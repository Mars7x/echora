//! Host GPU execution of NVN work (ECHORA_GPU=1): bound programs are
//! translated (`echora_gpu::shader`), their
//! bindings resolved through the inferred NVN2 binding map (docs/roadmap.md),
//! and clears and draws run through `echora_gpu`.
//!
//! Fixed-function state comes from record-time snapshots with NVN enum
//! values inferred from GL order (see compare_op, blend_factor, blend_op);
//! samplers keep only their filters and wrap modes;
//! textures upload their mip levels and layers. ASTC is decoded on the CPU
//! if compressed uploads are unavailable. Driver constant buffers 0..2
//! read as zeros.

use std::collections::{BTreeMap, VecDeque};
/// Maps here are keyed by addresses and names looked up on every call: the
/// fast hasher, not SipHash.
type HashMap<K, V> = super::FastMap<K, V>;
use std::sync::Arc;

use echora_gpu::ash::vk;
use echora_gpu::draw::{
    Blend, Draw, Fixed, ImageShape, Indices, MeshGroups, Program, Sampler, StencilFace,
    TextureSource, UploadRegion, VertexAttribute, VertexStream,
};
use echora_gpu::present::Presenter;
use echora_gpu::renderer::{Renderer, Scale};
use echora_gpu::shader::{self, DescriptorKind, MeshParameters, Parameters, Stage, Translated};

use super::{Command, Guest, Image, MAX_COLOR_TARGETS, Nvn, PagePool, StateSnapshots};

/// Reuse the allocation of the binding tables filled for each draw.
#[derive(Debug, Default)]
pub(super) struct DrawBindings {
    uniforms: echora_gpu::FastMap<(usize, u32), Vec<u8>>,
    guest_uniforms: echora_gpu::FastMap<(usize, u32), (u64, usize)>,
    textures: echora_gpu::FastMap<(usize, u32), (TextureSource, Sampler)>,
}

/// A built page table (see `Nvn::page_table`).
pub struct PageBuild {
    first: u64,
    entries: Vec<u64>,
    pools: HashMap<PagePool, (PoolShareVersion, bool)>,
    shared_mapping: std::cell::OnceCell<Option<Vec<writer_ranges::SharedPages>>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PoolShareVersion {
    Unshared,
    Resident(u64),
    Imported(usize),
}

#[derive(Default)]
struct TargetIndex {
    stamp: Option<u64>,
    storage: HashMap<u64, Vec<u64>>,
    mips: HashMap<u64, Vec<(u64, u32, (u32, u32))>>,
    by_target: HashMap<u64, (u64, Vec<u64>)>,
    changed: super::FastSet<u64>,
}

/// Guest storage as seen at successive byte offsets in a texture.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum TextureBacking {
    Linear((u64, std::ops::Range<u64>)),
    Virtual {
        size: u64,
        pieces: Vec<(u64, std::ops::Range<u64>)>,
    },
}

impl TextureBacking {
    fn linear(base: u64, size: u64) -> Self {
        Self::Linear((0, base..base + size))
    }

    pub(super) fn pieces(&self) -> &[(u64, std::ops::Range<u64>)] {
        match self {
            Self::Linear(piece) => std::slice::from_ref(piece),
            Self::Virtual { pieces, .. } => pieces,
        }
    }

    fn size(&self) -> u64 {
        match self {
            Self::Linear((_, range)) => range.end - range.start,
            Self::Virtual { size, .. } => *size,
        }
    }

    /// Offsets within a request and the physical ranges that back them.
    pub(super) fn subranges(
        &self,
        offset: u64,
        length: usize,
    ) -> impl Iterator<Item = (usize, std::ops::Range<u64>)> + '_ {
        let end = offset + length as u64;
        let pieces = self.pieces();
        let first =
            pieces.partition_point(|(start, range)| start + (range.end - range.start) <= offset);
        pieces[first..]
            .iter()
            .take_while(move |(start, _)| *start < end)
            .map(move |(start, range)| {
                let left = offset.max(*start);
                let right = end.min(start + (range.end - range.start));
                let physical = range.start + (left - start);
                (
                    (left - offset) as usize,
                    physical..physical + (right - left),
                )
            })
    }

    fn read_into(&self, core: &Guest, offset: u64, bytes: &mut [u8]) {
        if let Self::Linear((_, range)) = self {
            read_storage_into(core, range.start + offset, bytes);
            return;
        }
        bytes.fill(0);
        for (start, range) in self.subranges(offset, bytes.len()) {
            let end = start + (range.end - range.start) as usize;
            read_storage_into(core, range.start, &mut bytes[start..end]);
        }
    }

    fn read(&self, core: &Guest) -> Vec<u8> {
        let mut bytes = vec![0; self.size() as usize];
        self.read_into(core, 0, &mut bytes);
        bytes
    }

    fn equals(&self, core: &Guest, expected: &[u8]) -> bool {
        if expected.len() as u64 != self.size() {
            return false;
        }
        let mut end = 0;
        for (start, range) in self.pieces() {
            let start = *start as usize;
            if expected[end..start].iter().any(|&byte| byte != 0) {
                return false;
            }
            end = start + (range.end - range.start) as usize;
            if !core
                .memory
                .equals(range.start, &expected[start..end])
                .unwrap_or(false)
            {
                return false;
            }
        }
        expected[end..].iter().all(|&byte| byte == 0)
    }

    fn sample(&self, core: &Guest, format: u32, dimensions: [u32; 3]) -> u64 {
        if let Self::Linear((_, range)) = self {
            return texture_sample(core, range.start, self.size(), format, dimensions);
        }
        let seed = texture_sample_seed(format, dimensions);
        sample_chunks(seed, 0, self.size(), |offset, bytes| {
            let length = bytes.len().min((self.size() - offset) as usize);
            self.read_into(core, offset, &mut bytes[..length]);
            true
        })
    }
}

fn texture_row_pitch(
    props: &BTreeMap<String, super::Args>,
    info: crate::nvn_format::FormatInfo,
    width: u32,
) -> u64 {
    // Buffer textures are linear arrays of texels.
    if props.get("Target").is_some_and(|target| target.x[0] == 10) {
        u64::from(width.div_ceil(info.block_width)) * u64::from(info.bytes_per_block)
    } else {
        props.get("Stride").map_or(0, |value| value.x[0])
    }
}

fn texture_height_and_depth(target: u32, height: u32, depth: u32) -> (u32, u32) {
    // For a 1D array, SetSize2D supplies the number of layers in Height.
    if target == 3 {
        (1, height.max(1))
    } else {
        (height, depth.max(1))
    }
}

impl TargetIndex {
    fn insert(
        &mut self,
        key: u64,
        base: u64,
        levels: impl Iterator<Item = (u32, u64, (u32, u32))>,
    ) {
        // Alias searches choose the first key in renderer order, even if
        // that target was inserted after the other candidates.
        let targets = self.storage.entry(base).or_default();
        targets.insert(targets.partition_point(|&target| target < key), key);
        let mut mip_storage = Vec::new();
        for (level, offset, extent) in levels {
            let address = base + offset;
            let mips = self.mips.entry(address).or_default();
            let at = mips.partition_point(|&(target, mip, _)| (target, mip) < (key, level));
            mips.insert(at, (key, level, extent));
            mip_storage.push(address);
        }
        self.by_target.insert(key, (base, mip_storage));
    }

    fn remove(&mut self, key: u64) {
        use std::collections::hash_map::Entry;

        let Some((base, mip_storage)) = self.by_target.remove(&key) else {
            return;
        };
        if let Entry::Occupied(mut entry) = self.storage.entry(base) {
            entry.get_mut().retain(|&target| target != key);
            if entry.get().is_empty() {
                entry.remove();
            }
        }
        for address in mip_storage {
            if let Entry::Occupied(mut entry) = self.mips.entry(address) {
                entry.get_mut().retain(|&(target, ..)| target != key);
                if entry.get().is_empty() {
                    entry.remove();
                }
            }
        }
    }
}

impl PagePool {
    fn pages(self, first: u64, count: usize) -> std::ops::Range<usize> {
        use echora_shader_abi::{PAGE_SHIFT, PAGE_SIZE};
        let start = (self.gpu >> PAGE_SHIFT).saturating_sub(first);
        let end = (self.gpu + self.size)
            .div_ceil(PAGE_SIZE)
            .saturating_sub(first);
        start as usize..end.min(count as u64) as usize
    }
}

#[path = "nvn_write_ranges.rs"]
mod writer_ranges;

pub struct Backend {
    pub renderer: Renderer,
    /// Translations by stage key and the parameters that shape them.
    translations: HashMap<TranslationKey, Result<Arc<Translated>, String>>,
    compute_jumps: HashMap<u64, Option<ComputeJumps>>,
    /// Bindless lookups by translation identity. The translation cache
    /// keeps each addressed translation alive.
    handle_reads: HashMap<usize, Arc<shader::handle_reads::HandleReads>>,
    global_writes: HashMap<usize, (shader::handle_reads::GlobalWrites, [u32; 3])>,
    uniform_writes: HashMap<usize, shader::handle_reads::UniformWriteRanges>,
    /// Last unbounded store and bounded writes in submission order.
    unbounded_write_serial: u64,
    shared_writes: VecDeque<(u64, BoundedWrites)>,
    /// Compute stage keys translated with raw storage pointers
    /// (`Parameters::native_pointers`).
    native_programs: std::collections::HashSet<u64>,
    /// The GPU page table and the pool set it was built
    /// for (count, and the sum of their GPU addresses and sizes).
    page_table: Option<((usize, u64), echora_shader_abi::PageTable)>,
    /// The last page table built: its entries and, per pool (base, size,
    /// GPU address, flags), the sharing it was mapped with (None: some
    /// pages unmapped, so mapped again next time).
    page_build: Option<PageBuild>,
    /// ECHORA_LOOP_TRACE (debugging): each traced compute program's
    /// counter buffer (`Renderer::debug_buffer` index and address) and the
    /// present it was last reported at.
    loop_traces: HashMap<u64, (usize, u64, u64)>,
    /// ECHORA_VERTEX_TRACE (debugging): each traced graphics stage's record
    /// buffer, by (vertex key, fragment key, stage index): its
    /// `Renderer::debug_buffer` index and the present it was last reported
    /// at.
    vertex_traces: HashMap<(u64, u64, usize), (usize, u64)>,
    /// Each debug buffer's device address, by its index.
    vertex_trace_addresses: Vec<u64>,
    /// Linked graphics programs by translation identities and trace mode.
    programs: HashMap<GraphicsProgramKey, Program>,
    /// Compute programs by translation identity and specialization bits.
    compute_programs: HashMap<(usize, u64), Program>,
    /// Counts of work skipped or approximated, by reason.
    pub notes: BTreeMap<String, u64>,
    /// "stat: " counters by their static name (`stat`; merged into the
    /// notes for the report by `notes_with_stats`).
    stats: super::FastMap<&'static str, u64>,
    /// Draws described in ECHORA_GPU_LOG so far (the first few are).
    traced: u32,
    /// Draws probed (ECHORA_GPU_PROBE).
    probed: u32,
    /// The watched pixel's last value in each watched target (ECHORA_GPU_PIXEL_WATCH).
    watched_pixel: HashMap<u64, [f32; 4]>,
    /// The window swapchain (ECHORA_WINDOW), once created; None after a
    /// failure, which is noted and not retried.
    presenter: Option<Result<Presenter, ()>>,
    /// Untiled guest textures with a sample of their raw storage and its
    /// guest address range (see `texture_source`).
    textures: HashMap<u64, (u64, TextureSource, TextureBacking)>,
    /// The present during which each cached texture's sample last matched:
    /// within one frame a texture is sampled once, not on every draw.
    texture_checked: HashMap<u64, u64>,
    /// The largest image extent the host GPU supports.
    max_extent: u32,
    /// Byte ranges (from the storage start) of cached uploads that
    /// application flushes overlapped: re-checked before the next use.
    dirty: HashMap<u64, Vec<std::ops::Range<u64>>>,
    /// Hashes of each upload's raw storage in STORAGE_CHUNK pieces, so a
    /// flush re-checks only the pieces it overlapped.
    storage_hashes: HashMap<u64, Vec<u64>>,
    /// Merged storage of the cached uploads, rebuilt when the set changes.
    cached_texture_ranges: Option<Vec<std::ops::Range<u64>>>,
    /// The linear rows a whole-level copy last wrote into a texture
    /// (`copy_texels`), with the texture's storage sample after it: while
    /// the storage samples the same they are its untiled level 0.
    pub(super) copied_levels: HashMap<u64, (u64, Arc<Vec<u8>>)>,
    /// Raw storage of uploads from CPU-uncached pools (up to
    /// SHADOW_LIMIT bytes). The application writes those without flushes
    /// (it flushes only CPU-cached pools), so each frame's first use
    /// compares the whole storage: a sample misses a few new glyphs in a
    /// font atlas.
    shadows: HashMap<u64, Vec<u8>>,
    /// Dispatch results not yet stored to guest memory: (renderer read
    /// token, guest range, binding, program, GPU address). They are
    /// stored at the end of the submission, or before work that may read
    /// that memory (`resolve_storage`): one wait for several dispatches.
    pending_storage: Vec<(u64, std::ops::Range<u64>, u32, u64, u64)>,
    /// Guest ranges compute shaders wrote in place (storage buffers bound
    /// over imported guest memory) since the GPU last finished: a CPU read
    /// of one waits for the GPU first (see `resolve_storage`).
    in_place_writes: Vec<std::ops::Range<u64>>,
    /// Bumped whenever GPU writes not yet in guest memory are recorded
    /// (in place, pending storage, resident ranges): a memoized binding
    /// checked at this generation needs no new check.
    gpu_writes: u64,
    /// Bumped whenever a resident pool's `gpu_dirty` ranges shrink.
    dirty_removals: u64,
    /// Every resident pool's `gpu_dirty` ranges, merged and sorted, as of
    /// `dirty_removals`: `gpu_writes_pending` searches it instead of every
    /// pool. A range added to a pool is added here too; it is rebuilt only
    /// after ranges shrink (rebuilding per GPU write slowed spd200 ~27%).
    dirty_index: std::cell::RefCell<(u64, Vec<std::ops::Range<u64>>)>,
    /// A program storing through the page table was recorded since the
    /// last wait for it: its stores land in guest memory or a resident
    /// pool's copy with no range recorded (see `resolve_gpu_written`).
    page_table_writes: bool,
    /// The submission serial of the last such program recorded, and of the
    /// work before the last fence or event the guest could see signaled:
    /// a CPU read waits for stores up to that point only (it is entitled to
    /// no later ones, which the console's GPU may not have made yet).
    page_table_write_serial: u64,
    observable_serial: u64,
    /// The renderer's marker at the last such signal (`Renderer::mark`):
    /// a wait for it ends when the GPU passes it, not when its batch ends.
    observable_marker: Option<u64>,
    /// A later store can share the marker's submission without being
    /// covered by the marker.
    page_table_wrote_after_signal: bool,
    /// Work storing through the page table has run at some point: the
    /// resident pools' ranges a CPU read covers may hold its results.
    page_table_wrote: bool,
    /// Storage buffers kept on the GPU between dispatches (when guest
    /// memory cannot be imported), by guest address: see `Resident`.
    resident: HashMap<u64, ResidentPool>,
    /// GPU addresses SAMPLES_PASSED reports were written to on the GPU
    /// (`gpu_samples_report`): conditions on them are evaluated there.
    samples_reports: std::collections::HashSet<u64>,
    /// Memory pools the GPU can read in place through guest memory's file
    /// (see `Renderer::import_file`), by guest base: size, file and the
    /// file offset of the base. Learned at presents (`share_new_pools`).
    shared_pools: HashMap<u64, (u64, i32, u64)>,
    /// Pools no single span shares (some pages unmapped, or their backing
    /// not contiguous): the pieces that are, by pool base, as (offset,
    /// length, file, offset in the file).
    shared_pieces: HashMap<u64, Vec<(u64, u64, i32, u64)>>,
    shared_guest_index: std::cell::RefCell<GuestFileIndex>,
    /// The publication of each live share, even when its span count or
    /// file offsets match those of an earlier share.
    pool_share_versions: HashMap<u64, usize>,
    /// Bumped whenever a pool is shared (or shared again): the page table
    /// is rebuilt after it changes.
    share_generation: usize,
    /// Render targets and mip levels by their guest storage address.
    target_index: TargetIndex,
    retired_since: HashMap<u64, u64>,
}

impl std::fmt::Debug for Backend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Backend")
            .field("device", &self.renderer.gpu.name)
            .field("programs", &self.programs.len())
            .field("notes", &self.notes)
            .finish()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SharedRange {
    file: i32,
    start: u64,
    end: u64,
}

/// Files a submission can change and the virtual ranges that need to be
/// projected again if its page table receives new entries.
struct BoundedWrites {
    shared: Vec<SharedRange>,
    virtual_ranges: Vec<std::ops::Range<u64>>,
}

impl From<Vec<SharedRange>> for BoundedWrites {
    fn from(shared: Vec<SharedRange>) -> Self {
        Self {
            shared,
            virtual_ranges: Vec::new(),
        }
    }
}

fn merge_shared_writes(ranges: &mut Vec<SharedRange>) {
    ranges.sort_unstable_by_key(|range| (range.file, range.start));
    ranges.dedup_by(|next, previous| {
        if next.file == previous.file && next.start <= previous.end {
            previous.end = previous.end.max(next.end);
            true
        } else {
            false
        }
    });
}

impl SharedRange {
    fn overlaps(self, other: Self) -> bool {
        self.file == other.file && self.start < other.end && other.start < self.end
    }
}

/// Imported guest spans indexed by address, in their original lookup
/// order when several spans cover a request.
#[derive(Default)]
struct GuestFileIndex {
    generation: usize,
    spans: Vec<GuestFileSpan>,
}

struct GuestFileSpan {
    guest: u64,
    end: u64,
    furthest_end: u64,
    order: usize,
    file: i32,
    offset: u64,
}

impl GuestFileIndex {
    fn new(generation: usize, mappings: impl Iterator<Item = (u64, u64, i32, u64)>) -> Self {
        let mut spans: Vec<_> = mappings
            .enumerate()
            .filter_map(|(order, (guest, length, file, offset))| {
                Some(GuestFileSpan {
                    guest,
                    end: guest.checked_add(length)?,
                    furthest_end: 0,
                    order,
                    file,
                    offset,
                })
            })
            .collect();
        spans.sort_unstable_by_key(|span| span.guest);
        let mut furthest_end = 0;
        for span in &mut spans {
            furthest_end = furthest_end.max(span.end);
            span.furthest_end = furthest_end;
        }
        Self { generation, spans }
    }

    fn overlapping(&self, guest: std::ops::Range<u64>) -> impl Iterator<Item = SharedRange> + '_ {
        let after = self.spans.partition_point(|span| span.guest < guest.end);
        self.spans[..after]
            .iter()
            .rev()
            .take_while(move |span| span.furthest_end > guest.start)
            .filter_map(move |span| {
                let first = span.guest.max(guest.start);
                let last = span.end.min(guest.end);
                if first >= last {
                    return None;
                }
                Some(SharedRange {
                    file: span.file,
                    start: span.offset.checked_add(first - span.guest)?,
                    end: span.offset.checked_add(last - span.guest)?,
                })
            })
    }

    fn range(&self, guest: u64, length: u64) -> Option<SharedRange> {
        let end = guest.checked_add(length)?;
        let after = self.spans.partition_point(|span| span.guest <= guest);
        let span = self.spans[..after]
            .iter()
            .rev()
            .take_while(|span| span.furthest_end >= end)
            .filter(|span| span.end >= end)
            .min_by_key(|span| span.order)?;
        let start = span.offset.checked_add(guest - span.guest)?;
        Some(SharedRange {
            file: span.file,
            start,
            end: start.checked_add(length)?,
        })
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        if let Some(Ok(presenter)) = self.presenter.take() {
            self.renderer.destroy_presenter(presenter);
        }
        let programs = std::mem::take(&mut self.programs)
            .into_values()
            .chain(std::mem::take(&mut self.compute_programs).into_values());
        if let Err(error) = self.renderer.destroy_programs(programs) {
            eprintln!("GPU program cleanup: {error}");
        }
    }
}

impl Backend {
    fn configure_render_scale(renderer: &mut Renderer, guest_height: Option<u32>) {
        if Scale::env_override().is_some() {
            return;
        }
        let Some((numerator, denominator)) = crate::settings::render_scale(guest_height) else {
            return;
        };
        let scale = Scale::new(numerator, denominator).unwrap_or_else(|| {
            eprintln!(
                "settings: render scale {numerator}/{denominator} is outside 1/2..4; using 1"
            );
            Scale::ONE
        });
        renderer.set_resolution_scale(scale);
    }

    /// A backend when ECHORA_GPU is set, the recompiler is linked and a
    /// device with shader objects exists.
    pub fn create(guest_display_height: Option<u32>) -> Result<Option<Self>, String> {
        if super::debug_var_os("ECHORA_GPU").is_none_or(|value| value == "0") {
            return Ok(None);
        }
        if !shader::available() {
            eprintln!(
                "gpu: no shader translator plugin is installed (ECHORA_SHADER_PLUGIN): programs the Rust translator does not handle are skipped"
            );
        }
        let gpu = echora_gpu::vulkan::Gpu::new()?;
        let mut renderer = Renderer::new(gpu)?;
        // ECHORA_SHADER_CACHE=DIR (set by `echora run`): compiled shaders
        // are kept there for the next run (not with ECHORA_SHADER_STATS,
        // which needs the driver to compile each).
        if let Some(directory) = super::debug_var_os("ECHORA_SHADER_CACHE")
            .filter(|_| super::debug_var_os("ECHORA_SHADER_STATS").is_none())
        {
            renderer.set_shader_cache(directory.into());
        }
        Self::configure_render_scale(&mut renderer, guest_display_height);
        Ok(Some(Self {
            max_extent: renderer.max_image_extent(),
            renderer,
            translations: HashMap::default(),
            compute_jumps: HashMap::default(),
            handle_reads: HashMap::default(),
            global_writes: HashMap::default(),
            uniform_writes: HashMap::default(),
            unbounded_write_serial: 0,
            shared_writes: VecDeque::new(),
            native_programs: Default::default(),
            page_table: None,
            page_build: None,
            loop_traces: HashMap::default(),
            vertex_traces: HashMap::default(),
            vertex_trace_addresses: Vec::new(),
            programs: HashMap::default(),
            compute_programs: HashMap::default(),
            notes: BTreeMap::new(),
            stats: Default::default(),
            traced: 0,
            probed: 0,
            watched_pixel: HashMap::default(),
            presenter: None,
            textures: HashMap::default(),
            texture_checked: HashMap::default(),
            dirty: HashMap::default(),
            storage_hashes: HashMap::default(),
            cached_texture_ranges: None,
            copied_levels: HashMap::default(),
            shadows: HashMap::default(),
            pending_storage: Vec::new(),
            in_place_writes: Vec::new(),
            gpu_writes: 0,
            dirty_removals: 0,
            dirty_index: std::cell::RefCell::new((u64::MAX, Vec::new())),
            page_table_writes: false,
            page_table_write_serial: 0,
            observable_serial: 0,
            observable_marker: None,
            page_table_wrote_after_signal: false,
            page_table_wrote: false,
            resident: HashMap::default(),
            samples_reports: std::collections::HashSet::new(),
            shared_pools: HashMap::default(),
            shared_pieces: HashMap::default(),
            shared_guest_index: Default::default(),
            pool_share_versions: HashMap::default(),
            share_generation: 0,
            target_index: TargetIndex::default(),
            retired_since: HashMap::default(),
        }))
    }

    fn record_page_table_write(&mut self, serial: u64, bounded: Option<BoundedWrites>) {
        self.page_table_writes = true;
        self.page_table_wrote = true;
        self.page_table_wrote_after_signal = true;
        self.page_table_write_serial = serial;
        if let Some(mut writes) = bounded {
            let done = self.renderer.completed_serial();
            self.discard_shared_writes_through(done);
            if serial > self.unbounded_write_serial
                && (!writes.shared.is_empty() || !writes.virtual_ranges.is_empty())
            {
                if let Some((previous_serial, previous)) = self.shared_writes.back_mut()
                    && *previous_serial == serial
                {
                    if !writes.shared.is_empty() && previous.shared != writes.shared {
                        previous.shared.append(&mut writes.shared);
                        merge_shared_writes(&mut previous.shared);
                    }
                    for range in writes.virtual_ranges {
                        add_range(&mut previous.virtual_ranges, range);
                    }
                } else {
                    debug_assert!(
                        self.shared_writes
                            .back()
                            .is_none_or(|&(previous, _)| previous < serial)
                    );
                    merge_shared_writes(&mut writes.shared);
                    self.shared_writes.push_back((serial, writes));
                }
            }
        } else {
            self.unbounded_write_serial = serial;
            self.shared_writes.clear();
        }
    }

    fn set_page_table(
        &mut self,
        key: (usize, u64),
        built: PageBuild,
        modified: Option<&[std::ops::Range<u64>]>,
    ) -> Result<echora_shader_abi::PageTable, String> {
        let previous_address = self.page_table.map(|(_, table)| table.address);
        let (table, extended) = if let Some(modified) = modified {
            self.stat("stat: page table uses patches", 1);
            let patches: Vec<_> = modified
                .iter()
                .map(|range| {
                    let start = range.start as usize;
                    (start, &built.entries[start..range.end as usize])
                })
                .collect();
            self.renderer
                .patch_page_table(built.first as u32, built.entries.len(), &patches)?
        } else {
            self.renderer
                .set_page_table(built.first as u32, &built.entries)?
        };
        debug_assert!(table.address == 0 || self.renderer.page_table_entries() == built.entries);
        self.page_table = Some((key, table));
        self.page_build = Some(built);
        if extended {
            self.extend_pending_writes();
        } else if previous_address != Some(table.address) {
            // Outstanding programs still read the retired table. Its
            // existing file ranges remain valid until their submissions finish.
            for (_, writes) in &mut self.shared_writes {
                writes.virtual_ranges.clear();
            }
        }
        Ok(table)
    }

    fn record_pool_share(&mut self, base: u64) {
        self.share_generation += 1;
        self.pool_share_versions.insert(base, self.share_generation);
    }

    fn extend_pending_writes(&mut self) {
        self.discard_shared_writes_through(self.renderer.completed_serial());
        let Some(latest) = self
            .shared_writes
            .iter()
            .rev()
            .find(|(_, writes)| !writes.virtual_ranges.is_empty())
            .map(|&(serial, _)| serial)
        else {
            return;
        };
        let projected = self
            .shared_writes
            .iter()
            .map(|(_, writes)| {
                if writes.virtual_ranges.is_empty() {
                    Some(Vec::new())
                } else {
                    self.project_shared_writes(&writes.virtual_ranges)
                }
            })
            .collect::<Option<Vec<_>>>();
        if let Some(projected) = projected {
            // Keep the original file ranges too: queued programs can have
            // observed the table before the new entries became visible.
            for ((_, writes), mut additional) in self.shared_writes.iter_mut().zip(projected) {
                if !additional.is_empty() {
                    writes.shared.append(&mut additional);
                    merge_shared_writes(&mut writes.shared);
                }
            }
        } else {
            self.unbounded_write_serial = self.unbounded_write_serial.max(latest);
            self.discard_shared_writes_through(latest);
        }
    }

    fn project_shared_writes(&self, ranges: &[std::ops::Range<u64>]) -> Option<Vec<SharedRange>> {
        let built = self.page_build.as_ref()?;
        let mapping = built
            .shared_mapping
            .get_or_init(|| {
                use echora_shader_abi::{PAGE_SHIFT, PAGE_SIZE};
                let (file, imports) = self.renderer.shared_device_file_mappings()?;
                let imports = imports
                    .into_iter()
                    .map(|(device, offset)| writer_ranges::ImportedSpan {
                        file,
                        device,
                        offset,
                    })
                    .collect();
                writer_ranges::project_table(
                    built.first << PAGE_SHIFT,
                    PAGE_SIZE,
                    &built.entries,
                    imports,
                )
            })
            .as_ref()?;
        Some(
            writer_ranges::project_writes(mapping, ranges)?
                .into_iter()
                .map(|range| SharedRange {
                    file: range.file,
                    start: range.bytes.start,
                    end: range.bytes.end,
                })
                .collect(),
        )
    }

    fn discard_shared_writes_through(&mut self, serial: u64) {
        while self
            .shared_writes
            .front()
            .is_some_and(|&(recorded, _)| recorded <= serial)
        {
            self.shared_writes.pop_front();
        }
    }

    /// Newest submission that may write these file bytes.
    fn shared_write_serial(&self, range: SharedRange) -> u64 {
        let minimum = self
            .unbounded_write_serial
            .max(self.renderer.completed_serial());
        self.shared_writes
            .iter()
            .rev()
            .take_while(|&&(serial, _)| serial > minimum)
            .find(|(_, written)| {
                let first = written
                    .shared
                    .partition_point(|span| (span.file, span.end) <= (range.file, range.start));
                written
                    .shared
                    .get(first)
                    .is_some_and(|span| range.overlaps(*span))
            })
            .map_or(minimum, |(serial, _)| *serial)
    }

    fn wait_for_table_writes(&mut self, range: Option<SharedRange>) -> Result<(), String> {
        if !self.page_table_writes
            || self.renderer.completed_serial() >= self.page_table_write_serial
        {
            self.page_table_writes = false;
            return Ok(());
        }
        let serial = range.map_or(self.page_table_write_serial, |range| {
            self.shared_write_serial(range)
        });
        self.renderer.wait_for(serial)?;
        if self.renderer.completed_serial() >= self.page_table_write_serial {
            self.page_table_writes = false;
        }
        Ok(())
    }

    /// Whether a resident GPU copy holds bytes newer than guest memory
    /// anywhere in this range.
    fn resident_writes_pending(&self, range: &std::ops::Range<u64>) -> bool {
        if self.resident.is_empty() {
            return false;
        }
        let mut index = self.dirty_index.borrow_mut();
        let stamp = self.dirty_removals;
        if index.0 != stamp {
            let mut ranges: Vec<std::ops::Range<u64>> = self
                .resident
                .values()
                .flat_map(|pool| pool.gpu_dirty.iter().cloned())
                .collect();
            merge_ranges(&mut ranges);
            *index = (stamp, ranges);
        }
        ranges_overlap(&index.1, range)
    }

    fn remove_texture_upload(&mut self, texture: u64) {
        if self.textures.remove(&texture).is_some() {
            self.cached_texture_ranges = None;
        }
    }

    fn cached_textures_overlap(&mut self, range: &std::ops::Range<u64>) -> bool {
        let ranges = self.cached_texture_ranges.get_or_insert_with(|| {
            let mut ranges: Vec<_> = self
                .textures
                .values()
                .flat_map(|(_, _, backing)| backing.pieces().iter().map(|(_, range)| range.clone()))
                .collect();
            merge_ranges(&mut ranges);
            ranges
        });
        ranges_overlap(ranges, range)
    }

    /// A physical write can cover parts of several textures, including
    /// more than one virtual offset in the same texture.
    fn mark_texture_bytes_changed(&mut self, range: &std::ops::Range<u64>) -> bool {
        let mut count = 0;
        if self.cached_textures_overlap(range) {
            for (&key, (_, _, backing)) in &self.textures {
                for (offset, storage) in backing.pieces() {
                    let start = range.start.max(storage.start);
                    let end = range.end.min(storage.end);
                    if start < end {
                        let start = offset + (start - storage.start);
                        let end = offset + (end - storage.start);
                        self.texture_checked.remove(&key);
                        self.dirty.entry(key).or_default().push(start..end);
                        count += 1;
                    }
                }
            }
        }
        self.stat("stat: textures invalidated", count);
        count != 0
    }

    /// Add `amount` to a counter in the notes (named "stat: ..."), for
    /// the run report; nothing is logged.
    pub(super) fn stat(&mut self, name: &'static str, amount: u64) {
        // Called dozens of times per draw: a hashed lookup of the static
        // name (the notes are a string-ordered map thousands long).
        *self.stats.entry(name).or_default() += amount;
    }

    /// The notes with the stat counters added in.
    pub(super) fn notes_with_stats(&self) -> BTreeMap<String, u64> {
        let mut notes = self.notes.clone();
        for (&name, &amount) in &self.stats {
            *notes.entry(name.to_string()).or_default() += amount;
        }
        notes
    }

    /// Count a note; its first occurrence is appended to ECHORA_GPU_LOG
    /// when set, so reasons are visible while a run is in progress.
    fn note(&mut self, reason: impl Into<String>) {
        let reason = reason.into();
        let count = self.notes.entry(reason.clone()).or_default();
        *count += 1;
        if *count == 1 {
            if let Some(path) = super::debug_var_os("ECHORA_GPU_LOG") {
                use std::io::Write;
                if let Ok(mut file) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                {
                    let _ = writeln!(file, "{reason}");
                }
            }
        }
    }
}

/// The translated vertex/fragment or task/mesh/fragment stages of a
/// graphics pipeline. The cached Program keeps these Arc allocations alive.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct GraphicsProgramKey {
    stages: [usize; 3],
    traced: bool,
}

/// Translated stages and binding counts shared by draws of the program.
#[derive(Debug)]
pub(super) struct BoundProgram {
    keys: (u64, u64),
    pipeline: GraphicsProgramKey,
    stages: Box<[Arc<Translated>]>,
    entries: Box<[super::Args]>,
    uniforms: usize,
    textures: usize,
}

impl BoundProgram {
    fn new(keys: (u64, u64), stages: Box<[Arc<Translated>]>, entries: Box<[super::Args]>) -> Self {
        let mut uniforms = 0;
        let mut textures = 0;
        for descriptor in stages.iter().flat_map(|stage| &stage.descriptors) {
            match descriptor.kind() {
                DescriptorKind::UniformBuffer => uniforms += 1,
                DescriptorKind::Texture | DescriptorKind::TextureBuffer => textures += 1,
                _ => {}
            }
        }
        let mut identities = [0; 3];
        for (identity, stage) in identities[..stages.len()].iter_mut().zip(stages.iter()) {
            *identity = Arc::as_ptr(stage) as usize;
        }
        Self {
            keys,
            pipeline: GraphicsProgramKey {
                stages: identities,
                traced: false,
            },
            stages,
            entries,
            uniforms,
            textures,
        }
    }
}

/// Stage key, binding base, previous stage's stores, generic input types,
/// compute and mesh state, native pointers, and proved compute jump targets.
type TranslationKey = (
    u64,
    [u32; 3],
    Option<[u64; 8]>,
    [u8; 32],
    [u8; MAX_COLOR_TARGETS],
    [u32; 5],
    Option<MeshParameters>,
    bool,
    Option<Arc<shader::JumpTargets>>,
    Vec<shader::SampledTexture>,
);

struct ComputeJumps {
    tables: shader::JumpTables,
    last: Option<(Vec<Box<[u8]>>, Option<Arc<shader::JumpTargets>>)>,
}

impl ComputeJumps {
    fn from_program(program: &[u8]) -> Option<Self> {
        Some(Self {
            tables: shader::JumpTables::from_compute_program(program)?,
            last: None,
        })
    }

    fn resolve(&mut self, constants: &[u8]) -> Option<Arc<shader::JumpTargets>> {
        if let Some((saved, targets)) = &self.last
            && self
                .tables
                .table_bytes(constants)
                .zip(saved)
                .all(|(current, saved)| current == Some(saved.as_ref()))
        {
            return targets.clone();
        }
        let bytes = self
            .tables
            .table_bytes(constants)
            .map(|table| table.map(Box::from))
            .collect::<Option<Vec<_>>>()?;
        let targets = self.tables.resolve(constants).map(Arc::new);
        self.last = Some((bytes, targets.clone()));
        targets
    }
}

/// Shader input types for the bound vertex attributes.
fn generic_input_types(attributes: &[BTreeMap<String, super::Args>]) -> [u8; 32] {
    let mut types = [0; 32];
    for (location, props) in attributes.iter().enumerate().take(32) {
        let format = props.get("Format").map_or(0, |args| args.x[0]) as u32;
        let name = crate::nvn_format::info(format).map_or("", |info| info.name);
        types[location] = if name.ends_with("UI") {
            2
        } else if name.ends_with('I') {
            1
        } else {
            0
        };
    }
    types
}

/// Vulkan format of an NVN format, by its name in `nvn_format`.
pub fn vk_format(format: u32) -> Option<vk::Format> {
    use vk::Format as F;
    Some(match crate::nvn_format::info(format)?.name {
        "R8" => F::R8_UNORM,
        "R8SN" => F::R8_SNORM,
        "R8UI" => F::R8_UINT,
        "R8I" => F::R8_SINT,
        "R16F" => F::R16_SFLOAT,
        "R16" => F::R16_UNORM,
        "R16SN" => F::R16_SNORM,
        "R16UI" => F::R16_UINT,
        "R16I" => F::R16_SINT,
        "R32F" => F::R32_SFLOAT,
        "R32UI" => F::R32_UINT,
        "R32I" => F::R32_SINT,
        "RG8" => F::R8G8_UNORM,
        "RG8SN" => F::R8G8_SNORM,
        "RG8UI" => F::R8G8_UINT,
        "RG8I" => F::R8G8_SINT,
        "RG16F" => F::R16G16_SFLOAT,
        "RG16" => F::R16G16_UNORM,
        "RG16SN" => F::R16G16_SNORM,
        "RG16UI" => F::R16G16_UINT,
        "RG16I" => F::R16G16_SINT,
        "RG32F" => F::R32G32_SFLOAT,
        "RG32UI" => F::R32G32_UINT,
        "RG32I" => F::R32G32_SINT,
        "RGB32F" => F::R32G32B32_SFLOAT,
        "RGB32UI" => F::R32G32B32_UINT,
        "RGB32I" => F::R32G32B32_SINT,
        "RGBA8" => F::R8G8B8A8_UNORM,
        "RGBA8SN" => F::R8G8B8A8_SNORM,
        "RGBA8UI" => F::R8G8B8A8_UINT,
        "RGBA8I" => F::R8G8B8A8_SINT,
        "RGBA16F" => F::R16G16B16A16_SFLOAT,
        "RGBA16" => F::R16G16B16A16_UNORM,
        "RGBA16SN" => F::R16G16B16A16_SNORM,
        "RGBA16UI" => F::R16G16B16A16_UINT,
        "RGBA16I" => F::R16G16B16A16_SINT,
        "RGBA32F" => F::R32G32B32A32_SFLOAT,
        "RGBA32UI" => F::R32G32B32A32_UINT,
        "RGBA32I" => F::R32G32B32A32_SINT,
        "DEPTH16" => F::D16_UNORM,
        "DEPTH32F" => F::D32_SFLOAT,
        // AMD GPUs have no D24S8; D32F_S8 holds every D24 value.
        "DEPTH24_STENCIL8" => F::D32_SFLOAT_S8_UINT,
        "DEPTH32F_STENCIL8" => F::D32_SFLOAT_S8_UINT,
        "RGBA8_SRGB" => F::R8G8B8A8_SRGB,
        "RGB10A2" => F::A2B10G10R10_UNORM_PACK32,
        "RGB10A2UI" => F::A2B10G10R10_UINT_PACK32,
        "RGB10A2SN" => F::A2B10G10R10_SNORM_PACK32,
        "RGB10A2I" => F::A2B10G10R10_SINT_PACK32,
        "R11G11B10F" => F::B10G11R11_UFLOAT_PACK32,
        "RGB9E5F" => F::E5B9G9R9_UFLOAT_PACK32,
        "BC1_UNORM" => F::BC1_RGBA_UNORM_BLOCK,
        "BC2_UNORM" => F::BC2_UNORM_BLOCK,
        "BC3_UNORM" => F::BC3_UNORM_BLOCK,
        "BC1_SRGB" => F::BC1_RGBA_SRGB_BLOCK,
        "BC2_SRGB" => F::BC2_SRGB_BLOCK,
        "BC3_SRGB" => F::BC3_SRGB_BLOCK,
        "BC4_UNORM" => F::BC4_UNORM_BLOCK,
        "BC4_SNORM" => F::BC4_SNORM_BLOCK,
        "BC5_UNORM" => F::BC5_UNORM_BLOCK,
        "BC5_SNORM" => F::BC5_SNORM_BLOCK,
        "BC7_UNORM" => F::BC7_UNORM_BLOCK,
        "BC7_SRGB" => F::BC7_SRGB_BLOCK,
        "BC6H_SFLOAT" => F::BC6H_SFLOAT_BLOCK,
        "BC6H_UFLOAT" => F::BC6H_UFLOAT_BLOCK,
        "BGRA8" => F::B8G8R8A8_UNORM,
        "BGRA8_SRGB" => F::B8G8R8A8_SRGB,
        _ => return None,
    })
}

fn astc_vk_format(format: u32) -> Option<vk::Format> {
    // NVN lists the same block footprints as Vulkan, first UNORM and then
    // sRGB. Vulkan interleaves the two variants.
    let (block, srgb) = match format {
        121..=134 => (format - 121, 0),
        135..=148 => (format - 135, 1),
        _ => return None,
    };
    Some(vk::Format::from_raw(
        vk::Format::ASTC_4X4_UNORM_BLOCK.as_raw() + (block * 2 + srgb) as i32,
    ))
}

/// How a vertex program pulling its inputs decodes an NVN vertex format
/// (`echora_shader_abi::VertexFormat`); 0 (no format) reads as unbound.
pub fn vertex_format(format: u32) -> Option<echora_shader_abi::VertexFormat> {
    use echora_shader_abi::{VertexFormat, VertexLayout as L, VertexNumeric as N};
    if format == 0 {
        return Some(VertexFormat::default());
    }
    let name = crate::nvn_format::info(format)?.name;
    let packed = |components, layout, numeric| VertexFormat {
        components,
        layout,
        numeric,
        bgra: false,
    };
    match name {
        "RGB10A2" => return Some(packed(4, L::Packed1010102, N::Unorm)),
        "RGB10A2UI" => return Some(packed(4, L::Packed1010102, N::Uint)),
        // Packed normals and tangents.
        "RGB10A2SN" => return Some(packed(4, L::Packed1010102, N::Snorm)),
        "RGB10A2I" => return Some(packed(4, L::Packed1010102, N::Sint)),
        "R11G11B10F" => return Some(packed(3, L::Packed111110, N::Float)),
        "BGRA8" => {
            return Some(VertexFormat {
                bgra: true,
                ..packed(4, L::Bits8, N::Unorm)
            });
        }
        _ => {}
    }
    // R/RG/RGB/RGBA, then 8/16/32, then "" (unorm), SN, UI, I or F.
    let channels = name.len() - name.trim_start_matches(['R', 'G', 'B', 'A']).len();
    let rest = &name[channels..];
    let digits = rest.len() - rest.trim_start_matches(|c: char| c.is_ascii_digit()).len();
    let layout = match &rest[..digits] {
        "8" => L::Bits8,
        "16" => L::Bits16,
        "32" => L::Bits32,
        _ => return None,
    };
    let numeric = match &rest[digits..] {
        "" => N::Unorm,
        "SN" => N::Snorm,
        "UI" => N::Uint,
        "I" => N::Sint,
        "F" => N::Float,
        _ => return None,
    };
    if !(1..=4).contains(&channels) || (numeric == N::Float && layout == L::Bits8) {
        return None;
    }
    Some(packed(channels as u32, layout, numeric))
}

/// NVN stage index (as in BindTexture/BindUniformBuffer) of a stage.
fn nvn_stage(stage: Stage) -> u32 {
    match stage {
        Stage::Vertex => 0,
        Stage::Fragment => 1,
        Stage::Geometry => 2,
        Stage::TessControl => 3,
        Stage::TessEval => 4,
        Stage::Compute => 5,
        // The application binds a task or mesh program's resources at
        // these stages (the bound uniform buffers show them).
        Stage::Task => 6,
        Stage::Mesh => 7,
    }
}

/// Where NVN stage `nvn`'s storage buffer table starts in the driver
/// constants c0 (16 bytes a slot): 0x110 + 0x100 * the hardware's stage
/// order (vertex, tessellation control, tessellation evaluation,
/// geometry, fragment); task programs read theirs from 0x670 and mesh
/// programs from 0x770 (the offsets the game's own task and mesh programs
/// load their buffer addresses from).
fn c0_storage_table(nvn: u32) -> u32 {
    match nvn {
        0 => 0x110,
        3 => 0x210,
        4 => 0x310,
        2 => 0x410,
        6 => 0x670,
        7 => 0x770,
        _ => 0x510,
    }
}

fn shader_word(bytes: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        bytes.get(offset..offset + 4)?.try_into().ok()?,
    ))
}

/// The translator sizes a bank through its last referenced word.
fn set_shader_word(bytes: &mut [u8], offset: usize, word: u32) {
    if let Some(destination) = bytes.get_mut(offset..offset + 4) {
        destination.copy_from_slice(&word.to_le_bytes());
    }
}

fn storage_contains(base: u64, size: u64, address: u64, length: u64) -> bool {
    address
        .checked_sub(base)
        .and_then(|offset| offset.checked_add(length))
        .is_some_and(|end| end <= size)
}

/// Draw primitive modes, GL numbering (inferred: DrawElementsBaseVertex
/// mode 4 draws six indices per quad).
fn topology(mode: u64) -> Option<vk::PrimitiveTopology> {
    use vk::PrimitiveTopology as T;
    Some(match mode {
        0 => T::POINT_LIST,
        1 => T::LINE_LIST,
        3 => T::LINE_STRIP,
        4 => T::TRIANGLE_LIST,
        5 => T::TRIANGLE_STRIP,
        6 => T::TRIANGLE_FAN,
        _ => return None,
    })
}

/// NVN depth/stencil compare functions: GL order from 1 (inferred; the
/// game's depth func 7 with depth testing off).
fn compare_op(value: u64) -> vk::CompareOp {
    use vk::CompareOp as C;
    match value {
        1 => C::NEVER,
        2 => C::LESS,
        3 => C::EQUAL,
        4 => C::LESS_OR_EQUAL,
        5 => C::GREATER,
        6 => C::NOT_EQUAL,
        7 => C::GREATER_OR_EQUAL,
        _ => C::ALWAYS,
    }
}

/// NVN stencil operations: GL order from 1 (inferred, as the compare
/// functions are).
fn stencil_op(value: u64) -> vk::StencilOp {
    use vk::StencilOp as S;
    match value {
        2 => S::ZERO,
        3 => S::REPLACE,
        4 => S::INCREMENT_AND_CLAMP,
        5 => S::DECREMENT_AND_CLAMP,
        6 => S::INVERT,
        7 => S::INCREMENT_AND_WRAP,
        8 => S::DECREMENT_AND_WRAP,
        _ => S::KEEP,
    }
}

/// NVN blend factors: GL order from 1 (inferred; the game's UI blends with
/// 5/6, source alpha over one minus source alpha).
fn blend_factor(value: u64) -> Option<vk::BlendFactor> {
    use vk::BlendFactor as F;
    Some(match value {
        1 => F::ZERO,
        2 => F::ONE,
        3 => F::SRC_COLOR,
        4 => F::ONE_MINUS_SRC_COLOR,
        5 => F::SRC_ALPHA,
        6 => F::ONE_MINUS_SRC_ALPHA,
        7 => F::DST_ALPHA,
        8 => F::ONE_MINUS_DST_ALPHA,
        9 => F::DST_COLOR,
        10 => F::ONE_MINUS_DST_COLOR,
        11 => F::SRC_ALPHA_SATURATE,
        // 0x10..0x13 are the dual-source factors (SRC1_COLOR ..
        // ONE_MINUS_SRC1_ALPHA): programs do not write a second source
        // yet, so they are refused (`None`).
        0x61 => F::CONSTANT_COLOR,
        0x62 => F::ONE_MINUS_CONSTANT_COLOR,
        0x63 => F::CONSTANT_ALPHA,
        0x64 => F::ONE_MINUS_CONSTANT_ALPHA,
        _ => return None,
    })
}

/// NVN blend equations: add, subtract, reverse subtract, min, max from 1
/// (inferred).
fn blend_op(value: u64) -> vk::BlendOp {
    use vk::BlendOp as O;
    match value {
        2 => O::SUBTRACT,
        3 => O::REVERSE_SUBTRACT,
        4 => O::MIN,
        5 => O::MAX,
        _ => O::ADD,
    }
}

impl Nvn {
    pub(super) fn record_window_resolution(&mut self, texture: u64) {
        if self.guest_display_height.is_some() {
            return;
        }
        let Ok((_, height)) = self.texture_extent(texture) else {
            return;
        };
        self.guest_display_height = Some(height);
        if let Some(backend) = self.gpu.as_mut()
            && backend.renderer.targets_created == 0
            && backend.translations.is_empty()
            && backend.programs.is_empty()
            && backend.compute_programs.is_empty()
        {
            // Device setup can precede window initialization. Until targets
            // or programs exist, the real guest height can replace the default.
            Backend::configure_render_scale(&mut backend.renderer, Some(height));
        }
    }

    /// Restart the GPU statistics ("stat: " notes; ECHORA_PROFILE_FROM).
    pub(crate) fn reset_gpu_stats(&mut self) {
        if let Some(backend) = self.gpu.as_mut() {
            backend.notes.retain(|name, _| !name.starts_with("stat: "));
            backend.stats.clear();
        }
    }

    /// The GPU backend, created on first use when enabled.
    pub(super) fn gpu(&mut self) -> Option<&mut Backend> {
        if !self.gpu_checked {
            self.gpu_checked = true;
            match Backend::create(self.guest_display_height) {
                Ok(backend) => self.gpu = backend,
                Err(error) => {
                    // A driver known to hang the GPU ends the run: going
                    // on without a picture would only hide it.
                    if error.starts_with(echora_gpu::vulkan::UNSUPPORTED_DRIVER) {
                        eprintln!("echora: {error}");
                        if let Some(path) = super::debug_var_os("ECHORA_GPU_LOG") {
                            let _ = std::fs::write(path, format!("{error}\n"));
                        }
                        std::process::exit(i32::from(echora_config::UNSUPPORTED_DRIVER_EXIT));
                    }
                    let message = format!("GPU backend disabled: {error}");
                    eprintln!("nvn: {message}");
                    if let Some(path) = super::debug_var_os("ECHORA_GPU_LOG") {
                        let _ = std::fs::write(path, message + "\n");
                    }
                }
            }
        }
        self.gpu.as_mut()
    }

    /// Count a GPU approximation or skip by reason.
    pub(super) fn gpu_note(&mut self, reason: impl Into<String>) {
        let reason = reason.into();
        self.sequence_note(&reason);
        if let Some(backend) = self.gpu.as_mut() {
            backend.note(reason);
        }
    }

    /// Whether present being prepared is sequenced:
    /// ECHORA_GPU_SEQUENCE_FRAME=N or N-M (a range, inclusive).
    pub(super) fn sequencing(&self) -> bool {
        if self.sequence_chosen == Some(self.presented_frames) {
            return true;
        }
        let Ok(value) = super::debug_var("ECHORA_GPU_SEQUENCE_FRAME") else {
            return false;
        };
        let (first, last) = value.split_once('-').unwrap_or((&value, &value));
        match (first.parse::<u64>(), last.parse::<u64>()) {
            (Ok(first), Ok(last)) => (first..=last).contains(&self.presented_frames),
            _ => false,
        }
    }

    /// ECHORA_GPU_SEQUENCE_FRAME: flow notes made while a sequenced
    /// present is being prepared are also logged with their order (and
    /// that present's number).
    pub(super) fn sequence_note(&mut self, line: &str) {
        if self.sequencing() && line.starts_with("flow") {
            self.sequence += 1;
            let numbered = format!("seq {:05} f{} {line}", self.sequence, self.presented_frames);
            if let Some(backend) = self.gpu.as_mut() {
                backend.note(numbered);
            }
        }
    }

    /// Size and Vulkan format of a texture used as a render target.
    fn target_shape(&self, texture: u64) -> Result<(u32, u32, vk::Format), String> {
        if self.target_shapes.0 == super::debug_generation()
            && let Some(&(width, height, format, ..)) = self.target_shapes.1.get(&texture)
        {
            return Ok((width, height, format));
        }
        let (width, height) = self.texture_extent(texture)?;
        let object = self.object(texture, "Texture")?;
        let format = object
            .builder
            .get("Format")
            .map_or(0, |value| value.x[0] as u32);
        let vk_format =
            vk_format(format).ok_or_else(|| format!("render target format {format}"))?;
        Ok((width, height, vk_format))
    }

    fn color_view_format(&self, location: usize) -> Option<vk::Format> {
        self.render
            .color_formats
            .get(location)
            .copied()
            .filter(|format| *format != 0)
            .and_then(vk_format)
    }

    fn fragment_output_types(&self) -> [u8; MAX_COLOR_TARGETS] {
        std::array::from_fn(|location| {
            let Some(texture) = self
                .render
                .colors
                .get(location)
                .copied()
                .filter(|key| *key != 0)
            else {
                return 0;
            };
            let format = self
                .color_view_format(location)
                .or_else(|| self.target_shape(texture).ok().map(|(_, _, format)| format));
            match format.map(echora_gpu::renderer::sampled_texture_type) {
                Some(shader::SampleType::Signed) => 1,
                Some(shader::SampleType::Unsigned) => 2,
                _ => 0,
            }
        })
    }

    /// Draws recorded so far (the statistics' count).
    pub(super) fn gpu_draw_count(&self) -> u64 {
        self.gpu
            .as_ref()
            .and_then(|backend| backend.stats.get("stat: draws").copied())
            .unwrap_or(0)
    }

    /// Resolve texture bindings anew (`binding_memo`), counted by `why`.
    pub(super) fn bump_binding_serial(&mut self, why: &'static str) {
        self.binding_serial += 1;
        if let Some(backend) = self.gpu.as_mut() {
            backend.stat(why, 1);
        }
    }

    /// Drop cached textures whose storage overlaps guest `range`, or the
    /// texture object `texture` itself: the application wrote it.
    pub(super) fn gpu_invalidate(&mut self, range: std::ops::Range<u64>, texture: Option<u64>) {
        // Resolved bindings (`binding_memo`) change only with a texture's
        // bytes: one named, or one under the range. Descriptors are
        // registered, not read from the range.
        if self.stage_cache.get_mut().invalidate(&range)
            || self
                .code_pools
                .iter()
                .any(|(&start, &size)| start < range.end && range.start < start + size)
        {
            self.flush_epoch += 1;
        }
        if texture.is_some() {
            self.bump_binding_serial("stat: binding serial by an invalidated texture");
        }
        if let Some(backend) = self.gpu.as_mut() {
            for pool in backend.resident.values_mut() {
                let start = range.start.max(pool.base);
                let end = range.end.min(pool.base + pool.size);
                if start < end {
                    // The CPU's bytes are newer there than the GPU's.
                    if remove_range(&mut pool.gpu_dirty, start..end) {
                        backend.dirty_removals += 1;
                    }
                    add_range(&mut pool.guest_dirty, start..end);
                }
            }
            // WriteTexels names the texture: its contents changed.
            if let Some(texture) = texture {
                backend.remove_texture_upload(texture);
            }
            if backend.mark_texture_bytes_changed(&range) {
                self.bump_binding_serial("stat: binding serial by touched textures");
            }
        }
    }

    /// ClearBuffer over guest `range` (already written in guest memory):
    /// cached textures over it are dropped as for a flush, and resident
    /// copies are filled with `value` on the GPU instead of uploaded.
    pub(super) fn gpu_clear_buffer(
        &mut self,
        range: std::ops::Range<u64>,
        value: u32,
    ) -> Result<(), String> {
        self.gpu_invalidate(range.clone(), None);
        let Some(backend) = self.gpu.as_mut() else {
            return Ok(());
        };
        let mut fills = Vec::new();
        for pool in backend.resident.values_mut() {
            let piece = range.start.max(pool.base)..range.end.min(pool.base + pool.size);
            if piece.start < piece.end {
                // Both copies hold the cleared words once the fill runs.
                remove_range(&mut pool.guest_dirty, piece.clone());
                fills.push((pool.base, piece));
            }
        }
        for (base, piece) in fills {
            backend.stat(
                "stat: resident bytes cleared on the GPU",
                piece.end - piece.start,
            );
            backend.renderer.fill_resident(
                base,
                piece.start - base,
                piece.end - piece.start,
                value,
            )?;
        }
        Ok(())
    }

    /// ClearColor on the GPU: `color` as the four captured 32-bit words.
    pub(super) fn gpu_clear_color(
        &mut self,
        core: &Guest,
        (texture, level, layer): (u64, u32, u32),
        color: [u32; 4],
        rect: Option<[i32; 4]>,
    ) -> Result<(), String> {
        // A texture object over one level of another target clears that
        // target's level, as draws into it render there.
        let (texture, level) = self.resolve_color_target(texture, level)?;
        let started = std::time::Instant::now();
        self.ensure_render_target(texture)?;
        let ensured = std::time::Instant::now();
        crate::timeline::span(u64::MAX - 2, "clear: ensure target", started, ensured);
        if self
            .gpu
            .as_ref()
            .unwrap()
            .renderer
            .clear_needs_initialization(texture, level, layer, rect)
        {
            self.initialize_new_targets(core);
        }
        let backend = self.gpu.as_mut().unwrap();
        backend.renderer.defined(texture);
        let value = vk::ClearColorValue {
            float32: color.map(f32::from_bits),
        };
        let cleared = backend
            .renderer
            .clear_color_level(texture, level, layer, rect, value);
        crate::timeline::span(
            u64::MAX - 2,
            "clear: clear",
            ensured,
            std::time::Instant::now(),
        );
        cleared
    }

    /// ClearDepthStencil on the GPU: the masked-off aspects keep their
    /// contents (partial stencil masks clear all bits; not modeled).
    pub(super) fn gpu_clear_depth(
        &mut self,
        core: &Guest,
        texture: u64,
        depth: Option<f32>,
        stencil: Option<u32>,
    ) -> Result<(), String> {
        let (_, _, format) = self.target_shape(texture)?;
        let has_stencil = echora_gpu::renderer::has_stencil(format);
        let stencil = stencil.filter(|_| has_stencil);
        if depth.is_none() && stencil.is_none() {
            return Ok(());
        }
        self.ensure_render_target(texture)?;
        let scissor = self.render.scissor;
        let layer = self.render.depth_layer;
        if !has_stencil
            && self
                .gpu
                .as_ref()
                .unwrap()
                .renderer
                .clear_needs_initialization(texture, 0, layer, scissor)
        {
            self.initialize_new_targets(core);
        }
        let backend = self.gpu.as_mut().unwrap();
        backend.renderer.defined(texture);
        backend
            .renderer
            .clear_depth_stencil(texture, layer, scissor, depth, stencil)
    }

    /// Release what was cached for texture object `texture`, finalized:
    /// its upload (CPU texels and GPU image), saved storage copy and hashes.
    /// Gameplay streams textures, and these grew without bound. Render
    /// targets are retained by storage, independently of the texture object.
    pub(super) fn gpu_forget_texture(&mut self, texture: u64) {
        let Some(backend) = self.gpu.as_mut() else {
            return;
        };
        backend.remove_texture_upload(texture);
        backend.texture_checked.remove(&texture);
        backend.shadows.remove(&texture);
        backend.storage_hashes.remove(&texture);
        backend.dirty.remove(&texture);
        backend.copied_levels.remove(&texture);
        backend.renderer.forget_upload(texture);
    }

    /// Discard guest texture uploads whose virtual storage may change.
    pub(super) fn invalidate_virtual_texture_uploads(&mut self, pool: u64, mappings: &[[u64; 4]]) {
        let Some(backend) = self.gpu.as_ref() else {
            return;
        };
        if mappings.is_empty() {
            return;
        }
        let mut affected: Vec<_> = backend
            .textures
            .keys()
            .chain(backend.copied_levels.keys())
            .copied()
            .filter(|&texture| {
                let Some(storage) = self
                    .objects
                    .get(&texture)
                    .and_then(|o| o.builder.get("Storage"))
                else {
                    return false;
                };
                if storage.x[0] != pool {
                    return false;
                }
                let start = storage.x[1];
                let end = self
                    .texture_storage_size(texture)
                    .map_or(u64::MAX, |length| start.saturating_add(length));
                mappings.iter().any(|&[_, _, offset, length]| {
                    length != 0 && start < offset.saturating_add(length) && offset < end
                })
            })
            .collect();
        if !affected.is_empty() {
            affected.sort_unstable();
            affected.dedup();
            self.bump_binding_serial("stat: binding serial by virtual texture remapping");
            for texture in affected {
                self.gpu_forget_texture(texture);
            }
        }
    }

    /// Forget the host target and cached upload kept for texture object
    /// `texture`, whose structure now describes another texture.
    /// Texture structure `texture` now describes `identity` (storage and
    /// format) instead of `previous`: what the GPU rendered stays with the
    /// memory, as on the console. Its target moves to the retired key of
    /// `previous` (where a descriptor registered before still finds it),
    /// and a target kept for `identity` from an earlier texture there
    /// becomes its own.
    pub(super) fn gpu_move_texture(
        &mut self,
        texture: u64,
        previous: (u64, u64),
        identity: Option<(u64, u64)>,
    ) {
        self.texture_epoch += 1;
        self.binding_evictions.push((
            texture,
            [previous.0, identity.map_or(0, |(storage, _)| storage)],
        ));
        if self.depth_traced(texture) {
            eprintln!(
                "depth trace present {} texture {texture:x} now {identity:x?} (was {previous:x?}): targets move",
                self.presented_frames
            );
        }
        if let Some(backend) = self.gpu.as_mut() {
            backend.remove_texture_upload(texture);
            backend.texture_checked.remove(&texture);
            let renderer = &mut backend.renderer;
            let retired = super::retired_texture(previous);
            if renderer.target_info(texture).is_some() {
                backend.retired_since.insert(retired, self.presented_frames);
            }
            if let Some(identity) = identity {
                backend
                    .retired_since
                    .remove(&super::retired_texture(identity));
            }
            let result = renderer
                .move_target(texture, retired)
                .and_then(|_| match identity {
                    Some(identity) => {
                        renderer.move_target(super::retired_texture(identity), texture)
                    }
                    None => Ok(()),
                });
            if let Err(error) = result {
                backend.note(format!("moving a reused texture's target: {error}"));
            }
            backend.stat("stat: targets moved with their memory", 1);
        }
    }

    /// Copy the GPU's latest texels for a rendered texture. Return false when
    /// the texture's bytes are in guest memory.
    pub(super) fn gpu_read_texels(
        &mut self,
        core: &mut Guest,
        texture: u64,
        transfer: super::TexelTransfer,
        destination: u64,
    ) -> Result<bool, String> {
        let Some((key, base_level)) = self.resolve_target(texture) else {
            return Ok(false);
        };
        let object = self.object(texture, "Texture")?;
        let get = |key: &str, default| {
            object
                .builder
                .get(key)
                .map_or(default, |args| args.x[0] as u32)
        };
        let format = get("Format", 0);
        let info = crate::nvn_format::info(format)
            .ok_or_else(|| format!("texture format {format} is not modeled"))?;
        let [x, y, z, width, height, layers] = transfer.region_for(get("Target", 1));
        let linear = transfer.layout(object)?;
        if linear.size == 0 {
            return Ok(true);
        }
        let [level, layer] = transfer.bases;
        let level = u32::try_from(level)
            .ok()
            .and_then(|level| base_level.checked_add(level))
            .ok_or("target copy mip level overflows")?;
        let layer = u32::try_from(layer)
            .ok()
            .and_then(|layer| z.checked_add(layer))
            .ok_or("target copy layer overflows")?;
        let backend = self.gpu.as_mut().unwrap();
        backend.renderer.restore_archived_target(key)?;
        let target = backend.renderer.existing_target(key).unwrap();
        let exact_format = vk_format(format).is_some_and(|format| {
            format == target.format
                || echora_gpu::renderer::srgb_twin(target.format) == Some(format)
        });
        if info.block_width != 1
            || info.block_height != 1
            || !exact_format
            || matches!(
                target.format,
                vk::Format::D24_UNORM_S8_UINT | vk::Format::D32_SFLOAT_S8_UINT
            )
        {
            return Err(format!(
                "raw readback of rendered format {} is unsupported",
                info.name
            ));
        }
        let planes = match target.shape {
            ImageShape::D3 { depth } if level < 32 => (depth >> level).max(1),
            shape => shape.layers(),
        };
        if level >= target.levels || layer.checked_add(layers).is_none_or(|end| end > planes) {
            return Err("texture copy subresources are outside the rendered target".into());
        }
        let texel = info.bytes_per_block;
        let row_bytes = width as usize * texel as usize;
        for plane in 0..layers {
            let bytes = backend.renderer.read_target_plane_bytes(
                key,
                level,
                layer + plane,
                texel,
                [x, y, width, height],
            )?;
            let destination = destination + u64::from(plane) * linear.image_stride;
            if linear.row_stride == row_bytes as u64 {
                core.memory
                    .write(destination, &bytes)
                    .map_err(|trap| format!("texture copy destination: {trap:?}"))?;
            } else {
                for (row, bytes) in bytes.chunks_exact(row_bytes).enumerate() {
                    core.memory
                        .write(destination + row as u64 * linear.row_stride, bytes)
                        .map_err(|trap| format!("texture copy destination: {trap:?}"))?;
                }
            }
        }
        backend.stat("stat: texture readbacks", 1);
        backend.stat(
            "stat: texture readback bytes",
            row_bytes as u64 * u64::from(height) * u64::from(layers),
        );
        if super::debug_var_os("ECHORA_GPU_FLOW").is_some() {
            backend.note(format!(
                "texels read from GPU {key:x} level {level} layers {layer}+{layers}: {} region {x},{y} {width}x{height}",
                info.name
            ));
        }
        Ok(true)
    }

    /// The presented texture's texels, read back from its GPU target.
    pub(super) fn gpu_read_back(&mut self, texture: u64) -> Result<Option<Image>, String> {
        let backend = self.gpu.as_mut().unwrap();
        let Some(target) = backend.renderer.existing_target(texture) else {
            return Ok(None);
        };
        let format = target.format;
        let (mut texels, width, height) = backend.renderer.read_target_scaled(texture)?;
        // Readback decodes sRGB targets to linear values; a presented
        // frame is the display texture's stored (encoded) texels.
        if matches!(
            format,
            vk::Format::R8G8B8A8_SRGB | vk::Format::B8G8R8A8_SRGB
        ) {
            for texel in &mut texels {
                for channel in &mut texel[..3] {
                    *channel = linear_to_srgb(*channel);
                }
            }
        }
        Ok(Some(Image {
            width,
            height,
            texels,
        }))
    }

    /// Debug aid: ECHORA_CACHE_SIZES=<file> appends the sizes of the NVN
    /// and GPU caches every 600th present (finding memory growth).
    pub(super) fn log_cache_sizes(&self) {
        let Some(path) = super::debug_var_os("ECHORA_CACHE_SIZES") else {
            return;
        };
        // ECHORA_CACHE_SIZES_EVERY=N presents (600).
        let every = super::debug_var("ECHORA_CACHE_SIZES_EVERY")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(600)
            .max(1);
        if !self.presented_frames.is_multiple_of(every) {
            return;
        }
        let commands = self.command_list_commands;
        let captured: usize = self
            .command_lists
            .values()
            .flat_map(|list| list.commands.iter())
            .map(|command| command.captured.len() * 8 + command.states.len() * 512)
            .sum();
        let mut line = format!(
            "present {} objects {} bags {} views {} finalized {} images {} lists {} commands {commands} captured~{}MB recording {}",
            self.presented_frames,
            self.objects.len(),
            self.bags.len(),
            self.views.len(),
            self.finalized_storage.len(),
            self.images.len(),
            self.command_lists.len(),
            captured >> 20,
            super::read_shared(&self.recording)
                .values()
                .map(|recording| super::lock_shared(recording).commands.len())
                .sum::<usize>(),
        );
        if let Some(backend) = self.gpu.as_ref() {
            let mut seen = std::collections::HashSet::new();
            let texels: usize = backend
                .textures
                .values()
                .filter_map(|(_, source, _)| match source {
                    TextureSource::Upload { texels, .. } => seen
                        .insert(std::sync::Arc::as_ptr(texels))
                        .then_some(texels.len()),
                    _ => None,
                })
                .sum();
            let shadows: usize = backend.shadows.values().map(Vec::len).sum();
            line += &format!(
                " | textures {} texels {}MB shadows {} {}MB hashes {} dirty {} checked {} notes {} pending {} index {}",
                backend.textures.len(),
                texels >> 20,
                backend.shadows.len(),
                shadows >> 20,
                backend.storage_hashes.len(),
                backend.dirty.len(),
                backend.texture_checked.len(),
                backend.notes.len(),
                backend.pending_storage.len(),
                backend.target_index.storage.len(),
            );
            for (name, size) in backend.renderer.cache_sizes() {
                line += &format!(" {name} {size}");
            }
        }
        use std::io::Write;
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            let _ = writeln!(file, "{line}");
        }
    }

    /// Debug aid: with ECHORA_GPU_DUMP set to a directory, every 300th
    /// present writes each color target as target-<present>-<texture>.ppm,
    /// to see where draws land when the presented image does not change.
    /// ECHORA_GPU_DUMP_FROM=N starts at present N (past a long movie).
    pub(super) fn gpu_dump_targets(&mut self) -> Result<(), String> {
        let Some(directory) = super::debug_var_os("ECHORA_GPU_DUMP") else {
            return Ok(());
        };
        let from = super::debug_var("ECHORA_GPU_DUMP_FROM")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(0);
        if !self.presented_frames.is_multiple_of(300) || self.presented_frames < from {
            return Ok(());
        }
        let directory = std::path::PathBuf::from(directory);
        std::fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
        let present = self.presented_frames;
        let backend = self.gpu.as_mut().unwrap();
        for key in backend.renderer.target_keys() {
            let target = backend.renderer.existing_target(key).unwrap();
            if echora_gpu::renderer::is_depth(target.format) {
                // Depth: the probed values only.
                let (width, height) = (target.width as usize, target.height as usize);
                if let Ok(probes) = super::debug_var("ECHORA_GPU_PROBE") {
                    let bytes = backend.renderer.read_target_bytes(key, 4)?;
                    let mut line = format!("{key:x} {width}x{height} depth\n");
                    for probe in probes.split_whitespace() {
                        let Some((x, y)) = probe.split_once(',') else {
                            continue;
                        };
                        let (Ok(x), Ok(y)) = (x.parse::<usize>(), y.parse::<usize>()) else {
                            continue;
                        };
                        let offset = ((y * height / 1080) * width + x * width / 1920) * 4;
                        if let Some(word) = bytes.get(offset..offset + 4) {
                            let value = f32::from_le_bytes(word.try_into().unwrap());
                            line += &format!("  probe {probe} = {value}\n");
                        }
                    }
                    use std::io::Write;
                    if let Ok(mut file) = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(directory.join(format!("stats-{present:05}.txt")))
                    {
                        let _ = file.write_all(line.as_bytes());
                    }
                }
                continue;
            }
            let (width, height) = (target.width, target.height);
            let (vk_format, shape) = (target.format, target.shape);
            let nvn_format = self
                .objects
                .get(&key)
                .and_then(|texture| texture.builder.get("Format"))
                .map_or(0, |format| format.x[0]);
            // Deeper mip levels: statistics only.
            for level in 1..target.levels {
                let texels = backend.renderer.read_target_level(key, level)?;
                let finite = texels.iter().filter(|t| t.iter().all(|v| v.is_finite()));
                let (mut sum, mut count, mut max) = (0f64, 0usize, 0f32);
                for texel in finite {
                    sum += f64::from(texel[0]);
                    count += 1;
                    max = max.max(texel[0]);
                }
                use std::io::Write;
                if let Ok(mut file) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(directory.join(format!("stats-{present:05}.txt")))
                {
                    let _ = writeln!(
                        file,
                        "{key:x} level {level} mean-r {:.4} max-r {max:.3} nonfinite {}",
                        sum / count.max(1) as f64,
                        texels.len() - count
                    );
                }
            }
            let raw = backend.renderer.read_target(key)?;
            // Float statistics of every target (dumps clip HDR values).
            {
                use std::io::Write;
                let finite: Vec<&[f32; 4]> = raw
                    .iter()
                    .filter(|t| t.iter().all(|v| v.is_finite()))
                    .collect();
                let nan = raw.len() - finite.len();
                let mut mean = [0f64; 4];
                let mut max = 0f32;
                for texel in &finite {
                    for (channel, value) in texel.iter().enumerate() {
                        mean[channel] += f64::from(*value);
                    }
                    max = max.max(texel[0].max(texel[1]).max(texel[2]));
                }
                let count = finite.len().max(1) as f64;
                let mut line = format!(
                    "{key:x} {width}x{height} mean {:.4} {:.4} {:.4} {:.4} max {max:.3} nonfinite {nan} {vk_format:?} nvn {nvn_format:#x} {shape:?}\n",
                    mean[0] / count,
                    mean[1] / count,
                    mean[2] / count,
                    mean[3] / count
                );
                // ECHORA_GPU_PROBE="x,y x,y" (1920x1080 coordinates,
                // scaled to the target): exact texels, to compare with a
                // reference capture.
                if let Ok(probes) = super::debug_var("ECHORA_GPU_PROBE") {
                    for probe in probes.split_whitespace() {
                        let Some((x, y)) = probe.split_once(',') else {
                            continue;
                        };
                        let (Ok(x), Ok(y)) = (x.parse::<usize>(), y.parse::<usize>()) else {
                            continue;
                        };
                        let (x, y) = (x * width as usize / 1920, y * height as usize / 1080);
                        if let Some(texel) = raw.get(y * width as usize + x) {
                            line += &format!("  probe {probe} = {texel:?}\n");
                        }
                    }
                }
                if let Ok(mut file) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(directory.join(format!("stats-{present:05}.txt")))
                {
                    let _ = file.write_all(line.as_bytes());
                }
            }
            // Tiny targets (exposure values, ...) also as exact floats.
            if width * height <= 64 {
                let text: String = raw.iter().map(|texel| format!("{texel:?}\n")).collect();
                let path = directory.join(format!("target-{present:05}-{key:x}.txt"));
                std::fs::write(&path, text).map_err(|error| error.to_string())?;
            }
            // Probing compares numbers: no images (they fill the disk).
            if super::debug_var_os("ECHORA_GPU_PROBE").is_some() {
                continue;
            }
            let texels = raw;
            let image = Image {
                width,
                height,
                texels,
            };
            let path = directory.join(format!("target-{present:05}-{key:x}.ppm"));
            std::fs::write(&path, image.ppm()).map_err(|error| error.to_string())?;
        }
        Ok(())
    }

    /// The cached translation and code key of the stage at GPU address `gpu`.
    fn translate_stage(
        &mut self,
        core: &Guest,
        gpu: u64,
        // First binding, texture scaling bit and image scaling bit.
        base: [u32; 3],
        previous_stores: Option<[u64; 8]>,
        generic_input_types: [u8; 32],
        compute: [u32; 5],
        mesh: Option<MeshParameters>,
        compute_constants: Option<&[u8]>,
        sampled_textures: &[shader::SampledTexture],
    ) -> Result<(Arc<Translated>, u64), String> {
        let (code, key) = self
            .stage_code(core, gpu)
            .ok_or_else(|| format!("stage at {gpu:#x} is in no memory pool"))?;
        // ECHORA_SHADER_DIR: a stage whose code was not yet readable when
        // its program was set is written when first translated.
        if let Some(directory) = super::debug_var_os("ECHORA_SHADER_DIR") {
            let path = std::path::Path::new(&directory).join(format!("stage-{key:016x}.code"));
            if !path.exists() {
                let _ = std::fs::create_dir_all(&directory);
                let _ = std::fs::write(path, code.as_slice());
            }
        }
        let fragment_output_types = if previous_stores.is_some() {
            self.fragment_output_types()
        } else {
            [0; MAX_COLOR_TARGETS]
        };
        let typed_outputs = fragment_output_types != [0; MAX_COLOR_TARGETS];
        let backend = self.gpu.as_mut().unwrap();
        let native_pointers = backend.native_programs.contains(&key);
        let jump_targets = compute_constants.and_then(|constants| {
            backend
                .compute_jumps
                .entry(key)
                .or_insert_with(|| ComputeJumps::from_program(&code))
                .as_mut()?
                .resolve(constants)
        });
        let constants_for_translation = jump_targets.as_ref().and(compute_constants);
        let cache_key = (
            key,
            base,
            previous_stores,
            generic_input_types,
            fragment_output_types,
            compute,
            mesh,
            native_pointers,
            jump_targets,
            sampled_textures.to_vec(),
        );
        if let Some(found) = backend.translations.get(&cache_key) {
            return found.clone().map(|stage| (stage, key));
        }
        let resolution_scale = self
            .gpu
            .as_ref()
            .unwrap()
            .renderer
            .resolution_scale()
            .terms();
        // Programs reading the texture pool need the device to hold it.
        let pool_ready = self
            .gpu
            .as_ref()
            .unwrap()
            .renderer
            .gpu
            .texture_pool_layout
            .is_some();
        let mut notes: Vec<String> = Vec::new();
        let mut builtin = false;
        let translated = {
            let parameters = Parameters {
                binding_base: base[0],
                resolution_scale,
                scaling_base: [base[1], base[2]],
                mesh,
                previous_stores,
                generic_input_types,
                fragment_output_types,
                workgroup_size: [compute[0], compute[1], compute[2]],
                // Compute shared memory from the control section; zero
                // (graphics) is raised to 48 KiB, as zero aborts the
                // recompiler on shared accesses.
                shared_memory_size: if compute[3] == 0 { 0xc000 } else { compute[3] },
                // Local (per-invocation stack) memory: without it the
                // recompiler drops stores at constant offsets.
                local_memory_size: compute[4],
                native_pointers,
                // Instrument only these stage keys when a trace of every
                // translated stage would be too expensive to reach the frame.
                vertex_trace: std::env::var("ECHORA_VERTEX_TRACE_PROGRAMS")
                    .ok()
                    .map(|keys| {
                        keys.split(',').any(|item| {
                            u64::from_str_radix(item.trim().trim_start_matches("0x"), 16) == Ok(key)
                        })
                    }),
            };
            // ECHORA_SHADER_TRACE=FILE (debugging): each translation's
            // stage key and parameters, appended before it runs, so the
            // last line names a translation that aborts the process.
            if let Some(path) = super::debug_var_os("ECHORA_SHADER_TRACE") {
                use std::io::Write as _;
                if let Ok(mut file) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                {
                    let _ = writeln!(file, "{key:016x} {parameters:?}");
                }
            }
            // Compute programs see the bindings of the dispatch that
            // first runs them (which constant buffer words are
            // pointers); graphics stages bind every candidate.
            let plugin = |nvn: &Self| {
                if compute[0] != 0 {
                    shader::translate(
                        &code,
                        parameters,
                        &mut PointerProbe {
                            nvn,
                            core,
                            c1: compute_constants.unwrap_or_default(),
                        },
                    )
                } else {
                    shader::translate(&code, parameters, &mut shader::NoEnvironment)
                }
            };
            // ECHORA_SHADER_TRANSLATOR: the Rust translator for programs it
            // handles (`rust`), or beside the plugin for a report
            // (`compare`); what it does not handle is noted by reason.
            // ECHORA_PLUGIN_PROGRAMS=<stage key>[,...] (debugging): those
            // stages from the plugin whatever the translator, to tell a Rust
            // translation's fault from its neighbours'.
            // The plugin ABI has no attachment numeric types.
            let translator = if typed_outputs {
                shader::Translator::Rust
            } else if super::debug_var("ECHORA_PLUGIN_PROGRAMS").is_ok_and(|keys| {
                keys.split(',')
                    .any(|item| item.trim() == format!("{key:016x}"))
            }) {
                shader::Translator::Plugin
            } else {
                shader::Translator::from_env()
            };
            // A program reading the texture pool (set 1) needs a device
            // that holds it; elsewhere such programs stay with the plugin.
            let rust = (translator != shader::Translator::Plugin)
                .then(|| {
                    shader::translate_rust_with_sampled_textures(
                        &code,
                        &parameters,
                        constants_for_translation,
                        sampled_textures,
                    )
                })
                .map(|result| {
                    result.and_then(|translated| {
                        if translated.texture_pool_bindings == 0 || pool_ready {
                            Ok(translated)
                        } else {
                            Err(echora_shader_abi::Unsupported {
                                reason: "the device cannot hold the texture pool".into(),
                                offset: None,
                            })
                        }
                    })
                });
            let result = match (translator, rust) {
                (shader::Translator::Rust, Some(Ok(translated))) => {
                    builtin = true;
                    notes.push("stat: rust translator programs".into());
                    Ok(translated)
                }
                (_, Some(Err(unsupported))) => {
                    notes.push(format!("rust translator: {}", unsupported.reason));
                    if shader::available() && !typed_outputs {
                        plugin(self)
                    } else {
                        Err(match unsupported.offset {
                            Some(offset) => format!("{} (at byte {offset:#x})", unsupported.reason),
                            None => unsupported.reason,
                        })
                    }
                }
                (shader::Translator::Compare, Some(Ok(rust))) => {
                    let plugin = plugin(self);
                    if let Ok(plugin) = &plugin {
                        notes.push(match shader::compare(plugin, &rust) {
                            Some(difference) => format!("rust translator differs: {difference}"),
                            None => "stat: rust translator alike".into(),
                        });
                    }
                    plugin
                }
                _ => plugin(self),
            };
            result
                .map(Arc::new)
                .map_err(|error| format!("stage {key:016x}: {error}"))
        };
        let backend = self.gpu.as_mut().unwrap();
        for note in notes {
            backend.note(note);
        }
        // Without a translator plugin, a program the Rust translator does
        // not handle stops the run with its report: nothing is drawn wrong
        // in silence.
        if let Err(error) = &translated
            && (typed_outputs || !shader::available())
        {
            backend.renderer.fatal = Some(format!("untranslated shader: {error}"));
        }
        if let Ok(stage) = &translated
            && stage.texture_pool_bindings != 0
            && let Some(reads) = shader::handle_reads::HandleReads::from_program(&code)
        {
            backend
                .handle_reads
                .insert(Arc::as_ptr(stage) as usize, Arc::new(reads));
        }
        if builtin
            && let Ok(stage) = &translated
            && stage.stage == Stage::Compute
            && stage.push_fields & echora_shader_abi::PUSH_PAGE_TABLE != 0
            && stage.push_fields & echora_shader_abi::PUSH_PAGE_TABLE_READ_ONLY == 0
        {
            let id = Arc::as_ptr(stage) as usize;
            if let Some(writes) = shader::handle_reads::GlobalWrites::from_program(&code) {
                let local = [compute[0], compute[1], compute[2]].map(|size| size.max(1));
                backend.global_writes.insert(id, (writes, local));
            }
            if let Some(writes) = shader::handle_reads::UniformWriteRanges::from_program(&code) {
                backend.uniform_writes.insert(id, writes);
            }
        }
        backend.translations.insert(cache_key, translated.clone());
        translated.map(|stage| (stage, key))
    }

    /// The memory a raw storage pointer candidate of a compute program
    /// names (`Descriptor::flags` bit 5): the eight bytes at its place in
    /// c0 (`c0`, as filled), the stage's constants (c1) or bound uniform
    /// buffer `index - 3`, read through `indirect_offset` when set. The
    /// pointer carries no size: the binding runs to the end of its memory
    /// pool, at most `NATIVE_STORAGE_SPAN`. `None` when it maps nowhere.
    fn native_pointer(
        &mut self,
        core: &mut Guest,
        descriptor: &shader::Descriptor,
        c0: &[u8],
        c1: &[u8],
    ) -> Result<Option<(u64, u64)>, String> {
        let at = descriptor.cbuf_offset as usize;
        let mut word = [0u8; 8];
        match descriptor.cbuf_index {
            0 => match c0.get(at..at + 8) {
                Some(bytes) => word.copy_from_slice(bytes),
                None => return Ok(None),
            },
            1 => match c1.get(at..at + 8) {
                Some(bytes) => word.copy_from_slice(bytes),
                _ => return Ok(None),
            },
            2 => return Ok(None),
            index => {
                let Some(bind) = self
                    .render
                    .bound_uniforms
                    .get(5, u64::from(index - 3))
                    .map(|bind| (bind.x[3], bind.x[4]))
                else {
                    return Ok(None);
                };
                if at as u64 + 8 > bind.1 {
                    return Ok(None);
                }
                if !self.read_gpu_word(core, bind.0 + at as u64, &mut word)? {
                    return Ok(None);
                }
            }
        }
        let mut pointer = u64::from_le_bytes(word);
        if descriptor.indirect_offset != u32::MAX {
            let at = pointer.wrapping_add(u64::from(descriptor.indirect_offset));
            if !self.read_gpu_word(core, at, &mut word)? {
                return Ok(None);
            }
            pointer = u64::from_le_bytes(word);
        }
        let base = pointer & !15;
        Ok(self
            .pool_for(base)
            .map(|(_, remaining)| (base, remaining.min(NATIVE_STORAGE_SPAN))))
    }

    /// The eight bytes at GPU address `address`, after the dispatch results
    /// that may have written them; false when no pool maps it.
    fn read_gpu_word(
        &mut self,
        core: &mut Guest,
        address: u64,
        word: &mut [u8; 8],
    ) -> Result<bool, String> {
        let Ok(guest) = self.gpu_to_guest(address, 8) else {
            return Ok(false);
        };
        self.resolve_storage(core, Some(guest..guest + 8))?;
        Ok(core.memory.read(guest, word).is_ok())
    }

    /// The constants a stage's control section places beside its code.
    fn stage_constants_of(&self, core: &Guest, stage: super::Args) -> Option<Vec<u8>> {
        let (code, control) = (stage.x[0], stage.x[1]);
        let size = core.memory.read_u32(control + 0x3c).ok()?;
        let offset = core.memory.read_u32(control + 0x40).ok()?;
        if size == 0 || size > 0x10000 {
            return None;
        }
        let guest = self
            .gpu_to_guest(code + u64::from(offset), u64::from(size))
            .ok()?;
        let mut bytes = vec![0u8; size as usize];
        core.memory.read(guest, &mut bytes).ok()?;
        Some(bytes)
    }

    /// The stage of `program` bound for stage bit `bit` (0 vertex, 1
    /// fragment): a program may hold one stage, or several (SetShadersExt
    /// gives one program both), told apart by the shader program header's
    /// type (word 0 bits 10..13: 1 vertex, 5 pixel; Maxwell SPH layout).
    fn program_stage(&self, core: &Guest, program: u64, bit: u32) -> Result<super::Args, String> {
        let program = self
            .objects
            .get(&program)
            .ok_or("bound program is not initialized")?;
        let mut stages = super::PROGRAM_STAGE_KEYS
            .iter()
            .map_while(|key| program.state.get(*key).copied());
        let first = stages.next().ok_or("bound program has no shaders")?;
        let Some(second) = stages.next() else {
            return Ok(first);
        };
        let wanted = match bit {
            0 => 1,
            1 => 5,
            _ => return Err(format!("stage bit {bit} is not modeled")),
        };
        [first, second]
            .into_iter()
            .chain(stages)
            .find(|stage| {
                self.stage_code(core, stage.x[0]).is_some_and(|(code, _)| {
                    code.len() >= 4
                        && (u32::from_le_bytes([code[0], code[1], code[2], code[3]]) >> 10) & 0xf
                            == wanted
                })
            })
            .ok_or_else(|| format!("bound program has no stage of type {wanted}"))
    }

    fn bound_program(&mut self, core: &Guest) -> Result<Arc<BoundProgram>, String> {
        // The same programs as the last draw, within a present and with no
        // flush of guest memory since (program code is revalidated once a
        // present, as textures are): its stages again.
        let bind = |bit: u32| {
            self.render
                .state
                .get(super::program_key(bit))
                .map(|command| command.x[1])
        };
        let attributes = self
            .render
            .state
            .get("BindVertexAttribState")
            .map(|command| command.states.clone())
            .unwrap_or_default();
        let stamp = (self.presented_frames, self.flush_epoch);
        if self.program_memo.0 != stamp {
            self.program_memo = (stamp, HashMap::default());
        }
        let key = bind(0)
            .zip(bind(1))
            .map(|(vertex, fragment)| (vertex, fragment, self.fragment_output_types()));
        if let Some(key) = key
            && let Some((memo_attributes, bound)) = self.program_memo.1.get(&key)
            && attributes.ptr_eq(memo_attributes)
        {
            return Ok(bound.clone());
        }
        let bound = Arc::new(self.resolve_bound_program(core)?);
        if let Some(key) = key {
            self.program_memo.1.insert(key, (attributes, bound.clone()));
        }
        Ok(bound)
    }

    fn resolve_bound_program(&mut self, core: &Guest) -> Result<BoundProgram, String> {
        let entry = |bit| {
            let bind = self
                .render
                .state
                .get(super::program_key(bit))
                .ok_or_else(|| format!("no program bound for stage {bit}"))?;
            self.program_stage(core, bind.x[1], bit)
        };
        let entries = [entry(0)?, entry(1)?];
        let addresses = entries.map(|stage| stage.x[0]);
        let types = self
            .render
            .state
            .get("BindVertexAttribState")
            .map_or([0; 32], |bind| generic_input_types(&bind.states));
        let (vertex, vertex_key) = self.translate_stage(
            core,
            addresses[0],
            [0; 3],
            None,
            types,
            [0; 5],
            None,
            None,
            &[],
        )?;
        let binding_base = vertex
            .descriptors
            .iter()
            .map(|descriptor| descriptor.binding + descriptor.count)
            .max()
            .unwrap_or(0);
        let count = |kind| {
            vertex
                .descriptors
                .iter()
                .filter(|descriptor| descriptor.kind() == kind)
                .count() as u32
        };
        let base = [
            binding_base,
            count(DescriptorKind::Texture),
            count(DescriptorKind::Image),
        ];
        // Fragment inputs the vertex stage does not write read as zero, so
        // the fragment stage is translated against the vertex stage's stores.
        let (fragment, fragment_key) = self.translate_stage(
            core,
            addresses[1],
            base,
            Some(vertex.stores),
            [0; 32],
            [0; 5],
            None,
            None,
            &[],
        )?;
        if vertex.stage != Stage::Vertex || fragment.stage != Stage::Fragment {
            return Err(format!(
                "bound stages are {:?} and {:?}",
                vertex.stage, fragment.stage
            ));
        }
        Ok(BoundProgram::new(
            (vertex_key, fragment_key),
            Box::new([vertex, fragment]),
            Box::new(entries),
        ))
    }

    /// The task (if any), mesh and fragment stages of the programs bound for
    /// a mesh draw: their keys, translations and stage entries. A mesh
    /// program writes the stage buffer, and its control
    /// section gives its largest vertex and primitive counts at +0x60 (low
    /// and high half) and its threads at +0x58; a task program has nothing
    /// at +0x60, and its +0x58 is laid out as SET_MESH_INIT_SHADER (threads
    /// 5:0, payload lines 30:22); a vertex program has neither. Without a task the mesh program sits in
    /// the vertex slot (its header names a vertex program), after one in the
    /// tessellation slot; a pass-through geometry program bound with it
    /// lists the per-primitive outputs.
    fn bound_mesh_program(&mut self, core: &Guest) -> Result<Arc<BoundProgram>, String> {
        // As `bound_program`: the same binds within a present and with no
        // shader-code flush or program change since give the same stages.
        let binds = std::array::from_fn(|slot| {
            if slot == nvn_stage(Stage::Compute) as usize {
                return None;
            }
            self.render
                .state
                .get(super::program_key(slot as u32))
                .map(|bind| bind.x[1])
        });
        let stamp = (self.presented_frames, self.flush_epoch);
        if self.mesh_program_memo.0 != stamp {
            self.mesh_program_memo = (stamp, HashMap::default());
        }
        let key = (binds, self.fragment_output_types());
        if let Some(found) = self.mesh_program_memo.1.get(&key) {
            return Ok(found.clone());
        }
        let found = Arc::new(self.resolve_bound_mesh_program(core)?);
        self.mesh_program_memo.1.insert(key, found.clone());
        Ok(found)
    }

    fn resolve_bound_mesh_program(&mut self, core: &Guest) -> Result<BoundProgram, String> {
        // The graphics stages' programs: the COMPUTE bit's (5, NVN's
        // 0x20) is not one. A program bound for every bit (0xff, as the
        // game binds its tile classifier) stays in the compute slot while
        // later graphics binds (0xdf) replace the rest; its fragment stage
        // then drew every following mesh draw.
        let programs: Vec<u64> = self
            .render
            .state
            .iter()
            .filter(|(key, _)| key.starts_with("BindProgram@") && **key != "BindProgram@5")
            .map(|(_, bind)| bind.x[1])
            .collect();
        // For the skip reasons: each bound bit's program and its stages'
        // header types and +0x60 words.
        let bound: Vec<String> = self
            .render
            .state
            .iter()
            .filter_map(|(key, bind)| Some((key.strip_prefix("BindProgram@")?, bind)))
            .map(|(bit, bind)| {
                let stages: Vec<String> = (0..6)
                    .filter_map(|index| {
                        let stage = self
                            .objects
                            .get(&bind.x[1])?
                            .state
                            .get(&format!("Stage{index}"))?;
                        let (code, _) = self.stage_code(core, stage.x[0])?;
                        let kind = code.get(..4).map_or(0, |word| {
                            (u32::from_le_bytes(word.try_into().unwrap()) >> 10) & 0xf
                        });
                        let sizes = core.memory.read_u32(stage.x[1] + 0x60).unwrap_or(0);
                        Some(format!("t{kind}/{sizes:x}"))
                    })
                    .collect();
                format!("{bit}:{}", stages.join("+"))
            })
            .collect();
        let bound = bound.join(" ");
        let mut stages: Vec<super::Args> = Vec::new();
        for program in programs {
            let Some(object) = self.objects.get(&program) else {
                continue;
            };
            for index in 0..6 {
                if let Some(stage) = object.state.get(&format!("Stage{index}")) {
                    if !stages.iter().any(|known| known.x[0] == stage.x[0]) {
                        stages.push(*stage);
                    }
                }
            }
        }
        let control = |stage: &super::Args, offset: u64| {
            core.memory.read_u32(stage.x[1] + offset).unwrap_or(0)
        };
        let (mut task, mut mesh, mut fragment, mut geometry) = (None, None, None, None);
        for stage in &stages {
            let Some((code, _)) = self.stage_code(core, stage.x[0]) else {
                continue;
            };
            if code.len() < 0x50 {
                continue;
            }
            let kind = (u32::from_le_bytes([code[0], code[1], code[2], code[3]]) >> 10) & 0xf;
            match kind {
                5 => fragment = Some(*stage),
                4 => {
                    let mut header = [0u8; 0x50];
                    header.copy_from_slice(&code[..0x50]);
                    geometry = Some(header);
                }
                1 | 3 if control(stage, 0x60) != 0 => mesh = Some((*stage, kind)),
                // A task program's threads (+0x58 bits 5:0) are never 0; a
                // vertex program has 0 there. One is still bound from the
                // draws before when a mesh program replaces only other
                // stages (the game's first mesh draws after the opening
                // movie): it takes no part in the mesh draw.
                1 if control(stage, 0x58) & 0x3f != 0 => task = Some(*stage),
                _ => {}
            }
        }
        let (mesh, _) = mesh.ok_or_else(|| format!("no mesh program is bound ({bound})"))?;
        let fragment = fragment.ok_or_else(|| {
            format!("no fragment program is bound with the mesh program ({bound})")
        })?;
        let sizes = control(&mesh, 0x60);
        let (vertices, primitives) = (sizes & 0xffff, sizes >> 16);
        // Triangles, the only mesh output topology the recompiler takes.
        let shader_a = 4 | (vertices & 0x1ff) << 4 | (primitives & 0x3ff) << 13;
        let shader_b = (control(&mesh, 0x58) & 0x3f) << 20;
        let task_init = task.map(|task| control(&task, 0x58));
        let init = task_init.unwrap_or(control(&mesh, 0x58) & !0x3f);
        let advance = |stage: &Translated, base: [u32; 3]| {
            let count = |kind| {
                stage
                    .descriptors
                    .iter()
                    .filter(|descriptor| descriptor.kind() == kind)
                    .count() as u32
            };
            [
                stage
                    .descriptors
                    .iter()
                    .map(|descriptor| descriptor.binding + descriptor.count)
                    .max()
                    .unwrap_or(base[0])
                    .max(base[0]),
                base[1] + count(DescriptorKind::Texture),
                base[2] + count(DescriptorKind::Image),
            ]
        };
        let (mut translated, mut entries) = (Vec::new(), Vec::new());
        let mut base = [0u32; 3];
        let mut keys = 0u64;
        if let (Some(task), Some(init)) = (task, task_init) {
            let parameters = MeshParameters {
                task: true,
                registers: [shader_a, shader_b, init],
                primitive_header: None,
            };
            let (stage, key) = self.translate_stage(
                core,
                task.x[0],
                base,
                None,
                [0; 32],
                [0; 5],
                Some(parameters),
                None,
                &[],
            )?;
            if stage.stage != Stage::Task {
                return Err(format!("the task program translated as {:?}", stage.stage));
            }
            base = advance(&stage, base);
            keys = key;
            translated.push(stage);
            entries.push(task);
        }
        let parameters = MeshParameters {
            task: false,
            registers: [shader_a, shader_b, init],
            primitive_header: geometry,
        };
        let (stage, key) = self.translate_stage(
            core,
            mesh.x[0],
            base,
            None,
            [0; 32],
            [0; 5],
            Some(parameters),
            None,
            &[],
        )?;
        if stage.stage != Stage::Mesh {
            return Err(format!("the mesh program translated as {:?}", stage.stage));
        }
        base = advance(&stage, base);
        let stores = stage.stores;
        keys = keys.rotate_left(1) ^ key;
        translated.push(stage);
        entries.push(mesh);
        let (stage, fragment_key) = self.translate_stage(
            core,
            fragment.x[0],
            base,
            Some(stores),
            [0; 32],
            [0; 5],
            None,
            None,
            &[],
        )?;
        if stage.stage != Stage::Fragment {
            return Err(format!(
                "the fragment program translated as {:?}",
                stage.stage
            ));
        }
        translated.push(stage);
        entries.push(fragment);
        // Mesh programs never share a key with vertex programs' pipelines.
        Ok(BoundProgram::new(
            (keys ^ 0x6d65_7368, fragment_key),
            translated.into(),
            entries.into(),
        ))
    }

    /// A mesh draw's workgroups: DrawMeshTasks(cmd, first, count) draws
    /// `count` groups (from 0 only), DrawMeshTasksIndirect(cmd, address)
    /// reads one u32 group count at that GPU address, from the device
    /// buffer when the GPU may have written it.
    fn mesh_groups(&mut self, core: &mut Guest, command: &Command) -> Result<MeshGroups, String> {
        match command.name {
            "nvnCommandBufferDrawMeshTasks" => {
                let (first, count) = (command.x[1] as u32, command.x[2] as u32);
                if first != 0 {
                    return Err(format!("mesh tasks from group {first}"));
                }
                // ECHORA_DIAG_TASK_GROUPS (diagnostic runs only), as the
                // indirect draws' bound.
                let bound = super::debug_var("ECHORA_DIAG_TASK_GROUPS")
                    .ok()
                    .and_then(|value| value.parse::<u32>().ok())
                    .unwrap_or(u32::MAX);
                Ok(MeshGroups::Direct([count.min(bound), 1, 1]))
            }
            "nvnCommandBufferDrawMeshTasksIndirect" => {
                let length = 4;
                let address = command.x[1];
                let guest = self.gpu_to_guest(address, length)?;
                let read_count = |core: &Guest<'_>| {
                    core.memory
                        .read_u32(guest)
                        .map_err(|trap| format!("reading GPU {address:#x}: {trap:?}"))
                };
                // ECHORA_MESH_LOG (debugging): each indirect mesh draw's
                // group counts as the GPU wrote them (waits for the GPU).
                if super::debug_var_os("ECHORA_MESH_LOG").is_some() {
                    self.resolve_gpu_written(core, guest..guest + length)?;
                    let groups = [read_count(core)?, 1, 1];
                    eprintln!(
                        "mesh indirect present {} at {address:#x}: groups {groups:?}",
                        self.presented_frames
                    );
                }
                self.diagnostic_probe(core)?;
                if let Some((buffer, offset)) = self.indirect_arguments(core, guest, length)? {
                    return Ok(MeshGroups::Indirect { buffer, offset });
                }
                Ok(MeshGroups::Direct([read_count(core)?, 1, 1]))
            }
            other => Err(format!("{other} is not modeled on the GPU")),
        }
    }

    /// ECHORA_DIAG_PROBE=GPU address,... (diagnostic): at the first
    /// indirect mesh draw, each address's page table entry and the u32 the
    /// CPU sees there (printed now), and the u32 the GPU reads through that
    /// entry in command order (printed with the indirect counts, also when
    /// a wait stalls or the device is lost).
    fn diagnostic_probe(&mut self, core: &mut Guest) -> Result<(), String> {
        static DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let Ok(list) = super::debug_var("ECHORA_DIAG_PROBE") else {
            return Ok(());
        };
        if DONE.swap(true, std::sync::atomic::Ordering::Relaxed) {
            return Ok(());
        }
        use echora_shader_abi::PAGE_SHIFT;
        let table = self.page_table(core)?;
        let addresses: Vec<u64> = list
            .split(',')
            .filter_map(|item| u64::from_str_radix(item.trim().trim_start_matches("0x"), 16).ok())
            .collect();
        let mut device = Vec::new();
        for address in addresses {
            let entry = self.gpu.as_ref().and_then(|backend| {
                let build = backend.page_build.as_ref()?;
                let index = (address >> PAGE_SHIFT).checked_sub(build.first)? as usize;
                build.entries.get(index).copied()
            });
            let cpu = self
                .gpu_to_guest(address, 4)
                .ok()
                .and_then(|guest| core.memory.read_u32(guest).ok());
            eprintln!(
                "probe present {} GPU {address:#x}: table {:#x}+{} pages, entry {entry:x?}, CPU word {cpu:x?}",
                self.presented_frames, table.first_page, table.page_count
            );
            if let Some(entry) = entry.filter(|entry| *entry != 0) {
                device.push(entry + (address & ((1 << PAGE_SHIFT) - 1)));
            }
        }
        self.gpu
            .as_mut()
            .unwrap()
            .renderer
            .probe_device_words(&device)
    }

    /// Guest bytes at GPU address `address`, `size` long.
    fn gpu_bytes(&self, core: &Guest, address: u64, size: u64) -> Result<Vec<u8>, String> {
        let guest = self.gpu_to_guest(address, size)?;
        let mut bytes = vec![0u8; size as usize];
        core.memory
            .read(guest, &mut bytes)
            .map_err(|trap| format!("reading GPU {address:#x}: {trap:?}"))?;
        Ok(bytes)
    }

    /// Level 0 of a sampled guest texture, untiled (block-linear) into
    /// rows of format blocks.
    /// Sampled textures for the flow trace: `t<key>` for render targets,
    /// `u<key>:<w>x<h>:<format>` for uploads of guest memory, marked `~`
    /// when their storage overlaps a render target's texture.
    fn describe_sources<'a>(&self, sources: impl Iterator<Item = &'a TextureSource>) -> String {
        let targets: Vec<(u64, u64)> = self
            .gpu
            .as_ref()
            .map(|backend| backend.renderer.target_keys())
            .unwrap_or_default()
            .into_iter()
            .filter_map(|key| Some((key, self.texture_storage(key)?)))
            .collect();
        let mut keys: Vec<String> = sources
            .map(|source| match source {
                TextureSource::Target(key) => {
                    format!("t{key:x}@{:x}", self.texture_storage(*key).unwrap_or(0))
                }
                TextureSource::TargetView {
                    key,
                    base_level,
                    level_count,
                    base_layer,
                    layer_count,
                } => format!(
                    "t{key:x}@{:x}[{base_level}+{level_count}/{base_layer}+{layer_count}]",
                    self.texture_storage(*key).unwrap_or(0)
                ),
                TextureSource::Upload {
                    key,
                    width,
                    height,
                    shape,
                    format,
                    texels,
                    ..
                } => {
                    let value = if width * height <= 4 {
                        // Every layer of tiny textures, as floats when
                        // 16-byte texels (R32G32B32A32).
                        if texels.len() % 16 == 0 && texels.len() <= 256 {
                            let floats: Vec<f32> = texels
                                .as_chunks::<4>()
                                .0
                                .iter()
                                .map(|word| f32::from_le_bytes(*word))
                                .collect();
                            format!("={floats:?}")
                        } else {
                            format!("={:02x?}", &texels[..texels.len().min(64)])
                        }
                    } else {
                        String::new()
                    };
                    let base = self.texture_storage(*key);
                    let size = self.texture_storage_size(*key).unwrap_or(0);
                    // Targets over the same storage (~), or inside it (a
                    // mip level or layer of this texture: ^).
                    let shared = targets
                        .iter()
                        .filter_map(|(target, storage)| {
                            let base = base?;
                            if *storage == base {
                                Some(format!("~t{target:x}"))
                            } else if (base..base + size).contains(storage) {
                                Some(format!("^t{target:x}+{:#x}", storage - base))
                            } else {
                                None
                            }
                        })
                        .collect::<String>();
                    format!(
                        "u{key:x}:{width}x{height}:{shape:?}:{format:?}@{:x}{value}{shared}",
                        base.unwrap_or(0)
                    )
                }
            })
            .collect();
        keys.sort();
        format!("[{}]", keys.join(" "))
    }

    /// `key:WxH@storage` of written targets, for the flow trace.
    fn describe_targets(&self, keys: impl Iterator<Item = u64>) -> String {
        let described: Vec<String> = keys
            .map(|key| {
                let (width, height) = self.texture_extent(key).unwrap_or((0, 0));
                let levels = self
                    .objects
                    .get(&key)
                    .and_then(|object| object.builder.get("Levels"))
                    .map_or(1, |value| value.x[0]);
                let format = self
                    .objects
                    .get(&key)
                    .and_then(|object| object.builder.get("Format"))
                    .and_then(|value| crate::nvn_format::info(value.x[0] as u32))
                    .map_or("?", |info| info.name);
                format!(
                    "{key:x}:{width}x{height}/{levels}:{format}@{:x}",
                    self.texture_storage(key).unwrap_or(0)
                )
            })
            .collect();
        format!("[{}]", described.join(" "))
    }

    /// Bytes of a texture object's storage (all levels and layers).
    pub(super) fn texture_storage_size(&self, texture: u64) -> Option<u64> {
        let mut memo = self.size_memo.borrow_mut();
        if let Some(&size) = memo.get(&texture) {
            return size;
        }
        let size = self.texture_storage_size_uncached(texture);
        memo.insert(texture, size);
        size
    }

    fn texture_storage_size_uncached(&self, texture: u64) -> Option<u64> {
        let object = self.objects.get(&texture).filter(|o| o.kind == "Texture")?;
        let get = |key: &str| object.builder.get(key).map(|value| value.x[0] as u32);
        let info = crate::nvn_format::info(get("Format")?)?;
        let width = get("Width").unwrap_or(1);
        let target = get("Target").unwrap_or(1);
        let (height, depth) = texture_height_and_depth(
            target,
            get("Height").unwrap_or(1),
            get("Depth").unwrap_or(1),
        );
        let stride = texture_row_pitch(&object.builder, info, width);
        if stride != 0 {
            return stride.checked_mul(u64::from(height.div_ceil(info.block_height)));
        }
        let (depth, layers) = if target == 2 { (depth, 1) } else { (1, depth) };
        Some(
            crate::nvn_format::block_linear_size(
                info,
                width,
                height,
                depth,
                super::texture_layout_layers(
                    &object.builder,
                    info,
                    [width, height, depth],
                    layers,
                    get("Levels").unwrap_or(1),
                ),
                get("Levels").unwrap_or(1),
            )
            .0,
        )
    }

    /// The mip levels (base, count; 0: the rest) the texture view
    /// registered with a texture or image handle selects, if any.
    fn handle_view_levels(&self, handle: u64) -> Option<(u32, u32)> {
        let pool = self.render.state.get("SetTexturePool")?.x[1];
        let view = self.render_texture_view(pool, (handle & 0xf_ffff) as u32)?;
        let levels = view.props.get("Levels")?;
        Some((levels.x[0] as u32, levels.x[1] as u32))
    }

    /// The format the texture view registered with a handle sets, if any
    /// (the albedo and decal textures are UNORM and sampled through sRGB
    /// views, as a PokeDinner capture shows them).
    fn handle_view_format(&self, handle: u64) -> Option<vk::Format> {
        let pool = self.render.state.get("SetTexturePool")?.x[1];
        let view = self.render_texture_view(pool, (handle & 0xf_ffff) as u32)?;
        let format = view.props.get("Format")?.x[0] as u32;
        vk_format(format).or_else(|| astc_vk_format(format))
    }

    fn pool_list_stamp(&self, stage: u32) -> super::PoolListStamp {
        (
            (
                self.texture_epoch,
                self.gpu.as_ref().unwrap().renderer.target_changes,
                self.binding_serial,
                self.presented_frames,
            ),
            self.gpu_writes_now(),
            self.pool_inputs,
            self.stage_binds
                .get(&u64::from(stage))
                .copied()
                .unwrap_or(0),
        )
    }

    /// Resolve a shader's draw-uniform handles against the same constant
    /// and storage buffers that are about to be passed to the GPU.
    fn shader_pool_handles(
        &mut self,
        core: &mut Guest,
        stage: &Arc<Translated>,
        mut constant: impl FnMut(&Guest, u8, usize) -> Option<u32>,
        bound_storage: impl Fn(u64, u64) -> bool,
    ) -> Vec<u64> {
        let Some(reads) = self
            .gpu
            .as_ref()
            .and_then(|backend| backend.handle_reads.get(&(Arc::as_ptr(stage) as usize)))
            .cloned()
        else {
            return Vec::new();
        };
        let mut handles = Vec::new();
        reads.evaluate(
            |input| match input {
                shader::handle_reads::Input::Constant { bank, offset } => {
                    constant(core, bank, usize::from(offset))
                }
                shader::handle_reads::Input::GlobalWord {
                    address,
                    word,
                    words,
                } => {
                    let length = 4 * u64::from(words);
                    if !address.is_multiple_of(length) || !bound_storage(address, length) {
                        return None;
                    }
                    // Validate the full shader load, then synchronize only
                    // the word the handle expression actually reads.
                    let offset = 4 * u64::from(word);
                    let guest = self.gpu_to_guest(address, length).ok()? + offset;
                    self.resolve_queued_buffer_read(core, address + offset, guest, 4)
                        .ok()?;
                    core.memory.read_u32(guest).ok()
                }
            },
            |handle| {
                if handle & echora_shader_abi::TEXTURE_INDEX_MASK
                    < echora_shader_abi::TEXTURE_POOL_CAPACITY
                {
                    handles.push(u64::from(handle));
                }
            },
        );
        handles.sort_unstable();
        handles.dedup();
        handles
    }

    /// Texture pool entries for a program stage that reads the pool: the
    /// textures bound to NVN stage `nvn` (c2 from 0x20, 8 bytes per slot),
    /// plus handles read from the stage's other inputs. Handles that do
    /// not resolve are left out.
    fn pool_entries(
        &mut self,
        core: &mut Guest,
        nvn: u32,
        additional: &[u64],
    ) -> Result<std::sync::Arc<[echora_gpu::pool::PoolEntry]>, String> {
        // (The renderer's counts from pool writes reach the statistics
        // with the page table's at each present.)
        let key = self.pool_list_stamp(nvn);
        if let Some(cached) = self.pool_entry_cache.get(&nvn)
            && cached.stamp == key
            && cached.additional == additional
        {
            let list = cached.entries.clone();
            self.gpu
                .as_mut()
                .unwrap()
                .stat("stat: pool entry lists reused", 1);
            return Ok(list);
        }
        // Keep the same shared array when a GPU write changed the generation
        // but all its bindings still resolve to the same source and sampler.
        self.evict_bindings();
        let key = self.pool_list_stamp(nvn);
        if let Some(mut cached) = self.pool_entry_cache.remove(&nvn) {
            let previous = cached.stamp;
            if cached.additional == additional
                && (previous.0, previous.2, previous.3) == (key.0, key.2, key.3)
                && cached
                    .handles
                    .iter()
                    .zip(cached.entries.iter())
                    .all(|(&handle, entry)| {
                        self.with_memoized_binding(handle, |binding| {
                            entry.matches_binding(&binding.0, &binding.1)
                        }) == Some(true)
                    })
            {
                cached.stamp = key;
                let list = cached.entries.clone();
                self.pool_entry_cache.insert(nvn, cached);
                self.gpu
                    .as_mut()
                    .unwrap()
                    .stat("stat: pool entry lists reused", 1);
                return Ok(list);
            }
            let backend = self.gpu.as_mut().unwrap();
            for (changed, name) in [
                (
                    previous.0 != key.0,
                    "stat: pool entry lists remade: memo stamp",
                ),
                (
                    previous.1 != key.1,
                    "stat: pool entry lists remade: GPU writes",
                ),
                (
                    previous.2 != key.2,
                    "stat: pool entry lists remade: registrations or queue",
                ),
                (
                    previous.3 != key.3,
                    "stat: pool entry lists remade: stage binds",
                ),
            ] {
                if changed {
                    backend.stat(name, 1);
                }
            }
        }
        let stage = u64::from(nvn);
        let mut handles: Vec<u64> = self.render.bound_textures.handles(stage).collect();
        handles.extend_from_slice(additional);
        handles.sort_unstable();
        handles.dedup();
        // Resolved handles are kept for the present as the draw path's
        // bindings are (`binding_memo`, same stamp).
        let stamp = (
            self.binding_epoch,
            0,
            self.binding_serial,
            // Across presents: entries are revalidated per present
            // (`memoized_binding`); texture and target changes evict what
            // they touch (`evict_bindings`).
            0,
        );
        if self.binding_memo.0 != stamp {
            self.binding_memo = (stamp, Default::default());
        }
        let mut entries = Vec::with_capacity(handles.len());
        let mut resolved_handles = Vec::with_capacity(handles.len());
        for handle in handles {
            if let Some((source, sampler_state)) = self.memoized_binding(handle) {
                resolved_handles.push(handle);
                entries.push(echora_gpu::pool::PoolEntry {
                    texture: (handle & u64::from(echora_shader_abi::TEXTURE_INDEX_MASK)) as u32,
                    sampler: (handle >> echora_shader_abi::SAMPLER_INDEX_SHIFT & 0xfff) as u32,
                    source,
                    sampler_state,
                });
                continue;
            }
            let Ok(texture) = self.handle_texture(handle) else {
                self.gpu
                    .as_mut()
                    .unwrap()
                    .stat("stat: texture pool handles unresolved", 1);
                continue;
            };
            let (source, storage) = self.resolve_sampled_texture(core, texture)?;
            let source = self.through_view(source, handle);
            let sampler_state = self.binding_sampler(handle, texture, &source);
            self.binding_memo.1.insert(
                handle,
                (
                    source.clone(),
                    sampler_state,
                    storage,
                    self.gpu_writes_now(),
                    self.presented_frames,
                    texture,
                ),
            );
            resolved_handles.push(handle);
            entries.push(echora_gpu::pool::PoolEntry {
                texture: (handle & u64::from(echora_shader_abi::TEXTURE_INDEX_MASK)) as u32,
                sampler: (handle >> echora_shader_abi::SAMPLER_INDEX_SHIFT & 0xfff) as u32,
                source,
                sampler_state,
            });
        }
        let list: std::sync::Arc<[echora_gpu::pool::PoolEntry]> = entries.into();
        // Resolving may have moved the memo stamp: kept under the key now.
        self.pool_entry_cache.insert(
            nvn,
            super::CachedPoolEntries {
                stamp: self.pool_list_stamp(nvn),
                entries: list.clone(),
                handles: resolved_handles,
                additional: additional.to_vec(),
            },
        );
        Ok(list)
    }

    /// The sampler a texture binding samples with: the handle's sampler,
    /// its view's swizzle and format and, for uploads, its view's levels
    /// (a view of those levels: its level 0 is the base, as on the
    /// console; the guest sampler itself stays as the game set it, one
    /// Vulkan sampler for every texture paired with it).
    fn binding_sampler(&self, handle: u64, texture: u64, source: &TextureSource) -> Sampler {
        let mut sampler = self.handle_sampler(handle);
        sampler.components = self.handle_components(handle);
        sampler.format = self.handle_view_format(handle);
        // A separate texture can name the same render storage in the other
        // color space even when the descriptor has no explicit view format.
        if sampler.format.is_none()
            && let TextureSource::Target(key) | TextureSource::TargetView { key, .. } = source
            && *key != texture
            && let Some(format) = self
                .objects
                .get(&texture)
                .and_then(|object| object.builder.get("Format"))
                .and_then(|value| vk_format(value.x[0] as u32))
            && let Some(target) = self
                .gpu
                .as_ref()
                .and_then(|backend| backend.renderer.target_info(*key))
            && echora_gpu::renderer::srgb_twin(target.format) == Some(format)
        {
            sampler.format = Some(format);
        }
        // A decoded ASTC upload is an RGBA image. Its alternate ASTC view
        // still selects the other color space through the RGBA format.
        if let TextureSource::Upload { key, format, .. } = source
            && matches!(
                *format,
                vk::Format::R8G8B8A8_UNORM | vk::Format::R8G8B8A8_SRGB
            )
            && let Some(view) = sampler.format
            && let Some(encoded) = self
                .objects
                .get(key)
                .and_then(|object| object.builder.get("Format"))
                .and_then(|value| astc_vk_format(value.x[0] as u32))
            && echora_gpu::renderer::srgb_twin(encoded) == Some(view)
        {
            sampler.format = echora_gpu::renderer::srgb_twin(*format);
        }
        if sampler.format.is_some() && super::debug_var_os("ECHORA_GPU_FLOW").is_some() {
            let own = self
                .objects
                .get(&texture)
                .and_then(|object| object.builder.get("Format"))
                .and_then(|value| vk_format(value.x[0] as u32));
            if own != sampler.format {
                eprintln!(
                    "flow sampled view format {:?} of {own:?} on {texture:x}",
                    sampler.format
                );
            }
        }
        // ECHORA_GPU_LOD0=all|uploads|targets (debugging): those textures
        // sample level 0 only.
        if let Ok(which) = super::debug_var("ECHORA_GPU_LOD0") {
            let upload = matches!(source, TextureSource::Upload { .. });
            if which == "all" || (which == "uploads") == upload {
                sampler.base_level_only = true;
            }
        }
        if matches!(source, TextureSource::Upload { .. }) {
            sampler.view_levels = self
                .handle_view_levels(handle)
                .filter(|&range| range != (0, 0));
            sampler.view_layers = self
                .handle_view_layers(handle)
                .filter(|&range| range != (0, 0));
        }
        sampler
    }

    /// The component mapping of a handle's texture view swizzle: NVN
    /// swizzle values 0 ZERO, 1 ONE, 2..5 R, G, B, A (2, 3, 4, 5 is the
    /// identity; inferred from the application's views, which use that and
    /// 4, 3, 2, 5).
    fn handle_components(&self, handle: u64) -> vk::ComponentMapping {
        let Some(swizzle) = self.handle_swizzle(handle) else {
            return vk::ComponentMapping::default();
        };
        let component = |value: u64, identity: vk::ComponentSwizzle| match value {
            0 => vk::ComponentSwizzle::ZERO,
            1 => vk::ComponentSwizzle::ONE,
            2 => vk::ComponentSwizzle::R,
            3 => vk::ComponentSwizzle::G,
            4 => vk::ComponentSwizzle::B,
            5 => vk::ComponentSwizzle::A,
            _ => identity,
        };
        let mapping = vk::ComponentMapping {
            r: component(swizzle[0], vk::ComponentSwizzle::R),
            g: component(swizzle[1], vk::ComponentSwizzle::G),
            b: component(swizzle[2], vk::ComponentSwizzle::B),
            a: component(swizzle[3], vk::ComponentSwizzle::A),
        };
        // The identity as IDENTITY, so unswizzled views are shared.
        if [mapping.r, mapping.g, mapping.b, mapping.a]
            == [
                vk::ComponentSwizzle::R,
                vk::ComponentSwizzle::G,
                vk::ComponentSwizzle::B,
                vk::ComponentSwizzle::A,
            ]
        {
            vk::ComponentMapping::default()
        } else {
            mapping
        }
    }

    /// The swizzle (R, G, B, A sources) the texture view registered with
    /// a handle sets, if any.
    fn handle_swizzle(&self, handle: u64) -> Option<[u64; 4]> {
        let pool = self.render.state.get("SetTexturePool")?.x[1];
        let view = self.render_texture_view(pool, (handle & 0xf_ffff) as u32)?;
        let swizzle = view.props.get("Swizzle")?;
        Some([swizzle.x[0], swizzle.x[1], swizzle.x[2], swizzle.x[3]])
    }

    /// A render target sampled through a handle whose view selects array
    /// layers (or cube faces) samples only those, and a view's mip levels
    /// only those (`mip_targets`).
    fn through_view(&self, source: TextureSource, handle: u64) -> TextureSource {
        let (key, source_level, source_levels) = match source {
            TextureSource::Target(key) => (key, 0, 0),
            TextureSource::TargetView {
                key,
                base_level,
                level_count,
                ..
            } => (key, base_level, level_count),
            TextureSource::Upload { .. } => return source,
        };
        let (mut base_level, mut level_count) = if super::mip_targets() {
            self.handle_view_levels(handle).unwrap_or((0, 0))
        } else {
            (0, 0)
        };
        let (base_layer, layer_count) = self.handle_view_layers(handle).unwrap_or((0, 0));
        if (base_level, level_count, base_layer, layer_count) == (0, 0, 0, 0) {
            return source;
        }
        // The view's levels are relative to the texture object, including
        // when it names just part of the render target's mip chain.
        if source_levels != 0 {
            base_level = base_level.min(source_levels - 1);
            let remaining = source_levels - base_level;
            level_count = if level_count == 0 {
                remaining
            } else {
                level_count.min(remaining)
            };
        }
        TextureSource::TargetView {
            key,
            base_level: source_level + base_level,
            level_count,
            base_layer,
            layer_count,
        }
    }

    /// The array layers (base, count; 0: the rest) the texture view
    /// registered with a handle selects, if any.
    fn handle_view_layers(&self, handle: u64) -> Option<(u32, u32)> {
        let pool = self.render.state.get("SetTexturePool")?.x[1];
        let view = self.render_texture_view(pool, (handle & 0xf_ffff) as u32)?;
        let layers = view.props.get("Layers")?;
        Some((layers.x[0] as u32, layers.x[1] as u32))
    }

    /// Mip levels a texture object was built with, for its render target
    /// (1 with ECHORA_GPU_NO_MIPS, where every level renders into level 0).
    fn texture_levels(&self, texture: u64) -> u32 {
        if !super::mip_targets() {
            return 1;
        }
        self.objects
            .get(&texture)
            .and_then(|object| object.builder.get("Levels"))
            .map_or(1, |value| (value.x[0] as u32).clamp(1, 16))
    }

    /// Guest address of a texture object's storage. A virtual pool has no
    /// storage of its own: its offset resolves through the pool's GPU
    /// range to the physical pool mapped there (MapVirtual).
    /// A texture's storage and NVN format: what tells two textures over one
    /// structure (or one memory) apart.
    pub(super) fn texture_identity(&self, texture: u64) -> Option<(u64, u64)> {
        let format = self.objects.get(&texture)?.builder.get("Format")?.x[0];
        Some((self.texture_storage(texture)?, format))
    }

    pub(super) fn texture_storage(&self, texture: u64) -> Option<u64> {
        let mut memo = self.storage_memo.borrow_mut();
        if let Some(&address) = memo.get(&texture) {
            return address;
        }
        let address = self.texture_storage_uncached(texture);
        memo.insert(texture, address);
        address
    }

    fn texture_storage_uncached(&self, texture: u64) -> Option<u64> {
        let texture = self.objects.get(&texture).filter(|o| o.kind == "Texture")?;
        let storage = texture.builder.get("Storage")?;
        let pool = self
            .objects
            .get(&storage.x[0])
            .filter(|o| o.kind == "MemoryPool")?;
        let base = pool.builder.get("Storage")?.x[0];
        if base != 0 {
            return Some(base + storage.x[1]);
        }
        let gpu = pool.state.get("GpuAddress")?.x[0] + storage.x[1];
        self.pool_for(gpu).map(|(guest, _)| guest)
    }

    pub(super) fn texture_backing(
        &self,
        storage: &super::Args,
        size: u64,
    ) -> Result<TextureBacking, String> {
        let pool = self.object(storage.x[0], "MemoryPool")?;
        let [base, pool_size, ..] = pool
            .builder
            .get("Storage")
            .ok_or("texture pool has no storage")?
            .x;
        let start = storage.x[1];
        let end = start
            .checked_add(size)
            .ok_or("texture storage size overflows")?;
        if base != 0 {
            base.checked_add(end)
                .ok_or("texture storage address overflows")?;
            return Ok(TextureBacking::linear(base + start, size));
        }
        if end > pool_size {
            return Err("texture storage exceeds its virtual pool".into());
        }
        let mut pieces: Vec<(u64, std::ops::Range<u64>)> = Vec::new();
        // Fixed-width keys give increasing virtual offsets.
        let bounds = (
            std::ops::Bound::Included(super::VIRTUAL_MAPPING),
            std::ops::Bound::Unbounded,
        );
        for (name, mapping) in pool.state.range::<str, _>(bounds) {
            let Some(hex) = name.strip_prefix(super::VIRTUAL_MAPPING) else {
                break;
            };
            let Ok(offset) = u64::from_str_radix(hex, 16) else {
                continue;
            };
            if offset >= end {
                break;
            }
            let Some(range) = self.virtual_mapping_storage(mapping) else {
                continue;
            };
            let left = start.max(offset);
            let right = end.min(offset + (range.end - range.start));
            if left >= right {
                continue;
            }
            let physical = range.start + (left - offset)..range.start + (right - offset);
            if let Some((last_offset, last)) = pieces.last_mut()
                && *last_offset + (last.end - last.start) == left - start
                && last.end == physical.start
            {
                last.end = physical.end;
            } else {
                pieces.push((left - start, physical));
            }
        }
        if let [(0, range)] = pieces.as_slice()
            && range.end - range.start == size
        {
            return Ok(TextureBacking::linear(range.start, size));
        }
        if pieces.is_empty() {
            return Err("texture storage is in no mapped memory".into());
        }
        Ok(TextureBacking::Virtual { size, pieces })
    }

    /// A render target whose texture object shares `texture`'s storage and
    /// size, if any.
    fn target_sharing_storage(&self, texture: u64) -> Option<u64> {
        let base = self.texture_storage(texture)?;
        let extent = self.texture_extent(texture).ok()?;
        let backend = self.gpu.as_ref()?;
        let renderer = &backend.renderer;
        let matches = |key: &u64| {
            self.texture_storage(*key) == Some(base)
                && renderer
                    .target_info(*key)
                    .is_some_and(|target| (target.width, target.height) == extent)
        };
        if self.target_index_current() {
            return backend
                .target_index
                .storage
                .get(&base)?
                .iter()
                .copied()
                .find(|key| matches(key));
        }
        renderer
            .target_keys_in_order()
            .iter()
            .copied()
            .find(matches)
    }

    /// Whether `target_index` describes the current targets and storage.
    fn target_index_current(&self) -> bool {
        self.gpu.as_ref().is_some_and(|backend| {
            backend.target_index.stamp == Some(backend.renderer.target_changes)
                && backend.target_index.changed.is_empty()
        })
    }

    /// A texture changed, or a pool changed where textures map.
    pub(super) fn invalidate_target_index(&mut self, texture: Option<u64>) {
        if let Some(backend) = self.gpu.as_mut() {
            if let Some(key) = texture {
                if backend.renderer.target_info(key).is_some() {
                    backend.target_index.changed.insert(key);
                }
            } else {
                backend.target_index.stamp = None;
            }
        }
    }

    /// Update changed targets. A pool change can move many textures at
    /// once, so it still rebuilds the entire index.
    fn refresh_target_index(&mut self) {
        if self.target_index_current() {
            return;
        }
        let Some(backend) = self.gpu.as_mut() else {
            return;
        };
        let started = std::time::Instant::now();
        let mut index = std::mem::take(&mut backend.target_index);
        let backend = self.gpu.as_ref().unwrap();
        let renderer = &backend.renderer;
        let generation = renderer.target_changes;
        let add_target = |index: &mut TargetIndex, key| {
            if let Some(target) = renderer.target_info(key)
                && let Some(base) = self.texture_storage(key)
            {
                let levels = (target.levels > 1)
                    .then(|| self.target_level_offsets(key, target))
                    .into_iter()
                    .flatten();
                index.insert(key, base, levels);
            }
        };
        let rebuild = index.stamp.is_none();
        if rebuild {
            index.storage.values_mut().for_each(Vec::clear);
            index.mips.values_mut().for_each(Vec::clear);
            index.by_target.clear();
            index.changed.clear();
            for &key in renderer.target_keys_in_order() {
                add_target(&mut index, key);
            }
            index.storage.retain(|_, targets| !targets.is_empty());
            index.mips.retain(|_, mips| !mips.is_empty());
        } else {
            if index.stamp != Some(generation) {
                index
                    .changed
                    .extend(renderer.changed_targets().iter().copied());
            }
            let mut changed = std::mem::take(&mut index.changed);
            for key in changed.drain() {
                index.remove(key);
                if renderer.target_keys_in_order().binary_search(&key).is_ok() {
                    add_target(&mut index, key);
                }
            }
            index.changed = changed;
        }
        index.stamp = Some(generation);
        let backend = self.gpu.as_mut().unwrap();
        backend.target_index = index;
        let (count, time) = if rebuild {
            (
                "stat: target index rebuilds",
                "stat: target index rebuild us",
            )
        } else {
            (
                "stat: target index incremental updates",
                "stat: target index incremental update us",
            )
        };
        backend.stat(count, 1);
        backend.stat(time, started.elapsed().as_micros() as u64);
    }

    /// The render target (and its mip level) a texture object's storage
    /// is, as a GPU texture cache resolves addresses: the target itself,
    /// another target at the same address and size, or a target whose mip
    /// level `level` starts at this storage with this size (a separate
    /// texture object per level, as bloom chains use; levels above 0 as
    /// `mip_targets` allows).
    fn resolve_target(&self, texture: u64) -> Option<(u64, u32)> {
        self.resolve_target_with_levels(texture, self.texture_levels(texture))
    }

    fn resolve_target_with_levels(&self, texture: u64, levels: u32) -> Option<(u64, u32)> {
        let renderer = &self.gpu.as_ref()?.renderer;
        // Its own format (or its sRGB twin): a texture object
        // reinterpreting the memory in another format (RGB9E5 over the HDR
        // scene's R11G11B10) is not the same target.
        let (extent, format) = match self.target_shape(texture) {
            Ok((width, height, format)) => (Some((width, height)), Some(format)),
            Err(_) => (self.texture_extent(texture).ok(), None),
        };
        let same_format = |target_format: vk::Format| {
            format.is_some_and(|format| {
                target_format == format
                    || echora_gpu::renderer::srgb_twin(target_format) == Some(format)
            })
        };
        // The object's own target, while the object still describes it:
        // applications rotate texture objects through a ring and initialize
        // one again over other storage or in another format (the target
        // the object last named is then another texture).
        if renderer.target_info(texture).is_some_and(|target| {
            (format.is_none() || same_format(target.format))
                && extent.is_none_or(|extent| (target.width, target.height) == extent)
                && target.levels >= levels
        }) {
            return Some((texture, 0));
        }
        let mips = super::mip_targets();
        if !mips {
            return self.target_sharing_storage(texture).map(|key| (key, 0));
        }
        let base = self.texture_storage(texture)?;
        let extent = extent?;
        // From the index when it is current: the targets at this storage,
        // then the target levels that start there.
        if self.target_index_current() {
            let backend = self.gpu.as_ref()?;
            for &key in backend
                .target_index
                .storage
                .get(&base)
                .into_iter()
                .flatten()
            {
                if renderer.target_info(key).is_some_and(|target| {
                    (target.width, target.height) == extent
                        && same_format(target.format)
                        && target.levels >= levels
                }) {
                    return Some((key, 0));
                }
            }
            return backend
                .target_index
                .mips
                .get(&base)?
                .iter()
                .find(|(key, level, level_extent)| {
                    *level_extent == extent
                        && renderer.target_info(*key).is_some_and(|target| {
                            same_format(target.format) && target.levels - level >= levels
                        })
                })
                .map(|&(key, level, _)| (key, level));
        }
        let mut mip_match = None;
        for &key in renderer.target_keys_in_order() {
            let Some(target) = renderer
                .target_info(key)
                .filter(|target| same_format(target.format))
            else {
                continue;
            };
            let Some(target_base) = self.texture_storage(key) else {
                continue;
            };
            if target_base == base
                && (target.width, target.height) == extent
                && target.levels >= levels
            {
                return Some((key, 0));
            }
            if mip_match.is_some() || target.levels == 1 || base <= target_base {
                continue;
            }
            for (level, offset, level_extent) in self.target_level_offsets(key, target) {
                if target_base + offset == base
                    && level_extent == extent
                    && target.levels - level >= levels
                {
                    mip_match = Some((key, level));
                    break;
                }
            }
        }
        mip_match
    }

    /// A renderable view of a compatible image covering the texture's
    /// mip chain. When extending a same-layout image, keep the levels
    /// already written through its shorter alias.
    fn resolve_color_target(&mut self, texture: u64, level: u32) -> Result<(u64, u32), String> {
        let levels = self.texture_levels(texture);
        if let Some((key, base)) = self.resolve_target_with_levels(texture, levels) {
            return Ok((key, base + level));
        }
        if levels > 1
            && self
                .gpu
                .as_ref()
                .unwrap()
                .renderer
                .target_info(texture)
                .is_none()
            && let Some((alias, 0)) = self.resolve_target_with_levels(texture, 1)
            && self.flat_alias_layout_matches(texture, alias)
        {
            self.gpu
                .as_mut()
                .unwrap()
                .renderer
                .move_target(alias, texture)?;
            self.ensure_render_target(texture)?;
        }
        Ok((texture, level))
    }

    /// The layouts that the renderer can extend in place: a single flat
    /// surface with the same format, storage, stride and level-0 tiling.
    fn flat_alias_layout_matches(&self, texture: u64, alias: u64) -> bool {
        let Some(identity) = self.texture_identity(texture) else {
            return false;
        };
        let Some(info) = crate::nvn_format::info(identity.1 as u32) else {
            return false;
        };
        let shape = ImageShape::D2 { layers: 1 };
        let renderer = &self.gpu.as_ref().unwrap().renderer;
        let Some(target) = renderer.target_info(alias) else {
            return false;
        };
        if target.shape != shape
            || self.render_shape(texture) != shape
            || self.texture_identity(alias) != Some(identity)
            || Some(target.format) != vk_format(identity.1 as u32)
        {
            return false;
        }
        let layout = |key| -> Option<_> {
            let props = &self.objects.get(&key)?.builder;
            let value = |name| props.get(name).map_or(0, |args| args.x[0]);
            let dims = [target.width, target.height, 1];
            let layers =
                super::texture_layout_layers(props, info, dims, 1, self.texture_levels(key));
            let block = crate::nvn_format::base_block_shape(info, target.height, 1, layers);
            Some((value("Stride"), value("Samples").max(1), block))
        };
        layout(texture)
            .zip(layout(alias))
            .is_some_and(|(a, b)| a == b)
    }

    /// Each level above 0 of render target `key`: its offset from the
    /// target's storage and its extent (none when the texture object is
    /// gone or its format unknown).
    fn target_level_offsets(
        &self,
        key: u64,
        target: echora_gpu::renderer::TargetInfo,
    ) -> impl Iterator<Item = (u32, u64, (u32, u32))> + use<> {
        self.objects
            .get(&key)
            .and_then(|object| {
                let get = |name: &str| object.builder.get(name).map(|value| value.x[0] as u32);
                let info = get("Format").and_then(crate::nvn_format::info)?;
                let depth = if get("Target") == Some(2) {
                    get("Depth").unwrap_or(1).max(1)
                } else {
                    1
                };
                let (width, height) = (target.width, target.height);
                let dimensions = [width, height, depth];
                let layout = super::texture_layout_layers(
                    &object.builder,
                    info,
                    dimensions,
                    target.shape.layers(),
                    target.levels,
                );
                Some(
                    crate::nvn_format::block_linear_mips(info, dimensions, layout, target.levels)
                        .enumerate()
                        .skip(1)
                        .map(move |(level, mip)| {
                            let extent = ((width >> level).max(1), (height >> level).max(1));
                            (level as u32, mip.offset, extent)
                        }),
                )
            })
            .into_iter()
            .flatten()
    }

    /// The alias target (and its size) a draw renders the blocks of
    /// block-compressed texture `texture` into through a color view of
    /// format `view`, when each of the view's texels is one block (an
    /// RGBA16UI view of BC4: a pixel shader compressing, one pixel per 4x4
    /// block). The texture's own image takes them before it is sampled
    /// (`texture_source`), as on the console both name one memory.
    fn block_alias_target(
        &mut self,
        texture: u64,
        view: vk::Format,
    ) -> Result<Option<(u64, (u32, u32))>, String> {
        let Ok((width, height, format)) = self.target_shape(texture) else {
            return Ok(None);
        };
        let Some(bytes) = echora_gpu::renderer::bc_block_bytes(format) else {
            return Ok(None);
        };
        let texel = match echora_gpu::renderer::uint_alias(view) {
            Some(vk::Format::R32G32_UINT) => 8,
            Some(vk::Format::R32G32B32A32_UINT) => 16,
            _ => 0,
        };
        if texel != bytes {
            return Ok(None);
        }
        self.ensure_render_target(texture)?;
        let alias = 1 << 62 | texture;
        let shape = self.render_shape(texture);
        let levels = self.texture_levels(texture);
        let size = (width.div_ceil(4), height.div_ceil(4));
        self.gpu
            .as_mut()
            .unwrap()
            .renderer
            .shaped_target(alias, size.0, size.1, shape, view, levels)?;
        self.block_aliases.insert(texture, alias);
        Ok(Some((alias, size)))
    }

    fn resolve_sampled_texture(
        &mut self,
        core: &mut Guest,
        texture: u64,
    ) -> Result<(TextureSource, Option<TextureBacking>), String> {
        let storage = self.texture_object_backing(texture);
        if let Some(backing) = &storage {
            for (_, range) in backing.pieces() {
                self.resolve_storage(core, Some(range.clone()))?;
            }
        }
        Ok((self.texture_source(core, texture)?, storage))
    }

    fn texture_object_backing(&self, texture: u64) -> Option<TextureBacking> {
        let object = self.objects.get(&texture)?;
        let size = self.texture_storage_size(texture)?;
        self.texture_backing(object.builder.get("Storage")?, size)
            .ok()
    }

    pub(super) fn resolve_texture_copy_read(
        &mut self,
        core: &mut Guest,
        texture: u64,
    ) -> Result<(), String> {
        if let Some(backing) = self.texture_object_backing(texture) {
            for (_, range) in backing.pieces() {
                self.resolve_gpu_written(core, range.clone())?;
            }
        }
        Ok(())
    }

    fn texture_source(&mut self, core: &Guest, texture: u64) -> Result<TextureSource, String> {
        let source = self.texture_source_resolved(core, texture)?;
        // Blocks drawn since it was last sampled come in first.
        if let TextureSource::Target(key) | TextureSource::TargetView { key, .. } = &source {
            let backend = self.gpu.as_mut().unwrap();
            if let Some(age) = backend.retired_since.get_mut(key) {
                *age = self.presented_frames;
            }
            backend.renderer.restore_archived_target(*key)?;
            if let Some(alias) = self.block_aliases.remove(key) {
                backend.renderer.copy_blocks(alias, *key)?;
                backend.stat("stat: compressed blocks drawn copied in", 1);
            }
        }
        Ok(source)
    }

    fn texture_source_resolved(
        &mut self,
        core: &Guest,
        texture: u64,
    ) -> Result<TextureSource, String> {
        self.refresh_target_index();
        // The render target this texture is (its format and size, at its
        // address), or a mip level of one. Engines reuse transient memory
        // for textures of other formats within a frame (an RGB9E5 image
        // over the HDR scene's R11G11B10 memory once the scene is used):
        // the target in the texture's own format is the one it reads.
        let levels = self.texture_levels(texture);
        if let Some((key, level)) = self.resolve_target_with_levels(texture, levels) {
            if level > 0
                || (key != texture
                    && self
                        .gpu
                        .as_ref()
                        .unwrap()
                        .renderer
                        .target_info(key)
                        .is_some_and(|target| target.levels > levels))
            {
                return Ok(TextureSource::TargetView {
                    key,
                    base_level: level,
                    level_count: levels,
                    base_layer: 0,
                    layer_count: 0,
                });
            }
            return Ok(TextureSource::Target(key));
        }
        // Otherwise a target over the same memory in another format (a
        // depth target sampled as R32F, say).
        if self
            .gpu
            .as_ref()
            .unwrap()
            .renderer
            .target_info(texture)
            .is_some()
        {
            return Ok(TextureSource::Target(texture));
        }
        // Another texture object over the same storage may be the one the
        // GPU rendered into (a separate object for sampling is common):
        // sample that render target.
        if let Some(target) = self.target_sharing_storage(texture) {
            return Ok(TextureSource::Target(target));
        }
        self.guest_texture(core, texture)
    }

    /// Start new render targets with their guest memory's contents, so
    /// later passes can read subresources the renderer has yet to write.
    pub(super) fn initialize_new_targets(&mut self, core: &Guest) {
        let fresh: Vec<u64> = self.gpu.as_mut().unwrap().renderer.take_fresh_targets();
        for key in fresh {
            let Ok(source) = self.guest_texture(core, key) else {
                continue;
            };
            let backend = self.gpu.as_mut().unwrap();
            if backend.renderer.initialize_target(key, &source).is_ok() {
                backend.stat("stat: targets initialized from guest memory", 1);
            }
        }
    }

    /// A cached upload of 64 KiB or more whose image the renderer holds
    /// keeps no host copy of its texels: the renderer reads them back from
    /// the image when it needs them again (an alias image, a texel
    /// buffer). The forest's 436 uploads kept 2.7 GB on the host.
    fn release_upload_texels(&mut self, texture: u64) {
        let backend = self.gpu.as_mut().unwrap();
        let Some((
            _,
            TextureSource::Upload {
                key,
                generation,
                texels,
                ..
            },
            _,
        )) = backend.textures.get_mut(&texture)
        else {
            return;
        };
        if texels.len() < 1 << 16 || !backend.renderer.has_upload(*key, *generation) {
            return;
        }
        *texels = Arc::new(Vec::new());
        backend.stat("stat: upload texels released", 1);
    }

    /// The upload of a texture's guest memory, cached while it holds.
    fn guest_texture(&mut self, core: &Guest, texture: u64) -> Result<TextureSource, String> {
        // Flushed pieces of a cached upload: unchanged bytes keep it.
        if let Some(ranges) = self.gpu.as_mut().unwrap().dirty.remove(&texture) {
            // The CPU wrote the storage: a copy's rows no longer stand for it.
            self.gpu.as_mut().unwrap().copied_levels.remove(&texture);
            let backend = self.gpu.as_ref().unwrap();
            let unchanged = match (
                backend.textures.get(&texture),
                backend.storage_hashes.get(&texture),
            ) {
                (Some((_, _, backing)), Some(hashes)) => {
                    // A flush can change bytes missed by the periodic texture
                    // sample. Check every chunk the flush touches.
                    let mut chunks: Vec<_> = ranges
                        .into_iter()
                        .map(|range| range.start / STORAGE_CHUNK..range.end.div_ceil(STORAGE_CHUNK))
                        .collect();
                    merge_ranges(&mut chunks);
                    let mut bytes = vec![0; STORAGE_CHUNK as usize];
                    chunks.into_iter().flatten().all(|chunk| {
                        let start = chunk * STORAGE_CHUNK;
                        let length = STORAGE_CHUNK.min(backing.size() - start);
                        let bytes = &mut bytes[..length as usize];
                        backing.read_into(core, start, bytes);
                        hashes.get(chunk as usize) == Some(&hash_bytes(bytes))
                    })
                }
                _ => false,
            };
            let backend = self.gpu.as_mut().unwrap();
            if unchanged {
                backend.stat("stat: texture uploads kept after flush", 1);
            } else {
                backend.remove_texture_upload(texture);
            }
        }
        let frame = self.presented_frames;
        let backend = self.gpu.as_ref().unwrap();
        if backend.texture_checked.get(&texture) == Some(&frame) {
            if backend.textures.contains_key(&texture) {
                self.release_upload_texels(texture);
                let backend = self.gpu.as_ref().unwrap();
                return Ok(backend.textures[&texture].1.clone());
            }
        }
        let object = self.object(texture, "Texture")?;
        let get = |key: &str| object.builder.get(key).map(|value| value.x[0] as u32);
        let format = get("Format").unwrap_or(0);
        let info =
            crate::nvn_format::info(format).ok_or_else(|| format!("texture format {format}"))?;
        let width = get("Width").unwrap_or(1);
        let target = get("Target").unwrap_or(1);
        let (height, depth_or_layers) = texture_height_and_depth(
            target,
            get("Height").unwrap_or(1),
            get("Depth").unwrap_or(1),
        );
        let astc = info.name.starts_with("ASTC");
        let mut vk_format = match vk_format(format) {
            Some(vk_format) => vk_format,
            // ASTC has no host format on most desktop GPUs: decoded below.
            None if info.name == "ASTC" => vk::Format::R8G8B8A8_UNORM,
            None if info.name == "ASTC_SRGB" => vk::Format::R8G8B8A8_SRGB,
            None => {
                return Err(format!(
                    "texture format {} has no host equivalent",
                    info.name
                ));
            }
        };
        // Wider than the host's limit (buffer textures a million texels
        // long): out of specification, but the driver here samples them;
        // noted.
        let limit = self.gpu.as_ref().unwrap().max_extent;
        let beyond_limit = width > limit || height > limit;
        // Combined depth and stencil require two separate Vulkan copies.
        if echora_gpu::renderer::has_stencil(vk_format) {
            return Err(format!(
                "{} texture needs separate upload planes",
                info.name
            ));
        }
        // Vulkan buffer offsets for a depth copy are four-byte aligned.
        let upload_alignment = if echora_gpu::renderer::is_depth(vk_format) {
            4
        } else {
            1
        };
        let storage = object
            .builder
            .get("Storage")
            .ok_or("texture has no storage")?;
        let pool = self.object(storage.x[0], "MemoryPool")?;
        // NVN memory pool flags (same bits as NVN1): CPU_UNCACHED 2,
        // CPU_CACHED 4; the builder default 0x21 gives the CPU no access.
        let pool_flags = pool.builder.get("Flags").map_or(0x21, |value| value.x[0]);
        let uncached = pool_flags & 2 != 0 && pool_flags & 4 == 0;
        let stride = texture_row_pitch(&object.builder, info, width);
        let block = info.bytes_per_block as usize;
        // Untiling and hashing whole textures on every draw costs more than
        // the draw: a sample of 256 evenly spaced 64-byte chunks of the raw
        // storage (and the shape) decides whether the last upload still
        // holds. CPU writes the application flushes (memory pool flushes,
        // WriteTexels) drop overlapping entries (`gpu_invalidate`), so
        // small updates such as new glyphs in a font atlas are not missed.
        // NVN targets (inferred, GL order): 0 1D, 1 2D, 2 3D, 3 1D array,
        // 4 2D array, 7 rectangle, 8 cube, 9 cube array, 10 buffer (read as
        // a 1D image here); 2D arrays and cubes keep their layers in Depth.
        let shape = match target {
            0 | 10 => ImageShape::D1 { layers: 1 },
            3 => ImageShape::D1 {
                layers: depth_or_layers,
            },
            2 => ImageShape::D3 {
                depth: depth_or_layers,
            },
            4 => ImageShape::D2 {
                layers: depth_or_layers,
            },
            8 | 9 => ImageShape::Cube {
                layers: depth_or_layers.max(6) / 6 * 6,
            },
            _ => ImageShape::D2 { layers: 1 },
        };
        let (depth, layers) = (shape.depth(), shape.layers());
        let levels_set = get("Levels").unwrap_or(1);
        let layout = super::texture_layout_layers(
            &object.builder,
            info,
            [width, height, depth],
            layers,
            levels_set,
        );
        let storage_size = if stride != 0 {
            stride * u64::from(height.div_ceil(info.block_height))
        } else {
            crate::nvn_format::block_linear_size(info, width, height, depth, layout, levels_set).0
        };
        let backing = self.texture_backing(storage, storage_size)?;
        let base = backing.pieces().first().map_or(0, |(_, range)| range.start);
        let check_started = std::time::Instant::now();
        let backend = self.gpu.as_ref().unwrap();
        let mapping_changed = backend
            .textures
            .get(&texture)
            .is_some_and(|(_, _, previous)| *previous != backing);
        let shadow = backend.shadows.get(&texture);
        let shadowed = shadow.is_some();
        // Uncached memory can change in any frame without a flush.
        let shadow_changed = shadow.is_some_and(|shadow| !backing.equals(core, shadow));
        // An unchanged copy proves the cached upload current: its sample
        // stands for the storage.
        let kept_sample = (shadowed && !shadow_changed)
            .then(|| backend.textures.get(&texture).map(|entry| entry.0))
            .flatten();
        let sample = kept_sample
            .unwrap_or_else(|| backing.sample(core, format, [width, height, depth_or_layers]));
        let check_micros = check_started.elapsed().as_micros() as u64;
        // ECHORA_GPU_VERIFY_UPLOADS=1 (debugging): a cached upload whose
        // sample still matches is also checked against its whole storage.
        let stale = super::debug_var_os("ECHORA_GPU_VERIFY_UPLOADS").is_some() && {
            let backend = self.gpu.as_ref().unwrap();
            let sampled = backend.textures.get(&texture).map(|entry| entry.0);
            sampled == Some(sample)
                && backend.storage_hashes.get(&texture).is_some_and(|hashes| {
                    let raw = backing.read(core);
                    let now: Vec<u64> =
                        raw.chunks(STORAGE_CHUNK as usize).map(hash_bytes).collect();
                    let changed = now.iter().zip(hashes).filter(|(a, b)| a != b).count();
                    changed > 0
                })
        };
        if stale {
            eprintln!(
                "stale upload missed by its sample: {} {width}x{height} at {base:x} (present {frame})",
                info.name
            );
        }
        if let Some((sampled, _, _)) = self.gpu.as_ref().unwrap().textures.get(&texture) {
            if *sampled == sample && !stale && !shadow_changed && !mapping_changed {
                self.release_upload_texels(texture);
                let backend = self.gpu.as_mut().unwrap();
                let source = backend.textures[&texture].1.clone();
                backend.texture_checked.insert(texture, frame);
                backend.stat("stat: texture checks", 1);
                backend.stat("stat: texture check microseconds", check_micros);
                if shadowed {
                    backend.stat("stat: texture checks against a shadow", 1);
                }
                return Ok(source);
            }
        }
        // Every mip level the texture has (linear textures have one), each
        // untiled to packed rows of blocks and, for ASTC, decoded to RGBA.
        let level_count = if stride != 0 {
            1
        } else {
            let levels = get("Levels").unwrap_or(1).clamp(1, 16);
            levels.min(32 - width.max(height).max(depth).leading_zeros())
        };
        let compressed_astc = astc_vk_format(format).filter(|&format| {
            self.gpu.as_ref().unwrap().renderer.supports_astc_upload(
                format,
                shape,
                width,
                height,
                level_count,
            )
        });
        let decode_astc = astc && compressed_astc.is_none();
        if let Some(format) = compressed_astc {
            vk_format = format;
        }
        // The rows of the last whole-level copy into the texture, when the
        // storage still samples as the copy left it: they are level 0
        // untiled, so neither the storage read nor untiling is needed.
        let copied = (level_count == 1
            && layers == 1
            && depth == 1
            && !astc
            && !uncached
            && !mapping_changed
            && matches!(backing, TextureBacking::Linear(_)))
        .then(|| {
            let backend = self.gpu.as_ref().unwrap();
            let (copied_sample, rows) = backend.copied_levels.get(&texture)?;
            (*copied_sample == sample).then(|| rows.clone())
        })
        .flatten();
        // The whole storage, read once: untiling reads it block by block.
        let raw = if copied.is_some() {
            self.gpu
                .as_mut()
                .unwrap()
                .stat("stat: texture uploads from copies", 1);
            Vec::new()
        } else {
            backing.read(core)
        };
        if matches!(shape, ImageShape::D3 { .. })
            && super::debug_var_os("ECHORA_GPU_FLOW").is_some()
        {
            let nonzero = raw.iter().filter(|byte| **byte != 0).count();
            let sum: u64 = raw.iter().map(|byte| u64::from(*byte)).sum();
            eprintln!(
                "present {frame}: 3D upload {texture:x} at {base:x} {width}x{height}x{depth} {}: {nonzero} of {} bytes nonzero, mean {:.1}",
                info.name,
                raw.len(),
                sum as f64 / raw.len().max(1) as f64
            );
        }
        // ECHORA_GPU_RAW_DUMP=<storage hex>[,...] with ECHORA_DUMP_DIR
        // (debugging): those textures' storage bytes as read for upload,
        // with the builder properties that decide their layout ("packaged":
        // every texture with a packaged layout, once each).
        if let (Ok(wanted), Some(directory)) = (
            super::debug_var("ECHORA_GPU_RAW_DUMP"),
            super::debug_var_os("ECHORA_DUMP_DIR"),
        ) && wanted.split(',').any(|item| {
            u64::from_str_radix(item.trim().trim_start_matches("0x"), 16) == Ok(base)
                || (item.trim() == "packaged"
                    && self
                        .object(texture, "Texture")
                        .is_ok_and(|object| object.builder.contains_key("PackagedTextureLayout"))
                    && !std::path::Path::new(&directory)
                        .join(format!("raw-{base:x}.txt"))
                        .exists())
        }) {
            let directory = std::path::Path::new(&directory);
            let _ = std::fs::create_dir_all(directory);
            let _ = std::fs::write(directory.join(format!("raw-{base:x}-{frame}.bin")), &raw);
            let _ = std::fs::write(directory.join(format!("raw-{base:x}.txt")), b"");
            let _ = std::fs::write(
                directory.join(format!("raw-{base:x}-{frame}.txt")),
                format!(
                    "{} block {}x{} bytes {} storage {storage_size:#x} levels {level_count} shape {shape:?} stride {stride} pool flags {pool_flags:#x}\n{:?}\n",
                    info.name,
                    info.block_width,
                    info.block_height,
                    info.bytes_per_block,
                    self.object(texture, "Texture")
                        .map(|object| &object.builder)
                ),
            );
        }
        let storage_hashes: Vec<u64> = raw.chunks(STORAGE_CHUNK as usize).map(hash_bytes).collect();
        let raw_copy = if uncached && storage_size <= SHADOW_LIMIT {
            raw.clone()
        } else {
            Vec::new()
        };
        let upload_start = std::time::Instant::now();
        // ECHORA_IDENTITY_LUT=1 (debugging) replaces RGBA8 3D textures
        // (colour lookup tables) with an identity table: neutral grading.
        let identity = matches!(shape, ImageShape::D3 { .. })
            && block == 4
            && !astc
            && super::debug_var_os("ECHORA_IDENTITY_LUT").is_some();
        let raw = if identity {
            let channel = |value: u32, extent: u32| (value * 255 / (extent - 1).max(1)) as u8;
            let mut table = raw;
            for z in 0..depth {
                for y in 0..height {
                    for x in 0..width {
                        let offset = crate::nvn_format::block_linear_offset(
                            info, width, height, depth, 1, x, y, z,
                        ) as usize;
                        if let Some(texel) = table.get_mut(offset..offset + 4) {
                            texel.copy_from_slice(&[
                                channel(x, width),
                                channel(y, height),
                                channel(z, depth),
                                255,
                            ]);
                        }
                    }
                }
            }
            self.gpu
                .as_mut()
                .unwrap()
                .note("identity lookup table for a 3D RGBA8 texture (approximation)");
            table
        } else {
            raw
        };
        // Array layers each hold a whole mip chain at a fixed stride.
        let layer_stride = storage_size / u64::from(layers);
        let copy_block = |out: &mut [u8], at: u64| {
            let at = at as usize;
            match raw.get(at..at + out.len()) {
                Some(bytes) => out.copy_from_slice(bytes),
                None => out.fill(0),
            }
        };
        let extent_at = |level: u32| {
            [
                (width >> level).max(1),
                (height >> level).max(1),
                (depth >> level).max(1),
            ]
        };
        let layer_bytes = |[w, h, d]: [u32; 3]| {
            let (columns, rows, block_bytes) = if decode_astc {
                (w, h, 4)
            } else {
                (
                    w.div_ceil(info.block_width),
                    h.div_ceil(info.block_height),
                    block,
                )
            };
            (columns as usize)
                .checked_mul(rows as usize)?
                .checked_mul(d as usize)?
                .checked_mul(block_bytes)
        };
        let upload_bytes = (0..level_count)
            .try_fold(0usize, |total, level| {
                let stride =
                    layer_bytes(extent_at(level))?.checked_next_multiple_of(upload_alignment)?;
                total.checked_add(stride.checked_mul(layers as usize)?)
            })
            .ok_or("texture upload size overflow")?;
        let stream = decode_astc
            && layers > 1
            && super::debug_var_os("ECHORA_TEXTURE_DUMP").is_none()
            && super::debug_var_os("ECHORA_GPU_TEXTURE_DUMP").is_none()
            && self.gpu.as_ref().unwrap().renderer.can_generate_upload(
                texture,
                upload_bytes,
                layer_bytes([width, height, depth]).unwrap(),
            );
        let mut texels = Vec::new();
        if !stream {
            texels
                .try_reserve_exact(upload_bytes)
                .map_err(|err| format!("texture upload: {err}"))?;
        }
        let mut levels = Vec::with_capacity(level_count as usize * layers as usize);
        let mips: Vec<_> =
            crate::nvn_format::block_linear_mips(info, [width, height, depth], layout, level_count)
                .collect();
        let pack_layers =
            |level: u32, first_layer: usize, output: &mut [u8]| -> Result<(), String> {
                let [level_width, level_height, level_depth] = extent_at(level);
                let (columns, rows) = (
                    level_width.div_ceil(info.block_width),
                    level_height.div_ceil(info.block_height),
                );
                let mip = &mips[level as usize];
                let level_offset = mip.offset;
                let level_shape = mip.shape;
                let row_bytes = columns as usize * block;
                let slice_bytes = row_bytes * rows as usize;
                let gather_layer = |layer: usize, data: &mut [u8]| {
                    let layer_base = (first_layer + layer) as u64 * layer_stride + level_offset;
                    if let Some(rows) = copied.as_ref().filter(|rows| rows.len() == data.len()) {
                        data.copy_from_slice(rows);
                        return;
                    }
                    for z in 0..level_depth {
                        for row in 0..rows {
                            let from = z as usize * slice_bytes + row as usize * row_bytes;
                            let line = &mut data[from..from + row_bytes];
                            if stride != 0 {
                                copy_block(line, u64::from(row) * stride);
                                continue;
                            }
                            // Blocks dividing 16 bytes: the row's 16-byte runs.
                            if 16 % block == 0 {
                                for (start, offset) in crate::nvn_format::block_linear_row_runs(
                                    info,
                                    level_width,
                                    level_height,
                                    level_shape,
                                    row,
                                    z,
                                ) {
                                    let start = start as usize;
                                    let end = (start + 16).min(row_bytes);
                                    copy_block(&mut line[start..end], layer_base + offset);
                                }
                                continue;
                            }
                            // Copy complete blocks at consecutive storage addresses.
                            let mut column = 0;
                            while column < columns {
                                let run = crate::nvn_format::block_linear_run(info, column)
                                    .min(columns - column);
                                let offset = crate::nvn_format::block_linear_offset_in(
                                    info,
                                    level_width,
                                    level_height,
                                    level_shape,
                                    column,
                                    row,
                                    z,
                                );
                                copy_block(
                                    &mut line[column as usize * block..][..run as usize * block],
                                    layer_base + offset,
                                );
                                column += run;
                            }
                        }
                    }
                };
                let length = layer_bytes([level_width, level_height, level_depth])
                    .expect("the size of each mip was checked");
                let upload_stride = length.next_multiple_of(upload_alignment);
                if decode_astc {
                    crate::nvn_format::decode_astc_layers(
                        output,
                        [level_width, level_height, level_depth],
                        info,
                        gather_layer,
                    )
                    .map_err(|error| format!("ASTC decode: {error}"))?;
                } else {
                    for (layer, slice) in output.chunks_exact_mut(upload_stride).enumerate() {
                        gather_layer(layer, &mut slice[..length]);
                    }
                }
                Ok(())
            };
        let mut offset = 0;
        for level in 0..level_count {
            let [level_width, level_height, level_depth] = extent_at(level);
            let length = layer_bytes([level_width, level_height, level_depth])
                .expect("the size of each mip was checked");
            let upload_stride = length.next_multiple_of(upload_alignment);
            if !stream {
                texels.resize(offset + upload_stride * layers as usize, 0);
                let output = &mut texels[offset..];
                pack_layers(level, 0, output)?;
                let upload = &output[..length];
                // ECHORA_GPU_TEXTURE_DUMP=<dir> (debugging): the base level
                // of large 4-byte 2D textures, as last uploaded.
                if let Some(directory) = super::debug_var_os("ECHORA_GPU_TEXTURE_DUMP") {
                    let path = std::path::Path::new(&directory).join(format!(
                        "tex-{base:x}-{level_width}x{level_height}-{}.ppm",
                        info.name
                    ));
                    if level == 0 && block == 4 && level_depth == 1 && level_width >= 512 {
                        let mut ppm =
                            format!("P6\n{level_width} {level_height}\n255\n").into_bytes();
                        for texel in upload.as_chunks::<4>().0 {
                            ppm.extend_from_slice(&texel[..3]);
                        }
                        let _ = std::fs::write(&path, ppm);
                        let mut pgm =
                            format!("P5\n{level_width} {level_height}\n255\n").into_bytes();
                        pgm.extend(upload.as_chunks::<4>().0.iter().map(|texel| texel[3]));
                        let _ = std::fs::write(path.with_extension("alpha.pgm"), pgm);
                    }
                    // Compressed (16-byte block) textures: the untiled
                    // blocks, to compare with a reference capture.
                    if level == 0 && block == 16 && level_width >= 512 {
                        let _ = std::fs::write(path.with_extension("bin"), upload);
                    }
                }
            }
            levels.extend((0..layers).map(|layer| UploadRegion {
                level,
                layer,
                width: level_width,
                height: level_height,
                depth: level_depth,
                offset: (offset + layer as usize * upload_stride) as u64,
            }));
            offset += upload_stride * layers as usize;
        }
        // Content-keyed: guest writes that change the texels upload again.
        let generation = if stream {
            // The new image is generated from these storage bytes. Use
            // their hash without rereading the mapped Vulkan staging ring.
            hash_bytes(&raw)
        } else {
            hash_bytes(&texels)
        };
        // ECHORA_TEXTURE_DUMP: each distinct upload, named by key, format,
        // size and content hash: 4-byte texels as a PAM image (RGBA), other
        // formats as their raw blocks.
        if let Some(directory) = super::debug_var_os("ECHORA_TEXTURE_DUMP") {
            // Level 0 only.
            let texels = &texels[..levels
                .get(1)
                .map_or(texels.len(), |level| level.offset as usize)];
            if texels.len() != width as usize * height as usize * 4 {
                let path = std::path::Path::new(&directory).join(format!(
                    "texture-{texture:x}-{}-{width}x{height}-{generation:016x}.bin",
                    info.name
                ));
                if !path.exists() {
                    let _ = std::fs::create_dir_all(directory);
                    let _ = std::fs::write(path, texels);
                }
            } else {
                let path = std::path::Path::new(&directory).join(format!(
                    "texture-{texture:x}-{}-{generation:016x}.pam",
                    info.name
                ));
                if !path.exists() {
                    let mut image = format!(
                        "P7\nWIDTH {width}\nHEIGHT {height}\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n"
                    )
                    .into_bytes();
                    image.extend_from_slice(texels);
                    let _ = std::fs::create_dir_all(directory);
                    let _ = std::fs::write(path, image);
                }
            }
        }
        // Compare the packaged and derived layouts when an upload is made.
        let layout_note = layout.shape.and_then(|shape| {
            let derived = crate::nvn_format::base_block_shape(info, height, depth, layers);
            (shape != derived).then(|| {
                let builder = &self.objects[&texture].builder;
                format!(
                    "layout shape {shape:?} instead of {derived:?}: {} {width}x{height}x{depth} layers {layers} levels {levels_set} target {:?} packaged {:016x}",
                    info.name,
                    builder.get("Target").map(|value| value.x[0] as u32),
                    builder.get("PackagedTextureLayout").map_or(0, |value| value.x[1])
                )
            })
        });
        let levels: Arc<[UploadRegion]> = levels.into();
        let source = TextureSource::Upload {
            key: texture,
            generation,
            width,
            height,
            shape,
            format: vk_format,
            texels: Arc::new(texels),
            levels: levels.clone(),
        };
        let backend = self.gpu.as_mut().unwrap();
        if stream {
            backend
                .renderer
                .upload_generated(&source, upload_bytes, |first, output| {
                    let end = first + output.len() as u64;
                    let start = levels.partition_point(|region| region.offset < first);
                    let limit = levels.partition_point(|region| region.offset < end);
                    let mut index = start;
                    while index < limit {
                        let region = levels[index];
                        let count =
                            levels[index..limit].partition_point(|next| next.level == region.level);
                        let next_offset = levels
                            .get(index + count)
                            .map_or(upload_bytes as u64, |next| next.offset);
                        let start = (region.offset - first) as usize;
                        let finish = (next_offset - first) as usize;
                        pack_layers(
                            region.level,
                            region.layer as usize,
                            &mut output[start..finish],
                        )?;
                        index += count;
                    }
                    Ok(())
                })?;
            backend.stat("stat: streamed texture upload bytes", upload_bytes as u64);
        }
        if compressed_astc.is_some() {
            backend.stat("stat: compressed ASTC uploads", 1);
            backend.stat("stat: compressed ASTC upload bytes", upload_bytes as u64);
        }
        if matches!(&backing, TextureBacking::Virtual { .. }) {
            backend.stat("stat: texture uploads across virtual mappings", 1);
        }
        backend
            .textures
            .insert(texture, (sample, source.clone(), backing));
        backend.cached_texture_ranges = None;
        backend.storage_hashes.insert(texture, storage_hashes);
        if uncached && storage_size <= SHADOW_LIMIT {
            if backend.shadows.contains_key(&texture) {
                backend.stat("stat: uncached uploads changed", 1);
            }
            backend.shadows.insert(texture, raw_copy);
        } else {
            backend.shadows.remove(&texture);
        }
        backend.stat("stat: texture uploads", 1);
        if let Some(note) = layout_note {
            backend.note(note);
        }
        // ECHORA_UPLOAD_LOG (debugging): uploads counted by shape, to find
        // textures re-uploaded every present.
        if super::debug_var_os("ECHORA_UPLOAD_LOG").is_some() {
            let layout = if stride != 0 { "pitch" } else { "block" };
            let note = format!(
                "upload {} {width}x{height} {layout} at {base:#x} (texture {texture:#x})",
                info.name
            );
            *backend.notes.entry(note).or_default() += 1;
        }
        backend.stat("stat: texture checks", 1);
        backend.stat("stat: texture check microseconds", check_micros);
        if beyond_limit {
            backend.stat("stat: uploads beyond the host's image limit", 1);
        }
        backend.stat(
            "stat: texture upload microseconds",
            upload_start.elapsed().as_micros() as u64,
        );
        self.gpu
            .as_mut()
            .unwrap()
            .texture_checked
            .insert(texture, frame);
        Ok(source)
    }

    /// The texture object a bound handle names through the bound pool.
    fn handle_texture(&self, handle: u64) -> Result<u64, String> {
        let pool = self
            .render
            .state
            .get("SetTexturePool")
            .ok_or("no texture pool is set")?
            .x[1];
        let id = (handle & 0xf_ffff) as u32;
        let entry = match self.render_texture_descriptor(pool, id) {
            Some(entry) => entry,
            None => {
                self.object(pool, "TexturePool")?;
                return Err(format!("texture {id} is not registered"));
            }
        };
        // The structure was reused for other storage or in another format
        // since registration: the descriptor still names the texture it
        // held then.
        let registered = (entry.x[2], entry.x[3] & 0xffff_ffff);
        let current = self.texture_identity(entry.x[0]);
        let mut resolved = entry.x[0];
        if registered.0 != 0 && current != Some(registered) {
            let retired = super::retired_texture(registered);
            if self.objects.contains_key(&retired) {
                resolved = retired;
            }
        }
        // ECHORA_HANDLE_LOG=<storage hex>[,...] (debugging): each lookup of
        // a descriptor registered over that storage (or whose texture is
        // over it now), with the identity
        // (storage, format) registered and the object's now.
        if super::debug_var("ECHORA_HANDLE_LOG").is_ok_and(|wanted| {
            wanted.split(',').any(|item| {
                let item = u64::from_str_radix(item.trim().trim_start_matches("0x"), 16);
                item == Ok(registered.0) || current.is_some_and(|(storage, _)| item == Ok(storage))
            })
        }) {
            eprintln!(
                "handle present {} id {id} object {:x} registered {:x?} now {current:x?} at {} resolves {resolved:x}",
                self.presented_frames,
                entry.x[0],
                registered,
                entry.x[3] >> 32
            );
        }
        Ok(resolved)
    }

    /// The filters and wrap modes of the sampler a texture handle names
    /// (bits 20.. index the bound sampler pool). An unregistered sampler
    /// samples linearly, clamped to the edge. NVN filter and wrap values are
    /// inferred: min filters NEAREST, LINEAR, then NEAREST/LINEAR with the
    /// nearest mip level and NEAREST/LINEAR blending two levels, mag NEAREST,
    /// LINEAR; wrap CLAMP, REPEAT, MIRROR_CLAMP, MIRROR_CLAMP_TO_EDGE,
    /// MIRROR_CLAMP_TO_BORDER, CLAMP_TO_BORDER, MIRRORED_REPEAT,
    /// CLAMP_TO_EDGE.
    fn handle_sampler(&self, handle: u64) -> Sampler {
        let id = (handle >> 20 & 0xfff) as u32;
        self.render
            .state
            .get("SetSamplerPool")
            .and_then(|bind| self.sampler_states.get(&bind.x[1]))
            .and_then(|pool| pool.get(&id))
            .copied()
            .unwrap_or_default()
    }

    pub(super) fn sampler_from_properties(props: &BTreeMap<String, super::Args>) -> Sampler {
        let mut sampler = Sampler::default();
        let filter = |value: u64| match value {
            0 | 2 | 4 => vk::Filter::NEAREST,
            _ => vk::Filter::LINEAR,
        };
        let wrap = |value: u64| match value {
            1 => vk::SamplerAddressMode::REPEAT,
            // Mirror-once modes approximate as mirrored repeat (the device
            // feature for mirror clamp is not enabled).
            2..=4 | 6 => vk::SamplerAddressMode::MIRRORED_REPEAT,
            5 => vk::SamplerAddressMode::CLAMP_TO_BORDER,
            _ => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        };
        if let Some(args) = props.get("MinMagFilter") {
            sampler.min_filter = filter(args.x[0]);
            sampler.mag_filter = filter(args.x[1]);
            // Mipmapped min filters: 2/3 pick the nearest level, 4/5 blend
            // two levels; 0/1 sample only level 0.
            sampler.base_level_only = args.x[0] < 2;
            sampler.mipmap = if matches!(args.x[0], 4 | 5) {
                vk::SamplerMipmapMode::LINEAR
            } else {
                vk::SamplerMipmapMode::NEAREST
            };
        }
        if let Some(args) = props.get("WrapMode") {
            sampler.address = [wrap(args.x[0]), wrap(args.x[1]), wrap(args.x[2])];
        }
        // SetLodClamp(builder, float min, float max) and SetLodBias(builder,
        // float bias): the floats are the first vector registers.
        let float = |bits: u64| f32::from_bits(bits as u32);
        if let Some(args) = props.get("MaxAnisotropy") {
            let value = float(args.v[0]);
            if value.is_finite() {
                sampler.max_anisotropy = value.max(1.0);
            }
        }
        if let Some(args) = props.get("LodClamp") {
            let (min, max) = (float(args.v[0]), float(args.v[1]));
            if min.is_finite() && max.is_finite() {
                sampler.lod[0] = min.max(0.0);
                sampler.lod[1] = max.max(0.0);
            }
        }
        if let Some(args) = props.get("LodBias") {
            let bias = float(args.v[0]);
            if bias.is_finite() {
                sampler.lod[2] = bias.clamp(-16.0, 16.0);
            }
        }
        // SetCompare(builder, mode, function): mode 1 (compare R to the
        // texture) makes a shadow-map sampler; the function in the depth
        // test's numbering.
        if let Some(args) = props.get("Compare")
            && args.x[0] != 0
        {
            sampler.compare = Some(compare_op(args.x[1]));
        }
        // SetBorderColor (floats): the colour clamp-to-border reads outside
        // the texture (1.0 for the game's shadow maps: outside is lit).
        if let Some(args) = props.get("BorderColor") {
            sampler.border = args.x.map(|word| word as u32);
        }
        sampler
    }

    /// Front and back stencil state: the snapshot's StencilFunc(face, func)
    /// and StencilOp(face, fail, depth fail, pass) with the command
    /// buffer's SetStencilRef / SetStencilValueMask / SetStencilMask(face,
    /// value). Faces 1 front, 2 back, 3 both (inferred); each keeps only its
    /// latest call, which applies to the faces it names.
    fn stencil_faces(&self, depth_state: &BTreeMap<String, super::Args>) -> [StencilFace; 2] {
        let mut faces = [StencilFace {
            fail: vk::StencilOp::KEEP,
            pass: vk::StencilOp::KEEP,
            depth_fail: vk::StencilOp::KEEP,
            compare: vk::CompareOp::ALWAYS,
            compare_mask: 0xff,
            write_mask: 0xff,
            reference: 0,
        }; 2];
        let applies = |face: u64, index: usize| face >> index & 1 != 0;
        for (index, face) in faces.iter_mut().enumerate() {
            if let Some(args) = depth_state.get("StencilFunc") {
                if applies(args.x[0], index) {
                    face.compare = compare_op(args.x[1]);
                }
            }
            if let Some(args) = depth_state.get("StencilOp") {
                if applies(args.x[0], index) {
                    face.fail = stencil_op(args.x[1]);
                    face.depth_fail = stencil_op(args.x[2]);
                    face.pass = stencil_op(args.x[3]);
                }
            }
            for (name, value) in [
                ("SetStencilRef", &mut face.reference),
                ("SetStencilValueMask", &mut face.compare_mask),
                ("SetStencilMask", &mut face.write_mask),
            ] {
                if let Some(command) = self.render.state.get(name) {
                    if applies(command.x[1], index) {
                        *value = command.x[2] as u32;
                    }
                }
            }
        }
        faces
    }

    /// Show presented texture `texture` in the host window (ECHORA_WINDOW).
    /// Returns false when there is no window to present to.
    /// Submit the recorded work at each present without waiting for it (the
    /// GPU runs a frame behind; reads of results wait in `resolve_storage`).
    /// The GPU has finished all work given to it: a fence the CPU waits
    /// for has then really signaled. Frames otherwise run on the GPU while
    /// the CPU prepares the next, and the application rewrites what it
    /// believes finished frames read (its per-frame parameter blocks, read
    /// in place from guest memory).
    pub(super) fn gpu_finish(&mut self) -> Result<(), String> {
        let Some(backend) = self.gpu.as_mut() else {
            return Ok(());
        };
        backend.stat("stat: GPU waits for a CPU sync wait", 1);
        backend.renderer.flush()
    }

    /// Whether GPU results wait to be stored in guest memory at a CPU
    /// wait (`resolve_storage`), and the renderer's completed serial.
    pub(super) fn gpu_resolve_state(
        &self,
    ) -> (bool, Option<std::sync::Arc<std::sync::atomic::AtomicU64>>) {
        let Some(backend) = self.gpu.as_ref() else {
            return (false, None);
        };
        let pending = !backend.pending_storage.is_empty()
            || !backend.in_place_writes.is_empty()
            || backend
                .resident
                .values()
                .any(|pool| pool.uncached && !pool.gpu_dirty.is_empty());
        (pending, Some(backend.renderer.completed_handle()))
    }

    /// Serial of the GPU submission holding everything recorded so far
    /// (0 without a GPU).
    pub(super) fn gpu_recorded_serial(&self) -> u64 {
        self.gpu
            .as_ref()
            .map_or(0, |backend| backend.renderer.recorded_serial())
    }

    /// A CPU wait on a signaled sync: waits for the submission that held
    /// the work before its fence, or for everything when that is unknown.
    /// The submission a CPU read would wait for first (stores through the
    /// page table, `resolve_gpu_written`), submitted if still recording;
    /// None when none or it has finished. The caller waits for it without
    /// the NVN state (`gpu_poll_completed`, `gpu_completed_serial`).
    pub(super) fn gpu_read_wait_serial(&mut self) -> Result<Option<(u64, u64)>, String> {
        let Some(backend) = self.gpu.as_mut() else {
            return Ok(None);
        };
        if !backend.page_table_wrote || !backend.page_table_writes {
            return Ok(None);
        }
        let serial = backend
            .page_table_write_serial
            .min(backend.observable_serial);
        // Up to the last signal: its marker, when it is in that batch.
        let marker = backend
            .observable_marker
            .filter(|&marker| backend.renderer.marker_serial(marker) == Some(serial))
            .unwrap_or(0);
        if serial <= backend.renderer.completed_serial()
            || (marker != 0 && backend.renderer.marker_done(marker))
        {
            return Ok(None);
        }
        backend.renderer.submit_through(serial)?;
        Ok(Some((serial, marker)))
    }

    /// Submit the batch with this serial if it is still being recorded.
    pub(super) fn gpu_submit_through(&mut self, serial: u64) -> Result<(), String> {
        match self.gpu.as_mut() {
            Some(backend) => backend.renderer.submit_through(serial),
            None => Ok(()),
        }
    }

    /// Note finished submissions without waiting.
    pub(super) fn gpu_poll_completed(&mut self) -> Result<(), String> {
        match self.gpu.as_mut() {
            Some(backend) => backend.renderer.poll_completed(),
            None => Ok(()),
        }
    }

    fn archive_inactive_targets(&mut self) -> Result<(), String> {
        let Some(backend) = self.gpu.as_mut() else {
            return Ok(());
        };
        use echora_gpu::renderer::ArchiveRequest;
        backend.renderer.finish_target_archives()?;
        let present = self.presented_frames;
        let mut candidates: Vec<_> = backend
            .retired_since
            .iter()
            .filter_map(|(&key, &seen)| {
                if present.saturating_sub(seen) < 24 {
                    return None;
                }
                let target = backend.renderer.existing_target(key)?;
                let (width, height) = target.extent(0);
                Some((std::cmp::Reverse(u64::from(width) * u64::from(height)), key))
            })
            .collect();
        candidates.sort_unstable();
        for (_, key) in candidates.into_iter().take(6) {
            match backend.renderer.queue_target_archive(key)? {
                ArchiveRequest::Queued | ArchiveRequest::Ineligible => {
                    backend.retired_since.remove(&key);
                }
                ArchiveRequest::Deferred => {}
            }
        }
        Ok(())
    }

    /// The serial of the last finished submission.
    pub(super) fn gpu_completed_serial(&self) -> u64 {
        self.gpu
            .as_ref()
            .map_or(u64::MAX, |backend| backend.renderer.completed_serial())
    }

    /// Work recorded before submission `serial` ends at a fence or event
    /// the guest can see signaled.
    /// A marker is set here (`Renderer::mark`): a CPU wait for this point
    /// (a sync wait, a read of what the work before it stored) then ends
    /// when the GPU passes it, not when the rest of its batch, what the
    /// replay records after it, has run too. The marker, if one was set.
    pub(super) fn gpu_observable(&mut self, serial: u64) -> Option<u64> {
        let backend = self.gpu.as_mut()?;
        backend.observable_serial = backend.observable_serial.max(serial);
        let marker = match backend.renderer.mark() {
            Ok(marker) => marker,
            Err(error) => {
                backend.note(format!("marking a signal: {error}"));
                None
            }
        };
        backend.observable_marker = marker;
        backend.page_table_wrote_after_signal = false;
        marker
    }

    /// Whether the GPU passed marker `marker` (0: none).
    pub(super) fn gpu_marker_done(&self, marker: u64) -> bool {
        marker != 0
            && self
                .gpu
                .as_ref()
                .is_some_and(|backend| backend.renderer.marker_done(marker))
    }

    pub(super) fn gpu_wait_sync(&mut self, serial: Option<u64>) -> Result<(), String> {
        let Some(backend) = self.gpu.as_mut() else {
            return Ok(());
        };
        backend.stat("stat: GPU waits for a CPU sync wait", 1);
        // Census: how far the waited-for work is from what is recorded.
        let recorded = backend.renderer.recorded_serial();
        let census = match serial {
            None => "stat: sync waits for everything (no serial)",
            Some(serial) if serial >= recorded && recorded > backend.renderer.flushes => {
                "stat: sync waits on work still recording"
            }
            Some(serial) if serial + 1 >= recorded => "stat: sync waits on the last submission",
            Some(_) => "stat: sync waits on earlier submissions",
        };
        backend.stat(census, 1);
        match serial {
            Some(serial) => backend.renderer.wait_for(serial),
            None => backend.renderer.flush(),
        }
    }

    pub(super) fn gpu_submit(&mut self) -> Result<(), String> {
        self.archive_inactive_targets()?;
        match self.gpu.as_mut() {
            Some(backend) => backend.renderer.submit_frame(),
            None => Ok(()),
        }
    }

    pub(super) fn gpu_has_presenter(&self) -> bool {
        self.gpu
            .as_ref()
            .and_then(|backend| backend.presenter.as_ref())
            .is_some_and(Result::is_ok)
    }

    pub(super) fn gpu_present_window(&mut self, texture: u64) -> Result<bool, String> {
        let Some(window) = crate::window::shared() else {
            return Ok(false);
        };
        let backend = self.gpu.as_mut().unwrap();
        if backend.presenter.is_none() {
            use echora_config::VsyncMode;
            use vk::PresentModeKHR as Mode;
            let preferred_modes: &'static [Mode] = match crate::settings::get().display.vsync {
                VsyncMode::On => &[Mode::MAILBOX],
                VsyncMode::Adaptive => &[Mode::FIFO_RELAXED],
                VsyncMode::Off => &[Mode::IMMEDIATE, Mode::MAILBOX],
            };
            let created = window
                .handle()
                .ok_or_else(|| "no window".to_string())
                .and_then(|handle| Presenter::new(&backend.renderer.gpu, handle, preferred_modes));
            backend.presenter = Some(created.map_err(|error| {
                backend.note(format!("window presentation failed: {error}"));
            }));
        }
        let Some(Ok(presenter)) = backend.presenter.as_mut() else {
            return Ok(false);
        };
        if backend.renderer.existing_target(texture).is_none() {
            return Ok(true);
        }
        backend
            .renderer
            .present(presenter, texture, window.size())?;
        Ok(true)
    }

    /// Execute DispatchCompute(cmd, x, y, z) on the GPU. Bindings follow
    /// the inferred map: uniform buffer slot s is c(3+s), texture slot s's
    /// handle is at c0 + 0x40 + 8 * s; storage buffer slot s's address and size sit in driver
    /// constant buffer c0 at 0x20 + 16 * s (inferred: two programs read c0
    /// +0x20 and +0x30, and share the buffer bound to slot 1), so c0 is
    /// synthesized with the bound storage buffers. Each
    /// storage buffer is passed from its 16-byte-aligned base, as the
    /// program indexes it, and written back to guest memory afterwards.
    /// CopyTextureToTexture between render targets: a scaled blit of the
    /// source region into the destination region. Copies from textures
    /// that were never rendered to, and views of other levels or layers,
    /// are not modeled.
    /// The image a texture renders into: 2D arrays keep their layers and
    /// cubes their faces (in Depth, as for sampling), so a view's layer
    /// selects the same subresource when rendering and sampling.
    fn render_shape(&self, texture: u64) -> ImageShape {
        let Ok(object) = self.object(texture, "Texture") else {
            return ImageShape::D2 { layers: 1 };
        };
        let get = |name: &str| object.builder.get(name).map(|value| value.x[0] as u32);
        let depth = get("Depth").unwrap_or(1).max(1);
        match get("Target").unwrap_or(1) {
            // TARGET_3D: a volume of all its slices (the fog's froxel
            // volumes are 96 deep), as the dispatch image binding makes it;
            // one 2D layer would drop every store past slice 0.
            2 => ImageShape::D3 { depth },
            4 => ImageShape::D2 { layers: depth },
            // TARGET_CUBEMAP (and arrays): six faces even when Depth is
            // left at 1, as the upload path makes cubes; one 2D layer
            // left a cube sample nothing to read.
            8 | 9 => ImageShape::Cube {
                layers: depth.max(6) / 6 * 6,
            },
            _ => ImageShape::D2 { layers: 1 },
        }
    }

    /// The render target of texture `texture`, created at its size,
    /// format, levels and layers.
    #[track_caller]
    fn ensure_render_target(&mut self, texture: u64) -> Result<(u32, u32), String> {
        if self.depth_traced(texture)
            && self
                .gpu
                .as_ref()
                .is_some_and(|backend| backend.renderer.existing_target(texture).is_none())
        {
            eprintln!(
                "depth trace present {} target {texture:x} created from {}",
                self.presented_frames,
                std::panic::Location::caller()
            );
        }
        let generation = super::debug_generation();
        if self.target_shapes.0 != generation {
            self.target_shapes = (generation, HashMap::default());
        }
        let (width, height, format, levels, shape, storage) =
            match self.target_shapes.1.get(&texture) {
                Some(known) => *known,
                None => {
                    let (width, height, format) = self.target_shape(texture)?;
                    const TEXTURE_FLAG_IMAGE: u64 = 0x4;
                    let storage = self
                        .object(texture, "Texture")?
                        .builder
                        .get("Flags")
                        .is_some_and(|flags| flags.x[0] & TEXTURE_FLAG_IMAGE != 0);
                    let known = (
                        width,
                        height,
                        format,
                        self.texture_levels(texture),
                        self.render_shape(texture),
                        storage,
                    );
                    self.target_shapes.1.insert(texture, known);
                    known
                }
            };
        let backend = self.gpu.as_mut().unwrap();
        if storage {
            backend
                .renderer
                .guest_sized_target(texture, width, height, shape, format, levels)?;
        } else {
            backend
                .renderer
                .shaped_target(texture, width, height, shape, format, levels)?;
        }
        Ok((width, height))
    }

    /// ECHORA_DEPTH_TRACE=<storage hex> (debugging): whether `texture`'s
    /// storage is the traced one; draws, clears and copies over it are
    /// logged.
    pub(super) fn depth_traced(&self, texture: u64) -> bool {
        // ECHORA_DEPTH_TRACE=<storage>[,...].
        super::debug_var("ECHORA_DEPTH_TRACE").is_ok_and(|wanted| {
            let storage = self.texture_storage(texture);
            wanted.split(',').any(|item| {
                u64::from_str_radix(item.trim().trim_start_matches("0x"), 16).ok() == storage
            })
        })
    }

    pub(super) fn gpu_copy_texture(
        &mut self,
        _core: &Guest,
        command: &Command,
    ) -> Result<(), String> {
        if self.depth_traced(command.x[1]) || self.depth_traced(command.x[4]) {
            eprintln!(
                "depth trace present {} copy {:x} ({:x?}) -> {:x} ({:x?})",
                self.presented_frames,
                command.x[1],
                self.texture_storage(command.x[1]),
                command.x[4],
                self.texture_storage(command.x[4])
            );
        }
        // Views of level 0 and layer 0 are the whole texture.
        if command
            .captured
            .get(12..16)
            .is_none_or(|bases| bases.iter().any(|base| *base != 0))
        {
            return Err("texture views of other levels or layers".into());
        }
        let (from, to) = (command.x[1], command.x[4]);
        let rect = |base: usize| -> [i32; 4] {
            std::array::from_fn(|index| command.captured[base + [0, 1, 3, 4][index]] as u32 as i32)
        };
        let (source, destination) = (rect(0), rect(6));
        // Either end may be another texture object over a render target's
        // storage (a depth buffer copied for a later pass to test against):
        // that target.
        let target = |this: &Self, texture: u64| {
            let renderer = &this.gpu.as_ref().unwrap().renderer;
            if renderer.target_info(texture).is_some() {
                Some(texture)
            } else {
                this.target_sharing_storage(texture)
            }
        };
        let from = target(self, from).ok_or("source is not a render target")?;
        let to = target(self, to).unwrap_or(to);
        if self.depth_traced(command.x[1]) || self.depth_traced(command.x[4]) {
            eprintln!(
                "depth trace present {} copy resolved {from:x} -> {to:x} source {source:?} destination {destination:?}",
                self.presented_frames
            );
        }
        self.ensure_render_target(to)?;
        let traced = self.depth_traced(command.x[1]) || self.depth_traced(command.x[4]);
        let present = self.presented_frames;
        let renderer = &mut self.gpu.as_mut().unwrap().renderer;
        renderer.restore_archived_target(from)?;
        renderer.defined(to);
        renderer.copy_target(from, source, to, destination)?;
        // A traced copy: both ends read back after it, and how many texels
        // differ (a depth copy a later EQUAL test relies on must be exact).
        if traced {
            renderer.flush()?;
            let formats = (
                renderer.existing_target(from).map(|target| target.format),
                renderer.existing_target(to).map(|target| target.format),
            );
            let (a, b) = (
                renderer.read_target_level(from, 0),
                renderer.read_target_level(to, 0),
            );
            match (a, b) {
                (Ok(a), Ok(b)) => {
                    let mean = |texels: &[[f32; 4]]| {
                        texels.iter().map(|texel| f64::from(texel[0])).sum::<f64>()
                            / texels.len().max(1) as f64
                    };
                    let differ = a.iter().zip(&b).filter(|(x, y)| x[0] != y[0]).count();
                    eprintln!(
                        "depth trace present {present} copy check formats {formats:?} means {:.6} {:.6} texels {} {} differ {differ}",
                        mean(&a),
                        mean(&b),
                        a.len(),
                        b.len()
                    );
                }
                (a, b) => eprintln!(
                    "depth trace present {present} copy check unreadable {:?} {:?}",
                    a.err(),
                    b.err()
                ),
            }
        }
        Ok(())
    }

    pub(super) fn gpu_dispatch(
        &mut self,
        core: &mut Guest,
        command: &Command,
    ) -> Result<(), String> {
        // What later texture bindings resolve to is not changed by running
        // it: images it writes are targets (their bindings name them), and
        // results it stores back into a texture's memory invalidate it
        // (`gpu_invalidate`, which moves `binding_serial`); guest textures
        // are checked again each present.
        let bind = self
            .render
            .state
            .get("BindProgram@5")
            .ok_or("no compute program bound")?;
        let program = self
            .objects
            .get(&bind.x[1])
            .ok_or("bound compute program is not initialized")?;
        let stage_args = program
            .state
            .get("Stage0")
            .ok_or("bound program has no shaders")?;
        let (address, control) = (stage_args.x[0], stage_args.x[1]);
        // Control section words (inferred from two programs): workgroup
        // size x, y, z at 0x58, 0x5c, 0x60, shared memory bytes at 0x64 and
        // local memory bytes per invocation at 0x68 (inferred: in all 321
        // captured compute programs it covers every constant local offset).
        let mut compute = [0u32; 5];
        for (index, value) in compute.iter_mut().enumerate() {
            *value = core
                .memory
                .read_u32(control + 0x58 + 4 * index as u64)
                .map_err(|trap| format!("compute control section: {trap:?}"))?;
        }
        if compute[..3].iter().any(|size| *size == 0 || *size > 1024) {
            return Err(format!("compute workgroup {:?}", &compute[..3]));
        }
        // The jump proof and the shader bind see the same snapshot of
        // bank 1. Only the bytes read by the table shape the translation.
        let constants = self.stage_constants_of(core, *stage_args);
        let (mut stage, mut code_key) = self.translate_stage(
            core,
            address,
            [0; 3],
            None,
            [0; 32],
            compute,
            None,
            constants.as_deref(),
            &[],
        )?;
        // Storage pointers the NVN tracking found outside c0 were constants,
        // not addresses (floats 1.0 and 10.0 in c1): such a program reads
        // its pointers as raw arguments instead, picking among every
        // mapped candidate at run time.
        if stage.descriptors.iter().any(|descriptor| {
            descriptor.kind() == DescriptorKind::StorageBuffer
                && descriptor.cbuf_index != 0
                && descriptor.flags & 32 == 0
        }) && self.gpu.as_mut().unwrap().native_programs.insert(code_key)
        {
            (stage, code_key) = self.translate_stage(
                core,
                address,
                [0; 3],
                None,
                [0; 32],
                compute,
                None,
                constants.as_deref(),
                &[],
            )?;
        }
        if stage.stage != Stage::Compute {
            return Err(format!("bound compute stage is {:?}", stage.stage));
        }
        // ECHORA_GPU_C1_PROBE (debugging): for programs translated with raw
        // pointers, the words c1 could hold (the stage's constants, each
        // pushed root table, the first bound uniform buffers), every 300th
        // present.
        if super::debug_var_os("ECHORA_GPU_C1_PROBE").is_some()
            && self.presented_frames.is_multiple_of(300)
            && self
                .gpu
                .as_ref()
                .unwrap()
                .native_programs
                .contains(&code_key)
        {
            let words = |bytes: &[u8]| -> Vec<String> {
                bytes
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .take(4)
                    .map(|word| format!("{:#x}", u64::from_le_bytes(*word)))
                    .collect()
            };
            let constants = constants.as_deref().unwrap_or_default();
            eprintln!(
                "c1probe present {} {:016x} constants {:?}",
                self.presented_frames,
                code_key,
                words(constants)
            );
            for (table, root) in self.render.roots[1].iter().enumerate().take(4) {
                let bytes: Vec<u8> = root[..8]
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect();
                eprintln!("c1probe   root {table} {:?}", words(&bytes));
            }
            for slot in 0..4 {
                if let Some(bind) = self.render.bound_uniforms.get(5, slot) {
                    eprintln!("c1probe   ubo {slot} {:#x}+{:#x}", bind.x[3], bind.x[4]);
                }
            }
        }
        // ECHORA_GPU_SKIP_DISPATCH=<program key>[,...] (debugging): those
        // compute programs are not run.
        if super::debug_var("ECHORA_GPU_SKIP_DISPATCH").is_ok_and(|skipped| {
            skipped
                .split(',')
                .any(|skip| skip == format!("{:016x}", code_key))
        }) {
            return Ok(());
        }
        // For skipped dispatches: the program's bindings and the compute
        // slots bound, from which the driver constant layout is inferred.
        let layout = |nvn: &Self| {
            let descriptors: Vec<String> = stage
                .descriptors
                .iter()
                .map(|d| format!("{:?}@c{}+{:#x}", d.kind(), d.cbuf_index, d.cbuf_offset))
                .collect();
            let bound: Vec<&str> = nvn
                .render
                .state
                .keys()
                .filter(|key| key.contains("/5/"))
                .copied()
                .collect();
            format!(
                "program {:#x} uses [{}], bound [{}]",
                code_key,
                descriptors.join(" "),
                bound.join(" ")
            )
        };
        // The driver constant buffer c0 of a compute program is a table of
        // its resources, in this order: uniform buffers (16 bytes each:
        // address and size), storage buffers (16 bytes), texture handles
        // (8 bytes), image handles (8 bytes). The counts are bytes 0xb1 (+1),
        // 0xb2, 0xb3 and 0xb6 of the control section, whose word 0xb8 holds
        // the table's size in its high half (inferred from the compute
        // programs dumped).
        let mut header = [0u8; 12];
        core.memory
            .read(control + 0xb0, &mut header)
            .map_err(|trap| format!("compute control section: {trap:?}"))?;
        // The resources are packed against the table's end: uniform
        // buffers may be followed by padding (program 0xf9e5b02b5fabff78:
        // one uniform buffer, textures from 0x50 in a 0x70-byte table), so
        // the bases count back from the size.
        let table_end = u32::from(u16::from_le_bytes([header[10], header[11]]));
        let image_base = table_end.saturating_sub(8 * u32::from(header[6]));
        let texture_base = image_base.saturating_sub(8 * u32::from(header[3]));
        let storage_base = texture_base.saturating_sub(16 * u32::from(header[2]));
        if storage_base < 16 * (u32::from(header[1]) + 1) {
            return Err(format!(
                "compute resource table {header:02x?} overlaps its uniform buffers"
            ));
        }
        let slot = |descriptor: &shader::Descriptor, kind: DescriptorKind| -> Option<u32> {
            if descriptor.cbuf_index != 0 {
                return None;
            }
            let offset = descriptor.cbuf_offset;
            match kind {
                DescriptorKind::StorageBuffer if (storage_base..texture_base).contains(&offset) => {
                    Some((offset - storage_base) / 16)
                }
                DescriptorKind::Texture | DescriptorKind::TextureBuffer
                    if (texture_base..image_base).contains(&offset) =>
                {
                    Some((offset - texture_base) / 8)
                }
                DescriptorKind::Image if (image_base..table_end).contains(&offset) => {
                    Some((offset - image_base) / 8)
                }
                _ => None,
            }
        };
        let unplaced = |nvn: &Self, descriptor: &shader::Descriptor| {
            format!(
                "{:?} binding at c{}+{:#x} in compute (control 0xb0.. {:02x?}): {}",
                descriptor.kind(),
                descriptor.cbuf_index,
                descriptor.cbuf_offset,
                header,
                layout(nvn),
            )
        };
        let mut c0 = vec![0u8; (table_end as usize).max(0x100)];
        // The 16-byte head: the table's byte length, then the dispatch's
        // workgroup counts (x, y, z), which programs read to size their
        // work (a reduction's 1 / invocation count). The SDK's own buffers
        // under Lichii read [0xb0, 8, 9, 12] for an 8x9x12 dispatch whose
        // tables end at 0xb0. An indirect dispatch the GPU reads itself
        // (`indirect` below) keeps zero counts.
        c0[..4].copy_from_slice(&table_end.to_le_bytes());
        if !command.name.ends_with("DispatchComputeIndirect") {
            for (axis, count) in command.x[1..4].iter().enumerate() {
                let at = 4 + axis * 4;
                c0[at..at + 4].copy_from_slice(&(*count as u32).to_le_bytes());
            }
        }
        // After the head: root tables 0-7 (words pushed with
        // PushComputeConstants), their word counts at control +0xa9..0xb0,
        // each padded to four words; then the uniform buffers' addresses
        // and sizes (16 bytes each) up to the storage buffers.
        let mut root_counts = [0u8; 8];
        core.memory
            .read(control + 0xa9, &mut root_counts)
            .map_err(|trap| format!("compute control section: {trap:?}"))?;
        let mut at = 0x10usize;
        for (table, &count) in root_counts.iter().enumerate() {
            let words = usize::from(count).min(64);
            if let Some(root) = self.render.roots[1].get(table) {
                for (index, word) in root[..words].iter().enumerate() {
                    if let Some(slot) = c0.get_mut(at + index * 4..at + index * 4 + 4) {
                        slot.copy_from_slice(&word.to_le_bytes());
                    }
                }
            }
            at += words.next_multiple_of(4) * 4;
        }
        for slot in 0..u32::from(header[1]) {
            let entry = at + slot as usize * 16;
            if entry + 12 > storage_base as usize {
                break;
            }
            if let Some(bind) = self.render.bound_uniforms.get(5, u64::from(slot)) {
                c0[entry..entry + 8].copy_from_slice(&bind.x[3].to_le_bytes());
                c0[entry + 8..entry + 12].copy_from_slice(&(bind.x[4] as u32).to_le_bytes());
            }
        }
        // The storage buffers' addresses and sizes, every bound slot of the
        // table: a program reaching global memory through the page table
        // reads them from c0 itself (a counter's address, for one), with no
        // storage binding of its own.
        if stage.push_fields & echora_shader_abi::PUSH_PAGE_TABLE != 0 {
            for slot in 0..u32::from(header[2]) {
                let entry = (storage_base + slot * 16) as usize;
                if entry + 12 > texture_base as usize || entry + 12 > c0.len() {
                    break;
                }
                if let Some(bind) =
                    self.render
                        .state
                        .get(bind_key("BindStorageBuffer", Some(5), slot as u64))
                {
                    c0[entry..entry + 8].copy_from_slice(&bind.x[3].to_le_bytes());
                    c0[entry + 8..entry + 12].copy_from_slice(&(bind.x[4] as u32).to_le_bytes());
                }
            }
        }
        let mut storage = std::collections::HashMap::new();
        let mut storage_in_place = std::collections::HashMap::new();
        // Each storage binding's GPU range (ECHORA_DUMP_DISPATCH).
        let mut storage_ranges = Vec::new();
        let mut written_back = Vec::new();
        let mut uniforms = std::collections::HashMap::new();
        let mut textures = std::collections::HashMap::new();
        let mut images = std::collections::HashMap::new();
        // The view formats images are written through (see
        // `Dispatch::image_formats`).
        let mut image_formats = std::collections::HashMap::new();
        for descriptor in &stage.descriptors {
            let entry = descriptor.cbuf_offset as usize;
            match descriptor.kind() {
                DescriptorKind::StorageBuffer => {
                    let native = descriptor.flags & 32 != 0;
                    let (gpu, size) = match slot(descriptor, DescriptorKind::StorageBuffer) {
                        // A raw pointer candidate (the program picks among
                        // them by address): one that maps nowhere binds
                        // zeros it never selects.
                        _ if native => match self.native_pointer(
                            core,
                            descriptor,
                            &c0,
                            constants.as_deref().unwrap_or_default(),
                        )? {
                            Some(found) => found,
                            None => {
                                storage.insert(descriptor.binding, vec![0; 16]);
                                self.gpu
                                    .as_mut()
                                    .unwrap()
                                    .stat("stat: compute native pointers unmapped", 1);
                                continue;
                            }
                        },
                        Some(slot) => {
                            let bind = self
                                .render
                                .state
                                .get(bind_key("BindStorageBuffer", Some(5), slot as u64))
                                .ok_or_else(|| {
                                    format!(
                                        "storage buffer 5/{slot} is not bound: {}",
                                        layout(self)
                                    )
                                })?;
                            (bind.x[3], bind.x[4])
                        }
                        // Global memory through an address and size kept
                        // elsewhere in c0: a uniform buffer's entry or the
                        // pushed constants (OoT's post-movie passes).
                        None => {
                            let pointer = constant_pointer(descriptor, &c0);
                            let (gpu, size) = pointer
                                .filter(|&(gpu, size)| gpu != 0 && size != 0)
                                .ok_or_else(|| unplaced(self, descriptor))?;
                            self.gpu.as_mut().unwrap().note(format!(
                                "compute storage through c{}+{:#x}",
                                descriptor.cbuf_index, descriptor.cbuf_offset
                            ));
                            (gpu, size)
                        }
                    };
                    let base = gpu & !15;
                    let length = gpu - base + size;
                    // ECHORA_STORAGE_WRITE_LOG=<gpu start>-<end>[,...] or
                    // `all` (buffers up to 64 KiB) (debugging): each
                    // dispatch binding a storage buffer it writes over that
                    // range, with the present.
                    if let Ok(ranges) = super::debug_var("ECHORA_STORAGE_WRITE_LOG")
                        && descriptor.flags & 1 != 0
                        && super::debug_var("ECHORA_STORAGE_WRITE_LOG_FROM")
                            .ok()
                            .and_then(|value| value.parse::<u64>().ok())
                            .is_none_or(|from| self.presented_frames >= from)
                        && super::debug_var("ECHORA_STORAGE_WRITE_LOG_TO")
                            .ok()
                            .and_then(|value| value.parse::<u64>().ok())
                            .is_none_or(|to| self.presented_frames <= to)
                        && (ranges == "all" && size <= 0x10000
                            || ranges.split(',').any(|range| {
                            range.split_once('-').is_some_and(|(start, end)| {
                                let parse = |text: &str| {
                                    u64::from_str_radix(text.trim().trim_start_matches("0x"), 16).ok()
                                };
                                matches!((parse(start), parse(end)), (Some(start), Some(end)) if gpu < end && start < gpu + size)
                            })
                        }))
                    {
                        eprintln!(
                            "storage write present {} dispatch {:016x} binding {} gpu {gpu:#x}+{size:#x}",
                            self.presented_frames, code_key, descriptor.binding
                        );
                    }
                    if descriptor.cbuf_index == 0 && !native {
                        c0[entry..entry + 8].copy_from_slice(&gpu.to_le_bytes());
                        c0[entry + 8..entry + 12].copy_from_slice(&(size as u32).to_le_bytes());
                    }
                    let guest = self.gpu_to_guest(base, length)?;
                    storage_ranges.push((descriptor.binding, base, length));
                    // ECHORA_GPU_STORAGE_PROBE=<program key> (debugging): each
                    // storage buffer that program binds, its flags and the
                    // path it takes, at every 300th present.
                    let probing = super::debug_var("ECHORA_GPU_STORAGE_PROBE")
                        .is_ok_and(|probe| probe == format!("{:016x}", code_key))
                        && self.presented_frames.is_multiple_of(300);
                    if probing {
                        eprintln!(
                            "probe present {} {:016x} binding {} c{}+{:#x} gpu {gpu:#x} size {size:#x} flags {:#x}",
                            self.presented_frames,
                            code_key,
                            descriptor.binding,
                            descriptor.cbuf_index,
                            descriptor.cbuf_offset,
                            descriptor.flags
                        );
                    }
                    // In place when guest memory can be imported: the GPU
                    // reads and writes it directly, with no copies either
                    // way (earlier in-place writes are already ordered on
                    // the GPU).
                    let in_place = match core.memory.host_span(guest, length as usize) {
                        Some(pointer) => self
                            .gpu
                            .as_mut()
                            .unwrap()
                            .renderer
                            .import_host(pointer, length)?,
                        None => None,
                    };
                    if let Some((buffer, offset)) = in_place {
                        if probing {
                            eprintln!("probe   in place");
                        }
                        self.resolve_copies(core, Some(guest..guest + length))?;
                        storage_in_place.insert(descriptor.binding, (buffer, offset, length));
                        let written = &mut self.gpu.as_mut().unwrap().in_place_writes;
                        if descriptor.flags & 1 != 0 && !written.contains(&(guest..guest + length))
                        {
                            written.push(guest..guest + length);
                            self.gpu.as_mut().unwrap().gpu_writes += 1;
                        }
                        continue;
                    }
                    // Otherwise the pool holding it stays on the GPU between
                    // dispatches (see `ResidentPool`); outside any pool the
                    // bytes are copied in and out. Copied results still
                    // pending for the range land first, so the pool's copy
                    // takes them (a counter an earlier pass wrote).
                    self.resolve_copies(core, Some(guest..guest + length))?;
                    match self.resident_pool_binding(core, guest, length)? {
                        Some((buffer, offset)) => {
                            if probing {
                                eprintln!("probe   resident at {offset:#x}");
                            }
                            storage_in_place.insert(descriptor.binding, (buffer, offset, length));
                            if descriptor.flags & 1 != 0 {
                                let backend = self.gpu.as_mut().unwrap();
                                if let Some(pool) = backend
                                    .resident
                                    .values_mut()
                                    .find(|pool| pool.contains(guest))
                                {
                                    add_range(&mut pool.gpu_dirty, guest..guest + length);
                                    add_range(
                                        &mut backend.dirty_index.get_mut().1,
                                        guest..guest + length,
                                    );
                                    backend.gpu_writes += 1;
                                }
                            }
                        }
                        None => {
                            if probing {
                                eprintln!("probe   copied");
                            }
                            self.resolve_storage(core, Some(guest..guest + length))?;
                            storage.insert(descriptor.binding, self.gpu_bytes(core, base, length)?);
                            if descriptor.flags & 1 != 0 {
                                written_back.push((descriptor.binding, base));
                            }
                        }
                    }
                }
                // Texel buffers (buffer textures) bind like textures.
                kind @ (DescriptorKind::Texture | DescriptorKind::TextureBuffer) => {
                    let slot = slot(descriptor, kind).ok_or_else(|| unplaced(self, descriptor))?;
                    let &handle = self
                        .render
                        .bound_textures
                        .get(5, u64::from(slot))
                        .ok_or_else(|| {
                            format!("texture 5/{slot} is not bound: {}", layout(self))
                        })?;
                    c0[entry..entry + 8].copy_from_slice(&handle.to_le_bytes());
                    let texture = self.handle_texture(handle)?;
                    let started = std::time::Instant::now();
                    let (source, _) = self.resolve_sampled_texture(core, texture)?;
                    let source = self.through_view(source, handle);
                    if let Some(swizzle) = self.handle_swizzle(handle) {
                        if super::debug_var_os("ECHORA_GPU_FLOW").is_some() {
                            let note = format!("flow view swizzle {swizzle:?} on {texture:x}");
                            self.gpu_note(note);
                        }
                    }
                    self.gpu.as_mut().unwrap().stat(
                        "stat: dispatch texture microseconds",
                        started.elapsed().as_micros() as u64,
                    );
                    let sampler = self.binding_sampler(handle, texture, &source);
                    textures.insert(descriptor.binding, (source, sampler));
                }
                // Images a compute program writes become render targets, so
                // later draws sample what it wrote.
                DescriptorKind::Image => {
                    let slot = slot(descriptor, DescriptorKind::Image)
                        .ok_or_else(|| unplaced(self, descriptor))?;
                    let handle = self
                        .render
                        .state
                        .get(bind_key("BindImage", Some(5), slot as u64))
                        .ok_or_else(|| format!("image 5/{slot} is not bound: {}", layout(self)))?
                        .x[3];
                    c0[entry..entry + 8].copy_from_slice(&handle.to_le_bytes());
                    let texture = self.handle_texture(handle)?;
                    let (width, height, format) = self.target_shape(texture)?;
                    let levels = self.texture_levels(texture);
                    // 3D textures (colour lookup tables a pass blends) are
                    // written as volumes.
                    let builder = &self.object(texture, "Texture")?.builder;
                    let shape = if builder.get("Target").map(|value| value.x[0]) == Some(2) {
                        ImageShape::D3 {
                            depth: builder
                                .get("Depth")
                                .map_or(1, |value| value.x[0] as u32)
                                .max(1),
                        }
                    } else {
                        ImageShape::D2 { layers: 1 }
                    };
                    self.gpu
                        .as_mut()
                        .unwrap()
                        .renderer
                        .guest_sized_target(texture, width, height, shape, format, levels)?;
                    if matches!(shape, ImageShape::D3 { .. })
                        && super::debug_var_os("ECHORA_GPU_FLOW").is_some()
                    {
                        eprintln!(
                            "present {}: dispatch writes 3D image {texture:x} {width}x{height} {shape:?}",
                            self.presented_frames
                        );
                    }
                    let level = if super::mip_targets() {
                        self.handle_view_levels(handle).map_or(0, |(base, _)| base)
                    } else {
                        0
                    };
                    if self.depth_traced(texture) {
                        eprintln!(
                            "depth trace present {} dispatch writes {texture:x} (binding {})",
                            self.presented_frames, descriptor.binding
                        );
                    }
                    images.insert(descriptor.binding, (texture, level));
                    if let Some(view) = self
                        .handle_view_format(handle)
                        .filter(|view| echora_gpu::renderer::uint_alias(format) == Some(*view))
                    {
                        image_formats.insert(descriptor.binding, view);
                    }
                }
                DescriptorKind::UniformBuffer if descriptor.index >= 3 => {
                    let slot = descriptor.index - 3;
                    let bind = self
                        .render
                        .bound_uniforms
                        .get(5, u64::from(slot))
                        .ok_or_else(|| format!("uniform buffer 5/{slot} is not bound"))?;
                    let used = u64::from(stage.cbuf_used_sizes[descriptor.index as usize]);
                    let size = bind.x[4].min(used.max(16)).min(64 << 10);
                    let address = bind.x[3];
                    let guest = self.gpu_to_guest(address, size)?;
                    self.resolve_storage(core, Some(guest..guest + size))?;
                    uniforms.insert(descriptor.binding, self.gpu_bytes(core, address, size)?);
                }
                DescriptorKind::UniformBuffer if descriptor.index == 0 => {}
                DescriptorKind::UniformBuffer if descriptor.index == 1 => {
                    // Only the constants themselves: reads past them return
                    // zeros (robust buffer access).
                    let mut bytes = constants.clone().unwrap_or_default();
                    bytes.resize(bytes.len().max(16).next_multiple_of(16), 0);
                    uniforms.insert(descriptor.binding, bytes);
                }
                DescriptorKind::UniformBuffer => {
                    let used = stage.cbuf_used_sizes[descriptor.index as usize].max(16);
                    self.gpu.as_mut().unwrap().note(format!(
                        "driver constant buffer {} read as zeros",
                        descriptor.index
                    ));
                    uniforms.insert(descriptor.binding, vec![0; used as usize]);
                }
                _ => return Err(unplaced(self, descriptor)),
            }
        }
        self.initialize_new_targets(core);
        // c0 itself, once every handle in it is known.
        for descriptor in &stage.descriptors {
            if descriptor.kind() == DescriptorKind::UniformBuffer && descriptor.index == 0 {
                let used = (stage.cbuf_used_sizes[0] as usize).max(16);
                let mut bytes = c0.clone();
                self.graphics_addons
                    .apply_compute_constants(code_key, &mut bytes);
                bytes.resize(used.max(bytes.len()), 0);
                uniforms.insert(descriptor.binding, bytes);
            }
        }
        if super::debug_var_os("ECHORA_GPU_FLOW").is_some() {
            let line = format!(
                "flow dispatch {:016x} reads {} writes {} storage {} flags {:?}",
                code_key,
                self.describe_sources(textures.values().map(|(source, _)| source)),
                self.describe_targets(images.values().map(|(key, _)| *key)),
                storage.len(),
                stage
                    .descriptors
                    .iter()
                    .filter(|descriptor| descriptor.kind() == DescriptorKind::StorageBuffer)
                    .map(|descriptor| descriptor.flags)
                    .collect::<Vec<_>>()
            );
            self.sequence_note(&line);
            self.gpu.as_mut().unwrap().note(line);
            // The uniform words the program sees (its histogram range, say).
            let mut bindings: Vec<_> = uniforms.iter().collect();
            bindings.sort_by_key(|(binding, _)| **binding);
            for (binding, bytes) in bindings {
                let words: Vec<String> = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .take(48)
                    .map(|word| {
                        let bits = u32::from_le_bytes(*word);
                        format!("{bits:x}/{}", f32::from_bits(bits))
                    })
                    .collect();
                let line = format!(
                    "flow dispatch {:016x} uniform {binding} [{}]",
                    code_key,
                    words.join(" ")
                );
                self.sequence_note(&line);
            }
        }
        let image_keys: Vec<u64> = images.values().map(|(key, _)| *key).collect();
        // ECHORA_DISPATCH_SOURCE_LOG=<program key hex>[,...] (debugging):
        // each dispatch of those programs logs the textures it reads, with
        // the program and present that last wrote each, and its images.
        if let Ok(wanted) = super::debug_var("ECHORA_DISPATCH_SOURCE_LOG") {
            if wanted
                .split(',')
                .any(|item| item.trim() == format!("{:016x}", code_key))
            {
                let mut bound: Vec<_> = textures.iter().collect();
                bound.sort_by_key(|(binding, _)| **binding);
                let reads: Vec<String> =
                    bound
                        .iter()
                        .map(|(binding, (source, _))| {
                            let writer = match source {
                            TextureSource::Target(target)
                            | TextureSource::TargetView { key: target, .. } => self
                                .target_writers
                                .get(target)
                                .map_or("never written".to_string(), |(program, present, storage)| {
                                    format!("written by {program:016x} at {present} to {storage:x}")
                                }),
                            _ => "upload".to_string(),
                        };
                            format!(
                                "{binding}={} ({writer})",
                                self.describe_sources(std::iter::once(source))
                            )
                        })
                        .collect();
                eprintln!(
                    "dispatch sources present {} {:016x}: reads {} writes {}",
                    self.presented_frames,
                    code_key,
                    reads.join(", "),
                    self.describe_targets(image_keys.iter().copied())
                );
            }
            for target in &image_keys {
                let storage = self.texture_storage(*target).unwrap_or(0);
                self.target_writers
                    .insert(*target, (code_key, self.presented_frames, storage));
            }
        }
        let images_written = image_keys.clone();
        let volume_mean = |backend: &mut Backend, key: u64| -> Option<f64> {
            let is_volume = backend
                .renderer
                .existing_target(key)
                .is_some_and(|target| matches!(target.shape, ImageShape::D3 { .. }));
            let texels = backend
                .renderer
                .read_target(key)
                .ok()
                .filter(|_| is_volume)?;
            Some(
                texels.iter().map(|texel| f64::from(texel[0])).sum::<f64>()
                    / texels.len().max(1) as f64,
            )
        };
        let before: Vec<Option<f64>> = if super::debug_var_os("ECHORA_GPU_FLOW").is_some() {
            let backend = self.gpu.as_mut().unwrap();
            image_keys
                .iter()
                .map(|key| volume_mean(backend, *key))
                .collect()
        } else {
            Vec::new()
        };
        let uniform_summary: Vec<(u32, Vec<f32>)> = uniforms
            .iter()
            .map(|(binding, bytes)| {
                let floats = bytes.as_chunks::<4>().0.iter().take(8);
                (
                    *binding,
                    floats.map(|word| f32::from_le_bytes(*word)).collect(),
                )
            })
            .collect();
        // ECHORA_DISPATCH_UNIFORM_DUMP=<program key hex>[,...] with
        // ECHORA_DUMP_DIR (debugging): each dispatch of those programs
        // writes its uniform buffers as uniforms-<key>-<present>-<n>.bin
        // (n counting dispatches within the present), with
        // ECHORA_DUMP_DISPATCH_FROM as the first present.
        if let (Ok(wanted), Some(directory)) = (
            super::debug_var("ECHORA_DISPATCH_UNIFORM_DUMP"),
            super::debug_var_os("ECHORA_DUMP_DIR"),
        ) && wanted
            .split(',')
            .any(|item| item.trim() == format!("{:016x}", code_key))
            && super::debug_var("ECHORA_DUMP_DISPATCH_FROM")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .is_none_or(|from| self.presented_frames >= from)
        {
            let directory = std::path::Path::new(&directory);
            let _ = std::fs::create_dir_all(directory);
            let mut index = 0;
            while directory
                .join(format!(
                    "uniforms-{:016x}-{}-{index}-0.bin",
                    code_key, self.presented_frames
                ))
                .exists()
            {
                index += 1;
            }
            for (binding, bytes) in &uniforms {
                let _ = std::fs::write(
                    directory.join(format!(
                        "uniforms-{:016x}-{}-{index}-{binding}.bin",
                        code_key, self.presented_frames
                    )),
                    bytes,
                );
            }
            // An empty marker so the index above finds this dispatch.
            let _ = std::fs::write(
                directory.join(format!(
                    "uniforms-{:016x}-{}-{index}-0.bin",
                    code_key, self.presented_frames
                )),
                uniforms.get(&0).map_or(&[][..], |bytes| &bytes[..]),
            );
        }
        // DispatchComputeIndirect(cmd, NVNbufferAddress): the GPU reads the
        // three group counts at the address when the dispatch runs, from
        // the pool's GPU copy (an earlier pass's writes, a counter its
        // atomics advanced, are there in order).
        let indirect = if command.name.ends_with("DispatchComputeIndirect") {
            let guest = self.gpu_to_guest(command.x[1], 12)?;
            Some(
                self.indirect_arguments(core, guest, 12)?
                    .ok_or("indirect dispatch arguments are in no memory pool")?,
            )
        } else {
            None
        };
        self.graphics_addons.apply_compute_textures(
            code_key,
            &stage,
            &mut self.gpu.as_mut().unwrap().renderer,
            &mut textures,
        )?;
        let sampled_textures: Vec<_> = stage
            .descriptors
            .iter()
            .filter_map(|descriptor| {
                let (source, _) = textures.get(&descriptor.binding)?;
                let format = self.gpu.as_ref()?.renderer.source_format(source)?;
                let sample_type = echora_gpu::renderer::sampled_texture_type(format);
                (sample_type != shader::SampleType::Float).then_some(shader::SampledTexture {
                    bank: descriptor.cbuf_index,
                    offset: descriptor.cbuf_offset,
                    view: descriptor.texture_type,
                    sample_type,
                })
            })
            .collect();
        if !sampled_textures.is_empty() {
            let (typed, _) = self.translate_stage(
                core,
                address,
                [0; 3],
                None,
                [0; 32],
                compute,
                None,
                constants.as_deref(),
                &sampled_textures,
            )?;
            if typed.descriptors != stage.descriptors {
                return Err("integer texture formats changed the compute descriptor layout".into());
            }
            stage = typed;
        }
        // The program, specialized for what is bound: an IntegerFormat
        // constant is 1 when its binding's image or texture is integer. Each
        // combination has its own program for this exact translation.
        let backend = self.gpu.as_mut().unwrap();
        let mut specialization = Vec::new();
        for constant in &stage.spec_constants {
            let binding = match constant.meaning {
                echora_shader_abi::SpecMeaning::IntegerFormat(binding) => binding,
                // The renderer gives every program the pool's capacity.
                echora_shader_abi::SpecMeaning::TexturePoolCapacity => continue,
                // ECHORA_LOOP_TRACE: the program's counter buffer, 64 KiB.
                meaning @ (echora_shader_abi::SpecMeaning::LoopTraceLow
                | echora_shader_abi::SpecMeaning::LoopTraceHigh) => {
                    let address = match backend.loop_traces.get(&code_key) {
                        Some(&(_, address, _)) => address,
                        None => {
                            let (index, address) = backend.renderer.debug_buffer(64 << 10)?;
                            backend
                                .renderer
                                .mark_loop_trace(index, format!("dispatch {:016x}", code_key));
                            backend.loop_traces.insert(code_key, (index, address, 0));
                            address
                        }
                    };
                    let half = if meaning == echora_shader_abi::SpecMeaning::LoopTraceLow {
                        address as u32
                    } else {
                        (address >> 32) as u32
                    };
                    specialization.push((constant.id, half));
                    continue;
                }
                other => return Err(format!("compute spec constant {other:?}")),
            };
            let format = match images.get(&binding) {
                Some((texture, _)) => image_formats.get(&binding).copied().or_else(|| {
                    backend
                        .renderer
                        .existing_target(*texture)
                        .map(|target| target.format)
                }),
                None => textures
                    .get(&binding)
                    .and_then(|(source, _)| backend.renderer.source_format(source)),
            };
            let integer = format.is_some_and(echora_gpu::renderer::integer_format);
            specialization.push((constant.id, u32::from(integer)));
        }
        if specialization.len() > 32 {
            return Err(format!("{} spec constants", specialization.len()));
        }
        let specialization_bits = specialization
            .iter()
            .enumerate()
            .fold(0u64, |bits, (index, (_, value))| {
                bits | u64::from(*value != 0) << index
            });
        let program_key = (Arc::as_ptr(&stage) as usize, specialization_bits);
        if !backend.compute_programs.contains_key(&program_key) {
            // ECHORA_SHADER_STATS (debugging): which program compiles next,
            // for the driver's statistics (RADV_DEBUG=shaderstats) that follow.
            if super::debug_var_os("ECHORA_SHADER_STATS").is_some() {
                eprintln!("shader stats: dispatch {:016x}", code_key);
            }
            let program = backend
                .renderer
                .create_program_specialized(vec![stage.clone()], &[specialization])?;
            backend.compute_programs.insert(program_key, program);
        }
        // Every 60 presents, a traced program's loops that ran: for loop k,
        // total iterations, the most one invocation made, invocations the
        // budget stopped, and the header's guest offset.
        if let Some(&(index, address, reported)) = backend.loop_traces.get(&code_key)
            && self.presented_frames >= reported + 60
        {
            let words: Vec<u32> = backend
                .renderer
                .read_debug_buffer(index)
                .chunks_exact(4)
                .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
                .collect();
            for (loop_index, entry) in words.chunks_exact(4).enumerate() {
                if entry[0] != 0 {
                    eprintln!(
                        "loop trace present {} program {:016x} loop {loop_index} at {:#x}: {} iterations, at most {} per invocation, {} stopped by the budget",
                        self.presented_frames,
                        code_key,
                        entry[3].wrapping_sub(1),
                        entry[0],
                        entry[1],
                        entry[2]
                    );
                }
            }
            backend
                .loop_traces
                .insert(code_key, (index, address, self.presented_frames));
        }
        // The GPU page table, for programs that read global memory through
        // it.
        let page_table = if self.gpu.as_ref().unwrap().compute_programs[&program_key]
            .stages
            .iter()
            .any(|stage| stage.push_fields & echora_shader_abi::PUSH_PAGE_TABLE != 0)
        {
            self.page_table(core)?
        } else {
            Default::default()
        };
        // Only a program that stores through the table makes later CPU
        // reads wait (one that only reads it, the translator says, does not).
        let writes_page_table = page_table.address != 0
            && self.gpu.as_ref().unwrap().compute_programs[&program_key]
                .stages
                .iter()
                .any(|stage| {
                    stage.push_fields & echora_shader_abi::PUSH_PAGE_TABLE != 0
                        && stage.push_fields & echora_shader_abi::PUSH_PAGE_TABLE_READ_ONLY == 0
                });
        let pool = if stage.texture_pool_bindings != 0 {
            let additional = self.shader_pool_handles(
                core,
                &stage,
                |_, bank, offset| {
                    let descriptor = stage.descriptors.iter().find(|descriptor| {
                        descriptor.kind() == DescriptorKind::UniformBuffer
                            && descriptor.index == u32::from(bank)
                    })?;
                    shader_word(uniforms.get(&descriptor.binding)?, offset)
                },
                |address, length| {
                    c0[storage_base as usize..texture_base as usize]
                        .chunks_exact(16)
                        .any(|entry| {
                            let base = u64::from_le_bytes(entry[..8].try_into().unwrap());
                            let size = u32::from_le_bytes(entry[8..12].try_into().unwrap());
                            storage_contains(base, u64::from(size), address, length)
                        })
                },
            );
            self.pool_entries(core, 5, &additional)?
        } else {
            Default::default()
        };
        // ECHORA_DUMP_DISPATCH=<program key>[,...] (debugging): the first
        // dispatch of each listed program, for the workgroup simulator
        // (`dump_dispatch`).
        if let Some(keys) = super::debug_var_os("ECHORA_DUMP_DISPATCH") {
            let wanted = keys
                .to_string_lossy()
                .split(',')
                .any(|item| u64::from_str_radix(item.trim(), 16) == Ok(code_key));
            // ECHORA_DUMP_DISPATCH_FROM=N: from present N on (a temporal
            // pass's live frame, not its first run).
            let from = super::debug_var("ECHORA_DUMP_DISPATCH_FROM")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(0);
            if wanted && self.presented_frames >= from && self.dumped_dispatches.insert(code_key) {
                let groups = [command.x[1], command.x[2], command.x[3]].map(|count| count as u32);
                if let Err(error) = self.dump_dispatch(
                    core,
                    code_key,
                    &stage,
                    groups,
                    &uniforms,
                    &storage_ranges,
                    page_table,
                ) {
                    eprintln!("dispatch dump {:016x}: {error}", code_key);
                    self.gpu_note(format!("dispatch dump {:016x}: {error}", code_key));
                }
                let read: Vec<(u32, TextureSource)> =
                    textures
                        .iter()
                        .map(|(binding, (source, _))| (*binding, source.clone()))
                        .chain(images.iter().map(|(binding, (target, _))| {
                            (*binding, TextureSource::Target(*target))
                        }))
                        .collect();
                eprintln!(
                    "dispatch dump {:016x}: {} textures and images to read",
                    code_key,
                    read.len()
                );
                if let Err(error) = self.dump_images(code_key, &read) {
                    eprintln!("dispatch dump {:016x} images: {error}", code_key);
                    self.gpu_note(format!("dispatch dump {:016x} images: {error}", code_key));
                }
            }
            // ECHORA_DUMP_ONLY: listed programs are dumped, never run.
            if wanted && super::debug_var_os("ECHORA_DUMP_ONLY").is_some() {
                return Ok(());
            }
        }
        let groups = [command.x[1], command.x[2], command.x[3]].map(|count| count as u32);
        let bounded = if writes_page_table {
            self.compute_shared_writes(
                &stage,
                indirect.is_none().then_some(groups),
                |bank, offset| {
                    let descriptor = stage.descriptors.iter().find(|descriptor| {
                        descriptor.kind() == DescriptorKind::UniformBuffer
                            && descriptor.index == u32::from(bank)
                    })?;
                    shader_word(uniforms.get(&descriptor.binding)?, usize::from(offset))
                },
            )
        } else {
            None
        };
        let backend = self.gpu.as_mut().unwrap();
        let dispatch = echora_gpu::compute::Dispatch {
            program: &backend.compute_programs[&program_key],
            groups,
            uniforms,
            storage,
            storage_in_place,
            textures,
            images,
            image_formats,
            indirect,
            page_table,
            pool,
        };
        let started = std::time::Instant::now();
        let lengths: HashMap<u32, u64> = dispatch
            .storage
            .iter()
            .map(|(binding, bytes)| (*binding, bytes.len() as u64))
            .collect();
        if echora_gpu::renderer::Renderer::labelling() {
            let label = format!(
                "present {} dispatch {:016x} groups {} reads [{}] writes {:x?}",
                self.presented_frames,
                code_key,
                match dispatch.indirect {
                    Some(_) => format!("indirect at {:#x}", command.x[1]),
                    None => format!("{:?}", dispatch.groups),
                },
                label_sources(dispatch.textures.values().map(|(source, _)| source)),
                dispatch.images.values().collect::<Vec<_>>()
            );
            backend.renderer.label(label);
        }
        let tokens: HashMap<u32, u64> = backend
            .renderer
            .dispatch_deferred(&dispatch)?
            .into_iter()
            .collect();
        if writes_page_table {
            let serial = backend.renderer.recorded_serial();
            if bounded
                .as_ref()
                .is_some_and(|writes| !writes.virtual_ranges.is_empty())
            {
                backend.stat("stat: bounded uniform writes", 1);
            }
            backend.record_page_table_write(serial, bounded);
        }
        backend.stat(
            "stat: dispatch renderer microseconds",
            started.elapsed().as_micros() as u64,
        );
        if super::debug_var_os("ECHORA_TARGET_TRACE").is_some()
            || super::debug_var_os("ECHORA_MEMORY_TRACE").is_some()
        {
            let label = format!("dispatch {:016x}", code_key);
            self.trace_targets(&images_written, &[], &label)?;
            if let Err(error) = self.trace_memory(core, &label) {
                eprintln!("memory trace: {error}");
            }
        }
        let backend = self.gpu.as_mut().unwrap();
        if super::debug_var_os("ECHORA_GPU_FLOW").is_some() {
            // Volumes a dispatch wrote, as their first slice reads back.
            for (key, before) in image_keys.into_iter().zip(before) {
                if let Some(after) = volume_mean(backend, key) {
                    eprintln!(
                        "dispatch wrote volume {key:x}: slice 0 mean red {before:?} -> {after:.4}, uniforms {uniform_summary:?}"
                    );
                }
            }
        }
        for (binding, base) in written_back {
            let (Some(token), Some(length)) = (tokens.get(&binding), lengths.get(&binding)) else {
                continue;
            };
            let guest = self.gpu_to_guest(base, *length)?;
            self.gpu.as_mut().unwrap().gpu_writes += 1;
            self.gpu.as_mut().unwrap().pending_storage.push((
                *token,
                guest..guest + length,
                binding,
                code_key,
                base,
            ));
        }
        Ok(())
    }

    /// The current GPU write generation (`gpu_writes`).
    fn gpu_writes_now(&self) -> u64 {
        self.gpu.as_ref().map_or(0, |backend| backend.gpu_writes)
    }

    /// A memoized binding of `handle`, unless GPU writes recorded since it
    /// was last checked lie under its texture (it resolves anew then).
    fn memoized_binding(
        &mut self,
        handle: u64,
    ) -> Option<(echora_gpu::draw::TextureSource, echora_gpu::draw::Sampler)> {
        self.with_memoized_binding(handle, |entry| (entry.0.clone(), entry.1))
    }

    fn with_memoized_binding<T>(
        &mut self,
        handle: u64,
        read: impl FnOnce(&super::ResolvedBinding) -> T,
    ) -> Option<T> {
        let now = self.gpu_writes_now();
        let frame = self.presented_frames;
        let entry = self.binding_memo.1.get(&handle)?;
        // The common case: checked at this present and GPU-write generation.
        if entry.4 == frame && entry.3 == now {
            return Some(read(entry));
        }
        // From an earlier present: a render target's binding holds while
        // the target set is unchanged (the stamp) and no blocks drawn since
        // wait to be copied into it (`texture_source` brings those in); an
        // upload is checked against guest memory once per present, so it
        // is resolved again.
        if entry.4 != frame {
            let current = match &entry.0 {
                echora_gpu::draw::TextureSource::Target(key)
                | echora_gpu::draw::TextureSource::TargetView { key, .. } => {
                    !self.block_aliases.contains_key(key)
                }
                _ => false,
            };
            if !current {
                return None;
            }
        }
        if entry.3 != now
            && entry.2.as_ref().is_some_and(|backing| {
                backing
                    .pieces()
                    .iter()
                    .any(|(_, range)| self.gpu_writes_pending(range))
            })
        {
            return None;
        }
        let entry = self.binding_memo.1.get_mut(&handle)?;
        entry.3 = now;
        entry.4 = frame;
        Some(read(entry))
    }

    /// Drop the memoized bindings that targets made, remade or moved since
    /// (the renderer's keys) and textures moved or finalized
    /// (`binding_evictions`) may change: those sampling such a target, of
    /// such a texture, or over its storage (another object's binding there
    /// may now resolve to a render target). The other bindings stay.
    fn evict_bindings(&mut self) {
        let targets = self
            .gpu
            .as_mut()
            .map(|backend| {
                let targets = backend.renderer.take_changed_targets();
                if backend
                    .target_index
                    .stamp
                    .is_some_and(|stamp| stamp != backend.renderer.target_changes)
                {
                    backend.target_index.changed.extend(targets.iter().copied());
                }
                targets
            })
            .unwrap_or_default();
        if targets.is_empty() && self.binding_evictions.is_empty() {
            return;
        }
        let mut keys: super::FastSet<u64> = super::FastSet::default();
        let mut bases: super::FastSet<u64> = super::FastSet::default();
        for key in targets {
            keys.insert(key);
            if let Some(base) = self.texture_storage(key) {
                bases.insert(base);
            }
        }
        for (texture, storages) in std::mem::take(&mut self.binding_evictions) {
            keys.insert(texture);
            bases.extend(storages.into_iter().filter(|base| *base != 0));
        }
        let stale = |entry: &super::ResolvedBinding| {
            let source_key = match &entry.0 {
                TextureSource::Target(key) | TextureSource::TargetView { key, .. } => Some(key),
                _ => None,
            };
            source_key.is_some_and(|key| keys.contains(&key))
                || keys.contains(&entry.5)
                || entry.2.as_ref().is_some_and(|backing| {
                    backing
                        .pieces()
                        .iter()
                        .any(|(_, range)| bases.iter().any(|base| range.contains(base)))
                })
        };
        let before = self.binding_memo.1.len();
        self.binding_memo.1.retain(|_, entry| !stale(entry));
        let mut evicted = before != self.binding_memo.1.len();
        for (_, memo) in self.queue_memos.values_mut() {
            let before = memo.len();
            memo.retain(|_, entry| !stale(entry));
            evicted |= before != memo.len();
        }
        if evicted {
            // Pool entry lists made from them are made again.
            self.pool_inputs += 1;
            self.gpu
                .as_mut()
                .unwrap()
                .stat("stat: binding memo evictions", 1);
        }
    }

    /// Whether GPU writes not yet in guest memory overlap `range` (what
    /// `resolve_storage` would bring back there).
    fn gpu_writes_pending(&self, range: &std::ops::Range<u64>) -> bool {
        let Some(backend) = self.gpu.as_ref() else {
            return false;
        };
        let overlaps =
            |written: &std::ops::Range<u64>| written.start < range.end && range.start < written.end;
        if backend.in_place_writes.iter().any(overlaps)
            || backend
                .pending_storage
                .iter()
                .any(|(_, guest, ..)| overlaps(guest))
        {
            return true;
        }
        backend.resident_writes_pending(range)
    }

    /// Make a mapped rendering input current before comparing it on the
    /// CPU or copying it into the upload ring.
    pub(super) fn resolve_draw_input(
        &mut self,
        core: &mut Guest,
        guest: u64,
        size: u64,
    ) -> Result<(), String> {
        if size == 0 {
            return Ok(());
        }
        let range = guest
            ..guest
                .checked_add(size)
                .ok_or("guest draw input overflows")?;
        let serial = self.gpu.as_ref().and_then(|backend| {
            if !backend.page_table_writes {
                return None;
            }
            self.shared_guest_index()?
                .overlapping(range.clone())
                .map(|span| backend.shared_write_serial(span))
                .max()
        });
        if let Some(serial) = serial {
            let backend = self.gpu.as_mut().unwrap();
            backend.renderer.wait_for(serial)?;
            if backend.renderer.completed_serial() >= backend.page_table_write_serial {
                backend.page_table_writes = false;
            }
        }
        self.resolve_storage(core, Some(range))
    }

    /// A queued consumer sees preceding GPU writes whether or not a
    /// fence has exposed those writes to guest CPU calls.
    pub(super) fn resolve_queued_buffer_read(
        &mut self,
        core: &mut Guest,
        address: u64,
        guest: u64,
        length: u64,
    ) -> Result<(), String> {
        if length == 0 || self.gpu.is_none() {
            return Ok(());
        }
        let end = guest
            .checked_add(length)
            .ok_or("guest buffer range overflows")?;
        let shared = self.shared_table_range(address, guest, length);
        self.gpu.as_mut().unwrap().wait_for_table_writes(shared)?;
        let range = guest..end;
        if shared.is_some() {
            self.resolve_storage(core, Some(range))
        } else {
            self.resolve_gpu_written(core, range)
        }
    }

    pub(super) fn resolve_storage(
        &mut self,
        core: &mut Guest,
        range: Option<std::ops::Range<u64>>,
    ) -> Result<(), String> {
        // Writes made in place reach guest memory when the GPU finishes.
        if let Some(backend) = self.gpu.as_mut() {
            let waits = match &range {
                Some(range) => backend
                    .in_place_writes
                    .iter()
                    .any(|written| written.start < range.end && range.start < written.end),
                None => !backend.in_place_writes.is_empty(),
            };
            if waits {
                backend.stat("stat: GPU waits for in-place writes", 1);
                backend.renderer.flush()?;
                let mut textures_changed = false;
                for written in std::mem::take(&mut backend.in_place_writes) {
                    textures_changed |= backend.mark_texture_bytes_changed(&written);
                }
                if textures_changed {
                    self.bump_binding_serial("stat: binding serial by in-place texture writes");
                }
            }
        }
        self.resolve_copies(core, range.clone())?;
        self.resolve_resident(core, range)
    }

    /// A guest range the GPU may have written through the page table (a
    /// dispatch's arguments a compute program wrote): after such a dispatch
    /// the work is finished and, where a resident pool holds the range, its
    /// GPU copy is read back; then `resolve_storage` as usual.
    #[track_caller]
    pub(super) fn resolve_gpu_written(
        &mut self,
        core: &mut Guest,
        range: std::ops::Range<u64>,
    ) -> Result<(), String> {
        if let Some(backend) = self.gpu.as_mut()
            && backend.page_table_wrote
        {
            // Work storing through the table recorded since the last wait
            // finishes first; once it has, later reads need no wait until
            // more such work is recorded.
            if backend.page_table_writes {
                let started = std::time::Instant::now();
                let waited_serial = backend
                    .page_table_write_serial
                    .min(backend.observable_serial);
                if crate::timeline::on() {
                    crate::timeline::span(
                        u64::MAX - 3,
                        &format!(
                            "page-table-wait serial {waited_serial} recorded {} completed {}",
                            backend.renderer.recorded_serial(),
                            backend.renderer.completed_serial()
                        ),
                        started,
                        started,
                    );
                }
                // A marker at the last signal the GPU has passed already
                // covers what is waited for up to it.
                let marked = backend.observable_marker.is_some_and(|marker| {
                    backend.renderer.marker_serial(marker) == Some(backend.observable_serial)
                        && backend.renderer.marker_done(marker)
                });
                if backend.page_table_write_serial <= backend.observable_serial {
                    backend.stat("stat: GPU waits for page-table writes", 1);
                    if marked && backend.page_table_write_serial == backend.observable_serial {
                        backend.page_table_writes = backend.page_table_wrote_after_signal
                            && backend.renderer.completed_serial()
                                < backend.page_table_write_serial;
                    } else {
                        backend.renderer.wait_for(backend.page_table_write_serial)?;
                        backend.page_table_writes = false;
                    }
                } else {
                    // Stores recorded after the last signal the guest saw
                    // are not waited for; the ones before it are.
                    backend.stat("stat: GPU waits for page-table writes up to a signal", 1);
                    if !marked {
                        backend.renderer.wait_for(backend.observable_serial)?;
                    }
                }
                backend.stat(
                    "stat: GPU waits for page-table writes us",
                    started.elapsed().as_micros() as u64,
                );
                crate::timeline::span(
                    u64::MAX - 3,
                    "page-table-wait",
                    started,
                    std::time::Instant::now(),
                );
            }
            for pool in backend.resident.values_mut() {
                let piece = range.start.max(pool.base)..range.end.min(pool.base + pool.size);
                if piece.start < piece.end {
                    // Ranges the CPU wrote since (flushed, not yet uploaded)
                    // are newer in guest memory: the mirror's bytes there
                    // are older and stay on the GPU.
                    let mut pieces = vec![piece];
                    for newer in &pool.guest_dirty {
                        pieces = pieces
                            .into_iter()
                            .flat_map(|piece| {
                                let mut left = vec![piece.clone()];
                                remove_range(&mut left, newer.clone());
                                left
                            })
                            .collect();
                    }
                    for piece in pieces {
                        add_range(&mut pool.gpu_dirty, piece.clone());
                        add_range(&mut backend.dirty_index.get_mut().1, piece);
                    }
                    backend.gpu_writes += 1;
                }
            }
        }
        self.resolve_storage(core, Some(range))
    }

    /// Copy GPU-newer resident pool ranges back to the guest: what overlaps
    /// `range`, or with `None` (a CPU wait) what uncached pools hold (the
    /// CPU must invalidate a cached pool's range before reading it).
    #[track_caller]
    fn resolve_resident(
        &mut self,
        core: &mut Guest,
        range: Option<std::ops::Range<u64>>,
    ) -> Result<(), String> {
        let Some(backend) = self.gpu.as_mut() else {
            return Ok(());
        };
        if range
            .as_ref()
            .is_some_and(|range| !backend.resident_writes_pending(range))
        {
            return Ok(());
        }
        let mut batches = Vec::new();
        for pool in backend.resident.values() {
            let mut ranges = Vec::new();
            for dirty in &pool.gpu_dirty {
                let piece = match &range {
                    Some(range) => range.start.max(dirty.start)..range.end.min(dirty.end),
                    None if pool.uncached => dirty.clone(),
                    None => continue,
                };
                if !piece.is_empty() {
                    ranges.push((piece.start - pool.base, piece.end - piece.start));
                }
            }
            if !ranges.is_empty() {
                batches.push((pool.base, ranges));
            }
        }
        if batches.is_empty() {
            return Ok(());
        }

        let mut ids = Vec::new();
        for (base, ranges) in &batches {
            ids.extend(backend.renderer.read_resident_ranges(*base, ranges)?);
        }
        let started = std::time::Instant::now();
        backend.renderer.flush()?;
        backend.stat(
            "stat: resident copy-back wait us",
            started.elapsed().as_micros() as u64,
        );
        crate::timeline::span(
            u64::MAX - 3,
            "resident-copy-back-flush",
            started,
            std::time::Instant::now(),
        );
        // Only these: pending dispatch and draw results stay for
        // `resolve_copies`.
        let mut completed = backend.renderer.take_reads(&ids);
        backend.stat("stat: resident ranges copied back", ids.len() as u64);
        // Which call site asked (`#[track_caller]` up the resolve chain).
        backend.stat(caller_stat(std::panic::Location::caller()), 1);
        let mut ids = ids.into_iter();
        let mut textures_changed = false;
        let result = (|| {
            for (base, ranges) in batches {
                for (offset, _) in ranges {
                    let token = ids.next().ok_or("resident read token missing")?;
                    let bytes = completed
                        .remove(&token)
                        .ok_or_else(|| format!("resident read {token} never completed"))?;
                    let start = base + offset;
                    core.memory
                        .write(start, &bytes)
                        .map_err(|trap| format!("resident write-back: {trap:?}"))?;
                    textures_changed |=
                        backend.mark_texture_bytes_changed(&(start..start + bytes.len() as u64));
                }
                // Keep this pool's ranges pending until every selected write
                // has reached guest memory.
                if let Some(pool) = backend.resident.get_mut(&base) {
                    if let Some(range) = &range {
                        remove_range(&mut pool.gpu_dirty, range.clone());
                    } else {
                        pool.gpu_dirty.clear();
                    }
                    backend.dirty_removals += 1;
                }
            }
            Ok(())
        })();
        if textures_changed {
            self.bump_binding_serial("stat: binding serial by resident texture writes");
        }
        result
    }

    /// A memory pool at guest `base` is finalized: its mirror is dropped
    /// (the memory may hold something else next). GPU writes the guest has
    /// not seen are lost with it, as the pool is.
    pub(super) fn gpu_forget_pool(&mut self, base: u64) {
        if let Some(backend) = self.gpu.as_mut() {
            if backend
                .resident
                .remove(&base)
                .is_some_and(|pool| !pool.gpu_dirty.is_empty())
            {
                backend.dirty_removals += 1;
            }
            // Its pages may be released and reused: shared spans of them
            // would keep the old ones.
            let whole = backend.shared_pools.remove(&base);
            let pieces = backend.shared_pieces.remove(&base);
            let released = whole.iter().copied().chain(
                pieces
                    .iter()
                    .flatten()
                    .map(|&(_, size, file, offset)| (size, file, offset)),
            );
            for (size, _, offset) in released.clone() {
                backend.renderer.release_file_range(offset, size);
            }
            if backend.pool_share_versions.remove(&base).is_none() {
                return;
            }
            backend.share_generation += 1;
            // Other guest pools referring to the same file bytes can have
            // pages pointing at the retired Vulkan buffer as well.
            let overlaps = |size: u64, file, offset: u64| {
                released.clone().any(|(length, backing, start)| {
                    file == backing && offset < start + length && start < offset + size
                })
            };
            let whole = backend
                .shared_pools
                .iter()
                .filter_map(|(&base, &(size, file, offset))| {
                    overlaps(size, file, offset).then_some(base)
                });
            let split = backend.shared_pieces.iter().filter_map(|(&base, pieces)| {
                pieces
                    .iter()
                    .any(|&(_, size, file, offset)| overlaps(size, file, offset))
                    .then_some(base)
            });
            for alias in whole.chain(split) {
                backend
                    .pool_share_versions
                    .insert(alias, backend.share_generation);
            }
        }
    }

    /// Record the file offsets of memory pools not yet known, where the GPU
    /// can share them in place (needs the page table: `core` under the
    /// machine lock).
    /// A vertex fetch table entry (`echora_shader_abi::VertexAttribute`)
    /// for an attribute state reading NVN `format`: its stream's bound
    /// buffer, stride and divisor. A stream with no buffer reads as
    /// unbound.
    /// The vertex attribute and stream snapshots bound now, parsed once
    /// per pair of snapshots (`vertex_state_memo`): their props are
    /// string-keyed maps, looked up per attribute and stream per draw.
    fn parsed_vertex_state(
        &mut self,
        attributes: &StateSnapshots,
        streams: &StateSnapshots,
    ) -> Arc<ParsedVertexState> {
        if let Some((known_attributes, known_streams, parsed)) = &self.vertex_state_memo
            && known_attributes.ptr_eq(attributes)
            && known_streams.ptr_eq(streams)
        {
            return parsed.clone();
        }
        let value = |props: &BTreeMap<String, super::Args>, key: &str| {
            props.get(key).map_or(0, |args| args.x[0])
        };
        let parsed = Arc::new(ParsedVertexState {
            attributes: attributes
                .iter()
                .map(|props| ParsedAttribute {
                    format: value(props, "Format") as u32,
                    stream: value(props, "StreamIndex"),
                    // SetFormat(state, format, relative offset).
                    offset: props.get("Format").map_or(0, |args| args.x[1]),
                })
                .collect(),
            streams: streams
                .iter()
                .map(|props| ParsedStream {
                    stride: value(props, "Stride"),
                    divisor: value(props, "Divisor"),
                })
                .collect(),
        });
        self.vertex_state_memo = Some((attributes.clone(), streams.clone(), parsed.clone()));
        parsed
    }

    fn fetch_entry(
        &self,
        attribute: &ParsedAttribute,
        streams: &[ParsedStream],
    ) -> Result<echora_shader_abi::VertexAttribute, String> {
        let format = attribute.format;
        let encoded =
            vertex_format(format).ok_or_else(|| format!("vertex attribute format {format}"))?;
        let stream = attribute.stream as usize;
        let offset = attribute.offset;
        let (Some(props), Some(bind)) = (
            streams.get(stream),
            self.render
                .state
                .get(bind_key("BindVertexBuffer", None, stream as u64)),
        ) else {
            return Ok(Default::default());
        };
        if encoded.components == 0 || bind.x[2] == 0 {
            return Ok(Default::default());
        }
        Ok(echora_shader_abi::VertexAttribute {
            address: bind.x[2] + offset,
            stride: props.stride as u32,
            divisor: props.divisor as u32,
            format: encoded.encode(),
            size: bind.x[3].saturating_sub(offset).min(u64::from(u32::MAX)) as u32,
            reserved: [0; 2],
        })
    }

    /// The GPU page table programs reach global memory through
    /// (`echora_shader_abi::PageTable`; ECHORA_PAGE_TABLE=0, debugging,
    /// leaves it empty), rebuilt when the
    /// memory pools change. A host-visible pool's pages point into shared
    /// spans of guest memory (the CPU's bytes, coherent as on the console);
    /// a GPU-only pool's into its resident copy. A page the GPU cannot reach
    /// stays 0 (reads 0).
    pub(super) fn page_table(
        &mut self,
        core: &mut Guest,
    ) -> Result<echora_shader_abi::PageTable, String> {
        use echora_shader_abi::{PAGE_SHIFT, PAGE_SIZE};
        if super::debug_var_os("ECHORA_PAGE_TABLE").is_some_and(|value| value == "0") {
            return Ok(Default::default());
        }
        // Host-visible pools reach the GPU through the file spans presents
        // record (`share_new_pools`, under the machine lock): a queue
        // submission's memory view cannot look them up. The table is rebuilt
        // when a pool is shared (one made between presents is mapped from
        // the next present on) or the pools change (`pool_epoch`), not by
        // walking the objects on every dispatch.
        let shared = self
            .gpu
            .as_ref()
            .map_or(0, |backend| backend.share_generation);
        let key = (shared, self.pool_epoch);
        if let Some((built, table)) = self.gpu.as_ref().and_then(|backend| backend.page_table)
            && built == key
        {
            return Ok(table);
        }
        let started = std::time::Instant::now();
        // Host-visible pools with no recorded span (never shared, or their
        // span dropped with another pool over the same memory): shared at
        // the next present. Mapped meanwhile only by a build that can look
        // spans up, they read zero to the GPU; the forest's task programs,
        // reading a header of zeros there, loop in every workgroup.
        // ECHORA_QUEUE_UNSHARED=0 (debugging): not queued, as before.
        let queue_unshared =
            !super::debug_var_os("ECHORA_QUEUE_UNSHARED").is_some_and(|value| value == "0");
        let unshared: Vec<u64> = self
            .gpu
            .as_ref()
            .filter(|_| queue_unshared)
            .map(|backend| {
                self.memory_pools
                    .iter()
                    .copied()
                    .filter(|address| {
                        self.objects.get(address).is_some_and(|pool| {
                            let flags = pool.builder.get("Flags").map_or(0x21, |flags| flags.x[0]);
                            pool.kind == "MemoryPool"
                                && flags & 6 != 0
                                && pool.builder.get("Storage").is_some_and(|storage| {
                                    storage.x[0] != 0
                                        && !backend.shared_pools.contains_key(&storage.x[0])
                                        && !backend.shared_pieces.contains_key(&storage.x[0])
                                })
                        })
                    })
                    .filter(|address| !self.new_pools.contains(address))
                    .collect()
            })
            .unwrap_or_default();
        if !unshared.is_empty() {
            self.gpu.as_mut().unwrap().stat(
                "stat: page table pools queued for sharing",
                unshared.len() as u64,
            );
            self.new_pools.extend(unshared);
        }
        let pools = self.indexed_pool_ranges().page_pools.clone();
        let first = pools
            .iter()
            .map(|pool| pool.gpu >> PAGE_SHIFT)
            .min()
            .unwrap_or(0);
        let end = pools
            .iter()
            .map(|pool| (pool.gpu + pool.size).div_ceil(PAGE_SIZE))
            .max()
            .unwrap_or(first);
        // A table past 16 million pages (64 GiB) means the pools are not
        // what this model expects.
        // Rounded up to 262144 pages (1 GiB): pools are made at rising GPU
        // addresses, about one a present, and a table that grows by a page
        // each time would be built whole each time.
        let count = (end - first).next_multiple_of(1 << 18).min(16 << 20) as usize;
        let traced = super::debug_var("ECHORA_POOL_TRACE")
            .ok()
            .and_then(|value| u64::from_str_radix(value.trim_start_matches("0x"), 16).ok());
        if let Some(traced) = traced {
            eprintln!(
                "pool trace page table (present {}): listed {:x?}, recorded {:x?}, generation {}",
                self.presented_frames,
                pools.iter().find(|pool| pool.base == traced),
                self.gpu
                    .as_ref()
                    .and_then(|backend| backend.shared_pools.get(&traced)),
                self.gpu
                    .as_ref()
                    .map_or(0, |backend| backend.share_generation),
            );
        }
        // Pools already in the table whose sharing has not changed keep
        // their pages: a present typically shares one new pool, and building
        // the whole table again (every pool's spans, ~256k entries) took
        // about 12 ms. A pool gone or moved, or a table of another extent,
        // builds it whole.
        let signatures: Vec<_> = pools
            .iter()
            .map(|&pool| (pool, self.pool_share_signature(pool.base, pool.flags)))
            .collect();
        let previous = self.gpu.as_mut().unwrap().page_build.take();
        let current: std::collections::HashSet<_> =
            signatures.iter().map(|(pool, _)| *pool).collect();
        // Spans joined into larger ones, or dropped with other pools' pages
        // in them, leave entries only a whole build replaces.
        let (mut superseded, superseded_bytes, dropped) =
            self.gpu.as_ref().unwrap().renderer.superseded_spans();
        let mut whole = match &previous {
            None => Some("stat: page table built whole (first)"),
            Some(built) if built.first != first => Some("stat: page table built whole (extent)"),
            Some(_) if dropped => Some("stat: page table built whole (span dropped)"),
            Some(_) if superseded >= 64 || superseded_bytes >= 256 << 20 => {
                Some("stat: page table built whole (spans joined)")
            }
            Some(_) => None,
        };
        let backend = self.gpu.as_mut().unwrap();
        if !super::debug_var_os("ECHORA_PACK_DMA_SPANS").is_some_and(|value| value == "0") {
            if whole.is_some() {
                backend.renderer.pack_addressed_spans()?;
            } else if backend.renderer.compact_page_table_imports()? {
                whole = Some("stat: page table built whole (stable spans packed)");
            }
        }
        if let Some(reason) = whole {
            backend.stat(reason, 1);
            superseded = backend.renderer.superseded_spans().0;
        }
        let previous = previous.filter(|_| whole.is_none());
        let mut modified = previous
            .as_ref()
            .filter(|built| built.entries.len() == count)
            .map(|_| Vec::new());
        let mut built_pools = HashMap::default();
        let mut entries = match previous {
            Some(built) => {
                built_pools = built.pools;
                built.entries
            }
            None => vec![0u64; count],
        };
        // The table grew or shrank at its end; pools gone (finalized, or
        // made again elsewhere) lose their pages.
        entries.resize(count, 0);
        let gone: Vec<_> = built_pools
            .keys()
            .filter(|pool| !current.contains(pool))
            .copied()
            .collect();
        for pool in gone {
            built_pools.remove(&pool);
            let pages = pool.pages(first, count);
            if let Some(range) = entries.get_mut(pages.clone()) {
                range.fill(0);
                if let Some(modified) = &mut modified {
                    modified.push(pages.start as u64..pages.end as u64);
                }
            }
            self.gpu
                .as_mut()
                .unwrap()
                .stat("stat: page table pools unmapped", 1);
            // A pool kept over the same GPU pages is mapped again.
            built_pools.retain(|kept, _| {
                kept.gpu + kept.size <= pool.gpu || pool.gpu + pool.size <= kept.gpu
            });
        }
        self.gpu.as_mut().unwrap().stat(
            "stat: page table walk microseconds",
            started.elapsed().as_micros() as u64,
        );
        let checked = core.memory.can_share();
        for (pool, signature) in signatures {
            // Mapped with this sharing already; one with unmapped pages
            // again only when it was mapped without the guest's page table
            // to look spans up in, which this build has.
            match built_pools.get(&pool) {
                Some(&(built, retry)) if built == signature && !(retry && checked) => continue,
                _ => {}
            }
            // ECHORA_PAGE_TABLE_LOG (debugging): each pool mapped and why.
            if super::debug_var_os("ECHORA_PAGE_TABLE_LOG").is_some() {
                eprintln!(
                    "page table map {pool:x?} present {} checked {checked}: was {:x?}, now {signature:x?}",
                    self.presented_frames,
                    built_pools.get(&pool)
                );
            }
            let backend = self.gpu.as_mut().unwrap();
            backend.stat("stat: page table pools mapped", 1);
            if built_pools.contains_key(&pool) {
                backend.stat("stat: page table pools mapped again", 1);
                backend.stat("stat: page table bytes mapped again", pool.size);
            }
            let pages = pool.pages(first, count);
            let complete = if let Some(range) = entries.get_mut(pages.clone()) {
                range.fill(0);
                self.map_pool_pages(core, pool, first + pages.start as u64, range)?;
                if let Some(modified) = &mut modified {
                    modified.push(pages.start as u64..pages.end as u64);
                }
                range.iter().all(|&entry| entry != 0)
            } else {
                false
            };
            // A pool with unmapped pages, mapped without the guest's page
            // table, is mapped again by a build that has it.
            // Its signature after mapping (a GPU-only pool's resident
            // buffer exists from its first mapping on).
            let mapped = self.pool_share_signature(pool.base, pool.flags);
            built_pools.insert(pool, (mapped, !complete && !checked));
        }
        let backend = self.gpu.as_mut().unwrap();
        // The renderer's counts from the spans just shared (refusals).
        for (name, amount) in backend.renderer.take_stats() {
            backend.stat(name, amount);
        }
        backend.stat(
            "stat: page table map microseconds",
            started.elapsed().as_micros() as u64,
        );
        if let Some(modified) = &mut modified {
            merge_ranges(modified);
        }
        let table = backend.set_page_table(
            key,
            PageBuild {
                first,
                entries,
                pools: built_pools,
                shared_mapping: std::cell::OnceCell::new(),
            },
            modified.as_deref(),
        )?;
        if whole.is_some() {
            backend.renderer.take_superseded(superseded);
        }
        backend.stat(
            "stat: page table pinned MiB",
            backend.renderer.addressed_bytes() >> 20,
        );
        backend.stat("stat: page table builds", 1);
        backend.stat(
            "stat: page table build microseconds",
            started.elapsed().as_micros() as u64,
        );
        Ok(table)
    }

    /// What decides how pool `base` maps into the page table besides its
    /// own properties: its recorded share (whole or in pieces) and, for a
    /// GPU-only pool, its resident buffer's address.
    fn pool_share_signature(&self, base: u64, flags: u64) -> PoolShareVersion {
        let Some(backend) = self.gpu.as_ref() else {
            return PoolShareVersion::Unshared;
        };
        if flags & 6 == 0 {
            return PoolShareVersion::Resident(
                backend.renderer.resident_address(base).unwrap_or(0),
            );
        }
        backend
            .pool_share_versions
            .get(&base)
            .copied()
            .map_or(PoolShareVersion::Unshared, PoolShareVersion::Imported)
    }

    /// Write the GPU page entries for both a direct pool and a virtual alias.
    fn map_pool_pages(
        &mut self,
        core: &mut Guest,
        pool: PagePool,
        first: u64,
        entries: &mut [u64],
    ) -> Result<(), String> {
        use echora_shader_abi::{PAGE_SHIFT, PAGE_SIZE};
        let PagePool {
            base,
            backing_size,
            offset: backing_offset,
            size,
            gpu,
            flags,
        } = pool;
        // A page under the largest span shared at once, so a chunk whose
        // file offset is not page-aligned still fits once rounded to pages.
        const CHUNK: u64 = (64 << 20) - PAGE_SIZE;
        let host_visible = flags & 6 != 0;
        // A pool shared in pieces: each piece, in chunks.
        let pieces = self
            .gpu
            .as_ref()
            .and_then(|backend| backend.shared_pieces.get(&base))
            .filter(|_| host_visible)
            .cloned();
        if let Some(pieces) = pieces {
            for (piece_offset, piece_length, file, at) in pieces {
                let end = (piece_offset + piece_length).min(backing_offset + size);
                let mut current = piece_offset.max(backing_offset);
                while current < end {
                    let length = (end - current).min(CHUNK);
                    let address = self.gpu.as_mut().unwrap().renderer.address_span(
                        file,
                        at + current - piece_offset,
                        length,
                    )?;
                    if let Some(address) = address {
                        let first_page = (gpu + current - backing_offset) >> PAGE_SHIFT;
                        for page in 0..length.div_ceil(PAGE_SIZE) {
                            if let Some(entry) =
                                entries.get_mut((first_page + page - first) as usize)
                            {
                                *entry = address + page * PAGE_SIZE;
                            }
                        }
                    }
                    current += length;
                }
            }
            return Ok(());
        }
        let mut offset = 0;
        while offset < size {
            let length = (size - offset).min(CHUNK);
            let mut recorded_span = false;
            let address = if host_visible {
                let recorded = self
                    .gpu
                    .as_ref()
                    .and_then(|backend| backend.shared_pools.get(&base))
                    .filter(|(shared_size, ..)| *shared_size >= backing_offset + offset + length)
                    .map(|&(_, file, at)| (file, at + backing_offset + offset));
                recorded_span = recorded.is_some();
                match recorded.or_else(|| {
                    core.memory
                        .shared_span(base + backing_offset + offset, length)
                }) {
                    Some((file, at)) => self
                        .gpu
                        .as_mut()
                        .unwrap()
                        .renderer
                        .address_span(file, at, length)?,
                    None => None,
                }
            } else {
                self.resident_pool_binding(core, base, backing_size)?;
                self.gpu
                    .as_ref()
                    .unwrap()
                    .renderer
                    .resident_address(base)
                    .map(|address| address + backing_offset + offset)
            };
            // A chunk no single span shares (a page unmapped, or the
            // backing not contiguous): the runs of pages that are.
            if address.is_none() && host_visible {
                let mut runs: Vec<(u64, u64, i32, u64)> = Vec::new();
                let mut page = 0;
                while page * PAGE_SIZE < length {
                    let at = offset + page * PAGE_SIZE;
                    if let Some((file, file_at)) = core
                        .memory
                        .shared_span(base + backing_offset + at, PAGE_SIZE)
                    {
                        match runs.last_mut() {
                            Some(run)
                                if run.2 == file
                                    && run.0 + run.1 == at
                                    && run.3 + run.1 == file_at =>
                            {
                                run.1 += PAGE_SIZE;
                            }
                            _ => runs.push((at, PAGE_SIZE, file, file_at)),
                        }
                    }
                    page += 1;
                }
                let backend = self.gpu.as_mut().unwrap();
                if runs.is_empty() {
                    let known = backend.shared_pools.get(&base).map(|&(shared, ..)| shared);
                    backend.note(format!(
                        "page table: pool at GPU {gpu:#x} (+{size:#x}) flags {flags:#x} not mapped ({}), guest {base:#x}, recorded size {known:x?}",
                        if recorded_span { "its recorded span did not import" } else { "no shared span" }
                    ));
                } else {
                    backend.stat("stat: page table chunks mapped in runs", 1);
                }
                for (at, run_length, file, file_at) in runs {
                    let mapped = self
                        .gpu
                        .as_mut()
                        .unwrap()
                        .renderer
                        .address_span(file, file_at, run_length)?;
                    if let Some(mapped) = mapped {
                        let first_page = (gpu + at) >> PAGE_SHIFT;
                        for page in 0..run_length / PAGE_SIZE {
                            if let Some(entry) =
                                entries.get_mut((first_page + page - first) as usize)
                            {
                                *entry = mapped + page * PAGE_SIZE;
                            }
                        }
                    }
                }
                offset += length;
                continue;
            }
            if address.is_none() {
                let note = format!(
                    "page table: pool at GPU {gpu:#x} (+{size:#x}) flags {flags:#x} not mapped{}",
                    if !host_visible {
                        " (no resident copy)"
                    } else if recorded_span {
                        " (its recorded span did not import)"
                    } else {
                        " (no shared span)"
                    }
                );
                let known = self
                    .gpu
                    .as_ref()
                    .and_then(|backend| backend.shared_pools.get(&base))
                    .map(|&(shared, ..)| shared);
                let note = format!("{note}, guest {base:#x}, recorded size {known:x?}");
                self.gpu.as_mut().unwrap().note(note);
            }
            if let Some(address) = address {
                let first_page = (gpu + offset) >> PAGE_SHIFT;
                for page in 0..length.div_ceil(PAGE_SIZE) {
                    if let Some(entry) = entries.get_mut((first_page + page - first) as usize) {
                        *entry = address + page * PAGE_SIZE;
                    }
                }
            }
            offset += length;
        }
        Ok(())
    }

    /// Write one dispatch's inputs for the workgroup simulator (`cargo run
    /// -p echora-shader --features sim --example simulate`) under
    /// ECHORA_DUMP_DIR (default the cache's dispatch-dumps), one directory
    /// per program: the SPIR-V, the group counts, the push constants, each
    /// uniform and storage buffer by binding, the page table and the memory
    /// behind it of every pool the dispatch can name (one its constant
    /// buffers hold an address in).
    #[allow(clippy::too_many_arguments)]
    fn dump_dispatch<S: std::hash::BuildHasher>(
        &self,
        core: &Guest,
        key: u64,
        stage: &Translated,
        groups: [u32; 3],
        uniforms: &std::collections::HashMap<u32, Vec<u8>, S>,
        storage: &[(u32, u64, u64)],
        page_table: echora_shader_abi::PageTable,
    ) -> Result<(), String> {
        use echora_shader_abi::{PAGE_SHIFT, PAGE_SIZE};
        let directory = self.dump_directory(key)?;
        std::fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
        let write = |name: String, bytes: &[u8]| {
            std::fs::write(directory.join(&name), bytes).map_err(|error| format!("{name}: {error}"))
        };
        let words: Vec<u8> = stage
            .spirv
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        write("program.spv".into(), &words)?;
        // Each binding's guest source: kind, constant bank (`index`), its
        // used size, and where a handle or pointer for it is read.
        let mut listing = String::new();
        for descriptor in &stage.descriptors {
            listing += &format!(
                "binding {} kind {:?} bank {} used {} at c{}+{:#x} flags {:#x}\n",
                descriptor.binding,
                descriptor.kind(),
                descriptor.index,
                stage
                    .cbuf_used_sizes
                    .get(descriptor.index as usize)
                    .copied()
                    .unwrap_or(0),
                descriptor.cbuf_index,
                descriptor.cbuf_offset,
                descriptor.flags
            );
        }
        write("descriptors.txt".into(), listing.as_bytes())?;
        write(
            "dispatch.txt".into(),
            format!(
                "groups {} {} {}\nsubgroup 32\n",
                groups[0], groups[1], groups[2]
            )
            .as_bytes(),
        )?;
        let push = echora_shader_abi::PushConstants {
            page_table,
            ..Default::default()
        };
        write("push.bin".into(), &push.to_bytes())?;
        for (binding, bytes) in uniforms {
            write(format!("uniform-0-{binding}.bin"), bytes)?;
        }
        // Storage buffers as guest memory holds them now, from their
        // 16-byte aligned start (the translation's offsets count from it).
        for &(binding, base, length) in storage {
            write(
                format!("storage-0-{binding}.bin"),
                &self.gpu_bytes(core, base, length)?,
            )?;
        }
        let entries = self.gpu.as_ref().unwrap().renderer.page_table_entries();
        if page_table.address == 0 || entries.is_empty() {
            return Ok(());
        }
        let table: Vec<u8> = entries
            .iter()
            .flat_map(|entry| entry.to_le_bytes())
            .collect();
        write(format!("memory-{:x}.bin", page_table.address), &table)?;
        // ECHORA_DUMP_MEMORY=all: every mapped page, in runs whose device
        // addresses follow on (the GPU addresses do too), at most 64 MiB a
        // file.
        // ECHORA_DUMP_MEMORY=START-END (hex guest GPU addresses): also the
        // mapped pages in that range (memory the program reaches through a
        // pointer it loads), beside the pools its constant buffers name.
        let mode = super::debug_var("ECHORA_DUMP_MEMORY").ok();
        let range = mode.as_deref().and_then(|mode| {
            let (start, end) = mode.split_once('-')?;
            let parse =
                |text: &str| u64::from_str_radix(text.trim().trim_start_matches("0x"), 16).ok();
            Some(parse(start)?..parse(end)?)
        });
        if mode.as_deref() == Some("all") || range.is_some() {
            let first = u64::from(page_table.first_page);
            let (mut index, last) = match &range {
                Some(range) => (
                    ((range.start >> PAGE_SHIFT).saturating_sub(first) as usize).min(entries.len()),
                    ((range.end >> PAGE_SHIFT).saturating_sub(first) as usize).min(entries.len()),
                ),
                None => (0, entries.len()),
            };
            let entries = &entries[..last];
            while index < entries.len() {
                let base = entries[index];
                if base == 0 {
                    index += 1;
                    continue;
                }
                let mut end = index + 1;
                while end < entries.len()
                    && end - index < 16384
                    && entries[end] == base + (end - index) as u64 * PAGE_SIZE
                {
                    end += 1;
                }
                let gpu = (first + index as u64) << PAGE_SHIFT;
                let length = (end - index) as u64 * PAGE_SIZE;
                // Read from the file behind guest memory where there is one
                // (pread leaves the guest mapping's pages out of this
                // process's resident set; reading through the mapping
                // faulted gigabytes in). A page the guest cannot read stays
                // out (it reads as unmapped in the simulator).
                let shared = self
                    .gpu_to_guest(gpu, length)
                    .ok()
                    .and_then(|guest| core.memory.shared_span(guest, length));
                let bytes = shared
                    .and_then(|(file, at)| read_file_range(file, at, length))
                    .or_else(|| self.gpu_bytes(core, gpu, length).ok());
                if let Some(bytes) = bytes {
                    write(format!("memory-{base:x}.bin"), &bytes)?;
                }
                index = end;
            }
            if range.is_none() {
                return Ok(());
            }
        }
        // Pools an address in the constant buffers falls in.
        let pools: Vec<(u64, u64)> = self
            .memory_pools
            .iter()
            .filter_map(|address| self.objects.get(address))
            .filter(|object| object.kind == "MemoryPool")
            .filter_map(|pool| {
                let size = pool.builder.get("Storage")?.x[1];
                let gpu = pool.state.get("GpuAddress")?.x[0];
                (gpu != 0 && size != 0).then_some((gpu, size))
            })
            .collect();
        let mut named: Vec<(u64, u64)> = Vec::new();
        for bytes in uniforms.values() {
            for word in bytes.chunks_exact(8) {
                let address = u64::from_le_bytes(word.try_into().unwrap());
                if let Some(&pool) = pools
                    .iter()
                    .find(|&&(gpu, size)| (gpu..gpu + size).contains(&address))
                    && !named.contains(&pool)
                {
                    named.push(pool);
                }
            }
        }
        let first = u64::from(page_table.first_page);
        let mut total = 0u64;
        for (gpu, size) in named {
            if total + size > 4 << 30 {
                return Err(format!("pools past 4 GiB; {gpu:#x} (+{size:#x}) left out"));
            }
            total += size;
            let bytes = self.gpu_bytes(core, gpu, size)?;
            // Runs of pages whose device addresses follow on.
            let pages = size.div_ceil(PAGE_SIZE);
            let mut start = 0;
            while start < pages {
                let device = |page: u64| {
                    ((gpu >> PAGE_SHIFT) + page)
                        .checked_sub(first)
                        .and_then(|index| entries.get(index as usize))
                        .copied()
                        .unwrap_or(0)
                };
                let base = device(start);
                let mut end = start + 1;
                while end < pages && base != 0 && device(end) == base + (end - start) * PAGE_SIZE {
                    end += 1;
                }
                if base != 0 {
                    let from = (start * PAGE_SIZE) as usize;
                    let to = ((end * PAGE_SIZE) as usize).min(bytes.len());
                    write(format!("memory-{base:x}.bin"), &bytes[from..to])?;
                }
                start = end;
            }
        }
        Ok(())
    }

    /// Where `dump_dispatch` writes program `key`'s dispatch this present.
    fn dump_directory(&self, key: u64) -> Result<std::path::PathBuf, String> {
        dump_directory_at(key, self.presented_frames)
    }

    /// ECHORA_DUMP_DRAW for a mesh draw: its task and mesh stages as
    /// dispatches (`dump_dispatch`, with the draw's group count; a task
    /// program's payload is left to the reader), in directories named by
    /// the fragment key xor "task" or "mesh".
    #[allow(clippy::too_many_arguments)]
    fn dump_mesh_stages(
        &mut self,
        core: &mut Guest,
        command: &Command,
        fragment: u64,
        stages: &[std::sync::Arc<Translated>],
        uniforms: &echora_gpu::FastMap<(usize, u32), Vec<u8>>,
        textures: &echora_gpu::FastMap<(usize, u32), (TextureSource, echora_gpu::draw::Sampler)>,
        page_table: echora_shader_abi::PageTable,
    ) {
        let groups = match command.name {
            "nvnCommandBufferDrawMeshTasksIndirect" => {
                let address = command.x[1];
                let count = self
                    .gpu_to_guest(address, 4)
                    .and_then(|guest| self.resolve_gpu_written(core, guest..guest + 4))
                    .and_then(|()| self.gpu_bytes(core, address, 4));
                match count {
                    Ok(bytes) => u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                    Err(error) => {
                        eprintln!("draw dump {fragment:016x}: indirect count: {error}");
                        return;
                    }
                }
            }
            _ => command.x[2] as u32,
        };
        for (index, stage) in stages.iter().enumerate() {
            let tag: u64 = match stage.stage {
                echora_shader_abi::Stage::Task => 0x7461_736b,
                echora_shader_abi::Stage::Mesh => 0x6d65_7368,
                _ => continue,
            };
            let key = fragment ^ tag;
            self.dumped_dispatches.insert(key);
            let stage_uniforms: HashMap<u32, Vec<u8>> = uniforms
                .iter()
                .filter(|((stage_index, _), _)| *stage_index == index)
                .map(|((_, binding), bytes)| (*binding, bytes.clone()))
                .collect();
            if let Err(error) = self.dump_dispatch(
                core,
                key,
                stage,
                [groups, 1, 1],
                &stage_uniforms,
                &[],
                page_table,
            ) {
                eprintln!("draw dump {key:016x}: {error}");
            }
            let read: Vec<(u32, TextureSource)> = textures
                .iter()
                .filter(|((stage_index, _), _)| *stage_index == index)
                .map(|((_, binding), (source, _))| (*binding, source.clone()))
                .collect();
            if let Err(error) = self.dump_images(key, &read) {
                eprintln!("draw dump {key:016x} images: {error}");
            }
            eprintln!(
                "draw dump {key:016x}: {:?} stage, {groups} groups",
                stage.stage
            );
        }
    }

    /// ECHORA_DUMP_DRAW for an ordinary draw: its vertex stage as a
    /// dispatch (`dump_dispatch`) of one invocation, the vertex its index
    /// buffer names first, in a directory named by the fragment key xor
    /// "vert". The vertex fetch table goes into `memory-` at a made-up
    /// device address the push constants name, and `dispatch.txt` gives the
    /// draw's VertexIndex, BaseVertex, InstanceIndex and BaseInstance.
    #[allow(clippy::too_many_arguments)]
    fn dump_vertex_stage(
        &mut self,
        core: &mut Guest,
        command: &Command,
        fragment: u64,
        stages: &[std::sync::Arc<Translated>],
        uniforms: &echora_gpu::FastMap<(usize, u32), Vec<u8>>,
        textures: &echora_gpu::FastMap<(usize, u32), (TextureSource, echora_gpu::draw::Sampler)>,
        page_table: echora_shader_abi::PageTable,
        (fetch, base_vertex, first_instance): (&[echora_shader_abi::VertexAttribute], i32, u32),
    ) {
        let Some((index, stage)) = stages
            .iter()
            .enumerate()
            .find(|(_, stage)| stage.stage == echora_shader_abi::Stage::Vertex)
        else {
            return;
        };
        let key = fragment ^ 0x7665_7274;
        self.dumped_dispatches.insert(key);
        // The first index of an indexed draw (else vertex 0 of the draw).
        let first = if command.name.starts_with("nvnCommandBufferDrawElements") {
            let size = match command.x[2] {
                0 => 1,
                1 => 2,
                _ => 4,
            };
            self.gpu_bytes(core, command.x[4], size).map_or(0, |bytes| {
                bytes
                    .iter()
                    .rev()
                    .fold(0u32, |value, &byte| value << 8 | u32::from(byte))
            })
        } else {
            0
        };
        let stage_uniforms: HashMap<u32, Vec<u8>> = uniforms
            .iter()
            .filter(|((stage_index, _), _)| *stage_index == index)
            .map(|((_, binding), bytes)| (*binding, bytes.clone()))
            .collect();
        if let Err(error) = self.dump_dispatch(
            core,
            key,
            stage,
            [1, 1, 1],
            &stage_uniforms,
            &[],
            page_table,
        ) {
            eprintln!("draw dump {key:016x}: {error}");
            return;
        }
        let read: Vec<(u32, TextureSource)> = textures
            .iter()
            .filter(|((stage_index, _), _)| *stage_index == index)
            .map(|((_, binding), (source, _))| (*binding, source.clone()))
            .collect();
        if let Err(error) = self.dump_images(key, &read) {
            eprintln!("draw dump {key:016x} images: {error}");
        }
        let Ok(directory) = self.dump_directory(key) else {
            return;
        };
        // The fetch table at a device address no pool uses.
        const TABLE: u64 = 0x7f00_0000_0000;
        let table: Vec<u8> = fetch
            .iter()
            .flat_map(|entry| {
                let mut bytes = Vec::with_capacity(32);
                bytes.extend(entry.address.to_le_bytes());
                for word in [entry.stride, entry.divisor, entry.format, entry.size, 0, 0] {
                    bytes.extend(word.to_le_bytes());
                }
                bytes
            })
            .collect();
        let push = echora_shader_abi::PushConstants {
            vertex_fetch: echora_shader_abi::VertexFetch {
                table: if fetch.is_empty() { 0 } else { TABLE },
                base_instance: first_instance,
                attributes: (1u64 << fetch.len()).wrapping_sub(1) as u32,
            },
            resolution_scale: [1.0; 4],
            page_table,
            ..Default::default()
        };
        let vertex = (first as i64 + i64::from(base_vertex)) as u32;
        let lines = format!(
            "builtin 42 {vertex}\nbuiltin 4424 {}\nbuiltin 43 {first_instance}\nbuiltin 4425 {first_instance}\n",
            base_vertex as u32
        );
        let result = std::fs::write(directory.join(format!("memory-{TABLE:x}.bin")), &table)
            .and_then(|()| std::fs::write(directory.join("push.bin"), push.to_bytes()))
            .and_then(|()| {
                use std::io::Write;
                std::fs::OpenOptions::new()
                    .append(true)
                    .open(directory.join("dispatch.txt"))?
                    .write_all(lines.as_bytes())
            });
        match result {
            Ok(()) => eprintln!(
                "draw dump {key:016x}: vertex stage, vertex {vertex} (index {first} + base {base_vertex}), instance {first_instance}"
            ),
            Err(error) => eprintln!("draw dump {key:016x}: {error}"),
        }
    }

    /// A dumped draw's fragment program and inputs, for the simulator: as
    /// `dump_dispatch`, with `dispatch.txt` naming the target's middle
    /// pixel (`fragcoord`; its attributes are left to the reader) and each
    /// sampled texture's level 0, every slice of a 3D one.
    fn dump_draw(
        renderer: &mut echora_gpu::renderer::Renderer,
        directory: &std::path::Path,
        draw: &echora_gpu::draw::Draw,
        target_extent: Option<(u32, u32)>,
    ) -> Result<(), String> {
        std::fs::create_dir_all(directory).map_err(|error| error.to_string())?;
        let write = |name: String, bytes: &[u8]| {
            std::fs::write(directory.join(&name), bytes).map_err(|error| format!("{name}: {error}"))
        };
        let (index, stage) = draw
            .program
            .stages
            .iter()
            .enumerate()
            .find(|(_, stage)| stage.stage == echora_shader_abi::Stage::Fragment)
            .ok_or("no fragment stage")?;
        let words: Vec<u8> = stage
            .spirv
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        write("program.spv".into(), &words)?;
        let (width, height) = target_extent.unwrap_or((1, 1));
        write(
            "dispatch.txt".into(),
            format!(
                "groups 1 1 1\nfragcoord {}.5 {}.5 0.5 1.0\n",
                width / 2,
                height / 2
            )
            .as_bytes(),
        )?;
        let push = echora_shader_abi::PushConstants {
            resolution_scale: [1.0; 4],
            page_table: draw.page_table,
            ..Default::default()
        };
        write("push.bin".into(), &push.to_bytes())?;
        for ((stage_index, binding), bytes) in draw.uniforms {
            if *stage_index == index {
                write(format!("uniform-0-{binding}.bin"), bytes)?;
            }
        }
        for ((stage_index, binding), &(guest, length)) in draw.guest_uniforms {
            if *stage_index == index
                && let Some(read) = draw.read_guest
            {
                let mut bytes = vec![0; length];
                read(guest, &mut bytes)?;
                write(format!("uniform-0-{binding}.bin"), &bytes)?;
            }
        }
        for ((stage_index, binding), (source, _)) in draw.textures {
            if *stage_index != index {
                continue;
            }
            // Every slice of a volume and every layer of an array (the
            // simulator reads them all), slice or layer after slice.
            let (texels, width, height, depth, layers) = match source {
                TextureSource::Upload {
                    key,
                    width,
                    height,
                    shape,
                    ..
                } => {
                    renderer.texture_view_for(source)?;
                    let (depth, layers) = (shape.depth(), shape.layers());
                    let mut texels = Vec::new();
                    for slice in 0..depth.max(layers) {
                        texels.extend(renderer.read_upload_slice(*key, 0, slice)?);
                    }
                    (texels, *width, *height, depth, layers)
                }
                TextureSource::Target(key) | TextureSource::TargetView { key, .. } => {
                    let target = renderer.existing_target(*key).ok_or("no such target")?;
                    let (width, height) = (target.width, target.height);
                    let (depth, layers) = (target.shape.depth(), target.shape.layers());
                    let mut texels = Vec::new();
                    for slice in 0..depth.max(layers) {
                        texels.extend(renderer.read_target_slice(*key, 0, slice)?);
                    }
                    (texels, width, height, depth, layers)
                }
            };
            let bytes: Vec<u8> = texels
                .iter()
                .flat_map(|texel| texel.iter().flat_map(|part| part.to_bits().to_le_bytes()))
                .collect();
            write(format!("image-0-{binding}.bin"), &bytes)?;
            write(
                format!("image-0-{binding}.txt"),
                format!("{width} {height} {depth} {layers}\n").as_bytes(),
            )?;
        }
        Ok(())
    }

    /// A dumped dispatch's textures and storage images, by binding: level
    /// 0 read back from the GPU as RGBA f32 texels (`image-0-B.bin`, layer
    /// 0, every slice of a volume) with `WIDTH HEIGHT DEPTH LAYERS`
    /// (`image-0-B.txt`). Integer
    /// formats read back as floats here.
    fn dump_images(&mut self, key: u64, read: &[(u32, TextureSource)]) -> Result<(), String> {
        let directory = self.dump_directory(key)?;
        let renderer = &mut self.gpu.as_mut().unwrap().renderer;
        // A binding that cannot be read is reported and skipped, the
        // others still written.
        let mut failed = Vec::new();
        for (binding, source) in read {
            eprintln!("dispatch dump {key:016x}: reading binding {binding} from {source:?}");
            let mut one = || -> Result<(), String> {
                // Every slice of a volume and every layer of an array (the
                // simulator reads them all).
                let (depth, layers);
                let texels = match source {
                    TextureSource::Upload { key, shape, .. } => {
                        renderer.texture_view_for(source)?;
                        (depth, layers) = (shape.depth(), shape.layers());
                        let mut texels = Vec::new();
                        for slice in 0..depth.max(layers) {
                            texels.extend(renderer.read_upload_slice(*key, 0, slice)?);
                        }
                        texels
                    }
                    TextureSource::Target(key) | TextureSource::TargetView { key, .. } => {
                        (depth, layers) = renderer.existing_target(*key).map_or((1, 1), |target| {
                            (target.shape.depth(), target.shape.layers())
                        });
                        let mut texels = Vec::new();
                        for slice in 0..depth.max(layers) {
                            texels.extend(renderer.read_target_slice(*key, 0, slice)?);
                        }
                        texels
                    }
                };
                let (width, height) = match source {
                    TextureSource::Upload { width, height, .. } => (*width, *height),
                    TextureSource::Target(key) | TextureSource::TargetView { key, .. } => renderer
                        .existing_target(*key)
                        .map(|target| (target.width, target.height))
                        .ok_or("no such target")?,
                };
                let count = (width * height * depth.max(layers)) as usize;
                let bytes: Vec<u8> = texels
                    .iter()
                    .take(count)
                    .flat_map(|texel| texel.iter().flat_map(|part| part.to_bits().to_le_bytes()))
                    .collect();
                std::fs::write(directory.join(format!("image-0-{binding}.bin")), bytes)
                    .map_err(|error| error.to_string())?;
                std::fs::write(
                    directory.join(format!("image-0-{binding}.txt")),
                    format!("{width} {height} {depth} {layers}\n"),
                )
                .map_err(|error| error.to_string())
            };
            if let Err(error) = one() {
                failed.push(format!("binding {binding}: {error}"));
            }
        }
        if failed.is_empty() {
            Ok(())
        } else {
            Err(failed.join("; "))
        }
    }

    /// ECHORA_TARGET_TRACE=<target key>[,...] or `all` (debugging), from
    /// present ECHORA_TARGET_TRACE_FROM to ECHORA_TARGET_TRACE_TO: after
    /// each draw or dispatch that writes
    /// one of them, its level 0 read back, as its mean colour and how many
    /// texels are not finite (where a pass first leaves NaN or white).
    /// ECHORA_MEMORY_TRACE=START-END (hex GPU addresses, debugging), from
    /// present ECHORA_MEMORY_TRACE_FROM to ECHORA_MEMORY_TRACE_TO: after
    /// each draw or dispatch, that range as the guest sees it (GPU writes
    /// resolved), logged when it changed since the last look: which pass
    /// (or the CPU, between passes) wrote a buffer. Each look waits for the
    /// GPU, so keep the presents few.
    fn trace_memory(&mut self, core: &mut Guest, what: &str) -> Result<(), String> {
        let Ok(range) = super::debug_var("ECHORA_MEMORY_TRACE") else {
            return Ok(());
        };
        let present = |name: &'static str, default: u64| {
            super::debug_var(name)
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(default)
        };
        let (from, to) = (
            present("ECHORA_MEMORY_TRACE_FROM", 0),
            present("ECHORA_MEMORY_TRACE_TO", u64::MAX),
        );
        if !(from..=to).contains(&self.presented_frames) {
            return Ok(());
        }
        // START-END[,START-END...]: each range on its own.
        for item in range.split(',') {
            let Some((start, end)) = item.split_once('-').and_then(|(start, end)| {
                let parse =
                    |text: &str| u64::from_str_radix(text.trim().trim_start_matches("0x"), 16).ok();
                Some((parse(start)?, parse(end)?))
            }) else {
                return Err(format!("ECHORA_MEMORY_TRACE={range}: not START-END"));
            };
            self.resolve_gpu_written(core, start..end)?;
            let bytes = self.gpu_bytes(core, start, end - start)?;
            if self.memory_trace_last.get(&start) != Some(&bytes) {
                let words: Vec<String> = bytes
                    .chunks_exact(4)
                    .take(16)
                    .map(|word| {
                        format!(
                            "{:08x}",
                            u32::from_le_bytes([word[0], word[1], word[2], word[3]])
                        )
                    })
                    .collect();
                eprintln!(
                    "memory trace present {} after {what}: {start:#x}.. now {}",
                    self.presented_frames,
                    words.join(" ")
                );
                self.memory_trace_last.insert(start, bytes);
            }
        }
        Ok(())
    }

    fn trace_targets(
        &mut self,
        written: &[u64],
        places: &[(u32, u32)],
        what: &str,
    ) -> Result<(), String> {
        let Ok(keys) = super::debug_var("ECHORA_TARGET_TRACE") else {
            return Ok(());
        };
        let from = super::debug_var("ECHORA_TARGET_TRACE_FROM")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(0);
        let to = super::debug_var("ECHORA_TARGET_TRACE_TO")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(u64::MAX);
        if !(from..=to).contains(&self.presented_frames) {
            return Ok(());
        }
        // Writes of the traced programs count as `sampled` too (their
        // result beside their inputs).
        let traced_write = what.starts_with("draw ")
            && super::debug_var("ECHORA_GPU_TRACE_PROGRAM").is_ok_and(|wanted| {
                wanted
                    .split(',')
                    .any(|item| what.ends_with(&format!("/{}", item.trim())))
            });
        let present = self.presented_frames;
        let storages: Vec<u64> = written
            .iter()
            .map(|key| self.texture_storage(*key).unwrap_or(0))
            .collect();
        let backend = self.gpu.as_mut().unwrap();
        for (index, &key) in written.iter().enumerate() {
            // The (level, layer) a draw rendered into (level 0, layer 0
            // otherwise).
            let (level, layer) = places.get(index).copied().unwrap_or((0, 0));
            // `sampled` in the list: the targets traced draws sample;
            // `@<storage hex>`: the target over that memory.
            if keys != "all"
                && !keys.split(',').any(|item| {
                    item.trim() == format!("{key:x}")
                        || item.trim() == format!("@{:x}", storages[index])
                        || (item.trim() == "sampled"
                            && (traced_write || what.starts_with("sampled by")))
                })
            {
                continue;
            }
            let Some(target) = backend.renderer.existing_target(key) else {
                continue;
            };
            let (width, height, format) = (target.width, target.height, target.format);
            let (width, height) = ((width >> level).max(1), (height >> level).max(1));
            let depth = target.shape.depth();
            backend.renderer.flush()?;
            // A depth target past 4096² (a 12288² shadow map is 2.4 GB as
            // RGBA f32): only the requested rectangles, read as raw depth.
            if echora_gpu::renderer::is_depth(format)
                && u64::from(width) * u64::from(height) > 4096 * 4096
            {
                let depth_of = |bytes: &[u8]| -> f32 {
                    match format {
                        vk::Format::D16_UNORM => {
                            f32::from(u16::from_le_bytes([bytes[0], bytes[1]])) / 65535.0
                        }
                        vk::Format::D24_UNORM_S8_UINT | vk::Format::X8_D24_UNORM_PACK32 => {
                            (u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                                & 0xff_ffff) as f32
                                / 16_777_215.0
                        }
                        _ => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                    }
                };
                let size = if format == vk::Format::D16_UNORM {
                    2
                } else {
                    4
                };
                let rects = super::debug_var("ECHORA_TARGET_TRACE_RECT").unwrap_or_default();
                for rect in rects.split(';') {
                    let values: Vec<u32> = rect
                        .split(',')
                        .filter_map(|value| value.trim().parse().ok())
                        .collect();
                    let [x, y, w, h] = values[..] else {
                        continue;
                    };
                    if x + w > width || y + h > height || level != 0 || layer != 0 {
                        continue;
                    }
                    let bytes =
                        backend
                            .renderer
                            .read_target_region_bytes(key, size, Some([x, y, w, h]))?;
                    for (row, line) in bytes.chunks_exact((w * size) as usize).enumerate() {
                        let values: Vec<String> = line
                            .chunks_exact(size as usize)
                            .map(|texel| format!("{:.6}", depth_of(texel)))
                            .collect();
                        eprintln!(
                            "target trace present {present} {what} {key:x} row {} from {x}: {}",
                            y as usize + row,
                            values.join(" ")
                        );
                    }
                }
                if let (Ok(regions), Some(directory)) = (
                    super::debug_var("ECHORA_TARGET_TRACE_PGM"),
                    super::debug_var_os("ECHORA_DUMP_DIR"),
                ) {
                    for region in regions.split(';') {
                        let values: Vec<u32> = region
                            .split(',')
                            .filter_map(|value| value.trim().parse().ok())
                            .collect();
                        let [x, y, w, h, step] = values[..] else {
                            continue;
                        };
                        let step = step.max(1);
                        if x + w > width || y + h > height || level != 0 || layer != 0 {
                            continue;
                        }
                        let (columns, rows) = (w / step, h / step);
                        let mut image = format!("P5\n{columns} {rows}\n65535\n").into_bytes();
                        // Row by row: a whole large map is never held at once.
                        for row in 0..rows {
                            let bytes = backend.renderer.read_target_region_bytes(
                                key,
                                size,
                                Some([x, y + row * step, w, 1]),
                            )?;
                            for column in 0..columns {
                                let at = (column * step * size) as usize;
                                let value = depth_of(&bytes[at..at + size as usize]);
                                let value = (value.clamp(0.0, 1.0) * 65535.0).round() as u16;
                                image.extend_from_slice(&value.to_be_bytes());
                            }
                        }
                        let directory = std::path::Path::new(&directory);
                        let _ = std::fs::create_dir_all(directory);
                        let _ = std::fs::write(
                            directory.join(format!("trace-{key:x}-{present}-{x}-{y}.pgm")),
                            image,
                        );
                    }
                }
                eprintln!(
                    "target trace present {present} {what} wrote {key:x} level {level} layer {layer} {width}x{height} {format:?}: too large for a full readback (rectangles only)"
                );
                continue;
            }
            let texels = match backend.renderer.read_target_slice(key, level, layer) {
                Ok(texels) => texels,
                Err(error) => {
                    eprintln!(
                        "target trace present {present} {what} wrote {key:x} level {level} layer {layer} {width}x{height} {format:?}: not read back: {error}"
                    );
                    continue;
                }
            };
            // A volume: also the mean first channel of a few slices
            // (the trace's other figures are slice 0's).
            let slices = if depth > 1 {
                let mut means = Vec::new();
                for slice in [1, depth / 8, depth / 2, depth - 1] {
                    let Ok(texels) = backend.renderer.read_target_slice(key, 0, slice) else {
                        continue;
                    };
                    let sum: f64 = texels.iter().map(|texel| f64::from(texel[0])).sum();
                    means.push(format!("{slice}:{:.4}", sum / texels.len().max(1) as f64));
                }
                format!(" depth {depth} slices {}", means.join(" "))
            } else {
                String::new()
            };
            // ECHORA_TARGET_TRACE_RECT=<x>,<y>,<w>,<h>[;...] (texels): also
            // those rectangles' first channels, row by row, in targets that
            // hold them (a shadow map's texels around a lookup).
            if let Ok(rects) = super::debug_var("ECHORA_TARGET_TRACE_RECT") {
                for rect in rects.split(';') {
                    let values: Vec<usize> = rect
                        .split(',')
                        .filter_map(|value| value.trim().parse().ok())
                        .collect();
                    let [x, y, w, h] = values[..] else {
                        continue;
                    };
                    if x + w > width as usize || y + h > height as usize {
                        continue;
                    }
                    for row in y..y + h {
                        let line: Vec<String> = (x..x + w)
                            .map(|column| {
                                format!("{:.6}", texels[row * width as usize + column][0])
                            })
                            .collect();
                        eprintln!(
                            "target trace present {present} {what} {key:x} row {row} from {x}: {}",
                            line.join(" ")
                        );
                    }
                }
            }
            // ECHORA_TARGET_TRACE_PGM=<x>,<y>,<w>,<h>,<step>[;...] with
            // ECHORA_DUMP_DIR: those regions' first channels (every step-th
            // texel, 0..1 as 0..65535) as 16-bit PGM images,
            // trace-<key>-<present>-<x>-<y>.pgm (a shadow map's casters).
            if let (Ok(regions), Some(directory)) = (
                super::debug_var("ECHORA_TARGET_TRACE_PGM"),
                super::debug_var_os("ECHORA_DUMP_DIR"),
            ) {
                for region in regions.split(';') {
                    let values: Vec<usize> = region
                        .split(',')
                        .filter_map(|value| value.trim().parse().ok())
                        .collect();
                    let [x, y, w, h, step] = values[..] else {
                        continue;
                    };
                    let step = step.max(1);
                    if x + w > width as usize || y + h > height as usize {
                        continue;
                    }
                    let (columns, rows) = (w / step, h / step);
                    let mut image = format!("P5\n{columns} {rows}\n65535\n").into_bytes();
                    for row in 0..rows {
                        for column in 0..columns {
                            let texel =
                                texels[(y + row * step) * width as usize + x + column * step][0];
                            let value = (texel.clamp(0.0, 1.0) * 65535.0).round() as u16;
                            image.extend_from_slice(&value.to_be_bytes());
                        }
                    }
                    let directory = std::path::Path::new(&directory);
                    let _ = std::fs::create_dir_all(directory);
                    let _ = std::fs::write(
                        directory.join(format!("trace-{key:x}-{present}-{x}-{y}.pgm")),
                        image,
                    );
                }
            }
            let count = texels.len().max(1) as f64;
            let nonfinite = texels
                .iter()
                .filter(|texel| texel.iter().any(|part| !part.is_finite()))
                .count();
            let mut mean = [0f64; 4];
            for texel in texels
                .iter()
                .filter(|texel| texel.iter().all(|part| part.is_finite()))
            {
                for (sum, part) in mean.iter_mut().zip(texel) {
                    *sum += f64::from(*part) / count;
                }
            }
            // ECHORA_TARGET_TRACE_PIXEL=<x>,<y> (fractions of the width and
            // height): also the texel there, the same place in targets of
            // every size (follows one point of the picture through passes).
            let pixel = super::debug_var("ECHORA_TARGET_TRACE_PIXEL")
                .ok()
                .and_then(|value| {
                    let (x, y) = value.split_once(',')?;
                    Some((x.parse::<f64>().ok()?, y.parse::<f64>().ok()?))
                })
                .and_then(|(x, y)| {
                    let column = ((x * f64::from(width)) as usize).min(width as usize - 1);
                    let row = ((y * f64::from(height)) as usize).min(height as usize - 1);
                    texels.get(row * width as usize + column)
                })
                .map_or(String::new(), |texel| format!(" pixel {texel:?}"));
            eprintln!(
                "target trace present {present} {what} wrote {key:x} level {level} layer {layer} {width}x{height} {format:?} @{:x}: mean {:.4} {:.4} {:.4} {:.4}, {nonfinite} not finite{pixel}{slices}",
                storages[index], mean[0], mean[1], mean[2], mean[3]
            );
        }
        Ok(())
    }

    /// Pool sharing at creation needs the machine's checked memory mappings.
    pub(crate) fn share_pools_at_init(&self) -> bool {
        self.gpu
            .as_ref()
            .is_some_and(|backend| backend.renderer.gpu.dma_buf_import.is_some())
            && super::debug_var_os("ECHORA_SHARE_AT_INIT").is_none_or(|value| value != "0")
    }

    pub(super) fn share_new_pools(&mut self, core: &Guest) {
        // ECHORA_POOL_LOG (debugging): every 300th present, the memory
        // pools a GPU page table would cover: count, bytes by CPU access
        // (uncached, cached, none) and the GPU address range.
        if super::debug_var_os("ECHORA_POOL_LOG").is_some()
            && self.presented_frames.is_multiple_of(300)
        {
            let (mut count, mut bytes, mut low, mut high) = (0u64, [0u64; 3], u64::MAX, 0u64);
            for pool in self
                .objects
                .values()
                .filter(|object| object.kind == "MemoryPool")
            {
                let Some(storage) = pool.builder.get("Storage") else {
                    continue;
                };
                let flags = pool.builder.get("Flags").map_or(0x21, |flags| flags.x[0]);
                let class = if flags & 2 != 0 {
                    0
                } else if flags & 4 != 0 {
                    1
                } else {
                    2
                };
                count += 1;
                bytes[class] += storage.x[1];
                if let Some(gpu) = pool.state.get("GpuAddress").map(|address| address.x[0]) {
                    low = low.min(gpu);
                    high = high.max(gpu + storage.x[1]);
                }
            }
            eprintln!(
                "pools present {}: {count} pools, uncached {} MiB, cached {} MiB, gpu-only {} MiB, gpu {low:#x}..{high:#x} ({} MiB span)",
                self.presented_frames,
                bytes[0] >> 20,
                bytes[1] >> 20,
                bytes[2] >> 20,
                high.saturating_sub(low) >> 20
            );
        }
        let Some(backend) = self.gpu.as_ref() else {
            return;
        };
        if backend.renderer.gpu.dma_buf_import.is_none() {
            self.new_pools.clear();
            return;
        }
        // ECHORA_POOL_TRACE=<guest base> (debugging): that pool's sharing,
        // at each present and each page table build.
        let traced = super::debug_var("ECHORA_POOL_TRACE")
            .ok()
            .and_then(|value| u64::from_str_radix(value.trim_start_matches("0x"), 16).ok());
        if let Some(traced) = traced {
            let object = self.objects.values().find(|object| {
                object.kind == "MemoryPool"
                    && object.builder.get("Storage").map(|storage| storage.x[0]) == Some(traced)
            });
            eprintln!(
                "pool trace present {}: object {}, storage {:x?}, recorded {:x?}, pieces {}, whole span {:x?}",
                self.presented_frames,
                object.is_some(),
                object
                    .and_then(|object| object.builder.get("Storage"))
                    .map(|storage| (storage.x[0], storage.x[1])),
                backend.shared_pools.get(&traced),
                backend.shared_pieces.contains_key(&traced),
                object
                    .and_then(|object| object.builder.get("Storage"))
                    .and_then(|storage| core.memory.shared_span(storage.x[0], storage.x[1])),
            );
        }
        let scanned = std::time::Instant::now();
        // Only pools made since the last present: the others were shared
        // (or found not shareable) then, and a pool made again over the
        // same memory is made anew.
        let new_pools = std::mem::take(&mut self.new_pools);
        let pools: Vec<(u64, u64)> = new_pools
            .iter()
            .filter_map(|address| self.objects.get(address))
            .filter(|object| object.kind == "MemoryPool")
            .filter_map(|pool| {
                pool.builder
                    .get("Storage")
                    .map(|storage| (storage.x[0], storage.x[1]))
            })
            // A pool at a known base whose size changed (made again over
            // the same memory, larger) is shared again.
            .filter(|&(base, size)| {
                base != 0
                    && size != 0
                    && backend
                        .shared_pools
                        .get(&base)
                        .is_none_or(|&(shared, ..)| shared != size)
                    && !backend.shared_pieces.contains_key(&base)
            })
            .collect();
        let backend = self.gpu.as_mut().unwrap();
        backend.stat("stat: pools tried for sharing", pools.len() as u64);
        backend.stat(
            "stat: present us scanning pools",
            scanned.elapsed().as_micros() as u64,
        );
        backend.stat("stat: pools scanned", new_pools.len() as u64);
        for (base, size) in pools {
            if let Some((file, offset)) = core.memory.shared_span(base, size) {
                let backend = self.gpu.as_mut().unwrap();
                backend.stat("stat: pools shared whole", 1);
                backend.shared_pools.insert(base, (size, file, offset));
                backend.record_pool_share(base);
                continue;
            }
            // Page by page: runs of pages whose backing follows on.
            use echora_shader_abi::PAGE_SIZE;
            let mut pieces: Vec<(u64, u64, i32, u64)> = Vec::new();
            let mut offset = 0;
            while offset < size {
                let length = (size - offset).min(PAGE_SIZE);
                if let Some((file, at)) = core.memory.shared_span(base + offset, length) {
                    match pieces.last_mut() {
                        Some(last)
                            if last.2 == file
                                && last.0 + last.1 == offset
                                && last.3 + last.1 == at =>
                        {
                            last.1 += length;
                        }
                        _ => pieces.push((offset, length, file, at)),
                    }
                }
                offset += length;
            }
            let backend = self.gpu.as_mut().unwrap();
            backend.stat("stat: pools shared in pieces", 1);
            backend.shared_pieces.insert(base, pieces);
            backend.record_pool_share(base);
        }
    }

    /// The file bytes that back a guest range represented by one known
    /// in-place span. Comparing file offsets covers aliases of those pages.
    fn shared_guest_range(&self, guest: u64, length: u64) -> Option<SharedRange> {
        self.shared_guest_index()?.range(guest, length)
    }

    fn shared_guest_index(&self) -> Option<std::cell::RefMut<'_, GuestFileIndex>> {
        let backend = self.gpu.as_ref()?;
        let mut index = backend.shared_guest_index.borrow_mut();
        if index.generation != backend.share_generation {
            let whole = backend
                .shared_pools
                .iter()
                .map(|(&base, &(length, file, offset))| (base, length, file, offset));
            let pieces = backend.shared_pieces.iter().flat_map(|(&base, spans)| {
                spans
                    .iter()
                    .map(move |&(at, length, file, offset)| (base + at, length, file, offset))
            });
            *index = GuestFileIndex::new(backend.share_generation, whole.chain(pieces));
        }
        Some(index)
    }

    /// Match the bound page table to the guest file before relying on
    /// shared memory. The same guest range can also have a separate
    /// device-local buffer whose writes must be copied back.
    fn shared_table_range(&self, gpu: u64, guest: u64, length: u64) -> Option<SharedRange> {
        use echora_shader_abi::{PAGE_SHIFT, PAGE_SIZE};
        let backend = self.gpu.as_ref()?;
        let (key, table) = backend.page_table?;
        if key != (backend.share_generation, self.pool_epoch) {
            return None;
        }
        let shared = self.shared_guest_range(guest, length)?;
        let entries = backend.renderer.page_table_entries();
        let mut done = 0;
        while done < length {
            let address = gpu.checked_add(done)?;
            let page = (address >> PAGE_SHIFT).checked_sub(u64::from(table.first_page))?;
            let inside = address & (PAGE_SIZE - 1);
            let device = *entries.get(usize::try_from(page).ok()?)?;
            let bytes = (length - done).min(PAGE_SIZE - inside);
            if device == 0
                || !backend.renderer.shares_device_range(
                    shared.file,
                    shared.start + done,
                    bytes,
                    device + inside,
                )
            {
                return None;
            }
            done += bytes;
        }
        Some(shared)
    }

    /// Bound writes to shared guest memory, first using the dispatch's
    /// invocation counts, then the full ranges of its uniform pointers.
    fn compute_shared_writes(
        &self,
        stage: &Arc<Translated>,
        groups: Option<[u32; 3]>,
        mut constant: impl FnMut(u8, u16) -> Option<u32>,
    ) -> Option<BoundedWrites> {
        let backend = self.gpu.as_ref()?;
        let key = Arc::as_ptr(stage) as usize;
        if let Some(groups) = groups
            && let Some((plan, local)) = backend.global_writes.get(&key)
            && let Some(exact) = plan.intervals(groups, *local, &mut constant)
            && let Some(shared) = exact
                .into_iter()
                .map(|range| {
                    let length = range.end - range.start;
                    let guest = self.gpu_to_guest(range.start, length).ok()?;
                    self.shared_table_range(range.start, guest, length)
                })
                .collect::<Option<Vec<_>>>()
        {
            return Some(shared.into());
        }

        let mut virtual_ranges = backend.uniform_writes.get(&key)?.intervals(constant)?;
        let (stamp, _) = backend.page_table?;
        if stamp != (backend.share_generation, self.pool_epoch) {
            return None;
        }
        merge_ranges(&mut virtual_ranges);
        Some(BoundedWrites {
            shared: backend.project_shared_writes(&virtual_ranges)?,
            virtual_ranges,
        })
    }

    /// The buffer and offset that share `length` guest bytes at `guest` with
    /// the GPU in place, when they lie in a shared memory pool, or in one
    /// piece of a pool shared in pieces, and the span imports; None
    /// otherwise (the caller copies).
    fn shared_binding(
        &mut self,
        guest: u64,
        length: u64,
    ) -> Result<Option<(vk::Buffer, u64)>, String> {
        let Some(range) = self.shared_guest_range(guest, length) else {
            return Ok(None);
        };
        self.gpu
            .as_mut()
            .unwrap()
            .renderer
            .import_file(range.file, range.start, length)
    }

    /// Where the GPU reads `length` bytes of indirect arguments at guest
    /// `guest` from: the pool's GPU copy when a GPU write through a binding
    /// made it newer there, otherwise guest memory itself, shared in place,
    /// for a CPU-visible pool (what the CPU wrote, and what programs stored
    /// through the page table, which reaches such pools in guest memory and
    /// not their copy: a culling pass's counts were read from the stale
    /// copy, and foliage and particles were not drawn); the copy otherwise.
    fn indirect_arguments(
        &mut self,
        core: &mut Guest,
        guest: u64,
        length: u64,
    ) -> Result<Option<(vk::Buffer, u64)>, String> {
        let range = guest..guest + length;
        // A storage binding may have written to a temporary GPU buffer.
        // Its results must reach guest memory before choosing an argument
        // source, for draws and dispatches alike.
        self.resolve_copies(core, Some(range.clone()))?;
        let copy_newer = self.gpu.as_ref().unwrap().resident.values().any(|pool| {
            pool.contains(guest)
                && pool
                    .gpu_dirty
                    .iter()
                    .any(|dirty| dirty.start < range.end && range.start < dirty.end)
        });
        if !copy_newer {
            if let Some(shared) = self.shared_binding(guest, length)? {
                self.gpu
                    .as_mut()
                    .unwrap()
                    .stat("stat: indirect arguments read in place", 1);
                return Ok(Some(shared));
            }
        }
        self.resident_pool_binding(core, guest, length)
    }

    /// The device buffer and offset for `length` storage bytes at guest
    /// `guest`: their memory pool, mirrored on the GPU (created and filled
    /// on first use, with the ranges the CPU changed since uploaded first).
    /// `None` outside a pool with storage of its own.
    fn resident_pool_binding(
        &mut self,
        core: &mut Guest,
        guest: u64,
        length: u64,
    ) -> Result<Option<(vk::Buffer, u64)>, String> {
        self.resident_pool_binding_for(core, guest, length, false)
    }

    /// `resident_pool_binding` for GPU work that writes all `length` bytes
    /// before anything reads them (a counter report) when `overwritten`:
    /// the guest's bytes there are neither stored back nor uploaded first.
    fn resident_pool_binding_for(
        &mut self,
        core: &mut Guest,
        guest: u64,
        length: u64,
        overwritten: bool,
    ) -> Result<Option<(vk::Buffer, u64)>, String> {
        // A pool already mirrored first (no scan of every NVN object); a
        // pool finalized since is dropped with its buffer key reused only
        // by a pool at the same address and size.
        let known = self
            .gpu
            .as_ref()
            .unwrap()
            .resident
            .values()
            .find(|pool| pool.base <= guest && guest + length <= pool.base + pool.size)
            .map(|pool| {
                let flags = if pool.uncached {
                    2
                } else if pool.host_visible {
                    4
                } else {
                    0
                };
                (pool.base, pool.size, flags)
            });
        let Some((base, size, flags)) = known.or_else(|| {
            self.memory_pools
                .iter()
                .filter_map(|address| self.objects.get(address))
                .filter(|object| object.kind == "MemoryPool")
                .filter_map(|pool| {
                    let storage = pool.builder.get("Storage")?;
                    let flags = pool.builder.get("Flags").map_or(0, |flags| flags.x[0]);
                    Some((storage.x[0], storage.x[1], flags))
                })
                .find(|&(base, size, _)| {
                    base != 0 && base <= guest && guest + length <= base + size
                })
        }) else {
            return Ok(None);
        };
        // A CPU-uncached pool is coherent on the console: the CPU writes it
        // without flushing (a list it builds for a GPU pass to walk). Its
        // copy here takes the CPU's bytes for the bound range before each
        // binding, after any GPU-newer bytes there are stored back.
        let uncached = flags & 2 != 0 && flags & 4 == 0;
        // A range the GPU wrote last and the CPU has not read back since
        // (an occlusion report a condition reads) is newest in the GPU
        // copy already: neither copied back nor uploaded over again (each
        // round trip is a submission and a wait, hundreds a frame).
        let gpu_newer = self
            .gpu
            .as_ref()
            .unwrap()
            .resident
            .get(&base)
            .is_some_and(|pool| {
                let at = pool.gpu_dirty.partition_point(|dirty| dirty.end <= guest);
                pool.gpu_dirty
                    .get(at)
                    .is_some_and(|dirty| dirty.start <= guest && guest + length <= dirty.end)
            });
        if uncached && !gpu_newer && !overwritten {
            self.resolve_resident(core, Some(guest..guest + length))?;
        }
        let backend = self.gpu.as_mut().unwrap();
        let (buffer, created) = backend.renderer.resident_buffer(base, size)?;
        let fresh = !backend.resident.contains_key(&base);
        let pool = backend
            .resident
            .entry(base)
            .or_insert_with(|| ResidentPool {
                base,
                size,
                uncached: flags & 2 != 0,
                host_visible: flags & 6 != 0,
                guest_dirty: Vec::new(),
                gpu_dirty: Vec::new(),
            });
        let mut dropped_dirty = false;
        if created || fresh || pool.size != size {
            dropped_dirty = !pool.gpu_dirty.is_empty();
            *pool = ResidentPool {
                base,
                size,
                uncached: flags & 2 != 0,
                host_visible: flags & 6 != 0,
                guest_dirty: vec![base..base + size],
                gpu_dirty: Vec::new(),
            };
        }
        // Its GPU-dirty ranges went with the old mirror: the index rebuilds.
        if dropped_dirty {
            backend.dirty_removals += 1;
        }
        if overwritten {
            remove_range(&mut pool.guest_dirty, guest..guest + length);
        } else if uncached && !gpu_newer {
            add_range(&mut pool.guest_dirty, guest..guest + length);
        }
        // Only what this binding can read goes up; the pool's other dirty
        // ranges wait for a binding over them (a 16-byte report used to
        // bring up everything flushed anywhere in its pool).
        // A GPU-only pool (the page table reaches it through this copy, not
        // through guest memory) takes all of them, as before.
        let bound = if flags & 6 == 0 {
            base..base + size
        } else {
            guest..guest + length
        };
        let uploads = ranges_within(&pool.guest_dirty, bound.clone());
        if !uploads.is_empty() {
            remove_range(&mut pool.guest_dirty, bound);
        }
        for range in uploads {
            let length = range.end - range.start;
            let backend = self.gpu.as_mut().unwrap();
            backend.stat("stat: dispatch storage bytes", length);
            // Guest bytes go straight into the staging ring.
            backend.renderer.upload_resident_with(
                base,
                range.start - base,
                length,
                |done, piece| {
                    read_storage_into(core, range.start + done, piece);
                },
            )?;
        }
        Ok(Some((buffer, guest - base)))
    }

    /// ResetCounter(SAMPLES_PASSED) on the GPU (`Renderer::samples_reset`).
    pub(super) fn gpu_samples_reset(&mut self) -> Result<(), String> {
        self.gpu.as_mut().unwrap().renderer.samples_reset()
    }

    /// ReportCounter(SAMPLES_PASSED, `address`) on the GPU, into the device
    /// copy of the resident pool holding the address (later CPU reads of
    /// it store that back). False when no pool holds it.
    pub(super) fn gpu_samples_report(
        &mut self,
        core: &mut Guest,
        address: u64,
        timestamp: u64,
    ) -> Result<bool, String> {
        let guest = self.gpu_to_guest(address, 16)?;
        // The report writes all 16 bytes (the count and the timestamp).
        let Some((buffer, offset)) = self.resident_pool_binding_for(core, guest, 16, true)? else {
            return Ok(false);
        };
        let backend = self.gpu.as_mut().unwrap();
        backend.renderer.samples_report(buffer, offset, timestamp)?;
        if let Some(pool) = backend
            .resident
            .values_mut()
            .find(|pool| pool.base <= guest && guest + 16 <= pool.base + pool.size)
        {
            add_range(&mut pool.gpu_dirty, guest..guest + 16);
            add_range(&mut backend.dirty_index.get_mut().1, guest..guest + 16);
            backend.gpu_writes += 1;
        }
        backend.samples_reports.insert(address);
        Ok(true)
    }

    /// SetRenderEnableConditional(`mode`, `address`) on the GPU when the
    /// counter there is a SAMPLES_PASSED report (`gpu_samples_report`) and
    /// the one 16 bytes on is CPU memory holding 0, as the game's
    /// occlusion conditions are: rendering then holds while the report's
    /// count is nonzero (mode 1, differ) or zero (mode 0, equal), read on
    /// the GPU in command order. False otherwise (the CPU evaluates it).
    pub(super) fn gpu_condition(
        &mut self,
        core: &mut Guest,
        mode: u64,
        address: u64,
    ) -> Result<bool, String> {
        let backend = self.gpu.as_ref().unwrap();
        if backend.renderer.supports_conditions()
            && backend.samples_reports.contains(&address)
            && !backend.samples_reports.contains(&(address + 16))
            && mode <= 1
        {
            let other = self.gpu_to_guest(address + 16, 8)?;
            self.resolve_draw_input(core, other, 8)?;
            if core
                .memory
                .read_u64(other)
                .map_err(|trap| format!("{trap:?}"))?
                == 0
            {
                let guest = self.gpu_to_guest(address, 8)?;
                if let Some((buffer, offset)) = self.resident_pool_binding(core, guest, 8)? {
                    let renderer = &mut self.gpu.as_mut().unwrap().renderer;
                    renderer.set_condition(Some((buffer, offset, mode == 0)));
                    return Ok(true);
                }
            }
        }
        self.gpu_clear_condition();
        Ok(false)
    }

    /// Rendering is no longer gated on the GPU.
    pub(super) fn gpu_clear_condition(&mut self) {
        if let Some(backend) = self.gpu.as_mut() {
            backend.renderer.set_condition(None);
        }
    }

    /// Store deferred dispatch results (storage copied back from the GPU)
    /// that overlap `range` (all of them for `None`).
    fn resolve_copies(
        &mut self,
        core: &mut Guest,
        range: Option<std::ops::Range<u64>>,
    ) -> Result<(), String> {
        let Some(backend) = self.gpu.as_mut() else {
            return Ok(());
        };
        if backend.pending_storage.is_empty() {
            return Ok(());
        }
        if let Some(range) = &range {
            let overlaps = backend
                .pending_storage
                .iter()
                .any(|(_, guest, ..)| guest.start < range.end && range.start < guest.end);
            if !overlaps {
                return Ok(());
            }
        }
        let reason = if range.is_some() {
            "stat: dispatch results stored for a GPU read"
        } else {
            "stat: dispatch results stored at a CPU wait"
        };
        backend.stat(reason, 1);
        if backend.renderer.reads_pending() {
            backend.renderer.flush()?;
        }
        let pending = std::mem::take(&mut backend.pending_storage);
        let ids: Vec<u64> = pending.iter().map(|(token, ..)| *token).collect();
        let mut completed = backend.renderer.take_reads(&ids);
        for (token, guest, binding, program, base) in pending {
            let Some(bytes) = completed.remove(&token) else {
                return Err(format!("dispatch result {token} never completed"));
            };
            if bytes.len() <= 1024 && super::debug_var_os("ECHORA_GPU_FLOW").is_some() {
                let words: Vec<String> = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|word| u32::from_le_bytes(*word).to_string())
                    .collect();
                let line = format!(
                    "flow dispatch {program:016x} stored {binding} @{base:x} [{}]",
                    words.join(" ")
                );
                self.sequence_note(&line);
            }
            core.memory
                .write(guest.start, &bytes)
                .map_err(|trap| format!("storing compute results: {trap:?}"))?;
            // Textures over the written memory (a histogram the next pass
            // reads as a texture, say) are checked again before their next
            // use, even within this frame.
            self.gpu_invalidate(guest, None);
        }
        Ok(())
    }

    /// Execute a draw command on the GPU.
    pub(super) fn gpu_draw(&mut self, core: &mut Guest, command: &Command) -> Result<(), String> {
        let x = &command.x;
        let empty = match command.name {
            "nvnCommandBufferDrawArrays"
            | "nvnCommandBufferDrawElements"
            | "nvnCommandBufferDrawElementsBaseVertex" => x[3] as u32 == 0,
            "nvnCommandBufferDrawArraysInstanced" => x[3] as u32 == 0 || x[5] as u32 == 0,
            "nvnCommandBufferDrawElementsInstanced" => x[3] as u32 == 0 || x[7] as u32 == 0,
            "nvnCommandBufferDrawMeshTasks" => x[2] as u32 == 0,
            _ => false,
        };
        if empty {
            if let Some(backend) = &mut self.gpu {
                backend.stat("stat: empty draws", 1);
            }
            return Ok(());
        }
        let mut bindings = std::mem::take(&mut self.draw_bindings);
        let result = self.gpu_draw_with_bindings(core, command, &mut bindings);
        bindings.uniforms.clear();
        bindings.guest_uniforms.clear();
        bindings.textures.clear();
        self.draw_bindings = bindings;
        result
    }

    fn gpu_draw_with_bindings(
        &mut self,
        core: &mut Guest,
        command: &Command,
        bindings: &mut DrawBindings,
    ) -> Result<(), String> {
        // Time spent in each phase (stats, microseconds), measured only
        // when GPU timing or the timeline is on: the clock reads were ~4%
        // of the replay thread's time.
        let timing = draw_phase_timing();
        let phase = |nvn: &mut Self, name: &'static str, since: &mut Option<std::time::Instant>| {
            let Some(previous) = since.as_mut() else {
                return;
            };
            let now = std::time::Instant::now();
            nvn.gpu
                .as_mut()
                .unwrap()
                .stat(name, (now - *previous).as_micros() as u64);
            *previous = now;
        };
        let mut since = timing.then(std::time::Instant::now);
        let mesh_draw = command.name.contains("DrawMeshTasks");
        let program = if mesh_draw {
            self.bound_mesh_program(core)?
        } else {
            self.bound_program(core)?
        };
        let keys = program.keys;
        let stages = &*program.stages;
        let mesh_groups = if mesh_draw {
            let groups = self.mesh_groups(core, command)?;
            if matches!(groups, MeshGroups::Direct(groups) if groups.contains(&0)) {
                return Ok(());
            }
            Some(groups)
        } else {
            None
        };
        phase(self, "stat: draw us program", &mut since);
        // ECHORA_GPU_SEQUENCE_AFTER=<fragment key>: the present after the
        // first draw of that program is sequenced (scenes come at frames
        // that vary from run to run).
        if self.sequence_chosen.is_none()
            && super::debug_var("ECHORA_GPU_SEQUENCE_AFTER")
                .is_ok_and(|wanted| wanted == format!("{:016x}", keys.1))
        {
            self.sequence_chosen = Some(self.presented_frames + 1);
            eprintln!("sequencing present {}", self.presented_frames + 1);
        }
        // ECHORA_GPU_SKIP=<fragment key>[,...] (debugging): those
        // programs' draws are not rendered.
        if super::debug_var("ECHORA_GPU_SKIP").is_ok_and(|skipped| {
            skipped
                .split(',')
                .any(|key| key == format!("{:016x}", keys.1))
        }) {
            return Ok(());
        }
        // ECHORA_GPU_SNAP=<target>: during the sequenced present, that
        // target after each draw into it, as snap-<sequence>.ppm (x/(1+x)).
        let snap = super::debug_var("ECHORA_GPU_SNAP")
            .ok()
            .and_then(|key| u64::from_str_radix(&key, 16).ok())
            .zip(super::debug_var_os("ECHORA_GPU_DUMP"))
            .filter(|_| self.sequencing());
        // The translation identity includes the vertex input types, mesh
        // control state and any outputs the previous stage provides.
        let built = program.pipeline;
        let traced_variant = GraphicsProgramKey {
            traced: true,
            ..built
        };
        let backend = self.gpu.as_mut().unwrap();
        if !backend.programs.contains_key(&built) {
            // ECHORA_VERTEX_TRACE: each traced stage's record buffer (64
            // records of 128 words), its address in the stage's constants.
            let mut specialization = Vec::new();
            for (index, stage) in stages.iter().enumerate() {
                let mut values = Vec::new();
                for constant in &stage.spec_constants {
                    // ECHORA_FRAGMENT_TRACE_PIXEL=X,Y (debugging): the
                    // pixel a traced fragment program records (none
                    // without it).
                    if constant.meaning == echora_shader_abi::SpecMeaning::FragmentTracePixel {
                        let pixel = super::debug_var("ECHORA_FRAGMENT_TRACE_PIXEL")
                            .ok()
                            .and_then(|value| {
                                let (x, y) = value.split_once(',')?;
                                Some(
                                    x.trim().parse::<u32>().ok()?
                                        | y.trim().parse::<u32>().ok()? << 16,
                                )
                            })
                            .unwrap_or(u32::MAX);
                        values.push((constant.id, pixel));
                        continue;
                    }
                    // ECHORA_LOOP_TRACE: the stage's loop counters, as a
                    // dispatch's (`draw_loop_trace_key`).
                    if let meaning @ (echora_shader_abi::SpecMeaning::LoopTraceLow
                    | echora_shader_abi::SpecMeaning::LoopTraceHigh) = constant.meaning
                    {
                        let trace = draw_loop_trace_key(keys, index);
                        let address = match backend.loop_traces.get(&trace) {
                            Some(&(_, address, _)) => address,
                            None => {
                                let (buffer, address) = backend.renderer.debug_buffer(64 << 10)?;
                                backend.renderer.mark_loop_trace(
                                    buffer,
                                    format!("draw {:016x}/{:016x} stage {index}", keys.0, keys.1),
                                );
                                backend
                                    .loop_traces
                                    .insert(trace, (buffer, address, self.presented_frames));
                                address
                            }
                        };
                        let half = if meaning == echora_shader_abi::SpecMeaning::LoopTraceLow {
                            address as u32
                        } else {
                            (address >> 32) as u32
                        };
                        values.push((constant.id, half));
                        continue;
                    }
                    let low = match constant.meaning {
                        echora_shader_abi::SpecMeaning::VertexTraceLow => true,
                        echora_shader_abi::SpecMeaning::VertexTraceHigh => false,
                        _ => continue,
                    };
                    let trace = (keys.0, keys.1, index);
                    let address = match backend.vertex_traces.get(&trace) {
                        Some(&(buffer, _)) => backend.vertex_trace_addresses[buffer],
                        None => {
                            let (buffer, address) = backend.renderer.debug_buffer(32 << 10)?;
                            backend
                                .vertex_traces
                                .insert(trace, (buffer, self.presented_frames));
                            if backend.vertex_trace_addresses.len() <= buffer {
                                backend.vertex_trace_addresses.resize(buffer + 1, 0);
                            }
                            backend.vertex_trace_addresses[buffer] = address;
                            address
                        }
                    };
                    let half = if low {
                        address as u32
                    } else {
                        (address >> 32) as u32
                    };
                    values.push((constant.id, half));
                }
                specialization.push(values);
            }
            // ECHORA_VERTEX_TRACE_BASE=N: the program traces nothing; draws
            // with base vertex N use a variant that does (`traced_variant`),
            // so each program's records come from the same patch.
            let untraced: Vec<Vec<(u32, u32)>> = specialization
                .iter()
                .map(|values| values.iter().map(|&(id, _)| (id, 0)).collect())
                .collect();
            let base_filter = super::debug_var_os("ECHORA_VERTEX_TRACE_BASE").is_some();
            if super::debug_var_os("ECHORA_SHADER_STATS").is_some() {
                eprintln!("shader stats: draw {:016x}/{:016x}", keys.0, keys.1);
            }
            let program = backend.renderer.create_program_specialized(
                stages.to_vec(),
                if base_filter {
                    &untraced
                } else {
                    &specialization
                },
            )?;
            backend.programs.insert(built, program);
            if base_filter {
                eprintln!(
                    "vertex trace program {:016x}/{:016x} trace constants per stage {:?}",
                    keys.0,
                    keys.1,
                    specialization.iter().map(Vec::len).collect::<Vec<_>>()
                );
            }
            if base_filter && specialization.iter().any(|values| !values.is_empty()) {
                let traced = backend
                    .renderer
                    .create_program_specialized(stages.to_vec(), &specialization)?;
                backend.programs.insert(traced_variant, traced);
            }
        }
        // ECHORA_LOOP_TRACE: every 60 presents, each traced stage's loops
        // that ran, as for dispatches.
        for index in 0..stages.len() {
            let trace = draw_loop_trace_key(keys, index);
            let Some(&(buffer, address, reported)) = backend.loop_traces.get(&trace) else {
                continue;
            };
            if self.presented_frames < reported + 60 {
                continue;
            }
            let words: Vec<u32> = backend
                .renderer
                .read_debug_buffer(buffer)
                .chunks_exact(4)
                .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
                .collect();
            for (loop_index, entry) in words.chunks_exact(4).enumerate() {
                if entry[0] != 0 {
                    eprintln!(
                        "loop trace present {} draw {:016x}/{:016x} stage {index} loop {loop_index} at {:#x}: {} iterations, at most {} per invocation, {} stopped by the budget",
                        self.presented_frames,
                        keys.0,
                        keys.1,
                        entry[3].wrapping_sub(1),
                        entry[0],
                        entry[1],
                        entry[2]
                    );
                }
            }
            backend
                .loop_traces
                .insert(trace, (buffer, address, self.presented_frames));
        }
        // ECHORA_VERTEX_TRACE_REPORT=<stage key>[,...]: every 60 presents,
        // the records a traced program's stages wrote (from earlier draws).
        if let Ok(wanted) = super::debug_var("ECHORA_VERTEX_TRACE_REPORT")
            && wanted
                .split(',')
                .any(|key| key == format!("{:016x}", keys.0) || key == format!("{:016x}", keys.1))
            && super::debug_var("ECHORA_VERTEX_TRACE_REPORT_FROM")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .is_none_or(|from| self.presented_frames >= from)
        {
            for index in 0..stages.len() {
                let trace = (keys.0, keys.1, index);
                let Some(&(buffer, reported)) = backend.vertex_traces.get(&trace) else {
                    continue;
                };
                // ECHORA_VERTEX_TRACE_REPORT_EVERY=N presents (60).
                let every = super::debug_var("ECHORA_VERTEX_TRACE_REPORT_EVERY")
                    .ok()
                    .and_then(|value| value.parse::<u64>().ok())
                    .unwrap_or(60);
                if self.presented_frames < reported + every {
                    continue;
                }
                let words: Vec<u32> = backend
                    .renderer
                    .read_debug_buffer(buffer)
                    .chunks_exact(4)
                    .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
                    .collect();
                for (record, entry) in words.chunks_exact(128).enumerate() {
                    // Any record written (an invocation that left by an
                    // early exit has no "reached its end" bit).
                    if entry.iter().all(|word| *word == 0) {
                        continue;
                    }
                    let position: Vec<f32> = entry[4..8]
                        .iter()
                        .map(|bits| f32::from_bits(*bits))
                        .collect();
                    let mut line = format!(
                        "vertex trace present {} program {:016x}/{:016x} stage {index} record {record}: done {:#x} flags {:#x} ids {} {} position {position:?}",
                        self.presented_frames,
                        keys.0,
                        keys.1,
                        entry[0],
                        entry[1],
                        entry[2],
                        entry[3]
                    );
                    // A fragment program's registers r0..r3 at its end.
                    if entry[104..108].iter().any(|word| *word != 0) {
                        line += &format!(
                            " | out {:?}",
                            entry[104..108]
                                .iter()
                                .map(|bits| f32::from_bits(*bits))
                                .collect::<Vec<_>>()
                        );
                    }
                    for (index, texture) in entry[8..8 + 6 * 16].chunks_exact(6).enumerate() {
                        if texture[0] != 0 {
                            line += &format!(
                                " | tex at {:#x} x{} {:08x} {:08x} {:08x} {:08x}",
                                texture[0] - 1,
                                texture[1],
                                texture[2],
                                texture[3],
                                texture[4],
                                texture[5]
                            );
                            if entry[echora_shader_abi::VERTEX_TRACE_HANDLE_MASK] & (1 << index)
                                != 0
                            {
                                let handle =
                                    entry[echora_shader_abi::VERTEX_TRACE_HANDLE_WORD + index];
                                line += &format!(" handle {handle:#010x}");
                            }
                        }
                    }
                    eprintln!("{line}");
                }
                backend
                    .vertex_traces
                    .insert(trace, (buffer, self.presented_frames));
            }
        }

        // ECHORA_GPU_SKIP_DRAW=<stage key>[,...] (debugging): draws whose
        // program has one of those stages are not recorded.
        // ECHORA_GPU_SKIP_DRAW_DEPTH=<storage hex> (debugging): the skip
        // applies only to draws whose depth target is that memory (a
        // shadow map's casters, not the same programs' other passes).
        let skip_depth = super::debug_var("ECHORA_GPU_SKIP_DRAW_DEPTH")
            .ok()
            .and_then(|value| u64::from_str_radix(value.trim_start_matches("0x"), 16).ok());
        if super::debug_var("ECHORA_GPU_SKIP_DRAW").is_ok_and(|skipped| {
            skipped.split(',').any(|skip| {
                skip == format!("{:016x}", keys.0) || skip == format!("{:016x}", keys.1)
            })
        }) && skip_depth.is_none_or(|storage| {
            self.render.depth != 0 && self.texture_storage(self.render.depth) == Some(storage)
        }) {
            return Ok(());
        }
        let keys_for_note = [keys.0, keys.1];
        // Resolve color aliases before gathering sampled texture bindings.
        // Each color keeps its NVN target index as its attachment location
        // (a fragment program writes target n's registers to location n);
        // an unbound target between bound ones is a gap (0), trailing ones
        // are dropped.
        let bound = self
            .render
            .colors
            .iter()
            .rposition(|texture| *texture != 0)
            .map_or(0, |last| last + 1);
        let mut colors = [0; MAX_COLOR_TARGETS];
        let mut color_levels = [0; MAX_COLOR_TARGETS];
        let mut color_layers = [0; MAX_COLOR_TARGETS];
        let mut color_formats = [None; MAX_COLOR_TARGETS];
        let mips = super::mip_targets();
        for index in 0..bound {
            let texture = self.render.colors[index];
            let view = self.render.color_levels.get(index).copied().unwrap_or(0);
            color_layers[index] = (view >> 32) as u32;
            let level = if mips { view as u32 } else { 0 };
            let (target, attachment_level) = if texture == 0 {
                (0, 0)
            } else {
                self.resolve_color_target(texture, level)?
            };
            colors[index] = target;
            color_levels[index] = attachment_level;
            // An sRGB view encodes writes to a UNORM target. An integer
            // view one block wide per texel renders into compressed blocks.
            color_formats[index] = self.color_view_format(index);
        }
        let DrawBindings {
            uniforms,
            guest_uniforms,
            textures,
        } = bindings;
        uniforms.reserve(program.uniforms);
        guest_uniforms.reserve(program.uniforms);
        textures.reserve(program.textures);
        let mut storage = echora_gpu::FastMap::default();
        let mut storage_in_place = echora_gpu::FastMap::default();
        // Copied storage buffers a stage writes: guest address and GPU
        // address, by (stage index, binding).
        let mut storage_written = HashMap::default();
        // Driver constant entries of bound storage buffers: (stage index,
        // c0 offset, GPU address, size).
        let mut storage_entries: Vec<(usize, u32, u64, u64)> = Vec::new();
        let (mut uniform_time, mut texture_time) = Default::default();
        let mut uniform_bytes = 0u64;
        // Time per kind of binding (the statistics' "draw us binding ...").
        let mut binding_times = [std::time::Duration::ZERO; 5];
        let mut memo_hits = 0u64;
        let mut binding_lap: Option<(usize, std::time::Instant)> = None;
        for (index, stage) in stages.iter().enumerate() {
            let nvn = nvn_stage(stage.stage);
            for descriptor in &stage.descriptors {
                let now = timing.then(std::time::Instant::now);
                if let (Some((kind, since)), Some(now)) = (binding_lap, now) {
                    binding_times[kind] += now - since;
                }
                let kind = match descriptor.kind() {
                    DescriptorKind::UniformBuffer
                        if descriptor.index >= 3 && descriptor.index != 24 =>
                    {
                        0
                    }
                    DescriptorKind::UniformBuffer => 1,
                    DescriptorKind::StorageBuffer => 2,
                    DescriptorKind::Texture | DescriptorKind::TextureBuffer => 3,
                    _ => 4,
                };
                binding_lap = now.map(|now| (kind, now));
                match descriptor.kind() {
                    // Constant bank 24: root table 0 of the graphics
                    // pipeline (PushGraphicsConstants).
                    DescriptorKind::UniformBuffer if descriptor.index == 24 => {
                        let words = self.render.roots[0].first().copied().unwrap_or([0; 64]);
                        let bytes: Vec<u8> =
                            words.iter().flat_map(|word| word.to_le_bytes()).collect();
                        uniforms.insert((index, descriptor.binding), bytes);
                    }
                    DescriptorKind::UniformBuffer if descriptor.index >= 3 => {
                        let slot = descriptor.index - 3;
                        let bind = self
                            .render
                            .bound_uniforms
                            .get(u64::from(nvn), u64::from(slot))
                            .ok_or_else(|| {
                                // The slots that are bound, by stage, and
                                // the program's stages (which bank it reads).
                                let mut bound: Vec<&str> = self
                                    .render
                                    .named_uniform_bindings()
                                    .map(|(key, _)| key)
                                    .collect();
                                bound.sort_unstable();
                                format!(
                                    "uniform buffer {nvn}/{slot} is not bound (bound [{}], {:?} stage {:016x})",
                                    bound.join(" "),
                                    stage.stage,
                                    keys_for_note.get(index).copied().unwrap_or(0)
                                )
                            })?;
                        let used = u64::from(stage.cbuf_used_sizes[descriptor.index as usize]);
                        let size = bind.x[4].min(used.max(16)).min(64 << 10);
                        let address = bind.x[3];
                        // Read into the upload ring as the draw is recorded
                        // (nothing writes the range before then: what the
                        // GPU left there was just stored).
                        let started = timing.then(std::time::Instant::now);
                        let guest = self.gpu_to_guest(address, size)?;
                        if let Some(started) = started {
                            uniform_time += started.elapsed();
                        }
                        self.resolve_draw_input(core, guest, size)?;
                        uniform_bytes += size;
                        guest_uniforms.insert((index, descriptor.binding), (guest, size as usize));
                    }
                    DescriptorKind::UniformBuffer if descriptor.index == 1 => {
                        let mut bytes = self
                            .stage_constants_of(core, program.entries[index])
                            .unwrap_or_default();
                        bytes.resize(bytes.len().max(16).next_multiple_of(16), 0);
                        uniforms.insert((index, descriptor.binding), bytes);
                    }
                    DescriptorKind::UniformBuffer => {
                        let used = stage.cbuf_used_sizes[descriptor.index as usize].max(16);
                        // Every draw takes this path: the note is formatted
                        // only when tracing the flow.
                        if super::debug_var_os("ECHORA_GPU_FLOW").is_some() {
                            self.gpu.as_mut().unwrap().note(format!(
                                "driver constant buffer {} read as zeros ({used} bytes, {:?} stage {:016x})",
                                descriptor.index, stage.stage, keys_for_note.get(index).copied().unwrap_or(0)
                            ));
                        }
                        uniforms.insert((index, descriptor.binding), vec![0; used as usize]);
                    }
                    // Storage buffer i of graphics stage s (NVN numbering):
                    // its address and size at c0+0x110+0x100*s+16*i, 16 per
                    // stage (vertex shaders read c0+0x170 and fragment
                    // shaders, stage 4, c0+0x570 where the application binds
                    // slot 6).
                    // The contents are read now; graphics stages' writes are
                    // not returned.
                    DescriptorKind::StorageBuffer
                        if descriptor.cbuf_index == 0
                            && descriptor.cbuf_offset >= c0_storage_table(nvn)
                            && descriptor.cbuf_offset < c0_storage_table(nvn) + 0x100
                            && descriptor.cbuf_offset % 16 == 0 =>
                    {
                        let slot = (descriptor.cbuf_offset - c0_storage_table(nvn)) / 16;
                        let bind = self
                            .render
                            .state
                            .get(bind_key("BindStorageBuffer", Some(nvn as u64), slot as u64))
                            .ok_or_else(|| {
                                let bound: Vec<&str> = self
                                    .render
                                    .state
                                    .keys()
                                    .filter(|key| key.starts_with(&format!("BindStorageBuffer/{nvn}/")))
                                    .copied()
                                    .collect();
                                format!(
                                    "storage buffer {nvn}/{slot} (c0+{:#x}) is not bound (bound [{}])",
                                    descriptor.cbuf_offset,
                                    bound.join(" ")
                                )
                            })?;
                        let (gpu, size) = (bind.x[3], bind.x[4]);
                        let base = gpu & !15;
                        let length = gpu - base + size;
                        let guest = self.gpu_to_guest(base, length)?;
                        if descriptor.flags & 1 != 0 {
                            storage_written.insert((index, descriptor.binding), (guest, base));
                        }
                        storage_entries.push((index, descriptor.cbuf_offset, gpu, size));
                        // What dispatches wrote comes back to guest memory
                        // first, for the copy or the shared span alike.
                        self.resolve_storage(core, Some(guest..guest + length))?;
                        // A buffer the stage only reads is shared in place
                        // when its pool's file span imports (whole buffers
                        // copied per draw were most of the host's time after
                        // the movie). A buffer the stage writes keeps the
                        // copy, whose writes come back like a dispatch's
                        // (`resolve_storage`).
                        // ECHORA_DRAW_STORAGE_RESIDENT (measuring) reads it
                        // from the pool's GPU-memory copy instead, to compare
                        // the GPU's time reading system memory.
                        if descriptor.flags & 1 == 0 {
                            let in_place =
                                if super::debug_var_os("ECHORA_DRAW_STORAGE_RESIDENT").is_some() {
                                    self.resident_pool_binding(core, guest, length)?
                                } else {
                                    self.shared_binding(guest, length)?
                                };
                            if let Some((buffer, offset)) = in_place {
                                storage_in_place
                                    .insert((index, descriptor.binding), (buffer, offset, length));
                                self.gpu
                                    .as_mut()
                                    .unwrap()
                                    .stat("stat: draw storage shared", 1);
                                continue;
                            }
                        }
                        self.gpu
                            .as_mut()
                            .unwrap()
                            .stat("stat: draw storage bytes", length);
                        storage.insert(
                            (index, descriptor.binding),
                            self.gpu_bytes(core, base, length)?,
                        );
                    }
                    // Texel buffers (buffer textures) bind like textures.
                    DescriptorKind::Texture | DescriptorKind::TextureBuffer
                        if descriptor.cbuf_index == 2 && descriptor.cbuf_offset >= 0x20 =>
                    {
                        let slot = (descriptor.cbuf_offset - 0x20) / 8;
                        let &handle = self
                            .render
                            .bound_textures
                            .get(u64::from(nvn), u64::from(slot))
                            .ok_or_else(|| format!("texture {nvn}/{slot} is not bound"))?;
                        // Valid within a present (textures are revalidated
                        // once per present); registrations evict their ids.
                        let stamp = (
                            self.binding_epoch,
                            0,
                            self.binding_serial,
                            // Across presents: entries are revalidated per
                            // present (`memoized_binding`); texture and
                            // target changes evict what they touch
                            // (`evict_bindings`).
                            0,
                        );
                        self.evict_bindings();
                        if self.binding_memo.0 != stamp {
                            let old = self.binding_memo.0;
                            let backend = self.gpu.as_mut().unwrap();
                            for (changed, name) in [
                                (
                                    old.0 != stamp.0,
                                    "stat: binding memo resets by texture epoch",
                                ),
                                (
                                    old.1 != stamp.1,
                                    "stat: binding memo resets by target changes",
                                ),
                                (
                                    old.2 != stamp.2,
                                    "stat: binding memo resets by binding serial",
                                ),
                                (old.3 != stamp.3, "stat: binding memo resets by present"),
                            ] {
                                if changed {
                                    backend.stat(name, 1);
                                }
                            }
                            self.binding_memo = (stamp, Default::default());
                        }
                        if let Some(binding) = self.memoized_binding(handle) {
                            textures.insert((index, descriptor.binding), binding);
                            memo_hits += 1;
                            continue;
                        }
                        self.gpu
                            .as_mut()
                            .unwrap()
                            .stat("stat: texture binding memo misses", 1);
                        let started = timing.then(std::time::Instant::now);
                        let texture = self.handle_texture(handle).map_err(|error| {
                            format!(
                                "{error} ({:?} stage {:016x}, binding {})",
                                stage.stage,
                                keys_for_note.get(index).copied().unwrap_or(0),
                                descriptor.binding
                            )
                        })?;
                        let (source, storage) = self.resolve_sampled_texture(core, texture)?;
                        if let Some(started) = started {
                            texture_time += started.elapsed();
                        }
                        let source = self.through_view(source, handle);
                        if let Some(swizzle) = self.handle_swizzle(handle) {
                            if super::debug_var_os("ECHORA_GPU_FLOW").is_some() {
                                let note = format!("flow view swizzle {swizzle:?} on {texture:x}");
                                self.gpu_note(note);
                            }
                        }
                        let sampler = self.binding_sampler(handle, texture, &source);
                        self.binding_memo.1.insert(
                            handle,
                            (
                                source.clone(),
                                sampler,
                                storage,
                                self.gpu_writes_now(),
                                self.presented_frames,
                                texture,
                            ),
                        );
                        textures.insert((index, descriptor.binding), (source, sampler));
                    }
                    other => {
                        // The stage and its bound storage buffers, from
                        // which the driver constant layout can be inferred.
                        let bound: Vec<String> = self
                            .render
                            .state
                            .iter()
                            .filter(|(key, _)| {
                                key.starts_with(&format!("BindStorageBuffer/{nvn}/"))
                            })
                            .map(|(key, bind)| format!("{key}={:#x}+{:#x}", bind.x[3], bind.x[4]))
                            .collect();
                        return Err(format!(
                            "{other:?} binding at c{}+{:#x} in {:?} stage {:016x} (bound [{}])",
                            descriptor.cbuf_index,
                            descriptor.cbuf_offset,
                            stage.stage,
                            keys_for_note.get(index).copied().unwrap_or(0),
                            bound.join(" ")
                        ));
                    }
                }
            }
        }

        if let Some((kind, since)) = binding_lap.take() {
            binding_times[kind] += since.elapsed();
        }
        let tail_started = timing.then(std::time::Instant::now);
        // Each stage's c0 holds its bound storage buffers' addresses and
        // sizes (slot i at the stage's table + 16 * i, `c0_storage_table`),
        // as the driver keeps them: a program that
        // reaches global memory through the page table reads them there
        // itself, with no storage binding of its own.
        for (index, stage) in stages.iter().enumerate() {
            if stage.push_fields & echora_shader_abi::PUSH_PAGE_TABLE == 0 {
                continue;
            }
            let nvn = nvn_stage(stage.stage);
            let mut slots = self.render.storage_slots[nvn as usize];
            while slots != 0 {
                let slot = u64::from(slots.trailing_zeros());
                slots &= slots - 1;
                if let Some(bind) =
                    self.render
                        .state
                        .get(bind_key("BindStorageBuffer", Some(nvn as u64), slot))
                {
                    let offset = u64::from(c0_storage_table(nvn)) + 16 * slot;
                    storage_entries.push((index, offset as u32, bind.x[3], bind.x[4]));
                }
            }
        }
        // Each stage's c0 holds its storage buffers' addresses and sizes,
        // which translated code reads to find offsets in the buffer.
        for &(index, offset, gpu, size) in &storage_entries {
            let Some(c0) = stages[index].descriptors.iter().find(|descriptor| {
                descriptor.kind() == DescriptorKind::UniformBuffer && descriptor.index == 0
            }) else {
                continue;
            };
            if let Some(bytes) = uniforms.get_mut(&(index, c0.binding)) {
                let entry = offset as usize;
                set_shader_word(bytes, entry, gpu as u32);
                set_shader_word(bytes, entry + 4, (gpu >> 32) as u32);
                set_shader_word(bytes, entry + 8, size as u32);
            }
        }
        // The driver's other words in each stage's c0: the largest point
        // size (0x610, the hardware's 2047; GLSLC clamps gl_PointSize to
        // it) and the draw's base instance (0x644; DrawElementsInstanced's
        // and DrawArraysInstanced's, 0 for other direct draws). Programs
        // read the base instance there to find per-instance data.
        let base_instance = match command.name {
            "nvnCommandBufferDrawArraysInstanced" => command.x[4] as u32,
            "nvnCommandBufferDrawElementsInstanced" => command.x[6] as u32,
            _ => 0,
        };
        for (index, stage) in stages.iter().enumerate() {
            let Some(c0) = stage.descriptors.iter().find(|descriptor| {
                descriptor.kind() == DescriptorKind::UniformBuffer && descriptor.index == 0
            }) else {
                continue;
            };
            if let Some(bytes) = uniforms.get_mut(&(index, c0.binding)) {
                for (offset, word) in [(0x610, 2047.0f32.to_bits()), (0x644, base_instance)] {
                    set_shader_word(bytes, offset, word);
                }
            }
        }
        let pool_started = timing.then(std::time::Instant::now);
        // Texture pool entries, for stages that read the pool.
        // One stage's list is shared as it is (unchanged lists are then
        // known without comparing); several are joined.
        let mut lists = Vec::new();
        for (index, stage) in stages.iter().enumerate() {
            if stage.texture_pool_bindings != 0 {
                let additional = self.shader_pool_handles(
                    core,
                    stage,
                    |core, bank, offset| {
                        let descriptor = stage.descriptors.iter().find(|descriptor| {
                            descriptor.kind() == DescriptorKind::UniformBuffer
                                && descriptor.index == u32::from(bank)
                        })?;
                        let key = (index, descriptor.binding);
                        if let Some(bytes) = uniforms.get(&key) {
                            return shader_word(bytes, offset);
                        }
                        let &(guest, length) = guest_uniforms.get(&key)?;
                        (offset + 4 <= length).then_some(())?;
                        core.memory.read_u32(guest + offset as u64).ok()
                    },
                    |address, length| {
                        storage_entries.iter().any(|&(owner, _, base, size)| {
                            owner == index && storage_contains(base, size, address, length)
                        })
                    },
                );
                lists.push(self.pool_entries(core, nvn_stage(stage.stage), &additional)?);
            }
        }
        let pool: std::sync::Arc<[echora_gpu::pool::PoolEntry]> = match lists.len() {
            0 => Default::default(),
            1 => lists.pop().unwrap(),
            _ => lists.iter().flat_map(|list| list.iter().cloned()).collect(),
        };
        {
            let backend = self.gpu.as_mut().unwrap();
            if let (Some(tail_started), Some(pool_started)) = (tail_started, pool_started) {
                backend.stat(
                    "stat: draw ns driver words",
                    (pool_started - tail_started).as_nanos() as u64,
                );
                backend.stat(
                    "stat: draw ns texture pool entries",
                    pool_started.elapsed().as_nanos() as u64,
                );
            }
            backend.stat("stat: texture binding memo hits", memo_hits);
            for (name, time) in [
                "stat: draw ns binding uniform buffers",
                "stat: draw ns binding driver constants",
                "stat: draw ns binding storage buffers",
                "stat: draw ns binding textures",
                "stat: draw ns binding images and others",
            ]
            .into_iter()
            .zip(binding_times)
            .filter(|_| timing)
            {
                backend.stat(name, time.as_nanos() as u64);
            }
        }
        phase(self, "stat: draw us bindings", &mut since);
        {
            let backend = self.gpu.as_mut().unwrap();
            if timing {
                let time: std::time::Duration = uniform_time;
                backend.stat("stat: draw us uniform reads", time.as_micros() as u64);
                let time: std::time::Duration = texture_time;
                backend.stat("stat: draw us texture sources", time.as_micros() as u64);
            }
            backend.stat("stat: draw uniform bytes", uniform_bytes);
        }
        // Vertex input from the snapshot states and bound streams.
        let attribute_states = self
            .render
            .state
            .get("BindVertexAttribState")
            .map(|bind| bind.states.clone())
            .unwrap_or_default();
        let stream_states = self
            .render
            .state
            .get("BindVertexStreamState")
            .map(|bind| bind.states.clone())
            .unwrap_or_default();
        let value = |props: &BTreeMap<String, super::Args>, key: &str| {
            props.get(key).map_or(0, |args| args.x[0])
        };
        let mut attributes = Vec::new();
        // Vertex pulling (a Rust-translated vertex program): the program
        // fetches its inputs through the page table from this table, and
        // no vertex data is copied.
        let pulling =
            !mesh_draw && stages[0].push_fields & echora_shader_abi::PUSH_VERTEX_FETCH != 0;
        let mut vertex_fetch = Vec::new();
        if pulling {
            let locations = 32 - stages[0].loaded_generics.leading_zeros() as usize;
            vertex_fetch = vec![echora_shader_abi::VertexAttribute::default(); locations];
        }
        // Attributes the shader loads but the state leaves without a format
        // (0): read as (0, 0, 0, 1), the usual value of an attribute with no
        // vertex data (inferred; NVN's documentation is not public).
        let mut unformatted = Vec::new();
        let parsed = self.parsed_vertex_state(&attribute_states, &stream_states);
        for (location, attribute) in parsed.attributes.iter().enumerate() {
            if stages[0].loaded_generics >> location & 1 == 0 {
                continue;
            }
            let format = attribute.format;
            if pulling {
                vertex_fetch[location] = self.fetch_entry(attribute, &parsed.streams)?;
                continue;
            }
            if format == 0 {
                unformatted.push(location as u32);
                continue;
            }
            let vk_format =
                vk_format(format).ok_or_else(|| format!("vertex attribute format {format}"))?;
            attributes.push(VertexAttribute {
                location: location as u32,
                stream: attribute.stream as u32,
                offset: attribute.offset as u32,
                format: vk_format,
            });
        }
        // The draw itself (a mesh pipeline's output is triangles).
        let mode = if mesh_draw { 4 } else { command.x[1] };
        // Quads (mode 7, GL order): drawn as two triangles each.
        let quads = mode == 7;
        let topology = if quads {
            vk::PrimitiveTopology::TRIANGLE_LIST
        } else {
            topology(mode).ok_or_else(|| format!("primitive mode {mode}"))?
        };
        // Instance count and base instance (1 and 0 for plain draws).
        let (instances, first_instance) = match command.name {
            // DrawArraysInstanced(cmd, mode, first, count, baseInstance,
            // instanceCount) (inferred NVN1 order).
            "nvnCommandBufferDrawArraysInstanced" => (command.x[5] as u32, command.x[4] as u32),
            // DrawElementsInstanced(cmd, mode, type, count, indexBuffer,
            // baseVertex, baseInstance, instanceCount) (inferred NVN1 order).
            "nvnCommandBufferDrawElementsInstanced" => (command.x[7] as u32, command.x[6] as u32),
            _ => (1, 0),
        };
        // Instanced and quad draws (ECHORA_GPU_NO_INSTANCED skips them,
        // debugging). The counts the application passes are recorded: 1 to
        // 256 instances from base 0 in gameplay, confirming the argument
        // order, and 4,112 to 5,136 for the forest's ground cover (moss and
        // grass blades); implausible counts are refused.
        let new_kind =
            instances != 1 || first_instance != 0 || quads || command.name.ends_with("Instanced");
        if new_kind {
            let backend = self.gpu.as_mut().unwrap();
            backend.stat("stat: instanced or quad draws", 1);
            let bucket = match instances {
                0 => "0",
                1 => "1",
                2..=16 => "2-16",
                17..=256 => "17-256",
                257..=4096 => "257-4096",
                _ => "over 4096",
            };
            if super::debug_var_os("ECHORA_GPU_FLOW").is_some() {
                backend.note(format!(
                    "instanced draw {} instances {bucket} base instance {}",
                    super::without_prefix(&command.name, "nvnCommandBuffer"),
                    if first_instance > 4096 {
                        "over 4096".to_string()
                    } else {
                        first_instance.to_string()
                    }
                ));
            }
            if super::debug_var_os("ECHORA_GPU_NO_INSTANCED").is_some() {
                return Err(format!(
                    "{} is not drawn (ECHORA_GPU_NO_INSTANCED)",
                    command.name
                ));
            }
            if instances > 1 << 20 || first_instance > 1 << 20 {
                return Err(format!("{instances} instances from {first_instance}"));
            }
        }
        // SetPrimitiveRestart(cmd, enable, index): in a strip or fan, an
        // index equal to `index` starts a new primitive. Vulkan restarts
        // at the index type's all-ones value only, so matching indices are
        // rewritten to it (8-bit indices were widened to 16 bits).
        let restart = self
            .render
            .state
            .get("SetPrimitiveRestart")
            .filter(|command| command.x[1] & 0xff != 0)
            .map(|command| command.x[2] as u32);
        let strip = matches!(
            topology,
            vk::PrimitiveTopology::LINE_STRIP
                | vk::PrimitiveTopology::TRIANGLE_STRIP
                | vk::PrimitiveTopology::TRIANGLE_FAN
        );
        let (mut indices, mut count, mut first, mut base_vertex) = match command.name {
            // DrawArrays(cmd, mode, first, count).
            "nvnCommandBufferDrawArrays" | "nvnCommandBufferDrawArraysInstanced" => {
                (Indices::None, command.x[3] as u32, command.x[2] as u32, 0)
            }
            // DrawElementsBaseVertex(cmd, mode, type, count, indexBuffer,
            // baseVertex); type 0 = 8-bit, 1 = 16-bit, 2 = 32-bit (NVN1's
            // NVNindexType order). 8-bit indices widen to 16-bit, which the
            // renderer takes.
            "nvnCommandBufferDrawElementsBaseVertex"
            | "nvnCommandBufferDrawElements"
            | "nvnCommandBufferDrawElementsInstanced" => {
                let count = command.x[3] as u32;
                let (size, wrap): (u64, fn(Vec<u8>) -> Indices) = match command.x[2] {
                    0 => (1, |bytes: Vec<u8>| {
                        Indices::U16(
                            bytes
                                .iter()
                                .flat_map(|&index| u16::from(index).to_le_bytes())
                                .collect(),
                        )
                    }),
                    1 => (2, Indices::U16),
                    2 => (4, Indices::U32),
                    other => return Err(format!("index type {other}")),
                };
                let address = command.x[4];
                let length = u64::from(count) * size;
                let guest = self.gpu_to_guest(address, length)?;
                self.resolve_draw_input(core, guest, length)?;
                // Vertex pulling needs the indices only on the GPU. Upload
                // them as-is when the CPU does not widen or rewrite them.
                let rewrites_indices = size == 1 || quads || strip && restart.is_some();
                let indices = if pulling && !rewrites_indices && count > 0 {
                    let kind = if size == 2 {
                        vk::IndexType::UINT16
                    } else {
                        vk::IndexType::UINT32
                    };
                    Indices::Guest {
                        address: guest,
                        bytes: length,
                        kind,
                    }
                } else {
                    let mut bytes = vec![0; length as usize];
                    core.memory
                        .read(guest, &mut bytes)
                        .map_err(|trap| format!("reading GPU {address:#x}: {trap:?}"))?;
                    wrap(bytes)
                };
                let base = if command.name.ends_with("BaseVertex")
                    || command.name.ends_with("Instanced")
                {
                    command.x[5] as i32
                } else {
                    0
                };
                (indices, count, 0, base)
            }
            _ if mesh_draw => (Indices::None, 0, 0, 0),
            other => return Err(format!("{other} is not modeled on the GPU")),
        };
        // A draw this large can only come from misread state; it would hold
        // the GPU for seconds (the kernel resets it past 10 s).
        if u64::from(count) * u64::from(instances) > 1 << 24 {
            return Err(format!("{count} vertices x {instances} instances"));
        }
        if quads {
            let corners: Vec<u32> = match &indices {
                Indices::None => (first..first + count).collect(),
                Indices::U16(bytes) => bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|index| u32::from(u16::from_le_bytes(*index)))
                    .collect(),
                Indices::U32(bytes) => bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|index| u32::from_le_bytes(*index))
                    .collect(),
                Indices::Guest { .. } => unreachable!("quad indices are decoded on the CPU"),
            };
            let triangles: Vec<u8> = corners
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|[a, b, c, d]| [*a, *b, *c, *a, *c, *d])
                .flat_map(u32::to_le_bytes)
                .collect();
            if triangles.is_empty() {
                return Ok(());
            }
            count = (triangles.len() / 4) as u32;
            first = 0;
            indices = Indices::U32(triangles);
        }

        // One past the highest vertex the draw references, for CPU-selected
        // vertex stream ranges. Guest indices need no CPU-selected range.
        let vertex_end = match &indices {
            Indices::None => u64::from(first) + u64::from(count),
            Indices::U16(bytes) => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|index| u64::from(u16::from_le_bytes(*index)))
                .max()
                .map_or(0, |max| {
                    (max as i64 + i64::from(base_vertex) + 1).max(0) as u64
                }),
            Indices::U32(bytes) => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|index| u64::from(u32::from_le_bytes(*index)))
                .max()
                .map_or(0, |max| {
                    (max as i64 + i64::from(base_vertex) + 1).max(0) as u64
                }),
            Indices::Guest { .. } => 0,
        };
        // The lowest vertex it references: meshes that share one large
        // vertex buffer read their own range only (the draw's first or base
        // vertex moves down by as much).
        let lowest_index = |indices: &Indices| -> Option<u64> {
            match indices {
                Indices::None | Indices::Guest { .. } => None,
                Indices::U16(bytes) => bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|index| u64::from(u16::from_le_bytes(*index)))
                    .min(),
                Indices::U32(bytes) => bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|index| u64::from(u32::from_le_bytes(*index)))
                    .min(),
            }
        };
        let vertex_start = match lowest_index(&indices) {
            None if matches!(indices, Indices::None) => u64::from(first),
            None => 0,
            Some(min) => (min as i64 + i64::from(base_vertex)).max(0) as u64,
        }
        .min(vertex_end);
        let per_vertex = |props: &ParsedStream| props.divisor == 0 && props.stride != 0;
        let shifted = !pulling && vertex_start > 0 && parsed.streams.iter().any(per_vertex);
        // Each stream's bytes per vertex the loaded attributes reach (past
        // the furthest attribute's offset and size): a stream no attribute
        // reads is not read, and a stride-0 one (the same element for every
        // vertex) only that far, not its whole buffer (full-screen passes
        // bound megabytes for three vertices).
        let mut needed: HashMap<u32, u64> = HashMap::default();
        for attribute in &attributes {
            let size = attribute_format_size(attribute.format);
            let end = u64::from(attribute.offset) + size;
            let entry = needed.entry(attribute.stream).or_default();
            *entry = (*entry).max(end);
        }
        let mut streams = Vec::new();
        for (index, props) in parsed.streams.iter().enumerate() {
            // Only the vertices this draw can reach are read.
            let stride = props.stride;
            let Some(&element) = needed.get(&(index as u32)) else {
                streams.push(VertexStream {
                    stride: stride as u32,
                    divisor: props.divisor as u32,
                    data: Vec::new(),
                    guest: None,
                });
                continue;
            };
            let skip = if shifted && per_vertex(props) {
                vertex_start * stride
            } else {
                0
            };
            let data = match self
                .render
                .state
                .get(bind_key("BindVertexBuffer", None, index as u64))
            {
                Some(bind) if bind.x[3] > skip => {
                    let divisor = props.divisor;
                    let reach = if stride == 0 {
                        element.min(bind.x[3])
                    } else if divisor != 0 {
                        // Per-instance data: the instances drawn.
                        let elements = (u64::from(first_instance) + u64::from(instances.max(1)))
                            .div_ceil(divisor);
                        (elements * stride).min(bind.x[3])
                    } else {
                        (vertex_end * stride).min(bind.x[3])
                    };
                    let (address, size) = (bind.x[2] + skip, reach.saturating_sub(skip));
                    let guest = self.gpu_to_guest(address, size)?;
                    self.resolve_draw_input(core, guest, size)?;
                    // Read into the ring when recorded, not copied here.
                    Some((guest, size as usize))
                }
                _ => None,
            };
            streams.push(VertexStream {
                stride: props.stride as u32,
                divisor: props.divisor as u32,
                data: Vec::new(),
                guest: data,
            });
        }
        // ECHORA_FETCH_BEYOND=<stage key>[,...] (debugging): each bound
        // stream's bytes at the draw's first vertex (base vertex and first
        // index applied), whether or not they are inside the bound size.
        if super::debug_var("ECHORA_FETCH_BEYOND").is_ok_and(|wanted| {
            wanted
                .split(',')
                .any(|key| key == format!("{:016x}", keys.0) || key == format!("{:016x}", keys.1))
        }) {
            for index in 0..16 {
                let (Some(bind), Some(props)) = (
                    self.render
                        .state
                        .get(bind_key("BindVertexBuffer", None, index as u64)),
                    stream_states.get(index),
                ) else {
                    continue;
                };
                let stride = value(props, "Stride");
                if bind.x[2] == 0 {
                    continue;
                }
                let vertex = (i64::from(first) + i64::from(base_vertex)).max(0) as u64;
                let at = bind.x[2] + vertex * stride;
                let words: Vec<f32> = self
                    .gpu_bytes(core, at, 64)
                    .unwrap_or_default()
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|word| f32::from_le_bytes(*word))
                    .collect();
                eprintln!(
                    "fetch beyond present {} {:016x}/{:016x} stream {index} at {:#x} size {:#x} stride {stride} vertex {vertex} -> {at:#x}: {words:?} state {props:?}",
                    self.presented_frames, keys.0, keys.1, bind.x[2], bind.x[3]
                );
            }
            eprintln!(
                "fetch beyond present {} attributes {:?}",
                self.presented_frames,
                attribute_states.iter().take(4).collect::<Vec<_>>()
            );
        }
        if shifted {
            if matches!(indices, Indices::None) {
                first -= vertex_start as u32;
            } else {
                base_vertex -= vertex_start as i32;
            }
        }
        if !unformatted.is_empty() {
            let stream = streams.len() as u32;
            streams.push(VertexStream {
                stride: 0,
                divisor: 0,
                data: [0.0f32, 0.0, 0.0, 1.0]
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
                guest: None,
            });
            attributes.extend(unformatted.into_iter().map(|location| VertexAttribute {
                location,
                stream,
                offset: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
            }));
        }

        let page_table = if pulling
            || stages
                .iter()
                .any(|stage| stage.push_fields & echora_shader_abi::PUSH_PAGE_TABLE != 0)
        {
            self.page_table(core)?
        } else {
            Default::default()
        };
        // Stages store through the table (a vertex program writing texels
        // back for the CPU): later CPU reads of those ranges wait for them.
        // Stages that only read through it (the translator says) do not.
        let writes_page_table = page_table.address != 0
            && stages.iter().any(|stage| {
                stage.push_fields & echora_shader_abi::PUSH_PAGE_TABLE != 0
                    && stage.push_fields & echora_shader_abi::PUSH_PAGE_TABLE_READ_ONLY == 0
            });
        phase(self, "stat: draw us geometry", &mut since);
        let geometry_bytes = streams.iter().map(VertexStream::len).sum::<usize>();
        if geometry_bytes > 1 << 20 && super::debug_var_os("ECHORA_GPU_FLOW").is_some() {
            let strides: Vec<(u32, u32, usize)> = streams
                .iter()
                .map(|stream| (stream.stride, stream.divisor, stream.len()))
                .collect();
            let note = format!(
                "flow large geometry {:016x}/{:016x} {} {:?} vertex range {vertex_start}..{vertex_end} count {count} streams {strides:?}",
                keys.0,
                keys.1,
                command.name,
                matches!(indices, Indices::None)
            );
            self.gpu_note(note);
        }
        self.gpu
            .as_mut()
            .unwrap()
            .stat("stat: draw geometry bytes", geometry_bytes as u64);
        let mut extent = None;
        for (color, format) in colors[..bound].iter_mut().zip(&mut color_formats[..bound]) {
            if *color != 0
                && let Some(view) = *format
                && let Some((alias, size)) = self.block_alias_target(*color, view)?
            {
                *color = alias;
                *format = None;
                extent.get_or_insert(size);
            }
        }
        let (colors, color_levels, color_layers, color_formats) = (
            &colors[..bound],
            &color_levels[..bound],
            &color_layers[..bound],
            &color_formats[..bound],
        );
        for (index, &texture) in colors.iter().enumerate() {
            if texture == 0 || self.block_aliases.values().any(|&alias| alias == texture) {
                continue;
            }
            if self.depth_traced(texture) {
                eprintln!(
                    "depth trace present {} draw {:016x}/{:016x} renders {texture:x} as view format {:?} level {:?}",
                    self.presented_frames,
                    keys.0,
                    keys.1,
                    self.render.color_formats.get(index),
                    color_levels.get(index)
                );
            }
            extent.get_or_insert(self.ensure_render_target(texture)?);
        }
        // A depth texture object over the storage of an existing depth
        // target is that target (the game builds a new object over the
        // same depth memory for its lit pass, testing EQUAL against what
        // the prepass wrote).
        let depth = (self.render.depth != 0).then(|| {
            let texture = self.render.depth;
            let renderer = &self.gpu.as_ref().unwrap().renderer;
            if renderer.existing_target(texture).is_some() {
                return texture;
            }
            self.target_sharing_storage(texture)
                .filter(|&target| {
                    renderer
                        .existing_target(target)
                        .is_some_and(|target| echora_gpu::renderer::is_depth(target.format))
                })
                .unwrap_or(texture)
        });
        if let Some(texture) = depth.filter(|&texture| self.depth_traced(texture)) {
            eprintln!(
                "depth trace present {} draw {:016x}/{:016x} depth {texture:x} (bound {:x}) viewport {:?} scissor {:?}",
                self.presented_frames,
                keys.0,
                keys.1,
                self.render.depth,
                self.render.state.get("SetViewport").map(|bind| [
                    bind.x[1] as i32,
                    bind.x[2] as i32,
                    bind.x[3] as i32,
                    bind.x[4] as i32
                ]),
                self.render.scissor
            );
        }
        if let Some(texture) = depth {
            extent.get_or_insert(self.ensure_render_target(texture)?);
        }
        let depth_layer = self.render.depth_layer;
        // With no target bound the draw still runs its vertex stage (the
        // game reads texels back into a buffer from a vertex program, with
        // a fragment program that only exits): the area is the viewport's.
        let extent = extent.or_else(|| {
            let bind = self.render.state.get("SetViewport")?;
            let (width, height) = (bind.x[3] as i32, bind.x[4] as i32);
            (width > 0 && height > 0).then_some((width as u32, height as u32))
        });
        let (width, height) = extent.ok_or_else(|| {
            format!(
                "draw without render targets or viewport ({:016x}/{:016x})",
                keys.0, keys.1
            )
        })?;
        self.initialize_new_targets(core);
        let viewport = self
            .render
            .state
            .get("SetViewport")
            .map(|bind| {
                [
                    bind.x[1] as i32 as f32,
                    bind.x[2] as i32 as f32,
                    bind.x[3] as i32 as f32,
                    bind.x[4] as i32 as f32,
                ]
            })
            .unwrap_or([0.0, 0.0, width as f32, height as f32]);
        // The application passes 1 to SetWindowOriginMode: UPPER_LEFT, the
        // D3D convention (window row 0 at the top, clip +Y up to it). A
        // pass sampling the depth buffer by clip-space position showed the
        // scene mirrored vertically under the lower-left reading, and the
        // trainer portrait copy flips rows the way only this one undoes.
        // Viewports and scissors keep their window coordinates; the Vulkan
        // viewport's negative height maps clip +Y to row 0.
        let scissor = self
            .render
            .scissor
            .unwrap_or([0, 0, width as i32, height as i32]);
        // SetDepthRange(cmd, float near, float far) (S0, S1): the window
        // depth range. The intro's backdrop mesh is drawn with [0, 0], at
        // the far plane of its reverse-Z depth, so the sky behind it still
        // passes; at [0, 1] it covered the sky (a PokeDinner capture shows
        // that viewport).
        let depth_range = self
            .render
            .state
            .get("SetDepthRange")
            .map(|bind| {
                [
                    f32::from_bits(bind.v[0] as u32).clamp(0.0, 1.0),
                    f32::from_bits(bind.v[1] as u32).clamp(0.0, 1.0),
                ]
            })
            .unwrap_or([0.0, 1.0]);
        // Fixed-function state from the record-time snapshots.
        let arg = |props: &BTreeMap<String, super::Args>, key: &str, index: usize| {
            props.get(key).map(|args| args.x[index])
        };
        let polygon = self.render.properties("BindPolygonState");
        // The polygon state the draw binds (its address) and the props taken
        // for it, with their element's offset from that address (a nonzero
        // offset is another object's bag: the bound one has none).
        let depth_state = self.render.properties("BindDepthStencilState");
        let color_state = self.render.properties("BindColorState");
        let mask_state = self.render.properties("BindChannelMaskState");
        let primitive_restart = match (restart, &mut indices) {
            (Some(value), Indices::U16(bytes)) if strip => {
                for index in bytes.as_chunks_mut::<2>().0 {
                    if u32::from(u16::from_le_bytes(*index)) == value & 0xffff {
                        *index = [0xff; 2];
                    }
                }
                true
            }
            (Some(value), Indices::U32(bytes)) if strip => {
                for index in bytes.as_chunks_mut::<4>().0 {
                    if u32::from_le_bytes(*index) == value {
                        *index = [0xff; 4];
                    }
                }
                true
            }
            _ => false,
        };
        let fixed = Fixed {
            topology,
            primitive_restart,
            // ECHORA_GPU_NO_CULL=<fragment key>[,...] (debugging): those
            // programs' draws cull nothing (shows whether culling is what
            // drops a draw's triangles).
            cull: match arg(polygon, "CullFace", 0).unwrap_or(0) {
                _ if super::debug_var("ECHORA_GPU_NO_CULL").is_ok_and(|keys| {
                    keys.split(',')
                        .any(|key| key == format!("{:016x}", keys_for_note[1]))
                }) =>
                {
                    vk::CullModeFlags::NONE
                }
                1 => vk::CullModeFlags::FRONT,
                2 => vk::CullModeFlags::BACK,
                3 => vk::CullModeFlags::FRONT_AND_BACK,
                _ => vk::CullModeFlags::NONE,
            },
            // NVN's FrontFace: 0 clockwise, 1 counter-clockwise, in window
            // coordinates (the upper-left origin's, as Vulkan's; the other
            // way the Pokedex showed its back cover and Ditto its inside).
            // ECHORA_GPU_INVERT_FACE (debugging): every draw's front face
            // the other way (tests the facing convention against the
            // picture).
            front_face: match (
                arg(polygon, "FrontFace", 0).unwrap_or(1),
                super::debug_var_os("ECHORA_GPU_INVERT_FACE").is_some(),
            ) {
                (0, false) | (1.., true) => vk::FrontFace::CLOCKWISE,
                _ => vk::FrontFace::COUNTER_CLOCKWISE,
            },
            polygon_mode: match arg(polygon, "PolygonMode", 0).unwrap_or(2) {
                0 => vk::PolygonMode::POINT,
                1 => vk::PolygonMode::LINE,
                _ => vk::PolygonMode::FILL,
            },
            // ECHORA_GPU_NODEPTH=<fragment key>[,...] (debugging): those
            // programs draw without the depth test.
            depth_test: arg(depth_state, "DepthTestEnable", 0).unwrap_or(0) != 0
                && !super::debug_var("ECHORA_GPU_NODEPTH").is_ok_and(|keys| {
                    keys.split(',')
                        .any(|key| key == format!("{:016x}", keys_for_note[1]))
                }),
            depth_write: arg(depth_state, "DepthWriteEnable", 0).unwrap_or(0) != 0,
            // ECHORA_GPU_EQUAL_AS_LEQUAL=<fragment key>[,...] (debugging):
            // those programs' EQUAL depth tests as LESS_OR_EQUAL (tells a
            // depth that differs by rounding from one never written).
            depth_compare: match compare_op(arg(depth_state, "DepthFunc", 0).unwrap_or(8)) {
                vk::CompareOp::EQUAL
                    if super::debug_var("ECHORA_GPU_EQUAL_AS_LEQUAL").is_ok_and(|keys| {
                        keys.split(',')
                            .any(|key| key == format!("{:016x}", keys_for_note[1]))
                    }) =>
                {
                    vk::CompareOp::LESS_OR_EQUAL
                }
                // ECHORA_GPU_LESS_AS_LEQUAL=<fragment key>[,...] (debugging):
                // the same for LESS (tells whether a pass meets depth an
                // earlier pass already wrote exactly).
                vk::CompareOp::LESS
                    if super::debug_var("ECHORA_GPU_LESS_AS_LEQUAL").is_ok_and(|keys| {
                        keys.split(',')
                            .any(|key| key == format!("{:016x}", keys_for_note[1]))
                    }) =>
                {
                    vk::CompareOp::LESS_OR_EQUAL
                }
                op => op,
            },
            viewport: [
                viewport[0],
                viewport[1] + viewport[3],
                viewport[2],
                -viewport[3],
                depth_range[0],
                depth_range[1],
            ],
            scissor,
            // The device's depth mode (nvnDeviceSetDepthMode): 0,
            // NEAR_IS_MINUS_W (the OpenGL range, which this application
            // sets), maps clip z in [-w, w] to the depth range; 1,
            // NEAR_IS_ZERO, maps [0, w].
            depth_minus_one_to_one: self.depth_mode() == 0,
            // PolygonState SetPolygonOffsetEnables (bit 0x4 fill, after
            // point 0x1 and line 0x2, as NVN1; inferred) with the command
            // buffer's SetPolygonOffsetClamp(factor, units, clamp) (S0-S2):
            // ground decals drawn at the ground's depth rely on it, and
            // z-fought with the ground in alternating rows without it.
            depth_bias: (arg(polygon, "PolygonOffsetEnables", 0).unwrap_or(0) & 4 != 0)
                .then(|| self.render.state.get("SetPolygonOffsetClamp"))
                .flatten()
                .map(|bind| {
                    let float = |index: usize| f32::from_bits(bind.v[index] as u32);
                    [float(1), float(2), float(0)]
                }),
            stencil: (arg(depth_state, "StencilTestEnable", 0).unwrap_or(0) != 0)
                .then(|| self.stencil_faces(depth_state)),
            // MultisampleState SetAlphaToCoverageEnable(state, enable).
            alpha_to_coverage: arg(
                self.render.properties("BindMultisampleState"),
                "AlphaToCoverageEnable",
                0,
            )
            .is_some_and(|enable| enable != 0),
            // SetBlendColor(cmd, const float rgba[4]): the four floats
            // captured at record time; NVN's default is zero.
            blend_constants: self
                .render
                .state
                .get("SetBlendColor")
                .filter(|bind| bind.captured.len() == 4)
                .map_or([0.0; 4], |bind| {
                    std::array::from_fn(|channel| f32::from_bits(bind.captured[channel] as u32))
                }),
            // DEPTH24_STENCIL8 (NVN format 53) is held as D32F_S8 (`vk_format`).
            depth_bias_d24: self
                .object(self.render.depth, "Texture")
                .ok()
                .and_then(|texture| texture.builder.get("Format"))
                .is_some_and(|format| format.x[0] as u32 == 53),
        };
        if let Some([units, clamp, factor]) = fixed.depth_bias
            && super::debug_var_os("ECHORA_GPU_FLOW").is_some()
        {
            self.gpu.as_mut().unwrap().note(format!(
                "polygon offset units {units} factor {factor} clamp {clamp}"
            ));
        }
        // Blending: ColorState enables per target, each target's latest
        // BlendState its factors and equations, ChannelMaskState write
        // masks.
        // ECHORA_GPU_NO_BLEND=<fragment key hex>[,...] (debugging): those
        // programs draw without blending.
        let no_blend = super::debug_var("ECHORA_GPU_NO_BLEND").is_ok_and(|wanted| {
            wanted
                .split(',')
                .any(|item| item.trim() == format!("{:016x}", keys.1))
        });
        let mut unknown_factors = Vec::new();
        let mut blends = [Blend::default(); MAX_COLOR_TARGETS];
        macro_rules! target_state_keys {
            ($($slot:literal),*) => {
                [$([
                    concat!("BlendEnable/", $slot),
                    concat!("BindBlendState/", $slot),
                    concat!("ChannelMask/", $slot),
                ]),*]
            };
        }
        const TARGET_STATE_KEYS: [[&str; 3]; MAX_COLOR_TARGETS] =
            target_state_keys!(0, 1, 2, 3, 4, 5, 6, 7);
        for (blend, [enable_key, state_key, mask_key]) in
            blends[..bound].iter_mut().zip(TARGET_STATE_KEYS)
        {
            // SetBlendEnable(state, index, enable): each index's latest.
            if let Some(enable) = color_state.get(enable_key) {
                blend.enable = enable.x[1] != 0 && !no_blend;
            }
            let blend_state = self.render.properties(state_key);
            if let Some(func) = blend_state.get("BlendFunc") {
                let factors = func.x.map(blend_factor);
                if let [Some(src), Some(dst), Some(src_alpha), Some(dst_alpha)] = factors {
                    blend.color.0 = src;
                    blend.color.1 = dst;
                    blend.alpha.0 = src_alpha;
                    blend.alpha.1 = dst_alpha;
                } else {
                    unknown_factors.push(func.x);
                }
            }
            if let Some(equation) = blend_state.get("BlendEquation") {
                blend.color.2 = blend_op(equation.x[0]);
                blend.alpha.2 = blend_op(equation.x[1]);
            }
            if let Some(mask) = mask_state.get(mask_key) {
                let mut flags = vk::ColorComponentFlags::empty();
                // r, g, b in X2..X4; alpha (X5) recorded in v[1]. A
                // G-buffer pass binds one texture as targets 2 and 4
                // and masks target 4 off: its alpha writes overwrote
                // target 2's material alpha.
                let alpha = mask.v[1];
                for (enabled, flag) in [
                    (mask.x[1], vk::ColorComponentFlags::R),
                    (mask.x[2], vk::ColorComponentFlags::G),
                    (mask.x[3], vk::ColorComponentFlags::B),
                    (alpha, vk::ColorComponentFlags::A),
                ] {
                    if enabled & 0xff != 0 {
                        flags |= flag;
                    }
                }
                blend.write_mask = flags;
            }
        }
        let blends = &blends[..bound];
        // A target with factors NVN defines but the device path does not
        // map keeps the default (ONE, ZERO), never a guessed factor.
        for factors in unknown_factors {
            self.gpu.as_mut().unwrap().note(format!(
                "blend factors {factors:#x?} not supported: target drawn unblended (program {:016x})",
                keys.1
            ));
        }
        if super::debug_var_os("ECHORA_GPU_FLOW").is_some() {
            let line = format!(
                "flow draw {:016x}/{:016x} count {count} reads {} writes {} level {:?} layers {:?} depth layer {} viewport {:?} depth {:x?} blend {:?} masks {:?} factors {:?} constants {:?}",
                keys.0,
                keys.1,
                self.describe_sources(textures.values().map(|(source, _)| source)),
                self.describe_targets(colors.iter().copied().filter(|key| *key != 0)),
                color_levels,
                color_layers,
                depth_layer,
                fixed.viewport,
                depth,
                blends.iter().map(|blend| blend.enable).collect::<Vec<_>>(),
                blends
                    .iter()
                    .map(|blend| blend.write_mask.as_raw())
                    .collect::<Vec<_>>(),
                blends
                    .iter()
                    .map(|blend| (blend.color, blend.alpha))
                    .collect::<Vec<_>>(),
                fixed.blend_constants
            );
            self.sequence_note(&line);
            self.gpu.as_mut().unwrap().note(line);
        }
        // ECHORA_GPU_TRACE_PROGRAM=<fragment key hex>[,...]: trace those
        // programs' draws, with their fixed-function state.
        let traced_program = super::debug_var("ECHORA_GPU_TRACE_PROGRAM").is_ok_and(|wanted| {
            wanted.split(',').any(|item| {
                item == format!("{:016x}", keys.1) || item == format!("{:016x}", keys.0)
            })
        });
        let trace_details = traced_program
            || (self.gpu.as_ref().unwrap().traced < 40
                && super::debug_var_os("ECHORA_GPU_LOG").is_some());
        let depth_storage = depth.filter(|_| trace_details).map(|texture| {
            (
                self.texture_storage(texture),
                self.texture_extent(texture).ok(),
                self.texture_storage(self.render.depth),
            )
        });
        // Each traced binding's texture, in binding order.
        let traced_textures = if traced_program {
            let mut bound_textures: Vec<_> = textures.iter().collect();
            bound_textures.sort_by_key(|(key, _)| **key);
            bound_textures
                .iter()
                .map(|(key, (source, sampler))| {
                    format!(
                        "{key:?}={} sampler {sampler:?}",
                        self.describe_sources(std::iter::once(source))
                    )
                })
                .collect::<Vec<_>>()
                .join(" ")
        } else {
            String::new()
        };
        let bound: Vec<String> = if traced_program {
            self.render
                .state
                .iter()
                .filter(|(key, _)| key.starts_with("BindProgram@"))
                .map(|(key, bind)| {
                    format!(
                        "{key}:program {:#x} bits {:#x} from buffer {:#x}",
                        bind.x[1], bind.x[2], bind.x[0]
                    )
                })
                .chain(
                    self.render
                        .named_uniform_bindings()
                        .map(|(key, bind)| format!("{key}:{:#x}+{:#x}", bind.x[3], bind.x[4])),
                )
                .collect()
        } else {
            Vec::new()
        };
        // A traced program's sampled targets, as the draw meets them, also
        // go through ECHORA_TARGET_TRACE (their contents when read, beside
        // those when written).
        if !traced_textures.is_empty() {
            let mut sampled: Vec<u64> = textures
                .values()
                .filter_map(|(source, _)| match source {
                    TextureSource::Target(key) | TextureSource::TargetView { key, .. } => {
                        Some(*key)
                    }
                    _ => None,
                })
                .collect();
            sampled.sort_unstable();
            sampled.dedup();
            self.trace_targets(
                &sampled,
                &[],
                &format!("sampled by draw {:016x}/{:016x}", keys.0, keys.1),
            )?;
        }
        // ECHORA_DRAW_UNIFORM_DUMP=<fragment or vertex key hex>[,...] with
        // ECHORA_DUMP_DIR (debugging): each draw of those programs writes
        // its uniform buffers whole, as uniforms-<key>-<present>-<n>-<stage>-
        // <binding>.bin (n counting draws within the present), from present
        // ECHORA_DRAW_UNIFORM_DUMP_FROM on (as ECHORA_DISPATCH_UNIFORM_DUMP).
        if let (Ok(wanted), Some(directory)) = (
            super::debug_var("ECHORA_DRAW_UNIFORM_DUMP"),
            super::debug_var_os("ECHORA_DUMP_DIR"),
        ) && wanted.split(',').any(|item| {
            item.trim() == format!("{:016x}", keys.1) || item.trim() == format!("{:016x}", keys.0)
        }) && super::debug_var("ECHORA_DRAW_UNIFORM_DUMP_FROM")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .is_none_or(|from| self.presented_frames >= from)
        {
            let directory = std::path::Path::new(&directory);
            let _ = std::fs::create_dir_all(directory);
            let prefix = format!(
                "uniforms-{:016x}-{:016x}-{}",
                keys.0, keys.1, self.presented_frames
            );
            let mut index = 0;
            while directory.join(format!("{prefix}-{index}.done")).exists() {
                index += 1;
            }
            // ECHORA_DRAW_UNIFORM_DUMP_MAX=N: the first N draws per present.
            let most = super::debug_var("ECHORA_DRAW_UNIFORM_DUMP_MAX")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(usize::MAX);
            for ((stage, binding), bytes) in uniforms.iter().filter(|_| index < most) {
                let _ = std::fs::write(
                    directory.join(format!("{prefix}-{index}-{stage}-{binding}.bin")),
                    bytes,
                );
            }
            for ((stage, binding), &(guest, length)) in
                guest_uniforms.iter().filter(|_| index < most)
            {
                let mut bytes = vec![0; length];
                if core.memory.read(guest, &mut bytes).is_ok() {
                    let _ = std::fs::write(
                        directory.join(format!("{prefix}-{index}-{stage}-{binding}.bin")),
                        bytes,
                    );
                }
            }
            // Its storage buffers too, as the draw reads them, named by
            // stage and constant-buffer offset (with their GPU address in
            // a listing).
            if index < most {
                let mut listing = String::new();
                for &(stage, offset, gpu, size) in &storage_entries {
                    if let Ok(bytes) = self.gpu_bytes(core, gpu, size) {
                        let _ = std::fs::write(
                            directory
                                .join(format!("{prefix}-{index}-storage-{stage}-{offset:x}.bin")),
                            bytes,
                        );
                    }
                    listing.push_str(&format!(
                        "stage {stage} c0+{offset:#x} gpu {gpu:#x} size {size:#x}\n"
                    ));
                }
                let _ = std::fs::write(directory.join(format!("{prefix}-{index}.done")), listing);
            }
        }
        let backend = self.gpu.as_mut().unwrap();
        if traced_program {
            let line = format!(
                "trace textures present {} {:016x}/{:016x} base {base_vertex} buffer {:#x} {traced_textures}",
                self.presented_frames, keys.0, keys.1, command.x[0]
            );
            eprintln!("{line}");
            backend.note(line);
            // The colour state as snapshotted for this draw (blend enables).
            let color_state = self.render.properties("BindColorState");
            eprintln!(
                "trace color state {:016x}/{:016x} {color_state:?} blends {blends:?}",
                keys.0, keys.1
            );
            let line = format!(
                "trace program {:016x}/{:016x} bound {bound:?} depth test {} write {} {:?} bias {:?} stencil {:?} cull {:?} viewport {:?} scissor {:?} blends {:?} colors {:x?} view formats {:?} levels {:x?} uniforms {:?}",
                keys.0,
                keys.1,
                fixed.depth_test,
                fixed.depth_write,
                fixed.depth_compare,
                fixed.depth_bias,
                fixed.stencil,
                fixed.cull,
                fixed.viewport,
                fixed.scissor,
                blends,
                colors,
                self.render.color_formats,
                self.render.color_levels,
                uniforms
                    .iter()
                    .map(|(key, bytes)| (
                        *key,
                        bytes.len(),
                        bytes
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .take(300)
                            .map(|word| f32::from_le_bytes(*word))
                            .collect::<Vec<_>>()
                    ))
                    .collect::<Vec<_>>()
            );
            eprintln!("{line}");
            backend.note(line);
        }
        if trace_details {
            backend.traced += 1;
            let floats = |bytes: &[u8], count: usize| -> Vec<f32> {
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .take(count)
                    .map(|word| f32::from_le_bytes(*word))
                    .collect()
            };
            let vertices = streams
                .first()
                .map(|stream| {
                    let mut bytes = stream.data.clone();
                    if let Some((guest, length)) = stream.guest {
                        bytes = vec![0; length];
                        let _ = core.memory.read(guest, &mut bytes);
                    }
                    floats(&bytes, if traced_program { 64 } else { 9 })
                })
                .unwrap_or_default();
            let mut uniform_summary: Vec<String> = uniforms
                .iter()
                .map(|(key, bytes)| {
                    let count = if traced_program { 16 } else { 8 };
                    format!("{key:?}:{}B {:?}", bytes.len(), floats(bytes, count))
                })
                .collect();
            uniform_summary.sort();
            let polygon = self.render.properties("BindPolygonState");
            let polygon_trace = format!(
                "polygon bound {:#x?} props {polygon:?}",
                self.render
                    .state
                    .get("BindPolygonState")
                    .map(|bind| bind.x[1])
            );
            let index_sample = match &indices {
                Indices::U16(bytes) => bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .take(16)
                    .map(|index| u32::from(u16::from_le_bytes(*index)))
                    .collect::<Vec<_>>(),
                Indices::U32(bytes) => bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .take(16)
                    .map(|index| u32::from_le_bytes(*index))
                    .collect(),
                &Indices::Guest {
                    address,
                    bytes,
                    kind,
                } => {
                    let width = if kind == vk::IndexType::UINT16 { 2 } else { 4 };
                    let mut sample = [0u8; 64];
                    let sample = &mut sample[..(bytes as usize).min(16 * width)];
                    core.memory.read(address, sample).map_err(|trap| {
                        format!("reading guest indices at {address:#x}: {trap:?}")
                    })?;
                    if width == 2 {
                        sample
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .map(|word| u32::from(u16::from_le_bytes(*word)))
                            .collect()
                    } else {
                        sample
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .map(|word| u32::from_le_bytes(*word))
                            .collect()
                    }
                }
                Indices::None => Vec::new(),
            };
            let line = format!(
                "trace draw {:016x}/{:016x} {polygon_trace} {:?} count {count} indices {index_sample:?} first {first} base {base_vertex} instances {instances}+{first_instance} fetch {:x?} cull {:?} {:?} blends {:?} mesh {:?} depth {:?} {depth_storage:x?} test {} write {} {:?} viewport {:?} scissor {:?} colors {:x?} attrs {:?} v0 {vertices:?} ubo {uniform_summary:?} textures {}",
                keys.0,
                keys.1,
                topology,
                vertex_fetch
                    .iter()
                    .map(|entry| (
                        entry.address,
                        entry.stride,
                        entry.divisor,
                        entry.format,
                        entry.size
                    ))
                    .collect::<Vec<_>>(),
                fixed.cull,
                fixed.front_face,
                blends
                    .iter()
                    .map(|blend| (blend.enable, blend.color, blend.alpha, blend.write_mask))
                    .collect::<Vec<_>>(),
                mesh_groups,
                depth,
                fixed.depth_test,
                fixed.depth_write,
                fixed.depth_compare,
                fixed.viewport,
                fixed.scissor,
                colors,
                attributes
                    .iter()
                    .map(|a| (a.location, a.offset, a.format))
                    .collect::<Vec<_>>(),
                textures.len(),
            );
            // Also to the log as it is made: a run stopped early (the
            // watchdog, a hang) never writes the report the notes go to.
            if traced_program {
                eprintln!("{line}");
            }
            backend.note(line);
        }
        // ECHORA_DUMP_DRAW also dumps a mesh draw's task and mesh stages
        // (as dispatches, at the end): their inputs, kept here.
        // ECHORA_DUMP_DRAW_FROM=N: draws are dumped from present N on.
        let dump_from = super::debug_var("ECHORA_DUMP_DRAW_FROM")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(0);
        // ECHORA_DUMP_DRAW_BASE=N: only draws with base vertex N (the same
        // patch drawn by two programs, compared in the simulator).
        let base_wanted = super::debug_var("ECHORA_DUMP_DRAW_BASE")
            .ok()
            .and_then(|value| value.parse::<i32>().ok())
            .is_none_or(|wanted| wanted == base_vertex);
        let stage_dump = (self.presented_frames >= dump_from
            && base_wanted
            && super::debug_var("ECHORA_DUMP_DRAW").is_ok_and(|wanted| {
                wanted
                    .split(',')
                    .any(|item| u64::from_str_radix(item.trim(), 16) == Ok(keys.1))
            })
            && !self.dumped_dispatches.contains(&(keys.1 ^ 0x7461_736b))
            && !self.dumped_dispatches.contains(&(keys.1 ^ 0x6d65_7368))
            && !self.dumped_dispatches.contains(&(keys.1 ^ 0x7665_7274)))
        .then(|| {
            // Every uniform buffer's bytes, the guest-read ones included.
            let mut all = uniforms.clone();
            for (&key, &(guest, length)) in guest_uniforms.iter() {
                let mut bytes = vec![0; length];
                if core.memory.read(guest, &mut bytes).is_ok() {
                    all.insert(key, bytes);
                }
            }
            (stages.to_vec(), all, textures.clone(), page_table)
        });
        let vertex_fetch_dump = if stage_dump.is_some() {
            vertex_fetch.clone()
        } else {
            Vec::new()
        };
        let program_keys = if super::debug_var("ECHORA_VERTEX_TRACE_BASE")
            .ok()
            .and_then(|value| value.parse::<i32>().ok())
            == Some(base_vertex)
            && backend.programs.contains_key(&traced_variant)
        {
            traced_variant
        } else {
            built
        };
        let read_guest = |guest: u64, bytes: &mut [u8]| {
            core.memory
                .read(guest, bytes)
                .map_err(|trap| format!("reading guest bytes at {guest:#x}: {trap:?}"))
        };
        let draw = Draw {
            program: &backend.programs[&program_keys],
            blends,
            mesh: mesh_groups,
            colors,
            color_levels,
            color_layers,
            color_formats,
            depth,
            depth_layer,
            attributes,
            streams,
            indices,
            count,
            first,
            base_vertex,
            instances,
            first_instance,
            uniforms,
            guest_uniforms,
            read_guest: Some(&read_guest),
            textures,
            storage,
            storage_in_place,
            fixed,
            page_table,
            vertex_fetch,
            pool,
        };
        let mut colors_written = [0; MAX_COLOR_TARGETS];
        let mut color_places = [(0, 0); MAX_COLOR_TARGETS];
        let mut written = 0;
        for (index, &key) in draw.colors.iter().enumerate() {
            if key == 0 {
                continue;
            }
            *self.frame_target_draws.entry(key).or_default() += 1;
            colors_written[written] = key;
            color_places[written] = (draw.color_levels[index], draw.color_layers[index]);
            written += 1;
        }
        let colors_written = &colors_written[..written];
        let probe_target = colors_written.first().copied();
        let (draw_count, draw_instances) = (draw.count, draw.instances);
        let probe_level = draw.color_levels.first().copied().unwrap_or(0);
        let setup_end = timing.then(std::time::Instant::now);
        if echora_gpu::renderer::Renderer::labelling() {
            let label = format!(
                "present {} draw {:016x}/{:016x} count {} instances {} colors {:x?} depth {:x?} reads [{}]",
                self.presented_frames,
                keys.0,
                keys.1,
                draw.count,
                draw.instances,
                draw.colors,
                draw.depth,
                label_sources(draw.textures.values().map(|(source, _)| source))
            );
            backend.renderer.label(label);
        }
        let mut storage_results = 0;
        for (key, token) in backend.renderer.draw(&draw)? {
            let (Some(&(guest, base)), Some(bytes)) =
                (storage_written.get(&key), draw.storage.get(&key))
            else {
                continue;
            };
            storage_results += 1;
            backend.gpu_writes += 1;
            backend.pending_storage.push((
                token,
                guest..guest + bytes.len() as u64,
                key.1,
                keys.1,
                base,
            ));
        }
        let table_write_serial = writes_page_table.then(|| backend.renderer.recorded_serial());
        if storage_results > 0 {
            // (Not `stat`: the draw still borrows the programs.)
            *backend
                .notes
                .entry("stat: draw storage results".into())
                .or_default() += storage_results;
        }
        let timings = setup_end.zip(since).map(|(setup_end, since)| {
            [
                (setup_end - since).as_micros() as u64,
                setup_end.elapsed().as_micros() as u64,
            ]
        });
        // Only the trace switches below read it (a format per draw else).
        let draw_label = if super::debug_var_os("ECHORA_TARGET_TRACE").is_some()
            || super::debug_var_os("ECHORA_MEMORY_TRACE").is_some()
            || super::debug_var_os("ECHORA_DEPTH_TRACE_TEXEL").is_some()
        {
            format!("draw {:016x}/{:016x}", keys.0, keys.1)
        } else {
            String::new()
        };
        // ECHORA_GPU_UPLOAD_DUMP=<dir> with a traced program: each level of
        // the uploads it samples, once, as upload-<key>-<level>.ppm (RGB)
        // and .pgm (alpha).
        if let Some(directory) =
            super::debug_var_os("ECHORA_GPU_UPLOAD_DUMP").filter(|_| traced_program)
        {
            for (source, _) in draw.textures.values() {
                let TextureSource::Upload {
                    key,
                    width,
                    height,
                    levels,
                    ..
                } = source
                else {
                    continue;
                };
                for region in levels.iter().filter(|region| region.layer == 0) {
                    let level = region.level;
                    let path = std::path::Path::new(&directory)
                        .join(format!("upload-{key:x}-{width}x{height}-{level}.ppm"));
                    if path.exists() {
                        continue;
                    }
                    let Ok(texels) = backend.renderer.read_upload_level(*key, level) else {
                        continue;
                    };
                    let image = Image {
                        width: region.width,
                        height: region.height,
                        texels: texels.clone(),
                    };
                    let _ = std::fs::write(&path, image.ppm());
                    let mut pgm =
                        format!("P5\n{} {}\n255\n", region.width, region.height).into_bytes();
                    pgm.extend(texels.iter().map(|t| (t[3].clamp(0.0, 1.0) * 255.0) as u8));
                    let _ = std::fs::write(path.with_extension("pgm"), pgm);
                }
            }
        }
        // ECHORA_DUMP_DRAW=<fragment key>[,...] (debugging): the first draw
        // of each listed fragment program, for the simulator (`dump_draw`).
        if let Some(wanted) = super::debug_var_os("ECHORA_DUMP_DRAW")
            .filter(|_| self.presented_frames >= dump_from && base_wanted)
            && wanted
                .to_string_lossy()
                .split(',')
                .any(|item| u64::from_str_radix(item.trim(), 16) == Ok(keys.1))
            && self.dumped_dispatches.insert(keys.1)
        {
            let extent = probe_target
                .and_then(|key| backend.renderer.existing_target(key))
                .map(|target| (target.width, target.height));
            let result = dump_directory_at(keys.1, self.presented_frames).and_then(|directory| {
                Self::dump_draw(&mut backend.renderer, &directory, &draw, extent)
            });
            if let Err(error) = result {
                eprintln!("draw dump {:016x}: {error}", keys.1);
            }
        }
        if let Some(serial) = table_write_serial {
            backend.record_page_table_write(serial, None);
        }
        if let Some((key, directory)) = snap.filter(|snap| Some(snap.0) == probe_target) {
            let sequence = self.sequence;
            let target = backend.renderer.existing_target(key).unwrap();
            let width = target.width;
            let texels = backend.renderer.read_target_level(key, probe_level)?;
            let peak = texels
                .iter()
                .map(|t| t[0].max(t[1]).max(t[2]))
                .fold(0f32, f32::max);
            let image = Image {
                width,
                height: texels.len() as u32 / width.max(1),
                texels: texels
                    .into_iter()
                    .map(|t| t.map(|value| value / (1.0 + value.max(0.0))))
                    .collect(),
            };
            let name = format!("snap-{sequence:05}-{:016x}.ppm", keys.1);
            std::fs::write(std::path::Path::new(&directory).join(name), image.ppm())
                .map_err(|error| error.to_string())?;
            backend.note(format!("snap {sequence:05} {:016x} peak {peak}", keys.1));
        }
        // ECHORA_GPU_PROBE=<fragment key>: the first 40 draws of that
        // program note the range of what they left in color target 0.
        if super::debug_var("ECHORA_GPU_PROBE")
            .is_ok_and(|wanted| wanted == format!("{:016x}", keys.1))
            && backend.probed < 40
        {
            backend.probed += 1;
            if let Some(target) = probe_target {
                let texels = backend.renderer.read_target_level(target, probe_level)?;
                let red: Vec<f32> = texels.iter().map(|texel| texel[0]).collect();
                let max = red.iter().copied().fold(0f32, f32::max);
                let mean = red.iter().map(|value| f64::from(*value)).sum::<f64>()
                    / red.len().max(1) as f64;
                backend.note(format!(
                    "probe {:016x} {target:x} level {probe_level} mean-r {mean:.4} max-r {max:.3} #{}",
                    keys.1, backend.probed
                ));
            }
        }
        // ECHORA_GPU_PIXEL_WATCH=<first>:<end>:<target key>[+<key>...]:<x>,<y>:
        // in every 60th present from first to end, each draw into a watched
        // target that changes its pixel notes its program, instances and the
        // new value (finds which draw left a wrong pixel; reads the target
        // back after each draw). Watching several targets follows a pixel
        // through a chain of passes of the same size.
        if let Ok(watch) = super::debug_var("ECHORA_GPU_PIXEL_WATCH") {
            let fields: Vec<&str> = watch.split([':', ',']).collect();
            if let [first, end, targets, x, y] = fields[..] {
                let (first, end) = (first.parse::<u64>(), end.parse::<u64>());
                let targets: Result<Vec<u64>, _> = targets
                    .split('+')
                    .map(|target| u64::from_str_radix(target, 16))
                    .collect();
                let (x, y) = (x.parse::<usize>(), y.parse::<usize>());
                if let (Ok(first), Ok(end), Ok(targets), Ok(x), Ok(y)) = (first, end, targets, x, y)
                {
                    let present = self.presented_frames;
                    let watched = targets
                        .into_iter()
                        .filter(|target| colors_written.contains(target));
                    if (first..end).contains(&present) && present.is_multiple_of(60) {
                        for target in watched {
                            let width = backend
                                .renderer
                                .existing_target(target)
                                .map_or(1, |target| target.width as usize);
                            let texels = backend.renderer.read_target_level(target, 0)?;
                            let Some(value) = texels.get(y * width + x).copied() else {
                                continue;
                            };
                            if backend.watched_pixel.insert(target, value) != Some(value) {
                                let line = format!(
                                    "pixel watch present {present} {target:x} ({x},{y}) after {:016x}/{:016x} count {} instances {}: {value:?}",
                                    keys.0, keys.1, draw_count, draw_instances
                                );
                                backend.note(line);
                            }
                        }
                    }
                }
            }
        }
        let backend = self.gpu.as_mut().unwrap();
        if let Some([setup, renderer]) = timings {
            backend.stat("stat: draw us setup", setup);
            backend.stat("stat: draw us renderer", renderer);
        }
        if let Some((stages, uniforms, textures, page_table)) = stage_dump.as_ref()
            && !mesh_draw
        {
            self.dump_vertex_stage(
                core,
                command,
                keys.1,
                stages,
                uniforms,
                textures,
                *page_table,
                (&vertex_fetch_dump, base_vertex, first_instance),
            );
        }
        if let Some((stages, uniforms, textures, page_table)) = stage_dump.filter(|_| mesh_draw) {
            self.dump_mesh_stages(
                core, command, keys.1, &stages, &uniforms, &textures, page_table,
            );
        }
        if super::debug_var_os("ECHORA_DISPATCH_SOURCE_LOG").is_some() {
            for key in colors_written {
                let storage = self.texture_storage(*key).unwrap_or(0);
                self.target_writers
                    .insert(*key, (keys.1, self.presented_frames, storage));
            }
        }
        self.trace_targets(colors_written, &color_places[..written], &draw_label)?;
        // ECHORA_DEPTH_TRACE_TEXEL=<x>,<y> with ECHORA_DEPTH_TRACE (debugging):
        // after each draw into the traced depth memory, that texel's depth
        // when the draw changed it (which draw put a shadow caster there).
        if let (Some(target), Ok(texel)) = (depth, super::debug_var("ECHORA_DEPTH_TRACE_TEXEL"))
            && self.depth_traced(target)
            && let Some((x, y)) = texel.split_once(',').and_then(|(x, y)| {
                Some((x.trim().parse::<u32>().ok()?, y.trim().parse::<u32>().ok()?))
            })
        {
            let format = self
                .gpu
                .as_ref()
                .unwrap()
                .renderer
                .existing_target(target)
                .map(|target| target.format);
            let size = if format == Some(vk::Format::D16_UNORM) {
                2
            } else {
                4
            };
            let renderer = &mut self.gpu.as_mut().unwrap().renderer;
            renderer.flush()?;
            if let Ok(bytes) = renderer.read_target_region_bytes(target, size, Some([x, y, 1, 1]))
                && bytes.len() >= size as usize
            {
                let value = if size == 2 {
                    f32::from(u16::from_le_bytes([bytes[0], bytes[1]])) / 65535.0
                } else {
                    f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                };
                if self.depth_texel_last != Some(value.to_bits()) {
                    self.depth_texel_last = Some(value.to_bits());
                    eprintln!(
                        "depth texel present {} after {draw_label} ({x},{y}) = {value:.6}",
                        self.presented_frames
                    );
                }
            }
        }
        // A debugging trace that cannot read its range says so; the draw
        // stands.
        if let Err(error) = self.trace_memory(core, &draw_label) {
            eprintln!("memory trace: {error}");
        }
        Ok(())
    }
}

/// A memory pool mirrored in a device-local buffer while compute shaders
/// use storage in it (`Renderer::resident_buffer`, keyed by `base`). The
/// pools the game puts storage in are CPU-cached: the CPU flushes ranges it
/// wrote (FlushMappedRange) and invalidates ranges it will read
/// (InvalidateMappedRange), which keeps the two copies in step without
/// comparing bytes. `guest_dirty` ranges are newer in guest memory (uploaded
/// before the next use), `gpu_dirty` ranges newer on the GPU (copied back
/// when the CPU or a CPU-executed command needs them).
#[derive(Clone, Debug)]
struct ResidentPool {
    base: u64,
    size: u64,
    /// The CPU reads the pool without invalidating (CPU_UNCACHED): GPU
    /// writes come back at every CPU wait.
    uncached: bool,
    /// The CPU can reach it (CPU_UNCACHED or CPU_CACHED): the page table
    /// maps its guest memory, not this copy.
    host_visible: bool,
    guest_dirty: Vec<std::ops::Range<u64>>,
    gpu_dirty: Vec<std::ops::Range<u64>>,
}

impl ResidentPool {
    fn contains(&self, guest: u64) -> bool {
        (self.base..self.base + self.size).contains(&guest)
    }
}

/// Add `range` to a sorted list of disjoint ranges, merging neighbours.
/// A stat name for a copy-back's call site, made once per site.
fn caller_stat(location: &'static std::panic::Location<'static>) -> &'static str {
    thread_local! {
        static NAMES: std::cell::RefCell<HashMap<(&'static str, u32), &'static str>> =
            std::cell::RefCell::new(HashMap::default());
    }
    NAMES.with(|names| {
        *names
            .borrow_mut()
            .entry((location.file(), location.line()))
            .or_insert_with(|| {
                Box::leak(
                    format!(
                        "stat: copy-back from {}:{}",
                        location.file(),
                        location.line()
                    )
                    .into_boxed_str(),
                )
            })
    })
}

pub(super) fn merge_ranges(ranges: &mut Vec<std::ops::Range<u64>>) {
    ranges.sort_unstable_by_key(|range| range.start);
    ranges.dedup_by(|next, previous| {
        if next.start <= previous.end {
            previous.end = previous.end.max(next.end);
            true
        } else {
            false
        }
    });
}

fn ranges_overlap(ranges: &[std::ops::Range<u64>], range: &std::ops::Range<u64>) -> bool {
    let at = ranges.partition_point(|entry| entry.end <= range.start);
    ranges.get(at).is_some_and(|entry| entry.start < range.end)
}

fn add_range(ranges: &mut Vec<std::ops::Range<u64>>, range: std::ops::Range<u64>) {
    // The list is sorted and disjoint. Search for the first touching
    // range, or the position where a new range belongs.
    let low = ranges.partition_point(|existing| existing.end < range.start);
    let Some(first) = ranges
        .get(low)
        .filter(|existing| existing.start <= range.end)
    else {
        ranges.insert(low, range);
        return;
    };
    if first.start <= range.start && range.end <= first.end {
        return;
    }
    let start = first.start.min(range.start);
    let end = first.end.max(range.end);
    let following = &ranges[low + 1..];
    if let Some(next) = following.first().filter(|next| next.start <= end) {
        if following.get(1).is_some_and(|next| next.start <= end) {
            let high = low + 1 + following.partition_point(|existing| existing.start <= end);
            ranges.splice(low..high, [start..end.max(ranges[high - 1].end)]);
        } else {
            ranges[low] = start..end.max(next.end);
            ranges.remove(low + 1);
        }
    } else {
        ranges[low] = start..end;
    }
}

/// Remove `range` from sorted, disjoint ranges. Returns whether any bytes
/// were removed.
fn remove_range(ranges: &mut Vec<std::ops::Range<u64>>, range: std::ops::Range<u64>) -> bool {
    if range.is_empty() {
        return false;
    }
    let low = ranges.partition_point(|existing| existing.end <= range.start);
    let high = ranges.partition_point(|existing| existing.start < range.end);
    if low >= high {
        return false;
    }
    let remaining = [
        ranges[low].start..range.start,
        range.end..ranges[high - 1].end,
    ];
    let first = usize::from(remaining[0].is_empty());
    let end = remaining.len() - usize::from(remaining[1].is_empty());
    ranges.splice(low..high, remaining[first..end].iter().cloned());
    true
}

/// The parts of a sorted list of disjoint ranges inside `range`.
fn ranges_within(
    ranges: &[std::ops::Range<u64>],
    range: std::ops::Range<u64>,
) -> Vec<std::ops::Range<u64>> {
    let low = ranges.partition_point(|existing| existing.end <= range.start);
    ranges[low..]
        .iter()
        .take_while(|existing| existing.start < range.end)
        .map(|existing| existing.start.max(range.start)..existing.end.min(range.end))
        .collect()
}

/// Uploads from CPU-uncached pools up to this size keep a copy of their
/// storage for exact change checks (see `Backend::shadows`).
const SHADOW_LIMIT: u64 = 16 << 20;

/// Granularity of upload storage hashes (see `Backend::storage_hashes`).
const STORAGE_CHUNK: u64 = 64 << 10;

/// Read guest storage into `bytes`, substituting zeros for unreadable pages.
fn read_storage_into(core: &Guest, base: u64, mut bytes: &mut [u8]) {
    if core.memory.read(base, bytes).is_ok() {
        return;
    }
    let mut address = base;
    let page_size = echora_shader_abi::PAGE_SIZE;
    while !bytes.is_empty() {
        let available = (page_size - address % page_size) as usize;
        let (chunk, rest) = bytes.split_at_mut(available.min(bytes.len()));
        if core.memory.read(address, chunk).is_err() {
            chunk.fill(0);
        }
        address += chunk.len() as u64;
        bytes = rest;
    }
}

/// A texture storage's sample: its shape, and 256 evenly spaced 64-byte
/// chunks of its `storage_size` bytes at `base` (see `guest_texture`).
pub(super) fn texture_sample(
    core: &Guest,
    base: u64,
    storage_size: u64,
    format: u32,
    dimensions: [u32; 3],
) -> u64 {
    let sample = texture_sample_seed(format, dimensions);
    // At this size the evenly spaced samples cover the entire texture.
    // Read it once, without hashing overlapping chunks or touching bytes
    // beyond the texture when it occupies less than one sample.
    const FULL_SAMPLE: usize = echora_shader_abi::PAGE_SIZE as usize;
    if storage_size <= FULL_SAMPLE as u64 {
        let mut bytes = [0; FULL_SAMPLE];
        let bytes = &mut bytes[..storage_size as usize];
        if core.memory.read(base, bytes).is_ok() {
            return (sample ^ hash_bytes(bytes)).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        }
    }
    match &core.memory {
        super::GuestMemory::Checked(memory) => {
            sample_chunks(sample, base, storage_size, |address, bytes| {
                memory.read(address, bytes).is_ok()
            })
        }
        super::GuestMemory::View(view) => {
            sample_chunks(sample, base, storage_size, |address, bytes| {
                view.read(address, bytes).is_ok()
            })
        }
    }
}

fn texture_sample_seed(format: u32, [width, height, depth_or_layers]: [u32; 3]) -> u64 {
    u64::from(format) << 48
        ^ u64::from(width) << 24
        ^ u64::from(height)
        ^ u64::from(depth_or_layers) << 40
}

fn sample_chunks(
    sample: u64,
    base: u64,
    storage_size: u64,
    read_chunk: impl Fn(u64, &mut [u8; 64]) -> bool,
) -> u64 {
    // Gather the same chunks before hashing so their cache misses
    // overlap. A bitmap distinguishes unreadable chunks from zeroes.
    const CHUNK_COUNT: usize = 256;
    const CHUNK_BYTES: usize = 64;
    let mut data = [0u8; CHUNK_COUNT * CHUNK_BYTES + CHUNK_COUNT / 8];
    let (chunks, read_mask) = data.split_at_mut(CHUNK_COUNT * CHUNK_BYTES);
    let span = storage_size.saturating_sub(CHUNK_BYTES as u64);
    for (index, chunk) in chunks
        .as_chunks_mut::<CHUNK_BYTES>()
        .0
        .iter_mut()
        .enumerate()
    {
        let offset = span * index as u64 / (CHUNK_COUNT - 1) as u64;
        if read_chunk(base + offset, chunk) {
            read_mask[index / 8] |= 1 << (index % 8);
        } else {
            chunk.fill(0);
        }
    }
    (sample ^ hash_bytes(&data)).wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    twox_hash::XxHash3_64::oneshot(bytes)
}

/// The sRGB encoding of a linear value in [0, 1].
fn linear_to_srgb(value: f32) -> f32 {
    let value = value.clamp(0.0, 1.0);
    if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

/// Bytes one vertex attribute of `format` occupies.
fn attribute_format_size(format: vk::Format) -> u64 {
    use vk::Format as F;
    match format {
        F::R8_UNORM | F::R8_SNORM | F::R8_UINT | F::R8_SINT | F::R8_USCALED | F::R8_SSCALED => 1,
        F::R8G8_UNORM
        | F::R8G8_SNORM
        | F::R8G8_UINT
        | F::R8G8_SINT
        | F::R16_UNORM
        | F::R16_SNORM
        | F::R16_UINT
        | F::R16_SINT
        | F::R16_SFLOAT
        | F::R8G8_USCALED
        | F::R8G8_SSCALED
        | F::R16_USCALED
        | F::R16_SSCALED => 2,
        F::R8G8B8_UNORM | F::R8G8B8_SNORM | F::R8G8B8_UINT | F::R8G8B8_SINT => 3,
        F::R16G16B16_UNORM
        | F::R16G16B16_SNORM
        | F::R16G16B16_UINT
        | F::R16G16B16_SINT
        | F::R16G16B16_SFLOAT => 6,
        F::R16G16B16A16_UNORM
        | F::R16G16B16A16_SNORM
        | F::R16G16B16A16_UINT
        | F::R16G16B16A16_SINT
        | F::R16G16B16A16_SFLOAT
        | F::R32G32_SFLOAT
        | F::R32G32_UINT
        | F::R32G32_SINT
        | F::R16G16B16A16_USCALED
        | F::R16G16B16A16_SSCALED => 8,
        F::R32G32B32_SFLOAT | F::R32G32B32_UINT | F::R32G32B32_SINT => 12,
        F::R32G32B32A32_SFLOAT | F::R32G32B32A32_UINT | F::R32G32B32A32_SINT => 16,
        // Every other modeled vertex format is four bytes (RGBA8, RGB10A2,
        // R32, RG16, ...); 16 covers any larger one safely.
        F::R8G8B8A8_UNORM
        | F::R8G8B8A8_SNORM
        | F::R8G8B8A8_UINT
        | F::R8G8B8A8_SINT
        | F::R8G8B8A8_USCALED
        | F::R8G8B8A8_SSCALED
        | F::A2B10G10R10_UNORM_PACK32
        | F::A2B10G10R10_SNORM_PACK32
        | F::A2B10G10R10_UINT_PACK32
        | F::A2B10G10R10_SINT_PACK32
        | F::R32_SFLOAT
        | F::R32_UINT
        | F::R32_SINT
        | F::R16G16_UNORM
        | F::R16G16_SNORM
        | F::R16G16_UINT
        | F::R16G16_SINT
        | F::R16G16_SFLOAT
        | F::B10G11R11_UFLOAT_PACK32 => 4,
        _ => 16,
    }
}

#[cfg(test)]
mod program_tests {
    use super::{Arc, BoundProgram, Stage, Translated};

    #[test]
    fn distinct_translations_of_the_same_guest_code_get_distinct_pipelines() {
        let translation = |stage| {
            Arc::new(Translated {
                abi_version: echora_shader_abi::ABI_VERSION,
                stage,
                spirv: Vec::new(),
                descriptors: Vec::new(),
                loaded_generics: 0,
                stored_generics: 0,
                stores: [0; 8],
                frag_colors: 0,
                cbuf_used_sizes: [0; 18],
                spec_constants: Vec::new(),
                push_fields: 0,
                texture_pool_bindings: 0,
            })
        };
        let mesh = translation(Stage::Mesh);
        let fragment = translation(Stage::Fragment);
        let bind = |mesh: Arc<Translated>| {
            BoundProgram::new(
                (4, 8),
                Box::new([mesh, fragment.clone()]),
                Box::new([crate::nvn::Args::default(); 2]),
            )
        };

        let first = bind(mesh.clone());
        let same = bind(mesh);
        let variant = bind(translation(Stage::Mesh));
        assert_eq!(first.keys, variant.keys);
        assert_eq!(first.pipeline, same.pipeline);
        assert_ne!(first.pipeline, variant.pipeline);
    }
}

#[cfg(test)]
mod range_tests {
    use super::{add_range, remove_range};

    #[test]
    fn ranges_merge_and_split() {
        let mut ranges = Vec::new();
        add_range(&mut ranges, 10..20);
        add_range(&mut ranges, 30..40);
        add_range(&mut ranges, 18..32);
        assert_eq!(ranges, [10..40]);
        add_range(&mut ranges, 0..5);
        add_range(&mut ranges, 12..20);
        add_range(&mut ranges, 0..2);
        assert_eq!(ranges, [0..5, 10..40]);
        let mut bridged = vec![0..5, 10..15, 20..25, 30..35];
        add_range(&mut bridged, 5..30);
        assert_eq!(bridged.as_slice(), std::slice::from_ref(&(0..35)));
        remove_range(&mut ranges, 15..25);
        assert_eq!(ranges, [0..5, 10..15, 25..40]);
        remove_range(&mut ranges, 0..12);
        assert_eq!(ranges, [12..15, 25..40]);
        assert!(!remove_range(&mut ranges, 100..200));
        assert!(!remove_range(&mut ranges, 13..13));
        assert_eq!(ranges, [12..15, 25..40]);
    }
}

/// The recompiler's view of a compute dispatch's bindings while it
/// translates the program: which constant buffer words are mapped GPU
/// addresses (c1, the stage's constants, and the bound uniform buffers;
/// c0 is built later, so its words count as pointers).
struct PointerProbe<'a, 'g> {
    nvn: &'a Nvn,
    core: &'a Guest<'g>,
    c1: &'a [u8],
}

impl PointerProbe<'_, '_> {
    /// The eight bytes at `offset` of constant buffer `index`.
    fn word(&self, index: u32, offset: u32) -> Option<u64> {
        let at = offset as usize;
        let mut word = [0u8; 8];
        match index {
            0 | 2 => return None,
            1 => word.copy_from_slice(self.c1.get(at..at + 8)?),
            index => {
                let bind = self
                    .nvn
                    .render
                    .bound_uniforms
                    .get(5, u64::from(index - 3))?;
                if u64::from(offset) + 8 > bind.x[4] {
                    return None;
                }
                self.read(bind.x[3] + u64::from(offset), &mut word)?;
            }
        }
        Some(u64::from_le_bytes(word))
    }

    fn read(&self, gpu: u64, word: &mut [u8; 8]) -> Option<()> {
        let guest = self.nvn.gpu_to_guest(gpu, 8).ok()?;
        self.core.memory.read(guest, word).ok()
    }

    fn mapped(&self, pointer: u64) -> bool {
        pointer != 0 && self.nvn.pool_for(pointer & !15).is_some()
    }
}

impl shader::Environment for PointerProbe<'_, '_> {
    fn is_storage_pointer(&mut self, index: u32, offset: u32) -> bool {
        index == 0
            || self
                .word(index, offset)
                .is_some_and(|pointer| self.mapped(pointer))
    }

    fn is_indirect_storage_pointer(&mut self, index: u32, offset: u32, indirect: u32) -> bool {
        if index == 0 {
            return true;
        }
        let Some(table) = self.word(index, offset) else {
            return false;
        };
        let mut word = [0u8; 8];
        self.read(table.wrapping_add(u64::from(indirect)), &mut word)
            .is_some_and(|()| self.mapped(u64::from_le_bytes(word)))
    }
}

/// The most a raw storage pointer binds: its memory pool's rest, up to this
/// (copied bindings carry it whole).
const NATIVE_STORAGE_SPAN: u64 = 4 << 20;

/// The storage buffer address and size (eight and four bytes) a compute
/// program reads at `descriptor`'s place in its c0 (`c0`, as filled)
/// outside the storage buffer entries. Words the tracking finds in other
/// constant buffers were constants, not addresses (floats 1.0 and 10.0 in
/// c1), so only c0 is read.
fn constant_pointer(descriptor: &shader::Descriptor, c0: &[u8]) -> Option<(u64, u64)> {
    if descriptor.cbuf_index != 0 {
        return None;
    }
    let at = descriptor.cbuf_offset as usize;
    let bytes = c0.get(at..at + 12)?;
    let gpu = u64::from_le_bytes(bytes[..8].try_into().unwrap());
    let size = u64::from(u32::from_le_bytes(bytes[8..].try_into().unwrap()));
    Some((gpu, size))
}

/// Texture sources for a submission label: targets as `t<key>`, uploads
/// as `u<key>`.
fn label_sources<'a>(sources: impl Iterator<Item = &'a TextureSource>) -> String {
    sources
        .map(|source| match source {
            TextureSource::Target(key) | TextureSource::TargetView { key, .. } => {
                format!("t{key:x}")
            }
            TextureSource::Upload { key, .. } => format!("u{key:x}"),
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod vertex_format_tests {
    use super::vertex_format;
    use echora_shader_abi::{VertexLayout as L, VertexNumeric as N};

    #[test]
    fn vertex_formats_decode_by_name() {
        let decoded =
            |format| vertex_format(format).map(|f| (f.components, f.layout, f.numeric, f.bgra));
        assert_eq!(decoded(0).map(|f| f.0), Some(0));
        assert_eq!(decoded(1), Some((1, L::Bits8, N::Unorm, false))); // R8
        assert_eq!(decoded(19), Some((2, L::Bits16, N::Snorm, false))); // RG16SN
        assert_eq!(decoded(36), Some((3, L::Bits32, N::Sint, false))); // RGB32I
        assert_eq!(decoded(41), Some((4, L::Bits16, N::Float, false))); // RGBA16F
        assert_eq!(decoded(47), Some((4, L::Bits32, N::Uint, false))); // RGBA32UI
        assert_eq!(decoded(61), Some((4, L::Packed1010102, N::Unorm, false)));
        assert_eq!(decoded(105), Some((4, L::Packed1010102, N::Snorm, false)));
        assert_eq!(decoded(106), Some((4, L::Packed1010102, N::Sint, false)));
        assert_eq!(decoded(63), Some((3, L::Packed111110, N::Float, false)));
        assert_eq!(decoded(154), Some((4, L::Bits8, N::Unorm, true))); // BGRA8
        assert_eq!(decoded(50), None); // DEPTH16
        assert_eq!(decoded(66), None); // BC1
    }
}

/// `length` bytes of the file behind descriptor `file` at `at` (debugging
/// dumps), read without touching any mapping of it.
#[cfg(unix)]
fn read_file_range(file: i32, at: u64, length: u64) -> Option<Vec<u8>> {
    use std::os::unix::fs::FileExt as _;
    let mut bytes = vec![0u8; length as usize];
    std::fs::File::open(format!("/proc/self/fd/{file}"))
        .and_then(|file| file.read_exact_at(&mut bytes, at))
        .ok()?;
    Some(bytes)
}

/// Elsewhere the caller reads through guest memory instead.
#[cfg(not(unix))]
fn read_file_range(_file: i32, _at: u64, _length: u64) -> Option<Vec<u8>> {
    None
}

/// The directory a dump of program `key` at present `present` goes in:
/// ECHORA_DUMP_DIR, else the cache's echora/dispatch-dumps.
fn dump_directory_at(key: u64, present: u64) -> Result<std::path::PathBuf, String> {
    let root = match super::debug_var_os("ECHORA_DUMP_DIR") {
        Some(directory) => std::path::PathBuf::from(directory),
        None => std::env::var_os("XDG_CACHE_HOME")
            .map(std::path::PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME").map(|home| std::path::Path::new(&home).join(".cache"))
            })
            .ok_or("no cache directory")?
            .join("echora/dispatch-dumps"),
    };
    Ok(root.join(format!("{key:016x}-{present}")))
}

/// Whether gpu_draw measures its phases (the "stat: draw us/ns ..."
/// lines): with ECHORA_GPU_TIME (not 0) or the timeline on, as the
/// renderer's own GPU timing.
fn draw_phase_timing() -> bool {
    static GPU_TIME: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GPU_TIME.get_or_init(|| std::env::var_os("ECHORA_GPU_TIME").is_some_and(|value| value != "0"))
        || crate::timeline::on()
}

/// The render state's key for a binding of `kind` (stage `stage`, if any)
/// at `slot`, as "{kind}/{stage}/{slot}" or "{kind}/{slot}": made once
/// each (draws looked a dozen up, each formatted anew).
pub(super) fn bind_key(kind: &'static str, stage: Option<u64>, slot: u64) -> &'static str {
    // The common kinds, stages and slots come from a table made once (a
    // map entry per lookup showed ~5% of the replay thread's samples).
    const KINDS: [&str; 5] = [
        "BindTexture",
        "BindImage",
        "BindUniformBuffer",
        "BindStorageBuffer",
        "BindVertexBuffer",
    ];
    const STAGES: usize = 9;
    const SLOTS: usize = 128;
    static TABLE: std::sync::OnceLock<Vec<&'static str>> = std::sync::OnceLock::new();
    if let Some(kind_index) = KINDS.iter().position(|known| *known == kind)
        && stage.is_none_or(|stage| stage < STAGES as u64 - 1)
        && slot < SLOTS as u64
    {
        let table = TABLE.get_or_init(|| {
            let mut table = Vec::with_capacity(KINDS.len() * STAGES * SLOTS);
            for kind in KINDS {
                for stage in 0..STAGES {
                    for slot in 0..SLOTS {
                        let key = match stage {
                            0 => format!("{kind}/{slot}"),
                            stage => format!("{kind}/{}/{slot}", stage - 1),
                        };
                        table.push(&*Box::leak(key.into_boxed_str()));
                    }
                }
            }
            table
        });
        let stage_index = stage.map_or(0, |stage| stage as usize + 1);
        return table[(kind_index * STAGES + stage_index) * SLOTS + slot as usize];
    }
    thread_local! {
        static KEYS: std::cell::RefCell<
            HashMap<(usize, Option<u64>, u64), &'static str>,
        > = std::cell::RefCell::new(HashMap::default());
    }
    KEYS.with(|keys| {
        *keys
            .borrow_mut()
            .entry((kind.as_ptr() as usize, stage, slot))
            .or_insert_with(|| {
                let key = match stage {
                    Some(stage) => format!("{kind}/{stage}/{slot}"),
                    None => format!("{kind}/{slot}"),
                };
                Box::leak(key.into_boxed_str())
            })
    })
}

/// A vertex attribute's props (`parsed_vertex_state`): SetFormat's
/// format and relative offset, and its stream.
#[derive(Clone, Copy, Debug)]
pub(super) struct ParsedAttribute {
    format: u32,
    stream: u64,
    offset: u64,
}

/// A vertex stream's props (`parsed_vertex_state`).
#[derive(Clone, Copy, Debug)]
pub(super) struct ParsedStream {
    stride: u64,
    divisor: u64,
}

/// The vertex attribute and stream snapshots' props a draw reads.
#[derive(Debug)]
pub(super) struct ParsedVertexState {
    attributes: Vec<ParsedAttribute>,
    streams: Vec<ParsedStream>,
}

/// ECHORA_LOOP_TRACE: the `loop_traces` key of a draw program's stage
/// (dispatches use their program key).
fn draw_loop_trace_key(keys: (u64, u64), stage: usize) -> u64 {
    keys.0 ^ keys.1.rotate_left(21) ^ (stage as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

#[cfg(test)]
mod bind_key_tests {
    use super::bind_key;

    #[test]
    fn bind_keys_name_kind_stage_and_slot() {
        assert_eq!(bind_key("BindTexture", Some(4), 7), "BindTexture/4/7");
        assert_eq!(bind_key("BindVertexBuffer", None, 3), "BindVertexBuffer/3");
        assert_eq!(
            bind_key("BindSeparateSampler", Some(1), 2),
            "BindSeparateSampler/1/2"
        );
        assert_eq!(bind_key("BindImage", Some(0), 200), "BindImage/0/200");
        let kind = String::from("BindUniformBuffer").leak();
        assert!(std::ptr::eq(
            bind_key(kind, Some(2), 5),
            bind_key("BindUniformBuffer", Some(2), 5)
        ));
    }
}

#[cfg(test)]
mod target_tests {
    use super::{Backend, Nvn};
    use crate::nvn::{Args, Command, Guest, GuestMemory};
    use echora_guest::{cpu::Cpu, memory::Memory};
    use std::collections::BTreeMap;

    #[test]
    fn depth_textures_sample_initialized_guest_storage() {
        use crate::nvn::{Guest, GuestMemory};
        use echora_guest::{
            cpu::Cpu,
            memory::{Memory, READ, STATE_NORMAL, WRITE},
        };

        let Ok(Some(backend)) = Backend::create(None) else {
            return;
        };
        let mut nvn = Nvn {
            gpu: Some(backend),
            gpu_checked: true,
            ..Nvn::default()
        };
        let base = 0x10_0000;
        let (pool, texture) = (1, 2);
        let args = |x| crate::nvn::Args {
            x,
            ..Default::default()
        };
        let scalar = |value| args([value, 0, 0, 0]);
        let builder = BTreeMap::from([
            ("Storage".into(), args([pool, 0, 0, 0])),
            ("Width".into(), scalar(1)),
            ("Height".into(), scalar(1)),
            ("Depth".into(), scalar(2)),
            ("Target".into(), scalar(4)),
            ("Format".into(), scalar(50)),
        ]);
        let info = crate::nvn_format::info(50).unwrap();
        let layout = crate::nvn::texture_layout_layers(&builder, info, [1, 1, 1], 2, 1);
        let bytes = crate::nvn_format::block_linear_size(info, 1, 1, 1, layout, 1).0;
        nvn.initialize(
            pool,
            "MemoryPool",
            BTreeMap::from([
                ("Storage".into(), args([base, 0x10000, 0, 0])),
                ("Flags".into(), scalar(0x24)),
            ]),
        )
        .unwrap();
        nvn.initialize(texture, "Texture", builder).unwrap();
        let mut memory = Memory::new(1 << 39);
        memory
            .map(base, 0x10000, READ | WRITE, STATE_NORMAL, "depth texture")
            .unwrap();
        for (layer, value) in [0x4000u16, 0xc000].into_iter().enumerate() {
            memory
                .write(base + layer as u64 * bytes / 2, &value.to_le_bytes())
                .unwrap();
        }
        let mut cpu = Cpu::default();
        let guest = Guest {
            cpu: &mut cpu,
            memory: GuestMemory::Checked(&mut memory),
            counter: None,
        };
        let source = nvn.guest_texture(&guest, texture).unwrap();
        let echora_gpu::draw::TextureSource::Upload { levels, .. } = &source else {
            panic!("initial depth comes from guest storage");
        };
        assert_eq!(
            levels.iter().map(|level| level.offset).collect::<Vec<_>>(),
            [0, 4]
        );
        let renderer = &mut nvn.gpu.as_mut().unwrap().renderer;
        renderer.texture_view_for(&source).unwrap();
        for (layer, value) in [0x4000u16, 0xc000].into_iter().enumerate() {
            let image = renderer
                .read_upload_slice(texture, 0, layer as u32)
                .unwrap();
            assert_eq!(image.len(), 1);
            assert!((image[0][0] - f32::from(value) / 65535.0).abs() < 1e-5);
        }
    }

    fn texture_cache_fixture(flags: u64) -> Option<(Nvn, Memory, u64, u64)> {
        use echora_guest::memory::{READ, STATE_NORMAL, WRITE};

        let backend = Backend::create(None).ok()??;
        let mut nvn = Nvn {
            gpu: Some(backend),
            gpu_checked: true,
            ..Nvn::default()
        };
        let (pool, texture) = (1, 2);
        let base = 0x10_0000;
        let size = 2 << 20;
        let args = |x| Args {
            x,
            ..Default::default()
        };
        let scalar = |value| args([value, 0, 0, 0]);
        nvn.initialize(
            pool,
            "MemoryPool",
            BTreeMap::from([
                ("Storage".into(), args([base, size, 0, 0])),
                ("Flags".into(), scalar(flags)),
            ]),
        )
        .unwrap();
        nvn.initialize(
            texture,
            "Texture",
            BTreeMap::from([
                ("Storage".into(), args([pool, 0, 0, 0])),
                ("Width".into(), scalar(1024)),
                ("Height".into(), scalar(512)),
                ("Stride".into(), scalar(4096)),
                ("Format".into(), scalar(37)),
            ]),
        )
        .unwrap();
        let mut memory = Memory::new(1 << 39);
        memory
            .map(base, size, READ | WRITE, STATE_NORMAL, "texture storage")
            .unwrap();
        Some((nvn, memory, base, texture))
    }

    #[test]
    fn one_dimensional_array_upload_uses_height_for_layers() {
        use echora_gpu::draw::ImageShape;
        let Some((mut nvn, mut memory, base, texture)) = texture_cache_fixture(0x24) else {
            return;
        };
        let props = &mut nvn.objects.get_mut(&texture).unwrap().builder;
        for (key, value) in [("Width", 8), ("Height", 3), ("Target", 3), ("Levels", 2)] {
            props.insert(
                key.into(),
                Args {
                    x: [value, 0, 0, 0],
                    ..Default::default()
                },
            );
        }
        props.remove("Stride");
        let (size, _) = crate::nvn::texture_storage(props).unwrap();
        assert_eq!(nvn.texture_storage_size(texture), Some(size));

        let mut cpu = Cpu::default();
        let mut guest = Guest {
            cpu: &mut cpu,
            memory: GuestMemory::Checked(&mut memory),
            counter: Some(0),
        };
        let linear = base + (1 << 20);
        for layer in 0..3 {
            guest
                .memory
                .write(linear + layer * 32, &[0x40 + layer as u8; 32])
                .unwrap();
            guest
                .memory
                .write(linear + 96 + layer * 16, &[0x80 + layer as u8; 16])
                .unwrap();
        }
        for (level, width, address) in [(0, 8, linear), (1, 4, linear + 96)] {
            assert!(
                nvn.copy_texels(
                    &mut guest,
                    texture,
                    crate::nvn::TexelTransfer {
                        region: [0, 0, 0, width, 3, 1],
                        bases: [level, 0],
                        strides: [0; 2],
                    },
                    address,
                    true,
                )
                .is_ok()
            );
        }
        let source = nvn.guest_texture(&guest, texture).unwrap();
        let super::TextureSource::Upload {
            height,
            shape,
            texels,
            levels,
            ..
        } = &source
        else {
            panic!("expected a texture upload");
        };
        assert_eq!(*height, 1);
        assert_eq!(*shape, ImageShape::D1 { layers: 3 });
        assert_eq!(levels.len(), 6);
        for (index, region) in levels.iter().enumerate() {
            let (level, layer) = (index / 3, index % 3);
            assert_eq!(
                (region.level as usize, region.layer as usize),
                (level, layer)
            );
            assert_eq!(region.height, 1);
            let start = region.offset as usize;
            let end = start + region.width as usize * 4;
            assert!(
                texels[start..end]
                    .iter()
                    .all(|&byte| byte == ((level + 1) * 0x40 + layer) as u8)
            );
        }
        let renderer = &mut nvn.gpu.as_mut().unwrap().renderer;
        renderer.texture_view_for(&source).unwrap();
        for level in 0..2 {
            for layer in 0..3 {
                let pixels = renderer.read_upload_slice(texture, level, layer).unwrap();
                assert_eq!(pixels.len(), 8 >> level);
                let expected = ((level + 1) * 0x40 + layer) as f32 / 255.0;
                for channel in pixels.iter().flatten() {
                    assert!((channel - expected).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn array_render_targets_keep_layers_above_sixty_four() {
        let Some((mut nvn, mut memory, _, array)) = texture_cache_fixture(0x24) else {
            return;
        };
        let mut cpu = Cpu::default();
        let guest = Guest {
            cpu: &mut cpu,
            memory: GuestMemory::Checked(&mut memory),
            counter: None,
        };
        let builder = nvn.objects[&array].builder.clone();
        let red = [1.0_f32, 0.0, 0.0, 1.0];
        let green = [0.0_f32, 1.0, 0.0, 1.0];
        for (texture, target, last_layer) in [(array + 1, 4, 65), (array + 2, 9, 71)] {
            let mut props = builder.clone();
            props.remove("Stride");
            props.get_mut("Storage").unwrap().x[1] = (texture - array) << 16;
            for (key, value) in [
                ("Width", 4),
                ("Height", 4),
                ("Depth", last_layer + 1),
                ("Target", target),
            ] {
                props.insert(
                    key.into(),
                    Args {
                        x: [value, 0, 0, 0],
                        ..Default::default()
                    },
                );
            }
            nvn.initialize(texture, "Texture", props).unwrap();
            let clears = [(0, green), (last_layer - 1, green), (last_layer, red)];
            for (layer, color) in clears {
                nvn.gpu_clear_color(
                    &guest,
                    (texture, 0, layer as u32),
                    color.map(f32::to_bits),
                    None,
                )
                .unwrap();
            }
            let renderer = &mut nvn.gpu.as_mut().unwrap().renderer;
            for (layer, color) in clears {
                let pixels = renderer
                    .read_target_slice(texture, 0, layer as u32)
                    .unwrap();
                assert_eq!(pixels, vec![color; 16]);
            }
        }
    }

    #[test]
    fn clearing_the_base_layer_preserves_the_other_texture_subresources() {
        let Some((mut nvn, mut memory, base, texture)) = texture_cache_fixture(0x24) else {
            return;
        };
        let props = &mut nvn.objects.get_mut(&texture).unwrap().builder;
        props.remove("Stride");
        for (key, value) in [
            ("Width", 8),
            ("Height", 8),
            ("Depth", 2),
            ("Target", 4),
            ("Levels", 2),
        ] {
            props.insert(
                key.into(),
                Args {
                    x: [value, 0, 0, 0],
                    ..Default::default()
                },
            );
        }
        let mut cpu = Cpu::default();
        let mut guest = Guest {
            cpu: &mut cpu,
            memory: GuestMemory::Checked(&mut memory),
            counter: None,
        };
        let address = base + (1 << 20);
        for (level, byte) in [(0, 0x40), (1, 0x80)] {
            let extent = 8 >> level;
            guest
                .memory
                .write(address, &vec![byte; (extent * extent * 2 * 4) as usize])
                .unwrap();
            assert!(
                nvn.copy_texels(
                    &mut guest,
                    texture,
                    crate::nvn::TexelTransfer {
                        region: [0, 0, 0, extent, extent, 2],
                        bases: [level, 0],
                        strides: [0; 2],
                    },
                    address,
                    true,
                )
                .is_ok()
            );
        }
        let red = [1.0_f32, 0.0, 0.0, 1.0];
        nvn.gpu_clear_color(&guest, (texture, 0, 0), red.map(f32::to_bits), None)
            .unwrap();
        let renderer = &mut nvn.gpu.as_mut().unwrap().renderer;
        assert_eq!(
            renderer.read_target_slice(texture, 0, 0).unwrap(),
            vec![red; 64]
        );
        for (level, layer, byte) in [(0, 1, 0x40), (1, 0, 0x80), (1, 1, 0x80)] {
            let pixels = renderer.read_target_slice(texture, level, layer).unwrap();
            let channels = (8 >> level) * (8 >> level);
            assert_eq!(pixels.len(), channels);
            let expected = byte as f32 / 255.0;
            assert!(
                pixels
                    .iter()
                    .flatten()
                    .all(|pixel| (pixel - expected).abs() < 1e-6)
            );
        }
    }

    #[test]
    fn a_small_flush_refreshes_a_large_cached_texture() {
        use echora_gpu::draw::TextureSource;

        let Some((mut nvn, mut memory, base, texture)) = texture_cache_fixture(0x24) else {
            return;
        };
        let mut cpu = Cpu::default();
        let mut guest = Guest {
            cpu: &mut cpu,
            memory: GuestMemory::Checked(&mut memory),
            counter: None,
        };
        let rgba = [255, 128, 64, 255];
        let offset = 256;
        let TextureSource::Upload { generation, .. } = nvn.guest_texture(&guest, texture).unwrap()
        else {
            panic!("expected the initial texture upload");
        };
        nvn.gpu_invalidate(base + offset..base + offset + 4, None);
        nvn.guest_texture(&guest, texture).unwrap();
        assert_eq!(nvn.gpu.as_ref().unwrap().stats["stat: texture uploads"], 1);
        guest.memory.write(base + offset, &rgba).unwrap();
        nvn.gpu_invalidate(base + offset..base + offset + 4, None);
        let TextureSource::Upload {
            generation: updated,
            texels,
            ..
        } = nvn.guest_texture(&guest, texture).unwrap()
        else {
            panic!("expected the refreshed texture upload");
        };
        assert_ne!(updated, generation);
        assert_eq!(texels[offset as usize..][..4], rgba);
    }

    #[test]
    fn virtual_texture_uploads_follow_each_mapping_and_its_flushes() {
        let Some((mut nvn, mut memory, base, texture)) = texture_cache_fixture(0x24) else {
            return;
        };
        let mut cpu = Cpu::default();
        let mut guest = Guest {
            cpu: &mut cpu,
            memory: GuestMemory::Checked(&mut memory),
            counter: Some(0),
        };
        let physical_pool = 1;
        let virtual_pool = 3;
        let args = |x| Args {
            x,
            ..Args::default()
        };
        nvn.initialize(
            virtual_pool,
            "MemoryPool",
            BTreeMap::from([("Storage".into(), args([0, 2 << 20, 0, 0]))]),
        )
        .unwrap();
        nvn.objects
            .get_mut(&texture)
            .unwrap()
            .builder
            .insert("Storage".into(), args([virtual_pool, 0, 0, 0]));
        let request = base + 0x10_0000;
        let map = |nvn: &mut Nvn, guest: &mut Guest, words: [u64; 4]| {
            for (index, word) in words.into_iter().enumerate() {
                guest
                    .memory
                    .write(request + index as u64 * 8, &word.to_le_bytes())
                    .unwrap();
            }
            guest.cpu.regs[..3].copy_from_slice(&[virtual_pool, 1, request]);
            assert!(nvn.call("nvnMemoryPoolMapVirtual", guest).unwrap());
            assert_eq!(guest.cpu.regs[0], 1);
        };
        map(&mut nvn, &mut guest, [physical_pool, 0x60_000, 0, 0x1000]);
        map(
            &mut nvn,
            &mut guest,
            [physical_pool, 0x40_000, 0x2000, 0x1000],
        );
        let red = [255, 0, 0, 255];
        let green = [0, 255, 0, 255];
        let blue = [0, 0, 255, 255];
        guest.memory.write(base + 0x60_000, &red).unwrap();
        // These bytes follow the first allocation physically but are not
        // mapped into the corresponding virtual offsets.
        guest.memory.write(base + 0x61_000, &blue).unwrap();
        guest.memory.write(base + 0x62_100, &blue).unwrap();
        guest.memory.write(base + 0x2100, &blue).unwrap();
        let physical = base + 0x40_100;
        guest.memory.write(physical, &green).unwrap();
        let (descriptor_pool, index) = (4, 256);
        nvn.render.state.insert(
            "SetTexturePool",
            Command {
                name: "nvnCommandBufferSetTexturePool",
                x: [0, descriptor_pool, 0, 0, 0, 0, 0, 0],
                v: [0; 8],
                captured: Vec::new(),
                states: Default::default(),
            },
        );
        nvn.texture_descriptors
            .entry(descriptor_pool)
            .or_default()
            .insert(index, args([texture, 0, 0, 0]));
        let upload = |nvn: &mut Nvn, guest: &mut Guest| {
            let entries = nvn.pool_entries(guest, 0, &[u64::from(index)]).unwrap();
            let super::TextureSource::Upload { texels, .. } = &entries[0].source else {
                panic!("expected a texture upload");
            };
            texels.clone()
        };
        let texels = upload(&mut nvn, &mut guest);
        assert_eq!(texels[..4], red);
        assert_eq!(texels[0x1000..0x1004], [0; 4]);
        assert_eq!(texels[0x2100..0x2104], green);

        // A newer GPU mirror is outside the range starting at the first
        // physical allocation. Refresh the memoized upload in this present.
        let backend = nvn.gpu.as_mut().unwrap();
        let resident = base + 0x40_000;
        let gpu_write = physical..physical + 4;
        backend.renderer.resident_buffer(resident, 0x3000).unwrap();
        backend
            .renderer
            .upload_resident(resident, 0x100, &blue)
            .unwrap();
        backend.resident.insert(
            resident,
            super::ResidentPool {
                base: resident,
                size: 0x3000,
                uncached: true,
                host_visible: true,
                guest_dirty: Vec::new(),
                gpu_dirty: vec![gpu_write.clone()],
            },
        );
        super::add_range(&mut backend.dirty_index.get_mut().1, gpu_write);
        backend.gpu_writes += 1;
        assert_eq!(upload(&mut nvn, &mut guest)[0x2100..0x2104], blue);
        assert_eq!(
            guest.memory.read_u32(physical).unwrap(),
            u32::from_le_bytes(blue)
        );

        guest.memory.write(physical, &red).unwrap();
        nvn.gpu_invalidate(physical..physical + 4, None);
        assert_eq!(upload(&mut nvn, &mut guest)[0x2100..0x2104], red);

        // Transfer three pitch rows with an unmapped middle row. The
        // neighboring physical pages hold unrelated data.
        let input = base + 0x12_0000;
        let output = base + 0x13_0000;
        for (row, sample) in [0x31, 0x52, 0x73].into_iter().enumerate() {
            guest
                .memory
                .write(input + row as u64 * 0x1000, &[sample; 0x1000])
                .unwrap();
        }
        let transfer = crate::nvn::TexelTransfer {
            region: [0, 0, 0, 1024, 3, 1],
            bases: [0; 2],
            strides: [0; 2],
        };
        assert!(
            nvn.copy_texels(&mut guest, texture, transfer, input, true)
                .is_ok()
        );
        assert_eq!(
            guest.memory.read_u32(base + 0x61_000).unwrap(),
            u32::from_le_bytes(blue)
        );
        assert_eq!(
            guest.memory.read_u32(base + 0x62_100).unwrap(),
            u32::from_le_bytes(blue)
        );
        let updated = upload(&mut nvn, &mut guest);
        assert_eq!(updated[0..4], [0x31; 4]);
        assert_eq!(updated[0x2100..0x2104], [0x73; 4]);
        assert!(
            nvn.copy_texels(&mut guest, texture, transfer, output, false)
                .is_ok()
        );
        let mut read = [0xff; 0x3000];
        guest.memory.read(output, &mut read).unwrap();
        assert_eq!(&read[..0x1000], &[0x31; 0x1000]);
        assert_eq!(&read[0x1000..0x2000], &[0; 0x1000]);
        assert_eq!(&read[0x2000..], &[0x73; 0x1000]);

        map(&mut nvn, &mut guest, [0, 0, 0, 0x1000]);
        map(
            &mut nvn,
            &mut guest,
            [physical_pool, 0x2000, 0x2000, 0x1000],
        );
        let texels = upload(&mut nvn, &mut guest);
        assert_eq!(texels[..4], [0; 4]);
        assert_eq!(texels[0x2100..0x2104], blue);
    }

    #[test]
    fn decoded_astc_views_keep_their_color_space_and_mip_range() {
        use super::{TextureSource, vk};
        use echora_gpu::draw::ImageShape;

        let mut nvn = Nvn::default();
        let (pool, index) = (1, 3);
        let args = |x| Args {
            x,
            ..Args::default()
        };
        nvn.render.state.insert(
            "SetTexturePool",
            Command {
                name: "nvnCommandBufferSetTexturePool",
                x: [0, pool, 0, 0, 0, 0, 0, 0],
                v: [0; 8],
                captured: Vec::new(),
                states: Default::default(),
            },
        );
        for (case, (encoded, view, upload, expected)) in [
            (
                121,
                135,
                vk::Format::R8G8B8A8_UNORM,
                vk::Format::R8G8B8A8_SRGB,
            ),
            (
                135,
                121,
                vk::Format::R8G8B8A8_SRGB,
                vk::Format::R8G8B8A8_UNORM,
            ),
            (
                121,
                135,
                vk::Format::ASTC_4X4_UNORM_BLOCK,
                vk::Format::ASTC_4X4_SRGB_BLOCK,
            ),
            (
                121,
                136,
                vk::Format::R8G8B8A8_UNORM,
                vk::Format::ASTC_5X4_SRGB_BLOCK,
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let texture = 2 + case as u64;
            nvn.initialize(
                texture,
                "Texture",
                BTreeMap::from([("Format".into(), args([encoded, 0, 0, 0]))]),
            )
            .unwrap();
            nvn.views.insert(
                (pool, index),
                crate::nvn::Bag {
                    kind: "TextureView".into(),
                    props: BTreeMap::from([
                        ("Format".into(), args([view, 0, 0, 0])),
                        ("Levels".into(), args([1, 0, 0, 0])),
                        ("Layers".into(), args([1, 2, 0, 0])),
                    ]),
                    ..Default::default()
                },
            );
            let source = TextureSource::Upload {
                key: texture,
                generation: 1,
                width: 4,
                height: 4,
                shape: ImageShape::D2 { layers: 4 },
                format: upload,
                texels: Default::default(),
                levels: Default::default(),
            };
            let sampler = nvn.binding_sampler(u64::from(index), texture, &source);
            assert_eq!(sampler.format, Some(expected));
            assert_eq!(sampler.view_levels, Some((1, 0)));
            assert_eq!(sampler.view_layers, Some((1, 2)));
        }
    }

    #[test]
    fn an_uncached_texture_sees_a_write_after_many_unchanged_frames() {
        use echora_gpu::draw::TextureSource;

        let Some((mut nvn, mut memory, base, texture)) = texture_cache_fixture(0x22) else {
            return;
        };
        let mut cpu = Cpu::default();
        let mut guest = Guest {
            cpu: &mut cpu,
            memory: GuestMemory::Checked(&mut memory),
            counter: None,
        };
        let TextureSource::Upload { generation, .. } = nvn.guest_texture(&guest, texture).unwrap()
        else {
            panic!("expected the initial texture upload");
        };
        for _ in 0..128 {
            nvn.presented_frames += 1;
            nvn.guest_texture(&guest, texture).unwrap();
        }
        assert_eq!(nvn.gpu.as_ref().unwrap().stats["stat: texture uploads"], 1);
        nvn.presented_frames += 1;
        let rgba = [32, 64, 128, 255];
        let offset = 256;
        guest.memory.write(base + offset, &rgba).unwrap();
        let TextureSource::Upload {
            generation: updated,
            texels,
            ..
        } = nvn.guest_texture(&guest, texture).unwrap()
        else {
            panic!("expected the refreshed texture upload");
        };
        assert_ne!(updated, generation);
        assert_eq!(texels[offset as usize..][..4], rgba);
    }

    #[test]
    fn finalized_render_target_stays_visible_through_a_storage_alias() {
        use crate::nvn::{Bag, Guest, GuestMemory, retired_texture, write_shared};
        use echora_gpu::draw::TextureSource;
        use echora_guest::{cpu::Cpu, memory::Memory};

        let Ok(Some(backend)) = Backend::create(None) else {
            return;
        };
        let mut nvn = Nvn {
            gpu: Some(backend),
            gpu_checked: true,
            ..Nvn::default()
        };
        let args = |x| Args { x, v: [0; 2] };
        let scalar = |value| args([value, 0, 0, 0]);
        nvn.initialize(
            1,
            "MemoryPool",
            BTreeMap::from([("Storage".into(), args([0x10000, 0x10000, 0, 0]))]),
        )
        .unwrap();
        nvn.initialize(5, "Device", BTreeMap::new()).unwrap();
        let builder = BTreeMap::from([
            ("Device".into(), scalar(5)),
            ("Width".into(), scalar(8)),
            ("Height".into(), scalar(4)),
            ("Format".into(), scalar(52)),
            ("Storage".into(), args([1, 0, 0, 0])),
        ]);
        nvn.initialize(2, "Texture", builder.clone()).unwrap();
        nvn.initialize(3, "Texture", builder.clone()).unwrap();
        write_shared(nvn.bags.at(4)).insert(
            4,
            Bag {
                kind: "TextureBuilder".into(),
                props: builder,
                ..Bag::default()
            },
        );
        let mut cpu = Cpu::default();
        let mut memory = Memory::new(1 << 39);
        cpu.regs[0] = 2;
        let mut core = Guest {
            cpu: &mut cpu,
            memory: GuestMemory::Checked(&mut memory),
            counter: None,
        };
        nvn.gpu_clear_depth(&core, 2, Some(0.5), None).unwrap();
        assert_eq!(nvn.target_shape(2).unwrap().0, 8);
        assert!(nvn.call("nvnTextureFinalize", &mut core).unwrap());
        assert!(nvn.target_shape(2).is_err());

        let TextureSource::Target(key) = nvn.texture_source(&core, 3).unwrap() else {
            panic!("the alias must sample the retained target");
        };
        assert_eq!(key, retired_texture((0x10000, 52)));
        let renderer = &mut nvn.gpu.as_mut().unwrap().renderer;
        assert!(renderer.existing_target(2).is_none());
        assert_eq!(
            renderer.read_target_bytes(key, 4).unwrap(),
            0.5_f32.to_le_bytes().repeat(32)
        );

        core.cpu.regs[..2].copy_from_slice(&[2, 4]);
        assert!(nvn.call("nvnTextureInitialize", &mut core).unwrap());
        assert_eq!(nvn.resolve_target(3), Some((2, 0)));
        let renderer = &mut nvn.gpu.as_mut().unwrap().renderer;
        assert!(renderer.existing_target(key).is_none());
        assert_eq!(
            renderer.read_target_bytes(2, 4).unwrap(),
            0.5_f32.to_le_bytes().repeat(32)
        );
    }

    #[test]
    fn archived_target_stays_visible_through_a_storage_alias() {
        use crate::nvn::{Bag, Guest, GuestMemory, retired_texture, write_shared};
        use echora_gpu::draw::TextureSource;
        use echora_guest::{cpu::Cpu, memory::Memory};

        let Ok(Some(mut backend)) = Backend::create(None) else {
            return;
        };
        backend
            .renderer
            .set_resolution_scale(echora_gpu::renderer::Scale::ONE);
        let mut nvn = Nvn {
            gpu: Some(backend),
            gpu_checked: true,
            ..Nvn::default()
        };
        let args = |x| Args { x, v: [0; 2] };
        let scalar = |value| args([value, 0, 0, 0]);
        nvn.initialize(
            1,
            "MemoryPool",
            BTreeMap::from([("Storage".into(), args([0x10000, 0x20_0000, 0, 0]))]),
        )
        .unwrap();
        nvn.initialize(5, "Device", BTreeMap::new()).unwrap();
        let builder = BTreeMap::from([
            ("Device".into(), scalar(5)),
            ("Width".into(), scalar(512)),
            ("Height".into(), scalar(512)),
            ("Format".into(), scalar(52)),
            ("Storage".into(), args([1, 0, 0, 0])),
        ]);
        nvn.initialize(2, "Texture", builder.clone()).unwrap();
        nvn.initialize(3, "Texture", builder.clone()).unwrap();
        write_shared(nvn.bags.at(4)).insert(
            4,
            Bag {
                kind: "TextureBuilder".into(),
                props: builder,
                ..Bag::default()
            },
        );
        let mut cpu = Cpu::default();
        let mut memory = Memory::new(1 << 39);
        cpu.regs[0] = 2;
        let mut core = Guest {
            cpu: &mut cpu,
            memory: GuestMemory::Checked(&mut memory),
            counter: None,
        };
        nvn.gpu_clear_depth(&core, 2, Some(0.5), None).unwrap();
        assert!(nvn.call("nvnTextureFinalize", &mut core).unwrap());
        let archive = retired_texture((0x10000, 52));
        let renderer = &mut nvn.gpu.as_mut().unwrap().renderer;
        let image = renderer.existing_target(archive).unwrap();
        assert_eq!(image.extent(0), (512, 512));
        assert_eq!(image.levels, 1);
        assert_eq!(
            renderer.queue_target_archive(archive).unwrap(),
            echora_gpu::renderer::ArchiveRequest::Queued
        );
        renderer.flush().unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while renderer.existing_target(archive).is_some() {
            assert!(std::time::Instant::now() < deadline);
            renderer.finish_target_archives().unwrap();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        let TextureSource::Target(key) = nvn.texture_source(&core, 3).unwrap() else {
            panic!("the alias must sample the retained target");
        };
        assert_eq!(key, retired_texture((0x10000, 52)));
        let renderer = &mut nvn.gpu.as_mut().unwrap().renderer;
        assert!(renderer.existing_target(2).is_none());
        assert_eq!(
            renderer.read_target_bytes(key, 4).unwrap(),
            0.5_f32.to_le_bytes().repeat(512 * 512)
        );

        core.cpu.regs[..2].copy_from_slice(&[2, 4]);
        assert!(nvn.call("nvnTextureInitialize", &mut core).unwrap());
        assert_eq!(nvn.resolve_target(3), Some((2, 0)));
        let renderer = &mut nvn.gpu.as_mut().unwrap().renderer;
        assert!(renderer.existing_target(key).is_none());
        assert_eq!(
            renderer.read_target_bytes(2, 4).unwrap(),
            0.5_f32.to_le_bytes().repeat(512 * 512)
        );
    }

    #[test]
    fn first_scissored_depth_clear_preserves_the_rest_of_guest_storage() {
        let Some((mut nvn, mut memory, base, texture)) = texture_cache_fixture(0x24) else {
            return;
        };
        let props = &mut nvn.objects.get_mut(&texture).unwrap().builder;
        for (key, value) in [("Width", 8), ("Height", 4), ("Stride", 32), ("Format", 52)] {
            props.insert(
                key.into(),
                Args {
                    x: [value, 0, 0, 0],
                    ..Default::default()
                },
            );
        }
        memory
            .write(base, &0.25_f32.to_le_bytes().repeat(32))
            .unwrap();
        let mut cpu = Cpu::default();
        let guest = Guest {
            cpu: &mut cpu,
            memory: GuestMemory::Checked(&mut memory),
            counter: None,
        };
        nvn.render.scissor = Some([0, 0, 4, 4]);
        nvn.gpu_clear_depth(&guest, texture, Some(0.75), None)
            .unwrap();
        let renderer = &mut nvn.gpu.as_mut().unwrap().renderer;
        let pixels = renderer.read_target_bytes(texture, 4).unwrap();
        let expected: Vec<_> = [0.75_f32; 4]
            .into_iter()
            .chain([0.25; 4])
            .flat_map(f32::to_le_bytes)
            .collect();
        assert_eq!(pixels, expected.repeat(4));
    }

    #[test]
    fn partial_and_conditional_color_clears_keep_guest_texels() {
        use echora_guest::memory::{READ, STATE_NORMAL, WRITE};

        let Ok(Some(mut backend)) = Backend::create(None) else {
            return;
        };
        backend
            .renderer
            .set_resolution_scale(echora_gpu::renderer::Scale::ONE);
        let mut nvn = Nvn {
            gpu: Some(backend),
            gpu_checked: true,
            ..Nvn::default()
        };
        let base = 0x10000;
        let args = |x| Args {
            x,
            ..Args::default()
        };
        let scalar = |value| args([value, 0, 0, 0]);
        nvn.initialize(
            1,
            "MemoryPool",
            BTreeMap::from([("Storage".into(), args([base, 0x2000, 0, 0]))]),
        )
        .unwrap();
        let mut memory = Memory::new(1 << 39);
        memory
            .map(base, 0x2000, READ | WRITE, STATE_NORMAL, "color targets")
            .unwrap();
        for (texture, offset) in [(2, 0), (3, 0x1000)] {
            nvn.initialize(
                texture,
                "Texture",
                BTreeMap::from([
                    ("Width".into(), scalar(2)),
                    ("Height".into(), scalar(1)),
                    ("Format".into(), scalar(10)),
                    ("Storage".into(), args([1, offset, 0, 0])),
                ]),
            )
            .unwrap();
            memory.write_u32(base + offset, 0.25_f32.to_bits()).unwrap();
            memory
                .write_u32(base + offset + 4, 0.5_f32.to_bits())
                .unwrap();
        }
        let mut cpu = Cpu::default();
        let mut core = Guest {
            cpu: &mut cpu,
            memory: GuestMemory::Checked(&mut memory),
            counter: None,
        };
        let mut clear = Command {
            name: "nvnCommandBufferClearTexture",
            x: [0, 2, 0, 0, 0, 0xf, 0, 0],
            v: [0; 8],
            captured: vec![
                u64::from(0.75_f32.to_bits()),
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                0,
                0,
            ],
            states: Default::default(),
        };
        nvn.execute_command(&clear, &mut core).unwrap();
        let renderer = &mut nvn.gpu.as_mut().unwrap().renderer;
        let expected = [0.75_f32, 0.5].map(f32::to_le_bytes).concat();
        assert_eq!(renderer.read_target_bytes(2, 4).unwrap(), expected);

        if renderer.supports_conditions() {
            let (predicate, _) = renderer.resident_buffer(4, 16).unwrap();
            renderer.fill_resident(4, 0, 16, 0).unwrap();
            renderer.set_condition(Some((predicate, 0, false)));
            clear.x[1] = 3;
            clear.captured.truncate(6);
            nvn.execute_command(&clear, &mut core).unwrap();
            let renderer = &mut nvn.gpu.as_mut().unwrap().renderer;
            let expected = [0.25_f32, 0.5].map(f32::to_le_bytes).concat();
            assert_eq!(renderer.read_target_bytes(3, 4).unwrap(), expected);
        }
    }

    #[test]
    fn mip_storage_aliases_require_compatible_formats() {
        use super::{TextureSource, vk};

        let Ok(Some(backend)) = Backend::create(None) else {
            return;
        };
        let mut nvn = Nvn {
            gpu: Some(backend),
            gpu_checked: true,
            ..Nvn::default()
        };
        let args = |x| Args { x, v: [0; 2] };
        let scalar = |value| args([value, 0, 0, 0]);
        let pool = 1;
        nvn.initialize(
            pool,
            "MemoryPool",
            BTreeMap::from([("Storage".into(), args([0x10000, 0x10000, 0, 0]))]),
        )
        .unwrap();
        let texture = |size, levels, offset, format| {
            BTreeMap::from([
                ("Width".into(), scalar(size)),
                ("Height".into(), scalar(size)),
                ("Levels".into(), scalar(levels)),
                ("Format".into(), scalar(format)),
                ("Storage".into(), args([pool, offset, 0, 0])),
            ])
        };
        let parent = 2;
        nvn.initialize(parent, "Texture", texture(16, 4, 0, 37))
            .unwrap();
        nvn.ensure_render_target(parent).unwrap();
        let target = nvn
            .gpu
            .as_ref()
            .unwrap()
            .renderer
            .existing_target(parent)
            .unwrap();
        let (_, offset, _) = nvn
            .target_level_offsets(parent, target.into())
            .next()
            .unwrap();
        nvn.initialize(3, "Texture", texture(8, 2, offset, 37))
            .unwrap();
        nvn.initialize(4, "Texture", texture(8, 1, offset, 56))
            .unwrap();
        nvn.initialize(5, "Texture", texture(8, 1, offset, 10))
            .unwrap();

        for indexed in [false, true] {
            if indexed {
                nvn.refresh_target_index();
            }
            assert_eq!(nvn.resolve_target(3), Some((parent, 1)));
            assert_eq!(nvn.resolve_target(4), Some((parent, 1)));
            assert_eq!(nvn.resolve_target(5), None);
        }
        let mut cpu = Cpu::default();
        let mut memory = Memory::new(1 << 39);
        let core = Guest {
            cpu: &mut cpu,
            memory: GuestMemory::Checked(&mut memory),
            counter: None,
        };
        let source = nvn.texture_source_resolved(&core, 3).unwrap();
        assert!(matches!(
            source,
            TextureSource::TargetView {
                key: 2,
                base_level: 1,
                level_count: 2,
                ..
            }
        ));
        nvn.render.state.insert(
            "SetTexturePool",
            Command {
                name: "nvnCommandBufferSetTexturePool",
                x: [0, pool, 0, 0, 0, 0, 0, 0],
                v: [0; 8],
                captured: Vec::new(),
                states: Default::default(),
            },
        );
        nvn.views.insert(
            (pool, 3),
            crate::nvn::Bag {
                kind: "TextureView".into(),
                props: BTreeMap::from([("Levels".into(), args([1, 0, 0, 0]))]),
                ..Default::default()
            },
        );
        assert!(matches!(
            nvn.through_view(source, 3),
            TextureSource::TargetView {
                key: 2,
                base_level: 2,
                level_count: 1,
                ..
            }
        ));
        let srgb = nvn.texture_source_resolved(&core, 4).unwrap();
        let sampler = nvn.binding_sampler(4, 4, &srgb);
        assert_eq!(sampler.format, Some(vk::Format::R8G8B8A8_SRGB));
        nvn.ensure_render_target(3).unwrap();
        assert_eq!(nvn.resolve_target(4), Some((3, 0)));
        nvn.refresh_target_index();
        assert_eq!(nvn.resolve_target(4), Some((3, 0)));

        // Two compatible images of the same extent and storage are
        // present. Only one covers the bound texture's full mip chain.
        for (key, levels) in [(6, 1), (7, 3), (8, 3)] {
            nvn.initialize(key, "Texture", texture(16, levels, 0x4000, 37))
                .unwrap();
        }
        nvn.ensure_render_target(6).unwrap();
        nvn.ensure_render_target(7).unwrap();
        for indexed in [false, true] {
            if indexed {
                nvn.refresh_target_index();
            }
            assert_eq!(nvn.resolve_color_target(8, 2).unwrap(), (7, 2));
        }

        // When only a shallow image covers the storage, extending it
        // keeps pixels already written at level 0.
        nvn.initialize(9, "Texture", texture(16, 1, 0x8000, 37))
            .unwrap();
        nvn.initialize(10, "Texture", texture(16, 3, 0x8000, 37))
            .unwrap();
        let red = [1.0f32, 0.0, 0.0, 1.0];
        let blue = [0.0f32, 0.0, 1.0, 1.0];
        nvn.gpu_clear_color(&core, (9, 0, 0), red.map(f32::to_bits), None)
            .unwrap();
        nvn.refresh_target_index();
        nvn.gpu_clear_color(&core, (10, 2, 0), blue.map(f32::to_bits), None)
            .unwrap();
        assert_eq!(nvn.resolve_target(9), Some((10, 0)));
        let renderer = &mut nvn.gpu.as_mut().unwrap().renderer;
        assert_eq!(renderer.read_target_level(10, 0).unwrap(), [red; 256]);
        assert_eq!(renderer.read_target_level(10, 2).unwrap(), [blue; 16]);
    }
}

#[cfg(test)]
mod pool_tests {
    use super::{
        Backend, GuestFileIndex, Nvn, Sampler, SharedRange, TextureBacking, TextureSource, vk,
    };
    use crate::nvn::{Args, Command, Guest, GuestMemory};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    #[test]
    fn virtual_gpu_pages_use_the_physical_pool_and_follow_remapping() {
        let Ok(Some(backend)) = Backend::create(None) else {
            return;
        };
        if !backend.renderer.gpu.buffer_device_address {
            return;
        }
        let mut nvn = Nvn {
            gpu: Some(backend),
            gpu_checked: true,
            ..Nvn::default()
        };
        use echora_guest::memory::{Memory, READ, STATE_NORMAL, WRITE};
        use echora_shader_abi::{PAGE_SHIFT, PAGE_SIZE};
        let storage = 0x10_0000;
        let request = 0x12_0000;
        let mut memory = Memory::new(1 << 39);
        memory
            .map(
                storage,
                4 * PAGE_SIZE,
                READ | WRITE,
                STATE_NORMAL,
                "physical pool",
            )
            .unwrap();
        memory
            .map(
                request,
                PAGE_SIZE,
                READ | WRITE,
                STATE_NORMAL,
                "mapping requests",
            )
            .unwrap();
        let mut cpu = echora_guest::cpu::Cpu::default();
        let mut guest = Guest {
            cpu: &mut cpu,
            memory: GuestMemory::Checked(&mut memory),
            counter: Some(0),
        };
        let call = |nvn: &mut Nvn, guest: &mut Guest, name: &str, args: &[u64]| {
            guest.cpu.regs[..args.len()].copy_from_slice(args);
            assert!(nvn.call(name, guest).unwrap());
            guest.cpu.regs[0]
        };
        let physical = 0x30_0000;
        let virtual_pool = physical + 0x100;
        let builder = physical + 0x200;
        let device = physical + 0x300;
        let device_builder = physical + 0x400;
        call(
            &mut nvn,
            &mut guest,
            "nvnDeviceBuilderSetDefaults",
            &[device_builder],
        );
        call(
            &mut nvn,
            &mut guest,
            "nvnDeviceInitialize",
            &[device, device_builder],
        );
        for (object, base, size, flags) in [
            (physical, storage, 4 * PAGE_SIZE, 8),
            (virtual_pool, 0, 8 * PAGE_SIZE, 0x100),
        ] {
            call(
                &mut nvn,
                &mut guest,
                "nvnMemoryPoolBuilderSetDefaults",
                &[builder],
            );
            call(
                &mut nvn,
                &mut guest,
                "nvnMemoryPoolBuilderSetDevice",
                &[builder, device],
            );
            call(
                &mut nvn,
                &mut guest,
                "nvnMemoryPoolBuilderSetFlags",
                &[builder, flags],
            );
            call(
                &mut nvn,
                &mut guest,
                "nvnMemoryPoolBuilderSetStorage",
                &[builder, base, size],
            );
            assert_eq!(
                call(
                    &mut nvn,
                    &mut guest,
                    "nvnMemoryPoolInitialize",
                    &[object, builder]
                ),
                1
            );
        }
        let gpu = call(
            &mut nvn,
            &mut guest,
            "nvnMemoryPoolGetBufferAddress",
            &[virtual_pool],
        );
        nvn.page_table(&mut guest).unwrap();
        let page = |nvn: &mut Nvn, guest: &mut Guest, address: u64| {
            let table = nvn.page_table(guest).unwrap();
            let index = (address >> PAGE_SHIFT).checked_sub(u64::from(table.first_page));
            index
                .and_then(|index| {
                    nvn.gpu
                        .as_ref()?
                        .renderer
                        .page_table_entries()
                        .get(index as usize)
                })
                .copied()
                .unwrap_or(0)
        };
        let physical_gpu = call(
            &mut nvn,
            &mut guest,
            "nvnMemoryPoolGetBufferAddress",
            &[physical],
        );
        // The CPU can address the physical pool, while the GPU can only
        // access it through a virtual alias.
        assert_eq!(nvn.gpu_to_guest(physical_gpu, 4), Ok(storage));
        assert_eq!(page(&mut nvn, &mut guest, physical_gpu), 0);
        let address = gpu + 3 * PAGE_SIZE;
        let texture = builder + 0x800;
        let scalar = |value| Args {
            x: [value, 0, 0, 0],
            ..Default::default()
        };
        nvn.initialize(
            texture,
            "Texture",
            BTreeMap::from([
                (
                    "Storage".into(),
                    Args {
                        x: [virtual_pool, 3 * PAGE_SIZE, 0, 0],
                        ..Default::default()
                    },
                ),
                ("Width".into(), scalar(1)),
                ("Height".into(), scalar(1)),
                ("Stride".into(), scalar(4)),
                ("Format".into(), scalar(37)),
            ]),
        )
        .unwrap();
        for offset in [PAGE_SIZE, 2 * PAGE_SIZE] {
            guest
                .memory
                .write(storage + offset, &[(offset / PAGE_SIZE) as u8, 0, 0, 255])
                .unwrap();
        }
        for (backing, offset) in [(physical, 2 * PAGE_SIZE), (physical, PAGE_SIZE), (0, 0)] {
            for (index, word) in [backing, offset, 3 * PAGE_SIZE, PAGE_SIZE]
                .into_iter()
                .enumerate()
            {
                guest
                    .memory
                    .write(request + index as u64 * 8, &word.to_le_bytes())
                    .unwrap();
            }
            assert_eq!(
                call(
                    &mut nvn,
                    &mut guest,
                    "nvnMemoryPoolMapVirtual",
                    &[virtual_pool, 1, request]
                ),
                1
            );
            let actual = page(&mut nvn, &mut guest, address);
            let expected = if backing == 0 {
                0
            } else {
                nvn.gpu
                    .as_ref()
                    .unwrap()
                    .renderer
                    .resident_address(storage)
                    .unwrap()
                    + offset
            };
            assert_eq!(actual, expected);
            assert_eq!(page(&mut nvn, &mut guest, address - PAGE_SIZE), 0);
            if backing == 0 {
                assert!(nvn.guest_texture(&guest, texture).is_err());
            } else {
                let TextureSource::Upload { texels, .. } =
                    nvn.guest_texture(&guest, texture).unwrap()
                else {
                    panic!("expected a texture uploaded from the mapped backing");
                };
                assert_eq!(texels.as_slice(), &[(offset / PAGE_SIZE) as u8, 0, 0, 255]);
            }
        }

        let update = |nvn: &mut Nvn, guest: &mut Guest, requests: &[[u64; 4]]| {
            for (index, words) in requests.iter().enumerate() {
                for (field, value) in words.iter().enumerate() {
                    guest
                        .memory
                        .write(
                            request + (index * 0x28 + field * 8) as u64,
                            &value.to_le_bytes(),
                        )
                        .unwrap();
                }
            }
            guest.cpu.regs[..3].copy_from_slice(&[virtual_pool, requests.len() as u64, request]);
            nvn.call("nvnMemoryPoolMapVirtual", guest)
        };
        // The adjacent pages keep their backing when the middle page unmaps
        // or is replaced, with the original offset retained on the right.
        update(
            &mut nvn,
            &mut guest,
            &[[physical, 0, 2 * PAGE_SIZE, 3 * PAGE_SIZE]],
        )
        .unwrap();
        let neighbor = |nvn: &mut Nvn, guest: &mut Guest| {
            [
                page(nvn, guest, address - PAGE_SIZE),
                page(nvn, guest, address + PAGE_SIZE),
            ]
        };
        let sides = neighbor(&mut nvn, &mut guest);
        assert_ne!(sides[0], 0);
        assert_eq!(sides[1], sides[0] + 2 * PAGE_SIZE);
        update(&mut nvn, &mut guest, &[[0, 0, 3 * PAGE_SIZE, PAGE_SIZE]]).unwrap();
        assert_eq!(page(&mut nvn, &mut guest, address), 0);
        assert_eq!(neighbor(&mut nvn, &mut guest), sides);
        update(
            &mut nvn,
            &mut guest,
            &[[physical, 3 * PAGE_SIZE, 3 * PAGE_SIZE, PAGE_SIZE]],
        )
        .unwrap();
        assert_eq!(
            page(&mut nvn, &mut guest, address),
            sides[0] + 3 * PAGE_SIZE
        );
        assert_eq!(neighbor(&mut nvn, &mut guest), sides);

        // A request list with an invalid range leaves every mapping intact.
        assert!(
            update(
                &mut nvn,
                &mut guest,
                &[
                    [0, 0, 3 * PAGE_SIZE, PAGE_SIZE],
                    [0, 0, 8 * PAGE_SIZE, PAGE_SIZE]
                ],
            )
            .is_err()
        );
        assert_eq!(
            page(&mut nvn, &mut guest, address),
            sides[0] + 3 * PAGE_SIZE
        );
        assert_eq!(neighbor(&mut nvn, &mut guest), sides);
    }

    #[test]
    fn mesh_indirect_accepts_a_count_at_the_end_of_a_pool() {
        let Ok(Some(backend)) = Backend::create(None) else {
            return;
        };
        let pool = 1;
        let base = 0x10_0000;
        let address = 0x20_0000;
        let size = 4096;
        let mut nvn = Nvn {
            gpu: Some(backend),
            gpu_checked: true,
            ..Nvn::default()
        };
        nvn.initialize(
            pool,
            "MemoryPool",
            BTreeMap::from([(
                "Storage".into(),
                Args {
                    x: [base, size, 0, 0],
                    ..Args::default()
                },
            )]),
        )
        .unwrap();
        nvn.objects.get_mut(&pool).unwrap().state.insert(
            "GpuAddress".into(),
            Args {
                x: [address, 0, 0, 0],
                ..Args::default()
            },
        );
        use echora_guest::memory::{Memory, READ, STATE_NORMAL, WRITE};
        let mut memory = Memory::new(1 << 39);
        memory
            .map(base, size, READ | WRITE, STATE_NORMAL, "mesh count")
            .unwrap();
        memory.write(base + size - 4, &73u32.to_le_bytes()).unwrap();
        let mut cpu = echora_guest::cpu::Cpu::default();
        let mut guest = Guest {
            cpu: &mut cpu,
            memory: GuestMemory::Checked(&mut memory),
            counter: Some(0),
        };
        let mut command = Command {
            name: "nvnCommandBufferDrawMeshTasksIndirect",
            x: [0; 8],
            v: [0; 8],
            captured: Vec::new(),
            states: Default::default(),
        };
        command.x[1] = address + size - 4;
        let groups = nvn.mesh_groups(&mut guest, &command).unwrap();
        assert!(matches!(groups, super::MeshGroups::Indirect { .. }));

        // An earlier dispatch wrote a temporary storage binding at the
        // same guest address. The next draw must pick up its new count.
        let temporary = 0x30_0000;
        let count = 31u32.to_le_bytes();
        let backend = nvn.gpu.as_mut().unwrap();
        backend.renderer.resident_buffer(temporary, 16).unwrap();
        backend
            .renderer
            .upload_resident(temporary, 0, &count)
            .unwrap();
        let token = backend.renderer.read_resident(temporary, 0, 4).unwrap();
        backend
            .pending_storage
            .push((token, base + size - 4..base + size, 0, 0, command.x[1]));
        let groups = nvn.mesh_groups(&mut guest, &command).unwrap();
        assert!(matches!(groups, super::MeshGroups::Indirect { .. }));
        assert_eq!(guest.memory.read_u32(base + size - 4).unwrap(), 31);
        let renderer = &mut nvn.gpu.as_mut().unwrap().renderer;
        let token = renderer.read_resident(base, size - 4, 4).unwrap();
        renderer.flush().unwrap();
        assert_eq!(renderer.take_reads(&[token])[&token], count);
    }

    #[test]
    fn observable_marker_leaves_later_writes_in_the_same_submission_pending() {
        let Ok(Some(mut backend)) = Backend::create(None) else {
            return;
        };
        let base = 0x10_0000;
        backend.renderer.resident_buffer(base, 4096).unwrap();
        backend.renderer.upload_resident(base, 0, &[1; 16]).unwrap();
        let serial = backend.renderer.recorded_serial();
        backend.record_page_table_write(serial, None);
        let mut nvn = Nvn {
            gpu: Some(backend),
            gpu_checked: true,
            ..Nvn::default()
        };
        let marker = nvn.gpu_observable(serial).unwrap();
        let backend = nvn.gpu.as_mut().unwrap();
        backend
            .renderer
            .upload_resident(base, 16, &[2; 16])
            .unwrap();
        assert_eq!(backend.renderer.recorded_serial(), serial);
        backend.record_page_table_write(serial, None);
        backend.renderer.submit_through(serial).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !nvn.gpu_marker_done(marker) {
            assert!(
                std::time::Instant::now() < deadline,
                "GPU marker did not complete"
            );
            std::thread::yield_now();
        }

        let mut memory = echora_guest::memory::Memory::new(1 << 39);
        let mut cpu = echora_guest::cpu::Cpu::default();
        let mut guest = Guest {
            cpu: &mut cpu,
            memory: GuestMemory::Checked(&mut memory),
            counter: Some(0),
        };
        assert_eq!(nvn.gpu_read_wait_serial().unwrap(), None);
        nvn.resolve_gpu_written(&mut guest, 0..0).unwrap();
        let backend = nvn.gpu.as_mut().unwrap();
        assert!(backend.renderer.completed_serial() < serial);
        assert!(backend.page_table_writes);
        backend.wait_for_table_writes(None).unwrap();
        assert!(backend.renderer.completed_serial() >= serial);
        assert!(!backend.page_table_writes);
    }

    #[test]
    fn shared_write_wait_uses_the_newest_overlapping_submission() {
        let Ok(Some(mut backend)) = Backend::create(None) else {
            return;
        };
        let range = |file, start, end| SharedRange { file, start, end };
        backend.record_page_table_write(1, None);
        backend.record_page_table_write(2, Some(vec![range(1, 0, 16)].into()));
        backend.record_page_table_write(3, Some(vec![range(2, 0, 16)].into()));
        backend.record_page_table_write(
            4,
            Some(
                vec![
                    range(1, 16, 32),
                    range(2, 24, 32),
                    range(1, 8, 16),
                    range(1, 12, 24),
                ]
                .into(),
            ),
        );

        assert_eq!(backend.shared_write_serial(range(1, 0, 8)), 2);
        assert_eq!(backend.shared_write_serial(range(1, 7, 9)), 4);
        assert_eq!(backend.shared_write_serial(range(2, 7, 9)), 3);
        assert_eq!(backend.shared_write_serial(range(2, 21, 24)), 1);
        assert_eq!(backend.shared_write_serial(range(2, 24, 25)), 4);
        assert_eq!(backend.shared_write_serial(range(1, 32, 40)), 1);
        backend.discard_shared_writes_through(2);
        assert_eq!(backend.shared_writes.front().unwrap().0, 3);
        assert_eq!(backend.shared_write_serial(range(1, 7, 9)), 4);

        backend.record_page_table_write(5, None);
        assert!(backend.shared_writes.is_empty());
        assert_eq!(backend.shared_write_serial(range(1, 0, 16)), 5);
    }

    #[test]
    fn projected_writes_follow_only_additions_to_their_own_table() {
        let Ok(Some(mut backend)) = Backend::create(None) else {
            return;
        };
        let (buffer, _) = backend.renderer.resident_buffer(7, 8192).unwrap();
        let Some(address) = backend.renderer.buffer_address(buffer) else {
            return;
        };
        let page = echora_shader_abi::PAGE_SIZE;
        let initial = 10 * page;
        let entries = |addresses: &[u64]| {
            let mapping = super::writer_ranges::project_table(
                initial,
                page,
                addresses,
                vec![super::writer_ranges::ImportedSpan {
                    file: 1,
                    device: address..address + 2 * page,
                    offset: 0,
                }],
            );
            super::PageBuild {
                first: 10,
                entries: addresses.to_vec(),
                pools: Default::default(),
                shared_mapping: std::cell::OnceCell::from(mapping),
            }
        };
        let first = backend
            .set_page_table((0, 0), entries(&[address, 0, 0]), None)
            .unwrap();
        let touches = |start| SharedRange {
            file: 1,
            start,
            end: start + 8,
        };
        backend.record_page_table_write(
            1,
            Some(super::BoundedWrites {
                shared: vec![touches(0)],
                virtual_ranges: std::iter::once(initial..initial + page + 8).collect(),
            }),
        );
        let unchanged = backend
            .set_page_table((0, 1), entries(&[address, 0, 0]), None)
            .unwrap();
        assert_eq!(first, unchanged);
        let extended = backend
            .set_page_table((0, 2), entries(&[address, address + page, 0]), None)
            .unwrap();
        assert_eq!(extended, first);
        assert_eq!(backend.shared_write_serial(touches(0)), 1);
        assert_eq!(backend.shared_write_serial(touches(page)), 1);
        assert_eq!(backend.shared_write_serial(touches(page + 8)), 0);
        assert_eq!(backend.unbounded_write_serial, 0);

        // A replacement gives newer work a different buffer. The previous
        // serial's virtual ranges must not be projected through it.
        let replaced = backend
            .set_page_table((0, 3), entries(&[address + page, 0, 0]), None)
            .unwrap();
        assert_ne!(replaced.address, first.address);
        backend
            .set_page_table((0, 4), entries(&[address + page, address, 0]), None)
            .unwrap();
        assert_eq!(backend.shared_write_serial(touches(page + 8)), 0);
        assert_eq!(backend.shared_write_serial(touches(page)), 1);
    }

    #[test]
    fn empty_draws_do_not_resolve_program_or_buffer_bindings() {
        let mut nvn = Nvn::default();
        let mut memory = echora_guest::memory::Memory::new(1 << 39);
        let mut cpu = echora_guest::cpu::Cpu::default();
        let mut guest = Guest {
            cpu: &mut cpu,
            memory: GuestMemory::Checked(&mut memory),
            counter: Some(0),
        };
        for (name, count_arg) in [
            ("nvnCommandBufferDrawArrays", 3),
            ("nvnCommandBufferDrawElements", 3),
            ("nvnCommandBufferDrawElementsBaseVertex", 3),
            ("nvnCommandBufferDrawArraysInstanced", 5),
            ("nvnCommandBufferDrawElementsInstanced", 7),
            ("nvnCommandBufferDrawMeshTasks", 2),
        ] {
            let mut command = Command {
                name,
                x: [1; 8],
                v: [0; 8],
                captured: Vec::new(),
                states: Default::default(),
            };
            command.x[count_arg] = 0;
            nvn.gpu_draw(&mut guest, &command).unwrap();
            command.x[count_arg] = 1;
            assert!(nvn.gpu_draw(&mut guest, &command).is_err());
        }
    }

    #[test]
    fn target_index_updates_when_backing_storage_changes() {
        let Ok(Some(backend)) = Backend::create(None) else {
            return;
        };
        let mut nvn = Nvn {
            gpu: Some(backend),
            gpu_checked: true,
            ..Nvn::default()
        };
        let args = |x| Args {
            x,
            ..Args::default()
        };
        nvn.initialize(
            0x1000,
            "MemoryPool",
            BTreeMap::from([("Storage".to_owned(), args([0x10_0000, 0x10000, 0, 0]))]),
        )
        .unwrap();
        let props = BTreeMap::from([
            ("Storage".to_owned(), args([0x1000, 0x2000, 0, 0])),
            ("Width".to_owned(), args([16, 0, 0, 0])),
            ("Height".to_owned(), args([16, 0, 0, 0])),
            ("Format".to_owned(), args([37, 0, 0, 0])),
        ]);
        nvn.initialize(0x2000, "Texture", props.clone()).unwrap();
        nvn.ensure_render_target(0x2000).unwrap();
        nvn.refresh_target_index();
        assert_eq!(
            nvn.gpu.as_ref().unwrap().target_index.storage[&0x10_2000],
            [0x2000]
        );
        nvn.initialize(0x3000, "Texture", props.clone()).unwrap();
        assert!(nvn.target_index_current());

        nvn.gpu
            .as_mut()
            .unwrap()
            .renderer
            .move_target(0x2000, 0x3000)
            .unwrap();
        nvn.evict_bindings();
        assert!(!nvn.target_index_current());
        nvn.refresh_target_index();
        assert_eq!(
            nvn.gpu.as_ref().unwrap().target_index.storage[&0x10_2000],
            [0x3000]
        );

        let mut props = props;
        props.get_mut("Storage").unwrap().x[1] = 0x3000;
        nvn.objects.get_mut(&0x3000).unwrap().builder = props;
        nvn.object_changed(0x3000, "Texture");
        assert!(!nvn.target_index_current());
        nvn.refresh_target_index();
        let index = &nvn.gpu.as_ref().unwrap().target_index;
        assert!(!index.storage.contains_key(&0x10_2000));
        assert_eq!(index.storage[&0x10_3000], [0x3000]);

        nvn.objects.remove(&0x1000);
        nvn.object_changed(0x1000, "MemoryPool");
        assert!(!nvn.target_index_current());
        nvn.refresh_target_index();
        assert!(nvn.gpu.as_ref().unwrap().target_index.storage.is_empty());
        nvn.initialize(
            0x1000,
            "MemoryPool",
            BTreeMap::from([("Storage".to_owned(), args([0x20_0000, 0x10000, 0, 0]))]),
        )
        .unwrap();
        nvn.refresh_target_index();
        assert_eq!(
            nvn.gpu.as_ref().unwrap().target_index.storage[&0x20_3000],
            [0x3000]
        );
    }

    #[test]
    fn texture_storage_index_preserves_flushes_and_named_invalidations() {
        let Ok(Some(mut backend)) = Backend::create(None) else {
            return;
        };
        backend.textures.insert(
            1,
            (
                0,
                TextureSource::Target(1),
                TextureBacking::linear(0x1000, 0x80),
            ),
        );
        backend.textures.insert(
            2,
            (
                0,
                TextureSource::Target(2),
                TextureBacking::linear(0x1080, 0x80),
            ),
        );
        let mut nvn = Nvn {
            gpu: Some(backend),
            gpu_checked: true,
            ..Nvn::default()
        };
        nvn.gpu_invalidate(0x2000..0x2010, None);
        assert!(nvn.gpu.as_ref().unwrap().dirty.is_empty());

        nvn.gpu_invalidate(0x1040..0x10c0, None);
        let dirtied = &nvn.gpu.as_ref().unwrap().dirty;
        assert_eq!(dirtied[&1].len(), 1);
        assert_eq!(dirtied[&1][0], 0x40..0x80);
        assert_eq!(dirtied[&2].len(), 1);
        assert_eq!(dirtied[&2][0], 0..0x40);

        nvn.gpu_invalidate(0..0, Some(1));
        let backend = nvn.gpu.as_mut().unwrap();
        assert!(!backend.textures.contains_key(&1));
        assert!(!backend.cached_textures_overlap(&(0x1000..0x1080)));
        assert!(backend.cached_textures_overlap(&(0x1080..0x1081)));
    }

    #[test]
    fn shared_span_index_reuses_lookup_order_and_checks_the_whole_range() {
        let index = GuestFileIndex::new(
            1,
            [
                (0x1000, 0x400, 3, 0x8000),
                (0x1100, 0x80, 4, 0xa000),
                (0x1300, 0x400, 5, 0xc000),
            ]
            .into_iter(),
        );
        assert_eq!(
            index.range(0x1120, 8),
            Some(SharedRange {
                file: 3,
                start: 0x8120,
                end: 0x8128
            })
        );
        assert_eq!(
            index.range(0x1380, 0x100),
            Some(SharedRange {
                file: 5,
                start: 0xc080,
                end: 0xc180
            })
        );
        assert_eq!(index.range(0x13ff, 0x400), None);
        assert_eq!(index.range(u64::MAX, 8), None);
    }

    #[test]
    fn finalizing_a_pool_drops_all_of_its_shared_pieces() {
        let Ok(Some(mut backend)) = Backend::create(None) else {
            return;
        };
        backend.shared_pieces.insert(
            0x1000,
            vec![(0, 0x100, 3, 0x8000), (0x200, 0x100, 3, 0xa000)],
        );
        backend.record_pool_share(0x1000);
        backend.shared_pools.insert(0x2000, (0x100, 3, 0x8000));
        backend.record_pool_share(0x2000);
        backend
            .shared_pieces
            .insert(0x3000, vec![(0, 0x40, 3, 0xa040)]);
        backend.record_pool_share(0x3000);
        backend.shared_pools.insert(0x4000, (0x100, 3, 0xc000));
        backend.record_pool_share(0x4000);
        let mut nvn = Nvn {
            gpu: Some(backend),
            gpu_checked: true,
            ..Nvn::default()
        };
        assert_eq!(nvn.shared_guest_range(0x1020, 8).unwrap().start, 0x8020);
        assert_eq!(nvn.shared_guest_range(0x1200, 8).unwrap().start, 0xa000);
        let previous = nvn.pool_share_signature(0x1000, 4);
        let whole_alias = nvn.pool_share_signature(0x2000, 4);
        let split_alias = nvn.pool_share_signature(0x3000, 4);
        let unrelated = nvn.pool_share_signature(0x4000, 4);
        nvn.gpu_forget_pool(0x1000);
        assert_eq!(nvn.shared_guest_range(0x1020, 8), None);
        assert_eq!(nvn.shared_guest_range(0x1200, 8), None);
        assert_ne!(nvn.pool_share_signature(0x2000, 4), whole_alias);
        assert_ne!(nvn.pool_share_signature(0x3000, 4), split_alias);
        assert_eq!(nvn.pool_share_signature(0x4000, 4), unrelated);
        assert_eq!(nvn.shared_guest_range(0x2000, 8).unwrap().start, 0x8000);
        assert_eq!(nvn.shared_guest_range(0x3000, 8).unwrap().start, 0xa040);

        // A replacement with the same span count can point elsewhere.
        let backend = nvn.gpu.as_mut().unwrap();
        backend.shared_pieces.insert(
            0x1000,
            vec![(0, 0x100, 3, 0x9000), (0x200, 0x100, 3, 0xb000)],
        );
        backend.record_pool_share(0x1000);
        assert_ne!(nvn.pool_share_signature(0x1000, 4), previous);
        assert_eq!(nvn.shared_guest_range(0x1020, 8).unwrap().start, 0x9020);
        assert_eq!(nvn.shared_guest_range(0x1200, 8).unwrap().start, 0xb000);
    }

    #[test]
    fn texture_sample_retains_all_chunks_and_their_readability() {
        let size = 256 * 64;
        let empty = super::sample_chunks(7, 0, size, |_, _| true);
        for index in 0..256 {
            let changed = super::sample_chunks(7, 0, size, |address, bytes| {
                if address == index * 64 {
                    bytes[(index % 64) as usize] = 1;
                }
                true
            });
            assert_ne!(changed, empty);
        }
        let missing = super::sample_chunks(7, 0, size, |address, _| address != 64);
        assert_ne!(missing, empty);
        let failed = super::sample_chunks(7, 0, size, |address, bytes| {
            if address == 64 {
                bytes.fill(255);
                false
            } else {
                true
            }
        });
        assert_eq!(failed, missing);
    }

    #[test]
    fn unaligned_storage_reads_keep_bytes_next_to_unmapped_pages() {
        use crate::nvn::{Guest, GuestMemory};
        use echora_guest::{
            cpu::Cpu,
            memory::{Memory, READ, STATE_NORMAL, WRITE},
        };

        let mut memory = Memory::new(1 << 39);
        for base in [0x10_0000, 0x10_2000] {
            memory
                .map(base, 0x1000, READ | WRITE, STATE_NORMAL, "storage")
                .unwrap();
        }
        memory.write(0x10_0ffe, &[1, 2]).unwrap();
        memory.write(0x10_2000, &[3, 4]).unwrap();
        let mut cpu = Cpu::default();
        let guest = Guest {
            cpu: &mut cpu,
            memory: GuestMemory::Checked(&mut memory),
            counter: None,
        };
        let mut data = [0xff; 0x1004];
        super::read_storage_into(&guest, 0x10_0ffe, &mut data);
        assert_eq!(data[..2], [1, 2]);
        assert!(data[2..0x1002].iter().all(|&byte| byte == 0));
        assert_eq!(data[0x1002..], [3, 4]);
    }

    #[test]
    fn small_texture_samples_cover_the_storage_without_its_neighbors() {
        use echora_guest::memory::{Memory, READ, STATE_NORMAL, WRITE};
        let mut cpu = echora_guest::cpu::Cpu::default();
        let mut memory = Memory::new(1 << 39);
        let base = 0x10000;
        memory
            .map(base, 1 << 12, READ | WRITE, STATE_NORMAL, "test texture")
            .unwrap();
        let mut core = Guest {
            cpu: &mut cpu,
            memory: GuestMemory::Checked(&mut memory),
            counter: None,
        };
        let snapshot = |core: &Guest| super::texture_sample(core, base, 4, 37, [1, 1, 1]);
        let initial = snapshot(&core);
        core.memory.write(base + 4, &[1]).unwrap();
        assert_eq!(snapshot(&core), initial);
        core.memory.write(base + 3, &[1]).unwrap();
        assert_ne!(snapshot(&core), initial);
    }

    #[test]
    fn sampler_registration_captures_the_builder_properties() {
        let mut nvn = Nvn::default();
        let mut cpu = echora_guest::cpu::Cpu::default();
        let mut memory = echora_guest::memory::Memory::new(1 << 39);
        let mut core = Guest {
            cpu: &mut cpu,
            memory: GuestMemory::Checked(&mut memory),
            counter: Some(0),
        };
        let pool = 0x1000;
        let builder = 0x2000;
        let device = 0x3000;
        let sampler = 0x4000;
        nvn.initialize(device, "Device", BTreeMap::new()).unwrap();
        nvn.initialize(
            pool,
            "SamplerPool",
            BTreeMap::from([(
                "Storage".into(),
                Args {
                    x: [0, 0, 16, 0],
                    v: [0; 2],
                },
            )]),
        )
        .unwrap();
        nvn.render.state.insert(
            "SetSamplerPool",
            Command {
                name: "nvnCommandBufferSetSamplerPool",
                x: [0, pool, 0, 0, 0, 0, 0, 0],
                v: [0; 8],
                captured: Vec::new(),
                states: Default::default(),
            },
        );
        let mut call = |nvn: &mut Nvn, name: &str, args: [u64; 4], v0: f32| {
            core.cpu.regs[..4].copy_from_slice(&args);
            core.cpu.vectors[0] = u128::from(v0.to_bits());
            assert!(nvn.call(name, &mut core).unwrap());
        };
        call(
            &mut nvn,
            "nvnSamplerBuilderSetDefaults",
            [builder, 0, 0, 0],
            0.0,
        );
        call(
            &mut nvn,
            "nvnSamplerBuilderSetDevice",
            [builder, device, 0, 0],
            0.0,
        );
        call(
            &mut nvn,
            "nvnSamplerBuilderSetMinMagFilter",
            [builder, 0, 0, 0],
            0.0,
        );
        call(
            &mut nvn,
            "nvnSamplerBuilderSetMaxAnisotropy",
            [builder, 0, 0, 0],
            2.5,
        );
        call(
            &mut nvn,
            "nvnSamplerInitialize",
            [sampler, builder, 0, 0],
            0.0,
        );
        call(
            &mut nvn,
            "nvnSamplerPoolRegisterSamplerBuilder",
            [pool, 2, builder, 0],
            0.0,
        );
        assert_eq!(nvn.handle_sampler(2 << 20).min_filter, vk::Filter::NEAREST);
        assert_eq!(nvn.handle_sampler(2 << 20).max_anisotropy, 2.5);
        call(
            &mut nvn,
            "nvnSamplerBuilderSetMinMagFilter",
            [builder, 1, 1, 0],
            0.0,
        );
        call(
            &mut nvn,
            "nvnSamplerBuilderSetMaxAnisotropy",
            [builder, 0, 0, 0],
            8.0,
        );
        assert_eq!(nvn.handle_sampler(2 << 20).min_filter, vk::Filter::NEAREST);
        assert_eq!(nvn.handle_sampler(2 << 20).max_anisotropy, 2.5);
        call(
            &mut nvn,
            "nvnSamplerPoolRegisterSamplerBuilder",
            [pool, 2, builder, 0],
            0.0,
        );
        assert_eq!(nvn.handle_sampler(2 << 20).min_filter, vk::Filter::LINEAR);
        assert_eq!(nvn.handle_sampler(2 << 20).max_anisotropy, 8.0);
        call(
            &mut nvn,
            "nvnSamplerPoolRegisterSampler",
            [pool, 3, sampler, 0],
            0.0,
        );
        assert_eq!(nvn.handle_sampler(3 << 20).min_filter, vk::Filter::NEAREST);
        assert_eq!(nvn.handle_sampler(3 << 20).max_anisotropy, 2.5);
        call(&mut nvn, "nvnSamplerFinalize", [sampler, 0, 0, 0], 0.0);
        assert!(!nvn.holds_object(sampler));
        assert_eq!(nvn.handle_sampler(3 << 20).min_filter, vk::Filter::NEAREST);
        assert_eq!(nvn.handle_sampler(3 << 20).max_anisotropy, 2.5);
        call(
            &mut nvn,
            "nvnSamplerInitialize",
            [sampler, builder, 0, 0],
            0.0,
        );
        assert_eq!(nvn.handle_sampler(3 << 20).min_filter, vk::Filter::NEAREST);
        assert_eq!(nvn.handle_sampler(3 << 20).max_anisotropy, 2.5);
    }

    #[test]
    fn sampler_anisotropy_preserves_requests_and_normalizes_invalid_values() {
        let defaults = Nvn::sampler_from_properties(&BTreeMap::new());
        assert_eq!(defaults.max_anisotropy, 1.0);
        for (requested, expected) in [
            (4.0_f32, 4.0),
            (16.0, 16.0),
            (0.0, 1.0),
            (-2.0, 1.0),
            (f32::NAN, 1.0),
            (f32::INFINITY, 1.0),
        ] {
            let properties = BTreeMap::from([(
                "MaxAnisotropy".into(),
                Args {
                    x: [0; 4],
                    v: [u64::from(requested.to_bits()), 0],
                },
            )]);
            let sampler = Nvn::sampler_from_properties(&properties);
            assert_eq!(sampler.max_anisotropy, expected);
            assert_eq!(sampler == defaults, expected == 1.0);
        }
    }

    #[test]
    fn invalidating_a_mapped_buffer_reads_its_requested_cached_range() {
        use crate::nvn::{RegisteredBuffer, RegisteredPool, write_shared};
        use echora_guest::{
            cpu::Cpu,
            memory::{Memory, READ, STATE_NORMAL, WRITE},
        };

        let Ok(Some(mut backend)) = Backend::create(None) else {
            return;
        };
        let base = 0x10_0000;
        let size = 4096;
        backend.renderer.resident_buffer(base, size).unwrap();
        backend
            .renderer
            .upload_resident(base, 16, &[0x42; 24])
            .unwrap();
        backend.resident.insert(
            base,
            super::ResidentPool {
                base,
                size,
                uncached: false,
                host_visible: false,
                guest_dirty: Vec::new(),
                gpu_dirty: std::iter::once(base + 16..base + 40).collect(),
            },
        );
        let mut nvn = Nvn {
            gpu: Some(backend),
            gpu_checked: true,
            ..Nvn::default()
        };
        {
            let mut registry = write_shared(&nvn.registry);
            registry.pools.insert(
                1,
                RegisteredPool {
                    storage: base,
                    size,
                    gpu: 0,
                    flags: None,
                    buffers: 1,
                },
            );
            registry.buffers.insert(
                2,
                RegisteredBuffer {
                    pool: 1,
                    offset: 12,
                    size: 32,
                    state: BTreeMap::new(),
                },
            );
        }

        let mut memory = Memory::new(1 << 39);
        memory
            .map(base, size, READ | WRITE, STATE_NORMAL, "buffer invalidate")
            .unwrap();
        let mut cpu = Cpu::default();
        cpu.regs[..3].copy_from_slice(&[2, 8, 4]);
        let mut guest = Guest {
            cpu: &mut cpu,
            memory: GuestMemory::Checked(&mut memory),
            counter: None,
        };
        assert!(
            nvn.call("nvnBufferInvalidateMappedRange", &mut guest)
                .unwrap()
        );
        assert_eq!(guest.memory.read_u32(base + 16).unwrap(), 0);
        assert_eq!(guest.memory.read_u32(base + 20).unwrap(), 0x4242_4242);
        assert_eq!(guest.memory.read_u32(base + 24).unwrap(), 0);
    }

    #[test]
    fn draw_input_waits_for_overlapping_page_table_writes() {
        use echora_guest::{cpu::Cpu, memory::Memory};

        let Ok(Some(mut backend)) = Backend::create(None) else {
            return;
        };
        let guest = 0x10_0000;
        backend
            .shared_pieces
            .insert(guest, vec![(0, 8, 1, 0), (8, 8, 2, 48)]);
        backend.record_pool_share(guest);
        backend.renderer.resident_buffer(guest, 4096).unwrap();
        backend
            .renderer
            .upload_resident(guest, 0, &[0x42; 16])
            .unwrap();
        let serial = backend.renderer.recorded_serial();
        backend.record_page_table_write(
            serial,
            Some(
                vec![super::SharedRange {
                    file: 2,
                    start: 49,
                    end: 50,
                }]
                .into(),
            ),
        );

        let mut nvn = Nvn {
            gpu: Some(backend),
            gpu_checked: true,
            ..Nvn::default()
        };
        let mut memory = Memory::new(1 << 39);
        let mut cpu = Cpu::default();
        let mut core = Guest {
            cpu: &mut cpu,
            memory: GuestMemory::Checked(&mut memory),
            counter: Some(0),
        };
        nvn.resolve_draw_input(&mut core, guest, 9).unwrap();
        assert!(nvn.gpu.as_ref().unwrap().renderer.completed_serial() < serial);
        nvn.resolve_draw_input(&mut core, guest + 6, 5).unwrap();
        assert!(nvn.gpu.as_ref().unwrap().renderer.completed_serial() >= serial);
    }

    #[test]
    fn cpu_render_condition_waits_for_imported_gpu_writes() {
        use echora_guest::{
            cpu::Cpu,
            memory::{Memory, READ, STATE_NORMAL, WRITE},
        };

        let Ok(Some(mut backend)) = Backend::create(None) else {
            return;
        };
        let base = 0x10_0000;
        let size = 4096;
        backend.shared_pieces.insert(base, vec![(0, size, 1, 0)]);
        backend.record_pool_share(base);
        backend.renderer.resident_buffer(base, size).unwrap();
        backend.renderer.upload_resident(base, 16, &[0; 8]).unwrap();
        let serial = backend.renderer.recorded_serial();
        backend.record_page_table_write(
            serial,
            Some(
                vec![super::SharedRange {
                    file: 1,
                    start: 16,
                    end: 24,
                }]
                .into(),
            ),
        );

        let mut nvn = Nvn {
            gpu: Some(backend),
            gpu_checked: true,
            ..Nvn::default()
        };
        nvn.initialize(
            1,
            "MemoryPool",
            BTreeMap::from([(
                "Storage".into(),
                Args {
                    x: [base, size, 0, 0],
                    ..Args::default()
                },
            )]),
        )
        .unwrap();
        nvn.objects.get_mut(&1).unwrap().state.insert(
            "GpuAddress".into(),
            Args {
                x: [base, 0, 0, 0],
                ..Args::default()
            },
        );
        nvn.render.render_enable = Some((0, base));

        let mut memory = Memory::new(1 << 39);
        memory
            .map(base, size, READ | WRITE, STATE_NORMAL, "render condition")
            .unwrap();
        let mut cpu = Cpu::default();
        let mut core = Guest {
            cpu: &mut cpu,
            memory: GuestMemory::Checked(&mut memory),
            counter: None,
        };
        assert!(nvn.gpu.as_ref().unwrap().renderer.completed_serial() < serial);
        assert!(nvn.render_enabled(&mut core).unwrap());
        assert!(nvn.gpu.as_ref().unwrap().renderer.completed_serial() >= serial);
    }

    #[test]
    fn draw_input_reads_only_overlapping_resident_bytes() {
        let Ok(Some(mut backend)) = Backend::create(None) else {
            return;
        };
        let base = 0x10_0000;
        backend.renderer.resident_buffer(base, 4096).unwrap();
        backend
            .renderer
            .upload_resident(base, 8, &[0x42; 16])
            .unwrap();
        backend.resident.insert(
            base,
            super::ResidentPool {
                base,
                size: 4096,
                uncached: true,
                host_visible: true,
                guest_dirty: Vec::new(),
                gpu_dirty: std::iter::once(base + 8..base + 24).collect(),
            },
        );
        let mut nvn = Nvn {
            gpu: Some(backend),
            gpu_checked: true,
            ..Nvn::default()
        };
        use echora_guest::memory::{Memory, READ, STATE_NORMAL, WRITE};
        let mut memory = Memory::new(1 << 39);
        memory
            .map(base, 4096, READ | WRITE, STATE_NORMAL, "resident test")
            .unwrap();
        let mut cpu = echora_guest::cpu::Cpu::default();
        let mut core = Guest {
            cpu: &mut cpu,
            memory: GuestMemory::Checked(&mut memory),
            counter: Some(0),
        };
        assert!(nvn.gpu.as_ref().unwrap().pending_storage.is_empty());
        nvn.resolve_draw_input(&mut core, base, 8).unwrap();
        nvn.resolve_draw_input(&mut core, base + 12, 0).unwrap();
        assert_eq!(nvn.gpu.as_ref().unwrap().renderer.flushes, 0);
        nvn.resolve_draw_input(&mut core, base + 12, 4).unwrap();
        assert_eq!(core.memory.read_u32(base + 12).unwrap(), 0x4242_4242);
        assert_eq!(core.memory.read_u32(base + 16).unwrap(), 0);
        let backend = nvn.gpu.as_ref().unwrap();
        assert!(!backend.resident_writes_pending(&(base + 12..base + 16)));
        assert!(backend.resident_writes_pending(&(base + 16..base + 24)));
    }

    #[test]
    fn deferred_texture_bind_reads_the_gpu_written_handle() {
        use echora_guest::{
            cpu::Cpu,
            memory::{Memory, READ, STATE_NORMAL, WRITE},
        };

        let Ok(Some(mut backend)) = Backend::create(None) else {
            return;
        };
        let base = 0x10_0000;
        let size = 4096;
        let handle = 0x1000_00ab_u64;
        backend.renderer.resident_buffer(base, size).unwrap();
        backend
            .renderer
            .upload_resident(base, 0, &[handle.to_le_bytes(); 2].concat())
            .unwrap();
        backend.resident.insert(
            base,
            super::ResidentPool {
                base,
                size,
                uncached: true,
                host_visible: true,
                guest_dirty: Vec::new(),
                gpu_dirty: std::iter::once(base..base + 16).collect(),
            },
        );
        let mut nvn = Nvn {
            gpu: Some(backend),
            gpu_checked: true,
            ..Nvn::default()
        };
        let mut memory = Memory::new(1 << 39);
        memory
            .map(
                base,
                size,
                READ | WRITE,
                STATE_NORMAL,
                "deferred texture binds",
            )
            .unwrap();
        let mut cpu = Cpu::default();
        let mut core = Guest {
            cpu: &mut cpu,
            memory: GuestMemory::Checked(&mut memory),
            counter: None,
        };
        nvn.execute(
            &Command {
                name: "nvnCommandBufferBindTexturesDeferred",
                x: [0, 0, 1, 1, base, 0, 0, 0],
                v: [0; 8],
                captured: Vec::new(),
                states: Default::default(),
            },
            &mut core,
        )
        .unwrap();

        assert_eq!(nvn.render.bound_textures.get(0, 1), Some(&handle));
        assert_eq!(core.memory.read_u64(base + 8).unwrap(), 0);
        assert!(
            nvn.gpu
                .as_ref()
                .unwrap()
                .resident_writes_pending(&(base + 8..base + 16))
        );
    }

    #[test]
    fn failed_resident_writeback_keeps_unwritten_ranges_pending() {
        let Ok(Some(mut backend)) = Backend::create(None) else {
            return;
        };
        let base = 0x10_0000;
        let page = 4096;
        backend.renderer.resident_buffer(base, 2 * page).unwrap();
        backend
            .renderer
            .upload_resident(base, 8, &[0x42; 4])
            .unwrap();
        backend
            .renderer
            .upload_resident(base, page + 8, &[0x7b; 4])
            .unwrap();
        let first = base + 8..base + 12;
        let second = base + page + 8..base + page + 12;
        backend.resident.insert(
            base,
            super::ResidentPool {
                base,
                size: 2 * page,
                uncached: true,
                host_visible: true,
                guest_dirty: Vec::new(),
                gpu_dirty: vec![first.clone(), second.clone()],
            },
        );
        let mut nvn = Nvn {
            gpu: Some(backend),
            gpu_checked: true,
            ..Nvn::default()
        };
        use echora_guest::memory::{Memory, READ, STATE_NORMAL, WRITE};
        let mut memory = Memory::new(1 << 39);
        memory
            .map(
                base,
                page,
                READ | WRITE,
                STATE_NORMAL,
                "writeback first page",
            )
            .unwrap();
        let mut cpu = echora_guest::cpu::Cpu::default();
        {
            let mut core = Guest {
                cpu: &mut cpu,
                memory: GuestMemory::Checked(&mut memory),
                counter: Some(0),
            };
            assert!(nvn.resolve_resident(&mut core, None).is_err());
            assert_eq!(core.memory.read_u32(first.start).unwrap(), 0x4242_4242);
        }
        assert!(nvn.gpu.as_ref().unwrap().resident_writes_pending(&second));
        memory
            .map(
                base + page,
                page,
                READ | WRITE,
                STATE_NORMAL,
                "writeback second page",
            )
            .unwrap();
        let mut core = Guest {
            cpu: &mut cpu,
            memory: GuestMemory::Checked(&mut memory),
            counter: Some(0),
        };
        nvn.resolve_resident(&mut core, None).unwrap();
        assert_eq!(core.memory.read_u32(second.start).unwrap(), 0x7b7b_7b7b);
        let backend = nvn.gpu.as_ref().unwrap();
        assert!(!backend.resident_writes_pending(&first));
        assert!(!backend.resident_writes_pending(&second));
    }

    #[test]
    fn texture_copy_sees_an_earlier_page_table_write() {
        let Ok(Some(mut backend)) = Backend::create(None) else {
            return;
        };
        let pool = 1;
        let texture = 2;
        let base = 0x10_0000;
        let address = 0x20_0000;
        let page = 4096;
        backend.renderer.resident_buffer(base, page).unwrap();
        backend
            .renderer
            .upload_resident(base, 0, &[0x7b; 16])
            .unwrap();
        backend.resident.insert(
            base,
            super::ResidentPool {
                base,
                size: page,
                uncached: true,
                host_visible: true,
                guest_dirty: Vec::new(),
                gpu_dirty: Vec::new(),
            },
        );
        backend.record_page_table_write(backend.renderer.recorded_serial(), None);
        let mut nvn = Nvn {
            gpu: Some(backend),
            gpu_checked: true,
            ..Nvn::default()
        };
        let args = |x| Args {
            x,
            ..Args::default()
        };
        nvn.initialize(
            pool,
            "MemoryPool",
            BTreeMap::from([("Storage".into(), args([base, 2 * page, 0, 0]))]),
        )
        .unwrap();
        nvn.objects
            .get_mut(&pool)
            .unwrap()
            .state
            .insert("GpuAddress".into(), args([address, 0, 0, 0]));
        nvn.initialize(
            texture,
            "Texture",
            BTreeMap::from([
                ("Storage".into(), args([pool, page, 0, 0])),
                ("Width".into(), args([2, 0, 0, 0])),
                ("Height".into(), args([2, 0, 0, 0])),
                ("Format".into(), args([37, 0, 0, 0])),
            ]),
        )
        .unwrap();
        use echora_guest::memory::{Memory, READ, STATE_NORMAL, WRITE};
        let mut memory = Memory::new(1 << 39);
        memory
            .map(base, 2 * page, READ | WRITE, STATE_NORMAL, "texture copy")
            .unwrap();
        let mut cpu = echora_guest::cpu::Cpu::default();
        let mut core = Guest {
            cpu: &mut cpu,
            memory: GuestMemory::Checked(&mut memory),
            counter: Some(0),
        };
        nvn.resolve_storage(&mut core, Some(base..base + 16))
            .unwrap();
        assert_eq!(core.memory.read_u32(base).unwrap(), 0);
        nvn.execute_command(
            &Command {
                name: "nvnCommandBufferCopyBufferToTexture",
                x: [0, address, texture, 0, 0, 0, 0, 0],
                v: [0; 8],
                captured: vec![0, 0, 0, 2, 2, 1, 0, 0, 0],
                states: Default::default(),
            },
            &mut core,
        )
        .unwrap();
        assert_eq!(core.memory.read_u32(base + page).unwrap(), 0x7b7b_7b7b);
    }

    #[test]
    fn texture_copy_reads_the_rendered_mip_and_preserves_buffer_padding() {
        let Ok(Some(mut backend)) = Backend::create(None) else {
            return;
        };
        backend
            .renderer
            .set_resolution_scale(echora_gpu::renderer::Scale::new(3, 2).unwrap());
        let pool = 1;
        let texture = 2;
        let base = 0x10_0000;
        let address = 0x20_0000;
        let page = 4096;
        let mut nvn = Nvn {
            gpu: Some(backend),
            gpu_checked: true,
            ..Nvn::default()
        };
        let args = |x| Args {
            x,
            ..Args::default()
        };
        nvn.initialize(
            pool,
            "MemoryPool",
            BTreeMap::from([("Storage".into(), args([base, 8 * page, 0, 0]))]),
        )
        .unwrap();
        nvn.objects
            .get_mut(&pool)
            .unwrap()
            .state
            .insert("GpuAddress".into(), args([address, 0, 0, 0]));
        nvn.initialize(
            texture,
            "Texture",
            BTreeMap::from([
                ("Storage".into(), args([pool, page, 0, 0])),
                ("Width".into(), args([8, 0, 0, 0])),
                ("Height".into(), args([8, 0, 0, 0])),
                ("Depth".into(), args([2, 0, 0, 0])),
                ("Target".into(), args([4, 0, 0, 0])),
                ("Levels".into(), args([2, 0, 0, 0])),
                ("Format".into(), args([37, 0, 0, 0])),
            ]),
        )
        .unwrap();
        let blue = [0.0f32, 0.0, 1.0, 1.0].map(f32::to_bits);
        let red = [1.0f32, 0.0, 0.0, 1.0].map(f32::to_bits);
        use echora_guest::memory::{Memory, READ, STATE_NORMAL, WRITE};
        let mut memory = Memory::new(1 << 39);
        memory
            .map(base, 8 * page, READ | WRITE, STATE_NORMAL, "texture copy")
            .unwrap();
        memory.write(base, &[0x4b; 64]).unwrap();
        let mut cpu = echora_guest::cpu::Cpu::default();
        let mut core = Guest {
            cpu: &mut cpu,
            memory: GuestMemory::Checked(&mut memory),
            counter: Some(0),
        };
        nvn.gpu_clear_color(&core, (texture, 1, 0), blue, None)
            .unwrap();
        nvn.gpu_clear_color(&core, (texture, 1, 1), red, None)
            .unwrap();
        nvn.execute_command(
            &Command {
                name: "nvnCommandBufferCopyTextureToBuffer",
                x: [0, texture, 0, 0, address, 0, 0, 0],
                v: [0; 8],
                captured: vec![1, 1, 0, 2, 2, 1, 1, 1, 1, 12, 0],
                states: Default::default(),
            },
            &mut core,
        )
        .unwrap();
        let mut output = [0; 24];
        core.memory.read(base, &mut output).unwrap();
        let red_row = [255, 0, 0, 255, 255, 0, 0, 255];
        assert_eq!(&output[..8], &red_row);
        assert_eq!(&output[8..12], &[0x4b; 4]);
        assert_eq!(&output[12..20], &red_row);
        assert_eq!(&output[20..], &[0x4b; 4]);
    }

    #[test]
    fn pool_list_is_reused_only_while_bindings_stay_current() {
        let Ok(Some(backend)) = Backend::create(None) else {
            return;
        };
        let mut nvn = Nvn {
            gpu: Some(backend),
            gpu_checked: true,
            ..Nvn::default()
        };
        let mut cpu = echora_guest::cpu::Cpu::default();
        let mut memory = echora_guest::memory::Memory::new(1 << 39);
        let mut core = Guest {
            cpu: &mut cpu,
            memory: GuestMemory::Checked(&mut memory),
            counter: Some(0),
        };
        let handle = 19;
        nvn.render.bound_textures.insert((0, 0), handle);
        nvn.binding_memo.1.insert(
            handle,
            (
                TextureSource::Target(7),
                Sampler::default(),
                Some(TextureBacking::linear(4096, 4096)),
                0,
                0,
                7,
            ),
        );

        let original = nvn.pool_entries(&mut core, 0, &[]).unwrap();
        assert_eq!(original.len(), 1);
        let backend = nvn.gpu.as_mut().unwrap();
        backend.in_place_writes.push(8192..8196);
        backend.gpu_writes += 1;
        let unchanged = nvn.pool_entries(&mut core, 0, &[]).unwrap();
        assert!(Arc::ptr_eq(&original, &unchanged));

        let dynamic = 20;
        nvn.binding_memo.1.insert(
            dynamic,
            (TextureSource::Target(8), Sampler::default(), None, 1, 0, 8),
        );
        let including_dynamic = nvn.pool_entries(&mut core, 0, &[dynamic]).unwrap();
        assert_eq!(including_dynamic.len(), 2);
        assert_eq!(including_dynamic[1].texture, dynamic as u32);
        let same = nvn.pool_entries(&mut core, 0, &[dynamic]).unwrap();
        assert!(Arc::ptr_eq(&including_dynamic, &same));
        let static_only = nvn.pool_entries(&mut core, 0, &[]).unwrap();
        assert_eq!(static_only.len(), 1);

        let backend = nvn.gpu.as_mut().unwrap();
        backend.in_place_writes.push(4096..4100);
        backend.gpu_writes += 1;
        let changed = nvn.pool_entries(&mut core, 0, &[]).unwrap();
        // There is no registered handle to rebuild this fixture's stale
        // binding. It must be resolved anew and disappear from the list.
        assert!(changed.is_empty());
    }
}
