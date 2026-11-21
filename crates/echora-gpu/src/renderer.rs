//! Executes NVN work on the host GPU.
//!
//! Guest render targets are Vulkan images keyed by the guest texture's
//! address. Work is recorded into one command buffer and submitted when the
//! NVN layer needs the results (a present, a readback, a fence), then waited
//! for: simple and ordered, not fast.

use std::collections::HashMap;

use crate::fast_map::{FastMap, FastSet};

use ash::vk;

use crate::compute::StorageImage;
use crate::draw::ImageShape;
use crate::vulkan::Gpu;

pub struct Target {
    pub image: vk::Image,
    pub view: vk::ImageView,
    /// The view shaders sample: `view`, or for depth-stencil formats a
    /// depth-only view (a sampled view may name one aspect only).
    pub sampled_view: vk::ImageView,
    /// Dimensionality and mip levels the image was created with.
    pub shape: crate::draw::ImageShape,
    pub levels: u32,
    memory: vk::DeviceMemory,
    pub format: vk::Format,
    /// Uses supported by views with this image's own format.
    view_usage: vk::ImageUsageFlags,
    /// The guest's extent; the image is `scale` times as wide and high.
    pub width: u32,
    pub height: u32,
    /// Resolution scale (none, or the renderer's scale for render targets).
    pub scale: Scale,
    /// Layout the image is in after the commands recorded so far.
    layout: vk::ImageLayout,
}

/// Image description available even when the texels are in a host archive.
#[derive(Clone, Copy)]
pub struct TargetInfo {
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    pub shape: ImageShape,
    pub levels: u32,
    pub scale: Scale,
    /// The image has been initialized or used by recorded GPU work.
    pub initialized: bool,
}

impl From<&Target> for TargetInfo {
    fn from(target: &Target) -> Self {
        Self {
            width: target.width,
            height: target.height,
            format: target.format,
            shape: target.shape,
            levels: target.levels,
            scale: target.scale,
            initialized: target.layout != vk::ImageLayout::UNDEFINED,
        }
    }
}

struct ArchivedTarget {
    info: TargetInfo,
    decoded_bytes: usize,
    compressed: std::sync::Arc<[u8]>,
}

struct PendingArchive {
    read: Option<u64>,
    bytes: u64,
    image: Option<vk::Image>,
}

struct ArchiveCompressor {
    jobs: std::sync::mpsc::Sender<(u64, Vec<u8>)>,
    results: std::sync::mpsc::Receiver<(u64, Option<Vec<u8>>)>,
    thread: std::thread::JoinHandle<()>,
}

impl ArchiveCompressor {
    fn start() -> Option<Self> {
        let (jobs, pending) = std::sync::mpsc::channel::<(u64, Vec<u8>)>();
        let (finished, results) = std::sync::mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("echora-target-archive".into())
            .spawn(move || {
                let mut compressor = zstd::bulk::Compressor::new(1).ok();
                for (key, raw) in pending {
                    let compressed = compressor.as_mut().and_then(|c| c.compress(&raw).ok());
                    if finished.send((key, compressed)).is_err() {
                        break;
                    }
                }
            })
            .ok()?;
        Some(Self {
            jobs,
            results,
            thread,
        })
    }

    fn stop(self) {
        drop(self.results);
        drop(self.jobs);
        let _ = self.thread.join();
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArchiveRequest {
    Queued,
    Ineligible,
    Deferred,
}

impl Target {
    /// The image's extent at mip `level`.
    pub fn extent(&self, level: u32) -> (u32, u32) {
        (
            (self.scale.apply(self.width) >> level).max(1),
            (self.scale.apply(self.height) >> level).max(1),
        )
    }

    /// `rect` (x, y, width, height in guest texels) in image texels.
    fn scaled(&self, rect: [i32; 4]) -> [i32; 4] {
        self.scale.rect(rect)
    }
}

/// A resolution scale, numerator over denominator in lowest terms: image
/// texels per guest texel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scale {
    numerator: u32,
    denominator: u32,
}

impl Scale {
    pub const ONE: Self = Self {
        numerator: 1,
        denominator: 1,
    };

    /// `numerator / denominator`, reduced; None outside 1/2 to 4.
    pub fn new(numerator: u32, denominator: u32) -> Option<Self> {
        let n = u64::from(numerator);
        let d = u64::from(denominator);
        if d == 0 || n * 2 < d || n > d * 4 {
            return None;
        }
        let (mut a, mut b) = (numerator, denominator);
        while b != 0 {
            (a, b) = (b, a % b);
        }
        Some(Self {
            numerator: numerator / a,
            denominator: denominator / a,
        })
    }

    /// The scale chosen by ECHORA_RESOLUTION=WIDTHxHEIGHT (its height over
    /// the console's 1080 rows: 2560x1440 is 4/3) or ECHORA_RESOLUTION_SCALE
    /// (an integer, or N/D); none by default or when out of range.
    pub fn from_env() -> Self {
        Self::env_override().unwrap_or(Self::ONE)
    }

    /// The scale the environment asks for, if it asks.
    pub fn env_override() -> Option<Self> {
        let resolution = std::env::var("ECHORA_RESOLUTION").ok().and_then(|value| {
            let (_, height) = value.trim().split_once(['x', 'X'])?;
            Self::new(height.trim().parse().ok()?, 1080)
        });
        let factor = || {
            let value = std::env::var("ECHORA_RESOLUTION_SCALE").ok()?;
            let value = value.trim().trim_end_matches('x');
            match value.split_once('/') {
                Some((numerator, denominator)) => {
                    Self::new(numerator.parse().ok()?, denominator.parse().ok()?)
                }
                None => Self::new(value.parse().ok()?, 1),
            }
        };
        resolution.or_else(factor)
    }

    pub fn is_one(self) -> bool {
        self == Self::ONE
    }

    /// (numerator, denominator), as the shader translator takes them.
    pub fn terms(self) -> (u32, u32) {
        (self.numerator, self.denominator)
    }

    pub fn factor(self) -> f32 {
        self.numerator as f32 / self.denominator as f32
    }

    /// A guest extent in image texels.
    pub fn apply(self, value: u32) -> u32 {
        let scaled = u64::from(value) * u64::from(self.numerator) / u64::from(self.denominator);
        (scaled.min(u64::from(u32::MAX)) as u32).max(u32::from(value > 0))
    }

    /// A guest coordinate in image texels (rounded down, as the shaders'
    /// rescaled texel fetches are).
    fn coordinate(self, value: i32) -> i32 {
        (i64::from(value) * i64::from(self.numerator))
            .div_euclid(i64::from(self.denominator))
            .clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
    }

    /// A rect (x, y, width, height) in image texels: its edges scaled, so
    /// neighbouring rects still meet.
    pub fn rect(self, [x, y, width, height]: [i32; 4]) -> [i32; 4] {
        let (left, top) = (self.coordinate(x), self.coordinate(y));
        let right = self.coordinate(x.saturating_add(width));
        let bottom = self.coordinate(y.saturating_add(height));
        [
            left,
            top,
            right.saturating_sub(left),
            bottom.saturating_sub(top),
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct TargetDescription {
    width: u32,
    height: u32,
    shape: ImageShape,
    format: vk::Format,
    levels: u32,
}

impl From<&Target> for TargetDescription {
    fn from(target: &Target) -> Self {
        Self {
            width: target.width,
            height: target.height,
            shape: target.shape,
            format: target.format,
            levels: target.levels,
        }
    }
}

/// Allocations at least this large are logged with their caller.
const LARGE_ALLOCATION: u64 = 512 << 20;

/// ECHORA_ALLOC_LOG: running totals of device allocations by caller.
fn log_allocation(caller: &'static std::panic::Location<'static>, size: u64) {
    use std::collections::HashMap;
    use std::sync::Mutex;
    static TOTALS: Mutex<(u64, Option<HashMap<String, (u64, u64)>>)> = Mutex::new((0, None));
    let mut guard = TOTALS.lock().unwrap_or_else(|poison| poison.into_inner());
    let (sum, totals) = &mut *guard;
    let totals = totals.get_or_insert_with(HashMap::new);
    let entry = totals.entry(caller.to_string()).or_default();
    entry.0 += 1;
    entry.1 += size;
    let before = *sum >> 30;
    *sum += size;
    if *sum >> 30 > before {
        let mut ranked: Vec<_> = totals.iter().collect();
        ranked.sort_by_key(|(_, (_, bytes))| std::cmp::Reverse(*bytes));
        let top: Vec<String> = ranked
            .iter()
            .take(6)
            .map(|(site, (count, bytes))| format!("{site} {} MiB/{count}", bytes >> 20))
            .collect();
        eprintln!("gpu alloc: {} MiB total; {}", *sum >> 20, top.join(", "));
    }
}
/// The largest span `import_file` shares: udmabuf's default size limit.
const MAX_SHARED_SPAN: u64 = 64 << 20;
#[cfg(target_os = "linux")]
const SPAN_PACK_INTERVAL: u64 = 2048;
#[cfg(target_os = "linux")]
const SPAN_PACK_MIN_IMPORTS: usize = 128;
#[cfg(target_os = "linux")]
const SPAN_PACK_MAX_BYTES: u64 = 512 << 20;
/// Select the most stable imports that fit the packing byte budget. The
/// flags are returned in file offset order for the packer's range lookups.
#[cfg(target_os = "linux")]
fn select_stable_spans(sizes: impl ExactSizeIterator<Item = Option<u64>>) -> Vec<bool> {
    let mut selected = vec![false; sizes.len()];
    let mut candidates: Vec<_> = sizes
        .enumerate()
        .filter_map(|(index, size)| size.map(|size| (size, index)))
        .collect();
    candidates.sort_unstable();
    let mut remaining = SPAN_PACK_MAX_BYTES;
    for (size, index) in candidates {
        if size > remaining {
            break;
        }
        selected[index] = true;
        remaining -= size;
    }
    selected
}

#[cfg(target_os = "linux")]
const UDMABUF_FLAGS_CLOEXEC: u32 = 1;
/// Guest memory pinned by shared spans before the least recently used are
/// dropped.
const MAX_SHARED_BYTES: u64 = 1 << 30;
/// The largest texture upload: 1.5 GiB, above the largest the game makes
/// (a 1365x1365, 90-layer ASTC array with 11 levels, 852 MiB as RGBA8).
const MAX_UPLOAD_BYTES: u64 = 1536 << 20;

/// The thread that submits recorded batches, in order, so the recording
/// thread does not wait in vkQueueSubmit (about 1 ms a batch on RADV with
/// the ~1,700 imported buffers the game's forest holds). The command
/// buffer is ended before it is handed over: its pool is shared with the
/// slots still recorded. ECHORA_SUBMIT_THREAD=0 submits on the recording
/// thread.
struct Submitter {
    sender: Option<std::sync::mpsc::Sender<Submission>>,
    progress: std::sync::Arc<SubmissionProgress>,
    /// Nanoseconds the thread spent in vkQueueSubmit, for the statistics.
    submit_ns: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// Batches handed to the thread, in submission order.
    handed: u64,
    thread: Option<std::thread::JoinHandle<()>>,
}

struct Submission {
    commands: vk::CommandBuffer,
    fence: vk::Fence,
    signal: Option<vk::Semaphore>,
}

/// A batch's fence and command buffer belong to vkQueueSubmit until that call
/// returns, even if the fence has already signaled.
#[derive(Default)]
struct SubmissionProgress {
    returned: std::sync::atomic::AtomicU64,
    failure: std::sync::Mutex<Option<String>>,
    changed: std::sync::Condvar,
}

impl SubmissionProgress {
    /// The submitter stopped, leaving the failed batch and any later ones
    /// without a fence signal.
    const FAILED: u64 = u64::MAX;

    fn returned(&self, handed: u64) -> bool {
        let returned = self.returned.load(std::sync::atomic::Ordering::Acquire);
        returned != Self::FAILED && returned >= handed
    }

    fn failure(&self) -> Option<String> {
        if self.returned.load(std::sync::atomic::Ordering::Acquire) != Self::FAILED {
            return None;
        }
        self.failure
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    fn finish(&self, result: Result<(), String>) -> bool {
        use std::sync::atomic::Ordering::Release;

        let succeeded = result.is_ok();
        {
            // Use the same mutex as the wait predicate so a notification
            // cannot be lost between the check and parking.
            let mut failure = self
                .failure
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            match result {
                Ok(()) => {
                    self.returned.fetch_add(1, Release);
                }
                Err(error) => {
                    *failure = Some(error);
                    self.returned.store(Self::FAILED, Release);
                }
            }
        }
        self.changed.notify_all();
        succeeded
    }

    fn wait(&self, handed: u64) -> Result<(), String> {
        if self.returned(handed) {
            return Ok(());
        }
        let failure = self
            .failure
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let failure = self
            .changed
            .wait_while(failure, |failure| {
                failure.is_none() && !self.returned(handed)
            })
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        failure.clone().map_or(Ok(()), Err)
    }
}

impl Submitter {
    fn new(gpu: &Gpu, queue_lock: std::sync::Arc<std::sync::Mutex<()>>) -> Result<Self, String> {
        let (sender, receiver) = std::sync::mpsc::channel::<Submission>();
        let progress = std::sync::Arc::new(SubmissionProgress::default());
        let submit_ns = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let (device, queue) = (gpu.device.clone(), gpu.queue);
        let thread = std::thread::Builder::new()
            .name("echora-submit".into())
            .spawn({
                let (progress, submit_ns) = (progress.clone(), submit_ns.clone());
                move || {
                    for Submission {
                        commands,
                        fence,
                        signal,
                    } in receiver
                    {
                        let buffers = [commands];
                        let submit = [vk::SubmitInfo::default()
                            .command_buffers(&buffers)
                            .signal_semaphores(signal.as_slice())];
                        let start = std::time::Instant::now();
                        let held = queue_lock
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        // SAFETY: the buffer was ended and is not touched
                        // again until its fence signals; the queue is held.
                        let result = unsafe { device.queue_submit(queue, &submit, fence) };
                        drop(held);
                        submit_ns.fetch_add(
                            start.elapsed().as_nanos() as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                        if !progress.finish(result.map_err(vk_error("vkQueueSubmit"))) {
                            break;
                        }
                    }
                }
            })
            .map_err(|error| format!("the submit thread: {error}"))?;
        Ok(Self {
            sender: Some(sender),
            progress,
            submit_ns,
            handed: 0,
            thread: Some(thread),
        })
    }

    /// Join the thread after it has drained the queue or hit a failure.
    fn stop(&mut self) {
        self.sender = None;
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Which backing file owns the offsets tracked by the page-table imports.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum AddressedFile {
    #[default]
    Unset,
    Single(i32),
    Multiple,
}

/// Discontiguous byte ranges packed back-to-back in the same imported buffer.
struct PackedSpans {
    ranges: Vec<(u64, u64, u64)>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    address: u64,
}

impl PackedSpans {
    fn offset(&self, start: u64, length: u64) -> Option<u64> {
        let after = self.ranges.partition_point(|&(known, ..)| known <= start);
        self.ranges[..after]
            .iter()
            .rev()
            .take_while(|&&(known, ..)| known + MAX_SHARED_SPAN >= start)
            .find_map(|&(known, size, offset)| {
                (start + length <= known + size).then_some(offset + (start - known))
            })
    }
}

struct GeneratedTextureUpload<'a> {
    size: usize,
    fill: &'a mut dyn FnMut(u64, &mut [u8]) -> Result<(), String>,
}

pub struct Renderer {
    /// The submission slot being recorded: its command buffer, fence,
    /// ring, descriptor pool and what they hold until it completes (the
    /// per-draw recording helpers work on it).
    slot: Slot,
    /// Graphics state remains valid across render passes in one batch.
    pub(crate) graphics: crate::draw::GraphicsState,
    pub(crate) shader_clear_programs: [Option<crate::draw::Program>; 4],
    pub gpu: Gpu,
    /// Held while the queue is used (submissions, presents).
    pub(crate) queue_lock: std::sync::Arc<std::sync::Mutex<()>>,
    submitter: Option<Submitter>,
    /// The texture pool's set (`crate::pool`), once a program reads it.
    pub(crate) texture_pool: Option<crate::pool::TexturePool>,
    pub(crate) spare_texture_pool: Option<crate::pool::TexturePool>,
    pub(crate) tried_spare_texture_pool: bool,
    /// The texture pool entries last bound, the target changes then, and
    /// the targets they read (`bind_pool`).
    pub(crate) pool_cache: Option<(
        std::sync::Arc<[crate::pool::PoolEntry]>,
        u64,
        Vec<vk::Image>,
    )>,
    /// Each uploaded pool entry's resolved descriptors, by (texture,
    /// sampler) index: reused while the entry is the same (`bind_pool`).
    pub(crate) pool_entry_memo: FastMap<(u32, u32), crate::pool::ResolvedEntry>,
    /// Counts for the runtime's statistics (`take_stats`).
    pub(crate) stats: FastMap<&'static str, u64>,
    /// Rendering passes ended, by the source line that ended them (for
    /// the statistics: a pass that ends between draws costs a store and
    /// load of every attachment).
    pub(crate) pass_ends: FastMap<&'static std::panic::Location<'static>, u64>,
    /// Debugging buffers shaders write through device addresses
    /// (`debug_buffer`).
    debug_buffers: Vec<HostBuffer>,
    /// ECHORA_LOOP_TRACE buffers (index, the program it counts), reported
    /// when a wait stalls or the device is lost (`report_loop_traces`).
    loop_trace_buffers: Vec<(usize, String)>,
    recording: bool,
    /// No draws, dispatches or copies follow the last full barrier in the
    /// current command buffer.
    full_barrier_recorded: bool,
    memory_types: vk::PhysicalDeviceMemoryProperties,
    targets: FastMap<u64, Target>,
    archived_targets: FastMap<u64, ArchivedTarget>,
    archived_bytes: FastMap<u64, std::sync::Weak<[u8]>>,
    pending_archives: FastMap<u64, PendingArchive>,
    pending_archive_bytes: u64,
    archive_compressor: std::cell::OnceCell<Option<ArchiveCompressor>>,
    guest_target_keys: Vec<u64>,
    /// Guest textures uploaded for sampling, with the generation uploaded.
    uploads: FastMap<u64, (u64, Target)>,
    /// The texel bytes each upload was made from: a caller may drop its
    /// copy once the image holds them (`has_upload`), and `with_texels`
    /// reads them back from the image when needed again.
    upload_lengths: FastMap<u64, u64>,
    /// Texel buffers made from guest buffer textures (shaders read them as
    /// samplerBuffer), by upload key: the generation copied, the buffer and
    /// its view.
    texel_buffers: FastMap<u64, (u64, HostBuffer, vk::BufferView)>,
    /// Debugging (ECHORA_LAST_SUBMIT / ECHORA_SLOW_SUBMITS files): what
    /// the current submission records, to find work that stalls the GPU.
    labels: Vec<String>,
    /// Draws and dispatches recorded since the last submission.
    work_items: u32,
    samplers: FastMap<SamplerKey, vk::Sampler>,
    /// Additional views retired with their image, with their range, format,
    /// and usage used to identify them.
    extra_views: FastMap<(vk::Image, ExtraView), vk::ImageView>,
    /// Host memory imported as storage buffers (VK_EXT_external_memory_host),
    /// by page-aligned host address and length.
    imported: FastMap<(usize, u64), (vk::Buffer, vk::DeviceMemory)>,
    /// The GPU page table (`set_page_table`): its buffer and the entries
    /// written there.
    page_table: Option<PageTableBuffer>,
    /// Buffers from completed submissions, ready for a table of the same size.
    free_page_tables: Vec<RecycledPageTable>,
    /// Spans the driver refused to import (file-backed memory, on drivers
    /// that only import anonymous memory): not tried again.
    import_refused: std::collections::HashSet<(usize, u64)>,
    /// Spans of guest memory's file shared as dma-bufs (see `import_file`),
    /// by page-aligned file offset and length, oldest use first.
    shared: Vec<((u64, u64), vk::Buffer, vk::DeviceMemory)>,
    /// Shared spans the GPU page table points into (`address_span`): kept
    /// until their file range is released, never evicted while the table
    /// may use them. Start, span, buffer, memory, device address and the
    /// submission serial at import.
    addressed: Vec<(u64, u64, vk::Buffer, vk::DeviceMemory, u64, u64)>,
    /// Page-table spans another joined (`address_span`): the table may
    /// still point into them until it is built whole.
    superseded: Vec<(u64, u64, vk::Buffer, vk::DeviceMemory, u64, u64)>,
    #[cfg(target_os = "linux")]
    last_span_pack_serial: u64,
    packed: Vec<PackedSpans>,
    addressed_file: AddressedFile,
    /// A released range dropped a span that held more than it: other pools
    /// mapped through it are mapped again by a whole build.
    dropped_shared_span: bool,
    /// Bytes `shared` pins in host memory (udmabuf pins its pages).
    pub shared_bytes: u64,
    /// /dev/udmabuf, opened on first use; None after it failed to open.
    #[cfg(target_os = "linux")]
    udmabuf: Option<Option<std::fs::File>>,
    /// Spans that could not be shared: not tried again.
    shared_refused: std::collections::HashSet<(u64, u64)>,
    /// Storage buffers kept on the GPU between dispatches (see
    /// `resident_buffer`), by the caller's key: buffer, memory, size.
    resident: FastMap<u64, (vk::Buffer, vk::DeviceMemory, u64)>,
    /// How each sampled view was made (image, type, format, range), so a
    /// swizzled copy of it can be made.
    view_infos: FastMap<vk::ImageView, ViewInfo>,
    /// Targets created since the caller last took them (to initialize).
    fresh_targets: Vec<u64>,
    /// Submissions that waited for the GPU, and the microseconds waited.
    pub flushes: u64,
    completed_reads: Vec<(u64, Vec<u8>)>,
    next_read_token: u64,
    /// The other submission slots, the oldest submission first: while
    /// their work runs on the GPU, the CPU records into the current one
    /// (see `flush_async`).
    spares: Vec<Slot>,
    markers: FastMap<u64, (vk::Event, u64)>,
    free_events: Vec<vk::Event>,
    next_marker: u64,
    pub(crate) completed_serial: u64,
    /// `completed_serial`, readable by other threads (a CPU wait answered
    /// without the NVN state lock).
    completed_shared: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// ECHORA_HANG_MARKERS (debugging): the marker buffer and the labels of
    /// the work after each marker.
    hang: Option<HangMarkers>,
    /// Labels of the last submission (ECHORA_SLOW_SUBMITS reports them).
    submitted_labels: Vec<String>,
    /// Render targets created, and recreated because their shape changed.
    pub targets_created: u64,
    /// Changes to the set of targets (created, recreated, moved,
    /// removed): an index of them by key is current while this holds.
    pub target_changes: u64,
    /// The keys of the targets made, remade or moved since the last
    /// `take_changed_targets` (what `target_changes` counted).
    changed_targets: Vec<u64>,
    pub targets_recreated: u64,
    /// Scaled targets replaced by guest-resolution ones (`keep_unscaled`).
    pub descaled: u64,
    /// Render target resolution scale (`Scale::from_env`).
    scale: Scale,
    /// Descriptions kept at guest resolution by storage writes or by a
    /// pass with mixed attachment scales. A key can be reused for a
    /// different image description.
    unscaled: FastSet<(u64, TargetDescription)>,
    /// Targets marked before an image exists for the key.
    pending_unscaled: FastSet<u64>,
    /// Color clears limited by the scissor to part of the target.
    pub partial_clears: u64,
    pub flush_wait_microseconds: u64,
    /// Images rendered to or cleared since the last memory barrier.
    /// A later pass or transfer using one of them waits for those writes.
    written: FastSet<vk::Image>,
    /// Memory barriers recorded for those hazards.
    pub hazard_barriers: u64,
    /// GPU time per submission (ECHORA_GPU_TIME=1).
    timer: Option<crate::timing::GpuTimer>,
    /// The SAMPLES_PASSED counter and its occlusion queries, once used.
    pub(crate) occlusion: Option<crate::occlusion::Occlusion>,
    /// The indirect mesh draws' bounded group counts, once used
    /// (`bounded_mesh_count`).
    pub(crate) mesh_count: Option<crate::mesh_count::MeshCount>,
    /// The condition draws and dispatches are gated on (`set_condition`):
    /// wanted, and active in the command buffer being recorded.
    condition: Option<(vk::Buffer, u64, bool)>,
    /// A failure that must end the run rather than skip one draw (an upload
    /// over MAX_UPLOAD_BYTES): the runtime takes it after each draw and
    /// dispatch and stops with it.
    pub fatal: Option<String>,
    /// Where linked shader objects' driver binaries are kept between runs
    /// (see `crate::shader_cache`), if anywhere.
    pub(crate) shader_cache: Option<std::path::PathBuf>,
}

#[derive(PartialEq, Eq, Hash)]
enum ExtraView {
    SampledRange {
        view_type: vk::ImageViewType,
        base_level: u32,
        level_count: u32,
        base_layer: u32,
        layer_count: u32,
    },
    RelativeLevels {
        source: vk::ImageView,
        base: u32,
        count: u32,
    },
    Swizzled {
        source: vk::ImageView,
        format: vk::Format,
        components: [vk::ComponentSwizzle; 4],
    },
    Level(u32),
    StorageImage {
        format: vk::Format,
        view_type: vk::ImageViewType,
        level: u32,
        base_layer: u32,
        layer_count: u32,
    },
    Attachment {
        format: vk::Format,
        level: u32,
        layer: u32,
    },
}

type SamplerKey = (i32, i32, i32, [i32; 3], [u32; 3], u32, Option<i32>, i32);

/// What one submission owns until the GPU finishes it: its command
/// buffer and fence, the ring its uploads went to, its descriptor pool,
/// the reads deferred from its ring and the resources retired while it
/// was recorded. The renderer's own fields hold the slot being recorded.
struct Slot {
    command_pool: vk::CommandPool,
    commands: vk::CommandBuffer,
    fence: vk::Fence,
    ring: HostBuffer,
    ring_used: u64,
    descriptor_pool: vk::DescriptorPool,
    sets_used: u32,
    descriptors_used: [u32; 4],
    pending_reads: Vec<(u64, u64, usize)>,
    retired_uploads: Vec<Target>,
    retired_texel_buffers: Vec<(HostBuffer, vk::BufferView)>,
    retired_page_tables: Vec<RecycledPageTable>,
    retired_shared: Vec<(vk::Buffer, vk::DeviceMemory)>,
    /// Submitted and not yet waited for.
    in_flight: bool,
    /// The submit thread's batch number of this slot's last submission (0:
    /// submitted on the recording thread).
    handed: u64,
    /// Serial of its last submission.
    serial: u64,
    /// Markers its commands set (`mark`), freed once it completes.
    markers: Vec<u64>,
    /// The rendering pass left open after a draw. `commands` ends it.
    open_pass: Option<OpenPass>,
    /// The wanted condition (`Renderer::condition`) is active in this
    /// slot's command buffer, and began inside its open rendering pass.
    condition_active: bool,
    condition_inside: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct ColorAttachment {
    target: u64,
    level: u32,
    layer: u32,
    format: Option<vk::Format>,
}

struct OpenPass {
    /// Bindings of the last draw that used this pass. A target can be
    /// replaced under the same key while the pass is open.
    target_changes: u64,
    colors: Vec<Option<ColorAttachment>>,
    depth: Option<(u64, u32, u32)>,
    /// The requested area only matters when there are no attachments.
    requested_area: Option<[i32; 4]>,
    views: Vec<vk::ImageView>,
    depth_view: Option<vk::ImageView>,
    area: [i32; 4],
}

impl Slot {
    fn new(device: &ash::Device, queue_family: u32) -> Result<Self, String> {
        let mut slot = Self {
            command_pool: vk::CommandPool::null(),
            commands: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            ring: HostBuffer::empty(),
            ring_used: 0,
            descriptor_pool: vk::DescriptorPool::null(),
            sets_used: 0,
            descriptors_used: [0; 4],
            pending_reads: Vec::new(),
            retired_uploads: Vec::new(),
            retired_texel_buffers: Vec::new(),
            retired_page_tables: Vec::new(),
            retired_shared: Vec::new(),
            in_flight: false,
            handed: 0,
            serial: 0,
            markers: Vec::new(),
            open_pass: None,
            condition_active: false,
            condition_inside: false,
        };
        let created = (|| {
            let pool = vk::CommandPoolCreateInfo::default().queue_family_index(queue_family);
            // SAFETY: the device and its queue family are live.
            slot.command_pool = unsafe { device.create_command_pool(&pool, None) }
                .map_err(vk_error("vkCreateCommandPool"))?;
            let allocate = vk::CommandBufferAllocateInfo::default()
                .command_pool(slot.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            // SAFETY: the pool belongs to this slot and is unused.
            slot.commands = unsafe { device.allocate_command_buffers(&allocate) }
                .map_err(vk_error("vkAllocateCommandBuffers"))?[0];
            // SAFETY: a valid create info on the live device.
            slot.fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
                .map_err(vk_error("vkCreateFence"))?;
            slot.descriptor_pool = create_descriptor_pool(device)?;
            Ok(())
        })();
        if let Err(error) = created {
            slot.destroy_pools_and_fence(device);
            return Err(error);
        }
        Ok(slot)
    }

    /// Destroy the slot after setup failed or the last submission finished.
    fn destroy_pools_and_fence(&self, device: &ash::Device) {
        // SAFETY: any created handles belong to the device and are idle.
        // The destroy operations accept null handles from partial setup.
        unsafe {
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_fence(self.fence, None);
            device.destroy_command_pool(self.command_pool, None);
        }
    }

    /// Whether `size` bytes at `alignment` fit this slot's ring.
    fn ring_fits(&self, size: u64, alignment: u64) -> bool {
        self.ring_used.next_multiple_of(alignment) + size <= RING_SIZE
    }

    /// The offset for `size` bytes at `alignment` in this slot's ring, or
    /// none when they would not fit (the slot is then sealed and the next
    /// one takes them).
    fn ring_place(&mut self, size: u64, alignment: u64) -> Option<u64> {
        let offset = self.ring_used.next_multiple_of(alignment);
        (offset + size <= RING_SIZE).then(|| {
            self.ring_used = offset + size;
            offset
        })
    }

    /// Whether one more draw's `bytes` of ring uploads (at the 256-byte
    /// uniform alignment) and one descriptor set with `descriptors` of each
    /// pooled type fit this slot.
    fn holds(&self, bytes: u64, descriptors: [u32; 4]) -> bool {
        let capacity = [MAX_SETS * 8, MAX_SETS * 16, MAX_SETS * 4, MAX_SETS * 4];
        let pool_full = self.sets_used + 1 > MAX_SETS
            || (0..4).any(|kind| self.descriptors_used[kind] + descriptors[kind] > capacity[kind]);
        !pool_full && self.ring_used.next_multiple_of(256) + bytes <= RING_SIZE
    }

    /// Count one descriptor set with `descriptors` of each pooled type.
    fn count_set(&mut self, descriptors: [u32; 4]) {
        self.sets_used += 1;
        for (used, more) in self.descriptors_used.iter_mut().zip(descriptors) {
            *used += more;
        }
    }

    /// Read `length` ring bytes at `offset` back under `token` once this
    /// slot's work has completed.
    fn defer_read(&mut self, token: u64, offset: u64, length: usize) {
        self.pending_reads.push((token, offset, length));
    }

    /// `bytes` copied into this slot's ring at `alignment`: the buffer
    /// and offset, or none when the ring is full (the slot is sealed and
    /// the next one takes them).
    fn upload(&mut self, bytes: &[u8], alignment: u64) -> Option<(vk::Buffer, u64)> {
        let offset = self.ring_place(bytes.len() as u64, alignment)?;
        self.ring.write(offset as usize, bytes);
        Some((self.ring.buffer, offset))
    }

    /// `size` bytes of this slot's ring that `fill` writes in place, or
    /// none when the ring is full (`fill` is then not called).
    fn upload_with(
        &mut self,
        size: u64,
        alignment: u64,
        fill: impl FnOnce(&mut [u8]) -> Result<(), String>,
    ) -> Result<Option<(vk::Buffer, u64)>, String> {
        let Some(offset) = self.ring_place(size, alignment) else {
            return Ok(None);
        };
        fill(self.ring.slice_mut(offset as usize, size as usize))?;
        Ok(Some((self.ring.buffer, offset)))
    }

    /// A descriptor set of `layout` from this slot's pool, or none when
    /// the pool is full.
    fn allocate_set(
        &self,
        device: &ash::Device,
        layout: vk::DescriptorSetLayout,
    ) -> Result<Option<vk::DescriptorSet>, String> {
        let layouts = [layout];
        let info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);
        // SAFETY: the pool and layout are live.
        match unsafe { device.allocate_descriptor_sets(&info) } {
            Ok(sets) => Ok(Some(sets[0])),
            Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY | vk::Result::ERROR_FRAGMENTED_POOL) => {
                Ok(None)
            }
            Err(error) => Err(format!("vkAllocateDescriptorSets: {error}")),
        }
    }
}

/// Submission slots: the CPU records into one while the GPU runs the
/// others' batches, so a submission waits only for the batch this many
/// before it.
pub(crate) const SLOTS: usize = 6;

/// Draws and dispatches a submission holds by default. Smaller batches let
/// the GPU start a frame while the replay thread is still recording it.
/// Two forest runs on a Radeon RX 9070 XT averaged 18.44 FPS at 100 items
/// and 16.95 FPS at 200 with packed page-table imports.
const SUBMIT_ITEMS: u32 = 100;

/// Draws and dispatches one submission may hold (see `bound_submission`).
const MAX_WORK_ITEMS: u32 = 1000;

/// Ring size: everything recorded between flushes must fit.
const RING_SIZE: u64 = 64 << 20;
/// Descriptor sets per flush.
const MAX_SETS: u32 = 4096;

pub(crate) fn descriptor_buffer_range(length: usize) -> u64 {
    (length as u64).max(16).next_multiple_of(16)
}

/// A descriptor pool for one submission's sets.
fn create_descriptor_pool(device: &ash::Device) -> Result<vk::DescriptorPool, String> {
    let sizes = [
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: MAX_SETS * 8,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
            descriptor_count: MAX_SETS * 8,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: MAX_SETS * 16,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: MAX_SETS * 4,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_IMAGE,
            descriptor_count: MAX_SETS * 4,
        },
        // Counted with the combined image samplers (`reserve`).
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
            descriptor_count: MAX_SETS * 16,
        },
    ];
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(MAX_SETS)
        .pool_sizes(&sizes);
    // SAFETY: valid create info; destroyed in Drop.
    let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None) }
        .map_err(vk_error("vkCreateDescriptorPool"))?;
    Ok(descriptor_pool)
}

pub(crate) fn vk_error(context: &'static str) -> impl Fn(vk::Result) -> String {
    move |error| format!("{context}: {error}")
}

pub fn is_depth(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::D16_UNORM
            | vk::Format::D32_SFLOAT
            | vk::Format::D24_UNORM_S8_UINT
            | vk::Format::D32_SFLOAT_S8_UINT
    )
}

pub fn has_stencil(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::D24_UNORM_S8_UINT | vk::Format::D32_SFLOAT_S8_UINT
    )
}

fn aspect(format: vk::Format) -> vk::ImageAspectFlags {
    if has_stencil(format) {
        vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
    } else if is_depth(format) {
        vk::ImageAspectFlags::DEPTH
    } else {
        vk::ImageAspectFlags::COLOR
    }
}

/// Buffer/image transfers name one aspect for each plane. For a combined
/// depth-stencil upload, the four-byte depth texels precede the stencil bytes.
fn transfer_aspects(flags: vk::ImageAspectFlags) -> impl Iterator<Item = vk::ImageAspectFlags> {
    let stencil = vk::ImageAspectFlags::STENCIL;
    [flags & !stencil, flags & stencil]
        .into_iter()
        .filter(|flags| !flags.is_empty())
}

fn upload_copies(
    region: &crate::draw::UploadRegion,
    format: vk::Format,
    offset: u64,
) -> impl Iterator<Item = vk::BufferImageCopy> {
    transfer_aspects(aspect(format)).map(move |aspect_mask| {
        let mut buffer_offset = offset + region.offset;
        if aspect_mask == vk::ImageAspectFlags::STENCIL {
            buffer_offset +=
                4 * u64::from(region.width) * u64::from(region.height) * u64::from(region.depth);
        }
        vk::BufferImageCopy {
            buffer_offset,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask,
                mip_level: region.level,
                base_array_layer: region.layer,
                layer_count: 1,
            },
            image_extent: vk::Extent3D {
                width: region.width,
                height: region.height,
                depth: region.depth,
            },
            ..Default::default()
        }
    })
}

impl Renderer {
    pub fn new(gpu: Gpu) -> Result<Self, String> {
        let queue_lock = std::sync::Arc::new(std::sync::Mutex::new(()));
        let submitter =
            if std::env::var_os("ECHORA_SUBMIT_THREAD").is_some_and(|value| value == "0") {
                None
            } else {
                Some(Submitter::new(&gpu, queue_lock.clone())?)
            };
        let device = &gpu.device;
        let mut slots = Vec::with_capacity(SLOTS);
        for _ in 0..SLOTS {
            match Slot::new(device, gpu.queue_family) {
                Ok(slot) => slots.push(slot),
                Err(error) => {
                    for slot in slots {
                        slot.destroy_pools_and_fence(device);
                    }
                    return Err(error);
                }
            }
        }
        let mut fences = [vk::Fence::null(); SLOTS];
        for (fence, slot) in fences.iter_mut().zip(&slots) {
            *fence = slot.fence;
        }
        let slot = slots.remove(0);
        let spares = slots;
        let memory_types = gpu.memory_properties();
        // SAFETY: the physical device belongs to the live instance.
        let (period, valid_bits) = unsafe {
            let limits = gpu
                .instance
                .get_physical_device_properties(gpu.physical)
                .limits;
            let families = gpu
                .instance
                .get_physical_device_queue_family_properties(gpu.physical);
            let bits = families
                .get(gpu.queue_family as usize)
                .map_or(0, |family| family.timestamp_valid_bits);
            (limits.timestamp_period, bits)
        };
        let timer = crate::timing::GpuTimer::from_env(device, period, valid_bits, fences);
        let mut renderer = Self {
            texture_pool: None,
            spare_texture_pool: None,
            tried_spare_texture_pool: false,
            pool_cache: None,
            pool_entry_memo: FastMap::default(),
            stats: FastMap::default(),
            pass_ends: FastMap::default(),
            debug_buffers: Vec::new(),
            loop_trace_buffers: Vec::new(),
            timer,
            shader_cache: None,
            fatal: None,
            gpu,
            queue_lock,
            submitter,
            recording: false,
            full_barrier_recorded: false,
            graphics: Default::default(),
            shader_clear_programs: Default::default(),
            memory_types,
            targets: FastMap::default(),
            archived_targets: FastMap::default(),
            archived_bytes: FastMap::default(),
            pending_archives: FastMap::default(),
            pending_archive_bytes: 0,
            archive_compressor: std::cell::OnceCell::new(),
            guest_target_keys: Vec::new(),
            uploads: FastMap::default(),
            upload_lengths: FastMap::default(),
            texel_buffers: FastMap::default(),
            labels: Vec::new(),
            work_items: 0,
            samplers: FastMap::default(),
            extra_views: FastMap::default(),
            imported: FastMap::default(),
            addressed: Vec::new(),
            superseded: Vec::new(),
            #[cfg(target_os = "linux")]
            last_span_pack_serial: 0,
            packed: Vec::new(),
            addressed_file: AddressedFile::Unset,
            dropped_shared_span: false,
            page_table: None,
            free_page_tables: Vec::new(),
            import_refused: Default::default(),
            shared: Vec::new(),
            shared_bytes: 0,
            #[cfg(target_os = "linux")]
            udmabuf: None,
            shared_refused: Default::default(),
            resident: FastMap::default(),
            view_infos: FastMap::default(),
            fresh_targets: Vec::new(),
            flushes: 0,
            completed_reads: Vec::new(),
            next_read_token: 0,
            targets_created: 0,
            target_changes: 0,
            changed_targets: Vec::new(),
            targets_recreated: 0,
            descaled: 0,
            scale: Scale::from_env(),
            unscaled: FastSet::default(),
            pending_unscaled: FastSet::default(),
            partial_clears: 0,
            flush_wait_microseconds: 0,
            written: FastSet::default(),
            hazard_barriers: 0,
            spares,
            slot,
            markers: FastMap::default(),
            free_events: Vec::new(),
            next_marker: 1,
            completed_serial: 0,
            completed_shared: Default::default(),
            hang: None,
            submitted_labels: Vec::new(),
            occlusion: None,
            mesh_count: None,
            condition: None,
        };
        let usage = vk::BufferUsageFlags::UNIFORM_BUFFER
            | vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::VERTEX_BUFFER
            | vk::BufferUsageFlags::INDEX_BUFFER
            | vk::BufferUsageFlags::INDIRECT_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        // Render conditions are read from a ring copy (`apply_condition`).
        let usage = if renderer.gpu.conditional_rendering.is_some() {
            usage | vk::BufferUsageFlags::CONDITIONAL_RENDERING_EXT
        } else {
            usage
        };
        // Addressable, where the device allows: vertex fetch tables are
        // read through their device address.
        let addressable = renderer.gpu.buffer_device_address;
        renderer.slot.ring = renderer.host_buffer_with(RING_SIZE, usage, addressable)?;
        for index in 0..renderer.spares.len() {
            let ring = renderer.host_buffer_with(RING_SIZE, usage, addressable)?;
            renderer.spares[index].ring = ring;
        }
        Ok(renderer)
    }

    /// Whether a rendering pass is open in the recorded slot.
    pub(crate) fn pass_open(&self) -> bool {
        self.slot.open_pass.is_some()
    }

    pub(crate) fn device(&self) -> &ash::Device {
        &self.gpu.device
    }

    /// The command buffer, begun if nothing is recording.
    /// The command buffer, for recording outside a rendering pass (an open
    /// one ends first).
    #[track_caller]
    pub(crate) fn commands(&mut self) -> Result<vk::CommandBuffer, String> {
        let commands = self.recording_commands()?;
        self.end_pass();
        Ok(commands)
    }

    /// End the rendering pass a draw left open, if any.
    #[track_caller]
    pub(crate) fn end_pass(&mut self) {
        if self.slot.open_pass.is_some() {
            *self
                .pass_ends
                .entry(std::panic::Location::caller())
                .or_default() += 1;
            // A condition or a segment counting samples begun inside the
            // pass ends in it (the next draw begins the condition again).
            if self.slot.condition_active && self.slot.condition_inside {
                self.end_condition();
            }
            let counting = self.occlusion_pass_ends(self.slot.commands);
            self.slot.open_pass = None;
            // SAFETY: recording, inside the pass begun by `begin_rendering`.
            unsafe { self.gpu.device.cmd_end_rendering(self.slot.commands) };
            if counting {
                self.occlusion_pass_ended(self.slot.commands);
            }
        }
    }

    /// The command buffer, for recording inside the open rendering pass.
    pub(crate) fn recording_commands(&mut self) -> Result<vk::CommandBuffer, String> {
        self.full_barrier_recorded = false;
        if !self.recording {
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            // SAFETY: the buffer is idle (its slot's last submission was
            // waited for before it became current again).
            unsafe {
                self.device()
                    .begin_command_buffer(self.slot.commands, &begin)
            }
            .map_err(vk_error("vkBeginCommandBuffer"))?;
            self.recording = true;
            if let Some(timer) = self.timer.as_mut() {
                timer.begin(&self.gpu.device, self.slot.commands, self.slot.fence);
            }
            if self.spares.iter().any(|spare| spare.in_flight) {
                // Other slots' batches may still run: this one starts after
                // them, so batches finish in submission order and each
                // slot's retired resources are free once its fence signals.
                let barrier = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                    .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
                // SAFETY: recording, outside any rendering pass.
                unsafe {
                    *self
                        .stats
                        .entry("stat: barriers in recording_commands")
                        .or_default() += 1;
                    self.gpu.device.cmd_pipeline_barrier(
                        self.slot.commands,
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        vk::DependencyFlags::empty(),
                        &[barrier],
                        &[],
                        &[],
                    )
                };
            }
            self.occlusion_after_begin(self.slot.commands);
        }
        Ok(self.slot.commands)
    }

    /// Whether render conditions can be evaluated on the GPU
    /// (VK_EXT_conditional_rendering).
    pub fn supports_conditions(&self) -> bool {
        self.gpu.conditional_rendering.is_some()
    }

    /// The fences of the submission slots (they keep their handles when
    /// the slots swap).
    pub(crate) fn slot_fences(&self) -> [vk::Fence; SLOTS] {
        let mut fences = [self.slot.fence; SLOTS];
        for (slot, spare) in fences[1..].iter_mut().zip(&self.spares) {
            *slot = spare.fence;
        }
        fences
    }

    pub(crate) fn current_fence(&self) -> vk::Fence {
        self.slot.fence
    }

    /// Gate draws, dispatches and attachment clears on the 32-bit
    /// word at `offset` in `buffer` (the guest memory's device copy): they
    /// run while it is nonzero, or zero when `inverted`; None: always.
    /// The GPU reads the word in place, as the console reads a render
    /// condition from memory for the draws it gates; reports not yet
    /// written are written first.
    pub fn set_condition(&mut self, condition: Option<(vk::Buffer, u64, bool)>) {
        if self.condition != condition {
            self.end_condition();
            self.condition = condition;
        }
    }

    /// Whether beginning the wanted condition must first write reports
    /// (outside a rendering pass): one still waiting goes to the word it
    /// reads (a condition on an earlier, written report needs none).
    fn condition_needs_resolve(&self) -> bool {
        let Some((buffer, offset, _)) = self.condition else {
            return false;
        };
        !self.slot.condition_active
            && self
                .occlusion
                .as_ref()
                .is_some_and(|occlusion| occlusion.pending_at(buffer, offset))
    }

    /// Before a rendering pass begins: the report its condition
    /// reads is written (ending an open pass) when it still waits.
    pub(crate) fn prepare_condition(&mut self) -> Result<(), String> {
        if self.condition_needs_resolve() {
            let commands = self.commands()?;
            self.occlusion_resolve(commands);
        }
        Ok(())
    }

    /// Begin the wanted condition if it is not active (before recording a
    /// draw, attachment clear or dispatch, inside a rendering pass or outside any):
    /// one begun inside a pass ends with it.
    pub(crate) fn apply_condition(&mut self) -> Result<(), String> {
        let Some((buffer, offset, inverted)) = self.condition else {
            return Ok(());
        };
        if self.slot.condition_active {
            return Ok(());
        }
        let Some(conditional) = self.gpu.conditional_rendering.clone() else {
            return Ok(());
        };
        let commands = if self.condition_needs_resolve() {
            // Reports not yet written are, before the condition reads one.
            let commands = self.commands()?;
            self.occlusion_resolve(commands);
            commands
        } else {
            self.recording_commands()?
        };
        let begin = vk::ConditionalRenderingBeginInfoEXT::default()
            .buffer(buffer)
            .offset(offset)
            .flags(if inverted {
                vk::ConditionalRenderingFlagsEXT::INVERTED
            } else {
                vk::ConditionalRenderingFlagsEXT::empty()
            });
        // SAFETY: recording; the buffer is live, made for conditional
        // rendering, and its writes were made visible to all reads.
        unsafe { (conditional.fp().cmd_begin_conditional_rendering_ext)(commands, &begin) };
        self.slot.condition_active = true;
        self.slot.condition_inside = self.slot.open_pass.is_some();
        *self.stats.entry("stat: conditions begun").or_default() += 1;
        Ok(())
    }

    /// End the active condition (inside the rendering pass it began in, or
    /// outside any for one begun outside).
    pub(crate) fn end_condition(&mut self) {
        if !self.slot.condition_active {
            return;
        }
        if !self.slot.condition_inside {
            self.end_pass();
        }
        if let Some(conditional) = self.gpu.conditional_rendering.as_ref() {
            // SAFETY: recording; conditional rendering is active, and this
            // is where it began (its pass still open, or outside any).
            unsafe { (conditional.fp().cmd_end_conditional_rendering_ext)(self.slot.commands) };
        }
        self.slot.condition_active = false;
    }

    /// Copy `bytes` into the ring at an `alignment`-aligned offset, flushing
    /// first when the ring is full.
    /// A device-local storage buffer of `size` bytes kept under `key`
    /// across dispatches (created, or recreated at another size, as
    /// needed), and whether it is new (its contents undefined).
    #[track_caller]
    pub fn resident_buffer(&mut self, key: u64, size: u64) -> Result<(vk::Buffer, bool), String> {
        let size = size.max(16).next_multiple_of(16);
        if let Some(&(buffer, _, held)) = self.resident.get(&key) {
            if held == size {
                return Ok((buffer, false));
            }
            // Another size: the old one retires after the work using it.
            self.flush()?;
            let (buffer, memory, _) = self.resident.remove(&key).unwrap();
            // SAFETY: the flush finished every use.
            unsafe {
                self.gpu.device.destroy_buffer(buffer, None);
                self.gpu.device.free_memory(memory, None);
            }
        }
        let device = self.gpu.device.clone();
        let addressable = self.gpu.buffer_device_address;
        // Render conditions read reports here in place (`apply_condition`).
        let conditional_usage = if self.gpu.conditional_rendering.is_some() {
            vk::BufferUsageFlags::CONDITIONAL_RENDERING_EXT
        } else {
            vk::BufferUsageFlags::empty()
        };
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(addressed_usage(
                // INDIRECT_BUFFER: dispatches read their group counts here.
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::INDIRECT_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | conditional_usage,
                addressable,
            ))
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: valid create info; destroyed with the renderer.
        let buffer =
            unsafe { device.create_buffer(&info, None) }.map_err(vk_error("vkCreateBuffer"))?;
        // SAFETY: the buffer is live.
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let memory = self.allocate_with(
            requirements,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            addressable,
        )?;
        // SAFETY: fresh buffer and memory of the required size.
        unsafe { device.bind_buffer_memory(buffer, memory, 0) }
            .map_err(vk_error("vkBindBufferMemory"))?;
        self.resident.insert(key, (buffer, memory, size));
        Ok((buffer, true))
    }

    /// The device address of `buffer` (made by this renderer with device
    /// addresses: resident buffers, shared file spans), or None where the
    /// device has none.
    pub fn buffer_address(&self, buffer: vk::Buffer) -> Option<u64> {
        let info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
        // SAFETY: a live buffer created with SHADER_DEVICE_ADDRESS usage
        // (addressed_usage) when the device supports addresses.
        self.gpu
            .buffer_device_address
            .then(|| unsafe { self.gpu.device.get_buffer_device_address(&info) })
    }

    /// The device address of resident buffer `key`.
    pub fn resident_address(&self, key: u64) -> Option<u64> {
        let &(buffer, _, _) = self.resident.get(&key)?;
        self.buffer_address(buffer)
    }

    /// The GPU page table: `entries[i]` is the device address of guest page
    /// `first_page + i` (0: unmapped; see `echora_shader_abi::PageTable`).
    /// A replacement buffer when mappings change (the old one lives until
    /// the work that may read it has completed). Programs get this value
    /// in their push constants. The flag reports new mappings added to
    /// the existing table, which already-recorded programs can observe.
    pub fn set_page_table(
        &mut self,
        first_page: u32,
        entries: &[u64],
    ) -> Result<(echora_shader_abi::PageTable, bool), String> {
        if !self.gpu.buffer_device_address {
            return Ok((echora_shader_abi::PageTable::default(), false));
        }
        if self.page_table.as_ref().is_some_and(|table| {
            table.first_page == first_page && table.entries.len() == entries.len()
        }) {
            return self.patch_page_table(first_page, entries.len(), &[(0, entries)]);
        }
        self.replace_page_table(
            first_page,
            entries.len(),
            std::iter::once(0..entries.len()),
            |current| {
                current.clear();
                current.extend_from_slice(entries);
            },
        )?;
        Ok((self.page_table_binding(), false))
    }

    /// Change sorted, non-overlapping ranges of an existing page table.
    /// The existing mappings outside the patch, and the table's extent,
    /// carry over. All addresses are kept stable for submitted commands.
    pub fn patch_page_table(
        &mut self,
        first_page: u32,
        count: usize,
        patches: &[(usize, &[u64])],
    ) -> Result<(echora_shader_abi::PageTable, bool), String> {
        if !self.gpu.buffer_device_address {
            return Ok((echora_shader_abi::PageTable::default(), false));
        }
        let table = self
            .page_table
            .as_ref()
            .filter(|table| table.first_page == first_page && table.entries.len() == count)
            .ok_or("page table patch does not match the current extent")?;
        let mut last = 0;
        for &(start, patch) in patches {
            if start < last {
                return Err("page table patches overlap or are unordered".into());
            }
            last = start
                .checked_add(patch.len())
                .filter(|&end| end <= count)
                .ok_or("page table patch runs past the end of the table")?;
        }

        let mut added = Vec::new();
        let mut replace = false;
        const CHUNK: usize = 512;
        'compare: for &(start, patch) in patches {
            let old = &table.entries[start..start + patch.len()];
            for (index, (old, new)) in old.chunks(CHUNK).zip(patch.chunks(CHUNK)).enumerate() {
                if old == new {
                    continue;
                }
                if old
                    .iter()
                    .zip(new)
                    .any(|(old, new)| old != new && *old != 0)
                {
                    replace = true;
                    break 'compare;
                }
                added.push((start + index * CHUNK, new));
            }
        }
        if !replace && !added.is_empty() && table.submitted_read > self.completed_serial {
            self.poll_completed()?;
            if self.page_table.as_ref().unwrap().submitted_read > self.completed_serial {
                // Submitted shaders keep the table entries they can still read.
                replace = true;
                self.stat("stat: page tables replaced with submitted readers", 1);
            }
        }
        if replace {
            let changed = patches
                .iter()
                .map(|&(start, patch)| start..start + patch.len());
            self.replace_page_table(first_page, count, changed, |entries| {
                for &(start, patch) in patches {
                    entries[start..start + patch.len()].copy_from_slice(patch);
                }
            })?;
            return Ok((self.page_table_binding(), false));
        }
        let extended = !added.is_empty();
        if extended {
            self.mark_recycled_page_tables_dirty(
                added
                    .iter()
                    .map(|&(start, entries)| start..start + entries.len()),
            );
            let table = self.page_table.as_mut().unwrap();
            for (start, entries) in added {
                table.buffer.write_page_addresses(start * 8, entries);
                table.entries[start..start + entries.len()].copy_from_slice(entries);
            }
            self.stat("stat: page tables extended in place", 1);
        }
        Ok((self.page_table_binding(), extended))
    }

    /// Remember changes without touching buffers that the GPU can still read.
    fn mark_recycled_page_tables_dirty(
        &mut self,
        changed: impl Iterator<Item = std::ops::Range<usize>> + Clone,
    ) {
        let retired = std::iter::once(&mut self.slot)
            .chain(&mut self.spares)
            .flat_map(|slot| &mut slot.retired_page_tables);
        for table in self.free_page_tables.iter_mut().chain(retired) {
            for range in changed.clone() {
                table.mark_dirty(range);
            }
        }
    }

    fn replace_page_table(
        &mut self,
        first_page: u32,
        count: usize,
        changed: impl Iterator<Item = std::ops::Range<usize>> + Clone,
        update: impl FnOnce(&mut Vec<u64>),
    ) -> Result<(), String> {
        let bytes = (count.max(1) * 8) as u64;
        // Acquire the new buffer before retiring the current one.
        let mut storage = if self
            .free_page_tables
            .last()
            .is_some_and(|table| table.buffer.size == bytes)
        {
            self.stat("stat: page table buffers reused", 1);
            self.free_page_tables.pop().unwrap()
        } else {
            let buffer =
                self.host_buffer_with(bytes, vk::BufferUsageFlags::STORAGE_BUFFER, true)?;
            self.stat("stat: page table buffers allocated", 1);
            for old in self.free_page_tables.drain(..) {
                old.buffer.destroy(&self.gpu.device);
            }
            RecycledPageTable {
                buffer,
                dirty: std::iter::once(0..count).collect(),
            }
        };
        self.mark_recycled_page_tables_dirty(changed.clone());
        for range in changed.clone() {
            storage.mark_dirty(range);
        }
        let (mut entries, retired) = self
            .page_table
            .take()
            .map_or_else(|| (Vec::new(), None), |old| (old.entries, Some(old.buffer)));
        update(&mut entries);
        debug_assert_eq!(entries.len(), count);
        // A recycled buffer holds every entry outside its dirty ranges.
        for range in storage.dirty.drain(..) {
            let range = range.start.min(count)..range.end.min(count);
            storage
                .buffer
                .write_page_addresses(range.start * 8, &entries[range]);
        }
        let buffer = storage.buffer;
        if let Some(retired) = retired {
            // Work already recorded or submitted can still read the old
            // buffer. The end of this slot orders all those uses.
            let mut storage = RecycledPageTable {
                buffer: retired,
                dirty: storage.dirty,
            };
            for range in changed {
                storage.mark_dirty(range);
            }
            self.slot.retired_page_tables.push(storage);
        }
        self.page_table = Some(PageTableBuffer {
            address: self.buffer_address(buffer.buffer).unwrap_or(0),
            buffer,
            entries,
            first_page,
            recorded_read: false,
            submitted_read: 0,
        });
        Ok(())
    }

    fn page_table_binding(&self) -> echora_shader_abi::PageTable {
        let table = self.page_table.as_ref().unwrap();
        echora_shader_abi::PageTable {
            address: table.address,
            first_page: table.first_page,
            page_count: table.entries.len() as u32,
        }
    }

    /// Remember a command recorded with the current page-table buffer.
    pub(crate) fn record_page_table_read(&mut self, address: u64) {
        if address != 0
            && let Some(table) = &mut self.page_table
            && table.address == address
        {
            table.recorded_read = true;
        }
    }

    /// The page table's entries as last written (`set_page_table`).
    pub fn page_table_entries(&self) -> &[u64] {
        self.page_table
            .as_ref()
            .map_or(&[], |table| table.entries.as_slice())
    }

    /// Device address ranges and their offsets in the imported guest file.
    pub fn shared_device_file_mappings(&self) -> Option<(i32, Vec<(std::ops::Range<u64>, u64)>)> {
        let AddressedFile::Single(file) = self.addressed_file else {
            return None;
        };
        let ordinary = self
            .addressed
            .iter()
            .chain(&self.superseded)
            .map(|&(offset, size, _, _, address, _)| (address..address + size, offset));
        let packed = self.packed.iter().flat_map(|group| {
            group
                .ranges
                .iter()
                .map(|&(offset, size, at)| (group.address + at..group.address + at + size, offset))
        });
        Some((file, ordinary.chain(packed).collect()))
    }

    /// Whether a device address covers these exact bytes of the shared
    /// guest file. Previous imports remain valid until the table replaces
    /// their device addresses.
    pub fn shares_device_range(&self, file: i32, offset: u64, length: u64, device: u64) -> bool {
        if self.addressed_file != AddressedFile::Single(file) {
            return false;
        }
        let after = self
            .addressed
            .partition_point(|&(start, ..)| start <= offset);
        self.addressed[..after]
            .iter()
            .rev()
            .take_while(|&&(start, ..)| start + MAX_SHARED_SPAN >= offset)
            .chain(&self.superseded)
            .any(|&(start, size, _, _, address, _)| {
                offset
                    .checked_sub(start)
                    .is_some_and(|inside| inside + length <= size && address + inside == device)
            })
            || self.packed.iter().any(|group| {
                group
                    .offset(offset, length)
                    .is_some_and(|inside| group.address + inside == device)
            })
    }

    /// Record a copy of `bytes` into resident buffer `key` at `at` (before
    /// work that reads it, after work that used it). Uploads larger than
    /// the ring go in pieces.
    pub fn upload_resident(&mut self, key: u64, at: u64, bytes: &[u8]) -> Result<(), String> {
        self.upload_resident_with(key, at, bytes.len() as u64, |done, piece| {
            piece.copy_from_slice(&bytes[done as usize..done as usize + piece.len()]);
        })
    }

    /// `upload_resident` of `length` bytes that `fill(offset, piece)`
    /// writes straight into the staging ring, piece by piece (no copy of
    /// the whole range on the side first).
    pub fn upload_resident_with(
        &mut self,
        key: u64,
        at: u64,
        length: u64,
        mut fill: impl FnMut(u64, &mut [u8]),
    ) -> Result<(), String> {
        const PIECE: u64 = 8 << 20;
        let mut done = 0;
        while done < length {
            let piece = (length - done).min(PIECE);
            let (source, offset) = self.ring_alloc(piece, 16)?;
            fill(
                done,
                self.slot.ring.slice_mut(offset as usize, piece as usize),
            );
            self.copy_to_resident(key, at + done, source, offset, piece)?;
            done += piece;
        }
        Ok(())
    }

    /// Record a fill of `length` bytes of resident buffer `key` at `at`
    /// with the 32-bit `value` (in order with other work): ClearBuffer on
    /// the GPU's copy. Both must be multiples of 4.
    pub fn fill_resident(
        &mut self,
        key: u64,
        at: u64,
        length: u64,
        value: u32,
    ) -> Result<(), String> {
        let &(buffer, _, size) = self.resident.get(&key).ok_or("no such resident buffer")?;
        let length = length.min(size.saturating_sub(at));
        if length == 0 {
            return Ok(());
        }
        let commands = self.commands()?;
        let before = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
        let after = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
        // SAFETY: recording outside rendering; the buffer is live.
        unsafe {
            let device = &self.gpu.device;
            *self
                .stats
                .entry("stat: barriers in fill_resident")
                .or_default() += 1;
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[before],
                &[],
                &[],
            );
            device.cmd_fill_buffer(commands, buffer, at, length, value);
            *self
                .stats
                .entry("stat: barriers in fill_resident")
                .or_default() += 1;
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[after],
                &[],
                &[],
            );
        }
        Ok(())
    }

    /// Record a copy of `length` ring bytes at `offset` of `source` into
    /// resident buffer `key` at `at`, ordered after earlier work.
    fn copy_to_resident(
        &mut self,
        key: u64,
        at: u64,
        source: vk::Buffer,
        offset: u64,
        length: u64,
    ) -> Result<(), String> {
        let &(buffer, _, size) = self.resident.get(&key).ok_or("no such resident buffer")?;
        let commands = self.commands()?;
        let length = length.min(size.saturating_sub(at));
        let before = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
        let after = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
        // SAFETY: recording outside rendering; both buffers are live.
        unsafe {
            let device = &self.gpu.device;
            *self
                .stats
                .entry("stat: barriers in copy_to_resident")
                .or_default() += 1;
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[before],
                &[],
                &[],
            );
            device.cmd_copy_buffer(
                commands,
                source,
                buffer,
                &[vk::BufferCopy {
                    src_offset: offset,
                    dst_offset: at,
                    size: length,
                }],
            );
            *self
                .stats
                .entry("stat: barriers in copy_to_resident")
                .or_default() += 1;
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[after],
                &[],
                &[],
            );
        }
        Ok(())
    }

    /// Record a copy of `length` bytes of resident buffer `key` for the
    /// host: its token completes with the next flush (see
    /// `take_reads`).
    pub fn read_resident(&mut self, key: u64, at: u64, length: u64) -> Result<u64, String> {
        Ok(self.read_resident_ranges(key, &[(at, length)])?[0])
    }

    /// `read_resident` of several `(at, length)` ranges of buffer `key`,
    /// recorded as one copy (a frame's copy-back reads hundreds of small
    /// pieces of a pool). Submission makes the results visible to the
    /// host along with the other deferred reads. One token per range, in
    /// order.
    pub fn read_resident_ranges(
        &mut self,
        key: u64,
        ranges: &[(u64, u64)],
    ) -> Result<Vec<u64>, String> {
        let &(buffer, _, size) = self.resident.get(&key).ok_or("no such resident buffer")?;
        let ranges: Vec<(u64, u64)> = ranges
            .iter()
            .map(|&(at, length)| (at, length.min(size.saturating_sub(at))))
            .collect();
        let mut tokens = Vec::with_capacity(ranges.len());
        let mut first = 0;
        while first < ranges.len() {
            // As many ranges as one ring allocation holds (allocating them
            // one by one could flush the ring between them).
            let mut total = 0u64;
            let mut last = first;
            while last < ranges.len() {
                let next = total.next_multiple_of(16) + ranges[last].1;
                if next > RING_SIZE && last > first {
                    break;
                }
                total = next;
                last += 1;
            }
            let (target, base) = self.ring_alloc(total, 16)?;
            let commands = self.commands()?;
            // Reports still pending are written first (they come before
            // this read in command order; the batch is otherwise resolved
            // only at submission, after the copy). A condition gating the
            // resolve's dispatch ends first; the next draw begins it again.
            self.end_condition();
            self.occlusion_resolve(commands);
            let mut copies = Vec::with_capacity(last - first);
            let mut offset = base;
            for &(at, length) in &ranges[first..last] {
                offset = offset.next_multiple_of(16);
                if length != 0 {
                    copies.push(vk::BufferCopy {
                        src_offset: at,
                        dst_offset: offset,
                        size: length,
                    });
                }
                tokens.push(self.defer_ring_read(offset, length as usize));
                offset += length;
            }
            if !copies.is_empty() {
                let before = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
                self.stat("stat: barriers in read_resident", 1);
                // SAFETY: recording outside rendering; both buffers are live.
                unsafe {
                    let device = &self.gpu.device;
                    device.cmd_pipeline_barrier(
                        commands,
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[before],
                        &[],
                        &[],
                    );
                    device.cmd_copy_buffer(commands, buffer, target, &copies);
                }
            }
            first = last;
        }
        Ok(tokens)
    }

    /// Room for `size` bytes in the ring (flushing when it is full).
    fn ring_alloc(&mut self, size: u64, alignment: u64) -> Result<(vk::Buffer, u64), String> {
        if size > RING_SIZE {
            return Err(format!("{size}-byte read exceeds the ring"));
        }
        let offset = self.ring_place(size, alignment)?;
        Ok((self.slot.ring.buffer, offset))
    }

    /// Where `size` bytes at `alignment` go in the recorded slot's ring,
    /// sealing the slot (a flush) first when they would not fit.
    fn ring_place(&mut self, size: u64, alignment: u64) -> Result<u64, String> {
        if let Some(offset) = self.slot.ring_place(size, alignment) {
            return Ok(offset);
        }
        self.flush()?;
        self.slot
            .ring_place(size, alignment)
            .ok_or_else(|| format!("{size} bytes do not fit an empty ring"))
    }

    pub(crate) fn ring_upload(
        &mut self,
        bytes: &[u8],
        alignment: u64,
    ) -> Result<(vk::Buffer, u64), String> {
        let size = bytes.len() as u64;
        if size > RING_SIZE {
            return Err(format!("{size}-byte upload exceeds the ring"));
        }
        if let Some(placed) = self.slot.upload(bytes, alignment) {
            return Ok(placed);
        }
        self.flush()?;
        self.slot
            .upload(bytes, alignment)
            .ok_or_else(|| format!("{size} bytes do not fit an empty ring"))
    }

    /// `ring_upload` of `size` bytes that `fill` writes in place in the
    /// ring (from guest memory, without a copy in between).
    pub(crate) fn ring_upload_with(
        &mut self,
        size: u64,
        alignment: u64,
        fill: impl FnOnce(&mut [u8]) -> Result<(), String>,
    ) -> Result<(vk::Buffer, u64), String> {
        if size > RING_SIZE {
            return Err(format!("{size}-byte upload exceeds the ring"));
        }
        // `fill` runs once, in whichever slot takes the bytes.
        if !self.slot.ring_fits(size, alignment) {
            self.flush()?;
        }
        self.slot
            .upload_with(size, alignment, fill)?
            .ok_or_else(|| format!("{size} bytes do not fit an empty ring"))
    }

    /// Upload a descriptor's bytes and initialize the rest of its range.
    pub(crate) fn ring_upload_padded(
        &mut self,
        bytes: &[u8],
    ) -> Result<(vk::Buffer, u64, u64), String> {
        let range = descriptor_buffer_range(bytes.len());
        let (buffer, offset) = if range == bytes.len() as u64 {
            self.ring_upload(bytes, 256)?
        } else {
            self.ring_upload_with(range, 256, |ring| {
                let (data, padding) = ring.split_at_mut(bytes.len());
                data.copy_from_slice(bytes);
                padding.fill(0);
                Ok(())
            })?
        };
        Ok((buffer, offset, range))
    }

    /// Make room, before recording one draw or dispatch, for `bytes` of
    /// ring uploads (alignment included) and one descriptor set holding
    /// `descriptors` of each pooled type, flushing first if either would
    /// overflow. A flush in the middle would reset the pool and the ring
    /// under the work being recorded (its set would be stale and its data
    /// overwritten).
    pub(crate) fn reserve(&mut self, bytes: u64, descriptors: [u32; 4]) -> Result<(), String> {
        // Room for the render condition's word too (`apply_condition`).
        let bytes = bytes + if self.condition.is_some() { 16 } else { 0 };
        if bytes > RING_SIZE {
            return Err(format!("{bytes} bytes of uploads exceed the ring"));
        }
        if !self.slot.holds(bytes, descriptors) {
            self.flush_async()?;
        }
        self.slot.count_set(descriptors);
        Ok(())
    }

    /// `length` bytes of the ring at `offset`, after the work writing them
    /// completed (flushed) and before the ring is reused.
    /// Read `length` ring bytes at `offset` once the recorded work has
    /// completed: returns a token `take_reads` answers after
    /// the next flush.
    pub(crate) fn defer_ring_read(&mut self, offset: u64, length: usize) -> u64 {
        let token = self.next_read_token;
        self.next_read_token += 1;
        self.slot.defer_read(token, offset, length);
        token
    }

    /// Whether deferred reads wait for a flush.
    /// Whether deferred reads wait in either slot.
    pub fn reads_pending(&self) -> bool {
        !self.slot.pending_reads.is_empty()
            || self
                .spares
                .iter()
                .any(|spare| !spare.pending_reads.is_empty())
    }

    /// The completed deferred reads among `tokens`, by token; other
    /// readers' results stay for them.
    pub fn take_reads(&mut self, tokens: &[u64]) -> HashMap<u64, Vec<u8>> {
        let (Some(&first), Some(&last)) = (tokens.first(), tokens.last()) else {
            return HashMap::new();
        };
        // Readback batches usually request consecutive tokens.
        let contiguous = tokens
            .windows(2)
            .all(|pair| pair[0].checked_add(1) == Some(pair[1]));
        let sparse = (!contiguous).then(|| tokens.iter().copied().collect::<FastSet<_>>());
        let mut reads = HashMap::with_capacity(tokens.len().min(self.completed_reads.len()));
        self.completed_reads.retain_mut(|(token, bytes)| {
            let wanted = match &sparse {
                Some(tokens) => tokens.contains(token),
                None => (first..=last).contains(token),
            };
            if wanted {
                reads.insert(*token, std::mem::take(bytes));
            }
            !wanted
        });
        reads
    }

    pub(crate) fn ring_read(&self, offset: u64, length: usize) -> Vec<u8> {
        assert!(offset + length as u64 <= self.slot.ring.size);
        // SAFETY: in bounds of the mapped ring; the GPU work has completed.
        unsafe { std::slice::from_raw_parts(self.slot.ring.mapped.add(offset as usize), length) }
            .to_vec()
    }

    /// A descriptor set of `layout` from this flush's pool.
    pub(crate) fn descriptor_set(
        &mut self,
        layout: vk::DescriptorSetLayout,
    ) -> Result<vk::DescriptorSet, String> {
        if let Some(set) = self.slot.allocate_set(&self.gpu.device, layout)? {
            return Ok(set);
        }
        // Sets already written stay valid until their work is done. The
        // flush makes another slot the one recorded: the set comes from
        // that slot's pool, not from the pool of the submission just sent
        // (reset when that slot is reused, under work recorded now that
        // still uses the set).
        self.flush()?;
        *self
            .stats
            .entry("stat: descriptor pool full mid-draw")
            .or_default() += 1;
        self.slot
            .allocate_set(&self.gpu.device, layout)?
            .ok_or_else(|| "vkAllocateDescriptorSets: an empty pool is full".to_string())
    }

    /// The image view a texture binding of the shader's `texture_type`
    /// (the recompiler's TextureType: 1D, 1D array, 2D, 2D array, 3D, cube,
    /// cube array, buffer, rectangle) samples, uploading a guest texture
    /// when its generation changed.
    /// `texture_view` with `components` applied (a texture view's swizzle)
    /// and, when it is the image's sRGB twin, `format` (a texture view's
    /// format: sRGB decoding of a UNORM texture, or none of an sRGB one).
    pub(crate) fn texture_view_swizzled(
        &mut self,
        source: &crate::draw::TextureSource,
        texture_type: u32,
        sampler: &crate::draw::Sampler,
    ) -> Result<vk::ImageView, String> {
        let (components, format) = (sampler.components, sampler.format);
        let plain = self.texture_view_with_layers(source, texture_type, sampler.view_layers)?;
        let plain = match sampler.view_levels {
            Some((base, count)) => self.level_view_of(plain, base, count)?,
            None => plain,
        };
        let swizzle = [components.r, components.g, components.b, components.a];
        let identity = swizzle
            .into_iter()
            .all(|part| part == vk::ComponentSwizzle::IDENTITY);
        if identity && format.is_none() {
            return Ok(plain);
        }
        let Some(info) = self.view_infos.get(&plain).copied() else {
            return Ok(plain);
        };
        let format = format
            .filter(|format| srgb_twin(info.format) == Some(*format))
            .unwrap_or(info.format);
        if identity && format == info.format {
            return Ok(plain);
        }
        let cache = (
            info.image,
            ExtraView::Swizzled {
                source: plain,
                format,
                components: swizzle,
            },
        );
        if let Some(view) = self.extra_views.get(&cache) {
            return Ok(*view);
        }
        // A twin-format view samples only (an sRGB format has no storage
        // use its image may have).
        let mut usage = vk::ImageViewUsageCreateInfo::default().usage(vk::ImageUsageFlags::SAMPLED);
        let mut create = vk::ImageViewCreateInfo::default()
            .image(info.image)
            .view_type(info.view_type)
            .format(format)
            .components(components)
            .subresource_range(info.range);
        if format != info.format {
            create = create.push_next(&mut usage);
        }
        // SAFETY: the image is live; the view is destroyed with it.
        let view = unsafe { self.gpu.device.create_image_view(&create, None) }
            .map_err(vk_error("vkCreateImageView"))?;
        self.extra_views.insert(cache, view);
        Ok(view)
    }

    /// A view of levels `base`.. (`count` of them) of what `view` shows,
    /// as an NVN texture view selecting them: its level 0 is `base`.
    fn level_view_of(
        &mut self,
        view: vk::ImageView,
        base: u32,
        count: u32,
    ) -> Result<vk::ImageView, String> {
        let Some(info) = self.view_infos.get(&view).copied() else {
            return Ok(view);
        };
        let levels = info.range.level_count.max(1);
        let base = base.min(levels - 1);
        let count = if count == 0 {
            levels - base
        } else {
            count.min(levels - base)
        };
        if base == 0 && count == levels {
            return Ok(view);
        }
        let cache = (
            info.image,
            ExtraView::RelativeLevels {
                source: view,
                base,
                count,
            },
        );
        if let Some(existing) = self.extra_views.get(&cache) {
            return Ok(*existing);
        }
        let mut usage = vk::ImageViewUsageCreateInfo::default().usage(vk::ImageUsageFlags::SAMPLED);
        let create = vk::ImageViewCreateInfo::default()
            .image(info.image)
            .view_type(info.view_type)
            .format(info.format)
            .subresource_range(vk::ImageSubresourceRange {
                base_mip_level: info.range.base_mip_level + base,
                level_count: count,
                ..info.range
            })
            .push_next(&mut usage);
        // SAFETY: the image is live; the view is destroyed with it.
        let selected = unsafe { self.gpu.device.create_image_view(&create, None) }
            .map_err(vk_error("vkCreateImageView"))?;
        self.extra_views.insert(cache, selected);
        self.view_infos.insert(selected, ViewInfo::of(&create));
        Ok(selected)
    }

    /// Note how a new target's own views were made (see `view_infos`).
    fn register_views(&mut self, target: &Target) {
        let mut range = vk::ImageSubresourceRange {
            level_count: target.levels,
            layer_count: target.shape.layers(),
            ..range(target.format)
        };
        let info = ViewInfo {
            image: target.image,
            view_type: own_view_type(target.shape),
            format: target.format,
            range,
        };
        self.view_infos.insert(target.view, info);
        if has_stencil(target.format) {
            range.aspect_mask = vk::ImageAspectFlags::DEPTH;
        }
        self.view_infos
            .insert(target.sampled_view, ViewInfo { range, ..info });
    }

    pub(crate) fn texture_view(
        &mut self,
        source: &crate::draw::TextureSource,
        texture_type: u32,
    ) -> Result<vk::ImageView, String> {
        self.texture_view_with_layers(source, texture_type, None)
    }

    fn texture_view_with_layers(
        &mut self,
        source: &crate::draw::TextureSource,
        texture_type: u32,
        layer_range: Option<(u32, u32)>,
    ) -> Result<vk::ImageView, String> {
        use crate::draw::{ImageShape, TextureSource};
        if let TextureSource::Target(key) | TextureSource::TargetView { key, .. } = source {
            self.cancel_target_archive(*key);
            self.restore_archived_target(*key)?;
        }
        // Incompatible image dimensions need separate uploads: a row of
        // a 2D image as 1D, a flat image as one 3D slice, or the slices of
        // level 0 of a volume as 2D array layers.
        if let TextureSource::Upload {
            key,
            generation,
            width,
            height,
            shape,
            format,
            levels,
            ..
        } = source
            && !block_compressed(*format)
        {
            let alias = match *shape {
                ImageShape::D2 { layers } if matches!(texture_type, 0 | 1) => {
                    Some((key ^ 1 << 63, 1, ImageShape::D1 { layers }))
                }
                ImageShape::D2 { layers: 1 } if texture_type == 4 => {
                    Some((key ^ 1 << 62, *height, ImageShape::D3 { depth: 1 }))
                }
                ImageShape::D3 { depth } if depth > 0 && matches!(texture_type, 2 | 3) => levels
                    .iter()
                    .any(|region| region.level == 0)
                    .then_some((key ^ 1 << 61, *height, ImageShape::D2 { layers: depth })),
                _ => None,
            };
            if let Some((alias, alias_height, alias_shape)) = alias {
                let current = self.uploads.get(&alias).is_some_and(|(uploaded, target)| {
                    uploaded == generation
                        && (target.width, target.height, target.format, target.shape)
                            == (*width, alias_height, *format, alias_shape)
                });
                // The alias already holds this generation. The source's
                // texels may have been released after its own upload.
                let mut source = if current {
                    source.clone()
                } else {
                    self.with_texels(source)?
                };
                let TextureSource::Upload {
                    key,
                    height,
                    shape,
                    texels,
                    levels,
                    ..
                } = &mut source
                else {
                    unreachable!("with_texels preserves an upload source");
                };
                *key = alias;
                *height = alias_height;
                *shape = alias_shape;
                if !current {
                    match alias_shape {
                        ImageShape::D1 { .. } => {
                            *levels = levels
                                .iter()
                                .map(|region| crate::draw::UploadRegion {
                                    height: 1,
                                    ..*region
                                })
                                .collect();
                        }
                        ImageShape::D2 { layers } => {
                            let base = levels
                                .iter()
                                .find(|region| region.level == 0)
                                .copied()
                                .expect("the volume has level 0");
                            let end = levels
                                .iter()
                                .map(|region| region.offset)
                                .filter(|&offset| offset > base.offset)
                                .min()
                                .unwrap_or(texels.len() as u64);
                            let slice = (end - base.offset) / u64::from(layers);
                            *levels = (0..layers)
                                .map(|layer| crate::draw::UploadRegion {
                                    layer,
                                    depth: 1,
                                    offset: base.offset + u64::from(layer) * slice,
                                    ..base
                                })
                                .collect();
                        }
                        _ => {}
                    }
                }
                return self.texture_view_with_layers(&source, texture_type, layer_range);
            }
        }
        // A view of some mip levels and/or array layers (cube faces) of a
        // render target.
        if let TextureSource::TargetView {
            key,
            base_level,
            level_count,
            base_layer,
            layer_count,
        } = *source
        {
            let whole = TextureSource::Target(key);
            let Some(target) = self.targets.get(&key) else {
                return Err("sampled target does not exist".into());
            };
            let base_level = base_level.min(target.levels - 1);
            let level_count = if level_count == 0 {
                target.levels - base_level
            } else {
                level_count.min(target.levels - base_level)
            };
            let layers = target.shape.layers();
            let base_layer = base_layer.min(layers - 1);
            let layer_count = if layer_count == 0 {
                layers - base_layer
            } else {
                layer_count.min(layers - base_layer)
            };
            let layered = matches!(
                target.shape,
                ImageShape::D2 { .. } | ImageShape::Cube { .. }
            );
            let cube = matches!(target.shape, ImageShape::Cube { .. });
            let one_dimensional = matches!(target.shape, ImageShape::D1 { .. });
            // The binding's view type over the selected range, when the
            // range can hold it.
            let view = match texture_type {
                0 if one_dimensional => Some((vk::ImageViewType::TYPE_1D, 1)),
                1 if one_dimensional => Some((vk::ImageViewType::TYPE_1D_ARRAY, layer_count)),
                2 | 8 if layered => Some((vk::ImageViewType::TYPE_2D, 1)),
                3 if layered => Some((vk::ImageViewType::TYPE_2D_ARRAY, layer_count)),
                4 if matches!(target.shape, ImageShape::D3 { .. }) => {
                    Some((vk::ImageViewType::TYPE_3D, 1))
                }
                5 if cube && layer_count >= 6 => Some((vk::ImageViewType::CUBE, 6)),
                6 if cube && layer_count >= 6 => {
                    Some((vk::ImageViewType::CUBE_ARRAY, layer_count / 6 * 6))
                }
                _ => None,
            };
            let Some((view_type, layer_count)) = view else {
                return self.texture_view(&whole, texture_type);
            };
            if (base_level, level_count, base_layer, layer_count) == (0, target.levels, 0, layers)
                && view_type == own_view_type(target.shape)
            {
                return self.texture_view(&whole, texture_type);
            }
            self.transition(key, vk::ImageLayout::GENERAL)?;
            let target = &self.targets[&key];
            return self.sampled_range_view(ViewInfo {
                image: target.image,
                view_type,
                format: target.format,
                range: vk::ImageSubresourceRange {
                    base_mip_level: base_level,
                    level_count,
                    base_array_layer: base_layer,
                    layer_count,
                    ..range(target.format)
                },
            });
        }
        // A 3D binding of a 2D render target: the application renders a 3D
        // colour lookup table slice by slice, which 2D targets do not hold
        // yet. An identity table of the same size (neutral grading) stands
        // in (an approximation).
        if let TextureSource::Target(key) = source {
            let flat = self.targets.get(key).is_some_and(|target| {
                matches!(target.shape, ImageShape::D2 { .. }) && target.width == target.height
            });
            if texture_type == 4 && flat {
                let size = self.targets[key].width.clamp(2, 64);
                let identity_key = u64::MAX - u64::from(size);
                let shape = ImageShape::D3 { depth: size };
                let format = vk::Format::R8G8B8A8_UNORM;
                if let Some((0, target)) = self.uploads.get(&identity_key)
                    && (target.width, target.height, target.shape, target.format)
                        == (size, size, shape, format)
                {
                    return Ok(target.sampled_view);
                }
                let texels: Vec<u8> = (0..size * size * size)
                    .flat_map(|index| {
                        let channel = |value: u32| (value * 255 / (size - 1)) as u8;
                        [
                            channel(index % size),
                            channel(index / size % size),
                            channel(index / (size * size)),
                            255,
                        ]
                    })
                    .collect();
                let identity = TextureSource::Upload {
                    key: identity_key,
                    generation: 0,
                    width: size,
                    height: size,
                    shape,
                    format,
                    texels: std::sync::Arc::new(texels),
                    levels: vec![crate::draw::UploadRegion {
                        level: 0,
                        layer: 0,
                        width: size,
                        height: size,
                        depth: size,
                        offset: 0,
                    }]
                    .into(),
                };
                return self.texture_view(&identity, texture_type);
            }
        }
        // A volume target read by a 2D or 2D-array binding: the device
        // samples no 2D view of a 3D image, so its slices are copied into a
        // 2D array beside it (refreshed at each such binding) and that is
        // sampled: a 2D read of a volume reads slice 0, as the hardware's.
        if let TextureSource::Target(key) | TextureSource::TargetView { key, .. } = source
            && matches!(texture_type, 2 | 3)
            && let Some(&Target {
                width,
                height,
                format,
                shape: ImageShape::D3 { depth },
                ..
            }) = self.targets.get(key)
        {
            let alias = key ^ 1 << 61;
            self.volume_as_layers(*key, alias, width, height, depth, format)?;
            return self.texture_view(&TextureSource::Target(alias), texture_type);
        }
        let key = match source {
            TextureSource::Target(key) | TextureSource::TargetView { key, .. } => {
                self.transition(*key, vk::ImageLayout::GENERAL)?;
                if !self.targets.contains_key(key) {
                    return Err("sampled target does not exist".into());
                }
                *key
            }
            TextureSource::Upload { key, .. } => {
                self.upload(source)?;
                *key
            }
        };
        let target = match source {
            TextureSource::Target(_) | TextureSource::TargetView { .. } => &self.targets[&key],
            TextureSource::Upload { .. } => &self.uploads[&key].1,
        };
        let wanted = match texture_type {
            0 => vk::ImageViewType::TYPE_1D,
            1 => vk::ImageViewType::TYPE_1D_ARRAY,
            3 => vk::ImageViewType::TYPE_2D_ARRAY,
            4 => vk::ImageViewType::TYPE_3D,
            5 => vk::ImageViewType::CUBE,
            6 => vk::ImageViewType::CUBE_ARRAY,
            _ => vk::ImageViewType::TYPE_2D,
        };
        if wanted == own_view_type(target.shape) && layer_range.is_none() {
            return Ok(target.sampled_view);
        }
        let layers = target.shape.layers();
        let (base_layer, count) = layer_range.unwrap_or_default();
        let remaining = layers
            .checked_sub(base_layer)
            .filter(|&remaining| remaining != 0)
            .ok_or_else(|| format!("texture {key:x} has no layer {base_layer}"))?;
        let selected = if count == 0 {
            remaining
        } else {
            count.min(remaining)
        };
        let layer_count = match (wanted, target.shape) {
            (vk::ImageViewType::TYPE_1D, ImageShape::D1 { .. })
            | (vk::ImageViewType::TYPE_2D, ImageShape::D2 { .. } | ImageShape::Cube { .. })
            | (vk::ImageViewType::TYPE_3D, ImageShape::D3 { .. }) => 1,
            (vk::ImageViewType::TYPE_1D_ARRAY, ImageShape::D1 { .. })
            | (vk::ImageViewType::TYPE_2D_ARRAY, ImageShape::D2 { .. } | ImageShape::Cube { .. }) => {
                selected
            }
            (vk::ImageViewType::CUBE, ImageShape::Cube { .. }) if selected >= 6 => 6,
            (vk::ImageViewType::CUBE_ARRAY, ImageShape::Cube { .. })
                if selected.is_multiple_of(6) =>
            {
                selected
            }
            _ => {
                let what = match source {
                    TextureSource::Target(_) | TextureSource::TargetView { .. } => "render target",
                    TextureSource::Upload { .. } => "upload",
                };
                return Err(format!(
                    "a {wanted:?} binding samples {what} {key:x}, a {:?} image",
                    target.shape
                ));
            }
        };
        if wanted == own_view_type(target.shape) && (base_layer, layer_count) == (0, layers) {
            return Ok(target.sampled_view);
        }
        self.sampled_range_view(ViewInfo {
            image: target.image,
            view_type: wanted,
            format: target.format,
            range: vk::ImageSubresourceRange {
                level_count: target.levels,
                base_array_layer: base_layer,
                layer_count,
                ..range(target.format)
            },
        })
    }

    /// Read-only view of the selected mips and layers of a live image.
    fn sampled_range_view(&mut self, mut view: ViewInfo) -> Result<vk::ImageView, String> {
        let cache = (
            view.image,
            ExtraView::SampledRange {
                view_type: view.view_type,
                base_level: view.range.base_mip_level,
                level_count: view.range.level_count,
                base_layer: view.range.base_array_layer,
                layer_count: view.range.layer_count,
            },
        );
        if let Some(existing) = self.extra_views.get(&cache) {
            return Ok(*existing);
        }
        if has_stencil(view.format) {
            view.range.aspect_mask = vk::ImageAspectFlags::DEPTH;
        }
        let mut usage = vk::ImageViewUsageCreateInfo::default().usage(vk::ImageUsageFlags::SAMPLED);
        let info = vk::ImageViewCreateInfo::default()
            .image(view.image)
            .view_type(view.view_type)
            .format(view.format)
            .subresource_range(view.range)
            .push_next(&mut usage);
        // SAFETY: the image is live; the view is destroyed with it.
        let view = unsafe { self.gpu.device.create_image_view(&info, None) }
            .map_err(vk_error("vkCreateImageView"))?;
        self.extra_views.insert(cache, view);
        self.view_infos.insert(view, ViewInfo::of(&info));
        Ok(view)
    }

    /// Upload a guest texture when it is new or its generation changed.
    fn upload(&mut self, source: &crate::draw::TextureSource) -> Result<(), String> {
        self.upload_contents(source, None)
    }

    /// Whether a first upload is larger than the ring and can split into
    /// whole subresources that each fit into the ring.
    pub fn can_generate_upload(&self, key: u64, size: usize, largest_region: usize) -> bool {
        size as u64 > RING_SIZE
            && size as u64 <= MAX_UPLOAD_BYTES
            && largest_region as u64 <= RING_SIZE
            && !self.uploads.contains_key(&key)
    }

    /// A first upload, with source bytes generated into successive ring
    /// chunks. A chunk always holds complete subresources at their offsets.
    pub fn upload_generated(
        &mut self,
        source: &crate::draw::TextureSource,
        size: usize,
        mut fill: impl FnMut(u64, &mut [u8]) -> Result<(), String>,
    ) -> Result<(), String> {
        self.upload_contents(
            source,
            Some(GeneratedTextureUpload {
                size,
                fill: &mut fill,
            }),
        )
    }

    fn upload_contents(
        &mut self,
        source: &crate::draw::TextureSource,
        mut generated: Option<GeneratedTextureUpload<'_>>,
    ) -> Result<(), String> {
        let crate::draw::TextureSource::Upload {
            key,
            generation,
            width,
            height,
            shape,
            format,
            texels,
            levels,
        } = source
        else {
            return Ok(());
        };
        if let Some((uploaded, target)) = self.uploads.get(key) {
            if uploaded == generation
                && (target.width, target.height, target.format, target.shape)
                    == (*width, *height, *format, *shape)
            {
                return Ok(());
            }
        }
        if generated.is_some() && (self.uploads.contains_key(key) || levels.is_empty()) {
            return Err("a generated upload needs an unused key and subresource layout".into());
        }
        if generated.is_none() && texels.is_empty() && !levels.is_empty() {
            return Err(format!(
                "upload {key:#x}: texels released without a current image"
            ));
        }
        let size = generated
            .as_ref()
            .map_or(texels.len(), |generated| generated.size);
        if has_stencil(*format)
            && (levels.is_empty()
                || levels.iter().any(|region| {
                    !region.offset.is_multiple_of(4)
                        || u64::from(region.width)
                            .checked_mul(u64::from(region.height))
                            .and_then(|texels| texels.checked_mul(u64::from(region.depth)))
                            .and_then(|texels| texels.checked_mul(5))
                            .and_then(|length| region.offset.checked_add(length))
                            .is_none_or(|end| end > size as u64)
                }))
        {
            return Err("depth-stencil upload does not contain both planes".into());
        }
        let level_count = levels
            .iter()
            .map(|region| region.level + 1)
            .max()
            .unwrap_or(1);
        // A texture whose contents change keeps its image when its shape
        // does (each movie picture): the new texels are copied over the
        // old, ordered after the work that read them, with no flush and no
        // new allocation. Another shape retires the old image, freed once
        // the work submitted so far has completed.
        let mut reused = None;
        if let Some((_, old)) = self.uploads.remove(key) {
            if (old.width, old.height, old.format, old.shape) == (*width, *height, *format, *shape)
                && old.levels == level_count
            {
                reused = Some(old);
            } else {
                self.slot.retired_uploads.push(old);
            }
        }
        // A guest texture's upload never needs more than this: a larger one
        // means a misread size, shape or format, which stops the run with
        // the texture's description instead of exhausting memory.
        if size as u64 > MAX_UPLOAD_BYTES {
            let message = format!(
                "upload {key:#x}: {width}x{height} {shape:?} {format:?}, {level_count} levels, is {} MiB, over the {} MiB limit",
                size >> 20,
                MAX_UPLOAD_BYTES >> 20
            );
            self.fatal = Some(message.clone());
            return Err(message);
        }
        // Each region's bytes: up to the next region's offset, the last to
        // the end.
        let mut by_offset: Vec<&crate::draw::UploadRegion> = levels.iter().collect();
        by_offset.sort_by_key(|region| region.offset);
        if generated.is_some()
            && (by_offset.first().is_none_or(|region| region.offset != 0)
                || by_offset
                    .last()
                    .is_none_or(|region| region.offset >= size as u64)
                || by_offset
                    .windows(2)
                    .any(|pair| pair[0].offset == pair[1].offset))
        {
            return Err("invalid offsets for generated texture upload".into());
        }
        let region_bytes = |index: usize| {
            by_offset
                .get(index + 1)
                .map_or(size as u64, |next| next.offset)
                - by_offset[index].offset
        };
        // An upload larger than the ring (a big ASTC array decoded to
        // RGBA8) whose regions each fit goes through the ring a batch of
        // regions at a time, with no staging buffer of its own (two of
        // 852 MiB pushed host memory past its limit).
        let chunked = (generated.is_some() || size as u64 > RING_SIZE)
            && (0..by_offset.len()).all(|index| region_bytes(index) <= RING_SIZE);
        if generated.is_some() && !chunked {
            return Err("a generated upload region exceeds the ring".into());
        }
        let big = size as u64 > RING_SIZE;
        if big || Self::labelling() {
            let text = format!(
                "present - upload {key:#x} {width}x{height} {shape:?} {format:?} levels {level_count} bytes {} chunked {chunked} reused {}",
                size,
                reused.is_some()
            );
            if big {
                eprintln!("gpu: {}", &text[10..]);
            }
            if Self::labelling() {
                self.label(text);
            }
        }
        // The texels are staged before the image exists, so a failure
        // leaves nothing behind. Other uploads larger than the ring get a
        // staging buffer of their own, destroyed with the submission's
        // retired buffers.
        let (buffer, offset) = if chunked {
            (vk::Buffer::null(), 0)
        } else if texels.len() as u64 > RING_SIZE {
            let staging =
                self.host_buffer(texels.len() as u64, vk::BufferUsageFlags::TRANSFER_SRC)?;
            staging.write(0, texels);
            let buffer = staging.buffer;
            self.slot
                .retired_texel_buffers
                .push((staging, vk::BufferView::null()));
            (buffer, 0)
        } else {
            self.ring_upload(texels, 16)?
        };
        let reusing = reused.is_some();
        let mut target = match reused {
            Some(target) => target,
            None => self.create_shaped_image(
                *width,
                *height,
                *shape,
                *format,
                vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::TRANSFER_SRC,
                level_count,
            )?,
        };
        let commands = match self.commands() {
            Ok(commands) => commands,
            Err(error) => {
                self.slot.retired_uploads.push(target);
                return Err(error);
            }
        };
        let barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(if reusing {
                vk::AccessFlags::SHADER_READ
            } else {
                vk::AccessFlags::empty()
            })
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(target.image)
            .subresource_range(vk::ImageSubresourceRange {
                level_count: target.levels,
                layer_count: vk::REMAINING_ARRAY_LAYERS,
                ..range(*format)
            });
        let after = vk::ImageMemoryBarrier {
            src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            dst_access_mask: vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_READ,
            old_layout: vk::ImageLayout::GENERAL,
            ..barrier
        };
        // Levels the image cannot hold (create_shaped_image clamps) are
        // not copied.
        let copies: Vec<vk::BufferImageCopy> = levels
            .iter()
            .filter(|region| region.level < target.levels && region.layer < shape.layers())
            .flat_map(|region| upload_copies(region, *format, offset))
            .collect();
        // SAFETY: recording; the image and ring are live.
        unsafe {
            let device = &self.gpu.device;
            // A reused image: after every earlier read of it.
            *self.stats.entry("stat: barriers in upload").or_default() += 1;
            device.cmd_pipeline_barrier(
                commands,
                if reusing {
                    vk::PipelineStageFlags::ALL_COMMANDS
                } else {
                    vk::PipelineStageFlags::TOP_OF_PIPE
                },
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
            if !chunked {
                device.cmd_copy_buffer_to_image(
                    commands,
                    buffer,
                    target.image,
                    vk::ImageLayout::GENERAL,
                    &copies,
                );
            }
        }
        let result = (|| {
            if chunked {
                // Batches of consecutive regions that fill the ring; a batch
                // may flush first (the barrier above is then submitted before
                // its copies).
                let mut start = 0;
                while start < by_offset.len() {
                    let mut end = start;
                    let mut bytes = 0;
                    while end < by_offset.len() && bytes + region_bytes(end) <= RING_SIZE {
                        bytes += region_bytes(end);
                        end += 1;
                    }
                    let first = by_offset[start].offset;
                    let (ring, at) = if let Some(generated) = generated.as_mut() {
                        self.ring_upload_with(bytes, 16, |output| (generated.fill)(first, output))?
                    } else {
                        self.ring_upload(&texels[first as usize..(first + bytes) as usize], 16)?
                    };
                    let batch: Vec<vk::BufferImageCopy> = copies
                        .iter()
                        .filter(|copy| {
                            let source = copy.buffer_offset - offset;
                            source >= first && source < first + bytes
                        })
                        .map(|copy| vk::BufferImageCopy {
                            buffer_offset: at + (copy.buffer_offset - offset - first),
                            ..*copy
                        })
                        .collect();
                    let commands = self.commands()?;
                    // SAFETY: recording; the image and ring are live.
                    unsafe {
                        self.gpu.device.cmd_copy_buffer_to_image(
                            commands,
                            ring,
                            target.image,
                            vk::ImageLayout::GENERAL,
                            &batch,
                        );
                    }
                    start = end;
                }
            }
            let commands = self.commands()?;
            // SAFETY: recording; the image is live.
            unsafe {
                *self.stats.entry("stat: barriers in upload").or_default() += 1;
                self.gpu.device.cmd_pipeline_barrier(
                    commands,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::ALL_GRAPHICS
                        | vk::PipelineStageFlags::COMPUTE_SHADER
                        | vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[after],
                );
            }
            Ok(())
        })();
        if let Err(error) = result {
            self.slot.retired_uploads.push(target);
            return Err(error);
        }
        target.layout = vk::ImageLayout::GENERAL;
        if !reusing {
            self.register_views(&target);
        }
        self.uploads.insert(*key, (*generation, target));
        self.upload_lengths.insert(*key, size as u64);
        Ok(())
    }

    /// Whether upload `key` holds its texels at `generation` (its image
    /// was made from them).
    pub fn has_upload(&self, key: u64, generation: u64) -> bool {
        self.uploads
            .get(&key)
            .is_some_and(|(uploaded, _)| *uploaded == generation)
    }

    /// `source` with its texels: when the caller released them (an
    /// upload source with no texels, its image current), they are read
    /// back from the upload's image, region by region at the offsets
    /// they were uploaded from.
    pub(crate) fn with_texels(
        &mut self,
        source: &crate::draw::TextureSource,
    ) -> Result<crate::draw::TextureSource, String> {
        let crate::draw::TextureSource::Upload {
            key,
            generation,
            texels,
            levels,
            ..
        } = source
        else {
            return Ok(source.clone());
        };
        if !texels.is_empty() {
            return Ok(source.clone());
        }
        let length = *self
            .upload_lengths
            .get(key)
            .filter(|_| self.has_upload(*key, *generation))
            .ok_or_else(|| format!("upload {key:#x}: texels released and no current image"))?;
        let (image, image_levels, image_layers, format) = {
            let (_, target) = &self.uploads[key];
            (
                target.image,
                target.levels,
                target.shape.layers(),
                target.format,
            )
        };
        let buffer = self.host_buffer(length.max(16), vk::BufferUsageFlags::TRANSFER_DST)?;
        let copies: Vec<vk::BufferImageCopy> = levels
            .iter()
            .filter(|region| region.level < image_levels && region.layer < image_layers)
            .flat_map(|region| upload_copies(region, format, 0))
            .collect();
        if length > RING_SIZE {
            eprintln!("gpu: with_texels {key:#x} reads back {length} bytes");
        }
        if Self::labelling() {
            self.label(format!("present - with_texels {key:#x} bytes {length}"));
        }
        let commands = self.commands()?;
        // SAFETY: recording; the upload image is live, in GENERAL layout
        // and made with TRANSFER_SRC; the buffer holds every region.
        unsafe {
            let device = &self.gpu.device;
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            *self
                .stats
                .entry("stat: barriers in with_texels")
                .or_default() += 1;
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
            device.cmd_copy_image_to_buffer(
                commands,
                image,
                vk::ImageLayout::GENERAL,
                buffer.buffer,
                &copies,
            );
        }
        self.flush()?;
        // SAFETY: in bounds; the copy has completed.
        let bytes =
            unsafe { std::slice::from_raw_parts(buffer.mapped.cast::<u8>(), length as usize) }
                .to_vec();
        buffer.destroy(self.device());
        let mut filled = source.clone();
        if let crate::draw::TextureSource::Upload { texels, .. } = &mut filled {
            *texels = std::sync::Arc::new(bytes);
        }
        Ok(filled)
    }

    pub(crate) fn sampler(
        &mut self,
        sampler: &crate::draw::Sampler,
        integer: bool,
    ) -> Result<vk::Sampler, String> {
        let anisotropy = sampler
            .max_anisotropy
            .max(1.0)
            .min(self.gpu.max_sampler_anisotropy);
        let lod_bias = sampler.lod[2].clamp(
            -self.gpu.max_sampler_lod_bias,
            self.gpu.max_sampler_lod_bias,
        );
        let (mipmap, min_lod, max_lod) = if sampler.base_level_only {
            // A positive maximum preserves the minification filter while
            // nearest mip selection keeps the sample in level zero.
            (vk::SamplerMipmapMode::NEAREST, 0.0, 0.25)
        } else {
            (
                sampler.mipmap,
                sampler.lod[0],
                sampler.lod[1].max(sampler.lod[0]),
            )
        };
        let address = if self.gpu.sampler_mirror_clamp_to_edge {
            sampler.address
        } else {
            sampler.address.map(|mode| match mode {
                vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE => {
                    vk::SamplerAddressMode::MIRRORED_REPEAT
                }
                mode => mode,
            })
        };
        let border = if address.contains(&vk::SamplerAddressMode::CLAMP_TO_BORDER) {
            border_color(sampler.border, integer)
        } else if integer {
            vk::BorderColor::INT_TRANSPARENT_BLACK
        } else {
            vk::BorderColor::FLOAT_TRANSPARENT_BLACK
        };
        let key: SamplerKey = (
            sampler.min_filter.as_raw(),
            sampler.mag_filter.as_raw(),
            mipmap.as_raw(),
            address.map(|mode| mode.as_raw()),
            [min_lod, max_lod, lod_bias].map(f32::to_bits),
            anisotropy.to_bits(),
            sampler.compare.map(|op| op.as_raw()),
            border.as_raw(),
        );
        if let Some(existing) = self.samplers.get(&key) {
            return Ok(*existing);
        }
        let info = vk::SamplerCreateInfo::default()
            .min_filter(sampler.min_filter)
            .mag_filter(sampler.mag_filter)
            .mipmap_mode(mipmap)
            .address_mode_u(address[0])
            .address_mode_v(address[1])
            .address_mode_w(address[2])
            .min_lod(min_lod)
            .max_lod(max_lod)
            .mip_lod_bias(lod_bias)
            .anisotropy_enable(anisotropy > 1.0)
            .max_anisotropy(anisotropy)
            .compare_enable(sampler.compare.is_some())
            .compare_op(sampler.compare.unwrap_or(vk::CompareOp::ALWAYS))
            .border_color(border);
        // SAFETY: valid create info; destroyed in Drop.
        let created = unsafe { self.gpu.device.create_sampler(&info, None) }
            .map_err(vk_error("vkCreateSampler"))?;
        self.samplers.insert(key, created);
        Ok(created)
    }

    /// Submit what was recorded and wait until all submitted work has
    /// completed: deferred reads are then available and retired resources
    /// are freed.
    pub fn flush(&mut self) -> Result<(), String> {
        if self.recording {
            self.submit_current(None)?;
        }
        for index in 0..self.spares.len() {
            if self.spares[index].in_flight {
                self.swap_with(index);
                let finished = self.finish_current(false);
                self.swap_with(index);
                finished?;
            }
        }
        self.finish_current(false)
    }

    /// The highest completed submission serial, shared (see `wait_for`).
    pub fn completed_handle(&self) -> std::sync::Arc<std::sync::atomic::AtomicU64> {
        self.completed_shared.clone()
    }

    /// The highest completed submission serial while holding the renderer.
    pub fn completed_serial(&self) -> u64 {
        self.completed_serial
    }

    /// Serial of the submission that holds everything recorded so far:
    /// `wait_for` with it waits for that work and no later work.
    pub fn recorded_serial(&self) -> u64 {
        self.flushes + u64::from(self.recording)
    }

    /// Wait until the submission with this serial (and every earlier one)
    /// has completed. Work recorded after it keeps running; when the serial
    /// is still being recorded this is `flush`.
    pub fn wait_for(&mut self, serial: u64) -> Result<(), String> {
        if serial <= self.completed_serial {
            return Ok(());
        }
        if serial > self.flushes {
            return self.flush();
        }
        // Submitted: spare slots up to it can still be running (the
        // recorded one was waited for before it became current again).
        if self.slot.in_flight {
            self.finish_current(false)?;
        }
        for index in 0..self.spares.len() {
            if self.spares[index].in_flight && self.spares[index].serial <= serial {
                self.swap_with(index);
                let finished = self.finish_current(false);
                self.swap_with(index);
                finished?;
            }
        }
        Ok(())
    }

    /// A point in the commands recorded so far that the host can see the GPU
    /// pass (`marker_done`) before the rest of the batch runs: everything
    /// before it has completed and its writes are visible to the host. None
    /// when nothing is being recorded (the last submission's serial says).
    pub fn mark(&mut self) -> Result<Option<u64>, String> {
        if !self.recording {
            return Ok(None);
        }
        let commands = self.commands()?;
        // A report before the marker belongs to the work it signals.
        // Resolve it outside conditional rendering, before making writes
        // visible to the host.
        self.end_condition();
        self.occlusion_resolve(commands);
        let event = match self.free_events.pop() {
            Some(event) => event,
            // SAFETY: a valid create info on a live device.
            None => unsafe {
                self.gpu
                    .device
                    .create_event(&vk::EventCreateInfo::default(), None)
            }
            .map_err(vk_error("vkCreateEvent"))?,
        };
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ);
        // SAFETY: recording outside rendering; the event is unsignaled.
        unsafe {
            self.gpu.device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
            self.gpu
                .device
                .cmd_set_event(commands, event, vk::PipelineStageFlags::ALL_COMMANDS);
        }
        let id = self.next_marker;
        self.next_marker += 1;
        self.markers.insert(id, (event, self.flushes + 1));
        self.slot.markers.push(id);
        self.stat("stat: markers set", 1);
        Ok(Some(id))
    }

    /// Whether the GPU has passed marker `id` (or its batch completed).
    pub fn marker_done(&self, id: u64) -> bool {
        let Some(&(event, serial)) = self.markers.get(&id) else {
            return true;
        };
        serial <= self.completed_serial
            // SAFETY: the event is live.
            || unsafe { self.gpu.device.get_event_status(event) }.unwrap_or(false)
    }

    /// The batch serial marker `id` belongs to.
    pub fn marker_serial(&self, id: u64) -> Option<u64> {
        self.markers.get(&id).map(|&(_, serial)| serial)
    }

    /// Submit the batch with this serial if it is still being recorded
    /// (without waiting for it), so `poll_completed` can see it finish.
    pub fn submit_through(&mut self, serial: u64) -> Result<(), String> {
        if serial > self.flushes && self.recording {
            self.flush_async()?;
        }
        Ok(())
    }

    /// Start GPU work early when the CPU may need a result. Keep small
    /// recordings together so an early submission can overlap useful work.
    pub fn submit_for_cpu_dependency(&mut self) -> Result<(), String> {
        const MIN_BATCH_ITEMS: u32 = 32;
        if self.recording && self.work_items >= MIN_BATCH_ITEMS {
            self.flush_async()?;
            self.stat("stat: early CPU dependency submissions", 1);
        }
        Ok(())
    }

    /// Whether the submit thread's vkQueueSubmit of batch `handed` has
    /// returned (always, for batches submitted on this thread).
    fn submit_returned(&self, handed: u64) -> bool {
        self.submitter
            .as_ref()
            .is_none_or(|submitter| submitter.progress.returned(handed))
    }

    /// Note the submissions that have finished, without waiting: their
    /// slots are freed and `completed_handle` moves on.
    pub fn poll_completed(&mut self) -> Result<(), String> {
        if let Some(error) = self
            .submitter
            .as_ref()
            .and_then(|submitter| submitter.progress.failure())
        {
            return Err(error);
        }
        let signaled = |renderer: &Self, slot: &Slot| {
            // SAFETY: the fence is live, and no vkQueueSubmit uses it now.
            renderer.submit_returned(slot.handed)
                && unsafe { renderer.gpu.device.get_fence_status(slot.fence) }.unwrap_or(false)
        };
        for index in 0..self.spares.len() {
            if self.spares[index].in_flight && signaled(self, &self.spares[index]) {
                self.swap_with(index);
                let finished = self.finish_current(true);
                self.swap_with(index);
                finished?;
            }
        }
        if self.slot.in_flight && signaled(self, &self.slot) {
            self.finish_current(true)?;
        }
        Ok(())
    }

    /// End of a frame: submit what was recorded without waiting for it, so
    /// the GPU renders this frame while the CPU prepares the next. Reads of
    /// results still go through `flush`, which waits.
    pub fn submit_frame(&mut self) -> Result<(), String> {
        self.flush_async()
    }

    /// Submit what was recorded and continue in the other slot, waiting
    /// only for that slot's previous submission: the GPU runs one batch
    /// while the CPU records the next. For capacity limits (work items,
    /// ring and descriptor space), where nothing reads results.
    pub(crate) fn flush_async(&mut self) -> Result<(), String> {
        self.flush_async_with_signal(None)
    }

    /// Queue the blit and its signal. Keep the Vulkan submission ahead of
    /// the caller's present operation without waiting for the GPU.
    pub(crate) fn submit_for_present(&mut self, signal: vk::Semaphore) -> Result<(), String> {
        self.flush_async_with_signal(Some(signal))?;
        if let Some(submitter) = &self.submitter {
            submitter.progress.wait(submitter.handed)?;
        }
        Ok(())
    }

    /// Release the window before shutting down this renderer and device.
    pub fn destroy_presenter(&mut self, presenter: crate::present::Presenter) {
        let _ = self.flush();
        if let Some(submitter) = &mut self.submitter {
            submitter.stop();
        }
        presenter.destroy(&self.gpu);
    }

    fn flush_async_with_signal(&mut self, signal: Option<vk::Semaphore>) -> Result<(), String> {
        if !self.recording {
            return Ok(());
        }
        let start = std::time::Instant::now();
        self.submit_current(signal)?;
        let submitted = std::time::Instant::now();
        self.stat(
            "stat: flush ns submitting",
            (submitted - start).as_nanos() as u64,
        );
        self.swap_slot();
        // Its previous submission (if any) must finish before reuse.
        let finished = self.finish_current(false);
        self.stat(
            "stat: flush ns waiting for the other slot",
            submitted.elapsed().as_nanos() as u64,
        );
        finished
    }

    /// After a submission: the slot just submitted joins the spares as
    /// the newest, and the oldest becomes the one recorded.
    fn swap_slot(&mut self) {
        self.swap_with(0);
        self.spares.rotate_left(1);
    }

    /// Exchange the recorded slot with spare `index`.
    fn swap_with(&mut self, index: usize) {
        std::mem::swap(&mut self.slot, &mut self.spares[index]);
    }

    /// End and submit the current slot's command buffer.
    fn submit_current(&mut self, signal: Option<vk::Semaphore>) -> Result<(), String> {
        self.end_pass();
        self.end_condition();
        self.occlusion_before_submit(self.slot.commands);
        // Ring bytes the host reads back (storage a draw wrote) are made
        // visible to it once the submission completes.
        if !self.slot.pending_reads.is_empty() {
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ);
            // SAFETY: recording, outside rendering (the pass just ended).
            unsafe {
                *self
                    .stats
                    .entry("stat: barriers in submit_current")
                    .or_default() += 1;
                self.gpu.device.cmd_pipeline_barrier(
                    self.slot.commands,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[barrier],
                    &[],
                    &[],
                );
            }
        }
        self.recording = false;
        self.graphics.reset();
        self.flushes += 1;
        self.slot.serial = self.flushes;
        self.work_items = 0;
        let labels = std::mem::take(&mut self.labels);
        // The one before stays beside it (`.prev`): with two submissions in
        // flight, a hang can be in either.
        if let Some(path) = std::env::var_os("ECHORA_LAST_SUBMIT") {
            let mut previous = path.clone();
            previous.push(".prev");
            let _ = std::fs::rename(&path, previous);
            let _ = std::fs::write(path, labels.join("\n"));
        }
        // ECHORA_SUBMIT_LOG=<file> (debugging): every submission's labels
        // from present ECHORA_SUBMIT_LOG_FROM on, appended.
        if let Some(path) = std::env::var_os("ECHORA_SUBMIT_LOG") {
            let from = std::env::var("ECHORA_SUBMIT_LOG_FROM")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(0);
            let present = |label: &String| {
                label
                    .strip_prefix("present ")
                    .and_then(|rest| rest.split(' ').next())
                    .and_then(|number| number.parse::<u64>().ok())
            };
            let lines: Vec<&String> = labels
                .iter()
                .filter(|label| present(label).is_some_and(|number| number >= from))
                .collect();
            if !lines.is_empty() {
                use std::io::Write as _;
                if let Ok(mut file) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                {
                    for line in lines {
                        let _ = writeln!(file, "{line}");
                    }
                    let _ = writeln!(file, "-- submit");
                }
            }
        }
        self.submitted_labels = labels;
        if let Some(timer) = &self.timer {
            timer.end(&self.gpu.device, self.slot.commands, self.slot.fence);
        }
        *self.stats.entry("stat: submissions").or_default() += 1;
        *self
            .stats
            .entry("stat: shared spans held at submissions")
            .or_default() += (self.shared.len() + self.addressed.len() + self.packed.len()) as u64;
        // ECHORA_SPAN_LOG (debugging): every 200th submission, how the
        // page table's spans lie in the file (sizes and the gaps between).
        if self.flushes.is_multiple_of(200) && std::env::var_os("ECHORA_SPAN_LOG").is_some() {
            let gaps: Vec<u64> = self
                .addressed
                .windows(2)
                .map(|pair| pair[1].0.saturating_sub(pair[0].0 + pair[0].1))
                .collect();
            let small = gaps.iter().filter(|gap| **gap <= 64 << 10).count();
            let sizes: u64 = self.addressed.iter().map(|(_, span, ..)| span).sum();
            eprintln!(
                "spans: {} page-table ({} MiB, {} gaps of 64 KiB or less, first gaps {:x?}), {} for bindings",
                self.addressed.len(),
                sizes >> 20,
                small,
                &gaps[..gaps.len().min(24)],
                self.shared.len()
            );
        }
        *self
            .stats
            .entry("stat: page-table spans held at submissions")
            .or_default() +=
            (self.addressed.len() + self.superseded.len() + self.packed.len()) as u64;
        let start = std::time::Instant::now();
        let device = &self.gpu.device;
        // SAFETY: the buffer was begun.
        unsafe { device.end_command_buffer(self.slot.commands) }
            .map_err(vk_error("vkEndCommandBuffer"))?;
        if let Some(submitter) = &mut self.submitter {
            if let Some(error) = submitter.progress.failure() {
                return Err(error);
            }
            submitter
                .sender
                .as_ref()
                .ok_or("the submit thread has stopped")?
                .send(Submission {
                    commands: self.slot.commands,
                    fence: self.slot.fence,
                    signal,
                })
                .map_err(|_| {
                    submitter
                        .progress
                        .failure()
                        .unwrap_or_else(|| "the submit thread has stopped".to_string())
                })?;
            submitter.handed += 1;
            self.slot.handed = submitter.handed;
            *self
                .stats
                .entry("stat: flush ns handing to the submit thread")
                .or_default() += start.elapsed().as_nanos() as u64;
            *self
                .stats
                .entry("stat: submit thread ns in vkQueueSubmit")
                .or_default() += submitter
                .submit_ns
                .swap(0, std::sync::atomic::Ordering::Relaxed);
        } else {
            let buffers = [self.slot.commands];
            let submit = [vk::SubmitInfo::default()
                .command_buffers(&buffers)
                .signal_semaphores(signal.as_slice())];
            let held = self
                .queue_lock
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            // SAFETY: the buffer was ended; the fence is unsignaled; the
            // queue is held.
            unsafe { device.queue_submit(self.gpu.queue, &submit, self.slot.fence) }
                .map_err(vk_error("vkQueueSubmit"))?;
            drop(held);
            *self
                .stats
                .entry("stat: flush ns in vkQueueSubmit")
                .or_default() += start.elapsed().as_nanos() as u64;
        }
        self.slot.in_flight = true;
        if let Some(table) = &mut self.page_table
            && std::mem::take(&mut table.recorded_read)
        {
            table.submitted_read = self.slot.serial;
        }
        Ok(())
    }

    /// Wait for the current slot's submission, then free what it retired,
    /// collect its deferred reads and make the slot ready to record.
    fn finish_current(&mut self, fence_signaled: bool) -> Result<(), String> {
        if !self.slot.in_flight {
            return Ok(());
        }
        let wait_start = std::time::Instant::now();
        // A fence is the submit thread's while its vkQueueSubmit runs (the
        // call holds it exclusively). A failed submission never signals.
        if let Some(submitter) = &self.submitter {
            submitter.progress.wait(self.slot.handed)?;
        }
        let device = &self.gpu.device;
        // Under ECHORA_HANG_MARKERS a wait over 2 s reports where the GPU
        // stopped, then goes on waiting.
        // So does one with ECHORA_MESH_COUNT_LOG: the indirect mesh counts
        // read since the last report.
        if !fence_signaled {
            let mut limit = if self.hang.is_some()
                || crate::mesh_count::logging()
                || !self.loop_trace_buffers.is_empty()
            {
                2_000_000_000
            } else {
                u64::MAX
            };
            loop {
                // SAFETY: the submit thread has released the fence and the
                // queue signals it when it finishes this batch.
                match unsafe { device.wait_for_fences(&[self.slot.fence], true, limit) } {
                    Ok(()) => break,
                    Err(vk::Result::TIMEOUT) if limit != u64::MAX => {
                        if let Some(hang) = &self.hang {
                            hang.report(self.slot.serial);
                        }
                        if let Some(mesh_count) = &self.mesh_count {
                            mesh_count.report_pending("GPU stalled over 2 s");
                        }
                        self.report_loop_traces("GPU stalled over 2 s");
                        limit = u64::MAX;
                    }
                    Err(vk::Result::TIMEOUT) => {}
                    Err(error) => {
                        if let Some(mesh_count) = &self.mesh_count {
                            mesh_count.report_pending("fence wait failed");
                        }
                        self.report_loop_traces("fence wait failed");
                        return Err(vk_error("vkWaitForFences")(error));
                    }
                }
            }
        }
        // SAFETY: the fence has signaled and this slot is no longer in use.
        unsafe {
            device
                .reset_fences(&[self.slot.fence])
                .map_err(vk_error("vkResetFences"))?;
            device
                .reset_command_pool(self.slot.command_pool, vk::CommandPoolResetFlags::empty())
                .map_err(vk_error("vkResetCommandPool"))?;
            device
                .reset_descriptor_pool(
                    self.slot.descriptor_pool,
                    vk::DescriptorPoolResetFlags::empty(),
                )
                .map_err(vk_error("vkResetDescriptorPool"))?;
        }
        self.slot.in_flight = false;
        // Its markers are passed: their events are reset for reuse.
        for id in std::mem::take(&mut self.slot.markers) {
            if let Some((event, _)) = self.markers.remove(&id) {
                // SAFETY: the batch that set it has completed.
                if unsafe { self.gpu.device.reset_event(event) }.is_ok() {
                    self.free_events.push(event);
                }
            }
        }
        self.completed_serial = self.completed_serial.max(self.slot.serial);
        self.completed_shared
            .store(self.completed_serial, std::sync::atomic::Ordering::Release);
        self.flush_wait_microseconds += wait_start.elapsed().as_micros() as u64;
        if let Some(timer) = self.timer.as_mut() {
            timer.finished(
                &self.gpu.device,
                self.slot.fence,
                self.slot.serial,
                self.flush_wait_microseconds,
            );
        }
        // ECHORA_SLOW_SUBMITS_MS: the wait that counts as slow (1000).
        let slow = std::env::var("ECHORA_SLOW_SUBMITS_MS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(1000);
        if wait_start.elapsed() > std::time::Duration::from_millis(slow) {
            if let Some(path) = std::env::var_os("ECHORA_SLOW_SUBMITS") {
                use std::io::Write;
                if let Ok(mut file) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                {
                    let _ = writeln!(
                        file,
                        "submission waited {:?}:\n{}\n",
                        wait_start.elapsed(),
                        self.submitted_labels.join("\n")
                    );
                }
            }
        }
        // The work has completed: forgotten uploads can go.
        for target in std::mem::take(&mut self.slot.retired_uploads) {
            self.destroy_target(target);
        }
        for (buffer, view) in std::mem::take(&mut self.slot.retired_texel_buffers) {
            self.destroy_texel_buffer(buffer, view);
        }
        for storage in self.slot.retired_page_tables.drain(..) {
            if self.free_page_tables.len() < 2
                && self
                    .page_table
                    .as_ref()
                    .is_some_and(|table| table.buffer.size == storage.buffer.size)
            {
                self.free_page_tables.push(storage);
            } else {
                storage.buffer.destroy(&self.gpu.device);
            }
        }
        for (buffer, memory) in std::mem::take(&mut self.slot.retired_shared) {
            // SAFETY: the work reading it has completed.
            unsafe {
                self.gpu.device.destroy_buffer(buffer, None);
                self.gpu.device.free_memory(memory, None);
            }
        }
        // Deferred results leave the ring before its next use overwrites
        // them.
        let mut pending_reads = std::mem::take(&mut self.slot.pending_reads);
        for (token, offset, length) in pending_reads.drain(..) {
            let bytes = self.ring_read(offset, length);
            self.completed_reads.push((token, bytes));
        }
        self.slot.pending_reads = pending_reads;
        self.slot.ring_used = 0;
        self.slot.sets_used = 0;
        self.slot.descriptors_used = [0; 4];
        Ok(())
    }

    /// The texel-buffer view of guest buffer texture `source`: its level-0
    /// texels copied into a buffer once per generation. Buffer textures
    /// can be far longer than the largest 1D image (light lists a million
    /// texels long), which only a texel buffer reads correctly.
    pub(crate) fn texel_buffer_view(
        &mut self,
        source: &crate::draw::TextureSource,
    ) -> Result<vk::BufferView, String> {
        let crate::draw::TextureSource::Upload {
            key,
            generation,
            format,
            texels,
            levels,
            ..
        } = source
        else {
            return Err("texel buffers from render targets are not modeled".into());
        };
        if let Some((cached, _, view)) = self.texel_buffers.get(key) {
            if cached == generation {
                return Ok(*view);
            }
        }
        // Released texels come back from the upload's image.
        if texels.is_empty() {
            let filled = self.with_texels(source)?;
            return self.texel_buffer_view(&filled);
        }
        let start = levels.first().map_or(0, |region| region.offset as usize);
        let end = levels
            .get(1)
            .map_or(texels.len(), |region| region.offset as usize)
            .min(texels.len());
        let texel_size = format_bytes(*format) as usize;
        if texel_size == 0 {
            return Err(format!("{format:?} has no texel-buffer form"));
        }
        let bytes = &texels[start.min(end)..end];
        let length = bytes.len() / texel_size * texel_size;
        let buffer = self.host_buffer(
            length.max(texel_size) as u64,
            vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER,
        )?;
        buffer.write(0, &bytes[..length]);
        let info = vk::BufferViewCreateInfo::default()
            .buffer(buffer.buffer)
            .format(bit_preserving_float(*format))
            .offset(0)
            .range(length.max(texel_size) as u64);
        // SAFETY: the buffer is live; the view is destroyed with it.
        let view = match unsafe { self.device().create_buffer_view(&info, None) } {
            Ok(view) => view,
            Err(error) => {
                buffer.destroy(self.device());
                return Err(format!("vkCreateBufferView {format:?}: {error}"));
            }
        };
        if let Some((_, old, old_view)) =
            self.texel_buffers.insert(*key, (*generation, buffer, view))
        {
            self.slot.retired_texel_buffers.push((old, old_view));
        }
        Ok(view)
    }

    fn destroy_texel_buffer(&self, buffer: HostBuffer, view: vk::BufferView) {
        // SAFETY: the work reading it has completed.
        unsafe { self.device().destroy_buffer_view(view, None) };
        buffer.destroy(self.device());
    }

    /// `length` bytes of guest memory's file `file` at `offset` as a storage
    /// buffer the GPU reads and writes in place: the buffer and the offset
    /// in it. The page-aligned span is shared as a dma-buf (a udmabuf of the
    /// file, which must be sealed against shrinking), so the GPU sees the
    /// CPU's writes without flushes, as the console's does. `None` where
    /// that is unavailable (no dma-buf import, no /dev/udmabuf, a span over
    /// udmabuf's 64 MiB limit, an import the driver refuses): the caller
    /// copies instead. Spans are kept for reuse, the least recently used
    /// dropped beyond MAX_SHARED_BYTES (udmabuf pins their pages).
    pub fn import_file(
        &mut self,
        file: i32,
        offset: u64,
        length: u64,
    ) -> Result<Option<(vk::Buffer, u64)>, String> {
        const PAGE: u64 = 4096;
        // Imports are made a chunk at a time and shared by every binding
        // inside one, with the page table's spans too: each imported
        // memory object is listed in every submission, so thousands of
        // binding-sized ones made each vkQueueSubmit take milliseconds.
        const CHUNK: u64 = 16 << 20;
        let start = offset & !(PAGE - 1);
        let span = (offset - start + length.max(1)).next_multiple_of(PAGE);
        let key = (start, span);
        if self.addressed_file == AddressedFile::Single(file)
            && let Some((group, inside)) = self
                .packed
                .iter()
                .find_map(|group| group.offset(start, span).map(|inside| (group, inside)))
        {
            return Ok(Some((group.buffer, inside + offset - start)));
        }
        let after = self
            .addressed
            .partition_point(|(known, ..)| *known <= start);
        if let Some(&(known, _, buffer, ..)) = self.addressed[..after]
            .iter()
            .rev()
            .take_while(|(known, ..)| known + MAX_SHARED_SPAN >= start)
            .find(|(known, held, ..)| start + span <= known + held)
        {
            return Ok(Some((buffer, offset - known)));
        }
        if let Some(index) = self
            .shared
            .iter()
            .position(|((known, held), _, _)| *known <= start && start + span <= known + held)
        {
            let entry = self.shared.remove(index);
            let ((known, _), buffer, _) = entry;
            self.shared.push(entry);
            return Ok(Some((buffer, offset - known)));
        }
        if span > MAX_SHARED_SPAN || self.shared_refused.contains(&key) {
            return Ok(None);
        }
        // The chunk around the span, where the file holds it all; else the
        // span alone.
        let chunk_start = start & !(CHUNK - 1);
        let chunk_span = (start + span - chunk_start).next_multiple_of(CHUNK);
        let widened = if chunk_span <= MAX_SHARED_SPAN && (chunk_start, chunk_span) != key {
            self.share_span(file, chunk_start, chunk_span)?
                .map(|imported| ((chunk_start, chunk_span), imported))
        } else {
            None
        };
        let (key, imported) = match widened {
            Some(widened) => widened,
            None => match self.share_span(file, start, span)? {
                Some(imported) => (key, imported),
                None => {
                    self.shared_refused.insert(key);
                    return Ok(None);
                }
            },
        };
        self.stat("stat: spans imported for bindings", 1);
        let (start, span) = key;
        self.shared.push((key, imported.0, imported.1));
        self.shared_bytes += span;
        while self.shared_bytes > MAX_SHARED_BYTES && self.shared.len() > 1 {
            let ((_, dropped), buffer, memory) = self.shared.remove(0);
            self.shared_bytes -= dropped;
            self.slot.retired_shared.push((buffer, memory));
        }
        Ok(Some((imported.0, offset - start)))
    }

    /// The device address of `length` bytes of guest memory's file `file`
    /// at `offset`, shared in place (a udmabuf, as `import_file`) for the
    /// GPU page table, at most `MAX_SHARED_SPAN` bytes. The span stays until
    /// `release_file_range` covers it. None where sharing or device
    /// addresses are unavailable.
    pub fn address_span(
        &mut self,
        file: i32,
        offset: u64,
        length: u64,
    ) -> Result<Option<u64>, String> {
        const PAGE: u64 = 4096;
        let start = offset & !(PAGE - 1);
        let span = (offset - start + length.max(1)).next_multiple_of(PAGE);
        if self.addressed_file == AddressedFile::Single(file)
            && let Some(address) = self.packed.iter().find_map(|group| {
                group
                    .offset(start, span)
                    .map(|inside| group.address + inside)
            })
        {
            return Ok(Some(address + offset - start));
        }
        self.addressed_file = match self.addressed_file {
            AddressedFile::Unset => AddressedFile::Single(file),
            AddressedFile::Single(known) if known != file => AddressedFile::Multiple,
            known => known,
        };
        // Kept sorted by start; a span holding this one starts at most
        // MAX_SHARED_SPAN before it.
        let after = self
            .addressed
            .partition_point(|(known, ..)| *known <= start);
        if let Some(&(known, _, _, _, address, _)) = self.addressed[..after]
            .iter()
            .rev()
            .take_while(|(known, ..)| known + MAX_SHARED_SPAN >= start)
            .find(|(known, held, ..)| start + span <= known + held)
        {
            return Ok(Some(address + (offset - known)));
        }
        if span > MAX_SHARED_SPAN || !self.gpu.buffer_device_address {
            self.stat("stat: spans too large to share", 1);
            return Ok(None);
        }
        // Spans that touch this one in the file, or lie within a few pages
        // of it (pools are carved one after another with a page or so of
        // the allocator's between), join it: one import of their union pins
        // about the same pages, and every imported memory object is listed
        // in each submission (pools made one after another left thousands).
        // The spans it replaces stay until the table is built whole again
        // (`take_superseded`).
        const GAP: u64 = 64 << 10;
        let (mut low, mut high) = (start, start + span);
        loop {
            let grown = self
                .addressed
                .iter()
                .filter(|(known, held, ..)| *known <= high + GAP && low <= known + held + GAP)
                .fold((low, high), |(low, high), (known, held, ..)| {
                    (low.min(*known), high.max(known + held))
                });
            if grown == (low, high) || grown.1 - grown.0 > MAX_SHARED_SPAN {
                break;
            }
            (low, high) = grown;
        }
        let joined = if (low, high) != (start, start + span) {
            self.share_span(file, low, high - low)?
                .map(|imported| (low, high - low, imported))
        } else {
            None
        };
        let (start, span, (buffer, memory)) = match joined {
            Some(joined) => joined,
            None => match self.share_span(file, start, span)? {
                Some(imported) => (start, span, imported),
                None => return Ok(None),
            },
        };
        let Some(address) = self.buffer_address(buffer) else {
            self.stat("stat: shared spans without a device address", 1);
            self.slot.retired_shared.push((buffer, memory));
            return Ok(None);
        };
        let (replaced, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut self.addressed)
            .into_iter()
            .partition(|(known, held, ..)| start <= *known && known + held <= start + span);
        self.addressed = kept;
        if !replaced.is_empty() {
            self.stat("stat: page-table spans joined", replaced.len() as u64);
        }
        self.superseded.extend(replaced);
        let at = self
            .addressed
            .partition_point(|(known, ..)| *known <= start);
        self.addressed
            .insert(at, (start, span, buffer, memory, address, self.flushes));
        Ok(Some(address + (offset - start)))
    }

    /// Pack imports when the caller already needs to rebuild the table.
    #[cfg(target_os = "linux")]
    pub fn pack_addressed_spans(&mut self) -> Result<(), String> {
        self.pack_addressed_spans_before(None).map(|_| ())
    }

    /// Pack long-lived imports when there are enough to justify rebuilding
    /// the table. Keep recent imports separate so releasing one of them does
    /// not force the other ranges in a packed buffer to be mapped again.
    #[cfg(target_os = "linux")]
    pub fn compact_page_table_imports(&mut self) -> Result<bool, String> {
        if self.addressed.len() < SPAN_PACK_MIN_IMPORTS
            || self.flushes - self.last_span_pack_serial < SPAN_PACK_INTERVAL
        {
            return Ok(false);
        }
        let cutoff = self.flushes - SPAN_PACK_INTERVAL;
        let stable = self.addressed.iter().filter(|entry| entry.5 <= cutoff);
        if stable.take(SPAN_PACK_MIN_IMPORTS).count() < SPAN_PACK_MIN_IMPORTS {
            return Ok(false);
        }
        self.pack_addressed_spans_before(Some(cutoff))
    }

    #[cfg(target_os = "linux")]
    fn pack_addressed_spans_before(&mut self, cutoff: Option<u64>) -> Result<bool, String> {
        if self.addressed.len() < 64 {
            return Ok(false);
        }
        let AddressedFile::Single(file) = self.addressed_file else {
            return Ok(false);
        };
        self.last_span_pack_serial = self.flushes;
        let mut selected = cutoff
            .map(|cutoff| {
                select_stable_spans(
                    self.addressed
                        .iter()
                        .map(|&(_, size, _, _, _, imported)| (imported <= cutoff).then_some(size)),
                )
            })
            .into_iter()
            .flatten();
        let (old, mut kept): (Vec<_>, Vec<_>) = std::mem::take(&mut self.addressed)
            .into_iter()
            .partition(|_| selected.next().unwrap_or(true));
        let mut changed = false;
        let mut start = 0;
        while start < old.len() {
            let mut length = 0;
            let mut end = start;
            let mut ranges = Vec::new();
            for &(offset, size, ..) in old[start..].iter().take(128) {
                if length + size > MAX_SHARED_SPAN {
                    break;
                }
                ranges.push((offset, size, length));
                length += size;
                end += 1;
            }
            debug_assert!(end > start);
            let packed = if ranges.len() >= 4 {
                self.share_packed_spans(file, &ranges, length)
            } else {
                Ok(None)
            };
            let group = match packed {
                Ok(Some((buffer, memory))) => {
                    if let Some(address) = self.buffer_address(buffer) {
                        Some(PackedSpans {
                            ranges,
                            buffer,
                            memory,
                            address,
                        })
                    } else {
                        self.slot.retired_shared.push((buffer, memory));
                        None
                    }
                }
                Ok(None) => None,
                Err(error) => {
                    kept.extend_from_slice(&old[start..]);
                    kept.sort_unstable_by_key(|&(offset, ..)| offset);
                    self.addressed = kept;
                    return Err(error);
                }
            };
            if let Some(group) = group {
                self.stat("stat: individual imports packed", (end - start) as u64);
                self.stat("stat: packed DMA imports", 1);
                self.packed.push(group);
                self.superseded.extend_from_slice(&old[start..end]);
                changed = true;
            } else {
                kept.extend_from_slice(&old[start..end]);
            }
            start = end;
        }
        kept.sort_unstable_by_key(|&(offset, ..)| offset);
        self.addressed = kept;
        Ok(changed)
    }

    #[cfg(not(target_os = "linux"))]
    pub fn pack_addressed_spans(&mut self) -> Result<(), String> {
        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    pub fn compact_page_table_imports(&mut self) -> Result<bool, String> {
        Ok(false)
    }

    /// How many joined spans wait for the table to be built whole, the
    /// bytes they pin, and whether a released range dropped a span holding
    /// other pools too.
    pub fn superseded_spans(&self) -> (usize, u64, bool) {
        let bytes = self.superseded.iter().map(|(_, span, ..)| span).sum();
        (self.superseded.len(), bytes, self.dropped_shared_span)
    }

    /// The table was built whole: the first `count` joined-away spans
    /// (replaced before the build began, so it points into none of them)
    /// go once work recorded so far has finished.
    pub fn take_superseded(&mut self, count: usize) {
        let count = count.min(self.superseded.len());
        for (_, _, buffer, memory, ..) in self.superseded.drain(..count) {
            self.slot.retired_shared.push((buffer, memory));
        }
        self.dropped_shared_span = false;
    }

    /// Bytes of guest memory the page table's spans pin.
    pub fn addressed_bytes(&self) -> u64 {
        self.addressed.iter().map(|(_, span, ..)| span).sum::<u64>()
            + self
                .packed
                .iter()
                .flat_map(|group| &group.ranges)
                .map(|&(_, span, _)| span)
                .sum::<u64>()
    }

    /// Stop sharing file range `[offset, offset + length)`: its pages may be
    /// released (a udmabuf would keep the old ones).
    pub fn release_file_range(&mut self, offset: u64, length: u64) {
        let end = offset + length;
        self.packed.retain(|group| {
            let after = group.ranges.partition_point(|&(start, ..)| start < end);
            if group.ranges[..after]
                .iter()
                .rev()
                .take_while(|&&(start, ..)| start + MAX_SHARED_SPAN > offset)
                .any(|&(start, size, _)| offset < start + size)
            {
                self.dropped_shared_span = true;
                self.slot.retired_shared.push((group.buffer, group.memory));
                false
            } else {
                true
            }
        });
        let (gone, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut self.addressed)
            .into_iter()
            .partition(|(start, span, ..)| *start < end && offset < start + span);
        self.addressed = kept;
        for (start, span, buffer, memory, ..) in gone {
            if start < offset || end < start + span {
                self.dropped_shared_span = true;
            }
            self.slot.retired_shared.push((buffer, memory));
        }
        // Joined-away spans of the range keep old pages too: the table is
        // built whole before more work is recorded.
        let (gone, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut self.superseded)
            .into_iter()
            .partition(|(start, span, ..)| *start < end && offset < start + span);
        self.superseded = kept;
        for (_, _, buffer, memory, ..) in gone {
            self.dropped_shared_span = true;
            self.slot.retired_shared.push((buffer, memory));
        }
        let (dropped, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut self.shared)
            .into_iter()
            .partition(|((start, span), _, _)| *start < end && offset < start + span);
        self.shared = kept;
        for ((_, span), buffer, memory) in dropped {
            self.shared_bytes -= span;
            self.slot.retired_shared.push((buffer, memory));
        }
        self.shared_refused
            .retain(|(start, span)| !(*start < end && offset < start + span));
    }

    /// A udmabuf of `span` bytes of `file` at `start`, imported as a storage
    /// buffer, or None where that is unavailable.
    #[cfg(target_os = "linux")]
    fn share_span(
        &mut self,
        file: i32,
        start: u64,
        span: u64,
    ) -> Result<Option<(vk::Buffer, vk::DeviceMemory)>, String> {
        use std::os::fd::AsRawFd;
        let Some(external) = self.gpu.dma_buf_import.clone() else {
            return Ok(None);
        };
        let udmabuf = self.udmabuf.get_or_insert_with(|| {
            std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open("/dev/udmabuf")
                .ok()
        });
        let Some(udmabuf) = udmabuf else {
            self.stat("stat: udmabuf device unavailable", 1);
            return Ok(None);
        };
        // struct udmabuf_create { u32 memfd; u32 flags; u64 offset; u64 size; }
        // and UDMABUF_CREATE = _IOW('u', 0x42, struct udmabuf_create).
        #[repr(C)]
        struct Create {
            memfd: u32,
            flags: u32,
            offset: u64,
            size: u64,
        }
        let mut create = Create {
            memfd: file as u32,
            flags: UDMABUF_FLAGS_CLOEXEC,
            offset: start,
            size: span,
        };
        // SAFETY: a valid request for the udmabuf device; it returns a new
        // descriptor (or -1) and does not keep the pointer.
        let dma_buf = unsafe {
            libc::ioctl(
                udmabuf.as_raw_fd(),
                libc::_IOW::<Create>(u32::from(b'u'), 0x42),
                std::ptr::addr_of_mut!(create),
            )
        };
        if dma_buf < 0 {
            // The kernel's reason, the first few times (an import the page
            // table needs that fails leaves its pages unmapped).
            self.stat("stat: udmabuf creations refused", 1);
            if self.stats.get("stat: udmabuf creations refused") <= Some(&4) {
                eprintln!(
                    "gpu: udmabuf of {span:#x} bytes at file offset {start:#x} refused: {}",
                    std::io::Error::last_os_error()
                );
            }
            return Ok(None);
        }
        self.import_shared_fd(&external, dma_buf, span)
    }

    #[cfg(target_os = "linux")]
    fn share_packed_spans(
        &mut self,
        file: i32,
        ranges: &[(u64, u64, u64)],
        span: u64,
    ) -> Result<Option<(vk::Buffer, vk::DeviceMemory)>, String> {
        use std::os::fd::AsRawFd;
        let (Some(external), Some(Some(udmabuf))) =
            (self.gpu.dma_buf_import.clone(), self.udmabuf.as_ref())
        else {
            return Ok(None);
        };
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct Item {
            memfd: u32,
            padding: u32,
            offset: u64,
            size: u64,
        }

        #[repr(C)]
        struct ListHeader {
            flags: u32,
            count: u32,
        }

        #[repr(C)]
        struct CreateList {
            header: ListHeader,
            items: [Item; 128],
        }

        let mut create = CreateList {
            header: ListHeader {
                flags: UDMABUF_FLAGS_CLOEXEC,
                count: ranges.len() as u32,
            },
            items: [Item {
                memfd: file as u32,
                padding: 0,
                offset: 0,
                size: 0,
            }; 128],
        };
        for (item, &(offset, size, _)) in create.items.iter_mut().zip(ranges) {
            item.offset = offset;
            item.size = size;
        }
        // SAFETY: the request includes count initialized items. The kernel
        // copies them when making the new descriptor. The ioctl request
        // encodes the fixed header size, excluding the trailing items.
        let dma_buf = unsafe {
            libc::ioctl(
                udmabuf.as_raw_fd(),
                libc::_IOW::<ListHeader>(u32::from(b'u'), 0x43),
                &mut create,
            )
        };
        if dma_buf < 0 {
            self.stat("stat: packed DMA imports refused", 1);
            return Ok(None);
        }
        self.import_shared_fd(&external, dma_buf, span)
    }

    #[cfg(target_os = "linux")]
    fn import_shared_fd(
        &mut self,
        external: &ash::khr::external_memory_fd::Device,
        dma_buf: i32,
        span: u64,
    ) -> Result<Option<(vk::Buffer, vk::DeviceMemory)>, String> {
        let handle = vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;
        let mut properties = vk::MemoryFdPropertiesKHR::default();
        // SAFETY: `dma_buf` is a dma-buf descriptor we own.
        let queried =
            unsafe { external.get_memory_fd_properties(handle, dma_buf, &mut properties) };
        let device = self.gpu.device.clone();
        let close = || {
            // SAFETY: ours, and not imported.
            unsafe { libc::close(dma_buf) };
        };
        if queried.is_err() {
            close();
            self.stat("stat: dma-buf memory properties refused", 1);
            return Ok(None);
        }
        let addressable = self.gpu.buffer_device_address;
        let mut external_info = vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle);
        let info = vk::BufferCreateInfo::default()
            .size(span)
            .usage(addressed_usage(
                // INDIRECT_BUFFER: dispatches read their group counts here.
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::INDIRECT_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
                addressable,
            ))
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut external_info);
        // SAFETY: valid create info; destroyed with the span.
        let buffer = match unsafe { device.create_buffer(&info, None) } {
            Ok(buffer) => buffer,
            Err(error) => {
                close();
                return Err(format!("vkCreateBuffer: {error}"));
            }
        };
        // SAFETY: the buffer is live.
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let bits = properties.memory_type_bits & requirements.memory_type_bits;
        let refuse = |close: &dyn Fn()| {
            close();
            // SAFETY: unused buffer.
            unsafe { device.destroy_buffer(buffer, None) };
            Ok(None)
        };
        if bits == 0 || requirements.size > span {
            self.stat("stat: dma-buf imports refused (memory type or size)", 1);
            return refuse(&close);
        }
        let mut import = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(handle)
            .fd(dma_buf);
        let mut address_flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let mut allocate = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(bits.trailing_zeros())
            .push_next(&mut import);
        if addressable {
            allocate = allocate.push_next(&mut address_flags);
        }
        // SAFETY: a successful import takes ownership of `dma_buf`.
        let memory = match unsafe { device.allocate_memory(&allocate, None) } {
            Ok(memory) => memory,
            Err(_) => {
                self.stat("stat: dma-buf imports refused (allocation)", 1);
                return refuse(&close);
            }
        };
        // SAFETY: fresh buffer and memory of the required size.
        if unsafe { device.bind_buffer_memory(buffer, memory, 0) }.is_err() {
            // SAFETY: unused buffer and memory.
            unsafe {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
            }
            return Ok(None);
        }
        Ok(Some((buffer, memory)))
    }

    #[cfg(not(target_os = "linux"))]
    fn share_span(
        &mut self,
        _file: i32,
        _start: u64,
        _span: u64,
    ) -> Result<Option<(vk::Buffer, vk::DeviceMemory)>, String> {
        Ok(None)
    }

    /// `length` bytes of host memory at `pointer` as a storage buffer the
    /// GPU reads and writes in place (no copies): the buffer and the offset
    /// of `pointer` in it. `None` without host-memory import. The memory
    /// must stay mapped while the renderer lives; imports are kept per
    /// page-aligned span and reused.
    pub fn import_host(
        &mut self,
        pointer: *mut u8,
        length: u64,
    ) -> Result<Option<(vk::Buffer, u64)>, String> {
        let Some(host) = self.gpu.host_import.clone() else {
            return Ok(None);
        };
        const PAGE: usize = 4096;
        let start = pointer as usize & !(PAGE - 1);
        let offset = (pointer as usize - start) as u64;
        let Some(span) = offset
            .checked_add(length.max(1))
            .and_then(|bytes| bytes.checked_next_multiple_of(PAGE as u64))
        else {
            return Ok(None);
        };
        let alignment = self.gpu.host_import_alignment;
        // The caller only guarantees the guest pages in this span. Do
        // not round outward to the device's larger import alignment.
        if pointer.is_null()
            || !alignment.is_power_of_two()
            || ((start as u64 | span) & (alignment - 1)) != 0
        {
            return Ok(None);
        }
        if let Some((buffer, _)) = self.imported.get(&(start, span)) {
            return Ok(Some((*buffer, offset)));
        }
        if self.import_refused.contains(&(start, span)) {
            return Ok(None);
        }
        let handle = vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;
        let mut properties = vk::MemoryHostPointerPropertiesEXT::default();
        // SAFETY: `start` is a page-aligned host address the caller keeps
        // mapped (guest memory).
        let queried = unsafe {
            (host.fp().get_memory_host_pointer_properties_ext)(
                host.device(),
                handle,
                start as *const std::ffi::c_void,
                &mut properties,
            )
        }
        .result();
        if queried == Err(vk::Result::ERROR_INVALID_EXTERNAL_HANDLE) {
            self.import_refused.insert((start, span));
            return Ok(None);
        }
        queried.map_err(vk_error("vkGetMemoryHostPointerPropertiesEXT"))?;
        let device = self.gpu.device.clone();
        let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle);
        let info = vk::BufferCreateInfo::default()
            .size(span)
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut external);
        // SAFETY: valid create info; destroyed with the renderer.
        let buffer =
            unsafe { device.create_buffer(&info, None) }.map_err(vk_error("vkCreateBuffer"))?;
        // SAFETY: the buffer is live.
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let bits = properties.memory_type_bits & requirements.memory_type_bits;
        let coherent =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let Some(index) = (0..self.memory_types.memory_type_count).find(|index| {
            requirements.size <= span
                && bits & (1 << index) != 0
                && self.memory_types.memory_types[*index as usize]
                    .property_flags
                    .contains(coherent)
        }) else {
            // SAFETY: unused buffer.
            unsafe { device.destroy_buffer(buffer, None) };
            self.import_refused.insert((start, span));
            return Ok(None);
        };
        let mut import = vk::ImportMemoryHostPointerInfoEXT::default()
            .handle_type(handle)
            .host_pointer(start as *mut std::ffi::c_void);
        if span >= LARGE_ALLOCATION {
            eprintln!("gpu: {} MiB host import at {start:#x}", span >> 20);
        }
        let allocate = vk::MemoryAllocateInfo::default()
            .allocation_size(span)
            .memory_type_index(index)
            .push_next(&mut import);
        // SAFETY: the pointer is page aligned and `span` bytes stay mapped.
        let memory = match unsafe { device.allocate_memory(&allocate, None) } {
            Ok(memory) => memory,
            Err(_) => {
                // The caller copies instead.
                // SAFETY: unused buffer.
                unsafe { device.destroy_buffer(buffer, None) };
                self.import_refused.insert((start, span));
                return Ok(None);
            }
        };
        // SAFETY: fresh buffer and memory of the required size.
        if let Err(error) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
            // SAFETY: no work has referenced the buffer or its allocation.
            unsafe {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
            }
            return Err(vk_error("vkBindBufferMemory")(error));
        }
        self.imported.insert((start, span), (buffer, memory));
        Ok(Some((buffer, offset)))
    }

    #[track_caller]
    pub(crate) fn allocate(
        &self,
        requirements: vk::MemoryRequirements,
        flags: vk::MemoryPropertyFlags,
    ) -> Result<vk::DeviceMemory, String> {
        self.allocate_with(requirements, flags, false)
    }

    /// `allocate`, and with `addressable` memory buffers bound to it can
    /// have device addresses (`buffer_address`).
    #[track_caller]
    pub(crate) fn allocate_with(
        &self,
        requirements: vk::MemoryRequirements,
        flags: vk::MemoryPropertyFlags,
        addressable: bool,
    ) -> Result<vk::DeviceMemory, String> {
        // ECHORA_ALLOC_LOG (debugging): device memory allocated, summed by
        // caller, printed at each further GiB (frees are not counted).
        if std::env::var_os("ECHORA_ALLOC_LOG").is_some() {
            log_allocation(std::panic::Location::caller(), requirements.size);
        }
        // Allocations this large are unusual for guest-sized resources:
        // named with their caller, so a runaway size can be traced.
        if requirements.size >= LARGE_ALLOCATION {
            eprintln!(
                "gpu: {} MiB {flags:?} allocation at {}",
                requirements.size >> 20,
                std::panic::Location::caller()
            );
        }
        let index = (0..self.memory_types.memory_type_count)
            .find(|index| {
                requirements.memory_type_bits & (1 << index) != 0
                    && self.memory_types.memory_types[*index as usize]
                        .property_flags
                        .contains(flags)
            })
            .ok_or_else(|| format!("no memory type with {flags:?}"))?;
        let mut address_flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let mut info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(index);
        if addressable {
            info = info.push_next(&mut address_flags);
        }
        // SAFETY: valid allocate info; freed by the owner.
        unsafe { self.device().allocate_memory(&info, None) }.map_err(vk_error("vkAllocateMemory"))
    }

    /// The target for guest texture `key`, created (or recreated when its
    /// size or format changed) as needed.
    pub fn target(
        &mut self,
        key: u64,
        width: u32,
        height: u32,
        format: vk::Format,
    ) -> Result<&Target, String> {
        // Keeps the mip levels an existing target was created with.
        let levels = self.targets.get(&key).map_or(1, |target| target.levels);
        self.target_with_levels(key, width, height, format, levels)
    }

    /// The target for guest texture `key` with `levels` mip levels (draws
    /// and compute passes may render into any of them).
    pub fn target_with_levels(
        &mut self,
        key: u64,
        width: u32,
        height: u32,
        format: vk::Format,
        levels: u32,
    ) -> Result<&Target, String> {
        let shape = crate::draw::ImageShape::D2 { layers: 1 };
        self.shaped_target(key, width, height, shape, format, levels)
    }

    /// Create a storage image at the guest's extent so that shader writes
    /// address the guest's texels. A retained image keeps its contents.
    pub fn guest_sized_target(
        &mut self,
        key: u64,
        width: u32,
        height: u32,
        shape: crate::draw::ImageShape,
        format: vk::Format,
        levels: u32,
    ) -> Result<(), String> {
        if !self.targets.contains_key(&key) {
            self.pending_unscaled.insert(key);
        }
        let scaled = !self
            .shaped_target(key, width, height, shape, format, levels)?
            .scale
            .is_one();
        if scaled {
            self.keep_unscaled(key)?;
        }
        Ok(())
    }

    /// The target for guest texture `key` as a `shape` image: a 3D one for
    /// volumes compute passes write (colour lookup tables).
    pub fn shaped_target(
        &mut self,
        key: u64,
        width: u32,
        height: u32,
        shape: crate::draw::ImageShape,
        format: vk::Format,
        levels: u32,
    ) -> Result<&Target, String> {
        self.restore_archived_target(key)?;
        let largest = width.max(height).max(shape.depth()).max(1);
        let levels = levels.clamp(1, 32 - largest.leading_zeros());
        let stale = self.targets.get(&key).is_some_and(|target| {
            (
                target.width,
                target.height,
                target.format,
                target.levels,
                target.shape,
            ) != (width, height, format, levels, shape)
        });
        if stale {
            self.targets_recreated += 1;
            if std::env::var_os("ECHORA_GPU_FLOW").is_some() {
                let old = &self.targets[&key];
                eprintln!(
                    "target {key:x} recreated: {}x{} {:?} levels {} {:?} -> {width}x{height} {format:?} levels {levels} {shape:?}",
                    old.width, old.height, old.format, old.levels, old.shape
                );
            }
            // The old image goes once the work recorded so far has run
            // (retired with this batch), not after a wait for it all.
            self.target_changes += 1;
            self.changed_targets.push(key);
            let old = self.remove_target(key).unwrap();
            // The same memory taller or shorter, as the application
            // re-initializes a texture over it (a shadow map 2048 wide and
            // 2048 or 4096 high): rows both shapes have keep their texels,
            // as block-linear rows of one width lie alike in memory.
            let flat = |shape: ImageShape| matches!(shape, ImageShape::D2 { layers: 1 });
            if (old.width, old.format) == (width, format) && flat(old.shape) && flat(shape) {
                let result = self.create_target_keeping(key, &old, height, levels);
                self.slot.retired_uploads.push(old);
                result?;
                return Ok(&self.targets[&key]);
            }
            self.slot.retired_uploads.push(old);
        }
        if !self.targets.contains_key(&key) {
            let description = TargetDescription {
                width,
                height,
                shape,
                format,
                levels,
            };
            let scale = self.scale_for(key, description);
            let target = self.create_target(width, height, shape, format, levels, scale)?;
            if self.pending_unscaled.remove(&key) {
                self.unscaled.insert((key, description));
            }
            // Tiny targets (exposure values adapting from their previous
            // frame) start from guest memory, as on the console; mip chains
            // (bloom) start zeroed, as levels sampled before they are drawn
            // showed garbage. Other targets keep the device's initial
            // contents: zeroing them (or copying their guest bytes) lost the
            // 3D scene (measured on the trainer selection screen).
            self.targets_created += 1;
            let tiny = target.width * target.height <= 64;
            // (A block-compressed image cannot be cleared: its blocks
            // come in whole from `copy_blocks`.)
            let chain = target.levels > 1 && bc_block_bytes(target.format).is_none();
            self.register_views(&target);
            self.target_changes += 1;
            self.changed_targets.push(key);
            self.insert_target(key, target);
            if tiny {
                self.fresh_targets.push(key);
            } else if chain || std::env::var_os("ECHORA_ZERO_TARGETS").is_some() {
                self.zero_target(key)?;
            }
        }
        Ok(&self.targets[&key])
    }

    /// Replace a flat target at the same width and format, preserving
    /// the overlapping rows of each retained mip level.
    fn create_target_keeping(
        &mut self,
        key: u64,
        old: &Target,
        height: u32,
        levels: u32,
    ) -> Result<(), String> {
        let shape = ImageShape::D2 { layers: 1 };
        let target = self.create_target(old.width, height, shape, old.format, levels, old.scale)?;
        let compressed = bc_block_bytes(old.format).is_some();
        let retained = if old.layout == vk::ImageLayout::UNDEFINED {
            0
        } else {
            old.levels.min(levels)
        };
        let regions: Vec<_> = (0..retained)
            .filter_map(|mip_level| {
                let (width, old_height) = old.extent(mip_level);
                let (_, new_height) = target.extent(mip_level);
                let mut height = old_height.min(new_height);
                // A partial compressed block must end at both image edges.
                if compressed && old_height != new_height {
                    height -= height % 4;
                }
                let layers = vk::ImageSubresourceLayers {
                    aspect_mask: aspect(old.format),
                    mip_level,
                    base_array_layer: 0,
                    layer_count: 1,
                };
                (height > 0).then_some(vk::ImageCopy {
                    src_subresource: layers,
                    dst_subresource: layers,
                    extent: vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    },
                    ..Default::default()
                })
            })
            .collect();
        self.targets_created += 1;
        self.register_views(&target);
        self.target_changes += 1;
        self.changed_targets.push(key);
        self.insert_target(key, target);
        if levels > 1 && retained < levels && !compressed {
            self.zero_target_levels(key, retained)?;
        }
        if regions.is_empty() {
            return Ok(());
        }
        self.transition(key, vk::ImageLayout::GENERAL)?;
        let commands = self.commands()?;
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
        let image = self.targets[&key].image;
        // SAFETY: recording; both images are live and in GENERAL layout.
        unsafe {
            let device = &self.gpu.device;
            *self
                .stats
                .entry("stat: barriers in create_target_keeping")
                .or_default() += 1;
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
            device.cmd_copy_image(
                commands,
                old.image,
                vk::ImageLayout::GENERAL,
                image,
                vk::ImageLayout::GENERAL,
                &regions,
            );
            *self
                .stats
                .entry("stat: barriers in create_target_keeping")
                .or_default() += 1;
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
        Ok(())
    }

    /// Clear every mip level of a target to zero: images start with
    /// undefined contents, and levels the application samples before
    /// writing (a bloom chain's) would show garbage.
    fn zero_target(&mut self, key: u64) -> Result<(), String> {
        self.zero_target_levels(key, 0)
    }

    fn zero_target_levels(&mut self, key: u64, first_level: u32) -> Result<(), String> {
        self.transition(key, vk::ImageLayout::GENERAL)?;
        let commands = self.commands()?;
        let target = &self.targets[&key];
        let subresource = vk::ImageSubresourceRange {
            base_mip_level: first_level,
            level_count: target.levels - first_level,
            layer_count: target.shape.layers(),
            ..range(target.format)
        };
        // SAFETY: recording; the image is live and in GENERAL layout.
        unsafe {
            if is_depth(target.format) {
                self.gpu.device.cmd_clear_depth_stencil_image(
                    commands,
                    target.image,
                    vk::ImageLayout::GENERAL,
                    &vk::ClearDepthStencilValue::default(),
                    &[subresource],
                );
            } else {
                self.gpu.device.cmd_clear_color_image(
                    commands,
                    target.image,
                    vk::ImageLayout::GENERAL,
                    &vk::ClearColorValue::default(),
                    &[subresource],
                );
            }
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
            *self
                .stats
                .entry("stat: barriers in zero_target")
                .or_default() += 1;
            self.gpu.device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
        Ok(())
    }

    pub fn existing_target(&self, key: u64) -> Option<&Target> {
        self.targets.get(&key)
    }

    pub fn target_info(&self, key: u64) -> Option<TargetInfo> {
        self.targets
            .get(&key)
            .map(TargetInfo::from)
            .or_else(|| self.archived_targets.get(&key).map(|target| target.info))
    }

    /// Whether a live allocation still has its initial undefined layout.
    pub fn target_is_undefined(&self, key: u64) -> bool {
        self.targets
            .get(&key)
            .is_some_and(|target| target.layout == vk::ImageLayout::UNDEFINED)
    }

    /// Retire an allocation that has no texels to keep under another key.
    pub fn discard_undefined_target(&mut self, key: u64) {
        if self.target_is_undefined(key) {
            let target = self.remove_target(key).unwrap();
            self.slot.retired_uploads.push(target);
            self.fresh_targets.retain(|fresh| *fresh != key);
            self.target_changes += 1;
            self.changed_targets.push(key);
        }
    }

    /// Schedule one lossless snapshot while the target is inactive. Leave
    /// the image resident until the copy completes and only use ring space
    /// available in this submission.
    pub fn queue_target_archive(&mut self, key: u64) -> Result<ArchiveRequest, String> {
        let Some(target) = self.targets.get(&key) else {
            return Ok(ArchiveRequest::Ineligible);
        };
        if target.layout == vk::ImageLayout::UNDEFINED
            || self.fresh_targets.contains(&key)
            || key >= STAGING_KEYS
        {
            return Ok(ArchiveRequest::Ineligible);
        }
        let Ok((_, size)) = self.archive_copy(key) else {
            return Ok(ArchiveRequest::Ineligible);
        };
        if !(1 << 20..=RING_SIZE).contains(&size) {
            return Ok(ArchiveRequest::Ineligible);
        }
        if self.pending_archives.contains_key(&key)
            || self.pending_archive_bytes + size > RING_SIZE
            || !self.slot.ring_fits(size, 16)
        {
            return Ok(ArchiveRequest::Deferred);
        }
        let image = target.image;
        let token = self.queue_target_archive_read(key)?;
        self.pending_archives.insert(
            key,
            PendingArchive {
                read: Some(token),
                bytes: size,
                image: Some(image),
            },
        );
        self.pending_archive_bytes += size;
        Ok(ArchiveRequest::Queued)
    }

    fn cancel_target_archive(&mut self, key: u64) {
        if let Some(read) = self.pending_archives.get_mut(&key) {
            read.image = None;
        }
    }

    /// Move completed copies to the compression worker and collect its
    /// results. A target remains live, and can cancel its archive, until
    /// the result is collected. The readback limit covers both stages.
    pub fn finish_target_archives(&mut self) -> Result<(), String> {
        if self.pending_archives.is_empty() {
            return Ok(());
        }
        self.archived_bytes
            .retain(|_, block| block.strong_count() != 0);
        let compressed: Vec<_> = self
            .archive_compressor
            .get()
            .and_then(Option::as_ref)
            .into_iter()
            .flat_map(|worker| worker.results.try_iter())
            .collect();
        for (key, bytes) in compressed {
            self.finish_target_archive(key, bytes);
        }
        let mut tokens: Vec<_> = self
            .pending_archives
            .values()
            .filter_map(|archive| archive.read)
            .collect();
        if tokens.is_empty() {
            return Ok(());
        }
        self.poll_completed()?;
        tokens.sort_unstable();
        let mut ready = self.take_reads(&tokens);
        let completed: Vec<_> = self
            .pending_archives
            .iter()
            .filter_map(|(&key, archive)| Some((key, ready.remove(&archive.read?)?)))
            .collect();
        for (key, raw) in completed {
            let archive = self.pending_archives.get_mut(&key).unwrap();
            archive.read = None;
            if archive.image.is_none() {
                self.finish_target_archive(key, None);
                continue;
            }
            let worker = self
                .archive_compressor
                .get_or_init(ArchiveCompressor::start);
            let raw = match worker {
                Some(worker) => match worker.jobs.send((key, raw)) {
                    Ok(()) => continue,
                    Err(error) => error.0.1,
                },
                None => raw,
            };
            self.finish_target_archive(key, zstd::bulk::compress(&raw, 1).ok());
        }
        Ok(())
    }

    fn finish_target_archive(&mut self, key: u64, compressed: Option<Vec<u8>>) {
        let archive = self.pending_archives.remove(&key).unwrap();
        self.pending_archive_bytes -= archive.bytes;
        let Some(compressed) = compressed.filter(|bytes| bytes.len() < archive.bytes as usize)
        else {
            return;
        };
        let Some(target) = self
            .targets
            .get(&key)
            .filter(|target| Some(target.image) == archive.image)
        else {
            return;
        };
        let info = TargetInfo::from(target);
        use std::hash::Hasher;
        let mut hasher = crate::fast_map::FastHasher::default();
        hasher.write(&compressed);
        let hash = hasher.finish();
        let compressed = match self
            .archived_bytes
            .get(&hash)
            .and_then(std::sync::Weak::upgrade)
        {
            Some(shared) if shared.as_ref() == compressed => shared,
            _ => {
                let shared: std::sync::Arc<[u8]> = compressed.into_boxed_slice().into();
                self.archived_bytes
                    .insert(hash, std::sync::Arc::downgrade(&shared));
                shared
            }
        };
        let target = self.targets.remove(&key).unwrap();
        self.slot.retired_uploads.push(target);
        self.stat("stat: inactive target image bytes archived", archive.bytes);
        self.stat(
            "stat: inactive target compressed bytes",
            compressed.len() as u64,
        );
        self.archived_targets.insert(
            key,
            ArchivedTarget {
                info,
                decoded_bytes: archive.bytes as usize,
                compressed,
            },
        );
        self.target_changes += 1;
        self.changed_targets.push(key);
    }

    /// Reconstitute an image before a command refers to it. Keep the
    /// archive until recording the upload has succeeded.
    pub fn restore_archived_target(&mut self, key: u64) -> Result<(), String> {
        let Some(archived) = self.archived_targets.get(&key) else {
            return Ok(());
        };
        let info = archived.info;
        let bytes = zstd::bulk::decompress(&archived.compressed, archived.decoded_bytes)
            .map_err(|error| error.to_string())?;
        let target = self.create_target(
            info.width,
            info.height,
            info.shape,
            info.format,
            info.levels,
            info.scale,
        )?;
        self.register_views(&target);
        self.targets.insert(key, target);
        if let Err(error) = self.restore_target_archive(key, &bytes) {
            self.slot
                .retired_uploads
                .push(self.targets.remove(&key).unwrap());
            return Err(error);
        }
        self.archived_targets.remove(&key);
        self.target_changes += 1;
        self.changed_targets.push(key);
        Ok(())
    }

    /// Keys of the guest render targets (not internal staging images).
    /// Before recording a draw or dispatch: submit once a submission holds
    /// MAX_WORK_ITEMS of them. One submission is one GPU job, and the
    /// kernel resets the whole GPU when a job runs past its 10 s limit
    /// (headless runs once gathered 270 frames of work into one). Smaller
    /// submissions also let the GPU run a frame's work while the CPU still
    /// records the rest of it, instead of all of it after the present
    /// (`ECHORA_SUBMIT_ITEMS`, defaults to `SUBMIT_ITEMS`).
    pub(crate) fn bound_submission(&mut self) -> Result<(), String> {
        static LIMIT: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
        let limit = *LIMIT.get_or_init(|| {
            std::env::var("ECHORA_SUBMIT_ITEMS")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(SUBMIT_ITEMS)
                .clamp(1, MAX_WORK_ITEMS)
        });
        if self.work_items >= limit {
            self.flush_async()?;
        }
        self.work_items += 1;
        Ok(())
    }

    /// Whether submissions are labelled (`label`).
    pub fn labelling() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| {
            std::env::var_os("ECHORA_LAST_SUBMIT").is_some()
                || std::env::var_os("ECHORA_HANG_MARKERS").is_some()
                || std::env::var_os("ECHORA_SLOW_SUBMITS").is_some()
                || std::env::var_os("ECHORA_SUBMIT_LOG").is_some()
                || std::env::var_os("ECHORA_GPU_TIME").is_some_and(|value| value == "items")
        })
    }

    /// Debugging (ECHORA_GPU_TIME=items): time the draw or dispatch about
    /// to be recorded (`start`), or end its timing; it is named by the
    /// latest label without its present number.
    pub(crate) fn time_item(&mut self, start: bool) {
        let Some(timer) = self.timer.as_mut() else {
            return;
        };
        if !timer.timing_items() {
            return;
        }
        let name = self
            .labels
            .last()
            .map(|label| {
                let words: Vec<&str> = label.split_whitespace().collect();
                // "present N dispatch HASH ..." or "present N draw A/B ...".
                words.get(2..4).map_or(label.clone(), |kind| kind.join(" "))
            })
            .unwrap_or_default();
        timer.item(
            &self.gpu.device,
            self.slot.commands,
            self.slot.fence,
            &name,
            start,
        );
    }

    /// Debugging: name a piece of work in the current submission.
    pub fn label(&mut self, label: String) {
        if self.gpu.buffer_marker.is_some() && Self::hang_mark_kind(&label) {
            self.hang_mark(&label);
        }
        if self.labels.len() < 4096 {
            self.labels.push(label);
        }
    }

    /// ECHORA_HANG_MARKER_KINDS=draw,dispatch,upload,with_texels
    /// (debugging): mark only work
    /// whose label names one of these kinds ("present N <kind> ..."); all
    /// work is marked without it.
    fn hang_mark_kind(label: &str) -> bool {
        static KINDS: std::sync::OnceLock<Option<Vec<String>>> = std::sync::OnceLock::new();
        let kinds = KINDS.get_or_init(|| {
            std::env::var("ECHORA_HANG_MARKER_KINDS")
                .ok()
                .map(|kinds| kinds.split(',').map(str::to_owned).collect())
        });
        kinds.as_ref().is_none_or(|kinds| {
            label
                .split_whitespace()
                .nth(2)
                .is_some_and(|kind| kinds.iter().any(|wanted| wanted == kind))
        })
    }

    /// ECHORA_HANG_MARKERS: a marker before the work `label` names, written
    /// when the GPU has finished what came before it.
    /// ECHORA_HANG_MARKER_STAGE=top writes it when the GPU reaches it
    /// instead (the work before it may still run), and
    /// ECHORA_HANG_MARKER_IN_PASS=1 leaves an open rendering pass open:
    /// together they mark the work without ordering it.
    fn hang_mark(&mut self, label: &str) {
        static TOP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let top = *TOP.get_or_init(|| {
            std::env::var_os("ECHORA_HANG_MARKER_STAGE").is_some_and(|value| value == "top")
        });
        static IN_PASS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let in_pass = *IN_PASS.get_or_init(|| {
            std::env::var_os("ECHORA_HANG_MARKER_IN_PASS").is_some_and(|value| value == "1")
        });
        let Some(marker) = self.gpu.buffer_marker.clone() else {
            return;
        };
        if self.hang.is_none() {
            let Ok(buffer) = self.host_buffer(4, vk::BufferUsageFlags::TRANSFER_DST) else {
                return;
            };
            buffer.write(0, &[0; 4]);
            self.hang = Some(HangMarkers {
                buffer,
                next: 0,
                labels: std::collections::VecDeque::new(),
                path: std::env::var_os("ECHORA_HANG_MARKERS").unwrap_or_default(),
            });
        }
        let commands = if in_pass {
            self.recording_commands()
        } else {
            self.commands()
        };
        let Ok(commands) = commands else {
            return;
        };
        let hang = self.hang.as_mut().unwrap();
        hang.next += 1;
        // SAFETY: a recording command buffer and a live buffer of 4 bytes.
        unsafe {
            marker.cmd_write_buffer_marker(
                commands,
                if top {
                    vk::PipelineStageFlags::TOP_OF_PIPE
                } else {
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE
                },
                hang.buffer.buffer,
                0,
                hang.next,
            );
        }
        hang.labels.push_back((hang.next, label.to_owned()));
        if hang.labels.len() > 200_000 {
            hang.labels.pop_front();
        }
    }

    /// Debugging: the sizes of the renderer's caches.
    pub fn cache_sizes(&self) -> Vec<(&'static str, usize)> {
        vec![
            ("targets", self.targets.len()),
            ("uploads", self.uploads.len()),
            ("retired uploads", self.slot.retired_uploads.len()),
            ("samplers", self.samplers.len()),
            ("extra views", self.extra_views.len()),
            ("view infos", self.view_infos.len()),
            ("written", self.written.len()),
            ("pending reads", self.slot.pending_reads.len()),
            ("completed reads", self.completed_reads.len()),
            ("fresh targets", self.fresh_targets.len()),
        ]
    }

    fn insert_target(&mut self, key: u64, target: Target) -> Option<Target> {
        let previous = self.targets.insert(key, target);
        if previous.is_none() && key < STAGING_KEYS {
            let index = self.guest_target_keys.binary_search(&key).unwrap_err();
            self.guest_target_keys.insert(index, key);
        }
        previous
    }

    fn remove_target(&mut self, key: u64) -> Option<Target> {
        self.cancel_target_archive(key);
        let target = self.targets.remove(&key)?;
        if key < STAGING_KEYS {
            let index = self.guest_target_keys.binary_search(&key).unwrap();
            self.guest_target_keys.remove(index);
        }
        Some(target)
    }

    /// Guest render-target keys in address order.
    pub fn target_keys_in_order(&self) -> &[u64] {
        &self.guest_target_keys
    }

    pub fn target_keys(&self) -> Vec<u64> {
        self.target_keys_in_order().to_vec()
    }

    /// Collect guest render-target keys in address order, reusing the buffer.
    pub fn collect_target_keys(&self, keys: &mut Vec<u64>) {
        keys.clear();
        keys.extend_from_slice(self.target_keys_in_order());
    }

    /// The largest 1D/2D image extent the device supports.
    pub fn max_image_extent(&self) -> u32 {
        // SAFETY: the instance and physical device are live.
        let limits = unsafe {
            self.gpu
                .instance
                .get_physical_device_properties(self.gpu.physical)
        }
        .limits;
        limits
            .max_image_dimension1_d
            .min(limits.max_image_dimension2_d)
    }

    /// Whether images of `format` can be storage images (compute writes).
    /// The UNORM twin an sRGB format stores through, when that twin
    /// supports storage.
    pub fn storage_twin(&self, format: vk::Format) -> Option<vk::Format> {
        let srgb = matches!(
            format,
            vk::Format::R8G8B8A8_SRGB | vk::Format::B8G8R8A8_SRGB
        );
        srgb_twin(format).filter(|twin| srgb && self.supports_storage(*twin))
    }

    /// The format a storage image binding views `format` as: the format
    /// the program declares (an unsigned or signed word format of the same
    /// texel size), else `format` itself or its storage twin.
    pub(crate) fn storage_view_format(
        &self,
        format: vk::Format,
        declared: Option<vk::Format>,
    ) -> Option<vk::Format> {
        if let Some(declared) = declared.filter(|declared| *declared != format) {
            return (uint_alias(format) == uint_alias(declared)).then_some(declared);
        }
        if self.supports_storage(format) {
            Some(format)
        } else {
            self.storage_twin(format)
        }
    }

    pub fn supports_storage(&self, format: vk::Format) -> bool {
        // SAFETY: the instance and physical device are live.
        unsafe {
            self.gpu
                .instance
                .get_physical_device_format_properties(self.gpu.physical, format)
        }
        .optimal_tiling_features
        .contains(vk::FormatFeatureFlags::STORAGE_IMAGE)
    }

    /// Whether this ASTC upload can use the renderer's sampled image layout.
    pub fn supports_astc_upload(
        &self,
        format: vk::Format,
        shape: crate::draw::ImageShape,
        width: u32,
        height: u32,
        levels: u32,
    ) -> bool {
        use crate::draw::ImageShape;
        let Some(twin) = self
            .gpu
            .texture_compression_astc_ldr
            .then(|| astc_srgb_twin(format))
            .flatten()
        else {
            return false;
        };
        let flags = match shape {
            ImageShape::D2 { .. } => vk::ImageCreateFlags::MUTABLE_FORMAT,
            ImageShape::Cube { .. } => {
                vk::ImageCreateFlags::MUTABLE_FORMAT | vk::ImageCreateFlags::CUBE_COMPATIBLE
            }
            _ => return false,
        };
        let views = [format, twin];
        let mut format_list = vk::ImageFormatListCreateInfo::default().view_formats(&views);
        let info = vk::PhysicalDeviceImageFormatInfo2::default()
            .format(format)
            .ty(vk::ImageType::TYPE_2D)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST,
            )
            .flags(flags)
            .push_next(&mut format_list);
        let mut properties = vk::ImageFormatProperties2::default();
        // SAFETY: the instance and physical device are live.
        let supported = unsafe {
            self.gpu
                .instance
                .get_physical_device_image_format_properties2(
                    self.gpu.physical,
                    &info,
                    &mut properties,
                )
        };
        let limits = properties.image_format_properties;
        supported.is_ok()
            && width <= limits.max_extent.width
            && height <= limits.max_extent.height
            && levels <= limits.max_mip_levels
            && shape.layers() <= limits.max_array_layers
    }

    /// The resolution scale a new target `key` of this extent gets: the
    /// renderer's for 2D targets larger than a few texels (tiny ones hold
    /// values such as exposure, not images).
    fn scale_for(&self, key: u64, description: TargetDescription) -> Scale {
        let TargetDescription {
            width,
            height,
            shape,
            ..
        } = description;
        let scalable = matches!(shape, ImageShape::D2 { .. })
            && width * height > 64
            && key < STAGING_KEYS
            && !self.pending_unscaled.contains(&key)
            && !self.unscaled.contains(&(key, description))
            && self.scale.apply(width.max(height)) <= self.max_image_extent();
        if scalable { self.scale } else { Scale::ONE }
    }

    /// The common scale of a pass's attachments: targets bound with ones
    /// of another scale are kept unscaled from then on.
    pub(crate) fn attachment_scale(
        &mut self,
        colors: &[u64],
        depth: Option<u64>,
    ) -> Result<Scale, String> {
        if self.scale.is_one() {
            return Ok(Scale::ONE);
        }
        let attachments = || colors.iter().filter(|&&key| key != 0).chain(depth.iter());
        let mut scales = attachments().filter_map(|key| self.targets.get(key).map(|t| t.scale));
        let first = scales.next().unwrap_or(Scale::ONE);
        if scales.all(|scale| scale == first) {
            return Ok(first);
        }
        for key in attachments() {
            self.keep_unscaled(*key)?;
        }
        Ok(Scale::ONE)
    }

    /// Whether a texture binding of `source` samples a scaled target.
    /// The format a texture source's texels are in (a target's, or the
    /// decoded upload's).
    pub fn source_format(&self, source: &crate::draw::TextureSource) -> Option<vk::Format> {
        use crate::draw::TextureSource;
        match source {
            TextureSource::Target(key) | TextureSource::TargetView { key, .. } => {
                self.targets.get(key).map(|target| target.format)
            }
            TextureSource::Upload { format, .. } => Some(*format),
        }
    }

    pub(crate) fn samples_scaled(&self, source: &crate::draw::TextureSource) -> bool {
        use crate::draw::TextureSource;
        match source {
            TextureSource::Target(key) | TextureSource::TargetView { key, .. } => self
                .targets
                .get(key)
                .is_some_and(|target| !target.scale.is_one()),
            TextureSource::Upload { .. } => false,
        }
    }

    /// The renderer's resolution scale (shaders are translated for it).
    /// Chooses the render target scale. Call before any target exists.
    pub fn set_resolution_scale(&mut self, scale: Scale) {
        self.scale = scale;
    }

    /// Keep compiled shader binaries in `directory` and load them from
    /// there in later runs instead of compiling the same SPIR-V again.
    pub fn set_shader_cache(&mut self, directory: std::path::PathBuf) {
        self.shader_cache = Some(directory);
    }

    pub fn resolution_scale(&self) -> Scale {
        self.scale
    }

    /// Keep the current image at guest resolution, replacing a scaled
    /// image by a downscaled copy. Storage image writes use guest coordinates.
    /// If the key has no image, mark its next image.
    pub fn keep_unscaled(&mut self, key: u64) -> Result<(), String> {
        let Some(old) = self.targets.get(&key) else {
            self.pending_unscaled.insert(key);
            return Ok(());
        };
        let description = TargetDescription::from(old);
        if old.scale.is_one() {
            self.unscaled.insert((key, description));
            return Ok(());
        }
        let (width, height, shape, format, levels) =
            (old.width, old.height, old.shape, old.format, old.levels);
        self.transition(key, vk::ImageLayout::GENERAL)?;
        self.order_target(key)?;
        let target = self.create_target(width, height, shape, format, levels, Scale::ONE)?;
        self.register_views(&target);
        let old = self.insert_target(key, target).unwrap();
        let filter = if is_depth(format) || integer_format(format) {
            vk::Filter::NEAREST
        } else {
            vk::Filter::LINEAR
        };
        let new = &self.targets[&key];
        let blits: Vec<vk::ImageBlit> = (0..levels)
            .map(|level| {
                let layers = vk::ImageSubresourceLayers {
                    aspect_mask: range(format).aspect_mask,
                    mip_level: level,
                    base_array_layer: 0,
                    layer_count: shape.layers(),
                };
                let corner = |(width, height): (u32, u32)| vk::Offset3D {
                    x: width as i32,
                    y: height as i32,
                    z: 1,
                };
                vk::ImageBlit {
                    src_subresource: layers,
                    src_offsets: [vk::Offset3D::default(), corner(old.extent(level))],
                    dst_subresource: layers,
                    dst_offsets: [vk::Offset3D::default(), corner(new.extent(level))],
                }
            })
            .collect();
        let source = old.image;
        let destination = new.image;
        self.slot.retired_uploads.push(old);
        self.unscaled.insert((key, description));
        self.descaled += 1;
        self.target_changes += 1;
        self.changed_targets.push(key);
        self.transition(key, vk::ImageLayout::GENERAL)?;
        let commands = self.commands()?;
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
        // SAFETY: recording; both images are live and in GENERAL layout.
        unsafe {
            let device = &self.gpu.device;
            device.cmd_blit_image(
                commands,
                source,
                vk::ImageLayout::GENERAL,
                destination,
                vk::ImageLayout::GENERAL,
                &blits,
                filter,
            );
            *self
                .stats
                .entry("stat: barriers in keep_unscaled")
                .or_default() += 1;
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
        Ok(())
    }

    fn create_target(
        &self,
        width: u32,
        height: u32,
        shape: crate::draw::ImageShape,
        format: vk::Format,
        levels: u32,
        scale: Scale,
    ) -> Result<Target, String> {
        let storage = if self.supports_storage(format) || self.storage_twin(format).is_some() {
            vk::ImageUsageFlags::STORAGE
        } else {
            vk::ImageUsageFlags::empty()
        };
        // A block-compressed target is never an attachment or a storage
        // image: draws write its blocks through an integer alias that is
        // copied in (`copy_blocks`).
        let usage = if is_depth(format) {
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
        } else if bc_block_bytes(format).is_some() {
            vk::ImageUsageFlags::empty()
        } else {
            vk::ImageUsageFlags::COLOR_ATTACHMENT | storage
        } | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST;
        let mut target = self.create_shaped_image(
            scale.apply(width),
            scale.apply(height),
            shape,
            format,
            usage,
            levels,
        )?;
        (target.width, target.height, target.scale) = (width, height, scale);
        Ok(target)
    }

    /// Keep target `from` (if any) under key `to` instead, dropping what
    /// `to` held.
    pub fn move_target(&mut self, from: u64, to: u64) -> Result<(), String> {
        self.restore_archived_target(from)?;
        let Some(target) = self.remove_target(from) else {
            return Ok(());
        };
        self.target_changes += 1;
        self.changed_targets.push(from);
        self.changed_targets.push(to);
        // What `to` held goes once the work recorded so far has run
        // (retired with this batch), not after a wait for it all.
        if let Some(replaced) = self.remove_target(to) {
            self.slot.retired_uploads.push(replaced);
        } else if self.archived_targets.remove(&to).is_some() {
            let index = self.guest_target_keys.binary_search(&to).unwrap();
            self.guest_target_keys.remove(index);
        }
        if self.pending_unscaled.remove(&to) {
            self.unscaled.insert((to, TargetDescription::from(&target)));
        }
        self.insert_target(to, target);
        let needs_initialization = self.fresh_targets.contains(&from);
        self.fresh_targets
            .retain(|fresh| *fresh != from && *fresh != to);
        if needs_initialization {
            self.fresh_targets.push(to);
        }
        Ok(())
    }

    /// The keys of targets made, remade or moved since the last call.
    pub fn take_changed_targets(&mut self) -> Vec<u64> {
        std::mem::take(&mut self.changed_targets)
    }

    /// Target changes that have not been consumed by binding invalidation.
    pub fn changed_targets(&self) -> &[u64] {
        &self.changed_targets
    }

    /// Targets created since the last call.
    pub fn take_fresh_targets(&mut self) -> Vec<u64> {
        std::mem::take(&mut self.fresh_targets)
    }

    pub fn target_needs_initialization(&self, key: u64) -> bool {
        self.fresh_targets.contains(&key)
    }

    /// A write to a fresh target leaves some of its original texels intact.
    pub fn partial_write_needs_initialization(
        &self,
        key: u64,
        level: u32,
        layer: u32,
        rect: Option<[i32; 4]>,
    ) -> bool {
        if !self.target_needs_initialization(key) {
            return false;
        }
        let Some(target) = self.targets.get(&key) else {
            return false;
        };
        self.condition.is_some()
            || level != 0
            || layer != 0
            || target.levels > 1
            || target.shape.layers() > 1
            || rect.is_some_and(|rect| {
                let (width, height) = target.extent(level);
                let whole = [0, 0, width as i32, height as i32];
                clip(target.scaled(rect), whole) != whole
            })
    }

    /// Target `key` gets its contents otherwise (a clear or a copy): it
    /// needs no initialization.
    pub fn defined(&mut self, key: u64) {
        self.cancel_target_archive(key);
        self.fresh_targets.retain(|fresh| *fresh != key);
    }

    /// Initialize target `key` from a matching upload's subresources.
    pub fn initialize_target(
        &mut self,
        key: u64,
        source: &crate::draw::TextureSource,
    ) -> Result<(), String> {
        let crate::draw::TextureSource::Upload {
            key: upload,
            width,
            height,
            format,
            shape,
            levels,
            ..
        } = source
        else {
            return Err("target initialization needs a guest upload".into());
        };
        let target = self.targets.get(&key).ok_or("no such target")?;
        if (target.width, target.height, target.format, target.shape)
            != (*width, *height, *format, *shape)
        {
            return Err("upload and target differ".into());
        }
        let subresources =
            |(region, aspect_mask): (&crate::draw::UploadRegion, _)| vk::ImageSubresourceLayers {
                aspect_mask,
                mip_level: region.level,
                base_array_layer: region.layer,
                layer_count: 1,
            };
        let scaled = target.extent(0) != (*width, *height);
        let blit_filter = if scaled {
            // SAFETY: the instance and physical device are live.
            let features = unsafe {
                self.gpu
                    .instance
                    .get_physical_device_format_properties(self.gpu.physical, *format)
            }
            .optimal_tiling_features;
            if !features
                .contains(vk::FormatFeatureFlags::BLIT_SRC | vk::FormatFeatureFlags::BLIT_DST)
            {
                return Err(format!("cannot initialize a scaled {format:?} image"));
            }
            if !is_depth(*format)
                && features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
            {
                vk::Filter::LINEAR
            } else {
                vk::Filter::NEAREST
            }
        } else {
            vk::Filter::NEAREST
        };
        self.upload(source)?;
        let source = &self.uploads[upload].1;
        let target = &self.targets[&key];
        let regions = levels
            .iter()
            .filter(|region| {
                region.level < source.levels.min(target.levels) && region.layer < shape.layers()
            })
            .flat_map(|region| {
                transfer_aspects(aspect(*format)).map(move |aspect| (region, aspect))
            });
        let mut copies = Vec::new();
        let mut blits = Vec::new();
        if scaled {
            let corner = |width: u32, height: u32, depth: u32| vk::Offset3D {
                x: width as i32,
                y: height as i32,
                z: depth as i32,
            };
            blits.extend(regions.map(|(region, aspect)| {
                let layers = subresources((region, aspect));
                let (scaled_width, scaled_height) = target.extent(region.level);
                vk::ImageBlit {
                    src_subresource: layers,
                    src_offsets: [
                        vk::Offset3D::default(),
                        corner(region.width, region.height, region.depth),
                    ],
                    dst_subresource: layers,
                    dst_offsets: [
                        vk::Offset3D::default(),
                        corner(scaled_width, scaled_height, region.depth),
                    ],
                }
            }));
        } else {
            copies.extend(regions.map(|(region, aspect)| {
                let layers = subresources((region, aspect));
                vk::ImageCopy::default()
                    .src_subresource(layers)
                    .dst_subresource(layers)
                    .extent(vk::Extent3D {
                        width: region.width,
                        height: region.height,
                        depth: region.depth,
                    })
            }));
        }
        if copies.is_empty() && blits.is_empty() {
            return Ok(());
        }
        let (from, to) = (source.image, target.image);
        self.transition(key, vk::ImageLayout::GENERAL)?;
        let commands = self.commands()?;
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
        // SAFETY: recording; both images are live and in GENERAL layout.
        unsafe {
            if scaled {
                self.gpu.device.cmd_blit_image(
                    commands,
                    from,
                    vk::ImageLayout::GENERAL,
                    to,
                    vk::ImageLayout::GENERAL,
                    &blits,
                    blit_filter,
                );
            } else {
                self.gpu.device.cmd_copy_image(
                    commands,
                    from,
                    vk::ImageLayout::GENERAL,
                    to,
                    vk::ImageLayout::GENERAL,
                    &copies,
                );
            }
            *self
                .stats
                .entry("stat: barriers in initialize_target")
                .or_default() += 1;
            self.gpu.device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
        Ok(())
    }

    /// Select one mip and the layers addressed by the storage image shader.
    pub(crate) fn storage_image_view(
        &mut self,
        image: StorageImage,
        texture_type: u32,
        format: vk::Format,
    ) -> Result<vk::ImageView, String> {
        let StorageImage {
            target: key,
            level,
            base_layer,
            layer_count,
        } = image;
        let target = self.targets.get(&key).ok_or("no such target")?;
        if level >= target.levels {
            return Err(format!("level {level} of a {}-level target", target.levels));
        }
        let total = target.shape.layers();
        let own_type = own_view_type(target.shape);
        let view_type = match (texture_type, target.shape) {
            (0, ImageShape::D1 { .. }) => vk::ImageViewType::TYPE_1D,
            (1, ImageShape::D1 { .. }) => vk::ImageViewType::TYPE_1D_ARRAY,
            (2, ImageShape::D2 { .. } | ImageShape::Cube { .. }) => vk::ImageViewType::TYPE_2D,
            (3, ImageShape::D2 { .. } | ImageShape::Cube { .. }) => {
                vk::ImageViewType::TYPE_2D_ARRAY
            }
            _ => own_type,
        };
        let (base_layer, layer_count) = if view_type == vk::ImageViewType::TYPE_3D {
            // A storage volume's z coordinate selects a slice.
            (0, 1)
        } else {
            let available = total.saturating_sub(base_layer);
            if available == 0 || layer_count > available {
                return Err(format!(
                    "storage image layer range {base_layer}+{layer_count} exceeds {total} layers"
                ));
            }
            let layer_count = match view_type {
                vk::ImageViewType::TYPE_2D_ARRAY | vk::ImageViewType::TYPE_1D_ARRAY => {
                    if layer_count == 0 {
                        available
                    } else {
                        layer_count
                    }
                }
                _ => 1,
            };
            (base_layer, layer_count)
        };
        if format == target.format
            && view_type == own_type
            && base_layer == 0
            && layer_count == total
        {
            return self.level_view(key, level);
        }
        let cache = (
            target.image,
            ExtraView::StorageImage {
                format,
                view_type,
                level,
                base_layer,
                layer_count,
            },
        );
        if let Some(view) = self.extra_views.get(&cache) {
            return Ok(*view);
        }
        let mut usage = vk::ImageViewUsageCreateInfo::default().usage(vk::ImageUsageFlags::STORAGE);
        let info = vk::ImageViewCreateInfo::default()
            .image(target.image)
            .view_type(view_type)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                base_mip_level: level,
                level_count: 1,
                base_array_layer: base_layer,
                layer_count,
                ..range(target.format)
            })
            .push_next(&mut usage);
        // SAFETY: the image is live; the view is destroyed with it.
        let view = unsafe { self.gpu.device.create_image_view(&info, None) }
            .map_err(vk_error("vkCreateImageView"))?;
        self.extra_views.insert(cache, view);
        Ok(view)
    }

    pub(crate) fn level_view(&mut self, key: u64, level: u32) -> Result<vk::ImageView, String> {
        let target = self.targets.get(&key).ok_or("no such target")?;
        if target.levels == 1 && level == 0 {
            return Ok(target.view);
        }
        if level >= target.levels {
            return Err(format!("level {level} of a {}-level target", target.levels));
        }
        let cache = (target.image, ExtraView::Level(level));
        if let Some(view) = self.extra_views.get(&cache) {
            return Ok(*view);
        }
        let mut usage = vk::ImageViewUsageCreateInfo::default().usage(target.view_usage);
        let info = vk::ImageViewCreateInfo::default()
            .image(target.image)
            .view_type(own_view_type(target.shape))
            .format(target.format)
            .subresource_range(vk::ImageSubresourceRange {
                base_mip_level: level,
                layer_count: target.shape.layers(),
                ..range(target.format)
            })
            .push_next(&mut usage);
        // SAFETY: the image is live; the view is destroyed with it.
        let view = unsafe { self.gpu.device.create_image_view(&info, None) }
            .map_err(vk_error("vkCreateImageView"))?;
        self.extra_views.insert(cache, view);
        Ok(view)
    }

    /// A view of a single mip and layer, cube face, or volume slice.
    /// An sRGB or UNORM view selects how the attachment encodes writes.
    pub(crate) fn attachment_view(
        &mut self,
        key: u64,
        level: u32,
        layer: u32,
        view_format: Option<vk::Format>,
    ) -> Result<vk::ImageView, String> {
        let target = self.targets.get(&key).ok_or("no such target")?;
        if level >= target.levels {
            return Err(format!(
                "mip level {level} of a {}-level attachment",
                target.levels
            ));
        }
        // A volume's attachment is one slice of the level (a 3D view is no
        // attachment), viewed 2D through TYPE_2D_ARRAY_COMPATIBLE.
        let layers = match target.shape {
            ImageShape::D3 { depth } => (depth >> level).max(1),
            shape => shape.layers(),
        };
        if layer >= layers {
            return Err(format!(
                "level {level} layer {layer} of a {}-level {layers}-layer target",
                target.levels
            ));
        }
        let format = attachment_format(target.format, view_format);
        if format == target.format && layers == 1 && !matches!(target.shape, ImageShape::D3 { .. })
        {
            return self.level_view(key, level);
        }
        let cache = (
            target.image,
            ExtraView::Attachment {
                format,
                level,
                layer,
            },
        );
        if let Some(view) = self.extra_views.get(&cache) {
            return Ok(*view);
        }
        let attachment = if is_depth(format) {
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
        } else {
            vk::ImageUsageFlags::COLOR_ATTACHMENT
        };
        let mut usage = vk::ImageViewUsageCreateInfo::default().usage(attachment);
        let info = vk::ImageViewCreateInfo::default()
            .image(target.image)
            .view_type(if matches!(target.shape, ImageShape::D1 { .. }) {
                vk::ImageViewType::TYPE_1D
            } else {
                vk::ImageViewType::TYPE_2D
            })
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                base_mip_level: level,
                base_array_layer: layer,
                ..range(format)
            })
            .push_next(&mut usage);
        // SAFETY: the image is live; the view is destroyed with it.
        let view = unsafe { self.gpu.device.create_image_view(&info, None) }
            .map_err(vk_error("vkCreateImageView"))?;
        self.extra_views.insert(cache, view);
        Ok(view)
    }

    #[track_caller]
    fn create_shaped_image(
        &self,
        width: u32,
        height: u32,
        shape: crate::draw::ImageShape,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        levels: u32,
    ) -> Result<Target, String> {
        use crate::draw::ImageShape;
        let largest = width.max(height).max(shape.depth()).max(1);
        let levels = levels.clamp(1, 32 - largest.leading_zeros());
        let (image_type, flags) = match shape {
            ImageShape::D1 { .. } => (vk::ImageType::TYPE_1D, vk::ImageCreateFlags::empty()),
            ImageShape::D2 { .. } => (vk::ImageType::TYPE_2D, vk::ImageCreateFlags::empty()),
            // 2D views of its slices are a draw's attachments.
            ImageShape::D3 { .. } => (
                vk::ImageType::TYPE_3D,
                vk::ImageCreateFlags::TYPE_2D_ARRAY_COMPATIBLE,
            ),
            ImageShape::Cube { .. } => (
                vk::ImageType::TYPE_2D,
                vk::ImageCreateFlags::CUBE_COMPATIBLE,
            ),
        };
        // Formats with an sRGB twin may be viewed as either (an NVN texture
        // view's format): listed, so the driver keeps its compression.
        // Storage images stay single-format: sRGB formats have no storage
        // use, which a listed view format would need.
        // An sRGB format stores through its UNORM twin (image stores do
        // no sRGB encoding): STORAGE then applies to the twin's view only,
        // which EXTENDED_USAGE allows. A UNORM storage target keeps its
        // sRGB twin the same way (rendered through an NVN RGBA8_SRGB view,
        // as the game's window textures are): STORAGE stays with its own
        // format, and twin views name their usage (`twin_view_usage`).
        let store_twin = self.storage_twin(format);
        let unorm_with_twin =
            srgb_twin(format).is_some() && store_twin.is_none() && !is_srgb(format);
        let twin = srgb_twin(format).filter(|_| {
            !usage.contains(vk::ImageUsageFlags::STORAGE) || store_twin.is_some() || unorm_with_twin
        });
        // A storage target may also be written as unsigned words of its
        // texel size (a shader's R32UI view of an R11G11B10 or RGBA8
        // target): that alias is listed too.
        let alias = uint_alias(format)
            .filter(|alias| *alias != format && usage.contains(vk::ImageUsageFlags::STORAGE));
        let mut listed = vec![format];
        listed.extend(twin);
        listed.extend(alias);
        // R8 and R16 words may also be declared signed.
        listed.extend(alias.and_then(|alias| match alias {
            vk::Format::R8_UINT => Some(vk::Format::R8_SINT),
            vk::Format::R16_UINT => Some(vk::Format::R16_SINT),
            _ => None,
        }));
        listed.retain({
            let mut seen = Vec::new();
            move |format| {
                let new = !seen.contains(format);
                seen.push(*format);
                new
            }
        });
        let mut format_list = vk::ImageFormatListCreateInfo::default().view_formats(&listed);
        let twin = twin.or(alias);
        let flags = if twin.is_some() {
            flags | vk::ImageCreateFlags::MUTABLE_FORMAT
        } else {
            flags
        };
        let flags = if (store_twin.is_some() || unorm_with_twin)
            && usage.contains(vk::ImageUsageFlags::STORAGE)
        {
            flags | vk::ImageCreateFlags::EXTENDED_USAGE
        } else {
            flags
        };
        let mut info = vk::ImageCreateInfo::default()
            .flags(flags)
            .image_type(image_type)
            .format(format)
            .extent(vk::Extent3D {
                width,
                height: if image_type == vk::ImageType::TYPE_1D {
                    1
                } else {
                    height
                },
                depth: shape.depth(),
            })
            .mip_levels(levels)
            .array_layers(shape.layers())
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let mut uncompressed =
            vk::ImageCompressionControlEXT::default().flags(vk::ImageCompressionFlagsEXT::DISABLED);
        let uncompressed_depth = self.gpu.uncompressed_depth && is_depth(format);
        if info.array_layers > 1 {
            let mut query_formats = format_list;
            let mut query_compression = uncompressed;
            let mut query = vk::PhysicalDeviceImageFormatInfo2::default()
                .format(format)
                .ty(image_type)
                .tiling(info.tiling)
                .usage(usage)
                .flags(flags);
            if twin.is_some() {
                query = query.push_next(&mut query_formats);
            }
            if uncompressed_depth {
                query = query.push_next(&mut query_compression);
            }
            let mut properties = vk::ImageFormatProperties2::default();
            // SAFETY: the instance and physical device are live.
            unsafe {
                self.gpu
                    .instance
                    .get_physical_device_image_format_properties2(
                        self.gpu.physical,
                        &query,
                        &mut properties,
                    )
            }
            .map_err(vk_error("vkGetPhysicalDeviceImageFormatProperties2"))?;
            let max_layers = properties.image_format_properties.max_array_layers;
            if info.array_layers > max_layers {
                return Err(format!(
                    "image has {} layers, but {format:?} supports at most {max_layers}",
                    info.array_layers
                ));
            }
        }
        if twin.is_some() {
            info = info.push_next(&mut format_list);
        }
        if uncompressed_depth {
            info = info.push_next(&mut uncompressed);
        }
        let device = self.device();
        // SAFETY: valid create info; destroyed by destroy_target.
        let image =
            unsafe { device.create_image(&info, None) }.map_err(vk_error("vkCreateImage"))?;
        // SAFETY: the image is live.
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let mut memory = vk::DeviceMemory::null();
        let mut view = vk::ImageView::null();
        let target = (|| {
            memory = self.allocate(requirements, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
            // SAFETY: fresh image and memory of the required size.
            unsafe { device.bind_image_memory(image, memory, 0) }
                .map_err(vk_error("vkBindImageMemory"))?;
            let view_usage = if store_twin.is_some() {
                usage & !vk::ImageUsageFlags::STORAGE
            } else {
                usage
            };
            let mut own_usage = vk::ImageViewUsageCreateInfo::default().usage(view_usage);
            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(own_view_type(shape))
                .format(format)
                .subresource_range(vk::ImageSubresourceRange {
                    level_count: levels,
                    layer_count: shape.layers(),
                    ..range(format)
                })
                .push_next(&mut own_usage);
            // SAFETY: as above.
            view = unsafe { device.create_image_view(&view_info, None) }
                .map_err(vk_error("vkCreateImageView"))?;
            let sampled_view = if has_stencil(format) {
                let depth_only = view_info.subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    ..view_info.subresource_range
                });
                // SAFETY: as above.
                unsafe { device.create_image_view(&depth_only, None) }
                    .map_err(vk_error("vkCreateImageView"))?
            } else {
                view
            };
            Ok(Target {
                image,
                view,
                sampled_view,
                shape,
                levels,
                memory,
                format,
                view_usage,
                width,
                height,
                scale: Scale::ONE,
                layout: vk::ImageLayout::UNDEFINED,
            })
        })();
        if target.is_err() {
            // SAFETY: construction did not publish any resource or record any use.
            // A resource whose creation failed still holds a null handle.
            unsafe {
                device.destroy_image_view(view, None);
                device.destroy_image(image, None);
                device.free_memory(memory, None);
            }
        }
        target
    }

    fn destroy_target(&mut self, target: Target) {
        self.view_infos.retain(|_, info| info.image != target.image);
        let mut extra = Vec::new();
        self.extra_views.retain(|(image, _), view| {
            let other = *image != target.image;
            if !other {
                extra.push(*view);
            }
            other
        });
        let mut views = extra.clone();
        views.extend([target.view, target.sampled_view]);
        self.forget_pool_views(&views);
        // SAFETY: the batch containing the target's last use completed.
        unsafe {
            for view in extra {
                self.device().destroy_image_view(view, None);
            }
            if target.sampled_view != target.view {
                self.device().destroy_image_view(target.sampled_view, None);
            }
            self.device().destroy_image_view(target.view, None);
            self.device().destroy_image(target.image, None);
            self.device().free_memory(target.memory, None);
        }
    }

    /// Record a full barrier moving `key`'s image to `layout`. An image
    /// already there records nothing and leaves an open rendering pass open
    /// (a draw sampling a target would otherwise end the pass each time).
    #[track_caller]
    pub(crate) fn transition(&mut self, key: u64, layout: vk::ImageLayout) -> Result<(), String> {
        self.cancel_target_archive(key);
        if self.targets.get(&key).ok_or("no such target")?.layout == layout {
            return Ok(());
        }
        let commands = self.commands()?;
        let target = self.targets.get_mut(&key).ok_or("no such target")?;
        let barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
            .old_layout(target.layout)
            .new_layout(layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(target.image)
            // Every level and layer: the whole image changes layout (other
            // levels and layers were left in UNDEFINED before).
            .subresource_range(vk::ImageSubresourceRange {
                level_count: vk::REMAINING_MIP_LEVELS,
                layer_count: vk::REMAINING_ARRAY_LAYERS,
                ..range(target.format)
            });
        target.layout = layout;
        // SAFETY: recording; the image is live.
        unsafe {
            *self
                .stats
                .entry("stat: barriers in transition")
                .or_default() += 1;
            self.gpu.device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            )
        };
        Ok(())
    }

    /// Whether one of these images has render or clear writes awaiting a barrier.
    pub(crate) fn rendering_pending(&self, images: impl IntoIterator<Item = vk::Image>) -> bool {
        images
            .into_iter()
            .any(|image| self.written.contains(&image))
    }

    /// Make earlier rendering visible before work that reads (or writes)
    /// `images`: one memory barrier when any of them was rendered to since
    /// the last one. Rendering passes are recorded one per draw, and
    /// nothing else orders a pass's attachment writes before a later
    /// sample, copy or read back of the same image (stale tiles otherwise).
    pub(crate) fn order_after_rendering(&mut self, images: &[vk::Image]) -> Result<(), String> {
        if !self.rendering_pending(images.iter().copied()) {
            return Ok(());
        }
        self.memory_barrier()
    }

    /// The game's nvnCommandBufferBarrier: its work so far is ordered
    /// before what follows and its writes made visible (a full barrier,
    /// ending an open pass, whatever the order and invalidate bits ask).
    /// `bits` (NVNbarrierBits) are counted per combination.
    pub fn guest_barrier(&mut self, bits: u64) -> Result<(), String> {
        static NAMES: std::sync::OnceLock<Vec<&'static str>> = std::sync::OnceLock::new();
        let names = NAMES.get_or_init(|| {
            (0..256)
                .map(|bits| &*format!("stat: guest barriers bits {bits:#04x}").leak())
                .collect()
        });
        let total = self.stats.entry("stat: guest barriers").or_default();
        *total += 1;
        // A run that hangs never reports its stats: the counts so far go
        // to the log every 20,000 barriers.
        let report = *total % 20_000 == 0;
        *self.stats.entry(names[(bits & 0xff) as usize]).or_default() += 1;
        if report {
            let mut counts: Vec<(&str, u64)> = self
                .stats
                .iter()
                .filter(|(name, _)| name.starts_with("stat: guest barriers"))
                .map(|(name, count)| (*name, *count))
                .collect();
            counts.sort_unstable();
            eprintln!("gpu: guest barriers so far {counts:?}");
        }
        self.memory_barrier()
    }

    /// A full memory barrier; everything written before is visible after.
    #[track_caller]
    pub(crate) fn memory_barrier(&mut self) -> Result<(), String> {
        if self.recording && self.full_barrier_recorded {
            // The previous barrier already orders all earlier work before
            // later work.
            self.stat("stat: redundant full barriers", 1);
            return Ok(());
        }
        let commands = self.commands()?;
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
        // SAFETY: recording outside rendering.
        unsafe {
            *self
                .stats
                .entry("stat: barriers in memory_barrier")
                .or_default() += 1;
            self.gpu.device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            )
        };
        self.full_barrier_recorded = true;
        self.written.clear();
        self.hazard_barriers += 1;
        Ok(())
    }

    /// Order a transfer reading or writing target `key` after rendering.
    fn order_target(&mut self, key: u64) -> Result<(), String> {
        let image = self.targets.get(&key).ok_or("no such target")?.image;
        self.order_after_rendering(&[image])
    }

    /// Copy level 0 of volume target `volume` (its `depth` slices) into
    /// the array layers of target `alias`, created beside it.
    fn volume_as_layers(
        &mut self,
        volume: u64,
        alias: u64,
        width: u32,
        height: u32,
        depth: u32,
        format: vk::Format,
    ) -> Result<(), String> {
        use crate::draw::ImageShape;
        self.shaped_target(
            alias,
            width,
            height,
            ImageShape::D2 { layers: depth },
            format,
            1,
        )?;
        self.transition(volume, vk::ImageLayout::GENERAL)?;
        self.transition(alias, vk::ImageLayout::GENERAL)?;
        self.order_target(volume)?;
        self.order_target(alias)?;
        let commands = self.commands()?;
        let (from, to) = (&self.targets[&volume], &self.targets[&alias]);
        let aspect = range(format).aspect_mask;
        let region = vk::ImageCopy {
            src_subresource: vk::ImageSubresourceLayers {
                aspect_mask: aspect,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            src_offset: vk::Offset3D::default(),
            dst_subresource: vk::ImageSubresourceLayers {
                aspect_mask: aspect,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: depth,
            },
            dst_offset: vk::Offset3D::default(),
            extent: vk::Extent3D {
                width: from.width.min(to.width),
                height: from.height.min(to.height),
                depth,
            },
        };
        let after = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(
                vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::TRANSFER_READ
                    | vk::AccessFlags::TRANSFER_WRITE,
            );
        // SAFETY: recording outside rendering; both images are live and in
        // GENERAL layout; a volume's depth maps to the array's layers
        // (Vulkan 1.1).
        unsafe {
            let device = &self.gpu.device;
            device.cmd_copy_image(
                commands,
                from.image,
                vk::ImageLayout::GENERAL,
                to.image,
                vk::ImageLayout::GENERAL,
                &[region],
            );
            *self
                .stats
                .entry("stat: barriers in volume_as_layers")
                .or_default() += 1;
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[after],
                &[],
                &[],
            );
        }
        Ok(())
    }

    /// Copy every level and layer (or slice) of target `from`, whose
    /// texels are the blocks of block-compressed target `to` (an RGBA16UI
    /// image of 8-byte BC4 blocks, a quarter of its width and height),
    /// into `to`: a size-compatible copy, its extent in `from`'s texels.
    pub fn copy_blocks(&mut self, from: u64, to: u64) -> Result<(), String> {
        self.transition(from, vk::ImageLayout::GENERAL)?;
        self.transition(to, vk::ImageLayout::GENERAL)?;
        self.order_target(from)?;
        self.order_target(to)?;
        let commands = self.commands()?;
        let (source, destination) = (&self.targets[&from], &self.targets[&to]);
        let levels = source.levels.min(destination.levels);
        let layers = source.shape.layers().min(destination.shape.layers());
        let regions: Vec<vk::ImageCopy> = (0..levels)
            .map(|level| {
                let subresource = |format| vk::ImageSubresourceLayers {
                    aspect_mask: range(format).aspect_mask,
                    mip_level: level,
                    base_array_layer: 0,
                    layer_count: layers,
                };
                // The destination level's blocks bound the copy (a level
                // smaller than a block still holds one).
                let blocks = |extent: u32| (extent >> level).max(1).div_ceil(4);
                vk::ImageCopy {
                    src_subresource: subresource(source.format),
                    src_offset: vk::Offset3D::default(),
                    dst_subresource: subresource(destination.format),
                    dst_offset: vk::Offset3D::default(),
                    extent: vk::Extent3D {
                        width: blocks(destination.width).min((source.width >> level).max(1)),
                        height: blocks(destination.height).min((source.height >> level).max(1)),
                        depth: (source.shape.depth() >> level)
                            .max(1)
                            .min((destination.shape.depth() >> level).max(1)),
                    },
                }
            })
            .collect();
        let after = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        // SAFETY: recording outside rendering; both images are live and in
        // GENERAL layout; an uncompressed texel the size of a compressed
        // block copies to that block (Vulkan 1.0 size-compatible copies).
        unsafe {
            let device = &self.gpu.device;
            device.cmd_copy_image(
                commands,
                source.image,
                vk::ImageLayout::GENERAL,
                destination.image,
                vk::ImageLayout::GENERAL,
                &regions,
            );
            *self
                .stats
                .entry("stat: barriers in copy_blocks")
                .or_default() += 1;
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[after],
                &[],
                &[],
            );
        }
        Ok(())
    }

    /// Copy `source` rect (x, y, width, height) of target `from` into
    /// `destination` rect of target `to`, scaling (linear filter) when the
    /// sizes differ. Rects are clipped to their targets.
    pub fn copy_target(
        &mut self,
        from: u64,
        source: [i32; 4],
        to: u64,
        destination: [i32; 4],
    ) -> Result<(), String> {
        self.copy_target_level((from, 0, 0), source, (to, 0, 0), destination)
    }

    /// Copy between selected mip levels and layers, with rectangles in
    /// their guest texel grids.
    pub fn copy_target_level(
        &mut self,
        (from, source_level, source_layer): (u64, u32, u32),
        source: [i32; 4],
        (to, destination_level, destination_layer): (u64, u32, u32),
        destination: [i32; 4],
    ) -> Result<(), String> {
        if (from, source_level, source_layer) == (to, destination_level, destination_layer) {
            return Err("copy within one render target subresource".into());
        }
        let bounds = |key, level, layer| -> Result<[i32; 4], String> {
            let target = self.targets.get(&key).ok_or("no such copy target")?;
            if level >= target.levels || layer >= target.shape.layers() {
                return Err("texture copy layer or mip level is outside the render target".into());
            }
            let (width, height) = target.extent(level);
            Ok([0, 0, width as i32, height as i32])
        };
        let source_bounds = bounds(from, source_level, source_layer)?;
        let destination_bounds = bounds(to, destination_level, destination_layer)?;
        let source = clip(self.targets[&from].scaled(source), source_bounds);
        let destination = clip(self.targets[&to].scaled(destination), destination_bounds);
        if source[2] <= 0 || source[3] <= 0 || destination[2] <= 0 || destination[3] <= 0 {
            return Ok(());
        }
        self.transition(from, vk::ImageLayout::GENERAL)?;
        self.transition(to, vk::ImageLayout::GENERAL)?;
        let commands = self.commands()?;
        let (source_target, destination_target) = (&self.targets[&from], &self.targets[&to]);
        if (matches!(source_target.shape, ImageShape::D3 { .. })
            || matches!(destination_target.shape, ImageShape::D3 { .. }))
            && (source_layer != 0 || destination_layer != 0)
        {
            return Err("texture copy between an array layer and volume".into());
        }
        if aspect(source_target.format) != aspect(destination_target.format)
            || (is_depth(source_target.format) && source_target.format != destination_target.format)
        {
            return Err("texture copy between incompatible depth or stencil formats".into());
        }
        let corners = |[x, y, width, height]: [i32; 4]| {
            [
                vk::Offset3D { x, y, z: 0 },
                vk::Offset3D {
                    x: x + width,
                    y: y + height,
                    z: 1,
                },
            ]
        };
        let subresource = |aspect_mask, mip_level, base_array_layer| vk::ImageSubresourceLayers {
            aspect_mask,
            mip_level,
            base_array_layer,
            layer_count: 1,
        };
        let regions: Vec<_> = transfer_aspects(aspect(source_target.format))
            .map(|aspect_mask| {
                vk::ImageBlit::default()
                    .src_subresource(subresource(aspect_mask, source_level, source_layer))
                    .src_offsets(corners(source))
                    .dst_subresource(subresource(
                        aspect_mask,
                        destination_level,
                        destination_layer,
                    ))
                    .dst_offsets(corners(destination))
            })
            .collect();
        let filter = if is_depth(source_target.format)
            || integer_format(source_target.format)
            || source[2..] == destination[2..]
        {
            vk::Filter::NEAREST
        } else {
            vk::Filter::LINEAR
        };
        let barrier = |target: &Target, mip_level, layer, src, dst| {
            vk::ImageMemoryBarrier::default()
                .src_access_mask(src)
                .dst_access_mask(dst)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(target.image)
                .subresource_range(vk::ImageSubresourceRange {
                    base_mip_level: mip_level,
                    base_array_layer: layer,
                    ..range(target.format)
                })
        };
        let read_write = vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE;
        let before = [
            barrier(
                source_target,
                source_level,
                source_layer,
                vk::AccessFlags::MEMORY_WRITE,
                vk::AccessFlags::TRANSFER_READ,
            ),
            barrier(
                destination_target,
                destination_level,
                destination_layer,
                read_write,
                vk::AccessFlags::TRANSFER_WRITE,
            ),
        ];
        let after = barrier(
            destination_target,
            destination_level,
            destination_layer,
            vk::AccessFlags::TRANSFER_WRITE,
            read_write,
        );
        // SAFETY: recording; both images are live and in GENERAL layout.
        unsafe {
            let device = &self.gpu.device;
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &before,
            );
            device.cmd_blit_image(
                commands,
                source_target.image,
                vk::ImageLayout::GENERAL,
                destination_target.image,
                vk::ImageLayout::GENERAL,
                &regions,
                filter,
            );
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[after],
            );
        };
        self.stat("stat: barriers in copy_target", 2);
        self.defined(to);
        Ok(())
    }

    /// Clear `rect` (x, y, width, height; the whole target when None) of a
    /// color target to `color`.
    pub fn clear_color(
        &mut self,
        key: u64,
        rect: Option<[i32; 4]>,
        color: vk::ClearColorValue,
    ) -> Result<(), String> {
        self.clear_color_level(key, 0, 0, rect, color)
    }

    /// Clear `rect` of mip level `level` of array layer `layer` of a color
    /// target (see `clear_color`).
    pub fn clear_color_level(
        &mut self,
        key: u64,
        level: u32,
        layer: u32,
        rect: Option<[i32; 4]>,
        color: vk::ClearColorValue,
    ) -> Result<(), String> {
        self.clear_color_view(
            (key, level, layer),
            None,
            rect,
            color,
            vk::ColorComponentFlags::RGBA,
        )
    }

    /// Clear selected channels of a color attachment, using its view format.
    pub fn clear_color_view(
        &mut self,
        (key, level, layer): (u64, u32, u32),
        view_format: Option<vk::Format>,
        rect: Option<[i32; 4]>,
        color: vk::ClearColorValue,
        mask: vk::ColorComponentFlags,
    ) -> Result<(), String> {
        if mask.is_empty() {
            return Ok(());
        }
        let target = self.targets.get(&key).ok_or("no such target")?;
        if level >= target.levels || layer >= target.shape.layers() {
            return Err(format!(
                "level {level} layer {layer} of a {}-level {}-layer target",
                target.levels,
                target.shape.layers()
            ));
        }
        let (whole_width, whole_height) = target.extent(level);
        let format = attachment_format(target.format, view_format);
        let reinterpret = format != target.format;
        let whole = [0, 0, whole_width as i32, whole_height as i32];
        let rect = rect.map(|rect| target.scaled(rect));
        let [x, y, width, height] = clip(rect.unwrap_or(whole), whole);
        if [x, y, width, height] != whole {
            self.partial_clears += 1;
        }
        if width <= 0 || height <= 0 {
            return Ok(());
        }
        if mask != vk::ColorComponentFlags::RGBA {
            self.stat("stat: masked color clears", 1);
            // SAFETY: the union contains four initialized 32-bit words.
            let words = unsafe { color.uint32 };
            return self.clear_with_shader(
                (key, level, layer),
                (whole_width, whole_height),
                [x, y, width, height],
                crate::draw::ShaderClear::Color {
                    format,
                    words,
                    mask,
                },
            );
        }
        if [x, y, width, height] == whole && self.condition.is_none() && !reinterpret {
            self.transition(key, vk::ImageLayout::GENERAL)?;
            self.order_target(key)?;
            let commands = self.commands()?;
            let target = &self.targets[&key];
            // SAFETY: recording; the image is in GENERAL layout.
            unsafe {
                self.gpu.device.cmd_clear_color_image(
                    commands,
                    target.image,
                    vk::ImageLayout::GENERAL,
                    &color,
                    &[vk::ImageSubresourceRange {
                        base_mip_level: level,
                        base_array_layer: layer,
                        ..range(target.format)
                    }],
                )
            };
            self.written.insert(target.image);
            return Ok(());
        }
        // Attachment clears support scissoring and conditional rendering.
        self.prepare_condition()?;
        self.begin_rendering(&[key], &[level], &[layer], &[view_format], None, [0; 4])?;
        self.apply_condition()?;
        let commands = self.recording_commands()?;
        let attachment = vk::ClearAttachment {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            color_attachment: 0,
            clear_value: vk::ClearValue { color },
        };
        let rect = vk::ClearRect {
            rect: vk::Rect2D {
                offset: vk::Offset2D { x, y },
                extent: vk::Extent2D {
                    width: width as u32,
                    height: height as u32,
                },
            },
            base_array_layer: 0,
            layer_count: 1,
        };
        // SAFETY: inside the rendering begun above.
        unsafe {
            self.gpu
                .device
                .cmd_clear_attachments(commands, &[attachment], &[rect]);
        }
        Ok(())
    }

    /// Clear the depth and/or stencil of array layer `layer` of a depth
    /// target within `rect`.
    pub fn clear_depth_stencil(
        &mut self,
        key: u64,
        layer: u32,
        rect: Option<[i32; 4]>,
        depth: Option<f32>,
        stencil: Option<u32>,
    ) -> Result<(), String> {
        self.clear_depth_stencil_level(key, (0, layer), rect, depth, stencil)
    }

    /// Clear depth and/or stencil of the selected mip and array layer.
    pub fn clear_depth_stencil_level(
        &mut self,
        key: u64,
        (level, layer): (u32, u32),
        rect: Option<[i32; 4]>,
        depth: Option<f32>,
        stencil: Option<u32>,
    ) -> Result<(), String> {
        let target = self.targets.get(&key).ok_or("no such target")?;
        if level >= target.levels {
            return Err(format!(
                "mip level {level} of a {}-level depth target",
                target.levels
            ));
        }
        let format = target.format;
        let mut aspects = vk::ImageAspectFlags::empty();
        if depth.is_some() {
            aspects |= vk::ImageAspectFlags::DEPTH;
        }
        if stencil.is_some() && has_stencil(format) {
            aspects |= vk::ImageAspectFlags::STENCIL;
        }
        if aspects.is_empty() {
            return Ok(());
        }
        let (whole_width, whole_height) = target.extent(level);
        let whole = [0, 0, whole_width as i32, whole_height as i32];
        let rect = rect.map(|rect| target.scaled(rect));
        let [x, y, width, height] = clip(rect.unwrap_or(whole), whole);
        if width <= 0 || height <= 0 {
            return Ok(());
        }
        self.prepare_condition()?;
        self.begin_rendering(&[], &[], &[], &[], Some((key, level, layer)), [0; 4])?;
        self.apply_condition()?;
        let commands = self.recording_commands()?;
        let attachment = vk::ClearAttachment {
            aspect_mask: aspects,
            color_attachment: 0,
            clear_value: vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: depth.unwrap_or(0.0),
                    stencil: stencil.unwrap_or(0),
                },
            },
        };
        let rect = vk::ClearRect {
            rect: vk::Rect2D {
                offset: vk::Offset2D { x, y },
                extent: vk::Extent2D {
                    width: width as u32,
                    height: height as u32,
                },
            },
            base_array_layer: 0,
            layer_count: 1,
        };
        // SAFETY: inside the rendering begun above.
        unsafe {
            self.gpu
                .device
                .cmd_clear_attachments(commands, &[attachment], &[rect]);
        }
        Ok(())
    }

    /// Begin dynamic rendering into color targets `colors` and an optional
    /// depth target, limited to `area`. A color key of 0 is a gap: no
    /// attachment at that location (a fragment output there is dropped).
    /// Each color target renders into its mip level in `levels` and its
    /// layer in `layers` (0 when missing), as the format in `formats`
    /// (its own when missing). The depth binding gives its key, mip and layer.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn begin_rendering(
        &mut self,
        colors: &[u64],
        levels: &[u32],
        layers: &[u32],
        formats: &[Option<vk::Format>],
        depth: Option<(u64, u32, u32)>,
        area: [i32; 4],
    ) -> Result<(), String> {
        let level = |index: usize| levels.get(index).copied().unwrap_or(0);
        let layer = |index: usize| layers.get(index).copied().unwrap_or(0);
        let format = |index: usize| formats.get(index).copied().flatten();
        let color_bindings = colors.iter().enumerate().map(|(index, &target)| {
            (target != 0).then(|| ColorAttachment {
                target,
                level: level(index),
                layer: layer(index),
                format: format(index),
            })
        });
        if self.slot.open_pass.as_ref().is_some_and(|pass| {
            pass.target_changes == self.target_changes
                && pass.depth == depth
                && pass
                    .requested_area
                    .is_none_or(|requested| requested == area)
                && pass.colors.len() == colors.len()
                && pass.colors.iter().copied().eq(color_bindings.clone())
        }) {
            return Ok(());
        }
        let mut views = Vec::with_capacity(colors.len());
        for binding in color_bindings.clone() {
            let Some(ColorAttachment {
                target,
                level,
                layer,
                format,
            }) = binding
            else {
                views.push(vk::ImageView::null());
                continue;
            };
            let view = self.attachment_view(target, level, layer, format)?;
            views.push(view);
        }
        let depth_view = match depth {
            Some((key, level, layer)) => Some(self.attachment_view(key, level, layer, None)?),
            None => None,
        };
        // The render area is the attachments' common extent (the
        // scissor limits each draw, a clear its rectangle), so draws with
        // other scissors continue one pass.
        let extent = |key: &u64, level: u32| {
            let (width, height) = self.targets[key].extent(level);
            (width as i32, height as i32)
        };
        let (smallest_width, smallest_height) = colors
            .iter()
            .enumerate()
            .filter(|(_, key)| **key != 0)
            .map(|(index, key)| extent(key, level(index)))
            .chain(depth.iter().map(|(key, level, _)| extent(key, *level)))
            .fold((i32::MAX, i32::MAX), |(w, h), (tw, th)| {
                (w.min(tw), h.min(th))
            });
        // A pass with no attachments renders over `area` (the draw's
        // scissor): only its vertex stage's stores have effect.
        let requested_area = (smallest_width == i32::MAX).then_some(area);
        let whole = requested_area.unwrap_or([0, 0, smallest_width, smallest_height]);
        if let Some(pass) = &mut self.slot.open_pass
            && pass.views == views
            && pass.depth_view == depth_view
            && pass.area == whole
        {
            pass.target_changes = self.target_changes;
            pass.colors.clear();
            pass.colors.extend(color_bindings);
            pass.depth = depth;
            pass.requested_area = requested_area;
            return Ok(());
        }
        self.end_pass();
        let present: Vec<u64> = colors
            .iter()
            .copied()
            .filter(|key| *key != 0)
            .chain(depth.map(|(key, _, _)| key))
            .collect();
        for key in &present {
            self.transition(*key, vk::ImageLayout::GENERAL)?;
        }
        // Separate rendering passes need a memory dependency before
        // loading attachments that an earlier pass wrote.
        let attachments: Vec<vk::Image> =
            present.iter().map(|key| self.targets[key].image).collect();
        self.order_after_rendering(&attachments)?;
        self.written.extend(attachments.iter().copied());
        let commands = self.commands()?;
        let attachments: Vec<_> = views
            .iter()
            .map(|view| {
                vk::RenderingAttachmentInfo::default()
                    .image_view(*view)
                    .image_layout(vk::ImageLayout::GENERAL)
                    .load_op(vk::AttachmentLoadOp::LOAD)
                    .store_op(vk::AttachmentStoreOp::STORE)
            })
            .collect();
        let depth_attachment = depth_view.map(|view| {
            vk::RenderingAttachmentInfo::default()
                .image_view(view)
                .image_layout(vk::ImageLayout::GENERAL)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE)
        });
        // The render area may not exceed any attachment (the guest binds
        // targets of different sizes together; it renders within each).
        let [x, y, width, height] = whole;
        let mut info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x, y },
                extent: vk::Extent2D {
                    width: width.max(0) as u32,
                    height: height.max(0) as u32,
                },
            })
            .layer_count(1)
            .color_attachments(&attachments);
        if let Some(depth_attachment) = depth_attachment.as_ref() {
            info = info.depth_attachment(depth_attachment);
            if depth.is_some_and(|(key, _, _)| has_stencil(self.targets[&key].format)) {
                info = info.stencil_attachment(depth_attachment);
            }
        }
        // SAFETY: recording; attachments are live and in GENERAL layout.
        *self.stats.entry("stat: render passes begun").or_default() += 1;
        let counting = self.occlusion_pass_begins(commands);
        // SAFETY: recording; attachments are live and in GENERAL layout.
        unsafe { self.gpu.device.cmd_begin_rendering(commands, &info) };
        if counting {
            self.occlusion_pass_begun(commands);
        }
        self.slot.open_pass = Some(OpenPass {
            target_changes: self.target_changes,
            colors: color_bindings.collect(),
            depth,
            requested_area,
            views,
            depth_view,
            area: whole,
        });
        Ok(())
    }

    /// Read a color target back as RGBA f32 texels (row-major).
    /// Record a scaled blit of color target `key` into a swapchain image,
    /// leaving that image ready to present.
    pub(crate) fn blit_to_swapchain(
        &mut self,
        key: u64,
        image: vk::Image,
        (width, height): (u32, u32),
        extent: vk::Extent2D,
    ) -> Result<(), String> {
        self.transition(key, vk::ImageLayout::GENERAL)?;
        let commands = self.commands()?;
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let to_transfer = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(image)
            .subresource_range(range);
        let to_present = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::empty())
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .image(image)
            .subresource_range(range);
        let layers = vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };
        // By default the frame keeps its aspect ratio: scaled to fit and
        // centred, with black bars filling the rest of the window
        // (ECHORA_SCALING_MODE chooses another fit).
        let (mode, filter) = *output_options();
        let fitted = present_rect(mode, (width, height), (extent.width, extent.height));
        let blit = vk::ImageBlit {
            src_subresource: layers,
            // Targets keep the console's row order: row 0 is the top.
            src_offsets: [
                vk::Offset3D::default(),
                vk::Offset3D {
                    x: width as i32,
                    y: height as i32,
                    z: 1,
                },
            ],
            dst_subresource: layers,
            dst_offsets: [
                vk::Offset3D {
                    x: fitted[0],
                    y: fitted[1],
                    z: 0,
                },
                vk::Offset3D {
                    x: fitted[2],
                    y: fitted[3],
                    z: 1,
                },
            ],
        };
        let letterboxed = fitted != [0, 0, extent.width as i32, extent.height as i32];
        let source = self.targets[&key].image;
        // SAFETY: recording; the target is in GENERAL layout and the
        // swapchain image was acquired.
        unsafe {
            let device = &self.gpu.device;
            *self
                .stats
                .entry("stat: barriers in blit_to_swapchain")
                .or_default() += 1;
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)],
                &[],
                &[to_transfer],
            );
            if letterboxed {
                device.cmd_clear_color_image(
                    commands,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                    &[range],
                );
                *self
                    .stats
                    .entry("stat: barriers in blit_to_swapchain")
                    .or_default() += 1;
                device.cmd_pipeline_barrier(
                    commands,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)],
                    &[],
                    &[],
                );
            }
            device.cmd_blit_image(
                commands,
                source,
                vk::ImageLayout::GENERAL,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[blit],
                filter,
            );
            *self
                .stats
                .entry("stat: barriers in blit_to_swapchain")
                .or_default() += 1;
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_present],
            );
        }
        Ok(())
    }

    pub fn read_target(&mut self, key: u64) -> Result<Vec<[f32; 4]>, String> {
        self.read_target_level(key, 0)
    }

    /// Release the upload kept for `key` (and its one-dimensional alias):
    /// the texture it held was finalized.
    pub fn forget_upload(&mut self, key: u64) {
        for key in [key, key ^ 1 << 63] {
            if let Some((_, target)) = self.uploads.remove(&key) {
                self.slot.retired_uploads.push(target);
            }
            self.upload_lengths.remove(&key);
        }
    }

    /// Debugging: a zeroed host-visible buffer of `size` bytes that shaders
    /// can write through its device address; returns its index (for
    /// `read_debug_buffer`) and that address.
    pub fn debug_buffer(&mut self, size: u64) -> Result<(usize, u64), String> {
        let buffer = self.host_buffer_with(size, vk::BufferUsageFlags::STORAGE_BUFFER, true)?;
        buffer.write(0, &vec![0; size as usize]);
        let address = self
            .buffer_address(buffer.buffer)
            .ok_or("debug buffers need buffer device addresses")?;
        self.debug_buffers.push(buffer);
        Ok((self.debug_buffers.len() - 1, address))
    }

    /// Debugging: a `debug_buffer`'s bytes as they are now.
    /// Name debug buffer `index` as a loop trace of `program`.
    pub fn mark_loop_trace(&mut self, index: usize, program: String) {
        self.loop_trace_buffers.push((index, program));
    }

    /// Each loop trace's loops with invocations the budget stopped, and
    /// each loop's most iterations (when a wait stalls or fails: the trace
    /// otherwise reports every 60 presents, after a hang never).
    fn report_loop_traces(&self, why: &str) {
        for (index, program) in &self.loop_trace_buffers {
            let Some(buffer) = self.debug_buffers.get(*index) else {
                continue;
            };
            let words: Vec<u32> = buffer
                .read_bytes(64 << 10)
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| u32::from_le_bytes(*word))
                .collect();
            for (number, entry) in words.chunks_exact(4).enumerate() {
                if entry[0] != 0 && (entry[2] != 0 || entry[1] >= 256) {
                    eprintln!(
                        "loop trace ({why}) {program} loop {number} at {:#x}: {} iterations, at most {} per invocation, {} stopped by the budget",
                        entry[3].wrapping_sub(1),
                        entry[0],
                        entry[1],
                        entry[2]
                    );
                }
            }
        }
    }

    pub fn read_debug_buffer(&self, index: usize) -> Vec<u8> {
        self.debug_buffers
            .get(index)
            .map_or_else(Vec::new, |buffer| buffer.read_bytes(buffer.size as usize))
    }

    /// Make sure a guest texture is uploaded (debugging reads).
    pub fn texture_view_for(&mut self, source: &crate::draw::TextureSource) -> Result<(), String> {
        match source {
            crate::draw::TextureSource::Upload { .. } => self.upload(source),
            _ => self.texture_view(source, 2).map(|_| ()),
        }
    }

    /// Debugging: read mip level `level` of upload `key` back as RGBA f32
    /// texels (as a target would be).
    pub fn read_upload_level(&mut self, key: u64, level: u32) -> Result<Vec<[f32; 4]>, String> {
        self.read_upload_slice(key, level, 0)
    }

    /// Debugging: as `read_upload_level`, slice `slice` of a 3D upload.
    pub fn read_upload_slice(
        &mut self,
        key: u64,
        level: u32,
        slice: u32,
    ) -> Result<Vec<[f32; 4]>, String> {
        let (generation, target) = self.uploads.remove(&key).ok_or("no such upload")?;
        let temporary = u64::MAX - 1;
        self.insert_target(temporary, target);
        let texels = self
            .read_level(temporary, level, false, slice)
            .map(|(texels, ..)| texels);
        let target = self.remove_target(temporary).unwrap();
        self.uploads.insert(key, (generation, target));
        texels
    }

    /// Read mip level `level` of a color target back as RGBA f32 texels
    /// (at the guest's resolution).
    pub fn read_target_level(&mut self, key: u64, level: u32) -> Result<Vec<[f32; 4]>, String> {
        self.read_level(key, level, false, 0)
            .map(|(texels, ..)| texels)
    }

    /// As `read_target_level`, slice `slice` of a 3D target (array layer
    /// `slice` of any other).
    pub fn read_target_slice(
        &mut self,
        key: u64,
        level: u32,
        slice: u32,
    ) -> Result<Vec<[f32; 4]>, String> {
        self.read_level(key, level, false, slice)
            .map(|(texels, ..)| texels)
    }

    /// Level 0 of a color target at the image's own (scaled) resolution,
    /// with that width and height: the frame as the window shows it.
    pub fn read_target_scaled(&mut self, key: u64) -> Result<(Vec<[f32; 4]>, u32, u32), String> {
        self.read_level(key, 0, true, 0)
    }

    /// Level `level` (slice `slice` of a 3D target) as RGBA f32 texels.
    fn read_level(
        &mut self,
        key: u64,
        level: u32,
        full: bool,
        slice: u32,
    ) -> Result<(Vec<[f32; 4]>, u32, u32), String> {
        let (width, height, format, scaled) = {
            let target = self.targets.get(&key).ok_or("no such target")?;
            if level >= target.levels {
                return Err(format!("level {level} of a {}-level target", target.levels));
            }
            let scaled = target.extent(level);
            let (width, height) = if full {
                scaled
            } else {
                (
                    (target.width >> level).max(1),
                    (target.height >> level).max(1),
                )
            };
            (width, height, target.format, scaled)
        };
        // Depth targets cannot be blitted to a color image: their depth
        // aspect is copied as it is and read as (depth, 0, 0, 1).
        if is_depth(format) {
            if full && !self.targets[&key].scale.is_one() {
                return Err("depth readback uses the guest resolution".into());
            }
            let bytes =
                self.read_target_plane_bytes(key, level, slice, 4, [0, 0, width, height])?;
            let depth = |bytes: &[u8]| match format {
                vk::Format::D16_UNORM => {
                    f32::from(u16::from_le_bytes([bytes[0], bytes[1]])) / 65535.0
                }
                vk::Format::D24_UNORM_S8_UINT | vk::Format::X8_D24_UNORM_PACK32 => {
                    let word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                    (word & 0xff_ffff) as f32 / 16_777_215.0
                }
                _ => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            };
            let size = if format == vk::Format::D16_UNORM {
                2
            } else {
                4
            };
            let texels = bytes
                .chunks_exact(size)
                .map(|texel| [depth(texel), 0.0, 0.0, 1.0])
                .collect();
            return Ok((texels, width, height));
        }
        // Blit to an RGBA32F image so every color format reads back alike.
        let staging_key = u64::MAX;
        self.target(staging_key, width, height, vk::Format::R32G32B32A32_SFLOAT)?;
        self.transition(key, vk::ImageLayout::GENERAL)?;
        self.order_target(key)?;
        self.transition(staging_key, vk::ImageLayout::GENERAL)?;
        let size = u64::from(width) * u64::from(height) * 16;
        let buffer = self.host_buffer(size, vk::BufferUsageFlags::TRANSFER_DST)?;
        let commands = self.commands()?;
        // `slice` is a depth slice of a volume, else an array layer.
        let volume = matches!(self.targets[&key].shape, crate::draw::ImageShape::D3 { .. });
        let (slice, layer) = if volume { (slice, 0) } else { (0, slice) };
        let layers = vk::ImageSubresourceLayers {
            aspect_mask: aspect(format),
            mip_level: level,
            base_array_layer: layer,
            layer_count: 1,
        };
        let corner = vk::Offset3D {
            x: width as i32,
            y: height as i32,
            z: 1,
        };
        let blit = vk::ImageBlit {
            src_subresource: layers,
            src_offsets: [
                vk::Offset3D {
                    x: 0,
                    y: 0,
                    z: slice as i32,
                },
                vk::Offset3D {
                    x: scaled.0 as i32,
                    y: scaled.1 as i32,
                    z: slice as i32 + 1,
                },
            ],
            dst_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                ..layers
            },
            dst_offsets: [vk::Offset3D::default(), corner],
        };
        let copy = vk::BufferImageCopy {
            image_subresource: blit.dst_subresource,
            image_extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            ..Default::default()
        };
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
        let (source, staging) = (self.targets[&key].image, self.targets[&staging_key].image);
        // SAFETY: recording; images in GENERAL layout; the buffer fits.
        unsafe {
            let device = &self.gpu.device;
            device.cmd_blit_image(
                commands,
                source,
                vk::ImageLayout::GENERAL,
                staging,
                vk::ImageLayout::GENERAL,
                &[blit],
                vk::Filter::NEAREST,
            );
            *self
                .stats
                .entry("stat: barriers in read_level")
                .or_default() += 1;
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
            device.cmd_copy_image_to_buffer(
                commands,
                staging,
                vk::ImageLayout::GENERAL,
                buffer.buffer,
                &[copy],
            );
        }
        self.flush()?;
        let texels = buffer.read_f32x4((width * height) as usize);
        buffer.destroy(self.device());
        Ok((texels, width, height))
    }

    /// Readback layout for a single-level, single-layer target in its
    /// real host extent and format, with no resampling or conversion.
    fn archive_copy(&self, key: u64) -> Result<(vk::BufferImageCopy, u64), String> {
        let target = self.targets.get(&key).ok_or("no such target")?;
        if target.shape != (ImageShape::D2 { layers: 1 }) || target.levels != 1 {
            return Err("target has multiple subresources".into());
        }
        let texel_bytes = match target.format {
            vk::Format::D16_UNORM => 2,
            vk::Format::D32_SFLOAT => 4,
            format if !is_depth(format) => format_bytes(format),
            _ => 0,
        };
        if !texel_bytes.is_power_of_two() {
            return Err("target format needs a separate archive layout".into());
        }
        let (width, height) = target.extent(0);
        let size = u64::from(width) * u64::from(height) * u64::from(texel_bytes);
        let copy = vk::BufferImageCopy::default()
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(aspect(target.format))
                    .layer_count(1),
            )
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            });
        Ok((copy, size))
    }

    pub fn queue_target_archive_read(&mut self, key: u64) -> Result<u64, String> {
        let (copy, size) = self.archive_copy(key)?;
        let (buffer, offset) = self.ring_alloc(size, 16)?;
        self.transition(key, vk::ImageLayout::GENERAL)?;
        self.order_target(key)?;
        let image = self.targets[&key].image;
        self.end_pass();
        self.end_condition();
        let commands = self.commands()?;
        // SAFETY: recording outside rendering. Both resources are live,
        // and the ring has room for the whole plane.
        unsafe {
            self.gpu.device.cmd_copy_image_to_buffer(
                commands,
                image,
                vk::ImageLayout::GENERAL,
                buffer,
                &[copy.buffer_offset(offset)],
            );
            // Later commands can reuse this image before the copy returns.
            // Order their writes after the transfer has read the snapshot.
            self.gpu.device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[],
            );
        }
        Ok(self.defer_ring_read(offset, size as usize))
    }

    pub fn restore_target_archive(&mut self, key: u64, bytes: &[u8]) -> Result<(), String> {
        let (copy, size) = self.archive_copy(key)?;
        if bytes.len() as u64 != size {
            return Err("archived texel count does not match the target".into());
        }
        let (buffer, offset) = self.ring_upload(bytes, 16)?;
        self.transition(key, vk::ImageLayout::GENERAL)?;
        self.order_target(key)?;
        let image = self.targets[&key].image;
        let commands = self.commands()?;
        // SAFETY: recording outside rendering. Both resources are live.
        // The upload contains the entire plane in the image's format.
        unsafe {
            self.gpu.device.cmd_copy_buffer_to_image(
                commands,
                buffer,
                image,
                vk::ImageLayout::GENERAL,
                &[copy.buffer_offset(offset)],
            );
        }
        self.written.insert(image);
        self.defined(key);
        Ok(())
    }

    /// Level 0 of a color target as its own format's bytes, rows packed:
    /// `texel_bytes` is that format's texel size, which the caller knows.
    pub fn read_target_bytes(&mut self, key: u64, texel_bytes: u32) -> Result<Vec<u8>, String> {
        self.read_target_region_bytes(key, texel_bytes, None)
    }

    /// `read_target_bytes` of the rectangle `rect` (x, y, width, height in
    /// texels, clipped to the target) only, row after row: a few texels of
    /// a large target without its whole readback.
    pub fn read_target_region_bytes(
        &mut self,
        key: u64,
        texel_bytes: u32,
        rect: Option<[u32; 4]>,
    ) -> Result<Vec<u8>, String> {
        let target = self.targets.get(&key).ok_or("no such target")?;
        let (full_width, full_height) = (target.width, target.height);
        let [x, y, width, height] = rect.unwrap_or([0, 0, full_width, full_height]);
        let (x, y) = (x.min(full_width), y.min(full_height));
        let (width, height) = (width.min(full_width - x), height.min(full_height - y));
        self.read_target_plane_bytes(key, 0, 0, texel_bytes, [x, y, width, height])
    }

    /// A rectangle of a mip level's array layer or 3D slice, in the
    /// guest's texel grid, with no buffer padding between rows.
    pub fn read_target_plane_bytes(
        &mut self,
        key: u64,
        level: u32,
        plane: u32,
        texel_bytes: u32,
        rect: [u32; 4],
    ) -> Result<Vec<u8>, String> {
        // Depth targets: their depth aspect (2 bytes per texel for D16, 4
        // for the others: a float for the D32 formats, the depth in the
        // low 24 bits for D24).
        let format = self.targets.get(&key).ok_or("no such target")?.format;
        let (aspect_mask, texel_bytes) = if is_depth(format) {
            let bytes = if format == vk::Format::D16_UNORM {
                2
            } else {
                4
            };
            (vk::ImageAspectFlags::DEPTH, bytes)
        } else {
            (vk::ImageAspectFlags::COLOR, texel_bytes)
        };
        self.read_target_aspect_bytes(key, (level, plane), rect, (aspect_mask, texel_bytes))
    }

    /// Read a depth and stencil plane in one transfer. Four-byte depth
    /// texels are followed by one stencil byte for each texel.
    pub fn read_target_depth_stencil_plane_bytes(
        &mut self,
        key: u64,
        level: u32,
        plane: u32,
        rect: [u32; 4],
    ) -> Result<Vec<u8>, String> {
        let format = self.targets.get(&key).ok_or("no such target")?.format;
        if !has_stencil(format) {
            return Err("depth-stencil readback requires both aspects".into());
        }
        self.read_target_aspect_bytes(key, (level, plane), rect, (aspect(format), 5))
    }

    fn read_target_aspect_bytes(
        &mut self,
        key: u64,
        (level, plane): (u32, u32),
        [x, y, width, height]: [u32; 4],
        (aspect_mask, texel_bytes): (vk::ImageAspectFlags, u32),
    ) -> Result<Vec<u8>, String> {
        let target = self.targets.get(&key).ok_or("no such target")?;
        if level >= target.levels
            || x.checked_add(width)
                .is_none_or(|end| end > (target.width >> level).max(1))
            || y.checked_add(height)
                .is_none_or(|end| end > (target.height >> level).max(1))
        {
            return Err("target copy rectangle is outside the mip level".into());
        }
        let (slice, layer, planes) = if matches!(target.shape, ImageShape::D3 { .. }) {
            (plane, 0, (target.shape.depth() >> level).max(1))
        } else {
            (0, plane, target.shape.layers())
        };
        if plane >= planes {
            return Err("target copy layer or slice is outside the mip level".into());
        }
        if width == 0 || height == 0 {
            return Ok(Vec::new());
        }
        let (key, level, layer, slice) = if target.scale.is_one() {
            (key, level, layer, slice)
        } else {
            (
                self.unscaled_copy(key, level, layer, slice, aspect_mask)?,
                0,
                0,
                0,
            )
        };
        let image = self.targets[&key].image;
        self.transition(key, vk::ImageLayout::GENERAL)?;
        self.order_target(key)?;
        let size = u64::from(width) * u64::from(height) * u64::from(texel_bytes);
        let buffer = self.host_buffer(size, vk::BufferUsageFlags::TRANSFER_DST)?;
        let commands = self.commands()?;
        let copies: Vec<_> = transfer_aspects(aspect_mask)
            .enumerate()
            .map(|(index, aspect_mask)| vk::BufferImageCopy {
                buffer_offset: index as u64 * 4 * u64::from(width) * u64::from(height),
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask,
                    mip_level: level,
                    base_array_layer: layer,
                    layer_count: 1,
                },
                image_offset: vk::Offset3D {
                    x: x as i32,
                    y: y as i32,
                    z: slice as i32,
                },
                image_extent: vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                },
                ..Default::default()
            })
            .collect();
        // SAFETY: recording; the image is in GENERAL layout; the buffer fits.
        unsafe {
            self.gpu.device.cmd_copy_image_to_buffer(
                commands,
                image,
                vk::ImageLayout::GENERAL,
                buffer.buffer,
                &copies,
            );
        }
        self.flush()?;
        // SAFETY: in bounds; the copy has completed.
        let bytes =
            unsafe { std::slice::from_raw_parts(buffer.mapped.cast::<u8>(), size as usize) }
                .to_vec();
        buffer.destroy(self.device());
        Ok(bytes)
    }

    /// Copy one plane of a scaled target into a staging target at the
    /// guest's resolution.
    fn unscaled_copy(
        &mut self,
        key: u64,
        level: u32,
        layer: u32,
        slice: u32,
        aspect_mask: vk::ImageAspectFlags,
    ) -> Result<u64, String> {
        let target = self.targets.get(&key).ok_or("no such target")?;
        let (width, height, format) = (
            (target.width >> level).max(1),
            (target.height >> level).max(1),
            target.format,
        );
        let (scaled_width, scaled_height) = target.extent(level);
        let staging_key = STAGING_KEYS + 1;
        self.target_with_levels(staging_key, width, height, format, 1)?;
        self.transition(key, vk::ImageLayout::GENERAL)?;
        self.order_target(key)?;
        self.transition(staging_key, vk::ImageLayout::GENERAL)?;
        let commands = self.commands()?;
        let subresource = vk::ImageSubresourceLayers {
            aspect_mask,
            mip_level: level,
            base_array_layer: layer,
            layer_count: 1,
        };
        let corner = |width: u32, height: u32| vk::Offset3D {
            x: width as i32,
            y: height as i32,
            z: 1,
        };
        let blits: Vec<_> = transfer_aspects(aspect_mask)
            .map(|aspect_mask| {
                let subresource = vk::ImageSubresourceLayers {
                    aspect_mask,
                    ..subresource
                };
                vk::ImageBlit {
                    src_subresource: subresource,
                    src_offsets: [
                        vk::Offset3D {
                            z: slice as i32,
                            ..Default::default()
                        },
                        vk::Offset3D {
                            z: slice as i32 + 1,
                            ..corner(scaled_width, scaled_height)
                        },
                    ],
                    dst_subresource: vk::ImageSubresourceLayers {
                        mip_level: 0,
                        base_array_layer: 0,
                        ..subresource
                    },
                    dst_offsets: [vk::Offset3D::default(), corner(width, height)],
                }
            })
            .collect();
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
        let (source, staging) = (self.targets[&key].image, self.targets[&staging_key].image);
        // SAFETY: recording; both images are live and in GENERAL layout.
        unsafe {
            let device = &self.gpu.device;
            device.cmd_blit_image(
                commands,
                source,
                vk::ImageLayout::GENERAL,
                staging,
                vk::ImageLayout::GENERAL,
                &blits,
                vk::Filter::NEAREST,
            );
            *self
                .stats
                .entry("stat: barriers in unscaled_copy")
                .or_default() += 1;
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
        Ok(staging_key)
    }

    /// A host-visible, coherent buffer.
    #[track_caller]
    pub(crate) fn host_buffer(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
    ) -> Result<HostBuffer, String> {
        self.host_buffer_with(size, usage, false)
    }

    /// `host_buffer`, with a device address when `addressable`.
    #[track_caller]
    pub(crate) fn host_buffer_with(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
        addressable: bool,
    ) -> Result<HostBuffer, String> {
        let usage = addressed_usage(usage, addressable);
        let info = vk::BufferCreateInfo::default()
            .size(size.max(4))
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let device = self.device();
        // SAFETY: valid create info; destroyed by HostBuffer::destroy.
        let buffer =
            unsafe { device.create_buffer(&info, None) }.map_err(vk_error("vkCreateBuffer"))?;
        // SAFETY: the buffer is live.
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        // Host-cached memory when there is some: the CPU reads results back
        // from these buffers, and reads from uncached (often device-local,
        // PCIe-mapped) memory are an order of magnitude slower.
        let coherent =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let memory = self
            .allocate_with(
                requirements,
                coherent | vk::MemoryPropertyFlags::HOST_CACHED,
                addressable,
            )
            .or_else(|_| self.allocate_with(requirements, coherent, addressable))?;
        // SAFETY: fresh buffer and memory; mapped for the buffer's lifetime.
        let mapped = unsafe {
            device
                .bind_buffer_memory(buffer, memory, 0)
                .map_err(vk_error("vkBindBufferMemory"))?;
            device
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .map_err(vk_error("vkMapMemory"))?
        };
        Ok(HostBuffer {
            buffer,
            memory,
            mapped: mapped.cast(),
            size: size.max(4),
        })
    }
}

struct RecycledPageTable {
    buffer: HostBuffer,
    // Covers changes since this memory was the current table. The buffer
    // itself stays immutable until its last submitted references finish.
    dirty: Vec<std::ops::Range<usize>>,
}

impl RecycledPageTable {
    fn mark_dirty(&mut self, mut changed: std::ops::Range<usize>) {
        if changed.is_empty() {
            return;
        }
        let first = self
            .dirty
            .partition_point(|range| range.end < changed.start);
        let end = self
            .dirty
            .partition_point(|range| range.start <= changed.end);
        if first < end {
            changed.start = changed.start.min(self.dirty[first].start);
            changed.end = changed.end.max(self.dirty[end - 1].end);
        }
        self.dirty.splice(first..end, [changed]);
    }
}

struct PageTableBuffer {
    address: u64,
    buffer: HostBuffer,
    entries: Vec<u64>,
    first_page: u32,
    recorded_read: bool,
    submitted_read: u64,
}

/// A mapped host-visible buffer.
pub(crate) struct HostBuffer {
    pub buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
    pub size: u64,
}

// SAFETY: the mapping belongs to this buffer alone; whoever owns the buffer
// may use it from any thread.
unsafe impl Send for HostBuffer {}

impl HostBuffer {
    /// A placeholder owning nothing.
    fn empty() -> Self {
        Self {
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            mapped: std::ptr::null_mut(),
            size: 0,
        }
    }

    /// `length` bytes of the mapping at `offset`, to write in place.
    pub fn slice_mut(&mut self, offset: usize, length: usize) -> &mut [u8] {
        assert!(offset + length <= self.size as usize);
        // SAFETY: in bounds of the mapping, which lives as long as self;
        // borrowed mutably, so no other slice of it exists meanwhile.
        unsafe { std::slice::from_raw_parts_mut(self.mapped.add(offset), length) }
    }

    pub fn write(&self, offset: usize, bytes: &[u8]) {
        assert!(offset + bytes.len() <= self.size as usize);
        // SAFETY: in bounds of the mapping, which lives as long as self.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.mapped.add(offset), bytes.len())
        };
    }

    fn write_page_addresses(&self, offset: usize, entries: &[u64]) {
        #[cfg(target_endian = "little")]
        {
            // SAFETY: u64 has no padding, and bytes have alignment 1.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    entries.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(entries),
                )
            };
            self.write(offset, bytes);
        }
        #[cfg(target_endian = "big")]
        for (index, entry) in entries.iter().enumerate() {
            self.write(offset + index * 8, &entry.to_le_bytes());
        }
    }

    /// The first `length` bytes as they are now (debugging: work may still
    /// be writing them).
    pub(crate) fn read_bytes(&self, length: usize) -> Vec<u8> {
        let length = length.min(self.size as usize);
        // SAFETY: in bounds of the mapping.
        unsafe { std::slice::from_raw_parts(self.mapped, length) }.to_vec()
    }

    fn read_f32x4(&self, count: usize) -> Vec<[f32; 4]> {
        assert!(count * 16 <= self.size as usize);
        // SAFETY: in bounds; the GPU work writing it has completed.
        let floats = unsafe { std::slice::from_raw_parts(self.mapped.cast::<f32>(), count * 4) };
        floats.as_chunks::<4>().0.to_vec()
    }

    pub fn destroy(self, device: &ash::Device) {
        if self.buffer == vk::Buffer::null() {
            return;
        }
        // SAFETY: no pending work uses the buffer (callers flush first).
        unsafe {
            device.unmap_memory(self.memory);
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

fn range(format: vk::Format) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: aspect(format),
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

/// `rect` limited to `bounds`, both (x, y, width, height).
pub(crate) fn clip(rect: [i32; 4], bounds: [i32; 4]) -> [i32; 4] {
    let x0 = rect[0].max(bounds[0]);
    let y0 = rect[1].max(bounds[1]);
    let x1 = rect[0]
        .saturating_add(rect[2])
        .min(bounds[0].saturating_add(bounds[2]));
    let y1 = rect[1]
        .saturating_add(rect[3])
        .min(bounds[1].saturating_add(bounds[3]));
    [x0, y0, x1.saturating_sub(x0), y1.saturating_sub(y0)]
}

impl Drop for Renderer {
    fn drop(&mut self) {
        if let Some(Some(compressor)) = self.archive_compressor.take() {
            compressor.stop();
        }
        let flushed = self.flush();
        if let Some(mut submitter) = self.submitter.take() {
            submitter.stop();
        }
        if flushed.is_err() {
            let _held = self
                .queue_lock
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            // SAFETY: the submit thread has joined and earlier submissions
            // may still own these resources after a submission failed.
            let _ = unsafe { self.gpu.device.device_wait_idle() };
        }
        if let Some(occlusion) = self.occlusion.take() {
            occlusion.destroy(&self.gpu.device);
        }
        let clear_programs = std::mem::take(&mut self.shader_clear_programs);
        let _ = self.destroy_idle_programs(clear_programs.into_iter().flatten());
        if let Some(mesh_count) = self.mesh_count.take() {
            mesh_count.destroy(&self.gpu.device);
        }
        if let Some(timer) = self.timer.take() {
            timer.destroy(&self.gpu.device);
        }
        for (_, (buffer, memory, _)) in std::mem::take(&mut self.resident) {
            // SAFETY: the flush above finished every use of the buffer.
            unsafe {
                self.gpu.device.destroy_buffer(buffer, None);
                self.gpu.device.free_memory(memory, None);
            }
        }
        if let Some(table) = self.page_table.take() {
            table.buffer.destroy(&self.gpu.device);
        }
        for storage in self.free_page_tables.drain(..) {
            storage.buffer.destroy(&self.gpu.device);
        }
        for slot in std::iter::once(&mut self.slot).chain(&mut self.spares) {
            for storage in slot.retired_page_tables.drain(..) {
                storage.buffer.destroy(&self.gpu.device);
            }
        }
        for (_, (buffer, memory)) in std::mem::take(&mut self.imported) {
            // SAFETY: the flush above finished every use of the buffer.
            unsafe {
                self.gpu.device.destroy_buffer(buffer, None);
                self.gpu.device.free_memory(memory, None);
            }
        }
        let shared = std::mem::take(&mut self.shared)
            .into_iter()
            .map(|(_, buffer, memory)| (buffer, memory))
            .chain(
                std::mem::take(&mut self.addressed)
                    .into_iter()
                    .chain(std::mem::take(&mut self.superseded))
                    .map(|(_, _, buffer, memory, ..)| (buffer, memory)),
            )
            .chain(
                std::mem::take(&mut self.packed)
                    .into_iter()
                    .map(|group| (group.buffer, group.memory)),
            );
        let retired = std::mem::take(&mut self.slot.retired_shared)
            .into_iter()
            .chain(
                self.spares
                    .iter_mut()
                    .flat_map(|spare| std::mem::take(&mut spare.retired_shared))
                    .collect::<Vec<_>>(),
            );
        for (buffer, memory) in shared.chain(retired) {
            // SAFETY: the flush above finished every use of the buffer.
            unsafe {
                self.gpu.device.destroy_buffer(buffer, None);
                self.gpu.device.free_memory(memory, None);
            }
        }
        for (_, target) in std::mem::take(&mut self.targets) {
            self.destroy_target(target);
        }
        for (_, (_, target)) in std::mem::take(&mut self.uploads) {
            self.destroy_target(target);
        }
        let mut retired_uploads = std::mem::take(&mut self.slot.retired_uploads);
        let mut retired_texel_buffers = std::mem::take(&mut self.slot.retired_texel_buffers);
        for spare in &mut self.spares {
            retired_uploads.append(&mut spare.retired_uploads);
            retired_texel_buffers.append(&mut spare.retired_texel_buffers);
        }
        for target in retired_uploads {
            self.destroy_target(target);
        }
        for (_, (_, buffer, view)) in std::mem::take(&mut self.texel_buffers) {
            self.destroy_texel_buffer(buffer, view);
        }
        for (buffer, view) in retired_texel_buffers {
            self.destroy_texel_buffer(buffer, view);
        }
        let ring = std::mem::replace(&mut self.slot.ring, HostBuffer::empty());
        ring.destroy(&self.gpu.device);
        for index in 0..self.spares.len() {
            let ring = std::mem::replace(&mut self.spares[index].ring, HostBuffer::empty());
            ring.destroy(&self.gpu.device);
        }
        self.destroy_texture_pool();
        for buffer in std::mem::take(&mut self.debug_buffers) {
            buffer.destroy(&self.gpu.device);
        }
        // SAFETY: all work has completed.
        unsafe {
            for (_, sampler) in std::mem::take(&mut self.samplers) {
                self.gpu.device.destroy_sampler(sampler, None);
            }
            for slot in std::iter::once(&self.slot).chain(&self.spares) {
                slot.destroy_pools_and_fence(&self.gpu.device);
            }
            for (_, (event, _)) in std::mem::take(&mut self.markers) {
                self.gpu.device.destroy_event(event, None);
            }
            for event in std::mem::take(&mut self.free_events) {
                self.gpu.device.destroy_event(event, None);
            }
        }
    }
}

/// Whether `format` stores texels in compressed blocks (BC, ETC2, EAC,
/// ASTC), which 1D images cannot hold.
fn block_compressed(format: vk::Format) -> bool {
    let raw = format.as_raw();
    (131..=184).contains(&raw) || (1_000_066_000..=1_000_066_013).contains(&raw)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_import_falls_back_when_the_guest_span_is_too_small() {
        #[repr(align(4096))]
        struct HostPage([u8; 4096]);

        let mut page = Box::new(HostPage([0; 4096]));
        let Some(mut renderer) = renderer() else {
            return;
        };
        if renderer.gpu.host_import.is_none() {
            return;
        }
        renderer.gpu.host_import_alignment = 64 << 10;
        assert!(
            renderer
                .import_host(page.0.as_mut_ptr(), page.0.len() as u64)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn failed_submission_releases_waiters_for_queued_batches() {
        let progress = SubmissionProgress::default();
        assert!(progress.finish(Ok(())));
        assert!(progress.returned(1));
        assert!(!progress.returned(2));
        std::thread::scope(|scope| {
            let failed = scope.spawn(|| progress.wait(2));
            let queued = scope.spawn(|| progress.wait(3));
            assert!(!progress.finish(Err("submission failed".into())));
            assert_eq!(failed.join().unwrap(), Err("submission failed".into()));
            assert_eq!(queued.join().unwrap(), Err("submission failed".into()));
        });
        assert_eq!(progress.failure().as_deref(), Some("submission failed"));
        assert!(!progress.returned(2));
    }

    #[test]
    fn sampler_cache_uses_effective_filtering_limits() {
        use crate::draw::Sampler;

        let Some(mut renderer) = renderer() else {
            return;
        };
        let limit = renderer.gpu.max_sampler_anisotropy;
        let requested = Sampler {
            max_anisotropy: limit + 1.0,
            ..Sampler::default()
        };
        let first = renderer.sampler(&requested, false).unwrap();
        let clamped = Sampler {
            max_anisotropy: limit,
            ..requested
        };
        let count = renderer.samplers.len();
        assert_eq!(renderer.sampler(&clamped, false).unwrap(), first);
        assert_eq!(renderer.samplers.len(), count);

        for sign in [-1.0, 1.0] {
            let limit = renderer.gpu.max_sampler_lod_bias;
            let mut requested = Sampler::default();
            requested.lod[2] = sign * (limit + 1.0);
            let first = renderer.sampler(&requested, false).unwrap();
            requested.lod[2] = sign * limit;
            let count = renderer.samplers.len();
            assert_eq!(renderer.sampler(&requested, false).unwrap(), first);
            assert_eq!(renderer.samplers.len(), count);
        }

        let base_level = Sampler {
            base_level_only: true,
            lod: [8.0, 12.0, 0.0],
            mipmap: vk::SamplerMipmapMode::LINEAR,
            border: [1.0f32.to_bits(); 4],
            ..Sampler::default()
        };
        let first = renderer.sampler(&base_level, false).unwrap();
        let equivalent = Sampler {
            lod: [0.0, 0.25, 0.0],
            mipmap: vk::SamplerMipmapMode::NEAREST,
            border: [0; 4],
            ..base_level
        };
        assert_eq!(renderer.sampler(&equivalent, false).unwrap(), first);
    }

    fn renderer() -> Option<Renderer> {
        // No device (CI): the GPU tests have nothing to run on.
        Gpu::new().ok().map(|gpu| Renderer::new(gpu).unwrap())
    }

    #[test]
    fn reads_an_upload_before_separate_submission() {
        use crate::draw::{ImageShape, TextureSource, UploadRegion};

        let Some(mut renderer) = renderer() else {
            return;
        };
        for (generation, bytes, rgba) in [
            (1, [255, 0, 0, 255], [1.0, 0.0, 0.0, 1.0]),
            (2, [0, 0, 255, 255], [0.0, 0.0, 1.0, 1.0]),
        ] {
            renderer
                .texture_view_for(&TextureSource::Upload {
                    key: 1,
                    generation,
                    width: 1,
                    height: 1,
                    shape: ImageShape::D2 { layers: 1 },
                    format: vk::Format::R8G8B8A8_UNORM,
                    texels: std::sync::Arc::new(bytes.to_vec()),
                    levels: [UploadRegion {
                        level: 0,
                        layer: 0,
                        width: 1,
                        height: 1,
                        depth: 1,
                        offset: 0,
                    }]
                    .into(),
                })
                .unwrap();
            assert_eq!(renderer.read_upload_level(1, 0).unwrap(), [rgba]);
        }
    }

    #[test]
    fn flush_collects_reads_from_queued_and_recorded_work() {
        let Some(mut renderer) = renderer() else {
            return;
        };
        renderer.resident_buffer(1, 4).unwrap();
        renderer.fill_resident(1, 0, 4, 0x1111_1111).unwrap();
        let first = renderer.read_resident(1, 0, 4).unwrap();
        renderer.submit_frame().unwrap();
        renderer.fill_resident(1, 0, 4, 0x2222_2222).unwrap();
        let second = renderer.read_resident(1, 0, 4).unwrap();
        let serial = renderer.recorded_serial();
        assert!(renderer.reads_pending());

        renderer.flush().unwrap();

        assert_eq!(renderer.completed_serial, serial);
        assert!(!renderer.reads_pending());
        let reads = renderer.take_reads(&[first, second]);
        assert_eq!(reads[&first], [0x11; 4]);
        assert_eq!(reads[&second], [0x22; 4]);
    }

    #[test]
    fn resident_readback_batches_keep_ranges_in_order() {
        let Some(mut renderer) = renderer() else {
            return;
        };
        renderer.resident_buffer(1, 16).unwrap();
        renderer.fill_resident(1, 0, 8, 0x11_11_11_11).unwrap();
        renderer.fill_resident(1, 8, 8, 0x22_22_22_22).unwrap();
        let tokens = renderer
            .read_resident_ranges(1, &[(8, 4), (16, 0), (4, 4)])
            .unwrap();
        renderer.flush().unwrap();

        assert!(renderer.take_reads(&[]).is_empty());
        let reads = renderer.take_reads(&[tokens[2], tokens[0], tokens[2]]);
        assert_eq!(reads.len(), 2);
        assert_eq!(reads[&tokens[0]], [0x22; 4]);
        assert_eq!(reads[&tokens[2]], [0x11; 4]);
        let remaining = renderer.take_reads(&[tokens[1]]);
        assert!(remaining[&tokens[1]].is_empty());
        assert!(renderer.take_reads(&tokens).is_empty());
        let stats = renderer.take_stats();
        assert_eq!(stats["stat: barriers in read_resident"], 1);
        assert_eq!(stats["stat: barriers in submit_current"], 1);
    }

    #[test]
    fn polling_collects_reads_after_the_submitter_and_fence_finish() {
        let Some(mut renderer) = renderer() else {
            return;
        };
        renderer.resident_buffer(1, 4).unwrap();
        renderer.fill_resident(1, 0, 4, 0x1234_5678).unwrap();
        let token = renderer.read_resident(1, 0, 4).unwrap();
        let serial = renderer.recorded_serial();
        renderer.submit_frame().unwrap();

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while renderer.completed_serial() < serial {
            renderer.poll_completed().unwrap();
            assert!(std::time::Instant::now() < deadline);
            std::thread::yield_now();
        }
        let reads = renderer.take_reads(&[token]);
        assert_eq!(reads[&token], 0x1234_5678_u32.to_le_bytes());
    }

    #[test]
    fn consecutive_full_barriers_share_one_dependency() {
        let Some(mut renderer) = renderer() else {
            return;
        };
        renderer.guest_barrier(0x11).unwrap();
        renderer.guest_barrier(0x12).unwrap();
        assert_eq!(renderer.hazard_barriers, 1);

        renderer
            .target(1, 8, 4, vk::Format::R8G8B8A8_UNORM)
            .unwrap();
        renderer
            .clear_color(
                1,
                None,
                vk::ClearColorValue {
                    float32: [1.0, 0.0, 0.0, 1.0],
                },
            )
            .unwrap();
        renderer.guest_barrier(0x11).unwrap();
        assert_eq!(renderer.hazard_barriers, 2);
        assert_eq!(renderer.read_target(1).unwrap()[0], [1.0, 0.0, 0.0, 1.0]);

        renderer.guest_barrier(0x12).unwrap();
        assert_eq!(renderer.hazard_barriers, 3);
        renderer.flush().unwrap();
        renderer.guest_barrier(0x12).unwrap();
        assert_eq!(renderer.hazard_barriers, 4);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn packed_dma_spans_share_noncontiguous_file_pages() {
        use std::io::{Seek, SeekFrom, Write};
        use std::os::fd::{AsRawFd, FromRawFd};

        let Some(mut renderer) = renderer() else {
            return;
        };
        let page = 4096;
        let name = c"echora-packed-test";
        // SAFETY: a valid name and flags. Ownership goes to File below.
        let fd = unsafe {
            libc::memfd_create(name.as_ptr(), libc::MFD_ALLOW_SEALING | libc::MFD_CLOEXEC)
        };
        assert!(fd >= 0);
        let mut backing = unsafe { std::fs::File::from_raw_fd(fd) };
        backing.set_len(9 * page).unwrap();
        assert_eq!(
            unsafe { libc::fcntl(fd, libc::F_ADD_SEALS, libc::F_SEAL_SHRINK) },
            0
        );

        let pages = [7, 1, 5, 3];
        let ranges: Vec<_> = pages
            .iter()
            .enumerate()
            .map(|(index, &slot)| {
                backing.seek(SeekFrom::Start(slot * page)).unwrap();
                backing.write_all(&[slot as u8; 32]).unwrap();
                (slot * page, page, index as u64 * page)
            })
            .collect();
        let Some((single, single_memory)) = renderer
            .share_span(backing.as_raw_fd(), page, page)
            .unwrap()
        else {
            assert!(std::env::var_os("ECHORA_REQUIRE_PACKED_DMA").is_none());
            return;
        };
        let (buffer, memory) = renderer
            .share_packed_spans(backing.as_raw_fd(), &ranges, page * 4)
            .unwrap()
            .unwrap();
        let output = renderer
            .host_buffer(page * 4, vk::BufferUsageFlags::TRANSFER_DST)
            .unwrap();
        let commands = renderer.commands().unwrap();
        let regions: Vec<_> = (0..4)
            .map(|index| vk::BufferCopy {
                src_offset: index * page,
                dst_offset: index * page,
                size: 32,
            })
            .collect();
        // SAFETY: both buffers are live and the regions fit. Queue completion
        // makes the writes visible to the coherent output mapping.
        unsafe {
            let device = &renderer.gpu.device;
            device.cmd_copy_buffer(commands, buffer, output.buffer, &regions);
            device.cmd_fill_buffer(commands, buffer, 64, 4, 0x1234_5678);
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ);
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
        renderer.flush().unwrap();
        let mut updated = [0; 4];
        use std::os::unix::fs::FileExt;
        backing
            .read_exact_at(&mut updated, pages[0] * page + 64)
            .unwrap();
        assert_eq!(updated, 0x1234_5678_u32.to_le_bytes());
        let bytes = output.read_bytes((page * 4) as usize);
        for (index, slot) in pages.into_iter().enumerate() {
            assert_eq!(&bytes[index * page as usize..][..32], &[slot as u8; 32]);
        }
        // SAFETY: idle buffers. The Vulkan imports own their DMA descriptors.
        unsafe {
            for (buffer, memory) in [(buffer, memory), (single, single_memory)] {
                renderer.gpu.device.destroy_buffer(buffer, None);
                renderer.gpu.device.free_memory(memory, None);
            }
            output.destroy(&renderer.gpu.device);
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn stable_span_packing_favors_smaller_imports_within_its_budget() {
        let large = Some(64 << 20);
        let medium = Some(16 << 20);
        let small = Some(1 << 20);
        let mut sizes = [large; 21];
        sizes[16..].copy_from_slice(&[medium, None, small, medium, small]);
        let selected = select_stable_spans(sizes.into_iter());
        assert!(selected[..7].iter().all(|&entry| entry));
        assert!(selected[7..16].iter().all(|&entry| !entry));
        assert_eq!(selected[16..], [true, false, true, true, true]);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn compacted_dma_spans_preserve_buffer_offsets_and_retire_together() {
        use std::os::fd::{AsRawFd, FromRawFd};

        let Some(mut renderer) = renderer() else {
            return;
        };
        let page = 4096;
        let gap = 128 * 1024;
        // SAFETY: a valid name and flags. Ownership goes to File below.
        let fd = unsafe {
            libc::memfd_create(
                c"echora-addressed-test".as_ptr(),
                libc::MFD_ALLOW_SEALING | libc::MFD_CLOEXEC,
            )
        };
        assert!(fd >= 0);
        // SAFETY: successful memfd_create returned a newly owned fd.
        let backing = unsafe { std::fs::File::from_raw_fd(fd) };
        backing.set_len(64 * gap).unwrap();
        // SAFETY: the memfd is live and allows seals.
        assert_eq!(
            unsafe { libc::fcntl(fd, libc::F_ADD_SEALS, libc::F_SEAL_SHRINK) },
            0
        );
        let file = backing.as_raw_fd();
        let Some(original) = renderer.address_span(file, 0, page).unwrap() else {
            assert!(std::env::var_os("ECHORA_REQUIRE_PACKED_DMA").is_none());
            return;
        };
        for index in 1..64 {
            assert!(
                renderer
                    .address_span(file, index * gap, page)
                    .unwrap()
                    .is_some()
            );
        }
        renderer.pack_addressed_spans().unwrap();
        if renderer.packed.is_empty() {
            assert!(std::env::var_os("ECHORA_REQUIRE_PACKED_DMA").is_none());
            return;
        }
        assert_eq!(renderer.addressed.len(), 0);
        assert_eq!(renderer.packed.len(), 1);
        assert_eq!(renderer.superseded_spans().0, 64);
        let base = renderer.packed[0].address;
        let buffer = renderer.packed[0].buffer;
        assert_ne!(base, original);
        assert!(renderer.shares_device_range(file, 24, 64, original + 24));
        for index in 0..64 {
            assert_eq!(
                renderer.address_span(file, index * gap, page).unwrap(),
                Some(base + index * page)
            );
            assert_eq!(
                renderer.import_file(file, index * gap + 24, 64).unwrap(),
                Some((buffer, index * page + 24))
            );
            assert!(renderer.shares_device_range(
                file,
                index * gap + 24,
                64,
                base + index * page + 24,
            ));
        }
        assert!(!renderer.shares_device_range(file, gap + 24, 64, base + 24));
        assert!(!renderer.shares_device_range(file, gap + page - 32, 64, base + 2 * page - 32));
        renderer.take_superseded(64);
        assert!(!renderer.shares_device_range(file, 24, 64, original + 24));
        renderer.release_file_range(gap, page);
        assert!(renderer.packed.is_empty());
        assert!(renderer.superseded_spans().2);
    }

    #[test]
    fn page_table_extension_preserves_a_submitted_readers_buffer() {
        let Some(mut renderer) = renderer() else {
            return;
        };
        let (buffer, _) = renderer.resident_buffer(9, 8192).unwrap();
        let Some(address) = renderer.buffer_address(buffer) else {
            return;
        };
        let (original, _) = renderer.set_page_table(8, &[address, 0]).unwrap();
        let device = renderer.device().clone();
        let mut timeline = vk::PhysicalDeviceVulkan12Features::default();
        let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut timeline);
        // SAFETY: the instance and physical device are live.
        unsafe {
            renderer
                .gpu
                .instance
                .get_physical_device_features2(renderer.gpu.physical, &mut features);
        }
        if timeline.timeline_semaphore == 0 {
            return;
        }
        let mut kind =
            vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
        let info = vk::SemaphoreCreateInfo::default().push_next(&mut kind);
        // SAFETY: the feature is enabled and the device is live.
        let gate = unsafe { device.create_semaphore(&info, None) }.unwrap();

        renderer.commands().unwrap();
        renderer.record_page_table_read(original.address);
        let wait = [gate];
        let stages = [vk::PipelineStageFlags::ALL_COMMANDS];
        let mut value = vk::TimelineSemaphoreSubmitInfo::default().wait_semaphore_values(&[1]);
        let prior = vk::SubmitInfo::default()
            .wait_semaphores(&wait)
            .wait_dst_stage_mask(&stages)
            .push_next(&mut value);
        let queue = renderer.gpu.queue;
        {
            let _hold = renderer.queue_lock.lock().unwrap();
            // SAFETY: the queue is held and the live timeline semaphore
            // is signaled below, before waiting for either submission.
            unsafe { device.queue_submit(queue, &[prior], vk::Fence::null()) }.unwrap();
        }
        let submitted = renderer.submit_frame();
        let replacement = renderer.set_page_table(8, &[address, address + 4096]);
        let signal = vk::SemaphoreSignalInfo::default().semaphore(gate).value(1);
        // SAFETY: signal a new value on a live timeline semaphore.
        let released = unsafe { device.signal_semaphore(&signal) };
        let finished = renderer.flush();
        submitted.unwrap();
        released.unwrap();
        finished.unwrap();
        // SAFETY: the last submission's fence orders the earlier wait.
        unsafe { device.destroy_semaphore(gate, None) };

        let (after, extended) = replacement.unwrap();
        assert!(!extended);
        assert_ne!(original.address, after.address);
        assert_eq!(renderer.page_table_entries(), [address, address + 4096]);
    }

    #[test]
    fn recycled_page_table_includes_changes_after_its_retirement() {
        let Some(mut renderer) = renderer() else {
            return;
        };
        if !renderer.gpu.buffer_device_address {
            return;
        }
        renderer.resident_buffer(7, 8192).unwrap();
        let address = renderer.resident_address(7).unwrap();
        let mut entries = vec![0; 1025];
        entries[128] = address;
        entries[131] = address + 4096;
        entries[1024] = address;
        let (original, _) = renderer.set_page_table(0x100, &entries).unwrap();
        renderer.commands().unwrap();
        renderer.record_page_table_read(original.address);
        entries[128] = address + 4096;
        let (replacement, extended) = renderer
            .patch_page_table(0x100, entries.len(), &[(128, &entries[128..129])])
            .unwrap();
        assert!(!extended);
        assert_ne!(original.address, replacement.address);

        entries[130] = address + 4096;
        entries[900] = address + 4096;
        let (extended_in_place, extended) = renderer
            .patch_page_table(
                0x100,
                entries.len(),
                &[(130, &entries[130..131]), (900, &entries[900..901])],
            )
            .unwrap();
        assert!(extended);
        assert_eq!(replacement.address, extended_in_place.address);
        assert_eq!(
            renderer.slot.retired_page_tables[0].dirty,
            [128..129, 130..131, 900..901]
        );
        let old = &renderer.slot.retired_page_tables[0].buffer;
        let prefix = old.read_bytes(901 * 8);
        assert_eq!(&prefix[128 * 8..129 * 8], &address.to_le_bytes());
        assert_eq!(&prefix[130 * 8..131 * 8], &[0; 8]);
        assert_eq!(&prefix[900 * 8..901 * 8], &[0; 8]);
        renderer.flush().unwrap();

        entries[131] = address;
        let (reused, extended) = renderer
            .patch_page_table(0x100, entries.len(), &[(131, &entries[131..132])])
            .unwrap();
        assert!(!extended);
        assert_eq!(reused.address, original.address);
        let expected: Vec<_> = entries
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        assert_eq!(
            renderer
                .page_table
                .as_ref()
                .unwrap()
                .buffer
                .read_bytes(expected.len()),
            expected,
        );
    }

    #[test]
    fn the_page_table_and_resident_buffers_have_device_addresses() {
        let Some(mut renderer) = renderer() else {
            return;
        };
        if !renderer.gpu.buffer_device_address {
            return;
        }
        let (_, created) = renderer.resident_buffer(7, 8192).unwrap();
        assert!(created);
        let resident = renderer.resident_address(7).unwrap();
        assert_ne!(resident, 0);
        let mut entries = vec![0; 1025];
        entries[0] = resident;
        entries[2] = resident + 4096;
        let (table, extended) = renderer.set_page_table(0x100, &entries).unwrap();
        assert!(!extended);
        assert_ne!(table.address, 0);
        assert_eq!((table.first_page, table.page_count), (0x100, 1025));

        let (again, extended) = renderer.set_page_table(0x100, &entries).unwrap();
        assert_eq!(again, table);
        assert!(!extended);

        entries[1] = resident + 4096;
        entries[1024] = resident;
        let before = renderer.page_table_entries().to_vec();
        assert!(
            renderer
                .patch_page_table(
                    0x100,
                    entries.len(),
                    &[(1, &entries[1..3]), (2, &entries[2..3])]
                )
                .is_err()
        );
        assert!(
            renderer
                .patch_page_table(
                    0x100,
                    entries.len(),
                    &[(1, &entries[1..2]), (1025, &entries[1024..])]
                )
                .is_err()
        );
        assert_eq!(renderer.page_table_entries(), before);
        let (same, extended) = renderer
            .patch_page_table(
                0x100,
                entries.len(),
                &[(1, &entries[1..2]), (1024, &entries[1024..])],
            )
            .unwrap();
        assert_eq!(same, table);
        assert!(extended);
        assert_eq!(renderer.page_table_entries(), entries);
        let expected: Vec<_> = entries
            .iter()
            .flat_map(|entry| entry.to_le_bytes())
            .collect();
        assert_eq!(
            renderer
                .page_table
                .as_ref()
                .unwrap()
                .buffer
                .read_bytes(expected.len()),
            expected
        );

        renderer.commands().unwrap();
        entries[0] = 0;
        entries[1] = resident;
        let (replaced, extended) = renderer
            .patch_page_table(0x100, entries.len(), &[(0, &entries[..2])])
            .unwrap();
        assert_eq!(renderer.page_table_entries(), entries);
        assert!(!extended);
        assert_ne!(replaced.address, table.address);
        assert!(renderer.free_page_tables.is_empty());
        assert_eq!(renderer.slot.retired_page_tables.len(), 1);

        renderer.flush().unwrap();
        entries[0] = resident + 4096;
        entries[1] = 0;
        entries[2] = resident;
        let (reused, extended) = renderer.set_page_table(0x200, &entries).unwrap();
        assert!(!extended);
        assert_eq!(reused.address, table.address);
        assert_eq!((reused.first_page, reused.page_count), (0x200, 1025));
        assert_eq!(renderer.page_table_entries(), entries);
        let expected: Vec<_> = entries
            .iter()
            .flat_map(|entry| entry.to_le_bytes())
            .collect();
        assert_eq!(
            renderer
                .page_table
                .as_ref()
                .unwrap()
                .buffer
                .read_bytes(expected.len()),
            expected
        );
        let (rebased, extended) = renderer.set_page_table(0x201, &entries).unwrap();
        assert!(!extended);
        assert_ne!(rebased.address, reused.address);
    }

    #[test]
    fn clip_limits_to_bounds() {
        assert_eq!(clip([-5, 2, 20, 4], [0, 0, 10, 10]), [0, 2, 10, 4]);
        assert_eq!(clip([12, 0, 4, 4], [0, 0, 10, 10])[2], -2);
        assert!(clip([i32::MAX, 0, 4, 4], [0, 0, 10, 10])[2] < 0);
    }

    #[test]
    fn relative_mip_views_keep_their_source_ranges() {
        use crate::draw::{ImageShape, Sampler, TextureSource};

        let Some(mut renderer) = renderer() else {
            return;
        };
        renderer
            .target_with_levels(7, 8, 8, vk::Format::R8G8B8A8_UNORM, 4)
            .unwrap();
        let sampler = Sampler {
            view_levels: Some((0, 1)),
            ..Default::default()
        };
        for base_level in [1, 2, 1] {
            let source = TextureSource::TargetView {
                key: 7,
                base_level,
                level_count: 2,
                base_layer: 0,
                layer_count: 1,
            };
            let view = renderer
                .texture_view_swizzled(&source, 2, &sampler)
                .unwrap();
            let range = renderer.view_infos[&view].range;
            assert_eq!((range.base_mip_level, range.level_count), (base_level, 1));
        }
        for relative_base in [1, 2, 3] {
            let sampler = Sampler {
                view_levels: Some((relative_base, 0)),
                ..Default::default()
            };
            let view = renderer
                .texture_view_swizzled(&TextureSource::Target(7), 2, &sampler)
                .unwrap();
            let range = renderer.view_infos[&view].range;
            assert_eq!(
                (range.base_mip_level, range.level_count),
                (relative_base, 4 - relative_base)
            );
        }
        renderer
            .shaped_target(
                9,
                8,
                1,
                ImageShape::D1 { layers: 3 },
                vk::Format::R8G8B8A8_UNORM,
                4,
            )
            .unwrap();
        let source = TextureSource::TargetView {
            key: 9,
            base_level: 1,
            level_count: 2,
            base_layer: 1,
            layer_count: 2,
        };
        for (texture_type, view_type, layer_count) in [
            (0, vk::ImageViewType::TYPE_1D, 1),
            (1, vk::ImageViewType::TYPE_1D_ARRAY, 2),
        ] {
            let view = renderer.texture_view(&source, texture_type).unwrap();
            let info = &renderer.view_infos[&view];
            assert_eq!(info.view_type, view_type);
            assert_eq!(
                (
                    info.range.base_mip_level,
                    info.range.level_count,
                    info.range.base_array_layer,
                    info.range.layer_count,
                ),
                (1, 2, 1, layer_count),
            );
        }
        renderer
            .shaped_target(
                8,
                8,
                8,
                ImageShape::D3 { depth: 8 },
                vk::Format::R8G8B8A8_UNORM,
                4,
            )
            .unwrap();
        let volume = TextureSource::TargetView {
            key: 8,
            base_level: 1,
            level_count: 2,
            base_layer: 0,
            layer_count: 1,
        };
        let view = renderer.texture_view(&volume, 4).unwrap();
        let range = renderer.view_infos[&view].range;
        assert_eq!((range.base_mip_level, range.level_count), (1, 2));
    }

    #[test]
    fn combined_depth_stencil_uploads_copy_each_plane() {
        use crate::draw::{ImageShape, TextureSource, UploadRegion};

        let Some(mut renderer) = renderer() else {
            return;
        };
        let format = vk::Format::D32_SFLOAT_S8_UINT;
        let shape = ImageShape::D2 { layers: 2 };
        if renderer.shaped_target(9, 2, 1, shape, format, 2).is_err() {
            // D32_SFLOAT_S8_UINT is optional.
            return;
        }
        let contents = [
            (0, 0, vec![0.125_f32, 0.25_f32], vec![9, 25]),
            (0, 1, vec![0.375, 0.5], vec![72, 255]),
            (1, 0, vec![0.625], vec![94]),
            (1, 1, vec![0.75], vec![141]),
        ];
        let mut texels = Vec::new();
        let levels: std::sync::Arc<[UploadRegion]> = contents
            .iter()
            .map(|(level, layer, depth, stencil)| {
                let offset = texels.len() as u64;
                texels.extend(depth.iter().flat_map(|value| value.to_le_bytes()));
                texels.extend(stencil);
                texels.resize(texels.len().next_multiple_of(4), 0);
                UploadRegion {
                    level: *level,
                    layer: *layer,
                    width: depth.len() as u32,
                    height: 1,
                    depth: 1,
                    offset,
                }
            })
            .collect();
        let mut source = TextureSource::Upload {
            key: 6,
            generation: 1,
            width: 2,
            height: 1,
            shape,
            format,
            texels: std::sync::Arc::new(texels),
            levels: levels.clone(),
        };
        renderer.initialize_target(9, &source).unwrap();
        if let TextureSource::Upload { texels, .. } = &mut source {
            *texels = Default::default();
        }
        let TextureSource::Upload { texels, .. } = renderer.with_texels(&source).unwrap() else {
            unreachable!();
        };
        renderer.set_resolution_scale(Scale::new(2, 1).unwrap());
        renderer.shaped_target(10, 2, 1, shape, format, 2).unwrap();
        renderer.initialize_target(10, &source).unwrap();
        for (region, (level, layer, depth, stencil)) in levels.iter().zip(contents) {
            let rect = [0, 0, region.width, 1];
            let start = region.offset as usize;
            let depth: Vec<_> = depth.iter().flat_map(|value| value.to_le_bytes()).collect();
            let split = start + depth.len();
            assert_eq!(texels[start..split], depth);
            assert_eq!(texels[split..split + stencil.len()], stencil);
            assert_eq!(
                renderer
                    .read_target_plane_bytes(9, level, layer, 4, rect)
                    .unwrap(),
                depth
            );
            assert_eq!(
                renderer
                    .read_target_aspect_bytes(
                        9,
                        (level, layer),
                        rect,
                        (vk::ImageAspectFlags::STENCIL, 1)
                    )
                    .unwrap(),
                stencil
            );
            let expected = [depth, stencil].concat();
            for key in [9, 10] {
                assert_eq!(
                    renderer
                        .read_target_depth_stencil_plane_bytes(key, level, layer, rect)
                        .unwrap(),
                    expected
                );
            }
        }
    }

    #[test]
    fn upload_views_select_layers_and_mips_together() {
        use crate::draw::{ImageShape, Sampler, TextureSource, UploadRegion};

        let Some(mut renderer) = renderer() else {
            return;
        };
        let mut texels = Vec::new();
        let mut levels = Vec::new();
        for level in 0..2 {
            let size = 2 >> level;
            for layer in 0..4 {
                levels.push(UploadRegion {
                    level,
                    layer,
                    width: size,
                    height: size,
                    depth: 1,
                    offset: texels.len() as u64,
                });
                texels.extend(std::iter::repeat_n(layer as u8, (size * size * 4) as usize));
            }
        }
        let source = TextureSource::Upload {
            key: 19,
            generation: 1,
            width: 2,
            height: 2,
            shape: ImageShape::D2 { layers: 4 },
            format: vk::Format::R8G8B8A8_UNORM,
            texels: std::sync::Arc::new(texels),
            levels: levels.into(),
        };
        for (texture_type, layers, selected) in [
            (2, (2, 0), (2, 1)),
            (3, (2, 0), (2, 2)),
            (3, (1, 1), (1, 1)),
            (3, (2, 0), (2, 2)),
        ] {
            let sampler = Sampler {
                view_levels: Some((1, 0)),
                view_layers: Some(layers),
                ..Default::default()
            };
            let view = renderer
                .texture_view_swizzled(&source, texture_type, &sampler)
                .unwrap();
            let range = renderer.view_infos[&view].range;
            assert_eq!((range.base_mip_level, range.level_count), (1, 1));
            assert_eq!((range.base_array_layer, range.layer_count), selected);
        }
    }

    #[test]
    fn generated_upload_retries_after_a_partial_failure() {
        use crate::draw::{ImageShape, TextureSource, UploadRegion};

        let Some(mut renderer) = renderer() else {
            return;
        };
        let key = 19;
        // Padding forces several ring batches with only a few actual
        // pixels to read back from the GPU.
        let stride = super::RING_SIZE / 2;
        let levels: Vec<_> = (0..4)
            .map(|index| {
                let level = index / 2;
                let extent = 2 >> level;
                UploadRegion {
                    level,
                    layer: index % 2,
                    width: extent,
                    height: extent,
                    depth: 1,
                    offset: u64::from(index) * stride,
                }
            })
            .collect();
        let source = TextureSource::Upload {
            key,
            generation: 1,
            width: 2,
            height: 2,
            shape: ImageShape::D2 { layers: 2 },
            format: vk::Format::R8G8B8A8_UNORM,
            texels: Default::default(),
            levels: levels.clone().into(),
        };
        let size = (3 * stride + 4) as usize;
        let fill = |offset, bytes: &mut [u8]| {
            let end = offset + bytes.len() as u64;
            for region in levels
                .iter()
                .filter(|region| (offset..end).contains(&region.offset))
            {
                let start = (region.offset - offset) as usize;
                let length = (region.width * region.height * 4) as usize;
                bytes[start..start + length].fill(10 + (region.offset / stride) as u8);
            }
        };
        let failure = renderer.upload_generated(&source, size, |offset, bytes| {
            if offset != 0 {
                return Err("generated texels unavailable".into());
            }
            fill(offset, bytes);
            Ok(())
        });
        assert_eq!(failure, Err("generated texels unavailable".into()));
        assert!(!renderer.has_upload(key, 1));

        renderer
            .upload_generated(&source, size, |offset, bytes| {
                fill(offset, bytes);
                Ok(())
            })
            .unwrap();
        renderer.texture_view_for(&source).unwrap();
        for region in levels {
            let pixels = renderer
                .read_upload_slice(key, region.level, region.layer)
                .unwrap();
            let value = (10 + region.offset / stride) as f32 / 255.0;
            assert_eq!(pixels.len(), (region.width * region.height) as usize);
            assert!(
                pixels
                    .iter()
                    .flatten()
                    .all(|pixel| (pixel - value).abs() < 1e-6)
            );
        }
    }

    #[test]
    fn srgb_target_views_keep_storage_on_the_unorm_twin() {
        use crate::draw::{ImageShape, Sampler, TextureSource};

        let Some(mut renderer) = renderer() else {
            return;
        };
        let format = vk::Format::R8G8B8A8_SRGB;
        let Some(storage) = renderer.storage_twin(format) else {
            return;
        };
        let target = renderer
            .shaped_target(7, 8, 8, ImageShape::D2 { layers: 2 }, format, 4)
            .unwrap();
        assert!(!target.view_usage.contains(vk::ImageUsageFlags::STORAGE));
        assert!(target.view_usage.contains(vk::ImageUsageFlags::SAMPLED));
        let mip = renderer.level_view(7, 2).unwrap();
        let layer = renderer.attachment_view(7, 2, 1, None).unwrap();
        let writable = renderer
            .storage_image_view(
                StorageImage {
                    target: 7,
                    level: 2,
                    ..Default::default()
                },
                2,
                storage,
            )
            .unwrap();
        assert_ne!(mip, writable);
        assert_ne!(layer, writable);

        renderer.texture_view(&TextureSource::Target(7), 2).unwrap();
        let source = TextureSource::TargetView {
            key: 7,
            base_level: 1,
            level_count: 2,
            base_layer: 1,
            layer_count: 1,
        };
        let sampler = Sampler {
            view_levels: Some((1, 1)),
            ..Default::default()
        };
        let view = renderer
            .texture_view_swizzled(&source, 2, &sampler)
            .unwrap();
        let range = renderer.view_infos[&view].range;
        assert_eq!((range.base_mip_level, range.base_array_layer), (2, 1));
    }

    #[test]
    fn repeated_identity_volume_binds_keep_the_uploaded_texels() {
        use crate::draw::{ImageShape, TextureSource, UploadRegion};

        let Some(mut renderer) = renderer() else {
            return;
        };
        let format = vk::Format::R8G8B8A8_UNORM;
        renderer.target(18, 8, 8, format).unwrap();
        let source = TextureSource::Target(18);
        let first = renderer.texture_view(&source, 4).unwrap();
        assert_eq!(renderer.texture_view(&source, 4).unwrap(), first);

        let upload = TextureSource::Upload {
            key: u64::MAX - 8,
            generation: 0,
            width: 8,
            height: 8,
            shape: ImageShape::D3 { depth: 8 },
            format,
            texels: Default::default(),
            levels: [UploadRegion {
                level: 0,
                layer: 0,
                width: 8,
                height: 8,
                depth: 8,
                offset: 0,
            }]
            .into(),
        };
        let TextureSource::Upload { texels, .. } = renderer.with_texels(&upload).unwrap() else {
            unreachable!();
        };
        assert_eq!(&texels[..4], &[0, 0, 0, 255]);
        assert_eq!(&texels[4 * 511..], &[255, 255, 255, 255]);
        assert_eq!(&texels[4 * 163..4 * 164], &[109, 145, 72, 255]);
    }

    #[test]
    fn repeated_volume_array_copies_are_ordered() {
        let Some(mut renderer) = renderer() else {
            return;
        };
        let format = vk::Format::R8G8B8A8_UNORM;
        renderer
            .shaped_target(1, 4, 4, ImageShape::D3 { depth: 2 }, format, 1)
            .unwrap();
        renderer.zero_target(1).unwrap();
        renderer.volume_as_layers(1, 2, 4, 4, 2, format).unwrap();
        renderer.volume_as_layers(1, 2, 4, 4, 2, format).unwrap();
        assert_eq!(renderer.read_target_bytes(2, 4).unwrap(), vec![0; 64]);
    }

    #[test]
    fn a_cached_upload_alias_needs_no_readback_of_the_original() {
        use crate::draw::{ImageShape, TextureSource, UploadRegion};

        let Some(mut renderer) = renderer() else {
            return;
        };
        for (key, shape, depth, original_type, alias_type) in [
            (20, ImageShape::D2 { layers: 1 }, 1, 2, 0),
            (21, ImageShape::D2 { layers: 1 }, 1, 2, 4),
            (22, ImageShape::D3 { depth: 4 }, 4, 4, 3),
        ] {
            let mut source = TextureSource::Upload {
                key,
                generation: 1,
                width: 4,
                height: 4,
                shape,
                format: vk::Format::R8G8B8A8_UNORM,
                texels: std::sync::Arc::new(vec![0x3a; 64 * depth as usize]),
                levels: [UploadRegion {
                    level: 0,
                    layer: 0,
                    width: 4,
                    height: 4,
                    depth,
                    offset: 0,
                }]
                .into(),
            };
            renderer.texture_view(&source, original_type).unwrap();
            let alias = renderer.texture_view(&source, alias_type).unwrap();
            renderer.flush().unwrap();
            let TextureSource::Upload { texels, .. } = &mut source else {
                unreachable!();
            };
            *texels = Default::default();
            let submissions = renderer.flushes;
            assert_eq!(renderer.texture_view(&source, alias_type).unwrap(), alias);
            assert_eq!(renderer.flushes, submissions);
            if let TextureSource::Upload { generation, .. } = &mut source {
                *generation += 1;
            }
            assert!(renderer.texture_view(&source, alias_type).is_err());
        }
    }

    #[test]
    fn reusing_a_target_cancels_its_pending_archive() {
        let Some(mut renderer) = renderer() else {
            return;
        };
        let key = 7;
        renderer
            .target(key, 512, 512, vk::Format::R8G8B8A8_UNORM)
            .unwrap();
        let clear = |renderer: &mut Renderer, red| {
            renderer.defined(key);
            renderer
                .clear_color(
                    key,
                    None,
                    vk::ClearColorValue {
                        float32: [red, 0.0, 0.0, 1.0],
                    },
                )
                .unwrap();
        };
        clear(&mut renderer, 0.0);
        renderer.flush().unwrap();
        assert_eq!(
            renderer.queue_target_archive(key).unwrap(),
            ArchiveRequest::Queued
        );
        clear(&mut renderer, 1.0);
        renderer.flush().unwrap();
        renderer.finish_target_archives().unwrap();
        assert!(renderer.archived_targets.is_empty());
        assert_eq!(renderer.pending_archive_bytes, 0);
        assert_eq!(
            renderer.read_target_bytes(key, 4).unwrap(),
            [255, 0, 0, 255].repeat(512 * 512)
        );

        assert_eq!(
            renderer.queue_target_archive(key).unwrap(),
            ArchiveRequest::Queued
        );
        renderer.flush().unwrap();
        renderer.finish_target_archives().unwrap();
        if let Some(worker) = renderer.archive_compressor.get().and_then(Option::as_ref) {
            let (result, compressed) = worker
                .results
                .recv_timeout(std::time::Duration::from_secs(5))
                .unwrap();
            assert_eq!(result, key);
            clear(&mut renderer, 0.0);
            renderer.finish_target_archive(result, compressed);
            assert!(renderer.archived_targets.is_empty());
            assert_eq!(renderer.pending_archive_bytes, 0);
            assert_eq!(
                renderer.read_target_bytes(key, 4).unwrap(),
                [0, 0, 0, 255].repeat(512 * 512)
            );
        }
    }

    #[test]
    fn open_pass_tracks_the_image_under_each_target_key() {
        let Some(mut renderer) = renderer() else {
            return;
        };
        let format = vk::Format::R8G8B8A8_UNORM;
        renderer.target(1, 4, 4, format).unwrap();
        renderer
            .begin_rendering(&[1], &[], &[], &[], None, [0, 0, 2, 2])
            .unwrap();
        renderer.target(3, 4, 4, format).unwrap();
        renderer
            .begin_rendering(&[1], &[0], &[0], &[Some(format)], None, [2, 2, 2, 2])
            .unwrap();
        assert_eq!(renderer.stats["stat: render passes begun"], 1);

        renderer.move_target(1, 2).unwrap();
        assert_eq!(renderer.target_keys_in_order(), [2, 3]);
        renderer.target(1, 4, 4, format).unwrap();
        renderer.target(STAGING_KEYS + 1, 4, 4, format).unwrap();
        assert_eq!(renderer.target_keys_in_order(), [1, 2, 3]);
        renderer
            .begin_rendering(&[1], &[0], &[0], &[Some(format)], None, [2, 2, 2, 2])
            .unwrap();
        renderer.flush().unwrap();
        assert_eq!(renderer.stats["stat: render passes begun"], 2);
        renderer.move_target(2, 3).unwrap();
        assert_eq!(renderer.target_keys_in_order(), [1, 3]);
        renderer.move_target(3, STAGING_KEYS + 1).unwrap();
        assert_eq!(renderer.target_keys_in_order(), [1]);
        renderer.move_target(STAGING_KEYS + 1, 4).unwrap();
        assert_eq!(renderer.target_keys_in_order(), [1, 4]);
    }

    #[test]
    fn moving_a_target_keeps_its_guest_initialization_state() {
        let Some(mut renderer) = renderer() else {
            return;
        };
        let format = vk::Format::R32_SFLOAT;
        renderer.target(1, 1, 1, format).unwrap();
        renderer.target(2, 1, 1, format).unwrap();
        renderer.defined(2);
        renderer.move_target(1, 2).unwrap();
        assert_eq!(renderer.take_fresh_targets(), [2]);

        renderer.target(3, 1, 1, format).unwrap();
        renderer.move_target(2, 3).unwrap();
        assert!(renderer.take_fresh_targets().is_empty());
    }

    #[test]
    fn resizing_a_target_keeps_all_overlapping_mip_levels() {
        let Some(mut renderer) = renderer() else {
            return;
        };
        let format = vk::Format::R8G8B8A8_UNORM;
        let red = [1.0, 0.0, 0.0, 1.0];
        let green = [0.0, 1.0, 0.0, 1.0];
        renderer.target_with_levels(1, 16, 8, format, 2).unwrap();
        renderer
            .clear_color_level(1, 0, 0, None, vk::ClearColorValue { float32: red })
            .unwrap();
        renderer
            .clear_color_level(1, 1, 0, None, vk::ClearColorValue { float32: green })
            .unwrap();

        renderer.target_with_levels(1, 16, 16, format, 3).unwrap();
        let base = renderer.read_target_level(1, 0).unwrap();
        assert_eq!(&base[..16 * 8], &[red; 16 * 8]);
        let mip = renderer.read_target_level(1, 1).unwrap();
        assert_eq!(&mip[..8 * 4], &[green; 8 * 4]);
        assert_eq!(
            renderer.read_target_level(1, 2).unwrap(),
            vec![[0.0; 4]; 16]
        );

        renderer.target_with_levels(1, 16, 4, format, 2).unwrap();
        assert_eq!(renderer.read_target_level(1, 0).unwrap(), vec![red; 16 * 4]);
        assert_eq!(
            renderer.read_target_level(1, 1).unwrap(),
            vec![green; 8 * 2]
        );
    }

    #[test]
    fn target_allocation_failure_can_be_retried() {
        let Some(mut renderer) = renderer() else {
            return;
        };
        let memories = renderer.memory_types.memory_type_count;
        renderer.memory_types.memory_type_count = 0;
        let format = vk::Format::R8G8B8A8_UNORM;
        let key = 1;
        let error = renderer.target(key, 2, 2, format).err();
        renderer.memory_types.memory_type_count = memories;
        assert_eq!(error.as_deref(), Some("no memory type with DEVICE_LOCAL"));
        assert!(renderer.existing_target(key).is_none());

        renderer.target(key, 2, 2, format).unwrap();
        let color = [1.0, 0.0, 1.0, 1.0];
        renderer
            .clear_color(key, None, vk::ClearColorValue { float32: color })
            .unwrap();
        assert_eq!(renderer.read_target(key).unwrap(), vec![color; 4]);
    }

    #[test]
    fn whole_and_partial_clears_read_back() {
        let Some(mut renderer) = renderer() else {
            return;
        };
        renderer
            .target(1, 8, 4, vk::Format::R8G8B8A8_UNORM)
            .unwrap();
        renderer
            .clear_color(
                1,
                None,
                vk::ClearColorValue {
                    float32: [1.0, 0.0, 0.0, 1.0],
                },
            )
            .unwrap();
        renderer
            .clear_color(
                1,
                Some([4, 0, 4, 4]),
                vk::ClearColorValue {
                    float32: [0.0, 0.0, 1.0, 1.0],
                },
            )
            .unwrap();
        let texels = renderer.read_target(1).unwrap();
        assert_eq!(texels.len(), 32);
        assert_eq!(texels[0], [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(texels[7], [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(texels[3 * 8 + 3], [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn reopening_the_same_attachment_orders_the_previous_pass() {
        let Some(mut renderer) = renderer() else {
            return;
        };
        renderer
            .target(1, 8, 4, vk::Format::R8G8B8A8_UNORM)
            .unwrap();
        let red = [1.0, 0.0, 0.0, 1.0];
        let blue = [0.0, 0.0, 1.0, 1.0];
        renderer
            .clear_color(1, Some([0, 0, 4, 4]), vk::ClearColorValue { float32: red })
            .unwrap();
        renderer.end_pass();
        let before = renderer.hazard_barriers;
        renderer
            .clear_color(1, Some([4, 0, 4, 4]), vk::ClearColorValue { float32: blue })
            .unwrap();
        let ordered = renderer.hazard_barriers > before;
        let texels = renderer.read_target(1).unwrap();
        assert_eq!(texels[0..8], [red, red, red, red, blue, blue, blue, blue]);
        assert!(
            ordered,
            "a new rendering pass must wait for the prior store"
        );
    }

    #[test]
    fn retiring_an_image_keeps_its_scaled_texels_and_logical_target_key() {
        let Some(mut renderer) = renderer() else {
            return;
        };
        let format = vk::Format::R16G16B16A16_SFLOAT;
        renderer.set_resolution_scale(Scale::new(3, 2).unwrap());
        let (width, height) = renderer.target(7, 512, 256, format).unwrap().extent(0);
        assert_eq!((width, height), (768, 384));
        let color = [0.5, 4.0, 0.25, 1.0];
        renderer
            .clear_color(7, None, vk::ClearColorValue { float32: color })
            .unwrap();
        assert_eq!(
            renderer.queue_target_archive(7).unwrap(),
            ArchiveRequest::Queued
        );
        renderer.flush().unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while renderer.existing_target(7).is_some() {
            assert!(std::time::Instant::now() < deadline);
            renderer.finish_target_archives().unwrap();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let archive = &renderer.archived_targets[&7];
        assert_eq!(archive.decoded_bytes, (width * height * 8) as usize);
        assert!(archive.compressed.len() < archive.decoded_bytes);
        assert!(renderer.existing_target(7).is_none());
        renderer.move_target(7, 8).unwrap();
        assert_eq!(renderer.target_keys_in_order(), [8]);
        assert_eq!(
            renderer.existing_target(8).unwrap().extent(0),
            (width, height)
        );
        let texels = renderer.read_target_scaled(8).unwrap().0;
        assert!(texels.iter().all(|texel| *texel == color));
    }

    #[test]
    fn marker_includes_pending_counter_reports() {
        let Some(mut renderer) = renderer() else {
            return;
        };
        let (buffer, _) = renderer.resident_buffer(1, 16).unwrap();
        renderer.fill_resident(1, 0, 16, 0).unwrap();
        renderer.samples_reset().unwrap();
        renderer.samples_report(buffer, 0, 0x1234).unwrap();
        let reports_pending =
            |renderer: &Renderer| renderer.occlusion.as_ref().unwrap().pending_at(buffer, 0);
        assert!(reports_pending(&renderer));
        let marker = renderer.mark().unwrap().unwrap();
        assert!(!reports_pending(&renderer));
        let token = renderer.read_resident(1, 0, 16).unwrap();
        renderer.flush().unwrap();
        assert!(renderer.marker_done(marker));
        assert_eq!(
            renderer.take_reads(&[token])[&token],
            [0_u64, 0x1234].map(u64::to_le_bytes).concat()
        );
    }

    #[test]
    fn overlapping_counter_reports_follow_command_order() {
        let Some(mut renderer) = renderer() else {
            return;
        };
        let (buffer, _) = renderer.resident_buffer(1, 24).unwrap();
        renderer.samples_reset().unwrap();
        renderer.samples_report(buffer, 0, 0x1234).unwrap();
        renderer.samples_report(buffer, 8, 0x2345).unwrap();
        renderer.samples_report(buffer, 8, 0x3456).unwrap();
        renderer.mark().unwrap();
        let token = renderer.read_resident(1, 0, 24).unwrap();
        renderer.flush().unwrap();
        assert_eq!(
            renderer.take_reads(&[token])[&token],
            [0_u64, 0, 0x3456].map(u64::to_le_bytes).concat()
        );
    }

    #[test]
    fn color_and_depth_clears_follow_a_gpu_condition() {
        let Some(mut renderer) = renderer() else {
            return;
        };
        if !renderer.supports_conditions() {
            return;
        }
        let (predicate, _) = renderer.resident_buffer(1, 16).unwrap();
        renderer.fill_resident(1, 0, 16, 0).unwrap();
        renderer
            .target(2, 8, 4, vk::Format::R8G8B8A8_UNORM)
            .unwrap();
        renderer.target(3, 8, 4, vk::Format::D32_SFLOAT).unwrap();
        let red = [1.0, 0.0, 0.0, 1.0];
        let blue = [0.0, 0.0, 1.0, 1.0];
        renderer
            .clear_color(2, None, vk::ClearColorValue { float32: red })
            .unwrap();
        renderer
            .clear_depth_stencil(3, 0, None, Some(0.25), None)
            .unwrap();

        renderer.set_condition(Some((predicate, 0, false)));
        renderer
            .clear_color(2, None, vk::ClearColorValue { float32: blue })
            .unwrap();
        renderer
            .clear_depth_stencil(3, 0, None, Some(0.75), None)
            .unwrap();
        assert_eq!(renderer.read_target(2).unwrap(), vec![red; 32]);
        let depth = renderer.read_target_bytes(3, 4).unwrap();
        assert_eq!(depth, 0.25_f32.to_le_bytes().repeat(32));

        renderer.set_condition(Some((predicate, 0, true)));
        renderer
            .clear_color(2, Some([4, 0, 4, 4]), vk::ClearColorValue { float32: blue })
            .unwrap();
        renderer
            .clear_depth_stencil(3, 0, Some([4, 0, 4, 4]), Some(0.75), None)
            .unwrap();
        let colors = renderer.read_target(2).unwrap();
        let depth = renderer.read_target_bytes(3, 4).unwrap();
        for row in colors.chunks_exact(8) {
            assert_eq!(row, [red, red, red, red, blue, blue, blue, blue]);
        }
        for row in depth.chunks_exact(32) {
            assert_eq!(&row[..16], 0.25_f32.to_le_bytes().repeat(4));
            assert_eq!(&row[16..], 0.75_f32.to_le_bytes().repeat(4));
        }
    }

    #[test]
    fn masked_color_clears_preserve_channels_and_use_the_view_format() {
        let Some(mut renderer) = renderer() else {
            return;
        };
        if !renderer.supports_conditions() {
            return;
        }
        let (predicate, _) = renderer.resident_buffer(1, 32).unwrap();
        renderer.fill_resident(1, 0, 32, 0).unwrap();
        renderer
            .target(2, 4, 2, vk::Format::R8G8B8A8_UNORM)
            .unwrap();
        let initial = vk::ClearColorValue {
            float32: [0.0, 1.0, 1.0, 1.0],
        };
        let color = vk::ClearColorValue {
            float32: [0.25, 0.0, 0.0, 0.5],
        };
        let srgb = Some(vk::Format::R8G8B8A8_SRGB);
        let mask = vk::ColorComponentFlags::R | vk::ColorComponentFlags::A;
        renderer.clear_color(2, None, initial).unwrap();
        renderer.samples_reset().unwrap();
        renderer.set_condition(Some((predicate, 0, false)));
        renderer
            .clear_color_view((2, 0, 0), srgb, None, color, mask)
            .unwrap();
        renderer.set_condition(Some((predicate, 0, true)));
        renderer
            .clear_color_view((2, 0, 0), srgb, Some([2, 0, 2, 2]), color, mask)
            .unwrap();
        renderer.set_condition(None);
        renderer
            .clear_color_view(
                (2, 0, 0),
                srgb,
                None,
                color,
                vk::ColorComponentFlags::empty(),
            )
            .unwrap();
        renderer.samples_report(predicate, 16, 0x1234).unwrap();
        let token = renderer.read_resident(1, 16, 16).unwrap();
        let bytes = renderer.read_target_bytes(2, 4).unwrap();
        for row in bytes.chunks_exact(16) {
            assert_eq!(&row[..8], &[0, 255, 255, 255].repeat(2));
            for pixel in row[8..].chunks_exact(4) {
                assert!(pixel[0].abs_diff(137) <= 1, "{pixel:?}");
                assert_eq!(&pixel[1..3], &[255, 255]);
                assert!(pixel[3].abs_diff(128) <= 1, "{pixel:?}");
            }
        }
        assert_eq!(
            renderer.take_reads(&[token])[&token],
            [0_u64, 0x1234].map(u64::to_le_bytes).concat()
        );

        renderer
            .clear_color_view((2, 0, 0), srgb, None, color, vk::ColorComponentFlags::RGBA)
            .unwrap();
        for pixel in renderer.read_target_bytes(2, 4).unwrap().chunks_exact(4) {
            assert!(pixel[0].abs_diff(137) <= 1, "{pixel:?}");
            assert_eq!(&pixel[1..3], &[0, 0]);
            assert!(pixel[3].abs_diff(128) <= 1, "{pixel:?}");
        }
    }

    #[test]
    fn masked_integer_color_clears_keep_exact_words() {
        let Some(mut renderer) = renderer() else {
            return;
        };
        let initial = [0x8123_4567_u32, 45, 0x0100_0001, 111];
        let update = [0xFEDC_BA98_u32, 1, 2, 0x8ABD_EFFF];
        let mask = vk::ColorComponentFlags::R | vk::ColorComponentFlags::A;
        for (key, format) in [vk::Format::R32G32B32A32_UINT, vk::Format::R32G32B32A32_SINT]
            .into_iter()
            .enumerate()
        {
            let key = key as u64 + 1;
            renderer.target(key, 2, 2, format).unwrap();
            renderer
                .clear_color(key, None, vk::ClearColorValue { uint32: initial })
                .unwrap();
            renderer
                .clear_color_view(
                    (key, 0, 0),
                    None,
                    None,
                    vk::ClearColorValue { uint32: update },
                    mask,
                )
                .unwrap();
            let texel = [update[0], initial[1], initial[2], update[3]]
                .map(u32::to_le_bytes)
                .concat();
            assert_eq!(
                renderer.read_target_bytes(key, 16).unwrap(),
                texel.repeat(4)
            );
        }
    }

    #[test]
    fn masked_stencil_clears_follow_conditions_without_counting_samples() {
        let Some(mut renderer) = renderer() else {
            return;
        };
        if !renderer.supports_conditions() {
            return;
        }
        let (buffer, _) = renderer.resident_buffer(1, 32).unwrap();
        renderer.fill_resident(1, 0, 32, 0).unwrap();
        renderer
            .target(2, 4, 2, vk::Format::D32_SFLOAT_S8_UINT)
            .unwrap();
        renderer
            .clear_depth_stencil(2, 0, None, Some(0.25), Some(0x49))
            .unwrap();
        renderer.samples_reset().unwrap();
        renderer.set_condition(Some((buffer, 0, false)));
        renderer
            .clear_stencil_masked(2, 0, None, 0x88, 0xff)
            .unwrap();
        renderer.set_condition(Some((buffer, 0, true)));
        renderer
            .clear_stencil_masked(2, 0, Some([2, 0, 2, 2]), 0x36, 0x0f)
            .unwrap();
        renderer.set_condition(None);
        renderer.samples_report(buffer, 16, 0x1234).unwrap();
        let token = renderer.read_resident(1, 16, 16).unwrap();
        let pixels = renderer
            .read_target_depth_stencil_plane_bytes(2, 0, 0, [0, 0, 4, 2])
            .unwrap();
        assert_eq!(&pixels[..32], 0.25_f32.to_le_bytes().repeat(8));
        assert_eq!(&pixels[32..], [0x49, 0x49, 0x46, 0x46].repeat(2));
        assert_eq!(
            renderer.take_reads(&[token])[&token],
            [0_u64, 0x1234].map(u64::to_le_bytes).concat()
        );

        renderer.set_resolution_scale(Scale::new(1, 2).unwrap());
        renderer
            .target(3, 1, 1, vk::Format::D32_SFLOAT_S8_UINT)
            .unwrap();
        renderer
            .clear_depth_stencil(3, 0, None, Some(0.5), Some(0x84))
            .unwrap();
        renderer
            .clear_stencil_masked(3, 0, None, 0x21, 0x0f)
            .unwrap();
        let pixel = renderer
            .read_target_depth_stencil_plane_bytes(3, 0, 0, [0, 0, 1, 1])
            .unwrap();
        assert_eq!(pixel, [0.5_f32.to_le_bytes().as_slice(), &[0x81]].concat());
    }
}

/// The view type an image of `shape` is created with.
fn own_view_type(shape: crate::draw::ImageShape) -> vk::ImageViewType {
    use crate::draw::ImageShape;
    match shape {
        ImageShape::D1 { layers } if layers > 1 => vk::ImageViewType::TYPE_1D_ARRAY,
        ImageShape::D1 { .. } => vk::ImageViewType::TYPE_1D,
        ImageShape::D2 { layers } if layers > 1 => vk::ImageViewType::TYPE_2D_ARRAY,
        ImageShape::D2 { .. } => vk::ImageViewType::TYPE_2D,
        ImageShape::D3 { .. } => vk::ImageViewType::TYPE_3D,
        ImageShape::Cube { layers } if layers > 6 => vk::ImageViewType::CUBE_ARRAY,
        ImageShape::Cube { .. } => vk::ImageViewType::CUBE,
    }
}

/// The parameters a view was created with.
#[derive(Clone, Copy, Debug)]
struct ViewInfo {
    image: vk::Image,
    view_type: vk::ImageViewType,
    format: vk::Format,
    range: vk::ImageSubresourceRange,
}

impl ViewInfo {
    fn of(info: &vk::ImageViewCreateInfo) -> Self {
        Self {
            image: info.image,
            view_type: info.view_type,
            format: info.format,
            range: info.subresource_range,
        }
    }
}

/// The built-in border colour nearest `rgba` (float bits): transparent
/// black, opaque black or opaque white. Other colours (custom border
/// colours are not enabled) take the nearest of these by alpha and
/// brightness.
fn border_color(rgba: [u32; 4], integer: bool) -> vk::BorderColor {
    use vk::BorderColor as Color;
    let [transparent_black, opaque_black, opaque_white] = if integer {
        [
            Color::INT_TRANSPARENT_BLACK,
            Color::INT_OPAQUE_BLACK,
            Color::INT_OPAQUE_WHITE,
        ]
    } else {
        [
            Color::FLOAT_TRANSPARENT_BLACK,
            Color::FLOAT_OPAQUE_BLACK,
            Color::FLOAT_OPAQUE_WHITE,
        ]
    };
    let [r, g, b, a] = rgba.map(f32::from_bits);
    if a < 0.5 {
        transparent_black
    } else if (r + g + b) / 3.0 < 0.5 {
        opaque_black
    } else {
        opaque_white
    }
}

/// The sRGB or UNORM counterpart of `format` (the pairs NVN texture views
/// reinterpret between), if it has one.
/// The bytes of one 4x4 block of a BC (block-compressed) format.
pub fn bc_block_bytes(format: vk::Format) -> Option<u32> {
    use vk::Format as F;
    match format {
        F::BC1_RGB_UNORM_BLOCK
        | F::BC1_RGB_SRGB_BLOCK
        | F::BC1_RGBA_UNORM_BLOCK
        | F::BC1_RGBA_SRGB_BLOCK
        | F::BC4_UNORM_BLOCK
        | F::BC4_SNORM_BLOCK => Some(8),
        F::BC2_UNORM_BLOCK
        | F::BC2_SRGB_BLOCK
        | F::BC3_UNORM_BLOCK
        | F::BC3_SRGB_BLOCK
        | F::BC5_UNORM_BLOCK
        | F::BC5_SNORM_BLOCK
        | F::BC6H_UFLOAT_BLOCK
        | F::BC6H_SFLOAT_BLOCK
        | F::BC7_UNORM_BLOCK
        | F::BC7_SRGB_BLOCK => Some(16),
        _ => None,
    }
}

/// UNORM formats and their sRGB-encoded twins.
const SRGB_PAIRS: [(vk::Format, vk::Format); 6] = [
    (vk::Format::R8G8B8A8_UNORM, vk::Format::R8G8B8A8_SRGB),
    (vk::Format::B8G8R8A8_UNORM, vk::Format::B8G8R8A8_SRGB),
    (
        vk::Format::BC1_RGBA_UNORM_BLOCK,
        vk::Format::BC1_RGBA_SRGB_BLOCK,
    ),
    (vk::Format::BC2_UNORM_BLOCK, vk::Format::BC2_SRGB_BLOCK),
    (vk::Format::BC3_UNORM_BLOCK, vk::Format::BC3_SRGB_BLOCK),
    (vk::Format::BC7_UNORM_BLOCK, vk::Format::BC7_SRGB_BLOCK),
];

/// Whether `format` is one of the sRGB-encoded twins.
fn is_srgb(format: vk::Format) -> bool {
    astc_ldr_format_index(format).is_some_and(|index| index % 2 == 1)
        || SRGB_PAIRS.iter().any(|&(_, srgb)| srgb == format)
}

pub fn srgb_twin(format: vk::Format) -> Option<vk::Format> {
    if let Some(twin) = astc_srgb_twin(format) {
        return Some(twin);
    }
    SRGB_PAIRS.iter().find_map(|&(unorm, srgb)| {
        if format == unorm {
            Some(srgb)
        } else if format == srgb {
            Some(unorm)
        } else {
            None
        }
    })
}

fn attachment_format(format: vk::Format, view: Option<vk::Format>) -> vk::Format {
    view.filter(|&view| srgb_twin(format) == Some(view))
        .unwrap_or(format)
}

fn astc_srgb_twin(format: vk::Format) -> Option<vk::Format> {
    astc_ldr_format_index(format)
        .map(|index| vk::Format::from_raw(vk::Format::ASTC_4X4_UNORM_BLOCK.as_raw() + (index ^ 1)))
}

fn astc_ldr_format_index(format: vk::Format) -> Option<i32> {
    // Vulkan assigns consecutive UNORM/sRGB pairs to the ASTC LDR formats.
    (vk::Format::ASTC_4X4_UNORM_BLOCK.as_raw()..=vk::Format::ASTC_12X12_SRGB_BLOCK.as_raw())
        .contains(&format.as_raw())
        .then(|| format.as_raw() - vk::Format::ASTC_4X4_UNORM_BLOCK.as_raw())
}

/// The float format of the same texel size for a 32-bit integer format.
/// Translated shaders declare every texture with a float sampled type and
/// reinterpret the bits they read, as the console's texel fetch returns raw
/// 32-bit values; reading an integer format through a float sampler is
/// undefined in Vulkan (RADV converts the value, so light-list indices came
/// back as garbage). A float view of the same bits keeps them intact.
pub(crate) fn bit_preserving_float(format: vk::Format) -> vk::Format {
    use vk::Format as F;
    match format {
        F::R32_UINT | F::R32_SINT => F::R32_SFLOAT,
        F::R32G32_UINT | F::R32G32_SINT => F::R32G32_SFLOAT,
        F::R32G32B32_UINT | F::R32G32B32_SINT => F::R32G32B32_SFLOAT,
        F::R32G32B32A32_UINT | F::R32G32B32A32_SINT => F::R32G32B32A32_SFLOAT,
        other => other,
    }
}

/// Bytes per texel of the uncompressed formats buffer textures use (0 for
/// others).
pub(crate) fn format_bytes(format: vk::Format) -> u32 {
    use vk::Format as F;
    match format {
        F::R8_UNORM | F::R8_SNORM | F::R8_UINT | F::R8_SINT => 1,
        F::R8G8_UNORM | F::R8G8_SNORM | F::R8G8_UINT | F::R8G8_SINT => 2,
        F::R16_UNORM | F::R16_SNORM | F::R16_UINT | F::R16_SINT | F::R16_SFLOAT => 2,
        F::R8G8B8A8_UNORM
        | F::R8G8B8A8_SNORM
        | F::R8G8B8A8_UINT
        | F::R8G8B8A8_SINT
        | F::R8G8B8A8_SRGB
        | F::B8G8R8A8_UNORM
        | F::A2B10G10R10_UNORM_PACK32
        | F::B10G11R11_UFLOAT_PACK32 => 4,
        F::R16G16_UNORM | F::R16G16_SNORM | F::R16G16_UINT | F::R16G16_SINT | F::R16G16_SFLOAT => 4,
        F::R32_UINT | F::R32_SINT | F::R32_SFLOAT => 4,
        F::R16G16B16A16_UNORM
        | F::R16G16B16A16_SNORM
        | F::R16G16B16A16_UINT
        | F::R16G16B16A16_SINT
        | F::R16G16B16A16_SFLOAT => 8,
        F::R32G32_UINT | F::R32G32_SINT | F::R32G32_SFLOAT => 8,
        F::R32G32B32_UINT | F::R32G32B32_SINT | F::R32G32B32_SFLOAT => 12,
        F::R32G32B32A32_UINT | F::R32G32B32A32_SINT | F::R32G32B32A32_SFLOAT => 16,
        _ => 0,
    }
}

/// The rectangle `[x0, y0, x1, y1]` that shows a `source`-sized frame in a
/// `window`-sized surface at its own aspect ratio: as large as fits,
/// centred (letterboxed or pillarboxed).
/// Keys from here up are the renderer's own staging targets (never scaled).
const STAGING_KEYS: u64 = u64::MAX - 0x10;

/// How a presented frame fills the window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScalingMode {
    /// Largest size with the frame's aspect ratio, centred (black bars).
    Fit,
    /// The whole window, whatever its aspect ratio.
    Stretch,
    /// The largest whole multiple of the frame's size that fits, centred
    /// (sharp pixels); Fit when even one does not fit.
    Integer,
}

/// ECHORA_SCALING_MODE (fit, stretch or integer; default fit) and
/// ECHORA_SCALING_FILTER (linear or nearest; default linear).
fn output_options() -> &'static (ScalingMode, vk::Filter) {
    static OPTIONS: std::sync::OnceLock<(ScalingMode, vk::Filter)> = std::sync::OnceLock::new();
    OPTIONS.get_or_init(|| {
        let var = |name| std::env::var(name).unwrap_or_default().to_ascii_lowercase();
        let mode = match var("ECHORA_SCALING_MODE").as_str() {
            "stretch" => ScalingMode::Stretch,
            "integer" => ScalingMode::Integer,
            _ => ScalingMode::Fit,
        };
        let filter = match var("ECHORA_SCALING_FILTER").as_str() {
            "nearest" => vk::Filter::NEAREST,
            _ => vk::Filter::LINEAR,
        };
        (mode, filter)
    })
}

/// The window rect (x0, y0, x1, y1) a `source`-sized frame is shown in.
pub(crate) fn present_rect(mode: ScalingMode, source: (u32, u32), window: (u32, u32)) -> [i32; 4] {
    match mode {
        ScalingMode::Fit => fit_rect(source, window),
        ScalingMode::Stretch => [0, 0, window.0 as i32, window.1 as i32],
        ScalingMode::Integer => {
            let factor = (window.0 / source.0.max(1)).min(window.1 / source.1.max(1));
            if factor == 0 {
                return fit_rect(source, window);
            }
            let (width, height) = (source.0 * factor, source.1 * factor);
            let (x, y) = ((window.0 - width) / 2, (window.1 - height) / 2);
            [x as i32, y as i32, (x + width) as i32, (y + height) as i32]
        }
    }
}

pub(crate) fn fit_rect(source: (u32, u32), window: (u32, u32)) -> [i32; 4] {
    let (sw, sh) = (u64::from(source.0.max(1)), u64::from(source.1.max(1)));
    let (ww, wh) = (u64::from(window.0), u64::from(window.1));
    // Fit the width unless that makes the frame taller than the window.
    let (w, h) = if ww * sh <= wh * sw {
        (ww, (ww * sh + sw / 2) / sw)
    } else {
        ((wh * sw + sh / 2) / sh, wh)
    };
    let (x, y) = ((ww - w) / 2, (wh - h) / 2);
    [x as i32, y as i32, (x + w) as i32, (y + h) as i32]
}

#[cfg(test)]
mod fit_tests {
    use super::{ScalingMode, fit_rect, present_rect};

    #[test]
    fn scales_are_rational() {
        use super::Scale;
        let qhd = Scale::new(1440, 1080).unwrap();
        assert_eq!(qhd.terms(), (4, 3));
        assert_eq!((qhd.apply(1920), qhd.apply(1080)), (2560, 1440));
        // Edges scale, so rects that meet still meet.
        assert_eq!(qhd.rect([0, 0, 960, 540]), [0, 0, 1280, 720]);
        assert_eq!(qhd.rect([1, 1, 2, 2]), [1, 1, 3, 3]);
        assert_eq!(Scale::new(2160, 1080).unwrap().terms(), (2, 1));
        assert_eq!(Scale::new(720, 1080).unwrap().apply(1920), 1280);
        assert!(Scale::new(500, 1080).is_none());
        assert!(Scale::new(5000, 1080).is_none());
        assert_eq!(Scale::new(u32::MAX, u32::MAX), Some(Scale::ONE));
        assert!(Scale::new(u32::MAX, 1).is_none());
        assert!(Scale::new(1, u32::MAX).is_none());
        assert!(Scale::new(0, 0).is_none());

        let fourfold = Scale::new(4, 1).unwrap();
        assert_eq!(fourfold.apply(u32::MAX / 4 + 1), u32::MAX);
        assert_eq!(
            fourfold.rect([-1_000_000_000, 0, 2_000_000_000, 1]),
            [i32::MIN, 0, i32::MAX, 4]
        );
    }

    #[test]
    fn frames_keep_their_aspect_ratio() {
        // Same shape: the whole window.
        assert_eq!(fit_rect((1920, 1080), (2560, 1440)), [0, 0, 2560, 1440]);
        // Wider window: bars at the sides.
        assert_eq!(fit_rect((1920, 1080), (2560, 1080)), [320, 0, 2240, 1080]);
        // Taller window: bars above and below.
        assert_eq!(fit_rect((1920, 1080), (1280, 1024)), [0, 152, 1280, 872]);
        let integer = |source, window| present_rect(ScalingMode::Integer, source, window);
        assert_eq!(integer((1920, 1080), (3840, 2160)), [0, 0, 3840, 2160]);
        assert_eq!(integer((1920, 1080), (2560, 1440)), [320, 180, 2240, 1260]);
        assert_eq!(integer((1920, 1080), (1280, 720)), [0, 0, 1280, 720]);
        assert_eq!(
            present_rect(ScalingMode::Stretch, (1920, 1080), (3440, 1440)),
            [0, 0, 3440, 1440]
        );
        assert_eq!(fit_rect((1920, 1080), (0, 0)), [0, 0, 0, 0]);
    }
}

#[cfg(test)]
mod rescaling_tests {
    use super::*;

    #[test]
    fn storage_targets_keep_the_guest_extent() {
        let Ok(gpu) = Gpu::new() else {
            return;
        };
        let mut renderer = Renderer::new(gpu).unwrap();
        let shape = ImageShape::D2 { layers: 1 };
        let format = vk::Format::R8G8B8A8_UNORM;
        renderer.set_resolution_scale(Scale::new(2, 1).unwrap());

        renderer
            .guest_sized_target(7, 16, 8, shape, format, 1)
            .unwrap();
        assert_eq!(renderer.targets[&7].extent(0), (16, 8));
        assert_eq!(renderer.descaled, 0);

        renderer.shaped_target(8, 16, 8, shape, format, 1).unwrap();
        assert_eq!(renderer.targets[&8].extent(0), (32, 16));
        renderer
            .clear_color(
                8,
                None,
                vk::ClearColorValue {
                    float32: [1.0, 0.0, 0.0, 1.0],
                },
            )
            .unwrap();
        renderer
            .guest_sized_target(8, 16, 8, shape, format, 1)
            .unwrap();
        assert_eq!(renderer.targets[&8].extent(0), (16, 8));
        assert_eq!(renderer.descaled, 1);
        let pixels = renderer.read_target_bytes(8, 4).unwrap();
        assert_eq!(pixels.len(), 16 * 8 * 4);
        assert!(
            pixels
                .chunks_exact(4)
                .all(|texel| texel == [255, 0, 0, 255])
        );
    }

    #[test]
    fn reusing_a_target_keeps_unscaled_only_for_the_marked_description() {
        let Ok(gpu) = Gpu::new() else {
            return;
        };
        let mut renderer = Renderer::new(gpu).unwrap();
        let format = vk::Format::R8G8B8A8_UNORM;
        renderer.set_resolution_scale(Scale::new(2, 1).unwrap());

        assert_eq!(
            renderer.target(7, 16, 8, format).unwrap().extent(0),
            (32, 16)
        );
        renderer.keep_unscaled(7).unwrap();
        renderer.move_target(7, 8).unwrap();
        assert_eq!(
            renderer.target(7, 32, 8, format).unwrap().extent(0),
            (64, 16)
        );
        assert_eq!(renderer.existing_target(8).unwrap().extent(0), (16, 8));
        assert_eq!(
            renderer.target(7, 16, 8, format).unwrap().extent(0),
            (16, 8)
        );

        renderer.keep_unscaled(9).unwrap();
        assert_eq!(
            renderer.target(9, 16, 8, format).unwrap().extent(0),
            (16, 8)
        );
        assert_eq!(
            renderer.target(9, 32, 8, format).unwrap().extent(0),
            (64, 16)
        );
    }

    #[test]
    fn chained_target_copies_see_previous_writes() {
        let Ok(gpu) = Gpu::new() else {
            return;
        };
        let mut renderer = Renderer::new(gpu).unwrap();
        renderer.set_resolution_scale(Scale::new(3, 2).unwrap());
        renderer.keep_unscaled(8).unwrap();
        for key in [7, 8, 9] {
            renderer
                .target(key, 16, 8, vk::Format::R8G8B8A8_UNORM)
                .unwrap();
            let rgba = if key == 7 {
                [1.0, 0.0, 0.0, 1.0]
            } else {
                [0.0, 0.0, 1.0, 1.0]
            };
            renderer
                .clear_color(key, None, vk::ClearColorValue { float32: rgba })
                .unwrap();
        }

        let rect = [0, 0, 16, 8];
        renderer.copy_target(7, rect, 8, rect).unwrap();
        renderer.copy_target(8, rect, 9, rect).unwrap();
        renderer.copy_target(9, rect, 7, rect).unwrap();
        let (pixels, width, height) = renderer.read_target_scaled(7).unwrap();
        assert_eq!((width, height), (24, 12));
        assert!(pixels.iter().all(|pixel| *pixel == [1.0, 0.0, 0.0, 1.0]));
    }

    #[test]
    fn descales_a_replaced_attachment_in_command_order() {
        let Ok(gpu) = Gpu::new() else {
            return;
        };
        let mut renderer = Renderer::new(gpu).unwrap();
        let scale = Scale::new(2, 1).unwrap();
        let format = vk::Format::R8G8B8A8_UINT;
        renderer.set_resolution_scale(scale);
        renderer.keep_unscaled(7).unwrap();
        renderer.keep_unscaled(8).unwrap();
        renderer.target(8, 16, 8, format).unwrap();
        assert_eq!(renderer.target(9, 16, 8, format).unwrap().scale, scale);
        let source = renderer.targets[&9].image;
        let left = [200, 25, 41, 255];
        let right = [8, 65, 79, 255];
        renderer
            .clear_color(9, None, vk::ClearColorValue { uint32: left })
            .unwrap();
        renderer
            .clear_color(9, Some([8, 0, 8, 8]), vk::ClearColorValue { uint32: right })
            .unwrap();
        renderer.move_target(9, 7).unwrap();

        let submissions = renderer.flushes;
        assert_eq!(
            renderer.attachment_scale(&[0, 7, 8], None).unwrap(),
            Scale::ONE
        );
        assert_eq!(renderer.targets[&7].extent(0), (16, 8));
        assert_eq!(renderer.flushes, submissions);
        assert!(
            renderer
                .slot
                .retired_uploads
                .iter()
                .any(|old| old.image == source)
        );

        let bytes = renderer.read_target_bytes(7, 4).unwrap();
        assert_eq!(bytes.len(), 16 * 8 * 4);
        for row in bytes.chunks_exact(16 * 4) {
            assert_eq!(&row[..8 * 4], &left.map(|v| v as u8).repeat(8));
            assert_eq!(&row[8 * 4..], &right.map(|v| v as u8).repeat(8));
        }
    }
}

/// The unsigned word format with `format`'s texel size, which a storage
/// image of it can be viewed as (None: no such size, or compressed).
pub fn uint_alias(format: vk::Format) -> Option<vk::Format> {
    use vk::Format as F;
    let bytes = match format {
        F::R8_UNORM | F::R8_SNORM | F::R8_UINT | F::R8_SINT => 1,
        F::R16_SFLOAT | F::R16_UNORM | F::R16_SNORM | F::R16_UINT | F::R16_SINT => 2,
        F::R8G8_UNORM | F::R8G8_SNORM | F::R8G8_UINT | F::R8G8_SINT => 2,
        F::R32_SFLOAT | F::R32_UINT | F::R32_SINT => 4,
        F::R16G16_SFLOAT | F::R16G16_UNORM | F::R16G16_SNORM | F::R16G16_UINT => 4,
        F::R16G16_SINT => 4,
        F::R8G8B8A8_UNORM | F::R8G8B8A8_SNORM | F::R8G8B8A8_UINT | F::R8G8B8A8_SINT => 4,
        F::R8G8B8A8_SRGB | F::B8G8R8A8_UNORM | F::B8G8R8A8_SRGB => 4,
        F::A2B10G10R10_UNORM_PACK32 | F::A2B10G10R10_UINT_PACK32 => 4,
        F::B10G11R11_UFLOAT_PACK32 | F::E5B9G9R9_UFLOAT_PACK32 => 4,
        F::R32G32_SFLOAT | F::R32G32_UINT | F::R32G32_SINT => 8,
        F::R16G16B16A16_SFLOAT | F::R16G16B16A16_UNORM | F::R16G16B16A16_SNORM => 8,
        F::R16G16B16A16_UINT | F::R16G16B16A16_SINT => 8,
        F::R32G32B32A32_SFLOAT | F::R32G32B32A32_UINT | F::R32G32B32A32_SINT => 16,
        _ => return None,
    };
    Some(match bytes {
        1 => F::R8_UINT,
        2 => F::R16_UINT,
        4 => F::R32_UINT,
        8 => F::R32G32_UINT,
        _ => F::R32G32B32A32_UINT,
    })
}

/// The format a storage image descriptor declares (`Descriptor::flags` bits
/// 8-11: the recompiler's ImageFormat), None when typeless.
pub(crate) fn declared_image_format(flags: u32) -> Option<vk::Format> {
    use vk::Format as F;
    Some(match flags >> 8 & 0xf {
        1 => F::R8_UINT,
        2 => F::R8_SINT,
        3 => F::R16_UINT,
        4 => F::R16_SINT,
        5 => F::R32_UINT,
        6 => F::R32G32_UINT,
        7 => F::R32G32B32A32_UINT,
        _ => return None,
    })
}

/// Whether texels of `format` are integers (read through int or uint
/// images), as SpecMeaning::IntegerFormat asks.
pub fn integer_format(format: vk::Format) -> bool {
    sampled_texture_type(format) != crate::shader::SampleType::Float
}

/// SPIR-V sampled type required by a texture format.
pub fn sampled_texture_type(format: vk::Format) -> crate::shader::SampleType {
    use crate::shader::SampleType;
    use vk::Format as F;
    match format {
        F::R8_UINT
        | F::R8G8_UINT
        | F::R8G8B8A8_UINT
        | F::R16_UINT
        | F::R16G16_UINT
        | F::R16G16B16A16_UINT
        | F::R32_UINT
        | F::R32G32_UINT
        | F::R32G32B32_UINT
        | F::R32G32B32A32_UINT
        | F::A2B10G10R10_UINT_PACK32
        | F::S8_UINT => SampleType::Unsigned,
        F::R8_SINT
        | F::R8G8_SINT
        | F::R8G8B8A8_SINT
        | F::R16_SINT
        | F::R16G16_SINT
        | F::R16G16B16A16_SINT
        | F::R32_SINT
        | F::R32G32_SINT
        | F::R32G32B32_SINT
        | F::R32G32B32A32_SINT => SampleType::Signed,
        _ => SampleType::Float,
    }
}

/// `usage`, plus device addresses when the device supports them.
fn addressed_usage(usage: vk::BufferUsageFlags, addressable: bool) -> vk::BufferUsageFlags {
    if addressable {
        usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
    } else {
        usage
    }
}

/// ECHORA_HANG_MARKERS (debugging): each labelled piece of work is preceded
/// by a VK_AMD_buffer_marker write of its number, done when the GPU has
/// finished what came before; a stalled wait reads the last number written
/// and names the work after it.
struct HangMarkers {
    buffer: HostBuffer,
    next: u32,
    labels: std::collections::VecDeque<(u32, String)>,
    path: std::ffi::OsString,
}

impl HangMarkers {
    /// Write where the GPU stopped: the work after the last marker passed
    /// (the one it is in) and its neighbours.
    fn report(&self, serial: u64) {
        let passed = u32::from_le_bytes(self.buffer.read_bytes(4).try_into().unwrap_or([0; 4]));
        let mut text = format!(
            "GPU stalled over 2 s in submission {serial}: last marker passed {passed} of {}\n",
            self.next
        );
        for (number, label) in &self.labels {
            if number + 5 >= passed && *number <= passed + 20 {
                let mark = if *number == passed { "-> " } else { "   " };
                text.push_str(&format!("{mark}{number} {label}\n"));
            }
        }
        eprintln!("{}", text.lines().next().unwrap_or_default());
        let _ = std::fs::write(&self.path, text);
    }
}
