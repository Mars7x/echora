//! Host GPU execution of NVN work (ECHORA_GPU=1): bound programs are
//! translated (`echora_gpu::shader`), their
//! bindings resolved through the inferred NVN2 binding map (docs/roadmap.md),
//! and clears and draws run through `echora_gpu`.
//!
//! Fixed-function state comes from record-time snapshots with NVN enum
//! values inferred from GL order (see compare_op, blend_factor, blend_op);
//! samplers keep only their filters and wrap modes;
//! textures upload every mip level of their first layer (ASTC decoded on
//! the CPU); the driver constant buffers 0..2 read as
//! zeros.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use echora_gpu::ash::vk;
use echora_gpu::draw::{
    Blend, Draw, Fixed, ImageShape, Indices, MeshGroups, Program, Sampler, StencilFace,
    TextureSource, UploadRegion, VertexAttribute, VertexStream,
};
use echora_gpu::present::Presenter;
use echora_gpu::renderer::Renderer;
use echora_gpu::shader::{self, DescriptorKind, MeshParameters, Parameters, Stage, Translated};

use super::{Command, Guest, Image, Nvn};

/// A built page table (see `Nvn::page_table`).
pub struct PageBuild {
    first: u64,
    entries: Vec<u64>,
    pools: HashMap<(u64, u64, u64, u64), ((u64, u64, usize), bool)>,
}

pub struct Backend {
    pub renderer: Renderer,
    /// Translations by stage key and the parameters that shape them.
    translations: HashMap<TranslationKey, Result<Arc<Translated>, String>>,
    /// Compute stage keys translated with raw storage pointers
    /// (`Parameters::native_pointers`).
    native_programs: std::collections::HashSet<u64>,
    /// The GPU page table (ECHORA_PAGE_TABLE) and the pool set it was built
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
    /// Linked programs by (vertex key, fragment key).
    programs: HashMap<(u64, u64), Program>,
    /// Counts of work skipped or approximated, by reason.
    pub notes: BTreeMap<String, u64>,
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
    textures: HashMap<u64, (u64, TextureSource, std::ops::Range<u64>)>,
    /// The present during which each cached texture's sample last matched:
    /// within one frame a texture is sampled once, not on every draw.
    texture_checked: HashMap<u64, u64>,
    /// The largest image extent the host GPU supports.
    max_extent: u32,
    /// Uploads an application flush overlapped, with a hash of all their
    /// raw storage: if a later use finds the storage unchanged, the upload
    /// is kept instead of untiled again.
    /// Byte ranges (from the storage start) of cached uploads that
    /// application flushes overlapped: re-checked before the next use.
    dirty: HashMap<u64, Vec<std::ops::Range<u64>>>,
    /// Hash of the whole raw storage of each upload.
    /// Hashes of each upload's raw storage in STORAGE_CHUNK pieces, so a
    /// flush re-checks only the pieces it overlapped.
    storage_hashes: HashMap<u64, Vec<u64>>,
    /// For uploads over DENSE_SAMPLE_ABOVE bytes: a hash of 4096 pieces
    /// spread over the storage, checked instead of every flushed byte.
    dense_samples: HashMap<u64, u64>,
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
    /// Consecutive whole-storage checks each shadowed texture passed
    /// unchanged: a texture quiet for long is compared in full only every
    /// 2nd, then 4th present, with a 256-chunk sample in between (see
    /// `guest_texture`; ECHORA_EXACT_TEXTURE_CHECKS=1 compares every
    /// present).
    shadow_quiet: HashMap<u64, u32>,
    /// Dispatch results not yet stored to guest memory: (renderer read
    /// token, guest range, binding, program, GPU address). They are
    /// stored at the end of the submission, or before work that may read
    /// that memory (`resolve_storage`): one wait for several dispatches.
    pending_storage: Vec<(u64, std::ops::Range<u64>, u32, u64, u64)>,
    /// Guest ranges compute shaders wrote in place (storage buffers bound
    /// over imported guest memory) since the GPU last finished: a CPU read
    /// of one waits for the GPU first (see `resolve_storage`).
    in_place_writes: Vec<std::ops::Range<u64>>,
    /// A program storing through the page table was recorded since the
    /// last wait for it: its stores land in guest memory or a resident
    /// pool's copy with no range recorded (see `resolve_gpu_written`).
    page_table_writes: bool,
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
    /// Bumped whenever a pool is shared (or shared again): the page table
    /// is rebuilt after it changes.
    share_generation: usize,
    /// Render targets by their texture's storage address, and the
    /// (renderer target changes, texture epoch) it was built at: texture
    /// lookups scanned every target twice per binding.
    target_index: HashMap<u64, Vec<u64>>,
    /// The render targets with more than one level, in the same index (a
    /// texture over one of their levels resolves to them).
    target_chains: Vec<u64>,
    target_index_stamp: Option<(u64, u64)>,
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

impl Backend {
    /// A backend when ECHORA_GPU is set, the recompiler is linked and a
    /// device with shader objects exists.
    pub fn create() -> Result<Option<Self>, String> {
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
        // are kept there for the next run.
        if let Some(directory) = super::debug_var_os("ECHORA_SHADER_CACHE") {
            renderer.set_shader_cache(directory.into());
        }
        if echora_gpu::renderer::Scale::env_override().is_none()
            && let Some((numerator, denominator)) = crate::settings::render_scale()
        {
            match echora_gpu::renderer::Scale::new(numerator, denominator) {
                Some(scale) => renderer.set_resolution_scale(scale),
                None => eprintln!(
                    "settings: render scale {numerator}/{denominator} is outside 1/2..4; using 1"
                ),
            }
        }
        Ok(Some(Self {
            max_extent: renderer.max_image_extent(),
            renderer,
            translations: HashMap::new(),
            native_programs: Default::default(),
            page_table: None,
            page_build: None,
            loop_traces: HashMap::new(),
            vertex_traces: HashMap::new(),
            vertex_trace_addresses: Vec::new(),
            programs: HashMap::new(),
            notes: BTreeMap::new(),
            traced: 0,
            probed: 0,
            watched_pixel: HashMap::new(),
            presenter: None,
            textures: HashMap::new(),
            texture_checked: HashMap::new(),
            dirty: HashMap::new(),
            storage_hashes: HashMap::new(),
            dense_samples: HashMap::new(),
            copied_levels: HashMap::new(),
            shadows: HashMap::new(),
            shadow_quiet: HashMap::new(),
            pending_storage: Vec::new(),
            in_place_writes: Vec::new(),
            page_table_writes: false,
            page_table_wrote: false,
            resident: HashMap::new(),
            samples_reports: std::collections::HashSet::new(),
            shared_pools: HashMap::new(),
            shared_pieces: HashMap::new(),
            share_generation: 0,
            target_index: HashMap::new(),
            target_chains: Vec::new(),
            target_index_stamp: None,
        }))
    }

    /// Add `amount` to a counter in the notes (named "stat: ..."), for
    /// the run report; nothing is logged.
    pub(super) fn stat(&mut self, name: &'static str, amount: u64) {
        // Called several times per draw: no allocation once a name exists.
        match self.notes.get_mut(name) {
            Some(total) => *total += amount,
            None => {
                self.notes.insert(name.into(), amount);
            }
        }
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

/// The key bits marking a program's vertex-traced variant
/// (ECHORA_VERTEX_TRACE_BASE).
const TRACED_VARIANT: u64 = 0x7472_6163_6564_0000;

/// Stage keys (vertex, fragment) and the stages' translations.
pub(super) type BoundProgram = ((u64, u64), Vec<Arc<Translated>>);
/// Stage key, binding base, previous stage's stores, generic input types,
/// compute workgroup size and shared memory, and the task/mesh state.
type TranslationKey = (
    u64,
    [u32; 3],
    Option<[u64; 8]>,
    [u8; 32],
    [u32; 5],
    Option<MeshParameters>,
    bool,
);

/// Shader::AttributeType of each vertex attribute's format: integer formats
/// must reach integer shader inputs.
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
    /// Restart the GPU statistics ("stat: " notes; ECHORA_PROFILE_FROM).
    pub(crate) fn reset_gpu_stats(&mut self) {
        if let Some(backend) = self.gpu.as_mut() {
            backend.notes.retain(|name, _| !name.starts_with("stat: "));
        }
    }

    /// The GPU backend, created on first use when enabled.
    pub(super) fn gpu(&mut self) -> Option<&mut Backend> {
        if !self.gpu_checked {
            self.gpu_checked = true;
            match Backend::create() {
                Ok(backend) => self.gpu = backend,
                Err(error) => {
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

    /// Drop cached textures whose storage overlaps guest `range`, or the
    /// texture object `texture` itself: the application wrote it.
    pub(super) fn gpu_invalidate(&mut self, range: std::ops::Range<u64>, texture: Option<u64>) {
        // Resolved bindings (`binding_memo`) change only with a texture's
        // bytes: one named, or one under the range. Descriptors are
        // registered, not read from the range.
        self.flush_epoch += 1;
        if texture.is_some() {
            self.binding_serial += 1;
        }
        if let Some(backend) = self.gpu.as_mut() {
            for pool in backend.resident.values_mut() {
                let start = range.start.max(pool.base);
                let end = range.end.min(pool.base + pool.size);
                if start < end {
                    // The CPU's bytes are newer there than the GPU's.
                    remove_range(&mut pool.gpu_dirty, start..end);
                    add_range(&mut pool.guest_dirty, start..end);
                }
            }
            let mut touched = Vec::new();
            backend.textures.retain(|key, (_, _, storage)| {
                // WriteTexels names the texture: its contents changed.
                if Some(*key) == texture {
                    return false;
                }
                if storage.end > range.start && range.end > storage.start {
                    touched.push((
                        *key,
                        range.start.max(storage.start) - storage.start
                            ..range.end.min(storage.end) - storage.start,
                    ));
                }
                true
            });
            backend.stat("stat: textures invalidated", touched.len() as u64);
            if !touched.is_empty() {
                self.binding_serial += 1;
            }
            let backend = self.gpu.as_mut().unwrap();
            for (key, overlap) in touched {
                backend.texture_checked.remove(&key);
                backend.dirty.entry(key).or_default().push(overlap);
            }
        }
    }

    /// ClearColor on the GPU: `color` as the four captured 32-bit words.
    pub(super) fn gpu_clear_color(
        &mut self,
        texture: u64,
        level: u32,
        layer: u32,
        color: [u32; 4],
        rect: Option<[i32; 4]>,
    ) -> Result<(), String> {
        // A texture object over one level of another target clears that
        // target's level, as draws into it render there.
        let (texture, level) = match self.resolve_target(texture) {
            Some((key, base)) if key != texture => (key, base + level),
            _ => (texture, level),
        };
        self.ensure_render_target(texture)?;
        let backend = self.gpu.as_mut().unwrap();
        backend.renderer.defined(texture);
        let value = vk::ClearColorValue {
            float32: color.map(f32::from_bits),
        };
        backend
            .renderer
            .clear_color_level(texture, level, layer, rect, value)
    }

    /// ClearDepthStencil on the GPU: the masked-off aspects keep their
    /// contents (partial stencil masks clear all bits; not modeled).
    pub(super) fn gpu_clear_depth(
        &mut self,
        texture: u64,
        depth: Option<f32>,
        stencil: Option<u32>,
    ) -> Result<(), String> {
        self.ensure_render_target(texture)?;
        let scissor = self.render.scissor;
        let layer = self.render.depth_layer;
        let backend = self.gpu.as_mut().unwrap();
        backend.renderer.defined(texture);
        backend
            .renderer
            .clear_depth_stencil(texture, layer, scissor, depth, stencil)
    }

    /// Release what was cached for texture object `texture`, finalized:
    /// its upload (CPU texels and GPU image), saved storage copy and hashes.
    /// Gameplay streams textures, and these grew without bound. Render
    /// targets stay (what the GPU rendered stays with its memory).
    pub(super) fn gpu_forget_texture(&mut self, texture: u64) {
        let Some(backend) = self.gpu.as_mut() else {
            return;
        };
        backend.textures.remove(&texture);
        backend.texture_checked.remove(&texture);
        backend.shadows.remove(&texture);
        backend.storage_hashes.remove(&texture);
        backend.dense_samples.remove(&texture);
        backend.dirty.remove(&texture);
        backend.renderer.forget_upload(texture);
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
        if self.depth_traced(texture) {
            eprintln!(
                "depth trace present {} texture {texture:x} now {identity:x?} (was {previous:x?}): targets move",
                self.presented_frames
            );
        }
        if let Some(backend) = self.gpu.as_mut() {
            backend.textures.remove(&texture);
            backend.texture_checked.remove(&texture);
            let renderer = &mut backend.renderer;
            let retired = super::retired_texture(previous);
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

    /// ReadTexels of a texture the GPU rendered: guest memory holds only
    /// what the CPU wrote there, so the region comes from the host target
    /// (its own format's bytes, rows as the guest stores them). False when
    /// the texture is not a rendered single-level region this can serve.
    pub(super) fn gpu_read_texels(
        &mut self,
        core: &mut Guest,
        texture: u64,
        region: [u32; 6],
        destination: u64,
    ) -> Result<bool, String> {
        if self.gpu.is_none() {
            return Ok(false);
        }
        let Some((key, 0)) = self.resolve_target(texture) else {
            return Ok(false);
        };
        let format = self
            .object(texture, "Texture")?
            .builder
            .get("Format")
            .map_or(0, |value| value.x[0] as u32);
        let Some(info) = crate::nvn_format::info(format) else {
            return Ok(false);
        };
        let backend = self.gpu.as_mut().unwrap();
        let target = backend.renderer.existing_target(key).unwrap();
        let (width, height) = (target.width, target.height);
        let [x, y, z, columns, rows, depth] = region;
        if info.block_width != 1
            || info.block_height != 1
            || vk_format(format) != Some(target.format)
            || z != 0
            || depth > 1
            || x + columns > width
            || y + rows > height
        {
            backend.note(format!(
                "texels read from guest memory for rendered {} {width}x{height}",
                info.name
            ));
            return Ok(false);
        }
        let texel = info.bytes_per_block as usize;
        let bytes = backend.renderer.read_target_bytes(key, texel as u32)?;
        backend.note(format!(
            "texels read from the GPU: {} {width}x{height} region {columns}x{rows}",
            info.name
        ));
        if super::debug_var_os("ECHORA_GPU_FLOW").is_some() {
            let words: Vec<String> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .step_by((bytes.len() / 4 / 48).max(1))
                .take(48)
                .map(|word| format!("{}", f32::from_le_bytes(*word)))
                .collect();
            backend.note(format!(
                "flow texels read {key:x} frame {} {}: [{}]",
                self.presented_frames,
                info.name,
                words.join(" ")
            ));
        }
        let row_bytes = columns as usize * texel;
        for row in 0..rows as usize {
            let start = ((y as usize + row) * width as usize + x as usize) * texel;
            core.memory
                .write(
                    destination + (row * row_bytes) as u64,
                    &bytes[start..start + row_bytes],
                )
                .map_err(|trap| format!("ReadTexels destination: {trap:?}"))?;
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
        let commands: usize = self
            .command_lists
            .values()
            .map(|(list, ..)| list.len())
            .sum();
        let captured: usize = self
            .command_lists
            .values()
            .flat_map(|(list, ..)| list.iter())
            .map(|command| command.captured.len() * 8 + command.states.len() * 512)
            .sum();
        let mut line = format!(
            "present {} objects {} bags {} views {} finalized {} images {} lists {} commands {commands} captured~{}MB recording {}",
            self.presented_frames,
            self.objects.len(),
            super::read_shared(&self.bags).len(),
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
                backend.target_index.len(),
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

    /// The translation of the stage at GPU address `gpu`, cached.
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
    ) -> Result<Arc<Translated>, String> {
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
        let native_pointers = self.gpu.as_ref().unwrap().native_programs.contains(&key);
        let cache_key = (
            key,
            base,
            previous_stores,
            generic_input_types,
            compute,
            mesh,
            native_pointers,
        );
        if let Some(found) = self.gpu.as_ref().unwrap().translations.get(&cache_key) {
            return found.clone();
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
        let translated = {
            let parameters = Parameters {
                binding_base: base[0],
                resolution_scale,
                scaling_base: [base[1], base[2]],
                mesh,
                previous_stores,
                generic_input_types,
                workgroup_size: [compute[0], compute[1], compute[2]],
                // Compute shared memory from the control section; zero
                // (graphics) is raised to 48 KiB, as zero aborts the
                // recompiler on shared accesses.
                shared_memory_size: if compute[3] == 0 { 0xc000 } else { compute[3] },
                // Local (per-invocation stack) memory: without it the
                // recompiler drops stores at constant offsets.
                local_memory_size: compute[4],
                native_pointers,
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
                    shader::translate(&code, parameters, &mut PointerProbe { nvn, core })
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
            let translator = if super::debug_var("ECHORA_PLUGIN_PROGRAMS").is_ok_and(|keys| {
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
                .then(|| shader::translate_rust(&code, &parameters))
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
                    notes.push("stat: rust translator programs".into());
                    Ok(translated)
                }
                (_, Some(Err(unsupported))) => {
                    notes.push(format!("rust translator: {}", unsupported.reason));
                    if shader::available() {
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
            && !shader::available()
        {
            backend.renderer.fatal = Some(format!("untranslated shader: {error}"));
        }
        backend.translations.insert(cache_key, translated.clone());
        translated
    }

    /// Stage keys and translations of the bound vertex and fragment programs.
    /// Constant buffer 1 of the stage bound at NVN stage `bit`: the
    /// shader's own constant data, which the compiler places in its code
    /// block (NVN2 control section: size at +0x3c, offset in the code at
    /// +0x40; from the reference implementation's GLSLC notes).
    fn stage_constants(&self, core: &Guest, bit: u32) -> Option<Vec<u8>> {
        let bind = self.render.state.get(&format!("BindProgram@{bit}"))?;
        let stage = self.program_stage(core, bind.x[1], bit).ok()?;
        self.stage_constants_of(core, stage)
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
    ) -> Result<Option<(u64, u64)>, String> {
        let at = descriptor.cbuf_offset as usize;
        let mut word = [0u8; 8];
        match descriptor.cbuf_index {
            0 => match c0.get(at..at + 8) {
                Some(bytes) => word.copy_from_slice(bytes),
                None => return Ok(None),
            },
            1 => match self.stage_constants(core, 5) {
                Some(bytes) if bytes.len() >= at + 8 => word.copy_from_slice(&bytes[at..at + 8]),
                _ => return Ok(None),
            },
            2 => return Ok(None),
            index => {
                let Some(bind) = self
                    .render
                    .state
                    .get(&format!("BindUniformBuffer/5/{}", index - 3))
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
        let stages: Vec<super::Args> = (0..6)
            .map_while(|index| program.state.get(&format!("Stage{index}")).copied())
            .collect();
        if stages.len() <= 1 {
            return stages
                .first()
                .copied()
                .ok_or_else(|| "bound program has no shaders".into());
        }
        let wanted = match bit {
            0 => 1,
            1 => 5,
            _ => return Err(format!("stage bit {bit} is not modeled")),
        };
        stages
            .iter()
            .find(|stage| {
                self.stage_code(core, stage.x[0]).is_some_and(|(code, _)| {
                    code.len() >= 4
                        && (u32::from_le_bytes([code[0], code[1], code[2], code[3]]) >> 10) & 0xf
                            == wanted
                })
            })
            .copied()
            .ok_or_else(|| format!("bound program has no stage of type {wanted}"))
    }

    fn bound_program(&mut self, core: &Guest) -> Result<BoundProgram, String> {
        // The same programs as the last draw, within a present and with no
        // flush of guest memory since (program code is revalidated once a
        // present, as textures are): its stages again.
        let bind = |bit: u32| {
            self.render
                .state
                .get(if bit == 0 {
                    "BindProgram@0"
                } else {
                    "BindProgram@1"
                })
                .map(|command| command.x)
        };
        let attributes = self
            .render
            .state
            .get("BindVertexAttribState")
            .map(|command| command.states.clone());
        let stamp = bind(0)
            .zip(bind(1))
            .map(|(vertex, fragment)| (self.presented_frames, self.flush_epoch, vertex, fragment));
        if let (Some(stamp), Some((memo, memo_attributes, bound))) = (stamp, &self.program_memo)
            && *memo == stamp
            && match (&attributes, memo_attributes) {
                (Some(now), Some(then)) => Arc::ptr_eq(now, then),
                (None, None) => true,
                _ => false,
            }
        {
            return Ok(bound.clone());
        }
        let bound = self.resolve_bound_program(core)?;
        if let Some(stamp) = stamp {
            self.program_memo = Some((stamp, attributes, bound.clone()));
        }
        Ok(bound)
    }

    fn resolve_bound_program(&mut self, core: &Guest) -> Result<BoundProgram, String> {
        let mut addresses = Vec::new();
        for bit in [0u32, 1] {
            let bind = self
                .render
                .state
                .get(&format!("BindProgram@{bit}"))
                .ok_or_else(|| format!("no program bound for stage {bit}"))?;
            let stage = self.program_stage(core, bind.x[1], bit)?;
            addresses.push(stage.x[0]);
        }
        let types = self
            .render
            .state
            .get("BindVertexAttribState")
            .map_or([0; 32], |bind| generic_input_types(&bind.states));
        let vertex = self.translate_stage(core, addresses[0], [0; 3], None, types, [0; 5], None)?;
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
        let fragment = self.translate_stage(
            core,
            addresses[1],
            base,
            Some(vertex.stores),
            [0; 32],
            [0; 5],
            None,
        )?;
        if vertex.stage != Stage::Vertex || fragment.stage != Stage::Fragment {
            return Err(format!(
                "bound stages are {:?} and {:?}",
                vertex.stage, fragment.stage
            ));
        }
        let keys = (
            self.stage_code(core, addresses[0]).unwrap().1,
            self.stage_code(core, addresses[1]).unwrap().1,
        );
        Ok((keys, vec![vertex, fragment]))
    }

    /// The task (if any), mesh and fragment stages of the programs bound for
    /// a mesh draw: their keys, translations, NVN resource slots and stage
    /// entries. A mesh program writes the stage buffer, and its control
    /// section gives its largest vertex and primitive counts at +0x60 (low
    /// and high half) and its threads at +0x58; a task program has nothing
    /// at +0x60, and its +0x58 is laid out as SET_MESH_INIT_SHADER (threads
    /// 5:0, payload lines 30:22); a vertex program has neither. Without a task the mesh program sits in
    /// the vertex slot (its header names a vertex program), after one in the
    /// tessellation slot; a pass-through geometry program bound with it
    /// lists the per-primitive outputs.
    fn bound_mesh_program(
        &mut self,
        core: &Guest,
    ) -> Result<(BoundProgram, Vec<u32>, Vec<super::Args>), String> {
        // The graphics stages' programs: the COMPUTE bit's (5, NVN's
        // 0x20) is not one. A program bound for every bit (0xff, as the
        // game binds its tile classifier) stays in the compute slot while
        // later graphics binds (0xdf) replace the rest; its fragment stage
        // then drew every following mesh draw.
        let programs: Vec<u64> = self
            .render
            .state
            .iter()
            .filter(|(key, _)| key.starts_with("BindProgram@") && *key != "BindProgram@5")
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
        let (mut translated, mut slots, mut entries) = (Vec::new(), Vec::new(), Vec::new());
        let mut base = [0u32; 3];
        let mut keys = 0u64;
        if let (Some(task), Some(init)) = (task, task_init) {
            let parameters = MeshParameters {
                task: true,
                registers: [shader_a, shader_b, init],
                primitive_header: None,
            };
            let stage = self.translate_stage(
                core,
                task.x[0],
                base,
                None,
                [0; 32],
                [0; 5],
                Some(parameters),
            )?;
            if stage.stage != Stage::Task {
                return Err(format!("the task program translated as {:?}", stage.stage));
            }
            base = advance(&stage, base);
            keys = self.stage_code(core, task.x[0]).map_or(0, |(_, key)| key);
            translated.push(stage);
            slots.push(6);
            entries.push(task);
        }
        let parameters = MeshParameters {
            task: false,
            registers: [shader_a, shader_b, init],
            primitive_header: geometry,
        };
        let stage = self.translate_stage(
            core,
            mesh.x[0],
            base,
            None,
            [0; 32],
            [0; 5],
            Some(parameters),
        )?;
        if stage.stage != Stage::Mesh {
            return Err(format!("the mesh program translated as {:?}", stage.stage));
        }
        base = advance(&stage, base);
        let stores = stage.stores;
        keys = keys.rotate_left(1) ^ self.stage_code(core, mesh.x[0]).map_or(0, |(_, key)| key);
        translated.push(stage);
        slots.push(7);
        entries.push(mesh);
        let stage = self.translate_stage(
            core,
            fragment.x[0],
            base,
            Some(stores),
            [0; 32],
            [0; 5],
            None,
        )?;
        if stage.stage != Stage::Fragment {
            return Err(format!(
                "the fragment program translated as {:?}",
                stage.stage
            ));
        }
        translated.push(stage);
        slots.push(1);
        entries.push(fragment);
        let fragment_key = self
            .stage_code(core, fragment.x[0])
            .map_or(0, |(_, key)| key);
        // Mesh programs never share a key with vertex programs' pipelines.
        Ok((
            ((keys ^ 0x6d65_7368, fragment_key), translated),
            slots,
            entries,
        ))
    }

    /// A mesh draw's workgroups: DrawMeshTasks(cmd, first, count) draws
    /// `count` groups (from 0 only), DrawMeshTasksIndirect(cmd, address)
    /// reads three u32 group counts at that GPU address (from the device
    /// buffer of a resident pool when the GPU may have written them).
    fn mesh_groups(&mut self, core: &mut Guest, command: &Command) -> Result<MeshGroups, String> {
        match command.name.as_str() {
            "nvnCommandBufferDrawMeshTasks" => {
                let (first, count) = (command.x[1] as u32, command.x[2] as u32);
                if first != 0 {
                    return Err(format!("mesh tasks from group {first}"));
                }
                Ok(MeshGroups::Direct([count, 1, 1]))
            }
            "nvnCommandBufferDrawMeshTasksIndirect" => {
                let address = command.x[1];
                let guest = self.gpu_to_guest(address, 12)?;
                // ECHORA_MESH_LOG (debugging): each indirect mesh draw's
                // group counts as the GPU wrote them (waits for the GPU).
                if super::debug_var_os("ECHORA_MESH_LOG").is_some() {
                    self.resolve_gpu_written(core, guest..guest + 12)?;
                    let bytes = self.gpu_bytes(core, address, 12)?;
                    let words: Vec<u32> = bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|word| u32::from_le_bytes(*word))
                        .collect();
                    eprintln!(
                        "mesh indirect present {} at {address:#x}: groups {words:?}",
                        self.presented_frames
                    );
                }
                if let Some((buffer, offset)) = self.resident_pool_binding(core, guest, 12)? {
                    return Ok(MeshGroups::Indirect { buffer, offset });
                }
                let bytes = self.gpu_bytes(core, address, 12)?;
                let word = |index: usize| {
                    u32::from_le_bytes(bytes[index * 4..index * 4 + 4].try_into().unwrap())
                };
                Ok(MeshGroups::Direct([word(0), word(1), word(2)]))
            }
            other => Err(format!("{other} is not modeled on the GPU")),
        }
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

    /// Bytes of a texture object's block-linear storage (all levels and
    /// layers).
    fn texture_storage_size(&self, texture: u64) -> Option<u64> {
        let object = self.objects.get(&texture)?;
        let get = |key: &str| object.builder.get(key).map(|value| value.x[0] as u32);
        let info = crate::nvn_format::info(get("Format")?)?;
        let depth = get("Depth").unwrap_or(1).max(1);
        let (depth, layers) = if get("Target") == Some(2) {
            (depth, 1)
        } else {
            (1, depth)
        };
        Some(
            crate::nvn_format::block_linear_size(
                info,
                get("Width").unwrap_or(1),
                get("Height").unwrap_or(1),
                depth,
                super::texture_layout_layers(
                    &object.builder,
                    info,
                    [get("Width").unwrap_or(1), get("Height").unwrap_or(1), depth],
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
        let view = self.views.get(&(pool, (handle & 0xf_ffff) as u32))?;
        let levels = view.props.get("Levels")?;
        Some((levels.x[0] as u32, levels.x[1] as u32))
    }

    /// The format the texture view registered with a handle sets, if any
    /// (the albedo and decal textures are UNORM and sampled through sRGB
    /// views, as a PokeDinner capture shows them).
    fn handle_view_format(&self, handle: u64) -> Option<vk::Format> {
        let pool = self.render.state.get("SetTexturePool")?.x[1];
        let view = self.views.get(&(pool, (handle & 0xf_ffff) as u32))?;
        vk_format(view.props.get("Format")?.x[0] as u32)
    }

    /// Texture pool entries for a program stage that reads the pool: the
    /// textures bound to NVN stage `nvn`. Bound handles are what such
    /// programs index (they sit in c2 from 0x20, 8 bytes per slot); a
    /// handle formed elsewhere reads a null entry unless a bound one names
    /// it. Handles that do not resolve are left out.
    fn pool_entries(
        &mut self,
        core: &mut Guest,
        nvn: u32,
    ) -> Result<Vec<echora_gpu::pool::PoolEntry>, String> {
        // The renderer's counts from earlier pool writes.
        let backend = self.gpu.as_mut().unwrap();
        for (name, amount) in backend.renderer.take_stats() {
            backend.stat(name, amount);
        }
        let prefix = format!("BindTexture/{nvn}/");
        let mut handles: Vec<u64> = self
            .render
            .state
            .iter()
            .filter(|(key, _)| key.starts_with(&prefix))
            .map(|(_, bind)| bind.x[3])
            .collect();
        handles.sort_unstable();
        handles.dedup();
        // Resolved handles are kept for the present as the draw path's
        // bindings are (`binding_memo`, same stamp).
        let stamp = (
            self.texture_epoch,
            self.gpu.as_ref().unwrap().renderer.target_changes,
            self.binding_serial,
            self.presented_frames,
        );
        if self.binding_memo.0 != stamp {
            self.binding_memo = (stamp, Default::default());
        }
        let mut entries = Vec::new();
        for handle in handles {
            if let Some((source, sampler_state)) = self.binding_memo.1.get(&handle) {
                entries.push(echora_gpu::pool::PoolEntry {
                    texture: (handle & u64::from(echora_shader_abi::TEXTURE_INDEX_MASK)) as u32,
                    sampler: (handle >> echora_shader_abi::SAMPLER_INDEX_SHIFT & 0xfff) as u32,
                    source: source.clone(),
                    sampler_state: *sampler_state,
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
            if let (Some(base), Some(size)) = (
                self.texture_storage(texture),
                self.texture_storage_size(texture),
            ) {
                self.resolve_storage(core, Some(base..base + size))?;
            }
            // As the draw path does before it memoizes: GPU-written
            // bytes under the texture come back first.
            if let (Some(base), Some(size)) = (
                self.texture_storage(texture),
                self.texture_storage_size(texture),
            ) {
                self.resolve_storage(core, Some(base..base + size))?;
            }
            let source = self.texture_source(core, texture)?;
            let source = self.through_view(source, handle);
            let sampler_state = self.binding_sampler(handle, &source);
            self.binding_memo
                .1
                .insert(handle, (source.clone(), sampler_state));
            entries.push(echora_gpu::pool::PoolEntry {
                texture: (handle & u64::from(echora_shader_abi::TEXTURE_INDEX_MASK)) as u32,
                sampler: (handle >> echora_shader_abi::SAMPLER_INDEX_SHIFT & 0xfff) as u32,
                source,
                sampler_state,
            });
        }
        Ok(entries)
    }

    /// The sampler a texture binding samples with: the handle's sampler,
    /// its view's swizzle and format and, for uploads, its view's levels
    /// (a view of those levels: its level 0 is the base, as on the
    /// console; the guest sampler itself stays as the game set it, one
    /// Vulkan sampler for every texture paired with it).
    fn binding_sampler(&self, handle: u64, source: &TextureSource) -> Sampler {
        let mut sampler = self.handle_sampler(handle);
        sampler.components = self.handle_components(handle);
        sampler.format = self.handle_view_format(handle);
        if sampler.format.is_some() && super::debug_var_os("ECHORA_GPU_FLOW").is_some() {
            let texture = self.handle_texture(handle).unwrap_or(0);
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
        if let (TextureSource::Upload { .. }, Some((base, count))) =
            (source, self.handle_view_levels(handle))
        {
            if count > 0 {
                sampler.view_levels = Some((base, count));
            }
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
        let view = self.views.get(&(pool, (handle & 0xf_ffff) as u32))?;
        let swizzle = view.props.get("Swizzle")?;
        Some([swizzle.x[0], swizzle.x[1], swizzle.x[2], swizzle.x[3]])
    }

    /// A render target sampled through a handle whose view selects array
    /// layers (or cube faces) samples only those, and a view's mip levels
    /// only those (`mip_targets`).
    fn through_view(&self, source: TextureSource, handle: u64) -> TextureSource {
        let TextureSource::Target(key) = source else {
            return source;
        };
        let (base_level, level_count) = if super::mip_targets() {
            self.handle_view_levels(handle).unwrap_or((0, 0))
        } else {
            (0, 0)
        };
        let (base_layer, layer_count) = self.handle_view_layers(handle).unwrap_or((0, 0));
        if (base_level, level_count, base_layer, layer_count) == (0, 0, 0, 0) {
            return source;
        }
        TextureSource::TargetView {
            key,
            base_level,
            level_count,
            base_layer,
            layer_count,
        }
    }

    /// The array layers (base, count; 0: the rest) the texture view
    /// registered with a handle selects, if any.
    fn handle_view_layers(&self, handle: u64) -> Option<(u32, u32)> {
        let pool = self.render.state.get("SetTexturePool")?.x[1];
        let view = self.views.get(&(pool, (handle & 0xf_ffff) as u32))?;
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
        // Remembered until an object is created or removed, a texture is
        // initialized or finalized, or a virtual pool is remapped: the
        // property walk below was a third of the draw path's time.
        let generation = (self.object_generation, self.texture_epoch);
        if let Ok(mut memo) = self.storage_memo.try_borrow_mut() {
            if memo.0 != generation {
                *memo = (generation, Default::default());
            }
            if let Some(found) = memo.1.get(&texture) {
                return *found;
            }
        }
        let found = self.texture_storage_uncached(texture);
        if let Ok(mut memo) = self.storage_memo.try_borrow_mut() {
            memo.1.insert(texture, found);
        }
        found
    }

    fn texture_storage_uncached(&self, texture: u64) -> Option<u64> {
        let storage = self.objects.get(&texture)?.builder.get("Storage")?;
        let pool = self.objects.get(&storage.x[0])?;
        let base = pool.builder.get("Storage")?.x[0];
        if base != 0 {
            return Some(base + storage.x[1]);
        }
        let gpu = pool.state.get("GpuAddress")?.x[0] + storage.x[1];
        self.pool_for(gpu).map(|(guest, _)| guest)
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
                    .existing_target(*key)
                    .is_some_and(|target| (target.width, target.height) == extent)
        };
        if self.target_index_current() {
            return backend
                .target_index
                .get(&base)?
                .iter()
                .copied()
                .find(|key| matches(key));
        }
        renderer.target_keys().into_iter().find(|key| matches(key))
    }

    /// Whether `target_index` describes the current targets and storage.
    fn target_index_current(&self) -> bool {
        self.gpu.as_ref().is_some_and(|backend| {
            backend.target_index_stamp
                == Some((backend.renderer.target_changes, self.texture_epoch))
        })
    }

    /// Rebuild `target_index` if targets or texture storage changed.
    fn refresh_target_index(&mut self) {
        if self.target_index_current() {
            return;
        }
        let Some(backend) = self.gpu.as_ref() else {
            return;
        };
        let mut index: HashMap<u64, Vec<u64>> = HashMap::new();
        let mut chains = Vec::new();
        for key in backend.renderer.target_keys() {
            if let Some(base) = self.texture_storage(key) {
                index.entry(base).or_default().push(key);
            }
            if backend
                .renderer
                .existing_target(key)
                .is_some_and(|target| target.levels > 1)
            {
                chains.push(key);
            }
        }
        let backend = self.gpu.as_mut().unwrap();
        backend.target_index = index;
        backend.target_chains = chains;
        backend.target_index_stamp = Some((backend.renderer.target_changes, self.texture_epoch));
    }

    /// The render target (and its mip level) a texture object's storage
    /// is, as a GPU texture cache resolves addresses: the target itself,
    /// another target at the same address and size, or a target whose mip
    /// level `level` starts at this storage with this size (a separate
    /// texture object per level, as bloom chains use; levels above 0 as
    /// `mip_targets` allows).
    fn resolve_target(&self, texture: u64) -> Option<(u64, u32)> {
        let renderer = &self.gpu.as_ref()?.renderer;
        let extent = self.texture_extent(texture).ok();
        // Its own format (or its sRGB twin): a texture object
        // reinterpreting the memory in another format (RGB9E5 over the HDR
        // scene's R11G11B10) is not the same target.
        let format = self.target_shape(texture).ok().map(|(_, _, format)| format);
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
        if renderer.existing_target(texture).is_some_and(|target| {
            (format.is_none() || same_format(target.format))
                && extent.is_none_or(|extent| (target.width, target.height) == extent)
        }) {
            return Some((texture, 0));
        }
        let mips = super::mip_targets();
        if !mips {
            return self.target_sharing_storage(texture).map(|key| (key, 0));
        }
        let base = self.texture_storage(texture)?;
        let extent = extent?;
        // The targets at this storage and those with levels, from the
        // index when it is current (every target otherwise).
        let candidates: Vec<u64> = if self.target_index_current() {
            let backend = self.gpu.as_ref()?;
            backend
                .target_index
                .get(&base)
                .into_iter()
                .flatten()
                .chain(backend.target_chains.iter())
                .copied()
                .collect()
        } else {
            renderer.target_keys()
        };
        for key in candidates {
            let Some(target) = renderer.existing_target(key) else {
                continue;
            };
            let Some(target_base) = self.texture_storage(key) else {
                continue;
            };
            if target_base == base
                && (target.width, target.height) == extent
                && same_format(target.format)
            {
                return Some((key, 0));
            }
            if !mips || target.levels == 1 || base <= target_base {
                continue;
            }
            let object = self.objects.get(&key)?;
            let get = |name: &str| object.builder.get(name).map(|value| value.x[0] as u32);
            let Some(info) = get("Format").and_then(crate::nvn_format::info) else {
                continue;
            };
            let depth = if get("Target") == Some(2) {
                get("Depth").unwrap_or(1).max(1)
            } else {
                1
            };
            for level in 1..target.levels {
                let offset = crate::nvn_format::block_linear_level_offset(
                    info,
                    target.width,
                    target.height,
                    depth,
                    super::texture_layout_layers(
                        &object.builder,
                        info,
                        [target.width, target.height, depth],
                        target.shape.layers(),
                        target.levels,
                    ),
                    level,
                );
                let level_extent = (
                    (target.width >> level).max(1),
                    (target.height >> level).max(1),
                );
                if target_base + offset == base && level_extent == extent {
                    return Some((key, level));
                }
            }
        }
        None
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

    fn texture_source(&mut self, core: &Guest, texture: u64) -> Result<TextureSource, String> {
        let source = self.texture_source_resolved(core, texture)?;
        // Blocks drawn since it was last sampled come in first.
        if let TextureSource::Target(key) | TextureSource::TargetView { key, .. } = &source
            && let Some(alias) = self.block_aliases.remove(key)
        {
            let backend = self.gpu.as_mut().unwrap();
            backend.renderer.copy_blocks(alias, *key)?;
            backend.stat("stat: compressed blocks drawn copied in", 1);
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
        if let Some((key, level)) = self.resolve_target(texture) {
            if level > 0 {
                return Ok(TextureSource::TargetView {
                    key,
                    base_level: level,
                    level_count: 1,
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
            .existing_target(texture)
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

    /// Render targets created since the last call start with their guest
    /// memory's contents (level 0), as on the console, where a target is
    /// that memory: values a pass reads before it writes them (an
    /// auto-exposure adapting from its last value) are then defined.
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
                (Some((_, _, storage)), _) if backend.dense_samples.contains_key(&texture) => {
                    backend.dense_samples.get(&texture)
                        == Some(&dense_sample(
                            core,
                            storage.start,
                            storage.end - storage.start,
                        ))
                }
                (Some((_, _, storage)), Some(hashes)) => {
                    let mut chunks: Vec<u64> = ranges
                        .iter()
                        .flat_map(|range| {
                            range.start / STORAGE_CHUNK..range.end.div_ceil(STORAGE_CHUNK)
                        })
                        .collect();
                    chunks.sort_unstable();
                    chunks.dedup();
                    chunks.iter().all(|chunk| {
                        let start = chunk * STORAGE_CHUNK;
                        let length = STORAGE_CHUNK.min(storage.end - storage.start - start);
                        let bytes = read_storage(core, storage.start + start, length);
                        hashes.get(*chunk as usize) == Some(&hash_bytes(&bytes))
                    })
                }
                _ => false,
            };
            let backend = self.gpu.as_mut().unwrap();
            if unchanged {
                backend.stat("stat: texture uploads kept after flush", 1);
            } else {
                backend.textures.remove(&texture);
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
        let (width, height) = (get("Width").unwrap_or(1), get("Height").unwrap_or(1));
        let astc = info.name.starts_with("ASTC");
        let vk_format = match vk_format(format) {
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
        // Depth textures hold what the GPU rendered; their guest bytes are
        // not uploaded (the upload path copies color data only).
        if echora_gpu::renderer::is_depth(vk_format) {
            return Err(format!("{} texture never rendered to", info.name));
        }
        let storage = object
            .builder
            .get("Storage")
            .ok_or("texture has no storage")?;
        let pool = self.object(storage.x[0], "MemoryPool")?;
        // Through texture_storage: a virtual pool's offset resolves to the
        // physical memory mapped there.
        let base = self
            .texture_storage(texture)
            .ok_or("texture storage is in no mapped memory")?;
        // NVN memory pool flags (same bits as NVN1): CPU_UNCACHED 2,
        // CPU_CACHED 4; the builder default 0x21 gives the CPU no access.
        let pool_flags = pool.builder.get("Flags").map_or(0x21, |value| value.x[0]);
        let uncached = pool_flags & 2 != 0 && pool_flags & 4 == 0;
        // Buffer textures (target 10) are linear arrays of texels.
        let buffer_texture = get("Target") == Some(10);
        let stride = if buffer_texture {
            u64::from(width.div_ceil(info.block_width)) * u64::from(info.bytes_per_block)
        } else {
            object.builder.get("Stride").map_or(0, |value| value.x[0])
        };
        let block = info.bytes_per_block as usize;
        // Untiling and hashing whole textures on every draw costs more than
        // the draw: a sample of 256 evenly spaced 64-byte chunks of the raw
        // storage (and the shape) decides whether the last upload still
        // holds. CPU writes the application flushes (memory pool flushes,
        // WriteTexels) drop overlapping entries (`gpu_invalidate`), so
        // small updates such as new glyphs in a font atlas are not missed.
        // NVN targets (inferred, GL order): 0 1D, 1 2D, 2 3D, 3 1D array,
        // 4 2D array, 7 rectangle, 8 cube, 9 cube array, 10 buffer (read as
        // a 1D image here); arrays and cubes
        // keep their layers (faces) in Depth.
        let depth_or_layers = get("Depth").unwrap_or(1).max(1);
        let shape = match get("Target").unwrap_or(1) {
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
        // Shapes a texture's own layout gives that the extent alone would
        // not: noted by shape, so a report shows which textures they are.
        let mut layout_note = None;
        if let Some(shape) = layout.shape {
            let derived = crate::nvn_format::base_block_shape(info, height, depth, layers);
            if shape != derived {
                layout_note = Some(format!(
                    "layout shape {shape:?} instead of {derived:?}: {} {width}x{height}x{depth} layers {layers} levels {levels_set} target {:?} packaged {:016x}",
                    info.name,
                    get("Target"),
                    object
                        .builder
                        .get("PackagedTextureLayout")
                        .map_or(0, |value| value.x[1])
                ));
            }
        }
        let storage_size = if stride != 0 {
            stride * u64::from(height.div_ceil(info.block_height))
        } else {
            crate::nvn_format::block_linear_size(info, width, height, depth, layout, levels_set).0
        };
        let check_started = std::time::Instant::now();
        let shadowed = self.gpu.as_ref().unwrap().shadows.contains_key(&texture);
        // A shadowed texture that stayed unchanged for many presents is
        // compared in full every 2nd, then 4th present; between those the
        // 256-chunk sample below checks it (most textures are written once
        // and compared whole every present otherwise, the largest part of
        // texture validation).
        let skip_full = shadowed && {
            let quiet = self
                .gpu
                .as_ref()
                .unwrap()
                .shadow_quiet
                .get(&texture)
                .copied()
                .unwrap_or(0);
            let interval = match quiet {
                0..16 => 1,
                16..32 => 2,
                _ => 4,
            };
            !frame.is_multiple_of(interval)
                && super::debug_var_os("ECHORA_EXACT_TEXTURE_CHECKS").is_none()
        };
        // Uploads from uncached pools: the whole storage against its copy.
        let shadow_changed = !skip_full && {
            let backend = self.gpu.as_ref().unwrap();
            backend.shadows.get(&texture).is_some_and(|shadow| {
                shadow.len() as u64 != storage_size
                    || !core.memory.equals(base, shadow).unwrap_or(false)
            })
        };
        // Whether a whole-storage check passed (recorded below, with the
        // backend borrowed mutably).
        let full_check_passed = (shadowed && !skip_full).then_some(!shadow_changed);
        // An unchanged copy proves the cached upload current: its sample
        // stands for the storage.
        let kept_sample = (shadowed && !shadow_changed && !skip_full)
            .then(|| {
                self.gpu
                    .as_ref()
                    .unwrap()
                    .textures
                    .get(&texture)
                    .map(|entry| entry.0)
            })
            .flatten();
        let sample = kept_sample.unwrap_or_else(|| {
            texture_sample(
                core,
                base,
                storage_size,
                format,
                [width, height, depth_or_layers],
            )
        });
        let check_micros = check_started.elapsed().as_micros() as u64;
        // ECHORA_GPU_VERIFY_UPLOADS=1 (debugging): a cached upload whose
        // sample still matches is also checked against its whole storage.
        let stale = super::debug_var_os("ECHORA_GPU_VERIFY_UPLOADS").is_some() && {
            let backend = self.gpu.as_ref().unwrap();
            let sampled = backend.textures.get(&texture).map(|entry| entry.0);
            sampled == Some(sample)
                && backend.storage_hashes.get(&texture).is_some_and(|hashes| {
                    let raw = read_storage(core, base, storage_size);
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
            if *sampled == sample && !stale && !shadow_changed {
                self.release_upload_texels(texture);
                let backend = self.gpu.as_mut().unwrap();
                let source = backend.textures[&texture].1.clone();
                backend.texture_checked.insert(texture, frame);
                if full_check_passed == Some(true) {
                    let quiet = backend.shadow_quiet.entry(texture).or_default();
                    *quiet = quiet.saturating_add(1);
                }
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
        // The rows of the last whole-level copy into the texture, when the
        // storage still samples as the copy left it: they are level 0
        // untiled, so neither the storage read nor untiling is needed.
        let copied = (level_count == 1 && layers == 1 && depth == 1 && !astc && !uncached)
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
            read_storage(core, base, storage_size)
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
        let mut texels = Vec::new();
        let mut levels = Vec::new();
        for level in 0..level_count {
            let (level_width, level_height, level_depth) = (
                (width >> level).max(1),
                (height >> level).max(1),
                (depth >> level).max(1),
            );
            let (columns, rows) = (
                level_width.div_ceil(info.block_width),
                level_height.div_ceil(info.block_height),
            );
            let level_offset = crate::nvn_format::block_linear_level_offset(
                info, width, height, depth, layout, level,
            );
            let level_shape =
                crate::nvn_format::level_block_shape(info, height, depth, layout, level);
            let row_bytes = columns as usize * block;
            for layer in 0..layers {
                let layer_base = u64::from(layer) * layer_stride + level_offset;
                let slice_bytes = row_bytes * rows as usize;
                let mut data = vec![0u8; slice_bytes * level_depth as usize];
                if let Some(rows) = copied.as_ref().filter(|rows| rows.len() == data.len()) {
                    data.copy_from_slice(rows);
                } else {
                    for z in 0..level_depth {
                        for row in 0..rows {
                            let line = &mut data
                                [z as usize * slice_bytes + row as usize * row_bytes..]
                                [..row_bytes];
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
                            // A run of blocks at consecutive addresses at a time.
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
                }
                if astc {
                    let texels_per_slice = level_width as usize * level_height as usize * 4;
                    let mut rgba = vec![0u8; texels_per_slice * level_depth as usize];
                    let footprint =
                        astc_decode::Footprint::new(info.block_width, info.block_height);
                    for z in 0..level_depth as usize {
                        let slice = &mut rgba[z * texels_per_slice..][..texels_per_slice];
                        astc_decode::astc_decode(
                            &data[z * slice_bytes..][..slice_bytes],
                            level_width,
                            level_height,
                            footprint,
                            |x, y, texel| {
                                let at = (y as usize * level_width as usize + x as usize) * 4;
                                slice[at..at + 4].copy_from_slice(&texel);
                            },
                        )
                        .map_err(|error| format!("ASTC decode: {error}"))?;
                    }
                    data = rgba;
                }
                // ECHORA_GPU_TEXTURE_DUMP=<dir> (debugging): the base level
                // of large 4-byte 2D textures, as last uploaded.
                if let Some(directory) = super::debug_var_os("ECHORA_GPU_TEXTURE_DUMP") {
                    let path = std::path::Path::new(&directory).join(format!(
                        "tex-{base:x}-{level_width}x{level_height}-{}.ppm",
                        info.name
                    ));
                    if level == 0
                        && layer == 0
                        && block == 4
                        && level_depth == 1
                        && level_width >= 512
                    {
                        let mut ppm =
                            format!("P6\n{level_width} {level_height}\n255\n").into_bytes();
                        for texel in data.as_chunks::<4>().0 {
                            ppm.extend_from_slice(&texel[..3]);
                        }
                        let _ = std::fs::write(&path, ppm);
                        let mut pgm =
                            format!("P5\n{level_width} {level_height}\n255\n").into_bytes();
                        pgm.extend(data.as_chunks::<4>().0.iter().map(|texel| texel[3]));
                        let _ = std::fs::write(path.with_extension("alpha.pgm"), pgm);
                    }
                    // Compressed (16-byte block) textures: the untiled
                    // blocks, to compare with a reference capture.
                    if level == 0 && layer == 0 && block == 16 && level_width >= 512 {
                        let _ = std::fs::write(path.with_extension("bin"), &data);
                    }
                }
                levels.push(UploadRegion {
                    level,
                    layer,
                    width: level_width,
                    height: level_height,
                    depth: level_depth,
                    offset: texels.len() as u64,
                });
                texels.extend_from_slice(&data);
            }
        }
        // Content-keyed: guest writes that change the texels upload again.
        let generation = hash_bytes(&texels);
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
        let source = TextureSource::Upload {
            key: texture,
            generation,
            width,
            height,
            shape,
            format: vk_format,
            texels: Arc::new(texels),
            levels,
        };
        self.gpu
            .as_mut()
            .unwrap()
            .textures
            .insert(texture, (sample, source.clone(), base..base + storage_size));
        let backend = self.gpu.as_mut().unwrap();
        backend.storage_hashes.insert(texture, storage_hashes);
        if uncached && storage_size <= SHADOW_LIMIT {
            if backend.shadows.contains_key(&texture) {
                backend.stat("stat: uncached uploads changed", 1);
            }
            backend.shadows.insert(texture, raw_copy);
        } else {
            backend.shadows.remove(&texture);
        }
        if storage_size > DENSE_SAMPLE_ABOVE {
            let sample = dense_sample(core, base, storage_size);
            self.gpu
                .as_mut()
                .unwrap()
                .dense_samples
                .insert(texture, sample);
        }
        let backend = self.gpu.as_mut().unwrap();
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
        backend.shadow_quiet.remove(&texture);
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
        let id = handle & 0xf_ffff;
        let entry = self
            .object(pool, "TexturePool")?
            .state
            .get(&format!("Descriptor{id}"))
            .ok_or_else(|| format!("texture {id} is not registered"))?;
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
        let mut sampler = Sampler::default();
        let Some(pool) = self
            .render
            .state
            .get("SetSamplerPool")
            .map(|bind| bind.x[1])
        else {
            return sampler;
        };
        let id = handle >> 20 & 0xfff;
        let Some(entry) = self
            .object(pool, "SamplerPool")
            .ok()
            .and_then(|pool| pool.state.get(&format!("Descriptor{id}")))
        else {
            return sampler;
        };
        let props = match self.object(entry.x[0], "Sampler") {
            Ok(object) => object.builder.clone(),
            Err(_) => match self.bag(entry.x[0], "SamplerBuilder") {
                Ok(props) => props,
                Err(_) => return sampler,
            },
        };
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

    /// Serial of the GPU submission holding everything recorded so far
    /// (0 without a GPU).
    pub(super) fn gpu_recorded_serial(&self) -> u64 {
        self.gpu
            .as_ref()
            .map_or(0, |backend| backend.renderer.recorded_serial())
    }

    /// A CPU wait on a signaled sync: waits for the submission that held
    /// the work before its fence, or for everything when that is unknown.
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
        match self.gpu.as_mut() {
            Some(backend) => backend.renderer.submit_frame(),
            None => Ok(()),
        }
    }

    pub(super) fn gpu_present_window(&mut self, texture: u64) -> Result<bool, String> {
        let Some(window) = crate::window::shared() else {
            return Ok(false);
        };
        let backend = self.gpu.as_mut().unwrap();
        if backend.presenter.is_none() {
            let created = window
                .handle()
                .ok_or_else(|| "no window".to_string())
                .and_then(|handle| Presenter::new(&backend.renderer.gpu, handle));
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
    /// cubes their faces (in Depth, as for sampling; NVN targets inferred,
    /// GL order), so a view's layer selects one of them. Others, and arrays
    /// of more than 64 layers, are one 2D layer.
    fn render_shape(&self, texture: u64) -> ImageShape {
        let Ok(object) = self.object(texture, "Texture") else {
            return ImageShape::D2 { layers: 1 };
        };
        let get = |name: &str| object.builder.get(name).map(|value| value.x[0] as u32);
        let depth = get("Depth").unwrap_or(1).clamp(1, 64);
        match get("Target").unwrap_or(1) {
            // TARGET_3D: a volume of all its slices (the fog's froxel
            // volumes are 96 deep), as the dispatch image binding makes it;
            // one 2D layer would drop every store past slice 0.
            2 => ImageShape::D3 {
                depth: get("Depth").unwrap_or(1).max(1),
            },
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
        if self.target_shapes.0 != self.texture_epoch {
            self.target_shapes = (self.texture_epoch, HashMap::new());
        }
        let (width, height, format, levels, shape) = match self.target_shapes.1.get(&texture) {
            Some(known) => *known,
            None => {
                let (width, height, format) = self.target_shape(texture)?;
                let known = (
                    width,
                    height,
                    format,
                    self.texture_levels(texture),
                    self.render_shape(texture),
                );
                self.target_shapes.1.insert(texture, known);
                known
            }
        };
        self.gpu
            .as_mut()
            .unwrap()
            .renderer
            .shaped_target(texture, width, height, shape, format, levels)?;
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
            if renderer.existing_target(texture).is_some() {
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
        // Its results may change what later texture bindings resolve to.
        self.binding_serial += 1;
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
        let mut stage =
            self.translate_stage(core, address, [0; 3], None, [0; 32], compute, None)?;
        // Storage pointers the NVN tracking found outside c0 were constants,
        // not addresses (floats 1.0 and 10.0 in c1): such a program reads
        // its pointers as raw arguments instead, picking among every
        // mapped candidate at run time.
        if stage.descriptors.iter().any(|descriptor| {
            descriptor.kind() == DescriptorKind::StorageBuffer
                && descriptor.cbuf_index != 0
                && descriptor.flags & 32 == 0
        }) {
            let code_key = self.stage_code(core, address).map(|(_, key)| key);
            if let Some(code_key) = code_key
                && self.gpu.as_mut().unwrap().native_programs.insert(code_key)
            {
                stage =
                    self.translate_stage(core, address, [0; 3], None, [0; 32], compute, None)?;
            }
        }
        if stage.stage != Stage::Compute {
            return Err(format!("bound compute stage is {:?}", stage.stage));
        }
        let key = (self.stage_code(core, address).unwrap().1, u64::MAX);
        // ECHORA_GPU_C1_PROBE (debugging): for programs translated with raw
        // pointers, the words c1 could hold (the stage's constants, each
        // pushed root table, the first bound uniform buffers), every 300th
        // present.
        if super::debug_var_os("ECHORA_GPU_C1_PROBE").is_some()
            && self.presented_frames.is_multiple_of(300)
            && self.gpu.as_ref().unwrap().native_programs.contains(&key.0)
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
            let constants = self.stage_constants(core, 5).unwrap_or_default();
            eprintln!(
                "c1probe present {} {:016x} constants {:?}",
                self.presented_frames,
                key.0,
                words(&constants)
            );
            for (table, root) in self.render.roots[1].iter().enumerate().take(4) {
                let bytes: Vec<u8> = root[..8]
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect();
                eprintln!("c1probe   root {table} {:?}", words(&bytes));
            }
            for slot in 0..4 {
                if let Some(bind) = self
                    .render
                    .state
                    .get(&format!("BindUniformBuffer/5/{slot}"))
                {
                    eprintln!("c1probe   ubo {slot} {:#x}+{:#x}", bind.x[3], bind.x[4]);
                }
            }
        }
        // ECHORA_GPU_SKIP_DISPATCH=<program key>[,...] (debugging): those
        // compute programs are not run.
        if super::debug_var("ECHORA_GPU_SKIP_DISPATCH").is_ok_and(|skipped| {
            skipped
                .split(',')
                .any(|skip| skip == format!("{:016x}", key.0))
        }) {
            return Ok(());
        }
        // For skipped dispatches: the program's bindings and the compute
        // slots bound, from which the driver constant layout is inferred.
        let layout = {
            let descriptors: Vec<String> = stage
                .descriptors
                .iter()
                .map(|d| format!("{:?}@c{}+{:#x}", d.kind(), d.cbuf_index, d.cbuf_offset))
                .collect();
            let bound: Vec<&str> = self
                .render
                .state
                .keys()
                .filter(|key| key.contains("/5/"))
                .map(String::as_str)
                .collect();
            format!(
                "program {:#x} uses [{}], bound [{}]",
                key.0,
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
        let unplaced = |descriptor: &shader::Descriptor| {
            format!(
                "{:?} binding at c{}+{:#x} in compute (control 0xb0.. {:02x?}): {layout}",
                descriptor.kind(),
                descriptor.cbuf_index,
                descriptor.cbuf_offset,
                header
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
            if let Some(bind) = self
                .render
                .state
                .get(&format!("BindUniformBuffer/5/{slot}"))
            {
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
                if let Some(bind) = self
                    .render
                    .state
                    .get(&format!("BindStorageBuffer/5/{slot}"))
                {
                    c0[entry..entry + 8].copy_from_slice(&bind.x[3].to_le_bytes());
                    c0[entry + 8..entry + 12].copy_from_slice(&(bind.x[4] as u32).to_le_bytes());
                }
            }
        }
        let mut storage = HashMap::new();
        let mut storage_in_place = HashMap::new();
        // Each storage binding's GPU range (ECHORA_DUMP_DISPATCH).
        let mut storage_ranges = Vec::new();
        let mut written_back = Vec::new();
        let mut uniforms = HashMap::new();
        let mut textures = HashMap::new();
        let mut images = HashMap::new();
        // The view formats images are written through (see
        // `Dispatch::image_formats`).
        let mut image_formats = HashMap::new();
        for descriptor in &stage.descriptors {
            let entry = descriptor.cbuf_offset as usize;
            match descriptor.kind() {
                DescriptorKind::StorageBuffer => {
                    let native = descriptor.flags & 32 != 0;
                    let (gpu, size) = match slot(descriptor, DescriptorKind::StorageBuffer) {
                        // A raw pointer candidate (the program picks among
                        // them by address): one that maps nowhere binds
                        // zeros it never selects.
                        _ if native => match self.native_pointer(core, descriptor, &c0)? {
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
                                .get(&format!("BindStorageBuffer/5/{slot}"))
                                .ok_or_else(|| {
                                    format!("storage buffer 5/{slot} is not bound: {layout}")
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
                                .ok_or_else(|| unplaced(descriptor))?;
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
                            self.presented_frames, key.0, descriptor.binding
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
                        .is_ok_and(|probe| probe == format!("{:016x}", key.0))
                        && self.presented_frames.is_multiple_of(300);
                    if probing {
                        eprintln!(
                            "probe present {} {:016x} binding {} c{}+{:#x} gpu {gpu:#x} size {size:#x} flags {:#x}",
                            self.presented_frames,
                            key.0,
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
                    let slot = slot(descriptor, kind).ok_or_else(|| unplaced(descriptor))?;
                    let handle = self
                        .render
                        .state
                        .get(&format!("BindTexture/5/{slot}"))
                        .ok_or_else(|| format!("texture 5/{slot} is not bound: {layout}"))?
                        .x[3];
                    c0[entry..entry + 8].copy_from_slice(&handle.to_le_bytes());
                    let texture = self.handle_texture(handle)?;
                    let started = std::time::Instant::now();
                    if let (Some(base), Some(size)) = (
                        self.texture_storage(texture),
                        self.texture_storage_size(texture),
                    ) {
                        self.resolve_storage(core, Some(base..base + size))?;
                    }
                    let source = self.texture_source(core, texture)?;
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
                    let sampler = self.binding_sampler(handle, &source);
                    textures.insert(descriptor.binding, (source, sampler));
                }
                // Images a compute program writes become render targets, so
                // later draws sample what it wrote.
                DescriptorKind::Image => {
                    let slot = slot(descriptor, DescriptorKind::Image)
                        .ok_or_else(|| unplaced(descriptor))?;
                    let handle = self
                        .render
                        .state
                        .get(&format!("BindImage/5/{slot}"))
                        .ok_or_else(|| format!("image 5/{slot} is not bound: {layout}"))?
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
                        .shaped_target(texture, width, height, shape, format, levels)?;
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
                        .state
                        .get(&format!("BindUniformBuffer/5/{slot}"))
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
                    let mut bytes = self.stage_constants(core, 5).unwrap_or_default();
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
                _ => return Err(unplaced(descriptor)),
            }
        }
        self.initialize_new_targets(core);
        // c0 itself, once every handle in it is known.
        for descriptor in &stage.descriptors {
            if descriptor.kind() == DescriptorKind::UniformBuffer && descriptor.index == 0 {
                let used = (stage.cbuf_used_sizes[0] as usize).max(16);
                let mut bytes = c0.clone();
                bytes.resize(used.max(bytes.len()), 0);
                uniforms.insert(descriptor.binding, bytes);
            }
        }
        if super::debug_var_os("ECHORA_GPU_FLOW").is_some() {
            let line = format!(
                "flow dispatch {:016x} reads {} writes {} storage {} flags {:?}",
                key.0,
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
                    key.0,
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
                .any(|item| item.trim() == format!("{:016x}", key.0))
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
                    key.0,
                    reads.join(", "),
                    self.describe_targets(image_keys.iter().copied())
                );
            }
            for target in &image_keys {
                let storage = self.texture_storage(*target).unwrap_or(0);
                self.target_writers
                    .insert(*target, (key.0, self.presented_frames, storage));
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
            .any(|item| item.trim() == format!("{:016x}", key.0))
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
                    key.0, self.presented_frames
                ))
                .exists()
            {
                index += 1;
            }
            for (binding, bytes) in &uniforms {
                let _ = std::fs::write(
                    directory.join(format!(
                        "uniforms-{:016x}-{}-{index}-{binding}.bin",
                        key.0, self.presented_frames
                    )),
                    bytes,
                );
            }
            // An empty marker so the index above finds this dispatch.
            let _ = std::fs::write(
                directory.join(format!(
                    "uniforms-{:016x}-{}-{index}-0.bin",
                    key.0, self.presented_frames
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
            self.resolve_copies(core, Some(guest..guest + 12))?;
            Some(
                self.resident_pool_binding(core, guest, 12)?
                    .ok_or("indirect dispatch arguments are in no memory pool")?,
            )
        } else {
            None
        };
        // The program, specialized for what is bound: an IntegerFormat
        // constant is 1 when its binding's image or texture is integer. Each
        // combination is its own program (key.1 holds the values as bits).
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
                    let address = match backend.loop_traces.get(&key.0) {
                        Some(&(_, address, _)) => address,
                        None => {
                            let (index, address) = backend.renderer.debug_buffer(64 << 10)?;
                            backend.loop_traces.insert(key.0, (index, address, 0));
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
        let key = if specialization.is_empty() {
            key
        } else {
            let bits = specialization
                .iter()
                .enumerate()
                .fold(0u64, |bits, (index, (_, value))| {
                    bits | u64::from(*value != 0) << index
                });
            (key.0, 1 << 63 | bits)
        };
        if !backend.programs.contains_key(&key) {
            let program = backend
                .renderer
                .create_program_specialized(vec![stage.clone()], &[specialization])?;
            backend.programs.insert(key, program);
        }
        // Every 60 presents, a traced program's loops that ran: for loop k,
        // total iterations, the most one invocation made, invocations the
        // budget stopped, and the header's guest offset.
        if let Some(&(index, address, reported)) = backend.loop_traces.get(&key.0)
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
                        key.0,
                        entry[3].wrapping_sub(1),
                        entry[0],
                        entry[1],
                        entry[2]
                    );
                }
            }
            backend
                .loop_traces
                .insert(key.0, (index, address, self.presented_frames));
        }
        // The GPU page table, for programs that read global memory through
        // it (ECHORA_PAGE_TABLE).
        let page_table = if self.gpu.as_ref().unwrap().programs[&key]
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
        if page_table.address != 0
            && self.gpu.as_ref().unwrap().programs[&key]
                .stages
                .iter()
                .any(|stage| {
                    stage.push_fields & echora_shader_abi::PUSH_PAGE_TABLE != 0
                        && stage.push_fields & echora_shader_abi::PUSH_PAGE_TABLE_READ_ONLY == 0
                })
        {
            let backend = self.gpu.as_mut().unwrap();
            backend.page_table_writes = true;
            backend.page_table_wrote = true;
        }
        let pool = if stage.texture_pool_bindings != 0 {
            self.pool_entries(core, 5)?
        } else {
            Vec::new()
        };
        // ECHORA_DUMP_DISPATCH=<program key>[,...] (debugging): the first
        // dispatch of each listed program, for the workgroup simulator
        // (`dump_dispatch`).
        if let Some(keys) = super::debug_var_os("ECHORA_DUMP_DISPATCH") {
            let wanted = keys
                .to_string_lossy()
                .split(',')
                .any(|item| u64::from_str_radix(item.trim(), 16) == Ok(key.0));
            // ECHORA_DUMP_DISPATCH_FROM=N: from present N on (a temporal
            // pass's live frame, not its first run).
            let from = super::debug_var("ECHORA_DUMP_DISPATCH_FROM")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(0);
            if wanted && self.presented_frames >= from && self.dumped_dispatches.insert(key.0) {
                let groups = [command.x[1], command.x[2], command.x[3]].map(|count| count as u32);
                if let Err(error) = self.dump_dispatch(
                    core,
                    key.0,
                    &stage,
                    groups,
                    &uniforms,
                    &storage_ranges,
                    page_table,
                ) {
                    eprintln!("dispatch dump {:016x}: {error}", key.0);
                    self.gpu_note(format!("dispatch dump {:016x}: {error}", key.0));
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
                    key.0,
                    read.len()
                );
                if let Err(error) = self.dump_images(key.0, &read) {
                    eprintln!("dispatch dump {:016x} images: {error}", key.0);
                    self.gpu_note(format!("dispatch dump {:016x} images: {error}", key.0));
                }
            }
            // ECHORA_DUMP_ONLY: listed programs are dumped, never run.
            if wanted && super::debug_var_os("ECHORA_DUMP_ONLY").is_some() {
                return Ok(());
            }
        }
        let backend = self.gpu.as_mut().unwrap();
        let dispatch = echora_gpu::compute::Dispatch {
            program: &backend.programs[&key],
            groups: [
                command.x[1] as u32,
                command.x[2] as u32,
                command.x[3] as u32,
            ],
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
                key.0,
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
        backend.stat(
            "stat: dispatch renderer microseconds",
            started.elapsed().as_micros() as u64,
        );
        self.trace_targets(&images_written, &[], &format!("dispatch {:016x}", key.0))?;
        if let Err(error) = self.trace_memory(core, &format!("dispatch {:016x}", key.0)) {
            eprintln!("memory trace: {error}");
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
            self.gpu.as_mut().unwrap().pending_storage.push((
                *token,
                guest..guest + length,
                binding,
                key.0,
                base,
            ));
        }
        Ok(())
    }

    /// `resolve_storage` for `size` bytes at GPU address `address` (memory
    /// a draw or dispatch is about to read).
    #[track_caller]
    fn resolve_gpu(&mut self, core: &mut Guest, address: u64, size: u64) -> Result<(), String> {
        if self
            .gpu
            .as_ref()
            .is_none_or(|backend| backend.pending_storage.is_empty())
        {
            return Ok(());
        }
        let guest = self.gpu_to_guest(address, size)?;
        self.resolve_storage(core, Some(guest..guest + size))
    }

    /// Store finished dispatch results to guest memory: all of them
    /// (`None`), or, when any overlaps `range`, all of them (results come
    /// back in dispatch order, so an earlier one never lands on a later).
    #[track_caller]
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
                backend.in_place_writes.clear();
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
                backend.stat("stat: GPU waits for page-table writes", 1);
                backend.renderer.flush()?;
                backend.page_table_writes = false;
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
                        add_range(&mut pool.gpu_dirty, piece);
                    }
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
        let mut wanted = Vec::new();
        for pool in backend.resident.values() {
            for dirty in &pool.gpu_dirty {
                let piece = match &range {
                    Some(range) => range.start.max(dirty.start)..range.end.min(dirty.end),
                    None if pool.uncached => dirty.clone(),
                    None => continue,
                };
                if piece.start < piece.end {
                    wanted.push((pool.base, piece));
                }
            }
        }
        if wanted.is_empty() {
            return Ok(());
        }
        let mut tokens = Vec::new();
        for (base, piece) in &wanted {
            let token = backend.renderer.read_resident(
                *base,
                piece.start - base,
                piece.end - piece.start,
            )?;
            tokens.push((token, *base, piece.clone()));
        }
        backend.renderer.flush()?;
        // Only these: pending dispatch and draw results stay for
        // `resolve_copies`.
        let ids: Vec<u64> = tokens.iter().map(|(token, ..)| *token).collect();
        let mut completed = backend.renderer.take_reads(&ids);
        backend.stat("stat: resident ranges copied back", wanted.len() as u64);
        // Which call site asked (`#[track_caller]` up the resolve chain).
        backend.stat(caller_stat(std::panic::Location::caller()), 1);
        for (token, base, piece) in tokens {
            let bytes = completed
                .remove(&token)
                .ok_or_else(|| format!("resident read {token} never completed"))?;
            core.memory
                .write(piece.start, &bytes)
                .map_err(|trap| format!("resident write-back: {trap:?}"))?;
            if let Some(pool) = self.gpu.as_mut().unwrap().resident.get_mut(&base) {
                remove_range(&mut pool.gpu_dirty, piece);
            }
        }
        Ok(())
    }

    /// A memory pool at guest `base` is finalized: its mirror is dropped
    /// (the memory may hold something else next). GPU writes the guest has
    /// not seen are lost with it, as the pool is.
    pub(super) fn gpu_forget_pool(&mut self, base: u64) {
        if let Some(backend) = self.gpu.as_mut() {
            backend.resident.remove(&base);
            // Its pages may be released and reused: shared spans of them
            // would keep the old ones.
            if let Some((size, _, offset)) = backend.shared_pools.remove(&base) {
                backend.renderer.release_file_range(offset, size);
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
    fn fetch_entry(
        &self,
        attribute: &BTreeMap<String, super::Args>,
        streams: &[BTreeMap<String, super::Args>],
        format: u32,
    ) -> Result<echora_shader_abi::VertexAttribute, String> {
        let value = |props: &BTreeMap<String, super::Args>, key: &str| {
            props.get(key).map_or(0, |args| args.x[0])
        };
        let encoded =
            vertex_format(format).ok_or_else(|| format!("vertex attribute format {format}"))?;
        let stream = value(attribute, "StreamIndex") as usize;
        // SetFormat(state, format, relative offset).
        let offset = attribute.get("Format").map_or(0, |args| args.x[1]);
        let (Some(props), Some(bind)) = (
            streams.get(stream),
            self.render.state.get(&format!("BindVertexBuffer/{stream}")),
        ) else {
            return Ok(Default::default());
        };
        if encoded.components == 0 || bind.x[2] == 0 {
            return Ok(Default::default());
        }
        Ok(echora_shader_abi::VertexAttribute {
            address: bind.x[2] + offset,
            stride: value(props, "Stride") as u32,
            divisor: value(props, "Divisor") as u32,
            format: encoded.encode(),
            size: bind.x[3].saturating_sub(offset).min(u64::from(u32::MAX)) as u32,
            reserved: [0; 2],
        })
    }

    /// The GPU page table programs reach global memory through
    /// (ECHORA_PAGE_TABLE; `echora_shader_abi::PageTable`), rebuilt when the
    /// memory pools change. A host-visible pool's pages point into shared
    /// spans of guest memory (the CPU's bytes, coherent as on the console);
    /// a GPU-only pool's into its resident copy. A page the GPU cannot reach
    /// stays 0 (reads 0).
    pub(super) fn page_table(
        &mut self,
        core: &mut Guest,
    ) -> Result<echora_shader_abi::PageTable, String> {
        use echora_shader_abi::{PAGE_SHIFT, PAGE_SIZE};
        if super::debug_var_os("ECHORA_PAGE_TABLE").is_none() {
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
        let pools: Vec<(u64, u64, u64, u64)> = self
            .memory_pools
            .iter()
            .filter_map(|address| self.objects.get(address))
            .filter(|object| object.kind == "MemoryPool")
            .filter_map(|pool| {
                let storage = pool.builder.get("Storage")?;
                let gpu = pool.state.get("GpuAddress")?.x[0];
                let flags = pool.builder.get("Flags").map_or(0x21, |flags| flags.x[0]);
                Some((storage.x[0], storage.x[1], gpu, flags))
            })
            .filter(|&(base, size, gpu, _)| base != 0 && size != 0 && gpu != 0)
            // GPU_NO_ACCESS (8): a physical pool the GPU reaches only
            // through a virtual pool's mappings, never at its own address.
            .filter(|&(_, _, _, flags)| flags & 8 == 0)
            .collect();
        let first = pools
            .iter()
            .map(|(_, _, gpu, _)| gpu >> PAGE_SHIFT)
            .min()
            .unwrap_or(0);
        let end = pools
            .iter()
            .map(|(_, size, gpu, _)| (gpu + size).div_ceil(PAGE_SIZE))
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
                pools.iter().find(|pool| pool.0 == traced),
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
            .map(|&(base, size, gpu, flags)| {
                (
                    (base, size, gpu, flags),
                    self.pool_share_signature(base, flags),
                )
            })
            .collect();
        let previous = self.gpu.as_mut().unwrap().page_build.take();
        let current: std::collections::HashSet<_> =
            signatures.iter().map(|(pool, _)| *pool).collect();
        let whole = match &previous {
            None => Some("stat: page table built whole (first)"),
            Some(built) if built.first != first => Some("stat: page table built whole (extent)"),
            Some(_) => None,
        };
        if let Some(reason) = whole {
            self.gpu.as_mut().unwrap().stat(reason, 1);
        }
        let previous = previous.filter(|_| whole.is_none());
        let mut built_pools = HashMap::new();
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
            let (_, size, gpu, _) = pool;
            let pages = (gpu >> PAGE_SHIFT).saturating_sub(first) as usize
                ..((gpu + size).div_ceil(PAGE_SIZE).saturating_sub(first)).min(count as u64)
                    as usize;
            if let Some(range) = entries.get_mut(pages) {
                range.fill(0);
            }
            self.gpu
                .as_mut()
                .unwrap()
                .stat("stat: page table pools unmapped", 1);
            // A pool kept over the same GPU pages is mapped again.
            built_pools.retain(|&(_, kept_size, kept_gpu, _), _| {
                kept_gpu + kept_size <= gpu || gpu + size <= kept_gpu
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
            let (base, size, gpu, flags) = pool;
            let backend = self.gpu.as_mut().unwrap();
            backend.stat("stat: page table pools mapped", 1);
            if built_pools.contains_key(&pool) {
                backend.stat("stat: page table pools mapped again", 1);
                backend.stat("stat: page table bytes mapped again", size);
            }
            let pages = (gpu >> PAGE_SHIFT).saturating_sub(first) as usize
                ..((gpu + size).div_ceil(PAGE_SIZE) - first).min(count as u64) as usize;
            if let Some(range) = entries.get_mut(pages.clone()) {
                range.fill(0);
            }
            self.map_pool_pages(core, base, size, gpu, flags, first, &mut entries)?;
            // A pool with unmapped pages, mapped without the guest's page
            // table, is mapped again by a build that has it.
            let complete = entries
                .get(pages)
                .is_some_and(|range| range.iter().all(|&entry| entry != 0));
            // Its signature after mapping (a GPU-only pool's resident
            // buffer exists from its first mapping on).
            let mapped = self.pool_share_signature(base, flags);
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
        let table = backend.renderer.set_page_table(first as u32, &entries)?;
        backend.page_build = Some(PageBuild {
            first,
            entries,
            pools: built_pools,
        });
        backend.stat(
            "stat: page table pinned MiB",
            backend.renderer.addressed_bytes() >> 20,
        );
        backend.page_table = Some((key, table));
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
    fn pool_share_signature(&self, base: u64, flags: u64) -> (u64, u64, usize) {
        let Some(backend) = self.gpu.as_ref() else {
            return (0, 0, 0);
        };
        if flags & 6 == 0 {
            return (backend.renderer.resident_address(base).unwrap_or(0), 0, 0);
        }
        let shared = backend
            .shared_pools
            .get(&base)
            .map_or((0, 0), |&(size, file, at)| (size, (file as u64) << 48 ^ at));
        let pieces = backend
            .shared_pieces
            .get(&base)
            .map_or(0, |pieces| pieces.len());
        (shared.0, shared.1, pieces)
    }

    /// Write pool `base`'s page entries (pages from `first`) into `entries`.
    #[allow(clippy::too_many_arguments)]
    fn map_pool_pages(
        &mut self,
        core: &mut Guest,
        base: u64,
        size: u64,
        gpu: u64,
        flags: u64,
        first: u64,
        entries: &mut [u64],
    ) -> Result<(), String> {
        use echora_shader_abi::{PAGE_SHIFT, PAGE_SIZE};
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
                let mut done = 0;
                while done < piece_length {
                    let length = (piece_length - done).min(CHUNK);
                    let address = self.gpu.as_mut().unwrap().renderer.address_span(
                        file,
                        at + done,
                        length,
                    )?;
                    if let Some(address) = address {
                        let first_page = (gpu + piece_offset + done) >> PAGE_SHIFT;
                        for page in 0..length.div_ceil(PAGE_SIZE) {
                            if let Some(entry) =
                                entries.get_mut((first_page + page - first) as usize)
                            {
                                *entry = address + page * PAGE_SIZE;
                            }
                        }
                    }
                    done += length;
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
                    .filter(|(shared_size, ..)| *shared_size >= offset + length)
                    .map(|&(_, file, at)| (file, at + offset));
                recorded_span = recorded.is_some();
                match recorded.or_else(|| core.memory.shared_span(base + offset, length)) {
                    Some((file, at)) => self
                        .gpu
                        .as_mut()
                        .unwrap()
                        .renderer
                        .address_span(file, at, length)?,
                    None => None,
                }
            } else {
                self.resident_pool_binding(core, base, size)?;
                self.gpu
                    .as_ref()
                    .unwrap()
                    .renderer
                    .resident_address(base)
                    .map(|address| address + offset)
            };
            // A chunk no single span shares (a page unmapped, or the
            // backing not contiguous): the runs of pages that are.
            if address.is_none() && host_visible {
                let mut runs: Vec<(u64, u64, i32, u64)> = Vec::new();
                let mut page = 0;
                while page * PAGE_SIZE < length {
                    let at = offset + page * PAGE_SIZE;
                    if let Some((file, file_at)) = core.memory.shared_span(base + at, PAGE_SIZE) {
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
    fn dump_dispatch(
        &self,
        core: &Guest,
        key: u64,
        stage: &Translated,
        groups: [u32; 3],
        uniforms: &HashMap<u32, Vec<u8>>,
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
        uniforms: &HashMap<(usize, u32), Vec<u8>>,
        textures: &HashMap<(usize, u32), (TextureSource, echora_gpu::draw::Sampler)>,
        page_table: echora_shader_abi::PageTable,
    ) {
        let groups = match command.name.as_str() {
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
        uniforms: &HashMap<(usize, u32), Vec<u8>>,
        textures: &HashMap<(usize, u32), (TextureSource, echora_gpu::draw::Sampler)>,
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
        for ((stage_index, binding), bytes) in &draw.uniforms {
            if *stage_index == index {
                write(format!("uniform-0-{binding}.bin"), bytes)?;
            }
        }
        for ((stage_index, binding), (source, _)) in &draw.textures {
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
        let present = |name: &str, default: u64| {
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
        let pools: Vec<(u64, u64)> = self
            .memory_pools
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
        self.gpu
            .as_mut()
            .unwrap()
            .stat("stat: pools tried for sharing", pools.len() as u64);
        for (base, size) in pools {
            if let Some((file, offset)) = core.memory.shared_span(base, size) {
                let backend = self.gpu.as_mut().unwrap();
                backend.stat("stat: pools shared whole", 1);
                backend.shared_pools.insert(base, (size, file, offset));
                backend.share_generation += 1;
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
            backend.share_generation += 1;
        }
    }

    /// The buffer and offset that share `length` guest bytes at `guest` with
    /// the GPU in place, when they lie in a shared memory pool and the span
    /// imports; None otherwise (the caller copies).
    fn shared_binding(
        &mut self,
        guest: u64,
        length: u64,
    ) -> Result<Option<(vk::Buffer, u64)>, String> {
        let backend = self.gpu.as_mut().unwrap();
        let Some((file, offset)) = backend
            .shared_pools
            .iter()
            .find(|(base, (size, _, _))| **base <= guest && guest + length <= **base + *size)
            .map(|(base, &(_, file, offset))| (file, offset + (guest - base)))
        else {
            return Ok(None);
        };
        backend.renderer.import_file(file, offset, length)
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
            .map(|pool| (pool.base, pool.size, if pool.uncached { 2 } else { 0 }));
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
                pool.gpu_dirty
                    .iter()
                    .any(|dirty| dirty.start <= guest && guest + length <= dirty.end)
            });
        if uncached && !gpu_newer {
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
                guest_dirty: Vec::new(),
                gpu_dirty: Vec::new(),
            });
        if created || fresh || pool.size != size {
            *pool = ResidentPool {
                base,
                size,
                uncached: flags & 2 != 0,
                guest_dirty: vec![base..base + size],
                gpu_dirty: Vec::new(),
            };
        }
        if uncached && !gpu_newer {
            add_range(&mut pool.guest_dirty, guest..guest + length);
        }
        let uploads = std::mem::take(&mut pool.guest_dirty);
        for range in uploads {
            let bytes = read_storage(core, range.start, range.end - range.start);
            let backend = self.gpu.as_mut().unwrap();
            backend.stat("stat: dispatch storage bytes", bytes.len() as u64);
            backend
                .renderer
                .upload_resident(base, range.start - base, &bytes)?;
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
        let Some((buffer, offset)) = self.resident_pool_binding(core, guest, 16)? else {
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
        // Time spent in each phase (stats, microseconds).
        let phase_start = std::time::Instant::now();
        let phase = |nvn: &mut Self, name: &'static str, since: &mut std::time::Instant| {
            let now = std::time::Instant::now();
            nvn.gpu
                .as_mut()
                .unwrap()
                .stat(name, (now - *since).as_micros() as u64);
            *since = now;
        };
        let mut since = phase_start;
        let mesh_draw = command.name.contains("DrawMeshTasks");
        let ((keys, stages), slots, mesh_entries) = if mesh_draw {
            self.bound_mesh_program(core)?
        } else {
            let bound = self.bound_program(core)?;
            let slots = bound.1.iter().map(|stage| nvn_stage(stage.stage)).collect();
            (bound, slots, Vec::new())
        };
        let mesh_groups = if mesh_draw {
            Some(self.mesh_groups(core, command)?)
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
        let backend = self.gpu.as_mut().unwrap();
        if !backend.programs.contains_key(&keys) {
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
            let program = backend.renderer.create_program_specialized(
                stages.clone(),
                if base_filter {
                    &untraced
                } else {
                    &specialization
                },
            )?;
            backend.programs.insert(keys, program);
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
                    .create_program_specialized(stages.clone(), &specialization)?;
                backend
                    .programs
                    .insert((keys.0, keys.1 ^ TRACED_VARIANT), traced);
            }
        }
        // ECHORA_VERTEX_TRACE_REPORT=<stage key>[,...]: every 60 presents,
        // the records a traced program's stages wrote (from earlier draws).
        if let Ok(wanted) = super::debug_var("ECHORA_VERTEX_TRACE_REPORT")
            && wanted
                .split(',')
                .any(|key| key == format!("{:016x}", keys.0) || key == format!("{:016x}", keys.1))
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
                    for texture in entry[8..8 + 6 * 16].chunks_exact(6) {
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
        // Bindings, through the inferred map.
        let mut uniforms = HashMap::new();
        let mut textures = HashMap::new();
        let mut storage = HashMap::new();
        let mut storage_in_place = HashMap::new();
        // Copied storage buffers a stage writes: guest address and GPU
        // address, by (stage index, binding).
        let mut storage_written = HashMap::new();
        // Driver constant entries of bound storage buffers: (stage index,
        // c0 offset, GPU address, size).
        let mut storage_entries: Vec<(usize, u32, u64, u64)> = Vec::new();
        let (mut uniform_time, mut texture_time) = Default::default();
        let mut uniform_bytes = 0u64;
        for (index, stage) in stages.iter().enumerate() {
            let nvn = slots[index];
            for descriptor in &stage.descriptors {
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
                            .state
                            .get(&format!("BindUniformBuffer/{nvn}/{slot}"))
                            .ok_or_else(|| {
                                // The slots that are bound, by stage, and
                                // the program's stages (which bank it reads).
                                let mut bound: Vec<&str> = self
                                    .render
                                    .state
                                    .keys()
                                    .filter(|key| key.starts_with("BindUniformBuffer/"))
                                    .map(String::as_str)
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
                        self.resolve_gpu(core, address, size)?;
                        let started = std::time::Instant::now();
                        let bytes = self.gpu_bytes(core, address, size)?;
                        uniform_time += started.elapsed();
                        uniform_bytes += bytes.len() as u64;
                        uniforms.insert((index, descriptor.binding), bytes);
                    }
                    DescriptorKind::UniformBuffer if descriptor.index == 1 => {
                        let mut bytes = match mesh_entries.get(index) {
                            Some(&entry) => self.stage_constants_of(core, entry),
                            None => self.stage_constants(core, index as u32),
                        }
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
                            .get(&format!("BindStorageBuffer/{nvn}/{slot}"))
                            .ok_or_else(|| {
                                let bound: Vec<&str> = self
                                    .render
                                    .state
                                    .keys()
                                    .filter(|key| key.starts_with(&format!("BindStorageBuffer/{nvn}/")))
                                    .map(String::as_str)
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
                        if descriptor.flags & 1 == 0 {
                            if let Some((buffer, offset)) = self.shared_binding(guest, length)? {
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
                        let handle = self
                            .render
                            .state
                            .get(&format!("BindTexture/{nvn}/{slot}"))
                            .ok_or_else(|| format!("texture {nvn}/{slot} is not bound"))?
                            .x[3];
                        // Valid within a present (textures are revalidated
                        // once per present); registrations evict their ids.
                        let stamp = (
                            self.texture_epoch,
                            self.gpu.as_ref().unwrap().renderer.target_changes,
                            self.binding_serial,
                            self.presented_frames,
                        );
                        if self.binding_memo.0 != stamp {
                            self.binding_memo = (stamp, Default::default());
                        }
                        if let Some((source, sampler)) = self.binding_memo.1.get(&handle) {
                            textures
                                .insert((index, descriptor.binding), (source.clone(), *sampler));
                            self.gpu
                                .as_mut()
                                .unwrap()
                                .stat("stat: texture binding memo hits", 1);
                            continue;
                        }
                        self.gpu
                            .as_mut()
                            .unwrap()
                            .stat("stat: texture binding memo misses", 1);
                        let started = std::time::Instant::now();
                        let texture = self.handle_texture(handle).map_err(|error| {
                            format!(
                                "{error} ({:?} stage {:016x}, binding {})",
                                stage.stage,
                                keys_for_note.get(index).copied().unwrap_or(0),
                                descriptor.binding
                            )
                        })?;
                        if let (Some(base), Some(size)) = (
                            self.texture_storage(texture),
                            self.texture_storage_size(texture),
                        ) {
                            self.resolve_storage(core, Some(base..base + size))?;
                        }
                        let source = self.texture_source(core, texture)?;
                        texture_time += started.elapsed();
                        let source = self.through_view(source, handle);
                        if let Some(swizzle) = self.handle_swizzle(handle) {
                            if super::debug_var_os("ECHORA_GPU_FLOW").is_some() {
                                let note = format!("flow view swizzle {swizzle:?} on {texture:x}");
                                self.gpu_note(note);
                            }
                        }
                        let sampler = self.binding_sampler(handle, &source);
                        self.binding_memo
                            .1
                            .insert(handle, (source.clone(), sampler));
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

        // Each stage's c0 holds its bound storage buffers' addresses and
        // sizes (slot i at the stage's table + 16 * i, `c0_storage_table`),
        // as the driver keeps them: a program that
        // reaches global memory through the page table reads them there
        // itself, with no storage binding of its own.
        for (index, stage) in stages.iter().enumerate() {
            if stage.push_fields & echora_shader_abi::PUSH_PAGE_TABLE == 0 {
                continue;
            }
            let nvn = slots[index];
            for slot in 0..16u64 {
                if let Some(bind) = self
                    .render
                    .state
                    .get(&format!("BindStorageBuffer/{nvn}/{slot}"))
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
                if bytes.len() < entry + 12 {
                    bytes.resize((entry + 16).next_multiple_of(16), 0);
                }
                bytes[entry..entry + 8].copy_from_slice(&gpu.to_le_bytes());
                bytes[entry + 8..entry + 12].copy_from_slice(&(size as u32).to_le_bytes());
            }
        }
        // The driver's other words in each stage's c0: the largest point
        // size (0x610, the hardware's 2047; GLSLC clamps gl_PointSize to
        // it) and the draw's base instance (0x644; DrawElementsInstanced's
        // and DrawArraysInstanced's, 0 for other direct draws). Programs
        // read the base instance there to find per-instance data.
        let base_instance = match command.name.as_str() {
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
                    if bytes.len() < offset + 4 {
                        bytes.resize((offset + 4).next_multiple_of(16), 0);
                    }
                    bytes[offset..offset + 4].copy_from_slice(&word.to_le_bytes());
                }
            }
        }
        // Texture pool entries, for stages that read the pool.
        let mut pool = Vec::new();
        for (index, stage) in stages.iter().enumerate() {
            if stage.texture_pool_bindings != 0 {
                pool.extend(self.pool_entries(core, slots[index])?);
            }
        }
        phase(self, "stat: draw us bindings", &mut since);
        {
            let backend = self.gpu.as_mut().unwrap();
            let time: std::time::Duration = uniform_time;
            backend.stat("stat: draw us uniform reads", time.as_micros() as u64);
            let time: std::time::Duration = texture_time;
            backend.stat("stat: draw us texture sources", time.as_micros() as u64);
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
        for (location, props) in attribute_states.iter().enumerate() {
            if stages[0].loaded_generics >> location & 1 == 0 {
                continue;
            }
            let format = value(props, "Format") as u32;
            if pulling {
                vertex_fetch[location] = self.fetch_entry(props, &stream_states, format)?;
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
                stream: value(props, "StreamIndex") as u32,
                // SetFormat(state, format, relative offset).
                offset: props.get("Format").map_or(0, |args| args.x[1]) as u32,
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
        let (instances, first_instance) = match command.name.as_str() {
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
                    command.name.trim_start_matches("nvnCommandBuffer"),
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
        let (mut indices, mut count, mut first, mut base_vertex) = match command.name.as_str() {
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
                self.resolve_gpu(core, command.x[4], u64::from(count) * size)?;
                let bytes = self.gpu_bytes(core, command.x[4], u64::from(count) * size)?;
                let base = if command.name.ends_with("BaseVertex")
                    || command.name.ends_with("Instanced")
                {
                    command.x[5] as i32
                } else {
                    0
                };
                (wrap(bytes), count, 0, base)
            }
            _ if mesh_draw => (Indices::None, 0, 0, 0),
            other => return Err(format!("{other} is not modeled on the GPU")),
        };
        // A draw this large can only come from misread state; it would hold
        // the GPU for seconds (the kernel resets it past 10 s).
        if u64::from(count) * u64::from(instances.max(1)) > 1 << 24 {
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
            };
            let triangles: Vec<u8> = corners
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|[a, b, c, d]| [*a, *b, *c, *a, *c, *d])
                .flat_map(u32::to_le_bytes)
                .collect();
            count = (triangles.len() / 4) as u32;
            first = 0;
            indices = Indices::U32(triangles);
        }

        // One past the highest vertex the draw references.
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
        };
        // The lowest vertex it references: meshes that share one large
        // vertex buffer read their own range only (the draw's first or base
        // vertex moves down by as much).
        let lowest_index = |indices: &Indices| -> Option<u64> {
            match indices {
                Indices::None => None,
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
        let per_vertex = |props: &BTreeMap<String, super::Args>| {
            value(props, "Divisor") == 0 && value(props, "Stride") != 0
        };
        let shifted = !pulling && vertex_start > 0 && stream_states.iter().any(per_vertex);
        // Each stream's bytes per vertex the loaded attributes reach (past
        // the furthest attribute's offset and size): a stream no attribute
        // reads is not read, and a stride-0 one (the same element for every
        // vertex) only that far, not its whole buffer (full-screen passes
        // bound megabytes for three vertices).
        let mut needed: HashMap<u32, u64> = HashMap::new();
        for attribute in &attributes {
            let size = attribute_format_size(attribute.format);
            let end = u64::from(attribute.offset) + size;
            let entry = needed.entry(attribute.stream).or_default();
            *entry = (*entry).max(end);
        }
        let mut streams = Vec::new();
        for (index, props) in stream_states.iter().enumerate() {
            // Only the vertices this draw can reach are read.
            let stride = value(props, "Stride");
            let Some(&element) = needed.get(&(index as u32)) else {
                streams.push(VertexStream {
                    stride: stride as u32,
                    divisor: value(props, "Divisor") as u32,
                    data: Vec::new(),
                });
                continue;
            };
            let skip = if shifted && per_vertex(props) {
                vertex_start * stride
            } else {
                0
            };
            let data = match self.render.state.get(&format!("BindVertexBuffer/{index}")) {
                Some(bind) if bind.x[3] > skip => {
                    let divisor = value(props, "Divisor");
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
                    self.resolve_gpu(core, address, size)?;
                    self.gpu_bytes(core, address, size)?
                }
                _ => Vec::new(),
            };
            streams.push(VertexStream {
                stride: value(props, "Stride") as u32,
                divisor: value(props, "Divisor") as u32,
                data,
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
                    self.render.state.get(&format!("BindVertexBuffer/{index}")),
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
        if page_table.address != 0
            && stages.iter().any(|stage| {
                stage.push_fields & echora_shader_abi::PUSH_PAGE_TABLE != 0
                    && stage.push_fields & echora_shader_abi::PUSH_PAGE_TABLE_READ_ONLY == 0
            })
        {
            let backend = self.gpu.as_mut().unwrap();
            backend.page_table_writes = true;
            backend.page_table_wrote = true;
        }
        phase(self, "stat: draw us geometry", &mut since);
        let geometry_bytes = streams
            .iter()
            .map(|stream| stream.data.len())
            .sum::<usize>();
        if geometry_bytes > 1 << 20 && super::debug_var_os("ECHORA_GPU_FLOW").is_some() {
            let strides: Vec<(u32, u32, usize)> = streams
                .iter()
                .map(|stream| (stream.stride, stream.divisor, stream.data.len()))
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
        // Targets: the bound colors and depth, created at their sizes.
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
        let colors: Vec<u64> = self.render.colors[..bound].to_vec();
        // Each color target's mip level (`mip_targets`) and array
        // layer, from its view.
        let (color_levels, color_layers): (Vec<u32>, Vec<u32>) = self
            .render
            .colors
            .iter()
            .zip(self.render.color_levels.iter().chain(std::iter::repeat(&0)))
            .take(bound)
            .map(|(_, view)| {
                let level = if !super::mip_targets() {
                    0
                } else {
                    *view as u32
                };
                (level, (*view >> 32) as u32)
            })
            .unzip();
        // The format each color view renders its target as (an sRGB view
        // of a UNORM texture encodes).
        let color_formats: Vec<Option<vk::Format>> = (0..bound)
            .map(|index| {
                self.render
                    .color_formats
                    .get(index)
                    .copied()
                    .filter(|format| *format != 0)
                    .and_then(vk_format)
            })
            .collect();
        // A color texture that is a mip level of another render target (by
        // address) renders into that target's level.
        let (colors, color_levels): (Vec<u64>, Vec<u32>) = colors
            .iter()
            .zip(color_levels)
            .map(|(texture, level)| match self.resolve_target(*texture) {
                Some((key, base)) if *texture != 0 && key != *texture => (key, base + level),
                _ => (*texture, level),
            })
            .unzip();
        // A block-compressed texture viewed in an integer format one block
        // wide per texel (RGBA16UI over BC4) renders into its blocks.
        let mut color_formats = color_formats;
        let mut colors = colors;
        let mut extent = None;
        for (index, color) in colors.iter_mut().enumerate() {
            let Some(view) = color_formats.get(index).copied().flatten() else {
                continue;
            };
            if *color == 0 {
                continue;
            }
            if let Some((alias, size)) = self.block_alias_target(*color, view)? {
                *color = alias;
                color_formats[index] = None;
                extent.get_or_insert(size);
            }
        }
        let aliases: Vec<u64> = self.block_aliases.values().copied().collect();
        for (index, texture) in colors
            .iter()
            .enumerate()
            .filter(|(_, texture)| **texture != 0 && !aliases.contains(texture))
        {
            if self.depth_traced(*texture) {
                eprintln!(
                    "depth trace present {} draw {:016x}/{:016x} renders {texture:x} as view format {:?} level {:?}",
                    self.presented_frames,
                    keys.0,
                    keys.1,
                    self.render.color_formats.get(index),
                    color_levels.get(index)
                );
            }
            extent.get_or_insert(self.ensure_render_target(*texture)?);
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
        let snapshot = |key: &str| -> BTreeMap<String, super::Args> {
            self.render
                .state
                .get(key)
                .and_then(|bind| bind.states.first().cloned())
                .unwrap_or_default()
        };
        let arg = |props: &BTreeMap<String, super::Args>, key: &str, index: usize| {
            props.get(key).map(|args| args.x[index])
        };
        let polygon = snapshot("BindPolygonState");
        // The polygon state the draw binds (its address) and the props taken
        // for it, with their element's offset from that address (a nonzero
        // offset is another object's bag: the bound one has none).
        let depth_state = snapshot("BindDepthStencilState");
        let color_state = snapshot("BindColorState");
        let mask_state = snapshot("BindChannelMaskState");
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
            cull: match arg(&polygon, "CullFace", 0).unwrap_or(0) {
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
                arg(&polygon, "FrontFace", 0).unwrap_or(1),
                super::debug_var_os("ECHORA_GPU_INVERT_FACE").is_some(),
            ) {
                (0, false) | (1.., true) => vk::FrontFace::CLOCKWISE,
                _ => vk::FrontFace::COUNTER_CLOCKWISE,
            },
            polygon_mode: match arg(&polygon, "PolygonMode", 0).unwrap_or(2) {
                0 => vk::PolygonMode::POINT,
                1 => vk::PolygonMode::LINE,
                _ => vk::PolygonMode::FILL,
            },
            // ECHORA_GPU_NODEPTH=<fragment key>[,...] (debugging): those
            // programs draw without the depth test.
            depth_test: arg(&depth_state, "DepthTestEnable", 0).unwrap_or(0) != 0
                && !super::debug_var("ECHORA_GPU_NODEPTH").is_ok_and(|keys| {
                    keys.split(',')
                        .any(|key| key == format!("{:016x}", keys_for_note[1]))
                }),
            depth_write: arg(&depth_state, "DepthWriteEnable", 0).unwrap_or(0) != 0,
            // ECHORA_GPU_EQUAL_AS_LEQUAL=<fragment key>[,...] (debugging):
            // those programs' EQUAL depth tests as LESS_OR_EQUAL (tells a
            // depth that differs by rounding from one never written).
            depth_compare: match compare_op(arg(&depth_state, "DepthFunc", 0).unwrap_or(8)) {
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
            depth_bias: (arg(&polygon, "PolygonOffsetEnables", 0).unwrap_or(0) & 4 != 0)
                .then(|| self.render.state.get("SetPolygonOffsetClamp"))
                .flatten()
                .map(|bind| {
                    let float = |index: usize| f32::from_bits(bind.v[index] as u32);
                    [float(1), float(2), float(0)]
                }),
            stencil: (arg(&depth_state, "StencilTestEnable", 0).unwrap_or(0) != 0)
                .then(|| self.stencil_faces(&depth_state)),
            // MultisampleState SetAlphaToCoverageEnable(state, enable).
            alpha_to_coverage: arg(
                &snapshot("BindMultisampleState"),
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
        let blends: Vec<Blend> = (0..colors.len())
            .map(|target| {
                let mut blend = Blend::default();
                // SetBlendEnable(state, index, enable): each index's latest.
                if let Some(enable) = color_state.get(&format!("BlendEnable/{target}")) {
                    blend.enable = enable.x[1] != 0 && !no_blend;
                }
                let blend_state = snapshot(&format!("BindBlendState/{target}"));
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
                if let Some(mask) = mask_state.get(&format!("ChannelMask/{target}")) {
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
                blend
            })
            .collect();
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
        // Each traced binding's texture, in binding order.
        // The depth target's storage and extent, for the trace below.
        let depth_storage = depth.map(|texture| {
            (
                self.texture_storage(texture),
                self.texture_extent(texture).ok(),
                self.texture_storage(self.render.depth),
            )
        });
        let traced_textures = if super::debug_var("ECHORA_GPU_TRACE_PROGRAM").is_ok_and(|wanted| {
            wanted.split(',').any(|item| {
                item == format!("{:016x}", keys.1) || item == format!("{:016x}", keys.0)
            })
        }) {
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
        let bound: Vec<String> = if super::debug_var_os("ECHORA_GPU_TRACE_PROGRAM").is_some() {
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
                        .state
                        .iter()
                        .filter(|(key, _)| key.starts_with("BindUniformBuffer/"))
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
        // ECHORA_GPU_TRACE_PROGRAM=<fragment key hex>[,...]: also trace those
        // programs' draws, with their fixed-function state.
        let traced_program = super::debug_var("ECHORA_GPU_TRACE_PROGRAM").is_ok_and(|wanted| {
            wanted.split(',').any(|item| {
                item == format!("{:016x}", keys.1) || item == format!("{:016x}", keys.0)
            })
        });
        if traced_program {
            let line = format!(
                "trace textures present {} {:016x}/{:016x} base {base_vertex} buffer {:#x} {traced_textures}",
                self.presented_frames, keys.0, keys.1, command.x[0]
            );
            eprintln!("{line}");
            backend.note(line);
            // The colour state as snapshotted for this draw (blend enables).
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
        if traced_program
            || (backend.traced < 40 && super::debug_var_os("ECHORA_GPU_LOG").is_some())
        {
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
                .map(|stream| floats(&stream.data, if traced_program { 64 } else { 9 }))
                .unwrap_or_default();
            let mut uniform_summary: Vec<String> = uniforms
                .iter()
                .map(|(key, bytes)| {
                    let count = if traced_program { 16 } else { 8 };
                    format!("{key:?}:{}B {:?}", bytes.len(), floats(bytes, count))
                })
                .collect();
            uniform_summary.sort();
            let polygon_trace = format!(
                "polygon bound {:#x?} props {polygon:?}",
                self.render
                    .state
                    .get("BindPolygonState")
                    .map(|bind| bind.x[1])
            );
            let line = format!(
                "trace draw {:016x}/{:016x} {polygon_trace} {:?} count {count} indices {:?} first {first} base {base_vertex} instances {instances}+{first_instance} fetch {:x?} cull {:?} {:?} blends {:?} mesh {:?} depth {:?} {depth_storage:x?} test {} write {} {:?} viewport {:?} scissor {:?} colors {:x?} attrs {:?} v0 {vertices:?} ubo {uniform_summary:?} textures {}",
                keys.0,
                keys.1,
                topology,
                match &indices {
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
                    Indices::None => Vec::new(),
                },
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
            (
                stages.clone(),
                uniforms.clone(),
                textures.clone(),
                page_table,
            )
        });
        let vertex_fetch_dump = if stage_dump.is_some() {
            vertex_fetch.clone()
        } else {
            Vec::new()
        };
        let traced_variant = (keys.0, keys.1 ^ TRACED_VARIANT);
        let program_keys = if super::debug_var("ECHORA_VERTEX_TRACE_BASE")
            .ok()
            .and_then(|value| value.parse::<i32>().ok())
            == Some(base_vertex)
            && backend.programs.contains_key(&traced_variant)
        {
            traced_variant
        } else {
            keys
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
            textures,
            storage,
            storage_in_place,
            fixed,
            page_table,
            vertex_fetch,
            pool,
        };
        let probe_target = draw.colors.iter().copied().find(|key| *key != 0);
        let colors_written: Vec<u64> = draw
            .colors
            .iter()
            .copied()
            .filter(|key| *key != 0)
            .collect();
        for key in &colors_written {
            *self.frame_target_draws.entry(*key).or_default() += 1;
        }
        let color_places: Vec<(u32, u32)> = draw
            .colors
            .iter()
            .enumerate()
            .filter(|(_, key)| **key != 0)
            .map(|(index, _)| {
                (
                    draw.color_levels.get(index).copied().unwrap_or(0),
                    draw.color_layers.get(index).copied().unwrap_or(0),
                )
            })
            .collect();
        let (draw_count, draw_instances) = (draw.count, draw.instances);
        let probe_level = draw.color_levels.first().copied().unwrap_or(0);
        let setup_end = std::time::Instant::now();
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
            backend.pending_storage.push((
                token,
                guest..guest + bytes.len() as u64,
                key.1,
                keys.1,
                base,
            ));
        }
        if storage_results > 0 {
            // (Not `stat`: the draw still borrows the programs.)
            *backend
                .notes
                .entry("stat: draw storage results".into())
                .or_default() += storage_results;
        }
        let timings = [
            (setup_end - since).as_micros() as u64,
            setup_end.elapsed().as_micros() as u64,
        ];
        let draw_label = format!("draw {:016x}/{:016x}", keys.0, keys.1);
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
        backend.stat("stat: draw us setup", timings[0]);
        backend.stat("stat: draw us renderer", timings[1]);
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
            for key in &colors_written {
                let storage = self.texture_storage(*key).unwrap_or(0);
                self.target_writers
                    .insert(*key, (keys.1, self.presented_frames, storage));
            }
        }
        self.trace_targets(&colors_written, &color_places, &draw_label)?;
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

/// Uploads larger than this are re-checked after a flush by a dense sample
/// (see `Backend::dense_samples`), not by hashing every flushed byte.
const DENSE_SAMPLE_ABOVE: u64 = 1 << 20;

/// A hash of 4096 64-byte pieces spread evenly over `length` bytes at
/// `base` (a whole-storage hash costs more than the work it saves).
fn dense_sample(core: &Guest, base: u64, length: u64) -> u64 {
    let mut piece = [0u8; 64];
    let mut sample = length;
    for index in 0..4096u64 {
        let offset = length.saturating_sub(64) * index / 4095;
        if core.memory.read(base + offset, &mut piece).is_err() {
            piece.fill(0);
        }
        sample = sample.rotate_left(7) ^ hash_bytes(&piece);
    }
    sample
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
            std::cell::RefCell::new(HashMap::new());
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

fn add_range(ranges: &mut Vec<std::ops::Range<u64>>, range: std::ops::Range<u64>) {
    let mut merged = range;
    ranges.retain(|existing| {
        if existing.start <= merged.end && merged.start <= existing.end {
            merged = merged.start.min(existing.start)..merged.end.max(existing.end);
            false
        } else {
            true
        }
    });
    let at = ranges.partition_point(|existing| existing.start < merged.start);
    ranges.insert(at, merged);
}

/// Remove `range` from a list of disjoint ranges.
fn remove_range(ranges: &mut Vec<std::ops::Range<u64>>, range: std::ops::Range<u64>) {
    let mut kept = Vec::with_capacity(ranges.len() + 1);
    for existing in ranges.drain(..) {
        if existing.end <= range.start || range.end <= existing.start {
            kept.push(existing);
            continue;
        }
        if existing.start < range.start {
            kept.push(existing.start..range.start);
        }
        if range.end < existing.end {
            kept.push(range.end..existing.end);
        }
    }
    *ranges = kept;
}

/// Uploads from CPU-uncached pools up to this size keep a copy of their
/// storage for exact change checks (see `Backend::shadows`).
const SHADOW_LIMIT: u64 = 16 << 20;

/// Granularity of upload storage hashes (see `Backend::storage_hashes`).
const STORAGE_CHUNK: u64 = 64 << 10;

/// `length` bytes of guest memory at `base` (unreadable bytes read as
/// zeros).
fn read_storage(core: &Guest, base: u64, length: u64) -> Vec<u8> {
    let mut bytes = vec![0u8; length as usize];
    if core.memory.read(base, &mut bytes).is_ok() {
        return bytes;
    }
    for (index, chunk) in bytes.chunks_mut(1 << 12).enumerate() {
        if core
            .memory
            .read(base + (index << 12) as u64, chunk)
            .is_err()
        {
            chunk.fill(0);
        }
    }
    bytes
}

/// A fast 64-bit hash of `bytes`.
/// A texture storage's sample: its shape, and 256 evenly spaced 64-byte
/// chunks of its `storage_size` bytes at `base` (see `guest_texture`).
pub(super) fn texture_sample(
    core: &Guest,
    base: u64,
    storage_size: u64,
    format: u32,
    [width, height, depth_or_layers]: [u32; 3],
) -> u64 {
    let mut sample = u64::from(format) << 48 ^ u64::from(width) << 24 ^ u64::from(height);
    sample ^= u64::from(depth_or_layers) << 40;
    let mut chunk = [0u8; 64];
    for index in 0..256u64 {
        let offset = (storage_size.saturating_sub(64)) * index / 255;
        if core.memory.read(base + offset, &mut chunk).is_ok() {
            sample = (sample ^ hash_bytes(&chunk)).wrapping_mul(0x100_0000_01b3);
        }
    }
    sample
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut hash = bytes.len() as u64 ^ 0x9e37_79b9_7f4a_7c15;
    let (words, tail) = bytes.as_chunks::<8>();
    for word in words {
        hash = (hash ^ u64::from_le_bytes(*word))
            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
            .rotate_left(29);
    }
    for byte in tail {
        hash = (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3);
    }
    hash
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
        assert_eq!(ranges, [0..5, 10..40]);
        remove_range(&mut ranges, 15..25);
        assert_eq!(ranges, [0..5, 10..15, 25..40]);
        remove_range(&mut ranges, 0..12);
        assert_eq!(ranges, [12..15, 25..40]);
        remove_range(&mut ranges, 100..200);
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
}

impl PointerProbe<'_, '_> {
    /// The eight bytes at `offset` of constant buffer `index`.
    fn word(&self, index: u32, offset: u32) -> Option<u64> {
        let at = offset as usize;
        let mut word = [0u8; 8];
        match index {
            0 | 2 => return None,
            1 => word.copy_from_slice(self.nvn.stage_constants(self.core, 5)?.get(at..at + 8)?),
            index => {
                let bind = self
                    .nvn
                    .render
                    .state
                    .get(&format!("BindUniformBuffer/5/{}", index - 3))?;
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
