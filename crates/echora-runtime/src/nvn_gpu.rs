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
    /// ECHORA_LOOP_TRACE (debugging): each traced compute program's
    /// counter buffer (`Renderer::debug_buffer` index and address) and the
    /// present it was last reported at.
    loop_traces: HashMap<u64, (usize, u64, u64)>,
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
    /// A compute program that reaches global memory through the page table
    /// was dispatched: its stores land in guest memory or a resident pool's
    /// copy with no range recorded (see `resolve_gpu_written`).
    page_table_writes: bool,
    /// Storage buffers kept on the GPU between dispatches (when guest
    /// memory cannot be imported), by guest address: see `Resident`.
    resident: HashMap<u64, ResidentPool>,
    /// Memory pools the GPU can read in place through guest memory's file
    /// (see `Renderer::import_file`), by guest base: size, file and the
    /// file offset of the base. Learned at presents (`share_new_pools`).
    shared_pools: HashMap<u64, (u64, i32, u64)>,
    /// Render targets by their texture's storage address, and the
    /// (renderer target changes, texture epoch) it was built at: texture
    /// lookups scanned every target twice per binding.
    target_index: HashMap<u64, Vec<u64>>,
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
            loop_traces: HashMap::new(),
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
            resident: HashMap::new(),
            shared_pools: HashMap::new(),
            target_index: HashMap::new(),
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

/// Stage keys (vertex, fragment) and the stages' translations.
type BoundProgram = ((u64, u64), Vec<Arc<Translated>>);
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
        // Bound in the vertex and tessellation slots (a mesh program without
        // a task in the vertex slot: `bound_mesh_program` gives its slot).
        Stage::Task => 0,
        Stage::Mesh => 4,
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
fn blend_factor(value: u64) -> vk::BlendFactor {
    use vk::BlendFactor as F;
    match value {
        1 => F::ZERO,
        3 => F::SRC_COLOR,
        4 => F::ONE_MINUS_SRC_COLOR,
        5 => F::SRC_ALPHA,
        6 => F::ONE_MINUS_SRC_ALPHA,
        7 => F::DST_ALPHA,
        8 => F::ONE_MINUS_DST_ALPHA,
        9 => F::DST_COLOR,
        10 => F::ONE_MINUS_DST_COLOR,
        11 => F::SRC_ALPHA_SATURATE,
        _ => F::ONE,
    }
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
        self.binding_serial += 1;
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
    ) -> Result<(), String> {
        self.ensure_render_target(texture)?;
        let scissor = self.render.scissor;
        let backend = self.gpu.as_mut().unwrap();
        backend.renderer.defined(texture);
        let value = vk::ClearColorValue {
            float32: color.map(f32::from_bits),
        };
        backend
            .renderer
            .clear_color_level(texture, level, layer, scissor, value)
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
    /// Texture structure `texture` now describes `storage` instead of
    /// `previous`: what the GPU rendered stays with the memory, as on the
    /// console. Its target moves to the retired key of `previous` (where a
    /// descriptor registered before still finds it), and a target kept
    /// for `storage` from an earlier texture there becomes its own.
    pub(super) fn gpu_move_texture(&mut self, texture: u64, previous: u64, storage: Option<u64>) {
        self.texture_epoch += 1;
        if let Some(backend) = self.gpu.as_mut() {
            backend.textures.remove(&texture);
            backend.texture_checked.remove(&texture);
            let renderer = &mut backend.renderer;
            let retired = super::retired_texture(previous);
            let result = renderer
                .move_target(texture, retired)
                .and_then(|_| match storage {
                    Some(storage) => renderer.move_target(super::retired_texture(storage), texture),
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
        if !self.presented_frames.is_multiple_of(600) {
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
            let translator = shader::Translator::from_env();
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
        let programs: Vec<u64> = self
            .render
            .state
            .iter()
            .filter(|(key, _)| key.starts_with("BindProgram@"))
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
        let (mesh, mesh_kind) =
            mesh.ok_or_else(|| format!("no mesh program is bound ({bound})"))?;
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
            slots.push(0);
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
        slots.push(if mesh_kind == 1 { 0 } else { 4 });
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
                TextureSource::Target(key) => format!("t{key:x}"),
                TextureSource::TargetView {
                    key,
                    base_level,
                    level_count,
                    base_layer,
                    layer_count,
                } => format!("t{key:x}[{base_level}+{level_count}/{base_layer}+{layer_count}]"),
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
                layers,
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
        let mut entries = Vec::new();
        for handle in handles {
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
            let source = self.texture_source(core, texture)?;
            let source = self.through_view(source, handle);
            let sampler_state = self.binding_sampler(handle, &source);
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
    /// as a level of detail clamp (sampling levels base..base+count of the
    /// whole image picks what a view of those levels picks).
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
                let (base, last) = (base as f32, (base + count - 1) as f32);
                sampler.lod[0] = sampler.lod[0].max(base).min(last);
                sampler.lod[1] = sampler.lod[1].clamp(base, last);
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
    /// layers (or cube faces) samples only those, and with
    /// ECHORA_GPU_MIPS, a view's mip levels only those.
    fn through_view(&self, source: TextureSource, handle: u64) -> TextureSource {
        let TextureSource::Target(key) = source else {
            return source;
        };
        let (base_level, level_count) = if super::debug_var_os("ECHORA_GPU_MIPS").is_some() {
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
    /// (1 unless ECHORA_GPU_MIPS is set: rendering into mip levels is
    /// still being validated; without it every level renders into level 0).
    fn texture_levels(&self, texture: u64) -> u32 {
        if super::debug_var_os("ECHORA_GPU_MIPS").is_none() {
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
    pub(super) fn texture_storage(&self, texture: u64) -> Option<u64> {
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
        for key in backend.renderer.target_keys() {
            if let Some(base) = self.texture_storage(key) {
                index.entry(base).or_default().push(key);
            }
        }
        let backend = self.gpu.as_mut().unwrap();
        backend.target_index = index;
        backend.target_index_stamp = Some((backend.renderer.target_changes, self.texture_epoch));
    }

    /// The render target (and its mip level) a texture object's storage
    /// is, as a GPU texture cache resolves addresses: the target itself,
    /// another target at the same address and size, or a target whose mip
    /// level `level` starts at this storage with this size (a separate
    /// texture object per level, as bloom chains use). With
    /// ECHORA_GPU_MIPS for levels above 0.
    fn resolve_target(&self, texture: u64) -> Option<(u64, u32)> {
        let renderer = &self.gpu.as_ref()?.renderer;
        if renderer.existing_target(texture).is_some() {
            return Some((texture, 0));
        }
        let mips = super::debug_var_os("ECHORA_GPU_MIPS").is_some();
        if !mips {
            return self.target_sharing_storage(texture).map(|key| (key, 0));
        }
        let base = self.texture_storage(texture)?;
        let extent = self.texture_extent(texture).ok()?;
        for key in renderer.target_keys() {
            let Some(target) = renderer.existing_target(key) else {
                continue;
            };
            let Some(target_base) = self.texture_storage(key) else {
                continue;
            };
            if target_base == base && (target.width, target.height) == extent {
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

    fn texture_source(&mut self, core: &Guest, texture: u64) -> Result<TextureSource, String> {
        self.refresh_target_index();
        // A mip level of another render target, by address.
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
        }
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
            if let Some((_, source, _)) = backend.textures.get(&texture) {
                return Ok(source.clone());
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
        let storage_size = if stride != 0 {
            stride * u64::from(height.div_ceil(info.block_height))
        } else {
            crate::nvn_format::block_linear_size(info, width, height, depth, layers, levels_set).0
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
        if let Some((sampled, source, _)) = self.gpu.as_ref().unwrap().textures.get(&texture) {
            if *sampled == sample && !stale && !shadow_changed {
                let source = source.clone();
                let backend = self.gpu.as_mut().unwrap();
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
                            info, width, height, depth, x, y, z,
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
            let level_offset =
                crate::nvn_format::block_linear_level_offset(info, width, height, depth, level);
            let level_shape = crate::nvn_format::level_block_shape(info, height, depth, level);
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
        // The structure was reused for other storage since registration:
        // the descriptor still names the texture it held then.
        let registered = entry.x[2];
        if registered != 0 && self.texture_storage(entry.x[0]) != Some(registered) {
            let retired = super::retired_texture(registered);
            if self.objects.contains_key(&retired) {
                return Ok(retired);
            }
        }
        Ok(entry.x[0])
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
            4 => ImageShape::D2 { layers: depth },
            8 | 9 if depth >= 6 => ImageShape::Cube {
                layers: depth / 6 * 6,
            },
            _ => ImageShape::D2 { layers: 1 },
        }
    }

    /// The render target of texture `texture`, created at its size,
    /// format, levels and layers.
    fn ensure_render_target(&mut self, texture: u64) -> Result<(u32, u32), String> {
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

    pub(super) fn gpu_copy_texture(
        &mut self,
        _core: &Guest,
        command: &Command,
    ) -> Result<(), String> {
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
        if self.gpu().unwrap().renderer.existing_target(from).is_none() {
            return Err("source is not a render target".into());
        }
        self.ensure_render_target(to)?;
        let renderer = &mut self.gpu.as_mut().unwrap().renderer;
        renderer.defined(to);
        renderer.copy_target(from, source, to, destination)
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
        // After a 16-byte head: root tables 0-7 (words pushed with
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
        let mut storage = HashMap::new();
        let mut storage_in_place = HashMap::new();
        // Each storage binding's GPU range (ECHORA_DUMP_DISPATCH).
        let mut storage_ranges = Vec::new();
        let mut written_back = Vec::new();
        let mut uniforms = HashMap::new();
        let mut textures = HashMap::new();
        let mut images = HashMap::new();
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
                    let level = if super::debug_var_os("ECHORA_GPU_MIPS").is_some() {
                        self.handle_view_levels(handle).map_or(0, |(base, _)| base)
                    } else {
                        0
                    };
                    images.insert(descriptor.binding, (texture, level));
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
                Some((texture, _)) => backend
                    .renderer
                    .existing_target(*texture)
                    .map(|target| target.format),
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
        if page_table.address != 0 {
            self.gpu.as_mut().unwrap().page_table_writes = true;
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
            if wanted && self.dumped_dispatches.insert(key.0) {
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
                if let Err(error) = self.dump_images(key.0, &read) {
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
    pub(super) fn resolve_gpu_written(
        &mut self,
        core: &mut Guest,
        range: std::ops::Range<u64>,
    ) -> Result<(), String> {
        if let Some(backend) = self.gpu.as_mut()
            && backend.page_table_writes
        {
            backend.stat("stat: GPU waits for page-table writes", 1);
            backend.renderer.flush()?;
            for pool in backend.resident.values_mut() {
                let piece = range.start.max(pool.base)..range.end.min(pool.base + pool.size);
                if piece.start < piece.end {
                    add_range(&mut pool.gpu_dirty, piece);
                }
            }
        }
        self.resolve_storage(core, Some(range))
    }

    /// Copy GPU-newer resident pool ranges back to the guest: what overlaps
    /// `range`, or with `None` (a CPU wait) what uncached pools hold (the
    /// CPU must invalidate a cached pool's range before reading it).
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
            .map_or(0, |backend| backend.shared_pools.len());
        let key = (shared, self.pool_epoch);
        if let Some((built, table)) = self.gpu.as_ref().and_then(|backend| backend.page_table)
            && built == key
        {
            return Ok(table);
        }
        let pools: Vec<(u64, u64, u64, u64)> = self
            .objects
            .values()
            .filter(|object| object.kind == "MemoryPool")
            .filter_map(|pool| {
                let storage = pool.builder.get("Storage")?;
                let gpu = pool.state.get("GpuAddress")?.x[0];
                let flags = pool.builder.get("Flags").map_or(0x21, |flags| flags.x[0]);
                Some((storage.x[0], storage.x[1], gpu, flags))
            })
            .filter(|&(base, size, gpu, _)| base != 0 && size != 0 && gpu != 0)
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
        let count = (end - first).min(16 << 20) as usize;
        let mut entries = vec![0u64; count];
        const CHUNK: u64 = 64 << 20;
        for (base, size, gpu, flags) in pools {
            let host_visible = flags & 6 != 0;
            let mut offset = 0;
            while offset < size {
                let length = (size - offset).min(CHUNK);
                let address = if host_visible {
                    let recorded = self
                        .gpu
                        .as_ref()
                        .and_then(|backend| backend.shared_pools.get(&base))
                        .filter(|(shared_size, ..)| *shared_size >= offset + length)
                        .map(|&(_, file, at)| (file, at + offset));
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
        }
        let backend = self.gpu.as_mut().unwrap();
        let table = backend.renderer.set_page_table(first as u32, &entries)?;
        backend.stat(
            "stat: page table pinned MiB",
            backend.renderer.addressed_bytes() >> 20,
        );
        backend.page_table = Some((key, table));
        Ok(table)
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
        if super::debug_var("ECHORA_DUMP_MEMORY").is_ok_and(|mode| mode == "all") {
            let first = u64::from(page_table.first_page);
            let mut index = 0;
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
                // A page the guest cannot read stays out (it reads as
                // unmapped in the simulator).
                if let Ok(bytes) = self.gpu_bytes(core, gpu, length) {
                    write(format!("memory-{base:x}.bin"), &bytes)?;
                }
                index = end;
            }
            return Ok(());
        }
        // Pools an address in the constant buffers falls in.
        let pools: Vec<(u64, u64)> = self
            .objects
            .values()
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
        Ok(root.join(format!("{key:016x}-{}", self.presented_frames)))
    }

    /// A dumped dispatch's textures and storage images, by binding: level
    /// 0 read back from the GPU as RGBA f32 texels (`image-0-B.bin`, layer
    /// 0) with `WIDTH HEIGHT DEPTH LAYERS` (`image-0-B.txt`). Integer
    /// formats read back as floats here.
    fn dump_images(&mut self, key: u64, read: &[(u32, TextureSource)]) -> Result<(), String> {
        let directory = self.dump_directory(key)?;
        let renderer = &mut self.gpu.as_mut().unwrap().renderer;
        for (binding, source) in read {
            let texels = match source {
                TextureSource::Upload { key, .. } => {
                    renderer.texture_view_for(source)?;
                    renderer.read_upload_level(*key, 0)?
                }
                TextureSource::Target(key) | TextureSource::TargetView { key, .. } => {
                    renderer.read_target_level(*key, 0)?
                }
            };
            let (width, height) = match source {
                TextureSource::Upload { width, height, .. } => (*width, *height),
                TextureSource::Target(key) | TextureSource::TargetView { key, .. } => renderer
                    .existing_target(*key)
                    .map(|target| (target.width, target.height))
                    .ok_or("no such target")?,
            };
            let count = (width * height) as usize;
            let bytes: Vec<u8> = texels
                .iter()
                .take(count)
                .flat_map(|texel| texel.iter().flat_map(|part| part.to_bits().to_le_bytes()))
                .collect();
            std::fs::write(directory.join(format!("image-0-{binding}.bin")), bytes)
                .map_err(|error| error.to_string())?;
            std::fs::write(
                directory.join(format!("image-0-{binding}.txt")),
                format!("{width} {height} 1 1\n"),
            )
            .map_err(|error| error.to_string())?;
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
        let pools: Vec<(u64, u64)> = self
            .objects
            .values()
            .filter(|object| object.kind == "MemoryPool")
            .filter_map(|pool| {
                pool.builder
                    .get("Storage")
                    .map(|storage| (storage.x[0], storage.x[1]))
            })
            .filter(|&(base, size)| {
                base != 0 && size != 0 && !backend.shared_pools.contains_key(&base)
            })
            .collect();
        for (base, size) in pools {
            if let Some((file, offset)) = core.memory.shared_span(base, size) {
                self.gpu
                    .as_mut()
                    .unwrap()
                    .shared_pools
                    .insert(base, (size, file, offset));
            }
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
            self.objects
                .values()
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
        if uncached {
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
        if uncached {
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
            let program = backend.renderer.create_program(stages.clone())?;
            backend.programs.insert(keys, program);
        }

        // ECHORA_GPU_SKIP_DRAW=<stage key>[,...] (debugging): draws whose
        // program has one of those stages are not recorded.
        if super::debug_var("ECHORA_GPU_SKIP_DRAW").is_ok_and(|skipped| {
            skipped.split(',').any(|skip| {
                skip == format!("{:016x}", keys.0) || skip == format!("{:016x}", keys.1)
            })
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
                            .ok_or_else(|| format!("uniform buffer {nvn}/{slot} is not bound"))?;
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
                        self.gpu.as_mut().unwrap().note(format!(
                            "driver constant buffer {} read as zeros ({used} bytes, {:?} stage {:016x})",
                            descriptor.index, stage.stage, keys_for_note.get(index).copied().unwrap_or(0)
                        ));
                        uniforms.insert((index, descriptor.binding), vec![0; used as usize]);
                    }
                    // Storage buffer i of graphics stage s (NVN numbering):
                    // its address and size at c0+0x110+0x400*s+16*i
                    // (inferred: vertex shaders read c0+0x170 and fragment
                    // shaders c0+0x570 where the application binds slot 6).
                    // The contents are read now; graphics stages' writes are
                    // not returned.
                    DescriptorKind::StorageBuffer
                        if descriptor.cbuf_index == 0
                            && descriptor.cbuf_offset >= 0x110 + 0x400 * nvn
                            && descriptor.cbuf_offset < 0x110 + 0x400 * nvn + 0x200
                            && descriptor.cbuf_offset % 16 == 0 =>
                    {
                        let slot = (descriptor.cbuf_offset - 0x110 - 0x400 * nvn) / 16;
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
                        let texture = self.handle_texture(handle)?;
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

        // Each stage's c0 holds every bound storage buffer's address and
        // size (graphics stage s, slot i at 0x110 + 0x400 * s + 16 * i), as
        // the driver keeps them: a program that reaches global memory
        // through the page table reads them there itself, with no storage
        // binding of its own.
        let bound_storage: Vec<(u64, u64, u64)> = (0..5u64)
            .flat_map(|nvn| (0..16u64).map(move |slot| (nvn, slot)))
            .filter_map(|(nvn, slot)| {
                let bind = self
                    .render
                    .state
                    .get(&format!("BindStorageBuffer/{nvn}/{slot}"))?;
                Some((0x110 + 0x400 * nvn + 16 * slot, bind.x[3], bind.x[4]))
            })
            .collect();
        for (index, stage) in stages.iter().enumerate() {
            if stage.push_fields & echora_shader_abi::PUSH_PAGE_TABLE == 0 {
                continue;
            }
            for &(offset, gpu, size) in &bound_storage {
                storage_entries.push((index, offset as u32, gpu, size));
            }
        }
        // Each stage's c0 holds its storage buffers' addresses and sizes,
        // which translated code reads to find offsets in the buffer.
        for (index, offset, gpu, size) in storage_entries {
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
        // order; implausible counts are refused.
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
            backend.note(format!(
                "instanced draw {} instances {bucket} base instance {}",
                command.name.trim_start_matches("nvnCommandBuffer"),
                if first_instance > 4096 {
                    "over 4096".to_string()
                } else {
                    first_instance.to_string()
                }
            ));
            if super::debug_var_os("ECHORA_GPU_NO_INSTANCED").is_some() {
                return Err(format!(
                    "{} is not drawn (ECHORA_GPU_NO_INSTANCED)",
                    command.name
                ));
            }
            if instances > 4096 || first_instance > 1 << 20 {
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
        let colors: Vec<u64> = self
            .render
            .colors
            .iter()
            .copied()
            .filter(|texture| *texture != 0)
            .collect();
        // Each color target's mip level (with ECHORA_GPU_MIPS) and array
        // layer, from its view.
        let (color_levels, color_layers): (Vec<u32>, Vec<u32>) = self
            .render
            .colors
            .iter()
            .zip(self.render.color_levels.iter().chain(std::iter::repeat(&0)))
            .filter(|(texture, _)| **texture != 0)
            .map(|(_, view)| {
                let level = if super::debug_var_os("ECHORA_GPU_MIPS").is_none() {
                    0
                } else {
                    *view as u32
                };
                (level, (*view >> 32) as u32)
            })
            .unzip();
        // A color texture that is a mip level of another render target (by
        // address) renders into that target's level.
        let (colors, color_levels): (Vec<u64>, Vec<u32>) = colors
            .iter()
            .zip(color_levels)
            .map(|(texture, level)| match self.resolve_target(*texture) {
                Some((key, base)) if key != *texture => (key, base + level),
                _ => (*texture, level),
            })
            .unzip();
        let mut extent = None;
        for texture in &colors {
            extent.get_or_insert(self.ensure_render_target(*texture)?);
        }
        let depth = (self.render.depth != 0).then_some(self.render.depth);
        if let Some(texture) = depth {
            extent.get_or_insert(self.ensure_render_target(texture)?);
        }
        let depth_layer = self.render.depth_layer;
        let (width, height) = extent.ok_or("draw without render targets")?;
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
        let depth_state = snapshot("BindDepthStencilState");
        let color_state = snapshot("BindColorState");
        let mask_state = snapshot("BindChannelMaskState");
        let fixed = Fixed {
            topology,
            primitive_restart: false,
            cull: match arg(&polygon, "CullFace", 0).unwrap_or(0) {
                1 => vk::CullModeFlags::FRONT,
                2 => vk::CullModeFlags::BACK,
                3 => vk::CullModeFlags::FRONT_AND_BACK,
                _ => vk::CullModeFlags::NONE,
            },
            // NVN's FrontFace: 0 clockwise, 1 counter-clockwise, in window
            // coordinates (the upper-left origin's, as Vulkan's; the other
            // way the Pokedex showed its back cover and Ditto its inside).
            front_face: match arg(&polygon, "FrontFace", 0).unwrap_or(1) {
                0 => vk::FrontFace::CLOCKWISE,
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
            depth_compare: compare_op(arg(&depth_state, "DepthFunc", 0).unwrap_or(8)),
            viewport: [
                viewport[0],
                viewport[1] + viewport[3],
                viewport[2],
                -viewport[3],
                depth_range[0],
                depth_range[1],
            ],
            scissor,
            // The application passes 1 to SetDepthMode: NEAR_IS_ZERO (the
            // D3D range; its projection maps the near plane to z/w 1 and
            // about 17700 units to 0, reverse-Z), so clip z in [0, w] is
            // depth as it is (measured from the intro scene's matrix).
            depth_minus_one_to_one: false,
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
        };
        if let Some([units, clamp, factor]) = fixed.depth_bias {
            self.gpu.as_mut().unwrap().note(format!(
                "polygon offset units {units} factor {factor} clamp {clamp}"
            ));
        }
        // Blending: ColorState enables per target, each target's latest
        // BlendState its factors and equations, ChannelMaskState write
        // masks.
        let blends: Vec<Blend> = (0..colors.len())
            .map(|target| {
                let mut blend = Blend::default();
                // SetBlendEnable(state, index, enable): each index's latest.
                if let Some(enable) = color_state.get(&format!("BlendEnable/{target}")) {
                    blend.enable = enable.x[1] != 0;
                }
                let blend_state = snapshot(&format!("BindBlendState/{target}"));
                if let Some(func) = blend_state.get("BlendFunc") {
                    blend.color.0 = blend_factor(func.x[0]);
                    blend.color.1 = blend_factor(func.x[1]);
                    blend.alpha.0 = blend_factor(func.x[2]);
                    blend.alpha.1 = blend_factor(func.x[3]);
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
        if super::debug_var_os("ECHORA_GPU_FLOW").is_some() {
            let line = format!(
                "flow draw {:016x}/{:016x} count {count} reads {} writes {} level {:?} layers {:?} depth layer {} viewport {:?} depth {:x?} blend {:?} masks {:?} factors {:?}",
                keys.0,
                keys.1,
                self.describe_sources(textures.values().map(|(source, _)| source)),
                self.describe_targets(colors.iter().copied()),
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
                    .collect::<Vec<_>>()
            );
            self.sequence_note(&line);
            self.gpu.as_mut().unwrap().note(line);
        }
        // Each traced binding's texture, in binding order.
        let traced_textures = if super::debug_var("ECHORA_GPU_TRACE_PROGRAM")
            .is_ok_and(|wanted| wanted == format!("{:016x}", keys.1))
        {
            let mut bound_textures: Vec<_> = textures.iter().collect();
            bound_textures.sort_by_key(|(key, _)| **key);
            bound_textures
                .iter()
                .map(|(key, (source, _))| {
                    format!("{key:?}={}", self.describe_sources(std::iter::once(source)))
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
                .filter(|(key, _)| key.starts_with("BindUniformBuffer/"))
                .map(|(key, bind)| format!("{key}:{:#x}+{:#x}", bind.x[3], bind.x[4]))
                .collect()
        } else {
            Vec::new()
        };
        let backend = self.gpu.as_mut().unwrap();
        // ECHORA_GPU_TRACE_PROGRAM=<fragment key hex>: also trace that
        // program's draws, with their fixed-function state.
        let traced_program = super::debug_var("ECHORA_GPU_TRACE_PROGRAM")
            .is_ok_and(|wanted| wanted == format!("{:016x}", keys.1));
        if traced_program {
            backend.note(format!(
                "trace textures {:016x}/{:016x} {traced_textures}",
                keys.0, keys.1
            ));
            let line = format!(
                "trace program {:016x}/{:016x} bound {bound:?} depth test {} write {} {:?} stencil {:?} cull {:?} viewport {:?} scissor {:?} blends {:?} colors {:x?} uniforms {:?}",
                keys.0,
                keys.1,
                fixed.depth_test,
                fixed.depth_write,
                fixed.depth_compare,
                fixed.stencil,
                fixed.cull,
                fixed.viewport,
                fixed.scissor,
                blends,
                colors,
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
                .map(|(key, bytes)| format!("{key:?}:{}B {:?}", bytes.len(), floats(bytes, 8)))
                .collect();
            uniform_summary.sort();
            let line = format!(
                "trace draw {:016x}/{:016x} {:?} count {count} colors {:x?} attrs {:?} v0 {vertices:?} ubo {uniform_summary:?} textures {}",
                keys.0,
                keys.1,
                topology,
                colors,
                attributes
                    .iter()
                    .map(|a| (a.location, a.offset, a.format))
                    .collect::<Vec<_>>(),
                textures.len(),
            );
            backend.note(line);
        }
        let draw = Draw {
            program: &backend.programs[&keys],
            blends,
            mesh: mesh_groups,
            colors,
            color_levels,
            color_layers,
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
        let probe_target = draw.colors.first().copied();
        let colors_written = draw.colors.clone();
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
        assert_eq!(decoded(63), Some((3, L::Packed111110, N::Float, false)));
        assert_eq!(decoded(154), Some((4, L::Bits8, N::Unorm, true))); // BGRA8
        assert_eq!(decoded(50), None); // DEPTH16
        assert_eq!(decoded(66), None); // BC1
    }
}
