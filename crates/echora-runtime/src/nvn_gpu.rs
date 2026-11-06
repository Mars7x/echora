//! Host GPU execution of NVN work (ECHORA_GPU=1, with the shader recompiler
//! linked through ECHORA_SHADER_LIB): bound programs are translated, their
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
    Blend, Draw, Fixed, ImageShape, Indices, Program, Sampler, StencilFace, TextureSource,
    UploadRegion, VertexAttribute, VertexStream,
};
use echora_gpu::present::Presenter;
use echora_gpu::renderer::Renderer;
use echora_gpu::shader::{self, DescriptorKind, Parameters, Stage, Translated};

use super::{Command, Guest, Image, Nvn};

pub struct Backend {
    pub renderer: Renderer,
    /// Translations by stage key and the parameters that shape them.
    translations: HashMap<TranslationKey, Result<Arc<Translated>, String>>,
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
        if super::debug_var_os("ECHORA_GPU").is_none() {
            return Ok(None);
        }
        if !shader::AVAILABLE {
            return Err("ECHORA_GPU needs the shader recompiler (ECHORA_SHADER_LIB)".into());
        }
        let gpu = echora_gpu::vulkan::Gpu::new()?;
        let mut renderer = Renderer::new(gpu)?;
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
            shadows: HashMap::new(),
            shadow_quiet: HashMap::new(),
            pending_storage: Vec::new(),
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
/// compute workgroup size and shared memory.
type TranslationKey = (u64, [u32; 3], Option<[u64; 8]>, [u8; 32], [u32; 5]);

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

/// NVN stage index (as in BindTexture/BindUniformBuffer) of a stage.
fn nvn_stage(stage: Stage) -> u32 {
    match stage {
        Stage::Vertex => 0,
        Stage::Fragment => 1,
        Stage::Geometry => 2,
        Stage::TessControl => 3,
        Stage::TessEval => 4,
        Stage::Compute => 5,
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
            self.bags.len(),
            self.views.len(),
            self.finalized_storage.len(),
            self.images.len(),
            self.command_lists.len(),
            captured >> 20,
            self.recording.values().map(Vec::len).sum::<usize>(),
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
    pub(super) fn gpu_dump_targets(&mut self) -> Result<(), String> {
        let Some(directory) = super::debug_var_os("ECHORA_GPU_DUMP") else {
            return Ok(());
        };
        if !self.presented_frames.is_multiple_of(300) {
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
                    "{key:x} {width}x{height} mean {:.4} {:.4} {:.4} {:.4} max {max:.3} nonfinite {nan}\n",
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
    ) -> Result<Arc<Translated>, String> {
        let (code, key) = self
            .stage_code(core, gpu)
            .ok_or_else(|| format!("stage at {gpu:#x} is in no memory pool"))?;
        let backend = self.gpu.as_mut().unwrap();
        let resolution_scale = backend.renderer.resolution_scale().terms();
        backend
            .translations
            .entry((key, base, previous_stores, generic_input_types, compute))
            .or_insert_with(|| {
                let parameters = Parameters {
                    binding_base: base[0],
                    resolution_scale,
                    scaling_base: [base[1], base[2]],
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
                };
                shader::translate(&code, parameters, &mut shader::NoEnvironment)
                    .map(Arc::new)
                    .map_err(|error| format!("stage {key:016x}: {error}"))
            })
            .clone()
    }

    /// Stage keys and translations of the bound vertex and fragment programs.
    /// Constant buffer 1 of the stage bound at NVN stage `bit`: the
    /// shader's own constant data, which the compiler places in its code
    /// block (NVN2 control section: size at +0x3c, offset in the code at
    /// +0x40; from the reference implementation's GLSLC notes).
    fn stage_constants(&self, core: &Guest, bit: u32) -> Option<Vec<u8>> {
        let bind = self.render.state.get(&format!("BindProgram@{bit}"))?;
        let stage = self.program_stage(core, bind.x[1], bit).ok()?;
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
        let vertex = self.translate_stage(core, addresses[0], [0; 3], None, types, [0; 5])?;
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
        let base = pool.builder["Storage"].x[0] + storage.x[1];
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
        let mut sample = u64::from(format) << 48 ^ u64::from(width) << 24 ^ u64::from(height);
        sample ^= u64::from(depth_or_layers) << 40;
        let mut chunk = [0u8; 64];
        for index in 0..if kept_sample.is_some() { 0 } else { 256u64 } {
            let offset = (storage_size.saturating_sub(64)) * index / 255;
            if core.memory.read(base + offset, &mut chunk).is_ok() {
                sample = (sample ^ hash_bytes(&chunk)).wrapping_mul(0x100_0000_01b3);
            }
        }
        let sample = kept_sample.unwrap_or(sample);
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
        // The whole storage, read once: untiling reads it block by block.
        let raw = read_storage(core, base, storage_size);
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
                for z in 0..level_depth {
                    for row in 0..rows {
                        let line = &mut data[z as usize * slice_bytes + row as usize * row_bytes..]
                            [..row_bytes];
                        if stride != 0 {
                            copy_block(line, u64::from(row) * stride);
                            continue;
                        }
                        for column in 0..columns {
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
                                &mut line[column as usize * block..][..block],
                                layer_base + offset,
                            );
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
        let generation = texels.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
        });
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
    /// Submit the recorded work and wait for it (at each present).
    pub(super) fn gpu_submit(&mut self) -> Result<(), String> {
        match self.gpu.as_mut() {
            Some(backend) => backend.renderer.flush(),
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
        let stage = self.translate_stage(core, address, [0; 3], None, [0; 32], compute)?;
        if stage.stage != Stage::Compute {
            return Err(format!("bound compute stage is {:?}", stage.stage));
        }
        let key = (self.stage_code(core, address).unwrap().1, u64::MAX);
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
        let backend = self.gpu.as_mut().unwrap();
        if !backend.programs.contains_key(&key) {
            let program = backend.renderer.create_program(vec![stage.clone()])?;
            backend.programs.insert(key, program);
        }

        // The driver constant buffer c0 of a compute program is a table of
        // its resources, in this order: uniform buffers (16 bytes each:
        // address and size), storage buffers (16 bytes), texture handles
        // (8 bytes), image handles (8 bytes). The counts are bytes 0xb1 (+1),
        // 0xb2, 0xb3 and 0xb6 of the control section, whose word 0xb8 holds
        // the table's size in its high half (inferred; it matches all 472
        // compute programs dumped).
        let mut header = [0u8; 12];
        core.memory
            .read(control + 0xb0, &mut header)
            .map_err(|trap| format!("compute control section: {trap:?}"))?;
        let storage_base = 16 * (u32::from(header[1]) + 1);
        let texture_base = storage_base + 16 * u32::from(header[2]);
        let image_base = texture_base + 8 * u32::from(header[3]);
        let table_end = image_base + 8 * u32::from(header[6]);
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
                "{:?} binding at c{}+{:#x} in compute: {layout}",
                descriptor.kind(),
                descriptor.cbuf_index,
                descriptor.cbuf_offset
            )
        };
        let mut c0 = vec![0u8; (table_end as usize).max(0x100)];
        let mut storage = HashMap::new();
        let mut written_back = Vec::new();
        let mut uniforms = HashMap::new();
        let mut textures = HashMap::new();
        let mut images = HashMap::new();
        for descriptor in &stage.descriptors {
            let entry = descriptor.cbuf_offset as usize;
            match descriptor.kind() {
                DescriptorKind::StorageBuffer => {
                    let slot = slot(descriptor, DescriptorKind::StorageBuffer)
                        .ok_or_else(|| unplaced(descriptor))?;
                    let bind = self
                        .render
                        .state
                        .get(&format!("BindStorageBuffer/5/{slot}"))
                        .ok_or_else(|| format!("storage buffer 5/{slot} is not bound: {layout}"))?;
                    let (gpu, size) = (bind.x[3], bind.x[4]);
                    let base = gpu & !15;
                    let length = gpu - base + size;
                    c0[entry..entry + 8].copy_from_slice(&gpu.to_le_bytes());
                    c0[entry + 8..entry + 12].copy_from_slice(&(size as u32).to_le_bytes());
                    let guest = self.gpu_to_guest(base, length)?;
                    self.resolve_storage(core, Some(guest..guest + length))?;
                    storage.insert(descriptor.binding, self.gpu_bytes(core, base, length)?);
                    // Buffers the shader only reads need no write-back.
                    if descriptor.flags & 1 != 0 {
                        written_back.push((descriptor.binding, base));
                    }
                    self.gpu
                        .as_mut()
                        .unwrap()
                        .stat("stat: dispatch storage bytes", length);
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
            textures,
            images,
        };
        let started = std::time::Instant::now();
        let lengths: HashMap<u32, u64> = dispatch
            .storage
            .iter()
            .map(|(binding, bytes)| (*binding, bytes.len() as u64))
            .collect();
        if echora_gpu::renderer::Renderer::labelling() {
            let label = format!(
                "present {} dispatch {:016x} groups {:?}",
                self.presented_frames, key.0, dispatch.groups
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
        let mut completed: HashMap<u64, Vec<u8>> = backend
            .renderer
            .take_completed_reads()
            .into_iter()
            .collect();
        let pending = std::mem::take(&mut backend.pending_storage);
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
        let (keys, stages) = self.bound_program(core)?;
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

        let keys_for_note = [keys.0, keys.1];
        // Bindings, through the inferred map.
        let mut uniforms = HashMap::new();
        let mut textures = HashMap::new();
        let mut storage = HashMap::new();
        // Driver constant entries of bound storage buffers: (stage index,
        // c0 offset, GPU address, size).
        let mut storage_entries: Vec<(usize, u32, u64, u64)> = Vec::new();
        let (mut uniform_time, mut texture_time) = Default::default();
        let mut uniform_bytes = 0u64;
        for (index, stage) in stages.iter().enumerate() {
            let nvn = nvn_stage(stage.stage);
            for descriptor in &stage.descriptors {
                match descriptor.kind() {
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
                        let mut bytes =
                            self.stage_constants(core, index as u32).unwrap_or_default();
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
                        self.resolve_storage(core, Some(guest..guest + length))?;
                        if descriptor.flags & 1 != 0 {
                            self.gpu
                                .as_mut()
                                .unwrap()
                                .note("graphics-stage storage buffer writes are not returned");
                        }
                        storage_entries.push((index, descriptor.cbuf_offset, gpu, size));
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
        // Attributes the shader loads but the state leaves without a format
        // (0): read as (0, 0, 0, 1), the usual value of an attribute with no
        // vertex data (inferred; NVN's documentation is not public).
        let mut unformatted = Vec::new();
        for (location, props) in attribute_states.iter().enumerate() {
            if stages[0].loaded_generics >> location & 1 == 0 {
                continue;
            }
            let format = value(props, "Format") as u32;
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
        // The draw itself.
        let mode = command.x[1];
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
        let shifted = vertex_start > 0 && stream_states.iter().any(per_vertex);
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
            fixed,
        };
        let probe_target = draw.colors.first().copied();
        let colors_written = draw.colors.clone();
        let (draw_count, draw_instances) = (draw.count, draw.instances);
        let probe_level = draw.color_levels.first().copied().unwrap_or(0);
        let setup_end = std::time::Instant::now();
        if echora_gpu::renderer::Renderer::labelling() {
            let label = format!(
                "present {} draw {:016x}/{:016x} count {} instances {} colors {:x?} depth {:x?}",
                self.presented_frames,
                keys.0,
                keys.1,
                draw.count,
                draw.instances,
                draw.colors,
                draw.depth
            );
            backend.renderer.label(label);
        }
        backend.renderer.draw(&draw)?;
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
