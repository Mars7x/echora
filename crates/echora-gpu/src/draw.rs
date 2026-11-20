//! Draws: linked shader objects for a vertex/fragment pair and the dynamic
//! state, buffers and descriptors one draw needs.
//!
//! Everything a draw reads is copied into a host-visible ring at record time,
//! so the guest may change its memory before the work is submitted.

use std::sync::Arc;

use ash::vk;

use crate::renderer::{Renderer, descriptor_buffer_range, has_stencil};
use crate::shader::{Descriptor, DescriptorKind, Translated};
use crate::vulkan::stage_flags;

/// A vertex/fragment pair, translated so its bindings share set 0.
pub struct Program {
    pub stages: Vec<Arc<Translated>>,
    pub(crate) layout: vk::DescriptorSetLayout,
    pub(crate) pipeline_layout: vk::PipelineLayout,
    pub(crate) shaders: Vec<vk::ShaderEXT>,
    shader_stages: Vec<vk::ShaderStageFlags>,
    /// With resolution scaling: the stages given the rescaling push
    /// constants, and each texture binding's (stage index, binding) by its
    /// scaling bit (the recompiler numbers texture descriptors in binding
    /// order across the stages).
    pub(crate) push_stages: vk::ShaderStageFlags,
    pub(crate) texture_bits: Vec<(usize, u32)>,
    /// The stages that read the GPU page table (bytes 112..128 of the push
    /// constants, `echora_shader_abi::PageTable`).
    /// Stages that read the ABI push-constant fields (`PUSH_ABI_FIELDS`):
    /// they take the whole block; `push_stages` excludes them.
    pub(crate) abi_stages: vk::ShaderStageFlags,
    /// A stage reads the texture pool: set 1 is the renderer's pool set
    /// (`Renderer::bind_pool`).
    pub(crate) pool: bool,
    dynamic_uniforms: Vec<u32>,
}

/// Bytes of the recompiler's rescaling push constants (RescalingLayout:
/// texture bits at 0, image bits at 16, the down factor at 24).
pub(crate) const RESCALING_BYTES: u32 = 32;

/// The rescaling push constants: bit `i` of the texture words set when
/// texture binding `i` samples a scaled target.
pub(crate) fn rescaling_constants(scaled_textures: &[bool], down_factor: f32) -> [u8; 32] {
    let mut words = [0u32; 8];
    for (bit, _) in scaled_textures
        .iter()
        .enumerate()
        .filter(|(_, scaled)| **scaled)
    {
        if bit < 128 {
            words[bit / 32] |= 1 << (bit % 32);
        }
    }
    words[6] = down_factor.to_bits();
    let mut bytes = [0u8; 32];
    for (chunk, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
        chunk.copy_from_slice(&word.to_le_bytes());
    }
    bytes
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VertexAttribute {
    pub location: u32,
    pub stream: u32,
    pub offset: u32,
    pub format: vk::Format,
}

#[derive(Clone, Debug, Default)]
pub struct VertexStream {
    pub stride: u32,
    pub divisor: u32,
    pub data: Vec<u8>,
    /// Guest bytes (address, length) read straight into the ring through
    /// the draw's `read_guest`, instead of copied into `data` first.
    pub guest: Option<(u64, usize)>,
}

impl VertexStream {
    /// The stream's length in bytes.
    pub fn len(&self) -> usize {
        self.guest.map_or(self.data.len(), |(_, length)| length)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Blend {
    pub enable: bool,
    pub color: (vk::BlendFactor, vk::BlendFactor, vk::BlendOp),
    pub alpha: (vk::BlendFactor, vk::BlendFactor, vk::BlendOp),
    pub write_mask: vk::ColorComponentFlags,
}

impl Default for Blend {
    fn default() -> Self {
        Self {
            enable: false,
            color: (
                vk::BlendFactor::ONE,
                vk::BlendFactor::ZERO,
                vk::BlendOp::ADD,
            ),
            alpha: (
                vk::BlendFactor::ONE,
                vk::BlendFactor::ZERO,
                vk::BlendOp::ADD,
            ),
            write_mask: vk::ColorComponentFlags::RGBA,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Fixed {
    pub topology: vk::PrimitiveTopology,
    pub primitive_restart: bool,
    pub cull: vk::CullModeFlags,
    pub front_face: vk::FrontFace,
    pub polygon_mode: vk::PolygonMode,
    pub depth_test: bool,
    pub depth_write: bool,
    pub depth_compare: vk::CompareOp,
    /// x, y, width, height, min depth, max depth.
    pub viewport: [f32; 6],
    /// x, y, width, height.
    pub scissor: [i32; 4],
    /// Front and back stencil state; None disables the test.
    pub stencil: Option<[StencilFace; 2]>,
    /// Clip-space Z runs from -1 to 1 (OpenGL style) instead of 0 to 1;
    /// honored when the device has depth clip control.
    pub depth_minus_one_to_one: bool,
    /// Polygon offset: constant (units), clamp and slope factors; None
    /// leaves depth unbiased.
    pub depth_bias: Option<[f32; 3]>,
    /// The depth target is a D24 format held as D32F: polygon offset
    /// units are D24 steps (2^-24) where the device can force that.
    pub depth_bias_d24: bool,
    /// Alpha to coverage (MultisampleState): with one sample a fragment
    /// is kept where its target 0 alpha covers it (leaf and grass cards'
    /// cut-out).
    pub alpha_to_coverage: bool,
    /// The blend constant (SetBlendColor) the CONSTANT factors read.
    pub blend_constants: [f32; 4],
}

/// The graphics state written to the current Vulkan command buffer.
#[derive(Default)]
pub(crate) struct GraphicsState {
    program: Option<vk::PipelineLayout>,
    raster: Option<RasterState>,
    blends: Vec<Blend>,
    // Valid for the current command buffer and program. Submitting resets it.
    dynamic_set: Option<DynamicSet>,
    pool_set: Option<vk::DescriptorSet>,
    vertex_input: Option<VertexInput>,
}

impl GraphicsState {
    pub(crate) fn reset(&mut self) {
        self.program = None;
        self.raster = None;
        self.blends.clear();
        self.reset_program_bindings();
    }

    fn reset_program_bindings(&mut self) {
        self.dynamic_set = None;
        self.pool_set = None;
        self.vertex_input = None;
    }
}

#[derive(Default)]
struct VertexInput {
    attributes: Vec<VertexAttribute>,
    streams: Vec<(u32, u32)>,
}

struct DynamicSet {
    set: vk::DescriptorSet,
    bindings: Vec<(u32, DescriptorSource)>,
}

enum DescriptorSource {
    Buffer(vk::DescriptorType, vk::DescriptorBufferInfo),
    Image(vk::DescriptorImageInfo),
    TexelBuffer(vk::BufferView),
}

impl DescriptorSource {
    fn same_as(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Buffer(a_kind, a), Self::Buffer(b_kind, b)) => {
                (a_kind, a.buffer, a.offset, a.range) == (b_kind, b.buffer, b.offset, b.range)
            }
            (Self::Image(a), Self::Image(b)) => {
                (a.sampler, a.image_view, a.image_layout)
                    == (b.sampler, b.image_view, b.image_layout)
            }
            (Self::TexelBuffer(a), Self::TexelBuffer(b)) => a == b,
            _ => false,
        }
    }

    fn write(&self, set: vk::DescriptorSet, binding: u32) -> vk::WriteDescriptorSet<'_> {
        let write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(binding);
        match self {
            Self::Buffer(kind, info) => write
                .descriptor_type(*kind)
                .buffer_info(std::slice::from_ref(info)),
            Self::Image(info) => write
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(info)),
            Self::TexelBuffer(view) => write
                .descriptor_type(vk::DescriptorType::UNIFORM_TEXEL_BUFFER)
                .texel_buffer_view(std::slice::from_ref(view)),
        }
    }
}

#[derive(PartialEq, Eq)]
struct RasterState {
    viewport: [u32; 6],
    scissor: [i32; 4],
    topology: vk::PrimitiveTopology,
    primitive_restart: bool,
    cull: vk::CullModeFlags,
    front_face: vk::FrontFace,
    polygon_mode: vk::PolygonMode,
    depth_test: bool,
    depth_write: bool,
    depth_compare: vk::CompareOp,
    stencil: Option<[StencilFace; 2]>,
    depth_minus_one_to_one: bool,
    depth_bias: Option<([u32; 3], bool)>,
    alpha_to_coverage: bool,
    blend_constants: [u32; 4],
}

/// One face's stencil test and update.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StencilFace {
    pub fail: vk::StencilOp,
    pub pass: vk::StencilOp,
    pub depth_fail: vk::StencilOp,
    pub compare: vk::CompareOp,
    pub compare_mask: u32,
    pub write_mask: u32,
    pub reference: u32,
}

/// What a texture binding samples.
#[derive(Clone, Debug)]
pub enum TextureSource {
    /// A render target, by its key.
    Target(u64),
    /// Mip levels `base_level`.. (`level_count` of them, 0: the rest) of a
    /// render target, as a texture view selects them.
    TargetView {
        key: u64,
        base_level: u32,
        level_count: u32,
        /// Array layers (or cube faces) `base_layer`.. (`layer_count` of
        /// them, 0: the rest).
        base_layer: u32,
        layer_count: u32,
    },
    /// A guest texture: texels already untiled and decoded to `format`,
    /// cached under `key` while `generation` is unchanged. `levels` gives
    /// each (mip level, layer) region's place in `texels`, level 0 first.
    Upload {
        key: u64,
        generation: u64,
        width: u32,
        height: u32,
        shape: ImageShape,
        format: vk::Format,
        texels: Arc<Vec<u8>>,
        levels: std::sync::Arc<[UploadRegion]>,
    },
}

impl TextureSource {
    /// Whether two sources name the same texels: an upload by its key and
    /// generation (its texels are cached under them and never compared).
    pub(crate) fn same_as(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Target(a), Self::Target(b)) => a == b,
            (
                Self::TargetView {
                    key,
                    base_level,
                    level_count,
                    base_layer,
                    layer_count,
                },
                Self::TargetView {
                    key: other_key,
                    base_level: other_base_level,
                    level_count: other_level_count,
                    base_layer: other_base_layer,
                    layer_count: other_layer_count,
                },
            ) => {
                (key, base_level, level_count, base_layer, layer_count)
                    == (
                        other_key,
                        other_base_level,
                        other_level_count,
                        other_base_layer,
                        other_layer_count,
                    )
            }
            (
                Self::Upload {
                    key,
                    generation,
                    width,
                    height,
                    shape,
                    format,
                    ..
                },
                Self::Upload {
                    key: other_key,
                    generation: other_generation,
                    width: other_width,
                    height: other_height,
                    shape: other_shape,
                    format: other_format,
                    ..
                },
            ) => {
                (key, generation, width, height, shape, format)
                    == (
                        other_key,
                        other_generation,
                        other_width,
                        other_height,
                        other_shape,
                        other_format,
                    )
            }
            _ => false,
        }
    }
}

/// The dimensionality of an image and its layers (or depth).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImageShape {
    D1 {
        layers: u32,
    },
    D2 {
        layers: u32,
    },
    D3 {
        depth: u32,
    },
    /// Six layers per cube.
    Cube {
        layers: u32,
    },
}

impl ImageShape {
    pub fn layers(self) -> u32 {
        match self {
            Self::D1 { layers } | Self::D2 { layers } | Self::Cube { layers } => layers.max(1),
            Self::D3 { .. } => 1,
        }
    }

    pub fn depth(self) -> u32 {
        match self {
            Self::D3 { depth } => depth.max(1),
            _ => 1,
        }
    }
}

/// One mip level of one layer of an upload: its texel extent and the byte
/// offset of its packed texels (slices of a 3D level follow each other).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UploadRegion {
    pub level: u32,
    pub layer: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub offset: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct Sampler {
    pub min_filter: vk::Filter,
    pub mag_filter: vk::Filter,
    pub mipmap: vk::SamplerMipmapMode,
    /// Sample only level 0 (filters without mipmapping).
    pub base_level_only: bool,
    pub address: [vk::SamplerAddressMode; 3],
    /// Level of detail clamp (minimum, maximum) and bias.
    pub lod: [f32; 3],
    /// Requested anisotropy; 1 disables it. Clamped to the host limit at creation.
    pub max_anisotropy: f32,
    /// Depth comparison (a shadow-map sampler): the reference is compared
    /// with the texel by this function, giving 0 or 1.
    pub compare: Option<vk::CompareOp>,
    /// Border colour for clamp-to-border addressing, as float bits (RGBA).
    pub border: [u32; 4],
    /// Channel sources (a texture view's swizzle); not part of the
    /// Vulkan sampler.
    pub components: vk::ComponentMapping,
    /// A texture view's format, when it differs from the texture's (its
    /// sRGB twin); not part of the Vulkan sampler.
    pub format: Option<vk::Format>,
    /// A texture view's mip levels (base, count) of an upload; not part of
    /// the Vulkan sampler (one guest sampler serves textures of any size).
    pub view_levels: Option<(u32, u32)>,
}

impl PartialEq for Sampler {
    fn eq(&self, other: &Self) -> bool {
        let components =
            |mapping: vk::ComponentMapping| [mapping.r, mapping.g, mapping.b, mapping.a];
        (
            self.min_filter,
            self.mag_filter,
            self.mipmap,
            self.base_level_only,
            self.address,
            self.lod.map(f32::to_bits),
            self.max_anisotropy.to_bits(),
            self.compare,
            self.border,
            components(self.components),
            self.format,
            self.view_levels,
        ) == (
            other.min_filter,
            other.mag_filter,
            other.mipmap,
            other.base_level_only,
            other.address,
            other.lod.map(f32::to_bits),
            other.max_anisotropy.to_bits(),
            other.compare,
            other.border,
            components(other.components),
            other.format,
            other.view_levels,
        )
    }
}

impl Default for Sampler {
    fn default() -> Self {
        Self {
            min_filter: vk::Filter::LINEAR,
            mag_filter: vk::Filter::LINEAR,
            mipmap: vk::SamplerMipmapMode::NEAREST,
            base_level_only: true,
            address: [vk::SamplerAddressMode::CLAMP_TO_EDGE; 3],
            lod: [0.0, vk::LOD_CLAMP_NONE, 0.0],
            max_anisotropy: 1.0,
            compare: None,
            border: [0; 4],
            components: vk::ComponentMapping::default(),
            format: None,
            view_levels: None,
        }
    }
}

pub enum Indices {
    None,
    U16(Vec<u8>),
    U32(Vec<u8>),
    /// Unmodified indices uploaded directly from guest memory.
    Guest {
        address: u64,
        bytes: u64,
        kind: vk::IndexType,
    },
}

pub struct Draw<'a> {
    pub program: &'a Program,
    pub colors: &'a [u64],
    /// The mip level each color target renders into.
    pub color_levels: &'a [u32],
    /// The array layer (or cube face) each color target renders into.
    pub color_layers: &'a [u32],
    /// The format each color target is rendered as, when its view names
    /// one (the target's sRGB or UNORM twin: an sRGB view encodes what is
    /// written).
    pub color_formats: &'a [Option<vk::Format>],
    pub depth: Option<u64>,
    /// The array layer the depth target renders into.
    pub depth_layer: u32,
    pub attributes: Vec<VertexAttribute>,
    pub streams: Vec<VertexStream>,
    pub indices: Indices,
    pub count: u32,
    pub first: u32,
    pub base_vertex: i32,
    pub instances: u32,
    pub first_instance: u32,
    /// Bytes for each uniform buffer binding, by (stage index in the
    /// program, binding).
    pub uniforms: &'a crate::FastMap<(usize, u32), Vec<u8>>,
    /// Uniform buffer bindings read from guest memory (address, length)
    /// straight into the upload ring as the draw is recorded, through
    /// `read_guest`, instead of copied into `uniforms` first.
    pub guest_uniforms: &'a crate::FastMap<(usize, u32), (u64, usize)>,
    /// Reads guest memory for `guest_uniforms` and guest vertex streams.
    pub read_guest: Option<&'a dyn Fn(u64, &mut [u8]) -> Result<(), String>>,
    /// Texture bindings by (stage index, binding).
    pub textures: &'a crate::FastMap<(usize, u32), (TextureSource, Sampler)>,
    /// Storage buffer contents by (stage index, binding). What a stage
    /// writes into them comes back (`Renderer::draw`).
    pub storage: crate::FastMap<(usize, u32), Vec<u8>>,
    /// Storage buffers shared with the GPU in place instead (see
    /// `Renderer::import_file`): buffer, offset and length.
    pub storage_in_place: crate::FastMap<(usize, u32), (vk::Buffer, u64, u64)>,
    pub fixed: Fixed,
    pub blends: &'a [Blend],
    /// A task/mesh pipeline's workgroups (the program's first stages are
    /// task and mesh): drawn with vkCmdDrawMeshTasks*EXT, with no vertex
    /// input.
    pub mesh: Option<MeshGroups>,
    /// The GPU page table, for stages that read it.
    pub page_table: echora_shader_abi::PageTable,
    /// Vertex pulling (`PUSH_VERTEX_FETCH`): the fetch table entries, one
    /// per generic input location, uploaded with the draw.
    pub vertex_fetch: Vec<echora_shader_abi::VertexAttribute>,
    /// The texture pool entries a program reading the pool may name.
    pub pool: std::sync::Arc<[crate::pool::PoolEntry]>,
}

/// The workgroups of a mesh draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshGroups {
    /// X, Y and Z group counts.
    Direct([u32; 3]),
    /// A VkDrawMeshTasksIndirectCommandEXT (three u32 group counts) at this
    /// offset of a device buffer (a resident pool), as the GPU last wrote it.
    Indirect { buffer: vk::Buffer, offset: u64 },
}

fn vk_error(context: &'static str) -> impl Fn(vk::Result) -> String {
    move |error| format!("{context}: {error}")
}

impl Renderer {
    /// Link translated stages (vertex first) into shader objects sharing one
    /// set-0 layout. Each stage must have been translated with its binding
    /// base after the previous stage's bindings.
    pub fn create_program(&self, stages: Vec<Arc<Translated>>) -> Result<Program, String> {
        self.create_program_specialized(stages, &[])
    }

    /// `create_program`, with specialization constants: for each stage (by
    /// index; missing stages take none), its (constant id, value) pairs.
    pub fn create_program_specialized(
        &self,
        stages: Vec<Arc<Translated>>,
        specialization: &[Vec<(u32, u32)>],
    ) -> Result<Program, String> {
        let objects = self.gpu.shader_objects()?;
        let mut descriptors: Vec<(Descriptor, vk::ShaderStageFlags)> = Vec::new();
        for stage in &stages {
            for descriptor in &stage.descriptors {
                descriptors.push((*descriptor, stage_flags(stage.stage)));
            }
        }
        let mut dynamic_uniforms = Vec::new();
        if stages
            .iter()
            .all(|stage| stage.stage != echora_shader_abi::Stage::Compute)
            && !std::env::var_os("ECHORA_DYNAMIC_UNIFORMS").is_some_and(|value| value == "0")
        {
            dynamic_uniforms.extend(descriptors.iter().filter_map(|(descriptor, _)| {
                (descriptor.kind() == DescriptorKind::UniformBuffer).then_some(descriptor.binding)
            }));
            if dynamic_uniforms.len() <= self.gpu.dynamic_uniform_limit as usize {
                dynamic_uniforms.sort_unstable();
            } else {
                dynamic_uniforms.clear();
            }
        }
        let layout = self
            .gpu
            .descriptor_layout(&descriptors, !dynamic_uniforms.is_empty())?;
        let pool = stages.iter().any(|stage| stage.texture_pool_bindings != 0);
        let mut layouts = vec![layout];
        if pool {
            match self.gpu.texture_pool_layout {
                Some(pool_layout) => layouts.push(pool_layout),
                None => {
                    // SAFETY: the layout is not used.
                    unsafe { self.gpu.device.destroy_descriptor_set_layout(layout, None) };
                    return Err("the device cannot hold the texture pool".into());
                }
            }
        }
        let push_stages = if !self.resolution_scale().is_one() {
            stages
                .iter()
                .fold(vk::ShaderStageFlags::empty(), |flags, stage| {
                    flags | stage_flags(stage.stage)
                })
        } else {
            vk::ShaderStageFlags::empty()
        };
        let mut texture_bits = Vec::new();
        for (index, stage) in stages.iter().enumerate() {
            let mut bindings: Vec<u32> = stage
                .descriptors
                .iter()
                .filter(|descriptor| descriptor.kind() == DescriptorKind::Texture)
                .map(|descriptor| descriptor.binding)
                .collect();
            bindings.sort_unstable();
            texture_bits.extend(bindings.into_iter().map(|binding| (index, binding)));
        }
        let abi_stages = stages
            .iter()
            .filter(|stage| stage.push_fields & echora_shader_abi::PUSH_ABI_FIELDS != 0)
            .fold(vk::ShaderStageFlags::empty(), |flags, stage| {
                flags | stage_flags(stage.stage)
            });
        // One range per stage (Vulkan allows no more): the whole ABI block
        // for stages that read its fields, the rescaling words for the
        // others.
        let push_stages = push_stages & !abi_stages;
        let mut push_ranges = Vec::new();
        if !abi_stages.is_empty() {
            push_ranges.push(vk::PushConstantRange {
                stage_flags: abi_stages,
                offset: 0,
                size: echora_shader_abi::PUSH_CONSTANTS_SIZE,
            });
        }
        if !push_stages.is_empty() {
            push_ranges.push(vk::PushConstantRange {
                stage_flags: push_stages,
                offset: 0,
                size: RESCALING_BYTES,
            });
        }
        let push_ranges = push_ranges.as_slice();
        let pipeline_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(push_ranges);
        // SAFETY: valid create info; destroyed with the program.
        let pipeline_layout =
            unsafe { self.gpu.device.create_pipeline_layout(&pipeline_info, None) }
                .map_err(vk_error("vkCreatePipelineLayout"))?;
        let codes: Vec<&[u8]> = stages
            .iter()
            .map(|stage| words_as_bytes(&stage.spirv))
            .collect();
        // Each stage's constants as 4-byte entries in one data block, with
        // the texture pool's capacity where a stage reads the pool.
        let specialization: Vec<Vec<(u32, u32)>> = stages
            .iter()
            .enumerate()
            .map(|(index, stage)| {
                let mut values = specialization.get(index).cloned().unwrap_or_default();
                if stage.spec_constants.iter().any(|constant| {
                    constant.meaning == echora_shader_abi::SpecMeaning::TexturePoolCapacity
                }) {
                    values.push((
                        echora_shader_abi::TEXTURE_POOL_CAPACITY_SPEC_ID,
                        echora_shader_abi::TEXTURE_POOL_CAPACITY,
                    ));
                }
                values
            })
            .collect();
        let constants: Vec<(Vec<vk::SpecializationMapEntry>, Vec<u8>)> = (0..stages.len())
            .map(|index| {
                let values = specialization.get(index).map_or(&[][..], Vec::as_slice);
                let entries = values
                    .iter()
                    .enumerate()
                    .map(|(at, &(id, _))| vk::SpecializationMapEntry {
                        constant_id: id,
                        offset: at as u32 * 4,
                        size: 4,
                    })
                    .collect();
                let data = values
                    .iter()
                    .flat_map(|(_, value)| value.to_le_bytes())
                    .collect();
                (entries, data)
            })
            .collect();
        let specialization_infos: Vec<_> = constants
            .iter()
            .map(|(entries, data)| {
                vk::SpecializationInfo::default()
                    .map_entries(entries)
                    .data(data)
            })
            .collect();
        // Compute stages run in 32-lane subgroups, as guest warps, where the
        // device can require it.
        let mut subgroup_32 =
            vk::ShaderRequiredSubgroupSizeCreateInfoEXT::default().required_subgroup_size(32);
        let compute_32 = self.gpu.compute_subgroup_32
            && stages.len() == 1
            && stages[0].stage == crate::shader::Stage::Compute;
        let has_task = stages
            .iter()
            .any(|stage| stage.stage == crate::shader::Stage::Task);
        let mut infos: Vec<_> = stages
            .iter()
            .enumerate()
            .map(|(index, stage)| {
                let next = stages
                    .get(index + 1)
                    .map_or(vk::ShaderStageFlags::empty(), |next| {
                        stage_flags(next.stage)
                    });
                let mut flags = vk::ShaderCreateFlagsEXT::empty();
                if stages.len() > 1 {
                    flags |= vk::ShaderCreateFlagsEXT::LINK_STAGE;
                }
                if stage.stage == crate::shader::Stage::Mesh && !has_task {
                    flags |= vk::ShaderCreateFlagsEXT::NO_TASK_SHADER;
                }
                vk::ShaderCreateInfoEXT::default()
                    .flags(flags)
                    .stage(stage_flags(stage.stage))
                    .next_stage(next)
                    .code_type(vk::ShaderCodeTypeEXT::SPIRV)
                    .code(codes[index])
                    .name(c"main")
                    .set_layouts(&layouts)
                    .push_constant_ranges(push_ranges)
                    .specialization_info(&specialization_infos[index])
            })
            .collect();
        if compute_32 {
            infos[0] = infos[0].push_next(&mut subgroup_32);
            if self.gpu.compute_full_subgroups && crate::vulkan::whole_subgroups(&stages[0].spirv) {
                infos[0] = infos[0].flags(vk::ShaderCreateFlagsEXT::REQUIRE_FULL_SUBGROUPS);
            }
        }
        let mut shaders = crate::shader_cache::create_shaders(
            objects,
            self.shader_cache.as_deref(),
            self.gpu.shader_binary_version,
            &infos,
            &format!(
                "{descriptors:?} {push_ranges:?} {compute_32} {specialization:?} {pool}{}",
                if dynamic_uniforms.is_empty() {
                    ""
                } else {
                    " dynamic uniforms"
                },
            ),
        )?;
        let mut shader_stages: Vec<_> = stages
            .iter()
            .map(|stage| stage_flags(stage.stage))
            .collect();
        let graphics_stages = [
            vk::ShaderStageFlags::VERTEX,
            vk::ShaderStageFlags::TESSELLATION_CONTROL,
            vk::ShaderStageFlags::TESSELLATION_EVALUATION,
            vk::ShaderStageFlags::GEOMETRY,
            vk::ShaderStageFlags::TASK_EXT,
            vk::ShaderStageFlags::MESH_EXT,
        ];
        let enabled = if self.gpu.mesh_shader.is_some() { 6 } else { 4 };
        if shader_stages != [vk::ShaderStageFlags::COMPUTE] {
            // Bind all graphics stages in one command, unbinding any
            // stages a previous program used and this program does not.
            for &stage in &graphics_stages[..enabled] {
                if !shader_stages.contains(&stage) {
                    shader_stages.push(stage);
                    shaders.push(vk::ShaderEXT::null());
                }
            }
        }
        Ok(Program {
            stages,
            layout,
            pipeline_layout,
            shaders,
            shader_stages,
            push_stages,
            texture_bits,
            abi_stages,
            pool,
            dynamic_uniforms,
        })
    }

    pub fn destroy_program(&mut self, program: Program) -> Result<(), String> {
        self.destroy_programs(std::iter::once(program))
    }

    /// Release several programs after their last queued use completes.
    pub fn destroy_programs(
        &mut self,
        programs: impl IntoIterator<Item = Program>,
    ) -> Result<(), String> {
        let mut programs = programs.into_iter().peekable();
        if programs.peek().is_none() {
            return Ok(());
        }
        self.flush()?;
        let objects = self.gpu.shader_objects()?;
        for program in programs {
            // SAFETY: flushed, so nothing uses this program.
            unsafe {
                for shader in program.shaders {
                    if shader != vk::ShaderEXT::null() {
                        objects.destroy_shader(shader, None);
                    }
                }
                self.gpu
                    .device
                    .destroy_pipeline_layout(program.pipeline_layout, None);
                self.gpu
                    .device
                    .destroy_descriptor_set_layout(program.layout, None);
            }
        }
        Ok(())
    }

    /// Record one draw.
    /// Record `draw`. Returns, for each copied storage buffer a stage
    /// writes, its (stage index, binding) and the token of its contents
    /// after the next flush (`take_reads`).
    pub fn draw(&mut self, draw: &Draw) -> Result<Vec<((usize, u32), u64)>, String> {
        let empty = match draw.mesh {
            Some(MeshGroups::Direct(groups)) => groups.contains(&0),
            Some(MeshGroups::Indirect { .. }) => false,
            None => draw.count == 0 || draw.instances == 0,
        };
        if empty {
            return Ok(Vec::new());
        }
        // Attachments render at one scale (before anything is recorded:
        // keeping a target unscaled flushes).
        static PROFILE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let profile = *PROFILE.get_or_init(|| {
            std::env::var_os("ECHORA_GPU_TIME").is_some_and(|mode| mode != "0")
                || std::env::var_os("ECHORA_TIMELINE").is_some()
        });
        let mut since = profile.then(std::time::Instant::now);
        let scale = self.attachment_scale(draw.colors, draw.depth)?;
        self.bound_submission()?;
        self.lap("stat: renderer ns submission", &mut since);
        let program = draw.program;
        // The texture pool's entries (writing them may flush).
        let pool = if program.pool {
            Some(self.bind_pool(&draw.pool)?)
        } else {
            None
        };
        self.lap("stat: renderer ns texture pool", &mut since);
        // Textures first: an upload may flush. Then room for everything
        // else, so nothing flushes while this draw is recorded.
        let mut bytes = 0;
        let mut descriptors = [0u32; 4];
        for (stage_index, stage) in program.stages.iter().enumerate() {
            for descriptor in &stage.descriptors {
                let key = (stage_index, descriptor.binding);
                match descriptor.kind() {
                    DescriptorKind::Texture => {
                        if let Some((source, _)) = draw.textures.get(&key) {
                            self.texture_view(source, descriptor.texture_type)?;
                        }
                        descriptors[1] += 1;
                    }
                    DescriptorKind::TextureBuffer => {
                        if let Some((source, _)) = draw.textures.get(&key) {
                            self.texel_buffer_view(source)?;
                        }
                        descriptors[1] += 1;
                    }
                    DescriptorKind::StorageBuffer => {
                        if !draw.storage_in_place.contains_key(&key) {
                            let length = draw.storage.get(&key).map_or(16, Vec::len);
                            bytes += descriptor_buffer_range(length) + 256;
                        }
                        descriptors[2] += 1;
                    }
                    _ => {
                        let length = match draw.uniforms.get(&key) {
                            Some(bytes) => bytes.len(),
                            None => draw.guest_uniforms.get(&key).map_or(16, |read| read.1),
                        };
                        bytes += descriptor_buffer_range(length) + 256;
                        descriptors[0] += 1;
                    }
                }
            }
        }
        bytes += draw
            .streams
            .iter()
            .map(|stream| stream.len() as u64 + 16)
            .sum::<u64>();
        bytes += match &draw.indices {
            Indices::None => 0,
            Indices::U16(data) | Indices::U32(data) => data.len() as u64 + 16,
            Indices::Guest { bytes, .. } => bytes + 16,
        };
        let vertex_fetch_bytes = draw.vertex_fetch.len() as u64 * 32;
        bytes += vertex_fetch_bytes + 256;
        if matches!(draw.mesh, Some(MeshGroups::Indirect { .. })) {
            bytes += 32;
        }
        self.lap("stat: renderer ns views and sizes", &mut since);
        self.reserve(bytes, descriptors)?;
        self.lap("stat: renderer ns reserve", &mut since);
        // Uploads and descriptor writes first: they record copies that must
        // precede rendering.
        let mut writes = Vec::with_capacity(descriptors.iter().sum::<u32>() as usize);
        let mut written = Vec::new();
        let mut dynamic_offsets = vec![0; program.dynamic_uniforms.len()];
        for (stage_index, stage) in program.stages.iter().enumerate() {
            for descriptor in &stage.descriptors {
                let key = (stage_index, descriptor.binding);
                let source = match descriptor.kind() {
                    DescriptorKind::UniformBuffer => {
                        let (buffer, offset, range) = match (
                            draw.uniforms.get(&key),
                            draw.guest_uniforms.get(&key),
                            draw.read_guest,
                        ) {
                            (Some(bytes), ..) => self.ring_upload_padded(bytes)?,
                            (None, Some(&(guest, length)), Some(read)) => {
                                let range = descriptor_buffer_range(length);
                                let (buffer, offset) =
                                    self.ring_upload_with(range, 256, |ring| {
                                        let (data, padding) = ring.split_at_mut(length);
                                        if !data.is_empty() {
                                            read(guest, data)?;
                                        }
                                        padding.fill(0);
                                        Ok(())
                                    })?;
                                (buffer, offset, range)
                            }
                            _ => self.ring_upload_padded(&[])?,
                        };
                        let mut info = vk::DescriptorBufferInfo {
                            buffer,
                            offset,
                            range,
                        };
                        if dynamic_offsets.is_empty() {
                            DescriptorSource::Buffer(vk::DescriptorType::UNIFORM_BUFFER, info)
                        } else {
                            // Vulkan consumes dynamic offsets in binding order.
                            let slot = program
                                .dynamic_uniforms
                                .binary_search(&descriptor.binding)
                                .expect("program layout includes this uniform binding");
                            dynamic_offsets[slot] = offset as u32;
                            info.offset = 0;
                            DescriptorSource::Buffer(
                                vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                                info,
                            )
                        }
                    }
                    DescriptorKind::StorageBuffer => {
                        let (buffer, offset, range) = if let Some(&(buffer, offset, length)) =
                            draw.storage_in_place.get(&key)
                        {
                            (buffer, offset, length.max(16).next_multiple_of(16))
                        } else {
                            let bytes = draw
                                .storage
                                .get(&key)
                                .map(Vec::as_slice)
                                .unwrap_or(&[0; 16]);
                            let (buffer, offset, range) = self.ring_upload_padded(bytes)?;
                            // Only buffers the stage writes come back
                            // (flags bit 0).
                            if descriptor.flags & 1 != 0 {
                                written.push((key, offset, bytes.len()));
                            }
                            (buffer, offset, range)
                        };
                        DescriptorSource::Buffer(
                            vk::DescriptorType::STORAGE_BUFFER,
                            vk::DescriptorBufferInfo {
                                buffer,
                                offset,
                                range,
                            },
                        )
                    }
                    DescriptorKind::Texture => {
                        let Some((source, sampler)) = draw.textures.get(&key) else {
                            return Err(format!(
                                "no texture for stage {stage_index} binding {}",
                                descriptor.binding
                            ));
                        };
                        let view =
                            self.texture_view_swizzled(source, descriptor.texture_type, sampler)?;
                        let sampler = self.sampler(sampler)?;
                        DescriptorSource::Image(vk::DescriptorImageInfo {
                            sampler,
                            image_view: view,
                            image_layout: vk::ImageLayout::GENERAL,
                        })
                    }
                    DescriptorKind::TextureBuffer => {
                        let Some((source, _)) = draw.textures.get(&key) else {
                            return Err(format!(
                                "no texel buffer for stage {stage_index} binding {}",
                                descriptor.binding
                            ));
                        };
                        DescriptorSource::TexelBuffer(self.texel_buffer_view(source)?)
                    }
                    other => return Err(format!("{other:?} bindings are not modeled")),
                };
                writes.push((descriptor.binding, source));
            }
        }
        self.lap("stat: renderer ns uniforms storage textures", &mut since);
        let matching = if !dynamic_offsets.is_empty()
            && self.graphics.program == Some(program.pipeline_layout)
        {
            self.graphics.dynamic_set.as_ref().filter(|previous| {
                previous.bindings.len() == writes.len()
                    && previous.bindings.iter().zip(&writes).all(
                        |((old_binding, old), (binding, source))| {
                            binding == old_binding && source.same_as(old)
                        },
                    )
            })
        } else {
            None
        };
        let reused = matching.is_some();
        let set = if let Some(previous) = matching {
            let set = previous.set;
            self.stat("stat: dynamic graphics descriptor sets reused", 1);
            set
        } else {
            let set = self.descriptor_set(program.layout)?;
            let updates: Vec<_> = writes
                .iter()
                .map(|(binding, source)| source.write(set, *binding))
                .collect();
            // SAFETY: the allocated set and its resources are live.
            unsafe { self.gpu.device.update_descriptor_sets(&updates, &[]) };
            set
        };
        self.lap("stat: renderer ns descriptor set", &mut since);
        // Vertex streams and indices.
        let mut vertex_buffers = Vec::new();
        for stream in &draw.streams {
            vertex_buffers.push(match (stream.guest, draw.read_guest) {
                (Some((guest, length)), Some(read)) if length > 0 => {
                    self.ring_upload_with(length as u64, 16, |bytes| read(guest, bytes))?
                }
                _ => self.ring_upload(&stream.data, 16)?,
            });
        }
        let index = match &draw.indices {
            Indices::None => None,
            Indices::U16(bytes) => Some((self.ring_upload(bytes, 16)?, vk::IndexType::UINT16)),
            Indices::U32(bytes) => Some((self.ring_upload(bytes, 16)?, vk::IndexType::UINT32)),
            &Indices::Guest {
                address,
                bytes,
                kind,
            } => {
                let read = draw.read_guest.ok_or("no reader for guest indices")?;
                Some((
                    self.ring_upload_with(bytes, 16, |ring| read(address, ring))?,
                    kind,
                ))
            }
        };

        // Targets this draw samples were maybe rendered just before.
        let sampled = draw
            .textures
            .values()
            .filter_map(|(source, _)| match source {
                TextureSource::Target(key) | TextureSource::TargetView { key, .. } => {
                    self.existing_target(*key).map(|target| target.image)
                }
                TextureSource::Upload { .. } => None,
            });
        if self.rendering_pending(sampled) {
            self.memory_barrier()?;
        }
        // An indirect mesh draw's record holds one group count (NVN mesh
        // draws are one-dimensional: DrawMeshTasks(first, count)), not
        // Vulkan's three: the count is copied, on the GPU, into a ring slot
        // whose y and z are 1, before the rendering pass.
        let mesh_arguments = match draw.mesh {
            Some(MeshGroups::Indirect { buffer, offset }) => {
                let words = [0u32, 1, 1].map(u32::to_le);
                let (ring, at) = self.ring_upload(words_as_bytes(&words), 16)?;
                // Bounded by the device's limit on the GPU (`mesh_count`);
                // copied as it is where buffers have no addresses.
                if !self.bounded_indirect(
                    (buffer, offset),
                    (ring, at),
                    crate::mesh_count::IndirectCounts::Mesh,
                )? {
                    self.stat("stat: mesh counts copied unbounded", 1);
                    self.copy_mesh_count((buffer, offset), (ring, at))?;
                }
                Some((ring, at))
            }
            _ => None,
        };
        let area = draw.fixed.scissor;
        // Reports a condition reads resolve outside the pass; the condition
        // then begins inside it (a pass change ends one begun there).
        self.prepare_condition()?;
        self.begin_rendering(
            draw.colors,
            draw.color_levels,
            draw.color_layers,
            draw.color_formats,
            draw.depth,
            draw.depth_layer,
            area,
        )?;
        self.apply_condition()?;
        let commands = self.recording_commands()?;
        let objects = self.gpu.shader_objects()?.clone();
        let device = self.gpu.device.clone();
        let fixed = &draw.fixed;
        if draw.mesh.is_some() && self.gpu.mesh_shader.is_none() {
            return Err("the device has no task and mesh stages".into());
        }
        let [x, y, width, height, near, far] = fixed.viewport;
        // A negative height (Vulkan 1.1) flips Y; the caller decides.
        // Viewport and scissor are in guest texels; scaled targets have
        // `scale` image texels per guest texel.
        let factor = scale.factor();
        let viewport = vk::Viewport {
            x: x * factor,
            y: y * factor,
            width: width * factor,
            height: height * factor,
            min_depth: near,
            max_depth: far,
        };
        let area = scale.rect(area.map(|value| value.max(0)));
        let scissor = vk::Rect2D {
            offset: vk::Offset2D {
                x: area[0],
                y: area[1],
            },
            extent: vk::Extent2D {
                width: area[2] as u32,
                height: area[3] as u32,
            },
        };
        let raster = RasterState {
            viewport: [
                viewport.x,
                viewport.y,
                viewport.width,
                viewport.height,
                near,
                far,
            ]
            .map(f32::to_bits),
            scissor: area,
            topology: fixed.topology,
            primitive_restart: fixed.primitive_restart,
            cull: fixed.cull,
            front_face: fixed.front_face,
            polygon_mode: fixed.polygon_mode,
            depth_test: fixed.depth_test && draw.depth.is_some(),
            depth_write: fixed.depth_write && draw.depth.is_some(),
            depth_compare: fixed.depth_compare,
            stencil: fixed.stencil.filter(|_| {
                draw.depth
                    .and_then(|key| self.existing_target(key))
                    .is_some_and(|target| has_stencil(target.format))
            }),
            depth_minus_one_to_one: fixed.depth_minus_one_to_one,
            depth_bias: fixed.depth_bias.map(|[constant, clamp, slope]| {
                let clamp = if self.gpu.depth_bias_clamp {
                    clamp
                } else {
                    0.0
                };
                (
                    [constant, clamp, slope].map(f32::to_bits),
                    fixed.depth_bias_d24,
                )
            }),
            alpha_to_coverage: fixed.alpha_to_coverage,
            blend_constants: fixed.blend_constants.map(f32::to_bits),
        };
        let rescaling = (!program.push_stages.is_empty()).then(|| {
            let scaled: Vec<bool> = program
                .texture_bits
                .iter()
                .map(|key| {
                    draw.textures
                        .get(key)
                        .is_some_and(|(source, _)| self.samples_scaled(source))
                })
                .collect();
            rescaling_constants(&scaled, 1.0 / factor)
        });
        // The ABI block: the vertex fetch table (uploaded now, before
        // rendering is recorded), the resolution scale and the page table.
        let abi = if program.abi_stages.is_empty() {
            None
        } else {
            let mut fetch = echora_shader_abi::VertexFetch {
                base_instance: draw.first_instance,
                ..Default::default()
            };
            if !draw.vertex_fetch.is_empty() {
                let (buffer, offset) = self.ring_upload_with(vertex_fetch_bytes, 256, |ring| {
                    for (entry, attribute) in ring
                        .as_chunks_mut::<32>()
                        .0
                        .iter_mut()
                        .zip(&draw.vertex_fetch)
                    {
                        *entry = attribute.to_bytes();
                    }
                    Ok(())
                })?;
                fetch.table = self
                    .buffer_address(buffer)
                    .ok_or("vertex pulling needs buffer device addresses")?
                    + offset;
                let count = draw
                    .vertex_fetch
                    .len()
                    .min(echora_shader_abi::VERTEX_ATTRIBUTES);
                fetch.attributes = if count == 32 {
                    u32::MAX
                } else {
                    (1 << count) - 1
                };
            }
            Some(
                echora_shader_abi::PushConstants {
                    vertex_fetch: fetch,
                    resolution_scale: [factor, factor, 1.0 / factor, 1.0 / factor],
                    page_table: draw.page_table,
                    ..Default::default()
                }
                .to_bytes(),
            )
        };
        self.lap("stat: renderer ns geometry pass state", &mut since);
        self.time_item(true);
        if pool.is_some() {
            self.record_texture_pool_use();
        }
        // SAFETY: recording inside rendering; every handle is live.
        unsafe {
            if self.graphics.program != Some(program.pipeline_layout) {
                objects.cmd_bind_shaders(commands, &program.shader_stages, &program.shaders);
                self.graphics.program = Some(program.pipeline_layout);
                self.graphics.reset_program_bindings();
            }
            device.cmd_bind_descriptor_sets(
                commands,
                vk::PipelineBindPoint::GRAPHICS,
                program.pipeline_layout,
                0,
                &[set],
                &dynamic_offsets,
            );
            if !reused && !dynamic_offsets.is_empty() {
                self.graphics.dynamic_set = Some(DynamicSet {
                    set,
                    bindings: writes,
                });
            }
            if let Some(pool) = pool
                && self.graphics.pool_set != Some(pool)
            {
                device.cmd_bind_descriptor_sets(
                    commands,
                    vk::PipelineBindPoint::GRAPHICS,
                    program.pipeline_layout,
                    echora_shader_abi::TEXTURE_POOL_SET,
                    &[pool],
                    &[],
                );
                self.graphics.pool_set = Some(pool);
                self.stat("stat: graphics texture pool binds", 1);
            }
            if let Some(constants) = &rescaling {
                // Both stage ranges include this prefix.
                device.cmd_push_constants(
                    commands,
                    program.pipeline_layout,
                    program.push_stages | program.abi_stages,
                    0,
                    constants,
                );
            }
            if let Some(block) = &abi {
                let start = if rescaling.is_some() {
                    RESCALING_BYTES
                } else {
                    0
                };
                device.cmd_push_constants(
                    commands,
                    program.pipeline_layout,
                    program.abi_stages,
                    start,
                    &block[start as usize..],
                );
            }
            if draw.mesh.is_none() {
                self.set_vertex_input(commands, &objects, draw);
            }
            for (index, (buffer, offset)) in vertex_buffers.iter().enumerate() {
                device.cmd_bind_vertex_buffers(commands, index as u32, &[*buffer], &[*offset]);
            }
            if self.graphics.raster.is_none() {
                objects.cmd_set_rasterizer_discard_enable(commands, false);
                objects.cmd_set_depth_clamp_enable(commands, false);
                objects.cmd_set_depth_bounds_test_enable(commands, false);
                objects.cmd_set_rasterization_samples(commands, vk::SampleCountFlags::TYPE_1);
                objects.cmd_set_sample_mask(commands, vk::SampleCountFlags::TYPE_1, &[u32::MAX]);
            }
            if self.graphics.raster.as_ref() != Some(&raster) {
                objects.cmd_set_viewport_with_count(commands, &[viewport]);
                objects.cmd_set_scissor_with_count(commands, &[scissor]);
                objects.cmd_set_primitive_topology(commands, raster.topology);
                objects.cmd_set_primitive_restart_enable(commands, raster.primitive_restart);
                objects.cmd_set_cull_mode(commands, raster.cull);
                objects.cmd_set_front_face(commands, raster.front_face);
                if self.gpu.depth_clip_control {
                    objects.cmd_set_depth_clip_negative_one_to_one(
                        commands,
                        raster.depth_minus_one_to_one,
                    );
                }
                objects.cmd_set_polygon_mode(commands, raster.polygon_mode);
                objects.cmd_set_alpha_to_coverage_enable(commands, raster.alpha_to_coverage);
                device.cmd_set_blend_constants(commands, &fixed.blend_constants);
                objects.cmd_set_depth_bias_enable(commands, raster.depth_bias.is_some());
                if let Some((bias, force_unorm)) = raster.depth_bias {
                    let [constant, clamp, slope] = bias.map(f32::from_bits);
                    match self.gpu.depth_bias_control.as_ref() {
                        Some(control) if force_unorm => {
                            let mut representation = vk::DepthBiasRepresentationInfoEXT::default()
                                .depth_bias_representation(
                                    vk::DepthBiasRepresentationEXT::LEAST_REPRESENTABLE_VALUE_FORCE_UNORM,
                                )
                                .depth_bias_exact(true);
                            let info = vk::DepthBiasInfoEXT::default()
                                .depth_bias_constant_factor(constant)
                                .depth_bias_clamp(clamp)
                                .depth_bias_slope_factor(slope)
                                .push_next(&mut representation);
                            (control.fp().cmd_set_depth_bias2_ext)(commands, &info);
                        }
                        _ => device.cmd_set_depth_bias(commands, constant, clamp, slope),
                    }
                }
                objects.cmd_set_depth_test_enable(commands, raster.depth_test);
                objects.cmd_set_depth_write_enable(commands, raster.depth_write);
                objects.cmd_set_depth_compare_op(commands, raster.depth_compare);
                objects.cmd_set_stencil_test_enable(commands, raster.stencil.is_some());
                if let Some(faces) = raster.stencil {
                    for (flags, face) in [
                        (vk::StencilFaceFlags::FRONT, faces[0]),
                        (vk::StencilFaceFlags::BACK, faces[1]),
                    ] {
                        objects.cmd_set_stencil_op(
                            commands,
                            flags,
                            face.fail,
                            face.pass,
                            face.depth_fail,
                            face.compare,
                        );
                        device.cmd_set_stencil_compare_mask(commands, flags, face.compare_mask);
                        device.cmd_set_stencil_write_mask(commands, flags, face.write_mask);
                        device.cmd_set_stencil_reference(commands, flags, face.reference);
                    }
                }
                self.graphics.raster = Some(raster);
            }
            let blends = (0..draw.colors.len().max(1))
                .map(|index| draw.blends.get(index).copied().unwrap_or_default());
            if !self.graphics.blends.iter().copied().eq(blends.clone()) {
                self.graphics.blends.clear();
                self.graphics.blends.extend(blends);
                let blends = &self.graphics.blends;
                let enables: Vec<_> = blends.iter().map(|blend| u32::from(blend.enable)).collect();
                let equations: Vec<_> = blends
                    .iter()
                    .map(|blend| vk::ColorBlendEquationEXT {
                        src_color_blend_factor: blend.color.0,
                        dst_color_blend_factor: blend.color.1,
                        color_blend_op: blend.color.2,
                        src_alpha_blend_factor: blend.alpha.0,
                        dst_alpha_blend_factor: blend.alpha.1,
                        alpha_blend_op: blend.alpha.2,
                    })
                    .collect();
                let masks: Vec<_> = blends.iter().map(|blend| blend.write_mask).collect();
                objects.cmd_set_color_blend_enable(commands, 0, &enables);
                objects.cmd_set_color_blend_equation(commands, 0, &equations);
                objects.cmd_set_color_write_mask(commands, 0, &masks);
            }
            if let Some(groups) = draw.mesh {
                let mesh = self.gpu.mesh_shader.as_ref().expect("checked above");
                match groups {
                    MeshGroups::Direct([x, y, z]) => mesh.cmd_draw_mesh_tasks(commands, x, y, z),
                    MeshGroups::Indirect { .. } => {
                        let (buffer, offset) = mesh_arguments.expect("prepared above");
                        mesh.cmd_draw_mesh_tasks_indirect(commands, buffer, offset, 1, 12);
                    }
                }
            } else if let Some(((buffer, offset), kind)) = index {
                device.cmd_bind_index_buffer(commands, buffer, offset, kind);
                device.cmd_draw_indexed(
                    commands,
                    draw.count,
                    draw.instances,
                    draw.first,
                    draw.base_vertex,
                    draw.first_instance,
                );
            } else {
                device.cmd_draw(
                    commands,
                    draw.count,
                    draw.instances,
                    draw.first,
                    draw.first_instance,
                );
            }
            // The pass stays open for the next draw into the same targets.
        }
        // ECHORA_PASS_PER_DRAW (debugging): end the pass after every draw,
        // as hang markers before each draw did (they kept the forest load
        // from hanging), without the markers' cache flushes and events.
        static PASS_PER_DRAW: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *PASS_PER_DRAW.get_or_init(|| std::env::var_os("ECHORA_PASS_PER_DRAW").is_some()) {
            self.end_pass();
        }
        self.time_item(false);
        self.lap("stat: renderer ns commands", &mut since);
        Ok(written
            .into_iter()
            .map(|(key, offset, length)| (key, self.defer_ring_read(offset, length)))
            .collect())
    }
}

impl Renderer {
    fn set_vertex_input(
        &mut self,
        commands: vk::CommandBuffer,
        objects: &ash::ext::shader_object::Device,
        draw: &Draw<'_>,
    ) {
        let streams = draw
            .streams
            .iter()
            .map(|stream| (stream.stride, stream.divisor));
        if self.graphics.vertex_input.as_ref().is_some_and(|last| {
            last.attributes == draw.attributes && last.streams.iter().copied().eq(streams.clone())
        }) {
            return;
        }
        let bindings: Vec<_> = streams
            .clone()
            .enumerate()
            .map(|(index, (stride, divisor))| {
                vk::VertexInputBindingDescription2EXT::default()
                    .binding(index as u32)
                    .stride(stride)
                    .input_rate(if divisor == 0 {
                        vk::VertexInputRate::VERTEX
                    } else {
                        vk::VertexInputRate::INSTANCE
                    })
                    .divisor(divisor.max(1))
            })
            .collect();
        let attributes: Vec<_> = draw
            .attributes
            .iter()
            .map(|attribute| {
                vk::VertexInputAttributeDescription2EXT::default()
                    .location(attribute.location)
                    .binding(attribute.stream)
                    .format(attribute.format)
                    .offset(attribute.offset)
            })
            .collect();
        // SAFETY: commands is recording with live vertex bindings.
        unsafe { objects.cmd_set_vertex_input(commands, &bindings, &attributes) };
        let last = self
            .graphics
            .vertex_input
            .get_or_insert_with(VertexInput::default);
        last.attributes.clone_from(&draw.attributes);
        last.streams.clear();
        last.streams.extend(streams);
        self.stat("stat: graphics vertex input sets", 1);
    }

    /// Count the time since `since` under `name` and restart it (the
    /// draw's phases, in the statistics).
    fn lap(&mut self, name: &'static str, since: &mut Option<std::time::Instant>) {
        if let Some(since) = since {
            let now = std::time::Instant::now();
            self.stat(name, (now - *since).as_nanos() as u64);
            *since = now;
        }
    }
}

pub(crate) fn words_as_bytes(words: &[u32]) -> &[u8] {
    // SAFETY: u32 has no padding and u8 has alignment 1.
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast(), words.len() * 4) }
}

impl Renderer {
    /// Copy an indirect mesh draw's group count from `source` into
    /// `target` as it is (no bound), on devices whose buffers have no
    /// addresses.
    fn copy_mesh_count(
        &mut self,
        (buffer, offset): (vk::Buffer, u64),
        (ring, at): (vk::Buffer, u64),
    ) -> Result<(), String> {
        let commands = self.commands()?;
        let copy = vk::BufferCopy {
            src_offset: offset,
            dst_offset: at,
            size: 4,
        };
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::INDIRECT_COMMAND_READ);
        // The count an earlier pass wrote is read by the copy.
        let written = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
        // SAFETY: recording, outside rendering; both ranges exist.
        unsafe {
            *self.stats.entry("stat: barriers in draw").or_default() += 1;
            self.gpu.device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[written],
                &[],
                &[],
            );
            self.gpu
                .device
                .cmd_copy_buffer(commands, buffer, ring, &[copy]);
            *self.stats.entry("stat: barriers in draw").or_default() += 1;
            self.gpu.device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::DRAW_INDIRECT,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::shader::Stage;
    use crate::vulkan::Gpu;

    pub(crate) fn spirv(source: &str, stage: naga::ShaderStage) -> Vec<u32> {
        let module = naga::front::wgsl::parse_str(source).unwrap();
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::TEXTURE_AND_SAMPLER_BINDING_ARRAY
                | naga::valid::Capabilities::IMMEDIATES,
        )
        .validate(&module)
        .unwrap();
        let options = naga::back::spv::Options {
            lang_version: (1, 3),
            ..Default::default()
        };
        let pipeline = naga::back::spv::PipelineOptions {
            shader_stage: stage,
            entry_point: "main".into(),
        };
        naga::back::spv::write_vec(&module, &info, &options, Some(&pipeline)).unwrap()
    }

    pub(crate) fn translated(
        stage: Stage,
        spirv: Vec<u32>,
        descriptors: Vec<Descriptor>,
    ) -> Arc<Translated> {
        Arc::new(Translated {
            stage,
            spirv,
            descriptors,
            loaded_generics: 0,
            stored_generics: 0,
            stores: [0; 8],
            frag_colors: 1,
            cbuf_used_sizes: [0; 18],
            abi_version: crate::shader::ABI_VERSION,
            spec_constants: Vec::new(),
            push_fields: 0,
            texture_pool_bindings: 0,
        })
    }

    #[test]
    fn a_triangle_draws_with_a_uniform_color() {
        let Ok(gpu) = Gpu::new() else {
            return; // No device (CI).
        };
        let mut renderer = Renderer::new(gpu).unwrap();
        renderer.set_resolution_scale(crate::renderer::Scale::new(2, 1).unwrap());
        renderer.keep_unscaled(7).unwrap();
        let vertex = spirv(
            "struct Rescaling { words: array<vec4<f32>, 2>, };
            var<immediate> scaling: Rescaling;
            @vertex fn main(@location(0) position: vec2<f32>) -> @builtin(position) vec4<f32> {
                return vec4<f32>(position * scaling.words[1].z, 0.0, 1.0);
            }",
            naga::ShaderStage::Vertex,
        );
        let fragment = spirv(
            "@group(0) @binding(0) var<uniform> color: vec4<f32>;
            @group(0) @binding(3) var<uniform> tint: vec4<f32>;
            struct Abi { words: array<vec4<u32>, 8>, };
            var<immediate> abi: Abi;
            @fragment fn main() -> @location(0) vec4<f32> {
                let blue = max(color.b, f32(abi.words[7].z) / 255.0);
                return vec4<f32>(color.r, tint.g, blue, tint.a);
            }",
            naga::ShaderStage::Fragment,
        );
        let uniform = Descriptor {
            kind: 0,
            binding: 0,
            count: 1,
            index: 3,
            ..Default::default()
        };
        let mut fragment = translated(
            Stage::Fragment,
            fragment,
            vec![
                Descriptor {
                    binding: 3,
                    index: 4,
                    ..uniform
                },
                uniform,
            ],
        );
        let has_pool = renderer.gpu.texture_pool_layout.is_some();
        let fragment_data = Arc::get_mut(&mut fragment).unwrap();
        fragment_data.texture_pool_bindings = u32::from(has_pool);
        fragment_data.push_fields = echora_shader_abi::PUSH_PAGE_TABLE;
        let program = renderer
            .create_program(vec![translated(Stage::Vertex, vertex, vec![]), fragment])
            .unwrap();
        renderer
            .target(7, 4, 4, vk::Format::R8G8B8A8_UNORM)
            .unwrap();
        renderer
            .clear_color(7, None, vk::ClearColorValue { float32: [0.0; 4] })
            .unwrap();
        // A triangle covering the left half: (-1,-1), (0,-1)... in NDC.
        let positions: [f32; 8] = [-1.0, -1.0, 0.0, -1.0, -1.0, 1.0, 0.0, 1.0];
        let tint: [f32; 4] = [0.0, 1.0, 0.0, 1.0];
        let red_only = [Blend {
            write_mask: vk::ColorComponentFlags::R,
            ..Blend::default()
        }];
        let blend = [Blend::default()];
        let make_uniforms = |color: Option<[f32; 4]>| {
            let bytes = |rgba: [f32; 4]| rgba.into_iter().flat_map(f32::to_le_bytes).collect();
            let mut uniforms = crate::FastMap::from_iter([((1, 3), bytes(tint))]);
            if let Some(color) = color {
                uniforms.insert((1, 0), bytes(color));
            }
            uniforms
        };
        let black = make_uniforms(Some([0.0; 4]));
        let red = make_uniforms(Some([1.0, 0.0, 0.0, 0.0]));
        let blue = make_uniforms(Some([0.0, 0.0, 1.0, 0.0]));
        let tint_only = make_uniforms(None);
        let guest_uniform = crate::FastMap::from_iter([((1, 0), (0x1000, 4))]);
        let mut draw = Draw {
            program: &program,
            colors: &[7],
            color_levels: &[0],
            color_layers: &[0],
            color_formats: &[],
            depth: None,
            depth_layer: 0,
            attributes: vec![VertexAttribute {
                location: 0,
                stream: 0,
                offset: 0,
                format: vk::Format::R32G32_SFLOAT,
            }],
            streams: vec![VertexStream {
                stride: 8,
                divisor: 0,
                data: positions
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
                guest: None,
            }],
            indices: Indices::None,
            count: 4,
            first: 0,
            base_vertex: 0,
            instances: 1,
            first_instance: 0,
            uniforms: &black,
            guest_uniforms: &crate::FastMap::default(),
            read_guest: None,
            textures: &crate::FastMap::default(),
            pool: Default::default(),
            storage: crate::FastMap::default(),
            storage_in_place: crate::FastMap::default(),
            fixed: Fixed {
                topology: vk::PrimitiveTopology::TRIANGLE_STRIP,
                primitive_restart: false,
                cull: vk::CullModeFlags::NONE,
                front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                polygon_mode: vk::PolygonMode::FILL,
                depth_test: false,
                depth_write: false,
                depth_compare: vk::CompareOp::ALWAYS,
                viewport: [0.0, 0.0, 4.0, 4.0, 0.0, 1.0],
                scissor: [0, 0, 4, 4],
                stencil: None,
                depth_minus_one_to_one: false,
                depth_bias: None,
                depth_bias_d24: false,
                alpha_to_coverage: false,
                blend_constants: [0.0; 4],
            },
            blends: &blend,
            mesh: None,
            page_table: Default::default(),
            vertex_fetch: Vec::new(),
        };
        renderer.draw(&draw).unwrap();
        renderer.draw(&draw).unwrap();

        draw.fixed.scissor = [0, 0, 1, 4];
        draw.blends = &red_only;
        draw.uniforms = &red;
        renderer.draw(&draw).unwrap();
        let stats = renderer.take_stats();
        if !program.dynamic_uniforms.is_empty() {
            assert_eq!(stats["stat: dynamic graphics descriptor sets reused"], 2);
        }
        if has_pool {
            assert_eq!(stats["stat: graphics texture pool binds"], 1);
        }
        assert_eq!(stats["stat: graphics vertex input sets"], 1);
        let texels = renderer.read_target(7).unwrap();
        // The red-only write reaches the first column, retaining green.
        assert_eq!(texels[0], [1.0, 1.0, 0.0, 1.0]);
        assert_eq!(texels[4 + 1], [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(texels[3], [0.0; 4]);

        // This draw starts a new batch with the same raster and blend state.
        draw.uniforms = &black;
        renderer.draw(&draw).unwrap();
        let stats = renderer.take_stats();
        if !program.dynamic_uniforms.is_empty() {
            assert!(!stats.contains_key("stat: dynamic graphics descriptor sets reused"));
        }
        if has_pool {
            assert_eq!(stats["stat: graphics texture pool binds"], 1);
        }
        assert_eq!(stats["stat: graphics vertex input sets"], 1);
        let texels = renderer.read_target(7).unwrap();
        assert_eq!(texels[0], [0.0, 1.0, 0.0, 1.0]);

        draw.blends = &blend;
        draw.uniforms = &blue;
        renderer.draw(&draw).unwrap();
        let texels = renderer.read_target(7).unwrap();
        assert_eq!(texels[0], [0.0, 1.0, 1.0, 1.0]);

        // Bind only the red word. The blue word from the previous upload
        // must be cleared before the shader reads the full vector.
        draw.uniforms = &tint_only;
        draw.guest_uniforms = &guest_uniform;
        let read_guest = |address, data: &mut [u8]| {
            assert_eq!(address, 0x1000);
            data.copy_from_slice(&1.0f32.to_le_bytes());
            Ok(())
        };
        draw.read_guest = Some(&read_guest);
        renderer.draw(&draw).unwrap();
        let texels = renderer.read_target(7).unwrap();
        assert_eq!(texels[0], [1.0, 1.0, 0.0, 1.0]);

        renderer.take_stats();
        renderer.draw(&draw).unwrap();
        draw.page_table.first_page = 255;
        let stream = &mut draw.streams[0];
        stream.data = stream
            .data
            .chunks_exact(8)
            .flat_map(|position| position.iter().copied().chain([0; 8]))
            .collect();
        stream.stride = 16;
        renderer.draw(&draw).unwrap();
        renderer.draw(&draw).unwrap();
        let stats = renderer.take_stats();
        assert_eq!(stats["stat: graphics vertex input sets"], 2);

        let serial = renderer.recorded_serial();
        renderer.guest_barrier(0x12).unwrap();
        renderer.draw(&draw).unwrap();
        assert_eq!(renderer.recorded_serial(), serial);
        let stats = renderer.take_stats();
        assert!(!stats.contains_key("stat: graphics vertex input sets"));

        let texels = renderer.read_target(7).unwrap();
        assert_eq!(texels[0], [1.0, 1.0, 1.0, 1.0]);

        // An empty draw leaves both the image and the command buffer alone.
        draw.uniforms = &black;
        draw.instances = 0;
        let serial = renderer.recorded_serial();
        assert!(renderer.draw(&draw).unwrap().is_empty());
        assert_eq!(renderer.recorded_serial(), serial);
        draw.indices = Indices::U16(vec![0, 0, 1, 0, 2, 0, 3, 0]);
        assert!(renderer.draw(&draw).unwrap().is_empty());
        assert_eq!(renderer.recorded_serial(), serial);
        draw.instances = 1;
        draw.count = 0;
        assert!(renderer.draw(&draw).unwrap().is_empty());
        assert_eq!(renderer.recorded_serial(), serial);
        assert_eq!(renderer.read_target(7).unwrap(), texels);

        draw.count = 4;
        draw.uniforms = &blue;
        let no_guest_uniforms = crate::FastMap::default();
        draw.guest_uniforms = &no_guest_uniforms;
        let read_indices = |address, bytes: &mut [u8]| {
            let width = match address {
                0x2000 => 2,
                0x3000 => 4,
                _ => panic!("unexpected guest index address {address:#x}"),
            };
            assert_eq!(bytes.len(), 4 * width);
            for (source, destination) in (0u32..4).zip(bytes.chunks_exact_mut(width)) {
                destination.copy_from_slice(&source.to_le_bytes()[..width]);
            }
            Ok(())
        };
        draw.read_guest = Some(&read_indices);
        for (address, bytes, kind) in [
            (0x2000, 8, vk::IndexType::UINT16),
            (0x3000, 16, vk::IndexType::UINT32),
        ] {
            renderer
                .clear_color(7, None, vk::ClearColorValue { float32: [0.0; 4] })
                .unwrap();
            draw.indices = Indices::Guest {
                address,
                bytes,
                kind,
            };
            renderer.draw(&draw).unwrap();
            let texels = renderer.read_target(7).unwrap();
            assert_eq!(texels[0], [0.0, 1.0, 1.0, 1.0]);
            assert_eq!(texels[3], [0.0; 4]);
        }

        renderer.destroy_program(program).unwrap();
    }
}
