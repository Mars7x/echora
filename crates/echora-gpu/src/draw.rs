//! Draws: linked shader objects for a vertex/fragment pair and the dynamic
//! state, buffers and descriptors one draw needs.
//!
//! Everything a draw reads is copied into a host-visible ring at record time,
//! so the guest may change its memory before the work is submitted.

use std::sync::Arc;

use ash::vk;

use crate::renderer::{Renderer, Scale, descriptor_buffer_range, has_stencil};
use crate::shader::{Descriptor, DescriptorKind, SampleType, Translated};
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
    /// Each texture binding's (in descriptor order) last guest texture
    /// upload and its resolved descriptor (`TextureMemo`).
    texture_memo: std::sync::Mutex<Vec<Option<TextureMemo>>>,
    /// The descriptor set last written for this program, while the
    /// submission it was allocated for is recorded: a draw with the same
    /// descriptors binds it again.
    last_set: std::sync::Mutex<Option<DynamicSet>>,
}

/// A texture binding's upload, sampler state and the descriptor they
/// resolved to: the same again resolves to it while the renderer's
/// `texture_epoch` is unchanged (an upload's views and samplers live as
/// long as its image, and an image holds one key's texels while its
/// generation is unchanged).
struct TextureMemo {
    epoch: u64,
    upload: (u64, u64, u32, u32, ImageShape, vk::Format),
    sampler: Sampler,
    info: vk::DescriptorImageInfo,
}

impl TextureMemo {
    fn upload(source: &TextureSource) -> Option<(u64, u64, u32, u32, ImageShape, vk::Format)> {
        match *source {
            TextureSource::Upload {
                key,
                generation,
                width,
                height,
                shape,
                format,
                ..
            } => Some((key, generation, width, height, shape, format)),
            _ => None,
        }
    }
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
    pool_set: Option<vk::DescriptorSet>,
    // Dynamic vertex input remains set across shader object bindings.
    vertex_input: Option<VertexInput>,
    // Per-draw working space, kept to reuse its allocations.
    scratch: DrawScratch,
}

#[derive(Default)]
struct DrawScratch {
    /// The draw's descriptors by binding.
    writes: Vec<(u32, DescriptorSource)>,
    /// Texture and texel buffer descriptors, in the program's order.
    textures: Vec<DescriptorSource>,
    dynamic_offsets: Vec<u32>,
    vertex_buffers: Vec<vk::Buffer>,
    vertex_offsets: Vec<u64>,
}

impl GraphicsState {
    pub(crate) fn reset(&mut self) {
        self.program = None;
        self.raster = None;
        self.blends.clear();
        self.vertex_input = None;
        self.reset_program_bindings();
    }

    fn reset_program_bindings(&mut self) {
        self.pool_set = None;
    }
}

#[derive(Default)]
struct VertexInput {
    attributes: Vec<VertexAttribute>,
    streams: Vec<(u32, u32)>,
}

struct DynamicSet {
    /// The submission (`Renderer::flushes`) whose descriptor pool holds it.
    flush: u64,
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

/// One mip level of one layer of an upload and the byte offset of its
/// texels. Slices of a 3D level follow each other. For combined depth and
/// stencil, the depth plane uses four bytes per texel, followed by a
/// one-byte-per-texel stencil plane.
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
    /// A texture view's mip levels (base, count; zero count means the
    /// remaining levels) of an upload; not part of the Vulkan sampler.
    pub view_levels: Option<(u32, u32)>,
    /// An upload's view of array layers (base, count; zero means the rest).
    pub view_layers: Option<(u32, u32)>,
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
        ) && self.view_layers == other.view_layers
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
            view_layers: None,
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
    /// Unmodified indices the GPU wrote, read where they are (a buffer
    /// over shared guest memory): ordered after the writing work by its
    /// barrier, so the draw does not wait for that work to finish.
    Shared {
        buffer: vk::Buffer,
        offset: u64,
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
    /// The mip level and array layer the depth target renders into.
    pub depth_level: u32,
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
    /// Uniform buffer bindings the GPU reads in place, as (buffer, offset,
    /// length): guest memory that GPU work recorded before the draw writes
    /// (the offset a multiple of `uniform_offset_alignment`).
    pub uniforms_in_place: crate::FastMap<(usize, u32), (vk::Buffer, u64, u64)>,
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

/// Stages for a fullscreen clear. A stencil clear has no color output.
fn clear_stage(stage: crate::shader::Stage, output: Option<SampleType>) -> Arc<Translated> {
    use echora_shader::spirv::{
        Function, Module, Type, capability, decoration, execution, op, storage,
    };

    let mut module = Module::new();
    module.capability(capability::SHADER);
    module.memory_model(0, 1);
    let void = module.void();
    let signature = module.ty(Type::Function {
        result: void,
        parameters: Vec::new(),
    });
    let mut main = Function::new(&mut module, void, signature);
    let mut descriptors = Vec::new();
    if stage == crate::shader::Stage::Vertex {
        let index_type = module.u32();
        let index_pointer = module.pointer(storage::INPUT, index_type);
        let index = module.global_variable(index_pointer, storage::INPUT);
        const VERTEX_INDEX: u32 = 42;
        module.decorate(index, decoration::BUILT_IN, &[VERTEX_INDEX]);
        let float = module.f32();
        let vector = module.ty(Type::Vector {
            component: float,
            count: 4,
        });
        let position_pointer = module.pointer(storage::OUTPUT, vector);
        let position = module.global_variable(position_pointer, storage::OUTPUT);
        const POSITION: u32 = 0;
        module.decorate(position, decoration::BUILT_IN, &[POSITION]);
        module.entry_point(execution::VERTEX, main.id, "main", &[index, position]);
        let index = main.value(&mut module, op::LOAD, index_type, &[index]);
        let bool_type = module.bool();
        let neg = module.constant(float, (-1.0f32).to_bits());
        let edge = module.constant(float, 3.0f32.to_bits());
        let zero = module.constant(float, 0);
        let one = module.constant(float, 1.0f32.to_bits());
        let [x, y] = [1, 2].map(|corner| {
            let corner = module.u32_constant(corner);
            let outer = main.value(&mut module, op::I_EQUAL, bool_type, &[index, corner]);
            main.value(&mut module, op::SELECT, float, &[outer, edge, neg])
        });
        let vertex = main.value(
            &mut module,
            op::COMPOSITE_CONSTRUCT,
            vector,
            &[x, y, zero, one],
        );
        main.emit(op::STORE, &[position, vertex]);
    } else {
        let mut interface = Vec::new();
        if let Some(output) = output {
            let component = match output {
                SampleType::Float => module.f32(),
                SampleType::Signed => module.ty(Type::Int {
                    width: 32,
                    signed: true,
                }),
                SampleType::Unsigned => module.u32(),
            };
            let vector = module.ty(Type::Vector {
                component,
                count: 4,
            });
            let block = module.ty(Type::Struct {
                members: vec![vector],
            });
            module.decorate(block, decoration::BLOCK, &[]);
            module.member_decorate(block, 0, decoration::OFFSET, &[0]);
            let uniform = module.pointer(storage::UNIFORM, block);
            let color = module.global_variable(uniform, storage::UNIFORM);
            module.decorate(color, decoration::DESCRIPTOR_SET, &[0]);
            module.decorate(color, decoration::BINDING, &[0]);
            let output = module.pointer(storage::OUTPUT, vector);
            let attachment = module.global_variable(output, storage::OUTPUT);
            const LOCATION: u32 = 30;
            module.decorate(attachment, LOCATION, &[0]);
            let pointer = module.pointer(storage::UNIFORM, vector);
            let zero = module.u32_constant(0);
            let color_value = main.value(&mut module, op::ACCESS_CHAIN, pointer, &[color, zero]);
            let color_value = main.value(&mut module, op::LOAD, vector, &[color_value]);
            main.emit(op::STORE, &[attachment, color_value]);
            interface.extend([color, attachment]);
            descriptors.push(Descriptor {
                count: 1,
                ..Descriptor::default()
            });
        }
        module.entry_point(execution::FRAGMENT, main.id, "main", &interface);
        module.execution_mode(main.id, execution::ORIGIN_UPPER_LEFT, &[]);
    }
    main.emit(op::RETURN, &[]);
    main.finish(&mut module);
    Arc::new(Translated {
        abi_version: crate::shader::ABI_VERSION,
        stage,
        spirv: module.finish(),
        frag_colors: u32::from(!descriptors.is_empty()),
        descriptors,
        loaded_generics: 0,
        stored_generics: 0,
        stores: [0; 8],
        cbuf_used_sizes: [0; 18],
        spec_constants: Vec::new(),
        push_fields: 0,
        texture_pool_bindings: 0,
    })
}

pub(crate) enum ShaderClear {
    Color {
        format: vk::Format,
        words: [u32; 4],
        mask: vk::ColorComponentFlags,
    },
    Stencil(StencilFace),
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
        let mut subgroup_sizes = vec![
            vk::ShaderRequiredSubgroupSizeCreateInfoEXT::default()
                .required_subgroup_size(32);
            stages.len()
        ];
        let compute_32 = self.gpu.compute_subgroup_32
            && stages.len() == 1
            && stages[0].stage == crate::shader::Stage::Compute;
        let has_task = stages
            .iter()
            .any(|stage| stage.stage == crate::shader::Stage::Task);
        let infos: Vec<_> = stages
            .iter()
            .zip(&mut subgroup_sizes)
            .enumerate()
            .map(|(index, (stage, subgroup))| {
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
                if compute_32
                    && self.gpu.compute_full_subgroups
                    && crate::vulkan::whole_subgroups(&stage.spirv)
                {
                    flags |= vk::ShaderCreateFlagsEXT::REQUIRE_FULL_SUBGROUPS;
                }
                let info = vk::ShaderCreateInfoEXT::default()
                    .flags(flags)
                    .stage(stage_flags(stage.stage))
                    .next_stage(next)
                    .code_type(vk::ShaderCodeTypeEXT::SPIRV)
                    .code(codes[index])
                    .name(c"main")
                    .set_layouts(&layouts)
                    .push_constant_ranges(push_ranges)
                    .specialization_info(&specialization_infos[index]);
                if self.gpu.require_subgroup_32(stage) {
                    info.push_next(subgroup)
                } else {
                    info
                }
            })
            .collect();
        let mut cache_layout = format!(
            "{descriptors:?} {push_ranges:?} {compute_32} {specialization:?} {pool}{}",
            if dynamic_uniforms.is_empty() {
                ""
            } else {
                " dynamic uniforms"
            },
        );
        if pool {
            use std::fmt::Write as _;
            write!(
                cache_layout,
                " texture pool {} {} {}",
                echora_shader_abi::texture_pool::COUNT,
                echora_shader_abi::texture_pool::SAMPLER_DESCRIPTORS,
                echora_shader_abi::TEXTURE_POOL_CAPACITY,
            )
            .expect("writing to a String");
        }
        let mut shaders = crate::shader_cache::create_shaders(
            objects,
            self.shader_cache.as_deref(),
            self.gpu.shader_binary_version,
            &infos,
            &cache_layout,
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
            texture_memo: Default::default(),
            last_set: Default::default(),
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
        self.destroy_idle_programs(programs)
    }

    /// Release programs whose last queued use has already completed.
    pub(crate) fn destroy_idle_programs(
        &self,
        programs: impl IntoIterator<Item = Program>,
    ) -> Result<(), String> {
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

    /// Replace only the requested stencil bits inside a depth attachment.
    pub fn clear_stencil_masked(
        &mut self,
        key: u64,
        layer: u32,
        rect: Option<[i32; 4]>,
        value: u32,
        mask: u8,
    ) -> Result<(), String> {
        self.clear_stencil_masked_level(key, (0, layer), rect, value, mask)
    }

    /// Replace stencil bits at a particular mip and array layer.
    pub fn clear_stencil_masked_level(
        &mut self,
        key: u64,
        (level, layer): (u32, u32),
        rect: Option<[i32; 4]>,
        value: u32,
        mask: u8,
    ) -> Result<(), String> {
        let target = self.target_info(key).ok_or("no such stencil target")?;
        if level >= target.levels {
            return Err(format!(
                "mip level {level} of a {}-level stencil target",
                target.levels
            ));
        }
        if mask == 0 || !has_stencil(target.format) {
            return Ok(());
        }
        let width = (target.scale.apply(target.width) >> level).max(1);
        let height = (target.scale.apply(target.height) >> level).max(1);
        let whole = [0, 0, width as i32, height as i32];
        let rect = rect.map(|rect| target.scale.rect(rect));
        let scissor = crate::renderer::clip(rect.unwrap_or(whole), whole);
        if scissor[2] <= 0 || scissor[3] <= 0 {
            return Ok(());
        }
        let face = StencilFace {
            fail: vk::StencilOp::KEEP,
            pass: vk::StencilOp::REPLACE,
            depth_fail: vk::StencilOp::KEEP,
            compare: vk::CompareOp::ALWAYS,
            compare_mask: u32::MAX,
            write_mask: u32::from(mask),
            reference: value & 0xff,
        };
        self.stat("stat: masked stencil clears", 1);
        self.clear_with_shader(
            (key, level, layer),
            (width, height),
            scissor,
            ShaderClear::Stencil(face),
        )
    }

    pub(crate) fn clear_with_shader(
        &mut self,
        (key, level, layer): (u64, u32, u32),
        (width, height): (u32, u32),
        scissor: [i32; 4],
        clear: ShaderClear,
    ) -> Result<(), String> {
        let mut uniforms = crate::FastMap::default();
        let (format, stencil, blend) = match clear {
            ShaderClear::Color {
                format,
                words,
                mask,
            } => {
                uniforms.insert(
                    (1, 0),
                    words.iter().flat_map(|word| word.to_le_bytes()).collect(),
                );
                (
                    Some(format),
                    None,
                    Some(Blend {
                        write_mask: mask,
                        ..Default::default()
                    }),
                )
            }
            ShaderClear::Stencil(face) => (None, Some([face; 2]), None),
        };
        let output = format.map(crate::renderer::sampled_texture_type);
        let slot = match output {
            None => 0,
            Some(SampleType::Float) => 1,
            Some(SampleType::Signed) => 2,
            Some(SampleType::Unsigned) => 3,
        };
        let program = match self.shader_clear_programs[slot].take() {
            Some(program) => program,
            None => self.create_program(vec![
                clear_stage(crate::shader::Stage::Vertex, None),
                clear_stage(crate::shader::Stage::Fragment, output),
            ])?,
        };
        let color_target = format.map(|_| key);
        let color_level = format.map(|_| level);
        let color_layer = format.map(|_| layer);
        let color_format = format.map(Some);
        let draw = Draw {
            program: &program,
            colors: color_target.as_slice(),
            color_levels: color_level.as_slice(),
            color_layers: color_layer.as_slice(),
            color_formats: color_format.as_slice(),
            depth: stencil.map(|_| key),
            depth_level: level,
            depth_layer: layer,
            attributes: Vec::new(),
            streams: Vec::new(),
            indices: Indices::None,
            count: 3,
            first: 0,
            base_vertex: 0,
            instances: 1,
            first_instance: 0,
            uniforms: &uniforms,
            guest_uniforms: &crate::FastMap::default(),
            read_guest: None,
            textures: &crate::FastMap::default(),
            storage: crate::FastMap::default(),
            storage_in_place: crate::FastMap::default(),
            uniforms_in_place: crate::FastMap::default(),
            fixed: Fixed {
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                primitive_restart: false,
                cull: vk::CullModeFlags::NONE,
                front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                polygon_mode: vk::PolygonMode::FILL,
                depth_test: false,
                depth_write: false,
                depth_compare: vk::CompareOp::ALWAYS,
                viewport: [0.0, 0.0, width as f32, height as f32, 0.0, 1.0],
                scissor,
                stencil,
                depth_minus_one_to_one: false,
                depth_bias: None,
                depth_bias_d24: false,
                alpha_to_coverage: false,
                blend_constants: [0.0; 4],
            },
            blends: blend.as_slice(),
            mesh: None,
            page_table: Default::default(),
            vertex_fetch: Vec::new(),
            pool: Default::default(),
        };
        let result = self.without_occlusion_count(|renderer| {
            renderer.draw_with_raster_scale(&draw, Some(Scale::ONE))?;
            Ok(())
        });
        self.shader_clear_programs[slot] = Some(program);
        result
    }

    /// Record one draw.
    /// Record `draw`. Returns, for each copied storage buffer a stage
    /// writes, its (stage index, binding) and the token of its contents
    /// after the next flush (`take_reads`).
    pub fn draw(&mut self, draw: &Draw) -> Result<Vec<((usize, u32), u64)>, String> {
        self.draw_with_raster_scale(draw, None)
    }

    fn draw_with_raster_scale(
        &mut self,
        draw: &Draw,
        raster_scale: Option<Scale>,
    ) -> Result<Vec<((usize, u32), u64)>, String> {
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
        // Each texture's view and sampler are resolved here, once, and
        // written below.
        let mut scratch = std::mem::take(&mut self.graphics.scratch);
        scratch.textures.clear();
        let mut bytes = 0;
        let mut descriptors = [0u32; 4];
        for (stage_index, stage) in program.stages.iter().enumerate() {
            for descriptor in &stage.descriptors {
                let key = (stage_index, descriptor.binding);
                match descriptor.kind() {
                    DescriptorKind::Texture => {
                        let Some((source, sampler)) = draw.textures.get(&key) else {
                            return Err(format!(
                                "no texture for stage {stage_index} binding {}",
                                descriptor.binding
                            ));
                        };
                        let slot = scratch.textures.len();
                        let upload = TextureMemo::upload(source);
                        let memoized = upload.and_then(|upload| {
                            let memo = program.texture_memo.lock().unwrap();
                            memo.get(slot)?
                                .as_ref()
                                .filter(|memo| {
                                    memo.epoch == self.texture_epoch
                                        && memo.upload == upload
                                        && memo.sampler == *sampler
                                })
                                .map(|memo| memo.info)
                        });
                        let info = match memoized {
                            Some(info) => info,
                            None => {
                                let view = self.texture_view_swizzled(
                                    source,
                                    descriptor.texture_type,
                                    sampler,
                                )?;
                                let info = vk::DescriptorImageInfo {
                                    sampler: self.sampler(
                                        sampler,
                                        descriptor.flags & Descriptor::INTEGER != 0,
                                    )?,
                                    image_view: view,
                                    image_layout: vk::ImageLayout::GENERAL,
                                };
                                if let Some(upload) = upload {
                                    let mut memo = program.texture_memo.lock().unwrap();
                                    if memo.len() <= slot {
                                        memo.resize_with(slot + 1, || None);
                                    }
                                    memo[slot] = Some(TextureMemo {
                                        epoch: self.texture_epoch,
                                        upload,
                                        sampler: *sampler,
                                        info,
                                    });
                                }
                                info
                            }
                        };
                        scratch.textures.push(DescriptorSource::Image(info));
                        descriptors[1] += 1;
                    }
                    DescriptorKind::TextureBuffer => {
                        let Some((source, _)) = draw.textures.get(&key) else {
                            return Err(format!(
                                "no texel buffer for stage {stage_index} binding {}",
                                descriptor.binding
                            ));
                        };
                        scratch.textures.push(DescriptorSource::TexelBuffer(
                            self.texel_buffer_view(source)?,
                        ));
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
                        if !draw.uniforms_in_place.contains_key(&key) {
                            let length = match draw.uniforms.get(&key) {
                                Some(bytes) => bytes.len(),
                                None => draw.guest_uniforms.get(&key).map_or(16, |read| read.1),
                            };
                            bytes += descriptor_buffer_range(length) + 256;
                        }
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
            Indices::Shared { .. } => 0,
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
        let mut writes = std::mem::take(&mut scratch.writes);
        writes.clear();
        let mut written = Vec::new();
        let mut dynamic_offsets = std::mem::take(&mut scratch.dynamic_offsets);
        dynamic_offsets.clear();
        dynamic_offsets.resize(program.dynamic_uniforms.len(), 0);
        let mut textures = scratch.textures.drain(..);
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
                            _ if let Some(&(buffer, offset, length)) =
                                draw.uniforms_in_place.get(&key) =>
                            {
                                (buffer, offset, length.max(16).next_multiple_of(16))
                            }
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
                    DescriptorKind::Texture | DescriptorKind::TextureBuffer => textures
                        .next()
                        .expect("each texture was resolved before the uploads"),
                    other => return Err(format!("{other:?} bindings are not modeled")),
                };
                writes.push((descriptor.binding, source));
            }
        }
        drop(textures);
        self.lap("stat: renderer ns uniforms storage textures", &mut since);
        let matching = program
            .last_set
            .lock()
            .unwrap()
            .as_ref()
            .filter(|previous| {
                previous.flush == self.flushes
                    && previous.bindings.len() == writes.len()
                    && previous.bindings.iter().zip(&writes).all(
                        |((old_binding, old), (binding, source))| {
                            binding == old_binding && source.same_as(old)
                        },
                    )
            })
            .map(|previous| previous.set);
        let set = if let Some(set) = matching {
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
            drop(updates);
            // The replaced set's list is the next draw's to fill.
            let replaced = program.last_set.lock().unwrap().replace(DynamicSet {
                flush: self.flushes,
                set,
                bindings: std::mem::take(&mut writes),
            });
            if let Some(old) = replaced {
                writes = old.bindings;
            }
            set
        };
        self.lap("stat: renderer ns descriptor set", &mut since);
        // Vertex streams and indices.
        scratch.vertex_buffers.clear();
        scratch.vertex_offsets.clear();
        for stream in &draw.streams {
            let (buffer, offset) = match (stream.guest, draw.read_guest) {
                (Some((guest, length)), Some(read)) if length > 0 => {
                    self.ring_upload_with(length as u64, 16, |bytes| read(guest, bytes))?
                }
                _ => self.ring_upload(&stream.data, 16)?,
            };
            scratch.vertex_buffers.push(buffer);
            scratch.vertex_offsets.push(offset);
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
            &Indices::Shared {
                buffer,
                offset,
                kind,
            } => Some(((buffer, offset), kind)),
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
            draw.depth
                .map(|key| (key, draw.depth_level, draw.depth_layer)),
            area,
        )?;
        self.apply_condition()?;
        let commands = self.recording_commands()?;
        // The device's function tables by reference (cloning them copies
        // hundreds of pointers per draw).
        let gpu = self.gpu.clone();
        let objects = gpu.shader_objects()?;
        let device = &gpu.device;
        let fixed = &draw.fixed;
        if draw.mesh.is_some() && self.gpu.mesh_shader.is_none() {
            return Err("the device has no task and mesh stages".into());
        }
        let [x, y, width, height, near, far] = fixed.viewport;
        // A negative height (Vulkan 1.1) flips Y; the caller decides.
        // Guest draws scale their viewport and scissor to image texels.
        // Attachment clear draws specify them directly in image texels.
        let factor = scale.factor();
        let raster_factor = raster_scale.map_or(factor, Scale::factor);
        let raster_scale = raster_scale.unwrap_or(scale);
        let viewport = vk::Viewport {
            x: x * raster_factor,
            y: y * raster_factor,
            width: width * raster_factor,
            height: height * raster_factor,
            min_depth: near,
            max_depth: far,
        };
        let area = raster_scale.rect(area.map(|value| value.max(0)));
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
        if abi.is_some() {
            self.record_page_table_read(draw.page_table.address);
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
                self.set_vertex_input(commands, objects, draw);
            }
            if !scratch.vertex_buffers.is_empty() {
                device.cmd_bind_vertex_buffers(
                    commands,
                    0,
                    &scratch.vertex_buffers,
                    &scratch.vertex_offsets,
                );
            }
            if self.graphics.raster.is_none() {
                objects.cmd_set_rasterizer_discard_enable(commands, false);
                objects.cmd_set_depth_clamp_enable(commands, false);
                objects.cmd_set_depth_bounds_test_enable(commands, false);
                objects.cmd_set_rasterization_samples(commands, vk::SampleCountFlags::TYPE_1);
                objects.cmd_set_sample_mask(commands, vk::SampleCountFlags::TYPE_1, &[u32::MAX]);
                // Shader objects need the line width set for any draw that
                // rasterizes lines (wideLines is not enabled: always 1).
                device.cmd_set_line_width(commands, 1.0);
            }
            if self.graphics.raster.as_ref() != Some(&raster) {
                // Dynamic state remains set for this command buffer. Emit
                // only the groups that changed.
                let previous = self.graphics.raster.as_ref();
                macro_rules! changed {
                    ($($field:ident),+) => {
                        previous.is_none_or(|last| $(last.$field != raster.$field)||+)
                    };
                }
                if changed!(viewport, scissor) {
                    objects.cmd_set_viewport_with_count(commands, &[viewport]);
                    objects.cmd_set_scissor_with_count(commands, &[scissor]);
                }
                if changed!(topology, primitive_restart) {
                    objects.cmd_set_primitive_topology(commands, raster.topology);
                    objects.cmd_set_primitive_restart_enable(commands, raster.primitive_restart);
                }
                if changed!(cull, front_face, polygon_mode) {
                    objects.cmd_set_cull_mode(commands, raster.cull);
                    objects.cmd_set_front_face(commands, raster.front_face);
                    let mode = if self.gpu.fill_mode_non_solid {
                        raster.polygon_mode
                    } else {
                        vk::PolygonMode::FILL
                    };
                    objects.cmd_set_polygon_mode(commands, mode);
                }
                if self.gpu.depth_clip_control && changed!(depth_minus_one_to_one) {
                    objects.cmd_set_depth_clip_negative_one_to_one(
                        commands,
                        raster.depth_minus_one_to_one,
                    );
                }
                if changed!(alpha_to_coverage) {
                    objects.cmd_set_alpha_to_coverage_enable(commands, raster.alpha_to_coverage);
                }
                if changed!(blend_constants) {
                    device.cmd_set_blend_constants(commands, &fixed.blend_constants);
                }
                if changed!(depth_bias) {
                    objects.cmd_set_depth_bias_enable(commands, raster.depth_bias.is_some());
                }
                if changed!(depth_bias)
                    && let Some((bias, force_unorm)) = raster.depth_bias
                {
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
                if changed!(depth_test, depth_write, depth_compare) {
                    objects.cmd_set_depth_test_enable(commands, raster.depth_test);
                    objects.cmd_set_depth_write_enable(commands, raster.depth_write);
                    objects.cmd_set_depth_compare_op(commands, raster.depth_compare);
                }
                if changed!(stencil) {
                    objects.cmd_set_stencil_test_enable(commands, raster.stencil.is_some());
                }
                if changed!(stencil)
                    && let Some(faces) = raster.stencil
                {
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
        scratch.writes = writes;
        scratch.dynamic_offsets = dynamic_offsets;
        self.graphics.scratch = scratch;
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
            depth_level: 0,
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
            uniforms_in_place: crate::FastMap::default(),
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

        let alternate = renderer.create_program(program.stages.clone()).unwrap();
        draw.program = &alternate;
        renderer.draw(&draw).unwrap();
        let stats = renderer.take_stats();
        assert!(!stats.contains_key("stat: graphics vertex input sets"));
        draw.program = &program;

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

        renderer.destroy_program(alternate).unwrap();
        renderer.destroy_program(program).unwrap();
    }
}
