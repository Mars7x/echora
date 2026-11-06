//! Draws: linked shader objects for a vertex/fragment pair and the dynamic
//! state, buffers and descriptors one draw needs.
//!
//! Everything a draw reads is copied into a host-visible ring at record time,
//! so the guest may change its memory before the work is submitted.

use std::collections::HashMap;
use std::sync::Arc;

use ash::vk;

use crate::renderer::{Renderer, has_stencil};
use crate::shader::{Descriptor, DescriptorKind, Translated};
use crate::vulkan::stage_flags;

/// A vertex/fragment pair, translated so its bindings share set 0.
pub struct Program {
    pub stages: Vec<Arc<Translated>>,
    pub(crate) layout: vk::DescriptorSetLayout,
    pub(crate) pipeline_layout: vk::PipelineLayout,
    pub(crate) shaders: Vec<vk::ShaderEXT>,
    /// With resolution scaling: the stages given the rescaling push
    /// constants, and each texture binding's (stage index, binding) by its
    /// scaling bit (the recompiler numbers texture descriptors in binding
    /// order across the stages).
    pub(crate) push_stages: vk::ShaderStageFlags,
    pub(crate) texture_bits: Vec<(usize, u32)>,
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

#[derive(Clone, Copy, Debug)]
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
}

#[derive(Clone, Copy, Debug)]
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
        levels: Vec<UploadRegion>,
    },
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
    /// Channel sources (a texture view's swizzle); not part of the
    /// Vulkan sampler.
    pub components: vk::ComponentMapping,
    /// A texture view's format, when it differs from the texture's (its
    /// sRGB twin); not part of the Vulkan sampler.
    pub format: Option<vk::Format>,
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
            components: vk::ComponentMapping::default(),
            format: None,
        }
    }
}

pub enum Indices {
    None,
    U16(Vec<u8>),
    U32(Vec<u8>),
}

pub struct Draw<'a> {
    pub program: &'a Program,
    pub colors: Vec<u64>,
    /// The mip level each color target renders into.
    pub color_levels: Vec<u32>,
    /// The array layer (or cube face) each color target renders into.
    pub color_layers: Vec<u32>,
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
    pub uniforms: HashMap<(usize, u32), Vec<u8>>,
    /// Texture bindings by (stage index, binding).
    pub textures: HashMap<(usize, u32), (TextureSource, Sampler)>,
    /// Storage buffer contents by (stage index, binding). Graphics stages
    /// read them; writes are not returned.
    pub storage: HashMap<(usize, u32), Vec<u8>>,
    pub fixed: Fixed,
    pub blends: Vec<Blend>,
}

fn vk_error(context: &'static str) -> impl Fn(vk::Result) -> String {
    move |error| format!("{context}: {error}")
}

impl Renderer {
    /// Link translated stages (vertex first) into shader objects sharing one
    /// set-0 layout. Each stage must have been translated with its binding
    /// base after the previous stage's bindings.
    pub fn create_program(&self, stages: Vec<Arc<Translated>>) -> Result<Program, String> {
        let objects = self.gpu.shader_objects()?;
        let mut descriptors: Vec<(Descriptor, vk::ShaderStageFlags)> = Vec::new();
        for stage in &stages {
            for descriptor in &stage.descriptors {
                descriptors.push((*descriptor, stage_flags(stage.stage)));
            }
        }
        let layout = self.gpu.descriptor_set_layout_for(&descriptors)?;
        let layouts = [layout];
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
        let push_ranges = [vk::PushConstantRange {
            stage_flags: push_stages,
            offset: 0,
            size: RESCALING_BYTES,
        }];
        let push_ranges: &[vk::PushConstantRange] = if push_stages.is_empty() {
            &[]
        } else {
            &push_ranges
        };
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
        // Compute stages run in 32-lane subgroups, as guest warps, where the
        // device can require it.
        let mut subgroup_32 =
            vk::ShaderRequiredSubgroupSizeCreateInfoEXT::default().required_subgroup_size(32);
        let compute_32 = self.gpu.compute_subgroup_32
            && stages.len() == 1
            && stages[0].stage == crate::shader::Stage::Compute;
        let mut infos: Vec<_> = stages
            .iter()
            .enumerate()
            .map(|(index, stage)| {
                let next = stages
                    .get(index + 1)
                    .map_or(vk::ShaderStageFlags::empty(), |next| {
                        stage_flags(next.stage)
                    });
                vk::ShaderCreateInfoEXT::default()
                    .flags(if stages.len() > 1 {
                        vk::ShaderCreateFlagsEXT::LINK_STAGE
                    } else {
                        vk::ShaderCreateFlagsEXT::empty()
                    })
                    .stage(stage_flags(stage.stage))
                    .next_stage(next)
                    .code_type(vk::ShaderCodeTypeEXT::SPIRV)
                    .code(codes[index])
                    .name(c"main")
                    .set_layouts(&layouts)
                    .push_constant_ranges(push_ranges)
            })
            .collect();
        if compute_32 {
            infos[0] = infos[0].push_next(&mut subgroup_32);
        }
        // SAFETY: the code and layouts outlive the call.
        let shaders = unsafe { objects.create_shaders(&infos, None) }
            .map_err(|(_, error)| format!("vkCreateShadersEXT: {error}"))?;
        Ok(Program {
            stages,
            layout,
            pipeline_layout,
            shaders,
            push_stages,
            texture_bits,
        })
    }

    pub fn destroy_program(&mut self, program: Program) -> Result<(), String> {
        self.flush()?;
        let objects = self.gpu.shader_objects()?;
        // SAFETY: flushed, so nothing uses the program.
        unsafe {
            for shader in program.shaders {
                objects.destroy_shader(shader, None);
            }
            self.gpu
                .device
                .destroy_pipeline_layout(program.pipeline_layout, None);
            self.gpu
                .device
                .destroy_descriptor_set_layout(program.layout, None);
        }
        Ok(())
    }

    /// Record one draw.
    pub fn draw(&mut self, draw: &Draw) -> Result<(), String> {
        // Attachments render at one scale (before anything is recorded:
        // keeping a target unscaled flushes).
        let scale = self.attachment_scale(&draw.colors, draw.depth)?;
        self.bound_submission()?;
        let program = draw.program;
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
                        let length = draw.storage.get(&key).map_or(16, Vec::len) as u64;
                        bytes += length.max(16) + 256;
                        descriptors[2] += 1;
                    }
                    _ => {
                        let length = draw.uniforms.get(&key).map_or(16, Vec::len) as u64;
                        bytes += length.max(16) + 256;
                        descriptors[0] += 1;
                    }
                }
            }
        }
        bytes += draw
            .streams
            .iter()
            .map(|stream| stream.data.len() as u64 + 16)
            .sum::<u64>();
        bytes += match &draw.indices {
            Indices::None => 0,
            Indices::U16(data) | Indices::U32(data) => data.len() as u64 + 16,
        };
        self.reserve(bytes, descriptors)?;
        // Uploads and descriptor writes first: they record copies that must
        // precede rendering.
        let mut writes: Vec<(
            u32,
            vk::DescriptorType,
            Option<vk::DescriptorBufferInfo>,
            Option<vk::DescriptorImageInfo>,
        )> = Vec::new();
        let mut texel_views: HashMap<u32, vk::BufferView> = HashMap::new();
        for (stage_index, stage) in program.stages.iter().enumerate() {
            for descriptor in &stage.descriptors {
                let key = (stage_index, descriptor.binding);
                match descriptor.kind() {
                    DescriptorKind::UniformBuffer => {
                        let bytes = draw
                            .uniforms
                            .get(&key)
                            .map(Vec::as_slice)
                            .unwrap_or(&[0; 16]);
                        let (buffer, offset) = self.ring_upload(bytes, 256)?;
                        let range = (bytes.len() as u64).max(16).next_multiple_of(16);
                        writes.push((
                            descriptor.binding,
                            vk::DescriptorType::UNIFORM_BUFFER,
                            Some(vk::DescriptorBufferInfo {
                                buffer,
                                offset,
                                range,
                            }),
                            None,
                        ));
                    }
                    DescriptorKind::StorageBuffer => {
                        let bytes = draw
                            .storage
                            .get(&key)
                            .map(Vec::as_slice)
                            .unwrap_or(&[0; 16]);
                        let (buffer, offset) = self.ring_upload(bytes, 256)?;
                        let range = (bytes.len() as u64).max(16).next_multiple_of(16);
                        writes.push((
                            descriptor.binding,
                            vk::DescriptorType::STORAGE_BUFFER,
                            Some(vk::DescriptorBufferInfo {
                                buffer,
                                offset,
                                range,
                            }),
                            None,
                        ));
                    }
                    DescriptorKind::Texture => {
                        let Some((source, sampler)) = draw.textures.get(&key) else {
                            return Err(format!(
                                "no texture for stage {stage_index} binding {}",
                                descriptor.binding
                            ));
                        };
                        let view = self.texture_view_swizzled(
                            source,
                            descriptor.texture_type,
                            sampler.components,
                            sampler.format,
                        )?;
                        let sampler = self.sampler(sampler)?;
                        writes.push((
                            descriptor.binding,
                            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            None,
                            Some(vk::DescriptorImageInfo {
                                sampler,
                                image_view: view,
                                image_layout: vk::ImageLayout::GENERAL,
                            }),
                        ));
                    }
                    DescriptorKind::TextureBuffer => {
                        let Some((source, _)) = draw.textures.get(&key) else {
                            return Err(format!(
                                "no texel buffer for stage {stage_index} binding {}",
                                descriptor.binding
                            ));
                        };
                        texel_views.insert(descriptor.binding, self.texel_buffer_view(source)?);
                        writes.push((
                            descriptor.binding,
                            vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
                            None,
                            None,
                        ));
                    }
                    other => return Err(format!("{other:?} bindings are not modeled")),
                }
            }
        }
        let set = self.descriptor_set(program.layout)?;
        let texel_infos: Vec<[vk::BufferView; 1]> = writes
            .iter()
            .map(|write| [texel_views.get(&write.0).copied().unwrap_or_default()])
            .collect();
        let buffer_infos: Vec<[vk::DescriptorBufferInfo; 1]> = writes
            .iter()
            .map(|write| [write.2.unwrap_or_default()])
            .collect();
        let image_infos: Vec<[vk::DescriptorImageInfo; 1]> = writes
            .iter()
            .map(|write| [write.3.unwrap_or_default()])
            .collect();
        let descriptor_writes: Vec<_> = writes
            .iter()
            .enumerate()
            .map(|(index, (binding, kind, buffer, _))| {
                let write = vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(*binding)
                    .descriptor_type(*kind);
                if *kind == vk::DescriptorType::UNIFORM_TEXEL_BUFFER {
                    write.texel_buffer_view(&texel_infos[index])
                } else if buffer.is_some() {
                    write.buffer_info(&buffer_infos[index])
                } else {
                    write.image_info(&image_infos[index])
                }
            })
            .collect();
        // SAFETY: the set and resources are live.
        unsafe {
            self.gpu
                .device
                .update_descriptor_sets(&descriptor_writes, &[])
        };

        // Vertex streams and indices.
        let mut vertex_buffers = Vec::new();
        for stream in &draw.streams {
            vertex_buffers.push(self.ring_upload(&stream.data, 16)?);
        }
        let index = match &draw.indices {
            Indices::None => None,
            Indices::U16(bytes) => Some((self.ring_upload(bytes, 16)?, vk::IndexType::UINT16)),
            Indices::U32(bytes) => Some((self.ring_upload(bytes, 16)?, vk::IndexType::UINT32)),
        };

        // Targets this draw samples were maybe rendered just before.
        let sampled: Vec<vk::Image> = draw
            .textures
            .values()
            .filter_map(|(source, _)| match source {
                TextureSource::Target(key) | TextureSource::TargetView { key, .. } => {
                    self.existing_target(*key).map(|target| target.image)
                }
                TextureSource::Upload { .. } => None,
            })
            .collect();
        self.order_after_rendering(&sampled)?;
        let area = draw.fixed.scissor;
        self.begin_rendering(
            &draw.colors,
            &draw.color_levels,
            &draw.color_layers,
            draw.depth,
            draw.depth_layer,
        )?;
        let commands = self.recording_commands()?;
        let objects = self.gpu.shader_objects()?.clone();
        let device = self.gpu.device.clone();
        let fixed = &draw.fixed;
        let stage_bits: Vec<vk::ShaderStageFlags> = program
            .stages
            .iter()
            .map(|stage| stage_flags(stage.stage))
            .collect();
        // Unused graphics stages are bound to nothing.
        let unused = [
            vk::ShaderStageFlags::TESSELLATION_CONTROL,
            vk::ShaderStageFlags::TESSELLATION_EVALUATION,
            vk::ShaderStageFlags::GEOMETRY,
        ];
        let null_shaders = [vk::ShaderEXT::null(); 3];
        let bindings: Vec<_> = vertex_buffers
            .iter()
            .enumerate()
            .map(|(index, _)| {
                let stream = &draw.streams[index];
                vk::VertexInputBindingDescription2EXT::default()
                    .binding(index as u32)
                    .stride(stream.stride)
                    .input_rate(if stream.divisor == 0 {
                        vk::VertexInputRate::VERTEX
                    } else {
                        vk::VertexInputRate::INSTANCE
                    })
                    .divisor(stream.divisor.max(1))
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
        let count = draw.colors.len().max(1);
        let blends: Vec<Blend> = (0..count)
            .map(|index| draw.blends.get(index).copied().unwrap_or_default())
            .collect();
        let enables: Vec<u32> = blends.iter().map(|blend| u32::from(blend.enable)).collect();
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
        // SAFETY: recording inside rendering; every handle is live.
        unsafe {
            objects.cmd_bind_shaders(commands, &stage_bits, &program.shaders);
            objects.cmd_bind_shaders(commands, &unused, &null_shaders);
            device.cmd_bind_descriptor_sets(
                commands,
                vk::PipelineBindPoint::GRAPHICS,
                program.pipeline_layout,
                0,
                &[set],
                &[],
            );
            if let Some(constants) = &rescaling {
                device.cmd_push_constants(
                    commands,
                    program.pipeline_layout,
                    program.push_stages,
                    0,
                    constants,
                );
            }
            objects.cmd_set_vertex_input(commands, &bindings, &attributes);
            for (index, (buffer, offset)) in vertex_buffers.iter().enumerate() {
                device.cmd_bind_vertex_buffers(commands, index as u32, &[*buffer], &[*offset]);
            }
            objects.cmd_set_viewport_with_count(commands, &[viewport]);
            objects.cmd_set_scissor_with_count(commands, &[scissor]);
            objects.cmd_set_rasterizer_discard_enable(commands, false);
            // Required with shader objects whenever depthClamp is enabled.
            objects.cmd_set_depth_clamp_enable(commands, false);
            objects.cmd_set_primitive_topology(commands, fixed.topology);
            objects.cmd_set_primitive_restart_enable(commands, fixed.primitive_restart);
            objects.cmd_set_cull_mode(commands, fixed.cull);
            objects.cmd_set_front_face(commands, fixed.front_face);
            if self.gpu.depth_clip_control {
                objects
                    .cmd_set_depth_clip_negative_one_to_one(commands, fixed.depth_minus_one_to_one);
            }
            objects.cmd_set_polygon_mode(commands, fixed.polygon_mode);
            objects.cmd_set_rasterization_samples(commands, vk::SampleCountFlags::TYPE_1);
            objects.cmd_set_sample_mask(commands, vk::SampleCountFlags::TYPE_1, &[u32::MAX]);
            objects.cmd_set_alpha_to_coverage_enable(commands, false);
            objects.cmd_set_depth_bias_enable(commands, fixed.depth_bias.is_some());
            if let Some([constant, clamp, slope]) = fixed.depth_bias {
                let clamp = if self.gpu.depth_bias_clamp {
                    clamp
                } else {
                    0.0
                };
                self.gpu
                    .device
                    .cmd_set_depth_bias(commands, constant, clamp, slope);
            }
            objects.cmd_set_depth_test_enable(commands, fixed.depth_test && draw.depth.is_some());
            objects.cmd_set_depth_write_enable(commands, fixed.depth_write && draw.depth.is_some());
            objects.cmd_set_depth_compare_op(commands, fixed.depth_compare);
            objects.cmd_set_depth_bounds_test_enable(commands, false);
            let stencil = fixed.stencil.filter(|_| {
                draw.depth
                    .and_then(|key| self.existing_target(key))
                    .is_some_and(|target| has_stencil(target.format))
            });
            objects.cmd_set_stencil_test_enable(commands, stencil.is_some());
            if let Some(faces) = stencil {
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
            objects.cmd_set_color_blend_enable(commands, 0, &enables);
            objects.cmd_set_color_blend_equation(commands, 0, &equations);
            objects.cmd_set_color_write_mask(commands, 0, &masks);
            if let Some(((buffer, offset), kind)) = index {
                device.cmd_bind_index_buffer(commands, buffer, offset, kind);
                device.cmd_draw_indexed(
                    commands,
                    draw.count,
                    draw.instances.max(1),
                    draw.first,
                    draw.base_vertex,
                    draw.first_instance,
                );
            } else {
                device.cmd_draw(
                    commands,
                    draw.count,
                    draw.instances.max(1),
                    draw.first,
                    draw.first_instance,
                );
            }
            // The pass stays open for the next draw into the same targets.
        }
        Ok(())
    }
}

pub(crate) fn words_as_bytes(words: &[u32]) -> &[u8] {
    // SAFETY: u32 has no padding and u8 has alignment 1.
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast(), words.len() * 4) }
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
            naga::valid::Capabilities::empty(),
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
        })
    }

    #[test]
    fn a_triangle_draws_with_a_uniform_color() {
        let Ok(gpu) = Gpu::new() else {
            return; // No device (CI).
        };
        let mut renderer = Renderer::new(gpu).unwrap();
        let vertex = spirv(
            "@vertex fn main(@location(0) position: vec2<f32>) -> @builtin(position) vec4<f32> {
                return vec4<f32>(position, 0.0, 1.0);
            }",
            naga::ShaderStage::Vertex,
        );
        let fragment = spirv(
            "@group(0) @binding(0) var<uniform> color: vec4<f32>;
            @fragment fn main() -> @location(0) vec4<f32> { return color; }",
            naga::ShaderStage::Fragment,
        );
        let uniform = Descriptor {
            kind: 0,
            binding: 0,
            count: 1,
            index: 3,
            ..Default::default()
        };
        let program = renderer
            .create_program(vec![
                translated(Stage::Vertex, vertex, vec![]),
                translated(Stage::Fragment, fragment, vec![uniform]),
            ])
            .unwrap();
        renderer
            .target(7, 4, 4, vk::Format::R8G8B8A8_UNORM)
            .unwrap();
        renderer
            .clear_color(7, None, vk::ClearColorValue { float32: [0.0; 4] })
            .unwrap();
        // A triangle covering the left half: (-1,-1), (0,-1)... in NDC.
        let positions: [f32; 8] = [-1.0, -1.0, 0.0, -1.0, -1.0, 1.0, 0.0, 1.0];
        let color: [f32; 4] = [0.0, 1.0, 0.0, 1.0];
        let draw = Draw {
            program: &program,
            colors: vec![7],
            color_levels: vec![0],
            color_layers: vec![0],
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
            }],
            indices: Indices::None,
            count: 4,
            first: 0,
            base_vertex: 0,
            instances: 1,
            first_instance: 0,
            uniforms: HashMap::from([(
                (1, 0),
                color.iter().flat_map(|value| value.to_le_bytes()).collect(),
            )]),
            textures: HashMap::new(),
            storage: HashMap::new(),
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
            },
            blends: vec![Blend::default()],
        };
        renderer.draw(&draw).unwrap();
        let texels = renderer.read_target(7).unwrap();
        // Left half green, right half still clear.
        assert_eq!(texels[0], [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(texels[4 + 1], [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(texels[3], [0.0; 4]);
        renderer.destroy_program(program).unwrap();
    }
}
