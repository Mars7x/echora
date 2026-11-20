//! Compute dispatches: one translated compute stage, its uniform buffers and
//! its storage buffers. Storage buffers travel through the host-visible ring
//! and are read back after the dispatch, so the caller can store them into
//! guest memory.

use std::collections::HashMap;

use ash::vk;

use crate::draw::{Program, Sampler, TextureSource};
use crate::renderer::{Renderer, descriptor_buffer_range};
use crate::shader::DescriptorKind;

pub struct Dispatch<'a> {
    /// A program of one compute stage (see `Renderer::create_program`).
    pub program: &'a Program,
    pub groups: [u32; 3],
    /// Uniform buffer bytes by binding.
    pub uniforms: HashMap<u32, Vec<u8>>,
    /// Storage buffer contents by binding, before the dispatch.
    pub storage: HashMap<u32, Vec<u8>>,
    /// Storage buffers bound in place instead (imported host memory, see
    /// `Renderer::import_host`): buffer, offset and range by binding. The
    /// shader's writes land in that memory; nothing is uploaded or read back.
    pub storage_in_place: HashMap<u32, (vk::Buffer, u64, u64)>,
    /// Sampled textures by binding.
    pub textures: HashMap<u32, (TextureSource, Sampler)>,
    /// Storage images by binding: the render target each binding writes.
    pub images: HashMap<u32, (u64, u32)>,
    /// The format of the NVN texture view an image binding writes through,
    /// when it is the target's unsigned word alias (`uint_alias`) (a program packing
    /// RGB9E5 or R11G11B10 itself stores words through an R32UI view).
    pub image_formats: HashMap<u32, vk::Format>,
    /// Indirect: the group counts are three u32 in this buffer at this
    /// offset, read by the GPU when the dispatch runs (after every write
    /// recorded before it); `groups` is then unused.
    pub indirect: Option<(vk::Buffer, u64)>,
    /// The GPU page table, for a program that reads it.
    pub page_table: echora_shader_abi::PageTable,
    /// The texture pool entries a program reading the pool may name.
    pub pool: std::sync::Arc<[crate::pool::PoolEntry]>,
}

impl Renderer {
    /// Record `dispatch` without waiting: returns, for each storage
    /// buffer it writes, the token of its contents after the next flush
    /// (`take_reads`).
    pub fn dispatch_deferred(&mut self, dispatch: &Dispatch) -> Result<Vec<(u32, u64)>, String> {
        // Storage images are written at the guest's coordinates: their
        // targets stay unscaled (before recording: that may flush).
        for (key, _) in dispatch.images.values() {
            if !self.resolution_scale().is_one() {
                self.keep_unscaled(*key)?;
            }
        }
        self.bound_submission()?;
        let program = dispatch.program;
        let stage = program.stages.first().ok_or("program has no stages")?;
        // The texture pool's entries (writing them may flush).
        let pool = if program.pool {
            Some(self.bind_pool(&dispatch.pool)?)
        } else {
            None
        };
        // Textures first (an upload may flush), then room for the rest, so
        // nothing flushes while this dispatch is recorded.
        let mut bytes = 0;
        let mut descriptors = [0u32; 4];
        for descriptor in &stage.descriptors {
            let binding = descriptor.binding;
            match descriptor.kind() {
                DescriptorKind::Texture => {
                    if let Some((source, _)) = dispatch.textures.get(&binding) {
                        self.texture_view(source, descriptor.texture_type)?;
                    }
                    descriptors[1] += 1;
                }
                DescriptorKind::TextureBuffer => {
                    if let Some((source, _)) = dispatch.textures.get(&binding) {
                        self.texel_buffer_view(source)?;
                    }
                    descriptors[1] += 1;
                }
                DescriptorKind::Image => descriptors[3] += 1,
                DescriptorKind::StorageBuffer => {
                    if !dispatch.storage_in_place.contains_key(&binding) {
                        let length = dispatch.storage.get(&binding).map_or(16, Vec::len);
                        bytes += descriptor_buffer_range(length) + 256;
                    }
                    descriptors[2] += 1;
                }
                _ => {
                    let length = dispatch.uniforms.get(&binding).map_or(16, Vec::len);
                    bytes += descriptor_buffer_range(length) + 256;
                    descriptors[0] += 1;
                }
            }
        }
        // An indirect dispatch's bounded group counts (`bounded_indirect`).
        if dispatch.indirect.is_some() {
            bytes += 32;
        }
        self.reserve(bytes, descriptors)?;
        let mut writes = Vec::new();
        let mut images = Vec::new();
        let mut storage_images = Vec::new();
        let mut storage = Vec::new();
        let mut texel_buffers = Vec::new();
        for descriptor in &stage.descriptors {
            let binding = descriptor.binding;
            if descriptor.kind() == DescriptorKind::TextureBuffer {
                let (source, _) = dispatch
                    .textures
                    .get(&binding)
                    .ok_or_else(|| format!("no texel buffer for compute binding {binding}"))?;
                texel_buffers.push((binding, [self.texel_buffer_view(source)?]));
                continue;
            }
            if descriptor.kind() == DescriptorKind::Image {
                let (key, level) = *dispatch
                    .images
                    .get(&binding)
                    .ok_or_else(|| format!("no image for compute binding {binding}"))?;
                let format = self
                    .existing_target(key)
                    .ok_or("compute image is not a target")?
                    .format;
                let declared = crate::renderer::declared_image_format(descriptor.flags)
                    .or_else(|| dispatch.image_formats.get(&binding).copied());
                let view_format = self.storage_view_format(format, declared).ok_or_else(|| {
                    format!(
                        "storage image format {format:?} (declared {declared:?}) is not supported"
                    )
                })?;
                self.transition(key, vk::ImageLayout::GENERAL)?;
                // Storage image views name one mip level.
                let image_view = self.level_view_as(key, level, view_format)?;
                storage_images.push((
                    binding,
                    [vk::DescriptorImageInfo {
                        sampler: vk::Sampler::null(),
                        image_view,
                        image_layout: vk::ImageLayout::GENERAL,
                    }],
                ));
                continue;
            }
            if descriptor.kind() == DescriptorKind::Texture {
                let (source, sampler) = dispatch
                    .textures
                    .get(&binding)
                    .ok_or_else(|| format!("no texture for compute binding {binding}"))?;
                let image_view =
                    self.texture_view_swizzled(source, descriptor.texture_type, sampler)?;
                let sampler = self.sampler(sampler)?;
                images.push((
                    binding,
                    [vk::DescriptorImageInfo {
                        sampler,
                        image_view,
                        image_layout: vk::ImageLayout::GENERAL,
                    }],
                ));
                continue;
            }
            if descriptor.kind() == DescriptorKind::StorageBuffer {
                if let Some(&(buffer, offset, range)) = dispatch.storage_in_place.get(&binding) {
                    writes.push((
                        binding,
                        vk::DescriptorType::STORAGE_BUFFER,
                        [vk::DescriptorBufferInfo {
                            buffer,
                            offset,
                            range: range.max(16).next_multiple_of(16),
                        }],
                    ));
                    continue;
                }
            }
            let (kind, bytes) = match descriptor.kind() {
                DescriptorKind::UniformBuffer => (
                    vk::DescriptorType::UNIFORM_BUFFER,
                    dispatch.uniforms.get(&binding).map(Vec::as_slice),
                ),
                DescriptorKind::StorageBuffer => (
                    vk::DescriptorType::STORAGE_BUFFER,
                    dispatch.storage.get(&binding).map(Vec::as_slice),
                ),
                other => return Err(format!("{other:?} bindings are not modeled in compute")),
            };
            let bytes = bytes.unwrap_or(&[0; 16]);
            let (buffer, offset, range) = self.ring_upload_padded(bytes)?;
            // Only buffers the shader writes come back (flags bit 0).
            if kind == vk::DescriptorType::STORAGE_BUFFER && descriptor.flags & 1 != 0 {
                storage.push((binding, offset, bytes.len()));
            }
            writes.push((
                binding,
                kind,
                [vk::DescriptorBufferInfo {
                    buffer,
                    offset,
                    range,
                }],
            ));
        }
        let set = self.descriptor_set(program.layout)?;
        let descriptor_writes: Vec<_> = writes
            .iter()
            .map(|(binding, kind, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(*binding)
                    .descriptor_type(*kind)
                    .buffer_info(info)
            })
            .chain(images.iter().map(|(binding, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(*binding)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(info)
            }))
            .chain(storage_images.iter().map(|(binding, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(*binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(info)
            }))
            .chain(texel_buffers.iter().map(|(binding, view)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(*binding)
                    .descriptor_type(vk::DescriptorType::UNIFORM_TEXEL_BUFFER)
                    .texel_buffer_view(view)
            }))
            .collect();
        // SAFETY: the set and the ring are live.
        unsafe {
            self.gpu
                .device
                .update_descriptor_sets(&descriptor_writes, &[])
        };
        let objects = self.gpu.shader_objects()?.clone();
        // The group counts an indirect dispatch reads, bounded by the
        // device's limit on the GPU (`mesh_count`), before the condition.
        let indirect = match dispatch.indirect {
            Some(source) => {
                let (ring, at) = self.ring_upload(&[0; 12], 16)?;
                if self.bounded_indirect(
                    source,
                    (ring, at),
                    crate::mesh_count::IndirectCounts::Dispatch,
                )? {
                    Some((ring, at))
                } else {
                    self.stat("stat: dispatch counts read unbounded", 1);
                    Some(source)
                }
            }
            None => None,
        };
        // Gated on the guest's render condition, as the hardware gates
        // compute too.
        let commands = self.commands()?;
        self.apply_condition()?;
        let before = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::HOST_WRITE)
            .dst_access_mask(
                vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::UNIFORM_READ
                    | vk::AccessFlags::INDIRECT_COMMAND_READ,
            );
        let after = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ | vk::AccessFlags::MEMORY_READ);
        let rescaling = (!program.push_stages.is_empty()).then(|| {
            let scaled: Vec<bool> = program
                .texture_bits
                .iter()
                .map(|(_, binding)| {
                    dispatch
                        .textures
                        .get(binding)
                        .is_some_and(|(source, _)| self.samples_scaled(source))
                })
                .collect();
            crate::draw::rescaling_constants(&scaled, 1.0)
        });
        self.time_item(true);
        if pool.is_some() {
            self.record_texture_pool_use();
        }
        // SAFETY: recording outside rendering; the program and set are live.
        unsafe {
            let device = &self.gpu.device;
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
                vk::DependencyFlags::empty(),
                &[before],
                &[],
                &[],
            );
            objects.cmd_bind_shaders(commands, &[vk::ShaderStageFlags::COMPUTE], &program.shaders);
            if let Some(constants) = &rescaling {
                device.cmd_push_constants(
                    commands,
                    program.pipeline_layout,
                    program.push_stages,
                    0,
                    constants,
                );
            }
            if !program.abi_stages.is_empty() {
                let table = dispatch.page_table;
                let mut bytes = [0u8; 16];
                bytes[..8].copy_from_slice(&table.address.to_le_bytes());
                bytes[8..12].copy_from_slice(&table.first_page.to_le_bytes());
                bytes[12..].copy_from_slice(&table.page_count.to_le_bytes());
                device.cmd_push_constants(
                    commands,
                    program.pipeline_layout,
                    program.abi_stages,
                    echora_shader_abi::PUSH_PAGE_TABLE_OFFSET,
                    &bytes,
                );
            }
            device.cmd_bind_descriptor_sets(
                commands,
                vk::PipelineBindPoint::COMPUTE,
                program.pipeline_layout,
                0,
                &[set],
                &[],
            );
            if let Some(pool) = pool {
                device.cmd_bind_descriptor_sets(
                    commands,
                    vk::PipelineBindPoint::COMPUTE,
                    program.pipeline_layout,
                    echora_shader_abi::TEXTURE_POOL_SET,
                    &[pool],
                    &[],
                );
            }
            match indirect {
                Some((buffer, offset)) => device.cmd_dispatch_indirect(commands, buffer, offset),
                None => {
                    let [x, y, z] = dispatch.groups;
                    device.cmd_dispatch(commands, x, y, z);
                }
            }
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST | vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[after],
                &[],
                &[],
            );
        }
        self.time_item(false);
        // Only written storage buffers come back to the guest, after the
        // next flush (the ring keeps them until it is reused).
        Ok(storage
            .into_iter()
            .map(|(binding, offset, length)| (binding, self.defer_ring_read(offset, length)))
            .collect())
    }

    /// Run `dispatch` and wait for it; returns each written storage
    /// buffer's contents afterwards, by binding.
    pub fn dispatch(&mut self, dispatch: &Dispatch) -> Result<HashMap<u32, Vec<u8>>, String> {
        let tokens = self.dispatch_deferred(dispatch)?;
        if tokens.is_empty() {
            return Ok(HashMap::new());
        }
        self.flush()?;
        let ids: Vec<u64> = tokens.iter().map(|(_, token)| *token).collect();
        let mut completed = self.take_reads(&ids);
        Ok(tokens
            .into_iter()
            .map(|(binding, token)| (binding, completed.remove(&token).unwrap_or_default()))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::draw::tests::{spirv, translated};
    use crate::shader::{Descriptor, Stage};
    use crate::vulkan::Gpu;

    #[test]
    fn compute_fetch_matches_the_texture_view_numeric_type() {
        use crate::draw::{ImageShape, UploadRegion};
        use echora_shader::compute::{Layout, SampledTexture, translate};
        use echora_shader_abi::{PAGE_SHIFT, PAGE_SIZE};
        use std::sync::Arc;

        let Ok(gpu) = Gpu::new() else {
            return;
        };
        if gpu.shader_objects().is_err() || !gpu.buffer_device_address {
            return;
        }
        let mut renderer = Renderer::new(gpu).unwrap();
        let word = |fields: &[(u32, u128)]| {
            echora_shader::decode(
                fields
                    .iter()
                    .fold(7 << 12, |word, &(shift, value)| word | value << shift),
            )
        };
        let mov = |register, value| word(&[(0, 0x802), (16, register), (32, value), (72, 15)]);
        let guest_address = 0x1_0000u32;
        let fetch = word(&[
            (0, 0xb66),
            (16, 10),
            (24, 255),
            (32, 255),
            (40, 0x10),
            (61, 1),
            (64, 255),
            (72, 1),
            (81, 7),
            (87, 1),
        ]);
        let store = word(&[(0, 0x386), (24, 4), (32, 10), (72, 1), (73, 4)]);
        let exit = word(&[(0, 0x94d), (87, 7)]);
        let code = [
            mov(4, u128::from(guest_address)),
            mov(5, 0),
            fetch,
            store,
            exit,
        ];
        let output = 1;
        let (buffer, _) = renderer.resident_buffer(output, PAGE_SIZE).unwrap();
        let address = renderer.buffer_address(buffer).unwrap();
        let (table, _) = renderer
            .set_page_table(guest_address >> PAGE_SHIFT, &[address])
            .unwrap();
        for (index, (view, texel, expected)) in [
            (vk::Format::R16G16_UINT, 0x4321u16, 0x4321),
            (vk::Format::R16G16_SINT, (-123i16) as u16, (-123i32) as u32),
            (vk::Format::R16G16_UNORM, u16::MAX, 1.0f32.to_bits()),
        ]
        .into_iter()
        .enumerate()
        {
            renderer.fill_resident(output, 0, 4, 0).unwrap();
            let formats = [SampledTexture {
                bank: 0,
                offset: 0x40,
                view: 2,
                sample_type: crate::renderer::sampled_texture_type(view),
            }];
            let stage = Arc::new(
                translate(
                    &code,
                    &Layout {
                        sampled_textures: &formats,
                        ..Layout::workgroup([1, 1, 1])
                    },
                )
                .unwrap(),
            );
            let binding = stage
                .descriptors
                .iter()
                .find(|d| d.kind() == DescriptorKind::Texture)
                .unwrap()
                .binding;
            let program = renderer.create_program(vec![stage]).unwrap();
            let source = TextureSource::Upload {
                key: 2 + index as u64,
                generation: 1,
                width: 1,
                height: 1,
                shape: ImageShape::D2 { layers: 1 },
                format: view,
                texels: Arc::new([texel, 0].into_iter().flat_map(u16::to_le_bytes).collect()),
                levels: vec![UploadRegion {
                    level: 0,
                    layer: 0,
                    width: 1,
                    height: 1,
                    depth: 1,
                    offset: 0,
                }]
                .into(),
            };
            let sampler = Sampler {
                min_filter: vk::Filter::NEAREST,
                mag_filter: vk::Filter::NEAREST,
                ..Sampler::default()
            };
            let dispatch = Dispatch {
                program: &program,
                groups: [1, 1, 1],
                uniforms: HashMap::new(),
                storage: HashMap::new(),
                storage_in_place: HashMap::new(),
                textures: [(binding, (source, sampler))].into(),
                images: HashMap::new(),
                image_formats: HashMap::new(),
                indirect: None,
                page_table: table,
                pool: Default::default(),
            };
            renderer.dispatch(&dispatch).unwrap();
            let token = renderer.read_resident(output, 0, 4).unwrap();
            renderer.flush().unwrap();
            assert_eq!(
                renderer.take_reads(&[token])[&token],
                expected.to_le_bytes(),
                "{view:?}"
            );
            renderer.destroy_program(program).unwrap();
        }
    }

    #[test]
    fn image_reductions_can_use_device_scope() {
        use echora_shader::compute::{Layout, translate};
        use std::sync::Arc;

        let Ok(gpu) = Gpu::new() else {
            return;
        };
        if gpu.shader_objects().is_err() {
            return;
        }
        let mut renderer = Renderer::new(gpu).unwrap();
        let word = |fields: &[(u32, u128)]| {
            echora_shader::decode(
                fields
                    .iter()
                    .fold(7 << 12, |word, &(shift, value)| word | value << shift),
            )
        };
        let mov = |value| word(&[(0, 0x802), (32, value), (72, 15)]);
        let reduce = word(&[
            (0, 0x59f),
            (16, 255),
            (24, 255),
            (32, 0),
            (40, 0x18),
            (61, 3),
            (77, 7),
            (84, 1),
            (87, 6),
        ]);
        let exit = word(&[(0, 0x94d), (87, 7)]);
        let shader = translate(
            &[mov(0x12), reduce, mov(0x20), reduce, exit],
            &Layout::workgroup([1, 1, 1]),
        )
        .unwrap();
        let image = shader
            .descriptors
            .iter()
            .find(|d| d.kind() == DescriptorKind::Image)
            .unwrap()
            .binding;
        let program = renderer.create_program(vec![Arc::new(shader)]).unwrap();
        let output = 1;
        renderer.target(output, 1, 1, vk::Format::R32_UINT).unwrap();
        renderer
            .clear_color(output, None, vk::ClearColorValue { uint32: [0; 4] })
            .unwrap();
        let dispatch = Dispatch {
            program: &program,
            groups: [1, 1, 1],
            uniforms: HashMap::new(),
            storage: HashMap::new(),
            storage_in_place: HashMap::new(),
            textures: HashMap::new(),
            images: [(image, (output, 0))].into(),
            image_formats: HashMap::new(),
            indirect: None,
            page_table: Default::default(),
            pool: Default::default(),
        };
        renderer.dispatch(&dispatch).unwrap();
        assert_eq!(
            renderer.read_target_bytes(output, 4).unwrap(),
            0x32u32.to_le_bytes()
        );
        renderer.destroy_program(program).unwrap();
    }

    #[test]
    fn a_dispatch_stores_a_uniform_into_a_storage_buffer() {
        let Ok(gpu) = Gpu::new() else {
            return; // No device (CI).
        };
        let mut renderer = Renderer::new(gpu).unwrap();
        let code = spirv(
            "struct Words { data: array<u32> }
            @group(0) @binding(0) var<uniform> value: vec4<u32>;
            @group(0) @binding(1) var<storage, read_write> words: Words;
            @compute @workgroup_size(1) fn main() { words.data[1] = value.x + value.y; }",
            naga::ShaderStage::Compute,
        );
        let uniform = Descriptor {
            kind: 0,
            binding: 0,
            count: 1,
            index: 3,
            ..Default::default()
        };
        let storage = Descriptor {
            kind: 1,
            binding: 1,
            count: 1,
            // Written: only written buffers come back.
            flags: 1,
            ..Default::default()
        };
        let program = renderer
            .create_program(vec![translated(
                Stage::Compute,
                code,
                vec![uniform, storage],
            )])
            .unwrap();
        let mut value = vec![0u8; 16];
        value[..4].copy_from_slice(&0x1234_5678u32.to_le_bytes());
        value[4] = 1;
        let mut dispatch = Dispatch {
            program: &program,
            groups: [1, 1, 1],
            uniforms: HashMap::from([(0, value)]),
            storage: HashMap::from([(1, vec![0xaa; 8])]),
            storage_in_place: HashMap::new(),
            textures: HashMap::new(),
            images: HashMap::new(),
            image_formats: HashMap::new(),
            indirect: None,
            page_table: Default::default(),
            pool: Default::default(),
        };
        let results = renderer.dispatch(&dispatch).unwrap();
        assert_eq!(
            results[&1],
            [0xaa, 0xaa, 0xaa, 0xaa, 0x79, 0x56, 0x34, 0x12]
        );
        // The previous dispatch left value.y in this ring slot.
        dispatch.uniforms.get_mut(&0).unwrap().truncate(4);
        let results = renderer.dispatch(&dispatch).unwrap();
        assert_eq!(
            results[&1],
            [0xaa, 0xaa, 0xaa, 0xaa, 0x78, 0x56, 0x34, 0x12]
        );
        renderer.destroy_program(program).unwrap();
    }

    #[test]
    fn texture_pool_rewrite_preserves_bindings_for_pending_work() {
        for use_spare in [false, true] {
            check_texture_pool_rewrite(use_spare);
        }
    }

    fn check_texture_pool_rewrite(use_spare: bool) {
        let Ok(gpu) = Gpu::new() else {
            return;
        };
        if gpu.texture_pool_layout.is_none() || (use_spare && !gpu.spare_texture_pool) {
            return;
        }
        let mut renderer = Renderer::new(gpu).unwrap();
        renderer.tried_spare_texture_pool = !use_spare;
        for key in 1..=3 {
            renderer
                .target(key, 4, 4, vk::Format::R8G8B8A8_UNORM)
                .unwrap();
        }
        for (key, red) in [(1, 0.0), (2, 1.0)] {
            renderer
                .clear_color(
                    key,
                    None,
                    vk::ClearColorValue {
                        float32: [red, 0.0, 0.0, 1.0],
                    },
                )
                .unwrap();
        }
        renderer.flush().unwrap();

        let code = spirv(
            "enable wgpu_binding_array;
             @group(0) @binding(0) var<storage, read_write> output: array<u32>;
             @group(1) @binding(3) var textures: binding_array<texture_2d<f32>, 8>;
             @compute @workgroup_size(1) fn main() {
                 output[0] = u32(textureLoad(textures[1], vec2<i32>(0, 0), 0).r);
             }",
            naga::ShaderStage::Compute,
        );
        let mut stage = translated(
            Stage::Compute,
            code,
            vec![Descriptor {
                kind: 1,
                binding: 0,
                count: 1,
                flags: 1,
                ..Default::default()
            }],
        );
        std::sync::Arc::get_mut(&mut stage)
            .unwrap()
            .texture_pool_bindings = 1 << echora_shader_abi::texture_pool::SAMPLED_2D;
        let program = renderer.create_program(vec![stage]).unwrap();
        let dispatch = |key| Dispatch {
            program: &program,
            groups: [1, 1, 1],
            uniforms: HashMap::new(),
            storage: HashMap::from([(0, vec![0xff; 16])]),
            storage_in_place: HashMap::new(),
            textures: HashMap::new(),
            images: HashMap::new(),
            image_formats: HashMap::new(),
            indirect: None,
            page_table: Default::default(),
            pool: vec![crate::pool::PoolEntry {
                texture: 1,
                sampler: 1,
                source: TextureSource::Target(key),
                sampler_state: Sampler::default(),
            }]
            .into(),
        };
        let first = renderer.dispatch_deferred(&dispatch(1)).unwrap()[0].1;
        if !use_spare {
            // Without a second set, the rewrite must wait for only the
            // submission using the first set and leave newer work recorded.
            renderer.submit_frame().unwrap();
        }
        renderer
            .clear_color(3, None, vk::ClearColorValue { float32: [0.0; 4] })
            .unwrap();
        let serial = renderer.recorded_serial();
        let second = renderer.dispatch_deferred(&dispatch(2)).unwrap()[0].1;
        assert_eq!(
            renderer.recorded_serial(),
            serial,
            "work stayed recorded through the descriptor rewrite"
        );
        let stats = renderer.take_stats();
        let count = |name| stats.get(name).copied().unwrap_or(0);
        assert_eq!(count("stat: texture pool rotations"), u64::from(use_spare));
        assert_eq!(
            count("stat: texture pool flushes for rewritten entries"),
            u64::from(!use_spare)
        );
        renderer.flush().unwrap();
        let reads = renderer.take_reads(&[first, second]);
        assert_eq!(reads[&first][..4], 0u32.to_le_bytes());
        assert_eq!(reads[&second][..4], 1u32.to_le_bytes());
        renderer.destroy_program(program).unwrap();
    }
}
