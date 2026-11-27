//! Compute dispatches: one translated compute stage, its uniform buffers and
//! its storage buffers. Storage buffers travel through the host-visible ring
//! and are read back after the dispatch, so the caller can store them into
//! guest memory.

use std::collections::HashMap;

use ash::vk;

use crate::draw::{Program, Sampler, TextureSource};
use crate::renderer::{Renderer, descriptor_buffer_range};
use crate::shader::DescriptorKind;

/// A compute binding to one mip and a range of a target's array layers.
#[derive(Clone, Copy, Debug, Default)]
pub struct StorageImage {
    pub target: u64,
    pub level: u32,
    pub base_layer: u32,
    /// Zero selects the remaining layers.
    pub layer_count: u32,
}

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
    pub images: HashMap<u32, StorageImage>,
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
    /// The Vulkan format a storage descriptor will use for the requested
    /// image. A typed shader translation uses this format as well.
    pub fn storage_image_binding_format(
        &self,
        image: StorageImage,
        declaration: u32,
        view: Option<vk::Format>,
    ) -> Result<vk::Format, String> {
        let format = self
            .existing_target(image.target)
            .ok_or("compute image is not a target")?
            .format;
        let declared = crate::renderer::declared_image_format(declaration).or(view);
        self.storage_view_format(format, declared).ok_or_else(|| {
            format!("storage image format {format:?} (declared {declared:?}) is not supported")
        })
    }

    /// Record `dispatch` without waiting: returns, for each storage
    /// buffer it writes, the token of its contents after the next flush
    /// (`take_reads`).
    pub fn dispatch_deferred(&mut self, dispatch: &Dispatch) -> Result<Vec<(u32, u64)>, String> {
        self.dispatch_deferred_with_grid_uniform(dispatch, None)
    }

    /// An NVN driver uniform starts with the table size followed by the three
    /// workgroup counts. Indirect dispatches write those counts on the GPU so
    /// the command and the shader see the same values after preceding work.
    /// The indirect source must also support transfer reads on devices without
    /// buffer addresses.
    pub fn dispatch_deferred_with_grid_uniform(
        &mut self,
        dispatch: &Dispatch,
        grid_uniform: Option<u32>,
    ) -> Result<Vec<(u32, u64)>, String> {
        // NVN's render enable gates 3D work. Vulkan also predicates dispatches,
        // so leave its active scope and keep the wanted predicate for the next draw.
        self.end_condition();
        // Storage images are written at the guest's coordinates: their
        // targets stay unscaled (before recording: that may flush).
        for image in dispatch.images.values() {
            if !self.resolution_scale().is_one() {
                self.keep_unscaled(image.target)?;
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
        let mut grid_target = None;
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
                let image = dispatch
                    .images
                    .get(&binding)
                    .ok_or_else(|| format!("no image for compute binding {binding}"))?;
                let key = image.target;
                let view_format = self.storage_image_binding_format(
                    *image,
                    descriptor.flags,
                    dispatch.image_formats.get(&binding).copied(),
                )?;
                self.transition(key, vk::ImageLayout::GENERAL)?;
                // Storage image views name one mip level.
                let image_view =
                    self.storage_image_view(*image, descriptor.texture_type, view_format)?;
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
                self.check_sampler_view(sampler, image_view)?;
                let sampler = self.sampler(
                    sampler,
                    descriptor.flags & crate::shader::Descriptor::INTEGER != 0,
                )?;
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
                    if range == 0 {
                        return Err(format!(
                            "empty shared storage for compute binding {binding}"
                        ));
                    }
                    writes.push((
                        binding,
                        vk::DescriptorType::STORAGE_BUFFER,
                        [vk::DescriptorBufferInfo {
                            buffer,
                            offset,
                            range,
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
            let (buffer, offset, range) = if kind == vk::DescriptorType::STORAGE_BUFFER {
                self.ring_upload_storage(bytes)?
            } else {
                self.ring_upload_padded(bytes)?
            };
            if dispatch.indirect.is_some() && grid_uniform == Some(binding) {
                if kind != vk::DescriptorType::UNIFORM_BUFFER || bytes.len() < 16 {
                    return Err("indirect grid header needs a 16-byte uniform".into());
                }
                grid_target = Some((buffer, offset + 4));
            }
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
        // SAFETY: `self.gpu` stays in place for this call.
        let gpu = unsafe { self.detached_gpu() };
        let objects = gpu.shader_objects()?;
        // The group counts an indirect dispatch reads, bounded by the
        // device's limit on the GPU (`mesh_count`).
        let indirect = match dispatch.indirect {
            Some(source) => {
                let target = match grid_target {
                    Some(target) => target,
                    None => self.ring_upload(&[0; 12], 16)?,
                };
                if self.bounded_indirect(
                    source,
                    target,
                    crate::mesh_count::IndirectCounts::Dispatch,
                )? {
                    Some(target)
                } else {
                    self.stat("stat: dispatch counts read unbounded", 1);
                    if grid_target.is_some() {
                        self.copy_indirect_counts(source, target, 12)?;
                        Some(target)
                    } else {
                        Some(source)
                    }
                }
            }
            None => None,
        };
        let commands = self.commands()?;
        // Compute can overwrite images and buffers that earlier work wrote.
        // Publish those writes to shader writes as well as shader reads.
        let before = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::HOST_WRITE)
            .dst_access_mask(
                vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::SHADER_WRITE
                    | vk::AccessFlags::UNIFORM_READ
                    | vk::AccessFlags::INDIRECT_COMMAND_READ,
            );
        let after = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(
                vk::AccessFlags::HOST_READ
                    | vk::AccessFlags::MEMORY_READ
                    | vk::AccessFlags::MEMORY_WRITE,
            );
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
        if !program.abi_stages.is_empty() {
            self.record_page_table_read(dispatch.page_table.address);
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
    fn indirect_driver_uniform_matches_the_gpu_written_dispatch_grid() {
        for use_addresses in [true, false] {
            let Ok(mut gpu) = Gpu::new() else { return };
            if gpu.shader_objects().is_err() {
                return;
            }
            gpu.buffer_device_address &= use_addresses;
            let mut renderer = Renderer::new(gpu).unwrap();
            let code = spirv(
                "@group(0) @binding(0) var<uniform> header: vec4<u32>;
                 @group(0) @binding(1) var<storage, read_write> output: array<u32>;
                 @compute @workgroup_size(1) fn main(
                    @builtin(workgroup_id) id: vec3<u32>, @builtin(num_workgroups) grid: vec3<u32>
                 ) {
                    if any(id != vec3<u32>(0)) { return; }
                    output[0] = header.x; output[1] = header.y;
                    output[2] = header.z; output[3] = header.w;
                    output[4] = grid.x; output[5] = grid.y; output[6] = grid.z;
                 }",
                naga::ShaderStage::Compute,
            );
            let descriptors = vec![
                Descriptor {
                    kind: DescriptorKind::UniformBuffer as u32,
                    binding: 0,
                    count: 1,
                    ..Default::default()
                },
                Descriptor {
                    kind: DescriptorKind::StorageBuffer as u32,
                    binding: 1,
                    count: 1,
                    flags: 1,
                    ..Default::default()
                },
            ];
            let program = renderer
                .create_program(vec![translated(Stage::Compute, code, descriptors)])
                .unwrap();
            let (args, _) = renderer.resident_buffer(1, 32).unwrap();
            let mut dispatch = Dispatch {
                program: &program,
                groups: [1; 3],
                uniforms: HashMap::new(),
                storage: HashMap::new(),
                storage_in_place: HashMap::new(),
                textures: HashMap::new(),
                images: HashMap::new(),
                image_formats: HashMap::new(),
                indirect: None,
                page_table: Default::default(),
                pool: Default::default(),
            };
            let mut pending = Vec::new();
            for grid in [[2u32, 3, 1], [1, 2, 4]] {
                let [x, y, z] = grid;
                dispatch.indirect = None;
                dispatch.storage_in_place.insert(1, (args, 0, 32));
                dispatch.uniforms.insert(
                    0,
                    [0u32, x, y, z]
                        .map(u32::to_le_bytes)
                        .as_flattened()
                        .to_vec(),
                );
                renderer.dispatch_deferred(&dispatch).unwrap();

                dispatch.indirect = Some((args, 4));
                dispatch.storage_in_place.clear();
                dispatch.storage.insert(1, vec![0; 28]);
                dispatch.uniforms.insert(
                    0,
                    [208u32, 0, 0, 0]
                        .map(u32::to_le_bytes)
                        .as_flattened()
                        .to_vec(),
                );
                let result = renderer
                    .dispatch_deferred_with_grid_uniform(&dispatch, Some(0))
                    .unwrap();
                pending.push((result[0].1, [208, x, y, z, x, y, z]));
            }
            renderer.flush().unwrap();
            for (token, expected) in pending {
                assert_eq!(
                    renderer.take_reads(&[token])[&token],
                    expected.map(u32::to_le_bytes).as_flattened()
                );
            }
            renderer.destroy_program(program).unwrap();
        }
    }

    #[test]
    fn storage_array_length_uses_the_bound_bytes_for_shared_and_copied_buffers() {
        let Ok(gpu) = Gpu::new() else { return };
        if gpu.shader_objects().is_err() {
            return;
        }
        let mut renderer = Renderer::new(gpu).unwrap();
        let code = spirv(
            "@group(0) @binding(0) var<storage, read> words: array<u32>;
             @group(0) @binding(1) var<storage, read_write> result: u32;
             @compute @workgroup_size(1) fn main() { result = arrayLength(&words); }",
            naga::ShaderStage::Compute,
        );
        let descriptors = [0, 1].map(|binding| Descriptor {
            kind: DescriptorKind::StorageBuffer as u32,
            binding,
            count: 1,
            flags: binding,
            ..Default::default()
        });
        let program = renderer
            .create_program(vec![translated(Stage::Compute, code, descriptors.into())])
            .unwrap();
        let (buffer, _) = renderer.resident_buffer(1, 64).unwrap();
        let mut dispatch = Dispatch {
            program: &program,
            groups: [1; 3],
            storage: HashMap::from([(1, vec![0; 4])]),
            storage_in_place: HashMap::new(),
            uniforms: HashMap::new(),
            textures: HashMap::new(),
            images: HashMap::new(),
            image_formats: HashMap::new(),
            indirect: None,
            page_table: Default::default(),
            pool: Default::default(),
        };
        for shared in [false, true] {
            for words in [1_u32, 3, 4, 5] {
                let length = words * 4;
                if shared {
                    dispatch
                        .storage_in_place
                        .insert(0, (buffer, 0, u64::from(length)));
                } else {
                    dispatch.storage.insert(0, vec![0; length as usize]);
                }
                assert_eq!(
                    renderer.dispatch(&dispatch).unwrap()[&1],
                    words.to_le_bytes(),
                    "shared {shared}, bytes {length}"
                );
            }
        }
        renderer.destroy_program(program).unwrap();
    }

    #[test]
    fn compute_texture_reads_match_the_view_numeric_type() {
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
        let fetch_code = [
            mov(4, u128::from(guest_address)),
            mov(5, 0),
            fetch,
            store,
            exit,
        ];
        let sample = word(&[
            (0, 0xb60),
            (16, 10),
            (32, 2),
            (40, 0x10),
            (61, 1),
            (64, 255),
            (72, 1),
            (81, 7),
            (87, 3),
        ]);
        let border_code = [
            mov(0, u128::from(1.75f32.to_bits())),
            mov(1, u128::from(0.5f32.to_bits())),
            mov(2, 0),
            mov(4, u128::from(guest_address)),
            mov(5, 0),
            sample,
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
            let formats = [SampledTexture {
                bank: 0,
                offset: 0x40,
                view: 2,
                sample_type: crate::renderer::sampled_texture_type(view),
            }];
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
            let expected_border = if crate::renderer::integer_format(view) {
                1
            } else {
                1.0f32.to_bits()
            };
            let border = Sampler {
                address: [vk::SamplerAddressMode::CLAMP_TO_BORDER; 3],
                border: [expected_border; 4],
                ..sampler
            };
            for (code, sampler, expected) in [
                (fetch_code.as_slice(), sampler, expected),
                (border_code.as_slice(), border, expected_border),
            ] {
                renderer.fill_resident(output, 0, 4, 0).unwrap();
                let stage = Arc::new(
                    translate(
                        code,
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
                let dispatch = Dispatch {
                    program: &program,
                    groups: [1, 1, 1],
                    uniforms: HashMap::new(),
                    storage: HashMap::new(),
                    storage_in_place: HashMap::new(),
                    textures: [(binding, (source.clone(), sampler))].into(),
                    images: HashMap::new(),
                    image_formats: HashMap::new(),
                    indirect: None,
                    page_table: table,
                    pool: Default::default(),
                };
                renderer.dispatch(&dispatch).unwrap();
                let token = renderer.read_resident(output, 0, 4).unwrap();
                renderer.flush().unwrap();
                let actual = renderer.take_reads(&[token]).remove(&token).unwrap();
                renderer.destroy_program(program).unwrap();
                assert_eq!(
                    actual,
                    expected.to_le_bytes(),
                    "{view:?}, {:?}",
                    sampler.address,
                );
            }
        }
    }

    #[test]
    fn reserved_null_texture_samples_and_queries_four_zero_words() {
        use echora_shader::compute::{Layout, translate};
        use echora_shader_abi::{PAGE_SHIFT, PAGE_SIZE};
        use std::sync::Arc;
        let Ok(gpu) = Gpu::new() else { return };
        if gpu.shader_objects().is_err() || !gpu.buffer_device_address || !gpu.null_descriptor {
            return;
        }
        let mut renderer = Renderer::new(gpu).unwrap();
        let word = |fields: &[(u32, u128)]| {
            echora_shader::decode(
                fields
                    .iter()
                    .fold(7 << 12, |w, &(shift, value)| w | value << shift),
            )
        };
        let mov = |register, value| word(&[(0, 0x802), (16, register), (32, value), (72, 15)]);
        let guest = 0x1_0000u32;
        let key = 1;
        let (output, _) = renderer.resident_buffer(key, PAGE_SIZE).unwrap();
        let address = renderer.buffer_address(output).unwrap();
        let (table, _) = renderer
            .set_page_table(guest >> PAGE_SHIFT, &[address])
            .unwrap();
        for bindless in [false, true] {
            if bindless && renderer.gpu.texture_pool_layout.is_none() {
                continue;
            }
            let mut code = vec![mov(0, 0), mov(1, 0), mov(2, 0), mov(5, 0)];
            // TEX.LZ RGBA and TXQ.DIMENSION through the same 2D image.
            code.push(word(&[
                (0, if bindless { 0x361 } else { 0xb60 }),
                (16, 8),
                (32, if bindless { 2 } else { 1 }),
                (40, if bindless { 0 } else { 0x10 }),
                (59, u128::from(bindless)),
                (61, 1),
                (64, 10),
                (72, 15),
                (81, 7),
                (87, 1),
            ]));
            code.push(word(&[
                (0, if bindless { 0x370 } else { 0xb6f }),
                (16, 12),
                (32, if bindless { 0 } else { 255 }),
                (40, if bindless { 0 } else { 0x10 }),
                (59, u128::from(bindless)),
                (64, 255),
                (72, 15),
            ]));
            for index in 0..8 {
                code.push(mov(4, u128::from(guest) + index * 4));
                code.push(word(&[
                    (0, 0x386),
                    (24, 4),
                    (32, 8 + index),
                    (72, 1),
                    (73, 4),
                ]));
            }
            code.push(word(&[(0, 0x94d), (87, 7)]));
            let stage = Arc::new(translate(&code, &Layout::workgroup([1, 1, 1])).unwrap());
            assert_eq!(stage.texture_pool_bindings != 0, bindless);
            let textures = stage
                .descriptors
                .iter()
                .filter(|d| d.kind() == DescriptorKind::Texture)
                .map(|d| (d.binding, (TextureSource::Null, Sampler::default())))
                .collect();
            let program = renderer.create_program(vec![stage]).unwrap();
            // A skipped store or a normalized RGB descriptor with alpha=1 must fail.
            renderer.upload_resident(key, 0, &[0xff; 32]).unwrap();
            renderer
                .dispatch(&Dispatch {
                    program: &program,
                    groups: [1, 1, 1],
                    uniforms: HashMap::new(),
                    storage: HashMap::new(),
                    storage_in_place: HashMap::new(),
                    textures,
                    images: HashMap::new(),
                    image_formats: HashMap::new(),
                    indirect: None,
                    page_table: table,
                    pool: Default::default(),
                })
                .unwrap();
            let token = renderer.read_resident(key, 0, 32).unwrap();
            renderer.flush().unwrap();
            assert_eq!(renderer.take_reads(&[token])[&token], [0; 32]);
            renderer.destroy_program(program).unwrap();
        }
    }

    #[test]
    fn sampling_respects_mirror_clamp_base_levels_and_reduction() {
        use crate::draw::{ImageShape, UploadRegion};
        use echora_shader::compute::{Layout, translate};
        use echora_shader_abi::{PAGE_SHIFT, PAGE_SIZE};
        use std::sync::Arc;

        let Ok(gpu) = Gpu::new() else {
            return;
        };
        if gpu.shader_objects().is_err() || !gpu.buffer_device_address {
            return;
        }
        let supported = gpu.sampler_mirror_clamp_to_edge;
        let mut renderer = Renderer::new(gpu).unwrap();
        let word = |fields: &[(u32, u128)]| {
            echora_shader::decode(
                fields
                    .iter()
                    .fold(7 << 12, |word, &(shift, value)| word | value << shift),
            )
        };
        let mov = |register, value: u32| {
            word(&[
                (0, 0x802),
                (16, register),
                (32, u128::from(value)),
                (72, 15),
            ])
        };
        let guest_address = 0x1_0000u32;
        // TEX.LL through c[0][0x40], reading (x, y, level) from R0..R2.
        let sample = word(&[
            (0, 0xb60),
            (16, 10),
            (32, 2),
            (40, 0x10),
            (61, 1),
            (64, 255),
            (72, 1),
            (81, 7),
            (87, 3),
        ]);
        let store = word(&[(0, 0x386), (24, 4), (32, 10), (72, 1), (73, 4)]);
        let exit = word(&[(0, 0x94d), (87, 7)]);
        let stage = Arc::new(
            translate(
                &[
                    mov(0, 1.625f32.to_bits()),
                    mov(1, 0.5f32.to_bits()),
                    mov(2, 1.0f32.to_bits()),
                    mov(4, guest_address),
                    mov(5, 0),
                    sample,
                    store,
                    exit,
                ],
                &Layout::workgroup([1, 1, 1]),
            )
            .unwrap(),
        );
        let binding = stage
            .descriptors
            .iter()
            .find(|descriptor| descriptor.kind() == DescriptorKind::Texture)
            .unwrap()
            .binding;
        let program = renderer.create_program(vec![stage]).unwrap();
        let output = 1;
        let (buffer, _) = renderer.resident_buffer(output, PAGE_SIZE).unwrap();
        let address = renderer.buffer_address(buffer).unwrap();
        let (table, _) = renderer
            .set_page_table(guest_address >> PAGE_SHIFT, &[address])
            .unwrap();
        let source = TextureSource::Upload {
            key: 2,
            generation: 1,
            width: 2,
            height: 1,
            shape: ImageShape::D2 { layers: 1 },
            format: vk::Format::R32_SFLOAT,
            texels: Arc::new(
                [0.25f32, 0.75, 0.125]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect(),
            ),
            levels: vec![
                UploadRegion {
                    level: 0,
                    layer: 0,
                    width: 2,
                    height: 1,
                    depth: 1,
                    offset: 0,
                },
                UploadRegion {
                    level: 1,
                    layer: 0,
                    width: 1,
                    height: 1,
                    depth: 1,
                    offset: 8,
                },
            ]
            .into(),
        };
        let mirror = Sampler {
            min_filter: vk::Filter::NEAREST,
            mag_filter: vk::Filter::NEAREST,
            base_level_only: false,
            address: [
                vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE,
                vk::SamplerAddressMode::CLAMP_TO_EDGE,
                vk::SamplerAddressMode::CLAMP_TO_EDGE,
            ],
            lod: [0.0; 3],
            ..Sampler::default()
        };
        let mut repeated = mirror;
        repeated.address[0] = vk::SamplerAddressMode::MIRRORED_REPEAT;
        let edge = if supported { 0.75 } else { 0.25 };
        let mut cases = vec![
            (repeated, 0.25f32),
            (mirror, edge),
            (
                Sampler {
                    lod: [1.0, 1.0, 0.0],
                    ..mirror
                },
                0.125,
            ),
            (
                Sampler {
                    base_level_only: true,
                    mipmap: vk::SamplerMipmapMode::LINEAR,
                    lod: [8.0, 12.0, 0.0],
                    ..mirror
                },
                edge,
            ),
        ];
        // At x=1.625 clamp-to-edge selects 0.75 in the base mip and 0.125
        // in mip 1. Fractional LOD gives both texels non-zero weight.
        let trilinear = Sampler {
            address: [vk::SamplerAddressMode::CLAMP_TO_EDGE; 3],
            lod: [0.5, 0.5, 0.0],
            mipmap: vk::SamplerMipmapMode::LINEAR,
            ..mirror
        };
        cases.push((trilinear, 0.4375));
        // Wrapping x to 0.625 interpolates adjacent base-level texels at
        // weights 0.25 and 0.75; both contribute to the min/max footprint.
        let spatial = Sampler {
            address: [vk::SamplerAddressMode::REPEAT; 3],
            ..Sampler::default()
        };
        cases.push((spatial, 0.625));
        // SAFETY: this instance owns the live physical device.
        let formats = unsafe {
            renderer.gpu.instance.get_physical_device_format_properties(
                renderer.gpu.physical,
                vk::Format::R32_SFLOAT,
            )
        };
        if renderer.gpu.sampler_filter_minmax
            && renderer.gpu.filter_minmax_image_component_mapping
            && formats
                .optimal_tiling_features
                .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_MINMAX)
        {
            cases.push((
                Sampler {
                    reduction: vk::SamplerReductionMode::MIN,
                    ..spatial
                },
                0.25,
            ));
            cases.push((
                Sampler {
                    reduction: vk::SamplerReductionMode::MAX,
                    ..spatial
                },
                0.75,
            ));
        }
        for (sampler, expected) in cases {
            let dispatch = Dispatch {
                program: &program,
                groups: [1, 1, 1],
                uniforms: HashMap::new(),
                storage: HashMap::new(),
                storage_in_place: HashMap::new(),
                textures: [(binding, (source.clone(), sampler))].into(),
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
                "{sampler:?}"
            );
        }
        renderer.destroy_program(program).unwrap();
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
            images: [(
                image,
                StorageImage {
                    target: output,
                    ..Default::default()
                },
            )]
            .into(),
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
    fn storage_image_view_addresses_layers_relative_to_its_base() {
        use crate::draw::ImageShape;

        let Ok(gpu) = Gpu::new() else {
            return;
        };
        if gpu.shader_objects().is_err() {
            return;
        }
        let mut renderer = Renderer::new(gpu).unwrap();
        for (target, height, shape) in [
            (1, 2, ImageShape::D2 { layers: 3 }),
            (2, 1, ImageShape::D1 { layers: 3 }),
        ] {
            renderer
                .shaped_target(target, 2, height, shape, vk::Format::R32_UINT, 2)
                .unwrap();
            for layer in 0..3 {
                renderer
                    .clear_color_level(
                        target,
                        1,
                        layer,
                        None,
                        vk::ClearColorValue {
                            uint32: [10 + layer; 4],
                        },
                    )
                    .unwrap();
            }
        }
        let cases = [
            (1, "2d_array", "vec2<i32>(0), 0", 3, 1, 2, [10_u32, 77, 12]),
            (1, "2d", "vec2<i32>(0)", 2, 2, 1, [10, 77, 77]),
            (2, "1d", "0", 0, 1, 1, [10, 77, 12]),
        ];
        for (target, shape, coords, texture_type, base_layer, layer_count, expected) in cases {
            let code = spirv(
                &format!(
                    "@group(0) @binding(0) var image: texture_storage_{shape}<r32uint, write>;
                    @compute @workgroup_size(1) fn main() {{
                        textureStore(image, {coords}, vec4<u32>(77u));
                    }}"
                ),
                naga::ShaderStage::Compute,
            );
            let program = renderer
                .create_program(vec![translated(
                    Stage::Compute,
                    code,
                    vec![Descriptor {
                        kind: DescriptorKind::Image as u32,
                        binding: 0,
                        count: 1,
                        texture_type,
                        ..Default::default()
                    }],
                )])
                .unwrap();
            let image = StorageImage {
                target,
                level: 1,
                base_layer,
                layer_count,
            };
            let dispatch = Dispatch {
                program: &program,
                groups: [1; 3],
                uniforms: HashMap::new(),
                storage: HashMap::new(),
                storage_in_place: HashMap::new(),
                textures: HashMap::new(),
                images: [(0, image)].into(),
                image_formats: HashMap::new(),
                indirect: None,
                page_table: Default::default(),
                pool: Default::default(),
            };
            renderer.dispatch(&dispatch).unwrap();
            renderer.destroy_program(program).unwrap();
            for (layer, expected) in expected.into_iter().enumerate() {
                assert_eq!(
                    renderer
                        .read_target_plane_bytes(target, 1, layer as u32, 4, [0, 0, 1, 1])
                        .unwrap(),
                    expected.to_le_bytes(),
                    "{shape} view, array layer {layer}",
                );
            }
        }

        let target = 3;
        let format = vk::Format::R32_UINT;
        renderer
            .shaped_target(target, 2, 1, ImageShape::D1 { layers: 1 }, format, 2)
            .unwrap();
        let image = StorageImage {
            target,
            level: 1,
            base_layer: 0,
            layer_count: 1,
        };
        let array_view = renderer.storage_image_view(image, 1, format).unwrap();
        let single_view = renderer.storage_image_view(image, 0, format).unwrap();
        assert_ne!(array_view, single_view);
    }

    #[test]
    fn typed_guest_storage_instructions_address_one_dimensional_array_views() {
        use crate::draw::ImageShape;
        use echora_shader::compute::{BoundStorageImage, Layout, SampleType, translate};
        use std::sync::Arc;

        let word = |fields: &[(u32, u128)]| {
            echora_shader::decode(
                fields
                    .iter()
                    .fold(7 << 12, |word, &(shift, value)| word | value << shift),
            )
        };
        let mov = |register, value| word(&[(0, 0x802), (16, register), (32, value), (72, 15)]);
        let surface = |store| {
            word(&[
                (0, if store { 0xb9b } else { 0xb97 }),
                (16, if store { 0 } else { 4 }),
                (24, 0),
                (32, if store { 4 } else { 0 }),
                (40, 0x18),
                (61, 2),
                (72, 1),
                (81, if store { 0 } else { 7 }),
            ])
        };
        let Ok(gpu) = Gpu::new() else {
            return;
        };
        if gpu.shader_objects().is_err() {
            return;
        }
        let mut renderer = Renderer::new(gpu).unwrap();
        for (index, (format, sample_type, texels)) in [
            (
                vk::Format::R32_SFLOAT,
                SampleType::Float,
                [0.25f32, 0.5, 0.75].map(f32::to_bits),
            ),
            (
                vk::Format::R32_UINT,
                SampleType::Unsigned,
                [5, 2_000_000_000, u32::MAX - 1],
            ),
            (
                vk::Format::R32_SINT,
                SampleType::Signed,
                [-5i32, -10, 7].map(|value| value as u32),
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let index = index as u64;
            // Read logical texel (0, layer 0) and store it at (1, layer 0).
            let stage = Arc::new(
                translate(
                    &[
                        mov(0, 0),
                        mov(1, 0),
                        surface(false),
                        mov(0, 1),
                        surface(true),
                        word(&[(0, 0x94d), (87, 7)]),
                    ],
                    &Layout {
                        storage_images: &[BoundStorageImage {
                            bank: 0,
                            offset: 0x60,
                            view: 1,
                            sample_type,
                        }],
                        ..Layout::workgroup([1; 3])
                    },
                )
                .unwrap(),
            );
            let binding = stage.descriptors[0].binding;
            assert_eq!(stage.descriptors[0].texture_type, 1);
            assert!(stage.spec_constants.is_empty());
            let program = renderer.create_program(vec![stage]).unwrap();
            for (target, layers, base_layer) in [(index * 2 + 1, 1, 0), (index * 2 + 2, 3, 1)] {
                renderer
                    .shaped_target(target, 4, 1, ImageShape::D1 { layers }, format, 2)
                    .unwrap();
                for layer in 0..layers {
                    let value = texels[layer as usize];
                    renderer
                        .clear_color_level(
                            target,
                            1,
                            layer,
                            None,
                            vk::ClearColorValue {
                                uint32: [value, 0, 0, 0],
                            },
                        )
                        .unwrap();
                    renderer
                        .clear_color_level(
                            target,
                            1,
                            layer,
                            Some([1, 0, 1, 1]),
                            vk::ClearColorValue { uint32: [0; 4] },
                        )
                        .unwrap();
                }
                let dispatch = Dispatch {
                    program: &program,
                    groups: [1; 3],
                    uniforms: HashMap::new(),
                    storage: HashMap::new(),
                    storage_in_place: HashMap::new(),
                    textures: HashMap::new(),
                    images: [(
                        binding,
                        StorageImage {
                            target,
                            level: 1,
                            base_layer,
                            layer_count: 1,
                        },
                    )]
                    .into(),
                    image_formats: HashMap::new(),
                    indirect: None,
                    page_table: Default::default(),
                    pool: Default::default(),
                };
                renderer.dispatch(&dispatch).unwrap();
                for layer in 0..layers {
                    let value = texels[layer as usize];
                    let expected = [value, if layer == base_layer { value } else { 0 }];
                    assert_eq!(
                        renderer
                            .read_target_plane_bytes(target, 1, layer, 4, [0, 0, 2, 1])
                            .unwrap(),
                        expected.map(u32::to_le_bytes).as_flattened(),
                    );
                }
            }
            renderer.destroy_program(program).unwrap();
        }
    }

    #[test]
    fn dispatches_and_clears_can_overwrite_the_same_image() {
        let Ok(gpu) = Gpu::new() else {
            return;
        };
        if gpu.shader_objects().is_err() {
            return;
        }
        let mut renderer = Renderer::new(gpu).unwrap();
        let code = spirv(
            "@group(0) @binding(0) var<uniform> value: vec4<u32>;
            @group(0) @binding(1) var image: texture_storage_2d<r32uint, write>;
            @compute @workgroup_size(1) fn main() { textureStore(image, vec2<i32>(0), value); }",
            naga::ShaderStage::Compute,
        );
        let program = renderer
            .create_program(vec![translated(
                Stage::Compute,
                code,
                vec![
                    Descriptor {
                        kind: DescriptorKind::UniformBuffer as u32,
                        binding: 0,
                        count: 1,
                        index: 3,
                        ..Default::default()
                    },
                    Descriptor {
                        kind: DescriptorKind::Image as u32,
                        binding: 1,
                        count: 1,
                        ..Default::default()
                    },
                ],
            )])
            .unwrap();
        let target = 1;
        renderer.target(target, 1, 1, vk::Format::R32_UINT).unwrap();
        let mut dispatch = Dispatch {
            program: &program,
            groups: [1; 3],
            uniforms: HashMap::from([(0, vec![7; 16])]),
            storage: HashMap::new(),
            storage_in_place: HashMap::new(),
            textures: HashMap::new(),
            images: HashMap::from([(
                1,
                StorageImage {
                    target,
                    ..Default::default()
                },
            )]),
            image_formats: HashMap::new(),
            indirect: None,
            page_table: Default::default(),
            pool: Default::default(),
        };
        renderer.dispatch_deferred(&dispatch).unwrap();
        dispatch.uniforms.get_mut(&0).unwrap().fill(9);
        renderer.dispatch_deferred(&dispatch).unwrap();
        assert_eq!(renderer.read_target_bytes(target, 4).unwrap(), [9; 4]);

        renderer.dispatch_deferred(&dispatch).unwrap();
        renderer
            .clear_color(target, None, vk::ClearColorValue { uint32: [15; 4] })
            .unwrap();
        assert_eq!(
            renderer.read_target_bytes(target, 4).unwrap(),
            15_u32.to_le_bytes()
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
        if renderer.supports_conditions() {
            let (predicate, _) = renderer.resident_buffer(1, 16).unwrap();
            renderer.fill_resident(1, 0, 16, 0).unwrap();
            let (groups, _) = renderer.resident_buffer(2, 16).unwrap();
            renderer.fill_resident(2, 0, 16, 1).unwrap();
            renderer.target(3, 1, 1, vk::Format::R32_UINT).unwrap();
            let clear = |renderer: &mut Renderer, value| {
                renderer
                    .clear_color(3, None, vk::ClearColorValue { uint32: [value; 4] })
                    .unwrap();
            };
            clear(&mut renderer, 7);
            renderer.set_condition(Some((predicate, 0, false)));
            let mut tokens = Vec::new();
            for indirect in [None, Some((groups, 0))] {
                clear(&mut renderer, 9); // Starts a false rendering scope.
                dispatch.indirect = indirect;
                let pending = renderer.dispatch_deferred(&dispatch).unwrap();
                assert_eq!(pending.len(), 1);
                tokens.push(pending[0].1);
            }
            clear(&mut renderer, 11);
            assert_eq!(
                renderer.read_target_bytes(3, 4).unwrap(),
                7_u32.to_le_bytes()
            );
            let results = renderer.take_reads(&tokens);
            for token in tokens {
                assert_eq!(
                    results[&token],
                    [0xaa, 0xaa, 0xaa, 0xaa, 0x78, 0x56, 0x34, 0x12]
                );
            }
        }
        renderer.destroy_program(program).unwrap();
    }

    #[test]
    fn combined_depth_stencil_arrays_sample_every_depth_layer() {
        use crate::draw::ImageShape;

        let Ok(gpu) = Gpu::new() else {
            return;
        };
        if gpu.texture_pool_layout.is_none() || gpu.shader_objects().is_err() {
            return;
        }
        let mut renderer = Renderer::new(gpu).unwrap();
        let key = 1;
        renderer
            .shaped_target(
                key,
                4,
                4,
                ImageShape::D2 { layers: 2 },
                vk::Format::D32_SFLOAT_S8_UINT,
                1,
            )
            .unwrap();
        for (layer, depth) in [(0, 0.25), (1, 0.75)] {
            renderer
                .clear_depth_stencil(key, layer, None, Some(depth), Some(0))
                .unwrap();
        }
        let code = spirv(
            "enable wgpu_binding_array;
             @group(0) @binding(0) var<storage, read_write> output: array<u32>;
             @group(1) @binding(4) var textures: binding_array<texture_depth_2d_array, 8>;
             @compute @workgroup_size(1) fn main() {
                 output[0] = bitcast<u32>(textureLoad(textures[1], vec2<i32>(1, 1), 0, 0));
                 output[1] = bitcast<u32>(textureLoad(textures[1], vec2<i32>(1, 1), 1, 0));
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
            .texture_pool_bindings = 1 << echora_shader_abi::texture_pool::SAMPLED_2D_ARRAY;
        let program = renderer.create_program(vec![stage]).unwrap();
        let dispatch = Dispatch {
            program: &program,
            groups: [1, 1, 1],
            uniforms: HashMap::new(),
            storage: HashMap::from([(0, vec![0; 16])]),
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
        let results = renderer.dispatch(&dispatch).unwrap();
        renderer.destroy_program(program).unwrap();
        let expected: Vec<_> = [0.25_f32, 0.75]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        assert_eq!(results[&0][..8], expected);
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
