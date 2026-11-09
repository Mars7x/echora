//! Compute dispatches: one translated compute stage, its uniform buffers and
//! its storage buffers. Storage buffers travel through the host-visible ring
//! and are read back after the dispatch, so the caller can store them into
//! guest memory.

use std::collections::HashMap;

use ash::vk;

use crate::draw::{Program, Sampler, TextureSource};
use crate::renderer::Renderer;
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
                        let length = dispatch.storage.get(&binding).map_or(16, Vec::len) as u64;
                        bytes += length.max(16) + 256;
                    }
                    descriptors[2] += 1;
                }
                _ => {
                    let length = dispatch.uniforms.get(&binding).map_or(16, Vec::len) as u64;
                    bytes += length.max(16) + 256;
                    descriptors[0] += 1;
                }
            }
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
                let declared = crate::renderer::declared_image_format(descriptor.flags);
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
                let image_view = self.texture_view_swizzled(
                    source,
                    descriptor.texture_type,
                    sampler.components,
                    sampler.format,
                )?;
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
            let (buffer, offset) = self.ring_upload(bytes, 256)?;
            let range = (bytes.len() as u64).max(16).next_multiple_of(16);
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
        let commands = self.commands()?;
        let before = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::HOST_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::UNIFORM_READ);
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
        // SAFETY: recording outside rendering; the program and set are live.
        unsafe {
            let device = &self.gpu.device;
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::COMPUTE_SHADER,
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
            device.cmd_bind_descriptor_sets(
                commands,
                vk::PipelineBindPoint::COMPUTE,
                program.pipeline_layout,
                0,
                &[set],
                &[],
            );
            let [x, y, z] = dispatch.groups;
            device.cmd_dispatch(commands, x, y, z);
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
    fn a_dispatch_stores_a_uniform_into_a_storage_buffer() {
        let Ok(gpu) = Gpu::new() else {
            return; // No device (CI).
        };
        let mut renderer = Renderer::new(gpu).unwrap();
        let code = spirv(
            "struct Words { data: array<u32> }
            @group(0) @binding(0) var<uniform> value: vec4<u32>;
            @group(0) @binding(1) var<storage, read_write> words: Words;
            @compute @workgroup_size(1) fn main() { words.data[1] = value.x; }",
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
        let dispatch = Dispatch {
            program: &program,
            groups: [1, 1, 1],
            uniforms: HashMap::from([(0, value)]),
            storage: HashMap::from([(1, vec![0xaa; 8])]),
            storage_in_place: HashMap::new(),
            textures: HashMap::new(),
            images: HashMap::new(),
        };
        let results = renderer.dispatch(&dispatch).unwrap();
        assert_eq!(
            results[&1],
            [0xaa, 0xaa, 0xaa, 0xaa, 0x78, 0x56, 0x34, 0x12]
        );
    }
}
