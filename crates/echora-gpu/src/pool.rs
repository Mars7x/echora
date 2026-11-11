//! The texture pool (`echora_shader_abi::TEXTURE_POOL_SET`): one descriptor
//! set, shared by every program that reads it, whose arrays hold the guest
//! texture and sampler pool entries bindless handles name.
//!
//! Entries are written as draws and dispatches name them (`PoolEntry`),
//! not all at once: a guest pool holds every texture the game registered,
//! most never sampled through a computed handle. Every entry starts as a
//! null descriptor (samplers as a default sampler), which a shader reading
//! an entry nothing wrote gets.

use std::collections::{HashMap, HashSet};

use ash::vk;
use ash::vk::Handle;
use echora_shader_abi::texture_pool as binding;

use crate::draw::{ImageShape, Sampler, TextureSource};
use crate::renderer::{HostBuffer, Renderer, integer_format};
use crate::vulkan::{SAMPLER_POOL_CAPACITY, TEXTURE_POOL_CAPACITY};

/// One texture pool entry a draw or dispatch may read: the texture at
/// `texture` (handle bits 0..20) and the sampler at `sampler` (bits
/// 20..32), as the handle that names both resolves.
#[derive(Clone)]
pub struct PoolEntry {
    pub texture: u32,
    pub sampler: u32,
    pub source: TextureSource,
    pub sampler_state: Sampler,
}

pub(crate) struct TexturePool {
    descriptor_pool: vk::DescriptorPool,
    pub(crate) set: vk::DescriptorSet,
    /// `texture_pool::FLAGS`: one word per texture index, bit 0 set when
    /// the entry is integer (read through the uint arrays).
    flags: HostBuffer,
    /// Indices whose flag is set.
    integer: HashSet<u32>,
    /// What each written (binding, index) holds: a view's or sampler's
    /// raw handle.
    written: HashMap<(u32, u32), u64>,
    /// Recorded or submitted work since the last flush bound the set: an
    /// entry it may read is then not rewritten before a flush.
    pub(crate) in_use: bool,
}

/// The pool arrays a texture of `shape` is read through (the recompiler's
/// texture types: 0 1D, 1 1D array, 2 2D, 3 2D array, 4 3D, 5 cube, 6 cube
/// array), with the binding of each; integer textures through the uint
/// arrays only (sampling them as float is undefined).
fn arrays(shape: ImageShape, integer: bool) -> Vec<(u32, u32)> {
    let types: &[u32] = match shape {
        ImageShape::D1 { .. } => &[0, 1],
        ImageShape::D2 { layers } if layers > 1 => &[3, 2],
        ImageShape::D2 { .. } => &[2, 3],
        ImageShape::D3 { .. } => &[4],
        ImageShape::Cube { .. } => &[5, 6, 3],
    };
    types
        .iter()
        .filter_map(|&kind| {
            let array = if integer {
                match kind {
                    2 => binding::UINT_2D,
                    3 => binding::UINT_2D_ARRAY,
                    4 => binding::UINT_3D,
                    _ => return None,
                }
            } else {
                binding::SAMPLED_1D + kind
            };
            Some((kind, array))
        })
        .collect()
}

impl Renderer {
    /// The pool set, created on first use with every entry null.
    fn texture_pool(&mut self) -> Result<&mut TexturePool, String> {
        if self.texture_pool.is_none() {
            let layout = self
                .gpu
                .texture_pool_layout
                .ok_or("the device cannot hold the texture pool")?;
            let sizes = [
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::SAMPLER,
                    descriptor_count: SAMPLER_POOL_CAPACITY,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: TEXTURE_POOL_CAPACITY * 10,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: TEXTURE_POOL_CAPACITY * 2,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
                    descriptor_count: TEXTURE_POOL_CAPACITY,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 1,
                },
            ];
            let info = vk::DescriptorPoolCreateInfo::default()
                .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
                .max_sets(1)
                .pool_sizes(&sizes);
            let device = &self.gpu.device;
            // SAFETY: valid create info; destroyed with the renderer.
            let descriptor_pool = unsafe { device.create_descriptor_pool(&info, None) }
                .map_err(|error| format!("vkCreateDescriptorPool (texture pool): {error}"))?;
            let layouts = [layout];
            let allocate = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&layouts);
            // SAFETY: the pool and layout are live.
            let set = match unsafe { device.allocate_descriptor_sets(&allocate) } {
                Ok(sets) => sets[0],
                Err(error) => {
                    // SAFETY: nothing was allocated from it.
                    unsafe { device.destroy_descriptor_pool(descriptor_pool, None) };
                    return Err(format!("vkAllocateDescriptorSets (texture pool): {error}"));
                }
            };
            let flags_size = u64::from(TEXTURE_POOL_CAPACITY) * 4;
            let flags = self.host_buffer(flags_size, vk::BufferUsageFlags::STORAGE_BUFFER)?;
            flags.write(0, &vec![0; flags_size as usize]);
            let flags_info = [vk::DescriptorBufferInfo {
                buffer: flags.buffer,
                offset: 0,
                range: flags_size,
            }];
            let sampler = self.sampler(&Sampler::default())?;
            let samplers = vec![
                vk::DescriptorImageInfo {
                    sampler,
                    ..Default::default()
                };
                SAMPLER_POOL_CAPACITY as usize
            ];
            let images = vec![vk::DescriptorImageInfo::default(); TEXTURE_POOL_CAPACITY as usize];
            let buffers = vec![vk::BufferView::null(); TEXTURE_POOL_CAPACITY as usize];
            let writes: Vec<_> = (0..binding::COUNT)
                .map(|array| {
                    let write = vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(array);
                    match array {
                        binding::SAMPLERS => write
                            .descriptor_type(vk::DescriptorType::SAMPLER)
                            .image_info(&samplers),
                        binding::SAMPLED_BUFFER => write
                            .descriptor_type(vk::DescriptorType::UNIFORM_TEXEL_BUFFER)
                            .texel_buffer_view(&buffers),
                        binding::STORAGE_2D | binding::STORAGE_3D => write
                            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                            .image_info(&images),
                        binding::FLAGS => write
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&flags_info),
                        _ => write
                            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                            .image_info(&images),
                    }
                })
                .collect();
            // SAFETY: the set is live and unused; null descriptors are
            // enabled with the layout (`Gpu::new`).
            unsafe { self.gpu.device.update_descriptor_sets(&writes, &[]) };
            self.texture_pool = Some(TexturePool {
                descriptor_pool,
                set,
                flags,
                integer: HashSet::new(),
                written: HashMap::new(),
                in_use: false,
            });
        }
        Ok(self.texture_pool.as_mut().unwrap())
    }

    /// Write `entries` into the pool set and return it, for a program
    /// that reads the pool (bound as set 1). An entry that changes while
    /// recorded or submitted work may read it is written after a flush.
    pub(crate) fn bind_pool(&mut self, entries: &[PoolEntry]) -> Result<vk::DescriptorSet, String> {
        self.texture_pool()?;
        // (binding, index, view or sampler) for every entry, resolved
        // before the set is touched (views can upload).
        let mut wanted: Vec<(u32, u32, u64)> = Vec::new();
        let mut kinds: Vec<(u32, bool)> = Vec::new();
        let mut targets = Vec::new();
        for entry in entries {
            if entry.texture >= TEXTURE_POOL_CAPACITY || entry.sampler >= SAMPLER_POOL_CAPACITY {
                self.stat("stat: texture pool entries past capacity", 1);
                continue;
            }
            let sampler = self.sampler(&entry.sampler_state)?;
            wanted.push((binding::SAMPLERS, entry.sampler, sampler.as_raw()));
            let (shape, format) = self.source_shape(&entry.source)?;
            let integer = integer_format(format);
            if integer {
                self.stat("stat: texture pool integer entries", 1);
            }
            kinds.push((entry.texture, integer));
            for (kind, array) in arrays(shape, integer) {
                // A type the texture has no view of stays null.
                let Ok(view) =
                    self.texture_view_swizzled(&entry.source, kind, &entry.sampler_state)
                else {
                    continue;
                };
                wanted.push((array, entry.texture, view.as_raw()));
            }
            if let TextureSource::Target(key) | TextureSource::TargetView { key, .. } = entry.source
                && let Some(target) = self.existing_target(key)
            {
                targets.push(target.image);
            }
        }
        let pool = self.texture_pool.as_ref().unwrap();
        let changed: Vec<(u32, u32, u64)> = wanted
            .into_iter()
            .collect::<HashSet<_>>()
            .into_iter()
            .filter(|(array, index, handle)| pool.written.get(&(*array, *index)) != Some(handle))
            .collect();
        let flips: Vec<(u32, bool)> = kinds
            .into_iter()
            .filter(|(index, integer)| pool.integer.contains(index) != *integer)
            .collect();
        let rewrites = !flips.is_empty()
            || changed
                .iter()
                .any(|(array, index, _)| pool.written.contains_key(&(*array, *index)));
        if rewrites && pool.in_use {
            // ECHORA_POOL_REWRITE_LOG (debugging): the first rewrites of
            // entries work in flight may read, with what each entry asks.
            if std::env::var_os("ECHORA_POOL_REWRITE_LOG").is_some()
                && self
                    .stats
                    .get("stat: texture pool flushes for rewritten entries")
                    .copied()
                    .unwrap_or(0)
                    < 300
            {
                for (array, index, handle) in &changed {
                    if let Some(old) = pool.written.get(&(*array, *index)) {
                        let asked: Vec<String> = entries
                            .iter()
                            .filter(|entry| {
                                if *array == binding::SAMPLERS {
                                    entry.sampler == *index
                                } else {
                                    entry.texture == *index
                                }
                            })
                            .map(|entry| {
                                let source = match &entry.source {
                                    TextureSource::Upload {
                                        key,
                                        generation,
                                        width,
                                        height,
                                        shape,
                                        format,
                                        ..
                                    } => format!(
                                        "upload {key:x} gen {generation} {width}x{height} {shape:?} {format:?}"
                                    ),
                                    other => format!("{other:?}"),
                                };
                                format!(
                                    "texture {} sampler {} {source} components {:?} format {:?}",
                                    entry.texture,
                                    entry.sampler,
                                    entry.sampler_state.components,
                                    entry.sampler_state.format
                                )
                            })
                            .collect();
                        eprintln!(
                            "pool rewrite array {array} index {index}: {old:#x} -> {handle:#x} for {asked:?}"
                        );
                    }
                }
                for (index, integer) in &flips {
                    eprintln!("pool rewrite flag index {index}: integer -> {integer}");
                }
            }
            self.stat("stat: texture pool flushes for rewritten entries", 1);
            self.flush()?;
        }
        // Targets read through the pool were maybe rendered just before.
        self.order_after_rendering(&targets)?;
        let pool = self.texture_pool.as_mut().unwrap();
        let set = pool.set;
        for (index, integer) in flips {
            pool.flags
                .write(index as usize * 4, &u32::from(integer).to_le_bytes());
            if integer {
                pool.integer.insert(index);
            } else {
                pool.integer.remove(&index);
            }
        }
        let infos: Vec<[vk::DescriptorImageInfo; 1]> = changed
            .iter()
            .map(|&(array, _, handle)| {
                [if array == binding::SAMPLERS {
                    vk::DescriptorImageInfo {
                        sampler: vk::Sampler::from_raw(handle),
                        ..Default::default()
                    }
                } else {
                    vk::DescriptorImageInfo {
                        image_view: vk::ImageView::from_raw(handle),
                        image_layout: vk::ImageLayout::GENERAL,
                        ..Default::default()
                    }
                }]
            })
            .collect();
        let writes: Vec<_> = changed
            .iter()
            .zip(&infos)
            .map(|(&(array, index, _), info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(array)
                    .dst_array_element(index)
                    .descriptor_type(if array == binding::SAMPLERS {
                        vk::DescriptorType::SAMPLER
                    } else {
                        vk::DescriptorType::SAMPLED_IMAGE
                    })
                    .image_info(info)
            })
            .collect();
        if !writes.is_empty() {
            // SAFETY: the set and the views and samplers are live; work
            // that may read a rewritten entry has completed (above).
            unsafe { self.gpu.device.update_descriptor_sets(&writes, &[]) };
        }
        for (array, index, handle) in changed {
            pool.written.insert((array, index), handle);
        }
        pool.in_use = true;
        Ok(set)
    }

    /// Entries holding any of `views` (an image about to be destroyed,
    /// after its work completed) become null again.
    pub(crate) fn forget_pool_views(&mut self, views: &[vk::ImageView]) {
        let Some(pool) = self.texture_pool.as_mut() else {
            return;
        };
        let raw: HashSet<u64> = views.iter().map(|view| view.as_raw()).collect();
        let stale: Vec<(u32, u32)> = pool
            .written
            .iter()
            .filter(|(_, handle)| raw.contains(handle))
            .map(|(key, _)| *key)
            .collect();
        if stale.is_empty() {
            return;
        }
        let null = [vk::DescriptorImageInfo::default()];
        let writes: Vec<_> = stale
            .iter()
            .map(|&(array, index)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(pool.set)
                    .dst_binding(array)
                    .dst_array_element(index)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&null)
            })
            .collect();
        // SAFETY: the set is live; null descriptors are enabled.
        unsafe { self.gpu.device.update_descriptor_sets(&writes, &[]) };
        for key in stale {
            pool.written.remove(&key);
        }
    }

    /// A source's shape and format.
    fn source_shape(&self, source: &TextureSource) -> Result<(ImageShape, vk::Format), String> {
        match source {
            TextureSource::Upload { shape, format, .. } => Ok((*shape, *format)),
            TextureSource::Target(key) | TextureSource::TargetView { key, .. } => self
                .existing_target(*key)
                .map(|target| (target.shape, target.format))
                .ok_or_else(|| format!("no render target {key:x}")),
        }
    }

    /// Count `amount` under `name` (`take_stats`).
    pub(crate) fn stat(&mut self, name: &'static str, amount: u64) {
        *self.stats.entry(name).or_default() += amount;
    }

    /// The counts noted since the last call.
    pub fn take_stats(&mut self) -> HashMap<&'static str, u64> {
        std::mem::take(&mut self.stats)
    }

    pub(crate) fn destroy_texture_pool(&mut self) {
        if let Some(pool) = self.texture_pool.take() {
            pool.flags.destroy(&self.gpu.device);
            // SAFETY: all work has completed.
            unsafe {
                self.gpu
                    .device
                    .destroy_descriptor_pool(pool.descriptor_pool, None)
            };
        }
    }
}
