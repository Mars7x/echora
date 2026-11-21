//! The texture pool (`echora_shader_abi::TEXTURE_POOL_SET`): descriptor
//! arrays holding the guest texture and sampler entries that bindless
//! handles name. Each draw or dispatch binds the current version.
//!
//! Entries are written as draws and dispatches name them (`PoolEntry`),
//! not all at once: a guest pool holds every texture the game registered,
//! most never sampled through a computed handle. Every entry starts as a
//! null descriptor (samplers as a default sampler), which a shader reading
//! an entry nothing wrote gets.

use std::collections::HashSet;

use ash::vk;
use ash::vk::Handle;
use echora_shader_abi::texture_pool as binding;

use crate::draw::{ImageShape, Sampler, TextureSource};
use crate::fast_map::FastMap;
use crate::renderer::{HostBuffer, Renderer, sampled_texture_type};
use crate::shader::SampleType;
use crate::vulkan::{SAMPLER_POOL_CAPACITY, TEXTURE_POOL_CAPACITY};

/// One texture pool entry a draw or dispatch may read: the texture at
/// `texture` (handle bits 0..20) and the sampler at `sampler` (bits
/// 20..32), as the handle that names both resolves.
#[derive(Clone, Debug)]
pub struct PoolEntry {
    pub texture: u32,
    pub sampler: u32,
    pub source: TextureSource,
    pub sampler_state: Sampler,
}

impl PoolEntry {
    /// Whether the handle still names this texture revision and sampler.
    pub fn matches_binding(&self, source: &TextureSource, sampler: &Sampler) -> bool {
        self.sampler_state == *sampler && self.source.same_as(source)
    }
}

/// A pool entry of an upload as `bind_pool` resolved it: the entry, its
/// (binding, index, sampler or view) descriptors and its numeric flags.
/// Valid while no view it names is destroyed (`forget_pool_views`).
pub(crate) struct ResolvedEntry {
    source: TextureSource,
    sampler_state: Sampler,
    wanted: Vec<(u32, u32, u64)>,
    numeric: u32,
}

pub(crate) struct TexturePool {
    descriptor_pool: vk::DescriptorPool,
    pub(crate) set: vk::DescriptorSet,
    /// `texture_pool::FLAGS`: one numeric class word per texture index.
    flags: HostBuffer,
    numeric: FastMap<u32, u32>,
    default_samplers: [vk::Sampler; 2],
    /// What each written (binding, index) holds: a view's or sampler's
    /// raw handle.
    written: FastMap<(u32, u32), u64>,
    /// The slots containing each image view, for retirement by handle.
    view_slots: FastMap<u64, Vec<(u32, u32)>>,
    /// Last batch that bound this set. Its draws may access any descriptor
    /// in the pool, so replacing a descriptor here requires completion.
    last_use: u64,
}

impl TexturePool {
    /// Write the changed sampled views or samplers. A zero view clears
    /// a descriptor whose image was discarded.
    fn write_entries(&mut self, device: &ash::Device, changes: &[(u32, u32, u64)]) {
        if changes.is_empty() {
            return;
        }
        let infos: Vec<[vk::DescriptorImageInfo; 1]> = changes
            .iter()
            .map(|&(array, index, handle)| {
                [if array == binding::SAMPLERS {
                    let sampler = if handle == 0 {
                        self.default_samplers[usize::from(index >= SAMPLER_POOL_CAPACITY)]
                    } else {
                        vk::Sampler::from_raw(handle)
                    };
                    vk::DescriptorImageInfo::default().sampler(sampler)
                } else if handle != 0 {
                    vk::DescriptorImageInfo::default()
                        .image_view(vk::ImageView::from_raw(handle))
                        .image_layout(vk::ImageLayout::GENERAL)
                } else {
                    vk::DescriptorImageInfo::default()
                }]
            })
            .collect();
        let writes: Vec<_> = changes
            .iter()
            .zip(&infos)
            .map(|(&(array, index, _), info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(self.set)
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
        // SAFETY: the set and its new descriptors are live. Replaced
        // entries can only have been used by work that completed.
        unsafe { device.update_descriptor_sets(&writes, &[]) };
        for &(array, index, handle) in changes {
            let slot = (array, index);
            let previous = if handle == 0 {
                self.written.remove(&slot)
            } else {
                self.written.insert(slot, handle)
            };
            if array == binding::SAMPLERS || previous == Some(handle) {
                continue;
            }
            if let Some(previous) = previous
                && let std::collections::hash_map::Entry::Occupied(mut entry) =
                    self.view_slots.entry(previous)
            {
                let slots = entry.get_mut();
                slots.retain(|&held| held != slot);
                if slots.is_empty() {
                    entry.remove();
                }
            }
            if handle != 0 {
                self.view_slots.entry(handle).or_default().push(slot);
            }
        }
    }
}

/// The pool arrays a texture of `shape` is read through (the recompiler's
/// texture types: 0 1D, 1 1D array, 2 2D, 3 2D array, 4 3D, 5 cube, 6 cube
/// array), with the binding of each; integer textures through the integer
/// arrays only (sampling them as float is undefined).
fn arrays(shape: ImageShape, integer: bool) -> impl Iterator<Item = (u32, u32)> {
    let types: &[u32] = match shape {
        ImageShape::D1 { .. } => &[0, 1],
        ImageShape::D2 { layers } if layers > 1 => &[3, 2],
        ImageShape::D2 { .. } => &[2, 3],
        ImageShape::D3 { .. } => &[4],
        ImageShape::Cube { .. } => &[5, 6, 3],
    };
    types.iter().filter_map(move |&kind| {
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
}

impl Renderer {
    /// A new pool set, with every entry null or the default sampler.
    fn create_texture_pool(&mut self) -> Result<TexturePool, String> {
        let layout = self
            .gpu
            .texture_pool_layout
            .ok_or("the device cannot hold the texture pool")?;
        let default_samplers = [
            self.sampler(&Sampler::default(), false)?,
            self.sampler(&Sampler::default(), true)?,
        ];
        let sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLER,
                descriptor_count: binding::SAMPLER_DESCRIPTORS,
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
        let flags = match self.host_buffer(flags_size, vk::BufferUsageFlags::STORAGE_BUFFER) {
            Ok(flags) => flags,
            Err(error) => {
                // SAFETY: the pool's set has never been used.
                unsafe {
                    self.gpu
                        .device
                        .destroy_descriptor_pool(descriptor_pool, None)
                };
                return Err(error);
            }
        };
        flags.write(0, &vec![0; flags_size as usize]);
        let flags_info = [vk::DescriptorBufferInfo {
            buffer: flags.buffer,
            offset: 0,
            range: flags_size,
        }];
        let samplers: Vec<_> = default_samplers
            .iter()
            .flat_map(|&sampler| {
                std::iter::repeat_n(
                    vk::DescriptorImageInfo::default().sampler(sampler),
                    SAMPLER_POOL_CAPACITY as usize,
                )
            })
            .collect();
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
        Ok(TexturePool {
            descriptor_pool,
            set,
            flags,
            numeric: FastMap::default(),
            default_samplers,
            written: FastMap::default(),
            view_slots: FastMap::default(),
            last_use: 0,
        })
    }

    /// Copy current bindings into an idle spare and use it for the next
    /// draw, so current work keeps the descriptors it already bound.
    fn rotate_texture_pool(&mut self) -> bool {
        if !self.tried_spare_texture_pool && self.gpu.spare_texture_pool {
            self.tried_spare_texture_pool = true;
            self.spare_texture_pool = self.create_texture_pool().ok();
        }
        let Some(spare) = self.spare_texture_pool.as_mut() else {
            return false;
        };
        if spare.last_use > self.completed_serial {
            return false;
        }
        let active = self.texture_pool.as_ref().unwrap();
        let mut changes = Vec::new();
        for (&(array, index), &handle) in &active.written {
            if spare.written.get(&(array, index)) != Some(&handle) {
                changes.push((array, index, handle));
            }
        }
        for &(array, index) in spare.written.keys() {
            if !active.written.contains_key(&(array, index)) {
                changes.push((array, index, 0));
            }
        }
        spare.write_entries(&self.gpu.device, &changes);
        for (&index, &numeric) in &active.numeric {
            if spare.numeric.get(&index) != Some(&numeric) {
                spare
                    .flags
                    .write(index as usize * 4, &numeric.to_le_bytes());
            }
        }
        for &index in spare.numeric.keys() {
            if !active.numeric.contains_key(&index) {
                spare.flags.write(index as usize * 4, &0u32.to_le_bytes());
            }
        }
        spare.numeric.clone_from(&active.numeric);
        std::mem::swap(&mut self.texture_pool, &mut self.spare_texture_pool);
        self.stat("stat: texture pool rotations", 1);
        self.stat(
            "stat: texture pool descriptors synchronized",
            changes.len() as u64,
        );
        true
    }

    /// Write `entries` and return the texture pool's current set. Keep
    /// old descriptor contents live for work that already bound them.
    pub(crate) fn bind_pool(
        &mut self,
        entries: &std::sync::Arc<[PoolEntry]>,
    ) -> Result<vk::DescriptorSet, String> {
        if self.texture_pool.is_none() {
            self.texture_pool = Some(self.create_texture_pool()?);
        }
        // The same entries as the last call, with no target made again
        // since (a view of one would change): the set already holds them.
        if let Some((last, changes, targets)) = &self.pool_cache
            && *changes == self.target_changes
            // The same shared list (the runtime keeps one while nothing it
            // was made from changed), or equal entries.
            && (std::sync::Arc::ptr_eq(last, entries)
                || last.len() == entries.len()
                    && last.iter().zip(entries.iter()).all(|(last, entry)| {
                        (last.texture, last.sampler) == (entry.texture, entry.sampler)
                            && last.matches_binding(&entry.source, &entry.sampler_state)
                    }))
        {
            let pending = self.rendering_pending(targets.iter().copied());
            self.stat("stat: texture pool sets unchanged", 1);
            if pending {
                self.memory_barrier()?;
            }
            return Ok(self.texture_pool.as_ref().unwrap().set);
        }
        // (binding, index, view or sampler) for every entry, resolved
        // before the set is touched (views can upload).
        let mut wanted: Vec<(u32, u32, u64)> = Vec::new();
        let mut kinds: Vec<(u32, u32)> = Vec::new();
        let mut targets = Vec::new();
        for entry in entries.iter() {
            if entry.texture >= TEXTURE_POOL_CAPACITY || entry.sampler >= SAMPLER_POOL_CAPACITY {
                self.stat("stat: texture pool entries past capacity", 1);
                continue;
            }
            // An upload resolved before with the same source and sampler
            // names the same sampler and views (render targets resolve
            // every time: sampling them transitions and refreshes copies).
            let upload = matches!(entry.source, TextureSource::Upload { .. });
            if upload
                && let Some(memo) = self.pool_entry_memo.get(&(entry.texture, entry.sampler))
                && memo.sampler_state == entry.sampler_state
                && memo.source.same_as(&entry.source)
            {
                wanted.extend_from_slice(&memo.wanted);
                kinds.push((entry.texture, memo.numeric));
                self.stat("stat: texture pool entries reused", 1);
                continue;
            }
            let first = wanted.len();
            let sampler = self.sampler(&entry.sampler_state, false)?;
            wanted.push((binding::SAMPLERS, entry.sampler, sampler.as_raw()));
            let (shape, format) = self.source_shape(&entry.source)?;
            let numeric = match sampled_texture_type(format) {
                SampleType::Float => 0,
                SampleType::Unsigned => binding::FLAG_INTEGER,
                SampleType::Signed => binding::FLAG_INTEGER | binding::FLAG_SIGNED,
            };
            let integer = numeric != 0;
            if integer {
                let sampler = self.sampler(&entry.sampler_state, true)?;
                wanted.push((
                    binding::SAMPLERS,
                    entry.sampler + SAMPLER_POOL_CAPACITY,
                    sampler.as_raw(),
                ));
                self.stat("stat: texture pool integer entries", 1);
            }
            kinds.push((entry.texture, numeric));
            for (kind, array) in arrays(shape, integer) {
                // A type the texture has no view of stays null.
                let Ok(view) =
                    self.texture_view_swizzled(&entry.source, kind, &entry.sampler_state)
                else {
                    continue;
                };
                wanted.push((array, entry.texture, view.as_raw()));
            }
            if upload {
                let resolved = ResolvedEntry {
                    source: entry.source.clone(),
                    sampler_state: entry.sampler_state,
                    wanted: wanted[first..].to_vec(),
                    numeric,
                };
                self.pool_entry_memo
                    .insert((entry.texture, entry.sampler), resolved);
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
            .filter(|(array, index, handle)| pool.written.get(&(*array, *index)) != Some(handle))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        let flips: Vec<(u32, u32)> = kinds
            .into_iter()
            .filter(|(index, numeric)| {
                pool.numeric.get(index).copied().unwrap_or_default() != *numeric
            })
            .collect();
        let rewrites = !flips.is_empty()
            || changed
                .iter()
                .any(|(array, index, _)| pool.written.contains_key(&(*array, *index)));
        let last_use = pool.last_use;
        if rewrites && last_use > self.completed_serial {
            // ECHORA_POOL_REWRITE_LOG (debugging): the first rewrites of
            // entries work in flight may read, with what each entry asks.
            if std::env::var_os("ECHORA_POOL_REWRITE_LOG").is_some()
                && self
                    .stats
                    .get("stat: texture pool flushes for rewritten entries")
                    .copied()
                    .unwrap_or(0)
                    + self
                        .stats
                        .get("stat: texture pool rotations")
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
                for (index, numeric) in &flips {
                    eprintln!("pool rewrite flag index {index}: numeric -> {numeric}");
                }
            }
            if !self.rotate_texture_pool() {
                self.stat("stat: texture pool flushes for rewritten entries", 1);
                let waiting = std::time::Instant::now();
                self.wait_for(last_use)?;
                self.stat(
                    "stat: texture pool rewrite wait ns",
                    waiting.elapsed().as_nanos() as u64,
                );
            }
        }
        // Targets read through the pool were maybe rendered just before.
        self.order_after_rendering(&targets)?;
        let pool = self.texture_pool.as_mut().unwrap();
        let set = pool.set;
        for (index, numeric) in flips {
            pool.flags.write(index as usize * 4, &numeric.to_le_bytes());
            if numeric != 0 {
                pool.numeric.insert(index, numeric);
            } else {
                pool.numeric.remove(&index);
            }
        }
        pool.write_entries(&self.gpu.device, &changed);
        self.pool_cache = Some((entries.clone(), self.target_changes, targets));
        Ok(set)
    }

    /// Called when recording a draw or dispatch that binds the pool.
    pub(crate) fn record_texture_pool_use(&mut self) {
        let serial = self.recorded_serial();
        self.texture_pool.as_mut().unwrap().last_use = serial;
    }

    /// Entries holding any of `views` (an image about to be destroyed,
    /// after its work completed) become null again.
    pub(crate) fn forget_pool_views(&mut self, views: &[vk::ImageView]) {
        self.pool_cache = None;
        self.pool_entry_memo.clear();
        for pool in self
            .texture_pool
            .iter_mut()
            .chain(self.spare_texture_pool.iter_mut())
        {
            let mut stale: Vec<_> = views
                .iter()
                .filter_map(|view| pool.view_slots.get(&view.as_raw()))
                .flatten()
                .map(|&(array, index)| (array, index, 0))
                .collect();
            stale.sort_unstable();
            stale.dedup();
            pool.write_entries(&self.gpu.device, &stale);
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
    pub fn take_stats(&mut self) -> crate::fast_map::FastMap<&'static str, u64> {
        // Named once per source line, so each name is kept for the run.
        static NAMES: std::sync::Mutex<
            Vec<(&'static std::panic::Location<'static>, &'static str)>,
        > = std::sync::Mutex::new(Vec::new());
        let mut names = NAMES.lock().unwrap();
        for (location, amount) in std::mem::take(&mut self.pass_ends) {
            let name = match names.iter().find(|(known, _)| *known == location) {
                Some((_, name)) => name,
                None => {
                    let file = location.file().rsplit('/').next().unwrap_or("");
                    let name: &'static str = Box::leak(
                        format!("stat: rendering passes ended at {file}:{}", location.line())
                            .into_boxed_str(),
                    );
                    names.push((location, name));
                    name
                }
            };
            *self.stats.entry(name).or_default() += amount;
        }
        std::mem::take(&mut self.stats)
    }

    pub(crate) fn destroy_texture_pool(&mut self) {
        for pool in [self.texture_pool.take(), self.spare_texture_pool.take()]
            .into_iter()
            .flatten()
        {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bindless_integer_and_float_views_can_share_a_guest_sampler() {
        use crate::compute::{Dispatch, StorageImage};
        use crate::shader::{self, DescriptorKind, Parameters};
        use crate::vulkan::Gpu;
        use std::sync::Arc;

        let Ok(gpu) = Gpu::new() else {
            return;
        };
        if gpu.texture_pool_layout.is_none() {
            return;
        }
        let mut renderer = Renderer::new(gpu).unwrap();
        if !renderer.supports_storage(vk::Format::R32_UINT) {
            return;
        }
        let word =
            |fields: &[(u32, u128)]| fields.iter().fold(7 << 12, |w, &(at, val)| w | val << at);
        let mov = |register, value| word(&[(0, 0x802), (16, register), (32, value), (72, 15)]);
        let sampler_idx = 1u32;
        let tex_idx = 6u32;
        let handle = u128::from(sampler_idx << 20 | tex_idx);
        let parameters = Parameters {
            workgroup_size: [3, 1, 1],
            ..Default::default()
        };
        let sampler = Sampler {
            min_filter: vk::Filter::LINEAR,
            mag_filter: vk::Filter::LINEAR,
            base_level_only: true,
            address: [vk::SamplerAddressMode::CLAMP_TO_BORDER; 3],
            border: [1.0f32.to_bits(); 4],
            ..Default::default()
        };
        let inputs = [10u64, 11, 12];
        for (key, format, value) in [
            (
                inputs[0],
                vk::Format::R16G16_UINT,
                vk::ClearColorValue {
                    uint32: [24, 0, 0, 0],
                },
            ),
            (
                inputs[1],
                vk::Format::R16G16_SINT,
                vk::ClearColorValue {
                    int32: [-11, 0, 0, 0],
                },
            ),
            (
                inputs[2],
                vk::Format::R16G16_UNORM,
                vk::ClearColorValue { float32: [0.0; 4] },
            ),
        ] {
            renderer.target(key, 1, 1, format).unwrap();
            renderer.clear_color(key, None, value).unwrap();
        }
        for (name, position, expected) in [
            ("interior", 0.5f32, [24u32, (-11i32) as u32, 0]),
            (
                "interpolated border",
                0.25f32,
                [24, (-11i32) as u32, 0.25f32.to_bits()],
            ),
            ("border", 1.75f32, [1, 1, 1.0f32.to_bits()]),
        ] {
            let sample: Vec<u8> = [
                mov(0, u128::from(position.to_bits())),
                mov(1, u128::from(0.5f32.to_bits())),
                word(&[(0, 0x919), (16, 4), (72, 0x21)]),
                mov(5, 0),
                mov(6, handle),
                mov(7, 0),
                word(&[
                    (0, 0x210),
                    (16, 6),
                    (24, 6),
                    (32, 4),
                    (64, 255),
                    (81, 7),
                    (84, 7),
                ]),
                word(&[
                    (0, 0x361),
                    (16, 10),
                    (32, 6),
                    (59, 1),
                    (61, 1),
                    (64, 255),
                    (72, 1),
                    (81, 7),
                    (87, 3),
                ]),
                word(&[(0, 0xb9b), (24, 4), (32, 10), (40, 0x12), (61, 3), (72, 1)]),
                word(&[(0, 0x94d), (87, 7)]),
            ]
            .into_iter()
            .flat_map(u128::to_le_bytes)
            .collect();
            let discovery = shader::translate_rust(&sample, &parameters).unwrap();
            let desc = discovery
                .descriptors
                .iter()
                .find(|d| d.kind() == DescriptorKind::Image)
                .unwrap();
            let storage = shader::BoundStorageImage {
                bank: desc.cbuf_index,
                offset: desc.cbuf_offset,
                view: desc.texture_type,
                sample_type: shader::SampleType::Unsigned,
            };
            let stage = shader::translate_rust_with_bound_images(
                &sample,
                &parameters,
                None,
                &[],
                &[storage],
            )
            .unwrap();
            let image = stage
                .descriptors
                .iter()
                .find(|d| d.kind() == DescriptorKind::Image)
                .unwrap()
                .binding;
            let program = renderer.create_program(vec![Arc::new(stage)]).unwrap();
            let originals = [0usize, 1, 2];
            let swapped = [2usize, 0, 1];
            let output = [20u64, 21];
            for (out, order) in output.into_iter().zip([originals, swapped]) {
                renderer.target(out, 3, 1, vk::Format::R32_UINT).unwrap();
                renderer
                    .clear_color(out, None, vk::ClearColorValue { uint32: [0; 4] })
                    .unwrap();
                let pool = order
                    .into_iter()
                    .enumerate()
                    .map(|(i, ix)| PoolEntry {
                        texture: tex_idx + i as u32,
                        sampler: sampler_idx,
                        source: TextureSource::Target(inputs[ix]),
                        sampler_state: sampler,
                    })
                    .collect::<Vec<_>>()
                    .into();
                let dispatch = Dispatch {
                    program: &program,
                    groups: [1; 3],
                    uniforms: Default::default(),
                    storage: Default::default(),
                    storage_in_place: Default::default(),
                    textures: Default::default(),
                    images: std::collections::HashMap::from([(
                        image,
                        StorageImage {
                            target: out,
                            ..Default::default()
                        },
                    )]),
                    image_formats: Default::default(),
                    indirect: None,
                    page_table: Default::default(),
                    pool,
                };
                renderer.dispatch_deferred(&dispatch).unwrap();
            }
            let mut pixels = Vec::new();
            for (out, order) in output.into_iter().zip([originals, swapped]) {
                let data = renderer.read_target_bytes(out, 4).unwrap();
                let actual: Vec<_> = data
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|bytes| u32::from_le_bytes(*bytes))
                    .collect();
                let expected = order.map(|at| expected[at]);
                pixels.push((actual, expected));
            }
            renderer.destroy_program(program).unwrap();
            for (actual, expected) in pixels {
                assert_eq!(actual, expected, "{name}");
            }
        }
    }

    #[test]
    fn retiring_a_view_clears_its_aliases_after_descriptor_rewrites() {
        let Ok(gpu) = crate::vulkan::Gpu::new() else {
            return;
        };
        if gpu.texture_pool_layout.is_none() {
            return;
        }
        let mut renderer = Renderer::new(gpu).unwrap();
        for key in [7, 8] {
            renderer
                .target(key, 4, 4, vk::Format::R8G8B8A8_UNORM)
                .unwrap();
        }
        let entry = |texture, key| PoolEntry {
            texture,
            sampler: 1,
            source: TextureSource::Target(key),
            sampler_state: Sampler::default(),
        };
        renderer
            .bind_pool(&vec![entry(1, 7), entry(2, 7), entry(3, 7)].into())
            .unwrap();
        renderer.bind_pool(&vec![entry(1, 8)].into()).unwrap();
        let view = renderer.existing_target(7).unwrap().sampled_view;
        let new = renderer.existing_target(8).unwrap().sampled_view;
        renderer.flush().unwrap();
        renderer.forget_pool_views(&[view, view]);

        let written = &renderer.texture_pool.as_ref().unwrap().written;
        assert_eq!(written[&(binding::SAMPLED_2D, 1)], new.as_raw());
        assert!(!written.contains_key(&(binding::SAMPLED_2D, 2)));
        assert!(!written.contains_key(&(binding::SAMPLED_2D, 3)));
        assert!(written.contains_key(&(binding::SAMPLERS, 1)));
    }
}
