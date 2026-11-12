//! Texture instructions whose handle is a word in a constant bank: each
//! (handle word, view type, depth comparison) is a combined image sampler
//! the runtime binds from that word (`Descriptor::cbuf_index` and
//! `cbuf_offset`), so a translation holds no texture state.

use echora_shader_abi::{
    Descriptor, SAMPLER_INDEX_SHIFT, TEXTURE_INDEX_MASK, TEXTURE_POOL_CAPACITY_SPEC_ID,
    TEXTURE_POOL_SET, Unsupported, texture_pool,
};

use super::{Emitter, Known, unsupported};
use crate::decode::{Texture, TextureOp};
use crate::spirv::{Type, capability, decoration, glsl, op, storage};

/// SPIR-V image operand bits.
/// SPIR-V's SpecId decoration.
const DECORATION_SPEC_ID: u32 = 1;

/// SPIR-V's NonWritable decoration.
const DECORATION_NON_WRITABLE: u32 = 24;

const OPERAND_BIAS: u32 = 0x1;
const OPERAND_LOD: u32 = 0x2;
const OPERAND_GRAD: u32 = 0x4;
const OPERAND_OFFSET: u32 = 0x10;
const OPERAND_MIN_LOD: u32 = 0x80;

/// A view type: SPIR-V's dimension and arrayed flag, the runtime's
/// texture type, and how many coordinates (array index included) it takes.
struct View {
    dim: u32,
    arrayed: bool,
    texture_type: u32,
    coordinates: u32,
}

fn view(dimension: u8) -> Option<View> {
    let (dim, arrayed, texture_type, coordinates) = match dimension {
        0 => (0, false, 0, 1),
        1 => (1, false, 2, 2),
        2 => (2, false, 4, 3),
        3 => (3, false, 5, 3),
        4 => (0, true, 1, 2),
        5 => (1, true, 3, 3),
        7 => (3, true, 6, 4),
        _ => return None,
    };
    Some(View {
        dim,
        arrayed,
        texture_type,
        coordinates,
    })
}

/// The runtime's descriptor kind for a texture.
const KIND_TEXTURE: u32 = 4;
/// `Descriptor::flags`: read, and a depth texture.
const FLAG_READ: u32 = 2;
const FLAG_DEPTH: u32 = 8;

/// A texture's SPIR-V types: the sampled image and the image.
#[derive(Clone, Copy)]
struct Types {
    sampled: u32,
    image: u32,
}

/// A pool texture's float and uint views (bindings and types), its index,
/// its sampler and whether the entry is integer. Each branch loads only
/// its own view's descriptor (an unwritten slot of the other view is never
/// read) and makes its sampled image, as SPIR-V wants one in the block that
/// uses it.
#[derive(Clone, Copy, Debug)]
pub(super) struct PoolViews {
    integer: u32,
    sampler: u32,
    index: u32,
    float_binding: u32,
    float_types: (u32, u32),
    uint_binding: u32,
    uint_types: (u32, u32),
    uniform: bool,
}

impl Emitter {
    fn texture_types(&mut self, view: &View, depth: bool) -> Types {
        if view.dim == 0 {
            self.module.capability(capability::SAMPLED_1D);
        }
        if view.dim == 3 && view.arrayed {
            self.module.capability(capability::SAMPLED_CUBE_ARRAY);
        }
        let image = self.module.ty(Type::Image {
            sampled_type: self.f32,
            dim: view.dim,
            depth: u32::from(depth),
            arrayed: view.arrayed,
            multisampled: false,
            sampled: 1,
            format: 0,
        });
        let sampled = self.module.ty(Type::SampledImage { image });
        Types { sampled, image }
    }

    /// The combined image sampler for a handle word and view, declared and
    /// reported on first use.
    fn texture_variable(&mut self, texture: &Texture, view: &View, depth: bool) -> (u32, Types) {
        let types = self.texture_types(view, depth);
        let key = (
            texture.bank,
            texture.handle_offset,
            view.texture_type,
            depth,
        );
        if let Some(&variable) = self.textures.get(&key) {
            return (variable, types);
        }
        let pointer = self
            .module
            .pointer(storage::UNIFORM_CONSTANT, types.sampled);
        let variable = self
            .module
            .global_variable(pointer, storage::UNIFORM_CONSTANT);
        self.add_descriptor(
            variable,
            Descriptor {
                kind: KIND_TEXTURE,
                count: 1,
                cbuf_index: u32::from(texture.bank),
                cbuf_offset: u32::from(texture.handle_offset),
                texture_type: view.texture_type,
                flags: FLAG_READ | if depth { FLAG_DEPTH } else { 0 },
                ..Descriptor::default()
            },
        );
        self.textures.insert(key, variable);
        (variable, types)
    }

    /// The texture as a sampled image: its combined image sampler, or for
    /// a handle computed at run time, the texture pool's entries.
    ///
    /// `handle` is the handle's value for the pool, with whether it is the
    /// same across the warp (a uniform register); otherwise the uniform
    /// register in `texture.handle` gives it, or the texture is bound.
    fn sampled_image(
        &mut self,
        index: usize,
        texture: &Texture,
        view: &View,
        depth: bool,
        handle: Option<(u32, bool)>,
    ) -> Result<(u32, Types), Unsupported> {
        let handle = match (handle, texture.handle) {
            (Some(handle), _) => handle,
            (None, Some(register)) => (self.read_uniform(register.0), true),
            (None, None) => {
                let (variable, types) = self.texture_variable(texture, view, depth);
                let sampled = self.value(op::LOAD, types.sampled, &[variable]);
                return Ok((sampled, types));
            }
        };
        let (handle, uniform) = handle;
        let binding = match (view.dim, view.arrayed) {
            (0, false) => texture_pool::SAMPLED_1D,
            (0, true) => texture_pool::SAMPLED_1D_ARRAY,
            (1, false) => texture_pool::SAMPLED_2D,
            (1, true) => texture_pool::SAMPLED_2D_ARRAY,
            (2, _) => texture_pool::SAMPLED_3D,
            (3, false) => texture_pool::SAMPLED_CUBE,
            (3, true) => texture_pool::SAMPLED_CUBE_ARRAY,
            _ => return Err(unsupported(index, "texture pool view")),
        };
        // The pool's images take no depth flag: the sampler's state
        // decides a comparison.
        let types = self.texture_types(view, false);
        let (texture_index, inside, sampler_index) = self.pool_handle(handle);
        self.pool_inside = Some(match self.pool_inside {
            Some(before) => self.value(op::LOGICAL_AND, self.bool, &[before, inside]),
            None => inside,
        });
        let image = self.pool_entry(binding, types.image, texture_index, uniform);
        let sampler_type = self.module.ty(Type::Sampler);
        let sampler = self.pool_entry(texture_pool::SAMPLERS, sampler_type, sampler_index, uniform);
        let sampled = self.value(op::SAMPLED_IMAGE, types.sampled, &[image, sampler]);
        if !uniform {
            self.module.decorate(sampled, decoration::NON_UNIFORM, &[]);
        }
        // An integer entry is read through its uint view, where it has one.
        let uint_binding = match (view.dim, view.arrayed) {
            (1, false) => Some(texture_pool::UINT_2D),
            (1, true) => Some(texture_pool::UINT_2D_ARRAY),
            (2, _) => Some(texture_pool::UINT_3D),
            _ => None,
        };
        self.pool_integer = match uint_binding.filter(|_| !depth) {
            Some(uint_binding) => {
                let uint_image = self.module.ty(Type::Image {
                    sampled_type: self.u32,
                    dim: view.dim,
                    depth: 0,
                    arrayed: view.arrayed,
                    multisampled: false,
                    sampled: 1,
                    format: 0,
                });
                let uint_sampled = self.module.ty(Type::SampledImage { image: uint_image });
                let integer = self.pool_flag(texture_index);
                Some(PoolViews {
                    integer,
                    sampler,
                    index: texture_index,
                    float_binding: binding,
                    float_types: (types.sampled, types.image),
                    uint_binding,
                    uint_types: (uint_sampled, uint_image),
                    uniform,
                })
            }
            None => None,
        };
        Ok((sampled, types))
    }

    /// A pool handle's texture index, whether it is inside the pool, and
    /// its sampler index. An index past the pool's arrays reads as an
    /// unwritten entry: the access uses entry 0 and the results become
    /// zeros.
    fn pool_handle(&mut self, handle: u32) -> (u32, u32, u32) {
        if let (Some(index), Some(inside), Some(sampler)) = (
            self.known(Known::PoolIndex(handle)),
            self.known(Known::PoolInside(handle)),
            self.known(Known::PoolSampler(handle)),
        ) {
            return (index, inside, sampler);
        }
        let (mask, shift) = (
            self.constant(TEXTURE_INDEX_MASK),
            self.constant(SAMPLER_INDEX_SHIFT),
        );
        let texture_index = self.value(op::BITWISE_AND, self.u32, &[handle, mask]);
        let capacity = self.pool_capacity();
        let inside = self.value(op::U_LESS_THAN, self.bool, &[texture_index, capacity]);
        let zero = self.constant(0);
        let texture_index = self.value(op::SELECT, self.u32, &[inside, texture_index, zero]);
        let sampler = self.value(op::SHIFT_RIGHT_LOGICAL, self.u32, &[handle, shift]);
        self.remember(Known::PoolIndex(handle), texture_index);
        self.remember(Known::PoolInside(handle), inside);
        self.remember(Known::PoolSampler(handle), sampler);
        (texture_index, inside, sampler)
    }

    /// Whether pool entry `index` is integer: bit 0 of its word in the
    /// flags buffer (`texture_pool::FLAGS`).
    fn pool_flag(&mut self, index: u32) -> u32 {
        if let Some(flag) = self.known(Known::PoolFlag(index)) {
            return flag;
        }
        let variable = match self.pool_flags {
            Some(variable) => variable,
            None => {
                let array = self.module.ty(Type::RuntimeArray { element: self.u32 });
                self.module.decorate(array, decoration::ARRAY_STRIDE, &[4]);
                let block = self.module.ty(Type::Struct {
                    members: vec![array],
                });
                self.module.decorate(block, decoration::BLOCK, &[]);
                self.module
                    .member_decorate(block, 0, decoration::OFFSET, &[0]);
                self.module
                    .member_decorate(block, 0, DECORATION_NON_WRITABLE, &[]);
                let pointer = self.module.pointer(storage::STORAGE_BUFFER, block);
                let variable = self
                    .module
                    .global_variable(pointer, storage::STORAGE_BUFFER);
                self.module
                    .decorate(variable, decoration::DESCRIPTOR_SET, &[TEXTURE_POOL_SET]);
                self.module
                    .decorate(variable, decoration::BINDING, &[texture_pool::FLAGS]);
                self.interface.push(variable);
                self.pool_bindings |= 1 << texture_pool::FLAGS;
                self.pool_flags = Some(variable);
                variable
            }
        };
        let pointer = self.module.pointer(storage::STORAGE_BUFFER, self.u32);
        let zero = self.constant(0);
        let word = self.value(op::ACCESS_CHAIN, pointer, &[variable, zero, index]);
        let word = self.value(op::LOAD, self.u32, &[word]);
        let one = self.constant(1);
        let bit = self.value(op::BITWISE_AND, self.u32, &[word, one]);
        let flag = self.value(op::I_NOT_EQUAL, self.bool, &[bit, zero]);
        self.remember(Known::PoolFlag(index), flag);
        flag
    }

    /// The texture pool's capacity, a specialization constant
    /// (`SpecMeaning::TexturePoolCapacity`).
    fn pool_capacity(&mut self) -> u32 {
        if let Some(capacity) = self.pool_capacity {
            return capacity;
        }
        let capacity = self.module.u32_spec_constant(TEXTURE_INDEX_MASK + 1);
        self.module.decorate(
            capacity,
            DECORATION_SPEC_ID,
            &[TEXTURE_POOL_CAPACITY_SPEC_ID],
        );
        self.spec_constants.push(echora_shader_abi::SpecConstant {
            id: TEXTURE_POOL_CAPACITY_SPEC_ID,
            meaning: echora_shader_abi::SpecMeaning::TexturePoolCapacity,
        });
        self.pool_capacity = Some(capacity);
        capacity
    }

    /// Entry `index` of texture-pool binding `binding` (an unsized array of
    /// `element`); a non-uniform index unless `uniform`.
    fn pool_entry(&mut self, binding: u32, element: u32, index: u32, uniform: bool) -> u32 {
        let key = Known::PoolEntry(binding, element, index, uniform);
        if let Some(value) = self.known(key) {
            return value;
        }
        self.module.capability(capability::RUNTIME_DESCRIPTOR_ARRAY);
        if !uniform {
            self.module.capability(capability::SHADER_NON_UNIFORM);
            self.module
                .capability(capability::SAMPLED_IMAGE_ARRAY_NON_UNIFORM_INDEXING);
        }
        let array_type = self.module.ty(Type::RuntimeArray { element });
        let variable = match self.pool_arrays.get(&binding) {
            Some(&variable) => variable,
            None => {
                let pointer = self.module.pointer(storage::UNIFORM_CONSTANT, array_type);
                let variable = self
                    .module
                    .global_variable(pointer, storage::UNIFORM_CONSTANT);
                self.module
                    .decorate(variable, decoration::DESCRIPTOR_SET, &[TEXTURE_POOL_SET]);
                self.module
                    .decorate(variable, decoration::BINDING, &[binding]);
                self.interface.push(variable);
                self.pool_arrays.insert(binding, variable);
                self.pool_bindings |= 1 << binding;
                variable
            }
        };
        let pointer = self.module.pointer(storage::UNIFORM_CONSTANT, element);
        let entry = self.value(op::ACCESS_CHAIN, pointer, &[variable, index]);
        let value = self.value(op::LOAD, element, &[entry]);
        if !uniform {
            self.module.decorate(entry, decoration::NON_UNIFORM, &[]);
            self.module.decorate(value, decoration::NON_UNIFORM, &[]);
        }
        self.remember(key, value);
        value
    }

    /// The instruction's operand values, in order (RZ reads zero).
    fn texture_values(&mut self, texture: &Texture, count: u32) -> Vec<u32> {
        self.texture_values_split(texture, count, None)
    }

    /// As `texture_values`; with `coordinates` given (TEX.B, an unpacked
    /// form or a least-level mode) Ra holds that many values when unpacked
    /// (bit 60 clear) or with five values or more, Rb the rest; a packed
    /// form (bit 60 set) of up to four values has them half in each (NAK's
    /// SM70 scalar form: 1+1, 2+1, 2+2).
    fn texture_values_split(
        &mut self,
        texture: &Texture,
        count: u32,
        coordinates: Option<u32>,
    ) -> Vec<u32> {
        let in_a = match coordinates {
            Some(coordinates) if !texture.packed || count > 4 => coordinates,
            _ if count <= 4 => count.div_ceil(2),
            _ => 4,
        };
        (0..count)
            .map(|i| {
                let (base, n) = if i < in_a {
                    (texture.a.0, i)
                } else {
                    (texture.b.0, i - in_a)
                };
                if base == 255 {
                    self.constant(0)
                } else {
                    self.read_register(base.saturating_add(n as u8))
                }
            })
            .collect()
    }

    /// A vector (or scalar) of `values` of type `component`.
    pub(super) fn vector(&mut self, component: u32, values: &[u32]) -> u32 {
        if values.len() == 1 {
            return values[0];
        }
        let ty = self.module.ty(Type::Vector {
            component,
            count: values.len() as u32,
        });
        let id = self.module.id();
        let mut operands = vec![ty, id];
        operands.extend_from_slice(values);
        self.function.emit(op::COMPOSITE_CONSTRUCT, &operands);
        id
    }

    pub(super) fn texture(&mut self, index: usize, texture: &Texture) -> Result<(), Unsupported> {
        // A bindless handle binds like a bound one when its uniform
        // register only ever holds one constant-bank word.
        let mut texture = *texture;
        if let Some(register) = texture.handle
            && let Some(&(bank, offset)) = self.handle_words.get(&register.0)
        {
            texture.bank = bank;
            texture.handle_offset = offset;
            texture.handle = None;
        }
        let texture = &texture;
        let view = view(texture.dimension).ok_or_else(|| {
            unsupported(index, format!("texture dimension {}", texture.dimension))
        })?;
        if texture.mask == 0 {
            return Ok(());
        }
        self.pool_inside = None;
        self.pool_integer = None;
        if texture.handle.is_none()
            && !texture.handle_value
            && usize::from(texture.bank) >= self.cbuf_used_sizes.len()
        {
            return Err(unsupported(index, "texture handle in an unbound bank"));
        }
        let results = match texture.op {
            TextureOp::Query => self.texture_query(index, texture)?,
            TextureOp::LevelQuery => self.texture_level_query(index, texture, &view)?,
            TextureOp::Gradient => self.texture_gradient(index, texture, &view)?,
            _ => self.texture_read(index, texture, &view)?,
        };
        // A handle past the texture pool reads zeros.
        let results = match self.pool_inside.take() {
            Some(inside) => {
                let zero = self.constant(0);
                results
                    .iter()
                    .map(|&value| self.value(op::SELECT, self.u32, &[inside, value, zero]))
                    .collect()
            }
            None => results,
        };
        self.vertex_trace_texture(index, &results);
        // The written components, in mask order.
        let components: Vec<u32> = (0..4)
            .filter(|n| texture.mask >> n & 1 != 0)
            .map(|n| results[n])
            .collect();
        let components = if texture.half {
            // Pairs packed as halves, the first in the low half.
            let zero = self.constant(0);
            components
                .chunks(2)
                .map(|pair| {
                    let floats: Vec<u32> = [pair[0], *pair.get(1).unwrap_or(&zero)]
                        .iter()
                        .map(|&bits| self.value(op::BITCAST, self.f32, &[bits]))
                        .collect();
                    let vector = self.vector(self.f32, &floats);
                    self.glsl(self.u32, glsl::PACK_HALF_2X16, &[vector])
                })
                .collect()
        } else {
            components
        };
        // Components 0-1 go to Rd and 2-3 to Rd2 (else the registers after
        // Rd's): as words, or as one packed word each with halves.
        let per_register = if texture.half { 1 } else { 2 };
        for (n, value) in components.into_iter().enumerate() {
            let target = if texture.d2.0 == 255 || n < per_register {
                if texture.d.0 == 255 {
                    continue;
                }
                texture.d.0.saturating_add(n as u8)
            } else {
                texture.d2.0.saturating_add((n - per_register) as u8)
            };
            self.write_register(target, value);
        }
        Ok(())
    }

    /// TEX, TLD and TLD4: four result words (as integers' bits).
    fn texture_read(
        &mut self,
        index: usize,
        texture: &Texture,
        view: &View,
    ) -> Result<Vec<u32>, Unsupported> {
        let fetch = texture.op == TextureOp::Fetch;
        // Level modes: TEX implicit, zero, bias, explicit, implicit with a
        // least level (.LC) or biased with one (.LB.LC); TLD zero or
        // explicit; TLD4 none (NAK's SM70 numbering, 0-5).
        let valid = match texture.op {
            TextureOp::Sample => texture.lod <= 5,
            TextureOp::Fetch => matches!(texture.lod, 1 | 3),
            _ => true,
        };
        if !valid {
            return Err(unsupported(
                index,
                format!("texture level mode {}", texture.lod),
            ));
        }
        if texture.offsets && view.dim == 3 {
            return Err(unsupported(index, "cube texture with offsets"));
        }
        if fetch && (view.dim == 3 || texture.depth_compare) {
            return Err(unsupported(index, "cube or depth texel fetch"));
        }
        let level_value = matches!(texture.lod, 2 | 3 | 5);
        // .LC: the least level in the high half of the register an array's
        // layer takes (its low half), one of its own before the coordinates
        // otherwise, as 8.8 fixed point (NAK: f2u32(min_lod * 256), packed
        // by PRMT 0x1054 with the layer).
        let clamp = texture.op == TextureOp::Sample && matches!(texture.lod, 4 | 5);
        let extra = u32::from(clamp && !view.arrayed);
        let handle_value = texture.handle_value;
        let count = view.coordinates
            + extra
            + u32::from(handle_value)
            + u32::from(level_value)
            + u32::from(texture.offsets)
            + 2 * u32::from(texture.per_tap)
            + u32::from(texture.depth_compare);
        // Unpacked (bit 60 clear), as NAK lays it out: the array index (or
        // least level) and the coordinates in Ra, the rest (level, offsets,
        // reference) in Rb. The game's one unpacked form, a 3D TEX.LL with
        // offsets, reads its level and offsets from Rb and leaves Ra's
        // fourth register unwritten.
        let values = self.texture_values_split(
            texture,
            count,
            (handle_value || clamp || !texture.packed).then_some(view.coordinates + extra),
        );
        let (values, least_level) = if clamp {
            let first = values[0];
            let sixteen = self.constant(16);
            let fixed = self.value(op::SHIFT_RIGHT_LOGICAL, self.u32, &[first, sixteen]);
            let fixed = self.value(op::CONVERT_U_TO_F, self.f32, &[fixed]);
            let step = self.float_constant(1.0 / 256.0);
            let level = self.value(op::F_MUL, self.f32, &[fixed, step]);
            let values = if view.arrayed {
                let mask = self.constant(0xffff);
                let layer = self.value(op::BITWISE_AND, self.u32, &[first, mask]);
                std::iter::once(layer)
                    .chain(values[1..].iter().copied())
                    .collect()
            } else {
                values[1..].to_vec()
            };
            (values, Some(level))
        } else {
            (values, None)
        };
        let mut next = view.coordinates as usize;
        // TEX.B: the handle follows the coordinates, a value that may
        // differ across the warp.
        let handle = handle_value.then(|| {
            next += 1;
            (values[next - 1], false)
        });
        let (sampled, types) =
            self.sampled_image(index, texture, view, texture.depth_compare, handle)?;
        let level_operand = level_value.then(|| {
            next += 1;
            values[next - 1]
        });
        let packed_offsets = texture.offsets.then(|| {
            next += 1;
            values[next - 1]
        });
        let per_tap = texture.per_tap.then(|| {
            next += 2;
            [values[next - 2], values[next - 1]]
        });
        let reference = texture
            .depth_compare
            .then(|| self.value(op::BITCAST, self.f32, &[values[next]]));
        // The array index comes first; SPIR-V wants it last.
        let mut coordinates: Vec<u32> = values[..view.coordinates as usize].to_vec();
        if view.arrayed {
            coordinates.rotate_left(1);
        }
        let spatial = (view.coordinates - u32::from(view.arrayed)) as usize;
        // Texel offsets: a gather takes them as an operand; a fetch adds
        // them to its coordinates, and a sample adds them scaled by the
        // level's size (Vulkan takes only constant offsets there).
        // Packed signed offsets: 8 bits a component for TLD4, 4 for the
        // others (as NAK packs them: nak_nir_lower_tex.c).
        let gather = matches!(texture.op, TextureOp::Gather { .. });
        let field = if gather { 8 } else { 4 };
        let offsets: Option<Vec<u32>> = packed_offsets.map(|packed| {
            let width = self.constant(field);
            (0..spatial as u32)
                .map(|n| {
                    let at = self.constant(n * field);
                    self.value(op::BIT_FIELD_S_EXTRACT, self.u32, &[packed, at, width])
                })
                .collect()
        });
        // Only fragment programs have derivatives for an implicit level.
        let implicit = self.stage == echora_shader_abi::Stage::Fragment
            && matches!(texture.lod, 0 | 2 | 4 | 5);
        let coordinate = if fetch {
            if let Some(offsets) = &offsets {
                for (n, &offset) in offsets.iter().enumerate() {
                    coordinates[n] = self.value(op::I_ADD, self.u32, &[coordinates[n], offset]);
                }
            }
            self.vector(self.u32, &coordinates)
        } else {
            let mut floats: Vec<u32> = coordinates
                .iter()
                .enumerate()
                .map(|(n, &value)| {
                    if view.arrayed && n == coordinates.len() - 1 {
                        // The layer is an integer.
                        self.value(op::CONVERT_U_TO_F, self.f32, &[value])
                    } else {
                        self.value(op::BITCAST, self.f32, &[value])
                    }
                })
                .collect();
            if let (false, Some(offsets)) = (gather, &offsets) {
                self.module.capability(capability::IMAGE_QUERY);
                let image = self.value(op::IMAGE, types.image, &[sampled]);
                let level = match level_operand {
                    Some(value) => {
                        let level = self.value(op::BITCAST, self.f32, &[value]);
                        let level = self.glsl(self.f32, glsl::FLOOR, &[level]);
                        self.value(op::CONVERT_F_TO_U, self.u32, &[level])
                    }
                    None => self.constant(0),
                };
                // A level past the image's last is undefined to query: the
                // last one instead.
                let levels = self.value(op::IMAGE_QUERY_LEVELS, self.u32, &[image]);
                let one = self.constant(1);
                let levels = self.glsl(self.u32, glsl::U_MAX, &[levels, one]);
                let last = self.value(op::I_SUB, self.u32, &[levels, one]);
                let level = self.glsl(self.u32, glsl::U_MIN, &[level, last]);
                let size_type = self.module.ty(Type::Vector {
                    component: self.u32,
                    count: (spatial + usize::from(view.arrayed)) as u32,
                });
                let size = if spatial + usize::from(view.arrayed) == 1 {
                    self.value(op::IMAGE_QUERY_SIZE_LOD, self.u32, &[image, level])
                } else {
                    self.value(op::IMAGE_QUERY_SIZE_LOD, size_type, &[image, level])
                };
                for (n, &offset) in offsets.iter().enumerate() {
                    let extent = if spatial + usize::from(view.arrayed) == 1 {
                        size
                    } else {
                        self.value(op::COMPOSITE_EXTRACT, self.u32, &[size, n as u32])
                    };
                    // At least a texel (an unbound texture's size is 0, and
                    // 0 / 0 would make the coordinate NaN).
                    let extent = self.glsl(self.u32, glsl::U_MAX, &[extent, one]);
                    let extent = self.value(op::CONVERT_U_TO_F, self.f32, &[extent]);
                    let offset = self.value(op::CONVERT_S_TO_F, self.f32, &[offset]);
                    let step = self.value(op::F_DIV, self.f32, &[offset, extent]);
                    floats[n] = self.value(op::F_ADD, self.f32, &[floats[n], step]);
                }
            }
            self.vector(self.f32, &floats)
        };
        let vec4 = self.module.ty(Type::Vector {
            component: self.f32,
            count: 4,
        });
        // Image operands: a level (explicit; a compute program has no
        // implicit one), or for a gather the offsets.
        let mut mask = 0;
        let mut operands = Vec::new();
        match texture.op {
            TextureOp::Fetch => {
                mask |= OPERAND_LOD;
                operands.push(level_operand.unwrap_or_else(|| self.constant(0)));
            }
            // A fragment program samples at the implicit level, biased.
            TextureOp::Sample if implicit => {
                if matches!(texture.lod, 2 | 5) {
                    mask |= OPERAND_BIAS;
                    let bias = level_operand.unwrap_or_else(|| self.float_constant(0.0));
                    let bias = self.value(op::BITCAST, self.f32, &[bias]);
                    operands.push(bias);
                }
                if let Some(least) = least_level {
                    self.module.capability(capability::MIN_LOD);
                    mask |= OPERAND_MIN_LOD;
                    operands.push(least);
                }
            }
            // Elsewhere a bias over level zero is that level (no less than
            // a least level).
            TextureOp::Sample => {
                mask |= OPERAND_LOD;
                let mut lod = match level_operand {
                    Some(value) => self.value(op::BITCAST, self.f32, &[value]),
                    None => self.float_constant(0.0),
                };
                if let Some(least) = least_level {
                    lod = self.glsl(self.f32, glsl::N_MAX, &[lod, least]);
                }
                operands.push(lod);
            }
            _ => {}
        }
        if let (true, Some(offsets)) = (gather, &offsets) {
            self.module.capability(capability::IMAGE_GATHER_EXTENDED);
            mask |= OPERAND_OFFSET;
            let offset = self.vector(self.u32, offsets);
            operands.push(offset);
        }
        let image_operands = |mut list: Vec<u32>| {
            if mask != 0 {
                list.push(mask);
                list.extend_from_slice(&operands);
            }
            list
        };
        if let (Some(packed), TextureOp::Gather { component }) = (per_tap, texture.op) {
            // One gather per tap at that tap's offset (signed six bits per
            // byte: x0, y0, x1, y1 then x2, y2, x3, y3), keeping its tap.
            self.module.capability(capability::IMAGE_GATHER_EXTENDED);
            let six = self.constant(6);
            let component = self.constant(u32::from(component));
            let mut taps = Vec::new();
            for tap in 0..4u32 {
                let word = packed[tap as usize / 2];
                let at = |e: &mut Self, byte: u32| {
                    let shift = e.constant(byte * 8);
                    e.value(op::BIT_FIELD_S_EXTRACT, e.u32, &[word, shift, six])
                };
                let x = at(self, tap % 2 * 2);
                let y = at(self, tap % 2 * 2 + 1);
                let offset = self.vector(self.u32, &[x, y]);
                let operands = match reference {
                    Some(reference) => vec![sampled, coordinate, reference, OPERAND_OFFSET, offset],
                    None => vec![sampled, coordinate, component, OPERAND_OFFSET, offset],
                };
                let opcode = if reference.is_some() {
                    op::IMAGE_DREF_GATHER
                } else {
                    op::IMAGE_GATHER
                };
                let gathered = self.value(opcode, vec4, &operands);
                let tap_value = self.value(op::COMPOSITE_EXTRACT, self.f32, &[gathered, tap]);
                taps.push(self.value(op::BITCAST, self.u32, &[tap_value]));
            }
            return Ok(taps);
        }
        // A pool entry that may be integer: its uint view where the flag
        // says so, the float view otherwise.
        let integer = self.pool_integer.take().filter(|_| reference.is_none());
        if let Some(views) = integer {
            let uvec4 = self.module.ty(Type::Vector {
                component: self.u32,
                count: 4,
            });
            let words: Vec<u32> = (0..4)
                .map(|_| self.function.variable(&mut self.module, self.u32_ptr, None))
                .collect();
            // An implicit level needs derivatives, which a branch on the
            // integer flag may not keep: the level a sample would use is
            // queried before it (plus the bias) and both views sample at it
            // explicitly.
            let level = (implicit && texture.op == TextureOp::Sample).then(|| {
                self.module.capability(capability::IMAGE_QUERY);
                let vec2 = self.module.ty(Type::Vector {
                    component: self.f32,
                    count: 2,
                });
                let query = self.value(op::IMAGE_QUERY_LOD, vec2, &[sampled, coordinate]);
                let mut level = self.value(op::COMPOSITE_EXTRACT, self.f32, &[query, 0]);
                if mask & OPERAND_BIAS != 0 {
                    level = self.value(op::F_ADD, self.f32, &[level, operands[0]]);
                }
                if let Some(least) = least_level {
                    level = self.glsl(self.f32, glsl::N_MAX, &[level, least]);
                }
                level
            });
            self.choose(views.integer, |e, integer| {
                let (binding, (sampled_type, image_type), ty) = if integer {
                    (views.uint_binding, views.uint_types, uvec4)
                } else {
                    (views.float_binding, views.float_types, vec4)
                };
                let image = e.pool_entry(binding, image_type, views.index, views.uniform);
                let sampled = e.value(op::SAMPLED_IMAGE, sampled_type, &[image, views.sampler]);
                if !views.uniform {
                    e.module.decorate(sampled, decoration::NON_UNIFORM, &[]);
                }
                let result = match texture.op {
                    TextureOp::Fetch => {
                        let image = e.value(op::IMAGE, image_type, &[sampled]);
                        e.value(
                            op::IMAGE_FETCH,
                            ty,
                            &image_operands(vec![image, coordinate]),
                        )
                    }
                    TextureOp::Gather { component } => {
                        let component = e.constant(u32::from(component));
                        e.value(
                            op::IMAGE_GATHER,
                            ty,
                            &image_operands(vec![sampled, coordinate, component]),
                        )
                    }
                    _ => match level {
                        Some(level) => e.value(
                            op::IMAGE_SAMPLE_EXPLICIT_LOD,
                            ty,
                            &[sampled, coordinate, OPERAND_LOD, level],
                        ),
                        None => e.value(
                            if implicit {
                                op::IMAGE_SAMPLE_IMPLICIT_LOD
                            } else {
                                op::IMAGE_SAMPLE_EXPLICIT_LOD
                            },
                            ty,
                            &image_operands(vec![sampled, coordinate]),
                        ),
                    },
                };
                for (n, &word) in words.iter().enumerate() {
                    let value = if integer {
                        e.value(op::COMPOSITE_EXTRACT, e.u32, &[result, n as u32])
                    } else {
                        let value = e.value(op::COMPOSITE_EXTRACT, e.f32, &[result, n as u32]);
                        e.value(op::BITCAST, e.u32, &[value])
                    };
                    e.function.emit(op::STORE, &[word, value]);
                }
            });
            return Ok(words
                .into_iter()
                .map(|word| self.value(op::LOAD, self.u32, &[word]))
                .collect());
        }
        let result = match (texture.op, reference) {
            (TextureOp::Fetch, _) => {
                let image = self.value(op::IMAGE, types.image, &[sampled]);
                self.value(
                    op::IMAGE_FETCH,
                    vec4,
                    &image_operands(vec![image, coordinate]),
                )
            }
            (TextureOp::Sample, None) => self.value(
                if implicit {
                    op::IMAGE_SAMPLE_IMPLICIT_LOD
                } else {
                    op::IMAGE_SAMPLE_EXPLICIT_LOD
                },
                vec4,
                &image_operands(vec![sampled, coordinate]),
            ),
            (TextureOp::Sample, Some(reference)) => {
                let value = self.value(
                    if implicit {
                        op::IMAGE_SAMPLE_DREF_IMPLICIT_LOD
                    } else {
                        op::IMAGE_SAMPLE_DREF_EXPLICIT_LOD
                    },
                    self.f32,
                    &image_operands(vec![sampled, coordinate, reference]),
                );
                let zero = self.float_constant(0.0);
                let one = self.float_constant(1.0);
                self.vector(self.f32, &[value, zero, zero, one])
            }
            (TextureOp::Gather { component }, None) => {
                let component = self.constant(u32::from(component));
                self.value(
                    op::IMAGE_GATHER,
                    vec4,
                    &image_operands(vec![sampled, coordinate, component]),
                )
            }
            (TextureOp::Gather { .. }, Some(reference)) => self.value(
                op::IMAGE_DREF_GATHER,
                vec4,
                &image_operands(vec![sampled, coordinate, reference]),
            ),
            (TextureOp::Query | TextureOp::LevelQuery | TextureOp::Gradient, _) => {
                unreachable!("queries are not reads")
            }
        };
        Ok((0..4)
            .map(|n| {
                let component = self.value(op::COMPOSITE_EXTRACT, self.f32, &[result, n]);
                self.value(op::BITCAST, self.u32, &[component])
            })
            .collect())
    }

    /// TMML: the level a sample would use and the unclamped level, as 8.8
    /// fixed point. Only a fragment program has the derivatives it needs.
    fn texture_level_query(
        &mut self,
        index: usize,
        texture: &Texture,
        view: &View,
    ) -> Result<Vec<u32>, Unsupported> {
        // Only fragment programs have the derivatives a level comes from.
        if self.stage != echora_shader_abi::Stage::Fragment {
            return Err(unsupported(index, "TMML outside a fragment program"));
        }
        self.module.capability(capability::IMAGE_QUERY);
        // The coordinates: all in Ra when Rb is RZ, else half in each.
        let values: Vec<u32> = if texture.b.0 == 255 {
            (0..view.coordinates as u8)
                .map(|n| self.read_register(texture.a.0.saturating_add(n)))
                .collect()
        } else {
            self.texture_values(texture, view.coordinates)
        };
        let (sampled, _) = self.sampled_image(index, texture, view, false, None)?;
        // An array's layer comes first (NAK gives it 0: the level does not
        // depend on it); the query takes the coordinates alone.
        let floats: Vec<u32> = values
            .iter()
            .skip(usize::from(view.arrayed))
            .map(|&value| self.value(op::BITCAST, self.f32, &[value]))
            .collect();
        let coordinate = self.vector(self.f32, &floats);
        let vec2 = self.module.ty(Type::Vector {
            component: self.f32,
            count: 2,
        });
        let lod = self.value(op::IMAGE_QUERY_LOD, vec2, &[sampled, coordinate]);
        let scale = self.float_constant(256.0);
        let clamped = self.value(op::COMPOSITE_EXTRACT, self.f32, &[lod, 0]);
        let clamped = self.value(op::F_MUL, self.f32, &[clamped, scale]);
        let clamped = self.value(op::CONVERT_F_TO_U, self.u32, &[clamped]);
        let raw = self.value(op::COMPOSITE_EXTRACT, self.f32, &[lod, 1]);
        let raw = self.value(op::F_MUL, self.f32, &[raw, scale]);
        let raw = self.value(op::CONVERT_F_TO_S, self.u32, &[raw]);
        let zero = self.constant(0);
        Ok(vec![clamped, raw, zero, zero])
    }

    /// TXD: a 2D or 2D-array sample at the derivatives given.
    fn texture_gradient(
        &mut self,
        index: usize,
        texture: &Texture,
        view: &View,
    ) -> Result<Vec<u32>, Unsupported> {
        // Ra: the coordinates, then an array's layer (bits 0-15; with
        // offsets they would share its register, bits 16-31); Rb: each
        // coordinate's x and y derivative in turn (as NAK lays TXD out,
        // nak_nir_lower_tex.c). Offsets move a sample by texels of the level
        // the gradients pick, which Vulkan only takes as constants.
        if texture.offsets {
            return Err(unsupported(index, "TXD with offsets"));
        }
        let read = |e: &mut Self, base: u8, n: u8| {
            if base == 255 {
                e.float_constant(0.0)
            } else {
                let bits = e.read_register(base.saturating_add(n));
                e.value(op::BITCAST, e.f32, &[bits])
            }
        };
        let spatial = (view.coordinates - u32::from(view.arrayed)) as u8;
        let mut coordinates: Vec<u32> = (0..spatial).map(|n| read(self, texture.a.0, n)).collect();
        if view.arrayed {
            let layer = if texture.a.0 == 255 {
                self.constant(0)
            } else {
                self.read_register(texture.a.0.saturating_add(spatial))
            };
            let mask = self.constant(0xffff);
            let layer = self.value(op::BITWISE_AND, self.u32, &[layer, mask]);
            coordinates.push(self.value(op::CONVERT_U_TO_F, self.f32, &[layer]));
        }
        let coordinate = self.vector(self.f32, &coordinates);
        let d: Vec<u32> = (0..2 * spatial)
            .map(|n| read(self, texture.b.0, n))
            .collect();
        let dx: Vec<u32> = d.iter().step_by(2).copied().collect();
        let dy: Vec<u32> = d.iter().skip(1).step_by(2).copied().collect();
        let (dx, dy) = (self.vector(self.f32, &dx), self.vector(self.f32, &dy));
        let (sampled, _) = self.sampled_image(index, texture, view, false, None)?;
        let vec4 = self.module.ty(Type::Vector {
            component: self.f32,
            count: 4,
        });
        let result = self.value(
            op::IMAGE_SAMPLE_EXPLICIT_LOD,
            vec4,
            &[sampled, coordinate, OPERAND_GRAD, dx, dy],
        );
        Ok((0..4)
            .map(|n| {
                let component = self.value(op::COMPOSITE_EXTRACT, self.f32, &[result, n]);
                self.value(op::BITCAST, self.u32, &[component])
            })
            .collect())
    }

    /// TXQ (the dimension query): the texture's size, as textureSize gives
    /// it, in components 0-2 (width, height, then layers or depth: 1D
    /// arrays have their layers second), 1 where it has none, and the level
    /// count in component 3 (NAK's build_txq_size and build_txq_levels).
    /// The instruction names no view type: a bound handle's comes from the
    /// program's other uses of it (`texture_views`), else it is queried as
    /// a 2D array, which the runtime binds for it (a 2D texture as one
    /// layer); a bindless handle is queried as 2D.
    fn texture_query(&mut self, index: usize, texture: &Texture) -> Result<Vec<u32>, Unsupported> {
        self.module.capability(capability::IMAGE_QUERY);
        let dimension = if texture.handle_value {
            1
        } else {
            match self
                .texture_views
                .get(&(texture.bank, texture.handle_offset))
            {
                Some(&Some(dimension)) if view(dimension).is_some() => {
                    self.txq_views.0 += 1;
                    dimension
                }
                _ => {
                    self.txq_views.1 += 1;
                    let offset = u32::try_from(index * 16).unwrap_or(u32::MAX);
                    self.txq_fallbacks.push((texture.mask, offset));
                    5
                }
            }
        };
        let view = view(dimension).expect("a known dimension");
        // TXQ.B: the handle in Ra, the level in Ra+1; TXQ: the level in Ra.
        let value = |e: &mut Self, register: u8| {
            if register == 255 {
                e.constant(0)
            } else {
                e.read_register(register)
            }
        };
        let (handle, level_register) = if texture.handle_value {
            let next = if texture.a.0 == 255 {
                255
            } else {
                texture.a.0.saturating_add(1)
            };
            (Some((value(self, texture.a.0), false)), next)
        } else {
            (None, texture.a.0)
        };
        let (sampled, types) = self.sampled_image(index, texture, &view, false, handle)?;
        let level = value(self, level_register);
        let image = self.value(op::IMAGE, types.image, &[sampled]);
        // textureSize's components: 1D 1, 1D array and 2D and cube 2, the
        // rest 3.
        let count = match (view.dim, view.arrayed) {
            (0, false) => 1,
            (0, true) | (1, false) | (3, false) => 2,
            _ => 3,
        };
        let size_type = if count == 1 {
            self.u32
        } else {
            self.module.ty(Type::Vector {
                component: self.u32,
                count,
            })
        };
        let size = self.value(op::IMAGE_QUERY_SIZE_LOD, size_type, &[image, level]);
        let one = self.constant(1);
        let mut results: Vec<u32> = (0..3)
            .map(|n| match (n < count, count) {
                (false, _) => one,
                (true, 1) => size,
                (true, _) => self.value(op::COMPOSITE_EXTRACT, self.u32, &[size, n]),
            })
            .collect();
        let levels = self.value(op::IMAGE_QUERY_LEVELS, self.u32, &[image]);
        results.push(levels);
        Ok(results)
    }
}
