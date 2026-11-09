//! Texture instructions whose handle is a word in a constant bank: each
//! (handle word, view type, depth comparison) is a combined image sampler
//! the runtime binds from that word (`Descriptor::cbuf_index` and
//! `cbuf_offset`), so a translation holds no texture state.

use echora_shader_abi::{
    Descriptor, SAMPLER_INDEX_SHIFT, TEXTURE_INDEX_MASK, TEXTURE_POOL_SET, Unsupported,
    texture_pool,
};

use super::{Emitter, unsupported};
use crate::decode::{Texture, TextureOp};
use crate::spirv::{Type, capability, decoration, glsl, op, storage};

/// SPIR-V image operand bits.
const OPERAND_BIAS: u32 = 0x1;
const OPERAND_LOD: u32 = 0x2;
const OPERAND_OFFSET: u32 = 0x10;

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
        let (mask, shift) = (
            self.constant(TEXTURE_INDEX_MASK),
            self.constant(SAMPLER_INDEX_SHIFT),
        );
        let texture_index = self.value(op::BITWISE_AND, self.u32, &[handle, mask]);
        let sampler_index = self.value(op::SHIFT_RIGHT_LOGICAL, self.u32, &[handle, shift]);
        let image = self.pool_entry(binding, types.image, texture_index, uniform);
        let sampler_type = self.module.ty(Type::Sampler);
        let sampler = self.pool_entry(texture_pool::SAMPLERS, sampler_type, sampler_index, uniform);
        let sampled = self.value(op::SAMPLED_IMAGE, types.sampled, &[image, sampler]);
        if !uniform {
            self.module.decorate(sampled, decoration::NON_UNIFORM, &[]);
        }
        Ok((sampled, types))
    }

    /// Entry `index` of texture-pool binding `binding` (an unsized array of
    /// `element`); a non-uniform index unless `uniform`.
    fn pool_entry(&mut self, binding: u32, element: u32, index: u32, uniform: bool) -> u32 {
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
        value
    }

    /// The instruction's operand values, in order (RZ reads zero).
    fn texture_values(&mut self, texture: &Texture, count: u32) -> Vec<u32> {
        self.texture_values_split(texture, count, None)
    }

    /// As `texture_values`; TEX.B (`coordinates` given) keeps all its
    /// coordinates in Ra when unpacked (bit 60 clear) or with five values
    /// or more.
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
        if texture.handle.is_none()
            && !texture.handle_value
            && usize::from(texture.bank) >= self.cbuf_used_sizes.len()
        {
            return Err(unsupported(index, "texture handle in an unbound bank"));
        }
        let results = match texture.op {
            TextureOp::Query => self.texture_query(index, texture)?,
            TextureOp::LevelQuery => self.texture_level_query(index, texture, &view)?,
            _ => self.texture_read(index, texture, &view)?,
        };
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
        for (n, value) in components.into_iter().enumerate() {
            let target = if texture.d2.0 == 255 || n < 2 {
                if texture.d.0 == 255 {
                    continue;
                }
                texture.d.0.saturating_add(n as u8)
            } else {
                texture.d2.0.saturating_add(n as u8 - 2)
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
        // Level modes: TEX implicit, zero, bias or explicit; TLD zero or
        // explicit; TLD4 none.
        let valid = match texture.op {
            TextureOp::Sample => texture.lod <= 3,
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
        let level_value = texture.lod >= 2;
        let handle_value = texture.handle_value;
        let count = view.coordinates
            + u32::from(handle_value)
            + u32::from(level_value)
            + u32::from(texture.offsets)
            + 2 * u32::from(texture.per_tap)
            + u32::from(texture.depth_compare);
        let values =
            self.texture_values_split(texture, count, handle_value.then_some(view.coordinates));
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
        let offsets: Option<Vec<u32>> = packed_offsets.map(|packed| {
            let four = self.constant(4);
            (0..spatial as u32)
                .map(|n| {
                    let at = self.constant(n * 4);
                    self.value(op::BIT_FIELD_S_EXTRACT, self.u32, &[packed, at, four])
                })
                .collect()
        });
        let gather = matches!(texture.op, TextureOp::Gather { .. });
        // Only fragment programs have derivatives for an implicit level.
        let implicit =
            self.stage == echora_shader_abi::Stage::Fragment && matches!(texture.lod, 0 | 2);
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
                if texture.lod == 2 {
                    mask |= OPERAND_BIAS;
                    let bias = level_operand.unwrap_or_else(|| self.float_constant(0.0));
                    let bias = self.value(op::BITCAST, self.f32, &[bias]);
                    operands.push(bias);
                }
            }
            // Elsewhere a bias over level zero is that level.
            TextureOp::Sample => {
                mask |= OPERAND_LOD;
                let lod = match level_operand {
                    Some(value) => self.value(op::BITCAST, self.f32, &[value]),
                    None => self.float_constant(0.0),
                };
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
            (TextureOp::Query | TextureOp::LevelQuery, _) => {
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
        if self.stage != echora_shader_abi::Stage::Fragment || view.arrayed {
            return Err(unsupported(
                index,
                "TMML outside a fragment program or on an array",
            ));
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
        let floats: Vec<u32> = values
            .iter()
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

    /// TXQ: width, height, depth or layers, and the level count. The
    /// instruction names no view type, so the texture is read as 2D.
    fn texture_query(&mut self, index: usize, texture: &Texture) -> Result<Vec<u32>, Unsupported> {
        self.module.capability(capability::IMAGE_QUERY);
        let view = View {
            dim: 1,
            arrayed: false,
            texture_type: 2,
            coordinates: 2,
        };
        let (sampled, types) = self.sampled_image(index, texture, &view, false, None)?;
        let level = if texture.a.0 == 255 {
            self.constant(0)
        } else {
            self.read_register(texture.a.0)
        };
        let image = self.value(op::IMAGE, types.image, &[sampled]);
        let uvec2 = self.module.ty(Type::Vector {
            component: self.u32,
            count: 2,
        });
        let size = self.value(op::IMAGE_QUERY_SIZE_LOD, uvec2, &[image, level]);
        let width = self.value(op::COMPOSITE_EXTRACT, self.u32, &[size, 0]);
        let height = self.value(op::COMPOSITE_EXTRACT, self.u32, &[size, 1]);
        let levels = self.value(op::IMAGE_QUERY_LEVELS, self.u32, &[image]);
        let one = self.constant(1);
        Ok(vec![width, height, one, levels])
    }
}
