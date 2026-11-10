//! Formatted storage image loads and stores (SULD.P, SUST.P) whose handle
//! is a constant-bank word. The image's format is the bound image's
//! (typeless, without a format in the SPIR-V); whether it is integer is
//! run-time state, so each image has a float and an unsigned view at one
//! binding and a specialization constant (`SpecMeaning::IntegerFormat`)
//! picks the view, one translation serving both. SURED reduces through
//! a descriptor of its own, declared R32_UINT (image atomics need a
//! format, and the runtime makes one view per binding).

use echora_shader_abi::{Descriptor, Unsupported};

use super::{Emitter, unsupported};
use crate::spirv::{Type, capability, op, storage};

/// SPIR-V decorations.
const DECORATION_SPEC_ID: u32 = 1;

/// The runtime's descriptor kinds, and flags.
const KIND_IMAGE_BUFFER: u32 = 3;
const KIND_IMAGE: u32 = 5;
const FLAG_WRITTEN: u32 = 1;
const FLAG_READ: u32 = 2;

/// SPIR-V's R32ui image format, and the runtime's (descriptor flags bits
/// 8-11, `Shader::ImageFormat`).
const FORMAT_R32UI: u32 = 33;
const FLAG_FORMAT_R32_UINT: u32 = 5 << 8;

/// SPIR-V's device scope.
const SCOPE_DEVICE: u32 = 1;

/// An image's views and the constant that picks between them.
#[derive(Clone, Copy, Debug)]
pub(super) struct ImageViews {
    float: u32,
    unsigned: u32,
    float_type: u32,
    unsigned_type: u32,
    integer: u32,
    /// The descriptor's index (before bindings are assigned).
    descriptor: usize,
}

/// SPIR-V's dimension and arrayed flag, the runtime's texture type and
/// descriptor kind, and the coordinate count of a surface dimension.
fn surface(dimension: u8) -> Option<(u32, bool, u32, u32, u32)> {
    Some(match dimension {
        0 => (0, false, 0, KIND_IMAGE, 1),
        1 => (5, false, 7, KIND_IMAGE_BUFFER, 1),
        3 => (1, false, 2, KIND_IMAGE, 2),
        4 => (1, true, 3, KIND_IMAGE, 3),
        5 => (2, false, 4, KIND_IMAGE, 3),
        _ => return None,
    })
}

impl Emitter {
    fn image_views(
        &mut self,
        bank: u8,
        offset: u16,
        dimension: u8,
        flag: u32,
    ) -> Option<ImageViews> {
        let key = (bank, offset, dimension);
        if let Some(&views) = self.images.get(&key) {
            self.descriptors[views.descriptor].flags |= flag;
            return Some(views);
        }
        let (dim, arrayed, texture_type, kind, _) = surface(dimension)?;
        match dim {
            0 => self.module.capability(capability::IMAGE_1D),
            5 => self.module.capability(capability::IMAGE_BUFFER),
            _ => {}
        }
        self.module
            .capability(capability::STORAGE_IMAGE_READ_WITHOUT_FORMAT);
        self.module
            .capability(capability::STORAGE_IMAGE_WRITE_WITHOUT_FORMAT);
        let view = |e: &mut Self, sampled_type: u32| {
            let ty = e.module.ty(Type::Image {
                sampled_type,
                dim,
                depth: 0,
                arrayed,
                multisampled: false,
                sampled: 2,
                format: 0,
            });
            let pointer = e.module.pointer(storage::UNIFORM_CONSTANT, ty);
            (
                e.module.global_variable(pointer, storage::UNIFORM_CONSTANT),
                ty,
            )
        };
        let (float, float_type) = view(self, self.f32);
        let (unsigned, unsigned_type) = view(self, self.u32);
        let descriptor = self.descriptors.len();
        self.add_descriptor(
            float,
            Descriptor {
                kind,
                count: 1,
                cbuf_index: u32::from(bank),
                cbuf_offset: u32::from(offset),
                texture_type,
                flags: flag,
                ..Descriptor::default()
            },
        );
        self.descriptor_aliases.push((descriptor, unsigned));
        self.interface.push(unsigned);
        let integer = self.module.bool_spec_constant();
        let id = self.image_spec_ids.len() as u32 + IMAGE_SPEC_BASE;
        self.module.decorate(integer, DECORATION_SPEC_ID, &[id]);
        self.image_spec_ids.push((descriptor, id));
        let views = ImageViews {
            float,
            unsigned,
            float_type,
            unsigned_type,
            integer,
            descriptor,
        };
        self.images.insert(key, views);
        Some(views)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn surface(
        &mut self,
        index: usize,
        store: bool,
        d: u8,
        a: u8,
        data: u8,
        bank: u8,
        offset: u16,
        dimension: u8,
        mask: u8,
    ) -> Result<(), Unsupported> {
        let (_, _, _, _, count) = surface(dimension)
            .ok_or_else(|| unsupported(index, format!("surface dimension {dimension}")))?;
        if usize::from(bank) >= self.cbuf_used_sizes.len() || mask == 0 {
            return Err(unsupported(index, "surface handle in an unbound bank"));
        }
        let flag = if store { FLAG_WRITTEN } else { FLAG_READ };
        let views = self
            .image_views(bank, offset, dimension, flag)
            .ok_or_else(|| unsupported(index, "surface dimension"))?;
        let coordinates: Vec<u32> = (0..count as u8)
            .map(|n| {
                if a == 255 {
                    self.constant(0)
                } else {
                    self.read_register(a.saturating_add(n))
                }
            })
            .collect();
        let coordinate = self.vector(self.u32, &coordinates);
        let components: Vec<usize> = (0..4).filter(|n| mask >> n & 1 != 0).collect();
        let vec4 = self.module.ty(Type::Vector {
            component: self.f32,
            count: 4,
        });
        let uvec4 = self.module.ty(Type::Vector {
            component: self.u32,
            count: 4,
        });
        if store {
            // The masked components from data.., the rest zero.
            let zero = self.constant(0);
            let mut words = [zero; 4];
            for (n, &component) in components.iter().enumerate() {
                words[component] = if data == 255 {
                    zero
                } else {
                    self.read_register(data.saturating_add(n as u8))
                };
            }
            let unsigned_texel = self.vector(self.u32, &words);
            let float_texel = self.value(op::BITCAST, vec4, &[unsigned_texel]);
            self.choose(views.integer, |e, integer| {
                let (variable, ty, texel) = if integer {
                    (views.unsigned, views.unsigned_type, unsigned_texel)
                } else {
                    (views.float, views.float_type, float_texel)
                };
                let image = e.value(op::LOAD, ty, &[variable]);
                e.function
                    .emit(op::IMAGE_WRITE, &[image, coordinate, texel]);
            });
        } else {
            let pointer = self.module.pointer(storage::FUNCTION, uvec4);
            let result = self.function.variable(&mut self.module, pointer, None);
            self.choose(views.integer, |e, integer| {
                let (variable, ty) = if integer {
                    (views.unsigned, views.unsigned_type)
                } else {
                    (views.float, views.float_type)
                };
                let image = e.value(op::LOAD, ty, &[variable]);
                let texel = if integer {
                    e.value(op::IMAGE_READ, uvec4, &[image, coordinate])
                } else {
                    let value = e.value(op::IMAGE_READ, vec4, &[image, coordinate]);
                    e.value(op::BITCAST, uvec4, &[value])
                };
                e.function.emit(op::STORE, &[result, texel]);
            });
            let texel = self.value(op::LOAD, uvec4, &[result]);
            for (n, &component) in components.iter().enumerate() {
                if d != 255 {
                    let value =
                        self.value(op::COMPOSITE_EXTRACT, self.u32, &[texel, component as u32]);
                    self.write_register(d.saturating_add(n as u8), value);
                }
            }
        }
        Ok(())
    }

    /// SURED: an atomic reduction into a 32-bit unsigned texel, through an
    /// R32ui view at the image's binding (image atomics need a format).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn surface_reduce(
        &mut self,
        index: usize,
        a: u8,
        data: u8,
        bank: u8,
        offset: u16,
        dimension: u8,
        operation: u8,
    ) -> Result<(), Unsupported> {
        let opcode = match operation {
            0 => op::ATOMIC_I_ADD,
            1 => op::ATOMIC_U_MIN,
            2 => op::ATOMIC_U_MAX,
            5 => op::ATOMIC_AND,
            6 => op::ATOMIC_OR,
            7 => op::ATOMIC_XOR,
            _ => return Err(unsupported(index, format!("surface reduction {operation}"))),
        };
        let (dim, arrayed, texture_type, kind, count) = surface(dimension)
            .filter(|_| matches!(dimension, 1 | 3))
            .ok_or_else(|| {
                unsupported(index, format!("surface reduction dimension {dimension}"))
            })?;
        if usize::from(bank) >= self.cbuf_used_sizes.len() {
            return Err(unsupported(index, "surface handle in an unbound bank"));
        }
        let key = (bank, offset, dimension);
        let variable = match self.atomic_images.get(&key) {
            Some(&variable) => variable,
            None => {
                let ty = self.module.ty(Type::Image {
                    sampled_type: self.u32,
                    dim,
                    depth: 0,
                    arrayed,
                    multisampled: false,
                    sampled: 2,
                    format: FORMAT_R32UI,
                });
                let pointer = self.module.pointer(storage::UNIFORM_CONSTANT, ty);
                let variable = self
                    .module
                    .global_variable(pointer, storage::UNIFORM_CONSTANT);
                self.add_descriptor(
                    variable,
                    Descriptor {
                        kind,
                        count: 1,
                        cbuf_index: u32::from(bank),
                        cbuf_offset: u32::from(offset),
                        texture_type,
                        flags: FLAG_READ | FLAG_WRITTEN | FLAG_FORMAT_R32_UINT,
                        ..Descriptor::default()
                    },
                );
                if dim == 5 {
                    self.module.capability(capability::IMAGE_BUFFER);
                }
                self.atomic_images.insert(key, variable);
                variable
            }
        };
        let coordinates: Vec<u32> = (0..count as u8)
            .map(|n| {
                if a == 255 {
                    self.constant(0)
                } else {
                    self.read_register(a.saturating_add(n))
                }
            })
            .collect();
        let coordinate = self.vector(self.u32, &coordinates);
        let value = self.read_register(data);
        let texel_pointer = self.module.pointer(storage::IMAGE, self.u32);
        let sample = self.constant(0);
        let pointer = self.value(
            op::IMAGE_TEXEL_POINTER,
            texel_pointer,
            &[variable, coordinate, sample],
        );
        let scope = self.constant(SCOPE_DEVICE);
        let relaxed = self.constant(0);
        self.value(opcode, self.u32, &[pointer, scope, relaxed, value]);
        Ok(())
    }

    /// Runs `body(true)` where `condition` holds, else `body(false)`.
    /// Values known before the selection stay known in both bodies, and
    /// after it unless a body replaced them.
    pub(super) fn choose(&mut self, condition: u32, mut body: impl FnMut(&mut Self, bool)) {
        let (yes, no, merge) = (self.module.id(), self.module.id(), self.module.id());
        self.function
            .emit(op::SELECTION_MERGE, &[merge, crate::spirv::NONE]);
        self.function
            .emit(op::BRANCH_CONDITIONAL, &[condition, yes, no]);
        let before = self.function.block();
        self.dominators.push(before);
        self.function.label(yes);
        body(self, true);
        self.function.emit(op::BRANCH, &[merge]);
        self.function.label(no);
        body(self, false);
        self.function.emit(op::BRANCH, &[merge]);
        self.dominators.pop();
        self.function.label(merge);
        self.carry_known(before);
    }
}

/// Specialization constant ids for images start here.
pub(super) const IMAGE_SPEC_BASE: u32 = 32;
