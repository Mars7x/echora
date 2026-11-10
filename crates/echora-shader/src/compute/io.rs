//! Graphics inputs and outputs: the attribute space ALD, AST and IPA
//! address, by hardware byte address (0x6c point size, 0x70 position,
//! 0x80 plus 16 times n generic vector n, 0x2f8 instance id, 0x2fc vertex
//! id), and the fragment outputs a program leaves in its registers at
//! exit.
//!
//! Generic vectors between stages are `vec4` variables at location n.
//! Vertex inputs are fetched by the program itself (`vertex_fetch`):
//! through the page table from a per-draw table of attribute addresses,
//! strides and formats, so no translation depends on attribute formats.

use std::collections::BTreeMap;

use echora_shader_abi::{PUSH_RESOLUTION_SCALE, Stage, Unsupported};

use super::{Emitter, unsupported};
use crate::decode::Pred;
use crate::header::Interpolation;
use crate::spirv::{Type, capability, decoration, op, storage};

/// SPIR-V built-ins used here.
const BUILTIN_POSITION: u32 = 0;
const BUILTIN_POINT_SIZE: u32 = 1;
const BUILTIN_FRAG_COORD: u32 = 15;
const BUILTIN_FRONT_FACING: u32 = 17;
const BUILTIN_SAMPLE_MASK: u32 = 20;
const BUILTIN_FRAG_DEPTH: u32 = 22;
const BUILTIN_VERTEX_INDEX: u32 = 42;
const BUILTIN_INSTANCE_INDEX: u32 = 43;
const BUILTIN_BASE_INSTANCE: u32 = 4425;

/// Decorations.
const DECORATION_NO_PERSPECTIVE: u32 = 13;
const DECORATION_FLAT: u32 = 14;
const DECORATION_CENTROID: u32 = 16;
const DECORATION_LOCATION: u32 = 30;
const DECORATION_COMPONENT: u32 = 31;

/// A program's inputs and outputs so far.
#[derive(Debug, Default)]
pub(super) struct Io {
    /// Generic input and output vectors, by location.
    inputs: BTreeMap<u32, u32>,
    /// Fragment inputs of a vector whose components the header interpolates
    /// differently: one scalar per component, by (location, component).
    component_inputs: BTreeMap<(u32, u32), u32>,
    outputs: BTreeMap<u32, u32>,
    /// Built-in outputs, by built-in.
    builtin_outputs: BTreeMap<u32, u32>,
    /// Fragment inputs read at the centroid.
    centroid: Vec<u32>,
    pub(super) loaded_generics: u32,
    pub(super) stored_generics: u32,
    /// Bit (address / 4) of every attribute word written.
    pub(super) stores: [u64; 8],
}

impl Emitter {
    pub(super) fn vec4(&mut self) -> u32 {
        self.module.ty(Type::Vector {
            component: self.f32,
            count: 4,
        })
    }

    /// The generic vector `location` as an input or output variable.
    fn generic(&mut self, location: u32, output: bool) -> u32 {
        let existing = if output {
            self.io.outputs.get(&location)
        } else {
            self.io.inputs.get(&location)
        };
        if let Some(&variable) = existing {
            return variable;
        }
        let vec4 = self.vec4();
        let class = if output {
            storage::OUTPUT
        } else {
            storage::INPUT
        };
        let pointer = self.module.pointer(class, vec4);
        let variable = self.module.global_variable(pointer, class);
        self.module
            .decorate(variable, DECORATION_LOCATION, &[location]);
        if !output && self.stage == Stage::Fragment {
            let mode = self.vector_mode(location);
            self.decorate_interpolation(variable, mode);
        }
        self.interface.push(variable);
        if output {
            self.io.outputs.insert(location, variable);
        } else {
            self.io.inputs.insert(location, variable);
        }
        variable
    }

    /// The interpolation the header gives a fragment input vector's
    /// components (the first one used; perspective when none is).
    fn vector_mode(&self, location: u32) -> Interpolation {
        self.header.map_or(Interpolation::Perspective, |header| {
            (0..4)
                .map(|c| header.imap[(location * 4 + c) as usize])
                .find(|&mode| mode != Interpolation::Unused)
                .unwrap_or(Interpolation::Perspective)
        })
    }

    /// Whether the header interpolates a fragment input vector's used
    /// components in more than one way (it maps each component apart).
    fn mixed_vector(&self, location: u32) -> bool {
        let Some(header) = self.header.filter(|_| self.stage == Stage::Fragment) else {
            return false;
        };
        let used: Vec<Interpolation> = (0..4)
            .map(|c| header.imap[(location * 4 + c) as usize])
            .filter(|&mode| mode != Interpolation::Unused)
            .collect();
        used.windows(2).any(|pair| pair[0] != pair[1])
    }

    /// The interpolation a fragment input component is declared with:
    /// its own in a mixed vector, else its vector's.
    fn declared_mode(&self, location: u32, component: u32) -> Interpolation {
        let own = self
            .header
            .map(|header| header.imap[(location * 4 + component) as usize])
            .filter(|&mode| mode != Interpolation::Unused);
        match own {
            Some(mode) if self.mixed_vector(location) => mode,
            _ => self.vector_mode(location),
        }
    }

    fn decorate_interpolation(&mut self, variable: u32, mode: Interpolation) {
        match mode {
            Interpolation::Constant => {
                self.module.decorate(variable, DECORATION_FLAT, &[]);
            }
            Interpolation::ScreenLinear => {
                self.module
                    .decorate(variable, DECORATION_NO_PERSPECTIVE, &[]);
            }
            _ => {}
        }
    }

    /// Component `component` of a mixed fragment input vector, as a scalar
    /// input of its own (Location and Component) with its interpolation.
    fn component_input(&mut self, location: u32, component: u32) -> u32 {
        if let Some(&variable) = self.io.component_inputs.get(&(location, component)) {
            return variable;
        }
        let pointer = self.module.pointer(storage::INPUT, self.f32);
        let variable = self.module.global_variable(pointer, storage::INPUT);
        self.module
            .decorate(variable, DECORATION_LOCATION, &[location]);
        self.module
            .decorate(variable, DECORATION_COMPONENT, &[component]);
        let mode = self.declared_mode(location, component);
        self.decorate_interpolation(variable, mode);
        self.interface.push(variable);
        self.io
            .component_inputs
            .insert((location, component), variable);
        variable
    }

    /// A built-in output variable of type `ty`, declared once.
    fn builtin_output(&mut self, builtin: u32, ty: u32) -> u32 {
        if let Some(&variable) = self.io.builtin_outputs.get(&builtin) {
            return variable;
        }
        let pointer = self.module.pointer(storage::OUTPUT, ty);
        let variable = self.module.global_variable(pointer, storage::OUTPUT);
        self.module
            .decorate(variable, decoration::BUILT_IN, &[builtin]);
        if builtin == BUILTIN_POSITION {
            // The same guest math in two programs (a depth prepass and
            // its lit pass) must give the same position, as on the
            // console: every float operation is NoContraction, and the
            // position invariant.
            self.module.decorate(variable, decoration::INVARIANT, &[]);
        }
        self.io.builtin_outputs.insert(builtin, variable);
        self.interface.push(variable);
        variable
    }

    /// Component `component` of a `vec4` variable.
    fn component_pointer(&mut self, variable: u32, class: u32, component: u32) -> u32 {
        let pointer = self.module.pointer(class, self.f32);
        let component = self.constant(component);
        self.value(op::ACCESS_CHAIN, pointer, &[variable, component])
    }

    /// One word of the attribute space, read.
    fn attribute(
        &mut self,
        index: usize,
        address: u16,
        centroid: bool,
    ) -> Result<u32, Unsupported> {
        let address = u32::from(address);
        let word = (address / 4) % 4;
        Ok(match (self.stage, address) {
            (Stage::Vertex, 0x80..=0x27f) => {
                let location = (address - 0x80) / 16;
                self.io.loaded_generics |= 1 << location;
                return self.vertex_fetch(index, location, word);
            }
            (_, 0x80..=0x27f) => {
                let location = (address - 0x80) / 16;
                let mixed = self.mixed_vector(location);
                let variable = if mixed {
                    self.component_input(location, word)
                } else {
                    self.generic(location, false)
                };
                if centroid && !self.io.centroid.contains(&variable) {
                    self.module.decorate(variable, DECORATION_CENTROID, &[]);
                    self.io.centroid.push(variable);
                }
                self.io.loaded_generics |= 1 << location;
                let pointer = if mixed {
                    variable
                } else {
                    self.component_pointer(variable, storage::INPUT, word)
                };
                let value = self.value(op::LOAD, self.f32, &[pointer]);
                self.value(op::BITCAST, self.u32, &[value])
            }
            // The hardware's vertex id includes the draw's base vertex (GL's
            // gl_VertexID; NAK maps NIR's vertex_id to a[0x2fc] unlowered,
            // yuzu reads Vulkan's VertexIndex), its instance id excludes
            // the base instance (NAK's instance_id; yuzu subtracts it).
            (Stage::Vertex, 0x2fc) => self.scalar_builtin(BUILTIN_VERTEX_INDEX),
            (Stage::Vertex, 0x2f8) => {
                self.module.capability(capability::DRAW_PARAMETERS);
                let id = self.scalar_builtin(BUILTIN_INSTANCE_INDEX);
                let base = self.scalar_builtin(BUILTIN_BASE_INSTANCE);
                self.value(op::I_SUB, self.u32, &[id, base])
            }
            (Stage::Fragment, 0x3fc) => {
                // Front facing: all ones, else zero.
                let variable = self.builtin_variable(BUILTIN_FRONT_FACING, self.bool);
                let front = self.value(op::LOAD, self.bool, &[variable]);
                let (all, zero) = (self.constant(u32::MAX), self.constant(0));
                self.value(op::SELECT, self.u32, &[front, all, zero])
            }
            (Stage::Fragment, 0x70..=0x7f) => {
                let vec4 = self.vec4();
                let variable = self.builtin_variable(BUILTIN_FRAG_COORD, vec4);
                let pointer = self.component_pointer(variable, storage::INPUT, word);
                let mut value = self.value(op::LOAD, self.f32, &[pointer]);
                if word < 2 {
                    // Rendering is at the resolution scale; the guest sees
                    // its own resolution.
                    let inverse = self.resolution_scale(2 + word);
                    value = self.value(op::F_MUL, self.f32, &[value, inverse]);
                }
                self.value(op::BITCAST, self.u32, &[value])
            }
            _ => {
                return Err(unsupported(index, format!("attribute {address:#x} read")));
            }
        })
    }

    /// Word `word` of the push constants' resolution scale (x, y, inverse
    /// x, inverse y).
    fn resolution_scale(&mut self, word: u32) -> u32 {
        let variable = self.push_block();
        self.push_fields |= PUSH_RESOLUTION_SCALE;
        let pointer = self.module.pointer(storage::PUSH_CONSTANT, self.f32);
        let (member, word) = (self.constant(4), self.constant(word));
        let element = self.value(op::ACCESS_CHAIN, pointer, &[variable, member, word]);
        self.value(op::LOAD, self.f32, &[element])
    }

    /// One word of the attribute space, written.
    fn store_attribute(
        &mut self,
        index: usize,
        address: u16,
        value: u32,
    ) -> Result<(), Unsupported> {
        let address = u32::from(address);
        let word = (address / 4) % 4;
        // A per-primitive hint with no meaning here.
        if address == 0x5c {
            return Ok(());
        }
        let float = self.value(op::BITCAST, self.f32, &[value]);
        let pointer = match address {
            0x6c => self.builtin_output(BUILTIN_POINT_SIZE, self.f32),
            0x70..=0x7f => {
                let vec4 = self.vec4();
                let variable = self.builtin_output(BUILTIN_POSITION, vec4);
                self.component_pointer(variable, storage::OUTPUT, word)
            }
            0x80..=0x27f => {
                let location = (address - 0x80) / 16;
                let variable = self.generic(location, true);
                self.io.stored_generics |= 1 << location;
                self.component_pointer(variable, storage::OUTPUT, word)
            }
            _ => {
                return Err(unsupported(index, format!("attribute {address:#x} write")));
            }
        };
        let bit = address / 4;
        self.io.stores[(bit / 64) as usize] |= 1 << (bit % 64);
        self.function.emit(op::STORE, &[pointer, float]);
        Ok(())
    }

    /// ALD: `count` words from `address` into d...
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attribute_load(
        &mut self,
        index: usize,
        d: u8,
        offset: u8,
        vertex: u8,
        address: u16,
        count: u8,
        patch_or_output: bool,
    ) -> Result<(), Unsupported> {
        if offset != 255 || vertex != 255 || patch_or_output {
            return Err(unsupported(index, "indexed, per-vertex or output ALD"));
        }
        for n in 0..count {
            let value = self.attribute(index, address + u16::from(n) * 4, false)?;
            if d != 255 {
                self.write_register(d.saturating_add(n), value);
            }
        }
        Ok(())
    }

    /// AST: `count` words from data.. to `address`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attribute_store(
        &mut self,
        index: usize,
        data: u8,
        offset: u8,
        vertex: u8,
        address: u16,
        count: u8,
        patch: bool,
    ) -> Result<(), Unsupported> {
        if offset != 255 || vertex != 255 || patch {
            return Err(unsupported(index, "indexed or per-vertex AST"));
        }
        for n in 0..count {
            let value = if data == 255 {
                self.constant(0)
            } else {
                self.read_register(data.saturating_add(n))
            };
            self.store_attribute(index, address + u16::from(n) * 4, value)?;
        }
        Ok(())
    }

    /// IPA: an interpolated fragment input.
    pub(super) fn interpolate(
        &mut self,
        index: usize,
        d: u8,
        address: u16,
        location: u8,
    ) -> Result<(), Unsupported> {
        if self.stage != Stage::Fragment {
            return Err(unsupported(index, "IPA outside a fragment program"));
        }
        if location > 1 {
            return Err(unsupported(index, "IPA at an offset"));
        }
        let value = self.attribute(index, address, location == 1)?;
        let value = self.as_interpolated(address, value);
        self.write_register(d, value);
        Ok(())
    }

    /// Whether the attribute word at `address` reads the same for every
    /// fragment of a primitive: a system value (front facing, ...), or a
    /// generic the header maps as constant (flat).
    pub(super) fn flat_input(&self, address: u16) -> bool {
        !(0x80..0x280).contains(&address) || {
            let component = u32::from((address - 0x80) / 4);
            self.declared_mode(component / 4, component % 4) == Interpolation::Constant
        }
    }

    /// An input as IPA reads it: a generic the header interpolates with
    /// perspective (or leaves unused) still multiplied by 1/w (Vulkan's
    /// FragCoord.w), as the hardware interpolates attr/w linearly and the
    /// program divides by a[0x7c] itself (as NAK on SM70+ and yuzu read
    /// it: IPA(attr) * rcp(IPA(w))). Vulkan hands over attr already
    /// corrected.
    fn as_interpolated(&mut self, address: u16, value: u32) -> u32 {
        if !(0x80..0x280).contains(&address) {
            return value;
        }
        // As the input is declared (what Vulkan delivered).
        let component = u32::from((address - 0x80) / 4);
        let mode = self.declared_mode(component / 4, component % 4);
        if mode != Interpolation::Perspective {
            return value;
        }
        let vec4 = self.vec4();
        let variable = self.builtin_variable(BUILTIN_FRAG_COORD, vec4);
        let pointer = self.component_pointer(variable, storage::INPUT, 3);
        let inverse_w = self.value(op::LOAD, self.f32, &[pointer]);
        let float = self.value(op::BITCAST, self.f32, &[value]);
        let product = self.exact(op::F_MUL, &[float, inverse_w]);
        self.value(op::BITCAST, self.u32, &[product])
    }

    /// IPA with the address offset by a uniform register: each generic
    /// component the header's input map uses, read, and the one at the
    /// address selected (zero where none is).
    pub(super) fn interpolate_indexed(
        &mut self,
        index: usize,
        d: u8,
        base: u16,
        location: u8,
        register: u8,
    ) -> Result<(), Unsupported> {
        let Some(header) = self.header.filter(|_| self.stage == Stage::Fragment) else {
            return Err(unsupported(index, "indexed IPA outside a fragment program"));
        };
        if location > 1 {
            return Err(unsupported(index, "IPA at an offset"));
        }
        let offset = self.read_uniform(register);
        let base = self.constant(u32::from(base));
        let address = self.value(op::I_ADD, self.u32, &[base, offset]);
        let mut value = self.constant(0);
        for component in 0..128u16 {
            if header.imap[usize::from(component)] == Interpolation::Unused {
                continue;
            }
            let at = 0x80 + component * 4;
            let read = self.attribute(index, at, location == 1)?;
            let read = self.as_interpolated(at, read);
            let here = self.constant(u32::from(at));
            let matches = self.value(op::I_EQUAL, self.bool, &[address, here]);
            value = self.value(op::SELECT, self.u32, &[matches, read, value]);
        }
        self.write_register(d, value);
        Ok(())
    }

    /// KILL: demote where `condition` holds.
    pub(super) fn kill(&mut self, condition: Pred) -> Result<(), Unsupported> {
        self.module
            .capability(capability::DEMOTE_TO_HELPER_INVOCATION);
        if condition.is_true() {
            self.function.emit(op::DEMOTE_TO_HELPER_INVOCATION, &[]);
            return Ok(());
        }
        let condition = self.read_predicate(condition);
        self.when(condition, |e| {
            e.function.emit(op::DEMOTE_TO_HELPER_INVOCATION, &[]);
            Ok(())
        })
    }

    /// A fragment program's outputs, from its registers at exit: four
    /// registers per render target the header enables, then the sample
    /// mask's place and the depth.
    pub(super) fn fragment_outputs(&mut self) {
        let Some(header) = self.header else {
            return;
        };
        if self.stage != Stage::Fragment {
            return;
        }
        let mut register = 0u8;
        for target in 0..8u32 {
            if header.targets >> (target * 4) & 0xf == 0 {
                continue;
            }
            let variable = self.generic(target, true);
            let components: Vec<u32> = (0..4)
                .map(|n| {
                    let bits = self.read_register(register + n);
                    self.value(op::BITCAST, self.f32, &[bits])
                })
                .collect();
            let vector = self.vector(self.f32, &components);
            self.function.emit(op::STORE, &[variable, vector]);
            self.frag_colors |= 1 << target;
            register += 4;
        }
        if header.writes_sample_mask || header.writes_depth {
            if header.writes_sample_mask {
                let one = self.constant(1);
                let array = self.module.ty(Type::Array {
                    element: self.u32,
                    length: one,
                });
                let variable = self.builtin_output(BUILTIN_SAMPLE_MASK, array);
                let pointer = self.module.pointer(storage::OUTPUT, self.u32);
                let zero = self.constant(0);
                let element = self.value(op::ACCESS_CHAIN, pointer, &[variable, zero]);
                let mask = self.read_register(register);
                self.function.emit(op::STORE, &[element, mask]);
            }
            register += 1;
            if header.writes_depth {
                let variable = self.builtin_output(BUILTIN_FRAG_DEPTH, self.f32);
                let bits = self.read_register(register);
                let depth = self.value(op::BITCAST, self.f32, &[bits]);
                self.function.emit(op::STORE, &[variable, depth]);
            }
        }
    }
}
