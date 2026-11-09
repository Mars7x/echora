//! Vertex pulling: a vertex program fetches its generic inputs from the
//! guest's vertex buffers itself, by the draw's fetch table
//! (`echora_shader_abi::VertexFetch`, `VertexAttribute`, `VertexFormat`),
//! through the page table. One helper function decodes any format at run
//! time; the program's prologue fetches each location it reads once.

use std::collections::BTreeSet;

use echora_shader_abi::{PUSH_VERTEX_FETCH, Stage, Unsupported};

use super::{Emitter, unsupported};
use crate::decode::{Instruction, Op};
use crate::spirv::{Function, Type, glsl, op, storage};

/// SPIR-V built-ins.
const BUILTIN_VERTEX_INDEX: u32 = 42;
const BUILTIN_INSTANCE_INDEX: u32 = 43;

/// The memory-access operand for an aligned access.
const ALIGNED: u32 = 2;

/// 1.0's bits.
const ONE: u32 = 0x3f80_0000;

/// `VertexFormat` layouts and numeric types.
const LAYOUT_BITS8: u32 = 0;
const LAYOUT_BITS16: u32 = 1;
const LAYOUT_1010102: u32 = 3;
const LAYOUT_111110: u32 = 4;
const UNORM: u32 = 0;
const SNORM: u32 = 1;
const UINT: u32 = 2;
const SINT: u32 = 3;
const USCALED: u32 = 5;

impl Emitter {
    /// The generic input locations a vertex program's ALDs read.
    fn read_locations(code: &[Instruction]) -> BTreeSet<u32> {
        let mut locations = BTreeSet::new();
        for instruction in code {
            if let Op::Ald { address, count, .. } = instruction.op {
                for n in 0..u32::from(count) {
                    let at = u32::from(address) + n * 4;
                    if (0x80..0x280).contains(&at) {
                        locations.insert((at - 0x80) / 16);
                    }
                }
            }
        }
        locations
    }

    /// A vertex program's prologue: each generic input it reads, fetched.
    pub(super) fn vertex_prologue(&mut self, code: &[Instruction]) {
        if self.stage != Stage::Vertex {
            return;
        }
        let locations = Self::read_locations(code);
        if locations.is_empty() {
            return;
        }
        self.push_fields |= PUSH_VERTEX_FETCH;
        let fetch = self.fetch_function();
        let uvec4 = self.uvec4();
        let pointer = self.module.pointer(storage::FUNCTION, uvec4);
        for location in locations {
            let argument = self.constant(location);
            let value = self.value(op::FUNCTION_CALL, uvec4, &[fetch, argument]);
            let variable = self.function.variable(&mut self.module, pointer, None);
            self.function.emit(op::STORE, &[variable, value]);
            self.fetched.insert(location, variable);
            self.io.loaded_generics |= 1 << location;
        }
    }

    /// Word `word` of vertex input `location`, as the prologue fetched it.
    pub(super) fn vertex_fetch(
        &mut self,
        index: usize,
        location: u32,
        word: u32,
    ) -> Result<u32, Unsupported> {
        let variable = *self
            .fetched
            .get(&location)
            .ok_or_else(|| unsupported(index, "vertex input the prologue did not fetch"))?;
        let uvec4 = self.uvec4();
        let vector = self.value(op::LOAD, uvec4, &[variable]);
        Ok(self.value(op::COMPOSITE_EXTRACT, self.u32, &[vector, word]))
    }

    fn uvec4(&mut self) -> u32 {
        self.module.ty(Type::Vector {
            component: self.u32,
            count: 4,
        })
    }

    /// The helper `uvec4 fetch(uint location)`, emitted as its own
    /// function.
    fn fetch_function(&mut self) -> u32 {
        let uvec4 = self.uvec4();
        let function_type = self.module.ty(Type::Function {
            result: uvec4,
            parameters: vec![self.u32],
        });
        let (function, parameters) =
            Function::with_parameters(&mut self.module, uvec4, function_type, &[self.u32]);
        let id = function.id;
        let main = std::mem::replace(&mut self.function, function);
        let result = self.fetch_body(parameters[0]);
        self.function.emit(op::RETURN_VALUE, &[result]);
        let helper = std::mem::replace(&mut self.function, main);
        helper.finish(&mut self.module);
        id
    }

    /// A push-constant word of the vertex fetch fields (members 0-3).
    fn fetch_field(&mut self, member: u32) -> u32 {
        let block = self.push_block();
        let pointer = self.module.pointer(storage::PUSH_CONSTANT, self.u32);
        let member = self.constant(member);
        let element = self.value(op::ACCESS_CHAIN, pointer, &[block, member]);
        self.value(op::LOAD, self.u32, &[element])
    }

    /// The fetch helper's body; returns the fetched words.
    fn fetch_body(&mut self, location: u32) -> u32 {
        let uvec4 = self.uvec4();
        let pointer = self.module.pointer(storage::FUNCTION, uvec4);
        let result = self.function.variable(&mut self.module, pointer, None);
        let (zero, one_float) = (self.constant(0), self.constant(ONE));
        let unbound = self.vector(self.u32, &[zero, zero, zero, one_float]);
        self.function.emit(op::STORE, &[result, unbound]);
        let u64 = self.u64();
        let (low, high) = (self.fetch_field(0), self.fetch_field(1));
        let table = self.join(low, high);
        let base_instance = self.fetch_field(2);
        let attributes = self.fetch_field(3);
        let bit = self.value(op::SHIFT_RIGHT_LOGICAL, self.u32, &[attributes, location]);
        let one = self.constant(1);
        let bit = self.value(op::BITWISE_AND, self.u32, &[bit, one]);
        let listed = self.value(op::I_NOT_EQUAL, self.bool, &[bit, zero]);
        let zero64 = self.constant64(0);
        let has_table = self.value(op::I_NOT_EQUAL, self.bool, &[table, zero64]);
        let present = self.value(op::LOGICAL_AND, self.bool, &[listed, has_table]);
        let _ = self.when(present, |e| {
            // The entry: address (two words), stride, divisor, format, size.
            let location64 = e.widen(location, false);
            let thirty_two = e.constant64(32);
            let offset = e.value(op::I_MUL, u64, &[location64, thirty_two]);
            let entry = e.value(op::I_ADD, u64, &[table, offset]);
            let word_pointer = e.module.pointer(storage::PHYSICAL_STORAGE_BUFFER, e.u32);
            let words: Vec<u32> = (0..6u64)
                .map(|n| {
                    let at = e.constant64(n * 4);
                    let address = e.value(op::I_ADD, u64, &[entry, at]);
                    let pointer = e.value(op::CONVERT_U_TO_PTR, word_pointer, &[address]);
                    e.value(op::LOAD, e.u32, &[pointer, ALIGNED, 4])
                })
                .collect();
            let (address, stride, divisor, format, size) = (
                e.join(words[0], words[1]),
                words[2],
                words[3],
                words[4],
                words[5],
            );
            let seven = e.constant(7);
            let components = e.value(op::BITWISE_AND, e.u32, &[format, seven]);
            let formatted = e.value(op::I_NOT_EQUAL, e.bool, &[components, zero]);
            e.when(formatted, |e| {
                let decoded = e.decode_element(
                    address,
                    stride,
                    divisor,
                    format,
                    size,
                    components,
                    base_instance,
                );
                e.function.emit(op::STORE, &[result, decoded]);
                Ok(())
            })
        });
        self.value(op::LOAD, uvec4, &[result])
    }

    /// The word at guest address `va` (4-aligned), or 0 where `condition`
    /// fails or the page is unmapped.
    fn guest_word(&mut self, va: u32, condition: u32) -> u32 {
        let result = self.result_variables(1)[0];
        let _ = self.mapped(va, Some(condition), |e, device| {
            let pointer = e.module.pointer(storage::PHYSICAL_STORAGE_BUFFER, e.u32);
            let pointer = e.value(op::CONVERT_U_TO_PTR, pointer, &[device]);
            let word = e.value(op::LOAD, e.u32, &[pointer, ALIGNED, 4]);
            e.function.emit(op::STORE, &[result, word]);
            Ok(())
        });
        self.value(op::LOAD, self.u32, &[result])
    }

    /// `a == b` for u32 values, `b` a constant.
    fn equals(&mut self, a: u32, b: u32) -> u32 {
        let b = self.constant(b);
        self.value(op::I_EQUAL, self.bool, &[a, b])
    }

    fn pick(&mut self, condition: u32, a: u32, b: u32) -> u32 {
        self.value(op::SELECT, self.u32, &[condition, a, b])
    }

    /// One element's four words, decoded by `format`.
    #[allow(clippy::too_many_arguments)]
    fn decode_element(
        &mut self,
        address: u32,
        stride: u32,
        divisor: u32,
        format: u32,
        size: u32,
        components: u32,
        base_instance: u32,
    ) -> u32 {
        let u64 = self.u64();
        let (four, eight, fifteen) = (self.constant(4), self.constant(8), self.constant(15));
        let layout = self.value(op::SHIFT_RIGHT_LOGICAL, self.u32, &[format, four]);
        let layout = self.value(op::BITWISE_AND, self.u32, &[layout, fifteen]);
        let numeric = self.value(op::SHIFT_RIGHT_LOGICAL, self.u32, &[format, eight]);
        let numeric = self.value(op::BITWISE_AND, self.u32, &[numeric, fifteen]);
        let twelve = self.constant(12);
        let bgra = self.value(op::SHIFT_RIGHT_LOGICAL, self.u32, &[format, twelve]);
        let one = self.constant(1);
        let bgra = self.value(op::BITWISE_AND, self.u32, &[bgra, one]);
        let zero = self.constant(0);
        let bgra = self.value(op::I_NOT_EQUAL, self.bool, &[bgra, zero]);
        // The element: the vertex, or the instance over the divisor.
        let vertex = self.scalar_builtin(BUILTIN_VERTEX_INDEX);
        let instance = self.scalar_builtin(BUILTIN_INSTANCE_INDEX);
        let since = self.value(op::I_SUB, self.u32, &[instance, base_instance]);
        let safe = self.glsl(self.u32, glsl::U_MAX, &[divisor, one]);
        let steps = self.value(op::U_DIV, self.u32, &[since, safe]);
        let per_instance = self.value(op::I_ADD, self.u32, &[base_instance, steps]);
        let per_vertex = self.equals(divisor, 0);
        let element = self.pick(per_vertex, vertex, per_instance);
        let element = self.widen(element, false);
        let stride = self.widen(stride, false);
        let start = self.value(op::I_MUL, u64, &[element, stride]);
        let base = self.value(op::I_ADD, u64, &[address, start]);
        let size = self.widen(size, false);
        // Bytes per component (packed layouts: one word for all).
        let is8 = self.equals(layout, LAYOUT_BITS8);
        let is16 = self.equals(layout, LAYOUT_BITS16);
        let is1010102 = self.equals(layout, LAYOUT_1010102);
        let is111110 = self.equals(layout, LAYOUT_111110);
        let packed = self.value(op::LOGICAL_OR, self.bool, &[is1010102, is111110]);
        let (c1, c2, c4) = (self.constant(1), self.constant(2), self.constant(4));
        let bytes = self.pick(is16, c2, c4);
        let bytes = self.pick(is8, c1, bytes);
        let integer = {
            let uint = self.equals(numeric, UINT);
            let sint = self.equals(numeric, SINT);
            self.value(op::LOGICAL_OR, self.bool, &[uint, sint])
        };
        let mut words = Vec::new();
        for c in 0..4u32 {
            // Where component c's bits are.
            let (packed_shift, packed_bits_a, packed_bits_b) = match c {
                0 => (0, 10, 11),
                1 => (10, 10, 11),
                2 => (20, 10, 10),
                _ => (30, 2, 0),
            };
            let packed_shift_b = [0, 11, 22, 0][c as usize];
            let index = self.constant(c);
            let offset = self.value(op::I_MUL, self.u32, &[index, bytes]);
            let offset = self.pick(packed, zero, offset);
            let offset64 = self.widen(offset, false);
            let va = self.value(op::I_ADD, u64, &[base, offset64]);
            // Inside the stream: the component's last byte below `size`.
            let read = self.pick(packed, c4, bytes);
            let read64 = self.widen(read, false);
            let end = self.value(op::I_ADD, u64, &[start, offset64]);
            let end = self.value(op::I_ADD, u64, &[end, read64]);
            let inside = self.value(op::U_LESS_THAN_EQUAL, self.bool, &[end, size]);
            let present = self.value(op::U_LESS_THAN, self.bool, &[index, components]);
            let wanted = self.value(op::LOGICAL_AND, self.bool, &[inside, present]);
            let not_three = self.constant64(!3);
            let aligned = self.value(op::BITWISE_AND, u64, &[va, not_three]);
            let word = self.guest_word(aligned, wanted);
            let low = self.value(op::U_CONVERT, self.u32, &[va]);
            let three = self.constant(3);
            let lane = self.value(op::BITWISE_AND, self.u32, &[low, three]);
            let byte_shift = self.value(op::I_MUL, self.u32, &[lane, eight]);
            let shift_a = self.constant(packed_shift);
            let shift_b = self.constant(packed_shift_b);
            let packed_shift = self.pick(is111110, shift_b, shift_a);
            let shift = self.pick(packed, packed_shift, byte_shift);
            let bits_a = self.constant(packed_bits_a);
            let bits_b = self.constant(packed_bits_b);
            let packed_bits = self.pick(is111110, bits_b, bits_a);
            let plain_bits = self.value(op::I_MUL, self.u32, &[bytes, eight]);
            let bits = self.pick(packed, packed_bits, plain_bits);
            let value = self.decode_component(word, shift, bits, numeric, is111110);
            // Components the format lacks: (0, 0, 0, 1).
            let missing = if c == 3 {
                let one_float = self.constant(ONE);
                self.pick(integer, one, one_float)
            } else {
                zero
            };
            words.push(self.pick(present, value, missing));
        }
        // BGRA: components 0 and 2 swapped.
        let first = self.pick(bgra, words[2], words[0]);
        let third = self.pick(bgra, words[0], words[2]);
        words[0] = first;
        words[2] = third;
        self.vector(self.u32, &words)
    }

    /// A component of `bits` bits at `shift` in `word`, as its numeric
    /// type reads: float bits, or an integer.
    fn decode_component(
        &mut self,
        word: u32,
        shift: u32,
        bits: u32,
        numeric: u32,
        ufloat: u32,
    ) -> u32 {
        let raw = self.value(op::BIT_FIELD_U_EXTRACT, self.u32, &[word, shift, bits]);
        let signed = self.value(op::BIT_FIELD_S_EXTRACT, self.u32, &[word, shift, bits]);
        let all = self.constant(u32::MAX);
        let zero = self.constant(0);
        let one = self.constant(1);
        let as_float = |e: &mut Self, value: u32| e.value(op::BITCAST, e.u32, &[value]);
        // Normalized: raw over the largest value.
        let largest = self.value(op::BIT_FIELD_U_EXTRACT, self.u32, &[all, zero, bits]);
        let largest_f = self.value(op::CONVERT_U_TO_F, self.f32, &[largest]);
        let raw_f = self.value(op::CONVERT_U_TO_F, self.f32, &[raw]);
        let unorm = self.value(op::F_DIV, self.f32, &[raw_f, largest_f]);
        let unorm = as_float(self, unorm);
        let below = self.value(op::I_SUB, self.u32, &[bits, one]);
        let largest_s = self.value(op::BIT_FIELD_U_EXTRACT, self.u32, &[all, zero, below]);
        let largest_s = self.value(op::CONVERT_U_TO_F, self.f32, &[largest_s]);
        let signed_f = self.value(op::CONVERT_S_TO_F, self.f32, &[signed]);
        let snorm = self.value(op::F_DIV, self.f32, &[signed_f, largest_s]);
        let minus_one = self.float_constant(-1.0);
        let snorm = self.glsl(self.f32, glsl::F_MAX, &[snorm, minus_one]);
        let snorm = as_float(self, snorm);
        let uscaled = as_float(self, raw_f);
        let sscaled = as_float(self, signed_f);
        // Floats: a half, a single, or an unsigned 11- or 10-bit float.
        let vec2 = self.module.ty(Type::Vector {
            component: self.f32,
            count: 2,
        });
        let pair = self.glsl(vec2, glsl::UNPACK_HALF_2X16, &[raw]);
        let half = self.value(op::COMPOSITE_EXTRACT, self.f32, &[pair, 0]);
        let half = as_float(self, half);
        let small = self.small_float(raw, bits);
        let is16 = self.equals(bits, 16);
        let float = self.pick(is16, half, raw);
        let float = self.pick(ufloat, small, float);
        let mut value = float;
        for (kind, candidate) in [
            (UNORM, unorm),
            (SNORM, snorm),
            (UINT, raw),
            (SINT, signed),
            (USCALED, uscaled),
        ] {
            let matches = self.equals(numeric, kind);
            value = self.pick(matches, candidate, value);
        }
        let sscaled_kind = self.equals(numeric, USCALED + 1);
        self.pick(sscaled_kind, sscaled, value)
    }

    /// An unsigned 11- or 10-bit float (five exponent bits) as float bits.
    fn small_float(&mut self, raw: u32, bits: u32) -> u32 {
        let five = self.constant(5);
        let mantissa_bits = self.value(op::I_SUB, self.u32, &[bits, five]);
        let exponent = self.value(op::SHIFT_RIGHT_LOGICAL, self.u32, &[raw, mantissa_bits]);
        let zero = self.constant(0);
        let mantissa = self.value(
            op::BIT_FIELD_U_EXTRACT,
            self.u32,
            &[raw, zero, mantissa_bits],
        );
        let twenty_three = self.constant(23);
        let up = self.value(op::I_SUB, self.u32, &[twenty_three, mantissa_bits]);
        let fraction = self.value(op::SHIFT_LEFT_LOGICAL, self.u32, &[mantissa, up]);
        // Normal: the exponent rebased from 15 to 127.
        let rebase = self.constant(112);
        let biased = self.value(op::I_ADD, self.u32, &[exponent, rebase]);
        let biased = self.value(op::SHIFT_LEFT_LOGICAL, self.u32, &[biased, twenty_three]);
        let normal = self.value(op::BITWISE_OR, self.u32, &[biased, fraction]);
        // Infinity or NaN.
        let top = self.constant(0xff << 23);
        let special = self.value(op::BITWISE_OR, self.u32, &[top, fraction]);
        // Denormal: mantissa * 2^-14 / 2^mantissa_bits.
        let mantissa_f = self.value(op::CONVERT_U_TO_F, self.f32, &[mantissa]);
        let six = self.equals(mantissa_bits, 6);
        let (scale6, scale5) = (
            self.float_constant(2f32.powi(-20)),
            self.float_constant(2f32.powi(-19)),
        );
        let scale = self.value(op::SELECT, self.f32, &[six, scale6, scale5]);
        let denormal = self.value(op::F_MUL, self.f32, &[mantissa_f, scale]);
        let denormal = self.value(op::BITCAST, self.u32, &[denormal]);
        let is_zero = self.equals(exponent, 0);
        let is_special = self.equals(exponent, 31);
        let value = self.pick(is_special, special, normal);
        self.pick(is_zero, denormal, value)
    }
}
