//! HMMA: the warp-wide 16x8 matrix multiply-accumulate, on the subgroup.
//!
//! Each lane holds the PTX fragment layout (g = lane / 4, t = lane % 4):
//! A (16xK halves, two per register) rows g and g + 8, columns 2t, 2t + 1,
//! and with K = 16 the same plus 8 (registers: row g, row g + 8, then the
//! upper columns); B (Kx8) rows 2t, 2t + 1 (and + 8) of column g; C and D
//! rows g and g + 8, columns 2t and 2t + 1 (four 32-bit floats, or two
//! registers of halves). A lane gathers its A row from its own quad and
//! its B columns from the quads of lanes 8t and 8t + 4 by subgroup
//! shuffles.
//!
//! Arithmetic: half products are exact in 32-bit floats; they are summed
//! into the accumulator in order of k with 32-bit round-to-nearest, and a
//! half result is rounded once at the end.
//!
//! `MatrixPath::Exact` instead models the A100's tensor cores as published
//! (Fasi, Higham, Mikaitis and Pranesh, 2021; Khattak and Mikaitis, 2025):
//! each block of eight k (one block for HMMA.1688, two for HMMA.16816,
//! the first block's result the second's accumulator) adds its eight exact
//! products and the accumulator in one step. Every addend is aligned to
//! the largest exponent among them and truncated toward zero to 24 bits
//! below that exponent's leading bit, the aligned values are summed
//! exactly, and the sum is normalised once: truncated to a 32-bit float,
//! or, for the last block of a half result, rounded to the nearest half
//! (through a round-to-odd 32-bit value, so rounding happens once). What
//! the papers do not settle is inferred: the sign-magnitude truncation,
//! the accumulator of the second block being the first block's 32-bit
//! result, results past the 32-bit range (the largest finite value, as
//! truncation gives) and below it (subnormals, truncated), and blocks with
//! an infinity or NaN (their plain 32-bit sum).

use echora_shader_abi::Unsupported;

use std::collections::{BTreeMap, BTreeSet};

use super::{Block, Emitter, SCOPE_SUBGROUP};
use crate::decode::{Instruction, Op};
use crate::spirv::{Type, capability, glsl, op, storage};

/// Elements of a 16x16 matrix.
const MATRIX_ELEMENTS: u32 = 256;

/// SPIR-V's SubgroupId built-in.
const BUILTIN_SUBGROUP_ID: u32 = 40;

/// OpFConvert.
const F_CONVERT: u16 = 115;

/// HMMAs (both HMMA.16816 or both HMMA.1688) run in pairs on one 16x16x16 multiply: two in a row in a
/// block, both unguarded, with the same A and accumulator type, the second
/// reading nothing the first writes (its d is not among the second's a, b
/// and c registers). The first's index maps to the second's (d, b, c).
#[derive(Debug, Default)]
pub(super) struct Pairs {
    firsts: BTreeMap<usize, (u8, u8, u8)>,
    seconds: BTreeSet<usize>,
}

/// The registers `count` from `base` (none for RZ).
fn registers(base: u8, count: u8) -> std::ops::Range<u16> {
    if base == 255 {
        0..0
    } else {
        u16::from(base)..u16::from(base) + u16::from(count)
    }
}

fn overlap(x: &std::ops::Range<u16>, y: &std::ops::Range<u16>) -> bool {
    x.start < y.end && y.start < x.end
}

pub(super) fn pairs(code: &[Instruction], blocks: &[Block]) -> Pairs {
    let starts: BTreeSet<usize> = blocks.iter().map(|block| block.start).collect();
    let mut pairs = Pairs::default();
    let mut index = 0;
    while index + 1 < code.len() {
        let (first, second) = (&code[index], &code[index + 1]);
        let paired = match (first.op, second.op) {
            (
                Op::Hmma {
                    d, a, k16, single, ..
                },
                Op::Hmma {
                    a: a2,
                    b: b2,
                    c: c2,
                    k16: k16_2,
                    single: single2,
                    ..
                },
            ) if a == a2
                && k16 == k16_2
                && single == single2
                && first.guard.is_true()
                && second.guard.is_true()
                && !starts.contains(&(index + 1)) =>
            {
                let wide = if single { 4 } else { 2 };
                let written = registers(d.0, wide);
                !overlap(&written, &registers(a2.0, 4))
                    && !overlap(&written, &registers(b2.0, 2))
                    && !overlap(&written, &registers(c2.0, wide))
            }
            _ => false,
        };
        if paired {
            if let Op::Hmma { d, b, c, .. } = second.op {
                pairs.firsts.insert(index, (d.0, b.0, c.0));
                pairs.seconds.insert(index + 1);
            }
            index += 2;
        } else {
            index += 1;
        }
    }
    pairs
}

impl Emitter {
    /// A register's two halves as 32-bit floats.
    fn halves(&mut self, bits: u32) -> [u32; 2] {
        self.module.capability(capability::FLOAT16);
        let f16 = self.module.ty(Type::Float { width: 16 });
        let pair = self.module.ty(Type::Vector {
            component: f16,
            count: 2,
        });
        let both = self.value(op::BITCAST, pair, &[bits]);
        [0, 1].map(|n| {
            let half = self.value(op::COMPOSITE_EXTRACT, f16, &[both, n]);
            self.value(F_CONVERT, self.f32, &[half])
        })
    }

    /// Two 32-bit floats as a register of halves (the first low).
    fn pack(&mut self, values: [u32; 2]) -> u32 {
        let f16 = self.module.ty(Type::Float { width: 16 });
        let pair = self.module.ty(Type::Vector {
            component: f16,
            count: 2,
        });
        let halves = values.map(|value| self.value(F_CONVERT, f16, &[value]));
        let vector = self.value(op::COMPOSITE_CONSTRUCT, pair, &halves);
        self.value(op::BITCAST, self.u32, &[vector])
    }

    fn register_or_zero(&mut self, n: u8) -> u32 {
        if n == 255 {
            self.constant(0)
        } else {
            self.read_register(n)
        }
    }

    /// The workgroup scratch for cooperative matrices of `component`
    /// (`f16` for A, B and a half accumulator, `f32` for a single one):
    /// per warp, three 16x16 matrices (A, B, C) row-major. Declared once
    /// per component type.
    fn matrix_scratch(&mut self, component: u32) -> u32 {
        if let Some(&variable) = self.matrix_scratch.get(&component) {
            return variable;
        }
        let threads: u32 = self.layout.workgroup.iter().product();
        let warps = threads.div_ceil(32).max(1);
        let length = self.constant(warps * MATRIX_ELEMENTS * 3);
        let array = self.module.ty(Type::Array {
            element: component,
            length,
        });
        let pointer = self.module.pointer(storage::WORKGROUP, array);
        let variable = self.module.global_variable(pointer, storage::WORKGROUP);
        self.interface.push(variable);
        self.matrix_scratch.insert(component, variable);
        variable
    }

    /// A pointer to element `(row, column)` of matrix `which` (0 A, 1 B, 2
    /// C) of this warp's scratch region at `base`.
    #[allow(clippy::too_many_arguments)]
    fn scratch_element(
        &mut self,
        scratch: u32,
        component: u32,
        base: u32,
        which: u32,
        row: u32,
        column: u32,
    ) -> u32 {
        let sixteen = self.constant(16);
        let offset = self.constant(which * MATRIX_ELEMENTS);
        let at = self.value(op::I_MUL, self.u32, &[row, sixteen]);
        let at = self.value(op::I_ADD, self.u32, &[at, column]);
        let at = self.value(op::I_ADD, self.u32, &[at, offset]);
        let at = self.value(op::I_ADD, self.u32, &[at, base]);
        let pointer = self.module.pointer(storage::WORKGROUP, component);
        self.value(op::ACCESS_CHAIN, pointer, &[scratch, at])
    }

    /// A subgroup barrier that orders workgroup memory.
    fn subgroup_barrier(&mut self) {
        let scope = self.constant(SCOPE_SUBGROUP);
        let semantics = self.constant(0x8 | 0x100);
        self.function
            .emit(op::CONTROL_BARRIER, &[scope, scope, semantics]);
    }

    /// HMMA (HMMA.1688 with K padded by zeros) on a 16x16x16 cooperative matrix multiply: the lanes
    /// write their fragments (PTX layout) to the warp's scratch, the
    /// subgroup multiplies, and each lane reads its part of D back. `halves`
    /// holds one HMMA's (d, b, c), for columns 0-7 (B's upper eight
    /// columns then zero), or two sharing A, the second's in columns 8-15.
    fn cooperative_multiply(
        &mut self,
        index: usize,
        a: u8,
        halves_in: &[(u8, u8, u8)],
        k16: bool,
        single: bool,
    ) -> Result<(), Unsupported> {
        self.module.capability(capability::COOPERATIVE_MATRIX);
        self.module.extension("SPV_KHR_cooperative_matrix");
        self.module.capability(capability::FLOAT16);
        let f16 = self.module.ty(Type::Float { width: 16 });
        let pair = self.module.ty(Type::Vector {
            component: f16,
            count: 2,
        });
        let accumulator_type = if single { self.f32 } else { f16 };
        let halves = self.matrix_scratch(f16);
        let wide = if single {
            self.matrix_scratch(self.f32)
        } else {
            halves
        };
        // This warp's region: by subgroup id (Vulkan does not promise that
        // invocation index / 32 names the subgroup).
        self.module.capability(capability::GROUP_NON_UNIFORM);
        let warp = self.scalar_builtin(BUILTIN_SUBGROUP_ID);
        // The previous multiply's lanes may still be reading D: wait
        // before writing the fragments.
        self.subgroup_barrier();
        let size = self.constant(MATRIX_ELEMENTS * 3);
        let base = self.value(op::I_MUL, self.u32, &[warp, size]);
        let lane = self.special_register(index, 0)?;
        let (one, two, three, four) = (
            self.constant(1),
            self.constant(2),
            self.constant(3),
            self.constant(4),
        );
        let group = self.value(op::SHIFT_RIGHT_LOGICAL, self.u32, &[lane, two]);
        let thread = self.value(op::BITWISE_AND, self.u32, &[lane, three]);
        let column_pair = self.value(op::I_MUL, self.u32, &[thread, two]);
        let next = |base: u8, n: u8| {
            if base == 255 {
                255
            } else {
                base.saturating_add(n)
            }
        };
        let half_of = |e: &mut Self, register: u8, h: u32| {
            let bits = e.register_or_zero(register);
            let both = e.value(op::BITCAST, pair, &[bits]);
            e.value(op::COMPOSITE_EXTRACT, f16, &[both, h])
        };
        let plus = |e: &mut Self, value: u32, n: u32| {
            let n = e.constant(n);
            e.value(op::I_ADD, e.u32, &[value, n])
        };
        // A: register j, half h at (g + 8 (j & 1), 2t + h + 8 (j >> 1)); with
        // K = 8 (HMMA.1688), columns 8-15 are zero, and so are B's rows 8-15:
        // the multiply is then exactly the 16x8x8 one.
        let zero_half = self.module.constant(f16, 0);
        for j in 0..4u32 {
            let row = plus(self, group, 8 * (j & 1));
            for h in 0..2 {
                let column = plus(self, column_pair, h + 8 * (j >> 1));
                let value = if k16 || j < 2 {
                    half_of(self, next(a, j as u8), h)
                } else {
                    zero_half
                };
                let pointer = self.scratch_element(halves, f16, base, 0, row, column);
                self.function.emit(op::STORE, &[pointer, value]);
            }
        }
        // B: register j, half h at (2t + h + 8j, g + 8 n) for HMMA n; with
        // one, columns 8-15 zero (lane l zeroes row l / 2, four columns from
        // 8 + 4 (l & 1)).
        for (n, &(_, b, _)) in halves_in.iter().enumerate() {
            let column = plus(self, group, 8 * n as u32);
            for j in 0..2u32 {
                for h in 0..2 {
                    let row = plus(self, column_pair, h + 8 * j);
                    let value = if k16 || j == 0 {
                        half_of(self, next(b, j as u8), h)
                    } else {
                        zero_half
                    };
                    let pointer = self.scratch_element(halves, f16, base, 1, row, column);
                    self.function.emit(op::STORE, &[pointer, value]);
                }
            }
        }
        if halves_in.len() == 1 {
            let zero = self.module.constant(f16, 0);
            let zero_row = self.value(op::SHIFT_RIGHT_LOGICAL, self.u32, &[lane, one]);
            let odd = self.value(op::BITWISE_AND, self.u32, &[lane, one]);
            let quarter = self.value(op::I_MUL, self.u32, &[odd, four]);
            for n in 0..4 {
                let column = plus(self, quarter, 8 + n);
                let pointer = self.scratch_element(halves, f16, base, 1, zero_row, column);
                self.function.emit(op::STORE, &[pointer, zero]);
            }
        }
        // C: element i at (g + 8 (i >> 1), 2t + (i & 1) + 8 n) for HMMA n.
        for (n, &(_, _, c)) in halves_in.iter().enumerate() {
            for i in 0..4u32 {
                let row = plus(self, group, 8 * (i >> 1));
                let column = plus(self, column_pair, (i & 1) + 8 * n as u32);
                let value = if single {
                    let bits = self.register_or_zero(next(c, i as u8));
                    self.value(op::BITCAST, self.f32, &[bits])
                } else {
                    half_of(self, next(c, (i / 2) as u8), i % 2)
                };
                let pointer = self.scratch_element(wide, accumulator_type, base, 2, row, column);
                self.function.emit(op::STORE, &[pointer, value]);
            }
        }
        self.subgroup_barrier();
        // The multiply: row-major, stride 16.
        let (scope, sixteen) = (self.constant(SCOPE_SUBGROUP), self.constant(16));
        let row_major = self.constant(0);
        let matrix = |e: &mut Self, component: u32, usage: u32| {
            let usage = e.constant(usage);
            e.module.ty(Type::CooperativeMatrix {
                component,
                scope,
                rows: sixteen,
                columns: sixteen,
                usage,
            })
        };
        let (a_type, b_type) = (matrix(self, f16, 0), matrix(self, f16, 1));
        let c_type = matrix(self, accumulator_type, 2);
        let zero_index = self.constant(0);
        let a_pointer = self.scratch_element(halves, f16, base, 0, zero_index, zero_index);
        let b_pointer = self.scratch_element(halves, f16, base, 1, zero_index, zero_index);
        let c_pointer =
            self.scratch_element(wide, accumulator_type, base, 2, zero_index, zero_index);
        let load = |e: &mut Self, ty: u32, pointer: u32| {
            e.value(
                op::COOPERATIVE_MATRIX_LOAD,
                ty,
                &[pointer, row_major, sixteen],
            )
        };
        let a_matrix = load(self, a_type, a_pointer);
        let b_matrix = load(self, b_type, b_pointer);
        let c_matrix = load(self, c_type, c_pointer);
        let product = self.value(
            op::COOPERATIVE_MATRIX_MUL_ADD,
            c_type,
            &[a_matrix, b_matrix, c_matrix],
        );
        self.function.emit(
            op::COOPERATIVE_MATRIX_STORE,
            &[c_pointer, product, row_major, sixteen],
        );
        self.subgroup_barrier();
        self.matrices_used = true;
        // D back, where C was.
        for (n, &(d, _, _)) in halves_in.iter().enumerate() {
            if d == 255 {
                continue;
            }
            let values: Vec<u32> = (0..4u32)
                .map(|i| {
                    let row = plus(self, group, 8 * (i >> 1));
                    let column = plus(self, column_pair, (i & 1) + 8 * n as u32);
                    let pointer =
                        self.scratch_element(wide, accumulator_type, base, 2, row, column);
                    self.value(op::LOAD, accumulator_type, &[pointer])
                })
                .collect();
            if single {
                for (m, value) in values.into_iter().enumerate() {
                    self.write_float(d.saturating_add(m as u8), value);
                }
            } else {
                for m in 0..2 {
                    let vector = self.value(
                        op::COMPOSITE_CONSTRUCT,
                        pair,
                        &[values[2 * m], values[2 * m + 1]],
                    );
                    let bits = self.value(op::BITCAST, self.u32, &[vector]);
                    self.write_register(d.saturating_add(m as u8), bits);
                }
            }
        }
        Ok(())
    }

    /// One tensor-core step (see the module notes): `accumulator` plus
    /// `products`, aligned to the largest exponent and truncated, summed
    /// exactly and normalised by truncation, or with `odd` by rounding to
    /// odd (a half rounded from it is then correctly rounded).
    fn block_sum(&mut self, accumulator: u32, products: &[u32], odd: bool) -> u32 {
        let u32_ = self.u32;
        let c = |e: &mut Self, value: u32| e.constant(value);
        let (zero, one, twenty_three, thirty_one, thirty_two) = (
            c(self, 0),
            c(self, 1),
            c(self, 23),
            c(self, 31),
            c(self, 32),
        );
        let (exponent_mask, fraction_mask, leading) =
            (c(self, 0xff), c(self, 0x7f_ffff), c(self, 0x80_0000));
        // A logical right shift by `shift`, 0 from 32 on.
        let shift_right = |e: &mut Self, value: u32, shift: u32| {
            let clamped = e.glsl(u32_, glsl::U_MIN, &[shift, thirty_one]);
            let shifted = e.value(op::SHIFT_RIGHT_LOGICAL, u32_, &[value, clamped]);
            let inside = e.value(op::U_LESS_THAN, e.bool, &[shift, thirty_two]);
            e.value(op::SELECT, u32_, &[inside, shifted, zero])
        };
        let addends: Vec<u32> = std::iter::once(accumulator)
            .chain(products.iter().copied())
            .collect();
        // Each addend's sign, biased exponent (subnormals and zero as 1)
        // and 24-bit significand.
        let mut parts = Vec::new();
        let mut special = None;
        let mut largest = one;
        for &addend in &addends {
            let bits = self.value(op::BITCAST, u32_, &[addend]);
            let sign = self.value(op::SHIFT_RIGHT_LOGICAL, u32_, &[bits, thirty_one]);
            let field = self.value(op::SHIFT_RIGHT_LOGICAL, u32_, &[bits, twenty_three]);
            let field = self.value(op::BITWISE_AND, u32_, &[field, exponent_mask]);
            let normal = self.value(op::I_NOT_EQUAL, self.bool, &[field, zero]);
            let exponent = self.glsl(u32_, glsl::U_MAX, &[field, one]);
            let fraction = self.value(op::BITWISE_AND, u32_, &[bits, fraction_mask]);
            let hidden = self.value(op::SELECT, u32_, &[normal, leading, zero]);
            let significand = self.value(op::BITWISE_OR, u32_, &[fraction, hidden]);
            let infinite = self.value(op::I_EQUAL, self.bool, &[field, exponent_mask]);
            special = Some(match special {
                None => infinite,
                Some(any) => self.value(op::LOGICAL_OR, self.bool, &[any, infinite]),
            });
            largest = self.glsl(u32_, glsl::U_MAX, &[largest, exponent]);
            parts.push((sign, exponent, significand));
        }
        // In units of 2^(largest - 151), 24 bits below the leading bit: an
        // addend at the largest exponent is its significand times 2, one
        // below it its significand shifted right (truncated) by the
        // difference less one.
        let mut sum = zero;
        for (sign, exponent, significand) in parts {
            let gap = self.value(op::I_SUB, u32_, &[largest, exponent]);
            let at_top = self.value(op::I_EQUAL, self.bool, &[gap, zero]);
            let doubled = self.value(op::SHIFT_LEFT_LOGICAL, u32_, &[significand, one]);
            // (At the top the shift wraps; that value is not selected.)
            let below = self.value(op::I_SUB, u32_, &[gap, one]);
            let shifted = shift_right(self, significand, below);
            let magnitude = self.value(op::SELECT, u32_, &[at_top, doubled, shifted]);
            let negative = self.value(op::I_SUB, u32_, &[zero, magnitude]);
            let is_negative = self.value(op::I_NOT_EQUAL, self.bool, &[sign, zero]);
            let term = self.value(op::SELECT, u32_, &[is_negative, negative, magnitude]);
            sum = self.value(op::I_ADD, u32_, &[sum, term]);
        }
        // Normalise: sign, magnitude M (under 2^29), its leading bit p.
        // The value is M * 2^(largest - 151): biased exponent
        // largest + p - 24, kept signed in 32 bits.
        let sign = self.value(op::SHIFT_RIGHT_LOGICAL, u32_, &[sum, thirty_one]);
        let magnitude = self.glsl(u32_, glsl::S_ABS, &[sum]);
        let p = self.glsl(u32_, glsl::FIND_U_MSB, &[magnitude]);
        let twenty_four = c(self, 24);
        let exponent = self.value(op::I_ADD, u32_, &[largest, p]);
        let exponent = self.value(op::I_SUB, u32_, &[exponent, twenty_four]);
        let subnormal = self.value(op::S_LESS_THAN, self.bool, &[exponent, one]);
        // The significand is M shifted right by p - 23 (a subnormal's,
        // M * 2^(largest - 2), by 2 - largest); a negative shift is left.
        let two = c(self, 2);
        let normal_shift = self.value(op::I_SUB, u32_, &[p, twenty_three]);
        let subnormal_shift = self.value(op::I_SUB, u32_, &[two, largest]);
        let shift = self.value(
            op::SELECT,
            u32_,
            &[subnormal, subnormal_shift, normal_shift],
        );
        let right = self.value(op::S_GREATER_THAN, self.bool, &[shift, zero]);
        let kept = shift_right(self, magnitude, shift);
        let left = self.value(op::I_SUB, u32_, &[zero, shift]);
        let left = self.glsl(u32_, glsl::U_MIN, &[left, thirty_one]);
        let raised = self.value(op::SHIFT_LEFT_LOGICAL, u32_, &[magnitude, left]);
        let mut significand = self.value(op::SELECT, u32_, &[right, kept, raised]);
        if odd {
            // Round to odd: any dropped bit sets the lowest kept one.
            let back = self.glsl(u32_, glsl::U_MIN, &[shift, thirty_one]);
            let restored = self.value(op::SHIFT_LEFT_LOGICAL, u32_, &[kept, back]);
            let inexact = self.value(op::I_NOT_EQUAL, self.bool, &[restored, magnitude]);
            let small = self.value(op::U_LESS_THAN, self.bool, &[shift, thirty_two]);
            let lost = self.value(op::LOGICAL_AND, self.bool, &[small, inexact]);
            // Shifted wholly out: inexact unless M is 0.
            let gone = self.value(op::LOGICAL_NOT, self.bool, &[small]);
            let nonzero = self.value(op::I_NOT_EQUAL, self.bool, &[magnitude, zero]);
            let gone = self.value(op::LOGICAL_AND, self.bool, &[gone, nonzero]);
            let inexact = self.value(op::LOGICAL_OR, self.bool, &[lost, gone]);
            let inexact = self.value(op::LOGICAL_AND, self.bool, &[right, inexact]);
            let sticky = self.value(op::SELECT, u32_, &[inexact, one, zero]);
            significand = self.value(op::BITWISE_OR, u32_, &[significand, sticky]);
        }
        // Assemble; a subnormal's field is 0 (its significand has no
        // leading bit), a zero sum is +0.
        let field = self.value(op::SELECT, u32_, &[subnormal, zero, exponent]);
        let fraction = self.value(op::BITWISE_AND, u32_, &[significand, fraction_mask]);
        let field_bits = self.value(op::SHIFT_LEFT_LOGICAL, u32_, &[field, twenty_three]);
        let sign_bit = self.value(op::SHIFT_LEFT_LOGICAL, u32_, &[sign, thirty_one]);
        let bits = self.value(op::BITWISE_OR, u32_, &[field_bits, fraction]);
        let bits = self.value(op::BITWISE_OR, u32_, &[bits, sign_bit]);
        // Past the range: the largest finite value (truncation).
        let maximum = c(self, 0x7f7f_ffff);
        let maximum = self.value(op::BITWISE_OR, u32_, &[maximum, sign_bit]);
        let limit = c(self, 255);
        let over = self.value(op::S_GREATER_THAN_EQUAL, self.bool, &[exponent, limit]);
        let bits = self.value(op::SELECT, u32_, &[over, maximum, bits]);
        let is_zero = self.value(op::I_EQUAL, self.bool, &[sum, zero]);
        let bits = self.value(op::SELECT, u32_, &[is_zero, zero, bits]);
        let result = self.value(op::BITCAST, self.f32, &[bits]);
        // Infinities and NaN: the plain sum.
        let mut plain = accumulator;
        for &product in products {
            plain = self.exact(op::F_ADD, &[plain, product]);
        }
        let special = special.expect("an accumulator");
        self.value(op::SELECT, self.f32, &[special, plain, result])
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn matrix_multiply(
        &mut self,
        index: usize,
        d: u8,
        a: u8,
        b: u8,
        c: u8,
        k16: bool,
        single: bool,
    ) -> Result<(), Unsupported> {
        if self.layout.matrices == super::MatrixPath::Cooperative {
            if self.hmma_pairs.seconds.contains(&index) {
                return Ok(());
            }
            let mut halves = vec![(d, b, c)];
            halves.extend(self.hmma_pairs.firsts.get(&index).copied());
            return self.cooperative_multiply(index, a, &halves, k16, single);
        }
        self.module
            .capability(capability::GROUP_NON_UNIFORM_SHUFFLE);
        let next = |base: u8, n: u8| {
            if base == 255 {
                255
            } else {
                base.saturating_add(n)
            }
        };
        let a_registers: Vec<u32> = (0..if k16 { 4 } else { 2 })
            .map(|n| self.register_or_zero(next(a, n)))
            .collect();
        let b_registers: Vec<u32> = (0..if k16 { 2 } else { 1 })
            .map(|n| self.register_or_zero(next(b, n)))
            .collect();
        // The accumulator: rows g, g + 8; columns 2t, 2t + 1.
        let mut accumulator: Vec<u32> = if single {
            (0..4)
                .map(|n| {
                    let bits = self.register_or_zero(next(c, n));
                    self.value(op::BITCAST, self.f32, &[bits])
                })
                .collect()
        } else {
            let low = self.register_or_zero(c);
            let high = self.register_or_zero(next(c, 1));
            let [c0, c1] = self.halves(low);
            let [c2, c3] = self.halves(high);
            vec![c0, c1, c2, c3]
        };
        let lane = self.special_register(index, 0)?;
        let (two, three) = (self.constant(2), self.constant(3));
        let group = self.value(op::SHIFT_RIGHT_LOGICAL, self.u32, &[lane, two]);
        let thread = self.value(op::BITWISE_AND, self.u32, &[lane, three]);
        let subgroup = self.constant(SCOPE_SUBGROUP);
        let four = self.constant(4);
        let quad = self.value(op::I_MUL, self.u32, &[group, four]);
        // Columns 2t and 2t + 1 live in the quads of lanes 8t and 8t + 4.
        let eight = self.constant(8);
        let column_quad = self.value(op::I_MUL, self.u32, &[thread, eight]);
        let shuffle = |e: &mut Self, value: u32, source: u32| {
            e.value(
                op::GROUP_NON_UNIFORM_SHUFFLE,
                e.u32,
                &[subgroup, value, source],
            )
        };
        // Each output's products, in k order (for the exact sums).
        let exact = self.layout.matrices == super::MatrixPath::Exact;
        let mut products: [Vec<u32>; 4] = Default::default();
        for pair in 0..if k16 { 8u32 } else { 4 } {
            // k = 8 * upper + 2 * source + h, h the half.
            let (source, upper) = (pair % 4, (pair / 4) as usize);
            let source = self.constant(source);
            let from_quad = self.value(op::I_ADD, self.u32, &[quad, source]);
            let row_low = shuffle(self, a_registers[2 * upper], from_quad);
            let row_high = shuffle(self, a_registers[2 * upper + 1], from_quad);
            let first = self.value(op::I_ADD, self.u32, &[column_quad, source]);
            let column_0 = shuffle(self, b_registers[upper], first);
            let second = self.value(op::I_ADD, self.u32, &[first, four]);
            let column_1 = shuffle(self, b_registers[upper], second);
            let [a_g0, a_g1] = self.halves(row_low);
            let [a_h0, a_h1] = self.halves(row_high);
            let [b00, b01] = self.halves(column_0);
            let [b10, b11] = self.halves(column_1);
            // Each output sums its two products of this pair, in k order.
            let terms = [
                (0, a_g0, b00, a_g1, b01),
                (1, a_g0, b10, a_g1, b11),
                (2, a_h0, b00, a_h1, b01),
                (3, a_h0, b10, a_h1, b11),
            ];
            for (output, x0, y0, x1, y1) in terms {
                for (x, y) in [(x0, y0), (x1, y1)] {
                    if exact {
                        let product = self.exact(op::F_MUL, &[x, y]);
                        products[output].push(product);
                    } else {
                        // A product of two halves is exact in 32 bits, so a
                        // fused multiply-add rounds as the add after it would
                        // (fused or not, the result is the same).
                        accumulator[output] =
                            self.glsl(self.f32, glsl::FMA, &[x, y, accumulator[output]]);
                    }
                }
            }
        }
        if exact {
            for (output, products) in products.iter().enumerate() {
                let blocks: Vec<&[u32]> = products.chunks(8).collect();
                for (n, block) in blocks.iter().enumerate() {
                    let odd = !single && n + 1 == blocks.len();
                    accumulator[output] = self.block_sum(accumulator[output], block, odd);
                }
            }
        }
        if d == 255 {
            return Ok(());
        }
        if single {
            for (n, value) in accumulator.into_iter().enumerate() {
                self.write_float(d.saturating_add(n as u8), value);
            }
        } else {
            let low = self.pack([accumulator[0], accumulator[1]]);
            let high = self.pack([accumulator[2], accumulator[3]]);
            self.write_register(d, low);
            self.write_register(d.saturating_add(1), high);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    /// `block_sum`'s arithmetic step by step on the CPU (the SPIR-V is
    /// validated, not run, by the tests).
    fn block_sum(accumulator: f32, products: &[f32], odd: bool) -> f32 {
        let addends: Vec<f32> = std::iter::once(accumulator)
            .chain(products.iter().copied())
            .collect();
        if addends.iter().any(|x| !x.is_finite()) {
            return addends.iter().sum();
        }
        let parts: Vec<(bool, u32, u32)> = addends
            .iter()
            .map(|x| {
                let bits = x.to_bits();
                let field = bits >> 23 & 0xff;
                let hidden = if field != 0 { 0x80_0000 } else { 0 };
                (bits >> 31 != 0, field.max(1), bits & 0x7f_ffff | hidden)
            })
            .collect();
        let largest = parts.iter().map(|part| part.1).max().unwrap();
        let sum: i32 = parts
            .iter()
            .map(|&(negative, exponent, significand)| {
                let gap = largest - exponent;
                let magnitude = if gap == 0 {
                    significand << 1
                } else {
                    significand.checked_shr(gap - 1).unwrap_or(0)
                } as i32;
                if negative { -magnitude } else { magnitude }
            })
            .sum();
        if sum == 0 {
            return 0.0;
        }
        let magnitude = sum.unsigned_abs();
        let p = 31 - magnitude.leading_zeros() as i32;
        let exponent = largest as i32 + p - 24;
        let shift = if exponent < 1 {
            2 - largest as i32
        } else {
            p - 23
        };
        let mut significand = if shift > 0 {
            magnitude >> shift
        } else {
            magnitude << -shift
        };
        if odd && shift > 0 && significand << shift != magnitude {
            significand |= 1;
        }
        let sign = (sum < 0) as u32;
        if exponent >= 255 {
            return f32::from_bits(0x7f7f_ffff | sign << 31);
        }
        let field = if exponent < 1 { 0 } else { exponent as u32 };
        f32::from_bits(sign << 31 | field << 23 | significand & 0x7f_ffff)
    }

    #[test]
    fn block_sums_align_truncate_and_normalise_once() {
        // Exact sums are exact.
        assert_eq!(block_sum(1.0, &[0.5; 8], false), 5.0);
        assert_eq!(block_sum(-3.0, &[0.25, -0.75], false), -3.5);
        // 1 + 1.5 * 2^-24: the addend truncates to 2^-24 at alignment and
        // the sum to 1 at normalisation; round-to-nearest gives 1 + 2^-23.
        let small = 1.5 * 2f32.powi(-24);
        assert_eq!(block_sum(1.0, &[small], false), 1.0);
        assert_eq!(1.0 + small, 1.0 + 2f32.powi(-23));
        // Toward zero for negative sums too.
        assert_eq!(block_sum(-1.0, &[-small], false), -1.0);
        // Alignment loses what lies under the largest addend, even when
        // the large addends cancel.
        assert_eq!(block_sum(1.0, &[-1.0, 2f32.powi(-30)], false), 0.0);
        // Round to odd keeps the lost bit for a later rounding to half.
        assert_eq!(block_sum(1.0, &[small], true), 1.0 + 2f32.powi(-23));
        // Subnormal results, and past the range.
        let tiny = f32::from_bits(3);
        assert_eq!(
            block_sum(tiny, &[f32::from_bits(2)], false),
            f32::from_bits(5)
        );
        assert_eq!(block_sum(f32::MAX, &[f32::MAX], false), f32::MAX);
        assert!(block_sum(1.0, &[f32::NAN], false).is_nan());
    }

    /// The PTX fragment positions, written from the layout's own
    /// definition (NAK's `compute_mat`): A element i of a lane, (row, k).
    fn a_position(lane: usize, i: usize) -> (usize, usize) {
        let (g, t) = (lane / 4, lane % 4);
        (g + 8 * (i >> 1 & 1), t * 2 + (i & 1) + 8 * (i >> 2))
    }

    /// B element i (k, column).
    fn b_position(lane: usize, i: usize) -> (usize, usize) {
        let (g, t) = (lane / 4, lane % 4);
        (t * 2 + (i & 1) + 8 * (i >> 1), g)
    }

    /// C/D element i (row, column).
    fn c_position(lane: usize, i: usize) -> (usize, usize) {
        let (g, t) = (lane / 4, lane % 4);
        (g + 8 * (i >> 1), t * 2 + (i & 1))
    }

    #[test]
    fn the_shuffle_schedule_computes_the_matrix_product() {
        // Small integers, so every sum is exact.
        let a: Vec<Vec<f32>> = (0..16)
            .map(|r| {
                (0..16)
                    .map(|k| ((r * 7 + k * 3) % 5) as f32 - 2.0)
                    .collect()
            })
            .collect();
        let b: Vec<Vec<f32>> = (0..16)
            .map(|k| (0..8).map(|n| ((k * 5 + n) % 7) as f32 - 3.0).collect())
            .collect();
        let c: Vec<Vec<f32>> = (0..16)
            .map(|r| (0..8).map(|n| (r + n) as f32).collect())
            .collect();
        // Each lane's registers, from the layout (halves as values).
        let a_regs: Vec<[[f32; 2]; 4]> = (0..32)
            .map(|lane| {
                let mut regs = [[0.0; 2]; 4];
                for i in 0..8 {
                    let (r, k) = a_position(lane, i);
                    regs[i / 2][i % 2] = a[r][k];
                }
                regs
            })
            .collect();
        let b_regs: Vec<[[f32; 2]; 2]> = (0..32)
            .map(|lane| {
                let mut regs = [[0.0; 2]; 2];
                for i in 0..4 {
                    let (k, n) = b_position(lane, i);
                    regs[i / 2][i % 2] = b[k][n];
                }
                regs
            })
            .collect();
        for lane in 0..32 {
            let (g, t) = (lane / 4, lane % 4);
            let mut acc: Vec<f32> = (0..4)
                .map(|i| {
                    let (r, n) = c_position(lane, i);
                    c[r][n]
                })
                .collect();
            // The schedule `matrix_multiply` emits.
            for pair in 0..8 {
                let (source, upper) = (pair % 4, pair / 4);
                let row_low = a_regs[4 * g + source][2 * upper];
                let row_high = a_regs[4 * g + source][2 * upper + 1];
                let column_0 = b_regs[8 * t + source][upper];
                let column_1 = b_regs[8 * t + 4 + source][upper];
                for h in 0..2 {
                    acc[0] += row_low[h] * column_0[h];
                    acc[1] += row_low[h] * column_1[h];
                    acc[2] += row_high[h] * column_0[h];
                    acc[3] += row_high[h] * column_1[h];
                }
            }
            for (i, &value) in acc.iter().enumerate() {
                let (r, n) = c_position(lane, i);
                let expected: f32 = c[r][n] + (0..16).map(|k| a[r][k] * b[k][n]).sum::<f32>();
                assert_eq!(value, expected, "lane {lane} element {i}");
            }
        }
    }
}
