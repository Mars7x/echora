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

use echora_shader_abi::Unsupported;

use super::{Emitter, SCOPE_SUBGROUP};
use crate::spirv::{Type, capability, op, storage};

/// Elements of a 16x16 matrix.
const MATRIX_ELEMENTS: u32 = 256;

/// SPIR-V's SubgroupId built-in.
const BUILTIN_SUBGROUP_ID: u32 = 40;

/// OpFConvert.
const F_CONVERT: u16 = 115;

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

    /// HMMA.16816 on a 16x16x16 cooperative matrix multiply: the lanes
    /// write their fragments (PTX layout) to the warp's scratch, B's upper
    /// eight columns zero, the subgroup multiplies, and each lane reads its
    /// part of D back.
    #[allow(clippy::too_many_arguments)]
    fn cooperative_multiply(
        &mut self,
        index: usize,
        d: u8,
        a: u8,
        b: u8,
        c: u8,
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
        // A: register j, half h at (g + 8 (j & 1), 2t + h + 8 (j >> 1)).
        for j in 0..4u32 {
            let row = plus(self, group, 8 * (j & 1));
            for h in 0..2 {
                let column = plus(self, column_pair, h + 8 * (j >> 1));
                let value = half_of(self, next(a, j as u8), h);
                let pointer = self.scratch_element(halves, f16, base, 0, row, column);
                self.function.emit(op::STORE, &[pointer, value]);
            }
        }
        // B: register j, half h at (2t + h + 8j, g); columns 8-15 zero
        // (lane l zeroes row l / 2, four columns from 8 + 4 (l & 1)).
        for j in 0..2u32 {
            for h in 0..2 {
                let row = plus(self, column_pair, h + 8 * j);
                let value = half_of(self, next(b, j as u8), h);
                let pointer = self.scratch_element(halves, f16, base, 1, row, group);
                self.function.emit(op::STORE, &[pointer, value]);
            }
        }
        let zero = self.module.constant(f16, 0);
        let zero_row = self.value(op::SHIFT_RIGHT_LOGICAL, self.u32, &[lane, one]);
        let odd = self.value(op::BITWISE_AND, self.u32, &[lane, one]);
        let quarter = self.value(op::I_MUL, self.u32, &[odd, four]);
        for n in 0..4 {
            let column = plus(self, quarter, 8 + n);
            let pointer = self.scratch_element(halves, f16, base, 1, zero_row, column);
            self.function.emit(op::STORE, &[pointer, zero]);
        }
        // C: element i at (g + 8 (i >> 1), 2t + (i & 1)).
        for i in 0..4u32 {
            let row = plus(self, group, 8 * (i >> 1));
            let column = plus(self, column_pair, i & 1);
            let value = if single {
                let bits = self.register_or_zero(next(c, i as u8));
                self.value(op::BITCAST, self.f32, &[bits])
            } else {
                half_of(self, next(c, (i / 2) as u8), i % 2)
            };
            let pointer = self.scratch_element(wide, accumulator_type, base, 2, row, column);
            self.function.emit(op::STORE, &[pointer, value]);
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
        if d == 255 {
            return Ok(());
        }
        let values: Vec<u32> = (0..4u32)
            .map(|i| {
                let row = plus(self, group, 8 * (i >> 1));
                let column = plus(self, column_pair, i & 1);
                let pointer = self.scratch_element(wide, accumulator_type, base, 2, row, column);
                self.value(op::LOAD, accumulator_type, &[pointer])
            })
            .collect();
        if single {
            for (n, value) in values.into_iter().enumerate() {
                self.write_float(d.saturating_add(n as u8), value);
            }
        } else {
            for n in 0..2 {
                let vector = self.value(
                    op::COMPOSITE_CONSTRUCT,
                    pair,
                    &[values[2 * n], values[2 * n + 1]],
                );
                let bits = self.value(op::BITCAST, self.u32, &[vector]);
                self.write_register(d.saturating_add(n as u8), bits);
            }
        }
        Ok(())
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
        if k16 && self.layout.matrices == super::MatrixPath::Cooperative {
            return self.cooperative_multiply(index, d, a, b, c, single);
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
                    let product = self.exact(op::F_MUL, &[x, y]);
                    accumulator[output] = self.exact(op::F_ADD, &[accumulator[output], product]);
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
