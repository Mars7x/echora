//! Denormals kept: non-`.FTZ` FADD, FMUL and FFMA computed exactly in f64
//! (`Layout::float64`), so that a host which flushes 32-bit denormals (RADV
//! and NVK do by default) still gives what the hardware gives without
//! `.FTZ`: IEEE results with denormal operands and results.
//!
//! A 32-bit denormal operand becomes its exact f64 value from its integer
//! bits (mantissa times 2^-149), never through a 32-bit float operation.
//! The f64 work is exact or rounds once where that is harmless:
//!
//! - FMUL: the product of two 32-bit floats is exact in f64 (48 bits).
//! - FADD: one f64 rounding, then one to 32 bits; for addition a format of
//!   at least 2p + 2 bits (53 >= 50) makes the double rounding innocuous.
//! - FFMA: the exact product plus c, rounded to odd in f64 (the two-sum's
//!   exact error decides the odd neighbour), then to 32 bits: rounding to
//!   odd in p + 2 bits or more and then to nearest is the correct rounding
//!   (Boldo and Melquiond).
//!
//! The result below the smallest normal is rounded to the 2^-149 grid with
//! integer operations, so no 32-bit float operation sees a denormal; every
//! f64 value involved is at least 2^-298, far above f64's denormals. `.SAT`
//! clamps the result's bits. Only round-to-nearest forms without `.DNZ` take
//! this path, and FMUL without a scale (whether the hardware scales before or
//! after rounding is not known); the others, and every `.FTZ` form, are
//! translated as before.

use echora_shader_abi::Unsupported;

use super::Emitter;
use crate::decode::{FloatModes, Operand};
use crate::spirv::{Type, capability, decoration, glsl, op};

const F_CONVERT: u16 = 115;
const ROUND_EVEN: u32 = 2;

impl Emitter {
    /// Whether a 32-bit float operation with `modes` keeps denormals here.
    pub(super) fn keeps_denormals(&self, modes: FloatModes) -> bool {
        self.layout.float64 && !modes.flush_to_zero && modes.rounding == 0
    }

    fn f64(&mut self) -> u32 {
        self.module.capability(capability::FLOAT64);
        self.module.ty(Type::Float { width: 64 })
    }

    fn double_constant(&mut self, value: f64) -> u32 {
        let bits = self.constant64(value.to_bits());
        let f64 = self.f64();
        self.value(op::BITCAST, f64, &[bits])
    }

    /// An f64 operation the driver must not fuse.
    fn exact64(&mut self, opcode: u16, operands: &[u32]) -> u32 {
        let f64 = self.f64();
        let value = self.value(opcode, f64, operands);
        self.module.decorate(value, decoration::NO_CONTRACTION, &[]);
        value
    }

    /// The f64 value of 32-bit float `bits` exactly, with `operand`'s
    /// modifiers.
    fn wide(&mut self, bits: u32, operand: Operand) -> u32 {
        let f64 = self.f64();
        let exponent_mask = self.constant(0x7f80_0000);
        let exponent = self.value(op::BITWISE_AND, self.u32, &[bits, exponent_mask]);
        let zero = self.constant(0);
        let small = self.value(op::I_EQUAL, self.bool, &[exponent, zero]);
        // Zeros and denormals: the mantissa times 2^-149, signed.
        let mantissa_mask = self.constant(0x007f_ffff);
        let mantissa = self.value(op::BITWISE_AND, self.u32, &[bits, mantissa_mask]);
        let count = self.value(op::CONVERT_U_TO_F, f64, &[mantissa]);
        let step = self.double_constant(2f64.powi(-149));
        let magnitude = self.exact64(op::F_MUL, &[count, step]);
        let negative_magnitude = self.value(op::F_NEGATE, f64, &[magnitude]);
        let sign_bit = self.constant(0x8000_0000);
        let sign = self.value(op::BITWISE_AND, self.u32, &[bits, sign_bit]);
        let negative = self.value(op::I_NOT_EQUAL, self.bool, &[sign, zero]);
        let tiny = self.value(op::SELECT, f64, &[negative, negative_magnitude, magnitude]);
        // Others (normal, Inf, NaN): converted.
        let float = self.value(op::BITCAST, self.f32, &[bits]);
        let wide = self.value(F_CONVERT, f64, &[float]);
        let mut value = self.value(op::SELECT, f64, &[small, tiny, wide]);
        if operand.abs {
            value = self.glsl(f64, glsl::F_ABS, &[value]);
        }
        if operand.neg {
            value = self.value(op::F_NEGATE, f64, &[value]);
        }
        value
    }

    /// Whether 32-bit float `bits` is a denormal.
    fn is_denormal(&mut self, bits: u32) -> u32 {
        let exponent_mask = self.constant(0x7f80_0000);
        let exponent = self.value(op::BITWISE_AND, self.u32, &[bits, exponent_mask]);
        let mantissa_mask = self.constant(0x007f_ffff);
        let mantissa = self.value(op::BITWISE_AND, self.u32, &[bits, mantissa_mask]);
        let zero = self.constant(0);
        let no_exponent = self.value(op::I_EQUAL, self.bool, &[exponent, zero]);
        let some_mantissa = self.value(op::I_NOT_EQUAL, self.bool, &[mantissa, zero]);
        self.value(op::LOGICAL_AND, self.bool, &[no_exponent, some_mantissa])
    }

    /// The bits of the 32-bit float nearest `x` (an f64), denormals kept,
    /// then `.SAT` on the bits.
    fn narrow(&mut self, x: u32, saturate: bool) -> u32 {
        let f64 = self.f64();
        let u64 = self.u64();
        let magnitude = self.glsl(f64, glsl::F_ABS, &[x]);
        let smallest = self.double_constant(f64::from(f32::MIN_POSITIVE));
        let tiny = self.value(op::F_ORD_LESS_THAN, self.bool, &[magnitude, smallest]);
        // Below the smallest normal: steps of 2^-149, rounded to nearest
        // even (2^23 steps is the smallest normal itself).
        // (Clamped first, so that the conversion is in range wherever the
        // result is not used.)
        let held = self.glsl(f64, glsl::N_MIN, &[magnitude, smallest]);
        let scale = self.double_constant(2f64.powi(149));
        let steps = self.exact64(op::F_MUL, &[held, scale]);
        let steps = self.glsl(f64, ROUND_EVEN, &[steps]);
        let steps = self.value(op::CONVERT_F_TO_U, self.u32, &[steps]);
        let wide_bits = self.value(op::BITCAST, u64, &[x]);
        let thirty_two = self.constant(32);
        let high = self.value(op::SHIFT_RIGHT_LOGICAL, u64, &[wide_bits, thirty_two]);
        let high = self.value(op::U_CONVERT, self.u32, &[high]);
        let sign_bit = self.constant(0x8000_0000);
        let sign = self.value(op::BITWISE_AND, self.u32, &[high, sign_bit]);
        let tiny_bits = self.value(op::BITWISE_OR, self.u32, &[sign, steps]);
        // Otherwise (normal, Inf, NaN) the conversion's result is no
        // denormal.
        let narrow = self.value(F_CONVERT, self.f32, &[x]);
        let normal_bits = self.value(op::BITCAST, self.u32, &[narrow]);
        let mut bits = self.value(op::SELECT, self.u32, &[tiny, tiny_bits, normal_bits]);
        if saturate {
            // Negative (or -0) and NaN give +0, above 1.0 gives 1.0: on the
            // bits, positive floats order as integers.
            let infinity = self.constant(0x7f80_0000);
            let one = self.constant(1.0f32.to_bits());
            let zero = self.constant(0);
            let low = self.value(op::U_GREATER_THAN, self.bool, &[bits, infinity]);
            let capped = self.glsl(self.u32, glsl::U_MIN, &[bits, one]);
            bits = self.value(op::SELECT, self.u32, &[low, zero, capped]);
        }
        bits
    }

    /// FADD, FMUL (no scale) or FFMA without `.FTZ` (`operation` FADD,
    /// FMUL or FFMA's operands, 2 or 3): computed in 32 bits, and again
    /// exactly through f64 where an operand or the result is a denormal
    /// (a result below the smallest normal, or a zero a host flush may have
    /// made): elsewhere no denormal is involved and both agree.
    pub(super) fn keeping_denormals(
        &mut self,
        index: usize,
        d: u8,
        operation: Fused,
        operands: &[Operand],
        modes: FloatModes,
    ) -> Result<(), Unsupported> {
        // Every source read before d is written.
        let bits: Vec<u32> = operands
            .iter()
            .map(|operand| self.source(index, operand.src))
            .collect::<Result<_, _>>()?;
        let floats: Vec<u32> = bits
            .iter()
            .zip(operands)
            .map(|(&bits, operand)| {
                let mut value = self.value(op::BITCAST, self.f32, &[bits]);
                if operand.abs {
                    value = self.glsl(self.f32, glsl::F_ABS, &[value]);
                }
                if operand.neg {
                    value = self.value(op::F_NEGATE, self.f32, &[value]);
                }
                value
            })
            .collect();
        let value = match operation {
            Fused::Add => self.exact(op::F_ADD, &[floats[0], floats[1]]),
            Fused::Mul => self.exact(op::F_MUL, &[floats[0], floats[1]]),
            Fused::Fma => self.glsl(self.f32, glsl::FMA, &[floats[0], floats[1], floats[2]]),
        };
        let mut fast = value;
        if modes.saturate {
            let (zero, one) = (self.float_constant(0.0), self.float_constant(1.0));
            fast = self.saturate(self.f32, self.bool, fast, zero, one);
        }
        let fast = self.value(op::BITCAST, self.u32, &[fast]);
        self.write_register(d, fast);
        // Where denormals may be involved: exactly.
        let magnitude = self.glsl(self.f32, glsl::F_ABS, &[value]);
        let smallest = self.float_constant(f32::MIN_POSITIVE);
        let mut involved = self.value(op::F_ORD_LESS_THAN, self.bool, &[magnitude, smallest]);
        for &bits in &bits {
            let denormal = self.is_denormal(bits);
            involved = self.value(op::LOGICAL_OR, self.bool, &[involved, denormal]);
        }
        self.when(involved, |e| {
            let wide: Vec<u32> = bits
                .iter()
                .zip(operands)
                .map(|(&bits, &operand)| e.wide(bits, operand))
                .collect();
            let exact = match operation {
                Fused::Add => e.exact64(op::F_ADD, &[wide[0], wide[1]]),
                Fused::Mul => e.exact64(op::F_MUL, &[wide[0], wide[1]]),
                Fused::Fma => e.fused64(wide[0], wide[1], wide[2]),
            };
            let result = e.narrow(exact, modes.saturate);
            e.write_register(d, result);
            Ok(())
        })
    }

    /// a * b + c (f64 values of 32-bit floats), rounded to odd in f64.
    fn fused64(&mut self, a: u32, b: u32, c: u32) -> u32 {
        let f64 = self.f64();
        let u64 = self.u64();
        // The product exactly, the sum rounded and its exact error.
        let product = self.exact64(op::F_MUL, &[a, b]);
        let sum = self.exact64(op::F_ADD, &[product, c]);
        let c_part = self.exact64(op::F_SUB, &[sum, product]);
        let product_part = self.exact64(op::F_SUB, &[sum, c_part]);
        let product_error = self.exact64(op::F_SUB, &[product, product_part]);
        let c_error = self.exact64(op::F_SUB, &[c, c_part]);
        let error = self.exact64(op::F_ADD, &[product_error, c_error]);
        // Rounded to odd: an inexact sum with an even last bit moves to its
        // neighbour on the error's side (a NaN error, from Inf or NaN,
        // leaves it).
        let zero = self.double_constant(0.0);
        let inexact = self.value(op::F_ORD_NOT_EQUAL, self.bool, &[error, zero]);
        let bits = self.value(op::BITCAST, u64, &[sum]);
        let one = self.constant64(1);
        let last = self.value(op::BITWISE_AND, u64, &[bits, one]);
        let zero64 = self.constant64(0);
        let even = self.value(op::I_EQUAL, self.bool, &[last, zero64]);
        let adjust = self.value(op::LOGICAL_AND, self.bool, &[inexact, even]);
        let error_positive = self.value(op::F_ORD_GREATER_THAN, self.bool, &[error, zero]);
        let sum_positive = self.value(op::F_ORD_GREATER_THAN, self.bool, &[sum, zero]);
        let away = self.value(
            op::LOGICAL_EQUAL,
            self.bool,
            &[error_positive, sum_positive],
        );
        let up = self.value(op::I_ADD, u64, &[bits, one]);
        let down = self.value(op::I_SUB, u64, &[bits, one]);
        let neighbour = self.value(op::SELECT, u64, &[away, up, down]);
        let odd = self.value(op::SELECT, u64, &[adjust, neighbour, bits]);
        self.value(op::BITCAST, f64, &[odd])
    }
}

/// The operations `keeping_denormals` computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Fused {
    Add,
    Mul,
    Fma,
}
