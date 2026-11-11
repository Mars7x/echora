//! Half-precision pair operations (HADD2, HMUL2, HFMA2, HMNMX2) on SPIR-V's
//! 16-bit floats, so each lane rounds as a half does, and F2FP's packing.

use echora_shader_abi::Unsupported;

use super::{Emitter, unsupported};
use crate::decode::{Combine, FloatCompare, HalfOp, HalfOperand, Operand, Pred, Reg};
use crate::spirv::{Type, capability, glsl, op};

/// OpVectorShuffle.
const VECTOR_SHUFFLE: u16 = 79;
/// OpFConvert.
const F_CONVERT: u16 = 115;

impl Emitter {
    fn f16(&mut self) -> u32 {
        self.module.capability(capability::FLOAT16);
        self.module.ty(Type::Float { width: 16 })
    }

    fn f16_pair(&mut self) -> u32 {
        let f16 = self.f16();
        self.module.ty(Type::Vector {
            component: f16,
            count: 2,
        })
    }

    /// A half-pair source, swizzled and with its modifiers.
    fn half_operand(&mut self, index: usize, operand: HalfOperand) -> Result<u32, Unsupported> {
        let pair = self.f16_pair();
        let f16 = self.f16();
        let raw = self.source(index, operand.src)?;
        let mut value = match operand.swizzle {
            // A 32-bit float, converted, in both halves.
            1 => {
                let float = self.value(op::BITCAST, self.f32, &[raw]);
                let half = self.value(F_CONVERT, f16, &[float]);
                self.value(op::COMPOSITE_CONSTRUCT, pair, &[half, half])
            }
            swizzle => {
                let both = self.value(op::BITCAST, pair, &[raw]);
                match swizzle {
                    2 => self.value(VECTOR_SHUFFLE, pair, &[both, both, 0, 0]),
                    3 => self.value(VECTOR_SHUFFLE, pair, &[both, both, 1, 1]),
                    _ => both,
                }
            }
        };
        if operand.abs {
            value = self.glsl(pair, glsl::F_ABS, &[value]);
        }
        if operand.neg {
            value = self.value(op::F_NEGATE, pair, &[value]);
        }
        Ok(value)
    }

    /// A half constant pair.
    fn half_constant(&mut self, bits: u32) -> u32 {
        let f16 = self.f16();
        let pair = self.f16_pair();
        let half = self.module.constant(f16, bits);
        self.module.composite_constant(pair, &[half, half])
    }

    /// Each lane of `value`, or of `instead` where a or b is ±0 (the
    /// inputs already flushed when `.FTZ`): `.DNZ`'s zero factors.
    fn zero_factors(&mut self, a: u32, b: u32, instead: u32, value: u32) -> u32 {
        let pair = self.f16_pair();
        let bool2 = self.module.ty(crate::spirv::Type::Vector {
            component: self.bool,
            count: 2,
        });
        let zero = self.half_constant(0);
        let a_zero = self.value(op::F_ORD_EQUAL, bool2, &[a, zero]);
        let b_zero = self.value(op::F_ORD_EQUAL, bool2, &[b, zero]);
        let either = self.value(op::LOGICAL_OR, bool2, &[a_zero, b_zero]);
        self.value(op::SELECT, pair, &[either, instead, value])
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn half(
        &mut self,
        index: usize,
        operation: HalfOp,
        d: Reg,
        a: HalfOperand,
        b: HalfOperand,
        c: Option<HalfOperand>,
        saturate: bool,
        relu: Option<Pred>,
        flush: bool,
        dnz: bool,
    ) -> Result<(), Unsupported> {
        let pair = self.f16_pair();
        let mut a = self.half_operand(index, a)?;
        let mut b = self.half_operand(index, b)?;
        if flush {
            a = self.flush_halves(a);
            b = self.flush_halves(b);
        }
        let exact = |e: &mut Self, opcode: u16, operands: &[u32]| {
            let value = e.value(opcode, pair, operands);
            e.module
                .decorate(value, crate::spirv::decoration::NO_CONTRACTION, &[]);
            value
        };
        let mut value = match operation {
            HalfOp::Add => exact(self, op::F_ADD, &[a, b]),
            HalfOp::Mul => exact(self, op::F_MUL, &[a, b]),
            HalfOp::Fma => {
                let mut c = match c {
                    Some(c) => self.half_operand(index, c)?,
                    None => self.half_constant(0),
                };
                if flush {
                    c = self.flush_halves(c);
                }
                let product = self.glsl(pair, glsl::FMA, &[a, b, c]);
                if dnz {
                    // A zero factor: +0 + c, exactly.
                    let zero = self.half_constant(0);
                    let sum = exact(self, op::F_ADD, &[zero, c]);
                    self.zero_factors(a, b, sum, product)
                } else {
                    product
                }
            }
            HalfOp::MinMax { min } => self.half_min_max(min, a, b),
        };
        // HMUL2.DNZ: a zero factor's product is +0.
        if dnz && operation == HalfOp::Mul {
            let zero = self.half_constant(0);
            value = self.zero_factors(a, b, zero, value);
        }
        if flush {
            value = self.flush_halves(value);
        }
        if saturate {
            let (zero, one) = (self.half_constant(0), self.half_constant(0x3c00));
            let bool2 = self.module.ty(crate::spirv::Type::Vector {
                component: self.bool,
                count: 2,
            });
            value = self.saturate(pair, bool2, value, zero, one);
        }
        let mut bits = self.value(op::BITCAST, self.u32, &[value]);
        if let Some(condition) = relu {
            let clamped = self.relu_pair(bits);
            bits = if condition.is_true() {
                clamped
            } else {
                let holds = self.read_predicate(condition);
                self.value(op::SELECT, self.u32, &[holds, clamped, bits])
            };
        }
        self.write_register(d.0, bits);
        Ok(())
    }

    /// A pair of halves with each subnormal lane made a zero of its sign
    /// (`.FTZ`).
    fn flush_halves(&mut self, value: u32) -> u32 {
        let pair = self.f16_pair();
        let bits = self.value(op::BITCAST, self.u32, &[value]);
        let mut result = self.constant(0);
        for lane in 0..2u32 {
            let (shift, sixteen) = (self.constant(lane * 16), self.constant(16));
            let half = self.value(op::BIT_FIELD_U_EXTRACT, self.u32, &[bits, shift, sixteen]);
            let exponent_mask = self.constant(0x7c00);
            let exponent = self.value(op::BITWISE_AND, self.u32, &[half, exponent_mask]);
            let zero = self.constant(0);
            let subnormal = self.value(op::I_EQUAL, self.bool, &[exponent, zero]);
            let sign_mask = self.constant(0x8000);
            let sign = self.value(op::BITWISE_AND, self.u32, &[half, sign_mask]);
            let kept = self.value(op::SELECT, self.u32, &[subnormal, sign, half]);
            let placed = self.value(op::SHIFT_LEFT_LOGICAL, self.u32, &[kept, shift]);
            result = self.value(op::BITWISE_OR, self.u32, &[result, placed]);
        }
        self.value(op::BITCAST, pair, &[result])
    }

    /// The half in lane `lane` (0 low, 1 high) of `bits`, as a 32-bit float
    /// (exactly).
    pub(super) fn half_to_float(&mut self, bits: u32, lane: u8) -> u32 {
        let pair = self.f16_pair();
        let f16 = self.f16();
        let halves = self.value(op::BITCAST, pair, &[bits]);
        let half = self.value(op::COMPOSITE_EXTRACT, f16, &[halves, u32::from(lane)]);
        self.value(F_CONVERT, self.f32, &[half])
    }

    /// `float` (32-bit) as a half's bits (in the low 16), rounded by
    /// `rounding` (0 to nearest even, 1 down, 2 up, 3 toward zero): the
    /// nearest half, moved one step where it went past `float` the wrong
    /// way (so too large a value gives the largest finite half where the
    /// rounding goes toward zero).
    pub(super) fn round_to_half(&mut self, float: u32, rounding: u8) -> u32 {
        let f16 = self.f16();
        let pair = self.f16_pair();
        let zero_half = self.module.constant(f16, 0);
        let nearest = self.value(F_CONVERT, f16, &[float]);
        let both = self.value(op::COMPOSITE_CONSTRUCT, pair, &[nearest, zero_half]);
        let bits = self.value(op::BITCAST, self.u32, &[both]);
        if rounding == 0 {
            return bits;
        }
        let back = self.value(F_CONVERT, self.f32, &[nearest]);
        let sign_mask = self.constant(0x8000);
        let sign = self.value(op::BITWISE_AND, self.u32, &[bits, sign_mask]);
        let zero = self.constant(0);
        let negative = self.value(op::I_NOT_EQUAL, self.bool, &[sign, zero]);
        // Whether the nearest half is past `float` in the rounding's
        // direction, and so must step back by one in magnitude (or, for a
        // zero that should have moved away from it, forward).
        let (above, below) = (
            self.value(op::F_ORD_GREATER_THAN, self.bool, &[back, float]),
            self.value(op::F_ORD_LESS_THAN, self.bool, &[back, float]),
        );
        let one = self.constant(1);
        let down = self.value(op::I_SUB, self.u32, &[bits, one]);
        let up = self.value(op::I_ADD, self.u32, &[bits, one]);
        // Moving toward minus infinity: a positive half shrinks, a negative
        // one grows in magnitude; toward plus infinity the reverse.
        let toward_minus = self.value(op::SELECT, self.u32, &[negative, up, down]);
        let toward_plus = self.value(op::SELECT, self.u32, &[negative, down, up]);
        let adjusted = match rounding {
            1 => self.value(op::SELECT, self.u32, &[above, toward_minus, bits]),
            2 => self.value(op::SELECT, self.u32, &[below, toward_plus, bits]),
            _ => {
                let magnitude_above = self.value(op::SELECT, self.bool, &[negative, below, above]);
                self.value(op::SELECT, self.u32, &[magnitude_above, down, bits])
            }
        };
        // A negative zero from rounding up to -0 and similar stay as they
        // are: only the 16 bits matter.
        let mask = self.constant(0xffff);
        self.value(op::BITWISE_AND, self.u32, &[adjusted, mask])
    }

    /// `.RELU` on a register of halves, lane by lane on the bits: a NaN
    /// becomes the canonical 0x7fff, a negative value (either zero, or
    /// minus infinity) +0, anything else stays.
    fn relu_pair(&mut self, bits: u32) -> u32 {
        let mut result = self.constant(0);
        for lane in 0..2u32 {
            let (shift, sixteen) = (self.constant(lane * 16), self.constant(16));
            let half = self.value(op::BIT_FIELD_U_EXTRACT, self.u32, &[bits, shift, sixteen]);
            let magnitude_mask = self.constant(0x7fff);
            let magnitude = self.value(op::BITWISE_AND, self.u32, &[half, magnitude_mask]);
            let infinity = self.constant(0x7c00);
            let nan = self.value(op::U_GREATER_THAN, self.bool, &[magnitude, infinity]);
            let sign_bit = self.constant(0x8000);
            let sign = self.value(op::BITWISE_AND, self.u32, &[half, sign_bit]);
            let zero = self.constant(0);
            let negative = self.value(op::I_NOT_EQUAL, self.bool, &[sign, zero]);
            let kept = self.value(op::SELECT, self.u32, &[negative, zero, half]);
            let canonical = self.constant(0x7fff);
            let value = self.value(op::SELECT, self.u32, &[nan, canonical, kept]);
            let placed = self.value(op::SHIFT_LEFT_LOGICAL, self.u32, &[value, shift]);
            result = self.value(op::BITWISE_OR, self.u32, &[result, placed]);
        }
        result
    }

    /// HADD2.F32: the operands' lane-0 halves added as 32-bit floats, one
    /// float result.
    pub(super) fn half_single(
        &mut self,
        index: usize,
        operation: HalfOp,
        d: Reg,
        a: HalfOperand,
        b: HalfOperand,
        saturate: bool,
    ) -> Result<(), Unsupported> {
        if operation != HalfOp::Add {
            return Err(unsupported(index, "half operation with a 32-bit result"));
        }
        let f16 = self.f16();
        let a = self.half_operand(index, a)?;
        let b = self.half_operand(index, b)?;
        let lane = |e: &mut Self, pair: u32| {
            let half = e.value(op::COMPOSITE_EXTRACT, f16, &[pair, 0]);
            e.value(F_CONVERT, e.f32, &[half])
        };
        let (a, b) = (lane(self, a), lane(self, b));
        let mut value = self.exact(op::F_ADD, &[a, b]);
        if saturate {
            let (zero, one) = (self.float_constant(0.0), self.float_constant(1.0));
            value = self.saturate(self.f32, self.bool, value, zero, one);
        }
        self.write_float(d.0, value);
        Ok(())
    }

    fn half_min_max(&mut self, min: Pred, a: u32, b: u32) -> u32 {
        let pair = self.f16_pair();
        if min.index == 7 {
            let f = if min.negated {
                glsl::N_MAX
            } else {
                glsl::N_MIN
            };
            return self.glsl(pair, f, &[a, b]);
        }
        let condition = self.read_predicate(min);
        let low = self.glsl(pair, glsl::N_MIN, &[a, b]);
        let high = self.glsl(pair, glsl::N_MAX, &[a, b]);
        let bool_pair = self.module.ty(Type::Vector {
            component: self.bool,
            count: 2,
        });
        let both = self.value(op::COMPOSITE_CONSTRUCT, bool_pair, &[condition, condition]);
        self.value(op::SELECT, pair, &[both, low, high])
    }

    /// HSETP2: p from lane 0's comparison, q from lane 1's.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn half_set_predicates(
        &mut self,
        index: usize,
        p: u8,
        q: u8,
        a: HalfOperand,
        b: HalfOperand,
        compare: FloatCompare,
        combine: Combine,
        accumulate: Pred,
    ) -> Result<(), Unsupported> {
        let f16 = self.f16();
        let a = self.half_operand(index, a)?;
        let b = self.half_operand(index, b)?;
        let mut results = [0; 2];
        for (lane, result) in results.iter_mut().enumerate() {
            let x = self.value(op::COMPOSITE_EXTRACT, f16, &[a, lane as u32]);
            let y = self.value(op::COMPOSITE_EXTRACT, f16, &[b, lane as u32]);
            let compared = self.float_compare(compare, x, y);
            *result = self.combined(compared, combine, accumulate);
        }
        self.write_predicate(p, results[0]);
        self.write_predicate(q, results[1]);
        Ok(())
    }

    /// F2FP.PACK_AB: b and a (or c's low half) as halves, b low.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn pack_halves(
        &mut self,
        index: usize,
        d: Reg,
        a: Reg,
        b: Operand,
        c: Reg,
        merge: bool,
        rounding: u8,
    ) -> Result<(), Unsupported> {
        // A float's half in the low 16 bits.
        let half_bits = |e: &mut Self, word: u32| {
            let float = e.value(op::BITCAST, e.f32, &[word]);
            e.round_to_half(float, rounding)
        };
        let b = self.integer(index, b)?;
        let low = half_bits(self, b);
        let high = if merge {
            let c = self.read_register(c.0);
            let mask = self.constant(0xffff);
            self.value(op::BITWISE_AND, self.u32, &[c, mask])
        } else {
            let a = self.read_register(a.0);
            half_bits(self, a)
        };
        let sixteen = self.constant(16);
        let high = self.value(op::SHIFT_LEFT_LOGICAL, self.u32, &[high, sixteen]);
        let value = self.value(op::BITWISE_OR, self.u32, &[high, low]);
        self.write_register(d.0, value);
        Ok(())
    }
}
