//! Semantic differential test for integer SIMD and data movement: lifted
//! instructions against a reference interpreter written from the Arm
//! pseudocode.
//!
//! `build.rs` picks random words in the SIMD encoding space that decode to
//! the forms covered here (vector and scalar integer arithmetic, compares,
//! shifts, saturating and halving forms, pairwise and across-lanes
//! reductions, widening and narrowing, dot products, polynomial multiply,
//! permutes, EXT, TBL/TBX, DUP, INS, UMOV/SMOV, MOVI and the logical
//! forms), each followed by SVC #7, and lifts them through split emission.
//! Each runs natively from random vector, general-register and flag states;
//! the vector registers, general registers and NZCV afterwards must match
//! the interpreter. FPSR.QC is not compared.
#![cfg(any(target_os = "linux", windows))]
#![allow(clippy::unwrap_used)]

use echora_aot_tests::{Machine, Trap, simd_semantic_code};
use echora_lift::aarch64::{Instruction, decode};
use echora_lift::simd::{
    SimdOp, VecAcrossOp, VecIntOp, VecLongOp, VecNarrowOp, VecPermuteOp, VecShiftOp, VecUnaryOp,
    VectorLogicalOp,
};

const STATES_PER_WORD: usize = 16;
const STRIDE: u64 = 12;

#[derive(Clone, PartialEq, Eq)]
struct State {
    x: [u64; 31],
    v: [u128; 32],
    nzcv: u8,
}

impl std::fmt::Debug for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "State {{ ")?;
        for (n, value) in self.v.iter().enumerate() {
            write!(f, "v{n}: {value:#034x}, ")?;
        }
        write!(f, "x: {:x?}, nzcv: {} }}", self.x, self.nzcv)
    }
}

struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A vector: random, or built from edge lanes of one size.
    fn vector(&mut self) -> u128 {
        let pick = self.next();
        if pick.is_multiple_of(3) {
            let bits = 8 << ((pick >> 8) % 4);
            let mut value = 0;
            for lane in 0..128 / bits {
                let edge: u128 = match self.next() % 6 {
                    0 => 0,
                    1 => 1,
                    2 => mask(bits),
                    3 => 1 << (bits - 1),
                    4 => mask(bits - 1),
                    _ => u128::from(self.next()) & mask(bits),
                };
                value |= edge << (lane * bits);
            }
            value
        } else {
            (u128::from(self.next()) << 64) | u128::from(self.next())
        }
    }
}

fn mask(bits: u32) -> u128 {
    if bits >= 128 {
        u128::MAX
    } else {
        (1 << bits) - 1
    }
}

fn lane(v: u128, bits: u32, index: u32) -> u128 {
    (v >> (bits * index)) & mask(bits)
}

fn with_lane(v: u128, bits: u32, index: u32, value: u128) -> u128 {
    let at = bits * index;
    (v & !(mask(bits) << at)) | ((value & mask(bits)) << at)
}

fn signed(value: u128, bits: u32) -> i128 {
    let value = value & mask(bits);
    if bits < 128 && value >> (bits - 1) & 1 == 1 {
        value as i128 - (1i128 << bits)
    } else {
        value as i128
    }
}

/// Lanes as integers, signed or not.
fn int(value: u128, bits: u32, is_signed: bool) -> i128 {
    if is_signed {
        signed(value, bits)
    } else {
        value as i128
    }
}

fn saturate(value: i128, bits: u32, is_signed: bool) -> u128 {
    let (low, high) = if is_signed {
        (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1)
    } else {
        (0, (1i128 << bits) - 1)
    };
    (value.clamp(low, high) as u128) & mask(bits)
}

/// Rounding arithmetic shift right of an unbounded integer.
fn round_shift(value: i128, amount: u32) -> i128 {
    if amount == 0 {
        value
    } else if amount >= 100 {
        0
    } else {
        (value + (1i128 << (amount - 1))) >> amount
    }
}

/// A 64- or 128-bit result: the upper half cleared unless `q`.
fn width(value: u128, q: bool) -> u128 {
    if q { value } else { value & mask(64) }
}

fn lanes(q: bool, bits: u32) -> u32 {
    if q { 128 / bits } else { 64 / bits }
}

/// SSHL/USHL/SRSHL/URSHL: shift by the signed low byte of `amount`.
fn register_shift(a: u128, amount: u128, bits: u32, is_signed: bool, round: bool) -> u128 {
    let shift = i64::from(amount as u8 as i8);
    let element = int(a, bits, is_signed);
    let result = if shift >= 0 {
        if shift >= 64 { 0 } else { element << shift }
    } else {
        let right = (-shift) as u32;
        if round {
            round_shift(element, right)
        } else if right >= 100 {
            if element < 0 { -1 } else { 0 }
        } else {
            element >> right
        }
    };
    (result as u128) & mask(bits)
}

fn vec_int(op: VecIntOp, a: u128, b: u128, d: u128, bits: u32) -> u128 {
    let (sa, sb) = (signed(a, bits), signed(b, bits));
    let (ua, ub) = (a as i128, b as i128);
    let all = |condition: bool| if condition { mask(bits) } else { 0 };
    let result: u128 = match op {
        VecIntOp::Add => a.wrapping_add(b),
        VecIntOp::Sub => a.wrapping_sub(b),
        VecIntOp::Mul => a.wrapping_mul(b),
        VecIntOp::Mla => d.wrapping_add(a.wrapping_mul(b)),
        VecIntOp::Mls => d.wrapping_sub(a.wrapping_mul(b)),
        VecIntOp::CmEq => all(a == b),
        VecIntOp::CmTst => all(a & b != 0),
        VecIntOp::CmGt => all(sa > sb),
        VecIntOp::CmGe => all(sa >= sb),
        VecIntOp::CmHi => all(a > b),
        VecIntOp::CmHs => all(a >= b),
        VecIntOp::Sshl => register_shift(a, b, bits, true, false),
        VecIntOp::Ushl => register_shift(a, b, bits, false, false),
        VecIntOp::Srshl => register_shift(a, b, bits, true, true),
        VecIntOp::Urshl => register_shift(a, b, bits, false, true),
        VecIntOp::Smax => sa.max(sb) as u128,
        VecIntOp::Smin => sa.min(sb) as u128,
        VecIntOp::Umax => a.max(b),
        VecIntOp::Umin => a.min(b),
        VecIntOp::Uabd => (ua - ub).unsigned_abs(),
        VecIntOp::Sabd => (sa - sb).unsigned_abs(),
        VecIntOp::Sqadd => saturate(sa + sb, bits, true),
        VecIntOp::Uqadd => saturate(ua + ub, bits, false),
        VecIntOp::Sqsub => saturate(sa - sb, bits, true),
        VecIntOp::Uqsub => saturate(ua - ub, bits, false),
        VecIntOp::Shadd => ((sa + sb) >> 1) as u128,
        VecIntOp::Uhadd => ((ua + ub) >> 1) as u128,
        VecIntOp::Shsub => ((sa - sb) >> 1) as u128,
        VecIntOp::Uhsub => ((ua - ub) >> 1) as u128,
        VecIntOp::Srhadd => ((sa + sb + 1) >> 1) as u128,
        VecIntOp::Urhadd => ((ua + ub + 1) >> 1) as u128,
        VecIntOp::Sqdmulh => saturate((2 * sa * sb) >> bits, bits, true),
        VecIntOp::Sqrdmulh => saturate((2 * sa * sb + (1i128 << (bits - 1))) >> bits, bits, true),
        other => panic!("no reference for {other:?}"),
    };
    result & mask(bits)
}

fn vec_shift(op: VecShiftOp, a: u128, d: u128, bits: u32, shift: u32) -> u128 {
    let (sa, ua) = (signed(a, bits), a as i128);
    let right = |value: i128| {
        if shift >= 100 {
            value >> 99
        } else {
            value >> shift
        }
    };
    let result = match op {
        VecShiftOp::Shl => a << shift,
        VecShiftOp::Sshr => right(sa) as u128,
        VecShiftOp::Ushr => right(ua) as u128,
        VecShiftOp::Ssra => d.wrapping_add(right(sa) as u128),
        VecShiftOp::Usra => d.wrapping_add(right(ua) as u128),
        VecShiftOp::Srshr => round_shift(sa, shift) as u128,
        VecShiftOp::Urshr => round_shift(ua, shift) as u128,
        VecShiftOp::Srsra => d.wrapping_add(round_shift(sa, shift) as u128),
        VecShiftOp::Ursra => d.wrapping_add(round_shift(ua, shift) as u128),
        other => panic!("no reference for {other:?}"),
    };
    result & mask(bits)
}

fn vec_unary(op: VecUnaryOp, a: u128, bits: u32) -> u128 {
    let sa = signed(a, bits);
    let all = |condition: bool| if condition { mask(bits) } else { 0 };
    let result = match op {
        VecUnaryOp::Cnt => u128::from(a.count_ones()),
        VecUnaryOp::Not => !a,
        VecUnaryOp::Rbit => a.reverse_bits() >> (128 - bits),
        VecUnaryOp::Neg => (-sa) as u128,
        VecUnaryOp::Abs => sa.unsigned_abs(),
        VecUnaryOp::CmEqZero => all(sa == 0),
        VecUnaryOp::CmGeZero => all(sa >= 0),
        VecUnaryOp::CmGtZero => all(sa > 0),
        VecUnaryOp::CmLeZero => all(sa <= 0),
        VecUnaryOp::CmLtZero => all(sa < 0),
        VecUnaryOp::Clz => u128::from(a.leading_zeros() - (128 - bits)),
        VecUnaryOp::Cls => {
            let x = (a ^ (a >> 1)) & mask(bits - 1);
            u128::from(x.leading_zeros() - (128 - (bits - 1)))
        }
        // Lane reversals are handled per container by the caller.
        VecUnaryOp::Rev64 | VecUnaryOp::Rev32 | VecUnaryOp::Rev16 => a,
    };
    result & mask(bits)
}

fn carryless(a: u128, b: u128, bits: u32) -> u128 {
    let mut result = 0;
    for i in 0..bits {
        if (b >> i) & 1 == 1 {
            result ^= a << i;
        }
    }
    result
}

/// Runs `instruction` on `s`.
fn interpret(instruction: &SimdOp, s: &mut State) {
    use SimdOp::*;
    let v = |s: &State, n: u8| s.v[n as usize & 31];
    let x = |s: &State, n: u8| if n == 31 { 0 } else { s.x[n as usize] };
    match *instruction {
        VecInt {
            op,
            bits,
            q,
            scalar,
            rd,
            rn,
            rm,
        }
        | VecIntElement {
            op,
            bits,
            q,
            scalar,
            rd,
            rn,
            rm,
            ..
        } => {
            let bits = u32::from(bits);
            let count = if scalar { 1 } else { lanes(q, bits) };
            let (n, m, d) = (v(s, rn), v(s, rm), v(s, rd));
            let mut result = 0;
            for i in 0..count {
                let b = match *instruction {
                    VecIntElement { index, .. } => lane(m, bits, u32::from(index)),
                    _ => lane(m, bits, i),
                };
                let value = vec_int(op, lane(n, bits, i), b, lane(d, bits, i), bits);
                result = with_lane(result, bits, i, value);
            }
            s.v[rd as usize] = if scalar { result } else { width(result, q) };
        }
        VecShift {
            op,
            bits,
            q,
            scalar,
            rd,
            rn,
            shift,
        } => {
            let bits = u32::from(bits);
            let shift = u32::from(shift);
            let (n, d) = (v(s, rn), v(s, rd));
            if op == VecShiftOp::Shrn {
                // SHRN/SHRN2: `bits` is the source lane; `q` writes the
                // upper half and keeps the lower.
                let half = bits / 2;
                let mut result = if q { d & mask(64) } else { 0 };
                for i in 0..64 / half {
                    let value = (lane(n, bits, i) >> shift) & mask(half);
                    let at = if q { i + 64 / half } else { i };
                    result = with_lane(result, half, at, value);
                }
                s.v[rd as usize] = result;
                return;
            }
            let count = if scalar { 1 } else { lanes(q, bits) };
            let mut result = 0;
            for i in 0..count {
                let value = vec_shift(op, lane(n, bits, i), lane(d, bits, i), bits, shift);
                result = with_lane(result, bits, i, value);
            }
            s.v[rd as usize] = if scalar { result } else { width(result, q) };
        }
        VecIntPairwise {
            op,
            bits,
            q,
            scalar,
            rd,
            rn,
            rm,
        } => {
            let bits = u32::from(bits);
            let (n, m) = (v(s, rn), v(s, rm));
            if scalar {
                s.v[rd as usize] = vec_int(op, lane(n, 64, 0), lane(n, 64, 1), 0, 64);
                return;
            }
            let count = lanes(q, bits);
            let concat: Vec<u128> = (0..count)
                .map(|i| lane(n, bits, i))
                .chain((0..count).map(|i| lane(m, bits, i)))
                .collect();
            let mut result = 0;
            for i in 0..count {
                let (a, b) = (concat[2 * i as usize], concat[2 * i as usize + 1]);
                result = with_lane(result, bits, i, vec_int(op, a, b, 0, bits));
            }
            s.v[rd as usize] = width(result, q);
        }
        VecLong {
            op,
            signed: is_signed,
            bits,
            upper,
            rd,
            rn,
            rm,
        }
        | VecLongElement {
            op,
            signed: is_signed,
            bits,
            upper,
            rd,
            rn,
            rm,
            ..
        } => {
            let bits = u32::from(bits);
            let wide = 2 * bits;
            let (n, m, d) = (v(s, rn), v(s, rm), v(s, rd));
            let offset = if upper { 64 / bits } else { 0 };
            let mut result = 0;
            for i in 0..64 / bits {
                let a = match op {
                    VecLongOp::Addw | VecLongOp::Subw => int(lane(n, wide, i), wide, is_signed),
                    _ => int(lane(n, bits, i + offset), bits, is_signed),
                };
                let b = match *instruction {
                    VecLongElement { index, .. } => {
                        int(lane(m, bits, u32::from(index)), bits, is_signed)
                    }
                    _ => int(lane(m, bits, i + offset), bits, is_signed),
                };
                let accumulator = lane(d, wide, i) as i128;
                let value = match op {
                    VecLongOp::Mlal => accumulator + a * b,
                    VecLongOp::Mlsl => accumulator - a * b,
                    VecLongOp::Mull => a * b,
                    VecLongOp::Addl | VecLongOp::Addw => a + b,
                    VecLongOp::Subl | VecLongOp::Subw => a - b,
                    VecLongOp::Abal => accumulator + (a - b).abs(),
                    VecLongOp::Abdl => (a - b).abs(),
                };
                result = with_lane(result, wide, i, value as u128);
            }
            s.v[rd as usize] = result;
        }
        VecShiftLong {
            signed: is_signed,
            bits,
            upper,
            rd,
            rn,
            shift,
        } => {
            let bits = u32::from(bits);
            let n = v(s, rn);
            let offset = if upper { 64 / bits } else { 0 };
            let mut result = 0;
            for i in 0..64 / bits {
                let a = int(lane(n, bits, i + offset), bits, is_signed);
                result = with_lane(result, 2 * bits, i, (a << shift) as u128);
            }
            s.v[rd as usize] = result;
        }
        VecNarrow {
            op,
            round,
            bits,
            upper,
            scalar,
            rd,
            rn,
            shift,
        } => {
            let bits = u32::from(bits);
            let half = bits / 2;
            let (n, d) = (v(s, rn), v(s, rd));
            let count = if scalar { 1 } else { 64 / half };
            let mut result = if upper && !scalar { d & mask(64) } else { 0 };
            for i in 0..count {
                let a = int(lane(n, bits, i), bits, op != VecNarrowOp::Unsigned);
                let shifted = if round {
                    round_shift(a, u32::from(shift))
                } else {
                    a >> shift
                };
                let value = match op {
                    VecNarrowOp::Truncate => shifted as u128 & mask(half),
                    VecNarrowOp::Signed => saturate(shifted, half, true),
                    VecNarrowOp::Unsigned | VecNarrowOp::SignedToUnsigned => {
                        saturate(shifted, half, false)
                    }
                };
                let at = if upper && !scalar { i + count } else { i };
                result = with_lane(result, half, at, value);
            }
            s.v[rd as usize] = result;
        }
        VecPairwiseLong {
            signed: is_signed,
            accumulate,
            bits,
            q,
            rd,
            rn,
        } => {
            let bits = u32::from(bits);
            let (n, d) = (v(s, rn), v(s, rd));
            let mut result = 0;
            for i in 0..lanes(q, 2 * bits) {
                let sum = int(lane(n, bits, 2 * i), bits, is_signed)
                    + int(lane(n, bits, 2 * i + 1), bits, is_signed);
                let sum = if accumulate {
                    sum + lane(d, 2 * bits, i) as i128
                } else {
                    sum
                };
                result = with_lane(result, 2 * bits, i, sum as u128);
            }
            s.v[rd as usize] = width(result, q);
        }
        VecAcross {
            op,
            bits,
            q,
            rd,
            rn,
        } => {
            let bits = u32::from(bits);
            let n = v(s, rn);
            let values = (0..lanes(q, bits)).map(|i| lane(n, bits, i));
            let result = match op {
                VecAcrossOp::AddV => values.fold(0u128, u128::wrapping_add) & mask(bits),
                VecAcrossOp::UMaxV => values.max().unwrap_or(0),
                VecAcrossOp::UMinV => values.min().unwrap_or(0),
                VecAcrossOp::SMaxV => {
                    values.map(|a| signed(a, bits)).max().unwrap_or(0) as u128 & mask(bits)
                }
                VecAcrossOp::SMinV => {
                    values.map(|a| signed(a, bits)).min().unwrap_or(0) as u128 & mask(bits)
                }
                VecAcrossOp::SAddLV => {
                    values.map(|a| signed(a, bits)).sum::<i128>() as u128 & mask(2 * bits)
                }
                VecAcrossOp::UAddLV => values.sum::<u128>() & mask(2 * bits),
            };
            s.v[rd as usize] = result;
        }
        VecUnary {
            op,
            bits,
            q,
            scalar,
            rd,
            rn,
        } => {
            let bits = u32::from(bits);
            let n = v(s, rn);
            let count = if scalar { 1 } else { lanes(q, bits) };
            let container = match op {
                VecUnaryOp::Rev64 => Some(64),
                VecUnaryOp::Rev32 => Some(32),
                VecUnaryOp::Rev16 => Some(16),
                _ => None,
            };
            let mut result = 0;
            for i in 0..count {
                let value = match container {
                    // Reverse the lanes within each container.
                    Some(container) => {
                        let per = container / bits;
                        let source = (i / per) * per + (per - 1 - i % per);
                        lane(n, bits, source)
                    }
                    None => vec_unary(op, lane(n, bits, i), bits),
                };
                result = with_lane(result, bits, i, value);
            }
            s.v[rd as usize] = if scalar { result } else { width(result, q) };
        }
        VecPermute {
            op,
            bits,
            q,
            rd,
            rn,
            rm,
        } => {
            let bits = u32::from(bits);
            let (n, m) = (v(s, rn), v(s, rm));
            let count = lanes(q, bits);
            let pairs = count / 2;
            let mut result = 0;
            for i in 0..count {
                let value = match op {
                    VecPermuteOp::Uzp1 | VecPermuteOp::Uzp2 => {
                        let source = 2 * i + u32::from(op == VecPermuteOp::Uzp2);
                        if source < count {
                            lane(n, bits, source)
                        } else {
                            lane(m, bits, source - count)
                        }
                    }
                    VecPermuteOp::Trn1 | VecPermuteOp::Trn2 => {
                        let source = (i & !1) + u32::from(op == VecPermuteOp::Trn2);
                        lane(if i % 2 == 0 { n } else { m }, bits, source)
                    }
                    VecPermuteOp::Zip1 | VecPermuteOp::Zip2 => {
                        let base = if op == VecPermuteOp::Zip2 { pairs } else { 0 };
                        lane(if i % 2 == 0 { n } else { m }, bits, base + i / 2)
                    }
                };
                result = with_lane(result, bits, i, value);
            }
            s.v[rd as usize] = width(result, q);
        }
        VecExt {
            q,
            rd,
            rn,
            rm,
            index,
        } => {
            let bytes = if q { 16 } else { 8 };
            let (n, m) = (v(s, rn), v(s, rm));
            let mut result = 0;
            for i in 0..bytes {
                let at = u32::from(index) + i;
                let value = if at < bytes {
                    lane(n, 8, at)
                } else {
                    lane(m, 8, at - bytes)
                };
                result = with_lane(result, 8, i, value);
            }
            s.v[rd as usize] = result;
        }
        VecTable {
            q,
            extend,
            rd,
            rn,
            regs,
            rm,
        } => {
            let table: Vec<u128> = (0..u32::from(regs) * 16)
                .map(|i| lane(v(s, rn.wrapping_add((i / 16) as u8)), 8, i % 16))
                .collect();
            let (m, d) = (v(s, rm), v(s, rd));
            let mut result = 0;
            for i in 0..if q { 16 } else { 8 } {
                let index = lane(m, 8, i) as usize;
                let value = match table.get(index) {
                    Some(&value) => value,
                    None if extend => lane(d, 8, i),
                    None => 0,
                };
                result = with_lane(result, 8, i, value);
            }
            s.v[rd as usize] = result;
        }
        VecDot {
            signed: is_signed,
            q,
            rd,
            rn,
            rm,
        } => {
            let (n, m, d) = (v(s, rn), v(s, rm), v(s, rd));
            let mut result = 0;
            for i in 0..lanes(q, 32) {
                let mut sum = lane(d, 32, i) as i128;
                for k in 0..4 {
                    sum += int(lane(n, 8, 4 * i + k), 8, is_signed)
                        * int(lane(m, 8, 4 * i + k), 8, is_signed);
                }
                result = with_lane(result, 32, i, sum as u128);
            }
            s.v[rd as usize] = width(result, q);
        }
        VecPolyMulLong {
            bits,
            upper,
            rd,
            rn,
            rm,
        } => {
            let bits = u32::from(bits);
            let (n, m) = (v(s, rn), v(s, rm));
            let offset = if upper { 64 / bits } else { 0 };
            let mut result = 0;
            for i in 0..64 / bits {
                let product = carryless(lane(n, bits, i + offset), lane(m, bits, i + offset), bits);
                result = with_lane(result, 2 * bits, i, product);
            }
            s.v[rd as usize] = result;
        }
        VectorLogical { op, rd, rn, rm, q } => {
            let (n, m, d) = (v(s, rn), v(s, rm), v(s, rd));
            let result = match op {
                VectorLogicalOp::And => n & m,
                VectorLogicalOp::Bic => n & !m,
                VectorLogicalOp::Orr => n | m,
                VectorLogicalOp::Orn => n | !m,
                VectorLogicalOp::Eor => n ^ m,
                VectorLogicalOp::Bsl => (n & d) | (m & !d),
                VectorLogicalOp::Bit => (d & !m) | (n & m),
                VectorLogicalOp::Bif => (d & m) | (n & !m),
            };
            s.v[rd as usize] = width(result, q);
        }
        DupElement {
            rd,
            rn,
            bytes,
            index,
            q,
            scalar,
        } => {
            let bits = u32::from(bytes) * 8;
            let value = lane(v(s, rn), bits, u32::from(index));
            let count = if scalar { 1 } else { lanes(q, bits) };
            s.v[rd as usize] = (0..count).fold(0, |r, i| with_lane(r, bits, i, value));
        }
        DupGeneral { rd, rn, bytes, q } => {
            let bits = u32::from(bytes) * 8;
            let value = u128::from(x(s, rn));
            s.v[rd as usize] = (0..lanes(q, bits)).fold(0, |r, i| with_lane(r, bits, i, value));
        }
        InsertElement {
            rd,
            rn,
            bytes,
            dst,
            src,
        } => {
            let bits = u32::from(bytes) * 8;
            let value = lane(v(s, rn), bits, u32::from(src));
            s.v[rd as usize] = with_lane(v(s, rd), bits, u32::from(dst), value);
        }
        InsertGeneral {
            rd,
            rn,
            bytes,
            index,
        } => {
            let bits = u32::from(bytes) * 8;
            let value = u128::from(x(s, rn));
            s.v[rd as usize] = with_lane(v(s, rd), bits, u32::from(index), value);
        }
        MoveToGeneral {
            rd,
            rn,
            bytes,
            index,
            signed: is_signed,
            gp_bits,
        } => {
            let bits = u32::from(bytes) * 8;
            let value = lane(v(s, rn), bits, u32::from(index));
            let value = if is_signed {
                signed(value, bits) as u128 & mask(u32::from(gp_bits))
            } else {
                value
            };
            if rd != 31 {
                s.x[rd as usize] = value as u64;
            }
        }
        MoveImmediate { rd, value } => s.v[rd as usize] = value,
        ModifyImmediate {
            rd,
            value,
            clear,
            q,
        } => {
            let d = v(s, rd);
            let result = if clear { d & !value } else { d | value };
            s.v[rd as usize] = width(result, q);
        }
        ref other => panic!("no reference for {other:?}"),
    }
}

fn run_lifted(m: &mut Machine, entry: u64, state: &State) -> Result<State, Trap> {
    m.core.cpu.regs = state.x;
    m.core.cpu.vectors = state.v;
    m.core.cpu.nzcv = state.nzcv;
    m.core.cpu.pc = entry;
    let trap = loop {
        match simd_semantic_code::dispatch(&mut m.core) {
            Ok(()) => continue,
            Err(trap) => break trap,
        }
    };
    m.core.settle_block_trap();
    if !matches!(trap, Trap::SupervisorCall { number: 7 }) {
        return Err(trap);
    }
    Ok(State {
        x: m.core.cpu.regs,
        v: m.core.cpu.vectors,
        nzcv: m.core.cpu.nzcv,
    })
}

fn difference(expected: &State, actual: &State) -> String {
    let mut parts = Vec::new();
    for n in 0..32 {
        if expected.v[n] != actual.v[n] {
            parts.push(format!(
                "v{n}: expected {:#034x}, lifted {:#034x}",
                expected.v[n], actual.v[n]
            ));
        }
    }
    for n in 0..31 {
        if expected.x[n] != actual.x[n] {
            parts.push(format!(
                "x{n}: expected {:#x}, lifted {:#x}",
                expected.x[n], actual.x[n]
            ));
        }
    }
    if expected.nzcv != actual.nzcv {
        parts.push(format!(
            "nzcv: expected {}, lifted {}",
            expected.nzcv, actual.nzcv
        ));
    }
    parts.join("; ")
}

#[test]
fn lifted_integer_simd_matches_the_reference() {
    let words = simd_semantic_code::WORDS;
    let code_size = (words.len() as u64 * STRIDE).next_multiple_of(0x1000);
    let mut image = Vec::new();
    for word in words {
        image.extend(word.to_le_bytes());
        image.extend(0xd400_00e1_u32.to_le_bytes());
        image.extend(0xd400_0101_u32.to_le_bytes());
    }
    let mut m = Machine::with_layout(
        simd_semantic_code::BASE,
        &image,
        code_size,
        0x80_0000,
        0x1000,
    );
    simd_semantic_code::install();
    let mut random = Random(0x7a3c_91e5_02bd_4f68);
    let mut failures = Vec::new();
    let mut runs = 0;
    for (index, &word) in words.iter().enumerate() {
        let entry = simd_semantic_code::BASE + STRIDE * index as u64;
        let Instruction::Simd(op) = decode(word, entry) else {
            panic!("{word:#010x} is not a SIMD instruction");
        };
        for _ in 0..STATES_PER_WORD {
            let before = State {
                x: std::array::from_fn(|_| random.next()),
                v: std::array::from_fn(|_| random.vector()),
                nzcv: (random.next() & 15) as u8,
            };
            let mut expected = before.clone();
            interpret(&op, &mut expected);
            runs += 1;
            let problem = match run_lifted(&mut m, entry, &before) {
                Ok(actual) if actual == expected => continue,
                Ok(actual) => difference(&expected, &actual),
                Err(trap) => format!("lifted trapped: {trap:?}"),
            };
            failures.push(format!(
                "{word:#010x} {op:?}: {problem}\n    from {before:?}"
            ));
            break;
        }
    }
    println!(
        "{} instructions, {runs} runs, {} mismatching instructions",
        words.len(),
        failures.len()
    );
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures
            .iter()
            .take(30)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}
