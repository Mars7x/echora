//! Semantic differential test for floating point: lifted scalar and vector
//! FP instructions against a reference written from the Arm pseudocode,
//! rounding to nearest even with FPCR.FZ (flush to zero) and FPCR.DN
//! (default NaN) each set at random, comparing the FPSR flags the runtime
//! models: IOC, DZC, OFC and IDC.
//!
//! The reference follows FPProcessNaNs (first SNaN, then first QNaN, in
//! operand order, quieted) and FPProcessNaNs3 for fused operations (the
//! addend first, and the default NaN when a QNaN addend meets 0 x inf);
//! the default NaN for invalid operations; FPNeg before fusing (FMSUB,
//! FNMADD, FNMSUB, FMLS), which flips a propagated NaN's sign; the signed
//! zero rules of FMAX/FMIN and the QNaN rule of FMAXNM/FMINNM; FMULX's
//! 0 x inf = 2; FCVT's NaN conversion; and saturating conversions to
//! integers in every rounding mode. Inputs include NaNs, infinities,
//! zeros, denormals and edge integers. Under FZ, denormal inputs of
//! arithmetic, compares and conversions become signed zeros (IDC) and tiny
//! results flush to zero; under DN every NaN result is the default NaN.
//! UFC, IXC and QC are not modeled by the runtime and not compared. A run
//! whose FZ result rounds to exactly the smallest normal is skipped: the
//! rounded value alone cannot tell whether the exact one was tiny.
#![cfg(any(target_os = "linux", windows))]
#![allow(clippy::unwrap_used, clippy::float_cmp)]

use echora_aot_tests::{Machine, Trap, fp_semantic_code};
use echora_lift::aarch64::{Instruction, decode};
use echora_lift::simd::{
    FpBinaryOp, FpPrecision, FpSize, FpTernaryOp, FpUnaryOp, Rounding, SimdOp, VecFpOp,
    VecFpUnaryOp,
};

const STATES_PER_WORD: usize = 16;
const STRIDE: u64 = 12;

#[derive(Clone, PartialEq, Eq)]
struct State {
    x: [u64; 31],
    v: [u128; 32],
    nzcv: u8,
    fpcr: u32,
    /// Only `FLAGS` are compared.
    fpsr: u32,
}

const IOC: u32 = 1;
const DZC: u32 = 1 << 1;
const OFC: u32 = 1 << 2;
const IDC: u32 = 1 << 7;
/// The cumulative FPSR flags the runtime models.
const FLAGS: u32 = IOC | DZC | OFC | IDC;
const FPCR_FZ: u32 = 1 << 24;
const FPCR_DN: u32 = 1 << 25;

/// The FP mode of the current reference run and what it raised.
#[derive(Clone, Copy, Default)]
struct Context {
    flush: bool,
    default_nan: bool,
    flags: u32,
    /// An FZ result rounded to exactly the smallest normal.
    ambiguous: bool,
}

thread_local! {
    static CONTEXT: std::cell::Cell<Context> = const {
        std::cell::Cell::new(Context {
            flush: false,
            default_nan: false,
            flags: 0,
            ambiguous: false,
        })
    };
}

fn context() -> Context {
    CONTEXT.with(std::cell::Cell::get)
}

fn update(f: impl FnOnce(&mut Context)) {
    CONTEXT.with(|cell| {
        let mut context = cell.get();
        f(&mut context);
        cell.set(context);
    });
}

fn raise(flag: u32) {
    update(|context| context.flags |= flag);
}

fn min_normal(size: u32) -> u64 {
    1 << fraction_bits(size)
}

fn is_denormal(bits: u64, size: u32) -> bool {
    let magnitude = absolute(bits, size);
    magnitude != 0 && magnitude < min_normal(size)
}

/// FPUnpack's flush: under FZ a denormal input is a zero of its sign (IDC).
fn input(bits: u64, size: u32) -> u64 {
    if context().flush && is_denormal(bits, size) {
        raise(IDC);
        bits & sign_bit(size)
    } else {
        bits
    }
}

/// A propagated NaN, or the default NaN under DN.
fn nan_result(nan: u64, size: u32) -> u64 {
    if context().default_nan {
        default_nan(size)
    } else {
        nan
    }
}

fn finite(values: &[u64], size: u32) -> bool {
    values
        .iter()
        .all(|&value| value_of(value, size).is_finite())
}

/// FPRound's view of a computed result: an invalid operation gives the
/// default NaN (IOC), an infinity from finite inputs is an overflow (OFC),
/// and under FZ a tiny result flushes to a zero of its sign.
fn round_result(bits: u64, size: u32, finite: bool) -> u64 {
    if is_nan(bits, size) {
        raise(IOC);
        return default_nan(size);
    }
    if finite && is_infinite(bits, size) {
        raise(OFC);
    }
    if context().flush {
        if absolute(bits, size) == min_normal(size) {
            update(|context| context.ambiguous = true);
        }
        if is_denormal(bits, size) {
            return bits & sign_bit(size);
        }
    }
    bits
}

impl std::fmt::Debug for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "State {{ ")?;
        for (n, value) in self.v.iter().enumerate() {
            write!(f, "v{n}: {value:#034x}, ")?;
        }
        write!(
            f,
            "x: {:x?}, nzcv: {}, fpcr: {:#x}, fpsr: {:#x} }}",
            self.x, self.nzcv, self.fpcr, self.fpsr
        )
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

    /// A floating-point value of `size` bits, often a special one.
    fn float(&mut self, size: u32) -> u64 {
        let pick = self.next() % 16;
        let sign = (self.next() & 1) << (size - 1);
        let (exponent_bits, fraction_bits) = if size == 32 { (8, 23) } else { (11, 52) };
        let exponent_max = (1u64 << exponent_bits) - 1;
        let fraction = self.next() & ((1 << fraction_bits) - 1);
        let quiet = 1 << (fraction_bits - 1);
        let value = match pick {
            0 => 0,
            1 => exponent_max << fraction_bits,
            2 => (exponent_max << fraction_bits) | quiet | (fraction & (quiet - 1)),
            3 => (exponent_max << fraction_bits) | ((fraction & (quiet - 1)) | 1),
            4 => fraction.max(1),
            5 => {
                let one: f64 = 1.0;
                bits_of(one * f64::from((self.next() % 9) as u32) * 0.5, size)
            }
            6 => {
                let integer = (self.next() % 0x1_0000) as f64;
                bits_of(integer * 2f64.powi((self.next() % 80) as i32 - 16), size)
            }
            7 => ((exponent_max - 1) << fraction_bits) | fraction,
            _ => self.next() & mask(size),
        };
        (value | sign) & mask(size)
    }

    fn vector(&mut self, size: u32) -> u128 {
        let mut value = 0;
        for lane in 0..128 / size {
            value |= u128::from(self.float(size)) << (lane * size);
        }
        value
    }
}

fn mask(bits: u32) -> u64 {
    if bits == 64 {
        u64::MAX
    } else {
        (1 << bits) - 1
    }
}

fn bits_of(value: f64, size: u32) -> u64 {
    if size == 32 {
        u64::from((value as f32).to_bits())
    } else {
        value.to_bits()
    }
}

/// The value as an f64 (exact for both sizes).
fn value_of(bits: u64, size: u32) -> f64 {
    if size == 32 {
        f64::from(f32::from_bits(bits as u32))
    } else {
        f64::from_bits(bits)
    }
}

fn fraction_bits(size: u32) -> u32 {
    if size == 32 { 23 } else { 52 }
}

fn is_nan(bits: u64, size: u32) -> bool {
    value_of(bits, size).is_nan()
}

fn is_snan(bits: u64, size: u32) -> bool {
    is_nan(bits, size) && bits >> (fraction_bits(size) - 1) & 1 == 0
}

fn is_qnan(bits: u64, size: u32) -> bool {
    is_nan(bits, size) && !is_snan(bits, size)
}

fn quiet(bits: u64, size: u32) -> u64 {
    bits | 1 << (fraction_bits(size) - 1)
}

fn default_nan(size: u32) -> u64 {
    if size == 32 {
        0x7fc0_0000
    } else {
        0x7ff8_0000_0000_0000
    }
}

fn sign_bit(size: u32) -> u64 {
    1 << (size - 1)
}

fn negate(bits: u64, size: u32) -> u64 {
    bits ^ sign_bit(size)
}

fn absolute(bits: u64, size: u32) -> u64 {
    bits & !sign_bit(size)
}

fn is_zero(bits: u64, size: u32) -> bool {
    absolute(bits, size) == 0
}

fn is_infinite(bits: u64, size: u32) -> bool {
    value_of(bits, size).is_infinite()
}

fn infinity(negative: bool, size: u32) -> u64 {
    let bits = bits_of(f64::INFINITY, size);
    if negative { negate(bits, size) } else { bits }
}

/// FPProcessNaN: a signalling NaN raises IOC; DN gives the default NaN.
fn process_nan(a: u64, size: u32) -> Option<u64> {
    if is_snan(a, size) {
        raise(IOC);
        Some(nan_result(quiet(a, size), size))
    } else if is_nan(a, size) {
        Some(nan_result(a, size))
    } else {
        None
    }
}

/// FPProcessNaNs.
fn process_nans(a: u64, b: u64, size: u32) -> Option<u64> {
    if is_snan(a, size) {
        process_nan(a, size)
    } else if is_snan(b, size) {
        process_nan(b, size)
    } else if is_nan(a, size) {
        process_nan(a, size)
    } else if is_nan(b, size) {
        process_nan(b, size)
    } else {
        None
    }
}

/// FPProcessNaNs3.
fn process_nans3(a: u64, b: u64, c: u64, size: u32) -> Option<u64> {
    let first = [a, b, c]
        .into_iter()
        .find(|&value| is_snan(value, size))
        .or_else(|| [a, b, c].into_iter().find(|&value| is_nan(value, size)))?;
    process_nan(first, size)
}

/// An arithmetic result of non-NaN inputs in the operation's own
/// precision, through `round_result`.
fn compute(size: u32, finite: bool, f32_op: impl Fn() -> f32, f64_op: impl Fn() -> f64) -> u64 {
    let bits = if size == 32 {
        u64::from(f32_op().to_bits())
    } else {
        f64_op().to_bits()
    };
    round_result(bits, size, finite)
}

fn f32_of(bits: u64) -> f32 {
    f32::from_bits(bits as u32)
}

fn f64_of(bits: u64) -> f64 {
    f64::from_bits(bits)
}

fn add(a: u64, b: u64, size: u32) -> u64 {
    let (a, b) = (input(a, size), input(b, size));
    process_nans(a, b, size).unwrap_or_else(|| {
        compute(
            size,
            finite(&[a, b], size),
            || f32_of(a) + f32_of(b),
            || f64_of(a) + f64_of(b),
        )
    })
}

fn sub(a: u64, b: u64, size: u32) -> u64 {
    let (a, b) = (input(a, size), input(b, size));
    process_nans(a, b, size).unwrap_or_else(|| {
        compute(
            size,
            finite(&[a, b], size),
            || f32_of(a) - f32_of(b),
            || f64_of(a) - f64_of(b),
        )
    })
}

fn mul(a: u64, b: u64, size: u32) -> u64 {
    let (a, b) = (input(a, size), input(b, size));
    process_nans(a, b, size).unwrap_or_else(|| {
        compute(
            size,
            finite(&[a, b], size),
            || f32_of(a) * f32_of(b),
            || f64_of(a) * f64_of(b),
        )
    })
}

fn div(a: u64, b: u64, size: u32) -> u64 {
    let (a, b) = (input(a, size), input(b, size));
    process_nans(a, b, size).unwrap_or_else(|| {
        let dividend = value_of(a, size);
        if is_zero(b, size) && dividend.is_finite() && dividend != 0.0 {
            raise(DZC);
        }
        // A quotient by zero is an exact infinity, not an overflow.
        let finite = finite(&[a, b], size) && !is_zero(b, size);
        compute(
            size,
            finite,
            || f32_of(a) / f32_of(b),
            || f64_of(a) / f64_of(b),
        )
    })
}

fn mulx(a: u64, b: u64, size: u32) -> u64 {
    let (a, b) = (input(a, size), input(b, size));
    if let Some(nan) = process_nans(a, b, size) {
        return nan;
    }
    let zero_times_infinity =
        (is_zero(a, size) && is_infinite(b, size)) || (is_infinite(a, size) && is_zero(b, size));
    if zero_times_infinity {
        let negative = (a ^ b) & sign_bit(size) != 0;
        let two = bits_of(2.0, size);
        return if negative { negate(two, size) } else { two };
    }
    mul(a, b, size)
}

/// FPMax/FPMin.
fn max_min(a: u64, b: u64, size: u32, max: bool) -> u64 {
    let (a, b) = (input(a, size), input(b, size));
    if let Some(nan) = process_nans(a, b, size) {
        return nan;
    }
    let (x, y) = (value_of(a, size), value_of(b, size));
    if x == 0.0 && y == 0.0 {
        let (negative_a, negative_b) = (a & sign_bit(size) != 0, b & sign_bit(size) != 0);
        let negative = if max {
            negative_a && negative_b
        } else {
            negative_a || negative_b
        };
        return if negative { sign_bit(size) } else { 0 };
    }
    let first = if max { x > y } else { x < y };
    if first { a } else { b }
}

/// FPMaxNum/FPMinNum: a quiet NaN against a number loses.
fn max_min_number(mut a: u64, mut b: u64, size: u32, max: bool) -> u64 {
    let (quiet_a, quiet_b) = (is_qnan(a, size), is_qnan(b, size));
    if quiet_a && !quiet_b {
        a = infinity(max, size);
    } else if quiet_b && !quiet_a {
        b = infinity(max, size);
    }
    max_min(a, b, size, max)
}

/// FPMulAdd: addend + n * m, fused.
fn mul_add(addend: u64, n: u64, m: u64, size: u32) -> u64 {
    let (addend, n, m) = (input(addend, size), input(n, size), input(m, size));
    let nan = process_nans3(addend, n, m, size);
    let invalid_product =
        (is_infinite(n, size) && is_zero(m, size)) || (is_zero(n, size) && is_infinite(m, size));
    if is_qnan(addend, size) && invalid_product {
        raise(IOC);
        return default_nan(size);
    }
    if let Some(nan) = nan {
        return nan;
    }
    compute(
        size,
        finite(&[addend, n, m], size),
        || f32_of(n).mul_add(f32_of(m), f32_of(addend)),
        || f64_of(n).mul_add(f64_of(m), f64_of(addend)),
    )
}

fn sqrt(a: u64, size: u32) -> u64 {
    let a = input(a, size);
    process_nan(a, size).unwrap_or_else(|| {
        compute(
            size,
            finite(&[a], size),
            || f32_of(a).sqrt(),
            || f64_of(a).sqrt(),
        )
    })
}

fn round_value(value: f64, rounding: Rounding) -> f64 {
    match rounding {
        Rounding::Nearest => value.round_ties_even(),
        Rounding::Plus => value.ceil(),
        Rounding::Minus => value.floor(),
        Rounding::Zero => value.trunc(),
        Rounding::Away => value.round(),
    }
}

/// FRINT<r>: the result keeps the input's sign, even when zero.
fn round_integral(a: u64, rounding: Rounding, size: u32) -> u64 {
    let a = input(a, size);
    if let Some(nan) = process_nan(a, size) {
        return nan;
    }
    let rounded = round_value(value_of(a, size), rounding);
    let bits = bits_of(rounded, size);
    if rounded == 0.0 {
        (a & sign_bit(size)) | absolute(bits, size)
    } else {
        bits
    }
}

/// FPCompare flags: unordered 0011, equal 0110, less 1000, greater 0010.
/// A signalling NaN raises IOC, and so does any NaN when `signaling`.
fn compare(a: u64, b: u64, size: u32, signaling: bool) -> u8 {
    let (a, b) = (input(a, size), input(b, size));
    let (x, y) = (value_of(a, size), value_of(b, size));
    if x.is_nan() || y.is_nan() {
        if signaling || is_snan(a, size) || is_snan(b, size) {
            raise(IOC);
        }
        0b0011
    } else if x == y {
        0b0110
    } else if x < y {
        0b1000
    } else {
        0b0010
    }
}

/// A vector compare (FCMEQ, FCMGE, FCMGT and friends): true or false, with
/// FPCompareEQ's or FPCompareGE/GT's exceptions.
fn compare_lanes(a: u64, b: u64, size: u32, signaling: bool, test: fn(f64, f64) -> bool) -> u64 {
    let (a, b) = (input(a, size), input(b, size));
    let (x, y) = (value_of(a, size), value_of(b, size));
    if x.is_nan() || y.is_nan() {
        if signaling || is_snan(a, size) || is_snan(b, size) {
            raise(IOC);
        }
        return 0;
    }
    if test(x, y) { mask(size) } else { 0 }
}

fn condition_holds(condition: u8, flags: u8) -> bool {
    let (n, z, c, v) = (
        flags & 8 != 0,
        flags & 4 != 0,
        flags & 2 != 0,
        flags & 1 != 0,
    );
    let result = match condition >> 1 {
        0 => z,
        1 => c,
        2 => n,
        3 => v,
        4 => c && !z,
        5 => n == v,
        6 => n == v && !z,
        _ => true,
    };
    if condition & 1 == 1 && condition != 15 {
        !result
    } else {
        result
    }
}

/// FCVT between single and double precision.
fn convert(a: u64, from: u32, to: u32) -> u64 {
    if from == to {
        return a;
    }
    let a = input(a, from);
    if is_nan(a, from) {
        if is_snan(a, from) {
            raise(IOC);
        }
        if context().default_nan {
            return default_nan(to);
        }
        let negative = a & sign_bit(from) != 0;
        let fraction = a & mask(fraction_bits(from));
        let fraction = if to > from {
            fraction << 29
        } else {
            fraction >> 29
        };
        let base = quiet(bits_of(f64::INFINITY, to) | fraction, to);
        return if negative { negate(base, to) } else { base };
    }
    round_result(bits_of(value_of(a, from), to), to, finite(&[a], from))
}

/// FPToFixed: scaled by 2^fbits, rounded, saturated; NaN gives 0. A NaN or
/// out-of-range value raises IOC.
fn to_integer(a: u64, size: u32, fbits: u8, rounding: Rounding, bits: u32, signed: bool) -> u64 {
    let a = input(a, size);
    let value = value_of(a, size) * 2f64.powi(i32::from(fbits));
    if value.is_nan() {
        raise(IOC);
        return 0;
    }
    // `as i128` saturates beyond the i128 range; clamp as integers (an
    // f64 bound such as 2^63 - 1 is not representable).
    let rounded = round_value(value, rounding) as i128;
    let (low, high) = if signed {
        (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1)
    } else {
        (0, (1i128 << bits) - 1)
    };
    if rounded < low || rounded > high {
        raise(IOC);
    }
    (rounded.clamp(low, high) as u128 as u64) & mask(bits)
}

/// FixedToFP: rounded to nearest even, then scaled by 2^-fbits (exact).
fn from_integer(value: u64, bits: u32, signed: bool, fbits: u8, size: u32) -> u64 {
    let value = value & mask(bits);
    let scale = 2f64.powi(-i32::from(fbits));
    if size == 32 {
        let converted = if signed {
            (((value << (64 - bits)) as i64) >> (64 - bits)) as f32
        } else {
            value as f32
        };
        u64::from((converted * scale as f32).to_bits())
    } else {
        let converted = if signed {
            (((value << (64 - bits)) as i64) >> (64 - bits)) as f64
        } else {
            value as f64
        };
        (converted * scale).to_bits()
    }
}

fn lane(v: u128, size: u32, index: u32) -> u64 {
    ((v >> (size * index)) as u64) & mask(size)
}

fn with_lane(v: u128, size: u32, index: u32, value: u64) -> u128 {
    let at = size * index;
    let m = u128::from(mask(size)) << at;
    (v & !m) | (u128::from(value & mask(size)) << at)
}

fn fp_size(size: FpSize) -> u32 {
    match size {
        FpSize::S => 32,
        FpSize::D => 64,
    }
}

fn precision(p: FpPrecision) -> u32 {
    match p {
        FpPrecision::H => 16,
        FpPrecision::S => 32,
        FpPrecision::D => 64,
    }
}

fn vec_fp(op: VecFpOp, n: u64, m: u64, d: u64, size: u32) -> u64 {
    let (abs_n, abs_m) = (absolute(n, size), absolute(m, size));
    match op {
        VecFpOp::Add => add(n, m, size),
        VecFpOp::Sub => sub(n, m, size),
        VecFpOp::Mul => mul(n, m, size),
        VecFpOp::Div => div(n, m, size),
        VecFpOp::Max => max_min(n, m, size, true),
        VecFpOp::Min => max_min(n, m, size, false),
        VecFpOp::MaxNum => max_min_number(n, m, size, true),
        VecFpOp::MinNum => max_min_number(n, m, size, false),
        VecFpOp::AbsDiff => absolute(sub(n, m, size), size),
        VecFpOp::MulExtended => mulx(n, m, size),
        VecFpOp::CmEq => compare_lanes(n, m, size, false, |x, y| x == y),
        VecFpOp::CmGe => compare_lanes(n, m, size, true, |x, y| x >= y),
        VecFpOp::CmGt => compare_lanes(n, m, size, true, |x, y| x > y),
        VecFpOp::AcGe => compare_lanes(abs_n, abs_m, size, true, |x, y| x >= y),
        VecFpOp::AcGt => compare_lanes(abs_n, abs_m, size, true, |x, y| x > y),
        VecFpOp::MulAdd => mul_add(d, n, m, size),
        VecFpOp::MulSub => mul_add(d, negate(n, size), m, size),
        other => panic!("no reference for {other:?}"),
    }
}

fn vec_fp_unary(op: VecFpUnaryOp, a: u64, size: u32, fbits: u8) -> u64 {
    match op {
        VecFpUnaryOp::Abs => absolute(a, size),
        VecFpUnaryOp::Neg => negate(a, size),
        VecFpUnaryOp::Sqrt => sqrt(a, size),
        VecFpUnaryOp::Round(rounding) => round_integral(a, rounding, size),
        VecFpUnaryOp::ToSigned(rounding) => to_integer(a, size, fbits, rounding, size, true),
        VecFpUnaryOp::ToUnsigned(rounding) => to_integer(a, size, fbits, rounding, size, false),
        VecFpUnaryOp::FromSigned => from_integer(a, size, true, fbits, size),
        VecFpUnaryOp::FromUnsigned => from_integer(a, size, false, fbits, size),
        // FPCompareEQ/GE/GT against +0.0; LE and LT swap the operands.
        VecFpUnaryOp::CmEqZero => compare_lanes(a, 0, size, false, |x, y| x == y),
        VecFpUnaryOp::CmGeZero => compare_lanes(a, 0, size, true, |x, y| x >= y),
        VecFpUnaryOp::CmGtZero => compare_lanes(a, 0, size, true, |x, y| x > y),
        VecFpUnaryOp::CmLeZero => compare_lanes(0, a, size, true, |x, y| x >= y),
        VecFpUnaryOp::CmLtZero => compare_lanes(0, a, size, true, |x, y| x > y),
        other => panic!("no reference for {other:?}"),
    }
}

fn pairwise(op: VecFpOp, a: u64, b: u64, size: u32) -> u64 {
    vec_fp(op, a, b, 0, size)
}

/// Runs `instruction` on `s`.
fn interpret(instruction: &SimdOp, s: &mut State) {
    use SimdOp::*;
    let v = |s: &State, n: u8| s.v[n as usize];
    let x = |s: &State, n: u8| if n == 31 { 0 } else { s.x[n as usize] };
    let scalar = |value: u64| u128::from(value);
    match *instruction {
        FpUnary { op, size, rd, rn } => {
            let size = fp_size(size);
            let a = lane(v(s, rn), size, 0);
            let result = match op {
                FpUnaryOp::Mov => a,
                FpUnaryOp::Abs => absolute(a, size),
                FpUnaryOp::Neg => negate(a, size),
                FpUnaryOp::Sqrt => sqrt(a, size),
                FpUnaryOp::Round(rounding) => round_integral(a, rounding, size),
                FpUnaryOp::RoundExact => round_integral(a, Rounding::Nearest, size),
            };
            s.v[rd as usize] = scalar(result);
        }
        FpConvert { rd, rn, from, to } => {
            let (from, to) = (fp_size(from), fp_size(to));
            s.v[rd as usize] = scalar(convert(lane(v(s, rn), from, 0), from, to));
        }
        FpBinary {
            op,
            size,
            rd,
            rn,
            rm,
        } => {
            let size = fp_size(size);
            let (a, b) = (lane(v(s, rn), size, 0), lane(v(s, rm), size, 0));
            let result = match op {
                FpBinaryOp::Mul => mul(a, b, size),
                FpBinaryOp::Div => div(a, b, size),
                FpBinaryOp::Add => add(a, b, size),
                FpBinaryOp::Sub => sub(a, b, size),
                FpBinaryOp::Max => max_min(a, b, size, true),
                FpBinaryOp::Min => max_min(a, b, size, false),
                FpBinaryOp::MaxNum => max_min_number(a, b, size, true),
                FpBinaryOp::MinNum => max_min_number(a, b, size, false),
                FpBinaryOp::NegMul => negate(mul(a, b, size), size),
            };
            s.v[rd as usize] = scalar(result);
        }
        FpTernary {
            op,
            size,
            rd,
            rn,
            rm,
            ra,
        } => {
            let size = fp_size(size);
            let (n, m, a) = (
                lane(v(s, rn), size, 0),
                lane(v(s, rm), size, 0),
                lane(v(s, ra), size, 0),
            );
            let result = match op {
                FpTernaryOp::MulAdd => mul_add(a, n, m, size),
                FpTernaryOp::MulSub => mul_add(a, negate(n, size), m, size),
                FpTernaryOp::NegMulAdd => mul_add(negate(a, size), negate(n, size), m, size),
                FpTernaryOp::NegMulSub => mul_add(negate(a, size), n, m, size),
            };
            s.v[rd as usize] = scalar(result);
        }
        FpCompare {
            size,
            rn,
            rm,
            signaling,
        } => {
            let size = fp_size(size);
            let b = rm.map_or(0, |rm| lane(v(s, rm), size, 0));
            s.nzcv = compare(lane(v(s, rn), size, 0), b, size, signaling);
        }
        FpConditionalCompare {
            size,
            rn,
            rm,
            condition,
            nzcv,
            signaling,
        } => {
            let size = fp_size(size);
            s.nzcv = if condition_holds(condition, s.nzcv) {
                compare(
                    lane(v(s, rn), size, 0),
                    lane(v(s, rm), size, 0),
                    size,
                    signaling,
                )
            } else {
                nzcv
            };
        }
        FpSelect {
            size,
            rd,
            rn,
            rm,
            condition,
        } => {
            let size = fp_size(size);
            let source = if condition_holds(condition, s.nzcv) {
                rn
            } else {
                rm
            };
            s.v[rd as usize] = scalar(lane(v(s, source), size, 0));
        }
        FpMoveImmediate { rd, bits, .. } => s.v[rd as usize] = scalar(bits),
        FpMoveToGeneral {
            rd,
            rn,
            bits,
            upper,
        } => {
            let value = if upper {
                lane(v(s, rn), 64, 1)
            } else {
                lane(v(s, rn), u32::from(bits), 0)
            };
            if rd != 31 {
                s.x[rd as usize] = value;
            }
        }
        FpMoveFromGeneral {
            rd,
            rn,
            bits,
            upper,
        } => {
            let value = x(s, rn) & mask(u32::from(bits));
            s.v[rd as usize] = if upper {
                with_lane(v(s, rd) & u128::from(u64::MAX), 64, 1, value)
            } else {
                scalar(value)
            };
        }
        IntToFp {
            rd,
            rn,
            gp_bits,
            size,
            signed,
        } => {
            let size = fp_size(size);
            s.v[rd as usize] = scalar(from_integer(x(s, rn), u32::from(gp_bits), signed, 0, size));
        }
        IntToFpFixed {
            rd,
            rn,
            gp_bits,
            size,
            signed,
            fbits,
        } => {
            let size = fp_size(size);
            let value = from_integer(x(s, rn), u32::from(gp_bits), signed, fbits, size);
            s.v[rd as usize] = scalar(value);
        }
        SimdIntToFp {
            rd,
            rn,
            size,
            signed,
        } => {
            let size = fp_size(size);
            let value = from_integer(lane(v(s, rn), size, 0), size, signed, 0, size);
            s.v[rd as usize] = scalar(value);
        }
        FpToInt {
            rd,
            rn,
            gp_bits,
            size,
            signed,
            rounding,
        } => {
            let size = fp_size(size);
            let value = to_integer(
                lane(v(s, rn), size, 0),
                size,
                0,
                rounding,
                u32::from(gp_bits),
                signed,
            );
            if rd != 31 {
                s.x[rd as usize] = value;
            }
        }
        FpToFixed {
            rd,
            rn,
            gp_bits,
            size,
            signed,
            fbits,
        } => {
            let size = fp_size(size);
            let value = to_integer(
                lane(v(s, rn), size, 0),
                size,
                fbits,
                Rounding::Zero,
                u32::from(gp_bits),
                signed,
            );
            if rd != 31 {
                s.x[rd as usize] = value;
            }
        }
        VecFp {
            op,
            size,
            q,
            scalar: is_scalar,
            rd,
            rn,
            rm,
        }
        | VecFpElement {
            op,
            size,
            q,
            scalar: is_scalar,
            rd,
            rn,
            rm,
            ..
        } => {
            let size = fp_size(size);
            let count = if is_scalar {
                1
            } else if q {
                128 / size
            } else {
                64 / size
            };
            let (n, m, d) = (v(s, rn), v(s, rm), v(s, rd));
            let mut result = 0;
            for i in 0..count {
                let b = match *instruction {
                    VecFpElement { index, .. } => lane(m, size, u32::from(index)),
                    _ => lane(m, size, i),
                };
                let value = vec_fp(op, lane(n, size, i), b, lane(d, size, i), size);
                result = with_lane(result, size, i, value);
            }
            s.v[rd as usize] = result;
        }
        VecFpUnary {
            op,
            size,
            q,
            scalar: is_scalar,
            rd,
            rn,
            fbits,
        } => {
            let size = fp_size(size);
            let count = if is_scalar {
                1
            } else if q {
                128 / size
            } else {
                64 / size
            };
            let n = v(s, rn);
            let mut result = 0;
            for i in 0..count {
                result = with_lane(
                    result,
                    size,
                    i,
                    vec_fp_unary(op, lane(n, size, i), size, fbits),
                );
            }
            s.v[rd as usize] = result;
        }
        VecFpPairwise {
            op,
            size,
            q,
            scalar: is_scalar,
            rd,
            rn,
            rm,
        } => {
            let size = fp_size(size);
            let n = v(s, rn);
            if is_scalar {
                s.v[rd as usize] = scalar(pairwise(op, lane(n, size, 0), lane(n, size, 1), size));
                return;
            }
            let count = if q { 128 / size } else { 64 / size };
            let m = v(s, rm);
            let concat: Vec<u64> = (0..count)
                .map(|i| lane(n, size, i))
                .chain((0..count).map(|i| lane(m, size, i)))
                .collect();
            let mut result = 0;
            for i in 0..count {
                let (a, b) = (concat[2 * i as usize], concat[2 * i as usize + 1]);
                result = with_lane(result, size, i, pairwise(op, a, b, size));
            }
            s.v[rd as usize] = result;
        }
        VecFpAcross { op, rd, rn } => {
            let n = v(s, rn);
            let low = pairwise(op, lane(n, 32, 0), lane(n, 32, 1), 32);
            let high = pairwise(op, lane(n, 32, 2), lane(n, 32, 3), 32);
            s.v[rd as usize] = scalar(pairwise(op, low, high, 32));
        }
        VecFpConvert {
            from,
            to,
            upper,
            rd,
            rn,
        } => {
            let (from, to) = (precision(from), precision(to));
            let n = v(s, rn);
            s.v[rd as usize] = if to > from {
                // FCVTL/FCVTL2: two narrow lanes from a half.
                let offset = if upper { 2 } else { 0 };
                (0..2).fold(0, |r, i| {
                    with_lane(r, to, i, convert(lane(n, from, i + offset), from, to))
                })
            } else {
                // FCVTN/FCVTN2: into the lower half, or the upper keeping
                // the lower.
                let mut result = if upper {
                    v(s, rd) & u128::from(u64::MAX)
                } else {
                    0
                };
                let offset = if upper { 2 } else { 0 };
                for i in 0..2 {
                    result = with_lane(result, to, i + offset, convert(lane(n, from, i), from, to));
                }
                result
            };
        }
        ref other => panic!("no reference for {other:?}"),
    }
}

fn run_lifted(m: &mut Machine, entry: u64, state: &State) -> Result<State, Trap> {
    m.core.cpu.regs = state.x;
    m.core.cpu.vectors = state.v;
    m.core.cpu.nzcv = state.nzcv;
    m.core.cpu.fpcr = state.fpcr;
    m.core.cpu.fpsr = state.fpsr;
    m.core.cpu.pc = entry;
    let trap = loop {
        match fp_semantic_code::dispatch(&mut m.core) {
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
        fpcr: m.core.cpu.fpcr,
        fpsr: m.core.cpu.fpsr & FLAGS,
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
            "nzcv: expected {:#06b}, lifted {:#06b}",
            expected.nzcv, actual.nzcv
        ));
    }
    if expected.fpsr != actual.fpsr {
        parts.push(format!(
            "fpsr flags: expected {:#x}, lifted {:#x}",
            expected.fpsr, actual.fpsr
        ));
    }
    parts.join("; ")
}

/// A known echora-guest difference, accepted until fixed: FNMUL is
/// FPNeg(FPMul(n, m)), and with FPCR.AH = 0 FPNeg flips a NaN's sign too,
/// but the lifted FNMUL returns the propagated NaN unflipped. Accepts
/// exactly that: the same state except Vd's NaN with the other sign.
fn fnmul_nan_sign(op: &SimdOp, expected: &State, actual: &State) -> bool {
    let SimdOp::FpBinary {
        op: FpBinaryOp::NegMul,
        size,
        rd,
        ..
    } = *op
    else {
        return false;
    };
    let size = fp_size(size);
    let result = lane(expected.v[rd as usize], size, 0);
    let mut flipped = expected.clone();
    flipped.v[rd as usize] = u128::from(negate(result, size));
    is_nan(result, size) && flipped == *actual
}

/// The lane size an instruction's random vectors are built for.
fn lane_size(op: &SimdOp) -> u32 {
    use SimdOp::*;
    match *op {
        FpUnary { size, .. }
        | FpBinary { size, .. }
        | FpTernary { size, .. }
        | FpCompare { size, .. }
        | FpConditionalCompare { size, .. }
        | FpSelect { size, .. }
        | SimdIntToFp { size, .. }
        | FpToInt { size, .. }
        | FpToFixed { size, .. }
        | VecFp { size, .. }
        | VecFpElement { size, .. }
        | VecFpUnary { size, .. }
        | VecFpPairwise { size, .. } => fp_size(size),
        FpConvert { from, .. } => fp_size(from),
        VecFpConvert { from, .. } => precision(from),
        _ => 32,
    }
}

#[test]
fn lifted_floating_point_matches_the_reference() {
    let words = fp_semantic_code::WORDS;
    let code_size = (words.len() as u64 * STRIDE).next_multiple_of(0x1000);
    let mut image = Vec::new();
    for word in words {
        image.extend(word.to_le_bytes());
        image.extend(0xd400_00e1_u32.to_le_bytes());
        image.extend(0xd400_0101_u32.to_le_bytes());
    }
    let mut m = Machine::with_layout(fp_semantic_code::BASE, &image, code_size, 0x80_0000, 0x1000);
    fp_semantic_code::install();
    let mut random = Random(0x3c6e_f372_fe94_f82b);
    let mut failures = Vec::new();
    let mut runs = 0;
    let mut known = 0;
    let mut skipped = 0;
    for (index, &word) in words.iter().enumerate() {
        let entry = fp_semantic_code::BASE + STRIDE * index as u64;
        let Instruction::Simd(op) = decode(word, entry) else {
            panic!("{word:#010x} is not a SIMD instruction");
        };
        let size = lane_size(&op);
        for _ in 0..STATES_PER_WORD {
            let before = State {
                x: std::array::from_fn(|_| {
                    if random.next().is_multiple_of(2) {
                        random.float(64)
                    } else {
                        random.next()
                    }
                }),
                v: std::array::from_fn(|_| random.vector(size)),
                nzcv: (random.next() & 15) as u8,
                fpcr: match random.next() % 4 {
                    0 => 0,
                    1 => FPCR_FZ,
                    2 => FPCR_DN,
                    _ => FPCR_FZ | FPCR_DN,
                },
                fpsr: 0,
            };
            let mut expected = before.clone();
            CONTEXT.with(|cell| {
                cell.set(Context {
                    flush: before.fpcr & FPCR_FZ != 0,
                    default_nan: before.fpcr & FPCR_DN != 0,
                    ..Context::default()
                })
            });
            interpret(&op, &mut expected);
            let raised = context();
            expected.fpsr |= raised.flags;
            if raised.ambiguous {
                skipped += 1;
                continue;
            }
            runs += 1;
            let problem = match run_lifted(&mut m, entry, &before) {
                Ok(actual) if actual == expected => continue,
                Ok(actual) if fnmul_nan_sign(&op, &expected, &actual) => {
                    known += 1;
                    continue;
                }
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
        "{} instructions, {runs} runs, {} mismatching instructions, \
         {known} runs with the known FNMUL NaN sign difference, \
         {skipped} ambiguous flush-to-zero runs skipped",
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
