//! Scalar floating-point and Advanced SIMD execution helpers called by
//! generated code. Decoding lives in the recompiler crate.

use serde::Serialize;

use crate::Trap;
use crate::cpu::Cpu;
use crate::view::GuestView;

pub const FPCR_AH: u32 = 1 << 1;
pub const FPCR_FZ16: u32 = 1 << 19;
pub const FPCR_RMODE: u32 = 3 << 22;
pub const FPCR_FZ: u32 = 1 << 24;
pub const FPCR_DN: u32 = 1 << 25;
pub const FPCR_AHP: u32 = 1 << 26;

pub const FPSR_IOC: u32 = 1 << 0;
pub const FPSR_DZC: u32 = 1 << 1;
pub const FPSR_OFC: u32 = 1 << 2;
pub const FPSR_UFC: u32 = 1 << 3;
pub const FPSR_IXC: u32 = 1 << 4;
pub const FPSR_IDC: u32 = 1 << 7;
/// Cumulative saturation of the Advanced SIMD saturating instructions.
pub const FPSR_QC: u32 = 1 << 27;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum FpSize {
    S,
    D,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum FpUnaryOp {
    Mov,
    Abs,
    Neg,
    Sqrt,
    /// FRINT<r>: round to integral in floating-point format.
    Round(Rounding),
    /// FRINTX: as FRINTI, but signals inexact (not accumulated here).
    RoundExact,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum FpBinaryOp {
    Mul,
    Div,
    Add,
    Sub,
    Max,
    Min,
    MaxNum,
    MinNum,
    NegMul,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum FpTernaryOp {
    /// Rd = Ra + Rn*Rm
    MulAdd,
    /// Rd = Ra - Rn*Rm
    MulSub,
    /// Rd = -Ra - Rn*Rm
    NegMulAdd,
    /// Rd = -Ra + Rn*Rm
    NegMulSub,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Rounding {
    /// Ties to even.
    Nearest,
    /// Towards +infinity.
    Plus,
    /// Towards -infinity.
    Minus,
    /// Towards zero.
    Zero,
    /// Ties away from zero.
    Away,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum VectorLogicalOp {
    And,
    Bic,
    Orr,
    Orn,
    Eor,
    Bsl,
    Bit,
    Bif,
}

// ----------------------------------------------------------------------------
// Register access
// ----------------------------------------------------------------------------

fn bits_of(size: FpSize) -> u32 {
    match size {
        FpSize::S => 32,
        FpSize::D => 64,
    }
}

fn read_fp(cpu: &Cpu, index: u8, size: FpSize) -> u64 {
    let value = cpu.vectors[index as usize];
    match size {
        FpSize::S => value as u32 as u64,
        FpSize::D => value as u64,
    }
}

/// Scalar writes zero every bit above the written element.
fn write_fp(cpu: &mut Cpu, index: u8, size: FpSize, value: u64) {
    cpu.vectors[index as usize] = match size {
        FpSize::S => u128::from(value as u32),
        FpSize::D => u128::from(value),
    };
}

fn element_mask(bytes: u8) -> u128 {
    if bytes >= 16 {
        u128::MAX
    } else {
        (1u128 << (u32::from(bytes) * 8)) - 1
    }
}

fn read_element(cpu: &Cpu, index: u8, bytes: u8, lane: u8) -> u128 {
    (cpu.vectors[index as usize] >> (u32::from(lane) * u32::from(bytes) * 8)) & element_mask(bytes)
}

// ----------------------------------------------------------------------------
// FPCR / NaN handling
// ----------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Mode {
    fz: bool,
    dn: bool,
}

fn mode(cpu: &Cpu) -> Result<Mode, Trap> {
    let fpcr = cpu.fpcr;
    if fpcr & FPCR_RMODE != 0 {
        return Err(Trap::InvalidState {
            message: format!(
                "FPCR rounding mode {} is not implemented at {:#x}",
                (fpcr >> 22) & 3,
                cpu.pc
            ),
        });
    }
    if fpcr & (FPCR_AH | FPCR_FZ16) != 0 {
        return Err(Trap::InvalidState {
            message: format!("FPCR.AH/FZ16 behaviour is not implemented at {:#x}", cpu.pc),
        });
    }
    Ok(Mode {
        fz: fpcr & FPCR_FZ != 0,
        dn: fpcr & FPCR_DN != 0,
    })
}

/// Raw-bit view of one IEEE format.
#[derive(Clone, Copy)]
struct Format {
    bits: u32,
    exponent_bits: u32,
}

impl Format {
    const fn of(size: FpSize) -> Self {
        match size {
            FpSize::S => Self {
                bits: 32,
                exponent_bits: 8,
            },
            FpSize::D => Self {
                bits: 64,
                exponent_bits: 11,
            },
        }
    }
    fn fraction_bits(self) -> u32 {
        self.bits - 1 - self.exponent_bits
    }
    fn sign_bit(self) -> u64 {
        1 << (self.bits - 1)
    }
    fn exponent_mask(self) -> u64 {
        ((1u64 << self.exponent_bits) - 1) << self.fraction_bits()
    }
    fn fraction_mask(self) -> u64 {
        (1u64 << self.fraction_bits()) - 1
    }
    fn quiet_bit(self) -> u64 {
        1 << (self.fraction_bits() - 1)
    }
    fn is_nan(self, value: u64) -> bool {
        value & self.exponent_mask() == self.exponent_mask() && value & self.fraction_mask() != 0
    }
    fn is_signaling(self, value: u64) -> bool {
        self.is_nan(value) && value & self.quiet_bit() == 0
    }
    fn is_denormal(self, value: u64) -> bool {
        value & self.exponent_mask() == 0 && value & self.fraction_mask() != 0
    }
    fn default_nan(self) -> u64 {
        self.exponent_mask() | self.quiet_bit()
    }
}

struct Fp<'a> {
    cpu: &'a mut Cpu,
    mode: Mode,
    format: Format,
    size: FpSize,
}

impl<'a> Fp<'a> {
    fn new(cpu: &'a mut Cpu, size: FpSize) -> Result<Self, Trap> {
        let mode = mode(cpu)?;
        Ok(Self {
            cpu,
            mode,
            format: Format::of(size),
            size,
        })
    }

    fn raise(&mut self, flags: u32) {
        self.cpu.fpsr |= flags;
    }

    /// FPUnpack's flush of denormal inputs when FPCR.FZ is set.
    fn input(&mut self, value: u64) -> u64 {
        if self.mode.fz && self.format.is_denormal(value) {
            self.raise(FPSR_IDC);
            value & self.format.sign_bit()
        } else {
            value
        }
    }

    fn quiet(&self, value: u64) -> u64 {
        if self.mode.dn {
            self.format.default_nan()
        } else {
            value | self.format.quiet_bit()
        }
    }

    /// FPProcessNaNs over the operands in architectural order.
    fn process_nans(&mut self, operands: &[u64]) -> Option<u64> {
        if let Some(&nan) = operands.iter().find(|v| self.format.is_signaling(**v)) {
            self.raise(FPSR_IOC);
            return Some(self.quiet(nan));
        }
        operands
            .iter()
            .find(|v| self.format.is_nan(**v))
            .map(|&nan| self.quiet(nan))
    }

    fn to_f64(&self, value: u64) -> f64 {
        match self.size {
            FpSize::S => f64::from(f32::from_bits(value as u32)),
            FpSize::D => f64::from_bits(value),
        }
    }

    /// Final rounding of an exactly computed or natively rounded result:
    /// default NaN for invalid operations, overflow and FZ output flushing.
    fn output(&mut self, value: u64, finite_inputs: bool) -> u64 {
        if self.format.is_nan(value) {
            self.raise(FPSR_IOC);
            return self.format.default_nan();
        }
        let infinite = value & !self.format.sign_bit() == self.format.exponent_mask();
        if infinite && finite_inputs {
            self.raise(FPSR_OFC);
        }
        if self.mode.fz && self.format.is_denormal(value) {
            return value & self.format.sign_bit();
        }
        value
    }

    fn is_finite(&self, value: u64) -> bool {
        value & self.format.exponent_mask() != self.format.exponent_mask()
    }
}

macro_rules! native {
    ($size:expr, $a:expr, $b:expr, |$x:ident, $y:ident| $body:expr) => {
        match $size {
            FpSize::S => {
                let $x = f32::from_bits($a as u32);
                let $y = f32::from_bits($b as u32);
                u64::from(($body).to_bits())
            }
            FpSize::D => {
                let $x = f64::from_bits($a);
                let $y = f64::from_bits($b);
                ($body).to_bits()
            }
        }
    };
}

fn is_zero(format: Format, value: u64) -> bool {
    value & !format.sign_bit() == 0
}

fn is_infinity(format: Format, value: u64) -> bool {
    value & !format.sign_bit() == format.exponent_mask()
}

// ----------------------------------------------------------------------------
// Execution
// ----------------------------------------------------------------------------

pub fn fp_unary(cpu: &mut Cpu, op: FpUnaryOp, size: FpSize, rd: u8, rn: u8) -> Result<(), Trap> {
    let value = read_fp(cpu, rn, size);
    let format = Format::of(size);
    let result = match op {
        // FMOV, FABS and FNEG are bitwise and never trap or flush.
        FpUnaryOp::Mov => value,
        FpUnaryOp::Abs => value & !format.sign_bit(),
        FpUnaryOp::Neg => value ^ format.sign_bit(),
        FpUnaryOp::Sqrt => {
            let mut fp = Fp::new(cpu, size)?;
            let value = fp.input(value);
            if let Some(nan) = fp.process_nans(&[value]) {
                nan
            } else if value & format.sign_bit() != 0 && !is_zero(format, value) {
                fp.raise(FPSR_IOC);
                format.default_nan()
            } else {
                let raw = native!(size, value, 0, |x, _y| x.sqrt());
                fp.output(raw, false)
            }
        }
        FpUnaryOp::Round(_) | FpUnaryOp::RoundExact => {
            let rounding = if let FpUnaryOp::Round(r) = op {
                r
            } else {
                Rounding::Nearest
            };
            let mut fp = Fp::new(cpu, size)?;
            let value = fp.input(value);
            if let Some(nan) = fp.process_nans(&[value]) {
                nan
            } else {
                native!(size, value, 0, |x, _y| round(x, rounding))
            }
        }
    };
    write_fp(cpu, rd, size, result);
    Ok(())
}

trait Rint: Copy {
    fn round_with(self, rounding: Rounding) -> Self;
}
impl Rint for f32 {
    fn round_with(self, rounding: Rounding) -> Self {
        match rounding {
            Rounding::Nearest => self.round_ties_even(),
            Rounding::Plus => self.ceil(),
            Rounding::Minus => self.floor(),
            Rounding::Zero => self.trunc(),
            Rounding::Away => self.round(),
        }
    }
}
impl Rint for f64 {
    fn round_with(self, rounding: Rounding) -> Self {
        match rounding {
            Rounding::Nearest => self.round_ties_even(),
            Rounding::Plus => self.ceil(),
            Rounding::Minus => self.floor(),
            Rounding::Zero => self.trunc(),
            Rounding::Away => self.round(),
        }
    }
}
fn round<T: Rint>(value: T, rounding: Rounding) -> T {
    value.round_with(rounding)
}

pub fn fp_convert(cpu: &mut Cpu, rd: u8, rn: u8, from: FpSize, to: FpSize) -> Result<(), Trap> {
    let mode = mode(cpu)?;
    let source = Format::of(from);
    let target = Format::of(to);
    let mut value = read_fp(cpu, rn, from);
    if mode.fz && source.is_denormal(value) {
        cpu.fpsr |= FPSR_IDC;
        value &= source.sign_bit();
    }
    let result = if source.is_nan(value) {
        if source.is_signaling(value) {
            cpu.fpsr |= FPSR_IOC;
        }
        if mode.dn {
            target.default_nan()
        } else {
            // FPConvertNaN: keep the sign and the most significant payload bits.
            let sign = if value & source.sign_bit() != 0 {
                target.sign_bit()
            } else {
                0
            };
            let payload = value & source.fraction_mask();
            let shifted = if target.fraction_bits() > source.fraction_bits() {
                payload << (target.fraction_bits() - source.fraction_bits())
            } else {
                payload >> (source.fraction_bits() - target.fraction_bits())
            };
            sign | target.exponent_mask() | target.quiet_bit() | shifted
        }
    } else {
        let converted = match (from, to) {
            (FpSize::S, FpSize::D) => f64::from(f32::from_bits(value as u32)).to_bits(),
            (FpSize::D, FpSize::S) => u64::from((f64::from_bits(value) as f32).to_bits()),
            _ => value,
        };
        if is_infinity(target, converted) && !is_infinity(source, value) {
            cpu.fpsr |= FPSR_OFC;
        }
        if mode.fz && target.is_denormal(converted) {
            converted & target.sign_bit()
        } else {
            converted
        }
    };
    write_fp(cpu, rd, to, result);
    Ok(())
}

/// Fast paths: with FPCR in its default mode (round to nearest, no flush,
/// no default NaN) and no NaN or infinity in the result, the exact helpers
/// return the host's native result unchanged and raise no flag. The fast
/// paths below are inlined into translated code and fall back to the exact
/// helpers otherwise.
#[inline(always)]
fn default_mode(cpu: &Cpu) -> bool {
    cpu.fpcr == 0
}

/// Whether raw `value` of `size` is finite (not an infinity or NaN).
#[inline(always)]
fn finite(size: FpSize, value: u64) -> bool {
    let exponent = Format::of(size).exponent_mask();
    value & exponent != exponent
}

#[inline]
pub fn fp_binary(
    cpu: &mut Cpu,
    op: FpBinaryOp,
    size: FpSize,
    rd: u8,
    rn: u8,
    rm: u8,
) -> Result<(), Trap> {
    if default_mode(cpu) {
        let a = read_fp(cpu, rn, size);
        let b = read_fp(cpu, rm, size);
        let raw = match op {
            FpBinaryOp::Add => Some(native!(size, a, b, |x, y| x + y)),
            FpBinaryOp::Sub => Some(native!(size, a, b, |x, y| x - y)),
            FpBinaryOp::Mul => Some(native!(size, a, b, |x, y| x * y)),
            FpBinaryOp::Div => Some(native!(size, a, b, |x, y| x / y)),
            _ => None,
        };
        if let Some(raw) = raw.filter(|raw| finite(size, *raw)) {
            write_fp(cpu, rd, size, raw);
            return Ok(());
        }
    }
    fp_binary_exact(cpu, op, size, rd, rn, rm)
}

#[inline(never)]
fn fp_binary_exact(
    cpu: &mut Cpu,
    op: FpBinaryOp,
    size: FpSize,
    rd: u8,
    rn: u8,
    rm: u8,
) -> Result<(), Trap> {
    let a = read_fp(cpu, rn, size);
    let b = read_fp(cpu, rm, size);
    let mut fp = Fp::new(cpu, size)?;
    let result = binary_lane(&mut fp, op, a, b);
    write_fp(cpu, rd, size, result);
    Ok(())
}

/// One lane of a two-operand FP operation (FPProcessNaNs, then the op).
fn binary_lane(fp: &mut Fp<'_>, op: FpBinaryOp, a: u64, b: u64) -> u64 {
    let size = fp.size;
    let format = fp.format;
    let a = fp.input(a);
    let b = fp.input(b);
    let numeric = matches!(op, FpBinaryOp::MaxNum | FpBinaryOp::MinNum);
    let (a, b) = if numeric {
        // FPMaxNum/FPMinNum: a single quiet NaN is replaced by the other
        // operand's "missing" value before ordinary processing.
        let quiet_a = format.is_nan(a) && !format.is_signaling(a);
        let quiet_b = format.is_nan(b) && !format.is_signaling(b);
        let infinity =
            |negative: bool| format.exponent_mask() | if negative { format.sign_bit() } else { 0 };
        let max = op == FpBinaryOp::MaxNum;
        match (quiet_a, quiet_b) {
            (true, false) => (infinity(max), b),
            (false, true) => (a, infinity(max)),
            _ => (a, b),
        }
    } else {
        (a, b)
    };
    if let Some(nan) = fp.process_nans(&[a, b]) {
        nan
    } else {
        let finite = fp.is_finite(a) && fp.is_finite(b);
        match op {
            FpBinaryOp::Max | FpBinaryOp::Min | FpBinaryOp::MaxNum | FpBinaryOp::MinNum => {
                let max = matches!(op, FpBinaryOp::Max | FpBinaryOp::MaxNum);
                if is_zero(format, a) && is_zero(format, b) {
                    // +0 is larger than -0.
                    if max { a & b } else { a | b }
                } else {
                    let (x, y) = (fp.to_f64(a), fp.to_f64(b));
                    let pick_a = if max { x > y } else { x < y };
                    let chosen = if pick_a { a } else { b };
                    fp.output(chosen, false)
                }
            }
            FpBinaryOp::Div => {
                if is_zero(format, b) && fp.is_finite(a) && !is_zero(format, a) {
                    fp.raise(FPSR_DZC);
                }
                let raw = native!(size, a, b, |x, y| x / y);
                fp.output(raw, finite && !is_zero(format, b))
            }
            FpBinaryOp::NegMul => {
                let raw = native!(size, a, b, |x, y| x * y);
                let out = fp.output(raw, finite);
                if format.is_nan(out) {
                    out
                } else {
                    out ^ format.sign_bit()
                }
            }
            FpBinaryOp::Mul => {
                let raw = native!(size, a, b, |x, y| x * y);
                fp.output(raw, finite)
            }
            FpBinaryOp::Add => {
                let raw = native!(size, a, b, |x, y| x + y);
                fp.output(raw, finite)
            }
            FpBinaryOp::Sub => {
                let raw = native!(size, a, b, |x, y| x - y);
                fp.output(raw, finite)
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub fn fp_ternary(
    cpu: &mut Cpu,
    op: FpTernaryOp,
    size: FpSize,
    rd: u8,
    rn: u8,
    rm: u8,
    ra: u8,
) -> Result<(), Trap> {
    if default_mode(cpu) {
        let sign = Format::of(size).sign_bit();
        let (negate_addend, negate_product) = match op {
            FpTernaryOp::MulAdd => (false, false),
            FpTernaryOp::MulSub => (false, true),
            FpTernaryOp::NegMulAdd => (true, true),
            FpTernaryOp::NegMulSub => (true, false),
        };
        let n = read_fp(cpu, rn, size) ^ if negate_product { sign } else { 0 };
        let m = read_fp(cpu, rm, size);
        let a = read_fp(cpu, ra, size) ^ if negate_addend { sign } else { 0 };
        let raw = match size {
            FpSize::S => u64::from(
                f32::from_bits(n as u32)
                    .mul_add(f32::from_bits(m as u32), f32::from_bits(a as u32))
                    .to_bits(),
            ),
            FpSize::D => f64::from_bits(n)
                .mul_add(f64::from_bits(m), f64::from_bits(a))
                .to_bits(),
        };
        if finite(size, raw) {
            write_fp(cpu, rd, size, raw);
            return Ok(());
        }
    }
    fp_ternary_exact(cpu, op, size, rd, rn, rm, ra)
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn fp_ternary_exact(
    cpu: &mut Cpu,
    op: FpTernaryOp,
    size: FpSize,
    rd: u8,
    rn: u8,
    rm: u8,
    ra: u8,
) -> Result<(), Trap> {
    let n = read_fp(cpu, rn, size);
    let m = read_fp(cpu, rm, size);
    let a = read_fp(cpu, ra, size);
    let mut fp = Fp::new(cpu, size)?;
    let result = ternary_lane(&mut fp, op, n, m, a);
    write_fp(cpu, rd, size, result);
    Ok(())
}

/// One lane of a fused multiply-add family operation: Ra +/- Rn*Rm.
fn ternary_lane(fp: &mut Fp<'_>, op: FpTernaryOp, n: u64, m: u64, a: u64) -> u64 {
    let size = fp.size;
    let format = fp.format;
    let sign = format.sign_bit();
    // Negations are applied to operands before the fused operation, as in
    // the pseudocode (FMSUB negates op1; FNMADD negates addend and op1).
    let (negate_addend, negate_product) = match op {
        FpTernaryOp::MulAdd => (false, false),
        FpTernaryOp::MulSub => (false, true),
        FpTernaryOp::NegMulAdd => (true, true),
        FpTernaryOp::NegMulSub => (true, false),
    };
    let n = fp.input(n);
    let m = fp.input(m);
    let a = fp.input(a);
    let n = if negate_product { n ^ sign } else { n };
    let a = if negate_addend { a ^ sign } else { a };
    let invalid_product = (is_infinity(format, n) && is_zero(format, m))
        || (is_zero(format, n) && is_infinity(format, m));
    let quiet_addend = format.is_nan(a) && !format.is_signaling(a);
    if let Some(nan) = fp.process_nans(&[a, n, m]) {
        if quiet_addend && invalid_product {
            fp.raise(FPSR_IOC);
            format.default_nan()
        } else {
            nan
        }
    } else {
        let finite = fp.is_finite(a) && fp.is_finite(n) && fp.is_finite(m);
        let raw = match size {
            FpSize::S => u64::from(
                f32::from_bits(n as u32)
                    .mul_add(f32::from_bits(m as u32), f32::from_bits(a as u32))
                    .to_bits(),
            ),
            FpSize::D => f64::from_bits(n)
                .mul_add(f64::from_bits(m), f64::from_bits(a))
                .to_bits(),
        };
        fp.output(raw, finite)
    }
}

fn compare_flags(fp: &mut Fp<'_>, a: u64, b: u64, signaling: bool) -> u8 {
    let format = fp.format;
    if format.is_nan(a) || format.is_nan(b) {
        if signaling || format.is_signaling(a) || format.is_signaling(b) {
            fp.raise(FPSR_IOC);
        }
        return 0b0011;
    }
    let (x, y) = (fp.to_f64(a), fp.to_f64(b));
    if x == y {
        0b0110
    } else if x < y {
        0b1000
    } else {
        0b0010
    }
}

#[inline]
pub fn fp_compare(
    cpu: &mut Cpu,
    size: FpSize,
    rn: u8,
    rm: Option<u8>,
    signaling: bool,
) -> Result<(), Trap> {
    if default_mode(cpu) {
        let a = read_fp(cpu, rn, size);
        let b = rm.map_or(0, |rm| read_fp(cpu, rm, size));
        let format = Format::of(size);
        if !format.is_nan(a) && !format.is_nan(b) {
            let (x, y) = match size {
                FpSize::S => (
                    f64::from(f32::from_bits(a as u32)),
                    f64::from(f32::from_bits(b as u32)),
                ),
                FpSize::D => (f64::from_bits(a), f64::from_bits(b)),
            };
            cpu.nzcv = if x == y {
                0b0110
            } else if x < y {
                0b1000
            } else {
                0b0010
            };
            return Ok(());
        }
    }
    fp_compare_exact(cpu, size, rn, rm, signaling)
}

#[inline(never)]
fn fp_compare_exact(
    cpu: &mut Cpu,
    size: FpSize,
    rn: u8,
    rm: Option<u8>,
    signaling: bool,
) -> Result<(), Trap> {
    let a = read_fp(cpu, rn, size);
    let b = rm.map_or(0, |rm| read_fp(cpu, rm, size));
    let mut fp = Fp::new(cpu, size)?;
    let a = fp.input(a);
    let b = fp.input(b);
    let flags = compare_flags(&mut fp, a, b, signaling);
    cpu.nzcv = flags;
    Ok(())
}

pub fn fp_conditional_compare(
    cpu: &mut Cpu,
    size: FpSize,
    rn: u8,
    rm: u8,
    condition: u8,
    nzcv: u8,
    signaling: bool,
) -> Result<(), Trap> {
    if !cpu.condition(condition) {
        cpu.nzcv = nzcv & 15;
        return Ok(());
    }
    fp_compare(cpu, size, rn, Some(rm), signaling)
}

#[inline]
pub fn fp_select(cpu: &mut Cpu, size: FpSize, rd: u8, rn: u8, rm: u8, condition: u8) {
    let source = if cpu.condition(condition) { rn } else { rm };
    let value = read_fp(cpu, source, size);
    write_fp(cpu, rd, size, value);
}

#[inline]
pub fn fp_move_immediate(cpu: &mut Cpu, size: FpSize, rd: u8, bits: u64) {
    write_fp(cpu, rd, size, bits);
}

pub fn fp_move_to_general(cpu: &mut Cpu, rd: u8, rn: u8, bits: u8, upper: bool) {
    let vector = cpu.vectors[rn as usize];
    let value = if upper {
        (vector >> 64) as u64
    } else if bits == 32 {
        vector as u32 as u64
    } else {
        vector as u64
    };
    cpu.write_reg(rd, bits, false, value);
}

#[inline]
pub fn fp_move_from_general(cpu: &mut Cpu, rd: u8, rn: u8, bits: u8, upper: bool) {
    let value = cpu.read_reg(rn, bits, false);
    let vector = &mut cpu.vectors[rd as usize];
    if upper {
        *vector = (*vector & u128::from(u64::MAX)) | (u128::from(value) << 64);
    } else {
        *vector = u128::from(value);
    }
}

fn int_to_fp_bits(value: u64, int_bits: u32, signed: bool, size: FpSize) -> u64 {
    match (size, signed, int_bits) {
        (FpSize::S, true, 32) => u64::from((value as u32 as i32 as f32).to_bits()),
        (FpSize::S, true, _) => u64::from((value as i64 as f32).to_bits()),
        (FpSize::S, false, 32) => u64::from((value as u32 as f32).to_bits()),
        (FpSize::S, false, _) => u64::from((value as f32).to_bits()),
        (FpSize::D, true, 32) => (value as u32 as i32 as f64).to_bits(),
        (FpSize::D, true, _) => (value as i64 as f64).to_bits(),
        (FpSize::D, false, 32) => (value as u32 as f64).to_bits(),
        (FpSize::D, false, _) => (value as f64).to_bits(),
    }
}

pub fn int_to_fp(
    cpu: &mut Cpu,
    rd: u8,
    rn: u8,
    gp_bits: u8,
    size: FpSize,
    signed: bool,
) -> Result<(), Trap> {
    let mode = mode(cpu)?;
    let value = cpu.read_reg(rn, gp_bits, false);
    let bits = int_to_fp_bits(value, u32::from(gp_bits), signed, size);
    // Integer conversions cannot produce denormals; FZ needs no flush here.
    let _ = mode;
    write_fp(cpu, rd, size, bits);
    Ok(())
}

pub fn simd_int_to_fp(
    cpu: &mut Cpu,
    rd: u8,
    rn: u8,
    size: FpSize,
    signed: bool,
) -> Result<(), Trap> {
    mode(cpu)?;
    let value = read_fp(cpu, rn, size);
    let bits = int_to_fp_bits(value, bits_of(size), signed, size);
    write_fp(cpu, rd, size, bits);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn fp_to_int(
    cpu: &mut Cpu,
    rd: u8,
    rn: u8,
    gp_bits: u8,
    size: FpSize,
    signed: bool,
    rounding: Rounding,
) -> Result<(), Trap> {
    let raw = read_fp(cpu, rn, size);
    let mut fp = Fp::new(cpu, size)?;
    let raw = fp.input(raw);
    let format = fp.format;
    let value = if format.is_nan(raw) {
        fp.raise(FPSR_IOC);
        0
    } else {
        let x = round(fp.to_f64(raw), rounding);
        // f32 inputs are exact in f64, so the rounding above is exact too.
        let (min, max) = match (signed, gp_bits) {
            (true, 32) => (f64::from(i32::MIN), f64::from(i32::MAX)),
            (true, _) => (i64::MIN as f64, 9_223_372_036_854_775_808.0),
            (false, 32) => (0.0, f64::from(u32::MAX)),
            (false, _) => (0.0, 18_446_744_073_709_551_616.0),
        };
        let out_of_range = if gp_bits == 64 {
            x < min || x >= max
        } else {
            x < min || x > max
        };
        if out_of_range {
            fp.raise(FPSR_IOC);
        }
        match (signed, gp_bits) {
            (true, 32) => x as i32 as u32 as u64,
            (true, _) => x as i64 as u64,
            (false, 32) => x as u32 as u64,
            (false, _) => x as u64,
        }
    };
    cpu.write_reg(rd, gp_bits, false, value);
    Ok(())
}

#[inline]
pub fn move_immediate(cpu: &mut Cpu, rd: u8, value: u128) {
    cpu.vectors[rd as usize] = value;
}

pub fn modify_immediate(cpu: &mut Cpu, rd: u8, value: u128, clear: bool, q: bool) {
    let current = cpu.vectors[rd as usize];
    let result = if clear {
        current & !value
    } else {
        current | value
    };
    cpu.vectors[rd as usize] = if q {
        result
    } else {
        result & u128::from(u64::MAX)
    };
}

pub fn vector_logical(cpu: &mut Cpu, op: VectorLogicalOp, rd: u8, rn: u8, rm: u8, q: bool) {
    let n = cpu.vectors[rn as usize];
    let m = cpu.vectors[rm as usize];
    let d = cpu.vectors[rd as usize];
    let result = match op {
        VectorLogicalOp::And => n & m,
        VectorLogicalOp::Bic => n & !m,
        VectorLogicalOp::Orr => n | m,
        VectorLogicalOp::Orn => n | !m,
        VectorLogicalOp::Eor => n ^ m,
        VectorLogicalOp::Bsl => (d & n) | (!d & m),
        VectorLogicalOp::Bit => (n & m) | (d & !m),
        VectorLogicalOp::Bif => (d & m) | (n & !m),
    };
    cpu.vectors[rd as usize] = if q {
        result
    } else {
        result & u128::from(u64::MAX)
    };
}

pub fn insert_element(cpu: &mut Cpu, rd: u8, rn: u8, bytes: u8, dst: u8, src: u8) {
    let value = read_element(cpu, rn, bytes, src);
    let shift = u32::from(dst) * u32::from(bytes) * 8;
    let mask = element_mask(bytes) << shift;
    let vector = &mut cpu.vectors[rd as usize];
    *vector = (*vector & !mask) | (value << shift);
}

pub fn insert_general(cpu: &mut Cpu, rd: u8, rn: u8, bytes: u8, index: u8) {
    let value = u128::from(cpu.read_reg(rn, 64, false)) & element_mask(bytes);
    let shift = u32::from(index) * u32::from(bytes) * 8;
    let mask = element_mask(bytes) << shift;
    let vector = &mut cpu.vectors[rd as usize];
    *vector = (*vector & !mask) | (value << shift);
}

pub fn move_to_general(
    cpu: &mut Cpu,
    rd: u8,
    rn: u8,
    bytes: u8,
    index: u8,
    signed: bool,
    gp_bits: u8,
) {
    let value = read_element(cpu, rn, bytes, index) as u64;
    let value = if signed {
        crate::cpu::sign_extend(value, bytes * 8)
    } else {
        value
    };
    cpu.write_reg(rd, gp_bits, false, value);
}

fn replicate(value: u128, bytes: u8, q: bool) -> u128 {
    let mut result = 0u128;
    let lanes = if q { 16 } else { 8 } / u32::from(bytes);
    for lane in 0..lanes {
        result |= value << (lane * u32::from(bytes) * 8);
    }
    result
}

pub fn dup_element(cpu: &mut Cpu, rd: u8, rn: u8, bytes: u8, index: u8, q: bool, scalar: bool) {
    let value = read_element(cpu, rn, bytes, index);
    cpu.vectors[rd as usize] = if scalar {
        value
    } else {
        replicate(value, bytes, q)
    };
}

pub fn dup_general(cpu: &mut Cpu, rd: u8, rn: u8, bytes: u8, q: bool) {
    let value = u128::from(cpu.read_reg(rn, 64, false)) & element_mask(bytes);
    cpu.vectors[rd as usize] = replicate(value, bytes, q);
}

#[inline]
pub fn load(
    cpu: &mut Cpu,
    memory: &GuestView,
    address: u64,
    rt: u8,
    bytes: u8,
) -> Result<(), Trap> {
    let mut buffer = [0u8; 16];
    memory.read(address, &mut buffer[..usize::from(bytes)])?;
    cpu.vectors[rt as usize] = u128::from_le_bytes(buffer);
    Ok(())
}

#[inline]
pub fn store(
    cpu: &Cpu,
    memory: &mut GuestView,
    address: u64,
    rt: u8,
    bytes: u8,
) -> Result<(), Trap> {
    let bytes_le = cpu.vectors[rt as usize].to_le_bytes();
    memory.write(address, &bytes_le[..usize::from(bytes)])
}

/// Pair transfers check both halves before changing any register or byte.
#[inline]
pub fn load_pair(
    cpu: &mut Cpu,
    memory: &GuestView,
    address: u64,
    rt: u8,
    rt2: u8,
    bytes: u8,
) -> Result<(), Trap> {
    let len = usize::from(bytes);
    let mut buffer = [0u8; 32];
    memory.read(address, &mut buffer[..len * 2])?;
    let mut first = [0u8; 16];
    let mut second = [0u8; 16];
    first[..len].copy_from_slice(&buffer[..len]);
    second[..len].copy_from_slice(&buffer[len..len * 2]);
    cpu.vectors[rt as usize] = u128::from_le_bytes(first);
    cpu.vectors[rt2 as usize] = u128::from_le_bytes(second);
    Ok(())
}

#[inline]
pub fn store_pair(
    cpu: &Cpu,
    memory: &mut GuestView,
    address: u64,
    rt: u8,
    rt2: u8,
    bytes: u8,
) -> Result<(), Trap> {
    let len = usize::from(bytes);
    let mut buffer = [0u8; 32];
    buffer[..len].copy_from_slice(&cpu.vectors[rt as usize].to_le_bytes()[..len]);
    buffer[len..len * 2].copy_from_slice(&cpu.vectors[rt2 as usize].to_le_bytes()[..len]);
    memory.write(address, &buffer[..len * 2])
}

// ----------------------------------------------------------------------------
// Advanced SIMD vector operations
// ----------------------------------------------------------------------------

fn lane_mask(bits: u32) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

fn get_lane(value: u128, bits: u32, index: u32) -> u64 {
    ((value >> (index * bits)) as u64) & lane_mask(bits)
}

fn set_lane(value: &mut u128, bits: u32, index: u32, lane: u64) {
    let shift = index * bits;
    let mask = u128::from(lane_mask(bits)) << shift;
    *value = (*value & !mask) | ((u128::from(lane & lane_mask(bits))) << shift);
}

/// Number of lanes: a scalar operation uses only lane 0.
fn lane_count(bits: u32, q: bool, scalar: bool) -> u32 {
    if scalar {
        1
    } else if q {
        128 / bits
    } else {
        64 / bits
    }
}

/// Clear the bits above the written part: 64-bit vectors and scalars zero the
/// rest of the register.
fn finish_vector(value: u128, bits: u32, q: bool, scalar: bool) -> u128 {
    if scalar {
        value & u128::from(lane_mask(bits))
    } else if q {
        value
    } else {
        value & u128::from(u64::MAX)
    }
}

fn sign_extend_lane(value: u64, bits: u32) -> i64 {
    if bits >= 64 {
        value as i64
    } else {
        ((value << (64 - bits)) as i64) >> (64 - bits)
    }
}

fn size_bits(size: FpSize) -> u32 {
    match size {
        FpSize::S => 32,
        FpSize::D => 64,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum VecFpOp {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
    MaxNum,
    MinNum,
    /// |a - b|
    AbsDiff,
    MulExtended,
    CmEq,
    CmGe,
    CmGt,
    AcGe,
    AcGt,
    /// d + n*m (fused)
    MulAdd,
    /// d - n*m (fused)
    MulSub,
    /// FRECPS: 2 - n*m (fused).
    RecipStep,
    /// FRSQRTS: (3 - n*m) / 2 (fused).
    RSqrtStep,
}

fn fp_compare_lane(fp: &mut Fp<'_>, op: VecFpOp, a: u64, b: u64) -> bool {
    let format = fp.format;
    let (a, b) = if matches!(op, VecFpOp::AcGe | VecFpOp::AcGt) {
        (a & !format.sign_bit(), b & !format.sign_bit())
    } else {
        (a, b)
    };
    if format.is_nan(a) || format.is_nan(b) {
        // FCMEQ is a quiet comparison; the ordered comparisons signal.
        if op != VecFpOp::CmEq || format.is_signaling(a) || format.is_signaling(b) {
            fp.raise(FPSR_IOC);
        }
        return false;
    }
    let (x, y) = (fp.to_f64(a), fp.to_f64(b));
    match op {
        VecFpOp::CmEq => x == y,
        VecFpOp::CmGe | VecFpOp::AcGe => x >= y,
        _ => x > y,
    }
}

fn vec_fp_lane(fp: &mut Fp<'_>, op: VecFpOp, a: u64, b: u64, d: u64) -> u64 {
    let format = fp.format;
    let all_ones = lane_mask(format.bits);
    match op {
        VecFpOp::Add => binary_lane(fp, FpBinaryOp::Add, a, b),
        VecFpOp::Sub => binary_lane(fp, FpBinaryOp::Sub, a, b),
        VecFpOp::Mul => binary_lane(fp, FpBinaryOp::Mul, a, b),
        VecFpOp::Div => binary_lane(fp, FpBinaryOp::Div, a, b),
        VecFpOp::Max => binary_lane(fp, FpBinaryOp::Max, a, b),
        VecFpOp::Min => binary_lane(fp, FpBinaryOp::Min, a, b),
        VecFpOp::MaxNum => binary_lane(fp, FpBinaryOp::MaxNum, a, b),
        VecFpOp::MinNum => binary_lane(fp, FpBinaryOp::MinNum, a, b),
        VecFpOp::AbsDiff => binary_lane(fp, FpBinaryOp::Sub, a, b) & !format.sign_bit(),
        VecFpOp::MulExtended => {
            // FMULX: infinity times zero is +/-2.0 instead of the default NaN.
            let ia = fp.input(a);
            let ib = fp.input(b);
            let special = (is_infinity(format, ia) && is_zero(format, ib))
                || (is_zero(format, ia) && is_infinity(format, ib));
            if special && !format.is_nan(ia) && !format.is_nan(ib) {
                let sign = (ia ^ ib) & format.sign_bit();
                let two = match fp.size {
                    FpSize::S => u64::from(2.0f32.to_bits()),
                    FpSize::D => 2.0f64.to_bits(),
                };
                two | sign
            } else {
                binary_lane(fp, FpBinaryOp::Mul, a, b)
            }
        }
        VecFpOp::CmEq | VecFpOp::CmGe | VecFpOp::CmGt | VecFpOp::AcGe | VecFpOp::AcGt => {
            let ia = fp.input(a);
            let ib = fp.input(b);
            if fp_compare_lane(fp, op, ia, ib) {
                all_ones
            } else {
                0
            }
        }
        VecFpOp::MulAdd => ternary_lane(fp, FpTernaryOp::MulAdd, a, b, d),
        VecFpOp::MulSub => ternary_lane(fp, FpTernaryOp::MulSub, a, b, d),
        VecFpOp::RecipStep | VecFpOp::RSqrtStep => step_lane(fp, op == VecFpOp::RSqrtStep, a, b),
    }
}

/// FPRecipStepFused / FPRSqrtStepFused. Only the two operands take part in
/// NaN processing; infinity times zero yields exactly 2.0 or 1.5. The square
/// root step halves the fused result, which is exact outside the subnormal
/// range.
fn step_lane(fp: &mut Fp<'_>, rsqrt: bool, a: u64, b: u64) -> u64 {
    let size = fp.size;
    let format = fp.format;
    let a = fp.input(a);
    let b = fp.input(b);
    if let Some(nan) = fp.process_nans(&[a, b]) {
        return nan;
    }
    let (special, constant) = match (rsqrt, size) {
        (false, FpSize::S) => (u64::from(2.0f32.to_bits()), u64::from(2.0f32.to_bits())),
        (false, FpSize::D) => (2.0f64.to_bits(), 2.0f64.to_bits()),
        (true, FpSize::S) => (u64::from(1.5f32.to_bits()), u64::from(3.0f32.to_bits())),
        (true, FpSize::D) => (1.5f64.to_bits(), 3.0f64.to_bits()),
    };
    if (is_infinity(format, a) && is_zero(format, b))
        || (is_zero(format, a) && is_infinity(format, b))
    {
        return special;
    }
    let finite = fp.is_finite(a) && fp.is_finite(b);
    let raw = match size {
        FpSize::S => {
            let fused = (-f32::from_bits(a as u32))
                .mul_add(f32::from_bits(b as u32), f32::from_bits(constant as u32));
            u64::from(if rsqrt { fused * 0.5 } else { fused }.to_bits())
        }
        FpSize::D => {
            let fused = (-f64::from_bits(a)).mul_add(f64::from_bits(b), f64::from_bits(constant));
            if rsqrt { fused * 0.5 } else { fused }.to_bits()
        }
    };
    fp.output(raw, finite)
}

/// RecipEstimate: 9-bit reciprocal of a 9-bit input in [256, 512).
const fn recip_estimate(a: u64) -> u64 {
    let a = a * 2 + 1;
    let b = (1 << 19) / a;
    b.div_ceil(2)
}

/// RecipSqrtEstimate: 9-bit reciprocal square root of an input in [128, 512).
const fn rsqrt_estimate(a: u64) -> u64 {
    let a = if a < 256 {
        a * 2 + 1
    } else {
        ((a >> 1) << 1) * 2 + 2
    };
    let mut b = 512u64;
    while a * (b + 1) * (b + 1) < (1 << 28) {
        b += 1;
    }
    b.div_ceil(2)
}

const RSQRT_TABLE: [u16; 384] = {
    let mut table = [0u16; 384];
    let mut index = 0;
    while index < 384 {
        table[index] = rsqrt_estimate(index as u64 + 128) as u16;
        index += 1;
    }
    table
};

/// The 52-bit fraction and biased exponent FPRecipEstimate and
/// FPRSqrtEstimate work from (fraction left-aligned to bit 51).
fn estimate_parts(format: Format, value: u64) -> (i64, u64) {
    let exponent = ((value & format.exponent_mask()) >> format.fraction_bits()) as i64;
    let fraction = (value & format.fraction_mask()) << (52 - format.fraction_bits());
    (exponent, fraction)
}

const FRACTION52: u64 = (1 << 52) - 1;

/// FPRecipEstimate (FRECPE) with FPCR.AH clear and round-to-nearest.
fn recip_estimate_lane(fp: &mut Fp<'_>, value: u64) -> u64 {
    let format = fp.format;
    let value = fp.input(value);
    let sign = value & format.sign_bit();
    let bias = (1i64 << (format.exponent_bits - 1)) - 1;
    if let Some(nan) = fp.process_nans(&[value]) {
        return nan;
    }
    if is_infinity(format, value) {
        return sign;
    }
    if is_zero(format, value) {
        fp.raise(FPSR_DZC);
        return sign | format.exponent_mask();
    }
    let magnitude = fp.to_f64(value).abs();
    // |x| < 2^-(bias+1) overflows; round-to-nearest gives infinity.
    if magnitude < f64::from_bits(((1023 - bias - 1) as u64) << 52) {
        fp.raise(FPSR_OFC | FPSR_IXC);
        return sign | format.exponent_mask();
    }
    if fp.mode.fz && magnitude >= f64::from_bits(((1023 + bias - 1) as u64) << 52) {
        fp.raise(FPSR_UFC);
        return sign;
    }
    let (mut exponent, mut fraction) = estimate_parts(format, value);
    if exponent == 0 {
        if fraction & (1 << 51) == 0 {
            exponent = -1;
            fraction = (fraction << 2) & FRACTION52;
        } else {
            fraction = (fraction << 1) & FRACTION52;
        }
    }
    let scaled = 256 | (fraction >> 44);
    let mut result_exponent = 2 * bias - 1 - exponent;
    let estimate = recip_estimate(scaled);
    let mut fraction = (estimate & 0xff) << 44;
    if result_exponent == 0 {
        fraction = (1 << 51) | (fraction >> 1);
    } else if result_exponent == -1 {
        fraction = (1 << 50) | (fraction >> 2);
        result_exponent = 0;
    }
    sign | ((result_exponent as u64) << format.fraction_bits())
        | (fraction >> (52 - format.fraction_bits()))
}

/// FPRSqrtEstimate (FRSQRTE) with FPCR.AH clear.
fn rsqrt_estimate_lane(fp: &mut Fp<'_>, value: u64) -> u64 {
    let format = fp.format;
    let value = fp.input(value);
    let sign = value & format.sign_bit();
    let bias = (1i64 << (format.exponent_bits - 1)) - 1;
    if let Some(nan) = fp.process_nans(&[value]) {
        return nan;
    }
    if is_zero(format, value) {
        fp.raise(FPSR_DZC);
        return sign | format.exponent_mask();
    }
    if sign != 0 {
        fp.raise(FPSR_IOC);
        return format.default_nan();
    }
    if is_infinity(format, value) {
        return 0;
    }
    let (mut exponent, mut fraction) = estimate_parts(format, value);
    if exponent == 0 {
        while fraction & (1 << 51) == 0 {
            fraction <<= 1;
            exponent -= 1;
        }
        fraction = (fraction << 1) & FRACTION52;
    }
    let scaled = if exponent & 1 == 0 {
        256 | (fraction >> 44)
    } else {
        128 | (fraction >> 45)
    };
    let result_exponent = (3 * bias - 1 - exponent) / 2;
    let estimate = u64::from(RSQRT_TABLE[(scaled - 128) as usize]);
    ((result_exponent as u64) << format.fraction_bits())
        | ((estimate & 0xff) << (format.fraction_bits() - 8))
}

/// Lanewise FP operation (vector, or scalar with `scalar`).
#[allow(clippy::too_many_arguments)]
pub fn vec_fp(
    cpu: &mut Cpu,
    op: VecFpOp,
    size: FpSize,
    q: bool,
    scalar: bool,
    rd: u8,
    rn: u8,
    rm: u8,
) -> Result<(), Trap> {
    let (n, m, d) = (
        cpu.vectors[rn as usize],
        cpu.vectors[rm as usize],
        cpu.vectors[rd as usize],
    );
    let bits = size_bits(size);
    let mut fp = Fp::new(cpu, size)?;
    let mut result = 0u128;
    for lane in 0..lane_count(bits, q, scalar) {
        let value = vec_fp_lane(
            &mut fp,
            op,
            get_lane(n, bits, lane),
            get_lane(m, bits, lane),
            get_lane(d, bits, lane),
        );
        set_lane(&mut result, bits, lane, value);
    }
    cpu.vectors[rd as usize] = finish_vector(result, bits, q, scalar);
    Ok(())
}

/// Pairwise FP operation. Vector: result lane i combines lanes 2i and 2i+1
/// of the concatenation Vm:Vn (Vn supplies the low half). Scalar: the two
/// lanes of Vn give one element.
#[allow(clippy::too_many_arguments)]
pub fn vec_fp_pairwise(
    cpu: &mut Cpu,
    op: VecFpOp,
    size: FpSize,
    q: bool,
    scalar: bool,
    rd: u8,
    rn: u8,
    rm: u8,
) -> Result<(), Trap> {
    let (n, m) = (cpu.vectors[rn as usize], cpu.vectors[rm as usize]);
    let bits = size_bits(size);
    let mut fp = Fp::new(cpu, size)?;
    let mut result = 0u128;
    if scalar {
        let value = vec_fp_lane(&mut fp, op, get_lane(n, bits, 0), get_lane(n, bits, 1), 0);
        set_lane(&mut result, bits, 0, value);
    } else {
        let lanes = lane_count(bits, q, false);
        let half = lanes / 2;
        for lane in 0..lanes {
            let (source, pair) = if lane < half {
                (n, lane)
            } else {
                (m, lane - half)
            };
            let value = vec_fp_lane(
                &mut fp,
                op,
                get_lane(source, bits, pair * 2),
                get_lane(source, bits, pair * 2 + 1),
                0,
            );
            set_lane(&mut result, bits, lane, value);
        }
    }
    cpu.vectors[rd as usize] = finish_vector(result, bits, q, scalar);
    Ok(())
}

/// FP operation by element: every lane of Vn with lane `index` of Vm.
#[allow(clippy::too_many_arguments)]
pub fn vec_fp_element(
    cpu: &mut Cpu,
    op: VecFpOp,
    size: FpSize,
    q: bool,
    scalar: bool,
    rd: u8,
    rn: u8,
    rm: u8,
    index: u8,
) -> Result<(), Trap> {
    let bits = size_bits(size);
    let element = get_lane(cpu.vectors[rm as usize], bits, u32::from(index));
    let (n, d) = (cpu.vectors[rn as usize], cpu.vectors[rd as usize]);
    let mut fp = Fp::new(cpu, size)?;
    let mut result = 0u128;
    for lane in 0..lane_count(bits, q, scalar) {
        let value = vec_fp_lane(
            &mut fp,
            op,
            get_lane(n, bits, lane),
            element,
            get_lane(d, bits, lane),
        );
        set_lane(&mut result, bits, lane, value);
    }
    cpu.vectors[rd as usize] = finish_vector(result, bits, q, scalar);
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum VecIntOp {
    Add,
    Sub,
    Mul,
    CmEq,
    CmTst,
    CmGt,
    CmGe,
    CmHi,
    CmHs,
    Sshl,
    Ushl,
    Smax,
    Smin,
    Umax,
    Umin,
    /// Unsigned/signed absolute difference.
    Uabd,
    Sabd,
    /// Saturating add/subtract (FPSR.QC on saturation).
    Sqadd,
    Uqadd,
    Sqsub,
    Uqsub,
    /// Halving add/subtract, and rounding halving add.
    Shadd,
    Uhadd,
    Shsub,
    Uhsub,
    Srhadd,
    Urhadd,
    /// Saturating (rounding) doubling multiply returning the high half.
    Sqdmulh,
    Sqrdmulh,
    /// d + n*m and d - n*m.
    Mla,
    Mls,
    /// Register shifts by the signed low byte of Vm: rounding, saturating,
    /// and saturating rounding.
    Srshl,
    Urshl,
    Sqshl,
    Uqshl,
    Sqrshl,
    Uqrshl,
}

/// SSHL-family shift of one lane by a signed amount (negative shifts right).
fn shift_register_lane(
    a: u64,
    bits: u32,
    shift: i64,
    signed: bool,
    round: bool,
    saturate: bool,
    saturated: &mut bool,
) -> u64 {
    let mask = lane_mask(bits);
    let value = if signed {
        i128::from(sign_extend_lane(a, bits))
    } else {
        i128::from(a)
    };
    let shifted = if shift >= 0 {
        if value == 0 {
            0
        } else if shift >= i64::from(bits) {
            // Every significant bit leaves the lane.
            if saturate {
                if value < 0 {
                    i128::MIN / 2
                } else {
                    i128::MAX / 2
                }
            } else {
                0
            }
        } else {
            value << shift
        }
    } else {
        let right = (-shift).min(i64::from(bits) + 1);
        if round {
            rounding_shift(value, right)
        } else {
            value >> right
        }
    };
    match (saturate, signed) {
        (true, true) => saturate_signed(shifted, bits, saturated),
        (true, false) => saturate_unsigned(shifted, bits, saturated),
        _ => shifted as u64 & mask,
    }
}

fn saturate_signed(value: i128, bits: u32, saturated: &mut bool) -> u64 {
    let max = (1i128 << (bits - 1)) - 1;
    let min = -(1i128 << (bits - 1));
    let clamped = value.clamp(min, max);
    *saturated |= clamped != value;
    clamped as u64 & lane_mask(bits)
}

fn saturate_unsigned(value: i128, bits: u32, saturated: &mut bool) -> u64 {
    let clamped = value.clamp(0, i128::from(lane_mask(bits)));
    *saturated |= clamped != value;
    clamped as u64
}

/// One lane of a three-same integer operation; `d` is the destination lane
/// for the accumulating forms.
fn int_lane(op: VecIntOp, bits: u32, a: u64, b: u64, d: u64, saturated: &mut bool) -> u64 {
    let mask = lane_mask(bits);
    let (sa, sb) = (sign_extend_lane(a, bits), sign_extend_lane(b, bits));
    let (wa, wb) = (i128::from(sa), i128::from(sb));
    let (ua, ub) = (i128::from(a), i128::from(b));
    let flag = |condition: bool| if condition { mask } else { 0 };
    let value = match op {
        VecIntOp::Add => a.wrapping_add(b),
        VecIntOp::Sub => a.wrapping_sub(b),
        VecIntOp::Mul => a.wrapping_mul(b),
        VecIntOp::CmEq => flag(a == b),
        VecIntOp::CmTst => flag(a & b != 0),
        VecIntOp::CmGt => flag(sa > sb),
        VecIntOp::CmGe => flag(sa >= sb),
        VecIntOp::CmHi => flag(a > b),
        VecIntOp::CmHs => flag(a >= b),
        VecIntOp::Sshl => shift_lane(a, bits, i64::from(b as u8 as i8), true),
        VecIntOp::Ushl => shift_lane(a, bits, i64::from(b as u8 as i8), false),
        VecIntOp::Smax => sa.max(sb) as u64,
        VecIntOp::Smin => sa.min(sb) as u64,
        VecIntOp::Umax => a.max(b),
        VecIntOp::Umin => a.min(b),
        VecIntOp::Uabd => a.abs_diff(b),
        VecIntOp::Sabd => sa.abs_diff(sb),
        VecIntOp::Sqadd => saturate_signed(wa + wb, bits, saturated),
        VecIntOp::Uqadd => saturate_unsigned(ua + ub, bits, saturated),
        VecIntOp::Sqsub => saturate_signed(wa - wb, bits, saturated),
        VecIntOp::Uqsub => saturate_unsigned(ua - ub, bits, saturated),
        VecIntOp::Shadd => ((wa + wb) >> 1) as u64,
        VecIntOp::Uhadd => ((ua + ub) >> 1) as u64,
        VecIntOp::Shsub => ((wa - wb) >> 1) as u64,
        VecIntOp::Uhsub => ((ua - ub) >> 1) as u64,
        VecIntOp::Srhadd => ((wa + wb + 1) >> 1) as u64,
        VecIntOp::Urhadd => ((ua + ub + 1) >> 1) as u64,
        VecIntOp::Sqdmulh => saturate_signed((2 * wa * wb) >> bits, bits, saturated),
        VecIntOp::Sqrdmulh => saturate_signed(
            (2 * wa * wb + (1i128 << (bits - 1))) >> bits,
            bits,
            saturated,
        ),
        VecIntOp::Mla => d.wrapping_add(a.wrapping_mul(b)),
        VecIntOp::Mls => d.wrapping_sub(a.wrapping_mul(b)),
        VecIntOp::Srshl
        | VecIntOp::Urshl
        | VecIntOp::Sqshl
        | VecIntOp::Uqshl
        | VecIntOp::Sqrshl
        | VecIntOp::Uqrshl => {
            let signed = matches!(op, VecIntOp::Srshl | VecIntOp::Sqshl | VecIntOp::Sqrshl);
            let round = matches!(
                op,
                VecIntOp::Srshl | VecIntOp::Urshl | VecIntOp::Sqrshl | VecIntOp::Uqrshl
            );
            let saturate = matches!(
                op,
                VecIntOp::Sqshl | VecIntOp::Uqshl | VecIntOp::Sqrshl | VecIntOp::Uqrshl
            );
            let shift = i64::from(b as u8 as i8);
            shift_register_lane(a, bits, shift, signed, round, saturate, saturated)
        }
    };
    value & mask
}

/// Lanewise integer operation on explicit Vn and Vm values.
#[allow(clippy::too_many_arguments)]
fn vec_int_values(
    cpu: &mut Cpu,
    op: VecIntOp,
    bits: u32,
    q: bool,
    scalar: bool,
    rd: u8,
    n: u128,
    m: u128,
) {
    let d = cpu.vectors[rd as usize];
    let mut saturated = false;
    let mut result = 0u128;
    for lane in 0..lane_count(bits, q, scalar) {
        let value = int_lane(
            op,
            bits,
            get_lane(n, bits, lane),
            get_lane(m, bits, lane),
            get_lane(d, bits, lane),
            &mut saturated,
        );
        set_lane(&mut result, bits, lane, value);
    }
    if saturated {
        cpu.fpsr |= FPSR_QC;
    }
    cpu.vectors[rd as usize] = finish_vector(result, bits, q, scalar);
}

/// Every `bits`-wide lane set to lane `index` of `value`.
fn broadcast(value: u128, bits: u32, index: u8) -> u128 {
    let element = get_lane(value, bits, u32::from(index));
    let mut result = 0u128;
    for lane in 0..128 / bits {
        set_lane(&mut result, bits, lane, element);
    }
    result
}

/// Integer operation by element: every lane of Vn with lane `index` of Vm.
#[allow(clippy::too_many_arguments)]
pub fn vec_int_element(
    cpu: &mut Cpu,
    op: VecIntOp,
    bits: u8,
    q: bool,
    scalar: bool,
    rd: u8,
    rn: u8,
    rm: u8,
    index: u8,
) {
    let bits = u32::from(bits);
    let (n, m) = (
        cpu.vectors[rn as usize],
        broadcast(cpu.vectors[rm as usize], bits, index),
    );
    vec_int_values(cpu, op, bits, q, scalar, rd, n, m);
}

/// ADDP/SMAXP/SMINP/UMAXP/UMINP. Vector: result lane i combines lanes 2i and
/// 2i+1 of Vm:Vn. Scalar (ADDP Dd, Vn.2D): the two lanes of Vn.
#[allow(clippy::too_many_arguments)]
pub fn vec_int_pairwise(
    cpu: &mut Cpu,
    op: VecIntOp,
    bits: u8,
    q: bool,
    scalar: bool,
    rd: u8,
    rn: u8,
    rm: u8,
) {
    let bits = u32::from(bits);
    let (n, m) = (cpu.vectors[rn as usize], cpu.vectors[rm as usize]);
    let mut unused = false;
    let mut result = 0u128;
    let lanes = lane_count(bits, q, scalar);
    let half = (lanes / 2).max(1);
    for lane in 0..lanes {
        let (source, pair) = if scalar || lane < half {
            (n, lane)
        } else {
            (m, lane - half)
        };
        let value = int_lane(
            op,
            bits,
            get_lane(source, bits, pair * 2),
            get_lane(source, bits, pair * 2 + 1),
            0,
            &mut unused,
        );
        set_lane(&mut result, bits, lane, value);
    }
    cpu.vectors[rd as usize] = finish_vector(result, bits, q, scalar);
}

fn shift_lane(value: u64, bits: u32, amount: i64, signed: bool) -> u64 {
    let mask = lane_mask(bits);
    if amount >= 0 {
        if amount >= i64::from(bits) {
            0
        } else {
            (value << amount) & mask
        }
    } else {
        let right = (-amount) as u32;
        if signed {
            let wide = sign_extend_lane(value, bits);
            let shifted = if right >= bits {
                wide >> (bits - 1)
            } else {
                wide >> right
            };
            (shifted as u64) & mask
        } else if right >= bits {
            0
        } else {
            value >> right
        }
    }
}

/// Lanewise integer operation on `bits`-wide lanes.
#[allow(clippy::too_many_arguments)]
pub fn vec_int(
    cpu: &mut Cpu,
    op: VecIntOp,
    bits: u8,
    q: bool,
    scalar: bool,
    rd: u8,
    rn: u8,
    rm: u8,
) {
    let (n, m) = (cpu.vectors[rn as usize], cpu.vectors[rm as usize]);
    vec_int_values(cpu, op, u32::from(bits), q, scalar, rd, n, m);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum VecUnaryOp {
    Cnt,
    Not,
    Rbit,
    Neg,
    Abs,
    /// Reverse lanes within each 64-, 32- or 16-bit container.
    Rev64,
    Rev32,
    Rev16,
    CmEqZero,
    CmGeZero,
    CmGtZero,
    CmLeZero,
    CmLtZero,
    /// Count leading zero bits / leading bits equal to the sign bit.
    Clz,
    Cls,
}

fn leading_zeros_lane(value: u64, bits: u32) -> u64 {
    u64::from(value.leading_zeros() - (64 - bits))
}

pub fn vec_unary(cpu: &mut Cpu, op: VecUnaryOp, bits: u8, q: bool, scalar: bool, rd: u8, rn: u8) {
    let bits = u32::from(bits);
    let n = cpu.vectors[rn as usize];
    let mask = lane_mask(bits);
    let count = lane_count(bits, q, scalar);
    let mut result = 0u128;
    let container = match op {
        VecUnaryOp::Rev64 => 64,
        VecUnaryOp::Rev32 => 32,
        VecUnaryOp::Rev16 => 16,
        _ => 0,
    };
    for lane in 0..count {
        let a = get_lane(n, bits, lane);
        let signed = sign_extend_lane(a, bits);
        let flag = |condition: bool| if condition { mask } else { 0 };
        let value = match op {
            VecUnaryOp::Cnt => u64::from(a.count_ones()),
            VecUnaryOp::Not => !a,
            VecUnaryOp::Rbit => a.reverse_bits() >> (64 - bits),
            VecUnaryOp::Neg => (signed.wrapping_neg()) as u64,
            VecUnaryOp::Abs => signed.wrapping_abs() as u64,
            VecUnaryOp::Rev64 | VecUnaryOp::Rev32 | VecUnaryOp::Rev16 => {
                let per = container / bits;
                let group = lane / per;
                let within = lane % per;
                get_lane(n, bits, group * per + (per - 1 - within))
            }
            VecUnaryOp::CmEqZero => flag(a == 0),
            VecUnaryOp::CmGeZero => flag(signed >= 0),
            VecUnaryOp::CmGtZero => flag(signed > 0),
            VecUnaryOp::CmLeZero => flag(signed <= 0),
            VecUnaryOp::CmLtZero => flag(signed < 0),
            VecUnaryOp::Clz => leading_zeros_lane(a, bits),
            VecUnaryOp::Cls => leading_zeros_lane((a ^ (signed >> 1) as u64) & mask, bits) - 1,
        };
        set_lane(&mut result, bits, lane, value & mask);
    }
    cpu.vectors[rd as usize] = finish_vector(result, bits, q, scalar);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum VecFpUnaryOp {
    Abs,
    Neg,
    Sqrt,
    Round(Rounding),
    /// FCVT<r>S / FCVT<r>U with `fbits` fraction bits (0 for integer).
    ToSigned(Rounding),
    ToUnsigned(Rounding),
    FromSigned,
    FromUnsigned,
    CmEqZero,
    CmGeZero,
    CmGtZero,
    CmLeZero,
    CmLtZero,
    /// FRECPE and FRSQRTE.
    RecipEstimate,
    RSqrtEstimate,
}

fn fp_to_integer_lane(
    fp: &mut Fp<'_>,
    value: u64,
    signed: bool,
    rounding: Rounding,
    fbits: u8,
) -> u64 {
    let bits = fp.format.bits;
    let input = fp.input(value);
    if fp.format.is_nan(input) {
        fp.raise(FPSR_IOC);
        return 0;
    }
    let scaled = fp.to_f64(input) * 2f64.powi(i32::from(fbits));
    let x = round(scaled, rounding);
    let (min, max) = match (signed, bits) {
        (true, 32) => (f64::from(i32::MIN), f64::from(i32::MAX)),
        (true, _) => (i64::MIN as f64, 9_223_372_036_854_775_808.0),
        (false, 32) => (0.0, f64::from(u32::MAX)),
        (false, _) => (0.0, 18_446_744_073_709_551_616.0),
    };
    let out_of_range = if bits == 64 {
        x < min || x >= max
    } else {
        x < min || x > max
    };
    if out_of_range {
        fp.raise(FPSR_IOC);
    }
    match (signed, bits) {
        (true, 32) => x as i32 as u32 as u64,
        (true, _) => x as i64 as u64,
        (false, 32) => x as u32 as u64,
        (false, _) => x as u64,
    }
}

fn integer_to_fp_lane(value: u64, size: FpSize, signed: bool, fbits: u8) -> u64 {
    let bits = size_bits(size);
    let scale = 2f64.powi(-i32::from(fbits));
    match size {
        FpSize::S => {
            // Convert through f64 only when exact; otherwise use the direct
            // single-rounding conversion (fbits == 0).
            let converted = if fbits == 0 {
                if signed {
                    sign_extend_lane(value, bits) as f32
                } else {
                    value as u32 as f32
                }
            } else {
                let wide = if signed {
                    sign_extend_lane(value, bits) as f64
                } else {
                    value as u32 as f64
                };
                (wide * scale) as f32
            };
            u64::from(converted.to_bits())
        }
        FpSize::D => {
            let converted = if signed {
                value as i64 as f64
            } else {
                value as f64
            };
            (if fbits == 0 {
                converted
            } else {
                converted * scale
            })
            .to_bits()
        }
    }
}

/// Lanewise FP unary operation; `fbits` applies to the fixed-point converts.
#[allow(clippy::too_many_arguments)]
pub fn vec_fp_unary(
    cpu: &mut Cpu,
    op: VecFpUnaryOp,
    size: FpSize,
    q: bool,
    scalar: bool,
    rd: u8,
    rn: u8,
    fbits: u8,
) -> Result<(), Trap> {
    let bits = size_bits(size);
    let n = cpu.vectors[rn as usize];
    let mut fp = Fp::new(cpu, size)?;
    let format = fp.format;
    let mut result = 0u128;
    for lane in 0..lane_count(bits, q, scalar) {
        let a = get_lane(n, bits, lane);
        let value = match op {
            VecFpUnaryOp::Abs => a & !format.sign_bit(),
            VecFpUnaryOp::Neg => a ^ format.sign_bit(),
            VecFpUnaryOp::Sqrt => {
                let a = fp.input(a);
                if let Some(nan) = fp.process_nans(&[a]) {
                    nan
                } else if a & format.sign_bit() != 0 && !is_zero(format, a) {
                    fp.raise(FPSR_IOC);
                    format.default_nan()
                } else {
                    let raw = native!(size, a, 0, |x, _y| x.sqrt());
                    fp.output(raw, false)
                }
            }
            VecFpUnaryOp::Round(rounding) => {
                let a = fp.input(a);
                if let Some(nan) = fp.process_nans(&[a]) {
                    nan
                } else {
                    native!(size, a, 0, |x, _y| round(x, rounding))
                }
            }
            VecFpUnaryOp::ToSigned(rounding) => {
                fp_to_integer_lane(&mut fp, a, true, rounding, fbits)
            }
            VecFpUnaryOp::ToUnsigned(rounding) => {
                fp_to_integer_lane(&mut fp, a, false, rounding, fbits)
            }
            VecFpUnaryOp::FromSigned => integer_to_fp_lane(a, size, true, fbits),
            VecFpUnaryOp::FromUnsigned => integer_to_fp_lane(a, size, false, fbits),
            VecFpUnaryOp::CmEqZero
            | VecFpUnaryOp::CmGeZero
            | VecFpUnaryOp::CmGtZero
            | VecFpUnaryOp::CmLeZero
            | VecFpUnaryOp::CmLtZero => {
                let a = fp.input(a);
                let (compare, swap) = match op {
                    VecFpUnaryOp::CmEqZero => (VecFpOp::CmEq, false),
                    VecFpUnaryOp::CmGeZero => (VecFpOp::CmGe, false),
                    VecFpUnaryOp::CmGtZero => (VecFpOp::CmGt, false),
                    VecFpUnaryOp::CmLeZero => (VecFpOp::CmGe, true),
                    _ => (VecFpOp::CmGt, true),
                };
                let (x, y) = if swap { (0, a) } else { (a, 0) };
                if fp_compare_lane(&mut fp, compare, x, y) {
                    lane_mask(bits)
                } else {
                    0
                }
            }
            VecFpUnaryOp::RecipEstimate => recip_estimate_lane(&mut fp, a),
            VecFpUnaryOp::RSqrtEstimate => rsqrt_estimate_lane(&mut fp, a),
        };
        set_lane(&mut result, bits, lane, value);
    }
    cpu.vectors[rd as usize] = finish_vector(result, bits, q, scalar);
    Ok(())
}

/// FCVTZS/FCVTZU (scalar, fixed-point) to a general register.
#[allow(clippy::too_many_arguments)]
pub fn fp_to_fixed(
    cpu: &mut Cpu,
    rd: u8,
    rn: u8,
    gp_bits: u8,
    size: FpSize,
    signed: bool,
    fbits: u8,
) -> Result<(), Trap> {
    let raw = read_fp(cpu, rn, size);
    let mut fp = Fp::new(cpu, size)?;
    let input = fp.input(raw);
    let value = if fp.format.is_nan(input) {
        fp.raise(FPSR_IOC);
        0
    } else {
        let x = (fp.to_f64(input) * 2f64.powi(i32::from(fbits))).trunc();
        let (min, max) = match (signed, gp_bits) {
            (true, 32) => (f64::from(i32::MIN), f64::from(i32::MAX)),
            (true, _) => (i64::MIN as f64, 9_223_372_036_854_775_808.0),
            (false, 32) => (0.0, f64::from(u32::MAX)),
            (false, _) => (0.0, 18_446_744_073_709_551_616.0),
        };
        let out_of_range = if gp_bits == 64 {
            x < min || x >= max
        } else {
            x < min || x > max
        };
        if out_of_range {
            fp.raise(FPSR_IOC);
        }
        match (signed, gp_bits) {
            (true, 32) => x as i32 as u32 as u64,
            (true, _) => x as i64 as u64,
            (false, 32) => x as u32 as u64,
            (false, _) => x as u64,
        }
    };
    cpu.write_reg(rd, gp_bits, false, value);
    Ok(())
}

/// SCVTF/UCVTF (scalar, fixed-point) from a general register.
pub fn int_to_fp_fixed(
    cpu: &mut Cpu,
    rd: u8,
    rn: u8,
    gp_bits: u8,
    size: FpSize,
    signed: bool,
    fbits: u8,
) -> Result<(), Trap> {
    mode(cpu)?;
    let value = cpu.read_reg(rn, gp_bits, false);
    let wide = match (signed, gp_bits) {
        (true, 32) => f64::from(value as u32 as i32),
        (true, _) => value as i64 as f64,
        (false, 32) => f64::from(value as u32),
        (false, _) => value as f64,
    };
    let scaled = wide * 2f64.powi(-i32::from(fbits));
    let bits = match size {
        FpSize::S => u64::from((scaled as f32).to_bits()),
        FpSize::D => scaled.to_bits(),
    };
    write_fp(cpu, rd, size, bits);
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum VecAcrossOp {
    AddV,
    SMaxV,
    SMinV,
    UMaxV,
    UMinV,
    /// Widening sums: the result is twice the lane width.
    SAddLV,
    UAddLV,
}

pub fn vec_across(cpu: &mut Cpu, op: VecAcrossOp, bits: u8, q: bool, rd: u8, rn: u8) {
    let bits = u32::from(bits);
    let n = cpu.vectors[rn as usize];
    let lanes: Vec<u64> = (0..lane_count(bits, q, false))
        .map(|lane| get_lane(n, bits, lane))
        .collect();
    let (value, out_bits) = match op {
        VecAcrossOp::AddV => (lanes.iter().fold(0u64, |acc, x| acc.wrapping_add(*x)), bits),
        VecAcrossOp::UMaxV => (
            *lanes.iter().max().expect("a vector has at least one lane"),
            bits,
        ),
        VecAcrossOp::UMinV => (
            *lanes.iter().min().expect("a vector has at least one lane"),
            bits,
        ),
        VecAcrossOp::SMaxV => (
            lanes
                .iter()
                .map(|x| sign_extend_lane(*x, bits))
                .max()
                .expect("a vector has at least one lane") as u64,
            bits,
        ),
        VecAcrossOp::SMinV => (
            lanes
                .iter()
                .map(|x| sign_extend_lane(*x, bits))
                .min()
                .expect("a vector has at least one lane") as u64,
            bits,
        ),
        VecAcrossOp::SAddLV => (
            lanes
                .iter()
                .map(|x| sign_extend_lane(*x, bits))
                .fold(0i64, |acc, x| acc.wrapping_add(x)) as u64,
            bits * 2,
        ),
        VecAcrossOp::UAddLV => (
            lanes.iter().fold(0u64, |acc, x| acc.wrapping_add(*x)),
            bits * 2,
        ),
    };
    cpu.vectors[rd as usize] = u128::from(value & lane_mask(out_bits));
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum VecShiftOp {
    Shl,
    Sshr,
    Ushr,
    Ssra,
    Usra,
    /// Shift right and narrow to half-width lanes (SHRN/SHRN2).
    Shrn,
    /// Rounding shift right (and accumulate).
    Srshr,
    Urshr,
    Srsra,
    Ursra,
    /// Saturating shift left: signed, unsigned, signed to unsigned.
    Sqshl,
    Uqshl,
    Sqshlu,
}

/// Rounding shift right by 1..=bits on a lane value widened to i128.
fn rounding_shift(value: i128, shift: i64) -> i128 {
    (value + (1i128 << (shift - 1))) >> shift
}

/// Shift by immediate on `bits`-wide lanes.
#[allow(clippy::too_many_arguments)]
pub fn vec_shift(
    cpu: &mut Cpu,
    op: VecShiftOp,
    bits: u8,
    q: bool,
    scalar: bool,
    rd: u8,
    rn: u8,
    shift: u8,
) {
    let bits = u32::from(bits);
    let (n, d) = (cpu.vectors[rn as usize], cpu.vectors[rd as usize]);
    let shift = i64::from(shift);
    if op == VecShiftOp::Shrn {
        // Source lanes are `bits` wide; results are bits/2 wide and fill the
        // lower (Q=0) or upper (Q=1) half of Vd.
        let half = bits / 2;
        let mut result = if q { d & u128::from(u64::MAX) } else { 0 };
        for lane in 0..(64 / bits * 2).min(128 / bits) {
            let value = get_lane(n, bits, lane) >> shift;
            let position = if q { lane + 64 / half } else { lane };
            set_lane(&mut result, half, position, value);
        }
        cpu.vectors[rd as usize] = result;
        return;
    }
    let mut saturated = false;
    let mut result = 0u128;
    for lane in 0..lane_count(bits, q, scalar) {
        let a = get_lane(n, bits, lane);
        let value = match op {
            VecShiftOp::Shl => shift_lane(a, bits, shift, false),
            VecShiftOp::Sshr => shift_lane(a, bits, -shift, true),
            VecShiftOp::Ushr => shift_lane(a, bits, -shift, false),
            VecShiftOp::Ssra => {
                get_lane(d, bits, lane).wrapping_add(shift_lane(a, bits, -shift, true))
            }
            VecShiftOp::Usra => {
                get_lane(d, bits, lane).wrapping_add(shift_lane(a, bits, -shift, false))
            }
            VecShiftOp::Shrn => unreachable!(),
            VecShiftOp::Srshr | VecShiftOp::Srsra => {
                let shifted = rounding_shift(i128::from(sign_extend_lane(a, bits)), shift) as u64;
                let base = if op == VecShiftOp::Srsra {
                    get_lane(d, bits, lane)
                } else {
                    0
                };
                base.wrapping_add(shifted)
            }
            VecShiftOp::Urshr | VecShiftOp::Ursra => {
                let shifted = rounding_shift(i128::from(a), shift) as u64;
                let base = if op == VecShiftOp::Ursra {
                    get_lane(d, bits, lane)
                } else {
                    0
                };
                base.wrapping_add(shifted)
            }
            VecShiftOp::Sqshl => saturate_signed(
                i128::from(sign_extend_lane(a, bits)) << shift,
                bits,
                &mut saturated,
            ),
            VecShiftOp::Uqshl => saturate_unsigned(i128::from(a) << shift, bits, &mut saturated),
            VecShiftOp::Sqshlu => saturate_unsigned(
                i128::from(sign_extend_lane(a, bits)) << shift,
                bits,
                &mut saturated,
            ),
        };
        set_lane(&mut result, bits, lane, value);
    }
    if saturated {
        cpu.fpsr |= FPSR_QC;
    }
    cpu.vectors[rd as usize] = finish_vector(result, bits, q, scalar);
}

/// SSHLL/USHLL (and SXTL/UXTL): widen the lower (`upper` false) or upper half
/// of Vn's `bits`-wide lanes to 2*bits and shift left.
pub fn vec_shift_long(
    cpu: &mut Cpu,
    signed: bool,
    bits: u8,
    upper: bool,
    rd: u8,
    rn: u8,
    shift: u8,
) {
    let bits = u32::from(bits);
    let n = cpu.vectors[rn as usize];
    let wide = bits * 2;
    let count = 64 / bits;
    let mut result = 0u128;
    for lane in 0..count {
        let source = get_lane(n, bits, lane + if upper { count } else { 0 });
        let extended = if signed {
            sign_extend_lane(source, bits) as u64
        } else {
            source
        };
        set_lane(&mut result, wide, lane, extended << shift);
    }
    cpu.vectors[rd as usize] = result;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum VecLongOp {
    /// d += n*m (widening)
    Mlal,
    /// d -= n*m (widening)
    Mlsl,
    Mull,
    Addl,
    Subl,
    /// SADDW/UADDW, SSUBW/USUBW: Vn is already wide.
    Addw,
    Subw,
}

/// Widening three-operand operations on the lower or upper half (the "2"
/// forms) of `bits`-wide lanes.
#[allow(clippy::too_many_arguments)]
pub fn vec_long(
    cpu: &mut Cpu,
    op: VecLongOp,
    signed: bool,
    bits: u8,
    upper: bool,
    rd: u8,
    rn: u8,
    rm: u8,
) {
    let (n, m) = (cpu.vectors[rn as usize], cpu.vectors[rm as usize]);
    vec_long_values(cpu, op, signed, u32::from(bits), upper, rd, n, m);
}

/// Widening operation by element (SMULL/UMULL/SMLAL/UMLAL/SMLSL/UMLSL).
#[allow(clippy::too_many_arguments)]
pub fn vec_long_element(
    cpu: &mut Cpu,
    op: VecLongOp,
    signed: bool,
    bits: u8,
    upper: bool,
    rd: u8,
    rn: u8,
    rm: u8,
    index: u8,
) {
    let bits = u32::from(bits);
    let (n, m) = (
        cpu.vectors[rn as usize],
        broadcast(cpu.vectors[rm as usize], bits, index),
    );
    vec_long_values(cpu, op, signed, bits, upper, rd, n, m);
}

#[allow(clippy::too_many_arguments)]
fn vec_long_values(
    cpu: &mut Cpu,
    op: VecLongOp,
    signed: bool,
    bits: u32,
    upper: bool,
    rd: u8,
    n: u128,
    m: u128,
) {
    let d = cpu.vectors[rd as usize];
    let wide = bits * 2;
    let count = 64 / bits;
    let extend = |value: u64| {
        if signed {
            sign_extend_lane(value, bits) as u64
        } else {
            value
        }
    };
    let mut result = 0u128;
    for lane in 0..count {
        let source = lane + if upper { count } else { 0 };
        let a = if matches!(op, VecLongOp::Addw | VecLongOp::Subw) {
            get_lane(n, wide, lane)
        } else {
            extend(get_lane(n, bits, source))
        };
        let b = extend(get_lane(m, bits, source));
        let acc = get_lane(d, wide, lane);
        let value = match op {
            VecLongOp::Mlal => acc.wrapping_add(a.wrapping_mul(b)),
            VecLongOp::Mlsl => acc.wrapping_sub(a.wrapping_mul(b)),
            VecLongOp::Mull => a.wrapping_mul(b),
            VecLongOp::Addl | VecLongOp::Addw => a.wrapping_add(b),
            VecLongOp::Subl | VecLongOp::Subw => a.wrapping_sub(b),
        };
        set_lane(&mut result, wide, lane, value);
    }
    cpu.vectors[rd as usize] = result;
}

/// SADDLP/UADDLP (and SADALP/UADALP with `accumulate`): add adjacent pairs of
/// `bits`-wide lanes into 2*bits-wide lanes.
pub fn vec_pairwise_long(
    cpu: &mut Cpu,
    signed: bool,
    accumulate: bool,
    bits: u8,
    q: bool,
    rd: u8,
    rn: u8,
) {
    let bits = u32::from(bits);
    let (n, d) = (cpu.vectors[rn as usize], cpu.vectors[rd as usize]);
    let wide = bits * 2;
    let pairs = lane_count(bits, q, false) / 2;
    let extend = |value: u64| {
        if signed {
            sign_extend_lane(value, bits) as u64
        } else {
            value
        }
    };
    let mut result = 0u128;
    for pair in 0..pairs {
        let sum = extend(get_lane(n, bits, pair * 2)).wrapping_add(extend(get_lane(
            n,
            bits,
            pair * 2 + 1,
        )));
        let value = if accumulate {
            get_lane(d, wide, pair).wrapping_add(sum)
        } else {
            sum
        };
        set_lane(&mut result, wide, pair, value);
    }
    cpu.vectors[rd as usize] = finish_vector(result, wide, q, false);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum VecNarrowOp {
    /// Truncate (XTN, SHRN, RSHRN).
    Truncate,
    /// Signed to signed saturating (SQXTN, SQSHRN, SQRSHRN).
    Signed,
    /// Unsigned to unsigned saturating (UQXTN, UQSHRN, UQRSHRN).
    Unsigned,
    /// Signed to unsigned saturating (SQXTUN, SQSHRUN, SQRSHRUN).
    SignedToUnsigned,
}

/// Narrow `bits`-wide source lanes to bits/2 after an optional (rounding)
/// right shift. Vector results fill the lower half of Vd (upper zeroed) or,
/// for the "2" forms (`upper`), the upper half keeping the lower; a scalar
/// result is one element with the rest of Vd zeroed.
#[allow(clippy::too_many_arguments)]
pub fn vec_narrow(
    cpu: &mut Cpu,
    op: VecNarrowOp,
    round: bool,
    bits: u8,
    upper: bool,
    scalar: bool,
    rd: u8,
    rn: u8,
    shift: u8,
) {
    let bits = u32::from(bits);
    let half = bits / 2;
    let (n, d) = (cpu.vectors[rn as usize], cpu.vectors[rd as usize]);
    let shift = i64::from(shift);
    let count = if scalar { 1 } else { 64 / half };
    let mut saturated = false;
    let mut result = if upper && !scalar {
        d & u128::from(u64::MAX)
    } else {
        0
    };
    for lane in 0..count {
        let a = get_lane(n, bits, lane);
        let wide = if matches!(op, VecNarrowOp::Unsigned) {
            i128::from(a)
        } else {
            i128::from(sign_extend_lane(a, bits))
        };
        let shifted = if shift == 0 {
            wide
        } else if round {
            rounding_shift(wide, shift)
        } else {
            wide >> shift
        };
        let value = match op {
            VecNarrowOp::Truncate => shifted as u64,
            VecNarrowOp::Signed => saturate_signed(shifted, half, &mut saturated),
            VecNarrowOp::Unsigned | VecNarrowOp::SignedToUnsigned => {
                saturate_unsigned(shifted, half, &mut saturated)
            }
        };
        let position = if upper && !scalar { lane + count } else { lane };
        set_lane(&mut result, half, position, value);
    }
    if saturated {
        cpu.fpsr |= FPSR_QC;
    }
    cpu.vectors[rd as usize] = result;
}

/// TBL/TBX: byte lookup in the `regs` consecutive registers from Vn (wrapping
/// at V31). Out-of-range indices give zero (TBL) or keep Vd's byte (TBX).
pub fn vec_table(cpu: &mut Cpu, q: bool, extend: bool, rd: u8, rn: u8, regs: u8, rm: u8) {
    let mut table = [0u8; 64];
    for r in 0..usize::from(regs) {
        let bytes = cpu.vectors[(usize::from(rn) + r) % 32].to_le_bytes();
        table[r * 16..r * 16 + 16].copy_from_slice(&bytes);
    }
    let limit = usize::from(regs) * 16;
    let indices = cpu.vectors[rm as usize].to_le_bytes();
    let mut result = cpu.vectors[rd as usize].to_le_bytes();
    let count = if q { 16 } else { 8 };
    for lane in 0..count {
        let index = usize::from(indices[lane]);
        if index < limit {
            result[lane] = table[index];
        } else if !extend {
            result[lane] = 0;
        }
    }
    let value = u128::from_le_bytes(result);
    cpu.vectors[rd as usize] = if q {
        value
    } else {
        value & u128::from(u64::MAX)
    };
}

/// Floating-point precisions for conversions that include half precision.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum FpPrecision {
    H,
    S,
    D,
}

impl FpPrecision {
    const fn format(self) -> Format {
        match self {
            FpPrecision::H => Format {
                bits: 16,
                exponent_bits: 5,
            },
            FpPrecision::S => Format::of(FpSize::S),
            FpPrecision::D => Format::of(FpSize::D),
        }
    }
}

/// Exact value of a finite H, S or D encoding.
fn precision_to_f64(precision: FpPrecision, value: u64) -> f64 {
    match precision {
        FpPrecision::S => f64::from(f32::from_bits(value as u32)),
        FpPrecision::D => f64::from_bits(value),
        FpPrecision::H => {
            let magnitude = {
                let exponent = ((value >> 10) & 31) as i32;
                let fraction = (value & 0x3ff) as f64;
                if exponent == 0 {
                    fraction * 2f64.powi(-24)
                } else if exponent == 31 {
                    f64::INFINITY
                } else {
                    (1024.0 + fraction) * 2f64.powi(exponent - 25)
                }
            };
            if value & 0x8000 != 0 {
                -magnitude
            } else {
                magnitude
            }
        }
    }
}

/// IEEE half-precision encoding of `value`, rounded to nearest even.
fn f64_to_half(value: f64) -> u64 {
    let sign = if value.is_sign_negative() { 0x8000 } else { 0 };
    let magnitude = value.abs();
    if magnitude.is_infinite() {
        return sign | 0x7c00;
    }
    if magnitude < 2f64.powi(-14) {
        // Subnormal (or zero): units of 2^-24; 1024 becomes the minimum normal.
        return sign | (magnitude * 2f64.powi(24)).round_ties_even() as u64;
    }
    let mut exponent = ((magnitude.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    let mantissa = magnitude / 2f64.powi(exponent);
    let mut fraction = ((mantissa - 1.0) * 1024.0).round_ties_even() as u64;
    if fraction == 1024 {
        fraction = 0;
        exponent += 1;
    }
    if exponent > 15 {
        return sign | 0x7c00;
    }
    sign | (((exponent + 15) as u64) << 10) | fraction
}

/// FPConvert between any two of H, S and D (IEEE half precision only).
fn convert_precision(
    cpu: &mut Cpu,
    mode: Mode,
    value: u64,
    from: FpPrecision,
    to: FpPrecision,
) -> u64 {
    let source = from.format();
    let target = to.format();
    let mut value = value;
    // FZ flushes single/double inputs and outputs; half precision uses FZ16,
    // which is required to be clear.
    if mode.fz && from != FpPrecision::H && source.is_denormal(value) {
        cpu.fpsr |= FPSR_IDC;
        value &= source.sign_bit();
    }
    let negative = value & source.sign_bit() != 0;
    let sign = if negative { target.sign_bit() } else { 0 };
    if source.is_nan(value) {
        if source.is_signaling(value) {
            cpu.fpsr |= FPSR_IOC;
        }
        if mode.dn {
            return target.default_nan();
        }
        let payload = value & source.fraction_mask();
        let shifted = if target.fraction_bits() > source.fraction_bits() {
            payload << (target.fraction_bits() - source.fraction_bits())
        } else {
            payload >> (source.fraction_bits() - target.fraction_bits())
        };
        return sign | target.exponent_mask() | target.quiet_bit() | shifted;
    }
    let exact = precision_to_f64(from, value);
    let converted = match to {
        FpPrecision::H => f64_to_half(exact),
        FpPrecision::S => u64::from((exact as f32).to_bits()),
        FpPrecision::D => exact.to_bits(),
    };
    if is_infinity(target, converted) && !is_infinity(source, value) {
        cpu.fpsr |= FPSR_OFC | FPSR_IXC;
    }
    if mode.fz && to != FpPrecision::H && target.is_denormal(converted) {
        cpu.fpsr |= FPSR_UFC;
        return converted & target.sign_bit();
    }
    converted
}

fn half_mode(cpu: &Cpu) -> Result<Mode, Trap> {
    if cpu.fpcr & FPCR_AHP != 0 {
        return Err(Trap::InvalidState {
            message: format!(
                "FPCR.AHP alternative half precision is not implemented at {:#x}",
                cpu.pc
            ),
        });
    }
    mode(cpu)
}

/// Scalar FCVT where one side is half precision.
pub fn fp_convert_precision(
    cpu: &mut Cpu,
    rd: u8,
    rn: u8,
    from: FpPrecision,
    to: FpPrecision,
) -> Result<(), Trap> {
    let mode = half_mode(cpu)?;
    let source_bits = from.format().bits;
    let value = (cpu.vectors[rn as usize] as u64) & lane_mask(source_bits);
    let result = convert_precision(cpu, mode, value, from, to);
    cpu.vectors[rd as usize] = u128::from(result);
    Ok(())
}

/// FCVTL/FCVTL2 (`narrow` false: widen the lower or upper half of Vn) and
/// FCVTN/FCVTN2 (`narrow` true: write the lower or upper half of Vd).
pub fn vec_fp_convert(
    cpu: &mut Cpu,
    from: FpPrecision,
    to: FpPrecision,
    upper: bool,
    rd: u8,
    rn: u8,
) -> Result<(), Trap> {
    let mode = half_mode(cpu)?;
    let (source_bits, target_bits) = (from.format().bits, to.format().bits);
    let (n, d) = (cpu.vectors[rn as usize], cpu.vectors[rd as usize]);
    let narrow = target_bits < source_bits;
    let count = 64 / source_bits.min(target_bits);
    let mut result = if narrow && upper {
        d & u128::from(u64::MAX)
    } else {
        0
    };
    for lane in 0..count {
        let source_lane = if !narrow && upper { lane + count } else { lane };
        let target_lane = if narrow && upper { lane + count } else { lane };
        let value = convert_precision(cpu, mode, get_lane(n, source_bits, source_lane), from, to);
        set_lane(&mut result, target_bits, target_lane, value);
    }
    cpu.vectors[rd as usize] = result;
    Ok(())
}

/// PMULL/PMULL2: carry-less multiply of the lower or upper 8-bit (into
/// 16-bit) or 64-bit (into 128-bit) elements.
pub fn vec_poly_mul_long(cpu: &mut Cpu, bits: u8, upper: bool, rd: u8, rn: u8, rm: u8) {
    let bits = u32::from(bits);
    let (n, m) = (cpu.vectors[rn as usize], cpu.vectors[rm as usize]);
    let count = 64 / bits;
    let mut result = 0u128;
    for lane in 0..count {
        let source = lane + if upper { count } else { 0 };
        let (a, b) = (
            u128::from(get_lane(n, bits, source)),
            u128::from(get_lane(m, bits, source)),
        );
        let mut product = 0u128;
        for bit in 0..bits {
            if b >> bit & 1 != 0 {
                product ^= a << bit;
            }
        }
        let wide = bits * 2;
        let shift = lane * wide;
        let lane_mask = if wide == 128 {
            u128::MAX
        } else {
            (1u128 << wide) - 1
        };
        result |= (product & lane_mask) << shift;
    }
    cpu.vectors[rd as usize] = result;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum CryptoOp {
    Aese,
    Aesd,
    Aesmc,
    Aesimc,
    Sha1c,
    Sha1p,
    Sha1m,
    Sha1h,
    Sha1su0,
    Sha1su1,
    Sha256h,
    Sha256h2,
    Sha256su0,
    Sha256su1,
}

const fn gf_mul(mut a: u8, mut b: u8) -> u8 {
    let mut product = 0u8;
    while b != 0 {
        if b & 1 != 0 {
            product ^= a;
        }
        let carry = a & 0x80 != 0;
        a <<= 1;
        if carry {
            a ^= 0x1b;
        }
        b >>= 1;
    }
    product
}

/// The AES S-box and its inverse, derived from the GF(2^8) inverse and the
/// FIPS-197 affine transform.
const AES_SBOXES: ([u8; 256], [u8; 256]) = {
    let mut forward = [0u8; 256];
    let mut inverse = [0u8; 256];
    let mut x = 0usize;
    while x < 256 {
        // x^254 is the multiplicative inverse (0 maps to 0).
        let mut power = 1u8;
        let mut index = 0;
        while index < 254 {
            power = gf_mul(power, x as u8);
            index += 1;
        }
        let b = if x == 0 { 0 } else { power };
        let value =
            b ^ b.rotate_left(1) ^ b.rotate_left(2) ^ b.rotate_left(3) ^ b.rotate_left(4) ^ 0x63;
        forward[x] = value;
        inverse[value as usize] = x as u8;
        x += 1;
    }
    (forward, inverse)
};

fn mix_columns(state: [u8; 16], inverse: bool) -> [u8; 16] {
    let coefficients: [u8; 4] = if inverse {
        [14, 11, 13, 9]
    } else {
        [2, 3, 1, 1]
    };
    let mut out = [0u8; 16];
    for column in 0..4 {
        for row in 0..4 {
            let mut value = 0;
            for k in 0..4 {
                value ^= gf_mul(coefficients[(k + 4 - row) % 4], state[column * 4 + k]);
            }
            out[column * 4 + row] = value;
        }
    }
    out
}

fn sha_choose(x: u32, y: u32, z: u32) -> u32 {
    ((y ^ z) & x) ^ z
}

fn sha_majority(x: u32, y: u32, z: u32) -> u32 {
    (x & y) | ((x | y) & z)
}

fn lanes_u32(value: u128) -> [u32; 4] {
    std::array::from_fn(|i| (value >> (32 * i)) as u32)
}

fn from_lanes_u32(lanes: [u32; 4]) -> u128 {
    lanes
        .iter()
        .enumerate()
        .fold(0, |acc, (i, &lane)| acc | (u128::from(lane) << (32 * i)))
}

/// SHA256hash from the Arm pseudocode; `part1` returns X (SHA256H), otherwise
/// Y (SHA256H2).
fn sha256_hash(x: u128, y: u128, w: u128, part1: bool) -> u128 {
    let (mut x, mut y, w) = (lanes_u32(x), lanes_u32(y), lanes_u32(w));
    for word in w {
        let choose = sha_choose(y[0], y[1], y[2]);
        let majority = sha_majority(x[0], x[1], x[2]);
        let sigma1 = y[0].rotate_right(6) ^ y[0].rotate_right(11) ^ y[0].rotate_right(25);
        let sigma0 = x[0].rotate_right(2) ^ x[0].rotate_right(13) ^ x[0].rotate_right(22);
        let t = y[3]
            .wrapping_add(sigma1)
            .wrapping_add(choose)
            .wrapping_add(word);
        x[3] = t.wrapping_add(x[3]);
        y[3] = t.wrapping_add(sigma0).wrapping_add(majority);
        // <Y, X> = ROL(Y : X, 32)
        let (x3, y3) = (x[3], y[3]);
        x = [y3, x[0], x[1], x[2]];
        y = [x3, y[0], y[1], y[2]];
    }
    from_lanes_u32(if part1 { x } else { y })
}

/// AES, SHA-1 and SHA-256 instructions (FEAT_AES, FEAT_SHA1, FEAT_SHA256).
pub fn crypto(cpu: &mut Cpu, op: CryptoOp, rd: u8, rn: u8, rm: u8) {
    let (d, n, m) = (
        cpu.vectors[rd as usize],
        cpu.vectors[rn as usize],
        cpu.vectors[rm as usize],
    );
    let (sbox, inverse_sbox) = (&AES_SBOXES.0, &AES_SBOXES.1);
    let result = match op {
        CryptoOp::Aese | CryptoOp::Aesd => {
            let state = (d ^ n).to_le_bytes();
            let mut out = [0u8; 16];
            for column in 0..4 {
                for row in 0..4 {
                    if op == CryptoOp::Aese {
                        // ShiftRows, then SubBytes.
                        out[row + 4 * column] =
                            sbox[state[row + 4 * ((column + row) % 4)] as usize];
                    } else {
                        // InvShiftRows, then InvSubBytes.
                        out[row + 4 * ((column + row) % 4)] =
                            inverse_sbox[state[row + 4 * column] as usize];
                    }
                }
            }
            u128::from_le_bytes(out)
        }
        CryptoOp::Aesmc | CryptoOp::Aesimc => {
            u128::from_le_bytes(mix_columns(n.to_le_bytes(), op == CryptoOp::Aesimc))
        }
        CryptoOp::Sha1c | CryptoOp::Sha1p | CryptoOp::Sha1m => {
            let mut x = lanes_u32(d);
            let mut y = n as u32;
            let w = lanes_u32(m);
            for word in w {
                let t = match op {
                    CryptoOp::Sha1c => sha_choose(x[1], x[2], x[3]),
                    CryptoOp::Sha1p => x[1] ^ x[2] ^ x[3],
                    _ => sha_majority(x[1], x[2], x[3]),
                };
                y = y
                    .wrapping_add(x[0].rotate_left(5))
                    .wrapping_add(t)
                    .wrapping_add(word);
                x[1] = x[1].rotate_left(30);
                // <Y, X> = ROL(Y : X, 32)
                let top = x[3];
                x = [y, x[0], x[1], x[2]];
                y = top;
            }
            from_lanes_u32(x)
        }
        CryptoOp::Sha1h => u128::from((n as u32).rotate_left(30)),
        CryptoOp::Sha1su0 => ((n << 64) | (d >> 64)) ^ d ^ m,
        CryptoOp::Sha1su1 => {
            let t = lanes_u32(d ^ (n >> 32));
            from_lanes_u32([
                t[0].rotate_left(1),
                t[1].rotate_left(1),
                t[2].rotate_left(1),
                t[3].rotate_left(1) ^ t[0].rotate_left(2),
            ])
        }
        CryptoOp::Sha256h => sha256_hash(d, n, m, true),
        CryptoOp::Sha256h2 => sha256_hash(n, d, m, false),
        CryptoOp::Sha256su0 => {
            let operand1 = lanes_u32(d);
            let t = lanes_u32((n << 96) | (d >> 32));
            from_lanes_u32(std::array::from_fn(|e| {
                let element = t[e];
                (element.rotate_right(7) ^ element.rotate_right(18) ^ (element >> 3))
                    .wrapping_add(operand1[e])
            }))
        }
        CryptoOp::Sha256su1 => {
            let operand1 = lanes_u32(d);
            let t0 = lanes_u32((m << 96) | (n >> 32));
            let sigma = |element: u32| {
                element.rotate_right(17) ^ element.rotate_right(19) ^ (element >> 10)
            };
            let mut result = [0u32; 4];
            let t1 = lanes_u32(m >> 64);
            for e in 0..2 {
                result[e] = sigma(t1[e]).wrapping_add(operand1[e]).wrapping_add(t0[e]);
            }
            for e in 2..4 {
                result[e] = sigma(result[e - 2])
                    .wrapping_add(operand1[e])
                    .wrapping_add(t0[e]);
            }
            from_lanes_u32(result)
        }
    };
    cpu.vectors[rd as usize] = result;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum VecPermuteOp {
    Uzp1,
    Uzp2,
    Trn1,
    Trn2,
    Zip1,
    Zip2,
}

pub fn vec_permute(cpu: &mut Cpu, op: VecPermuteOp, bits: u8, q: bool, rd: u8, rn: u8, rm: u8) {
    let bits = u32::from(bits);
    let (n, m) = (cpu.vectors[rn as usize], cpu.vectors[rm as usize]);
    let count = lane_count(bits, q, false);
    let half = count / 2;
    let mut result = 0u128;
    for lane in 0..count {
        let value = match op {
            VecPermuteOp::Uzp1 | VecPermuteOp::Uzp2 => {
                let odd = u32::from(op == VecPermuteOp::Uzp2);
                let source = lane * 2 + odd;
                if source < count {
                    get_lane(n, bits, source)
                } else {
                    get_lane(m, bits, source - count)
                }
            }
            VecPermuteOp::Trn1 | VecPermuteOp::Trn2 => {
                let odd = u32::from(op == VecPermuteOp::Trn2);
                let base = (lane & !1) + odd;
                if lane % 2 == 0 {
                    get_lane(n, bits, base)
                } else {
                    get_lane(m, bits, base)
                }
            }
            VecPermuteOp::Zip1 | VecPermuteOp::Zip2 => {
                let base = if op == VecPermuteOp::Zip2 { half } else { 0 };
                let source = base + lane / 2;
                if lane % 2 == 0 {
                    get_lane(n, bits, source)
                } else {
                    get_lane(m, bits, source)
                }
            }
        };
        set_lane(&mut result, bits, lane, value);
    }
    cpu.vectors[rd as usize] = finish_vector(result, bits, q, false);
}

/// EXT: bytes `index`.. of the concatenation Vm:Vn (Vn in the low part).
pub fn vec_ext(cpu: &mut Cpu, q: bool, rd: u8, rn: u8, rm: u8, index: u8) {
    let width = if q { 16 } else { 8 };
    let n = cpu.vectors[rn as usize].to_le_bytes();
    let m = cpu.vectors[rm as usize].to_le_bytes();
    let mut bytes = [0u8; 16];
    for (position, byte) in bytes.iter_mut().take(width).enumerate() {
        let source = usize::from(index) + position;
        *byte = if source < width {
            n[source]
        } else {
            m[source - width]
        };
    }
    cpu.vectors[rd as usize] = u128::from_le_bytes(bytes);
}

/// LD1-LD4/ST1-ST4 (multiple structures). `structure` is the interleave
/// (1 for LD1 of `registers` consecutive registers). Returns the bytes moved.
#[allow(clippy::too_many_arguments)]
pub fn transfer_multiple(
    cpu: &mut Cpu,
    memory: &mut GuestView,
    address: u64,
    rt: u8,
    registers: u8,
    structure: u8,
    bits: u8,
    q: bool,
    load: bool,
) -> Result<u64, Trap> {
    let esize = u64::from(bits / 8);
    let per_register = if q { 16 } else { 8 };
    let lanes = per_register / esize;
    let total = u64::from(registers) * per_register;
    let register = |r: u64| ((u64::from(rt) + r) % 32) as usize;
    let mut buffer = vec![0u8; total as usize];
    if load {
        memory.read(address, &mut buffer)?;
        let mut values = [0u128; 4];
        for r in 0..u64::from(registers) {
            for lane in 0..lanes {
                let offset = if structure == 1 {
                    r * per_register + lane * esize
                } else {
                    (lane * u64::from(structure) + r) * esize
                };
                let mut chunk = [0u8; 8];
                chunk[..esize as usize]
                    .copy_from_slice(&buffer[offset as usize..(offset + esize) as usize]);
                set_lane(
                    &mut values[r as usize],
                    u32::from(bits),
                    lane as u32,
                    u64::from_le_bytes(chunk),
                );
            }
        }
        for r in 0..u64::from(registers) {
            cpu.vectors[register(r)] = values[r as usize];
        }
    } else {
        for r in 0..u64::from(registers) {
            let value = cpu.vectors[register(r)];
            for lane in 0..lanes {
                let offset = if structure == 1 {
                    r * per_register + lane * esize
                } else {
                    (lane * u64::from(structure) + r) * esize
                };
                let lane_value = get_lane(value, u32::from(bits), lane as u32).to_le_bytes();
                buffer[offset as usize..(offset + esize) as usize]
                    .copy_from_slice(&lane_value[..esize as usize]);
            }
        }
        memory.write(address, &buffer)?;
    }
    Ok(total)
}

/// LD1-LD4/ST1-ST4 (single structure) to/from lane `index` of `registers`
/// consecutive registers; other lanes are preserved. Returns the bytes moved.
#[allow(clippy::too_many_arguments)]
pub fn transfer_single(
    cpu: &mut Cpu,
    memory: &mut GuestView,
    address: u64,
    rt: u8,
    registers: u8,
    bits: u8,
    index: u8,
    load: bool,
) -> Result<u64, Trap> {
    let esize = usize::from(bits / 8);
    let total = usize::from(registers) * esize;
    let register = |r: usize| (usize::from(rt) + r) % 32;
    let mut buffer = vec![0u8; total];
    if load {
        memory.read(address, &mut buffer)?;
        for r in 0..usize::from(registers) {
            let mut chunk = [0u8; 8];
            chunk[..esize].copy_from_slice(&buffer[r * esize..(r + 1) * esize]);
            set_lane(
                &mut cpu.vectors[register(r)],
                u32::from(bits),
                u32::from(index),
                u64::from_le_bytes(chunk),
            );
        }
    } else {
        for r in 0..usize::from(registers) {
            let lane = get_lane(cpu.vectors[register(r)], u32::from(bits), u32::from(index));
            buffer[r * esize..(r + 1) * esize].copy_from_slice(&lane.to_le_bytes()[..esize]);
        }
        memory.write(address, &buffer)?;
    }
    Ok(total as u64)
}

/// LD1R-LD4R: load one structure and replicate each element to all lanes.
#[allow(clippy::too_many_arguments)]
pub fn load_replicate(
    cpu: &mut Cpu,
    memory: &GuestView,
    address: u64,
    rt: u8,
    registers: u8,
    bits: u8,
    q: bool,
) -> Result<u64, Trap> {
    let esize = usize::from(bits / 8);
    let total = usize::from(registers) * esize;
    let mut buffer = vec![0u8; total];
    memory.read(address, &mut buffer)?;
    for r in 0..usize::from(registers) {
        let mut chunk = [0u8; 8];
        chunk[..esize].copy_from_slice(&buffer[r * esize..(r + 1) * esize]);
        let element = u64::from_le_bytes(chunk);
        let mut value = 0u128;
        for lane in 0..lane_count(u32::from(bits), q, false) {
            set_lane(&mut value, u32::from(bits), lane, element);
        }
        cpu.vectors[(usize::from(rt) + r) % 32] = value;
    }
    Ok(total as u64)
}

#[cfg(test)]
mod vector_tests {
    use super::*;
    #[cfg(target_os = "linux")]
    use crate::memory::{Memory, READ, STATE_NORMAL, WRITE};

    fn lanes32(values: [u32; 4]) -> u128 {
        values
            .iter()
            .enumerate()
            .fold(0, |acc, (i, &v)| acc | (u128::from(v) << (32 * i)))
    }

    #[test]
    fn lanewise_float_and_integer_arithmetic() {
        let mut cpu = Cpu::default();
        cpu.vectors[1] = lanes32([1.0f32, 2.0, 3.0, 4.0].map(f32::to_bits));
        cpu.vectors[2] = lanes32([10.0f32, 20.0, 30.0, 40.0].map(f32::to_bits));
        vec_fp(&mut cpu, VecFpOp::Add, FpSize::S, true, false, 0, 1, 2).unwrap();
        assert_eq!(
            cpu.vectors[0],
            lanes32([11.0f32, 22.0, 33.0, 44.0].map(f32::to_bits))
        );
        cpu.vectors[1] = lanes32([1, 2, 3, u32::MAX]);
        cpu.vectors[2] = lanes32([5, 6, 7, 1]);
        vec_int(&mut cpu, VecIntOp::Add, 32, true, false, 0, 1, 2);
        assert_eq!(cpu.vectors[0], lanes32([6, 8, 10, 0]));
    }

    #[test]
    fn pairwise_float_takes_adjacent_lanes_of_n_then_m() {
        let mut cpu = Cpu::default();
        cpu.vectors[1] = lanes32([1.0f32, 2.0, 3.0, 4.0].map(f32::to_bits));
        cpu.vectors[2] = lanes32([10.0f32, 20.0, 30.0, -40.0].map(f32::to_bits));
        vec_fp_pairwise(&mut cpu, VecFpOp::Add, FpSize::S, true, false, 0, 1, 2).unwrap();
        assert_eq!(
            cpu.vectors[0],
            lanes32([3.0f32, 7.0, 30.0, -10.0].map(f32::to_bits))
        );
        vec_fp_pairwise(&mut cpu, VecFpOp::Max, FpSize::S, false, false, 0, 1, 2).unwrap();
        assert_eq!(
            cpu.vectors[0],
            lanes32([2.0f32, 20.0, 0.0, 0.0].map(f32::to_bits)) & u128::from(u64::MAX)
        );
        // Scalar form: one element from the two low lanes of Vn.
        vec_fp_pairwise(&mut cpu, VecFpOp::Add, FpSize::S, false, true, 0, 1, 0).unwrap();
        assert_eq!(cpu.vectors[0], u128::from(3.0f32.to_bits()));
    }

    #[test]
    fn reciprocal_estimates_match_the_architectural_tables() {
        let mut cpu = Cpu::default();
        cpu.vectors[1] = lanes32([1.0f32, 2.0, 4.0, 0.0].map(f32::to_bits));
        vec_fp_unary(
            &mut cpu,
            VecFpUnaryOp::RecipEstimate,
            FpSize::S,
            true,
            false,
            0,
            1,
            0,
        )
        .unwrap();
        assert_eq!(
            cpu.vectors[0],
            lanes32([0x3f7f_8000, 0x3eff_8000, 0x3e7f_8000, 0x7f80_0000])
        );
        assert_ne!(cpu.fpsr & FPSR_DZC, 0);
        vec_fp_unary(
            &mut cpu,
            VecFpUnaryOp::RSqrtEstimate,
            FpSize::S,
            true,
            false,
            0,
            1,
            0,
        )
        .unwrap();
        assert_eq!(
            cpu.vectors[0],
            lanes32([0x3f7f_8000, 0x3f34_8000, 0x3eff_8000, 0x7f80_0000])
        );
        // Newton step: FRSQRTS(a, b) = (3 - a*b) / 2; FRECPS = 2 - a*b.
        cpu.vectors[1] = lanes32([1.0f32, 0.5, 2.0, f32::INFINITY].map(f32::to_bits));
        cpu.vectors[2] = lanes32([1.0f32, 2.0, 0.25, 0.0].map(f32::to_bits));
        vec_fp(
            &mut cpu,
            VecFpOp::RSqrtStep,
            FpSize::S,
            true,
            false,
            0,
            1,
            2,
        )
        .unwrap();
        assert_eq!(
            cpu.vectors[0],
            lanes32([1.0f32, 1.0, 1.25, 1.5].map(f32::to_bits))
        );
        vec_fp(
            &mut cpu,
            VecFpOp::RecipStep,
            FpSize::S,
            true,
            false,
            0,
            1,
            2,
        )
        .unwrap();
        assert_eq!(
            cpu.vectors[0],
            lanes32([1.0f32, 1.0, 1.5, 2.0].map(f32::to_bits))
        );
    }

    #[test]
    fn table_lookup_saturation_and_half_precision() {
        let mut cpu = Cpu::default();
        cpu.vectors[1] = u128::from_le_bytes(std::array::from_fn(|i| i as u8 + 0x10));
        cpu.vectors[2] = u128::from_le_bytes(std::array::from_fn(|i| 0x20 + i as u8));
        cpu.vectors[3] =
            u128::from_le_bytes([0, 15, 16, 31, 32, 1, 0xff, 2, 0, 0, 0, 0, 0, 0, 0, 0]);
        cpu.vectors[0] = u128::MAX;
        vec_table(&mut cpu, false, false, 0, 1, 2, 3);
        assert_eq!(
            cpu.vectors[0],
            u128::from(u64::from_le_bytes([
                0x10, 0x1f, 0x20, 0x2f, 0, 0x11, 0, 0x12
            ]))
        );
        cpu.vectors[0] = u128::MAX;
        vec_table(&mut cpu, false, true, 0, 1, 1, 3);
        assert_eq!(
            cpu.vectors[0],
            u128::from(u64::from_le_bytes([
                0x10, 0x1f, 0xff, 0xff, 0xff, 0x11, 0xff, 0x12
            ]))
        );

        cpu.vectors[1] = lanes32([0x7fff, 0xffff_8000, 0x1_0000, 0xfff0_0000]);
        cpu.fpsr = 0;
        vec_narrow(
            &mut cpu,
            VecNarrowOp::Signed,
            false,
            32,
            false,
            false,
            0,
            1,
            0,
        );
        assert_eq!(cpu.vectors[0], 0x8000_7fff_8000_7fff);
        assert_ne!(cpu.fpsr & FPSR_QC, 0);
        vec_narrow(
            &mut cpu,
            VecNarrowOp::Truncate,
            false,
            32,
            false,
            false,
            0,
            1,
            0,
        );
        assert_eq!(cpu.vectors[0], 0x0000_0000_8000_7fff);

        cpu.vectors[1] = lanes32([0x4000_0000, 0x8000_0000, 0x2000_0000, 0]);
        cpu.vectors[2] = lanes32([0x4000_0000, 0x8000_0000, 0xc000_0000, 0]);
        cpu.fpsr = 0;
        vec_int(&mut cpu, VecIntOp::Sqrdmulh, 32, true, false, 0, 1, 2);
        assert_eq!(
            cpu.vectors[0],
            lanes32([0x2000_0000, 0x7fff_ffff, 0xf000_0000, 0])
        );
        assert_ne!(cpu.fpsr & FPSR_QC, 0);

        // 1.0, -2.0, 65504 (max half) and a subnormal 2^-24 widen exactly.
        cpu.vectors[1] = 0x0001_7bff_c000_3c00;
        vec_fp_convert(&mut cpu, FpPrecision::H, FpPrecision::S, false, 0, 1).unwrap();
        assert_eq!(
            cpu.vectors[0],
            lanes32([1.0f32, -2.0, 65504.0, 2f32.powi(-24)].map(f32::to_bits))
        );
        cpu.vectors[1] = cpu.vectors[0];
        vec_fp_convert(&mut cpu, FpPrecision::S, FpPrecision::H, false, 0, 1).unwrap();
        assert_eq!(cpu.vectors[0], 0x0001_7bff_c000_3c00);
    }

    fn bytes(hex: &str) -> u128 {
        let mut out = [0u8; 16];
        for (i, byte) in out.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
        }
        u128::from_le_bytes(out)
    }

    #[test]
    fn aes_rounds_match_fips_197() {
        let mut cpu = Cpu::default();
        // FIPS-197 appendix B, round 1 (state bytes in column order).
        let start = bytes("193de3bea0f4e22b9ac68d2ae9f84808");
        cpu.vectors[0] = start;
        crypto(&mut cpu, CryptoOp::Aese, 0, 1, 0);
        assert_eq!(cpu.vectors[0], bytes("d4bf5d30e0b452aeb84111f11e2798e5"));
        crypto(&mut cpu, CryptoOp::Aesmc, 2, 0, 0);
        assert_eq!(cpu.vectors[2], bytes("046681e5e0cb199a48f8d37a2806264c"));
        crypto(&mut cpu, CryptoOp::Aesimc, 3, 2, 0);
        assert_eq!(cpu.vectors[3], cpu.vectors[0]);
        crypto(&mut cpu, CryptoOp::Aesd, 0, 1, 0);
        assert_eq!(cpu.vectors[0], start);
    }

    /// One padded block holding "abc", as big-endian 32-bit words.
    fn abc_block() -> [u32; 16] {
        let mut words = [0u32; 16];
        words[0] = 0x6162_6380;
        words[15] = 24;
        words
    }

    fn lanes(words: &[u32]) -> u128 {
        lanes32([words[0], words[1], words[2], words[3]])
    }

    #[test]
    fn sha256_instructions_hash_abc() {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];
        let initial = [
            0x6a09e667u32,
            0xbb67ae85,
            0x3c6ef372,
            0xa54ff53a,
            0x510e527f,
            0x9b05688c,
            0x1f83d9ab,
            0x5be0cd19,
        ];
        let mut cpu = Cpu::default();
        let block = abc_block();
        // v0/v1 state, v4..v7 message, v2 W+K, v3 saved state.
        cpu.vectors[0] = lanes(&initial[..4]);
        cpu.vectors[1] = lanes(&initial[4..]);
        for i in 0..4 {
            cpu.vectors[4 + i] = lanes(&block[i * 4..]);
        }
        for group in 0..16 {
            let message = 4 + group % 4;
            let w = lanes32(std::array::from_fn(|e| {
                let word = (cpu.vectors[message] >> (32 * e)) as u32;
                word.wrapping_add(K[group * 4 + e])
            }));
            cpu.vectors[2] = w;
            cpu.vectors[3] = cpu.vectors[0];
            crypto(&mut cpu, CryptoOp::Sha256h, 0, 1, 2);
            crypto(&mut cpu, CryptoOp::Sha256h2, 1, 3, 2);
            if group < 12 {
                let (m0, m1, m2, m3) = (
                    message,
                    4 + (group + 1) % 4,
                    4 + (group + 2) % 4,
                    4 + (group + 3) % 4,
                );
                crypto(&mut cpu, CryptoOp::Sha256su0, m0 as u8, m1 as u8, 0);
                crypto(&mut cpu, CryptoOp::Sha256su1, m0 as u8, m2 as u8, m3 as u8);
            }
        }
        let digest: Vec<u32> = (0..8)
            .map(|i| ((cpu.vectors[i / 4] >> (32 * (i % 4))) as u32).wrapping_add(initial[i]))
            .collect();
        assert_eq!(
            digest,
            [
                0xba7816bf, 0x8f01cfea, 0x414140de, 0x5dae2223, 0xb00361a3, 0x96177a9c, 0xb410ff61,
                0xf20015ad
            ]
        );
    }

    #[test]
    fn sha1_instructions_hash_abc() {
        let initial = [
            0x67452301u32,
            0xefcdab89,
            0x98badcfe,
            0x10325476,
            0xc3d2e1f0,
        ];
        let mut cpu = Cpu::default();
        let block = abc_block();
        // v0 ABCD, v1 E, v2 W+K, v3 next E, v4..v7 message.
        cpu.vectors[0] = lanes(&initial[..4]);
        cpu.vectors[1] = u128::from(initial[4]);
        for i in 0..4 {
            cpu.vectors[4 + i] = lanes(&block[i * 4..]);
        }
        for group in 0..20 {
            let message = 4 + group % 4;
            let (k, op) = match group / 5 {
                0 => (0x5a827999u32, CryptoOp::Sha1c),
                1 => (0x6ed9eba1, CryptoOp::Sha1p),
                2 => (0x8f1bbcdc, CryptoOp::Sha1m),
                _ => (0xca62c1d6, CryptoOp::Sha1p),
            };
            cpu.vectors[2] = lanes32(std::array::from_fn(|e| {
                ((cpu.vectors[message] >> (32 * e)) as u32).wrapping_add(k)
            }));
            crypto(&mut cpu, CryptoOp::Sha1h, 3, 0, 0);
            crypto(&mut cpu, op, 0, 1, 2);
            cpu.vectors[1] = cpu.vectors[3];
            if group < 16 {
                let (m0, m1, m2, m3) = (
                    message,
                    4 + (group + 1) % 4,
                    4 + (group + 2) % 4,
                    4 + (group + 3) % 4,
                );
                crypto(&mut cpu, CryptoOp::Sha1su0, m0 as u8, m1 as u8, m2 as u8);
                crypto(&mut cpu, CryptoOp::Sha1su1, m0 as u8, m3 as u8, 0);
            }
        }
        let digest: Vec<u32> = (0..5)
            .map(|i| {
                let lane = if i < 4 {
                    (cpu.vectors[0] >> (32 * i)) as u32
                } else {
                    cpu.vectors[1] as u32
                };
                lane.wrapping_add(initial[i])
            })
            .collect();
        assert_eq!(
            digest,
            [0xa9993e36, 0x4706816a, 0xba3e2571, 0x7850c26c, 0x9cd0d89d]
        );
    }

    #[test]
    fn polynomial_multiply_and_register_shifts() {
        let mut cpu = Cpu::default();
        cpu.vectors[1] = 0x8000_0000_0000_0003;
        cpu.vectors[2] = 0x3;
        vec_poly_mul_long(&mut cpu, 64, false, 0, 1, 2);
        // (x^63 + x + 1)(x + 1) = x^64 + x^63 + x^2 + 1
        assert_eq!(cpu.vectors[0], (1u128 << 64) | (1 << 63) | 0b101);

        cpu.vectors[1] = lanes32([7, 0xffff_fff9, 0x4000_0000, 5]);
        cpu.vectors[2] = lanes32([0xff, 0xff, 1, 0xfe]);
        vec_int(&mut cpu, VecIntOp::Srshl, 32, true, false, 0, 1, 2);
        // 7 >> 1 rounds to 4; -7 >> 1 rounds to -3; 5 >> 2 rounds to 1.
        assert_eq!(cpu.vectors[0], lanes32([4, 0xffff_fffd, 0x8000_0000, 1]));
        cpu.fpsr = 0;
        vec_int(&mut cpu, VecIntOp::Sqshl, 32, true, false, 0, 1, 2);
        assert_eq!(cpu.vectors[0], lanes32([3, 0xffff_fffc, 0x7fff_ffff, 1]));
        assert_ne!(cpu.fpsr & FPSR_QC, 0);
    }

    #[test]
    fn count_permute_extract_reduce_and_shift() {
        let mut cpu = Cpu::default();
        cpu.vectors[1] = 0xffff_ffff_ffff_ffff_0000_0000_0003_0fff;
        cpu.vectors[0] = u128::MAX;
        vec_unary(&mut cpu, VecUnaryOp::Cnt, 8, false, false, 0, 1);
        // 64-bit form: upper half is zeroed.
        assert_eq!(cpu.vectors[0], 0x0000_0000_0000_0000_0000_0000_0002_0408);

        cpu.vectors[1] = lanes32([0, 1, 2, 3]);
        cpu.vectors[2] = lanes32([10, 11, 12, 13]);
        vec_permute(&mut cpu, VecPermuteOp::Zip2, 32, true, 0, 1, 2);
        assert_eq!(cpu.vectors[0], lanes32([2, 12, 3, 13]));
        vec_permute(&mut cpu, VecPermuteOp::Uzp1, 32, true, 0, 1, 2);
        assert_eq!(cpu.vectors[0], lanes32([0, 2, 10, 12]));
        vec_ext(&mut cpu, true, 0, 1, 2, 4);
        assert_eq!(cpu.vectors[0], lanes32([1, 2, 3, 10]));

        cpu.vectors[1] = lanes32([1, 2, 3, 4]);
        vec_across(&mut cpu, VecAcrossOp::AddV, 32, true, 0, 1);
        assert_eq!(cpu.vectors[0], 10);

        cpu.vectors[1] = lanes32([8, 16, 0xffff_fff8, 7]);
        vec_shift(&mut cpu, VecShiftOp::Ushr, 32, true, false, 0, 1, 3);
        assert_eq!(cpu.vectors[0], lanes32([1, 2, 0x1fff_ffff, 0]));
        vec_shift(&mut cpu, VecShiftOp::Sshr, 32, true, false, 0, 1, 3);
        assert_eq!(cpu.vectors[0], lanes32([1, 2, u32::MAX, 0]));
    }

    #[test]
    fn widening_multiply_uses_selected_half() {
        let mut cpu = Cpu::default();
        cpu.vectors[1] = lanes32([3, 0xffff_ffff, 5, 6]);
        cpu.vectors[2] = lanes32([7, 2, 11, 12]);
        vec_long(&mut cpu, VecLongOp::Mull, false, 32, false, 0, 1, 2);
        assert_eq!(cpu.vectors[0], 21 | (0x1_ffff_fffe_u128 << 64));
        vec_long(&mut cpu, VecLongOp::Mull, true, 32, false, 0, 1, 2);
        assert_eq!(cpu.vectors[0], 21 | (u128::from((-2i64) as u64) << 64));
        vec_long(&mut cpu, VecLongOp::Mull, false, 32, true, 0, 1, 2);
        assert_eq!(cpu.vectors[0], 55 | (72u128 << 64));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn structure_loads_deinterleave_and_stores_interleave() {
        let mut memory = Memory::new(0x20000);
        memory
            .map(0x1000, 0x1000, READ | WRITE, STATE_NORMAL, "data")
            .unwrap();
        let words: Vec<u8> = (0u32..8).flat_map(u32::to_le_bytes).collect();
        memory.write(0x1000, &words).unwrap();
        let mut view = memory.view();
        let mut cpu = Cpu::default();
        // LD2 {v31.4s, v0.4s}: register numbers wrap modulo 32.
        let moved =
            transfer_multiple(&mut cpu, &mut view, 0x1000, 31, 2, 2, 32, true, true).unwrap();
        assert_eq!(moved, 32);
        assert_eq!(cpu.vectors[31], lanes32([0, 2, 4, 6]));
        assert_eq!(cpu.vectors[0], lanes32([1, 3, 5, 7]));
        // ST2 writes the same interleaving back.
        transfer_multiple(&mut cpu, &mut view, 0x1100, 31, 2, 2, 32, true, false).unwrap();
        let mut out = [0u8; 32];
        memory.read(0x1100, &mut out).unwrap();
        assert_eq!(out.as_slice(), words.as_slice());
    }
}

#[cfg(test)]
mod fast_path_tests {
    use super::*;

    /// Raw values covering zeros, subnormals, normals near overflow,
    /// infinities and quiet and signaling NaNs of each size.
    fn samples(size: FpSize) -> Vec<u64> {
        match size {
            FpSize::S => [
                0.0f32, -0.0, 1.0, -1.5, 3.25e-39, -1.0e-45, 3.0e38, -3.4e38, 1.0e-20, 7.0,
            ]
            .iter()
            .map(|value| u64::from(value.to_bits()))
            .chain([
                0x7f80_0000,
                0xff80_0000,
                0x7fc0_0001,
                0x7f80_0001,
                0xffc0_1234,
            ])
            .collect(),
            FpSize::D => [
                0.0f64, -0.0, 1.0, -1.5, 2.0e-310, -5.0e-324, 1.7e308, -1.0e308, 1.0e-200, 7.0,
            ]
            .iter()
            .map(|value| value.to_bits())
            .chain([
                0x7ff0_0000_0000_0000,
                0xfff0_0000_0000_0000,
                0x7ff8_0000_0000_0001,
                0x7ff0_0000_0000_0001,
            ])
            .collect(),
        }
    }

    fn cpu_with(fpcr: u32, values: &[u64]) -> Cpu {
        let mut cpu = Cpu {
            fpcr,
            ..Cpu::default()
        };
        for (index, value) in values.iter().enumerate() {
            cpu.vectors[index + 1] = u128::from(*value) | 0xdead << 100;
        }
        cpu
    }

    #[test]
    fn fast_paths_match_the_exact_helpers() {
        let binary = [
            FpBinaryOp::Add,
            FpBinaryOp::Sub,
            FpBinaryOp::Mul,
            FpBinaryOp::Div,
            FpBinaryOp::Max,
        ];
        let ternary = [
            FpTernaryOp::MulAdd,
            FpTernaryOp::MulSub,
            FpTernaryOp::NegMulAdd,
            FpTernaryOp::NegMulSub,
        ];
        for fpcr in [0, FPCR_FZ, FPCR_DN] {
            for size in [FpSize::S, FpSize::D] {
                let values = samples(size);
                for &a in &values {
                    for &b in &values {
                        for op in binary {
                            let (mut fast, mut exact) =
                                (cpu_with(fpcr, &[a, b]), cpu_with(fpcr, &[a, b]));
                            fp_binary(&mut fast, op, size, 0, 1, 2).unwrap();
                            fp_binary_exact(&mut exact, op, size, 0, 1, 2).unwrap();
                            assert_eq!(fast, exact, "{op:?} {size:?} {a:#x} {b:#x}");
                        }
                        for signaling in [false, true] {
                            let (mut fast, mut exact) =
                                (cpu_with(fpcr, &[a, b]), cpu_with(fpcr, &[a, b]));
                            fp_compare(&mut fast, size, 1, Some(2), signaling).unwrap();
                            fp_compare_exact(&mut exact, size, 1, Some(2), signaling).unwrap();
                            assert_eq!(fast, exact, "compare {size:?} {a:#x} {b:#x}");
                        }
                        for &c in values.iter().step_by(3) {
                            for op in ternary {
                                let (mut fast, mut exact) =
                                    (cpu_with(fpcr, &[a, b, c]), cpu_with(fpcr, &[a, b, c]));
                                fp_ternary(&mut fast, op, size, 0, 1, 2, 3).unwrap();
                                fp_ternary_exact(&mut exact, op, size, 0, 1, 2, 3).unwrap();
                                assert_eq!(fast, exact, "{op:?} {size:?} {a:#x} {b:#x} {c:#x}");
                            }
                        }
                    }
                }
            }
        }
    }
}
