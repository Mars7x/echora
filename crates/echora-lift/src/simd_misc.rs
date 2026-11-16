//! Native code for the remaining vector operations the hot code calls
//! helpers for (`EmitOptions::inline_vector_misc`): TBL/TBX, the lane,
//! replicate and multiple-structure loads and stores, FABD and FMULX,
//! FMAXV/FMINV/FMAXNMV/FMINNMV, and the conversions with fraction bits or
//! a directed rounding, and vector SCVTF/UCVTF.
//!
//! The memory forms make the helper's own single `GuestView::read` or
//! `write` of the whole transfer (so the same access, the same fault and
//! nothing written on a fault), into a buffer on the stack instead of a
//! vector, and place the lanes as the helper does. The float forms run a
//! lane natively when it is the helper's result with no flag raised and
//! otherwise through the exact single-lane helper (or the whole helper).

use crate::simd::{FP_MODE, FpSize, Rounding, SimdOp, VecFpOp, VecFpUnaryOp, emit_call, float_kit};

thread_local! {
    /// `EmitOptions::inline_vector_misc` of the emission on this thread.
    static ON: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Sets whether the remaining vector operations are inlined; returns the
/// previous setting.
pub(crate) fn set_inline_vector_misc(on: bool) -> bool {
    ON.replace(on)
}

fn lane_mask(bits: u32) -> u128 {
    if bits >= 64 {
        u128::from(u64::MAX)
    } else {
        (1u128 << bits) - 1
    }
}

/// 2^`exponent` as an f64 literal (exact).
fn power_of_two(exponent: i32) -> String {
    format!(
        "f64::from_bits({:#x}u64)",
        ((1023 + i64::from(exponent)) as u64) << 52
    )
}

fn rounding_method(rounding: Rounding) -> &'static str {
    match rounding {
        Rounding::Nearest => "round_ties_even",
        Rounding::Plus => "ceil",
        Rounding::Minus => "floor",
        Rounding::Zero => "trunc",
        Rounding::Away => "round",
    }
}

/// The `u64` lane value of bytes `from..from + esize` of `buffer`.
fn buffer_lane(from: usize, esize: usize) -> String {
    let bytes: Vec<String> = (0..8)
        .map(|byte| {
            if byte < esize {
                format!("buffer[{}]", from + byte)
            } else {
                "0".to_owned()
            }
        })
        .collect();
    format!("u64::from_le_bytes([{}])", bytes.join(", "))
}

/// The expression a structure load or store's helper call becomes (the
/// bytes moved, as `structure_transfer` expects), or `None` to keep the
/// call. `address` is the computed address.
pub(crate) fn emit_transfer(op: &SimdOp) -> Option<String> {
    if !ON.get() {
        return None;
    }
    match *op {
        SimdOp::TransferSingle {
            rt,
            registers,
            bits,
            index,
            load,
            ..
        } => {
            let esize = usize::from(bits / 8);
            let total = usize::from(registers) * esize;
            let shift = u32::from(index) * u32::from(bits);
            let mask = lane_mask(u32::from(bits)) << shift;
            let mut body = String::new();
            for r in 0..usize::from(registers) {
                let register = (usize::from(rt) + r) % 32;
                if load {
                    body.push_str(&format!(
                        "{{ let value = {}; let vector = m.cpu.vectors[{register}]; m.cpu.vectors[{register}] = (vector & !{mask:#x}u128) | (u128::from(value) << {shift}); }} ",
                        buffer_lane(r * esize, esize)
                    ));
                } else {
                    body.push_str(&format!(
                        "{{ let lane = ((m.cpu.vectors[{register}] >> {shift}) as u64).to_le_bytes(); buffer[{}..{}].copy_from_slice(&lane[..{esize}]); }} ",
                        r * esize,
                        (r + 1) * esize
                    ));
                }
            }
            Some(if load {
                format!(
                    "{{ let mut buffer = [0u8; {total}]; m.memory.read(address, &mut buffer)?; {body}{total}u64 }}"
                )
            } else {
                format!(
                    "{{ let mut buffer = [0u8; {total}]; {body}m.memory.write(address, &buffer)?; {total}u64 }}"
                )
            })
        }
        SimdOp::LoadReplicate {
            rt,
            registers,
            bits,
            q,
            ..
        } => {
            let esize = usize::from(bits / 8);
            let total = usize::from(registers) * esize;
            let lanes = if q { 128 } else { 64 } / u32::from(bits);
            let replicate = (0..lanes).fold(0u128, |value, lane| {
                value | 1u128 << (lane * u32::from(bits))
            });
            let mut body = String::new();
            for r in 0..usize::from(registers) {
                let register = (usize::from(rt) + r) % 32;
                body.push_str(&format!(
                    "m.cpu.vectors[{register}] = u128::from({}) * {replicate:#x}u128; ",
                    buffer_lane(r * esize, esize)
                ));
            }
            Some(format!(
                "{{ let mut buffer = [0u8; {total}]; m.memory.read(address, &mut buffer)?; {body}{total}u64 }}"
            ))
        }
        SimdOp::TransferMultiple {
            rt,
            registers,
            structure,
            bits,
            q,
            load,
            ..
        } => {
            let esize = usize::from(bits / 8);
            let per_register = if q { 16 } else { 8 };
            let lanes = per_register / esize;
            let total = usize::from(registers) * per_register;
            let offset = |r: usize, lane: usize| {
                if structure == 1 {
                    r * per_register + lane * esize
                } else {
                    (lane * usize::from(structure) + r) * esize
                }
            };
            let mut body = String::new();
            if load {
                // Every register's value first, then the writes (as the
                // helper: a register both read for the address and loaded
                // is not, here, as the address is computed before).
                for r in 0..usize::from(registers) {
                    let parts: Vec<String> = (0..lanes)
                        .map(|lane| {
                            format!(
                                "(u128::from({}) << {})",
                                buffer_lane(offset(r, lane), esize),
                                lane * esize * 8
                            )
                        })
                        .collect();
                    body.push_str(&format!("let value{r}: u128 = {}; ", parts.join(" | ")));
                }
                for r in 0..usize::from(registers) {
                    let register = (usize::from(rt) + r) % 32;
                    body.push_str(&format!("m.cpu.vectors[{register}] = value{r}; "));
                }
                Some(format!(
                    "{{ let mut buffer = [0u8; {total}]; m.memory.read(address, &mut buffer)?; {body}{total}u64 }}"
                ))
            } else {
                for r in 0..usize::from(registers) {
                    let register = (usize::from(rt) + r) % 32;
                    body.push_str(&format!("{{ let value = m.cpu.vectors[{register}]; "));
                    for lane in 0..lanes {
                        let at = offset(r, lane);
                        body.push_str(&format!(
                            "buffer[{at}..{}].copy_from_slice(&((value >> {}) as u64).to_le_bytes()[..{esize}]); ",
                            at + esize,
                            lane * esize * 8
                        ));
                    }
                    body.push_str("} ");
                }
                Some(format!(
                    "{{ let mut buffer = [0u8; {total}]; {body}m.memory.write(address, &buffer)?; {total}u64 }}"
                ))
            }
        }
        _ => None,
    }
}

/// Native code for TBL/TBX, FABD, FMULX, FMAXV and its kin, and the
/// conversions (`inline_vector_misc`), or `None`.
pub(crate) fn emit_misc(op: &SimdOp) -> Option<String> {
    if !ON.get() {
        return None;
    }
    let p = "echora_guest::simd";
    match *op {
        SimdOp::VecTable {
            q,
            extend,
            rd,
            rn,
            regs,
            rm,
        } => {
            let table: Vec<String> = (0..regs)
                .map(|r| format!("m.cpu.vectors[{}]", (usize::from(rn) + usize::from(r)) % 32))
                .collect();
            let limit = usize::from(regs) * 16;
            let count = if q { 16 } else { 8 };
            let otherwise = if extend {
                "(old >> (lane * 8)) as u8"
            } else {
                "0u8"
            };
            Some(format!(
                "{{ let table: [u128; {regs}] = [{}]; let indices = m.cpu.vectors[{rm}]; let old = m.cpu.vectors[{rd}]; let mut out = 0u128; \
                 for lane in 0..{count}u32 {{ let index = (indices >> (lane * 8)) as u8 as usize; \
                 let byte: u8 = if index < {limit} {{ (table[index / 16] >> ((index % 16) * 8)) as u8 }} else {{ {otherwise} }}; \
                 out |= u128::from(byte) << (lane * 8); }} \
                 m.cpu.vectors[{rd}] = out; }}",
                table.join(", ")
            ))
        }
        SimdOp::VecFp {
            op: fp_op @ (VecFpOp::AbsDiff | VecFpOp::MulExtended),
            size,
            q,
            scalar,
            rd,
            rn,
            rm,
        } => lanewise_binary(fp_op, size, q, scalar, rd, rn, rm, None, &emit_call(op)?),
        SimdOp::VecFpElement {
            op: fp_op @ VecFpOp::MulExtended,
            size,
            q,
            scalar,
            rd,
            rn,
            rm,
            index,
        } => lanewise_binary(
            fp_op,
            size,
            q,
            scalar,
            rd,
            rn,
            rm,
            Some(index),
            &emit_call(op)?,
        ),
        // FMAXV/FMINV/FMAXNMV/FMINNMV (four singles): with every lane
        // normal or zero the NaN-handling forms agree, and nothing is
        // raised; the tree of the helper.
        SimdOp::VecFpAcross {
            op: fp_op @ (VecFpOp::Max | VecFpOp::Min | VecFpOp::MaxNum | VecFpOp::MinNum),
            rd,
            rn,
        } => {
            let (_, _, _, kit) = float_kit(FpSize::S);
            let max = matches!(fp_op, VecFpOp::Max | VecFpOp::MaxNum);
            let pick = if max {
                "|x: u32, y: u32| if zero(x) && zero(y) { x & y } else if f32::from_bits(x) > f32::from_bits(y) { x } else { y }"
            } else {
                "|x: u32, y: u32| if zero(x) && zero(y) { x | y } else if f32::from_bits(x) < f32::from_bits(y) { x } else { y }"
            };
            Some(format!(
                "{{ let vn = m.cpu.vectors[{rn}]; {kit} let lane = |i: u32| (vn >> (i * 32)) as u32; \
                 if {FP_MODE} && usable(lane(0)) && usable(lane(1)) && usable(lane(2)) && usable(lane(3)) {{ \
                 let pick = {pick}; m.cpu.vectors[{rd}] = u128::from(pick(pick(lane(0), lane(1)), pick(lane(2), lane(3)))); \
                 }} else {{\n{}\n}} }}",
                emit_call(op)?
            ))
        }
        // FCVTZS/FCVTZU (scalar, fixed point) to a general register: as
        // FCVTZS, the value scaled by 2^fbits first (exactly).
        SimdOp::FpToFixed {
            rd,
            rn,
            gp_bits,
            size,
            signed,
            fbits,
        } => {
            let (float, int, _, kit) = float_kit(size);
            let (min, max) = match (signed, gp_bits) {
                (true, 32) => ("-2147483648.0", "2147483647.0"),
                (true, _) => ("-9223372036854775808.0", "9223372036854775808.0"),
                (false, 32) => ("0.0", "4294967295.0"),
                (false, _) => ("0.0", "18446744073709551616.0"),
            };
            let upper = if gp_bits == 64 { "<" } else { "<=" };
            let convert = match (signed, gp_bits) {
                (true, 32) => "rounded as i32 as u32 as u64",
                (true, _) => "rounded as i64 as u64",
                (false, 32) => "rounded as u32 as u64",
                (false, _) => "rounded as u64",
            };
            let write = if rd == 31 {
                String::new()
            } else {
                format!("m.cpu.regs[{rd}] = value;")
            };
            Some(format!(
                "{{ let x = m.cpu.vectors[{rn}] as {int}; {kit} let rounded = (f64::from({float}::from_bits(x)) * {}).trunc(); \
                 if {FP_MODE} && usable(x) && rounded >= {min} && rounded {upper} {max} {{ let value = {convert}; {write} }} else {{\n{}\n}} }}",
                power_of_two(i32::from(fbits)),
                emit_call(op)?
            ))
        }
        // SCVTF/UCVTF (scalar, fixed point) from a general register: the
        // helper's own arithmetic once the FPCR is accepted.
        SimdOp::IntToFpFixed {
            rd,
            rn,
            gp_bits,
            size,
            signed,
            fbits,
        } => {
            let value = if rn == 31 {
                "0u64".to_owned()
            } else if gp_bits == 32 {
                format!("(m.cpu.regs[{rn}] & 0xffff_ffffu64)")
            } else {
                format!("m.cpu.regs[{rn}]")
            };
            let wide = match (signed, gp_bits) {
                (true, 32) => "f64::from(value as u32 as i32)",
                (true, _) => "value as i64 as f64",
                (false, 32) => "f64::from(value as u32)",
                (false, _) => "value as f64",
            };
            let bits = match size {
                FpSize::S => "u64::from((scaled as f32).to_bits())",
                FpSize::D => "scaled.to_bits()",
            };
            let write = match size {
                FpSize::S => format!("m.cpu.vectors[{rd}] = u128::from(bits as u32);"),
                FpSize::D => format!("m.cpu.vectors[{rd}] = u128::from(bits);"),
            };
            Some(format!(
                "{{ let value = {value}; if {FP_MODE} {{ let scaled = ({wide}) * {}; let bits: u64 = {bits}; {write} }} else {{\n{}\n}} }}",
                power_of_two(-i32::from(fbits)),
                emit_call(op)?
            ))
        }
        // Vector SCVTF/UCVTF (any fraction bits): `integer_to_fp_lane`
        // for every lane once the FPCR is accepted.
        SimdOp::VecFpUnary {
            op: unary @ (VecFpUnaryOp::FromSigned | VecFpUnaryOp::FromUnsigned),
            size,
            q,
            scalar,
            rd,
            rn,
            fbits,
        } if scalar || q || size == FpSize::S => {
            let signed = unary == VecFpUnaryOp::FromSigned;
            let (width, lanes) = lanes_of(size, q, scalar);
            let scale = power_of_two(-i32::from(fbits));
            let value = match (size, fbits == 0, signed) {
                (FpSize::S, true, true) => {
                    "u64::from((a as u32 as i32 as f32).to_bits())".to_owned()
                }
                (FpSize::S, true, false) => "u64::from((a as u32 as f32).to_bits())".to_owned(),
                (FpSize::S, false, true) => {
                    format!("u64::from(((f64::from(a as u32 as i32) * {scale}) as f32).to_bits())")
                }
                (FpSize::S, false, false) => {
                    format!("u64::from(((f64::from(a as u32) * {scale}) as f32).to_bits())")
                }
                (FpSize::D, true, true) => "(a as i64 as f64).to_bits()".to_owned(),
                (FpSize::D, true, false) => "(a as f64).to_bits()".to_owned(),
                (FpSize::D, false, true) => format!("((a as i64 as f64) * {scale}).to_bits()"),
                (FpSize::D, false, false) => format!("((a as f64) * {scale}).to_bits()"),
            };
            let mask = lane_mask(width);
            Some(format!(
                "{{ let vn = m.cpu.vectors[{rn}]; if {FP_MODE} {{ let mut out = 0u128; \
                 for lane in 0..{lanes}u32 {{ let a = ((vn >> (lane * {width})) as u64) & {:#x}u64; let value: u64 = {value}; out |= (u128::from(value) & {mask:#x}u128) << (lane * {width}); }} \
                 m.cpu.vectors[{rd}] = out; }} else {{\n{}\n}} }}",
                mask as u64,
                emit_call(op)?
            ))
        }
        // Vector FCVT*S/U in a directed rounding or with fraction bits
        // (`inline_vector_unary` takes toward zero without): per lane, a
        // normal or zero lane whose scaled, rounded value fits; any other
        // lane through the helper's scalar form, in Vd (written last).
        SimdOp::VecFpUnary {
            op: unary @ (VecFpUnaryOp::ToSigned(rounding) | VecFpUnaryOp::ToUnsigned(rounding)),
            size,
            q,
            scalar,
            rd,
            rn,
            fbits,
        } if (scalar || q || size == FpSize::S) && (fbits != 0 || rounding != Rounding::Zero) => {
            let signed = matches!(unary, VecFpUnaryOp::ToSigned(_));
            let (float, int, width, kit) = float_kit(size);
            let (_, lanes) = lanes_of(size, q, scalar);
            let (min, max, upper, convert) = match (signed, width) {
                (true, 32) => (
                    "-2147483648.0",
                    "2147483647.0",
                    "<=",
                    "rounded as i32 as u32 as u64",
                ),
                (true, _) => (
                    "-9223372036854775808.0",
                    "9223372036854775808.0",
                    "<",
                    "rounded as i64 as u64",
                ),
                (false, 32) => ("0.0", "4294967295.0", "<=", "rounded as u32 as u64"),
                (false, _) => ("0.0", "18446744073709551616.0", "<", "rounded as u64"),
            };
            let path = match unary {
                VecFpUnaryOp::ToSigned(r) => {
                    format!("VecFpUnaryOp::ToSigned(echora_guest::simd::Rounding::{r:?})")
                }
                _ => {
                    format!("VecFpUnaryOp::ToUnsigned(echora_guest::simd::Rounding::{rounding:?})")
                }
            };
            Some(format!(
                "{{ let vn = m.cpu.vectors[{rn}]; {kit} if {FP_MODE} {{ let mut out = 0u128; \
                 for lane in 0..{lanes}u32 {{ let x = (vn >> (lane * {width})) as {int}; \
                 let rounded = (f64::from({float}::from_bits(x)) * {}).{}(); \
                 let raw: u64 = if usable(x) && rounded >= {min} && rounded {upper} {max} {{ {convert} }} else {{ \
                 m.cpu.vectors[{rd}] = u128::from(x); \
                 {p}::vec_fp_unary(&mut m.cpu, {p}::{path}, {p}::FpSize::{size:?}, false, true, {rd}, {rd}, {fbits})?; \
                 m.cpu.vectors[{rd}] as {int} as u64 }}; \
                 out |= u128::from(raw as {int}) << (lane * {width}); }} \
                 m.cpu.vectors[{rd}] = out; }} else {{\n{}\n}} }}",
                power_of_two(i32::from(fbits)),
                rounding_method(rounding),
                emit_call(op)?
            ))
        }
        _ => None,
    }
}

/// Lane width and count of a float form.
fn lanes_of(size: FpSize, q: bool, scalar: bool) -> (u32, u32) {
    let width = if size == FpSize::S { 32 } else { 64 };
    let lanes = if scalar {
        1
    } else if q {
        128 / width
    } else {
        64 / width
    };
    (width, lanes)
}

/// FABD or FMULX per lane: natively for normal or zero operands whose
/// result is normal (or an exact zero), under the FPCR the helpers accept
/// and FPSR.IXC already set, as the inlined arithmetic; any other lane
/// through the exact single-lane helper.
#[allow(clippy::too_many_arguments)]
fn lanewise_binary(
    fp_op: VecFpOp,
    size: FpSize,
    q: bool,
    scalar: bool,
    rd: u8,
    rn: u8,
    rm: u8,
    element: Option<u8>,
    call: &str,
) -> Option<String> {
    if !(scalar || q || size == FpSize::S) {
        return None;
    }
    let p = "echora_guest::simd";
    let (float, int, width, kit) = float_kit(size);
    let (_, lanes) = lanes_of(size, q, scalar);
    let (value, exact) = match fp_op {
        // |x - y|: a zero difference is exact (equal operands).
        VecFpOp::AbsDiff => (
            format!("({float}::from_bits(x) - {float}::from_bits(y)).to_bits() & !sign"),
            "true",
        ),
        // FMULX of normal or zero operands is FMUL (no infinity).
        _ => (
            format!("({float}::from_bits(x) * {float}::from_bits(y)).to_bits()"),
            "zero(x) | zero(y)",
        ),
    };
    let y = match element {
        Some(index) => format!(
            "(m.cpu.vectors[{rm}] >> {}) as {int}",
            u32::from(index) * width
        ),
        None => format!("(vm >> (lane * {width})) as {int}"),
    };
    Some(format!(
        "{{ let vn = m.cpu.vectors[{rn}]; let vm = m.cpu.vectors[{rm}]; {kit} \
         if {FP_MODE} && m.cpu.fpsr & {p}::FPSR_IXC != 0 {{ let mut out = 0u128; \
         for lane in 0..{lanes}u32 {{ let x = (vn >> (lane * {width})) as {int}; let y = {y}; let raw: {int} = {value}; \
         let raw = if usable(x) & usable(y) & (normal(raw) | (zero(raw) & ({exact}))) {{ raw }} else {{ \
         let exact = {p}::vec_fp_lane(&mut m.cpu, {p}::VecFpOp::{fp_op:?}, {p}::FpSize::{size:?}, x as u64, y as u64, 0)?; exact as {int} }}; \
         out |= u128::from(raw) << (lane * {width}); }} \
         m.cpu.vectors[{rd}] = out; }} else {{\n{call}\n}} }}"
    ))
}
