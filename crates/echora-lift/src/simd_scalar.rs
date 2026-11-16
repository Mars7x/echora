//! Native code for scalar float moves, conversions, square roots, rounding
//! and conditional compares (`EmitOptions::inline_fp_scalar`), in place of
//! their helper calls.
//!
//! FMOV between general and FP registers is a bit move (`fp_move_*`).
//! SCVTF/UCVTF (from a general or an FP register) are the helper's own host
//! conversion once the FPCR is one the helpers accept. The others run
//! natively when the operand is normal or zero and the result is what the
//! helper gives with no flag raised, and call the helper otherwise.

use crate::simd::{
    FP_MODE, FpSize, FpUnaryOp, Rounding, SimdOp, emit_call, emit_inline, float_kit,
};

thread_local! {
    /// `EmitOptions::inline_fp_scalar` of the emission on this thread.
    static ON: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Sets whether scalar float moves and conversions are inlined; returns
/// the previous setting.
pub(crate) fn set_inline_fp_scalar(on: bool) -> bool {
    ON.replace(on)
}

/// General register `rn` read as `bits` (`Cpu::read_reg`, the zero
/// register as 0).
fn general(rn: u8, bits: u8) -> String {
    if rn == 31 {
        "0u64".to_owned()
    } else if bits == 32 {
        format!("(m.cpu.regs[{rn}] & 0xffff_ffffu64)")
    } else {
        format!("m.cpu.regs[{rn}]")
    }
}

/// `Cpu::write_reg(rd, bits, false, value)` of an already masked value.
fn write_general(rd: u8, value: &str) -> String {
    if rd == 31 {
        String::new()
    } else {
        format!("m.cpu.regs[{rd}] = {value};")
    }
}

/// `write_fp`: the element, every bit above it cleared.
fn write_fp(rd: u8, size: FpSize, value: &str) -> String {
    match size {
        FpSize::S => format!("m.cpu.vectors[{rd}] = u128::from(({value}) as u32);"),
        FpSize::D => format!("m.cpu.vectors[{rd}] = u128::from({value});"),
    }
}

/// `int_to_fp_bits` of the u64 `value`.
fn int_to_fp(value: &str, int_bits: u32, signed: bool, size: FpSize) -> String {
    let source = match (signed, int_bits) {
        (true, 32) => format!("({value} as u32 as i32)"),
        (true, _) => format!("({value} as i64)"),
        (false, 32) => format!("({value} as u32)"),
        (false, _) => value.to_owned(),
    };
    match size {
        FpSize::S => format!("u64::from(({source} as f32).to_bits())"),
        FpSize::D => format!("({source} as f64).to_bits()"),
    }
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

/// Native code for a scalar float move, conversion, square root, rounding
/// or conditional compare, or `None` to keep its helper call (or another
/// inlined form). A fallback helper call goes on a line of its own: register
/// locals are stored before and reloaded after the line that calls it, which
/// would undo a native write on the same line.
pub(crate) fn emit_fp_scalar(op: &SimdOp) -> Option<String> {
    if !ON.get() {
        return None;
    }
    let call = || emit_call(op);
    match *op {
        SimdOp::FpMoveToGeneral {
            rd,
            rn,
            bits,
            upper,
        } => {
            let value = if upper {
                format!("((m.cpu.vectors[{rn}] >> 64) as u64)")
            } else if bits == 32 {
                format!("(m.cpu.vectors[{rn}] as u32 as u64)")
            } else {
                format!("(m.cpu.vectors[{rn}] as u64)")
            };
            let value = if bits == 32 && upper {
                format!("({value} & 0xffff_ffffu64)")
            } else {
                value
            };
            Some(write_general(rd, &value))
        }
        SimdOp::FpMoveFromGeneral {
            rd,
            rn,
            bits,
            upper,
        } => {
            let value = general(rn, bits);
            Some(if upper {
                format!(
                    "{{ let value = {value}; let vector = m.cpu.vectors[{rd}]; m.cpu.vectors[{rd}] = (vector & u128::from(u64::MAX)) | (u128::from(value) << 64); }}"
                )
            } else {
                format!("m.cpu.vectors[{rd}] = u128::from({value});")
            })
        }
        // The helper's only check is the FPCR (`mode`): then the host
        // conversion, rounding to nearest.
        SimdOp::IntToFp {
            rd,
            rn,
            gp_bits,
            size,
            signed,
        } => {
            let value = int_to_fp("value", u32::from(gp_bits), signed, size);
            Some(format!(
                "{{ let value = {}; if {FP_MODE} {{ {} }} else {{\n{}\n}} }}",
                general(rn, gp_bits),
                write_fp(rd, size, &value),
                call()?
            ))
        }
        SimdOp::SimdIntToFp {
            rd,
            rn,
            size,
            signed,
        } => {
            let (read, bits) = match size {
                FpSize::S => (format!("(m.cpu.vectors[{rn}] as u32 as u64)"), 32),
                FpSize::D => (format!("(m.cpu.vectors[{rn}] as u64)"), 64),
            };
            let value = int_to_fp("value", bits, signed, size);
            Some(format!(
                "{{ let value = {read}; if {FP_MODE} {{ {} }} else {{\n{}\n}} }}",
                write_fp(rd, size, &value),
                call()?
            ))
        }
        // A normal or zero operand whose rounded value fits: no IOC, and
        // the helper models no IXC for conversions.
        SimdOp::FpToInt {
            rd,
            rn,
            gp_bits,
            size,
            signed,
            rounding,
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
            let method = rounding_method(rounding);
            Some(format!(
                "{{ let x = m.cpu.vectors[{rn}] as {int}; {kit} let rounded = f64::from({float}::from_bits(x)).{method}(); \
                 if {FP_MODE} && usable(x) && rounded >= {min} && rounded {upper} {max} {{ let value = {convert}; {} }} else {{\n{}\n}} }}",
                write_general(rd, "value"),
                call()?
            ))
        }
        // FCVT between single and double: a normal or zero operand whose
        // converted value is normal or zero (no flush, no overflow).
        SimdOp::FpConvert {
            rd,
            rn,
            from: from @ (FpSize::S | FpSize::D),
            to,
        } if from != to => {
            let (_, int, _, kit) = float_kit(from);
            let (_, _, _, target_kit) = float_kit(to);
            let (converted, target_int) = match from {
                FpSize::S => ("f64::from(f32::from_bits(x)).to_bits()", "u64"),
                FpSize::D => ("(f64::from_bits(x) as f32).to_bits()", "u32"),
            };
            // The target's tests, renamed.
            let target_kit = target_kit
                .replace("let normal =", "let target_normal =")
                .replace("let zero =", "let target_zero =")
                .replace("let usable = |value", "let target_usable = |value")
                .replace(
                    "normal(value) | zero(value)",
                    "target_normal(value) | target_zero(value)",
                )
                .replace("let sign =", "let target_sign =")
                .replace("let _ = sign;", "let _ = target_sign;");
            Some(format!(
                "{{ let x = m.cpu.vectors[{rn}] as {int}; {kit} {target_kit} let converted: {target_int} = {converted}; \
                 if {FP_MODE} && usable(x) && target_usable(converted) {{ {} }} else {{\n{}\n}} }}",
                write_fp(rd, to, "converted"),
                call()?
            ))
        }
        SimdOp::FpUnary {
            op: unary @ (FpUnaryOp::Sqrt | FpUnaryOp::Round(_) | FpUnaryOp::RoundExact),
            size,
            rd,
            rn,
        } => {
            let (float, int, _, kit) = float_kit(size);
            let (native, value) = match unary {
                FpUnaryOp::Sqrt => (
                    "usable(x) && ((x & sign) == 0 || zero(x))",
                    "f.sqrt()".to_owned(),
                ),
                FpUnaryOp::Round(rounding) => {
                    ("usable(x)", format!("f.{}()", rounding_method(rounding)))
                }
                // FRINTX: to nearest (the FPCR's mode here); the helper
                // models no IXC.
                _ => ("usable(x)", "f.round_ties_even()".to_owned()),
            };
            Some(format!(
                "{{ let x = m.cpu.vectors[{rn}] as {int}; {kit} let f = {float}::from_bits(x); \
                 if {FP_MODE} && {native} {{ {} }} else {{\n{}\n}} }}",
                write_fp(rd, size, &format!("{value}.to_bits()")),
                call()?
            ))
        }
        // FCCMP/FCCMPE: the immediate flags when the condition fails, else
        // the compare (inlined as FCMP is).
        SimdOp::FpConditionalCompare {
            size,
            rn,
            rm,
            condition,
            nzcv,
            signaling,
        } => {
            let compare = emit_inline(&SimdOp::FpCompare {
                size,
                rn,
                rm: Some(rm),
                signaling,
            })?;
            Some(format!(
                "if m.cpu.condition({condition}) {{ {compare} }} else {{ m.cpu.nzcv = {}; }}",
                nzcv & 15
            ))
        }
        _ => None,
    }
}
