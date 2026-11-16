//! Native code for vector integer operations, element moves and float
//! pairwise operations (`EmitOptions::inline_vector_int`,
//! `EmitOptions::inline_vector_pairwise`), in place of their helper calls.
//!
//! The integer forms are the helpers' own lane functions (`int_lane`,
//! `shift_register_lane`, `vec_unary`, `vec_shift`, `vec_long`,
//! `vec_narrow`, ...) written out with the lane width, shift and lane
//! layout fixed at lift time: the same arithmetic (i128 where the helper
//! widens), the same saturation and FPSR.QC, the same zeroing of the
//! unwritten part of Vd. They raise nothing else and never trap. Float
//! pairwise lanes run natively under the inlined float operations' rules
//! and otherwise through the exact single-lane helper.

use crate::simd::{
    FP_MODE, FpSize, SimdOp, VecAcrossOp, VecFpOp, VecIntOp, VecLongOp, VecNarrowOp, VecShiftOp,
    VecUnaryOp, float_kit,
};

thread_local! {
    /// `EmitOptions::inline_vector_int` of the emission on this thread.
    static INT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// `EmitOptions::inline_vector_pairwise` of the emission on this thread.
    static PAIRWISE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Sets whether vector integer operations and moves are inlined; returns
/// the previous setting.
pub(crate) fn set_inline_vector_int(on: bool) -> bool {
    INT.replace(on)
}

/// Sets whether float pairwise operations are inlined; returns the
/// previous setting.
pub(crate) fn set_inline_vector_pairwise(on: bool) -> bool {
    PAIRWISE.replace(on)
}

const QC: &str = "m.cpu.fpsr = m.cpu.fpsr | echora_guest::simd::FPSR_QC;";

/// A `bits`-wide lane's mask as a u64 literal.
fn mask(bits: u32) -> String {
    if bits >= 64 {
        "u64::MAX".to_owned()
    } else {
        format!("{:#x}u64", (1u64 << bits) - 1)
    }
}

/// `var` (a u64 lane) sign-extended from `bits`, as an i64.
fn sext(var: &str, bits: u32) -> String {
    if bits >= 64 {
        format!("({var} as i64)")
    } else {
        format!("((({var} << {s}) as i64) >> {s})", s = 64 - bits)
    }
}

/// Lane `lane` (an expression) of the u128 `vector`, `bits` wide.
fn get(vector: &str, bits: u32, lane: &str) -> String {
    format!(
        "((({vector} >> (({lane}) * {bits})) as u64) & {})",
        mask(bits)
    )
}

fn lane_count(bits: u32, q: bool, scalar: bool) -> u32 {
    if scalar {
        1
    } else if q {
        128 / bits
    } else {
        64 / bits
    }
}

/// `out` with the bits above the written part cleared (`finish_vector`).
fn finish(bits: u32, q: bool, scalar: bool) -> String {
    if scalar {
        format!("out & u128::from({})", mask(bits))
    } else if q {
        "out".to_owned()
    } else {
        "out & u128::from(u64::MAX)".to_owned()
    }
}

/// `saturate_signed` of the i128 `value` to `bits`.
fn saturate_signed(value: &str, bits: u32) -> String {
    let max = (1i128 << (bits - 1)) - 1;
    let min = -(1i128 << (bits - 1));
    format!(
        "{{ let value: i128 = {value}; let clamped = value.clamp({min}i128, {max}i128); saturated |= clamped != value; clamped as u64 & {} }}",
        mask(bits)
    )
}

/// `saturate_unsigned` of the i128 `value` to `bits`.
fn saturate_unsigned(value: &str, bits: u32) -> String {
    let max = if bits >= 64 {
        i128::from(u64::MAX)
    } else {
        (1i128 << bits) - 1
    };
    format!(
        "{{ let value: i128 = {value}; let clamped = value.clamp(0i128, {max}i128); saturated |= clamped != value; clamped as u64 }}"
    )
}

/// All-ones in the lane when `condition` holds.
fn flag(condition: &str, bits: u32) -> String {
    format!("if {condition} {{ {} }} else {{ 0 }}", mask(bits))
}

/// `int_lane` of the u64 lanes `a`, `b` and `d` (`saturated` set on
/// saturation): the expression and whether the operation can saturate.
fn int_lane(op: VecIntOp, bits: u32) -> Option<(String, bool)> {
    let m = mask(bits);
    let (sa, sb) = (sext("a", bits), sext("b", bits));
    let (wa, wb) = (format!("i128::from({sa})"), format!("i128::from({sb})"));
    let (ua, ub) = ("i128::from(a)", "i128::from(b)");
    Some(match op {
        VecIntOp::Add => ("a.wrapping_add(b)".to_owned(), false),
        VecIntOp::Sub => ("a.wrapping_sub(b)".to_owned(), false),
        VecIntOp::Mul => ("a.wrapping_mul(b)".to_owned(), false),
        VecIntOp::CmEq => (flag("a == b", bits), false),
        VecIntOp::CmTst => (flag("a & b != 0", bits), false),
        VecIntOp::CmGt => (flag(&format!("{sa} > {sb}"), bits), false),
        VecIntOp::CmGe => (flag(&format!("{sa} >= {sb}"), bits), false),
        VecIntOp::CmHi => (flag("a > b", bits), false),
        VecIntOp::CmHs => (flag("a >= b", bits), false),
        VecIntOp::Sshl | VecIntOp::Ushl => {
            // `shift_lane` by the signed low byte of b.
            let right = if op == VecIntOp::Sshl {
                format!(
                    "{{ let wide = {sa}; let shifted = if right >= {bits} {{ wide >> {} }} else {{ wide >> right }}; (shifted as u64) & {m} }}",
                    bits - 1
                )
            } else {
                format!("if right >= {bits} {{ 0 }} else {{ a >> right }}")
            };
            (
                format!(
                    "{{ let amount = i64::from(b as u8 as i8); if amount >= 0 {{ if amount >= {bits} {{ 0 }} else {{ (a << (amount as u32)) & {m} }} }} else {{ let right = (-amount) as u32; {right} }} }}"
                ),
                false,
            )
        }
        VecIntOp::Smax => (format!("{sa}.max({sb}) as u64"), false),
        VecIntOp::Smin => (format!("{sa}.min({sb}) as u64"), false),
        VecIntOp::Umax => ("a.max(b)".to_owned(), false),
        VecIntOp::Umin => ("a.min(b)".to_owned(), false),
        VecIntOp::Uabd => ("a.abs_diff(b)".to_owned(), false),
        VecIntOp::Sabd => (format!("{sa}.abs_diff({sb})"), false),
        VecIntOp::Sqadd => (saturate_signed(&format!("{wa} + {wb}"), bits), true),
        VecIntOp::Uqadd => (saturate_unsigned(&format!("{ua} + {ub}"), bits), true),
        VecIntOp::Sqsub => (saturate_signed(&format!("{wa} - {wb}"), bits), true),
        VecIntOp::Uqsub => (saturate_unsigned(&format!("{ua} - {ub}"), bits), true),
        VecIntOp::Shadd => (format!("(({wa} + {wb}) >> 1) as u64"), false),
        VecIntOp::Uhadd => (format!("(({ua} + {ub}) >> 1) as u64"), false),
        VecIntOp::Shsub => (format!("(({wa} - {wb}) >> 1) as u64"), false),
        VecIntOp::Uhsub => (format!("(({ua} - {ub}) >> 1) as u64"), false),
        VecIntOp::Srhadd => (format!("(({wa} + {wb} + 1) >> 1) as u64"), false),
        VecIntOp::Urhadd => (format!("(({ua} + {ub} + 1) >> 1) as u64"), false),
        // 16- and 32-bit lanes only (as decoded): the doubled product fits.
        VecIntOp::Sqdmulh if bits <= 32 => (
            saturate_signed(&format!("(2 * {wa} * {wb}) >> {bits}"), bits),
            true,
        ),
        VecIntOp::Sqrdmulh if bits <= 32 => (
            saturate_signed(
                &format!("(2 * {wa} * {wb} + (1i128 << {})) >> {bits}", bits - 1),
                bits,
            ),
            true,
        ),
        VecIntOp::Sqdmulh | VecIntOp::Sqrdmulh => return None,
        VecIntOp::Mla => ("d.wrapping_add(a.wrapping_mul(b))".to_owned(), false),
        VecIntOp::Mls => ("d.wrapping_sub(a.wrapping_mul(b))".to_owned(), false),
        VecIntOp::Srshl
        | VecIntOp::Urshl
        | VecIntOp::Sqshl
        | VecIntOp::Uqshl
        | VecIntOp::Sqrshl
        | VecIntOp::Uqrshl => {
            // `shift_register_lane`.
            let signed = matches!(op, VecIntOp::Srshl | VecIntOp::Sqshl | VecIntOp::Sqrshl);
            let round = matches!(
                op,
                VecIntOp::Srshl | VecIntOp::Urshl | VecIntOp::Sqrshl | VecIntOp::Uqrshl
            );
            let saturate = matches!(
                op,
                VecIntOp::Sqshl | VecIntOp::Uqshl | VecIntOp::Sqrshl | VecIntOp::Uqrshl
            );
            let value = if signed {
                format!("i128::from({sa})")
            } else {
                "i128::from(a)".to_owned()
            };
            let overflow = if saturate {
                "if value < 0 { i128::MIN / 2 } else { i128::MAX / 2 }"
            } else {
                "0"
            };
            let right = if round {
                "(value + (1i128 << (right - 1))) >> right"
            } else {
                "value >> right"
            };
            let result = match (saturate, signed) {
                (true, true) => saturate_signed("shifted", bits),
                (true, false) => saturate_unsigned("shifted", bits),
                _ => format!("shifted as u64 & {m}"),
            };
            (
                format!(
                    "{{ let shift = i64::from(b as u8 as i8); let value: i128 = {value}; \
                     let shifted: i128 = if shift >= 0 {{ if value == 0 {{ 0 }} else if shift >= {bits} {{ {overflow} }} else {{ value << shift }} }} \
                     else {{ let right = (-shift).min({}); {right} }}; {result} }}",
                    bits + 1
                ),
                saturate,
            )
        }
    })
}

/// A lanewise integer operation: `value` of the lanes `a` (Vn), `b` (Vm,
/// or `element` for every lane) and `d` (Vd), QC after.
#[allow(clippy::too_many_arguments)]
fn lanewise(
    value: &str,
    saturates: bool,
    bits: u32,
    q: bool,
    scalar: bool,
    rd: u8,
    rn: u8,
    rm: u8,
    element: Option<u8>,
) -> String {
    let m = mask(bits);
    let lanes = lane_count(bits, q, scalar);
    let b = match element {
        Some(_) => "element".to_owned(),
        None => get("vm", bits, "lane"),
    };
    let element = match element {
        Some(index) => format!("let element = {};", get("vm", bits, &index.to_string())),
        None => String::new(),
    };
    let qc = if saturates {
        format!("if saturated {{ {QC} }}")
    } else {
        String::new()
    };
    format!(
        "{{ let vn = m.cpu.vectors[{rn}]; let vm = m.cpu.vectors[{rm}]; let vd = m.cpu.vectors[{rd}]; {element} \
         let mut saturated = false; let mut out = 0u128; \
         for lane in 0..{lanes}u32 {{ let a = {}; let b = {b}; let d = {}; \
         let value: u64 = {value}; out |= u128::from(value & {m}) << (lane * {bits}); }} \
         {qc} m.cpu.vectors[{rd}] = {}; }}",
        get("vn", bits, "lane"),
        get("vd", bits, "lane"),
        finish(bits, q, scalar)
    )
}

/// Native code for an integer vector operation or move, or `None` to keep
/// its helper call (or another inlined form).
pub(crate) fn emit_int(op: &SimdOp) -> Option<String> {
    if !INT.get() {
        return None;
    }
    match *op {
        // Vector Add and Sub have their own inlined form (`emit_inline`).
        SimdOp::VecInt {
            op: VecIntOp::Add | VecIntOp::Sub,
            scalar: false,
            ..
        } => None,
        SimdOp::VecInt {
            op: int_op,
            bits,
            q,
            scalar,
            rd,
            rn,
            rm,
        } => {
            let bits = u32::from(bits);
            let (value, saturates) = int_lane(int_op, bits)?;
            Some(lanewise(
                &value, saturates, bits, q, scalar, rd, rn, rm, None,
            ))
        }
        SimdOp::VecIntElement {
            op: int_op,
            bits,
            q,
            scalar,
            rd,
            rn,
            rm,
            index,
        } => {
            let bits = u32::from(bits);
            let (value, saturates) = int_lane(int_op, bits)?;
            Some(lanewise(
                &value,
                saturates,
                bits,
                q,
                scalar,
                rd,
                rn,
                rm,
                Some(index),
            ))
        }
        SimdOp::VecIntPairwise {
            op: int_op,
            bits,
            q,
            scalar,
            rd,
            rn,
            rm,
        } => {
            let bits = u32::from(bits);
            let (value, saturates) = int_lane(int_op, bits)?;
            if saturates {
                return None;
            }
            let lanes = lane_count(bits, q, scalar);
            let half = (lanes / 2).max(1);
            let mut body = String::new();
            for lane in 0..lanes {
                let (source, pair) = if scalar || lane < half {
                    ("vn", lane)
                } else {
                    ("vm", lane - half)
                };
                body.push_str(&format!(
                    "{{ let a = {}; let b = {}; let d = 0u64; let value: u64 = {value}; out |= u128::from(value & {}) << {}; }} ",
                    get(source, bits, &(pair * 2).to_string()),
                    get(source, bits, &(pair * 2 + 1).to_string()),
                    mask(bits),
                    lane * bits
                ));
            }
            Some(format!(
                "{{ let vn = m.cpu.vectors[{rn}]; let vm = m.cpu.vectors[{rm}]; let mut saturated = false; let mut out = 0u128; {body}m.cpu.vectors[{rd}] = {}; }}",
                finish(bits, q, scalar)
            ))
        }
        SimdOp::VecUnary {
            op: unary,
            bits,
            q,
            scalar,
            rd,
            rn,
        } => {
            let bits = u32::from(bits);
            let m = mask(bits);
            let lanes = lane_count(bits, q, scalar);
            let sa = sext("a", bits);
            let container = match unary {
                VecUnaryOp::Rev64 => 64,
                VecUnaryOp::Rev32 => 32,
                VecUnaryOp::Rev16 => 16,
                _ => 0,
            };
            if container != 0 {
                if bits >= container {
                    return None;
                }
                let per = container / bits;
                let mut body = String::new();
                for lane in 0..lanes {
                    let source = lane / per * per + (per - 1 - lane % per);
                    body.push_str(&format!(
                        "out |= u128::from({}) << {}; ",
                        get("vn", bits, &source.to_string()),
                        lane * bits
                    ));
                }
                return Some(format!(
                    "{{ let vn = m.cpu.vectors[{rn}]; let mut out = 0u128; {body}m.cpu.vectors[{rd}] = {}; }}",
                    finish(bits, q, scalar)
                ));
            }
            let value = match unary {
                VecUnaryOp::Cnt => "u64::from(a.count_ones())".to_owned(),
                VecUnaryOp::Not => "!a".to_owned(),
                VecUnaryOp::Rbit => format!("a.reverse_bits() >> {}", 64 - bits),
                VecUnaryOp::Neg => format!("{sa}.wrapping_neg() as u64"),
                VecUnaryOp::Abs => format!("{sa}.wrapping_abs() as u64"),
                VecUnaryOp::CmEqZero => flag("a == 0", bits),
                VecUnaryOp::CmGeZero => flag(&format!("{sa} >= 0"), bits),
                VecUnaryOp::CmGtZero => flag(&format!("{sa} > 0"), bits),
                VecUnaryOp::CmLeZero => flag(&format!("{sa} <= 0"), bits),
                VecUnaryOp::CmLtZero => flag(&format!("{sa} < 0"), bits),
                VecUnaryOp::Clz => format!("u64::from(a.leading_zeros() - {})", 64 - bits),
                VecUnaryOp::Cls => format!(
                    "u64::from(((a ^ ({sa} >> 1) as u64) & {m}).leading_zeros() - {}) - 1",
                    64 - bits
                ),
                VecUnaryOp::Rev64 | VecUnaryOp::Rev32 | VecUnaryOp::Rev16 => unreachable!(),
            };
            Some(format!(
                "{{ let vn = m.cpu.vectors[{rn}]; let mut out = 0u128; \
                 for lane in 0..{lanes}u32 {{ let a = {}; let value: u64 = {value}; out |= u128::from(value & {m}) << (lane * {bits}); }} \
                 m.cpu.vectors[{rd}] = {}; }}",
                get("vn", bits, "lane"),
                finish(bits, q, scalar)
            ))
        }
        SimdOp::VecShift {
            op: shift_op,
            bits,
            q,
            scalar,
            rd,
            rn,
            shift,
        } => {
            let bits = u32::from(bits);
            let s = u32::from(shift);
            let m = mask(bits);
            let sa = sext("a", bits);
            if shift_op == VecShiftOp::Shrn {
                let half = bits / 2;
                let base = if q {
                    "vd & u128::from(u64::MAX)"
                } else {
                    "0u128"
                };
                let mut body = String::new();
                for lane in 0..128 / bits {
                    let position = if q { lane + 64 / half } else { lane };
                    body.push_str(&format!(
                        "out |= u128::from(({} >> {s}) & {}) << {}; ",
                        get("vn", bits, &lane.to_string()),
                        mask(half),
                        position * half
                    ));
                }
                return Some(format!(
                    "{{ let vn = m.cpu.vectors[{rn}]; let vd = m.cpu.vectors[{rd}]; let mut out = {base}; {body}m.cpu.vectors[{rd}] = out; }}"
                ));
            }
            // `shift_lane(a, bits, -shift, signed)`.
            let signed_right = if s == 0 {
                "a".to_owned()
            } else if s >= bits {
                format!("(({sa} >> {}) as u64 & {m})", bits - 1)
            } else {
                format!("(({sa} >> {s}) as u64 & {m})")
            };
            let unsigned_right = if s == 0 {
                "a".to_owned()
            } else if s >= bits {
                "0u64".to_owned()
            } else {
                format!("(a >> {s})")
            };
            let rounding =
                |value: &str| format!("((({value}) + (1i128 << {})) >> {s}) as u64", s - 1);
            let (value, saturates) = match shift_op {
                VecShiftOp::Shl => (
                    if s >= bits {
                        "0u64".to_owned()
                    } else {
                        format!("(a << {s}) & {m}")
                    },
                    false,
                ),
                VecShiftOp::Sshr => (signed_right, false),
                VecShiftOp::Ushr => (unsigned_right, false),
                VecShiftOp::Ssra => (format!("d.wrapping_add({signed_right})"), false),
                VecShiftOp::Usra => (format!("d.wrapping_add({unsigned_right})"), false),
                VecShiftOp::Shrn => unreachable!(),
                _ if s == 0
                    && matches!(
                        shift_op,
                        VecShiftOp::Srshr
                            | VecShiftOp::Srsra
                            | VecShiftOp::Urshr
                            | VecShiftOp::Ursra
                    ) =>
                {
                    return None;
                }
                VecShiftOp::Srshr => (rounding(&format!("i128::from({sa})")), false),
                VecShiftOp::Srsra => (
                    format!("d.wrapping_add({})", rounding(&format!("i128::from({sa})"))),
                    false,
                ),
                VecShiftOp::Urshr => (rounding("i128::from(a)"), false),
                VecShiftOp::Ursra => (
                    format!("d.wrapping_add({})", rounding("i128::from(a)")),
                    false,
                ),
                VecShiftOp::Sqshl => (
                    saturate_signed(&format!("i128::from({sa}) << {s}"), bits),
                    true,
                ),
                VecShiftOp::Uqshl => (
                    saturate_unsigned(&format!("i128::from(a) << {s}"), bits),
                    true,
                ),
                VecShiftOp::Sqshlu => (
                    saturate_unsigned(&format!("i128::from({sa}) << {s}"), bits),
                    true,
                ),
            };
            let lanes = lane_count(bits, q, scalar);
            let qc = if saturates {
                format!("if saturated {{ {QC} }}")
            } else {
                String::new()
            };
            Some(format!(
                "{{ let vn = m.cpu.vectors[{rn}]; let vd = m.cpu.vectors[{rd}]; let mut saturated = false; let mut out = 0u128; \
                 for lane in 0..{lanes}u32 {{ let a = {}; let d = {}; let value: u64 = {value}; out |= u128::from(value & {m}) << (lane * {bits}); }} \
                 {qc} m.cpu.vectors[{rd}] = {}; }}",
                get("vn", bits, "lane"),
                get("vd", bits, "lane"),
                finish(bits, q, scalar)
            ))
        }
        SimdOp::VecShiftLong {
            signed,
            bits,
            upper,
            rd,
            rn,
            shift,
        } => {
            let bits = u32::from(bits);
            let wide = bits * 2;
            let count = 64 / bits;
            let mut body = String::new();
            for lane in 0..count {
                let source = get(
                    "vn",
                    bits,
                    &(lane + if upper { count } else { 0 }).to_string(),
                );
                let extended = if signed {
                    format!("({} as u64)", sext("source", bits))
                } else {
                    "source".to_owned()
                };
                body.push_str(&format!(
                    "{{ let source = {source}; out |= u128::from(({extended} << {shift}) & {}) << {}; }} ",
                    mask(wide),
                    lane * wide
                ));
            }
            Some(format!(
                "{{ let vn = m.cpu.vectors[{rn}]; let mut out = 0u128; {body}m.cpu.vectors[{rd}] = out; }}"
            ))
        }
        SimdOp::VecLong {
            op: long_op,
            signed,
            bits,
            upper,
            rd,
            rn,
            rm,
        } => Some(long(
            long_op,
            signed,
            u32::from(bits),
            upper,
            rd,
            rn,
            rm,
            None,
        )),
        SimdOp::VecLongElement {
            op: long_op,
            signed,
            bits,
            upper,
            rd,
            rn,
            rm,
            index,
        } => Some(long(
            long_op,
            signed,
            u32::from(bits),
            upper,
            rd,
            rn,
            rm,
            Some(index),
        )),
        SimdOp::VecNarrow {
            op: narrow,
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
            let s = u32::from(shift);
            let count = if scalar { 1 } else { 64 / half };
            let base = if upper && !scalar {
                "vd & u128::from(u64::MAX)"
            } else {
                "0u128"
            };
            let wide = if narrow == VecNarrowOp::Unsigned {
                "i128::from(a)".to_owned()
            } else {
                format!("i128::from({})", sext("a", bits))
            };
            let shifted = if s == 0 {
                "wide".to_owned()
            } else if round {
                format!("(wide + (1i128 << {})) >> {s}", s - 1)
            } else {
                format!("wide >> {s}")
            };
            let (value, saturates) = match narrow {
                VecNarrowOp::Truncate => ("shifted as u64".to_owned(), false),
                VecNarrowOp::Signed => (saturate_signed("shifted", half), true),
                VecNarrowOp::Unsigned | VecNarrowOp::SignedToUnsigned => {
                    (saturate_unsigned("shifted", half), true)
                }
            };
            let offset = if upper && !scalar { count } else { 0 };
            let qc = if saturates {
                format!("if saturated {{ {QC} }}")
            } else {
                String::new()
            };
            Some(format!(
                "{{ let vn = m.cpu.vectors[{rn}]; let vd = m.cpu.vectors[{rd}]; let mut saturated = false; let mut out = {base}; \
                 for lane in 0..{count}u32 {{ let a = {}; let wide: i128 = {wide}; let shifted: i128 = {shifted}; let value: u64 = {value}; \
                 out |= u128::from(value & {}) << ((lane + {offset}) * {half}); }} \
                 {qc} m.cpu.vectors[{rd}] = out; }}",
                get("vn", bits, "lane"),
                mask(half)
            ))
        }
        SimdOp::VecPairwiseLong {
            signed,
            accumulate,
            bits,
            q,
            rd,
            rn,
        } => {
            let bits = u32::from(bits);
            let wide = bits * 2;
            let pairs = lane_count(bits, q, false) / 2;
            let extend = |value: String| {
                if signed {
                    format!("({} as u64)", sext(&value, bits))
                } else {
                    value
                }
            };
            let mut body = String::new();
            for pair in 0..pairs {
                let sum = format!(
                    "{}.wrapping_add({})",
                    extend(get("vn", bits, &(pair * 2).to_string())),
                    extend(get("vn", bits, &(pair * 2 + 1).to_string()))
                );
                let value = if accumulate {
                    format!("{}.wrapping_add({sum})", get("vd", wide, &pair.to_string()))
                } else {
                    sum
                };
                body.push_str(&format!(
                    "out |= u128::from(({value}) & {}) << {}; ",
                    mask(wide),
                    pair * wide
                ));
            }
            Some(format!(
                "{{ let vn = m.cpu.vectors[{rn}]; let vd = m.cpu.vectors[{rd}]; let mut out = 0u128; {body}m.cpu.vectors[{rd}] = {}; }}",
                finish(wide, q, false)
            ))
        }
        SimdOp::VecAcross {
            op: across,
            bits,
            q,
            rd,
            rn,
        } => {
            let bits = u32::from(bits);
            let lanes = lane_count(bits, q, false);
            let lane = |index: u32| get("vn", bits, &index.to_string());
            let signed = |index: u32| sext(&lane(index), bits);
            let (value, out_bits) = match across {
                VecAcrossOp::AddV | VecAcrossOp::UAddLV => (
                    (1..lanes).fold(lane(0), |acc, index| {
                        format!("{acc}.wrapping_add({})", lane(index))
                    }),
                    if across == VecAcrossOp::AddV {
                        bits
                    } else {
                        bits * 2
                    },
                ),
                VecAcrossOp::UMaxV | VecAcrossOp::UMinV => {
                    let method = if across == VecAcrossOp::UMaxV {
                        "max"
                    } else {
                        "min"
                    };
                    (
                        (1..lanes).fold(lane(0), |acc, index| {
                            format!("{acc}.{method}({})", lane(index))
                        }),
                        bits,
                    )
                }
                VecAcrossOp::SMaxV | VecAcrossOp::SMinV => {
                    let method = if across == VecAcrossOp::SMaxV {
                        "max"
                    } else {
                        "min"
                    };
                    (
                        format!(
                            "({}) as u64",
                            (1..lanes).fold(signed(0), |acc, index| {
                                format!("{acc}.{method}({})", signed(index))
                            })
                        ),
                        bits,
                    )
                }
                VecAcrossOp::SAddLV => (
                    format!(
                        "({}) as u64",
                        (1..lanes).fold(signed(0), |acc, index| {
                            format!("{acc}.wrapping_add({})", signed(index))
                        })
                    ),
                    bits * 2,
                ),
            };
            Some(format!(
                "{{ let vn = m.cpu.vectors[{rn}]; let value: u64 = {value}; m.cpu.vectors[{rd}] = u128::from(value & {}); }}",
                mask(out_bits)
            ))
        }
        SimdOp::MoveImmediate { rd, value } => {
            Some(format!("m.cpu.vectors[{rd}] = {value:#x}u128;"))
        }
        SimdOp::ModifyImmediate {
            rd,
            value,
            clear,
            q,
        } => {
            let combined = if clear {
                format!("current & !{value:#x}u128")
            } else {
                format!("current | {value:#x}u128")
            };
            let keep = if q { "" } else { " & u128::from(u64::MAX)" };
            Some(format!(
                "{{ let current = m.cpu.vectors[{rd}]; m.cpu.vectors[{rd}] = ({combined}){keep}; }}"
            ))
        }
        SimdOp::DupElement {
            rd,
            rn,
            bytes,
            index,
            q,
            scalar,
        } => {
            let element = element(rn, bytes, index)?;
            let value = if scalar {
                "value".to_owned()
            } else {
                format!("value * {:#x}u128", replicate(bytes, q))
            };
            Some(format!(
                "{{ let value = {element}; m.cpu.vectors[{rd}] = {value}; }}"
            ))
        }
        SimdOp::DupGeneral { rd, rn, bytes, q } => {
            let value = format!(
                "u128::from({}) & {:#x}u128",
                general(rn),
                element_mask(bytes)?
            );
            Some(format!(
                "{{ let value = {value}; m.cpu.vectors[{rd}] = value * {:#x}u128; }}",
                replicate(bytes, q)
            ))
        }
        SimdOp::InsertElement {
            rd,
            rn,
            bytes,
            dst,
            src,
        } => {
            let value = element(rn, bytes, src)?;
            Some(insert(&value, rd, bytes, dst)?)
        }
        SimdOp::InsertGeneral {
            rd,
            rn,
            bytes,
            index,
        } => {
            let value = format!(
                "(u128::from({}) & {:#x}u128)",
                general(rn),
                element_mask(bytes)?
            );
            Some(insert(&value, rd, bytes, index)?)
        }
        SimdOp::MoveToGeneral {
            rd,
            rn,
            bytes,
            index,
            signed,
            gp_bits,
        } => {
            let element_bits = u32::from(bytes) * 8;
            if element_bits > 64 {
                return None;
            }
            let value = format!("({} as u64)", element(rn, bytes, index)?);
            let value = if signed {
                format!("({} as u64)", sext(&value, element_bits))
            } else {
                value
            };
            let value = if gp_bits == 32 {
                format!("(({value}) & 0xffff_ffffu64)")
            } else {
                value
            };
            if rd == 31 {
                // The zero register: nothing is written.
                Some(String::new())
            } else {
                Some(format!("m.cpu.regs[{rd}] = {value};"))
            }
        }
        _ => None,
    }
}

/// `vec_long_values`: widening operations on the lower or upper half.
#[allow(clippy::too_many_arguments)]
fn long(
    op: VecLongOp,
    signed: bool,
    bits: u32,
    upper: bool,
    rd: u8,
    rn: u8,
    rm: u8,
    element: Option<u8>,
) -> String {
    let wide = bits * 2;
    let count = 64 / bits;
    let extend = |value: String| {
        if signed {
            format!("({} as u64)", sext(&value, bits))
        } else {
            value
        }
    };
    let abs_diff = if signed {
        "(a as i64).abs_diff(b as i64)"
    } else {
        "a.abs_diff(b)"
    };
    let value = match op {
        VecLongOp::Abal => format!("acc.wrapping_add({abs_diff})"),
        VecLongOp::Abdl => abs_diff.to_owned(),
        VecLongOp::Mlal => "acc.wrapping_add(a.wrapping_mul(b))".to_owned(),
        VecLongOp::Mlsl => "acc.wrapping_sub(a.wrapping_mul(b))".to_owned(),
        VecLongOp::Mull => "a.wrapping_mul(b)".to_owned(),
        VecLongOp::Addl | VecLongOp::Addw => "a.wrapping_add(b)".to_owned(),
        VecLongOp::Subl | VecLongOp::Subw => "a.wrapping_sub(b)".to_owned(),
    };
    let mut body = String::new();
    for lane in 0..count {
        let source = lane + if upper { count } else { 0 };
        let a = if matches!(op, VecLongOp::Addw | VecLongOp::Subw) {
            get("vn", wide, &lane.to_string())
        } else {
            extend(get("vn", bits, &source.to_string()))
        };
        let b = match element {
            Some(index) => extend(get("vm", bits, &index.to_string())),
            None => extend(get("vm", bits, &source.to_string())),
        };
        body.push_str(&format!(
            "{{ let a: u64 = {a}; let b: u64 = {b}; let acc = {}; let value: u64 = {value}; out |= u128::from(value & {}) << {}; }} ",
            get("vd", wide, &lane.to_string()),
            mask(wide),
            lane * wide
        ));
    }
    format!(
        "{{ let vn = m.cpu.vectors[{rn}]; let vm = m.cpu.vectors[{rm}]; let vd = m.cpu.vectors[{rd}]; let mut out = 0u128; {body}m.cpu.vectors[{rd}] = out; }}"
    )
}

fn element_mask(bytes: u8) -> Option<u128> {
    match bytes {
        1 | 2 | 4 | 8 => Some((1u128 << (u32::from(bytes) * 8)) - 1),
        _ => None,
    }
}

/// Element `index` of Vn, `bytes` wide, as a u128.
fn element(rn: u8, bytes: u8, index: u8) -> Option<String> {
    Some(format!(
        "((m.cpu.vectors[{rn}] >> {}) & {:#x}u128)",
        u32::from(index) * u32::from(bytes) * 8,
        element_mask(bytes)?
    ))
}

/// The constant whose product with an element replicates it across 16
/// bytes (`q`) or 8.
fn replicate(bytes: u8, q: bool) -> u128 {
    let total = if q { 16 } else { 8 };
    (0..total / u32::from(bytes)).fold(0u128, |value, lane| {
        value | 1u128 << (lane * u32::from(bytes) * 8)
    })
}

/// General register `rn` read as 64 bits, the zero register as 0.
fn general(rn: u8) -> String {
    if rn == 31 {
        "0u64".to_owned()
    } else {
        format!("m.cpu.regs[{rn}]")
    }
}

/// `value` (a u128 element) written into element `index` of Vd.
fn insert(value: &str, rd: u8, bytes: u8, index: u8) -> Option<String> {
    let shift = u32::from(index) * u32::from(bytes) * 8;
    let mask = element_mask(bytes)? << shift;
    Some(format!(
        "{{ let value = {value}; let vector = m.cpu.vectors[{rd}]; m.cpu.vectors[{rd}] = (vector & !{mask:#x}u128) | (value << {shift}); }}"
    ))
}

/// Native code for a float pairwise operation (FADDP, FMAXP, FMINP,
/// FMAXNMP, FMINNMP): each result lane natively when its pair is normal or
/// zero and (FADDP) the sum is normal or zero, under the FPCR the helpers
/// accept and, for FADDP, FPSR.IXC already set; otherwise that lane alone
/// through the exact single-lane helper.
pub(crate) fn emit_pairwise(op: &SimdOp) -> Option<String> {
    if !PAIRWISE.get() {
        return None;
    }
    let SimdOp::VecFpPairwise {
        op: fp_op,
        size,
        q,
        scalar,
        rd,
        rn,
        rm,
    } = *op
    else {
        return None;
    };
    let (float, int, width, kit) = float_kit(size);
    let (native, raw, needs_ixc) = match fp_op {
        VecFpOp::Add => (
            "usable(x) & usable(y) & (normal(raw) | zero(raw))",
            format!("({float}::from_bits(x) + {float}::from_bits(y)).to_bits()"),
            true,
        ),
        VecFpOp::Max | VecFpOp::MaxNum => (
            "usable(x) & usable(y)",
            format!(
                "if zero(x) && zero(y) {{ x & y }} else if {float}::from_bits(x) > {float}::from_bits(y) {{ x }} else {{ y }}"
            ),
            false,
        ),
        VecFpOp::Min | VecFpOp::MinNum => (
            "usable(x) & usable(y)",
            format!(
                "if zero(x) && zero(y) {{ x | y }} else if {float}::from_bits(x) < {float}::from_bits(y) {{ x }} else {{ y }}"
            ),
            false,
        ),
        _ => return None,
    };
    if size == FpSize::D && !q && !scalar {
        return None;
    }
    let p = "echora_guest::simd";
    let mode = if needs_ixc {
        format!("{FP_MODE} && m.cpu.fpsr & {p}::FPSR_IXC != 0")
    } else {
        FP_MODE.to_owned()
    };
    let lanes = lane_count(width, q, scalar);
    let half = lanes / 2;
    let mut body = String::new();
    for lane in 0..lanes {
        let (source, pair) = if scalar || lane < half {
            ("vn", lane)
        } else {
            ("vm", lane - half)
        };
        body.push_str(&format!(
            "{{ let x = ({source} >> {}) as {int}; let y = ({source} >> {}) as {int}; let raw: {int} = {raw}; \
             let raw = if {native} {{ raw }} else {{ let exact = {p}::vec_fp_lane(&mut m.cpu, {p}::VecFpOp::{fp_op:?}, {p}::FpSize::{size:?}, x as u64, y as u64, 0)?; exact as {int} }}; \
             out |= u128::from(raw) << {}; }} ",
            pair * 2 * width,
            (pair * 2 + 1) * width,
            lane * width
        ));
    }
    let call = crate::simd::emit_call(op)?;
    Some(format!(
        "{{ let vn = m.cpu.vectors[{rn}]; let vm = m.cpu.vectors[{rm}]; {kit} \
         if {mode} {{ let mut out = 0u128; {body}m.cpu.vectors[{rd}] = {}; }} else {{ {call} }} }}",
        finish(width, q, scalar)
    ))
}
