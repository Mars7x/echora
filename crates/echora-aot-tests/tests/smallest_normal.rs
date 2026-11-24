//! Results at the smallest normal: Arm detects underflow before rounding,
//! so a product, quotient, fused result or narrowed value whose exact
//! magnitude is just below the smallest normal and rounds up to it raises
//! UFC, and under FZ is a zero. The host's rounded value is a normal, so
//! the inline paths must not take it.
//!
//! `build.rs` (`smallest_normal_code`) lifts each multiply, divide, fused
//! operation and narrowing FCVT the inline options take natively, alone,
//! and a vector and a scalar float run of multiplies, with every inline
//! and run option on and with none (the helpers alone). Each runs from
//! operands whose exact result is at that boundary (and, as a check, just
//! above and below it), under FZ on and off and FPSR with and without IXC:
//! the vector registers, FPCR, FPSR and the trap must agree.
//!
//! The vector fused forms (FMLA, FMLS, by element, single precision) are
//! also checked against Arm's rounding itself (`arm_fused`), independently
//! of the helpers. `tests/rounded_up.rs` checks the same boundary against
//! known values; this test adds the run options, FMULX, the narrowing
//! FCVT, operands far from the lowest binade whose product rounds up, and
//! the FPCR and FPSR mix of the other float tests.
#![cfg(any(target_os = "linux", windows))]
#![allow(clippy::unwrap_used)]

mod common;

use common::*;
use echora_aot_tests::{Machine, Trap};

/// The lifts this test runs (`build.rs` writes each to its own file).
#[allow(
    missing_debug_implementations,
    clippy::all,
    dead_code,
    unused_imports,
    unused_parens,
    unreachable_code
)]
mod lifted {
    include!(concat!(env!("OUT_DIR"), "/lifted_smallest_normal.rs"));
}
use lifted::{smallest_normal_code, smallest_normal_reference};

const FPCR_FZ: u32 = 1 << 24;
const FPCR_DN: u32 = 1 << 25;
const FPSR_IXC: u32 = 1 << 4;
const FPSR_UFC: u32 = 1 << 3;

/// Single-precision (Vn, Vm, Va) lanes: the exact product or quotient is
/// just below 2^-126 and rounds to it (ties to even, or nearer), for
/// multiplies (by 1 - 2^-24, by a half, by 1 - 2^-23 of 1 + 2^-23) and
/// divides (by two), either sign; an exact 2^-126 and one that rounds to
/// the largest denormal, which every path already treats right; and
/// operands far from the lowest binade whose product or quotient rounds to
/// 2^-126 (the result tests alone must catch those).
const SINGLES: &[(u32, u32, u32)] = &[
    (0x0080_0000, 0x3f7f_ffff, 0),
    (0x8080_0000, 0x3f7f_ffff, 0x8000_0000),
    (0x00ff_ffff, 0x3f00_0000, 0),
    (0x0080_0001, 0x3f7f_fffe, 0x8000_0000),
    (0x00ff_ffff, 0x4000_0000, 0),
    (0x80ff_ffff, 0x4000_0000, 0),
    (0x0080_0000, 0x3f80_0000, 0),
    (0x0080_0000, 0x3f80_0001, 0),
    // Operands far from the lowest binade: 2^-63 times 2^-64 (2 - 2^-23),
    // and 2^-63 (2 - 2^-23) over 2^64.
    (0x2000_0000, 0x1fff_ffff, 0),
    (0xa000_0000, 0x1fff_ffff, 0x8000_0000),
    (0x207f_ffff, 0x5f80_0000, 0),
];
/// The same in double precision.
const DOUBLES: &[(u64, u64, u64)] = &[
    (0x0010_0000_0000_0000, 0x3fef_ffff_ffff_ffff, 0),
    (
        0x001f_ffff_ffff_ffff,
        0x3fe0_0000_0000_0000,
        0x8000_0000_0000_0000,
    ),
    (0x801f_ffff_ffff_ffff, 0x4000_0000_0000_0000, 0),
    (0x0010_0000_0000_0000, 0x3ff0_0000_0000_0000, 0),
    // 2^-511 times 2^-512 (2 - 2^-52).
    (0x2000_0000_0000_0000, 0x1fff_ffff_ffff_ffff, 0),
    (0xa000_0000_0000_0000, 0x1fff_ffff_ffff_ffff, 0),
];
/// Doubles just below 2^-126 that narrow to it (FCVT S, D), and one that
/// narrows to the largest single denormal.
const NARROWED: &[u64] = &[
    0x380f_ffff_f000_0000,
    0xb80f_ffff_f800_0000,
    0x380f_ffff_e000_0000,
    0x3810_0000_0000_0000,
];

#[derive(Debug, PartialEq)]
struct Outcome {
    vectors: [u128; 5],
    fpcr: u32,
    fpsr: u32,
    trap: String,
}

fn run(
    dispatch: fn(&mut echora_guest::Core) -> Result<(), Trap>,
    m: &mut Machine,
    entry: u64,
    vectors: [u128; 5],
    (fpcr, fpsr): (u32, u32),
) -> Outcome {
    m.core.cpu.vectors[..5].copy_from_slice(&vectors);
    m.core.cpu.fpcr = fpcr;
    m.core.cpu.fpsr = fpsr;
    m.core.cpu.pc = entry;
    m.core.host_check_at = u64::MAX;
    m.core.rearm();
    let trap = loop {
        match dispatch(&mut m.core) {
            Ok(()) => continue,
            Err(trap) => break trap,
        }
    };
    m.core.settle_block_trap();
    Outcome {
        vectors: std::array::from_fn(|v| m.core.cpu.vectors[v]),
        fpcr: m.core.cpu.fpcr,
        fpsr: m.core.cpu.fpsr,
        trap: format!("{trap:?} at {:#x}", m.core.cpu.pc),
    }
}

/// `value` in every 32-bit lane, or in all but lane 2 (1.0 there).
fn singles(value: u32, mixed: bool) -> u128 {
    (0..4).fold(0u128, |vector, lane| {
        let lane_value = if mixed && lane == 2 {
            0x3f80_0000
        } else {
            value
        };
        vector | u128::from(lane_value) << (32 * lane)
    })
}

/// `value` in both 64-bit lanes, or 1.0 in the upper one.
fn doubles(value: u64, mixed: bool) -> u128 {
    let upper = if mixed { 0x3ff0_0000_0000_0000 } else { value };
    u128::from(value) | u128::from(upper) << 64
}

/// The word a program starts with.
fn first_word(entry: u64) -> u32 {
    smallest_normal_code::WORDS[((entry - smallest_normal_code::BASE) / 4) as usize]
}

/// A vector fused form (single precision): the lane computation (`d +
/// n * m`, or `d - n * m`) and the element of Vm every lane takes (by
/// element), or `None`.
struct Fused {
    subtract: bool,
    element: Option<u32>,
}

fn fused_model(entry: u64) -> Option<Fused> {
    let (subtract, element) = match first_word(entry) {
        0x4e22_cc20 => (false, None),    // FMLA 4S
        0x4ea2_cc20 => (true, None),     // FMLS 4S
        0x4f82_1020 => (false, Some(0)), // FMLA 4S by element
        _ => return None,
    };
    Some(Fused { subtract, element })
}

/// Arm's single-precision rounding of `value + error` (round to nearest;
/// `error` what the double `value` left out, too small to move the rounding
/// here): underflow is detected before rounding, so a nonzero value below
/// 2^-126 is a zero with UFC under FZ, and otherwise raises UFC when
/// inexact.
fn arm_round(value: f64, error: f64, fz: bool) -> (u32, u32) {
    let tiny = value != 0.0 && value.abs() < f64::from(f32::MIN_POSITIVE);
    if tiny && fz {
        return (u32::from(value.is_sign_negative()) << 31, FPSR_UFC);
    }
    let rounded = value as f32;
    let inexact = f64::from(rounded) != value || error != 0.0;
    let flags = if inexact { FPSR_IXC } else { 0 } | if tiny && inexact { FPSR_UFC } else { 0 };
    (rounded.to_bits(), flags)
}

/// A vector fused form by Arm's rounding, for these inputs (a product of
/// singles is exact in a double; its sum with Vd's lane is the double sum
/// and that sum's error, which here is nonzero only far from any tie).
fn arm_fused(model: &Fused, vectors: [u128; 5], (fpcr, fpsr): (u32, u32), trap: &str) -> Outcome {
    let lane =
        |vector: u128, index: u32| f64::from(f32::from_bits((vector >> (32 * index)) as u32));
    let mut flags = 0;
    let mut result = 0u128;
    for index in 0..4 {
        let product = lane(vectors[1], index) * lane(vectors[2], model.element.unwrap_or(index));
        let addend = lane(vectors[0], index);
        let product = if model.subtract { -product } else { product };
        // Knuth's two-sum: the double sum and what it left out.
        let sum = addend + product;
        let back = sum - product;
        let error = (addend - back) + (product - (sum - back));
        let (bits, raised) = arm_round(sum, error, fpcr & FPCR_FZ != 0);
        flags |= raised;
        result |= u128::from(bits) << (32 * index);
    }
    let mut out = vectors;
    out[0] = result;
    Outcome {
        vectors: out,
        fpcr,
        fpsr: fpsr | flags,
        trap: trap.to_owned(),
    }
}

#[test]
fn results_rounding_to_the_smallest_normal_match_the_helpers() {
    let words = smallest_normal_code::WORDS;
    let code_size = (words.len() as u64 * 4).next_multiple_of(0x1000);
    let image: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let mut m = Machine::with_layout(
        smallest_normal_code::BASE,
        &image,
        code_size,
        DATA,
        DATA_SIZE,
    );
    // The vectors, and whether they hold single-precision lanes.
    let mut inputs: Vec<([u128; 5], bool)> = Vec::new();
    for mixed in [false, true] {
        for &(n, mm, a) in SINGLES {
            inputs.push((
                [
                    singles(a, mixed),
                    singles(n, mixed),
                    singles(mm, mixed),
                    singles(a, mixed),
                    0,
                ],
                true,
            ));
        }
        for &(n, mm, a) in DOUBLES {
            inputs.push((
                [
                    doubles(a, mixed),
                    doubles(n, mixed),
                    doubles(mm, mixed),
                    doubles(a, mixed),
                    0,
                ],
                false,
            ));
        }
    }
    for &value in NARROWED {
        inputs.push(([0, doubles(value, false), 0, 0, 0], false));
    }
    let statuses = [
        (0, FPSR_IXC),
        (FPCR_FZ, FPSR_IXC),
        (FPCR_FZ | FPCR_DN, FPSR_IXC),
        (0, 0),
        (FPCR_FZ, FPSR_IXC | FPSR_UFC),
    ];
    let mut runs = 0;
    let mut failures = Vec::new();
    for &entry in smallest_normal_code::ENTRIES {
        for (vectors, single) in &inputs {
            let fused = fused_model(entry);
            if fused.is_some() && !single {
                continue;
            }
            for &status in &statuses {
                let actual = run(
                    smallest_normal_code::dispatch,
                    &mut m,
                    entry,
                    *vectors,
                    status,
                );
                let expected = match &fused {
                    Some(model) => arm_fused(model, *vectors, status, &actual.trap),
                    None => run(
                        smallest_normal_reference::dispatch,
                        &mut m,
                        entry,
                        *vectors,
                        status,
                    ),
                };
                runs += 1;
                if actual != expected && failures.len() < 8 {
                    failures.push(format!(
                        "program at {entry:#x} from {vectors:x?}, {status:x?}: expected {expected:x?}, got {actual:x?}"
                    ));
                }
            }
        }
    }
    println!("{runs} runs, {} mismatches", failures.len());
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
