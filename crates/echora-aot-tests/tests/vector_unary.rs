//! Vector FSQRT, FCVTZS/FCVTZU (toward zero), FRECPE, FRSQRTE and FRINT*
//! inlined lane by lane (`EmitOptions::inline_vector_unary`), against the
//! same instructions lifted with the helper for every lane.
//!
//! `build.rs` (`vector_unary_code`) lifts each instruction in each form
//! (4S, 2S, 2D, and the scalar S and D forms that exist), from V1 into V0
//! and into V1 itself. Each runs from vectors of edge lanes (zeros,
//! denormals, the normal range's ends, integer range ends for the
//! conversions, the estimates' boundary exponents, infinities and NaNs)
//! and random bits, under FPCR.FZ and FPCR.DN on and off, from random
//! FPSR flags, and under a rounding mode the helpers do not implement
//! (both must trap alike). The vector registers, FPCR, FPSR and the trap
//! must agree. The estimates are also compared for every fraction of one
//! even and one odd exponent.
#![cfg(any(target_os = "linux", windows))]
#![allow(clippy::unwrap_used)]

mod common;

use common::*;
use echora_aot_tests::{Machine, Trap};

/// The lifts this test runs (`build.rs` writes each to its own file, so
/// that no one compilation holds every lift).
#[allow(
    missing_debug_implementations,
    clippy::all,
    dead_code,
    unused_imports,
    unused_parens,
    unreachable_code
)]
mod lifted {
    include!(concat!(env!("OUT_DIR"), "/lifted_vector_unary.rs"));
}
use lifted::{vector_unary_code, vector_unary_reference};

const FPCR_FZ: u32 = 1 << 24;
const FPCR_DN: u32 = 1 << 25;
/// Round toward plus infinity: the helpers do not implement it and trap.
const FPCR_RP: u32 = 1 << 22;

/// Single-precision lanes where a native path could differ from the
/// helper's.
const SINGLES: &[u32] = &[
    0x0000_0000, // +0
    0x8000_0000, // -0
    0x3f80_0000, // 1
    0xbf80_0000, // -1
    0x3fc0_0000, // 1.5
    0xc020_0000, // -2.5
    0x3f00_0000, // 0.5
    0xbf00_0000, // -0.5
    0xbf7d_70a4, // -0.99: FCVTZU to 0, in range
    0x4f00_0000, // 2^31: FCVTZS out of range
    0xcf00_0000, // -2^31: FCVTZS in range
    0xcf00_0001, // below -2^31
    0x4eff_ffff, // the largest single below 2^31
    0x4f80_0000, // 2^32: FCVTZU out of range
    0x4f7f_ffff, // the largest single below 2^32
    0x0000_0001, // the smallest denormal
    0x007f_ffff, // the largest denormal
    0x8000_0001, // a negative denormal
    0x0080_0000, // the smallest normal
    0x8080_0000, // its negative
    0x7f7f_ffff, // the largest finite
    0xff7f_ffff, // its negative
    0x7e80_0000, // 2^126: FRECPE's result denormal
    0x7e7f_ffff, // just below 2^126
    0x7f00_0000, // 2^127
    0x7f80_0000, // +inf
    0xff80_0000, // -inf
    0x7fc0_0000, // quiet NaN
    0x7fa0_0000, // signaling NaN
    0xffc0_0001, // negative quiet NaN with payload
    0x4b40_0001, // an odd integer near 2^23: FRINT exact
    0x3f40_0000, // 0.75: FRINT ties and directions
    0xbfc0_0000, // -1.5: FRINT ties
    0x4049_0fdb, // pi
];

/// Double-precision lanes, as `SINGLES`.
const DOUBLES: &[u64] = &[
    0x0000_0000_0000_0000,
    0x8000_0000_0000_0000,
    0x3ff0_0000_0000_0000, // 1
    0xbff0_0000_0000_0000, // -1
    0x3ff8_0000_0000_0000, // 1.5
    0xc004_0000_0000_0000, // -2.5
    0xbfe0_0000_0000_0000, // -0.5
    0xbfef_ae14_7ae1_47ae, // -0.99
    0x43e0_0000_0000_0000, // 2^63: FCVTZS out of range
    0xc3e0_0000_0000_0000, // -2^63: in range
    0xc3e0_0000_0000_0001, // below -2^63
    0x43df_ffff_ffff_ffff, // the largest double below 2^63
    0x43f0_0000_0000_0000, // 2^64: FCVTZU out of range
    0x43ef_ffff_ffff_ffff, // the largest double below 2^64
    0x41df_ffff_ffc0_0000, // 2^31 - 1: in range for 64 bits
    0x0000_0000_0000_0001, // the smallest denormal
    0x000f_ffff_ffff_ffff, // the largest denormal
    0x8000_0000_0000_0001,
    0x0010_0000_0000_0000, // the smallest normal
    0x8010_0000_0000_0000,
    0x7fef_ffff_ffff_ffff, // the largest finite
    0x7fd0_0000_0000_0000, // 2^1022: FRECPE's result denormal
    0x7fcf_ffff_ffff_ffff, // just below
    0x7fe0_0000_0000_0000, // 2^1023
    0x7ff0_0000_0000_0000, // +inf
    0xfff0_0000_0000_0000, // -inf
    0x7ff8_0000_0000_0000, // quiet NaN
    0x7ff4_0000_0000_0000, // signaling NaN
    0x4330_0000_0000_0001, // an odd integer at 2^52
    0x3fe8_0000_0000_0000, // 0.75
    0xbff8_0000_0000_0000, // -1.5
    0x4009_21fb_5444_2d18, // pi
];

#[derive(Debug, PartialEq)]
struct Outcome {
    vectors: [u128; 2],
    fpcr: u32,
    fpsr: u32,
    trap: String,
}

fn machine() -> Machine {
    let words = vector_unary_code::WORDS;
    let code_size = (words.len() as u64 * 4).next_multiple_of(0x1000);
    let image: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    Machine::with_layout(
        vector_unary_code::BASE,
        &image,
        code_size,
        0x1000_0000,
        0x1000,
    )
}

fn run(
    dispatch: fn(&mut echora_guest::Core) -> Result<(), Trap>,
    m: &mut Machine,
    entry: u64,
    vectors: [u128; 2],
    (fpcr, fpsr): (u32, u32),
) -> Outcome {
    m.core.cpu.vectors[..2].copy_from_slice(&vectors);
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
        vectors: [m.core.cpu.vectors[0], m.core.cpu.vectors[1]],
        fpcr: m.core.cpu.fpcr,
        fpsr: m.core.cpu.fpsr,
        trap: format!("{trap:?} at {:#x}", m.core.cpu.pc),
    }
}

/// A vector of `form`'s lanes, each an edge value or (one in four) random
/// bits.
fn lanes(random: &mut Random, form: &str) -> u128 {
    let double = matches!(form, "2d" | "d");
    let (width, count) = if double { (64, 2) } else { (32, 4) };
    (0..count).fold(0u128, |vector, lane| {
        let value = match (double, random.next() % 4) {
            (false, 0) => random.next() & 0xffff_ffff,
            (false, _) => u64::from(SINGLES[random.next() as usize % SINGLES.len()]),
            (true, 0) => random.next(),
            (true, _) => DOUBLES[random.next() as usize % DOUBLES.len()],
        };
        vector | u128::from(value) << (width * lane)
    })
}

#[test]
fn every_target_instruction_has_a_native_lane_path() {
    assert_eq!(
        vector_unary_code::INLINED,
        vector_unary_code::PROGRAMS.len()
    );
    assert_eq!(vector_unary_reference::INLINED, 0);
}

#[test]
fn inlined_unary_operations_match_the_helper_on_edge_lanes() {
    let mut m = machine();
    let mut random = Random(0x0b5e_55ed_0000_0001);
    let fpcrs = [0, FPCR_FZ, FPCR_DN, FPCR_FZ | FPCR_DN];
    let fpsrs = [0, 0x10, 0x9f];
    let mut runs = 0;
    let mut failures = Vec::new();
    for &(name, form, entry, in_place) in vector_unary_code::PROGRAMS {
        'program: for &fpcr in &fpcrs {
            for &fpsr in &fpsrs {
                for _ in 0..60 {
                    let vectors = [
                        u128::from(random.next()) << 64 | u128::from(random.next()),
                        lanes(&mut random, form),
                    ];
                    let expected = run(
                        vector_unary_reference::dispatch,
                        &mut m,
                        entry,
                        vectors,
                        (fpcr, fpsr),
                    );
                    let actual = run(
                        vector_unary_code::dispatch,
                        &mut m,
                        entry,
                        vectors,
                        (fpcr, fpsr),
                    );
                    runs += 1;
                    if actual != expected {
                        failures.push(format!(
                            "{name} {form} (in place: {in_place}) from {vectors:x?}, FPCR {fpcr:#x}, FPSR {fpsr:#x}: expected {expected:x?}, got {actual:x?}"
                        ));
                        break 'program;
                    }
                }
            }
        }
    }
    println!("{runs} runs, {} mismatching programs", failures.len());
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn an_unimplemented_rounding_mode_traps_as_the_helper_does() {
    let mut m = machine();
    let mut random = Random(0x7a9_0000_0002);
    for &(name, form, entry, _) in vector_unary_code::PROGRAMS {
        let vectors = [
            u128::from(random.next()) << 64 | u128::from(random.next()),
            lanes(&mut random, form),
        ];
        let status = (FPCR_RP, 0x10);
        let expected = run(
            vector_unary_reference::dispatch,
            &mut m,
            entry,
            vectors,
            status,
        );
        let actual = run(vector_unary_code::dispatch, &mut m, entry, vectors, status);
        assert!(
            expected.trap.starts_with("InvalidState"),
            "{name} {form}: {expected:x?}"
        );
        assert_eq!(actual, expected, "{name} {form}");
    }
}

/// FRECPE and FRSQRTE (4S) for every fraction of an even and an odd
/// exponent, four lanes a run, positive and (for FRECPE) negative.
#[test]
fn estimates_match_the_helper_for_every_fraction() {
    let mut m = machine();
    let mut checked = 0u64;
    for &(name, form, entry, in_place) in vector_unary_code::PROGRAMS {
        if form != "4s" || in_place || !matches!(name, "frecpe" | "frsqrte") {
            continue;
        }
        for exponent in [0x7f_u32, 0x80] {
            for sign in [0, 0x8000_0000_u32] {
                if name == "frsqrte" && sign != 0 {
                    continue;
                }
                for first in (0..1u32 << 23).step_by(4) {
                    let vector = (0..4).fold(0u128, |vector, lane| {
                        let bits = sign | exponent << 23 | (first + lane);
                        vector | u128::from(bits) << (32 * lane)
                    });
                    let vectors = [0, vector];
                    let expected = run(
                        vector_unary_reference::dispatch,
                        &mut m,
                        entry,
                        vectors,
                        (0, 0),
                    );
                    let actual = run(vector_unary_code::dispatch, &mut m, entry, vectors, (0, 0));
                    assert_eq!(actual, expected, "{name} of {vector:#x}");
                    checked += 4;
                }
            }
        }
    }
    assert_eq!(checked, 3 << 24);
}
