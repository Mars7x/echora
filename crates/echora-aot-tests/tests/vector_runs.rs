//! Vector float runs (`EmitOptions::vector_float_runs`): two to six vector
//! float arithmetic operations in a row checked once on a fast copy, the
//! operations as each is emitted alone otherwise, against the same programs
//! lifted without runs (`build.rs`, `vector_run_code`).
//!
//! The programs use V0 to V3, so results feed later operations and
//! registers alias; a quarter write FPCR from X5 in the middle (which ends
//! a run there). Each runs from vectors whose lanes are mostly ordinary
//! values and sometimes float edges (NaN payloads, signalling NaNs,
//! denormals, infinities, zeros, the extremes, and adjacent lanes whose
//! pairwise sum overflows, is denormal or meets a NaN), so an exceptional lane
//! falls in the middle of a run, in an operand or a result; under the
//! three FPCR and FPSR settings of the edge-table tests, FPSR without IXC
//! (no run commits) and a rounding mode the helpers trap on, with X5 any
//! of those modes. The vector registers, FPCR, FPSR and the trap must
//! agree, every program from every state. The lift with runs has every
//! per-instruction inline option (`inline-vector-unary`, `-int`,
//! `-pairwise`, `inline-fp-scalar`, `inline-vector-misc`) in every other
//! 1 KiB region as well (`split_alternate`), so a run's exact copy is
//! there those options' code, the helpers' elsewhere: where a run and an
//! option take the same form, the run's fast copy runs and falls back to
//! the option's code.
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
    include!(concat!(env!("OUT_DIR"), "/lifted_vector_run.rs"));
}
use lifted::{vector_run_code, vector_run_reference};

const FPCR_FZ: u32 = 1 << 24;
const FPCR_DN: u32 = 1 << 25;
const FPCR_RP: u32 = 1 << 22;
const FPSR_IXC: u32 = 1 << 4;

const SINGLE_EDGES: [u32; 16] = [
    0x7fc0_0001,
    0xffc1_2345,
    0x7f80_0001,
    0xffa0_0000,
    0x0000_0001,
    0x807f_ffff,
    0x7f80_0000,
    0xff80_0000,
    0x0000_0000,
    0x8000_0000,
    0x7f7f_ffff,
    0xff7f_ffff,
    0x0080_0000,
    0x8080_0000,
    0x0080_0001,
    0x3f80_0000,
];
const DOUBLE_EDGES: [u64; 14] = [
    0x7ff8_0000_0000_0001,
    0xfff8_1234_5678_9abc,
    0x7ff0_0000_0000_0001,
    0x0000_0000_0000_0001,
    0x7ff0_0000_0000_0000,
    0xfff0_0000_0000_0000,
    0x0000_0000_0000_0000,
    0x8000_0000_0000_0000,
    0x7fef_ffff_ffff_ffff,
    0xffef_ffff_ffff_ffff,
    0x0010_0000_0000_0000,
    0x8010_0000_0000_0000,
    0x0010_0000_0000_0001,
    0x3ff0_0000_0000_0000,
];
/// Adjacent lanes whose pairwise sum (FADDP) overflows, is denormal, or
/// meets a NaN, in singles and doubles.
const SINGLE_PAIRS: [[u32; 2]; 3] = [
    [0x7f7f_ffff, 0x7f7f_ffff],
    [0x0080_0001, 0x8080_0000],
    [0x7fc0_0001, 0x3f80_0000],
];
const DOUBLE_PAIRS: [[u64; 2]; 3] = [
    [0x7fef_ffff_ffff_ffff, 0x7fef_ffff_ffff_ffff],
    [0x0010_0000_0000_0001, 0x8010_0000_0000_0000],
    [0x7ff8_0000_0000_0001, 0x3ff0_0000_0000_0000],
];
/// Ordinary values: results stay normal through a few operations.
const SINGLE_ORDINARY: [f32; 8] = [1.5, -2.25, 0.75, 3.0, -0.5, 1.25, 10.0, -7.5];

#[derive(Debug, PartialEq)]
struct Outcome {
    vectors: [u128; 4],
    fpcr: u32,
    fpsr: u32,
    trap: String,
}

fn run(
    dispatch: fn(&mut echora_guest::Core) -> Result<(), Trap>,
    m: &mut Machine,
    entry: u64,
    vectors: [u128; 4],
    x5: u64,
    (fpcr, fpsr): (u32, u32),
) -> Outcome {
    m.core.cpu.vectors[..4].copy_from_slice(&vectors);
    m.core.cpu.regs[5] = x5;
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

/// A vector of single or double lanes, mostly ordinary, a quarter edges;
/// one in eight of adjacent edge pairs (`SINGLE_PAIRS`, `DOUBLE_PAIRS`).
fn vector(random: &mut Random) -> u128 {
    if random.next().is_multiple_of(8) {
        return if random.next().is_multiple_of(2) {
            (0..2).fold(0u128, |vector, pair| {
                let [a, b] = SINGLE_PAIRS[random.next() as usize % SINGLE_PAIRS.len()];
                vector | (u128::from(a) | u128::from(b) << 32) << (64 * pair)
            })
        } else {
            let [a, b] = DOUBLE_PAIRS[random.next() as usize % DOUBLE_PAIRS.len()];
            u128::from(a) | u128::from(b) << 64
        };
    }
    if random.next().is_multiple_of(2) {
        (0..4).fold(0u128, |vector, lane| {
            let value = if random.next().is_multiple_of(4) {
                SINGLE_EDGES[random.next() as usize % SINGLE_EDGES.len()]
            } else {
                SINGLE_ORDINARY[random.next() as usize % SINGLE_ORDINARY.len()].to_bits()
            };
            vector | u128::from(value) << (32 * lane)
        })
    } else {
        (0..2).fold(0u128, |vector, lane| {
            let value = if random.next().is_multiple_of(4) {
                DOUBLE_EDGES[random.next() as usize % DOUBLE_EDGES.len()]
            } else {
                f64::from(SINGLE_ORDINARY[random.next() as usize % SINGLE_ORDINARY.len()]).to_bits()
            };
            vector | u128::from(value) << (64 * lane)
        })
    }
}

#[test]
fn runs_are_formed() {
    const { assert!(vector_run_code::RUNS > 300) };
    const { assert!(vector_run_reference::RUNS == 0) };
    const { assert!(vector_run_code::FORMS >= 30) };
}

#[test]
fn vector_float_runs_match_the_operations_alone() {
    let words = vector_run_code::WORDS;
    let code_size = (words.len() as u64 * 4).next_multiple_of(0x1000);
    let image: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let mut m = Machine::with_layout(
        vector_run_code::BASE,
        &image,
        code_size,
        0x1000_0000,
        0x1000,
    );
    let mut random = Random(0x7e57_a11e_0000_0017);
    let statuses = [
        (0, 0),
        (FPCR_FZ, FPSR_IXC | (1 << 27)),
        (FPCR_FZ | FPCR_DN, FPSR_IXC),
        (FPCR_FZ, 0),
        (0, FPSR_IXC),
        (FPCR_RP, FPSR_IXC),
    ];
    let fpcrs = [
        0u64,
        u64::from(FPCR_FZ),
        u64::from(FPCR_FZ | FPCR_DN),
        u64::from(FPCR_RP),
    ];
    let mut failures = Vec::new();
    let mut runs = 0;
    for (index, &entry) in vector_run_code::ENTRIES.iter().enumerate() {
        'states: for round in 0..48 {
            let vectors: [u128; 4] = std::array::from_fn(|_| vector(&mut random));
            let x5 = fpcrs[random.next() as usize % fpcrs.len()];
            let status = statuses[round % statuses.len()];
            let expected = run(
                vector_run_reference::dispatch,
                &mut m,
                entry,
                vectors,
                x5,
                status,
            );
            let actual = run(
                vector_run_code::dispatch,
                &mut m,
                entry,
                vectors,
                x5,
                status,
            );
            runs += 1;
            if actual != expected {
                failures.push(format!(
                    "program {index} at {entry:#x} from {vectors:x?}, X5 {x5:#x}, {status:x?}: expected {expected:x?}, got {actual:x?}"
                ));
                break 'states;
            }
        }
    }
    println!("{runs} runs, {} mismatching programs", failures.len());
    assert!(
        failures.is_empty(),
        "{} programs: {}",
        failures.len(),
        failures
            .iter()
            .take(3)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}
