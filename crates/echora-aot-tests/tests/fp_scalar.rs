//! Scalar float moves, conversions, FSQRT, FRINT* and FCCMP as native code
//! (`EmitOptions::inline_fp_scalar`), against the same words lifted with
//! their helpers.
//!
//! `build.rs` (`fp_scalar_code`) samples random words of FMOV to and from
//! general registers, SCVTF/UCVTF (from general and FP registers),
//! FCVT*S/U to general registers in every rounding, FCVT between single
//! and double, FSQRT, FRINT*, FRINTX and FCCMP/FCCMPE, one word and an
//! SVC each. Each runs from FP registers holding float edges (zeros,
//! denormals, the normal range's ends, the integer range ends, halves,
//! infinities, NaNs) or random bits, general registers likewise, random
//! NZCV, under FPCR.FZ and FPCR.DN on and off and a rounding mode the
//! helpers trap on. Every register, NZCV, FPSR and the trap must agree.
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
    include!(concat!(env!("OUT_DIR"), "/lifted_fp_scalar.rs"));
}
use lifted::{fp_scalar_code, fp_scalar_reference};

const FPCR_FZ: u32 = 1 << 24;
const FPCR_DN: u32 = 1 << 25;
const FPCR_RP: u32 = 1 << 22;

/// Single-precision values where a native path could differ.
const SINGLES: &[u32] = &[
    0x0000_0000,
    0x8000_0000,
    0x3f80_0000,
    0xbf80_0000,
    0x3f00_0000,
    0xbf00_0000,
    0x3fc0_0000,
    0xc020_0000,
    0xbf7d_70a4,
    0x4f00_0000,
    0xcf00_0000,
    0xcf00_0001,
    0x4eff_ffff,
    0x4f80_0000,
    0x4f7f_ffff,
    0x5f00_0000,
    0xdf00_0000,
    0x5f80_0000,
    0x0000_0001,
    0x007f_ffff,
    0x8000_0001,
    0x0080_0000,
    0x7f7f_ffff,
    0xff7f_ffff,
    0x7f80_0000,
    0xff80_0000,
    0x7fc0_0000,
    0x7fa0_0000,
    0xffc0_0001,
    0x4049_0fdb,
];

/// Double-precision values, as `SINGLES` (and those whose single is
/// denormal, infinite or zero).
const DOUBLES: &[u64] = &[
    0x0000_0000_0000_0000,
    0x8000_0000_0000_0000,
    0x3ff0_0000_0000_0000,
    0xbff0_0000_0000_0000,
    0x3fe0_0000_0000_0000,
    0xbfe0_0000_0000_0000,
    0xbfef_ae14_7ae1_47ae,
    0x41e0_0000_0000_0000,
    0xc1e0_0000_0000_0000,
    0x41df_ffff_ffc0_0000,
    0x41ef_ffff_ffe0_0000,
    0x41f0_0000_0000_0000,
    0x43e0_0000_0000_0000,
    0xc3e0_0000_0000_0000,
    0x43df_ffff_ffff_ffff,
    0x43f0_0000_0000_0000,
    0x43ef_ffff_ffff_ffff,
    0x0000_0000_0000_0001,
    0x000f_ffff_ffff_ffff,
    0x0010_0000_0000_0000,
    0x7fef_ffff_ffff_ffff,
    0x7ff0_0000_0000_0000,
    0xfff0_0000_0000_0000,
    0x7ff8_0000_0000_0000,
    0x7ff4_0000_0000_0000,
    0x47f0_0000_0000_0000,
    0x47ef_ffff_f000_0000,
    0x3810_0000_0000_0000,
    0x380f_ffff_ffff_ffff,
    0x3690_0000_0000_0000,
    0x36a0_0000_0000_0000,
    0x4009_21fb_5444_2d18,
];

#[derive(Debug, PartialEq)]
struct Outcome {
    vectors: [u128; 32],
    regs: [u64; 31],
    nzcv: u8,
    fpsr: u32,
    trap: String,
}

fn run(
    dispatch: fn(&mut echora_guest::Core) -> Result<(), Trap>,
    m: &mut Machine,
    entry: u64,
    (vectors, regs, nzcv): (&[u128; 32], &[u64; 31], u8),
    (fpcr, fpsr): (u32, u32),
) -> Outcome {
    m.core.cpu.vectors = *vectors;
    m.core.cpu.regs = *regs;
    m.core.cpu.nzcv = nzcv;
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
        vectors: m.core.cpu.vectors,
        regs: m.core.cpu.regs,
        nzcv: m.core.cpu.nzcv,
        fpsr: m.core.cpu.fpsr,
        trap: format!("{trap:?} at {:#x}", m.core.cpu.pc),
    }
}

/// A float or integer edge value, or random bits: the low 32 bits a
/// single, or all 64 a double, or an integer edge.
fn value(random: &mut Random) -> u64 {
    match random.next() % 4 {
        0 => u64::from(SINGLES[random.next() as usize % SINGLES.len()]),
        1 => DOUBLES[random.next() as usize % DOUBLES.len()],
        2 => random.value(),
        _ => random.next(),
    }
}

#[test]
fn moves_leave_no_helper_call() {
    const { assert!(fp_scalar_code::MOVE_CALLS == 0) };
    const { assert!(fp_scalar_reference::MOVE_CALLS > 20) };
    const { assert!(fp_scalar_code::FORMS > 100) };
}

#[test]
fn inlined_scalar_operations_match_the_helpers() {
    let words = fp_scalar_code::WORDS;
    let code_size = (words.len() as u64 * 4).next_multiple_of(0x1000);
    let image: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let mut m = Machine::with_layout(fp_scalar_code::BASE, &image, code_size, 0x1000_0000, 0x1000);
    let mut random = Random(0x5ca1_a2f0_0000_0009);
    let fpcrs = [0, FPCR_FZ, FPCR_DN, FPCR_FZ | FPCR_DN, FPCR_RP];
    let fpsrs = [0, 0x10, 0x9f];
    let mut runs = 0;
    let mut failures = Vec::new();
    for (index, &entry) in fp_scalar_code::ENTRIES.iter().enumerate() {
        for run_index in 0..60 {
            let vectors: [u128; 32] = std::array::from_fn(|_| {
                u128::from(value(&mut random)) | u128::from(random.next()) << 64
            });
            let regs: [u64; 31] = std::array::from_fn(|_| value(&mut random));
            let nzcv = (random.next() & 15) as u8;
            let status = (
                fpcrs[run_index % fpcrs.len()],
                fpsrs[random.next() as usize % fpsrs.len()],
            );
            let state = (&vectors, &regs, nzcv);
            let expected = run(fp_scalar_reference::dispatch, &mut m, entry, state, status);
            let actual = run(fp_scalar_code::dispatch, &mut m, entry, state, status);
            runs += 1;
            if actual != expected {
                let word = words[index * 2];
                let mut changed: Vec<String> = (0..32)
                    .filter(|&v| actual.vectors[v] != expected.vectors[v])
                    .map(|v| {
                        format!(
                            "V{v} {:#x} (from {:#x}) vs {:#x}",
                            expected.vectors[v], vectors[v], actual.vectors[v]
                        )
                    })
                    .chain(
                        (0..31)
                            .filter(|&x| actual.regs[x] != expected.regs[x])
                            .map(|x| {
                                format!("X{x} {:#x} vs {:#x}", expected.regs[x], actual.regs[x])
                            }),
                    )
                    .collect();
                if actual.nzcv != expected.nzcv {
                    changed.push(format!("NZCV {:#x} vs {:#x}", expected.nzcv, actual.nzcv));
                }
                if actual.fpsr != expected.fpsr {
                    changed.push(format!("FPSR {:#x} vs {:#x}", expected.fpsr, actual.fpsr));
                }
                if actual.trap != expected.trap {
                    changed.push(format!("{} vs {}", expected.trap, actual.trap));
                }
                failures.push(format!(
                    "{:?} ({word:#010x}) under FPCR {:#x}, FPSR {:#x}, NZCV {nzcv:#x}: {changed:?}",
                    echora_lift::aarch64::decode(word, entry),
                    status.0,
                    status.1
                ));
                break;
            }
        }
    }
    println!("{runs} runs, {} mismatching words", failures.len());
    assert!(
        failures.is_empty(),
        "{} mismatching words, first: {}",
        failures.len(),
        failures
            .iter()
            .take(5)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// Every word from the edge tables (`common::edge_vectors`, the low lane
/// of each the operand), general registers the register edges and the
/// float edges as bits, every NZCV, under three FPCR and FPSR settings.
#[test]
fn inlined_scalar_operations_match_the_helpers_on_edge_tables() {
    let words = fp_scalar_code::WORDS;
    let code_size = (words.len() as u64 * 4).next_multiple_of(0x1000);
    let image: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let mut m = Machine::with_layout(fp_scalar_code::BASE, &image, code_size, 0x1000_0000, 0x1000);
    let edges = edge_vectors();
    let integers: Vec<u64> = REGISTER_EDGES
        .iter()
        .copied()
        .chain(SINGLES.iter().map(|&value| u64::from(value)))
        .chain(DOUBLES.iter().copied())
        .collect();
    let statuses = [(0, 0), (FPCR_FZ, 0x10), (FPCR_FZ | FPCR_DN, 0x10)];
    let mut failures = Vec::new();
    'words: for (index, &entry) in fp_scalar_code::ENTRIES.iter().enumerate() {
        for step in [7, 11] {
            for first in 0..edges.len() {
                let vectors: [u128; 32] =
                    std::array::from_fn(|v| edges[(v * step + first) % edges.len()]);
                let regs: [u64; 31] =
                    std::array::from_fn(|x| integers[(x * step + first) % integers.len()]);
                let nzcv = (first % 16) as u8;
                for status in statuses {
                    let state = (&vectors, &regs, nzcv);
                    let expected = run(fp_scalar_reference::dispatch, &mut m, entry, state, status);
                    let actual = run(fp_scalar_code::dispatch, &mut m, entry, state, status);
                    if actual != expected {
                        failures.push(format!(
                            "{:?} with edges from {first} (step {step}) under {status:x?}",
                            echora_lift::aarch64::decode(words[index * 2], entry)
                        ));
                        continue 'words;
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} words: {}",
        failures.len(),
        failures
            .iter()
            .take(5)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}
