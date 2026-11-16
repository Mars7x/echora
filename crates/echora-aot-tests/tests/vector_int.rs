//! Vector integer operations, element moves and float pairwise operations
//! as native code (`EmitOptions::inline_vector_int`,
//! `inline_vector_pairwise`), against the same words lifted with their
//! helpers.
//!
//! `build.rs` (`vector_int_code`) samples random words of every such
//! operation and form (three-same, by element, pairwise, unary, shifts by
//! immediate, widening, narrowing, across lanes, DUP, INS, UMOV/SMOV,
//! MOVI, ORR/BIC immediate, FADDP/FMAXP/FMINP/FMAXNMP/FMINNMP), one word
//! and an SVC each. Each runs from vector registers whose lanes (of a
//! random width) are edge values (0, 1, all ones, the sign bit alone, the
//! largest positive, small signed shift counts, float edges) or random
//! bits, with general registers likewise, under FPCR.FZ and FPCR.DN on and
//! off, from FPSR with and without QC and IXC, and under a rounding mode
//! the float helpers trap on. Every vector and general register, FPSR and
//! the trap must agree.
#![cfg(any(target_os = "linux", windows))]
#![allow(clippy::unwrap_used)]

mod common;

use common::*;
use echora_aot_tests::{Machine, Trap, vector_int_code, vector_int_reference};

const FPCR_FZ: u32 = 1 << 24;
const FPCR_DN: u32 = 1 << 25;
const FPCR_RP: u32 = 1 << 22;
const FPSR_QC: u32 = 1 << 27;
const FPSR_IXC: u32 = 1 << 4;

#[derive(Debug, PartialEq)]
struct Outcome {
    vectors: [u128; 32],
    regs: [u64; 31],
    fpsr: u32,
    trap: String,
}

fn run(
    dispatch: fn(&mut echora_guest::Core) -> Result<(), Trap>,
    m: &mut Machine,
    entry: u64,
    (vectors, regs): (&[u128; 32], &[u64; 31]),
    (fpcr, fpsr): (u32, u32),
) -> Outcome {
    m.core.cpu.vectors = *vectors;
    m.core.cpu.regs = *regs;
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
        fpsr: m.core.cpu.fpsr,
        trap: format!("{trap:?} at {:#x}", m.core.cpu.pc),
    }
}

/// A lane of `bits`: an edge value or random bits.
fn lane(random: &mut Random, bits: u32) -> u64 {
    let mask = if bits == 64 {
        u64::MAX
    } else {
        (1 << bits) - 1
    };
    let sign = 1u64 << (bits - 1);
    let value = match random.next() % 12 {
        0 => 0,
        1 => 1,
        2 => mask,
        3 => sign,
        4 => sign - 1,
        5 => sign + 1,
        // Signed shift counts at and around every lane width (a shift
        // operand's lane may be narrower than this one).
        6 => {
            const COUNTS: [i64; 17] = [
                0, 1, -1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, -128,
            ];
            let count = COUNTS[random.next() as usize % COUNTS.len()];
            (if random.next().is_multiple_of(2) {
                count
            } else {
                count.wrapping_neg()
            }) as u64
        }
        // Float edges.
        7 if bits == 32 => [
            0x3f80_0000,
            0x8000_0000,
            0x0000_0001,
            0x7f80_0000,
            0x7fc0_0000,
            0x7fa0_0000,
            0x7f7f_ffff,
            0x3eaa_aaab,
        ][random.next() as usize % 8],
        7 if bits == 64 => [
            0x3ff0_0000_0000_0000,
            0x8000_0000_0000_0000,
            0x0000_0000_0000_0001,
            0x7ff0_0000_0000_0000,
            0x7ff8_0000_0000_0000,
            0x7ff4_0000_0000_0000,
            0x7fef_ffff_ffff_ffff,
            0x3fd5_5555_5555_5555,
        ][random.next() as usize % 8],
        _ => random.next(),
    };
    value & mask
}

/// A vector of lanes of one random width, or (one in three) one shift
/// count in every byte, so that a shift operand's low byte holds it
/// whatever the lane width.
fn vector(random: &mut Random) -> u128 {
    if random.next().is_multiple_of(3) {
        const COUNTS: [i8; 8] = [0, 1, 8, 16, 32, 63, 64, 65];
        let count = COUNTS[random.next() as usize % COUNTS.len()];
        let count = if random.next().is_multiple_of(2) {
            count
        } else {
            count.wrapping_neg()
        };
        return u128::from_ne_bytes([count as u8; 16]);
    }
    let bits = [8, 16, 32, 64][random.next() as usize % 4];
    (0..128 / bits).fold(0u128, |vector, index| {
        vector | u128::from(lane(random, bits)) << (index * bits)
    })
}

#[test]
fn integer_and_move_operations_leave_no_helper_call() {
    const { assert!(vector_int_code::HELPER_CALLS == 0) };
    const { assert!(vector_int_reference::HELPER_CALLS > 2000) };
    // The float pairwise words keep the single-lane helper for their
    // fallback lanes; the reference has none.
    const { assert!(vector_int_code::LANE_CALLS > 0) };
    const { assert!(vector_int_reference::LANE_CALLS == 0) };
    const { assert!(vector_int_code::FORMS > 1500) };
}

#[test]
fn inlined_integer_and_pairwise_operations_match_the_helpers() {
    let words = vector_int_code::WORDS;
    let code_size = (words.len() as u64 * 4).next_multiple_of(0x1000);
    let image: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let mut m = Machine::with_layout(
        vector_int_code::BASE,
        &image,
        code_size,
        0x1000_0000,
        0x1000,
    );
    let mut random = Random(0x1a7e_9e44_0000_0003);
    let fpcrs = [0, FPCR_FZ, FPCR_DN, FPCR_FZ | FPCR_DN, FPCR_RP];
    let fpsrs = [0, FPSR_IXC, FPSR_QC | FPSR_IXC, 0x9f];
    let mut runs = 0;
    let mut failures = Vec::new();
    for (index, &entry) in vector_int_code::ENTRIES.iter().enumerate() {
        for run_index in 0..24 {
            let vectors: [u128; 32] = std::array::from_fn(|_| vector(&mut random));
            let regs: [u64; 31] = std::array::from_fn(|_| lane(&mut random, 64));
            let status = (
                fpcrs[run_index % fpcrs.len()],
                fpsrs[random.next() as usize % fpsrs.len()],
            );
            let expected = run(
                vector_int_reference::dispatch,
                &mut m,
                entry,
                (&vectors, &regs),
                status,
            );
            let actual = run(
                vector_int_code::dispatch,
                &mut m,
                entry,
                (&vectors, &regs),
                status,
            );
            runs += 1;
            if actual != expected {
                let word = words[index * 2];
                let changed: Vec<String> = (0..32)
                    .filter(|&v| actual.vectors[v] != expected.vectors[v])
                    .map(|v| {
                        format!(
                            "V{v}: expected {:#x}, got {:#x}",
                            expected.vectors[v], actual.vectors[v]
                        )
                    })
                    .chain(
                        (0..31)
                            .filter(|&x| actual.regs[x] != expected.regs[x])
                            .map(|x| {
                                format!(
                                    "X{x}: expected {:#x}, got {:#x}",
                                    expected.regs[x], actual.regs[x]
                                )
                            }),
                    )
                    .collect();
                failures.push(format!(
                    "{:?} ({word:#010x}) under FPCR {:#x}, FPSR {:#x}: {changed:?}, FPSR {:#x} vs {:#x}, {} vs {}",
                    echora_lift::aarch64::decode(word, entry),
                    status.0,
                    status.1,
                    expected.fpsr,
                    actual.fpsr,
                    expected.trap,
                    actual.trap
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
            .take(10)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}
