//! TBL/TBX, the lane, replicate and multiple-structure loads and stores,
//! FABD, FMULX, FMAXV and its kin and the remaining conversions as native
//! code (`EmitOptions::inline_vector_misc`), against the same words lifted
//! with their helpers.
//!
//! `build.rs` (`vector_misc_code`) samples random words of each such
//! operation and form, one word and an SVC each. Each runs from vector
//! registers of float and integer edges and random bits, general registers
//! (and SP) pointing into the mapped data (an access outside it ends the
//! process either way), data memory of random bytes,
//! under FPCR.FZ and FPCR.DN on and off and a rounding mode the float
//! helpers trap on. Every register, the data memory, FPSR and the trap
//! must agree.
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
    include!(concat!(env!("OUT_DIR"), "/lifted_vector_misc.rs"));
}
use lifted::{vector_misc_code, vector_misc_reference};

const FPCR_FZ: u32 = 1 << 24;
const FPCR_DN: u32 = 1 << 25;
const FPCR_RP: u32 = 1 << 22;
const DATA: u64 = 0x1000_0000;
const DATA_SIZE: u64 = 0x1000;

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
    sp: u64,
    data: Vec<u8>,
    fpsr: u32,
    trap: String,
}

struct State {
    vectors: [u128; 32],
    regs: [u64; 31],
    sp: u64,
    data: Vec<u8>,
    status: (u32, u32),
}

fn run(
    dispatch: fn(&mut echora_guest::Core) -> Result<(), Trap>,
    m: &mut Machine,
    entry: u64,
    state: &State,
) -> Outcome {
    m.memory.write(DATA, &state.data).unwrap();
    m.core.cpu.vectors = state.vectors;
    m.core.cpu.regs = state.regs;
    m.core.cpu.sp = state.sp;
    m.core.cpu.fpcr = state.status.0;
    m.core.cpu.fpsr = state.status.1;
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
    let mut data = vec![0u8; DATA_SIZE as usize];
    m.memory.read(DATA, &mut data).unwrap();
    Outcome {
        vectors: m.core.cpu.vectors,
        regs: m.core.cpu.regs,
        sp: m.core.cpu.sp,
        data,
        fpsr: m.core.cpu.fpsr,
        trap: format!("{trap:?} at {:#x}", m.core.cpu.pc),
    }
}

/// A float or integer edge value, or random bits.
fn value(random: &mut Random) -> u64 {
    match random.next() % 4 {
        0 => u64::from(SINGLES[random.next() as usize % SINGLES.len()]),
        1 => DOUBLES[random.next() as usize % DOUBLES.len()],
        2 => random.value(),
        _ => random.next(),
    }
}

/// A vector of edge values in its lanes (single or double), table
/// indices around the tables' sizes, or random bits.
fn vector(random: &mut Random) -> u128 {
    match random.next() % 4 {
        0 => (0..4).fold(0u128, |vector, lane| {
            vector | u128::from(SINGLES[random.next() as usize % SINGLES.len()]) << (32 * lane)
        }),
        1 => u128::from(value(random)) | u128::from(value(random)) << 64,
        2 => u128::from_le_bytes(std::array::from_fn(|_| (random.next() % 72) as u8)),
        _ => u128::from(random.next()) << 64 | u128::from(random.next()),
    }
}

#[test]
fn tables_and_structure_transfers_leave_no_helper_call() {
    const { assert!(vector_misc_code::HELPER_CALLS == 0) };
    const { assert!(vector_misc_reference::HELPER_CALLS > 100) };
    const { assert!(vector_misc_code::FORMS > 200) };
}

#[test]
fn inlined_operations_match_the_helpers() {
    let words = vector_misc_code::WORDS;
    let code_size = (words.len() as u64 * 4).next_multiple_of(0x1000);
    let image: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let mut m = Machine::with_layout(vector_misc_code::BASE, &image, code_size, DATA, DATA_SIZE);
    let mut random = Random(0x3c5a_96e1_0000_0011);
    let fpcrs = [0, FPCR_FZ, FPCR_DN, FPCR_FZ | FPCR_DN, FPCR_RP];
    let fpsrs = [0, 0x10, 0x9f];
    let mut runs = 0;
    let mut failures = Vec::new();
    for (index, &entry) in vector_misc_code::ENTRIES.iter().enumerate() {
        let memory = matches!(
            echora_lift::aarch64::decode(words[index * 2], entry),
            echora_lift::aarch64::Instruction::Simd(
                echora_lift::simd::SimdOp::TransferSingle { .. }
                    | echora_lift::simd::SimdOp::TransferMultiple { .. }
                    | echora_lift::simd::SimdOp::LoadReplicate { .. }
            )
        );
        for run_index in 0..40 {
            // For a load or store, every general register and SP inside
            // the data, with room for the largest transfer (64 bytes) and a
            // post-index of up to 64 (a guest access outside mapped memory
            // ends the process, with the helper as with the inlined code:
            // both make the same `GuestView` access); for the other
            // operations, every other run, edge values.
            let address = |random: &mut Random| DATA + random.next() % (DATA_SIZE - 128);
            let edges = !memory && run_index % 2 == 1;
            let state = State {
                vectors: std::array::from_fn(|_| vector(&mut random)),
                regs: std::array::from_fn(|_| {
                    if edges {
                        value(&mut random)
                    } else {
                        address(&mut random)
                    }
                }),
                sp: address(&mut random) & !15,
                data: (0..DATA_SIZE).map(|_| random.next() as u8).collect(),
                status: (
                    fpcrs[run_index % fpcrs.len()],
                    fpsrs[random.next() as usize % fpsrs.len()],
                ),
            };
            let expected = run(vector_misc_reference::dispatch, &mut m, entry, &state);
            let actual = run(vector_misc_code::dispatch, &mut m, entry, &state);
            runs += 1;
            if actual != expected {
                let word = words[index * 2];
                let mut changed: Vec<String> = (0..32)
                    .filter(|&v| actual.vectors[v] != expected.vectors[v])
                    .map(|v| {
                        format!(
                            "V{v} {:#x} vs {:#x}",
                            expected.vectors[v], actual.vectors[v]
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
                if actual.sp != expected.sp {
                    changed.push(format!("SP {:#x} vs {:#x}", expected.sp, actual.sp));
                }
                if actual.data != expected.data {
                    changed.push("data memory".to_owned());
                }
                if actual.fpsr != expected.fpsr {
                    changed.push(format!("FPSR {:#x} vs {:#x}", expected.fpsr, actual.fpsr));
                }
                if actual.trap != expected.trap {
                    changed.push(format!("{} vs {}", expected.trap, actual.trap));
                }
                failures.push(format!(
                    "{:?} ({word:#010x}) under FPCR {:#x}, FPSR {:#x}: {changed:?}",
                    echora_lift::aarch64::decode(word, entry),
                    state.status.0,
                    state.status.1
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

/// Every word from the edge tables (`common::edge_vectors`, which include
/// table indices around 16 * n as bytes), general registers inside the
/// data for loads and stores and the register edges for the rest, data of
/// a fixed pattern, under three FPCR and FPSR settings.
#[test]
fn inlined_operations_match_the_helpers_on_edge_tables() {
    let words = vector_misc_code::WORDS;
    let code_size = (words.len() as u64 * 4).next_multiple_of(0x1000);
    let image: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let mut m = Machine::with_layout(vector_misc_code::BASE, &image, code_size, DATA, DATA_SIZE);
    let mut edges = edge_vectors();
    // Table indices at the ends of one to four table registers.
    for index in [15u8, 16, 31, 32, 47, 48, 63, 64, 255] {
        edges.push(u128::from_le_bytes([index; 16]));
    }
    let statuses = [(0, 0), (FPCR_FZ, 0x10), (FPCR_FZ | FPCR_DN, 0x10)];
    let data: Vec<u8> = (0..DATA_SIZE)
        .map(|index| (index * 37 + 11) as u8)
        .collect();
    let mut failures = Vec::new();
    'words: for (index, &entry) in vector_misc_code::ENTRIES.iter().enumerate() {
        let memory = matches!(
            echora_lift::aarch64::decode(words[index * 2], entry),
            echora_lift::aarch64::Instruction::Simd(
                echora_lift::simd::SimdOp::TransferSingle { .. }
                    | echora_lift::simd::SimdOp::TransferMultiple { .. }
                    | echora_lift::simd::SimdOp::LoadReplicate { .. }
            )
        );
        for step in [7, 11] {
            for first in 0..edges.len() {
                for status in statuses {
                    let state = State {
                        vectors: std::array::from_fn(|v| edges[(v * step + first) % edges.len()]),
                        regs: std::array::from_fn(|x| {
                            if memory {
                                DATA + ((x * step + first) as u64 * 24) % (DATA_SIZE - 128)
                            } else {
                                REGISTER_EDGES[(x + first) % REGISTER_EDGES.len()]
                            }
                        }),
                        sp: DATA + 0x400,
                        data: data.clone(),
                        status,
                    };
                    let expected = run(vector_misc_reference::dispatch, &mut m, entry, &state);
                    let actual = run(vector_misc_code::dispatch, &mut m, entry, &state);
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
