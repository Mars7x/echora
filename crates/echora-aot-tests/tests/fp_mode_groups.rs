//! `EmitOptions::hoist_fp_mode`: consecutive inlined float operations test
//! the FPCR mode and sticky IXC once, before the first, and each keeps its
//! exact fallback.
//!
//! `build.rs` (`fp_mode_group_code`) lifts groups of scalar operations
//! (FMUL, FADD, FDIV, FMADD, FSUB; doubles; FCMP and FMAX among them) with
//! every inline path and the option, alone and with an MSR FPSR (which can
//! clear IXC) or an MSR FPCR between them, against the helpers alone. Each
//! runs from operands that are normal, inexact, denormal, zero, infinite
//! or NaN, under FPCR with FZ, DN and a directed rounding mode on and off,
//! FPSR with and without IXC, and each value the MSRs write: the vector
//! registers, FPCR, FPSR, NZCV and the trap must agree.
#![cfg(any(target_os = "linux", windows))]
#![allow(clippy::unwrap_used)]

mod common;

use common::*;
use echora_aot_tests::{Machine, Trap};

#[allow(
    missing_debug_implementations,
    clippy::all,
    dead_code,
    unused_imports,
    unused_parens,
    unreachable_code
)]
mod lifted {
    include!(concat!(env!("OUT_DIR"), "/lifted_fp_mode_group.rs"));
}
use lifted::{fp_mode_group_code as code, fp_mode_group_reference as reference};

const FPCR_FZ: u64 = 1 << 24;
const FPCR_DN: u64 = 1 << 25;
/// Round toward minus infinity.
const FPCR_RM: u64 = 2 << 22;
const FPSR_IXC: u64 = 1 << 4;
const FPSR_UFC: u64 = 1 << 3;

#[derive(Debug, PartialEq)]
struct Outcome {
    vectors: [u128; 8],
    fpcr: u32,
    fpsr: u32,
    nzcv: u8,
    trap: String,
}

fn run(
    dispatch: fn(&mut echora_guest::Core) -> Result<(), Trap>,
    m: &mut Machine,
    entry: u64,
    operands: [u128; 3],
    (fpcr, fpsr, x5, x6): (u64, u64, u64, u64),
) -> Outcome {
    m.core.cpu.vectors = [0; 32];
    m.core.cpu.vectors[1..4].copy_from_slice(&operands);
    m.core.cpu.fpcr = fpcr as u32;
    m.core.cpu.fpsr = fpsr as u32;
    m.core.cpu.nzcv = 0;
    m.core.cpu.regs[5] = x5;
    m.core.cpu.regs[6] = x6;
    m.core.cpu.pc = entry;
    let trap = match m.run(dispatch) {
        Err(trap) => format!("{trap:?} at {:#x}", m.core.cpu.pc),
        Ok(()) => unreachable!("run returns only a trap"),
    };
    Outcome {
        vectors: std::array::from_fn(|index| m.core.cpu.vectors[index]),
        fpcr: m.core.cpu.fpcr,
        fpsr: m.core.cpu.fpsr,
        nzcv: m.core.cpu.nzcv,
        trap,
    }
}

#[test]
fn the_groups_are_hoisted() {
    let source = include_str!(concat!(env!("OUT_DIR"), "/lifted_fp_mode_group.rs"));
    let (grouped, helpers) =
        source.split_at(source.find("pub mod fp_mode_group_reference").unwrap());
    // One test per group: the five singles; before and after each MSR
    // (two groups each); the four after an MSR FPSR that follows the
    // first; the doubles. FMAX tests no IXC and FCMP is not inlined with
    // this test, so the compare program forms no group.
    // With regalloc each block is emitted in its window and its structured
    // function: twice.
    let groups = grouped.matches("let fp_mode_").count();
    let copies = if cfg!(feature = "regalloc") { 2 } else { 1 };
    assert_eq!(groups, 7 * copies);
    assert!(!helpers.contains("let fp_mode_"));
}

#[test]
fn grouped_operations_match_the_helpers() {
    let words = code::WORDS;
    let code_size = (words.len() as u64 * 4).next_multiple_of(0x1000);
    let image: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let mut m = Machine::with_layout(code::BASE, &image, code_size, DATA, DATA_SIZE);
    let singles: [u32; 8] = [
        0x3fc0_0000, // 1.5
        0x4040_0000, // 3.0
        0x3dcc_cccd, // 0.1 (products inexact)
        0x0000_0001, // the smallest denormal
        0x0000_0000,
        0x7f80_0000, // infinity
        0x7fc0_0001, // a quiet NaN
        0x7f00_0000, // 2^127 (products overflow)
    ];
    let doubles: [u64; 6] = [
        0x3ff8_0000_0000_0000,
        0x4008_0000_0000_0000,
        0x3fb9_9999_9999_999a,
        0x0000_0000_0000_0001,
        0x7ff0_0000_0000_0000,
        0x7fe0_0000_0000_0000,
    ];
    let statuses = [
        (FPCR_FZ, FPSR_IXC),
        (FPCR_FZ, 0),
        (FPCR_FZ | FPCR_DN, FPSR_IXC),
        (0, FPSR_IXC),
        (FPCR_FZ | FPCR_RM, FPSR_IXC),
    ];
    let writes = [
        (0, FPCR_FZ),
        (FPSR_IXC, 0),
        (FPSR_IXC | FPSR_UFC, FPCR_FZ | FPCR_RM),
        (FPSR_IXC, FPCR_FZ | FPCR_DN),
    ];
    let mut runs = 0;
    let mut failures = Vec::new();
    for (index, &entry) in code::ENTRIES.iter().enumerate() {
        let double = code::NAMES[index] == "doubles";
        let values: Vec<u128> = if double {
            doubles.iter().map(|&value| u128::from(value)).collect()
        } else {
            singles.iter().map(|&value| u128::from(value)).collect()
        };
        for (a, &first) in values.iter().enumerate() {
            for &second in &values {
                let third = values[(a + 3) % values.len()];
                for &(fpcr, fpsr) in &statuses {
                    for &(x5, x6) in &writes {
                        let state = (fpcr, fpsr, x5, x6);
                        let operands = [first, second, third];
                        let expected = run(reference::dispatch, &mut m, entry, operands, state);
                        let actual = run(code::dispatch, &mut m, entry, operands, state);
                        runs += 1;
                        if actual != expected && failures.len() < 8 {
                            failures.push(format!(
                                "{} from {operands:x?}, {state:x?}: expected {expected:x?}, got {actual:x?}",
                                code::NAMES[index]
                            ));
                        }
                    }
                }
            }
        }
    }
    println!("{runs} runs, {} mismatches", failures.len());
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
