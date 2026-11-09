//! FPCR and FPSR in window locals (`EmitOptions::fp_status_locals`),
//! against the same programs lifted without them.
//!
//! `build.rs` (`fp_status_code`) lifts programs that write FPCR and FPSR
//! with MSR and read them with MRS around float operations (scalar and
//! vector) whose helpers raise FPSR flags or read FPCR: a local the window
//! wrote must reach the helper, and what the helper raised must reach the
//! window's next read and the end of the window. Each runs from random
//! FPCR and FPSR values and operands (NaNs, infinities, denormals and
//! inexact sums among them) under both lifts; registers, FPCR, FPSR and the
//! vector registers must agree.
#![cfg(any(target_os = "linux", windows))]
#![allow(clippy::unwrap_used)]

mod common;

use common::*;
use echora_aot_tests::{Machine, Trap, fp_status_code, fp_status_reference};

/// Float bit patterns that make the helpers raise or check something.
const FLOATS: &[u32] = &[
    0x3f80_0000, // 1.0
    0x3eaa_aaab, // 1/3: inexact sums
    0x7f80_0000, // +inf
    0x7fc0_0000, // quiet NaN
    0x7fa0_0000, // signaling NaN
    0x0000_0001, // the smallest denormal
    0x8000_0000, // -0
    0x7f7f_ffff, // the largest finite: overflow
];

#[derive(Debug, PartialEq)]
struct Outcome {
    regs: [u64; 31],
    vectors: [u128; 3],
    fpcr: u32,
    fpsr: u32,
    nzcv: u8,
}

fn run(
    dispatch: fn(&mut echora_guest::Core) -> Result<(), Trap>,
    m: &mut Machine,
    entry: u64,
    regs: &[u64; 31],
    vectors: [u128; 3],
    (fpcr, fpsr): (u32, u32),
) -> Outcome {
    m.core.cpu.regs = *regs;
    m.core.cpu.vectors[..3].copy_from_slice(&vectors);
    m.core.cpu.fpcr = fpcr;
    m.core.cpu.fpsr = fpsr;
    m.core.cpu.nzcv = 0;
    m.core.cpu.pc = entry;
    m.core.host_check_at = u64::MAX;
    let trap = loop {
        match dispatch(&mut m.core) {
            Ok(()) => continue,
            Err(trap) => break trap,
        }
    };
    m.core.settle_block_trap();
    assert!(
        matches!(trap, Trap::SupervisorCall { number: 7 }),
        "{entry:#x}: {trap:?}"
    );
    Outcome {
        regs: m.core.cpu.regs,
        vectors: std::array::from_fn(|n| m.core.cpu.vectors[n]),
        fpcr: m.core.cpu.fpcr,
        fpsr: m.core.cpu.fpsr,
        nzcv: m.core.cpu.nzcv,
    }
}

#[test]
fn fp_status_locals_match_the_core_fields() {
    let words = fp_status_code::WORDS;
    let code_size = (words.len() as u64 * 4).next_multiple_of(0x1000);
    let image: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let mut m = Machine::with_layout(fp_status_code::BASE, &image, code_size, DATA, DATA_SIZE);
    let mut random = Random(0xf9c5_7a7e_0000_0001);
    let lanes = |random: &mut Random| -> u128 {
        (0..4).fold(0u128, |vector, lane| {
            let value = FLOATS[random.next() as usize % FLOATS.len()];
            vector | u128::from(value) << (32 * lane)
        })
    };
    // FPCR: FZ and DN under round to nearest (AH, FZ16 and the directed
    // rounding modes trap in some helpers, unimplemented);
    // FPSR: cumulative flags.
    let fpcrs = [0, 0x0100_0000, 0x0200_0000, 0x0300_0000];
    let mut runs = 0;
    let mut failures = Vec::new();
    for &entry in fp_status_code::ENTRIES {
        for _ in 0..200 {
            let mut regs: [u64; 31] = std::array::from_fn(|_| random.value());
            regs[3] = random.value() & 0x9f; // an FPSR value for MSR
            regs[5] = fpcrs[random.next() as usize % fpcrs.len()];
            let vectors = [lanes(&mut random), lanes(&mut random), lanes(&mut random)];
            let status = (
                fpcrs[random.next() as usize % fpcrs.len()] as u32,
                (random.value() & 0x9f) as u32,
            );
            let expected = run(
                fp_status_reference::dispatch,
                &mut m,
                entry,
                &regs,
                vectors,
                status,
            );
            let actual = run(
                fp_status_code::dispatch,
                &mut m,
                entry,
                &regs,
                vectors,
                status,
            );
            runs += 1;
            if actual != expected {
                failures.push(format!(
                    "program at {entry:#x}: expected {expected:x?}, got {actual:x?}"
                ));
                break;
            }
        }
    }
    println!("{runs} runs, {} mismatching programs", failures.len());
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
