//! The helper fallback counters (`EmitOptions::count_fallbacks`):
//! `build.rs` (`fallback_count_code`) lifts FADD S0, S1, S2, FSQRT V0.4S,
//! V1.4S and FMLA V0.4S, V1.4S, V2.4S with them. Operands the inlined
//! code takes natively count nothing; a NaN operand counts one call of
//! the helper each inlined form falls back to (the whole helper for FADD,
//! the scalar form of `vec_fp_unary` per lane for FSQRT, the single-lane
//! helper per lane for FMLA). FPCR.FZ and FPSR.IXC are set, as in the
//! game.
#![cfg(any(target_os = "linux", windows))]
#![allow(clippy::unwrap_used)]

use echora_aot_tests::{Machine, Trap, fallback_count_code as code};

fn machine() -> Machine {
    let words = code::WORDS;
    let image: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    Machine::with_layout(code::BASE, &image, 0x1000, 0x1000_0000, 0x1000)
}

fn run(m: &mut Machine, entry: u64, vectors: [u128; 3]) {
    m.core.cpu.vectors[..3].copy_from_slice(&vectors);
    // Sticky inexact and flush-to-zero, as the game runs (the inlined
    // scalar arithmetic's fast path needs both).
    m.core.cpu.fpsr = 0x10;
    m.core.cpu.fpcr = 0x0100_0000;
    m.core.cpu.pc = entry;
    m.core.host_check_at = u64::MAX;
    m.core.rearm();
    let trap = loop {
        match code::dispatch(&mut m.core) {
            Ok(()) => continue,
            Err(trap) => break trap,
        }
    };
    assert!(
        matches!(trap, Trap::SupervisorCall { number: 7 }),
        "{trap:?}"
    );
}

/// The counter of `helper`.
fn count(helper: &str) -> u64 {
    let index = code::FALLBACK_HELPERS
        .iter()
        .position(|name| *name == helper)
        .unwrap();
    code::fallback_counts()[index]
}

#[test]
fn fallbacks_are_counted_per_helper_and_native_operands_count_nothing() {
    let mut m = machine();
    let ones = u128::from_le_bytes([[0x00, 0x00, 0x80, 0x3f]; 4].concat().try_into().unwrap());
    let nan_lane = ones & !u128::from(u32::MAX) | 0x7fc0_0000;
    let [fadd, fsqrt, fmla] = [code::ENTRIES[0], code::ENTRIES[1], code::ENTRIES[2]];
    let before = code::fallback_counts();
    run(&mut m, fadd, [0, ones, ones]);
    run(&mut m, fsqrt, [0, ones, 0]);
    run(&mut m, fmla, [ones, ones, ones]);
    assert_eq!(code::fallback_counts(), before);
    // One NaN lane each.
    run(&mut m, fadd, [0, nan_lane, ones]);
    assert_eq!(
        count("fp_binary"),
        before[code::FALLBACK_HELPERS
            .iter()
            .position(|n| *n == "fp_binary")
            .unwrap()]
            + 1
    );
    run(&mut m, fsqrt, [0, nan_lane, 0]);
    assert_eq!(count("vec_fp_unary"), 1);
    run(&mut m, fmla, [ones, nan_lane, ones]);
    assert_eq!(count("vec_fp_lane"), 1);
    assert_eq!(count("other"), 0);
}
