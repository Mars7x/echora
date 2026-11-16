//! Register locals stored around calls only where they may have changed
//! (`RegisterLocals::rewrite`): the callee of `build.rs`'s `spill_code`
//! reads X1 (written by its callers before the call), X2 and X4 (written
//! after it, so needed by a loop's later passes), and X7 (never touched by
//! a caller). Its sum in X6 says which values it saw.
#![cfg(any(target_os = "linux", windows))]
#![allow(clippy::unwrap_used)]

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
    include!(concat!(env!("OUT_DIR"), "/lifted_spill.rs"));
}
use lifted::spill_code;

fn run(entry: u64, registers: &[(usize, u64)]) -> [u64; 31] {
    let words = spill_code::WORDS;
    let image: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let mut m = Machine::with_layout(
        spill_code::BASE,
        &image,
        (image.len() as u64).next_multiple_of(0x1000),
        0x80_0000,
        0x1000,
    );
    spill_code::install();
    for &(n, value) in registers {
        m.core.cpu.regs[n] = value;
    }
    m.core.cpu.pc = entry;
    loop {
        match spill_code::dispatch(&mut m.core) {
            Ok(()) => continue,
            Err(Trap::SupervisorCall { number: 7 }) => break,
            Err(trap) => panic!("unexpected trap {trap:?} at {:#x}", m.core.cpu.pc),
        }
    }
    m.core.cpu.regs
}

#[test]
fn a_looping_caller_s_later_calls_see_what_its_previous_pass_wrote() {
    let regs = run(
        spill_code::LOOPING,
        &[(1, 10), (2, 20), (4, 30), (5, 3), (6, 0), (7, 100)],
    );
    // Pass i: X1 = 10 + i, X2 = 20 + 5 (i - 1), X4 = 30 + 7 (i - 1).
    assert_eq!(
        regs[6],
        (11 + 20 + 30 + 100) + (12 + 25 + 37 + 100) + (13 + 30 + 44 + 100)
    );
    assert_eq!((regs[1], regs[2], regs[4], regs[5]), (13, 35, 51, 0));
}

#[test]
fn a_call_stores_only_the_registers_written_before_it() {
    // X2 is written after the call, so its value at the call is the one
    // the Core already holds.
    let expected = (true, !spill_code::SELECTIVE);
    assert_eq!(spill_code::STRAIGHT_STORES, expected);
    let regs = run(
        spill_code::STRAIGHT,
        &[(1, 10), (2, 20), (4, 30), (6, 0), (7, 100)],
    );
    assert_eq!(regs[6], 11 + 20 + 30 + 100);
    assert_eq!((regs[1], regs[2], regs[3]), (11, 25, 161));
}

#[test]
fn a_call_in_one_branch_leaves_the_other_branch_s_stores_alone() {
    // X1 is written before a CBZ whose taken side calls; the other side's
    // call, in another arm, still needs X1 stored.
    for x8 in [0, 1] {
        let regs = run(
            spill_code::BRANCHING,
            &[(1, 10), (2, 20), (4, 30), (6, 0), (7, 100), (8, x8)],
        );
        assert_eq!(regs[6], 11 + 20 + 30 + 100, "X8 = {x8}");
    }
}
