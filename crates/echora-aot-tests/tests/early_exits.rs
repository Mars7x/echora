//! Summarized calls of leaves that leave early (`build.rs`'s
//! `early_exit_code`): leaf A branches out of its region to code that reads
//! a register the caller held in a local across the call, and leaf B ends
//! the run with an SVC. Both must see the caller's value in the `Core`.
#![cfg(any(target_os = "linux", windows))]
#![allow(clippy::unwrap_used)]

use echora_aot_tests::{
    Core, Machine, Trap, early_exit_code, early_exit_table_code, two_level_code,
    two_level_table_code,
};

fn run(dispatch: fn(&mut Core) -> Result<(), Trap>, x5: u64) -> [u64; 31] {
    let words = early_exit_code::WORDS;
    let image: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let mut m = Machine::with_layout(
        early_exit_code::BASE,
        &image,
        (image.len() as u64).next_multiple_of(0x1000),
        0x80_0000,
        0x1000,
    );
    m.core.cpu.regs[1] = 10;
    m.core.cpu.regs[2] = 100;
    m.core.cpu.regs[5] = x5;
    m.core.cpu.pc = early_exit_code::BASE;
    loop {
        match dispatch(&mut m.core) {
            Ok(()) => continue,
            Err(Trap::SupervisorCall { number: 7 }) => break,
            Err(trap) => panic!("unexpected trap {trap:?}"),
        }
    }
    m.core.cpu.regs
}

#[test]
fn early_exits_see_the_caller_s_held_registers() {
    const { assert!(early_exit_code::SUMMARIZED_CALLS >= 2) };
    const { assert!(early_exit_table_code::SUMMARIZED_CALLS >= 2) };
    early_exit_table_code::install();
    for dispatch in [early_exit_code::dispatch, early_exit_table_code::dispatch] {
        for x5 in [0, 1] {
            let regs = run(dispatch, x5);
            // X1: 10 + 1 + 2; X6: X4 + X1 after leaf A; X2: + X4 through the
            // early exit.
            assert_eq!(regs[1], 13, "x5 {x5}");
            assert_eq!(regs[4], 7, "x5 {x5}");
            assert_eq!(regs[6], 18, "x5 {x5}");
            assert_eq!(regs[2], if x5 == 0 { 100 } else { 107 }, "x5 {x5}");
        }
    }
}

fn run_two_levels(dispatch: fn(&mut Core) -> Result<(), Trap>, x5: u64) -> [u64; 31] {
    let words = two_level_code::WORDS;
    let image: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let mut m = Machine::with_layout(
        two_level_code::BASE,
        &image,
        (image.len() as u64).next_multiple_of(0x1000),
        0x80_0000,
        0x1000,
    );
    m.core.cpu.regs[1] = 10;
    m.core.cpu.regs[2] = 100;
    m.core.cpu.regs[5] = x5;
    m.core.cpu.sp = 0x80_1000;
    m.core.cpu.pc = two_level_code::BASE;
    loop {
        match dispatch(&mut m.core) {
            Ok(()) => continue,
            Err(Trap::SupervisorCall { number: 7 }) => break,
            Err(trap) => panic!("unexpected trap {trap:?}"),
        }
    }
    m.core.cpu.regs
}

#[test]
fn an_inner_early_exit_sees_every_caller_s_held_registers() {
    // A's call of C and C's call of D are summarized, and C's returns from
    // C when D returns early.
    const { assert!(two_level_code::SUMMARIZED_CALLS >= 2) };
    const { assert!(two_level_code::PROPAGATING >= 1) };
    const { assert!(two_level_table_code::PROPAGATING >= 1) };
    two_level_table_code::install();
    for dispatch in [two_level_code::dispatch, two_level_table_code::dispatch] {
        for x5 in [0, 1] {
            let regs = run_two_levels(dispatch, x5);
            // X1: 10 + X8 (3) + 1 + X7 (5); X6: X4 + X1; X2: + X4 + X7
            // through the early exit.
            assert_eq!(regs[1], 19, "x5 {x5}");
            assert_eq!(regs[6], 26, "x5 {x5}");
            assert_eq!(regs[2], if x5 == 0 { 100 } else { 112 }, "x5 {x5}");
        }
    }
}
