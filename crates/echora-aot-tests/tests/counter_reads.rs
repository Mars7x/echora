//! A function reading the physical counter (derived from the step count),
//! as windows and as a structured function (`regalloc`, `build.rs`'s
//! `counter_code`): the same registers, steps and host boundaries, whole
//! and in slices of every length up to 23.
#![cfg(any(target_os = "linux", windows))]
#![allow(clippy::unwrap_used)]

use echora_aot_tests::{Core, Machine, Trap};

#[allow(
    missing_debug_implementations,
    clippy::all,
    dead_code,
    unused_imports,
    unused_parens,
    unreachable_code
)]
mod lifted {
    include!(concat!(env!("OUT_DIR"), "/lifted_counter.rs"));
}
use lifted::{counter_regalloc_code, counter_window_code};

/// Registers, steps and every host boundary (PC, steps) of a run of 40
/// calls with a boundary every `slice` steps (None: none).
fn run(
    dispatch: fn(&mut Core) -> Result<(), Trap>,
    slice: Option<u64>,
) -> ([u64; 31], u64, Vec<(u64, u64)>) {
    let words = counter_window_code::WORDS;
    let image: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let mut m = Machine::with_layout(
        counter_window_code::BASE,
        &image,
        (image.len() as u64).next_multiple_of(0x1000),
        0x80_0000,
        0x1000,
    );
    m.core.cpu.regs[0] = 40;
    m.core.cpu.pc = counter_window_code::BASE;
    // Guest time follows the steps (one tick each), as with
    // ECHORA_VIRTUAL_CLOCK.
    m.core.physical_counter.instructions_per_second = Some(31_250_000);
    m.core.max_steps = u64::MAX;
    m.core.host_check_at = slice.unwrap_or(u64::MAX);
    m.core.rearm();
    let mut boundaries = Vec::new();
    loop {
        match dispatch(&mut m.core) {
            Ok(()) => continue,
            Err(Trap::SupervisorCall { number: 7 }) => break,
            Err(Trap::HostBoundary) => {
                boundaries.push((m.core.cpu.pc, m.core.steps));
                m.core.host_check_at = m.core.steps + slice.unwrap();
                m.core.rearm();
            }
            Err(trap) => panic!("unexpected trap {trap:?} at {:#x}", m.core.cpu.pc),
        }
    }
    m.core.settle_block_trap();
    (m.core.cpu.regs, m.core.steps, boundaries)
}

#[test]
fn counter_reads_see_the_same_steps_in_structured_functions() {
    const { assert!(counter_regalloc_code::STRUCTURED_FUNCTIONS > 0) };
    const { assert!(counter_window_code::STRUCTURED_FUNCTIONS == 0) };
    counter_window_code::install();
    counter_regalloc_code::install();
    for slice in std::iter::once(None).chain((1..24).map(Some)) {
        let windows = run(counter_window_code::dispatch, slice);
        let structured = run(counter_regalloc_code::dispatch, slice);
        assert_eq!(structured, windows, "slice {slice:?}");
        assert_eq!(windows.0[3], 40);
    }
}
