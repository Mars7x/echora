//! PLT stubs inlined at their callers (`EmitOptions::stub_windows` off):
//! the synthetic PLT of `build.rs` (`plt_code`) runs with its GOT holding
//! the predicted words, and with slots rewritten to other functions; with
//! the step budget and a host boundary falling on an inlined stub, the run
//! stops at the stub and resumes there through its own window. Registers
//! and the instruction count must be what the guest code gives either way.
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
    include!(concat!(env!("OUT_DIR"), "/lifted_plt.rs"));
}
use lifted::plt_code;

const DATA_SIZE: u64 = 0x1000;
/// Instructions the caller runs: its own 7, four per stub run (four), the
/// tail-branching function's 1 and two per function run (four).
const STEPS: u64 = 7 + 4 * 4 + 1 + 2 * 4;

fn machine(got: [u64; 3]) -> Machine {
    let words = plt_code::WORDS;
    let image: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let mut m = Machine::with_layout(
        plt_code::BASE,
        &image,
        (image.len() as u64).next_multiple_of(0x1000),
        plt_code::GOT,
        DATA_SIZE,
    );
    let slots: Vec<u8> = got.iter().flat_map(|word| word.to_le_bytes()).collect();
    m.memory.write(plt_code::GOT, &slots).unwrap();
    plt_code::install();
    m.core.cpu.pc = plt_code::BASE;
    m
}

/// Runs to the final SVC, through every host boundary; the traps that
/// stopped it on the way, with the PC each left.
fn run(m: &mut Machine) -> Vec<(Trap, u64)> {
    let mut stops = Vec::new();
    loop {
        match plt_code::dispatch(&mut m.core) {
            Ok(()) => continue,
            Err(Trap::SupervisorCall { number: 7 }) => {
                m.core.settle_block_trap();
                return stops;
            }
            Err(trap @ (Trap::HostBoundary | Trap::Budget { .. })) => {
                m.core.settle_block_trap();
                stops.push((trap, m.core.cpu.pc));
                m.core.max_steps = u64::MAX;
                m.core.host_check_at = u64::MAX;
                m.core.rearm();
            }
            Err(trap) => panic!("unexpected trap {trap:?} at {:#x}", m.core.cpu.pc),
        }
    }
}

#[test]
fn every_stub_is_recognised_and_the_in_region_call_is_direct() {
    // Three stubs, all predicted and inlined; BLs to each, one of them (to
    // FA, in the caller's region) with a direct window call; one B.
    assert_eq!(plt_code::REPORT, (3, 3, 3, 3, 1, 1));
    assert_eq!(plt_code::DIRECT_CALLS, 1);
    // Running the predicted window for another word would still be right
    // (`Core::call_window` leaves a window that does not hold the PC), so
    // only the emitted guard shows the call is skipped then.
    const { assert!(plt_code::GUARDED) };
}

#[test]
fn stubs_whose_got_word_is_the_predicted_one_reach_their_targets() {
    let mut m = machine([plt_code::FA, plt_code::FB, plt_code::FC]);
    assert!(run(&mut m).is_empty());
    let regs = &m.core.cpu.regs;
    // x0: 5 + 1 (FA), doubled (FB); x1: x0 added twice (FC).
    assert_eq!(regs[0], 12);
    assert_eq!(regs[1], 24);
    assert_eq!(regs[16], plt_code::GOT + 16);
    assert_eq!(regs[17], plt_code::FC);
    assert_eq!(regs[30], plt_code::BASE + 24);
    assert_eq!(m.core.steps, STEPS);
}

#[test]
fn stubs_whose_got_word_was_rewritten_follow_the_word() {
    // The in-region slot now names FD (another region), the second FE (in
    // the caller's region, but not the prediction).
    let mut m = machine([plt_code::FD, plt_code::FE, plt_code::FC]);
    assert!(run(&mut m).is_empty());
    let regs = &m.core.cpu.regs;
    // x0: 5 - 2 (FD); x2 from FE; x1: x0 added twice (FC).
    assert_eq!(regs[0], 3);
    assert_eq!(regs[2], 9);
    assert_eq!(regs[1], 6);
    assert_eq!(regs[17], plt_code::FC);
    assert_eq!(m.core.steps, STEPS);
}

#[test]
fn a_slot_rewritten_to_a_function_of_the_caller_s_window_is_not_mistaken_for_the_prediction() {
    // The first slot names FE: in the caller's own window, like the
    // predicted FA, but another function.
    let mut m = machine([plt_code::FE, plt_code::FB, plt_code::FC]);
    assert!(run(&mut m).is_empty());
    let regs = &m.core.cpu.regs;
    assert_eq!(regs[2], 9);
    assert_eq!(regs[0], 10);
    assert_eq!(regs[1], 20);
    assert_eq!(m.core.steps, STEPS);
}

#[test]
fn the_budget_running_out_at_an_inlined_stub_stops_at_the_stub() {
    // The caller's first block (MOVZ, MOVZ, BL) leaves too little for the
    // stub's four instructions.
    let mut m = machine([plt_code::FA, plt_code::FB, plt_code::FC]);
    m.core.max_steps = 6;
    m.core.rearm();
    let stops = run(&mut m);
    assert_eq!(stops.len(), 1);
    assert!(matches!(stops[0].0, Trap::Budget { pc, steps: 3 } if pc == plt_code::STUBS[0]));
    assert_eq!(stops[0].1, plt_code::STUBS[0]);
    assert_eq!(m.core.cpu.regs[0], 12);
    assert_eq!(m.core.cpu.regs[1], 24);
    assert_eq!(m.core.steps, STEPS);
}

#[test]
fn a_host_boundary_at_an_inlined_stub_resumes_at_the_stub() {
    // Every stub's start is a boundary the host may take: with the host
    // check due after the caller's first block, the run returns there and
    // resumes the stub through its own window.
    for (due, stub) in [(3, 0), (10, 1)] {
        let mut m = machine([plt_code::FA, plt_code::FB, plt_code::FC]);
        m.core.host_check_at = due;
        m.core.rearm();
        let stops = run(&mut m);
        assert!(
            matches!(stops.as_slice(), [(Trap::HostBoundary, pc)] if *pc == plt_code::STUBS[stub]),
            "{stops:x?}"
        );
        assert_eq!(m.core.cpu.regs[0], 12);
        assert_eq!(m.core.cpu.regs[1], 24);
        assert_eq!(m.core.steps, STEPS);
    }
}
