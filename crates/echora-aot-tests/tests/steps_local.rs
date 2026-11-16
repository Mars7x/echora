//! The step count in a window local (`EmitOptions::steps_local`) against
//! per-block charging, on `build.rs`'s deferred-charges program (a loop
//! through a diamond, a join, a block leaving for another window, a call
//! of a leaf in a third, an SVC): final registers and steps, every host
//! boundary's PC and step count with the host check due at every step,
//! and the budget's trap at every budget. No window table is installed, so
//! each module's calls run through its own dispatcher (its own code only).
#![cfg(any(target_os = "linux", windows))]
#![allow(clippy::unwrap_used)]

use echora_aot_tests::{Core, Machine, Trap};

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
    include!(concat!(env!("OUT_DIR"), "/lifted_deferred_charge.rs"));
}
use lifted::{deferred_charge_code, deferred_charge_reference, steps_local_code};

type Dispatch = fn(&mut Core) -> Result<(), Trap>;

#[derive(Debug, PartialEq, Eq)]
struct Run {
    regs: [u64; 31],
    steps: u64,
    /// Each host boundary and budget stop: the trap, its PC and the steps.
    stops: Vec<(String, u64, u64)>,
}

fn run(dispatch: Dispatch, count: u64, flip: u64, due: u64, budget: u64) -> Run {
    let words = deferred_charge_code::WORDS;
    let image: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let mut m = Machine::with_layout(
        deferred_charge_code::BASE,
        &image,
        (image.len() as u64).next_multiple_of(0x1000),
        0x80_0000,
        0x1000,
    );
    m.core.cpu.regs[0] = count;
    m.core.cpu.regs[1] = flip;
    m.core.cpu.pc = deferred_charge_code::BASE;
    m.core.host_check_at = due;
    m.core.max_steps = budget;
    m.core.rearm();
    let mut stops = Vec::new();
    loop {
        match dispatch(&mut m.core) {
            Ok(()) => continue,
            Err(Trap::SupervisorCall { number: 7 }) => break,
            // The budget ends the run (it can stop in a block inlined into
            // another, which has no arm to resume at).
            Err(trap @ Trap::Budget { .. }) => {
                m.core.settle_block_trap();
                stops.push((format!("{trap:?}"), m.core.cpu.pc, m.core.steps));
                break;
            }
            Err(trap @ Trap::HostBoundary) => {
                m.core.settle_block_trap();
                stops.push((format!("{trap:?}"), m.core.cpu.pc, m.core.steps));
                m.core.host_check_at = u64::MAX;
                m.core.max_steps = u64::MAX;
                m.core.rearm();
            }
            Err(trap) => panic!("unexpected trap {trap:?}"),
        }
    }
    m.core.settle_block_trap();
    Run {
        regs: m.core.cpu.regs,
        steps: m.core.steps,
        stops,
    }
}

#[test]
fn a_local_step_count_charges_and_stops_exactly_as_the_core_s() {
    for count in [1, 2, 7] {
        for flip in [0, 1] {
            let expected = run(
                deferred_charge_reference::dispatch,
                count,
                flip,
                u64::MAX,
                u64::MAX,
            );
            for limit in (0..expected.steps + 2).chain([u64::MAX]) {
                for (due, budget) in [(limit, u64::MAX), (u64::MAX, limit)] {
                    let reference = run(
                        deferred_charge_reference::dispatch,
                        count,
                        flip,
                        due,
                        budget,
                    );
                    let local = run(steps_local_code::dispatch, count, flip, due, budget);
                    assert_eq!(
                        local, reference,
                        "count {count} flip {flip} due {due} budget {budget}"
                    );
                }
            }
        }
    }
}
