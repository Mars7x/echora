//! Deferred block charges (`EmitOptions::deferred_charges`) against the
//! same program charged per block (`build.rs`'s `deferred_charge_code`):
//! registers and the step count at the final SVC must match, also when a
//! host boundary falls at any point of the run and the host resumes it.
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
use lifted::{deferred_charge_code, deferred_charge_reference};

type Dispatch = fn(&mut Core) -> Result<(), Trap>;

/// Runs to the SVC from X0 = `count`, X1 = `flip`, with the first host
/// check due at `due` steps (then none): registers, steps and the host
/// boundaries taken.
fn run(
    dispatch: Dispatch,
    install: fn(),
    count: u64,
    flip: u64,
    due: u64,
) -> ([u64; 31], u64, usize) {
    let words = deferred_charge_code::WORDS;
    let image: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let mut m = Machine::with_layout(
        deferred_charge_code::BASE,
        &image,
        (image.len() as u64).next_multiple_of(0x1000),
        0x80_0000,
        0x1000,
    );
    install();
    m.core.cpu.regs[0] = count;
    m.core.cpu.regs[1] = flip;
    m.core.cpu.pc = deferred_charge_code::BASE;
    m.core.host_check_at = due;
    m.core.rearm();
    let mut boundaries = 0;
    loop {
        match dispatch(&mut m.core) {
            Ok(()) => continue,
            Err(Trap::SupervisorCall { number: 7 }) => break,
            Err(Trap::HostBoundary) => {
                boundaries += 1;
                m.core.settle_block_trap();
                m.core.host_check_at = u64::MAX;
                m.core.rearm();
            }
            Err(trap) => panic!("unexpected trap {trap:?}"),
        }
    }
    m.core.settle_block_trap();
    (m.core.cpu.regs, m.core.steps, boundaries)
}

#[test]
fn some_block_charges_are_deferred() {
    const { assert!(deferred_charge_code::DEFERRED > 0) };
    assert_eq!(deferred_charge_reference::DEFERRED, 0);
}

#[test]
fn deferred_charges_count_every_instruction() {
    // Both modules hold the same window addresses; the table installed
    // first serves both (their windows run the same guest code).
    for count in [1, 2, 7, 20] {
        for flip in [0, 1] {
            let (regs, steps, _) = run(
                deferred_charge_reference::dispatch,
                deferred_charge_reference::install,
                count,
                flip,
                u64::MAX,
            );
            for due in (0..steps + 2).chain([u64::MAX]) {
                let (deferred_regs, deferred_steps, _) = run(
                    deferred_charge_code::dispatch,
                    deferred_charge_code::install,
                    count,
                    flip,
                    due,
                );
                assert_eq!(deferred_regs, regs, "count {count} flip {flip} due {due}");
                assert_eq!(deferred_steps, steps, "count {count} flip {flip} due {due}");
            }
        }
    }
}
