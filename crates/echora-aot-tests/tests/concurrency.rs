//! Plain guest loads and stores seen from another host thread.
//!
//! Generated code reads and writes guest memory with non-atomic
//! `read_unaligned`/`write_unaligned`, so the host compiler may cache,
//! merge or reorder those accesses as if no other thread existed. Arm
//! requires a store to become visible to other observers in finite time,
//! so a polling loop on a plain LDR must see another thread's plain store
//! eventually. Here a second Core on the same Memory, on its own host
//! thread, sets the flag the loop polls. The window keeps running without
//! returning to the host (no slice ends), so a load the compiler hoisted
//! out of the loop would spin forever.
#![cfg(any(target_os = "linux", windows))]
#![allow(clippy::unwrap_used)]

use std::sync::mpsc;
use std::time::Duration;

use echora_aot_tests::{Core, Machine, Trap, concurrency_code};

const DATA: u64 = 0x200_0000;
const FLAG: u64 = DATA;
const OTHER: u64 = DATA + 0x40;

fn machine() -> Machine {
    let words = concurrency_code::WORDS;
    let image: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let size = (image.len() as u64).next_multiple_of(0x1000);
    concurrency_code::install();
    Machine::with_layout(concurrency_code::BASE, &image, size, DATA, 0x1000)
}

/// Runs from `entry` until SVC #7, resuming after host boundaries (the
/// spin hint ends a slice). Returns the steps run.
fn run(core: &mut Core, entry: u64) -> u64 {
    core.cpu.pc = entry;
    core.max_steps = u64::MAX;
    core.host_check_at = u64::MAX;
    core.rearm();
    loop {
        match concurrency_code::dispatch(core) {
            Ok(()) => {}
            Err(Trap::HostBoundary) => {
                core.spin_polls = 0;
                core.host_check_at = u64::MAX;
                core.rearm();
            }
            Err(Trap::SupervisorCall { number: 7 }) => {
                core.settle_block_trap();
                return core.steps;
            }
            Err(trap) => panic!("{trap:?} at {:#x}", core.cpu.pc),
        }
    }
}

/// Polls the flag with the program at `entry` on one thread while another
/// sets it after a while. Fails if the poll never sees the store.
fn poll_ends_after_another_threads_store(entry: u64) {
    let mut m = machine();
    let mut setter = Core::new(m.memory.view());
    m.core.cpu.regs[0] = FLAG;
    let (done, finished) = mpsc::channel();
    let mut poller = m.core;
    std::thread::scope(|scope| {
        scope.spawn(move || {
            let steps = run(&mut poller, entry);
            let _ = done.send((steps, poller.cpu.regs[2]));
        });
        scope.spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            setter.memory.store_u32(FLAG, 1);
        });
        match finished.recv_timeout(Duration::from_secs(10)) {
            Ok((steps, count)) => assert!(steps > 0 && count < steps, "{steps} {count}"),
            // The poller still spins: the process ends when the test does.
            Err(_) => {
                eprintln!("the poll at {entry:#x} never saw the other thread's store");
                std::process::exit(1);
            }
        }
    });
}

#[test]
fn a_pure_poll_sees_another_threads_store() {
    poll_ends_after_another_threads_store(concurrency_code::ENTRIES[0]);
}

#[test]
fn a_counted_poll_sees_another_threads_store() {
    poll_ends_after_another_threads_store(concurrency_code::ENTRIES[1]);
}

#[test]
fn a_store_then_a_load_of_another_address_run_in_order() {
    // Single-threaded meaning only; the machine code is what tells whether
    // the compiler kept both accesses (see docs/testing.md).
    let mut m = machine();
    m.core.cpu.regs[0] = FLAG;
    m.core.cpu.regs[1] = OTHER;
    m.core.cpu.regs[2] = 5;
    m.memory.write(OTHER, &9u32.to_le_bytes()).unwrap();
    run(&mut m.core, concurrency_code::ENTRIES[2]);
    assert_eq!(m.core.cpu.regs[3], 9);
    let mut flag = [0; 4];
    m.memory.read(FLAG, &mut flag).unwrap();
    assert_eq!(u32::from_le_bytes(flag), 5);
}

#[test]
fn two_loads_of_one_address_read_it() {
    let mut m = machine();
    m.core.cpu.regs[0] = FLAG;
    m.memory.write(FLAG, &3u32.to_le_bytes()).unwrap();
    run(&mut m.core, concurrency_code::ENTRIES[3]);
    assert_eq!((m.core.cpu.regs[1], m.core.cpu.regs[2]), (3, 3));
}
