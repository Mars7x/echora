//! Calls between windows before `install()` has run: no window table and,
//! with direct calls (`--features direct-calls`, with `small-regions` for
//! calls into other regions), empty call slots. Every BL must then fall
//! back to running its callee through the dispatcher, with the same
//! result. Its own test binary, so nothing in the process installs.
#![cfg(any(target_os = "linux", windows))]
#![allow(clippy::unwrap_used)]

use echora_aot_tests::kernels::{self, KERNEL_BASE, KERNEL_ENTRIES, KERNEL_SIZE};
use echora_aot_tests::{Machine, Trap, kernel_code};

#[test]
fn calls_run_before_install() {
    const ITERATIONS: u64 = 1000;
    let image = kernels::image();
    let mut m = Machine::with_layout(KERNEL_BASE, &image, KERNEL_SIZE, 0x20_0000, 0x1_0000);
    let entry = KERNEL_ENTRIES
        .iter()
        .find(|(name, _)| *name == "calls")
        .unwrap()
        .1;
    m.core.cpu.pc = KERNEL_BASE + entry;
    m.core.cpu.regs[0] = ITERATIONS;
    m.core.cpu.regs[1] = 7;
    m.core.cpu.regs[2] = 11;
    m.core.host_check_at = u64::MAX;
    let trap = loop {
        if let Err(trap) = kernel_code::dispatch(&mut m.core) {
            break trap;
        }
    };
    m.core.settle_block_trap();
    assert!(
        matches!(trap, Trap::SupervisorCall { number: 7 }),
        "{trap:?}"
    );
    // leaf_add: x1 += 1; leaf_mix: x2 = (x2 ^ (x1 << 1)) + 3.
    let (mut x1, mut x2) = (7u64, 11u64);
    for _ in 0..ITERATIONS {
        x1 = x1.wrapping_add(1);
        x2 = (x2 ^ (x1 << 1)).wrapping_add(3);
    }
    assert_eq!(m.core.cpu.regs[0], 0);
    assert_eq!(m.core.cpu.regs[1], x1);
    assert_eq!(m.core.cpu.regs[2], x2);
}
