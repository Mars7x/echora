//! Split emission installs its window table for the whole process
//! (`echora_guest::install_windows`), and `Core::call` uses that table for
//! every program. It runs in its own test binary, so the table cannot reach
//! the single-dispatch programs in `cpu.rs`.
#![cfg(any(target_os = "linux", windows))]
#![allow(clippy::unwrap_used)]

use echora_aot_tests::*;

#[test]
fn split_emission_runs_the_same_program() {
    let mut m = Machine::new();
    assert!(matches!(
        split_call_return::run(&mut m),
        Err(Trap::Exit { code: 0 })
    ));
    assert_eq!(m.core.cpu.regs[3], 17);
    assert_eq!(m.core.cpu.regs[30], 0x1004);
    assert_eq!(m.core.steps, 4);
}
