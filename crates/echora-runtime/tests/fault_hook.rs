//! The fault module's panic hook after runs have ended.
//!
//! While guest code runs, a panic gets a snapshot of every worker's core
//! (`echora_runtime::fault`). Once a run is over its cores are gone: a later
//! panic on the same thread must reach the default hook with its message,
//! not read a freed core (SIGSEGV) or abort.
//!
//! The panicking half runs in a child process (this test binary again), so
//! the parent can check how it ended.
#![cfg(unix)]
#![allow(clippy::unwrap_used)]

use echora_guest::memory::{EXECUTE, PAGE_SIZE, READ, STATE_CODE, STATE_TLS, WRITE};
use echora_guest::{Core, Trap};
use echora_runtime::runtime::{Layout, Machine};

const CODE: u64 = 0x7100000000;
const CHILD: &str = "ECHORA_FAULT_HOOK_CHILD";
const MESSAGE: &str = "a panic after the runs";

/// A process whose main thread exits at once, on `workers` workers.
fn run(workers: usize) {
    let layout = Layout {
        process_core_mask: Some(0x3f),
        process_priority_mask: Some(((1u64 << 60) - 1) & !((1u64 << 28) - 1)),
        process_ideal_core_id: 0,
        initial_used_memory: 0x4000,
        workers,
        ..Layout::default()
    };
    let mut m = Machine::new(layout, 1_000_000);
    m.memory
        .map(CODE, PAGE_SIZE, READ | EXECUTE, STATE_CODE, "code")
        .unwrap();
    m.memory
        .map(m.layout.tls_base, PAGE_SIZE, READ | WRITE, STATE_TLS, "TLS")
        .unwrap();
    m.register_main_thread_tls().unwrap();
    m.core.cpu.pc = CODE;
    m.core.cpu.tpidrro_el0 = m.layout.tls_base;
    let exit = |core: &mut Core| -> Result<(), Trap> {
        core.tick_block_at(core.cpu.pc, 1)?;
        Err(Trap::SupervisorCall { number: 0x07 })
    };
    let trap = m.run_generated(exit).unwrap_err();
    assert!(matches!(trap, Trap::Exit { .. }), "{trap:?}");
}

#[test]
fn a_panic_after_runs_reports_its_message() {
    if std::env::var_os(CHILD).is_some() {
        // One worker watches the core on this thread; two run parallel
        // workers whose cores are gone once the run returns.
        run(1);
        run(2);
        run(1);
        panic!("{MESSAGE}");
    }
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "a_panic_after_runs_reports_its_message",
            "--exact",
            "--nocapture",
        ])
        .env(CHILD, "1")
        .env("RUST_BACKTRACE", "0")
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    // A failed test exits with 101; a crash ends by a signal instead.
    assert_eq!(
        output.status.code(),
        Some(101),
        "{:?}: {stderr}",
        output.status
    );
    assert!(stderr.contains(MESSAGE), "{stderr}");
    assert!(
        !stderr.contains("core slot"),
        "a snapshot of ended runs: {stderr}"
    );
}
