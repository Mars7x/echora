//! Scheduling across guest cores folded onto one worker.
//!
//! With W workers, worker `w` serves every guest core `c` with
//! `c % W == w`, so with two workers cores 0 and 2 share worker 0. On
//! Horizon each core has its own run queue: a thread pinned to core 2 runs
//! however busy a better-priority thread pinned to core 0 is. Here the
//! main thread (priority 44, core 0) spins on a flag that a thread it
//! starts (priority 50, core 2) sets, and the run must end.
//!
//! "Guest code" here is a dispatch function standing in for translated
//! code: one block per call, ticked and ended at block boundaries as
//! generated code does, with SVCs returned as traps.
#![cfg(any(target_os = "linux", windows))]
#![allow(clippy::unwrap_used)]

use echora_guest::memory::{
    EXECUTE, PAGE_SIZE, READ, STATE_CODE, STATE_CODE_DATA, STATE_NORMAL, STATE_TLS, WRITE,
};
use echora_guest::{Core, Trap};
use echora_runtime::runtime::{Layout, Machine};

const CODE: u64 = 0x7100000000;
/// The main thread: create the setter, start it, then spin.
const MAIN_CREATE: u64 = CODE;
const MAIN_START: u64 = CODE + 0x10;
const MAIN_SPIN: u64 = CODE + 0x20;
const MAIN_EXIT: u64 = CODE + 0x30;
/// The setter: store the flag, exit.
const SETTER: u64 = CODE + 0x100;
const SETTER_EXIT: u64 = CODE + 0x110;
const DATA: u64 = 0x8000;
/// The flag, in a page of its own (the stack alias makes `DATA`
/// inaccessible at its own address).
const FLAG: u64 = 0x10000;

fn machine(workers: usize, max_steps: u64) -> Machine {
    let layout = Layout {
        process_core_mask: Some(0x3f),
        process_priority_mask: Some(((1u64 << 60) - 1) & !((1u64 << 28) - 1)),
        process_ideal_core_id: 0,
        initial_used_memory: 0x4000,
        workers,
        parallel_slice: 1000,
        scheduler_slice: 1000,
        ..Layout::default()
    };
    let mut m = Machine::new(layout, max_steps);
    m.memory
        .map(CODE, PAGE_SIZE, READ | EXECUTE, STATE_CODE, "code")
        .unwrap();
    m.memory
        .map(m.layout.tls_base, PAGE_SIZE, READ | WRITE, STATE_TLS, "TLS")
        .unwrap();
    m.register_main_thread_tls().unwrap();
    m.memory
        .map(
            DATA,
            PAGE_SIZE,
            READ | WRITE,
            STATE_CODE_DATA,
            "stack and data",
        )
        .unwrap();
    m.memory
        .map_stack_alias(m.layout.stack_region_base, DATA, PAGE_SIZE)
        .unwrap();
    m.memory
        .map(FLAG, PAGE_SIZE, READ | WRITE, STATE_NORMAL, "flag")
        .unwrap();
    m.core.cpu.pc = MAIN_CREATE;
    m.core.cpu.sp = m.layout.stack_region_base + 0x400;
    m.core.cpu.tpidrro_el0 = m.layout.tls_base;
    m
}

/// One block of the program at the core's PC.
fn block(core: &mut Core) -> Result<(), Trap> {
    let pc = core.cpu.pc;
    core.tick_block_at(pc, 1)?;
    match pc {
        MAIN_CREATE => {
            // CreateThread(entry, arg, stack top, priority 50, core 2).
            core.cpu.regs[1] = SETTER;
            core.cpu.regs[2] = 0;
            core.cpu.regs[3] = 0x7ff0000000 - 0x200000 + 0x400;
            core.cpu.regs[4] = 50;
            core.cpu.regs[5] = 2;
            return Err(Trap::SupervisorCall { number: 0x08 });
        }
        MAIN_START => {
            assert_eq!(core.cpu.regs[0], 0, "CreateThread failed");
            core.cpu.regs[0] = core.cpu.regs[1];
            return Err(Trap::SupervisorCall { number: 0x09 });
        }
        MAIN_SPIN => {
            let flag = core.memory.load_u32(FLAG);
            core.spin_hint();
            core.cpu.pc = if flag == 0 { MAIN_SPIN } else { MAIN_EXIT };
        }
        MAIN_EXIT => return Err(Trap::SupervisorCall { number: 0x07 }),
        SETTER => {
            core.memory.store_u32(FLAG, 1);
            core.cpu.pc = SETTER_EXIT;
        }
        SETTER_EXIT => return Err(Trap::SupervisorCall { number: 0x0a }),
        other => return Err(Trap::MissingBlock { pc: other }),
    }
    core.finish_block()
}

/// SVCs return to the next "instruction": the runtime adds 4 to the PC,
/// so each SVC block sits 4 below the block it returns to.
fn dispatch(core: &mut Core) -> Result<(), Trap> {
    let pc = core.cpu.pc;
    let at = match pc {
        _ if pc == MAIN_CREATE + 4 => MAIN_START,
        _ if pc == MAIN_START + 4 => MAIN_SPIN,
        _ => pc,
    };
    core.cpu.pc = at;
    block(core)
}

fn run(workers: usize) -> Trap {
    let mut m = machine(workers, 20_000_000);
    m.run_generated(dispatch).unwrap_err()
}

#[test]
fn one_worker_lets_the_setter_run() {
    let trap = run(1);
    assert!(matches!(trap, Trap::Exit { .. }), "{trap:?}");
}

/// Currently livelocks: worker 0 serves cores 0 and 2 and runs only the
/// best-priority runnable thread among both, so the spinning main thread
/// (44) keeps the processor and the setter (50) never runs. The run ends
/// at the step budget instead of the exit. See docs/multicore-review.md,
/// finding 1, and this test's commit message for the fix.
#[test]
#[ignore = "livelocks: strict priority across guest cores folded onto one worker"]
fn two_workers_let_a_worse_priority_thread_on_another_folded_core_run() {
    let trap = run(2);
    assert!(matches!(trap, Trap::Exit { .. }), "{trap:?}");
}
