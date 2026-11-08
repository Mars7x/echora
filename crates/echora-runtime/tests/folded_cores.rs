//! Scheduling across guest cores folded onto one worker.
//!
//! With W workers, worker `w` serves every guest core `c` with
//! `c % W == w`, so with two workers cores 0 and 2 share worker 0. On
//! Horizon each core has its own run queue: a thread pinned to core 2 runs
//! however busy a better-priority thread pinned to core 0 is. Here the
//! main thread (priority 44, core 0) spins on a flag that a thread it
//! starts (priority 50, on another core of worker 0) sets, and the run
//! must end. A second program checks that equal priorities on a folded
//! core take turns: a waiter spins (without a spin hint) for a flag a
//! second thread of its priority on its core sets.
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
/// The turns program: main creates and starts the waiter and the second
/// setter, then spins on `FLAG` (which the waiter sets).
const TURNS_CREATE_WAITER: u64 = CODE + 0x200;
const TURNS_START_WAITER: u64 = CODE + 0x210;
const TURNS_CREATE_SETTER: u64 = CODE + 0x220;
const TURNS_START_SETTER: u64 = CODE + 0x230;
/// Main's spin in the turns program, without a spin hint: its core always
/// takes its turn, so the waiter's core comes back to it from core 0.
const TURNS_SPIN: u64 = CODE + 0x240;
/// The waiter: spin until `FLAG_B`, then set `FLAG` and exit.
const WAITER: u64 = CODE + 0x300;
const WAITER_SET: u64 = CODE + 0x310;
/// The second setter: set `FLAG_B` and exit.
const SETTER_B: u64 = CODE + 0x400;
const DATA: u64 = 0x8000;
/// The flag, in a page of its own (the stack alias makes `DATA`
/// inaccessible at its own address).
const FLAG: u64 = 0x10000;
const FLAG_B: u64 = FLAG + 4;

/// The guest core the threads main creates are pinned to: one worker 0
/// serves besides core 0.
static SETTER_CORE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(2);

fn machine(workers: usize, max_steps: u64, entry: u64) -> Machine {
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
    m.core.cpu.pc = entry;
    m.core.cpu.sp = m.layout.stack_region_base + 0x400;
    m.core.cpu.tpidrro_el0 = m.layout.tls_base;
    m
}

/// CreateThread(entry, arg 0, a stack top, priority 50, `SETTER_CORE`).
fn create_thread(core: &mut Core, entry: u64, stack: u64) -> Result<(), Trap> {
    core.cpu.regs[1] = entry;
    core.cpu.regs[2] = 0;
    core.cpu.regs[3] = 0x7ff0000000 - 0x200000 + stack;
    core.cpu.regs[4] = 50;
    core.cpu.regs[5] = SETTER_CORE.load(std::sync::atomic::Ordering::Relaxed);
    Err(Trap::SupervisorCall { number: 0x08 })
}

/// StartThread on the handle CreateThread returned.
fn start_thread(core: &mut Core) -> Result<(), Trap> {
    assert_eq!(core.cpu.regs[0], 0, "CreateThread failed");
    core.cpu.regs[0] = core.cpu.regs[1];
    Err(Trap::SupervisorCall { number: 0x09 })
}

/// One block of the program at the core's PC.
fn block(core: &mut Core) -> Result<(), Trap> {
    let pc = core.cpu.pc;
    core.tick_block_at(pc, 1)?;
    match pc {
        MAIN_CREATE => return create_thread(core, SETTER, 0x400),
        MAIN_START | TURNS_START_WAITER | TURNS_START_SETTER => return start_thread(core),
        TURNS_CREATE_WAITER => return create_thread(core, WAITER, 0x400),
        TURNS_CREATE_SETTER => return create_thread(core, SETTER_B, 0x800),
        WAITER => {
            // No spin hint: only the turns on its core let SETTER_B run.
            let flag = core.memory.load_u32(FLAG_B);
            core.cpu.pc = if flag == 0 { WAITER } else { WAITER_SET };
        }
        WAITER_SET => {
            core.memory.store_u32(FLAG, 1);
            core.cpu.pc = SETTER_EXIT;
        }
        SETTER_B => {
            core.memory.store_u32(FLAG_B, 1);
            core.cpu.pc = SETTER_EXIT;
        }
        MAIN_SPIN => {
            let flag = core.memory.load_u32(FLAG);
            core.spin_hint();
            core.cpu.pc = if flag == 0 { MAIN_SPIN } else { MAIN_EXIT };
        }
        TURNS_SPIN => {
            let flag = core.memory.load_u32(FLAG);
            core.cpu.pc = if flag == 0 { TURNS_SPIN } else { MAIN_EXIT };
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
        _ if pc == MAIN_START + 4 => MAIN_SPIN,
        // Each other SVC block is followed by the next block, 0x10 on.
        _ if pc % 0x10 == 4 => pc + 0xc,
        _ => pc,
    };
    core.cpu.pc = at;
    block(core)
}

fn run(workers: usize, entry: u64) -> Trap {
    // Worker 0 serves cores 0 and `workers` below six workers; with six,
    // core 1 is worker 1's alone.
    let core = if workers < 6 { workers as u64 } else { 1 };
    SETTER_CORE.store(core, std::sync::atomic::Ordering::Relaxed);
    let mut m = machine(workers, 20_000_000, entry);
    m.run_generated(dispatch).unwrap_err()
}

/// The tests share `SETTER_CORE`: one at a time.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn ends(workers: usize, entry: u64) {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let trap = run(workers, entry);
    assert!(
        matches!(trap, Trap::Exit { .. }),
        "{workers} workers: {trap:?}"
    );
}

#[test]
fn one_worker_lets_the_setter_run() {
    ends(1, MAIN_CREATE);
}

/// Worker 0 serves cores 0 and W; it used to run only the best-priority
/// runnable thread among both, so the spinning main thread (44) kept the
/// processor and the setter (50) never ran (docs/multicore-review.md,
/// finding 1). Each folded core now runs its own best in turn.
#[test]
fn two_to_five_workers_let_a_worse_priority_thread_on_another_folded_core_run() {
    for workers in 2..=5 {
        ends(workers, MAIN_CREATE);
    }
}

/// The waiter and the second setter share a folded core at one priority:
/// the core's turns must alternate them, not rerun the lower id.
#[test]
fn equal_priorities_on_a_folded_core_take_turns() {
    for workers in 2..=5 {
        ends(workers, TURNS_CREATE_WAITER);
    }
}

/// One core per worker: no folding, scheduling as before.
#[test]
fn six_workers_run_both_programs() {
    ends(6, MAIN_CREATE);
    ends(6, TURNS_CREATE_WAITER);
}
