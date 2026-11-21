//! Window calls (`EmitOptions::window_calls`) against the build's own
//! emission: game-shaped callers and functions (`build.rs`'s
//! `window_call_code`: frames, BL and BLR, jumps through BR, early
//! returns elsewhere, SVCs the host completes, recursion, a UDF on some
//! paths) lifted without window calls, with them in one region, with them
//! across 1 KiB regions through the window-call table, and with them in
//! every other 1 KiB region. Every caller runs from random states to its
//! end, in slices (host boundaries every few instructions, resumed through
//! the dispatcher) and against a budget; registers, SP, NZCV, the PC, the
//! step count, the trap, the data memory and the stack must be the same
//! for every lift.
//!
//! The window table `Core::call` uses is one per process (the first lift
//! installed), so each lift runs in a child process of its own and prints a
//! digest per run, which the parent compares.
#![cfg(any(target_os = "linux", windows))]
#![allow(clippy::unwrap_used)]

mod common;

use common::*;
use echora_aot_tests::{Core, Machine, READ, STATE_NORMAL, Trap, WRITE};

#[allow(
    missing_debug_implementations,
    clippy::all,
    dead_code,
    unused_imports,
    unused_parens,
    unreachable_code
)]
mod lifted {
    include!(concat!(env!("OUT_DIR"), "/lifted_window_call.rs"));
}
use lifted::{
    window_call_base, window_call_mixed, window_call_on, window_call_program, window_call_table,
};

const STACK: u64 = 0xc0_0000;
const STACK_SIZE: u64 = 0x2000;
const STATES_PER_CALLER: usize = 12;
/// The child's environment variable: the lift to run.
const CHILD: &str = "ECHORA_WINDOW_CALL_LIFT";

type Dispatch = fn(&mut Core) -> Result<(), Trap>;

/// How a run is driven: to its end, in slices of this many instructions
/// (rounded to blocks), or with this budget.
#[derive(Clone, Copy, Debug)]
enum Mode {
    Whole,
    Sliced(u64),
    Budget(u64),
}

/// Runs `entry` from `state` to its SVC #7, a trap or the budget; the
/// host completes SVC #3 (X0 gets the step count) and resumes after it.
/// The outcome as text: the trap, PC, steps, NZCV, SP and registers, and
/// a hash of the data memory and the stack.
fn run(m: &mut Machine, dispatch: Dispatch, entry: u64, state: &State, mode: Mode) -> String {
    m.core.cpu.regs = state.x;
    m.core.cpu.sp = state.sp;
    m.core.cpu.nzcv = state.nzcv;
    m.core.cpu.pc = entry;
    m.core.steps = 0;
    m.core.spin_polls = 0;
    m.core.max_steps = match mode {
        Mode::Budget(budget) => budget,
        _ => 200_000,
    };
    let next_check = |steps: u64| match mode {
        Mode::Sliced(slice) => steps + slice,
        _ => u64::MAX,
    };
    m.core.host_check_at = next_check(0);
    m.core.rearm();
    m.memory.write(DATA, &state.memory).expect("data is mapped");
    let mut random = Random(state.x[0] | 1);
    let stack: Vec<u8> = (0..STACK_SIZE).map(|_| random.next() as u8).collect();
    m.memory.write(STACK, &stack).expect("stack is mapped");
    let mut boundaries = 0u64;
    let trap = loop {
        match dispatch(&mut m.core) {
            Ok(()) => continue,
            Err(Trap::HostBoundary) => {
                boundaries += 1;
                m.core.spin_polls = 0;
                m.core.host_check_at = next_check(m.core.steps);
                m.core.rearm();
            }
            Err(Trap::SupervisorCall { number: 3 }) => {
                m.core.settle_block_trap();
                m.core.memory.clear_exclusive();
                m.core.cpu.regs[0] = m.core.steps;
                m.core.cpu.pc += 4;
            }
            Err(trap) => break trap,
        }
    };
    m.core.settle_block_trap();
    let mut memory = vec![0; DATA_SIZE as usize];
    m.memory.read(DATA, &mut memory).expect("data is mapped");
    let mut stack = vec![0; STACK_SIZE as usize];
    m.memory.read(STACK, &mut stack).expect("stack is mapped");
    let fnv = |bytes: &[u8]| {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    };
    let cpu = &m.core.cpu;
    format!(
        "{trap:?} pc {:#x} steps {} boundaries {} nzcv {:x} sp {:#x} x {:x?} memory {:016x} stack {:016x}",
        cpu.pc,
        m.core.steps,
        // Slices end at the same block boundaries in every lift.
        boundaries,
        cpu.nzcv,
        cpu.sp,
        cpu.regs,
        fnv(&memory),
        fnv(&stack)
    )
}

/// Every run of every caller with `dispatch`, one line each.
fn runs(dispatch: Dispatch) -> Vec<String> {
    let words = window_call_program::WORDS;
    let code_size = (words.len() as u64 * 4).next_multiple_of(0x1000);
    let image: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let mut m = Machine::with_layout(
        window_call_program::BASE,
        &image,
        code_size,
        DATA,
        DATA_SIZE,
    );
    m.memory
        .map(STACK, STACK_SIZE, READ | WRITE, STATE_NORMAL, "stack")
        .expect("map stack");
    let mut random = Random(0x6a09_e667_f3bc_c909);
    let memory: Vec<u8> = (0..DATA_SIZE).map(|_| random.next() as u8).collect();
    let mut lines = Vec::new();
    for (caller, &entry) in window_call_program::CALLERS.iter().enumerate() {
        for index in 0..STATES_PER_CALLER {
            let mut state = State {
                x: std::array::from_fn(|_| random.value()),
                sp: STACK + STACK_SIZE - 0x100,
                nzcv: (random.next() & 15) as u8,
                memory: memory.clone(),
            };
            state.x[28] = BASE_ADDRESS;
            state.x[27] = random.next() % 0x100;
            // The recursion count.
            state.x[24] = random.next() % 6;
            let slice = 1 + random.next() % 40;
            let budget = 1 + random.next() % 400;
            for mode in [Mode::Whole, Mode::Sliced(slice), Mode::Budget(budget)] {
                let outcome = run(&mut m, dispatch, entry, &state, mode);
                lines.push(format!(
                    "caller {caller} at {entry:#x} state {index} {mode:?}: {outcome}"
                ));
            }
        }
    }
    lines
}

/// The child: runs one lift and prints its runs.
#[test]
fn window_call_lift_child() {
    let Ok(lift) = std::env::var(CHILD) else {
        return;
    };
    let (dispatch, install): (Dispatch, fn()) = match lift.as_str() {
        "base" => (window_call_base::dispatch, window_call_base::install),
        "on" => (window_call_on::dispatch, window_call_on::install),
        "table" => (window_call_table::dispatch, window_call_table::install),
        "mixed" => (window_call_mixed::dispatch, window_call_mixed::install),
        other => panic!("unknown lift {other}"),
    };
    install();
    for line in runs(dispatch) {
        println!("RUN {line}");
    }
}

#[test]
fn window_calls_match_the_default_emission() {
    if window_call_program::ACTIVE {
        // Window calls are emitted, directly and through the table, and
        // the base lift has none.
        const { assert!(window_call_on::DIRECT_CALLS > 0 && window_call_on::ENTERED_WINDOWS > 0) };
        const { assert!(window_call_table::TABLE_CALLS > 0) };
        const { assert!(window_call_mixed::DIRECT_CALLS + window_call_mixed::TABLE_CALLS > 0) };
    }
    const {
        assert!(
            window_call_base::DIRECT_CALLS
                + window_call_base::TABLE_CALLS
                + window_call_base::ENTERED_WINDOWS
                == 0
        )
    };
    let exe = std::env::current_exe().unwrap();
    let lines = |lift: &str| -> Vec<String> {
        let output = std::process::Command::new(&exe)
            .args([
                "--exact",
                "window_call_lift_child",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(CHILD, lift)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "the {lift} child failed: {output:?}"
        );
        String::from_utf8(output.stdout)
            .unwrap()
            .lines()
            // The harness's own `test ... ` can precede the first.
            .filter_map(|line| Some(line[line.find("RUN ")? + 4..].to_owned()))
            .collect()
    };
    let base = lines("base");
    assert_eq!(
        base.len(),
        window_call_program::CALLERS.len() * STATES_PER_CALLER * 3
    );
    let ended = |trap: &str| {
        base.iter()
            .filter(|line| {
                line.split_once(": ")
                    .is_some_and(|(_, outcome)| outcome.starts_with(trap))
            })
            .count()
    };
    println!(
        "{} runs: {} ended at SVC #7, {} by the budget, {} by a UDF",
        base.len(),
        ended("SupervisorCall { number: 7 }"),
        ended("Budget"),
        ended("UndefinedInstruction")
    );
    for lift in ["on", "table", "mixed"] {
        let other = lines(lift);
        assert_eq!(other.len(), base.len(), "{lift}: runs");
        let mismatches: Vec<String> = base
            .iter()
            .zip(&other)
            .filter(|(a, b)| a != b)
            .take(5)
            .map(|(a, b)| format!("expected {a}\n     got {b}"))
            .collect();
        assert!(
            mismatches.is_empty(),
            "{lift}: runs differ from the default emission:\n{}",
            mismatches.join("\n")
        );
    }
}
