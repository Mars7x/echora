//! Window calls (`EmitOptions::window_calls`) against the build's own
//! emission: game-shaped callers and functions (`build.rs`'s
//! `window_call_code`: frames, BL and BLR, B tail calls, a jump through
//! BR, BR X30 and RETAA returns, early returns to another window, SVCs the
//! host completes, recursion, UDF faults, PLT stub calls and tail calls,
//! float and SIMD instructions, a function falling through into the next
//! window, a jump into a block's middle) lifted in pairs without and with
//! window calls: the build's options (in one region, through the
//! window-call table across 1 KiB regions, in every other region, with
//! early exits, audited), and with deferred charges, the step count in a
//! local, dense dispatch, direct calls and stub windows. Every caller runs
//! from random states (registers, vectors, flags, FPCR, memory) to its
//! end, in slices (host boundaries every few instructions, resumed
//! through the dispatcher) and against a budget; general and vector
//! registers, SP, NZCV, FPCR, FPSR, the PC, the step and boundary counts,
//! the trap, the data memory and the stack must be the same with and
//! without window calls. The runs to the end are also checked against the
//! reference interpreter where it covers every instruction run.
//!
//! The window table `Core::call` uses is one per process (the first lift
//! installed), so each lift runs in a child process of its own and prints a
//! digest per run, which the parent compares.
#![cfg(any(target_os = "linux", windows))]
// The lifts' counts are constants that the build's features decide.
#![allow(clippy::unwrap_used, clippy::assertions_on_constants)]

mod common;

use common::*;
use echora_aot_tests::{Core, Machine, READ, STATE_NORMAL, Trap, WRITE};
use echora_lift::aarch64::{Address, Instruction, MemoryOp, Writeback, decode};

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
use lifted::*;

const STACK: u64 = 0xc0_0000;
const STACK_SIZE: u64 = 0x2000;
const STATES_PER_CALLER: usize = 12;
/// The child's environment variable: the lift to run.
const CHILD: &str = "ECHORA_WINDOW_CALL_LIFT";
const SVC_EXIT: u32 = 0xd400_00e1;

type Dispatch = fn(&mut Core) -> Result<(), Trap>;

/// A program's code and its fixed addresses (`window_call_program` of a
/// lift module).
struct Program {
    base: u64,
    words: &'static [u32],
    callers: &'static [u64],
    got_slot: u64,
    got_word: u64,
    missing: u64,
}

macro_rules! program {
    ($module:path) => {{
        use $module as program;
        Program {
            base: program::BASE,
            words: program::WORDS,
            callers: program::CALLERS,
            got_slot: program::GOT_SLOT,
            got_word: program::GOT_WORD,
            missing: program::MISSING,
        }
    }};
}

/// The program with float and SIMD bodies, and its integer twin.
fn programs() -> (Program, Program) {
    (
        program!(window_call_program),
        program!(window_call_int::window_call_program),
    )
}

/// Every lift by name: its dispatcher, `install`, whether every region
/// making window calls has its table, and the census and call counts.
struct Lift {
    name: &'static str,
    dispatch: Dispatch,
    install: fn(),
    installed: fn() -> bool,
    census: (usize, usize, usize, usize),
    summarized: usize,
}

macro_rules! lift {
    ($module:ident) => {
        Lift {
            name: stringify!($module),
            dispatch: $module::dispatch,
            install: $module::install,
            installed: $module::window_calls_installed,
            census: $module::CENSUS,
            summarized: $module::DIRECT_CALLS + $module::TABLE_CALLS + $module::FUNCTION_CALLS,
        }
    };
}

fn lifts() -> Vec<Lift> {
    vec![
        lift!(window_call_base),
        lift!(window_call_on),
        lift!(window_call_table),
        lift!(window_call_mixed),
        lift!(window_call_early),
        lift!(window_call_audit),
        lift!(window_call_deferred_base),
        lift!(window_call_deferred_on),
        lift!(window_call_steps_base),
        lift!(window_call_steps_on),
        lift!(window_call_dense_base),
        lift!(window_call_dense_on),
        lift!(window_call_direct_base),
        lift!(window_call_direct_on),
        lift!(window_call_stub_base),
        lift!(window_call_stub_on),
        lift!(function_on),
        lift!(function_regions),
        lift!(function_audit),
        lift!(function_direct),
        lift!(function_deferred),
        lift!(function_steps),
        lift!(regalloc_on),
        lift!(regalloc_regions),
        lift!(regalloc_direct),
        lift!(regalloc_loops),
        lift!(regalloc_vectors),
        lift!(regalloc_crates),
        lift!(regalloc_capped),
        lift!(regalloc_split),
        lift!(regalloc_returns),
        lift!(regalloc_summarized),
        lift!(regalloc_shared),
        lift!(regalloc_coarse),
        lift!(regalloc_returns_resume),
        lift!(regalloc_span),
        lift!(regalloc_resume),
        lift!(regalloc_resume_crates),
        lift!(regalloc_indirect_base),
        lift!(regalloc_indirect),
    ]
}

/// Each lift with window calls and the lift without them it must match.
const PAIRS: &[(&str, &str)] = &[
    ("window_call_base", "window_call_on"),
    ("window_call_base", "window_call_table"),
    ("window_call_base", "window_call_mixed"),
    ("window_call_base", "window_call_early"),
    ("window_call_base", "window_call_audit"),
    ("window_call_deferred_base", "window_call_deferred_on"),
    ("window_call_steps_base", "window_call_steps_on"),
    ("window_call_dense_base", "window_call_dense_on"),
    ("window_call_direct_base", "window_call_direct_on"),
    ("window_call_stub_base", "window_call_stub_on"),
    ("window_call_base", "function_on"),
    ("window_call_base", "function_regions"),
    ("window_call_base", "function_audit"),
    ("window_call_base", "function_direct"),
    ("window_call_deferred_base", "function_deferred"),
    ("window_call_steps_base", "function_steps"),
    ("window_call_base", "regalloc_on"),
    ("window_call_base", "regalloc_regions"),
    ("window_call_base", "regalloc_direct"),
    ("window_call_base", "regalloc_loops"),
    ("window_call_base", "regalloc_vectors"),
    ("window_call_base", "regalloc_crates"),
    ("window_call_base", "regalloc_capped"),
    ("window_call_base", "regalloc_split"),
    ("window_call_base", "regalloc_returns"),
    ("window_call_base", "regalloc_summarized"),
    ("window_call_base", "regalloc_shared"),
    ("window_call_base", "regalloc_returns_resume"),
    ("window_call_base", "regalloc_span"),
    ("window_call_base", "regalloc_resume"),
    ("window_call_base", "regalloc_resume_crates"),
    ("regalloc_indirect_base", "regalloc_indirect"),
];

/// How a run is driven: to its end, in slices of this many instructions
/// (rounded to blocks), or with this budget.
#[derive(Clone, Copy, Debug)]
enum Mode {
    Whole,
    Sliced(u64),
    Budget(u64),
}

/// A run's starting state: registers, SP, NZCV and data memory, vectors,
/// FPCR, the stack.
#[derive(Clone)]
struct Start {
    state: State,
    vectors: [u128; 32],
    fpcr: u32,
    stack: Vec<u8>,
}

/// What a run ends with (for the reference comparison).
#[derive(Debug, PartialEq)]
struct End {
    trap: String,
    x: [u64; 31],
    sp: u64,
    nzcv: u8,
    memory: Vec<u8>,
    stack: Vec<u8>,
}

fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// Runs `entry` from `start` to its SVC #7, a trap or the budget; the
/// host completes SVC #3 (X0 ^= 0x5555) and resumes after it. The outcome
/// as text (the trap, PC, steps, boundaries, NZCV, SP, FPCR, FPSR,
/// registers, and hashes of the vectors, the data memory and the stack),
/// and the end state.
fn run(
    m: &mut Machine,
    dispatch: Dispatch,
    entry: u64,
    start: &Start,
    mode: Mode,
) -> (String, End) {
    let state = &start.state;
    m.core.cpu.regs = state.x;
    m.core.cpu.sp = state.sp;
    m.core.cpu.nzcv = state.nzcv;
    m.core.cpu.vectors = start.vectors;
    m.core.cpu.fpcr = start.fpcr;
    m.core.cpu.fpsr = 0;
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
    m.memory
        .write(STACK, &start.stack)
        .expect("stack is mapped");
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
                m.core.cpu.regs[0] ^= 0x5555;
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
    let cpu = &m.core.cpu;
    let vectors: Vec<u8> = cpu.vectors.iter().flat_map(|v| v.to_le_bytes()).collect();
    let text = format!(
        "{trap:?} pc {:#x} steps {} boundaries {} nzcv {:x} sp {:#x} fpcr {:#x} fpsr {:#x} x {:x?} vectors {:016x} memory {:016x} stack {:016x}",
        cpu.pc,
        m.core.steps,
        // Slices end at the same block boundaries in every lift.
        boundaries,
        cpu.nzcv,
        cpu.sp,
        cpu.fpcr,
        cpu.fpsr,
        cpu.regs,
        fnv(&vectors),
        fnv(&memory),
        fnv(&stack)
    );
    let end = End {
        trap: format!("{trap:?}"),
        x: cpu.regs,
        sp: cpu.sp,
        nzcv: cpu.nzcv,
        memory,
        stack,
    };
    (text, end)
}

fn machine(program: &Program) -> Machine {
    let words = program.words;
    let code_size = (words.len() as u64 * 4).next_multiple_of(0x1000);
    let image: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let mut m = Machine::with_layout(program.base, &image, code_size, DATA, DATA_SIZE);
    m.memory
        .map(STACK, STACK_SIZE, READ | WRITE, STATE_NORMAL, "stack")
        .expect("map stack");
    m
}

/// Every caller's starting states, the same for every lift.
fn starts(program: &Program) -> Vec<(usize, u64, Start, u64, u64)> {
    let mut random = Random(0x6a09_e667_f3bc_c909);
    let mut memory: Vec<u8> = (0..DATA_SIZE).map(|_| random.next() as u8).collect();
    let slot = (program.got_slot - DATA) as usize;
    memory[slot..slot + 8].copy_from_slice(&program.got_word.to_le_bytes());
    let mut starts = Vec::new();
    for (caller, &entry) in program.callers.iter().enumerate() {
        for _ in 0..STATES_PER_CALLER {
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
            let vectors = std::array::from_fn(|_| {
                u128::from(random.value()) | u128::from(random.value()) << 64
            });
            // Round to nearest, plain, flush-to-zero (the game's), or with
            // default NaNs too (other rounding modes trap).
            let fpcr = [0, 0x0100_0000, 0x0300_0000][random.next() as usize % 3];
            let stack: Vec<u8> = (0..STACK_SIZE).map(|_| random.next() as u8).collect();
            let slice = 1 + random.next() % 40;
            let budget = 1 + random.next() % 400;
            starts.push((
                caller,
                entry,
                Start {
                    state,
                    vectors,
                    fpcr,
                    stack,
                },
                slice,
                budget,
            ));
        }
    }
    starts
}

/// Every run of every caller with `dispatch`, one line each; and the hash
/// of the recent-PC ring after each run (direct window calls skip the
/// entry's record).
fn runs(dispatch: Dispatch) -> (Vec<String>, u64) {
    let program = programs().0;
    let mut m = machine(&program);
    let mut lines = Vec::new();
    let mut ring = 0u64;
    for (caller, entry, start, slice, budget) in starts(&program) {
        for mode in [Mode::Whole, Mode::Sliced(slice), Mode::Budget(budget)] {
            let (outcome, _) = run(&mut m, dispatch, entry, &start, mode);
            lines.push(format!("caller {caller} at {entry:#x} {mode:?}: {outcome}"));
            let pcs: Vec<u8> = m
                .core
                .recent_thread_pcs()
                .iter()
                .flat_map(|record| record.pc.to_le_bytes())
                .collect();
            ring = ring.rotate_left(5) ^ fnv(&pcs);
        }
    }
    (lines, ring)
}

fn word_at(program: &Program, pc: u64) -> Option<u32> {
    let index = usize::try_from(pc.checked_sub(program.base)? / 4).ok()?;
    program.words.get(index).copied()
}

/// SP-relative LDP/STP of X registers on the stack.
fn stack_pair(start: &mut Start, instruction: &Instruction) -> Option<()> {
    let Instruction::LoadStorePair {
        rt,
        rt2,
        rn: 31,
        offset,
        writeback,
        op,
        ..
    } = *instruction
    else {
        return None;
    };
    let sp = start.state.sp;
    let moved = sp.wrapping_add(offset as u64);
    let at = if writeback == Writeback::Post {
        sp
    } else {
        moved
    };
    let offset = usize::try_from(at.checked_sub(STACK)?).ok()?;
    if offset + 16 > STACK_SIZE as usize {
        return None;
    }
    for (k, register) in [rt, rt2].into_iter().enumerate() {
        let bytes = &mut start.stack[offset + 8 * k..offset + 8 * k + 8];
        match op {
            MemoryOp::Store => {
                let value = if register == 31 {
                    0
                } else {
                    start.state.x[register as usize]
                };
                bytes.copy_from_slice(&value.to_le_bytes());
            }
            MemoryOp::Load { .. } => {
                if register != 31 {
                    start.state.x[register as usize] =
                        u64::from_le_bytes((&*bytes).try_into().unwrap());
                }
            }
        }
    }
    if writeback != Writeback::None {
        start.state.sp = moved;
    }
    Some(())
}

/// Runs a caller with the reference interpreter: the end it reaches, or
/// None where it runs an instruction it has no semantics for (float and
/// SIMD) or goes astray.
fn reference(program: &Program, entry: u64, start: &Start) -> Option<End> {
    let mut start = start.clone();
    let mut pc = entry;
    let trap = 'run: {
        for _ in 0..200_000 {
            if pc == program.missing {
                break 'run format!("{:?}", Trap::MissingBlock { pc });
            }
            let word = word_at(program, pc)?;
            if word == SVC_EXIT {
                break 'run format!("{:?}", Trap::SupervisorCall { number: 7 });
            }
            let instruction = decode(word, pc);
            let effects = echora_lift::defuse::effects(&instruction);
            let undefined = matches!(instruction, Instruction::Undefined);
            if !undefined && (effects.reads.vector != 0 || effects.writes.vector != 0) {
                return None;
            }
            let x = &mut start.state.x;
            pc = match instruction {
                Instruction::Undefined => {
                    break 'run format!("{:?}", Trap::UndefinedInstruction { pc, word });
                }
                Instruction::Svc { number: 3 } => {
                    x[0] ^= 0x5555;
                    pc + 4
                }
                Instruction::Branch { target, link } => {
                    if link {
                        x[30] = pc + 4;
                    }
                    target
                }
                Instruction::BranchReg { rn, link } => {
                    let target = get(&start.state, rn);
                    if link {
                        start.state.x[30] = pc + 4;
                    }
                    target
                }
                Instruction::PAuthDisabledReturn { .. } => x[30],
                Instruction::PAuthDisabledHint { .. } | Instruction::Nop => pc + 4,
                Instruction::Adr { rd, address } => {
                    set(&mut start.state, rd, address);
                    pc + 4
                }
                Instruction::LoadStorePair { rn: 31, .. } => {
                    stack_pair(&mut start, &instruction)?;
                    pc + 4
                }
                // A load from the code (the jump table's halfwords).
                Instruction::LoadStore {
                    rt,
                    rn,
                    bytes,
                    op: op @ MemoryOp::Load { .. },
                    address: Address::Register { rm, option, shift },
                } if {
                    let at = get_sp(&start.state, rn).wrapping_add(extend(
                        get(&start.state, rm),
                        option,
                        shift,
                        64,
                    ));
                    at.checked_sub(program.base)
                        .is_some_and(|offset| offset < 4 * program.words.len() as u64)
                } =>
                {
                    let at = get_sp(&start.state, rn).wrapping_add(extend(
                        get(&start.state, rm),
                        option,
                        shift,
                        64,
                    ));
                    let mut value = 0u64;
                    for byte in 0..u64::from(bytes) {
                        let word = word_at(program, (at + byte) & !3)?;
                        value |= u64::from((word >> (8 * ((at + byte) & 3))) as u8) << (8 * byte);
                    }
                    set(&mut start.state, rt, loaded(value, u64::from(bytes), op));
                    pc + 4
                }
                _ => {
                    let taken = interpret(&instruction, &mut start.state)?;
                    match instruction {
                        Instruction::BranchCond { target, .. }
                        | Instruction::CompareBranch { target, .. }
                        | Instruction::TestBranch { target, .. }
                            if taken =>
                        {
                            target
                        }
                        _ => pc + 4,
                    }
                }
            };
        }
        return None;
    };
    Some(End {
        trap,
        x: start.state.x,
        sp: start.state.sp,
        nzcv: start.state.nzcv,
        memory: start.state.memory,
        stack: start.stack,
    })
}

/// The child: runs one lift and prints its runs, or (`reference`) checks
/// the window-call lift's runs to the end against the reference.
#[test]
fn window_call_lift_child() {
    let Ok(name) = std::env::var(CHILD) else {
        return;
    };
    if let Some(lift) = name.strip_prefix("reference:") {
        let (dispatch, install): (Dispatch, fn()) = match lift {
            "base" => (
                window_call_int::window_call_base::dispatch,
                window_call_int::window_call_base::install,
            ),
            "on" => (
                window_call_int::window_call_on::dispatch,
                window_call_int::window_call_on::install,
            ),
            "table" => (
                window_call_int::window_call_table::dispatch,
                window_call_int::window_call_table::install,
            ),
            "function_on" => (
                window_call_int::function_on::dispatch,
                window_call_int::function_on::install,
            ),
            "function_regions" => (
                window_call_int::function_regions::dispatch,
                window_call_int::function_regions::install,
            ),
            "regalloc_on" => (
                window_call_int::regalloc_on::dispatch,
                window_call_int::regalloc_on::install,
            ),
            "regalloc_regions" => (
                window_call_int::regalloc_regions::dispatch,
                window_call_int::regalloc_regions::install,
            ),
            "regalloc_loops" => (
                window_call_int::regalloc_loops::dispatch,
                window_call_int::regalloc_loops::install,
            ),
            "regalloc_crates" => (
                window_call_int::regalloc_crates::dispatch,
                window_call_int::regalloc_crates::install,
            ),
            "regalloc_span" => (
                window_call_int::regalloc_span::dispatch,
                window_call_int::regalloc_span::install,
            ),
            "regalloc_coarse" => (
                window_call_int::regalloc_coarse::dispatch,
                window_call_int::regalloc_coarse::install,
            ),
            "regalloc_indirect_base" => (
                window_call_int::regalloc_indirect_base::dispatch,
                window_call_int::regalloc_indirect_base::install,
            ),
            "regalloc_indirect" => (
                window_call_int::regalloc_indirect::dispatch,
                window_call_int::regalloc_indirect::install,
            ),
            other => panic!("unknown lift {other}"),
        };
        install();
        let program = programs().1;
        let mut m = machine(&program);
        let (mut compared, mut skipped) = (0, 0);
        for (caller, entry, start, _, _) in starts(&program) {
            let Some(expected) = reference(&program, entry, &start) else {
                skipped += 1;
                continue;
            };
            let (_, actual) = run(&mut m, dispatch, entry, &start, Mode::Whole);
            compared += 1;
            if actual != expected {
                let state = |end: &End| State {
                    x: end.x,
                    sp: end.sp,
                    nzcv: end.nzcv,
                    memory: end.memory.clone(),
                };
                panic!(
                    "caller {caller} at {entry:#x}: {} -> {} ({}); stack {}",
                    expected.trap,
                    actual.trap,
                    difference(&state(&expected), &state(&actual)),
                    if expected.stack == actual.stack {
                        "same"
                    } else {
                        "differs"
                    }
                );
            }
        }
        println!("REFERENCE compared {compared} skipped {skipped}");
        return;
    }
    let (name, install) = match name.strip_suffix(",uninstalled") {
        Some(name) => (name.to_owned(), false),
        None => (name, true),
    };
    let lifts = lifts();
    let lift = lifts
        .iter()
        .find(|lift| lift.name == name)
        .unwrap_or_else(|| panic!("unknown lift {name}"));
    if install {
        (lift.install)();
    }
    println!("INSTALLED {}", (lift.installed)());
    let (lines, ring) = runs(lift.dispatch);
    for line in lines {
        println!("RUN {line}");
    }
    println!("RING {ring:016x}");
}

/// A child's output: its run lines, whether it had the window-call table,
/// and the recent-PC ring hash.
struct Output {
    runs: Vec<String>,
    installed: bool,
    ring: String,
}

fn child(lift: &str) -> Output {
    let exe = std::env::current_exe().unwrap();
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
        "the {lift} child failed: {}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    // The harness's own `test ... ` can precede the first line.
    let field = |key: &str| -> Vec<String> {
        text.lines()
            .filter_map(|line| Some(line[line.find(key)? + key.len()..].to_owned()))
            .collect()
    };
    Output {
        runs: field("RUN "),
        installed: field("INSTALLED ")
            .first()
            .is_some_and(|value| value == "true"),
        ring: field("RING ").first().cloned().unwrap_or_default(),
    }
}

/// `coarse-steps` (`regalloc_coarse`): every run to its end, whole or in
/// slices, ends as the default emission's does (trap, PC, registers,
/// vectors, memory, stack, flags) but for the step and boundary counts; the
/// steps never fall below the instructions the default emission counts
/// (each test charges the longest path to the next), and the charges are
/// coarse (some runs count more). Budgeted runs stop where the coarse count
/// reaches the budget, sooner: not compared.
#[test]
fn coarse_steps_end_as_the_default_emission() {
    let base = child("window_call_base");
    let coarse = child("regalloc_coarse");
    assert_eq!(base.runs.len(), coarse.runs.len());
    // The line without its step and boundary counts, and the steps.
    let split = |line: &str| -> (String, u64) {
        let at = line.find(" steps ").expect("a run line has its steps");
        let (steps, rest) = line[at + 7..].split_once(' ').expect("steps, then more");
        let rest = rest
            .strip_prefix("boundaries ")
            .expect("then the boundaries");
        let (_, rest) = rest.split_once(' ').expect("then more");
        (
            format!("{} {rest}", &line[..at]),
            steps.parse().expect("a count"),
        )
    };
    let (mut compared, mut above) = (0, 0);
    for (expected, actual) in base.runs.iter().zip(&coarse.runs) {
        if expected.contains(" Budget(") {
            continue;
        }
        let (expected_text, expected_steps) = split(expected);
        let (actual_text, actual_steps) = split(actual);
        assert_eq!(actual_text, expected_text);
        assert!(
            actual_steps >= expected_steps,
            "{actual_steps} steps counted, {expected_steps} run: {actual}"
        );
        compared += 1;
        above += usize::from(actual_steps > expected_steps);
    }
    println!("coarse steps: {compared} runs compared, {above} counted more");
    assert!(compared > 0 && above > 0, "{compared} {above}");
}

#[test]
fn window_calls_match_the_default_emission() {
    let lifts = lifts();
    for lift in &lifts {
        let (windows, spills, reloads, calls) = lift.census;
        println!(
            "{}: {windows} windows, {calls} calls, {} summarized; per window {:.1} spills, {:.1} reloads",
            lift.name,
            lift.summarized,
            spills as f64 / windows as f64,
            reloads as f64 / windows as f64
        );
        if lift.name.ends_with("base") {
            assert_eq!(lift.summarized, 0, "{}", lift.name);
        } else if window_call_program::ACTIVE {
            assert!(lift.summarized > 0, "{} has no window calls", lift.name);
        }
    }
    if window_call_program::ACTIVE {
        assert!(window_call_table::TABLE_CALLS > 0);
        // Early exits allow more targets (with long windows, the windows
        // of the program's targets already hold every block they reach).
        if cfg!(feature = "long-windows") {
            assert!(window_call_early::DIRECT_CALLS >= window_call_on::DIRECT_CALLS);
        } else {
            assert!(window_call_early::DIRECT_CALLS > window_call_on::DIRECT_CALLS);
        }
    }
    let mut outputs = std::collections::BTreeMap::new();
    for lift in &lifts {
        let output = child(lift.name);
        assert_eq!(
            output.runs.len(),
            programs().0.callers.len() * STATES_PER_CALLER * 3,
            "{}: runs",
            lift.name
        );
        if window_call_program::ACTIVE
            && !lift.name.ends_with("base")
            && !lift.name.starts_with("function_")
            && !lift.name.starts_with("regalloc_")
        {
            assert!(
                output.installed,
                "{}: the window-call table is not installed",
                lift.name
            );
        }
        outputs.insert(lift.name, output);
    }
    let base = &outputs["window_call_base"].runs;
    let ended = |trap: &str| {
        base.iter()
            .filter(|line| {
                line.split_once(": ")
                    .is_some_and(|(_, outcome)| outcome.starts_with(trap))
            })
            .count()
    };
    println!(
        "{} runs: {} ended at SVC #7, {} by the budget, {} by a UDF, {} by a missing block",
        base.len(),
        ended("SupervisorCall { number: 7 }"),
        ended("Budget"),
        ended("UndefinedInstruction"),
        ended("MissingBlock")
    );
    for &(plain, calling) in PAIRS {
        let (expected, actual) = (&outputs[plain].runs, &outputs[calling].runs);
        let mismatches: Vec<String> = expected
            .iter()
            .zip(actual)
            .filter(|(a, b)| a != b)
            .take(5)
            .map(|(a, b)| format!("expected {a}\n     got {b}"))
            .collect();
        assert!(
            mismatches.is_empty(),
            "{calling}: runs differ from {plain}:\n{}",
            mismatches.join("\n")
        );
    }
    if window_call_program::ACTIVE {
        // Calls through the table enter windows as direct calls do (no
        // entry record); without the table they would return early and
        // the dispatcher would enter the callee, recording it.
        // (Long windows enter fewer windows in `window_call_on`'s 64 KiB
        // region than in the table lift's 1 KiB regions.)
        if !cfg!(feature = "long-windows") {
            assert_eq!(
                outputs["window_call_table"].ring,
                outputs["window_call_on"].ring
            );
        }
        let uninstalled = child("window_call_table,uninstalled");
        assert!(!uninstalled.installed);
        assert_eq!(&uninstalled.runs, base, "uninstalled: still the same runs");
        assert_ne!(uninstalled.ring, outputs["window_call_table"].ring);
    }
}

/// The cases lifted functions meet in the game: a switch through a jump
/// table (its cases in the function, so it and the frame calling it get
/// summaries), a function over `FUNCTION_BLOCK_LIMIT` blocks (left to
/// the windows), a game-shaped function calling both in a loop, and
/// functions cut at region edges (1 KiB regions).
#[test]
fn lifted_functions_cover_the_game_s_cases() {
    use window_call_program::*;
    assert_eq!(JUMP_TABLES, 1);
    for function in [SWITCH, SWITCH_FRAME, GAME] {
        assert!(FUNCTIONS_16.contains(&function), "{function:#x} is lifted");
        assert!(
            SUMMARIES_16.contains(&function),
            "{function:#x} has a summary"
        );
    }
    assert!(!FUNCTIONS_16.contains(&LONG), "over the block limit");
    // In 1 KiB regions bodies end at region edges (the long function's
    // part in its first region may fit the limit).
    assert!(!FUNCTIONS_10.is_empty());
    assert!(function_on::FUNCTION_CALLS > 0);
    assert!(function_regions::FUNCTION_CALLS > 0);
    assert!(function_on::FUNCTIONS > 0);
    // Structured functions run, entered by calls and by window arms.
    assert!(regalloc_on::STRUCTURED_FUNCTIONS > 0);
    assert!(regalloc_regions::STRUCTURED_FUNCTIONS > 0);
    assert!(regalloc_on::STRUCTURED_ENTRIES > 0);
    // In crates of their own, the windows reach them through the table.
    assert!(regalloc_crates::STRUCTURED_FUNCTIONS > 0);
    assert!(regalloc_crates::FUNCTION_CRATES > 0);
    // Deferred roots as entries: more functions.
    assert!(regalloc_indirect::FUNCTIONS > regalloc_on::FUNCTIONS);
    // Resume functions: more window arms run structured code.
    // (With the resume features on, `regalloc_on` has them too.)
    if cfg!(feature = "regalloc-resume-loops") || cfg!(feature = "regalloc-resume-returns") {
        assert!(regalloc_resume::STRUCTURED_ENTRIES >= regalloc_on::STRUCTURED_ENTRIES);
    } else {
        assert!(regalloc_resume::STRUCTURED_ENTRIES > regalloc_on::STRUCTURED_ENTRIES);
    }
    assert!(regalloc_resume_crates::STRUCTURED_ENTRIES > 0);
    // Bodies across region edges exist (and run in `regalloc_span`).
    assert!(window_call_program::SPANNING > 0);
    // Over the size cap: stubs, the windows run them.
    assert!(regalloc_capped::CAPPED_FUNCTIONS > 0);
    assert!(regalloc_capped::STRUCTURED_FUNCTIONS > 0);
    assert!(regalloc_capped::CAPPED_FUNCTIONS > regalloc_on::CAPPED_FUNCTIONS);
    // In pieces instead: fewer left to the windows, several pieces each.
    assert!(regalloc_split::SPLIT_FUNCTIONS > 0);
    assert!(regalloc_split::PIECES > 2 * regalloc_split::SPLIT_FUNCTIONS);
    assert!(regalloc_split::CAPPED_FUNCTIONS < regalloc_capped::CAPPED_FUNCTIONS);
    // Return points as entries: more window arms run structured code.
    if cfg!(feature = "regalloc-return-entries") || cfg!(feature = "regalloc-resume-returns") {
        assert!(regalloc_returns::STRUCTURED_ENTRIES >= regalloc_on::STRUCTURED_ENTRIES);
    } else {
        assert!(regalloc_returns::STRUCTURED_ENTRIES > regalloc_on::STRUCTURED_ENTRIES);
    }
}

/// The integer twin, without and with window calls (in one region and
/// through the table), against the reference interpreter.
#[test]
fn window_calls_match_the_reference() {
    if window_call_program::ACTIVE {
        assert!(window_call_int::window_call_on::DIRECT_CALLS > 0);
        assert!(window_call_int::window_call_table::TABLE_CALLS > 0);
    }
    for lift in [
        "base",
        "on",
        "table",
        "function_on",
        "function_regions",
        "regalloc_on",
        "regalloc_regions",
        "regalloc_loops",
        "regalloc_crates",
        "regalloc_span",
        "regalloc_coarse",
        "regalloc_indirect_base",
        "regalloc_indirect",
    ] {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "window_call_lift_child",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(CHILD, format!("reference:{lift}"))
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success(),
            "{lift}: {text}{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let line = text
            .lines()
            .find_map(|line| Some(&line[line.find("REFERENCE ")?..]))
            .unwrap();
        println!("{lift}: {line}");
        let numbers: Vec<usize> = line
            .split_whitespace()
            .filter_map(|word| word.parse().ok())
            .collect();
        // Every run is compared.
        assert_eq!(numbers[1], 0, "{lift}: runs left out: {line}");
    }
}
