//! Call differential test: random callers with BL and BLR to leaf
//! functions (several in other dispatch windows, some calling a later leaf
//! behind an STP/LDP of X29/X30), lifted through split emission and run
//! natively, against the shared reference interpreter extended with BL,
//! BLR and RET.
//!
//! Calls go through `Core::call` and the register-local stores and reloads
//! around it, a path single instructions never reach. Registers, SP, NZCV,
//! the data memory and the stack must match at each caller's final SVC.
#![cfg(any(target_os = "linux", windows))]
#![allow(clippy::unwrap_used)]

mod common;

use common::*;
use echora_aot_tests::{Machine, READ, STATE_NORMAL, Trap, WRITE, call_code};
use echora_lift::aarch64::{Instruction, decode};

const STATES_PER_CALLER: usize = 16;
const STACK: u64 = 0xc0_0000;
const STACK_SIZE: u64 = 0x1000;
const SVC_EXIT: u32 = 0xd400_00e1;

#[derive(Clone, PartialEq, Eq, Debug)]
struct Run {
    state: State,
    stack: Vec<u8>,
}

fn word_at(pc: u64) -> Option<u32> {
    let index = usize::try_from(pc.checked_sub(call_code::BASE)? / 4).ok()?;
    call_code::WORDS.get(index).copied()
}

/// Reads and writes SP-relative pairs on the separate stack.
fn stack_access(run: &mut Run, instruction: &Instruction) -> Option<bool> {
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
    let sp = run.state.sp;
    let moved = sp.wrapping_add(offset as u64);
    let at = if writeback == echora_lift::aarch64::Writeback::Post {
        sp
    } else {
        moved
    };
    let offset = usize::try_from(at.checked_sub(STACK)?).ok()?;
    if offset + 16 > STACK_SIZE as usize {
        return None;
    }
    match op {
        echora_lift::aarch64::MemoryOp::Store => {
            for (k, register) in [rt, rt2].into_iter().enumerate() {
                let value = if register == 31 {
                    0
                } else {
                    run.state.x[register as usize]
                };
                run.stack[offset + 8 * k..offset + 8 * k + 8].copy_from_slice(&value.to_le_bytes());
            }
        }
        echora_lift::aarch64::MemoryOp::Load { .. } => {
            for (k, register) in [rt, rt2].into_iter().enumerate() {
                let bytes = run.stack[offset + 8 * k..offset + 8 * k + 8]
                    .try_into()
                    .unwrap();
                if register != 31 {
                    run.state.x[register as usize] = u64::from_le_bytes(bytes);
                }
            }
        }
    }
    if writeback != echora_lift::aarch64::Writeback::None {
        run.state.sp = moved;
    }
    Some(false)
}

/// Runs a caller with the reference; `None` if it went astray.
fn reference(entry: u64, run: &mut Run) -> Option<()> {
    let mut pc = entry;
    for _ in 0..10_000 {
        let word = word_at(pc)?;
        if word == SVC_EXIT {
            return Some(());
        }
        let instruction = decode(word, pc);
        pc = match instruction {
            Instruction::Branch { target, link: true } => {
                run.state.x[30] = pc + 4;
                target
            }
            Instruction::BranchReg { rn, link } => {
                let target = get(&run.state, rn);
                if link {
                    run.state.x[30] = pc + 4;
                }
                target
            }
            Instruction::LoadStorePair { rn: 31, .. } => {
                stack_access(run, &instruction)?;
                pc + 4
            }
            _ => {
                let taken = interpret(&instruction, &mut run.state)?;
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
    None
}

fn run_lifted(m: &mut Machine, entry: u64, run: &Run) -> Result<Run, Trap> {
    let state = &run.state;
    m.core.cpu.regs = state.x;
    m.core.cpu.sp = state.sp;
    m.core.cpu.nzcv = state.nzcv;
    m.core.cpu.pc = entry;
    m.memory.write(DATA, &state.memory).expect("data is mapped");
    m.memory.write(STACK, &run.stack).expect("stack is mapped");
    m.core.spin_polls = 0;
    let trap = loop {
        match call_code::dispatch(&mut m.core) {
            Ok(()) => continue,
            Err(Trap::HostBoundary) => {
                m.core.spin_polls = 0;
                m.core.host_check_at = u64::MAX;
                m.core.rearm();
            }
            Err(trap) => break trap,
        }
    };
    m.core.settle_block_trap();
    if !matches!(trap, Trap::SupervisorCall { number: 7 }) {
        return Err(trap);
    }
    let mut memory = vec![0; DATA_SIZE as usize];
    m.memory.read(DATA, &mut memory).expect("data is mapped");
    let mut stack = vec![0; STACK_SIZE as usize];
    m.memory.read(STACK, &mut stack).expect("stack is mapped");
    Ok(Run {
        state: State {
            x: m.core.cpu.regs,
            sp: m.core.cpu.sp,
            nzcv: m.core.cpu.nzcv,
            memory,
        },
        stack,
    })
}

#[test]
fn lifted_calls_match_the_reference() {
    let words = call_code::WORDS;
    let code_size = (words.len() as u64 * 4).next_multiple_of(0x1000);
    let image: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let mut m = Machine::with_layout(call_code::BASE, &image, code_size, DATA, DATA_SIZE);
    m.memory
        .map(STACK, STACK_SIZE, READ | WRITE, STATE_NORMAL, "stack")
        .expect("map stack");
    call_code::install();
    let mut random = Random(0x1f83_d9ab_fb41_bd6b);
    let memory: Vec<u8> = (0..DATA_SIZE).map(|_| random.next() as u8).collect();
    let stack: Vec<u8> = (0..STACK_SIZE).map(|_| random.next() as u8).collect();
    let mut failures = Vec::new();
    let mut runs = 0;
    for &entry in call_code::CALLERS {
        for _ in 0..STATES_PER_CALLER {
            let mut state = State {
                x: std::array::from_fn(|_| random.value()),
                sp: STACK + STACK_SIZE,
                nzcv: (random.next() & 15) as u8,
                memory: memory.clone(),
            };
            state.x[28] = BASE_ADDRESS;
            state.x[27] = random.next() % 0x100;
            let before = Run {
                state,
                stack: stack.clone(),
            };
            let mut expected = before.clone();
            let reference = reference(entry, &mut expected);
            runs += 1;
            let problem = match (reference, run_lifted(&mut m, entry, &before)) {
                (Some(()), Ok(actual)) if actual == expected => continue,
                (Some(()), Ok(actual)) => {
                    let mut text = difference(&expected.state, &actual.state);
                    if expected.stack != actual.stack {
                        text.push_str("; stack differs");
                    }
                    text
                }
                (Some(()), Err(trap)) => format!("lifted trapped: {trap:?}"),
                (None, _) => "the reference went astray".to_owned(),
            };
            failures.push(format!("caller at {entry:#x}: {problem}"));
            break;
        }
    }
    let calls = words
        .iter()
        .filter(|&&word| {
            matches!(
                decode(word, 0),
                Instruction::Branch { link: true, .. } | Instruction::BranchReg { link: true, .. }
            )
        })
        .count();
    println!(
        "{} callers ({calls} calls), {runs} runs, {} mismatching callers",
        call_code::CALLERS.len(),
        failures.len()
    );
    assert!(calls > 0);
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures
            .iter()
            .take(20)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}
