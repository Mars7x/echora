//! Loop and indirect-entry differential test, with and without scheduler
//! slices.
//!
//! Each program runs a counted loop whose middle is also an entry, jumps
//! with ADR + BR to a block reached only through the register, and calls
//! with ADRP + ADD + BLR into a leaf function at its start or partway
//! through its first block (leaves sit mid-window). With one- to
//! three-instruction slices, the host boundary comes at nearly every block
//! boundary and execution resumes through `dispatch`: blocks that codegen
//! merges or inlines must still be enterable from outside, at any
//! boundary and through any address the program builds. Registers, SP,
//! NZCV and memory must match the shared reference.
#![cfg(any(target_os = "linux", windows))]
#![allow(clippy::unwrap_used)]

mod common;

use common::*;
use echora_aot_tests::{Machine, Trap, loop_code};
use echora_lift::aarch64::{Instruction, decode};

const STATES_PER_PROGRAM: usize = 12;
const SVC_EXIT: u32 = 0xd400_00e1;

fn word_at(pc: u64) -> Option<u32> {
    let index = usize::try_from(pc.checked_sub(loop_code::BASE)? / 4).ok()?;
    loop_code::WORDS.get(index).copied()
}

/// Runs a program with the reference; `None` if it went astray.
fn reference(entry: u64, state: &mut State) -> Option<()> {
    let mut pc = entry;
    for _ in 0..10_000 {
        let word = word_at(pc)?;
        if word == SVC_EXIT {
            return Some(());
        }
        let instruction = decode(word, pc);
        pc = match instruction {
            Instruction::Adr { rd, address } => {
                state.x[rd as usize] = address;
                pc + 4
            }
            Instruction::BranchReg { rn, link } => {
                let target = get(state, rn);
                if link {
                    state.x[30] = pc + 4;
                }
                target
            }
            _ => {
                let taken = interpret(&instruction, state)?;
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

fn run_lifted(
    m: &mut Machine,
    entry: u64,
    state: &State,
    slice: Option<u64>,
    boundaries: &mut u64,
) -> Result<State, Trap> {
    m.core.cpu.regs = state.x;
    m.core.cpu.sp = state.sp;
    m.core.cpu.nzcv = state.nzcv;
    m.core.cpu.pc = entry;
    m.memory.write(DATA, &state.memory).expect("data is mapped");
    m.core.spin_polls = 0;
    let next_check = |steps: u64| slice.map_or(u64::MAX, |slice| steps + slice);
    m.core.host_check_at = next_check(m.core.steps);
    m.core.rearm();
    let trap = loop {
        match loop_code::dispatch(&mut m.core) {
            Ok(()) => continue,
            Err(Trap::HostBoundary) => {
                *boundaries += 1;
                m.core.spin_polls = 0;
                m.core.host_check_at = next_check(m.core.steps);
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
    Ok(State {
        x: m.core.cpu.regs,
        sp: m.core.cpu.sp,
        nzcv: m.core.cpu.nzcv,
        memory,
    })
}

fn check(slices: Option<u64>) {
    let words = loop_code::WORDS;
    let code_size = (words.len() as u64 * 4).next_multiple_of(0x1000);
    let image: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let mut m = Machine::with_layout(loop_code::BASE, &image, code_size, DATA, DATA_SIZE);
    loop_code::install();
    let mut random = Random(0xa409_3822_299f_31d0);
    let memory: Vec<u8> = (0..DATA_SIZE).map(|_| random.next() as u8).collect();
    let mut failures = Vec::new();
    let (mut runs, mut boundaries) = (0, 0);
    for &entry in loop_code::PROGRAMS {
        for _ in 0..STATES_PER_PROGRAM {
            let mut before = State {
                x: std::array::from_fn(|_| random.value()),
                sp: random.value(),
                nzcv: (random.next() & 15) as u8,
                memory: memory.clone(),
            };
            before.x[28] = BASE_ADDRESS;
            before.x[27] = random.next() % 0x100;
            let mut expected = before.clone();
            let reference = reference(entry, &mut expected);
            let slice = slices.map(|longest| 1 + random.next() % longest);
            runs += 1;
            let problem = match (
                reference,
                run_lifted(&mut m, entry, &before, slice, &mut boundaries),
            ) {
                (Some(()), Ok(actual)) if actual == expected => continue,
                (Some(()), Ok(actual)) => difference(&expected, &actual),
                (Some(()), Err(trap)) => format!("lifted trapped: {trap:?}"),
                (None, _) => "the reference went astray".to_owned(),
            };
            failures.push(format!(
                "program at {entry:#x} (slice {slice:?}): {problem}"
            ));
            break;
        }
    }
    println!(
        "{} programs, {runs} runs, {boundaries} host boundaries, {} mismatching programs",
        loop_code::PROGRAMS.len(),
        failures.len()
    );
    assert!(slices.is_none() || boundaries > runs as u64);
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

#[test]
fn loops_and_indirect_entries_match_the_reference() {
    check(None);
}

#[test]
fn loops_and_indirect_entries_resume_after_host_boundaries() {
    check(Some(3));
}
