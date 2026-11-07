//! Sequence differential test: random straight-line sequences of integer,
//! flag and memory instructions with forward branches, lifted through split
//! emission and run natively, against the shared reference interpreter.
//!
//! Single-instruction tests (`semantics.rs`) cannot see how instructions
//! interact inside a block: codegen keeps guest registers and NZCV in
//! locals, stores and reloads them around helpers, and splits blocks at
//! branch targets. Here each sequence of 4 to 12 instructions (a quarter of
//! them replaced by forward B.cond, CBZ/CBNZ, TBZ/TBNZ or B) runs from
//! random states, and the registers, SP, NZCV and data memory at its final
//! SVC must match the reference. X28 holds the data base and X27 a small
//! index; `build.rs` keeps every instruction from writing either.
#![cfg(any(target_os = "linux", windows))]
#![allow(clippy::unwrap_used)]

mod common;

use common::*;
use echora_aot_tests::{Machine, Trap, sequence_code};
use echora_lift::aarch64::{Instruction, decode};

const STATES_PER_SEQUENCE: usize = 24;

/// Runs a sequence from `entry` with the reference: `None` if an access
/// left the data.
fn reference(entry: u64, state: &mut State) -> Option<()> {
    let mut pc = entry;
    loop {
        let index = ((pc - sequence_code::BASE) / 4) as usize;
        let word = sequence_code::WORDS[index];
        if word == 0xd400_00e1 {
            return Some(());
        }
        let instruction = decode(word, pc);
        let taken = match instruction {
            Instruction::Branch { link: false, .. } => true,
            _ => interpret(&instruction, state)?,
        };
        pc = match instruction {
            Instruction::Branch { target, .. }
            | Instruction::BranchCond { target, .. }
            | Instruction::CompareBranch { target, .. }
            | Instruction::TestBranch { target, .. }
                if taken =>
            {
                target
            }
            _ => pc + 4,
        };
    }
}

/// Runs a sequence natively. With `slice`, the host boundary comes every
/// `slice` instructions (rounded to blocks) and execution resumes through
/// the dispatcher, as after a scheduler switch.
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
    let trap = loop {
        match sequence_code::dispatch(&mut m.core) {
            Ok(()) => continue,
            // The end of a slice, or (after CASes that cannot change
            // memory) a spin poll: resume, as the host would.
            Err(Trap::HostBoundary) => {
                *boundaries += 1;
                m.core.spin_polls = 0;
                m.core.host_check_at = next_check(m.core.steps);
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

fn listing(entry: u64) -> String {
    let mut pc = entry;
    let mut lines = Vec::new();
    loop {
        let word = sequence_code::WORDS[((pc - sequence_code::BASE) / 4) as usize];
        lines.push(format!("    {pc:#x}: {word:#010x} {:?}", decode(word, pc)));
        if word == 0xd400_00e1 {
            return lines.join("\n");
        }
        pc += 4;
    }
}

#[test]
fn lifted_sequences_match_the_reference() {
    check_sequences(None);
}

/// The same sequences with one- to three-instruction slices: nearly every
/// block boundary returns to the host and resumes through `dispatch`.
#[test]
fn lifted_sequences_resume_after_host_boundaries() {
    check_sequences(Some(3));
}

fn check_sequences(slices: Option<u64>) {
    let words = sequence_code::WORDS;
    let code_size = (words.len() as u64 * 4).next_multiple_of(0x1000);
    let image: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let mut m = Machine::with_layout(sequence_code::BASE, &image, code_size, DATA, DATA_SIZE);
    sequence_code::install();
    let mut random = Random(0x9b05_688c_2b3e_6c1f);
    let memory: Vec<u8> = (0..DATA_SIZE).map(|_| random.next() as u8).collect();
    let mut failures = Vec::new();
    let mut runs = 0;
    let mut boundaries = 0;
    for &entry in sequence_code::ENTRIES {
        for _ in 0..STATES_PER_SEQUENCE {
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
            runs += 1;
            let slice = slices.map(|longest| 1 + random.next() % longest);
            let problem = match (
                reference,
                run_lifted(&mut m, entry, &before, slice, &mut boundaries),
            ) {
                (Some(()), Ok(actual)) if actual == expected => continue,
                (Some(()), Ok(actual)) => difference(&expected, &actual),
                (Some(()), Err(trap)) => format!("lifted trapped: {trap:?}"),
                (None, _) => "the reference access left the data".to_owned(),
            };
            failures.push(format!(
                "sequence at {entry:#x}: {problem}\n{}\n    from {before:?}",
                listing(entry)
            ));
            break;
        }
    }
    let decoded: Vec<Instruction> = words.iter().map(|&word| decode(word, 0)).collect();
    let branches = decoded
        .iter()
        .filter(|i| {
            matches!(
                i,
                Instruction::Branch { .. }
                    | Instruction::BranchCond { .. }
                    | Instruction::CompareBranch { .. }
                    | Instruction::TestBranch { .. }
            )
        })
        .count();
    let memory_instructions = decoded
        .iter()
        .filter(|i| echora_lift::defuse::effects(i).loads || echora_lift::defuse::effects(i).stores)
        .count();
    println!(
        "{} sequences ({} instructions, {branches} branches, {memory_instructions} memory \
         accesses), {runs} runs, {boundaries} host boundaries, {} mismatching sequences",
        sequence_code::ENTRIES.len(),
        words.len(),
        failures.len()
    );
    assert!(branches > 0 && memory_instructions > 0);
    assert!(
        slices.is_none() || boundaries > runs as u64,
        "slices must end often"
    );
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures
            .iter()
            .take(10)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}
