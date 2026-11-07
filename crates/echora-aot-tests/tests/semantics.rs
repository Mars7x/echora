//! Semantic differential test: lifted integer and memory instructions
//! against an independent reference interpreter.
//!
//! `build.rs` picks random encodings of each integer class (ALU, shifts,
//! extends, compares, selects, multiplies, divides, bit operations, loads
//! and stores, conditional branches, CRC32, LSE atomics, acquire/release
//! and exclusive accesses, MRS/MSR NZCV), each followed by SVC #7 and SVC
//! #8 (branches jump to the second), and lifts them through split emission.
//! Every instruction runs natively from random register, SP, flag and
//! memory states; whether a branch was taken, and the registers, SP, NZCV
//! and memory afterwards, must equal what the interpreter below computes
//! from the Arm pseudocode. The
//! interpreter shares only the decoded operand fields with the lifter (the
//! decoder is checked separately against yaxpeax-arm), not its semantics.
#![cfg(any(target_os = "linux", windows))]
#![allow(clippy::unwrap_used)]

mod common;

use common::*;
use echora_aot_tests::{Machine, Trap, semantic_code};
use echora_lift::aarch64::{Address, Instruction, decode};

const STATES_PER_WORD: usize = 24;
/// Bytes per entry: the instruction, SVC #7 and SVC #8.
const STRIDE: u64 = 12;

/// Registers an access reads its address from, pointed into the data.
fn address_registers(instruction: &Instruction) -> Option<(u8, Option<u8>)> {
    match *instruction {
        Instruction::LoadStore { rn, address, .. } => Some((
            rn,
            match address {
                Address::Register { rm, .. } => Some(rm),
                Address::Immediate { .. } => None,
            },
        )),
        Instruction::LoadStorePair { rn, .. }
        | Instruction::CompareAndSwap { rn, .. }
        | Instruction::CompareAndSwapPair { rn, .. }
        | Instruction::AtomicAdd { rn, .. }
        | Instruction::AtomicMemory { rn, .. }
        | Instruction::LoadAcquireRcpc { rn, .. }
        | Instruction::LoadAcquireRcsc { rn, .. }
        | Instruction::StoreRelease { rn, .. }
        | Instruction::LoadExclusive { rn, .. }
        | Instruction::StoreExclusive { rn, .. } => Some((rn, None)),
        _ => None,
    }
}

/// A random state; `memory` is the data area's random contents.
fn random_state(random: &mut Random, instruction: &Instruction, memory: &[u8]) -> State {
    let mut state = State {
        x: std::array::from_fn(|_| random.value()),
        sp: random.value(),
        nzcv: (random.next() & 15) as u8,
        memory: memory.to_vec(),
    };
    if let Some((base, index)) = address_registers(instruction) {
        set_sp(&mut state, base, BASE_ADDRESS);
        if let Some(index) = index {
            // Small, so every extend and shift stays inside the data.
            set(&mut state, index, random.next() % 0x100);
        }
    }
    state
}

/// Runs one lifted instruction: whether it branched, and the state after.
fn run_lifted(m: &mut Machine, entry: u64, state: &State) -> Result<(bool, State), Trap> {
    m.core.memory.clear_exclusive();
    m.core.cpu.regs = state.x;
    m.core.cpu.sp = state.sp;
    m.core.cpu.nzcv = state.nzcv;
    m.core.cpu.pc = entry;
    m.memory.write(DATA, &state.memory).expect("data is mapped");
    m.core.spin_polls = 0;
    let trap = loop {
        match semantic_code::dispatch(&mut m.core) {
            Ok(()) => continue,
            // A CAS that cannot change memory counts as a spin poll, and
            // enough of them end the slice at a block boundary: resume, as
            // the host would.
            Err(Trap::HostBoundary) => {
                m.core.spin_polls = 0;
                m.core.host_check_at = u64::MAX;
            }
            Err(trap) => break trap,
        }
    };
    m.core.settle_block_trap();
    let taken = match trap {
        Trap::SupervisorCall { number: 7 } => false,
        Trap::SupervisorCall { number: 8 } => true,
        other => return Err(other),
    };
    let mut memory = vec![0; DATA_SIZE as usize];
    m.memory.read(DATA, &mut memory).expect("data is mapped");
    Ok((
        taken,
        State {
            x: m.core.cpu.regs,
            sp: m.core.cpu.sp,
            nzcv: m.core.cpu.nzcv,
            memory,
        },
    ))
}

#[test]
fn lifted_integer_and_memory_instructions_match_the_reference() {
    let code_size = (semantic_code::WORDS.len() as u64 * STRIDE).next_multiple_of(0x1000);
    let mut image = Vec::new();
    for word in semantic_code::WORDS {
        image.extend(word.to_le_bytes());
        image.extend(0xd400_00e1_u32.to_le_bytes());
        image.extend(0xd400_0101_u32.to_le_bytes());
    }
    let mut m = Machine::with_layout(semantic_code::BASE, &image, code_size, DATA, DATA_SIZE);
    semantic_code::install();
    let mut random = Random(0xd1ff_e2e4_7a11_0c8b);
    let memory: Vec<u8> = (0..DATA_SIZE).map(|_| random.next() as u8).collect();
    let mut failures = Vec::new();
    let mut runs = 0;
    for (index, &word) in semantic_code::WORDS.iter().enumerate() {
        let entry = semantic_code::BASE + STRIDE * index as u64;
        let instruction = decode(word, entry);
        for _ in 0..STATES_PER_WORD {
            let before = random_state(&mut random, &instruction, &memory);
            let mut expected = before.clone();
            let reference = interpret(&instruction, &mut expected);
            let lifted = run_lifted(&mut m, entry, &before);
            runs += 1;
            let problem = match (reference, lifted) {
                (Some(branch), Ok((taken, actual))) if branch == taken && actual == expected => {
                    continue;
                }
                (Some(branch), Ok((taken, _))) if branch != taken => {
                    format!("branch taken: expected {branch}, lifted {taken}")
                }
                (Some(_), Ok((_, actual))) => difference(&expected, &actual),
                (Some(_), Err(trap)) => format!("lifted trapped: {trap:?}"),
                (None, Ok(_)) => "the reference access left the data".to_owned(),
                (None, Err(_)) => continue,
            };
            failures.push(format!(
                "{word:#010x} {instruction:?}: {problem}\n    from {before:?}"
            ));
            break;
        }
    }
    println!(
        "{} instructions, {runs} runs, {} mismatching instructions",
        semantic_code::WORDS.len(),
        failures.len()
    );
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures
            .iter()
            .take(40)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}
