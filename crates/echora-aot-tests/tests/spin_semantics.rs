//! The spin-loop detector on generated programs, checked semantically.
//!
//! `find_spin_loops` claims a loop cannot exit by itself. On one thread
//! with nothing else changing memory, that means: once control has come
//! back to the head, it never leaves. Every reported loop is run with the
//! shared reference from random states, and must keep returning to its
//! head once it has returned at all. The detector runs over 400 random
//! loop candidates (bodies of loads from [X28], compares and random
//! instructions, closed by a backward B.cond, CBZ/CBNZ or TBZ/TBNZ) and over
//! the other semantic programs (sequences, counted loops, switches,
//! exclusives, calls). Pure polls (only loads into the probe register,
//! then a CBZ/CBNZ/TBZ on it) must all be found.
#![cfg(any(target_os = "linux", windows))]
#![allow(clippy::unwrap_used)]

mod common;

use std::collections::BTreeSet;

use common::*;
use echora_aot_tests::{
    call_code, exclusive_code, loop_code, sequence_code, spin_candidates, switch_code,
};
use echora_lift::aarch64::{Instruction, MemoryOp, decode};
use echora_lift::codegen::{CodeRegion, discover_with_tables};
use echora_lift::spin::{SpinLoop, find_spin_loops};

const STATES_PER_LOOP: usize = 32;

/// Name, base, words, entries and data regions of a generated program.
type Program<'a> = (&'a str, u64, &'a [u32], &'a [u64], &'a [CodeRegion]);
const ARRIVALS: usize = 40;

fn word_at(base: u64, words: &[u32], pc: u64) -> Option<u32> {
    let index = usize::try_from(pc.checked_sub(base)? / 4).ok()?;
    words.get(index).copied()
}

/// One reference step: the next PC, or `None` at SVC #7, an indirect
/// target outside the code or an access outside the data.
fn step(base: u64, words: &[u32], pc: u64, state: &mut State) -> Option<u64> {
    let word = word_at(base, words, pc)?;
    if word == 0xd400_00e1 {
        return None;
    }
    let instruction = decode(word, pc);
    Some(match instruction {
        Instruction::Branch { target, link } => {
            if link {
                state.x[30] = pc + 4;
            }
            target
        }
        Instruction::BranchReg { rn, link } => {
            let target = get(state, rn);
            if link {
                state.x[30] = pc + 4;
            }
            target
        }
        Instruction::Adr { rd, address } => {
            set(state, rd, address);
            pc + 4
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
    })
}

/// Runs from the head: whether it left after having come back.
fn exits_after_returning(base: u64, words: &[u32], head: u64, state: &mut State) -> bool {
    let mut pc = head;
    let mut arrivals = 0;
    for _ in 0..20_000 {
        match step(base, words, pc, state) {
            Some(next) => {
                if next == head {
                    arrivals += 1;
                    if arrivals == ARRIVALS {
                        return false;
                    }
                }
                pc = next;
            }
            // Leaving (or an inconclusive access) after a return is an exit.
            None => return arrivals > 0 && word_at(base, words, pc) == Some(0xd400_00e1),
        }
    }
    false
}

/// The detector's verdicts on one program, checked semantically.
fn check(
    name: &str,
    base: u64,
    words: &[u32],
    entries: &[u64],
    data: &[CodeRegion],
    random: &mut Random,
    memory: &[u8],
) -> Vec<SpinLoop> {
    let region = CodeRegion {
        name: name.into(),
        base,
        bytes: words.iter().flat_map(|word| word.to_le_bytes()).collect(),
    };
    let translation = discover_with_tables(
        std::slice::from_ref(&region),
        data,
        entries,
        &[],
        &BTreeSet::new(),
        words.len() + 16,
    )
    .unwrap();
    let loops = find_spin_loops(&translation.blocks);
    for spin in &loops {
        for _ in 0..STATES_PER_LOOP {
            let mut state = State {
                x: std::array::from_fn(|_| random.value()),
                sp: random.value(),
                nzcv: (random.next() & 15) as u8,
                memory: memory.to_vec(),
            };
            state.x[28] = BASE_ADDRESS;
            state.x[27] = random.next() % 0x100;
            let start = state.clone();
            assert!(
                !exits_after_returning(base, words, spin.head, &mut state),
                "{name}: the loop at {:#x} (back-edges {:x?}) was reported as a spin \
                 loop but exited after returning to its head, from {start:?}",
                spin.head,
                spin.back_edges
            );
        }
    }
    println!("{name}: {} spin loops", loops.len());
    loops
}

/// Candidate loops whose body only loads the probe register and whose
/// branch tests it: they must be found.
fn pure_polls() -> BTreeSet<u64> {
    let words = spin_candidates::WORDS;
    let base = spin_candidates::BASE;
    spin_candidates::ENTRIES
        .iter()
        .copied()
        .filter(|&head| {
            let mut pc = head;
            let mut probe = None;
            loop {
                let instruction = decode(word_at(base, words, pc).unwrap(), pc);
                match instruction {
                    Instruction::LoadStore {
                        rt,
                        rn: 28,
                        op: MemoryOp::Load { .. },
                        ..
                    } if probe.is_none_or(|probe| probe == rt) => probe = Some(rt),
                    Instruction::CompareBranch { rt, target, .. }
                    | Instruction::TestBranch { rt, target, .. } => {
                        return target == head && probe == Some(rt);
                    }
                    _ => return false,
                }
                pc += 4;
            }
        })
        .collect()
}

#[test]
fn reported_spin_loops_never_exit_by_themselves() {
    let mut random = Random(0xbe54_66cf_34e9_0c6c);
    let mut memory: Vec<u8> = (0..DATA_SIZE).map(|_| random.next() as u8).collect();
    // Zero a few words the candidates load, so some polls really spin.
    for word in 0..64 {
        if random.next().is_multiple_of(2) {
            let at = (BASE_ADDRESS - DATA) as usize + 8 * word;
            memory[at..at + 4].fill(0);
        }
    }
    let candidates = check(
        "candidates",
        spin_candidates::BASE,
        spin_candidates::WORDS,
        spin_candidates::ENTRIES,
        &[],
        &mut random,
        &memory,
    );
    let found: BTreeSet<u64> = candidates.iter().map(|spin| spin.head).collect();
    let polls = pure_polls();
    println!(
        "pure polls {}, all found: {}",
        polls.len(),
        polls.is_subset(&found)
    );
    assert!(!polls.is_empty());
    assert!(
        polls.is_subset(&found),
        "pure polls not found: {:x?}",
        polls.difference(&found).collect::<Vec<_>>()
    );
    // The other programs' loops all change state each time around.
    let tables = CodeRegion {
        name: "switch tables".into(),
        base: switch_code::TABLES,
        bytes: switch_code::TABLE_BYTES.to_vec(),
    };
    let programs: [Program; 5] = [
        (
            "sequences",
            sequence_code::BASE,
            sequence_code::WORDS,
            sequence_code::ENTRIES,
            &[],
        ),
        (
            "counted loops",
            loop_code::BASE,
            loop_code::WORDS,
            loop_code::PROGRAMS,
            &[],
        ),
        (
            "switches",
            switch_code::BASE,
            switch_code::WORDS,
            switch_code::ENTRIES,
            std::slice::from_ref(&tables),
        ),
        (
            "exclusives",
            exclusive_code::BASE,
            exclusive_code::WORDS,
            exclusive_code::ENTRIES,
            &[],
        ),
        (
            "calls",
            call_code::BASE,
            call_code::WORDS,
            call_code::CALLERS,
            &[],
        ),
    ];
    for (name, base, words, entries, data) in programs {
        check(name, base, words, entries, data, &mut random, &memory);
    }
}
