//! Flag setters fused with their B.cond (`EmitOptions::fused_flags`),
//! against the reference.
//!
//! `build.rs` (`flag_code`) lifts, with fused flags on, one program per
//! window: SUBS, CMP, CMN, ADDS (shifted, immediate and extended, 32 and
//! 64 bits) and ANDS/TST (register and immediate) before a B.cond with
//! every condition, with successors that overwrite NZCV first (the pair
//! fuses) or that need it: a CSEL in a successor, the SVC ending the
//! program, a CCMP before the branch or in a successor, an indirect
//! branch to a block reading NZCV (the pair must not fuse). Each program
//! runs from states with operands at the edges of every condition, with
//! and without one- to three-instruction slices; registers and NZCV must
//! match the reference.
#![cfg(any(target_os = "linux", windows))]
#![allow(clippy::unwrap_used)]

mod common;

use common::*;
use echora_aot_tests::{Machine, Trap, flag_code};
use echora_lift::aarch64::{Instruction, decode};

const SVC_EXIT: u32 = 0xd400_00e1;

/// Operands at the edges of the conditions, 32- and 64-bit.
const EDGES: &[u64] = &[
    0,
    1,
    5,
    0xff,
    0x7fff_ffff,
    0x8000_0000,
    0xffff_fffb,
    0xffff_ffff,
    0x1_0000_0005,
    0x7fff_ffff_ffff_ffff,
    0x8000_0000_0000_0000,
    u64::MAX - 4,
    u64::MAX,
];

fn word_at(pc: u64) -> Option<u32> {
    let index = usize::try_from(pc.checked_sub(flag_code::BASE)? / 4).ok()?;
    flag_code::WORDS.get(index).copied()
}

/// Runs a program with the reference; `None` if it went astray.
fn reference(entry: u64, state: &mut State) -> Option<()> {
    let mut pc = entry;
    for _ in 0..100 {
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
            Instruction::BranchReg { rn, .. } => get(state, rn),
            Instruction::Branch {
                target,
                link: false,
            } => target,
            _ => match (interpret(&instruction, state)?, instruction) {
                (true, Instruction::BranchCond { target, .. }) => target,
                _ => pc + 4,
            },
        };
    }
    None
}

fn run_lifted(
    m: &mut Machine,
    entry: u64,
    state: &State,
    slice: Option<u64>,
) -> Result<State, Trap> {
    m.core.cpu.regs = state.x;
    m.core.cpu.sp = state.sp;
    m.core.cpu.nzcv = state.nzcv;
    m.core.cpu.pc = entry;
    let next_check = |steps: u64| slice.map_or(u64::MAX, |slice| steps + slice);
    m.core.host_check_at = next_check(m.core.steps);
    let trap = loop {
        match flag_code::dispatch(&mut m.core) {
            Ok(()) => continue,
            Err(Trap::HostBoundary) => m.core.host_check_at = next_check(m.core.steps),
            Err(trap) => break trap,
        }
    };
    m.core.settle_block_trap();
    if !matches!(trap, Trap::SupervisorCall { number: 7 }) {
        return Err(trap);
    }
    Ok(State {
        x: m.core.cpu.regs,
        sp: m.core.cpu.sp,
        nzcv: m.core.cpu.nzcv,
        memory: state.memory.clone(),
    })
}

#[test]
fn pairs_fuse_exactly_where_nzcv_is_dead_after_them() {
    let wrong: Vec<_> = flag_code::FUSED
        .iter()
        .filter(|(_, expected, fused)| expected != fused)
        .collect();
    assert!(wrong.is_empty(), "(entry, should fuse, fused): {wrong:#x?}");
    let fused = flag_code::FUSED.iter().filter(|entry| entry.2).count();
    assert!(fused >= 200, "only {fused} programs fused");
}

fn check(slices: Option<u64>) {
    let words = flag_code::WORDS;
    let code_size = (words.len() as u64 * 4).next_multiple_of(0x1000);
    let image: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let mut m = Machine::with_layout(flag_code::BASE, &image, code_size, DATA, DATA_SIZE);
    flag_code::install();
    let mut random = Random(0x5eed_f1a6_0000_0001);
    let mut failures = Vec::new();
    let mut runs = 0;
    for &(entry, _, _) in flag_code::FUSED {
        'states: for (index, &x1) in EDGES.iter().enumerate() {
            for &x2 in EDGES.iter().skip(index % 3).step_by(3) {
                let mut before = State {
                    x: std::array::from_fn(|_| random.value()),
                    sp: random.value(),
                    nzcv: (random.next() & 15) as u8,
                    memory: Vec::new(),
                };
                before.x[1] = x1;
                before.x[2] = x2;
                let mut expected = before.clone();
                let reference = reference(entry, &mut expected);
                let slice = slices.map(|longest| 1 + random.next() % longest);
                runs += 1;
                let problem = match (reference, run_lifted(&mut m, entry, &before, slice)) {
                    (Some(()), Ok(actual)) if actual == expected => continue,
                    (Some(()), Ok(actual)) => difference(&expected, &actual),
                    (Some(()), Err(trap)) => format!("lifted trapped: {trap:?}"),
                    (None, _) => "the reference went astray".to_owned(),
                };
                failures.push(format!(
                    "program at {entry:#x}, x1 {x1:#x}, x2 {x2:#x} (slice {slice:?}): {problem}"
                ));
                break 'states;
            }
        }
    }
    println!("{runs} runs, {} mismatching programs", failures.len());
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
fn fused_pairs_match_the_reference() {
    check(None);
}

#[test]
fn fused_pairs_match_the_reference_across_host_boundaries() {
    check(Some(3));
}
