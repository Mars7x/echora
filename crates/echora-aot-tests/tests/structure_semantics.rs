//! The structured control flow of `echora_lift::structure`, checked
//! semantically: running a window's `Structured` tree (blocks, loops, ifs,
//! dispatch, breaks and leaves) on the shared reference interpreter must
//! give the same state, and leave the window at the same place, as running
//! the window's code plainly from the same entry.
//!
//! Calls inside a block run to their return in both. Windows of the
//! sequence, counted-loop, switch, call and loop-candidate programs are
//! checked from every control entry and several random states; a leave
//! back into the same window re-enters it there, as the dispatcher does.
#![cfg(any(target_os = "linux", windows))]
#![allow(clippy::unwrap_used)]

mod common;

use std::collections::{BTreeMap, BTreeSet};

use common::*;
use echora_aot_tests::{call_code, loop_code, sequence_code, spin_candidates, switch_code};
use echora_lift::aarch64::{Instruction, decode};
use echora_lift::codegen::{CodeRegion, discover_with_tables};
use echora_lift::structure::{DEFAULT_WINDOW_BITS, Structured, analyze_windows};

const STATES_PER_ENTRY: usize = 6;
const SVC_EXIT: u32 = 0xd400_00e1;
const STACK_TOP: u64 = DATA + 0x4000;
const STEP_LIMIT: usize = 3_000;

/// Name, base, words, entries and data regions of a generated program.
type Program<'a> = (&'a str, u64, &'a [u32], &'a [u64], &'a [CodeRegion]);

/// Where a run stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum End {
    /// Control left the window for this PC.
    Left(u64),
    /// SVC #7 at this PC.
    Exit(u64),
    /// The reference could not continue (an access outside the data or too
    /// many steps): nothing to compare.
    Stuck,
}

struct Code<'a> {
    base: u64,
    words: &'a [u32],
}

impl Code<'_> {
    fn word(&self, pc: u64) -> Option<u32> {
        let index = usize::try_from(pc.checked_sub(self.base)? / 4).ok()?;
        self.words.get(index).copied()
    }

    /// One instruction; calls run to their return. The next PC and
    /// whether a conditional branch was taken; `None` when stuck.
    fn step(&self, pc: u64, state: &mut State, budget: &mut usize) -> Option<(u64, bool)> {
        *budget = budget.checked_sub(1)?;
        let instruction = decode(self.word(pc)?, pc);
        let (next, taken) = match instruction {
            Instruction::Branch { target, link } => {
                if link {
                    state.x[30] = pc + 4;
                    return self
                        .call(target, pc + 4, state, budget)
                        .map(|next| (next, false));
                }
                (target, true)
            }
            Instruction::BranchReg { rn, link } => {
                let target = get(state, rn);
                if link {
                    state.x[30] = pc + 4;
                    return self
                        .call(target, pc + 4, state, budget)
                        .map(|next| (next, false));
                }
                (target, true)
            }
            Instruction::Adr { rd, address } => {
                set(state, rd, address);
                (pc + 4, false)
            }
            _ => {
                let taken = interpret(&instruction, state)?;
                let next = match instruction {
                    Instruction::BranchCond { target, .. }
                    | Instruction::CompareBranch { target, .. }
                    | Instruction::TestBranch { target, .. }
                        if taken =>
                    {
                        target
                    }
                    _ => pc + 4,
                };
                (next, taken)
            }
        };
        Some((next, taken))
    }

    /// Runs a callee until it returns to `back`.
    fn call(&self, target: u64, back: u64, state: &mut State, budget: &mut usize) -> Option<u64> {
        let mut pc = target;
        while pc != back {
            if self.word(pc)? == SVC_EXIT {
                return None;
            }
            pc = self.step(pc, state, budget)?.0;
        }
        Some(back)
    }

    /// Plain execution from `entry` until the window is left or SVC #7.
    fn plain(&self, entry: u64, window: u64, state: &mut State) -> End {
        let mut budget = STEP_LIMIT;
        let mut pc = entry;
        loop {
            if pc >> DEFAULT_WINDOW_BITS != window {
                return End::Left(pc);
            }
            match self.word(pc) {
                Some(SVC_EXIT) => return End::Exit(pc),
                None => return End::Stuck,
                Some(_) => {}
            }
            match self.step(pc, state, &mut budget) {
                Some((next, _)) => pc = next,
                None => return End::Stuck,
            }
        }
    }
}

/// The structured executor's state.
struct Run<'a, 'b> {
    code: &'a Code<'b>,
    blocks: &'a BTreeMap<u64, Vec<u64>>,
    state: State,
    budget: usize,
    /// The last block's final conditional branch: taken or not.
    taken: bool,
    /// Where the last block's own control transfer went.
    next: u64,
}

/// How a structured part ended.
enum Flow {
    Normal,
    Break(usize),
    End(End),
}

impl Run<'_, '_> {
    fn code_block(&mut self, start: u64) -> Option<End> {
        for &pc in &self.blocks[&start] {
            if self.code.word(pc) == Some(SVC_EXIT) {
                return Some(End::Exit(pc));
            }
            match self.code.step(pc, &mut self.state, &mut self.budget) {
                Some((next, taken)) => {
                    self.next = next;
                    self.taken = taken;
                }
                None => return Some(End::Stuck),
            }
        }
        None
    }

    fn run(&mut self, tree: &Structured, entry: u64) -> Flow {
        match tree {
            Structured::Code(start) => match self.code_block(*start) {
                Some(end) => Flow::End(end),
                None => Flow::Normal,
            },
            Structured::Seq(parts) => {
                for part in parts {
                    match self.run(part, entry) {
                        Flow::Normal => {}
                        other => return other,
                    }
                }
                Flow::Normal
            }
            Structured::Block(body) => match self.run(body, entry) {
                Flow::Break(0) => Flow::Normal,
                Flow::Break(depth) => Flow::Break(depth - 1),
                other => other,
            },
            Structured::Loop(body) => loop {
                if self.budget == 0 {
                    return Flow::End(End::Stuck);
                }
                match self.run(body, entry) {
                    Flow::Break(0) => continue,
                    Flow::Break(depth) => return Flow::Break(depth - 1),
                    other => return other,
                }
            },
            Structured::If {
                taken, fallthrough, ..
            } => {
                if self.taken {
                    self.run(taken, entry)
                } else {
                    self.run(fallthrough, entry)
                }
            }
            Structured::Dispatch(cases) => match cases.iter().find(|(pc, _)| *pc == entry) {
                Some((_, body)) => self.run(body, entry),
                None => panic!("no dispatch case for entry {entry:#x}"),
            },
            Structured::Br(depth) => Flow::Break(*depth),
            Structured::Leave(Some(pc)) => Flow::End(End::Left(*pc)),
            Structured::Leave(None) => Flow::End(End::Left(self.next)),
        }
    }
}

fn random_state(random: &mut Random, memory: &[u8]) -> State {
    let mut state = State {
        x: std::array::from_fn(|_| random.value()),
        sp: STACK_TOP,
        nzcv: (random.next() & 15) as u8,
        memory: memory.to_vec(),
    };
    state.x[28] = BASE_ADDRESS;
    state.x[27] = random.next() % 0x100;
    state
}

/// Checks every structured window of one program; returns (windows
/// checked, runs compared).
fn check(program: Program, random: &mut Random, memory: &[u8]) -> (usize, usize) {
    let (name, base, words, entries, data) = program;
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
    let blocks: BTreeMap<u64, Vec<u64>> = translation
        .blocks
        .iter()
        .map(|block| {
            (
                block.start,
                block.instructions.iter().map(|i| i.pc).collect(),
            )
        })
        .collect();
    let code = Code { base, words };
    let (mut windows, mut runs, mut loops) = (0, 0, 0);
    for window in analyze_windows(&translation, DEFAULT_WINDOW_BITS) {
        let Some(structure) = &window.structure else {
            continue;
        };
        windows += 1;
        loops += window.loops.len();
        for entry in window.control_entries() {
            for _ in 0..STATES_PER_ENTRY {
                let start = random_state(random, memory);
                let mut plain = start.clone();
                let plain_end = code.plain(entry, window.window, &mut plain);
                let mut run = Run {
                    code: &code,
                    blocks: &blocks,
                    state: start.clone(),
                    budget: STEP_LIMIT,
                    taken: false,
                    next: entry,
                };
                // A leave that lands back in the window re-enters it there,
                // as the dispatcher does (an indirect branch to one of the
                // window's own entries).
                // A leave to anything else in the window would be a
                // MissingBlock trap (an address built from random registers,
                // say): nothing to compare.
                let entries: BTreeSet<u64> = window.control_entries().collect();
                let mut at = entry;
                let structured_end = loop {
                    match run.run(structure, at) {
                        Flow::End(End::Left(pc)) if pc >> DEFAULT_WINDOW_BITS == window.window => {
                            if !entries.contains(&pc) {
                                break End::Stuck;
                            }
                            at = pc;
                        }
                        Flow::End(end) => break end,
                        Flow::Normal => panic!(
                            "{name}: the structure of window {:#x} fell off its end",
                            window.window
                        ),
                        Flow::Break(depth) => panic!(
                            "{name}: a break of depth {depth} left window {:#x}",
                            window.window
                        ),
                    }
                    if run.budget == 0 {
                        break End::Stuck;
                    }
                };
                if plain_end == End::Stuck || structured_end == End::Stuck {
                    continue;
                }
                runs += 1;
                assert_eq!(
                    structured_end, plain_end,
                    "{name}: window {:#x} from {entry:#x} ends differently",
                    window.window
                );
                assert!(
                    run.state == plain,
                    "{name}: window {:#x} from {entry:#x}: {}",
                    window.window,
                    difference(&plain, &run.state)
                );
            }
        }
    }
    println!("{name}: {windows} structured windows ({loops} natural loops), {runs} runs compared");
    (windows, runs)
}

#[test]
fn structured_windows_run_like_the_plain_code() {
    let mut random = Random(0x6a09_e667_bb67_ae85);
    let mut memory: Vec<u8> = (0..DATA_SIZE).map(|_| random.next() as u8).collect();
    let at = (switch_code::TABLES - DATA) as usize;
    memory[at..at + switch_code::TABLE_BYTES.len()].copy_from_slice(switch_code::TABLE_BYTES);
    let tables = CodeRegion {
        name: "switch tables".into(),
        base: switch_code::TABLES,
        bytes: switch_code::TABLE_BYTES.to_vec(),
    };
    let programs: [Program; 5] = [
        (
            "loop candidates",
            spin_candidates::BASE,
            spin_candidates::WORDS,
            spin_candidates::ENTRIES,
            &[],
        ),
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
            loop_code::ENTRIES,
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
            "calls",
            call_code::BASE,
            call_code::WORDS,
            call_code::CALLERS,
            &[],
        ),
    ];
    let mut total = 0;
    for program in programs {
        let (windows, runs) = check(program, &mut random, &memory);
        assert!(windows > 0, "{}: no structured window", program.0);
        total += runs;
    }
    assert!(total > 1000, "too few runs compared: {total}");
}
