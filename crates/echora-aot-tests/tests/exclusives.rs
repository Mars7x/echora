//! Exclusive-pair differential test: LDXR/STXR sequences on one guest
//! context and on two contexts interleaved on one memory.
//!
//! The reference monitor follows the runtime's documented model, which the
//! architecture permits: LDXR marks the address, size, value and the
//! granule's global generation; STXR stores (status 0) only if the mark is
//! for the same address and size, no exclusive store succeeded to the
//! granule since (on any context), and the bytes still hold the loaded
//! value; every STXR and CLREX opens the monitor. Plain stores do not
//! advance the generation (the value compare catches those that change the
//! value).
//!
//! Two `Core`s on one `Memory` stand for two processors. They run in turns
//! of one to four instructions, rounded to blocks, and the reference
//! replays exactly the instructions each turn retired, so a successful STXR
//! on one context must make the other's pending pair fail in both.
#![cfg(any(target_os = "linux", windows))]
#![allow(clippy::unwrap_used)]

mod common;

use std::collections::HashMap;

use common::*;
use echora_aot_tests::{Core, Machine, Trap, exclusive_code};
use echora_lift::aarch64::{Instruction, decode};

const STATES_PER_SEQUENCE: usize = 12;
const SVC_EXIT: u32 = 0xd400_00e1;

/// The marked address, size, loaded value and generation.
type Mark = (u64, u8, u64, u64);

/// One context's registers and local monitor.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Context {
    x: [u64; 31],
    sp: u64,
    nzcv: u8,
    pc: u64,
    mark: Option<Mark>,
    done: bool,
}

struct Reference {
    memory: Vec<u8>,
    /// Global monitor generations by 8-byte granule.
    generations: HashMap<u64, u64>,
    /// STXRs that failed only because another context's STXR to the
    /// granule succeeded since the LDXR.
    stolen: u64,
}

impl Reference {
    /// Runs one instruction of `context`; false once it reaches SVC #7.
    fn step(&mut self, context: &mut Context) -> Option<bool> {
        let index = usize::try_from((context.pc - exclusive_code::BASE) / 4).ok()?;
        let word = *exclusive_code::WORDS.get(index)?;
        if word == SVC_EXIT {
            context.done = true;
            return Some(false);
        }
        let instruction = decode(word, context.pc);
        let mut state = State {
            x: context.x,
            sp: context.sp,
            nzcv: context.nzcv,
            memory: std::mem::take(&mut self.memory),
        };
        let result = self.execute(&instruction, &mut state, &mut context.mark);
        context.x = state.x;
        context.sp = state.sp;
        context.nzcv = state.nzcv;
        self.memory = state.memory;
        let taken = result?;
        context.pc = match instruction {
            Instruction::Branch {
                target,
                link: false,
            } => target,
            Instruction::BranchCond { target, .. }
            | Instruction::CompareBranch { target, .. }
            | Instruction::TestBranch { target, .. }
                if taken =>
            {
                target
            }
            _ => context.pc + 4,
        };
        Some(true)
    }

    fn execute(
        &mut self,
        instruction: &Instruction,
        state: &mut State,
        mark: &mut Option<Mark>,
    ) -> Option<bool> {
        match *instruction {
            Instruction::LoadExclusive { rt, rn, bytes, .. } => {
                let address = get_sp(state, rn);
                let value = read(state, address, u64::from(bytes))?;
                let generation = self.generations.get(&(address >> 3)).copied().unwrap_or(0);
                *mark = Some((address, bytes, value, generation));
                set(state, rt, value);
                Some(false)
            }
            Instruction::StoreExclusive {
                rs, rt, rn, bytes, ..
            } => {
                let address = get_sp(state, rn);
                let generation = self.generations.get(&(address >> 3)).copied().unwrap_or(0);
                let stored = match mark.take() {
                    Some((marked, size, value, marked_generation)) => {
                        let unchanged = marked == address
                            && size == bytes
                            && read(state, address, u64::from(bytes))? == value;
                        if unchanged && marked_generation != generation {
                            self.stolen += 1;
                        }
                        unchanged && marked_generation == generation
                    }
                    None => false,
                };
                if stored {
                    let value = get(state, rt);
                    write(state, address, u64::from(bytes), value)?;
                    self.generations.insert(address >> 3, generation + 2);
                }
                set(state, rs, u64::from(!stored));
                Some(false)
            }
            Instruction::ClearExclusive => {
                *mark = None;
                Some(false)
            }
            Instruction::Branch { link: false, .. } => Some(true),
            _ => interpret(instruction, state),
        }
    }
}

fn random_context(random: &mut Random, entry: u64) -> Context {
    let mut x: [u64; 31] = std::array::from_fn(|_| random.value());
    x[28] = BASE_ADDRESS;
    x[27] = random.next() % 0x100;
    Context {
        x,
        sp: random.value(),
        nzcv: (random.next() & 15) as u8,
        pc: entry,
        mark: None,
        done: false,
    }
}

fn load(core: &mut Core, context: &Context) {
    core.cpu.regs = context.x;
    core.cpu.sp = context.sp;
    core.cpu.nzcv = context.nzcv;
    core.cpu.pc = context.pc;
    core.memory.clear_exclusive();
    core.spin_polls = 0;
}

/// Runs `core` for about `slice` instructions (or to its end): the number
/// retired, and whether it reached SVC #7.
fn turn(core: &mut Core, slice: u64) -> Result<(u64, bool), Trap> {
    let start = core.steps;
    core.host_check_at = core.steps + slice;
    core.rearm();
    loop {
        match exclusive_code::dispatch(core) {
            Ok(()) => continue,
            Err(Trap::HostBoundary) => return Ok((core.steps - start, false)),
            Err(Trap::SupervisorCall { number: 7 }) => {
                core.settle_block_trap();
                // The SVC itself is not replayed.
                return Ok((core.steps - start - 1, true));
            }
            Err(trap) => return Err(trap),
        }
    }
}

/// Runs `entries` as contexts on their own cores, in turns; the reference
/// replays each turn. Returns a mismatch description, if any.
fn interleave(
    m: &mut Machine,
    cores: &mut [Core],
    entries: &[u64],
    random: &mut Random,
    memory: &[u8],
    longest: u64,
    stolen: &mut u64,
) -> Option<String> {
    let mut contexts: Vec<Context> = entries
        .iter()
        .map(|&entry| random_context(random, entry))
        .collect();
    let mut reference = Reference {
        memory: memory.to_vec(),
        generations: HashMap::new(),
        stolen: 0,
    };
    m.memory.write(DATA, memory).expect("data is mapped");
    for (core, context) in cores.iter_mut().zip(&contexts) {
        load(core, context);
    }
    let before = contexts.clone();
    let mut finished = vec![false; contexts.len()];
    let mut turns = 0;
    while finished.iter().any(|done| !done) {
        turns += 1;
        if turns > 100_000 {
            return Some("no end".into());
        }
        let which = random.next() as usize % contexts.len();
        if finished[which] {
            continue;
        }
        let slice = 1 + random.next() % longest;
        let (retired, ended) = match turn(&mut cores[which], slice) {
            Ok(result) => result,
            Err(trap) => return Some(format!("context {which} trapped: {trap:?}")),
        };
        for _ in 0..retired {
            if reference.step(&mut contexts[which]) != Some(true) {
                return Some(format!("context {which}: the reference ended early"));
            }
        }
        if ended {
            if reference.step(&mut contexts[which]) != Some(false) {
                return Some(format!("context {which}: the reference did not end"));
            }
            finished[which] = true;
        }
    }
    *stolen += reference.stolen;
    let mut problems = Vec::new();
    for (which, (core, context)) in cores.iter().zip(&contexts).enumerate() {
        let expected = (context.x, context.sp, context.nzcv);
        let actual = (core.cpu.regs, core.cpu.sp, core.cpu.nzcv);
        if expected != actual {
            let lifted = State {
                x: actual.0,
                sp: actual.1,
                nzcv: actual.2,
                memory: Vec::new(),
            };
            let wanted = State {
                x: expected.0,
                sp: expected.1,
                nzcv: expected.2,
                memory: Vec::new(),
            };
            problems.push(format!("context {which}: {}", difference(&wanted, &lifted)));
        }
    }
    let mut lifted_memory = vec![0; DATA_SIZE as usize];
    m.memory
        .read(DATA, &mut lifted_memory)
        .expect("data is mapped");
    if lifted_memory != reference.memory {
        problems.push("memory differs".into());
    }
    if problems.is_empty() {
        None
    } else {
        Some(format!(
            "{} (starts: {:x?})",
            problems.join("; "),
            before.iter().map(|c| c.x).collect::<Vec<_>>()
        ))
    }
}

fn check(contexts: usize, longest: u64) {
    let words = exclusive_code::WORDS;
    let code_size = (words.len() as u64 * 4).next_multiple_of(0x1000);
    let image: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let mut m = Machine::with_layout(exclusive_code::BASE, &image, code_size, DATA, DATA_SIZE);
    exclusive_code::install();
    let mut cores: Vec<Core> = (0..contexts)
        .map(|_| {
            let mut core = Core::new(m.memory.view());
            core.max_steps = u64::MAX;
            core.rearm();
            core
        })
        .collect();
    let mut random = Random(0x0801_f2e2_858e_fc16 ^ contexts as u64 ^ (longest << 8));
    let memory: Vec<u8> = (0..DATA_SIZE).map(|_| random.next() as u8).collect();
    let entries = exclusive_code::ENTRIES;
    let mut failures = Vec::new();
    let mut runs = 0;
    let mut stolen = 0;
    for (index, _) in entries.iter().enumerate() {
        let group: Vec<u64> = (0..contexts)
            .map(|k| entries[(index + k) % entries.len()])
            .collect();
        for _ in 0..STATES_PER_SEQUENCE {
            runs += 1;
            if let Some(problem) = interleave(
                &mut m,
                &mut cores,
                &group,
                &mut random,
                &memory,
                longest,
                &mut stolen,
            ) {
                failures.push(format!("sequences at {group:#x?}: {problem}"));
                break;
            }
        }
    }
    let pairs = words
        .iter()
        .filter(|&&word| matches!(decode(word, 0), Instruction::StoreExclusive { .. }))
        .count();
    println!(
        "{contexts} context(s), {} sequences, {pairs} STXRs, {runs} runs, \
         {stolen} STXRs failed by the other context's, {} mismatching",
        entries.len(),
        failures.len()
    );
    assert!(
        contexts == 1 || stolen > 0,
        "the generation rule was never exercised"
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

/// One test, run in turn: the global monitor is process-wide and keyed by
/// guest address, so tests on separate `Memory`s at the same addresses
/// would advance each other's generations if they ran in parallel.
#[test]
fn exclusive_pairs_match_the_reference() {
    // One context, whole runs.
    check(1, 1 << 20);
    // One context, one- to three-instruction slices.
    check(1, 3);
    // Two contexts in turns of one to four instructions.
    check(2, 4);
}
