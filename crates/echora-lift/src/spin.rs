//! Static detection of guest spin-wait loops.
//!
//! A spin-wait loop polls memory that another thread changes and does
//! nothing else, so while it runs on a shared host worker it only burns that
//! worker's time slice. [`find_spin_loops`] finds loops that provably cannot
//! exit by themselves: every path from the loop head T back to T within a
//! small window only loads, does register and flag arithmetic, takes direct
//! branches and runs barriers, and writes nothing that is live at T. Each
//! such iteration leaves the thread exactly as it found it, so only another
//! thread can make the loop exit, and translated code may hand the core on at
//! the loop's back-edges.
//!
//! Three kinds of memory write are also allowed, because they too leave the
//! state unchanged across an iteration:
//!
//! * an exclusive add of a constant (LDXR, ADD/SUB #imm, STXR) whose sum over
//!   every path back to T is zero on each address, such as a reader count
//!   taken and dropped again;
//! * a lock taken with LDXR, a zero test and STXR, and released with a store
//!   of zero (STLR/STR WZR) to the same address on every path back to T;
//! * at most [`INVARIANT_STORE_LIMIT`] other stores whose value and address
//!   registers the loop never writes.
//!
//! An exclusive store counts only on the edge where its status register
//! reports success (CBZ/CBNZ on the status register right after it).
//! Addresses are compared by base register, which the loop must not write.
//!
//! Paths longer than the window are treated as exits. Indirect branches other
//! than returns, and branches to undiscovered code, reachable from T within
//! the window reject the loop, since they might lead back to T.

use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};

use serde::Serialize;

use crate::aarch64::{Address, Instruction, MemoryOp, Writeback};
use crate::codegen::BasicBlock;
use crate::defuse::{Effects, Regs, effects};

/// The default window: the longest path from the head back to the head, in
/// instructions, that the detector considers part of the loop.
pub const DEFAULT_SPIN_WINDOW: usize = 64;

/// The most loop-invariant plain stores a spin loop may contain.
pub const INVARIANT_STORE_LIMIT: usize = 2;

/// Upper bound on (instruction, path state) pairs explored per loop; a loop
/// that needs more is rejected.
const PATH_STATE_LIMIT: usize = 1 << 16;

/// A loop that cannot exit by itself.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct SpinLoop {
    /// The loop head: the target of the back-edges.
    pub head: u64,
    /// Addresses of the direct branches in the loop that jump to `head`,
    /// in ascending order. A handoff belongs on these branches' taken edge.
    pub back_edges: Vec<u64>,
}

/// [`find_spin_loops_within`] with [`DEFAULT_SPIN_WINDOW`].
pub fn find_spin_loops(blocks: &[BasicBlock]) -> Vec<SpinLoop> {
    find_spin_loops_within(blocks, DEFAULT_SPIN_WINDOW)
}

/// Lists the spin-wait loops among `blocks`, ordered by head address.
///
/// Candidate heads are the targets of backward direct branches (target at or
/// before the branch). `window` bounds, in instructions, the paths from the
/// head back to the head that make up the loop.
pub fn find_spin_loops_within(blocks: &[BasicBlock], window: usize) -> Vec<SpinLoop> {
    let code: BTreeMap<u64, &Instruction> = blocks
        .iter()
        .flat_map(|block| block.instructions.iter())
        .map(|record| (record.pc, &record.instruction))
        .collect();
    let heads: BTreeSet<u64> = code
        .iter()
        .filter_map(|(&pc, instruction)| direct_target(instruction).filter(|&target| target <= pc))
        .filter(|target| code.contains_key(target))
        .collect();
    heads
        .into_iter()
        .filter_map(|head| analyze(&code, head, window))
        .collect()
}

/// The taken target of a direct, non-linking branch.
fn direct_target(instruction: &Instruction) -> Option<u64> {
    match *instruction {
        Instruction::Branch {
            target,
            link: false,
        }
        | Instruction::BranchCond { target, .. }
        | Instruction::CompareBranch { target, .. }
        | Instruction::TestBranch { target, .. } => Some(target),
        _ => None,
    }
}

/// Where control can go after `instruction` runs at `pc`, with calls and
/// SVCs returning to the next instruction. `None` means unknown (an indirect
/// branch that is not a return); an empty list means the code is left (a
/// return or a trap).
fn flow(pc: u64, instruction: &Instruction) -> Option<Vec<u64>> {
    let next = pc.wrapping_add(4);
    match *instruction {
        Instruction::Branch {
            target,
            link: false,
        } => Some(vec![target]),
        Instruction::BranchCond { target, .. }
        | Instruction::CompareBranch { target, .. }
        | Instruction::TestBranch { target, .. } => Some(vec![target, next]),
        Instruction::BranchReg {
            rn: 30,
            link: false,
        }
        | Instruction::PAuthDisabledReturn { .. }
        | Instruction::Undefined
        | Instruction::Unsupported => Some(vec![]),
        Instruction::BranchReg { link: false, .. } => None,
        _ => Some(vec![next]),
    }
}

struct Graph<'a> {
    code: &'a BTreeMap<u64, &'a Instruction>,
    /// Instructions reachable from the head, with their shortest distance.
    distance: BTreeMap<u64, usize>,
    /// Successors of each explored instruction (only those explored).
    successors: BTreeMap<u64, Vec<u64>>,
    /// Instructions with a successor that was not explored or is unknown.
    leaves: BTreeSet<u64>,
}

fn analyze(code: &BTreeMap<u64, &Instruction>, head: u64, window: usize) -> Option<SpinLoop> {
    if window == 0 {
        return None;
    }
    let graph = explore(code, head, window)?;
    let members = loop_members(&graph, head, window);
    if !members.contains(&head) {
        return None;
    }
    let back_edges: Vec<u64> = members
        .iter()
        .copied()
        .filter(|pc| direct_target(code[pc]) == Some(head))
        .collect();
    if back_edges.is_empty() {
        return None;
    }

    let mut written = Regs::NONE;
    let mut plain_stores = BTreeSet::new();
    let mut exclusive_stores = false;
    let mut all_effects = BTreeMap::new();
    for &pc in &members {
        let e = effects(code[&pc]);
        if e.calls || e.svc || e.system || e.event_wait || e.traps || e.atomic {
            return None;
        }
        written = written.union(e.writes);
        all_effects.insert(pc, e);
    }
    for (&pc, e) in &all_effects {
        if !e.stores {
            continue;
        }
        if e.exclusive {
            exclusive_stores = true;
        } else if e.writes.is_empty() && !e.reads.intersects(written) {
            plain_stores.insert(pc);
        } else {
            return None;
        }
    }

    if written.intersects(live_at(&graph, head)) {
        return None;
    }

    let releases = if exclusive_stores {
        simulate_paths(&graph, &members, head, window, written)?
    } else {
        BTreeSet::new()
    };
    if plain_stores.difference(&releases).count() > INVARIANT_STORE_LIMIT {
        return None;
    }
    Some(SpinLoop { head, back_edges })
}

/// Breadth-first exploration from `head` up to `window - 1` instructions
/// deep. Fails when an unknown successor is reachable.
fn explore<'a>(
    code: &'a BTreeMap<u64, &'a Instruction>,
    head: u64,
    window: usize,
) -> Option<Graph<'a>> {
    let mut graph = Graph {
        code,
        distance: BTreeMap::from([(head, 0)]),
        successors: BTreeMap::new(),
        leaves: BTreeSet::new(),
    };
    let mut queue = VecDeque::from([head]);
    while let Some(pc) = queue.pop_front() {
        let depth = graph.distance[&pc];
        let targets = flow(pc, code[&pc])?;
        if targets.is_empty() {
            graph.leaves.insert(pc);
        }
        let mut explored = Vec::new();
        for target in targets {
            if !code.contains_key(&target) {
                return None;
            }
            if !graph.distance.contains_key(&target) {
                if depth + 1 >= window {
                    graph.leaves.insert(pc);
                    continue;
                }
                graph.distance.insert(target, depth + 1);
                queue.push_back(target);
            }
            explored.push(target);
        }
        graph.successors.insert(pc, explored);
    }
    Some(graph)
}

/// Instructions on some path from `head` back to `head` of at most `window`
/// instructions.
fn loop_members(graph: &Graph, head: u64, window: usize) -> BTreeSet<u64> {
    let mut predecessors: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    for (&pc, targets) in &graph.successors {
        for &target in targets {
            predecessors.entry(target).or_default().push(pc);
        }
    }
    // Instructions from each one up to and including the edge into the head.
    let mut remaining: BTreeMap<u64, usize> = BTreeMap::new();
    let mut queue = VecDeque::new();
    for &pc in predecessors.get(&head).into_iter().flatten() {
        if remaining.insert(pc, 1).is_none() {
            queue.push_back(pc);
        }
    }
    while let Some(pc) = queue.pop_front() {
        let count = remaining[&pc] + 1;
        for &before in predecessors.get(&pc).into_iter().flatten() {
            if let std::collections::btree_map::Entry::Vacant(slot) = remaining.entry(before) {
                slot.insert(count);
                queue.push_back(before);
            }
        }
    }
    remaining
        .into_iter()
        .filter(|(pc, count)| graph.distance[pc] + count <= window)
        .map(|(pc, _)| pc)
        .collect()
}

/// Registers that may be read on some path from `head` before being written.
/// Everything is live where exploration stops, at returns and traps, and
/// across calls and SVCs.
fn live_at(graph: &Graph, head: u64) -> Regs {
    let effects: BTreeMap<u64, Effects> = graph
        .successors
        .keys()
        .map(|&pc| (pc, effects(graph.code[&pc])))
        .collect();
    let mut live: BTreeMap<u64, Regs> = BTreeMap::new();
    let mut changed = true;
    while changed {
        changed = false;
        for (&pc, targets) in graph.successors.iter().rev() {
            let e = &effects[&pc];
            let live_in = if e.calls || e.svc || graph.leaves.contains(&pc) {
                Regs::ALL
            } else {
                let live_out = targets.iter().fold(Regs::NONE, |out, target| {
                    out.union(live.get(target).copied().unwrap_or_default())
                });
                e.reads.union(live_out.difference(e.writes))
            };
            if live.get(&pc) != Some(&live_in) {
                live.insert(pc, live_in);
                changed = true;
            }
        }
    }
    live.get(&head).copied().unwrap_or(Regs::ALL)
}

/// An address used by exclusive accesses: base register and access size.
type Location = (u8, u8);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Monitor {
    location: Location,
    /// The loaded value has been tested equal to zero on this path.
    zero: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Update {
    Add(i64),
    Acquire,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Pending {
    status: u8,
    location: Location,
    update: Update,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
struct PathState {
    monitor: Option<Monitor>,
    /// Registers holding the exclusively loaded value plus a constant.
    derived: BTreeMap<u8, i64>,
    /// An exclusive store whose status has not been tested yet.
    pending: Option<Pending>,
    /// Net exclusive adds per location, without zero entries.
    deltas: BTreeMap<Location, i64>,
    held: BTreeSet<Location>,
}

impl PathState {
    fn at_rest(&self) -> bool {
        self.pending.is_none() && self.deltas.is_empty() && self.held.is_empty()
    }

    fn commit(&mut self, pending: Pending) -> Option<()> {
        match pending.update {
            Update::Add(amount) => {
                let total = self
                    .deltas
                    .get(&pending.location)
                    .copied()
                    .unwrap_or(0)
                    .wrapping_add(amount);
                if total == 0 {
                    self.deltas.remove(&pending.location);
                } else {
                    self.deltas.insert(pending.location, total);
                }
            }
            Update::Acquire => {
                if !self.held.insert(pending.location) {
                    return None;
                }
            }
        }
        Some(())
    }
}

/// Follows every path from `head` back to `head` through the loop, checking
/// that exclusive adds cancel and locks are released. Returns the plain
/// stores that served as lock releases.
fn simulate_paths(
    graph: &Graph,
    members: &BTreeSet<u64>,
    head: u64,
    window: usize,
    written: Regs,
) -> Option<BTreeSet<u64>> {
    let mut releases = BTreeSet::new();
    let mut seen = HashSet::new();
    let mut stack = vec![(head, 0_usize, PathState::default())];
    while let Some((pc, steps, state)) = stack.pop() {
        if !seen.insert((pc, steps, state.clone())) {
            continue;
        }
        if seen.len() > PATH_STATE_LIMIT {
            return None;
        }
        for (target, next) in step(pc, graph.code[&pc], state, written, &mut releases)? {
            if target == head {
                if !next.at_rest() {
                    return None;
                }
            } else if members.contains(&target) && steps + 2 <= window {
                stack.push((target, steps + 1, next));
            }
        }
    }
    Some(releases)
}

/// One instruction of a path: its successors with the path state on each.
fn step(
    pc: u64,
    instruction: &Instruction,
    mut state: PathState,
    written: Regs,
    releases: &mut BTreeSet<u64>,
) -> Option<Vec<(u64, PathState)>> {
    let next = pc.wrapping_add(4);
    let e = effects(instruction);
    if let Some(pending) = state.pending {
        if let Instruction::CompareBranch {
            rt,
            nonzero,
            target,
            ..
        } = *instruction
            && rt == pending.status
        {
            // STXR writes 0 on success.
            let (success, failure) = if nonzero {
                (next, target)
            } else {
                (target, next)
            };
            state.pending = None;
            let failed = state.clone();
            state.commit(pending)?;
            return Some(vec![(success, state), (failure, failed)]);
        }
        let status = Regs::gpr(pending.status);
        if e.branch || e.reads.intersects(status) || e.writes.intersects(status) {
            return None;
        }
    }

    match *instruction {
        Instruction::LoadExclusive { rt, rn, bytes, .. } => {
            if written.has_gpr_or_sp(rn) {
                return None;
            }
            state.monitor = Some(Monitor {
                location: (rn, bytes),
                zero: false,
            });
            state.derived.clear();
            if rt != 31 {
                state.derived.insert(rt, 0);
            }
        }
        Instruction::AddSubImmediate {
            rd,
            rn,
            immediate,
            subtract,
            ..
        } => {
            let source = if rn == 31 {
                None
            } else {
                state.derived.get(&rn).copied()
            };
            state.derived.remove(&rd);
            if let (Some(base), true) = (source, rd != 31) {
                let amount = immediate as i64;
                let value = if subtract {
                    base.wrapping_sub(amount)
                } else {
                    base.wrapping_add(amount)
                };
                state.derived.insert(rd, value);
            }
        }
        Instruction::StoreExclusive {
            rs, rt, rn, bytes, ..
        } => {
            let monitor = state.monitor.filter(|m| m.location == (rn, bytes))?;
            let update = match state.derived.get(&rt) {
                Some(&amount) if rt != 31 => Update::Add(amount),
                _ if monitor.zero && !written.has_gpr(rt) => Update::Acquire,
                _ => return None,
            };
            state.pending = Some(Pending {
                status: rs,
                location: monitor.location,
                update,
            });
            state.monitor = None;
            state.derived.clear();
        }
        Instruction::CompareBranch {
            rt,
            nonzero,
            target,
            ..
        } if state.monitor.is_some() && state.derived.get(&rt) == Some(&0) => {
            let (zero, other) = if nonzero {
                (next, target)
            } else {
                (target, next)
            };
            let mut tested = state.clone();
            if let Some(monitor) = tested.monitor.as_mut() {
                monitor.zero = true;
            }
            return Some(vec![(zero, tested), (other, state)]);
        }
        Instruction::ClearExclusive => {
            state.monitor = None;
            state.derived.clear();
        }
        Instruction::StoreRelease { rt: 31, rn, bytes }
        | Instruction::LoadStore {
            rt: 31,
            rn,
            bytes,
            op: MemoryOp::Store,
            address:
                Address::Immediate {
                    offset: 0,
                    writeback: Writeback::None,
                },
        } => {
            if state.held.remove(&(rn, bytes)) {
                releases.insert(pc);
            }
        }
        _ => {
            state.derived.retain(|&reg, _| !e.writes.has_gpr(reg));
        }
    }
    let targets = flow(pc, instruction)?;
    Some(
        targets
            .into_iter()
            .map(|target| (target, state.clone()))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::{CodeRegion, discover};

    const BASE: u64 = 0x8000;

    fn at(index: u64) -> u64 {
        BASE + 4 * index
    }

    fn spin_loops_within(words: &[u32], window: usize) -> Vec<SpinLoop> {
        let region = CodeRegion {
            name: "test".into(),
            base: BASE,
            bytes: words.iter().flat_map(|word| word.to_le_bytes()).collect(),
        };
        let translation = discover(&[region], &[BASE], 4096).expect("discover test code");
        find_spin_loops_within(&translation.blocks, window)
    }

    fn spin_loops(words: &[u32]) -> Vec<SpinLoop> {
        spin_loops_within(words, DEFAULT_SPIN_WINDOW)
    }

    fn found(head: u64, back_edges: &[u64]) -> SpinLoop {
        SpinLoop {
            head,
            back_edges: back_edges.to_vec(),
        }
    }

    #[test]
    fn finds_a_flag_poll() {
        let words = [
            0x08df_fc08, // 0: wait: ldarb w8, [x0]
            0x34ff_ffe8, // 1: cbz w8, wait
            0xd65f_03c0, // 2: ret
        ];
        assert_eq!(spin_loops(&words), vec![found(at(0), &[at(1)])]);
    }

    #[test]
    fn finds_a_poll_with_a_barrier_and_flag_compare() {
        let words = [
            0xb940_0808, // 0: wait: ldr w8, [x0, #8]
            0xd503_3bbf, // 1: dmb ish
            0x7100_051f, // 2: cmp w8, #1
            0x54ff_ffa1, // 3: b.ne wait
            0xd65f_03c0, // 4: ret
        ];
        assert_eq!(spin_loops(&words), vec![found(at(0), &[at(3)])]);
    }

    #[test]
    fn finds_a_ticket_lock_wait() {
        let words = [
            0x885f_fc08, // 0: lock: ldaxr w8, [x0]
            0x1140_4109, // 1: add w9, w8, #0x10, lsl #12
            0x880a_7c09, // 2: stxr w10, w9, [x0]
            0x35ff_ffaa, // 3: cbnz w10, lock
            0x5310_7d0b, // 4: lsr w11, w8, #16
            0x6b28_217f, // 5: cmp w11, w8, uxth
            0x5400_0080, // 6: b.eq done
            0x48df_fc08, // 7: wait: ldarh w8, [x0]
            0x6b0b_011f, // 8: cmp w8, w11
            0x54ff_ffc1, // 9: b.ne wait
            0xd65f_03c0, // 10: done: ret
        ];
        // The ticket is taken once: the STXR only loops back when it fails,
        // so the retry loop changes nothing either.
        assert_eq!(
            spin_loops(&words),
            vec![found(at(0), &[at(3)]), found(at(7), &[at(9)])]
        );
    }

    #[test]
    fn finds_an_exclusive_add_undone_on_every_path() {
        let words = [
            0x885f_fc08, // 0: retry: ldaxr w8, [x0]
            0x1100_0508, // 1: add w8, w8, #1
            0x8809_7c08, // 2: stxr w9, w8, [x0]
            0x35ff_ffa9, // 3: cbnz w9, retry
            0x88df_fc2a, // 4: ldar w10, [x1]
            0x3400_00ca, // 5: cbz w10, done
            0x885f_fc08, // 6: undo: ldaxr w8, [x0]
            0x5100_0508, // 7: sub w8, w8, #1
            0x8809_fc08, // 8: stlxr w9, w8, [x0]
            0x35ff_ffa9, // 9: cbnz w9, undo
            0x17ff_fff6, // 10: b retry
            0xd65f_03c0, // 11: done: ret
        ];
        let loops = spin_loops(&words);
        assert!(loops.contains(&found(at(0), &[at(3), at(10)])), "{loops:?}");
    }

    #[test]
    fn rejects_an_exclusive_add_left_in_place_on_one_path() {
        let words = [
            0x885f_fc08, // 0: retry: ldaxr w8, [x0]
            0x1100_0508, // 1: add w8, w8, #1
            0x8809_7c08, // 2: stxr w9, w8, [x0]
            0x35ff_ffa9, // 3: cbnz w9, retry
            0x88df_fc2a, // 4: ldar w10, [x1]
            0x3400_010a, // 5: cbz w10, done
            0xb940_004c, // 6: ldr w12, [x2]
            0x35ff_ff2c, // 7: cbnz w12, retry
            0x885f_fc08, // 8: undo: ldaxr w8, [x0]
            0x5100_0508, // 9: sub w8, w8, #1
            0x8809_fc08, // 10: stlxr w9, w8, [x0]
            0x35ff_ffa9, // 11: cbnz w9, undo
            0x17ff_fff4, // 12: b retry
            0xd65f_03c0, // 13: done: ret
        ];
        let loops = spin_loops(&words);
        assert!(loops.iter().all(|found| found.head != at(0)), "{loops:?}");
    }

    #[test]
    fn finds_a_lock_taken_and_released() {
        let words = [
            0x885f_fc08, // 0: retry: ldaxr w8, [x0]
            0x35ff_ffe8, // 1: cbnz w8, retry
            0x8809_7c0a, // 2: stxr w9, w10, [x0]
            0x35ff_ffa9, // 3: cbnz w9, retry
            0xb940_002b, // 4: ldr w11, [x1]
            0x889f_fc1f, // 5: stlr wzr, [x0]
            0x34ff_ff4b, // 6: cbz w11, retry
            0xd65f_03c0, // 7: ret
        ];
        assert_eq!(
            spin_loops(&words),
            vec![found(at(0), &[at(1), at(3), at(6)])]
        );
    }

    #[test]
    fn rejects_a_lock_still_held_at_the_head() {
        let words = [
            0x885f_fc08, // 0: retry: ldaxr w8, [x0]
            0x35ff_ffe8, // 1: cbnz w8, retry
            0x8809_7c0a, // 2: stxr w9, w10, [x0]
            0x35ff_ffa9, // 3: cbnz w9, retry
            0xb940_002b, // 4: ldr w11, [x1]
            0x34ff_ff6b, // 5: cbz w11, retry
            0x889f_fc1f, // 6: stlr wzr, [x0]
            0xd65f_03c0, // 7: ret
        ];
        assert_eq!(spin_loops(&words), vec![]);
    }

    #[test]
    fn rejects_a_loop_with_a_call() {
        let words = [
            0x9400_0004, // 0: wait: bl helper
            0xb940_0268, // 1: ldr w8, [x19]
            0x34ff_ffc8, // 2: cbz w8, wait
            0xd65f_03c0, // 3: ret
            0xd65f_03c0, // 4: helper: ret
        ];
        assert_eq!(spin_loops(&words), vec![]);
    }

    #[test]
    fn rejects_a_loop_with_a_supervisor_call() {
        let words = [
            0xd400_00e1, // 0: wait: svc #7
            0xb940_0268, // 1: ldr w8, [x19]
            0x34ff_ffc8, // 2: cbz w8, wait
            0xd65f_03c0, // 3: ret
        ];
        assert_eq!(spin_loops(&words), vec![]);
    }

    #[test]
    fn rejects_a_loop_that_counts() {
        let words = [
            0x9100_0694, // 0: wait: add x20, x20, #1
            0xb940_0268, // 1: ldr w8, [x19]
            0x34ff_ffc8, // 2: cbz w8, wait
            0xaa14_03e0, // 3: mov x0, x20
            0xd65f_03c0, // 4: ret
        ];
        assert_eq!(spin_loops(&words), vec![]);
    }

    #[test]
    fn rejects_a_loop_that_walks_its_pointer() {
        let words = [
            0xb840_4408, // 0: wait: ldr w8, [x0], #4
            0x34ff_ffe8, // 1: cbz w8, wait
            0xd65f_03c0, // 2: ret
        ];
        assert_eq!(spin_loops(&words), vec![]);
    }

    #[test]
    fn rejects_a_store_of_a_changing_value() {
        let words = [
            0xb940_0008, // 0: wait: ldr w8, [x0]
            0xb900_0028, // 1: str w8, [x1]
            0x34ff_ffc8, // 2: cbz w8, wait
            0xd65f_03c0, // 3: ret
        ];
        assert_eq!(spin_loops(&words), vec![]);
    }

    #[test]
    fn allows_two_invariant_stores_but_not_three() {
        let two = [
            0xb900_003f, // 0: wait: str wzr, [x1]
            0xa901_0c82, // 1: stp x2, x3, [x4, #16]
            0xb940_0008, // 2: ldr w8, [x0]
            0x34ff_ffa8, // 3: cbz w8, wait
            0xd65f_03c0, // 4: ret
        ];
        assert_eq!(spin_loops(&two), vec![found(at(0), &[at(3)])]);
        let three = [
            0xb900_003f, // 0: wait: str wzr, [x1]
            0xb900_0422, // 1: str w2, [x1, #4]
            0xf900_0083, // 2: str x3, [x4]
            0xb940_0008, // 3: ldr w8, [x0]
            0x34ff_ff88, // 4: cbz w8, wait
            0xd65f_03c0, // 5: ret
        ];
        assert_eq!(spin_loops(&three), vec![]);
    }

    #[test]
    fn rejects_an_indirect_branch_within_reach() {
        let words = [
            0xb940_0008, // 0: wait: ldr w8, [x0]
            0x34ff_ffe8, // 1: cbz w8, wait
            0xd61f_0200, // 2: br x16
        ];
        assert_eq!(spin_loops(&words), vec![]);
    }

    #[test]
    fn honours_the_window() {
        let mut words = vec![
            0xb940_0008, // 0: wait: ldr w8, [x0]
            0x3500_0908, // 1: cbnz w8, done
        ];
        words.extend([0xd503_201f; 70]); // 2..72: nop
        words.push(0x17ff_ffb8); // 72: b wait
        words.push(0xd65f_03c0); // 73: done: ret
        // ldr, cbnz, 70 nops and the branch: 73 instructions per iteration.
        assert_eq!(spin_loops(&words), vec![]);
        assert_eq!(spin_loops_within(&words, 72), vec![]);
        assert_eq!(spin_loops_within(&words, 73), vec![found(at(0), &[at(72)])]);
    }
}
