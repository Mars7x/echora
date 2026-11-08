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
//! Some memory writes are also allowed, because they too leave the state
//! unchanged across an iteration:
//!
//! * an atomic add of a constant whose sum over every path back to T is zero
//!   on each address, such as a reader count taken and dropped again: either
//!   exclusive (LDXR, ADD/SUB #imm, STXR) or LSE (LDADD/STADD of a register
//!   holding a known constant);
//! * a lock taken and released on the same address on every path back to T.
//!   It is taken with LDXR, a zero test and STXR; with CAS whose expected
//!   value is a known zero; or with SWP of a nonzero value whose old value
//!   is then tested for zero. It is released with a store of zero (STR/STLR
//!   of WZR or of a register known to be zero) or a SWP of zero;
//! * at most [`INVARIANT_STORE_LIMIT`] other stores whose value and address
//!   registers the loop never writes.
//!
//! STXR, CAS and SWP acquires count only on the edge where their result
//! reports success: CBZ/CBNZ on the status (STXR) or old value (CAS, SWP)
//! register, or CMP of it with #0 and B.EQ/B.NE. Constants are those moved
//! into registers on the path (MOVZ, MOVN, MOV of WZR or a bitmask
//! immediate), or before the loop in the straight-line code that falls
//! into T when nothing else branches to T. Addresses are compared by base
//! register, which the loop must not write. The other register of a
//! CASP and the remaining LSE operations are not recognised.
//!
//! Paths longer than the window are treated as exits, and so are indirect
//! branches (such as a tail call through X16 after the loop's exit): the
//! detector does not see where they go. Branches to undiscovered code within
//! the window reject the loop.
//!
//! Loops containing YIELD or WFE are not reported: generated code already
//! counts every such instruction as a poll (`Core::spin_hint`).
//!
//! A wrong answer costs only speed, never correctness: generated code can
//! hand the core on at any block boundary.

use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};

use serde::Serialize;

use crate::aarch64::{Address, AtomicOp, Instruction, LogicalOp, MemoryOp, WideOp, Writeback};
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

/// Instructions walked back from the head to find constants set before it.
const ENTRY_CONSTANT_REACH: usize = 64;

/// A loop that cannot exit by itself.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct SpinLoop {
    /// The loop head: the target of the back-edges.
    pub head: u64,
    /// Addresses of the direct branches in the loop that jump to `head`,
    /// in ascending order. A handoff belongs on these branches' taken edge.
    pub back_edges: Vec<u64>,
}

/// Detector settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpinOptions {
    /// The longest path from the head back to the head, in instructions,
    /// that belongs to the loop.
    pub window: usize,
    /// Leave out back-edges that are only taken after a failed STXR, and
    /// loops that only have those. Such a loop retries a store that lost
    /// the exclusive monitor; it does not wait for another thread.
    pub exclude_store_retries: bool,
}

impl Default for SpinOptions {
    fn default() -> Self {
        Self {
            window: DEFAULT_SPIN_WINDOW,
            exclude_store_retries: true,
        }
    }
}

/// [`find_spin_loops_with`] with the default [`SpinOptions`].
pub fn find_spin_loops(blocks: &[BasicBlock]) -> Vec<SpinLoop> {
    find_spin_loops_with(blocks, SpinOptions::default())
}

/// [`find_spin_loops_with`] with the default options and another window.
pub fn find_spin_loops_within(blocks: &[BasicBlock], window: usize) -> Vec<SpinLoop> {
    find_spin_loops_with(
        blocks,
        SpinOptions {
            window,
            ..SpinOptions::default()
        },
    )
}

/// Lists the spin-wait loops among `blocks`, ordered by head address.
///
/// Candidate heads are the targets of backward direct branches (target at or
/// before the branch).
pub fn find_spin_loops_with(blocks: &[BasicBlock], options: SpinOptions) -> Vec<SpinLoop> {
    let code: BTreeMap<u64, &Instruction> = blocks
        .iter()
        .flat_map(|block| block.instructions.iter())
        .map(|record| (record.pc, &record.instruction))
        .collect();
    let mut targets: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    for (&pc, instruction) in &code {
        if let Some(target) = direct_target(instruction) {
            targets.entry(target).or_default().push(pc);
        }
    }
    let heads: BTreeSet<u64> = targets
        .iter()
        .filter(|(target, sources)| {
            code.contains_key(target) && sources.iter().any(|source| source >= target)
        })
        .map(|(&target, _)| target)
        .collect();
    let program = Program {
        code: &code,
        targets: &targets,
    };
    heads
        .into_iter()
        .filter_map(|head| analyze(&program, head, options))
        .collect()
}

struct Program<'a> {
    code: &'a BTreeMap<u64, &'a Instruction>,
    /// Direct branch target -> the branches to it.
    targets: &'a BTreeMap<u64, Vec<u64>>,
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
/// SVCs returning to the next instruction. An empty list means the code is
/// left (a return, an indirect branch or a trap).
fn flow(pc: u64, instruction: &Instruction) -> Vec<u64> {
    let next = pc.wrapping_add(4);
    match *instruction {
        Instruction::Branch {
            target,
            link: false,
        } => vec![target],
        Instruction::BranchCond { target, .. }
        | Instruction::CompareBranch { target, .. }
        | Instruction::TestBranch { target, .. } => vec![target, next],
        Instruction::BranchReg { link: false, .. }
        | Instruction::PAuthDisabledReturn { .. }
        | Instruction::Undefined
        | Instruction::Unsupported => vec![],
        _ => vec![next],
    }
}

struct Graph<'a> {
    code: &'a BTreeMap<u64, &'a Instruction>,
    /// Instructions reachable from the head, with their shortest distance.
    distance: BTreeMap<u64, usize>,
    /// Successors of each explored instruction (only those explored).
    successors: BTreeMap<u64, Vec<u64>>,
    /// Instructions with a successor that was not explored or that leave.
    leaves: BTreeSet<u64>,
}

/// What the path simulation found.
struct Simulation {
    /// Plain zero stores that released a lock.
    releases: BTreeSet<u64>,
    /// Back-edges taken on some path without a failed STXR.
    waiting_back_edges: BTreeSet<u64>,
}

fn analyze(program: &Program, head: u64, options: SpinOptions) -> Option<SpinLoop> {
    let window = options.window;
    if window == 0 {
        return None;
    }
    let code = program.code;
    let graph = explore(code, head, window)?;
    let members = loop_members(&graph, head, window);
    if !members.contains(&head) {
        return None;
    }
    let mut back_edges: Vec<u64> = members
        .iter()
        .copied()
        .filter(|pc| direct_target(code[pc]) == Some(head))
        .collect();
    if back_edges.is_empty() {
        return None;
    }

    let mut written = Regs::NONE;
    let mut all_effects = BTreeMap::new();
    for &pc in &members {
        let e = effects(code[&pc]);
        if e.calls || e.svc || e.system || e.event_wait || e.traps {
            return None;
        }
        written = written.union(e.writes);
        all_effects.insert(pc, e);
    }
    let mut plain_stores = BTreeSet::new();
    let mut must_release = BTreeSet::new();
    let mut simulate = false;
    for (&pc, e) in &all_effects {
        if e.atomic || (e.stores && e.exclusive) {
            simulate = true;
        } else if !e.stores {
            continue;
        } else if e.writes.is_empty() && !e.reads.intersects(written) {
            plain_stores.insert(pc);
        } else if release_base(code[&pc]).is_some_and(|rn| !written.has_gpr_or_sp(rn)) {
            // A store of a changing register: only a zero store releasing a
            // lock on every path is allowed.
            must_release.insert(pc);
            simulate = true;
        } else {
            return None;
        }
    }

    if written.intersects(live_at(&graph, head)) {
        return None;
    }

    let simulation = if simulate {
        let constants = entry_constants(program, head, &members, written);
        simulate_paths(
            &graph,
            &members,
            head,
            window,
            written,
            constants,
            &must_release,
        )?
    } else {
        Simulation {
            releases: BTreeSet::new(),
            waiting_back_edges: back_edges.iter().copied().collect(),
        }
    };
    if plain_stores.difference(&simulation.releases).count() > INVARIANT_STORE_LIMIT {
        return None;
    }
    if options.exclude_store_retries {
        back_edges.retain(|pc| simulation.waiting_back_edges.contains(pc));
        if back_edges.is_empty() {
            return None;
        }
    }
    Some(SpinLoop { head, back_edges })
}

/// The base register of a store that can release a lock: STLR, or STR with
/// a zero immediate offset and no writeback.
fn release_base(instruction: &Instruction) -> Option<u8> {
    match *instruction {
        Instruction::StoreRelease { rn, .. }
        | Instruction::LoadStore {
            rn,
            op: MemoryOp::Store,
            address:
                Address::Immediate {
                    offset: 0,
                    writeback: Writeback::None,
                },
            ..
        } => Some(rn),
        _ => None,
    }
}

/// Breadth-first exploration from `head` up to `window - 1` instructions
/// deep. Fails when a branch to undiscovered code is reachable.
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
        let targets = flow(pc, code[&pc]);
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
/// Everything is live where exploration stops, at returns, indirect
/// branches and traps, and across calls and SVCs.
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

/// The constant `instruction` moves into a general register, if it does.
fn moved_constant(instruction: &Instruction) -> Option<(u8, u64)> {
    let mask = |bits: u8| {
        if bits == 64 {
            u64::MAX
        } else {
            u32::MAX as u64
        }
    };
    match *instruction {
        Instruction::MoveWide {
            rd,
            bits,
            op,
            immediate,
            shift,
        } if rd != 31 => {
            let value = u64::from(immediate) << shift;
            match op {
                WideOp::Zero => Some((rd, value & mask(bits))),
                WideOp::Not => Some((rd, !value & mask(bits))),
                WideOp::Keep => None,
            }
        }
        // MOV Rd, ZR.
        Instruction::LogicalShifted {
            rd,
            rn: 31,
            rm: 31,
            op: LogicalOp::Orr,
            invert: false,
            ..
        } if rd != 31 => Some((rd, 0)),
        // MOV Rd, #bitmask (ORR Rd, ZR, #imm). Rd = 31 is SP.
        Instruction::LogicalImmediate {
            rd,
            rn: 31,
            op: LogicalOp::Orr,
            mask,
            ..
        } if rd != 31 => Some((rd, mask)),
        _ => None,
    }
}

/// Constants set before the loop by the straight-line code that falls into
/// the head, when nothing outside the loop branches to the head. Only
/// registers the loop never writes are kept.
fn entry_constants(
    program: &Program,
    head: u64,
    members: &BTreeSet<u64>,
    written: Regs,
) -> BTreeMap<u8, u64> {
    let mut constants = BTreeMap::new();
    let outside_branch = program
        .targets
        .get(&head)
        .is_some_and(|sources| sources.iter().any(|source| !members.contains(source)));
    if outside_branch {
        return constants;
    }
    let mut decided = Regs::NONE;
    let mut pc = head;
    for _ in 0..ENTRY_CONSTANT_REACH {
        let Some(previous) = pc.checked_sub(4) else {
            break;
        };
        let Some(&instruction) = program.code.get(&previous) else {
            break;
        };
        if members.contains(&previous) || flow(previous, instruction) != vec![pc] {
            break;
        }
        let e = effects(instruction);
        if e.calls || e.svc || e.branch {
            break;
        }
        if let Some((rd, value)) = moved_constant(instruction)
            && !decided.has_gpr(rd)
        {
            constants.insert(rd, value);
        }
        decided = decided.union(e.writes);
        pc = previous;
        // Anything before a branch target may hold other values there.
        if program.targets.contains_key(&pc) {
            break;
        }
    }
    constants.retain(|&reg, _| !written.has_gpr(reg));
    constants
}

/// An address used by atomic accesses: base register and access size.
type Location = (u8, u8);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Monitor {
    location: Location,
    /// The loaded value has been tested equal to zero on this path.
    zero: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Update {
    Add(u64),
    Acquire,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Pending {
    /// Zero on success: the STXR status, or the CAS/SWP old value.
    status: u8,
    location: Location,
    update: Update,
    /// A STXR, which fails only when it loses the monitor.
    store_exclusive: bool,
    /// The status has been compared with zero into NZCV.
    compared: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
struct PathState {
    monitor: Option<Monitor>,
    /// Registers holding the exclusively loaded value plus a constant.
    derived: BTreeMap<u8, u64>,
    /// Registers holding known constants.
    constants: BTreeMap<u8, u64>,
    /// An atomic acquire or exclusive store whose result is untested.
    pending: Option<Pending>,
    /// Net atomic adds per location, without zero entries.
    deltas: BTreeMap<Location, u64>,
    held: BTreeSet<Location>,
    /// A STXR failed on this path.
    store_failed: bool,
}

fn size_mask(bytes: u8) -> u64 {
    if bytes >= 8 {
        u64::MAX
    } else {
        (1 << (8 * u32::from(bytes))) - 1
    }
}

impl PathState {
    fn at_rest(&self) -> bool {
        self.pending.is_none() && self.deltas.is_empty() && self.held.is_empty()
    }

    fn add(&mut self, location: Location, amount: u64) {
        let total = self
            .deltas
            .get(&location)
            .copied()
            .unwrap_or(0)
            .wrapping_add(amount)
            & size_mask(location.1);
        if total == 0 {
            self.deltas.remove(&location);
        } else {
            self.deltas.insert(location, total);
        }
    }

    fn commit(&mut self, pending: Pending) -> Option<()> {
        match pending.update {
            Update::Add(amount) => self.add(pending.location, amount),
            Update::Acquire => {
                if !self.held.insert(pending.location) {
                    return None;
                }
            }
        }
        Some(())
    }

    /// The edges of a result test: the success edge commits.
    fn split(
        mut self,
        pending: Pending,
        success: u64,
        failure: u64,
    ) -> Option<Vec<(u64, PathState)>> {
        self.pending = None;
        let mut failed = self.clone();
        failed.store_failed |= pending.store_exclusive;
        self.commit(pending)?;
        Some(vec![(success, self), (failure, failed)])
    }

    fn is_zero(&self, register: u8) -> bool {
        register == 31 || self.constants.get(&register) == Some(&0)
    }

    /// Whether `register` holds a value the loop leaves alone.
    fn is_fixed(&self, register: u8, written: Regs) -> bool {
        !written.has_gpr(register) || self.constants.contains_key(&register)
    }

    fn release(&mut self, location: Location) -> bool {
        self.held.remove(&location)
    }
}

/// Follows every path from `head` back to `head` through the loop, checking
/// that atomic adds cancel and locks are released.
fn simulate_paths(
    graph: &Graph,
    members: &BTreeSet<u64>,
    head: u64,
    window: usize,
    written: Regs,
    constants: BTreeMap<u8, u64>,
    must_release: &BTreeSet<u64>,
) -> Option<Simulation> {
    let mut simulation = Simulation {
        releases: BTreeSet::new(),
        waiting_back_edges: BTreeSet::new(),
    };
    let mut seen = HashSet::new();
    let start = PathState {
        constants,
        ..PathState::default()
    };
    let mut stack = vec![(head, 0_usize, start)];
    while let Some((pc, steps, state)) = stack.pop() {
        if !seen.insert((pc, steps, state.clone())) {
            continue;
        }
        if seen.len() > PATH_STATE_LIMIT {
            return None;
        }
        let instruction = graph.code[&pc];
        let successors = step(
            pc,
            instruction,
            state,
            written,
            must_release,
            &mut simulation,
        )?;
        for (target, next) in successors {
            if target == head {
                if !next.at_rest() {
                    return None;
                }
                if !next.store_failed && direct_target(instruction) == Some(head) {
                    simulation.waiting_back_edges.insert(pc);
                }
            } else if members.contains(&target) && steps + 2 <= window {
                stack.push((target, steps + 1, next));
            }
        }
    }
    Some(simulation)
}

/// One instruction of a path: its successors with the path state on each.
fn step(
    pc: u64,
    instruction: &Instruction,
    mut state: PathState,
    written: Regs,
    must_release: &BTreeSet<u64>,
    simulation: &mut Simulation,
) -> Option<Vec<(u64, PathState)>> {
    let next = pc.wrapping_add(4);
    let e = effects(instruction);
    if let Some(mut pending) = state.pending {
        match *instruction {
            Instruction::CompareBranch {
                rt,
                nonzero,
                target,
                ..
            } if !pending.compared && rt == pending.status => {
                let (success, failure) = if nonzero {
                    (next, target)
                } else {
                    (target, next)
                };
                return state.split(pending, success, failure);
            }
            // CMP status, #0.
            Instruction::AddSubImmediate {
                rd: 31,
                rn,
                immediate: 0,
                subtract: true,
                flags: true,
                ..
            } if !pending.compared && rn == pending.status => {
                pending.compared = true;
                state.pending = Some(pending);
                return Some(vec![(next, state)]);
            }
            // B.EQ (0) or B.NE (1) after the compare.
            Instruction::BranchCond { target, condition } if pending.compared => {
                let (success, failure) = match condition {
                    0 => (target, next),
                    1 => (next, target),
                    _ => return None,
                };
                return state.split(pending, success, failure);
            }
            _ => {}
        }
        let status = if pending.compared {
            Regs {
                nzcv: true,
                ..Regs::NONE
            }
        } else {
            Regs::gpr(pending.status)
        };
        if e.branch || e.reads.intersects(status) || e.writes.intersects(status) {
            return None;
        }
    }

    let constant = |state: &PathState, register: u8| {
        if register == 31 {
            Some(0)
        } else {
            state.constants.get(&register).copied()
        }
    };
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
                let value = if subtract {
                    base.wrapping_sub(immediate)
                } else {
                    base.wrapping_add(immediate)
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
                _ if monitor.zero && state.is_fixed(rt, written) => Update::Acquire,
                _ => return None,
            };
            state.pending = Some(Pending {
                status: rs,
                location: monitor.location,
                update,
                store_exclusive: true,
                compared: false,
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
        // CAS with a zero expected value: the lock is taken when the old
        // value that comes back in Rs is zero.
        Instruction::CompareAndSwap {
            rs, rt, rn, bytes, ..
        } => {
            if written.has_gpr_or_sp(rn)
                || rs == 31
                || rs == rt
                || !state.is_zero(rs)
                || !state.is_fixed(rt, written)
            {
                return None;
            }
            state.pending = Some(Pending {
                status: rs,
                location: (rn, bytes),
                update: Update::Acquire,
                store_exclusive: false,
                compared: false,
            });
        }
        // SWP of zero releases; SWP of a fixed nonzero value acquires when
        // the old value in Rt is zero.
        Instruction::AtomicMemory {
            op: AtomicOp::Swap,
            rs,
            rt,
            rn,
            bytes,
            ..
        } => {
            if written.has_gpr_or_sp(rn) {
                return None;
            }
            let location = (rn, bytes);
            if state.is_zero(rs) {
                if !state.release(location) {
                    return None;
                }
            } else {
                if rt == 31 || rt == rs || !state.is_fixed(rs, written) {
                    return None;
                }
                state.pending = Some(Pending {
                    status: rt,
                    location,
                    update: Update::Acquire,
                    store_exclusive: false,
                    compared: false,
                });
            }
        }
        Instruction::AtomicAdd { rs, rn, bytes, .. } => {
            if written.has_gpr_or_sp(rn) {
                return None;
            }
            state.add((rn, bytes), constant(&state, rs)?);
        }
        Instruction::CompareAndSwapPair { .. } | Instruction::AtomicMemory { .. } => return None,
        Instruction::StoreRelease { rt, rn, bytes }
        | Instruction::LoadStore {
            rt,
            rn,
            bytes,
            op: MemoryOp::Store,
            address:
                Address::Immediate {
                    offset: 0,
                    writeback: Writeback::None,
                },
        } if state.is_zero(rt) && state.release((rn, bytes)) => {
            simulation.releases.insert(pc);
        }
        _ => {
            if must_release.contains(&pc) {
                return None;
            }
            state.derived.retain(|&reg, _| !e.writes.has_gpr(reg));
        }
    }
    state.constants.retain(|&reg, _| !e.writes.has_gpr(reg));
    if let Some((rd, value)) = moved_constant(instruction) {
        state.constants.insert(rd, value);
    }
    Some(
        flow(pc, instruction)
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

    fn spin_loops_with(words: &[u32], options: SpinOptions) -> Vec<SpinLoop> {
        let region = CodeRegion {
            name: "test".into(),
            base: BASE,
            bytes: words.iter().flat_map(|word| word.to_le_bytes()).collect(),
        };
        let translation = discover(&[region], &[BASE], 4096).expect("discover test code");
        find_spin_loops_with(&translation.blocks, options)
    }

    fn spin_loops_within(words: &[u32], window: usize) -> Vec<SpinLoop> {
        spin_loops_with(
            words,
            SpinOptions {
                window,
                ..SpinOptions::default()
            },
        )
    }

    fn spin_loops(words: &[u32]) -> Vec<SpinLoop> {
        find_spin_loops_in(words)
    }

    /// Through the public default entry point.
    fn find_spin_loops_in(words: &[u32]) -> Vec<SpinLoop> {
        let region = CodeRegion {
            name: "test".into(),
            base: BASE,
            bytes: words.iter().flat_map(|word| word.to_le_bytes()).collect(),
        };
        let translation = discover(&[region], &[BASE], 4096).expect("discover test code");
        find_spin_loops(&translation.blocks)
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
        assert_eq!(spin_loops(&words), vec![found(at(7), &[at(9)])]);
        // The ticket STXR loops back only when it fails. That retry changes
        // nothing either, so without the exclusion it is a spin loop too.
        let all = SpinOptions {
            exclude_store_retries: false,
            ..SpinOptions::default()
        };
        assert_eq!(
            spin_loops_with(&words, all),
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
        assert!(loops.contains(&found(at(0), &[at(10)])), "{loops:?}");
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
        // The STXR-failure back-edge (3) is a retry, not a wait.
        assert_eq!(spin_loops(&words), vec![found(at(0), &[at(1), at(6)])]);
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
    fn allows_a_tail_call_after_the_exit() {
        let words = [
            0xb940_0008, // 0: wait: ldr w8, [x0]
            0x34ff_ffe8, // 1: cbz w8, wait
            0xd61f_0200, // 2: br x16
        ];
        assert_eq!(spin_loops(&words), vec![found(at(0), &[at(1)])]);
    }

    #[test]
    fn finds_a_cas_lock_released_by_a_zero_store() {
        let words = [
            0x2a1f_03e8, // 0: retry: mov w8, wzr
            0x88e8_7c09, // 1: casa w8, w9, [x0]
            0x35ff_ffc8, // 2: cbnz w8, retry
            0xb940_002a, // 3: ldr w10, [x1]
            0x889f_fc1f, // 4: stlr wzr, [x0]
            0x34ff_ff6a, // 5: cbz w10, retry
            0xd65f_03c0, // 6: ret
        ];
        // A failed CAS means the lock is held: that back-edge waits.
        assert_eq!(spin_loops(&words), vec![found(at(0), &[at(2), at(5)])]);
    }

    #[test]
    fn finds_a_cas_lock_tested_with_cmp_and_released_from_a_zero_register() {
        let words = [
            0x5280_0029, // 0: mov w9, #1
            0x2a1f_03e8, // 1: retry: mov w8, wzr
            0x88e8_7c09, // 2: casa w8, w9, [x0]
            0x7100_011f, // 3: cmp w8, #0
            0x54ff_ffa1, // 4: b.ne retry
            0xb940_002a, // 5: ldr w10, [x1]
            0x2a1f_03eb, // 6: mov w11, wzr
            0x889f_fc0b, // 7: stlr w11, [x0]
            0x34ff_ff2a, // 8: cbz w10, retry
            0xd65f_03c0, // 9: ret
        ];
        assert_eq!(spin_loops(&words), vec![found(at(1), &[at(4), at(8)])]);
    }

    #[test]
    fn rejects_a_cas_lock_still_held_at_the_head() {
        let words = [
            0x2a1f_03e8, // 0: retry: mov w8, wzr
            0x88e8_7c09, // 1: casa w8, w9, [x0]
            0x35ff_ffc8, // 2: cbnz w8, retry
            0xb940_002a, // 3: ldr w10, [x1]
            0x34ff_ff8a, // 4: cbz w10, retry
            0x889f_fc1f, // 5: stlr wzr, [x0]
            0xd65f_03c0, // 6: ret
        ];
        assert_eq!(spin_loops(&words), vec![]);
    }

    #[test]
    fn finds_a_swap_lock_released_by_a_swap_of_zero() {
        let words = [
            0xb8a9_8008, // 0: retry: swpa w9, w8, [x0]
            0x35ff_ffe8, // 1: cbnz w8, retry
            0xb940_002a, // 2: ldr w10, [x1]
            0xb87f_800b, // 3: swpl wzr, w11, [x0]
            0x34ff_ff8a, // 4: cbz w10, retry
            0xd65f_03c0, // 5: ret
        ];
        assert_eq!(spin_loops(&words), vec![found(at(0), &[at(1), at(4)])]);
    }

    #[test]
    fn finds_an_lse_add_undone_on_every_path() {
        let words = [
            0x5280_0029, // 0: mov w9, #1
            0x1280_000b, // 1: mov w11, #-1
            0xb8a9_0008, // 2: retry: ldadda w9, w8, [x0]
            0x88df_fc2a, // 3: ldar w10, [x1]
            0x3400_006a, // 4: cbz w10, done
            0xb86b_0008, // 5: ldaddl w11, w8, [x0]
            0x17ff_fffc, // 6: b retry
            0xd65f_03c0, // 7: done: ret
        ];
        assert_eq!(spin_loops(&words), vec![found(at(2), &[at(6)])]);
    }

    #[test]
    fn rejects_an_lse_add_left_in_place() {
        let words = [
            0x5280_0029, // 0: mov w9, #1
            0xb8a9_0008, // 1: retry: ldadda w9, w8, [x0]
            0x88df_fc2a, // 2: ldar w10, [x1]
            0x34ff_ffca, // 3: cbz w10, retry
            0xd65f_03c0, // 4: ret
        ];
        assert_eq!(spin_loops(&words), vec![]);
    }

    #[test]
    fn rejects_an_lse_add_of_an_unknown_amount() {
        let words = [
            0xb8a9_0008, // 0: retry: ldadda w9, w8, [x0]
            0x88df_fc2a, // 1: ldar w10, [x1]
            0x3400_008a, // 2: cbz w10, done
            0x1280_000b, // 3: mov w11, #-1
            0xb86b_0008, // 4: ldaddl w11, w8, [x0]
            0x17ff_fffb, // 5: b retry
            0xd65f_03c0, // 6: done: ret
        ];
        assert_eq!(spin_loops(&words), vec![]);
    }

    #[test]
    fn finds_an_exclusive_lock_acquire_at_its_head() {
        // lock: ldaxr w8, [x0]; cbnz w8, lock; stxr w9, w1, [x0];
        // cbnz w9, lock; ret. The poll edge makes it a spin; the hint goes
        // at the head, whichever edge returns there.
        let words = [
            0x885f_fc08,
            0x35ff_ffe8,
            0x8809_7c01,
            0x35ff_ffa9,
            0xd65f_03c0,
        ];
        let loops = spin_loops(&words);
        assert_eq!(loops.len(), 1, "{loops:?}");
        assert_eq!(loops[0].head, at(0));
    }

    #[test]
    fn rejects_an_exclusive_increment_retry() {
        // retry: ldaxr w8, [x0]; add w8, w8, #1; stlxr w9, w8, [x0];
        // cbnz w9, retry: the thread can finish it alone (the next STLXR
        // may succeed), so it is not a wait.
        let words = [
            0x885f_fc08,
            0x1100_0508,
            0x8809_fc08,
            0x35ff_ffa9,
            0xd65f_03c0,
        ];
        assert_eq!(spin_loops(&words), vec![]);
    }

    #[test]
    fn leaves_wfe_and_yield_waits_to_their_own_hints() {
        // Generated code already counts every WFE and YIELD as a poll
        // (codegen's SpinHint), so loops containing one are not reported
        // again: sevl; wait: wfe; ldaxr w8, [x0]; cbnz w8, wait; ...
        let wfe_first = [
            0xd503_20bf,
            0xd503_205f,
            0x885f_fc08,
            0x35ff_ffc8,
            0xd65f_03c0,
        ];
        // ... loop: ldaxr w8, [x0]; cbz w8, done; wfe; b loop; done: ret
        let wfe_after = [
            0x885f_fc08,
            0x3400_0068,
            0xd503_205f,
            0x17ff_fffd,
            0xd65f_03c0,
        ];
        // ... wait: yield; ldr w8, [x0]; cbz w8, wait
        let yield_poll = [0xd503_203f, 0xb940_0008, 0x34ff_ffc8, 0xd65f_03c0];
        for words in [&wfe_first[..], &wfe_after, &yield_poll] {
            assert_eq!(spin_loops(words), vec![], "{words:x?}");
        }
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
