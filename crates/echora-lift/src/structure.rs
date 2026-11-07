//! Call targets and structured control flow of discovered code.
//!
//! Groundwork for direct native calls and for generating plain loops and
//! `if`/`else` instead of a window's `match next_pc` dispatch loop:
//!
//! * [`call_sites`] classifies every BL and BLR: a direct target that is
//!   translated at lift time, untranslated code, or a register. With
//!   [`CallOptions::trust_aapcs64`] it marks X19-X29 and SP as preserved, so
//!   the caller only reloads the rest after a direct call.
//! * [`entry_windows`] maps every address control can enter a window at to
//!   that window.
//! * [`analyze_windows`] computes, for each window, its entries, dominators,
//!   natural loop nesting, irreducible edges and, when the window is
//!   reducible, a [`Structured`] tree of blocks, loops and ifs built with
//!   Ramsey's algorithm from "Beyond Relooper" (the WebAssembly stackifier).
//!
//! Windows are the dispatch windows of code generation: the blocks whose
//! start shares `start >> window_bits`.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use crate::aarch64::Instruction;
use crate::codegen::{BasicBlock, Translation};
use crate::defuse::Regs;

/// The window size of split code generation (1 KiB windows).
pub const DEFAULT_WINDOW_BITS: u32 = 10;

/// How much a call site may assume about its callee.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CallOptions {
    /// Trust the callee to preserve X19-X29 and SP as AAPCS64 requires.
    /// Off by default: guest code is not obliged to follow the convention.
    pub trust_aapcs64: bool,
}

/// Where a call goes, as known at lift time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum CallTarget {
    /// BL to translated code: a native call to `window`'s function is
    /// possible, entering at `entry`.
    Direct { entry: u64, window: u64 },
    /// BL to an address that was not translated (a host function or code
    /// outside the executable regions).
    Untranslated { target: u64 },
    /// BLR: the target is in `register`.
    Indirect { register: u8 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CallSite {
    /// The BL or BLR.
    pub pc: u64,
    /// Where the callee returns to.
    pub return_to: u64,
    pub target: CallTarget,
    /// Registers that hold the same value after the call returns.
    pub preserved: Regs,
    /// Registers the caller must reload after the call: everything else.
    pub clobbered: Regs,
}

/// Registers AAPCS64 makes callee-saved and the caller may trust: X19-X29
/// and SP. (Only the low halves of V8-V15 are callee-saved, so no vector
/// register is trusted.)
pub fn aapcs64_preserved() -> Regs {
    Regs {
        gpr: ((1 << 30) - 1) & !((1 << 19) - 1),
        sp: true,
        nzcv: false,
        vector: 0,
    }
}

/// Every BL and BLR in `translation`, in address order.
pub fn call_sites(
    translation: &Translation,
    window_bits: u32,
    options: CallOptions,
) -> Vec<CallSite> {
    let preserved = if options.trust_aapcs64 {
        aapcs64_preserved()
    } else {
        Regs::NONE
    };
    translation
        .instructions
        .values()
        .filter_map(|record| {
            let target = match record.instruction {
                Instruction::Branch { target, link: true } => {
                    if translation.instructions.contains_key(&target) {
                        CallTarget::Direct {
                            entry: target,
                            window: target >> window_bits,
                        }
                    } else {
                        CallTarget::Untranslated { target }
                    }
                }
                Instruction::BranchReg { rn, link: true } => CallTarget::Indirect { register: rn },
                _ => return None,
            };
            Some(CallSite {
                pc: record.pc,
                return_to: record.pc.wrapping_add(4),
                target,
                preserved,
                clobbered: Regs::ALL.difference(preserved),
            })
        })
        .collect()
}

/// Why control can enter a window at a block. Several can hold at once.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct EntryKinds {
    /// A translation entry (dispatch root).
    pub translation_entry: bool,
    /// The target of a BL.
    pub call_target: bool,
    /// The target of a direct branch from another window.
    pub branch_from_outside: bool,
    /// Reached by falling through from the end of another window.
    pub fallthrough_from_outside: bool,
    /// No direct edge reaches the block from its own window: it is reached
    /// some other way, such as a jump table or a tail call through a
    /// register.
    pub unreferenced: bool,
    /// The instruction after a BL or BLR. A callee that leaves through the
    /// dispatcher (a trap, a host boundary, an exceeded call depth) resumes
    /// here. This is a resume point, not a control entry: it is reached by
    /// the call's own fall-through edge inside the window.
    pub return_address: bool,
}

impl EntryKinds {
    /// Whether control (rather than only resumption after a call) can
    /// enter here.
    pub fn is_control(&self) -> bool {
        self.translation_entry
            || self.call_target
            || self.branch_from_outside
            || self.fallthrough_from_outside
            || self.unreferenced
    }

    fn any(&self) -> bool {
        self.is_control() || self.return_address
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct WindowEntry {
    pub pc: u64,
    pub kinds: EntryKinds,
}

/// Every address at which a window can be entered (control entries and
/// return addresses), mapped to its window (`pc >> window_bits`).
///
/// Any block start can also be resumed after `Trap::HostBoundary`; those
/// resumptions are not listed.
pub fn entry_windows(translation: &Translation, window_bits: u32) -> BTreeMap<u64, u64> {
    analyze_windows(translation, window_bits)
        .into_iter()
        .flat_map(|window| {
            window
                .entries
                .into_iter()
                .map(move |entry| (entry.pc, window.window))
        })
        .collect()
}

/// A natural loop: a header and the blocks that reach a back-edge into it
/// without passing through it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct NaturalLoop {
    pub header: u64,
    /// Blocks ending in a back-edge to the header.
    pub latches: Vec<u64>,
    /// Block starts of the loop body, header included, ascending.
    pub blocks: Vec<u64>,
    /// Index in [`WindowAnalysis::loops`] of the innermost enclosing loop.
    pub parent: Option<usize>,
    /// 1 for an outermost loop.
    pub depth: usize,
}

/// A retreating edge whose target does not dominate its source: the cycle
/// it closes can be entered at more than one block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct IrreducibleEdge {
    pub from: u64,
    pub to: u64,
}

/// Structured code for a reducible window.
///
/// Labels are counted from the innermost: `Br(0)` targets the closest
/// enclosing `Block` or `Loop`. Leaving a `Block` with `Br` continues after
/// it; `Br` to a `Loop` continues at its start; reaching the end of a
/// `Loop` body falls out of it (in Rust: `'l: loop { body; break }`, with
/// `continue 'l` for `Br`). `If` does not take a label.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum Structured {
    /// The instructions of the block starting here. Its final control
    /// transfer is expressed by what follows (an `If`, `Br` or `Leave`),
    /// except for calls, which return to the next instruction.
    Code(u64),
    Seq(Vec<Structured>),
    Block(Box<Structured>),
    Loop(Box<Structured>),
    /// Decided by the conditional branch ending `block`.
    If {
        block: u64,
        taken: Box<Structured>,
        fallthrough: Box<Structured>,
    },
    /// Selects by the entry PC; only at the top of a multi-entry window.
    Dispatch(Vec<(u64, Structured)>),
    Br(usize),
    /// Leave the window: to `Some(pc)` for a direct edge to code outside
    /// it, or `None` where the block computes the next PC itself (a return,
    /// an indirect branch or a trap).
    Leave(Option<u64>),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct WindowAnalysis {
    /// `start >> window_bits` of its blocks.
    pub window: u64,
    /// Block starts, ascending.
    pub blocks: Vec<u64>,
    /// Blocks where control or resumption can enter the window.
    pub entries: Vec<WindowEntry>,
    /// Immediate dominator of each block; `None` for blocks dominated only
    /// by the window entry as a whole (control entries, and blocks reached
    /// from more than one of them).
    pub idom: BTreeMap<u64, Option<u64>>,
    /// Natural loops, outer loops before the loops they contain.
    pub loops: Vec<NaturalLoop>,
    /// Empty for a reducible window.
    pub irreducible: Vec<IrreducibleEdge>,
    /// The structured form, for a reducible window.
    pub structure: Option<Structured>,
}

impl WindowAnalysis {
    /// Entries that control (not only resumption) can take.
    pub fn control_entries(&self) -> impl Iterator<Item = u64> + '_ {
        self.entries
            .iter()
            .filter(|entry| entry.kinds.is_control())
            .map(|entry| entry.pc)
    }

    pub fn is_multi_entry(&self) -> bool {
        self.control_entries().nth(1).is_some()
    }

    pub fn is_reducible(&self) -> bool {
        self.irreducible.is_empty()
    }
}

/// Successor PCs of a block's last instruction, taken edge first, and
/// whether the first is a jump rather than a fall-through. Calls continue
/// at the next instruction; an empty list leaves the window (return,
/// indirect branch, trap).
fn block_successors(block: &BasicBlock) -> (Vec<u64>, bool) {
    let last = block
        .instructions
        .last()
        .expect("blocks start with one instruction");
    let next = last.pc.wrapping_add(4);
    let successors = match last.instruction {
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
    };
    let jumps = matches!(
        last.instruction,
        Instruction::Branch { link: false, .. }
            | Instruction::BranchCond { .. }
            | Instruction::CompareBranch { .. }
            | Instruction::TestBranch { .. }
    );
    (successors, jumps)
}

/// The analysis of every window, in window order.
pub fn analyze_windows(translation: &Translation, window_bits: u32) -> Vec<WindowAnalysis> {
    let starts: BTreeMap<u64, &BasicBlock> = translation
        .blocks
        .iter()
        .map(|block| (block.start, block))
        .collect();
    let window_of = |pc: u64| pc >> window_bits;

    let mut kinds: BTreeMap<u64, EntryKinds> = BTreeMap::new();
    let mut referenced = BTreeSet::new();
    for &entry in &translation.entries {
        if starts.contains_key(&entry) {
            kinds.entry(entry).or_default().translation_entry = true;
        }
    }
    for block in &translation.blocks {
        let last = block
            .instructions
            .last()
            .expect("blocks start with one instruction");
        match last.instruction {
            Instruction::Branch { target, link: true } => {
                if starts.contains_key(&target) {
                    kinds.entry(target).or_default().call_target = true;
                }
                kinds.entry(last.pc + 4).or_default().return_address = true;
            }
            Instruction::BranchReg { link: true, .. } => {
                kinds.entry(last.pc + 4).or_default().return_address = true;
            }
            _ => {}
        }
        let (successors, jumps) = block_successors(block);
        for (index, successor) in successors.into_iter().enumerate() {
            if !starts.contains_key(&successor) {
                continue;
            }
            if window_of(successor) == window_of(block.start) {
                referenced.insert(successor);
            } else if jumps && index == 0 {
                kinds.entry(successor).or_default().branch_from_outside = true;
            } else {
                kinds.entry(successor).or_default().fallthrough_from_outside = true;
            }
        }
    }
    kinds.retain(|pc, _| starts.contains_key(pc));

    let mut windows: BTreeMap<u64, Vec<&BasicBlock>> = BTreeMap::new();
    for block in &translation.blocks {
        windows
            .entry(window_of(block.start))
            .or_default()
            .push(block);
    }
    windows
        .into_iter()
        .map(|(window, blocks)| {
            let mut entries: BTreeMap<u64, EntryKinds> = blocks
                .iter()
                .filter_map(|block| {
                    let mut entry = kinds.get(&block.start).copied().unwrap_or_default();
                    if !entry.is_control() && !referenced.contains(&block.start) {
                        entry.unreferenced = true;
                    }
                    entry.any().then_some((block.start, entry))
                })
                .collect();
            analyze_window(window, &blocks, &mut entries, window_bits)
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Edge {
    Node(usize),
    Leave(Option<u64>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Label {
    LoopHeadedBy(usize),
    BlockFollowedBy(usize),
}

/// One window's graph. Node 0 is a virtual root with an edge to each
/// control entry; node `i` is block `i - 1`.
struct Graph {
    starts: Vec<u64>,
    successors: Vec<Vec<Edge>>,
    /// Reverse postorder number of each reachable node.
    rpo: Vec<Option<usize>>,
    idom: Vec<usize>,
    children: Vec<Vec<usize>>,
    loop_header: Vec<bool>,
    merge: Vec<bool>,
}

fn analyze_window(
    window: u64,
    blocks: &[&BasicBlock],
    entries: &mut BTreeMap<u64, EntryKinds>,
    window_bits: u32,
) -> WindowAnalysis {
    let index: BTreeMap<u64, usize> = blocks
        .iter()
        .enumerate()
        .map(|(i, block)| (block.start, i + 1))
        .collect();
    let mut successors = vec![Vec::new()];
    for block in blocks {
        let (targets, _) = block_successors(block);
        let edges = if targets.is_empty() {
            vec![Edge::Leave(None)]
        } else {
            targets
                .into_iter()
                .map(|target| match index.get(&target) {
                    Some(&node) if target >> window_bits == window => Edge::Node(node),
                    _ => Edge::Leave(Some(target)),
                })
                .collect()
        };
        successors.push(edges);
    }
    let mut graph = Graph {
        starts: std::iter::once(0)
            .chain(blocks.iter().map(|block| block.start))
            .collect(),
        successors,
        rpo: Vec::new(),
        idom: Vec::new(),
        children: Vec::new(),
        loop_header: Vec::new(),
        merge: Vec::new(),
    };

    // Root edges; then make every block reachable, adding the first
    // unreachable block as an unreferenced entry until none is left (a
    // cycle reached only indirectly has internal predecessors).
    let mut retreating;
    loop {
        graph.successors[0] = entries
            .iter()
            .filter(|(_, kinds)| kinds.is_control())
            .map(|(pc, _)| Edge::Node(index[pc]))
            .collect();
        let (rpo, edges) = graph.depth_first();
        graph.rpo = rpo;
        retreating = edges;
        match (1..graph.starts.len()).find(|&node| graph.rpo[node].is_none()) {
            Some(node) => entries.entry(graph.starts[node]).or_default().unreferenced = true,
            None => break,
        }
    }
    graph.dominators();
    let irreducible: Vec<IrreducibleEdge> = retreating
        .iter()
        .filter(|&&(from, to)| !graph.dominates(to, from))
        .map(|&(from, to)| IrreducibleEdge {
            from: graph.starts[from],
            to: graph.starts[to],
        })
        .collect();
    graph.classify();
    let loops = graph.natural_loops();
    let structure = if irreducible.is_empty() {
        graph.do_tree(0, &mut Vec::new())
    } else {
        None
    };
    WindowAnalysis {
        window,
        blocks: graph.starts[1..].to_vec(),
        entries: entries
            .iter()
            .map(|(&pc, &kinds)| WindowEntry { pc, kinds })
            .collect(),
        idom: (1..graph.starts.len())
            .map(|node| {
                let idom = graph.idom[node];
                (graph.starts[node], (idom != 0).then(|| graph.starts[idom]))
            })
            .collect(),
        loops,
        irreducible,
        structure,
    }
}

impl Graph {
    fn nodes(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        self.successors[node].iter().filter_map(|edge| match *edge {
            Edge::Node(next) => Some(next),
            Edge::Leave(_) => None,
        })
    }

    /// Reverse postorder numbers from the root, and the retreating edges
    /// (to a node still on the depth-first stack).
    fn depth_first(&self) -> (Vec<Option<usize>>, Vec<(usize, usize)>) {
        let count = self.starts.len();
        let mut visited = vec![false; count];
        let mut on_stack = vec![false; count];
        let mut postorder = Vec::new();
        let mut retreating = Vec::new();
        let mut stack = vec![(0, 0)];
        visited[0] = true;
        on_stack[0] = true;
        while let Some(&mut (node, ref mut position)) = stack.last_mut() {
            let next = self.nodes(node).nth(*position);
            *position += 1;
            match next {
                Some(next) if on_stack[next] => retreating.push((node, next)),
                Some(next) if !visited[next] => {
                    visited[next] = true;
                    on_stack[next] = true;
                    stack.push((next, 0));
                }
                Some(_) => {}
                None => {
                    on_stack[node] = false;
                    postorder.push(node);
                    stack.pop();
                }
            }
        }
        let mut rpo = vec![None; count];
        for (number, &node) in postorder.iter().rev().enumerate() {
            rpo[node] = Some(number);
        }
        (rpo, retreating)
    }

    /// Cooper, Harvey and Kennedy's iterative dominator algorithm.
    fn dominators(&mut self) {
        let count = self.starts.len();
        let mut order: Vec<usize> = (0..count).filter(|&n| self.rpo[n].is_some()).collect();
        order.sort_by_key(|&n| self.rpo[n]);
        let mut predecessors = vec![Vec::new(); count];
        for &node in &order {
            for next in self.nodes(node) {
                predecessors[next].push(node);
            }
        }
        const UNDEFINED: usize = usize::MAX;
        let mut idom = vec![UNDEFINED; count];
        idom[0] = 0;
        let rpo = |n: usize| self.rpo[n].expect("reachable");
        let mut changed = true;
        while changed {
            changed = false;
            for &node in order.iter().skip(1) {
                let mut new = UNDEFINED;
                for &pred in &predecessors[node] {
                    if idom[pred] == UNDEFINED {
                        continue;
                    }
                    new = if new == UNDEFINED {
                        pred
                    } else {
                        let (mut a, mut b) = (pred, new);
                        while a != b {
                            while rpo(a) > rpo(b) {
                                a = idom[a];
                            }
                            while rpo(b) > rpo(a) {
                                b = idom[b];
                            }
                        }
                        a
                    };
                }
                if idom[node] != new {
                    idom[node] = new;
                    changed = true;
                }
            }
        }
        let mut children = vec![Vec::new(); count];
        for &node in order.iter().skip(1) {
            children[idom[node]].push(node);
        }
        self.idom = idom;
        self.children = children;
    }

    fn dominates(&self, a: usize, mut b: usize) -> bool {
        loop {
            if a == b {
                return true;
            }
            if b == 0 {
                return false;
            }
            b = self.idom[b];
        }
    }

    /// Back-edges (to a dominator) mark loop headers; nodes with two or more
    /// forward in-edges are merge nodes.
    fn classify(&mut self) {
        let count = self.starts.len();
        let mut forward = vec![0_usize; count];
        let mut loop_header = vec![false; count];
        for node in 0..count {
            for next in self.nodes(node) {
                if self.dominates(next, node) {
                    loop_header[next] = true;
                } else {
                    forward[next] += 1;
                }
            }
        }
        self.loop_header = loop_header;
        self.merge = forward.into_iter().map(|n| n >= 2).collect();
    }

    fn natural_loops(&self) -> Vec<NaturalLoop> {
        let count = self.starts.len();
        let mut predecessors = vec![Vec::new(); count];
        for node in 0..count {
            for next in self.nodes(node) {
                predecessors[next].push(node);
            }
        }
        let mut headers: Vec<usize> = (1..count).filter(|&n| self.loop_header[n]).collect();
        headers.sort_by_key(|&n| self.rpo[n]);
        let bodies: Vec<(usize, Vec<usize>, BTreeSet<usize>)> = headers
            .iter()
            .map(|&header| {
                let latches: Vec<usize> = predecessors[header]
                    .iter()
                    .copied()
                    .filter(|&pred| self.dominates(header, pred))
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect();
                let mut body = BTreeSet::from([header]);
                let mut work: Vec<usize> = latches.clone();
                while let Some(node) = work.pop() {
                    if body.insert(node) {
                        work.extend(predecessors[node].iter().copied());
                    }
                }
                (header, latches, body)
            })
            .collect();
        let mut loops: Vec<NaturalLoop> = Vec::new();
        for (i, (header, latches, body)) in bodies.iter().enumerate() {
            // Outer loops come first: the innermost container is the last
            // earlier loop whose body holds this header.
            let parent = (0..i).rev().find(|&j| bodies[j].2.contains(header));
            let depth = parent.map_or(1, |p| loops[p].depth + 1);
            loops.push(NaturalLoop {
                header: self.starts[*header],
                latches: latches.iter().map(|&n| self.starts[n]).collect(),
                blocks: body.iter().map(|&n| self.starts[n]).collect(),
                parent,
                depth,
            });
        }
        loops
    }

    /// Ramsey's `doTree`.
    fn do_tree(&self, node: usize, context: &mut Vec<Label>) -> Option<Structured> {
        let mut merges: Vec<usize> = self.children[node]
            .iter()
            .copied()
            .filter(|&child| self.merge[child])
            .collect();
        // Highest reverse postorder first: its code comes last, outermost.
        merges.sort_by_key(|&child| std::cmp::Reverse(self.rpo[child]));
        if self.loop_header[node] {
            context.push(Label::LoopHeadedBy(node));
            let body = self.node_within(node, &merges, context);
            context.pop();
            Some(Structured::Loop(Box::new(body?)))
        } else {
            self.node_within(node, &merges, context)
        }
    }

    fn node_within(
        &self,
        node: usize,
        merges: &[usize],
        context: &mut Vec<Label>,
    ) -> Option<Structured> {
        if let Some((&merge, rest)) = merges.split_first() {
            context.push(Label::BlockFollowedBy(merge));
            let inner = self.node_within(node, rest, context);
            context.pop();
            return Some(seq(vec![
                Structured::Block(Box::new(inner?)),
                self.do_tree(merge, context)?,
            ]));
        }
        let edges = &self.successors[node];
        let mut code = Vec::new();
        if node != 0 {
            code.push(Structured::Code(self.starts[node]));
        }
        let ending = match edges.as_slice() {
            [] => Structured::Leave(None),
            [edge] => self.do_branch(node, *edge, context)?,
            [taken, fallthrough] if node != 0 => Structured::If {
                block: self.starts[node],
                taken: Box::new(self.do_branch(node, *taken, context)?),
                fallthrough: Box::new(self.do_branch(node, *fallthrough, context)?),
            },
            _ => Structured::Dispatch(
                edges
                    .iter()
                    .map(|&edge| {
                        let pc = match edge {
                            Edge::Node(next) => self.starts[next],
                            Edge::Leave(pc) => pc?,
                        };
                        Some((pc, self.do_branch(node, edge, context)?))
                    })
                    .collect::<Option<_>>()?,
            ),
        };
        code.push(ending);
        Some(seq(code))
    }

    fn do_branch(&self, from: usize, edge: Edge, context: &mut Vec<Label>) -> Option<Structured> {
        let to = match edge {
            Edge::Node(to) => to,
            Edge::Leave(pc) => return Some(Structured::Leave(pc)),
        };
        let label = if self.rpo[to] <= self.rpo[from] {
            Label::LoopHeadedBy(to)
        } else if self.merge[to] {
            Label::BlockFollowedBy(to)
        } else {
            return self.do_tree(to, context);
        };
        let depth = context.iter().rev().position(|&entry| entry == label)?;
        Some(Structured::Br(depth))
    }
}

/// A sequence, flattened, without one-element wrappers.
fn seq(parts: Vec<Structured>) -> Structured {
    let mut flat = Vec::new();
    for part in parts {
        match part {
            Structured::Seq(inner) => flat.extend(inner),
            other => flat.push(other),
        }
    }
    if flat.len() == 1 {
        flat.pop().expect("one element")
    } else {
        Structured::Seq(flat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::{CodeRegion, discover};
    use Structured::{Br, Code, Leave};

    /// Window-aligned: window 0x20 with the default window bits.
    const BASE: u64 = 0x8000;
    const WINDOW: u64 = BASE >> DEFAULT_WINDOW_BITS;

    fn at(offset: u64) -> u64 {
        BASE + offset
    }

    /// Places each run of words at its byte offset from `BASE`.
    fn translate(runs: &[(u64, &[u32])], entries: &[u64]) -> Translation {
        let mut bytes = Vec::new();
        for &(offset, words) in runs {
            let offset = offset as usize;
            if bytes.len() < offset + words.len() * 4 {
                bytes.resize(offset + words.len() * 4, 0);
            }
            for (i, word) in words.iter().enumerate() {
                bytes[offset + i * 4..offset + i * 4 + 4].copy_from_slice(&word.to_le_bytes());
            }
        }
        let region = CodeRegion {
            name: "test".into(),
            base: BASE,
            bytes,
        };
        let entries: Vec<u64> = entries.iter().map(|&offset| at(offset)).collect();
        discover(&[region], &entries, 4096).expect("discover test code")
    }

    fn windows(runs: &[(u64, &[u32])], entries: &[u64]) -> Vec<WindowAnalysis> {
        analyze_windows(&translate(runs, entries), DEFAULT_WINDOW_BITS)
    }

    fn seq(parts: Vec<Structured>) -> Structured {
        Structured::Seq(parts)
    }
    fn block(body: Structured) -> Structured {
        Structured::Block(Box::new(body))
    }
    fn looped(body: Structured) -> Structured {
        Structured::Loop(Box::new(body))
    }
    fn if_(offset: u64, taken: Structured, fallthrough: Structured) -> Structured {
        Structured::If {
            block: at(offset),
            taken: Box::new(taken),
            fallthrough: Box::new(fallthrough),
        }
    }
    fn code(offset: u64) -> Structured {
        Code(at(offset))
    }

    fn natural(header: u64, latches: &[u64], blocks: &[u64], parent: Option<usize>) -> NaturalLoop {
        NaturalLoop {
            header: at(header),
            latches: latches.iter().map(|&o| at(o)).collect(),
            blocks: blocks.iter().map(|&o| at(o)).collect(),
            parent,
            depth: parent.map_or(1, |_| 2),
        }
    }

    #[test]
    fn counted_loop() {
        let words = [
            0xd280_0148, // 0x0: mov x8, #10
            0xf100_0508, // 0x4: loop: subs x8, x8, #1
            0x54ff_ffe1, // 0x8: b.ne loop
            0xd65f_03c0, // 0xc: ret
        ];
        let [window] = windows(&[(0, &words)], &[0])
            .try_into()
            .expect("one window");
        assert_eq!(window.window, WINDOW);
        assert_eq!(window.blocks, vec![at(0), at(4), at(0xc)]);
        assert!(window.is_reducible() && !window.is_multi_entry());
        assert_eq!(window.idom[&at(4)], Some(at(0)));
        assert_eq!(window.idom[&at(0)], None);
        assert_eq!(window.loops, vec![natural(4, &[4], &[4], None)]);
        assert_eq!(
            window.structure,
            Some(seq(vec![
                code(0),
                looped(seq(vec![
                    code(4),
                    if_(4, Br(0), seq(vec![code(0xc), Leave(None)])),
                ])),
            ]))
        );
    }

    #[test]
    fn nested_loops() {
        let words = [
            0xd280_0089, // 0x0: mov x9, #4
            0xd280_0148, // 0x4: outer: mov x8, #10
            0xf100_0508, // 0x8: inner: subs x8, x8, #1
            0x54ff_ffe1, // 0xc: b.ne inner
            0xf100_0529, // 0x10: subs x9, x9, #1
            0x54ff_ff81, // 0x14: b.ne outer
            0xd65f_03c0, // 0x18: ret
        ];
        let [window] = windows(&[(0, &words)], &[0])
            .try_into()
            .expect("one window");
        assert_eq!(
            window.loops,
            vec![
                natural(4, &[0x10], &[4, 8, 0x10], None),
                natural(8, &[8], &[8], Some(0)),
            ]
        );
        assert_eq!(
            window.structure,
            Some(seq(vec![
                code(0),
                looped(seq(vec![
                    code(4),
                    looped(seq(vec![
                        code(8),
                        if_(
                            8,
                            Br(0),
                            seq(vec![
                                code(0x10),
                                if_(0x10, Br(1), seq(vec![code(0x18), Leave(None)])),
                            ]),
                        ),
                    ])),
                ])),
            ]))
        );
    }

    #[test]
    fn if_else_diamond() {
        let words = [
            0xb400_0060, // 0x0: cbz x0, else
            0xd280_0021, // 0x4: mov x1, #1
            0x1400_0002, // 0x8: b join
            0xd280_0041, // 0xc: else: mov x1, #2
            0x9100_0c20, // 0x10: join: add x0, x1, #3
            0xd65f_03c0, // 0x14: ret
        ];
        let [window] = windows(&[(0, &words)], &[0])
            .try_into()
            .expect("one window");
        assert!(window.loops.is_empty());
        assert_eq!(window.idom[&at(0x10)], Some(at(0)));
        assert_eq!(
            window.structure,
            Some(seq(vec![
                block(seq(vec![
                    code(0),
                    if_(0, seq(vec![code(0xc), Br(0)]), seq(vec![code(4), Br(0)])),
                ])),
                code(0x10),
                Leave(None),
            ]))
        );
    }

    #[test]
    fn early_return() {
        let words = [
            0xb400_0080, // 0x0: cbz x0, early
            0xf940_0001, // 0x4: ldr x1, [x0]
            0x9100_0420, // 0x8: add x0, x1, #1
            0xd65f_03c0, // 0xc: ret
            0xd280_0000, // 0x10: early: mov x0, #0
            0xd65f_03c0, // 0x14: ret
        ];
        let [window] = windows(&[(0, &words)], &[0])
            .try_into()
            .expect("one window");
        assert_eq!(
            window.structure,
            Some(seq(vec![
                code(0),
                if_(
                    0,
                    seq(vec![code(0x10), Leave(None)]),
                    seq(vec![code(4), Leave(None)]),
                ),
            ]))
        );
    }

    #[test]
    fn calls_inside_a_loop() {
        let caller = [
            0xd280_0153, // 0x0: mov x19, #10
            0x9400_00ff, // 0x4: loop: bl g
            0xd63f_0280, // 0x8: blr x20
            0xf100_0673, // 0xc: subs x19, x19, #1
            0x54ff_ffa1, // 0x10: b.ne loop
            0x1400_00fc, // 0x14: b h
        ];
        let callees = [
            0xd65f_03c0, // 0x400: g: ret
            0xd65f_03c0, // 0x404: h: ret
        ];
        let runs: [(u64, &[u32]); 2] = [(0, &caller), (0x400, &callees)];
        let translation = translate(&runs, &[0]);

        let sites = call_sites(&translation, DEFAULT_WINDOW_BITS, CallOptions::default());
        assert_eq!(
            sites,
            vec![
                CallSite {
                    pc: at(4),
                    return_to: at(8),
                    target: CallTarget::Direct {
                        entry: at(0x400),
                        window: WINDOW + 1,
                    },
                    preserved: Regs::NONE,
                    clobbered: Regs::ALL,
                },
                CallSite {
                    pc: at(8),
                    return_to: at(0xc),
                    target: CallTarget::Indirect { register: 20 },
                    preserved: Regs::NONE,
                    clobbered: Regs::ALL,
                },
            ]
        );
        let trusted = call_sites(
            &translation,
            DEFAULT_WINDOW_BITS,
            CallOptions {
                trust_aapcs64: true,
            },
        );
        let preserved = trusted[0].preserved;
        assert!((19..=29).all(|n| preserved.has_gpr(n)) && preserved.sp);
        assert!(
            (0..=18)
                .chain([30])
                .all(|n| trusted[0].clobbered.has_gpr(n))
        );
        assert!(!preserved.nzcv && preserved.vector == 0);
        assert_eq!(preserved.union(trusted[0].clobbered), Regs::ALL);

        let [caller, callees] = analyze_windows(&translation, DEFAULT_WINDOW_BITS)
            .try_into()
            .expect("two windows");
        // Return addresses are resume points, not control entries.
        assert_eq!(caller.control_entries().collect::<Vec<_>>(), vec![at(0)]);
        let resume: Vec<u64> = caller
            .entries
            .iter()
            .filter(|entry| entry.kinds.return_address)
            .map(|entry| entry.pc)
            .collect();
        assert_eq!(resume, vec![at(8), at(0xc)]);
        assert_eq!(caller.loops, vec![natural(4, &[0xc], &[4, 8, 0xc], None)]);
        assert_eq!(
            caller.structure,
            Some(seq(vec![
                code(0),
                looped(seq(vec![
                    code(4),
                    code(8),
                    code(0xc),
                    if_(0xc, Br(0), seq(vec![code(0x14), Leave(Some(at(0x404)))])),
                ])),
            ]))
        );
        assert_eq!(
            callees.entries,
            vec![
                WindowEntry {
                    pc: at(0x400),
                    kinds: EntryKinds {
                        call_target: true,
                        ..EntryKinds::default()
                    },
                },
                WindowEntry {
                    pc: at(0x404),
                    kinds: EntryKinds {
                        branch_from_outside: true,
                        ..EntryKinds::default()
                    },
                },
            ]
        );
    }

    #[test]
    fn irreducible_two_entry_loop_is_reported() {
        let words = [
            0xb400_0040, // 0x0: cbz x0, b
            0x9100_0421, // 0x4: a: add x1, x1, #1
            0xf100_0442, // 0x8: b: subs x2, x2, #1
            0x54ff_ffc1, // 0xc: b.ne a
            0xd65f_03c0, // 0x10: ret
        ];
        let [window] = windows(&[(0, &words)], &[0])
            .try_into()
            .expect("one window");
        assert!(!window.is_reducible());
        assert_eq!(
            window.irreducible,
            vec![IrreducibleEdge {
                from: at(4),
                to: at(8),
            }]
        );
        assert_eq!(window.structure, None);
        assert!(window.loops.is_empty());
    }

    #[test]
    fn window_with_several_entries() {
        let low = [
            0xd280_0020, // 0x0: f1: mov x0, #1
            0xd65f_03c0, // 0x4: ret
            0xb400_0040, // 0x8: f2: cbz x0, out
            0x1400_00fd, // 0xc: b f3
            0xd65f_03c0, // 0x10: out: ret
        ];
        let high = [
            0xd280_0060, // 0x400: f3: mov x0, #3
            0xd65f_03c0, // 0x404: ret
        ];
        let runs: [(u64, &[u32]); 2] = [(0, &low), (0x400, &high)];
        let translation = translate(&runs, &[0, 8]);
        let [first, second] = analyze_windows(&translation, DEFAULT_WINDOW_BITS)
            .try_into()
            .expect("two windows");
        assert!(first.is_multi_entry() && !second.is_multi_entry());
        assert_eq!(
            first.control_entries().collect::<Vec<_>>(),
            vec![at(0), at(8)]
        );
        assert_eq!(
            first.structure,
            Some(Structured::Dispatch(vec![
                (at(0), seq(vec![code(0), Leave(None)])),
                (
                    at(8),
                    seq(vec![
                        code(8),
                        if_(
                            8,
                            seq(vec![code(0x10), Leave(None)]),
                            seq(vec![code(0xc), Leave(Some(at(0x400)))]),
                        ),
                    ]),
                ),
            ]))
        );
        assert_eq!(second.structure, Some(seq(vec![code(0x400), Leave(None)])));
        assert_eq!(
            entry_windows(&translation, DEFAULT_WINDOW_BITS),
            BTreeMap::from([(at(0), WINDOW), (at(8), WINDOW), (at(0x400), WINDOW + 1),])
        );
    }
}
