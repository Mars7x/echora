//! Lifted functions as structured host code (`EmitOptions::regalloc`).
//!
//! A lifted function (`codegen::functions`) whose blocks form a reducible
//! graph from its entry becomes one host function with real control flow
//! (`crate::structure::function_structure`: loops, ifs and labelled
//! blocks), so a constant successor is a direct jump instead of a trip
//! through a `match next_pc`, and the host compiler sees every guest
//! register local's whole live range. Small leaf callees run inlined:
//! their blocks are copied into the caller's graph, a BL to them only sets
//! X30, and their return goes on in the caller when X30 still holds the
//! return address (anything else leaves, as their RET would).
//!
//! Steps: each block adds its count to the local `n`, which goes into the
//! steps before every call that is not inlined and when the function
//! returns, so the count is exact wherever it can be observed (a callee's
//! charges, traps, SVCs, the host after the function). A trap inside one
//! of its blocks keeps the instructions up to and including the trapping
//! one and clears the block tag (`tail`). The entry, every loop head and
//! every call return point test that the steps, `n` and the longest run of
//! instructions before the next test stay within the stop word; where they
//! might not, the function leaves at that block (every such block keeps
//! its window arm), and the windows' per-block charges go on from there:
//! budgets and host boundaries fall exactly on the blocks they fall on in
//! window code. Only a preemption another thread requests is seen at the
//! function's next test instead of the next block.

use super::*;
use crate::aarch64::{Address, Writeback};

/// Block keys: the block's start, with the number of the inlined copy it
/// belongs to (0: the function's own blocks) above `KEY_SHIFT`.
const KEY_SHIFT: u32 = 40;

fn pc_of(key: u64) -> u64 {
    key & ((1 << KEY_SHIFT) - 1)
}

/// Most guest instructions a callee may hold to run inlined.
pub const INLINE_LIMIT: usize = 24;

/// With `EmitOptions::regalloc_loop_inlining`: most guest instructions a
/// callee called inside a loop may hold to run inlined.
pub const LOOP_INLINE_LIMIT: usize = 96;

/// The blocks of `blocks` on a cycle among them.
fn in_cycles(blocks: &[BasicBlock], jump_tables: &BTreeMap<u64, Vec<u64>>) -> BTreeSet<u64> {
    let next: BTreeMap<u64, Vec<u64>> = blocks
        .iter()
        .map(|block| {
            (
                block.start,
                successors(block, jump_tables)
                    .into_iter()
                    .flatten()
                    .collect(),
            )
        })
        .collect();
    let mut cyclic = BTreeSet::new();
    for &start in next.keys() {
        let mut seen = BTreeSet::new();
        let mut pending: Vec<u64> = next[&start].clone();
        while let Some(pc) = pending.pop() {
            if pc == start {
                cyclic.insert(start);
                break;
            }
            if let Some(targets) = next.get(&pc)
                && seen.insert(pc)
            {
                pending.extend(targets.iter().copied());
            }
        }
    }
    cyclic
}

/// Most guest instructions a function may hold with its inlined copies.
pub const INLINED_FUNCTION_LIMIT: usize = 1024;

/// A lifted function's successors as `crate::structure::function_structure`
/// takes them: as `functions` follows them, a recovered jump table's
/// cases then any other target (`None`), and nothing past an instruction
/// that ends its block otherwise (a return, BR, SVC or trap).
fn successors(block: &BasicBlock, jump_tables: &BTreeMap<u64, Vec<u64>>) -> Vec<Option<u64>> {
    use Instruction::*;
    let last = block
        .instructions
        .last()
        .expect("blocks have an instruction");
    let next = last.pc.wrapping_add(4);
    match &last.instruction {
        Branch {
            target,
            link: false,
        } => vec![Some(*target)],
        Branch { link: true, .. } | BranchReg { link: true, .. } => vec![Some(next)],
        BranchCond { target, .. } | CompareBranch { target, .. } | TestBranch { target, .. } => {
            vec![Some(*target), Some(next)]
        }
        BranchReg { link: false, .. } if let Some(cases) = jump_tables.get(&last.pc) => cases
            .iter()
            .map(|&case| Some(case))
            .chain(std::iter::once(None))
            .collect(),
        instruction if instruction.ends_block() => Vec::new(),
        _ => vec![Some(next)],
    }
}

fn is_call(block: &BasicBlock) -> bool {
    matches!(
        block.instructions.last().map(|record| &record.instruction),
        Some(Instruction::Branch { link: true, .. } | Instruction::BranchReg { link: true, .. })
    )
}

/// Blocks whose per-block boundaries the window form keeps: spin-wait
/// heads and traced blocks. (Exclusive accesses run structured: no
/// boundary falls between a load and store exclusive there, so the pair
/// succeeds unless another host thread takes the reservation, as Arm
/// allows.)
fn plain(blocks: &[&BasicBlock], spin_heads: &BTreeSet<u64>) -> bool {
    blocks
        .iter()
        .all(|block| !spin_heads.contains(&block.start) && !traced_pcs().contains(&block.start))
}

/// The body of each block as structured emission runs it, or None when
/// one reads the step count other than through the flushes before calls.
fn bodies(blocks: &[&BasicBlock], inlined_calls: &BTreeSet<u64>) -> Option<BTreeMap<u64, String>> {
    let outer = IN_FUNCTION.replace(true);
    let outer_body = REGALLOC_BODY.replace(true);
    let outer_inlined = REGALLOC_INLINED.replace(inlined_calls.clone());
    let bodies: BTreeMap<u64, String> = blocks
        .iter()
        .map(|block| {
            let mut body = String::new();
            emit_block_body(&mut body, block, false, false, true);
            (block.start, body)
        })
        .collect();
    REGALLOC_INLINED.set(outer_inlined);
    REGALLOC_BODY.set(outer_body);
    IN_FUNCTION.set(outer);
    let observes = ["trace_hit", "m.steps", "enter_window"];
    let observed = bodies.values().any(|body| {
        let body = body.replace("m.steps += *n; *n = 0;", "");
        observes.iter().any(|text| body.contains(text))
    });
    (!observed).then_some(bodies)
}

/// Writes a lifted function's structured tree.
struct Emitter<'a> {
    bodies: &'a BTreeMap<u64, String>,
    successors: &'a BTreeMap<u64, Vec<Option<u64>>>,
    /// The blocks that test the stop word, with the most instructions any
    /// path from their start runs before the next such test or a call.
    checks: &'a BTreeMap<u64, u64>,
    /// The tests that also leave where SP is not 16-byte aligned (`sp_hoisted`).
    sp_checks: BTreeSet<u64>,
    /// The block whose code was emitted last.
    last_code: u64,
    /// Enclosing labels, innermost last: (is a loop, number).
    labels: Vec<(bool, usize)>,
    next_label: usize,
}

impl<'a> Emitter<'a> {
    fn code(&mut self, output: &mut String, key: u64, body: &str) {
        // Where each block starts, for the stores at exits (`exit_stores`).
        writeln!(output, "                // @in {key:#x}").unwrap();
        self.last_code = key;
        if let Some(ahead) = self.checks.get(&key) {
            // Any block of the paths from here could take a budget or host
            // boundary: the windows run them, with per-block charges. So
            // do they where SP is not 16-byte aligned, when the function
            // checks it only at its entries and call returns.
            let sp = if self.sp_checks.contains(&key) {
                " || m.cpu.sp & 15 != 0"
            } else {
                ""
            };
            writeln!(
                output,
                "                if m.steps + *n + {ahead} > m.stop.load(std::sync::atomic::Ordering::Relaxed){sp} {{ m.cpu.pc = {:#x}; return Ok(()); }}",
                pc_of(key)
            )
            .unwrap();
        }
        output.push_str(body);
    }

    fn body(&self, key: u64) -> &'a str {
        &self.bodies[&pc_of(key)]
    }

    /// Marks that what follows runs after the block `key` ended.
    fn after(output: &mut String, key: u64) {
        writeln!(output, "                // @out {key:#x}").unwrap();
    }

    fn label(&mut self, is_loop: bool) -> usize {
        let number = self.next_label;
        self.next_label += 1;
        self.labels.push((is_loop, number));
        number
    }

    /// False where the tree holds a node a single-entry reducible graph
    /// does not produce.
    fn emit(&mut self, output: &mut String, tree: &crate::structure::Structured) -> bool {
        use crate::structure::Structured::*;
        match tree {
            Code(key) => self.code(output, *key, self.body(*key)),
            Seq(parts) => {
                let mut index = 0;
                while index < parts.len() {
                    // A block ending in a conditional branch, then the `If`
                    // it decides: branch on the condition itself.
                    if let (
                        Code(key),
                        Some(If {
                            block,
                            taken,
                            fallthrough,
                        }),
                    ) = (&parts[index], parts.get(index + 1))
                        && block == key
                        && let Some((head, condition, to_taken, to_other)) =
                            split_conditional(self.body(*key))
                        && self.successors[key].first().copied().flatten().map(pc_of)
                            == Some(to_taken)
                    {
                        self.code(output, *key, head);
                        writeln!(
                            output,
                            "                if {condition} {{\n                next_pc = {to_taken:#x};"
                        )
                        .unwrap();
                        Self::after(output, *key);
                        if !self.emit(output, taken) {
                            return false;
                        }
                        writeln!(
                            output,
                            "                }} else {{\n                next_pc = {to_other:#x};"
                        )
                        .unwrap();
                        Self::after(output, *key);
                        if !self.emit(output, fallthrough) {
                            return false;
                        }
                        output.push_str("                }\n");
                        index += 2;
                        continue;
                    }
                    if !self.emit(output, &parts[index]) {
                        return false;
                    }
                    index += 1;
                }
            }
            Block(body) => {
                let number = self.label(false);
                writeln!(output, "                'b{number}: {{").unwrap();
                if !self.emit(output, body) {
                    return false;
                }
                output.push_str("                }\n");
                self.labels.pop();
            }
            Loop(body) => {
                let number = self.label(true);
                writeln!(output, "                'l{number}: loop {{").unwrap();
                if !self.emit(output, body) {
                    return false;
                }
                writeln!(
                    output,
                    "                break 'l{number};\n                }}"
                )
                .unwrap();
                self.labels.pop();
            }
            If {
                block,
                taken,
                fallthrough,
            } => {
                let Some(Some(to_taken)) = self.successors[block].first().copied() else {
                    return false;
                };
                writeln!(
                    output,
                    "                if next_pc == {:#x} {{",
                    pc_of(to_taken)
                )
                .unwrap();
                Self::after(output, *block);
                if !self.emit(output, taken) {
                    return false;
                }
                output.push_str("                } else {\n");
                Self::after(output, *block);
                if !self.emit(output, fallthrough) {
                    return false;
                }
                output.push_str("                }\n");
            }
            Switch { cases, default } => {
                // Follows the code of the block it decides.
                let block = self.last_code;
                output.push_str("                match next_pc {\n");
                for (keys, body) in cases {
                    let pattern: Vec<String> = keys
                        .iter()
                        .map(|key| format!("{:#x}", pc_of(*key)))
                        .collect();
                    writeln!(output, "                {} => {{", pattern.join(" | ")).unwrap();
                    Self::after(output, block);
                    if !self.emit(output, body) {
                        return false;
                    }
                    output.push_str("                }\n");
                }
                output.push_str("                _ => {\n");
                Self::after(output, block);
                if !self.emit(output, default) {
                    return false;
                }
                output.push_str("                }\n                }\n");
            }
            Br(depth) => {
                let Some(&(is_loop, number)) = self
                    .labels
                    .len()
                    .checked_sub(1 + depth)
                    .map(|at| &self.labels[at])
                else {
                    return false;
                };
                if is_loop {
                    writeln!(output, "                continue 'l{number};").unwrap();
                } else {
                    writeln!(output, "                break 'b{number};").unwrap();
                }
            }
            Leave(_) => {
                output.push_str(
                    "                m.cpu.pc = next_pc;\n                return Ok(());\n",
                );
            }
            Dispatch(cases) => {
                // Entered at one of several blocks (`m.cpu.pc`).
                output.push_str("                match next_pc {\n");
                for (key, body) in cases {
                    writeln!(output, "                {:#x} => {{", pc_of(*key)).unwrap();
                    if !self.emit(output, body) {
                        return false;
                    }
                    output.push_str("                }\n");
                }
                output.push_str("                _ => {\n                // @none\n                m.cpu.pc = next_pc;\n                return Ok(());\n                }\n                }\n");
            }
            Region(_) => return false,
        }
        true
    }
}

/// The loop heads (each loop's first block) of a tree, or with
/// `outermost` only those of loops in no other loop.
fn loop_heads(tree: &crate::structure::Structured, heads: &mut BTreeSet<u64>, outermost: bool) {
    use crate::structure::Structured::*;
    match tree {
        Loop(body) => {
            if let Some(head) = first_block(body) {
                heads.insert(head);
            }
            if !outermost {
                loop_heads(body, heads, outermost);
            }
        }
        Seq(parts) => parts
            .iter()
            .for_each(|part| loop_heads(part, heads, outermost)),
        Block(body) => loop_heads(body, heads, outermost),
        If {
            taken, fallthrough, ..
        } => {
            loop_heads(taken, heads, outermost);
            loop_heads(fallthrough, heads, outermost);
        }
        Switch { cases, default } => {
            cases
                .iter()
                .for_each(|(_, body)| loop_heads(body, heads, outermost));
            loop_heads(default, heads, outermost);
        }
        Dispatch(cases) => cases
            .iter()
            .for_each(|(_, body)| loop_heads(body, heads, outermost)),
        _ => {}
    }
}

/// A lifted function as structured host code: the blocks it can be
/// entered at (its entry, its outermost loop heads and its small inner
/// loops' heads), each with the host function to run there (the entry's,
/// or an inner loop's resume function), or None, writing
/// nothing, when it does not qualify (an irreducible graph; a spin-wait head, a
/// traced block, an exclusive access or anything reading the step count),
/// and the caller emits the window form.
pub(super) fn emit_function(
    output: &mut String,
    entry: u64,
    blocks: &[BasicBlock],
    spin_heads: &BTreeSet<u64>,
    jump_tables: &BTreeMap<u64, Vec<u64>>,
    functions: &BTreeMap<u64, Vec<u64>>,
    by_start: &BTreeMap<u64, &BasicBlock>,
) -> Option<Vec<(u64, u64)>> {
    let own: Vec<&BasicBlock> = blocks.iter().collect();
    if !plain(&own, spin_heads) {
        return None;
    }
    // Callees small enough to run inlined, with what they call.
    // Call sites whose callee runs inlined: those of at most
    // `INLINE_LIMIT` instructions with what they call, and with
    // `EmitOptions::regalloc_loop_inlining` those of at most
    // `LOOP_INLINE_LIMIT` called inside a loop; then every call inside an
    // inlined callee.
    let loops = REGALLOC_LOOP_INLINING.get();
    let cap = if loops {
        LOOP_INLINE_LIMIT
    } else {
        INLINE_LIMIT
    };
    let cyclic = if loops {
        in_cycles(blocks, jump_tables)
    } else {
        BTreeSet::new()
    };
    let mut memo = BTreeMap::new();
    let mut inlined: BTreeSet<u64> = BTreeSet::new();
    let mut targets: Vec<u64> = Vec::new();
    let mut size: usize = blocks.iter().map(|block| block.instructions.len()).sum();
    for block in blocks {
        let last = block
            .instructions
            .last()
            .expect("blocks have an instruction");
        let limit = if cyclic.contains(&block.start) {
            cap
        } else {
            INLINE_LIMIT
        };
        if let Instruction::Branch { target, link: true } = last.instruction
            && target != entry
            && let Some(callee) = inlinable(
                target,
                functions,
                by_start,
                spin_heads,
                cap,
                &mut memo,
                &mut Vec::new(),
            )
            && callee <= limit
        {
            size += callee;
            inlined.insert(last.pc);
            targets.push(target);
        }
    }
    let own_size: usize = blocks.iter().map(|block| block.instructions.len()).sum();
    if own_size > FUNCTION_INSTRUCTIONS {
        capped(entry, "instructions");
        return None;
    }
    if size > INLINED_FUNCTION_LIMIT || size > FUNCTION_INSTRUCTIONS {
        inlined.clear();
    } else {
        // Every call inside an inlined callee runs inlined too.
        let mut seen = BTreeSet::new();
        while let Some(target) = targets.pop() {
            if !seen.insert(target) {
                continue;
            }
            for start in &functions[&target] {
                let last = by_start[start]
                    .instructions
                    .last()
                    .expect("blocks have an instruction");
                if let Instruction::Branch { target, link: true } = last.instruction {
                    inlined.insert(last.pc);
                    targets.push(target);
                }
            }
        }
    }
    let start = output.len();
    let mut emitted = emit_with(
        output,
        entry,
        &own,
        jump_tables,
        functions,
        by_start,
        &inlined,
    );
    // Inlining that makes the function's Rust larger than
    // FUNCTION_BYTES is given up: compile time and memory grow faster than
    // the function.
    if emitted.is_some() && !inlined.is_empty() && output.len() - start > function_bytes() {
        output.truncate(start);
        emitted = None;
    }
    let (inlined, (entries, inner)) = match emitted {
        Some(emitted) => (inlined, emitted),
        None if !inlined.is_empty() => (
            BTreeSet::new(),
            emit_with(
                output,
                entry,
                &own,
                jump_tables,
                functions,
                by_start,
                &BTreeSet::new(),
            )?,
        ),
        None => return None,
    };
    // Too large even without inlining: left to the windows.
    if output.len() - start > function_bytes() {
        output.truncate(start);
        capped(entry, "bytes");
        return None;
    }
    let mut entered: Vec<(u64, u64)> = entries.iter().map(|&block| (block, entry)).collect();
    // After a host boundary execution resumes at a loop head through the
    // windows. An inner loop's head is no entry (it would be a second way
    // into the loop around it), so with
    // `EmitOptions::regalloc_resume_loops` a small inner loop gets a resume
    // function of its own: its blocks reached from its head before any of
    // the function's entries, which it leaves to.
    let inner = if REGALLOC_RESUME_LOOPS.get() || REGALLOC_RESUME_RETURNS.get() {
        inner
    } else {
        Vec::new()
    };
    for head in inner {
        // Not a function's entry, nor a block another function already
        // runs (both would name a second `run_function_<head>`).
        if functions.contains_key(&head)
            || REGALLOC_ENTRIES.with_borrow(|entries| entries.contains_key(&head))
        {
            continue;
        }
        let next: BTreeMap<u64, Vec<u64>> = own
            .iter()
            .map(|block| {
                (
                    block.start,
                    successors(block, jump_tables)
                        .into_iter()
                        .flatten()
                        .collect(),
                )
            })
            .collect();
        if !next.contains_key(&head) {
            continue;
        }
        let mut reached = BTreeSet::from([head]);
        let mut pending = vec![head];
        while let Some(pc) = pending.pop() {
            for &target in next.get(&pc).into_iter().flatten() {
                if next.contains_key(&target)
                    && !entries.contains(&target)
                    && reached.insert(target)
                {
                    pending.push(target);
                }
            }
        }
        let part: Vec<&BasicBlock> = own
            .iter()
            .copied()
            .filter(|block| reached.contains(&block.start))
            .collect();
        let size: usize = part.iter().map(|block| block.instructions.len()).sum();
        if size > RESUME_LIMIT {
            continue;
        }
        if let Some((resumed, _)) = emit_with(
            output,
            head,
            &part,
            jump_tables,
            functions,
            by_start,
            &inlined,
        ) {
            entered.extend(resumed.into_iter().map(|block| (block, head)));
        }
    }
    Some(entered)
}

/// The bytes of Rust a structured function may take: a larger one is
/// emitted again without inlining, and if still larger not at all (its
/// entry gets `capped_stub`, and the windows run it).
pub const FUNCTION_BYTES: usize = 500_000;

/// `FUNCTION_BYTES`, or `EmitOptions::regalloc_function_bytes`.
fn function_bytes() -> usize {
    match REGALLOC_FUNCTION_BYTES.get() {
        0 => FUNCTION_BYTES,
        bytes => bytes,
    }
}

/// The guest instructions a structured function may hold with its
/// inlined callees: past it, without inlining; its own past it, not at all.
pub const FUNCTION_INSTRUCTIONS: usize = 4000;

/// The functions left to the windows by the caps, and which cap.
static CAPPED: std::sync::Mutex<Vec<(u64, &'static str)>> = std::sync::Mutex::new(Vec::new());

fn capped(entry: u64, cap: &'static str) {
    CAPPED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push((entry, cap));
    REGALLOC_CAPPED.with_borrow_mut(|capped| capped.insert(entry));
}

/// The functions the size caps left to the windows since the last call
/// (entry, and `"instructions"` or `"bytes"`), sorted.
pub fn take_capped_functions() -> Vec<(u64, &'static str)> {
    let mut capped = std::mem::take(
        &mut *CAPPED
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
    );
    capped.sort_unstable();
    capped
}

/// The host function of an entry the caps leave to the windows: it runs
/// nothing and returns at once, its PC still the entry, so a caller goes
/// on through the windows as after any early exit, having stored every
/// local it holds (and a window arm never runs it: the entry is not
/// structured). Running the function through the window table to X30
/// instead (`Core::call`) is not safe: a window runs on past a RET into
/// whatever of the caller shares the window, reading registers a
/// summarized caller still holds in locals (`tests/window_calls.rs`'s
/// `regalloc_capped` lift fails that way).
pub(super) fn capped_stub(output: &mut String, entry: u64) {
    writeln!(
        output,
        "\n/// Over the size caps: the windows run it.\n#[inline(never)]\npub fn run_function_{entry:x}(_m: &mut Machine) -> Result<(), Trap> {{ Ok(()) }}"
    )
    .unwrap();
}

/// Most guest instructions of an inner loop's resume function (its blocks
/// up to the function's next entry).
pub const RESUME_LIMIT: usize = 128;

/// The instructions `target` runs inlined (its own and those of what it
/// calls, all inlined), when it can: a lifted function, plain, calling
/// only such functions (no BLR), none of them recursive, at most `cap`
/// instructions in all.
#[allow(clippy::too_many_arguments)]
fn inlinable(
    target: u64,
    functions: &BTreeMap<u64, Vec<u64>>,
    by_start: &BTreeMap<u64, &BasicBlock>,
    spin_heads: &BTreeSet<u64>,
    cap: usize,
    memo: &mut BTreeMap<u64, Option<usize>>,
    stack: &mut Vec<u64>,
) -> Option<usize> {
    if let Some(&known) = memo.get(&target) {
        return known;
    }
    if stack.contains(&target) {
        return None;
    }
    let blocks: Vec<&BasicBlock> = functions
        .get(&target)?
        .iter()
        .map(|start| by_start[start])
        .collect();
    let mut size: Option<usize> = plain(&blocks, spin_heads)
        .then(|| blocks.iter().map(|block| block.instructions.len()).sum());
    stack.push(target);
    for block in &blocks {
        if size.is_none_or(|size| size > cap) {
            break;
        }
        if !is_call(block) {
            continue;
        }
        let last = block
            .instructions
            .last()
            .expect("blocks have an instruction");
        let callee = match last.instruction {
            Instruction::Branch { target, link: true } => {
                inlinable(target, functions, by_start, spin_heads, cap, memo, stack)
            }
            _ => None,
        };
        size = size.zip(callee).map(|(size, callee)| size + callee);
    }
    stack.pop();
    let size = size.filter(|&size| size <= cap);
    memo.insert(target, size);
    size
}

fn emit_with(
    output: &mut String,
    entry: u64,
    own: &[&BasicBlock],
    jump_tables: &BTreeMap<u64, Vec<u64>>,
    functions: &BTreeMap<u64, Vec<u64>>,
    by_start: &BTreeMap<u64, &BasicBlock>,
    inlined: &BTreeSet<u64>,
) -> Option<(Vec<u64>, Vec<u64>)> {
    // The graph: the function's own blocks, then a copy of each inlined
    // callee for each call of it (the calls inside a copy too).
    let mut graph: BTreeMap<u64, Vec<Option<u64>>> = BTreeMap::new();
    let mut counts: BTreeMap<u64, u64> = BTreeMap::new();
    let mut calls: BTreeSet<u64> = BTreeSet::new();
    let mut returns: BTreeSet<u64> = BTreeSet::new();
    let mut inlined_calls: BTreeSet<u64> = BTreeSet::new();
    let mut emitted: Vec<&BasicBlock> = own.to_vec();
    // (tag of the copy, its blocks, the key its returns go on at).
    let mut pending: Vec<(u64, Vec<&BasicBlock>, Option<u64>)> = vec![(0, own.to_vec(), None)];
    let mut copies: u64 = 0;
    while let Some((tag, blocks, returns_to)) = pending.pop() {
        let starts: BTreeSet<u64> = blocks.iter().map(|block| block.start).collect();
        let key = |pc: u64| if starts.contains(&pc) { pc | tag } else { pc };
        for block in blocks {
            let last = block
                .instructions
                .last()
                .expect("blocks have an instruction");
            let mut targets: Vec<Option<u64>> = successors(block, jump_tables)
                .into_iter()
                .map(|target| target.map(key))
                .collect();
            if let Instruction::Branch { target, link: true } = last.instruction
                && inlined.contains(&last.pc)
            {
                copies += 1;
                let copy = copies << KEY_SHIFT;
                inlined_calls.insert(last.pc);
                targets = vec![Some(target | copy)];
                let callee: Vec<&BasicBlock> = functions[&target]
                    .iter()
                    .map(|start| by_start[start])
                    .collect();
                emitted.extend(callee.iter().copied());
                pending.push((copy, callee, Some(key(last.pc + 4))));
            } else if is_call(block) {
                calls.insert(block.start | tag);
                returns.insert(key(last.pc + 4));
            } else if let Some(returns_to) = returns_to
                && targets.is_empty()
                && matches!(last.instruction, Instruction::BranchReg { link: false, .. })
            {
                // A return: on in the caller while X30 holds the return
                // address, anywhere else leaves.
                targets = vec![Some(returns_to), None];
            }
            graph.insert(block.start | tag, targets);
            counts.insert(block.start | tag, block.instructions.len() as u64);
        }
    }
    let edges: Vec<(u64, Vec<Option<u64>>)> = graph
        .iter()
        .map(|(key, targets)| (*key, targets.clone()))
        .collect();
    let single = crate::structure::function_structure(&[entry], &edges)?;
    // Also entered at the heads of its outermost loops of its own blocks,
    // where execution resumes after a host boundary (an inner loop's head
    // would be a second way into the loop around it).
    let mut outer = BTreeSet::new();
    loop_heads(&single, &mut outer, true);
    let entries: Vec<u64> = std::iter::once(entry)
        .chain(
            outer
                .into_iter()
                .filter(|&head| head != entry && head >> KEY_SHIFT == 0),
        )
        .collect();
    // The heads of its own inner loops.
    let mut inner = BTreeSet::new();
    loop_heads(&single, &mut inner, false);
    // With regalloc_resume_returns, also the blocks its calls return to
    // (a failed test there hands back to the windows).
    if REGALLOC_RESUME_RETURNS.get() {
        inner.extend(returns.iter().copied());
    }
    let inner: Vec<u64> = inner
        .into_iter()
        .filter(|&head| {
            // Its own blocks only: a call ending the body returns past it.
            head >> KEY_SHIFT == 0 && graph.contains_key(&head) && !entries.contains(&head)
        })
        .collect();
    let tree = if entries.len() > 1
        && let Some(tree) = crate::structure::function_structure(&entries, &edges)
    {
        tree
    } else {
        return emit_tree(
            output,
            entry,
            &[entry],
            single,
            &graph,
            &counts,
            &calls,
            &returns,
            &unique_blocks(own, &emitted),
            &inlined_calls,
        )
        .map(|entries| (entries, inner));
    };
    emit_tree(
        output,
        entry,
        &entries,
        tree,
        &graph,
        &counts,
        &calls,
        &returns,
        &unique_blocks(own, &emitted),
        &inlined_calls,
    )
    .map(|entries| (entries, inner))
}

/// Whether every SP write in `blocks` adds or subtracts a multiple of 16
/// (a pre- or post-indexed stack access, ADD or SUB of an immediate to SP
/// itself): then SP keeps the alignment it has at the entries and after
/// calls, and its alignment needs checking only there.
fn sp_hoisted(blocks: &[&BasicBlock]) -> bool {
    use Instruction::*;
    blocks.iter().all(|block| {
        block.instructions.iter().all(|record| {
            if !crate::defuse::effects(&record.instruction).writes.sp {
                return true;
            }
            match &record.instruction {
                LoadStore {
                    rn: 31,
                    address: Address::Immediate { offset, writeback },
                    ..
                } => *writeback != Writeback::None && offset % 16 == 0,
                LoadStorePair {
                    rn: 31,
                    offset,
                    writeback,
                    ..
                }
                | LoadStoreVectorPair {
                    rn: 31,
                    offset,
                    writeback,
                    ..
                } => *writeback != Writeback::None && offset % 16 == 0,
                AddSubImmediate {
                    rd: 31,
                    rn: 31,
                    bits: 64,
                    immediate,
                    flags: false,
                    ..
                } => immediate % 16 == 0,
                // Calls: the return point checks again.
                Branch { link: true, .. } | BranchReg { link: true, .. } => true,
                _ => false,
            }
        })
    })
}

/// The rewritten body (`RegisterLocals::rewrite`, with the emitter's
/// `// @in`, `// @out` and `// @none` marks) with a store before every
/// `return` of the locals possibly written on some path to it (`W`, a
/// fixpoint over the blocks: none at an entry), and without the stores
/// before calls and helpers of locals outside `W` (a local nothing wrote
/// still equals the `Core`: loaded from it, or reloaded after whatever
/// wrote it there).
fn exit_stores(
    text: &str,
    locals: &RegisterLocals,
    graph: &BTreeMap<u64, Vec<Option<u64>>>,
    entries: &[u64],
) -> String {
    let marker = |line: &str, prefix: &str| -> Option<u64> {
        let rest = line.trim_start().strip_prefix(prefix)?;
        u64::from_str_radix(rest.trim().trim_start_matches("0x"), 16).ok()
    };
    // What each block's own lines write.
    let mut writes: BTreeMap<u64, Dirty> = BTreeMap::new();
    let mut current: Option<u64> = None;
    for line in text.lines() {
        if line.trim_start().starts_with("// @") {
            current = marker(line, "// @in ");
            continue;
        }
        if let Some(key) = current {
            *writes.entry(key).or_default() |= local_writes(line);
        }
    }
    // Possibly written on entry to each block.
    let mut written_in: BTreeMap<u64, Dirty> = graph.keys().map(|&key| (key, 0)).collect();
    let mut changed = true;
    while changed {
        changed = false;
        for (&key, targets) in graph {
            let out = written_in[&key] | writes.get(&key).copied().unwrap_or(0);
            for target in targets.iter().flatten() {
                if let Some(to) = written_in.get_mut(target)
                    && *to | out != *to
                {
                    *to |= out;
                    changed = true;
                }
            }
        }
    }
    let _ = entries;
    let store = |dirty: Dirty| -> String {
        let mut text = locals.store_dirty(dirty);
        // FPCR and FPSR are loaded and stored whole (`store_dirty`).
        if !text.is_empty() && !text.ends_with(' ') {
            text.push(' ');
        }
        text
    };
    let mut output = String::with_capacity(text.len() * 2);
    let mut written: Dirty = 0;
    for line in text.lines() {
        if line.trim_start().starts_with("// @") {
            if let Some(key) = marker(line, "// @in ") {
                written = written_in[&key];
            } else if let Some(key) = marker(line, "// @out ") {
                written = written_in[&key] | writes.get(&key).copied().unwrap_or(0);
            } else {
                written = 0;
            }
            continue;
        }
        let mut line = line.to_owned();
        // Stores before calls and helpers of locals nothing wrote.
        for index in &locals.written {
            if written & 1 << index == 0 {
                line = line.replace(&format!("m.cpu.regs[{index}] = r.x{index};"), "");
            }
        }
        if written & DIRTY_SP == 0 {
            line = line.replace("m.cpu.sp = r.sp;", "");
        }
        if written & DIRTY_NZCV == 0 {
            line = line.replace("m.cpu.nzcv = r.fl.nzcv;", "");
        }
        let here = written | local_writes(&line);
        if line.contains("return Ok(())") || line.contains("return Err(") {
            let stores = store(here);
            line = line
                .replace("return Ok(())", &format!("{stores}return Ok(())"))
                .replace("return Err(", &format!("{stores}return Err("));
        }
        written = here;
        output.push_str(&line);
        output.push('\n');
    }
    output
}

/// A call's result returned as it is on its error path (`if
/// called.is_err() { .. return called; }`) instead of moved into a new
/// `Err`: the host compiler then keeps the `Result` where the callee wrote
/// it instead of copying it around every call.
fn returned_results(text: &str) -> String {
    const HEAD: &str = "if let Err(trap) = called { ";
    const TAIL: &str = "return Err(trap); }";
    let mut output = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find(HEAD) {
        let after = &rest[at + HEAD.len()..];
        let Some(end) = after.find(TAIL) else {
            break;
        };
        let stores = &after[..end];
        output.push_str(&rest[..at]);
        if stores.contains('}') || stores.contains('{') {
            // Not a plain error path: unchanged.
            output.push_str(HEAD);
            rest = after;
            continue;
        }
        output.push_str("if called.is_err() { ");
        output.push_str(stores);
        output.push_str("return called; }");
        rest = &after[end + TAIL.len()..];
    }
    output.push_str(rest);
    output
}

/// `text` with each `EXPR?` written out as `(match EXPR { Ok(value) =>
/// value, Err(error) => { return Err(error); } })`, so that every exit is
/// a `return` that `exit_stores` sees; None (and `?` stays) where an
/// operand cannot be told apart safely: a string or a closure around it.
fn explicit_errors(text: &str) -> Option<String> {
    let mut output = String::with_capacity(text.len() + text.len() / 8);
    for line in text.lines() {
        let bytes = line.as_bytes();
        let mut rewritten = String::with_capacity(line.len());
        let mut copied = 0;
        let mut quoted = false;
        let mut escaped = false;
        for (at, &byte) in bytes.iter().enumerate() {
            if quoted {
                if escaped {
                    escaped = false;
                } else if byte == b'\\' {
                    escaped = true;
                } else if byte == b'"' {
                    quoted = false;
                }
                continue;
            }
            match byte {
                b'"' => quoted = true,
                b'?' => {
                    // The operand: a chain of paths, calls and indexing.
                    let mut start = at;
                    loop {
                        while start > 0
                            && (bytes[start - 1].is_ascii_alphanumeric()
                                || matches!(bytes[start - 1], b'_' | b'.' | b':'))
                        {
                            start -= 1;
                        }
                        if start > 0 && matches!(bytes[start - 1], b')' | b']') {
                            let mut depth = 0i32;
                            let mut open = start;
                            loop {
                                if open == 0 {
                                    return None;
                                }
                                open -= 1;
                                match bytes[open] {
                                    b')' | b']' => depth += 1,
                                    b'(' | b'[' => {
                                        depth -= 1;
                                        if depth == 0 {
                                            break;
                                        }
                                    }
                                    b'"' | b'\'' | b'|' => return None,
                                    _ => {}
                                }
                            }
                            start = open;
                            continue;
                        }
                        break;
                    }
                    // Inside a closure (`|..| expr`, ending at its `;`)
                    // `return` would leave the closure: only past one.
                    let in_closure = line[..start]
                        .rfind('|')
                        .is_some_and(|bar| !line[bar..start].contains(';'));
                    if start == at || start < copied || in_closure {
                        return None;
                    }
                    rewritten.push_str(&line[copied..start]);
                    write!(
                        rewritten,
                        "(match {} {{ Ok(value) => value, Err(error) => {{ return Err(error); }} }})",
                        &line[start..at]
                    )
                    .unwrap();
                    copied = at + 1;
                }
                _ => {}
            }
        }
        rewritten.push_str(&line[copied..]);
        output.push_str(&rewritten);
        output.push('\n');
    }
    Some(output)
}

/// Whether a line hands the `Core` to code that may read or write vector
/// registers there: a helper taking the CPU, or any call.
fn hands_over_core(line: &str) -> bool {
    ["(m)", "(m,", "m.call"]
        .iter()
        .any(|text| line.contains(text))
        || helpers_outside_wrappers(line)
}

/// Whether a line hands the CPU to a helper other than in a wrapped call
/// (`let called = helper(&mut m.cpu, ..);`, `wrap_cpu_helpers`).
fn helpers_outside_wrappers(line: &str) -> bool {
    let mut rest = line;
    let mut wrapped = 0;
    while let Some(at) = rest.find("let called = ") {
        let call = &rest[at..];
        let end = statement_end(call).unwrap_or(call.len());
        wrapped += call[..end].matches("m.cpu").count() - call[..end].matches("m.cpu.").count();
        rest = &call[end..];
    }
    let all = line.matches("m.cpu").count() - line.matches("m.cpu.").count();
    all != wrapped
}

/// The index of the `;` ending the statement `text` starts with.
fn statement_end(text: &str) -> Option<usize> {
    let mut depth = 0i32;
    text.char_indices()
        .find(|&(_, c)| {
            match c {
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' => depth -= 1,
                ';' if depth == 0 => return true,
                _ => {}
            }
            false
        })
        .map(|(end, _)| end)
}

/// `line` with `reload` after each wrapped helper call that may write the
/// CPU.
fn reload_after_helpers(line: &str, reload: &str) -> String {
    let mut output = String::with_capacity(line.len() + 64);
    let mut rest = line;
    while let Some(at) = rest.find("let called = ") {
        let call = &rest[at..];
        let end = statement_end(call).map_or(call.len(), |end| end + 1);
        output.push_str(&rest[..at]);
        output.push_str(&call[..end]);
        if call[..end].contains("&mut m.cpu") {
            output.push(' ');
            output.push_str(reload);
        }
        rest = &call[end..];
    }
    output.push_str(rest);
    output
}

/// The structured body with vector registers in locals
/// (`EmitOptions::regalloc_vector_locals`): `r.vN` for `m.cpu.vectors[N]`,
/// written through (a write sets the local and stores the register at
/// once, so the `Core` always holds every vector register and no exit,
/// trap or call needs a store), read from the local. A line that hands
/// the `Core` to a helper or a call keeps its own accesses, and every
/// vector local is loaded again after it. None (unchanged) where a vector
/// register is named other than by a constant index. Also the registers
/// used.
fn vector_locals(text: &str) -> Option<(String, BTreeSet<u8>)> {
    const VECTORS: &str = "m.cpu.vectors";
    let mut used: BTreeSet<u8> = BTreeSet::new();
    for (at, _) in text.match_indices(VECTORS) {
        let rest = text[at + VECTORS.len()..].strip_prefix('[')?;
        let end = rest.find(']')?;
        used.insert(rest[..end].parse::<u8>().ok().filter(|&n| n < 32)?);
    }
    if used.is_empty() {
        return Some((text.to_owned(), used));
    }
    let reload: String = used
        .iter()
        .map(|n| format!("r.v{n} = m.cpu.vectors[{n}]; "))
        .collect();
    let mut output = String::with_capacity(text.len() * 2);
    for line in text.lines() {
        if hands_over_core(line) {
            // The reload must follow everything the line runs.
            let code = without_strings(line);
            if code.matches('{').count() != code.matches('}').count() {
                return None;
            }
            output.push_str(line);
            output.push('\n');
            output.push_str(&reload);
            output.push('\n');
            continue;
        }
        // Writes first (to a placeholder the reads leave alone), then reads.
        let mut rewritten = String::with_capacity(line.len() + 32);
        let mut rest = line;
        while let Some(at) = rest.find(VECTORS) {
            let after = &rest[at + VECTORS.len() + 1..];
            let close = after.find(']').expect("checked above");
            let number = &after[..close];
            let tail = &after[close + 1..];
            rewritten.push_str(&rest[..at]);
            if let Some(expression) = tail.strip_prefix(" = ") {
                // The statement's end: `;` outside brackets.
                let mut depth = 0i32;
                let end = expression
                    .char_indices()
                    .find(|&(_, c)| {
                        match c {
                            '(' | '[' | '{' => depth += 1,
                            ')' | ']' | '}' => depth -= 1,
                            ';' if depth == 0 => return true,
                            _ => {}
                        }
                        false
                    })
                    .map(|(end, _)| end)?;
                write!(
                    rewritten,
                    "r.v{number} = {}; \u{1}{number}\u{2} = r.v{number}",
                    &expression[..end]
                )
                .unwrap();
                rest = &expression[end..];
            } else {
                write!(rewritten, "r.v{number}").unwrap();
                rest = tail;
            }
        }
        rewritten.push_str(rest);
        // Reads inside the written expressions too; the placeholders are
        // the stores.
        for n in &used {
            rewritten = rewritten.replace(&format!("{VECTORS}[{n}]"), &format!("r.v{n}"));
        }
        let rewritten = rewritten
            .replace('\u{1}', "m.cpu.vectors[")
            .replace('\u{2}', "]");
        let rewritten = if rewritten.contains("&mut m.cpu") {
            reload_after_helpers(&rewritten, &reload)
        } else {
            rewritten
        };
        output.push_str(&rewritten);
        output.push('\n');
    }
    Some((output, used))
}

/// The PC stores in `text` (`m.cpu.pc = K;` lines) moved to where the
/// code after them can observe the PC (`EmitOptions::regalloc_lazy_pc`):
/// before each wrapped helper call and each `return` up to the next PC
/// store. Every instruction that may observe the PC stores its own
/// (`needs_pc`), and so do exits and stop-word bails, so the PC is the
/// store's value at each place it can be seen, as before. A store stays
/// where any line before the next one could see the PC otherwise: a call
/// taking the `Machine`, an unwrapped helper, a read or write of the PC,
/// any other use of `m`. Needs `text` without `?` (`explicit_errors`).
fn lazy_pc(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let store_of = |line: &str| {
        let line = line.trim();
        line.strip_prefix("m.cpu.pc = ")
            .and_then(|rest| rest.strip_suffix(';'))
            .filter(|value| {
                value.starts_with("0x") && value[2..].chars().all(|c| c.is_ascii_hexdigit())
            })
            .map(str::to_owned)
    };
    let mut output = String::with_capacity(text.len() + text.len() / 8);
    let mut index = 0;
    while index < lines.len() {
        let line = lines[index];
        index += 1;
        let Some(value) = store_of(line) else {
            output.push_str(line);
            output.push('\n');
            continue;
        };
        // The lines up to the next PC store, all of them able to take the
        // store where they observe the PC.
        let end = lines[index..]
            .iter()
            .position(|line| store_of(line).is_some())
            .map(|offset| index + offset);
        let sinkable = end.is_some_and(|end| {
            lines[index..end]
                .iter()
                .all(|line| pc_observed_only_at_exits(line))
        });
        if !sinkable {
            output.push_str(line);
            output.push('\n');
            continue;
        }
        let store = format!("m.cpu.pc = {value}; ");
        for line in &lines[index..end.expect("sinkable")] {
            output.push_str(
                &line
                    .replace("let called = ", &format!("{store}let called = "))
                    .replace("return ", &format!("{store}return ")),
            );
            output.push('\n');
        }
        index = end.expect("sinkable");
    }
    output
}

/// Whether `line` can observe the PC only at its wrapped helper calls
/// (`let called = helper(&mut m.cpu, ..)`) and its `return`s: it names
/// no other part of the `Machine` than CPU fields other than the PC, its
/// pure flag helpers (as `needs_pc`), memory and the step counters.
fn pc_observed_only_at_exits(line: &str) -> bool {
    const PURE: &[&str] = &[
        "add_sub(",
        "add_with_carry(",
        "condition(",
        "logical_flags(",
        "conditional_compare(",
        "conditional_compare_negative(",
    ];
    let code = without_strings(line);
    if code.contains('?') || code.contains("m.cpu.pc") || helpers_outside_wrappers(&code) {
        return false;
    }
    let bytes = code.as_bytes();
    let ident = |byte: u8| byte.is_ascii_alphanumeric() || byte == b'_';
    // The `Machine` itself, handed to a call or bound.
    let whole_machine = bytes.iter().enumerate().any(|(at, &byte)| {
        byte == b'm'
            && (at == 0 || !(ident(bytes[at - 1]) || bytes[at - 1] == b'.'))
            && bytes
                .get(at + 1)
                .is_none_or(|&next| !ident(next) && next != b'.')
    });
    if whole_machine {
        return false;
    }
    code.match_indices("m.").all(|(at, _)| {
        if at > 0
            && (bytes[at - 1].is_ascii_alphanumeric()
                || bytes[at - 1] == b'_'
                || bytes[at - 1] == b'.')
        {
            return true;
        }
        let rest = &code[at + 2..];
        if rest.starts_with("cpu")
            && !rest[3..].starts_with(|c: char| c.is_ascii_alphanumeric() || c == '_' || c == '.')
        {
            // The CPU handed to a wrapped helper (checked above).
            return true;
        }
        if let Some(field) = rest.strip_prefix("cpu.") {
            let end = field
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .unwrap_or(field.len());
            // A field, or a pure method.
            !field[end..].starts_with('(') || PURE.iter().any(|pure| field.starts_with(pure))
        } else {
            ["memory.load_u", "memory.store_u", "steps", "stop."]
                .iter()
                .any(|allowed| rest.starts_with(allowed))
        }
    })
}

/// The locals' declaration and load (`RegisterLocals::load`) with a
/// `u128` field per vector register in `vectors`.
fn with_vector_fields(load: &str, vectors: &BTreeSet<u8>) -> String {
    let fields: String = vectors.iter().map(|n| format!("v{n}: u128, ")).collect();
    let values: String = vectors
        .iter()
        .map(|n| format!("v{n}: m.cpu.vectors[{n}], "))
        .collect();
    load.replacen("struct R { ", &format!("struct R {{ {fields}"), 1)
        .replacen("let mut r = R { ", &format!("let mut r = R {{ {values}"), 1)
}

/// The NZCV local as a `cpu::LazyFlags` instead of a `cpu::Flags`: its writes `set(..)`, its
/// reads `nzcv()`, the flag operations by the same names.
fn lazy_flags(text: &str) -> String {
    let text = text
        .replace(
            "fl: cpu::Flags { nzcv: m.cpu.nzcv }",
            "fl: cpu::LazyFlags::from(m.cpu.nzcv)",
        )
        .replace("fl: cpu::Flags, ", "fl: cpu::LazyFlags, ");
    const WRITE: &str = "r.fl.nzcv = ";
    let mut output = String::with_capacity(text.len());
    let mut rest = text.as_str();
    while let Some(at) = rest.find(WRITE) {
        output.push_str(&rest[..at]);
        let expression = &rest[at + WRITE.len()..];
        let mut depth = 0i32;
        let end = expression
            .char_indices()
            .find(|&(_, c)| {
                match c {
                    '(' | '[' | '{' => depth += 1,
                    ')' | ']' | '}' => depth -= 1,
                    ';' if depth == 0 => return true,
                    _ => {}
                }
                false
            })
            .map_or(expression.len(), |(end, _)| end);
        output.push_str("r.fl.set(");
        output.push_str(&expression[..end]);
        output.push(')');
        rest = &expression[end..];
    }
    output.push_str(rest);
    output.replace("r.fl.nzcv", "r.fl.nzcv()")
}

/// The distinct blocks of a function and its inlined copies.
fn unique_blocks<'a>(_own: &[&'a BasicBlock], emitted: &[&'a BasicBlock]) -> Vec<&'a BasicBlock> {
    let mut unique: BTreeMap<u64, &BasicBlock> = BTreeMap::new();
    for block in emitted {
        unique.insert(block.start, block);
    }
    unique.into_values().collect()
}

/// Writes the function for its tree; None when a body reads the step
/// count, else the blocks it can be entered at.
#[allow(clippy::too_many_arguments)]
fn emit_tree(
    output: &mut String,
    entry: u64,
    entries: &[u64],
    tree: crate::structure::Structured,
    graph: &BTreeMap<u64, Vec<Option<u64>>>,
    counts: &BTreeMap<u64, u64>,
    calls: &BTreeSet<u64>,
    returns: &BTreeSet<u64>,
    unique: &[&BasicBlock],
    inlined_calls: &BTreeSet<u64>,
) -> Option<Vec<u64>> {
    let mut bodies = bodies(unique, inlined_calls)?;
    // Stack accesses check SP once, at the entries and call returns, where
    // every SP write keeps its alignment.
    let hoisted = sp_hoisted(unique) && bodies.values().any(|body| body.contains("unaligned_sp("));
    if hoisted {
        for body in bodies.values_mut() {
            *body = body
                .lines()
                .filter(|line| {
                    !(line.trim_start().starts_with("if base & 15 != 0 {")
                        && line.contains("unaligned_sp("))
                })
                .map(|line| format!("{line}\n"))
                .collect();
        }
    }
    // Where the stop word is tested: the entry, loop heads and the blocks
    // calls return to; and how far each test looks ahead (no cycle avoids
    // a loop head: the graph is reducible).
    let mut heads: BTreeSet<u64> = entries.iter().copied().collect();
    loop_heads(&tree, &mut heads, false);
    heads.extend(returns.iter().filter(|pc| graph.contains_key(pc)));
    fn ahead(
        key: u64,
        heads: &BTreeSet<u64>,
        counts: &BTreeMap<u64, u64>,
        graph: &BTreeMap<u64, Vec<Option<u64>>>,
        calls: &BTreeSet<u64>,
        memo: &mut BTreeMap<u64, u64>,
    ) -> u64 {
        if let Some(&known) = memo.get(&key) {
            return known;
        }
        let mut longest = 0;
        if !calls.contains(&key) {
            for next in graph[&key].iter().flatten() {
                if counts.contains_key(next) && !heads.contains(next) {
                    longest = longest.max(ahead(*next, heads, counts, graph, calls, memo));
                }
            }
        }
        let total = counts[&key] + longest;
        memo.insert(key, total);
        total
    }
    let mut memo = BTreeMap::new();
    let checks: BTreeMap<u64, u64> = heads
        .iter()
        .map(|&head| (head, ahead(head, &heads, counts, graph, calls, &mut memo)))
        .collect();
    let sp_checks: BTreeSet<u64> = if hoisted {
        entries
            .iter()
            .copied()
            .chain(returns.iter().copied().filter(|pc| graph.contains_key(pc)))
            .collect()
    } else {
        BTreeSet::new()
    };
    let mut emitter = Emitter {
        bodies: &bodies,
        successors: graph,
        checks: &checks,
        sp_checks,
        last_code: entry,
        labels: Vec::new(),
        next_label: 0,
    };
    // One top-level brace: the locals' rewrite then treats the function as
    // one arm with jumps, storing every written local at calls.
    let mut text = String::from("{\n");
    if !emitter.emit(&mut text, &tree) {
        return None;
    }
    text.push_str("}\n");
    let locals = RegisterLocals::of(&text);
    let rewritten = locals.rewrite(&text);
    let rewritten = explicit_errors(&rewritten).unwrap_or(rewritten);
    // With no `?` left every exit is a `return` in the text: each stores
    // only what may have been written before it, so registers written
    // before any exit need no load.
    let precise = !rewritten
        .lines()
        .any(|line| without_strings(line).contains('?'));
    let mut body = if precise {
        exit_stores(&rewritten, &locals, graph, entries)
    } else {
        rewritten
            .lines()
            .filter(|line| !line.trim_start().starts_with("// @"))
            .map(|line| format!("{line}\n"))
            .collect()
    };
    if WINDOW_MEMORY.get() {
        body = body
            .replace("m.memory.load_u", "wmem.load_u")
            .replace("m.memory.store_u", "wmem.store_u");
    }
    let body = lazy_flags(&returned_results(&body));
    let body = if precise && REGALLOC_LAZY_PC.get() {
        lazy_pc(&body)
    } else {
        body
    };
    let mut load = lazy_flags(&locals.load());
    let body = match REGALLOC_VECTOR_LOCALS
        .get()
        .then(|| vector_locals(&body))
        .flatten()
    {
        Some((body, vectors)) if !vectors.is_empty() => {
            load = with_vector_fields(&load, &vectors);
            body
        }
        _ => body,
    };
    writeln!(
        output,
        "\n#[inline(never)]\n#[allow(unused_parens, unused_mut, unused_assignments, unused_variables, unreachable_patterns, unreachable_code, unused_labels)]\npub fn run_function_{entry:x}(m: &mut Machine) -> Result<(), Trap> {{"
    )
    .unwrap();
    output.push_str(&load);
    writeln!(
        output,
        "    #[inline(always)]\n    fn body(m: &mut Machine, r: &mut R, n: &mut u64) -> Result<(), Trap> {{\n    let mut next_pc: u64 = {};",
        if entries.len() > 1 { "m.cpu.pc".to_owned() } else { format!("{entry:#x}") }
    )
    .unwrap();
    if WINDOW_MEMORY.get() {
        output.push_str("    let wmem = m.memory.window_memory();\n");
    }
    output.push_str(&body);
    output.push_str("    #[allow(unreachable_code)]\n    Ok(())\n    }\n");
    // The instructions of a trapping block after the trapping one.
    output.push_str("    #[cold]\n    fn tail(pc: u64) -> u64 {\n        match pc {\n");
    for block in unique {
        let end = block
            .instructions
            .last()
            .expect("blocks have an instruction")
            .pc;
        writeln!(
            output,
            "            {:#x}..={end:#x} => ({end:#x} - pc) / 4,",
            block.start
        )
        .unwrap();
    }
    output.push_str("            _ => 0,\n        }\n    }\n");
    output.push_str(
        "    let mut n = 0u64;\n    let result = body(m, &mut r, &mut n);\n    if n != 0 {\n        m.steps += n;\n        if result.is_err() {\n            m.steps -= tail(m.cpu.pc);\n            m.clear_block_tag();\n        }\n    }\n",
    );
    if !precise {
        output.push_str(&lazy_flags(&locals.store()));
    }
    output.push_str("    result\n}\n");
    Some(entries.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pc_stores_move_into_slow_paths_only_where_nothing_else_sees_the_pc() {
        let text = "m.cpu.pc = 0x100;\n\
            let x = r.x1; if x != 0 { { m.cpu.fpsr = r.fpsr; let called = helper(&mut m.cpu, x); r.fpsr = m.cpu.fpsr; (match called { Ok(value) => value, Err(error) => { return Err(error); } }) }; }\n\
            m.cpu.pc = 0x104;\n\
            let called = run_function_200(m);\n\
            m.cpu.pc = 0x108;\n\
            m.cpu.regs[1] = m.cpu.load_q_pair(&m.memory, 0, 0, 1);\n\
            m.cpu.pc = 0x10c;\n";
        let moved = lazy_pc(text);
        let lines: Vec<&str> = moved.lines().collect();
        // Into the helper call and the error return.
        assert_eq!(
            lines[0],
            "let x = r.x1; if x != 0 { { m.cpu.fpsr = r.fpsr; m.cpu.pc = 0x100; let called = helper(&mut m.cpu, x); r.fpsr = m.cpu.fpsr; (match called { Ok(value) => value, Err(error) => { m.cpu.pc = 0x100; return Err(error); } }) }; }"
        );
        // Not past a call taking the Machine or a CPU method.
        assert_eq!(lines[1], "m.cpu.pc = 0x104;");
        assert_eq!(lines[3], "m.cpu.pc = 0x108;");
        // Not past the last store, with nothing after it to bound the span.
        assert_eq!(lines[5], "m.cpu.pc = 0x10c;");
    }
}
