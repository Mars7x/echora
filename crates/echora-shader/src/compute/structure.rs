//! Structured control flow: a program's blocks as nested blocks, loops
//! and if/else, so the driver's compiler sees the real control flow. The
//! construction follows Norman Ramsey, "Beyond Relooper: recursive
//! translation of unstructured control flow to structured control flow"
//! (ICFP 2022): over the dominator tree in reverse postorder, a node's
//! children that several forward edges reach ("merge nodes") follow it
//! inside nested blocks, loop headers wrap their body in a loop, and every
//! branch becomes a block exit, a loop continue or the code of a node it
//! alone reaches.
//!
//! In SPIR-V each block or loop is a loop construct (a block runs once);
//! an exit or continue to the innermost one is a plain branch, and one to
//! an outer construct records its target in a variable and leaves each
//! construct in turn, each merge passing it on until it arrives. Each loop
//! iteration spends the invocation's loop budget.
//!
//! Programs whose graph is irreducible, or that branch to computed
//! addresses, keep the dispatch loop.

use std::collections::BTreeMap;

use echora_shader_abi::Unsupported;

use super::{Block, Emitter, End, Guard, LOOP_BUDGET};
use crate::decode::Instruction;
use crate::spirv::{self, op};

/// A structured statement.
#[derive(Debug)]
enum Statement {
    /// Block `n`'s instructions (without its terminator).
    Code(usize),
    Sequence(Vec<Statement>),
    If {
        condition: Guard,
        then: Box<Statement>,
        otherwise: Box<Statement>,
    },
    /// Construct `id`, run once; an exit from it continues after it.
    Block {
        id: usize,
        body: Box<Statement>,
    },
    /// Construct `id`, repeated; a continue to it runs it again.
    Loop {
        id: usize,
        body: Box<Statement>,
    },
    Exit(usize),
    Continue(usize),
    Return,
}

/// The graph of a program's blocks.
struct Graph {
    /// Reverse postorder numbers (`usize::MAX`: unreachable).
    rpo: Vec<usize>,
    loop_header: Vec<bool>,
    merge: Vec<bool>,
    /// Dominator-tree children, by reverse postorder.
    children: Vec<Vec<usize>>,
}

/// A frame of the context a branch is translated in.
#[derive(Clone, Copy, Debug)]
enum Frame {
    LoopHeadedBy(usize, usize),
    BlockFollowedBy(usize, usize),
}

impl Graph {
    /// The graph, or `None` for an irreducible one or one with computed
    /// branches.
    fn new(blocks: &[Block]) -> Option<Self> {
        let index_of: BTreeMap<usize, usize> = blocks
            .iter()
            .enumerate()
            .map(|(n, block)| (block.start, n))
            .collect();
        let count = blocks.len();
        let mut successors = vec![Vec::new(); count];
        for (n, block) in blocks.iter().enumerate() {
            let next = (n + 1 < count).then_some(n + 1);
            successors[n] = match block.end {
                End::Fall => next.into_iter().collect(),
                End::Branch { target, condition } => {
                    let target = *index_of.get(&target)?;
                    if condition.always() {
                        vec![target]
                    } else {
                        let mut list = vec![target];
                        list.extend(next.filter(|&next| next != target));
                        list
                    }
                }
                End::Exit { condition } => {
                    if condition.always() {
                        Vec::new()
                    } else {
                        next.into_iter().collect()
                    }
                }
                End::Indirect { .. } => return None,
            };
        }
        // Reverse postorder from the entry.
        let mut postorder = Vec::new();
        let mut seen = vec![false; count];
        let mut stack = vec![(0usize, 0usize)];
        seen[0] = true;
        while let Some(&mut (node, ref mut next)) = stack.last_mut() {
            if let Some(&successor) = successors[node].get(*next) {
                *next += 1;
                if !seen[successor] {
                    seen[successor] = true;
                    stack.push((successor, 0));
                }
            } else {
                postorder.push(node);
                stack.pop();
            }
        }
        let mut rpo = vec![usize::MAX; count];
        let order: Vec<usize> = postorder.iter().rev().copied().collect();
        for (number, &node) in order.iter().enumerate() {
            rpo[node] = number;
        }
        // Predecessors and immediate dominators (Cooper, Harvey and
        // Kennedy's iteration).
        let mut predecessors = vec![Vec::new(); count];
        for &node in &order {
            for &successor in &successors[node] {
                predecessors[successor].push(node);
            }
        }
        let mut idom = vec![usize::MAX; count];
        idom[0] = 0;
        let mut changed = true;
        while changed {
            changed = false;
            for &node in order.iter().skip(1) {
                let mut new = usize::MAX;
                for &p in &predecessors[node] {
                    if idom[p] == usize::MAX {
                        continue;
                    }
                    new = if new == usize::MAX {
                        p
                    } else {
                        let (mut a, mut b) = (p, new);
                        while a != b {
                            while rpo[a] > rpo[b] {
                                a = idom[a];
                            }
                            while rpo[b] > rpo[a] {
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
        let dominates = |a: usize, mut b: usize| loop {
            if a == b {
                return true;
            }
            if b == 0 {
                return false;
            }
            b = idom[b];
        };
        let mut loop_header = vec![false; count];
        let mut forward_in = vec![0usize; count];
        for &node in &order {
            for &successor in &successors[node] {
                if rpo[successor] <= rpo[node] {
                    // A retreating edge must be a back edge.
                    if !dominates(successor, node) {
                        return None;
                    }
                    loop_header[successor] = true;
                } else {
                    forward_in[successor] += 1;
                }
            }
        }
        let merge: Vec<bool> = forward_in.iter().map(|&n| n >= 2).collect();
        let mut children = vec![Vec::new(); count];
        for &node in order.iter().skip(1) {
            children[idom[node]].push(node);
        }
        Some(Self {
            rpo,
            loop_header,
            merge,
            children,
        })
    }
}

/// Translates a graph into statements.
struct Builder<'a> {
    graph: &'a Graph,
    blocks: &'a [Block],
    index_of: BTreeMap<usize, usize>,
    context: Vec<Frame>,
    constructs: usize,
}

impl Builder<'_> {
    fn construct(&mut self) -> usize {
        self.constructs += 1;
        self.constructs - 1
    }

    fn tree(&mut self, x: usize) -> Statement {
        let mut merges: Vec<usize> = self.graph.children[x]
            .iter()
            .copied()
            .filter(|&child| self.graph.merge[child])
            .collect();
        merges.sort_by_key(|&child| std::cmp::Reverse(self.graph.rpo[child]));
        if self.graph.loop_header[x] {
            let id = self.construct();
            self.context.push(Frame::LoopHeadedBy(x, id));
            let body = self.within(x, &merges);
            self.context.pop();
            Statement::Loop {
                id,
                body: Box::new(body),
            }
        } else {
            self.within(x, &merges)
        }
    }

    fn within(&mut self, x: usize, merges: &[usize]) -> Statement {
        match merges.split_first() {
            None => Statement::Sequence(vec![Statement::Code(x), self.terminator(x)]),
            Some((&follower, rest)) => {
                let id = self.construct();
                self.context.push(Frame::BlockFollowedBy(follower, id));
                let body = self.within(x, rest);
                self.context.pop();
                Statement::Sequence(vec![
                    Statement::Block {
                        id,
                        body: Box::new(body),
                    },
                    self.tree(follower),
                ])
            }
        }
    }

    fn branch(&mut self, x: usize, y: Option<usize>) -> Statement {
        let Some(y) = y else {
            return Statement::Return;
        };
        if self.graph.rpo[y] <= self.graph.rpo[x] {
            let id = self.context.iter().rev().find_map(|frame| match *frame {
                Frame::LoopHeadedBy(header, id) if header == y => Some(id),
                _ => None,
            });
            Statement::Continue(id.unwrap_or(usize::MAX))
        } else if self.graph.merge[y] {
            let id = self.context.iter().rev().find_map(|frame| match *frame {
                Frame::BlockFollowedBy(follower, id) if follower == y => Some(id),
                _ => None,
            });
            Statement::Exit(id.unwrap_or(usize::MAX))
        } else {
            self.tree(y)
        }
    }

    fn terminator(&mut self, x: usize) -> Statement {
        let next = (x + 1 < self.blocks.len()).then_some(x + 1);
        match self.blocks[x].end {
            End::Fall => self.branch(x, next),
            End::Branch { target, condition } => {
                let target = self.index_of[&target];
                if condition.always() {
                    self.branch(x, Some(target))
                } else {
                    Statement::If {
                        condition,
                        then: Box::new(self.branch(x, Some(target))),
                        otherwise: Box::new(self.branch(x, next)),
                    }
                }
            }
            End::Exit { condition } => {
                if condition.always() {
                    Statement::Return
                } else {
                    Statement::If {
                        condition,
                        then: Box::new(Statement::Return),
                        otherwise: Box::new(self.branch(x, next)),
                    }
                }
            }
            End::Indirect { .. } => Statement::Return,
        }
    }
}

/// An open construct while emitting.
#[derive(Clone, Copy, Debug)]
struct Open {
    id: usize,
    merge: u32,
    continuing: u32,
    /// Whether an exit or continue to an outer construct leaves through
    /// this construct's merge.
    escapes: bool,
}

/// Emission state.
struct Structured {
    open: Vec<Open>,
    /// The pending target: 0 none, 2 id + 1 an exit, 2 id + 2 a continue.
    pending: u32,
    budget: u32,
}

fn exit_code(id: usize) -> u32 {
    2 * id as u32 + 1
}

fn continue_code(id: usize) -> u32 {
    2 * id as u32 + 2
}

impl Emitter {
    /// The structured translation of `blocks`, or `false` (nothing
    /// emitted) when their graph needs the dispatch loop.
    pub(super) fn structured(
        &mut self,
        code: &[Instruction],
        blocks: &[Block],
    ) -> Result<bool, Unsupported> {
        let Some(graph) = Graph::new(blocks) else {
            return Ok(false);
        };
        let mut builder = Builder {
            graph: &graph,
            blocks,
            index_of: blocks
                .iter()
                .enumerate()
                .map(|(n, block)| (block.start, n))
                .collect(),
            context: Vec::new(),
            constructs: 0,
        };
        let root = builder.tree(0);
        self.vertex_prologue(code);
        let zero = self.constant(0);
        let pending = self
            .function
            .variable(&mut self.module, self.u32_ptr, Some(zero));
        let budget_value = self.constant(LOOP_BUDGET);
        let budget = self
            .function
            .variable(&mut self.module, self.u32_ptr, Some(budget_value));
        let mut state = Structured {
            open: Vec::new(),
            pending,
            budget,
        };
        self.statement(code, blocks, &root, &mut state)?;
        Ok(true)
    }

    fn statement(
        &mut self,
        code: &[Instruction],
        blocks: &[Block],
        statement: &Statement,
        state: &mut Structured,
    ) -> Result<(), Unsupported> {
        match statement {
            Statement::Code(n) => {
                let block = &blocks[*n];
                for (index, instruction) in code
                    .iter()
                    .enumerate()
                    .take(block.body_end)
                    .skip(block.start)
                {
                    self.instruction(index, instruction)?;
                }
            }
            Statement::Sequence(statements) => {
                for statement in statements {
                    self.statement(code, blocks, statement, state)?;
                }
            }
            Statement::If {
                condition,
                then,
                otherwise,
            } => {
                let value = self.condition(*condition);
                let (taken, other, merge) = (self.module.id(), self.module.id(), self.module.id());
                self.function
                    .emit(op::SELECTION_MERGE, &[merge, spirv::NONE]);
                self.function
                    .emit(op::BRANCH_CONDITIONAL, &[value, taken, other]);
                self.function.label(taken);
                self.statement(code, blocks, then, state)?;
                self.function.label(other);
                self.statement(code, blocks, otherwise, state)?;
                // Both arms leave: the merge is never reached.
                self.function.label(merge);
                self.function.emit(op::UNREACHABLE, &[]);
            }
            Statement::Block { id, body } | Statement::Loop { id, body } => {
                let repeats = matches!(statement, Statement::Loop { .. });
                let (header, start, continuing, merge) = (
                    self.module.id(),
                    self.module.id(),
                    self.module.id(),
                    self.module.id(),
                );
                self.function.emit(op::BRANCH, &[header]);
                self.function.label(header);
                self.function
                    .emit(op::LOOP_MERGE, &[merge, continuing, spirv::NONE]);
                self.function.emit(op::BRANCH, &[start]);
                self.function.label(start);
                if repeats {
                    // Each iteration spends the budget; none left stops.
                    let left = self.value(op::LOAD, self.u32, &[state.budget]);
                    let zero = self.constant(0);
                    let spent = self.value(op::I_EQUAL, self.bool, &[left, zero]);
                    let (stop, go) = (self.module.id(), self.module.id());
                    self.function.emit(op::SELECTION_MERGE, &[go, spirv::NONE]);
                    self.function
                        .emit(op::BRANCH_CONDITIONAL, &[spent, stop, go]);
                    self.function.label(stop);
                    self.leave();
                    self.function.label(go);
                }
                state.open.push(Open {
                    id: *id,
                    merge,
                    continuing,
                    escapes: false,
                });
                self.statement(code, blocks, body, state)?;
                let closed = state.open.pop().unwrap_or(Open {
                    id: *id,
                    merge,
                    continuing,
                    escapes: false,
                });
                self.function.label(continuing);
                if repeats {
                    let left = self.value(op::LOAD, self.u32, &[state.budget]);
                    let one = self.constant(1);
                    let less = self.value(op::I_SUB, self.u32, &[left, one]);
                    self.function.emit(op::STORE, &[state.budget, less]);
                }
                self.function.emit(op::BRANCH, &[header]);
                self.function.label(merge);
                if closed.escapes {
                    self.pass_on(state);
                }
                if repeats {
                    // A loop is left only by passing on: nothing follows.
                    self.function.emit(op::UNREACHABLE, &[]);
                }
            }
            Statement::Exit(id) | Statement::Continue(id) => {
                let exit = matches!(statement, Statement::Exit(_));
                let Some(innermost) = state.open.last().copied() else {
                    self.function.emit(op::UNREACHABLE, &[]);
                    return Ok(());
                };
                if innermost.id == *id {
                    let target = if exit {
                        innermost.merge
                    } else {
                        innermost.continuing
                    };
                    self.function.emit(op::BRANCH, &[target]);
                } else {
                    let code = if exit {
                        exit_code(*id)
                    } else {
                        continue_code(*id)
                    };
                    let code = self.constant(code);
                    self.function.emit(op::STORE, &[state.pending, code]);
                    for open in state.open.iter_mut().rev() {
                        if open.id == *id {
                            break;
                        }
                        open.escapes = true;
                    }
                    self.function.emit(op::BRANCH, &[innermost.merge]);
                }
            }
            Statement::Return => self.leave(),
        }
        Ok(())
    }

    /// The invocation's end: the fragment outputs, then return.
    fn leave(&mut self) {
        self.fragment_outputs();
        self.function.emit(op::RETURN, &[]);
    }

    /// At a construct's merge: a pending exit or continue to the enclosing
    /// construct is taken (and cleared), any other is passed on through
    /// its merge.
    fn pass_on(&mut self, state: &Structured) {
        let Some(outer) = state.open.last().copied() else {
            return;
        };
        let value = self.value(op::LOAD, self.u32, &[state.pending]);
        let zero = self.constant(0);
        let some = self.value(op::I_NOT_EQUAL, self.bool, &[value, zero]);
        let (passing, rest) = (self.module.id(), self.module.id());
        self.function
            .emit(op::SELECTION_MERGE, &[rest, spirv::NONE]);
        self.function
            .emit(op::BRANCH_CONDITIONAL, &[some, passing, rest]);
        self.function.label(passing);
        let exit = self.constant(exit_code(outer.id));
        let again = self.constant(continue_code(outer.id));
        let is_exit = self.value(op::I_EQUAL, self.bool, &[value, exit]);
        let is_continue = self.value(op::I_EQUAL, self.bool, &[value, again]);
        let arrived = self.value(op::LOGICAL_OR, self.bool, &[is_exit, is_continue]);
        let left = self.value(op::SELECT, self.u32, &[arrived, zero, value]);
        self.function.emit(op::STORE, &[state.pending, left]);
        self.function.emit(
            op::BRANCH_CONDITIONAL,
            &[is_continue, outer.continuing, outer.merge],
        );
        self.function.label(rest);
    }
}
