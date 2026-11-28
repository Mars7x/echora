//! Give each call site its own copy of a small leaf routine, so the
//! return becomes a direct branch and the shader can use structured
//! control flow. Keep the original dispatcher when a return is unclear.

use crate::decode::{Instruction, Op, Pred, Reg, Src};

use super::relative_target;

// Bound the additional code when many sites call the same routine.
const MAX_INLINED_INSTRUCTIONS: usize = 256;

// How far before a call its link assignment may be.
const MAX_LINK_DISTANCE: usize = 8;

pub(super) fn inline_leaf(code: &[Instruction]) -> Option<Vec<Instruction>> {
    // Each routine's entry and the calls to it.
    let mut routines: Vec<(usize, Vec<usize>)> = Vec::new();
    let mut returns = Vec::new();
    for (index, instruction) in code.iter().enumerate() {
        match instruction.op {
            Op::Call { offset } => {
                let start = relative_target(index, offset, code.len())?;
                match routines.iter_mut().find(|(entry, _)| *entry == start) {
                    Some((_, callers)) => callers.push(index),
                    None => routines.push((start, vec![index])),
                }
            }
            Op::Ret { target } if instruction.guard.is_true() && !instruction.uniform => {
                returns.push((index, target));
            }
            Op::Ret { .. } | Op::Brx { .. } => return None,
            _ => {}
        }
    }
    if routines.is_empty() {
        return None;
    }
    let callers: Vec<usize> = routines
        .iter()
        .flat_map(|(_, callers)| callers.iter().copied())
        .collect();
    // Each routine returns at the first RET after its entry, and no two
    // share one. Any other RET must be unreachable: compilers can leave
    // copies of a routine no call uses.
    // Instructions that a branch or call enters.
    let mut entered = vec![false; code.len()];
    for (index, &instruction) in code.iter().enumerate() {
        match instruction.op {
            Op::Bra { offset, .. } | Op::Call { offset } => {
                entered[relative_target(index, offset, code.len())?] = true;
            }
            _ => {}
        }
    }
    let mut leaves = Vec::new();
    for (start, callers) in &routines {
        let &(end, ret) = returns.iter().find(|&&(index, _)| index >= *start)?;
        if leaves.iter().any(|leaf: &Leaf| leaf.end == end) {
            return None;
        }
        leaves.push(Leaf::new(code, *start, end, ret, callers, &entered)?);
    }
    let inlined: usize = leaves
        .iter()
        .map(|leaf| leaf.callers.len() * (leaf.end - leaf.start + 1))
        .sum();
    if inlined > MAX_INLINED_INSTRUCTIONS {
        return None;
    }
    for (index, &instruction) in code.iter().enumerate() {
        if let Op::Bra { offset, .. } = instruction.op {
            let dest = relative_target(index, offset, code.len())?;
            // No other control edge may enter a leaf, the middle of its
            // return sequence or a call.
            if callers.contains(&dest)
                || leaves.iter().any(|leaf| {
                    let inside = |at: usize| (leaf.start..=leaf.end).contains(&at);
                    inside(index) != inside(dest)
                        || (leaf.copy_index + 1..=leaf.end).contains(&dest)
                })
            {
                return None;
            }
        }
    }
    let dead_returns: Vec<usize> = returns
        .iter()
        .map(|&(index, _)| index)
        .filter(|&index| leaves.iter().all(|leaf| leaf.end != index))
        .collect();
    // Straight-line code back to an unconditional end, entered by no
    // branch or call.
    let unreachable = |mut index: usize| loop {
        if entered[index] || index == 0 {
            return false;
        }
        let previous = code[index - 1];
        if ends_linear_path(previous) {
            return true;
        }
        if !matches!(previous.op, Op::Mov { .. }) {
            return false;
        }
        index -= 1;
    };
    if !dead_returns.iter().all(|&index| unreachable(index)) {
        return None;
    }

    // Relative branches inside the copy still refer to its own
    // instructions. The old routines are no longer reachable.
    let mut result = code.to_vec();
    let exit = Op::Exit {
        condition: Pred::TRUE,
    };
    for index in leaves.iter().map(|leaf| leaf.end).chain(dead_returns) {
        result[index].op = exit;
    }
    if !ends_linear_path(*result.last()?) {
        result.push(Instruction {
            raw: 0,
            guard: Pred::TRUE,
            op: exit,
            uniform: false,
        });
    }
    for leaf in &leaves {
        for &caller in &leaf.callers {
            let new_entry = result.len();
            result.extend_from_slice(&code[leaf.start..=leaf.end]);
            let new_end = result.len() - 1;
            result[caller].op = branch(caller, new_entry);
            result[new_end].op = branch(new_end, caller + 1);
        }
    }
    Some(result)
}

/// A leaf routine from `start` to its RET at `end`.
struct Leaf {
    start: usize,
    end: usize,
    /// No branch may enter the return sequence after this: between the
    /// link copy and RET, or (without a copy) RET itself.
    copy_index: usize,
    callers: Vec<usize>,
}

impl Leaf {
    /// The routine when every call reaches its RET with the caller's own
    /// return address in the RET's register.
    fn new(
        code: &[Instruction],
        start: usize,
        end: usize,
        ret: Reg,
        callers: &[usize],
        entered: &[bool],
    ) -> Option<Self> {
        if start == 0
            || start >= end
            || !ends_linear_path(code[start - 1])
            || callers
                .iter()
                .any(|&caller| (start..=end).contains(&caller))
        {
            return None;
        }
        // RET reads the caller's link through an ordinary register copy,
        // or straight from the register each caller sets just before its
        // CALL. Some routines clear the adjacent word immediately before
        // RET.
        let in_routine = |index: usize| (start..end).contains(&index);
        let plain = |index: usize| code[index].guard.is_true() && !code[index].uniform;
        let clears_high = in_routine(end - 1)
            && plain(end - 1)
            && matches!(code[end - 1].op, Op::Mov {
                d,
                src: Src::Imm(0),
                lanes: 15,
            } if d.0 == ret.0.saturating_add(1));
        let before_return = end - 1 - usize::from(clears_high);
        let (link, copy_index) = match code[before_return].op {
            Op::Mov {
                d,
                src: Src::Reg(link),
                lanes: 15,
            } if d == ret && in_routine(before_return) && plain(before_return) => {
                (link, before_return)
            }
            _ => (ret, end - 1),
        };
        if link == Reg::ZERO || ret == Reg::ZERO {
            return None;
        }
        // Every path to a call must execute the link assignment: shortly
        // before the call in straight-line code that no branch enters and
        // that leaves the link alone.
        for &call in callers {
            let set_link = Op::Mov {
                d: link,
                src: Src::Imm(((call + 1) * 16) as u32),
                lanes: 15,
            };
            let mut index = call;
            loop {
                if call - index > MAX_LINK_DISTANCE || index == 0 {
                    return None;
                }
                index -= 1;
                let instruction = code[index];
                if instruction.op == set_link && instruction.guard.is_true() && !instruction.uniform
                {
                    break;
                }
                if entered[index] || may_write(instruction, link)? {
                    return None;
                }
            }
        }
        // The link register stays unchanged inside.
        for &instruction in &code[start..end] {
            if may_write(instruction, link)? {
                return None;
            }
        }
        Some(Self {
            start,
            end,
            copy_index,
            callers: callers.to_vec(),
        })
    }
}

fn branch(from: usize, to: usize) -> Op {
    Op::Bra {
        offset: (to as i64 - from as i64 - 1) * 16,
        condition: Pred::TRUE,
        mode: 0,
    }
}

fn ends_linear_path(instruction: Instruction) -> bool {
    instruction.guard.is_true()
        && match instruction.op {
            Op::Ret { .. } => true,
            Op::Bra { condition, .. } | Op::Exit { condition } => condition.is_true(),
            _ => false,
        }
}

fn may_write(instruction: Instruction, link: Reg) -> Option<bool> {
    if instruction.uniform {
        return None;
    }
    let destination = match instruction.op {
        Op::Mov { d, .. }
        | Op::Iadd3 { d, .. }
        | Op::Imad { d, .. }
        | Op::Lop3 { d, .. }
        | Op::Fset { d, .. }
        | Op::Fadd { d, .. }
        | Op::Ffma { d, .. }
        | Op::Fmul { d, .. }
        | Op::Fmnmx { d, .. }
        | Op::Imnmx { d, .. }
        | Op::Mufu { d, .. }
        | Op::Sel { d, .. }
        | Op::P2r { d, .. }
        | Op::Shf { d, .. } => d,
        Op::Fsetp { .. }
        | Op::Isetp { .. }
        | Op::Plop3 { .. }
        | Op::Bra { .. }
        | Op::Exit { .. }
        | Op::Bssy
        | Op::Bsync
        | Op::Depbar => return Some(false),
        _ => return None,
    };
    Some(destination == link)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::{Layout, blocks, emit_blocks, translate_with_registers};
    use crate::decode::{Combine, IntCompare, Operand};
    use crate::sim::Machine;
    use std::collections::HashMap;

    fn word(op: Op) -> Instruction {
        Instruction {
            raw: 0,
            guard: Pred::TRUE,
            op,
            uniform: false,
        }
    }

    fn mov(d: u8, src: Src) -> Op {
        Op::Mov {
            d: Reg(d),
            src,
            lanes: 15,
        }
    }

    fn caller(from: usize, to: usize) -> Op {
        Op::Call {
            offset: (to as i64 - from as i64 - 1) * 16,
        }
    }

    fn operand(src: Src) -> Operand {
        Operand {
            src,
            abs: false,
            neg: false,
        }
    }

    fn program() -> Vec<Instruction> {
        let mut code: Vec<_> = [
            Op::S2r {
                d: Reg(1),
                special: 33,
            },
            Op::Isetp {
                p: 0,
                q: 7,
                a: operand(Src::Reg(Reg(1))),
                b: operand(Src::Imm(4)),
                compare: IntCompare::Lt,
                signed: false,
                combine: Combine::And,
                accumulate: Pred::TRUE,
                extended: None,
            },
            mov(0, Src::Imm(4)),
            mov(10, Src::Imm(5 * 16)),
            caller(4, 10),
            mov(10, Src::Imm(7 * 16)),
            caller(6, 10),
            branch(7, 9),
            Op::Exit {
                condition: Pred::TRUE,
            },
            Op::Exit {
                condition: Pred::TRUE,
            },
            branch(10, 12),
            Op::Iadd3 {
                d: Reg(0),
                a: operand(Src::Reg(Reg(0))),
                b: operand(Src::Imm(1)),
                c: operand(Src::Reg(Reg::ZERO)),
                carry_out: [Pred::TRUE; 2],
                carry_in: [Pred::TRUE; 2],
                extended: false,
            },
            mov(6, Src::Reg(Reg(10))),
            Op::Ret { target: Reg(6) },
        ]
        .into_iter()
        .map(word)
        .collect();
        code[4].guard = Pred {
            index: 0,
            negated: false,
        };
        code[10].guard = Pred {
            index: 0,
            negated: true,
        };
        code
    }

    /// The program with a routine that returns through the register its
    /// callers set (`MOV R10, link; CALL`), clearing the high word first.
    fn direct_program() -> Vec<Instruction> {
        let mut code = program();
        code[12] = word(mov(11, Src::Imm(0)));
        code[13] = word(Op::Ret { target: Reg(10) });
        code
    }

    /// Two routines returning through the register their callers set, the
    /// second's link assignment two instructions before its call, and an
    /// unreachable copy of a routine.
    fn two_routine_program() -> Vec<Instruction> {
        let add = |n| Op::Iadd3 {
            d: Reg(0),
            a: operand(Src::Reg(Reg(0))),
            b: operand(Src::Imm(n)),
            c: operand(Src::Reg(Reg::ZERO)),
            carry_out: [Pred::TRUE; 2],
            carry_in: [Pred::TRUE; 2],
            extended: false,
        };
        let exit = Op::Exit {
            condition: Pred::TRUE,
        };
        let mut code: Vec<_> = [
            Op::S2r {
                d: Reg(1),
                special: 33,
            },
            Op::Isetp {
                p: 0,
                q: 7,
                a: operand(Src::Reg(Reg(1))),
                b: operand(Src::Imm(4)),
                compare: IntCompare::Lt,
                signed: false,
                combine: Combine::And,
                accumulate: Pred::TRUE,
                extended: None,
            },
            mov(0, Src::Imm(4)),
            mov(10, Src::Imm(5 * 16)),
            caller(4, 11),
            mov(10, Src::Imm(8 * 16)),
            mov(2, Src::Imm(0)),
            caller(7, 14),
            branch(8, 10),
            exit,
            exit,
            add(1),
            mov(11, Src::Imm(0)),
            Op::Ret { target: Reg(10) },
            add(2),
            mov(11, Src::Imm(0)),
            Op::Ret { target: Reg(10) },
            mov(11, Src::Imm(0)),
            Op::Ret { target: Reg(10) },
        ]
        .into_iter()
        .map(word)
        .collect();
        code[4].guard = Pred {
            index: 0,
            negated: false,
        };
        code
    }

    #[test]
    fn separate_inlined_returns_match_the_register_link() {
        check_inlined_returns(program(), 22, [6, 6, 6, 6, 4, 4, 4, 4]);
    }

    #[test]
    fn returns_through_the_callers_register_are_inlined() {
        check_inlined_returns(direct_program(), 22, [6, 6, 6, 6, 4, 4, 4, 4]);
    }

    #[test]
    fn several_routines_with_scheduled_links_are_inlined() {
        check_inlined_returns(two_routine_program(), 25, [7, 7, 7, 7, 6, 6, 6, 6]);
        // A write to the link between its assignment and the call, or a
        // branch into the gap, keeps the dispatcher.
        let mut code = two_routine_program();
        code[6] = word(mov(10, Src::Imm(0)));
        assert!(inline_leaf(&code).is_none());
        let mut code = two_routine_program();
        code[8] = word(branch(8, 6));
        assert!(inline_leaf(&code).is_none());
        // So does a second copy that a branch reaches.
        let mut code = two_routine_program();
        code[8] = word(branch(8, 17));
        assert!(inline_leaf(&code).is_none());
    }

    fn check_inlined_returns(original: Vec<Instruction>, length: usize, expected: [u64; 8]) {
        let expanded = inline_leaf(&original).unwrap();
        assert_eq!(expanded.len(), length);
        let layout = Layout::workgroup([8, 1, 1]);
        let emitter = emit_blocks(&original, &layout, &blocks(&original).unwrap(), None).unwrap();
        let original_register = emitter.registers[&0];
        let original_spirv = emitter.finish().spirv;
        let (translated, registers, _) = translate_with_registers(&original, &layout).unwrap();
        let run = |spirv: &[u32], register| {
            if let Some(problem) = crate::spirv::tests::validate(spirv) {
                panic!("{problem}");
            }
            let report = Machine::new(spirv).run_workgroup(8, 32, |n| {
                HashMap::from([(27, vec![n as u64, 0, 0]), (41, vec![n as u64])])
            });
            assert!(report.problems.is_empty(), "{:?}", report.problems);
            report
                .invocations
                .iter()
                .map(|lane| lane[&register][0])
                .collect::<Vec<_>>()
        };
        assert_eq!(run(&original_spirv, original_register), expected);
        assert_eq!(run(&translated.spirv, registers[&0]), expected);
        let has_switch = |spirv: &[u32]| {
            let mut words = &spirv[5..];
            while let Some(&header) = words.first() {
                if header as u16 == crate::spirv::op::SWITCH {
                    return true;
                }
                words = &words[(header >> 16) as usize..];
            }
            false
        };
        assert!(has_switch(&original_spirv));
        assert!(!has_switch(&translated.spirv));
    }

    #[test]
    fn direct_returns_keep_the_dispatcher_when_the_register_changes() {
        // The routine writes the return register.
        let mut code = direct_program();
        code[11] = word(mov(10, Src::Imm(0)));
        assert!(inline_leaf(&code).is_none());
        // A caller does not set it just before calling.
        let mut code = direct_program();
        code[5] = word(mov(9, Src::Imm(7 * 16)));
        assert!(inline_leaf(&code).is_none());
    }

    #[test]
    fn impossible_call_targets_and_modified_links_keep_the_dispatcher() {
        let mut code = program();
        code[11] = word(mov(10, Src::Imm(0)));
        assert!(inline_leaf(&code).is_none());
        let mut code = program();
        code[8] = word(branch(8, 4));
        assert!(inline_leaf(&code).is_none());
    }
}
