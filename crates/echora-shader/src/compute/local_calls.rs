//! Give each call site its own copy of a small leaf routine, so the
//! return becomes a direct branch and the shader can use structured
//! control flow. Keep the original dispatcher when a return is unclear.

use crate::decode::{Instruction, Op, Pred, Reg, Src};

use super::relative_target;

// Bound the additional code when many sites call the same routine.
const MAX_INLINED_INSTRUCTIONS: usize = 256;

pub(super) fn inline_leaf(code: &[Instruction]) -> Option<Vec<Instruction>> {
    let mut entry = None;
    let mut callers = Vec::new();
    let mut returns = None;
    for (index, instruction) in code.iter().enumerate() {
        match instruction.op {
            Op::Call { offset } => {
                let start = relative_target(index, offset, code.len())?;
                if entry.is_some_and(|known| known != start) {
                    return None;
                }
                entry = Some(start);
                callers.push(index);
            }
            Op::Ret { target } if instruction.guard.is_true() && !instruction.uniform => {
                if returns.replace((index, target)).is_some() {
                    return None;
                }
            }
            Op::Ret { .. } | Op::Brx { .. } => return None,
            _ => {}
        }
    }
    let start = entry?;
    let (end, ret) = returns?;
    if start == 0
        || start >= end
        || !ends_linear_path(code[start - 1])
        || callers
            .iter()
            .any(|&caller| (start..=end).contains(&caller))
    {
        return None;
    }
    let leaf_len = end - start + 1;
    if callers.len().checked_mul(leaf_len)? > MAX_INLINED_INSTRUCTIONS {
        return None;
    }
    // RET reads the caller's link through an ordinary register copy.
    // Some routines clear the adjacent word immediately before RET.
    let last = &code[end - 1];
    let (copy, copy_index) = match last.op {
        Op::Mov {
            d,
            src: Src::Imm(0),
            lanes: 15,
        } if d.0 == ret.0.saturating_add(1) && last.guard.is_true() && !last.uniform => {
            (code.get(end.checked_sub(2)?)?, end - 2)
        }
        _ => (last, end - 1),
    };
    let Op::Mov {
        d,
        src: Src::Reg(link),
        lanes: 15,
    } = copy.op
    else {
        return None;
    };
    if d != ret
        || copy_index < start
        || !copy.guard.is_true()
        || copy.uniform
        || link == Reg::ZERO
        || ret == Reg::ZERO
    {
        return None;
    }
    // Every path to a call must execute the immediate link assignment.
    for &call in &callers {
        let set_link = code.get(call.checked_sub(1)?)?;
        if !set_link.guard.is_true()
            || set_link.uniform
            || set_link.op
                != (Op::Mov {
                    d: link,
                    src: Src::Imm(((call + 1) * 16) as u32),
                    lanes: 15,
                })
        {
            return None;
        }
    }
    // No other control edge may enter the leaf or the middle of
    // the return sequence. The link register stays unchanged inside.
    for (index, &instruction) in code.iter().enumerate() {
        if let Op::Bra { offset, .. } = instruction.op {
            let dest = relative_target(index, offset, code.len())?;
            let inside = (start..=end).contains(&index);
            if inside != (start..=end).contains(&dest)
                || (copy_index + 1..=end).contains(&dest)
                || callers.contains(&dest)
            {
                return None;
            }
        }
        if (start..end).contains(&index) && may_write(instruction, link)? {
            return None;
        }
    }

    // Relative branches inside the copy still refer to its own
    // instructions. The old routine is no longer reachable.
    let mut result = code.to_vec();
    let exit = Op::Exit {
        condition: Pred::TRUE,
    };
    result[end].op = exit;
    if !ends_linear_path(*result.last()?) {
        result.push(Instruction {
            raw: 0,
            guard: Pred::TRUE,
            op: exit,
            uniform: false,
        });
    }
    for caller in callers {
        let new_entry = result.len();
        result.extend_from_slice(&code[start..=end]);
        let new_end = result.len() - 1;
        result[caller].op = branch(caller, new_entry);
        result[new_end].op = branch(new_end, caller + 1);
    }
    Some(result)
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

    #[test]
    fn separate_inlined_returns_match_the_register_link() {
        let original = program();
        let expanded = inline_leaf(&original).unwrap();
        assert_eq!(expanded.len(), 22);
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
        let expected = [6, 6, 6, 6, 4, 4, 4, 4];
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
    fn impossible_call_targets_and_modified_links_keep_the_dispatcher() {
        let mut code = program();
        code[11] = word(mov(10, Src::Imm(0)));
        assert!(inline_leaf(&code).is_none());
        let mut code = program();
        code[8] = word(branch(8, 4));
        assert!(inline_leaf(&code).is_none());
    }
}
