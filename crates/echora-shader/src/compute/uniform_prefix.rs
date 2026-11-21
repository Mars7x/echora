//! Incoming uniforms and read-only loads from a common address have the
//! same value in every vertex invocation. The straight-line entry prefix
//! can copy those values directly to uniform registers.

use crate::{
    Instruction, Op,
    decode::{Address, Pred, Space, Src},
};

pub(super) fn broadcasts(code: &[Instruction], first_block_end: usize) -> Vec<usize> {
    if !code.iter().any(|i| matches!(i.op, Op::R2ur { .. }))
        || code.iter().any(|i| {
            matches!(
                i.op,
                Op::Brx { .. }
                    | Op::Store { .. }
                    | Op::Atomic { .. }
                    | Op::Surface { store: true, .. }
                    | Op::SurfaceReduce { .. }
            )
        })
    {
        return Vec::new();
    }
    let mut state = Prefix {
        words: [false; 256],
        uniforms: [false; 64],
        predicates: [false; 16],
    };
    let mut broadcasts = Vec::new();
    for (index, instruction) in code[..first_block_end].iter().enumerate() {
        if !instruction.guard.is_true() {
            break;
        }
        let uniform = instruction.uniform;
        let source = |state: &Prefix, src| state.source(src, uniform);
        match instruction.op {
            Op::Mov { d, src, lanes: 15 } => {
                state.set(d.0, uniform, source(&state, src));
            }
            Op::Uldc {
                d,
                src: Src::Const { .. },
                size: size @ 4..=6,
            } => {
                for word in 0..1 << (size - 4) {
                    state.set(d.0.saturating_add(word), true, true);
                }
            }
            Op::Lop3 { d, a, b, c, p, .. } => {
                let same =
                    source(&state, Src::Reg(a)) && source(&state, b.src) && source(&state, c.src);
                state.set(d.0, uniform, same);
                state.set_predicate(p.index, uniform, same);
            }
            Op::Iadd3 {
                d,
                a,
                b,
                c,
                extended: false,
                carry_out,
                ..
            } => {
                let same = source(&state, a.src) && source(&state, b.src) && source(&state, c.src);
                state.set(d.0, uniform, same);
                for pred in carry_out {
                    state.set_predicate(pred.index, uniform, same);
                }
            }
            Op::Isetp {
                p,
                q,
                a,
                b,
                accumulate,
                extended: None,
                ..
            } => {
                let same = source(&state, a.src)
                    && source(&state, b.src)
                    && state.predicate(accumulate, uniform);
                state.set_predicate(p, uniform, same);
                state.set_predicate(q, uniform, same);
            }
            Op::Sel {
                d, a, b, condition, ..
            } => {
                let same = source(&state, a.src)
                    && source(&state, b.src)
                    && state.predicate(condition, uniform);
                state.set(d.0, uniform, same);
            }
            Op::Shf {
                d,
                low,
                high,
                shift,
                ..
            } => {
                let same = source(&state, Src::Reg(low))
                    && source(&state, high.src)
                    && source(&state, shift.src);
                state.set(d.0, uniform, same);
            }
            Op::Ffma { d, .. } => state.set(d.0, uniform, false),
            Op::Load {
                d,
                space: Space::Global,
                address,
                size,
                condition,
                strong: false,
            } if !uniform && condition.is_true() => {
                let words = match size {
                    0..=4 => 1,
                    5 => 2,
                    6 => 4,
                    _ => break,
                };
                let same = state.address(address);
                for word in 0..words {
                    state.set(d.0.saturating_add(word), false, same);
                }
            }
            Op::Ald { d, count, .. } if !uniform => {
                for word in 0..count {
                    state.set(d.0.saturating_add(word), false, false);
                }
            }
            Op::R2ur { d, src } if !uniform => {
                let same = source(&state, Src::Reg(src));
                if same {
                    broadcasts.push(index);
                }
                state.set(d.0, true, same);
            }
            Op::Nop | Op::Depbar | Op::Ldgdepbar | Op::Errbar => {}
            _ => break,
        }
    }
    broadcasts
}

struct Prefix {
    words: [bool; 256],
    uniforms: [bool; 64],
    predicates: [bool; 16],
}

impl Prefix {
    fn source(&self, src: Src, uniform: bool) -> bool {
        match src {
            Src::Imm(_) | Src::Const { .. } => true,
            Src::Uniform(reg) => self.uniform_word(reg.0),
            Src::Reg(reg) if uniform => self.uniform_word(reg.0),
            Src::Reg(reg) => self.word(reg.0),
            Src::BindlessConst { .. } => false,
        }
    }

    fn word(&self, reg: u8) -> bool {
        reg == 255 || self.words[usize::from(reg)]
    }

    fn uniform_word(&self, reg: u8) -> bool {
        reg >= 63 || self.uniforms[usize::from(reg)]
    }

    fn set(&mut self, reg: u8, uniform: bool, same: bool) {
        if uniform {
            if let Some(word) = self.uniforms.get_mut(usize::from(reg)) {
                *word = same;
            }
        } else {
            self.words[usize::from(reg)] = same;
        }
    }

    fn address(&self, address: Address) -> bool {
        self.word(address.reg.0)
            && (!address.wide || self.word(address.reg.0.saturating_add(1)))
            && address.uniform.is_none_or(|(reg, wide)| {
                self.uniform_word(reg.0) && (!wide || self.uniform_word(reg.0.saturating_add(1)))
            })
    }

    fn predicate(&self, pred: Pred, uniform: bool) -> bool {
        pred.index & 7 == 7 || self.predicates[predicate_index(pred.index, uniform)]
    }

    fn set_predicate(&mut self, index: u8, uniform: bool, same: bool) {
        self.predicates[predicate_index(index, uniform)] = same;
    }
}

fn predicate_index(index: u8, uniform: bool) -> usize {
    usize::from(index | if uniform { 8 } else { 0 })
}
