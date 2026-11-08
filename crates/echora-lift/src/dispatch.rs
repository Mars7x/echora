//! An audit of a translation's indirect branches: how each BR and BLR gets
//! its target, and which look like jump-table dispatches that discovery did
//! not recover.
//!
//! A computed dispatch whose table was not found is a MissingBlock waiting
//! to happen: the run stops when the guest first takes a case no other root
//! reached. [`audit`] finds those ahead of time. For each indirect branch it
//! follows the target register back to the instruction that last wrote it
//! (through register copies) and classifies the target:
//!
//! - [`Target::JumpTable`], a recovered table (`Translation::jump_tables`);
//! - [`Target::Return`], RET;
//! - [`Target::Loaded`], a pointer loaded from memory (a vtable, a table
//!   of function pointers, a callback in a structure), whose targets come
//!   from relocation and code-address roots, not from the branch;
//! - [`Target::Computed`], an `add` of a shifted or extended register, the
//!   shape of a table dispatch, without a recovered table. These are the
//!   ones to look at;
//! - [`Target::ComputedCall`], a BLR through such an address: an indirect
//!   call, not a dispatch;
//! - [`Target::TailCall`], a BR right after a frame-restoring epilogue
//!   (`ldp x29, x30`, AUTIASP/AUTIBSP, `add sp`): a tail call;
//! - [`Target::Constant`], an address from ADR, or ADRP and ADD;
//! - [`Target::Parameter`], no write of the register in the
//!   [`WINDOW`] instructions before the branch (an argument, such as a
//!   callback or a tail call through a register).
//!
//! The walk goes back in address order, not along control flow, as the
//! jump-table walk does, so it is a heuristic: a write on another path can
//! decide the class. The audit only reports; it changes nothing.

use std::fmt::Write as _;

use serde::Serialize;

use crate::aarch64::{Instruction, MemoryOp};
use crate::codegen::Translation;
use crate::defuse::effects;

/// Instructions searched back from a branch for the write of its register.
pub const WINDOW: u64 = 512;
/// Register copies followed (`mov x8, x9`).
const COPIES: usize = 4;
/// Instructions before the branch included in a report's context.
const CONTEXT: u64 = 24;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Target {
    JumpTable {
        cases: usize,
    },
    Return,
    Loaded,
    Computed,
    /// A BLR through a computed address (a call through a function pointer
    /// computed from a table or an object): no table dispatch.
    ComputedCall,
    /// A BR right after a frame-restoring epilogue: a tail call through a
    /// register, whatever set it.
    TailCall,
    Constant,
    Parameter,
    /// Any other instruction wrote the register.
    Other,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Dispatch {
    pub pc: u64,
    /// BLR rather than BR.
    pub link: bool,
    pub target: Target,
    /// The instruction that set the target register, if one was found.
    pub definition: Option<u64>,
}

/// Every indirect branch of `translation`, in address order.
pub fn audit(translation: &Translation) -> Vec<Dispatch> {
    translation
        .instructions
        .values()
        .filter_map(|record| {
            let Instruction::BranchReg { rn, link } = record.instruction else {
                return None;
            };
            // RET is BR's encoding with opc 10 (0xd65f_xxxx).
            if record.word & 0xffff_fc1f == 0xd65f_0000 {
                return Some(Dispatch {
                    pc: record.pc,
                    link,
                    target: Target::Return,
                    definition: None,
                });
            }
            if let Some(&cases) = translation.jump_tables.get(&record.pc) {
                return Some(Dispatch {
                    pc: record.pc,
                    link,
                    target: Target::JumpTable { cases },
                    definition: None,
                });
            }
            if !link && follows_epilogue(translation, record.pc) {
                return Some(Dispatch {
                    pc: record.pc,
                    link,
                    target: Target::TailCall,
                    definition: None,
                });
            }
            let (mut target, definition) = classify(translation, record.pc, rn);
            if link && target == Target::Computed {
                target = Target::ComputedCall;
            }
            Some(Dispatch {
                pc: record.pc,
                link,
                target,
                definition,
            })
        })
        .collect()
}

/// Instructions before a BR searched for a frame-restoring epilogue.
const EPILOGUE: u64 = 4;

/// Whether one of the few instructions before `pc` restores the caller's
/// frame: `ldp x29, x30, [sp...]` (any addressing form), AUTIASP or
/// AUTIBSP, or `add sp, sp, #imm`. A BR after that leaves the function: a
/// tail call.
fn follows_epilogue(translation: &Translation, pc: u64) -> bool {
    (1..=EPILOGUE).any(|step| {
        let Some(record) = pc
            .checked_sub(step * 4)
            .and_then(|at| translation.instructions.get(&at))
        else {
            return false;
        };
        let word = record.word;
        let frame_pair = matches!(word & 0xffc0_0000, 0xa8c0_0000 | 0xa940_0000 | 0xa9c0_0000)
            && word & 0x7fff == (30 << 10) | (31 << 5) | 29;
        let authenticate = matches!(word, 0xd503_23bf | 0xd503_23ff);
        let release_frame = word & 0xff00_03ff == 0x9100_03ff;
        frame_pair || authenticate || release_frame
    })
}

/// How register `register` got its value before `pc`.
fn classify(translation: &Translation, pc: u64, register: u8) -> (Target, Option<u64>) {
    let mut register = register;
    let mut at = pc;
    let mut copies = 0;
    let start = pc.saturating_sub(WINDOW * 4);
    while at > start {
        at -= 4;
        let Some(record) = translation.instructions.get(&at) else {
            continue;
        };
        if register == 31 || effects(&record.instruction).writes.gpr & (1 << register) == 0 {
            continue;
        }
        use Instruction::*;
        let target = match record.instruction {
            LoadStore {
                op: MemoryOp::Load { .. },
                ..
            }
            | LoadStorePair {
                op: MemoryOp::Load { .. },
                ..
            }
            | LoadLiteral { .. }
            | LoadAcquireRcpc { .. }
            | LoadAcquireRcsc { .. }
            | LoadExclusive { .. } => Target::Loaded,
            AddSubShifted { rm, amount, .. } if amount > 0 && rm != 31 => Target::Computed,
            AddSubExtended { .. } => Target::Computed,
            Adr { .. } | AddSubImmediate { .. } => Target::Constant,
            // mov xD, xS (ORR with the zero register, no shift): follow xS.
            LogicalShifted {
                rn: 31,
                rm,
                invert: false,
                amount: 0,
                ..
            } if copies < COPIES => {
                copies += 1;
                register = rm;
                continue;
            }
            _ => Target::Other,
        };
        return (target, Some(at));
    }
    (Target::Parameter, None)
}

/// The indirect branches that look like unrecovered table dispatches.
pub fn unrecovered(dispatches: &[Dispatch]) -> impl Iterator<Item = &Dispatch> {
    dispatches
        .iter()
        .filter(|dispatch| dispatch.target == Target::Computed)
}

/// A text report: counts by class, then each unrecovered dispatch with the
/// words before it (address, word, disassembly), for reproducing it.
pub fn report(translation: &Translation, dispatches: &[Dispatch]) -> String {
    let mut text = String::new();
    let count = |wanted: fn(&Target) -> bool| {
        dispatches
            .iter()
            .filter(|dispatch| wanted(&dispatch.target))
            .count()
    };
    let _ = writeln!(
        text,
        "indirect branches: {} jump tables, {} returns, {} loaded pointers, {} computed \
         without a table, {} computed calls, {} tail calls, {} constants, {} parameters, \
         {} other",
        count(|t| matches!(t, Target::JumpTable { .. })),
        count(|t| *t == Target::Return),
        count(|t| *t == Target::Loaded),
        count(|t| *t == Target::Computed),
        count(|t| *t == Target::ComputedCall),
        count(|t| *t == Target::TailCall),
        count(|t| *t == Target::Constant),
        count(|t| *t == Target::Parameter),
        count(|t| *t == Target::Other),
    );
    for dispatch in unrecovered(dispatches) {
        let _ = writeln!(
            text,
            "\ncomputed {} at {:#x} (target set at {:#x}):",
            if dispatch.link { "blr" } else { "br" },
            dispatch.pc,
            dispatch.definition.unwrap_or(0)
        );
        let from = dispatch.pc.saturating_sub(CONTEXT * 4);
        for (pc, record) in translation.instructions.range(from..=dispatch.pc) {
            let _ = writeln!(
                text,
                "  {pc:#x}: {:08x}  {}",
                record.word, record.disassembly
            );
        }
    }
    text
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::codegen::{CodeRegion, discover};

    fn translation(words: &[u32]) -> Translation {
        let region = CodeRegion {
            name: "test".into(),
            base: 0x1000,
            bytes: words.iter().flat_map(|word| word.to_le_bytes()).collect(),
        };
        discover(&[region], &[0x1000], 64).unwrap()
    }

    fn only(words: &[u32]) -> Dispatch {
        let dispatches = audit(&translation(words));
        assert_eq!(dispatches.len(), 1, "{dispatches:?}");
        dispatches[0].clone()
    }

    #[test]
    fn returns_and_parameters() {
        assert_eq!(only(&[0xd65f_03c0]).target, Target::Return); // ret
        assert_eq!(only(&[0xd61f_0020]).target, Target::Parameter); // br x1
    }

    #[test]
    fn pointers_loaded_from_memory_through_copies() {
        // ldr x9, [x0, #8]; mov x8, x9; blr x8
        let dispatch = only(&[0xf940_0409, 0xaa09_03e8, 0xd63f_0100]);
        assert_eq!(dispatch.target, Target::Loaded);
        assert!(dispatch.link);
        assert_eq!(dispatch.definition, Some(0x1000));
    }

    #[test]
    fn computed_dispatches_without_a_table_are_flagged() {
        // ldrsw x8, [x9, w0, uxtw #2] with x9 unknown (no table found);
        // add x8, x9, x8, lsl #2; br x8
        let words = [0xb8a0_5928, 0x8b08_0928, 0xd61f_0100];
        let translation = translation(&words);
        let dispatches = audit(&translation);
        assert_eq!(dispatches[0].target, Target::Computed);
        assert_eq!(unrecovered(&dispatches).count(), 1);
        let text = report(&translation, &dispatches);
        assert!(text.contains("computed br at 0x1008"), "{text}");
        assert!(text.contains("0x1004: 8b080928"), "{text}");
    }

    #[test]
    fn computed_calls_are_not_dispatch_candidates() {
        // ldrsw x8, [x9, w0, uxtw #2]; add x8, x9, x8, lsl #2; blr x8; ret
        let dispatches = audit(&translation(&[
            0xb8a0_5928,
            0x8b08_0928,
            0xd63f_0100,
            0xd65f_03c0,
        ]));
        assert_eq!(dispatches[0].target, Target::ComputedCall);
        assert_eq!(unrecovered(&dispatches).count(), 0);
    }

    #[test]
    fn branches_after_an_epilogue_are_tail_calls() {
        // add x1, x2, x3, lsl #3 (computed); ldp x29, x30, [sp], #16;
        // autibsp; br x1, as at main+0x556308.
        let words = [0x8b03_0c41, 0xa8c1_7bfd, 0xd503_23ff, 0xd61f_0020];
        let dispatches = audit(&translation(&words));
        assert_eq!(dispatches[0].target, Target::TailCall);
        assert_eq!(unrecovered(&dispatches).count(), 0);
        // add sp, sp, #32; br x1
        let dispatches = audit(&translation(&[0x9100_83ff, 0xd61f_0020]));
        assert_eq!(dispatches[0].target, Target::TailCall);
    }

    #[test]
    fn constants() {
        // adr x8, 0x1008; br x8; ret
        let dispatches = audit(&translation(&[0x1000_0048, 0xd61f_0100, 0xd65f_03c0]));
        assert_eq!(dispatches[0].target, Target::Constant);
    }
}
