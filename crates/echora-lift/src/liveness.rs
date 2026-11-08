//! Register and flag liveness over a translation, and the writes it shows
//! dead.
//!
//! A backward data-flow pass over the translation's basic blocks, using
//! [`defuse::effects`](crate::defuse::effects) for what each instruction
//! reads and writes. It is conservative wherever control leaves what the
//! analysis can see: everything is live after a call (BL, BLR), an SVC, an
//! instruction that stops translated code, an indirect branch (including
//! jump tables and returns), and an edge to an address that starts no
//! block. A write is dead when nothing can read it before it is overwritten
//! or the state is discarded, so emission could skip computing it: an
//! NZCV result nobody tests, or a register result overwritten first.
//!
//! [`dead_writes`] counts them. Generated code keeps flags and registers
//! in locals, so the host compiler removes most dead computations itself.
//! The count is an estimate of the Rust that emission need not write at all,
//! which costs rustc front-end and LLVM time even when it compiles to
//! nothing.

use std::collections::BTreeMap;

use serde::Serialize;

use crate::aarch64::Instruction;
use crate::codegen::Translation;
use crate::defuse::{Regs, effects};

/// Counts of dead writes over a translation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct DeadWrites {
    pub instructions: usize,
    /// Instructions that write NZCV.
    pub flag_writes: usize,
    /// Those whose NZCV nothing reads.
    pub dead_flag_writes: usize,
    /// General-register writes (one per register written).
    pub register_writes: usize,
    /// Those nothing reads.
    pub dead_register_writes: usize,
    /// Instructions all of whose register and flag writes are dead and that
    /// have no other effect (no memory access, branch or system effect).
    pub dead_instructions: usize,
}

/// What `instruction` reads, with everything read where control leaves
/// the analysis (calls, SVCs, traps).
fn reads(instruction: &Instruction) -> Regs {
    let e = effects(instruction);
    if e.calls || e.svc || e.traps {
        Regs::ALL
    } else {
        e.reads
    }
}

/// Registers and flags live at the end of each block (by block start).
pub fn live_out(translation: &Translation) -> BTreeMap<u64, Regs> {
    let starts: std::collections::BTreeSet<u64> =
        translation.blocks.iter().map(|block| block.start).collect();
    // Successor blocks, or None when control leaves what the analysis sees.
    let successors: BTreeMap<u64, Option<Vec<u64>>> = translation
        .blocks
        .iter()
        .map(|block| {
            let last = block
                .instructions
                .last()
                .expect("blocks have an instruction");
            let next = last.instruction.successors(last.pc);
            let leaves = matches!(last.instruction, Instruction::BranchReg { .. })
                || next.is_empty()
                || next.iter().any(|pc| !starts.contains(pc));
            (block.start, (!leaves).then_some(next))
        })
        .collect();
    let mut live_in: BTreeMap<u64, Regs> = translation
        .blocks
        .iter()
        .map(|block| (block.start, Regs::NONE))
        .collect();
    let mut out: BTreeMap<u64, Regs> = BTreeMap::new();
    loop {
        let mut changed = false;
        for block in translation.blocks.iter().rev() {
            let after = match &successors[&block.start] {
                None => Regs::ALL,
                Some(next) => next
                    .iter()
                    .fold(Regs::NONE, |live, pc| live.union(live_in[pc])),
            };
            out.insert(block.start, after);
            let mut live = after;
            for record in block.instructions.iter().rev() {
                live = live
                    .difference(effects(&record.instruction).writes)
                    .union(reads(&record.instruction));
            }
            if live != live_in[&block.start] {
                live_in.insert(block.start, live);
                changed = true;
            }
        }
        if !changed {
            return out;
        }
    }
}

/// The dead NZCV and register writes of `translation`.
pub fn dead_writes(translation: &Translation) -> DeadWrites {
    let out = live_out(translation);
    let mut counts = DeadWrites::default();
    for block in &translation.blocks {
        let mut live = out[&block.start];
        for record in block.instructions.iter().rev() {
            let e = effects(&record.instruction);
            counts.instructions += 1;
            if e.writes.nzcv {
                counts.flag_writes += 1;
                if !live.nzcv {
                    counts.dead_flag_writes += 1;
                }
            }
            let written = e.writes.gpr.count_ones() as usize;
            let dead_registers = (e.writes.gpr & !live.gpr).count_ones() as usize;
            counts.register_writes += written;
            counts.dead_register_writes += dead_registers;
            let all_dead = !(e.writes.nzcv && live.nzcv)
                && e.writes.gpr & live.gpr == 0
                && !(e.writes.sp && live.sp)
                && e.writes.vector & live.vector == 0;
            let side_effects = e.loads
                || e.stores
                || e.calls
                || e.svc
                || e.system
                || e.barrier
                || e.event_wait
                || e.traps
                || record.instruction.ends_block();
            if all_dead && !side_effects && !e.writes.is_empty() {
                counts.dead_instructions += 1;
            }
            live = live.difference(e.writes).union(reads(&record.instruction));
        }
    }
    counts
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::codegen::{CodeRegion, discover};

    fn counts(words: &[u32]) -> DeadWrites {
        let region = CodeRegion {
            name: "test".into(),
            base: 0x1000,
            bytes: words.iter().flat_map(|word| word.to_le_bytes()).collect(),
        };
        dead_writes(&discover(&[region], &[0x1000], 64).unwrap())
    }

    #[test]
    fn flags_overwritten_before_a_test_are_dead() {
        // adds x0, x0, #1 (flags dead); subs x1, x1, #1; b.ne back; ret
        let c = counts(&[0xb100_0400, 0xf100_0421, 0x54ff_ffc1, 0xd65f_03c0]);
        assert_eq!((c.flag_writes, c.dead_flag_writes), (2, 1));
    }

    #[test]
    fn registers_overwritten_first_are_dead() {
        // mov x0, #1 (dead: overwritten); mov x0, #2; ret
        let c = counts(&[0xd280_0020, 0xd280_0040, 0xd65f_03c0]);
        assert_eq!(c.dead_register_writes, 1);
        assert_eq!(c.dead_instructions, 1);
    }

    #[test]
    fn everything_is_live_where_control_leaves() {
        // mov x0, #1; ret (the caller may read x0); mov x9, #1; bl (the
        // callee may read x9).
        assert_eq!(counts(&[0xd280_0020, 0xd65f_03c0]).dead_register_writes, 0);
        assert_eq!(
            counts(&[0xd280_0029, 0x9400_0002, 0xd65f_03c0, 0xd65f_03c0]).dead_register_writes,
            0
        );
    }
}
