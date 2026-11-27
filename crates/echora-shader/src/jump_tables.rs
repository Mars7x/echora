//! Prove the possible targets of the compiler's bounded constant-bank jumps.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::{Range, RangeInclusive};

use crate::decode::{Combine, Instruction, IntCompare, Op, Operand, Reg, Src};

/// Targets proved from the same bank-1 bytes that the command binds.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct JumpTargets(pub(crate) BTreeMap<usize, Vec<usize>>);

#[derive(Debug)]
struct Site {
    instruction: usize,
    words: Range<usize>,
    definitions: RangeInclusive<usize>,
}

/// Bounded jump table reads in a program. Its instructions bound the
/// indices and the command's constant bank supplies destinations.
#[derive(Debug)]
pub struct JumpTables {
    sites: Vec<Site>,
    instructions: usize,
}

impl JumpTables {
    pub fn from_compute_program(program: &[u8]) -> Option<Self> {
        Self::analyze(&crate::program_code(program).ok()?)
    }

    pub fn from_fragment_program(program: &[u8]) -> Option<Self> {
        if crate::header::parse(program)?.stage != echora_shader_abi::Stage::Fragment {
            return None;
        }
        let code = crate::program_code(program.get(crate::GRAPHICS_HEADER_SIZE..)?).ok()?;
        Self::analyze(&code)
    }

    pub(crate) fn analyze(code: &[Instruction]) -> Option<Self> {
        let mut entries = BTreeSet::from([0]);
        let mut sites = Vec::new();
        for (index, instruction) in code.iter().enumerate() {
            match instruction.op {
                Op::Bra { offset, .. } | Op::Call { offset } => {
                    entries.insert(crate::compute::relative_target(index, offset, code.len())?);
                    entries.insert(index + 1);
                }
                Op::Brx { target } => {
                    sites.push(bounded(code, index, target)?);
                    entries.insert(index + 1);
                }
                Op::Ret { .. } => return None,
                Op::Exit { .. } => {
                    entries.insert(index + 1);
                }
                _ => {}
            }
        }
        // No other entry may bypass the clamp, address calculation or load.
        if sites.is_empty()
            || sites
                .iter()
                .any(|site| entries.range(site.definitions.clone()).next().is_some())
        {
            return None;
        }
        Some(Self {
            sites,
            instructions: code.len(),
        })
    }

    /// The part of the bank that controls the jumps, in instruction order.
    pub fn table_bytes<'a>(
        &'a self,
        constants: &'a [u8],
    ) -> impl Iterator<Item = Option<&'a [u8]>> + 'a {
        self.sites
            .iter()
            .map(move |site| constants.get(site.words.clone()))
    }

    pub fn resolve(&self, constants: &[u8]) -> Option<JumpTargets> {
        let mut tables = BTreeMap::new();
        for (site, data) in self.sites.iter().zip(self.table_bytes(constants)) {
            let mut targets = Vec::new();
            for bytes in data?.as_chunks::<4>().0 {
                let byte = usize::try_from(u32::from_le_bytes(*bytes)).ok()?;
                let index = byte / 16;
                if byte % 16 != 0
                    || index >= self.instructions
                    || self
                        .sites
                        .iter()
                        .any(|site| site.definitions.contains(&index))
                {
                    return None;
                }
                targets.push(index);
            }
            targets.sort_unstable();
            targets.dedup();
            tables.insert(site.instruction, targets);
        }
        Some(JumpTargets(tables))
    }
}

fn simple_operand(value: Operand) -> Option<Src> {
    (!value.abs && !value.neg).then_some(value.src)
}

fn bounded(code: &[Instruction], index: usize, target: Src) -> Option<Site> {
    let (target, uniform, zero) = match target {
        Src::Reg(register) => (register, false, Reg::ZERO),
        Src::Uniform(register) => (Reg(register.0), true, Reg(63)),
        _ => return None,
    };
    if target.0 >= zero.0 {
        return None;
    }
    let mut previous = index;
    let mut instruction = || {
        previous = previous.checked_sub(1)?;
        let definition = code[previous];
        (definition.guard.is_true() && definition.uniform == uniform).then_some(definition.op)
    };
    let trailing = instruction()?;
    let load = match trailing {
        Op::Shf {
            d,
            low,
            high,
            right: true,
            high_word: true,
            ..
        } if d != target && low == zero && simple_operand(high) == Some(Src::Reg(target)) => {
            instruction()?
        }
        Op::Ldc { .. } => trailing,
        _ => return None,
    };
    let Op::Ldc {
        d,
        index: address,
        bank: 1,
        offset,
        size: 4,
        mode: 0,
    } = load
    else {
        return None;
    };
    if d != target || offset % 4 != 0 || address.0 >= zero.0 {
        return None;
    }
    let Op::Shf {
        d,
        low: bounded,
        shift,
        high,
        kind: 3,
        right: false,
        high_word: false,
        ..
    } = instruction()?
    else {
        return None;
    };
    if d != address
        || bounded.0 >= zero.0
        || simple_operand(shift) != Some(Src::Imm(2))
        || simple_operand(high) != Some(Src::Reg(zero))
    {
        return None;
    }
    let upper = match instruction()? {
        Op::Imnmx {
            d,
            b,
            min,
            signed: false,
            ..
        } if d == bounded && min.is_true() => {
            let Src::Imm(limit) = simple_operand(b)? else {
                return None;
            };
            limit
        }
        Op::Sel {
            d,
            a,
            b,
            condition,
            float: false,
        } if d == bounded && !condition.negated => {
            let Src::Imm(limit) = simple_operand(b)? else {
                return None;
            };
            let Op::Isetp {
                p,
                q,
                a: left,
                b: right,
                compare,
                signed: false,
                combine: Combine::And,
                accumulate,
                extended: None,
                ..
            } = instruction()?
            else {
                return None;
            };
            if p >= 7
                || q == p
                || p != condition.index
                || !accumulate.is_true()
                || simple_operand(a)? != simple_operand(left)?
                || simple_operand(right)? != Src::Imm(limit)
                || !matches!(compare, IntCompare::Lt | IntCompare::Le)
            {
                return None;
            }
            limit
        }
        _ => return None,
    };
    // Keep the resulting switches small, even with large constant banks.
    if upper >= 64 {
        return None;
    }
    let start = usize::try_from(offset).ok()?;
    let end = start + (upper as usize + 1) * 4;
    if end > 0x10000 {
        return None;
    }
    Some(Site {
        instruction: index,
        words: start..end,
        definitions: (previous + 1)..=index,
    })
}
