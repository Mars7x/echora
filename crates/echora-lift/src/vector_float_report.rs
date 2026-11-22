//! What keeps vector float operations out of runs
//! (`EmitOptions::vector_float_runs`): for the forms a run may hold
//! (`crate::simd::vector_run_op`), how many there are, how many sit in a run
//! today and by run length, how many are singles; what sits in the gaps
//! between them in their blocks; and how many would be in a run under looser
//! rules. `echora lift` prints it when `ECHORA_LIFT_REPORT` names
//! `vector-float`.
//!
//! Counts can be weighted per window (`parse_weights`: a profile's samples
//! per lifted window); every block then counts with its window's weight,
//! and a window without one counts nothing.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::aarch64::Instruction;
use crate::codegen::{BasicBlock, Translation};
use crate::defuse::{Regs, effects};
use crate::simd::{SimdOp, VectorLogicalOp};

/// The instruction classes of a gap between vector float operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Class {
    /// A vector float operation no run holds (FCVTL, FCVTN, the half
    /// precision forms, ...).
    VectorFloatOther,
    /// SIMD and FP register loads (LDR, LDP, LD1 and kin).
    VectorLoad,
    /// SIMD and FP register stores.
    VectorStore,
    /// Register moves and permutes: MOV (ORR of one register), EXT, DUP,
    /// INS, UMOV, MOVI, UZP/ZIP/TRN, TBL/TBX, FMOV between registers.
    Move,
    /// Vector integer and logic operations.
    VectorInt,
    /// Scalar float operations (arithmetic, compares, conversions, moves
    /// to and from general registers).
    ScalarFloat,
    /// General-register arithmetic and address computation.
    GeneralArithmetic,
    /// Compares (anything setting NZCV) and conditional selects.
    CompareSelect,
    /// General-register loads and stores.
    GeneralMemory,
    /// BL and BLR.
    Call,
    /// MRS/MSR of FPCR or FPSR, SVC, system operations, barriers, and what
    /// stops translated execution.
    StatusOrTrap,
    /// Branches and everything else.
    Other,
}

impl Class {
    const ALL: [Self; 12] = [
        Self::VectorFloatOther,
        Self::VectorLoad,
        Self::VectorStore,
        Self::Move,
        Self::VectorInt,
        Self::ScalarFloat,
        Self::GeneralArithmetic,
        Self::CompareSelect,
        Self::GeneralMemory,
        Self::Call,
        Self::StatusOrTrap,
        Self::Other,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::VectorFloatOther => "vector float, not a run form",
            Self::VectorLoad => "vector loads",
            Self::VectorStore => "vector stores",
            Self::Move => "register moves and permutes",
            Self::VectorInt => "vector integer",
            Self::ScalarFloat => "scalar float",
            Self::GeneralArithmetic => "general-register arithmetic",
            Self::CompareSelect => "compares and selects",
            Self::GeneralMemory => "general loads and stores",
            Self::Call => "calls",
            Self::StatusOrTrap => "FPCR/FPSR, SVC, system, traps",
            Self::Other => "branches and other",
        }
    }
}

/// Whether `instruction` is a form a vector float run holds.
fn held(instruction: &Instruction) -> bool {
    matches!(instruction, Instruction::Simd(op) if crate::simd::vector_run_op(op))
}

/// The class of an instruction that is not a run form.
pub fn class(instruction: &Instruction) -> Class {
    let effect = effects(instruction);
    if effect.calls {
        return Class::Call;
    }
    if matches!(
        instruction,
        Instruction::ReadFpControl { .. } | Instruction::WriteFpControl { .. }
    ) || effect.svc
        || effect.traps
        || effect.system
        || effect.barrier
    {
        return Class::StatusOrTrap;
    }
    if let Instruction::LoadStoreVectorPair { .. } = instruction {
        return if effect.loads {
            Class::VectorLoad
        } else {
            Class::VectorStore
        };
    }
    if let Instruction::Simd(op) = instruction {
        if effect.loads {
            return Class::VectorLoad;
        }
        if effect.stores {
            return Class::VectorStore;
        }
        return match *op {
            SimdOp::VectorLogical {
                op: VectorLogicalOp::Orr,
                rn,
                rm,
                ..
            } if rn == rm => Class::Move,
            SimdOp::FpUnary {
                op: crate::simd::FpUnaryOp::Mov,
                ..
            }
            | SimdOp::MoveImmediate { .. }
            | SimdOp::FpMoveImmediate { .. }
            | SimdOp::InsertElement { .. }
            | SimdOp::InsertGeneral { .. }
            | SimdOp::MoveToGeneral { .. }
            | SimdOp::DupElement { .. }
            | SimdOp::DupGeneral { .. }
            | SimdOp::VecPermute { .. }
            | SimdOp::VecExt { .. }
            | SimdOp::VecTable { .. } => Class::Move,
            SimdOp::FpUnary { .. }
            | SimdOp::FpConvert { .. }
            | SimdOp::FpBinary { .. }
            | SimdOp::FpTernary { .. }
            | SimdOp::FpCompare { .. }
            | SimdOp::FpConditionalCompare { .. }
            | SimdOp::FpSelect { .. }
            | SimdOp::FpMoveToGeneral { .. }
            | SimdOp::FpMoveFromGeneral { .. }
            | SimdOp::IntToFp { .. }
            | SimdOp::SimdIntToFp { .. }
            | SimdOp::FpToInt { .. }
            | SimdOp::FpToFixed { .. }
            | SimdOp::IntToFpFixed { .. } => Class::ScalarFloat,
            SimdOp::VecFp { .. }
            | SimdOp::VecFpPairwise { .. }
            | SimdOp::VecFpElement { .. }
            | SimdOp::VecFpUnary { .. }
            | SimdOp::VecFpAcross { .. }
            | SimdOp::VecFpConvert { .. }
            | SimdOp::FpConvertPrecision { .. } => Class::VectorFloatOther,
            _ => Class::VectorInt,
        };
    }
    if effect.loads || effect.stores {
        return Class::GeneralMemory;
    }
    if effect.writes.nzcv
        || matches!(
            instruction,
            Instruction::Select { .. }
                | Instruction::ConditionalCompare { .. }
                | Instruction::ConditionalCompareNegative { .. }
        )
    {
        return Class::CompareSelect;
    }
    if effect.branch {
        return Class::Other;
    }
    if matches!(instruction, Instruction::Nop) {
        return Class::Other;
    }
    Class::GeneralArithmetic
}

/// What a looser run rule lets a run hold besides the run forms.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Rule {
    /// Register moves and permutes (`Class::Move`).
    pub moves: bool,
    /// Vector loads whose address registers no instruction of the run
    /// writes (a load with writeback writes its base, so never).
    pub loads: bool,
    /// General-register arithmetic (`Class::GeneralArithmetic`), unless it
    /// writes a register an earlier load of the run addressed through.
    pub arithmetic: bool,
    /// A run may start at a block's first instruction and end at its last.
    pub edges: bool,
}

/// The block's run forms that a run would hold under `rule`, by index: a
/// run is two or more run forms with only what `rule` admits between them.
/// `Rule::default()` gives today's runs (`codegen::vector_runs_in`).
pub fn in_runs(block: &BasicBlock, rule: Rule) -> Vec<bool> {
    let count = block.instructions.len();
    let mut in_run = vec![false; count];
    let (low, high) = if rule.edges {
        (0, count)
    } else {
        (1, count.saturating_sub(1))
    };
    let mut current: Vec<usize> = Vec::new();
    let mut written = Regs::NONE;
    let mut addressed = Regs::NONE;
    let close = |current: &mut Vec<usize>, in_run: &mut Vec<bool>| {
        if current.len() >= 2 {
            for &index in current.iter() {
                in_run[index] = true;
            }
        }
        current.clear();
    };
    for index in low..high.max(low) {
        let instruction = &block.instructions[index].instruction;
        if held(instruction) {
            current.push(index);
            continue;
        }
        let effect = effects(instruction);
        let general = |regs: Regs| Regs {
            vector: 0,
            nzcv: false,
            ..regs
        };
        let admitted = match class(instruction) {
            Class::Move => rule.moves,
            Class::VectorLoad => {
                rule.loads
                    && !general(effect.reads).intersects(written)
                    && !general(effect.writes).intersects(general(effect.reads))
            }
            Class::GeneralArithmetic => {
                rule.arithmetic && !general(effect.writes).intersects(addressed)
            }
            _ => false,
        };
        if admitted {
            if class(instruction) == Class::VectorLoad {
                addressed = addressed.union(general(effect.reads));
            }
            written = written.union(general(effect.writes));
        } else {
            close(&mut current, &mut in_run);
            written = Regs::NONE;
            addressed = Regs::NONE;
        }
    }
    close(&mut current, &mut in_run);
    in_run
}

/// Gap lengths reported: the first is two operations next to each other
/// but in different runs (a block's first or last instruction is never in
/// one), the last a block's edge (no other operation on that side).
const GAP_LENGTHS: [&str; 7] = ["0", "1", "2", "3-4", "5-8", "9+", "block edge"];

fn gap_bucket(length: usize) -> usize {
    match length {
        0 => 0,
        1 => 1,
        2 => 2,
        3..=4 => 3,
        5..=8 => 4,
        _ => 5,
    }
}

/// The report (`vector_float_report`); every count weighted.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VectorFloatReport {
    /// Whether the counts are weighted by a profile.
    pub weighted: bool,
    /// Blocks counted.
    pub blocks: f64,
    /// Run forms.
    pub operations: f64,
    /// Run forms in a run today.
    pub in_runs: f64,
    /// Runs today, and the operations in them, by length 2, 3, 4, 5-8, 9+.
    pub runs_by_length: [(f64, f64); 5],
    /// Run forms in no run.
    pub singles: f64,
    /// Gaps around singles and runs (between neighbouring ones of a block,
    /// each counted once, and to the block's edges), by `GAP_LENGTHS`.
    pub gaps: [f64; 7],
    /// For the gaps between two of them: per class, the gaps holding at
    /// least one such instruction and the instructions.
    pub gap_classes: BTreeMap<Class, (f64, f64)>,
    /// Run forms that would be in a run: (a) with moves and permutes, (b)
    /// with those and loads, (c) with those and general arithmetic, (d)
    /// with runs at the block's edges (today's forms only), (c+d) both.
    pub candidates: [f64; 5],
}

/// The candidate rules of `VectorFloatReport::candidates`.
pub const CANDIDATES: [(&str, Rule); 5] = [
    (
        "(a) moves and permutes",
        Rule {
            moves: true,
            loads: false,
            arithmetic: false,
            edges: false,
        },
    ),
    (
        "(b) (a) and loads not addressed through the run's writes",
        Rule {
            moves: true,
            loads: true,
            arithmetic: false,
            edges: false,
        },
    ),
    (
        "(c) (b) and general-register arithmetic",
        Rule {
            moves: true,
            loads: true,
            arithmetic: true,
            edges: false,
        },
    ),
    (
        "(d) today's forms, runs at a block's first or last instruction",
        Rule {
            moves: false,
            loads: false,
            arithmetic: false,
            edges: true,
        },
    ),
    (
        "(c) and (d)",
        Rule {
            moves: true,
            loads: true,
            arithmetic: true,
            edges: true,
        },
    ),
];

/// Window weights from a profile: lines `WEIGHT REGIONHEX run_window_NAME`
/// (blank lines and lines starting with `#` ignored). The key is the
/// region and the window function's name.
pub fn parse_weights(text: &str) -> Result<BTreeMap<(u64, String), f64>, String> {
    let mut weights = BTreeMap::new();
    for (number, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        let [weight, region, window] = fields[..] else {
            return Err(format!(
                "line {}: expected WEIGHT REGIONHEX run_window_NAME",
                number + 1
            ));
        };
        let weight: f64 = weight
            .parse()
            .map_err(|_| format!("line {}: weight {weight:?}", number + 1))?;
        let region = u64::from_str_radix(
            region.trim_start_matches("gen-r-").trim_start_matches("0x"),
            16,
        )
        .map_err(|_| format!("line {}: region {region:?}", number + 1))?;
        *weights.entry((region, window.to_owned())).or_insert(0.0) += weight;
    }
    Ok(weights)
}

/// The report over the blocks of `translation` (only those of `regions`,
/// region ids as in the crate names, when given), each weighted by its
/// window's entry in `weights` (when given; 0 when absent) or 1.
pub fn vector_float_report(
    translation: &Translation,
    region_bits: u32,
    regions: Option<&BTreeSet<u64>>,
    weights: Option<&BTreeMap<(u64, String), f64>>,
) -> VectorFloatReport {
    let mut report = VectorFloatReport {
        weighted: weights.is_some(),
        ..VectorFloatReport::default()
    };
    for block in &translation.blocks {
        let region = block.start >> region_bits;
        if regions.is_some_and(|regions| !regions.contains(&region)) {
            continue;
        }
        let weight = match weights {
            Some(weights) => {
                let window = format!(
                    "run_window_{:x}",
                    block.start >> crate::codegen::SPLIT_WINDOW_BITS
                );
                weights.get(&(region, window)).copied().unwrap_or(0.0)
            }
            None => 1.0,
        };
        if weight == 0.0 {
            continue;
        }
        add_block(&mut report, block, weight);
    }
    report
}

fn add_block(report: &mut VectorFloatReport, block: &BasicBlock, weight: f64) {
    report.blocks += weight;
    let held_at: Vec<usize> = (0..block.instructions.len())
        .filter(|&index| held(&block.instructions[index].instruction))
        .collect();
    if held_at.is_empty() {
        return;
    }
    report.operations += weight * held_at.len() as f64;
    let today = in_runs(block, Rule::default());
    // Singles and runs, in order, as instruction ranges.
    let mut segments: Vec<(usize, usize)> = Vec::new();
    for &index in &held_at {
        match segments.last_mut() {
            Some((_, end)) if today[index] && *end == index && today[index - 1] => *end = index + 1,
            _ => segments.push((index, index + 1)),
        }
    }
    for &(start, end) in &segments {
        let length = end - start;
        if length >= 2 {
            report.in_runs += weight * length as f64;
            let bucket = match length {
                2 => 0,
                3 => 1,
                4 => 2,
                5..=8 => 3,
                _ => 4,
            };
            report.runs_by_length[bucket].0 += weight;
            report.runs_by_length[bucket].1 += weight * length as f64;
        } else {
            report.singles += weight;
        }
    }
    // Two edges per block, and the gaps between neighbours.
    report.gaps[6] += 2.0 * weight;
    for pair in segments.windows(2) {
        let (gap_start, gap_end) = (pair[0].1, pair[1].0);
        report.gaps[gap_bucket(gap_end - gap_start)] += weight;
        let mut seen = BTreeSet::new();
        for record in &block.instructions[gap_start..gap_end] {
            let class = class(&record.instruction);
            let entry = report.gap_classes.entry(class).or_insert((0.0, 0.0));
            entry.1 += weight;
            if seen.insert(class) {
                entry.0 += weight;
            }
        }
    }
    for (slot, (_, rule)) in report.candidates.iter_mut().zip(CANDIDATES) {
        *slot += weight
            * in_runs(block, rule)
                .iter()
                .filter(|&&in_run| in_run)
                .count() as f64;
    }
}

impl fmt::Display for VectorFloatReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let number = |value: f64| {
            if self.weighted {
                format!("{value:.1}")
            } else {
                format!("{value:.0}")
            }
        };
        let share = |value: f64| {
            if self.operations > 0.0 {
                format!("{:.1}%", value / self.operations * 100.0)
            } else {
                "-".to_owned()
            }
        };
        writeln!(
            f,
            "vector float report ({}): {} blocks, {} run forms, {} in runs ({}), {} singles",
            if self.weighted {
                "weighted by window"
            } else {
                "unweighted"
            },
            number(self.blocks),
            number(self.operations),
            number(self.in_runs),
            share(self.in_runs),
            number(self.singles)
        )?;
        for (label, (runs, operations)) in
            ["2", "3", "4", "5-8", "9+"].iter().zip(self.runs_by_length)
        {
            writeln!(
                f,
                "  runs of {label}: {} runs, {} operations",
                number(runs),
                number(operations)
            )?;
        }
        writeln!(
            f,
            "  gaps around singles and runs (between neighbours in a block, each once; edges: before the first and after the last):"
        )?;
        for (label, count) in GAP_LENGTHS.iter().zip(self.gaps) {
            let unit = if *label == "block edge" {
                "s"
            } else {
                " instructions"
            };
            writeln!(f, "    {label:>10}{unit}: {}", number(count))?;
        }
        writeln!(
            f,
            "  what the gaps between neighbours hold (gaps holding any, instructions):"
        )?;
        for class in Class::ALL {
            let (gaps, instructions) = self.gap_classes.get(&class).copied().unwrap_or_default();
            writeln!(
                f,
                "    {:<32} {} gaps, {} instructions",
                class.name(),
                number(gaps),
                number(instructions)
            )?;
        }
        writeln!(f, "  run forms that would be in a run:")?;
        writeln!(
            f,
            "    today: {} ({})",
            number(self.in_runs),
            share(self.in_runs)
        )?;
        for ((label, _), count) in CANDIDATES.iter().zip(self.candidates) {
            writeln!(f, "    {label}: {} ({})", number(count), share(count))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::{CodeRegion, discover};

    /// Block A (entry 0x1000):
    ///  0 NOP
    ///  1 FADD V0.4S, V1.4S, V2.4S     run today
    ///  2 FMUL V3.4S, V0.4S, V2.4S     run today
    ///  3 MOV V4.16B, V3.16B           (move)
    ///  4 FADD V5.4S, V4.4S, V2.4S     single
    ///  5 LDR Q6, [X0]                 (vector load)
    ///  6 FMUL V7.4S, V6.4S, V2.4S     single
    ///  7 ADD X1, X1, #1               (general arithmetic)
    ///  8 FADD V8.4S, V7.4S, V2.4S     single
    ///  9 LDR Q9, [X1]                 (vector load through X1, written at 7)
    /// 10 FADD V10.4S, V9.4S, V2.4S    single
    /// 11 NOP
    /// 12 RET
    /// Block B (entry 0x1034): FADD V12.4S; FADD V13.4S, V12.4S; RET: no
    /// run today (the first instruction is never in one).
    /// Block C (entry 0x1040): NOP; FADD V14.4S; LDR Q15, [X2]; ADD X2, X2,
    /// #16; FADD V16.4S, V15.4S; RET: the ADD writes the register the load
    /// addressed through, so no rule puts the FADDs in one run.
    fn translation() -> Translation {
        let words: [u32; 22] = [
            0xd503_201f,
            0x4e22_d420,
            0x6e22_dc03,
            0x4ea3_1c64,
            0x4e22_d485,
            0x3dc0_0006,
            0x6e22_dcc7,
            0x9100_0421,
            0x4e22_d4e8,
            0x3dc0_0029,
            0x4e22_d52a,
            0xd503_201f,
            0xd65f_03c0,
            0x4e22_d42c,
            0x4e22_d58d,
            0xd65f_03c0,
            0xd503_201f,
            0x4e22_d42e,
            0x3dc0_004f,
            0x9100_4042,
            0x4e22_d5f0,
            0xd65f_03c0,
        ];
        let region = CodeRegion {
            name: "report".into(),
            base: 0x1000,
            bytes: words.iter().flat_map(|word| word.to_le_bytes()).collect(),
        };
        discover(&[region], &[0x1000, 0x1034, 0x1040], 32).unwrap()
    }

    #[test]
    fn the_report_counts_a_known_block() {
        let translation = translation();
        assert_eq!(translation.blocks.len(), 3);
        // Today's rule is the emission's.
        for block in &translation.blocks {
            let mut expected = vec![false; block.instructions.len()];
            for (start, end) in crate::codegen::vector_runs_in(block) {
                expected[start..end].fill(true);
            }
            assert_eq!(in_runs(block, Rule::default()), expected);
        }
        let report = vector_float_report(&translation, 16, None, None);
        assert_eq!(report.blocks, 3.0);
        assert_eq!(report.operations, 10.0);
        assert_eq!(report.in_runs, 2.0);
        assert_eq!(report.runs_by_length[0], (1.0, 2.0));
        assert_eq!(report.singles, 8.0);
        // A: four gaps of one; B: its two operations side by side; C: a
        // gap of two; two edges per block.
        assert_eq!(report.gaps, [1.0, 4.0, 1.0, 0.0, 0.0, 0.0, 6.0]);
        assert_eq!(report.gap_classes[&Class::Move], (1.0, 1.0));
        assert_eq!(report.gap_classes[&Class::VectorLoad], (3.0, 3.0));
        assert_eq!(report.gap_classes[&Class::GeneralArithmetic], (2.0, 2.0));
        assert_eq!(report.gap_classes.len(), 3);
        // (a) 1, 2, 4; (b) 1, 2, 4, 6 (the ADD ends it) and 8, 10; (c) 1
        // to 8 (the LDR through X1, written in the run, ends it); (d)
        // today's and B's two; (c) and (d): (c)'s and B's two.
        assert_eq!(report.candidates, [3.0, 6.0, 5.0, 4.0, 7.0]);
    }

    #[test]
    fn weights_select_and_scale_windows() {
        let translation = translation();
        let weights = parse_weights("# hot windows\n2.5 0 run_window_4\n").unwrap();
        let report = vector_float_report(&translation, 16, None, Some(&weights));
        // All three blocks are in window 4 (0x1000 to 0x13ff).
        assert_eq!(report.operations, 25.0);
        assert_eq!(report.in_runs, 5.0);
        let other = parse_weights("1 1 run_window_4\n").unwrap();
        assert_eq!(
            vector_float_report(&translation, 16, None, Some(&other)).operations,
            0.0
        );
        assert_eq!(
            vector_float_report(&translation, 16, Some(&BTreeSet::from([1])), None).operations,
            0.0
        );
        assert!(parse_weights("1 zz run_window_4").is_err());
        assert!(parse_weights("1 0").is_err());
    }
}
