//! Lifts the synthetic programs into Rust modules compiled with this crate.

use std::fmt::Write as _;
use std::path::PathBuf;

use echora_lift::codegen::{
    CodeRegion, discover, discover_with_tables, emit_rust, emit_rust_split,
};

include!("src/programs.rs");

mod kernels {
    include!("src/kernels.rs");
}

fn region(words: &[u32]) -> CodeRegion {
    CodeRegion {
        name: "synthetic".into(),
        base: CODE_BASE,
        bytes: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

fn module(out: &mut String, name: &str, generated: &str) {
    writeln!(
        out,
        "pub mod {name} {{\n{generated}\n\
         pub fn run(m: &mut crate::Machine) -> Result<(), Trap> {{ m.run(dispatch) }}\n}}"
    )
    .expect("writing to a String cannot fail");
}

fn main() {
    println!("cargo::rerun-if-changed=src/programs.rs");
    println!("cargo::rerun-if-changed=src/kernels.rs");
    let mut out = String::new();
    for (name, words) in PROGRAMS {
        let translation = discover(&[region(words)], &[CODE_BASE], 100)
            .unwrap_or_else(|e| panic!("lifting {name}: {e:#}"));
        module(&mut out, name, &emit_rust(&translation));
    }

    // A call across a 4 KiB dispatch window, entered from every address.
    let mut words = vec![0xd503201f; 300];
    words[0] = 0x94000100; // BL to a block in the next window.
    words[1] = 0xd40000e1;
    words[256] = 0x91004463; // ADD X3, X3, #17.
    words[257] = 0xd65f03c0;
    let entries: Vec<u64> = (0..words.len()).map(|i| CODE_BASE + i as u64 * 4).collect();
    let translation = discover(&[region(&words)], &entries, 400).expect("sharded_call_return");
    module(&mut out, "sharded_call_return", &emit_rust(&translation));

    // The same program through split emission (the layout real builds use):
    // region sources become child modules named as the root expects.
    let (root, regions) = emit_rust_split(&translation, 10);
    writeln!(out, "pub mod split_call_return {{").expect("write");
    for (region, source) in regions {
        writeln!(out, "pub mod gen_r_{region:x} {{\n{source}\n}}").expect("write");
    }
    writeln!(
        out,
        "{root}\npub fn run(m: &mut crate::Machine) -> Result<(), Trap> {{ install(); m.run(dispatch) }}\n}}"
    )
    .expect("write");

    // A hole inside a window must trap, not fall through.
    let words = vec![0x14000000; 300];
    let entries: Vec<u64> = (0..words.len())
        .filter(|i| *i != 64)
        .map(|i| CODE_BASE + i as u64 * 4)
        .collect();
    let translation = discover(&[region(&words)], &entries, 400).expect("sharded_hole");
    module(&mut out, "sharded_hole", &emit_rust(&translation));

    kernel_code(&mut out);
    semantic_code(&mut out);
    simd_semantic_code(&mut out);
    fp_semantic_code(&mut out);
    simd_memory_code(&mut out);

    let path = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR")).join("programs.rs");
    std::fs::write(path, out).expect("write generated programs");
}

/// The benchmark kernels, lifted together through split emission (as real
/// builds are) and discovered with their jump table.
fn kernel_code(out: &mut String) {
    let image = CodeRegion {
        name: "kernels".into(),
        base: kernels::KERNEL_BASE,
        bytes: kernels::image(),
    };
    let entries: Vec<u64> = kernels::KERNEL_ENTRIES
        .iter()
        .map(|(_, offset)| kernels::KERNEL_BASE + offset)
        .collect();
    let translation = discover_with_tables(
        std::slice::from_ref(&image),
        std::slice::from_ref(&image),
        &entries,
        &[],
        &Default::default(),
        4096,
    )
    .expect("lifting the kernels");
    let report = translation.report();
    assert!(
        report.unsupported.is_empty() && report.unresolved_targets.is_empty(),
        "kernels must lift completely: {:?} {:?}",
        report.unsupported,
        report.unresolved_targets
    );
    assert_eq!(report.jump_tables, 1, "the switch kernel's jump table");
    let (root, regions) = emit_rust_split(&translation, 10);
    writeln!(out, "pub mod kernel_code {{").expect("write");
    for (region, source) in regions {
        writeln!(out, "pub mod gen_r_{region:x} {{\n{source}\n}}").expect("write");
    }
    writeln!(out, "{root}\n}}").expect("write");
}

/// Instructions for the semantic differential test (`tests/semantics.rs`):
/// random operand fields of each class, each word followed by SVC #7 and
/// SVC #8 and entered at its own address. Branches jump 8 bytes, to the
/// SVC #8, so the SVC number tells whether they were taken.
fn semantic_code(out: &mut String) {
    use echora_lift::aarch64::{Address, Instruction, decode};

    const BASE: u64 = 0x40_0000;
    const PER_CLASS: usize = 96;
    // (name, mask of fixed bits, their value).
    const CLASSES: &[(&str, u32, u32)] = &[
        ("add/sub immediate", 0x1f80_0000, 0x1100_0000),
        ("add/sub shifted", 0x1f20_0000, 0x0b00_0000),
        ("add/sub extended", 0x1fe0_0000, 0x0b20_0000),
        ("adc/sbc", 0x1fe0_fc00, 0x1a00_0000),
        ("logical shifted", 0x1f00_0000, 0x0a00_0000),
        ("logical immediate", 0x1f80_0000, 0x1200_0000),
        ("move wide", 0x1f80_0000, 0x1280_0000),
        ("bitfield", 0x1f80_0000, 0x1300_0000),
        ("extr", 0x7fa0_0000, 0x1380_0000),
        ("conditional compare", 0x3fe0_0410, 0x3a40_0000),
        ("conditional select", 0x3fe0_0800, 0x1a80_0000),
        ("madd/msub", 0x7fe0_0000, 0x1b00_0000),
        ("long multiply", 0xff60_0000, 0x9b20_0000),
        ("multiply high", 0xff60_fc00, 0x9b40_7c00),
        ("divide and variable shift", 0x7fe0_c000, 0x1ac0_0000),
        ("one source", 0x7fff_f000, 0x5ac0_0000),
        ("load/store unsigned offset", 0x3f00_0000, 0x3900_0000),
        ("load/store unscaled, pre, post", 0x3f20_0000, 0x3800_0000),
        ("load/store register offset", 0x3f20_0c00, 0x3820_0800),
        ("load/store pair", 0x3e00_0000, 0x2800_0000),
        ("B.cond", 0xff00_0010, 0x5400_0000),
        ("CBZ/CBNZ", 0x7e00_0000, 0x3400_0000),
        ("TBZ/TBNZ", 0x7e00_0000, 0x3600_0000),
        ("CRC32", 0x7fe0_e000, 0x1ac0_4000),
        ("CAS", 0x3fa0_7c00, 0x08a0_7c00),
        ("CASP", 0xbfa0_7c00, 0x0820_7c00),
        ("LSE atomics", 0x3f20_0c00, 0x3820_0000),
        ("load/store exclusive", 0x3f80_7c00, 0x0800_7c00),
        ("LDAPR", 0x3fff_fc00, 0x38bf_c000),
        ("LDAR/STLR", 0x3f7f_fc00, 0x089f_fc00),
        ("MRS/MSR NZCV", 0xffdf_ffe0, 0xd51b_4200),
    ];
    let mut state: u64 = 0x5eed_0f5e_3a17_1c5d;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 16) as u32
    };
    let mut words = Vec::new();
    for &(_, mask, value) in CLASSES {
        let mut taken = 0;
        while taken < PER_CLASS {
            let mut word = (next() & !mask) | value;
            // Branch offsets of 8 bytes (imm19 or imm14 = 2).
            if word & 0x7c00_0000 == 0x3400_0000 || word & 0xff00_0010 == 0x5400_0000 {
                word = if word & 0x7e00_0000 == 0x3600_0000 {
                    (word & !0x0007_ffe0) | (2 << 5)
                } else {
                    (word & !0x00ff_ffe0) | (2 << 5)
                };
            }
            let usable = match decode(word, 0) {
                // The forms the reference interpreter covers.
                Instruction::AddSubImmediate { .. }
                | Instruction::AddSubShifted { .. }
                | Instruction::AddSubExtended { .. }
                | Instruction::AddCarry { .. }
                | Instruction::LogicalShifted { .. }
                | Instruction::LogicalImmediate { .. }
                | Instruction::MoveWide { .. }
                | Instruction::Bitfield { .. }
                | Instruction::Extract { .. }
                | Instruction::ConditionalCompare { .. }
                | Instruction::ConditionalCompareNegative { .. }
                | Instruction::Select { .. }
                | Instruction::MultiplyAdd { .. }
                | Instruction::UnsignedMultiplyAddLong { .. }
                | Instruction::MultiplyAddLong { .. }
                | Instruction::MultiplyHigh { .. }
                | Instruction::Divide { .. }
                | Instruction::VariableShift { .. }
                | Instruction::CountLeading { .. }
                | Instruction::ReverseBits { .. }
                | Instruction::ReverseBytes { .. }
                | Instruction::LoadStorePair { .. }
                | Instruction::BranchCond { .. }
                | Instruction::CompareBranch { .. }
                | Instruction::TestBranch { .. }
                | Instruction::Crc32 { .. }
                | Instruction::CompareAndSwap { .. }
                | Instruction::CompareAndSwapPair { .. }
                | Instruction::AtomicAdd { .. }
                | Instruction::AtomicMemory { .. }
                | Instruction::LoadAcquireRcpc { .. }
                | Instruction::LoadAcquireRcsc { .. }
                | Instruction::StoreRelease { .. }
                | Instruction::LoadExclusive { .. }
                | Instruction::StoreExclusive { .. }
                | Instruction::ReadNzcv { .. }
                | Instruction::WriteNzcv { .. }
                | Instruction::LoadStore {
                    address: Address::Immediate { .. },
                    ..
                } => true,
                // The index register must not also be the base, so both
                // can point where the test wants.
                Instruction::LoadStore {
                    rn,
                    address: Address::Register { rm, .. },
                    ..
                } => rm != rn,
                _ => false,
            };
            if usable {
                words.push(word);
                taken += 1;
            }
        }
    }
    let mut image = Vec::new();
    for word in &words {
        image.extend(word.to_le_bytes());
        image.extend(0xd400_00e1_u32.to_le_bytes()); // SVC #7
        image.extend(0xd400_0101_u32.to_le_bytes()); // SVC #8
    }
    let entries: Vec<u64> = (0..words.len()).map(|i| BASE + 12 * i as u64).collect();
    let region = CodeRegion {
        name: "semantic".into(),
        base: BASE,
        bytes: image,
    };
    let translation = discover(&[region], &entries, words.len() * 3 + 16).expect("semantic");
    let (root, regions) = emit_rust_split(&translation, 10);
    writeln!(out, "pub mod semantic_code {{").expect("write");
    writeln!(out, "pub const BASE: u64 = {BASE:#x};").expect("write");
    writeln!(out, "pub const WORDS: &[u32] = &{words:#x?};").expect("write");
    for (region, source) in regions {
        writeln!(out, "pub mod gen_r_{region:x} {{\n{source}\n}}").expect("write");
    }
    writeln!(out, "{root}\n}}").expect("write");
}

/// Integer SIMD and data-movement instructions for `tests/simd_semantics.rs`.
fn simd_semantic_code(out: &mut String) {
    use echora_lift::simd::SimdOp;

    let covered_simd = |op: &SimdOp| {
        matches!(
            op,
            SimdOp::VecInt { .. }
                | SimdOp::VecShift { .. }
                | SimdOp::VecIntElement { .. }
                | SimdOp::VecIntPairwise { .. }
                | SimdOp::VecLong { .. }
                | SimdOp::VecLongElement { .. }
                | SimdOp::VecNarrow { .. }
                | SimdOp::VecPairwiseLong { .. }
                | SimdOp::VecAcross { .. }
                | SimdOp::VecUnary { .. }
                | SimdOp::VecPermute { .. }
                | SimdOp::VecExt { .. }
                | SimdOp::VecTable { .. }
                | SimdOp::VecShiftLong { .. }
                | SimdOp::VecDot { .. }
                | SimdOp::VecPolyMulLong { .. }
                | SimdOp::VectorLogical { .. }
                | SimdOp::DupElement { .. }
                | SimdOp::DupGeneral { .. }
                | SimdOp::InsertElement { .. }
                | SimdOp::InsertGeneral { .. }
                | SimdOp::MoveToGeneral { .. }
                | SimdOp::MoveImmediate { .. }
                | SimdOp::ModifyImmediate { .. }
        )
    };
    let covered = |instruction: &echora_lift::aarch64::Instruction| matches!(instruction, echora_lift::aarch64::Instruction::Simd(op) if covered_simd(op));
    sampled_simd_program(
        out,
        "simd_semantic_code",
        0x60_0000,
        0x51d0_5e3a_9c2b_7e41,
        &[
            (0x0e00_0000, 0x0e00_0000),
            (0xdf00_0000, 0x5e00_0000),
            (0x9f20_0c00, 0x0e20_0800),
        ],
        &covered,
    );
}

/// Floating-point instructions for `tests/fp_semantics.rs`.
fn fp_semantic_code(out: &mut String) {
    use echora_lift::simd::{FpPrecision, SimdOp, VecFpOp, VecFpUnaryOp};

    let covered_simd = |op: &SimdOp| match *op {
        SimdOp::VecFp { op, .. } | SimdOp::VecFpElement { op, .. } => {
            !matches!(op, VecFpOp::RecipStep | VecFpOp::RSqrtStep)
        }
        SimdOp::VecFpUnary { op, .. } => !matches!(
            op,
            VecFpUnaryOp::RecipEstimate | VecFpUnaryOp::RSqrtEstimate
        ),
        SimdOp::VecFpConvert { from, to, .. } => from != FpPrecision::H && to != FpPrecision::H,
        SimdOp::FpUnary { .. }
        | SimdOp::FpConvert { .. }
        | SimdOp::FpBinary { .. }
        | SimdOp::FpTernary { .. }
        | SimdOp::FpCompare { .. }
        | SimdOp::FpConditionalCompare { .. }
        | SimdOp::FpSelect { .. }
        | SimdOp::FpMoveImmediate { .. }
        | SimdOp::FpMoveToGeneral { .. }
        | SimdOp::FpMoveFromGeneral { .. }
        | SimdOp::IntToFp { .. }
        | SimdOp::IntToFpFixed { .. }
        | SimdOp::SimdIntToFp { .. }
        | SimdOp::FpToInt { .. }
        | SimdOp::FpToFixed { .. }
        | SimdOp::VecFpPairwise { .. }
        | SimdOp::VecFpAcross { .. } => true,
        _ => false,
    };
    let covered = |instruction: &echora_lift::aarch64::Instruction| matches!(instruction, echora_lift::aarch64::Instruction::Simd(op) if covered_simd(op));
    sampled_simd_program(
        out,
        "fp_semantic_code",
        0x70_0000,
        0x0f1e_2d3c_4b5a_6978,
        &[
            (0x5f20_0000, 0x1e20_0000),
            (0x5f00_0000, 0x1f00_0000),
            (0x9f20_0400, 0x0e20_0400),
            (0xdf20_0400, 0x5e20_0400),
            (0x9f00_0400, 0x0f00_0000),
            (0x9f3e_0c00, 0x0e20_0800),
            (0xdf3e_0c00, 0x5e20_0800),
            (0x9f3e_0c00, 0x0e30_0800),
            (0xdf3e_0c00, 0x5e30_0800),
            (0x5f20_0000, 0x1e00_0000),
        ],
        &covered,
    );
}

/// Random words under `generators` (mask, fixed value) that decode to
/// instructions `covered` accepts, up to 160 per variant, each followed by
/// SVC #7 and SVC #8, lifted through split emission as module `name`.
fn sampled_simd_program(
    out: &mut String,
    name: &str,
    base: u64,
    seed: u64,
    generators: &[(u32, u32)],
    covered: &dyn Fn(&echora_lift::aarch64::Instruction) -> bool,
) {
    use echora_lift::aarch64::{Instruction, decode};

    const PER_VARIANT: usize = 160;
    let mut state = seed;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 16) as u32
    };
    let mut counts = std::collections::BTreeMap::<String, usize>::new();
    let mut words = Vec::new();
    for _ in 0..4_000_000 {
        let word = next();
        let (mask, value) = generators[next() as usize % generators.len()];
        let word = (word & !mask) | value;
        let instruction = decode(word, 0);
        if !covered(&instruction) {
            continue;
        }
        let variant = match &instruction {
            Instruction::Simd(op) => format!("{op:?}"),
            other => format!("{other:?}"),
        };
        let variant = variant
            .split([' ', '{'])
            .next()
            .unwrap_or_default()
            .to_owned();
        let count = counts.entry(variant).or_default();
        if *count < PER_VARIANT {
            *count += 1;
            words.push(word);
        }
    }
    let mut image = Vec::new();
    for word in &words {
        image.extend(word.to_le_bytes());
        image.extend(0xd400_00e1_u32.to_le_bytes()); // SVC #7
        image.extend(0xd400_0101_u32.to_le_bytes()); // SVC #8
    }
    let entries: Vec<u64> = (0..words.len()).map(|i| base + 12 * i as u64).collect();
    let region = CodeRegion {
        name: name.into(),
        base,
        bytes: image,
    };
    let translation = discover(&[region], &entries, words.len() * 3 + 16).expect(name);
    let (root, regions) = emit_rust_split(&translation, 10);
    writeln!(out, "pub mod {name} {{").expect("write");
    writeln!(out, "pub const BASE: u64 = {base:#x};").expect("write");
    writeln!(out, "pub const WORDS: &[u32] = &{words:#x?};").expect("write");
    for (region, source) in regions {
        writeln!(out, "pub mod gen_r_{region:x} {{\n{source}\n}}").expect("write");
    }
    writeln!(out, "{root}\n}}").expect("write");
}

/// SIMD loads and stores for `tests/simd_memory_semantics.rs`.
fn simd_memory_code(out: &mut String) {
    use echora_lift::aarch64::Instruction;
    use echora_lift::simd::SimdOp;

    let covered = |instruction: &Instruction| match instruction {
        Instruction::LoadStoreVectorPair { .. } => true,
        Instruction::Simd(op) => matches!(
            op,
            SimdOp::LoadStore { .. }
                | SimdOp::LoadStorePair { .. }
                | SimdOp::TransferMultiple { .. }
                | SimdOp::TransferSingle { .. }
                | SimdOp::LoadReplicate { .. }
        ),
        _ => false,
    };
    sampled_simd_program(
        out,
        "simd_memory_code",
        0x78_0000,
        0x6a09_e667_f3bc_c908,
        &[
            (0x3f00_0000, 0x3d00_0000),
            (0x3f20_0000, 0x3c00_0000),
            (0x3f20_0c00, 0x3c20_0800),
            (0x3e00_0000, 0x2c00_0000),
            (0xbf00_0000, 0x0c00_0000),
            (0xbf00_0000, 0x0d00_0000),
        ],
        &covered,
    );
}
