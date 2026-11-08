//! Lifts the synthetic programs into Rust modules compiled with this crate.

use std::fmt::Write as _;
use std::path::PathBuf;

use echora_lift::codegen::{
    CodeRegion, EmitOptions, discover, discover_with_tables, emit_rust, emit_rust_split_with,
};

thread_local! {
    /// The options of a `codesize` pass, which override the features.
    static MODE: std::cell::Cell<Option<EmitOptions>> = const { std::cell::Cell::new(None) };
    /// Guest instructions lifted through split emission so far.
    static LIFTED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Codegen options: common vector operations as native code with the
/// `inline-vector` feature, structured emission with `structured`.
fn options() -> EmitOptions {
    if let Some(mode) = MODE.get() {
        return mode;
    }
    EmitOptions {
        inline_vector: std::env::var_os("CARGO_FEATURE_INLINE_VECTOR").is_some(),
        structured: std::env::var_os("CARGO_FEATURE_STRUCTURED").is_some(),
        direct_calls: direct_calls(),
    }
}

/// Direct BL calls with the `direct-calls` feature.
fn direct_calls() -> bool {
    std::env::var_os("CARGO_FEATURE_DIRECT_CALLS").is_some()
}

/// Region size of split emission: one 1 KiB window per region, or with
/// direct calls 64 KiB, so that most calls between windows stay in one
/// region. With `small-regions` it stays 1 KiB, so that direct calls go
/// through the call slots of other regions.
fn region_bits() -> u32 {
    let small = std::env::var_os("CARGO_FEATURE_SMALL_REGIONS").is_some();
    if direct_calls() && !small { 16 } else { 10 }
}

/// The benchmark kernels' options: structured also with
/// `structured-kernels`, which builds far faster than `structured`.
fn kernel_options() -> EmitOptions {
    if let Some(mode) = MODE.get() {
        return mode;
    }
    EmitOptions {
        structured: std::env::var_os("CARGO_FEATURE_STRUCTURED_KERNELS").is_some(),
        ..options()
    }
}

/// Split emission of `translation`, counting its instructions.
fn split(
    translation: &echora_lift::codegen::Translation,
    options: EmitOptions,
) -> (String, Vec<(u64, String)>) {
    LIFTED.set(LIFTED.get() + translation.instructions.len());
    emit_rust_split_with(translation, region_bits(), options)
}

/// Every program lifted through split emission (what real builds use).
fn split_programs(out: &mut String) {
    kernel_code(out);
    semantic_code(out);
    simd_semantic_code(out);
    fp_semantic_code(out);
    simd_memory_code(out);
    sequence_code(out);
    call_code(out);
    loop_code(out);
    exclusive_code(out);
    switch_code(out);
    spin_candidates(out);
    irreducible_candidates(out);
    concurrency_code(out);
    stress_code(out);
}

/// With the `codesize` feature: the split programs once per emission mode,
/// each a crate of its own for `benches/codesize.rs` to compile, and a
/// list of `mode instructions bytes`.
fn codesize(out_dir: &std::path::Path) {
    let modes = [
        ("flat", EmitOptions::default()),
        (
            "structured",
            EmitOptions {
                structured: true,
                ..EmitOptions::default()
            },
        ),
        (
            "inline_vector",
            EmitOptions {
                inline_vector: true,
                ..EmitOptions::default()
            },
        ),
    ];
    let dir = out_dir.join("codesize");
    std::fs::create_dir_all(&dir).expect("codesize directory");
    let mut list = String::new();
    for (name, mode) in modes {
        MODE.set(Some(mode));
        LIFTED.set(0);
        let mut text = String::new();
        split_programs(&mut text);
        writeln!(list, "{name} {} {}", LIFTED.get(), text.len()).expect("write");
        let source = format!(
            "#[allow(unused_imports)]\nuse echora_guest::{{Core as Machine, Trap}};\n{text}"
        );
        std::fs::write(dir.join(format!("{name}.rs")), source).expect("write");
    }
    MODE.set(None);
    std::fs::write(dir.join("modes.txt"), list).expect("write");
}

include!("src/programs.rs");

mod kernels {
    include!("src/kernels.rs");
}

mod stress {
    include!("src/stress.rs");
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
    println!("cargo::rerun-if-changed=src/stress.rs");
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
    let (root, regions) = split(&translation, options());
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

    split_programs(&mut out);

    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR"));
    std::fs::write(out_dir.join("programs.rs"), out).expect("write generated programs");
    if std::env::var_os("CARGO_FEATURE_CODESIZE").is_some() {
        codesize(&out_dir);
    }
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
    let (root, regions) = split(&translation, kernel_options());
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
    let (root, regions) = split(&translation, options());
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
                | SimdOp::Crypto { .. }
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
        None,
    );
}

/// Floating-point instructions for `tests/fp_semantics.rs`.
fn fp_semantic_code(out: &mut String) {
    use echora_lift::simd::SimdOp;

    let covered_simd = |op: &SimdOp| {
        matches!(
            op,
            SimdOp::VecFp { .. }
                | SimdOp::VecFpElement { .. }
                | SimdOp::VecFpUnary { .. }
                | SimdOp::VecFpConvert { .. }
                | SimdOp::FpConvertPrecision { .. }
                | SimdOp::FpUnary { .. }
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
                | SimdOp::VecFpAcross { .. }
        )
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
        Some(24),
    );
}

/// Random words under `generators` (mask, fixed value) that decode to
/// instructions `covered` accepts, up to 160 per variant (or, with
/// `per_operation`, that many per variant and operation), each followed by
/// SVC #7 and SVC #8, lifted through split emission as module `name`.
fn sampled_simd_program(
    out: &mut String,
    name: &str,
    base: u64,
    seed: u64,
    generators: &[(u32, u32)],
    covered: &dyn Fn(&echora_lift::aarch64::Instruction) -> bool,
    per_operation: Option<usize>,
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
        let text = match &instruction {
            Instruction::Simd(op) => format!("{op:?}"),
            other => format!("{other:?}"),
        };
        // The variant, or with `per_operation` the variant and its `op`.
        let (key, cap) = match per_operation {
            Some(cap) => (text.split(',').next().unwrap_or_default().to_owned(), cap),
            None => (
                text.split([' ', '{']).next().unwrap_or_default().to_owned(),
                PER_VARIANT,
            ),
        };
        let count = counts.entry(key).or_default();
        if *count < cap {
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
    let (root, regions) = split(&translation, options());
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
        None,
    );
}

/// Random straight-line sequences of integer, flag and memory instructions
/// with forward branches, each ending in SVC #7, for `tests/sequences.rs`.
/// Memory instructions use X28 as their base and X27 as their index, and
/// no instruction writes either, so every access stays in the test's data.
fn sequence_code(out: &mut String) {
    use echora_lift::defuse::effects;

    const BASE: u64 = 0x90_0000;
    const SEQUENCES: usize = 600;
    let mut state: u64 = 0x2b7e_1516_28ae_d2a6;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 16) as u32
    };
    let reserved = |instruction: &echora_lift::aarch64::Instruction| {
        let writes = effects(instruction).writes;
        writes.has_gpr(27) || writes.has_gpr(28)
    };
    let mut words: Vec<u32> = Vec::new();
    let mut entries = Vec::new();
    for _ in 0..SEQUENCES {
        let length = 4 + next() as usize % 9;
        entries.push(BASE + 4 * words.len() as u64);
        let mut body = Vec::new();
        while body.len() < length {
            body.push(body_instruction(&mut next, &reserved));
        }
        // Forward branches to a later instruction of the sequence (or its
        // SVC): conditional, compare, test and unconditional.
        for index in 0..body.len() {
            if next() % 4 != 0 {
                continue;
            }
            let distance = 1 + next() % (body.len() - index) as u32;
            let register = next() & 31;
            let branch = match next() % 4 {
                0 => 0x5400_0000 | (next() & 15),
                1 => 0x3400_0000 | ((next() & 1) << 24) | ((next() & 1) << 31) | register,
                2 => {
                    let bit = next() & 63;
                    0x3600_0000
                        | ((bit >> 5) << 31)
                        | ((bit & 31) << 19)
                        | ((next() & 1) << 24)
                        | register
                }
                _ => 0x1400_0000,
            };
            // B takes imm26 in bits 25:0; the others imm19 or imm14 from bit 5.
            body[index] = if branch & 0x7c00_0000 == 0x1400_0000 {
                branch | distance
            } else {
                branch | (distance << 5)
            };
        }
        words.extend(body);
        words.push(0xd400_00e1);
    }
    let image: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let region = CodeRegion {
        name: "sequences".into(),
        base: BASE,
        bytes: image,
    };
    let translation = discover(&[region], &entries, words.len() + 16).expect("sequences");
    let (root, regions) = split(&translation, options());
    writeln!(out, "pub mod sequence_code {{").expect("write");
    writeln!(out, "pub const BASE: u64 = {BASE:#x};").expect("write");
    writeln!(out, "pub const WORDS: &[u32] = &{words:#x?};").expect("write");
    writeln!(out, "pub const ENTRIES: &[u64] = &{entries:#x?};").expect("write");
    for (region, source) in regions {
        writeln!(out, "pub mod gen_r_{region:x} {{\n{source}\n}}").expect("write");
    }
    writeln!(out, "{root}\n}}").expect("write");
}

// (mask of fixed bits, their value, has a base register in bits 9:5).
/// Classes the sequence and call programs draw from.
const BODY_CLASSES: &[(u32, u32, bool)] = &[
    (0x1f80_0000, 0x1100_0000, false),
    (0x1f20_0000, 0x0b00_0000, false),
    (0x1fe0_0000, 0x0b20_0000, false),
    (0x1fe0_fc00, 0x1a00_0000, false),
    (0x1f00_0000, 0x0a00_0000, false),
    (0x1f80_0000, 0x1200_0000, false),
    (0x1f80_0000, 0x1280_0000, false),
    (0x1f80_0000, 0x1300_0000, false),
    (0x7fa0_0000, 0x1380_0000, false),
    (0x3fe0_0410, 0x3a40_0000, false),
    (0x3fe0_0800, 0x1a80_0000, false),
    (0x7fe0_0000, 0x1b00_0000, false),
    (0xff60_0000, 0x9b20_0000, false),
    (0x7fe0_c000, 0x1ac0_0000, false),
    (0x7fff_f000, 0x5ac0_0000, false),
    (0x7fe0_e000, 0x1ac0_4000, false),
    (0xffdf_ffe0, 0xd51b_4200, false),
    (0x3f00_0000, 0x3900_0000, true),
    (0x3f20_0000, 0x3800_0000, true),
    (0x3f20_0c00, 0x3820_0800, true),
    (0x3e00_0000, 0x2800_0000, true),
    (0x3fa0_7c00, 0x08a0_7c00, true),
    (0x3f20_0c00, 0x3820_0000, true),
    (0x3f7f_fc00, 0x089f_fc00, true),
];

/// A random instruction from `BODY_CLASSES` that `reserved` does not
/// reject. Memory instructions use X28 as their base and X27 as their
/// index; exclusives (monitor behaviour is implementation defined) and
/// PACGA are left out.
fn body_instruction(
    next: &mut dyn FnMut() -> u32,
    reserved: &dyn Fn(&echora_lift::aarch64::Instruction) -> bool,
) -> u32 {
    use echora_lift::aarch64::{Address, Instruction, decode};
    loop {
        let (mask, value, memory) = BODY_CLASSES[next() as usize % BODY_CLASSES.len()];
        let mut word = (next() & !mask) | value;
        if memory {
            word = (word & !(31 << 5)) | (28 << 5);
        }
        if let Instruction::LoadStore {
            address: Address::Register { .. },
            ..
        } = decode(word, 0)
        {
            word = (word & !(31 << 16)) | (27 << 16);
        }
        let instruction = decode(word, 0);
        let usable = !matches!(
            instruction,
            Instruction::Unsupported
                | Instruction::Nop
                | Instruction::PacGeneric { .. }
                | Instruction::LoadExclusive { .. }
                | Instruction::StoreExclusive { .. }
        ) && !reserved(&instruction);
        if usable {
            return word;
        }
    }
}

/// Programs with calls for `tests/calls.rs`: 8 leaf functions, six in
/// windows of their own and two beside the callers, some calling a later
/// leaf behind an STP/LDP of X29 and X30 on the stack; and 300 callers of
/// random instructions with BL to a leaf, or BLR X26 after MOVZ/MOVK, in
/// about a third of their slots, each ending in SVC #7.
fn call_code(out: &mut String) {
    use echora_lift::defuse::effects;

    const BASE: u64 = 0xa0_0000;
    const LEAVES: usize = 8;
    const CALLERS: usize = 300;
    const PROLOGUE: u32 = 0xa9bf_7bfd; // stp x29, x30, [sp, #-16]!
    const EPILOGUE: u32 = 0xa8c1_7bfd; // ldp x29, x30, [sp], #16
    const RET: u32 = 0xd65f_03c0;
    let mut state: u64 = 0x3243_f6a8_885a_308d;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 16) as u32
    };
    // X26 holds BLR targets, X27/X28 the data index and base, X30 the
    // return address; SP is the stack.
    let reserved = |instruction: &echora_lift::aarch64::Instruction| {
        let writes = effects(instruction).writes;
        [26, 27, 28, 30].iter().any(|&n| writes.has_gpr(n)) || writes.sp
    };
    let bl = |from: u64, to: u64| {
        0x9400_0000 | (((to.wrapping_sub(from) as i64 / 4) as u32) & 0x03ff_ffff)
    };

    // Leaf bodies first (calls patched once addresses are known): each is
    // its words and the index of a BL to patch with its callee.
    type Leaf = (Vec<u32>, Option<(usize, usize)>);
    let mut leaves: Vec<Leaf> = Vec::new();
    for leaf in 0..LEAVES {
        let mut words = Vec::new();
        let nested = leaf + 1 < LEAVES && next() % 2 == 0;
        if nested {
            words.push(PROLOGUE);
        }
        for _ in 0..1 + next() % 5 {
            words.push(body_instruction(&mut next, &reserved));
        }
        let call = if nested {
            let callee = leaf + 1 + next() as usize % (LEAVES - leaf - 1);
            words.push(0);
            let at = words.len() - 1;
            for _ in 0..next() % 3 {
                words.push(body_instruction(&mut next, &reserved));
            }
            words.push(EPILOGUE);
            Some((at, callee))
        } else {
            None
        };
        words.push(RET);
        leaves.push((words, call));
    }
    let mut callers: Vec<Vec<u32>> = Vec::new();
    let mut calls = Vec::new(); // (caller, index, leaf, by register)
    for caller in 0..CALLERS {
        let mut words = Vec::new();
        for _ in 0..4 + next() % 9 {
            if next() % 3 == 0 {
                let leaf = next() as usize % LEAVES;
                let register = next() % 2 == 0;
                if register {
                    words.extend([0, 0]);
                }
                words.push(0);
                calls.push((caller, words.len() - 1, leaf, register));
            } else {
                words.push(body_instruction(&mut next, &reserved));
            }
        }
        words.push(0xd400_00e1);
        callers.push(words);
    }
    // Layout: leaves 0-5 one window each, then the callers, then leaves 6
    // and 7.
    let mut addresses = [0; LEAVES];
    for (leaf, address) in addresses.iter_mut().enumerate().take(6) {
        *address = BASE + 0x400 * leaf as u64;
    }
    let callers_base = BASE + 0x400 * 6;
    let mut cursor = callers_base;
    let mut caller_addresses = Vec::new();
    for words in &callers {
        caller_addresses.push(cursor);
        cursor += 4 * words.len() as u64;
    }
    for (leaf, address) in addresses.iter_mut().enumerate().skip(6) {
        *address = cursor;
        cursor += 4 * leaves[leaf].0.len() as u64;
    }
    for leaf in 0..LEAVES {
        if let Some((at, callee)) = leaves[leaf].1 {
            let from = addresses[leaf] + 4 * at as u64;
            leaves[leaf].0[at] = bl(from, addresses[callee]);
        }
    }
    for &(caller, at, leaf, register) in &calls {
        let from = caller_addresses[caller] + 4 * at as u64;
        let target = addresses[leaf];
        if register {
            let low = (target & 0xffff) as u32;
            let high = (target >> 16) as u32;
            callers[caller][at - 2] = 0xd280_0000 | (low << 5) | 26; // movz x26
            callers[caller][at - 1] = 0xf2a0_0000 | (high << 5) | 26; // movk x26, lsl 16
            callers[caller][at] = 0xd63f_0000 | (26 << 5); // blr x26
        } else {
            callers[caller][at] = bl(from, target);
        }
    }
    let mut image = vec![0u8; (cursor - BASE) as usize];
    let mut place = |address: u64, words: &[u32]| {
        for (index, word) in words.iter().enumerate() {
            let at = (address - BASE) as usize + 4 * index;
            image[at..at + 4].copy_from_slice(&word.to_le_bytes());
        }
    };
    for leaf in 0..LEAVES {
        place(addresses[leaf], &leaves[leaf].0);
    }
    for (caller, words) in callers.iter().enumerate() {
        place(caller_addresses[caller], words);
    }
    let words: Vec<u32> = image
        .chunks(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four bytes")))
        .collect();
    let region = CodeRegion {
        name: "calls".into(),
        base: BASE,
        bytes: image,
    };
    let translation = discover(&[region], &caller_addresses, words.len() + 16).expect("calls");
    let (root, regions) = split(&translation, options());
    writeln!(out, "pub mod call_code {{").expect("write");
    writeln!(out, "pub const BASE: u64 = {BASE:#x};").expect("write");
    writeln!(out, "pub const WORDS: &[u32] = &{words:#x?};").expect("write");
    writeln!(out, "pub const CALLERS: &[u64] = &{caller_addresses:#x?};").expect("write");
    for (region, source) in regions {
        writeln!(out, "pub mod gen_r_{region:x} {{\n{source}\n}}").expect("write");
    }
    writeln!(out, "{root}\n}}").expect("write");
}

/// Loop programs for `tests/loops.rs`: counted loops (X25) with a mid-loop
/// label that is also an entry, an ADR + BR to a later block reached only
/// through the register, and an ADRP + ADD + BLR into a leaf function,
/// either at its start or partway through its first block. Leaves sit in
/// the middle of windows of their own. Indirect targets are entries, as
/// relocation roots would be.
fn loop_code(out: &mut String) {
    use echora_lift::defuse::effects;

    const BASE: u64 = 0xb0_0000;
    const LEAVES: usize = 4;
    const PROGRAMS: usize = 300;
    const RET: u32 = 0xd65f_03c0;
    let mut state: u64 = 0x1319_8a2e_0370_7344;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 16) as u32
    };
    // X24 holds indirect targets, X25 the loop count, X26-X28 and X30 as
    // in the other programs; SP stays.
    let reserved = |instruction: &echora_lift::aarch64::Instruction| {
        let writes = effects(instruction).writes;
        [24, 25, 26, 27, 28, 30].iter().any(|&n| writes.has_gpr(n)) || writes.sp
    };
    let adr = |from: u64, to: u64| {
        let offset = to.wrapping_sub(from) as u32;
        0x1000_0000 | ((offset & 3) << 29) | (((offset >> 2) & 0x7ffff) << 5) | 24
    };
    let adrp = |from: u64, to: u64| {
        let pages = ((to >> 12).wrapping_sub(from >> 12)) as u32;
        0x9000_0000 | ((pages & 3) << 29) | (((pages >> 2) & 0x7ffff) << 5) | 24
    };
    let add_low = |to: u64| 0x9100_0000 | (((to & 0xfff) as u32) << 10) | (24 << 5) | 24;
    let cbnz = |from: u64, to: u64| {
        let offset = (to.wrapping_sub(from) as i64 / 4) as u32;
        0xb500_0000 | ((offset & 0x7ffff) << 5) | 25
    };

    let mut image: Vec<u32> = Vec::new();
    let mut entries = Vec::new();
    let place = |image: &mut Vec<u32>, address: u64, words: &[u32]| {
        let at = ((address - BASE) / 4) as usize;
        if image.len() < at + words.len() {
            image.resize(at + words.len(), 0);
        }
        image[at..at + words.len()].copy_from_slice(words);
    };
    // Leaves at offset 0x100 of windows 0..LEAVES, and their BLR targets:
    // the start, or the second or third instruction of the first block.
    let mut targets = Vec::new();
    for leaf in 0..LEAVES {
        let address = BASE + 0x400 * leaf as u64 + 0x100;
        let mut words: Vec<u32> = (0..3 + next() % 4)
            .map(|_| body_instruction(&mut next, &reserved))
            .collect();
        words.push(RET);
        place(&mut image, address, &words);
        for skip in 0..3 {
            targets.push(address + 4 * skip);
            entries.push(address + 4 * skip);
        }
    }
    let mut cursor = BASE + 0x400 * LEAVES as u64;
    let mut programs = Vec::new();
    for _ in 0..PROGRAMS {
        let start = cursor;
        programs.push(start);
        entries.push(start);
        let mut words = vec![0xd280_0000 | ((1 + next() % 5) << 5) | 25]; // movz x25, #n
        let head = start + 4 * words.len() as u64;
        for _ in 0..1 + next() % 3 {
            words.push(body_instruction(&mut next, &reserved));
        }
        // A mid-loop block start, entered by fall-through and as an entry.
        entries.push(start + 4 * words.len() as u64);
        for _ in 0..1 + next() % 3 {
            words.push(body_instruction(&mut next, &reserved));
        }
        words.push(0xd100_0739); // sub x25, x25, #1
        let at = start + 4 * words.len() as u64;
        words.push(cbnz(at, head));
        // ADR + BR over a block that never runs, to one reached only so.
        let adr_at = start + 4 * words.len() as u64;
        words.push(0);
        words.push(0xd61f_0000 | (24 << 5)); // br x24
        words.push(0); // UDF: never reached
        let landing = start + 4 * words.len() as u64;
        words[((adr_at - start) / 4) as usize] = adr(adr_at, landing);
        entries.push(landing);
        for _ in 0..1 + next() % 3 {
            words.push(body_instruction(&mut next, &reserved));
        }
        // ADRP + ADD + BLR into a leaf.
        let target = targets[next() as usize % targets.len()];
        let adrp_at = start + 4 * words.len() as u64;
        words.push(adrp(adrp_at, target));
        words.push(add_low(target));
        words.push(0xd63f_0000 | (24 << 5)); // blr x24
        words.push(body_instruction(&mut next, &reserved));
        words.push(0xd400_00e1);
        place(&mut image, start, &words);
        cursor += 4 * words.len() as u64;
    }
    let bytes: Vec<u8> = image.iter().flat_map(|word| word.to_le_bytes()).collect();
    let region = CodeRegion {
        name: "loops".into(),
        base: BASE,
        bytes,
    };
    let translation = discover(&[region], &entries, image.len() + 16).expect("loops");
    let (root, regions) = split(&translation, options());
    writeln!(out, "pub mod loop_code {{").expect("write");
    writeln!(out, "pub const BASE: u64 = {BASE:#x};").expect("write");
    writeln!(out, "pub const WORDS: &[u32] = &{image:#x?};").expect("write");
    writeln!(out, "pub const PROGRAMS: &[u64] = &{programs:#x?};").expect("write");
    writeln!(out, "pub const ENTRIES: &[u64] = &{entries:#x?};").expect("write");
    for (region, source) in regions {
        writeln!(out, "pub mod gen_r_{region:x} {{\n{source}\n}}").expect("write");
    }
    writeln!(out, "{root}\n}}").expect("write");
}

/// Sequences with exclusive pairs for `tests/exclusives.rs`: random
/// instructions with LDXR/LDAXR then STXR/STLXR on [X28] (sometimes of
/// another size, with CLREX or a plain store to [X28] between them, or
/// storing back the loaded value), each ending in SVC #7.
fn exclusive_code(out: &mut String) {
    use echora_lift::defuse::effects;

    const BASE: u64 = 0xc8_0000;
    const SEQUENCES: usize = 400;
    let mut state: u64 = 0xa409_3822_299f_31d0;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 16) as u32
    };
    let reserved = |instruction: &echora_lift::aarch64::Instruction| {
        let writes = effects(instruction).writes;
        writes.has_gpr(27) || writes.has_gpr(28)
    };
    // A data register other than X27/X28.
    let register = |next: &mut dyn FnMut() -> u32| loop {
        let n = next() % 31;
        if n != 27 && n != 28 {
            return n;
        }
    };
    let mut words: Vec<u32> = Vec::new();
    let mut entries = Vec::new();
    for _ in 0..SEQUENCES {
        entries.push(BASE + 4 * words.len() as u64);
        for _ in 0..1 + next() % 3 {
            for _ in 0..next() % 3 {
                words.push(body_instruction(&mut next, &reserved));
            }
            let size = next() % 4;
            let loaded = register(&mut next);
            let acquire = (next() & 1) << 15;
            words.push(0x085f_7c00 | (size << 30) | acquire | (28 << 5) | loaded);
            // Half the pairs span two blocks (B to the next instruction), so
            // a turn can end between the load and the store.
            if next() % 2 == 0 {
                words.push(0x1400_0001);
            }
            for _ in 0..next() % 3 {
                let word = match next() % 6 {
                    0 => 0xd503_305f,                                   // clrex
                    1 => 0xb900_0000 | (28 << 5) | register(&mut next), // str w, [x28]
                    _ => body_instruction(&mut next, &reserved),
                };
                words.push(word);
            }
            let store_size = if next() % 5 == 0 { next() % 4 } else { size };
            // A third store back the loaded register: the value does not
            // change, so only the global generation can fail another
            // context's pending pair.
            let store_back = next() % 3 == 0;
            let (status, value) = loop {
                let status = register(&mut next);
                let value = if store_back {
                    loaded
                } else {
                    register(&mut next)
                };
                if status != value {
                    break (status, value);
                }
            };
            let release = (next() & 1) << 15;
            words.push(
                0x0800_7c00 | (store_size << 30) | (status << 16) | release | (28 << 5) | value,
            );
        }
        words.push(0xd400_00e1);
    }
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let region = CodeRegion {
        name: "exclusives".into(),
        base: BASE,
        bytes,
    };
    let translation = discover(&[region], &entries, words.len() + 16).expect("exclusives");
    let (root, regions) = split(&translation, options());
    writeln!(out, "pub mod exclusive_code {{").expect("write");
    writeln!(out, "pub const BASE: u64 = {BASE:#x};").expect("write");
    writeln!(out, "pub const WORDS: &[u32] = &{words:#x?};").expect("write");
    writeln!(out, "pub const ENTRIES: &[u64] = &{entries:#x?};").expect("write");
    for (region, source) in regions {
        writeln!(out, "pub mod gen_r_{region:x} {{\n{source}\n}}").expect("write");
    }
    writeln!(out, "{root}\n}}").expect("write");
}

/// Programs with jump tables for `tests/switches.rs`: random instructions,
/// a 2-, 4- or 8-way switch on a masked register through a halfword or
/// byte table (ADRP + ADD to the table, ADR + ADD + BR to the case), case
/// bodies that branch to a merge point, more instructions and SVC #7. The
/// tables sit in the test's data, below every address the random stores
/// reach, and discovery reads them there.
fn switch_code(out: &mut String) {
    use echora_lift::defuse::effects;

    const BASE: u64 = 0xd0_0000;
    const TABLES: u64 = 0x80_0100;
    const PROGRAMS: usize = 200;
    let mut state: u64 = 0x082e_fa98_ec4e_6c89;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 16) as u32
    };
    let reserved = |instruction: &echora_lift::aarch64::Instruction| {
        let writes = effects(instruction).writes;
        writes.has_gpr(27) || writes.has_gpr(28)
    };
    let branch = |from: u64, to: u64| {
        0x1400_0000 | (((to.wrapping_sub(from) as i64 / 4) as u32) & 0x03ff_ffff)
    };
    let mut words: Vec<u32> = Vec::new();
    let mut table_bytes: Vec<u8> = Vec::new();
    let mut entries = Vec::new();
    for _ in 0..PROGRAMS {
        let start = BASE + 4 * words.len() as u64;
        entries.push(start);
        let at = |words: &Vec<u32>| BASE + 4 * words.len() as u64;
        for _ in 0..next() % 4 {
            words.push(body_instruction(&mut next, &reserved));
        }
        let bits = 1 + next() % 3;
        let cases = 1u32 << bits;
        let source = next() % 31;
        let halfwords = next() % 2 == 0;
        // Table address, aligned for halfwords.
        table_bytes.resize(table_bytes.len().next_multiple_of(2), 0);
        let table = TABLES + table_bytes.len() as u64;
        // and w17, wSource, #(cases - 1)
        words.push(0x1200_0000 | ((bits - 1) << 10) | (source << 5) | 17);
        let adrp_at = at(&words);
        let pages = ((table >> 12).wrapping_sub(adrp_at >> 12)) as u32;
        words.push(0x9000_0000 | ((pages & 3) << 29) | (((pages >> 2) & 0x7ffff) << 5) | 16);
        words.push(0x9100_0000 | (((table & 0xfff) as u32) << 10) | (16 << 5) | 16);
        words.push(if halfwords { 0x7871_7a09 } else { 0x3871_6a09 });
        let adr_at = at(&words);
        words.push(0); // adr x3, cases
        words.push(0x8b09_0863); // add x3, x3, x9, lsl #2
        words.push(0xd61f_0060); // br x3
        let cases_at = at(&words);
        let offset = cases_at - adr_at;
        words[((adr_at - BASE) / 4) as usize] = 0x1000_0000
            | (((offset & 3) as u32) << 29)
            | ((((offset >> 2) & 0x7ffff) as u32) << 5)
            | 3;
        // Case bodies; the table may send several indices to one case.
        let mut starts = Vec::new();
        let mut patches = Vec::new();
        let bodies = 1 + next() % cases;
        for _ in 0..bodies {
            starts.push(((at(&words) - cases_at) / 4) as u16);
            for _ in 0..1 + next() % 2 {
                words.push(body_instruction(&mut next, &reserved));
            }
            patches.push(words.len());
            words.push(0);
        }
        let merge = at(&words);
        for index in patches {
            let from = BASE + 4 * index as u64;
            words[index] = branch(from, merge);
        }
        for case in 0..cases {
            let target = starts[if case < bodies {
                case as usize
            } else {
                next() as usize % starts.len()
            }];
            if halfwords {
                table_bytes.extend(target.to_le_bytes());
            } else {
                table_bytes.push(target as u8);
            }
        }
        for _ in 0..1 + next() % 3 {
            words.push(body_instruction(&mut next, &reserved));
        }
        words.push(0xd400_00e1);
    }
    let code = CodeRegion {
        name: "switches".into(),
        base: BASE,
        bytes: words.iter().flat_map(|word| word.to_le_bytes()).collect(),
    };
    let tables = CodeRegion {
        name: "switch tables".into(),
        base: TABLES,
        bytes: table_bytes.clone(),
    };
    let translation = discover_with_tables(
        std::slice::from_ref(&code),
        std::slice::from_ref(&tables),
        &entries,
        &[],
        &Default::default(),
        words.len() + 16,
    )
    .expect("switches");
    assert_eq!(
        translation.jump_tables.len(),
        PROGRAMS,
        "every switch's table must be recovered"
    );
    let (root, regions) = split(&translation, options());
    writeln!(out, "pub mod switch_code {{").expect("write");
    writeln!(out, "pub const BASE: u64 = {BASE:#x};").expect("write");
    writeln!(out, "pub const WORDS: &[u32] = &{words:#x?};").expect("write");
    writeln!(out, "pub const ENTRIES: &[u64] = &{entries:#x?};").expect("write");
    writeln!(out, "pub const TABLES: u64 = {TABLES:#x};").expect("write");
    writeln!(out, "pub const TABLE_BYTES: &[u8] = &{table_bytes:?};").expect("write");
    for (region, source) in regions {
        writeln!(out, "pub mod gen_r_{region:x} {{\n{source}\n}}").expect("write");
    }
    writeln!(out, "{root}\n}}").expect("write");
}

/// Loop candidates for `tests/spin_semantics.rs` (not lifted): a body of
/// one to four instructions, often loads from [X28] and compares, then a
/// conditional branch back to the head, then SVC #7.
fn spin_candidates(out: &mut String) {
    use echora_lift::defuse::effects;

    const BASE: u64 = 0xe0_0000;
    const LOOPS: usize = 400;
    let mut state: u64 = 0x4528_21e6_38d0_1377;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 16) as u32
    };
    let reserved = |instruction: &echora_lift::aarch64::Instruction| {
        let writes = effects(instruction).writes;
        writes.has_gpr(27) || writes.has_gpr(28)
    };
    let mut words: Vec<u32> = Vec::new();
    let mut entries = Vec::new();
    for _ in 0..LOOPS {
        let head = BASE + 4 * words.len() as u64;
        entries.push(head);
        let probe = next() % 31;
        let probe = if probe == 27 || probe == 28 { 5 } else { probe };
        for _ in 0..1 + next() % 4 {
            let word = match next() % 5 {
                // ldr wProbe, [x28, #8 * k]
                0 | 1 => 0xb940_0000 | ((next() % 64) << 10) | (28 << 5) | probe,
                // cmp wProbe, #imm
                2 => 0x7100_001f | ((next() % 4096) << 10) | (probe << 5),
                _ => body_instruction(&mut next, &reserved),
            };
            words.push(word);
        }
        let at = BASE + 4 * words.len() as u64;
        let back = (((head as i64 - at as i64) / 4) as u32) & 0x7ffff;
        let branch = match next() % 3 {
            0 => 0x5400_0000 | (back << 5) | (next() % 14),
            1 => 0x3400_0000 | ((next() & 1) << 24) | (back << 5) | probe,
            _ => {
                let bit = next() % 32;
                0x3600_0000 | (bit << 19) | ((next() & 1) << 24) | ((back & 0x3fff) << 5) | probe
            }
        };
        words.push(branch);
        words.push(0xd400_00e1);
    }
    // SVC returns, so the detector explores past each candidate's SVC; a
    // final RET keeps the last ones from running off the end of the code.
    words.push(0xd65f_03c0);
    writeln!(out, "pub mod spin_candidates {{").expect("write");
    writeln!(out, "pub const BASE: u64 = {BASE:#x};").expect("write");
    writeln!(out, "pub const WORDS: &[u32] = &{words:#x?};").expect("write");
    writeln!(out, "pub const ENTRIES: &[u64] = &{entries:#x?};").expect("write");
    writeln!(out, "}}").expect("write");
}

/// Irreducible loops with two exits for `tests/structure_semantics.rs` (not
/// lifted): a two-entry loop {A, B} counted in X25, entered at A or B,
/// left by A's conditional branch to one tail or by B's fall-through to
/// another.
fn irreducible_candidates(out: &mut String) {
    use echora_lift::defuse::effects;

    const BASE: u64 = 0xf0_0000;
    const PROGRAMS: usize = 300;
    let mut state: u64 = 0xc0ac_29b7_c97c_50dd;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 16) as u32
    };
    let reserved = |instruction: &echora_lift::aarch64::Instruction| {
        let writes = effects(instruction).writes;
        [25, 27, 28].iter().any(|&n| writes.has_gpr(n))
    };
    let probe = |next: &mut dyn FnMut() -> u32| loop {
        let n = next() % 31;
        if ![25, 27, 28].contains(&n) {
            return n;
        }
    };
    let conditional = |from: u64, to: u64, register: u32, nonzero: u32| {
        let offset = (((to as i64 - from as i64) / 4) as u32) & 0x7ffff;
        0xb400_0000 | (nonzero << 24) | (offset << 5) | register
    };
    let mut words: Vec<u32> = Vec::new();
    let mut entries = Vec::new();
    let at = |words: &Vec<u32>| BASE + 4 * words.len() as u64;
    for _ in 0..PROGRAMS {
        entries.push(at(&words));
        words.push(0xd280_0000 | ((1 + next() % 4) << 5) | 25); // movz x25, #n
        let enter_b = words.len();
        words.push(0); // cbz/cbnz probe, B
        let a = at(&words);
        for _ in 0..1 + next() % 3 {
            words.push(body_instruction(&mut next, &reserved));
        }
        let exit_a = words.len();
        words.push(0); // cbz/cbnz probe, tail 1
        let b = at(&words);
        for _ in 0..1 + next() % 3 {
            words.push(body_instruction(&mut next, &reserved));
        }
        words.push(0xd100_0739); // sub x25, x25, #1
        let back = at(&words);
        words.push(0xb500_0000 | (((((a as i64 - back as i64) / 4) as u32) & 0x7ffff) << 5) | 25);
        for _ in 0..next() % 3 {
            words.push(body_instruction(&mut next, &reserved));
        }
        words.push(0xd400_00e1);
        let tail = at(&words);
        for _ in 0..1 + next() % 3 {
            words.push(body_instruction(&mut next, &reserved));
        }
        words.push(0xd400_00e1);
        let (first, second) = (probe(&mut next), probe(&mut next));
        let from = BASE + 4 * enter_b as u64;
        words[enter_b] = conditional(from, b, first, next() & 1);
        let from = BASE + 4 * exit_a as u64;
        words[exit_a] = conditional(from, tail, second, next() & 1);
    }
    words.push(0xd65f_03c0);
    writeln!(out, "pub mod irreducible_candidates {{").expect("write");
    writeln!(out, "pub const BASE: u64 = {BASE:#x};").expect("write");
    writeln!(out, "pub const WORDS: &[u32] = &{words:#x?};").expect("write");
    writeln!(out, "pub const ENTRIES: &[u64] = &{entries:#x?};").expect("write");
    writeln!(out, "}}").expect("write");
}

/// Concurrency-sensitive patterns for `tests/concurrency.rs`, each at the
/// start of its own window: plain loads and stores that another host
/// thread changes or watches. X0 and X1 hold data addresses.
fn concurrency_code(out: &mut String) {
    const BASE: u64 = 0x100_0000;
    const WINDOW: usize = 0x400 / 4;
    const LDR_W1_X0: u32 = 0xb940_0001;
    const LDR_W2_X0: u32 = 0xb940_0002;
    const LDR_W3_X1: u32 = 0xb940_0023;
    const STR_W2_X0: u32 = 0xb900_0002;
    const ADD_X2_1: u32 = 0x9100_0442;
    const SVC_7: u32 = 0xd400_00e1;
    let programs: [&[u32]; 4] = [
        // poll: LDR W1, [X0]; CBZ W1, back to the LDR.
        &[LDR_W1_X0, 0x34ff_ffe1, SVC_7],
        // counted poll: LDR W1, [X0]; ADD X2, X2, #1; CBZ W1, back to the
        // LDR. The count makes it no pure poll (no spin hint).
        &[LDR_W1_X0, ADD_X2_1, 0x34ff_ffc1, SVC_7],
        // store then load: STR W2, [X0]; LDR W3, [X1].
        &[STR_W2_X0, LDR_W3_X1, SVC_7],
        // two loads of one address: LDR W1, [X0]; LDR W2, [X0].
        &[LDR_W1_X0, LDR_W2_X0, SVC_7],
    ];
    let mut words = vec![0u32; WINDOW * programs.len()];
    let mut entries = Vec::new();
    for (index, program) in programs.iter().enumerate() {
        words[index * WINDOW..index * WINDOW + program.len()].copy_from_slice(program);
        entries.push(BASE + (index * WINDOW * 4) as u64);
    }
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let region = CodeRegion {
        name: "concurrency".into(),
        base: BASE,
        bytes,
    };
    let translation = discover(&[region], &entries, words.len()).expect("concurrency");
    let (root, regions) = split(&translation, options());
    writeln!(out, "pub mod concurrency_code {{").expect("write");
    writeln!(out, "pub const BASE: u64 = {BASE:#x};").expect("write");
    writeln!(out, "pub const WORDS: &[u32] = &{words:#x?};").expect("write");
    writeln!(out, "pub const ENTRIES: &[u64] = &{entries:#x?};").expect("write");
    for (region, source) in regions {
        writeln!(out, "pub mod gen_r_{region:x} {{\n{source}\n}}").expect("write");
    }
    writeln!(out, "{root}\n}}").expect("write");
}

/// The concurrency stress programs of `tests/stress.rs` (`src/stress.rs`).
fn stress_code(out: &mut String) {
    let bytes: Vec<u8> = stress::STRESS_WORDS
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    let region = CodeRegion {
        name: "stress".into(),
        base: stress::STRESS_BASE,
        bytes,
    };
    let entries: Vec<u64> = stress::STRESS_ENTRIES
        .iter()
        .map(|(_, offset)| stress::STRESS_BASE + offset)
        .collect();
    let translation = discover(&[region], &entries, stress::STRESS_WORDS.len()).expect("stress");
    let report = translation.report();
    assert!(
        report.unsupported.is_empty(),
        "stress programs must lift completely: {:?}",
        report.unsupported
    );
    let (root, regions) = split(&translation, options());
    writeln!(out, "pub mod stress_code {{").expect("write");
    for (region, source) in regions {
        writeln!(out, "pub mod gen_r_{region:x} {{\n{source}\n}}").expect("write");
    }
    writeln!(out, "{root}\n}}").expect("write");
}
