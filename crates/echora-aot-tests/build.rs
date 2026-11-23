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
    /// Dead writes over the programs lifted so far (`echora_lift::liveness`).
    static DEAD: std::cell::Cell<echora_lift::liveness::DeadWrites> =
        const { std::cell::Cell::new(echora_lift::liveness::DeadWrites {
            instructions: 0,
            flag_writes: 0,
            dead_flag_writes: 0,
            register_writes: 0,
            dead_register_writes: 0,
            dead_instructions: 0,
        }) };
}

/// Codegen options: common vector operations as native code with the
/// `inline-vector` feature, structured emission with `structured`, flag
/// setters fused with their B.cond with `fused-flags`.
fn options() -> EmitOptions {
    if let Some(mode) = MODE.get() {
        return mode;
    }
    EmitOptions {
        inline_vector: std::env::var_os("CARGO_FEATURE_INLINE_VECTOR").is_some(),
        structured: std::env::var_os("CARGO_FEATURE_STRUCTURED").is_some(),
        fused_flags: std::env::var_os("CARGO_FEATURE_FUSED_FLAGS").is_some(),
        whole_vector_checks: std::env::var_os("CARGO_FEATURE_WHOLE_VECTOR_CHECKS").is_some(),
        inline_vector_unary: std::env::var_os("CARGO_FEATURE_INLINE_VECTOR_UNARY").is_some(),
        inline_vector_int: std::env::var_os("CARGO_FEATURE_INLINE_VECTOR_INT").is_some(),
        inline_vector_pairwise: std::env::var_os("CARGO_FEATURE_INLINE_VECTOR_PAIRWISE").is_some(),
        inline_fp_scalar: std::env::var_os("CARGO_FEATURE_INLINE_FP_SCALAR").is_some(),
        inline_vector_misc: std::env::var_os("CARGO_FEATURE_INLINE_VECTOR_MISC").is_some(),
        count_fallbacks: std::env::var_os("CARGO_FEATURE_COUNT_FALLBACKS").is_some(),
        // A build flag, not an emission choice: these programs are built
        // by this crate's own flags.
        align_windows: false,
        census: false,
        direct_q_pairs: std::env::var_os("CARGO_FEATURE_DIRECT_Q_PAIRS").is_some(),
        vector_float_runs: std::env::var_os("CARGO_FEATURE_VECTOR_FLOAT_RUNS").is_some(),
        fp_status_locals: std::env::var_os("CARGO_FEATURE_FP_STATUS_LOCALS").is_some(),
        window_memory: std::env::var_os("CARGO_FEATURE_WINDOW_MEMORY").is_some(),
        dense_dispatch: std::env::var_os("CARGO_FEATURE_DENSE_DISPATCH").is_some(),
        untagged_blocks: std::env::var_os("CARGO_FEATURE_UNTAGGED_BLOCKS").is_some(),
        split_ticks: std::env::var_os("CARGO_FEATURE_SPLIT_TICKS").is_some(),
        direct_calls: std::env::var_os("CARGO_FEATURE_DIRECT_CALLS").is_some(),
        functions: std::env::var_os("CARGO_FEATURE_FUNCTIONS").is_some(),
        function_table: std::env::var_os("CARGO_FEATURE_FUNCTION_TABLE").is_some(),
        steps_local: std::env::var_os("CARGO_FEATURE_STEPS_LOCAL").is_some(),
        call_summaries: std::env::var_os("CARGO_FEATURE_CALL_SUMMARIES").is_some(),
        transitive_summaries: std::env::var_os("CARGO_FEATURE_TRANSITIVE_SUMMARIES").is_some(),
        float_runs: std::env::var_os("CARGO_FEATURE_FLOAT_RUNS").is_some(),
        float_run_loads: std::env::var_os("CARGO_FEATURE_FLOAT_RUN_LOADS").is_some(),
        deferred_charges: std::env::var_os("CARGO_FEATURE_DEFERRED_CHARGES").is_some(),
        window_calls: std::env::var_os("CARGO_FEATURE_WINDOW_CALLS").is_some(),
        window_call_early_exits: std::env::var_os("CARGO_FEATURE_WINDOW_CALL_EARLY_EXITS")
            .is_some(),
        window_calls_audit: std::env::var_os("CARGO_FEATURE_WINDOW_CALLS_AUDIT").is_some(),
        long_windows: std::env::var_os("CARGO_FEATURE_LONG_WINDOWS").is_some(),
        known_returns: std::env::var_os("CARGO_FEATURE_KNOWN_RETURNS").is_some(),
        cold_fp_exact: std::env::var_os("CARGO_FEATURE_COLD_FP_EXACT").is_some(),
        hoist_fp_mode: std::env::var_os("CARGO_FEATURE_HOIST_FP_MODE").is_some(),
        call_summaries_audit: std::env::var_os("CARGO_FEATURE_CALL_SUMMARIES_AUDIT").is_some(),
        threaded_successors: std::env::var_os("CARGO_FEATURE_THREADED_SUCCESSORS").is_some(),
        cached_stop: std::env::var_os("CARGO_FEATURE_CACHED_STOP").is_some(),
        arm_charges: std::env::var_os("CARGO_FEATURE_ARM_CHARGES").is_some(),
        regalloc: std::env::var_os("CARGO_FEATURE_REGALLOC").is_some(),
        stub_windows: false,
    }
}

/// Region size of split emission: one 1 KiB window per region; with
/// `long-windows` (outside the golden modes) 64 KiB regions, as the game's,
/// so windows can hold blocks of their neighbours.
fn region_bits() -> u32 {
    if MODE.get().is_none() && std::env::var_os("CARGO_FEATURE_LONG_WINDOWS").is_some() {
        16
    } else {
        10
    }
}

/// The benchmark kernels' options: structured also with
/// `structured-kernels`, which builds far faster than `structured`.
fn kernel_options() -> EmitOptions {
    if let Some(mode) = MODE.get() {
        return mode;
    }
    EmitOptions {
        structured: std::env::var_os("CARGO_FEATURE_STRUCTURED_KERNELS").is_some(),
        // Markers for `KERNEL_CENSUS` (comments: the kernels compile the
        // same).
        census: true,
        ..options()
    }
}

/// Split emission of `translation`, counting its instructions.
fn split(
    translation: &echora_lift::codegen::Translation,
    options: EmitOptions,
) -> (String, Vec<(u64, String)>) {
    split_at(translation, options, region_bits())
}

/// [`split`] in regions of `region_bits`.
fn split_at(
    translation: &echora_lift::codegen::Translation,
    options: EmitOptions,
    region_bits: u32,
) -> (String, Vec<(u64, String)>) {
    LIFTED.set(LIFTED.get() + translation.instructions.len());
    if std::env::var_os("CARGO_FEATURE_CODESIZE").is_some() {
        let add = echora_lift::liveness::dead_writes(translation);
        let mut dead = DEAD.get();
        dead.instructions += add.instructions;
        dead.flag_writes += add.flag_writes;
        dead.dead_flag_writes += add.dead_flag_writes;
        dead.register_writes += add.register_writes;
        dead.dead_register_writes += add.dead_register_writes;
        dead.dead_instructions += add.dead_instructions;
        DEAD.set(dead);
    }
    emit_rust_split_with(translation, region_bits, options)
}

/// Split emission of `translation` with `listed` for every other 1 KiB
/// region (the even ones) and `base` for the rest, counting its
/// instructions: options that overlap on the same forms are tried both
/// with and without each other in one lift.
fn split_alternate(
    translation: &echora_lift::codegen::Translation,
    base: EmitOptions,
    listed: EmitOptions,
) -> (String, Vec<(u64, String)>) {
    let regions: std::collections::BTreeSet<u64> = translation
        .blocks
        .iter()
        .map(|block| block.start >> region_bits())
        .filter(|region| region % 2 == 0)
        .collect();
    LIFTED.set(LIFTED.get() + translation.instructions.len());
    let mut sources = Vec::new();
    let root = echora_lift::codegen::emit_rust_split_to_regions(
        translation,
        region_bits(),
        base,
        listed,
        &regions,
        &mut |region, source, _| {
            sources.push((region, source));
            Ok::<(), std::convert::Infallible>(())
        },
    )
    .expect("no sink errors");
    (root, sources)
}

/// Every program lifted through split emission (what real builds use).
fn split_programs(out: &mut String) {
    for (_, generate) in SPLIT_PROGRAMS {
        generate(out);
    }
}

/// The split programs' lifts, in the golden hashes' order, by the file of
/// `OUT_DIR` each is written to (`lifted_<name>.rs`, see `main`): a test
/// includes only the lifts it runs, so no one compilation holds them all
/// (one crate with every lift took 9 GB to check). `kernel` goes into the
/// library (`programs.rs`): the benchmarks and several tests use it.
type Generator = fn(&mut String);
const SPLIT_PROGRAMS: &[(&str, Generator)] = &[
    ("kernel", kernel_code),
    ("semantic", semantic_code),
    ("simd_semantic", simd_semantic_code),
    ("fp_semantic", fp_semantic_code),
    ("simd_memory", simd_memory_code),
    ("sequence", sequence_code),
    ("call", call_code),
    ("loop", loop_code),
    ("exclusive", exclusive_code),
    ("switch", switch_code),
    ("spin_candidates", spin_candidates),
    ("irreducible_candidates", irreducible_candidates),
    ("concurrency", concurrency_code),
    ("stress", stress_code),
    ("plt", plt_code),
    ("spill", spill_code),
    ("float_run", float_run_code),
    ("float_run_load", float_run_load_code),
    ("deferred_charge", deferred_charge_code),
    ("early_exit", early_exit_code),
    ("two_level", two_level_code),
    ("svc_wrapper", svc_wrapper_code),
];

/// Lifted outside the golden hashes, each to its own file as
/// `SPLIT_PROGRAMS`.
const OTHER_PROGRAMS: &[(&str, Generator)] = &[
    ("flag", flag_code),
    ("fp_status", fp_status_code),
    ("vector_unary", vector_unary_code),
    ("vector_int", vector_int_code),
    ("fp_scalar", fp_scalar_code),
    ("vector_misc", vector_misc_code),
    ("fallback_count", fallback_count_code),
    ("q_pair", q_pair_code),
    ("vector_run", vector_run_code),
    ("rounded_up", rounded_up_code),
    ("window_call", window_call_code),
    ("smallest_normal", smallest_normal_code),
    ("memory_fault", memory_fault_code),
    ("fp_mode_group", fp_mode_group_code),
];

/// Hashes of the split programs' emitted Rust with the default options and
/// with inline_vector alone (what game builds use), whatever this build's
/// features: `tests/golden.rs` checks them against the checked-in values,
/// so a change to default output is a deliberate one.
fn golden(out_dir: &std::path::Path) {
    let fnv = |text: &str| {
        text.bytes().fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    };
    let mut hashes = Vec::new();
    for mode in [
        EmitOptions::default(),
        EmitOptions {
            inline_vector: true,
            ..EmitOptions::default()
        },
    ] {
        MODE.set(Some(mode));
        let lifted = LIFTED.get();
        let mut text = String::new();
        split_programs(&mut text);
        LIFTED.set(lifted);
        hashes.push(fnv(&text));
    }
    MODE.set(None);
    std::fs::write(
        out_dir.join("golden.rs"),
        format!(
            "/// Hash of the default emission of the split programs.\n\
             pub const DEFAULT_OUTPUT: u64 = {:#018x};\n\
             /// Hash of their emission with inline_vector alone.\n\
             pub const INLINE_VECTOR_OUTPUT: u64 = {:#018x};\n",
            hashes[0], hashes[1]
        ),
    )
    .expect("write golden hashes");
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
        (
            "fused_flags",
            EmitOptions {
                fused_flags: true,
                ..EmitOptions::default()
            },
        ),
        (
            "whole_vector",
            EmitOptions {
                inline_vector: true,
                whole_vector_checks: true,
                ..EmitOptions::default()
            },
        ),
        (
            "fp_locals",
            EmitOptions {
                inline_vector: true,
                whole_vector_checks: true,
                fp_status_locals: true,
                ..EmitOptions::default()
            },
        ),
        // The set docs/codegen-speed.md recommends for the next rebuild.
        (
            "recommended",
            EmitOptions {
                inline_vector: true,
                fused_flags: true,
                whole_vector_checks: true,
                fp_status_locals: true,
                ..EmitOptions::default()
            },
        ),
    ];
    // The game set, and the game set plus each option
    // `docs/lift-options.md` weighs for the next relift.
    let game = game_set();
    let modes: Vec<(&str, EmitOptions)> = modes
        .into_iter()
        .chain([
            ("game", game),
            (
                "game_window_calls",
                EmitOptions {
                    window_calls: true,
                    ..game
                },
            ),
            (
                "game+unary",
                EmitOptions {
                    inline_vector_unary: true,
                    ..game
                },
            ),
            (
                "game+int",
                EmitOptions {
                    inline_vector_int: true,
                    ..game
                },
            ),
            (
                "game+pairwise",
                EmitOptions {
                    inline_vector_pairwise: true,
                    ..game
                },
            ),
            (
                "game+fp_scalar",
                EmitOptions {
                    inline_fp_scalar: true,
                    ..game
                },
            ),
            (
                "game+misc",
                EmitOptions {
                    inline_vector_misc: true,
                    ..game
                },
            ),
            (
                "game+q_pairs",
                EmitOptions {
                    direct_q_pairs: true,
                    ..game
                },
            ),
            (
                "game+vector_runs",
                EmitOptions {
                    vector_float_runs: true,
                    ..game
                },
            ),
            (
                "game+run_loads",
                EmitOptions {
                    float_run_loads: true,
                    ..game
                },
            ),
            (
                "game+steps_local",
                EmitOptions {
                    steps_local: true,
                    ..game
                },
            ),
            (
                "game+functions_all",
                EmitOptions {
                    functions: true,
                    function_table: true,
                    call_summaries: true,
                    transitive_summaries: true,
                    ..game
                },
            ),
            (
                "game+window_calls",
                EmitOptions {
                    window_calls: true,
                    ..game
                },
            ),
            (
                "game+long_windows",
                EmitOptions {
                    long_windows: true,
                    ..game
                },
            ),
            (
                "game+known_returns",
                EmitOptions {
                    known_returns: true,
                    ..game
                },
            ),
            ("game+recommended", recommended_set()),
            ("game+recommended+windows", recommended_windows_set()),
        ])
        .collect();
    let dir = out_dir.join("codesize");
    std::fs::create_dir_all(&dir).expect("codesize directory");
    let mut list = String::new();
    for (name, mode) in modes.iter().copied() {
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
    kernel_sizes(&dir);
    // Each mode lifted the same programs: one mode's share.
    let dead = DEAD.get();
    let per_mode = |count: usize| count / modes.len();
    std::fs::write(
        dir.join("liveness.txt"),
        format!(
            "{} instructions: {} of {} NZCV writes dead, {} of {} register writes dead, \
             {} instructions with nothing but dead writes\n",
            per_mode(dead.instructions),
            per_mode(dead.dead_flag_writes),
            per_mode(dead.flag_writes),
            per_mode(dead.dead_register_writes),
            per_mode(dead.register_writes),
            per_mode(dead.dead_instructions),
        ),
    )
    .expect("write");
}

/// The game build's default options (`echora lift`'s `DEFAULT_OPTIONS`).
fn game_set() -> EmitOptions {
    EmitOptions {
        inline_vector: true,
        fused_flags: true,
        whole_vector_checks: true,
        fp_status_locals: true,
        window_memory: true,
        untagged_blocks: true,
        split_ticks: true,
        float_runs: true,
        ..EmitOptions::default()
    }
}

/// The game set plus the options `docs/lift-options.md` recommends turning
/// on (everywhere or in the hot regions), but for the window calls.
fn recommended_set() -> EmitOptions {
    EmitOptions {
        inline_vector_unary: true,
        inline_vector_int: true,
        inline_vector_pairwise: true,
        inline_fp_scalar: true,
        inline_vector_misc: true,
        direct_q_pairs: true,
        vector_float_runs: true,
        steps_local: true,
        ..game_set()
    }
}

/// The recommended set plus the window-call options
/// `docs/lift-options.md` recommends for the hot regions.
fn recommended_windows_set() -> EmitOptions {
    EmitOptions {
        window_calls: true,
        long_windows: true,
        known_returns: true,
        ..recommended_set()
    }
}

/// With the `codesize` feature: the generated Rust of the benchmark
/// kernels (`kernel_code`) with the game set and with the game set plus
/// each opt-in option (and what it needs), as `name bytes lines` in
/// `kernels.txt` for `benches/codesize.rs` to print.
fn kernel_sizes(dir: &std::path::Path) {
    let game = game_set();
    let functions = EmitOptions {
        functions: true,
        ..game
    };
    let table = EmitOptions {
        function_table: true,
        ..functions
    };
    let summaries = EmitOptions {
        call_summaries: true,
        ..table
    };
    let options: Vec<(&str, EmitOptions)> = vec![
        ("game-set", game),
        (
            "inline-vector-unary",
            EmitOptions {
                inline_vector_unary: true,
                ..game
            },
        ),
        (
            "inline-vector-int",
            EmitOptions {
                inline_vector_int: true,
                ..game
            },
        ),
        (
            "inline-vector-pairwise",
            EmitOptions {
                inline_vector_pairwise: true,
                ..game
            },
        ),
        (
            "inline-fp-scalar",
            EmitOptions {
                inline_fp_scalar: true,
                ..game
            },
        ),
        (
            "inline-vector-misc",
            EmitOptions {
                inline_vector_misc: true,
                ..game
            },
        ),
        (
            "direct-q-pairs",
            EmitOptions {
                direct_q_pairs: true,
                ..game
            },
        ),
        (
            "vector-float-runs",
            EmitOptions {
                vector_float_runs: true,
                ..game
            },
        ),
        (
            "float-run-loads",
            EmitOptions {
                float_run_loads: true,
                ..game
            },
        ),
        (
            "steps-local",
            EmitOptions {
                steps_local: true,
                ..game
            },
        ),
        (
            "deferred-charges",
            EmitOptions {
                deferred_charges: true,
                ..game
            },
        ),
        ("functions", functions),
        ("+function-table", table),
        ("+call-summaries", summaries),
        (
            "+transitive-summaries",
            EmitOptions {
                transitive_summaries: true,
                ..summaries
            },
        ),
        (
            "direct-calls",
            EmitOptions {
                direct_calls: true,
                ..game
            },
        ),
        (
            "dense-dispatch",
            EmitOptions {
                dense_dispatch: true,
                ..game
            },
        ),
        (
            "stub-windows",
            EmitOptions {
                stub_windows: true,
                ..game
            },
        ),
        (
            "structured",
            EmitOptions {
                structured: true,
                ..game
            },
        ),
        (
            "count-fallbacks",
            EmitOptions {
                count_fallbacks: true,
                ..game
            },
        ),
        (
            "window-calls",
            EmitOptions {
                window_calls: true,
                ..game
            },
        ),
        (
            "+window-call-early-exits",
            EmitOptions {
                window_calls: true,
                window_call_early_exits: true,
                ..game
            },
        ),
        (
            "long-windows",
            EmitOptions {
                long_windows: true,
                ..game
            },
        ),
        (
            "known-returns",
            EmitOptions {
                known_returns: true,
                ..game
            },
        ),
        ("recommended", recommended_set()),
        ("recommended+windows", recommended_windows_set()),
    ];
    let lifted = LIFTED.get();
    let mut list = String::new();
    for (name, mode) in options {
        MODE.set(Some(mode));
        let mut text = String::new();
        kernel_code(&mut text);
        writeln!(list, "{name} {} {}", text.len(), text.lines().count()).expect("write");
    }
    MODE.set(None);
    LIFTED.set(lifted);
    std::fs::write(dir.join("kernels.txt"), list).expect("write");
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
    println!("cargo::rerun-if-env-changed=ECHORA_KERNEL_PAD");
    println!("cargo::rerun-if-env-changed=ECHORA_KERNEL_ABLATE");
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

    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR"));
    for &(name, generate) in SPLIT_PROGRAMS.iter().chain(OTHER_PROGRAMS) {
        if name == "kernel" {
            generate(&mut out);
            continue;
        }
        let mut text = String::new();
        generate(&mut text);
        std::fs::write(out_dir.join(format!("lifted_{name}.rs")), text)
            .expect("write a generated lift");
    }
    std::fs::write(out_dir.join("programs.rs"), out).expect("write generated programs");
    golden(&out_dir);
    if std::env::var_os("CARGO_FEATURE_CODESIZE").is_some() {
        codesize(&out_dir);
    }
}

/// Programs for `tests/fused_flags.rs`, one per 1 KiB window: a flag setter,
/// a B.cond, and successors that either overwrite NZCV before reading it
/// (so the pair fuses with `EmitOptions::fused_flags`) or read it in a way
/// that must keep it: a CSEL in a successor, the SVC that ends the
/// program, a CCMP before the branch or in a successor, and an indirect
/// branch to a block that reads it. Lifted with fused flags on; `FUSED`
/// says, per program, whether it should fuse and whether it did.
fn flag_code(out: &mut String) {
    const BASE: u64 = 0x140_0000;
    const SVC_EXIT: u32 = 0xd400_00e1;
    // ADDS X12, X12, #0: writes NZCV (and X12, unchanged).
    const KILL: u32 = 0xb100_018c;
    const MARK_NOT_TAKEN: u32 = 0xd280_002b; // MOVZ X11, #1
    const MARK_TAKEN: u32 = 0xd280_004b; // MOVZ X11, #2
    let setters: &[u32] = &[
        0xeb02_0023, // SUBS X3, X1, X2
        0x6b02_0023, // SUBS W3, W1, W2
        0xf100_143f, // CMP X1, #5
        0x7100_143f, // CMP W1, #5
        0xb100_143f, // CMN X1, #5
        0x3100_143f, // CMN W1, #5
        0xab02_0023, // ADDS X3, X1, X2
        0x2b02_0023, // ADDS W3, W1, W2
        0xeb22_4023, // SUBS X3, X1, W2, UXTW
        0x6b22_0023, // SUBS W3, W1, W2, UXTB
        0xea02_0023, // ANDS X3, X1, X2
        0x6a02_0023, // ANDS W3, W1, W2
        0xf240_003f, // TST X1, #1
        0x7201_003f, // TST W1, #0x80000000
        0xf240_1c3f, // TST X1, #0xff
    ];
    let b_cond = |from: u64, to: u64, condition: u32| {
        0x5400_0000 | ((((to - from) / 4) as u32 & 0x7ffff) << 5) | condition
    };
    let b = |from: u64, to: u64| 0x1400_0000 | (((to - from) / 4) as u32 & 0x3ff_ffff);
    let adr = |from: u64, to: u64, rd: u32| {
        let offset = to.wrapping_sub(from) as u32;
        0x1000_0000 | ((offset & 3) << 29) | (((offset >> 2) & 0x7ffff) << 5) | rd
    };
    let mut words: Vec<u32> = Vec::new();
    let mut entries = Vec::new();
    // (entry, should fuse)
    let mut programs: Vec<(u64, bool)> = Vec::new();
    let mut add = |kind: u32, setter: u32, condition: u32| {
        let start = BASE + 4 * words.len() as u64;
        let at = |index: u64| start + 4 * index;
        let mut program = vec![0u32; 16];
        // Taken: 9 onwards; the exit at 15.
        let taken = at(9);
        let exit = at(15);
        program[9] = MARK_TAKEN;
        program[10] = KILL;
        program[11] = b(at(11), exit);
        program[15] = SVC_EXIT;
        let fuses = match kind {
            // Both successors overwrite NZCV first: fuses.
            0 => {
                program[0] = setter;
                program[1] = b_cond(at(1), taken, condition);
                program[2] = MARK_NOT_TAKEN;
                program[3] = KILL;
                program[4] = b(at(4), exit);
                true
            }
            // A successor reads NZCV (CSEL X11, X1, X2, LT).
            1 => {
                program[0] = setter;
                program[1] = b_cond(at(1), taken, condition);
                program[2] = 0x9a82_b02b;
                program[3] = b(at(3), exit);
                false
            }
            // The program ends at a successor: NZCV is what it leaves.
            2 => {
                program[0] = setter;
                program[1] = b_cond(at(1), taken, condition);
                program[2] = SVC_EXIT;
                program[9] = SVC_EXIT;
                false
            }
            // A CCMP between the setter and the branch.
            3 => {
                program[0] = setter;
                program[1] = 0xfa44_1065; // CCMP X3, X4, #5, NE
                program[2] = b_cond(at(2), taken, condition);
                program[3] = MARK_NOT_TAKEN;
                program[4] = KILL;
                program[5] = b(at(5), exit);
                false
            }
            // A CCMP in a successor reads NZCV.
            4 => {
                program[0] = setter;
                program[1] = b_cond(at(1), taken, condition);
                program[2] = 0xfa44_1065; // CCMP X3, X4, #5, NE
                program[3] = b(at(3), exit);
                false
            }
            // An indirect branch from a successor to a block that reads
            // NZCV (CSEL X11, X1, X2, GE): it must stay.
            _ => {
                program[0] = setter;
                program[1] = b_cond(at(1), taken, condition);
                program[2] = adr(at(2), at(6), 24);
                program[3] = 0xd61f_0300; // BR X24
                program[6] = 0x9a82_a02b;
                program[7] = b(at(7), exit);
                entries.push(at(6));
                false
            }
        };
        words.extend_from_slice(&program);
        // One program per 1 KiB window.
        // The rest of the window UDF: each program ends at an SVC, and
        // discovery stops at the UDF after it (padding of SVCs was lifted
        // as hundreds of blocks a window).
        words.resize(words.len().next_multiple_of(256), 0);
        entries.push(start);
        programs.push((start, fuses));
    };
    for condition in 0..14 {
        for &setter in setters {
            add(0, setter, condition);
        }
    }
    for kind in 1..6 {
        for condition in [0, 1, 2, 5, 8, 11, 12] {
            for &setter in &[setters[0], setters[3], setters[6], setters[11]] {
                add(kind, setter, condition);
            }
        }
    }
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let region = CodeRegion {
        name: "flags".into(),
        base: BASE,
        bytes,
    };
    let translation = discover(&[region], &entries, words.len() + 16).expect("flags");
    let options = EmitOptions {
        fused_flags: true,
        ..options()
    };
    let (root, regions) = split(&translation, options);
    // Whether each program's window fused its pair.
    let fused: Vec<(u64, bool, bool)> = programs
        .iter()
        .map(|&(entry, expected)| {
            let window = entry >> 10;
            let function = format!("fn run_window_{window:x}(");
            let source = regions
                .iter()
                .map(|(_, source)| source.as_str())
                .find(|source| source.contains(&function))
                .expect("the program's window");
            let start = source.find(&function).expect("found above");
            let end = source[start + 1..]
                .find("fn run_window_")
                .map_or(source.len(), |end| start + 1 + end);
            (entry, expected, source[start..end].contains("let fused = "))
        })
        .collect();
    writeln!(out, "pub mod flag_code {{").expect("write");
    writeln!(out, "pub const BASE: u64 = {BASE:#x};").expect("write");
    writeln!(out, "pub const WORDS: &[u32] = &{words:#x?};").expect("write");
    writeln!(out, "/// (entry, should fuse, fused)").expect("write");
    writeln!(out, "pub const FUSED: &[(u64, bool, bool)] = &{fused:#x?};").expect("write");
    for (region, source) in regions {
        writeln!(out, "pub mod gen_r_{region:x} {{\n{source}\n}}").expect("write");
    }
    writeln!(out, "{root}\n}}").expect("write");
}

/// Programs for `tests/fp_status.rs`: FPCR and FPSR written with MSR, read
/// with MRS, around float operations whose helpers raise FPSR flags, one
/// program per window. Lifted twice: with `EmitOptions::fp_status_locals`
/// (`fp_status_code`) and without (`fp_status_reference`).
fn fp_status_code(out: &mut String) {
    const BASE: u64 = 0x180_0000;
    const SVC_EXIT: u32 = 0xd400_00e1;
    let msr_fpcr = |rt: u32| 0xd51b_4400 | rt;
    let msr_fpsr = |rt: u32| 0xd51b_4420 | rt;
    let mrs_fpcr = |rt: u32| 0xd53b_4400 | rt;
    let mrs_fpsr = |rt: u32| 0xd53b_4420 | rt;
    const FADD_S: u32 = 0x1e22_2820; // FADD S0, S1, S2
    const FMUL_S: u32 = 0x1e22_0820; // FMUL S0, S1, S2
    const FADD_4S: u32 = 0x4e22_d420; // FADD V0.4S, V1.4S, V2.4S
    const FMUL_4S: u32 = 0x6e22_dc20; // FMUL V0.4S, V1.4S, V2.4S
    let programs: Vec<Vec<u32>> = vec![
        vec![msr_fpsr(3), FADD_S, mrs_fpsr(4)],
        vec![msr_fpsr(3), FADD_4S, mrs_fpsr(4)],
        vec![msr_fpcr(5), FADD_S, mrs_fpcr(6), mrs_fpsr(4)],
        vec![msr_fpcr(5), FMUL_4S, mrs_fpsr(4)],
        vec![FADD_S, mrs_fpsr(4), msr_fpsr(3), FMUL_S, mrs_fpsr(7)],
        vec![
            msr_fpsr(3),
            FADD_4S,
            msr_fpcr(5),
            FMUL_4S,
            mrs_fpsr(4),
            mrs_fpcr(6),
        ],
        vec![msr_fpsr(3), FADD_S],
        vec![msr_fpcr(5), msr_fpsr(3), FMUL_4S],
    ];
    let mut words: Vec<u32> = Vec::new();
    let mut entries = Vec::new();
    for program in &programs {
        entries.push(BASE + 4 * words.len() as u64);
        words.extend_from_slice(program);
        words.push(SVC_EXIT);
        words.resize(words.len().next_multiple_of(256), SVC_EXIT);
    }
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let translation = discover(
        &[CodeRegion {
            name: "fp_status".into(),
            base: BASE,
            bytes,
        }],
        &entries,
        words.len() + 16,
    )
    .expect("fp_status");
    for (name, locals) in [("fp_status_code", true), ("fp_status_reference", false)] {
        let options = EmitOptions {
            inline_vector: true,
            whole_vector_checks: true,
            fp_status_locals: locals,
            ..options()
        };
        let (root, regions) = split(&translation, options);
        writeln!(out, "pub mod {name} {{").expect("write");
        writeln!(out, "pub const BASE: u64 = {BASE:#x};").expect("write");
        writeln!(out, "pub const WORDS: &[u32] = &{words:#x?};").expect("write");
        writeln!(out, "pub const ENTRIES: &[u64] = &{entries:#x?};").expect("write");
        for (region, source) in regions {
            writeln!(out, "pub mod gen_r_{region:x} {{\n{source}\n}}").expect("write");
        }
        writeln!(out, "{root}\n}}").expect("write");
    }
}

/// Programs for `tests/rounded_up.rs`: products, quotients and fused
/// products (scalar, vector, by element; alone and in float runs) from V1,
/// V2 and V3 into V0, an SVC after each. Lifted with helper calls only
/// (`rounded_up_reference`), with the inline paths checked lane by lane
/// (`rounded_up_inline`), with whole-vector checks and the scalar and
/// misc inline paths (`rounded_up_whole`), and with float runs and vector
/// float runs on top (`rounded_up_runs`).
fn rounded_up_code(out: &mut String) {
    const BASE: u64 = 0x260_0000;
    const SVC_EXIT: u32 = 0xd400_00e1;
    const NOP: u32 = 0xd503_201f;
    // Words and their disassembly, which `tests/rounded_up.rs` checks.
    let programs: &[(&str, &[u32])] = &[
        ("fmul v0.4s, v1.4s, v2.4s", &[0x6e22_dc20]),
        ("fmul v0.2d, v1.2d, v2.2d", &[0x6e62_dc20]),
        ("fdiv v0.4s, v1.4s, v2.4s", &[0x6e22_fc20]),
        ("fdiv v0.2d, v1.2d, v2.2d", &[0x6e62_fc20]),
        ("fmul s0, s1, s2", &[0x1e22_0820]),
        ("fmul d0, d1, d2", &[0x1e62_0820]),
        ("fdiv s0, s1, s2", &[0x1e22_1820]),
        ("fdiv d0, d1, d2", &[0x1e62_1820]),
        ("fnmul s0, s1, s2", &[0x1e22_8820]),
        ("fmadd s0, s1, s2, s3", &[0x1f02_0c20]),
        ("fmla v0.4s, v1.4s, v2.4s", &[0x4e22_cc20]),
        ("fmla v0.4s, v1.4s, v2.s[1]", &[0x4fa2_1020]),
        ("fmul v0.4s, v1.4s, v2.s[1]", &[0x4fa2_9020]),
        ("fmulx v0.4s, v1.4s, v2.4s", &[0x4e22_dc20]),
        // Runs (after a NOP: a run never starts a block): the product,
        // then exact adds of a zero V3; a float run needs eight
        // operations, a vector run two.
        (
            "fmul v0.4s, v1.4s, v2.4s; fadd v0.4s, v0.4s, v3.4s",
            &[NOP, 0x6e22_dc20, 0x4e23_d400],
        ),
        (
            "fmul s0, s1, s2; fadd s0, s0, s3 (7 times)",
            &[
                NOP,
                0x1e22_0820,
                0x1e23_2800,
                0x1e23_2800,
                0x1e23_2800,
                0x1e23_2800,
                0x1e23_2800,
                0x1e23_2800,
                0x1e23_2800,
            ],
        ),
    ];
    let mut words: Vec<u32> = Vec::new();
    let mut entries = Vec::new();
    for (_, program) in programs {
        entries.push(BASE + 4 * words.len() as u64);
        words.extend_from_slice(program);
        words.push(SVC_EXIT);
        words.resize(words.len().next_multiple_of(256), SVC_EXIT);
    }
    let names: Vec<&str> = programs.iter().map(|(name, _)| *name).collect();
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let translation = discover(
        &[CodeRegion {
            name: "rounded_up".into(),
            base: BASE,
            bytes,
        }],
        &entries,
        words.len() + 16,
    )
    .expect("rounded_up");
    let plain = EmitOptions {
        inline_vector: false,
        ..options()
    };
    let inline = EmitOptions {
        inline_vector: true,
        whole_vector_checks: false,
        ..options()
    };
    let whole = EmitOptions {
        inline_vector: true,
        whole_vector_checks: true,
        inline_fp_scalar: true,
        inline_vector_misc: true,
        ..options()
    };
    let runs = EmitOptions {
        float_runs: true,
        vector_float_runs: true,
        ..whole
    };
    for (name, options) in [
        ("rounded_up_reference", plain),
        ("rounded_up_inline", inline),
        ("rounded_up_whole", whole),
        ("rounded_up_runs", runs),
    ] {
        let (root, regions) = split(&translation, options);
        writeln!(out, "pub mod {name} {{").expect("write");
        writeln!(out, "pub const BASE: u64 = {BASE:#x};").expect("write");
        writeln!(out, "pub const WORDS: &[u32] = &{words:#x?};").expect("write");
        writeln!(out, "pub const ENTRIES: &[u64] = &{entries:#x?};").expect("write");
        writeln!(out, "pub const NAMES: &[&str] = &{names:?};").expect("write");
        for (region, source) in regions {
            writeln!(out, "pub mod gen_r_{region:x} {{\n{source}\n}}").expect("write");
        }
        writeln!(out, "{root}\n}}").expect("write");
    }
}

/// Programs for `tests/vector_unary.rs`: vector FSQRT, FCVTZS/FCVTZU,
/// FRECPE, FRSQRTE and FRINT* (4S, 2S, 2D, and the scalar forms that
/// exist), one instruction from V1 into V0 (or V1 itself) and an SVC each.
/// Lifted with `EmitOptions::inline_vector_unary` (`vector_unary_code`)
/// and without (`vector_unary_reference`: the helper for every lane).
fn vector_unary_code(out: &mut String) {
    const BASE: u64 = 0x1c0_0000;
    const SVC_EXIT: u32 = 0xd400_00e1;
    // (name, U, a, opcode, scalar form exists)
    const OPS: &[(&str, u32, u32, u32, bool)] = &[
        ("fsqrt", 1, 1, 0x1f, false),
        ("fcvtzs", 0, 1, 0x1b, true),
        ("fcvtzu", 1, 1, 0x1b, true),
        ("frecpe", 0, 1, 0x1d, true),
        ("frsqrte", 1, 1, 0x1d, true),
        ("frintn", 0, 0, 0x18, false),
        ("frintp", 0, 1, 0x18, false),
        ("frintm", 0, 0, 0x19, false),
        ("frintz", 0, 1, 0x19, false),
        ("frinta", 1, 0, 0x18, false),
    ];
    // (form, Q, sz, scalar)
    const FORMS: &[(&str, u32, u32, bool)] = &[
        ("4s", 1, 0, false),
        ("2s", 0, 0, false),
        ("2d", 1, 1, false),
        ("s", 0, 0, true),
        ("d", 0, 1, true),
    ];
    let mut words: Vec<u32> = Vec::new();
    // (name, form, entry, in place)
    let mut programs: Vec<(String, String, u64, bool)> = Vec::new();
    for &(name, u, a, opcode, has_scalar) in OPS {
        for &(form, q, sz, scalar) in FORMS {
            if scalar && !has_scalar {
                continue;
            }
            for in_place in [false, true] {
                let base = if scalar { 0x5e20_0800 } else { 0x0e20_0800 };
                let rd = u32::from(in_place);
                let word =
                    base | q << 30 | u << 29 | (a << 1 | sz) << 22 | opcode << 12 | 1 << 5 | rd;
                programs.push((
                    name.to_owned(),
                    form.to_owned(),
                    BASE + 4 * words.len() as u64,
                    in_place,
                ));
                words.push(word);
                words.push(SVC_EXIT);
            }
        }
    }
    let entries: Vec<u64> = programs.iter().map(|program| program.2).collect();
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let translation = discover(
        &[CodeRegion {
            name: "vector_unary".into(),
            base: BASE,
            bytes,
        }],
        &entries,
        words.len() + 16,
    )
    .expect("vector_unary");
    for (name, inline) in [
        ("vector_unary_code", true),
        ("vector_unary_reference", false),
    ] {
        let options = EmitOptions {
            inline_vector: true,
            whole_vector_checks: true,
            inline_vector_unary: inline,
            // As game builds lift: helper calls then pass FPCR and FPSR
            // through window locals.
            fp_status_locals: true,
            // The count is of the windows' code: no function copies.
            regalloc: false,
            ..options()
        };
        let (root, regions) = split(&translation, options);
        // Inlined instructions: those with a lane fallback in the emission.
        let inlined: usize = regions
            .iter()
            .map(|(_, source)| source.matches(", false, true, ").count())
            .sum();
        writeln!(out, "pub mod {name} {{").expect("write");
        writeln!(out, "pub const BASE: u64 = {BASE:#x};").expect("write");
        writeln!(out, "pub const WORDS: &[u32] = &{words:#x?};").expect("write");
        writeln!(out, "/// (operation, form, entry, Vd is Vn)").expect("write");
        writeln!(
            out,
            "pub const PROGRAMS: &[(&str, &str, u64, bool)] = &{programs:#x?};"
        )
        .expect("write");
        writeln!(out, "/// Instructions emitted with a native lane path.\npub const INLINED: usize = {inlined};").expect("write");
        for (region, source) in regions {
            writeln!(out, "pub mod gen_r_{region:x} {{\n{source}\n}}").expect("write");
        }
        writeln!(out, "{root}\n}}").expect("write");
    }
}

/// Programs for `tests/vector_int.rs`: random words of the vector integer
/// operations, element moves and float pairwise operations (up to 2 of
/// each operation and form, as decoded), one word and an SVC each. Lifted
/// with `EmitOptions::inline_vector_int` and `inline_vector_pairwise`
/// (`vector_int_code`) and without (`vector_int_reference`).
fn vector_int_code(out: &mut String) {
    use echora_lift::aarch64::{Instruction, decode};
    use echora_lift::simd::SimdOp;

    const BASE: u64 = 0x200_0000;
    const SVC_EXIT: u32 = 0xd400_00e1;
    // Helper calls the inlined lift must not keep (the float pairwise
    // helper stays for FPCR modes the inline code does not take).
    const INTEGER_HELPERS: &[&str] = &[
        "vec_int(",
        "vec_int_element(",
        "vec_int_pairwise(",
        "vec_unary(",
        "vec_shift(",
        "vec_shift_long(",
        "vec_long(",
        "vec_long_element(",
        "vec_narrow(",
        "vec_pairwise_long(",
        "vec_across(",
        "move_immediate(",
        "modify_immediate(",
        "dup_element(",
        "dup_general(",
        "insert_element(",
        "insert_general(",
        "move_to_general(",
    ];
    let family = |op: &SimdOp| {
        matches!(
            op,
            SimdOp::VecInt { .. }
                | SimdOp::VecIntElement { .. }
                | SimdOp::VecIntPairwise { .. }
                | SimdOp::VecUnary { .. }
                | SimdOp::VecShift { .. }
                | SimdOp::VecShiftLong { .. }
                | SimdOp::VecLong { .. }
                | SimdOp::VecLongElement { .. }
                | SimdOp::VecNarrow { .. }
                | SimdOp::VecPairwiseLong { .. }
                | SimdOp::VecAcross { .. }
                | SimdOp::MoveImmediate { .. }
                | SimdOp::ModifyImmediate { .. }
                | SimdOp::DupElement { .. }
                | SimdOp::DupGeneral { .. }
                | SimdOp::InsertElement { .. }
                | SimdOp::InsertGeneral { .. }
                | SimdOp::MoveToGeneral { .. }
                | SimdOp::VecFpPairwise { .. }
        )
    };
    let mut seen: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    let mut per_kind: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    let mut words: Vec<u32> = Vec::new();
    let mut entries = Vec::new();
    let mut state = 0x5eed_0f1e_2d3c_4b5au64;
    for _ in 0..3_000_000 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // The SIMD data-processing space (op0 x111).
        let word = (state as u32 & !0x0e00_0000) | 0x0e00_0000;
        let Instruction::Simd(op) = decode(word, 0) else {
            continue;
        };
        if !family(&op) {
            continue;
        }
        let text = format!("{op:?}");
        // The operation and form without its registers; at most 2 words
        // each, and 500 per kind of operation (immediates and indices make
        // many forms).
        let mut key = String::new();
        for (index, part) in text.split(", ").enumerate() {
            let register = ["rd: ", "rn: ", "rm: "].iter().any(|field| {
                part.trim_start_matches(|c: char| c != 'r')
                    .starts_with(field)
            });
            if index == 0 || !register {
                key.push_str(part);
                key.push(',');
            }
        }
        let kind = text.split(' ').next().unwrap_or_default().to_owned();
        if *per_kind.get(&kind).unwrap_or(&0) >= 500 {
            continue;
        }
        let count = seen.entry(key).or_default();
        if *count >= 2 {
            continue;
        }
        *count += 1;
        *per_kind.entry(kind).or_default() += 1;
        entries.push(BASE + 4 * words.len() as u64);
        words.push(word);
        words.push(SVC_EXIT);
    }
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let translation = discover(
        &[CodeRegion {
            name: "vector_int".into(),
            base: BASE,
            bytes,
        }],
        &entries,
        words.len() + 16,
    )
    .expect("vector_int");
    for (name, inline) in [("vector_int_code", true), ("vector_int_reference", false)] {
        let options = EmitOptions {
            inline_vector: true,
            whole_vector_checks: true,
            inline_vector_int: inline,
            inline_vector_pairwise: inline,
            // As game builds lift: helper calls then pass FPCR and FPSR
            // through window locals.
            fp_status_locals: true,
            ..options()
        };
        let (root, regions) = split(&translation, options);
        let helpers: usize = regions
            .iter()
            .map(|(_, source)| {
                INTEGER_HELPERS
                    .iter()
                    .map(|helper| {
                        source
                            .matches(&format!("echora_guest::simd::{helper}"))
                            .count()
                    })
                    .sum::<usize>()
            })
            .sum();
        let pairwise_lanes: usize = regions
            .iter()
            .map(|(_, source)| source.matches("echora_guest::simd::vec_fp_lane(").count())
            .sum();
        writeln!(out, "pub mod {name} {{").expect("write");
        writeln!(out, "pub const BASE: u64 = {BASE:#x};").expect("write");
        writeln!(out, "pub const WORDS: &[u32] = &{words:#x?};").expect("write");
        writeln!(out, "pub const ENTRIES: &[u64] = &{entries:#x?};").expect("write");
        writeln!(
            out,
            "/// Operations and forms sampled.\npub const FORMS: usize = {};",
            seen.len()
        )
        .expect("write");
        writeln!(out, "/// Integer and move helper calls left in the emission.\npub const HELPER_CALLS: usize = {helpers};").expect("write");
        writeln!(out, "/// Single-lane float helper calls (pairwise fallbacks).\npub const LANE_CALLS: usize = {pairwise_lanes};").expect("write");
        for (region, source) in regions {
            writeln!(out, "pub mod gen_r_{region:x} {{\n{source}\n}}").expect("write");
        }
        writeln!(out, "{root}\n}}").expect("write");
    }
}

/// Programs for `tests/fp_scalar.rs`: random words of the scalar float
/// moves, conversions, FSQRT, FRINT* and FCCMP (up to 2 of each operation
/// and form), one word and an SVC each. Lifted with
/// `EmitOptions::inline_fp_scalar` (`fp_scalar_code`) and without
/// (`fp_scalar_reference`).
fn fp_scalar_code(out: &mut String) {
    use echora_lift::aarch64::{Instruction, decode};
    use echora_lift::simd::SimdOp;

    const BASE: u64 = 0x220_0000;
    const SVC_EXIT: u32 = 0xd400_00e1;
    let family = |op: &SimdOp| {
        matches!(
            op,
            SimdOp::FpMoveToGeneral { .. }
                | SimdOp::FpMoveFromGeneral { .. }
                | SimdOp::IntToFp { .. }
                | SimdOp::SimdIntToFp { .. }
                | SimdOp::FpToInt { .. }
                | SimdOp::FpConvert { .. }
                | SimdOp::FpUnary { .. }
                | SimdOp::FpConditionalCompare { .. }
        )
    };
    let mut seen: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    let mut words: Vec<u32> = Vec::new();
    let mut entries = Vec::new();
    let mut state = 0x0f5c_a1a2_0000_0007u64;
    for _ in 0..3_000_000 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // The scalar floating-point space (bits 28..24 = 11110).
        let word = (state as u32 & !0x1f00_0000) | 0x1e00_0000;
        let Instruction::Simd(op) = decode(word, 0) else {
            continue;
        };
        if !family(&op) {
            continue;
        }
        // The operation and form without its registers.
        let text = format!("{op:?}");
        let mut key = String::new();
        for (index, part) in text.split(", ").enumerate() {
            let register = ["rd: ", "rn: ", "rm: "].iter().any(|field| {
                part.trim_start_matches(|c: char| c != 'r')
                    .starts_with(field)
            });
            if index == 0 || !register {
                key.push_str(part);
                key.push(',');
            }
        }
        let count = seen.entry(key).or_default();
        if *count >= 2 {
            continue;
        }
        *count += 1;
        entries.push(BASE + 4 * words.len() as u64);
        words.push(word);
        words.push(SVC_EXIT);
    }
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let translation = discover(
        &[CodeRegion {
            name: "fp_scalar".into(),
            base: BASE,
            bytes,
        }],
        &entries,
        words.len() + 16,
    )
    .expect("fp_scalar");
    for (name, inline) in [("fp_scalar_code", true), ("fp_scalar_reference", false)] {
        let options = EmitOptions {
            inline_vector: true,
            whole_vector_checks: true,
            inline_fp_scalar: inline,
            // As game builds lift: helper calls then pass FPCR and FPSR
            // through window locals.
            fp_status_locals: true,
            ..options()
        };
        let (root, regions) = split(&translation, options);
        // FMOV to and from general registers never needs its helper.
        let moves: usize = regions
            .iter()
            .map(|(_, source)| {
                source
                    .matches("echora_guest::simd::fp_move_to_general(")
                    .count()
                    + source
                        .matches("echora_guest::simd::fp_move_from_general(")
                        .count()
            })
            .sum();
        writeln!(out, "pub mod {name} {{").expect("write");
        writeln!(out, "pub const BASE: u64 = {BASE:#x};").expect("write");
        writeln!(out, "pub const WORDS: &[u32] = &{words:#x?};").expect("write");
        writeln!(out, "pub const ENTRIES: &[u64] = &{entries:#x?};").expect("write");
        writeln!(
            out,
            "/// Operations and forms sampled.\npub const FORMS: usize = {};",
            seen.len()
        )
        .expect("write");
        writeln!(
            out,
            "/// FMOV helper calls left in the emission.\npub const MOVE_CALLS: usize = {moves};"
        )
        .expect("write");
        for (region, source) in regions {
            writeln!(out, "pub mod gen_r_{region:x} {{\n{source}\n}}").expect("write");
        }
        writeln!(out, "{root}\n}}").expect("write");
    }
}

/// Programs for `tests/vector_misc.rs`: random words of TBL/TBX, the
/// lane, replicate and multiple-structure loads and stores, FABD, FMULX,
/// FMAXV and its kin and the remaining conversions (one of each operation
/// and form, 120 of each kind), one word and an SVC each. Lifted with
/// `EmitOptions::inline_vector_misc` (`vector_misc_code`) and without
/// (`vector_misc_reference`).
fn vector_misc_code(out: &mut String) {
    use echora_lift::aarch64::{Instruction, decode};
    use echora_lift::simd::{Rounding, SimdOp, VecFpOp, VecFpUnaryOp};

    const BASE: u64 = 0x240_0000;
    const SVC_EXIT: u32 = 0xd400_00e1;
    let family = |op: &SimdOp| match op {
        SimdOp::VecTable { .. }
        | SimdOp::TransferSingle { .. }
        | SimdOp::LoadReplicate { .. }
        | SimdOp::TransferMultiple { .. }
        | SimdOp::VecFpAcross { .. }
        | SimdOp::FpToFixed { .. }
        | SimdOp::IntToFpFixed { .. } => true,
        SimdOp::VecFp { op, .. } => matches!(op, VecFpOp::AbsDiff | VecFpOp::MulExtended),
        SimdOp::VecFpElement { op, .. } => *op == VecFpOp::MulExtended,
        SimdOp::VecFpUnary { op, fbits, .. } => match op {
            VecFpUnaryOp::FromSigned | VecFpUnaryOp::FromUnsigned => true,
            VecFpUnaryOp::ToSigned(rounding) | VecFpUnaryOp::ToUnsigned(rounding) => {
                *fbits != 0 || *rounding != Rounding::Zero
            }
            _ => false,
        },
        _ => false,
    };
    let mut seen: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    let mut per_kind: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    let mut words: Vec<u32> = Vec::new();
    let mut entries = Vec::new();
    let mut state = 0x3c5a_96e1_0000_000bu64;
    for attempt in 0..4_000_000u32 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // The SIMD data-processing space (op0 x111), or loads and stores
        // (op0 x1x0).
        let word = if attempt % 2 == 0 {
            (state as u32 & !0x0e00_0000) | 0x0e00_0000
        } else {
            (state as u32 & !0x0a00_0000) | 0x0800_0000
        };
        let Instruction::Simd(op) = decode(word, 0) else {
            continue;
        };
        if !family(&op) {
            continue;
        }
        let text = format!("{op:?}");
        let mut key = String::new();
        for (index, part) in text.split(", ").enumerate() {
            let register = ["rd: ", "rn: ", "rm: ", "rt: "].iter().any(|field| {
                part.trim_start_matches(|c: char| c != 'r')
                    .starts_with(field)
            });
            if index == 0 || !register {
                key.push_str(part);
                key.push(',');
            }
        }
        // One word per form, at most 120 per kind (the variant and its
        // operation): lane indices and structure shapes make many forms.
        let variant = text.split(' ').next().unwrap_or_default();
        let operation = text
            .split(", ")
            .find_map(|part| part.split("op: ").nth(1))
            .unwrap_or_default();
        let kind = format!("{variant} {operation}");
        if *per_kind.get(&kind).unwrap_or(&0) >= 120 {
            continue;
        }
        let count = seen.entry(key).or_default();
        if *count >= 1 {
            continue;
        }
        *count += 1;
        *per_kind.entry(kind).or_default() += 1;
        entries.push(BASE + 4 * words.len() as u64);
        words.push(word);
        words.push(SVC_EXIT);
    }
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let translation = discover(
        &[CodeRegion {
            name: "vector_misc".into(),
            base: BASE,
            bytes,
        }],
        &entries,
        words.len() + 16,
    )
    .expect("vector_misc");
    for (name, inline) in [("vector_misc_code", true), ("vector_misc_reference", false)] {
        let options = EmitOptions {
            inline_vector: true,
            whole_vector_checks: true,
            inline_vector_misc: inline,
            // As game builds lift: helper calls then pass FPCR and FPSR
            // through window locals; window memory too.
            fp_status_locals: true,
            window_memory: true,
            ..options()
        };
        let (root, regions) = split(&translation, options);
        // Helper calls that must not be left.
        let memory: usize = regions
            .iter()
            .map(|(_, source)| {
                [
                    "vec_table(",
                    "transfer_single(",
                    "load_replicate(",
                    "transfer_multiple(",
                ]
                .iter()
                .map(|helper| {
                    source
                        .matches(&format!("echora_guest::simd::{helper}"))
                        .count()
                })
                .sum::<usize>()
            })
            .sum();
        writeln!(out, "pub mod {name} {{").expect("write");
        writeln!(out, "pub const BASE: u64 = {BASE:#x};").expect("write");
        writeln!(out, "pub const WORDS: &[u32] = &{words:#x?};").expect("write");
        writeln!(out, "pub const ENTRIES: &[u64] = &{entries:#x?};").expect("write");
        writeln!(
            out,
            "/// Operations and forms sampled.\npub const FORMS: usize = {};",
            seen.len()
        )
        .expect("write");
        writeln!(out, "/// TBL and structure load and store helper calls left.\npub const HELPER_CALLS: usize = {memory};").expect("write");
        for (region, source) in regions {
            writeln!(out, "pub mod gen_r_{region:x} {{\n{source}\n}}").expect("write");
        }
        writeln!(out, "{root}\n}}").expect("write");
    }
}

/// Programs for `tests/fallback_counts.rs`: FADD S0, S1, S2, FSQRT V0.4S,
/// V1.4S and FMLA V0.4S, V1.4S, V2.4S, each followed by an SVC, lifted
/// with `EmitOptions::count_fallbacks` and the inlined float options.
fn fallback_count_code(out: &mut String) {
    const BASE: u64 = 0x260_0000;
    const SVC_EXIT: u32 = 0xd400_00e1;
    const PROGRAMS: [u32; 3] = [0x1e22_2820, 0x6ea1_f820, 0x4e22_cc20];
    let mut words = Vec::new();
    let mut entries = Vec::new();
    for word in PROGRAMS {
        entries.push(BASE + 4 * words.len() as u64);
        words.push(word);
        words.push(SVC_EXIT);
    }
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let translation = discover(
        &[CodeRegion {
            name: "fallback_count".into(),
            base: BASE,
            bytes,
        }],
        &entries,
        words.len() + 16,
    )
    .expect("fallback_count");
    let options = EmitOptions {
        inline_vector: true,
        whole_vector_checks: true,
        inline_vector_unary: true,
        fp_status_locals: true,
        count_fallbacks: true,
        ..options()
    };
    let (root, regions) = split(&translation, options);
    writeln!(out, "pub mod fallback_count_code {{").expect("write");
    writeln!(out, "pub const BASE: u64 = {BASE:#x};").expect("write");
    writeln!(out, "pub const WORDS: &[u32] = &{words:#x?};").expect("write");
    writeln!(out, "pub const ENTRIES: &[u64] = &{entries:#x?};").expect("write");
    for (region, source) in regions {
        writeln!(out, "pub mod gen_r_{region:x} {{\n{source}\n}}").expect("write");
    }
    writeln!(out, "{root}\n}}").expect("write");
}

/// Programs for `tests/q_pairs.rs`: LDP and STP of Q registers in every
/// addressing form (signed offset, pre-index, post-index) at offsets
/// across the immediate's range, with the base register an ordinary one or
/// SP, and STP followed by an LDP whose bytes overlap it; each followed by an SVC. Lifted with
/// `EmitOptions::direct_q_pairs` (`q_pair_code`) and without
/// (`q_pair_reference`), with window memory and FPCR/FPSR locals as game
/// builds.
fn q_pair_code(out: &mut String) {
    use echora_lift::aarch64::{Instruction, decode};

    const BASE: u64 = 0x280_0000;
    const SVC_EXIT: u32 = 0xd400_00e1;
    // (store opcode, load opcode) per form: signed offset, pre-index,
    // post-index.
    const FORMS: [(&str, u32, u32); 3] = [
        ("offset", 0xad00_0000, 0xad40_0000),
        ("pre", 0xad80_0000, 0xadc0_0000),
        ("post", 0xac80_0000, 0xacc0_0000),
    ];
    let encode = |opcode: u32, imm7: i32, rt2: u32, rn: u32, rt: u32| {
        opcode | ((imm7 as u32) & 0x7f) << 15 | rt2 << 10 | rn << 5 | rt
    };
    // (description, words)
    let mut programs: Vec<(String, Vec<u32>)> = Vec::new();
    for (form, store, load) in FORMS {
        for imm7 in [-64, -1, 0, 1, 63] {
            for (rt, rt2, rn) in [(0, 1, 2), (30, 31, 31), (5, 6, 7)] {
                programs.push((
                    format!("stp q{rt}, q{rt2}, x{rn} {form} {imm7}"),
                    vec![encode(store, imm7, rt2, rn, rt)],
                ));
                programs.push((
                    format!("ldp q{rt}, q{rt2}, x{rn} {form} {imm7}"),
                    vec![encode(load, imm7, rt2, rn, rt)],
                ));
            }
            // LDP with both destinations the same register is
            // CONSTRAINED UNPREDICTABLE and does not decode (a Q pair's
            // registers and its base never alias: one is a vector, the
            // other a general register).
        }
    }
    // STP then an LDP of overlapping bytes (16 apart, and the same).
    for (first, second) in [(0, 1), (0, 0), (1, 0)] {
        programs.push((
            format!("stp q0, q1, x2 offset {first}; ldp q3, q4, x2 offset {second}"),
            vec![
                encode(FORMS[0].1, first, 1, 2, 0),
                encode(FORMS[0].2, second, 4, 2, 3),
            ],
        ));
    }
    let mut words: Vec<u32> = Vec::new();
    let mut entries = Vec::new();
    let mut names = Vec::new();
    for (name, program) in &programs {
        for &word in program {
            assert!(
                matches!(decode(word, 0), Instruction::LoadStoreVectorPair { .. }),
                "{name}: {word:#010x}"
            );
        }
        entries.push(BASE + 4 * words.len() as u64);
        names.push(name.clone());
        words.extend_from_slice(program);
        words.push(SVC_EXIT);
    }
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let translation = discover(
        &[CodeRegion {
            name: "q_pairs".into(),
            base: BASE,
            bytes,
        }],
        &entries,
        words.len() + 16,
    )
    .expect("q_pairs");
    for (name, direct) in [("q_pair_code", true), ("q_pair_reference", false)] {
        let options = EmitOptions {
            direct_q_pairs: direct,
            window_memory: true,
            fp_status_locals: true,
            // The census counts the windows' calls: no function copies.
            regalloc: false,
            ..options()
        };
        let (root, regions) = split(&translation, options);
        let out_of_line: usize = regions
            .iter()
            .map(|(_, source)| {
                echora_lift::census::census(source)
                    .iter()
                    .map(|(_, counts)| counts.out_of_line)
                    .sum::<usize>()
            })
            .sum();
        writeln!(out, "pub mod {name} {{").expect("write");
        writeln!(out, "pub const BASE: u64 = {BASE:#x};").expect("write");
        writeln!(out, "pub const WORDS: &[u32] = &{words:#x?};").expect("write");
        writeln!(out, "pub const ENTRIES: &[u64] = &{entries:#x?};").expect("write");
        writeln!(out, "pub const NAMES: &[&str] = &{names:?};").expect("write");
        writeln!(out, "/// Out-of-line `load_q_pair`/`store_q_pair` calls left (`echora_lift::census`).\npub const OUT_OF_LINE: usize = {out_of_line};").expect("write");
        for (region, source) in regions {
            writeln!(out, "pub mod gen_r_{region:x} {{\n{source}\n}}").expect("write");
        }
        writeln!(out, "{root}\n}}").expect("write");
    }
}

/// Programs for `tests/vector_runs.rs`: two to six vector float arithmetic
/// operations in a row (`echora_lift::simd::vector_run_op`: FADD, FSUB,
/// FMUL, FMLA, FMLS, FDIV, FRECPS, FRSQRTS and the by-element forms, each
/// size), on V0 to V3 so that results feed later operations and registers
/// alias, a quarter of them with an MSR FPCR, X5 in the middle (which
/// splits the run), each followed by an SVC. Lifted with
/// `EmitOptions::vector_float_runs` (`vector_run_code`) and without
/// (`vector_run_reference`), with the game's other options.
/// Programs for `tests/smallest_normal.rs`: each multiply, divide, fused
/// operation and narrowing conversion the inline options take natively,
/// alone, and a vector float run and a scalar float run of multiplies,
/// lifted with every inline and run option on and with none (the helpers
/// alone).
fn smallest_normal_code(out: &mut String) {
    const BASE: u64 = 0x2c0_0000;
    const SVC_EXIT: u32 = 0xd400_00e1;
    const NOP: u32 = 0xd503_201f;
    let single: &[u32] = &[
        0x1e22_0820, // FMUL S0, S1, S2
        0x1e22_8820, // FNMUL S0, S1, S2
        0x1e22_1820, // FDIV S0, S1, S2
        0x1f02_0c20, // FMADD S0, S1, S2, S3
        0x1f22_0c20, // FNMADD S0, S1, S2, S3
        0x1f02_8c20, // FMSUB S0, S1, S2, S3
        0x1e62_0820, // FMUL D0, D1, D2
        0x1e62_1820, // FDIV D0, D1, D2
        0x1f42_0c20, // FMADD D0, D1, D2, D3
        0x6e22_dc20, // FMUL V0.4S, V1.4S, V2.4S
        0x6e62_dc20, // FMUL V0.2D, V1.2D, V2.2D
        0x4e22_cc20, // FMLA V0.4S, V1.4S, V2.4S
        0x4ea2_cc20, // FMLS V0.4S, V1.4S, V2.4S
        0x4f82_1020, // FMLA V0.4S, V1.4S, V2.S[0]
        0x6e22_fc20, // FDIV V0.4S, V1.4S, V2.4S
        0x4e22_dc20, // FMULX V0.4S, V1.4S, V2.4S
        0x4e62_dc20, // FMULX V0.2D, V1.2D, V2.2D
        0x5e22_dc20, // FMULX S0, S1, S2
        0x4f82_9020, // FMUL V0.4S, V1.4S, V2.S[0]
        0x4e22_fc20, // FRECPS V0.4S, V1.4S, V2.4S
        0x4ea2_fc20, // FRSQRTS V0.4S, V1.4S, V2.4S
        0x1e62_4020, // FCVT S0, D1
    ];
    let mut programs: Vec<Vec<u32>> = single.iter().map(|&word| vec![word]).collect();
    // A vector float run (FMUL V0.4S and V4.4S, the same operands) and a
    // scalar float run of eight FMUL S, neither first nor last in its
    // block.
    programs.push(vec![NOP, 0x6e22_dc20, 0x6e22_dc24, NOP]);
    let mut run = vec![NOP];
    for index in 0..8 {
        run.push(if index % 2 == 0 {
            0x1e22_0820
        } else {
            0x1e22_0824
        });
    }
    run.push(NOP);
    programs.push(run);
    let mut words: Vec<u32> = Vec::new();
    let mut entries = Vec::new();
    for program in &programs {
        entries.push(BASE + 4 * words.len() as u64);
        words.extend_from_slice(program);
        words.push(SVC_EXIT);
    }
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let translation = discover(
        &[CodeRegion {
            name: "smallest_normal".into(),
            base: BASE,
            bytes,
        }],
        &entries,
        words.len() + 16,
    )
    .expect("smallest_normal");
    for (name, on) in [
        ("smallest_normal_code", true),
        ("smallest_normal_reference", false),
    ] {
        let options = EmitOptions {
            inline_vector: on,
            whole_vector_checks: on,
            inline_vector_unary: on,
            inline_vector_int: on,
            inline_vector_pairwise: on,
            inline_fp_scalar: on,
            inline_vector_misc: on,
            vector_float_runs: on,
            float_runs: on,
            fp_status_locals: on,
            ..EmitOptions::default()
        };
        let (root, regions) = split(&translation, options);
        writeln!(out, "pub mod {name} {{").expect("write");
        writeln!(out, "pub const BASE: u64 = {BASE:#x};").expect("write");
        writeln!(out, "pub const WORDS: &[u32] = &{words:#x?};").expect("write");
        writeln!(out, "pub const ENTRIES: &[u64] = &{entries:#x?};").expect("write");
        for (region, source) in regions {
            writeln!(out, "pub mod gen_r_{region:x} {{\n{source}\n}}").expect("write");
        }
        writeln!(out, "{root}\n}}").expect("write");
    }
}

/// Programs for `tests/memory_faults.rs`: every plain load and store form
/// (each size, sign extension, writeback, S/D/Q, pairs) and the exclusive,
/// ordered, atomic and SP-based forms, each alone (an exclusive pair
/// together), lifted with the build's options.
fn memory_fault_code(out: &mut String) {
    const BASE: u64 = 0x2e0_0000;
    const SVC_EXIT: u32 = 0xd400_00e1;
    let forms: &[(&str, &[u32])] = &[
        ("ldrb w0, [x1]", &[0x3940_0020]),
        ("ldrh w0, [x1]", &[0x7940_0020]),
        ("ldr w0, [x1]", &[0xb940_0020]),
        ("ldr x0, [x1]", &[0xf940_0020]),
        ("ldrsb x0, [x1]", &[0x3980_0020]),
        ("ldrsh w0, [x1]", &[0x79c0_0020]),
        ("ldrsw x0, [x1]", &[0xb980_0020]),
        ("strb w2, [x1]", &[0x3900_0022]),
        ("strh w2, [x1]", &[0x7900_0022]),
        ("str w2, [x1]", &[0xb900_0022]),
        ("str x2, [x1]", &[0xf900_0022]),
        ("ldr x0, [x1, #8]!", &[0xf840_8c20]),
        ("str x2, [x1], #-8", &[0xf81f_8422]),
        ("ldur x0, [x1, #-1]", &[0xf85f_f020]),
        ("ldr s0, [x1]", &[0xbd40_0020]),
        ("ldr d0, [x1]", &[0xfd40_0020]),
        ("ldr q0, [x1]", &[0x3dc0_0020]),
        ("str d0, [x1]", &[0xfd00_0020]),
        ("str q0, [x1]", &[0x3d80_0020]),
        ("ldp x0, x3, [x1]", &[0xa940_0c20]),
        ("stp x2, x3, [x1]", &[0xa900_0c22]),
        ("ldp w0, w3, [x1]", &[0x2940_0c20]),
        ("ldxr x0, [x1]", &[0xc85f_7c20]),
        ("ldxr w0, [x1]", &[0x885f_7c20]),
        ("ldaxr x0, [x1]", &[0xc85f_fc20]),
        ("stxr w4, x2, [x1]", &[0xc804_7c22]),
        (
            "ldxr x0, [x1]; stxr w4, x2, [x1]",
            &[0xc85f_7c20, 0xc804_7c22],
        ),
        ("ldar x0, [x1]", &[0xc8df_fc20]),
        ("stlr x2, [x1]", &[0xc89f_fc22]),
        ("ldapr x0, [x1]", &[0xf8bf_c020]),
        ("cas x0, x2, [x1]", &[0xc8a0_7c22]),
        ("ldadd x2, x0, [x1]", &[0xf822_0020]),
        ("swp x2, x0, [x1]", &[0xf822_8020]),
        ("ldr x0, [sp]", &[0xf940_03e0]),
        ("str x2, [sp, #8]", &[0xf900_07e2]),
    ];
    let mut words: Vec<u32> = Vec::new();
    let mut entries = Vec::new();
    let mut names = Vec::new();
    for (name, program) in forms {
        entries.push(BASE + 4 * words.len() as u64);
        names.push(*name);
        words.extend_from_slice(program);
        words.push(SVC_EXIT);
    }
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let translation = discover(
        &[CodeRegion {
            name: "memory_fault".into(),
            base: BASE,
            bytes,
        }],
        &entries,
        words.len() + 16,
    )
    .expect("memory_fault");
    {
        let name = "memory_fault_code";
        let (root, regions) = split(&translation, options());
        writeln!(out, "pub mod {name} {{").expect("write");
        writeln!(out, "pub const BASE: u64 = {BASE:#x};").expect("write");
        writeln!(out, "pub const WORDS: &[u32] = &{words:#x?};").expect("write");
        writeln!(out, "pub const ENTRIES: &[u64] = &{entries:#x?};").expect("write");
        writeln!(out, "pub const NAMES: &[&str] = &{names:?};").expect("write");
        for (region, source) in regions {
            writeln!(out, "pub mod gen_r_{region:x} {{\n{source}\n}}").expect("write");
        }
        writeln!(out, "{root}\n}}").expect("write");
    }
}

/// Programs for `tests/fp_mode_groups.rs`: consecutive scalar float
/// operations `EmitOptions::hoist_fp_mode` groups, alone, with an MSR FPSR
/// (which can clear IXC) or MSR FPCR among them, in double precision, and
/// with a compare and FMAX; lifted with the build's options, every inline
/// path and `hoist_fp_mode` (`fp_mode_group_code`), and with the helpers
/// alone (`fp_mode_group_reference`).
fn fp_mode_group_code(out: &mut String) {
    const BASE: u64 = 0x2f0_0000;
    const NOP: u32 = 0xd503_201f;
    const SVC_EXIT: u32 = 0xd400_00e1;
    const SINGLES: [u32; 5] = [
        0x1e22_0820, // FMUL S0, S1, S2
        0x1e21_2803, // FADD S3, S0, S1
        0x1e22_1864, // FDIV S4, S3, S2
        0x1f01_0c85, // FMADD S5, S4, S1, S3
        0x1e20_38a6, // FSUB S6, S5, S0
    ];
    const MSR_FPSR_X5: u32 = 0xd51b_4425;
    const MSR_FPCR_X6: u32 = 0xd51b_4406;
    let with = |at: usize, word: u32| {
        let mut program = SINGLES.to_vec();
        program.insert(at, word);
        program
    };
    let programs: Vec<(&str, Vec<u32>)> = vec![
        ("singles", SINGLES.to_vec()),
        ("msr fpsr inside", with(2, MSR_FPSR_X5)),
        ("msr fpcr inside", with(2, MSR_FPCR_X6)),
        ("msr fpsr after the first", with(1, MSR_FPSR_X5)),
        (
            "doubles",
            vec![
                0x1e62_0820, // FMUL D0, D1, D2
                0x1e61_2803, // FADD D3, D0, D1
                0x1e62_1864, // FDIV D4, D3, D2
            ],
        ),
        (
            "compare and max",
            vec![
                0x1e22_0820, // FMUL S0, S1, S2
                0x1e22_2020, // FCMP S1, S2
                0x1e22_4827, // FMAX S7, S1, S2
                0x1e21_2803, // FADD S3, S0, S1
            ],
        ),
    ];
    let mut words: Vec<u32> = Vec::new();
    let mut entries = Vec::new();
    let mut names = Vec::new();
    for (name, program) in &programs {
        entries.push(BASE + 4 * words.len() as u64);
        names.push(*name);
        words.push(NOP);
        words.extend_from_slice(program);
        words.push(SVC_EXIT);
    }
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let translation = discover(
        &[CodeRegion {
            name: "fp_mode_group".into(),
            base: BASE,
            bytes,
        }],
        &entries,
        words.len() + 16,
    )
    .expect("fp_mode_group");
    let grouped = EmitOptions {
        inline_vector: true,
        inline_fp_scalar: true,
        fp_status_locals: true,
        hoist_fp_mode: true,
        ..options()
    };
    for (name, options) in [
        ("fp_mode_group_code", grouped),
        ("fp_mode_group_reference", EmitOptions::default()),
    ] {
        let (root, regions) = split(&translation, options);
        writeln!(out, "pub mod {name} {{").expect("write");
        writeln!(out, "pub const BASE: u64 = {BASE:#x};").expect("write");
        writeln!(out, "pub const WORDS: &[u32] = &{words:#x?};").expect("write");
        writeln!(out, "pub const ENTRIES: &[u64] = &{entries:#x?};").expect("write");
        writeln!(out, "pub const NAMES: &[&str] = &{names:?};").expect("write");
        for (region, source) in regions {
            writeln!(out, "pub mod gen_r_{region:x} {{\n{source}\n}}").expect("write");
        }
        writeln!(out, "{root}\n}}").expect("write");
    }
}

fn vector_run_code(out: &mut String) {
    use echora_lift::aarch64::{Instruction, decode};

    const BASE: u64 = 0x2a0_0000;
    const SVC_EXIT: u32 = 0xd400_00e1;
    const MSR_FPCR_X5: u32 = 0xd51b_4405;
    // Eligible words, one per form, registers narrowed to V0-V3 (Vm of a
    // by-element form: bits 16-19, M clear).
    let mut pool: Vec<u32> = Vec::new();
    let mut forms = std::collections::BTreeSet::new();
    let mut state = 0x7e57_a11e_0000_0013u64;
    for _ in 0..2_000_000 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let word = (state as u32 & !0x0e00_0000) | 0x0e00_0000;
        let Instruction::Simd(op) = decode(word, 0) else {
            continue;
        };
        if !echora_lift::simd::vector_run_op(&op) {
            continue;
        }
        // Rd, Rn and Vm narrowed; for a form with no Vm (bits 16-20 part of
        // its opcode, as FMAXV's), Rd and Rn alone.
        let random = (state >> 32) as u32;
        let narrowed = [
            word & !0x001f_03ff | random & 0x0003_0063,
            word & !0x0000_03ff | random & 0x0000_0063,
        ]
        .into_iter()
        .find(|&candidate| {
            matches!(decode(candidate, 0), Instruction::Simd(op) if echora_lift::simd::vector_run_op(&op))
        });
        let Some(narrowed) = narrowed else {
            continue;
        };
        let Instruction::Simd(narrowed_op) = decode(narrowed, 0) else {
            continue;
        };
        let text = format!("{narrowed_op:?}");
        let form: String = text.split(" rd:").next().unwrap_or(&text).to_owned();
        if forms.insert(form) {
            pool.push(narrowed);
        }
    }
    let mut words: Vec<u32> = Vec::new();
    let mut entries = Vec::new();
    for program in 0..600u64 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let length = 2 + (state % 5) as usize;
        entries.push(BASE + 4 * words.len() as u64);
        for index in 0..length {
            if program % 4 == 0 && index == length / 2 {
                words.push(MSR_FPCR_X5);
            }
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            words.push(pool[(state % pool.len() as u64) as usize]);
        }
        words.push(SVC_EXIT);
    }
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let translation = discover(
        &[CodeRegion {
            name: "vector_run".into(),
            base: BASE,
            bytes,
        }],
        &entries,
        words.len() + 16,
    )
    .expect("vector_run");
    for (name, runs) in [("vector_run_code", true), ("vector_run_reference", false)] {
        let options = EmitOptions {
            inline_vector: true,
            whole_vector_checks: true,
            fp_status_locals: true,
            window_memory: true,
            untagged_blocks: true,
            split_ticks: true,
            vector_float_runs: runs,
            ..options()
        };
        // The runs' lift has every per-instruction inline option in every
        // other region as well, so a run's exact copy is in some regions
        // those options' code and in the others the helpers'.
        let (root, regions) = if runs {
            split_alternate(
                &translation,
                options,
                EmitOptions {
                    inline_vector_unary: true,
                    inline_vector_int: true,
                    inline_vector_pairwise: true,
                    inline_fp_scalar: true,
                    inline_vector_misc: true,
                    ..options
                },
            )
        } else {
            split(&translation, options)
        };
        let formed: usize = regions
            .iter()
            .map(|(_, source)| source.matches("let o0 = ").count())
            .sum();
        writeln!(out, "pub mod {name} {{").expect("write");
        writeln!(out, "pub const BASE: u64 = {BASE:#x};").expect("write");
        writeln!(out, "pub const WORDS: &[u32] = &{words:#x?};").expect("write");
        writeln!(out, "pub const ENTRIES: &[u64] = &{entries:#x?};").expect("write");
        writeln!(
            out,
            "/// Forms sampled.\npub const FORMS: usize = {};",
            forms.len()
        )
        .expect("write");
        writeln!(
            out,
            "/// Vector float runs formed.\npub const RUNS: usize = {formed};"
        )
        .expect("write");
        for (region, source) in regions {
            writeln!(out, "pub mod gen_r_{region:x} {{\n{source}\n}}").expect("write");
        }
        writeln!(out, "{root}\n}}").expect("write");
    }
}

/// ECHORA_KERNEL_ABLATE (measuring only: the code is no longer faithful):
/// the kernels' emitted Rust without one class of per-block cost, to time
/// what it costs. A comma list of
/// - `charges`: no block charges (no budget, host boundaries or block
///   tags; the kernels still end at their SVC);
/// - `pc`: no stores of the PC that only a trap or a boundary would read
///   (an inlined successor's, and the window loop's after each block; a
///   window still stores the PC when it leaves);
/// - `memory`: guest loads and stores without the reservation and
///   alignment tests (volatile accesses at the host address).
fn ablate(source: &str, ablations: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let on = |name: &str| ablations.split(',').any(|a| a.trim() == name);
    let lines: Vec<&str> = source.lines().collect();
    for (index, line) in lines.iter().enumerate() {
        let next = lines.get(index + 1).copied().unwrap_or("");
        if on("charges") && line.starts_with("if !m.charge_block_at(") {
            continue;
        }
        if on("pc") {
            let constant = line
                .strip_prefix("m.cpu.pc = 0x")
                .and_then(|rest| rest.strip_suffix(';'))
                .is_some_and(|pc| u64::from_str_radix(pc, 16).is_ok());
            // An inlined successor's PC, followed by its charge (or, with
            // charges ablated, by its body).
            if constant
                && (next.starts_with("if !m.charge_block_at(") || on("charges"))
                && !next.contains("call")
                && !next.starts_with("m.cpu.regs[")
                && !next.contains("enter_window_")
            {
                continue;
            }
            if *line == "m.cpu.pc = next_pc;" && next == "m.finish_block()?;" {
                continue;
            }
            if let Some(rest) = line.strip_suffix(" => return Ok(()),") {
                // The window leaves: the PC is the next PC.
                let pc = if rest.starts_with("pc ") {
                    "pc"
                } else {
                    "next_pc"
                };
                out.push_str(&format!("{rest} => {{ m.cpu.pc = {pc}; return Ok(()) }}\n"));
                continue;
            }
        }
        out.push_str(line);
        out.push('\n');
    }
    if on("memory") {
        out = out
            .replace("wmem.load_u", "ablated_load_u")
            .replace("wmem.store_u", "ablated_store_u");
        for (name, ty) in [("8", "u8"), ("16", "u16"), ("32", "u32"), ("64", "u64")] {
            out = out
                .replace(
                    &format!("ablated_load_u{name}("),
                    &format!("ablated_load::<{ty}>(wmem, "),
                )
                .replace(
                    &format!("ablated_store_u{name}("),
                    &format!("ablated_store::<{ty}>(wmem, "),
                );
        }
        // The view's host base: the field of `WindowMemory` that is neither
        // the reservation's mask nor the size (the mask plus one).
        out.push_str(
            "#[allow(unsafe_code)]\n#[inline(always)]\nfn ablated_base(w: echora_guest::view::WindowMemory) -> usize { \
             let raw: [u64; 3] = unsafe { std::mem::transmute(w) }; \
             raw.into_iter().find(|v| !raw.iter().any(|m| m.wrapping_add(1).is_power_of_two() && (v == m || *v == m + 1))).unwrap_or(0) as usize }\n\
             #[allow(unsafe_code)]\n#[inline(always)]\nfn ablated_load<T: Copy>(w: echora_guest::view::WindowMemory, address: u64) -> T { \
             unsafe { std::ptr::read_volatile((ablated_base(w) + address as usize) as *const T) } }\n\
             #[allow(unsafe_code)]\n#[inline(always)]\nfn ablated_store<T: Copy>(w: echora_guest::view::WindowMemory, address: u64, value: T) { \
             unsafe { std::ptr::write_volatile((ablated_base(w) + address as usize) as *mut T, value) } }\n",
        );
    }
    out
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
    assert_eq!(
        report.jump_tables, 2,
        "the switch and gamefn kernels' jump tables"
    );
    // One 64 KiB region, as the game's regions hold code: calls between
    // the kernels' functions stay in their region.
    LIFTED.set(LIFTED.get() + translation.instructions.len());
    let (root, regions) = emit_rust_split_with(&translation, 16, kernel_options());
    let regions = match std::env::var("ECHORA_KERNEL_ABLATE") {
        Ok(ablations) if MODE.get().is_none() => regions
            .into_iter()
            .map(|(region, source)| (region, ablate(&source, &ablations)))
            .collect(),
        _ => regions,
    };
    // ECHORA_KERNEL_PAD=n (measuring layout sensitivity): a function of n
    // kept additions before the kernels' windows in their region, so the
    // same generated code lands at other addresses.
    let pad: usize = std::env::var("ECHORA_KERNEL_PAD")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    let padding = if pad == 0 {
        String::new()
    } else {
        let sum: String = (0..pad)
            .map(|index| format!(" + std::hint::black_box({index}u64)"))
            .collect();
        format!(
            "#[inline(never)]\nfn layout_pad() -> u64 {{ 0{sum} }}\n#[used]\nstatic LAYOUT_PAD: fn() -> u64 = layout_pad;\n"
        )
    };
    writeln!(out, "pub mod kernel_code {{").expect("write");
    // The census of each kernel's window (and the leaf functions'), as a
    // Markdown table (`echora_lift::census`).
    let mut rows = Vec::new();
    let mut named: Vec<(u64, &str)> = kernels::KERNEL_ENTRIES
        .iter()
        .map(|&(name, offset)| ((kernels::KERNEL_BASE + offset) >> 10, name))
        .collect();
    named.push(((kernels::KERNEL_BASE + 0x1000) >> 10, "(leaf functions)"));
    named.push((
        (kernels::KERNEL_BASE + kernels::GAMECALLS_FUNCTIONS[0]) >> 10,
        "(gamecalls update, hash)",
    ));
    named.push((
        (kernels::KERNEL_BASE + kernels::GAMECALLS_FUNCTIONS[1]) >> 10,
        "(gamecalls clamp, flags)",
    ));
    named.push((
        (kernels::KERNEL_BASE + kernels::CROSSING_FUNCTION) >> 10,
        "(crossing function head)",
    ));
    named.push((
        ((kernels::KERNEL_BASE + kernels::CROSSING_FUNCTION) >> 10) + 1,
        "(crossing function tail)",
    ));
    named.push((
        (kernels::KERNEL_BASE + kernels::GAMEFN_FUNCTION) >> 10,
        "(gamefn function)",
    ));
    for (_, source) in &regions {
        for (function, counts) in echora_lift::census::census(source) {
            // An entered window's body is its `enter_window_<w>`; its
            // `run_window_<w>` only calls it.
            if let Some(window) = function.strip_prefix("run_window_")
                && source.contains(&format!("pub fn enter_window_{window}("))
            {
                continue;
            }
            let label = named
                .iter()
                .find(|(window, _)| {
                    function == format!("run_window_{window:x}")
                        || function == format!("enter_window_{window:x}")
                })
                .map_or(function.clone(), |(_, name)| (*name).to_owned());
            rows.push((label, counts));
        }
    }
    // Not in the golden hashes' emissions (which hash this module's text).
    if MODE.get().is_none() {
        writeln!(
            out,
            "/// What keeps the kernels' vector float operations out of runs (`echora_lift::vector_float_report`).\npub const VECTOR_FLOAT_REPORT: &str = {:?};",
            echora_lift::vector_float_report::vector_float_report(&translation, 16, None, None)
                .to_string()
        )
        .expect("write");
        writeln!(
            out,
            "/// The census of the kernels' windows (`echora_lift::census::table`).\npub const KERNEL_CENSUS: &str = {:?};",
            echora_lift::census::table(&rows)
        )
        .expect("write");
    }
    for (region, source) in regions {
        let source = match source.find("\n#[inline(never)]") {
            Some(at) if pad != 0 => format!("{}\n{padding}{}", &source[..at], &source[at..]),
            _ => source,
        };
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
    // As split windows, and with lifted functions in one 64 KiB region (as
    // the game's regions hold code), whose calls between them are direct.
    for (name, functions, function_table, call_summaries, transitive_summaries, regalloc) in [
        ("call_code", false, false, false, false, false),
        ("call_function_code", true, false, false, false, false),
        ("call_function_table_code", true, true, false, false, false),
        ("call_summary_code", true, false, true, false, false),
        ("call_summary_table_code", true, true, true, false, false),
        ("call_transitive_code", true, true, true, true, false),
        ("call_regalloc_code", true, true, true, true, true),
    ] {
        let (root, regions) = if functions {
            LIFTED.set(LIFTED.get() + translation.instructions.len());
            emit_rust_split_with(
                &translation,
                16,
                EmitOptions {
                    functions: true,
                    function_table,
                    call_summaries,
                    transitive_summaries,
                    regalloc,
                    ..options()
                },
            )
        } else {
            split(&translation, options())
        };
        let direct = regions
            .iter()
            .map(|(_, source)| source.matches("run_function_").count())
            .sum::<usize>();
        let summarized = regions
            .iter()
            .map(|(_, source)| source.matches("if m.cpu.pc != 0x").count())
            .sum::<usize>();
        writeln!(out, "pub mod {name} {{").expect("write");
        writeln!(out, "pub const BASE: u64 = {BASE:#x};").expect("write");
        writeln!(out, "pub const WORDS: &[u32] = &{words:#x?};").expect("write");
        writeln!(out, "pub const CALLERS: &[u64] = &{caller_addresses:#x?};").expect("write");
        writeln!(out, "/// Lifted functions and direct calls named in the emission.\npub const FUNCTION_NAMES: usize = {direct};").expect("write");
        writeln!(out, "/// Calls of leaves with an effect summary.\npub const SUMMARIZED_CALLS: usize = {summarized};").expect("write");
        for (region, source) in regions {
            writeln!(out, "pub mod gen_r_{region:x} {{\n{source}\n}}").expect("write");
        }
        writeln!(out, "{root}\n}}").expect("write");
    }
    // With window calls (`EmitOptions::window_calls`), in one 64 KiB
    // region and in 1 KiB regions (calls across windows through the
    // window-call table). Not in the golden hashes' emissions.
    if MODE.get().is_none() {
        for (name, bits) in [("call_window_code", 16), ("call_window_table_code", 10)] {
            LIFTED.set(LIFTED.get() + translation.instructions.len());
            let (mut root, regions) = emit_rust_split_with(
                &translation,
                bits,
                EmitOptions {
                    window_calls: true,
                    ..options()
                },
            );
            if !root.contains("fn window_calls_installed") {
                root.push_str("\npub fn window_calls_installed() -> bool { false }\n");
            }
            let summarized = regions
                .iter()
                .map(|(_, source)| {
                    source.matches("= enter_window_").count()
                        + source.matches("= window_call_at(").count()
                })
                .sum::<usize>();
            writeln!(out, "pub mod {name} {{").expect("write");
            writeln!(
                out,
                "/// Summarized window calls.\npub const SUMMARIZED_CALLS: usize = {summarized};"
            )
            .expect("write");
            for (region, source) in regions {
                writeln!(out, "pub mod gen_r_{region:x} {{\n{source}\n}}").expect("write");
            }
            writeln!(out, "{root}\n}}").expect("write");
        }
    }
    // Mixed: every other 1 KiB region with functions, the table and
    // (transitive) summaries, the others with the base options, so calls
    // cross between regions with and without them both ways.
    let listed: std::collections::BTreeSet<u64> = translation
        .blocks
        .iter()
        .map(|block| block.start >> 10)
        .filter(|region| region % 2 == 0)
        .collect();
    LIFTED.set(LIFTED.get() + translation.instructions.len());
    let mut regions = Vec::new();
    let root = echora_lift::codegen::emit_rust_split_to_regions(
        &translation,
        10,
        // Without functions whatever the build's features.
        EmitOptions {
            functions: false,
            function_table: false,
            call_summaries: false,
            transitive_summaries: false,
            ..options()
        },
        EmitOptions {
            functions: true,
            function_table: true,
            call_summaries: true,
            transitive_summaries: true,
            ..options()
        },
        &listed,
        &mut |region, source, _| {
            regions.push((region, source));
            Ok::<(), std::convert::Infallible>(())
        },
    )
    .expect("no sink errors");
    let functions = regions
        .iter()
        .map(|(_, source)| source.matches("pub fn run_function_").count())
        .sum::<usize>();
    let unlisted_functions = regions
        .iter()
        .filter(|(region, _)| !listed.contains(region))
        .map(|(_, source)| {
            source.matches("run_function_").count() + source.matches("function_at(").count()
        })
        .sum::<usize>();
    writeln!(out, "pub mod call_mixed_code {{").expect("write");
    writeln!(out, "/// Lifted functions, and mentions of functions in unlisted regions.\npub const FUNCTIONS: usize = {functions};\npub const UNLISTED_FUNCTIONS: usize = {unlisted_functions};").expect("write");
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
    // No indirect branch looks like a table dispatch that was missed.
    let dispatches = echora_lift::dispatch::audit(&translation);
    assert_eq!(
        echora_lift::dispatch::unrecovered(&dispatches).count(),
        0,
        "{}",
        echora_lift::dispatch::report(&translation, &dispatches)
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
    let words = stress::stress_words();
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let region = CodeRegion {
        name: "stress".into(),
        base: stress::STRESS_BASE,
        bytes,
    };
    let entries: Vec<u64> = stress::STRESS_ENTRIES
        .iter()
        .map(|(_, offset)| stress::STRESS_BASE + offset)
        .collect();
    let translation = discover(&[region], &entries, words.len()).expect("stress");
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

/// A synthetic PLT for `tests/plt.rs`: a caller window (BL to three stubs,
/// and a BL to a function that tail-branches to the third), a window of
/// stubs `ADRP X16; LDR X17, [X16, #off]; ADD X16, X16, #off; BR X17`
/// loading GOT slots in the data page, and the functions they reach. The
/// translation predicts the slots hold FA (in the caller's window, so in
/// its region: a direct window call), FB and FC (another region); FD and FE
/// are what a test rewrites the slots to. Split emission with the default
/// options, which inline the stubs.
fn plt_code(out: &mut String) {
    use echora_lift::codegen::plt_report;

    const BASE: u64 = 0x120_0000;
    const GOT: u64 = 0x80_0000;
    const SVC_7: u32 = 0xd400_00e1;
    const RET: u32 = 0xd65f_03c0;
    let (stub_a, stub_b, stub_c) = (BASE + 0x400, BASE + 0x410, BASE + 0x420);
    let tailer = BASE + 0x100;
    let (fa, fe) = (BASE + 0x200, BASE + 0x300);
    let (fb, fc, fd) = (BASE + 0x800, BASE + 0x900, BASE + 0xa00);
    let branch = |link: bool, from: u64, to: u64| {
        (if link { 0x9400_0000 } else { 0x1400_0000 })
            | (((to.wrapping_sub(from) as i64 / 4) as u32) & 0x03ff_ffff)
    };
    let mut words = vec![0u32; 0xc00 / 4];
    let mut place = |address: u64, program: &[u32]| {
        let at = ((address - BASE) / 4) as usize;
        words[at..at + program.len()].copy_from_slice(program);
    };
    place(
        BASE,
        &[
            0xd280_00a0, // movz x0, #5
            0xd280_0001, // movz x1, #0
            branch(true, BASE + 8, stub_a),
            branch(true, BASE + 12, stub_b),
            branch(true, BASE + 16, tailer),
            branch(true, BASE + 20, stub_c),
            SVC_7,
        ],
    );
    place(tailer, &[branch(false, tailer, stub_c)]);
    place(fa, &[0x9100_0400, RET]); // add x0, x0, #1
    place(fe, &[0xd280_0122, RET]); // movz x2, #9
    place(fb, &[0x8b00_0000, RET]); // add x0, x0, x0
    place(fc, &[0x8b00_0021, RET]); // add x1, x1, x0
    place(fd, &[0xd100_0800, RET]); // sub x0, x0, #2
    for (index, stub) in [stub_a, stub_b, stub_c].into_iter().enumerate() {
        let offset = 8 * index as u32;
        let page = ((GOT >> 12) as i64 - (stub >> 12) as i64) as u32;
        place(
            stub,
            &[
                0x9000_0010 | (page & 3) << 29 | (page >> 2 & 0x7_ffff) << 5,
                0xf940_0211 | (offset / 8) << 10,
                0x9100_0210 | offset << 10,
                0xd61f_0220,
            ],
        );
    }
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let region = CodeRegion {
        name: "plt".into(),
        base: BASE,
        bytes,
    };
    let entries = [BASE, fa, fb, fc, fd, fe];
    let mut translation = discover(&[region], &entries, words.len()).expect("plt");
    translation.got_words = [(GOT, fa), (GOT + 8, fb), (GOT + 16, fc)].into();
    // 1 KiB regions whatever the build: the program puts its targets in
    // and out of the callers' regions.
    let report = plt_report(&translation, 10);
    // Counted as with lifted functions alone: no structured copies.
    let (root, regions) = split_at(
        &translation,
        EmitOptions {
            regalloc: false,
            ..options()
        },
        10,
    );
    let direct = regions
        .iter()
        .map(|(_, source)| {
            source
                .split("m.call_window(")
                .skip(1)
                .filter(|call| {
                    call.split(')')
                        .next()
                        .is_some_and(|arguments| arguments.contains(", run_window_"))
                })
                .count()
        })
        .sum::<usize>();
    // The direct call runs only when X17 holds the predicted word.
    let guarded = regions
        .iter()
        .any(|(_, source)| source.contains(&format!("if r.x17 == {fa:#x} {{\n")));
    writeln!(out, "pub mod plt_code {{").expect("write");
    writeln!(out, "pub const BASE: u64 = {BASE:#x};").expect("write");
    writeln!(out, "pub const GOT: u64 = {GOT:#x};").expect("write");
    writeln!(out, "pub const WORDS: &[u32] = &{words:#x?};").expect("write");
    writeln!(
        out,
        "pub const STUBS: [u64; 3] = [{stub_a:#x}, {stub_b:#x}, {stub_c:#x}];\n\
         pub const FA: u64 = {fa:#x};\n\
         pub const FB: u64 = {fb:#x};\n\
         pub const FC: u64 = {fc:#x};\n\
         pub const FD: u64 = {fd:#x};\n\
         pub const FE: u64 = {fe:#x};"
    )
    .expect("write");
    writeln!(
        out,
        "/// (stubs, predicted, inlined, BL sites, direct BL sites, B sites).\n\
         pub const REPORT: (usize, usize, usize, usize, usize, usize) = ({}, {}, {}, {}, {}, {});\n\
         /// Direct window calls in the emitted regions.\n\
         pub const DIRECT_CALLS: usize = {direct};\n\
         /// Whether the direct call is guarded by the predicted word.\n\
         pub const GUARDED: bool = {guarded};",
        report.stubs,
        report.predicted,
        report.inlined,
        report.call_sites,
        report.direct_call_sites,
        report.tail_sites
    )
    .expect("write");
    for (region, source) in regions {
        writeln!(out, "pub mod gen_r_{region:x} {{\n{source}\n}}").expect("write");
    }
    writeln!(out, "{root}\n}}").expect("write");
}

/// Two callers for `tests/spills.rs`, each in its own window, of one callee
/// in a third (ADD X3, X1, X2; ADD X3, X3, X4; ADD X3, X3, X7;
/// ADD X6, X6, X3; RET): a loop (ADD X1 before the BL; ADD X2 and ADD X4
/// after it; SUBS X5 and B.NE back) whose later passes need what the
/// previous one wrote stored before the call, and a straight caller (ADD
/// X1; BL; ADD X2; SVC) whose call needs only X1 stored. X7 no caller
/// touches.
fn spill_code(out: &mut String) {
    const BASE: u64 = 0x130_0000;
    const SVC_7: u32 = 0xd400_00e1;
    let (looping, straight, callee) = (BASE, BASE + 0x400, BASE + 0x800);
    let branching = BASE + 0xc00;
    let bl = |from: u64, to: u64| {
        0x9400_0000 | (((to.wrapping_sub(from) as i64 / 4) as u32) & 0x03ff_ffff)
    };
    let mut words = vec![0u32; 0x1000 / 4];
    let mut place = |address: u64, program: &[u32]| {
        let at = ((address - BASE) / 4) as usize;
        words[at..at + program.len()].copy_from_slice(program);
    };
    // ADD X1; CBZ X8 to a BL (inlined into the CBZ's arm); else a BL in an
    // arm of its own (an entry), whose call still needs X1 stored.
    place(
        branching,
        &[
            0x9100_0421,
            0xb400_0088, // cbz x8, +0x10
            bl(branching + 8, callee),
            SVC_7,
            0,
            bl(branching + 0x14, callee),
            SVC_7,
        ],
    );
    place(
        looping,
        &[
            0x9100_0421, // add x1, x1, #1
            bl(looping + 4, callee),
            0x9100_1442,                                      // add x2, x2, #5
            0x9100_1c84,                                      // add x4, x4, #7
            0xf100_04a5,                                      // subs x5, x5, #1
            0x5400_0001 | (((-5i32) as u32) & 0x7_ffff) << 5, // b.ne the add
            SVC_7,
        ],
    );
    place(
        straight,
        &[0x9100_0421, bl(straight + 4, callee), 0x9100_1442, SVC_7],
    );
    place(
        callee,
        &[
            0x8b02_0023,
            0x8b04_0063,
            0x8b07_0063,
            0x8b03_00c6,
            0xd65f_03c0,
        ],
    );
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let region = CodeRegion {
        name: "spills".into(),
        base: BASE,
        bytes,
    };
    let translation = discover(
        &[region],
        &[looping, straight, callee, branching, branching + 8],
        words.len(),
    )
    .expect("spills");
    let (root, regions) = split(&translation, options());
    // The straight caller's call (through the window table, or with
    // window calls its window's): the line storing before it.
    let window = format!("fn run_window_{:x}(", straight >> 10);
    let call_line = regions
        .iter()
        .find_map(|(_, source)| {
            let body = &source[source.find(&window)?..];
            body.lines()
                .find(|line| line.contains("let called = "))
                .map(str::to_owned)
        })
        .expect("the straight caller's call");
    let stores_x1 = call_line.contains("m.cpu.regs[1] = r.x1;");
    let stores_x2 = call_line.contains("m.cpu.regs[2] = r.x2;");
    let selective = !options().structured;
    writeln!(out, "pub mod spill_code {{").expect("write");
    writeln!(out, "pub const BASE: u64 = {BASE:#x};").expect("write");
    writeln!(out, "pub const WORDS: &[u32] = &{words:#x?};").expect("write");
    writeln!(
        out,
        "pub const LOOPING: u64 = {looping:#x};\npub const STRAIGHT: u64 = {straight:#x};\n\
         pub const BRANCHING: u64 = {branching:#x};\n\
         /// Whether the straight caller stores X1 and X2 before its call.\n\
         pub const STRAIGHT_STORES: (bool, bool) = ({stores_x1}, {stores_x2});\n\
         /// Whether stores are chosen per call (not with structured\n\
         /// emission, whose arms store every written local).\n\
         pub const SELECTIVE: bool = {selective};"
    )
    .expect("write");
    for (region, source) in regions {
        writeln!(out, "pub mod gen_r_{region:x} {{\n{source}\n}}").expect("write");
    }
    writeln!(out, "{root}\n}}").expect("write");
}

/// Float runs for `tests/float_runs.rs`: programs of a NOP, 8 to 16 scalar
/// single-precision operations (FADD, FSUB, FMUL, FNMUL, FDIV, FMADD,
/// FMSUB, FNMADD, FNMSUB, FSQRT, with FMOV, FABS and FNEG between them) on
/// S0-S7, and an SVC, one per 1 KiB window, lifted with float runs
/// (`float_run_code`) and without (`float_run_reference`).
fn float_run_code(out: &mut String) {
    use echora_lift::codegen::float_run_report;

    const BASE: u64 = 0x200_0000;
    const PROGRAMS: usize = 48;
    const WINDOW: usize = 0x400 / 4;
    let mut state: u64 = 0x5851_f42d_4c95_7f2d;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 16) as u32
    };
    let mut words = vec![0u32; PROGRAMS * WINDOW];
    let mut entries = Vec::new();
    for program in 0..PROGRAMS {
        let mut code = vec![0xd503_201f]; // nop
        let arithmetic = 8 + next() as usize % 9;
        let mut made = 0;
        while made < arithmetic {
            let (d, n, m, a) = (next() % 8, next() % 8, next() % 8, next() % 8);
            let word = match next() % 13 {
                0 => 0x1e20_2800 | m << 16 | n << 5 | d,     // fadd
                1 => 0x1e20_3800 | m << 16 | n << 5 | d,     // fsub
                2 | 3 => 0x1e20_0800 | m << 16 | n << 5 | d, // fmul
                4 => 0x1e20_8800 | m << 16 | n << 5 | d,     // fnmul
                5 => 0x1e20_1800 | m << 16 | n << 5 | d,     // fdiv
                6 | 7 => 0x1f00_0000 | m << 16 | a << 10 | n << 5 | d, // fmadd
                8 => 0x1f00_8000 | m << 16 | a << 10 | n << 5 | d, // fmsub
                9 => 0x1f20_0000 | m << 16 | a << 10 | n << 5 | d, // fnmadd
                10 => 0x1f20_8000 | m << 16 | a << 10 | n << 5 | d, // fnmsub
                11 => 0x1e21_c000 | n << 5 | d,              // fsqrt
                _ => {
                    // fmov, fabs, fneg (not counted)
                    let op = [0x1e20_4000, 0x1e20_c000, 0x1e21_4000][next() as usize % 3];
                    code.push(op | n << 5 | d);
                    continue;
                }
            };
            code.push(word);
            made += 1;
        }
        code.push(0xd400_00e1); // svc #7
        let at = program * WINDOW;
        words[at..at + code.len()].copy_from_slice(&code);
        entries.push(BASE + (at * 4) as u64);
    }
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let region = CodeRegion {
        name: "float runs".into(),
        base: BASE,
        bytes,
    };
    let translation = discover(&[region], &entries, words.len()).expect("float runs");
    let (runs, operations) = float_run_report(&translation, false);
    for (name, float_runs) in [("float_run_code", true), ("float_run_reference", false)] {
        let options = EmitOptions {
            float_runs,
            ..options()
        };
        // The runs' lift (not in the golden emissions) has
        // `inline-fp-scalar` in every other region as well, so a run's
        // FSQRT in its exact copy is in some regions that option's code.
        let (root, regions) = if float_runs && MODE.get().is_none() {
            split_alternate(
                &translation,
                options,
                EmitOptions {
                    inline_vector: true,
                    inline_fp_scalar: true,
                    ..options
                },
            )
        } else {
            split(&translation, options)
        };
        writeln!(out, "pub mod {name} {{").expect("write");
        writeln!(out, "pub const BASE: u64 = {BASE:#x};").expect("write");
        writeln!(out, "pub const WORDS: &[u32] = &{words:#x?};").expect("write");
        writeln!(out, "pub const ENTRIES: &[u64] = &{entries:#x?};").expect("write");
        writeln!(
            out,
            "/// Float runs found, and the arithmetic operations they hold.\n\
             pub const RUNS: (usize, usize) = ({runs}, {operations});"
        )
        .expect("write");
        for (region, source) in regions {
            writeln!(out, "pub mod gen_r_{region:x} {{\n{source}\n}}").expect("write");
        }
        writeln!(out, "{root}\n}}").expect("write");
    }
}

/// A loop for `tests/deferred_charges.rs`, lifted with deferred charges
/// (`deferred_charge_code`) and without (`deferred_charge_reference`), both
/// with split ticks: X3 counts to X0 through a diamond on X1 (one side
/// inlined into the CBZ's arm, the other its own arm) and a join, which
/// two predecessors reach forward, so neither it nor the second side
/// checks; then a block that does not check either leaves for another
/// window, which calls a leaf in a third and ends in an SVC.
fn deferred_charge_code(out: &mut String) {
    const BASE: u64 = 0x150_0000;
    let leaf = BASE + 0x400;
    let bl = |from: u64, to: u64| {
        0x9400_0000 | (((to.wrapping_sub(from) as i64 / 4) as u32) & 0x03ff_ffff)
    };
    let far = BASE + 0x800;
    let branch = |from: u64, to: u64| {
        0x1400_0000 | (((to.wrapping_sub(from) as i64 / 4) as u32) & 0x03ff_ffff)
    };
    let mut words = vec![0u32; 0xc00 / 4];
    let program = [
        0xd280_0003, // 0x00 movz x3, #0
        0xb400_0061, // 0x04 L: cbz x1, A (+0xc)
        0x9100_0c42, // 0x08 add x2, x2, #3
        0x1400_0002, // 0x0c b J
        0xca01_0042, // 0x10 A: eor x2, x2, x1
        0x9100_0463, // 0x14 J: add x3, x3, #1
        0xd200_0021, // 0x18 eor x1, x1, #1
        0xeb00_007f, // 0x1c cmp x3, x0
        0x54ff_ff2b, // 0x20 b.lt L
        0x9100_04c6, // 0x24 add x6, x6, #1 (deferred, then out of the window)
        branch(BASE + 0x28, far),
    ];
    words[..program.len()].copy_from_slice(&program);
    // The leaf: add x5, x5, #1; ret.
    words[0x100..0x102].copy_from_slice(&[0x9100_04a5, 0xd65f_03c0]);
    // Far: bl leaf; add x4, x2, x3; svc #7.
    words[0x200..0x203].copy_from_slice(&[bl(far, leaf), 0x8b03_0044, 0xd400_00e1]);
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let region = CodeRegion {
        name: "deferred charges".into(),
        base: BASE,
        bytes,
    };
    let translation =
        discover(&[region], &[BASE, leaf, far], words.len()).expect("deferred charges");
    for (name, deferred_charges, steps_local, cached_stop, arm_charges) in [
        ("deferred_charge_code", true, false, false, false),
        ("deferred_charge_reference", false, false, false, false),
        ("steps_local_code", false, true, false, false),
        ("cached_stop_code", false, false, true, false),
        ("arm_charge_code", false, false, false, true),
    ] {
        let options = EmitOptions {
            split_ticks: true,
            deferred_charges,
            steps_local,
            cached_stop,
            arm_charges,
            untagged_blocks: steps_local || cached_stop || arm_charges,
            ..options()
        };
        let (root, regions) = split(&translation, options);
        let deferred: usize = regions
            .iter()
            .map(|(_, source)| source.matches("*pending += ").count())
            .sum();
        let cached: usize = regions
            .iter()
            .map(|(_, source)| {
                source.matches(" <= stop {").count() + source.matches("if arm_fast {").count()
            })
            .sum();
        writeln!(out, "pub mod {name} {{").expect("write");
        writeln!(out, "pub const BASE: u64 = {BASE:#x};").expect("write");
        writeln!(out, "pub const WORDS: &[u32] = &{words:#x?};").expect("write");
        writeln!(
            out,
            "/// Deferred block charges in the emission, and charges against the window's stop word.\npub const DEFERRED: usize = {deferred};\npub const CACHED: usize = {cached};"
        )
        .expect("write");
        for (region, source) in regions {
            writeln!(out, "pub mod gen_r_{region:x} {{\n{source}\n}}").expect("write");
        }
        writeln!(out, "{root}\n}}").expect("write");
    }
}

/// Float runs with loads for `tests/float_runs.rs`: as `float_run_code`,
/// with LDR, LDUR and LDP of S registers among the operations (post- and
/// pre-indexed and offset forms on X9, register offsets with X10, and LDR
/// from SP), lifted with inline vector operations and float runs holding
/// loads (`float_run_load_code`) and without float runs
/// (`float_run_load_reference`).
fn float_run_load_code(out: &mut String) {
    use echora_lift::codegen::float_run_report;

    const BASE: u64 = 0x210_0000;
    const PROGRAMS: usize = 48;
    const WINDOW: usize = 0x400 / 4;
    let mut state: u64 = 0x1f83_d9ab_fb41_bd6b;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 16) as u32
    };
    let mut words = vec![0u32; PROGRAMS * WINDOW];
    let mut entries = Vec::new();
    for program in 0..PROGRAMS {
        let mut code = vec![0xd503_201f]; // nop
        let arithmetic = 8 + next() as usize % 9;
        let mut made = 0;
        while made < arithmetic {
            let (d, n, m, a) = (next() % 8, next() % 8, next() % 8, next() % 8);
            if next() % 3 == 0 {
                let imm9 = |value: i32| (value as u32 & 0x1ff) << 12;
                code.push(match next() % 7 {
                    0 => 0xbc40_0400 | imm9([4, 8, -4][next() as usize % 3]) | 9 << 5 | d, // ldr s, [x9], #imm
                    1 => 0xbc40_0c00 | imm9([4, -8][next() as usize % 2]) | 9 << 5 | d, // ldr s, [x9, #imm]!
                    2 => 0xbd40_0000 | (next() % 16) << 10 | 9 << 5 | d, // ldr s, [x9, #imm*4]
                    3 => 0xbc40_0000 | imm9(-(next() as i32 % 16) * 4) | 9 << 5 | d, // ldur s, [x9, #-imm]
                    4 => 0xbc60_7800 | 10 << 16 | 9 << 5 | d, // ldr s, [x9, x10, lsl #2]
                    5 => {
                        // ldp s, s, [x9, #imm]{!} or [x9], #imm
                        let form = [0x2d40_0000, 0x2dc0_0000, 0x2cc0_0000][next() as usize % 3];
                        let imm7 = ([2, -2, 4][next() as usize % 3] as u32 & 0x7f) << 15;
                        let second = (d + 1) % 8;
                        form | imm7 | second << 10 | 9 << 5 | d
                    }
                    _ => 0xbd40_0000 | 2 << 10 | 31 << 5 | d, // ldr s, [sp, #8]
                });
                continue;
            }
            let word = match next() % 13 {
                0 => 0x1e20_2800 | m << 16 | n << 5 | d,
                1 => 0x1e20_3800 | m << 16 | n << 5 | d,
                2 | 3 => 0x1e20_0800 | m << 16 | n << 5 | d,
                4 => 0x1e20_8800 | m << 16 | n << 5 | d,
                5 => 0x1e20_1800 | m << 16 | n << 5 | d,
                6 | 7 => 0x1f00_0000 | m << 16 | a << 10 | n << 5 | d,
                8 => 0x1f00_8000 | m << 16 | a << 10 | n << 5 | d,
                9 => 0x1f20_0000 | m << 16 | a << 10 | n << 5 | d,
                10 => 0x1f20_8000 | m << 16 | a << 10 | n << 5 | d,
                11 => 0x1e21_c000 | n << 5 | d,
                _ => {
                    let op = [0x1e20_4000, 0x1e20_c000, 0x1e21_4000][next() as usize % 3];
                    code.push(op | n << 5 | d);
                    continue;
                }
            };
            code.push(word);
            made += 1;
        }
        code.push(0xd400_00e1); // svc #7
        let at = program * WINDOW;
        words[at..at + code.len()].copy_from_slice(&code);
        entries.push(BASE + (at * 4) as u64);
    }
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let region = CodeRegion {
        name: "float runs with loads".into(),
        base: BASE,
        bytes,
    };
    let translation = discover(&[region], &entries, words.len()).expect("float runs with loads");
    let (runs, operations) = float_run_report(&translation, true);
    for (name, float_runs) in [
        ("float_run_load_code", true),
        ("float_run_load_reference", false),
    ] {
        let options = EmitOptions {
            inline_vector: true,
            float_runs,
            float_run_loads: true,
            ..options()
        };
        let (root, regions) = split(&translation, options);
        writeln!(out, "pub mod {name} {{").expect("write");
        writeln!(out, "pub const BASE: u64 = {BASE:#x};").expect("write");
        writeln!(out, "pub const WORDS: &[u32] = &{words:#x?};").expect("write");
        writeln!(out, "pub const ENTRIES: &[u64] = &{entries:#x?};").expect("write");
        writeln!(
            out,
            "/// Float runs found (with loads), and the arithmetic operations they hold.\n\
             pub const RUNS: (usize, usize) = ({runs}, {operations});"
        )
        .expect("write");
        for (region, source) in regions {
            writeln!(out, "pub mod gen_r_{region:x} {{\n{source}\n}}").expect("write");
        }
        writeln!(out, "{root}\n}}").expect("write");
    }
}

/// Leaves that leave early, for `tests/early_exits.rs`: a caller writes X4
/// (which neither leaf touches), calls leaf A (X1 += 1, and with X5 nonzero
/// it points X30 out of its region before its RET, to code that adds X4 to
/// X2 and branches back to the caller), adds X4 and X1 into X6, and calls
/// leaf B (X1 += 2, then SVC #7, which ends the run: B can end outside its
/// body, so it gets no summary). Lifted with functions and call summaries, directly
/// (`early_exit_code`) and through the function table
/// (`early_exit_table_code`).
fn early_exit_code(out: &mut String) {
    const BASE: u64 = 0x160_0000;
    let (leaf_a, leaf_b, far) = (BASE + 0x400, BASE + 0x800, BASE + 0x1_0000);
    let bl = |from: u64, to: u64| {
        0x9400_0000 | (((to.wrapping_sub(from) as i64 / 4) as u32) & 0x03ff_ffff)
    };
    let branch = |from: u64, to: u64| {
        0x1400_0000 | (((to.wrapping_sub(from) as i64 / 4) as u32) & 0x03ff_ffff)
    };
    let mut words = vec![0u32; 0x1_0400 / 4];
    let mut place = |address: u64, program: &[u32]| {
        let at = ((address - BASE) / 4) as usize;
        words[at..at + program.len()].copy_from_slice(program);
    };
    place(
        BASE,
        &[
            0xd280_00e4, // movz x4, #7
            bl(BASE + 4, leaf_a),
            0x8b01_0086, // add x6, x4, x1
            bl(BASE + 12, leaf_b),
            0xd400_00e1, // svc #7
        ],
    );
    let adr = |from: u64, to: u64, rd: u32| {
        let imm = to.wrapping_sub(from) as u32 & 0x1f_ffff;
        0x1000_0000 | (imm & 3) << 29 | (imm >> 2) << 5 | rd
    };
    place(
        leaf_a,
        &[
            0x9100_0421,              // add x1, x1, #1
            0xb400_0045,              // cbz x5, +8
            adr(leaf_a + 8, far, 30), // adr x30, far
            0xd65f_03c0,              // ret
        ],
    );
    place(leaf_b, &[0x9100_0821, 0xd400_00e1]); // add x1, x1, #2; svc #7
    // add x2, x2, x4; b back to the caller after its first call.
    place(far, &[0x8b04_0042, branch(far + 4, BASE + 8)]);
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let region = CodeRegion {
        name: "early exits".into(),
        base: BASE,
        bytes,
    };
    let translation = discover(&[region], &[BASE, far], words.len()).expect("early exits");
    for (name, function_table, regalloc) in [
        ("early_exit_code", false, false),
        ("early_exit_table_code", true, false),
        ("early_exit_regalloc_code", true, true),
    ] {
        LIFTED.set(LIFTED.get() + translation.instructions.len());
        let (root, regions) = emit_rust_split_with(
            &translation,
            16,
            EmitOptions {
                functions: true,
                function_table,
                call_summaries: true,
                regalloc,
                ..options()
            },
        );
        let summarized = regions
            .iter()
            .map(|(_, source)| source.matches("if m.cpu.pc != 0x").count())
            .sum::<usize>();
        writeln!(out, "pub mod {name} {{").expect("write");
        writeln!(out, "pub const BASE: u64 = {BASE:#x};").expect("write");
        writeln!(out, "pub const WORDS: &[u32] = &{words:#x?};").expect("write");
        writeln!(out, "/// Calls of leaves with an effect summary.\npub const SUMMARIZED_CALLS: usize = {summarized};").expect("write");
        for (region, source) in regions {
            writeln!(out, "pub mod gen_r_{region:x} {{\n{source}\n}}").expect("write");
        }
        writeln!(out, "{root}\n}}").expect("write");
    }
}

/// Two levels of summarized calls for `tests/early_exits.rs`: window A
/// writes X4 and X8 and calls function C, which saves X29/X30, writes X7,
/// calls leaf D, adds X7 to X1, restores X29/X30 and returns; D adds X8 to
/// X1 (only D reads X8: A stores it because C's summary includes D's), adds
/// one, and with X5 nonzero points X30 out of its region before its RET, to
/// code that adds X4 (held by A) and X7 (held by C) to X2 and branches back
/// into C after its call. Lifted with
/// transitive summaries, directly (`two_level_code`) and through the
/// table (`two_level_table_code`).
fn two_level_code(out: &mut String) {
    const BASE: u64 = 0x180_0000 + 0x40_0000;
    let (c, d, far) = (BASE + 0x400, BASE + 0x800, BASE + 0x1_0000);
    let bl = |from: u64, to: u64| {
        0x9400_0000 | (((to.wrapping_sub(from) as i64 / 4) as u32) & 0x03ff_ffff)
    };
    let branch = |from: u64, to: u64| {
        0x1400_0000 | (((to.wrapping_sub(from) as i64 / 4) as u32) & 0x03ff_ffff)
    };
    let mut words = vec![0u32; 0x1_0400 / 4];
    let mut place = |address: u64, program: &[u32]| {
        let at = ((address - BASE) / 4) as usize;
        words[at..at + program.len()].copy_from_slice(program);
    };
    place(
        BASE,
        &[
            0xd280_00e4, // movz x4, #7
            0xd280_0068, // movz x8, #3
            bl(BASE + 8, c),
            0x8b01_0086, // add x6, x4, x1
            0xd400_00e1, // svc #7
        ],
    );
    place(
        c,
        &[
            0xa9bf_7bfd, // stp x29, x30, [sp, #-16]!
            0xd280_00a7, // movz x7, #5
            bl(c + 8, d),
            0x8b07_0021, // add x1, x1, x7
            0xa8c1_7bfd, // ldp x29, x30, [sp], #16
            0xd65f_03c0, // ret
        ],
    );
    let adr = |from: u64, to: u64, rd: u32| {
        let imm = to.wrapping_sub(from) as u32 & 0x1f_ffff;
        0x1000_0000 | (imm & 3) << 29 | (imm >> 2) << 5 | rd
    };
    place(
        d,
        &[
            0x8b08_0021,          // add x1, x1, x8
            0x9100_0421,          // add x1, x1, #1
            0xb400_0045,          // cbz x5, +8
            adr(d + 12, far, 30), // adr x30, far
            0xd65f_03c0,          // ret
        ],
    );
    // add x2, x2, x4; add x2, x2, x7; b back into C after its call.
    place(far, &[0x8b04_0042, 0x8b07_0042, branch(far + 8, c + 12)]);
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let region = CodeRegion {
        name: "two levels".into(),
        base: BASE,
        bytes,
    };
    let translation = discover(&[region], &[BASE, far], words.len()).expect("two levels");
    for (name, function_table, regalloc) in [
        ("two_level_code", false, false),
        ("two_level_table_code", true, false),
        ("two_level_regalloc_code", true, true),
    ] {
        LIFTED.set(LIFTED.get() + translation.instructions.len());
        let (root, regions) = emit_rust_split_with(
            &translation,
            16,
            EmitOptions {
                functions: true,
                function_table,
                call_summaries: true,
                transitive_summaries: true,
                regalloc,
                ..options()
            },
        );
        let propagating = regions
            .iter()
            .map(|(_, source)| source.matches("{ return Ok(()); }").count())
            .sum::<usize>();
        let summarized = regions
            .iter()
            .map(|(_, source)| source.matches("if m.cpu.pc != 0x").count())
            .sum::<usize>();
        writeln!(out, "pub mod {name} {{").expect("write");
        writeln!(out, "pub const BASE: u64 = {BASE:#x};").expect("write");
        writeln!(out, "pub const WORDS: &[u32] = &{words:#x?};").expect("write");
        writeln!(out, "/// Summarized calls, and those that return early with their callee.\npub const SUMMARIZED_CALLS: usize = {summarized};\npub const PROPAGATING: usize = {propagating};").expect("write");
        for (region, source) in regions {
            writeln!(out, "pub mod gen_r_{region:x} {{\n{source}\n}}").expect("write");
        }
        writeln!(out, "{root}\n}}").expect("write");
    }
}

/// An SVC wrapper, as SDKs write them, for `tests/early_exits.rs`: a caller
/// loop (X3 counts to X9) calls a function that is `SVC #0x35` and a RET,
/// where the RET is a resume point (after the SVC) and so not in the
/// function's body; the host completes the SVC and resumes at the RET,
/// which needs the caller's X30 in the `Core`. Lifted with functions and
/// call summaries, directly (`svc_wrapper_code`) and through the table
/// (`svc_wrapper_table_code`).
fn svc_wrapper_code(out: &mut String) {
    const BASE: u64 = 0x1d0_0000;
    let wrapper = BASE + 0x400;
    let bl = |from: u64, to: u64| {
        0x9400_0000 | (((to.wrapping_sub(from) as i64 / 4) as u32) & 0x03ff_ffff)
    };
    let mut words = vec![0u32; 0x800 / 4];
    let program = [
        0xd280_0003,            // movz x3, #0
        0x9100_0463,            // L: add x3, x3, #1
        0x9100_06d6,            // add x22, x22, #1 (callee-saved, in the call's block)
        bl(BASE + 12, wrapper), // bl wrapper
        0x9100_0442,            // add x2, x2, #1
        0xeb09_007f,            // cmp x3, x9
        0x54ff_ff6b,            // b.lt L
        0xd400_00e1,            // svc #7
    ];
    words[..program.len()].copy_from_slice(&program);
    words[0x100..0x102].copy_from_slice(&[0xd400_06a1, 0xd65f_03c0]); // svc #0x35; ret
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let region = CodeRegion {
        name: "svc wrapper".into(),
        base: BASE,
        bytes,
    };
    let translation = discover(&[region], &[BASE], words.len()).expect("svc wrapper");
    for (name, function_table, regalloc) in [
        ("svc_wrapper_code", false, false),
        ("svc_wrapper_table_code", true, false),
        ("svc_wrapper_regalloc_code", true, true),
    ] {
        LIFTED.set(LIFTED.get() + translation.instructions.len());
        let (root, regions) = emit_rust_split_with(
            &translation,
            16,
            EmitOptions {
                functions: true,
                function_table,
                call_summaries: true,
                regalloc,
                ..options()
            },
        );
        let summarized = regions
            .iter()
            .map(|(_, source)| source.matches("if m.cpu.pc != 0x").count())
            .sum::<usize>();
        writeln!(out, "pub mod {name} {{").expect("write");
        writeln!(out, "pub const BASE: u64 = {BASE:#x};").expect("write");
        writeln!(out, "pub const WORDS: &[u32] = &{words:#x?};").expect("write");
        writeln!(out, "/// Calls of leaves with an effect summary.\npub const SUMMARIZED_CALLS: usize = {summarized};").expect("write");
        for (region, source) in regions {
            writeln!(out, "pub mod gen_r_{region:x} {{\n{source}\n}}").expect("write");
        }
        writeln!(out, "{root}\n}}").expect("write");
    }
}

/// Programs for `tests/window_calls.rs`, shaped as game code calls: 40
/// functions packed several to a 1 KiB window between NOP padding, and 60
/// callers, each a run of random instructions with BL or BLR X26 to a
/// function in about half its slots, ending in SVC #7. Bodies mix integer
/// instructions with scalar and vector float and SIMD ones (moves between
/// general and vector registers among them). A function is a leaf (some
/// skipping instructions behind a CBZ, some returning by BR X30 or RETAA),
/// keeps a frame (X29/X30 and X19/X20 on the stack) and calls later
/// functions by BL or BLR, tail-calls a later function by B, runs a
/// two-way jump through ADR and BR, can reach a UDF (a fault), makes an
/// SVC #3, returns elsewhere (X30 pointed at a block in another window,
/// which branches back through X25), recurses X24 times, or calls or
/// tail-calls a PLT stub whose GOT word (`GOT_SLOT`, in the data) holds a
/// function's address. Fixed cases: a caller whose callee's callee returns
/// elsewhere, to code reading a register the caller held across both
/// calls; a function straddling a window boundary (falling through into
/// the next window); and a function jumping into the middle of a block (a
/// MissingBlock trap). Lifted in pairs, each without and with window calls:
/// the build's options (in one 64 KiB region; in 1 KiB regions, every call
/// across windows through the window-call table; in every other 1 KiB
/// region; with early exits; audited), and those options with deferred
/// charges, the step count in a local, dense dispatch, direct calls and
/// stub windows.
fn window_call_code(out: &mut String) {
    window_call_lifts(out, true);
    // The same shapes with integer bodies only, which the reference
    // interpreter covers: without and with window calls, in one region
    // and through the table.
    writeln!(out, "pub mod window_call_int {{").expect("write");
    window_call_lifts(out, false);
    writeln!(out, "}}").expect("write");
}

/// `window_call_code`'s program and lifts; without `fp`, integer bodies
/// only and the base, one-region and table lifts.
fn window_call_lifts(out: &mut String, fp: bool) {
    use echora_lift::aarch64::{Instruction, decode};
    use echora_lift::defuse::effects;

    const BASE: u64 = 0xb0_0000;
    const FUNCTIONS: usize = 40;
    const CALLERS: usize = 60;
    const NOP: u32 = 0xd503_201f;
    const RET: u32 = 0xd65f_03c0;
    /// The PLT stub's GOT word: in the data, below what X28-based
    /// accesses reach.
    const GOT_SLOT: u64 = 0x80_0100;
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 16) as u32
    };
    // X24 counts recursion, X25 holds an early exit's return address, X26
    // BLR targets and jump targets, X27/X28 the data index and base, X30
    // the return address; SP is the stack. (X16/X17 are free: the PLT
    // stub writes them.)
    let reserved = |instruction: &Instruction| {
        let writes = effects(instruction).writes;
        [24, 25, 26, 27, 28, 30].iter().any(|&n| writes.has_gpr(n)) || writes.sp
    };
    let offset = |from: usize, to: usize| (to as i64 - from as i64) as u32;
    let b = |from: usize, to: usize| 0x1400_0000 | (offset(from, to) & 0x03ff_ffff);
    let cbz =
        |from: usize, to: usize, rt: u32| 0xb400_0000 | ((offset(from, to) & 0x7_ffff) << 5) | rt;
    let tbz = |from: usize, to: usize, rt: u32, bit: u32| {
        ((bit >> 5) << 31)
            | 0x3600_0000
            | ((bit & 31) << 19)
            | ((offset(from, to) & 0x3fff) << 5)
            | rt
    };
    let adr = |from: u64, to: u64, rd: u32| {
        let delta = to.wrapping_sub(from) as u32;
        0x1000_0000 | ((delta & 3) << 29) | (((delta >> 2) & 0x7_ffff) << 5) | rd
    };
    // Float and SIMD forms with random registers: (template, fields to
    // fill: Rd, Rn, Rm at bits 0, 5, 16, and a condition at 12).
    const FP_FORMS: &[(u32, bool)] = &[
        (0x1e20_2800, true),  // fadd s
        (0x1e60_0800, true),  // fmul d
        (0x4e20_cc00, true),  // fmla v.4s
        (0x4ea0_8400, true),  // add v.4s
        (0x9e66_0000, false), // fmov x, d
        (0x9e67_0000, false), // fmov d, x
        (0x9e62_0000, false), // scvtf d, x
        (0x9e78_0000, false), // fcvtzs x, d
        (0x1e60_2000, true),  // fcmp d, d (Rd field 0)
        (0x1e60_0c00, true),  // fcsel d (condition)
        (0x0e0c_3c00, false), // umov w, v.s[1]
        (0x4e04_0c00, false), // dup v.4s, w
    ];
    let fp_instruction = |next: &mut dyn FnMut() -> u32| loop {
        let (template, rm) = FP_FORMS[next() as usize % FP_FORMS.len()];
        let mut word = template | (next() % 32) | ((next() % 32) << 5);
        if rm {
            word |= (next() % 32) << 16;
        }
        if template == 0x1e60_2000 {
            word &= !31;
        }
        if template == 0x1e60_0c00 {
            word |= (next() % 15) << 12;
        }
        let instruction = decode(word, 0);
        assert!(
            !matches!(
                instruction,
                Instruction::Unsupported | Instruction::Undefined
            ),
            "{word:#x}"
        );
        if !reserved(&instruction) {
            return word;
        }
    };

    /// A word to patch once every piece has its address.
    enum Fix {
        /// BL to a piece.
        Bl(usize),
        /// B to a piece.
        B(usize),
        /// MOVZ/MOVK X26 with a function's address (the BLR follows).
        Blr(usize),
        /// ADR X30 to a piece (an early exit's other return).
        Adr30(usize),
        /// ADRP X16 to the GOT slot's page.
        Adrp16,
    }
    // Pieces: functions 0..FUNCTIONS, then the early exits' other returns,
    // then the callers, then the fixed cases. Each is its words (relative
    // branches inside it already encoded, in words), fixes, and extra
    // discovery entries (word indices).
    struct Piece {
        words: Vec<u32>,
        fixes: Vec<(usize, Fix)>,
        entries: Vec<usize>,
    }
    let piece = |words: Vec<u32>, fixes: Vec<(usize, Fix)>| Piece {
        words,
        fixes,
        entries: Vec::new(),
    };
    let mut pieces: Vec<Piece> = Vec::new();
    let mut others: Vec<Piece> = Vec::new();
    let body = |words: &mut Vec<u32>, count: u32, next: &mut dyn FnMut() -> u32| {
        for _ in 0..count {
            if fp && next().is_multiple_of(4) {
                words.push(fp_instruction(next));
            } else {
                words.push(body_instruction(next, &reserved));
            }
        }
    };
    let mut stub_uses: Vec<(usize, usize, bool)> = Vec::new(); // (piece, word, link)
    for function in 0..FUNCTIONS {
        let mut piece = Piece {
            words: Vec::new(),
            fixes: Vec::new(),
            entries: Vec::new(),
        };
        let last = function + 1 == FUNCTIONS;
        let kind = if last { 0 } else { next() % 16 };
        let words = &mut piece.words;
        match kind {
            // A frame and calls of later functions.
            4..=7 => {
                words.push(0xa9bf_7bfd); // stp x29, x30, [sp, #-16]!
                let saves = next() % 2 == 0;
                if saves {
                    words.push(0xa9bf_53f3); // stp x19, x20, [sp, #-16]!
                }
                for _ in 0..1 + next() % 3 {
                    body(words, next() % 3, &mut next);
                    let callee = function + 1 + next() as usize % (FUNCTIONS - function - 1);
                    if next() % 5 == 0 {
                        piece.fixes.push((words.len(), Fix::Blr(callee)));
                        words.extend([0, 0, 0xd63f_0000 | (26 << 5)]);
                    } else {
                        piece.fixes.push((words.len(), Fix::Bl(callee)));
                        words.push(0);
                    }
                }
                body(words, next() % 3, &mut next);
                if saves {
                    words.push(0xa8c1_53f3); // ldp x19, x20, [sp], #16
                }
                words.push(0xa8c1_7bfd); // ldp x29, x30, [sp], #16
                words.push(RET);
            }
            // A two-way jump: ADR X26, case 0; TBZ over an ADD of 8 to it;
            // BR X26; two cases of an instruction and a branch to the end.
            8 => {
                body(words, next() % 3, &mut next);
                let at = words.len();
                let register = next() % 24;
                let bit = next() % 64;
                // adr x26, case0 (at + 4); tbz; add x26, x26, #8; br x26
                words.push(adr(0, 4 * 4, 26));
                words.push(tbz(1, 3, register, bit));
                words.push(0x9100_235a); // add x26, x26, #8
                words.push(0xd61f_0000 | (26 << 5));
                let case0 = at + 4;
                for case in 0..2 {
                    body(words, 1, &mut next);
                    words.push(0);
                    piece.entries.push(case0 + 2 * case);
                }
                let end = words.len();
                words[case0 + 1] = b(case0 + 1, end);
                words[case0 + 3] = b(case0 + 3, end);
                body(words, 1 + next() % 2, &mut next);
                words.push(RET);
            }
            // A UDF behind a bit test: a fault on some states.
            9 => {
                body(words, next() % 3, &mut next);
                let at = words.len();
                words.push(tbz(at, at + 2, next() % 24, next() % 64));
                words.push(0); // udf #0
                body(words, 1 + next() % 3, &mut next);
                words.push(RET);
            }
            // An SVC #3 the host completes, in a leaf.
            10 => {
                body(words, 1 + next() % 2, &mut next);
                words.push(0xd400_0061);
                body(words, next() % 3, &mut next);
                words.push(RET);
            }
            // Returns elsewhere (X30 pointed at a block in another window,
            // which goes back through X25), or recurses X24 times.
            11 => {
                if next() % 2 == 0 {
                    words.push(0xaa1e_03f9); // mov x25, x30
                    body(words, 1 + next() % 3, &mut next);
                    piece
                        .fixes
                        .push((words.len(), Fix::Adr30(FUNCTIONS + others.len())));
                    words.push(0);
                    words.push(RET);
                    let mut other = Piece {
                        words: Vec::new(),
                        fixes: Vec::new(),
                        entries: Vec::new(),
                    };
                    body(&mut other.words, 1 + next() % 3, &mut next);
                    other.words.push(0xd61f_0000 | (25 << 5)); // br x25
                    others.push(other);
                } else {
                    // cbz x24, out; sub x24, x24, #1; stp; bl self; ldp; out: ret
                    words.push(0);
                    words.push(0xd100_0718);
                    words.push(0xa9bf_7bfd);
                    piece.fixes.push((words.len(), Fix::Bl(function)));
                    words.push(0);
                    words.push(0xa8c1_7bfd);
                    body(words, next() % 2, &mut next);
                    let out = words.len();
                    words[0] = cbz(0, out, 24);
                    words.push(RET);
                }
            }
            // A tail call of a later function.
            12 => {
                body(words, 1 + next() % 3, &mut next);
                let callee = function + 1 + next() as usize % (FUNCTIONS - function - 1);
                piece.fixes.push((words.len(), Fix::B(callee)));
                words.push(0);
            }
            // A leaf returning by BR X30, or by RETAA after PACIASP.
            13 => {
                let authenticated = next() % 2 == 0;
                if authenticated {
                    words.push(0xd503_233f); // paciasp
                }
                body(words, 1 + next() % 3, &mut next);
                words.push(if authenticated {
                    0xd65f_0bff
                } else {
                    0xd61f_03c0
                });
            }
            // A call or a tail call of the PLT stub.
            14 => {
                body(words, next() % 3, &mut next);
                if next() % 2 == 0 {
                    words.push(0xa9bf_7bfd);
                    stub_uses.push((function, words.len(), true));
                    words.push(0);
                    words.push(0xa8c1_7bfd);
                    body(words, next() % 2, &mut next);
                    words.push(RET);
                } else {
                    stub_uses.push((function, words.len(), false));
                    words.push(0);
                }
            }
            // A leaf, some with instructions skipped behind a CBZ.
            _ => {
                body(words, next() % 3, &mut next);
                if next() % 2 == 0 {
                    let at = words.len();
                    let skipped = 1 + next() as usize % 2;
                    words.push(cbz(at, at + 1 + skipped, next() % 24));
                    body(words, skipped as u32, &mut next);
                }
                body(words, 1 + next() % 3, &mut next);
                words.push(RET);
            }
        }
        pieces.push(piece);
    }
    pieces.extend(others);
    let callers_from = pieces.len();
    for _ in 0..CALLERS {
        let mut piece = Piece {
            words: Vec::new(),
            fixes: Vec::new(),
            entries: Vec::new(),
        };
        for _ in 0..4 + next() % 9 {
            if next() % 2 == 0 {
                let callee = next() as usize % FUNCTIONS;
                if next() % 6 == 0 {
                    piece.fixes.push((piece.words.len(), Fix::Blr(callee)));
                    piece.words.extend([0, 0, 0xd63f_0000 | (26 << 5)]);
                } else {
                    piece.fixes.push((piece.words.len(), Fix::Bl(callee)));
                    piece.words.push(0);
                }
            } else {
                let mut words = Vec::new();
                body(&mut words, 1, &mut next);
                piece.words.extend(words);
            }
        }
        piece.words.push(0xd400_00e1);
        pieces.push(piece);
    }
    // The PLT stub: ADRP X16, the slot's page; LDR X17, [X16, #0x100];
    // ADD X16, X16, #0x100; BR X17. Its GOT word holds function 0's
    // address (`run` writes it).
    let stub = pieces.len();
    pieces.push(piece(
        vec![0, 0xf940_8211, 0x9104_0210, 0xd61f_0220],
        vec![(0, Fix::Adrp16)],
    ));
    for (function, at, link) in stub_uses {
        pieces[function]
            .fixes
            .push((at, if link { Fix::Bl(stub) } else { Fix::B(stub) }));
    }
    // Two levels with an early return: a caller sets X9 and calls F, which
    // keeps a frame and calls E, which returns to a block in another
    // window reading X9 (held by the caller across both calls: neither
    // touches it) before going back to F through X25.
    let other = pieces.len();
    let early = other + 1;
    let frame = other + 2;
    let chain_caller = other + 3;
    // add x10, x9, #1; br x25
    pieces.push(piece(
        vec![0x9100_052a, 0xd61f_0000 | (25 << 5)],
        Vec::new(),
    ));
    // mov x25, x30; adr x30, other; ret
    pieces.push(piece(
        vec![0xaa1e_03f9, 0, RET],
        vec![(1, Fix::Adr30(other))],
    ));
    // stp x29, x30, [sp, #-16]!; bl early; ldp x29, x30, [sp], #16; ret
    pieces.push(piece(
        vec![0xa9bf_7bfd, 0, 0xa8c1_7bfd, RET],
        vec![(1, Fix::Bl(early))],
    ));
    // movz x9, #0x1234; bl frame; svc #7
    pieces.push(piece(
        vec![0xd282_4689, 0, 0xd400_00e1],
        vec![(1, Fix::Bl(frame))],
    ));
    // A jump into the middle of a block: adr x26, .; add x26, x26, #16;
    // br x26 (to the second ADD below, which starts no block).
    let missing = pieces.len();
    pieces.push(piece(
        vec![
            adr(0, 0, 26),
            0x9100_435a,
            0xd61f_0000 | (26 << 5),
            0x9100_0421,
            0x9100_0842,
            RET,
        ],
        Vec::new(),
    ));
    let missing_caller = pieces.len();
    pieces.push(piece(vec![0, 0xd400_00e1], vec![(0, Fix::Bl(missing))]));
    // A switch through a halfword jump table (a leaf): AND W17, W0, #3;
    // ADR X16, table; LDRH W9, [X16, X17, LSL #1]; ADR X3, cases;
    // ADD X3, X3, X9, LSL #2; BR X3; four cases of an ADD to X0 and a B to
    // the RET; the table (offsets in words from the cases).
    let switch = pieces.len();
    {
        let mut words = vec![
            0x1200_0411,
            adr(4, 4 * 15, 16),
            0x7871_7a09,
            adr(12, 4 * 6, 3),
            0x8b09_0863,
            0xd61f_0060,
        ];
        for case in 0..4u32 {
            words.push(0x9100_0000 | ((case * 3 + 1) << 10));
            words.push(b(words.len(), 14));
        }
        words.push(RET);
        words.extend([2 << 16, 4 | (6 << 16)]);
        assert_eq!(words.len(), 17);
        pieces.push(piece(words, Vec::new()));
    }
    // A frame calling the switch (a transitive summary over a jump table).
    let switch_frame = pieces.len();
    pieces.push(piece(
        vec![0xa9bf_7bfd, 0, 0xa8c1_7bfd, RET],
        vec![(1, Fix::Bl(switch))],
    ));
    // A function of more than FUNCTION_BLOCK_LIMIT blocks (and windows):
    // 300 CBZs over an ADD each.
    let long = pieces.len();
    {
        let mut words = Vec::new();
        for _ in 0..300 {
            let at = words.len();
            words.push(cbz(at, at + 2, 1 + next() % 15));
            words.push(0x9100_0421 | ((next() % 15) << 10)); // add x1, x1, #k
        }
        words.push(RET);
        pieces.push(piece(words, Vec::new()));
    }
    // A game-shaped function: a frame saving X19/X20, eight rounds of a
    // body (random instructions, CBZ skips) with calls of the switch and of
    // the switch's frame, then the frame back.
    let game = pieces.len();
    {
        let keep = |instruction: &Instruction| {
            let writes = effects(instruction).writes;
            reserved(instruction) || [19, 20, 29].iter().any(|&n| writes.has_gpr(n))
        };
        let mut words = vec![0xa9be_7bfd, 0xa901_53f3, 0xd280_0113]; // stp; stp x19, x20; mov x19, #8
        let mut fixes = Vec::new();
        let head = words.len();
        for round in 0..6 {
            for _ in 0..4 {
                words.push(body_instruction(&mut next, &keep));
            }
            let at = words.len();
            words.push(cbz(at, at + 3, 1 + next() % 15));
            words.push(body_instruction(&mut next, &keep));
            words.push(body_instruction(&mut next, &keep));
            // (Callees that leave X19, its round count, alone.)
            let callee = if round % 2 == 0 { switch } else { switch_frame };
            fixes.push((words.len(), Fix::Bl(callee)));
            words.push(0);
        }
        words.push(0xd100_0673); // sub x19, x19, #1
        let at = words.len();
        words.push(0xb500_0013 | ((offset(at, head) & 0x7_ffff) << 5)); // cbnz x19, head
        words.extend([0xa941_53f3, 0xa8c2_7bfd, RET]); // ldp x19, x20; ldp x29, x30
        pieces.push(Piece {
            words,
            fixes,
            entries: Vec::new(),
        });
    }
    let new_callers = pieces.len();
    pieces.push(piece(
        vec![0, 0, 0xd400_00e1],
        vec![(0, Fix::Bl(switch)), (1, Fix::Bl(switch_frame))],
    ));
    pieces.push(piece(vec![0, 0xd400_00e1], vec![(0, Fix::Bl(long))]));
    pieces.push(piece(vec![0, 0xd400_00e1], vec![(0, Fix::Bl(game))]));
    // A function straddling a window boundary (placed last), and its
    // caller.
    let straddle = pieces.len();
    {
        let mut words = Vec::new();
        for _ in 0..6 {
            words.push(body_instruction(&mut next, &reserved));
        }
        words.push(RET);
        pieces.push(piece(words, Vec::new()));
    }
    let straddle_caller = pieces.len();
    pieces.push(piece(
        vec![0, 0, 0xd400_00e1],
        vec![(0, Fix::Bl(straddle)), (1, Fix::Bl(0))],
    ));
    // Layout: the pieces shuffled, with NOP padding of 0 to 63 words
    // between them; the straddling function last, three words before a
    // window boundary.
    let mut order: Vec<usize> = (0..straddle).chain(straddle_caller..pieces.len()).collect();
    for index in (1..order.len()).rev() {
        order.swap(index, next() as usize % (index + 1));
    }
    let mut addresses = vec![0u64; pieces.len()];
    let mut words: Vec<u32> = Vec::new();
    for &piece in &order {
        let padding = if next() % 3 == 0 { next() % 64 } else { 0 };
        words.extend(std::iter::repeat_n(NOP, padding as usize));
        addresses[piece] = BASE + 4 * words.len() as u64;
        words.extend(&pieces[piece].words);
    }
    while !(words.len() + 3).is_multiple_of(256) {
        words.push(NOP);
    }
    addresses[straddle] = BASE + 4 * words.len() as u64;
    words.extend(&pieces[straddle].words);
    let mut entries: Vec<u64> = Vec::new();
    for (index, piece) in pieces.iter().enumerate() {
        let start = ((addresses[index] - BASE) / 4) as usize;
        entries.push(addresses[index]);
        for &at in &piece.entries {
            entries.push(addresses[index] + 4 * at as u64);
        }
        for (at, fix) in &piece.fixes {
            let pc = addresses[index] + 4 * *at as u64;
            let word = start + at;
            let relative = |to: u64| ((to.wrapping_sub(pc) as i64 / 4) as u32) & 0x03ff_ffff;
            match *fix {
                Fix::Bl(callee) => words[word] = 0x9400_0000 | relative(addresses[callee]),
                Fix::B(callee) => words[word] = 0x1400_0000 | relative(addresses[callee]),
                Fix::Blr(callee) => {
                    let target = addresses[callee];
                    words[word] = 0xd280_0000 | (((target & 0xffff) as u32) << 5) | 26;
                    words[word + 1] = 0xf2a0_0000 | (((target >> 16) as u32) << 5) | 26;
                }
                Fix::Adr30(other) => words[word] = adr(pc, addresses[other], 30),
                Fix::Adrp16 => {
                    let pages = ((GOT_SLOT >> 12) as i64 - (pc >> 12) as i64) as u32;
                    words[word] =
                        0x9000_0000 | ((pages & 3) << 29) | (((pages >> 2) & 0x7_ffff) << 5) | 16;
                }
            }
        }
    }
    let missing_target = addresses[missing] + 16;
    let mut callers: Vec<u64> = addresses[callers_from..callers_from + CALLERS].to_vec();
    callers.extend([
        addresses[chain_caller],
        addresses[missing_caller],
        addresses[straddle_caller],
        addresses[new_callers],
        addresses[new_callers + 1],
        addresses[new_callers + 2],
    ]);
    let image: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let region = CodeRegion {
        name: "window calls".into(),
        base: BASE,
        bytes: image,
    };
    let translation = discover(&[region], &entries, words.len() + 64).expect("window calls");
    let report = translation.report();
    // (Indirect branches' targets count as unresolved: they are entries.)
    assert!(
        report.unsupported.is_empty(),
        "window-call programs must lift completely: {:?}",
        report.unsupported
    );
    assert!(
        !translation
            .blocks
            .iter()
            .any(|block| block.start == missing_target),
        "the missing target starts no block"
    );
    assert!(
        echora_lift::codegen::plt_stubs(&translation).contains_key(&addresses[stub]),
        "the PLT stub is recognised"
    );
    // The base lifts never have window calls, whatever the features.
    let plain = EmitOptions {
        window_calls: false,
        window_call_early_exits: false,
        window_calls_audit: false,
        // Lifted functions only in the function lifts.
        functions: false,
        function_table: false,
        call_summaries: false,
        transitive_summaries: false,
        call_summaries_audit: false,
        ..options()
    };
    let calling = EmitOptions {
        window_calls: true,
        ..plain
    };
    let active = !calling.structured && !calling.functions;
    // (name, root, regions)
    type Lift = (&'static str, String, Vec<(u64, String)>);
    let mut lifts: Vec<Lift> = Vec::new();
    let mut lift = |name: &'static str, bits: u32, options: EmitOptions| {
        LIFTED.set(LIFTED.get() + translation.instructions.len());
        let (root, regions) = emit_rust_split_with(&translation, bits, options);
        lifts.push((name, root, regions));
    };
    let with = |change: fn(&mut EmitOptions)| {
        let mut options = plain;
        change(&mut options);
        options
    };
    lift("window_call_base", 16, plain);
    lift("window_call_on", 16, calling);
    lift("window_call_table", 10, calling);
    // (the lift without, the lift with, the options' change)
    type Variant = (&'static str, &'static str, fn(&mut EmitOptions));
    let variants: &[Variant] = if fp {
        &[
            (
                "window_call_deferred_base",
                "window_call_deferred_on",
                |o| {
                    o.split_ticks = true;
                    o.deferred_charges = true;
                },
            ),
            ("window_call_steps_base", "window_call_steps_on", |o| {
                o.split_ticks = true;
                o.deferred_charges = false;
                o.steps_local = true;
            }),
            ("window_call_dense_base", "window_call_dense_on", |o| {
                o.dense_dispatch = true
            }),
            ("window_call_direct_base", "window_call_direct_on", |o| {
                o.direct_calls = true
            }),
            ("window_call_stub_base", "window_call_stub_on", |o| {
                o.stub_windows = true
            }),
        ]
    } else {
        &[]
    };
    if fp {
        lift(
            "window_call_early",
            16,
            EmitOptions {
                window_call_early_exits: true,
                ..calling
            },
        );
        lift(
            "window_call_audit",
            16,
            EmitOptions {
                window_call_early_exits: true,
                window_calls_audit: true,
                ..calling
            },
        );
    }
    // Other options, each without and with window calls.
    for &(base, on, change) in variants {
        let options = with(change);
        lift(base, 16, options);
        lift(
            on,
            16,
            EmitOptions {
                window_calls: true,
                ..options
            },
        );
    }
    // Lifted functions with transitive summaries through the function
    // table: in one region, in 1 KiB regions (functions spanning
    // regions), and with float bodies also audited, called directly, with
    // deferred charges and with the step count in a local.
    let functions = EmitOptions {
        functions: true,
        function_table: true,
        call_summaries: true,
        transitive_summaries: true,
        ..plain
    };
    lift("function_on", 16, functions);
    lift("function_regions", 10, functions);
    // Structured functions (regalloc): in one region, in 1 KiB regions,
    // and called without the table.
    let structured_functions = EmitOptions {
        regalloc: true,
        ..functions
    };
    lift("regalloc_on", 16, structured_functions);
    lift("regalloc_regions", 10, structured_functions);
    if fp {
        lift(
            "regalloc_direct",
            16,
            EmitOptions {
                function_table: false,
                ..structured_functions
            },
        );
    }
    if fp {
        lift(
            "function_audit",
            16,
            EmitOptions {
                call_summaries_audit: true,
                ..functions
            },
        );
        lift(
            "function_direct",
            16,
            EmitOptions {
                function_table: false,
                ..functions
            },
        );
        lift(
            "function_deferred",
            16,
            EmitOptions {
                split_ticks: true,
                deferred_charges: true,
                ..functions
            },
        );
        lift(
            "function_steps",
            16,
            EmitOptions {
                split_ticks: true,
                deferred_charges: false,
                steps_local: true,
                ..functions
            },
        );
    }
    if fp {
        let listed: std::collections::BTreeSet<u64> = translation
            .blocks
            .iter()
            .map(|block| block.start >> 10)
            .filter(|region| region % 2 == 0)
            .collect();
        LIFTED.set(LIFTED.get() + translation.instructions.len());
        let mut regions = Vec::new();
        let root = echora_lift::codegen::emit_rust_split_to_regions(
            &translation,
            10,
            plain,
            calling,
            &listed,
            &mut |region, source, _| {
                regions.push((region, source));
                Ok::<(), std::convert::Infallible>(())
            },
        )
        .expect("no sink errors");
        lifts.push(("window_call_mixed", root, regions));
    }
    // Targets with summaries, early exits allowed, for the tests.
    let targets: Vec<u64> =
        echora_lift::codegen::window_call_summaries(&translation, &|_| Some(true))
            .into_keys()
            .collect();
    writeln!(out, "pub mod window_call_program {{").expect("write");
    writeln!(out, "pub const BASE: u64 = {BASE:#x};").expect("write");
    writeln!(out, "pub const WORDS: &[u32] = &{words:#x?};").expect("write");
    writeln!(out, "pub const CALLERS: &[u64] = &{callers:#x?};").expect("write");
    writeln!(
        out,
        "/// The PLT stub's GOT word and what it holds.\npub const GOT_SLOT: u64 = {GOT_SLOT:#x};\npub const GOT_WORD: u64 = {:#x};\n\
         /// Where the jump into a block's middle lands.\npub const MISSING: u64 = {missing_target:#x};\n\
         /// Targets with window-call summaries (early exits allowed).\npub const TARGETS: &[u64] = &{targets:#x?};",
        addresses[0]
    )
    .expect("write");
    writeln!(out, "/// Whether the build's options allow window calls (not structured, no lifted functions).\npub const ACTIVE: bool = {active};").expect("write");
    let lifted_16 = echora_lift::codegen::functions(&translation, 16);
    let lifted_10 = echora_lift::codegen::functions(&translation, 10);
    let summaries_16: Vec<u64> =
        echora_lift::codegen::transitive_summaries(&translation, &lifted_16, 16, true)
            .into_keys()
            .collect();
    writeln!(
        out,
        "/// The switch, its frame, the function over the block limit and the game-shaped function.\n\
         pub const SWITCH: u64 = {:#x};\npub const SWITCH_FRAME: u64 = {:#x};\npub const LONG: u64 = {:#x};\npub const GAME: u64 = {:#x};\n\
         pub const JUMP_TABLES: usize = {};\n\
         /// Lifted functions in 64 KiB and 1 KiB regions, and those with summaries (64 KiB, the table).\n\
         pub const FUNCTIONS_16: &[u64] = &{:#x?};\npub const FUNCTIONS_10: &[u64] = &{:#x?};\npub const SUMMARIES_16: &[u64] = &{:#x?};",
        addresses[switch],
        addresses[switch_frame],
        addresses[long],
        addresses[game],
        report.jump_tables,
        lifted_16.keys().collect::<Vec<_>>(),
        lifted_10.keys().collect::<Vec<_>>(),
        summaries_16,
    )
    .expect("write");
    writeln!(out, "}}").expect("write");
    for (name, mut root, regions) in lifts {
        // Every lift answers whether its window-call table is installed
        // (those without window calls: never).
        if !root.contains("fn window_calls_installed") {
            root.push_str("\npub fn window_calls_installed() -> bool { false }\n");
        }
        let count = |pattern: &str| {
            regions
                .iter()
                .map(|(_, source)| source.matches(pattern).count())
                .sum::<usize>()
        };
        // The census over every window (`echora_lift::census`): windows,
        // stores and reloads of register locals in their bodies, calls.
        let (mut windows, mut spills, mut reloads, mut calls) = (0, 0, 0, 0);
        for (_, source) in &regions {
            for (_, counts) in echora_lift::census::census(source) {
                windows += 1;
                spills += counts.spills;
                reloads += counts.reloads;
                calls += counts.calls;
            }
        }
        writeln!(out, "pub mod {name} {{").expect("write");
        writeln!(
            out,
            "/// The census of its windows: windows, spills, reloads and calls.\n\
             pub const CENSUS: (usize, usize, usize, usize) = ({windows}, {spills}, {reloads}, {calls});"
        )
        .expect("write");
        writeln!(
            out,
            "/// Summarized window calls, directly and through the table, and the windows they enter.\n\
             pub const DIRECT_CALLS: usize = {};\npub const TABLE_CALLS: usize = {};\npub const ENTERED_WINDOWS: usize = {};",
            count("= enter_window_"),
            count("= window_call_at("),
            count("pub fn enter_window_")
        )
        .expect("write");
        writeln!(
            out,
            "/// Lifted functions, and summarized calls of them (directly and through the table).
             pub const FUNCTIONS: usize = {};
pub const FUNCTION_CALLS: usize = {};",
            count("pub fn run_function_"),
            count("let called = run_function_") + count("let called = function_at(")
        )
        .expect("write");
        writeln!(
            out,
            "/// Functions emitted structured (regalloc), and window arms running them.\n\
             pub const STRUCTURED_FUNCTIONS: usize = {};\npub const STRUCTURED_ENTRIES: usize = {};",
            count("n: &mut u64) -> Result<(), Trap>"),
            count("= run_entry(m, run_function_")
        )
        .expect("write");
        for (region, source) in regions {
            writeln!(out, "pub mod gen_r_{region:x} {{\n{source}\n}}").expect("write");
        }
        writeln!(out, "{root}\n}}").expect("write");
    }
}
