//! Size and compile time of the generated Rust, per emission mode.
//!
//! `build.rs` (with the `codesize` feature) writes every program it lifts
//! through split emission once per mode (flat, structured, inline_vector,
//! fused_flags, whole_vector, fp_locals and the recommended set) as a crate
//! of its own. This compiles each with rustc as cargo would for a region of
//! the game build (an rlib, `opt-level` 1, 16 codegen units) and reports
//! the bytes of Rust per guest instruction, the time, and the machine code
//! (`.text`, by `size`) per guest instruction.
//! See `docs/testing.md`.
//!
//! ```text
//! cargo bench -p echora-aot-tests --features codesize --bench codesize
//!     [-- [MODE] [--opt N] [--runs N]]
//! ```
//!
//! `MODE` keeps modes whose name contains it; `--opt` sets the optimization
//! level (default 1, as the game build's regions); `--runs` compiles each
//! mode that many times and reports the fastest (default 1).

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

const SOURCES: &str = concat!(env!("OUT_DIR"), "/codesize");

/// The newest `lib{name}-*.rlib` next to this benchmark: the dependency the
/// generated code compiles against.
fn rlib(deps: &Path, name: &str) -> PathBuf {
    let prefix = format!("lib{name}-");
    std::fs::read_dir(deps)
        .expect("the deps directory")
        .flatten()
        .filter(|entry| {
            let file = entry.file_name().to_string_lossy().into_owned();
            file.starts_with(&prefix) && file.ends_with(".rlib")
        })
        .max_by_key(|entry| entry.metadata().and_then(|m| m.modified()).ok())
        .map(|entry| entry.path())
        .unwrap_or_else(|| panic!("no {prefix}*.rlib in {}", deps.display()))
}

/// The `.text` bytes of the rlib `compile` left in `scratch` (the sum over
/// its object files, by binutils' `size`), if `size` runs.
fn text_bytes(scratch: &Path) -> Option<u64> {
    let output = Command::new("size")
        .arg("-t")
        .arg(scratch.join("libcodesize.rlib"))
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&output.stdout);
    let totals = text.lines().find(|line| line.contains("(TOTALS)"))?;
    totals.split_whitespace().next()?.parse().ok()
}

fn compile(source: &Path, deps: &Path, guest: &Path, opt: u32, scratch: &Path) -> Duration {
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let start = Instant::now();
    let status = Command::new(rustc)
        .args([
            "--edition",
            "2024",
            "--crate-type",
            "rlib",
            "--cap-lints",
            "allow",
        ])
        .args(["--crate-name", "codesize"])
        .arg(format!("-Copt-level={opt}"))
        .arg("-Ccodegen-units=16")
        .arg("-L")
        .arg(format!("dependency={}", deps.display()))
        .arg("--extern")
        .arg(format!("echora_guest={}", guest.display()))
        .arg("--out-dir")
        .arg(scratch)
        .arg(source)
        .status()
        .expect("running rustc");
    let elapsed = start.elapsed();
    assert!(status.success(), "rustc failed on {}", source.display());
    elapsed
}

fn main() {
    let mut filter = None;
    let mut opt = 1;
    let mut runs = 1;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut number = |name: &str| {
            args.next()
                .and_then(|value| value.parse().ok())
                .unwrap_or_else(|| panic!("{name} takes a number"))
        };
        match arg.as_str() {
            "--opt" => opt = number("--opt"),
            "--runs" => runs = number("--runs").max(1),
            "--bench" => {}
            other if other.starts_with('-') => panic!("unknown option {other}"),
            other => filter = Some(other.to_owned()),
        }
    }
    let exe = std::env::current_exe().expect("the benchmark's path");
    let deps = exe.parent().expect("the deps directory").to_path_buf();
    let guest = rlib(&deps, "echora_guest");
    let scratch = std::env::temp_dir().join(format!("echora-codesize-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).expect("scratch directory");
    let list = std::fs::read_to_string(Path::new(SOURCES).join("modes.txt")).expect("modes.txt");
    println!(
        "{:<14} {:>12} {:>12} {:>10} {:>12} {:>14} {:>12}",
        "mode", "instructions", "bytes", "bytes/ins", "rustc s", "us/instruction", "text B/ins"
    );
    for line in list.lines() {
        let mut fields = line.split_whitespace();
        let (Some(name), Some(instructions), Some(bytes)) =
            (fields.next(), fields.next(), fields.next())
        else {
            continue;
        };
        if filter
            .as_deref()
            .is_some_and(|filter| !name.contains(filter))
        {
            continue;
        }
        let instructions: f64 = instructions.parse().expect("instruction count");
        let bytes: f64 = bytes.parse().expect("byte count");
        let source = Path::new(SOURCES).join(format!("{name}.rs"));
        let best = (0..runs)
            .map(|_| compile(&source, &deps, &guest, opt, &scratch))
            .min()
            .expect("at least one run");
        let text = text_bytes(&scratch).map_or_else(
            || "-".to_owned(),
            |text| format!("{:.1}", text as f64 / instructions),
        );
        println!(
            "{name:<14} {instructions:>12} {bytes:>12} {:>10.1} {:>12.2} {:>14.2} {text:>12}",
            bytes / instructions,
            best.as_secs_f64(),
            best.as_secs_f64() * 1e6 / instructions
        );
    }
    let _ = std::fs::remove_dir_all(&scratch);
    if let Ok(text) = std::fs::read_to_string(Path::new(SOURCES).join("liveness.txt")) {
        print!("dead writes (echora_lift::liveness): {text}");
    }
}
