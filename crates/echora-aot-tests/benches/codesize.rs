//! Size and compile time of the generated Rust, per emission mode.
//!
//! `build.rs` (with the `codesize` feature) writes every program it lifts
//! through split emission once per mode (flat, structured, inline_vector,
//! fused_flags, whole_vector, fp_locals and the recommended set) as a crate
//! of its own. This compiles each with rustc as cargo would for a region of
//! the game build (an rlib, `opt-level` 1, 16 codegen units) and reports
//! the bytes of Rust per guest instruction, the time, the machine code
//! (`.text`, by `size`) per guest instruction and rustc's peak memory
//! (`VmHWM`, sampled while it runs). Also the game set and the game set plus
//! each option the readiness report (`docs/lift-options.md`) recommends,
//! and the benchmark kernels' generated Rust per option.
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

/// rustc's peak resident memory so far (`VmHWM` in `/proc/<pid>/status`),
/// in KiB, where the system has it.
fn peak_kib(pid: u32) -> Option<u64> {
    let status = std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
    line.split_whitespace().nth(1)?.parse().ok()
}

/// Compiles `source`; the time and rustc's peak memory in KiB (0 where
/// `/proc` does not have it), sampled every 10 ms while it runs.
fn compile(source: &Path, deps: &Path, guest: &Path, opt: u32, scratch: &Path) -> (Duration, u64) {
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let start = Instant::now();
    let mut child = Command::new(rustc)
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
        .spawn()
        .expect("running rustc");
    let mut peak = 0;
    let status = loop {
        if let Some(status) = child.try_wait().expect("waiting for rustc") {
            break status;
        }
        peak = peak.max(peak_kib(child.id()).unwrap_or(0));
        std::thread::sleep(Duration::from_millis(10));
    };
    let elapsed = start.elapsed();
    assert!(status.success(), "rustc failed on {}", source.display());
    (elapsed, peak)
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
        "{:<24} {:>12} {:>12} {:>10} {:>12} {:>14} {:>12} {:>10}",
        "mode",
        "instructions",
        "bytes",
        "bytes/ins",
        "rustc s",
        "us/instruction",
        "text B/ins",
        "rustc MiB"
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
        let compiles: Vec<(Duration, u64)> = (0..runs)
            .map(|_| compile(&source, &deps, &guest, opt, &scratch))
            .collect();
        let best = compiles
            .iter()
            .map(|&(time, _)| time)
            .min()
            .expect("at least one run");
        let peak = compiles.iter().map(|&(_, peak)| peak).max().unwrap_or(0);
        let text = text_bytes(&scratch).map_or_else(
            || "-".to_owned(),
            |text| format!("{:.1}", text as f64 / instructions),
        );
        println!(
            "{name:<24} {instructions:>12} {bytes:>12} {:>10.1} {:>12.2} {:>14.2} {text:>12} {:>10.0}",
            bytes / instructions,
            best.as_secs_f64(),
            best.as_secs_f64() * 1e6 / instructions,
            peak as f64 / 1024.0
        );
    }
    let _ = std::fs::remove_dir_all(&scratch);
    if let Ok(text) = std::fs::read_to_string(Path::new(SOURCES).join("liveness.txt")) {
        print!("dead writes (echora_lift::liveness): {text}");
    }
    // The benchmark kernels' generated Rust per option (`build.rs`,
    // `kernel_sizes`): the game set, then the game set plus each option.
    if let Ok(text) = std::fs::read_to_string(Path::new(SOURCES).join("kernels.txt")) {
        let sizes: Vec<(&str, f64, f64)> = text
            .lines()
            .filter_map(|line| {
                let mut fields = line.split_whitespace();
                Some((
                    fields.next()?,
                    fields.next()?.parse().ok()?,
                    fields.next()?.parse().ok()?,
                ))
            })
            .collect();
        if let Some(&(_, base_bytes, base_lines)) = sizes.first() {
            println!("\nkernels' generated Rust, against the game set:");
            println!(
                "{:<24} {:>10} {:>8} {:>8} {:>8}",
                "option", "bytes", "change", "lines", "change"
            );
            for (name, bytes, lines) in sizes {
                println!(
                    "{name:<24} {bytes:>10} {:>+7.1}% {lines:>8} {:>+7.1}%",
                    (bytes / base_bytes - 1.0) * 100.0,
                    (lines / base_lines - 1.0) * 100.0
                );
            }
        }
    }
}
