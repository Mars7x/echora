//! `echora prepare`: everything between a game and a runnable translation,
//! so the launcher can set the game up with one command. It fetches the
//! shader recompiler sources, lifts the game and builds the runner, and
//! reports its progress on stdout in lines a front end can follow:
//!
//! ```text
//! echora-prepare: step fetch
//! echora-prepare: step lift
//! echora-prepare: step build
//! echora-prepare: built 340 of 1111
//! echora-prepare: ready
//! ```

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};

use anyhow::{Context, Result, bail};

/// The prefix of every progress line.
pub const PREFIX: &str = "echora-prepare: ";

/// Written into a translation once its runner is built; names the `echora`
/// binary that prepared it, so a new build of Echora prepares it again.
const STAMP: &str = "prepared";

/// The repository this binary was built from: the generated workspace
/// depends on its crates, and the fetch script lives there.
fn repository() -> Result<PathBuf> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .context("locate the Echora source tree this binary was built from")
}

/// Identifies the running `echora` binary by size and modification time.
pub fn binary_identity() -> String {
    let meta = std::env::current_exe().and_then(std::fs::metadata);
    match meta {
        Ok(meta) => {
            let modified = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_nanos());
            format!("{} {modified}\n", meta.len())
        }
        Err(_) => String::new(),
    }
}

pub fn runner_path(translation: &Path) -> PathBuf {
    translation
        .join("target/release/echora-run")
        .with_extension(std::env::consts::EXE_EXTENSION)
}

/// Whether the translation's runner exists and no other build of Echora
/// prepared it. A runner built by hand (no stamp) is used as it is, since
/// whoever built it manages it.
pub fn is_ready(translation: &Path) -> bool {
    runner_path(translation).is_file()
        && match std::fs::read_to_string(translation.join(STAMP)) {
            Ok(stamp) => stamp == binary_identity(),
            Err(_) => true,
        }
}

pub fn mark_ready(translation: &Path) -> Result<()> {
    std::fs::write(translation.join(STAMP), binary_identity())
        .context("record the prepared translation")
}

pub fn step(name: &str) {
    println!("{PREFIX}step {name}");
}

/// Fetches the shader recompiler sources into `sources` with the
/// repository's script. Existing sources are kept when the fetch fails
/// (offline, say).
pub fn fetch_shader_sources(sources: &Path) -> Result<()> {
    let script = repository()?.join("scripts/fetch-shader-sources.sh");
    let status = Command::new("bash")
        .arg(&script)
        .env("ECHORA_SHADER_SRC", sources)
        .stdout(Stdio::null())
        .status()
        .with_context(|| format!("run {}", script.display()))?;
    if status.success() {
        return Ok(());
    }
    if sources.join("lichii/src/shader_recompiler").is_dir() {
        eprintln!(
            "warning: fetching the shader recompiler failed; using the sources already there"
        );
        return Ok(());
    }
    bail!("fetching the shader recompiler failed ({status})");
}

/// `cargo`, from `PATH` or rustup's default location (a launcher started
/// from the desktop may not have `~/.cargo/bin` on its `PATH`).
fn cargo() -> PathBuf {
    if let Some(cargo) = std::env::var_os("CARGO") {
        return cargo.into();
    }
    let on_path = Command::new("cargo")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success());
    if !on_path {
        let home = std::env::var_os("CARGO_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cargo")));
        if let Some(cargo) = home.map(|h| h.join("bin/cargo")).filter(|c| c.is_file()) {
            return cargo;
        }
    }
    "cargo".into()
}

/// Build jobs that fit in memory: each region crate's compiler can use
/// about 1.2 GB, so jobs are capped by the host's memory as well as its
/// processors.
pub fn default_jobs() -> usize {
    let cpus = std::thread::available_parallelism().map_or(1, usize::from);
    let by_memory = total_memory_gib().map_or(cpus, |gib| ((gib - 4.0) / 1.5).max(1.0) as usize);
    cpus.min(by_memory).max(1)
}

fn total_memory_gib() -> Option<f64> {
    let info = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kib: f64 = info
        .lines()
        .find_map(|line| line.strip_prefix("MemTotal:"))?
        .trim()
        .trim_end_matches("kB")
        .trim()
        .parse()
        .ok()?;
    Some(kib / f64::from(1 << 20))
}

/// Whether a cargo JSON message reports a region crate as built.
pub fn built_region(message: &str) -> bool {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(message) else {
        return false;
    };
    value["reason"] == "compiler-artifact"
        && value["target"]["name"]
            .as_str()
            .is_some_and(|name| name.starts_with("gen_r_"))
}

/// The running cargo build's process id, 0 when none.
static CARGO: AtomicU32 = AtomicU32::new(0);

/// Ends the preparation once standard input closes: the launcher keeps it
/// open, so a launcher that quits or is killed stops the build with it
/// instead of leaving an hour of compiling behind.
pub fn stop_with_stdin() {
    std::thread::spawn(|| {
        let _ = std::io::copy(&mut std::io::stdin(), &mut std::io::sink());
        let pid = CARGO.load(Ordering::SeqCst);
        if pid != 0 {
            let _ = if cfg!(windows) {
                Command::new("taskkill")
                    .args(["/T", "/F", "/PID", &pid.to_string()])
                    .status()
            } else {
                Command::new("kill").arg(pid.to_string()).status()
            };
        }
        std::process::exit(1);
    });
}

/// Builds the runner in `translation`, reporting each region crate built.
pub fn build(translation: &Path, jobs: usize) -> Result<()> {
    let regions = std::fs::read_dir(translation.join("regions"))
        .context("read the translation's regions")?
        .count();
    let mut child = Command::new(cargo())
        .args(["build", "--release", "-p", "echora-run"])
        .args(["-j", &jobs.to_string()])
        .args(["--message-format", "json-render-diagnostics"])
        .current_dir(translation)
        .stdout(Stdio::piped())
        .spawn()
        .context("start cargo (is Rust installed?)")?;
    CARGO.store(child.id(), Ordering::SeqCst);
    let stdout = child.stdout.take().context("cargo's output")?;
    let mut built = 0;
    println!("{PREFIX}built 0 of {regions}");
    for line in BufReader::new(stdout).lines() {
        if built_region(&line?) {
            built += 1;
            println!("{PREFIX}built {built} of {regions}");
        }
    }
    let status = child.wait()?;
    CARGO.store(0, Ordering::SeqCst);
    if !status.success() {
        bail!("building the translation failed ({status})");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_artifacts_count_and_others_do_not() {
        let region =
            r#"{"reason":"compiler-artifact","target":{"name":"gen_r_7100ea"},"fresh":true}"#;
        let runtime = r#"{"reason":"compiler-artifact","target":{"name":"echora_runtime"}}"#;
        let script = r#"{"reason":"build-script-executed","package_id":"gen_r_1"}"#;
        assert!(built_region(region));
        assert!(!built_region(runtime));
        assert!(!built_region(script));
        assert!(!built_region("   Compiling gen_r_7100ea"));
    }

    #[test]
    fn a_translation_needs_its_runner_and_no_other_builds_stamp() {
        let dir = std::env::temp_dir().join(format!("echora-prepare-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("target/release")).unwrap();
        assert!(!is_ready(&dir));
        std::fs::write(runner_path(&dir), b"").unwrap();
        assert!(is_ready(&dir), "a runner built by hand is used as it is");
        mark_ready(&dir).unwrap();
        assert!(is_ready(&dir));
        std::fs::write(dir.join(STAMP), "another build\n").unwrap();
        assert!(!is_ready(&dir));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn jobs_are_at_least_one() {
        assert!(default_jobs() >= 1);
    }
}
