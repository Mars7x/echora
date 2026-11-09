//! Regenerates the checked-in programs from their assembly sources:
//! `src/kernels.rs` from `benches/kernels.s` and `src/stress.rs` from
//! `tests/stress.s`. Needs llvm-mc, llvm-objcopy, llvm-objdump and llvm-nm.
//!
//! `cargo run -p echora-aot-tests --bin assemble [kernels|stress]` (both
//! without an argument).

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

const NOP: u32 = 0xd503_201f;

const KERNELS: [&str; 9] = [
    "arith", "memcpy", "memset", "calls", "switch", "vector", "poll", "scalar", "ldst",
];

const STRESS_PROGRAMS: [&str; 10] = [
    "claim_cas",
    "claim_llsc",
    "claim_ldadd",
    "ticket",
    "seqlock",
    "pass_acquire",
    "pass_dmb",
    "casp_stack",
    "pair_halves",
    "vector_halves",
];

type Result<T> = std::result::Result<T, String>;

fn main() {
    let which: Vec<String> = std::env::args().skip(1).collect();
    let all = which.is_empty();
    let wants = |name: &str| all || which.iter().any(|arg| arg == name);
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut result = Ok(());
    if wants("kernels") {
        result = result.and_then(|()| kernels(crate_dir));
    }
    if wants("stress") {
        result = result.and_then(|()| stress(crate_dir));
    }
    if let Err(error) = result {
        eprintln!("assemble: {error}");
        std::process::exit(1);
    }
}

/// An assembled source: its `.text` bytes, symbols and the object file.
struct Assembled {
    text: Vec<u8>,
    labels: BTreeMap<String, u64>,
    object: PathBuf,
    _dir: TempDir,
}

struct TempDir(PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn run(program: &str, args: &[&str]) -> Result<String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|error| format!("{program}: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "{program} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    String::from_utf8(output.stdout).map_err(|error| format!("{program}: {error}"))
}

fn assemble(source: &Path, attributes: &[&str]) -> Result<Assembled> {
    let dir = std::env::temp_dir().join(format!("echora-assemble-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|error| error.to_string())?;
    let dir = TempDir(dir);
    let object = dir.0.join("program.o");
    let binary = dir.0.join("program.bin");
    let (source, object_str, binary_str) = (
        source.to_string_lossy(),
        object.to_string_lossy(),
        binary.to_string_lossy(),
    );
    let mut args = vec!["-triple=aarch64"];
    args.extend_from_slice(attributes);
    args.extend(["-filetype=obj", &source, "-o", &object_str]);
    run("llvm-mc", &args)?;
    run(
        "llvm-objcopy",
        &[
            "-O",
            "binary",
            "--only-section=.text",
            &object_str,
            &binary_str,
        ],
    )?;
    let text = std::fs::read(&binary).map_err(|error| error.to_string())?;
    let mut labels = BTreeMap::new();
    for line in run("llvm-nm", &[&object_str])?.lines() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        let [address, _, name] = fields[..] else {
            return Err(format!("unexpected llvm-nm line: {line}"));
        };
        let address =
            u64::from_str_radix(address, 16).map_err(|error| format!("{line}: {error}"))?;
        labels.insert(name.to_string(), address);
    }
    Ok(Assembled {
        text,
        labels,
        object,
        _dir: dir,
    })
}

fn label(labels: &BTreeMap<String, u64>, name: &str) -> Result<u64> {
    labels
        .get(name)
        .copied()
        .ok_or_else(|| format!("no symbol {name}"))
}

fn words(bytes: &[u8]) -> Vec<u32> {
    let (words, _) = bytes.as_chunks::<4>();
    words.iter().map(|word| u32::from_le_bytes(*word)).collect()
}

fn write(path: &Path, text: &str) -> Result<()> {
    std::fs::write(path, text).map_err(|error| format!("{}: {error}", path.display()))
}

fn kernels(crate_dir: &Path) -> Result<()> {
    let mut assembled = assemble(&crate_dir.join("benches/kernels.s"), &[])?;
    let disassembly = run("llvm-objdump", &["-d", &assembled.object.to_string_lossy()])?;
    let data = &mut assembled.text;
    data.resize(data.len().next_multiple_of(4), 0);
    let words = words(data);
    let labels = &assembled.labels;
    // Each instruction's disassembly, by offset.
    let mut comments: BTreeMap<u64, String> = BTreeMap::new();
    for line in disassembly.lines() {
        let parts: Vec<&str> = line.split('\t').collect();
        let head = parts[0].trim();
        if parts.len() >= 2 && head.ends_with(':') && head.contains(' ') {
            let offset = head.split(':').next().unwrap_or_default();
            let offset =
                u64::from_str_radix(offset, 16).map_err(|error| format!("{line}: {error}"))?;
            let comment: Vec<&str> = parts[1..parts.len().min(3)]
                .iter()
                .map(|part| part.trim())
                .collect();
            comments.insert(offset, comment.join(" ").trim().to_string());
        }
    }
    let size = (data.len() as u64).next_multiple_of(0x400);
    let table = label(labels, "table")?;
    for offset in (table..table + 16).step_by(4) {
        comments.insert(offset, "jump table halfwords".into());
    }
    let mut runs: Vec<(u64, &str)> = KERNELS
        .iter()
        .map(|&name| Ok((label(labels, name)?, name)))
        .collect::<Result<_>>()?;
    runs.push((label(labels, "leaf_add")?, "leaves"));
    runs.sort();

    let mut out = String::from(
        "// Generated by `src/bin/assemble.rs` from `benches/kernels.s`;\n\
         // shared by `build.rs`, which lifts the kernels, and the benchmark,\n\
         // which runs them. Original code, never copied from a game.\n\n\
         /// Where the kernel image is loaded: page and window aligned.\n\
         pub const KERNEL_BASE: u64 = 0x10_0000;\n\n\
         /// Every kernel by name, with its entry offset from `KERNEL_BASE`.\n\
         pub const KERNEL_ENTRIES: &[(&str, u64)] = &[\n",
    );
    for name in KERNELS {
        let _ = writeln!(out, "    (\"{name}\", {:#x}),", label(labels, name)?);
    }
    let _ = write!(
        out,
        "];\n\n\
         /// Bytes of the image: whole 1 KiB windows.\n\
         pub const KERNEL_SIZE: u64 = {size:#x};\n\n\
         /// The image as bytes from `KERNEL_BASE`, with zeros (UDF) in the gaps.\n\
         pub fn image() -> Vec<u8> {{\n\
         \x20   let mut bytes = vec![0; KERNEL_SIZE as usize];\n\
         \x20   for &(offset, words) in KERNEL_IMAGE {{\n\
         \x20       for (index, word) in words.iter().enumerate() {{\n\
         \x20           let at = offset as usize + index * 4;\n\
         \x20           bytes[at..at + 4].copy_from_slice(&word.to_le_bytes());\n\
         \x20       }}\n\
         \x20   }}\n\
         \x20   bytes\n\
         }}\n\n\
         /// `(offset, words)` runs of the image; the gaps are never executed.\n\
         pub const KERNEL_IMAGE: &[(u64, &[u32])] = &[\n"
    );
    for (index, &(start, _)) in runs.iter().enumerate() {
        let end = runs
            .get(index + 1)
            .map_or(words.len() as u64 * 4, |&(next, _)| next);
        let mut run = words[(start / 4) as usize..(end / 4) as usize].to_vec();
        while run.last().is_some_and(|&word| word == 0 || word == NOP) {
            run.pop();
        }
        let _ = write!(out, "    (\n        {start:#x},\n        &[\n");
        for (i, word) in run.iter().enumerate() {
            let comment = comments
                .get(&(start + i as u64 * 4))
                .map_or("", String::as_str);
            let _ = writeln!(
                out,
                "            0x{:04x}_{:04x}, // {comment}",
                word >> 16,
                word & 0xffff
            );
        }
        out.push_str("        ],\n    ),\n");
    }
    out.push_str("];\n");
    write(&crate_dir.join("src/kernels.rs"), &out)
}

fn stress(crate_dir: &Path) -> Result<()> {
    let mut assembled = assemble(&crate_dir.join("tests/stress.s"), &["-mattr=+lse"])?;
    let data = &mut assembled.text;
    data.resize(data.len().next_multiple_of(0x1000), 0);
    let words = words(data);
    let mut out = String::from(
        "// Generated by `src/bin/assemble.rs` from `tests/stress.s`;\n\
         // lifted by `build.rs` and run by `tests/stress.rs`. Original code.\n\n\
         /// Where the stress programs are loaded.\n\
         pub const STRESS_BASE: u64 = 0x300_0000;\n\n\
         /// Every program by name, with its entry offset from `STRESS_BASE`.\n\
         pub const STRESS_ENTRIES: &[(&str, u64)] = &[\n",
    );
    for name in STRESS_PROGRAMS {
        let _ = writeln!(
            out,
            "    (\"{name}\", {:#x}),",
            label(&assembled.labels, name)?
        );
    }
    out.push_str("];\n\n/// The image, whole pages.\npub const STRESS_WORDS: &[u32] = &[\n");
    for chunk in words.chunks(8) {
        let line: Vec<String> = chunk.iter().map(|word| format!("{word:#010x},")).collect();
        let _ = writeln!(out, "    {}", line.join(" "));
    }
    out.push_str("];\n");
    write(&crate_dir.join("src/stress.rs"), &out)
}
