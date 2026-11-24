//! `ECHORA_LIFT_REPORT=sizes`: the size of every emitted crate, measured on
//! the Rust it is given to rustc as, so a crate that will take too long or
//! too much memory to compile shows before the build.
//!
//! For each crate: its bytes, its functions (top-level `fn` items: windows,
//! window entries, lifted and structured functions, helpers) and its
//! largest function by bytes, with the guest blocks and instructions that
//! function holds:
//! - a window's or window function's blocks from their charges
//!   (`charge_block_at(PC, COUNT, ..)` and the `tick_*_at(PC, COUNT ..)`
//!   forms), each block once (an arm's exact and fast paths both charge);
//! - a structured function's (`regalloc`) from its `tail` table, which
//!   lists every block as `START..=END`.
//!
//! Blocks charged without their PC (`deferred_charges`' pending counts)
//! are not seen; nor are the copies `long_windows` adds, which charge as
//! the originals (counted once).

use std::collections::BTreeMap;
use std::fmt;

/// One top-level function of a crate.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FunctionSize {
    pub name: String,
    pub bytes: usize,
    pub blocks: usize,
    pub instructions: u64,
}

/// One emitted crate. A block in two crates (a window's and, with
/// `regalloc_crates`, a function crate's copy) counts in each.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CrateSize {
    /// The crate's id (its region, or a function crate's id).
    pub id: u64,
    pub bytes: usize,
    pub functions: usize,
    /// Guest instructions the crate's functions hold, each block once.
    pub instructions: u64,
    /// The function with the most bytes.
    pub largest: FunctionSize,
    /// Bytes of its structured functions (`regalloc`: a function with a
    /// `tail` table).
    pub structured_bytes: usize,
}

/// rustc's peak memory for one region crate at `opt-level` 1 and one
/// codegen unit, roughly: a fixed part and a part per byte of Rust, larger
/// for structured functions (one large host function holding every guest
/// register as a value) than for window code. Calibrated on measured
/// crates:
/// - `big_function`'s region, 240 blocks: the game set's 273 KB in 187 MB;
///   regalloc's 562 KB in one crate in 284 MB;
/// - the codesize crate of every test program with the game set, 26.4 MB in
///   4.5 GB;
/// - a game region with regalloc, 83 MB in 25 GB.
///
/// It fits those within about 20%; a crate it puts in the gigabytes is
/// worth a look before the build.
pub const RUSTC_BASE_MIB: f64 = 120.0;
/// Peak bytes per byte of window code.
pub const RUSTC_PER_WINDOW_BYTE: f64 = 175.0;
/// Peak bytes per byte of structured functions.
pub const RUSTC_PER_STRUCTURED_BYTE: f64 = 300.0;

impl CrateSize {
    /// rustc's estimated peak memory for this crate, in MiB.
    pub fn estimated_rustc_mib(&self) -> f64 {
        let other = self.bytes.saturating_sub(self.structured_bytes) as f64;
        RUSTC_BASE_MIB
            + (other * RUSTC_PER_WINDOW_BYTE
                + self.structured_bytes as f64 * RUSTC_PER_STRUCTURED_BYTE)
                / (1024.0 * 1024.0)
    }
}

/// The name of a top-level `fn` item starting `line`, if it does.
fn function_name(line: &str) -> Option<&str> {
    let rest = line
        .strip_prefix("pub fn ")
        .or_else(|| line.strip_prefix("pub(crate) fn "))
        .or_else(|| line.strip_prefix("fn "))?;
    rest.split(['(', '<']).next()
}

/// The hex number at the start of `text` (`0x...`), and what follows it.
fn hex(text: &str) -> Option<(u64, &str)> {
    let digits = text.strip_prefix("0x")?;
    let end = digits
        .find(|c: char| !c.is_ascii_hexdigit() && c != '_')
        .unwrap_or(digits.len());
    let value = u64::from_str_radix(&digits[..end].replace('_', ""), 16).ok()?;
    Some((value, &digits[end..]))
}

/// The decimal number at the start of `text`.
fn decimal(text: &str) -> Option<u64> {
    let end = text
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(text.len());
    text[..end].parse().ok()
}

/// The blocks a function's text charges or lists: start PC to instruction
/// count.
fn blocks_of(text: &str) -> BTreeMap<u64, u64> {
    let mut blocks = BTreeMap::new();
    let mut note = |pc: u64, count: u64| {
        let entry = blocks.entry(pc).or_insert(0);
        *entry = (*entry).max(count);
    };
    for line in text.lines() {
        // Charges: `charge_block_at(0xPC, COUNT` and `tick_..._at(0xPC, COUNT`.
        let mut rest = line;
        while let Some(at) = rest.find("_at(0x") {
            let call = &rest[at + 4..];
            rest = call;
            let before = &line[..line.len() - call.len() - 4];
            if !(before.ends_with("charge_block") || before.contains("m.tick_")) {
                continue;
            }
            if let Some((pc, after)) = hex(call)
                && let Some(count) = after.strip_prefix(", ").and_then(decimal)
            {
                note(pc, count);
            }
        }
        // A structured function's `tail` table: `0xSTART..=0xEND => ...`.
        let trimmed = line.trim_start();
        if let Some((start, after)) = hex(trimmed)
            && let Some((end, after)) = after.strip_prefix("..=").and_then(hex)
            && after.starts_with(" =>")
            && end >= start
        {
            note(start, (end - start) / 4 + 1);
        }
    }
    blocks
}

/// The sizes of one emitted crate's source.
pub fn crate_size(id: u64, source: &str) -> CrateSize {
    // Top-level `fn` items, delimited by brace depth (the emitted code is
    // unindented; braces in its strings come in pairs).
    let mut functions: Vec<FunctionSize> = Vec::new();
    let mut texts: Vec<String> = Vec::new();
    let mut depth = 0i64;
    let mut inside = false;
    for line in source.lines() {
        if depth == 0 {
            inside = false;
            if let Some(name) = function_name(line) {
                functions.push(FunctionSize {
                    name: name.to_owned(),
                    ..FunctionSize::default()
                });
                texts.push(String::new());
                inside = true;
            }
        }
        if inside && let Some(text) = texts.last_mut() {
            text.push_str(line);
            text.push('\n');
        }
        depth += line.matches('{').count() as i64 - line.matches('}').count() as i64;
        depth = depth.max(0);
    }
    let mut all_blocks = BTreeMap::new();
    let mut named = Vec::new();
    let mut structured_bytes = 0;
    for (mut function, text) in functions.into_iter().zip(texts) {
        if text.contains("fn tail(") {
            structured_bytes += text.len();
        }
        let blocks = blocks_of(&text);
        function.bytes = text.len();
        function.blocks = blocks.len();
        function.instructions = blocks.values().sum();
        for (pc, count) in blocks {
            let entry = all_blocks.entry(pc).or_insert(0);
            *entry = count.max(*entry);
        }
        named.push(function);
    }
    let largest = named
        .iter()
        .max_by_key(|function| function.bytes)
        .cloned()
        .unwrap_or_default();
    CrateSize {
        id,
        bytes: source.len(),
        functions: named.len(),
        instructions: all_blocks.values().sum(),
        largest,
        structured_bytes,
    }
}

/// The report over every crate: the `top` largest by bytes, and totals.
#[derive(Debug, Default)]
pub struct SizeReport {
    pub crates: Vec<CrateSize>,
    pub top: usize,
}

impl fmt::Display for SizeReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut crates: Vec<&CrateSize> = self.crates.iter().collect();
        crates.sort_by_key(|size| std::cmp::Reverse(size.bytes));
        let total: usize = crates.iter().map(|size| size.bytes).sum();
        let largest_estimate = crates
            .iter()
            .map(|size| size.estimated_rustc_mib())
            .fold(0.0, f64::max);
        writeln!(
            f,
            "crate sizes: {} crates, {:.1} MB of Rust, the largest rustc estimate {:.1} GiB; the {} largest:",
            crates.len(),
            total as f64 / 1e6,
            largest_estimate / 1024.0,
            self.top.min(crates.len())
        )?;
        writeln!(
            f,
            "  {:>10} {:>10} {:>10} {:>9} {:>11} {:>9}  largest function (bytes, blocks, instructions)",
            "crate", "bytes", "structured", "functions", "instructions", "rustc MiB"
        )?;
        for size in crates.iter().take(self.top) {
            writeln!(
                f,
                "  {:>10x} {:>10} {:>10} {:>9} {:>11} {:>9.0}  {} ({}, {}, {})",
                size.id,
                size.bytes,
                size.structured_bytes,
                size.functions,
                size.instructions,
                size.estimated_rustc_mib(),
                size.largest.name,
                size.largest.bytes,
                size.largest.blocks,
                size.largest.instructions
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::{CodeRegion, EmitOptions, discover, emit_rust_split_with};

    fn region(words: &[u32]) -> CodeRegion {
        CodeRegion {
            name: "synthetic".into(),
            base: 0x1000,
            bytes: words.iter().flat_map(|word| word.to_le_bytes()).collect(),
        }
    }

    /// A loop calling a leaf, the leaf, and an exit: blocks of 3, 1, 2 and
    /// 1 instructions.
    fn program() -> Vec<u32> {
        vec![
            0x9100_0400, // ADD X0, X0, #1
            0xf100_0421, // SUBS X1, X1, #1
            0x54ff_ffc1, // B.NE -8
            0x9400_0003, // BL +12
            0xd400_00e1, // SVC #7
            0xd503_201f, // NOP
            0x9100_0442, // ADD X2, X2, #1
            0xd65f_03c0, // RET
        ]
    }

    #[test]
    fn crate_sizes_count_every_block_once() {
        let translation = discover(&[region(&program())], &[0x1000], 16).unwrap();
        let expected_instructions: u64 = translation
            .blocks
            .iter()
            .map(|block| block.instructions.len() as u64)
            .sum();
        for options in [
            EmitOptions {
                split_ticks: true,
                untagged_blocks: true,
                ..EmitOptions::default()
            },
            EmitOptions::default(),
            EmitOptions {
                split_ticks: true,
                untagged_blocks: true,
                functions: true,
                function_table: true,
                regalloc: true,
                ..EmitOptions::default()
            },
            EmitOptions {
                split_ticks: true,
                untagged_blocks: true,
                functions: true,
                function_table: true,
                regalloc: true,
                regalloc_crates: true,
                ..EmitOptions::default()
            },
        ] {
            let (_, regions) = emit_rust_split_with(&translation, 16, options);
            let sizes: Vec<CrateSize> = regions
                .iter()
                .map(|(id, source)| crate_size(*id, source))
                .collect();
            let instructions: u64 = sizes.iter().map(|size| size.instructions).sum();
            if options.regalloc_crates {
                // A function crate holds copies of blocks its window crate
                // holds too; each crate counts what it compiles.
                assert!(sizes.len() > 1, "{options:?}");
                assert!(instructions > expected_instructions, "{options:?}");
                assert!(
                    sizes
                        .iter()
                        .all(|size| size.instructions <= expected_instructions)
                );
            } else {
                assert_eq!(instructions, expected_instructions, "{options:?}");
            }
            let bytes: usize = sizes.iter().map(|size| size.bytes).sum();
            let source_bytes: usize = regions.iter().map(|(_, source)| source.len()).sum();
            assert_eq!(bytes, source_bytes);
            let largest = sizes
                .iter()
                .map(|size| &size.largest)
                .max_by_key(|f| f.bytes)
                .unwrap();
            assert!(largest.name.starts_with("run_"), "{largest:?}");
            assert!(
                largest.blocks >= 1 && largest.instructions >= 1,
                "{largest:?}"
            );
        }
    }

    #[test]
    fn functions_and_their_blocks_are_read_from_the_text() {
        let source = "\
use echora_guest::cpu;
#[inline(never)]
pub fn run_window_4(m: &mut Machine) -> Result<(), Trap> {
    if !m.charge_block_at(0x1000, 3, false) { m.charge_block_slow(0x1000, 3, true)?; }
    if !m.charge_block_at(0x1000, 3, false) { m.charge_block_slow(0x1000, 3, true)?; }
    m.tick_untagged_block_at(0x100c, 1)?;
}
static WINDOWS: &[u64] = &[];
pub fn run_function_1018(m: &mut Machine) -> Result<(), Trap> {
    fn tail(pc: u64) -> u64 {
        match pc {
            0x1018..=0x101c => (0x101c - pc) / 4,
            0x1020..=0x1020 => (0x1020 - pc) / 4,
            _ => 0,
        }
    }
}
";
        let size = crate_size(0x4, source);
        assert_eq!(size.functions, 2);
        assert_eq!(size.instructions, 3 + 1 + 2 + 1);
        let window = FunctionSize {
            name: "run_window_4".into(),
            bytes: 0,
            blocks: 2,
            instructions: 4,
        };
        // The function with a `tail` table is structured.
        assert!(size.structured_bytes > 0 && size.structured_bytes < size.bytes);
        // The window is the larger one.
        assert_eq!(size.largest.name, window.name);
        assert_eq!(
            (size.largest.blocks, size.largest.instructions),
            (window.blocks, window.instructions)
        );
    }

    #[test]
    fn the_rustc_estimate_fits_its_calibration() {
        // (bytes, structured bytes, measured MiB)
        let measured = [
            (272_834, 0, 187.0),
            (562_318, 248_743, 284.0),
            (26_400_000, 0, 4_498.0),
            (83_000_000, 83_000_000, 25_600.0),
        ];
        for (bytes, structured_bytes, mib) in measured {
            let size = CrateSize {
                bytes,
                structured_bytes,
                ..CrateSize::default()
            };
            let estimate = size.estimated_rustc_mib();
            assert!(
                (estimate / mib - 1.0).abs() < 0.2,
                "{bytes} bytes: estimated {estimate:.0} MiB, measured {mib}"
            );
        }
    }
}
