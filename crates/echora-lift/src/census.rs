//! A static census of emitted code: per window or lifted function, what
//! its Rust spends operations on (guest memory accesses by width and
//! path, calls and indirect branches, flag computations, the stores and
//! reloads of register locals around calls and helpers, helper calls by
//! helper, inlined SIMD and float operations by the option that inlined
//! them, block charges). It reads the text split emission writes; the
//! inlined operations are told apart by the markers
//! `EmitOptions::census` adds (comments, which change nothing compiled).

use std::collections::BTreeMap;
use std::fmt::Write as _;

/// Counts for one function of emitted code.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Census {
    /// Guest loads and stores straight through the window's view of
    /// memory (`load_u8`..`store_u64`), by width: 1, 2, 4, 8 bytes.
    pub loads: [usize; 4],
    pub stores: [usize; 4],
    /// Guest accesses through the view's checked calls: `read`/`write`
    /// (inlined structure transfers, LDP/STP of Q registers through
    /// `Cpu::load_q_pair`/`store_q_pair`), atomics, exclusives and
    /// acquire/release forms. (LDR/STR of SIMD registers left to the
    /// `load`/`store` helpers count under `helpers`.)
    pub checked_accesses: usize,
    /// Calls of other guest code: `Core::call` (through the window table),
    /// `call_window` and direct calls of lifted functions or windows.
    pub calls: usize,
    /// Branches to a computed target (`next_pc = target`: BR, BLR's
    /// continuation, RET, jump tables).
    pub indirect_branches: usize,
    /// Flag computations kept (`add_sub`, `logical_flags` and the
    /// conditional compares with their flags written), and those fused
    /// with their branch instead (`fused_flags`).
    pub flags_set: usize,
    pub flags_fused: usize,
    /// Stores of register locals to the `Cpu` and reloads from it inside
    /// a function's body (around calls, helpers and exits; not its entry
    /// load or final store), and how many of each are NZCV, FPCR or FPSR.
    pub spills: usize,
    pub reloads: usize,
    pub status_spills: usize,
    pub status_reloads: usize,
    /// Block charges (`charge_block_at` or `tick_block_at`).
    pub charges: usize,
    /// Calls of `echora_guest::simd` helpers by helper (fallbacks inside
    /// inlined code included).
    pub helpers: BTreeMap<String, usize>,
    /// Operations inlined, by the option that inlined them (`vector` for
    /// `inline_vector` itself).
    pub inlined: BTreeMap<String, usize>,
}

/// The census of each function in `source` (a region's emitted Rust), in
/// order, by name (`run_window_…`, `run_function_…`).
pub fn census(source: &str) -> Vec<(String, Census)> {
    let mut functions = Vec::new();
    let mut current: Option<(String, Census)> = None;
    let mut in_body = false;
    for line in source.lines() {
        if let Some(name) = function_name(line) {
            if let Some(done) = current.take() {
                functions.push(done);
            }
            current = Some((name.to_owned(), Census::default()));
            in_body = false;
            continue;
        }
        let Some((_, counts)) = current.as_mut() else {
            continue;
        };
        if line.contains("fn body(") {
            in_body = true;
            continue;
        }
        if line.starts_with("let result = body(") {
            in_body = false;
            continue;
        }
        if !in_body {
            continue;
        }
        count_line(line, counts);
    }
    if let Some(done) = current {
        functions.push(done);
    }
    functions
}

fn function_name(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("pub fn ")?;
    let name = &rest[..rest.find('(')?];
    (name.starts_with("run_window_") || name.starts_with("run_function_")).then_some(name)
}

fn count_line(line: &str, counts: &mut Census) {
    for (index, width) in ["8", "16", "32", "64"].iter().enumerate() {
        for path in ["wmem.", "m.memory."] {
            counts.loads[index] += line.matches(&format!("{path}load_u{width}(")).count();
            counts.stores[index] += line.matches(&format!("{path}store_u{width}(")).count();
        }
    }
    for checked in [
        "load_q_pair(",
        "store_q_pair(",
        "m.memory.read(",
        "m.memory.write(",
        "m.memory.load_acquire",
        "m.memory.store_release",
        "m.memory.load_exclusive",
        "m.memory.store_exclusive",
        "m.memory.atomic",
        "m.memory.compare_and_swap",
    ] {
        counts.checked_accesses += line.matches(checked).count();
    }
    counts.calls += line.matches("m.call(").count()
        + line.matches("m.call_window(").count()
        + line.matches("= run_function_").count()
        + line.matches("= function_at(").count();
    counts.indirect_branches += line.matches("next_pc = target;").count();
    counts.flags_set += line
        .match_indices("add_sub(")
        .filter(|(at, _)| {
            line[*at..]
                .split(')')
                .next()
                .is_some_and(|call| call.ends_with("true"))
        })
        .count()
        + line.matches("logical_flags(").count()
        + line.matches("conditional_compare(").count()
        + line.matches("conditional_compare_negative(").count();
    counts.flags_fused += line.matches("let fused = ").count();
    for (at, _) in line.match_indices(" = r.") {
        let before = &line[..at];
        if let Some(place) = before.rsplit([' ', ';', '{']).next()
            && place.starts_with("m.cpu.")
        {
            counts.spills += 1;
            if ["m.cpu.nzcv", "m.cpu.fpcr", "m.cpu.fpsr"].contains(&place) {
                counts.status_spills += 1;
            }
        }
    }
    for (at, _) in line.match_indices(" = m.cpu.") {
        let before = &line[..at];
        if let Some(place) = before.rsplit([' ', ';', '{']).next()
            && place.starts_with("r.")
        {
            counts.reloads += 1;
            let field = &line[at + " = m.cpu.".len()..];
            if ["nzcv", "fpcr", "fpsr"]
                .iter()
                .any(|status| field.starts_with(status))
            {
                counts.status_reloads += 1;
            }
        }
    }
    counts.charges +=
        line.matches("m.charge_block_at(").count() + line.matches("m.tick_block_at(").count();
    const HELPER: &str = "echora_guest::simd::";
    for (at, _) in line.match_indices(HELPER) {
        let rest = &line[at + HELPER.len()..];
        let length = rest
            .find(|c: char| !(c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_'))
            .unwrap_or(rest.len());
        if length > 0 && rest[length..].starts_with('(') {
            *counts.helpers.entry(rest[..length].to_owned()).or_default() += 1;
        }
    }
    const MARKER: &str = "/*inline:";
    for (at, _) in line.match_indices(MARKER) {
        let rest = &line[at + MARKER.len()..];
        if let Some(end) = rest.find("*/") {
            *counts.inlined.entry(rest[..end].to_owned()).or_default() += 1;
        }
    }
}

/// `census` of several functions as a Markdown table, one row per
/// `(label, census)`: loads and stores (1/2/4/8 bytes), checked
/// accesses, calls, indirect branches, flags kept and fused, spills and
/// reloads (NZCV/FPCR/FPSR among them), charges, helper calls and
/// inlined operations.
pub fn table(rows: &[(String, Census)]) -> String {
    let mut text = String::from(
        "| function | loads 1/2/4/8 | stores 1/2/4/8 | checked | calls | indirect | flags kept/fused | spills (status) | reloads (status) | charges | helper calls | inlined |\n\
         |---|---|---|---|---|---|---|---|---|---|---|---|\n",
    );
    for (label, c) in rows {
        let join = |map: &BTreeMap<String, usize>| {
            if map.is_empty() {
                "-".to_owned()
            } else {
                map.iter()
                    .map(|(name, count)| format!("{name} {count}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            }
        };
        let widths = |values: &[usize; 4]| {
            values
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join("/")
        };
        writeln!(
            text,
            "| {label} | {} | {} | {} | {} | {} | {}/{} | {} ({}) | {} ({}) | {} | {} | {} |",
            widths(&c.loads),
            widths(&c.stores),
            c.checked_accesses,
            c.calls,
            c.indirect_branches,
            c.flags_set,
            c.flags_fused,
            c.spills,
            c.status_spills,
            c.reloads,
            c.status_reloads,
            c.charges,
            join(&c.helpers),
            join(&c.inlined)
        )
        .expect("writing to a String does not fail");
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small function written as split emission writes one: known counts
    /// of every class.
    const SAMPLE: &str = "\
pub fn run_window_400(m: &mut Machine) -> Result<(), Trap> {
m.enter_window()?;
let mut r = R { x0: m.cpu.regs[0], fl: cpu::Flags { nzcv: m.cpu.nzcv }, };
#[inline(always)]
fn body(m: &mut Machine, r: &mut R) -> Result<(), Trap> {
let wmem = m.memory.window_memory();
if !m.charge_block_at(0x100000, 3, false) { m.charge_block_slow(0x100000, 3, true)?; }
let loaded = wmem.load_u64(address) as u64;
let loaded = wmem.load_u8(address) as u64;
wmem.store_u32(address, (r.x0) as u32);
let value = r.fl.add_sub(lhs, rhs, true, 64, true);
let value = r.fl.add_sub(lhs, rhs, false, 64, false);
let fused = { a != b };
m.cpu.regs[0] = r.x0; m.cpu.nzcv = r.fl.nzcv; let called = m.call(0x100010); r.x0 = m.cpu.regs[0]; r.fl.nzcv = m.cpu.nzcv;
next_pc = target;
/*inline:int*/{ let vn = m.cpu.vectors[1]; }
/*inline:vector*/{ if fast { } else { echora_guest::simd::vec_fp(&mut m.cpu, echora_guest::simd::VecFpOp::Add, echora_guest::simd::FpSize::S, true, false, 0, 1, 2)?; } }
let moved: u64 = { m.memory.read(address, &mut buffer)?; 16u64 };
m.cpu.load_q_pair(&m.memory, address, 0, 1)?;
}
let result = body(m, &mut r);
m.cpu.regs[0] = r.x0;
result
}
pub fn run_function_100400(m: &mut Machine) -> Result<(), Trap> {
fn body(m: &mut Machine, r: &mut R) -> Result<(), Trap> {
if !m.charge_block_at(0x100400, 1, true) { m.charge_block_slow(0x100400, 1, true)?; }
}
let result = body(m, &mut r);
}
";

    #[test]
    fn counts_each_class_in_the_body_only() {
        let functions = census(SAMPLE);
        assert_eq!(functions.len(), 2);
        let (name, c) = &functions[0];
        assert_eq!(name, "run_window_400");
        assert_eq!(c.loads, [1, 0, 0, 1]);
        assert_eq!(c.stores, [0, 0, 1, 0]);
        assert_eq!(c.checked_accesses, 2);
        assert_eq!(c.calls, 1);
        assert_eq!(c.indirect_branches, 1);
        assert_eq!((c.flags_set, c.flags_fused), (1, 1));
        // The entry load and the final store after the body are not
        // counted.
        assert_eq!((c.spills, c.status_spills), (2, 1));
        assert_eq!((c.reloads, c.status_reloads), (2, 1));
        assert_eq!(c.charges, 1);
        assert_eq!(c.helpers.get("vec_fp"), Some(&1));
        assert_eq!(c.helpers.len(), 1);
        assert_eq!(c.inlined.get("int"), Some(&1));
        assert_eq!(c.inlined.get("vector"), Some(&1));
        let (name, c) = &functions[1];
        assert_eq!(name, "run_function_100400");
        assert_eq!(c.charges, 1);
        assert_eq!(c.loads, [0; 4]);
    }

    #[test]
    fn the_table_has_a_row_per_function() {
        let text = table(&census(SAMPLE));
        assert_eq!(text.lines().count(), 4);
        assert!(text.contains("| run_window_400 | 1/0/0/1 | 0/0/1/0 | 2 | 1 | 1 | 1/1 | 2 (1) | 2 (1) | 1 | vec_fp 1 | int 1, vector 1 |"));
    }
}
