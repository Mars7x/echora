//! `EmitOptions::mxcsr_groups` on every form a group admits: each computed
//! by the group's committing path must be the helpers' result bit for bit.
//!
//! `build.rs` (`mxcsr_form_code`) finds every form `mxcsr_group_op` or
//! `run_other_op` takes (vector, by-element and pairwise float arithmetic,
//! the compares, FMAX and FMIN in every size, FABD, FMULX, unary forms) and
//! lifts each alone, after an FMUL (so inside a group) and before an FADD,
//! and a block shaped like a game's angle wrap (FRINTM, FMUL by element,
//! FSUB, FCMGT, BSL, FMAX, FCMEQ, FMOV, TBZ) in 2S and 4S, with every inline
//! path, vector float runs and MXCSR groups, and with the helpers alone.
//! Each runs from normal operands with FPSR.IXC set, so the groups commit:
//! the vector registers, the general registers, NZCV, FPCR, FPSR and the
//! trap must agree. A group computing a form it does not implement (a
//! compare as a fused multiply-subtract, as before 2026-11) fails here.
#![cfg(target_arch = "x86_64")]
#![cfg(any(target_os = "linux", windows))]
#![allow(clippy::unwrap_used)]

mod common;

use common::*;
use echora_aot_tests::{Machine, Trap};

#[allow(
    missing_debug_implementations,
    clippy::all,
    dead_code,
    unused_imports,
    unused_parens,
    unreachable_code,
    unsafe_code
)]
mod lifted {
    include!(concat!(env!("OUT_DIR"), "/lifted_mxcsr_form.rs"));
}
use lifted::{mxcsr_form_code as code, mxcsr_form_reference as reference};

const FPCR_FZ: u32 = 1 << 24;
const FPCR_DN: u32 = 1 << 25;
const FPSR_IXC: u32 = 1 << 4;

#[derive(Debug, PartialEq)]
struct Outcome {
    vectors: [u128; 8],
    regs: [u64; 9],
    nzcv: u8,
    fpcr: u32,
    fpsr: u32,
    trap: String,
}

fn run(
    dispatch: fn(&mut echora_guest::Core) -> Result<(), Trap>,
    m: &mut Machine,
    entry: u64,
    vectors: [u128; 8],
    fpcr: u32,
) -> Outcome {
    m.core.cpu.vectors = [0; 32];
    m.core.cpu.vectors[..8].copy_from_slice(&vectors);
    m.core.cpu.regs = [0; 31];
    m.core.cpu.nzcv = 0;
    m.core.cpu.fpcr = fpcr;
    m.core.cpu.fpsr = FPSR_IXC;
    m.core.cpu.pc = entry;
    let trap = match m.run(dispatch) {
        Err(trap) => format!("{trap:?} at {:#x}", m.core.cpu.pc),
        Ok(()) => unreachable!("run returns only a trap"),
    };
    Outcome {
        vectors: std::array::from_fn(|index| m.core.cpu.vectors[index]),
        regs: std::array::from_fn(|index| m.core.cpu.regs[index]),
        nzcv: m.core.cpu.nzcv,
        fpcr: m.core.cpu.fpcr,
        fpsr: m.core.cpu.fpsr,
        trap,
    }
}

/// A xorshift stream, so every run is the same.
struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A normal number of either sign between 1/4 and 8, sometimes repeated
    /// from `previous` (so compares see equal lanes).
    fn normal(&mut self, previous: f64) -> f64 {
        if self.next().is_multiple_of(5) {
            return previous;
        }
        let fraction = 1.0 + (self.next() % 1024) as f64 / 1024.0;
        let scale = [0.25, 0.5, 1.0, 2.0, 4.0][(self.next() % 5) as usize];
        let sign = if self.next().is_multiple_of(2) {
            -1.0
        } else {
            1.0
        };
        sign * fraction * scale
    }

    fn vector(&mut self, single: bool) -> u128 {
        let mut previous = 1.5;
        if single {
            (0..4).fold(0u128, |vector, lane| {
                previous = self.normal(previous);
                vector | u128::from((previous as f32).to_bits()) << (lane * 32)
            })
        } else {
            (0..2).fold(0u128, |vector, lane| {
                previous = self.normal(previous);
                vector | u128::from(previous.to_bits()) << (lane * 64)
            })
        }
    }
}

#[test]
fn the_forms_are_lifted_in_groups() {
    let source = include_str!(concat!(env!("OUT_DIR"), "/lifted_mxcsr_form.rs"));
    let (grouped, helpers) = source.split_at(source.find("pub mod mxcsr_form_reference").unwrap());
    // Most programs hold a group (every form after an FMUL does).
    let groups = grouped.matches("mxcsr_read(bad)").count();
    assert!(groups >= code::ENTRIES.len() / 3, "{groups} groups");
    assert!(!helpers.contains("mxcsr_read(bad)"));
}

#[test]
fn every_form_in_a_group_matches_the_helpers() {
    let words = code::WORDS;
    let code_size = (words.len() as u64 * 4).next_multiple_of(0x1000);
    let image: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let mut m = Machine::with_layout(code::BASE, &image, code_size, DATA, DATA_SIZE);
    let mut random = Random(0x6d78_666f_726d_7331);
    let mut runs = 0;
    let mut failing: Vec<String> = Vec::new();
    for (index, &entry) in code::ENTRIES.iter().enumerate() {
        let mut first = None;
        for _ in 0..200 {
            let vectors: [u128; 8] = std::array::from_fn(|_| random.vector(code::SINGLES[index]));
            let fpcr = [0, FPCR_FZ, FPCR_FZ | FPCR_DN][(random.next() % 3) as usize];
            let expected = run(reference::dispatch, &mut m, entry, vectors, fpcr);
            let actual = run(code::dispatch, &mut m, entry, vectors, fpcr);
            runs += 1;
            if actual != expected && first.is_none() {
                first = Some(format!(
                    "{} from {vectors:x?}, fpcr {fpcr:#x}: expected {expected:x?}, got {actual:x?}",
                    code::NAMES[index]
                ));
            }
        }
        failing.extend(first);
    }
    println!(
        "{} programs, {runs} runs, {} failing",
        code::ENTRIES.len(),
        failing.len()
    );
    let shown: Vec<&String> = failing.iter().take(12).collect();
    assert!(
        failing.is_empty(),
        "{} programs differ:\n{}",
        failing.len(),
        shown
            .iter()
            .map(|line| line.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    );
}
