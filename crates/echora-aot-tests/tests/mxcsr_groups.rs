//! `EmitOptions::mxcsr_groups`: vector float groups computed natively and
//! checked by the host's exception flags and a test of each result's
//! lanes, against the helpers.
//!
//! `build.rs` (`mxcsr_group_code`) lifts each vector form a group takes
//! (FADD, FSUB, FMUL, FDIV, FMLA, FMLS; 4S, 2S, 2D; by element) alone and
//! in a chain reusing registers, with the build's options, every inline
//! path, vector float runs and MXCSR groups, and with the helpers alone.
//! Each program runs from vectors whose lanes are drawn from normal,
//! inexact, denormal, smallest normal, values rounding to it, zeros,
//! infinities, quiet and signalling NaNs and values that overflow, under
//! FPCR with FZ, DN and round toward minus infinity on and off and FPSR
//! with and without IXC: the vector registers, FPCR, FPSR and the trap
//! must agree.
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
    include!(concat!(env!("OUT_DIR"), "/lifted_mxcsr_group.rs"));
}
use lifted::{mxcsr_group_code as code, mxcsr_group_reference as reference};

const FPCR_FZ: u32 = 1 << 24;
const FPCR_DN: u32 = 1 << 25;
/// Round toward minus infinity.
const FPCR_RM: u32 = 2 << 22;
const FPSR_IXC: u32 = 1 << 4;

#[derive(Debug, PartialEq)]
struct Outcome {
    vectors: [u128; 5],
    fpcr: u32,
    fpsr: u32,
    trap: String,
}

fn run(
    dispatch: fn(&mut echora_guest::Core) -> Result<(), Trap>,
    m: &mut Machine,
    entry: u64,
    vectors: [u128; 5],
    (fpcr, fpsr): (u32, u32),
) -> Outcome {
    m.core.cpu.vectors = [0; 32];
    m.core.cpu.vectors[..5].copy_from_slice(&vectors);
    m.core.cpu.fpcr = fpcr;
    m.core.cpu.fpsr = fpsr;
    m.core.cpu.pc = entry;
    let trap = match m.run(dispatch) {
        Err(trap) => format!("{trap:?} at {:#x}", m.core.cpu.pc),
        Ok(()) => unreachable!("run returns only a trap"),
    };
    Outcome {
        vectors: std::array::from_fn(|index| m.core.cpu.vectors[index]),
        fpcr: m.core.cpu.fpcr,
        fpsr: m.core.cpu.fpsr,
        trap,
    }
}

/// Single-precision lanes: every class the groups must tell apart.
const SINGLES: &[u32] = &[
    0x3fc0_0000, // 1.5
    0x4040_0000, // 3.0
    0x3dcc_cccd, // 0.1, inexact products
    0xc020_0000, // -2.5
    0x0000_0000,
    0x8000_0000,
    0x0000_0001, // the smallest denormal
    0x007f_ffff, // the largest denormal
    0x0080_0000, // the smallest normal
    0x00ff_ffff, // halved: rounds up to the smallest normal
    0x3f00_0000, // 0.5
    0x3f7f_ffff, // just below 1: products round to the smallest normal
    0x7f80_0000, // infinity
    0xff80_0000,
    0x7fc0_0001, // a quiet NaN
    0x7f80_0001, // a signalling NaN
    0x7f00_0000, // 2^127: products overflow
    0x0100_0000, // 2^-125: products underflow
];
/// The same in double precision.
const DOUBLES: &[u64] = &[
    0x3ff8_0000_0000_0000,
    0x4008_0000_0000_0000,
    0x3fb9_9999_9999_999a,
    0xc004_0000_0000_0000,
    0x0000_0000_0000_0000,
    0x8000_0000_0000_0000,
    0x0000_0000_0000_0001,
    0x0010_0000_0000_0000,
    0x001f_ffff_ffff_ffff,
    0x3fe0_0000_0000_0000,
    0x3fef_ffff_ffff_ffff,
    0x7ff0_0000_0000_0000,
    0x7ff8_0000_0000_0001,
    0x7ff0_0000_0000_0001,
    0x7fe0_0000_0000_0000,
    0x0020_0000_0000_0000,
];

/// A xorshift stream, so every run is the same.
struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A vector of lanes from `pool` (mostly one class, sometimes mixed).
    fn vector<T: Copy + Into<u128>>(&mut self, pool: &[T], width: u32) -> u128 {
        let common = pool[(self.next() % pool.len() as u64) as usize];
        (0..128 / width).fold(0u128, |vector, lane| {
            let value = if self.next().is_multiple_of(4) {
                pool[(self.next() % pool.len() as u64) as usize]
            } else {
                common
            };
            vector | (value.into() << (lane * width))
        })
    }
}

#[test]
fn the_groups_are_lifted_with_the_host_flags() {
    let source = include_str!(concat!(env!("OUT_DIR"), "/lifted_mxcsr_group.rs"));
    let (grouped, helpers) = source.split_at(source.find("pub mod mxcsr_group_reference").unwrap());
    // One read per program (and per copy with regalloc).
    let reads = grouped.matches("mxcsr_read(bad)").count();
    assert!(reads >= code::ENTRIES.len(), "{reads}");
    assert!(!helpers.contains("mxcsr_read(bad)"));
}

#[test]
fn groups_match_the_helpers() {
    let words = code::WORDS;
    let code_size = (words.len() as u64 * 4).next_multiple_of(0x1000);
    let image: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let mut m = Machine::with_layout(code::BASE, &image, code_size, DATA, DATA_SIZE);
    let statuses = [
        (FPCR_FZ, FPSR_IXC),
        (FPCR_FZ | FPCR_DN, FPSR_IXC),
        (0, FPSR_IXC),
        (FPCR_FZ, 0),
        (FPCR_FZ | FPCR_RM, FPSR_IXC),
    ];
    let mut random = Random(0x6d78_6373_725f_6772);
    let mut runs = 0;
    let mut failures = Vec::new();
    for (index, &entry) in code::ENTRIES.iter().enumerate() {
        for _ in 0..4_000 {
            let vectors: [u128; 5] = std::array::from_fn(|_| {
                if code::SINGLES[index] {
                    random.vector(SINGLES, 32)
                } else {
                    random.vector(DOUBLES, 64)
                }
            });
            let status = statuses[(random.next() % statuses.len() as u64) as usize];
            let expected = run(reference::dispatch, &mut m, entry, vectors, status);
            let actual = run(code::dispatch, &mut m, entry, vectors, status);
            runs += 1;
            if actual != expected && failures.len() < 8 {
                failures.push(format!(
                    "{} from {vectors:x?}, {status:x?}: expected {expected:x?}, got {actual:x?}",
                    code::NAMES[index]
                ));
            }
        }
    }
    println!("{runs} runs, {} mismatches", failures.len());
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
