//! Float runs (`EmitOptions::float_runs`) against the same programs lifted
//! without them: runs of 8 to 16 scalar single-precision operations
//! (`build.rs`'s `float_run_code`) from states mixing ordinary normals with
//! zeros, denormals, the smallest normal, MAX, infinities, quiet and
//! signalling NaNs and operands that overflow or underflow, under FPCR with
//! FZ and DN each on or off and a directed rounding mode, and FPSR with the
//! sticky IXC set or not. Every vector register and FPSR must match bit for
//! bit, whether the fast copy committed or declined to the exact one.
#![cfg(any(target_os = "linux", windows))]
#![allow(clippy::unwrap_used)]

use echora_aot_tests::{Machine, Trap};

/// The lifts this test runs (`build.rs` writes each to its own file, so
/// that no one compilation holds every lift).
#[allow(
    missing_debug_implementations,
    clippy::all,
    dead_code,
    unused_imports,
    unused_parens,
    unreachable_code
)]
mod lifted {
    include!(concat!(env!("OUT_DIR"), "/lifted_float_run.rs"));
    include!(concat!(env!("OUT_DIR"), "/lifted_float_run_load.rs"));
}
use lifted::{float_run_code, float_run_load_code, float_run_load_reference, float_run_reference};

const STATES_PER_PROGRAM: usize = 400;
const FZ: u32 = 1 << 24;
const DN: u32 = 1 << 25;
const RMODE_PLUS: u32 = 1 << 22;
const IXC: u32 = 1 << 4;

struct Random(u64);

impl Random {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }

    /// A single-precision value: mostly ordinary normals (where the fast
    /// copy commits), else an edge case.
    fn float(&mut self) -> u32 {
        const EDGES: [u32; 16] = [
            // About 1e-31: products of two underflow to zero.
            0x0c00_0000,
            0,
            0x8000_0000,
            1,
            0x807f_ffff,
            0x0080_0000,
            0x8080_0001,
            0x7f7f_ffff,
            0x7f80_0000,
            0xff80_0000,
            0x7fc0_0000,
            0xffc0_1234,
            0x7fa0_0000,
            0x1e00_0000,
            0x5e00_0000,
            0x3f80_0000,
        ];
        match self.next() % 8 {
            0 | 1 => EDGES[self.next() as usize % EDGES.len()],
            2 => self.next(),
            // Normals of magnitude 1/8 to 8.
            _ => (0x3e00_0000 + self.next() % 0x0200_0000) | (self.next() & 0x8000_0000),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
struct State {
    v: [u128; 32],
    fpcr: u32,
    fpsr: u32,
}

fn machine() -> Machine {
    let words = float_run_code::WORDS;
    let image: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    Machine::with_layout(
        float_run_code::BASE,
        &image,
        (image.len() as u64).next_multiple_of(0x1000),
        0x80_0000,
        0x1000,
    )
}

fn run(
    m: &mut Machine,
    dispatch: fn(&mut echora_aot_tests::Core) -> Result<(), Trap>,
    entry: u64,
    state: &State,
) -> Result<State, String> {
    m.core.cpu.vectors = state.v;
    m.core.cpu.fpcr = state.fpcr;
    m.core.cpu.fpsr = state.fpsr;
    m.core.cpu.pc = entry;
    loop {
        match dispatch(&mut m.core) {
            Ok(()) => continue,
            Err(Trap::SupervisorCall { number: 7 }) => break,
            // A rounding mode the helpers do not implement stops the run
            // (both copies must stop the same way).
            Err(trap) => return Err(format!("{trap:?} at {:#x}", m.core.cpu.pc)),
        }
    }
    Ok(State {
        v: m.core.cpu.vectors,
        fpcr: m.core.cpu.fpcr,
        fpsr: m.core.cpu.fpsr,
    })
}

#[test]
fn every_program_is_one_float_run() {
    let programs = float_run_code::ENTRIES.len();
    assert_eq!(float_run_code::RUNS.0, programs);
    assert!(float_run_code::RUNS.1 >= 8 * programs);
}

#[test]
fn float_runs_match_the_exact_emission_bit_for_bit() {
    let (mut fast, mut exact) = (machine(), machine());
    let mut random = Random(0x2545_f491_4f6c_dd1d);
    let mut failures = Vec::new();
    let mut runs = 0;
    for &entry in float_run_code::ENTRIES {
        for _ in 0..STATES_PER_PROGRAM {
            let mut v = [0u128; 32];
            for register in &mut v {
                // Upper bits the scalar writes must clear.
                *register = u128::from(random.float())
                    | u128::from(random.next()) << 64
                    | u128::from(random.next()) << 96;
            }
            let mut fpcr = [0, 0, FZ, FZ, DN, FZ | DN, RMODE_PLUS][random.next() as usize % 7];
            let mut fpsr = if random.next().is_multiple_of(4) {
                0
            } else {
                IXC
            } | random.next() & 0x8f;
            if random.next().is_multiple_of(8) {
                // Tiny operands (about 1e-31) among normals, with only IXC
                // raised: products of two underflow to zero, which must
                // raise UFC.
                for register in &mut v[..8] {
                    *register = if random.next().is_multiple_of(2) {
                        u128::from(0x0c00_0000 | random.next() & 0x8000_0000)
                    } else {
                        u128::from(0x3f80_0000 + random.next() % 0x0080_0000)
                    };
                }
                fpcr = 0;
                fpsr = IXC;
            }
            let state = State { v, fpcr, fpsr };
            let expected = run(&mut exact, float_run_reference::dispatch, entry, &state);
            let actual = run(&mut fast, float_run_code::dispatch, entry, &state);
            runs += 1;
            let (actual, expected) = match (actual, expected) {
                (Ok(actual), Ok(expected)) => (actual, expected),
                (actual, expected) => {
                    if actual != expected && failures.len() < 10 {
                        failures.push(format!("program at {entry:#x}: {actual:?} vs {expected:?}"));
                    }
                    continue;
                }
            };
            if actual != expected && failures.len() < 10 {
                let registers: Vec<String> = (0..8)
                    .filter(|&n| actual.v[n] != expected.v[n])
                    .map(|n| format!("s{n} {:#x} (expected {:#x})", actual.v[n], expected.v[n]))
                    .collect();
                failures.push(format!(
                    "program at {entry:#x}, fpcr {fpcr:#x} fpsr {fpsr:#x} inputs {:x?}: {} fpsr {:#x} (expected {:#x})",
                    state.v[..8].iter().map(|v| *v as u32).collect::<Vec<_>>(),
                    registers.join(", "),
                    actual.fpsr,
                    expected.fpsr
                ));
            }
        }
    }
    println!("{runs} runs");
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Data the load programs read: X9 starts in its middle, SP 8 bytes below
/// its end (16-byte aligned or not).
const DATA: u64 = 0x80_0000;
const DATA_SIZE: u64 = 0x1000;

#[derive(Clone, PartialEq, Eq, Debug)]
struct LoadState {
    v: [u128; 32],
    x9: u64,
    x10: u64,
    sp: u64,
    fpsr: u32,
}

fn load_machine() -> Machine {
    let words = float_run_load_code::WORDS;
    let image: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    Machine::with_layout(
        float_run_load_code::BASE,
        &image,
        (image.len() as u64).next_multiple_of(0x1000),
        DATA,
        DATA_SIZE,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_loads(
    m: &mut Machine,
    dispatch: fn(&mut echora_aot_tests::Core) -> Result<(), Trap>,
    entry: u64,
    v: &[u128; 32],
    data: &[u8],
    (x9, x10, sp): (u64, u64, u64),
    fpcr: u32,
    fpsr: u32,
) -> Result<LoadState, String> {
    m.memory.write(DATA, data).unwrap();
    m.core.cpu.vectors = *v;
    m.core.cpu.regs[9] = x9;
    m.core.cpu.regs[10] = x10;
    m.core.cpu.sp = sp;
    m.core.cpu.fpcr = fpcr;
    m.core.cpu.fpsr = fpsr;
    m.core.cpu.pc = entry;
    loop {
        match dispatch(&mut m.core) {
            Ok(()) => continue,
            Err(Trap::SupervisorCall { number: 7 }) => break,
            Err(trap) => return Err(format!("{trap:?} at {:#x}", m.core.cpu.pc)),
        }
    }
    Ok(LoadState {
        v: m.core.cpu.vectors,
        x9: m.core.cpu.regs[9],
        x10: m.core.cpu.regs[10],
        sp: m.core.cpu.sp,
        fpsr: m.core.cpu.fpsr,
    })
}

#[test]
fn every_load_program_is_one_float_run() {
    let programs = float_run_load_code::ENTRIES.len();
    assert_eq!(float_run_load_code::RUNS.0, programs);
}

#[test]
fn float_runs_with_loads_match_the_exact_emission_bit_for_bit() {
    let (mut fast, mut exact) = (load_machine(), load_machine());
    let mut random = Random(0x9e37_79b9_7f4a_7c15);
    let mut failures = Vec::new();
    for &entry in float_run_load_code::ENTRIES {
        for _ in 0..STATES_PER_PROGRAM / 2 {
            let mut v = [0u128; 32];
            for register in &mut v {
                *register = u128::from(random.float()) | u128::from(random.next()) << 64;
            }
            let data: Vec<u8> = (0..DATA_SIZE / 4)
                .flat_map(|_| random.float().to_le_bytes())
                .collect();
            // X9 in the middle, X10 a small index, SP aligned or not.
            let registers = (
                DATA + 0x800,
                u64::from(random.next() % 32),
                DATA + DATA_SIZE - 0x10
                    + if random.next().is_multiple_of(4) {
                        4
                    } else {
                        0
                    },
            );
            let fpcr = [0, 0, FZ, FZ, DN, FZ | DN, RMODE_PLUS][random.next() as usize % 7];
            let fpsr = if random.next().is_multiple_of(4) {
                0
            } else {
                IXC
            } | random.next() & 0x8f;
            let expected = run_loads(
                &mut exact,
                float_run_load_reference::dispatch,
                entry,
                &v,
                &data,
                registers,
                fpcr,
                fpsr,
            );
            let actual = run_loads(
                &mut fast,
                float_run_load_code::dispatch,
                entry,
                &v,
                &data,
                registers,
                fpcr,
                fpsr,
            );
            if actual != expected && failures.len() < 10 {
                failures.push(format!(
                    "program at {entry:#x}, fpcr {fpcr:#x} fpsr {fpsr:#x} sp {:#x}: {actual:x?} vs {expected:x?}",
                    registers.2
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
