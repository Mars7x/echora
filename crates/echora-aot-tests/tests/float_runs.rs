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

use echora_aot_tests::{Machine, Trap, float_run_code, float_run_reference};

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
