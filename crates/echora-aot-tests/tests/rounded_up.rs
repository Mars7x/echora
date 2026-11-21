//! Products, quotients and fused products whose unrounded value is tiny
//! but rounds up to the smallest normal (`build.rs`, `rounded_up_code`).
//! Arm detects tininess before rounding, so these underflow: with FPCR.FZ
//! the result is a zero of its sign, without it the smallest normal, and
//! UFC is raised either way. The host's result is that normal, which no
//! inline path (lane by lane, whole vector, scalar, misc, float runs) nor
//! a helper's fast path may take. Each program runs with FPSR.IXC already
//! set (what the fast paths wait for), with FPCR.FZ clear and set, under
//! every lift, against values known from the architecture.
#![cfg(any(target_os = "linux", windows))]
#![allow(clippy::unwrap_used)]

mod common;

use common::*;
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
    include!(concat!(env!("OUT_DIR"), "/lifted_rounded_up.rs"));
}
use lifted::{rounded_up_inline, rounded_up_reference, rounded_up_runs, rounded_up_whole};

const FPCR_FZ: u32 = 1 << 24;
const FPSR_UFC: u32 = 1 << 3;
const FPSR_IXC: u32 = 1 << 4;

type Dispatch = fn(&mut echora_guest::Core) -> Result<(), Trap>;

fn run(
    dispatch: Dispatch,
    m: &mut Machine,
    entry: u64,
    vectors: [u128; 4],
    fpcr: u32,
) -> (u128, u32) {
    m.core.cpu.vectors[..4].copy_from_slice(&vectors);
    m.core.cpu.fpcr = fpcr;
    m.core.cpu.fpsr = FPSR_IXC;
    m.core.cpu.pc = entry;
    m.core.host_check_at = u64::MAX;
    m.core.rearm();
    let trap = loop {
        match dispatch(&mut m.core) {
            Ok(()) => continue,
            Err(trap) => break trap,
        }
    };
    m.core.settle_block_trap();
    assert!(
        matches!(trap, Trap::SupervisorCall { number: 7 }),
        "{entry:#x}: {trap:?}"
    );
    (m.core.cpu.vectors[0], m.core.cpu.fpsr)
}

/// `lanes` copies of `value` (`width` bits each).
fn splat(value: u64, width: u32, lanes: u32) -> u128 {
    (0..lanes).fold(0, |vector, lane| {
        vector | u128::from(value) << (lane * width)
    })
}

#[test]
fn results_rounded_up_to_the_smallest_normal_underflow_in_every_lift() {
    let words = rounded_up_reference::WORDS;
    // Each program's first word after any NOP is the instruction its name
    // gives.
    for (&entry, name) in rounded_up_reference::ENTRIES
        .iter()
        .zip(rounded_up_reference::NAMES)
    {
        let at = ((entry - rounded_up_reference::BASE) / 4) as usize;
        let word = *words[at..]
            .iter()
            .find(|&&word| word != 0xd503_201f)
            .unwrap();
        let first = name.split(';').next().unwrap();
        assert_eq!(echora_lift::aarch64::disassemble(word, 0), first);
    }
    let code_size = (words.len() as u64 * 4).next_multiple_of(0x1000);
    let image: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let mut m = Machine::with_layout(
        rounded_up_reference::BASE,
        &image,
        code_size,
        DATA,
        DATA_SIZE,
    );
    let lifts: [(&str, Dispatch); 4] = [
        ("reference", rounded_up_reference::dispatch),
        ("inline", rounded_up_inline::dispatch),
        ("whole", rounded_up_whole::dispatch),
        ("runs", rounded_up_runs::dispatch),
    ];
    let mut runs = 0;
    for (&entry, name) in rounded_up_reference::ENTRIES
        .iter()
        .zip(rounded_up_reference::NAMES)
    {
        let double = name.contains(".2d") || name.contains(" d0");
        let (width, smallest) = if double {
            (64, 0x0010_0000_0000_0000u64)
        } else {
            (32, 0x0080_0000)
        };
        let scalar = !name.contains(" v0.");
        let lanes = if scalar { 1 } else { 128 / width };
        // 2^-63 * (2^-63 - 2^-87) and (2^-125 - 2^-149) / 2 are
        // 2^-126 - 2^-150, which rounds to even: 2^-126; the doubles alike.
        let (a, b) = match (name.starts_with("fdiv"), double) {
            (false, false) => (0x2000_0000, 0x1fff_ffff),
            (false, true) => (0x2000_0000_0000_0000, 0x1fff_ffff_ffff_ffff),
            (true, false) => (0x00ff_ffff, u64::from(2.0f32.to_bits())),
            (true, true) => (0x001f_ffff_ffff_ffff, 2.0f64.to_bits()),
        };
        let sign = 1u64 << (width - 1);
        let negate = name.starts_with("fnmul");
        for negative in [false, true] {
            let a = if negative { a | sign } else { a };
            // The result's sign: the product's, flipped by FNMUL.
            let result_sign = if negative != negate { sign } else { 0 };
            // Every lane the same operands (by element: any index); a zero
            // V3 (the FMADD addend, the runs' exact add) and a zero
            // accumulator V0.
            let vectors = [0, splat(a, width, lanes), splat(b, width, lanes), 0];
            for fpcr in [0, FPCR_FZ] {
                let magnitude = if fpcr & FPCR_FZ != 0 { 0 } else { smallest };
                // The runs' add of +0 makes a flushed -0 a +0.
                let result_sign = if magnitude == 0 && name.contains("fadd") {
                    0
                } else {
                    result_sign
                };
                let expected = (
                    splat(result_sign | magnitude, width, lanes),
                    FPSR_UFC | FPSR_IXC,
                );
                for (lift, dispatch) in lifts {
                    let actual = run(dispatch, &mut m, entry, vectors, fpcr);
                    runs += 1;
                    assert_eq!(
                        actual, expected,
                        "{name} ({lift} lift) fpcr {fpcr:#x} a {a:#x}: expected {expected:x?}, got {actual:x?}"
                    );
                }
            }
        }
    }
    println!("{runs} runs");
}
