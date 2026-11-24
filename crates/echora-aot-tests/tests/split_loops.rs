//! Functions whose loops cross the pieces `regalloc_split_capped` cuts
//! them into (`build.rs`'s `split_loop_code`): an FP compare decides the
//! inner loop's exit, another a mid-loop exit, branches inside the body;
//! `split_big` past the block limit of one structured function, with a
//! frame, leaf calls storing and loading memory inside the loop, and a
//! caller holding a register across its calls. Each is lifted as windows,
//! as functions without pieces, and in pieces under the default cap and
//! three smaller ones (so the cuts, and the edges crossing them, fall at
//! different places): every lift ends with the same registers, vector
//! registers, flags, FP status, memory, steps and host boundaries, whole
//! and in slices of many lengths, so every piece boundary is crossed both
//! ways and entries are resumed at.
#![cfg(any(target_os = "linux", windows))]
#![allow(clippy::unwrap_used)]

use echora_aot_tests::{Core, Machine, Trap};

#[allow(
    missing_debug_implementations,
    clippy::all,
    dead_code,
    unused_imports,
    unused_parens,
    unreachable_code
)]
mod lifted {
    include!(concat!(env!("OUT_DIR"), "/lifted_split_loop.rs"));
}
use lifted::*;

const DATA: u64 = 0x80_0000;
const DATA_SIZE: u64 = 0x1000;

/// The guest state a run ends with.
#[derive(Debug, PartialEq)]
struct End {
    regs: [u64; 31],
    sp: u64,
    vectors: Vec<u128>,
    nzcv: u8,
    fpsr: u32,
    memory: u64,
    steps: u64,
    boundaries: Vec<(u64, u64)>,
}

type Lift = (&'static str, fn(), fn(&mut Core) -> Result<(), Trap>);

fn run(base: u64, words: &[u32], (_, install, dispatch): Lift, slice: Option<u64>) -> End {
    install();
    let image: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let mut m = Machine::with_layout(
        base,
        &image,
        (image.len() as u64).next_multiple_of(0x1000),
        DATA,
        DATA_SIZE,
    );
    let float = |value: f64| u128::from(value.to_bits());
    m.core.cpu.vectors[1] = float(0.01);
    m.core.cpu.vectors[3] = float(2.0);
    m.core.cpu.vectors[4] = float(0.5);
    m.core.cpu.regs[6] = 5;
    m.core.cpu.regs[10] = 3;
    m.core.cpu.regs[12] = DATA + 0x100;
    m.core.cpu.sp = DATA + 0x800;
    m.core.cpu.pc = base;
    m.core.max_steps = u64::MAX;
    m.core.host_check_at = slice.unwrap_or(u64::MAX);
    m.core.rearm();
    let mut boundaries = Vec::new();
    loop {
        match dispatch(&mut m.core) {
            Ok(()) => {}
            Err(Trap::SupervisorCall { number: 7 }) => break,
            Err(Trap::HostBoundary) => {
                boundaries.push((m.core.cpu.pc, m.core.steps));
                m.core.host_check_at = m.core.steps + slice.unwrap();
                m.core.rearm();
            }
            Err(trap) => panic!("unexpected trap {trap:?} at {:#x}", m.core.cpu.pc),
        }
        assert!(m.core.steps < 50_000_000, "no end");
    }
    m.core.settle_block_trap();
    let mut memory = [0u8; 8];
    m.memory.read(DATA + 0x100, &mut memory).unwrap();
    End {
        regs: m.core.cpu.regs,
        sp: m.core.cpu.sp,
        vectors: m.core.cpu.vectors[..8].to_vec(),
        nzcv: m.core.cpu.nzcv,
        fpsr: m.core.cpu.fpsr,
        memory: u64::from_le_bytes(memory),
        steps: m.core.steps,
        boundaries,
    }
}

/// Every lift's end against the windows', for each slice; the windows' end
/// of the first slice.
fn compare(base: u64, words: &[u32], windows: Lift, lifts: &[Lift], slices: &[Option<u64>]) -> End {
    let mut first = None;
    for &slice in slices {
        let expected = run(base, words, windows, slice);
        for &lift in lifts {
            assert_eq!(
                run(base, words, lift, slice),
                expected,
                "{}, slice {slice:?}",
                lift.0
            );
        }
        first.get_or_insert(expected);
    }
    first.unwrap()
}

macro_rules! lift {
    ($name:literal, $module:ident) => {
        (
            $name,
            $module::install as fn(),
            $module::dispatch as fn(&mut Core) -> Result<(), Trap>,
        )
    };
}

#[test]
fn loops_across_pieces_end_as_the_windows_and_the_whole_function_do() {
    const { assert!(split_loop_windows::STRUCTURED_FUNCTIONS == 0) };
    const { assert!(split_loop_whole::PIECES == 0) };
    // Each cap cuts differently (some pieces under the smallest stay
    // with the windows).
    const { assert!(split_loop_medium::PIECES > split_loop_large::PIECES) };
    const { assert!(split_loop_large::PIECES > 1) };
    let slices: Vec<Option<u64>> = std::iter::once(None).chain((1..61).map(Some)).collect();
    let end = compare(
        split_loop_windows::BASE,
        split_loop_windows::WORDS,
        lift!("windows", split_loop_windows),
        &[
            lift!("whole", split_loop_whole),
            lift!("default", split_loop_default),
            lift!("small", split_loop_small),
            lift!("medium", split_loop_medium),
            lift!("large", split_loop_large),
        ],
        &slices,
    );
    // The loops ran: the outer one to its end, the inner one's mid exit
    // taken.
    assert_eq!(end.regs[6], 0);
    assert_eq!(end.regs[2], 5);
}

#[test]
fn a_function_past_the_block_limit_runs_in_pieces_as_the_windows_run_it() {
    // Past the block limit: no structured function without pieces; in
    // pieces under every cap.
    const { assert!(split_big_windows::BLOCKS > 1024) };
    const { assert!(split_big_whole::PIECES == 0) };
    const { assert!(split_big_default::PIECES > 1) };
    const { assert!(split_big_large::PIECES > 1) };
    let slices = [
        None,
        Some(1),
        Some(2),
        Some(3),
        Some(7),
        Some(13),
        Some(29),
        Some(61),
        Some(127),
        Some(1000),
        Some(4093),
    ];
    let end = compare(
        split_big_windows::BASE,
        split_big_windows::WORDS,
        lift!("windows", split_big_windows),
        &[
            lift!("whole", split_big_whole),
            lift!("default", split_big_default),
            lift!("small", split_big_small),
            lift!("medium", split_big_medium),
            lift!("large", split_big_large),
        ],
        &slices,
    );
    // Three calls, the held register kept, the leaf called.
    assert_eq!(end.regs[9], 7 + 3 * 3);
    assert_eq!(end.regs[10], 0);
    assert!(end.regs[11] > 0);
}
