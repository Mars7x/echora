//! LDP/STP of Q registers with the body of `Cpu::load_q_pair`/
//! `store_q_pair` written out (`EmitOptions::direct_q_pairs`) against the
//! out-of-line calls (`build.rs`, `q_pair_code`): every
//! addressing form at offsets across the immediate's range, an ordinary
//! base or SP, and a store followed by an overlapping load. (LDP with both
//! destinations the same register does not decode; a pair's registers and
//! its base cannot alias.)
//!
//! - Inside mapped memory (a pair straddling a page boundary among them):
//!   registers, SP, the data and the trap must agree.
//! - Past the end of the guest address space (straddling it, at it, far
//!   past it, wrapping): the range test of `GuestView::read`/`write`
//!   fails, so both return its memory fault with nothing moved and no
//!   writeback.
//! - Inside the address space but on memory not mapped (straddling the
//!   data's end, or a page past it): the host faults, which ends the
//!   process either way; each such run is a child process of this test,
//!   whose fault message and signal must agree.
#![cfg(target_os = "linux")]
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
    include!(concat!(env!("OUT_DIR"), "/lifted_q_pair.rs"));
}
use lifted::{q_pair_code, q_pair_reference};

const DATA: u64 = 0x1000_0000;
const DATA_SIZE: u64 = 0x4000;
const CHILD: &str = "ECHORA_Q_PAIR_CHILD";

type Dispatch = fn(&mut echora_guest::Core) -> Result<(), Trap>;

#[derive(Debug, PartialEq)]
struct Outcome {
    vectors: [u128; 32],
    regs: [u64; 31],
    sp: u64,
    data: Vec<u8>,
    trap: String,
}

fn machine() -> Machine {
    let words = q_pair_code::WORDS;
    let code_size = (words.len() as u64 * 4).next_multiple_of(0x1000);
    let image: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    Machine::with_layout(q_pair_code::BASE, &image, code_size, DATA, DATA_SIZE)
}

/// Runs the program at `entry` with every general register and SP at
/// `base` (SP rounded down to 16), vectors and data of a fixed pattern.
fn run(dispatch: Dispatch, m: &mut Machine, entry: u64, base: u64) -> Outcome {
    let data: Vec<u8> = (0..DATA_SIZE).map(|index| (index * 29 + 7) as u8).collect();
    m.memory.write(DATA, &data).unwrap();
    m.core.cpu.vectors = std::array::from_fn(|v| {
        u128::from(v as u64 * 0x0101_0101_0101_0101) << 64 | u128::from(!(v as u64) << 8 | 0x5a)
    });
    m.core.cpu.regs = [base; 31];
    m.core.cpu.sp = base & !15;
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
    let mut data = vec![0u8; DATA_SIZE as usize];
    m.memory.read(DATA, &mut data).unwrap();
    Outcome {
        vectors: m.core.cpu.vectors,
        regs: m.core.cpu.regs,
        sp: m.core.cpu.sp,
        data,
        trap: format!("{trap:?} at {:#x}", m.core.cpu.pc),
    }
}

#[test]
fn the_option_leaves_no_out_of_line_pair_call() {
    const { assert!(q_pair_code::OUT_OF_LINE == 0) };
    const {
        assert!(
            q_pair_reference::OUT_OF_LINE == q_pair_code::WORDS.len() - q_pair_code::ENTRIES.len()
        )
    };
}

#[test]
fn pairs_inside_mapped_memory_match_the_checked_calls() {
    let mut m = machine();
    // Every offset (-1024 to 1008) from these bases stays inside the data;
    // DATA + 0x1ff8 puts an offset-0 pair across the page boundary at
    // DATA + 0x2000, the others straddle at other offsets.
    let bases = [
        DATA + 0x1800,
        DATA + 0x1808,
        DATA + 0x1ff8,
        DATA + 0x2004,
        DATA + 0x2ff0,
    ];
    for (index, &entry) in q_pair_code::ENTRIES.iter().enumerate() {
        for base in bases {
            let expected = run(q_pair_reference::dispatch, &mut m, entry, base);
            let actual = run(q_pair_code::dispatch, &mut m, entry, base);
            assert_eq!(
                actual,
                expected,
                "{} from base {base:#x}",
                q_pair_code::NAMES[index]
            );
        }
    }
}

#[test]
fn pairs_past_the_address_space_trap_as_the_checked_calls_do() {
    let mut m = machine();
    let end = m.memory.address_space_end();
    for (index, &entry) in q_pair_code::ENTRIES.iter().enumerate() {
        let name = q_pair_code::NAMES[index];
        // Only the plain-register forms at offset 0 (SP is rounded, and an
        // offset could bring the address back in range).
        if !name.contains(" 0") || name.contains("x31") || name.contains(';') {
            continue;
        }
        for base in [end - 16, end - 1, end, end + 0x1000, u64::MAX - 8] {
            let expected = run(q_pair_reference::dispatch, &mut m, entry, base);
            let actual = run(q_pair_code::dispatch, &mut m, entry, base);
            assert!(
                expected.trap.starts_with("MemoryFault"),
                "{name} at {base:#x}: {}",
                expected.trap
            );
            assert_eq!(actual, expected, "{name} from base {base:#x}");
        }
    }
}

/// In a child process (`CHILD` set): runs one program with one lift from
/// one base, which faults in the host and ends the process. Otherwise
/// runs each such case in a child, both lifts, and compares their fault
/// messages and signals.
#[test]
fn pairs_on_unmapped_memory_end_the_process_alike() {
    use std::os::unix::process::ExitStatusExt;

    if let Ok(case) = std::env::var(CHILD) {
        let mut parts = case.split(',');
        let lift = parts.next().unwrap();
        let index: usize = parts.next().unwrap().parse().unwrap();
        let base = u64::from_str_radix(parts.next().unwrap(), 16).unwrap();
        let dispatch: Dispatch = if lift == "code" {
            q_pair_code::dispatch
        } else {
            q_pair_reference::dispatch
        };
        let mut m = machine();
        let outcome = run(dispatch, &mut m, q_pair_code::ENTRIES[index], base);
        // Not reached when the access faults.
        println!("no fault: {}", outcome.trap);
        return;
    }
    let exe = std::env::current_exe().unwrap();
    let mut cases = 0;
    for (index, name) in q_pair_code::NAMES.iter().enumerate() {
        // The plain-register forms at offset 0 and 1.
        if !(name.contains(" 0") || name.contains(" 1"))
            || name.contains("x31")
            || name.contains(';')
        {
            continue;
        }
        // A pair straddling the data's end, and one a page past it.
        for base in [DATA + DATA_SIZE - 16, DATA + DATA_SIZE + 0x1000] {
            let outcome = |lift: &str| {
                let output = std::process::Command::new(&exe)
                    .args([
                        "--exact",
                        "pairs_on_unmapped_memory_end_the_process_alike",
                        "--nocapture",
                        "--test-threads=1",
                    ])
                    .env(CHILD, format!("{lift},{index},{base:x}"))
                    .output()
                    .unwrap();
                let message = String::from_utf8_lossy(&output.stderr)
                    .lines()
                    .find(|line| line.starts_with("guest memory fault"))
                    .map(str::to_owned);
                (
                    output.status.signal(),
                    message,
                    String::from_utf8_lossy(&output.stdout).contains("no fault"),
                )
            };
            let expected = outcome("reference");
            let actual = outcome("code");
            assert!(
                expected.0.is_some() && expected.1.is_some() && !expected.2,
                "{name} from {base:#x}: {expected:?}"
            );
            assert_eq!(actual, expected, "{name} from {base:#x}");
            cases += 1;
        }
    }
    assert!(cases >= 12, "{cases}");
}
