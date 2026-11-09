//! Guest instructions per second of lifted code, without the game.
//!
//! Runs the kernels of `src/kernels.rs` (lifted by `build.rs` through split
//! emission, the layout real builds use) natively and reports how many
//! guest instructions each retires per second. See `docs/testing.md`.
//!
//! ```text
//! cargo bench -p echora-aot-tests --bench kernels [-- [FILTER] [--scale N] [--runs N]
//!     [--save FILE] [--compare FILE]]
//! ```
//!
//! `FILTER` keeps kernels whose name contains it; `--scale` multiplies the
//! work of each kernel (default 1: about 15-140 million instructions);
//! `--runs` sets how many times each kernel runs (default 3; the best and
//! the median are reported). `--save` writes the results to a file and
//! `--compare` reads one written earlier and prints, per kernel, the earlier
//! and current best and median rates with the change: save before a codegen
//! change, compare after it.

use std::time::{Duration, Instant};

use echora_aot_tests::kernels::{self, KERNEL_BASE, KERNEL_ENTRIES, KERNEL_SIZE};
use echora_aot_tests::{Machine, Trap, kernel_code};

const DATA_BASE: u64 = 0x20_0000;
const DATA_SIZE: u64 = 0x40_0000;
/// 64 KiB buffers for the memory kernels.
const BUFFER: u64 = 0x1_0000;
const DESTINATION: u64 = DATA_BASE;
const SOURCE: u64 = DATA_BASE + BUFFER;
const VECTOR_A: u64 = DATA_BASE + 2 * BUFFER;
const VECTOR_B: u64 = DATA_BASE + 3 * BUFFER;
const FLAG: u64 = DATA_BASE + 4 * BUFFER;
/// Guest instructions between host boundaries, as a scheduler slice.
const SLICE: u64 = 100_000;
/// Slices the polling kernel spins before the other thread sets its flag.
const POLL_SLICES: u64 = 500;

struct Options {
    filter: Option<String>,
    scale: u64,
    runs: usize,
    save: Option<String>,
    compare: Option<String>,
}

fn options() -> Options {
    let mut options = Options {
        filter: None,
        scale: 1,
        runs: 3,
        save: None,
        compare: None,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut number = |name: &str| {
            args.next()
                .and_then(|value| value.parse().ok())
                .filter(|&value: &u64| value > 0)
                .unwrap_or_else(|| panic!("{name} takes a positive number"))
        };
        match arg.as_str() {
            "--scale" => options.scale = number("--scale"),
            "--runs" => options.runs = number("--runs") as usize,
            "--save" => options.save = Some(args.next().expect("--save takes a file")),
            "--compare" => options.compare = Some(args.next().expect("--compare takes a file")),
            // Passed by `cargo bench`.
            "--bench" => {}
            other if other.starts_with('-') => panic!("unknown option {other}"),
            other => options.filter = Some(other.to_owned()),
        }
    }
    options
}

/// Sets up the registers for `name`, scaled; returns the slices after which
/// another thread ends the kernel, if it waits for one.
fn prepare(m: &mut Machine, name: &str, scale: u64) -> Option<u64> {
    let regs = &mut m.core.cpu.regs;
    match name {
        "arith" => regs[0] = 20_000_000 * scale,
        "memcpy" => {
            regs[0] = 10_000 * scale;
            regs[1] = DESTINATION;
            regs[2] = SOURCE;
            regs[3] = BUFFER;
        }
        "memset" => {
            regs[0] = 5_000 * scale;
            regs[1] = DESTINATION;
            regs[3] = BUFFER;
            regs[4] = 0x0101_0101_0101_0101;
        }
        "calls" => regs[0] = 5_000_000 * scale,
        "switch" => regs[0] = 5_000_000 * scale,
        "vector" => {
            regs[0] = 500 * scale;
            regs[1] = VECTOR_A;
            regs[2] = VECTOR_B;
            regs[3] = BUFFER / 16;
            // Sticky inexact, as in any program that has rounded once:
            // the float helpers' fast paths need it.
            m.core.cpu.fpsr |= echora_guest::simd::FPSR_IXC;
        }
        "scalar" => {
            regs[0] = 200 * scale;
            regs[1] = VECTOR_A;
            regs[2] = VECTOR_B;
            regs[3] = BUFFER / 4;
            // Sticky inexact, as for the vector kernel, and flush-to-zero,
            // which the game runs with (FPCR 0x0100_0000).
            m.core.cpu.fpsr |= echora_guest::simd::FPSR_IXC;
            m.core.cpu.fpcr = echora_guest::simd::FPCR_FZ;
        }
        "ldst" => {
            regs[0] = 1_000 * scale;
            regs[1] = VECTOR_A;
            regs[2] = VECTOR_B;
            regs[3] = BUFFER / 8;
        }
        "poll" => {
            regs[0] = FLAG;
            return Some(POLL_SLICES * scale);
        }
        other => panic!("no setup for kernel {other}"),
    }
    None
}

/// One run: guest instructions retired and the time taken.
fn run(name: &str, entry: u64, scale: u64) -> (u64, Duration) {
    let image = kernels::image();
    let mut m = Machine::with_layout(
        KERNEL_BASE,
        &image,
        KERNEL_SIZE.next_multiple_of(0x1000),
        DATA_BASE,
        DATA_SIZE,
    );
    // A float pattern for the vector kernel: 1.0 in every lane.
    let ones: Vec<u8> = std::iter::repeat_n(1.0_f32.to_le_bytes(), (2 * BUFFER / 4) as usize)
        .flatten()
        .collect();
    m.memory
        .write(VECTOR_A, &ones)
        .expect("vector inputs are mapped");
    m.core.cpu.pc = KERNEL_BASE + entry;
    let other_thread = prepare(&mut m, name, scale);
    m.core.host_check_at = SLICE;
    let mut flag_set = false;
    let start = Instant::now();
    loop {
        let trap = match kernel_code::dispatch(&mut m.core) {
            Ok(()) => continue,
            Err(trap) => trap,
        };
        // A host boundary falls between blocks: nothing to refund. (Settling
        // there would refund a self-looping block, whose next PC lies in its
        // own range.)
        if !matches!(trap, Trap::HostBoundary) {
            m.core.settle_block_trap();
        }
        match trap {
            Trap::HostBoundary => {
                // Where the host would schedule another thread; here that
                // thread only ends the polling kernel once its time comes:
                // after as many instructions as its slices would take (the
                // spin hint ends a polling slice early, so counting slices
                // would end it almost at once).
                if !flag_set && other_thread.is_some_and(|slices| m.core.steps >= slices * SLICE) {
                    m.memory.write(FLAG, &1_u32.to_le_bytes()).expect("flag");
                    flag_set = true;
                }
                m.core.spin_polls = 0;
                m.core.host_check_at = m.core.steps + SLICE;
            }
            Trap::SupervisorCall { number: 7 } => break,
            other => panic!("{name}: {other:?} at {:#x}", m.core.cpu.pc),
        }
    }
    (m.core.steps, start.elapsed())
}

/// One kernel's result: name, best and median million instructions per
/// second.
type Result = (String, f64, f64);

/// The lines `--save` writes: `name best median`.
fn read_results(path: &str) -> Vec<Result> {
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|error| panic!("reading {path}: {error}"));
    text.lines()
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            let name = fields.next()?.to_owned();
            let best = fields.next()?.parse().ok()?;
            let median = fields.next()?.parse().ok()?;
            Some((name, best, median))
        })
        .collect()
}

fn change(before: f64, after: f64) -> String {
    if before > 0.0 {
        format!("{:+.1}%", (after / before - 1.0) * 100.0)
    } else {
        "-".into()
    }
}

fn print_comparison(before: &[Result], after: &[Result]) {
    println!(
        "\n{:<8} {:>12} {:>12} {:>8} {:>14} {:>14} {:>8}",
        "kernel", "best before", "best after", "change", "median before", "median after", "change"
    );
    for (name, best, median) in after {
        let Some((_, old_best, old_median)) = before.iter().find(|(old, _, _)| old == name) else {
            continue;
        };
        println!(
            "{name:<8} {old_best:>12.1} {best:>12.1} {:>8} {old_median:>14.1} {median:>14.1} {:>8}",
            change(*old_best, *best),
            change(*old_median, *median)
        );
    }
}

fn main() {
    let options = options();
    kernel_code::install();
    println!(
        "{:<8} {:>14} {:>12} {:>14} {:>14}",
        "kernel", "instructions", "best ms", "best Minstr/s", "median Minstr/s"
    );
    let mut results: Vec<Result> = Vec::new();
    for &(name, entry) in KERNEL_ENTRIES {
        if options
            .filter
            .as_deref()
            .is_some_and(|filter| !name.contains(filter))
        {
            continue;
        }
        let mut rates = Vec::new();
        let mut best = Duration::MAX;
        let mut instructions = 0;
        for _ in 0..options.runs {
            let (steps, elapsed) = run(name, entry, options.scale);
            instructions = steps;
            best = best.min(elapsed);
            rates.push(steps as f64 / elapsed.as_secs_f64() / 1e6);
        }
        rates.sort_by(f64::total_cmp);
        let fastest = rates.last().copied().unwrap_or(0.0);
        let median = rates[rates.len() / 2];
        println!(
            "{name:<8} {instructions:>14} {:>12.1} {fastest:>14.1} {median:>14.1}",
            best.as_secs_f64() * 1e3
        );
        results.push((name.to_owned(), fastest, median));
    }
    if let Some(path) = &options.save {
        let text: String = results
            .iter()
            .map(|(name, best, median)| format!("{name} {best:.3} {median:.3}\n"))
            .collect();
        std::fs::write(path, text).unwrap_or_else(|error| panic!("writing {path}: {error}"));
    }
    if let Some(path) = &options.compare {
        print_comparison(&read_results(path), &results);
    }
}
