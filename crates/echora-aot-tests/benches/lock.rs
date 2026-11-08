//! The multi-core cost of one global lock, without the game.
//!
//! The runtime's workers run guest code unlocked and take one machine mutex
//! for all host work: slice ends, SVCs, NVN calls
//! (docs/multicore-review.md, part 3). This models that. W workers, each
//! on its own host thread with its own machine, run the arith kernel in
//! slices of `--slice` guest instructions. At each slice end a worker takes
//! one shared mutex and does `--hold` microseconds of host work under it,
//! except on a `--fast` share of slice ends, which skip the lock, as part
//! 3's lock-free slice-end path would. For each worker count it reports the
//! aggregate guest rate, the speedup over one worker, and the shares of
//! wall time spent running guest code, waiting for the lock and holding it.
//!
//! ```text
//! cargo bench -p echora-aot-tests --bench lock [-- [--workers 1,2,4,6]
//!     [--slice N] [--hold US] [--fast F] [--millis MS]]
//! ```
//!
//! The defaults (50,000-instruction slices, 10 us of host work per slice
//! end) put about 40% of a single worker's time under the lock, as in the
//! game at six workers. The lock then saturates once W times the hold share
//! reaches 1: on four host CPUs the model levels off at about 1.5x from
//! three workers on, as the game's intro does (1.4-1.5x at six). With 80%
//! of slice ends lock-free (`--fast 0.8`), four workers reach about 3.5x.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use echora_aot_tests::kernels::{self, KERNEL_BASE, KERNEL_ENTRIES, KERNEL_SIZE};
use echora_aot_tests::{Machine, Trap, kernel_code};

const DATA_BASE: u64 = 0x20_0000;

struct Options {
    workers: Vec<usize>,
    slice: u64,
    hold: Duration,
    fast: f64,
    run_for: Duration,
}

fn options() -> Options {
    let mut options = Options {
        workers: vec![1, 2, 3, 4, 6],
        slice: 50_000,
        hold: Duration::from_micros(10),
        fast: 0.0,
        run_for: Duration::from_millis(1500),
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| panic!("{arg} takes a value"));
        match arg.as_str() {
            "--workers" => {
                options.workers = value()
                    .split(',')
                    .map(|n| n.trim().parse().expect("worker count"))
                    .collect();
            }
            "--slice" => options.slice = value().parse().expect("instructions"),
            "--hold" => {
                options.hold = Duration::from_micros(value().parse().expect("microseconds"))
            }
            "--fast" => options.fast = value().parse().expect("share 0..1"),
            "--millis" => options.run_for = Duration::from_millis(value().parse().expect("ms")),
            "--bench" => {}
            other => panic!("unknown option {other}"),
        }
    }
    options
}

/// One worker's totals.
#[derive(Default)]
struct Totals {
    instructions: u64,
    running: Duration,
    waiting: Duration,
    holding: Duration,
}

fn spin_for(duration: Duration) {
    let until = Instant::now() + duration;
    while Instant::now() < until {
        std::hint::spin_loop();
    }
}

fn worker(index: usize, options: &Options, lock: &Mutex<()>, start: Instant) -> Totals {
    let image = kernels::image();
    let mut m = Machine::with_layout(
        KERNEL_BASE,
        &image,
        KERNEL_SIZE.next_multiple_of(0x1000),
        DATA_BASE,
        0x1_0000,
    );
    let entry = KERNEL_ENTRIES
        .iter()
        .find(|(name, _)| *name == "arith")
        .expect("the arith kernel")
        .1;
    m.core.cpu.pc = KERNEL_BASE + entry;
    // Enough iterations to outlast the run.
    m.core.cpu.regs[0] = u64::MAX / 8;
    m.core.max_steps = u64::MAX;
    m.core.host_check_at = options.slice;
    let mut totals = Totals::default();
    // Which slice ends skip the lock: a fixed pattern per worker.
    let mut fast_credit = index as f64 * 0.37 % 1.0;
    while start.elapsed() < options.run_for {
        let ran = Instant::now();
        let trap = loop {
            if let Err(trap) = kernel_code::dispatch(&mut m.core) {
                break trap;
            }
        };
        totals.running += ran.elapsed();
        assert!(matches!(trap, Trap::HostBoundary), "{trap:?}");
        fast_credit += options.fast;
        if fast_credit >= 1.0 {
            fast_credit -= 1.0;
        } else {
            let waited = Instant::now();
            let guard = lock
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let held = Instant::now();
            totals.waiting += held - waited;
            spin_for(options.hold);
            drop(guard);
            totals.holding += held.elapsed();
        }
        m.core.spin_polls = 0;
        m.core.host_check_at = m.core.steps + options.slice;
    }
    totals.instructions = m.core.steps;
    totals
}

fn main() {
    let options = options();
    kernel_code::install();
    println!(
        "slice {} instructions, {:?} under the lock per slice end, {:.0}% of slice ends lock-free",
        options.slice,
        options.hold,
        options.fast * 100.0
    );
    println!(
        "{:>7} {:>12} {:>8} {:>9} {:>9} {:>9}",
        "workers", "Minstr/s", "speedup", "running", "waiting", "holding"
    );
    let mut single = None;
    for &workers in &options.workers {
        let lock = Arc::new(Mutex::new(()));
        let start = Instant::now();
        let totals: Vec<Totals> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..workers)
                .map(|index| {
                    let (options, lock) = (&options, &lock);
                    scope.spawn(move || worker(index, options, lock, start))
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("lock benchmark worker"))
                .collect()
        });
        let wall = start.elapsed().as_secs_f64();
        let instructions: u64 = totals.iter().map(|t| t.instructions).sum();
        let rate = instructions as f64 / wall / 1e6;
        let share = |field: fn(&Totals) -> Duration| {
            totals.iter().map(|t| field(t).as_secs_f64()).sum::<f64>() / (wall * workers as f64)
                * 100.0
        };
        let base = *single.get_or_insert(rate / workers as f64);
        println!(
            "{workers:>7} {rate:>12.0} {:>7.2}x {:>8.0}% {:>8.0}% {:>8.0}%",
            rate / base,
            share(|t| t.running),
            share(|t| t.waiting),
            share(|t| t.holding),
        );
    }
}
