//! The multi-core cost of one global lock, without the game.
//!
//! The runtime's workers run guest code unlocked and take one machine mutex
//! for host work (docs/multicore-review.md, part 3). This models that. W
//! workers, each on its own host thread with its own machine, run the
//! arith kernel in slices of `--slice` guest instructions. After each
//! slice a worker does `--hold` microseconds of host work, split by kind
//! with the game's measured shares (main session, W=6, 90 s: 78.3 s held,
//! 62% NVN host calls, 1.3% slice ends, the rest SVCs and IPC):
//!
//! - NVN work (`--nvn`, default 0.62), under the global lock, under an NVN
//!   lock of its own, or under none (`--nvn-lock global|own|none`;
//!   dea1c7f moved NVN to its own lock), or with recording unlocked and
//!   only the `--submit` share of it (default 0.2) under the NVN lock
//!   (`--nvn-lock recording`);
//! - slice-end scheduling (`--sched`, default 0.013), under the global lock
//!   except on a `--fast` share of slice ends (part 3's lock-free path);
//! - everything else, under the global lock.
//!
//! For each worker count it reports the aggregate guest rate, the speedup
//! over one worker, and the shares of time running guest code, waiting for
//! a lock and holding the global one.
//!
//! ```text
//! cargo bench -p echora-aot-tests --bench lock [-- [--workers 1,2,4,6]
//!     [--slice N] [--hold US] [--nvn F] [--nvn-lock global|own|recording|none] [--submit F]
//!     [--sched F] [--fast F] [--millis MS]]
//! ```
//!
//! The defaults (50,000-instruction slices, 10 us of host work after each)
//! put about 40% of a single worker's time in host work, as in the game.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use echora_aot_tests::kernels::{self, KERNEL_BASE, KERNEL_ENTRIES, KERNEL_SIZE};
use echora_aot_tests::{Machine, Trap, kernel_code};

const DATA_BASE: u64 = 0x20_0000;

#[derive(Clone, Copy, PartialEq, Eq)]
enum NvnLock {
    Global,
    Own,
    /// Recording without a lock (each worker records into its own command
    /// buffers); only the submit share takes the NVN lock.
    Recording,
    /// The planned NVN split: recording unlocked (per command buffer), the
    /// `--exec` share of NVN work (execution and the GPU backend) serial:
    /// under one execution lock, or with `--exec-async` on one execution
    /// thread that workers hand it to through a short queue.
    Split,
    None,
}

struct Options {
    workers: Vec<usize>,
    slice: u64,
    hold: Duration,
    nvn: f64,
    nvn_lock: NvnLock,
    /// With `NvnLock::Recording`: the share of NVN work that is submission.
    submit: f64,
    /// With `NvnLock::Split`: the share of NVN work that is execution.
    exec: f64,
    /// With `NvnLock::Split`: execution on its own thread, not the worker's.
    exec_async: bool,
    /// SVC and IPC work off the global lock (`--svc-lock none`).
    svc_unlocked: bool,
    sched: f64,
    fast: f64,
    run_for: Duration,
}

fn options() -> Options {
    let mut options = Options {
        workers: vec![1, 2, 3, 4, 6],
        slice: 50_000,
        hold: Duration::from_micros(10),
        nvn: 0.62,
        nvn_lock: NvnLock::Global,
        submit: 0.2,
        exec: 0.8,
        exec_async: false,
        svc_unlocked: false,
        sched: 0.013,
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
            "--nvn" => options.nvn = value().parse().expect("share 0..1"),
            "--sched" => options.sched = value().parse().expect("share 0..1"),
            "--submit" => options.submit = value().parse().expect("share 0..1"),
            "--exec" => options.exec = value().parse().expect("share 0..1"),
            "--exec-async" => options.exec_async = true,
            "--svc-lock" => {
                options.svc_unlocked = match value().as_str() {
                    "global" => false,
                    "none" => true,
                    other => panic!("--svc-lock takes global or none, not {other}"),
                }
            }
            "--nvn-lock" => {
                options.nvn_lock = match value().as_str() {
                    "global" => NvnLock::Global,
                    "own" => NvnLock::Own,
                    "recording" => NvnLock::Recording,
                    "split" => NvnLock::Split,
                    "none" => NvnLock::None,
                    other => panic!("--nvn-lock takes global, own, recording or none, not {other}"),
                }
            }
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

fn worker(
    index: usize,
    options: &Options,
    lock: &Mutex<()>,
    nvn_lock: &Mutex<()>,
    executor: &std::sync::mpsc::SyncSender<Duration>,
    start: Instant,
) -> Totals {
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
        let part = |share: f64| options.hold.mul_f64(share);
        let nvn = part(options.nvn);
        let sched = part(options.sched);
        let other = options.hold.saturating_sub(nvn + sched);
        // Work under `lock`: the wait and the hold are counted; `global`
        // says whether the hold counts as holding the global lock.
        let mut queue_wait = Duration::ZERO;
        let mut locked = |lock: &Mutex<()>, work: Duration, global: bool| {
            let waited = Instant::now();
            let guard = lock
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let held = Instant::now();
            totals.waiting += held - waited;
            spin_for(work);
            drop(guard);
            if global {
                totals.holding += held.elapsed();
            }
        };
        fast_credit += options.fast;
        let sched_locked = if fast_credit >= 1.0 {
            fast_credit -= 1.0;
            false
        } else {
            true
        };
        // The global lock's work in one hold, as the runtime's boundary.
        let mut global = if options.svc_unlocked {
            spin_for(other);
            Duration::ZERO
        } else {
            other
        };
        if sched_locked {
            global += sched;
        } else {
            spin_for(sched);
        }
        match options.nvn_lock {
            NvnLock::Global => global += nvn,
            NvnLock::Own => locked(nvn_lock, nvn, false),
            NvnLock::Recording => {
                let submit = nvn.mul_f64(options.submit);
                spin_for(nvn.saturating_sub(submit));
                locked(nvn_lock, submit, false);
            }
            NvnLock::Split => {
                let execution = nvn.mul_f64(options.exec);
                spin_for(nvn.saturating_sub(execution));
                if options.exec_async {
                    // Waits only while the execution queue is full.
                    let waited = Instant::now();
                    let _ = executor.send(execution);
                    queue_wait += waited.elapsed();
                } else {
                    locked(nvn_lock, execution, false);
                }
            }
            NvnLock::None => spin_for(nvn),
        }
        if !global.is_zero() {
            locked(lock, global, true);
        }
        totals.waiting += queue_wait;
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
        "slice {} instructions, {:?} of host work after each: NVN {:.0}% ({}), scheduling {:.1}% \
         ({:.0}% lock-free), the rest under the global lock",
        options.slice,
        options.hold,
        options.nvn * 100.0,
        match options.nvn_lock {
            NvnLock::Global => "global lock",
            NvnLock::Own => "own lock",
            NvnLock::Recording => "recording unlocked, submit under its lock",
            NvnLock::Split if options.exec_async =>
                "split: recording unlocked, execution on one thread",
            NvnLock::Split => "split: recording unlocked, execution under its lock",
            NvnLock::None => "unlocked",
        },
        options.sched * 100.0,
        options.fast * 100.0
    );
    println!(
        "{:>7} {:>12} {:>8} {:>9} {:>9} {:>9}",
        "workers", "Minstr/s", "speedup", "running", "waiting", "holding"
    );
    let mut single = None;
    for &workers in &options.workers {
        let lock = Arc::new(Mutex::new(()));
        let nvn_lock = Arc::new(Mutex::new(()));
        // The execution thread of `--exec-async`: runs handed-over work.
        let (executor, work) = std::sync::mpsc::sync_channel::<Duration>(4);
        let execution = std::thread::spawn(move || {
            while let Ok(duration) = work.recv() {
                spin_for(duration);
            }
        });
        let start = Instant::now();
        let totals: Vec<Totals> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..workers)
                .map(|index| {
                    let (options, lock, nvn_lock) = (&options, &lock, &nvn_lock);
                    let executor = executor.clone();
                    scope.spawn(move || worker(index, options, lock, nvn_lock, &executor, start))
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("lock benchmark worker"))
                .collect()
        });
        let wall = start.elapsed().as_secs_f64();
        drop(executor);
        let _ = execution.join();
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
