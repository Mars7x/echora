//! Concurrency stress: the patterns a job system uses (`src/stress.rs`), lifted
//! and run on two to four Cores that share one Memory, each on its own
//! host thread. Each program checks an invariant: no item claimed twice
//! or lost, no torn value inside a lock or a seqlock read, every message
//! seen whole.
//!
//! Every pattern also runs preempted: host boundaries end each Core's
//! slices at random points, and at each one the Core's exclusive monitor
//! is cleared and its host thread yields, as a context switch would.
//!
//! `ECHORA_STRESS_SCALE=N` multiplies the work (default 1: tens of
//! thousands of iterations per thread, which suits debug builds; use a
//! release build and a large scale for millions).
#![cfg(any(target_os = "linux", windows))]
#![allow(clippy::unwrap_used)]

use std::sync::mpsc;
use std::time::Duration;

use echora_aot_tests::stress::{STRESS_BASE, STRESS_ENTRIES, stress_words};
use echora_aot_tests::{Core, Machine, Trap, stress_code};

const DATA: u64 = 0x400_0000;
const DATA_SIZE: u64 = 0x1000_0000;
/// Where the claim programs count each item.
const CLAIMS: u64 = DATA + 0x1000;
/// A stalled run fails after this long.
const TIMEOUT: Duration = Duration::from_secs(120);

fn scale() -> u64 {
    std::env::var("ECHORA_STRESS_SCALE")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1)
}

fn entry(name: &str) -> u64 {
    STRESS_BASE + STRESS_ENTRIES.iter().find(|(n, _)| *n == name).unwrap().1
}

/// xorshift, for slice lengths.
struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Runs `program` on `threads` Cores, each on its own host thread, with
/// x3 = `count`. With `preempt`, slices end every 1 to 64 instructions
/// (rounded to blocks), and each end clears the monitor and yields.
/// Returns each thread's x10 and x11 and the machine, for its memory.
fn run(program: &str, threads: u64, count: u64, preempt: bool) -> (Vec<(u64, u64)>, Machine) {
    run_after(program, threads, count, preempt, |_| {})
}

/// As `run`, after `setup` has prepared the data.
fn run_after(
    program: &str,
    threads: u64,
    count: u64,
    preempt: bool,
    setup: impl FnOnce(&mut Machine),
) -> (Vec<(u64, u64)>, Machine) {
    let image: Vec<u8> = stress_words()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    let mut m = Machine::with_layout(STRESS_BASE, &image, image.len() as u64, DATA, DATA_SIZE);
    setup(&mut m);
    stress_code::install();
    let pc = entry(program);
    let (done, results) = mpsc::channel();
    let mut cores: Vec<Core> = (0..threads).map(|_| Core::new(m.memory.view())).collect();
    for (index, core) in cores.iter_mut().enumerate() {
        core.cpu.regs[0] = DATA;
        core.cpu.regs[1] = index as u64;
        core.cpu.regs[2] = threads;
        core.cpu.regs[3] = count;
        core.cpu.sp = DATA + DATA_SIZE - 0x1000 * (index as u64 + 1);
        core.cpu.pc = pc;
        core.max_steps = u64::MAX;
    }
    for (index, mut core) in cores.into_iter().enumerate() {
        let done = done.clone();
        std::thread::spawn(move || {
            let mut random = Random(0x9e37_79b9_7f4a_7c15 ^ ((index as u64 + 1) * 0x2545_f491));
            let slice = |random: &mut Random, steps: u64| {
                if preempt {
                    steps + 1 + random.next() % 64
                } else {
                    u64::MAX
                }
            };
            core.host_check_at = slice(&mut random, 0);
            let result = loop {
                match stress_code::dispatch(&mut core) {
                    Ok(()) => {}
                    Err(Trap::HostBoundary) => {
                        if preempt {
                            core.memory.clear_exclusive();
                            std::thread::yield_now();
                        }
                        core.spin_polls = 0;
                        core.host_check_at = slice(&mut random, core.steps);
                    }
                    Err(Trap::SupervisorCall { number: 7 }) => {
                        core.settle_block_trap();
                        break Ok((core.cpu.regs[10], core.cpu.regs[11]));
                    }
                    Err(trap) => break Err(format!("{trap:?} at {:#x}", core.cpu.pc)),
                }
            };
            let _ = done.send((index, result));
        });
    }
    let mut out = vec![(0, 0); threads as usize];
    for _ in 0..threads {
        match results.recv_timeout(TIMEOUT) {
            Ok((index, Ok(registers))) => out[index] = registers,
            Ok((index, Err(trap))) => panic!("{program}: thread {index} trapped: {trap}"),
            Err(_) => {
                // The stalled threads still spin: end the process.
                eprintln!("{program} ({threads} threads, preempt {preempt}): stalled");
                std::process::exit(1);
            }
        }
    }
    (out, m)
}

fn read_u64(m: &Machine, address: u64) -> u64 {
    let mut bytes = [0; 8];
    m.memory.read(address, &mut bytes).unwrap();
    u64::from_le_bytes(bytes)
}

fn check_claims(program: &str, threads: u64, preempt: bool) {
    let items = 100_000 * scale();
    assert!(CLAIMS + items * 4 < DATA + DATA_SIZE - 0x10000);
    let (results, m) = run(program, threads, items, preempt);
    let claimed: u64 = results.iter().map(|(claims, _)| claims).sum();
    let mut counts = vec![0u8; (items * 4) as usize];
    m.memory.read(CLAIMS, &mut counts).unwrap();
    let wrong: Vec<(u64, u32)> = counts
        .chunks(4)
        .enumerate()
        .map(|(i, c)| (i as u64, u32::from_le_bytes(c.try_into().unwrap())))
        .filter(|(_, count)| *count != 1)
        .take(10)
        .collect();
    assert!(
        claimed == items && wrong.is_empty(),
        "{program} ({threads} threads, preempt {preempt}): {claimed} of {items} claimed, \
         items not claimed once (index, count): {wrong:?}"
    );
}

fn check_ticket(threads: u64, preempt: bool) {
    let iterations = 20_000 * scale();
    let (results, m) = run("ticket", threads, iterations, preempt);
    let torn: u64 = results.iter().map(|(torn, _)| torn).sum();
    let (a, b) = (read_u64(&m, DATA + 0x80), read_u64(&m, DATA + 0x88));
    assert!(
        torn == 0 && a == threads * iterations && b == a,
        "ticket ({threads} threads, preempt {preempt}): pair ({a}, {b}), expected {}, \
         {torn} unequal pairs seen inside the lock",
        threads * iterations
    );
}

fn check_seqlock(threads: u64, preempt: bool) {
    let writes = 100_000 * scale();
    let (results, _) = run("seqlock", threads, writes, preempt);
    let torn: u64 = results.iter().skip(1).map(|(torn, _)| torn).sum();
    let reads: u64 = results.iter().skip(1).map(|(_, reads)| reads).sum();
    assert!(
        torn == 0 && reads > 0,
        "seqlock ({threads} threads, preempt {preempt}): {torn} torn of {reads} reads"
    );
}

fn check_passing(program: &str, preempt: bool) {
    let messages = 20_000 * scale();
    let (results, m) = run(program, 2, messages, preempt);
    let (wrong, _) = results[1];
    let acknowledged = read_u64(&m, DATA + 0x80);
    assert!(
        wrong == 0 && acknowledged == messages,
        "{program} (preempt {preempt}): {wrong} of {messages} messages seen wrong, \
         {acknowledged} acknowledged"
    );
}

/// Nodes of the CASP stack: few, so that threads contend for them.
const STACK_NODES: u64 = 16;

fn node(index: u64) -> u64 {
    DATA + 0x1000 + index * 64
}

fn write_u64(m: &mut Machine, address: u64, value: u64) {
    m.memory.write(address, &value.to_le_bytes()).unwrap();
}

fn check_stack(threads: u64, preempt: bool) {
    let rounds = 20_000 * scale();
    let (results, m) = run_after("casp_stack", threads, rounds, preempt, |m| {
        for index in 0..STACK_NODES {
            let next = if index + 1 < STACK_NODES {
                node(index + 1)
            } else {
                0
            };
            write_u64(m, node(index), next);
        }
        write_u64(m, DATA, node(0));
    });
    let busy: u64 = results.iter().map(|(busy, _)| busy).sum();
    // Every node on the stack exactly once, none still marked.
    let mut seen = Vec::new();
    let mut at = read_u64(&m, DATA);
    while at != 0 && seen.len() <= STACK_NODES as usize {
        seen.push(at);
        at = read_u64(&m, at);
    }
    let mut sorted = seen.clone();
    sorted.sort_unstable();
    sorted.dedup();
    let expected: Vec<u64> = (0..STACK_NODES).map(node).collect();
    let marked: Vec<u64> = expected
        .iter()
        .filter(|&&node| read_u64(&m, node + 8) != 0)
        .copied()
        .collect();
    assert!(
        busy == 0 && sorted == expected && seen.len() == expected.len() && marked.is_empty(),
        "casp_stack ({threads} threads, preempt {preempt}): {busy} pops of a busy node, \
         stack {seen:#x?}, still marked {marked:#x?}"
    );
}

fn check_halves(program: &str, threads: u64, preempt: bool) {
    let writes = 200_000 * scale();
    let (results, _) = run(program, threads, writes, preempt);
    let torn: u64 = results.iter().skip(1).map(|(torn, _)| torn).sum();
    let reads: u64 = results.iter().skip(1).map(|(_, reads)| reads).sum();
    assert!(
        torn == 0 && reads > 0,
        "{program} ({threads} threads, preempt {preempt}): {torn} torn 64-bit halves in {reads} reads"
    );
}

macro_rules! both {
    ($name:ident, $check:expr) => {
        mod $name {
            #[test]
            fn running() {
                let check: fn(bool) = $check;
                check(false);
            }
            #[test]
            fn preempted() {
                let check: fn(bool) = $check;
                check(true);
            }
        }
    };
}

both!(claims_by_cas_on_four_cores, |p| super::check_claims(
    "claim_cas",
    4,
    p
));
both!(claims_by_exclusives_on_four_cores, |p| super::check_claims(
    "claim_llsc",
    4,
    p
));
both!(claims_by_ldadd_on_four_cores, |p| super::check_claims(
    "claim_ldadd",
    4,
    p
));
both!(claims_by_cas_on_two_cores, |p| super::check_claims(
    "claim_cas",
    2,
    p
));
both!(claims_by_exclusives_on_two_cores, |p| super::check_claims(
    "claim_llsc",
    2,
    p
));
both!(ticket_lock_on_three_cores, |p| super::check_ticket(3, p));
both!(ticket_lock_on_two_cores, |p| super::check_ticket(2, p));
both!(seqlock_with_three_readers, |p| super::check_seqlock(4, p));
both!(messages_by_release_and_acquire, |p| super::check_passing(
    "pass_acquire",
    p
));
both!(messages_by_dmb, |p| super::check_passing("pass_dmb", p));
both!(casp_stack_on_four_cores, |p| super::check_stack(4, p));
both!(casp_stack_on_two_cores, |p| super::check_stack(2, p));
both!(pair_halves_with_three_readers, |p| super::check_halves(
    "pair_halves",
    4,
    p
));
both!(vector_halves_with_three_readers, |p| super::check_halves(
    "vector_halves",
    4,
    p
));
