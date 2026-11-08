//! Guest synchronization under the real scheduler, at one to six workers.
//!
//! Main starts `THREADS` threads spread over the six application cores
//! with mixed priorities. Each runs `ROUNDS` rounds of:
//!
//! 1. a counter increment under a Horizon user mutex (CAS fast path,
//!    ArbitrateLock / ArbitrateUnlock when contended), with the read and the
//!    write in separate blocks and, every other round, a yield between
//!    them, so others contend for the held mutex;
//! 2. its turn in a ring on one address: wait (WaitForAddress, if equal)
//!    until the turn counter names it, advance it, wake everyone
//!    (SignalToAddress), so a lost wake-up stalls the ring;
//! 3. a barrier on a condition variable (WaitProcessWideKeyAtomic /
//!    SignalProcessWideKey) under a second mutex, checking that each wait
//!    returns with the mutex owned.
//!
//! Main waits (WaitForAddress, if less than) for all to finish. The run
//! must end in the step budget with every count exact.
//!
//! "Guest code" here is a dispatch function standing in for translated
//! code, as in `folded_cores.rs`: one block per call, thread state in guest
//! registers, shared state in guest memory through the guest's atomics.
#![cfg(any(target_os = "linux", windows))]
#![allow(clippy::unwrap_used)]

use echora_guest::memory::{
    EXECUTE, PAGE_SIZE, READ, STATE_CODE, STATE_CODE_DATA, STATE_NORMAL, STATE_TLS, WRITE,
};
use echora_guest::{Core, Trap};
use echora_runtime::runtime::{Layout, Machine};

const THREADS: u64 = 12;
const ROUNDS: u64 = 40;

const CODE: u64 = 0x7100000000;
// Main: create and start each thread, then wait for all to finish.
const MAIN_CREATE: u64 = CODE;
const MAIN_START: u64 = CODE + 0x10;
const MAIN_NEXT: u64 = CODE + 0x20;
const MAIN_WAIT: u64 = CODE + 0x30;
const MAIN_WAITED: u64 = CODE + 0x40;
const MAIN_EXIT: u64 = CODE + 0x50;
// A thread's rounds.
const ENTRY: u64 = CODE + 0x1000;
const ROUND: u64 = CODE + 0x1010;
const READ_COUNTER: u64 = CODE + 0x1020;
const WRITE_COUNTER: u64 = CODE + 0x1030;
const RING: u64 = CODE + 0x1040;
const RING_WAITED: u64 = CODE + 0x1050;
const RING_ADVANCE: u64 = CODE + 0x1060;
const RING_SIGNALED: u64 = CODE + 0x1070;
const BARRIER_IN: u64 = CODE + 0x1080;
const BARRIER_LAST: u64 = CODE + 0x1090;
const BARRIER_SIGNALED: u64 = CODE + 0x10a0;
const BARRIER_WAIT: u64 = CODE + 0x10b0;
const BARRIER_WOKEN: u64 = CODE + 0x10c0;
const BARRIER_LEAVE: u64 = CODE + 0x10d0;
const ROUND_END: u64 = CODE + 0x10e0;
const FINISH: u64 = CODE + 0x10f0;
const FINISH_SIGNALED: u64 = CODE + 0x1100;
const HOLD_YIELD: u64 = CODE + 0x1110;
const HOLD_YIELDED: u64 = CODE + 0x1120;
// Mutex subroutines: the mutex address in x25, the return PC in x26.
const LOCK: u64 = CODE + 0x1200;
const LOCK_ARBITRATE: u64 = CODE + 0x1210;
const LOCK_ARBITRATED: u64 = CODE + 0x1220;
const UNLOCK: u64 = CODE + 0x1230;
const UNLOCK_ARBITRATE: u64 = CODE + 0x1240;
const UNLOCK_ARBITRATED: u64 = CODE + 0x1250;

const DATA: u64 = 0x8000;
const SHARED: u64 = 0x10000;
const MUTEX: u64 = SHARED;
const BARRIER_MUTEX: u64 = SHARED + 0x4;
const BARRIER_KEY: u64 = SHARED + 0x8;
const COUNTER: u64 = SHARED + 0xc;
const TURN: u64 = SHARED + 0x10;
const ARRIVED: u64 = SHARED + 0x14;
const GENERATION: u64 = SHARED + 0x18;
const FINISHED: u64 = SHARED + 0x1c;
/// Condition waits that returned without the mutex.
const UNOWNED_WAKES: u64 = SHARED + 0x20;
/// Ring turns taken out of order.
const RING_ERRORS: u64 = SHARED + 0x24;
/// Each thread's handle, by index.
const HANDLES: u64 = SHARED + 0x40;
/// Each thread's own increments, by index.
const MINE: u64 = SHARED + 0x100;

const WAIT_MASK: u32 = 0x4000_0000;

// Thread registers.
const INDEX: usize = 19;
const LEFT: usize = 20;
const TEMP: usize = 21;
const TAG: usize = 22;
const SEEN: usize = 24;
const MUTEX_REG: usize = 25;
const RETURN: usize = 26;

fn machine(workers: usize) -> Machine {
    let layout = Layout {
        process_core_mask: Some(0x3f),
        process_priority_mask: Some(((1u64 << 60) - 1) & !((1u64 << 28) - 1)),
        process_ideal_core_id: 0,
        initial_used_memory: 0x4000,
        workers,
        // Short slices: more slice ends inside the sections.
        parallel_slice: 37,
        scheduler_slice: 37,
        ..Layout::default()
    };
    let mut m = Machine::new(layout, 50_000_000);
    m.memory
        .map(CODE, 2 * PAGE_SIZE, READ | EXECUTE, STATE_CODE, "code")
        .unwrap();
    m.memory
        .map(m.layout.tls_base, PAGE_SIZE, READ | WRITE, STATE_TLS, "TLS")
        .unwrap();
    m.register_main_thread_tls().unwrap();
    m.memory
        .map(
            DATA,
            PAGE_SIZE,
            READ | WRITE,
            STATE_CODE_DATA,
            "stack and data",
        )
        .unwrap();
    m.memory
        .map_stack_alias(m.layout.stack_region_base, DATA, PAGE_SIZE)
        .unwrap();
    m.memory
        .map(SHARED, PAGE_SIZE, READ | WRITE, STATE_NORMAL, "shared")
        .unwrap();
    m.core.cpu.pc = MAIN_CREATE;
    m.core.cpu.sp = m.layout.stack_region_base + 0xf00;
    m.core.cpu.tpidrro_el0 = m.layout.tls_base;
    m
}

fn svc(number: u16) -> Result<(), Trap> {
    Err(Trap::SupervisorCall { number })
}

fn cas(core: &mut Core, address: u64, expected: u32, desired: u32) -> bool {
    core.memory
        .compare_exchange_single_thread(
            address,
            4,
            u64::from(expected),
            u64::from(desired),
            true,
            true,
        )
        .unwrap()
        == u64::from(expected)
}

fn add(core: &mut Core, address: u64, addend: u32) -> u32 {
    core.memory
        .fetch_add_single_thread(address, 4, u64::from(addend), true, true)
        .unwrap() as u32
}

/// One block of the program at the core's PC.
fn block(core: &mut Core) -> Result<(), Trap> {
    let pc = core.cpu.pc;
    core.tick_block_at(pc, 1)?;
    let r = &mut core.cpu.regs;
    let next = match pc {
        MAIN_CREATE => {
            // CreateThread(entry, index, stack top, priority, core): mixed
            // priorities 40..52 over all six cores.
            let i = r[INDEX];
            r[1] = ENTRY;
            r[2] = i;
            r[3] = 0x7ff0000000 - 0x200000 + 0x100 * (i + 1);
            r[4] = 40 + (i % 4) * 4;
            r[5] = i % 6;
            return svc(0x08);
        }
        MAIN_START => {
            assert_eq!(r[0], 0, "CreateThread failed");
            let handle = r[1] as u32;
            let i = r[INDEX];
            core.memory.store_u32(HANDLES + 4 * i, handle);
            core.cpu.regs[0] = u64::from(handle);
            return svc(0x09);
        }
        MAIN_NEXT => {
            assert_eq!(r[0], 0, "StartThread failed");
            r[INDEX] += 1;
            if r[INDEX] < THREADS {
                MAIN_CREATE
            } else {
                MAIN_WAIT
            }
        }
        MAIN_WAIT => {
            if u64::from(core.memory.load_u32(FINISHED)) >= THREADS {
                MAIN_EXIT
            } else {
                // WaitForAddress(FINISHED, WaitIfLessThan, THREADS, forever).
                let r = &mut core.cpu.regs;
                r[0] = FINISHED;
                r[1] = 0;
                r[2] = THREADS;
                r[3] = u64::MAX;
                return svc(0x34);
            }
        }
        MAIN_WAITED => MAIN_WAIT,
        MAIN_EXIT => return svc(0x07),

        ENTRY => {
            r[INDEX] = r[0];
            r[LEFT] = ROUNDS;
            r[TAG] = u64::from(core.memory.load_u32(HANDLES + 4 * r[0]));
            ROUND
        }
        ROUND => {
            r[MUTEX_REG] = MUTEX;
            r[RETURN] = READ_COUNTER;
            LOCK
        }
        READ_COUNTER => {
            core.cpu.regs[TEMP] = u64::from(core.memory.load_u32(COUNTER));
            // Every other round yields between the read and the write, so
            // others contend for the held mutex.
            if core.cpu.regs[LEFT].is_multiple_of(2) {
                HOLD_YIELD
            } else {
                WRITE_COUNTER
            }
        }
        HOLD_YIELD => {
            // SleepThread(0): a yield. TEMP survives in the saved context.
            r[0] = 0;
            return svc(0x0b);
        }
        HOLD_YIELDED => WRITE_COUNTER,
        WRITE_COUNTER => {
            // A plain read-modify-write: only the mutex makes it exact.
            let value = core.cpu.regs[TEMP] as u32 + 1;
            core.memory.store_u32(COUNTER, value);
            let mine = MINE + 4 * core.cpu.regs[INDEX];
            let value = core.memory.load_u32(mine) + 1;
            core.memory.store_u32(mine, value);
            let r = &mut core.cpu.regs;
            r[MUTEX_REG] = MUTEX;
            r[RETURN] = RING;
            UNLOCK
        }
        RING => {
            let turn = core.memory.load_u32(TURN);
            let r = &mut core.cpu.regs;
            if u64::from(turn) % THREADS == r[INDEX] {
                r[SEEN] = u64::from(turn);
                RING_ADVANCE
            } else {
                // WaitForAddress(TURN, WaitIfEqual, turn, forever).
                r[0] = TURN;
                r[1] = 2;
                r[2] = u64::from(turn);
                r[3] = u64::MAX;
                return svc(0x34);
            }
        }
        RING_WAITED => RING,
        RING_ADVANCE => {
            let seen = core.cpu.regs[SEEN] as u32;
            // Only this thread may advance the turn now.
            if !cas(core, TURN, seen, seen + 1) {
                add(core, RING_ERRORS, 1);
            }
            // SignalToAddress(TURN, Signal, 0, all).
            let r = &mut core.cpu.regs;
            r[0] = TURN;
            r[1] = 0;
            r[2] = 0;
            r[3] = u64::from(u32::MAX);
            return svc(0x35);
        }
        RING_SIGNALED => {
            r[MUTEX_REG] = BARRIER_MUTEX;
            r[RETURN] = BARRIER_IN;
            LOCK
        }
        BARRIER_IN => {
            let arrived = core.memory.load_u32(ARRIVED) + 1;
            if u64::from(arrived) == THREADS {
                core.memory.store_u32(ARRIVED, 0);
                let generation = core.memory.load_u32(GENERATION) + 1;
                core.memory.store_u32(GENERATION, generation);
                BARRIER_LAST
            } else {
                core.memory.store_u32(ARRIVED, arrived);
                core.cpu.regs[SEEN] = u64::from(core.memory.load_u32(GENERATION));
                BARRIER_WAIT
            }
        }
        BARRIER_LAST => {
            // SignalProcessWideKey(key, all).
            r[0] = BARRIER_KEY;
            r[1] = u64::from(u32::MAX);
            return svc(0x1d);
        }
        BARRIER_SIGNALED => BARRIER_LEAVE,
        BARRIER_WAIT => {
            if u64::from(core.memory.load_u32(GENERATION)) != core.cpu.regs[SEEN] {
                BARRIER_LEAVE
            } else {
                // WaitProcessWideKeyAtomic(mutex, key, tag, forever).
                let r = &mut core.cpu.regs;
                r[0] = BARRIER_MUTEX;
                r[1] = BARRIER_KEY;
                r[2] = r[TAG];
                r[3] = u64::MAX;
                return svc(0x1c);
            }
        }
        BARRIER_WOKEN => {
            // The wait returns with the mutex reacquired.
            let owner = core.memory.load_u32(BARRIER_MUTEX) & !WAIT_MASK;
            if u64::from(owner) != core.cpu.regs[TAG] {
                add(core, UNOWNED_WAKES, 1);
            }
            BARRIER_WAIT
        }
        BARRIER_LEAVE => {
            r[MUTEX_REG] = BARRIER_MUTEX;
            r[RETURN] = ROUND_END;
            UNLOCK
        }
        ROUND_END => {
            r[LEFT] -= 1;
            if r[LEFT] > 0 { ROUND } else { FINISH }
        }
        FINISH => {
            add(core, FINISHED, 1);
            let r = &mut core.cpu.regs;
            r[0] = FINISHED;
            r[1] = 0;
            r[2] = 0;
            r[3] = u64::from(u32::MAX);
            return svc(0x35);
        }
        FINISH_SIGNALED => return svc(0x0a),

        LOCK => {
            let (address, tag) = (r[MUTEX_REG], r[TAG] as u32);
            let word = core.memory.load_u32(address);
            if word == 0 {
                if cas(core, address, 0, tag) {
                    core.cpu.regs[RETURN]
                } else {
                    LOCK
                }
            } else if word & WAIT_MASK == 0 {
                if cas(core, address, word, word | WAIT_MASK) {
                    core.cpu.regs[TEMP] = u64::from(word);
                    LOCK_ARBITRATE
                } else {
                    LOCK
                }
            } else {
                core.cpu.regs[TEMP] = u64::from(word);
                LOCK_ARBITRATE
            }
        }
        LOCK_ARBITRATE => {
            // ArbitrateLock(owner, address, tag).
            r[0] = r[TEMP] & u64::from(!WAIT_MASK);
            r[1] = r[MUTEX_REG];
            r[2] = r[TAG];
            return svc(0x1a);
        }
        LOCK_ARBITRATED => {
            let owner = core.memory.load_u32(core.cpu.regs[MUTEX_REG]) & !WAIT_MASK;
            if u64::from(owner) == core.cpu.regs[TAG] {
                core.cpu.regs[RETURN]
            } else {
                LOCK
            }
        }
        UNLOCK => {
            let (address, tag) = (r[MUTEX_REG], r[TAG] as u32);
            if cas(core, address, tag, 0) {
                core.cpu.regs[RETURN]
            } else {
                UNLOCK_ARBITRATE
            }
        }
        UNLOCK_ARBITRATE => {
            r[0] = r[MUTEX_REG];
            return svc(0x1b);
        }
        UNLOCK_ARBITRATED => r[RETURN],
        other => return Err(Trap::MissingBlock { pc: other }),
    };
    core.cpu.pc = next;
    core.finish_block()
}

/// SVCs return to the next "instruction": the runtime adds 4 to the PC,
/// and each SVC block is followed by its continuation 0x10 on.
fn dispatch(core: &mut Core) -> Result<(), Trap> {
    let pc = core.cpu.pc;
    if pc % 0x10 == 4 {
        core.cpu.pc = pc + 0xc;
    }
    block(core)
}

fn run(workers: usize) {
    let mut m = machine(workers);
    let trap = m.run_generated(dispatch).unwrap_err();
    assert!(
        matches!(trap, Trap::Exit { .. }),
        "{workers} workers: {trap:?}"
    );
    let read = |address: u64| u64::from(m.memory.read_u32(address).unwrap());
    let total = THREADS * ROUNDS;
    assert_eq!(read(COUNTER), total, "{workers} workers: mutex counter");
    for i in 0..THREADS {
        assert_eq!(read(MINE + 4 * i), ROUNDS, "{workers} workers: thread {i}");
    }
    assert_eq!(read(TURN), total, "{workers} workers: ring turns");
    assert_eq!(read(RING_ERRORS), 0, "{workers} workers: ring order");
    assert_eq!(read(GENERATION), ROUNDS, "{workers} workers: barriers");
    assert_eq!(read(UNOWNED_WAKES), 0, "{workers} workers: condvar mutex");
    assert_eq!(read(MUTEX), 0, "{workers} workers: mutex left held");
    assert_eq!(read(BARRIER_MUTEX), 0, "{workers} workers: barrier mutex");
}

#[test]
fn mutex_ring_and_barrier_at_one_to_six_workers() {
    for workers in 1..=6 {
        run(workers);
    }
}
