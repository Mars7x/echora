//! Wait timeouts under the real scheduler, at one to six workers.
//!
//! For each wait kind (WaitForAddress, WaitProcessWideKeyAtomic and
//! WaitSynchronization) one thread, A, times out and then another, B,
//! waits on the same thing and is woken. A must time out with TimedOut, and
//! must have left the queue: a later signal for one waiter finds only B.
//! - The address signal (SignalAndModifyByWaitingCountIfEqual) shows how
//!   many waited: one, not two.
//! - The condition variable's key word is clear once its one waiter is
//!   signaled, and a timed-out waiter returns without the mutex (Horizon's
//!   user library takes it again itself).
//! - WaitSynchronization on a thread's handle times out with index -1; when
//!   the thread exits B wakes with index 0, and nothing writes A's
//!   registers again.
//!
//! A second program races signals against timeouts: a waiter waits with a
//! short timeout, round after round, while a signaler on another core
//! signals it after a varying delay. Each round must have exactly one
//! outcome.
//! - On an address, the signal reports whether it woke the waiter: it did
//!   exactly when the wait returned 0, never also when it timed out.
//! - On a condition variable (the signaler holding the mutex for a varying
//!   while after signaling), a wait returns 0 owning the mutex or TimedOut
//!   without it, nothing else.
//!
//! As in `sync_stress.rs`, "guest code" is a dispatch function, one block
//! per call, with thread state in guest registers and shared state in
//! guest memory.
#![cfg(any(target_os = "linux", windows))]
#![allow(clippy::unwrap_used)]

use echora_guest::memory::{
    EXECUTE, PAGE_SIZE, READ, STATE_CODE, STATE_CODE_DATA, STATE_NORMAL, STATE_TLS, WRITE,
};
use echora_guest::{Core, Trap};
use echora_runtime::runtime::{Layout, Machine};

const CODE: u64 = 0x7100000000;
const MAIN_CREATE: u64 = CODE;
const MAIN_SLEEP: u64 = CODE + 0x30;
const MAIN_EXIT: u64 = CODE + 0x50;
const ENTRY: u64 = CODE + 0x100;
/// The thread A and B wait on with WaitSynchronization: it exits when told.
const T: u64 = CODE + 0x200;
const A: u64 = CODE + 0x300;
const B: u64 = CODE + 0x400;
/// The signaler, which runs the phases in turn.
const S: u64 = CODE + 0x500;
// The race program: a waiter and a signaler on an address, and on a
// condition variable.
const ADDRESS_WAITER: u64 = CODE + 0x800;
const ADDRESS_SIGNALER: u64 = CODE + 0x900;
const CONDITION_WAITER: u64 = CODE + 0xa00;
const CONDITION_SIGNALER: u64 = CODE + 0xb00;
const LOCK: u64 = CODE + 0x700;
const UNLOCK: u64 = CODE + 0x730;

const QUEUE_ROLES: [u64; 4] = [T, A, B, S];
const RACE_ROLES: [u64; 4] = [
    ADDRESS_WAITER,
    ADDRESS_SIGNALER,
    CONDITION_WAITER,
    CONDITION_SIGNALER,
];
/// Which program the threads run: false for the queue program.
static RACE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// The tests share `RACE`: one at a time.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn roles() -> [u64; 4] {
    if RACE.load(std::sync::atomic::Ordering::Relaxed) {
        RACE_ROLES
    } else {
        QUEUE_ROLES
    }
}

/// Rounds of each race.
const ROUNDS: u64 = 120;
/// The waiter's timeout in round `r`: 0.2 to 3.7 ms. Sleeping workers
/// wake in a millisecond or two, so deadlines this long fall both before
/// and after the signal.
fn race_timeout(r: u64) -> u64 {
    200_000 + (r % 8) * 500_000
}
/// The signaler's delay before signaling in round `r`: 0 to 3.6 ms.
fn race_delay(r: u64) -> u64 {
    (r * 5 % 9) * 450_000
}
/// Each pair's handshake words: the waiter is ready / done with a round,
/// the signaler is done with it, the waiter's result.
const ADDRESS_PAIR: u64 = SHARED + 0x300;
const CONDITION_PAIR: u64 = SHARED + 0x340;
const READY: u64 = 0;
const WAITED: u64 = 4;
const SIGNALED: u64 = 8;
const RESULT: u64 = 0xc;
/// The race's address.
const RACE_ADDRESS: u64 = SHARED + 0x380;
/// Address rounds by (signal outcome: none, woke, more) x (wait result:
/// 0, TimedOut, InvalidState, other).
const ADDRESS_OUTCOMES: u64 = SHARED + 0x400;
/// Condition rounds by wait outcome: 0 owning the mutex, 0 without it,
/// TimedOut without it, TimedOut owning it, other.
const CONDITION_OUTCOMES: u64 = SHARED + 0x440;

const DATA: u64 = 0x8000;
const SHARED: u64 = 0x10000;
const ADDRESS: u64 = SHARED;
const MUTEX: u64 = SHARED + 0x4;
const KEY: u64 = SHARED + 0x8;
const HANDLE_LIST: u64 = SHARED + 0xc;
const ALL_DONE: u64 = SHARED + 0x10;
const T_EXIT: u64 = SHARED + 0x14;
const A_DONE1: u64 = SHARED + 0x18;
const GO_B1: u64 = SHARED + 0x1c;
const B_DONE1: u64 = SHARED + 0x20;
const GO_A2: u64 = SHARED + 0x24;
const A_DONE2: u64 = SHARED + 0x28;
const GO_B2: u64 = SHARED + 0x2c;
const B_READY2: u64 = SHARED + 0x30;
const B_DONE2: u64 = SHARED + 0x34;
const GO_A3: u64 = SHARED + 0x38;
const A_DONE3: u64 = SHARED + 0x3c;
const GO_B3: u64 = SHARED + 0x40;
const B_DONE3: u64 = SHARED + 0x44;
/// Results, checked after the run.
const A_RES1: u64 = SHARED + 0x80;
const B_RES1: u64 = SHARED + 0x84;
const WAITERS_SEEN1: u64 = SHARED + 0x88;
const A_RES2: u64 = SHARED + 0x8c;
const A_OWNED2: u64 = SHARED + 0x90;
const B_RES2: u64 = SHARED + 0x94;
const B_OWNED2: u64 = SHARED + 0x98;
const KEY_AFTER2: u64 = SHARED + 0x9c;
const A_RES3: u64 = SHARED + 0xa0;
const A_INDEX3: u64 = SHARED + 0xa4;
const B_RES3: u64 = SHARED + 0xa8;
const B_INDEX3: u64 = SHARED + 0xac;
const A_CLOBBERED: u64 = SHARED + 0xb0;
const HANDLES: u64 = SHARED + 0x100;

const WAIT_MASK: u32 = 0x4000_0000;
const TIMED_OUT: u64 = 0xea01;
const TIMEOUT_NS: u64 = 1_000_000;
const FOREVER: u64 = u64::MAX;
const CANARY: [u64; 2] = [0xa5a5_0000_1111, 0x5a5a_0000_2222];

const INDEX: usize = 19;
const TAG: usize = 22;
const SEEN: usize = 24;
const ROUND: usize = 20;
const TEMP: usize = 21;
const MUTEX_REG: usize = 25;
const RETURN: usize = 26;

fn machine(workers: usize) -> Machine {
    let layout = Layout {
        process_core_mask: Some(0x3f),
        process_priority_mask: Some(((1u64 << 60) - 1) & !((1u64 << 28) - 1)),
        process_ideal_core_id: 0,
        initial_used_memory: 0x4000,
        workers,
        parallel_slice: 37,
        scheduler_slice: 37,
        ..Layout::default()
    };
    let mut m = Machine::new(layout, 50_000_000);
    m.memory
        .map(CODE, PAGE_SIZE, READ | EXECUTE, STATE_CODE, "code")
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

fn load(core: &Core, address: u64) -> u64 {
    u64::from(core.memory.load_u32(address))
}

fn store(core: &mut Core, address: u64, value: u64) {
    core.memory.store_u32(address, value as u32);
}

/// Go on to `next` once `flag` is set; until then spin here (with the hint
/// a polling loop gets).
fn after(core: &mut Core, flag: u64, next: u64) -> u64 {
    if load(core, flag) != 0 {
        next
    } else {
        core.spin_hint();
        core.cpu.pc
    }
}

/// Go on to `next` once the word at `address` has reached this thread's
/// round + 1; until then spin here.
fn until(core: &mut Core, address: u64, next: u64) -> u64 {
    if load(core, address) > core.cpu.regs[ROUND] {
        next
    } else {
        core.spin_hint();
        core.cpu.pc
    }
}

fn add(core: &mut Core, address: u64, addend: u32) {
    core.memory
        .fetch_add_single_thread(address, 4, u64::from(addend), true, true)
        .unwrap();
}

/// Take the mutex, then go on at `next`.
fn lock(core: &mut Core, next: u64) -> u64 {
    core.cpu.regs[MUTEX_REG] = MUTEX;
    core.cpu.regs[RETURN] = next;
    LOCK
}

/// Release the mutex, then go on at `next`.
fn unlock(core: &mut Core, next: u64) -> u64 {
    core.cpu.regs[MUTEX_REG] = MUTEX;
    core.cpu.regs[RETURN] = next;
    UNLOCK
}

fn owns_mutex(core: &Core) -> bool {
    load(core, MUTEX) & u64::from(!WAIT_MASK) == core.cpu.regs[TAG]
}

/// Load the registers of an SVC and make it.
fn call(core: &mut Core, number: u16, args: &[u64]) -> Result<(), Trap> {
    core.cpu.regs[..args.len()].copy_from_slice(args);
    svc(number)
}

/// One block of the program at the core's PC.
fn block(core: &mut Core) -> Result<(), Trap> {
    let pc = core.cpu.pc;
    core.tick_block_at(pc, 1)?;
    let next = match pc {
        // Main: create and start the four threads, then sleep until done.
        MAIN_CREATE => {
            let i = core.cpu.regs[INDEX];
            let priority = if i == 0 { 50 } else { 44 };
            let stack = 0x7ff0000000 - 0x200000 + 0x100 * (i + 1);
            return call(core, 0x08, &[0, ENTRY, i, stack, priority, i % 6]);
        }
        p if p == MAIN_CREATE + 0x10 => {
            assert_eq!(core.cpu.regs[0], 0, "CreateThread failed");
            let handle = core.cpu.regs[1];
            let i = core.cpu.regs[INDEX];
            store(core, HANDLES + 4 * i, handle);
            return call(core, 0x09, &[handle]);
        }
        p if p == MAIN_CREATE + 0x20 => {
            assert_eq!(core.cpu.regs[0], 0, "StartThread failed");
            core.cpu.regs[INDEX] += 1;
            if core.cpu.regs[INDEX] < roles().len() as u64 {
                MAIN_CREATE
            } else {
                MAIN_SLEEP
            }
        }
        MAIN_SLEEP => {
            // The race ends when both signalers have finished.
            let needed = if RACE.load(std::sync::atomic::Ordering::Relaxed) {
                2
            } else {
                1
            };
            if load(core, ALL_DONE) >= needed {
                MAIN_EXIT
            } else {
                return call(core, 0x0b, &[TIMEOUT_NS]);
            }
        }
        p if p == MAIN_SLEEP + 0x10 => MAIN_SLEEP,
        MAIN_EXIT => return svc(0x07),

        ENTRY => {
            let i = core.cpu.regs[0];
            core.cpu.regs[INDEX] = i;
            core.cpu.regs[TAG] = load(core, HANDLES + 4 * i);
            roles()[i as usize]
        }

        // T: exit when told.
        T => after(core, T_EXIT, T + 0x10),
        p if p == T + 0x10 => return svc(0x0a),

        // A, phase 1: time out on the address.
        A => return call(core, 0x34, &[ADDRESS, 2, 0, TIMEOUT_NS]),
        p if p == A + 0x10 => {
            let result = core.cpu.regs[0];
            store(core, A_RES1, result);
            store(core, A_DONE1, 1);
            A + 0x20
        }
        // Phase 2: time out on the condition variable.
        p if p == A + 0x20 => after(core, GO_A2, A + 0x30),
        p if p == A + 0x30 => lock(core, A + 0x40),
        p if p == A + 0x40 => {
            let tag = core.cpu.regs[TAG];
            return call(core, 0x1c, &[MUTEX, KEY, tag, TIMEOUT_NS]);
        }
        p if p == A + 0x50 => {
            let result = core.cpu.regs[0];
            store(core, A_RES2, result);
            let owned = owns_mutex(core);
            store(core, A_OWNED2, u64::from(owned));
            if owned {
                A + 0x60
            } else {
                lock(core, A + 0x60)
            }
        }
        p if p == A + 0x60 => unlock(core, A + 0x70),
        p if p == A + 0x70 => {
            store(core, A_DONE2, 1);
            A + 0x80
        }
        // Phase 3: time out waiting for T.
        p if p == A + 0x80 => after(core, GO_A3, A + 0x90),
        p if p == A + 0x90 => {
            let t = load(core, HANDLES);
            store(core, HANDLE_LIST, t);
            return call(core, 0x18, &[0, HANDLE_LIST, 1, TIMEOUT_NS]);
        }
        p if p == A + 0xa0 => {
            let (result, index) = (core.cpu.regs[0], core.cpu.regs[1]);
            store(core, A_RES3, result);
            store(core, A_INDEX3, index);
            store(core, A_DONE3, 1);
            core.cpu.regs[..2].copy_from_slice(&CANARY);
            A + 0xb0
        }
        // Then nothing may write A's registers again.
        p if p == A + 0xb0 => {
            if core.cpu.regs[..2] != CANARY {
                store(core, A_CLOBBERED, 1);
                core.cpu.regs[..2].copy_from_slice(&CANARY);
            }
            after(core, ALL_DONE, A + 0xc0)
        }
        p if p == A + 0xc0 => return svc(0x0a),

        // B, phase 1: wait on the address until woken.
        B => after(core, GO_B1, B + 0x10),
        p if p == B + 0x10 => {
            let seen = load(core, ADDRESS);
            core.cpu.regs[SEEN] = seen;
            return call(core, 0x34, &[ADDRESS, 2, seen, FOREVER]);
        }
        p if p == B + 0x20 => {
            // InvalidState: the address moved on before the wait; again.
            let result = core.cpu.regs[0];
            if result == 0xfa01 {
                B + 0x10
            } else {
                store(core, B_RES1, result);
                store(core, B_DONE1, 1);
                B + 0x30
            }
        }
        // Phase 2: wait on the condition variable until signaled.
        p if p == B + 0x30 => after(core, GO_B2, B + 0x40),
        p if p == B + 0x40 => lock(core, B + 0x50),
        p if p == B + 0x50 => {
            store(core, B_READY2, 1);
            let tag = core.cpu.regs[TAG];
            return call(core, 0x1c, &[MUTEX, KEY, tag, FOREVER]);
        }
        p if p == B + 0x60 => {
            let result = core.cpu.regs[0];
            store(core, B_RES2, result);
            let owned = owns_mutex(core);
            store(core, B_OWNED2, u64::from(owned));
            if owned {
                B + 0x70
            } else {
                lock(core, B + 0x70)
            }
        }
        p if p == B + 0x70 => unlock(core, B + 0x80),
        p if p == B + 0x80 => {
            store(core, B_DONE2, 1);
            B + 0x90
        }
        // Phase 3: wait for T to exit.
        p if p == B + 0x90 => after(core, GO_B3, B + 0xa0),
        p if p == B + 0xa0 => return call(core, 0x18, &[0, HANDLE_LIST, 1, FOREVER]),
        p if p == B + 0xb0 => {
            let (result, index) = (core.cpu.regs[0], core.cpu.regs[1]);
            store(core, B_RES3, result);
            store(core, B_INDEX3, index);
            store(core, B_DONE3, 1);
            B + 0xc0
        }
        p if p == B + 0xc0 => after(core, ALL_DONE, B + 0xd0),
        p if p == B + 0xd0 => return svc(0x0a),

        // S, phase 1: once A has timed out, signal one waiter, B, with the
        // signal that counts the waiters.
        S => after(core, A_DONE1, S + 0x10),
        p if p == S + 0x10 => {
            store(core, GO_B1, 1);
            S + 0x20
        }
        p if p == S + 0x20 => {
            let seen = load(core, ADDRESS);
            core.cpu.regs[SEEN] = seen;
            return call(core, 0x35, &[ADDRESS, 2, seen, 1]);
        }
        p if p == S + 0x30 => {
            // No waiters: value + 1; one, all woken: value - 1; more: value.
            let (seen, now) = (core.cpu.regs[SEEN] as u32, load(core, ADDRESS) as u32);
            if now == seen.wrapping_add(1) {
                // B is not waiting yet: give it a moment, then again.
                return call(core, 0x0b, &[100_000]);
            }
            let waiters = if now == seen.wrapping_sub(1) { 1 } else { 2 };
            store(core, WAITERS_SEEN1, waiters);
            S + 0x50
        }
        p if p == S + 0x40 => S + 0x20,
        p if p == S + 0x50 => after(core, B_DONE1, S + 0x60),
        // Phase 2: once A has timed out, signal B's condition wait while
        // holding the mutex (taken once B's wait released it).
        p if p == S + 0x60 => {
            store(core, GO_A2, 1);
            S + 0x70
        }
        p if p == S + 0x70 => after(core, A_DONE2, S + 0x80),
        p if p == S + 0x80 => {
            store(core, GO_B2, 1);
            S + 0x90
        }
        p if p == S + 0x90 => after(core, B_READY2, S + 0xa0),
        p if p == S + 0xa0 => lock(core, S + 0xb0),
        p if p == S + 0xb0 => return call(core, 0x1d, &[KEY, 1]),
        p if p == S + 0xc0 => {
            let key = load(core, KEY);
            store(core, KEY_AFTER2, key);
            unlock(core, S + 0xd0)
        }
        p if p == S + 0xd0 => after(core, B_DONE2, S + 0xe0),
        // Phase 3: once A has timed out, let B wait, then end T.
        p if p == S + 0xe0 => {
            store(core, GO_A3, 1);
            S + 0xf0
        }
        p if p == S + 0xf0 => after(core, A_DONE3, S + 0x100),
        p if p == S + 0x100 => {
            store(core, GO_B3, 1);
            return call(core, 0x0b, &[2 * TIMEOUT_NS]);
        }
        p if p == S + 0x110 => {
            store(core, T_EXIT, 1);
            S + 0x120
        }
        p if p == S + 0x120 => after(core, B_DONE3, S + 0x130),
        p if p == S + 0x130 => {
            store(core, ALL_DONE, 1);
            return svc(0x0a);
        }

        // The address race: wait with a short timeout, round after round.
        ADDRESS_WAITER => ADDRESS_WAITER + 0x10,
        p if p == ADDRESS_WAITER + 0x10 => {
            let round = core.cpu.regs[ROUND];
            if round == ROUNDS {
                return svc(0x0a);
            }
            let seen = load(core, RACE_ADDRESS);
            store(core, ADDRESS_PAIR + READY, round + 1);
            return call(core, 0x34, &[RACE_ADDRESS, 2, seen, race_timeout(round)]);
        }
        p if p == ADDRESS_WAITER + 0x20 => {
            let (result, round) = (core.cpu.regs[0], core.cpu.regs[ROUND]);
            store(core, ADDRESS_PAIR + RESULT, result);
            store(core, ADDRESS_PAIR + WAITED, round + 1);
            ADDRESS_WAITER + 0x30
        }
        p if p == ADDRESS_WAITER + 0x30 => {
            until(core, ADDRESS_PAIR + SIGNALED, ADDRESS_WAITER + 0x40)
        }
        p if p == ADDRESS_WAITER + 0x40 => {
            core.cpu.regs[ROUND] += 1;
            ADDRESS_WAITER + 0x10
        }

        // Signal after a delay, with the signal that reports the waiters.
        ADDRESS_SIGNALER => {
            if core.cpu.regs[ROUND] == ROUNDS {
                add(core, ALL_DONE, 1);
                return svc(0x0a);
            }
            until(core, ADDRESS_PAIR + READY, ADDRESS_SIGNALER + 0x10)
        }
        p if p == ADDRESS_SIGNALER + 0x10 => {
            let delay = race_delay(core.cpu.regs[ROUND]);
            return call(core, 0x0b, &[delay]);
        }
        p if p == ADDRESS_SIGNALER + 0x20 => {
            let seen = load(core, RACE_ADDRESS);
            core.cpu.regs[SEEN] = seen;
            return call(core, 0x35, &[RACE_ADDRESS, 2, seen, 1]);
        }
        p if p == ADDRESS_SIGNALER + 0x30 => {
            let (seen, now) = (core.cpu.regs[SEEN] as u32, load(core, RACE_ADDRESS) as u32);
            core.cpu.regs[TEMP] = if now == seen.wrapping_add(1) {
                0
            } else if now == seen.wrapping_sub(1) {
                1
            } else {
                2
            };
            ADDRESS_SIGNALER + 0x40
        }
        p if p == ADDRESS_SIGNALER + 0x40 => {
            until(core, ADDRESS_PAIR + WAITED, ADDRESS_SIGNALER + 0x50)
        }
        p if p == ADDRESS_SIGNALER + 0x50 => {
            let result = match load(core, ADDRESS_PAIR + RESULT) {
                0 => 0,
                TIMED_OUT => 1,
                0xfa01 => 2,
                _ => 3,
            };
            let outcome = core.cpu.regs[TEMP] * 4 + result;
            add(core, ADDRESS_OUTCOMES + 4 * outcome, 1);
            let round = core.cpu.regs[ROUND];
            store(core, ADDRESS_PAIR + SIGNALED, round + 1);
            core.cpu.regs[ROUND] += 1;
            ADDRESS_SIGNALER
        }

        // The condition race: wait with a short timeout under the mutex.
        CONDITION_WAITER => CONDITION_WAITER + 0x10,
        p if p == CONDITION_WAITER + 0x10 => {
            if core.cpu.regs[ROUND] == ROUNDS {
                return svc(0x0a);
            }
            lock(core, CONDITION_WAITER + 0x20)
        }
        p if p == CONDITION_WAITER + 0x20 => {
            let round = core.cpu.regs[ROUND];
            store(core, CONDITION_PAIR + READY, round + 1);
            let tag = core.cpu.regs[TAG];
            return call(core, 0x1c, &[MUTEX, KEY, tag, race_timeout(round)]);
        }
        p if p == CONDITION_WAITER + 0x30 => {
            let (result, owned) = (core.cpu.regs[0], owns_mutex(core));
            let outcome = match (result, owned) {
                (0, true) => 0,
                (0, false) => 1,
                (TIMED_OUT, false) => 2,
                (TIMED_OUT, true) => 3,
                _ => 4,
            };
            store(core, CONDITION_PAIR + RESULT, outcome);
            if owned {
                CONDITION_WAITER + 0x40
            } else {
                lock(core, CONDITION_WAITER + 0x40)
            }
        }
        p if p == CONDITION_WAITER + 0x40 => unlock(core, CONDITION_WAITER + 0x50),
        p if p == CONDITION_WAITER + 0x50 => {
            let round = core.cpu.regs[ROUND];
            store(core, CONDITION_PAIR + WAITED, round + 1);
            CONDITION_WAITER + 0x60
        }
        p if p == CONDITION_WAITER + 0x60 => {
            until(core, CONDITION_PAIR + SIGNALED, CONDITION_WAITER + 0x70)
        }
        p if p == CONDITION_WAITER + 0x70 => {
            core.cpu.regs[ROUND] += 1;
            CONDITION_WAITER + 0x10
        }

        // Signal after a delay under the mutex, then hold it a while: the
        // waiter's deadline may pass while it waits for the mutex.
        CONDITION_SIGNALER => {
            if core.cpu.regs[ROUND] == ROUNDS {
                add(core, ALL_DONE, 1);
                return svc(0x0a);
            }
            until(core, CONDITION_PAIR + READY, CONDITION_SIGNALER + 0x10)
        }
        p if p == CONDITION_SIGNALER + 0x10 => {
            let delay = race_delay(core.cpu.regs[ROUND]);
            return call(core, 0x0b, &[delay]);
        }
        p if p == CONDITION_SIGNALER + 0x20 => lock(core, CONDITION_SIGNALER + 0x30),
        p if p == CONDITION_SIGNALER + 0x30 => return call(core, 0x1d, &[KEY, 1]),
        p if p == CONDITION_SIGNALER + 0x40 => {
            core.cpu.regs[TEMP] = core.cpu.regs[ROUND] % 3 * 200 + 1;
            CONDITION_SIGNALER + 0x50
        }
        p if p == CONDITION_SIGNALER + 0x50 => {
            core.cpu.regs[TEMP] -= 1;
            if core.cpu.regs[TEMP] != 0 {
                CONDITION_SIGNALER + 0x50
            } else {
                unlock(core, CONDITION_SIGNALER + 0x60)
            }
        }
        p if p == CONDITION_SIGNALER + 0x60 => {
            until(core, CONDITION_PAIR + WAITED, CONDITION_SIGNALER + 0x70)
        }
        p if p == CONDITION_SIGNALER + 0x70 => {
            let outcome = load(core, CONDITION_PAIR + RESULT);
            add(core, CONDITION_OUTCOMES + 4 * outcome, 1);
            let round = core.cpu.regs[ROUND];
            store(core, CONDITION_PAIR + SIGNALED, round + 1);
            core.cpu.regs[ROUND] += 1;
            CONDITION_SIGNALER
        }

        // The user mutex: its address in x25, the return PC in x26.
        LOCK => {
            let (address, tag) = (core.cpu.regs[MUTEX_REG], core.cpu.regs[TAG] as u32);
            let word = core.memory.load_u32(address);
            if word == 0 {
                if cas(core, address, 0, tag) {
                    core.cpu.regs[RETURN]
                } else {
                    LOCK
                }
            } else if word & WAIT_MASK == 0 && !cas(core, address, word, word | WAIT_MASK) {
                LOCK
            } else {
                let owner = u64::from(word & !WAIT_MASK);
                let address = core.cpu.regs[MUTEX_REG];
                return call(core, 0x1a, &[owner, address, u64::from(tag)]);
            }
        }
        p if p == LOCK + 0x10 => {
            if owns_mutex(core) {
                core.cpu.regs[RETURN]
            } else {
                LOCK
            }
        }
        UNLOCK => {
            let (address, tag) = (core.cpu.regs[MUTEX_REG], core.cpu.regs[TAG] as u32);
            if cas(core, address, tag, 0) {
                core.cpu.regs[RETURN]
            } else {
                let address = core.cpu.regs[MUTEX_REG];
                return call(core, 0x1b, &[address]);
            }
        }
        p if p == UNLOCK + 0x10 => core.cpu.regs[RETURN],
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
    let read = |address: u64| u64::from(m.memory.read_u32(address).unwrap());
    let minus_one = u64::from(u32::MAX);
    let checks = [
        ("address: A's result", read(A_RES1), TIMED_OUT),
        ("address: B's result", read(B_RES1), 0),
        ("address: waiters B's signal found", read(WAITERS_SEEN1), 1),
        ("condition: A's result", read(A_RES2), TIMED_OUT),
        ("condition: A returned owning the mutex", read(A_OWNED2), 0),
        ("condition: B's result", read(B_RES2), 0),
        ("condition: B returned owning the mutex", read(B_OWNED2), 1),
        ("condition: key word after the signal", read(KEY_AFTER2), 0),
        ("handles: A's result", read(A_RES3), TIMED_OUT),
        ("handles: A's index", read(A_INDEX3), minus_one),
        ("handles: B's result", read(B_RES3), 0),
        ("handles: B's index", read(B_INDEX3), 0),
        ("A's registers written after its wait", read(A_CLOBBERED), 0),
        ("mutex left held", read(MUTEX), 0),
    ];
    let mut failures: Vec<String> = checks
        .into_iter()
        .filter(|&(_, seen, expected)| seen != expected)
        .map(|(what, seen, expected)| format!("{what}: {seen:#x}, not {expected:#x}"))
        .collect();
    if !matches!(trap, Trap::Exit { .. }) {
        failures.insert(0, format!("ended with {trap:?}"));
    }
    assert!(
        failures.is_empty(),
        "{workers} workers: {}",
        failures.join("; ")
    );
}

fn serial(race: bool) -> std::sync::MutexGuard<'static, ()> {
    let guard = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    RACE.store(race, std::sync::atomic::Ordering::Relaxed);
    guard
}

#[test]
fn timed_out_waiters_leave_their_queues() {
    let _serial = serial(false);
    for workers in 1..=6 {
        run(workers);
    }
}

fn race(workers: usize) {
    let mut m = machine(workers);
    let trap = m.run_generated(dispatch).unwrap_err();
    let read = |address: u64| u64::from(m.memory.read_u32(address).unwrap());
    let address: Vec<u64> = (0..12).map(|i| read(ADDRESS_OUTCOMES + 4 * i)).collect();
    let condition: Vec<u64> = (0..5).map(|i| read(CONDITION_OUTCOMES + 4 * i)).collect();
    let mut failures = Vec::new();
    if !matches!(trap, Trap::Exit { .. }) {
        failures.push(format!("ended with {trap:?}"));
    }
    // Consistent address rounds: woke and 0; no waiter and TimedOut (the
    // deadline came first) or InvalidState (the signal came first).
    let consistent = address[4] + address[1] + address[2];
    if consistent != ROUNDS || address.iter().sum::<u64>() != ROUNDS {
        failures.push(format!(
            "address rounds (none, woke, more) x (0, TimedOut, InvalidState, other): {address:?}"
        ));
    }
    if condition[0] + condition[2] != ROUNDS || condition.iter().sum::<u64>() != ROUNDS {
        failures.push(format!(
            "condition rounds (0 owning, 0 not owning, TimedOut not owning, TimedOut \
             owning, other): {condition:?}"
        ));
    }
    assert!(
        failures.is_empty(),
        "{workers} workers: {}",
        failures.join("; ")
    );
    eprintln!(
        "{workers} workers: address woke {} / timed out {} / missed {}; condition woke {} / \
         timed out {}",
        address[4], address[1], address[2], condition[0], condition[2]
    );
}

#[test]
fn signals_racing_timeouts_have_one_outcome() {
    let _serial = serial(true);
    for workers in 1..=6 {
        race(workers);
    }
}
