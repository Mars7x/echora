//! Faults and traps of guest memory accesses, against a model of what each
//! must do.
//!
//! A plain load or store goes straight to the host guest view. Its only
//! test is the address space's range test, an address outside which
//! panics. A page the guest may not access faults in the host; the fault
//! handler names the guest address and the process ends. Exclusive and
//! ordered accesses check natural alignment, LSE atomics that they stay in
//! one 16-byte block, and SP-based accesses that SP is 16-byte aligned,
//! each a trap.
//!
//! `build.rs` (`memory_fault_code`) lifts each plain load and store form
//! (every size, sign extension, writeback, S/D/Q, pairs), the exclusive,
//! ordered, atomic and SP-based forms, and an exclusive pair, with the
//! build's options. The data's last page is read-only. Each form runs from
//! aligned, misaligned, 16-byte-crossing and page-crossing addresses, from
//! read-only, unmapped and translated-code pages, from the address space's
//! end and from outside it. `expect` gives the outcome:
//! - a completed access: registers, SP, Q0 and the data as the reference
//!   interpreter (`common::interpret`) computes them, or for the SIMD
//!   forms and the exclusive pair as modelled here;
//! - a trap: its kind and address, with nothing changed;
//! - an address outside the space: the panic, with its address (a trap
//!   for the checked copy S, D and Q transfers use without
//!   `inline-vector`);
//! - an access to bytes the guest may not access: in a child process, the
//!   end of the process by SIGSEGV, the handler naming one of those bytes.
#![cfg(any(target_os = "linux", windows))]
#![allow(clippy::unwrap_used, unsafe_code)]

mod common;

use common::*;
use echora_aot_tests::{ADDRESS_SPACE, Machine, READ, Trap};
use echora_lift::aarch64::decode;

#[allow(
    missing_debug_implementations,
    clippy::all,
    dead_code,
    unused_imports,
    unused_parens,
    unreachable_code
)]
mod lifted {
    include!(concat!(env!("OUT_DIR"), "/lifted_memory_fault.rs"));
}
use lifted::memory_fault_code as code;

const CHILD: &str = "ECHORA_MEMORY_FAULT_CHILD";
const END: u64 = DATA + DATA_SIZE;
const READ_ONLY: u64 = END - 0x1000;
/// The bits an in-range address may have (the view's range test).
const MASK: u64 = ADDRESS_SPACE.next_power_of_two() - 1;
const X2: u64 = 0x1122_3344_5566_7788;
const X3: u64 = 0x99aa_bbcc_ddee_ff00;
const Q0: u128 = 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210;

/// How a form reaches memory.
#[derive(Clone, Copy, PartialEq)]
enum Class {
    /// A plain access (two host accesses for `halves`).
    Plain,
    /// Exclusive and ordered accesses: natural alignment.
    Ordered,
    /// A store-exclusive without a monitor: the alignment check, then it
    /// fails without touching memory.
    StoreExclusiveAlone,
    /// LSE atomics: within one 16-byte block.
    Lse,
    /// SP-based: SP 16-byte aligned.
    Sp,
    /// S, D and Q transfers without `inline-vector`: the view's checked
    /// copy, a trap for bytes past the address space.
    Checked,
}

struct Form {
    /// The access's offset from the base register.
    offset: i64,
    bytes: u64,
    store: bool,
    halves: bool,
    class: Class,
}

fn form(name: &str) -> Form {
    let simd = [" s0", " d0", " q0"]
        .iter()
        .any(|register| name.contains(register));
    if simd && !cfg!(feature = "inline-vector") {
        let bytes = if name.contains(" s0") {
            4
        } else if name.contains(" d0") {
            8
        } else {
            16
        };
        return Form {
            offset: 0,
            bytes,
            store: name.starts_with("str"),
            halves: false,
            class: Class::Checked,
        };
    }
    let plain = |offset, bytes, store, halves| Form {
        offset,
        bytes,
        store,
        halves,
        class: Class::Plain,
    };
    let with = |class, offset, bytes, store| Form {
        offset,
        bytes,
        store,
        halves: false,
        class,
    };
    match name {
        "ldrb w0, [x1]" | "ldrsb x0, [x1]" => plain(0, 1, false, false),
        "ldrh w0, [x1]" | "ldrsh w0, [x1]" => plain(0, 2, false, false),
        "ldr w0, [x1]" | "ldrsw x0, [x1]" | "ldr s0, [x1]" => plain(0, 4, false, false),
        "ldr x0, [x1]" | "ldr d0, [x1]" => plain(0, 8, false, false),
        "strb w2, [x1]" => plain(0, 1, true, false),
        "strh w2, [x1]" => plain(0, 2, true, false),
        "str w2, [x1]" => plain(0, 4, true, false),
        "str x2, [x1]" | "str d0, [x1]" | "str x2, [x1], #-8" => plain(0, 8, true, false),
        "ldr x0, [x1, #8]!" => plain(8, 8, false, false),
        "ldur x0, [x1, #-1]" => plain(-1, 8, false, false),
        "ldr q0, [x1]" | "ldp x0, x3, [x1]" => plain(0, 16, false, true),
        "str q0, [x1]" | "stp x2, x3, [x1]" => plain(0, 16, true, true),
        "ldp w0, w3, [x1]" => plain(0, 8, false, true),
        "ldxr x0, [x1]" | "ldaxr x0, [x1]" | "ldar x0, [x1]" | "ldapr x0, [x1]" => {
            with(Class::Ordered, 0, 8, false)
        }
        "ldxr w0, [x1]" => with(Class::Ordered, 0, 4, false),
        "stlr x2, [x1]" | "ldxr x0, [x1]; stxr w4, x2, [x1]" => with(Class::Ordered, 0, 8, true),
        "stxr w4, x2, [x1]" => with(Class::StoreExclusiveAlone, 0, 8, true),
        "cas x0, x2, [x1]" | "ldadd x2, x0, [x1]" | "swp x2, x0, [x1]" => {
            with(Class::Lse, 0, 8, true)
        }
        "ldr x0, [sp]" => with(Class::Sp, 0, 8, false),
        "str x2, [sp, #8]" => with(Class::Sp, 8, 8, true),
        other => panic!("no model for {other}"),
    }
}

/// The data's bytes before every run: no two alike in a row, so a load of
/// the wrong width or at the wrong offset shows.
fn pattern() -> Vec<u8> {
    (0..DATA_SIZE as usize)
        .map(|index| (index * 37 + index / 251) as u8)
        .collect()
}

/// Whether the guest may access the byte at `address` (`store`: write it).
fn accessible(address: u64, store: bool) -> bool {
    let data = (DATA..END).contains(&address) && !(store && address >= READ_ONLY);
    let code = (code::BASE..code::BASE + 0x1000).contains(&address) && !store;
    data || code
}

#[derive(Debug, PartialEq)]
enum Expected {
    /// Registers x0-x4, SP, Q0 and the data.
    Done(Box<Snapshot>),
    /// A trap's kind and address, with the snapshot unchanged.
    Trap(String, Box<Snapshot>),
    /// The panic of an address outside the space.
    Panic(String),
    /// A host fault at one of these bytes.
    Fault(std::ops::Range<u64>),
}

#[derive(Debug, PartialEq, Clone)]
struct Snapshot {
    registers: [u64; 5],
    sp: u64,
    q0: u128,
    data: Vec<u8>,
}

fn start(address: u64) -> Snapshot {
    Snapshot {
        registers: [0, address, X2, X3, 7],
        sp: address,
        q0: Q0,
        data: pattern(),
    }
}

/// What `index`'s form must do from `address`, or `None` where the order
/// of a Q or pair access's halves decides it (one in the space and
/// inaccessible, the other outside) or for a completed load from the
/// code page.
fn expect(index: usize, address: u64) -> Option<Expected> {
    let name = code::NAMES[index];
    let form = form(name);
    let before = start(address);
    let target = address.wrapping_add(form.offset as u64);
    let trap = |kind: String| Some(Expected::Trap(kind, Box::new(before.clone())));
    match form.class {
        Class::Sp if address & 15 != 0 => return trap("InvalidState".into()),
        Class::Ordered | Class::StoreExclusiveAlone if !target.is_multiple_of(form.bytes) => {
            return trap(format!("MemoryFault at {target:#x}"));
        }
        Class::Lse if (target & 15) + form.bytes > 16 => {
            return trap(format!("MemoryFault at {target:#x}"));
        }
        Class::Checked
            if target
                .checked_add(form.bytes)
                .is_none_or(|end| end > ADDRESS_SPACE) =>
        {
            return trap(format!("MemoryFault at {target:#x}"));
        }
        _ => {}
    }
    if form.class == Class::StoreExclusiveAlone {
        let mut after = before;
        after.registers[4] = 1;
        return Some(Expected::Done(Box::new(after)));
    }
    let outside = |at: u64| at & !MASK != 0;
    let parts: Vec<(u64, u64)> = if form.halves {
        let half = form.bytes / 2;
        vec![(target, half), (target.wrapping_add(half), half)]
    } else {
        vec![(target, form.bytes)]
    };
    let inaccessible: Vec<u64> = parts
        .iter()
        .filter(|(at, _)| !outside(*at))
        .flat_map(|&(at, bytes)| (at..at + bytes).filter(|&byte| !accessible(byte, form.store)))
        .collect();
    if let Some(&(at, _)) = parts.iter().find(|(at, _)| outside(*at)) {
        if !inaccessible.is_empty() {
            return None;
        }
        return Some(Expected::Panic(format!(
            "guest memory access at {at:#x} lies outside the guest address space"
        )));
    }
    if let (Some(&first), Some(&last)) = (inaccessible.first(), inaccessible.last()) {
        return Some(Expected::Fault(first..last + 1));
    }
    // Loads from the code page complete too; the model knows only the data.
    if parts
        .iter()
        .any(|&(at, bytes)| at < DATA || at + bytes > END)
    {
        return None;
    }
    Some(Expected::Done(Box::new(complete(name, before))))
}

/// A completed access from `before`.
fn complete(name: &str, before: Snapshot) -> Snapshot {
    let offset = |at: u64| (at - DATA) as usize;
    let address = before.registers[1];
    let mut after = before.clone();
    match name {
        "ldr s0, [x1]" | "ldr d0, [x1]" | "ldr q0, [x1]" => {
            let bytes = form(name).bytes as usize;
            let mut value = 0u128;
            for (index, byte) in before.data[offset(address)..][..bytes].iter().enumerate() {
                value |= u128::from(*byte) << (8 * index);
            }
            after.q0 = value;
        }
        "str d0, [x1]" | "str q0, [x1]" => {
            let bytes = form(name).bytes as usize;
            after.data[offset(address)..][..bytes].copy_from_slice(&Q0.to_le_bytes()[..bytes]);
        }
        "ldxr x0, [x1]; stxr w4, x2, [x1]" => {
            let old = &before.data[offset(address)..][..8];
            after.registers[0] = u64::from_le_bytes(old.try_into().unwrap());
            after.registers[4] = 0;
            after.data[offset(address)..][..8].copy_from_slice(&X2.to_le_bytes());
        }
        _ => {
            let entry = code::ENTRIES[code::NAMES.iter().position(|n| *n == name).unwrap()];
            let word = code::WORDS[((entry - code::BASE) / 4) as usize];
            let mut state = State {
                x: [0; 31],
                sp: before.sp,
                nzcv: 0,
                memory: before.data.clone(),
            };
            state.x[..5].copy_from_slice(&before.registers);
            interpret(&decode(word, entry), &mut state).expect("an access inside the data");
            after.registers.copy_from_slice(&state.x[..5]);
            after.sp = state.sp;
            after.data = state.memory;
        }
    }
    after
}

fn machine() -> Machine {
    let words = code::WORDS;
    let code_size = (words.len() as u64 * 4).next_multiple_of(0x1000);
    let image: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let mut m = Machine::with_layout(code::BASE, &image, code_size, DATA, DATA_SIZE);
    m.memory.write(DATA, &pattern()).unwrap();
    m.memory.set_permissions(READ_ONLY, 0x1000, READ).unwrap();
    m
}

/// Runs `index`'s form from `address`: the snapshot after it and its trap
/// (`Exit` when it completed), or the panic's message.
fn run(m: &mut Machine, index: usize, address: u64) -> Result<(Snapshot, String), String> {
    let data = pattern();
    m.memory
        .write(DATA, &data[..(READ_ONLY - DATA) as usize])
        .unwrap();
    m.core.memory.clear_exclusive();
    let before = start(address);
    m.core.cpu.regs = [0; 31];
    m.core.cpu.regs[..5].copy_from_slice(&before.registers);
    m.core.cpu.sp = address;
    m.core.cpu.vectors[0] = Q0;
    m.core.cpu.pc = code::ENTRIES[index];
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| m.run(code::dispatch)));
    let trap = match result {
        Ok(Err(Trap::Exit { .. })) => "Exit".to_owned(),
        Ok(Err(Trap::MemoryFault { address, .. })) => format!("MemoryFault at {address:#x}"),
        Ok(Err(Trap::InvalidState { .. })) => "InvalidState".to_owned(),
        Ok(Err(other)) => format!("{other:?}"),
        Ok(Ok(())) => unreachable!("run returns only a trap"),
        Err(payload) => return Err(panic_message(payload.as_ref())),
    };
    let mut data = vec![0; DATA_SIZE as usize];
    m.memory.read(DATA, &mut data).unwrap();
    let snapshot = Snapshot {
        registers: std::array::from_fn(|index| m.core.cpu.regs[index]),
        sp: m.core.cpu.sp,
        q0: m.core.cpu.vectors[0],
        data,
    };
    Ok((snapshot, trap))
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| {
            payload
                .downcast_ref::<&str>()
                .map(|text| (*text).to_owned())
        })
        .unwrap_or_default()
}

/// Base addresses (X1, and SP) every form runs from.
const ADDRESSES: &[u64] = &[
    DATA + 0x8000,
    DATA + 0x8001,
    DATA + 0x8003,
    DATA + 0x8007,
    // Crossing a 16-byte boundary, and a page.
    DATA + 0x800d,
    DATA + 0x1000 - 3,
    DATA + 0x1000 - 1,
    // Into the read-only page, aligned and straddling into it.
    READ_ONLY,
    READ_ONLY - 3,
    // Straddling the data's end, and a page past it.
    END - 4,
    END + 0x1000,
    // Translated code (mapped without write).
    code::BASE,
    // Straddling the end of the address space (unmapped) into the
    // reservation's inaccessible tail.
    ADDRESS_SPACE - 4,
    // Outside the address space, and near the top of the 64-bit space
    // (far enough that no offset or writeback wraps it to a low address).
    ADDRESS_SPACE + 0x1000,
    ADDRESS_SPACE + DATA,
    u64::MAX - 0xfff,
];

#[test]
fn accesses_that_complete_trap_or_panic_do_as_modelled() {
    let mut m = machine();
    let mut counts = [0usize; 3];
    let mut failures = Vec::new();
    let cases: Vec<(usize, u64, Expected)> = (0..code::NAMES.len())
        .flat_map(|index| ADDRESSES.iter().map(move |&address| (index, address)))
        .filter_map(|(index, address)| match expect(index, address)? {
            Expected::Fault(_) => None,
            expected => Some((index, address, expected)),
        })
        .collect();
    // The default hook would print every expected panic.
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    for (index, address, expected) in cases {
        let name = code::NAMES[index];
        {
            let actual = match run(&mut m, index, address) {
                Ok((snapshot, trap)) if trap == "Exit" => Expected::Done(Box::new(snapshot)),
                Ok((snapshot, trap)) => Expected::Trap(trap, Box::new(snapshot)),
                Err(message) => Expected::Panic(message),
            };
            counts[match expected {
                Expected::Done(_) => 0,
                Expected::Trap(..) => 1,
                _ => 2,
            }] += 1;
            if actual != expected && failures.len() < 8 {
                let brief = |outcome: &Expected| match outcome {
                    Expected::Done(s) | Expected::Trap(_, s) => format!(
                        "{} {:x?} sp {:#x} q0 {:#x}",
                        match outcome {
                            Expected::Trap(kind, _) => kind.as_str(),
                            _ => "done",
                        },
                        s.registers,
                        s.sp,
                        s.q0
                    ),
                    other => format!("{other:?}"),
                };
                failures.push(format!(
                    "{name} from {address:#x}: expected {}, got {}",
                    brief(&expected),
                    brief(&actual)
                ));
            }
        }
    }
    std::panic::set_hook(hook);
    println!(
        "{} completed, {} traps, {} panics, {} mismatches",
        counts[0],
        counts[1],
        counts[2],
        failures.len()
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(
        counts[0] >= 150 && counts[1] >= 40 && counts[2] >= 80,
        "{counts:?}"
    );
}

/// In a child process (`CHILD` set): runs one form from one address,
/// which faults in the host and ends the process.
#[test]
fn accesses_to_bytes_the_guest_may_not_access_end_the_process() {
    if let Ok(case) = std::env::var(CHILD) {
        // No core dump for these intentional crashes.
        #[cfg(unix)]
        {
            let none = libc::rlimit {
                rlim_cur: 0,
                rlim_max: 0,
            };
            // SAFETY: setrlimit with a valid struct; affects only this process.
            assert_eq!(unsafe { libc::setrlimit(libc::RLIMIT_CORE, &none) }, 0);
        }
        // No Windows Error Reporting dialog for them.
        #[cfg(windows)]
        // SAFETY: changes only this process's error mode.
        unsafe {
            use windows_sys::Win32::System::Diagnostics::Debug::{
                SEM_NOGPFAULTERRORBOX, SetErrorMode,
            };
            SetErrorMode(SEM_NOGPFAULTERRORBOX);
        }
        let (index, address) = case.split_once(',').unwrap();
        let address = u64::from_str_radix(address, 16).unwrap();
        let outcome = run(&mut machine(), index.parse().unwrap(), address);
        // Not reached when the access faults.
        println!("no fault: {outcome:x?}");
        return;
    }
    let exe = std::env::current_exe().unwrap();
    let mut faults = 0;
    for (index, name) in code::NAMES.iter().enumerate() {
        for &address in ADDRESSES {
            let Some(Expected::Fault(bytes)) = expect(index, address) else {
                continue;
            };
            let output = std::process::Command::new(&exe)
                .args([
                    "--exact",
                    "accesses_to_bytes_the_guest_may_not_access_end_the_process",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(CHILD, format!("{index},{address:x}"))
                .output()
                .unwrap();
            let stderr = String::from_utf8_lossy(&output.stderr);
            let fault = stderr
                .lines()
                .find_map(|line| line.strip_prefix("guest memory fault at 0x"))
                .and_then(|at| u64::from_str_radix(at, 16).ok());
            assert!(
                fault.is_some_and(|at| bytes.contains(&at)),
                "{name} from {address:#x}: fault {fault:x?}, expected one in {bytes:x?}: {}{stderr}",
                String::from_utf8_lossy(&output.stdout)
            );
            assert!(!output.status.success(), "{name} from {address:#x}");
            #[cfg(unix)]
            {
                use std::os::unix::process::ExitStatusExt;
                assert_eq!(
                    output.status.signal(),
                    Some(libc::SIGSEGV),
                    "{name} from {address:#x}"
                );
            }
            faults += 1;
        }
    }
    println!("{faults} host faults");
    assert!(faults >= 90, "{faults}");
}
