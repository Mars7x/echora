//! Guest-fault context. The guest crate's handler reports the faulting guest
//! address and ends the process; this one runs first and adds the block PC
//! and link register of the core running guest code, then hands the signal
//! back to that handler. Generated code sets `cpu.pc` when a block exits, so
//! during a block it holds that block's start.
#![allow(unsafe_code)]

use echora_guest::Core;
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

static CORES: [AtomicUsize; 8] = [const { AtomicUsize::new(0) }; 8];
/// Snapshots reading `CORES` now. A core leaving `CORES` waits for them, so
/// no snapshot reads a core after its owner frees or moves it.
static READERS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// The core this host thread runs guest code on (parallel workers each
    /// have their own); the handler runs on the faulting thread.
    static CORE: Cell<usize> = const { Cell::new(0) };
}

/// Reports faults on this thread with `core`'s context until the returned
/// guard drops. The caller keeps `core` in place until then.
#[must_use = "the core is watched only until the guard drops"]
pub(crate) fn watch(core: &Core) -> Watch {
    CORE.with(|cell| cell.set(core as *const Core as usize));
    // Every watched core, so a fault on one worker also shows what the
    // others were running.
    let address = core as *const Core as usize;
    for slot in &CORES {
        match slot.compare_exchange(0, address, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(seen) if seen == address => break,
            Err(_) => {}
        }
    }
    #[cfg(unix)]
    unix::install();
    Watch { address }
}

/// A watched core; dropping it stops the watch, after which the core may
/// move or be freed.
pub(crate) struct Watch {
    address: usize,
}

impl Drop for Watch {
    fn drop(&mut self) {
        CORE.with(|cell| {
            if cell.get() == self.address {
                cell.set(0);
            }
        });
        for slot in &CORES {
            let _ = slot.compare_exchange(self.address, 0, Ordering::SeqCst, Ordering::SeqCst);
        }
        // A snapshot that saw the address before it left finishes first.
        while READERS.load(Ordering::SeqCst) != 0 {
            std::hint::spin_loop();
        }
    }
}

#[cfg(unix)]
mod unix {
    use super::{CORE, CORES, Core, READERS};
    use std::cell::Cell;
    use std::fmt::Write as _;
    use std::sync::OnceLock;
    use std::sync::atomic::Ordering;

    /// The handlers this one replaced (SIGSEGV, SIGBUS), restored on a fault.
    static PREVIOUS: OnceLock<[libc::sigaction; 2]> = OnceLock::new();

    pub(super) fn install() {
        static HOOK: std::sync::Once = std::sync::Once::new();
        HOOK.call_once(|| {
            // A panic in translated code (a guest access outside the
            // address space) gets the same snapshot as a fault.
            let previous = std::panic::take_hook();
            std::panic::set_hook(Box::new(move |info| {
                let core = CORE.with(Cell::get) as *const Core;
                if core.is_null() {
                    return previous(info);
                }
                snapshot(core);
                previous(info);
                // Other workers would wait for this one forever.
                std::process::abort();
            }));
        });
        PREVIOUS.get_or_init(|| {
            // SAFETY: installs a handler that reads atomics and two integers,
            // writes to stderr and restores the previous actions.
            unsafe {
                let mut action: libc::sigaction = std::mem::zeroed();
                action.sa_sigaction = report as *const () as usize;
                action.sa_flags = libc::SA_SIGINFO;
                let mut previous: [libc::sigaction; 2] = std::mem::zeroed();
                libc::sigaction(libc::SIGSEGV, &action, &mut previous[0]);
                libc::sigaction(libc::SIGBUS, &action, &mut previous[1]);
                previous
            }
        });
    }

    extern "C" fn report(_: libc::c_int, info: *mut libc::siginfo_t, context: *mut libc::c_void) {
        // Where the host faulted: the instruction, the address it touched and
        // the object holding the instruction (with its offset there, for
        // `addr2line`/`nm` on that file). A dying process can afford
        // `dladdr` and formatting.
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            // SAFETY: the kernel passes a valid siginfo and ucontext to an
            // SA_SIGINFO handler.
            let (rip, address) = unsafe {
                let context = &*(context as *const libc::ucontext_t);
                (
                    context.uc_mcontext.gregs[libc::REG_RIP as usize] as usize,
                    (*info).si_addr() as usize,
                )
            };
            let mut text = format!("host fault: rip {rip:#x} address {address:#x}");
            // SAFETY: dladdr reads the loader's tables; `found` is filled in
            // when it returns nonzero.
            unsafe {
                let mut found: libc::Dl_info = std::mem::zeroed();
                if libc::dladdr(rip as *const libc::c_void, &mut found) != 0 {
                    let file = if found.dli_fname.is_null() {
                        "?".into()
                    } else {
                        std::ffi::CStr::from_ptr(found.dli_fname).to_string_lossy()
                    };
                    let _ = write!(text, " in {file} +{:#x}", rip - found.dli_fbase as usize);
                    if !found.dli_sname.is_null() {
                        let symbol = std::ffi::CStr::from_ptr(found.dli_sname).to_string_lossy();
                        let _ = write!(text, " ({symbol})");
                    }
                }
            }
            text.push('\n');
            // How many mappings the process holds (a failed mmap leaves
            // libraries with pointers to nothing).
            if let Ok(maps) = std::fs::read_to_string("/proc/self/maps") {
                let _ = writeln!(text, "  mappings {}", maps.lines().count());
            }
            // The general registers, for what the faulting instruction used.
            // SAFETY: as above.
            let registers = unsafe { (*(context as *const libc::ucontext_t)).uc_mcontext.gregs };
            let names = [
                ("rax", libc::REG_RAX),
                ("rbx", libc::REG_RBX),
                ("rcx", libc::REG_RCX),
                ("rdx", libc::REG_RDX),
                ("rsi", libc::REG_RSI),
                ("rdi", libc::REG_RDI),
                ("rbp", libc::REG_RBP),
                ("rsp", libc::REG_RSP),
                ("r8", libc::REG_R8),
                ("r9", libc::REG_R9),
                ("r10", libc::REG_R10),
                ("r11", libc::REG_R11),
                ("r12", libc::REG_R12),
                ("r13", libc::REG_R13),
                ("r14", libc::REG_R14),
                ("r15", libc::REG_R15),
            ];
            for (index, (name, register)) in names.into_iter().enumerate() {
                let separator = if index % 4 == 3 { '\n' } else { ' ' };
                let _ = write!(
                    text,
                    "  {name} {:#018x}{separator}",
                    registers[register as usize] as u64
                );
            }
            // The first words each register points to, where readable (a
            // structure a library read its pointer from: freed or
            // overwritten). process_vm_readv reports an unreadable address
            // instead of faulting again.
            for (name, register) in names {
                let at = registers[register as usize] as usize;
                let mut words = [0u64; 6];
                let local = libc::iovec {
                    iov_base: words.as_mut_ptr().cast(),
                    iov_len: std::mem::size_of_val(&words),
                };
                let remote = libc::iovec {
                    iov_base: at as *mut libc::c_void,
                    iov_len: std::mem::size_of_val(&words),
                };
                // SAFETY: reads our own memory into `words`; a bad
                // address fails the call.
                let read =
                    unsafe { libc::process_vm_readv(libc::getpid(), &local, 1, &remote, 1, 0) };
                if read == std::mem::size_of_val(&words) as isize {
                    let _ = write!(text, "  [{name}]");
                    for word in words {
                        let _ = write!(text, " {word:016x}");
                    }
                    text.push('\n');
                }
            }
            // SAFETY: write(2) with a valid buffer.
            unsafe { libc::write(2, text.as_ptr().cast(), text.len()) };
        }
        #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
        let _ = (info, context);
        let core = CORE.with(Cell::get) as *const Core;
        if !core.is_null() {
            // SAFETY: `watch` registered a core its caller keeps alive while
            // guest code runs; only two integers are read.
            let (pc, lr) = unsafe { ((*core).cpu.pc, (*core).cpu.regs[30]) };
            let mut line = *b"guest block 0x0000000000000000, x30 0x0000000000000000\n";
            for index in 0..16 {
                let digit =
                    |value: u64| b"0123456789abcdef"[((value >> (60 - 4 * index)) & 0xf) as usize];
                line[14 + index] = digit(pc);
                line[38 + index] = digit(lr);
            }
            // SAFETY: write(2) is async-signal-safe; the buffer is valid.
            unsafe { libc::write(2, line.as_ptr().cast(), line.len()) };
            snapshot(core);
        }
        if let Some(previous) = PREVIOUS.get() {
            // Returning re-runs the faulting access under the previous
            // handler, which reports the guest address and ends the process.
            // SAFETY: restoring actions saved by `install`.
            unsafe {
                libc::sigaction(libc::SIGSEGV, &previous[0], std::ptr::null_mut());
                libc::sigaction(libc::SIGBUS, &previous[1], std::ptr::null_mut());
            }
        }
    }

    /// Every watched core's thread, PC and recent blocks, oldest first;
    /// `core` is marked as the one that failed. Collecting them allocates,
    /// which a dying process can afford: the fault is in translated code,
    /// not inside the allocator.
    fn snapshot(core: *const Core) {
        {
            let mut text = String::new();
            // Counted before any slot is read: a watch ending meanwhile
            // waits for this snapshot (see `Watch::drop`).
            READERS.fetch_add(1, Ordering::SeqCst);
            for (index, slot) in CORES.iter().enumerate() {
                let other = slot.load(Ordering::SeqCst) as *const Core;
                if other.is_null() {
                    continue;
                }
                // SAFETY: still watched (its watch ending waits for
                // READERS); another worker's core may be changing, so this
                // is a best-effort snapshot of a dying process.
                let (thread, pc, recent) = unsafe {
                    (
                        (*other).current_thread_id,
                        (*other).cpu.pc,
                        (*other).recent_thread_pcs(),
                    )
                };
                let mark = if other == core { " (faulted)" } else { "" };
                let _ = write!(text, "core slot {index}{mark}: thread {thread} pc {pc:#x}");
                // An NVN thunk: the entry point being run on the host.
                if let Some(name) = crate::nvn::thunk_name(pc) {
                    let _ = write!(text, " (NVN {name})");
                }
                text.push('\n');
                for entry in recent.iter().rev().take(24).rev() {
                    let _ = writeln!(text, "  recent t{} {:#x}", entry.thread_id, entry.pc);
                }
            }
            READERS.fetch_sub(1, Ordering::SeqCst);
            // SAFETY: write(2) is async-signal-safe; the buffer is valid.
            unsafe { libc::write(2, text.as_ptr().cast(), text.len()) };
        }
    }
}

/// Timed waits on this thread end close to their deadline: Linux lets a
/// thread's sleeps run up to 50 us late by default, so a guest's 20 us
/// SleepThread (games poll with them) took about 45, where the console
/// sleeps for close to what was asked.
pub(crate) fn precise_timers() {
    #[cfg(target_os = "linux")]
    // SAFETY: PR_SET_TIMERSLACK takes the slack in nanoseconds and affects
    // only the calling thread.
    unsafe {
        libc::prctl(libc::PR_SET_TIMERSLACK, 1 as libc::c_ulong, 0, 0, 0);
    }
}
