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

thread_local! {
    /// The core this host thread runs guest code on (parallel workers each
    /// have their own); the handler runs on the faulting thread.
    static CORE: Cell<usize> = const { Cell::new(0) };
}

/// Reports faults on this thread with `core`'s context until another core
/// is registered here. The caller keeps `core` in place while guest code
/// runs on it.
pub(crate) fn watch(core: &Core) {
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
}

#[cfg(unix)]
mod unix {
    use super::{CORE, CORES, Core};
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

    extern "C" fn report(_: libc::c_int, _: *mut libc::siginfo_t, _: *mut libc::c_void) {
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
            for (index, slot) in CORES.iter().enumerate() {
                let other = slot.load(Ordering::Relaxed) as *const Core;
                if other.is_null() {
                    continue;
                }
                // SAFETY: as above; another worker's core may be changing,
                // so this is a best-effort snapshot of a dying process.
                let (thread, pc, recent) = unsafe {
                    (
                        (*other).current_thread_id,
                        (*other).cpu.pc,
                        (*other).recent_thread_pcs(),
                    )
                };
                let mark = if other == core { " (faulted)" } else { "" };
                let _ = writeln!(text, "core slot {index}{mark}: thread {thread} pc {pc:#x}");
                for entry in recent.iter().rev().take(24).rev() {
                    let _ = writeln!(text, "  recent t{} {:#x}", entry.thread_id, entry.pc);
                }
            }
            // SAFETY: write(2) is async-signal-safe; the buffer is valid.
            unsafe { libc::write(2, text.as_ptr().cast(), text.len()) };
        }
    }
}
