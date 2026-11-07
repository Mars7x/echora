//! Guest-fault context. The guest crate's handler reports the faulting guest
//! address and ends the process; this one runs first and adds the block PC
//! and link register of the core running guest code, then hands the signal
//! back to that handler. Generated code sets `cpu.pc` when a block exits, so
//! during a block it holds that block's start.
#![allow(unsafe_code)]

use echora_guest::Core;
use std::cell::Cell;

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
    #[cfg(unix)]
    unix::install();
}

#[cfg(unix)]
mod unix {
    use super::{CORE, Core};
    use std::cell::Cell;
    use std::sync::OnceLock;

    /// The handlers this one replaced (SIGSEGV, SIGBUS), restored on a fault.
    static PREVIOUS: OnceLock<[libc::sigaction; 2]> = OnceLock::new();

    pub(super) fn install() {
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
            // The recent block starts, oldest first. Collecting them
            // allocates, which a dying process can afford: the fault is in
            // translated code, not inside the allocator.
            // SAFETY: as above.
            let recent = unsafe { (*core).recent_thread_pcs() };
            for entry in recent.iter().rev().take(24).rev() {
                let mut line = *b"  recent 0x0000000000000000\n";
                for index in 0..16 {
                    line[11 + index] =
                        b"0123456789abcdef"[((entry.pc >> (60 - 4 * index)) & 0xf) as usize];
                }
                // SAFETY: as above.
                unsafe { libc::write(2, line.as_ptr().cast(), line.len()) };
            }
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
}
