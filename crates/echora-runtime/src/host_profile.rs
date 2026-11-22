//! A host user-time sampler for hosts where `perf` is unavailable.
//!
//! `ECHORA_HOST_PROFILE=PATH` arms a process user-time timer (`SIGVTALRM`,
//! 1 kHz); each signal records the interrupted host instruction pointer.
//! When the run ends, PATH receives `/proc/self/maps` followed by one
//! `count address thread top` line per sampled address, host thread id (low
//! 16 bits) and the word at the top of the stack, for offline
//! symbolization. In a leaf without a frame (glibc's memcpy variants) that
//! word is the caller's return address. A fifth column names the guest
//! thread the sampled host thread was running (`set_guest_thread`; 0 for
//! none), so one guest thread's time can be told apart from the others
//! sharing its workers.
//!
//! `ECHORA_HOST_PROFILE_STACKS=PREFIX[,PREFIX...]` (or `*`) also keeps, for
//! samples of host threads whose name starts with a prefix, the thread's
//! registers and the top of its stack (`ECHORA_HOST_PROFILE_STACK_BYTES`,
//! default 16 KiB, at most 64 KiB; `ECHORA_HOST_PROFILE_STACK_MB` of them in
//! all, default 512), for an offline unwinder to walk with each object's
//! `.eh_frame` (no frame pointers needed). They go to `PATH.stacks`: per
//! sample, little-endian u64 rip, rsp, rbp, thread id, guest thread and
//! byte count, the thread's 16-byte name, then that many stack bytes from
//! rsp up.

#![allow(unsafe_code)]

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod imp {
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    use std::sync::atomic::AtomicPtr;

    const CAPACITY: usize = 1 << 21;
    /// A stack sample's fixed part: six words and the thread name.
    const STACK_HEADER: usize = 64;
    /// The stack buffer (`ECHORA_HOST_PROFILE_STACKS`), its slot size and
    /// slot count, and the next slot.
    static STACKS: AtomicPtr<u8> = AtomicPtr::new(std::ptr::null_mut());
    static STACK_SLOT: AtomicUsize = AtomicUsize::new(0);
    static STACK_SLOTS: AtomicUsize = AtomicUsize::new(0);
    static STACK_NEXT: AtomicUsize = AtomicUsize::new(0);
    static PID: AtomicU64 = AtomicU64::new(0);
    /// Thread name prefixes whose samples keep stacks (empty: all), each
    /// at most 15 bytes, NUL-padded.
    const MAX_PREFIXES: usize = 8;
    static PREFIXES: [AtomicU64; MAX_PREFIXES * 2] =
        [const { AtomicU64::new(0) }; MAX_PREFIXES * 2];
    static PREFIX_COUNT: AtomicUsize = AtomicUsize::new(0);
    static ALL_THREADS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    static SAMPLES: [AtomicU64; CAPACITY] = [const { AtomicU64::new(0) }; CAPACITY];
    static TOPS: [AtomicU64; CAPACITY] = [const { AtomicU64::new(0) }; CAPACITY];
    static GUESTS: [AtomicU64; CAPACITY] = [const { AtomicU64::new(0) }; CAPACITY];
    static NEXT: AtomicUsize = AtomicUsize::new(0);

    std::thread_local! {
        // Const-initialized and without a destructor: readable from the
        // signal handler.
        static GUEST: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    }

    /// The guest thread this host thread runs from now on (0: none).
    pub fn set_guest_thread(thread: u64) {
        GUEST.with(|guest| guest.set(thread));
    }

    extern "C" fn on_sample(_: libc::c_int, _: *mut libc::siginfo_t, context: *mut libc::c_void) {
        // SAFETY: the kernel passes a valid `ucontext_t` to SA_SIGINFO handlers.
        let (rip, rsp, rbp) = unsafe {
            let registers = &(*(context as *const libc::ucontext_t)).uc_mcontext.gregs;
            (
                registers[libc::REG_RIP as usize],
                registers[libc::REG_RSP as usize],
                registers[libc::REG_RBP as usize],
            )
        };
        // SAFETY: the interrupted thread's stack pointer addresses its
        // mapped stack.
        let top = unsafe { *(rsp as *const u64) };
        // User addresses fit 48 bits; the top 16 carry the thread's id.
        // SAFETY: gettid is async-signal-safe.
        let thread = unsafe { libc::gettid() } as u64 & 0xffff;
        let slot = NEXT.fetch_add(1, Ordering::Relaxed);
        if slot < CAPACITY {
            SAMPLES[slot].store(
                rip as u64 & 0xffff_ffff_ffff | thread << 48,
                Ordering::Relaxed,
            );
            TOPS[slot].store(top, Ordering::Relaxed);
            GUESTS[slot].store(GUEST.with(std::cell::Cell::get), Ordering::Relaxed);
        }
        if !STACKS.load(Ordering::Relaxed).is_null() {
            // SAFETY: errno is this thread's; the system calls made while
            // keeping the stack must not change what the interrupted code sees.
            unsafe {
                let errno = *libc::__errno_location();
                keep_stack(rip as u64, rsp as u64, rbp as u64);
                *libc::__errno_location() = errno;
            }
        }
    }

    /// Copy the interrupted thread's registers and the top of its stack
    /// into the next stack slot, when its name is one asked for. Only
    /// async-signal-safe calls: prctl, gettid and process_vm_readv (which
    /// stops at an unmapped page instead of faulting).
    ///
    /// # Safety
    /// From the sampling signal's handler, with `STACKS` set by `start`.
    unsafe fn keep_stack(rip: u64, rsp: u64, rbp: u64) {
        let mut name = [0u8; 16];
        // SAFETY: PR_GET_NAME writes at most 16 bytes.
        unsafe { libc::prctl(libc::PR_GET_NAME, name.as_mut_ptr()) };
        if !ALL_THREADS.load(Ordering::Relaxed) {
            let wanted = (0..PREFIX_COUNT.load(Ordering::Relaxed)).any(|index| {
                let mut prefix = [0u8; 16];
                prefix[..8]
                    .copy_from_slice(&PREFIXES[index * 2].load(Ordering::Relaxed).to_le_bytes());
                prefix[8..].copy_from_slice(
                    &PREFIXES[index * 2 + 1]
                        .load(Ordering::Relaxed)
                        .to_le_bytes(),
                );
                let length = prefix.iter().position(|byte| *byte == 0).unwrap_or(16);
                name[..length] == prefix[..length]
            });
            if !wanted {
                return;
            }
        }
        let slot = STACK_NEXT.fetch_add(1, Ordering::Relaxed);
        if slot >= STACK_SLOTS.load(Ordering::Relaxed) {
            return;
        }
        let size = STACK_SLOT.load(Ordering::Relaxed);
        // SAFETY: the slot lies inside the mapped buffer and is this
        // sample's alone (claimed above).
        let record = unsafe { STACKS.load(Ordering::Relaxed).add(slot * size) };
        let capacity = size - STACK_HEADER;
        // The stack one page at a time: a partial read stops at the first
        // page that is not mapped.
        let mut remote = [libc::iovec {
            iov_base: std::ptr::null_mut(),
            iov_len: 0,
        }; 17];
        let mut pieces = 0;
        let mut at = rsp;
        let end = rsp.saturating_add(capacity as u64);
        while at < end && pieces < remote.len() {
            let next = ((at | 4095) + 1).min(end);
            remote[pieces] = libc::iovec {
                iov_base: at as *mut libc::c_void,
                iov_len: (next - at) as usize,
            };
            pieces += 1;
            at = next;
        }
        let local = libc::iovec {
            // SAFETY: inside this slot, after the header.
            iov_base: unsafe { record.add(STACK_HEADER) }.cast(),
            iov_len: capacity,
        };
        // SAFETY: reads this process's own memory into the slot; unmapped
        // pages end the copy.
        let copied = unsafe {
            libc::process_vm_readv(
                PID.load(Ordering::Relaxed) as libc::pid_t,
                &local,
                1,
                remote.as_ptr(),
                pieces as libc::c_ulong,
                0,
            )
        }
        .max(0) as u64;
        // SAFETY: gettid is async-signal-safe.
        let thread = unsafe { libc::gettid() } as u64;
        let words = [
            rip,
            rsp,
            rbp,
            thread,
            GUEST.with(std::cell::Cell::get),
            copied,
        ];
        for (index, word) in words.iter().enumerate() {
            // SAFETY: the header's first 48 bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(word.to_le_bytes().as_ptr(), record.add(index * 8), 8)
            };
        }
        // SAFETY: the header's last 16 bytes.
        unsafe { std::ptr::copy_nonoverlapping(name.as_ptr(), record.add(48), 16) };
    }

    /// Map the stack buffer and note which threads keep stacks
    /// (`ECHORA_HOST_PROFILE_STACKS`).
    fn arm_stacks() {
        let Some(threads) = std::env::var_os("ECHORA_HOST_PROFILE_STACKS") else {
            return;
        };
        let threads = threads.to_string_lossy().into_owned();
        let mut count = 0;
        for prefix in threads.split(',').filter(|prefix| !prefix.is_empty()) {
            if prefix == "*" {
                ALL_THREADS.store(true, Ordering::Relaxed);
                continue;
            }
            if count == MAX_PREFIXES {
                break;
            }
            let mut bytes = [0u8; 16];
            let length = prefix.len().min(15);
            bytes[..length].copy_from_slice(&prefix.as_bytes()[..length]);
            PREFIXES[count * 2].store(
                u64::from_le_bytes(*bytes.first_chunk().expect("16 bytes")),
                Ordering::Relaxed,
            );
            PREFIXES[count * 2 + 1].store(
                u64::from_le_bytes(*bytes.last_chunk().expect("16 bytes")),
                Ordering::Relaxed,
            );
            count += 1;
        }
        PREFIX_COUNT.store(count, Ordering::Relaxed);
        let number = |name: &str, default: usize| {
            std::env::var(name)
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(default)
        };
        let bytes = number("ECHORA_HOST_PROFILE_STACK_BYTES", 16 << 10).clamp(512, 64 << 10);
        let slot = (STACK_HEADER + bytes).next_multiple_of(64);
        let total = number("ECHORA_HOST_PROFILE_STACK_MB", 512).clamp(1, 8192) << 20;
        // SAFETY: an anonymous private mapping, kept for the run.
        let buffer = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                total,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if buffer == libc::MAP_FAILED {
            eprintln!("host profile: no memory for stacks");
            return;
        }
        PID.store(std::process::id().into(), Ordering::Relaxed);
        STACK_SLOT.store(slot, Ordering::Relaxed);
        STACK_SLOTS.store(total / slot, Ordering::Relaxed);
        STACKS.store(buffer.cast(), Ordering::Release);
    }

    /// Write the stack samples to `PATH.stacks`.
    fn write_stacks(path: &std::ffi::OsStr) {
        let buffer = STACKS.load(Ordering::Acquire);
        if buffer.is_null() {
            return;
        }
        let size = STACK_SLOT.load(Ordering::Relaxed);
        let taken = STACK_NEXT
            .load(Ordering::Relaxed)
            .min(STACK_SLOTS.load(Ordering::Relaxed));
        let mut out = Vec::new();
        for slot in 0..taken {
            // SAFETY: a claimed slot inside the buffer; the timer is disarmed.
            let record = unsafe { std::slice::from_raw_parts(buffer.add(slot * size), size) };
            let copied =
                u64::from_le_bytes(*record[40..].first_chunk().expect("a full header")) as usize;
            out.extend_from_slice(&record[..STACK_HEADER + copied.min(size - STACK_HEADER)]);
        }
        let mut name = path.to_owned();
        name.push(".stacks");
        match std::fs::write(&name, out) {
            Ok(()) => eprintln!("host profile: {taken} stack samples"),
            Err(error) => eprintln!("host profile: stack write failed: {error}"),
        }
    }

    pub struct Profile {
        path: std::ffi::OsString,
    }

    /// Drop the samples taken so far (ECHORA_PROFILE_FROM: profile one
    /// phase of a run).
    pub fn restart() {
        NEXT.store(0, Ordering::Relaxed);
        STACK_NEXT.store(0, Ordering::Relaxed);
    }

    pub fn start() -> Option<Profile> {
        let path = std::env::var_os("ECHORA_HOST_PROFILE")?;
        arm_stacks();
        // SAFETY: installs a handler that only touches atomics, then arms
        // the process CPU-time timer.
        unsafe {
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = on_sample as *const () as usize;
            action.sa_flags = libc::SA_SIGINFO | libc::SA_RESTART;
            libc::sigaction(libc::SIGVTALRM, &action, std::ptr::null_mut());
            let period = libc::timeval {
                tv_sec: 0,
                tv_usec: 1000,
            };
            let timer = libc::itimerval {
                it_interval: period,
                it_value: period,
            };
            libc::setitimer(libc::ITIMER_VIRTUAL, &timer, std::ptr::null_mut());
        }
        Some(Profile { path })
    }

    impl Drop for Profile {
        fn drop(&mut self) {
            // SAFETY: disarms the timer armed in `start`.
            unsafe {
                let timer: libc::itimerval = std::mem::zeroed();
                libc::setitimer(libc::ITIMER_VIRTUAL, &timer, std::ptr::null_mut());
            }
            let taken = NEXT.load(Ordering::Relaxed).min(CAPACITY);
            let mut counts = std::collections::HashMap::<(u64, u64, u64), u64>::new();
            for ((sample, top), guest) in SAMPLES[..taken]
                .iter()
                .zip(&TOPS[..taken])
                .zip(&GUESTS[..taken])
            {
                let key = (
                    sample.load(Ordering::Relaxed),
                    top.load(Ordering::Relaxed),
                    guest.load(Ordering::Relaxed),
                );
                *counts.entry(key).or_default() += 1;
            }
            let mut counts: Vec<_> = counts.into_iter().collect();
            counts.sort_unstable_by(|a, b| b.1.cmp(&a.1));
            let mut out = std::fs::read_to_string("/proc/self/maps").unwrap_or_default();
            out.push_str("--\n");
            for ((sample, top, guest), count) in counts {
                let (thread, address) = (sample >> 48, sample & 0xffff_ffff_ffff);
                out.push_str(&format!("{count} {address:#x} {thread} {top:#x} {guest}\n"));
            }
            if let Err(error) = std::fs::write(&self.path, out) {
                eprintln!("host profile: write failed: {error}");
            } else {
                eprintln!("host profile: {taken} samples");
            }
            write_stacks(&self.path);
        }
    }
}

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
mod imp {
    pub struct Profile;

    pub fn start() -> Option<Profile> {
        None
    }

    pub fn restart() {}

    pub fn set_guest_thread(_: u64) {}
}

pub use imp::{Profile, restart, set_guest_thread, start};
