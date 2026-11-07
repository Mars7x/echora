//! A host user-time sampler for hosts where `perf` is unavailable.
//!
//! `ECHORA_HOST_PROFILE=PATH` arms a process user-time timer (`SIGVTALRM`,
//! 1 kHz); each signal records the interrupted host instruction pointer.
//! When the run ends, PATH receives `/proc/self/maps` followed by one
//! `count address thread` line per sampled address and host thread id (low
//! 16 bits), for offline symbolization.

#![allow(unsafe_code)]

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod imp {
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    const CAPACITY: usize = 1 << 21;
    static SAMPLES: [AtomicU64; CAPACITY] = [const { AtomicU64::new(0) }; CAPACITY];
    static NEXT: AtomicUsize = AtomicUsize::new(0);

    extern "C" fn on_sample(_: libc::c_int, _: *mut libc::siginfo_t, context: *mut libc::c_void) {
        // SAFETY: the kernel passes a valid `ucontext_t` to SA_SIGINFO handlers.
        let rip = unsafe {
            (*(context as *const libc::ucontext_t)).uc_mcontext.gregs[libc::REG_RIP as usize]
        };
        // User addresses fit 48 bits; the top 16 carry the thread's id.
        // SAFETY: gettid is async-signal-safe.
        let thread = unsafe { libc::gettid() } as u64 & 0xffff;
        let slot = NEXT.fetch_add(1, Ordering::Relaxed);
        if slot < CAPACITY {
            SAMPLES[slot].store(
                rip as u64 & 0xffff_ffff_ffff | thread << 48,
                Ordering::Relaxed,
            );
        }
    }

    pub struct Profile {
        path: std::ffi::OsString,
    }

    pub fn start() -> Option<Profile> {
        let path = std::env::var_os("ECHORA_HOST_PROFILE")?;
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
            let mut counts = std::collections::HashMap::<u64, u64>::new();
            for sample in &SAMPLES[..taken] {
                *counts.entry(sample.load(Ordering::Relaxed)).or_default() += 1;
            }
            let mut counts: Vec<_> = counts.into_iter().collect();
            counts.sort_unstable_by(|a, b| b.1.cmp(&a.1));
            let mut out = std::fs::read_to_string("/proc/self/maps").unwrap_or_default();
            out.push_str("--\n");
            for (sample, count) in counts {
                let (thread, address) = (sample >> 48, sample & 0xffff_ffff_ffff);
                out.push_str(&format!("{count} {address:#x} {thread}\n"));
            }
            if let Err(error) = std::fs::write(&self.path, out) {
                eprintln!("host profile: write failed: {error}");
            } else {
                eprintln!("host profile: {taken} samples");
            }
        }
    }
}

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
mod imp {
    pub struct Profile;

    pub fn start() -> Option<Profile> {
        None
    }
}

pub use imp::{Profile, start};
