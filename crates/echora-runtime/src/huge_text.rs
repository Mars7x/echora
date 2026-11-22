//! Translated code on 2 MiB pages.
//!
//! The translation's machine code is gigabytes of window functions in the
//! runner and its bundle libraries. Mapped from their files it sits on 4 KiB
//! pages, and the guest threads' hot code (some 150 MiB) misses the
//! instruction TLB far more often than its few hundred entries can hold.
//! At startup this copies the executable mappings of the runner and of the
//! libraries beside it into anonymous memory backed by transparent huge
//! pages and moves the copies over the originals (`mremap`). The addresses
//! and bytes stay the same, so nothing else changes.
//!
//! `ECHORA_HUGE_TEXT=1` turns it on (off by default). `ECHORA_HUGE_TEXT_MB`
//! bounds what each mapping gets from its start (0 or unset: all of it). It
//! stays off while the host profiler runs: samples in anonymous memory have
//! no file to be symbolized from.
//!
//! It pays with libraries linked hot code first (an lld symbol ordering
//! file from a profile, and 2 MiB aligned segments): forest scene +5%
//! presents/s with the first 128 MiB of each mapping (+0.35 GB resident),
//! instruction TLB misses -94%. With the default layout the misses fall as
//! much but the frame does not get faster, and all of the text costs
//! +1.4 GB resident, hence off by default.

// mmap, madvise, mprotect and mremap of the process's own code.
#![allow(unsafe_code)]

const HUGE: usize = 2 << 20;

/// Remap once, before any translated code runs.
pub(crate) fn remap_once() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let on = crate::nvn::debug_var("ECHORA_HUGE_TEXT").is_ok_and(|value| value == "1");
        if !on || std::env::var_os("ECHORA_HOST_PROFILE").is_some() {
            return;
        }
        let limit = crate::nvn::debug_var("ECHORA_HUGE_TEXT_MB")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&mb| mb > 0)
            .map_or(usize::MAX, |mb| mb << 20);
        let started = std::time::Instant::now();
        let (mappings, bytes) = remap(limit);
        eprintln!(
            "huge text: {mappings} mappings, {} MiB moved to huge-page memory ({} MiB on 2 MiB pages) in {} ms",
            bytes >> 20,
            anon_huge_bytes() >> 20,
            started.elapsed().as_millis()
        );
    });
}

/// The executable file mappings of the runner and the libraries in its
/// directory, as (start, end).
fn own_text() -> Vec<(usize, usize)> {
    let Ok(exe) = std::fs::read_link("/proc/self/exe") else {
        return Vec::new();
    };
    let Some(dir) = exe.parent() else {
        return Vec::new();
    };
    let Ok(maps) = std::fs::read_to_string("/proc/self/maps") else {
        return Vec::new();
    };
    maps.lines()
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            let range = fields.next()?;
            let perms = fields.next()?;
            let path = fields.nth(3)?;
            let path = std::path::Path::new(path);
            if !perms.starts_with("r-x") || (path != exe && path.parent() != Some(dir)) {
                return None;
            }
            let (start, end) = range.split_once('-')?;
            Some((
                usize::from_str_radix(start, 16).ok()?,
                usize::from_str_radix(end, 16).ok()?,
            ))
        })
        .collect()
}

/// Move each mapping's whole 2 MiB pages, up to `limit` bytes from its
/// start, onto huge-page memory. Returns the mappings and bytes moved.
fn remap(limit: usize) -> (usize, usize) {
    let mut mappings = 0;
    let mut moved = 0;
    for (start, end) in own_text() {
        let low = start.next_multiple_of(HUGE);
        let high = (end.min(start.saturating_add(limit))) & !(HUGE - 1);
        if high <= low {
            continue;
        }
        // SAFETY: `low..high` is mapped readable and executable; its copy is
        // made in fresh memory and moved over it in one `mremap`, which the
        // kernel performs under the address space's lock, so a thread
        // running this code (this one included) sees the same bytes at the
        // same addresses before and after.
        if unsafe { move_to_huge_pages(low, high - low) } {
            mappings += 1;
            moved += high - low;
        }
    }
    (mappings, moved)
}

/// Copy `len` bytes of code at `at` (2 MiB aligned) into 2 MiB aligned
/// anonymous memory advised for huge pages and move it over `at`.
unsafe fn move_to_huge_pages(at: usize, len: usize) -> bool {
    unsafe {
        let span = len + HUGE;
        let raw = libc::mmap(
            std::ptr::null_mut(),
            span,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
            -1,
            0,
        );
        if raw == libc::MAP_FAILED {
            return false;
        }
        let raw = raw as usize;
        let copy = raw.next_multiple_of(HUGE);
        if copy > raw {
            libc::munmap(raw as *mut libc::c_void, copy - raw);
        }
        let tail = raw + span - (copy + len);
        if tail > 0 {
            libc::munmap((copy + len) as *mut libc::c_void, tail);
        }
        let copy_ptr = copy as *mut libc::c_void;
        libc::madvise(copy_ptr, len, libc::MADV_HUGEPAGE);
        std::ptr::copy_nonoverlapping(at as *const u8, copy as *mut u8, len);
        let moved = libc::mprotect(copy_ptr, len, libc::PROT_READ | libc::PROT_EXEC) == 0
            && libc::mremap(
                copy_ptr,
                len,
                len,
                libc::MREMAP_MAYMOVE | libc::MREMAP_FIXED,
                at as *mut libc::c_void,
            ) != libc::MAP_FAILED;
        if !moved {
            libc::munmap(copy_ptr, len);
        }
        moved
    }
}

/// Anonymous huge-page bytes of the process (smaps_rollup AnonHugePages).
fn anon_huge_bytes() -> usize {
    std::fs::read_to_string("/proc/self/smaps_rollup")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|line| line.starts_with("AnonHugePages:"))?
                .split_whitespace()
                .nth(1)?
                .parse::<usize>()
                .ok()
        })
        .map_or(0, |kib| kib << 10)
}

#[cfg(test)]
mod tests {
    #[inline(never)]
    fn sum(values: &[u64]) -> u64 {
        values.iter().sum()
    }

    #[test]
    fn moved_code_keeps_running_with_the_same_bytes() {
        let text = super::own_text();
        assert!(
            !text.is_empty(),
            "the test binary has an executable mapping"
        );
        let probe = sum as *const () as usize;
        let before: Vec<u8> = (0..64)
            .map(|offset| unsafe { *((probe + offset) as *const u8) })
            .collect();
        let (mappings, bytes) = super::remap(usize::MAX);
        if bytes > 0 {
            assert!(mappings > 0);
        }
        let after: Vec<u8> = (0..64)
            .map(|offset| unsafe { *((probe + offset) as *const u8) })
            .collect();
        assert_eq!(before, after);
        assert_eq!(sum(&[1, 2, 3, 4]), 10);
    }
}
