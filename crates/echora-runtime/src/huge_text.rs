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
//! It pays with libraries linked hot code first (`echora layout`: an lld
//! symbol ordering file from a profile, and segments aligned to 2 MiB):
//! forest scene +5% presents/s with the first 128 MiB of each mapping
//! (+0.35 GB resident), instruction TLB misses -94%. With the default layout
//! the misses fall as much but the frame does not get faster, and all of the
//! text costs +1.4 GB resident. So by default only mappings whose code
//! segment the linker aligned to 2 MiB move, 128 MiB of each from its start.
//!
//! `ECHORA_HUGE_TEXT=0` turns it off, `=1` moves every mapping (all of it
//! unless bounded). `ECHORA_HUGE_TEXT_MB` bounds what each mapping gets from
//! its start (0: all of it). It stays off while the host profiler runs:
//! samples in anonymous memory have no file to be symbolized from.

// mmap, madvise, mprotect and mremap of the process's own code.
#![allow(unsafe_code)]

const HUGE: usize = 2 << 20;

/// What each aligned mapping gets by default: the hot code `echora layout`
/// places first (99.9% of a forest profile's samples: at most 106 MiB in
/// one library).
const DEFAULT_MB: usize = 128;

/// Remap once, before any translated code runs.
pub(crate) fn remap_once() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let mode = crate::nvn::debug_var("ECHORA_HUGE_TEXT").ok();
        let every = match mode.as_deref() {
            Some("0") => return,
            Some("1") => true,
            _ => false,
        };
        if std::env::var_os("ECHORA_HOST_PROFILE").is_some() {
            return;
        }
        let limit = match crate::nvn::debug_var("ECHORA_HUGE_TEXT_MB")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
        {
            Some(0) => usize::MAX,
            Some(mb) => mb << 20,
            None if every => usize::MAX,
            None => DEFAULT_MB << 20,
        };
        let started = std::time::Instant::now();
        let (mappings, bytes) = remap(limit, !every);
        if mappings > 0 {
            eprintln!(
                "huge text: {mappings} mappings, {} MiB moved to huge-page memory ({} MiB on 2 MiB pages) in {} ms",
                bytes >> 20,
                anon_huge_bytes() >> 20,
                started.elapsed().as_millis()
            );
        }
    });
}

/// The executable file mappings of the runner and the libraries in its
/// directory, as (start, end, whether the file's segment there is aligned
/// to 2 MiB).
fn own_text() -> Vec<(usize, usize, bool)> {
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
            let offset = u64::from_str_radix(fields.next()?, 16).ok()?;
            let path = fields.nth(2)?;
            let path = std::path::Path::new(path);
            if !perms.starts_with("r-x") || (path != exe && path.parent() != Some(dir)) {
                return None;
            }
            let (start, end) = range.split_once('-')?;
            Some((
                usize::from_str_radix(start, 16).ok()?,
                usize::from_str_radix(end, 16).ok()?,
                segment_align(path, offset).is_some_and(|align| align >= HUGE as u64),
            ))
        })
        .collect()
}

/// The alignment of the loadable segment of ELF file `path` that starts at
/// file offset `offset`.
fn segment_align(path: &std::path::Path, offset: u64) -> Option<u64> {
    use std::io::{Read, Seek};
    let mut file = std::fs::File::open(path).ok()?;
    let mut header = [0u8; 64];
    file.read_exact(&mut header).ok()?;
    if header[..6] != *b"\x7fELF\x02\x01" {
        return None;
    }
    let phoff = u64::from_le_bytes(header[0x20..0x28].try_into().ok()?);
    let entry = usize::from(u16::from_le_bytes(header[0x36..0x38].try_into().ok()?));
    let count = usize::from(u16::from_le_bytes(header[0x38..0x3a].try_into().ok()?));
    if entry < 0x38 {
        return None;
    }
    let mut table = vec![0u8; entry * count];
    file.seek(std::io::SeekFrom::Start(phoff)).ok()?;
    file.read_exact(&mut table).ok()?;
    table.chunks_exact(entry).find_map(|header| {
        let word = |at: usize| Some(u64::from_le_bytes(header.get(at..at + 8)?.try_into().ok()?));
        // PT_LOAD at that offset.
        if header[..4] == 1u32.to_le_bytes() && word(8)? == offset {
            word(0x30)
        } else {
            None
        }
    })
}

/// Move each mapping's whole 2 MiB pages, up to `limit` bytes from its
/// start, onto huge-page memory; with `only_aligned`, only mappings whose
/// segment is aligned to 2 MiB. Returns the mappings and bytes moved.
fn remap(limit: usize, only_aligned: bool) -> (usize, usize) {
    let mut mappings = 0;
    let mut moved = 0;
    for (start, end, aligned) in own_text() {
        if only_aligned && !aligned {
            continue;
        }
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
        let (mappings, bytes) = super::remap(usize::MAX, false);
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
