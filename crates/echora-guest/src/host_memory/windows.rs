//! Windows backend: guest RAM as pagefile-backed sections seen through two
//! placeholder-managed host views.
//!
//! The memory "file" is split into `CHUNK`-sized pagefile-backed sections,
//! created when a chunk is first written or mapped into the guest view, so
//! commit charge follows the memory in use rather than the size of the
//! guest address space. Offsets are canonical guest page addresses, as on
//! Linux: chunk `i` holds file offsets `[i * CHUNK, (i + 1) * CHUNK)`.
//!
//! Both views are address-space reservations made of placeholders
//! (`VirtualAlloc2` with `MEM_RESERVE_PLACEHOLDER`, Windows 10 1803 or
//! later). Replacing a placeholder with `MapViewOfFile3` maps a view at
//! page rather than 64 KiB granularity, which the guest view needs. The
//! physical view maps each present chunk read-write at its file offset;
//! absent chunks read as zero. The guest view maps each mapped guest page's
//! backing range with the guest's permissions and leaves everything else as
//! inaccessible placeholders.
//!
//! A view cannot be partly unmapped, so changing part of one unmaps it
//! whole, re-forms the placeholders around the changed range and maps the
//! untouched remainders again. `Space` keeps an exact record of the regions
//! (views and free placeholders) so every split and coalesce names whole
//! placeholders.
//!
//! Sections cannot have pages decommitted, so `release` returns storage only
//! for whole chunks (by replacing the section with a fresh one when the
//! guest view still maps it) and zeroes partial ranges in place.
#![allow(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::c_void;
use std::ptr::NonNull;

use windows_sys::Win32::Foundation::{
    CloseHandle, EXCEPTION_ACCESS_VIOLATION, HANDLE, INVALID_HANDLE_VALUE,
};
use windows_sys::Win32::System::Diagnostics::Debug::{
    AddVectoredExceptionHandler, EXCEPTION_CONTINUE_SEARCH, EXCEPTION_POINTERS,
};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER,
    MEM_RESERVE, MEM_RESERVE_PLACEHOLDER, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile3,
    PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE, SEC_COMMIT, UnmapViewOfFile2, VirtualAlloc2,
    VirtualFree,
};
use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};
use windows_sys::Win32::System::SystemServices::MEM_COALESCE_PLACEHOLDERS;
use windows_sys::Win32::System::Threading::GetCurrentProcess;

use crate::memory::PAGE_SIZE;

/// Bytes of memory file per section. A multiple of the 64 KiB allocation
/// granularity, so each chunk's physical view starts on a granule.
const CHUNK: u64 = 2 << 20;

/// Windows' allocation granularity: reservations are sized in these.
const GRANULE: u64 = 1 << 16;

const GUARD_BYTES: u64 = 1 << 20;

/// Bytes of the guest view's reservation for an address space of `size`
/// bytes: the next power of two, so generated code tests an address with one
/// mask (`GuestView::at`), plus an inaccessible guard for accesses that
/// start in range and run past its end.
pub(crate) fn guest_reservation(size: u64) -> u64 {
    size.next_power_of_two().max(GRANULE) + GUARD_BYTES
}

/// What the physical view reads for a page whose chunk does not exist yet.
static ZERO_PAGE: [u8; PAGE_SIZE as usize] = [0; PAGE_SIZE as usize];

/// An owned pagefile-backed section handle.
struct Section(HANDLE);

impl Section {
    fn new() -> Self {
        // SAFETY: an anonymous pagefile-backed section; the handle is owned
        // by the returned value.
        let handle = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_COMMIT,
                (CHUNK >> 32) as u32,
                CHUNK as u32,
                std::ptr::null(),
            )
        };
        assert!(
            !handle.is_null(),
            "guest memory section creation failed: {}",
            std::io::Error::last_os_error()
        );
        Self(handle)
    }
}

impl Drop for Section {
    fn drop(&mut self) {
        // SAFETY: the handle is ours; views keep the section alive on their own.
        unsafe { CloseHandle(self.0) };
    }
}

/// A view of `chunk` at section offset `offset` with page `protection`.
#[derive(Clone, Copy)]
struct View {
    chunk: usize,
    offset: u64,
    protection: u32,
}

/// `[start, end)` of a reservation: a view, or a free placeholder.
#[derive(Clone, Copy)]
struct Region {
    end: u64,
    view: Option<View>,
}

/// A placeholder reservation and the exact layout of its views and free
/// placeholders, keyed by offset from `base`.
struct Space {
    base: NonNull<u8>,
    regions: BTreeMap<u64, Region>,
    /// `(chunk, start)` of every view, to find the views of a chunk.
    by_chunk: BTreeSet<(usize, u64)>,
}

fn process() -> HANDLE {
    // SAFETY: returns the current-process pseudo handle.
    unsafe { GetCurrentProcess() }
}

impl Space {
    /// One free placeholder of `size` bytes (a multiple of `GRANULE`).
    fn reserve(size: u64) -> Self {
        debug_assert!(size.is_multiple_of(GRANULE));
        // SAFETY: a fresh placeholder at a system-chosen address.
        let address = unsafe {
            VirtualAlloc2(
                std::ptr::null_mut(),
                std::ptr::null(),
                size as usize,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            )
        };
        let base = NonNull::new(address.cast()).unwrap_or_else(|| {
            panic!(
                "guest memory reservation failed: {}",
                std::io::Error::last_os_error()
            )
        });
        Self {
            base,
            regions: BTreeMap::from([(
                0,
                Region {
                    end: size,
                    view: None,
                },
            )]),
            by_chunk: BTreeSet::new(),
        }
    }

    fn at(&self, offset: u64) -> *mut c_void {
        self.base.as_ptr().wrapping_add(offset as usize).cast()
    }

    /// Replace the free placeholder `[start, end)` with a view.
    fn map_view(&mut self, start: u64, end: u64, sections: &[Option<Section>], view: View) {
        let section = sections[view.chunk]
            .as_ref()
            .expect("a mapped chunk has a section");
        debug_assert!(matches!(
            self.regions.get(&start),
            Some(Region { end: e, view: None }) if *e == end
        ));
        // SAFETY: `[start, end)` is exactly one free placeholder in our
        // reservation, and the section range lies within the chunk.
        let mapped = unsafe {
            MapViewOfFile3(
                section.0,
                process(),
                self.at(start),
                view.offset,
                (end - start) as usize,
                MEM_REPLACE_PLACEHOLDER,
                view.protection,
                std::ptr::null_mut(),
                0,
            )
        };
        assert!(
            !mapped.Value.is_null(),
            "guest memory view mapping failed: {}",
            std::io::Error::last_os_error()
        );
        self.regions.insert(
            start,
            Region {
                end,
                view: Some(view),
            },
        );
        self.by_chunk.insert((view.chunk, start));
    }

    /// Turn the view at `start` back into a free placeholder of the same
    /// extent, returning the view.
    fn unmap_view(&mut self, start: u64) -> View {
        let region = self.regions[&start];
        let view = region.view.expect("unmapping a view");
        // SAFETY: a view we mapped; the placeholder is kept.
        let result = unsafe {
            UnmapViewOfFile2(
                process(),
                MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: self.at(start),
                },
                MEM_PRESERVE_PLACEHOLDER,
            )
        };
        assert!(
            result != 0,
            "guest memory view unmapping failed: {}",
            std::io::Error::last_os_error()
        );
        self.regions.insert(
            start,
            Region {
                end: region.end,
                view: None,
            },
        );
        self.by_chunk.remove(&(view.chunk, start));
        view
    }

    /// `VirtualFree` on placeholders: a split or a coalesce.
    fn free(&self, start: u64, end: u64, kind: u32) {
        // SAFETY: `[start, end)` is a placeholder range of our reservation;
        // with these flags the range stays reserved.
        let result = unsafe { VirtualFree(self.at(start), (end - start) as usize, kind) };
        assert!(
            result != 0,
            "guest memory placeholder update failed: {}",
            std::io::Error::last_os_error()
        );
    }

    /// Split the free placeholder starting at `start` so a placeholder ends
    /// at `at`.
    fn split(&mut self, start: u64, at: u64) {
        let end = self.regions[&start].end;
        debug_assert!(start < at && at < end);
        self.free(start, at, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
        self.regions.insert(
            start,
            Region {
                end: at,
                view: None,
            },
        );
        self.regions.insert(at, Region { end, view: None });
    }

    /// Make `[low, high)` exactly one free placeholder. Views that extend
    /// past either end keep their outside parts mapped as before.
    fn carve(&mut self, low: u64, high: u64, sections: &[Option<Section>]) {
        let first = *self
            .regions
            .range(..=low)
            .next_back()
            .expect("the reservation covers the range")
            .0;
        let starts: Vec<u64> = self.regions.range(first..high).map(|(s, _)| *s).collect();
        let last_end = self.regions[starts.last().expect("a region")].end;
        assert!(last_end >= high, "range lies outside the reservation");
        let mut left = None;
        let mut right = None;
        for &start in &starts {
            let end = self.regions[&start].end;
            if self.regions[&start].view.is_some() {
                let view = self.unmap_view(start);
                if start < low {
                    left = Some((start, view));
                }
                if end > high {
                    right = Some((
                        end,
                        View {
                            offset: view.offset + (high - start),
                            ..view
                        },
                    ));
                }
            }
        }
        if starts.len() > 1 {
            self.free(first, last_end, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS);
            for start in &starts[1..] {
                self.regions.remove(start);
            }
            self.regions.insert(
                first,
                Region {
                    end: last_end,
                    view: None,
                },
            );
        }
        if first < low {
            self.split(first, low);
        }
        if last_end > high {
            self.split(low, high);
        }
        if let Some((start, view)) = left {
            self.map_view(start, low, sections, view);
        }
        if let Some((end, view)) = right {
            self.map_view(high, end, sections, view);
        }
    }

    /// The views of `chunk` as `(start, end, view)`.
    fn views_of(&self, chunk: usize) -> Vec<(u64, u64, View)> {
        self.by_chunk
            .range((chunk, 0)..=(chunk, u64::MAX))
            .map(|&(_, start)| {
                let region = self.regions[&start];
                (start, region.end, region.view.expect("indexed view"))
            })
            .collect()
    }
}

impl Drop for Space {
    fn drop(&mut self) {
        for (&start, region) in &self.regions {
            // SAFETY: each region is a view we mapped or a placeholder we
            // reserved; nothing uses the reservation any more.
            unsafe {
                if region.view.is_some() {
                    UnmapViewOfFile2(
                        process(),
                        MEMORY_MAPPED_VIEW_ADDRESS {
                            Value: self.at(start),
                        },
                        0,
                    );
                } else {
                    VirtualFree(self.at(start), 0, MEM_RELEASE);
                }
            }
        }
    }
}

// Fields drop in order: the views go before the sections they map.
pub(crate) struct HostMemory {
    guest: Space,
    physical: Space,
    chunks: Vec<Option<Section>>,
    size: u64,
}

// SAFETY: the views are plain memory and the handles are process-wide;
// access is synchronized by the `Memory` that owns this object.
unsafe impl Send for HostMemory {}
unsafe impl Sync for HostMemory {}

impl HostMemory {
    /// An all-zero memory file of `size` bytes and both views of it.
    pub(crate) fn new(size: u64) -> Self {
        // SAFETY: SYSTEM_INFO is plain data; GetSystemInfo fills it.
        let info: SYSTEM_INFO = unsafe {
            let mut info = std::mem::zeroed();
            GetSystemInfo(&mut info);
            info
        };
        assert_eq!(
            info.dwPageSize, 4096,
            "guest memory needs 4 KiB host pages; this host uses {}-byte pages",
            info.dwPageSize
        );
        let chunk_count = size.div_ceil(CHUNK);
        let physical = Space::reserve(chunk_count * CHUNK);
        let guest = Space::reserve(guest_reservation(size).next_multiple_of(GRANULE));
        install_fault_report(guest.base.as_ptr() as usize, guest_reservation(size));
        Self {
            size,
            chunks: (0..chunk_count).map(|_| None).collect(),
            physical,
            guest,
        }
    }

    /// Create chunk `chunk`'s section and map it into the physical view.
    fn ensure_chunk(&mut self, chunk: usize) {
        if self.chunks[chunk].is_some() {
            return;
        }
        self.chunks[chunk] = Some(Section::new());
        let start = chunk as u64 * CHUNK;
        self.physical.carve(start, start + CHUNK, &self.chunks);
        self.physical.map_view(
            start,
            start + CHUNK,
            &self.chunks,
            View {
                chunk,
                offset: 0,
                protection: PAGE_READWRITE,
            },
        );
    }

    /// Read a `T` at guest address `address` through the guest view.
    #[inline(always)]
    pub(crate) fn load<T: Copy>(&self, address: u64) -> T {
        if address > self.size - std::mem::size_of::<T>() as u64 {
            outside_guest_view(address);
        }
        // SAFETY: the access lies within the guest reservation; the guest
        // view's protections fault on anything the guest may not read.
        unsafe { std::ptr::read_unaligned(self.guest.base.as_ptr().add(address as usize).cast()) }
    }

    /// Write `value` at guest address `address` through the guest view.
    #[inline(always)]
    pub(crate) fn store<T: Copy>(&mut self, address: u64, value: T) {
        if address > self.size - std::mem::size_of::<T>() as u64 {
            outside_guest_view(address);
        }
        // SAFETY: as in `load`, for writes.
        unsafe {
            std::ptr::write_unaligned(self.guest.base.as_ptr().add(address as usize).cast(), value)
        }
    }

    /// Generated code's path into the guest view.
    pub(crate) fn guest_access(&self) -> super::GuestAccess {
        super::GuestAccess::new(self.guest.base.as_ptr() as usize, self.size)
    }

    /// The page at file offset `page` in the physical view.
    #[inline(always)]
    pub(crate) fn page(&self, page: u64) -> &[u8; PAGE_SIZE as usize] {
        debug_assert!(page + PAGE_SIZE <= self.size);
        if self.chunks[(page / CHUNK) as usize].is_none() {
            return &ZERO_PAGE;
        }
        // SAFETY: the chunk's physical view is mapped read-write for as
        // long as the chunk exists; `&self` borrows it shared.
        unsafe { &*self.physical.base.as_ptr().add(page as usize).cast() }
    }

    /// The page at file offset `page`, writable.
    #[inline(always)]
    pub(crate) fn page_mut(&mut self, page: u64) -> &mut [u8; PAGE_SIZE as usize] {
        debug_assert!(page + PAGE_SIZE <= self.size);
        self.ensure_chunk((page / CHUNK) as usize);
        // SAFETY: as in `page`; `&mut self` makes the borrow exclusive.
        unsafe { &mut *self.physical.base.as_ptr().add(page as usize).cast() }
    }

    /// Host address of the physical view's page at file offset `page`.
    /// Valid once the page has been written through `page_mut`.
    pub(crate) fn page_address(&self, page: u64) -> usize {
        debug_assert!(self.chunks[(page / CHUNK) as usize].is_some());
        self.physical.base.as_ptr() as usize + page as usize
    }

    /// No file a device can share by offset here.
    pub(crate) fn shared_file(&self) -> Option<i32> {
        None
    }

    /// Zero file range `[start, end)`. Whole chunks return their storage
    /// to the host; partial chunks are zeroed in place.
    pub(crate) fn release(&mut self, start: u64, end: u64) {
        let end = end.min(self.size);
        if start >= end {
            return;
        }
        for chunk in (start / CHUNK) as usize..end.div_ceil(CHUNK) as usize {
            if self.chunks[chunk].is_none() {
                continue;
            }
            let chunk_start = chunk as u64 * CHUNK;
            let chunk_end = (chunk_start + CHUNK).min(self.size);
            let low = start.max(chunk_start);
            let high = end.min(chunk_end);
            if low == chunk_start && high == chunk_end {
                self.replace_chunk(chunk);
            } else {
                // SAFETY: the range lies in this chunk's read-write
                // physical view, which `&mut self` holds exclusively.
                unsafe {
                    std::ptr::write_bytes(
                        self.physical.base.as_ptr().add(low as usize),
                        0,
                        (high - low) as usize,
                    )
                };
            }
        }
    }

    /// Drop chunk `chunk`'s section, so its storage returns to the host. Guest
    /// views of it are moved to a fresh, zeroed section.
    fn replace_chunk(&mut self, chunk: usize) {
        let views = self.guest.views_of(chunk);
        for &(start, _, _) in &views {
            self.guest.unmap_view(start);
        }
        self.physical.unmap_view(chunk as u64 * CHUNK);
        self.chunks[chunk] = None;
        if views.is_empty() {
            return;
        }
        self.ensure_chunk(chunk);
        for (start, end, view) in views {
            self.guest.map_view(start, end, &self.chunks, view);
        }
    }

    /// Map guest range `[address, address + length)` to file offset
    /// `backing` with `access` (READ/WRITE bits), or make it inaccessible
    /// when `access` has no READ.
    pub(crate) fn map_guest(
        &mut self,
        address: u64,
        length: u64,
        backing: Option<u64>,
        access: u8,
    ) {
        let end = address + length;
        let Some(backing) = backing.filter(|_| access & crate::memory::READ != 0) else {
            self.guest.carve(address, end, &self.chunks);
            return;
        };
        let protection = if access & crate::memory::WRITE != 0 {
            PAGE_READWRITE
        } else {
            PAGE_READONLY
        };
        // One view per chunk the backing range touches.
        let (mut cursor, mut offset) = (address, backing);
        while cursor < end {
            let chunk = (offset / CHUNK) as usize;
            let chunk_start = chunk as u64 * CHUNK;
            let length = (end - cursor).min(chunk_start + CHUNK - offset);
            self.ensure_chunk(chunk);
            self.guest.carve(cursor, cursor + length, &self.chunks);
            self.guest.map_view(
                cursor,
                cursor + length,
                &self.chunks,
                View {
                    chunk,
                    offset: offset - chunk_start,
                    protection,
                },
            );
            cursor += length;
            offset += length;
        }
    }
}

#[cold]
#[inline(never)]
fn outside_guest_view(address: u64) -> ! {
    panic!("guest memory access at {address:#x} lies outside the guest address space")
}

/// The most recent guest view, for the fault report.
static GUEST_VIEW: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static GUEST_VIEW_SIZE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static REPORTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// An access violation inside the guest view is a guest access the guest may
/// not make (unmapped, protected or translated code): report the guest
/// address once and let the exception continue to its usual end. Other
/// exceptions are untouched.
fn install_fault_report(base: usize, size: u64) {
    use std::sync::atomic::Ordering;
    GUEST_VIEW.store(base, Ordering::Relaxed);
    GUEST_VIEW_SIZE.store(size, Ordering::Relaxed);
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        // SAFETY: the handler only reads the exception record and atomics
        // and writes to stderr.
        let handle = unsafe { AddVectoredExceptionHandler(1, Some(report_fault)) };
        assert!(!handle.is_null(), "AddVectoredExceptionHandler failed");
    });
}

unsafe extern "system" fn report_fault(info: *mut EXCEPTION_POINTERS) -> i32 {
    use std::sync::atomic::Ordering;
    use windows_sys::Win32::Storage::FileSystem::WriteFile;
    use windows_sys::Win32::System::Console::{GetStdHandle, STD_ERROR_HANDLE};
    // SAFETY: the system passes valid exception pointers to the handler.
    let record = unsafe { &*(*info).ExceptionRecord };
    if record.ExceptionCode != EXCEPTION_ACCESS_VIOLATION || record.NumberParameters < 2 {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    let address = record.ExceptionInformation[1];
    let base = GUEST_VIEW.load(Ordering::Relaxed);
    let size = GUEST_VIEW_SIZE.load(Ordering::Relaxed) as usize;
    if base != 0
        && address >= base
        && address - base < size
        && !REPORTED.swap(true, Ordering::Relaxed)
    {
        let mut message = *b"guest memory fault at 0x0000000000000000\n";
        let guest = (address - base) as u64;
        for index in 0..16 {
            let digit = (guest >> (60 - 4 * index)) & 0xf;
            message[24 + index] = b"0123456789abcdef"[digit as usize];
        }
        let mut written = 0;
        // SAFETY: a plain write of a valid buffer to the stderr handle.
        unsafe {
            WriteFile(
                GetStdHandle(STD_ERROR_HANDLE),
                message.as_ptr(),
                message.len() as u32,
                &mut written,
                std::ptr::null_mut(),
            )
        };
    }
    EXCEPTION_CONTINUE_SEARCH
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::memory::{READ, WRITE};

    #[test]
    fn views_split_alias_and_release_at_page_granularity() {
        let size = 4 * CHUNK;
        let mut host = HostMemory::new(size);
        // A run that crosses a chunk boundary, then a page carved out of it.
        host.map_guest(CHUNK - 0x2000, 0x4000, Some(CHUNK - 0x2000), READ | WRITE);
        host.store(CHUNK - 0x1004, 0x1122_3344_u32);
        host.store(CHUNK - 4, 0xaabb_ccdd_eeff_0011_u64); // across the chunks
        assert_eq!(host.load::<u64>(CHUNK - 4), 0xaabb_ccdd_eeff_0011);
        assert_eq!(host.page(CHUNK - 0x1000)[0xffc..], [0x11, 0x00, 0xff, 0xee]);
        host.map_guest(CHUNK - 0x1000, 0x1000, Some(CHUNK - 0x1000), READ);
        assert_eq!(host.load::<u32>(CHUNK - 0x1004), 0x1122_3344);
        // An alias of the first page at a distant guest address.
        host.map_guest(
            3 * CHUNK + 0x5000,
            0x1000,
            Some(CHUNK - 0x2000),
            READ | WRITE,
        );
        host.store(3 * CHUNK + 0x5ffc, 7_u32);
        assert_eq!(host.load::<u32>(CHUNK - 0x1004), 7);
        // Releasing a partial range zeroes it everywhere it is seen.
        host.release(CHUNK - 0x2000, CHUNK - 0x1000);
        assert_eq!(host.load::<u32>(3 * CHUNK + 0x5ffc), 0);
        // Releasing a whole chunk keeps its guest views, now zero.
        host.page_mut(CHUNK)[0] = 9;
        assert_eq!(host.load::<u8>(CHUNK), 9);
        host.release(CHUNK, 2 * CHUNK);
        assert_eq!(host.load::<u8>(CHUNK), 0);
        assert_eq!(host.page(CHUNK)[0], 0);
        // Unmapping restores an inaccessible placeholder, and the layout
        // record matches by remapping across every boundary.
        host.map_guest(0, size, None, 0);
        assert_eq!(host.guest.regions.len(), 2);
        host.map_guest(0, size, Some(0), READ | WRITE);
        host.store(2 * CHUNK + 0x10, 5_u8);
        assert_eq!(host.page(2 * CHUNK)[0x10], 5);
        assert!(host.chunks.iter().all(Option::is_some));
    }

    #[test]
    fn untouched_chunks_have_no_section() {
        let host = HostMemory::new(1 << 39);
        assert!(host.chunks.iter().all(Option::is_none));
        assert_eq!(host.page(0x1234_5000)[0], 0);
    }
}
