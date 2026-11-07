//! Checked backend for hosts whose pages are larger than the guest's 4 KiB
//! (Apple silicon uses 16 KiB): no host guest view, but a table that
//! translates each guest page.
//!
//! A host page larger than a guest page cannot give each guest page its own
//! backing and permissions, so generated code cannot load and store through
//! host page protections here. Instead every guest page has one entry in a
//! translation table: the host address of its backing page in the physical
//! view (canonical guest page addresses, as on the other backends) plus
//! READ/WRITE bits. Each access looks its page up, checks the permission and
//! reaches the backing bytes; an access the guest may not make prints the
//! same `guest memory fault at` report as the other backends and aborts the
//! process. Accesses that cross a guest page boundary are split, because
//! neighbouring guest pages need not have neighbouring backing.
//!
//! Both the physical view and the table are lazily allocated anonymous
//! memory, so untouched pages cost nothing. The backend uses only portable
//! `mmap`; building with `--cfg echora_checked_memory` selects it on Linux
//! too, which is how it is tested there.
#![allow(unsafe_code)]

use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::memory::PAGE_SIZE;

const PAGE_MASK: u64 = PAGE_SIZE - 1;
const ENTRY_READ: u64 = 1;
const ENTRY_WRITE: u64 = 2;

/// Generated code's path into guest memory: the translation table and the
/// address space size.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GuestAccess {
    /// `*const AtomicU64`, one entry per guest page.
    table: usize,
    size: u64,
    /// Bits an in-range address may have (the address space size rounded up
    /// to a power of two, minus one), as on the other backends.
    mask: u64,
}

impl GuestAccess {
    /// Checked memory has no contiguous host view to share.
    pub(crate) fn host_span(&self, _address: u64, _length: usize) -> Option<*mut u8> {
        None
    }

    fn new(table: usize, size: u64) -> Self {
        Self {
            table,
            size,
            mask: size.next_power_of_two() - 1,
        }
    }

    /// The host address of guest `address`, which the guest must be allowed
    /// to read (or, with `write`, to write). A forbidden access ends the
    /// process with the guest address reported.
    #[inline(always)]
    fn translate(&self, address: u64, write: bool) -> usize {
        if address & !self.mask != 0 {
            super::outside(address);
        }
        if address >= self.size {
            fault(address);
        }
        // SAFETY: the table has an entry for every page below `size` and
        // lives as long as the `HostMemory` that the view's `Memory` owns.
        let entry = unsafe { &*(self.table as *const AtomicU64).add((address >> 12) as usize) }
            .load(Ordering::Relaxed);
        let needed = if write { ENTRY_WRITE } else { ENTRY_READ };
        if entry & needed == 0 {
            fault(address);
        }
        (entry & !PAGE_MASK) as usize + (address & PAGE_MASK) as usize
    }

    /// Run `each(host, done, length)` for each guest-page piece of the
    /// `length` bytes at `address`, translating each piece.
    #[inline]
    fn pieces(
        &self,
        address: u64,
        length: usize,
        write: bool,
        mut each: impl FnMut(usize, usize, usize),
    ) {
        let mut done = 0;
        while done < length {
            let at = address.wrapping_add(done as u64);
            let piece = (length - done).min((PAGE_SIZE - (at & PAGE_MASK)) as usize);
            each(self.translate(at, write), done, piece);
            done += piece;
        }
    }

    #[inline(always)]
    fn fits_in_page<T>(address: u64) -> bool {
        (address & PAGE_MASK) + std::mem::size_of::<T>() as u64 <= PAGE_SIZE
    }

    #[inline(always)]
    pub(crate) fn load<T: Copy>(&self, address: u64) -> T {
        if Self::fits_in_page::<T>(address) {
            // SAFETY: the translated bytes lie in one readable backing page.
            return unsafe { std::ptr::read_unaligned(self.translate(address, false) as *const T) };
        }
        self.load_split(address)
    }

    #[cold]
    fn load_split<T: Copy>(&self, address: u64) -> T {
        let mut value = std::mem::MaybeUninit::<T>::uninit();
        let out = value.as_mut_ptr().cast::<u8>();
        self.pieces(
            address,
            std::mem::size_of::<T>(),
            false,
            |host, done, length| {
                // SAFETY: each piece lies in one readable backing page and in
                // `value`.
                unsafe { std::ptr::copy_nonoverlapping(host as *const u8, out.add(done), length) };
            },
        );
        // SAFETY: every byte of `value` was written; `T` is plain data.
        unsafe { value.assume_init() }
    }

    #[inline(always)]
    pub(crate) fn store<T: Copy>(&self, address: u64, value: T) {
        if Self::fits_in_page::<T>(address) {
            // SAFETY: the translated bytes lie in one writable backing page.
            unsafe { std::ptr::write_unaligned(self.translate(address, true) as *mut T, value) };
            return;
        }
        self.store_split(address, value);
    }

    #[cold]
    fn store_split<T: Copy>(&self, address: u64, value: T) {
        let bytes = (&raw const value).cast::<u8>();
        // Check every piece before writing any, as one host store would.
        self.pieces(address, std::mem::size_of::<T>(), true, |_, _, _| {});
        self.pieces(
            address,
            std::mem::size_of::<T>(),
            true,
            |host, done, length| {
                // SAFETY: each piece lies in one writable backing page and in
                // `value`.
                unsafe { std::ptr::copy_nonoverlapping(bytes.add(done), host as *mut u8, length) };
            },
        );
    }

    /// Host address of an aligned atomic access of up to 16 bytes, which
    /// therefore lies in one guest page.
    #[inline(always)]
    pub(crate) fn atomic(&self, address: u64, write: bool) -> *mut u8 {
        self.translate(address, write) as *mut u8
    }

    /// Copy guest bytes at `address` into `out`; false when they leave the
    /// guest address space.
    #[inline]
    pub(crate) fn read(&self, address: u64, out: &mut [u8]) -> bool {
        if self
            .size
            .checked_sub(out.len() as u64)
            .is_none_or(|last| address > last)
        {
            return false;
        }
        let target = out.as_mut_ptr();
        self.pieces(address, out.len(), false, |host, done, length| {
            // SAFETY: each piece lies in one readable backing page and in `out`.
            unsafe { std::ptr::copy_nonoverlapping(host as *const u8, target.add(done), length) };
        });
        true
    }

    /// Copy `bytes` to guest `address`; false when they leave the guest
    /// address space.
    #[inline]
    pub(crate) fn write(&self, address: u64, bytes: &[u8]) -> bool {
        if self
            .size
            .checked_sub(bytes.len() as u64)
            .is_none_or(|last| address > last)
        {
            return false;
        }
        self.pieces(address, bytes.len(), true, |host, done, length| {
            // SAFETY: each piece lies in one writable backing page and in
            // `bytes`.
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr().add(done), host as *mut u8, length)
            };
        });
        true
    }
}

/// A guest access the guest may not make: report the guest address the way
/// the other backends' fault handlers do, then end the process.
#[cold]
#[inline(never)]
fn fault(address: u64) -> ! {
    eprintln!("guest memory fault at {address:#018x}");
    std::process::abort()
}

fn host_page_size() -> u64 {
    // SAFETY: sysconf has no preconditions.
    let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    assert!(size > 0, "sysconf(_SC_PAGESIZE) failed");
    size as u64
}

/// Lazily allocated, zero-filled anonymous memory of `length` bytes at
/// `address` (or anywhere when null).
fn anonymous(address: *mut libc::c_void, length: u64) -> NonNull<u8> {
    #[cfg(target_os = "linux")]
    let flags = libc::MAP_NORESERVE;
    #[cfg(not(target_os = "linux"))]
    let flags = 0;
    let fixed = if address.is_null() {
        0
    } else {
        libc::MAP_FIXED
    };
    // SAFETY: an anonymous mapping; with MAP_FIXED, callers replace only
    // their own mapping.
    let mapped = unsafe {
        libc::mmap(
            address,
            length as usize,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON | flags | fixed,
            -1,
            0,
        )
    };
    assert!(mapped != libc::MAP_FAILED, "guest memory allocation failed");
    NonNull::new(mapped.cast()).expect("mmap returned null")
}

pub(crate) struct HostMemory {
    size: u64,
    host_page: u64,
    physical: NonNull<u8>,
    physical_length: u64,
    table: NonNull<AtomicU64>,
    table_length: u64,
}

// SAFETY: the views are plain memory owned by this object; access is
// synchronized by the `Memory` that owns it, and the table is atomic.
unsafe impl Send for HostMemory {}
unsafe impl Sync for HostMemory {}

impl HostMemory {
    /// A zero-filled memory file of `size` bytes and an empty translation
    /// table.
    pub(crate) fn new(size: u64) -> Self {
        let host_page = host_page_size();
        let physical_length = size.next_multiple_of(host_page);
        let table_length = (size.div_ceil(PAGE_SIZE) * 8).next_multiple_of(host_page);
        Self {
            size,
            host_page,
            physical: anonymous(std::ptr::null_mut(), physical_length),
            physical_length,
            table: anonymous(std::ptr::null_mut(), table_length).cast(),
            table_length,
        }
    }

    pub(crate) fn guest_access(&self) -> GuestAccess {
        GuestAccess::new(self.table.as_ptr() as usize, self.size)
    }

    /// Read a `T` at guest address `address` with the guest's permissions.
    #[inline(always)]
    pub(crate) fn load<T: Copy>(&self, address: u64) -> T {
        if address > self.size - std::mem::size_of::<T>() as u64 {
            super::outside(address);
        }
        self.guest_access().load(address)
    }

    /// Write `value` at guest address `address` with the guest's permissions.
    #[inline(always)]
    pub(crate) fn store<T: Copy>(&mut self, address: u64, value: T) {
        if address > self.size - std::mem::size_of::<T>() as u64 {
            super::outside(address);
        }
        self.guest_access().store(address, value)
    }

    /// The page at file offset `page` in the physical view.
    #[inline(always)]
    pub(crate) fn page(&self, page: u64) -> &[u8; PAGE_SIZE as usize] {
        debug_assert!(page + PAGE_SIZE <= self.size);
        // SAFETY: the physical view covers the whole file, readable and
        // writable for the object's lifetime; `&self` borrows it shared.
        unsafe { &*self.physical.as_ptr().add(page as usize).cast() }
    }

    /// The page at file offset `page`, writable.
    #[inline(always)]
    pub(crate) fn page_mut(&mut self, page: u64) -> &mut [u8; PAGE_SIZE as usize] {
        debug_assert!(page + PAGE_SIZE <= self.size);
        // SAFETY: as in `page`; `&mut self` makes the borrow exclusive.
        unsafe { &mut *self.physical.as_ptr().add(page as usize).cast() }
    }

    /// Host address of the physical view's page at file offset `page`.
    pub(crate) fn page_address(&self, page: u64) -> usize {
        self.physical.as_ptr() as usize + page as usize
    }

    /// Zero file range `[start, end)`. Whole host pages inside it are
    /// replaced with fresh anonymous memory, returning their storage; the
    /// guest-page edges are zeroed in place.
    pub(crate) fn release(&mut self, start: u64, end: u64) {
        let end = end.min(self.size);
        if start >= end {
            return;
        }
        let inner_start = start.next_multiple_of(self.host_page);
        let inner_end = end & !(self.host_page - 1);
        let zero = |from: u64, to: u64| {
            if from < to {
                // SAFETY: the range lies in the physical view, which
                // `&mut self` holds exclusively.
                unsafe {
                    std::ptr::write_bytes(
                        self.physical.as_ptr().add(from as usize),
                        0,
                        (to - from) as usize,
                    )
                };
            }
        };
        if inner_start < inner_end {
            zero(start, inner_start);
            anonymous(
                self.physical
                    .as_ptr()
                    .wrapping_add(inner_start as usize)
                    .cast(),
                inner_end - inner_start,
            );
            zero(inner_end, end);
        } else {
            zero(start, end);
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
        let backing = backing.filter(|_| access & crate::memory::READ != 0);
        let mut flags = ENTRY_READ;
        if access & crate::memory::WRITE != 0 {
            flags |= ENTRY_WRITE;
        }
        for page in 0..length / PAGE_SIZE {
            let entry = backing.map_or(0, |backing| {
                self.page_address(backing + page * PAGE_SIZE) as u64 | flags
            });
            let index = (address / PAGE_SIZE + page) as usize;
            // SAFETY: the guest range lies in the address space, which the
            // table covers.
            unsafe { &*self.table.as_ptr().add(index) }.store(entry, Ordering::Relaxed);
        }
    }
}

impl Drop for HostMemory {
    fn drop(&mut self) {
        // SAFETY: both mappings are ours and unused now.
        unsafe {
            libc::munmap(self.physical.as_ptr().cast(), self.physical_length as usize);
            libc::munmap(self.table.as_ptr().cast(), self.table_length as usize);
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::memory::{READ, WRITE};

    #[test]
    fn accesses_follow_aliases_split_at_pages_and_release_zeroes() {
        let mut host = HostMemory::new(1 << 24);
        let access = host.guest_access();
        // Two guest pages whose backings are far apart, so an access across
        // them must be split.
        host.map_guest(0x10000, 0x1000, Some(0x10000), READ | WRITE);
        host.map_guest(0x11000, 0x1000, Some(0x80000), READ | WRITE);
        access.store(0x10ffc, 0x1122_3344_5566_7788_u64);
        assert_eq!(access.load::<u64>(0x10ffc), 0x1122_3344_5566_7788);
        assert_eq!(host.page(0x10000)[0xffc..], [0x88, 0x77, 0x66, 0x55]);
        assert_eq!(host.page(0x80000)[..4], [0x44, 0x33, 0x22, 0x11]);
        let mut bytes = [0; 8];
        assert!(access.read(0x10ffc, &mut bytes));
        assert_eq!(u64::from_le_bytes(bytes), 0x1122_3344_5566_7788);
        assert!(!access.read((1 << 24) - 4, &mut bytes));
        // An alias sees the same bytes.
        host.map_guest(0x40000, 0x1000, Some(0x80000), READ);
        assert_eq!(access.load::<u32>(0x40000), 0x1122_3344);
        // A partial release zeroes in place; a release covering whole host
        // pages replaces them.
        host.release(0x80000, 0x80002);
        assert_eq!(access.load::<u32>(0x40000), 0x1122_0000);
        host.page_mut(0x100000)[7] = 9;
        host.release(0x100000 - 0x1000, 0x100000 + 2 * host.host_page);
        assert_eq!(host.page(0x100000)[7], 0);
    }
}
