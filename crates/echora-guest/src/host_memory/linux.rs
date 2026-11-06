//! Linux backend: guest RAM as one sparse memfd seen through two host views.
//!
//! Adapted from pokopia-recomp `crates/guest/src/host_memory.rs`.
//!
//! The file's offsets are canonical guest page addresses (the backing page
//! of an alias is its source). The physical view maps the whole file
//! read-write for host-side accesses, which follow their own permission
//! checks. The guest view reserves the guest address space at
//! `guest + address`: each mapped page there maps its backing page with the
//! guest's permissions (aliases map the source's file range again), and
//! everything else stays inaccessible. Generated code loads and stores
//! through the guest view directly; an access the guest may not make faults
//! in the host, where `install_fault_report` names the guest address.
#![allow(unsafe_code)]

use std::ptr::NonNull;

pub(crate) struct HostMemory {
    fd: libc::c_int,
    size: u64,
    physical: NonNull<u8>,
    guest: NonNull<u8>,
}

// SAFETY: the views are plain memory owned by this object; access is
// synchronized by the `Memory` that owns it.
unsafe impl Send for HostMemory {}
unsafe impl Sync for HostMemory {}

/// Bytes of the guest view's reservation for an address space of `size`
/// bytes: the next power of two, so generated code tests an address with one
/// mask (`GuestView::at`), plus an inaccessible guard for accesses that
/// start in range and run past its end.
pub(crate) fn guest_reservation(size: u64) -> u64 {
    size.next_power_of_two() + GUARD_BYTES
}

const GUARD_BYTES: u64 = 1 << 20;

fn reserve(size: u64, protection: libc::c_int, flags: libc::c_int, fd: libc::c_int) -> NonNull<u8> {
    // SAFETY: a fresh mapping at a kernel-chosen address.
    let address = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            size as usize,
            protection,
            flags | libc::MAP_NORESERVE,
            fd,
            0,
        )
    };
    assert!(
        address != libc::MAP_FAILED,
        "guest memory reservation failed"
    );
    NonNull::new(address.cast()).expect("mmap returned null")
}

impl HostMemory {
    /// A zero-filled file of `size` bytes and both views of it.
    pub(crate) fn new(size: u64) -> Self {
        // SAFETY: sysconf has no preconditions.
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        assert_eq!(
            page_size, 4096,
            "guest memory needs 4 KiB host pages; this host uses {page_size}-byte pages"
        );
        // SAFETY: the name is a valid C string; the descriptor is owned here.
        let fd = unsafe { libc::memfd_create(c"echora-guest".as_ptr(), libc::MFD_CLOEXEC) };
        assert!(fd >= 0, "memfd_create failed");
        // SAFETY: `fd` is our file; the size is sparse until written.
        assert_eq!(
            unsafe { libc::ftruncate(fd, size as libc::off_t) },
            0,
            "ftruncate failed"
        );
        let physical = reserve(
            size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            fd,
        );
        let guest = reserve(
            guest_reservation(size),
            libc::PROT_NONE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
        );
        // Keep both views out of core dumps. The memfd is shared memory, which
        // the default dump filter includes, and dumping it would fault in
        // every page of the (sparse, address-space-sized) file.
        for (view, length) in [(physical, size), (guest, guest_reservation(size))] {
            // SAFETY: advice on our own mappings; it changes no contents.
            let result = unsafe {
                libc::madvise(view.as_ptr().cast(), length as usize, libc::MADV_DONTDUMP)
            };
            assert_eq!(result, 0, "madvise(MADV_DONTDUMP) failed");
        }
        install_fault_report(guest.as_ptr() as usize, guest_reservation(size));
        Self {
            fd,
            size,
            physical,
            guest,
        }
    }

    /// Read a `T` at guest address `address` through the guest view.
    #[inline(always)]
    pub(crate) fn load<T: Copy>(&self, address: u64) -> T {
        if address > self.size - std::mem::size_of::<T>() as u64 {
            outside_guest_view(address);
        }
        // SAFETY: the access lies within the guest reservation; the guest
        // view's protections fault on anything the guest may not read.
        unsafe { std::ptr::read_unaligned(self.guest.as_ptr().add(address as usize).cast()) }
    }

    /// Write `value` at guest address `address` through the guest view.
    #[inline(always)]
    pub(crate) fn store<T: Copy>(&mut self, address: u64, value: T) {
        if address > self.size - std::mem::size_of::<T>() as u64 {
            outside_guest_view(address);
        }
        // SAFETY: as in `load`, for writes.
        unsafe {
            std::ptr::write_unaligned(self.guest.as_ptr().add(address as usize).cast(), value)
        }
    }

    /// Host address of the guest view's first byte.
    pub(crate) fn guest_base(&self) -> usize {
        self.guest.as_ptr() as usize
    }

    /// The page at file offset `page` in the physical view.
    #[inline(always)]
    pub(crate) fn page(&self, page: u64) -> &[u8; crate::memory::PAGE_SIZE as usize] {
        debug_assert!(page + crate::memory::PAGE_SIZE <= self.size);
        // SAFETY: the physical view covers the whole file, readable and
        // writable for the object's lifetime; `&self` borrows it shared.
        unsafe { &*self.physical.as_ptr().add(page as usize).cast() }
    }

    /// The page at file offset `page`, writable.
    #[inline(always)]
    pub(crate) fn page_mut(&mut self, page: u64) -> &mut [u8; crate::memory::PAGE_SIZE as usize] {
        debug_assert!(page + crate::memory::PAGE_SIZE <= self.size);
        // SAFETY: as in `page`; `&mut self` makes the borrow exclusive.
        unsafe { &mut *self.physical.as_ptr().add(page as usize).cast() }
    }

    /// Host address of the physical view's page at file offset `page`.
    pub(crate) fn page_address(&self, page: u64) -> usize {
        self.physical.as_ptr() as usize + page as usize
    }

    /// Zero file range `[start, end)` and return its storage to the host.
    pub(crate) fn release(&mut self, start: u64, end: u64) {
        if start >= end {
            return;
        }
        // SAFETY: `fd` is our file; punching a hole only zeroes its bytes.
        let result = unsafe {
            libc::fallocate(
                self.fd,
                libc::FALLOC_FL_PUNCH_HOLE | libc::FALLOC_FL_KEEP_SIZE,
                start as libc::off_t,
                (end - start) as libc::off_t,
            )
        };
        assert_eq!(result, 0, "fallocate punch hole failed");
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
        let target = self.guest.as_ptr().wrapping_add(address as usize).cast();
        let mapped = match backing {
            Some(offset) if access & crate::memory::READ != 0 => {
                let mut protection = libc::PROT_READ;
                if access & crate::memory::WRITE != 0 {
                    protection |= libc::PROT_WRITE;
                }
                // SAFETY: the target lies in our guest reservation; MAP_FIXED
                // replaces only our own mapping there.
                unsafe {
                    libc::mmap(
                        target,
                        length as usize,
                        protection,
                        libc::MAP_SHARED | libc::MAP_FIXED,
                        self.fd,
                        offset as libc::off_t,
                    )
                }
            }
            // SAFETY: as above, with an inaccessible anonymous mapping.
            _ => unsafe {
                libc::mmap(
                    target,
                    length as usize,
                    libc::PROT_NONE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_FIXED | libc::MAP_NORESERVE,
                    -1,
                    0,
                )
            },
        };
        assert!(mapped != libc::MAP_FAILED, "guest view mapping failed");
        // A fresh mapping does not inherit the view's no-dump advice.
        // SAFETY: advice on the mapping just created.
        let result = unsafe { libc::madvise(mapped, length as usize, libc::MADV_DONTDUMP) };
        assert_eq!(result, 0, "madvise(MADV_DONTDUMP) failed");
    }
}

#[cold]
#[inline(never)]
fn outside_guest_view(address: u64) -> ! {
    panic!("guest memory access at {address:#x} lies outside the guest address space")
}

impl Drop for HostMemory {
    fn drop(&mut self) {
        // SAFETY: both views and the descriptor are ours and unused now.
        unsafe {
            libc::munmap(self.physical.as_ptr().cast(), self.size as usize);
            libc::munmap(
                self.guest.as_ptr().cast(),
                guest_reservation(self.size) as usize,
            );
            libc::close(self.fd);
        }
    }
}

/// The most recent guest view, for the fault report.
static GUEST_VIEW: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static GUEST_VIEW_SIZE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A host fault inside the guest view is a guest access the guest may not
/// make (unmapped, protected or translated code): report the guest address
/// and let the default action end the process. Other faults are untouched.
fn install_fault_report(base: usize, size: u64) {
    use std::sync::atomic::Ordering;
    GUEST_VIEW.store(base, Ordering::Relaxed);
    GUEST_VIEW_SIZE.store(size, Ordering::Relaxed);
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        // SAFETY: installing a handler that only reads atomics, formats into
        // a stack buffer, writes to stderr and restores the default action.
        unsafe {
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = report_fault as *const () as usize;
            action.sa_flags = libc::SA_SIGINFO | libc::SA_RESETHAND;
            libc::sigaction(libc::SIGSEGV, &action, std::ptr::null_mut());
            libc::sigaction(libc::SIGBUS, &action, std::ptr::null_mut());
        }
    });
}

extern "C" fn report_fault(_: libc::c_int, info: *mut libc::siginfo_t, _: *mut libc::c_void) {
    use std::sync::atomic::Ordering;
    // SAFETY: the kernel passes a valid siginfo for SA_SIGINFO handlers.
    let address = unsafe { (*info).si_addr() } as usize;
    let base = GUEST_VIEW.load(Ordering::Relaxed);
    let size = GUEST_VIEW_SIZE.load(Ordering::Relaxed) as usize;
    if base != 0 && address >= base && address - base < size {
        let mut message = *b"guest memory fault at 0x0000000000000000\n";
        let guest = (address - base) as u64;
        for index in 0..16 {
            let digit = (guest >> (60 - 4 * index)) & 0xf;
            message[24 + index] = b"0123456789abcdef"[digit as usize];
        }
        // SAFETY: write(2) is async-signal-safe; the buffer is valid.
        unsafe { libc::write(2, message.as_ptr().cast(), message.len()) };
    }
    // SA_RESETHAND restored the default action: returning re-runs the
    // faulting access, which now ends the process with the usual signal.
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// VmFlags of every mapping that overlaps `[start, end)`.
    fn vm_flags(start: usize, end: usize) -> Vec<String> {
        let smaps = std::fs::read_to_string("/proc/self/smaps").unwrap();
        let mut out = Vec::new();
        let mut inside = false;
        for line in smaps.lines() {
            if let Some((range, _)) = line.split_once(' ')
                && let Some((a, b)) = range.split_once('-')
                && let (Ok(a), Ok(b)) = (usize::from_str_radix(a, 16), usize::from_str_radix(b, 16))
            {
                inside = a < end && b > start;
            } else if inside && let Some(flags) = line.strip_prefix("VmFlags:") {
                out.push(flags.trim().to_owned());
            }
        }
        out
    }

    #[test]
    fn views_are_excluded_from_core_dumps() {
        let size = 1 << 24;
        let mut host = HostMemory::new(size);
        host.map_guest(
            0x1000,
            0x2000,
            Some(0x1000),
            crate::memory::READ | crate::memory::WRITE,
        );
        host.map_guest(0x8000, 0x1000, Some(0x8000), crate::memory::READ);
        let guest = host.guest_base();
        let physical = host.page_address(0);
        for (start, len) in [(guest, guest_reservation(size)), (physical, size)] {
            let flags = vm_flags(start, start + len as usize);
            assert!(!flags.is_empty());
            for f in &flags {
                assert!(f.split(' ').any(|x| x == "dd"), "mapping without dd: {f}");
            }
        }
    }
}
