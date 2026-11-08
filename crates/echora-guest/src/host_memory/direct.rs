//! Guest access for backends with a host guest view (Linux, Windows): a
//! guest address is an offset into the view, whose page protections are
//! the guest's permissions, so a forbidden access faults in the host.
#![allow(unsafe_code)]

/// Generated code's path into guest memory: the guest view's base and the
/// address space size.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GuestAccess {
    base: usize,
    size: u64,
    /// Bits an in-range address may have (the reservation's power-of-two
    /// size minus one).
    mask: u64,
}

impl GuestAccess {
    pub(crate) fn new(base: usize, size: u64) -> Self {
        Self {
            base,
            size,
            mask: size.next_power_of_two() - 1,
        }
    }

    /// The host address of an access. One test against the reservation's
    /// power-of-two mask: an access starting past the address space (or
    /// running past its end) lands in the reservation's inaccessible tail
    /// and faults in the host, which reports the guest address.
    #[inline(always)]
    fn at<T>(&self, address: u64) -> *mut T {
        if address & !self.mask != 0 {
            super::outside(address);
        }
        (self.base + address as usize) as *mut T
    }

    /// A guest load. Naturally aligned accesses of up to 8 bytes are
    /// single-copy atomic on AArch64, and other guest threads (workers) may
    /// write the same memory: they are relaxed host atomics, the same
    /// instruction as a plain access, which the compiler may not merge,
    /// split or keep in a register across other code. Misaligned and wider
    /// accesses are plain.
    #[inline(always)]
    pub(crate) fn load<T: Copy>(&self, address: u64) -> T {
        use std::sync::atomic::{AtomicU8, AtomicU16, AtomicU32, AtomicU64, Ordering::Relaxed};
        let pointer = self.at::<T>(address);
        let size = size_of::<T>();
        if size <= 8 && address & (size as u64 - 1) == 0 {
            // SAFETY: inside the guest reservation (the view's protections
            // fault on anything the guest may not read), aligned to `size`
            // (the view's base is page-aligned), and T is `size` bytes.
            unsafe {
                match size {
                    1 => {
                        return std::mem::transmute_copy(
                            &AtomicU8::from_ptr(pointer.cast()).load(Relaxed),
                        );
                    }
                    2 => {
                        return std::mem::transmute_copy(
                            &AtomicU16::from_ptr(pointer.cast()).load(Relaxed),
                        );
                    }
                    4 => {
                        return std::mem::transmute_copy(
                            &AtomicU32::from_ptr(pointer.cast()).load(Relaxed),
                        );
                    }
                    8 => {
                        return std::mem::transmute_copy(
                            &AtomicU64::from_ptr(pointer.cast()).load(Relaxed),
                        );
                    }
                    _ => {}
                }
            }
        }
        // SAFETY: inside the guest reservation, as above.
        unsafe { std::ptr::read_unaligned(pointer) }
    }

    /// A guest store: relaxed host atomics for aligned accesses of up to 8
    /// bytes, as in `load`.
    #[inline(always)]
    pub(crate) fn store<T: Copy>(&self, address: u64, value: T) {
        use std::sync::atomic::{AtomicU8, AtomicU16, AtomicU32, AtomicU64, Ordering::Relaxed};
        let pointer = self.at::<T>(address);
        let size = size_of::<T>();
        if size <= 8 && address & (size as u64 - 1) == 0 {
            // SAFETY: as in `load`, for writes.
            unsafe {
                match size {
                    1 => {
                        return AtomicU8::from_ptr(pointer.cast())
                            .store(std::mem::transmute_copy(&value), Relaxed);
                    }
                    2 => {
                        return AtomicU16::from_ptr(pointer.cast())
                            .store(std::mem::transmute_copy(&value), Relaxed);
                    }
                    4 => {
                        return AtomicU32::from_ptr(pointer.cast())
                            .store(std::mem::transmute_copy(&value), Relaxed);
                    }
                    8 => {
                        return AtomicU64::from_ptr(pointer.cast())
                            .store(std::mem::transmute_copy(&value), Relaxed);
                    }
                    _ => {}
                }
            }
        }
        // SAFETY: as in `load`, for writes.
        unsafe { std::ptr::write_unaligned(pointer, value) }
    }

    /// Host address of an aligned atomic access of up to 16 bytes. The
    /// view's protections enforce read and write permission.
    #[inline(always)]
    pub(crate) fn atomic(&self, address: u64, _write: bool) -> *mut u8 {
        self.at(address)
    }

    /// Host pointer to `length` bytes at `address`, if they lie inside the
    /// guest address space.
    #[inline(always)]
    fn pointer(&self, address: u64, length: usize) -> Option<*mut u8> {
        (address <= self.size.checked_sub(length as u64)?)
            .then(|| (self.base + address as usize) as *mut u8)
    }

    /// Host address of `length` guest bytes at `address`, for sharing them
    /// with a device (Vulkan host-memory import); `None` outside the space.
    pub(crate) fn host_span(&self, address: u64, length: usize) -> Option<*mut u8> {
        self.pointer(address, length)
    }

    /// Copy guest bytes at `address` into `out`; false when they leave the
    /// guest address space.
    #[inline]
    pub(crate) fn read(&self, address: u64, out: &mut [u8]) -> bool {
        let Some(pointer) = self.pointer(address, out.len()) else {
            return false;
        };
        // SAFETY: as in `load`, for `out.len()` bytes.
        unsafe { std::ptr::copy_nonoverlapping(pointer, out.as_mut_ptr(), out.len()) };
        true
    }

    /// Copy `bytes` to guest `address`; false when they leave the guest
    /// address space.
    #[inline]
    pub(crate) fn write(&self, address: u64, bytes: &[u8]) -> bool {
        let Some(pointer) = self.pointer(address, bytes.len()) else {
            return false;
        };
        // SAFETY: as in `store`, for `bytes.len()` bytes.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), pointer, bytes.len()) };
        true
    }
}
