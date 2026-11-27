//! Guest access for backends with a host guest view (Linux, Windows): a
//! guest address is an offset into the view, whose page protections are
//! the guest's permissions, so a forbidden access faults in the host.
#![allow(unsafe_code)]

/// A value read at any alignment (`GuestAccess::load` on x86-64).
#[cfg(target_arch = "x86_64")]
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct Unaligned<T>(T);

/// Generated code's path into guest memory: the guest view's base and the
/// address space size.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GuestAccess {
    base: usize,
    size: u64,
    /// The last offset in the power-of-two reservation, including the
    /// inaccessible tail beyond the guest address space.
    last_offset: u64,
}

impl GuestAccess {
    pub(crate) fn new(base: usize, size: u64) -> Self {
        Self {
            base,
            size,
            last_offset: size.next_power_of_two() - 1,
        }
    }

    /// The host address of an access. A comparison rejects addresses
    /// outside the reservation. Accesses in its inaccessible tail fault
    /// through the host page protections, which report the guest address.
    #[inline(always)]
    fn at<T>(&self, address: u64) -> *mut T {
        if address > self.last_offset {
            super::outside(address);
        }
        (self.base + address as usize) as *mut T
    }

    /// A guest load. Naturally aligned accesses of up to 8 bytes are
    /// single-copy atomic on AArch64, and other guest threads (workers) may
    /// write the same memory: the compiler may not merge, split or keep
    /// them in a register across other code. On x86-64 every load of up to
    /// 8 bytes is one volatile `mov` at any alignment (`Unaligned`), which
    /// is single-copy atomic when aligned, as a relaxed atomic is there,
    /// with no branch on the alignment; elsewhere aligned loads are relaxed
    /// host atomics. Misaligned loads elsewhere, and wider ones, are plain.
    #[inline(always)]
    pub(crate) fn load<T: Copy>(&self, address: u64) -> T {
        use std::sync::atomic::{AtomicU8, AtomicU16, AtomicU32, AtomicU64, Ordering::Relaxed};
        let pointer = self.at::<T>(address);
        let size = size_of::<T>();
        #[cfg(target_arch = "x86_64")]
        if size <= 8 {
            // SAFETY: inside the guest reservation (the view's protections
            // fault on anything the guest may not read); `Unaligned<T>` has
            // T's size and alignment 1.
            return unsafe { std::ptr::read_volatile(pointer.cast::<Unaligned<T>>()).0 };
        }
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

    /// A guest store, as `load`: on x86-64 one `mov` at any alignment (an
    /// `asm!` store at the base plus the address, which the compiler may
    /// not merge or drop; a volatile store of a packed value goes through
    /// the stack), elsewhere relaxed host atomics for aligned accesses of up
    /// to 8 bytes.
    #[inline(always)]
    pub(crate) fn store<T: Copy>(&self, address: u64, value: T) {
        use std::sync::atomic::{AtomicU8, AtomicU16, AtomicU32, AtomicU64, Ordering::Relaxed};
        let pointer = self.at::<T>(address);
        let size = size_of::<T>();
        #[cfg(target_arch = "x86_64")]
        if size <= 8 {
            // SAFETY: inside the guest reservation (the view's protections
            // fault on anything the guest may not write, as `pointer` was
            // checked to be); T is `size` bytes.
            unsafe {
                use std::arch::asm;
                let (base, offset) = (self.base, address as usize);
                match size {
                    1 => {
                        let value: u8 = std::mem::transmute_copy(&value);
                        asm!("mov byte ptr [{b} + {o}], {v}", b = in(reg) base, o = in(reg) offset, v = in(reg_byte) value, options(nostack, preserves_flags));
                    }
                    2 => {
                        let value: u16 = std::mem::transmute_copy(&value);
                        asm!("mov word ptr [{b} + {o}], {v:x}", b = in(reg) base, o = in(reg) offset, v = in(reg) value, options(nostack, preserves_flags));
                    }
                    4 => {
                        let value: u32 = std::mem::transmute_copy(&value);
                        asm!("mov dword ptr [{b} + {o}], {v:e}", b = in(reg) base, o = in(reg) offset, v = in(reg) value, options(nostack, preserves_flags));
                    }
                    _ => {
                        let value: u64 = std::mem::transmute_copy(&value);
                        asm!("mov qword ptr [{b} + {o}], {v}", b = in(reg) base, o = in(reg) offset, v = in(reg) value, options(nostack, preserves_flags));
                    }
                }
            }
            let _ = pointer;
            return;
        }
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
