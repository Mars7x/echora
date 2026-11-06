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

    #[inline(always)]
    pub(crate) fn load<T: Copy>(&self, address: u64) -> T {
        // SAFETY: inside the guest reservation; the guest view's protections
        // fault on anything the guest may not read.
        unsafe { std::ptr::read_unaligned(self.at::<T>(address)) }
    }

    #[inline(always)]
    pub(crate) fn store<T: Copy>(&self, address: u64, value: T) {
        // SAFETY: as in `load`, for writes.
        unsafe { std::ptr::write_unaligned(self.at::<T>(address), value) }
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
