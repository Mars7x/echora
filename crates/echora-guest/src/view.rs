//! Generated code's handle on guest memory: the guest view of the memory
//! file (`host_memory`) plus this guest thread's exclusive monitor.
//!
//! A view is a base pointer and a size, so every guest thread can hold its
//! own. Loads and stores go straight to the guest view, where the guest's
//! permissions are host page protections: an access the guest may not make
//! faults in the host and ends the process (see `host_memory`). Atomics are
//! host atomics on the same bytes, so they stay indivisible when several
//! guest threads run at once. Size and alignment errors still return traps,
//! as the architecture requires; mapping and permission state, and every
//! checked host-side access, live in `memory::Memory`.
#![allow(unsafe_code)]

use crate::Trap;
use crate::memory::AtomicOp;
use std::sync::atomic::{AtomicU8, AtomicU16, AtomicU32, AtomicU64, Ordering};

/// The local exclusive monitor in Exclusive Access state: the marked address
/// and size, and the value LDXR observed. STXR stores only if the bytes
/// still hold that value (a compare-and-swap), which a concurrent store by
/// another guest thread defeats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Exclusive {
    address: u64,
    bytes: u8,
    value: u64,
}

#[derive(Debug)]
pub struct GuestView {
    base: usize,
    size: u64,
    /// Bits an in-range address may have (the reservation's power-of-two
    /// size minus one).
    mask: u64,
    exclusive: Option<Exclusive>,
}

fn element_size(bytes: u8, what: &str) -> Result<(), Trap> {
    if matches!(bytes, 1 | 2 | 4 | 8) {
        Ok(())
    } else {
        Err(Trap::InvalidState {
            message: format!("unsupported {what} element size {bytes}"),
        })
    }
}

fn aligned(address: u64, bytes: u8, what: &str) -> Result<(), Trap> {
    if address.is_multiple_of(u64::from(bytes)) {
        Ok(())
    } else {
        Err(Trap::MemoryFault {
            address,
            access: format!("unaligned {bytes}-byte {what}"),
        })
    }
}

fn orderings(acquire: bool, release: bool) -> (Ordering, Ordering) {
    match (acquire, release) {
        (true, true) => (Ordering::AcqRel, Ordering::Acquire),
        (true, false) => (Ordering::Acquire, Ordering::Acquire),
        (false, true) => (Ordering::Release, Ordering::Relaxed),
        (false, false) => (Ordering::Relaxed, Ordering::Relaxed),
    }
}

impl GuestView {
    /// A view of the guest view mapped at host address `base`, `size` bytes
    /// long. The mapping must outlive the view (the `Memory` that owns it
    /// lives as long as the machine).
    pub(crate) fn new(base: usize, size: u64) -> Self {
        Self {
            base,
            size,
            mask: size.next_power_of_two() - 1,
            exclusive: None,
        }
    }

    pub fn address_space_end(&self) -> u64 {
        self.size
    }

    /// Host pointer to `length` bytes at `address`, if they lie inside the
    /// guest address space.
    #[inline(always)]
    fn pointer(&self, address: u64, length: usize) -> Option<*mut u8> {
        (address <= self.size.checked_sub(length as u64)?)
            .then(|| (self.base + address as usize) as *mut u8)
    }

    /// The host address of a typed access. One test against the
    /// reservation's power-of-two mask: an access starting past the address
    /// space (or running past its end) lands in the reservation's
    /// inaccessible tail and faults in the host, which reports the guest
    /// address (`host_memory::install_fault_report`).
    #[inline(always)]
    fn at<T: Copy>(&self, address: u64) -> *mut T {
        if address & !self.mask != 0 {
            outside(address);
        }
        (self.base + address as usize) as *mut T
    }

    #[inline(always)]
    fn load<T: Copy>(&self, address: u64) -> T {
        // SAFETY: inside the guest reservation; the guest view's protections
        // fault on anything the guest may not read.
        unsafe { std::ptr::read_unaligned(self.at::<T>(address)) }
    }

    #[inline(always)]
    fn store<T: Copy>(&mut self, address: u64, value: T) {
        // SAFETY: as in `load`, for writes.
        unsafe { std::ptr::write_unaligned(self.at::<T>(address), value) }
    }

    #[inline(always)]
    pub fn load_u8(&self, address: u64) -> u8 {
        self.load(address)
    }
    #[inline(always)]
    pub fn load_u16(&self, address: u64) -> u16 {
        u16::from_le(self.load(address))
    }
    #[inline(always)]
    pub fn load_u32(&self, address: u64) -> u32 {
        u32::from_le(self.load(address))
    }
    #[inline(always)]
    pub fn load_u64(&self, address: u64) -> u64 {
        u64::from_le(self.load(address))
    }
    #[inline(always)]
    pub fn store_u8(&mut self, address: u64, value: u8) {
        self.store(address, value)
    }
    #[inline(always)]
    pub fn store_u16(&mut self, address: u64, value: u16) {
        self.store(address, value.to_le())
    }
    #[inline(always)]
    pub fn store_u32(&mut self, address: u64, value: u32) {
        self.store(address, value.to_le())
    }
    #[inline(always)]
    pub fn store_u64(&mut self, address: u64, value: u64) {
        self.store(address, value.to_le())
    }

    /// Host-side bulk access (graphics, HLE): bytes at `address`, or a
    /// fault trap when they leave the guest address space.
    #[inline]
    pub fn read(&self, address: u64, out: &mut [u8]) -> Result<(), Trap> {
        let pointer = self
            .pointer(address, out.len())
            .ok_or_else(|| Trap::MemoryFault {
                address,
                access: "read outside the guest address space".into(),
            })?;
        // SAFETY: as in `load`, for `out.len()` bytes.
        unsafe { std::ptr::copy_nonoverlapping(pointer, out.as_mut_ptr(), out.len()) };
        Ok(())
    }

    #[inline]
    pub fn write(&mut self, address: u64, bytes: &[u8]) -> Result<(), Trap> {
        let pointer = self
            .pointer(address, bytes.len())
            .ok_or_else(|| Trap::MemoryFault {
                address,
                access: "write outside the guest address space".into(),
            })?;
        // SAFETY: as in `store`, for `bytes.len()` bytes.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), pointer, bytes.len()) };
        Ok(())
    }

    pub fn read_u8(&self, address: u64) -> Result<u8, Trap> {
        let mut b = [0; 1];
        self.read(address, &mut b)?;
        Ok(b[0])
    }
    pub fn read_u16(&self, address: u64) -> Result<u16, Trap> {
        let mut b = [0; 2];
        self.read(address, &mut b)?;
        Ok(u16::from_le_bytes(b))
    }
    pub fn read_u32(&self, address: u64) -> Result<u32, Trap> {
        let mut b = [0; 4];
        self.read(address, &mut b)?;
        Ok(u32::from_le_bytes(b))
    }
    pub fn read_u64(&self, address: u64) -> Result<u64, Trap> {
        let mut b = [0; 8];
        self.read(address, &mut b)?;
        Ok(u64::from_le_bytes(b))
    }
    pub fn write_u8(&mut self, address: u64, value: u8) -> Result<(), Trap> {
        self.write(address, &[value])
    }
    pub fn write_u16(&mut self, address: u64, value: u16) -> Result<(), Trap> {
        self.write(address, &value.to_le_bytes())
    }
    pub fn write_u32(&mut self, address: u64, value: u32) -> Result<(), Trap> {
        self.write(address, &value.to_le_bytes())
    }
    pub fn write_u64(&mut self, address: u64, value: u64) -> Result<(), Trap> {
        self.write(address, &value.to_le_bytes())
    }

    fn atomic_load(&self, address: u64, bytes: u8, order: Ordering) -> u64 {
        let pointer = self.at::<u8>(address);
        // SAFETY: aligned (checked by callers) and inside the reservation.
        unsafe {
            match bytes {
                1 => AtomicU8::from_ptr(pointer).load(order) as u64,
                2 => AtomicU16::from_ptr(pointer.cast()).load(order) as u64,
                4 => AtomicU32::from_ptr(pointer.cast()).load(order) as u64,
                _ => AtomicU64::from_ptr(pointer.cast()).load(order),
            }
        }
    }

    fn atomic_store(&self, address: u64, bytes: u8, value: u64, order: Ordering) {
        let pointer = self.at::<u8>(address);
        // SAFETY: as in `atomic_load`.
        unsafe {
            match bytes {
                1 => AtomicU8::from_ptr(pointer).store(value as u8, order),
                2 => AtomicU16::from_ptr(pointer.cast()).store(value as u16, order),
                4 => AtomicU32::from_ptr(pointer.cast()).store(value as u32, order),
                _ => AtomicU64::from_ptr(pointer.cast()).store(value, order),
            }
        }
    }

    /// Compare-and-swap; returns the observed value (equal to `expected`
    /// exactly when the swap happened).
    fn atomic_cas(
        &self,
        address: u64,
        bytes: u8,
        expected: u64,
        new: u64,
        orders: (Ordering, Ordering),
    ) -> u64 {
        let pointer = self.at::<u8>(address);
        let (success, failure) = orders;
        // SAFETY: as in `atomic_load`.
        unsafe {
            match bytes {
                1 => AtomicU8::from_ptr(pointer)
                    .compare_exchange(expected as u8, new as u8, success, failure)
                    .unwrap_or_else(|observed| observed) as u64,
                2 => AtomicU16::from_ptr(pointer.cast())
                    .compare_exchange(expected as u16, new as u16, success, failure)
                    .unwrap_or_else(|observed| observed) as u64,
                4 => AtomicU32::from_ptr(pointer.cast())
                    .compare_exchange(expected as u32, new as u32, success, failure)
                    .unwrap_or_else(|observed| observed) as u64,
                _ => AtomicU64::from_ptr(pointer.cast())
                    .compare_exchange(expected, new, success, failure)
                    .unwrap_or_else(|observed| observed),
            }
        }
    }

    /// Atomically replace the value with `update(old)`; returns `old`.
    fn atomic_update(
        &self,
        address: u64,
        bytes: u8,
        orders: (Ordering, Ordering),
        update: impl Fn(u64) -> u64,
    ) -> u64 {
        let mask = u64::MAX >> (64 - u32::from(bytes) * 8);
        let mut old = self.atomic_load(address, bytes, Ordering::Relaxed);
        loop {
            let observed = self.atomic_cas(address, bytes, old, update(old) & mask, orders);
            if observed == old {
                return old;
            }
            old = observed;
        }
    }

    /// LSE CAS: the observed value; memory takes `desired` when it held
    /// `expected` (both truncated to the element).
    pub fn compare_exchange_single_thread(
        &mut self,
        address: u64,
        bytes: u8,
        expected: u64,
        desired: u64,
        acquire: bool,
        release: bool,
    ) -> Result<u64, Trap> {
        element_size(bytes, "CAS")?;
        aligned(address, bytes, "atomic compare-and-swap")?;
        let mask = u64::MAX >> (64 - u32::from(bytes) * 8);
        Ok(self.atomic_cas(
            address,
            bytes,
            expected & mask,
            desired & mask,
            orderings(acquire, release),
        ))
    }

    /// LSE CASP: compares the pair at `address` (first element at the lower
    /// address) with `expected` and, if equal, stores `desired`; returns the
    /// old pair. `bytes` is the element size (4 or 8) and the pair must be
    /// aligned to its total size.
    #[allow(clippy::too_many_arguments)]
    pub fn compare_exchange_pair_single_thread(
        &mut self,
        address: u64,
        bytes: u8,
        expected: (u64, u64),
        desired: (u64, u64),
        acquire: bool,
        release: bool,
    ) -> Result<(u64, u64), Trap> {
        let orders = orderings(acquire, release);
        match bytes {
            4 => {
                aligned(address, 8, "atomic pair compare-and-swap")?;
                let join = |(lo, hi): (u64, u64)| (lo & 0xffff_ffff) | hi << 32;
                let old = self.atomic_cas(address, 8, join(expected), join(desired), orders);
                Ok((old & 0xffff_ffff, old >> 32))
            }
            8 => {
                aligned(address, 16, "atomic pair compare-and-swap")?;
                let join = |(lo, hi): (u64, u64)| u128::from(lo) | u128::from(hi) << 64;
                let pointer = self.at::<u8>(address);
                // SAFETY: 16-byte aligned (checked above) and inside the
                // reservation; the guest view's protections fault on
                // anything the guest may not access, as in `atomic_load`.
                let atomic = unsafe { portable_atomic::AtomicU128::from_ptr(pointer.cast()) };
                let old = atomic
                    .compare_exchange(join(expected), join(desired), orders.0, orders.1)
                    .unwrap_or_else(|observed| observed);
                Ok((old as u64, (old >> 64) as u64))
            }
            _ => Err(Trap::InvalidState {
                message: format!("unsupported CASP element size {bytes}"),
            }),
        }
    }

    /// LSE LDADD: the old value; memory takes old + `addend`.
    pub fn fetch_add_single_thread(
        &mut self,
        address: u64,
        bytes: u8,
        addend: u64,
        acquire: bool,
        release: bool,
    ) -> Result<u64, Trap> {
        element_size(bytes, "LDADD")?;
        aligned(address, bytes, "atomic fetch-add")?;
        Ok(
            self.atomic_update(address, bytes, orderings(acquire, release), |old| {
                old.wrapping_add(addend)
            }),
        )
    }

    /// LSE LDCLR/LDEOR/LDSET/LD{S,U}{MAX,MIN}/SWP: the old value.
    pub fn atomic_rmw_single_thread(
        &mut self,
        address: u64,
        bytes: u8,
        op: AtomicOp,
        operand: u64,
        acquire: bool,
        release: bool,
    ) -> Result<u64, Trap> {
        element_size(bytes, "atomic")?;
        aligned(address, bytes, "atomic read-modify-write")?;
        let bits = u32::from(bytes) * 8;
        let mask = u64::MAX >> (64 - bits);
        let signed = |value: u64| ((value << (64 - bits)) as i64) >> (64 - bits);
        let operand = operand & mask;
        Ok(self.atomic_update(
            address,
            bytes,
            orderings(acquire, release),
            |old| match op {
                AtomicOp::Clear => old & !operand,
                AtomicOp::ExclusiveOr => old ^ operand,
                AtomicOp::Set => old | operand,
                AtomicOp::SignedMax => {
                    if signed(operand) > signed(old) {
                        operand
                    } else {
                        old
                    }
                }
                AtomicOp::SignedMin => {
                    if signed(operand) < signed(old) {
                        operand
                    } else {
                        old
                    }
                }
                AtomicOp::UnsignedMax => old.max(operand),
                AtomicOp::UnsignedMin => old.min(operand),
                AtomicOp::Swap => operand,
            },
        ))
    }

    /// LDAPR: an acquire load (RCpc).
    pub fn load_acquire_rcpc_single_thread(&self, address: u64, bytes: u8) -> Result<u64, Trap> {
        element_size(bytes, "LDAPR")?;
        aligned(address, bytes, "LDAPR without LSE2 alignment relaxation")?;
        Ok(self.atomic_load(address, bytes, Ordering::Acquire))
    }

    /// LDAR: an acquire load (RCsc).
    pub fn load_acquire_rcsc_single_thread(&self, address: u64, bytes: u8) -> Result<u64, Trap> {
        element_size(bytes, "LDAR")?;
        aligned(address, bytes, "LDAR without LSE2 alignment relaxation")?;
        Ok(self.atomic_load(address, bytes, Ordering::SeqCst))
    }

    /// STLR: a release store.
    pub fn store_release_single_thread(
        &mut self,
        address: u64,
        bytes: u8,
        value: u64,
    ) -> Result<(), Trap> {
        element_size(bytes, "STLR")?;
        aligned(address, bytes, "STLR without LSE2 alignment relaxation")?;
        self.atomic_store(address, bytes, value, Ordering::SeqCst);
        Ok(())
    }

    /// LDXR/LDAXR: load and mark the address in the local monitor.
    pub fn load_exclusive_single_thread(
        &mut self,
        address: u64,
        bytes: u8,
        acquire: bool,
    ) -> Result<u64, Trap> {
        element_size(bytes, "exclusive load")?;
        aligned(address, bytes, "load-exclusive")?;
        let order = if acquire {
            Ordering::Acquire
        } else {
            Ordering::Relaxed
        };
        let value = self.atomic_load(address, bytes, order);
        self.exclusive = Some(Exclusive {
            address,
            bytes,
            value,
        });
        Ok(value)
    }

    /// STXR/STLXR: `Ok(true)` (status 0) when stored. The store happens only
    /// if the monitor marks this address and size and the bytes still hold
    /// the value LDXR saw; every STXR leaves the monitor open.
    pub fn store_exclusive_single_thread(
        &mut self,
        address: u64,
        bytes: u8,
        value: u64,
        release: bool,
    ) -> Result<bool, Trap> {
        element_size(bytes, "exclusive store")?;
        aligned(address, bytes, "store-exclusive")?;
        let Some(marked) = self.exclusive.take() else {
            return Ok(false);
        };
        if marked.address != address || marked.bytes != bytes {
            return Ok(false);
        }
        let mask = u64::MAX >> (64 - u32::from(bytes) * 8);
        let orders = orderings(false, release);
        Ok(self.atomic_cas(address, bytes, marked.value, value & mask, orders) == marked.value)
    }

    /// CLREX, exception return and context switches open the monitor.
    pub fn clear_exclusive(&mut self) {
        self.exclusive = None;
    }

    /// The marked address and size, if the monitor is in Exclusive Access.
    pub fn exclusive_reservation(&self) -> Option<(u64, u8)> {
        self.exclusive.map(|marked| (marked.address, marked.bytes))
    }
}

#[cold]
#[inline(never)]
fn outside(address: u64) -> ! {
    panic!("guest memory access at {address:#x} lies outside the guest address space")
}

#[cfg(all(test, target_os = "linux"))]
#[allow(clippy::unwrap_used)]
mod tests {
    use crate::memory::{AtomicOp, Memory, READ, STATE_NORMAL, WRITE};

    fn memory() -> Memory {
        let mut memory = Memory::new(0x10000);
        memory
            .map(0x1000, 0x1000, READ | WRITE, STATE_NORMAL, "data")
            .unwrap();
        memory
    }

    #[test]
    fn atomics_see_and_change_the_bytes_memory_holds() {
        let memory = memory();
        let mut view = memory.view();
        view.store_u32(0x1000, 5);
        assert_eq!(
            view.fetch_add_single_thread(0x1000, 4, 3, true, true)
                .unwrap(),
            5
        );
        assert_eq!(memory.read_u32(0x1000).unwrap(), 8);
        assert_eq!(
            view.compare_exchange_single_thread(0x1000, 4, 7, 1, false, false)
                .unwrap(),
            8
        );
        assert_eq!(
            view.compare_exchange_single_thread(0x1000, 4, 8, 1, false, false)
                .unwrap(),
            8
        );
        assert_eq!(view.load_u32(0x1000), 1);
        view.store_u8(0x1008, 0xf0);
        let old = view
            .atomic_rmw_single_thread(0x1008, 1, AtomicOp::SignedMax, 0x10, false, false)
            .unwrap();
        assert_eq!((old, view.load_u8(0x1008)), (0xf0, 0x10));
        assert!(
            view.fetch_add_single_thread(0x1001, 4, 1, false, false)
                .is_err()
        );
    }

    #[test]
    fn store_exclusive_fails_after_another_store_changes_the_value() {
        let memory = memory();
        let mut first = memory.view();
        let mut second = memory.view();
        first.store_u64(0x1010, 1);
        assert_eq!(
            first.load_exclusive_single_thread(0x1010, 8, true).unwrap(),
            1
        );
        second.store_u64(0x1010, 2);
        assert!(
            !first
                .store_exclusive_single_thread(0x1010, 8, 3, true)
                .unwrap()
        );
        assert_eq!(
            first
                .load_exclusive_single_thread(0x1010, 8, false)
                .unwrap(),
            2
        );
        assert!(
            first
                .store_exclusive_single_thread(0x1010, 8, 3, false)
                .unwrap()
        );
        assert!(
            !first
                .store_exclusive_single_thread(0x1010, 8, 4, false)
                .unwrap()
        );
        assert_eq!(second.load_u64(0x1010), 3);
    }

    #[test]
    fn concurrent_fetch_adds_lose_nothing() {
        let memory = memory();
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let mut view = memory.view();
                scope.spawn(move || {
                    for _ in 0..10_000 {
                        view.fetch_add_single_thread(0x1020, 8, 1, true, true)
                            .unwrap();
                    }
                });
            }
        });
        assert_eq!(memory.read_u64(0x1020).unwrap(), 40_000);
    }
}
