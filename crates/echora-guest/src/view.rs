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
use crate::host_memory::GuestAccess;
use crate::memory::AtomicOp;
use portable_atomic::AtomicU128;
use std::sync::atomic::{AtomicU8, AtomicU16, AtomicU32, AtomicU64, Ordering};

/// The local exclusive monitor in Exclusive Access state: the marked address
/// and size, the value LDXR observed and the global monitor generation of
/// its granule then. STXR stores only if no exclusive store to the granule
/// succeeded since (on any thread) and the bytes still hold that value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Exclusive {
    address: u64,
    bytes: u8,
    value: u64,
    generation: u64,
}

/// The global exclusive monitor, as Arm's makes LL/SC exclusive across
/// cores: a generation per 8-byte granule bucket, advanced by every
/// successful STXR there (odd while one is in progress). A compare-and-swap
/// alone lets STXR succeed after other threads changed the value and changed
/// it back (ABA), which corrupts lock-free lists once guest threads run on
/// several host cores. Plain stores do not advance it; the value compare
/// still catches the ones that leave a different value.
static MONITOR: [AtomicU64; MONITOR_BUCKETS] = [const { AtomicU64::new(0) }; MONITOR_BUCKETS];
const MONITOR_BUCKETS: usize = 4096;

fn monitor_bucket(address: u64) -> &'static AtomicU64 {
    &MONITOR[((address >> 3) as usize) % MONITOR_BUCKETS]
}

#[derive(Debug)]
pub struct GuestView {
    access: GuestAccess,
    size: u64,
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

/// FEAT_LSE2: an LSE atomic (CAS, LD<op>, SWP) at an unaligned address is
/// single-copy atomic, and does not fault, when it stays within one aligned
/// 16-byte block; one that crosses a block boundary faults. OoT updates
/// reference counts in subobjects at odd offsets this way.
fn within_block(address: u64, bytes: u8, what: &str) -> Result<(), Trap> {
    if (address & 15) + u64::from(bytes) <= 16 {
        Ok(())
    } else {
        Err(Trap::MemoryFault {
            address,
            access: format!("{bytes}-byte {what} crossing a 16-byte boundary"),
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

/// A window's copy of a view's access path (`GuestView::window_memory`):
/// generated code keeps it in locals, where a guest store (which may alias
/// the `Core` through a raw pointer) cannot force its base and range test
/// to be reloaded. The same accesses as the view's plain loads and stores:
/// relaxed atomics, the same range test and faults.
#[derive(Clone, Copy, Debug)]
pub struct WindowMemory(GuestAccess);

impl WindowMemory {
    #[inline(always)]
    pub fn load_u8(self, address: u64) -> u8 {
        self.0.load(address)
    }
    #[inline(always)]
    pub fn load_u16(self, address: u64) -> u16 {
        u16::from_le(self.0.load(address))
    }
    #[inline(always)]
    pub fn load_u32(self, address: u64) -> u32 {
        u32::from_le(self.0.load(address))
    }
    #[inline(always)]
    pub fn load_u64(self, address: u64) -> u64 {
        u64::from_le(self.0.load(address))
    }
    #[inline(always)]
    pub fn store_u8(self, address: u64, value: u8) {
        self.0.store(address, value)
    }
    #[inline(always)]
    pub fn store_u16(self, address: u64, value: u16) {
        self.0.store(address, value.to_le())
    }
    #[inline(always)]
    pub fn store_u32(self, address: u64, value: u32) {
        self.0.store(address, value.to_le())
    }
    #[inline(always)]
    pub fn store_u64(self, address: u64, value: u64) {
        self.0.store(address, value.to_le())
    }
}

impl GuestView {
    /// The view's access path, for a window to keep in locals
    /// (`WindowMemory`). Valid while the view lives: its base and size
    /// never change (mappings change page protections, not the
    /// reservation).
    #[inline(always)]
    pub fn window_memory(&self) -> WindowMemory {
        WindowMemory(self.access)
    }

    /// A view of guest memory through `access`, `size` bytes long. The
    /// memory must outlive the view (the `Memory` that owns it lives as long
    /// as the machine).
    pub(crate) fn new(access: GuestAccess, size: u64) -> Self {
        Self {
            access,
            size,
            exclusive: None,
        }
    }

    /// Host address of `length` guest bytes at `address` when the backend
    /// keeps guest memory contiguous in the host (not checked memory), for
    /// importing it into a device. Accesses through it bypass the guest's
    /// page permissions.
    pub fn host_span(&self, address: u64, length: usize) -> Option<*mut u8> {
        self.access.host_span(address, length)
    }

    pub fn address_space_end(&self) -> u64 {
        self.size
    }

    #[inline(always)]
    fn load<T: Copy>(&self, address: u64) -> T {
        self.access.load(address)
    }

    #[inline(always)]
    fn store<T: Copy>(&mut self, address: u64, value: T) {
        self.access.store(address, value)
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
        if self.access.read(address, out) {
            Ok(())
        } else {
            Err(Trap::MemoryFault {
                address,
                access: "read outside the guest address space".into(),
            })
        }
    }

    #[inline]
    pub fn write(&mut self, address: u64, bytes: &[u8]) -> Result<(), Trap> {
        if self.access.write(address, bytes) {
            Ok(())
        } else {
            Err(Trap::MemoryFault {
                address,
                access: "write outside the guest address space".into(),
            })
        }
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

    /// The 16-byte block holding an unaligned `bytes`-byte element at
    /// `address` (see `within_block`), and the element's bit offset and mask
    /// in it.
    fn block(&self, address: u64, bytes: u8, write: bool) -> (&AtomicU128, u32, u128) {
        let pointer = self.access.atomic(address & !15, write);
        // SAFETY: 16-byte aligned and inside the reservation; the guest
        // view's protections fault on anything the guest may not access, as
        // in `atomic_load`.
        let block = unsafe { AtomicU128::from_ptr(pointer.cast()) };
        let mask = u128::from(u64::MAX >> (64 - u32::from(bytes) * 8));
        (block, (address & 15) as u32 * 8, mask)
    }

    fn atomic_load(&self, address: u64, bytes: u8, order: Ordering) -> u64 {
        if !address.is_multiple_of(u64::from(bytes)) {
            let (block, shift, mask) = self.block(address, bytes, false);
            return ((block.load(order) >> shift) & mask) as u64;
        }
        let pointer = self.access.atomic(address, false);
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
        if !address.is_multiple_of(u64::from(bytes)) {
            let (block, shift, mask) = self.block(address, bytes, true);
            let lane = (u128::from(value) & mask) << shift;
            let _ = block.fetch_update(order, Ordering::Relaxed, |old| {
                Some(old & !(mask << shift) | lane)
            });
            return;
        }
        let pointer = self.access.atomic(address, true);
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
        if !address.is_multiple_of(u64::from(bytes)) {
            let (block, shift, mask) = self.block(address, bytes, true);
            let mut current = block.load(Ordering::Relaxed);
            loop {
                let observed = ((current >> shift) & mask) as u64;
                if observed != expected {
                    return observed;
                }
                let next = current & !(mask << shift) | (u128::from(new) & mask) << shift;
                match block.compare_exchange_weak(current, next, orders.0, orders.1) {
                    Ok(_) => return observed,
                    Err(seen) => current = seen,
                }
            }
        }
        let pointer = self.access.atomic(address, true);
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
        within_block(address, bytes, "atomic compare-and-swap")?;
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
                let pointer = self.access.atomic(address, true);
                // SAFETY: 16-byte aligned (checked above) and inside the
                // reservation; the guest view's protections fault on
                // anything the guest may not access, as in `atomic_load`.
                let atomic = unsafe { AtomicU128::from_ptr(pointer.cast()) };
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
        within_block(address, bytes, "atomic fetch-add")?;
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
        within_block(address, bytes, "atomic read-modify-write")?;
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
        let bucket = monitor_bucket(address);
        // A generation read with no store in progress, unchanged across the
        // load: the value belongs to it.
        let (value, generation) = loop {
            let generation = bucket.load(Ordering::Acquire);
            if generation & 1 == 0 {
                let value = self.atomic_load(address, bytes, order);
                if bucket.load(Ordering::Acquire) == generation {
                    break (value, generation);
                }
            }
            std::hint::spin_loop();
        };
        self.exclusive = Some(Exclusive {
            address,
            bytes,
            value,
            generation,
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
        // Claim the granule only if no exclusive store succeeded since LDXR.
        let bucket = monitor_bucket(address);
        if bucket
            .compare_exchange(
                marked.generation,
                marked.generation | 1,
                Ordering::Acquire,
                Ordering::Relaxed,
            )
            .is_err()
        {
            return Ok(false);
        }
        let stored =
            self.atomic_cas(address, bytes, marked.value, value & mask, orders) == marked.value;
        let next = if stored {
            marked.generation + 2
        } else {
            marked.generation
        };
        bucket.store(next, Ordering::Release);
        Ok(stored)
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

#[cfg(all(test, any(target_os = "linux", windows)))]
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
    fn unaligned_atomics_within_a_16_byte_block_update_only_their_bytes() {
        let memory = memory();
        let mut view = memory.view();
        view.write(0x1000, &[0xaa; 16]).unwrap();
        // A reference count at an odd offset, as in OoT's subobjects.
        view.write_u32(0x1001, 5).unwrap();
        assert_eq!(
            view.fetch_add_single_thread(0x1001, 4, 1, true, true)
                .unwrap(),
            5
        );
        assert_eq!(memory.read_u32(0x1001).unwrap(), 6);
        assert_eq!(memory.read_u8(0x1000).unwrap(), 0xaa);
        assert_eq!(memory.read_u8(0x1005).unwrap(), 0xaa);
        assert_eq!(
            view.compare_exchange_single_thread(0x1009, 2, 0xaaaa, 0x1234, false, false)
                .unwrap(),
            0xaaaa
        );
        assert_eq!(memory.read_u16(0x1009).unwrap(), 0x1234);
        assert_eq!(
            view.compare_exchange_single_thread(0x1009, 2, 0, 7, false, false)
                .unwrap(),
            0x1234
        );
        assert_eq!(
            view.atomic_rmw_single_thread(0x1009, 2, AtomicOp::Swap, 9, false, true)
                .unwrap(),
            0x1234
        );
        assert_eq!(memory.read_u16(0x1009).unwrap(), 9);
        assert_eq!(memory.read_u8(0x1008).unwrap(), 0xaa);
        assert_eq!(memory.read_u8(0x100b).unwrap(), 0xaa);
        // Crossing a block boundary still faults, changing nothing.
        assert!(
            view.fetch_add_single_thread(0x100e, 4, 1, false, false)
                .is_err()
        );
        assert_eq!(memory.read_u16(0x100e).unwrap(), 0xaaaa);
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
            view.fetch_add_single_thread(0x100d, 4, 1, false, false)
                .is_err()
        );
    }

    #[test]
    fn store_exclusive_fails_after_another_thread_changes_the_value_and_back() {
        // ABA: the second view's exclusive stores leave the original value,
        // which a compare-and-swap alone would accept.
        let memory = memory();
        let mut first = memory.view();
        let mut second = memory.view();
        first.store_u64(0x1020, 5);
        assert_eq!(
            first.load_exclusive_single_thread(0x1020, 8, true).unwrap(),
            5
        );
        for value in [6, 5] {
            second
                .load_exclusive_single_thread(0x1020, 8, true)
                .unwrap();
            assert!(
                second
                    .store_exclusive_single_thread(0x1020, 8, value, true)
                    .unwrap()
            );
        }
        assert!(
            !first
                .store_exclusive_single_thread(0x1020, 8, 7, true)
                .unwrap()
        );
        assert_eq!(first.load_u64(0x1020), 5);
        // A fresh reservation succeeds.
        first.load_exclusive_single_thread(0x1020, 8, true).unwrap();
        assert!(
            first
                .store_exclusive_single_thread(0x1020, 8, 7, true)
                .unwrap()
        );
        assert_eq!(first.load_u64(0x1020), 7);
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
