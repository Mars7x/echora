use crate::Trap;
use crate::host_memory::HostMemory;
use serde::Serialize;
use std::collections::BTreeMap;

/// LSE read-modify-write selectors other than ADD (see `atomic_rmw_single_thread`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum AtomicOp {
    Clear,
    ExclusiveOr,
    Set,
    SignedMax,
    SignedMin,
    UnsignedMax,
    UnsignedMin,
    Swap,
}

pub const PAGE_SIZE: u64 = 0x1000;
pub const READ: u8 = 1;
pub const WRITE: u8 = 2;
pub const EXECUTE: u8 = 4;
pub const STATE_FREE: u32 = 0;
pub const STATE_CODE: u32 = 3;
pub const STATE_CODE_DATA: u32 = 4;
pub const STATE_NORMAL: u32 = 5;
/// Horizon MemoryState::Shared (KMemoryState_Shared).
pub const STATE_SHARED: u32 = 6;
pub const STATE_ALIAS_CODE_DATA: u32 = 9;
pub const STATE_STACK: u32 = 0x0b;
pub const STATE_TLS: u32 = 0x0c;
pub const STATE_INACCESSIBLE: u32 = 0x10;
/// Horizon MemoryAttribute::PermissionLocked.
pub const ATTRIBUTE_PERMISSION_LOCKED: u32 = 0x10;
pub const ATTRIBUTE_LOCKED: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Mapping {
    pub base: u64,
    pub size: u64,
    pub state: u32,
    pub permissions: u8,
    pub attributes: u32,
    pub label: String,
}

/// Memory-side MapMemory failures. The SVC wrapper additionally validates the
/// destination's configured stack region and the kernel's error precedence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapMemoryError {
    InvalidAddress,
    InvalidSize,
    InvalidCurrentMemory,
}

/// One destination range references canonical source pages. Source permissions
/// stay locked while this exists, so supported mappings cannot form chains.
#[derive(Clone, Debug, PartialEq, Eq)]
struct StackAlias {
    source: u64,
    size: u64,
}

impl Mapping {
    fn end(&self) -> u64 {
        self.base + self.size
    }
}

/// Sparse guest pages, independent of host virtual addresses.
pub struct Memory {
    address_space_end: u64,
    mappings: BTreeMap<u64, Mapping>,
    pages: Pages,
    stack_aliases: BTreeMap<u64, StackAlias>,
    translated_ranges: Vec<(u64, u64)>,
    exclusive: Option<ExclusiveReservation>,
    /// Software TLB: guest page -> backing page storage and the access the
    /// full checks would allow there. A cache only: every mapping, alias,
    /// protection or page-set change flushes it, and anything it cannot
    /// answer takes the full path.
    tlb: Box<[std::cell::Cell<TlbEntry>; TLB_ENTRIES]>,
}

/// One TLB entry. `tag` is the guest page address (u64::MAX when empty);
/// `page` is the host address of the backing page's storage (boxed pages
/// never move), valid until the page is removed (removals flush the TLB).
/// Holding the address, not a slot index, saves two dependent loads on
/// every fast access.
#[derive(Clone, Copy)]
struct TlbEntry {
    tag: u64,
    page: usize,
    access: u8,
}

/// Guest page storage: page contents live in the memory file at their
/// backing address (`HostMemory`), so every page reads as zero until
/// written and generated code can reach it through the guest view. The
/// resident set records the pages the host wrote, for reports and tests;
/// stores through the guest view are not tracked.
struct Pages {
    host: HostMemory,
    resident: std::collections::BTreeSet<u64>,
}

impl Pages {
    fn new(size: u64) -> Self {
        Self {
            host: HostMemory::new(size),
            resident: Default::default(),
        }
    }

    fn len(&self) -> usize {
        self.resident.len()
    }

    /// The host-written page at backing address `page` (its "slot" is that
    /// address): the TLB serves only those, so host writes to new pages take
    /// the slow path that records them as resident. Every other page still
    /// reads correctly through `get`.
    fn slot_of(&self, page: u64) -> Option<usize> {
        self.resident.contains(&page).then_some(page as usize)
    }

    /// Host address of a slot's page storage (stable for the file's life).
    fn slot_address(&self, slot: usize) -> usize {
        self.host.page_address(slot as u64)
    }

    fn get(&self, page: &u64) -> Option<&[u8; PAGE_SIZE as usize]> {
        Some(self.host.page(*page))
    }

    fn get_or_insert(&mut self, page: u64) -> &mut [u8; PAGE_SIZE as usize] {
        self.resident.insert(page);
        self.host.page_mut(page)
    }

    /// Zero backing pages `[start, end)` and return their storage.
    /// Callers flush the TLB.
    fn release(&mut self, start: u64, end: u64) {
        self.host.release(start, end);
        // Only the pages in the range, not a pass over every resident page:
        // the game maps and unmaps physical memory often.
        let released: Vec<u64> = self.resident.range(start..end).copied().collect();
        for page in released {
            self.resident.remove(&page);
        }
    }

    #[cfg(all(test, any(target_os = "linux", windows)))]
    fn contains_key(&self, page: &u64) -> bool {
        self.resident.contains(page)
    }

    #[cfg(all(test, any(target_os = "linux", windows)))]
    fn keys(&self) -> impl Iterator<Item = &u64> {
        self.resident.iter()
    }
}

#[cfg(test)]
impl Clone for Pages {
    fn clone(&self) -> Self {
        let mut copy = Pages::new(TEST_CLONE_SIZE);
        for &page in &self.resident {
            copy.get_or_insert(page)
                .copy_from_slice(self.host.page(page));
        }
        copy
    }
}

/// Size of the memory file behind a cloned page set (tests only).
#[cfg(test)]
const TEST_CLONE_SIZE: u64 = 1 << 39;

impl PartialEq for Pages {
    fn eq(&self, other: &Self) -> bool {
        self.resident == other.resident
            && self
                .resident
                .iter()
                .all(|page| self.host.page(*page) == other.host.page(*page))
    }
}

impl Eq for Pages {}

impl std::fmt::Debug for Pages {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_map()
            .entries(self.resident.iter().map(|page| (page, "<page>")))
            .finish()
    }
}

impl std::ops::Index<&u64> for Pages {
    type Output = [u8; PAGE_SIZE as usize];

    fn index(&self, page: &u64) -> &Self::Output {
        self.host.page(*page)
    }
}

const TLB_ENTRIES: usize = 4096;
const EMPTY_TLB_ENTRY: TlbEntry = TlbEntry {
    tag: u64::MAX,
    page: 0,
    access: 0,
};

/// The one guest PE's local exclusive monitor in Exclusive Access state.
///
/// Arm ARM (DDI0487) B2.9 marks the accessed address and size. `backing` is
/// the canonical sparse-page byte address, so stores through a stack alias of
/// the same storage are recognized. No global monitor or second PE exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ExclusiveReservation {
    address: u64,
    backing: u64,
    bytes: u8,
}

impl Memory {
    pub fn new(address_space_end: u64) -> Self {
        Self {
            address_space_end,
            mappings: BTreeMap::new(),
            pages: Pages::new(address_space_end),
            stack_aliases: BTreeMap::new(),
            translated_ranges: Vec::new(),
            exclusive: None,
            // A fixed-size array, so indexing needs no bounds check.
            tlb: (0..TLB_ENTRIES)
                .map(|_| std::cell::Cell::new(EMPTY_TLB_ENTRY))
                .collect::<Box<[_]>>()
                .try_into()
                .unwrap_or_else(|_| unreachable!()),
        }
    }

    /// Layout changes. Each runs the change, then brings the guest view of
    /// the ranges it may have touched up to date, whatever the result.
    pub fn map(
        &mut self,
        base: u64,
        size: u64,
        permissions: u8,
        state: u32,
        label: impl Into<String>,
    ) -> Result<(), Trap> {
        let result = self.map_unsynced(base, size, permissions, state, label);
        self.sync_guest_view(&[(base, size)]);
        result
    }

    pub fn map_stack_alias(
        &mut self,
        destination: u64,
        source: u64,
        size: u64,
    ) -> Result<(), MapMemoryError> {
        let result = self.map_stack_alias_unsynced(destination, source, size);
        self.sync_guest_view(&[(destination, size), (source, size)]);
        result
    }

    pub fn unmap_stack_alias(
        &mut self,
        destination: u64,
        source: u64,
        size: u64,
    ) -> Result<(), MapMemoryError> {
        let result = self.unmap_stack_alias_unsynced(destination, source, size);
        self.sync_guest_view(&[(destination, size), (source, size)]);
        result
    }

    pub fn protect_translated_code(&mut self, base: u64, size: u64) -> Result<(), Trap> {
        let result = self.protect_translated_code_unsynced(base, size);
        self.sync_guest_view(&[(base, size)]);
        result
    }

    pub fn unmap_exact(&mut self, base: u64, size: u64) -> Result<(), Trap> {
        let result = self.unmap_exact_unsynced(base, size);
        self.sync_guest_view(&[(base, size)]);
        result
    }

    pub fn resize_heap(&mut self, base: u64, old_size: u64, new_size: u64) -> Result<(), Trap> {
        let result = self.resize_heap_unsynced(base, old_size, new_size);
        self.sync_guest_view(&[(base, old_size.max(new_size))]);
        result
    }

    pub fn lock_range(
        &mut self,
        address: u64,
        size: u64,
        states: &[u32],
        required: u8,
        permissions: u8,
        attribute: u32,
    ) -> Result<bool, Trap> {
        let result =
            self.lock_range_unsynced(address, size, states, required, permissions, attribute);
        self.sync_guest_view(&[(address, size)]);
        result
    }

    pub fn unlock_range(
        &mut self,
        address: u64,
        size: u64,
        attribute: u32,
        permissions: u8,
    ) -> Result<(), Trap> {
        let result = self.unlock_range_unsynced(address, size, attribute, permissions);
        self.sync_guest_view(&[(address, size)]);
        result
    }

    pub fn set_permissions(
        &mut self,
        address: u64,
        size: u64,
        permissions: u8,
    ) -> Result<(), Trap> {
        let result = self.set_permissions_unsynced(address, size, permissions);
        self.sync_guest_view(&[(address, size)]);
        result
    }

    pub fn set_attributes(
        &mut self,
        address: u64,
        size: u64,
        mask: u32,
        value: u32,
    ) -> Result<(), Trap> {
        let result = self.set_attributes_unsynced(address, size, mask, value);
        self.sync_guest_view(&[(address, size)]);
        result
    }

    /// Remap the guest view over `(base, size)` ranges from the mappings:
    /// each page maps its backing page with the access `fill_tlb` would
    /// grant (READ/WRITE), and unmapped pages become inaccessible.
    /// Neighbouring pages with the same access and contiguous backing map
    /// as one run.
    fn sync_guest_view(&mut self, ranges: &[(u64, u64)]) {
        for &(base, size) in ranges {
            let start = base & !(PAGE_SIZE - 1);
            let end = base
                .saturating_add(size)
                .min(self.address_space_end)
                .next_multiple_of(PAGE_SIZE);
            let mut runs: Vec<(u64, u64, Option<u64>, u8)> = Vec::new();
            let mut page = start;
            while page < end {
                let (backing, access) = match self.guest_view_page(page) {
                    Some((backing, access)) => (Some(backing), access),
                    None => (None, 0),
                };
                match runs.last_mut() {
                    Some((_, length, run_backing, run_access))
                        if *run_access == access
                            && match (*run_backing, backing) {
                                (Some(first), Some(next)) => first + *length == next,
                                (None, None) => true,
                                _ => false,
                            } =>
                    {
                        *length += PAGE_SIZE;
                    }
                    _ => runs.push((page, PAGE_SIZE, backing, access)),
                }
                page += PAGE_SIZE;
            }
            for (address, length, backing, access) in runs {
                self.pages.host.map_guest(address, length, backing, access);
            }
        }
    }

    /// The backing page and READ/WRITE access of guest page `page` in the
    /// guest view, or None when unmapped.
    fn guest_view_page(&self, page: u64) -> Option<(u64, u8)> {
        let mapping = self.mapping_at(page)?;
        if mapping.state == STATE_FREE || mapping.end() < page + PAGE_SIZE {
            return None;
        }
        let backing = self.backing_page(page);
        let mut access = mapping.permissions & (READ | WRITE);
        let end = page + PAGE_SIZE;
        if self.translated_ranges.iter().any(|(start, stop)| {
            (page < *stop && *start < end) || (backing < *stop && *start < backing + PAGE_SIZE)
        }) {
            access &= !WRITE;
        }
        Some((backing, access))
    }

    fn flush_tlb(&self) {
        for entry in self.tlb.iter() {
            entry.set(EMPTY_TLB_ENTRY);
        }
    }

    /// Whether the 4-byte instruction at `pc` is known executable without a
    /// lookup: its page is in the TLB with execute permission. False means
    /// "check fully", not "not executable".
    #[inline(always)]
    pub fn executable_fast(&self, pc: u64) -> bool {
        let page = pc & !(PAGE_SIZE - 1);
        let entry = self.tlb[(page / PAGE_SIZE) as usize % TLB_ENTRIES].get();
        entry.tag == page && entry.access & EXECUTE != 0
    }

    /// Host storage for an access of `size` bytes at `address` within one
    /// page with every `mask` permission, from the TLB (filled on a miss).
    /// None sends the access down the full path: page crossings, missing
    /// permissions, unmapped or never-written pages, translated code (for
    /// writes) and anything else the full checks handle.
    /// The page a TLB entry's host address names.
    #[inline(always)]
    #[allow(unsafe_code)]
    fn tlb_page(&self, page: usize) -> &[u8; PAGE_SIZE as usize] {
        // SAFETY: `page` came from a live TLB entry: the address of a boxed
        // page owned by `self.pages`, which is not removed while cached.
        unsafe { &*(page as *const [u8; PAGE_SIZE as usize]) }
    }

    #[inline(always)]
    #[allow(unsafe_code)]
    fn tlb_page_mut(&mut self, page: usize) -> &mut [u8; PAGE_SIZE as usize] {
        // SAFETY: as in `tlb_page`; `&mut self` makes the access exclusive.
        unsafe { &mut *(page as *mut [u8; PAGE_SIZE as usize]) }
    }

    #[inline(always)]
    fn fast_page(&self, address: u64, size: usize, mask: u8) -> Option<(usize, usize)> {
        let page = address & !(PAGE_SIZE - 1);
        let offset = (address - page) as usize;
        if offset + size > PAGE_SIZE as usize {
            return None;
        }
        let slot = &self.tlb[(page / PAGE_SIZE) as usize % TLB_ENTRIES];
        let mut entry = slot.get();
        if entry.tag != page {
            entry = self.fill_tlb(page)?;
            slot.set(entry);
        }
        (entry.access & mask == mask).then_some((entry.page, offset))
    }

    /// TLB slot for a store that may skip `write_pages`: the page is
    /// writable (TLB write access already excludes translated code) and, with
    /// a reservation open, no alias could reach it from another address.
    /// The caller performing the store must then call `note_fast_store`.
    #[inline(always)]
    fn fast_store(&self, address: u64, size: usize) -> Option<(usize, usize)> {
        if self.exclusive.is_some() && !self.stack_aliases.is_empty() {
            return None;
        }
        self.fast_page(address, size, WRITE)
    }

    /// A fast store overlapping the reservation's addresses clears it.
    #[inline(always)]
    fn note_fast_store(&mut self, address: u64, size: usize) {
        if let Some(reservation) = self.exclusive
            && address < reservation.address + u64::from(reservation.bytes)
            && reservation.address < address + size as u64
        {
            self.exclusive = None;
        }
    }

    /// TLB slot for an atomic read-modify-write: readable and fast-storable.
    #[inline(always)]
    fn fast_rmw(&self, address: u64, size: usize) -> Option<(usize, usize)> {
        self.fast_page(address, size, READ)?;
        self.fast_store(address, size)
    }

    #[cold]
    fn fill_tlb(&self, page: u64) -> Option<TlbEntry> {
        let mapping = self.mapping_at(page)?;
        if mapping.state == STATE_FREE || mapping.end() < page + PAGE_SIZE {
            return None;
        }
        let backing = self.backing_page(page);
        let mut access = mapping.permissions & (READ | WRITE | EXECUTE);
        // A page never written reads as zeros from the memory file, so it
        // is cached for reads; its first write takes the full path, which
        // makes it resident.
        if self.pages.slot_of(backing).is_none() {
            access &= !WRITE;
        }
        let end = page + PAGE_SIZE;
        if self.translated_ranges.iter().any(|(start, stop)| {
            (page < *stop && *start < end) || (backing < *stop && *start < backing + PAGE_SIZE)
        }) {
            access &= !WRITE;
        }
        Some(TlbEntry {
            tag: page,
            page: self.pages.slot_address(backing as usize),
            access,
        })
    }

    /// The mapping containing `address`, without copying it.
    fn mapping_at(&self, address: u64) -> Option<&Mapping> {
        if address >= self.address_space_end {
            return None;
        }
        self.mappings
            .range(..=address)
            .next_back()
            .map(|(_, mapping)| mapping)
            .filter(|mapping| address < mapping.end())
    }

    /// Return the local monitor to Open Access (CLREX, exception return, or a
    /// conservative clear). Store-Exclusive is permitted to fail spuriously,
    /// so extra clears are architecturally safe; missing clears are not.
    pub fn clear_exclusive(&mut self) {
        self.exclusive = None;
    }

    /// Numeric (address, size) of the live reservation, for tests/reports.
    pub fn exclusive_reservation(&self) -> Option<(u64, u8)> {
        self.exclusive.map(|r| (r.address, r.bytes))
    }

    pub fn address_space_end(&self) -> u64 {
        self.address_space_end
    }

    /// A handle on the guest view for generated code (one per guest
    /// thread). It stays valid while this `Memory` lives.
    pub fn view(&self) -> crate::view::GuestView {
        crate::view::GuestView::new(self.pages.host.guest_access(), self.address_space_end)
    }

    pub fn mappings(&self) -> impl Iterator<Item = &Mapping> {
        self.mappings.values()
    }

    pub fn resident_bytes(&self) -> u64 {
        self.pages.len() as u64 * PAGE_SIZE
    }

    /// Numeric-only alias metadata: (destination, canonical source, byte size).
    /// This does not expose page contents or add a second physical allocation.
    pub fn stack_aliases(&self) -> impl Iterator<Item = (u64, u64, u64)> + '_ {
        self.stack_aliases
            .iter()
            .map(|(&destination, alias)| (destination, alias.source, alias.size))
    }

    fn range_end(&self, address: u64, size: u64, access: &str) -> Result<u64, Trap> {
        match address.checked_add(size) {
            Some(end) if end <= self.address_space_end => Ok(end),
            _ => Err(Trap::MemoryFault {
                address,
                access: format!("{access}: address range overflows guest space"),
            }),
        }
    }

    fn map_unsynced(
        &mut self,
        base: u64,
        size: u64,
        permissions: u8,
        state: u32,
        label: impl Into<String>,
    ) -> Result<(), Trap> {
        self.flush_tlb();
        // Layout changes conservatively end any reservation (spurious STXR
        // failure is permitted), even if the request is later rejected.
        self.clear_exclusive();
        let end = self.range_end(base, size, "map")?;
        if size == 0 || !base.is_multiple_of(PAGE_SIZE) || !size.is_multiple_of(PAGE_SIZE) {
            return Err(Trap::InvalidState {
                message: "mapping requires a nonempty page-aligned range".into(),
            });
        }
        if permissions & !7 != 0 {
            return Err(Trap::InvalidState {
                message: "invalid guest memory permissions".into(),
            });
        }
        if self
            .mappings
            .values()
            .any(|m| base < m.end() && m.base < end)
        {
            return Err(Trap::InvalidState {
                message: format!("overlapping guest mapping at {base:#x}"),
            });
        }
        self.mappings.insert(
            base,
            Mapping {
                base,
                size,
                state,
                permissions,
                attributes: 0,
                label: label.into(),
            },
        );
        Ok(())
    }

    /// Alias a uniform RW data range into a free destination, retaining the
    /// same sparse backing. The SVC wrapper must check stack-region containment.
    ///
    /// Pinned Atmosphere 6b0c077 / Eden 1a6d7d4 KPageTableBase::MapMemory checks
    /// CanAlias + UserReadWrite + no attributes, then Free destination. The
    /// supported CanAlias states are CodeData, Normal and AliasCodeData. It
    /// retains source state with user permission None/Locked, and maps the
    /// destination as Stack/RW/no attributes using the same page group.
    ///
    /// All metadata construction precedes publication. No data pages are copied
    /// or allocated here; even an untouched zero page has stable source identity.
    /// UnmapMemory and alias teardown are not implemented in this milestone.
    fn map_stack_alias_unsynced(
        &mut self,
        destination: u64,
        source: u64,
        size: u64,
    ) -> Result<(), MapMemoryError> {
        self.flush_tlb();
        // Layout changes conservatively end any reservation (spurious STXR
        // failure is permitted), even if the request is later rejected.
        self.clear_exclusive();
        if !destination.is_multiple_of(PAGE_SIZE) || !source.is_multiple_of(PAGE_SIZE) {
            return Err(MapMemoryError::InvalidAddress);
        }
        if size == 0 || !size.is_multiple_of(PAGE_SIZE) {
            return Err(MapMemoryError::InvalidSize);
        }
        let source_end = source
            .checked_add(size)
            .filter(|end| *end <= self.address_space_end)
            .ok_or(MapMemoryError::InvalidCurrentMemory)?;
        let destination_end = destination
            .checked_add(size)
            .filter(|end| *end <= self.address_space_end)
            .ok_or(MapMemoryError::InvalidCurrentMemory)?;
        let first = self.query(source);
        if !matches!(
            first.state,
            STATE_CODE_DATA | STATE_NORMAL | STATE_ALIAS_CODE_DATA
        ) {
            return Err(MapMemoryError::InvalidCurrentMemory);
        }
        let mut cursor = source;
        while cursor < source_end {
            let mapping = self.query(cursor);
            // The kernel requires the entire range's full state to match, not
            // just each constituent block independently carrying CanAlias.
            if mapping.state != first.state
                || mapping.permissions != (READ | WRITE)
                || mapping.attributes != 0
            {
                return Err(MapMemoryError::InvalidCurrentMemory);
            }
            cursor = mapping.end().min(source_end);
        }
        if self
            .mappings
            .values()
            .any(|mapping| destination < mapping.end() && mapping.base < destination_end)
        {
            return Err(MapMemoryError::InvalidCurrentMemory);
        }

        let mut mappings = self.mappings.clone();
        Self::split_mapping_at(&mut mappings, source);
        Self::split_mapping_at(&mut mappings, source_end);
        for (_, mapping) in mappings.range_mut(source..source_end) {
            mapping.permissions = 0;
            mapping.attributes = ATTRIBUTE_LOCKED;
        }
        mappings.insert(
            destination,
            Mapping {
                base: destination,
                size,
                state: STATE_STACK,
                permissions: READ | WRITE,
                attributes: 0,
                label: "stack alias".into(),
            },
        );
        let mut aliases = self.stack_aliases.clone();
        aliases.insert(destination, StackAlias { source, size });
        self.mappings = mappings;
        self.stack_aliases = aliases;
        Ok(())
    }

    /// UnmapMemory for a whole MapMemory alias: the destination must be the
    /// exact alias of `source` with `size`. The alias mapping is removed and
    /// the source returns to read-write and unlocked, re-merged with its
    /// neighbors. Pages stay in canonical source storage, so no data moves.
    fn unmap_stack_alias_unsynced(
        &mut self,
        destination: u64,
        source: u64,
        size: u64,
    ) -> Result<(), MapMemoryError> {
        self.flush_tlb();
        self.clear_exclusive();
        if !destination.is_multiple_of(PAGE_SIZE) || !source.is_multiple_of(PAGE_SIZE) {
            return Err(MapMemoryError::InvalidAddress);
        }
        if size == 0 || !size.is_multiple_of(PAGE_SIZE) {
            return Err(MapMemoryError::InvalidSize);
        }
        if self.stack_aliases.get(&destination) != Some(&StackAlias { source, size }) {
            return Err(MapMemoryError::InvalidCurrentMemory);
        }
        let mut mappings = self.mappings.clone();
        mappings.remove(&destination);
        let source_end = source + size;
        for (_, mapping) in mappings.range_mut(source..source_end) {
            mapping.permissions = READ | WRITE;
            mapping.attributes = 0;
        }
        Self::merge_mappings(&mut mappings, source, source_end);
        self.mappings = mappings;
        self.stack_aliases.remove(&destination);
        Ok(())
    }

    /// Merge adjacent mappings touching [start, end) whose state, permissions,
    /// attributes and label are identical (undoing a split).
    fn merge_mappings(mappings: &mut BTreeMap<u64, Mapping>, start: u64, end: u64) {
        let first = mappings
            .range(..start)
            .next_back()
            .filter(|(_, mapping)| mapping.end() == start)
            .map_or(start, |(&base, _)| base);
        let keys: Vec<u64> = mappings.range(first..=end).map(|(&base, _)| base).collect();
        let mut current: Option<u64> = None;
        for base in keys {
            if let Some(previous) = current {
                let (left, right) = (&mappings[&previous], &mappings[&base]);
                if left.end() == base
                    && left.state == right.state
                    && left.permissions == right.permissions
                    && left.attributes == right.attributes
                    && left.label == right.label
                {
                    let extra = right.size;
                    mappings.remove(&base);
                    mappings
                        .get_mut(&previous)
                        .expect("previous mapping was found above")
                        .size += extra;
                    continue;
                }
            }
            current = Some(base);
        }
    }

    /// The file backing guest range `[address, address + length)` and the
    /// range's offset in it, when every page of it is mapped and their
    /// backing pages are contiguous: a device can then share the range
    /// in place (on Linux, through a udmabuf of the file). None otherwise,
    /// or on hosts without such a file.
    pub fn shared_span(&self, address: u64, length: u64) -> Option<(i32, u64)> {
        let file = self.pages.host.shared_file()?;
        let start = address & !(PAGE_SIZE - 1);
        let end = address
            .checked_add(length.max(1))?
            .next_multiple_of(PAGE_SIZE);
        let (first, _) = self.guest_view_page(start)?;
        let mut page = start + PAGE_SIZE;
        while page < end {
            let (backing, _) = self.guest_view_page(page)?;
            if backing != first + (page - start) {
                return None;
            }
            page += PAGE_SIZE;
        }
        Some((file, first + (address - start)))
    }

    fn backing_page(&self, page: u64) -> u64 {
        if let Some((&destination, alias)) = self.stack_aliases.range(..=page).next_back()
            && page < destination + alias.size
        {
            return alias.source + (page - destination);
        }
        page
    }

    fn reject_active_alias_mutation(
        &self,
        address: u64,
        end: u64,
        operation: &str,
    ) -> Result<(), Trap> {
        if self.stack_aliases.iter().any(|(&destination, alias)| {
            (address < destination + alias.size && destination < end)
                || (address < alias.source + alias.size && alias.source < end)
        }) {
            return Err(Trap::InvalidState {
                message: format!(
                    "{operation} overlaps an active MapMemory source or destination; alias teardown is unsupported"
                ),
            });
        }
        Ok(())
    }

    fn protect_translated_code_unsynced(&mut self, base: u64, size: u64) -> Result<(), Trap> {
        self.flush_tlb();
        // Layout changes conservatively end any reservation (spurious STXR
        // failure is permitted), even if the request is later rejected.
        self.clear_exclusive();
        let end = self.range_end(base, size, "protect translated code")?;
        self.translated_ranges.push((base, end));
        Ok(())
    }

    pub fn query(&self, address: u64) -> Mapping {
        if address >= self.address_space_end {
            return Mapping {
                base: self.address_space_end,
                size: 0u64.wrapping_sub(self.address_space_end),
                state: STATE_INACCESSIBLE,
                permissions: 0,
                attributes: 0,
                label: "outside address space".into(),
            };
        }
        if let Some((_, mapping)) = self.mappings.range(..=address).next_back()
            && address < mapping.end()
        {
            return mapping.clone();
        }
        let start = self
            .mappings
            .range(..=address)
            .next_back()
            .map_or(0, |(_, m)| m.end());
        let end = self
            .mappings
            .range(address..)
            .next()
            .map_or(self.address_space_end, |(base, _)| *base);
        Mapping {
            base: start,
            size: end - start,
            state: STATE_FREE,
            permissions: 0,
            attributes: 0,
            label: "unmapped".into(),
        }
    }

    fn validate_access(&self, address: u64, size: usize, mask: u8) -> Result<(), Trap> {
        let access = match mask {
            WRITE => "write",
            EXECUTE => "execute",
            _ => "read",
        };
        let end = self.range_end(address, size as u64, access)?;
        let mut cursor = address;
        while cursor < end {
            let Some(mapping) = self.mapping_at(cursor).filter(|mapping| {
                mapping.state != STATE_FREE && mapping.permissions & mask == mask
            }) else {
                return Err(Trap::MemoryFault {
                    address: cursor,
                    access: access.into(),
                });
            };
            cursor = mapping.end().min(end);
        }
        if mask == WRITE
            && self
                .translated_ranges
                .iter()
                .any(|(start, stop)| address < *stop && *start < end)
        {
            return Err(Trap::InvalidState {
                message: format!("self-modifying translated code at {address:#x} is unsupported"),
            });
        }
        if mask == WRITE {
            // Virtual permissions belong to the address used by the guest,
            // but translated-code protection also follows physical backing.
            for (&destination, alias) in self.stack_aliases.range(..end) {
                let overlap_start = address.max(destination);
                let overlap_end = end.min(destination + alias.size);
                if overlap_start < overlap_end {
                    let backing_start = alias.source + (overlap_start - destination);
                    let backing_end = alias.source + (overlap_end - destination);
                    if self
                        .translated_ranges
                        .iter()
                        .any(|(start, stop)| backing_start < *stop && *start < backing_end)
                    {
                        return Err(Trap::InvalidState {
                            message: format!(
                                "self-modifying translated code through alias at {address:#x} is unsupported"
                            ),
                        });
                    }
                }
            }
        }
        Ok(())
    }

    /// The TLB hit is inlined into callers; everything else is out of line.
    #[inline(always)]
    pub fn read(&self, address: u64, out: &mut [u8]) -> Result<(), Trap> {
        if let Some((slot, offset)) = self.fast_page(address, out.len(), READ) {
            out.copy_from_slice(&self.tlb_page(slot)[offset..offset + out.len()]);
            return Ok(());
        }
        self.read_slow(address, out)
    }

    /// Whether the readable bytes at `address` equal `expected`, compared
    /// in place (host-side change checks of large guest buffers).
    pub fn equals(&self, address: u64, expected: &[u8]) -> Result<bool, Trap> {
        self.validate_access(address, expected.len(), READ)?;
        let mut done = 0;
        while done < expected.len() {
            let current = address + done as u64;
            let page = current & !(PAGE_SIZE - 1);
            let offset = (current - page) as usize;
            let count = (PAGE_SIZE as usize - offset).min(expected.len() - done);
            let same = match self.pages.get(&self.backing_page(page)) {
                Some(bytes) => bytes[offset..offset + count] == expected[done..done + count],
                None => expected[done..done + count].iter().all(|byte| *byte == 0),
            };
            if !same {
                return Ok(false);
            }
            done += count;
        }
        Ok(true)
    }

    #[cold]
    #[inline(never)]
    fn read_slow(&self, address: u64, out: &mut [u8]) -> Result<(), Trap> {
        self.validate_access(address, out.len(), READ)?;
        let mut done = 0;
        while done < out.len() {
            let current = address + done as u64;
            let page = current & !(PAGE_SIZE - 1);
            let offset = (current - page) as usize;
            let count = (PAGE_SIZE as usize - offset).min(out.len() - done);
            if let Some(bytes) = self.pages.get(&self.backing_page(page)) {
                out[done..done + count].copy_from_slice(&bytes[offset..offset + count]);
            } else {
                out[done..done + count].fill(0);
            }
            done += count;
        }
        Ok(())
    }

    #[inline]
    pub fn check_execute(&self, address: u64, size: usize) -> Result<(), Trap> {
        if self.fast_page(address, size, EXECUTE).is_some() {
            return Ok(());
        }
        self.validate_access(address, size, EXECUTE)
    }

    /// Preflight a bounded write before a non-memory service state change. The
    /// caller must keep the single-thread mapping state unchanged until write.
    pub fn check_write(&self, address: u64, size: usize) -> Result<(), Trap> {
        self.validate_access(address, size, WRITE)
    }

    #[inline(always)]
    pub fn write(&mut self, address: u64, bytes: &[u8]) -> Result<(), Trap> {
        if let Some((slot, offset)) = self.fast_store(address, bytes.len()) {
            self.note_fast_store(address, bytes.len());
            self.tlb_page_mut(slot)[offset..offset + bytes.len()].copy_from_slice(bytes);
            return Ok(());
        }
        self.write_slow(address, bytes)
    }

    #[cold]
    #[inline(never)]
    fn write_slow(&mut self, address: u64, bytes: &[u8]) -> Result<(), Trap> {
        // Validate the entire write first, so a fault cannot leave a partial store.
        self.validate_access(address, bytes.len(), WRITE)?;
        self.write_pages(address, bytes);
        Ok(())
    }

    /// One LSE CAS step in the prototype's single-guest-thread memory model.
    ///
    /// Arm DDI0557A.b B1.1 requires natural alignment and both read and write
    /// permission checks even when the comparison fails. Validate everything
    /// before exposing an observed value or changing memory. Failed comparisons
    /// leave memory alone, one of the architecturally permitted behaviors.
    ///
    /// Exclusive `&mut Memory` access and the absence of a guest scheduler make
    /// this indivisible to the sole guest thread. The fences conservatively
    /// honor encoded acquire/release bits, including stronger ordering for zero
    /// register forms; they do not make this a concurrent guest-memory engine.
    pub fn compare_exchange_single_thread(
        &mut self,
        address: u64,
        bytes: u8,
        expected: u64,
        desired: u64,
        acquire: bool,
        release: bool,
    ) -> Result<u64, Trap> {
        if !matches!(bytes, 1 | 2 | 4 | 8) {
            return Err(Trap::InvalidState {
                message: format!("unsupported CAS element size {bytes}"),
            });
        }
        if !address.is_multiple_of(u64::from(bytes)) {
            return Err(Trap::MemoryFault {
                address,
                access: format!("unaligned {bytes}-byte atomic compare-and-swap"),
            });
        }
        let len = usize::from(bytes);
        let mask = u64::MAX >> (64 - bytes * 8);
        if let Some((slot, offset)) = self.fast_rmw(address, len) {
            let mut encoded = [0; 8];
            encoded[..len].copy_from_slice(&self.tlb_page(slot)[offset..offset + len]);
            let observed = u64::from_le_bytes(encoded);
            if observed == expected & mask {
                if release {
                    std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
                }
                self.note_fast_store(address, len);
                self.tlb_page_mut(slot)[offset..offset + len]
                    .copy_from_slice(&desired.to_le_bytes()[..len]);
            }
            if acquire {
                std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
            }
            return Ok(observed);
        }
        self.validate_access(address, len, READ)?;
        // Check WRITE separately so translated-code protection also applies.
        self.validate_access(address, len, WRITE)?;
        let mut encoded = [0; 8];
        self.read(address, &mut encoded[..len])?;
        let observed = u64::from_le_bytes(encoded);
        if observed == expected & mask {
            if release {
                std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            }
            // All fallible checks are complete before committing this store.
            self.write_pages(address, &desired.to_le_bytes()[..len]);
        }
        if acquire {
            std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
        }
        Ok(observed)
    }

    /// One LSE LDADD step, returning the old value after an unconditional store.
    ///
    /// Arm DDI0557A.b B1.1 and B11.1.5-.7 require natural alignment and both
    /// READ and WRITE permissions, including zero addends and STADD aliases.
    /// Check the entire span and translated-code protection before observing
    /// data or committing a store. Integers use the prototype's little-endian
    /// memory model; arithmetic wraps at the selected element width.
    /// <https://documentation-service.arm.com/static/5fb7d32fd77dd807b9a80c61>
    ///
    /// The emitter supplies effective acquire (encoded A && Rt != 31) and
    /// encoded release R. Exclusive &mut Memory access with one guest thread
    /// makes this indivisible to that thread. The fences do not implement a
    /// concurrent guest-memory model, scheduler, or exclusive monitor.
    pub fn fetch_add_single_thread(
        &mut self,
        address: u64,
        bytes: u8,
        addend: u64,
        acquire: bool,
        release: bool,
    ) -> Result<u64, Trap> {
        if !matches!(bytes, 1 | 2 | 4 | 8) {
            return Err(Trap::InvalidState {
                message: format!("unsupported LDADD element size {bytes}"),
            });
        }
        if !address.is_multiple_of(u64::from(bytes)) {
            return Err(Trap::MemoryFault {
                address,
                access: format!("unaligned {bytes}-byte atomic fetch-add"),
            });
        }
        let len = usize::from(bytes);
        let mask = u64::MAX >> (64 - bytes * 8);
        if let Some((slot, offset)) = self.fast_rmw(address, len) {
            let mut encoded = [0; 8];
            encoded[..len].copy_from_slice(&self.tlb_page(slot)[offset..offset + len]);
            let observed = u64::from_le_bytes(encoded);
            let updated = observed.wrapping_add(addend) & mask;
            if release {
                std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            }
            self.note_fast_store(address, len);
            self.tlb_page_mut(slot)[offset..offset + len]
                .copy_from_slice(&updated.to_le_bytes()[..len]);
            if acquire {
                std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
            }
            return Ok(observed);
        }
        self.validate_access(address, len, READ)?;
        // WRITE must be checked separately so translated-code protection applies.
        self.validate_access(address, len, WRITE)?;
        let mut encoded = [0; 8];
        self.read(address, &mut encoded[..len])?;
        let observed = u64::from_le_bytes(encoded);
        let updated = observed.wrapping_add(addend) & mask;
        if release {
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        }
        // Even add-zero is an RMW. All fallible checks precede its store.
        self.write_pages(address, &updated.to_le_bytes()[..len]);
        if acquire {
            std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
        }
        Ok(observed)
    }

    /// LSE LD<op>/SWP read-modify-write on a single PE. Returns the old value.
    pub fn atomic_rmw_single_thread(
        &mut self,
        address: u64,
        bytes: u8,
        op: AtomicOp,
        operand: u64,
        acquire: bool,
        release: bool,
    ) -> Result<u64, Trap> {
        if !matches!(bytes, 1 | 2 | 4 | 8) {
            return Err(Trap::InvalidState {
                message: format!("unsupported atomic element size {bytes}"),
            });
        }
        if !address.is_multiple_of(u64::from(bytes)) {
            return Err(Trap::MemoryFault {
                address,
                access: format!("unaligned {bytes}-byte atomic {op:?}"),
            });
        }
        let len = usize::from(bytes);
        let fast = self.fast_rmw(address, len);
        if fast.is_none() {
            self.validate_access(address, len, READ)?;
            self.validate_access(address, len, WRITE)?;
        }
        let mut encoded = [0; 8];
        match fast {
            Some((slot, offset)) => {
                encoded[..len].copy_from_slice(&self.tlb_page(slot)[offset..offset + len])
            }
            None => self.read(address, &mut encoded[..len])?,
        }
        let observed = u64::from_le_bytes(encoded);
        let bits = u32::from(bytes) * 8;
        let mask = u64::MAX >> (64 - bits);
        let operand = operand & mask;
        let signed = |value: u64| crate::cpu::sign_extend(value, bytes * 8) as i64;
        let updated = match op {
            AtomicOp::Clear => observed & !operand,
            AtomicOp::ExclusiveOr => observed ^ operand,
            AtomicOp::Set => observed | operand,
            AtomicOp::SignedMax => {
                if signed(observed) >= signed(operand) {
                    observed
                } else {
                    operand
                }
            }
            AtomicOp::SignedMin => {
                if signed(observed) <= signed(operand) {
                    observed
                } else {
                    operand
                }
            }
            AtomicOp::UnsignedMax => observed.max(operand),
            AtomicOp::UnsignedMin => observed.min(operand),
            AtomicOp::Swap => operand,
        } & mask;
        if release {
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        }
        match fast {
            Some((slot, offset)) => {
                self.note_fast_store(address, len);
                self.tlb_page_mut(slot)[offset..offset + len]
                    .copy_from_slice(&updated.to_le_bytes()[..len]);
            }
            None => self.write_pages(address, &updated.to_le_bytes()[..len]),
        }
        if acquire {
            std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
        }
        Ok(observed)
    }

    /// LDAPR/B/H is an ordered read, not a read-modify-write operation.
    ///
    /// This prototype uses baseline natural alignment for ordered accesses;
    /// FEAT_LSE2/nAA alignment relaxation is not modeled. READ permission is
    /// sufficient. The acquire fence is conservatively stronger than RCpc in
    /// this one-guest-thread engine, including when the result is discarded.
    pub fn load_acquire_rcpc_single_thread(&self, address: u64, bytes: u8) -> Result<u64, Trap> {
        if !matches!(bytes, 1 | 2 | 4 | 8) {
            return Err(Trap::InvalidState {
                message: format!("unsupported LDAPR element size {bytes}"),
            });
        }
        if !address.is_multiple_of(u64::from(bytes)) {
            return Err(Trap::MemoryFault {
                address,
                access: format!("unaligned {bytes}-byte LDAPR without LSE2 alignment relaxation"),
            });
        }
        let mut encoded = [0; 8];
        self.read(address, &mut encoded[..usize::from(bytes)])?;
        std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
        Ok(u64::from_le_bytes(encoded))
    }

    /// LDAR/B/H is an RCsc acquire read, distinct from LDAPR's RCpc acquire.
    ///
    /// Both currently perform a successful read followed by a conservative
    /// host acquire fence in this single-guest-thread engine. Keeping separate
    /// helpers preserves the architectural distinction; these fences alone
    /// are not a concurrent guest RCsc memory model. The fence is conservative
    /// for Rt == 31, whose discarded read need not carry acquire ordering.
    /// Baseline natural alignment
    /// is enforced, without FEAT_LSE2/nAA relaxation. Only READ permission is
    /// required, and even a discarded result must perform and validate the read.
    pub fn load_acquire_rcsc_single_thread(&self, address: u64, bytes: u8) -> Result<u64, Trap> {
        if !matches!(bytes, 1 | 2 | 4 | 8) {
            return Err(Trap::InvalidState {
                message: format!("unsupported LDAR element size {bytes}"),
            });
        }
        if !address.is_multiple_of(u64::from(bytes)) {
            return Err(Trap::MemoryFault {
                address,
                access: format!("unaligned {bytes}-byte LDAR without LSE2 alignment relaxation"),
            });
        }
        let mut encoded = [0; 8];
        self.read(address, &mut encoded[..usize::from(bytes)])?;
        std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
        Ok(u64::from_le_bytes(encoded))
    }

    /// STLR/B/H is a WRITE-only ordered store in the single-guest-thread model.
    /// As with LDAPR, baseline natural alignment is enforced; FEAT_LSE2/nAA
    /// alignment relaxation is not modeled. All span/permission/code-protection
    /// checks precede the release fence and the store, so faults change nothing.
    pub fn store_release_single_thread(
        &mut self,
        address: u64,
        bytes: u8,
        value: u64,
    ) -> Result<(), Trap> {
        if !matches!(bytes, 1 | 2 | 4 | 8) {
            return Err(Trap::InvalidState {
                message: format!("unsupported STLR element size {bytes}"),
            });
        }
        if !address.is_multiple_of(u64::from(bytes)) {
            return Err(Trap::MemoryFault {
                address,
                access: format!("unaligned {bytes}-byte STLR without LSE2 alignment relaxation"),
            });
        }
        let len = usize::from(bytes);
        self.validate_access(address, len, WRITE)?;
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        self.write_pages(address, &value.to_le_bytes()[..len]);
        Ok(())
    }

    /// LDXR/LDAXR B/H/W/X: a read that marks the local exclusive monitor.
    ///
    /// Arm ARM DDI0487 C6.2 (LDXR/LDAXR) and B2.9: exclusive accesses require
    /// natural alignment even with FEAT_LSE2, and need only READ permission.
    /// A faulting access leaves the previous monitor state unchanged. The
    /// acquire fence is conservative in this single-guest-thread engine.
    pub fn load_exclusive_single_thread(
        &mut self,
        address: u64,
        bytes: u8,
        acquire: bool,
    ) -> Result<u64, Trap> {
        if !matches!(bytes, 1 | 2 | 4 | 8) {
            return Err(Trap::InvalidState {
                message: format!("unsupported exclusive load element size {bytes}"),
            });
        }
        if !address.is_multiple_of(u64::from(bytes)) {
            return Err(Trap::MemoryFault {
                address,
                access: format!("unaligned {bytes}-byte load-exclusive"),
            });
        }
        let mut encoded = [0; 8];
        self.read(address, &mut encoded[..usize::from(bytes)])?;
        let page = address & !(PAGE_SIZE - 1);
        self.exclusive = Some(ExclusiveReservation {
            address,
            backing: self.backing_page(page) + (address - page),
            bytes,
        });
        if acquire {
            std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
        }
        Ok(u64::from_le_bytes(encoded))
    }

    /// STXR/STLXR B/H/W/X. Returns `Ok(true)` when the store was performed
    /// (status 0) and `Ok(false)` when it was not (status 1).
    ///
    /// Follows AArch64.ExclusiveMonitorsPass in DDI0487: alignment is checked
    /// first and faults regardless of monitor state; a store whose address and
    /// size do not match the reservation fails without a permission check
    /// (mismatched sizes are CONSTRAINED UNPREDICTABLE, and failing is one
    /// permitted outcome); a matching store checks WRITE permission and
    /// translated-code protection before committing. A fault leaves memory and
    /// the monitor untouched. Every completed STXR leaves the monitor open.
    pub fn store_exclusive_single_thread(
        &mut self,
        address: u64,
        bytes: u8,
        value: u64,
        release: bool,
    ) -> Result<bool, Trap> {
        if !matches!(bytes, 1 | 2 | 4 | 8) {
            return Err(Trap::InvalidState {
                message: format!("unsupported exclusive store element size {bytes}"),
            });
        }
        if !address.is_multiple_of(u64::from(bytes)) {
            return Err(Trap::MemoryFault {
                address,
                access: format!("unaligned {bytes}-byte store-exclusive"),
            });
        }
        let passes = matches!(
            self.exclusive,
            Some(r) if r.address == address && r.bytes == bytes
        );
        if !passes {
            self.exclusive = None;
            return Ok(false);
        }
        let len = usize::from(bytes);
        if let Some((slot, offset)) = self.fast_store(address, len) {
            self.exclusive = None;
            if release {
                std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            }
            self.tlb_page_mut(slot)[offset..offset + len]
                .copy_from_slice(&value.to_le_bytes()[..len]);
            return Ok(true);
        }
        self.validate_access(address, len, WRITE)?;
        self.exclusive = None;
        if release {
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        }
        self.write_pages(address, &value.to_le_bytes()[..len]);
        Ok(true)
    }

    /// Loader-only initialization; it still requires every byte to be mapped.
    pub fn initialize(&mut self, address: u64, bytes: &[u8]) -> Result<(), Trap> {
        let end = self.range_end(address, bytes.len() as u64, "initialize")?;
        let mut cursor = address;
        while cursor < end {
            let mapping = self.query(cursor);
            if mapping.state == STATE_FREE {
                return Err(Trap::MemoryFault {
                    address: cursor,
                    access: "initialize unmapped region".into(),
                });
            }
            cursor = mapping.end().min(end);
        }
        self.write_pages(address, bytes);
        Ok(())
    }

    fn write_pages(&mut self, address: u64, bytes: &[u8]) {
        let mut done = 0;
        while done < bytes.len() {
            let current = address + done as u64;
            let page = current & !(PAGE_SIZE - 1);
            let offset = (current - page) as usize;
            let count = (PAGE_SIZE as usize - offset).min(bytes.len() - done);
            let backing = self.backing_page(page);
            if let Some(reservation) = self.exclusive {
                // Any store overlapping the marked bytes, through any alias,
                // clears the local monitor before data changes.
                let start = backing + offset as u64;
                if start < reservation.backing + u64::from(reservation.bytes)
                    && reservation.backing < start + count as u64
                {
                    self.exclusive = None;
                }
            }
            if self.pages.slot_of(backing).is_none() {
                // Cached read-only while never written (see `fill_tlb`).
                self.tlb[(page / PAGE_SIZE) as usize % TLB_ENTRIES].set(EMPTY_TLB_ENTRY);
            }
            let target = self.pages.get_or_insert(backing);
            target[offset..offset + count].copy_from_slice(&bytes[done..done + count]);
            done += count;
        }
    }

    fn unmap_exact_unsynced(&mut self, base: u64, size: u64) -> Result<(), Trap> {
        self.flush_tlb();
        // Layout changes conservatively end any reservation (spurious STXR
        // failure is permitted), even if the request is later rejected.
        self.clear_exclusive();
        let mapping = self.mappings.get(&base).ok_or_else(|| Trap::InvalidState {
            message: "unmap requires an existing exact mapping".into(),
        })?;
        if mapping.size != size {
            return Err(Trap::InvalidState {
                message: "unmap size does not match the existing mapping".into(),
            });
        }
        self.reject_active_alias_mutation(base, mapping.end(), "unmap")?;
        self.mappings.remove(&base);
        self.pages.release(base, base + size);
        Ok(())
    }

    fn resize_heap_unsynced(
        &mut self,
        base: u64,
        old_size: u64,
        new_size: u64,
    ) -> Result<(), Trap> {
        self.flush_tlb();
        // Layout changes conservatively end any reservation (spurious STXR
        // failure is permitted), even if the request is later rejected.
        self.clear_exclusive();
        let old_end = self.range_end(base, old_size, "resize heap")?;
        let new_end = self.range_end(base, new_size, "resize heap")?;
        if !base.is_multiple_of(PAGE_SIZE)
            || !old_size.is_multiple_of(PAGE_SIZE)
            || !new_size.is_multiple_of(PAGE_SIZE)
        {
            return Err(Trap::InvalidState {
                message: "heap must be page aligned".into(),
            });
        }
        let mut cursor = base;
        while cursor < old_end {
            let mapping = self.query(cursor);
            if mapping.state != STATE_NORMAL || mapping.base < base || mapping.end() > old_end {
                return Err(Trap::InvalidState {
                    message: "heap range contains an unexpected mapping".into(),
                });
            }
            cursor = mapping.end();
        }
        if new_size > old_size {
            self.map(
                old_end,
                new_size - old_size,
                READ | WRITE,
                STATE_NORMAL,
                "heap",
            )?;
        } else if new_size < old_size {
            // Growth and an unchanged/retained prefix do not discard backing.
            // Keep the existing non-alias permission policy; reject only the
            // removed tail if it intersects either side of a live alias.
            self.reject_active_alias_mutation(new_end, old_end, "shrink heap")?;
            self.split_at(new_end);
            self.split_at(old_end);
            self.mappings
                .retain(|start, _| *start < new_end || *start >= old_end);
            self.pages.release(new_end, old_end);
        }
        Ok(())
    }

    /// Lock a range for a kernel object (transfer memory, code memory):
    /// every page must be in one of `states`, have exactly `required`
    /// permissions and no attributes. On success the owner's permissions
    /// become `permissions` and `attribute` is set. Returns false, changing
    /// nothing, when the range does not meet the requirements.
    fn lock_range_unsynced(
        &mut self,
        address: u64,
        size: u64,
        states: &[u32],
        required: u8,
        permissions: u8,
        attribute: u32,
    ) -> Result<bool, Trap> {
        self.flush_tlb();
        self.clear_exclusive();
        if !address.is_multiple_of(PAGE_SIZE) || size == 0 || !size.is_multiple_of(PAGE_SIZE) {
            return Ok(false);
        }
        let Ok(end) = self.range_end(address, size, "lock") else {
            return Ok(false);
        };
        let mut cursor = address;
        while cursor < end {
            let mapping = self.query(cursor);
            if !states.contains(&mapping.state)
                || mapping.permissions != required
                || mapping.attributes != 0
            {
                return Ok(false);
            }
            cursor = mapping.end().min(end);
        }
        self.reject_active_alias_mutation(address, end, "lock")?;
        self.split_at(address);
        self.split_at(end);
        for (_, mapping) in self.mappings.range_mut(address..end) {
            mapping.permissions = permissions;
            mapping.attributes |= attribute;
        }
        Ok(true)
    }

    /// Reverse [`Memory::lock_range`]: every page must carry `attribute`;
    /// it is cleared and the owner's permissions become `permissions`.
    fn unlock_range_unsynced(
        &mut self,
        address: u64,
        size: u64,
        attribute: u32,
        permissions: u8,
    ) -> Result<(), Trap> {
        self.flush_tlb();
        self.clear_exclusive();
        let end = self.range_end(address, size, "unlock")?;
        let mut cursor = address;
        while cursor < end {
            let mapping = self.query(cursor);
            if mapping.attributes & attribute != attribute {
                return Err(Trap::InvalidState {
                    message: format!("unlock of {address:#x}+{size:#x} found an unlocked page"),
                });
            }
            cursor = mapping.end().min(end);
        }
        self.split_at(address);
        self.split_at(end);
        for (_, mapping) in self.mappings.range_mut(address..end) {
            mapping.permissions = permissions;
            mapping.attributes &= !attribute;
        }
        Ok(())
    }

    fn split_at(&mut self, point: u64) {
        Self::split_mapping_at(&mut self.mappings, point);
    }

    fn split_mapping_at(mappings: &mut BTreeMap<u64, Mapping>, point: u64) {
        let containing = mappings.range(..point).next_back().map(|(_, m)| m.clone());
        if let Some(mut left) = containing
            && point < left.end()
        {
            let mut right = left.clone();
            right.base = point;
            right.size = left.end() - point;
            left.size = point - left.base;
            mappings.insert(left.base, left);
            mappings.insert(right.base, right);
        }
    }

    fn set_permissions_unsynced(
        &mut self,
        address: u64,
        size: u64,
        permissions: u8,
    ) -> Result<(), Trap> {
        self.flush_tlb();
        // Layout changes conservatively end any reservation (spurious STXR
        // failure is permitted), even if the request is later rejected.
        self.clear_exclusive();
        if !address.is_multiple_of(PAGE_SIZE)
            || size == 0
            || !size.is_multiple_of(PAGE_SIZE)
            || permissions & !7 != 0
        {
            return Err(Trap::InvalidState {
                message: "invalid permission range".into(),
            });
        }
        let end = self.range_end(address, size, "permissions")?;
        self.reject_active_alias_mutation(address, end, "change permissions")?;
        let mut cursor = address;
        let mut unchanged = true;
        while cursor < end {
            let mapping = self.query(cursor);
            if mapping.state == STATE_FREE {
                return Err(Trap::MemoryFault {
                    address: cursor,
                    access: "permissions".into(),
                });
            }
            unchanged &= mapping.permissions == permissions;
            cursor = mapping.end().min(end);
        }
        if unchanged {
            return Ok(());
        }
        self.split_at(address);
        self.split_at(end);
        for (_, mapping) in self.mappings.range_mut(address..end) {
            mapping.permissions = permissions;
        }
        Ok(())
    }

    fn set_attributes_unsynced(
        &mut self,
        address: u64,
        size: u64,
        mask: u32,
        value: u32,
    ) -> Result<(), Trap> {
        self.flush_tlb();
        // Layout changes conservatively end any reservation (spurious STXR
        // failure is permitted), even if the request is later rejected.
        self.clear_exclusive();
        // Uncached (8) toggles; PermissionLocked (0x10) may only be set. A
        // locked range is then refused by SetMemoryPermission, which accepts
        // only unattributed mappings. Unlocking is never permitted.
        if mask & !(8 | ATTRIBUTE_PERMISSION_LOCKED) != 0
            || value & !mask != 0
            || (mask & ATTRIBUTE_PERMISSION_LOCKED != 0 && value & ATTRIBUTE_PERMISSION_LOCKED == 0)
        {
            return Err(Trap::InvalidState {
                message: "unsupported memory attribute mask".into(),
            });
        }
        if !address.is_multiple_of(PAGE_SIZE) || size == 0 || !size.is_multiple_of(PAGE_SIZE) {
            return Err(Trap::InvalidState {
                message: "invalid attribute range".into(),
            });
        }
        let end = self.range_end(address, size, "attributes")?;
        self.reject_active_alias_mutation(address, end, "change attributes")?;
        let mut cursor = address;
        let mut unchanged = true;
        while cursor < end {
            let mapping = self.query(cursor);
            if mapping.state == STATE_FREE {
                return Err(Trap::MemoryFault {
                    address: cursor,
                    access: "attributes".into(),
                });
            }
            unchanged &= mapping.attributes & mask == value;
            cursor = mapping.end().min(end);
        }
        if unchanged {
            return Ok(());
        }
        self.split_at(address);
        self.split_at(end);
        for (_, mapping) in self.mappings.range_mut(address..end) {
            mapping.attributes = (mapping.attributes & !mask) | value;
        }
        Ok(())
    }

    #[inline(always)]
    pub fn read_u8(&self, address: u64) -> Result<u8, Trap> {
        let mut b = [0; 1];
        self.read(address, &mut b)?;
        Ok(b[0])
    }
    #[inline(always)]
    pub fn read_u16(&self, address: u64) -> Result<u16, Trap> {
        let mut b = [0; 2];
        self.read(address, &mut b)?;
        Ok(u16::from_le_bytes(b))
    }
    #[inline(always)]
    pub fn read_u32(&self, address: u64) -> Result<u32, Trap> {
        let mut b = [0; 4];
        self.read(address, &mut b)?;
        Ok(u32::from_le_bytes(b))
    }
    #[inline(always)]
    pub fn read_u64(&self, address: u64) -> Result<u64, Trap> {
        let mut b = [0; 8];
        self.read(address, &mut b)?;
        Ok(u64::from_le_bytes(b))
    }
    /// Generated code's loads and stores: straight through the guest view,
    /// where the guest's permissions are host page protections (a
    /// forbidden access faults in the host; see `host_memory`). A store may
    /// leave an exclusive reservation in place, which the architecture
    /// permits for the same PE's own stores.
    #[inline(always)]
    pub fn load_u8(&self, address: u64) -> u8 {
        self.pages.host.load(address)
    }
    #[inline(always)]
    pub fn load_u16(&self, address: u64) -> u16 {
        u16::from_le(self.pages.host.load(address))
    }
    #[inline(always)]
    pub fn load_u32(&self, address: u64) -> u32 {
        u32::from_le(self.pages.host.load(address))
    }
    #[inline(always)]
    pub fn load_u64(&self, address: u64) -> u64 {
        u64::from_le(self.pages.host.load(address))
    }
    #[inline(always)]
    pub fn store_u8(&mut self, address: u64, value: u8) {
        self.pages.host.store(address, value)
    }
    #[inline(always)]
    pub fn store_u16(&mut self, address: u64, value: u16) {
        self.pages.host.store(address, value.to_le())
    }
    #[inline(always)]
    pub fn store_u32(&mut self, address: u64, value: u32) {
        self.pages.host.store(address, value.to_le())
    }
    #[inline(always)]
    pub fn store_u64(&mut self, address: u64, value: u64) {
        self.pages.host.store(address, value.to_le())
    }

    #[inline(always)]
    pub fn write_u8(&mut self, address: u64, value: u8) -> Result<(), Trap> {
        self.write(address, &[value])
    }
    #[inline(always)]
    pub fn write_u16(&mut self, address: u64, value: u16) -> Result<(), Trap> {
        self.write(address, &value.to_le_bytes())
    }
    #[inline(always)]
    pub fn write_u32(&mut self, address: u64, value: u32) -> Result<(), Trap> {
        self.write(address, &value.to_le_bytes())
    }
    #[inline(always)]
    pub fn write_u64(&mut self, address: u64, value: u64) -> Result<(), Trap> {
        self.write(address, &value.to_le_bytes())
    }
}

#[cfg(all(test, any(target_os = "linux", windows)))]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_os = "linux")]
    fn shared_spans_follow_contiguous_backing() {
        let mut memory = Memory::new(0x80_0000_0000);
        memory
            .map(0x10000, 0x4000, READ | WRITE, STATE_NORMAL, "heap")
            .unwrap();
        // A plain mapping is backed at its own address.
        let (file, offset) = memory.shared_span(0x10010, 0x3000).unwrap();
        assert!(file >= 0);
        assert_eq!(offset, 0x10010);
        // Past the mapping, or unmapped: not shareable.
        assert!(memory.shared_span(0x13000, 0x2000).is_none());
        assert!(memory.shared_span(0x20000, 0x10).is_none());
        // An alias is backed by its source pages.
        memory.map_stack_alias(0x40000, 0x11000, 0x1000).unwrap();
        assert_eq!(memory.shared_span(0x40008, 8).unwrap().1, 0x11008);
    }

    #[test]
    fn guest_view_follows_host_writes_aliases_and_unmaps() {
        let mut memory = Memory::new(0x80_0000_0000);
        memory
            .map(0x10000, 0x4000, READ | WRITE, STATE_NORMAL, "heap")
            .unwrap();
        memory.write_u64(0x10008, 0x1122_3344_5566_7788).unwrap();
        assert_eq!(memory.load_u64(0x10008), 0x1122_3344_5566_7788);
        memory.store_u32(0x10ffe, 0xaabb_ccdd); // across a page boundary
        assert_eq!(memory.read_u32(0x10ffe).unwrap(), 0xaabb_ccdd);
        memory.map_stack_alias(0x40000, 0x11000, 0x1000).unwrap();
        // The alias shows the source page's bytes, both ways.
        assert_eq!(memory.load_u16(0x40000) as u32, 0xaabb);
        memory.store_u8(0x40010, 0x5a);
        // The source is locked while aliased; afterwards it holds the store.
        assert!(memory.read_u8(0x11010).is_err());
        memory.unmap_stack_alias(0x40000, 0x11000, 0x1000).unwrap();
        assert_eq!(memory.read_u8(0x11010).unwrap(), 0x5a);
        assert_eq!(memory.load_u8(0x11010), 0x5a);
        // Unmapped then remapped storage reads as zero again.
        memory.unmap_exact(0x10000, 0x4000).unwrap();
        memory
            .map(0x10000, 0x4000, READ | WRITE, STATE_NORMAL, "heap")
            .unwrap();
        assert_eq!(memory.load_u64(0x10008), 0);
    }

    #[test]
    fn unmap_memory_restores_the_source_exactly() {
        let mut memory = Memory::new(0x80_0000_0000);
        memory
            .map(0x10000, 0x4000, READ | WRITE, STATE_NORMAL, "heap")
            .unwrap();
        memory
            .map(0x40000, 0x2000, READ | WRITE, STATE_STACK, "stack region")
            .unwrap();
        memory.unmap_exact(0x40000, 0x2000).unwrap();
        memory.write_u64(0x11000, 42).unwrap();
        memory.map_stack_alias(0x40000, 0x11000, 0x2000).unwrap();
        assert_eq!(memory.read_u64(0x40000).unwrap(), 42);
        assert_eq!(
            memory.unmap_stack_alias(0x40000, 0x12000, 0x2000),
            Err(MapMemoryError::InvalidCurrentMemory)
        );
        memory.unmap_stack_alias(0x40000, 0x11000, 0x2000).unwrap();
        assert!(memory.read_u64(0x40000).is_err());
        assert_eq!(memory.read_u64(0x11000).unwrap(), 42);
        let source = memory.query(0x11000);
        assert_eq!(
            (
                source.base,
                source.size,
                source.permissions,
                source.attributes
            ),
            (0x10000, 0x4000, READ | WRITE, 0)
        );
        assert_eq!(memory.stack_aliases().count(), 0);
    }

    #[derive(Debug, PartialEq, Eq)]
    struct MemorySnapshot {
        mappings: BTreeMap<u64, Mapping>,
        pages: Pages,
        aliases: BTreeMap<u64, StackAlias>,
        translated_ranges: Vec<(u64, u64)>,
    }

    fn snapshot(memory: &Memory) -> MemorySnapshot {
        MemorySnapshot {
            mappings: memory.mappings.clone(),
            pages: memory.pages.clone(),
            aliases: memory.stack_aliases.clone(),
            translated_ranges: memory.translated_ranges.clone(),
        }
    }

    #[test]
    fn stack_alias_metadata_and_sparse_backing_are_shared_for_each_data_state() {
        for state in [STATE_CODE_DATA, STATE_NORMAL, STATE_ALIAS_CODE_DATA] {
            let mut m = Memory::new(0x20000);
            m.map(0x1000, 0x4000, READ | WRITE, state, "data with BSS")
                .unwrap();
            m.write_u64(0x2ff8, 0x8877_6655_4433_2211).unwrap();
            assert_eq!(m.read_u64(0x3000).unwrap(), 0);
            let pages = m.pages.clone();
            m.map_stack_alias(0x8000, 0x2000, 0x2000).unwrap();
            assert_eq!(m.pages, pages); // No data allocation or byte copy.
            assert_eq!(m.resident_bytes(), PAGE_SIZE);
            assert_eq!(
                m.stack_aliases().collect::<Vec<_>>(),
                [(0x8000, 0x2000, 0x2000)]
            );
            assert_eq!(
                m.query(0x2100),
                Mapping {
                    base: 0x2000,
                    size: 0x2000,
                    state,
                    permissions: 0,
                    attributes: 1,
                    label: "data with BSS".into(),
                }
            );
            assert_eq!(
                m.query(0x8123),
                Mapping {
                    base: 0x8000,
                    size: 0x2000,
                    state: 0x0b,
                    permissions: 3,
                    attributes: 0,
                    label: "stack alias".into(),
                }
            );
            for base in [0x1000, 0x4000] {
                assert_eq!(
                    m.query(base),
                    Mapping {
                        base,
                        size: 0x1000,
                        state,
                        permissions: 3,
                        attributes: 0,
                        label: "data with BSS".into(),
                    }
                );
            }
            assert_eq!(m.read_u64(0x8ff8).unwrap(), 0x8877_6655_4433_2211);
            assert_eq!(m.read_u64(0x9000).unwrap(), 0);
            assert_eq!(m.resident_bytes(), PAGE_SIZE);
            assert!(matches!(m.read_u8(0x2000), Err(Trap::MemoryFault { .. })));
            assert!(matches!(
                m.write_u8(0x3000, 1),
                Err(Trap::MemoryFault { .. })
            ));
            assert!(matches!(
                m.check_execute(0x8000, 4),
                Err(Trap::MemoryFault { .. })
            ));
            // Loader-privileged source initialization bypasses user permissions.
            // Its immediate visibility through the alias proves this is shared
            // backing rather than a snapshot copied during MapMemory.
            m.initialize(0x2010, &[0x91, 0x82, 0x73, 0x64]).unwrap();
            assert_eq!(m.read_u32(0x8010).unwrap(), 0x6473_8291);
            // Loader initialization through the destination resolves the same
            // canonical source page, including its first materialization.
            m.initialize(0x9010, &[0x55, 0xaa]).unwrap();
            assert_eq!(&m.pages[&0x3000][0x10..0x12], &[0x55, 0xaa]);
            m.write_u64(0x8ffc, 0x1234_5678_9abc_def0).unwrap();
            assert_eq!(m.read_u64(0x8ffc).unwrap(), 0x1234_5678_9abc_def0);
            assert_eq!(&m.pages[&0x2000][0xffc..], &[0xf0, 0xde, 0xbc, 0x9a]);
            assert_eq!(&m.pages[&0x3000][..4], &[0x78, 0x56, 0x34, 0x12]);
            assert_eq!(
                m.pages.keys().copied().collect::<Vec<_>>(),
                [0x2000, 0x3000]
            );
            assert_eq!(m.resident_bytes(), 2 * PAGE_SIZE);
        }
    }

    #[test]
    fn stack_alias_atomics_use_virtual_permissions_and_canonical_storage() {
        let mut m = Memory::new(0x10000);
        m.map(0x1000, PAGE_SIZE, READ | WRITE, STATE_CODE_DATA, "data")
            .unwrap();
        m.write_u64(0x1000, u64::MAX).unwrap();
        m.map_stack_alias(0x8000, 0x1000, PAGE_SIZE).unwrap();
        assert_eq!(
            m.fetch_add_single_thread(0x8000, 8, 1, true, true).unwrap(),
            u64::MAX
        );
        assert_eq!(
            m.compare_exchange_single_thread(0x8000, 8, 0, 7, true, true)
                .unwrap(),
            0
        );
        assert_eq!(
            m.compare_exchange_single_thread(0x8000, 8, 0, 9, false, false)
                .unwrap(),
            7
        );
        assert_eq!(m.load_acquire_rcpc_single_thread(0x8000, 8).unwrap(), 7);
        assert_eq!(m.load_acquire_rcsc_single_thread(0x8000, 8).unwrap(), 7);
        m.store_release_single_thread(0x8008, 4, 0xfeed_beef)
            .unwrap();
        assert_eq!(
            &m.pages[&0x1000][..12],
            &[7, 0, 0, 0, 0, 0, 0, 0, 0xef, 0xbe, 0xed, 0xfe]
        );
        m.initialize(0x1020, &[5]).unwrap();
        assert_eq!(
            m.fetch_add_single_thread(0x8020, 1, 257, false, false)
                .unwrap(),
            5
        );
        assert_eq!(m.pages[&0x1000][0x20], 6);
        let before = snapshot(&m);
        assert!(matches!(
            m.fetch_add_single_thread(0x1000, 8, 1, false, false),
            Err(Trap::MemoryFault { .. })
        ));
        assert!(matches!(
            m.store_release_single_thread(0x1008, 4, 0),
            Err(Trap::MemoryFault { .. })
        ));
        assert_eq!(snapshot(&m), before);
        assert_eq!(m.resident_bytes(), PAGE_SIZE);
    }

    #[test]
    fn stack_alias_access_faults_and_translated_backing_protection_are_atomic() {
        let mut m = Memory::new(0x10000);
        m.map(0x1000, 0x2000, READ | WRITE, STATE_NORMAL, "data")
            .unwrap();
        m.write_u64(0x2040, 9).unwrap();
        m.map_stack_alias(0x8000, 0x1000, 0x2000).unwrap();
        let before = snapshot(&m);
        let mut out = [0xa5; 8];
        assert!(matches!(
            m.read(0x9ffc, &mut out),
            Err(Trap::MemoryFault { .. })
        ));
        assert_eq!(out, [0xa5; 8]);
        assert!(matches!(
            m.write_u64(0x9ffc, u64::MAX),
            Err(Trap::MemoryFault { .. })
        ));
        assert!(matches!(
            m.initialize(0x9ffc, &[0xff; 8]),
            Err(Trap::MemoryFault { .. })
        ));
        assert_eq!(snapshot(&m), before);
        m.protect_translated_code(0x2040, 8).unwrap();
        m.protect_translated_code(0x8010, 8).unwrap();
        let before = snapshot(&m);
        assert_eq!(m.read_u64(0x9040).unwrap(), 9);
        assert!(matches!(
            m.check_write(0x9040, 8),
            Err(Trap::InvalidState { .. })
        ));
        assert!(matches!(
            m.write_u64(0x9040, 0),
            Err(Trap::InvalidState { .. })
        ));
        assert!(matches!(
            m.write(0x903f, &[0xff; 10]),
            Err(Trap::InvalidState { .. })
        ));
        assert!(matches!(
            m.compare_exchange_single_thread(0x9040, 8, 0, 1, false, false),
            Err(Trap::InvalidState { .. })
        ));
        assert!(matches!(
            m.fetch_add_single_thread(0x9040, 8, 0, false, false),
            Err(Trap::InvalidState { .. })
        ));
        assert!(matches!(
            m.store_release_single_thread(0x9040, 8, 1),
            Err(Trap::InvalidState { .. })
        ));
        assert!(matches!(
            m.write_u64(0x8010, 0),
            Err(Trap::InvalidState { .. })
        ));
        assert_eq!(snapshot(&m), before);
        // Half-open bounds permit writes immediately before protected backing.
        m.write_u64(0x9038, 0x1234).unwrap();
        assert_eq!(m.read_u64(0x9038).unwrap(), 0x1234);
        assert_eq!(m.read_u64(0x9040).unwrap(), 9);
    }

    #[test]
    fn stack_alias_geometry_and_destination_failures_leave_all_state_unchanged() {
        let mut m = Memory::new(0x10000);
        m.map(0x1000, 0x3000, READ | WRITE, STATE_NORMAL, "data")
            .unwrap();
        m.initialize(0x1000, &[1, 2, 3, 4]).unwrap();
        m.map(0x9000, PAGE_SIZE, READ, STATE_CODE, "occupied")
            .unwrap();
        let before = snapshot(&m);
        let last_page = !(PAGE_SIZE - 1);
        for (destination, source, size, expected) in [
            (0x8001, 0x1000, PAGE_SIZE, MapMemoryError::InvalidAddress),
            (0x8000, 0x1001, PAGE_SIZE, MapMemoryError::InvalidAddress),
            (0x8000, 0x1000, 0, MapMemoryError::InvalidSize),
            (0x8000, 0x1000, PAGE_SIZE + 1, MapMemoryError::InvalidSize),
            (0xf000, 0x1000, 0x2000, MapMemoryError::InvalidCurrentMemory),
            (0x8000, 0xf000, 0x2000, MapMemoryError::InvalidCurrentMemory),
            (
                last_page,
                0x1000,
                PAGE_SIZE,
                MapMemoryError::InvalidCurrentMemory,
            ),
            (
                0x8000,
                last_page,
                PAGE_SIZE,
                MapMemoryError::InvalidCurrentMemory,
            ),
            (
                0x8000,
                0x4000,
                PAGE_SIZE,
                MapMemoryError::InvalidCurrentMemory,
            ),
            (0x8000, 0x3000, 0x2000, MapMemoryError::InvalidCurrentMemory),
            (0x8000, 0x1000, 0x2000, MapMemoryError::InvalidCurrentMemory),
            (
                0x1000,
                0x1000,
                PAGE_SIZE,
                MapMemoryError::InvalidCurrentMemory,
            ),
            (0x2000, 0x1000, 0x2000, MapMemoryError::InvalidCurrentMemory),
        ] {
            assert_eq!(m.map_stack_alias(destination, source, size), Err(expected));
            assert_eq!(snapshot(&m), before);
        }
        // A late destination failure must not have locked the source or left
        // an alias record behind; the valid retry still maps its original bytes.
        m.map_stack_alias(0x8000, 0x1000, PAGE_SIZE).unwrap();
        assert_eq!(m.read_u32(0x8000).unwrap(), 0x0403_0201);
    }

    #[test]
    fn stack_alias_source_requires_uniform_supported_rw_unattributed_memory() {
        for state in [STATE_CODE, STATE_STACK, STATE_TLS] {
            let mut m = Memory::new(0x10000);
            m.map(0x1000, 0x2000, READ | WRITE, state, "unsupported source")
                .unwrap();
            let before = snapshot(&m);
            assert_eq!(
                m.map_stack_alias(0x8000, 0x1000, 0x2000),
                Err(MapMemoryError::InvalidCurrentMemory)
            );
            assert_eq!(snapshot(&m), before);
        }
        for scenario in 0..5 {
            let mut m = Memory::new(0x10000);
            m.map(0x1000, PAGE_SIZE, READ | WRITE, STATE_CODE_DATA, "data")
                .unwrap();
            m.map(0x2000, PAGE_SIZE, READ | WRITE, STATE_CODE_DATA, "BSS")
                .unwrap();
            match scenario {
                0 => m.mappings.get_mut(&0x2000).unwrap().state = STATE_NORMAL,
                1 => m.set_permissions(0x2000, PAGE_SIZE, READ).unwrap(),
                2 => m
                    .set_permissions(0x2000, PAGE_SIZE, READ | WRITE | EXECUTE)
                    .unwrap(),
                3 => m.set_attributes(0x2000, PAGE_SIZE, 8, 8).unwrap(),
                4 => m.unmap_exact(0x2000, PAGE_SIZE).unwrap(),
                _ => unreachable!(),
            }
            let before = snapshot(&m);
            assert_eq!(
                m.map_stack_alias(0x8000, 0x1000, 0x2000),
                Err(MapMemoryError::InvalidCurrentMemory)
            );
            assert_eq!(snapshot(&m), before);
        }
        let mut m = Memory::new(0x10000);
        for (base, label) in [(0x1000, "data"), (0x2000, "BSS")] {
            m.map(base, PAGE_SIZE, READ | WRITE, STATE_CODE_DATA, label)
                .unwrap();
        }
        // Equal metadata across distinct backing ranges is accepted; labels
        // are diagnostic strings, not part of the kernel's state comparison.
        m.map_stack_alias(0x8000, 0x1000, 0x2000).unwrap();
        assert_eq!(m.resident_bytes(), 0);
        assert_eq!(m.read_u64(0x8ffc).unwrap(), 0);
    }

    #[test]
    fn stack_alias_chains_and_collisions_are_rejected_but_adjacent_aliases_resolve_independently() {
        let mut m = Memory::new(0x10000);
        m.map(0x1000, 0x3000, READ | WRITE, STATE_CODE_DATA, "data")
            .unwrap();
        m.initialize(0x1ffc, &[1, 2, 3, 4]).unwrap();
        m.initialize(0x3000, &[5, 6, 7, 8]).unwrap();
        m.map_stack_alias(0x8000, 0x1000, PAGE_SIZE).unwrap();
        let before = snapshot(&m);
        for (destination, source, size) in [
            (0x9000, 0x1000, PAGE_SIZE), // Already locked source.
            (0x9000, 0x8000, PAGE_SIZE), // Stack cannot be an alias source.
            (0x8000, 0x2000, PAGE_SIZE), // Occupied destination/collision.
            (0x7000, 0x2000, 0x2000),    // Partly overlaps existing destination.
        ] {
            assert_eq!(
                m.map_stack_alias(destination, source, size),
                Err(MapMemoryError::InvalidCurrentMemory)
            );
            assert_eq!(snapshot(&m), before);
        }
        m.map_stack_alias(0x9000, 0x3000, PAGE_SIZE).unwrap();
        assert_eq!(m.read_u64(0x8ffc).unwrap(), 0x0807_0605_0403_0201);
        m.write_u64(0x8ffc, 0x1234_5678_9abc_def0).unwrap();
        assert_eq!(&m.pages[&0x1000][0xffc..], &[0xf0, 0xde, 0xbc, 0x9a]);
        assert_eq!(&m.pages[&0x3000][..4], &[0x78, 0x56, 0x34, 0x12]);
        assert!(!m.pages.contains_key(&0x2000));
        assert_eq!(
            m.stack_aliases().collect::<Vec<_>>(),
            [(0x8000, 0x1000, PAGE_SIZE), (0x9000, 0x3000, PAGE_SIZE)]
        );
    }

    #[test]
    fn stack_alias_mutation_and_heap_shrink_guards_preserve_live_backing() {
        let mut m = Memory::new(0x20000);
        m.map(0x1000, 0x3000, READ | WRITE, STATE_NORMAL, "heap")
            .unwrap();
        m.map_stack_alias(0x8000, 0x2000, PAGE_SIZE).unwrap();
        m.write_u64(0x8000, 42).unwrap();
        let before = snapshot(&m);
        for address in [0x2000, 0x8000] {
            for permission in [0, READ, READ | WRITE] {
                assert!(matches!(
                    m.set_permissions(address, PAGE_SIZE, permission),
                    Err(Trap::InvalidState { .. })
                ));
                assert_eq!(snapshot(&m), before);
            }
            for (mask, value) in [(0, 0), (8, 8)] {
                assert!(matches!(
                    m.set_attributes(address, PAGE_SIZE, mask, value),
                    Err(Trap::InvalidState { .. })
                ));
                assert_eq!(snapshot(&m), before);
            }
            assert!(matches!(
                m.unmap_exact(address, PAGE_SIZE),
                Err(Trap::InvalidState { .. })
            ));
            assert_eq!(snapshot(&m), before);
        }
        // Ranges partially touching a live side must not mutate earlier pages.
        assert!(m.set_permissions(0x1000, 0x2000, READ).is_err());
        assert!(m.set_attributes(0x1000, 0x2000, 8, 8).is_err());
        assert!(m.set_attributes(0x1000, PAGE_SIZE, 0x10, 0).is_err());
        assert_eq!(snapshot(&m), before);
        m.resize_heap(0x1000, 0x3000, 0x5000).unwrap();
        assert_eq!(m.read_u64(0x8000).unwrap(), 42);
        assert_eq!(m.query(0x2000).attributes, ATTRIBUTE_LOCKED);
        assert_eq!(m.read_u64(0x4000).unwrap(), 0);
        m.write_u64(0x5000, 99).unwrap();
        m.resize_heap(0x1000, 0x5000, 0x3000).unwrap();
        assert_eq!(m.query(0x4000).state, STATE_FREE);
        assert!(!m.pages.contains_key(&0x5000));
        m.resize_heap(0x1000, 0x3000, 0x2000).unwrap();
        let retained = snapshot(&m);
        for size in [0x1000, 0] {
            assert!(matches!(
                m.resize_heap(0x1000, 0x2000, size),
                Err(Trap::InvalidState { .. })
            ));
            assert_eq!(snapshot(&m), retained);
        }
        assert_eq!(m.read_u64(0x8000).unwrap(), 42);
        m.resize_heap(0x1000, 0x2000, 0x3000).unwrap();
        // Independent non-alias metadata remains mutable under the existing API.
        m.set_permissions(0x1000, PAGE_SIZE, READ).unwrap();
        m.set_attributes(0x1000, PAGE_SIZE, 8, 8).unwrap();
        m.map(0xa000, PAGE_SIZE, READ | WRITE, STATE_NORMAL, "unrelated")
            .unwrap();
        m.write_u64(0xa000, 123).unwrap();
        m.unmap_exact(0xa000, PAGE_SIZE).unwrap();
        assert_eq!(m.read_u64(0x8000).unwrap(), 42);
        assert_eq!(
            m.stack_aliases().collect::<Vec<_>>(),
            [(0x8000, 0x2000, PAGE_SIZE)]
        );
    }

    #[test]
    fn stlr_needs_only_write_permission_and_preserves_neighbor_bytes() {
        for bytes in [1, 2, 4, 8] {
            let mut m = Memory::new(0x10000);
            m.map(0x1000, PAGE_SIZE, WRITE, STATE_NORMAL, "write-only")
                .unwrap();
            let initial = 0x1122_3344_5566_7788u64;
            let value = 0xfedc_ba98_7654_3210;
            let mask = u64::MAX >> (64 - bytes * 8);
            m.initialize(0x1000, &initial.to_le_bytes()).unwrap();
            m.store_release_single_thread(0x1000, bytes, value).unwrap();
            m.set_permissions(0x1000, PAGE_SIZE, READ).unwrap();
            assert_eq!(
                m.read_u64(0x1000).unwrap(),
                (initial & !mask) | (value & mask)
            );
        }
    }

    #[test]
    fn stlr_faults_before_writing_on_bad_alignment_permissions_bounds_or_code_protection() {
        let mut m = Memory::new(0x10000);
        m.map(0x1000, PAGE_SIZE, READ | WRITE, STATE_NORMAL, "data")
            .unwrap();
        m.map(0x2000, PAGE_SIZE, READ, STATE_NORMAL, "read-only")
            .unwrap();
        m.initialize(0x1000, &u64::MAX.to_le_bytes()).unwrap();
        m.initialize(0x2000, &u64::MAX.to_le_bytes()).unwrap();
        for bytes in [2, 4, 8] {
            assert!(matches!(
                m.store_release_single_thread(0x1001, bytes, 0),
                Err(Trap::MemoryFault {
                    address: 0x1001,
                    ..
                })
            ));
        }
        for address in [0x2000, 0x3000, 0x10000] {
            assert!(matches!(
                m.store_release_single_thread(address, 8, 0),
                Err(Trap::MemoryFault { .. })
            ));
        }
        assert!(matches!(
            m.store_release_single_thread(0x1000, 3, 0),
            Err(Trap::InvalidState { .. })
        ));
        m.protect_translated_code(0x1000, PAGE_SIZE).unwrap();
        assert!(matches!(
            m.store_release_single_thread(0x1000, 8, 0),
            Err(Trap::InvalidState { .. })
        ));
        assert_eq!(m.read_u64(0x1000).unwrap(), u64::MAX);
        assert_eq!(m.read_u64(0x2000).unwrap(), u64::MAX);
        assert_eq!(m.resident_bytes(), 2 * PAGE_SIZE);
    }

    #[test]
    fn ldapr_is_read_only_and_zero_extends_each_element_width() {
        let mut m = Memory::new(0x10000);
        m.map(
            0x1000,
            PAGE_SIZE,
            READ,
            STATE_NORMAL,
            "read-only atomic data",
        )
        .unwrap();
        let initial = 0xfedc_ba98_7654_3210u64;
        m.initialize(0x1000, &initial.to_le_bytes()).unwrap();
        for bytes in [1, 2, 4, 8] {
            let mask = u64::MAX >> (64 - bytes * 8);
            assert_eq!(
                m.load_acquire_rcpc_single_thread(0x1000, bytes).unwrap(),
                initial & mask
            );
            assert_eq!(m.read_u64(0x1000).unwrap(), initial);
        }
        // An acquire byte read has no wider-element alignment requirement.
        assert_eq!(m.load_acquire_rcpc_single_thread(0x1001, 1).unwrap(), 0x32);
    }

    #[test]
    fn ldapr_checks_baseline_alignment_read_permission_and_bounds() {
        let mut m = Memory::new(0x10000);
        m.map(0x1000, PAGE_SIZE, READ, STATE_NORMAL, "read-only")
            .unwrap();
        m.map(0x2000, PAGE_SIZE, WRITE, STATE_NORMAL, "write-only")
            .unwrap();
        for bytes in [2, 4, 8] {
            assert!(matches!(
                m.load_acquire_rcpc_single_thread(0x1001, bytes),
                Err(Trap::MemoryFault {
                    address: 0x1001,
                    ..
                })
            ));
        }
        for address in [0x2000, 0x3000, 0x10000] {
            assert!(matches!(
                m.load_acquire_rcpc_single_thread(address, 8),
                Err(Trap::MemoryFault { .. })
            ));
        }
        assert!(matches!(
            m.load_acquire_rcpc_single_thread(0x1000, 3),
            Err(Trap::InvalidState { .. })
        ));
        assert_eq!(m.resident_bytes(), 0);
    }

    #[test]
    fn ldar_reads_all_widths_unsigned_with_read_permission_only() {
        let mut m = Memory::new(0x10000);
        m.map(
            0x1000,
            PAGE_SIZE,
            READ,
            STATE_NORMAL,
            "read-only ordered data",
        )
        .unwrap();
        let initial = 0xf0e1_d2c3_b4a5_9687u64;
        m.initialize(0x1000, &initial.to_le_bytes()).unwrap();
        for bytes in [1, 2, 4, 8] {
            let mask = u64::MAX >> (64 - bytes * 8);
            assert_eq!(
                m.load_acquire_rcsc_single_thread(0x1000, bytes).unwrap(),
                initial & mask
            );
        }
        assert_eq!(m.load_acquire_rcsc_single_thread(0x1001, 1).unwrap(), 0x96);
        assert_eq!(m.read_u64(0x1000).unwrap(), initial);
        assert_eq!(m.resident_bytes(), PAGE_SIZE);
    }

    #[test]
    fn ldar_faults_on_bad_alignment_nonreadable_or_unmapped_memory_without_mutation() {
        let mut m = Memory::new(0x10000);
        m.map(0x1000, PAGE_SIZE, READ, STATE_NORMAL, "read-only")
            .unwrap();
        m.map(0x2000, PAGE_SIZE, WRITE, STATE_NORMAL, "write-only")
            .unwrap();
        for bytes in [2, 4, 8] {
            assert!(matches!(
                m.load_acquire_rcsc_single_thread(0x1001, bytes),
                Err(Trap::MemoryFault {
                    address: 0x1001,
                    ..
                })
            ));
        }
        for address in [0x2000, 0x3000, 0x10000] {
            assert!(matches!(
                m.load_acquire_rcsc_single_thread(address, 8),
                Err(Trap::MemoryFault { .. })
            ));
        }
        for bytes in [0, 3, 16] {
            assert!(matches!(
                m.load_acquire_rcsc_single_thread(0x1000, bytes),
                Err(Trap::InvalidState { .. })
            ));
        }
        assert_eq!(m.resident_bytes(), 0);
        assert_eq!(m.read_u64(0x1000).unwrap(), 0);
    }

    #[test]
    fn ldadd_returns_old_wraps_all_widths_and_preserves_neighbor_bytes_for_each_ordering() {
        for bytes in [1, 2, 4, 8] {
            for acquire in [false, true] {
                for release in [false, true] {
                    let mut m = Memory::new(0x10000);
                    m.map(0x1000, PAGE_SIZE, READ | WRITE, STATE_NORMAL, "atomic data")
                        .unwrap();
                    let mask = u64::MAX >> (64 - bytes * 8);
                    let initial = (0x1122_3344_5566_7788 & !mask) | mask;
                    let addend = (0xfedc_ba98_7654_3200 & !mask) | 1;
                    m.write_u64(0x1000, initial).unwrap();
                    assert_eq!(
                        m.fetch_add_single_thread(0x1000, bytes, addend, acquire, release)
                            .unwrap(),
                        mask
                    );
                    assert_eq!(m.read_u64(0x1000).unwrap(), initial & !mask);
                    assert_eq!(
                        m.fetch_add_single_thread(0x1000, bytes, 0, acquire, release)
                            .unwrap(),
                        0
                    );
                    assert_eq!(m.read_u64(0x1000).unwrap(), initial & !mask);
                }
            }
        }
    }

    #[test]
    fn ldadd_requires_read_and_write_even_for_zero_addends() {
        for permissions in [READ, WRITE, 0] {
            let mut m = Memory::new(0x10000);
            m.map(0x1000, PAGE_SIZE, permissions, STATE_NORMAL, "restricted")
                .unwrap();
            m.initialize(0x1000, &7u64.to_le_bytes()).unwrap();
            for addend in [0, 1] {
                let expected_access = if permissions & READ == 0 {
                    "read"
                } else {
                    "write"
                };
                assert!(matches!(
                    m.fetch_add_single_thread(0x1000, 8, addend, true, true),
                    Err(Trap::MemoryFault { address: 0x1000, access }) if access == expected_access
                ));
            }
            m.set_permissions(0x1000, PAGE_SIZE, READ).unwrap();
            assert_eq!(m.read_u64(0x1000).unwrap(), 7);
        }
    }

    #[test]
    fn ldadd_alignment_bounds_invalid_width_and_code_protection_fault_without_writing() {
        let mut m = Memory::new(0x10000);
        m.map(0x1000, PAGE_SIZE, READ | WRITE, STATE_NORMAL, "atomic data")
            .unwrap();
        m.write_u64(0x1000, u64::MAX).unwrap();
        for bytes in [2, 4, 8] {
            assert!(matches!(
                m.fetch_add_single_thread(0x1001, bytes, 1, true, true),
                Err(Trap::MemoryFault {
                    address: 0x1001,
                    ..
                })
            ));
        }
        for address in [0x2000, 0x10000] {
            assert!(matches!(
                m.fetch_add_single_thread(address, 8, 1, false, false),
                Err(Trap::MemoryFault { .. })
            ));
        }
        for bytes in [0, 3, 16] {
            assert!(matches!(
                m.fetch_add_single_thread(0x1000, bytes, 1, false, false),
                Err(Trap::InvalidState { .. })
            ));
        }
        m.protect_translated_code(0x1000, PAGE_SIZE).unwrap();
        for addend in [0, 1] {
            assert!(matches!(
                m.fetch_add_single_thread(0x1000, 8, addend, false, false),
                Err(Trap::InvalidState { .. })
            ));
        }
        assert_eq!(m.read_u64(0x1000).unwrap(), u64::MAX);
        assert_eq!(m.resident_bytes(), PAGE_SIZE);
    }

    #[test]
    fn cas_masks_inputs_returns_observed_value_and_preserves_neighbor_bytes() {
        for bytes in [1, 2, 4, 8] {
            let mut m = Memory::new(0x10000);
            m.map(0x1000, PAGE_SIZE, READ | WRITE, STATE_NORMAL, "atomic data")
                .unwrap();
            let initial = 0xfedc_ba98_7654_3210;
            let desired = 0x1234_5678_9abc_def0;
            let mask = u64::MAX >> (64 - bytes * 8);
            m.write_u64(0x1000, initial).unwrap();
            // High input bits outside the memory element must be ignored.
            let expected = (initial & mask) | (0xffff_aaaa_5555_0000 & !mask);
            assert_eq!(
                m.compare_exchange_single_thread(0x1000, bytes, expected, desired, true, true)
                    .unwrap(),
                initial & mask
            );
            assert_eq!(
                m.read_u64(0x1000).unwrap(),
                (initial & !mask) | (desired & mask)
            );
            let before = m.read_u64(0x1000).unwrap();
            assert_eq!(
                m.compare_exchange_single_thread(0x1000, bytes, desired ^ 1, 0, true, true)
                    .unwrap(),
                desired & mask
            );
            assert_eq!(m.read_u64(0x1000).unwrap(), before);
        }
    }

    #[test]
    fn cas_requires_read_and_write_permissions_even_when_comparison_would_fail() {
        for permissions in [READ, WRITE, 0] {
            let mut m = Memory::new(0x10000);
            m.map(0x1000, PAGE_SIZE, permissions, STATE_NORMAL, "restricted")
                .unwrap();
            m.initialize(0x1000, &7u64.to_le_bytes()).unwrap();
            for expected in [7, 8] {
                assert!(matches!(
                    m.compare_exchange_single_thread(0x1000, 8, expected, 9, true, true),
                    Err(Trap::MemoryFault {
                        address: 0x1000,
                        ..
                    })
                ));
            }
            m.set_permissions(0x1000, PAGE_SIZE, READ).unwrap();
            assert_eq!(m.read_u64(0x1000).unwrap(), 7);
        }
    }

    #[test]
    fn cas_alignment_unmapped_and_translated_code_faults_do_not_write() {
        let mut m = Memory::new(0x10000);
        m.map(0x1000, PAGE_SIZE, READ | WRITE, STATE_NORMAL, "atomic data")
            .unwrap();
        m.write_u64(0x1000, u64::MAX).unwrap();
        for bytes in [2, 4, 8] {
            assert!(matches!(
                m.compare_exchange_single_thread(0x1001, bytes, u64::MAX, 0, false, false),
                Err(Trap::MemoryFault {
                    address: 0x1001,
                    ..
                })
            ));
            assert_eq!(m.read_u64(0x1000).unwrap(), u64::MAX);
        }
        assert!(matches!(
            m.compare_exchange_single_thread(0x2000, 8, 0, 1, false, false),
            Err(Trap::MemoryFault {
                address: 0x2000,
                ..
            })
        ));
        assert!(matches!(
            m.compare_exchange_single_thread(0x10000, 8, 0, 1, false, false),
            Err(Trap::MemoryFault { .. })
        ));
        assert!(matches!(
            m.compare_exchange_single_thread(0x1000, 3, 0, 1, false, false),
            Err(Trap::InvalidState { .. })
        ));
        m.protect_translated_code(0x1000, PAGE_SIZE).unwrap();
        for expected in [u64::MAX, 0] {
            assert!(matches!(
                m.compare_exchange_single_thread(0x1000, 8, expected, 1, false, false),
                Err(Trap::InvalidState { .. })
            ));
        }
        assert_eq!(m.read_u64(0x1000).unwrap(), u64::MAX);
    }

    #[test]
    fn exclusive_pair_succeeds_once_zero_extends_and_preserves_neighbors() {
        let mut m = Memory::new(0x10000);
        m.map(0x1000, PAGE_SIZE, READ | WRITE, STATE_NORMAL, "data")
            .unwrap();
        for bytes in [1u8, 2, 4, 8] {
            for ordered in [false, true] {
                m.write_u64(0x1000, 0x8877_6655_4433_2211).unwrap();
                m.write_u64(0x1008, u64::MAX).unwrap();
                let mask = u64::MAX >> (64 - u32::from(bytes) * 8);
                let loaded =
                    m.load_exclusive_single_thread(0x1008 - u64::from(bytes), bytes, ordered);
                let shift = 64 - u32::from(bytes) * 8;
                assert_eq!(loaded.unwrap(), (0x8877_6655_4433_2211 >> shift) & mask);
                assert_eq!(
                    m.exclusive_reservation(),
                    Some((0x1008 - u64::from(bytes), bytes))
                );
                let stored = m
                    .store_exclusive_single_thread(
                        0x1008 - u64::from(bytes),
                        bytes,
                        0x0123_4567_89ab_cdef,
                        ordered,
                    )
                    .unwrap();
                assert!(stored);
                assert_eq!(m.exclusive_reservation(), None);
                assert_eq!(m.read_u64(0x1008).unwrap(), u64::MAX);
                let expected = if bytes == 8 {
                    0x0123_4567_89ab_cdef
                } else {
                    (0x8877_6655_4433_2211 & (u64::MAX >> (u32::from(bytes) * 8)))
                        | ((0x0123_4567_89ab_cdef & mask) << shift)
                };
                assert_eq!(m.read_u64(0x1000).unwrap(), expected);
                // The monitor is open again: a repeated store must fail.
                assert!(
                    !m.store_exclusive_single_thread(0x1008 - u64::from(bytes), bytes, 0, ordered)
                        .unwrap()
                );
                assert_eq!(m.read_u64(0x1000).unwrap(), expected);
            }
        }
    }

    #[test]
    fn exclusive_store_fails_without_matching_reservation_and_always_opens_monitor() {
        let mut m = Memory::new(0x10000);
        m.map(0x1000, PAGE_SIZE, READ | WRITE, STATE_NORMAL, "data")
            .unwrap();
        m.write_u64(0x1000, 7).unwrap();
        assert!(
            !m.store_exclusive_single_thread(0x1000, 8, 9, false)
                .unwrap()
        );
        for (address, bytes) in [(0x1008, 8), (0x1000, 4), (0x1004, 4)] {
            m.load_exclusive_single_thread(0x1000, 8, false).unwrap();
            assert!(
                !m.store_exclusive_single_thread(address, bytes, 9, true)
                    .unwrap()
            );
            assert_eq!(m.exclusive_reservation(), None);
        }
        // A newer LDXR replaces the older reservation.
        m.load_exclusive_single_thread(0x1000, 8, false).unwrap();
        m.load_exclusive_single_thread(0x1010, 4, false).unwrap();
        assert!(
            !m.store_exclusive_single_thread(0x1000, 8, 9, false)
                .unwrap()
        );
        m.load_exclusive_single_thread(0x1000, 8, false).unwrap();
        m.clear_exclusive();
        assert!(
            !m.store_exclusive_single_thread(0x1000, 8, 9, false)
                .unwrap()
        );
        assert_eq!(m.read_u64(0x1000).unwrap(), 7);
        assert_eq!(m.read_u64(0x1008).unwrap(), 0);
    }

    #[test]
    fn overlapping_stores_and_layout_changes_clear_the_reservation_but_disjoint_stores_do_not() {
        let mut m = Memory::new(0x20000);
        m.map(0x1000, PAGE_SIZE, READ | WRITE, STATE_CODE_DATA, "data")
            .unwrap();
        m.map(0x3000, PAGE_SIZE, READ | WRITE, STATE_NORMAL, "other")
            .unwrap();
        m.load_exclusive_single_thread(0x1010, 8, false).unwrap();
        m.write_u64(0x1008, 1).unwrap();
        m.write_u64(0x1018, 1).unwrap();
        m.write_u32(0x3010, 1).unwrap();
        assert!(
            m.store_exclusive_single_thread(0x1010, 8, 5, false)
                .unwrap()
        );
        // Overlap at the last byte, the whole span, and straddling the start.
        for (overlap, len) in [(0x1017u64, 1usize), (0x1010, 8), (0x100e, 4)] {
            m.load_exclusive_single_thread(0x1010, 8, false).unwrap();
            m.write(overlap, &vec![2; len]).unwrap();
            assert_eq!(m.exclusive_reservation(), None, "{overlap:#x}");
        }
        // Adjacent stores that end at or begin after the marked bytes do not.
        m.load_exclusive_single_thread(0x1010, 8, false).unwrap();
        m.write(0x100c, &[3; 4]).unwrap();
        m.write(0x1018, &[3; 4]).unwrap();
        assert_eq!(m.exclusive_reservation(), Some((0x1010, 8)));
        m.clear_exclusive();
        m.load_exclusive_single_thread(0x1010, 8, false).unwrap();
        m.compare_exchange_single_thread(0x1010, 8, 0, 0, false, false)
            .unwrap();
        // A failed CAS performs no store, so the reservation survives it.
        assert!(m.exclusive_reservation().is_some());
        m.fetch_add_single_thread(0x1014, 4, 0, false, false)
            .unwrap();
        assert_eq!(m.exclusive_reservation(), None);
        // A store through a stack alias of the same backing clears it.
        m.map_stack_alias(0x8000, 0x1000, PAGE_SIZE).unwrap();
        m.load_exclusive_single_thread(0x8020, 4, false).unwrap();
        m.write_u64(0x8028, 3).unwrap();
        assert!(m.exclusive_reservation().is_some());
        m.write_u8(0x8023, 3).unwrap();
        assert_eq!(m.exclusive_reservation(), None);
        m.load_exclusive_single_thread(0x3000, 8, false).unwrap();
        m.set_permissions(0x3000, PAGE_SIZE, READ).unwrap();
        assert_eq!(m.exclusive_reservation(), None);
        m.load_exclusive_single_thread(0x3000, 8, false).unwrap();
        m.map(0x5000, PAGE_SIZE, READ, STATE_NORMAL, "new").unwrap();
        assert_eq!(m.exclusive_reservation(), None);
    }

    #[test]
    fn exclusive_alignment_permission_and_code_protection_faults_change_nothing() {
        let mut m = Memory::new(0x10000);
        m.map(0x1000, PAGE_SIZE, READ | WRITE, STATE_NORMAL, "data")
            .unwrap();
        m.map(0x2000, PAGE_SIZE, WRITE, STATE_NORMAL, "write-only")
            .unwrap();
        m.map(0x3000, PAGE_SIZE, READ | WRITE, STATE_CODE_DATA, "code")
            .unwrap();
        m.protect_translated_code(0x3000, PAGE_SIZE).unwrap();
        m.write_u64(0x1000, 0x55).unwrap();
        for bytes in [2u8, 4, 8] {
            assert!(matches!(
                m.load_exclusive_single_thread(0x1001, bytes, false),
                Err(Trap::MemoryFault {
                    address: 0x1001,
                    ..
                })
            ));
        }
        // LDXR needs READ; an unmapped or write-only address faults and keeps
        // whatever reservation already existed.
        m.load_exclusive_single_thread(0x1000, 8, false).unwrap();
        for address in [0x2000, 0x5000, 0x10000] {
            assert!(matches!(
                m.load_exclusive_single_thread(address, 8, true),
                Err(Trap::MemoryFault { .. })
            ));
            assert_eq!(m.exclusive_reservation(), Some((0x1000, 8)));
        }
        // Unaligned STXR faults even when the monitor would fail.
        assert!(matches!(
            m.store_exclusive_single_thread(0x1004, 8, 1, false),
            Err(Trap::MemoryFault {
                address: 0x1004,
                ..
            })
        ));
        assert_eq!(m.exclusive_reservation(), Some((0x1000, 8)));
        for bytes in [0u8, 3, 16] {
            assert!(matches!(
                m.load_exclusive_single_thread(0x1000, bytes, false),
                Err(Trap::InvalidState { .. })
            ));
            assert!(matches!(
                m.store_exclusive_single_thread(0x1000, bytes, 0, false),
                Err(Trap::InvalidState { .. })
            ));
        }
        // A matching STXR checks WRITE before committing; the fault leaves
        // memory and the reservation intact so nothing is half-done.
        let mut read_only = Memory::new(0x10000);
        read_only
            .map(0x1000, PAGE_SIZE, READ, STATE_NORMAL, "ro")
            .unwrap();
        read_only
            .load_exclusive_single_thread(0x1000, 4, false)
            .unwrap();
        assert!(matches!(
            read_only.store_exclusive_single_thread(0x1000, 4, 9, false),
            Err(Trap::MemoryFault {
                address: 0x1000,
                ..
            })
        ));
        assert_eq!(read_only.exclusive_reservation(), Some((0x1000, 4)));
        assert_eq!(read_only.read_u32(0x1000).unwrap(), 0);
        // A mismatched STXR to read-only memory fails cleanly without faulting.
        read_only.clear_exclusive();
        assert!(
            !read_only
                .store_exclusive_single_thread(0x1000, 4, 9, false)
                .unwrap()
        );
        m.load_exclusive_single_thread(0x3000, 8, false).unwrap();
        assert!(matches!(
            m.store_exclusive_single_thread(0x3000, 8, 1, false),
            Err(Trap::InvalidState { .. })
        ));
        assert_eq!(m.read_u64(0x3000).unwrap(), 0);
        assert_eq!(m.read_u64(0x1000).unwrap(), 0x55);
    }

    #[test]
    fn sparse_memory_zeroes_and_crosses_pages() {
        let mut m = Memory::new(1 << 39);
        m.map(0x1000, 0x10000000, READ | WRITE, STATE_NORMAL, "heap")
            .unwrap();
        assert_eq!(m.resident_bytes(), 0);
        assert_eq!(m.read_u64(0x1fff).unwrap(), 0);
        m.write_u64(0x1fff, 0x123456789abcdef0).unwrap();
        assert_eq!(m.read_u64(0x1fff).unwrap(), 0x123456789abcdef0);
        assert_eq!(m.resident_bytes(), 2 * PAGE_SIZE);
    }

    #[test]
    fn faulting_cross_region_write_is_atomic() {
        let mut m = Memory::new(0x10000);
        m.map(0x1000, 0x1000, READ | WRITE, STATE_NORMAL, "rw")
            .unwrap();
        m.map(0x2000, 0x1000, READ, STATE_CODE, "r").unwrap();
        assert!(m.write_u32(0x1ffe, 0xffffffff).is_err());
        assert_eq!(m.read_u16(0x1ffe).unwrap(), 0);
        assert!(m.map(0x1000, 0x2000, READ, STATE_CODE, "overlap").is_err());
    }

    #[test]
    fn query_reports_whole_gaps_and_wrapping_end() {
        let mut m = Memory::new(0x10000);
        m.map(0x2000, 0x2000, READ, STATE_CODE, "code").unwrap();
        let gap = m.query(0x1000);
        assert_eq!((gap.base, gap.size, gap.state), (0, 0x2000, STATE_FREE));
        let code = m.query(0x2345);
        assert_eq!((code.base, code.size), (0x2000, 0x2000));
        let end = m.query(0x10000);
        assert_eq!(end.base.wrapping_add(end.size), 0);
        assert_eq!(end.state, STATE_INACCESSIBLE);
    }

    #[test]
    fn permission_changes_split_without_losing_pages() {
        let mut m = Memory::new(0x10000);
        m.map(0x1000, 0x3000, READ | WRITE, STATE_CODE_DATA, "data")
            .unwrap();
        m.write_u32(0x2000, 42).unwrap();
        m.set_permissions(0x2000, 0x1000, READ).unwrap();
        assert_eq!(m.read_u32(0x2000).unwrap(), 42);
        assert!(m.write_u32(0x2000, 43).is_err());
        m.write_u32(0x1000, 7).unwrap();
        assert_eq!(m.mappings().count(), 3);
    }

    #[test]
    fn translated_text_cannot_be_modified_after_permission_change() {
        let mut m = Memory::new(0x10000);
        m.map(0x1000, 0x1000, READ | EXECUTE, STATE_CODE, "text")
            .unwrap();
        m.initialize(0x1000, &[1, 2, 3, 4]).unwrap();
        m.protect_translated_code(0x1000, 0x1000).unwrap();
        m.set_permissions(0x1000, 0x1000, READ | WRITE).unwrap();
        assert!(m.write_u32(0x1000, 0).is_err());
    }
}
