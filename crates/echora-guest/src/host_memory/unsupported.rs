//! Placeholder for hosts without a guest-memory backend. Creating guest
//! memory fails immediately with an explanation; no other method can be
//! reached because the type cannot be constructed.

use std::convert::Infallible;

pub(crate) struct HostMemory {
    never: Infallible,
}

impl HostMemory {
    pub(crate) fn new(_size: u64) -> Self {
        panic!(
            "guest memory has no backend for this host yet: Echora's guest view needs \
             4 KiB host pages and page-granular file views (implemented for Linux and Windows only)"
        )
    }

    pub(crate) fn load<T: Copy>(&self, _address: u64) -> T {
        match self.never {}
    }

    pub(crate) fn store<T: Copy>(&mut self, _address: u64, _value: T) {
        match self.never {}
    }

    pub(crate) fn guest_access(&self) -> super::GuestAccess {
        match self.never {}
    }

    pub(crate) fn page(&self, _page: u64) -> &[u8; crate::memory::PAGE_SIZE as usize] {
        match self.never {}
    }

    pub(crate) fn page_mut(&mut self, _page: u64) -> &mut [u8; crate::memory::PAGE_SIZE as usize] {
        match self.never {}
    }

    pub(crate) fn page_address(&self, _page: u64) -> usize {
        match self.never {}
    }

    pub(crate) fn release(&mut self, _start: u64, _end: u64) {
        match self.never {}
    }

    pub(crate) fn map_guest(
        &mut self,
        _address: u64,
        _length: u64,
        _backing: Option<u64>,
        _access: u8,
    ) {
        match self.never {}
    }
}
