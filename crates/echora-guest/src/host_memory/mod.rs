//! Host backends for guest memory.
//!
//! Every backend provides the same `HostMemory` API: one zero-filled backing
//! store addressed by canonical guest page, and a host-side physical view of
//! it. Each also provides a `GuestAccess`, generated code's path into guest
//! memory with the guest's permissions.
//!
//! Hosts with 4 KiB pages map a guest view in which each mapped guest page
//! has the guest's permissions as host page protections, so generated code
//! loads and stores through it directly (`direct`). The Linux backend maps a
//! memfd twice; the Windows backend maps pagefile-backed sections into
//! placeholder reservations, which allow page-granular views (Windows 10
//! 1803 or later).

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
pub(crate) use linux::HostMemory;

#[cfg(windows)]
mod windows;
#[cfg(windows)]
pub(crate) use windows::HostMemory;

#[cfg(not(any(target_os = "linux", windows)))]
mod unsupported;
#[cfg(not(any(target_os = "linux", windows)))]
pub(crate) use unsupported::HostMemory;

mod direct;
pub(crate) use direct::GuestAccess;

#[cold]
#[inline(never)]
pub(crate) fn outside(address: u64) -> ! {
    panic!("guest memory access at {address:#x} lies outside the guest address space")
}
