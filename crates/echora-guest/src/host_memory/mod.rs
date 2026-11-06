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
//!
//! macOS uses the checked backend: Apple silicon's 16 KiB pages cannot carry
//! per-guest-page permissions, so each access looks its guest page up in a
//! translation table. `--cfg echora_checked_memory` selects that backend on
//! Linux, for testing.

#[cfg(all(target_os = "linux", not(echora_checked_memory)))]
mod linux;
#[cfg(all(target_os = "linux", not(echora_checked_memory)))]
pub(crate) use linux::HostMemory;

#[cfg(windows)]
mod windows;
#[cfg(windows)]
pub(crate) use windows::HostMemory;

#[cfg(any(target_os = "macos", all(target_os = "linux", echora_checked_memory)))]
mod checked;
#[cfg(any(target_os = "macos", all(target_os = "linux", echora_checked_memory)))]
pub(crate) use checked::{GuestAccess, HostMemory};

#[cfg(not(any(target_os = "linux", windows, target_os = "macos")))]
mod unsupported;
#[cfg(not(any(target_os = "linux", windows, target_os = "macos")))]
pub(crate) use unsupported::HostMemory;

#[cfg(not(any(target_os = "macos", all(target_os = "linux", echora_checked_memory))))]
mod direct;
#[cfg(not(any(target_os = "macos", all(target_os = "linux", echora_checked_memory))))]
pub(crate) use direct::GuestAccess;

#[cold]
#[inline(never)]
pub(crate) fn outside(address: u64) -> ! {
    panic!("guest memory access at {address:#x} lies outside the guest address space")
}
