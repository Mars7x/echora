//! Host backends for guest memory.
//!
//! Every backend provides the same `HostMemory` API: one zero-filled backing
//! store addressed by canonical guest page, a host-side physical view of it,
//! and a guest view in which each mapped guest page has the guest's
//! permissions, so generated code can load and store through it directly.
//!
//! The Linux backend maps a memfd twice. The Windows backend maps
//! pagefile-backed sections into placeholder reservations, which allow
//! page-granular views (Windows 10 1803 or later). Both need 4 KiB host
//! pages, which x86-64 and most ARM64 systems use. Hosts with larger pages
//! (Apple silicon's 16 KiB) need a different strategy and have no backend
//! yet.

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
