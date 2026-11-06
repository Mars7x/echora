//! Host backends for guest memory.
//!
//! Every backend provides the same `HostMemory` API: one zero-filled backing
//! store addressed by canonical guest page, a host-side physical view of it,
//! and a guest view in which each mapped guest page has the guest's
//! permissions, so generated code can load and store through it directly.
//!
//! The Linux backend maps a memfd twice. It needs 4 KiB host pages, which
//! x86-64 and most ARM64 Linux systems use. Hosts with larger pages
//! (Apple silicon's 16 KiB) or coarser view granularity (Windows maps file
//! views at 64 KiB) need a different strategy and have no backend yet.

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
pub(crate) use linux::HostMemory;

#[cfg(not(target_os = "linux"))]
mod unsupported;
#[cfg(not(target_os = "linux"))]
pub(crate) use unsupported::HostMemory;
