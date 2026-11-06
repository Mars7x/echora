//! The guest runtime: process state, the scheduler, kernel SVCs, IPC and
//! the system services the game reaches.
//!
//! Adapted from pokopia-recomp (same author, GPL-3.0-or-later). Gamepads
//! come from `echora-input`; audio from `echora-audio` (with `ECHORA_AUDIO`).

// Ported code keeps its original structure; these lints flag style only.
// Most unwraps convert fixed-length slices to arrays; replacing them is
// tracked as cleanup in docs/limitations.md.
#![allow(missing_debug_implementations, clippy::all)]

pub mod audio;
pub mod audio_renderer;
pub mod av1;
pub mod bootstrap;
mod bytes;
pub mod content;
pub mod content_metadata;
pub mod display;
mod fault;
pub mod fs;
pub mod hid;
pub mod hle;
pub mod input;
pub mod ipc;
pub mod loader;
pub mod logging;
pub mod npdm;
pub mod nvdrv;
#[allow(clippy::unwrap_used)] // ported NVN code: unwraps follow presence checks
pub mod nvn;
#[allow(clippy::unwrap_used)] // ported NVN code: unwraps follow presence checks
pub mod nvn_format;
pub mod romfs;
pub mod runtime;
#[allow(clippy::unwrap_used)] // ported NVN code: unwraps follow presence checks
pub mod sass;
pub mod services;
pub mod settings;
mod synchronization;
mod threads;
#[allow(clippy::unwrap_used)] // ported NVN code: unwraps follow presence checks
pub mod window;

pub use echora_guest::{cpu, memory};
