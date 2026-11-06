//! The guest runtime: process state, the scheduler, kernel SVCs, IPC and
//! the system services the game reaches.
//!
//! Adapted from pokopia-recomp (same author, GPL-3.0-or-later). Gamepads
//! come from `echora-input`; `audio` plays through PipeWire's `pw-cat` when
//! `ECHORA_AUDIO` is set and opens no device otherwise.

// Ported code keeps its original structure; these lints flag style only.
// Most unwraps convert fixed-length slices to arrays; replacing them is
// tracked as cleanup in docs/limitations.md.
#![allow(missing_debug_implementations, clippy::all, clippy::unwrap_used)]

pub mod audio;
pub mod bootstrap;
pub mod content;
pub mod content_metadata;
pub mod display;
pub mod fs;
pub mod hid;
pub mod hle;
pub mod input;
pub mod ipc;
pub mod loader;
pub mod logging;
pub mod npdm;
pub mod nvn;
pub mod nvn_format;
pub mod romfs;
pub mod runtime;
pub mod sass;
pub mod services;
mod synchronization;
mod threads;
pub mod window;

pub use echora_guest::{cpu, memory};
