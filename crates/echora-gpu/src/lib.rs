//! Host GPU side of the NVN layer: native shader translation and Vulkan.

// Vulkan and the shader library are FFI: unsafe code is inherent here.
#![allow(
    unsafe_code,
    missing_debug_implementations,
    clippy::all,
    clippy::unwrap_used
)]
pub use ash;
pub mod compute;
pub mod draw;
pub mod loop_guard;
pub mod present;
pub mod renderer;
pub mod shader;
pub mod vulkan;
