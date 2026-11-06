//! Host gamepads as a Switch controller state.
//!
//! The interface ([`Pad`], [`gamepads`], [`rumble`]) names nothing
//! platform-specific. Linux reads evdev devices directly; other platforms
//! have no gamepad backend yet and report no pads.

#[cfg(target_os = "linux")]
mod evdev;
#[cfg(target_os = "linux")]
pub use evdev::*;

#[cfg(not(target_os = "linux"))]
mod none;
#[cfg(not(target_os = "linux"))]
pub use none::*;
