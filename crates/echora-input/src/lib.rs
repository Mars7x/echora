//! Host gamepads as a Switch controller state.
//!
//! The interface ([`Pad`], [`gamepads`], [`rumble`]) names nothing
//! platform-specific. Linux reads evdev devices directly; Windows uses
//! gilrs; other platforms report no pads.

mod pad;
pub use pad::{Pad, STICK_MAX, button};

#[cfg(target_os = "linux")]
mod evdev;
#[cfg(target_os = "linux")]
pub use evdev::{Gamepads, gamepads, rumble};

#[cfg(target_os = "windows")]
mod gilrs_pads;
#[cfg(target_os = "windows")]
pub use gilrs_pads::{Gamepads, gamepads, rumble};

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
mod none;
#[cfg(not(any(target_os = "linux", target_os = "windows")))]
pub use none::{Gamepads, gamepads, rumble};
