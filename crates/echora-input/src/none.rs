//! Platforms without a gamepad backend: there are no pads, and rumble
//! requests have no effect.

use crate::pad::Pad;

#[derive(Debug, Default)]
pub struct Gamepads;

impl Gamepads {
    pub fn state(&self) -> Pad {
        Pad::default()
    }

    /// Whether any gamepad is connected (never, without a backend).
    pub fn connected(&self) -> bool {
        false
    }
}

pub fn gamepads() -> &'static Gamepads {
    static GAMEPADS: Gamepads = Gamepads;
    &GAMEPADS
}

/// A vibration request for `device`; without a backend it has no effect.
pub fn rumble(_device: u32, _low: f32, _high: f32) {}
