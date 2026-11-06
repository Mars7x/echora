//! The controller state every backend produces, and the stick handling
//! they share.

/// NpadButton bits of the Switch controller state.
pub mod button {
    pub const A: u64 = 1 << 0;
    pub const B: u64 = 1 << 1;
    pub const X: u64 = 1 << 2;
    pub const Y: u64 = 1 << 3;
    pub const STICK_L: u64 = 1 << 4;
    pub const STICK_R: u64 = 1 << 5;
    pub const L: u64 = 1 << 6;
    pub const R: u64 = 1 << 7;
    pub const ZL: u64 = 1 << 8;
    pub const ZR: u64 = 1 << 9;
    pub const PLUS: u64 = 1 << 10;
    pub const MINUS: u64 = 1 << 11;
    pub const LEFT: u64 = 1 << 12;
    pub const UP: u64 = 1 << 13;
    pub const RIGHT: u64 = 1 << 14;
    pub const DOWN: u64 = 1 << 15;
    /// Left stick directions (left, up, right, down), then the right's.
    pub const STICK_L_LEFT: u64 = 1 << 16;
    pub const STICK_R_LEFT: u64 = 1 << 20;
}

/// Full deflection of a Switch stick axis.
pub const STICK_MAX: i32 = 0x7fff;

/// One controller sample: NpadButton bits and the two sticks (x right,
/// y up, each within a circle of radius [`STICK_MAX`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Pad {
    pub buttons: u64,
    pub left: (i32, i32),
    pub right: (i32, i32),
}

impl Pad {
    /// Several pads as one: buttons held on any, the larger deflection of
    /// each stick.
    pub(crate) fn merge(self, other: Pad) -> Pad {
        let larger = |a: (i32, i32), b: (i32, i32)| {
            let length = |(x, y): (i32, i32)| i64::from(x).pow(2) + i64::from(y).pow(2);
            if length(b) > length(a) { b } else { a }
        };
        Pad {
            buttons: self.buttons | other.buttons,
            left: larger(self.left, other.left),
            right: larger(self.right, other.right),
        }
    }
}

/// Stick deflection (of 1) below which a stick reads as centred.
pub(crate) const DEADZONE: f32 = 0.12;

/// Deflection of an axis that sets its stick-direction button, and of an
/// analog trigger that presses ZL or ZR.
pub(crate) const PRESS: f32 = 0.5;

/// A stick position (each axis -1..=1) with a radial dead zone, scaled to
/// the Switch's range and kept within its circle.
pub(crate) fn stick((x, y): (f32, f32)) -> (i32, i32) {
    let length = (x * x + y * y).sqrt();
    if length <= DEADZONE {
        return (0, 0);
    }
    let scaled = ((length - DEADZONE) / (1.0 - DEADZONE)).min(1.0);
    let factor = scaled / length * STICK_MAX as f32;
    // Truncated toward zero, so a full diagonal stays inside the circle.
    ((x * factor) as i32, (y * factor) as i32)
}

/// The stick-direction buttons (left, up, right, down from `first`) of a
/// stick position.
pub(crate) fn directions((x, y): (i32, i32), first: u64) -> u64 {
    let press = (PRESS * STICK_MAX as f32) as i32;
    let bits = u64::from(x < -press)
        | u64::from(y > press) << 1
        | u64::from(x > press) << 2
        | u64::from(y < -press) << 3;
    bits * first
}
