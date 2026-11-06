//! The host window as the runtime sees it: a stop request and the keyboard
//! state that becomes controller input. No window backend is ported yet, so
//! `shared` returns `None` and controller input comes only from scripted
//! presses (ECHORA_AUTO_PRESS).

pub static STOP_REQUESTED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Controller buttons in the HID state's bit order.
pub mod button {
    pub const A: u64 = 1 << 0;
    pub const B: u64 = 1 << 1;
    pub const X: u64 = 1 << 2;
    pub const Y: u64 = 1 << 3;
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
    pub const STICK_L_LEFT: u64 = 1 << 16;
    pub const STICK_L_UP: u64 = 1 << 17;
    pub const STICK_L_RIGHT: u64 = 1 << 18;
    pub const STICK_L_DOWN: u64 = 1 << 19;

    /// A button by name (A, B, UP, PLUS, ...), for scripted input.
    pub fn named(name: &str) -> Option<u64> {
        Some(match name.trim().to_ascii_uppercase().as_str() {
            "A" => A,
            "B" => B,
            "X" => X,
            "Y" => Y,
            "L" => L,
            "R" => R,
            "ZL" => ZL,
            "ZR" => ZR,
            "PLUS" => PLUS,
            "MINUS" => MINUS,
            "LEFT" => LEFT,
            "UP" => UP,
            "RIGHT" => RIGHT,
            "DOWN" => DOWN,
            _ => return None,
        })
    }
}

/// State shared between the host window and the runtime.
#[derive(Debug, Default)]
pub struct Shared {
    /// Held buttons (`button` bits).
    pub buttons: std::sync::atomic::AtomicU64,
}

/// The host window, if one is open. None until a window backend exists.
pub fn shared() -> Option<&'static Shared> {
    None
}
