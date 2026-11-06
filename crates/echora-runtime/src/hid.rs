//! Controller state in the hid applet resource's shared memory.
//!
//! Layout from Eden's `hid_core/resources/shared_memory_format.h` (mirror
//! master, fetched 2026-10-31; the Switch 1 format, which the application's
//! shared memory size matches): npad entries of 0x5000 bytes at 0x9A00,
//! indexed Player1..8 = 0..7 and Handheld = 8, each starting with its style
//! set, colors and one 17-entry sample ring per style. One controller is
//! modeled, connected both as Player 1 (Pro Controller) and as the console's
//! handheld Joy-Cons, so either style the application asks for has input.

use crate::memory::Memory;
use crate::runtime::Trap;

const NPAD_BASE: u64 = 0x9A00;
const NPAD_ENTRY_SIZE: u64 = 0x5000;
const NPAD_CONDITION: u64 = 0x3E200;
const LIFO_ENTRIES: u64 = 17;
/// Lifo header (timestamp, total count, tail, count) then entries of a
/// sampling number and a 0x28-byte NPadGenericState.
const LIFO_HEADER: u64 = 0x20;
const LIFO_ENTRY_SIZE: u64 = 0x30;
const FULLKEY_LIFO: u64 = 0x28;
const HANDHELD_LIFO: u64 = 0x378;
const DEVICE_TYPE: u64 = 0x4188;
const SYSTEM_PROPERTIES: u64 = 0x4190;
const BATTERY_LEVELS: u64 = 0x419C;

const STYLE_FULLKEY: u32 = 1 << 0;
const STYLE_HANDHELD: u32 = 1 << 1;
/// DeviceType bits: fullkey, handheld left and right.
const DEVICE_FULLKEY: u32 = 1 << 0;
const DEVICE_HANDHELD: u32 = 1 << 2 | 1 << 3;
/// Powered Joy-Cons, plus, minus and directional buttons usable.
const SYSTEM_PROPERTIES_VALUE: u64 = 1 << 3 | 1 << 4 | 1 << 5 | 1 << 13 | 1 << 14 | 1 << 15;
const BATTERY_FULL: u32 = 4;
/// NpadAttribute: connected (fullkey); connected and wired with both
/// Joy-Cons attached (handheld).
const ATTRIBUTE_FULLKEY: u32 = 1;
const ATTRIBUTE_HANDHELD: u32 = 0x3f;

/// NpadButton bits of the stick directions, mirrored into the stick axes.
const STICK_LEFT: u64 = 1 << 16;
const STICK_UP: u64 = 1 << 17;
const STICK_RIGHT: u64 = 1 << 18;
const STICK_DOWN: u64 = 1 << 19;
const STICK_MAX: i32 = 0x7fff;

fn le32(value: u32) -> [u8; 4] {
    value.to_le_bytes()
}

/// The left stick of digital input: full deflection toward each held
/// stick-direction button.
pub fn digital_stick(buttons: u64) -> (i32, i32) {
    let axis = |negative: u64, positive: u64| {
        (buttons & positive != 0) as i32 * STICK_MAX - (buttons & negative != 0) as i32 * STICK_MAX
    };
    (axis(STICK_LEFT, STICK_RIGHT), axis(STICK_DOWN, STICK_UP))
}

/// Publish sample `sample` (counting from 1) with `buttons` (NpadButton
/// bits) and the `left` and `right` stick positions to the shared memory
/// mapped at `base`.
pub fn write_controller(
    memory: &mut Memory,
    base: u64,
    sample: u64,
    buttons: u64,
    left: (i32, i32),
    right: (i32, i32),
) -> Result<(), Trap> {
    for (index, style, lifo, device, attribute) in [
        (
            0,
            STYLE_FULLKEY,
            FULLKEY_LIFO,
            DEVICE_FULLKEY,
            ATTRIBUTE_FULLKEY,
        ),
        (
            8,
            STYLE_HANDHELD,
            HANDHELD_LIFO,
            DEVICE_HANDHELD,
            ATTRIBUTE_HANDHELD,
        ),
    ] {
        let entry = base + NPAD_BASE + index * NPAD_ENTRY_SIZE;
        // Style set, dual assignment, then full-key and Joy-Con colors with
        // attribute Ok (0) and opaque body/button colors.
        let mut header = Vec::with_capacity(0x28);
        header.extend_from_slice(&le32(style));
        header.extend_from_slice(&le32(0));
        header.extend_from_slice(&le32(0));
        header.extend_from_slice(&le32(0xff32_3232));
        header.extend_from_slice(&le32(0xffff_ffff));
        header.extend_from_slice(&le32(0));
        for color in [0xff32_3232u32, 0xffff_ffff, 0xff32_3232, 0xffff_ffff] {
            header.extend_from_slice(&le32(color));
        }
        memory.initialize(entry, &header)?;
        memory.initialize(entry + DEVICE_TYPE, &le32(device))?;
        memory.initialize(
            entry + SYSTEM_PROPERTIES,
            &SYSTEM_PROPERTIES_VALUE.to_le_bytes(),
        )?;
        for battery in 0..3 {
            memory.initialize(entry + BATTERY_LEVELS + battery * 4, &le32(BATTERY_FULL))?;
        }
        // The ring: entry `sample % 17` is the newest.
        let ring = entry + lifo;
        let tail = sample % LIFO_ENTRIES;
        let count = sample.min(LIFO_ENTRIES - 1);
        let mut lifo_header = Vec::with_capacity(0x20);
        lifo_header.extend_from_slice(&0u64.to_le_bytes());
        lifo_header.extend_from_slice(&LIFO_ENTRIES.to_le_bytes());
        lifo_header.extend_from_slice(&tail.to_le_bytes());
        lifo_header.extend_from_slice(&count.to_le_bytes());
        let mut state = Vec::with_capacity(LIFO_ENTRY_SIZE as usize);
        state.extend_from_slice(&(sample << 1).to_le_bytes());
        state.extend_from_slice(&sample.to_le_bytes());
        state.extend_from_slice(&buttons.to_le_bytes());
        for value in [left.0, left.1, right.0, right.1] {
            state.extend_from_slice(&value.to_le_bytes());
        }
        state.extend_from_slice(&le32(attribute));
        state.extend_from_slice(&le32(0));
        memory.initialize(ring + LIFO_HEADER + tail * LIFO_ENTRY_SIZE, &state)?;
        memory.initialize(ring, &lifo_header)?;
    }
    // NpadCondition: initialized, horizontal hold, valid.
    let mut condition = Vec::with_capacity(16);
    for value in [0u32, 1, 1, 1] {
        condition.extend_from_slice(&le32(value));
    }
    memory.initialize(base + NPAD_CONDITION, &condition)
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use crate::memory::{READ, STATE_SHARED};

    #[test]
    fn samples_land_in_the_player1_and_handheld_rings() {
        let base = 0x1000_0000;
        let mut memory = Memory::new(1 << 39);
        memory
            .map(base, 0x40000, READ, STATE_SHARED, "hid")
            .unwrap();
        for sample in 1..=20 {
            write_controller(
                &mut memory,
                base,
                sample,
                1 | STICK_RIGHT,
                digital_stick(1 | STICK_RIGHT),
                (0, 0),
            )
            .unwrap();
        }
        let player1 = base + NPAD_BASE;
        let handheld = base + NPAD_BASE + 8 * NPAD_ENTRY_SIZE;
        assert_eq!(memory.read_u32(player1).unwrap(), STYLE_FULLKEY);
        assert_eq!(memory.read_u32(handheld).unwrap(), STYLE_HANDHELD);
        let ring = handheld + HANDHELD_LIFO;
        let tail = memory.read_u64(ring + 0x10).unwrap();
        assert_eq!(tail, 20 % 17);
        assert_eq!(memory.read_u64(ring + 0x18).unwrap(), 16);
        let entry = ring + LIFO_HEADER + tail * LIFO_ENTRY_SIZE;
        assert_eq!(memory.read_u64(entry).unwrap(), 40);
        assert_eq!(memory.read_u64(entry + 8).unwrap(), 20);
        assert_eq!(memory.read_u64(entry + 0x10).unwrap(), 1 | STICK_RIGHT);
        assert_eq!(memory.read_u32(entry + 0x18).unwrap() as i32, STICK_MAX);
        assert_eq!(memory.read_u32(entry + 0x28).unwrap(), ATTRIBUTE_HANDHELD);
        assert_eq!(memory.read_u32(base + NPAD_CONDITION + 4).unwrap(), 1);
    }
}
