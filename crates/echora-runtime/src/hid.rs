//! Controller state in the hid applet resource's shared memory.
//!
//! Layout from Eden's `hid_core/resources/shared_memory_format.h` (mirror
//! master, fetched 2026-10-31; the Switch 1 format, which the application's
//! shared memory size matches): npad entries of 0x5000 bytes at 0x9A00,
//! indexed Player1..8 = 0..7 and Handheld = 8, each starting with its style
//! set, colors and one 17-entry sample ring per style. The generic System
//! and SystemExt readers use a separate sample ring on each entry. One
//! controller is modeled, connected both as Player 1 (Pro Controller) and
//! as the console's handheld Joy-Cons.

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
const SYSTEM_EXT_LIFO: u64 = FULLKEY_LIFO + 6 * (LIFO_HEADER + LIFO_ENTRIES * LIFO_ENTRY_SIZE);
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
    let tail = sample % LIFO_ENTRIES;
    let count = sample.min(LIFO_ENTRIES - 1);
    let mut lifo_header = [0; LIFO_HEADER as usize];
    lifo_header[8..16].copy_from_slice(&LIFO_ENTRIES.to_le_bytes());
    lifo_header[16..24].copy_from_slice(&tail.to_le_bytes());
    lifo_header[24..32].copy_from_slice(&count.to_le_bytes());
    let mut state = [0; LIFO_ENTRY_SIZE as usize];
    state[..8].copy_from_slice(&(sample << 1).to_le_bytes());
    state[8..16].copy_from_slice(&sample.to_le_bytes());
    state[16..24].copy_from_slice(&buttons.to_le_bytes());
    for (field, value) in state[24..40]
        .chunks_exact_mut(4)
        .zip([left.0, left.1, right.0, right.1])
    {
        field.copy_from_slice(&value.to_le_bytes());
    }
    let batteries = [le32(BATTERY_FULL); 3];
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
        let header = [
            style,
            0,
            0,
            0xff32_3232,
            0xffff_ffff,
            0,
            0xff32_3232,
            0xffff_ffff,
            0xff32_3232,
            0xffff_ffff,
        ]
        .map(le32);
        memory.initialize(entry, header.as_flattened())?;
        memory.initialize(entry + DEVICE_TYPE, &le32(device))?;
        memory.initialize(
            entry + SYSTEM_PROPERTIES,
            &SYSTEM_PROPERTIES_VALUE.to_le_bytes(),
        )?;
        memory.initialize(entry + BATTERY_LEVELS, batteries.as_flattened())?;
        // The ring: entry `sample % 17` is the newest.
        state[40..44].copy_from_slice(&le32(attribute));
        for offset in [lifo, SYSTEM_EXT_LIFO] {
            let ring = entry + offset;
            memory.initialize(ring + LIFO_HEADER + tail * LIFO_ENTRY_SIZE, &state)?;
            memory.initialize(ring, &lifo_header)?;
        }
    }
    // NpadCondition: initialized, horizontal hold, valid.
    let condition = [0, 1, 1, 1].map(le32);
    memory.initialize(base + NPAD_CONDITION, condition.as_flattened())
}

#[allow(clippy::unwrap_used)]
#[cfg(all(test, any(target_os = "linux", windows)))]
mod tests {
    use super::*;
    use crate::memory::{READ, STATE_SHARED};

    #[test]
    fn scripted_movement_drives_the_stick_and_preserves_dpad_input() {
        use crate::window::button::named;
        for (name, stick) in [
            ("STICK_L_UP", (0, STICK_MAX)),
            ("STICK_L_DOWN", (0, -STICK_MAX)),
            ("STICK_L_LEFT", (-STICK_MAX, 0)),
            ("STICK_L_RIGHT", (STICK_MAX, 0)),
            ("UP", (0, 0)),
            ("NONE", (0, 0)),
        ] {
            assert_eq!(digital_stick(named(name).unwrap()), stick);
        }
        assert_eq!(named("NONE"), Some(0));
        assert_ne!(named("UP"), Some(0));
    }

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
        for (controller, style_ring, attribute) in [
            (player1, FULLKEY_LIFO, ATTRIBUTE_FULLKEY),
            (handheld, HANDHELD_LIFO, ATTRIBUTE_HANDHELD),
        ] {
            for offset in [style_ring, 0x1408] {
                let ring = controller + offset;
                let tail = memory.read_u64(ring + 0x10).unwrap();
                assert_eq!(tail, 20 % 17);
                assert_eq!(memory.read_u64(ring + 0x18).unwrap(), 16);
                let entry = ring + LIFO_HEADER + tail * LIFO_ENTRY_SIZE;
                assert_eq!(memory.read_u64(entry).unwrap(), 40);
                assert_eq!(memory.read_u64(entry + 8).unwrap(), 20);
                assert_eq!(memory.read_u64(entry + 0x10).unwrap(), 1 | STICK_RIGHT);
                assert_eq!(memory.read_u32(entry + 0x18).unwrap() as i32, STICK_MAX);
                assert_eq!(memory.read_u32(entry + 0x28).unwrap(), attribute);
            }
        }
        assert_eq!(memory.read_u32(base + NPAD_CONDITION + 4).unwrap(), 1);
    }
}
