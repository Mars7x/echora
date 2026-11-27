//! Guest multisample settings and programmable sample locations.

use super::{Args, Guest, StateSnapshots};
use echora_guest::Trap;
use std::collections::BTreeMap;

const FIELDS: [(&str, u32, u32); 11] = [
    ("MultisampleEnable", 0, 1),
    ("AlphaToCoverageEnable", 1, 1),
    ("AlphaToCoverageDither", 2, 1),
    ("SampleLocationsEnable", 3, 1),
    ("SampleLocationsGridEnable", 4, 1),
    ("CoverageTruncationEnable", 5, 1),
    ("Samples", 8, 31),
    ("CoverageModulationMode", 13, 3),
    ("RasterSamples", 15, 31),
    ("CoverageToColorEnable", 20, 1),
    ("CoverageToColorOutput", 21, 7),
];

/// The integer pair is written only for supported sample counts. The grid
/// depends on the larger of framebuffer and raster samples.
fn grid(word: u32) -> Option<[u32; 2]> {
    match ((word >> 8) & 31).max((word >> 15) & 31) {
        0 | 1 | 16 => Some([1, 1]),
        2 => Some([2, 4]),
        4 => Some([2, 2]),
        8 => Some([1, 2]),
        _ => None,
    }
}

fn location_nibble(bits: u32, vector_lane: bool) -> u8 {
    let rounded = f64::from(f32::from_bits(bits) * 16.0) + 0.5;
    // The guest vector loop narrows an i64 conversion to its low i32.
    // The scalar loop converts straight to i32. These agree on [0, 1].
    let location = if vector_lane {
        (rounded as i64) as i32
    } else {
        rounded as i32
    };
    location.clamp(0, 15) as u8
}

fn sample_locations(core: &mut Guest, set: bool) -> Result<(), Trap> {
    let [state, first, count, points, ..] = core.cpu.regs;
    let count = i64::from(count as i32).max(0) as u64;
    let packed = state
        .wrapping_add(4)
        .wrapping_add_signed(i64::from(first as i32));
    // Overlapping arrays use the guest's forward scalar order. Disjoint
    // groups of four points are converted with the vector instructions.
    let overlap = packed < points.wrapping_add(8 * count) && points < packed.wrapping_add(count);
    let vector_count = if overlap { 0 } else { count & !3 };
    for i in 0..count {
        let at = packed.wrapping_add(i);
        let point = points.wrapping_add(8 * i);
        if set {
            let xy = core.memory.read_u64(point)?;
            let x = location_nibble(xy as u32, i < vector_count);
            let y = location_nibble((xy >> 32) as u32, i < vector_count);
            core.memory.write(at, &[x | (y << 4)])?;
        } else {
            let mut byte = [0];
            core.memory.read(at, &mut byte)?;
            let x = (f32::from(byte[0] & 15) / 16.0).to_bits();
            let y = (f32::from(byte[0] >> 4) / 16.0).to_bits();
            core.memory
                .write_u64(point, u64::from(x) | (u64::from(y) << 32))?;
        }
    }
    Ok(())
}

pub(super) fn call(core: &mut Guest, rest: &str) -> Result<bool, Trap> {
    let [state, value, out_y, ..] = core.cpu.regs;
    match rest {
        "SetDefaults" => {
            let mut bytes = [0; 24];
            bytes[0] = 5;
            bytes[4..20].fill(0x88);
            core.memory.write(state, &bytes)?;
        }
        "GetSampleLocationsGrid" => {
            if let Some([x, y]) = grid(core.memory.read_u32(state)?) {
                core.memory.write_u32(value, x)?;
                core.memory.write_u32(out_y, y)?;
            }
        }
        "SetSampleLocations" | "GetSampleLocations" => {
            sample_locations(core, rest == "SetSampleLocations")?;
        }
        _ => {
            let (set, name) = if let Some(name) = rest.strip_prefix("Set") {
                (true, name)
            } else if let Some(name) = rest.strip_prefix("Get") {
                (false, name)
            } else {
                return Ok(false);
            };
            let Some(&(_, shift, mask)) = FIELDS.iter().find(|&&(field, _, _)| name == field)
            else {
                return Ok(false);
            };
            if set {
                let value = if mask == 1 {
                    u32::from(value as u8 != 0)
                } else {
                    value as u32
                };
                let bits = (core.memory.read_u32(state)? & !(mask << shift)) | (value << shift);
                core.memory.write_u32(state, bits)?;
            } else {
                let bits = if shift < 8 {
                    let mut byte = [0];
                    core.memory.read(state, &mut byte)?;
                    u32::from(byte[0])
                } else {
                    core.memory.read_u32(state)?
                };
                core.cpu.regs[0] = u64::from((bits >> shift) & mask);
            }
        }
    }
    Ok(true)
}

/// Capture the guest values before the application reuses the structure.
/// The bind reads the location array only when programmable locations are enabled.
pub(super) fn snapshot(core: &Guest, address: u64) -> Result<StateSnapshots, Trap> {
    let word = core.memory.read_u32(address)?;
    let mut locations = [0; 16];
    if word & (1 << 3) != 0 {
        core.memory.read(address.wrapping_add(4), &mut locations)?;
    }
    type Snapshot = (u32, [u8; 16], StateSnapshots);
    thread_local! {
        static LAST: std::cell::RefCell<Option<Snapshot>> = const { std::cell::RefCell::new(None) };
    }
    Ok(LAST.with_borrow_mut(|last| {
        if let Some((known, points, snapshot)) = last
            && *known == word
            && *points == locations
        {
            return snapshot.clone();
        }
        let mut properties: BTreeMap<_, _> = FIELDS
            .map(|(name, shift, mask)| {
                (
                    name.to_owned(),
                    Args {
                        x: [u64::from((word >> shift) & mask), 0, 0, 0],
                        ..Args::default()
                    },
                )
            })
            .into();
        properties.insert("@Offset".into(), Args::default());
        if word & (1 << 3) != 0 {
            let packed = u128::from_le_bytes(locations);
            properties.insert(
                "PackedSampleLocations".into(),
                Args {
                    x: [packed as u64, (packed >> 64) as u64, 0, 0],
                    ..Args::default()
                },
            );
        }
        let snapshot = StateSnapshots::from(vec![properties]);
        *last = Some((word, locations, snapshot.clone()));
        snapshot
    }))
}

#[cfg(test)]
mod tests {
    use super::super::GuestMemory;
    use super::*;
    use echora_guest::{
        cpu::Cpu,
        memory::{Memory, READ, STATE_NORMAL, WRITE},
    };

    #[test]
    fn copied_state_and_recorded_locations_keep_their_guest_values() {
        let mut memory = Memory::new(1 << 39);
        memory
            .map(0x1000, 4096, READ | WRITE, STATE_NORMAL, "multisample")
            .unwrap();
        let mut cpu = Cpu::default();
        let mut guest = Guest {
            cpu: &mut cpu,
            memory: GuestMemory::Checked(&mut memory),
            counter: None,
        };
        let (state, copy, point) = (0x10c0, 0x1100, 0x1180);
        guest.cpu.regs[0] = state;
        guest.memory.write(state, &[0xff; 24]).unwrap();
        assert!(call(&mut guest, "SetDefaults").unwrap());
        let mut bytes = [0; 24];
        guest.memory.read(state, &mut bytes).unwrap();
        assert_eq!(&bytes[0..4], &[5, 0, 0, 0]);
        assert_eq!(&bytes[4..20], &[0x88; 16]);
        assert_eq!(&bytes[20..], &[0; 4]);
        guest.memory.write(copy, &bytes).unwrap();
        guest.cpu.regs[..2].copy_from_slice(&[copy, 2]);
        call(&mut guest, "SetSampleLocationsEnable").unwrap();
        guest.memory.write_u32(point, 0.03125f32.to_bits()).unwrap();
        guest.memory.write_u32(point + 4, 1.0f32.to_bits()).unwrap();
        guest.cpu.regs[..4].copy_from_slice(&[copy, 15, 1, point]);
        call(&mut guest, "SetSampleLocations").unwrap();
        let saved = snapshot(&guest, copy).unwrap();
        assert_eq!(saved[0]["PackedSampleLocations"].x[1] >> 56, 0xf1);
        call(&mut guest, "GetSampleLocations").unwrap();
        assert_eq!(
            guest.memory.read_u32(point).unwrap(),
            (1.0f32 / 16.0).to_bits()
        );
        assert_eq!(
            guest.memory.read_u32(point + 4).unwrap(),
            (15.0f32 / 16.0).to_bits()
        );
        guest.memory.write(copy + 19, &[0x32]).unwrap();
        assert_eq!(
            snapshot(&guest, copy).unwrap()[0]["PackedSampleLocations"].x[1] >> 56,
            0x32
        );
        assert_eq!(saved[0]["PackedSampleLocations"].x[1] >> 56, 0xf1);
        // The disabled locations are never accessed, even at the end of the mapping.
        guest.memory.write_u32(0x1ffc, 5).unwrap();
        assert!(!snapshot(&guest, 0x1ffc).unwrap()[0].contains_key("PackedSampleLocations"));
    }
}
