//! Primitive restart translation for indexed strips and fans.
use super::{Indices, vk};
use vk::PrimitiveTopology as Topology;

/// A marker wider than the guest index cannot match a zero-extended index.
pub(super) fn restart_marker(topology: Topology, register: Option<u32>, size: u64) -> Option<u32> {
    if !matches!(
        topology,
        Topology::LINE_STRIP | Topology::TRIANGLE_STRIP | Topology::TRIANGLE_FAN
    ) {
        return None;
    }
    register.filter(|&marker| marker <= index_max(size))
}

pub(super) fn index_max(size: u64) -> u32 {
    match size {
        1 => u32::from(u8::MAX),
        2 => u32::from(u16::MAX),
        4 => u32::MAX,
        _ => unreachable!("validated guest index type"),
    }
}

/// Called after a custom marker's input has been synchronized and copied from guest memory.
/// On return a true enable always means the supplied Vulkan index type's fixed restart.
pub(super) fn prepare_restart(
    indices: &mut Indices,
    marker: u32,
    topology: &mut Topology,
    count: &mut u32,
) -> Result<bool, String> {
    match indices {
        Indices::Guest { kind, .. } | Indices::Shared { kind, .. } => {
            let expected = match *kind {
                vk::IndexType::UINT8_KHR => u32::from(u8::MAX),
                vk::IndexType::UINT16 => u32::from(u16::MAX),
                vk::IndexType::UINT32 => u32::MAX,
                _ => return Err("invalid Vulkan index type".into()),
            };
            if marker != expected {
                return Err("custom restart indices require synchronized conversion".into());
            }
            Ok(true)
        }
        Indices::None => Ok(false),
        Indices::U16(bytes) => {
            if marker == u32::from(u16::MAX) {
                return Ok(true);
            }
            let words = bytes.as_chunks_mut::<2>().0;
            let restart = (marker as u16).to_le_bytes();
            if !words.contains(&restart) {
                return Ok(false);
            }
            if words.contains(&u16::MAX.to_le_bytes()) {
                let widened = words
                    .iter()
                    .flat_map(|&v| {
                        let index = if v == restart {
                            u32::MAX
                        } else {
                            u32::from(u16::from_le_bytes(v))
                        };
                        index.to_le_bytes()
                    })
                    .collect();
                *indices = Indices::U32(widened);
            } else {
                for v in words.iter_mut().filter(|v| **v == restart) {
                    *v = u16::MAX.to_le_bytes()
                }
            }
            Ok(true)
        }
        Indices::U32(bytes) => {
            if marker == u32::MAX {
                return Ok(true);
            }
            let words = bytes.as_chunks_mut::<4>().0;
            let restart = marker.to_le_bytes();
            if !words.contains(&restart) {
                return Ok(false);
            }
            if !words.contains(&u32::MAX.to_le_bytes()) {
                for v in words.iter_mut().filter(|v| **v == restart) {
                    *v = u32::MAX.to_le_bytes()
                }
                return Ok(true);
            }
            // No wider Vulkan index exists. Keep this as one draw so primitive IDs
            // continue across restarts. Preserve the topology's first provoking
            // vertex, winding, and all complete primitives, including degenerates.
            let (list, primitive_width) = if *topology == Topology::LINE_STRIP {
                (Topology::LINE_LIST, 2usize)
            } else {
                (Topology::TRIANGLE_LIST, 3)
            };
            let runs = || words.split(|v| *v == restart);
            let primitives: usize = runs()
                .map(|run| run.len().saturating_sub(primitive_width - 1))
                .sum();
            let output_count = primitives
                .checked_mul(primitive_width)
                .and_then(|n| u32::try_from(n).ok())
                .ok_or("translated restart draw is too large")?;
            let mut output = Vec::new();
            output
                .try_reserve_exact(
                    usize::try_from(output_count)
                        .ok()
                        .and_then(|n| n.checked_mul(4))
                        .ok_or("translated restart buffer is too large")?,
                )
                .map_err(|e| e.to_string())?;
            for run in runs() {
                for (i, window) in run.windows(primitive_width).enumerate() {
                    match *topology {
                        Topology::LINE_STRIP => output.extend_from_slice(window.as_flattened()),
                        Topology::TRIANGLE_FAN => {
                            output.extend_from_slice(window[1..].as_flattened());
                            output.extend_from_slice(&run[0]);
                        }
                        Topology::TRIANGLE_STRIP if i % 2 == 1 => {
                            for j in [0, 2, 1] {
                                output.extend_from_slice(&window[j])
                            }
                        }
                        Topology::TRIANGLE_STRIP => output.extend_from_slice(window.as_flattened()),
                        _ => unreachable!("restart applies only to strips and fans"),
                    }
                }
            }
            *bytes = output;
            *topology = list;
            *count = output_count;
            Ok(false)
        }
    }
}

/// Inclusive range of vertices referenced by copied indices, excluding fixed restart tokens.
pub(super) fn bounds(indices: &Indices, restart: bool) -> Option<(u32, u32)> {
    fn range(words: impl Iterator<Item = u32>, max: u32, restart: bool) -> Option<(u32, u32)> {
        let mut words = words.filter(|&x| !restart || x != max);
        let first = words.next()?;
        Some(words.fold((first, first), |(lo, hi), i| (lo.min(i), hi.max(i))))
    }
    match indices {
        Indices::U16(bytes) => range(
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&v| u32::from(u16::from_le_bytes(v))),
            u32::from(u16::MAX),
            restart,
        ),
        Indices::U32(bytes) => range(
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&v| u32::from_le_bytes(v)),
            u32::MAX,
            restart,
        ),
        _ => None,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn u32_indices(values: &[u32]) -> Indices {
        Indices::U32(values.iter().flat_map(|v| v.to_le_bytes()).collect())
    }
    fn u16_indices(values: &[u16]) -> Indices {
        Indices::U16(values.iter().flat_map(|v| v.to_le_bytes()).collect())
    }
    fn unpack(indices: &Indices) -> Vec<u32> {
        match indices {
            Indices::U32(v) => v
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&v| u32::from_le_bytes(v))
                .collect(),
            Indices::U16(v) => v
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&v| u32::from(u16::from_le_bytes(v)))
                .collect(),
            _ => panic!("expected owned indices"),
        }
    }

    #[test]
    fn narrow_restart_preserves_a_real_all_ones_index_and_bounds() {
        let mut indices = u16_indices(&[4, 7, 65535, 7, 6]);
        let mut topology = Topology::TRIANGLE_STRIP;
        assert!(prepare_restart(&mut indices, 7, &mut topology, &mut 5).unwrap());
        assert_eq!(unpack(&indices), [4, u32::MAX, 65535, u32::MAX, 6]);
        assert_eq!(bounds(&indices, true), Some((4, 65535)));
        assert_eq!(bounds(&u16_indices(&[65535, 65535]), true), None);
        // 8-bit guest indices already widened to 16 bits on devices without uint8.
        let mut widened = u16_indices(&[5, 255, 6]);
        assert!(prepare_restart(&mut widened, 255, &mut topology, &mut 3).unwrap());
        assert_eq!(unpack(&widened), [5, 65535, 6]);
        assert_eq!(bounds(&widened, true), Some((5, 6)));
        assert_eq!(restart_marker(topology, Some(0x100ff), 1), None);
        assert_eq!(restart_marker(topology, Some(0x10007), 2), None);
        assert_eq!(restart_marker(Topology::TRIANGLE_LIST, Some(7), 2), None);
    }

    #[test]
    fn resident_restart_requires_the_guest_and_host_markers_to_match() {
        let mut sources = [
            Indices::Guest {
                address: 0x4000,
                bytes: 12,
                kind: vk::IndexType::UINT16,
            },
            Indices::Shared {
                buffer: vk::Buffer::null(),
                offset: 0,
                kind: vk::IndexType::UINT16,
            },
        ];
        for source in &mut sources {
            let mut topology = Topology::TRIANGLE_STRIP;
            assert!(prepare_restart(source, 65535, &mut topology, &mut 6).unwrap());
            assert!(prepare_restart(source, 7, &mut topology, &mut 6).is_err());
        }
    }

    #[test]
    fn u32_collision_preserves_primitive_order_and_restarts_each_run() {
        let m = u32::MAX;
        let source = [7, 9, 7, 0, 1, m, 3, 3, 7, 7, 4, 5, 6, 7, 1, 7];
        for (primitive, expected_topology, expected) in [
            (
                Topology::TRIANGLE_STRIP,
                Topology::TRIANGLE_LIST,
                vec![0, 1, m, 1, 3, m, m, 3, 3, 4, 5, 6],
            ),
            (
                Topology::TRIANGLE_FAN,
                Topology::TRIANGLE_LIST,
                vec![1, m, 0, m, 3, 0, 3, 3, 0, 5, 6, 4],
            ),
            (
                Topology::LINE_STRIP,
                Topology::LINE_LIST,
                vec![0, 1, 1, m, m, 3, 3, 3, 4, 5, 5, 6],
            ),
        ] {
            let mut topology = primitive;
            let mut count = source.len() as u32;
            let mut indices = u32_indices(&source);
            assert!(!prepare_restart(&mut indices, 7, &mut topology, &mut count).unwrap());
            assert_eq!(topology, expected_topology);
            assert_eq!(count, expected.len() as u32);
            assert_eq!(unpack(&indices), expected);
            assert_eq!(bounds(&indices, false), Some((0, m)));
        }
    }

    #[test]
    fn u32_custom_marker_without_collision_keeps_the_input_topology() {
        let mut topology = Topology::TRIANGLE_STRIP;
        let mut count = 3;
        let mut with = u32_indices(&[3, 7, 3]);
        assert!(prepare_restart(&mut with, 7, &mut topology, &mut count).unwrap());
        assert_eq!(unpack(&with), [3, u32::MAX, 3]);
        assert_eq!(bounds(&with, true), Some((3, 3)));
        let mut without = u32_indices(&[0, 1, u32::MAX]);
        assert!(!prepare_restart(&mut without, 7, &mut topology, &mut count).unwrap());
        assert_eq!(unpack(&without), [0, 1, u32::MAX]);
        assert_eq!(bounds(&without, false), Some((0, u32::MAX)));
        assert_eq!(topology, Topology::TRIANGLE_STRIP);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod draw_tests {
    use super::super::Backend;
    use crate::nvn::{Args, Command, Guest, GuestMemory, Nvn, StateSnapshots};
    use echora_gpu::ash::vk;
    use echora_guest::{
        cpu::Cpu,
        memory::{Memory, READ, STATE_NORMAL, WRITE},
    };
    use std::collections::BTreeMap;

    /// Draw across the center with three real indices, the last being all ones.
    /// Moving the restart into that triangle must clear its coverage.
    #[test]
    fn guest_custom_restart_preserves_an_all_ones_vertex() {
        let Some(backend) = Backend::create(None).expect("initialize the requested Vulkan device")
        else {
            return;
        };
        let max_index = backend.renderer.gpu.max_index_value;
        let mut nvn = Nvn {
            gpu: Some(backend),
            gpu_checked: true,
            ..Default::default()
        };
        let base = 0x10_0000;
        let fragment = base + 0x400;
        let index_buffer = base + 0x800;
        let args = |x| Args {
            x,
            ..Default::default()
        };
        nvn.initialize(
            1,
            "MemoryPool",
            BTreeMap::from([("Storage".into(), args([base, 4096, 0, 0]))]),
        )
        .unwrap();
        nvn.objects
            .get_mut(&1)
            .unwrap()
            .state
            .insert("GpuAddress".into(), args([base, 0, 0, 0]));
        let mut memory = Memory::new(1 << 39);
        memory
            .map(
                base,
                4096,
                READ | WRITE,
                STATE_NORMAL,
                "primitive restart fixture",
            )
            .unwrap();
        let word = |fields: &[(u32, u128)]| {
            fields
                .iter()
                .fold(7 << 12, |word, &(shift, v)| word | (v << shift))
        };
        let mov = |r, v| word(&[(0, 0x802), (16, r), (32, v), (72, 15)]);
        let isequal = |v| word(&[(0, 0x80c), (24, 4), (32, v), (76, 2), (84, 7), (87, 7)]);
        let exit = word(&[(0, 0x94d), (87, 7)]);
        let shader = |tag: u32, words: &[u128]| {
            let mut code = vec![0; 128];
            code[..4].copy_from_slice(&tag.to_le_bytes());
            code.extend(words.iter().flat_map(|w| w.to_le_bytes()));
            code
        };
        let as_float = |v: f32| u128::from(v.to_bits());
        // Vertex zero starts at (-1,-1,0,1). Vertex 1 moves right, any index other
        // than 0 or 1 moves up. The other indices preceding the restart form
        // an incomplete strip.
        let pred_p0 = !(7u128 << 12);
        let vertex = shader(
            1 | (1 << 10),
            &[
                word(&[(0, 0x321), (16, 4), (24, 255), (32, 255), (40, 0x2fc)]),
                mov(0, as_float(-1.0)),
                mov(1, as_float(3.0)),
                mov(2, 0),
                mov(3, as_float(1.0)),
                isequal(0),
                mov(1, as_float(-1.0)) & pred_p0,
                isequal(1),
                mov(0, as_float(3.0)) & pred_p0,
                mov(1, as_float(-1.0)) & pred_p0,
                word(&[
                    (0, 0x322),
                    (24, 255),
                    (32, 0),
                    (40, 0x70),
                    (64, 255),
                    (74, 3),
                ]),
                exit,
            ],
        );
        let mut pixel = shader(2 | (5 << 10), &[mov(0, 123), exit]);
        pixel[576 / 8] = 1;
        memory.write(base, &vertex).unwrap();
        memory.write(fragment, &pixel).unwrap();
        nvn.initialize(2, "Program", BTreeMap::new()).unwrap();
        nvn.objects.get_mut(&2).unwrap().state.extend([
            ("Stage0".into(), args([base, 0, 0, 0])),
            ("Stage1".into(), args([fragment, 0, 0, 0])),
        ]);
        let output = 8;
        nvn.initialize(
            output,
            "Texture",
            BTreeMap::from([
                ("Width".into(), args([1, 0, 0, 0])),
                ("Height".into(), args([1, 0, 0, 0])),
                ("Format".into(), args([11, 0, 0, 0])),
            ]),
        )
        .unwrap();
        nvn.gpu
            .as_mut()
            .unwrap()
            .renderer
            .target(output, 1, 1, vk::Format::R32_UINT)
            .unwrap();
        let mut cpu = Cpu::default();
        let mut guest = Guest {
            cpu: &mut cpu,
            memory: GuestMemory::Checked(&mut memory),
            counter: None,
        };
        let cmd = |name, x| Command {
            name,
            x,
            v: [0; 8],
            captured: Vec::new(),
            states: StateSnapshots::default(),
        };
        nvn.execute(
            &cmd("nvnCommandBufferBindProgram", [0, 2, 3, 0, 0, 0, 0, 0]),
            &mut guest,
        )
        .unwrap();
        let mut targets = cmd("nvnCommandBufferSetRenderTargets", [0, 1, 0, 0, 0, 0, 0, 0]);
        targets.captured.push(output);
        nvn.execute(&targets, &mut guest).unwrap();
        for (index_type, bytes) in [
            (0, vec![4, 7, 0, 1, 255, 7]),
            (
                1,
                [4u16, 7, 0, 1, 65535, 7]
                    .into_iter()
                    .flat_map(u16::to_le_bytes)
                    .collect(),
            ),
            (
                2,
                [4u32, 7, 0, 1, u32::MAX, 7]
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect(),
            ),
        ] {
            // A host may limit regular 32-bit vertex indices below u32::MAX.
            if index_type == 2 && max_index != u32::MAX {
                continue;
            }
            // Leave stale zeros in guest memory. The transfer and its staged readback
            // are pending when the draw needs to translate its custom restart marker.
            guest
                .memory
                .write(index_buffer, &vec![0; bytes.len()])
                .unwrap();
            let length = bytes.len() as u64;
            let backend = nvn.gpu.as_mut().unwrap();
            backend.renderer.resident_buffer(index_buffer, 64).unwrap();
            backend
                .renderer
                .upload_resident(index_buffer, 0, &bytes)
                .unwrap();
            let token = backend
                .renderer
                .read_resident(index_buffer, 0, length)
                .unwrap();
            backend.pending_storage.push((
                token,
                index_buffer..index_buffer + length,
                0,
                0,
                index_buffer,
            ));

            for (marker, expected) in [(7, 123u32), (1, 0)] {
                nvn.execute(
                    &cmd(
                        "nvnCommandBufferSetPrimitiveRestart",
                        [0, 1, marker, 0, 0, 0, 0, 0],
                    ),
                    &mut guest,
                )
                .unwrap();
                nvn.gpu
                    .as_mut()
                    .unwrap()
                    .renderer
                    .clear_color(output, None, vk::ClearColorValue { uint32: [0; 4] })
                    .unwrap();
                nvn.gpu_draw(
                    &mut guest,
                    &cmd(
                        "nvnCommandBufferDrawElements",
                        [0, 5, index_type, 6, index_buffer, 0, 0, 0],
                    ),
                )
                .unwrap();
                assert_eq!(
                    nvn.gpu
                        .as_mut()
                        .unwrap()
                        .renderer
                        .read_target_bytes(output, 4)
                        .unwrap(),
                    expected.to_le_bytes(),
                    "index type {index_type} marker {marker}"
                );
            }
        }
    }
}
