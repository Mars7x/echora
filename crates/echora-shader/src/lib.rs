//! Translation of the Switch 2 GPU's shader programs (NVIDIA Ampere, SM 8.x
//! SASS) to SPIR-V, ahead of time and without runtime state. See
//! `docs/shader-translator.md` for the design and its status.
//!
//! So far: [`decode`], 128-bit instruction words into typed instructions,
//! and [`compute`], simple compute programs to SPIR-V. Anything
//! [`translate`] reports as `Unsupported` goes to the plugin translator.

pub mod compute;
pub mod decode;
pub mod header;
pub mod spirv;

pub use decode::{Instruction, Op, decode, decode_program};
use echora_shader_abi::{Parameters, Stage, Translate, Translated, Unsupported};

/// Bytes of the header in front of a graphics program's code.
pub const GRAPHICS_HEADER_SIZE: usize = header::SIZE;

/// The most instructions a program may have.
pub const MAX_INSTRUCTIONS: usize = 65536;

/// Translates one program (its bytes as NVN stores them at the stage's GPU
/// address) to SPIR-V: a pure function, callable from many threads at once.
/// Only compute programs are translated so far.
pub fn translate(program: &[u8], parameters: &Parameters) -> Result<Translated, Unsupported> {
    if let Some(stage) = parameters.stage_slot {
        return Err(whole(format!("{stage:?} stage slot")));
    }
    if let Some(header) = header::parse(program) {
        if !matches!(header.stage, Stage::Vertex | Stage::Fragment) {
            return Err(whole(format!(
                "{:?} programs are not translated yet",
                header.stage
            )));
        }
        let code = program_code(&program[GRAPHICS_HEADER_SIZE.min(program.len())..])?;
        let layout = compute::Layout {
            stage: header.stage,
            header: Some(header),
            workgroup: [1, 1, 1],
            shared_bytes: None,
            local_bytes: parameters.local_memory_size,
        };
        // Offsets in errors are from the program's start.
        return compute::translate(&code, &layout).map_err(|mut error| {
            error.offset = error
                .offset
                .map(|offset| offset + GRAPHICS_HEADER_SIZE as u32);
            error
        });
    }
    let workgroup = parameters
        .workgroup_size
        .ok_or_else(|| whole("compute program without a workgroup size".into()))?;
    let code = program_code(program)?;
    let layout = compute::Layout {
        stage: Stage::Compute,
        header: None,
        workgroup,
        shared_bytes: parameters.shared_memory_size,
        local_bytes: parameters.local_memory_size,
    };
    compute::translate(&code, &layout)
}

/// `translate` is the runtime's translator signature.
const _: Translate = translate;

fn whole(reason: String) -> Unsupported {
    Unsupported {
        reason,
        offset: None,
    }
}

/// Whether `program` starts with a graphics header (SPH): a VTG or pixel
/// header with a known stage. A compute program starts with an
/// instruction, whose guard bits (12-14) make the header's shader type
/// field 12 or more for any instruction guarded by PT.
pub fn is_graphics(program: &[u8]) -> bool {
    header::is_graphics(program)
}

/// The program's instructions, up to the unconditional EXIT that no earlier
/// forward branch jumps past.
pub fn program_code(program: &[u8]) -> Result<Vec<Instruction>, Unsupported> {
    let (words, _) = program.as_chunks::<16>();
    let mut code = Vec::new();
    let mut last_forward = 0usize;
    for (index, word) in words.iter().take(MAX_INSTRUCTIONS).enumerate() {
        let instruction = decode(u128::from_le_bytes(*word));
        code.push(instruction);
        match instruction.op {
            Op::Bra { offset, .. } => {
                let target = (index as i64 + 1) * 16 + offset;
                if target > 0 {
                    last_forward = last_forward.max((target / 16) as usize);
                }
            }
            // A subroutine's entry counts as a forward target.
            Op::Call { offset } => {
                let target = (index as i64 + 1) * 16 + offset;
                if target > 0 {
                    last_forward = last_forward.max((target / 16) as usize);
                }
            }
            // The end: an unconditional EXIT (or a subroutine's RET) that
            // no earlier branch or call reaches past.
            Op::Exit | Op::Ret { .. } if instruction.guard.is_true() && index >= last_forward => {
                return Ok(code);
            }
            _ => {}
        }
    }
    Err(whole("no end of program found".into()))
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;
    use echora_shader_abi::PUSH_RESOLUTION_SCALE;

    const EXIT: u128 = 0x794d | 0x7000;
    const MOV_R1_5: u128 = 0x7802 | (1 << 16) | (5 << 32);
    const NOP: u128 = 0x7918;

    fn bytes(words: &[u128]) -> Vec<u8> {
        words.iter().flat_map(|word| word.to_le_bytes()).collect()
    }

    fn bra(from: usize, to: usize) -> u128 {
        let offset = (to as i64 - from as i64 - 1) * 16;
        0x7947 | (u128::from(offset as u64 & ((1 << 50) - 1)) << 32) | (7 << 87)
    }

    #[test]
    fn a_compute_program_ends_at_its_last_reachable_exit() {
        // BRA over an EXIT; the real end is the second EXIT, and the padding
        // after it is not part of the program.
        let program = bytes(&[bra(0, 2), EXIT, MOV_R1_5, EXIT, NOP, NOP]);
        assert!(!is_graphics(&program));
        assert_eq!(program_code(&program).unwrap().len(), 4);
        let parameters = Parameters {
            workgroup_size: Some([32, 1, 1]),
            ..Parameters::default()
        };
        let translated = translate(&program, &parameters).unwrap();
        assert_eq!(translated.stage, Stage::Compute);
        if let Some(problem) = spirv::tests::validate(&translated.spirv) {
            panic!("{problem}");
        }
    }

    /// A word with `fields` (bit offset, width, value) set and a PT guard.
    fn word(fields: &[(u32, u32, u128)]) -> u128 {
        let mut raw = 7u128 << 12;
        for &(offset, _, value) in fields {
            raw |= value << offset;
        }
        raw
    }

    /// A 128-byte header: `sph_type`, `shader_type`, and bits set at the
    /// given positions.
    fn header(sph_type: u32, shader_type: u32, set: &[usize]) -> Vec<u8> {
        let mut bytes = vec![0u8; GRAPHICS_HEADER_SIZE];
        let first = sph_type | shader_type << 10;
        bytes[..4].copy_from_slice(&first.to_le_bytes());
        for &bit in set {
            bytes[bit / 8] |= 1 << (bit % 8);
        }
        bytes
    }

    fn validated(program: &[u8]) -> Translated {
        let translated = translate(program, &Parameters::default()).unwrap();
        if let Some(problem) = spirv::tests::validate(&translated.spirv) {
            panic!("{problem}");
        }
        translated
    }

    #[test]
    fn a_vertex_program_writes_position_and_generics() {
        let mut program = header(1, 1, &[]);
        program.extend(bytes(&[
            // ALD R0, a[0x2fc] (vertex id); I2F R1, R0
            word(&[
                (0, 12, 0x321),
                (16, 8, 0),
                (24, 8, 255),
                (32, 8, 255),
                (40, 10, 0x2fc),
            ]),
            word(&[
                (0, 12, 0x306),
                (16, 8, 1),
                (32, 8, 0),
                (75, 2, 2),
                (84, 2, 2),
            ]),
            // AST a[0x5c], R0 (a hint, dropped)
            word(&[
                (0, 12, 0x322),
                (24, 8, 255),
                (32, 8, 0),
                (40, 10, 0x5c),
                (64, 8, 255),
            ]),
            // AST a[0x70], R0..R3 (4 words); AST a[0x90], R1 (generic 1.x)
            word(&[
                (0, 12, 0x322),
                (24, 8, 255),
                (32, 8, 0),
                (40, 10, 0x70),
                (64, 8, 255),
                (74, 2, 3),
            ]),
            word(&[
                (0, 12, 0x322),
                (24, 8, 255),
                (32, 8, 1),
                (40, 10, 0x90),
                (64, 8, 255),
            ]),
            EXIT,
        ]));
        let translated = validated(&program);
        assert_eq!(translated.stage, Stage::Vertex);
        assert_eq!(translated.stored_generics, 1 << 1);
        // Position x..w (words 28-31) and generic 1.x (word 36).
        assert_eq!(translated.stores[0], 0xf << 28 | 1 << 36);
        // Generic inputs are fetched by the program from the draw's fetch
        // table, through the page table.
        let mut program = header(1, 1, &[]);
        program.extend(bytes(&[
            // ALD R0, a[0x90], 4 words (generic 1); AST a[0x70], R0..R3
            word(&[
                (0, 12, 0x321),
                (16, 8, 0),
                (24, 8, 255),
                (32, 8, 255),
                (40, 10, 0x90),
                (74, 2, 3),
            ]),
            word(&[
                (0, 12, 0x322),
                (24, 8, 255),
                (32, 8, 0),
                (40, 10, 0x70),
                (64, 8, 255),
                (74, 2, 3),
            ]),
            EXIT,
        ]));
        let translated = validated(&program);
        assert_eq!(translated.loaded_generics, 1 << 1);
        assert_eq!(
            translated.push_fields,
            echora_shader_abi::PUSH_VERTEX_FETCH | echora_shader_abi::PUSH_PAGE_TABLE
        );
    }

    #[test]
    fn a_fragment_program_interpolates_samples_and_writes_targets() {
        // Generic 0 perspective (x, y), generic 1 flat (x); render target
        // 0 all four components and depth.
        let mut set = vec![
            192 + 1,
            192 + 2 * 2 + 1,
            192 + 4 * 2,
            576,
            577,
            578,
            579,
            609,
        ];
        set.sort_unstable();
        let mut program = header(2, 5, &set);
        program.extend(bytes(&[
            // IPA R4, a[0x80]; IPA.CENTROID R5, a[0x84]; IPA R6, a[0x90]
            word(&[
                (0, 12, 0x326),
                (16, 8, 4),
                (32, 8, 255),
                (64, 8, 0x20),
                (81, 3, 7),
            ]),
            word(&[
                (0, 12, 0x326),
                (16, 8, 5),
                (32, 8, 255),
                (64, 8, 0x21),
                (76, 2, 1),
                (81, 3, 7),
            ]),
            word(&[
                (0, 12, 0x326),
                (16, 8, 6),
                (32, 8, 255),
                (64, 8, 0x24),
                (78, 2, 1),
                (81, 3, 7),
            ]),
            // IPA.CONSTANT R8, a[0x3fc] (front facing)
            word(&[
                (0, 12, 0x326),
                (16, 8, 8),
                (32, 8, 255),
                (64, 8, 0xff),
                (78, 2, 1),
                (81, 3, 7),
            ]),
            // IPA.IDX R9, a[UR4 + 0x80] (an indexed generic)
            word(&[
                (0, 12, 0x526),
                (16, 8, 9),
                (24, 8, 4),
                (32, 8, 255),
                (64, 8, 0x20),
                (81, 3, 7),
            ]),
            // IPA R7, a[0x70] (frag coord x)
            word(&[
                (0, 12, 0x326),
                (16, 8, 7),
                (32, 8, 255),
                (64, 8, 0x1c),
                (81, 3, 7),
            ]),
            // TEX R0, R4, R5, 2D, 0xf (implicit level), c[2][0x20]
            word(&[
                (0, 12, 0xb60),
                (16, 8, 0),
                (24, 8, 4),
                (32, 8, 5),
                (40, 14, 8),
                (54, 5, 2),
                (61, 3, 1),
                (64, 8, 255),
                (72, 4, 0xf),
                (81, 3, 7),
            ]),
            // TMML R12, R4, R5, 2D, 0x3 (the level and the raw level)
            word(&[
                (0, 12, 0xb69),
                (16, 8, 12),
                (24, 8, 4),
                (32, 8, 5),
                (40, 14, 8),
                (54, 5, 2),
                (61, 3, 1),
                (64, 8, 255),
                (72, 4, 3),
                (81, 3, 7),
            ]),
            // @P0 KILL; MOV R5, R7 (the depth after the sample mask slot)
            word(&[(0, 12, 0x95b), (12, 4, 0), (87, 3, 7)]),
            word(&[(0, 12, 0x202), (16, 8, 5), (32, 8, 7)]),
            EXIT,
        ]));
        let translated = validated(&program);
        assert_eq!(translated.stage, Stage::Fragment);
        assert_eq!(translated.frag_colors, 1);
        assert_eq!(translated.loaded_generics, 0b11);
        assert_eq!(translated.push_fields, PUSH_RESOLUTION_SCALE);
        // Fragment bindings start at slot 4.
        let texture = translated.descriptors[0];
        assert_eq!(
            (texture.binding, texture.cbuf_index, texture.cbuf_offset),
            (4 * compute::MAX_DESCRIPTORS, 2, 0x20)
        );
    }

    #[test]
    fn calls_return_through_the_address_register() {
        // MOV R2, 0x20 (return to instruction 2); CALL to 3; EXIT;
        // MOV R3, 0x1; RET R2. The program ends at the RET.
        let mov = |d: u128, value: u128| word(&[(0, 12, 0x802), (16, 8, d), (32, 32, value)]);
        let call = word(&[(0, 12, 0x944), (32, 50, 16)]);
        let ret = word(&[(0, 12, 0x950), (24, 8, 2)]);
        let program = bytes(&[mov(2, 0x20), call, EXIT, mov(3, 1), ret, NOP, NOP]);
        assert_eq!(program_code(&program).unwrap().len(), 5);
        let parameters = Parameters {
            workgroup_size: Some([32, 1, 1]),
            ..Parameters::default()
        };
        let translated = translate(&program, &parameters).unwrap();
        if let Some(problem) = spirv::tests::validate(&translated.spirv) {
            panic!("{problem}");
        }
    }

    #[test]
    fn graphics_and_endless_programs_are_unsupported() {
        // A geometry program is not translated yet.
        let geometry = header(1, 4, &[]);
        assert!(is_graphics(&geometry));
        assert!(translate(&geometry, &Parameters::default()).is_err());
        assert!(program_code(&bytes(&[MOV_R1_5, NOP])).is_err());
        let no_size = translate(&bytes(&[EXIT]), &Parameters::default());
        assert!(no_size.unwrap_err().reason.contains("workgroup"));
    }
}
