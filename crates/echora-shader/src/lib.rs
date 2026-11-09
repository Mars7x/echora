//! Translation of the Switch 2 GPU's shader programs (NVIDIA Ampere, SM 8.x
//! SASS) to SPIR-V, ahead of time and without runtime state. See
//! `docs/shader-translator.md` for the design and its status.
//!
//! So far: [`decode`], 128-bit instruction words into typed instructions,
//! and [`compute`], simple compute programs to SPIR-V. Anything
//! [`translate`] reports as `Unsupported` goes to the plugin translator.

pub mod compute;
pub mod decode;
pub mod spirv;

pub use decode::{Instruction, Op, decode, decode_program};
use echora_shader_abi::{Parameters, Translate, Translated, Unsupported};

/// Bytes of the header in front of a graphics program's code.
pub const GRAPHICS_HEADER_SIZE: usize = 128;

/// The most instructions a program may have.
pub const MAX_INSTRUCTIONS: usize = 65536;

/// Translates one program (its bytes as NVN stores them at the stage's GPU
/// address) to SPIR-V: a pure function, callable from many threads at once.
/// Only compute programs are translated so far.
pub fn translate(program: &[u8], parameters: &Parameters) -> Result<Translated, Unsupported> {
    if let Some(stage) = parameters.stage_slot {
        return Err(whole(format!("{stage:?} stage slot")));
    }
    if is_graphics(program) {
        return Err(whole("graphics stages are not translated yet".into()));
    }
    let workgroup = parameters
        .workgroup_size
        .ok_or_else(|| whole("compute program without a workgroup size".into()))?;
    let code = program_code(program)?;
    let layout = compute::Layout {
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

/// Whether `program` starts with a graphics header (SPH): its first word
/// names a VTG or pixel header (bits 0-4: 1 or 2) and a shader type (bits
/// 10-13: 1 vertex, 3 tessellation evaluation, 4 geometry, 5 pixel). A
/// compute program starts with an instruction, whose guard bits (12-14)
/// make that type field 12 or more for any instruction guarded by PT.
pub fn is_graphics(program: &[u8]) -> bool {
    let Some(&first) = program.first_chunk::<4>() else {
        return false;
    };
    let word = u32::from_le_bytes(first);
    matches!(word & 0x1f, 1 | 2) && matches!((word >> 10) & 0xf, 1 | 3 | 4 | 5)
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
            Op::Exit
                if instruction.guard.index == 7
                    && !instruction.guard.negated
                    && index >= last_forward =>
            {
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
    use echora_shader_abi::Stage;

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

    #[test]
    fn graphics_and_endless_programs_are_unsupported() {
        let mut header = vec![0u8; GRAPHICS_HEADER_SIZE];
        header[..4].copy_from_slice(&(2u32 | (5 << 10)).to_le_bytes());
        assert!(is_graphics(&header));
        assert!(translate(&header, &Parameters::default()).is_err());
        assert!(program_code(&bytes(&[MOV_R1_5, NOP])).is_err());
        let no_size = translate(&bytes(&[EXIT]), &Parameters::default());
        assert!(no_size.unwrap_err().reason.contains("workgroup"));
    }
}
