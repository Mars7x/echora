//! NVIDIA Ampere (SM 8.x) shader machine code, as used by NVN on the target.
//!
//! Instructions are 128 bits. Field positions here are inferences recorded
//! from public references (Mesa NAK's SM70+ encoder and an Ampere capture
//! routine in a GPL-3.0 yuzu-derived emulator), not NVIDIA documentation:
//!
//! * bits 0..12: opcode (12 bits);
//! * bits 12..16: guard predicate (bit 15 negates; index 7 is PT, "always");
//! * bits 16..24: destination register for most ALU forms.

/// One 128-bit instruction as four little-endian words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Instruction(pub [u32; 4]);

/// Opcodes this module interprets (inferred; see the module notes).
pub mod opcode {
    /// BRA with an 18+32-bit signed relative target.
    pub const BRA: u32 = 0x947;
    /// Branch form with a 32-bit signed relative target in word 1.
    pub const BRA_ABS32: u32 = 0x945;
    pub const CALL: u32 = 0x944;
    pub const EXIT: u32 = 0x94d;
    pub const RET: u32 = 0x950;
}

/// Guard predicate value meaning "always" (PT, not negated).
pub const ALWAYS: u32 = 7;

/// Most programs fit far below this; a longer scan is treated as malformed.
pub const MAX_INSTRUCTIONS: usize = 65536;

impl Instruction {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self(std::array::from_fn(|i| {
            u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap())
        }))
    }

    /// `count` bits (at most 32) starting at bit `offset`.
    pub fn bits(&self, offset: u32, count: u32) -> u32 {
        let word = (offset / 32) as usize;
        let mut value = u64::from(self.0[word]);
        if word + 1 < 4 {
            value |= u64::from(self.0[word + 1]) << 32;
        }
        ((value >> (offset % 32)) & ((1u64 << count) - 1)) as u32
    }

    pub fn opcode(&self) -> u32 {
        self.bits(0, 12)
    }

    pub fn predicate(&self) -> u32 {
        self.bits(12, 4)
    }

    /// Relative byte target of a branch or call, measured from the next
    /// instruction.
    pub fn branch_offset(&self) -> Option<i64> {
        match self.opcode() {
            opcode::BRA_ABS32 => Some(i64::from(self.0[1] as i32)),
            opcode::BRA | opcode::CALL => {
                let raw = u64::from(self.0[1]) | (u64::from(self.0[2] & 0x3ffff) << 32);
                // Sign-extend the 50-bit field.
                Some(((raw << 14) as i64) >> 14)
            }
            _ => None,
        }
    }
}

/// Number of instructions in the program at the start of `code`: up to the
/// unconditional EXIT that no earlier forward branch jumps past or, when the
/// program calls subroutines, the last unconditional RET after that. None
/// when no end is found or a branch leaves the scanned range.
pub fn program_length(code: &[u8]) -> Option<usize> {
    let count = (code.len() / 16).min(MAX_INSTRUCTIONS);
    let (mut last_call, mut last_forward) = (0usize, 0usize);
    let mut main_exited = false;
    for index in 0..count {
        let instruction = Instruction::from_bytes(&code[index * 16..index * 16 + 16]);
        if let Some(offset) = instruction.branch_offset() {
            let target = index as i64 * 16 + 16 + offset;
            if target < 0 || target % 16 != 0 || target >= (MAX_INSTRUCTIONS * 16) as i64 {
                return None;
            }
            let target = (target / 16) as usize;
            if instruction.opcode() == opcode::CALL {
                last_call = last_call.max(target);
            } else {
                last_forward = last_forward.max(target);
            }
        }
        let always = instruction.predicate() == ALWAYS;
        match instruction.opcode() {
            opcode::EXIT if always && index >= last_forward => {
                if last_call == 0 {
                    return Some(index + 1);
                }
                main_exited = true;
            }
            opcode::RET if always && main_exited && index >= last_call && index >= last_forward => {
                return Some(index + 1);
            }
            _ => {}
        }
    }
    None
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    fn op(opcode: u32, predicate: u32, word1: u32, word2: u32) -> [u8; 16] {
        let words = [opcode | (predicate << 12), word1, word2, 0];
        let mut bytes = [0u8; 16];
        for (i, word) in words.iter().enumerate() {
            bytes[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    fn program(instructions: &[[u8; 16]]) -> Vec<u8> {
        instructions.concat()
    }

    #[test]
    fn fields_and_branch_offsets() {
        let bra = Instruction::from_bytes(&op(opcode::BRA, 3, (-32i32) as u32, 0x3ffff));
        assert_eq!((bra.opcode(), bra.predicate()), (opcode::BRA, 3));
        assert_eq!(bra.branch_offset(), Some(-32));
        let forward = Instruction::from_bytes(&op(opcode::BRA_ABS32, ALWAYS, 48, 0));
        assert_eq!(forward.branch_offset(), Some(48));
        assert_eq!(
            Instruction::from_bytes(&op(0x210, ALWAYS, 0, 0)).branch_offset(),
            None
        );
    }

    #[test]
    fn program_ends_at_exit_not_jumped_over() {
        let nop = op(0x918, ALWAYS, 0, 0);
        // A guarded EXIT does not end the program; a branch past the first
        // unconditional EXIT moves the end to the second.
        let code = program(&[
            op(opcode::EXIT, 0, 0, 0),
            op(opcode::BRA, 0, 32, 0),
            op(opcode::EXIT, ALWAYS, 0, 0),
            nop,
            op(opcode::EXIT, ALWAYS, 0, 0),
            nop,
        ]);
        assert_eq!(program_length(&code), Some(5));
        assert_eq!(program_length(&code[..64]), None);
    }

    #[test]
    fn subroutines_extend_the_program_to_their_return() {
        let nop = op(0x918, ALWAYS, 0, 0);
        let code = program(&[
            op(opcode::CALL, ALWAYS, 16, 0),
            op(opcode::EXIT, ALWAYS, 0, 0),
            nop,
            op(opcode::RET, ALWAYS, 0, 0),
            nop,
        ]);
        assert_eq!(program_length(&code), Some(4));
    }
}
