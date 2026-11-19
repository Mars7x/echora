//! Build-specific guest options selected outside the graphics backend.

use echora_loader::{NsoModule, process::PlacedModule};

const OCARINA_TITLE_ID: u64 = 0x0400_4860_24de_c000;
const OCARINA_BUILD_ID: [u8; 32] = [
    0xe0, 0xf3, 0xcc, 0xfa, 0xf5, 0x00, 0x65, 0xae, 0x4e, 0xa0, 0x1e, 0xb5, 0xae, 0xb1, 0x3b, 0x1a,
    0x45, 0xda, 0x33, 0x48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];
const RESOLUTION_BRANCH: u64 = 0x80_5bd0;
const RESOLUTION_BRANCH_WORD: u32 = 0x3600_41e8;
const FIXED_RESOLUTION_PATH: u64 = 0x80_640c;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolutionBranch {
    pub address: u64,
    pub target: u64,
    pub test_register: usize,
}

/// Find the game's branch to its existing fixed-resolution path. Both
/// destinations are translated so the launcher option can change at startup.
pub fn resolution_branch(program_id: u64, modules: &[PlacedModule]) -> Option<ResolutionBranch> {
    let main = modules.iter().find(|placed| placed.module.name == "main")?;
    match_resolution_branch(program_id, &main.module, main.base)
}

fn match_resolution_branch(
    program_id: u64,
    module: &NsoModule,
    base: u64,
) -> Option<ResolutionBranch> {
    if program_id != OCARINA_TITLE_ID || module.build_id != OCARINA_BUILD_ID {
        return None;
    }
    let text_start = u64::from(module.text.memory_offset);
    let instruction = |offset: u64| -> Option<u32> {
        let at = usize::try_from(offset.checked_sub(text_start)?).ok()?;
        let bytes = module.text.bytes.get(at..at.checked_add(4)?)?;
        Some(u32::from_le_bytes(bytes.try_into().ok()?))
    };
    // On this build TBZ W8,#0 selects a fixed scale of 1.0. Keep the
    // original test when the user opts into the game's resolution changes.
    if instruction(RESOLUTION_BRANCH)? != RESOLUTION_BRANCH_WORD {
        return None;
    }
    instruction(FIXED_RESOLUTION_PATH)?;
    instruction(RESOLUTION_BRANCH + 4)?;
    Some(ResolutionBranch {
        address: base.checked_add(RESOLUTION_BRANCH)?,
        target: base.checked_add(FIXED_RESOLUTION_PATH)?,
        test_register: 8,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use echora_loader::{Compression, Segment};

    #[test]
    fn fixed_resolution_requires_the_matching_executable_and_branch() {
        let mut segment = Segment {
            memory_offset: 0x80_5000,
            file_offset: 0,
            stored_size: 0x2000,
            compression: Compression::None,
            sha256: [0; 32],
            hash_verified: false,
            bytes: vec![0; 0x2000],
        };
        let index = (RESOLUTION_BRANCH - u64::from(segment.memory_offset)) as usize;
        segment.bytes[index..index + 4].copy_from_slice(&RESOLUTION_BRANCH_WORD.to_le_bytes());
        let mut main = NsoModule {
            name: "main".into(),
            build_id: OCARINA_BUILD_ID,
            flags: 0,
            rodata: segment.clone(),
            data: segment.clone(),
            text: segment,
            bss_size: 0,
            image_size: 0x80_7000,
            mod0: None,
        };
        assert_eq!(
            match_resolution_branch(OCARINA_TITLE_ID, &main, 0x71_0000_0000),
            Some(ResolutionBranch {
                address: 0x71_0080_5bd0,
                target: 0x71_0080_640c,
                test_register: 8,
            })
        );
        assert!(match_resolution_branch(1, &main, 0).is_none());
        main.build_id[0] ^= 1;
        assert!(match_resolution_branch(OCARINA_TITLE_ID, &main, 0).is_none());
        main.build_id = OCARINA_BUILD_ID;
        main.text.bytes[index] ^= 1;
        assert!(match_resolution_branch(OCARINA_TITLE_ID, &main, 0).is_none());
    }
}
