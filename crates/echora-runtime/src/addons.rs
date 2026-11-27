//! Build-specific guest options selected outside the graphics backend.

use std::collections::HashMap;

use echora_config::OcarinaOfTimeAddons;
use echora_gpu::{
    ash::vk,
    draw::{ImageShape, Sampler, TextureSource},
    renderer::Renderer,
    shader::{DescriptorKind, Translated},
};
use echora_loader::{NsoModule, process::PlacedModule};

const OCARINA_TITLE_ID: u64 = 0x0400_4860_24de_c000;
const OCARINA_BUILD_ID: [u8; 32] = [
    0xe0, 0xf3, 0xcc, 0xfa, 0xf5, 0x00, 0x65, 0xae, 0x4e, 0xa0, 0x1e, 0xb5, 0xae, 0xb1, 0x3b, 0x1a,
    0x45, 0xda, 0x33, 0x48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];
const RESOLUTION_BRANCH: u64 = 0x80_5bd0;
const RESOLUTION_BRANCH_WORD: u32 = 0x3600_41e8;
const FIXED_RESOLUTION_PATH: u64 = 0x80_640c;
const NDSR_INPUT_PREP: u64 = 0x4259_3194_bb0b_2b32;
const NDSR_WEIGHT_HANDLE: u32 = 0xb8;
const NDSR_FIXED_RESOLVE: u64 = 0xda6d_f3e5_4c71_6df2;
const NDSR_ADAPTIVE_RESOLVE: u64 = 0xf48d_b748_1be8_56b9;
const NDSR_SHARPENING_STRENGTH: std::ops::Range<usize> = 0x80..0x84;
const NEUTRAL_WEIGHTS_TARGET: u64 = u64::MAX - 7;

#[derive(Debug, Default)]
pub(crate) struct GraphicsAddons {
    neutral_smaa_weights: bool,
    disable_ndsr_sharpening: bool,
}

impl GraphicsAddons {
    pub(crate) fn for_game(
        program_id: u64,
        modules: &[PlacedModule],
        options: &OcarinaOfTimeAddons,
    ) -> Self {
        let supported = modules.iter().any(|placed| {
            placed.module.name == "main" && matches_ocarina_build(program_id, &placed.module)
        });
        Self {
            neutral_smaa_weights: supported && !options.smaa,
            disable_ndsr_sharpening: supported && !options.sharpening,
        }
    }

    /// Both resolve shaders read the gain for their sharpening filter from
    /// c0. Zero gain keeps the rest of the resolve and its output intact.
    pub(crate) fn apply_compute_constants(&self, code_key: u64, constants: &mut [u8]) {
        if self.disable_ndsr_sharpening
            && matches!(code_key, NDSR_FIXED_RESOLVE | NDSR_ADAPTIVE_RESOLVE)
            && let Some(strength) = constants.get_mut(NDSR_SHARPENING_STRENGTH)
        {
            strength.fill(0);
        }
    }

    fn weight_binding(&self, code_key: u64, stage: &Translated) -> Option<u32> {
        if !self.neutral_smaa_weights || code_key != NDSR_INPUT_PREP {
            return None;
        }
        stage
            .descriptors
            .iter()
            .find(|descriptor| {
                descriptor.kind() == DescriptorKind::Texture
                    && descriptor.cbuf_index == 0
                    && descriptor.cbuf_offset == NDSR_WEIGHT_HANDLE
                    && descriptor.texture_type == 2
            })
            .map(|descriptor| descriptor.binding)
    }

    /// The game blends SMAA weights when it prepares input for NDSR.
    /// A cleared plane disables edge blending while the resolve and output pass
    /// continue to run, including the adaptive-resolve path.
    pub(crate) fn apply_compute_textures(
        &self,
        code_key: u64,
        stage: &Translated,
        renderer: &mut Renderer,
        textures: &mut HashMap<u32, (TextureSource, Sampler)>,
    ) -> Result<(), String> {
        let Some(binding) = self.weight_binding(code_key, stage) else {
            return Ok(());
        };
        let Some((source, _)) = textures.get_mut(&binding) else {
            return Ok(());
        };
        let target = match *source {
            TextureSource::Target(key)
            | TextureSource::TargetView {
                key,
                base_level: 0,
                base_layer: 0,
                ..
            } => key,
            _ => return Ok(()),
        };
        let Some(original) = renderer.existing_target(target).filter(|target| {
            target.format == vk::Format::R8G8B8A8_UNORM
                && target.shape == (ImageShape::D2 { layers: 1 })
                && target.levels == 1
        }) else {
            return Ok(());
        };
        let (width, height, shape, format) = (
            original.width,
            original.height,
            original.shape,
            original.format,
        );
        let reusable = renderer
            .existing_target(NEUTRAL_WEIGHTS_TARGET)
            .is_some_and(|target| (target.width, target.height) == (width, height));
        if !reusable {
            renderer.guest_sized_target(NEUTRAL_WEIGHTS_TARGET, width, height, shape, format, 1)?;
            renderer.clear_color(
                NEUTRAL_WEIGHTS_TARGET,
                None,
                vk::ClearColorValue { float32: [0.0; 4] },
            )?;
        }
        match source {
            TextureSource::Target(key) | TextureSource::TargetView { key, .. } => {
                *key = NEUTRAL_WEIGHTS_TARGET;
            }
            TextureSource::Null | TextureSource::Upload { .. } => unreachable!(),
        }
        Ok(())
    }
}

fn matches_ocarina_build(program_id: u64, module: &NsoModule) -> bool {
    program_id == OCARINA_TITLE_ID && module.build_id == OCARINA_BUILD_ID
}

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
    if !matches_ocarina_build(program_id, module) {
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
    fn smaa_override_only_selects_the_input_weights() {
        use echora_gpu::shader::Descriptor;
        use echora_shader_abi::{ABI_VERSION, Stage};
        let stage = Translated {
            abi_version: ABI_VERSION,
            stage: Stage::Compute,
            spirv: Vec::new(),
            descriptors: vec![
                Descriptor {
                    kind: 4,
                    binding: 1,
                    count: 1,
                    cbuf_offset: 0xa8,
                    texture_type: 2,
                    ..Default::default()
                },
                Descriptor {
                    kind: 4,
                    binding: 2,
                    count: 1,
                    cbuf_offset: NDSR_WEIGHT_HANDLE,
                    texture_type: 2,
                    ..Default::default()
                },
            ],
            cbuf_used_sizes: [0; 18],
            loaded_generics: 0,
            stored_generics: 0,
            stores: [0; 8],
            frag_colors: 0,
            frag_color_components: 0,
            spec_constants: Vec::new(),
            push_fields: 0,
            texture_pool_bindings: 0,
        };
        let off = GraphicsAddons {
            neutral_smaa_weights: true,
            ..Default::default()
        };
        assert_eq!(off.weight_binding(NDSR_INPUT_PREP, &stage), Some(2));
        assert_eq!(off.weight_binding(NDSR_INPUT_PREP + 1, &stage), None);
        assert_eq!(
            GraphicsAddons::default().weight_binding(NDSR_INPUT_PREP, &stage),
            None
        );
    }

    #[test]
    fn sharpening_override_only_zeros_the_resolve_gain() {
        let off = GraphicsAddons {
            disable_ndsr_sharpening: true,
            ..Default::default()
        };
        let original = [0xff; 192];
        let mut expected = original;
        expected[128..132].fill(0);
        for code_key in [NDSR_FIXED_RESOLVE, NDSR_ADAPTIVE_RESOLVE] {
            let mut constants = original;
            GraphicsAddons::default().apply_compute_constants(code_key, &mut constants);
            assert_eq!(constants, original);
            off.apply_compute_constants(code_key, &mut constants);
            assert_eq!(constants, expected);
        }
        let mut constants = original;
        off.apply_compute_constants(NDSR_INPUT_PREP, &mut constants);
        off.apply_compute_constants(NDSR_FIXED_RESOLVE, &mut constants[..131]);
        assert_eq!(constants, original);
    }

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
