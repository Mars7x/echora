//! What a translated program needs of the device beyond core Vulkan 1.3
//! (`translate_with_requirements`): the features (`VkPhysicalDevice*Features`
//! members, or the properties and extensions named) its SPIR-V uses, each
//! with the guest instruction that first needs it, so that a device
//! lacking one refuses the program by name (`missing`) instead of failing
//! validation.

use echora_shader_abi::{Options, Stage, Translated, Unsupported};

use crate::compute::Needs;
use crate::spirv::capability;

/// A device feature a program needs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Requirement {
    /// The Vulkan feature (a `VkPhysicalDevice*Features` member) or, for
    /// subgroup operations, the `subgroupSupportedOperations` bit.
    pub feature: &'static str,
    /// What needs it: the SPIR-V capability or the program's use.
    pub reason: &'static str,
    /// The guest instruction first needing it (byte offset from the
    /// program's start), when one does.
    pub first_use: Option<u32>,
}

/// The device features each capability the translator emits needs (none:
/// core Vulkan 1.3). Subgroup capabilities need the operations in
/// `subgroupSupportedOperations` (for the program's stage in
/// `subgroupSupportedStages`).
fn capability_features(capability: u32) -> &'static [(&'static str, &'static str)] {
    match capability {
        capability::FLOAT16 => &[("shaderFloat16", "Float16 (half arithmetic)")],
        capability::FLOAT64 => &[("shaderFloat64", "Float64 (exact denormals through f64)")],
        capability::INT64 => &[("shaderInt64", "Int64 (64-bit addresses and arithmetic)")],
        capability::IMAGE_GATHER_EXTENDED => &[(
            "shaderImageGatherExtended",
            "ImageGatherExtended (gathers with offsets)",
        )],
        capability::MIN_LOD => &[("shaderResourceMinLod", "MinLod (TEX .LC least levels)")],
        capability::SAMPLED_CUBE_ARRAY => &[("imageCubeArray", "SampledCubeArray (cube arrays)")],
        capability::STORAGE_IMAGE_READ_WITHOUT_FORMAT => &[(
            "shaderStorageImageReadWithoutFormat",
            "StorageImageReadWithoutFormat (typeless SULD)",
        )],
        capability::STORAGE_IMAGE_WRITE_WITHOUT_FORMAT => &[(
            "shaderStorageImageWriteWithoutFormat",
            "StorageImageWriteWithoutFormat (typeless SUST)",
        )],
        capability::GROUP_NON_UNIFORM => &[("subgroup basic", "GroupNonUniform")],
        capability::GROUP_NON_UNIFORM_VOTE => &[("subgroup vote", "GroupNonUniformVote (VOTE)")],
        capability::GROUP_NON_UNIFORM_ARITHMETIC => {
            &[("subgroup arithmetic", "GroupNonUniformArithmetic (REDUX)")]
        }
        capability::GROUP_NON_UNIFORM_BALLOT => &[(
            "subgroup ballot",
            "GroupNonUniformBallot (lane masks, R2UR)",
        )],
        capability::GROUP_NON_UNIFORM_SHUFFLE => {
            &[("subgroup shuffle", "GroupNonUniformShuffle (SHFL, FSWZADD)")]
        }
        capability::SHADER_CLOCK => &[("shaderSubgroupClock", "ShaderClockKHR (CS2R clock)")],
        capability::DRAW_PARAMETERS => &[("shaderDrawParameters", "DrawParameters")],
        capability::DEMOTE_TO_HELPER_INVOCATION => &[(
            "shaderDemoteToHelperInvocation",
            "DemoteToHelperInvocation (KILL)",
        )],
        capability::RUNTIME_DESCRIPTOR_ARRAY => &[(
            "runtimeDescriptorArray",
            "RuntimeDescriptorArray (texture pool)",
        )],
        capability::SAMPLED_IMAGE_ARRAY_NON_UNIFORM_INDEXING => &[(
            "shaderSampledImageArrayNonUniformIndexing",
            "SampledImageArrayNonUniformIndexing (texture pool handles)",
        )],
        capability::ATOMIC_FLOAT32_ADD => &[(
            "shaderBufferFloat32AtomicAdd",
            "AtomicFloat32AddEXT (RED/ATOM .F32 ADD)",
        )],
        capability::COOPERATIVE_MATRIX => &[("cooperativeMatrix", "CooperativeMatrixKHR (HMMA)")],
        capability::MESH_SHADING_EXT => &[("meshShader", "MeshShadingEXT")],
        capability::SHADER_LAYER => {
            &[("shaderOutputLayer", "ShaderLayer (layer outside geometry)")]
        }
        capability::SHADER_VIEWPORT_INDEX => &[(
            "shaderOutputViewportIndex",
            "ShaderViewportIndex (viewport outside geometry)",
        )],
        capability::PHYSICAL_STORAGE_BUFFER_ADDRESSES => &[(
            "bufferDeviceAddress",
            "PhysicalStorageBufferAddresses (global memory)",
        )],
        _ => &[],
    }
}

/// The requirements of `translated`, from what its translation recorded
/// (`needs`) and the `options` it was made with.
pub fn requirements(translated: &Translated, needs: &Needs, options: &Options) -> Vec<Requirement> {
    let mut list = Vec::new();
    let mut add = |feature, reason, first_use| {
        if !list.iter().any(|r: &Requirement| r.feature == feature) {
            list.push(Requirement {
                feature,
                reason,
                first_use,
            });
        }
    };
    for &(capability, site) in &needs.capabilities {
        for &(feature, reason) in capability_features(capability) {
            add(feature, reason, site);
        }
    }
    if translated.stage == Stage::Task {
        add("taskShader", "a task program", None);
    }
    if let Some(site) = needs.store {
        match translated.stage {
            Stage::Vertex | Stage::TessControl | Stage::TessEval | Stage::Geometry => add(
                "vertexPipelineStoresAndAtomics",
                "a buffer or image store or atomic in a vertex-pipeline stage",
                site,
            ),
            Stage::Fragment => add(
                "fragmentStoresAndAtomics",
                "a buffer or image store or atomic in a fragment program",
                site,
            ),
            _ => {}
        }
    }
    if options.dual_source_blend && translated.stage == Stage::Fragment {
        add(
            "dualSrcBlend",
            "Options::dual_source_blend (Index 1 output)",
            None,
        );
    }
    list
}

/// A refusal naming the first requirement `supported` (by feature name)
/// denies, if any.
pub fn missing(
    requirements: &[Requirement],
    supported: impl Fn(&str) -> bool,
) -> Result<(), Unsupported> {
    match requirements.iter().find(|r| !supported(r.feature)) {
        Some(requirement) => Err(Unsupported {
            reason: format!(
                "the device lacks {} (needed by {})",
                requirement.feature, requirement.reason
            ),
            offset: requirement.first_use,
        }),
        None => Ok(()),
    }
}
