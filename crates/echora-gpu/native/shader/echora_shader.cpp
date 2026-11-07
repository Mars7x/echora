// SPDX-License-Identifier: GPL-3.0-or-later
// Adapter from one native (Ampere) NVN program to the Eden-family (Lichii) shader
// recompiler: the program is captured and lowered to Maxwell form the way
// PokeDinner's GenericEnvironment::LowerNativeCode does (GPL-3.0-or-later,
// Eden Emulator Project and PokeDinner contributors), translated to IR and
// emitted as SPIR-V.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <vector>

#include "echora_shader.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"
#include "shader_recompiler/frontend/ampere/program_address_map.h"
#include "shader_recompiler/frontend/ampere/program_capture.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/object_pool.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"
#include "common/settings.h"

namespace {

using Shader::Ampere::Instruction;

/// Graphics stages carry a 128-byte header whose first 0x50 bytes are a
/// Maxwell shader program header (the recompiler reads it as one).
constexpr size_t NATIVE_GRAPHICS_HEADER = 128;

class Env final : public Shader::Environment {
public:
    std::vector<u64> code;
    u32 texture_bound = 0;
    u32 local_memory = 0;
    u32 shared_memory = 0;
    std::array<u32, 3> workgroup{1, 1, 1};
    EchoraShaderCallbacks callbacks{};

    void Setup(Shader::Stage stage_, const Shader::ProgramHeader& header, u32 start) {
        stage = stage_;
        sph = header;
        start_address = start;
        is_proprietary_driver = true;
    }

    void SetupMesh(const EchoraShaderInput& input) {
        std::copy(std::begin(input.mesh_registers), std::end(input.mesh_registers),
                  mesh_registers.begin());
        if (input.has_mesh_primitive_header != 0) {
            static_assert(sizeof(mesh_primitive_header) == sizeof(input.mesh_primitive_header));
            std::memcpy(&mesh_primitive_header, input.mesh_primitive_header,
                        sizeof(input.mesh_primitive_header));
        }
    }

    u64 ReadInstruction(u32 address) override {
        const size_t index = address / sizeof(u64);
        if (address % sizeof(u64) != 0 || index >= code.size()) {
            throw Shader::RuntimeError("Lowered instruction address out of range: {:#x}", address);
        }
        return code[index];
    }
    u32 TranslateProgramAddress(u32 address) const override {
        if (const auto translated = Shader::Ampere::NativeProgramAddress(code, address)) {
            return *translated;
        }
        throw Shader::RuntimeError("Invalid native indirect branch address {:#x}", address);
    }
    u32 ReadCbufValue(u32 index, u32 offset) override {
        return callbacks.read_cbuf ? callbacks.read_cbuf(callbacks.user, index, offset) : 0;
    }
    Shader::TextureType ReadTextureType(u32 handle) override {
        return callbacks.texture_type
                   ? static_cast<Shader::TextureType>(callbacks.texture_type(callbacks.user, handle))
                   : Shader::TextureType::Color2D;
    }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32) override {
        return Shader::TexturePixelFormat::A8B8G8R8_UNORM;
    }
    bool IsTexturePixelFormatInteger(u32) override { return false; }
    u32 ReadViewportTransformState() override { return 1; }
    u32 TextureBoundBuffer() const override { return texture_bound; }
    u32 LocalMemorySize() const override { return local_memory; }
    u32 SharedMemorySize() const override { return shared_memory; }
    std::array<u32, 3> WorkgroupSize() const override { return workgroup; }
    bool HasHLEMacroState() const override { return false; }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }
    void Dump(u64, u64) override {}
};

/// Maxwell SPH shader_type to stage (1 vertex, 2/3 tessellation, 4
/// geometry, 5 pixel).
bool StageFromHeader(u32 shader_type, Shader::Stage& stage) {
    switch (shader_type) {
    case 1: stage = Shader::Stage::VertexB; return true;
    case 2: stage = Shader::Stage::TessellationControl; return true;
    case 3: stage = Shader::Stage::TessellationEval; return true;
    case 4: stage = Shader::Stage::Geometry; return true;
    case 5: stage = Shader::Stage::Fragment; return true;
    default: return false;
    }
}

Shader::Profile MakeProfile() {
    return Shader::Profile{
        .supported_spirv = 0x00010300,
        .unified_descriptor_binding = true,
        .support_descriptor_aliasing = true,
        .support_int8 = true,
        .support_int16 = true,
        .support_int64 = true,
        .support_vertex_instance_id = false,
        .support_float_controls = true,
        .support_separate_denorm_behavior = true,
        .support_separate_rounding_mode = true,
        .support_fp16_denorm_preserve = true,
        .support_fp32_denorm_preserve = true,
        .support_fp16_denorm_flush = true,
        .support_fp32_denorm_flush = true,
        .support_fp16_signed_zero_nan_preserve = true,
        .support_fp32_signed_zero_nan_preserve = true,
        .support_fp64_signed_zero_nan_preserve = true,
        .support_vote = true,
        .support_shuffle_relative = true,
        .support_viewport_index_layer_non_geometry = true,
        .support_typeless_image_loads = true,
        .support_demote_to_helper_invocation = true,
        .support_derivative_control = true,
        .support_native_ndc = true,
        .support_scaled_attributes = false,
        .support_multi_viewport = true,
        .warp_size_potentially_larger_than_guest = true,
        .max_subgroup_size = 64,
        .min_ssbo_alignment = 16,
        .max_user_clip_distances = 8,
    };
}

Shader::HostTranslateInfo MakeHostInfo() {
    Shader::HostTranslateInfo info{
        .min_ssbo_alignment = 16,
        .support_float64 = true,
        .support_float16 = true,
        .support_int64 = true,
        .support_snorm_render_buffer = true,
        .support_viewport_index_layer = true,
        .support_conditional_barrier = true,
    };
    info.ApplyDescriptorLimitPolicy();
    return info;
}

/// Mirrors the binding assignment in EmitContext's Define* passes: buffers
/// advance by their count, texel buffers, textures and images by one.
u32 DescribeBindings(const Shader::Info& info, u32 base, EchoraShaderOutput* output) {
    u32 binding = base;
    const auto add = [&](EchoraShaderDescriptor descriptor, u32 advance) {
        descriptor.binding = binding;
        binding += advance;
        if (output->descriptor_count == ECHORA_SHADER_MAX_DESCRIPTORS) {
            throw Shader::RuntimeError("more than {} descriptors", ECHORA_SHADER_MAX_DESCRIPTORS);
        }
        output->descriptors[output->descriptor_count++] = descriptor;
    };
    for (const auto& desc : info.constant_buffer_descriptors) {
        add({.kind = ECHORA_DESCRIPTOR_UNIFORM_BUFFER, .count = desc.count, .index = desc.index},
            desc.count);
    }
    for (const auto& desc : info.storage_buffers_descriptors) {
        add({.kind = ECHORA_DESCRIPTOR_STORAGE_BUFFER, .count = desc.count,
             .cbuf_index = desc.cbuf_index, .cbuf_offset = desc.cbuf_offset,
             .flags = desc.is_written ? 1u : 0u},
            desc.count);
    }
    for (const auto& desc : info.texture_buffer_descriptors) {
        add({.kind = ECHORA_DESCRIPTOR_TEXTURE_BUFFER, .count = desc.count,
             .cbuf_index = desc.cbuf_index, .cbuf_offset = desc.cbuf_offset,
             .shift_left = desc.shift_left, .has_secondary = desc.has_secondary,
             .secondary_cbuf_index = desc.secondary_cbuf_index,
             .secondary_cbuf_offset = desc.secondary_cbuf_offset,
             .secondary_shift_left = desc.secondary_shift_left, .size_shift = desc.size_shift},
            1);
    }
    for (const auto& desc : info.image_buffer_descriptors) {
        add({.kind = ECHORA_DESCRIPTOR_IMAGE_BUFFER, .count = desc.count,
             .cbuf_index = desc.cbuf_index, .cbuf_offset = desc.cbuf_offset,
             .size_shift = desc.size_shift,
             .flags = (desc.is_written ? 1u : 0u) | (desc.is_read ? 2u : 0u) |
                      (desc.is_integer ? 4u : 0u)},
            1);
    }
    for (const auto& desc : info.texture_descriptors) {
        add({.kind = ECHORA_DESCRIPTOR_TEXTURE, .count = desc.count,
             .cbuf_index = desc.cbuf_index, .cbuf_offset = desc.cbuf_offset,
             .shift_left = desc.shift_left, .has_secondary = desc.has_secondary,
             .secondary_cbuf_index = desc.secondary_cbuf_index,
             .secondary_cbuf_offset = desc.secondary_cbuf_offset,
             .secondary_shift_left = desc.secondary_shift_left, .size_shift = desc.size_shift,
             .texture_type = static_cast<u32>(desc.type),
             .flags = (desc.is_integer ? 4u : 0u) | (desc.is_depth ? 8u : 0u) |
                      (desc.is_multisample ? 16u : 0u)},
            1);
    }
    for (const auto& desc : info.image_descriptors) {
        add({.kind = ECHORA_DESCRIPTOR_IMAGE, .count = desc.count,
             .cbuf_index = desc.cbuf_index, .cbuf_offset = desc.cbuf_offset,
             .size_shift = desc.size_shift, .texture_type = static_cast<u32>(desc.type),
             .flags = (desc.is_written ? 1u : 0u) | (desc.is_read ? 2u : 0u) |
                      (desc.is_integer ? 4u : 0u)},
            1);
    }
    for (u32 index = 0; index < 32; ++index) {
        if (info.loads.Generic(index)) output->loaded_generics |= 1u << index;
        if (info.stores.Generic(index)) output->stored_generics |= 1u << index;
    }
    for (size_t bit = 0; bit < info.stores.mask.size(); ++bit) {
        if (info.stores.mask[bit]) output->stores[bit / 64] |= u64{1} << (bit % 64);
    }
    for (u32 index = 0; index < 8; ++index) {
        if (info.stores_frag_color[index]) output->frag_colors |= 1u << index;
    }
    for (u32 index = 0; index < 18 && index < info.constant_buffer_used_sizes.size(); ++index) {
        output->cbuf_used_sizes[index] = info.constant_buffer_used_sizes[index];
    }
    return binding;
}

void Fail(EchoraShaderOutput* output, const char* message) {
    std::snprintf(output->error, sizeof(output->error), "%s", message);
}

} // namespace

extern "C" int echora_shader_translate(const EchoraShaderInput* input,
                                        EchoraShaderOutput* output) {
    std::memset(output, 0, sizeof(*output));
    try {
        // The rescaling pass reads this global (translation runs on one
        // thread at a time).
        // A rational scale: up_scale over up_scale * down_factor (patch
        // 0008).
        auto& resolution = Settings::values.resolution_info;
        const u32 numerator = std::max<u32>(input->resolution_scale_numerator, 1);
        const u32 denominator = std::max<u32>(input->resolution_scale_denominator, 1);
        resolution.up_scale = numerator;
        resolution.down_shift = 0;
        resolution.up_factor = static_cast<f32>(numerator) / static_cast<f32>(denominator);
        resolution.down_factor = static_cast<f32>(denominator) / static_cast<f32>(numerator);
        resolution.active = numerator != denominator;
        const u8* bytes = input->program;
        size_t length = input->length;
        Shader::ProgramHeader header{};
        Shader::Stage stage = Shader::Stage::Compute;
        u32 first_word = 0;
        if (length >= 4) std::memcpy(&first_word, bytes, 4);
        // A graphics header's first word has sph_type 1 (VTG) or 2 (PS) and
        // a nonzero shader type; a compute program starts with code.
        const u32 sph_type = first_word & 0x1f;
        const bool graphics =
            (sph_type == 1 || sph_type == 2) && StageFromHeader((first_word >> 10) & 0xf, stage);
        if (graphics && input->mesh_stage == ECHORA_MESH_TASK) stage = Shader::Stage::Task;
        if (graphics && input->mesh_stage == ECHORA_MESH_MESH) stage = Shader::Stage::Mesh;
        if (graphics) {
            if (length < NATIVE_GRAPHICS_HEADER) {
                Fail(output, "program shorter than its header");
                return 0;
            }
            std::memcpy(&header, bytes, sizeof(header));
            bytes += NATIVE_GRAPHICS_HEADER;
            length -= NATIVE_GRAPHICS_HEADER;
        } else {
            stage = Shader::Stage::Compute;
        }

        std::vector<Instruction> native;
        Shader::Ampere::ProgramCaptureState capture;
        for (size_t offset = 0; offset + sizeof(Instruction) <= length &&
                                native.size() < Shader::Ampere::ProgramCaptureState::MaximumInstructions;
             offset += sizeof(Instruction)) {
            Instruction instruction;
            std::memcpy(&instruction, bytes + offset, sizeof(instruction));
            native.push_back(instruction);
            const auto status = capture.Observe(instruction, static_cast<u32>(native.size() - 1));
            if (status == Shader::Ampere::ProgramCaptureState::Result::InvalidTarget) {
                Fail(output, "native branch outside capture bounds");
                return 0;
            }
            if (status == Shader::Ampere::ProgramCaptureState::Result::Complete) break;
        }
        const auto lowered = Shader::Ampere::LowerToMaxwell(native);
        if (!lowered) {
            const size_t failed = lowered.error_offset / sizeof(Instruction);
            const u32 opcode = failed < native.size() ? native[failed].Bits(0, 12) : 0;
            std::snprintf(output->error, sizeof(output->error), "lowering: %.*s (op %03x at %#x)",
                          static_cast<int>(lowered.error.size()), lowered.error.data(), opcode,
                          lowered.error_offset);
            return 0;
        }

        Env env;
        env.callbacks = input->callbacks;
        env.texture_bound = lowered.required_texture_buffer == ~0U ? 0 : lowered.required_texture_buffer;
        // SPH, then the lowered instructions with a scheduling word before
        // every three (Maxwell's 32-byte bundles), then the native address map.
        env.code.resize(sizeof(header) / sizeof(u64));
        std::memcpy(env.code.data(), &header, sizeof(header));
        for (const u64 instruction : lowered.instructions) {
            if ((env.code.size() * sizeof(u64)) % 32 == 0) env.code.push_back(0);
            env.code.push_back(instruction);
        }
        env.code.push_back(0);
        Shader::Ampere::AppendNativeAddressMap(env.code, lowered.source_instruction_offsets);

        u32 cfg_start;
        if (graphics) {
            env.Setup(stage, header, 0);
            env.SetupMesh(*input);
            // At least 16 bytes (local memory is allocated in 16-byte
            // units, inferred): 34 captured programs declare none yet store
            // at offsets up to 8, which the recompiler would drop.
            env.local_memory = std::max<u32>(static_cast<u32>(header.LocalMemorySize()) +
                                                 header.common3.shader_local_memory_crs_size,
                                             16);
            cfg_start = sizeof(header);
        } else {
            env.Setup(stage, header, sizeof(header));
            env.local_memory = input->local_memory_size;
            env.shared_memory = input->shared_memory_size;
            for (int axis = 0; axis < 3; ++axis) {
                env.workgroup[axis] = std::max<u32>(input->workgroup_size[axis], 1);
            }
            cfg_start = sizeof(header);
        }

        Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_blocks;
        Shader::ObjectPool<Shader::IR::Inst> insts;
        Shader::ObjectPool<Shader::IR::Block> blocks;
        const auto host_info = MakeHostInfo();
        Shader::Maxwell::Flow::CFG cfg(env, flow_blocks, cfg_start, stage == Shader::Stage::VertexB);
        auto program = Shader::Maxwell::TranslateProgram(insts, blocks, env, cfg, host_info);
        Shader::RuntimeInfo runtime_info{};
        for (size_t index = 0; index < runtime_info.generic_input_types.size(); ++index) {
            runtime_info.generic_input_types[index] =
                static_cast<Shader::AttributeType>(input->generic_input_types[index]);
        }
        if (input->has_previous_stores) {
            for (size_t bit = 0; bit < runtime_info.previous_stage_stores.mask.size(); ++bit) {
                runtime_info.previous_stage_stores.mask[bit] =
                    (input->previous_stores[bit / 64] >> (bit % 64)) & 1;
            }
        } else {
            runtime_info.previous_stage_stores.mask.set();
        }
        Shader::Maxwell::ConvertLegacyToGeneric(program, runtime_info);
        const auto profile = MakeProfile();
        Shader::Backend::Bindings bindings;
        bindings.unified = input->binding_base;
        bindings.texture_scaling_index = input->texture_scaling_base;
        bindings.image_scaling_index = input->image_scaling_base;
        std::vector<u32> spirv =
            Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);

        if (const u32 described = DescribeBindings(program.info, input->binding_base, output);
            described != bindings.unified) {
            throw Shader::RuntimeError("binding layout mismatch: emitted {}, described {}",
                                       bindings.unified, described);
        }
        output->words = spirv.size();
        output->spirv = new u32[spirv.size()];
        std::copy(spirv.begin(), spirv.end(), output->spirv);
        output->stage = static_cast<u32>(stage);
        return 1;
    } catch (const std::exception& error) {
        Fail(output, error.what());
        return 0;
    }
}

extern "C" void echora_shader_free(EchoraShaderOutput* output) {
    delete[] output->spirv;
    output->spirv = nullptr;
    output->words = 0;
}

extern "C" size_t echora_shader_output_size(void) {
    return sizeof(EchoraShaderOutput);
}

extern "C" size_t echora_shader_input_size(void) {
    return sizeof(EchoraShaderInput);
}
