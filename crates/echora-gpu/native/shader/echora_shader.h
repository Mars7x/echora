// SPDX-License-Identifier: GPL-3.0-or-later
// C interface to the shader recompiler: one native NVN program in, SPIR-V out.
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum EchoraShaderStage {
    ECHORA_SHADER_VERTEX = 0,
    ECHORA_SHADER_TESS_CONTROL = 1,
    ECHORA_SHADER_TESS_EVAL = 2,
    ECHORA_SHADER_GEOMETRY = 3,
    ECHORA_SHADER_FRAGMENT = 4,
    ECHORA_SHADER_COMPUTE = 5,
    /// Switch 2 task and mesh programs (the recompiler's Stage::Task and
    /// Stage::Mesh; 6 is its VertexA).
    ECHORA_SHADER_TASK = 7,
    ECHORA_SHADER_MESH = 8,
};

/// What a graphics program is translated as.
enum EchoraMeshStage {
    ECHORA_MESH_NONE = 0,
    ECHORA_MESH_TASK = 1,
    ECHORA_MESH_MESH = 2,
};

struct EchoraShaderCallbacks {
    void* user;
    /// Word at `offset` of constant buffer `index`; null reads zero.
    uint32_t (*read_cbuf)(void* user, uint32_t index, uint32_t offset);
    /// Shader::TextureType of a raw texture handle; null means Color2D.
    uint32_t (*texture_type)(void* user, uint32_t handle);
};

struct EchoraShaderInput {
    /// Program bytes as NVN stores them at the stage's GPU address: graphics
    /// stages start with their 128-byte header, compute programs with code.
    const uint8_t* program;
    size_t length;
    /// Compute only.
    uint32_t workgroup_size[3];
    uint32_t shared_memory_size;
    uint32_t local_memory_size;
    /// First set-0 binding this stage uses, so stages of one pipeline can
    /// share a set (each stage starts after the previous one's bindings).
    uint32_t binding_base;
    /// Nonzero when `previous_stores` holds the attribute stores (Shader
    /// IR::Attribute bit mask) of the stage before this one; otherwise every
    /// input is treated as provided.
    uint32_t has_previous_stores;
    uint64_t previous_stores[8];
    /// Vertex stages: Shader::AttributeType of each generic input (0 float,
    /// 1 signed integer, 2 unsigned integer).
    uint8_t generic_input_types[32];
    /// Resolution scale of render targets, numerator / denominator (0 or
    /// equal: none). Otherwise the recompiler's rescaling pass reads, from a
    /// push constant block (Shader::Backend::SPIRV::RescalingLayout), which
    /// textures and images are scaled and the render target's down factor.
    uint32_t resolution_scale_numerator;
    uint32_t resolution_scale_denominator;
    /// First texture and image scaling bits this stage uses (the counts of
    /// texture and image descriptors of the stages before it).
    uint32_t texture_scaling_base;
    uint32_t image_scaling_base;
    /// EchoraMeshStage: a task or mesh program (its header names the vertex
    /// or tessellation slot it is bound in) instead of the header's stage.
    uint32_t mesh_stage;
    /// Task and mesh programs: SET_MESH_SHADER_A (topology 3:0, max vertex
    /// 12:4, max primitive 22:13), SET_MESH_SHADER_B (mesh threads 25:20)
    /// and SET_MESH_INIT_SHADER (task threads 5:0, local buffer lines 20:12,
    /// lines passed to the mesh stage 30:22).
    uint32_t mesh_registers[3];
    /// Mesh programs: nonzero when `mesh_primitive_header` holds the program
    /// header of the pass-through geometry program bound with them, whose
    /// output map lists the per-primitive outputs.
    uint32_t has_mesh_primitive_header;
    uint8_t mesh_primitive_header[0x50];
    EchoraShaderCallbacks callbacks;
};

enum EchoraDescriptorKind {
    ECHORA_DESCRIPTOR_UNIFORM_BUFFER = 0,
    ECHORA_DESCRIPTOR_STORAGE_BUFFER = 1,
    ECHORA_DESCRIPTOR_TEXTURE_BUFFER = 2,
    ECHORA_DESCRIPTOR_IMAGE_BUFFER = 3,
    ECHORA_DESCRIPTOR_TEXTURE = 4,
    ECHORA_DESCRIPTOR_IMAGE = 5,
};

/// One SPIR-V binding (set 0) in the order the backend assigns them. A
/// uniform buffer names the guest constant buffer `index`; the other kinds
/// locate their handle or address in constant buffer `cbuf_index` at byte
/// `cbuf_offset` (texture handles may combine a secondary word).
struct EchoraShaderDescriptor {
    uint32_t kind;
    uint32_t binding;
    uint32_t count;
    uint32_t index;
    uint32_t cbuf_index;
    uint32_t cbuf_offset;
    uint32_t shift_left;
    uint32_t has_secondary;
    uint32_t secondary_cbuf_index;
    uint32_t secondary_cbuf_offset;
    uint32_t secondary_shift_left;
    uint32_t size_shift;
    /// Shader::TextureType for textures and images.
    uint32_t texture_type;
    /// Bit 0 written, 1 read, 2 integer, 3 depth, 4 multisample.
    uint32_t flags;
};

#define ECHORA_SHADER_MAX_DESCRIPTORS 64

struct EchoraShaderOutput {
    /// Free with echora_shader_free.
    uint32_t* spirv;
    size_t words;
    uint32_t stage;
    EchoraShaderDescriptor descriptors[ECHORA_SHADER_MAX_DESCRIPTORS];
    uint32_t descriptor_count;
    /// Generic attributes (vertex inputs or varyings) read and written.
    uint32_t loaded_generics;
    uint32_t stored_generics;
    /// Every attribute written (Shader IR::Attribute bit mask), for the next
    /// stage's `previous_stores`.
    uint64_t stores[8];
    /// Fragment color outputs written (bit per render target).
    uint32_t frag_colors;
    /// Bytes of each guest constant buffer the program reads.
    uint32_t cbuf_used_sizes[18];
    /// NUL-terminated reason when translation fails.
    char error[256];
};

/// Returns 1 and fills `spirv` on success; 0 with `error` set otherwise.
int echora_shader_translate(const EchoraShaderInput* input, EchoraShaderOutput* output);
void echora_shader_free(EchoraShaderOutput* output);
/// sizeof(EchoraShaderOutput) and sizeof(EchoraShaderInput), so bindings
/// can check their layout.
size_t echora_shader_output_size(void);
size_t echora_shader_input_size(void);

#ifdef __cplusplus
}
#endif
