//! Native NVN programs to SPIR-V through `native/shader` (the Lichii/Eden
//! recompiler behind a C interface). Mirrors `echora_shader.h`.

use std::ffi::c_void;

pub const MAX_DESCRIPTORS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Vertex,
    TessControl,
    TessEval,
    Geometry,
    Fragment,
    Compute,
    /// Switch 2 task and mesh programs (VK_EXT_mesh_shader stages).
    Task,
    Mesh,
}

impl Stage {
    fn from_raw(raw: u32) -> Option<Self> {
        Some(match raw {
            0 => Self::Vertex,
            1 => Self::TessControl,
            2 => Self::TessEval,
            3 => Self::Geometry,
            4 => Self::Fragment,
            5 => Self::Compute,
            7 => Self::Task,
            8 => Self::Mesh,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DescriptorKind {
    UniformBuffer,
    StorageBuffer,
    TextureBuffer,
    ImageBuffer,
    Texture,
    Image,
}

/// One set-0 binding, in the order the SPIR-V backend assigns them.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Descriptor {
    pub kind: u32,
    pub binding: u32,
    pub count: u32,
    /// Guest constant buffer index of a uniform buffer.
    pub index: u32,
    pub cbuf_index: u32,
    pub cbuf_offset: u32,
    pub shift_left: u32,
    pub has_secondary: u32,
    pub secondary_cbuf_index: u32,
    pub secondary_cbuf_offset: u32,
    pub secondary_shift_left: u32,
    pub size_shift: u32,
    pub texture_type: u32,
    /// Bit 0 written, 1 read, 2 integer, 3 depth, 4 multisample.
    pub flags: u32,
}

impl Descriptor {
    pub fn kind(&self) -> DescriptorKind {
        match self.kind {
            0 => DescriptorKind::UniformBuffer,
            1 => DescriptorKind::StorageBuffer,
            2 => DescriptorKind::TextureBuffer,
            3 => DescriptorKind::ImageBuffer,
            4 => DescriptorKind::Texture,
            _ => DescriptorKind::Image,
        }
    }
}

/// Answers the recompiler's questions about state outside the program.
pub trait Environment {
    /// Word at byte `offset` of guest constant buffer `index`.
    fn read_cbuf(&mut self, _index: u32, _offset: u32) -> u32 {
        0
    }
    /// Shader::TextureType of a raw texture handle (2 is Color2D).
    fn texture_type(&mut self, _handle: u32) -> u32 {
        2
    }
}

/// An environment that knows nothing beyond the program.
pub struct NoEnvironment;
impl Environment for NoEnvironment {}

#[derive(Clone, Copy, Debug, Default)]
pub struct Parameters {
    pub workgroup_size: [u32; 3],
    pub shared_memory_size: u32,
    pub local_memory_size: u32,
    /// First set-0 binding (graphics stages after the first in a pipeline).
    pub binding_base: u32,
    /// Attribute stores of the previous graphics stage (`Translated::stores`);
    /// None treats every input as provided (vertex stages).
    pub previous_stores: Option<[u64; 8]>,
    /// Vertex stages: Shader::AttributeType of each generic input (0 float,
    /// 1 signed integer, 2 unsigned integer).
    pub generic_input_types: [u8; 32],
    /// Render target resolution scale the program is translated for, as
    /// (numerator, denominator); (0, 0) or equal terms: none. See
    /// `Renderer::resolution_scale`.
    pub resolution_scale: (u32, u32),
    /// Texture and image descriptors of the stages before this one.
    pub scaling_base: [u32; 2],
    /// Translate a graphics program as a task or mesh program instead of the
    /// stage its header names (the slot NVN binds it in).
    pub mesh: Option<MeshParameters>,
}

/// The task or mesh pipeline state a program is translated against.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MeshParameters {
    /// The task stage (else the mesh stage).
    pub task: bool,
    /// SET_MESH_SHADER_A, SET_MESH_SHADER_B and SET_MESH_INIT_SHADER.
    pub registers: [u32; 3],
    /// Mesh programs: the program header (0x50 bytes) of the pass-through
    /// geometry program bound with them, whose output map lists the
    /// per-primitive outputs.
    pub primitive_header: Option<[u8; 0x50]>,
}

#[derive(Clone, Debug)]
pub struct Translated {
    pub stage: Stage,
    pub spirv: Vec<u32>,
    pub descriptors: Vec<Descriptor>,
    pub loaded_generics: u32,
    pub stored_generics: u32,
    /// Every attribute written, as a Shader IR::Attribute bit mask.
    pub stores: [u64; 8],
    pub frag_colors: u32,
    pub cbuf_used_sizes: [u32; 18],
}

#[repr(C)]
struct Callbacks {
    user: *mut c_void,
    read_cbuf: Option<unsafe extern "C" fn(*mut c_void, u32, u32) -> u32>,
    texture_type: Option<unsafe extern "C" fn(*mut c_void, u32) -> u32>,
}

#[repr(C)]
struct Input {
    program: *const u8,
    length: usize,
    workgroup_size: [u32; 3],
    shared_memory_size: u32,
    local_memory_size: u32,
    binding_base: u32,
    has_previous_stores: u32,
    previous_stores: [u64; 8],
    generic_input_types: [u8; 32],
    resolution_scale_numerator: u32,
    resolution_scale_denominator: u32,
    texture_scaling_base: u32,
    image_scaling_base: u32,
    mesh_stage: u32,
    mesh_registers: [u32; 3],
    has_mesh_primitive_header: u32,
    mesh_primitive_header: [u8; 0x50],
    callbacks: Callbacks,
}

#[repr(C)]
struct Output {
    spirv: *mut u32,
    words: usize,
    stage: u32,
    descriptors: [Descriptor; MAX_DESCRIPTORS],
    descriptor_count: u32,
    loaded_generics: u32,
    stored_generics: u32,
    stores: [u64; 8],
    frag_colors: u32,
    cbuf_used_sizes: [u32; 18],
    error: [u8; 256],
}

#[cfg(echora_shader_lib)]
unsafe extern "C" {
    fn echora_shader_translate(input: *const Input, output: *mut Output) -> i32;
    fn echora_shader_free(output: *mut Output);
    #[cfg_attr(not(test), allow(dead_code))]
    fn echora_shader_output_size() -> usize;
    #[cfg_attr(not(test), allow(dead_code))]
    fn echora_shader_input_size() -> usize;
}

unsafe extern "C" fn read_cbuf_thunk<E: Environment>(
    user: *mut c_void,
    index: u32,
    offset: u32,
) -> u32 {
    unsafe { (*(user as *mut E)).read_cbuf(index, offset) }
}

unsafe extern "C" fn texture_type_thunk<E: Environment>(user: *mut c_void, handle: u32) -> u32 {
    unsafe { (*(user as *mut E)).texture_type(handle) }
}

/// Whether this build links the shader recompiler.
pub const AVAILABLE: bool = cfg!(echora_shader_lib);

/// Translate one program as NVN stores it at its stage address (graphics
/// stages include their 128-byte header).
pub fn translate<E: Environment>(
    program: &[u8],
    compute: Parameters,
    environment: &mut E,
) -> Result<Translated, String> {
    let input = Input {
        program: program.as_ptr(),
        length: program.len(),
        workgroup_size: compute.workgroup_size,
        shared_memory_size: compute.shared_memory_size,
        local_memory_size: compute.local_memory_size,
        binding_base: compute.binding_base,
        has_previous_stores: compute.previous_stores.is_some().into(),
        previous_stores: compute.previous_stores.unwrap_or_default(),
        generic_input_types: compute.generic_input_types,
        resolution_scale_numerator: compute.resolution_scale.0,
        resolution_scale_denominator: compute.resolution_scale.1,
        texture_scaling_base: compute.scaling_base[0],
        image_scaling_base: compute.scaling_base[1],
        mesh_stage: compute.mesh.map_or(0, |mesh| if mesh.task { 1 } else { 2 }),
        mesh_registers: compute.mesh.map_or([0; 3], |mesh| mesh.registers),
        has_mesh_primitive_header: compute
            .mesh
            .is_some_and(|mesh| mesh.primitive_header.is_some())
            .into(),
        mesh_primitive_header: compute
            .mesh
            .and_then(|mesh| mesh.primitive_header)
            .unwrap_or([0; 0x50]),
        callbacks: Callbacks {
            user: environment as *mut E as *mut c_void,
            read_cbuf: Some(read_cbuf_thunk::<E>),
            texture_type: Some(texture_type_thunk::<E>),
        },
    };
    // SAFETY: Output is plain data; all-zero is its documented empty state.
    let mut output: Output = unsafe { std::mem::zeroed() };
    run(&input, &mut output)?;
    // SAFETY: on success the library filled `spirv` with `words` words.
    let spirv = unsafe { std::slice::from_raw_parts(output.spirv, output.words) }.to_vec();
    // Every loop ends within a bound (a runaway guest loop reset the GPU).
    // (ECHORA_NO_LOOP_GUARD, debugging: the translation as it is.)
    let spirv = if std::env::var_os("ECHORA_NO_LOOP_GUARD").is_some() {
        spirv
    } else {
        crate::loop_guard::guard_loops(&spirv)
    };
    let result = Translated {
        stage: Stage::from_raw(output.stage).ok_or("unknown stage")?,
        spirv,
        descriptors: output.descriptors[..output.descriptor_count as usize].to_vec(),
        loaded_generics: output.loaded_generics,
        stored_generics: output.stored_generics,
        stores: output.stores,
        frag_colors: output.frag_colors,
        cbuf_used_sizes: output.cbuf_used_sizes,
    };
    free(&mut output);
    Ok(result)
}

#[cfg(echora_shader_lib)]
fn run(input: &Input, output: &mut Output) -> Result<(), String> {
    // SAFETY: both structs match echora_shader.h; callbacks outlive the call.
    if unsafe { echora_shader_translate(input, output) } != 0 {
        return Ok(());
    }
    let end = output
        .error
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(output.error.len());
    Err(String::from_utf8_lossy(&output.error[..end]).into_owned())
}

#[cfg(not(echora_shader_lib))]
fn run(_input: &Input, _output: &mut Output) -> Result<(), String> {
    Err("built without the shader recompiler (set ECHORA_SHADER_LIB)".into())
}

#[cfg(echora_shader_lib)]
fn free(output: &mut Output) {
    // SAFETY: frees what echora_shader_translate allocated.
    unsafe { echora_shader_free(output) }
}

#[cfg(not(echora_shader_lib))]
fn free(_output: &mut Output) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_layout_matches_the_header() {
        assert_eq!(std::mem::size_of::<Descriptor>(), 56);
        // 8 + 8 + 4 + 64 * 56 + 3 * 4 (+ 4 padding) + 8 * 8 + 4 + 18 * 4 + 256,
        // padded to 8.
        assert_eq!(std::mem::size_of::<Output>(), 4016);
        #[cfg(echora_shader_lib)]
        // SAFETY: a pure function of the C type.
        assert_eq!(
            unsafe { echora_shader_output_size() },
            std::mem::size_of::<Output>()
        );
        #[cfg(echora_shader_lib)]
        // SAFETY: a pure function of the C type.
        assert_eq!(
            unsafe { echora_shader_input_size() },
            std::mem::size_of::<Input>()
        );
    }

    #[test]
    fn a_program_too_short_for_its_header_fails() {
        let header = 0x000c_d482u32.to_le_bytes();
        let result = translate(&header, Parameters::default(), &mut NoEnvironment);
        assert!(result.is_err());
    }
}
