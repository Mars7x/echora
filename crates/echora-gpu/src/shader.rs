//! Native NVN programs to SPIR-V: the Rust translator (`echora-shader`),
//! and an external translator plugin loaded at run time when one is
//! installed (`docs/shader-plugin.md` describes its C interface, which the
//! `Input` and `Output` structs here mirror).

use std::ffi::c_void;

pub use echora_shader_abi::{
    ABI_VERSION, Descriptor, DescriptorKind, PUSH_TRANSLATOR, Stage, Translated,
};

pub const MAX_DESCRIPTORS: usize = 64;

/// Which translator turns a program into SPIR-V (ECHORA_SHADER_TRANSLATOR).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Translator {
    /// The external translator plugin (the default when one is installed).
    Plugin,
    /// The Rust translator (`echora-shader`) for each program it handles,
    /// the plugin for the rest (the default without a plugin).
    Rust,
    /// Both: the plugin's program is used; what the Rust translator does
    /// differently is reported (`compare`).
    Compare,
}

impl Translator {
    /// `plugin`, `rust` or `compare` in ECHORA_SHADER_TRANSLATOR.
    pub fn from_env() -> Self {
        match std::env::var("ECHORA_SHADER_TRANSLATOR").as_deref() {
            Ok("rust") => Self::Rust,
            Ok("compare") => Self::Compare,
            Ok("plugin") => Self::Plugin,
            _ if available() => Self::Plugin,
            _ => Self::Rust,
        }
    }
}

/// The ahead-of-time parameters the Rust translator takes, from the
/// plugin's (which also carry pipeline state it gets at run time).
pub fn abi_parameters(parameters: &Parameters) -> echora_shader_abi::Parameters {
    let compute = parameters.workgroup_size != [0; 3];
    echora_shader_abi::Parameters {
        stage_slot: parameters
            .mesh
            .map(|mesh| if mesh.task { Stage::Task } else { Stage::Mesh }),
        workgroup_size: compute.then_some(parameters.workgroup_size),
        shared_memory_size: compute.then_some(parameters.shared_memory_size),
        local_memory_size: compute.then_some(parameters.local_memory_size),
        cooperative_matrix: COOPERATIVE_MATRIX.load(std::sync::atomic::Ordering::Relaxed),
    }
}

/// Whether HMMA translates to cooperative matrices: set when the device
/// has the shapes (`Gpu::new`) and ECHORA_HMMA=cooperative.
static COOPERATIVE_MATRIX: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Lets the Rust translator use cooperative matrices for HMMA when the
/// device multiplies 16x16x16 half matrices at subgroup scope.
pub fn set_cooperative_matrix(shapes: &[crate::vulkan::CooperativeMatrixShape]) {
    use ash::vk::{ComponentTypeKHR as T, ScopeKHR};
    let has = |result: T| {
        shapes.iter().any(|shape| {
            (shape.m, shape.n, shape.k) == (16, 16, 16)
                && (shape.a, shape.b) == (T::FLOAT16, T::FLOAT16)
                && (shape.c, shape.result) == (result, result)
                && shape.scope == ScopeKHR::SUBGROUP
        })
    };
    // Opt-in (ECHORA_HMMA=cooperative): the first in-game run of these
    // programs on RADV 25.2 under 32-lane subgroups lost the device.
    let on = has(T::FLOAT16)
        && has(T::FLOAT32)
        && std::env::var("ECHORA_HMMA").as_deref() == Ok("cooperative");
    COOPERATIVE_MATRIX.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// The Rust translator's program, if it handles `program`.
pub fn translate_rust(
    program: &[u8],
    parameters: &Parameters,
) -> Result<Translated, echora_shader_abi::Unsupported> {
    let translated = echora_shader::translate(program, &abi_parameters(parameters))?;
    if translated.abi_version != ABI_VERSION {
        return Err(echora_shader_abi::Unsupported {
            reason: format!(
                "made for shader ABI {}, not {ABI_VERSION}",
                translated.abi_version
            ),
            offset: None,
        });
    }
    Ok(translated)
}

/// How two translations of one program differ, in the respects the
/// runtime relies on (None: alike). The SPIR-V itself may differ freely.
pub fn compare(plugin: &Translated, rust: &Translated) -> Option<String> {
    if plugin.stage != rust.stage {
        return Some(format!("stage {:?} vs {:?}", plugin.stage, rust.stage));
    }
    let kinds = |translated: &Translated| -> Vec<DescriptorKind> {
        translated
            .descriptors
            .iter()
            .map(Descriptor::kind)
            .collect()
    };
    if kinds(plugin) != kinds(rust) {
        return Some(format!(
            "descriptors {:?} vs {:?}",
            kinds(plugin),
            kinds(rust)
        ));
    }
    if plugin.frag_colors != rust.frag_colors {
        return Some(format!(
            "fragment colors {:#x} vs {:#x}",
            plugin.frag_colors, rust.frag_colors
        ));
    }
    None
}

/// Answers the plugin's questions about state outside the program.
pub trait Environment {
    /// Word at byte `offset` of guest constant buffer `index`.
    fn read_cbuf(&mut self, _index: u32, _offset: u32) -> u32 {
        0
    }
    /// Shader::TextureType of a raw texture handle (2 is Color2D).
    fn texture_type(&mut self, _handle: u32) -> u32 {
        2
    }
    /// Programs with raw pointers: whether the eight bytes at byte
    /// `offset` of constant buffer `index` are a mapped GPU address (a
    /// candidate the program may read through).
    fn is_storage_pointer(&mut self, _index: u32, _offset: u32) -> bool {
        true
    }
    /// Whether the eight bytes `indirect` bytes past that pointer are one.
    fn is_indirect_storage_pointer(&mut self, _index: u32, _offset: u32, _indirect: u32) -> bool {
        true
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
    /// Read storage buffer pointers as raw 8-byte arguments from the start
    /// (see `Descriptor::flags` bit 5), for a program whose NVN-tracked
    /// pointer words are not addresses.
    pub native_pointers: bool,
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

#[repr(C)]
struct Callbacks {
    user: *mut c_void,
    read_cbuf: Option<unsafe extern "C" fn(*mut c_void, u32, u32) -> u32>,
    texture_type: Option<unsafe extern "C" fn(*mut c_void, u32) -> u32>,
    is_storage_pointer: Option<unsafe extern "C" fn(*mut c_void, u32, u32) -> u32>,
    is_indirect_storage_pointer: Option<unsafe extern "C" fn(*mut c_void, u32, u32, u32) -> u32>,
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
    native_pointers: u32,
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

unsafe extern "C" fn is_storage_pointer_thunk<E: Environment>(
    user: *mut c_void,
    index: u32,
    offset: u32,
) -> u32 {
    unsafe { (*(user as *mut E)).is_storage_pointer(index, offset) }.into()
}

unsafe extern "C" fn is_indirect_storage_pointer_thunk<E: Environment>(
    user: *mut c_void,
    index: u32,
    offset: u32,
    indirect: u32,
) -> u32 {
    unsafe { (*(user as *mut E)).is_indirect_storage_pointer(index, offset, indirect) }.into()
}

/// The external translator plugin's entry points.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
struct Plugin {
    translate: unsafe extern "C" fn(*const Input, *mut Output) -> i32,
    free: unsafe extern "C" fn(*mut Output),
}

/// The plugin: `ECHORA_SHADER_PLUGIN`, else `shader-plugin.so` in Echora's
/// cache directory, loaded once. One built for another `ABI_VERSION`, or
/// whose input and output structs differ in size, is not used.
fn plugin() -> Option<&'static Plugin> {
    static PLUGIN: std::sync::OnceLock<Option<Plugin>> = std::sync::OnceLock::new();
    PLUGIN
        .get_or_init(|| {
            let path = std::env::var_os("ECHORA_SHADER_PLUGIN")
                .map(std::path::PathBuf::from)
                .or_else(|| {
                    let cache = std::env::var_os("XDG_CACHE_HOME")
                        .map(std::path::PathBuf::from)
                        .or_else(|| {
                            std::env::var_os("HOME")
                                .map(|home| std::path::Path::new(&home).join(".cache"))
                        })?;
                    Some(cache.join("echora/shader-plugin.so"))
                })?;
            match load_plugin(&path) {
                Ok(plugin) => Some(plugin),
                Err(error) => {
                    if std::env::var_os("ECHORA_SHADER_PLUGIN").is_some() {
                        eprintln!("shader plugin {}: {error}", path.display());
                    }
                    None
                }
            }
        })
        .as_ref()
}

#[cfg(target_os = "linux")]
fn load_plugin(path: &std::path::Path) -> Result<Plugin, String> {
    use std::os::unix::ffi::OsStrExt;
    let name =
        std::ffi::CString::new(path.as_os_str().as_bytes()).map_err(|error| error.to_string())?;
    // SAFETY: a valid C string; the library stays loaded for the process.
    let library = unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
    if library.is_null() {
        return Err("not loadable".into());
    }
    let symbol = |name: &std::ffi::CStr| {
        // SAFETY: a live library handle and a valid C string.
        let address = unsafe { libc::dlsym(library, name.as_ptr()) };
        (!address.is_null())
            .then_some(address)
            .ok_or_else(|| format!("no {}", name.to_string_lossy()))
    };
    // SAFETY: each symbol has the plugin interface's signature.
    unsafe {
        let version: extern "C" fn() -> u32 =
            std::mem::transmute(symbol(c"echora_plugin_abi_version")?);
        let input_size: extern "C" fn() -> usize =
            std::mem::transmute(symbol(c"echora_plugin_input_size")?);
        let output_size: extern "C" fn() -> usize =
            std::mem::transmute(symbol(c"echora_plugin_output_size")?);
        if version() != ABI_VERSION {
            return Err(format!(
                "made for shader ABI {}, not {ABI_VERSION}",
                version()
            ));
        }
        if input_size() != std::mem::size_of::<Input>()
            || output_size() != std::mem::size_of::<Output>()
        {
            return Err("its input or output layout differs".into());
        }
        Ok(Plugin {
            translate: std::mem::transmute::<
                *mut c_void,
                unsafe extern "C" fn(*const Input, *mut Output) -> i32,
            >(symbol(c"echora_plugin_translate")?),
            free: std::mem::transmute::<*mut c_void, unsafe extern "C" fn(*mut Output)>(symbol(
                c"echora_plugin_free",
            )?),
        })
    }
}

#[cfg(not(target_os = "linux"))]
fn load_plugin(_path: &std::path::Path) -> Result<Plugin, String> {
    Err("translator plugins load on Linux only".into())
}

/// Whether an external translator plugin is installed.
pub fn available() -> bool {
    plugin().is_some()
}

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
            is_storage_pointer: Some(is_storage_pointer_thunk::<E>),
            is_indirect_storage_pointer: Some(is_indirect_storage_pointer_thunk::<E>),
        },
        native_pointers: compute.native_pointers.into(),
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
        abi_version: ABI_VERSION,
        // The plugin takes pipeline state as translation parameters and
        // may push texture rescaling words.
        spec_constants: Vec::new(),
        push_fields: PUSH_TRANSLATOR,
        texture_pool_bindings: 0,
    };
    free(&mut output);
    Ok(result)
}

fn run(input: &Input, output: &mut Output) -> Result<(), String> {
    let plugin = plugin().ok_or("no shader translator plugin is installed")?;
    // SAFETY: both structs match the plugin interface (sizes checked at
    // load); callbacks outlive the call.
    if unsafe { (plugin.translate)(input, output) } != 0 {
        return Ok(());
    }
    let end = output
        .error
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(output.error.len());
    Err(String::from_utf8_lossy(&output.error[..end]).into_owned())
}

fn free(output: &mut Output) {
    if let Some(plugin) = plugin() {
        // SAFETY: frees what the plugin's translate allocated.
        unsafe { (plugin.free)(output) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_layout_matches_the_header() {
        assert_eq!(std::mem::size_of::<Descriptor>(), 60);
        // 8 + 8 + 4 + 64 * 60 + 3 * 4 + 8 * 8 + 4 + 18 * 4 + 256, padded
        // to 8.
        assert_eq!(std::mem::size_of::<Output>(), 4272);
    }

    #[test]
    fn a_program_too_short_for_its_header_fails() {
        let header = 0x000c_d482u32.to_le_bytes();
        let result = translate(&header, Parameters::default(), &mut NoEnvironment);
        assert!(result.is_err());
    }
}
