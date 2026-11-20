//! Driver binaries of linked shader objects, kept between runs.
//!
//! Drivers do not cache VK_EXT_shader_object compiles the way they cache
//! pipelines, so every run would compile each translated program again
//! (on RADV, NIR and ACO passes take a third of the main thread while a
//! scene first draws). After compiling SPIR-V the binaries are read back
//! with vkGetShaderBinaryDataEXT and written to a file named by a hash of
//! everything that went into the compile; a later run creates the shaders
//! from that file. A binary the driver no longer accepts (its
//! shaderBinaryUUID or version changed, or the file is damaged) is
//! compiled again from the SPIR-V and replaced.

use ash::vk;
use std::hash::{Hash, Hasher};
use std::path::Path;

const MAGIC: &[u8; 4] = b"ECSB";
const FORMAT: u32 = 1;

/// Vulkan requires 16-byte alignment when exporting or loading a binary.
/// A byte vector alone only guarantees alignment of one.
struct ShaderBinary {
    storage: Vec<u8>,
    start: usize,
    length: usize,
}

impl ShaderBinary {
    fn zeroed(length: usize) -> Option<Self> {
        if length == 0 {
            return None;
        }
        const ALIGNMENT: usize = 16;
        let capacity = length.checked_add(ALIGNMENT - 1)?;
        let mut storage = Vec::new();
        storage.try_reserve_exact(capacity).ok()?;
        storage.resize(capacity, 0);
        let start = storage.as_ptr().align_offset(ALIGNMENT);
        Some(Self {
            storage,
            start,
            length,
        })
    }

    fn bytes(&self) -> &[u8] {
        &self.storage[self.start..self.start + self.length]
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.storage[self.start..self.start + self.length]
    }
}

fn get_binary(
    objects: &ash::ext::shader_object::Device,
    shader: vk::ShaderEXT,
) -> Option<ShaderBinary> {
    let query = objects.fp().get_shader_binary_data_ext;
    let mut size = 0;
    // SAFETY: shader belongs to this live device. A null data pointer
    // queries the exact size of its immutable binary.
    let status = unsafe { query(objects.device(), shader, &mut size, std::ptr::null_mut()) };
    if status != vk::Result::SUCCESS {
        return None;
    }
    let mut binary = ShaderBinary::zeroed(size)?;
    // SAFETY: the shader remains live and the data pointer has size bytes
    // of writable storage aligned to 16 bytes.
    let status = unsafe {
        query(
            objects.device(),
            shader,
            &mut size,
            binary.bytes_mut().as_mut_ptr().cast(),
        )
    };
    (status == vk::Result::SUCCESS && size == binary.length).then_some(binary)
}

/// Create the shaders of `infos` (SPIR-V), from cached binaries in
/// `directory` when it holds them. `layout` describes the set layouts,
/// push constant ranges and any other create state outside `infos`.
pub(crate) fn create_shaders(
    objects: &ash::ext::shader_object::Device,
    directory: Option<&Path>,
    binary_version: ([u8; 16], u32),
    infos: &[vk::ShaderCreateInfoEXT],
    layout: &str,
) -> Result<Vec<vk::ShaderEXT>, String> {
    let path = directory
        .map(|directory| directory.join(format!("{}.bin", key(binary_version, infos, layout))));
    if let Some(binaries) = path.as_deref().and_then(|path| read(path, binary_version)) {
        if binaries.len() == infos.len() {
            let binary_infos: Vec<_> = infos
                .iter()
                .zip(&binaries)
                .map(|(info, binary)| {
                    info.code_type(vk::ShaderCodeTypeEXT::BINARY)
                        .code(binary.bytes())
                })
                .collect();
            // SAFETY: the binaries and the layouts in `infos` outlive the call.
            match unsafe { objects.create_shaders(&binary_infos, None) } {
                Ok(shaders) => return Ok(shaders),
                Err((partial, _)) => destroy(objects, &partial),
            }
        }
    }
    // SAFETY: the code and layouts outlive the call.
    let shaders = unsafe { objects.create_shaders(infos, None) }.map_err(|(partial, error)| {
        destroy(objects, &partial);
        format!("vkCreateShadersEXT: {error}")
    })?;
    if let Some(path) = path {
        let binaries: Option<Vec<_>> = shaders
            .iter()
            .map(|shader| get_binary(objects, *shader))
            .collect();
        if let Some(binaries) = binaries {
            // A cache that cannot be written only costs the next run a compile.
            let _ = write(&path, binary_version, &binaries);
        }
    }
    Ok(shaders)
}

fn destroy(objects: &ash::ext::shader_object::Device, shaders: &[vk::ShaderEXT]) {
    for shader in shaders
        .iter()
        .filter(|shader| **shader != vk::ShaderEXT::null())
    {
        // SAFETY: created above and not used.
        unsafe { objects.destroy_shader(*shader, None) };
    }
}

/// 128 bits (two differently seeded hashes) of the driver binary version,
/// each stage's create state and code, and `layout`.
fn key(binary_version: ([u8; 16], u32), infos: &[vk::ShaderCreateInfoEXT], layout: &str) -> String {
    let hash = |seed: u64| {
        let mut hasher = std::hash::DefaultHasher::new();
        seed.hash(&mut hasher);
        binary_version.hash(&mut hasher);
        layout.hash(&mut hasher);
        for info in infos {
            (
                info.flags.as_raw(),
                info.stage.as_raw(),
                info.next_stage.as_raw(),
            )
                .hash(&mut hasher);
            (info.code_type.as_raw(), info.p_next.is_null()).hash(&mut hasher);
            // SAFETY: `code` points at `code_size` bytes for the call's duration.
            let code =
                unsafe { std::slice::from_raw_parts(info.p_code.cast::<u8>(), info.code_size) };
            code.hash(&mut hasher);
        }
        hasher.finish()
    };
    format!("{:016x}{:016x}", hash(0), hash(1))
}

fn read(path: &Path, binary_version: ([u8; 16], u32)) -> Option<Vec<ShaderBinary>> {
    let bytes = std::fs::read(path).ok()?;
    let mut rest = bytes.strip_prefix(header(binary_version).as_slice())?;
    let mut take = |count: usize| {
        let (taken, after) = rest.split_at_checked(count)?;
        rest = after;
        Some(taken)
    };
    let stages = u32::from_le_bytes(take(4)?.try_into().ok()?);
    let mut binaries = Vec::new();
    for _ in 0..stages {
        let size = usize::try_from(u64::from_le_bytes(take(8)?.try_into().ok()?)).ok()?;
        let encoded = take(size)?;
        let mut binary = ShaderBinary::zeroed(size)?;
        binary.bytes_mut().copy_from_slice(encoded);
        binaries.push(binary);
    }
    rest.is_empty().then_some(binaries)
}

fn write(
    path: &Path,
    binary_version: ([u8; 16], u32),
    binaries: &[ShaderBinary],
) -> std::io::Result<()> {
    let mut bytes = header(binary_version);
    bytes.extend_from_slice(&(binaries.len() as u32).to_le_bytes());
    for binary in binaries {
        bytes.extend_from_slice(&(binary.length as u64).to_le_bytes());
        bytes.extend_from_slice(binary.bytes());
    }
    if let Some(directory) = path.parent() {
        std::fs::create_dir_all(directory)?;
    }
    // Written beside and renamed over, so a reader never sees half a file.
    let partial = path.with_extension(format!("{}.part", std::process::id()));
    std::fs::write(&partial, &bytes)?;
    std::fs::rename(&partial, path)
}

fn header(binary_version: ([u8; 16], u32)) -> Vec<u8> {
    let mut bytes = MAGIC.to_vec();
    bytes.extend_from_slice(&FORMAT.to_le_bytes());
    bytes.extend_from_slice(&binary_version.0);
    bytes.extend_from_slice(&binary_version.1.to_le_bytes());
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_binaries_read_back_only_for_their_driver_version() {
        let directory =
            std::env::temp_dir().join(format!("echora-shader-cache-{}", std::process::id()));
        let path = directory.join("program.bin");
        let version = ([7; 16], 3);
        let expected = [vec![1, 2, 3], vec![4], vec![9; 40]];
        let binaries: Vec<_> = expected
            .iter()
            .map(|bytes| {
                let mut binary = ShaderBinary::zeroed(bytes.len()).unwrap();
                binary.bytes_mut().copy_from_slice(bytes);
                binary
            })
            .collect();
        write(&path, version, &binaries).unwrap();
        let decoded = read(&path, version).unwrap();
        assert_eq!(decoded.len(), expected.len());
        for (binary, expected) in decoded.iter().zip(expected) {
            assert_eq!(binary.bytes(), expected);
            assert!(binary.bytes().as_ptr().addr().is_multiple_of(16));
        }
        assert!(read(&path, ([7; 16], 4)).is_none());
        assert!(read(&path, ([8; 16], 3)).is_none());
        // A truncated file is not a cache entry.
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[..bytes.len() - 1]).unwrap();
        assert!(read(&path, version).is_none());
        std::fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn compiled_shader_binary_can_be_reloaded() {
        let Ok(gpu) = crate::vulkan::Gpu::new() else {
            return;
        };
        let Ok(objects) = gpu.shader_objects() else {
            return;
        };
        let code = crate::draw::tests::spirv(
            "@compute @workgroup_size(1) fn main() {}",
            naga::ShaderStage::Compute,
        );
        let info = vk::ShaderCreateInfoEXT::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .code_type(vk::ShaderCodeTypeEXT::SPIRV)
            .code(crate::draw::words_as_bytes(&code))
            .name(c"main");
        let shaders =
            create_shaders(objects, None, gpu.shader_binary_version, &[info], "").unwrap();
        let binary = get_binary(objects, shaders[0]);
        destroy(objects, &shaders);
        let binary = binary.unwrap();
        let info = info
            .code_type(vk::ShaderCodeTypeEXT::BINARY)
            .code(binary.bytes());
        // SAFETY: the shader binary came from this device and its aligned
        // storage and layout information remain live for the call.
        let shaders = unsafe { objects.create_shaders(&[info], None) }.unwrap();
        destroy(objects, &shaders);
    }

    #[test]
    fn keys_change_with_code_layout_and_driver() {
        let code = [0x0723_0203u32, 1, 2, 3];
        let other = [0x0723_0203u32, 1, 2, 4];
        fn info(code: &[u32]) -> vk::ShaderCreateInfoEXT<'_> {
            vk::ShaderCreateInfoEXT::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .code_type(vk::ShaderCodeTypeEXT::SPIRV)
                .code(crate::draw::words_as_bytes(code))
        }
        let version = ([1; 16], 1);
        let base = key(version, &[info(&code)], "layout");
        assert_eq!(base, key(version, &[info(&code)], "layout"));
        assert_ne!(base, key(version, &[info(&other)], "layout"));
        assert_ne!(base, key(version, &[info(&code)], "other layout"));
        assert_ne!(base, key(([1; 16], 2), &[info(&code)], "layout"));
    }
}
