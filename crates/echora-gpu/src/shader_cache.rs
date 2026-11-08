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
                .map(|(info, binary)| info.code_type(vk::ShaderCodeTypeEXT::BINARY).code(binary))
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
        let binaries: Result<Vec<Vec<u8>>, _> = shaders
            .iter()
            // SAFETY: the shaders were just created.
            .map(|shader| unsafe { objects.get_shader_binary_data(*shader) })
            .collect();
        if let Ok(binaries) = binaries {
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

fn read(path: &Path, binary_version: ([u8; 16], u32)) -> Option<Vec<Vec<u8>>> {
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
        let size = u64::from_le_bytes(take(8)?.try_into().ok()?);
        binaries.push(take(usize::try_from(size).ok()?)?.to_vec());
    }
    rest.is_empty().then_some(binaries)
}

fn write(
    path: &Path,
    binary_version: ([u8; 16], u32),
    binaries: &[Vec<u8>],
) -> std::io::Result<()> {
    let mut bytes = header(binary_version);
    bytes.extend_from_slice(&(binaries.len() as u32).to_le_bytes());
    for binary in binaries {
        bytes.extend_from_slice(&(binary.len() as u64).to_le_bytes());
        bytes.extend_from_slice(binary);
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
        let binaries = vec![vec![1, 2, 3], Vec::new(), vec![9; 40]];
        write(&path, version, &binaries).unwrap();
        assert_eq!(read(&path, version), Some(binaries));
        assert_eq!(read(&path, ([7; 16], 4)), None);
        assert_eq!(read(&path, ([8; 16], 3)), None);
        // A truncated file is not a cache entry.
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[..bytes.len() - 1]).unwrap();
        assert_eq!(read(&path, version), None);
        std::fs::remove_dir_all(&directory).unwrap();
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
