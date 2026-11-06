//! Builds the shader translation library when `ECHORA_SHADER_SRC` names the
//! fetched, patched sources (see `native/shader/README.md`): the PokeDinner
//! shader recompiler, sirit, and Echora's C ABI shim. Without it the crate
//! still builds and shader translation reports that the library is absent.

use std::path::{Path, PathBuf};

fn sources(dir: &Path, skip: &[&str], out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if skip.iter().any(|s| path.ends_with(s)) {
            continue;
        }
        if path.is_dir() {
            sources(&path, skip, out);
        } else if path.extension().is_some_and(|e| e == "cpp") {
            out.push(path);
        }
    }
}

fn main() {
    println!("cargo:rerun-if-env-changed=ECHORA_SHADER_SRC");
    println!("cargo::rustc-check-cfg=cfg(echora_shader_lib)");
    println!("cargo:rerun-if-changed=native/shader");
    let Ok(src) = std::env::var("ECHORA_SHADER_SRC") else {
        return;
    };
    let src = PathBuf::from(src);
    let native = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("manifest dir"))
        .join("native/shader");

    let mut files = Vec::new();
    sources(
        &src.join("pokedinner/src/shader_recompiler"),
        &["backend/glasm", "backend/glsl"],
        &mut files,
    );
    sources(&src.join("sirit/src"), &[], &mut files);
    for file in ["common/settings.cpp", "common/settings_common.cpp"] {
        files.push(src.join("pokedinner/src").join(file));
    }
    files.push(native.join("echora_shader.cpp"));
    files.push(native.join("echora_common.cpp"));

    cc::Build::new()
        .cpp(true)
        .std("c++20")
        .opt_level(2)
        .warnings(false)
        .define("FMT_HEADER_ONLY", None)
        .include(src.join("pokedinner/src"))
        .include(src.join("fmt/include"))
        .include(src.join("sirit/include"))
        .include(src.join("sirit/src"))
        .include(src.join("SPIRV-Headers/include"))
        .include(src.join("boost-include"))
        .include(&native)
        .files(&files)
        .compile("echora_shader");
    println!("cargo:rustc-cfg=echora_shader_lib");
}
