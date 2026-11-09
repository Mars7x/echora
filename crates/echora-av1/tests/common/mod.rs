//! Test streams: the small AOM conformance vectors checked in under
//! tests/vectors, and optionally more from ECHORA_AV1_VECTORS.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

fn read_dir_ivf(dir: &Path) -> Vec<(String, Vec<u8>)> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.expect("directory entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "ivf"))
        .collect();
    files.sort();
    files
        .into_iter()
        .map(|p| {
            let name = p
                .file_name()
                .expect("file name")
                .to_string_lossy()
                .into_owned();
            (name, std::fs::read(&p).expect("readable vector"))
        })
        .collect()
}

pub fn checked_in_vectors() -> Vec<(String, Vec<u8>)> {
    read_dir_ivf(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/vectors"))
}

/// The vectors in the directory ECHORA_AV1_VECTORS names, if set.
pub fn extra_vectors() -> Vec<(String, Vec<u8>)> {
    match std::env::var_os("ECHORA_AV1_VECTORS") {
        Some(dir) => read_dir_ivf(Path::new(&dir)),
        None => Vec::new(),
    }
}
