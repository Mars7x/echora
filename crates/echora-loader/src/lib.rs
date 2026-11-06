//! Loader for Switch executable modules.
//!
//! Loading parses, decompresses and verifies modules; it never executes guest
//! code or applies relocations (the guest's own rtld does that at startup).

pub mod mod0;
pub mod nso;
pub mod zbic;

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

pub use mod0::{DynamicInfo, Mod0Info, defined_symbol};
pub use nso::{Compression, NsoModule, Segment};

/// Combined mapped size of all modules of one process.
pub const MAX_EXEFS_IMAGE_SIZE: u64 = 1 << 30;

#[derive(Debug, thiserror::Error)]
pub enum LoaderError {
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("{0}")]
    Format(String),
    #[error("integrity check failed: {0}")]
    Integrity(String),
}

pub type Result<T> = std::result::Result<T, LoaderError>;

pub(crate) fn hex_serde<S: serde::Serializer>(
    v: &[u8; 32],
    s: S,
) -> std::result::Result<S::Ok, S::Error> {
    s.serialize_str(&hex::encode(v))
}

fn io_at(path: &Path) -> impl FnOnce(std::io::Error) -> LoaderError + '_ {
    move |source| LoaderError::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// Reads a regular file (not a symlink) of at most `limit` bytes.
fn read_regular_file(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let md = fs::symlink_metadata(path).map_err(io_at(path))?;
    if !md.is_file() {
        return Err(LoaderError::Format(format!(
            "{} must be a regular file",
            path.display()
        )));
    }
    if md.len() > limit {
        return Err(LoaderError::Format(format!(
            "{} exceeds the {limit}-byte input limit",
            path.display()
        )));
    }
    let mut out = Vec::with_capacity(md.len() as usize);
    fs::File::open(path)
        .map_err(io_at(path))?
        .take(limit)
        .read_to_end(&mut out)
        .map_err(io_at(path))?;
    Ok(out)
}

/// Loads one NSO file.
pub fn load_nso(name: &str, path: &Path) -> Result<NsoModule> {
    let bytes = read_regular_file(path, nso::MAX_NSO_FILE_SIZE)?;
    nso::parse(name, &bytes, nso::MAX_IMAGE_SIZE)
}

/// Module names of an ExeFS in process load order: `rtld`, `main`, present
/// `subsdk0`-`subsdk9`, then `sdk`. `rtld`, `main` and `sdk` are required,
/// and an unrecognized numbered module is refused rather than left out.
pub fn exefs_load_order(dir: &Path) -> Result<Vec<String>> {
    let md = fs::symlink_metadata(dir).map_err(io_at(dir))?;
    if !md.is_dir() {
        return Err(LoaderError::Format(format!(
            "ExeFS {} must be a directory",
            dir.display()
        )));
    }
    let mut subsdk = [false; 10];
    for entry in fs::read_dir(dir).map_err(io_at(dir))? {
        let entry = entry.map_err(io_at(dir))?;
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if let Some(suffix) = name.strip_prefix("subsdk")
            && !suffix.is_empty()
            && suffix.bytes().all(|b| b.is_ascii_digit())
        {
            match suffix.parse::<usize>() {
                Ok(i) if suffix.len() == 1 && i < 10 => subsdk[i] = true,
                _ => {
                    return Err(LoaderError::Format(format!(
                        "unsupported ExeFS module {name}"
                    )));
                }
            }
        }
    }
    for required in ["rtld", "main", "sdk"] {
        let p = dir.join(required);
        if !fs::symlink_metadata(&p).is_ok_and(|m| m.is_file()) {
            return Err(LoaderError::Format(format!(
                "required ExeFS module {required} is missing or not a regular file"
            )));
        }
    }
    let mut names = vec!["rtld".to_owned(), "main".to_owned()];
    names.extend((0..10).filter(|&i| subsdk[i]).map(|i| format!("subsdk{i}")));
    names.push("sdk".to_owned());
    Ok(names)
}

/// Loads every module of an ExeFS directory in load order, within a combined
/// image budget.
pub fn load_exefs(dir: &Path) -> Result<Vec<NsoModule>> {
    let mut remaining = MAX_EXEFS_IMAGE_SIZE;
    let mut out = Vec::new();
    for name in exefs_load_order(dir)? {
        let path = dir.join(&name);
        let bytes = read_regular_file(&path, nso::MAX_NSO_FILE_SIZE)?;
        let module = nso::parse(&name, &bytes, remaining.min(nso::MAX_IMAGE_SIZE))?;
        remaining = remaining
            .checked_sub(module.image_size)
            .ok_or_else(|| LoaderError::Format("combined ExeFS image exceeds the budget".into()))?;
        out.push(module);
    }
    Ok(out)
}
