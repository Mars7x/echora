//! Host-specific locations for Echora's caches. Generated artifacts never go
//! into the source tree.

use std::path::PathBuf;

use anyhow::{Result, bail};

/// `ECHORA_CACHE_DIR`, else the platform's per-user cache directory.
pub fn cache_dir() -> Result<PathBuf> {
    if let Some(dir) = std::env::var_os("ECHORA_CACHE_DIR") {
        return Ok(PathBuf::from(dir));
    }
    let env = |k| std::env::var_os(k).map(PathBuf::from);
    let dir = if cfg!(target_os = "windows") {
        env("LOCALAPPDATA").map(|d| d.join("Echora").join("cache"))
    } else {
        env("XDG_CACHE_HOME")
            .or_else(|| env("HOME").map(|h| h.join(".cache")))
            .map(|d| d.join("echora"))
    };
    match dir {
        Some(d) => Ok(d),
        None => bail!("no cache directory could be determined; set ECHORA_CACHE_DIR"),
    }
}

/// `ECHORA_DATA_DIR`, else the platform's per-user data directory: where
/// game saves live.
pub fn data_dir() -> Result<PathBuf> {
    if let Some(dir) = std::env::var_os("ECHORA_DATA_DIR") {
        return Ok(PathBuf::from(dir));
    }
    let env = |k| std::env::var_os(k).map(PathBuf::from);
    let dir = if cfg!(target_os = "windows") {
        env("APPDATA").map(|d| d.join("Echora"))
    } else {
        env("XDG_DATA_HOME")
            .or_else(|| env("HOME").map(|h| h.join(".local/share")))
            .map(|d| d.join("echora"))
    };
    match dir {
        Some(d) => Ok(d),
        None => bail!("no data directory could be determined; set ECHORA_DATA_DIR"),
    }
}

/// The settings file (see `echora_config::default_path`).
pub fn settings_file() -> Result<PathBuf> {
    match echora_config::default_path() {
        Some(path) => Ok(path),
        None => bail!("no configuration directory could be determined; set ECHORA_SETTINGS"),
    }
}
