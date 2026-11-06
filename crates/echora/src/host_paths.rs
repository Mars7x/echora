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
    } else if cfg!(target_os = "macos") {
        env("HOME").map(|h| h.join("Library/Caches/Echora"))
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

/// `ECHORA_SETTINGS`, else `settings.toml` in the platform's per-user
/// configuration directory.
pub fn settings_file() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("ECHORA_SETTINGS") {
        return Ok(PathBuf::from(path));
    }
    let env = |k| std::env::var_os(k).map(PathBuf::from);
    let dir = if cfg!(target_os = "windows") {
        env("APPDATA").map(|d| d.join("Echora"))
    } else if cfg!(target_os = "macos") {
        env("HOME").map(|h| h.join("Library/Application Support/Echora"))
    } else {
        env("XDG_CONFIG_HOME")
            .or_else(|| env("HOME").map(|h| h.join(".config")))
            .map(|d| d.join("echora"))
    };
    match dir {
        Some(d) => Ok(d.join("settings.toml")),
        None => bail!("no configuration directory could be determined; set ECHORA_SETTINGS"),
    }
}
