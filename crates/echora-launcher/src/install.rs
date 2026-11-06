//! Installing the game into Echora's own data folder, so the NUP or dump
//! the player chose is no longer needed afterwards.
//!
//! The copy is staged next to the install and swapped in only once it is
//! complete, so a failed or interrupted install leaves the previous one in
//! place.

use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// Where installed games live: `ECHORA_DATA_DIR`, else the platform's
/// per-user data folder.
pub fn data_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("ECHORA_DATA_DIR") {
        return Some(PathBuf::from(dir));
    }
    let env = |key| std::env::var_os(key).map(PathBuf::from);
    if cfg!(target_os = "windows") {
        env("LOCALAPPDATA").map(|dir| dir.join("Echora"))
    } else if cfg!(target_os = "macos") {
        env("HOME").map(|home| home.join("Library/Application Support/Echora"))
    } else {
        env("XDG_DATA_HOME")
            .or_else(|| env("HOME").map(|home| home.join(".local/share")))
            .map(|dir| dir.join("echora"))
    }
}

/// Progress of an install running on another thread.
#[derive(Default)]
pub struct Progress {
    copied: AtomicU64,
    total: AtomicU64,
    result: Mutex<Option<Result<PathBuf, String>>>,
}

impl Progress {
    /// Fraction copied, from 0 to 1.
    pub fn fraction(&self) -> f32 {
        let total = self.total.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        self.copied.load(Ordering::Relaxed) as f32 / total as f32
    }

    /// The installed path or the error, once the install has finished.
    pub fn take_result(&self) -> Option<Result<PathBuf, String>> {
        self.result.lock().ok()?.take()
    }

    fn finish(&self, result: Result<PathBuf, String>) {
        if let Ok(mut slot) = self.result.lock() {
            *slot = Some(result);
        }
    }
}

/// Copies `source` (a NUP file or a dump folder) into `games`, replacing any
/// earlier install, and reports the installed path through `progress`.
pub fn install(source: &Path, games: &Path, progress: &Progress) {
    let result = install_inner(source, games, progress).map_err(|error| error.to_string());
    progress.finish(result);
}

fn install_inner(source: &Path, games: &Path, progress: &Progress) -> io::Result<PathBuf> {
    let name = if source.is_dir() { "dump" } else { "game.nup" };
    let target = games.join(name);
    if source.canonicalize().ok() == target.canonicalize().ok() {
        return Ok(target);
    }
    fs::create_dir_all(games)?;
    progress.total.store(size_of(source)?, Ordering::Relaxed);
    let staging = games.join(".installing");
    remove(&staging)?;
    copy(source, &staging, progress)?;
    // Only one game is installed at a time: clear both kinds of install.
    remove(&games.join("dump"))?;
    remove(&games.join("game.nup"))?;
    fs::rename(&staging, &target)?;
    Ok(target)
}

fn size_of(path: &Path) -> io::Result<u64> {
    let meta = fs::metadata(path)?;
    if !meta.is_dir() {
        return Ok(meta.len());
    }
    fs::read_dir(path)?.try_fold(0, |sum, entry| Ok(sum + size_of(&entry?.path())?))
}

fn copy(source: &Path, target: &Path, progress: &Progress) -> io::Result<()> {
    if source.is_dir() {
        fs::create_dir_all(target)?;
        for entry in fs::read_dir(source)? {
            let entry = entry?;
            copy(&entry.path(), &target.join(entry.file_name()), progress)?;
        }
        return Ok(());
    }
    let mut input = fs::File::open(source)?;
    let mut output = fs::File::create(target)?;
    let mut buffer = vec![0; 8 << 20];
    loop {
        let read = input.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        output.write_all(&buffer[..read])?;
        progress.copied.fetch_add(read as u64, Ordering::Relaxed);
    }
    output.sync_all()
}

fn remove(path: &Path) -> io::Result<()> {
    match fs::symlink_metadata(path) {
        Ok(meta) if meta.is_dir() => fs::remove_dir_all(path),
        Ok(_) => fs::remove_file(path),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("echora-install-{name}-{}", std::process::id()));
        let _ = remove(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn installs_a_file_and_replaces_a_dump() {
        let dir = scratch("file");
        let games = dir.join("games");
        fs::create_dir_all(games.join("dump")).unwrap();
        fs::write(dir.join("source.nup"), vec![7u8; 20 << 20]).unwrap();
        let progress = Progress::default();
        install(&dir.join("source.nup"), &games, &progress);
        let installed = progress.take_result().unwrap().unwrap();
        assert_eq!(installed, games.join("game.nup"));
        assert_eq!(fs::read(&installed).unwrap().len(), 20 << 20);
        assert!(!games.join("dump").exists());
        assert!(!games.join(".installing").exists());
        assert_eq!(progress.fraction(), 1.0);
        remove(&dir).unwrap();
    }

    #[test]
    fn installs_a_dump_folder() {
        let dir = scratch("dump");
        fs::create_dir_all(dir.join("source/exefs")).unwrap();
        fs::write(dir.join("source/exefs/main"), b"code").unwrap();
        fs::write(dir.join("source/romfs.bin"), b"assets").unwrap();
        let progress = Progress::default();
        install(&dir.join("source"), &dir.join("games"), &progress);
        let installed = progress.take_result().unwrap().unwrap();
        assert_eq!(fs::read(installed.join("exefs/main")).unwrap(), b"code");
        assert_eq!(fs::read(installed.join("romfs.bin")).unwrap(), b"assets");
        remove(&dir).unwrap();
    }

    #[test]
    fn reinstalling_the_installed_copy_is_a_no_op() {
        let dir = scratch("same");
        let games = dir.join("games");
        fs::create_dir_all(&games).unwrap();
        fs::write(games.join("game.nup"), b"installed").unwrap();
        let progress = Progress::default();
        install(&games.join("game.nup"), &games, &progress);
        assert_eq!(
            progress.take_result().unwrap().unwrap(),
            games.join("game.nup")
        );
        assert_eq!(fs::read(games.join("game.nup")).unwrap(), b"installed");
        remove(&dir).unwrap();
    }
}
