//! Installing the game into Echora's own data folder, so the NUP or dump
//! the player chose can be moved, renamed or deleted afterwards.
//!
//! Each file is placed as cheaply as the disk allows: a copy-on-write clone
//! (Btrfs, XFS), else a hard link on the same filesystem, both instant and
//! taking no extra space, else a real copy, started only once there is room
//! for it. The install is staged next to the old one and swapped in only once
//! it is complete, so a failed or interrupted install leaves the previous one
//! in place.

// Cloning a file and asking for free space are system calls (FICLONE,
// statvfs) with no safe standard-library wrapper.
#![allow(unsafe_code)]

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
    let mut copying = Copying::Unchecked;
    if let Err(error) = place(source, &staging, progress, &mut copying) {
        // Leave nothing half-made behind; the previous install stays.
        let _ = remove(&staging);
        return Err(error);
    }
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

/// Whether a real copy has had its free space checked.
enum Copying {
    Unchecked,
    Checked,
}

/// Places `source` (a file, or a folder of them) at `target`: cloned, else
/// linked, else copied.
fn place(
    source: &Path,
    target: &Path,
    progress: &Progress,
    copying: &mut Copying,
) -> io::Result<()> {
    if source.is_dir() {
        fs::create_dir_all(target)?;
        for entry in fs::read_dir(source)? {
            let entry = entry?;
            place(
                &entry.path(),
                &target.join(entry.file_name()),
                progress,
                copying,
            )?;
        }
        return Ok(());
    }
    let size = fs::metadata(source)?.len();
    if clone(source, target).is_ok() || fs::hard_link(source, target).is_ok() {
        progress.copied.fetch_add(size, Ordering::Relaxed);
        return Ok(());
    }
    if let Copying::Unchecked = copying {
        let needed =
            progress.total.load(Ordering::Relaxed) - progress.copied.load(Ordering::Relaxed);
        check_space(target, needed)?;
        *copying = Copying::Checked;
    }
    copy(source, target, progress)
}

/// A copy-on-write clone of `source` at `target` (Linux `FICLONE`).
#[cfg(target_os = "linux")]
fn clone(source: &Path, target: &Path) -> io::Result<()> {
    use std::os::fd::AsRawFd;
    const FICLONE: libc::c_ulong = 0x4004_9409;
    let input = fs::File::open(source)?;
    let output = fs::File::create(target)?;
    // SAFETY: both descriptors are open for the duration of the call.
    let result = unsafe { libc::ioctl(output.as_raw_fd(), FICLONE, input.as_raw_fd()) };
    if result == 0 {
        return Ok(());
    }
    let error = io::Error::last_os_error();
    drop(output);
    let _ = fs::remove_file(target);
    Err(error)
}

#[cfg(not(target_os = "linux"))]
fn clone(_source: &Path, _target: &Path) -> io::Result<()> {
    Err(io::Error::from(io::ErrorKind::Unsupported))
}

/// Free space kept on top of the game when copying it.
const SPACE_HEADROOM: u64 = 512 << 20;

/// Fails with a plain message when the disk holding `target` has less than
/// `needed` bytes free (plus a little headroom).
fn check_space(target: &Path, needed: u64) -> io::Result<()> {
    let Some(free) = free_space(target.parent().unwrap_or(target)) else {
        return Ok(());
    };
    if free < needed + SPACE_HEADROOM {
        let gb = |bytes: u64| bytes as f64 / 1e9;
        return Err(io::Error::other(format!(
            "there isn't enough free space: the game needs {:.1} GB and the disk has {:.1} GB free",
            gb(needed),
            gb(free)
        )));
    }
    Ok(())
}

#[cfg(unix)]
fn free_space(dir: &Path) -> Option<u64> {
    use std::os::unix::ffi::OsStrExt;
    let path = std::ffi::CString::new(dir.as_os_str().as_bytes()).ok()?;
    // SAFETY: statvfs is plain old data, valid when zeroed.
    let mut stats: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: a valid C string and a buffer of the right type.
    if unsafe { libc::statvfs(path.as_ptr(), &mut stats) } != 0 {
        return None;
    }
    Some(stats.f_bavail as u64 * stats.f_frsize as u64)
}

#[cfg(not(unix))]
fn free_space(_dir: &Path) -> Option<u64> {
    None
}

fn copy(source: &Path, target: &Path, progress: &Progress) -> io::Result<()> {
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
    fn the_install_outlives_its_source() {
        let dir = scratch("outlives");
        fs::write(dir.join("source.nup"), b"game data").unwrap();
        let progress = Progress::default();
        install(&dir.join("source.nup"), &dir.join("games"), &progress);
        let installed = progress.take_result().unwrap().unwrap();
        // Renamed, then deleted: the install keeps its own copy of the data.
        fs::rename(dir.join("source.nup"), dir.join("renamed.nup")).unwrap();
        assert_eq!(fs::read(&installed).unwrap(), b"game data");
        fs::remove_file(dir.join("renamed.nup")).unwrap();
        assert_eq!(fs::read(&installed).unwrap(), b"game data");
        remove(&dir).unwrap();
    }

    #[test]
    fn a_copy_that_cannot_fit_says_so() {
        let dir = scratch("space");
        // Without a free-space query (not Unix) the copy just starts.
        let result = check_space(&dir.join("game.nup"), u64::MAX / 2);
        if cfg!(unix) {
            let error = result.unwrap_err();
            assert!(error.to_string().contains("free space"), "{error}");
        }
        // A small copy fits whenever the disk has the headroom free.
        if free_space(&dir).is_none_or(|free| free > SPACE_HEADROOM + 1) {
            assert!(check_space(&dir.join("game.nup"), 1).is_ok());
        }
        remove(&dir).unwrap();
    }

    /// With `ECHORA_TEST_NUP` set: installing the real game next to it (same
    /// filesystem) is a clone or a link, so it takes moments, not a copy.
    #[test]
    #[ignore]
    fn the_real_game_installs_without_copying() {
        let Some(nup) = std::env::var_os("ECHORA_TEST_NUP").map(PathBuf::from) else {
            return;
        };
        let games = nup.with_file_name(format!("echora-install-test-{}", std::process::id()));
        let progress = Progress::default();
        let start = std::time::Instant::now();
        install(&nup, &games, &progress);
        let installed = progress.take_result().unwrap().unwrap();
        let elapsed = start.elapsed();
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let links = fs::metadata(&nup).unwrap().nlink();
            let how = if links > 1 { "a hard link" } else { "a clone" };
            eprintln!("installed {} as {how} in {elapsed:?}", installed.display());
        }
        assert_eq!(
            fs::metadata(&installed).unwrap().len(),
            fs::metadata(&nup).unwrap().len()
        );
        remove(&games).unwrap();
        assert!(elapsed.as_secs() < 5, "installing took {elapsed:?}");
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
