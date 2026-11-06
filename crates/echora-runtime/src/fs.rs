//! Host-directory backing for guest IFileSystem sessions (save data).
//!
//! Guest paths are absolute ("/dir/file"), use '/' separators and may not
//! leave the root. Results use the fs module (2) numbering from yuzu and
//! Atmosphere: PathNotFound (2/1), PathAlreadyExists (2/2), and
//! OutOfRange (2/3005) for reads past the end of a file.

use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub const RESULT_PATH_NOT_FOUND: u32 = 2 | (1 << 9);
pub const RESULT_PATH_ALREADY_EXISTS: u32 = 2 | (2 << 9);
pub const RESULT_INVALID_PATH: u32 = 2 | (6005 << 9);
pub const RESULT_DIRECTORY_NOT_EMPTY: u32 = 2 | (7 << 9);

/// nn::fs::DirectoryEntryType.
pub const ENTRY_DIRECTORY: u32 = 0;
pub const ENTRY_FILE: u32 = 1;
/// OpenDirectory filter bits.
pub const FILTER_DIRECTORIES: u32 = 1;
pub const FILTER_FILES: u32 = 2;
/// Size of nn::fs::DirectoryEntry: name[0x301], attributes, type at 0x304,
/// size at 0x308.
pub const DIRECTORY_ENTRY_SIZE: usize = 0x310;

/// A guest result code for a failed file-system operation.
pub type FsResult<T> = Result<T, u32>;

#[derive(Clone, Debug)]
pub struct HostFileSystem {
    root: PathBuf,
}

/// One directory listing entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub name: String,
    pub kind: u32,
    pub size: u64,
}

impl Entry {
    pub fn encode(&self) -> [u8; DIRECTORY_ENTRY_SIZE] {
        let mut bytes = [0u8; DIRECTORY_ENTRY_SIZE];
        let name = self.name.as_bytes();
        let length = name.len().min(0x300);
        bytes[..length].copy_from_slice(&name[..length]);
        bytes[0x304] = self.kind as u8;
        bytes[0x308..0x310].copy_from_slice(&self.size.to_le_bytes());
        bytes
    }
}

fn not_found(error: std::io::Error) -> u32 {
    match error.kind() {
        std::io::ErrorKind::AlreadyExists => RESULT_PATH_ALREADY_EXISTS,
        std::io::ErrorKind::DirectoryNotEmpty => RESULT_DIRECTORY_NOT_EMPTY,
        _ => RESULT_PATH_NOT_FOUND,
    }
}

/// Decode a NUL-terminated guest path from a buffer.
pub fn guest_path(bytes: &[u8]) -> String {
    let end = bytes
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

impl HostFileSystem {
    pub fn new(root: impl Into<PathBuf>) -> std::io::Result<Self> {
        let root = root.into();
        fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The host path for a guest path, refusing components that would
    /// leave the root.
    pub fn resolve(&self, path: &str) -> FsResult<PathBuf> {
        let mut host = self.root.clone();
        for component in path.split('/') {
            match component {
                "" | "." => {}
                ".." => return Err(RESULT_INVALID_PATH),
                name if name.contains('\\') || name.contains(':') => {
                    return Err(RESULT_INVALID_PATH);
                }
                name => host.push(name),
            }
        }
        Ok(host)
    }

    pub fn create_file(&self, path: &str, size: u64) -> FsResult<()> {
        let host = self.resolve(path)?;
        if host.exists() {
            return Err(RESULT_PATH_ALREADY_EXISTS);
        }
        if !host.parent().is_some_and(Path::is_dir) {
            return Err(RESULT_PATH_NOT_FOUND);
        }
        let file = fs::File::create(&host).map_err(not_found)?;
        file.set_len(size).map_err(not_found)
    }

    pub fn delete_file(&self, path: &str) -> FsResult<()> {
        let host = self.resolve(path)?;
        if !host.is_file() {
            return Err(RESULT_PATH_NOT_FOUND);
        }
        fs::remove_file(host).map_err(not_found)
    }

    pub fn create_directory(&self, path: &str) -> FsResult<()> {
        let host = self.resolve(path)?;
        if host.exists() {
            return Err(RESULT_PATH_ALREADY_EXISTS);
        }
        fs::create_dir(host).map_err(not_found)
    }

    pub fn delete_directory(&self, path: &str, recursive: bool) -> FsResult<()> {
        let host = self.resolve(path)?;
        if !host.is_dir() || host == self.root {
            return Err(RESULT_PATH_NOT_FOUND);
        }
        if recursive {
            fs::remove_dir_all(host).map_err(not_found)
        } else {
            fs::remove_dir(host).map_err(not_found)
        }
    }

    /// Remove everything inside a directory, keeping the directory.
    pub fn clean_directory(&self, path: &str) -> FsResult<()> {
        let host = self.resolve(path)?;
        if !host.is_dir() {
            return Err(RESULT_PATH_NOT_FOUND);
        }
        for entry in fs::read_dir(&host).map_err(not_found)? {
            let entry = entry.map_err(not_found)?.path();
            if entry.is_dir() {
                fs::remove_dir_all(entry).map_err(not_found)?;
            } else {
                fs::remove_file(entry).map_err(not_found)?;
            }
        }
        Ok(())
    }

    pub fn rename(&self, from: &str, to: &str) -> FsResult<()> {
        let (from, to) = (self.resolve(from)?, self.resolve(to)?);
        if !from.exists() {
            return Err(RESULT_PATH_NOT_FOUND);
        }
        if to.exists() {
            return Err(RESULT_PATH_ALREADY_EXISTS);
        }
        fs::rename(from, to).map_err(not_found)
    }

    pub fn entry_type(&self, path: &str) -> FsResult<u32> {
        let host = self.resolve(path)?;
        if host.is_dir() {
            Ok(ENTRY_DIRECTORY)
        } else if host.is_file() {
            Ok(ENTRY_FILE)
        } else {
            Err(RESULT_PATH_NOT_FOUND)
        }
    }

    /// Entries of a directory selected by the OpenDirectory filter, sorted
    /// by name so listings are reproducible.
    pub fn list(&self, path: &str, filter: u32) -> FsResult<Vec<Entry>> {
        let host = self.resolve(path)?;
        if !host.is_dir() {
            return Err(RESULT_PATH_NOT_FOUND);
        }
        let mut entries = Vec::new();
        for entry in fs::read_dir(&host).map_err(not_found)? {
            let entry = entry.map_err(not_found)?;
            let metadata = entry.metadata().map_err(not_found)?;
            let (kind, wanted) = if metadata.is_dir() {
                (ENTRY_DIRECTORY, filter & FILTER_DIRECTORIES != 0)
            } else {
                (ENTRY_FILE, filter & FILTER_FILES != 0)
            };
            if wanted {
                entries.push(Entry {
                    name: entry.file_name().to_string_lossy().into_owned(),
                    kind,
                    size: if kind == ENTRY_FILE {
                        metadata.len()
                    } else {
                        0
                    },
                });
            }
        }
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(entries)
    }

    /// Check that a file exists before an IFile is handed out.
    pub fn open_file(&self, path: &str) -> FsResult<PathBuf> {
        let host = self.resolve(path)?;
        if !host.is_file() {
            return Err(RESULT_PATH_NOT_FOUND);
        }
        Ok(host)
    }
}

/// IFile operations on an opened host file (reopened per call so no host
/// descriptors outlive a request).
pub fn read_at(path: &Path, offset: u64, length: usize) -> FsResult<Vec<u8>> {
    let mut file = fs::File::open(path).map_err(not_found)?;
    let size = file.metadata().map_err(not_found)?.len();
    if offset >= size {
        return Ok(Vec::new());
    }
    let length = length.min((size - offset) as usize);
    let mut bytes = vec![0; length];
    file.seek(SeekFrom::Start(offset)).map_err(not_found)?;
    file.read_exact(&mut bytes).map_err(not_found)?;
    Ok(bytes)
}

/// Write at an offset, growing the file when the write option allows it
/// (nn::fs::WriteOption flush bit is irrelevant on the host).
pub fn write_at(path: &Path, offset: u64, bytes: &[u8]) -> FsResult<()> {
    let mut file = fs::OpenOptions::new()
        .write(true)
        .open(path)
        .map_err(not_found)?;
    file.seek(SeekFrom::Start(offset)).map_err(not_found)?;
    file.write_all(bytes).map_err(not_found)
}

pub fn set_size(path: &Path, size: u64) -> FsResult<()> {
    let file = fs::OpenOptions::new()
        .write(true)
        .open(path)
        .map_err(not_found)?;
    file.set_len(size).map_err(not_found)
}

pub fn size(path: &Path) -> FsResult<u64> {
    fs::metadata(path)
        .map(|metadata| metadata.len())
        .map_err(not_found)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> HostFileSystem {
        let root = std::env::temp_dir().join(format!("echora-fs-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        HostFileSystem::new(root).unwrap()
    }

    #[test]
    fn files_and_directories_round_trip() {
        let fs = scratch("round-trip");
        fs.create_directory("/save").unwrap();
        assert_eq!(
            fs.create_directory("/save"),
            Err(RESULT_PATH_ALREADY_EXISTS)
        );
        fs.create_file("/save/data.bin", 4).unwrap();
        assert_eq!(fs.create_file("/missing/x", 0), Err(RESULT_PATH_NOT_FOUND));
        assert_eq!(fs.entry_type("/save").unwrap(), ENTRY_DIRECTORY);
        assert_eq!(fs.entry_type("/save/data.bin").unwrap(), ENTRY_FILE);
        assert_eq!(fs.entry_type("/nope"), Err(RESULT_PATH_NOT_FOUND));

        let file = fs.open_file("/save/data.bin").unwrap();
        write_at(&file, 2, b"hello").unwrap();
        assert_eq!(size(&file).unwrap(), 7);
        assert_eq!(read_at(&file, 0, 16).unwrap(), b"\0\0hello");
        assert_eq!(read_at(&file, 9, 4).unwrap(), b"");
        set_size(&file, 3).unwrap();
        assert_eq!(read_at(&file, 0, 16).unwrap(), b"\0\0h");

        fs.create_file("/a.txt", 1).unwrap();
        let all = fs.list("/", FILTER_DIRECTORIES | FILTER_FILES).unwrap();
        assert_eq!(
            all.iter()
                .map(|entry| entry.name.as_str())
                .collect::<Vec<_>>(),
            ["a.txt", "save"]
        );
        assert_eq!(fs.list("/", FILTER_FILES).unwrap().len(), 1);
        let encoded = all[0].encode();
        assert_eq!(&encoded[..5], b"a.txt");
        assert_eq!(encoded[0x304], ENTRY_FILE as u8);
        assert_eq!(u64::from_le_bytes(encoded[0x308..].try_into().unwrap()), 1);

        fs.rename("/a.txt", "/b.txt").unwrap();
        assert_eq!(fs.delete_file("/a.txt"), Err(RESULT_PATH_NOT_FOUND));
        fs.delete_file("/b.txt").unwrap();
        assert_eq!(
            fs.delete_directory("/save", false),
            Err(RESULT_DIRECTORY_NOT_EMPTY)
        );
        fs.clean_directory("/save").unwrap();
        fs.delete_directory("/save", false).unwrap();
        assert!(fs.list("/", 3).unwrap().is_empty());
        fs::remove_dir_all(fs.root()).unwrap();
    }

    #[test]
    fn paths_cannot_leave_the_root() {
        let fs = scratch("escape");
        assert_eq!(fs.resolve("/../etc/passwd"), Err(RESULT_INVALID_PATH));
        assert_eq!(fs.resolve("/a/../../b"), Err(RESULT_INVALID_PATH));
        assert_eq!(
            fs.resolve("//a/./b").unwrap(),
            fs.root().join("a").join("b")
        );
        assert_eq!(guest_path(b"/x/y\0junk"), "/x/y");
        fs::remove_dir_all(fs.root()).unwrap();
    }
}
