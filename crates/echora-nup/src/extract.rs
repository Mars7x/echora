//! Safe extraction of a directory subtree to the host filesystem.

use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use crate::{DirId, Node, NupArchive, NupError, Result};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExtractStats {
    pub files: u64,
    pub dirs: u64,
    pub bytes: u64,
}

impl NupArchive {
    /// Extracts the tree below `dir` into `dest`, which must not exist or be an
    /// empty directory (not a symlink). Every file is verified against its
    /// recorded SHA-256 while it is written, and appears under its final name
    /// only after verification succeeds. Names were validated as plain path
    /// components when the archive was opened, so output cannot escape `dest`.
    pub fn extract_dir(&self, dir: DirId, dest: &Path) -> Result<ExtractStats> {
        let io_err = |path: &Path| {
            let path = path.to_path_buf();
            move |source| NupError::Io { path, source }
        };
        match fs::symlink_metadata(dest) {
            Ok(m) if m.is_dir() => {
                if fs::read_dir(dest).map_err(io_err(dest))?.next().is_some() {
                    return Err(NupError::Io {
                        path: dest.to_path_buf(),
                        source: io::Error::new(
                            io::ErrorKind::AlreadyExists,
                            "destination directory is not empty",
                        ),
                    });
                }
            }
            Ok(_) => {
                return Err(NupError::Io {
                    path: dest.to_path_buf(),
                    source: io::Error::new(
                        io::ErrorKind::AlreadyExists,
                        "destination exists and is not a directory",
                    ),
                });
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                fs::create_dir_all(dest).map_err(io_err(dest))?;
            }
            Err(e) => return Err(io_err(dest)(e)),
        }

        let mut stats = ExtractStats::default();
        let mut stack: Vec<(DirId, PathBuf)> = vec![(dir, dest.to_path_buf())];
        let mut buf = vec![0u8; 1 << 20];
        while let Some((d, out_dir)) = stack.pop() {
            for child in self.children(d) {
                let target = out_dir.join(self.name(child));
                match child {
                    Node::Dir(sub) => {
                        fs::create_dir(&target).map_err(io_err(&target))?;
                        stats.dirs += 1;
                        stack.push((sub, target));
                    }
                    Node::File(f) => {
                        let partial =
                            target.with_file_name(format!("{}.echora-partial", self.name(child)));
                        let result = (|| {
                            let mut out = OpenOptions::new()
                                .write(true)
                                .create_new(true)
                                .open(&partial)
                                .map_err(io_err(&partial))?;
                            let mut reader = self.open_file(f, true)?;
                            loop {
                                let n = io::Read::read(&mut reader, &mut buf)
                                    .map_err(|e| self.content_error(f, e))?;
                                if n == 0 {
                                    break;
                                }
                                out.write_all(&buf[..n]).map_err(io_err(&partial))?;
                            }
                            out.flush().map_err(io_err(&partial))?;
                            fs::rename(&partial, &target).map_err(io_err(&target))
                        })();
                        if result.is_err() {
                            let _ = fs::remove_file(&partial);
                        }
                        result?;
                        stats.files += 1;
                        stats.bytes += self.file(f).size;
                    }
                }
            }
        }
        Ok(stats)
    }
}
