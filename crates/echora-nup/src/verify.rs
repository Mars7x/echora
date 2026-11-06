//! Whole-archive content verification.

use std::io::{self, Read};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use crate::{FileId, NupArchive};

#[derive(Debug, Clone)]
pub struct VerifyProblem {
    pub file: FileId,
    pub message: String,
}

#[derive(Debug, Clone, Default)]
pub struct VerifyReport {
    /// File records covered, including records that share a verified extent.
    pub files: u64,
    /// Distinct extents decoded and hashed.
    pub extents: u64,
    /// Uncompressed bytes hashed.
    pub bytes: u64,
    pub problems: Vec<VerifyProblem>,
}

impl VerifyReport {
    pub fn ok(&self) -> bool {
        self.problems.is_empty()
    }
}

impl NupArchive {
    /// Decodes every distinct extent and checks its size and SHA-256, using
    /// `threads` workers (at least one). Shared (deduplicated) extents are
    /// verified once. `progress` receives the running byte count.
    pub fn verify_contents(&self, threads: usize, progress: &(dyn Fn(u64) + Sync)) -> VerifyReport {
        let mut unique: Vec<FileId> = self
            .files()
            .filter(|(_, f)| f.size > 0)
            .map(|(id, _)| id)
            .collect();
        unique.sort_unstable_by_key(|id| {
            let f = self.file(*id);
            (f.extent_offset, f.extent_size)
        });
        unique.dedup_by_key(|id| {
            let f = self.file(*id);
            (f.extent_offset, f.extent_size)
        });

        let next = AtomicUsize::new(0);
        let bytes = AtomicU64::new(0);
        let problems = Mutex::new(Vec::new());
        std::thread::scope(|s| {
            for _ in 0..threads.max(1) {
                s.spawn(|| {
                    let mut sink = vec![0u8; 1 << 20];
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(&id) = unique.get(i) else { break };
                        let result = self
                            .open_file(id, true)
                            .map_err(|e| e.to_string())
                            .and_then(|mut r| drain(&mut r, &mut sink).map_err(|e| e.to_string()));
                        match result {
                            Ok(n) => progress(bytes.fetch_add(n, Ordering::Relaxed) + n),
                            Err(message) => {
                                if let Ok(mut p) = problems.lock() {
                                    p.push(VerifyProblem { file: id, message });
                                }
                            }
                        }
                    }
                });
            }
        });

        let mut problems = problems.into_inner().unwrap_or_default();
        problems.sort_by_key(|p| p.file);
        VerifyReport {
            files: self.files().filter(|(_, f)| f.size > 0).count() as u64,
            extents: unique.len() as u64,
            bytes: bytes.into_inner(),
            problems,
        }
    }
}

fn drain(r: &mut impl Read, buf: &mut [u8]) -> io::Result<u64> {
    let mut total = 0u64;
    loop {
        match r.read(buf) {
            Ok(0) => return Ok(total),
            Ok(n) => total += n as u64,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
}
