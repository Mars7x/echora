//! Positional and streaming reads of file extents.

use std::fs::File;
use std::io::{self, BufReader, Read};

use sha2::{Digest, Sha256};

use crate::{Codec, FileId, NupArchive, NupError, Result};

#[cfg(unix)]
fn read_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    std::os::unix::fs::FileExt::read_at(file, buf, offset)
}

#[cfg(windows)]
fn read_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    std::os::windows::fs::FileExt::seek_read(file, buf, offset)
}

#[cfg(not(any(unix, windows)))]
fn read_at(_file: &File, _buf: &mut [u8], _offset: u64) -> io::Result<usize> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "positional file reads are not implemented on this host",
    ))
}

pub(crate) fn read_exact_at(file: &File, mut buf: &mut [u8], mut offset: u64) -> io::Result<()> {
    while !buf.is_empty() {
        match read_at(file, buf, offset) {
            Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
            Ok(n) => {
                buf = &mut buf[n..];
                offset += n as u64;
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Reads one extent without moving any shared file cursor, so several readers
/// can stream from the same archive concurrently.
struct ExtentReader<'a> {
    file: &'a File,
    pos: u64,
    end: u64,
}

impl Read for ExtentReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let left = self.end - self.pos;
        if left == 0 {
            return Ok(0);
        }
        let want = buf.len().min(usize::try_from(left).unwrap_or(usize::MAX));
        let n = read_at(self.file, &mut buf[..want], self.pos)?;
        if n == 0 {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        self.pos += n as u64;
        Ok(n)
    }
}

enum Source<'a> {
    Raw(ExtentReader<'a>),
    Zstd(Box<zstd::stream::read::Decoder<'static, BufReader<ExtentReader<'a>>>>),
}

/// Streams a file's uncompressed contents. When verification is enabled the
/// reader fails at end of stream unless the byte count and SHA-256 match the
/// TOC record; it also fails as soon as a decoder produces more bytes than the
/// recorded size.
pub struct FileReader<'a> {
    source: Source<'a>,
    expected_size: u64,
    expected_sha: [u8; 32],
    produced: u64,
    hasher: Option<Sha256>,
    finished: bool,
}

impl std::fmt::Debug for FileReader<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileReader")
            .field("expected_size", &self.expected_size)
            .field("produced", &self.produced)
            .finish_non_exhaustive()
    }
}

impl Read for FileReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.finished || buf.is_empty() {
            return Ok(0);
        }
        let n = match &mut self.source {
            Source::Raw(r) => r.read(buf)?,
            Source::Zstd(d) => d.read(buf)?,
        };
        self.produced += n as u64;
        if self.produced > self.expected_size {
            return Err(invalid_data("decoded data longer than recorded size"));
        }
        if let Some(h) = &mut self.hasher {
            h.update(&buf[..n]);
        }
        if n == 0 {
            self.finished = true;
            if self.produced != self.expected_size {
                return Err(invalid_data("decoded data shorter than recorded size"));
            }
            if let Some(h) = self.hasher.take() {
                let digest: [u8; 32] = h.finalize().into();
                if digest != self.expected_sha {
                    return Err(invalid_data("SHA-256 mismatch"));
                }
            }
        }
        Ok(n)
    }
}

fn invalid_data(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_owned())
}

impl NupArchive {
    /// Opens a streaming reader for a file. With `verify`, the full content
    /// hash is checked when the stream reaches its end.
    pub fn open_file(&self, id: FileId, verify: bool) -> Result<FileReader<'_>> {
        let entry = self.file(id);
        let extent = ExtentReader {
            file: &self.file,
            pos: entry.extent_offset,
            end: entry.extent_offset + entry.extent_size,
        };
        let source = match entry.codec {
            Codec::Raw => Source::Raw(extent),
            Codec::Zstd => {
                let mut decoder = zstd::stream::read::Decoder::with_buffer(
                    BufReader::with_capacity(256 << 10, extent),
                )
                .map_err(|source| NupError::Io {
                    path: self.path().to_path_buf(),
                    source,
                })?;
                decoder = decoder.single_frame();
                Source::Zstd(Box::new(decoder))
            }
        };
        Ok(FileReader {
            source,
            expected_size: entry.size,
            expected_sha: entry.sha256,
            produced: 0,
            hasher: verify.then(Sha256::new),
            finished: false,
        })
    }

    /// Reads a whole file into memory, verifying its hash. `limit` bounds the
    /// allocation; larger files must be streamed with [`Self::open_file`].
    pub fn read_file(&self, id: FileId, limit: u64) -> Result<Vec<u8>> {
        let size = self.file(id).size;
        if size > limit {
            return Err(NupError::Format(format!(
                "file of {size} bytes exceeds the {limit}-byte read limit"
            )));
        }
        let mut out = Vec::with_capacity(size as usize);
        self.open_file(id, true)?
            .read_to_end(&mut out)
            .map_err(|e| self.content_error(id, e))?;
        Ok(out)
    }

    pub(crate) fn content_error(&self, id: FileId, e: io::Error) -> NupError {
        if e.kind() == io::ErrorKind::InvalidData {
            NupError::Integrity(format!("file {}: {e}", id.0))
        } else {
            NupError::Io {
                path: self.path().to_path_buf(),
                source: e,
            }
        }
    }
}
