//! Executable-format helpers used by the runtime's metadata readers.
//! Modules themselves come from `echora-loader`.

use std::fs::{self, File};
use std::io::Read;
use std::ops::Range;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use sha2::{Digest, Sha256};

pub use echora_loader::NsoModule;

pub(crate) fn checked_range(
    offset: u64,
    size: u64,
    limit: u64,
    what: &str,
) -> Result<Range<usize>> {
    ensure!(
        offset <= limit && size <= limit - offset,
        "{what} is outside its containing region (offset {offset:#x}, size {size:#x}, limit {limit:#x})"
    );
    let start = usize::try_from(offset).context("offset does not fit host address space")?;
    let end =
        usize::try_from(offset + size).context("range end does not fit host address space")?;
    Ok(start..end)
}

pub(crate) fn u32_at(bytes: &[u8], offset: usize, what: &str) -> Result<u32> {
    let range = checked_range(offset as u64, 4, bytes.len() as u64, what)?;
    Ok(crate::bytes::le_u32(&bytes[range]))
}

pub(crate) fn u64_at(bytes: &[u8], offset: usize, what: &str) -> Result<u64> {
    let range = checked_range(offset as u64, 8, bytes.len() as u64, what)?;
    Ok(crate::bytes::le_u64(&bytes[range]))
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut output, "{byte:02x}").expect("writing to a String cannot fail");
    }
    output
}

pub(crate) fn sha256(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

pub fn read_regular_file(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let before =
        fs::symlink_metadata(path).with_context(|| format!("inspect {}", path.display()))?;
    ensure!(
        before.file_type().is_file(),
        "{} must be a regular file, not a symlink or special file",
        path.display()
    );
    ensure!(
        before.len() <= limit,
        "{} exceeds the {limit}-byte input limit",
        path.display()
    );
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let opened = file
        .metadata()
        .with_context(|| format!("inspect opened {}", path.display()))?;
    ensure!(
        opened.is_file() && opened.len() == before.len(),
        "{} changed while opening it",
        path.display()
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        ensure!(
            before.dev() == opened.dev() && before.ino() == opened.ino(),
            "{} changed while opening it",
            path.display()
        );
    }
    let capacity =
        usize::try_from(before.len()).context("input does not fit host address space")?;
    let mut bytes = Vec::with_capacity(capacity);
    file.take(limit.saturating_add(1))
        .read_to_end(&mut bytes)
        .with_context(|| format!("read {}", path.display()))?;
    if bytes.len() as u64 != before.len() {
        bail!("{} changed size while reading it", path.display());
    }
    Ok(bytes)
}
