//! A reader for IVF files, the container of the AOM test vectors.

use crate::Error;

/// The IVF file header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IvfHeader {
    pub fourcc: [u8; 4],
    pub width: u16,
    pub height: u16,
    pub timebase_denominator: u32,
    pub timebase_numerator: u32,
    pub frame_count: u32,
}

/// One IVF frame: a temporal unit for AV1.
#[derive(Clone, Copy, Debug)]
pub struct IvfFrame<'a> {
    pub timestamp: u64,
    pub data: &'a [u8],
}

/// Splits an IVF file into its header and frames.
pub fn parse_ivf(file: &[u8]) -> Result<(IvfHeader, Vec<IvfFrame<'_>>), Error> {
    if file.len() < 32 || &file[0..4] != b"DKIF" {
        return Err(Error::Invalid("not an IVF file"));
    }
    let u16_at = |o: usize| u16::from_le_bytes([file[o], file[o + 1]]);
    let u32_at = |o: usize| u32::from_le_bytes([file[o], file[o + 1], file[o + 2], file[o + 3]]);
    let header_len = usize::from(u16_at(6));
    let header = IvfHeader {
        fourcc: [file[8], file[9], file[10], file[11]],
        width: u16_at(12),
        height: u16_at(14),
        timebase_denominator: u32_at(16),
        timebase_numerator: u32_at(20),
        frame_count: u32_at(24),
    };
    let mut frames = Vec::new();
    let mut pos = header_len.max(32);
    while pos < file.len() {
        if pos + 12 > file.len() {
            return Err(Error::EndOfData);
        }
        let size = u32_at(pos) as usize;
        let timestamp = u64::from(u32_at(pos + 4)) | (u64::from(u32_at(pos + 8)) << 32);
        let start = pos + 12;
        let data = file.get(start..start + size).ok_or(Error::EndOfData)?;
        frames.push(IvfFrame { timestamp, data });
        pos = start + size;
    }
    Ok((header, frames))
}
