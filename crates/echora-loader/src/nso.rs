//! NSO executable modules, including ZBIC-compressed segments.

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::{LoaderError, Result, zbic};

pub const HEADER_SIZE: usize = 0x100;
pub const MAX_NSO_FILE_SIZE: u64 = 256 << 20;
pub const MAX_SEGMENT_SIZE: u64 = 256 << 20;
pub const MAX_IMAGE_SIZE: u64 = 512 << 20;

/// Flag bits: per-segment compression (0-2) and hash checks (3-5).
pub const FLAG_COMPRESSED_MASK: u32 = 0x07;
pub const FLAG_HASH_MASK: u32 = 0x38;
/// Compressed segments are ZBIC frames instead of raw LZ4 blocks.
pub const FLAG_ZBIC: u32 = 0x80;
/// Set together with [`FLAG_ZBIC`] in every ZBIC module seen so far. Its
/// meaning is not established; segment hashes still verify the output.
pub const FLAG_UNKNOWN_6: u32 = 0x40;

const SEGMENT_NAMES: [&str; 3] = ["text", "rodata", "data"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Compression {
    None,
    Lz4,
    Zbic,
}

#[derive(Clone, Serialize)]
pub struct Segment {
    pub memory_offset: u32,
    pub file_offset: u32,
    pub stored_size: u32,
    pub compression: Compression,
    #[serde(serialize_with = "crate::hex_serde")]
    pub sha256: [u8; 32],
    /// The header's hash for this segment was checked against `bytes`.
    pub hash_verified: bool,
    /// Guest bytes; never serialized into reports.
    #[serde(skip)]
    pub bytes: Vec<u8>,
}

impl std::fmt::Debug for Segment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Segment")
            .field("memory_offset", &self.memory_offset)
            .field("size", &self.bytes.len())
            .field("compression", &self.compression)
            .field("hash_verified", &self.hash_verified)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct NsoModule {
    pub name: String,
    #[serde(serialize_with = "crate::hex_serde")]
    pub build_id: [u8; 32],
    pub flags: u32,
    pub text: Segment,
    pub rodata: Segment,
    pub data: Segment,
    pub bss_size: u32,
    /// Size of the mapped image (segments plus BSS), rounded up to 4 KiB.
    pub image_size: u64,
    pub mod0: Option<crate::mod0::Mod0Info>,
}

impl NsoModule {
    pub fn segments(&self) -> [&Segment; 3] {
        [&self.text, &self.rodata, &self.data]
    }
}

fn u32_at(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

#[derive(Debug, Clone, Copy)]
struct Descriptor {
    file_offset: u32,
    memory_offset: u32,
    size: u32,
    stored_size: u32,
    compressed: bool,
}

/// Parses and decompresses an NSO, checking every bound and each enabled
/// segment hash. `image_limit` caps the mapped size.
pub fn parse(name: &str, bytes: &[u8], image_limit: u64) -> Result<NsoModule> {
    let ctx = |m: String| LoaderError::Format(format!("{name}: {m}"));
    if bytes.len() < HEADER_SIZE || &bytes[..4] != b"NSO0" {
        return Err(ctx("not an NSO (missing NSO0 header)".into()));
    }
    if bytes.len() as u64 > MAX_NSO_FILE_SIZE {
        return Err(ctx("file exceeds the input limit".into()));
    }
    let version = u32_at(bytes, 4);
    if version != 0 {
        return Err(ctx(format!("unsupported NSO version {version}")));
    }
    let flags = u32_at(bytes, 0x0c);
    let known = FLAG_COMPRESSED_MASK | FLAG_HASH_MASK | FLAG_ZBIC | FLAG_UNKNOWN_6;
    if flags & !known != 0 {
        return Err(ctx(format!("unsupported NSO flags {flags:#x}")));
    }
    if flags & FLAG_UNKNOWN_6 != 0 && flags & FLAG_ZBIC == 0 {
        return Err(ctx(format!(
            "NSO flag 0x40 without ZBIC is unsupported ({flags:#x})"
        )));
    }

    let mut descs = [Descriptor {
        file_offset: 0,
        memory_offset: 0,
        size: 0,
        stored_size: 0,
        compressed: false,
    }; 3];
    for (i, label) in SEGMENT_NAMES.iter().enumerate() {
        let h = 0x10 + i * 0x10;
        let compressed = flags & (1 << i) != 0;
        let size = u32_at(bytes, h + 8);
        let d = Descriptor {
            file_offset: u32_at(bytes, h),
            memory_offset: u32_at(bytes, h + 4),
            size,
            stored_size: if compressed {
                u32_at(bytes, 0x60 + i * 4)
            } else {
                size
            },
            compressed,
        };
        if u64::from(d.size) > MAX_SEGMENT_SIZE {
            return Err(ctx(format!("{label} segment exceeds the size limit")));
        }
        if d.size != 0 && d.stored_size == 0 {
            return Err(ctx(format!("nonempty {label} segment has no stored data")));
        }
        if d.stored_size != 0 && (d.file_offset as usize) < HEADER_SIZE {
            return Err(ctx(format!("{label} segment overlaps the header")));
        }
        let file_end = u64::from(d.file_offset) + u64::from(d.stored_size);
        if file_end > bytes.len() as u64 {
            return Err(ctx(format!("{label} segment extends past the file")));
        }
        if u64::from(d.memory_offset) + u64::from(d.size) > image_limit {
            return Err(ctx(format!("{label} segment exceeds the image budget")));
        }
        descs[i] = d;
    }
    if descs[0].size == 0 {
        return Err(ctx("empty text segment".into()));
    }
    if descs[0].memory_offset % 4 != 0 || descs[0].size % 4 != 0 {
        return Err(ctx("text segment is not instruction-aligned".into()));
    }
    let file_ranges: Vec<_> = descs
        .iter()
        .map(|d| (u64::from(d.file_offset), u64::from(d.stored_size)))
        .collect();
    check_disjoint(&file_ranges).map_err(|_| ctx("segment file ranges overlap".into()))?;

    let bss_size = u32_at(bytes, 0x3c);
    let bss_start = u64::from(descs[2].memory_offset) + u64::from(descs[2].size);
    let mut mem_ranges: Vec<_> = descs
        .iter()
        .map(|d| (u64::from(d.memory_offset), u64::from(d.size)))
        .collect();
    mem_ranges.push((bss_start, u64::from(bss_size)));
    check_disjoint(&mem_ranges).map_err(|_| ctx("segment memory ranges overlap".into()))?;
    let end = mem_ranges.iter().map(|(s, l)| s + l).max().unwrap_or(0);
    let image_size = (end + 0xfff) & !0xfff;
    if image_size > image_limit {
        return Err(ctx(format!(
            "mapped image exceeds the {image_limit}-byte budget"
        )));
    }

    let zbic = flags & FLAG_ZBIC != 0;
    let mut segments = Vec::with_capacity(3);
    for (i, d) in descs.iter().enumerate() {
        let label = SEGMENT_NAMES[i];
        let stored = &bytes[d.file_offset as usize..(d.file_offset + d.stored_size) as usize];
        let (compression, decoded) = if !d.compressed || (d.size == 0 && stored.is_empty()) {
            if stored.len() != d.size as usize {
                return Err(ctx(format!("uncompressed {label} size mismatch")));
            }
            (Compression::None, stored.to_vec())
        } else if zbic {
            let out = zbic::decompress(stored, d.size as usize)
                .map_err(|e| ctx(format!("{label} segment: {e}")))?;
            (Compression::Zbic, out)
        } else {
            let mut out = vec![0; d.size as usize];
            let n = lz4_flex::block::decompress_into(stored, &mut out)
                .map_err(|e| ctx(format!("invalid LZ4 {label} segment: {e}")))?;
            if n != out.len() {
                return Err(ctx(format!(
                    "{label} decompressed to {n} bytes instead of {}",
                    out.len()
                )));
            }
            (Compression::Lz4, out)
        };
        let sha256: [u8; 32] = Sha256::digest(&decoded).into();
        let hash_verified = flags & (1 << (i + 3)) != 0;
        if hash_verified && sha256[..] != bytes[0xa0 + i * 0x20..0xc0 + i * 0x20] {
            return Err(LoaderError::Integrity(format!(
                "{name}: {label} segment SHA-256 does not match the header"
            )));
        }
        segments.push(Segment {
            memory_offset: d.memory_offset,
            file_offset: d.file_offset,
            stored_size: d.stored_size,
            compression,
            sha256,
            hash_verified,
            bytes: decoded,
        });
    }
    let mut it = segments.into_iter();
    let mut build_id = [0; 32];
    build_id.copy_from_slice(&bytes[0x40..0x60]);
    let (Some(text), Some(rodata), Some(data)) = (it.next(), it.next(), it.next()) else {
        unreachable!("three segments were parsed")
    };
    let mut module = NsoModule {
        name: name.to_owned(),
        build_id,
        flags,
        text,
        rodata,
        data,
        bss_size,
        image_size,
        mod0: None,
    };
    module.mod0 = crate::mod0::analyze(&module).map_err(|e| ctx(format!("MOD0 metadata: {e}")))?;
    Ok(module)
}

/// Fails if any two nonempty `(start, len)` ranges overlap.
fn check_disjoint(ranges: &[(u64, u64)]) -> std::result::Result<(), ()> {
    let mut r: Vec<_> = ranges.iter().filter(|(_, l)| *l > 0).copied().collect();
    r.sort_unstable();
    if r.windows(2).any(|w| w[0].0 + w[0].1 > w[1].0) {
        return Err(());
    }
    Ok(())
}
