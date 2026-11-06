//! On-disk layout constants and raw record decoding for NUP v1.
//!
//! Field offsets follow `SPEC.md` of <https://github.com/a55uka/nup> (1.0-draft).

pub const MAGIC: [u8; 4] = *b"NUP\x01";
pub const TOC_MAGIC: [u8; 4] = *b"TOC\x01";
pub const SUPERBLOCK_SIZE: usize = 0x80;
pub const TOC_HEADER_SIZE: usize = 0x30;
pub const PACKAGE_SIZE: usize = 0x40;
pub const DIR_SIZE: usize = 0x18;
pub const FILE_SIZE: usize = 0x50;
pub const EXTENT_ALIGN: u64 = 16;
pub const NO_DIR: u32 = u32::MAX;

pub const PACKAGE_FLAG_STANDALONE_UPDATE: u8 = 0x01;

pub const CODEC_RAW: u8 = 0;
pub const CODEC_ZSTD: u8 = 1;

pub const TLV_NPDM: u16 = 0x0001;
pub const TLV_NACP: u16 = 0x0002;
pub const TLV_ICON: u16 = 0x0003;
pub const TLV_DISPLAY_VERSION: u16 = 0x0004;
pub const TLV_TITLE_NAME: u16 = 0x0005;
pub const TLV_PUBLISHER: u16 = 0x0006;
pub const TLV_BUILD_ID: u16 = 0x0007;
pub const TLV_SOURCE_DUMP: u16 = 0x0008;

/// Little-endian field reader over a fixed record. Callers pass slices whose
/// length was already checked against the record size.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Fields<'a>(pub &'a [u8]);

impl Fields<'_> {
    pub fn u8(self, off: usize) -> u8 {
        self.0[off]
    }

    pub fn u16(self, off: usize) -> u16 {
        u16::from_le_bytes([self.0[off], self.0[off + 1]])
    }

    pub fn u32(self, off: usize) -> u32 {
        let mut b = [0; 4];
        b.copy_from_slice(&self.0[off..off + 4]);
        u32::from_le_bytes(b)
    }

    pub fn u64(self, off: usize) -> u64 {
        let mut b = [0; 8];
        b.copy_from_slice(&self.0[off..off + 8]);
        u64::from_le_bytes(b)
    }

    pub fn bytes<const N: usize>(self, off: usize) -> [u8; N] {
        let mut b = [0; N];
        b.copy_from_slice(&self.0[off..off + N]);
        b
    }
}

/// Package kinds from SPEC §3.1.2. Kinds 3 and 4 are reserved but valid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PackageKind {
    Base,
    Update,
    Addon,
    AddonUpdate,
    System,
}

impl PackageKind {
    pub fn from_raw(raw: u8) -> Option<Self> {
        Some(match raw {
            0 => Self::Base,
            1 => Self::Update,
            2 => Self::Addon,
            3 => Self::AddonUpdate,
            4 => Self::System,
            _ => return None,
        })
    }

    pub fn raw(self) -> u8 {
        match self {
            Self::Base => 0,
            Self::Update => 1,
            Self::Addon => 2,
            Self::AddonUpdate => 3,
            Self::System => 4,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Base => "base",
            Self::Update => "update",
            Self::Addon => "addon",
            Self::AddonUpdate => "addon_update",
            Self::System => "system",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codec {
    Raw,
    Zstd,
}
