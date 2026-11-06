//! Header-level identification of ExeFS modules. Full segment loading and
//! decompression belong to the executable loader; this module only reads the
//! fixed headers needed to fingerprint a build.

use serde::Serialize;

/// Size of the fixed NSO header.
pub const NSO_HEADER_SIZE: usize = 0x100;
/// NSO flag bits defined by the original Switch format: per-segment
/// compression (bits 0-2) and per-segment hash checks (bits 3-5).
pub const NSO_KNOWN_FLAGS: u32 = 0x3f;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct NsoSegment {
    pub file_offset: u32,
    pub memory_offset: u32,
    pub memory_size: u32,
    pub file_size: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct NsoHeader {
    pub version: u32,
    pub flags: u32,
    /// Flag bits outside [`NSO_KNOWN_FLAGS`]. Newer SDKs set these (for
    /// example alongside ZBIC-compressed segments); a loader must handle them
    /// explicitly rather than ignore them.
    pub unknown_flags: u32,
    /// ModuleId, the 32-byte build ID that mods, cheats and translation
    /// caches key on.
    #[serde(serialize_with = "hex_bytes")]
    pub build_id: [u8; 32],
    pub text: NsoSegment,
    pub rodata: NsoSegment,
    pub data: NsoSegment,
    pub bss_size: u32,
}

impl NsoHeader {
    pub fn parse(h: &[u8]) -> Option<Self> {
        if h.len() < NSO_HEADER_SIZE || &h[..4] != b"NSO0" {
            return None;
        }
        let u32_at = |o: usize| u32::from_le_bytes([h[o], h[o + 1], h[o + 2], h[o + 3]]);
        let seg = |hdr: usize, size_field: usize| NsoSegment {
            file_offset: u32_at(hdr),
            memory_offset: u32_at(hdr + 4),
            memory_size: u32_at(hdr + 8),
            file_size: u32_at(size_field),
        };
        let flags = u32_at(0x0c);
        let mut build_id = [0; 32];
        build_id.copy_from_slice(&h[0x40..0x60]);
        Some(Self {
            version: u32_at(0x04),
            flags,
            unknown_flags: flags & !NSO_KNOWN_FLAGS,
            build_id,
            text: seg(0x10, 0x60),
            rodata: seg(0x20, 0x64),
            data: seg(0x30, 0x68),
            bss_size: u32_at(0x3c),
        })
    }

    /// The build ID in hex with trailing zero padding removed (at least 8
    /// bytes are kept).
    pub fn build_id_hex(&self) -> String {
        let len = self
            .build_id
            .iter()
            .rposition(|&b| b != 0)
            .map_or(0, |i| i + 1);
        hex::encode(&self.build_id[..len.max(8)])
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct NpdmInfo {
    pub name: String,
    pub program_id: Option<String>,
    pub main_thread_priority: u8,
    pub main_thread_core: u8,
    pub main_thread_stack_size: u32,
    pub address_space_64bit: bool,
}

impl NpdmInfo {
    /// Reads the META header and, when present and in bounds, the ACI0
    /// program ID. Signatures are not checked.
    pub fn parse(b: &[u8]) -> Option<Self> {
        if b.len() < 0x80 || &b[..4] != b"META" {
            return None;
        }
        let u32_at = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        let name_raw = &b[0x20..0x30];
        let name_len = name_raw
            .iter()
            .position(|&c| c == 0)
            .unwrap_or(name_raw.len());
        let aci_off = u32_at(0x70) as usize;
        let aci_size = u32_at(0x74) as usize;
        let program_id = (aci_size >= 0x18)
            .then(|| b.get(aci_off..aci_off + 0x18))
            .flatten()
            .filter(|aci| &aci[..4] == b"ACI0")
            .map(|aci| {
                let mut id = [0; 8];
                id.copy_from_slice(&aci[0x10..0x18]);
                format!("{:016X}", u64::from_le_bytes(id))
            });
        Some(Self {
            name: String::from_utf8_lossy(&name_raw[..name_len]).into_owned(),
            program_id,
            main_thread_priority: b[0x0e],
            main_thread_core: b[0x0f],
            main_thread_stack_size: u32_at(0x1c),
            address_space_64bit: b[0x0c] & 1 != 0,
        })
    }
}

fn hex_bytes<S: serde::Serializer>(v: &[u8; 32], s: S) -> Result<S::Ok, S::Error> {
    s.serialize_str(&hex::encode(v))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nso_header_fields() {
        let mut h = vec![0u8; NSO_HEADER_SIZE];
        h[..4].copy_from_slice(b"NSO0");
        h[0x0c..0x10].copy_from_slice(&0xffu32.to_le_bytes());
        h[0x10..0x14].copy_from_slice(&0x100u32.to_le_bytes());
        h[0x18..0x1c].copy_from_slice(&0x1000u32.to_le_bytes());
        h[0x40..0x48].copy_from_slice(&[0xe0, 0xf3, 1, 2, 3, 4, 5, 6]);
        h[0x48] = 9;
        h[0x60..0x64].copy_from_slice(&0x800u32.to_le_bytes());
        let n = NsoHeader::parse(&h).unwrap();
        assert_eq!(n.flags, 0xff);
        assert_eq!(n.unknown_flags, 0xc0);
        assert_eq!(n.text.file_offset, 0x100);
        assert_eq!(n.text.memory_size, 0x1000);
        assert_eq!(n.text.file_size, 0x800);
        assert_eq!(n.build_id_hex(), "e0f301020304050609");
        assert!(NsoHeader::parse(&h[..0x80]).is_none());
    }

    #[test]
    fn parses_npdm_program_id() {
        let mut b = vec![0u8; 0x100];
        b[..4].copy_from_slice(b"META");
        b[0x0c] = 1;
        b[0x0e] = 44;
        b[0x1c..0x20].copy_from_slice(&0x100000u32.to_le_bytes());
        b[0x20..0x27].copy_from_slice(b"Program");
        b[0x70..0x74].copy_from_slice(&0x80u32.to_le_bytes());
        b[0x74..0x78].copy_from_slice(&0x40u32.to_le_bytes());
        b[0x80..0x84].copy_from_slice(b"ACI0");
        b[0x90..0x98].copy_from_slice(&0x0400_4860_24DE_C000u64.to_le_bytes());
        let n = NpdmInfo::parse(&b).unwrap();
        assert_eq!(n.name, "Program");
        assert_eq!(n.program_id.as_deref(), Some("0400486024DEC000"));
        assert_eq!(n.main_thread_priority, 44);
        assert!(n.address_space_64bit);

        b[0x74..0x78].copy_from_slice(&0x1000u32.to_le_bytes());
        b[0x70..0x74].copy_from_slice(&0xf0u32.to_le_bytes());
        assert_eq!(NpdmInfo::parse(&b).unwrap().program_id, None);
    }
}
