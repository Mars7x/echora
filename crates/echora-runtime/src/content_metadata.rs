//! Bounded inspection of extracted NACP and plaintext packaged CNMT metadata.
//!
//! This does not decrypt an NCA, authenticate metadata, or verify the content
//! payloads named by a CNMT. IDs and version gates are reported as stored; no
//! relationships are inferred from a filename or title-ID arithmetic.
//!
//! Layout references:
//! - <https://switchbrew.org/wiki/NACP>
//! - <https://switchbrew.org/w/index.php?title=CNMT&oldid=13421>
//! - <https://switchbrew.org/w/index.php?title=NCM_services&oldid=14750>
//! - <https://github.com/Atmosphere-NX/Atmosphere/blob/master/libraries/libstratosphere/include/stratosphere/ncm/ncm_content_meta.hpp>

use anyhow::{Context, Result, ensure};
use serde::{Serialize, Serializer};
use std::path::Path;

use crate::loader::{hex, read_regular_file, sha256, u32_at, u64_at};

pub const NACP_SIZE: usize = 0x4000;
pub const MAX_CNMT_SIZE: usize = 16 * 1024 * 1024;
const CNMT_HEADER_SIZE: usize = 0x20;
const CNMT_DIGEST_SIZE: usize = 0x20;
const PACKAGED_CONTENT_INFO_SIZE: usize = 0x38;
const CONTENT_META_INFO_SIZE: usize = 0x10;

fn serialize_id<S: Serializer>(value: &u64, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(&format!("{value:016x}"))
}

fn serialize_optional_id<S: Serializer>(
    value: &Option<u64>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    match value {
        Some(value) => serializer.serialize_some(&format!("{value:016x}")),
        None => serializer.serialize_none(),
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ControlMetadata {
    pub format: &'static str,
    pub file_size: usize,
    pub file_sha256: String,
    pub display_version: String,
    pub display_version_raw: String,
    pub display_version_null_terminated: bool,
    #[serde(serialize_with = "serialize_id")]
    pub presence_group_id: u64,
    #[serde(serialize_with = "serialize_id")]
    pub add_on_content_base_id: u64,
    #[serde(serialize_with = "serialize_id")]
    pub save_data_owner_id: u64,
    pub add_on_content_registration_type_raw: u8,
    pub runtime_add_on_content_install_raw: u8,
    pub program_index: u8,
    pub titles_data_format_raw: u8,
    pub apparent_platform_raw: u8,
    pub metadata_authenticated: bool,
    pub interpretation_notes: &'static str,
}

/// The numeric CNMT `version` is a content version, not a firmware version.
/// The actual extended-header length distinguishes these two layouts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AddOnContentLayout {
    Legacy0x10,
    Since15_0_0Size0x18,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExtendedHeader {
    Application {
        #[serde(serialize_with = "serialize_id")]
        patch_id: u64,
        required_system_version: u32,
        required_application_version: u32,
    },
    Patch {
        #[serde(serialize_with = "serialize_id")]
        application_id: u64,
        required_system_version: u32,
        extended_data_size: u32,
    },
    AddOnContent {
        layout: AddOnContentLayout,
        #[serde(serialize_with = "serialize_id")]
        application_id: u64,
        required_application_version: u32,
        content_accessibilities: Option<u8>,
        #[serde(serialize_with = "serialize_optional_id")]
        data_patch_id: Option<u64>,
    },
    DataPatch {
        #[serde(serialize_with = "serialize_id")]
        data_id: u64,
        #[serde(serialize_with = "serialize_id")]
        application_id: u64,
        required_application_version: u32,
        extended_data_size: u32,
    },
    Delta {
        #[serde(serialize_with = "serialize_id")]
        application_id: u64,
        extended_data_size: u32,
    },
    SystemUpdate {
        extended_data_size: u32,
    },
    /// Unknown or otherwise unimplemented types have no inferred relationships.
    /// Their header and remaining extended data are retained as raw metadata.
    Uninterpreted {
        content_meta_type_raw: u8,
    },
}

/// Byte 0x35 of a PackagedContentInfo is ambiguous without a format-era
/// context: before 15.0.0 it is the sixth size byte; since 15.0.0 it contains
/// content attributes and only the preceding five bytes encode the size.
/// Both interpretations are explicit; neither is selected by this parser.
#[derive(Clone, Debug, Serialize)]
pub struct ContentSizeField {
    pub raw_six_bytes: String,
    pub pre_15_0_0_size_interpretation: u64,
    pub since_15_0_0_size_interpretation: u64,
    pub since_15_0_0_attributes_interpretation: u8,
}

#[derive(Clone, Debug, Serialize)]
pub struct PackagedContentInfo {
    pub expected_sha256: String,
    /// A 16-byte ContentId, distinct from the 8-byte title IDs in headers.
    pub content_id: String,
    pub size_field: ContentSizeField,
    pub content_type_raw: u8,
    pub content_type: &'static str,
    pub id_offset: u8,
}

#[derive(Clone, Debug, Serialize)]
pub struct ContentMetaInfo {
    #[serde(serialize_with = "serialize_id")]
    pub title_id: u64,
    pub version: u32,
    pub content_meta_type_raw: u8,
    pub content_meta_type: &'static str,
    pub attributes_raw: u8,
}

#[derive(Clone, Debug, Serialize)]
pub struct OpaqueRegion {
    pub offset: usize,
    pub size: usize,
    pub sha256: String,
    pub interpreted: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct ContentMeta {
    pub format: &'static str,
    pub file_size: usize,
    pub file_sha256: String,
    #[serde(serialize_with = "serialize_id")]
    pub title_id: u64,
    pub version: u32,
    pub content_meta_type_raw: u8,
    pub content_meta_type: &'static str,
    /// This byte was reserved before 17.0.0; the raw value is always retained.
    pub platform_raw: u8,
    pub platform_interpretation_since_17_0_0: &'static str,
    pub attributes_raw: u8,
    pub required_download_system_version: u32,
    pub header_raw: String,
    pub extended_header_size: u16,
    pub extended_header: ExtendedHeader,
    pub extended_header_raw: String,
    pub contents: Vec<PackagedContentInfo>,
    pub content_meta: Vec<ContentMetaInfo>,
    pub extended_data: OpaqueRegion,
    pub stored_digest: String,
    pub stored_digest_verified: bool,
    pub metadata_authenticated: bool,
    pub referenced_content_verified: bool,
    pub interpretation_notes: &'static str,
}

impl ContentMeta {
    /// The stored parent application ID, or this record's own ID when it is an
    /// Application. This is a metadata declaration, not an authenticated link.
    pub fn application_id(&self) -> Option<u64> {
        match &self.extended_header {
            ExtendedHeader::Application { .. } => Some(self.title_id),
            ExtendedHeader::Patch { application_id, .. }
            | ExtendedHeader::AddOnContent { application_id, .. }
            | ExtendedHeader::DataPatch { application_id, .. }
            | ExtendedHeader::Delta { application_id, .. } => Some(*application_id),
            _ => None,
        }
    }
}

pub fn load_nacp(path: &Path) -> Result<ControlMetadata> {
    parse_nacp(&read_regular_file(path, NACP_SIZE as u64)?)
        .with_context(|| format!("parse extracted NACP {}", path.display()))
}

pub fn load_cnmt(path: &Path) -> Result<ContentMeta> {
    parse_cnmt(&read_regular_file(path, MAX_CNMT_SIZE as u64)?)
        .with_context(|| format!("parse plaintext CNMT {}", path.display()))
}

/// Parse one extracted 0x4000-byte control.nacp. Localized titles are not read:
/// newer NACP files can store their entire title area in compressed form.
pub fn parse_nacp(bytes: &[u8]) -> Result<ControlMetadata> {
    ensure!(
        bytes.len() == NACP_SIZE,
        "NACP must be exactly {NACP_SIZE} bytes; got {} (provide extracted control.nacp)",
        bytes.len()
    );
    let raw_version = &bytes[0x3060..0x3070];
    let terminator = raw_version.iter().position(|byte| *byte == 0);
    let version = &raw_version[..terminator.unwrap_or(raw_version.len())];
    let display_version = std::str::from_utf8(version)
        .context("NACP DisplayVersion is not valid UTF-8")?
        .to_owned();
    Ok(ControlMetadata {
        format: "nacp",
        file_size: bytes.len(),
        file_sha256: sha256(bytes),
        display_version,
        display_version_raw: hex(raw_version),
        display_version_null_terminated: terminator.is_some(),
        presence_group_id: u64_at(bytes, 0x3038, "NACP PresenceGroupId")?,
        add_on_content_base_id: u64_at(bytes, 0x3070, "NACP AddOnContentBaseId")?,
        save_data_owner_id: u64_at(bytes, 0x3078, "NACP SaveDataOwnerId")?,
        add_on_content_registration_type_raw: bytes[0x3027],
        runtime_add_on_content_install_raw: bytes[0x30f2],
        program_index: bytes[0x3212],
        titles_data_format_raw: bytes[0x3215],
        apparent_platform_raw: bytes[0x3217],
        metadata_authenticated: false,
        interpretation_notes: "DisplayVersion is a stored label, not a mapping from a CNMT content version. AddOnContentBaseId does not establish which DLC is installed. Reserved or version-dependent bytes are retained as raw values. Localized titles and compressed title data are not interpreted. This file has not been authenticated or linked to any supplied CNMT/content payload.",
    })
}

fn u16_at(bytes: &[u8], offset: usize) -> Result<u16> {
    let end = offset.checked_add(2).context("CNMT offset overflow")?;
    let value = bytes.get(offset..end).context("truncated CNMT u16")?;
    Ok(u16::from_le_bytes([value[0], value[1]]))
}

fn meta_type_name(kind: u8) -> &'static str {
    match kind {
        0x00 => "invalid",
        0x01 => "system_program",
        0x02 => "system_data",
        0x03 => "system_update",
        0x04 => "boot_image_package",
        0x05 => "boot_image_package_safe",
        0x80 => "application",
        0x81 => "patch",
        0x82 => "add_on_content",
        0x83 => "delta",
        0x84 => "data_patch",
        _ => "unknown",
    }
}

fn content_type_name(kind: u8) -> &'static str {
    match kind {
        0 => "meta",
        1 => "program",
        2 => "data",
        3 => "control",
        4 => "html_document",
        5 => "legal_information",
        6 => "delta_fragment",
        _ => "unknown",
    }
}

fn require_header_size(bytes: &[u8], expected: usize, kind: u8) -> Result<()> {
    ensure!(
        bytes.len() == expected,
        "unsupported or malformed {} extended-header size: expected {expected:#x}, got {:#x}",
        meta_type_name(kind),
        bytes.len()
    );
    Ok(())
}

/// Returns the interpreted header and, where documented, its exact extended
/// data size. None leaves the remaining region uninterpreted for unknown types.
fn parse_extended_header(kind: u8, bytes: &[u8]) -> Result<(ExtendedHeader, Option<usize>)> {
    let result = match kind {
        0x80 => {
            require_header_size(bytes, 0x10, kind)?;
            (
                ExtendedHeader::Application {
                    patch_id: u64_at(bytes, 0, "Application PatchId")?,
                    required_system_version: u32_at(bytes, 8, "Application RequiredSystemVersion")?,
                    required_application_version: u32_at(
                        bytes,
                        12,
                        "Application RequiredApplicationVersion",
                    )?,
                },
                Some(0),
            )
        }
        0x81 => {
            require_header_size(bytes, 0x18, kind)?;
            let size = u32_at(bytes, 12, "Patch ExtendedDataSize")?;
            (
                ExtendedHeader::Patch {
                    application_id: u64_at(bytes, 0, "Patch ApplicationId")?,
                    required_system_version: u32_at(bytes, 8, "Patch RequiredSystemVersion")?,
                    extended_data_size: size,
                },
                Some(size as usize),
            )
        }
        0x82 => {
            ensure!(
                matches!(bytes.len(), 0x10 | 0x18),
                "unsupported or malformed AddOnContent extended-header size {:#x}; expected 0x10 or 0x18",
                bytes.len()
            );
            let modern = bytes.len() == 0x18;
            (
                ExtendedHeader::AddOnContent {
                    layout: if modern {
                        AddOnContentLayout::Since15_0_0Size0x18
                    } else {
                        AddOnContentLayout::Legacy0x10
                    },
                    application_id: u64_at(bytes, 0, "AddOnContent ApplicationId")?,
                    required_application_version: u32_at(
                        bytes,
                        8,
                        "AddOnContent RequiredApplicationVersion",
                    )?,
                    content_accessibilities: modern.then_some(bytes[12]),
                    data_patch_id: if modern {
                        Some(u64_at(bytes, 16, "AddOnContent DataPatchId")?)
                    } else {
                        None
                    },
                },
                Some(0),
            )
        }
        0x84 => {
            require_header_size(bytes, 0x20, kind)?;
            let size = u32_at(bytes, 20, "DataPatch ExtendedDataSize")?;
            (
                ExtendedHeader::DataPatch {
                    data_id: u64_at(bytes, 0, "DataPatch DataId")?,
                    application_id: u64_at(bytes, 8, "DataPatch ApplicationId")?,
                    required_application_version: u32_at(
                        bytes,
                        16,
                        "DataPatch RequiredApplicationVersion",
                    )?,
                    extended_data_size: size,
                },
                Some(size as usize),
            )
        }
        0x83 => {
            require_header_size(bytes, 0x10, kind)?;
            let size = u32_at(bytes, 8, "Delta ExtendedDataSize")?;
            (
                ExtendedHeader::Delta {
                    application_id: u64_at(bytes, 0, "Delta ApplicationId")?,
                    extended_data_size: size,
                },
                Some(size as usize),
            )
        }
        0x03 => {
            require_header_size(bytes, 4, kind)?;
            let size = u32_at(bytes, 0, "SystemUpdate ExtendedDataSize")?;
            (
                ExtendedHeader::SystemUpdate {
                    extended_data_size: size,
                },
                Some(size as usize),
            )
        }
        _ => (
            ExtendedHeader::Uninterpreted {
                content_meta_type_raw: kind,
            },
            None,
        ),
    };
    Ok(result)
}

fn checked_table_end(start: usize, count: usize, stride: usize, limit: usize) -> Result<usize> {
    let size = count
        .checked_mul(stride)
        .context("CNMT table size overflow")?;
    let end = start
        .checked_add(size)
        .context("CNMT table offset overflow")?;
    ensure!(
        end <= limit,
        "CNMT table is truncated or overlaps the trailing digest"
    );
    Ok(end)
}

/// Inspect one plaintext PackagedContentMeta (.cnmt), including its trailing
/// stored digest. The digest is not an authenticity check: production metadata
/// can retain the digest calculated from its development counterpart.
pub fn parse_cnmt(bytes: &[u8]) -> Result<ContentMeta> {
    ensure!(
        bytes.len() <= MAX_CNMT_SIZE,
        "CNMT exceeds the 16 MiB input limit"
    );
    ensure!(
        bytes.len() >= CNMT_HEADER_SIZE + CNMT_DIGEST_SIZE,
        "CNMT is truncated; expected a 0x20-byte header and a 0x20-byte digest (provide plaintext .cnmt, not an NCA)"
    );
    let kind = bytes[12];
    ensure!(
        kind != 0,
        "CNMT has the explicitly invalid content-meta type 0"
    );
    let extended_header_size = u16_at(bytes, 14)?;
    let content_count = u16_at(bytes, 16)? as usize;
    let meta_count = u16_at(bytes, 18)? as usize;
    let digest_offset = bytes.len() - CNMT_DIGEST_SIZE;
    let content_start = checked_table_end(
        CNMT_HEADER_SIZE,
        extended_header_size as usize,
        1,
        digest_offset,
    )?;
    let extended_header_bytes = &bytes[CNMT_HEADER_SIZE..content_start];
    let (extended_header, declared_data_size) = parse_extended_header(kind, extended_header_bytes)?;
    // Validate every range, including extended data, before allocating tables.
    let meta_start = checked_table_end(
        content_start,
        content_count,
        PACKAGED_CONTENT_INFO_SIZE,
        digest_offset,
    )?;
    let data_start = checked_table_end(
        meta_start,
        meta_count,
        CONTENT_META_INFO_SIZE,
        digest_offset,
    )?;
    let data_size = digest_offset - data_start;
    if let Some(expected_size) = declared_data_size {
        ensure!(
            data_size == expected_size,
            "CNMT extended-data length mismatch: declared {expected_size}, available {data_size}; truncated input or trailing bytes"
        );
    }

    let mut contents = Vec::with_capacity(content_count);
    for entry in bytes[content_start..meta_start]
        .as_chunks::<PACKAGED_CONTENT_INFO_SIZE>()
        .0
    {
        let mut raw_size = [0u8; 8];
        raw_size[..6].copy_from_slice(&entry[0x30..0x36]);
        let legacy_size = u64::from_le_bytes(raw_size);
        raw_size[5] = 0;
        contents.push(PackagedContentInfo {
            expected_sha256: hex(&entry[..0x20]),
            content_id: hex(&entry[0x20..0x30]),
            size_field: ContentSizeField {
                raw_six_bytes: hex(&entry[0x30..0x36]),
                pre_15_0_0_size_interpretation: legacy_size,
                since_15_0_0_size_interpretation: u64::from_le_bytes(raw_size),
                since_15_0_0_attributes_interpretation: entry[0x35],
            },
            content_type_raw: entry[0x36],
            content_type: content_type_name(entry[0x36]),
            id_offset: entry[0x37],
        });
    }
    let mut content_meta = Vec::with_capacity(meta_count);
    for entry in bytes[meta_start..data_start]
        .as_chunks::<CONTENT_META_INFO_SIZE>()
        .0
    {
        content_meta.push(ContentMetaInfo {
            title_id: u64_at(entry, 0, "ContentMetaInfo Id")?,
            version: u32_at(entry, 8, "ContentMetaInfo Version")?,
            content_meta_type_raw: entry[12],
            content_meta_type: meta_type_name(entry[12]),
            attributes_raw: entry[13],
        });
    }
    Ok(ContentMeta {
        format: "cnmt",
        file_size: bytes.len(),
        file_sha256: sha256(bytes),
        title_id: u64_at(bytes, 0, "CNMT Id")?,
        version: u32_at(bytes, 8, "CNMT Version")?,
        content_meta_type_raw: kind,
        content_meta_type: meta_type_name(kind),
        platform_raw: bytes[13],
        platform_interpretation_since_17_0_0: match bytes[13] {
            0 => "nx",
            1 => "ounce",
            _ => "unknown",
        },
        attributes_raw: bytes[20],
        required_download_system_version: u32_at(bytes, 24, "CNMT RequiredDownloadSystemVersion")?,
        header_raw: hex(&bytes[..CNMT_HEADER_SIZE]),
        extended_header_size,
        extended_header,
        extended_header_raw: hex(extended_header_bytes),
        contents,
        content_meta,
        extended_data: OpaqueRegion {
            offset: data_start,
            size: data_size,
            sha256: sha256(&bytes[data_start..digest_offset]),
            interpreted: false,
        },
        stored_digest: hex(&bytes[digest_offset..]),
        stored_digest_verified: false,
        metadata_authenticated: false,
        referenced_content_verified: false,
        interpretation_notes: "IDs and numeric version gates are declarations in this metadata; no title-ID arithmetic or display-version mapping is applied. ContentInfo's six-byte size area has distinct pre-15.0.0 and since-15.0.0 interpretations, both reported without selecting one. Extended data is opaque. The stored trailing digest may refer to a development representation retained in production, so it is not checked against this file's prefix. File SHA-256 fingerprints these input bytes only; no signatures or referenced content payloads have been verified.",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
        bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn cnmt(
        kind: u8,
        header: &[u8],
        contents: &[[u8; 0x38]],
        metas: &[[u8; 0x10]],
        data: &[u8],
    ) -> Vec<u8> {
        let mut bytes = vec![0u8; 0x20];
        put_u64(&mut bytes, 0, 0xf123_4567_89ab_cdef);
        put_u32(&mut bytes, 8, 327680);
        bytes[12] = kind;
        bytes[13] = 1;
        put_u16(&mut bytes, 14, header.len() as u16);
        put_u16(&mut bytes, 16, contents.len() as u16);
        put_u16(&mut bytes, 18, metas.len() as u16);
        bytes.extend_from_slice(header);
        for content in contents {
            bytes.extend_from_slice(content);
        }
        for meta in metas {
            bytes.extend_from_slice(meta);
        }
        bytes.extend_from_slice(data);
        // Deliberately not SHA256(prefix): a stored footer is not authentication.
        bytes.extend_from_slice(&[0xa5; 0x20]);
        bytes
    }

    #[test]
    fn nacp_version_and_ids_do_not_require_uncompressed_titles() {
        let mut bytes = vec![0; NACP_SIZE];
        bytes[..0x3000].fill(0xff);
        bytes[0x3215] = 1;
        bytes[0x3060..0x3065].copy_from_slice(b"2.0.1");
        put_u64(&mut bytes, 0x3070, 0xf123_4567_89ab_cdef);
        put_u64(&mut bytes, 0x3078, 0x0123_4567_89ab_cdef);
        let parsed = parse_nacp(&bytes).unwrap();
        assert_eq!(parsed.display_version, "2.0.1");
        assert!(parsed.display_version_null_terminated);
        assert_eq!(parsed.titles_data_format_raw, 1);
        assert_eq!(parsed.add_on_content_base_id, 0xf123_4567_89ab_cdef);
        assert_eq!(parsed.file_sha256, sha256(&bytes));
        assert!(!parsed.metadata_authenticated);
        let report = serde_json::to_value(parsed).unwrap();
        assert_eq!(report["add_on_content_base_id"], "f123456789abcdef");
        assert_eq!(report["save_data_owner_id"], "0123456789abcdef");
    }

    #[test]
    fn nacp_requires_exact_length_and_bounded_utf8_version() {
        let mut bytes = vec![0; NACP_SIZE];
        assert!(parse_nacp(&bytes[..NACP_SIZE - 1]).is_err());
        bytes.push(0);
        assert!(parse_nacp(&bytes).is_err());
        bytes.pop();
        bytes[0x3060] = 0xff;
        assert!(parse_nacp(&bytes).is_err());
        bytes[0x3060..0x3070].copy_from_slice(b"abcdefghijklmnop");
        let parsed = parse_nacp(&bytes).unwrap();
        assert_eq!(parsed.display_version, "abcdefghijklmnop");
        assert!(!parsed.display_version_null_terminated);
    }

    #[test]
    fn application_relationship_tables_and_size_ambiguity_are_explicit() {
        let mut header = [0; 0x10];
        put_u64(&mut header, 0, 0x1122_3344_5566_7788);
        put_u32(&mut header, 8, 0x1234_5678);
        put_u32(&mut header, 12, 19);
        let mut content = [0; 0x38];
        content[..0x20].fill(0x12);
        content[0x20..0x30].fill(0x34);
        content[0x30..0x36].copy_from_slice(&[0x11, 0x22, 0x33, 0x44, 0x55, 0x66]);
        content[0x36] = 3;
        content[0x37] = 7;
        let mut meta = [0; 0x10];
        put_u64(&mut meta, 0, 0x8877_6655_4433_2211);
        put_u32(&mut meta, 8, 23);
        meta[12] = 0x82;
        meta[13] = 9;
        let bytes = cnmt(0x80, &header, &[content], &[meta], &[]);
        let parsed = parse_cnmt(&bytes).unwrap();
        assert_eq!(parsed.application_id(), Some(parsed.title_id));
        assert_eq!(parsed.version, 327680);
        assert_eq!(parsed.platform_interpretation_since_17_0_0, "ounce");
        match &parsed.extended_header {
            ExtendedHeader::Application {
                patch_id,
                required_system_version,
                required_application_version,
            } => {
                assert_eq!(*patch_id, 0x1122_3344_5566_7788);
                assert_eq!(*required_system_version, 0x1234_5678);
                assert_eq!(*required_application_version, 19);
            }
            _ => panic!("wrong header"),
        }
        let entry = &parsed.contents[0];
        assert_eq!(entry.content_id, "34".repeat(16));
        assert_eq!(entry.expected_sha256, "12".repeat(32));
        assert_eq!(
            entry.size_field.pre_15_0_0_size_interpretation,
            0x6655_4433_2211
        );
        assert_eq!(
            entry.size_field.since_15_0_0_size_interpretation,
            0x55_4433_2211
        );
        assert_eq!(
            entry.size_field.since_15_0_0_attributes_interpretation,
            0x66
        );
        assert_eq!(entry.content_type, "control");
        assert_eq!(entry.id_offset, 7);
        assert_eq!(parsed.content_meta[0].title_id, 0x8877_6655_4433_2211);
        assert_eq!(parsed.content_meta[0].version, 23);
        assert_eq!(parsed.content_meta[0].attributes_raw, 9);
        assert_eq!(parsed.file_sha256, sha256(&bytes));
        assert_eq!(
            serde_json::to_value(parsed).unwrap()["title_id"],
            "f123456789abcdef"
        );
    }

    #[test]
    fn patch_bounds_opaque_data_and_does_not_authenticate_footer() {
        let mut header = [0; 0x18];
        put_u64(&mut header, 0, 0x1234);
        put_u32(&mut header, 8, 77);
        put_u32(&mut header, 12, 3);
        let mut bytes = cnmt(0x81, &header, &[], &[], &[1, 2, 3]);
        let parsed = parse_cnmt(&bytes).unwrap();
        assert_eq!(parsed.application_id(), Some(0x1234));
        assert_eq!(parsed.extended_data.offset, 0x38);
        assert_eq!(parsed.extended_data.size, 3);
        assert_eq!(parsed.extended_data.sha256, sha256(&[1, 2, 3]));
        assert_eq!(parsed.stored_digest, "a5".repeat(32));
        assert!(!parsed.stored_digest_verified);
        assert!(!parsed.metadata_authenticated);
        assert!(!parsed.referenced_content_verified);
        assert!(!parsed.extended_data.interpreted);
        bytes.push(0);
        assert!(parse_cnmt(&bytes).is_err());
        put_u32(&mut bytes, 0x20 + 12, u32::MAX);
        assert!(parse_cnmt(&bytes).is_err());
    }

    #[test]
    fn addon_layout_is_selected_by_header_length_not_content_version() {
        for size in [0x10, 0x18] {
            let mut header = vec![0; size];
            put_u64(&mut header, 0, 0x1020_3040_5060_7080);
            put_u32(&mut header, 8, 0x321);
            header[12] = 0x7f;
            if size == 0x18 {
                put_u64(&mut header, 16, 0x9091_9293_9495_9697);
            }
            let mut bytes = cnmt(0x82, &header, &[], &[], &[]);
            put_u32(&mut bytes, 8, 0); // Content versions do not encode the layout era.
            let parsed = parse_cnmt(&bytes).unwrap();
            assert_eq!(parsed.application_id(), Some(0x1020_3040_5060_7080));
            match parsed.extended_header {
                ExtendedHeader::AddOnContent {
                    layout,
                    required_application_version,
                    content_accessibilities,
                    data_patch_id,
                    ..
                } => {
                    assert_eq!(required_application_version, 0x321);
                    if size == 0x18 {
                        assert_eq!(layout, AddOnContentLayout::Since15_0_0Size0x18);
                        assert_eq!(content_accessibilities, Some(0x7f));
                        assert_eq!(data_patch_id, Some(0x9091_9293_9495_9697));
                    } else {
                        assert_eq!(layout, AddOnContentLayout::Legacy0x10);
                        assert_eq!(content_accessibilities, None);
                        assert_eq!(data_patch_id, None);
                    }
                }
                _ => panic!("wrong header"),
            }
        }
        assert!(parse_cnmt(&cnmt(0x82, &[0; 0x14], &[], &[], &[])).is_err());
    }

    #[test]
    fn data_patch_keeps_data_and_application_ids_distinct() {
        let mut header = [0; 0x20];
        put_u64(&mut header, 0, 0x1111_2222_3333_4444);
        put_u64(&mut header, 8, 0xaaaa_bbbb_cccc_dddd);
        put_u32(&mut header, 16, 0x10203);
        put_u32(&mut header, 20, 2);
        let parsed = parse_cnmt(&cnmt(0x84, &header, &[], &[], &[0x13, 0x37])).unwrap();
        assert_eq!(parsed.application_id(), Some(0xaaaa_bbbb_cccc_dddd));
        match parsed.extended_header {
            ExtendedHeader::DataPatch {
                data_id,
                application_id,
                required_application_version,
                extended_data_size,
            } => {
                assert_eq!(data_id, 0x1111_2222_3333_4444);
                assert_eq!(application_id, 0xaaaa_bbbb_cccc_dddd);
                assert_eq!(required_application_version, 0x10203);
                assert_eq!(extended_data_size, 2);
            }
            _ => panic!("wrong header"),
        }
    }

    #[test]
    fn truncation_and_forged_counts_are_rejected_before_table_allocation() {
        let bytes = cnmt(0x80, &[0; 0x10], &[[0; 0x38]], &[[0; 0x10]], &[]);
        for end in 0..bytes.len() {
            assert!(
                parse_cnmt(&bytes[..end]).is_err(),
                "accepted truncated length {end}"
            );
        }
        for offset in [14, 16, 18] {
            let mut malformed = bytes.clone();
            put_u16(&mut malformed, offset, u16::MAX);
            assert!(parse_cnmt(&malformed).is_err());
        }
        let mut malformed = bytes;
        malformed.push(0);
        assert!(parse_cnmt(&malformed).is_err());
        assert!(parse_cnmt(&vec![0; MAX_CNMT_SIZE + 1]).is_err());
        assert!(parse_cnmt(&cnmt(0, &[], &[], &[], &[])).is_err());
    }

    #[test]
    fn unknown_future_types_preserve_raw_regions_without_inferred_parent() {
        let bytes = cnmt(0xf1, &[0x42, 0x43, 0x44], &[], &[], &[0xab, 0xcd]);
        let parsed = parse_cnmt(&bytes).unwrap();
        assert_eq!(parsed.content_meta_type, "unknown");
        assert_eq!(parsed.application_id(), None);
        assert_eq!(parsed.extended_header_raw, "424344");
        assert_eq!(parsed.extended_data.size, 2);
        assert!(matches!(
            parsed.extended_header,
            ExtendedHeader::Uninterpreted {
                content_meta_type_raw: 0xf1
            }
        ));
    }

    #[test]
    fn delta_and_system_update_honor_declared_extended_data_sizes() {
        let mut header = [0; 0x10];
        put_u64(&mut header, 0, 0x123456);
        put_u32(&mut header, 8, 1);
        let parsed = parse_cnmt(&cnmt(0x83, &header, &[], &[], &[1])).unwrap();
        assert_eq!(parsed.application_id(), Some(0x123456));
        assert!(parse_cnmt(&cnmt(0x83, &header, &[], &[], &[])).is_err());
        let system = parse_cnmt(&cnmt(0x03, &2u32.to_le_bytes(), &[], &[], &[1, 2])).unwrap();
        assert_eq!(system.application_id(), None);
        assert_eq!(system.extended_data.size, 2);
        assert!(parse_cnmt(&cnmt(0x03, &2u32.to_le_bytes(), &[], &[], &[1])).is_err());
    }
}
