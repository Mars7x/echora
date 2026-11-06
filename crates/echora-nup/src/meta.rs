//! Package meta block (SPEC §5).

use crate::format::{self, Fields};
use crate::{NupError, Result};

/// Decoded package metadata. Text records that are not valid UTF-8 are kept
/// out of the string fields rather than lossily converted; their presence is
/// still visible through `record_types`.
#[derive(Debug, Clone, Default)]
pub struct PackageMeta {
    pub npdm: Option<Vec<u8>>,
    pub nacp: Option<Vec<u8>>,
    pub icon: Option<Vec<u8>>,
    pub display_version: Option<String>,
    pub title_name: Option<String>,
    pub publisher: Option<String>,
    pub build_id: Option<[u8; 32]>,
    pub source_dump: Option<String>,
    pub record_types: Vec<u16>,
}

pub(crate) fn parse(block: &[u8]) -> Result<PackageMeta> {
    let mut meta = PackageMeta::default();
    let mut pos = 0usize;
    while pos + 8 <= block.len() {
        let head = Fields(&block[pos..pos + 8]);
        let ty = head.u16(0);
        let size = head.u32(4) as usize;
        let start = pos + 8;
        let end = start
            .checked_add(size)
            .filter(|&e| e <= block.len())
            .ok_or_else(|| NupError::Format(format!("meta record {ty:#06x} overruns block")))?;
        let payload = &block[start..end];
        meta.record_types.push(ty);
        let text = || String::from_utf8(payload.to_vec()).ok();
        match ty {
            format::TLV_NPDM => meta.npdm = Some(payload.to_vec()),
            format::TLV_NACP => meta.nacp = Some(payload.to_vec()),
            format::TLV_ICON => meta.icon = Some(payload.to_vec()),
            format::TLV_DISPLAY_VERSION => meta.display_version = text(),
            format::TLV_TITLE_NAME => meta.title_name = text(),
            format::TLV_PUBLISHER => meta.publisher = text(),
            format::TLV_BUILD_ID => meta.build_id = payload.try_into().ok(),
            format::TLV_SOURCE_DUMP => meta.source_dump = text(),
            _ => {}
        }
        pos = end;
    }
    Ok(meta)
}
