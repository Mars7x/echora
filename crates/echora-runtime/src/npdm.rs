//! NPDM program metadata: process parameters, kernel capabilities and
//! service access control. Signatures are not checked.
//!
//! Adapted from pokopia-recomp `src/loader/npdm.rs` and the NPDM types in
//! `src/loader.rs`.

use serde::Serialize;

use crate::loader::{checked_range, read_regular_file, sha256, u32_at, u64_at};

pub const MAX_NPDM_FILE_SIZE: u64 = 1024 * 1024;

#[derive(Clone, Debug, Serialize)]
pub struct Npdm {
    pub program_id: u64,
    pub flags0: u8,
    pub system_resource_size: u32,
    pub main_thread_priority: u8,
    pub main_thread_core: u8,
    pub main_thread_stack_size: u32,
    pub is_64bit: bool,
    pub address_space_type: u8,
    pub acid_layout: String,
    pub file_sha256: String,
    pub kernel_capabilities: NpdmKernelCapabilities,
    /// ACI0 service access control: client service names this process may
    /// open through sm: (a trailing '*' is a prefix wildcard). Server
    /// registration entries are excluded.
    pub service_access: Vec<String>,
}

/// The CorePriority descriptor is named KernelFlags by the NPDM loader.
/// Priority numbers run in the opposite direction to scheduling precedence.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct KernelFlags {
    pub descriptor_index: usize,
    pub raw: u32,
    pub lowest_thread_priority: u8,
    pub highest_thread_priority: u8,
    pub minimum_core_id: u8,
    pub maximum_core_id: u8,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct KernelCapabilityRegion {
    pub descriptor_count: usize,
    pub padding_descriptor_count: usize,
    pub kernel_flags_descriptor_count: usize,
    /// ACID restrictions use the first matching descriptor, not a union.
    pub first_kernel_flags: Option<KernelFlags>,
    pub uninterpreted_descriptor_count: usize,
}

/// Bounded capability metadata, not a complete NPDM authorization check.
#[derive(Clone, Debug, Serialize)]
pub struct NpdmKernelCapabilities {
    pub aci0: KernelCapabilityRegion,
    pub acid: KernelCapabilityRegion,
    /// None means no active KernelFlags grant was present; it is not a default.
    pub active_core_mask: Option<u64>,
    pub active_priority_mask: Option<u64>,
    pub kernel_flags_ceiling_validated: bool,
    pub validation_scope: &'static str,
    pub other_capability_semantics_validated: bool,
    pub signature_authentication_verified: bool,
}

impl Default for NpdmKernelCapabilities {
    fn default() -> Self {
        Self {
            aci0: KernelCapabilityRegion::default(),
            acid: KernelCapabilityRegion::default(),
            active_core_mask: None,
            active_priority_mask: None,
            kernel_flags_ceiling_validated: false,
            validation_scope: "kernel_flags_only_when_present",
            other_capability_semantics_validated: false,
            signature_authentication_verified: false,
        }
    }
}

fn disjoint(ranges: &[(u64, u64, &str)], what: &str) -> Result<()> {
    let mut nonempty: Vec<_> = ranges
        .iter()
        .copied()
        .filter(|(_, size, _)| *size != 0)
        .collect();
    nonempty.sort_by_key(|entry| entry.0);
    for pair in nonempty.windows(2) {
        let (start, size, name) = pair[0];
        let (next, _, next_name) = pair[1];
        let end = start.checked_add(size).context("range overflow")?;
        ensure!(end <= next, "overlapping {what}: {name} and {next_name}");
    }
    Ok(())
}

/// Read a size-bounded regular input file, rejecting symlinks and special files.
/// Check the opened file's identity and size against the pre-open metadata.
use anyhow::{Context, Result, ensure};
use std::path::Path;

const META_SIZE: u64 = 0x80;
const ACCESS_HEADER_SIZE: u64 = 0x40;

pub fn load_npdm(path: &Path) -> Result<Npdm> {
    let bytes = read_regular_file(path, MAX_NPDM_FILE_SIZE)?;
    parse_npdm(&bytes).with_context(|| format!("load NPDM {}", path.display()))
}

fn parse_npdm(bytes: &[u8]) -> Result<Npdm> {
    ensure!(
        bytes.len() as u64 >= META_SIZE,
        "truncated NPDM META header"
    );
    ensure!(
        bytes.len() as u64 <= MAX_NPDM_FILE_SIZE,
        "NPDM exceeds input limit"
    );
    ensure!(&bytes[..4] == b"META", "missing NPDM META magic");
    let aci_offset = u32_at(bytes, 0x70, "ACI0 offset")? as u64;
    let aci_size = u32_at(bytes, 0x74, "ACI0 size")? as u64;
    let acid_offset = u32_at(bytes, 0x78, "ACID offset")? as u64;
    let acid_size = u32_at(bytes, 0x7c, "ACID size")? as u64;
    ensure!(
        aci_offset >= META_SIZE && acid_offset >= META_SIZE,
        "NPDM access-control region overlaps META"
    );
    ensure!(aci_size >= ACCESS_HEADER_SIZE, "truncated ACI0 header");
    let aci_range = checked_range(aci_offset, aci_size, bytes.len() as u64, "ACI0 region")?;
    let acid_range = checked_range(acid_offset, acid_size, bytes.len() as u64, "ACID region")?;
    disjoint(
        &[
            (aci_offset, aci_size, "ACI0"),
            (acid_offset, acid_size, "ACID"),
        ],
        "NPDM access-control regions",
    )?;
    let aci = &bytes[aci_range];
    let acid = &bytes[acid_range];
    ensure!(&aci[..4] == b"ACI0", "missing NPDM ACI0 magic");

    // The signature and public key occupy 0x200 bytes on the older layout and
    // 0x300 on the extended layout. Detect the body within the declared ACID
    // bounds; a magic sequence elsewhere in the file is not sufficient.
    let candidates: Vec<usize> = [0x200, 0x300]
        .into_iter()
        .filter(|&offset| {
            offset + ACCESS_HEADER_SIZE as usize <= acid.len()
                && &acid[offset..offset + 4] == b"ACID"
        })
        .collect();
    ensure!(
        candidates.len() == 1,
        "NPDM must contain exactly one supported ACID body (legacy 0x200 or extended 0x300)"
    );
    let body_offset = candidates[0];
    let acid_body = &acid[body_offset..body_offset + ACCESS_HEADER_SIZE as usize];

    validate_controls(aci, 0, ACCESS_HEADER_SIZE, "ACI0", 0x1c)?;
    validate_controls(
        acid,
        body_offset,
        body_offset as u64 + ACCESS_HEADER_SIZE,
        "ACID",
        0x2c,
    )?;

    let program_id = u64_at(aci, 0x10, "ACI0 program ID")?;
    let minimum = u64_at(acid_body, 0x10, "ACID minimum program ID")?;
    let maximum = u64_at(acid_body, 0x18, "ACID maximum program ID")?;
    ensure!(minimum <= maximum, "reversed ACID program-ID range");
    ensure!(
        (minimum..=maximum).contains(&program_id),
        "ACI0 program ID {program_id:#018x} is outside the ACID allowed range"
    );
    let kernel_capabilities = validate_kernel_flags(
        read_kernel_capabilities(aci, 0, "ACI0")?,
        read_kernel_capabilities(acid, body_offset, "ACID")?,
        bytes[0x0f],
        bytes[0x0e],
    )?;

    let service_access = read_service_access(aci)?;
    Ok(Npdm {
        service_access,
        program_id,
        flags0: bytes[0x0c],
        system_resource_size: u32_at(bytes, 0x14, "NPDM SystemResourceSize")?,
        main_thread_priority: bytes[0x0e],
        main_thread_core: bytes[0x0f],
        main_thread_stack_size: u32_at(bytes, 0x1c, "main thread stack size")?,
        is_64bit: bytes[0x0c] & 1 != 0,
        address_space_type: (bytes[0x0c] >> 1) & 7,
        acid_layout: if body_offset == 0x300 {
            "extended"
        } else {
            "legacy"
        }
        .to_owned(),
        file_sha256: sha256(bytes),
        kernel_capabilities,
    })
}

/// ACI0 service access control (offset/size at header 0x28/0x2c, relative to
/// ACI0). Each entry is a control byte (bits 0-2: name length - 1; bit 7:
/// server registration) followed by the ASCII name.
fn read_service_access(aci: &[u8]) -> Result<Vec<String>> {
    let offset = u32_at(aci, 0x28, "ACI0 service access offset")? as u64;
    let size = u32_at(aci, 0x2c, "ACI0 service access size")? as u64;
    if size == 0 {
        return Ok(Vec::new());
    }
    let range = checked_range(
        offset,
        size,
        aci.len() as u64,
        "ACI0 service access control",
    )?;
    let bytes = &aci[range];
    let mut names = Vec::new();
    let mut cursor = 0;
    while cursor < bytes.len() {
        let control = bytes[cursor];
        let length = usize::from(control & 7) + 1;
        let name = bytes
            .get(cursor + 1..cursor + 1 + length)
            .context("truncated ACI0 service access entry")?;
        ensure!(
            name.iter().all(|byte| byte.is_ascii_graphic()),
            "ACI0 service access entry is not a printable ASCII name"
        );
        if control & 0x80 == 0 {
            names.push(String::from_utf8_lossy(name).into_owned());
        }
        cursor += 1 + length;
    }
    Ok(names)
}

fn read_kernel_capabilities(
    parent: &[u8],
    body: usize,
    label: &str,
) -> Result<KernelCapabilityRegion> {
    // These offsets are relative to the containing ACI0/ACID region, including
    // the ACID signature area, rather than to the detected ACID body.
    let offset = u32_at(parent, body + 0x30, "kernel capabilities offset")? as u64;
    let size = u32_at(parent, body + 0x34, "kernel capabilities size")? as u64;
    let range = checked_range(offset, size, parent.len() as u64, label)?;
    ensure!(
        size.is_multiple_of(4),
        "{label} kernel capabilities are not complete words"
    );
    let mut region = KernelCapabilityRegion {
        descriptor_count: range.len() / 4,
        ..KernelCapabilityRegion::default()
    };
    // Iteration is bounded by the 1 MiB NPDM input limit; retain counts and the
    // first matching word without allocating a copy of every descriptor.
    for (index, bytes) in parent[range].as_chunks::<4>().0.iter().enumerate() {
        let word = u32::from_le_bytes(*bytes);
        if word == u32::MAX {
            region.padding_descriptor_count += 1;
        } else if word & 0xf == 0x7 {
            region.kernel_flags_descriptor_count += 1;
            region.first_kernel_flags.get_or_insert(KernelFlags {
                descriptor_index: index,
                raw: word,
                lowest_thread_priority: ((word >> 4) & 0x3f) as u8,
                highest_thread_priority: ((word >> 10) & 0x3f) as u8,
                minimum_core_id: ((word >> 16) & 0xff) as u8,
                maximum_core_id: (word >> 24) as u8,
            });
        } else {
            region.uninterpreted_descriptor_count += 1;
        }
    }
    Ok(region)
}

fn inclusive_mask(minimum: u8, maximum: u8) -> u64 {
    // Callers have checked minimum <= maximum < 64. This also handles bit 63
    // and the full 0..=63 range without shifting by the type's width.
    (u64::MAX << minimum) & (u64::MAX >> (63 - maximum))
}

fn validate_kernel_flags(
    aci0: KernelCapabilityRegion,
    acid: KernelCapabilityRegion,
    main_core: u8,
    main_priority: u8,
) -> Result<NpdmKernelCapabilities> {
    // Primary model: Atmosphere 6b0c077b4c608a3a2ee712b8dc35fee87ceb62e8,
    // stratosphere/loader/source/ldr_capabilities.cpp (first ACID ceiling),
    // libraries/libmesosphere/source/kern_k_capabilities.cpp (user grants),
    // include/mesosphere/kern_select_cpu.hpp (64 virtual, not physical, cores),
    // and source/svc/kern_svc_process.cpp (main-thread membership).
    ensure!(
        aci0.kernel_flags_descriptor_count <= 1,
        "duplicate ACI0 KernelFlags descriptors"
    );
    let mut capabilities = NpdmKernelCapabilities {
        aci0,
        acid,
        ..NpdmKernelCapabilities::default()
    };
    let Some(grant) = capabilities.aci0.first_kernel_flags.as_ref() else {
        // Keep metadata inspection possible, but bootstrap must refuse to
        // execute a process without validated active permissions.
        return Ok(capabilities);
    };
    ensure!(
        grant.minimum_core_id <= grant.maximum_core_id,
        "reversed ACI0 KernelFlags core range"
    );
    ensure!(
        grant.maximum_core_id < 64,
        "ACI0 KernelFlags core ID exceeds the 64 virtual-core limit"
    );
    ensure!(
        grant.highest_thread_priority <= grant.lowest_thread_priority,
        "reversed ACI0 KernelFlags priority range"
    );
    ensure!(
        grant.highest_thread_priority >= 4,
        "ACI0 KernelFlags grants kernel-only priorities 0 through 3"
    );
    let ceiling = capabilities
        .acid
        .first_kernel_flags
        .as_ref()
        .context("ACI0 KernelFlags has no matching ACID ceiling")?;
    ensure!(
        grant.minimum_core_id >= ceiling.minimum_core_id
            && grant.maximum_core_id <= ceiling.maximum_core_id
            && grant.highest_thread_priority >= ceiling.highest_thread_priority
            && grant.lowest_thread_priority <= ceiling.lowest_thread_priority,
        "ACI0 KernelFlags grant exceeds the first ACID ceiling"
    );
    // A restriction ceiling may mention priorities 0..3 or core IDs above 63;
    // only the active user grant is subject to the kernel's mask limits.
    let core_mask = inclusive_mask(grant.minimum_core_id, grant.maximum_core_id);
    let priority_mask = inclusive_mask(grant.highest_thread_priority, grant.lowest_thread_priority);
    ensure!(
        main_core < 64 && core_mask & (1u64 << main_core) != 0,
        "META main thread core is outside the ACI0 KernelFlags grant"
    );
    ensure!(
        main_priority < 64 && priority_mask & (1u64 << main_priority) != 0,
        "META main thread priority is outside the ACI0 KernelFlags grant"
    );
    capabilities.active_core_mask = Some(core_mask);
    capabilities.active_priority_mask = Some(priority_mask);
    capabilities.kernel_flags_ceiling_validated = true;
    Ok(capabilities)
}

fn validate_controls(
    parent: &[u8],
    body: usize,
    minimum_offset: u64,
    label: &str,
    minimum_fs_size: u64,
) -> Result<()> {
    let names = ["filesystem access", "service access", "kernel capabilities"];
    let mut ranges = Vec::with_capacity(3);
    for (index, name) in names.iter().enumerate() {
        let field = body + 0x20 + index * 8;
        let offset = u32_at(parent, field, "access-control offset")? as u64;
        let size = u32_at(parent, field + 4, "access-control size")? as u64;
        checked_range(offset, size, parent.len() as u64, name)
            .with_context(|| format!("{label} {name}"))?;
        if size == 0 {
            ensure!(
                offset == 0 || offset >= minimum_offset,
                "empty {label} {name} has an offset inside its header"
            );
        } else {
            ensure!(
                offset >= minimum_offset,
                "{label} {name} overlaps its header"
            );
            if index == 0 {
                ensure!(
                    size >= minimum_fs_size,
                    "truncated {label} filesystem access structure"
                );
            } else if index == 2 {
                ensure!(
                    size.is_multiple_of(4),
                    "{label} kernel capabilities size is not a multiple of four"
                );
            }
        }
        ranges.push((offset, size, *name));
    }
    disjoint(&ranges, label)
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;
    fn put32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put64(bytes: &mut [u8], offset: usize, value: u64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn fixture(extended: bool) -> Vec<u8> {
        let acid_offset = 0x80;
        let body = if extended { 0x300 } else { 0x200 };
        let acid_size = body + 0x80;
        let aci_offset = acid_offset + acid_size;
        let mut bytes = vec![0; aci_offset + 0x80];
        bytes[..4].copy_from_slice(b"META");
        bytes[0x0c] = 7;
        bytes[0x0e] = 0x2c;
        put32(&mut bytes, 0x1c, 0x100000);
        put32(&mut bytes, 0x70, aci_offset as u32);
        put32(&mut bytes, 0x74, 0x80);
        put32(&mut bytes, 0x78, acid_offset as u32);
        put32(&mut bytes, 0x7c, acid_size as u32);
        bytes[aci_offset..aci_offset + 4].copy_from_slice(b"ACI0");
        put64(&mut bytes, aci_offset + 0x10, 0x1234);
        put32(&mut bytes, aci_offset + 0x20, 0x40);
        put32(&mut bytes, aci_offset + 0x24, 0x1c);
        let header = acid_offset + body;
        bytes[header..header + 4].copy_from_slice(b"ACID");
        put64(&mut bytes, header + 0x10, 0x1234);
        put64(&mut bytes, header + 0x18, 0x1234);
        put32(&mut bytes, header + 0x20, (body + 0x40) as u32);
        put32(&mut bytes, header + 0x24, 0x2c);
        bytes
    }

    fn kernel_flags(min_core: u8, max_core: u8, highest: u8, lowest: u8) -> u32 {
        assert!(highest < 64 && lowest < 64);
        0x7 | ((lowest as u32) << 4)
            | ((highest as u32) << 10)
            | ((min_core as u32) << 16)
            | ((max_core as u32) << 24)
    }

    fn capability_fixture(extended: bool, aci_words: &[u32], acid_words: &[u32]) -> Vec<u8> {
        assert!(aci_words.len() <= 8 && acid_words.len() <= 4);
        let mut bytes = fixture(extended);
        let aci = u32_at(&bytes, 0x70, "ACI0").unwrap() as usize;
        let acid = u32_at(&bytes, 0x78, "ACID").unwrap() as usize;
        let body = if extended { 0x300 } else { 0x200 };
        for (parent, header, offset, words) in [
            (aci, aci, 0x60, aci_words),
            (acid, acid + body, body + 0x70, acid_words),
        ] {
            put32(&mut bytes, header + 0x30, offset as u32);
            put32(&mut bytes, header + 0x34, (words.len() * 4) as u32);
            for (index, word) in words.iter().enumerate() {
                put32(&mut bytes, parent + offset + index * 4, *word);
            }
        }
        bytes
    }

    #[test]
    fn kernel_flags_decode_bitfields_and_report_limited_validation() {
        let grant = kernel_flags(3, 9, 5, 59);
        let ceiling = kernel_flags(2, 12, 0, 63);
        for extended in [false, true] {
            let mut bytes = capability_fixture(extended, &[u32::MAX, 0, grant], &[0, ceiling]);
            bytes[0x0f] = 4;
            let npdm = parse_npdm(&bytes).unwrap();
            let caps = &npdm.kernel_capabilities;
            assert_eq!(caps.active_core_mask, Some(0x3f8));
            assert_eq!(caps.active_priority_mask, Some(((1u64 << 60) - 1) & !31));
            assert_eq!(
                caps.aci0.first_kernel_flags,
                Some(KernelFlags {
                    descriptor_index: 2,
                    raw: grant,
                    lowest_thread_priority: 59,
                    highest_thread_priority: 5,
                    minimum_core_id: 3,
                    maximum_core_id: 9,
                })
            );
            assert_eq!(caps.aci0.descriptor_count, 3);
            assert_eq!(caps.aci0.padding_descriptor_count, 1);
            assert_eq!(caps.aci0.uninterpreted_descriptor_count, 1);
            assert_eq!(caps.aci0.kernel_flags_descriptor_count, 1);
            assert_eq!(caps.acid.descriptor_count, 2);
            assert_eq!(
                caps.acid
                    .first_kernel_flags
                    .as_ref()
                    .unwrap()
                    .descriptor_index,
                1
            );
            assert!(caps.kernel_flags_ceiling_validated);
            assert_eq!(caps.validation_scope, "kernel_flags_only_when_present");
            assert!(!caps.other_capability_semantics_validated);
            assert!(!caps.signature_authentication_verified);
        }
    }

    #[test]
    fn active_masks_handle_bit_63_and_allow_a_wider_acid_ceiling() {
        let ceiling = kernel_flags(0, 255, 0, 63);
        let mut bytes = capability_fixture(false, &[kernel_flags(0, 63, 4, 63)], &[ceiling]);
        bytes[0x0f] = 63;
        bytes[0x0e] = 63;
        let caps = parse_npdm(&bytes).unwrap().kernel_capabilities;
        assert_eq!(caps.active_core_mask, Some(u64::MAX));
        assert_eq!(caps.active_priority_mask, Some(!0xfu64));
        assert_eq!(caps.acid.first_kernel_flags.unwrap().maximum_core_id, 255);
        let bytes = capability_fixture(true, &[kernel_flags(0, 64, 4, 63)], &[ceiling]);
        assert!(
            parse_npdm(&bytes)
                .unwrap_err()
                .to_string()
                .contains("64 virtual-core")
        );
    }

    #[test]
    fn rejects_duplicate_reversed_or_kernel_priority_active_grants() {
        let ceiling = kernel_flags(0, 255, 0, 63);
        for (grant, expected) in [
            (kernel_flags(5, 4, 4, 63), "reversed ACI0 KernelFlags core"),
            (
                kernel_flags(0, 5, 50, 40),
                "reversed ACI0 KernelFlags priority",
            ),
            (kernel_flags(0, 5, 3, 63), "kernel-only priorities"),
        ] {
            let bytes = capability_fixture(false, &[grant], &[ceiling]);
            assert!(
                parse_npdm(&bytes)
                    .unwrap_err()
                    .to_string()
                    .contains(expected)
            );
        }
        let grant = kernel_flags(0, 5, 28, 59);
        let bytes = capability_fixture(true, &[grant, grant], &[ceiling]);
        assert!(
            parse_npdm(&bytes)
                .unwrap_err()
                .to_string()
                .contains("duplicate ACI0")
        );
    }

    #[test]
    fn grants_must_fit_the_first_acid_kernel_flags_ceiling() {
        let ceiling = kernel_flags(2, 7, 12, 50);
        let broad_ceiling = kernel_flags(0, 255, 0, 63);
        for grant in [
            kernel_flags(1, 5, 12, 50),
            kernel_flags(2, 8, 12, 50),
            kernel_flags(2, 7, 11, 50),
            kernel_flags(2, 7, 12, 51),
        ] {
            let mut bytes = capability_fixture(true, &[grant], &[ceiling, broad_ceiling]);
            bytes[0x0f] = 3;
            assert!(
                parse_npdm(&bytes)
                    .unwrap_err()
                    .to_string()
                    .contains("first ACID ceiling")
            );
        }
        let grant = kernel_flags(3, 6, 14, 48);
        let mut bytes = capability_fixture(false, &[grant], &[ceiling, broad_ceiling]);
        bytes[0x0f] = 3;
        let caps = parse_npdm(&bytes).unwrap().kernel_capabilities;
        assert_eq!(caps.active_core_mask, Some(0x78));
        assert_eq!(caps.acid.kernel_flags_descriptor_count, 2);
        assert_eq!(caps.acid.first_kernel_flags.unwrap().raw, ceiling);
        // A later narrower ceiling does not replace an accepted first ceiling.
        let mut bytes =
            capability_fixture(false, &[grant], &[broad_ceiling, kernel_flags(0, 0, 4, 4)]);
        bytes[0x0f] = 3;
        assert!(parse_npdm(&bytes).is_ok());
    }

    #[test]
    fn missing_kernel_flags_never_invent_permissions() {
        let grant = kernel_flags(0, 5, 28, 59);
        for bytes in [
            fixture(false),
            capability_fixture(true, &[u32::MAX, 0], &[grant]),
        ] {
            let caps = parse_npdm(&bytes).unwrap().kernel_capabilities;
            assert_eq!(caps.active_core_mask, None);
            assert_eq!(caps.active_priority_mask, None);
            assert!(!caps.kernel_flags_ceiling_validated);
        }
        let bytes = capability_fixture(false, &[grant], &[]);
        assert!(
            parse_npdm(&bytes)
                .unwrap_err()
                .to_string()
                .contains("no matching ACID ceiling")
        );
    }

    #[test]
    fn main_thread_must_belong_to_active_core_and_priority_masks() {
        let grant = kernel_flags(0, 5, 28, 59);
        for (core, priority, expected) in [
            (6, 44, "main thread core"),
            (64, 44, "main thread core"),
            (255, 44, "main thread core"),
            (0, 27, "main thread priority"),
            (0, 60, "main thread priority"),
            (0, 255, "main thread priority"),
        ] {
            let mut bytes = capability_fixture(true, &[grant], &[grant]);
            bytes[0x0f] = core;
            bytes[0x0e] = priority;
            assert!(
                parse_npdm(&bytes)
                    .unwrap_err()
                    .to_string()
                    .contains(expected)
            );
        }
    }

    #[test]
    fn kernel_capability_regions_are_bounded_disjoint_word_sequences() {
        let grant = kernel_flags(0, 5, 28, 59);
        for (is_acid, offset, size, expected) in [
            (false, 0x60, 3, "multiple of four"),
            (false, 0x7f, 4, "outside its containing region"),
            (true, 0x37e, 4, "outside its containing region"),
            (false, 0x20, 4, "overlaps its header"),
            (false, 0x50, 4, "overlapping"),
        ] {
            let mut bytes = capability_fixture(true, &[grant], &[grant]);
            let header = if is_acid {
                0x80 + 0x300
            } else {
                u32_at(&bytes, 0x70, "ACI0").unwrap() as usize
            };
            put32(&mut bytes, header + 0x30, offset);
            put32(&mut bytes, header + 0x34, size);
            let error = parse_npdm(&bytes).unwrap_err();
            assert!(format!("{error:#}").contains(expected), "{error:#}");
        }
    }

    #[test]
    fn accepts_both_acid_layouts_and_reads_process_fields() {
        for extended in [false, true] {
            let npdm = parse_npdm(&fixture(extended)).unwrap();
            assert_eq!(npdm.program_id, 0x1234);
            assert_eq!(npdm.flags0, 7);
            assert_eq!(npdm.system_resource_size, 0);
            assert_eq!(npdm.main_thread_priority, 0x2c);
            assert_eq!(npdm.main_thread_stack_size, 0x100000);
            assert!(npdm.is_64bit);
            assert_eq!(npdm.address_space_type, 3);
            assert_eq!(
                npdm.acid_layout,
                if extended { "extended" } else { "legacy" }
            );
        }
    }

    #[test]
    fn rejects_truncated_and_overlapping_access_regions() {
        let mut bytes = fixture(true);
        bytes.pop();
        assert!(parse_npdm(&bytes).is_err());
        let mut bytes = fixture(true);
        put32(&mut bytes, 0x70, 0x80);
        assert!(
            parse_npdm(&bytes)
                .unwrap_err()
                .to_string()
                .contains("overlapping")
        );
    }

    #[test]
    fn nested_ranges_cannot_escape_their_own_parent() {
        let mut bytes = fixture(true);
        // This address is inside the complete file, but outside the ACID region.
        put32(&mut bytes, 0x80 + 0x300 + 0x20, 0x390);
        assert!(parse_npdm(&bytes).is_err());
        let mut bytes = fixture(false);
        let aci = u32_at(&bytes, 0x70, "ACI").unwrap() as usize;
        put32(&mut bytes, aci + 0x20, 0x20);
        assert!(
            parse_npdm(&bytes)
                .unwrap_err()
                .to_string()
                .contains("overlaps its header")
        );
    }

    #[test]
    fn rejects_ambiguous_acid_body_and_invalid_program_range() {
        let mut bytes = fixture(true);
        bytes[0x80 + 0x200..0x80 + 0x204].copy_from_slice(b"ACID");
        assert!(
            parse_npdm(&bytes)
                .unwrap_err()
                .to_string()
                .contains("exactly one")
        );
        let mut bytes = fixture(false);
        put64(&mut bytes, 0x80 + 0x200 + 0x18, 0x1233);
        assert!(parse_npdm(&bytes).is_err());
    }

    #[test]
    fn requires_meta_and_aci_magic() {
        let mut bytes = fixture(false);
        bytes[0] = 0;
        assert!(parse_npdm(&bytes).is_err());
        let mut bytes = fixture(true);
        let aci = u32_at(&bytes, 0x70, "ACI").unwrap() as usize;
        bytes[aci + 3] = b'D';
        assert!(
            parse_npdm(&bytes)
                .unwrap_err()
                .to_string()
                .contains("ACI0 magic")
        );
    }
}
