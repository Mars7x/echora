//! Process image layout and static discovery roots.
//!
//! Modules are placed at consecutive page-aligned bases in load order, as
//! the guest's loader does; the guest rtld relocates them at startup. The
//! root helpers find candidate code addresses for ahead-of-time discovery.
//! A root is only a candidate: it does not claim the guest will run it.
//!
//! Root extraction is adapted from pokopia-recomp `src/diagnostics.rs`.

use crate::nso::NsoModule;

/// Base of the first module (rtld) in the modeled 39-bit address space.
pub const FIRST_MODULE_BASE: u64 = 0x71_0000_0000;
/// `PACIBSP`: the prologue this toolchain emits for non-leaf functions.
pub const PACIBSP: u32 = 0xd503_237f;
const NOP: u32 = 0xd503_201f;

const R_AARCH64_RELATIVE: u32 = 1027;
const R_AARCH64_JUMP_SLOT: u32 = 1026;
const RELA_ENTRY_SIZE: usize = 24;

#[derive(Debug, Clone)]
pub struct PlacedModule {
    pub base: u64,
    pub module: NsoModule,
}

impl PlacedModule {
    /// Absolute address range of the text segment.
    pub fn text_range(&self) -> std::ops::Range<u64> {
        let start = self.base + u64::from(self.module.text.memory_offset);
        start..start + self.module.text.bytes.len() as u64
    }

    /// Absolute address of the rodata segment.
    pub fn rodata_base(&self) -> u64 {
        self.base + u64::from(self.module.rodata.memory_offset)
    }
}

/// Places modules (already in load order) one after another, each starting
/// at the previous module's page-rounded image end.
pub fn place_modules(modules: Vec<NsoModule>) -> Vec<PlacedModule> {
    let mut base = FIRST_MODULE_BASE;
    modules
        .into_iter()
        .map(|module| {
            let placed = PlacedModule { base, module };
            base += placed.module.image_size;
            placed
        })
        .collect()
}

/// Bytes at module offset `[offset, offset + size)` if they lie within one
/// initialized segment (relocation targets are read from original bytes).
fn original_view(module: &NsoModule, offset: u64, size: u64) -> Option<&[u8]> {
    let end = offset.checked_add(size)?;
    if end > module.image_size {
        return None;
    }
    for segment in module.segments() {
        let start = u64::from(segment.memory_offset);
        let seg_end = start.checked_add(segment.bytes.len() as u64)?;
        if offset >= start && end <= seg_end {
            let begin = usize::try_from(offset - start).ok()?;
            return segment
                .bytes
                .get(begin..begin.checked_add(usize::try_from(size).ok()?)?);
        }
    }
    None
}

fn word(bytes: &[u8], offset: usize) -> Option<u64> {
    Some(u64::from_le_bytes(
        bytes.get(offset..offset.checked_add(8)?)?.try_into().ok()?,
    ))
}

/// `(module-relative slot, absolute text target)` for every relative
/// relocation whose target is 4-byte aligned in this module's text, from RELA
/// (`R_AARCH64_RELATIVE`, symbol zero) and packed RELR tables.
pub fn relative_text_slots(placed: &PlacedModule) -> Vec<(u64, u64)> {
    let module = &placed.module;
    let text_start = u64::from(module.text.memory_offset);
    let text = text_start..text_start + module.text.bytes.len() as u64;
    let Some(dynamic) = module.mod0.as_ref().and_then(|m| m.dynamic.as_ref()) else {
        return Vec::new();
    };
    let mut slots = Vec::new();
    for table in &dynamic.relocation_tables {
        let Some(bytes) = original_view(module, table.memory_offset, table.byte_size) else {
            continue;
        };
        match table.kind.as_str() {
            "RELA" => {
                for entry in bytes.as_chunks::<RELA_ENTRY_SIZE>().0 {
                    let (Some(offset), Some(info), Some(addend)) =
                        (word(entry, 0), word(entry, 8), word(entry, 16))
                    else {
                        continue;
                    };
                    if info as u32 == R_AARCH64_RELATIVE
                        && info >> 32 == 0
                        && addend % 4 == 0
                        && text.contains(&addend)
                    {
                        slots.push((offset, placed.base + addend));
                    }
                }
            }
            // An even word is the next slot offset; an odd word is a bitmap
            // over the 63 slots after the cursor. The relocated value is the
            // stored word plus the load base.
            "RELR" if table.entry_size == 8 => {
                let mut push = |slot: u64| {
                    if let Some(value) = original_view(module, slot, 8).and_then(|b| word(b, 0))
                        && value % 4 == 0
                        && text.contains(&value)
                    {
                        slots.push((slot, placed.base + value));
                    }
                };
                let mut cursor = 0u64;
                for chunk in bytes.as_chunks::<8>().0 {
                    let entry = word(chunk, 0).unwrap_or(0);
                    if entry & 1 == 0 {
                        push(entry);
                        cursor = entry.wrapping_add(8);
                    } else {
                        for bit in 1..64 {
                            if entry >> bit & 1 != 0 {
                                push(cursor.wrapping_add((bit - 1) * 8));
                            }
                        }
                        cursor = cursor.wrapping_add(63 * 8);
                    }
                }
            }
            _ => {}
        }
    }
    slots
}

/// Distinct relative-relocation targets in text, sorted.
pub fn relative_text_targets(placed: &PlacedModule) -> Vec<u64> {
    let mut targets: Vec<u64> = relative_text_slots(placed)
        .into_iter()
        .map(|(_, t)| t)
        .collect();
    targets.sort_unstable();
    targets.dedup();
    targets
}

/// Exported functions in this module's text: entry points other modules
/// call through symbol-resolved GOT slots.
pub fn exported_function_targets(placed: &PlacedModule) -> Vec<u64> {
    let text = placed.text_range();
    crate::mod0::defined_functions(&placed.module)
        .unwrap_or_default()
        .into_iter()
        .map(|offset| placed.base + offset)
        .filter(|address| text.contains(address))
        .collect()
}

/// Lazy-binding stubs: the original values of `JUMP_SLOT` GOT entries that
/// point into this module's text. Until the first call resolves an import,
/// rtld leaves its slot holding this value plus the load base, so the PLT
/// jumps there (normally PLT0) and nothing else names the address.
pub fn lazy_binding_targets(placed: &PlacedModule) -> Vec<u64> {
    let module = &placed.module;
    let text_start = u64::from(module.text.memory_offset);
    let text = text_start..text_start + module.text.bytes.len() as u64;
    let Some(dynamic) = module.mod0.as_ref().and_then(|m| m.dynamic.as_ref()) else {
        return Vec::new();
    };
    let mut targets = Vec::new();
    for table in &dynamic.relocation_tables {
        if table.kind != "JMPREL/RELA" {
            continue;
        }
        let Some(bytes) = original_view(module, table.memory_offset, table.byte_size) else {
            continue;
        };
        for entry in bytes.as_chunks::<RELA_ENTRY_SIZE>().0 {
            let (Some(slot), Some(info)) = (word(entry, 0), word(entry, 8)) else {
                continue;
            };
            if info as u32 == R_AARCH64_JUMP_SLOT
                && let Some(value) = original_view(module, slot, 8).and_then(|b| word(b, 0))
                && value % 4 == 0
                && text.contains(&value)
            {
                targets.push(placed.base + value);
            }
        }
    }
    targets.sort_unstable();
    targets.dedup();
    targets
}

/// Static constructors: `DT_INIT_ARRAY` slots whose relative relocation
/// targets this module's text, in array order.
pub fn init_array_targets(placed: &PlacedModule) -> Vec<u64> {
    let Some(array) = placed
        .module
        .mod0
        .as_ref()
        .and_then(|m| m.dynamic.as_ref())
        .and_then(|d| d.init_array.as_ref())
    else {
        return Vec::new();
    };
    let range = array.memory_offset..array.memory_offset + array.byte_size;
    let mut slots: Vec<_> = relative_text_slots(placed)
        .into_iter()
        .filter(|(slot, _)| range.contains(slot))
        .collect();
    slots.sort_unstable();
    slots.into_iter().map(|(_, t)| t).collect()
}

/// Whether `word` ends straight-line code (return, unconditional branch,
/// trap, padding), so the next word can begin a function.
fn ends_code(word: u32) -> bool {
    word == 0
        || word == NOP
        || word & 0xfc00_0000 == 0x1400_0000 // B
        || word & 0xffff_fc1f == 0xd61f_0000 // BR
        || word & 0xffff_fc1f == 0xd65f_0000 // RET
        || word == 0xd65f_0bff // RETAA
        || word == 0xd65f_0fff // RETAB
        || word & 0xffe0_001f == 0xd420_0000 // BRK
}

/// Text addresses the module computes in code with `ADR`, or `ADRP` and a
/// following `ADD` to the same register, that sit where a function can
/// begin. Callbacks without a `PACIBSP` prologue and without a relocation
/// (leaf functions passed by address) are reached only this way.
pub fn code_address_targets(placed: &PlacedModule) -> Vec<u64> {
    let start = placed.text_range().start;
    let words: Vec<u32> = placed
        .module
        .text
        .bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect();
    let size = words.len() as u64 * 4;
    let mut targets = Vec::new();
    for (i, &word) in words.iter().enumerate() {
        let pc = i as u64 * 4;
        // ADR and ADRP share the 21-bit immhi:immlo field.
        let imm = ((word >> 5 & 0x7ffff) << 2 | word >> 29 & 3) as i64;
        let imm = imm << 43 >> 43;
        let target = match word & 0x9f00_0000 {
            0x1000_0000 => (pc as i64).wrapping_add(imm) as u64,
            0x9000_0000 => {
                let rd = word & 31;
                let page = ((pc & !0xfff) as i64).wrapping_add(imm << 12) as u64;
                let add = words[i + 1..words.len().min(i + 8)]
                    .iter()
                    .take_while(|&&next| {
                        next & 0xffc0_0000 == 0x9100_0000 && next >> 5 & 31 == rd || next & 31 != rd
                    })
                    .find(|&&next| next & 0xffc0_0000 == 0x9100_0000 && next >> 5 & 31 == rd);
                let Some(add) = add else { continue };
                page.wrapping_add(u64::from(add >> 10 & 0xfff))
            }
            _ => continue,
        };
        if target % 4 == 0
            && target < size
            && (target == 0 || ends_code(words[target as usize / 4 - 1]))
        {
            targets.push(start + target);
        }
    }
    targets.sort_unstable();
    targets.dedup();
    targets
}

/// Addresses of every `PACIBSP` in the module's text.
pub fn prologue_targets(placed: &PlacedModule) -> Vec<u64> {
    let start = placed.text_range().start;
    placed
        .module
        .text
        .bytes
        .as_chunks::<4>()
        .0
        .iter()
        .enumerate()
        .filter(|(_, w)| u32::from_le_bytes([w[0], w[1], w[2], w[3]]) == PACIBSP)
        .map(|(i, _)| start + i as u64 * 4)
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::mod0::{DynamicArrayInfo, DynamicInfo, Mod0Info, RelocationTableInfo};
    use crate::nso::{Compression, Segment};

    fn segment(offset: u32, bytes: Vec<u8>) -> Segment {
        Segment {
            memory_offset: offset,
            file_offset: 0,
            stored_size: bytes.len() as u32,
            compression: Compression::None,
            sha256: [0; 32],
            hash_verified: false,
            bytes,
        }
    }

    fn table(kind: &str, offset: u64, size: u64, entry: u64) -> RelocationTableInfo {
        RelocationTableInfo {
            kind: kind.into(),
            memory_offset: offset,
            byte_size: size,
            entry_size: entry,
            count: size / entry,
            types: Default::default(),
        }
    }

    /// Text 0x0..0x100 (a PACIBSP at 0x40), data 0x1000..0x1100 holding
    /// function pointers, a RELR table at 0x1080 and a RELA table at 0x10c0.
    fn module() -> NsoModule {
        let mut text = vec![0u8; 0x100];
        text[0x40..0x44].copy_from_slice(&PACIBSP.to_le_bytes());
        let code =
            |t: &mut Vec<u8>, at: usize, w: u32| t[at..at + 4].copy_from_slice(&w.to_le_bytes());
        code(&mut text, 0x00, 0x1000_0100); // adr x0, 0x20 (after padding)
        code(&mut text, 0x04, 0x9000_0001); // adrp x1, 0
        code(&mut text, 0x08, 0x9100_c021); // add x1, x1, #0x30 (after ret)
        code(&mut text, 0x0c, 0x1000_0042); // adr x2, 0x14 (after a mov: not a start)
        code(&mut text, 0x10, 0xaa04_03e3); // mov x3, x4
        code(&mut text, 0x2c, 0xd65f_03c0); // ret
        let mut data = vec![0u8; 0x100];
        let put =
            |d: &mut Vec<u8>, at: usize, v: u64| d[at..at + 8].copy_from_slice(&v.to_le_bytes());
        put(&mut data, 0x00, 0x40); // slot 0x1000 -> text 0x40 (init_array[0])
        put(&mut data, 0x08, 0x80); // slot 0x1008 -> text 0x80 (init_array[1])
        put(&mut data, 0x18, 0x81); // misaligned: ignored
        put(&mut data, 0x20, 0x2000); // outside text: ignored
        // RELR: slot 0x1000, then bitmap covering 0x1008 (bit 1), 0x1018 (bit 3), 0x1020 (bit 4).
        put(&mut data, 0x80, 0x1000);
        put(&mut data, 0x88, 1 | 1 << 1 | 1 << 3 | 1 << 4);
        // RELA: R_AARCH64_RELATIVE at 0x1030 with addend 0xc0.
        put(&mut data, 0xc0, 0x1030);
        put(&mut data, 0xc8, u64::from(R_AARCH64_RELATIVE));
        put(&mut data, 0xd0, 0xc0);
        // JMPREL: R_AARCH64_JUMP_SLOT at 0x1028, whose lazy value is text 0x60.
        put(&mut data, 0x28, 0x60);
        put(&mut data, 0xe0, 0x1028);
        put(&mut data, 0xe8, 1 << 32 | u64::from(R_AARCH64_JUMP_SLOT));
        NsoModule {
            name: "m".into(),
            build_id: [0; 32],
            flags: 0,
            text: segment(0, text),
            rodata: segment(0x100, Vec::new()),
            data: segment(0x1000, data),
            bss_size: 0,
            image_size: 0x2000,
            mod0: Some(Mod0Info {
                memory_offset: 0,
                dynamic_offset: None,
                bss_start: 0,
                bss_end: 0,
                unwind_start: 0,
                unwind_end: 0,
                module_object_offset: None,
                dynamic: Some(DynamicInfo {
                    entries: Vec::new(),
                    byte_size: 16,
                    needed_libraries: Vec::new(),
                    soname: None,
                    symbol_table: None,
                    relocation_tables: vec![
                        table("RELR", 0x1080, 16, 8),
                        table("RELA", 0x10c0, 24, 24),
                        table("JMPREL/RELA", 0x10e0, 24, 24),
                    ],
                    init_array: Some(DynamicArrayInfo {
                        memory_offset: 0x1000,
                        byte_size: 16,
                        count: 2,
                    }),
                    fini_array: None,
                }),
            }),
        }
    }

    #[test]
    fn placement_is_consecutive_and_page_aligned() {
        let placed = place_modules(vec![module(), module()]);
        assert_eq!(placed[0].base, FIRST_MODULE_BASE);
        assert_eq!(placed[1].base, FIRST_MODULE_BASE + 0x2000);
        assert_eq!(
            placed[1].text_range(),
            FIRST_MODULE_BASE + 0x2000..FIRST_MODULE_BASE + 0x2100
        );
    }

    #[test]
    fn relr_and_rela_targets_inside_text() {
        let placed = place_modules(vec![module()]).remove(0);
        let b = placed.base;
        let mut slots = relative_text_slots(&placed);
        slots.sort_unstable();
        assert_eq!(
            slots,
            [(0x1000, b + 0x40), (0x1008, b + 0x80), (0x1030, b + 0xc0)]
        );
        assert_eq!(
            relative_text_targets(&placed),
            [b + 0x40, b + 0x80, b + 0xc0]
        );
        assert_eq!(init_array_targets(&placed), [b + 0x40, b + 0x80]);
        assert_eq!(prologue_targets(&placed), [b + 0x40]);
        assert_eq!(lazy_binding_targets(&placed), [b + 0x60]);
        assert_eq!(code_address_targets(&placed), [b + 0x20, b + 0x30]);
    }
}
