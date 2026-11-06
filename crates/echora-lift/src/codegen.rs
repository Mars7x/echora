// Every unwrap below is on `write!`/`writeln!` into a `String`, which cannot fail.
#![allow(clippy::unwrap_used)]

//! Ahead-of-time discovery and Rust emission. Generated execution contains no decoder.

use crate::aarch64::{
    self, Address, AtomicOp, BitfieldOp, CompareOperand, Instruction, LogicalOp, MemoryOp, WideOp,
    Writeback,
};
use crate::simd::SimdOp;
use anyhow::{Context, Result, bail};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt::Write;

/// Keep the host compiler's largest generated function bounded. Small
/// translations retain the direct single-function form to avoid extra dispatch.
pub const MAX_BLOCKS_PER_FUNCTION: usize = 128;
/// Guest address window (4 KiB) per generated dispatch function.
const WINDOW_BITS: u32 = 12;
/// Split crates use 1 KiB windows: densely seeded windows of FP-heavy code
/// otherwise produce single functions too large to compile in bounded memory.
const SPLIT_WINDOW_BITS: u32 = 10;

#[derive(Clone, Debug)]
pub struct CodeRegion {
    pub name: String,
    pub base: u64,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, Serialize)]
pub struct TranslatedInstruction {
    pub pc: u64,
    pub word: u32,
    pub instruction: Instruction,
    pub disassembly: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct BasicBlock {
    pub start: u64,
    pub instructions: Vec<TranslatedInstruction>,
}

#[derive(Clone, Debug)]
pub struct Translation {
    pub instructions: BTreeMap<u64, TranslatedInstruction>,
    pub blocks: Vec<BasicBlock>,
    pub entries: Vec<u64>,
    pub unresolved_targets: Vec<u64>,
    pub truncated: bool,
    /// Recovered jump tables: BR address -> number of distinct case targets.
    pub jump_tables: BTreeMap<u64, usize>,
}

#[derive(Clone, Debug, Serialize)]
pub struct UnsupportedSite {
    pub pc: u64,
    pub word: u32,
    pub disassembly: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct TranslationReport {
    pub translated_instructions: usize,
    pub supported_instructions: usize,
    pub unsupported_instructions: usize,
    /// UDF sites: translated as an explicit undefined-instruction stop.
    pub undefined_instructions: usize,
    pub block_count: usize,
    pub entries: Vec<u64>,
    pub unresolved_targets: Vec<u64>,
    pub truncated: bool,
    pub jump_tables: usize,
    pub jump_table_targets: usize,
    pub unsupported: Vec<UnsupportedSite>,
}

impl Translation {
    pub fn report(&self) -> TranslationReport {
        let unsupported: Vec<_> = self
            .instructions
            .values()
            .filter(|record| record.instruction == Instruction::Unsupported)
            .map(|record| UnsupportedSite {
                pc: record.pc,
                word: record.word,
                disassembly: record.disassembly.clone(),
            })
            .collect();
        TranslationReport {
            translated_instructions: self.instructions.len(),
            supported_instructions: self.instructions.len() - unsupported.len(),
            unsupported_instructions: unsupported.len(),
            undefined_instructions: self
                .instructions
                .values()
                .filter(|record| record.instruction == Instruction::Undefined)
                .count(),
            block_count: self.blocks.len(),
            entries: self.entries.clone(),
            unresolved_targets: self.unresolved_targets.clone(),
            truncated: self.truncated,
            jump_tables: self.jump_tables.len(),
            jump_table_targets: self.jump_tables.values().sum(),
            unsupported,
        }
    }
}

pub fn discover(
    regions: &[CodeRegion],
    entries: &[u64],
    max_instructions: usize,
) -> Result<Translation> {
    discover_prioritized(regions, entries, &[], &BTreeSet::new(), max_instructions)
}

/// Two-phase discovery: `entries` are explored to completion (or the limit)
/// before any `deferred` root is queued. Deferred roots are data-derived
/// candidates (e.g. relocation targets); unreadable ones are skipped rather
/// than rejected, and those queued become dispatch entries like the others.
/// `host_entries` are addresses the runtime handles as host functions: they are
/// never translated, so every branch or fallthrough reaching one returns to the
/// host loop.
pub fn discover_prioritized(
    regions: &[CodeRegion],
    entries: &[u64],
    deferred: &[u64],
    host_entries: &BTreeSet<u64>,
    max_instructions: usize,
) -> Result<Translation> {
    discover_with_tables(
        regions,
        &[],
        entries,
        deferred,
        host_entries,
        max_instructions,
    )
}

/// `discover_prioritized`, also following compiler-generated jump tables
/// stored in the read-only `data` regions (see `jump_table_targets`).
pub fn discover_with_tables(
    regions: &[CodeRegion],
    data: &[CodeRegion],
    entries: &[u64],
    deferred: &[u64],
    host_entries: &BTreeSet<u64>,
    max_instructions: usize,
) -> Result<Translation> {
    if max_instructions == 0 {
        bail!("translation instruction limit must be nonzero");
    }
    if entries.is_empty() {
        bail!("at least one translation entry is required");
    }
    let mut ranges = Vec::new();
    for region in regions {
        if region.base & 3 != 0 {
            bail!("code region {} is not instruction aligned", region.name);
        }
        let end = region
            .base
            .checked_add(region.bytes.len() as u64)
            .context("code region address overflow")?;
        ranges.push((region.base, end, region.name.as_str()));
    }
    ranges.sort_unstable();
    for pair in ranges.windows(2) {
        if pair[0].1 > pair[1].0 {
            bail!("overlapping code regions {} and {}", pair[0].2, pair[1].2);
        }
    }
    let read_word = |pc: u64| -> Option<u32> {
        if pc & 3 != 0 {
            return None;
        }
        regions.iter().find_map(|region| {
            let offset = usize::try_from(pc.checked_sub(region.base)?).ok()?;
            let bytes = region.bytes.get(offset..offset.checked_add(4)?)?;
            Some(u32::from_le_bytes(bytes.try_into().ok()?))
        })
    };
    for &entry in entries {
        if read_word(entry).is_none() {
            bail!("entry {entry:#x} is outside aligned executable bytes");
        }
    }
    let mut pending: VecDeque<u64> = entries.iter().copied().collect();
    let mut instructions = BTreeMap::new();
    let mut starts: BTreeSet<u64> = entries.iter().copied().collect();
    let mut unresolved = BTreeSet::new();
    let mut truncated = false;
    let mut all_entries = entries.to_vec();
    let mut deferred_queued = deferred.is_empty();
    let mut jump_tables = BTreeMap::new();
    loop {
        let pc = match pending.pop_front() {
            Some(pc) => pc,
            None if !deferred_queued => {
                deferred_queued = true;
                for &root in deferred {
                    if read_word(root).is_some() && !instructions.contains_key(&root) {
                        starts.insert(root);
                        all_entries.push(root);
                        pending.push_back(root);
                    }
                }
                continue;
            }
            None => break,
        };
        if instructions.contains_key(&pc) || unresolved.contains(&pc) || host_entries.contains(&pc)
        {
            continue;
        }
        let Some(word) = read_word(pc) else {
            unresolved.insert(pc);
            continue;
        };
        if instructions.len() == max_instructions {
            truncated = true;
            break;
        }
        let instruction = aarch64::decode(word, pc);
        let mut successors = instruction.successors(pc);
        if is_plain_br(word) {
            let targets = jump_table_targets(pc, &read_word, regions, data);
            if !targets.is_empty() {
                jump_tables.insert(pc, targets.len());
                starts.extend(targets.iter().copied());
            }
            successors.extend(targets);
        }
        if instruction.ends_block() {
            starts.extend(successors.iter().copied());
        }
        pending.extend(successors);
        instructions.insert(
            pc,
            TranslatedInstruction {
                pc,
                word,
                instruction,
                disassembly: aarch64::disassemble(word, pc),
            },
        );
    }
    let mut blocks: Vec<BasicBlock> = Vec::new();
    for record in instructions.values() {
        let append = blocks.last().is_some_and(|block| {
            let previous = block
                .instructions
                .last()
                .expect("blocks start with one instruction");
            previous.pc.checked_add(4) == Some(record.pc)
                && !previous.instruction.ends_block()
                && !starts.contains(&record.pc)
                // A straight-line run never crosses a dispatch window, so
                // one block cannot grow a window function without bound.
                && record.pc & ((1 << SPLIT_WINDOW_BITS) - 1) != 0
        });
        if append {
            blocks
                .last_mut()
                .expect("append implies a last block")
                .instructions
                .push(record.clone());
        } else {
            blocks.push(BasicBlock {
                start: record.pc,
                instructions: vec![record.clone()],
            });
        }
    }
    Ok(Translation {
        instructions,
        blocks,
        entries: all_entries,
        unresolved_targets: unresolved.into_iter().collect(),
        truncated,
        jump_tables,
    })
}

/// BR Xn (not BLR, RET or the authenticated forms).
fn is_plain_br(word: u32) -> bool {
    word & 0xffff_fc1f == 0xd61f_0000
}

/// Cases of an index masked by `and Rd, Rn, #(2^k - 1)` (AND immediate with
/// rotation 0 and a run of k low ones): 2^k, or None for other words.
fn low_mask_cases(word: u32) -> Option<u64> {
    if word & 0x7f80_0000 != 0x1200_0000 || (word >> 16) & 0x3f != 0 {
        return None;
    }
    let (sf, n) = (word >> 31, (word >> 22) & 1);
    let ones = (word >> 10) & 0x3f;
    // Up to 16 low ones: wider masks (`and x, x, #0xffffffff`) are
    // zero-extensions, not bounds.
    if n != sf || ones >= 16 {
        return None;
    }
    Some(1 << (ones + 1))
}

/// Most cases a recovered jump table may have.
const MAX_JUMP_TABLE_ENTRIES: u64 = 65536;
/// Entries read from an unsigned byte table whose bound is not found.
const UNBOUNDED_BYTE_TABLE_ENTRIES: u64 = 64;
/// How far back from the BR the table setup is searched, in instructions.
const JUMP_TABLE_WINDOW: u64 = 160;

/// Case targets of a compiler-generated jump table ending in `br` at `pc`:
///
/// ```text
///   cmp   wIdx, #N ; b.hi default     (or b.hs with #N+1)
///   adrp  xTab, table ; add xTab, xTab, #lo
///   ldr{b,h,sb,sh,w,sw} wE, [xTab, xIdx, lsl #s]
///   adr   xBase, anchor               (or the table itself as base)
///   add   xT, xBase, xE, {lsl,sxtw,uxtw} #k
///   br    xT
/// ```
///
/// Instructions are matched by raw encoding while scanning backwards in
/// address order, so setup hoisted out of a loop (straight-line code above the
/// loop head) is still found. Any write to a tracked register by an
/// unrecognized instruction abandons the match. A wrong match only adds
/// discovery roots: blocks run solely when guest control reaches them.
fn jump_table_targets(
    pc: u64,
    read_word: &dyn Fn(u64) -> Option<u32>,
    regions: &[CodeRegion],
    data: &[CodeRegion],
) -> Vec<u64> {
    jump_table(pc, read_word, regions, data).unwrap_or_default()
}

/// Destination register of an instruction that writes a general register,
/// for the instruction classes a jump-table scan must not skip over. None for
/// stores, branches, compares and other instructions without a destination.
fn written_register(word: u32) -> Option<u8> {
    let rd = (word & 31) as u8;
    let class = (word >> 25) & 0xf;
    // Loads and stores. SIMD&FP transfers (bit 26) write no general register.
    if class & 0b0101 == 0b0100 {
        if (word >> 26) & 1 == 1 {
            return None;
        }
        let load = if word & 0x3800_0000 == 0x3800_0000 {
            // Single register (immediate, register offset, pre/post-index):
            // opc 00 is a store.
            (word >> 22) & 3 != 0
        } else if word & 0x3800_0000 == 0x2800_0000 || word & 0x3800_0000 == 0x0800_0000 {
            // Pairs and exclusives: L bit. Only the first destination is
            // reported; a jump-table scan never tracks a pair's second one.
            (word >> 22) & 1 == 1
        } else {
            // Literal loads.
            word & 0x3b00_0000 == 0x1800_0000
        };
        return load.then_some(rd);
    }
    // Data processing (immediate and register): every form writes Rd except
    // flag-setting compares that target the zero register.
    if class & 0b1110 == 0b1000 || class & 0b0111 == 0b0101 {
        if (word >> 29) & 1 == 1 && rd == 31 {
            return None;
        }
        return Some(rd);
    }
    // BL/BLR write the link register.
    if word & 0xfc00_0000 == 0x9400_0000 || word & 0xffff_fc1f == 0xd63f_0000 {
        return Some(30);
    }
    None
}

fn jump_table(
    pc: u64,
    read_word: &dyn Fn(u64) -> Option<u32>,
    regions: &[CodeRegion],
    data: &[CodeRegion],
) -> Option<Vec<u64>> {
    let br = read_word(pc)?;
    let target_reg = ((br >> 5) & 31) as u8;
    let start = pc.saturating_sub(JUMP_TABLE_WINDOW * 4);
    // Walk back over [start, pc) once, resolving registers in dependency order.
    let want = target_reg;
    let mut stage = 0;
    let (mut base_reg, mut entry_reg, mut shift, mut sign_extend32) = (0u8, 0u8, 0u32, false);
    let (mut table_reg, mut entry_size, mut entry_signed) = (0u8, 0u64, false);
    let mut base: Option<u64> = None;
    let mut table: Option<u64> = None;
    let mut table_low: Option<u64> = None;
    let mut base_low: Option<u64> = None;
    let mut compares: Vec<(u8, u64)> = Vec::new();
    let mut index_copies: Vec<u8> = Vec::new();
    // The tightest bound on the index or any copy of it.
    let bound = |compares: &[(u8, u64)], copies: &[u8]| {
        compares
            .iter()
            .filter(|(register, _)| copies.contains(register))
            .map(|(_, cases)| *cases)
            .min()
    };
    let mut cursor = pc;
    while cursor > start {
        let base_done = base.is_some() || base_low == Some(u64::MAX);
        if stage == 2 && base_done && table.is_some() && bound(&compares, &index_copies).is_some() {
            break;
        }
        cursor -= 4;
        let Some(word) = read_word(cursor) else {
            break;
        };
        let rd = (word & 31) as u8;
        let rn = ((word >> 5) & 31) as u8;
        let rm = ((word >> 16) & 31) as u8;
        match stage {
            // add xT, xBase, xE, (lsl #k | sxtw/uxtw #k)
            0 => {
                if written_register(word) != Some(want) {
                    continue;
                }
                if word & 0xff20_0000 == 0x8b00_0000 && (word >> 22) & 3 == 0 {
                    shift = (word >> 10) & 63;
                } else if word & 0xffe0_0000 == 0x8b20_0000 {
                    let option = (word >> 13) & 7;
                    shift = (word >> 10) & 7;
                    sign_extend32 = match option {
                        0b110 => true,
                        0b010 | 0b011 => false,
                        _ => return None,
                    };
                } else {
                    return None;
                }
                if shift > 4 {
                    return None;
                }
                base_reg = rn;
                entry_reg = rm;
                stage = 1;
            }
            // The entry load and the base definition, in either order.
            1 | 2 => {
                let written = written_register(word);
                if written == Some(entry_reg) && entry_size == 0 {
                    // ldr{b,h,w,sb,sh,sw} (register offset)
                    let (size, signed) = match word & 0xffe0_0c00 {
                        0x3860_0800 => (1, false),
                        0x7860_0800 => (2, false),
                        0xb860_0800 => (4, false),
                        0x38e0_0800 | 0x38a0_0800 => (1, true),
                        0x78e0_0800 | 0x78a0_0800 => (2, true),
                        0xb8a0_0800 => (4, true),
                        _ => return None,
                    };
                    let scaled = (word >> 12) & 1 == 1;
                    let option = (word >> 13) & 7;
                    if !matches!(option, 0b011 | 0b010 | 0b110 | 0b111) {
                        return None;
                    }
                    if !scaled && size != 1 {
                        return None;
                    }
                    entry_size = size;
                    entry_signed = signed;
                    table_reg = rn;
                    index_copies.push(rm);
                    if table_reg == base_reg {
                        // Table-relative entries: the base is the table.
                        base_low = Some(u64::MAX);
                    }
                    stage = 2;
                    continue;
                }
                if written == Some(base_reg) && base.is_none() && base_low != Some(u64::MAX) {
                    if word & 0x9f00_0000 == 0x1000_0000 {
                        base = Some(
                            adr_target(word, cursor, false).checked_add(base_low.unwrap_or(0))?,
                        );
                        continue;
                    }
                    if word & 0xffc0_0000 == 0x9100_0000 && rn == rd && base_low.is_none() {
                        base_low = Some(u64::from((word >> 10) & 0xfff));
                        continue;
                    }
                    if word & 0x9f00_0000 == 0x9000_0000 && base_low.is_some() {
                        base = Some(adr_target(word, cursor, true).checked_add(base_low?)?);
                        continue;
                    }
                    return None;
                }
                if entry_size != 0 && written == Some(table_reg) && table.is_none() {
                    if word & 0x9f00_0000 == 0x1000_0000 {
                        table = Some(
                            adr_target(word, cursor, false).checked_add(table_low.unwrap_or(0))?,
                        );
                        continue;
                    }
                    if word & 0xffc0_0000 == 0x9100_0000 && rn == rd && table_low.is_none() {
                        table_low = Some(u64::from((word >> 10) & 0xfff));
                        continue;
                    }
                    if word & 0x9f00_0000 == 0x9000_0000 && table_low.is_some() {
                        table = Some(adr_target(word, cursor, true).checked_add(table_low?)?);
                        continue;
                    }
                    return None;
                }
                // Bound: `cmp wN/xN, #imm` followed by b.hi / b.hs, on the
                // index register or a register it was copied from. Copies
                // (`mov wIdx, wN`) often sit above the loop head, after the
                // compare in this backwards walk, so compares are collected
                // and matched against the copies at the end.
                if entry_size != 0 && word & 0x7f80_001f == 0x7100_001f && (word >> 22) & 1 == 0 {
                    let limit = u64::from((word >> 10) & 0xfff);
                    if let Some(cases) = conditional_bound(read_word, cursor, limit) {
                        compares.push((rn, cases));
                    }
                    continue;
                }
                // `ccmp wN, #imm, #nzcv, cond` before the b.hi / b.hs: when
                // its condition fails the flags are `nzcv`, and with C set
                // and Z clear the branch to the default case is taken
                // either way, so the immediate still bounds the index (main
                // 0x71008f7e14: cmn w0, #1; ccmp w21, #9, #2, ne; b.hi).
                if entry_size != 0 && word & 0x7fe0_0c10 == 0x7a40_0800 && word & 0b0110 == 0b0010 {
                    let limit = u64::from((word >> 16) & 31);
                    if let Some(cases) = conditional_bound(read_word, cursor, limit) {
                        compares.push((rn, cases));
                    }
                    continue;
                }
                if entry_size != 0
                    && written.is_some_and(|register| index_copies.contains(&register))
                {
                    // mov wIdx, wN (ORR with the zero register, no shift).
                    if word & 0x7fe0_ffe0 == 0x2a00_03e0 {
                        index_copies.push(rm);
                    }
                    // and wIdx/xIdx, wN/xN, #(2^k - 1): the mask itself bounds
                    // the index to 2^k cases (a byte switch with no compare).
                    if let Some(cases) = low_mask_cases(word) {
                        compares.push((rd, cases));
                        index_copies.push(rn);
                    }
                    // Other rewrites (zero-extending `and`) keep the value.
                    continue;
                }
            }
            _ => {}
        }
    }
    let table = table?;
    let base = if base_low == Some(u64::MAX) {
        table
    } else {
        base?
    };
    // Unsigned byte entries scaled to instructions can only reach the 256
    // instructions after the base, nearby code of the same function. With no
    // bound in the window (a loop counter whose exit test is elsewhere, as at
    // main 0x71009c93f0), a fixed number of entries is read instead; extra
    // targets only add block entries.
    let unbounded = entry_size == 1 && !entry_signed && shift == 2 && !sign_extend32;
    let (count, exact) = match bound(&compares, &index_copies) {
        Some(count) => (count, true),
        None if unbounded => (UNBOUNDED_BYTE_TABLE_ENTRIES, false),
        None => return None,
    };
    if count == 0 || count > MAX_JUMP_TABLE_ENTRIES {
        return None;
    }
    let read_data = |address: u64, size: u64| -> Option<u64> {
        data.iter().chain(regions).find_map(|region| {
            let offset = usize::try_from(address.checked_sub(region.base)?).ok()?;
            let bytes = region
                .bytes
                .get(offset..offset.checked_add(size as usize)?)?;
            let mut value = [0u8; 8];
            value[..size as usize].copy_from_slice(bytes);
            Some(u64::from_le_bytes(value))
        })
    };
    let mut targets = Vec::new();
    for index in 0..count {
        let Some(raw) = read_data(table.checked_add(index * entry_size)?, entry_size) else {
            if exact {
                return None;
            }
            break;
        };
        let bits = entry_size * 8;
        let mut value = if entry_signed && bits < 64 {
            (((raw << (64 - bits)) as i64) >> (64 - bits)) as u64
        } else {
            raw
        };
        if sign_extend32 {
            value = value as u32 as i32 as i64 as u64;
        }
        let target = base.wrapping_add(value << shift);
        if read_word(target).is_some() {
            targets.push(target);
        } else if exact {
            return None;
        }
    }
    targets.sort_unstable();
    targets.dedup();
    Some(targets)
}

fn adr_target(word: u32, pc: u64, page: bool) -> u64 {
    let immlo = u64::from((word >> 29) & 3);
    let immhi = u64::from((word >> 5) & 0x7ffff);
    let imm = (((immhi << 2) | immlo) << 43) as i64 >> 43;
    if page {
        (pc & !0xfff).wrapping_add((imm << 12) as u64)
    } else {
        pc.wrapping_add(imm as u64)
    }
}

/// Number of in-range indices for `cmp idx, #limit` at `pc`, from the
/// conditional branch to the default case that follows it within a few
/// instructions: b.hi (idx > limit) leaves limit + 1 cases, b.hs leaves limit.
fn conditional_bound(read_word: &dyn Fn(u64) -> Option<u32>, pc: u64, limit: u64) -> Option<u64> {
    for step in 1..=4 {
        let word = read_word(pc + step * 4)?;
        if word & 0xff00_0010 == 0x5400_0000 {
            return match word & 15 {
                8 => Some(limit + 1),
                2 => Some(limit),
                _ => None,
            };
        }
    }
    None
}

/// A register read resolved at generation time (Cpu::read_reg semantics):
/// register 31 is SP or zero, and 32-bit reads take the low word.
fn reg(index: u8, bits: u8, sp: bool) -> String {
    let place = match (index, sp) {
        (31, false) => return "0u64".into(),
        (31, true) => "m.cpu.sp".to_string(),
        _ => format!("m.cpu.regs[{index}]"),
    };
    if bits == 64 {
        place
    } else {
        format!("({place} & {:#x})", (1u64 << bits) - 1)
    }
}
/// A register write resolved at generation time (Cpu::write_reg semantics):
/// a 32-bit write zero-extends, and XZR discards the (still evaluated) value.
fn put(output: &mut String, index: u8, bits: u8, sp: bool, value: &str) {
    let masked = if bits == 64 {
        value.to_string()
    } else {
        format!("({value}) & {:#x}", (1u64 << bits) - 1)
    };
    match (index, sp) {
        (31, false) => writeln!(output, "                let _ = {value};"),
        (31, true) => writeln!(output, "                m.cpu.sp = {masked};"),
        _ => writeln!(output, "                m.cpu.regs[{index}] = {masked};"),
    }
    .unwrap();
}
/// Shifted-register operand; a zero shift of an already width-masked
/// register read is the read itself.
fn shifted(value: String, kind: u8, amount: u8, bits: u8) -> String {
    if amount == 0 {
        value
    } else {
        format!("cpu::shift({value}, {kind}, {amount}, {bits})")
    }
}
fn logical(op: LogicalOp, a: &str, b: &str) -> String {
    match op {
        LogicalOp::And | LogicalOp::Ands => format!("({a}) & ({b})"),
        LogicalOp::Orr => format!("({a}) | ({b})"),
        LogicalOp::Eor => format!("({a}) ^ ({b})"),
    }
}
fn read_memory(bytes: u8, address: &str) -> String {
    format!("m.memory.load_u{}({address}) as u64", bytes * 8)
}
fn write_memory(output: &mut String, bytes: u8, address: &str, value: &str) {
    writeln!(
        output,
        "                m.memory.store_u{}({address}, ({value}) as u{});",
        bytes * 8,
        bytes * 8
    )
    .unwrap();
}
fn load_register(output: &mut String, rt: u8, bytes: u8, op: MemoryOp, value: &str) {
    let MemoryOp::Load { signed, bits } = op else {
        unreachable!()
    };
    let value = if signed {
        format!("cpu::sign_extend({value}, {})", bytes * 8)
    } else {
        value.to_owned()
    };
    put(output, rt, bits, false, &value);
}
fn address(output: &mut String, rn: u8, mode: Address) -> Option<String> {
    writeln!(output, "                let base = {};", reg(rn, 64, true)).unwrap();
    if rn == 31 {
        writeln!(output, "                {SP_ALIGNMENT_CHECK}").unwrap();
    }
    match mode {
        Address::Immediate { offset, writeback } => {
            writeln!(
                output,
                "                let adjusted = base.wrapping_add({:#x}u64);",
                offset as u64
            )
            .unwrap();
            if writeback == Writeback::Post {
                output.push_str("                let address = base;\n");
            } else {
                output.push_str("                let address = adjusted;\n");
            }
            (writeback != Writeback::None).then(|| "adjusted".to_owned())
        }
        Address::Register { rm, option, shift } => {
            writeln!(output, "                let address = base.wrapping_add(cpu::extend({}, {option}, {shift}));", reg(rm, 64, false)).unwrap();
            None
        }
    }
}

/// SIMD structure transfer at [Xn|SP]; post-index adds the transfer size
/// (Rm = 31) or Xm to the base after the transfer completes.
fn structure_transfer(output: &mut String, rn: u8, post: Option<u8>, call: &str) {
    address(
        output,
        rn,
        Address::Immediate {
            offset: 0,
            writeback: Writeback::None,
        },
    );
    writeln!(output, "                let moved: u64 = {call};").unwrap();
    match post {
        None => output.push_str("                let _ = moved;\n"),
        Some(31) => put(output, rn, 64, true, "base.wrapping_add(moved)"),
        Some(rm) => {
            let increment = reg(rm, 64, false);
            put(
                output,
                rn,
                64,
                true,
                &format!("base.wrapping_add({increment})"),
            );
        }
    }
}

/// `last` is false for an instruction followed by another in the same block:
/// that successor stores its own PC before anything can observe it, so the
/// fallthrough PC store is only emitted for a block's final instruction.
/// `block_size` is set for a block's first instruction, which charges the
/// whole block's accounting; every instruction still stores its own PC so a
/// fault reports the exact instruction.
fn emit_instruction(
    output: &mut String,
    record: &TranslatedInstruction,
    last: bool,
    block_size: Option<usize>,
    check_entry: bool,
) {
    let pc = record.pc;
    let mut body = String::new();
    emit_instruction_body(&mut body, record, last);
    // The PC is stored where it can be observed: before anything that can
    // trap, read the PC or call a helper that might. Plain register and flag arithmetic skips
    // it, which keeps the host code smaller.
    // A block's first instruction runs with the PC its `match` arm tested.
    // Elsewhere, an instruction whose only use of the PC is reporting a trap
    // sets it on the error path; anything else stores it up front.
    if block_size.is_none() && needs_pc(&body) {
        match pc_on_error_paths(&body, pc) {
            Some(rewritten) => body = rewritten,
            None => writeln!(output, "                m.cpu.pc = {pc:#x};").unwrap(),
        }
    }
    if let Some(count) = block_size {
        if check_entry {
            writeln!(output, "                m.enter_window()?;").unwrap();
        }
        writeln!(output, "                m.tick_block({count})?;").unwrap();
    }
    output.push_str(&body);
}

/// After a BL/BLR set the PC and link register: run the callee in nested
/// host frames until it returns to `next` (`Core::call`). A trap leaves with
/// the callee's PC, so this is not a `?` (those get the caller's PC stored
/// on their error path).
fn emit_call(output: &mut String, next: u64) {
    writeln!(
        output,
        "                if let Err(trap) = m.call({next:#x}) {{ return Err(trap); }}"
    )
    .unwrap();
}

/// Guest registers as locals of a split window function: the registers its
/// blocks use are read into locals on entry and the ones they write are
/// stored back when it returns, so the host compiler keeps them in host
/// registers (guest stores through raw pointers may alias the `Core`, which
/// otherwise forces a reload of every register after each store). Direct
/// calls and the helpers that read or write general registers through the
/// `Core` see it current: around them the written locals are stored and the
/// used ones reloaded.
struct RegisterLocals {
    used: BTreeSet<u8>,
    written: BTreeSet<u8>,
    sp_used: bool,
    sp_written: bool,
}

/// Helper calls that read general registers from the `Core`.
const READS_REGISTERS: &[&str] = &[
    "pauth_disabled_return(",
    "read_reg(",
    "::fp_move_from_general(",
    "::int_to_fp(",
    "::int_to_fp_fixed(",
    "::insert_general(",
    "::dup_general(",
    "m.trace_hit(",
];

/// Helper calls that write general registers in the `Core`.
const WRITES_REGISTERS: &[&str] = &[
    "write_reg(",
    "strip_lr_instruction_pac(",
    "strip_pac(",
    "::fp_move_to_general(",
    "::fp_to_int(",
    "::move_to_general(",
    "::fp_to_fixed(",
];

const CALL_PREFIX: &str = "if let Err(trap) = m.call(";

impl RegisterLocals {
    fn of(cases: &str) -> Self {
        let mut locals = Self {
            used: BTreeSet::new(),
            written: BTreeSet::new(),
            sp_used: false,
            sp_written: false,
        };
        for (at, _) in cases.match_indices("m.cpu.regs[") {
            let rest = &cases[at + "m.cpu.regs[".len()..];
            let Some(end) = rest.find(']') else { continue };
            let Ok(index) = rest[..end].parse::<u8>() else {
                continue;
            };
            locals.used.insert(index);
            if Self::assigned(&rest[end + 1..]) {
                locals.written.insert(index);
            }
        }
        for (at, _) in cases.match_indices("m.cpu.sp") {
            let rest = &cases[at + "m.cpu.sp".len()..];
            if rest.starts_with(|c: char| c.is_alphanumeric() || c == '_') {
                continue;
            }
            locals.sp_used = true;
            locals.sp_written |= Self::assigned(rest);
        }
        locals
    }

    /// Whether text following a place assigns to it (` = `, not ` == `).
    fn assigned(rest: &str) -> bool {
        rest.starts_with(" = ")
    }

    fn load(&self) -> String {
        let mut text = String::new();
        for index in &self.used {
            writeln!(text, "    let mut x{index} = m.cpu.regs[{index}];").unwrap();
        }
        if self.sp_used {
            text.push_str("    let mut sp = m.cpu.sp;\n");
        }
        text
    }

    fn store(&self) -> String {
        let mut text = String::new();
        for index in &self.written {
            writeln!(text, "    m.cpu.regs[{index}] = x{index};").unwrap();
        }
        if self.sp_written {
            text.push_str("    m.cpu.sp = sp;\n");
        }
        text
    }

    fn reload(&self) -> String {
        let mut text = String::new();
        for index in &self.used {
            write!(text, "x{index} = m.cpu.regs[{index}]; ").unwrap();
        }
        if self.sp_used {
            text.push_str("sp = m.cpu.sp; ");
        }
        text
    }

    fn rewrite(&self, cases: &str) -> String {
        let store = self.store().replace('\n', " ");
        let reload = self.reload();
        let mut text = String::with_capacity(cases.len());
        for line in cases.lines() {
            let mut line = line.replace("m.cpu.regs[", "x[");
            for index in &self.used {
                line = line.replace(&format!("x[{index}]"), &format!("x{index}"));
            }
            let line = replace_sp(&line);
            let indent = &line[..line.len() - line.trim_start().len()];
            if let Some(rest) = line.trim_start().strip_prefix(CALL_PREFIX) {
                // The callee runs on the Core: store before, reload after
                // (also on its error path, so the final store keeps what
                // the callee left).
                let target = &rest[..rest.find(')').expect("call target")];
                writeln!(
                    text,
                    "{indent}{store}let called = m.call({target}); {reload}if let Err(trap) = called {{ return Err(trap); }}"
                )
                .unwrap();
                continue;
            }
            let reads = READS_REGISTERS.iter().any(|helper| line.contains(helper));
            let writes = WRITES_REGISTERS.iter().any(|helper| line.contains(helper));
            if reads || writes {
                writeln!(text, "{indent}{store}").unwrap();
            }
            text.push_str(&line);
            text.push('\n');
            if writes {
                writeln!(text, "{indent}{reload}").unwrap();
            }
        }
        text
    }
}

/// `m.cpu.sp` (the place, not a longer name) replaced by the local `sp`.
fn replace_sp(line: &str) -> String {
    let mut text = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(at) = rest.find("m.cpu.sp") {
        let after = &rest[at + "m.cpu.sp".len()..];
        text.push_str(&rest[..at]);
        if after.starts_with(|c: char| c.is_alphanumeric() || c == '_') {
            text.push_str("m.cpu.sp");
        } else {
            text.push_str("sp");
        }
        rest = after;
    }
    text.push_str(rest);
    text
}

/// The stack alignment check `emit_sp_alignment_check` writes.
const SP_ALIGNMENT_CHECK: &str = "if base & 15 != 0 { return Err(Trap::InvalidState { message: format!(\"unaligned SP memory access at {:#x}\", m.cpu.pc) }); }";

/// `body` with the PC stored only where it leaves with a trap, when its
/// fallible calls (`)?`) and stack alignment checks are the only reasons it
/// needs the PC; otherwise None.
fn pc_on_error_paths(body: &str, pc: u64) -> Option<String> {
    let stripped = body.replace(SP_ALIGNMENT_CHECK, "").replace(")?", ")");
    if needs_pc(&stripped) || body.matches('?').count() != body.matches(")?").count() {
        return None;
    }
    let check =
        SP_ALIGNMENT_CHECK.replacen("{ return", &format!("{{ m.cpu.pc = {pc:#x}; return"), 1);
    Some(body.replace(SP_ALIGNMENT_CHECK, &check).replace(
        ")?",
        &format!(").map_err(|trap| {{ m.cpu.pc = {pc:#x}; trap }})?"),
    ))
}

/// Whether an instruction's emitted code might observe `m.cpu.pc`: it can
/// trap or return early, reads the PC, or calls a CPU or SIMD helper other
/// than the pure arithmetic ones.
fn needs_pc(body: &str) -> bool {
    const PURE: &[&str] = &[
        "m.cpu.add_sub(",
        "m.cpu.add_with_carry(",
        "m.cpu.condition(",
        "m.cpu.logical_flags(",
        "m.cpu.conditional_compare(",
        "m.cpu.conditional_compare_negative(",
    ];
    if body.contains('?') || body.contains("Err(") || body.contains("Trap") {
        return true;
    }
    let reads_pc = body
        .match_indices("m.cpu.pc")
        .any(|(at, _)| !body[at + 8..].starts_with(" ="));
    if reads_pc || body.contains("simd::") || body.contains("m.spin_hint") {
        return true;
    }
    body.match_indices("m.cpu.")
        .filter(|(at, _)| {
            let rest = &body[at + 6..];
            rest.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .is_some_and(|end| rest[end..].starts_with('('))
        })
        .any(|(at, _)| !PURE.iter().any(|pure| body[at..].starts_with(pure)))
}

fn emit_instruction_body(output: &mut String, record: &TranslatedInstruction, last: bool) {
    use Instruction::*;
    let pc = record.pc;
    let next = pc.wrapping_add(4);
    let mut sets_pc = false;
    match &record.instruction {
        Nop => {}
        SpinHint => writeln!(output, "                m.spin_hint();").unwrap(),
        PAuthDisabledHint { hint } => {
            writeln!(
                output,
                "                m.cpu.pauth_disabled_hint(cpu::PAuthHint::{hint:?});"
            )
            .unwrap();
        }
        PAuthDisabledReturn { key } => {
            writeln!(output, "                m.cpu.pc = m.cpu.pauth_disabled_return(cpu::InstructionAuthKey::{key:?});").unwrap();
            sets_pc = true;
        }
        StripLrInstructionPac => output.push_str(
            "                m.cpu.strip_lr_instruction_pac(m.memory.address_space_end())?;\n",
        ),
        ClearExclusive => output.push_str("                m.memory.clear_exclusive();\n"),
        Barrier => output.push_str(
            "                std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);\n",
        ),
        Adr { rd, address } => put(output, *rd, 64, false, &format!("{address:#x}")),
        Branch { target, link } => {
            if *link {
                put(output, 30, 64, false, &format!("{next:#x}"));
            }
            writeln!(output, "                m.cpu.pc = {target:#x};").unwrap();
            if *link {
                emit_call(output, next);
            }
            sets_pc = true;
        }
        BranchCond { target, condition } => {
            writeln!(output, "                m.cpu.pc = if m.cpu.condition({condition}) {{ {target:#x} }} else {{ {next:#x} }};").unwrap();
            sets_pc = true;
        }
        CompareBranch {
            rt,
            bits,
            nonzero,
            target,
        } => {
            let comparison = if *nonzero { "!=" } else { "==" };
            writeln!(output, "                m.cpu.pc = if {} {comparison} 0 {{ {target:#x} }} else {{ {next:#x} }};", reg(*rt, *bits, false)).unwrap();
            sets_pc = true;
        }
        TestBranch {
            rt,
            bit,
            nonzero,
            target,
        } => {
            let comparison = if *nonzero { "!=" } else { "==" };
            writeln!(output, "                m.cpu.pc = if ({} & (1u64 << {bit})) {comparison} 0 {{ {target:#x} }} else {{ {next:#x} }};", reg(*rt, 64, false)).unwrap();
            sets_pc = true;
        }
        BranchReg { rn, link } => {
            writeln!(
                output,
                "                let target = {};",
                reg(*rn, 64, false)
            )
            .unwrap();
            if *link {
                put(output, 30, 64, false, &format!("{next:#x}"));
            }
            output.push_str("                m.cpu.pc = target;\n");
            if *link {
                emit_call(output, next);
            }
            sets_pc = true;
        }
        Svc { number } => writeln!(output, "                m.svc({number})?;").unwrap(),
        MoveWide {
            rd,
            bits,
            op,
            immediate,
            shift,
        } => {
            let immediate = (*immediate as u64) << shift;
            let value = match op {
                WideOp::Zero => format!("{immediate:#x}u64"),
                WideOp::Not => format!("!{immediate:#x}u64"),
                WideOp::Keep => format!(
                    "({} & !{:#x}u64) | {immediate:#x}u64",
                    reg(*rd, *bits, false),
                    0xffff_u64 << shift
                ),
            };
            put(output, *rd, *bits, false, &value);
        }
        AddSubImmediate {
            rd,
            rn,
            bits,
            immediate,
            subtract,
            flags,
        } => {
            writeln!(output, "                let lhs = {};\n                let value = m.cpu.add_sub(lhs, {immediate:#x}, {subtract}, {bits}, {flags});", reg(*rn, *bits, true)).unwrap();
            put(output, *rd, *bits, !flags, "value");
        }
        AddSubShifted {
            rd,
            rn,
            rm,
            bits,
            shift,
            amount,
            subtract,
            flags,
        } => {
            writeln!(output, "                let lhs = {};\n                let rhs = {};\n                let value = m.cpu.add_sub(lhs, rhs, {subtract}, {bits}, {flags});", reg(*rn, *bits, false), shifted(reg(*rm, *bits, false), *shift, *amount, *bits)).unwrap();
            put(output, *rd, *bits, false, "value");
        }
        AddSubExtended {
            rd,
            rn,
            rm,
            bits,
            option,
            amount,
            subtract,
            flags,
        } => {
            writeln!(output, "                let lhs = {};\n                let rhs = cpu::extend({}, {option}, {amount});\n                let value = m.cpu.add_sub(lhs, rhs, {subtract}, {bits}, {flags});", reg(*rn, *bits, true), reg(*rm, 64, false)).unwrap();
            put(output, *rd, *bits, !flags, "value");
        }
        AddCarry {
            rd,
            rn,
            rm,
            bits,
            subtract,
            flags,
        } => {
            writeln!(output, "                let lhs = {};\n                let rhs = {}{};\n                let carry = m.cpu.nzcv & 2 != 0;\n                let value = m.cpu.add_with_carry(lhs, rhs, carry, {bits}, {flags});", reg(*rn, *bits, false), if *subtract { "!" } else { "" }, reg(*rm, *bits, false)).unwrap();
            put(output, *rd, *bits, false, "value");
        }
        LogicalShifted {
            rd,
            rn,
            rm,
            bits,
            op,
            invert,
            shift,
            amount,
        } => {
            writeln!(output, "                let lhs = {};\n                let rhs = {}({});\n                let value = {};", reg(*rn, *bits, false), if *invert { "!" } else { "" }, shifted(reg(*rm, *bits, false), *shift, *amount, *bits), logical(*op, "lhs", "rhs")).unwrap();
            if *op == LogicalOp::Ands {
                writeln!(
                    output,
                    "                m.cpu.logical_flags(value, {bits});"
                )
                .unwrap();
            }
            put(output, *rd, *bits, false, "value");
        }
        LogicalImmediate {
            rd,
            rn,
            bits,
            op,
            mask,
        } => {
            writeln!(
                output,
                "                let value = {};",
                logical(*op, &reg(*rn, *bits, false), &format!("{mask:#x}u64"))
            )
            .unwrap();
            if *op == LogicalOp::Ands {
                writeln!(
                    output,
                    "                m.cpu.logical_flags(value, {bits});"
                )
                .unwrap();
            }
            put(output, *rd, *bits, *op != LogicalOp::Ands, "value");
        }
        Bitfield {
            rd,
            rn,
            bits,
            op,
            rotation,
            sign_bit,
            write_mask,
            top_mask,
        } => {
            writeln!(output, "                let source = {};\n                let rotated = cpu::shift(source, 3, {rotation}, {bits});", reg(*rn, *bits, false)).unwrap();
            match op {
                BitfieldOp::Unsigned => writeln!(output, "                let value = rotated & {write_mask:#x}u64 & {top_mask:#x}u64;").unwrap(),
                BitfieldOp::Signed => writeln!(output, "                let top = if (source >> {sign_bit}) & 1 != 0 {{ cpu::width_mask({bits}) }} else {{ 0 }};\n                let value = (top & !{top_mask:#x}u64) | (rotated & {write_mask:#x}u64 & {top_mask:#x}u64);").unwrap(),
                BitfieldOp::Merge => writeln!(output, "                let old = {};\n                let bottom = (old & !{write_mask:#x}u64) | (rotated & {write_mask:#x}u64);\n                let value = (old & !{top_mask:#x}u64) | (bottom & {top_mask:#x}u64);", reg(*rd, *bits, false)).unwrap(),
            }
            put(output, *rd, *bits, false, "value");
        }
        Extract {
            rd,
            rn,
            rm,
            bits,
            lsb,
        } => {
            let value = if *lsb == 0 {
                reg(*rm, *bits, false)
            } else {
                format!(
                    "({} >> {lsb}) | ({} << {})",
                    reg(*rm, *bits, false),
                    reg(*rn, *bits, false),
                    bits - lsb
                )
            };
            put(output, *rd, *bits, false, &value);
        }
        Select {
            rd,
            rn,
            rm,
            bits,
            condition,
            operation,
        } => {
            let rhs = match operation {
                0 => reg(*rm, *bits, false),
                1 => format!("{}.wrapping_add(1)", reg(*rm, *bits, false)),
                2 => format!("!{}", reg(*rm, *bits, false)),
                3 => format!("{}.wrapping_neg()", reg(*rm, *bits, false)),
                _ => unreachable!(),
            };
            writeln!(output, "                let value = if m.cpu.condition({condition}) {{ {} }} else {{ {rhs} }};", reg(*rn, *bits, false)).unwrap();
            put(output, *rd, *bits, false, "value");
        }
        ConditionalCompare {
            rn,
            bits,
            rhs,
            condition,
            nzcv,
        } => {
            let rhs = match rhs {
                CompareOperand::Register(rm) => reg(*rm, *bits, false),
                CompareOperand::Immediate(value) => format!("{value}u64"),
            };
            writeln!(output, "                let lhs = {};\n                let rhs = {rhs};\n                m.cpu.conditional_compare(lhs, rhs, {bits}, {condition}, {nzcv});", reg(*rn, *bits, false)).unwrap();
        }
        MultiplyAdd {
            rd,
            rn,
            rm,
            ra,
            bits,
            subtract,
        } => {
            writeln!(output, "                let product = {}.wrapping_mul({});\n                let value = {}.{}(product);", reg(*rn, *bits, false), reg(*rm, *bits, false), reg(*ra, *bits, false), if *subtract { "wrapping_sub" } else { "wrapping_add" }).unwrap();
            put(output, *rd, *bits, false, "value");
        }
        UnsignedMultiplyAddLong { rd, rn, rm, ra } => {
            writeln!(
                output,
                "                let value = cpu::unsigned_multiply_add_long({}, {}, {});",
                reg(*rn, 32, false),
                reg(*rm, 32, false),
                reg(*ra, 64, false)
            )
            .unwrap();
            put(output, *rd, 64, false, "value");
        }
        MultiplyHigh { rd, rn, rm, signed } => {
            let cast = if *signed { "as i64 as i128" } else { "as u128" };
            let value = format!(
                "((({} {cast}) * ({} {cast})) >> 64) as u64",
                reg(*rn, 64, false),
                reg(*rm, 64, false)
            );
            put(output, *rd, 64, false, &value);
        }
        Divide {
            rd,
            rn,
            rm,
            bits,
            signed,
        } => {
            writeln!(
                output,
                "                let lhs = {};\n                let rhs = {};",
                reg(*rn, *bits, false),
                reg(*rm, *bits, false)
            )
            .unwrap();
            let expression = if *signed {
                format!(
                    "(cpu::sign_extend(lhs, {bits}) as i64).wrapping_div(cpu::sign_extend(rhs, {bits}) as i64) as u64"
                )
            } else {
                "lhs / rhs".to_owned()
            };
            writeln!(
                output,
                "                let value = if rhs == 0 {{ 0 }} else {{ {expression} }};"
            )
            .unwrap();
            put(output, *rd, *bits, false, "value");
        }
        VariableShift {
            rd,
            rn,
            rm,
            bits,
            shift,
        } => {
            let value = format!(
                "cpu::shift({}, {shift}, ({} & {}) as u8, {bits})",
                reg(*rn, *bits, false),
                reg(*rm, *bits, false),
                bits - 1
            );
            put(output, *rd, *bits, false, &value);
        }
        CountLeading { rd, rn, bits, sign } => {
            writeln!(
                output,
                "                let source = {};",
                reg(*rn, *bits, false)
            )
            .unwrap();
            let value = if *sign {
                format!(
                    "((if source & (1u64 << {}) != 0 {{ !source }} else {{ source }}) as u{bits}).leading_zeros() as u64 - 1",
                    bits - 1
                )
            } else {
                format!("(source as u{bits}).leading_zeros() as u64")
            };
            put(output, *rd, *bits, false, &value);
        }
        ReverseBits { rd, rn, bits } => put(
            output,
            *rd,
            *bits,
            false,
            &format!(
                "({} as u{bits}).reverse_bits() as u64",
                reg(*rn, *bits, false)
            ),
        ),
        ReverseBytes {
            rd,
            rn,
            bits,
            group_bits,
        } => {
            writeln!(
                output,
                "                let source = {};",
                reg(*rn, *bits, false)
            )
            .unwrap();
            let pieces: Vec<_> = (0..*bits).step_by(*group_bits as usize).map(|offset| format!("((((source >> {offset}) as u{group_bits}).swap_bytes() as u64) << {offset})")).collect();
            put(output, *rd, *bits, false, &pieces.join(" | "));
        }
        CompareAndSwap {
            rs,
            rt,
            rn,
            bytes,
            acquire,
            release,
        } => {
            address(
                output,
                *rn,
                Address::Immediate {
                    offset: 0,
                    writeback: Writeback::None,
                },
            );
            let bits = if *bytes == 8 { 64 } else { 32 };
            writeln!(
                output,
                "                let expected = {};\n                let desired = {};",
                reg(*rs, bits, false),
                reg(*rt, bits, false),
            )
            .unwrap();
            writeln!(output, "                let observed = m.memory.compare_exchange_single_thread(address, {bytes}, expected, desired, {acquire}, {release})?;").unwrap();
            // A swap that cannot change memory is an atomic read: a poll.
            writeln!(
                output,
                "                if expected == desired {{ m.spin_hint(); }}"
            )
            .unwrap();
            // Rs can alias either source, including Rn. Update it only after
            // every operand is captured and the memory operation succeeds.
            put(output, *rs, bits, false, "observed");
        }
        CompareAndSwapPair {
            rs,
            rt,
            rn,
            bytes,
            acquire,
            release,
        } => {
            address(
                output,
                *rn,
                Address::Immediate {
                    offset: 0,
                    writeback: Writeback::None,
                },
            );
            let bits = bytes * 8;
            writeln!(
                output,
                "                let expected = ({}, {});\n                let desired = ({}, {});",
                reg(*rs, bits, false),
                reg(*rs + 1, bits, false),
                reg(*rt, bits, false),
                reg(*rt + 1, bits, false),
            )
            .unwrap();
            writeln!(output, "                let observed = m.memory.compare_exchange_pair_single_thread(address, {bytes}, expected, desired, {acquire}, {release})?;").unwrap();
            writeln!(
                output,
                "                if expected == desired {{ m.spin_hint(); }}"
            )
            .unwrap();
            // As for CAS: write the pair only after every operand is
            // captured and the memory operation succeeded.
            put(output, *rs, bits, false, "observed.0");
            put(output, *rs + 1, bits, false, "observed.1");
        }
        AtomicAdd {
            rs,
            rt,
            rn,
            bytes,
            acquire,
            release,
        } => {
            address(
                output,
                *rn,
                Address::Immediate {
                    offset: 0,
                    writeback: Writeback::None,
                },
            );
            let bits = if *bytes == 8 { 64 } else { 32 };
            writeln!(
                output,
                "                let addend = {};",
                reg(*rs, bits, false)
            )
            .unwrap();
            // Rt==31 suppresses acquire, independently of the release bit.
            let effective_acquire = *acquire && *rt != 31;
            writeln!(output, "                let observed = m.memory.fetch_add_single_thread(address, {bytes}, addend, {effective_acquire}, {release})?;").unwrap();
            // Capture Rn/SP and Rs/ZR before Rt changes: all source/destination
            // aliases are legal, and faults must leave the destination intact.
            put(output, *rt, bits, false, "observed");
        }
        LoadAcquireRcpc { rt, rn, bytes } => {
            address(
                output,
                *rn,
                Address::Immediate {
                    offset: 0,
                    writeback: Writeback::None,
                },
            );
            writeln!(output, "                let loaded = m.memory.load_acquire_rcpc_single_thread(address, {bytes})?;").unwrap();
            put(
                output,
                *rt,
                if *bytes == 8 { 64 } else { 32 },
                false,
                "loaded",
            );
        }
        LoadAcquireRcsc { rt, rn, bytes } => {
            address(
                output,
                *rn,
                Address::Immediate {
                    offset: 0,
                    writeback: Writeback::None,
                },
            );
            writeln!(output, "                let loaded = m.memory.load_acquire_rcsc_single_thread(address, {bytes})?;").unwrap();
            put(
                output,
                *rt,
                if *bytes == 8 { 64 } else { 32 },
                false,
                "loaded",
            );
        }
        LoadExclusive {
            rt,
            rn,
            bytes,
            acquire,
        } => {
            address(
                output,
                *rn,
                Address::Immediate {
                    offset: 0,
                    writeback: Writeback::None,
                },
            );
            writeln!(output, "                let loaded = m.memory.load_exclusive_single_thread(address, {bytes}, {acquire})?;").unwrap();
            // Rn/SP was captured before Rt changes, so Rt == Rn is fine.
            put(
                output,
                *rt,
                if *bytes == 8 { 64 } else { 32 },
                false,
                "loaded",
            );
        }
        StoreExclusive {
            rs,
            rt,
            rn,
            bytes,
            release,
        } => {
            address(
                output,
                *rn,
                Address::Immediate {
                    offset: 0,
                    writeback: Writeback::None,
                },
            );
            writeln!(
                output,
                "                let value = {};",
                reg(*rt, if *bytes == 8 { 64 } else { 32 }, false)
            )
            .unwrap();
            writeln!(output, "                let stored = m.memory.store_exclusive_single_thread(address, {bytes}, value, {release})?;").unwrap();
            // Ws gets 0 on success and 1 on failure; a fault leaves it intact.
            put(output, *rs, 32, false, "u64::from(!stored)");
        }
        StoreRelease { rt, rn, bytes } => {
            address(
                output,
                *rn,
                Address::Immediate {
                    offset: 0,
                    writeback: Writeback::None,
                },
            );
            writeln!(
                output,
                "                let value = {};",
                reg(*rt, if *bytes == 8 { 64 } else { 32 }, false)
            )
            .unwrap();
            writeln!(
                output,
                "                m.memory.store_release_single_thread(address, {bytes}, value)?;"
            )
            .unwrap();
        }
        LoadStore {
            rt,
            rn,
            bytes,
            op,
            address: mode,
        } => {
            let writeback = address(output, *rn, *mode);
            match op {
                MemoryOp::Store => write_memory(
                    output,
                    *bytes,
                    "address",
                    &reg(*rt, if *bytes == 8 { 64 } else { 32 }, false),
                ),
                MemoryOp::Load { .. } => {
                    writeln!(
                        output,
                        "                let loaded = {};",
                        read_memory(*bytes, "address")
                    )
                    .unwrap();
                    load_register(output, *rt, *bytes, *op, "loaded");
                }
            }
            if let Some(value) = writeback {
                put(output, *rn, 64, true, &value);
            }
        }
        LoadStorePair {
            rt,
            rt2,
            rn,
            bytes,
            op,
            offset,
            writeback,
        } => {
            let update = address(
                output,
                *rn,
                Address::Immediate {
                    offset: *offset,
                    writeback: *writeback,
                },
            );
            writeln!(
                output,
                "                let second_address = address.wrapping_add({bytes});"
            )
            .unwrap();
            match op {
                MemoryOp::Store => {
                    write_memory(
                        output,
                        *bytes,
                        "address",
                        &reg(*rt, if *bytes == 8 { 64 } else { 32 }, false),
                    );
                    write_memory(
                        output,
                        *bytes,
                        "second_address",
                        &reg(*rt2, if *bytes == 8 { 64 } else { 32 }, false),
                    );
                }
                MemoryOp::Load { .. } => {
                    writeln!(
                        output,
                        "                let first = {};\n                let second = {};",
                        read_memory(*bytes, "address"),
                        read_memory(*bytes, "second_address")
                    )
                    .unwrap();
                    load_register(output, *rt, *bytes, *op, "first");
                    load_register(output, *rt2, *bytes, *op, "second");
                }
            }
            if let Some(value) = update {
                put(output, *rn, 64, true, &value);
            }
        }
        LoadStoreVectorPair {
            rt,
            rt2,
            rn,
            load,
            offset,
            writeback,
        } => {
            let update = address(
                output,
                *rn,
                Address::Immediate {
                    offset: *offset,
                    writeback: *writeback,
                },
            );
            if *load {
                writeln!(
                    output,
                    "                m.cpu.load_q_pair(&m.memory, address, {rt}, {rt2})?;"
                )
                .unwrap();
            } else {
                writeln!(
                    output,
                    "                m.cpu.store_q_pair(&mut m.memory, address, {rt}, {rt2})?;"
                )
                .unwrap();
            }
            if let Some(value) = update {
                put(output, *rn, 64, true, &value);
            }
        }
        LoadLiteral {
            rt,
            bytes,
            signed,
            bits,
            address,
        } => {
            writeln!(
                output,
                "                let loaded = {};",
                read_memory(*bytes, &format!("{address:#x}"))
            )
            .unwrap();
            load_register(
                output,
                *rt,
                *bytes,
                MemoryOp::Load {
                    signed: *signed,
                    bits: *bits,
                },
                "loaded",
            );
        }
        ReadPhysicalCounter { rt } => {
            // Sample once after the instruction's budget/tick check, including
            // an XZR destination. The runtime owns clock/epoch/frequency policy.
            output.push_str("                let counter = m.physical_counter_ticks();\n");
            put(output, *rt, 64, false, "counter");
        }
        ReadThreadPointer { rt, read_only } => put(
            output,
            *rt,
            64,
            false,
            if *read_only {
                "m.cpu.tpidrro_el0"
            } else {
                "m.cpu.tpidr_el0"
            },
        ),
        WriteThreadPointer { rt } => writeln!(
            output,
            "                m.cpu.tpidr_el0 = {};",
            reg(*rt, 64, false)
        )
        .unwrap(),
        StripPac { rd } => writeln!(
            output,
            "                m.cpu.strip_pac({rd}, m.memory.address_space_end())?;"
        )
        .unwrap(),
        PacGeneric { rd, rn, rm } => {
            let value = format!(
                "{{ let h = ({} ^ ({}).rotate_left(29)).wrapping_mul(0x9e37_79b9_7f4a_7c15); \
                 (h ^ (h >> 29)) & 0xffff_ffff_0000_0000 }}",
                reg(*rn, 64, false),
                reg(*rm, 64, true)
            );
            put(output, *rd, 64, false, &value);
        }
        PAuthDisabledData | CacheMaintenance => {}
        ReadNzcv { rt } => put(output, *rt, 64, false, "(u64::from(m.cpu.nzcv) << 28)"),
        WriteNzcv { rt } => writeln!(
            output,
            "                m.cpu.nzcv = (({} >> 28) & 0xf) as u8;",
            reg(*rt, 64, false)
        )
        .unwrap(),
        Crc32 {
            rd,
            rn,
            rm,
            bytes,
            castagnoli,
        } => {
            let data_bits = if *bytes == 8 { 64 } else { 32 };
            let value = format!(
                "u64::from(echora_guest::cpu::crc32(({}) as u32, {}, {bytes}, {castagnoli}))",
                reg(*rn, 32, false),
                reg(*rm, data_bits, false)
            );
            put(output, *rd, 32, false, &value);
        }
        ReadCacheType { rt } => {
            // Cortex-A78 cache geometry: 64-byte I/D lines, PIPT L1I, 64-byte
            // ERG/CWG; IDC/DIC report that maintenance is required.
            put(output, *rt, 64, false, "0x8444_c004u64");
        }
        ConditionalCompareNegative {
            rn,
            bits,
            rhs,
            condition,
            nzcv,
        } => {
            let rhs = match rhs {
                CompareOperand::Register(rm) => reg(*rm, *bits, false),
                CompareOperand::Immediate(value) => format!("{value}u64"),
            };
            writeln!(output, "                let lhs = {};\n                let rhs = {rhs};\n                m.cpu.conditional_compare_negative(lhs, rhs, {bits}, {condition}, {nzcv});", reg(*rn, *bits, false)).unwrap();
        }
        MultiplyAddLong {
            rd,
            rn,
            rm,
            ra,
            signed,
            subtract,
        } => {
            let widen = if *signed {
                "as u32 as i32 as i64 as u64"
            } else {
                "as u32 as u64"
            };
            writeln!(
                output,
                "                let product = (({}) {widen}).wrapping_mul(({}) {widen});\n                let value = ({}).{}(product);",
                reg(*rn, 32, false),
                reg(*rm, 32, false),
                reg(*ra, 64, false),
                if *subtract { "wrapping_sub" } else { "wrapping_add" }
            )
            .unwrap();
            put(output, *rd, 64, false, "value");
        }
        AtomicMemory {
            op,
            rs,
            rt,
            rn,
            bytes,
            acquire,
            release,
        } => {
            address(
                output,
                *rn,
                Address::Immediate {
                    offset: 0,
                    writeback: Writeback::None,
                },
            );
            let bits = if *bytes == 8 { 64 } else { 32 };
            writeln!(
                output,
                "                let operand = {};",
                reg(*rs, bits, false)
            )
            .unwrap();
            let effective_acquire = *acquire && (*rt != 31 || *op == AtomicOp::Swap);
            writeln!(output, "                let observed = m.memory.atomic_rmw_single_thread(address, {bytes}, echora_guest::memory::AtomicOp::{op:?}, operand, {effective_acquire}, {release})?;").unwrap();
            put(output, *rt, bits, false, "observed");
        }
        ReadFpControl { rt, status } => {
            let field = if *status { "fpsr" } else { "fpcr" };
            put(output, *rt, 64, false, &format!("u64::from(m.cpu.{field})"));
        }
        WriteFpControl { rt, status } => {
            let field = if *status { "fpsr" } else { "fpcr" };
            writeln!(
                output,
                "                m.cpu.{field} = {} as u32;",
                reg(*rt, 64, false)
            )
            .unwrap();
        }
        Simd(op) => match op {
            SimdOp::LoadStore {
                rt,
                rn,
                bytes,
                load,
                address: mode,
            } => {
                let writeback = address(output, *rn, *mode);
                if *load {
                    writeln!(output, "                echora_guest::simd::load(&mut m.cpu, &m.memory, address, {rt}, {bytes})?;").unwrap();
                } else {
                    writeln!(output, "                echora_guest::simd::store(&m.cpu, &mut m.memory, address, {rt}, {bytes})?;").unwrap();
                }
                if let Some(value) = writeback {
                    put(output, *rn, 64, true, &value);
                }
            }
            SimdOp::LoadStorePair {
                rt,
                rt2,
                rn,
                bytes,
                load,
                offset,
                writeback,
            } => {
                let update = address(
                    output,
                    *rn,
                    Address::Immediate {
                        offset: *offset,
                        writeback: *writeback,
                    },
                );
                if *load {
                    writeln!(output, "                echora_guest::simd::load_pair(&mut m.cpu, &m.memory, address, {rt}, {rt2}, {bytes})?;").unwrap();
                } else {
                    writeln!(output, "                echora_guest::simd::store_pair(&m.cpu, &mut m.memory, address, {rt}, {rt2}, {bytes})?;").unwrap();
                }
                if let Some(value) = update {
                    put(output, *rn, 64, true, &value);
                }
            }
            SimdOp::TransferMultiple {
                rt,
                rn,
                registers,
                structure,
                bits,
                q,
                load,
                post,
            } => {
                structure_transfer(
                    output,
                    *rn,
                    *post,
                    &format!(
                        "echora_guest::simd::transfer_multiple(&mut m.cpu, &mut m.memory, address, {rt}, {registers}, {structure}, {bits}, {q}, {load})?"
                    ),
                );
            }
            SimdOp::TransferSingle {
                rt,
                rn,
                registers,
                bits,
                index,
                load,
                post,
            } => {
                structure_transfer(
                    output,
                    *rn,
                    *post,
                    &format!(
                        "echora_guest::simd::transfer_single(&mut m.cpu, &mut m.memory, address, {rt}, {registers}, {bits}, {index}, {load})?"
                    ),
                );
            }
            SimdOp::LoadReplicate {
                rt,
                rn,
                registers,
                bits,
                q,
                post,
            } => {
                structure_transfer(
                    output,
                    *rn,
                    *post,
                    &format!(
                        "echora_guest::simd::load_replicate(&mut m.cpu, &m.memory, address, {rt}, {registers}, {bits}, {q})?"
                    ),
                );
            }
            SimdOp::LoadLiteral { rt, bytes, address } => {
                writeln!(output, "                echora_guest::simd::load(&mut m.cpu, &m.memory, {address:#x}, {rt}, {bytes})?;").unwrap();
            }
            other => {
                let call = crate::simd::emit_call(other).expect("non-memory SIMD operation");
                writeln!(output, "                {call}").unwrap();
            }
        },
        Undefined => {
            writeln!(output, "                return Err(Trap::UndefinedInstruction {{ pc: {pc:#x}, word: {:#x} }});", record.word).unwrap();
            sets_pc = true;
        }
        Unsupported => {
            writeln!(output, "                return Err(Trap::UnsupportedInstruction {{ pc: {pc:#x}, word: {:#x} }});", record.word).unwrap();
            sets_pc = true;
        }
    }
    if !sets_pc && last {
        writeln!(output, "                m.cpu.pc = {next:#x};").unwrap();
    }
}

/// `check_entry`: each block checks its page and records its start (single
/// dispatch functions); split windows check once on entry instead.
/// Block starts whose entries generated code records with `trace_hit`:
/// ECHORA_TRACE_PCS at lift time, comma-separated hexadecimal guest
/// addresses. Only the region crates holding them change.
fn traced_pcs() -> &'static std::collections::BTreeSet<u64> {
    static PCS: std::sync::OnceLock<std::collections::BTreeSet<u64>> = std::sync::OnceLock::new();
    PCS.get_or_init(|| {
        std::env::var("ECHORA_TRACE_PCS")
            .unwrap_or_default()
            .split(',')
            .filter_map(|pc| u64::from_str_radix(pc.trim().trim_start_matches("0x"), 16).ok())
            .collect()
    })
}

fn emit_block_cases(output: &mut String, blocks: &[BasicBlock], check_entry: bool) {
    for block in blocks {
        writeln!(output, "            {:#x} => {{", block.start).unwrap();
        if traced_pcs().contains(&block.start) {
            output.push_str("                m.trace_hit();\n");
        }
        let count = block.instructions.len();
        for (index, instruction) in block.instructions.iter().enumerate() {
            let block_size = (index == 0).then_some(count);
            emit_instruction(
                output,
                instruction,
                index + 1 == count,
                block_size,
                check_entry,
            );
        }
        output.push_str("            }\n");
    }
}

/// Split emission: one library source per region of `1 << region_bits` guest
/// bytes, each holding public 4 KiB window functions, plus a root dispatcher.
/// Unchanged regions produce byte-identical sources, so a caller that writes
/// only changed files lets cargo skip recompiling them entirely.
/// Returns (root `generated.rs`, [(region index, region `lib.rs`)]).
pub fn emit_rust_split(
    translation: &Translation,
    region_bits: u32,
) -> (String, Vec<(u64, String)>) {
    let mut regions = Vec::new();
    let root = emit_rust_split_to(translation, region_bits, &mut |region, source| {
        regions.push((region, source));
        Ok::<(), std::convert::Infallible>(())
    });
    match root {
        Ok(root) => (root, regions),
        Err(never) => match never {},
    }
}

/// [`emit_rust_split`], handing each region's source to `sink` as soon as it
/// is complete (in ascending region order) so that at most one region is
/// held in memory. Returns the root dispatcher source.
pub fn emit_rust_split_to<E>(
    translation: &Translation,
    region_bits: u32,
    sink: &mut dyn FnMut(u64, String) -> Result<(), E>,
) -> Result<String, E> {
    assert!(region_bits >= SPLIT_WINDOW_BITS);
    let mut windows: Vec<(u64, &[BasicBlock])> = Vec::new();
    let mut first = 0;
    for index in 1..=translation.blocks.len() {
        let window = translation.blocks[first].start >> SPLIT_WINDOW_BITS;
        if index == translation.blocks.len()
            || translation.blocks[index].start >> SPLIT_WINDOW_BITS != window
        {
            windows.push((window, &translation.blocks[first..index]));
            first = index;
        }
    }
    let shift = region_bits - SPLIT_WINDOW_BITS;
    let mut root = format!(
        "// Generated from user-provided executable bytes. Keep this file local.\n\
         use echora_guest::{{Core, Trap, WindowFn}};\n\n\
         /// Registers the window table used for direct calls between windows.\n\
         pub fn install() {{\n    echora_guest::install_windows({SPLIT_WINDOW_BITS}, WINDOWS);\n}}\n\n\
         /// Runs translated code from `m.cpu.pc` until its window returns or traps.\n\
         pub fn dispatch(m: &mut Core) -> Result<(), Trap> {{\n    let pc = m.cpu.pc;\n    match pc >> {SPLIT_WINDOW_BITS} {{\n"
    );
    let header = "// Generated from user-provided executable bytes. Keep this file local.\n#[allow(unused_imports)]\nuse echora_guest::cpu;\nuse echora_guest::{Core as Machine, Trap};\n";
    let mut current: Option<(u64, String)> = None;
    for (window, blocks) in &windows {
        let region = window >> shift;
        writeln!(
            root,
            "        {window:#x} => gen_r_{region:x}::run_window_{window:x}(m),"
        )
        .unwrap();
        if current.as_ref().map(|(index, _)| *index) != Some(region) {
            if let Some((index, source)) = current.take() {
                sink(index, source)?;
            }
            current = Some((region, String::from(header)));
        }
        let output = &mut current.as_mut().expect("region set above").1;
        // A window lies within one page, and page permissions change only in
        // SVCs, which leave the window: one execute check on entry covers
        // every block run before it returns.
        let mut cases = String::new();
        emit_block_cases(&mut cases, blocks, false);
        writeln!(
            cases,
            "            pc if pc >> {SPLIT_WINDOW_BITS} != {window:#x} => return Ok(()),"
        )
        .unwrap();
        // Every completed block passes the boundary check, so a loop that
        // stays inside one window still yields to the host scheduler.
        cases.push_str("            pc => return Err(Trap::MissingBlock { pc }),\n");
        let locals = RegisterLocals::of(&cases);
        writeln!(output, "\n#[inline(never)]\n#[allow(unused_parens, unused_mut, unused_assignments, unused_variables)]\npub fn run_window_{window:x}(m: &mut Machine) -> Result<(), Trap> {{\n    m.enter_window()?;").unwrap();
        output.push_str(&locals.load());
        output.push_str(
            "    let result = (|| -> Result<(), Trap> {\n    loop {\n        match m.cpu.pc {\n",
        );
        output.push_str(&locals.rewrite(&cases));
        output.push_str("        }\n        m.finish_block()?;\n    }\n    })();\n");
        output.push_str(&locals.store());
        output.push_str("    result\n}\n");
    }
    if let Some((index, source)) = current.take() {
        sink(index, source)?;
    }
    root.push_str("        _ => Err(Trap::MissingBlock { pc }),\n    }\n}\n");
    // The window table for direct calls between windows (`Core::call`).
    root.push_str("\nstatic WINDOWS: &[(u64, WindowFn)] = &[\n");
    for (window, _) in &windows {
        let region = window >> shift;
        writeln!(
            root,
            "    ({window:#x}, gen_r_{region:x}::run_window_{window:x}),"
        )
        .unwrap();
    }
    root.push_str("];\n");
    Ok(root)
}

pub fn emit_rust(translation: &Translation) -> String {
    let mut output = String::from(
        "// Generated from user-provided executable bytes. Keep this file local.\n// Guest opcodes were decoded before Rust compilation.\n#[allow(unused_imports)]\nuse echora_guest::cpu;\nuse echora_guest::{Core, Trap};\n\n/// Runs translated code from `m.cpu.pc` until it traps or leaves a window.\n#[allow(unused_parens)]\npub fn dispatch(m: &mut Core) -> Result<(), Trap> {\n    loop {\n        match m.cpu.pc {\n",
    );
    if translation.blocks.len() <= MAX_BLOCKS_PER_FUNCTION {
        emit_block_cases(&mut output, &translation.blocks, true);
        output.push_str(
            "            pc => return Err(Trap::MissingBlock { pc }),\n        }\n        m.finish_block()?;\n    }\n}\n",
        );
        return output;
    }

    // Large translations dispatch through one function per fixed 4 KiB guest
    // address window. Window functions are named by address, so adding a
    // block changes only its own window's code: unchanged windows emit
    // byte-identical Rust and incremental rebuilds can reuse them. Blocks are
    // sorted by start, so each window's blocks are contiguous in the list.
    let mut windows: Vec<(u64, &[BasicBlock])> = Vec::new();
    let mut first = 0;
    for index in 1..=translation.blocks.len() {
        let window = translation.blocks[first].start >> WINDOW_BITS;
        if index == translation.blocks.len()
            || translation.blocks[index].start >> WINDOW_BITS != window
        {
            windows.push((window, &translation.blocks[first..index]));
            first = index;
        }
    }
    // The host loop calls `dispatch` again whenever a window returns.
    output.push_str("            pc => match pc >> 12 {\n");
    for (window, _) in &windows {
        writeln!(
            output,
            "                {window:#x} => return run_window_{window:x}(m),"
        )
        .unwrap();
    }
    output.push_str(
        "                _ => return Err(Trap::MissingBlock { pc }),\n            },\n        }\n    }\n}\n",
    );
    for (window, blocks) in &windows {
        writeln!(output, "\n#[inline(never)]\n#[allow(unused_parens)]\nfn run_window_{window:x}(m: &mut Core) -> Result<(), Trap> {{\n    loop {{\n        match m.cpu.pc {{").unwrap();
        emit_block_cases(&mut output, blocks, true);
        // Leaving the window executes no guest instruction; the dispatcher
        // selects the next window or reports an absent target. A hole inside
        // this window still traps immediately here.
        writeln!(
            output,
            "            pc if pc >> 12 != {window:#x} => return Ok(()),"
        )
        .unwrap();
        output.push_str(
            "            pc => return Err(Trap::MissingBlock { pc }),\n        }\n        m.finish_block()?;\n    }\n}\n",
        );
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Expected text written with the Cpu::read_reg/write_reg helper calls,
    /// in the form the generator emits after resolving them statically.
    fn resolved(text: &str) -> String {
        let mut text = text.to_string();
        while let Some(start) = text.find("m.cpu.read_reg(") {
            let end = start + text[start..].find(')').unwrap();
            let args: Vec<&str> = text[start + 15..end].split(", ").collect();
            let (index, bits, sp): (u8, u8, bool) = (
                args[0].parse().unwrap(),
                args[1].parse().unwrap(),
                args[2] == "true",
            );
            let replacement = reg(index, bits, sp);
            text.replace_range(start..=end, &replacement);
        }
        while let Some(start) = text.find("m.cpu.write_reg(") {
            let mut depth = 1;
            let mut end = start + 16;
            for (offset, c) in text[start + 16..].char_indices() {
                match c {
                    '(' => depth += 1,
                    ')' => depth -= 1,
                    _ => {}
                }
                if depth == 0 {
                    end = start + 16 + offset;
                    break;
                }
            }
            let args: Vec<String> = text[start + 16..end]
                .splitn(4, ", ")
                .map(str::to_string)
                .collect();
            let mut line = String::new();
            put(
                &mut line,
                args[0].parse().unwrap(),
                args[1].parse().unwrap(),
                args[2] == "true",
                &args[3],
            );
            let replacement = line.trim().trim_end_matches(';').to_string();
            text.replace_range(start..=end, &replacement);
        }
        text
    }

    /// Guest instructions charged by the emitted per-block accounting.
    fn charged(output: &str) -> usize {
        output
            .split("m.tick_block(")
            .skip(1)
            .map(|rest| rest[..rest.find(')').unwrap()].parse::<usize>().unwrap())
            .sum()
    }
    fn region(words: &[u32]) -> CodeRegion {
        CodeRegion {
            name: "synthetic".into(),
            base: 0x1000,
            bytes: words.iter().flat_map(|word| word.to_le_bytes()).collect(),
        }
    }

    #[test]
    fn physical_counter_read_samples_once_after_tick_and_preserves_fallthrough() {
        for rt in [0u8, 1, 17, 30, 31] {
            let word = 0xd53b_e020 | u32::from(rt);
            let translation = discover(&[region(&[word, 0xd400_00e1])], &[0x1000], 2).unwrap();
            assert_eq!(translation.report().unsupported_instructions, 0);
            let output = emit_rust(&translation);
            // The block's first instruction runs with the PC its arm tested.
            assert!(!output.contains("m.cpu.pc = 0x1000;"));
            let current_pc = output.find("0x1000 => {").unwrap();
            let tick = output.find("m.tick_block(").unwrap();
            let sample = output
                .find("let counter = m.physical_counter_ticks();")
                .unwrap();
            let commit = output
                .find(&resolved(&format!(
                    "m.cpu.write_reg({rt}, 64, false, counter);"
                )))
                .unwrap();
            let next_pc = output.find("m.cpu.pc = 0x1004;").unwrap();
            assert!(current_pc < tick && tick < sample && sample < commit && commit < next_pc);
            assert_eq!(output.matches("m.physical_counter_ticks()").count(), 1);
            let write = resolved(&format!("m.cpu.write_reg({rt}, 64, false, counter);"));
            assert_eq!(output.matches(write.as_str()).count(), 1);
            assert_eq!(output.matches("m.cpu.regs[").count(), usize::from(rt != 31));
            assert_eq!(charged(&output), 2);
            assert!(!output.contains("m.cpu.nzcv"));
            assert!(!output.contains("m.cpu.sp"));
            assert!(!output.contains("aarch64::decode"));
        }
    }

    #[test]
    fn spin_waits_hint_the_scheduler() {
        // YIELD, WFE, then CAS W1, W1, [X0] (a swap that cannot change memory).
        let words = [0xd503_203f, 0xd503_205f, 0x88a1_7c01, 0xd400_00e1];
        let translation = discover(&[region(&words)], &[0x1000], 4).unwrap();
        let output = emit_rust(&translation);
        assert_eq!(output.matches("m.spin_hint();").count(), 3);
        assert!(output.contains("if expected == desired { m.spin_hint(); }"));
    }

    #[test]
    fn cas_emits_static_captured_operands_and_commits_result_after_memory_success() {
        for (word, rs, rt, rn, bytes, bits, acquire, release) in [
            (0x88e8_7c14, 8, 20, 0, 4, 32, true, false),
            (0x08e1_fc02, 1, 2, 0, 1, 32, true, true),
            (0x48a1_fc02, 1, 2, 0, 2, 32, false, true),
            (0xc8a0_7c01, 0, 1, 0, 8, 64, false, false),
            (0x88a1_ffff, 1, 31, 31, 4, 32, false, true),
        ] {
            let translation = discover(&[region(&[word, 0xd400_00e1])], &[0x1000], 2).unwrap();
            assert_eq!(translation.report().unsupported_instructions, 0);
            let output = emit_rust(&translation);
            let base = output
                .find(&resolved(&format!(
                    "let base = m.cpu.read_reg({rn}, 64, true)"
                )))
                .unwrap();
            let expected = output
                .find(&resolved(&format!(
                    "let expected = m.cpu.read_reg({rs}, {bits}, false)"
                )))
                .unwrap();
            let desired = output
                .find(&resolved(&format!(
                    "let desired = m.cpu.read_reg({rt}, {bits}, false)"
                )))
                .unwrap();
            let access = output.find(&format!("let observed = m.memory.compare_exchange_single_thread(address, {bytes}, expected, desired, {acquire}, {release})?;")).unwrap();
            let commit = output
                .find(&resolved(&format!(
                    "m.cpu.write_reg({rs}, {bits}, false, observed)"
                )))
                .unwrap();
            assert!(base < expected && expected < desired && desired < access && access < commit);
            if rn == 31 {
                assert!(output.contains("if base & 15 != 0"));
            }
            assert!(!output.contains("aarch64::decode"));
            assert_eq!(charged(&output), 2);
        }
    }

    #[test]
    fn ldadd_captures_operands_before_rmw_and_suppresses_acquire_only_for_rt31() {
        for (word, rs, rt, rn, bytes, bits, acquire, release) in [
            (0x3821_0002, 1, 2, 0, 1, 32, false, false),
            (0x78a1_0002, 1, 2, 0, 2, 32, true, false),
            (0xb8e8_0008, 8, 8, 0, 4, 32, true, true),
            (0xf861_0002, 1, 2, 0, 8, 64, false, true),
            (0xf821_0000, 1, 0, 0, 8, 64, false, false),
            (0xf820_0002, 0, 2, 0, 8, 64, false, false),
            (0xf8ff_03e2, 31, 2, 31, 8, 64, true, true),
            (0xf8e1_03ff, 1, 31, 31, 8, 64, false, true),
        ] {
            let translation = discover(&[region(&[word, 0xd400_00e1])], &[0x1000], 2).unwrap();
            assert_eq!(translation.report().unsupported_instructions, 0);
            let output = emit_rust(&translation);
            let base = output
                .find(&resolved(&format!(
                    "let base = m.cpu.read_reg({rn}, 64, true)"
                )))
                .unwrap();
            let source = output
                .find(&resolved(&format!(
                    "let addend = m.cpu.read_reg({rs}, {bits}, false)"
                )))
                .unwrap();
            let access = output.find(&format!("let observed = m.memory.fetch_add_single_thread(address, {bytes}, addend, {acquire}, {release})?;")).unwrap();
            let commit = output
                .find(&resolved(&format!(
                    "m.cpu.write_reg({rt}, {bits}, false, observed)"
                )))
                .unwrap();
            assert!(base < source && source < access && access < commit);
            if rn == 31 {
                assert!(output.contains("if base & 15 != 0"));
            }
            assert!(!output.contains("m.cpu.nzcv"));
            assert!(!output.contains("aarch64::decode"));
            assert_eq!(charged(&output), 2);
        }
    }

    #[test]
    fn ldapr_captures_base_and_commits_only_after_ordered_read_success() {
        for (word, rt, rn, bytes, bits) in [
            (0x38bf_c108, 8, 8, 1, 32),
            (0x78bf_c002, 2, 0, 2, 32),
            (0xb8bf_c01f, 31, 0, 4, 32),
            (0xf8bf_c3e2, 2, 31, 8, 64),
        ] {
            let translation = discover(&[region(&[word, 0xd400_00e1])], &[0x1000], 2).unwrap();
            assert_eq!(translation.report().unsupported_instructions, 0);
            let output = emit_rust(&translation);
            let base = output
                .find(&resolved(&format!(
                    "let base = m.cpu.read_reg({rn}, 64, true)"
                )))
                .unwrap();
            let access = output
                .find(&format!(
                    "let loaded = m.memory.load_acquire_rcpc_single_thread(address, {bytes})?;"
                ))
                .unwrap();
            let commit = output
                .find(&resolved(&format!(
                    "m.cpu.write_reg({rt}, {bits}, false, loaded)"
                )))
                .unwrap();
            assert!(base < access && access < commit);
            if rn == 31 {
                assert!(output.contains("if base & 15 != 0"));
            }
            assert!(!output.contains("aarch64::decode"));
            assert_eq!(charged(&output), 2);
        }
    }

    #[test]
    fn ldar_uses_rcsc_helper_and_commits_only_after_successful_read() {
        for (word, rt, rn, bytes, bits) in [
            (0x08df_fc63, 3, 3, 1, 32),
            (0x48df_fc25, 5, 1, 2, 32),
            (0x88df_fc3f, 31, 1, 4, 32),
            (0xc8df_fd00, 0, 8, 8, 64),
            (0xc8df_ffe5, 5, 31, 8, 64),
        ] {
            let translation = discover(&[region(&[word, 0xd400_00e1])], &[0x1000], 2).unwrap();
            assert_eq!(translation.report().unsupported_instructions, 0);
            let output = emit_rust(&translation);
            let base = output
                .find(&resolved(&format!(
                    "let base = m.cpu.read_reg({rn}, 64, true)"
                )))
                .unwrap();
            let access = output
                .find(&format!(
                    "let loaded = m.memory.load_acquire_rcsc_single_thread(address, {bytes})?;"
                ))
                .unwrap();
            let commit = output
                .find(&resolved(&format!(
                    "m.cpu.write_reg({rt}, {bits}, false, loaded)"
                )))
                .unwrap();
            assert!(base < access && access < commit);
            if rn == 31 {
                assert!(output.contains("if base & 15 != 0"));
            }
            assert!(!output.contains("load_acquire_rcpc_single_thread"));
            assert!(!output.contains("m.cpu.nzcv"));
            assert!(!output.contains("aarch64::decode"));
            assert_eq!(charged(&output), 2);
        }
    }

    #[test]
    fn stlr_captures_base_and_value_without_register_or_flag_writeback() {
        for (word, rt, rn, bytes, bits) in [
            (0x089f_fc15, 21, 0, 1, 32),
            (0x489f_fc02, 2, 0, 2, 32),
            (0x889f_fc02, 2, 0, 4, 32),
            (0xc89f_fc00, 0, 0, 8, 64),
            (0xc89f_ffff, 31, 31, 8, 64),
        ] {
            let translation = discover(&[region(&[word, 0xd400_00e1])], &[0x1000], 2).unwrap();
            assert_eq!(translation.report().unsupported_instructions, 0);
            let output = emit_rust(&translation);
            let base = output
                .find(&resolved(&format!(
                    "let base = m.cpu.read_reg({rn}, 64, true)"
                )))
                .unwrap();
            let source = output
                .find(&resolved(&format!(
                    "let value = m.cpu.read_reg({rt}, {bits}, false)"
                )))
                .unwrap();
            let store = output
                .find(&format!(
                    "m.memory.store_release_single_thread(address, {bytes}, value)?;"
                ))
                .unwrap();
            assert!(base < source && source < store);
            if rn == 31 {
                assert!(output.contains("if base & 15 != 0"));
            }
            assert!(!output.contains("] = ") && !output.contains("m.cpu.sp = "));
            assert!(!output.contains("m.cpu.nzcv"));
            assert!(!output.contains("aarch64::decode"));
            assert_eq!(charged(&output), 2);
        }
    }

    #[test]
    fn large_translations_shard_by_address_window_without_extra_guest_ticks() {
        // 1025 one-instruction blocks span windows 0x1 (0x1000..0x1fff) and 0x2.
        let count = 1025;
        let mut words = vec![0xd503_201f; count];
        // The call and return cross the window boundary.
        words[0] = 0x9400_0000 | 1024;
        words[1024] = 0xd65f_03c0;
        let entries: Vec<_> = (0..count).map(|index| 0x1000 + index as u64 * 4).collect();
        let translation = discover(&[region(&words)], &entries, count).unwrap();
        let output = emit_rust(&translation);
        assert_eq!(translation.blocks.len(), count);
        assert_eq!(output.matches("#[inline(never)]").count(), 2);
        assert!(output.contains("0x1 => return run_window_1(m),"));
        assert!(output.contains("0x2 => return run_window_2(m),"));
        assert!(output.contains("pc if pc >> 12 != 0x1 => return Ok(())"));
        assert_eq!(charged(&output), count);
        let windows: Vec<_> = output.split("fn run_window_").skip(1).collect();
        assert_eq!(
            windows.iter().map(|text| charged(text)).collect::<Vec<_>>(),
            vec![1024, 1]
        );
        assert!(windows[0].contains(&resolved("m.cpu.write_reg(30, 64, false, 0x1004)")));
        assert!(windows[0].contains("m.cpu.pc = 0x2000;"));
        assert!(windows[1].contains(&resolved("let target = m.cpu.read_reg(30, 64, false)")));
        // One absent-target arm per window plus the outer dispatcher's.
        assert_eq!(
            output
                .matches("return Err(Trap::MissingBlock { pc })")
                .count(),
            3
        );
    }

    #[test]
    fn small_translations_keep_direct_dispatch_and_windows_do_not_fill_holes() {
        let small = discover(&[region(&[0xd65f_03c0])], &[0x1000], 1).unwrap();
        assert!(!emit_rust(&small).contains("run_window_"));
        let count = MAX_BLOCKS_PER_FUNCTION + 1;
        let mut words = vec![0; count * 2];
        let entries: Vec<_> = (0..count)
            .map(|index| {
                words[index * 2] = 0x1400_0000; // Each known instruction branches to itself.
                0x1000 + index as u64 * 8
            })
            .collect();
        let translation = discover(&[region(&words)], &entries, count).unwrap();
        let output = emit_rust(&translation);
        assert!(output.contains("0x1 => return run_window_1(m),"));
        assert!(!output.contains("0x1004 => {"));
        let window = output.split("fn run_window_1").nth(1).unwrap();
        assert!(window.contains("pc if pc >> 12 != 0x1 => return Ok(())"));
        assert!(window.contains("pc => return Err(Trap::MissingBlock { pc })"));
    }

    #[test]
    fn split_windows_check_the_host_boundary_after_every_block() {
        // A self-loop stays inside one window; without the boundary check it
        // would never return to the host scheduler.
        let translation = discover(&[region(&[0x1400_0000])], &[0x1000], 1).unwrap();
        let (_, regions) = emit_rust_split(&translation, SPLIT_WINDOW_BITS);
        assert_eq!(regions.len(), 1);
        let window = regions[0].1.split("fn run_window_").nth(1).unwrap();
        assert!(window.contains("        }\n        m.finish_block()?;\n    }\n    })();\n"));
    }

    #[test]
    fn split_windows_keep_guest_registers_in_locals() {
        // ADD X1, X1, #1; BL +8: X1 lives in a local, stored before the
        // direct call and reloaded after it, and stored when the window
        // returns.
        let translation =
            discover(&[region(&[0x9100_0421, 0x9400_0002, 0, 0])], &[0x1000], 4).unwrap();
        let (_, regions) = emit_rust_split(&translation, SPLIT_WINDOW_BITS);
        let window = regions[0].1.split("fn run_window_").nth(1).unwrap();
        assert!(window.contains("    let mut x1 = m.cpu.regs[1];\n"));
        assert!(window.contains("x1 = (x1).wrapping_add(") || window.contains("x1 = "));
        assert!(!window.contains("match m.cpu.pc {\n") || window.contains("m.cpu.regs[1] = x1;"));
        let call = window.find("let called = m.call(0x1008);").unwrap();
        let stored = window[..call].rfind("m.cpu.regs[1] = x1;").unwrap();
        assert!(window[call..].find("x1 = m.cpu.regs[1];").unwrap() > 0);
        assert!(stored < call);
        assert!(
            window.ends_with("    m.cpu.regs[1] = x1;\n    m.cpu.regs[30] = x30;\n    result\n}\n")
        );
    }

    #[test]
    fn q_pair_emission_commits_writeback_after_successful_whole_pair_access() {
        let translation = discover(
            &[region(&[0xadbf_07e0, 0xacc1_07e0, 0xd65f_03c0])],
            &[0x1000],
            8,
        )
        .unwrap();
        let output = emit_rust(&translation);
        assert_eq!(translation.report().unsupported_instructions, 0);
        let store = output
            .find("m.cpu.store_q_pair(&mut m.memory, address, 0, 1)?;")
            .unwrap();
        let load = output
            .find("m.cpu.load_q_pair(&m.memory, address, 0, 1)?;")
            .unwrap();
        let first_writeback = output
            .find(&resolved("m.cpu.write_reg(31, 64, true, adjusted)"))
            .unwrap();
        let last_writeback = output
            .rfind(&resolved("m.cpu.write_reg(31, 64, true, adjusted)"))
            .unwrap();
        assert!(store < first_writeback && first_writeback < load && load < last_writeback);
        assert!(output.contains("if base & 15 != 0"));
        assert!(output.contains("let address = adjusted;"));
        assert!(output.contains("let address = base;"));
    }

    #[test]
    fn long_multiply_and_conditional_compare_emit_typed_static_semantics() {
        let translation = discover(
            &[region(&[
                0x9bb5_2508,
                0x7a48_1924,
                0xfa49_1100,
                0xd65f_03c0,
            ])],
            &[0x1000],
            8,
        )
        .unwrap();
        assert_eq!(translation.report().unsupported_instructions, 0);
        let output = emit_rust(&translation);
        assert!(output.contains(&resolved("cpu::unsigned_multiply_add_long(m.cpu.read_reg(8, 32, false), m.cpu.read_reg(21, 32, false), m.cpu.read_reg(9, 64, false))")));
        assert!(output.contains(&resolved("m.cpu.write_reg(8, 64, false, value)")));
        assert!(output.contains("let rhs = 8u64;"));
        assert!(output.contains("m.cpu.conditional_compare(lhs, rhs, 32, 1, 4)"));
        assert!(output.contains(&resolved("let rhs = m.cpu.read_reg(9, 64, false);")));
        assert!(output.contains("m.cpu.conditional_compare(lhs, rhs, 64, 1, 0)"));
        assert_eq!(charged(&output), 4);
        assert!(!output.contains("aarch64::decode"));
    }

    #[test]
    fn discovery_tracks_calls_and_fallthrough_without_decoding_data_as_reachable() {
        // BL 0x100c; NOP; B self; RET; unsupported data.
        let translation = discover(
            &[region(&[
                0x9400_0003,
                0xd503_201f,
                0x1400_0000,
                0xd65f_03c0,
                0,
            ])],
            &[0x1000],
            50,
        )
        .unwrap();
        assert_eq!(translation.instructions.len(), 4);
        assert!(!translation.instructions.contains_key(&0x1010));
        assert!(translation.blocks.iter().any(|block| block.start == 0x100c));
        let output = emit_rust(&translation);
        assert!(output.contains(&resolved("m.cpu.write_reg(30, 64, false, 0x1004)")));
        assert!(!output.contains("aarch64::decode"));
        assert_eq!(charged(&output), 4);
    }

    #[test]
    fn generic_authentication_codes_are_deterministic_upper_halves() {
        // pacga x2, x19, x17; ret
        let translation = discover(&[region(&[0x9ad1_3262, 0xd65f_03c0])], &[0x1000], 8).unwrap();
        assert_eq!(translation.report().unsupported_instructions, 0);
        let output = emit_rust(&translation);
        assert!(output.contains("rotate_left(29)"));
        assert!(output.contains("& 0xffff_ffff_0000_0000"));
    }

    #[test]
    fn jump_tables_seed_every_case() {
        // adrp x16, table page; add x16, x16, #0x10; cmp w17, #2; b.hi 0x1028;
        // ldrh w0, [x16, x17, lsl #1]; adr x3, 0x1020; add x3, x3, x0, lsl #2;
        // br x3; three RET cases. The table (u16 0, 1, 2) is in read-only data.
        let words = [
            0xb000_0010,
            0x9100_4210,
            0x7100_0a3f,
            0x5400_00e8,
            0x7871_7a00,
            0x1000_0063,
            0x8b00_0863,
            0xd61f_0060,
            0xd65f_03c0,
            0xd65f_03c0,
            0xd65f_03c0,
        ];
        let mut table = vec![0u8; 16];
        table.extend([0, 0, 1, 0, 2, 0]);
        let data = CodeRegion {
            name: "rodata".into(),
            base: 0x2000,
            bytes: table,
        };
        let host = BTreeSet::new();
        let with =
            discover_with_tables(&[region(&words)], &[data], &[0x1000], &[], &host, 64).unwrap();
        for case in [0x1020, 0x1024, 0x1028] {
            assert!(with.instructions.contains_key(&case), "{case:#x}");
            assert!(with.blocks.iter().any(|block| block.start == case));
        }
        // Without the table only the bound check's default case is reached.
        let without = discover(&[region(&words)], &[0x1000], 64).unwrap();
        assert!(!without.instructions.contains_key(&0x1024));
    }

    #[test]
    fn jump_table_bounds_follow_index_copies() {
        // As above, but the load indexes x26 = w3 (`mov w26, w3` before the
        // bound check `cmp w3, #2`), the form hoisted out of dispatch loops.
        let words = [
            0xb000_0010,
            0x9100_4210,
            0x2a03_03fa,
            0x7100_087f,
            0x5400_00e8,
            0x787a_7a00,
            0x1000_0063,
            0x8b00_0863,
            0xd61f_0060,
            0xd65f_03c0,
            0xd65f_03c0,
            0xd65f_03c0,
        ];
        let mut table = vec![0u8; 16];
        table.extend([0, 0, 1, 0, 2, 0]);
        let data = CodeRegion {
            name: "rodata".into(),
            base: 0x2000,
            bytes: table,
        };
        let translation = discover_with_tables(
            &[region(&words)],
            &[data],
            &[0x1000],
            &[],
            &BTreeSet::new(),
            64,
        )
        .unwrap();
        assert_eq!(translation.jump_tables.get(&0x1020), Some(&3));
        assert!(translation.instructions.contains_key(&0x1028));
    }

    #[test]
    fn conditional_compares_bound_jump_tables() {
        // adrp x8, 0x2000; add x8, x8, #0; ccmp w1, #2, #2, ne; b.hi 0x1024;
        // adr x9, 0x101c; ldrh w10, [x8, x1, lsl #1]; add x9, x9, x10, lsl
        // #2; br x9 (at 0x101c), then three RET cases.
        let words = [
            0xb000_0008, // adrp x8, 0x2000
            0x9100_0108, // add x8, x8, #0
            0x7a42_1822, // ccmp w1, #2, #2, ne
            0x5400_00a8, // b.hi 0x1020
            0x1000_00a9, // adr x9, 0x1024
            0x7861_790a, // ldrh w10, [x8, x1, lsl #1]
            0x8b0a_0929, // add x9, x9, x10, lsl #2
            0xd61f_0120, // br x9
            0xd65f_03c0, // 0x1020: default
            0xd65f_03c0, // 0x1024: case 0
            0xd65f_03c0, // case 1
            0xd65f_03c0, // case 2
        ];
        let data = CodeRegion {
            name: "rodata".into(),
            base: 0x2000,
            bytes: vec![0, 0, 1, 0, 2, 0],
        };
        let translation = discover_with_tables(
            &[region(&words)],
            &[data],
            &[0x1000],
            &[],
            &BTreeSet::new(),
            64,
        )
        .unwrap();
        assert_eq!(translation.jump_tables.get(&0x101c), Some(&3));
        assert!(translation.instructions.contains_key(&0x102c));
    }

    #[test]
    fn unbounded_byte_tables_read_a_fixed_number_of_entries() {
        // As the low-mask case, but the index (x8) has no bound in the
        // window: a loop counter tested elsewhere. Entries are read anyway.
        let words = [
            0xb000_0009, // adrp x9, 0x2000
            0x9100_0129, // add x9, x9, #0
            0x1000_008a, // adr x10, 0x1018
            0x3868_692b, // ldrb w11, [x9, x8]
            0x8b0b_094a, // add x10, x10, x11, lsl #2
            0xd61f_0140, // br x10
            0xd65f_03c0, // 0x1018: case 0
            0xd65f_03c0, // case 1
            0xd65f_03c0, // case 2
        ];
        let data = CodeRegion {
            name: "rodata".into(),
            base: 0x2000,
            bytes: vec![0, 2, 1],
        };
        let translation = discover_with_tables(
            &[region(&words)],
            &[data],
            &[0x1000],
            &[],
            &BTreeSet::new(),
            64,
        )
        .unwrap();
        assert_eq!(translation.jump_tables.get(&0x1014), Some(&3));
        assert!(translation.instructions.contains_key(&0x1020));
    }

    #[test]
    fn jump_table_bounds_follow_low_masks() {
        // A byte switch with no compare (as in update 655360's main
        // 0x7101f72440): x8 = x0 & 0xff indexes a byte table of word
        // offsets from `adr x10, 0x1010`; cases 0, 1 and 2 are seeded.
        let words = [
            0xb000_0009, // adrp x9, 0x2000
            0x9100_0129, // add x9, x9, #0
            0x9240_1c08, // and x8, x0, #0xff
            0x1000_008a, // adr x10, 0x101c
            0x3868_692b, // ldrb w11, [x9, x8]
            0x8b0b_094a, // add x10, x10, x11, lsl #2
            0xd61f_0140, // br x10
            0xd65f_03c0, // 0x101c: case 0
            0xd65f_03c0, // case 1
            0xd65f_03c0, // case 2
        ];
        let mut table = vec![0u8, 1, 2];
        table.resize(256, 0);
        let data = CodeRegion {
            name: "rodata".into(),
            base: 0x2000,
            bytes: table,
        };
        let translation = discover_with_tables(
            &[region(&words)],
            &[data],
            &[0x1000],
            &[],
            &BTreeSet::new(),
            64,
        )
        .unwrap();
        assert_eq!(translation.jump_tables.get(&0x1018), Some(&3));
        assert!(translation.instructions.contains_key(&0x1020));
        assert!(translation.instructions.contains_key(&0x1024));
        assert_eq!(low_mask_cases(0x9240_1c08), Some(256));
        assert_eq!(low_mask_cases(0x1200_0c00), Some(16)); // and w0, w0, #0xf
        assert_eq!(low_mask_cases(0x9100_0129), None);
        // and x17, x17, #0xffffffff zero-extends; it bounds nothing.
        assert_eq!(low_mask_cases(0x9240_7e31), None);
    }

    #[test]
    fn host_entries_are_never_translated() {
        // NOP, NOP, NOP, RET: the third NOP is a host entry, so the run from
        // 0x1000 stops before it and a root at it is ignored.
        let words = [0xd503_201f, 0xd503_201f, 0xd503_201f, 0xd65f_03c0];
        let host = BTreeSet::from([0x1008]);
        let translation =
            discover_prioritized(&[region(&words)], &[0x1000], &[0x1008], &host, 16).unwrap();
        assert!(translation.instructions.contains_key(&0x1004));
        assert!(!translation.instructions.contains_key(&0x1008));
        assert!(!translation.instructions.contains_key(&0x100c));
    }

    #[test]
    fn discovery_rejects_bad_entries_and_bounds_work() {
        assert!(discover(&[region(&[0])], &[0x1001], 4).is_err());
        assert!(discover(&[region(&[0])], &[0x1004], 4).is_err());
        let translation = discover(&[region(&[0xd503_201f; 8])], &[0x1000], 3).unwrap();
        assert!(translation.truncated);
        assert_eq!(translation.instructions.len(), 3);
        let translation = discover(&[region(&[0x1400_0020])], &[0x1000], 4).unwrap();
        assert_eq!(translation.unresolved_targets, vec![0x1080]);
    }

    #[test]
    fn loads_go_through_the_guest_view_and_svcs_set_the_pc_on_error() {
        // NOP, LDR X0, [X1], SVC #7: the load reads the guest view directly
        // (no error path, so no PC), and the SVC writes its PC only where
        // it leaves with a trap.
        let words = [0xd503_201f, 0xf940_0020, 0xd400_00e1];
        let translation = discover(&[region(&words)], &[0x1000], 3).unwrap();
        let output = emit_rust(&translation);
        assert!(output.contains("m.memory.load_u64(address)"));
        assert!(!output.contains("m.cpu.pc = 0x1004"));
        assert!(output.contains(".map_err(|trap| { m.cpu.pc = 0x1008; trap })?"));
    }

    #[test]
    fn udf_is_an_undefined_instruction_stop_not_an_unsupported_one() {
        // Execution cannot continue past UDF, so the NOP after it is not discovered.
        let translation = discover(&[region(&[0x0000_0123, 0xd503_201f])], &[0x1000], 3).unwrap();
        let report = translation.report();
        assert_eq!(report.translated_instructions, 1);
        assert_eq!(
            (
                report.unsupported_instructions,
                report.undefined_instructions
            ),
            (0, 1)
        );
        let output = emit_rust(&translation);
        assert!(output.contains("UndefinedInstruction { pc: 0x1000, word: 0x123 }"));
    }

    #[test]
    fn unknown_instruction_has_precise_pc_and_explicit_trap() {
        let translation = discover(&[region(&[0xffff_ffff])], &[0x1000], 3).unwrap();
        assert_eq!(translation.report().unsupported_instructions, 1);
        let output = emit_rust(&translation);
        assert!(output.contains("UnsupportedInstruction { pc: 0x1000, word: 0xffffffff }"));
        // The arm `0x1000 =>` fixes the PC; no store precedes the charge.
        assert!(output.find("0x1000 => {").unwrap() < output.find("m.tick_block(").unwrap());
        assert!(!output.contains("m.cpu.pc = 0x1000"));
    }

    #[test]
    fn indirect_link_reads_old_lr_before_replacing_it() {
        let translation = discover(&[region(&[0xd63f_03c0, 0])], &[0x1000], 3).unwrap();
        let output = emit_rust(&translation);
        assert!(
            output
                .find(&resolved("let target = m.cpu.read_reg(30, 64, false)"))
                .unwrap()
                < output
                    .find(&resolved("m.cpu.write_reg(30, 64, false, 0x1004)"))
                    .unwrap()
        );
    }

    #[test]
    fn disabled_auth_hints_emit_dedicated_calls_and_returns_end_discovery() {
        for (sign, authenticate, ret, hint_prefix, key) in [
            (0xd503_233f, 0xd503_23bf, 0xd65f_0bff, "ia", "A"),
            (0xd503_237f, 0xd503_23ff, 0xd65f_0fff, "ib", "B"),
        ] {
            let translation =
                discover(&[region(&[sign, authenticate, ret, 0])], &[0x1000], 8).unwrap();
            assert_eq!(translation.instructions.len(), 3);
            assert_eq!(translation.report().unsupported_instructions, 0);
            assert!(!translation.instructions.contains_key(&0x100c));
            let output = emit_rust(&translation);
            assert!(output.contains(&format!(
                "m.cpu.pauth_disabled_hint(cpu::PAuthHint::Pac{hint_prefix}Sp)"
            )));
            assert!(output.contains(&format!(
                "m.cpu.pauth_disabled_hint(cpu::PAuthHint::Aut{hint_prefix}Sp)"
            )));
            assert!(output.contains(&format!(
                "m.cpu.pc = m.cpu.pauth_disabled_return(cpu::InstructionAuthKey::{key})"
            )));
            assert_eq!(charged(&output), 3);
            assert!(!output.contains("m.cpu.regs[30] = "));
            assert!(!output.contains("m.cpu.pc = 0x100c"));
            assert!(!output.contains("aarch64::decode"));
        }
    }

    #[test]
    fn xpaclri_emits_separate_stripping_with_address_model_validation() {
        let translation = discover(&[region(&[0xd503_20ff])], &[0x1000], 8).unwrap();
        let output = emit_rust(&translation);
        assert_eq!(translation.report().unsupported_instructions, 0);
        assert!(output.contains("m.cpu.strip_lr_instruction_pac(m.memory.address_space_end())?;"));
        assert!(!output.contains("pauth_disabled_hint("));
    }
}
