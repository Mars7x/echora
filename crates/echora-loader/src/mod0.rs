//! Analysis of module-relative MOD0 and dynamic-section metadata. No address
//! is rebased or relocated; the guest's rtld does that at startup.

use std::collections::BTreeMap;

use serde::Serialize;

use crate::nso::NsoModule;
use crate::{LoaderError, Result};

macro_rules! ensure {
    ($cond:expr, $($fmt:tt)*) => {
        if !$cond {
            return Err(LoaderError::Format(format!($($fmt)*)));
        }
    };
}

macro_rules! bail {
    ($($fmt:tt)*) => {
        return Err(LoaderError::Format(format!($($fmt)*)))
    };
}

/// Attaches a message to a missing value or failed conversion.
trait Context<T> {
    fn context(self, msg: &str) -> Result<T>;
}

impl<T> Context<T> for Option<T> {
    fn context(self, msg: &str) -> Result<T> {
        self.ok_or_else(|| LoaderError::Format(msg.to_owned()))
    }
}

fn u32_at(b: &[u8], at: usize, label: &str) -> Result<u32> {
    b.get(at..at + 4)
        .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
        .context(label)
}

fn i32_at(b: &[u8], at: usize, label: &str) -> Result<i32> {
    u32_at(b, at, label).map(|v| v as i32)
}

fn u64_at(b: &[u8], at: usize, label: &str) -> Result<u64> {
    let lo = u32_at(b, at, label)?;
    let hi = u32_at(b, at + 4, label)?;
    Ok(u64::from(lo) | u64::from(hi) << 32)
}

#[derive(Clone, Debug, Serialize)]
pub struct Mod0Info {
    pub memory_offset: u64,
    pub dynamic_offset: Option<u64>,
    pub bss_start: u64,
    pub bss_end: u64,
    pub unwind_start: u64,
    pub unwind_end: u64,
    pub module_object_offset: Option<u64>,
    pub dynamic: Option<DynamicInfo>,
}

#[derive(Clone, Debug, Serialize)]
pub struct DynamicInfo {
    /// Entries before DT_NULL; pointers remain module-relative guest values.
    pub entries: Vec<DynamicEntry>,
    /// Includes the terminating DT_NULL entry.
    pub byte_size: u64,
    pub needed_libraries: Vec<String>,
    pub soname: Option<String>,
    pub symbol_table: Option<SymbolTableInfo>,
    pub relocation_tables: Vec<RelocationTableInfo>,
    pub init_array: Option<DynamicArrayInfo>,
    pub fini_array: Option<DynamicArrayInfo>,
}

#[derive(Clone, Debug, Serialize)]
pub struct DynamicEntry {
    pub tag: i64,
    pub value: u64,
    pub name: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct SymbolTableInfo {
    pub memory_offset: u64,
    pub entry_size: u64,
    /// A count is available only when a bounded SysV or GNU hash table gives it.
    pub count: Option<u64>,
    /// Which hash tables the module provides.
    pub has_sysv_hash: bool,
    pub has_gnu_hash: bool,
    pub string_table_offset: Option<u64>,
    pub string_table_size: Option<u64>,
    /// Excludes the reserved all-zero symbol at index zero.
    pub undefined_count: Option<u64>,
    pub defined_count: Option<u64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RelocationTableInfo {
    pub kind: String,
    pub memory_offset: u64,
    pub byte_size: u64,
    pub entry_size: u64,
    pub count: u64,
    /// ELF relocation type to count. RELR encodings have no individual type.
    pub types: BTreeMap<u32, u64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct DynamicArrayInfo {
    pub memory_offset: u64,
    pub byte_size: u64,
    pub count: u64,
}

const MAX_DYNAMIC_ENTRIES: usize = 4096;
const MAX_SYMBOLS: u64 = 1_000_000;
const MAX_RELOCATIONS: u64 = 1_000_000;
const MAX_STRING_TABLE: u64 = 32 * 1024 * 1024;
const MAX_LIBRARY_NAME: usize = 4096;

const DT_NEEDED: i64 = 1;
const DT_PLTRELSZ: i64 = 2;
const DT_HASH: i64 = 4;
const DT_STRTAB: i64 = 5;
const DT_SYMTAB: i64 = 6;
const DT_RELA: i64 = 7;
const DT_RELASZ: i64 = 8;
const DT_RELAENT: i64 = 9;
const DT_STRSZ: i64 = 10;
const DT_SYMENT: i64 = 11;
const DT_SONAME: i64 = 14;
const DT_REL: i64 = 17;
const DT_RELSZ: i64 = 18;
const DT_RELENT: i64 = 19;
const DT_PLTREL: i64 = 20;
const DT_JMPREL: i64 = 23;
const DT_INIT_ARRAY: i64 = 25;
const DT_FINI_ARRAY: i64 = 26;
const DT_INIT_ARRAYSZ: i64 = 27;
const DT_FINI_ARRAYSZ: i64 = 28;
const DT_RELRSZ: i64 = 35;
const DT_RELR: i64 = 36;
const DT_RELRENT: i64 = 37;
const DT_GNU_HASH: i64 = 0x6fff_fef5;

fn view<'a>(module: &'a NsoModule, offset: u64, size: u64, label: &str) -> Result<&'a [u8]> {
    for segment in [&module.text, &module.rodata, &module.data] {
        let start = segment.memory_offset as u64;
        let length = segment.bytes.len() as u64;
        if offset >= start && offset - start <= length && size <= length - (offset - start) {
            let begin = (offset - start) as usize;
            return Ok(&segment.bytes[begin..begin + size as usize]);
        }
    }
    bail!("{label} at {offset:#x}, length {size:#x}, is not contained in initialized module data")
}

fn relative(base: u64, displacement: i32, limit: u64, label: &str) -> Result<u64> {
    let value = (base as i64)
        .checked_add(displacement as i64)
        .context("MOD0 relative-address overflow")?;
    ensure!(
        value >= 0 && value as u64 <= limit,
        "{label} is outside the module image"
    );
    Ok(value as u64)
}

pub(crate) fn analyze(module: &NsoModule) -> Result<Option<Mod0Info>> {
    // NSO modules commonly carry their MOD0 image offset in the second word.
    // Some NSOs have no MOD0; a candidate without the magic is simply absent.
    let pointer = match view(module, 4, 4, "MOD0 pointer") {
        Ok(bytes) => u32_at(bytes, 0, "MOD0 pointer")? as u64,
        Err(_) => return Ok(None),
    };
    if pointer == 0 {
        return Ok(None);
    }
    match view(module, pointer, 4, "MOD0 magic") {
        Ok(b"MOD0") => {}
        _ => return Ok(None),
    }
    let header = view(module, pointer, 0x1c, "MOD0 header")?;
    let at = |field, label| {
        relative(
            pointer,
            i32_at(header, field, label)?,
            module.image_size,
            label,
        )
    };
    let dynamic_relative = i32_at(header, 4, "MOD0 dynamic offset")?;
    let dynamic_offset = if dynamic_relative == 0 {
        None
    } else {
        Some(at(4, "MOD0 dynamic offset")?)
    };
    let bss_start = at(8, "MOD0 BSS start")?;
    let bss_end = at(12, "MOD0 BSS end")?;
    ensure!(bss_start <= bss_end, "MOD0 BSS range is reversed");
    let unwind_start = at(16, "MOD0 unwind start")?;
    let unwind_end = at(20, "MOD0 unwind end")?;
    ensure!(unwind_start <= unwind_end, "MOD0 unwind range is reversed");
    if unwind_end != unwind_start {
        view(
            module,
            unwind_start,
            unwind_end - unwind_start,
            "MOD0 unwind data",
        )?;
    }
    let module_object_offset = if i32_at(header, 24, "MOD0 module object")? == 0 {
        None
    } else {
        Some(at(24, "MOD0 module object")?)
    };
    if let Some(offset) = module_object_offset {
        ensure!(
            offset < module.image_size,
            "MOD0 module object points past the image"
        );
    }
    let dynamic = dynamic_offset
        .map(|offset| analyze_dynamic(module, offset))
        .transpose()?;
    Ok(Some(Mod0Info {
        memory_offset: pointer,
        dynamic_offset,
        bss_start,
        bss_end,
        unwind_start,
        unwind_end,
        module_object_offset,
        dynamic,
    }))
}

fn unique(entries: &[DynamicEntry], tag: i64) -> Result<Option<u64>> {
    let mut value = None;
    for entry in entries.iter().filter(|entry| entry.tag == tag) {
        if let Some(previous) = value {
            ensure!(
                previous == entry.value,
                "conflicting {} dynamic entries",
                tag_name(tag)
            );
        } else {
            value = Some(entry.value);
        }
    }
    Ok(value)
}

fn table_pair(
    entries: &[DynamicEntry],
    pointer_tag: i64,
    size_tag: i64,
) -> Result<Option<(u64, u64)>> {
    let pointer = unique(entries, pointer_tag)?;
    let size = unique(entries, size_tag)?;
    match (pointer, size) {
        (None, None) => Ok(None),
        (Some(pointer), Some(size)) => Ok(Some((pointer, size))),
        (None, Some(0)) => Ok(None),
        _ => bail!(
            "{} and {} must describe a complete table",
            tag_name(pointer_tag),
            tag_name(size_tag)
        ),
    }
}

fn analyze_dynamic(module: &NsoModule, offset: u64) -> Result<DynamicInfo> {
    let mut entries = Vec::new();
    let mut byte_size = None;
    for index in 0..MAX_DYNAMIC_ENTRIES {
        let bytes = view(module, offset + index as u64 * 16, 16, "dynamic entry")?;
        let tag = u64_at(bytes, 0, "dynamic tag")? as i64;
        let value = u64_at(bytes, 8, "dynamic value")?;
        if tag == 0 {
            byte_size = Some((index as u64 + 1) * 16);
            break;
        }
        entries.push(DynamicEntry {
            tag,
            value,
            name: tag_name(tag).to_owned(),
        });
    }
    let byte_size =
        byte_size.context("dynamic table has no DT_NULL terminator within the entry limit")?;
    let strings = table_pair(&entries, DT_STRTAB, DT_STRSZ)?;
    let string_bytes = match strings {
        Some((address, length)) => {
            ensure!(
                length <= MAX_STRING_TABLE,
                "dynamic string table exceeds analysis limit"
            );
            Some(view(module, address, length, "dynamic string table")?)
        }
        None => None,
    };
    let mut needed_libraries = Vec::new();
    for entry in entries.iter().filter(|entry| entry.tag == DT_NEEDED) {
        needed_libraries.push(dynamic_string(string_bytes, entry.value)?);
    }
    let soname = unique(&entries, DT_SONAME)?
        .map(|value| dynamic_string(string_bytes, value))
        .transpose()?;
    let symbol_table = analyze_symbols(module, &entries, strings, string_bytes)?;
    let symbol_count = symbol_table.as_ref().and_then(|table| table.count);
    let mut relocation_tables = Vec::new();
    for (kind, address_tag, size_tag, entry_tag, expected_size) in [
        ("RELA", DT_RELA, DT_RELASZ, DT_RELAENT, 24),
        ("REL", DT_REL, DT_RELSZ, DT_RELENT, 16),
        ("RELR", DT_RELR, DT_RELRSZ, DT_RELRENT, 8),
    ] {
        if let Some((address, size)) = table_pair(&entries, address_tag, size_tag)? {
            let entry_size = unique(&entries, entry_tag)?.unwrap_or(expected_size);
            ensure!(
                entry_size == expected_size,
                "unsupported {kind} entry size {entry_size}"
            );
            relocation_tables.push(analyze_relocations(
                module,
                kind,
                address,
                size,
                entry_size,
                symbol_count,
            )?);
        }
    }
    if let Some((address, size)) = table_pair(&entries, DT_JMPREL, DT_PLTRELSZ)? {
        let relocation_kind =
            unique(&entries, DT_PLTREL)?.context("DT_JMPREL has no DT_PLTREL type")?;
        let (kind, entry_size) = match relocation_kind as i64 {
            DT_RELA => ("JMPREL/RELA", 24),
            DT_REL => ("JMPREL/REL", 16),
            _ => bail!("unsupported DT_PLTREL value {relocation_kind}"),
        };
        relocation_tables.push(analyze_relocations(
            module,
            kind,
            address,
            size,
            entry_size,
            symbol_count,
        )?);
    }
    let init_array = analyze_array(module, &entries, DT_INIT_ARRAY, DT_INIT_ARRAYSZ)?;
    let fini_array = analyze_array(module, &entries, DT_FINI_ARRAY, DT_FINI_ARRAYSZ)?;
    Ok(DynamicInfo {
        entries,
        byte_size,
        needed_libraries,
        soname,
        symbol_table,
        relocation_tables,
        init_array,
        fini_array,
    })
}

fn dynamic_string(strings: Option<&[u8]>, offset: u64) -> Result<String> {
    let strings = strings.context("dynamic string reference has no string table")?;
    ensure!(
        offset < strings.len() as u64,
        "dynamic string offset exceeds the string table"
    );
    let tail = &strings[offset as usize..];
    let end = tail
        .iter()
        .take(MAX_LIBRARY_NAME + 1)
        .position(|byte| *byte == 0)
        .context("unterminated or oversized dynamic library name")?;
    Ok(String::from_utf8_lossy(&tail[..end]).into_owned())
}

fn analyze_symbols(
    module: &NsoModule,
    entries: &[DynamicEntry],
    strings: Option<(u64, u64)>,
    string_bytes: Option<&[u8]>,
) -> Result<Option<SymbolTableInfo>> {
    let address = unique(entries, DT_SYMTAB)?;
    let Some(address) = address else {
        ensure!(
            unique(entries, DT_HASH)?.is_none() && unique(entries, DT_GNU_HASH)?.is_none(),
            "symbol hash table has no DT_SYMTAB"
        );
        return Ok(None);
    };
    let entry_size = unique(entries, DT_SYMENT)?.unwrap_or(24);
    ensure!(
        entry_size == 24,
        "unsupported ELF64 symbol entry size {entry_size}"
    );
    let sysv_count = unique(entries, DT_HASH)?
        .map(|address| sysv_symbol_count(module, address))
        .transpose()?;
    let gnu_count = unique(entries, DT_GNU_HASH)?
        .map(|address| gnu_symbol_count(module, address))
        .transpose()?;
    if let (Some(sysv), Some(gnu)) = (sysv_count, gnu_count) {
        ensure!(
            sysv == gnu,
            "SysV and GNU symbol hash tables disagree on the symbol count"
        );
    }
    let count = sysv_count.or(gnu_count);
    let extent = count
        .unwrap_or(1)
        .checked_mul(entry_size)
        .context("symbol table size overflow")?;
    let table = view(module, address, extent, "dynamic symbol table")?;
    let mut undefined_count = None;
    let mut defined_count = None;
    if let Some(count) = count {
        let strings = string_bytes.context("dynamic symbol table has no string table")?;
        let last_terminator = strings
            .iter()
            .rposition(|byte| *byte == 0)
            .context("symbol string table is not NUL-terminated")?;
        let mut undefined = 0;
        let mut defined = 0;
        for index in 0..count as usize {
            let symbol = &table[index * 24..index * 24 + 24];
            let name = u32_at(symbol, 0, "symbol name offset")? as usize;
            // Any suffix before the last NUL is terminated. This avoids scanning
            // overlapping strings once per symbol in adversarial input.
            ensure!(
                name <= last_terminator,
                "dynamic symbol name is outside the terminated string table"
            );
            if index != 0 {
                let section = u16::from_le_bytes([symbol[6], symbol[7]]);
                if section == 0 {
                    undefined += 1;
                } else {
                    defined += 1;
                }
            }
        }
        undefined_count = Some(undefined);
        defined_count = Some(defined);
    }
    Ok(Some(SymbolTableInfo {
        memory_offset: address,
        entry_size,
        count,
        has_sysv_hash: sysv_count.is_some(),
        has_gnu_hash: gnu_count.is_some(),
        string_table_offset: strings.map(|value| value.0),
        string_table_size: strings.map(|value| value.1),
        undefined_count,
        defined_count,
    }))
}

/// ELF symbol type of a function.
const STT_FUNC: u8 = 2;

/// Module offsets of every defined function (`STT_FUNC`) in the dynamic
/// symbol table. Other modules reach these through symbol-resolved GOT slots
/// (`JUMP_SLOT`, `GLOB_DAT`), so no relative relocation points at them.
pub fn defined_functions(module: &NsoModule) -> Result<Vec<u64>> {
    let Some(table) = module
        .mod0
        .as_ref()
        .and_then(|info| info.dynamic.as_ref())
        .and_then(|dynamic| dynamic.symbol_table.as_ref())
    else {
        return Ok(Vec::new());
    };
    let Some(count) = table.count else {
        return Ok(Vec::new());
    };
    let symbols = view(
        module,
        table.memory_offset,
        count * 24,
        "dynamic symbol table",
    )?;
    let mut functions = Vec::new();
    for index in 1..count as usize {
        let symbol = &symbols[index * 24..index * 24 + 24];
        let kind = symbol[4] & 0xf;
        let section = u16::from_le_bytes([symbol[6], symbol[7]]);
        let value = u64_at(symbol, 8, "symbol value")?;
        if kind == STT_FUNC && section != 0 && value != 0 {
            functions.push(value);
        }
    }
    functions.sort_unstable();
    functions.dedup();
    Ok(functions)
}

/// Module offset of the defined dynamic symbol `name`, if the module exports
/// one. Uses the bounded table already validated by [`analyze`].
pub fn defined_symbol(module: &NsoModule, name: &str) -> Result<Option<u64>> {
    let Some(table) = module
        .mod0
        .as_ref()
        .and_then(|info| info.dynamic.as_ref())
        .and_then(|dynamic| dynamic.symbol_table.as_ref())
    else {
        return Ok(None);
    };
    let (Some(count), Some(strings), Some(string_size)) = (
        table.count,
        table.string_table_offset,
        table.string_table_size,
    ) else {
        return Ok(None);
    };
    let symbols = view(
        module,
        table.memory_offset,
        count * 24,
        "dynamic symbol table",
    )?;
    let strings = view(module, strings, string_size, "dynamic string table")?;
    for index in 1..count as usize {
        let symbol = &symbols[index * 24..index * 24 + 24];
        let section = u16::from_le_bytes([symbol[6], symbol[7]]);
        if section == 0 {
            continue;
        }
        let offset = u32_at(symbol, 0, "symbol name offset")? as usize;
        let Some(tail) = strings.get(offset..) else {
            continue;
        };
        let end = tail
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(tail.len());
        if &tail[..end] == name.as_bytes() {
            return Ok(Some(u64_at(symbol, 8, "symbol value")?));
        }
    }
    Ok(None)
}

fn sysv_symbol_count(module: &NsoModule, address: u64) -> Result<u64> {
    let header = view(module, address, 8, "SysV hash header")?;
    let buckets = u32_at(header, 0, "SysV bucket count")? as u64;
    let chains = u32_at(header, 4, "SysV chain count")? as u64;
    ensure!(
        buckets <= MAX_SYMBOLS && chains <= MAX_SYMBOLS,
        "SysV hash table exceeds analysis limit"
    );
    let table = view(
        module,
        address,
        8 + (buckets + chains) * 4,
        "SysV hash table",
    )?;
    for entry in table[8..].as_chunks::<4>().0 {
        let index = u32::from_le_bytes(*entry) as u64;
        ensure!(
            index == 0 || index < chains,
            "SysV hash chain references an absent symbol"
        );
    }
    Ok(chains)
}

fn gnu_symbol_count(module: &NsoModule, address: u64) -> Result<u64> {
    let header = view(module, address, 16, "GNU hash header")?;
    let buckets = u32_at(header, 0, "GNU bucket count")? as u64;
    let first_symbol = u32_at(header, 4, "GNU first symbol")? as u64;
    let bloom_words = u32_at(header, 8, "GNU bloom size")? as u64;
    ensure!(
        buckets <= MAX_SYMBOLS && first_symbol <= MAX_SYMBOLS && bloom_words <= MAX_SYMBOLS,
        "GNU hash table exceeds analysis limit"
    );
    ensure!(bloom_words != 0, "GNU hash table has an empty bloom filter");
    let prefix_size = 16 + bloom_words * 8 + buckets * 4;
    let prefix = view(module, address, prefix_size, "GNU hash table prefix")?;
    let bucket_offset = (16 + bloom_words * 8) as usize;
    let mut last_bucket = None;
    for bytes in prefix[bucket_offset..].as_chunks::<4>().0 {
        let symbol = u32::from_le_bytes(*bytes) as u64;
        if symbol != 0 {
            ensure!(
                symbol >= first_symbol && symbol < MAX_SYMBOLS,
                "GNU hash bucket is outside the symbol limit"
            );
            last_bucket = Some(last_bucket.map_or(symbol, |previous: u64| previous.max(symbol)));
        }
    }
    let Some(mut symbol) = last_bucket else {
        return Ok(first_symbol);
    };
    // GNU-hashed symbols are grouped consecutively by bucket. The greatest
    // nonempty bucket index identifies the final chain in the symbol table.
    loop {
        ensure!(
            symbol < MAX_SYMBOLS,
            "GNU hash chain has no terminator within the symbol limit"
        );
        let offset = address + prefix_size + (symbol - first_symbol) * 4;
        let hash = u32_at(
            view(module, offset, 4, "GNU hash chain")?,
            0,
            "GNU hash value",
        )?;
        symbol += 1;
        if hash & 1 != 0 {
            return Ok(symbol);
        }
    }
}

fn analyze_relocations(
    module: &NsoModule,
    kind: &str,
    address: u64,
    size: u64,
    entry_size: u64,
    symbol_count: Option<u64>,
) -> Result<RelocationTableInfo> {
    ensure!(
        size.is_multiple_of(entry_size),
        "{kind} table length is not a whole number of entries"
    );
    let count = size / entry_size;
    ensure!(
        count <= MAX_RELOCATIONS,
        "{kind} table exceeds relocation analysis limit"
    );
    let table = view(module, address, size, kind)?;
    let mut types = BTreeMap::new();
    if entry_size != 8 {
        for relocation in table.chunks_exact(entry_size as usize) {
            let info = u64_at(relocation, 8, "relocation info")?;
            let relocation_type = info as u32;
            let symbol = info >> 32;
            if let Some(symbol_count) = symbol_count {
                ensure!(
                    symbol == 0 || symbol < symbol_count,
                    "{kind} relocation references an absent symbol"
                );
            }
            *types.entry(relocation_type).or_insert(0) += 1;
        }
    }
    Ok(RelocationTableInfo {
        kind: kind.to_owned(),
        memory_offset: address,
        byte_size: size,
        entry_size,
        count,
        types,
    })
}

fn analyze_array(
    module: &NsoModule,
    entries: &[DynamicEntry],
    address_tag: i64,
    size_tag: i64,
) -> Result<Option<DynamicArrayInfo>> {
    let Some((address, size)) = table_pair(entries, address_tag, size_tag)? else {
        return Ok(None);
    };
    ensure!(
        size % 8 == 0 && size / 8 <= MAX_SYMBOLS,
        "invalid or oversized dynamic function-pointer array"
    );
    view(module, address, size, tag_name(address_tag))?;
    Ok(Some(DynamicArrayInfo {
        memory_offset: address,
        byte_size: size,
        count: size / 8,
    }))
}

fn tag_name(tag: i64) -> &'static str {
    match tag {
        0 => "DT_NULL",
        1 => "DT_NEEDED",
        2 => "DT_PLTRELSZ",
        3 => "DT_PLTGOT",
        4 => "DT_HASH",
        5 => "DT_STRTAB",
        6 => "DT_SYMTAB",
        7 => "DT_RELA",
        8 => "DT_RELASZ",
        9 => "DT_RELAENT",
        10 => "DT_STRSZ",
        11 => "DT_SYMENT",
        12 => "DT_INIT",
        13 => "DT_FINI",
        14 => "DT_SONAME",
        15 => "DT_RPATH",
        16 => "DT_SYMBOLIC",
        17 => "DT_REL",
        18 => "DT_RELSZ",
        19 => "DT_RELENT",
        20 => "DT_PLTREL",
        21 => "DT_DEBUG",
        22 => "DT_TEXTREL",
        23 => "DT_JMPREL",
        24 => "DT_BIND_NOW",
        25 => "DT_INIT_ARRAY",
        26 => "DT_FINI_ARRAY",
        27 => "DT_INIT_ARRAYSZ",
        28 => "DT_FINI_ARRAYSZ",
        29 => "DT_RUNPATH",
        30 => "DT_FLAGS",
        32 => "DT_PREINIT_ARRAY",
        33 => "DT_PREINIT_ARRAYSZ",
        35 => "DT_RELRSZ",
        36 => "DT_RELR",
        37 => "DT_RELRENT",
        0x6fff_fef5 => "DT_GNU_HASH",
        0x6fff_fff0 => "DT_VERSYM",
        0x6fff_fff9 => "DT_RELACOUNT",
        0x6fff_fffa => "DT_RELCOUNT",
        0x6fff_fffb => "DT_FLAGS_1",
        0x6fff_fffc => "DT_VERDEF",
        0x6fff_fffd => "DT_VERDEFNUM",
        0x6fff_fffe => "DT_VERNEED",
        0x6fff_ffff => "DT_VERNEEDNUM",
        _ => "UNKNOWN",
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::nso::{Compression, Segment};

    fn put32(bytes: &mut [u8], at: usize, value: u32) {
        bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
    }
    fn put64(bytes: &mut [u8], at: usize, value: u64) {
        bytes[at..at + 8].copy_from_slice(&value.to_le_bytes());
    }
    fn segment(offset: u32, bytes: Vec<u8>) -> Segment {
        Segment {
            memory_offset: offset,
            file_offset: 0x100,
            stored_size: bytes.len() as u32,
            compression: Compression::None,
            sha256: [0; 32],
            hash_verified: false,
            bytes,
        }
    }

    fn fixture() -> NsoModule {
        let mut text = vec![0; 0x100];
        put32(&mut text, 4, 0x1000);
        let mut rodata = vec![0; 0x200];
        rodata[..4].copy_from_slice(b"MOD0");
        put32(&mut rodata, 4, 0x1000); // dynamic table in data
        put32(&mut rodata, 8, 0x1100); // BSS at 0x2100
        put32(&mut rodata, 12, 0x1200);
        let data = vec![0; 0x100]; // DT_NULL
        NsoModule {
            name: "synthetic".to_owned(),
            build_id: [0; 32],
            flags: 0,
            text: segment(0, text),
            rodata: segment(0x1000, rodata),
            data: segment(0x2000, data),
            bss_size: 0x100,
            image_size: 0x3000,
            mod0: None,
        }
    }

    #[test]
    fn finds_mod0_without_mutating_relocation_targets() {
        let mut module = fixture();
        // One RELA entry in rodata, targeting bytes in data. Analysis must not
        // write its addend into the target.
        put64(&mut module.data.bytes, 0, DT_RELA as u64);
        put64(&mut module.data.bytes, 8, 0x1080);
        put64(&mut module.data.bytes, 16, DT_RELASZ as u64);
        put64(&mut module.data.bytes, 24, 24);
        put64(&mut module.rodata.bytes, 0x80, 0x2080);
        put64(&mut module.rodata.bytes, 0x88, 1027);
        put64(&mut module.rodata.bytes, 0x90, 0x12345678);
        let before = module.data.bytes.clone();
        let info = analyze(&module).unwrap().unwrap();
        let relocations = &info.dynamic.unwrap().relocation_tables;
        assert_eq!(relocations[0].count, 1);
        assert_eq!(relocations[0].types[&1027], 1);
        assert_eq!(module.data.bytes, before);
    }

    #[test]
    fn signed_mod0_relative_offsets_can_point_backwards() {
        let mut module = fixture();
        put32(&mut module.rodata.bytes, 4, (-0x0fc0i32) as u32);
        let info = analyze(&module).unwrap().unwrap();
        assert_eq!(info.dynamic_offset, Some(0x40));
    }

    #[test]
    fn rejects_dynamic_table_that_runs_into_uninitialized_memory() {
        let mut module = fixture();
        for entry in module.data.bytes.as_chunks_mut::<16>().0 {
            entry[0] = 30;
        }
        assert!(analyze(&module).is_err());
    }

    #[test]
    fn rejects_relocation_ranges_outside_initialized_segments() {
        let mut module = fixture();
        put64(&mut module.data.bytes, 0, DT_RELA as u64);
        put64(&mut module.data.bytes, 8, 0x2ff0);
        put64(&mut module.data.bytes, 16, DT_RELASZ as u64);
        put64(&mut module.data.bytes, 24, 24);
        assert!(analyze(&module).is_err());
    }

    #[test]
    fn malformed_gnu_hash_cannot_cause_unbounded_chain_walk() {
        let mut module = fixture();
        put32(&mut module.rodata.bytes, 0x80, 1);
        put32(&mut module.rodata.bytes, 0x84, 1);
        put32(&mut module.rodata.bytes, 0x88, 1);
        put32(&mut module.rodata.bytes, 0x98, 1);
        // All following chain words are even, with no low-bit terminator.
        assert!(gnu_symbol_count(&module, 0x1080).is_err());
    }

    #[test]
    fn gnu_hash_count_includes_unhashed_initial_symbols() {
        let mut module = fixture();
        put32(&mut module.rodata.bytes, 0x80, 1);
        put32(&mut module.rodata.bytes, 0x84, 2);
        put32(&mut module.rodata.bytes, 0x88, 1);
        put32(&mut module.rodata.bytes, 0x98, 2);
        put32(&mut module.rodata.bytes, 0x9c, 0x10);
        put32(&mut module.rodata.bytes, 0xa0, 0x21);
        assert_eq!(gnu_symbol_count(&module, 0x1080).unwrap(), 4);
    }

    /// The layout OoT's modules use: GNU hash only (no DT_HASH) and RELR.
    #[test]
    fn gnu_only_symbol_table_with_relr() {
        let mut module = fixture();
        let ro = &mut module.rodata.bytes;
        // GNU hash at 0x1080: 1 bucket, first hashed symbol 2, 1 bloom word,
        // bucket -> symbol 2, chain words for symbols 2 and 3 (3 ends it).
        put32(ro, 0x80, 1);
        put32(ro, 0x84, 2);
        put32(ro, 0x88, 1);
        put32(ro, 0x98, 2);
        put32(ro, 0x9c, 0x10);
        put32(ro, 0xa0, 0x21);
        // Symbols at 0x1100: [null, foo (undefined), bar (defined), unnamed].
        put32(ro, 0x100 + 24, 1);
        put32(ro, 0x100 + 48, 5);
        ro[0x100 + 48 + 6] = 1;
        put64(ro, 0x100 + 48 + 8, 0x40);
        ro[0x100 + 72 + 6] = 1;
        put64(ro, 0x100 + 72 + 8, 0x50);
        ro[0x180..0x189].copy_from_slice(b"\0foo\0bar\0");
        let d = &mut module.data.bytes;
        for (i, (tag, value)) in [
            (DT_SYMTAB, 0x1100),
            (DT_STRTAB, 0x1180),
            (DT_STRSZ, 0x10),
            (DT_GNU_HASH, 0x1080),
            (DT_RELR, 0x11a0),
            (DT_RELRSZ, 16),
        ]
        .into_iter()
        .enumerate()
        {
            put64(d, i * 16, tag as u64);
            put64(d, i * 16 + 8, value);
        }
        module.mod0 = analyze(&module).unwrap();
        let dynamic = module.mod0.as_ref().unwrap().dynamic.as_ref().unwrap();
        let symbols = dynamic.symbol_table.as_ref().unwrap();
        assert_eq!(symbols.count, Some(4));
        assert!(symbols.has_gnu_hash && !symbols.has_sysv_hash);
        assert_eq!(
            (symbols.defined_count, symbols.undefined_count),
            (Some(2), Some(1))
        );
        assert_eq!(dynamic.relocation_tables[0].kind, "RELR");
        assert_eq!(dynamic.relocation_tables[0].count, 2);
        assert_eq!(defined_symbol(&module, "bar").unwrap(), Some(0x40));
        assert_eq!(defined_symbol(&module, "foo").unwrap(), None);
    }

    #[test]
    fn reports_unknown_dynamic_tags_without_executing_them() {
        let mut module = fixture();
        put64(&mut module.data.bytes, 0, 0x6000_1234);
        put64(&mut module.data.bytes, 8, u64::MAX);
        let dynamic = analyze(&module).unwrap().unwrap().dynamic.unwrap();
        assert_eq!(dynamic.entries[0].name, "UNKNOWN");
        assert_eq!(dynamic.entries[0].value, u64::MAX);
    }
}
