//! Names the symbol a module's relocation binds a slot to, for following
//! PLT stubs while disassembling guest code: `importname EXEFS MODULE SLOT`
//! (SLOT a module offset in hex, a GOT entry a stub loads).
#![allow(clippy::unwrap_used)]

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let [_, exefs, module, slot] = &args[..] else {
        panic!("usage: importname EXEFS MODULE SLOT");
    };
    let wanted = u64::from_str_radix(slot.trim_start_matches("0x"), 16).unwrap();
    let modules = echora_loader::load_exefs(std::path::Path::new(exefs)).unwrap();
    let m = modules.iter().find(|m| &m.name == module).unwrap();
    let view = |offset: u64, size: u64| -> Option<&[u8]> {
        m.segments().into_iter().find_map(|segment| {
            let start = u64::from(segment.memory_offset);
            let end = start + segment.bytes.len() as u64;
            (offset >= start && offset + size <= end).then(|| {
                let from = (offset - start) as usize;
                &segment.bytes[from..from + size as usize]
            })
        })
    };
    let word = |bytes: &[u8], at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
    let dynamic = m.mod0.as_ref().and_then(|m| m.dynamic.as_ref()).unwrap();
    let symbols = dynamic.symbol_table.as_ref().unwrap();
    let strings = symbols.string_table_offset.unwrap();
    for table in dynamic
        .relocation_tables
        .iter()
        .filter(|table| table.kind.ends_with("RELA"))
    {
        let Some(bytes) = view(table.memory_offset, table.byte_size) else {
            continue;
        };
        for entry in bytes.as_chunks::<24>().0 {
            if word(entry, 0) != wanted {
                continue;
            }
            let info = word(entry, 8);
            let symbol = view(
                symbols.memory_offset + (info >> 32) * symbols.entry_size,
                24,
            )
            .unwrap();
            let name_at = strings + u64::from(u32::from_le_bytes(symbol[..4].try_into().unwrap()));
            let name: Vec<u8> = (0..)
                .map_while(|index| view(name_at + index, 1).map(|byte| byte[0]))
                .take_while(|byte| *byte != 0)
                .collect();
            println!(
                "slot {wanted:#x}: type {} symbol {} addend {:#x}",
                info as u32,
                String::from_utf8_lossy(&name),
                word(entry, 16)
            );
        }
    }
}
