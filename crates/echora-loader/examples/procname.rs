//! Finds the function an SDK name table maps a name to, for disassembling
//! guest code: `procname EXEFS MODULE NAME`. Prints each relocated slot that
//! points at the NUL-terminated name and the module offset the next slot
//! (a `{name, function}` pair's function) relocates to.
#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let [_, exefs, module, name] = &args[..] else {
        panic!("usage: procname EXEFS MODULE NAME");
    };
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
    // Every relative relocation: slot -> module offset it points at.
    let mut targets: BTreeMap<u64, u64> = BTreeMap::new();
    let dynamic = m.mod0.as_ref().and_then(|m| m.dynamic.as_ref()).unwrap();
    for table in &dynamic.relocation_tables {
        let Some(bytes) = view(table.memory_offset, table.byte_size) else {
            continue;
        };
        match table.kind.as_str() {
            "RELA" => {
                for entry in bytes.as_chunks::<24>().0 {
                    if word(entry, 8) as u32 == 1027 {
                        targets.insert(word(entry, 0), word(entry, 16));
                    }
                }
            }
            "RELR" => {
                let mut cursor = 0u64;
                let push = |slot: u64, targets: &mut BTreeMap<u64, u64>| {
                    if let Some(value) = view(slot, 8) {
                        targets.insert(slot, word(value, 0));
                    }
                };
                for chunk in bytes.as_chunks::<8>().0 {
                    let entry = word(chunk, 0);
                    if entry & 1 == 0 {
                        push(entry, &mut targets);
                        cursor = entry + 8;
                    } else {
                        for bit in 1..64 {
                            if entry >> bit & 1 != 0 {
                                push(cursor + (bit - 1) * 8, &mut targets);
                            }
                        }
                        cursor += 63 * 8;
                    }
                }
            }
            _ => {}
        }
    }
    let wanted = [name.as_bytes(), b"\0"].concat();
    for segment in m.segments() {
        let mut from = 0;
        while let Some(at) = segment.bytes[from..]
            .windows(wanted.len())
            .position(|window| window == wanted.as_slice())
        {
            let offset = u64::from(segment.memory_offset) + (from + at) as u64;
            for (slot, target) in &targets {
                if *target == offset {
                    let function = targets.get(&(slot + 8)).copied();
                    println!(
                        "name at {offset:#x}, slot {slot:#x}, next slot -> {:#x?}",
                        function
                    );
                }
            }
            from += at + 1;
        }
    }
}
