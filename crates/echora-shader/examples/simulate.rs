//! Runs workgroups of a translated compute program on the CPU and reports
//! what would hang or fault on a GPU (see `echora_shader::sim`).
//!
//! ```text
//! cargo run -p echora-shader --features sim --example simulate -- DUMP [X,Y,Z ...]
//! ```
//!
//! DUMP is a directory holding one dispatch:
//! - `program.spv`: the SPIR-V (little-endian words);
//! - `dispatch.txt`: lines `groups X Y Z` (the dispatch's workgroup counts),
//!   optionally `subgroup N` (default 32) and `loop_limit N` (iterations
//!   of one loop by one invocation that count as runaway, default 65536)
//!   and `spec ID VALUE` (a specialization constant, else its default);
//! - `push.bin`: the push constants' bytes;
//! - `uniform-SET-BINDING.bin`: each bound uniform buffer's bytes;
//! - `memory-ADDRESS.bin` (ADDRESS in hex, no 0x): device memory from that
//!   address: the page table and every page the dispatch may touch;
//! - `image-SET-BINDING.txt` (`WIDTH HEIGHT DEPTH LAYERS`) and
//!   `image-SET-BINDING.bin`: a bound texture's or image's level 0, layers
//!   (or 3D slices) one after another, each texel four little-endian u32:
//!   the components as the shader reads them (float bits for float, unorm
//!   and snorm formats, integers for integer ones; missing ones 0, 0, 0,
//!   1). Textures and images without these read as zeros;
//! - `image-SET-BINDING-ELEMENT.txt` and `.bin`: likewise, element ELEMENT
//!   of an image array (a texture pool view; its flags buffer is a
//!   `uniform-` file).
//!
//! Each X,Y,Z argument names a workgroup to run (default 0,0,0). Each
//! `watch=START..END` argument (hex addresses, or a `watch START END` line
//! in dispatch.txt) lists every store and atomic into that range, guest
//! (through the page table) or device: the invocation, the instruction
//! (index in the entry function, as the ends), its SPIR-V opcode and the
//! bytes before and after; then each watched word's first and last
//! values.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use echora_shader::sim::{Image, Inputs, Machine};

fn main() {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let Some(directory) = arguments.first() else {
        eprintln!("usage: simulate DUMP [X,Y,Z ...]");
        std::process::exit(2);
    };
    let directory = Path::new(directory);
    let read = |name: &str| {
        std::fs::read(directory.join(name)).unwrap_or_else(|error| {
            eprintln!("{name}: {error}");
            std::process::exit(2);
        })
    };
    let words: Vec<u32> = read("program.spv")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&word| u32::from_le_bytes(word))
        .collect();
    let mut inputs = Inputs {
        workgroup_count: [1, 1, 1],
        push: std::fs::read(directory.join("push.bin")).unwrap_or_default(),
        ..Inputs::default()
    };
    let dispatch = String::from_utf8_lossy(&read("dispatch.txt")).into_owned();
    for line in dispatch.lines() {
        let fields: Vec<u32> = line
            .split_whitespace()
            .skip(1)
            .filter_map(|field| field.parse().ok())
            .collect();
        match (line.split_whitespace().next(), fields.as_slice()) {
            (Some("groups"), &[x, y, z]) => inputs.workgroup_count = [x, y, z],
            (Some("subgroup"), &[size]) => inputs.subgroup_size = size,
            (Some("loop_limit"), &[limit]) => inputs.loop_limit = limit,
            (Some("spec"), &[id, value]) => {
                inputs.specialization.insert(id, value);
            }
            (Some("watch"), _) => {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if let [_, start, end] = parts[..]
                    && let Some(range) = hex_range(&format!("{start}..{end}"))
                {
                    inputs.watch.push(range);
                }
            }
            _ => {}
        }
    }
    let mut uniforms = HashMap::new();
    let mut memory = BTreeMap::new();
    let mut images = HashMap::new();
    let mut image_arrays = HashMap::new();
    for entry in std::fs::read_dir(directory).into_iter().flatten().flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(stem) = name.strip_suffix(".bin") else {
            continue;
        };
        if let Some(key) = stem.strip_prefix("uniform-") {
            let parts: Vec<u32> = key
                .split('-')
                .filter_map(|part| part.parse().ok())
                .collect();
            if let [set, binding] = parts[..] {
                uniforms.insert((set, binding), read(&name));
            }
        } else if let Some(key) = stem.strip_prefix("image-") {
            let parts: Vec<u32> = key
                .split('-')
                .filter_map(|part| part.parse().ok())
                .collect();
            let header = String::from_utf8_lossy(&read(&format!("image-{key}.txt"))).into_owned();
            let size: Vec<u32> = header
                .split_whitespace()
                .filter_map(|field| field.parse().ok())
                .collect();
            if let [width, height, depth, layers] = size[..] {
                let texels = read(&name)
                    .as_chunks::<16>()
                    .0
                    .iter()
                    .map(|texel| {
                        let word = |n: usize| {
                            u32::from_le_bytes([
                                texel[4 * n],
                                texel[4 * n + 1],
                                texel[4 * n + 2],
                                texel[4 * n + 3],
                            ])
                        };
                        [word(0), word(1), word(2), word(3)]
                    })
                    .collect();
                let image = Image {
                    width,
                    height,
                    depth,
                    layers,
                    texels,
                };
                match parts[..] {
                    [set, binding] => {
                        images.insert((set, binding), image);
                    }
                    [set, binding, element] => {
                        image_arrays.insert((set, binding, element), image);
                    }
                    _ => {}
                }
            }
        } else if let Some(Ok(address)) = stem
            .strip_prefix("memory-")
            .map(|address| u64::from_str_radix(address, 16))
        {
            memory.insert(address, read(&name));
        }
    }
    inputs.uniforms = uniforms;
    inputs.memory = memory;
    inputs.images = images;
    inputs.image_arrays = image_arrays;
    let machine = Machine::new(&words);
    for argument in &arguments[1..] {
        if let Some(range) = argument.strip_prefix("watch=").and_then(hex_range) {
            inputs.watch.push(range);
        }
    }
    let group_arguments: Vec<&String> = arguments[1..]
        .iter()
        .filter(|argument| !argument.starts_with("watch="))
        .collect();
    let groups: Vec<[u32; 3]> = if !group_arguments.is_empty() {
        group_arguments
            .into_iter()
            .filter_map(|group| {
                let parts: Vec<u32> = group
                    .split(',')
                    .filter_map(|part| part.parse().ok())
                    .collect();
                (parts.len() == 3).then(|| [parts[0], parts[1], parts[2]])
            })
            .collect()
    } else {
        vec![[0, 0, 0]]
    };
    let mut failed = false;
    for group in groups {
        inputs.workgroup_id = group;
        let report = machine.simulate(&inputs);
        println!(
            "workgroup {group:?} ({:?} invocations): {} barriers passed",
            machine.local_size,
            report.barriers.len()
        );
        // Where invocations ended, as runs of invocations.
        let mut start = 0;
        for n in 1..=report.ends.len() {
            if n == report.ends.len() || report.ends[n] != report.ends[start] {
                println!(
                    "  invocations {start}-{} ended at instruction {}",
                    n - 1,
                    report.ends[start]
                );
                start = n;
            }
        }
        // The uniform buffer words read, as ranges per binding.
        let mut reads = report.uniform_reads.iter().peekable();
        while let Some(&(set, binding, first)) = reads.next() {
            let mut last = first;
            while let Some(&&(next_set, next_binding, next)) = reads.peek() {
                if (next_set, next_binding, next) != (set, binding, last + 4) {
                    break;
                }
                last = next;
                reads.next();
            }
            println!("  uniform {set}/{binding} read at {first:#x}..={last:#x}");
        }
        // Guest pages the page table does not map: the program reads 0
        // from them and its stores there are dropped.
        let misses: u32 = report.unmapped_pages.values().sum();
        for (&page, &count) in &report.unmapped_pages {
            println!(
                "  note: guest page {:#x} (address {:#x}) is not mapped: {count} accesses",
                page,
                page << echora_shader_abi::PAGE_SHIFT
            );
        }
        if misses > 0 {
            println!(
                "  note: {misses} accesses to {} unmapped pages",
                report.unmapped_pages.len()
            );
        }
        // Watched writes, then each watched word's first and last value.
        let hex = |bytes: &[u8]| -> String {
            bytes
                .iter()
                .rev()
                .map(|byte| format!("{byte:02x}"))
                .collect()
        };
        let mut words: BTreeMap<u64, (Vec<u8>, Vec<u8>, usize)> = BTreeMap::new();
        for write in &report.writes {
            let guest = write
                .guest
                .map_or_else(|| "unmapped".to_string(), |guest| format!("{guest:#x}"));
            println!(
                "  write: invocation {} instruction {} (op {}) guest {guest} device {:#x}: \
                 {} -> {}",
                write.invocation,
                write.instruction,
                write.opcode,
                write.device,
                hex(&write.old),
                hex(&write.new)
            );
            let at = write.guest.unwrap_or(write.device);
            let entry = words
                .entry(at)
                .or_insert_with(|| (write.old.clone(), Vec::new(), 0));
            entry.1.clone_from(&write.new);
            entry.2 += 1;
        }
        for (address, (first, last, count)) in &words {
            println!(
                "  watched {address:#x}: {} -> {} ({count} writes)",
                hex(first),
                hex(last)
            );
        }
        for note in &report.notes {
            println!("  note: {note}");
        }
        for problem in &report.problems {
            println!("  PROBLEM: {problem}");
        }
        failed |= !report.problems.is_empty();
    }
    std::process::exit(i32::from(failed));
}

/// `START..END` in hex (with or without 0x).
fn hex_range(text: &str) -> Option<std::ops::Range<u64>> {
    let (start, end) = text.trim().split_once("..")?;
    let number = |text: &str| {
        let text = text.trim();
        u64::from_str_radix(text.strip_prefix("0x").unwrap_or(text), 16).ok()
    };
    Some(number(start)?..number(end)?)
}
