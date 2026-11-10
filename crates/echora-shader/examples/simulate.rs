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
//!   of one loop by one invocation that count as runaway, default 65536);
//! - `push.bin`: the push constants' bytes;
//! - `uniform-SET-BINDING.bin`: each bound uniform buffer's bytes;
//! - `memory-ADDRESS.bin` (ADDRESS in hex, no 0x): device memory from that
//!   address: the page table and every page the dispatch may touch.
//!
//! Each X,Y,Z argument names a workgroup to run (default 0,0,0).

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use echora_shader::sim::{Inputs, Machine};

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
            _ => {}
        }
    }
    let mut uniforms = HashMap::new();
    let mut memory = BTreeMap::new();
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
        } else if let Some(Ok(address)) = stem
            .strip_prefix("memory-")
            .map(|address| u64::from_str_radix(address, 16))
        {
            memory.insert(address, read(&name));
        }
    }
    inputs.uniforms = uniforms;
    inputs.memory = memory;
    let machine = Machine::new(&words);
    let groups: Vec<[u32; 3]> = if arguments.len() > 1 {
        arguments[1..]
            .iter()
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
