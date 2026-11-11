//! Translate every dumped program stage and rank the refusals.
//!
//! `cargo run --release -p echora-shader --example survey -- DIR` reads
//! the `stage-<key>.code` / `.control` pairs `echora shaders --dump` (or
//! ECHORA_SHADER_DIR) writes, translates each as its header names it
//! (compute programs take their workgroup, shared and local memory sizes
//! from the control section's words at 0x58..0x6c, as the runtime does),
//! and prints each refusal reason (numbers dropped) with how many
//! programs it stopped. Task and mesh programs are told apart as the
//! runtime does and translated against the state their own control
//! section gives.

use std::collections::BTreeMap;

use echora_shader_abi::{MeshParameters, Parameters, Stage};

fn main() {
    let directory = std::env::args().nth(1).expect("usage: survey DIR");
    let mut reasons: BTreeMap<String, (u32, String)> = BTreeMap::new();
    let (mut total, mut translated, mut unread) = (0u32, 0u32, 0u32);
    let (mut mesh_count, mut task_count) = (0u32, 0u32);
    let mut entries: Vec<_> = std::fs::read_dir(&directory)
        .expect("read the directory")
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "code")
        })
        .collect();
    entries.sort();
    // A task program's INIT word (+0x58) is the mesh stage's when the two
    // are bound together; the pairs are not in the dump, so each mesh
    // program in the task program's slot (header kind 3, which NVN uses
    // when a task program holds the vertex slot) is translated with the
    // task program passing the most payload, and each task program
    // against the first such mesh program's output counts.
    let (mut task_init, mut paired_sizes) = (None::<u32>, None::<u32>);
    for path in &entries {
        let (Ok(code), Ok(control)) = (
            std::fs::read(path),
            std::fs::read(path.with_extension("control")),
        ) else {
            continue;
        };
        let word = |offset: usize| {
            control
                .get(offset..)
                .and_then(|rest| rest.first_chunk::<4>())
                .map_or(0, |bytes| u32::from_le_bytes(*bytes))
        };
        let kind = code
            .first_chunk::<4>()
            .map_or(0, |word| (u32::from_le_bytes(*word) >> 10) & 0xf);
        if !echora_shader::is_graphics(&code) {
            continue;
        }
        if kind == 1 && word(0x60) == 0 && word(0x58) & 0x3f != 0 {
            let passed = |init: u32| init >> 22 & 0x1ff;
            if task_init.is_none_or(|init| passed(word(0x58)) > passed(init)) {
                task_init = Some(word(0x58));
            }
        }
        if kind == 3 && word(0x60) != 0 && paired_sizes.is_none() {
            paired_sizes = Some(word(0x60));
        }
    }
    let outputs = |sizes: u32| {
        let (vertices, primitives) = (sizes & 0xffff, sizes >> 16);
        4 | (vertices & 0x1ff) << 4 | (primitives & 0x3ff) << 13
    };
    for path in entries {
        total += 1;
        let Ok(code) = std::fs::read(&path) else {
            unread += 1;
            continue;
        };
        let control = std::fs::read(path.with_extension("control")).unwrap_or_default();
        let word = |offset: usize| {
            control
                .get(offset..)
                .and_then(|rest| rest.first_chunk::<4>())
                .map_or(0, |bytes| u32::from_le_bytes(*bytes))
        };
        let mut parameters = Parameters::default();
        // Task and mesh programs as the runtime's `bound_mesh_program` tells
        // them apart (header kind, control words): a mesh program has its
        // vertex and primitive counts at +0x60, a task program its threads
        // at +0x58 bits 5:0.
        let kind = code
            .first_chunk::<4>()
            .map_or(0, |word| (u32::from_le_bytes(*word) >> 10) & 0xf);
        let graphics = echora_shader::is_graphics(&code);
        let sizes = word(0x60);
        if graphics && matches!(kind, 1 | 3) && sizes != 0 {
            mesh_count += 1;
            let init = match task_init {
                Some(init) if kind == 3 => init,
                _ => word(0x58) & !0x3f,
            };
            parameters.stage_slot = Some(Stage::Mesh);
            parameters.mesh = Some(MeshParameters {
                task: false,
                registers: [outputs(sizes), (word(0x58) & 0x3f) << 20, init],
                primitive_header: None,
            });
        } else if graphics && kind == 1 && word(0x58) & 0x3f != 0 {
            task_count += 1;
            parameters.stage_slot = Some(Stage::Task);
            parameters.mesh = Some(MeshParameters {
                task: true,
                registers: [outputs(paired_sizes.unwrap_or(0)), 0, word(0x58)],
                primitive_header: None,
            });
        } else if graphics && kind == 3 {
            eprintln!("{} kind 3 with +0x60 = 0", path.display());
        }
        if !graphics {
            let workgroup = [word(0x58), word(0x5c), word(0x60)];
            if workgroup.iter().all(|size| (1..=1024).contains(size)) {
                parameters.workgroup_size = Some(workgroup);
            }
            parameters.shared_memory_size = Some(word(0x64));
            parameters.local_memory_size = Some(word(0x68));
        }
        let key = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("")
            .trim_start_matches("stage-")
            .to_string();
        match echora_shader::translate(&code, &parameters) {
            Ok(_) => translated += 1,
            Err(error) => {
                if std::env::var_os("SURVEY_RAW").is_some() {
                    eprintln!("{key} {} at {:?}", error.reason, error.offset);
                }
                // One bucket per reason with its numbers dropped.
                let reason: String = error
                    .reason
                    .split_whitespace()
                    .map(|part| {
                        if part.starts_with("0x")
                            || part.chars().all(|c| c.is_ascii_digit() || c == ',')
                        {
                            "N"
                        } else {
                            part
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                let entry = reasons.entry(reason).or_insert((0, key));
                entry.0 += 1;
            }
        }
    }
    let mut ranked: Vec<_> = reasons.into_iter().collect();
    ranked.sort_by_key(|(_, (count, _))| std::cmp::Reverse(*count));
    println!(
        "{total} stages ({mesh_count} mesh, {task_count} task), {translated} translated, {unread} unreadable"
    );
    for (reason, (count, example)) in ranked {
        println!("{count:6}  {reason}  (e.g. {example})");
    }
}
