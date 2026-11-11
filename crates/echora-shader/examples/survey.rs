//! Translate every dumped program stage and rank the refusals.
//!
//! `cargo run --release -p echora-shader --example survey -- DIR` reads
//! the `stage-<key>.code` / `.control` pairs `echora shaders --dump` (or
//! ECHORA_SHADER_DIR) writes, translates each as its header names it
//! (compute programs take their workgroup, shared and local memory sizes
//! from the control section's words at 0x58..0x6c, as the runtime does),
//! and prints each refusal reason (numbers dropped) with how many
//! programs it stopped. Task and mesh programs need their pipeline state
//! and are counted apart.

use std::collections::BTreeMap;

use echora_shader_abi::Parameters;

fn main() {
    let directory = std::env::args().nth(1).expect("usage: survey DIR");
    let mut reasons: BTreeMap<String, (u32, String)> = BTreeMap::new();
    let (mut total, mut translated, mut unread) = (0u32, 0u32, 0u32);
    let mut entries: Vec<_> = std::fs::read_dir(&directory)
        .expect("read the directory")
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "code")
        })
        .collect();
    entries.sort();
    for path in entries {
        total += 1;
        let Ok(code) = std::fs::read(&path) else {
            unread += 1;
            continue;
        };
        let control = std::fs::read(path.with_extension("control")).unwrap_or_default();
        let word = |offset: usize| {
            control
                .get(offset..offset + 4)
                .map_or(0, |bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
        };
        let mut parameters = Parameters::default();
        if !echora_shader::is_graphics(&code) {
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
    ranked.sort_by(|a, b| b.1.0.cmp(&a.1.0));
    println!("{total} stages, {translated} translated, {unread} unreadable");
    for (reason, (count, example)) in ranked {
        println!("{count:6}  {reason}  (e.g. {example})");
    }
}
