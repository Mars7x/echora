//! Translate captured NVN programs (ECHORA_SHADER_DIR dumps) and compile each
//! stage on the Vulkan driver, grouping failures by reason.
//!   shader-check stage-*.code
use std::collections::BTreeMap;

use echora_gpu::shader::{self, NoEnvironment, Parameters};
use echora_gpu::vulkan::Gpu;

fn main() {
    let gpu = match Gpu::new() {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    };
    println!("device: {}", gpu.name);
    // Captures carry no dispatch parameters; allow the most shared memory.
    let compute = Parameters {
        shared_memory_size: 0xc000,
        ..Default::default()
    };
    let (mut translated, mut compiled, mut total) = (0, 0, 0);
    let mut failures: BTreeMap<String, usize> = BTreeMap::new();
    for path in std::env::args().skip(1) {
        total += 1;
        let Ok(bytes) = std::fs::read(&path) else {
            *failures.entry("unreadable".into()).or_default() += 1;
            continue;
        };
        let program = match shader::translate(&bytes, compute, &mut NoEnvironment) {
            Ok(program) => program,
            Err(error) => {
                // Group by message without addresses.
                let reason = error.split(" (op").next().unwrap_or(&error).to_string();
                *failures.entry(format!("translate: {reason}")).or_default() += 1;
                continue;
            }
        };
        translated += 1;
        // ECHORA_SHADER_SPIRV=<dir>: keep each translation as <name>.spv.
        if let Some(directory) = std::env::var_os("ECHORA_SHADER_SPIRV") {
            let name = std::path::Path::new(&path).file_name().unwrap_or_default();
            let target = std::path::Path::new(&directory)
                .join(name)
                .with_extension("spv");
            let bytes: Vec<u8> = program
                .spirv
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect();
            if let Err(error) = std::fs::write(&target, bytes) {
                eprintln!("{}: {error}", target.display());
            }
        }
        if std::env::var_os("ECHORA_SHADER_TRACE").is_some() {
            // path stage, then kind:binding:count:index:cbuf:offset per binding.
            let bindings: Vec<String> = program
                .descriptors
                .iter()
                .map(|d| {
                    let fields = [
                        d.kind,
                        d.binding,
                        d.count,
                        d.index,
                        d.cbuf_index,
                        d.cbuf_offset,
                    ];
                    fields.map(|field| field.to_string()).join(":")
                })
                .collect();
            eprintln!("trace {path} {:?} {}", program.stage, bindings.join(" "));
        }
        match gpu.check_stage(&program) {
            Ok(()) => compiled += 1,
            Err(error) => {
                eprintln!("{path}: {error}");
                *failures.entry(format!("driver: {error}")).or_default() += 1;
            }
        }
    }
    println!("{compiled}/{total} compiled on the driver ({translated} translated)");
    let mut sorted: Vec<_> = failures.into_iter().collect();
    sorted.sort_by_key(|entry| std::cmp::Reverse(entry.1));
    for (reason, count) in sorted {
        println!("{count:6} {reason}");
    }
}
