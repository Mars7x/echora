//! Offline check: translate captured program files (ECHORA_SHADER_DIR dumps)
//! and summarize the failure reasons, without a GPU. With `-o DIR`,
//! successful programs are written as `<name>.spv`.
//!   translate-dump [-o DIR] stage-*.code
use std::collections::BTreeMap;
use std::path::Path;

use echora_gpu::shader::{self, NoEnvironment, Parameters};

fn main() {
    let mut arguments: Vec<String> = std::env::args().skip(1).collect();
    if arguments.is_empty() {
        eprintln!("usage: translate-dump [-o DIR] program.code...");
        std::process::exit(2);
    }
    let out_dir = (arguments.len() > 2 && arguments[0] == "-o").then(|| {
        let directory = arguments.remove(1);
        arguments.remove(0);
        directory
    });
    let mut reasons: BTreeMap<String, usize> = BTreeMap::new();
    let mut stages: BTreeMap<String, usize> = BTreeMap::new();
    let (mut translated, mut total) = (0, 0);
    for path in &arguments {
        total += 1;
        let Ok(bytes) = std::fs::read(path) else {
            *reasons.entry("unreadable".into()).or_default() += 1;
            continue;
        };
        let parameters = Parameters {
            // Captures do not record dispatch parameters; assume the
            // largest shared memory a compute program can use.
            shared_memory_size: 0xc000,
            // A capture's control section (<name>.control beside it) gives
            // a compute program's local memory size at 0x68, as at run time.
            local_memory_size: control_word(path, 0x68).unwrap_or(0),
            // As the renderer translates them (ECHORA_RESOLUTION).
            resolution_scale: echora_gpu::renderer::Scale::from_env().terms(),
            ..Default::default()
        };
        // As at run time: a program whose control section gives mesh output
        // sizes (+0x60) is a mesh program. Without a capture of its draw, a
        // program that reads the stage buffer otherwise is taken as a task
        // program (its control +0x58 is then SET_MESH_INIT_SHADER).
        let sizes = control_word(path, 0x60).unwrap_or(0);
        let threads = control_word(path, 0x58).unwrap_or(0);
        let mesh = |task: bool| {
            let (vertices, primitives) = (sizes & 0xffff, sizes >> 16);
            shader::MeshParameters {
                task,
                registers: [
                    4 | (vertices & 0x1ff) << 4 | (primitives & 0x3ff) << 13,
                    (threads & 0x3f) << 20,
                    if task { threads } else { threads & !0x3f },
                ],
                primitive_header: None,
            }
        };
        let parameters = Parameters {
            mesh: (sizes != 0).then(|| mesh(false)),
            ..parameters
        };
        let mut result = shader::translate(&bytes, parameters, &mut NoEnvironment);
        if sizes == 0
            && matches!(&result, Err(error) if error.contains("outside task and mesh programs"))
        {
            let parameters = Parameters {
                mesh: Some(mesh(true)),
                ..parameters
            };
            result = shader::translate(&bytes, parameters, &mut NoEnvironment);
        }
        match result {
            Ok(program) => {
                translated += 1;
                *stages.entry(format!("{:?}", program.stage)).or_default() += 1;
                if let Some(directory) = &out_dir {
                    let name = Path::new(path).file_name().unwrap_or_default();
                    let mut target = Path::new(directory).join(name).into_os_string();
                    target.push(".spv");
                    let spirv: Vec<u8> = program
                        .spirv
                        .iter()
                        .flat_map(|word| word.to_le_bytes())
                        .collect();
                    if let Err(error) = std::fs::write(&target, spirv) {
                        eprintln!("{}: {error}", Path::new(&target).display());
                    }
                }
            }
            Err(error) => {
                // Group by message without the per-program address.
                let reason = error.split(" at 0x").next().unwrap_or(&error);
                *reasons.entry(reason.to_string()).or_default() += 1;
            }
        }
    }
    println!("{translated}/{total} translated");
    for (stage, count) in &stages {
        println!("  {stage}: {count}");
    }
    let mut sorted: Vec<(usize, String)> = reasons.into_iter().map(|(r, c)| (c, r)).collect();
    sorted.sort_by(|a, b| b.cmp(a));
    for (count, reason) in sorted {
        println!("{count:6} {reason}");
    }
}

/// The little-endian word at `offset` of the capture's control section.
fn control_word(code_path: &str, offset: usize) -> Option<u32> {
    let control = code_path.strip_suffix(".code")?.to_string() + ".control";
    let bytes = std::fs::read(control).ok()?;
    Some(u32::from_le_bytes(
        bytes.get(offset..offset + 4)?.try_into().ok()?,
    ))
}
