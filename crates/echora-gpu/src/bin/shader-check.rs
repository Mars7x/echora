//! Translate captured NVN programs (ECHORA_SHADER_DIR dumps) and compile each
//! stage on the Vulkan driver, grouping failures by reason.
//!   shader-check stage-*.code
//!   shader-check --compare stage-*.code
//! With --compare, each stage also goes through the Rust translator
//! (`echora-shader`): what it does not handle is grouped by reason, and
//! what it does is compiled on the driver and compared with the C++
//! recompiler's program (`shader::compare`).
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
    for shape in &gpu.cooperative_matrix {
        println!("cooperative matrix: {shape:?}");
    }
    // Captures carry no dispatch parameters; allow the most shared memory.
    let compute = Parameters {
        shared_memory_size: 0xc000,
        ..Default::default()
    };
    let (mut translated, mut compiled, mut total) = (0, 0, 0);
    let mut failures: BTreeMap<String, usize> = BTreeMap::new();
    let compare = std::env::args().any(|arg| arg == "--compare");
    let mut rust: BTreeMap<String, usize> = BTreeMap::new();
    for path in std::env::args().skip(1).filter(|arg| arg != "--compare") {
        total += 1;
        let Ok(bytes) = std::fs::read(&path) else {
            *failures.entry("unreadable".into()).or_default() += 1;
            continue;
        };
        // ECHORA_CHECK_MESH=1: a stage whose control section (the .control
        // file beside it) gives mesh output sizes at +0x60 is translated as
        // a mesh program with no task, with the registers the runtime
        // derives from that section (see `bound_mesh_program`).
        let mut parameters = compute;
        // A compute program's control section (the .control file beside
        // it): workgroup size, shared and local memory at +0x58..+0x68, as
        // the runtime reads them.
        let control = std::fs::read(std::path::Path::new(&path).with_extension("control"))
            .unwrap_or_default();
        let control_word = |offset: usize| {
            control
                .get(offset..)
                .and_then(<[u8]>::first_chunk::<4>)
                .map(|b| u32::from_le_bytes(*b))
        };
        let graphics = bytes
            .first_chunk::<4>()
            .is_some_and(|b| matches!(u32::from_le_bytes(*b) & 0x1f, 1 | 2));
        if !graphics
            && let (Some(x), Some(y), Some(z)) =
                (control_word(0x58), control_word(0x5c), control_word(0x60))
            && [x, y, z].iter().all(|size| (1..=1024).contains(size))
        {
            parameters.workgroup_size = [x, y, z];
            parameters.shared_memory_size = control_word(0x64).unwrap_or(0xc000);
            parameters.local_memory_size = control_word(0x68).unwrap_or(0);
        }
        if std::env::var_os("ECHORA_CHECK_MESH").is_some() {
            let control = std::fs::read(std::path::Path::new(&path).with_extension("control"))
                .unwrap_or_default();
            let word = |offset: usize| {
                control
                    .get(offset..)
                    .and_then(<[u8]>::first_chunk::<4>)
                    .map_or(0, |b| u32::from_le_bytes(*b))
            };
            let kind = bytes
                .first_chunk::<4>()
                .map_or(0, |b| (u32::from_le_bytes(*b) >> 10) & 0xf);
            let sizes = word(0x60);
            if matches!(kind, 1 | 3) && sizes != 0 {
                let (vertices, primitives) = (sizes & 0xffff, sizes >> 16);
                parameters.mesh = Some(shader::MeshParameters {
                    task: false,
                    registers: [
                        4 | (vertices & 0x1ff) << 4 | (primitives & 0x3ff) << 13,
                        (word(0x58) & 0x3f) << 20,
                        word(0x58) & !0x3f,
                    ],
                    primitive_header: None,
                });
            } else {
                continue;
            }
        }
        if compare {
            let mut compile = None;
            let outcome = match shader::translate_rust(&bytes, &parameters) {
                Err(unsupported) => format!("unsupported: {}", unsupported.reason),
                Ok(program) => match timed_check(&gpu, &program, &mut compile) {
                    Err(error) => format!("driver rejects: {error}"),
                    Ok(()) => match shader::translate(&bytes, parameters, &mut NoEnvironment) {
                        Err(_) => "translated (the plugin does not)".into(),
                        Ok(plugin) => match shader::compare(&plugin, &program) {
                            Some(difference) => format!("differs: {difference}"),
                            None => "alike".into(),
                        },
                    },
                },
            };
            // Split by stage kind (the SPH's shader type for graphics).
            let kind = if graphics {
                match bytes
                    .first_chunk::<4>()
                    .map_or(0, |b| (u32::from_le_bytes(*b) >> 10) & 0xf)
                {
                    1 => "vertex",
                    2 => "tess control",
                    3 => "tess eval",
                    4 => "geometry",
                    5 => "fragment",
                    _ => "graphics",
                }
            } else {
                "compute"
            };
            // ECHORA_CHECK_OUTCOMES: each stage's outcome, for finding
            // examples of one.
            if std::env::var_os("ECHORA_CHECK_OUTCOMES").is_some() {
                // The driver's compile time of the Rust translation, and
                // whether it reads the texture pool.
                let compile = compile.map_or(String::new(), |(ms, pool)| {
                    format!(" [compile {ms:.1} ms{}]", if pool { ", pool" } else { "" })
                });
                eprintln!("outcome {path} {kind} {outcome}{compile}");
            }
            *rust.entry(format!("{kind:8} {outcome}")).or_default() += 1;
        }
        let program = match shader::translate(&bytes, parameters, &mut NoEnvironment) {
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
            // path stage, then kind:binding:count:index:cbuf:offset:texture type
            // per binding.
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
                        d.texture_type,
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
    if compare {
        println!("rust translator:");
        let mut sorted: Vec<_> = rust.into_iter().collect();
        sorted.sort_by_key(|entry| std::cmp::Reverse(entry.1));
        for (outcome, count) in sorted {
            println!("{count:6} {outcome}");
        }
    }
}

/// `check_stage`, noting its compile time (ms) and whether the program
/// reads the texture pool.
fn timed_check(
    gpu: &Gpu,
    program: &shader::Translated,
    compile: &mut Option<(f64, bool)>,
) -> Result<(), String> {
    let started = std::time::Instant::now();
    let result = gpu.check_stage(program);
    *compile = Some((
        started.elapsed().as_secs_f64() * 1e3,
        program.texture_pool_bindings != 0,
    ));
    result
}
