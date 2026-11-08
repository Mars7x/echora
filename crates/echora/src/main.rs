//! The `echora` command-line front end.

mod host_paths;
mod prepare;

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use echora_content::{ContentInventory, ContentSource};
use echora_nup::{Node, NupArchive};

#[derive(Debug, Parser)]
#[command(
    name = "echora",
    version,
    about = "Echora: native recompilation runtime"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Describe a NUP file or extracted dump directory.
    Inspect {
        input: PathBuf,
        /// Print the full inventory as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Decode and hash-check every file in a NUP.
    Verify {
        input: PathBuf,
        /// Worker threads (default: available parallelism).
        #[arg(long)]
        threads: Option<usize>,
    },
    /// Validate the input and place the program's ExeFS in the content cache.
    Import {
        input: PathBuf,
        /// Cache root (default: the platform cache directory, or
        /// ECHORA_CACHE_DIR).
        #[arg(long)]
        cache: Option<PathBuf>,
    },
    /// Import the input, then load and verify every executable module.
    Load {
        input: PathBuf,
        /// Cache root (as for `import`).
        #[arg(long)]
        cache: Option<PathBuf>,
        /// Print the module report as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Load the modules and run ahead-of-time code discovery (no code
    /// generation), reporting coverage and unsupported instructions.
    Analyze {
        input: PathBuf,
        /// Cache root (as for `import`).
        #[arg(long)]
        cache: Option<PathBuf>,
        /// Modules whose prologues and relocation targets seed discovery
        /// (default: all). The process entry is always a root.
        #[arg(long, value_delimiter = ',')]
        seed: Vec<String>,
        /// Maximum instructions to discover.
        #[arg(long, default_value_t = 1_000_000)]
        limit: usize,
        /// Write the full report (including every unsupported site) as JSON.
        #[arg(long)]
        report: Option<PathBuf>,
    },
    /// Create the game process (modules, stack, TLS, entry registers)
    /// without executing guest code, and describe it.
    Boot {
        input: PathBuf,
        /// Cache root (as for `import`).
        #[arg(long)]
        cache: Option<PathBuf>,
    },
    /// Discover code and write the translation (a Cargo workspace of
    /// generated Rust) to the cache, rewriting only changed files.
    Lift {
        input: PathBuf,
        /// Cache root (as for `import`).
        #[arg(long)]
        cache: Option<PathBuf>,
        /// Maximum instructions to discover.
        #[arg(long, default_value_t = 20_000_000)]
        limit: usize,
        /// Guest bytes per generated crate, as a power of two.
        #[arg(long, default_value_t = 16)]
        region_bits: u32,
    },
    /// Set the game up to run: fetch the shader recompiler, lift the game
    /// and build its translation. Does nothing when this build of Echora
    /// already prepared it.
    Prepare {
        input: PathBuf,
        /// Cache root (as for `import`).
        #[arg(long)]
        cache: Option<PathBuf>,
        /// Build jobs (default: what the processors and memory allow).
        #[arg(long, short)]
        jobs: Option<usize>,
        /// Only report whether the game is prepared (exit status 0 if so).
        #[arg(long)]
        check: bool,
        /// Prepare again even when already prepared.
        #[arg(long)]
        force: bool,
        /// Stop when standard input closes (for a front end that keeps it
        /// open while it runs).
        #[arg(long)]
        stop_with_stdin: bool,
    },
    /// Run the built translation of the input's program (see `lift`).
    Run {
        input: PathBuf,
        /// Cache root (as for `import`).
        #[arg(long)]
        cache: Option<PathBuf>,
        /// Guest instruction budget.
        #[arg(long, default_value_t = 1_000_000_000)]
        budget: u64,
    },
    /// Extract a package subtree from a NUP, verifying every file.
    Extract {
        input: PathBuf,
        dest: PathBuf,
        /// Package index (see `inspect`).
        #[arg(long, default_value_t = 0)]
        package: usize,
        /// Path inside the package, for example `exefs` or `romfs/System`.
        #[arg(long)]
        subtree: Option<String>,
    },
}

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("ECHORA_LOG")
                .unwrap_or_else(|_| "info".into()),
        )
        .with_writer(std::io::stderr)
        .init();
    match run(Cli::parse()) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<ExitCode> {
    match cli.command {
        Command::Inspect { input, json } => {
            let inv = ContentSource::open(&input)?.inventory()?;
            if json {
                println!("{}", serde_json::to_string_pretty(&inv)?);
            } else {
                print_inventory(&inv);
            }
        }
        Command::Verify { input, threads } => {
            let archive = NupArchive::open(&input)?;
            let threads = threads
                .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
            let start = Instant::now();
            let report = archive.verify_contents(threads, &|_| {});
            let secs = start.elapsed().as_secs_f64();
            println!(
                "verified {} files ({} distinct extents, {:.2} GiB) in {secs:.1} s",
                report.files,
                report.extents,
                report.bytes as f64 / f64::from(1 << 30)
            );
            for p in &report.problems {
                println!("  file {}: {}", p.file.0, p.message);
            }
            if !report.ok() {
                return Ok(ExitCode::FAILURE);
            }
        }
        Command::Import { input, cache } => {
            let cache = match cache {
                Some(c) => c,
                None => host_paths::cache_dir()?,
            };
            let source = ContentSource::open(&input)?;
            let inv = source.inventory()?;
            print_inventory(&inv);
            let Some(pkg) = inv.program_package else {
                bail!("the input contains no package with an ExeFS");
            };
            let start = Instant::now();
            let dir = source.materialize_exefs(&inv, pkg, &cache)?;
            tracing::info!(elapsed = ?start.elapsed(), "ExeFS ready");
            println!("exefs: {}", dir.display());
        }
        Command::Load { input, cache, json } => {
            let cache = match cache {
                Some(c) => c,
                None => host_paths::cache_dir()?,
            };
            let source = ContentSource::open(&input)?;
            let inv = source.inventory()?;
            let Some(pkg) = inv.program_package else {
                bail!("the input contains no package with an ExeFS");
            };
            let dir = source.materialize_exefs(&inv, pkg, &cache)?;
            let start = Instant::now();
            let modules = echora_loader::load_exefs(&dir)?;
            let elapsed = start.elapsed();
            if json {
                println!("{}", serde_json::to_string_pretty(&modules)?);
            } else {
                for m in &modules {
                    println!(
                        "{:<8} build {} flags {:#04x} image {:#x} bss {:#x}",
                        m.name,
                        hex_prefix(&m.build_id),
                        m.flags,
                        m.image_size,
                        m.bss_size
                    );
                    for (label, s) in ["text", "rodata", "data"].iter().zip(m.segments()) {
                        println!(
                            "  {label:<6} {:>10} bytes at {:#09x}  {:?}  sha256 {}",
                            s.bytes.len(),
                            s.memory_offset,
                            s.compression,
                            if s.hash_verified {
                                "verified"
                            } else {
                                "not stored"
                            }
                        );
                    }
                    print_mod0(m);
                }
            }
            tracing::info!(?elapsed, modules = modules.len(), "ExeFS loaded");
        }
        Command::Analyze {
            input,
            cache,
            seed,
            limit,
            report,
        } => analyze(input, cache, &seed, limit, report.as_deref())?,
        Command::Boot { input, cache } => {
            let cache = match cache {
                Some(c) => c,
                None => host_paths::cache_dir()?,
            };
            let source = ContentSource::open(&input)?;
            let inv = source.inventory()?;
            let Some(pkg) = inv.program_package else {
                bail!("the input contains no package with an ExeFS");
            };
            let dir = source.materialize_exefs(&inv, pkg, &cache)?;
            let program = echora_runtime::bootstrap::load_program(&dir, 0)?;
            let n = &program.npdm;
            println!(
                "program {:016X}, main thread priority {} core {}, stack {:#x}, ACID layout {}",
                n.program_id,
                n.main_thread_priority,
                n.main_thread_core,
                n.main_thread_stack_size,
                n.acid_layout
            );
            println!("service access: {} entries", n.service_access.len());
            for m in &program.modules {
                println!(
                    "  {:<8} at {:#x} ({:#x} bytes)",
                    m.module.name, m.base, m.module.image_size
                );
            }
            let m = &program.machine;
            println!(
                "entry {:#x}, sp {:#x}, tls {:#x}, {} mappings",
                m.core.cpu.pc,
                m.core.cpu.sp,
                m.core.cpu.tpidrro_el0,
                m.memory.mappings().count()
            );
            println!("hooked host entries: {}", m.nvn.hook_addresses().count());
        }
        Command::Lift {
            input,
            cache,
            limit,
            region_bits,
        } => {
            lift(input, cache, limit, region_bits)?;
        }
        Command::Prepare {
            input,
            cache,
            jobs,
            check,
            force,
            stop_with_stdin,
        } => {
            if stop_with_stdin {
                prepare::stop_with_stdin();
            }
            let cache = match cache {
                Some(c) => c,
                None => host_paths::cache_dir()?,
            };
            let (_, translation) = translation_dir(&input, &cache)?;
            let ready = prepare::is_ready(&translation);
            if check {
                println!("{}", if ready { "prepared" } else { "not prepared" });
                return Ok(if ready {
                    ExitCode::SUCCESS
                } else {
                    ExitCode::FAILURE
                });
            }
            if !ready || force {
                prepare::step("fetch");
                prepare::fetch_shader_sources(&shader_sources(&cache))?;
                prepare::step("lift");
                let lifted = lift(input, Some(cache), 20_000_000, 16)?;
                prepare::step("build");
                prepare::build(&lifted, jobs.unwrap_or_else(prepare::default_jobs))?;
                prepare::mark_ready(&lifted)?;
            }
            println!("{}ready", prepare::PREFIX);
        }
        Command::Run {
            input,
            cache,
            budget,
        } => {
            let cache = match cache {
                Some(c) => c,
                None => host_paths::cache_dir()?,
            };
            let (exefs, translation) = translation_dir(&input, &cache)?;
            let runner = prepare::runner_path(&translation);
            if !runner.is_file() {
                bail!(
                    "no built translation at {}; run `echora prepare` (or `echora lift` and `cargo build --release -p echora-run` there)",
                    translation.display()
                );
            }
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs());
            let report = translation.join(format!("run-{stamp}.json"));
            // A NUP is read in place; a dump directory keeps its data in romfs/.
            let input = input.canonicalize()?;
            let romfs_path = if input.is_dir() {
                input.join("romfs")
            } else {
                input
            };
            println!("report: {}", report.display());
            let mut command = std::process::Command::new(&runner);
            // ECHORA_CAPTURE_EVERY=N: presented frames are saved as PNGs in
            // run-<stamp>-frames beside the report.
            if std::env::var_os("ECHORA_CAPTURE_EVERY").is_some()
                && std::env::var_os("ECHORA_CAPTURE_DIR").is_none()
            {
                let frames = translation.join(format!("run-{stamp}-frames"));
                println!("frames: {}", frames.display());
                command.env("ECHORA_CAPTURE_DIR", frames);
            }
            command
                .arg(&exefs)
                .arg(budget.to_string())
                .arg(&report)
                .env("ECHORA_ROMFS", romfs_path)
                .env("ECHORA_SETTINGS", host_paths::settings_file()?);
            // Draws run on the host GPU (Vulkan) unless the environment
            // chooses otherwise; every configurable backend is a GPU one.
            if std::env::var_os("ECHORA_GPU").is_none() {
                command.env("ECHORA_GPU", "1");
            }
            // The shader translator rejects tensor matrix multiplies unless
            // asked to expand them; the game's movie and upscaling passes
            // use them, so runs expand them unless the environment chooses.
            if std::env::var_os("EDEN_NX2_TENSOR").is_none() {
                command.env("EDEN_NX2_TENSOR", "1");
            }
            // Compiled shaders are kept beside the translation, for the
            // same program's next run.
            if std::env::var_os("ECHORA_SHADER_CACHE").is_none() {
                command.env("ECHORA_SHADER_CACHE", translation.join("shader-cache"));
            }
            // Saves go under the per-user data directory unless the
            // environment chooses a save or work directory.
            if std::env::var_os("ECHORA_SAVE_DIR").is_none()
                && std::env::var_os("ECHORA_WORK").is_none()
            {
                command.env("ECHORA_SAVE_DIR", host_paths::data_dir()?.join("save"));
            }
            let status = command
                .status()
                .with_context(|| format!("start {}", runner.display()))?;
            if !status.success() {
                return Ok(ExitCode::from(2));
            }
        }
        Command::Extract {
            input,
            dest,
            package,
            subtree,
        } => {
            let archive = NupArchive::open(&input)?;
            let pkg = archive
                .packages()
                .get(package)
                .with_context(|| format!("no package {package}"))?;
            let dir = match subtree.as_deref() {
                None | Some("") => pkg.root,
                Some(path) => match archive.resolve(pkg.root, path) {
                    Some(Node::Dir(d)) => d,
                    Some(Node::File(_)) => bail!("{path} is a file, not a directory"),
                    None => bail!("{path} not found in package {package}"),
                },
            };
            let stats = archive.extract_dir(dir, &dest)?;
            println!(
                "extracted {} files in {} directories ({} bytes) to {}",
                stats.files,
                stats.dirs,
                stats.bytes,
                dest.display()
            );
        }
    }
    Ok(ExitCode::SUCCESS)
}

struct Discovered {
    cache: PathBuf,
    exefs_key: String,
    placed: Vec<echora_loader::process::PlacedModule>,
    translation: echora_lift::codegen::Translation,
    elapsed: std::time::Duration,
}

/// Loads the program's modules, places them, seeds discovery and runs it.
fn discover_game(
    input: PathBuf,
    cache: Option<PathBuf>,
    seed: &[String],
    limit: usize,
) -> Result<Discovered> {
    use echora_lift::codegen::{CodeRegion, discover_with_tables};
    use echora_loader::process;

    let cache = match cache {
        Some(c) => c,
        None => host_paths::cache_dir()?,
    };
    let source = ContentSource::open(&input)?;
    let inv = source.inventory()?;
    let Some(pkg) = inv.program_package else {
        bail!("the input contains no package with an ExeFS");
    };
    let dir = source.materialize_exefs(&inv, pkg, &cache)?;
    let exefs_key = inv.packages[pkg]
        .exefs_key
        .clone()
        .context("program package has no ExeFS key")?;
    let placed = process::place_modules(echora_loader::load_exefs(&dir)?);

    let regions: Vec<CodeRegion> = placed
        .iter()
        .map(|p| CodeRegion {
            name: p.module.name.clone(),
            base: p.text_range().start,
            bytes: p.module.text.bytes.clone(),
        })
        .collect();
    let rodata: Vec<CodeRegion> = placed
        .iter()
        .map(|p| CodeRegion {
            name: format!("{}.rodata", p.module.name),
            base: p.rodata_base(),
            bytes: p.module.rodata.bytes.clone(),
        })
        .collect();

    // The process starts in rtld's text; static constructors and exported
    // functions are primary roots, prologues and relocation targets are
    // deferred candidates.
    let mut roots = vec![placed[0].text_range().start];
    let mut deferred = Vec::new();
    for p in &placed {
        if !seed.is_empty() && !seed.contains(&p.module.name) {
            continue;
        }
        let ctors = process::init_array_targets(p);
        let exports = process::exported_function_targets(p);
        let lazy = process::lazy_binding_targets(p);
        let relocs = process::relative_text_targets(p);
        let addresses = process::code_address_targets(p);
        let prologues = process::prologue_targets(p);
        println!(
            "{:<8} seeds: {} constructors, {} exported functions, {} lazy-binding stubs, {} relocation targets, {} code addresses, {} prologues",
            p.module.name,
            ctors.len(),
            exports.len(),
            lazy.len(),
            relocs.len(),
            addresses.len(),
            prologues.len()
        );
        roots.extend(ctors);
        // Exported functions are certain entry points, reached from other
        // modules by symbol rather than by relocation.
        roots.extend(exports);
        // Unresolved imports jump through their GOT slot's lazy value.
        roots.extend(lazy);
        deferred.extend(relocs);
        deferred.extend(addresses);
        deferred.extend(prologues);
    }
    roots.sort_unstable();
    roots.dedup();
    deferred.sort_unstable();
    deferred.dedup();

    // Hooked exports (the NVN bootstrap) and replaced library functions run
    // as host code, which only happens where no translated block exists;
    // exported-function roots and calls would otherwise translate them.
    let call_sites = echora_runtime::bootstrap::hle_call_sites(&placed);
    roots.extend(call_sites.iter().map(|site| site + 4));
    roots.sort_unstable();
    roots.dedup();
    let host_entries = echora_runtime::bootstrap::hooked_export_addresses(&placed)?
        .into_iter()
        .chain(echora_runtime::bootstrap::hle_function_addresses(&placed))
        .chain(call_sites)
        .collect();
    let start = Instant::now();
    let translation =
        discover_with_tables(&regions, &rodata, &roots, &deferred, &host_entries, limit)?;
    Ok(Discovered {
        cache,
        exefs_key,
        placed,
        translation,
        elapsed: start.elapsed(),
    })
}

fn hex_bytes(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// `main.rs` of a translation's runner crate. `// EXPECTED` is replaced by
/// the (module, text SHA-256) pairs the translation was made from.
const RUNNER_MAIN: &str = r#"//! Runs a translated program. Generated by `echora lift`.

use std::path::PathBuf;

use anyhow::{Context, bail};
use echora_runtime::runtime::Trap;

/// Modules and text-segment hashes this translation was made from.
const EXPECTED: &[(&str, &str)] = &[
    // EXPECTED
];

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args_os().skip(1);
    let exefs = PathBuf::from(args.next().context("usage: echora-run EXEFS [MAX_STEPS] [REPORT.json]")?);
    let budget: u64 = match args.next() {
        Some(value) => value.to_string_lossy().parse().context("MAX_STEPS")?,
        None => 1_000_000_000,
    };
    let report_path = args.next().map(PathBuf::from);
    let mut program = echora_runtime::bootstrap::load_program(&exefs, budget)?;
    let loaded: Vec<(String, String)> = program
        .modules
        .iter()
        .map(|p| (p.module.name.clone(), hex(&p.module.text.sha256)))
        .collect();
    let matches = loaded.len() == EXPECTED.len()
        && loaded.iter().zip(EXPECTED).all(|((n, h), (en, eh))| n == en && h == eh);
    if !matches {
        bail!("the ExeFS does not match this translation's build: {loaded:?}");
    }
    echora_translation::install();
    let start = std::time::Instant::now();
    let stop = match program.machine.run_generated(echora_translation::dispatch) {
        Ok(()) => Trap::InvalidState { message: "the runner returned without a guest stop".into() },
        Err(stop) => stop,
    };
    let elapsed = start.elapsed();
    let report = program.run_report(&stop);
    eprintln!("stop: {stop:?}");
    eprintln!("pc: {:#x} {}", program.machine.core.cpu.pc, report["pc_location"]);
    eprintln!("steps: {} in {:.2} s", program.machine.core.steps, elapsed.as_secs_f64());
    if let Some(path) = report_path {
        std::fs::write(&path, serde_json::to_vec_pretty(&report)?)
            .with_context(|| format!("write {}", path.display()))?;
    }
    if !matches!(stop, Trap::Exit { code: 0 }) {
        std::process::exit(2);
    }
    Ok(())
}
"#;

/// Writes `contents` unless the file already holds exactly that; returns
/// whether it wrote.
fn write_if_changed(path: &std::path::Path, contents: &str) -> std::io::Result<bool> {
    if std::fs::read(path).is_ok_and(|old| old == contents.as_bytes()) {
        return Ok(false);
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, contents)?;
    Ok(true)
}

/// The input program's ExeFS in `cache`, and where its translation lives.
fn translation_dir(input: &std::path::Path, cache: &std::path::Path) -> Result<(PathBuf, PathBuf)> {
    let source = ContentSource::open(input)?;
    let inv = source.inventory()?;
    let Some(pkg) = inv.program_package else {
        bail!("the input contains no package with an ExeFS");
    };
    let exefs = source.materialize_exefs(&inv, pkg, cache)?;
    let key = inv.packages[pkg]
        .exefs_key
        .as_ref()
        .context("program package has no ExeFS key")?;
    Ok((exefs, cache.join("translations").join(key)))
}

/// The shader recompiler sources: `ECHORA_SHADER_SRC`, else `shader-src`
/// in the cache, where `scripts/fetch-shader-sources.sh` puts them.
fn shader_sources(cache: &std::path::Path) -> PathBuf {
    std::env::var_os("ECHORA_SHADER_SRC").map_or_else(|| cache.join("shader-src"), PathBuf::from)
}

fn lift(input: PathBuf, cache: Option<PathBuf>, limit: usize, region_bits: u32) -> Result<PathBuf> {
    use echora_lift::codegen::{EmitOptions, emit_rust_split_to_with};

    if !(10..=24).contains(&region_bits) {
        bail!("region-bits must be between 10 and 24");
    }
    let d = discover_game(input, cache, &[], limit)?;
    let report = d.translation.report();
    println!(
        "discovered {} instructions in {} blocks ({:.1} s){}",
        report.translated_instructions,
        report.block_count,
        d.elapsed.as_secs_f64(),
        if report.truncated {
            ", truncated at the limit"
        } else {
            ""
        }
    );
    let out = d.cache.join("translations").join(&d.exefs_key);
    let guest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../echora-guest")
        .canonicalize()
        .context("locate the echora-guest crate")?;
    let guest = guest.to_string_lossy().into_owned();
    // The SIMD/FP operations generated code reaches through echora-guest's
    // declarations; only the shared library links them.
    let simd = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../echora-simd")
        .canonicalize()
        .context("locate the echora-simd crate")?;
    let simd = simd.to_string_lossy().into_owned();

    // ECHORA_LIFT_OPTIONS: code generation options, comma-separated
    // (structured, inline-vector, direct-calls); all off by default.
    let mut options = EmitOptions::default();
    for option in std::env::var("ECHORA_LIFT_OPTIONS")
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|option| !option.is_empty())
    {
        match option {
            "structured" => options.structured = true,
            "inline-vector" => options.inline_vector = true,
            "direct-calls" => options.direct_calls = true,
            other => bail!("unknown ECHORA_LIFT_OPTIONS entry {other:?}"),
        }
    }

    let start = Instant::now();
    let mut regions = Vec::new();
    let (mut written, mut source_bytes) = (0usize, 0u64);
    let root = emit_rust_split_to_with(
        &d.translation,
        region_bits,
        options,
        &mut |region, source| {
            let dir = out.join("regions").join(format!("{region:x}"));
            let manifest = format!(
                "[package]\nname = \"gen-r-{region:x}\"\nversion = \"0.0.0\"\nedition = \"2024\"\npublish = false\n\n\
             [dependencies]\nechora-guest = {{ path = {guest:?} }}\n"
            );
            write_if_changed(&dir.join("Cargo.toml"), &manifest)?;
            source_bytes += source.len() as u64;
            written += usize::from(write_if_changed(&dir.join("src/lib.rs"), &source)?);
            regions.push(region);
            Ok::<(), std::io::Error>(())
        },
    )?;
    let emit_time = start.elapsed();

    // Remove regions that no longer exist.
    if let Ok(entries) = std::fs::read_dir(out.join("regions")) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if u64::from_str_radix(&name, 16).map_or(true, |r| !regions.contains(&r)) {
                std::fs::remove_dir_all(entry.path())?;
            }
        }
    }

    // Regions are linked into a few shared libraries ("bundles"), each well
    // under the 2 GiB that x86-64 code can span with 32-bit relative
    // references (and that a Windows image may occupy). The dispatcher
    // reaches each region through its bundle's re-export.
    let mut dispatch = format!(
        "[package]\nname = \"echora-translation\"\nversion = \"0.0.0\"\nedition = \"2024\"\npublish = false\n\n\
         [dependencies]\nechora-guest = {{ path = {guest:?} }}\n"
    );
    // One shared library holds the guest crate, so the bundles and the
    // runner share a single copy of its state (the window table above all).
    // It lives here rather than as a dylib of echora-guest itself, which
    // would collide with the guest's ordinary build in Echora's workspace.
    write_if_changed(
        &out.join("guest/Cargo.toml"),
        &format!(
            "[package]\nname = \"echora-guest-shared\"\nversion = \"0.0.0\"\nedition = \"2024\"\npublish = false\n\n\
             [lib]\ncrate-type = [\"dylib\"]\n\n\
             [dependencies]\nechora-guest = {{ path = {guest:?} }}\nechora-simd = {{ path = {simd:?} }}\n"
        ),
    )?;
    write_if_changed(
        &out.join("guest/src/lib.rs"),
        "// Generated. Keep this file local.\npub use echora_guest::*;\nextern crate echora_simd;\n",
    )?;
    let mut imports = String::new();
    let bundles: Vec<&[u64]> = regions.chunks(BUNDLE_REGIONS).collect();
    for (index, members) in bundles.iter().enumerate() {
        let mut manifest = format!(
            "[package]\nname = \"echora-bundle-{index}\"\nversion = \"0.0.0\"\nedition = \"2024\"\npublish = false\n\n\
             [lib]\ncrate-type = [\"dylib\"]\n\n\
             [dependencies]\nechora-guest = {{ path = {guest:?} }}\n\
             echora-guest-shared = {{ path = \"../../guest\" }}\n"
        );
        // Linking the shared guest library keeps the guest crate out of
        // each bundle.
        let mut exports = String::from(
            "// Generated. Keep this file local.\nextern crate echora_guest_shared;\n",
        );
        for region in *members {
            manifest.push_str(&format!(
                "gen-r-{region:x} = {{ path = \"../../regions/{region:x}\" }}\n"
            ));
            exports.push_str(&format!("pub use gen_r_{region:x};\n"));
            imports.push_str(&format!("use echora_bundle_{index}::gen_r_{region:x};\n"));
        }
        let dir = out.join("bundles").join(index.to_string());
        write_if_changed(&dir.join("Cargo.toml"), &manifest)?;
        write_if_changed(&dir.join("src/lib.rs"), &exports)?;
        dispatch.push_str(&format!(
            "echora-bundle-{index} = {{ path = \"../bundles/{index}\" }}\n"
        ));
    }
    if let Ok(entries) = std::fs::read_dir(out.join("bundles")) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.parse::<usize>().map_or(true, |i| i >= bundles.len()) {
                std::fs::remove_dir_all(entry.path())?;
            }
        }
    }
    write_if_changed(&out.join("dispatch/Cargo.toml"), &dispatch)?;
    write_if_changed(&out.join("dispatch/src/lib.rs"), &(imports + &root))?;

    // The runner: boots the process, checks it is the build this
    // translation was made from, and runs it.
    let runtime = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../echora-runtime")
        .canonicalize()
        .context("locate the echora-runtime crate")?;
    let expected: String = d
        .placed
        .iter()
        .map(|p| {
            format!(
                "    ({:?}, {:?}),\n",
                p.module.name,
                hex_bytes(&p.module.text.sha256)
            )
        })
        .collect();
    write_if_changed(
        &out.join("runner/Cargo.toml"),
        &format!(
            "[package]\nname = \"echora-run\"\nversion = \"0.0.0\"\nedition = \"2024\"\npublish = false\n\n\
             [dependencies]\nanyhow = \"1\"\nserde_json = \"1\"\n\
             echora-runtime = {{ path = {:?} }}\n\
             echora-translation = {{ path = \"../dispatch\" }}\n",
            runtime.to_string_lossy()
        ),
    )?;
    write_if_changed(
        &out.join("runner/src/main.rs"),
        &RUNNER_MAIN.replace("    // EXPECTED\n", &expected),
    )?;
    // Generated code is compiled at a moderate level; the guest crate, the
    // runtime it calls into and the runtime's AV1 decoder (several times
    // slower at level 1) get full optimization.
    // `rpath` lets the runner find the bundles and the shared guest crate.
    // One codegen unit per crate: with a thousand region crates sharing the
    // cores, total CPU time decides the build, and splitting a region costs
    // CPU without helping it (its windows reach each other only through
    // function pointers). The three fully optimized crates keep the default.
    write_if_changed(
        &out.join("Cargo.toml"),
        "[workspace]\nresolver = \"3\"\nmembers = [\"guest\", \"dispatch\", \"runner\", \"bundles/*\", \"regions/*\"]\n\n\
         [profile.release]\nopt-level = 1\ncodegen-units = 1\nrpath = true\n\n\
         [profile.release.package.echora-guest]\nopt-level = 3\ncodegen-units = 16\n\n\
         [profile.release.package.echora-runtime]\nopt-level = 3\ncodegen-units = 16\n\n\
         [profile.release.package.rav1d]\nopt-level = 3\ncodegen-units = 16\n",
    )?;
    // Linking the bundles as shared libraries needs the standard library
    // linked dynamically too; on Unix the runner records where it lives.
    let mut flags = vec!["-C".to_owned(), "prefer-dynamic".to_owned()];
    if cfg!(unix) {
        let libdir =
            std::process::Command::new(std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into()))
                .args(["--print", "target-libdir"])
                .output()
                .context("run rustc --print target-libdir")?;
        let libdir = String::from_utf8_lossy(&libdir.stdout).trim().to_owned();
        flags.push("-C".into());
        flags.push(format!("link-arg=-Wl,-rpath,{libdir}"));
    }
    // The shader recompiler is built into the runtime when its fetched
    // sources are present (scripts/fetch-shader-sources.sh): the workspace
    // passes them on, so a plain `cargo build` of the runner can draw.
    let shader_sources = Some(shader_sources(&d.cache)).filter(|dir| dir.is_dir());
    let mut config = format!("[build]\nrustflags = {flags:?}\n");
    if let Some(sources) = shader_sources {
        config.push_str(&format!(
            "\n[env]\nECHORA_SHADER_SRC = {:?}\n",
            sources.to_string_lossy()
        ));
    }
    write_if_changed(&out.join(".cargo/config.toml"), &config)?;
    std::fs::write(
        out.join("translation-report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!(
        "wrote {} region crates ({:.1} MiB of Rust, {written} changed) in {:.1} s",
        regions.len(),
        source_bytes as f64 / f64::from(1 << 20),
        emit_time.as_secs_f64()
    );
    println!("translation: {}", out.display());
    Ok(out)
}

/// Regions per shared library: about 500 MB of code each at the current
/// code size, far from the 2 GiB limit.
const BUNDLE_REGIONS: usize = 192;

fn analyze(
    input: PathBuf,
    cache: Option<PathBuf>,
    seed: &[String],
    limit: usize,
    report_path: Option<&std::path::Path>,
) -> Result<()> {
    let Discovered {
        placed,
        translation,
        elapsed,
        ..
    } = discover_game(input, cache, seed, limit)?;
    let report = translation.report();
    println!(
        "discovered {} instructions in {} blocks ({:.1} s){}",
        report.translated_instructions,
        report.block_count,
        elapsed.as_secs_f64(),
        if report.truncated {
            ", truncated at the limit"
        } else {
            ""
        }
    );
    println!(
        "jump tables {} ({} targets), unresolved indirect targets {}",
        report.jump_tables,
        report.jump_table_targets,
        report.unresolved_targets.len()
    );
    for p in &placed {
        let range = p.text_range();
        let covered = translation.instructions.range(range.clone()).count();
        let total = (range.end - range.start) / 4;
        println!(
            "  {:<8} {covered:>9} of {total:>9} text words ({:.1}%)",
            p.module.name,
            covered as f64 * 100.0 / total as f64
        );
    }
    let mut by_mnemonic = std::collections::BTreeMap::<String, usize>::new();
    for site in &report.unsupported {
        let m = site
            .disassembly
            .split_whitespace()
            .next()
            .unwrap_or("?")
            .to_owned();
        *by_mnemonic.entry(m).or_default() += 1;
    }
    let mut ranked: Vec<_> = by_mnemonic.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    println!(
        "undefined (UDF) sites: {} (translated as undefined-instruction stops)",
        report.undefined_instructions
    );
    println!(
        "unsupported: {} sites, {} mnemonics",
        report.unsupported_instructions,
        ranked.len()
    );
    for (m, n) in ranked.iter().take(25) {
        println!("  {n:>8}  {m}");
    }
    if let Some(path) = report_path {
        std::fs::write(path, serde_json::to_vec_pretty(&report)?)
            .with_context(|| format!("write {}", path.display()))?;
    }
    Ok(())
}

fn print_mod0(m: &echora_loader::NsoModule) {
    let Some(info) = &m.mod0 else {
        println!("  MOD0   absent");
        return;
    };
    println!(
        "  MOD0   at {:#x}, bss {:#x}..{:#x}, unwind {:#x} bytes",
        info.memory_offset,
        info.bss_start,
        info.bss_end,
        info.unwind_end - info.unwind_start
    );
    let Some(d) = &info.dynamic else {
        println!("  dynamic absent");
        return;
    };
    print!("  dynamic {} entries", d.entries.len());
    if let Some(name) = &d.soname {
        print!(", soname {name}");
    }
    if !d.needed_libraries.is_empty() {
        print!(", needs {}", d.needed_libraries.join(" "));
    }
    println!();
    if let Some(t) = &d.symbol_table {
        let hashes = match (t.has_sysv_hash, t.has_gnu_hash) {
            (true, true) => "SysV+GNU",
            (true, false) => "SysV",
            (false, true) => "GNU only",
            (false, false) => "none",
        };
        println!(
            "  symbols {} ({} defined, {} undefined), hash {hashes}",
            t.count.map_or("?".into(), |c| c.to_string()),
            t.defined_count.unwrap_or(0),
            t.undefined_count.unwrap_or(0)
        );
    }
    for r in &d.relocation_tables {
        let types: Vec<_> = r.types.iter().map(|(t, n)| format!("{t}x{n}")).collect();
        println!(
            "  {:<12} {} entries  types {}",
            r.kind,
            r.count,
            types.join(" ")
        );
    }
    if let Some(a) = &d.init_array {
        println!("  init_array {} entries", a.count);
    }
}

fn hex_prefix(id: &[u8; 32]) -> String {
    id[..20].iter().map(|b| format!("{b:02x}")).collect()
}

fn print_inventory(inv: &ContentInventory) {
    println!("source:  {} ({})", inv.path.display(), inv.source);
    for p in &inv.packages {
        let v = p.version;
        println!(
            "package {}: {:?} {} version {v} ({}.{}.{}){}{}",
            p.index,
            p.kind,
            p.title_id.as_deref().unwrap_or("unknown-title"),
            v >> 16,
            (v >> 8) & 0xff,
            v & 0xff,
            p.display_version
                .as_deref()
                .map(|d| format!(", display {d}"))
                .unwrap_or_default(),
            if p.standalone_update {
                ", standalone update"
            } else {
                ""
            },
        );
        if let Some(name) = &p.title_name {
            println!("  title:    {name}");
        }
        if let Some(id) = p.main_build_id() {
            println!("  build id: {id}");
        }
        for m in &p.exefs {
            let detail = match (&m.nso, &m.npdm) {
                (Some(n), _) => format!(
                    "NSO flags {:#x}, text {:#x} rodata {:#x} data {:#x} bss {:#x}",
                    n.flags,
                    n.text.memory_size,
                    n.rodata.memory_size,
                    n.data.memory_size,
                    n.bss_size
                ),
                (_, Some(n)) => format!(
                    "NPDM {} program {}",
                    n.name,
                    n.program_id.as_deref().unwrap_or("?")
                ),
                _ => String::new(),
            };
            println!(
                "  exefs/{:<10} {:>10}  {}  {detail}",
                m.name,
                m.size,
                &m.sha256[..16]
            );
        }
        if let Some(r) = &p.romfs {
            println!(
                "  romfs:    {} files, {:.2} GiB",
                r.files,
                r.bytes as f64 / f64::from(1 << 30)
            );
        }
    }
    if let Some(p) = inv.program_package {
        println!("program package: {p}");
    }
    for n in &inv.notes {
        println!("note: {n}");
    }
}
