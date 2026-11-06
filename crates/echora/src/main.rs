//! The `echora` command-line front end.

mod host_paths;

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
                }
            }
            tracing::info!(?elapsed, modules = modules.len(), "ExeFS loaded");
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
