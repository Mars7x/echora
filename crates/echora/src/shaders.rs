//! `echora shaders`: an inventory of every GPU program the game ships,
//! read straight from a NUP without running anything.
//!
//! The game's programs live in BNSH shader containers (magic `BNSH`, block
//! `grsc`), embedded in its romfs files (material shader archives, `.bfsha`;
//! effect definitions, `.vsetb.byml`; and a few packs) and possibly in its
//! executable modules. Most romfs files are zstd frames, some compressed
//! against a dictionary from `Pack/ZsDic.pack.zs`. Every file is
//! decompressed in memory and searched for containers; each container's
//! variations name a program with up to eight stage slots, each a control
//! section and code.
//!
//! Programs are deduplicated by the key Echora's runtime gives a stage when
//! it dumps one (`ECHORA_SHADER_DIR`): FNV-1a over the code up to the end
//! `sass::program_length` finds. So the unique keys are comparable with the
//! stages a run has seen.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result, anyhow, bail};
use echora_nup::{Node, NupArchive};

/// Stage slots of a BNSH program, in order. Slots 6 and 7 are inferred:
/// the game fills slot 7 alongside fragment programs, with programs whose
/// header marks them as pre-rasterization stages, as NVN2 mesh programs
/// are; slot 6 (task) is never filled in this build.
const STAGES: [&str; 8] = [
    "vertex",
    "tess control",
    "tess eval",
    "geometry",
    "fragment",
    "compute",
    "task",
    "mesh",
];

/// Files that are media streams, never shader containers.
const MEDIA: [&str; 2] = [".webm", ".bwav"];

/// A stage's code and control section, kept for `--dump`.
type Dumped = (Vec<u8>, Vec<u8>);

#[derive(Default)]
struct Totals {
    /// Files holding at least one container, by source kind.
    files: BTreeMap<String, u64>,
    containers: BTreeMap<String, u64>,
    programs: BTreeMap<String, u64>,
    /// Stage entries (one per filled slot of each program), by stage.
    stages: BTreeMap<&'static str, u64>,
    /// Unique stage keys, with the stage they first appeared as and their
    /// code and control section (kept only when dumping).
    unique: HashMap<u64, (&'static str, Option<Dumped>)>,
    /// Files that could not be decompressed or read.
    unreadable: Vec<String>,
    scanned: u64,
}

/// The source kind a file name stands for: its type extension (`bfsha`,
/// `vsetb.byml`, `sarc`, ...) without the `.zs` compression suffix.
fn source_kind(name: &str) -> String {
    let base = name.rsplit('/').next().unwrap_or(name);
    let base = base.strip_suffix(".zs").unwrap_or(base);
    let parts: Vec<&str> = base.split('.').collect();
    match parts.as_slice() {
        [.., kind, "byml"] => format!("{kind}.byml"),
        [_, .., kind] => (*kind).to_owned(),
        _ => String::new(),
    }
}

fn u16_at(bytes: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_le_bytes(*bytes.get(at..)?.first_chunk()?))
}

fn u32_at(bytes: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(*bytes.get(at..)?.first_chunk()?))
}

fn u64_at(bytes: &[u8], at: usize) -> Option<u64> {
    Some(u64::from_le_bytes(*bytes.get(at..)?.first_chunk()?))
}

/// The runtime's stage key (see the module comment).
fn stage_key(code: &[u8]) -> u64 {
    let end = echora_runtime::sass::program_length(code).map_or(code.len(), |count| count * 16);
    code[..end.min(code.len())]
        .iter()
        .fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
        })
}

/// One program stage: (slot, control section, code).
type StageCode<'a> = (usize, &'a [u8], &'a [u8]);

/// The programs of the BNSH container at the start of `bnsh` (the bytes
/// from its magic to the end of the enclosing file), or None when the bytes
/// only look like one. Offsets inside a container are relative to its
/// start.
fn container_programs(bnsh: &[u8]) -> Option<Vec<Vec<StageCode<'_>>>> {
    // Header: magic, version, byte-order mark 0xfeff at +0x0c, first block
    // offset at +0x16, file size at +0x1c.
    if u16_at(bnsh, 0x0c)? != 0xfeff {
        return None;
    }
    let size = u32_at(bnsh, 0x1c)? as usize;
    let bnsh = bnsh.get(..size)?;
    let grsc = u16_at(bnsh, 0x16)? as usize;
    if bnsh.get(grsc..grsc + 4)? != b"grsc" {
        return None;
    }
    // grsc: variation count at +0x1c, variation array offset at +0x20;
    // each variation 0x40 bytes with its binary program offset at +0x10.
    let count = u32_at(bnsh, grsc + 0x1c)? as usize;
    let variations = u64_at(bnsh, grsc + 0x20)? as usize;
    let mut programs = Vec::with_capacity(count);
    for index in 0..count {
        let program = u64_at(bnsh, variations + index * 0x40 + 0x10)? as usize;
        if program == 0 {
            continue;
        }
        // Program: a word, then one stage code offset per slot. Stage code:
        // control offset at +0x08, code offset at +0x10, code size at
        // +0x18, control size at +0x1c.
        let mut stages = Vec::new();
        for (slot, _) in STAGES.iter().enumerate() {
            let entry = u64_at(bnsh, program + 8 + slot * 8)? as usize;
            if entry == 0 {
                continue;
            }
            let control = u64_at(bnsh, entry + 0x08)? as usize;
            let code = u64_at(bnsh, entry + 0x10)? as usize;
            let code_size = u32_at(bnsh, entry + 0x18)? as usize;
            let control_size = u32_at(bnsh, entry + 0x1c)? as usize;
            stages.push((
                slot,
                bnsh.get(control..control + control_size)?,
                bnsh.get(code..code + code_size)?,
            ));
        }
        programs.push(stages);
    }
    Some(programs)
}

/// Adds every container in `bytes` (one decompressed file) to `totals`.
fn scan(bytes: &[u8], source: &str, dump: bool, totals: &Mutex<Totals>) {
    let mut found = Vec::new();
    let mut at = 0;
    while let Some(offset) = bytes[at..].windows(8).position(|w| w == b"BNSH\0\0\0\0") {
        let start = at + offset;
        if let Some(programs) = container_programs(&bytes[start..]) {
            found.push(programs);
        }
        at = start + 8;
    }
    if found.is_empty() {
        return;
    }
    // Keys outside the lock: hashing is most of the work.
    let mut keyed = Vec::new();
    for programs in &found {
        for stages in programs {
            for &(slot, control, code) in stages {
                keyed.push((slot, stage_key(code), control, code));
            }
        }
    }
    let mut totals = totals.lock().unwrap_or_else(|poison| poison.into_inner());
    *totals.files.entry(source.to_owned()).or_default() += 1;
    *totals.containers.entry(source.to_owned()).or_default() += found.len() as u64;
    *totals.programs.entry(source.to_owned()).or_default() += found
        .iter()
        .map(|programs| programs.len() as u64)
        .sum::<u64>();
    for (slot, key, control, code) in keyed {
        *totals.stages.entry(STAGES[slot]).or_default() += 1;
        totals.unique.entry(key).or_insert_with(|| {
            (
                STAGES[slot],
                dump.then(|| (code.to_vec(), control.to_vec())),
            )
        });
    }
}

/// The zstd dictionaries in `Pack/ZsDic.pack.zs` (a SARC), by file name.
fn dictionaries(
    archive: &NupArchive,
    romfs: echora_nup::DirId,
) -> Result<HashMap<String, Vec<u8>>> {
    let Some(Node::File(file)) = archive.resolve(romfs, "Pack/ZsDic.pack.zs") else {
        return Ok(HashMap::new());
    };
    let packed = archive.read_file(file, u64::MAX)?;
    let sarc = zstd::stream::decode_all(&packed[..]).context("Pack/ZsDic.pack.zs")?;
    // SARC: header size at +4, data offset at +0x0c; SFAT after the header
    // with its node count at +6 and 16-byte nodes (name offset / 4 in the
    // attribute's low half, data start and end) from +0x0c; SFNT names
    // after the nodes, past its 8-byte header.
    let bad = || anyhow!("Pack/ZsDic.pack.zs: not a SARC");
    if sarc.get(..4) != Some(b"SARC") {
        return Err(bad());
    }
    let sfat = u16_at(&sarc, 4).ok_or_else(bad)? as usize;
    let data = u32_at(&sarc, 0x0c).ok_or_else(bad)? as usize;
    let nodes = u16_at(&sarc, sfat + 6).ok_or_else(bad)? as usize;
    let names = sfat + 0x0c + nodes * 16 + 8;
    let mut out = HashMap::new();
    for index in 0..nodes {
        let node = sfat + 0x0c + index * 16;
        let attributes = u32_at(&sarc, node + 4).ok_or_else(bad)?;
        let start = u32_at(&sarc, node + 8).ok_or_else(bad)? as usize;
        let end = u32_at(&sarc, node + 12).ok_or_else(bad)? as usize;
        let name_at = names + (attributes & 0xffff) as usize * 4;
        let name = sarc
            .get(name_at..)
            .and_then(|rest| rest.split(|&byte| byte == 0).next())
            .ok_or_else(bad)?;
        let bytes = sarc.get(data + start..data + end).ok_or_else(bad)?;
        out.insert(String::from_utf8_lossy(name).into_owned(), bytes.to_vec());
    }
    Ok(out)
}

/// A romfs file's bytes, decompressed when it is a zstd frame: with the
/// dictionary its type uses (`pack`, `bcett.byml`, `bkprb`, else `zs`), or
/// none.
fn decompress(
    name: &str,
    raw: Vec<u8>,
    dictionaries: &HashMap<String, zstd::dict::DecoderDictionary<'static>>,
) -> Option<Vec<u8>> {
    if !name.ends_with(".zs") {
        return Some(raw);
    }
    let dictionary = ["pack", "bcett.byml", "bkprb"]
        .into_iter()
        .find(|kind| name.ends_with(&format!(".{kind}.zs")))
        .map_or("zs.zsdic".to_owned(), |kind| format!("{kind}.zsdic"));
    let capacity = zstd::zstd_safe::get_frame_content_size(&raw)
        .ok()
        .flatten()
        .unwrap_or(1 << 30) as usize;
    if let Some(dictionary) = dictionaries.get(&dictionary) {
        let mut decoder = zstd::bulk::Decompressor::with_prepared_dictionary(dictionary).ok()?;
        if let Ok(bytes) = decoder.decompress(&raw, capacity) {
            return Some(bytes);
        }
    }
    zstd::stream::decode_all(&raw[..]).ok()
}

/// Counts every program in the NUP at `input`; with `dump`, also writes
/// each unique stage as `stage-<key>.code` and `.control` there (the names
/// and layout of `ECHORA_SHADER_DIR` dumps).
pub fn inventory(input: &Path, json: bool, dump: Option<&Path>) -> Result<()> {
    let archive = NupArchive::open(input)?;
    let totals = Mutex::new(Totals::default());
    for package in archive.packages() {
        if let Some(exefs) = package.exefs {
            for (name, file) in archive.walk_files(exefs) {
                if name.ends_with(".npdm") {
                    continue;
                }
                let bytes = archive.read_file(file, echora_loader::nso::MAX_NSO_FILE_SIZE)?;
                let Ok(module) =
                    echora_loader::nso::parse(&name, &bytes, echora_loader::nso::MAX_IMAGE_SIZE)
                else {
                    continue;
                };
                for segment in module.segments() {
                    scan(&segment.bytes, "exefs", dump.is_some(), &totals);
                }
            }
        }
        let Some(romfs) = package.romfs else {
            continue;
        };
        let dictionaries: HashMap<_, _> = dictionaries(&archive, romfs)?
            .into_iter()
            .map(|(name, bytes)| (name, zstd::dict::DecoderDictionary::copy(&bytes)))
            .collect();
        let files: Vec<_> = archive
            .walk_files(romfs)
            .into_iter()
            .filter(|(name, _)| !MEDIA.iter().any(|media| name.ends_with(media)))
            .collect();
        let next = std::sync::atomic::AtomicUsize::new(0);
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get().min(8));
        std::thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(|| {
                    loop {
                        let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some((name, file)) = files.get(index) else {
                            break;
                        };
                        let bytes = archive
                            .read_file(*file, u64::MAX)
                            .ok()
                            .and_then(|raw| decompress(name, raw, &dictionaries));
                        let Some(bytes) = bytes else {
                            totals
                                .lock()
                                .unwrap_or_else(|poison| poison.into_inner())
                                .unreadable
                                .push(name.clone());
                            continue;
                        };
                        scan(&bytes, &source_kind(name), dump.is_some(), &totals);
                        totals
                            .lock()
                            .unwrap_or_else(|poison| poison.into_inner())
                            .scanned += 1;
                    }
                });
            }
        });
    }
    let mut totals = totals
        .into_inner()
        .unwrap_or_else(|poison| poison.into_inner());
    totals.unreadable.sort();

    let mut unique_by_stage: BTreeMap<&str, u64> = BTreeMap::new();
    for (stage, _) in totals.unique.values() {
        *unique_by_stage.entry(stage).or_default() += 1;
    }
    if let Some(directory) = dump {
        std::fs::create_dir_all(directory)
            .with_context(|| format!("creating {}", directory.display()))?;
        for (key, (_, bytes)) in &totals.unique {
            let Some((code, control)) = bytes else {
                continue;
            };
            let code_path = directory.join(format!("stage-{key:016x}.code"));
            std::fs::write(&code_path, code)
                .with_context(|| format!("writing {}", code_path.display()))?;
            std::fs::write(code_path.with_extension("control"), control)
                .with_context(|| format!("writing {}", code_path.display()))?;
        }
    }
    let files: u64 = totals.files.values().sum();
    let containers: u64 = totals.containers.values().sum();
    let programs: u64 = totals.programs.values().sum();
    let stage_entries: u64 = totals.stages.values().sum();
    if json {
        let sources: BTreeMap<&String, serde_json::Value> = totals
            .files
            .iter()
            .map(|(source, files)| {
                (
                    source,
                    serde_json::json!({
                        "files": files,
                        "containers": totals.containers.get(source),
                        "programs": totals.programs.get(source),
                    }),
                )
            })
            .collect();
        let report = serde_json::json!({
            "files_scanned": totals.scanned,
            "files": files,
            "containers": containers,
            "programs": programs,
            "stages": stage_entries,
            "stages_by_kind": totals.stages,
            "unique_stages": totals.unique.len(),
            "unique_stages_by_kind": unique_by_stage,
            "sources": sources,
            "unreadable": totals.unreadable,
        });
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }
    println!(
        "{files} files with shader containers ({} files scanned)",
        totals.scanned
    );
    for (source, count) in &totals.files {
        println!(
            "  {source:12} {count:6} files {:6} containers {:7} programs",
            totals.containers.get(source).unwrap_or(&0),
            totals.programs.get(source).unwrap_or(&0),
        );
    }
    println!("{containers} containers, {programs} programs, {stage_entries} stages");
    println!("{} unique stages (by code)", totals.unique.len());
    for stage in STAGES {
        let entries = totals.stages.get(stage).copied().unwrap_or(0);
        if entries > 0 {
            println!(
                "  {stage:12} {entries:7} stages {:6} unique",
                unique_by_stage.get(stage).unwrap_or(&0)
            );
        }
    }
    if !totals.unreadable.is_empty() {
        println!("{} files could not be read:", totals.unreadable.len());
        for name in &totals.unreadable {
            println!("  {name}");
        }
    }
    if dump.is_some() && totals.unique.is_empty() {
        bail!("no shader containers found");
    }
    Ok(())
}
