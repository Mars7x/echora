//! Profile-driven code layout for a translation.
//!
//! The translation's hot code is a small part of gigabytes of window
//! functions spread across the bundles. `echora layout` reads a run's host
//! profile (`ECHORA_HOST_PROFILE`) and writes, per library of the
//! translation, the functions the guest threads ran in, hottest first, to
//! `layout/<file>.order`. The bundles' and the runner's build scripts link
//! with such a file when it exists: lld places those functions first, and
//! each segment starts on a 2 MiB boundary, so the runtime can move the hot
//! start of each library's code onto 2 MiB pages (`huge_text`). Only the
//! final links change; no region is compiled again.

use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// The directory of a translation's order files.
pub fn layout_dir(translation: &Path) -> PathBuf {
    translation.join("layout")
}

/// The build script of a bundle or runner crate whose linked file is named
/// `file` (`libechora_bundle_0.so`, `echora-run`): links with
/// `../../layout/<file>.order` (relative to `layout`, the path from the
/// crate to the layout directory) when it exists.
pub fn link_script(file: &str, layout: &str) -> String {
    format!(
        "// Generated. Keep this file local.\n\
         // With the profile's order file (echora layout), lld places its\n\
         // functions first and aligns segments to 2 MiB for huge-page text.\n\
         fn main() {{\n\
         \x20   let dir = std::path::Path::new(env!(\"CARGO_MANIFEST_DIR\")).join({layout:?});\n\
         \x20   println!(\"cargo:rerun-if-changed={{}}\", dir.display());\n\
         \x20   let order = dir.join({order:?});\n\
         \x20   if std::env::var(\"CARGO_CFG_TARGET_OS\").as_deref() != Ok(\"linux\") || !order.exists() {{\n\
         \x20       return;\n\
         \x20   }}\n\
         \x20   println!(\"cargo:rustc-link-arg=-Wl,--symbol-ordering-file={{}}\", order.display());\n\
         \x20   println!(\"cargo:rustc-link-arg=-Wl,--no-warn-symbol-ordering\");\n\
         \x20   println!(\"cargo:rustc-link-arg=-Wl,-z,max-page-size=0x200000\");\n\
         \x20   println!(\"cargo:rustc-link-arg=-Wl,-z,separate-loadable-segments\");\n\
         }}\n",
        order = format!("{file}.order"),
    )
}

/// Write the order files of `translation` from the host profile at
/// `profile`: per file of the translation's build (`target/release`), the
/// functions holding `coverage` (a fraction) of the samples taken on guest
/// threads, hottest first. Returns (file, functions) per file written.
pub fn write_order_files(
    translation: &Path,
    profile: &Path,
    coverage: f64,
) -> Result<Vec<(String, usize)>> {
    let text = std::fs::read_to_string(profile)
        .with_context(|| format!("read the host profile {}", profile.display()))?;
    let release = translation.join("target/release");
    let release = release.canonicalize().unwrap_or(release);
    let samples = profile_samples(&text, &release)?;
    let mut symbols: HashMap<PathBuf, Symbols> = HashMap::new();
    let mut weights: HashMap<(PathBuf, String), u64> = HashMap::new();
    let mut total = 0;
    for (path, offset, count) in samples {
        let table = match symbols.entry(path.clone()) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => entry
                .insert(Symbols::read(&path).with_context(|| format!("read {}", path.display()))?),
        };
        total += count;
        if let Some(name) = table.at_offset(offset) {
            *weights.entry((path, name.to_owned())).or_default() += count;
        }
    }
    if total == 0 {
        bail!(
            "the profile has no guest-thread samples in {}",
            release.display()
        );
    }
    let mut ranked: Vec<_> = weights.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let wanted = (total as f64 * coverage.clamp(0.0, 1.0)).ceil() as u64;
    let mut files: HashMap<String, Vec<String>> = HashMap::new();
    let mut taken = 0;
    for ((path, name), count) in ranked {
        if taken >= wanted {
            break;
        }
        taken += count;
        let file = path.file_name().unwrap_or_default().to_string_lossy();
        files.entry(file.into_owned()).or_default().push(name);
    }
    let dir = layout_dir(translation);
    std::fs::create_dir_all(&dir)?;
    let mut written: Vec<_> = files
        .into_iter()
        .map(|(file, names)| {
            std::fs::write(dir.join(format!("{file}.order")), names.join("\n") + "\n")?;
            Ok((file, names.len()))
        })
        .collect::<Result<_>>()?;
    written.sort();
    Ok(written)
}

/// The host profile's guest-thread samples in files under `release`, as
/// (file, file offset, count).
fn profile_samples(text: &str, release: &Path) -> Result<Vec<(PathBuf, u64, u64)>> {
    let Some((maps, samples)) = text.split_once("\n--\n") else {
        bail!("not a host profile (no `--` line after the address map)");
    };
    // Executable file mappings under `release`: (start, end, file offset, path).
    let mappings: Vec<(u64, u64, u64, PathBuf)> = maps
        .lines()
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            let range = fields.next()?;
            let perms = fields.next()?;
            let offset = u64::from_str_radix(fields.next()?, 16).ok()?;
            let path = PathBuf::from(fields.nth(2)?);
            if !perms.contains('x') || !path.starts_with(release) {
                return None;
            }
            let (start, end) = range.split_once('-')?;
            Some((
                u64::from_str_radix(start, 16).ok()?,
                u64::from_str_radix(end, 16).ok()?,
                offset,
                path,
            ))
        })
        .collect();
    let mut out = Vec::new();
    for line in samples.lines() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        let [count, address, _thread, _top, guest] = fields[..] else {
            continue;
        };
        if guest == "0" {
            continue;
        }
        let (Ok(count), Ok(address)) = (
            count.parse::<u64>(),
            u64::from_str_radix(address.trim_start_matches("0x"), 16),
        ) else {
            continue;
        };
        if let Some((start, _, offset, path)) = mappings
            .iter()
            .find(|(start, end, _, _)| (*start..*end).contains(&address))
        {
            out.push((path.clone(), address - start + offset, count));
        }
    }
    Ok(out)
}

/// The function symbols of an ELF64 little-endian file, by file offset.
struct Symbols {
    /// (file offset, size, name), sorted by offset.
    functions: Vec<(u64, u64, String)>,
}

impl Symbols {
    fn read(path: &Path) -> Result<Self> {
        let data = std::fs::read(path)?;
        Self::parse(&data)
    }

    fn parse(data: &[u8]) -> Result<Self> {
        let u16_at = |at: usize| -> Result<u64> {
            Ok(
                u16::from_le_bytes(data.get(at..at + 2).context("truncated ELF")?.try_into()?)
                    .into(),
            )
        };
        let u32_at = |at: usize| -> Result<u64> {
            Ok(
                u32::from_le_bytes(data.get(at..at + 4).context("truncated ELF")?.try_into()?)
                    .into(),
            )
        };
        let u64_at = |at: usize| -> Result<u64> {
            Ok(u64::from_le_bytes(
                data.get(at..at + 8).context("truncated ELF")?.try_into()?,
            ))
        };
        if data.get(..6) != Some(b"\x7fELF\x02\x01") {
            bail!("not a little-endian ELF64 file");
        }
        // Loadable segments: (virtual address, file offset, file size).
        let (phoff, phentsize, phnum) = (u64_at(0x20)?, u16_at(0x36)?, u16_at(0x38)?);
        let mut loads = Vec::new();
        for index in 0..phnum {
            let at = (phoff + index * phentsize) as usize;
            if u32_at(at)? == 1 {
                loads.push((u64_at(at + 0x10)?, u64_at(at + 8)?, u64_at(at + 0x20)?));
            }
        }
        let to_offset = |vaddr: u64| {
            loads.iter().find_map(|&(base, offset, size)| {
                (base..base + size)
                    .contains(&vaddr)
                    .then(|| vaddr - base + offset)
            })
        };
        let (shoff, shentsize, shnum) = (u64_at(0x28)?, u16_at(0x3a)?, u16_at(0x3c)?);
        let section = |index: u64| (shoff + index * shentsize) as usize;
        let mut functions = Vec::new();
        for index in 0..shnum {
            let at = section(index);
            // SHT_SYMTAB (2), else SHT_DYNSYM (11) in a stripped file.
            let kind = u32_at(at + 4)?;
            if kind != 2 && kind != 11 {
                continue;
            }
            let (offset, size, entsize) =
                (u64_at(at + 0x18)?, u64_at(at + 0x20)?, u64_at(at + 0x38)?);
            let strings = section(u32_at(at + 0x28)?);
            let (string_offset, string_size) = (u64_at(strings + 0x18)?, u64_at(strings + 0x20)?);
            let names = data
                .get(string_offset as usize..(string_offset + string_size) as usize)
                .context("truncated ELF string table")?;
            for entry in 0..size / entsize.max(1) {
                let symbol = (offset + entry * entsize) as usize;
                let info = *data.get(symbol + 4).context("truncated ELF symbol")?;
                let (value, length) = (u64_at(symbol + 8)?, u64_at(symbol + 0x10)?);
                // STT_FUNC with a size and an address.
                if info & 0xf != 2 || length == 0 || value == 0 {
                    continue;
                }
                let name_at = u32_at(symbol)? as usize;
                let name = names
                    .get(name_at..)
                    .and_then(|rest| rest.split(|&b| b == 0).next())
                    .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
                    .unwrap_or_default();
                if let (false, Some(file_offset)) = (name.is_empty(), to_offset(value)) {
                    functions.push((file_offset, length, name));
                }
            }
            if kind == 2 {
                break;
            }
        }
        functions.sort();
        functions.dedup_by_key(|function| function.0);
        Ok(Self { functions })
    }

    /// The function holding file offset `offset`.
    fn at_offset(&self, offset: u64) -> Option<&str> {
        let index = self.functions.partition_point(|f| f.0 <= offset);
        let (start, size, name) = self.functions.get(index.checked_sub(1)?)?;
        (offset < start + size).then_some(name.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_running_test_binary_resolves_its_own_functions() {
        let exe = std::env::current_exe().unwrap();
        let symbols = Symbols::read(&exe).unwrap();
        assert!(!symbols.functions.is_empty());
        // Every function resolves at its own start, and no two overlap.
        let (offset, size, name) = &symbols.functions[symbols.functions.len() / 2];
        assert_eq!(symbols.at_offset(*offset), Some(name.as_str()));
        assert_eq!(symbols.at_offset(offset + size - 1), Some(name.as_str()));
    }

    #[test]
    fn order_files_rank_guest_samples_by_function() {
        let exe = std::env::current_exe().unwrap().canonicalize().unwrap();
        let symbols = Symbols::read(&exe).unwrap();
        let release = std::env::temp_dir().join(format!("echora-layout-{}", std::process::id()));
        let target = release.join("target/release");
        std::fs::create_dir_all(&target).unwrap();
        let copy = target.join("echora-run");
        std::fs::copy(&exe, &copy).unwrap();
        // One mapping of the whole file at 0x10000000, offset 0.
        let len = std::fs::metadata(&copy).unwrap().len();
        let (hot, warm) = (&symbols.functions[10], &symbols.functions[20]);
        let profile = format!(
            "10000000-{:x} r-xp 00000000 00:00 0 {}\n--\n\
             7 {:#x} 1 0x0 5\n3 {:#x} 1 0x0 5\n50 {:#x} 1 0x0 0\n",
            0x1000_0000 + len,
            copy.display(),
            0x1000_0000 + hot.0,
            0x1000_0000 + warm.0,
            0x1000_0000 + warm.0,
        );
        let path = release.join("profile.txt");
        std::fs::write(&path, profile).unwrap();
        let written = write_order_files(&release, &path, 1.0).unwrap();
        assert_eq!(written, vec![("echora-run".to_owned(), 2)]);
        let order = std::fs::read_to_string(release.join("layout/echora-run.order")).unwrap();
        assert_eq!(order, format!("{}\n{}\n", hot.2, warm.2));
        std::fs::remove_dir_all(&release).unwrap();
    }

    #[test]
    fn link_script_names_the_order_file() {
        let script = link_script("libechora_bundle_3.so", "../../layout");
        assert!(script.contains("\"libechora_bundle_3.so.order\""));
        assert!(script.contains("\"../../layout\""));
        assert!(script.contains("max-page-size=0x200000"));
    }
}
