//! A RomFS image synthesized from an extracted directory tree.
//!
//! Layout follows the documented RomFS format (Switchbrew "RomFS"): a 0x50
//! byte header of ten u64 fields, file data from 0x200 with each file aligned
//! to 0x10, then the directory hash table, directory entries, file hash table
//! and file entries. Entry offsets are byte offsets into their tables, and an
//! empty link is 0xFFFFFFFF. The image is never materialized: reads are served
//! from the in-memory metadata and, for file data, from a `FileSource`: host
//! files under a directory, or a NUP package's RomFS read in place.

use anyhow::{Context, Result, bail, ensure};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const HEADER_SIZE: u64 = 0x50;
const DATA_OFFSET: u64 = 0x200;
const FILE_ALIGNMENT: u64 = 0x10;
const EMPTY: u32 = u32::MAX;

/// One file of the listing: its guest path and exact size.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ListedFile {
    pub path: String,
    pub size: u64,
}

#[derive(Clone, Debug)]
struct ImageFile {
    path: String,
    /// Absolute offset of the file's data in the image.
    offset: u64,
    size: u64,
}

/// Upper bound on decoded compressed files kept in memory.
const DECODED_CACHE_BYTES: u64 = 512 << 20;

/// Where file data comes from.
#[derive(Debug)]
pub enum FileSource {
    /// Host files: the root joined with each listed path.
    Directory(PathBuf),
    /// Files of a NUP package's RomFS, read in place.
    Nup(NupFiles),
}

/// A NUP RomFS read in place. Raw files are read at any offset directly;
/// zstd-compressed files are decoded whole (their SHA-256 is verified) into
/// a bounded cache of recently used files.
#[derive(Debug)]
pub struct NupFiles {
    archive: std::sync::Arc<echora_nup::NupArchive>,
    ids: std::collections::HashMap<String, echora_nup::FileId>,
    cache: std::sync::Mutex<DecodedCache>,
}

#[derive(Debug, Default)]
struct DecodedCache {
    /// Least recently used first.
    entries: std::collections::VecDeque<(echora_nup::FileId, std::sync::Arc<Vec<u8>>)>,
    bytes: u64,
}

impl NupFiles {
    /// The RomFS of `archive`'s package `package` and its file listing.
    pub fn open(
        archive: std::sync::Arc<echora_nup::NupArchive>,
        package: usize,
    ) -> Result<(Self, Vec<ListedFile>)> {
        let pkg = archive
            .packages()
            .get(package)
            .with_context(|| format!("NUP has no package {package}"))?;
        let romfs = pkg
            .romfs
            .with_context(|| format!("NUP package {package} has no RomFS"))?;
        let files = archive.walk_files(romfs);
        let listing = files
            .iter()
            .map(|(path, id)| ListedFile {
                path: path.clone(),
                size: archive.file(*id).size,
            })
            .collect();
        let ids = files.into_iter().collect();
        let source = Self {
            archive,
            ids,
            cache: std::sync::Mutex::new(DecodedCache::default()),
        };
        Ok((source, listing))
    }

    fn read(&self, path: &str, offset: u64, out: &mut [u8]) -> Result<()> {
        let id = *self
            .ids
            .get(path)
            .with_context(|| format!("RomFS file {path:?} is not in the NUP"))?;
        let entry = self.archive.file(id);
        match entry.codec {
            echora_nup::Codec::Raw => self
                .archive
                .read_raw_at(id, offset, out)
                .with_context(|| format!("read RomFS file {path:?}")),
            echora_nup::Codec::Zstd => {
                let data = self.decoded(id, path)?;
                let start = offset as usize;
                let bytes = data
                    .get(start..start + out.len())
                    .with_context(|| format!("read past the end of RomFS file {path:?}"))?;
                out.copy_from_slice(bytes);
                Ok(())
            }
        }
    }

    fn decoded(&self, id: echora_nup::FileId, path: &str) -> Result<std::sync::Arc<Vec<u8>>> {
        let mut cache = self
            .cache
            .lock()
            .map_err(|_| anyhow::anyhow!("RomFS cache poisoned"))?;
        if let Some(index) = cache.entries.iter().position(|(cached, _)| *cached == id) {
            let entry = cache.entries.remove(index).expect("index found above");
            let data = entry.1.clone();
            cache.entries.push_back(entry);
            return Ok(data);
        }
        let size = self.archive.file(id).size;
        ensure!(
            size <= DECODED_CACHE_BYTES,
            "compressed RomFS file {path:?} ({size} bytes) exceeds the decode cache"
        );
        let data = std::sync::Arc::new(
            self.archive
                .read_file(id, size)
                .with_context(|| format!("decode RomFS file {path:?}"))?,
        );
        while cache.bytes + size > DECODED_CACHE_BYTES {
            let Some((_, old)) = cache.entries.pop_front() else {
                break;
            };
            cache.bytes -= old.len() as u64;
        }
        cache.bytes += size;
        cache.entries.push_back((id, data.clone()));
        Ok(data)
    }
}

impl FileSource {
    fn read(&self, path: &str, size: u64, offset: u64, out: &mut [u8]) -> Result<()> {
        match self {
            Self::Directory(root) => {
                let full = root.join(path);
                let mut handle = File::open(&full)
                    .with_context(|| format!("RomFS file {path:?} is not staged locally"))?;
                let actual = handle.metadata()?.len();
                if actual != size {
                    bail!("RomFS file {path:?} is {actual} bytes locally, listed as {size}");
                }
                handle.seek(SeekFrom::Start(offset))?;
                handle
                    .read_exact(out)
                    .with_context(|| format!("read RomFS file {path:?}"))
            }
            Self::Nup(nup) => nup.read(path, offset, out),
        }
    }
}

#[derive(Debug)]
pub struct VirtualRomfs {
    source: FileSource,
    /// Header bytes (0..0x50).
    header: Vec<u8>,
    /// Hash tables and entries, starting at `tables_offset`.
    tables: Vec<u8>,
    tables_offset: u64,
    /// Files sorted by image offset.
    files: Vec<ImageFile>,
    size: u64,
}

/// RomFS name hash: start from the parent offset XOR 123456789, then fold in
/// each name byte with a 5-bit rotate right.
fn name_hash(parent: u32, name: &[u8]) -> u32 {
    let mut hash = parent ^ 123_456_789;
    for &byte in name {
        hash = hash.rotate_right(5) ^ u32::from(byte);
    }
    hash
}

/// Bucket count for `entries` entries: small counts use 3 or an odd count;
/// larger counts advance to the next value with no factor below 18.
fn bucket_count(entries: usize) -> usize {
    if entries < 3 {
        return 3;
    }
    if entries < 19 {
        return entries | 1;
    }
    let mut count = entries;
    while [2, 3, 5, 7, 11, 13, 17]
        .iter()
        .any(|prime| count.is_multiple_of(*prime))
    {
        count += 1;
    }
    count
}

#[derive(Default)]
struct DirNode {
    name: String,
    parent: usize,
    children: Vec<usize>,
    files: Vec<usize>,
}

fn align(value: u64, alignment: u64) -> u64 {
    value.div_ceil(alignment) * alignment
}

fn put_u32(table: &mut [u8], at: usize, value: u32) {
    table[at..at + 4].copy_from_slice(&value.to_le_bytes());
}

impl VirtualRomfs {
    /// Read a listing of `SIZE PATH` lines (one regular file per line, POSIX
    /// paths relative to the RomFS root).
    pub fn read_listing(path: &Path) -> Result<Vec<ListedFile>> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read RomFS listing {}", path.display()))?;
        let mut files = Vec::new();
        for (number, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let (size, name) = line
                .split_once(' ')
                .with_context(|| format!("listing line {} is not SIZE PATH", number + 1))?;
            let size = size
                .parse()
                .with_context(|| format!("listing line {} has an invalid size", number + 1))?;
            files.push(ListedFile {
                path: name.to_string(),
                size,
            });
        }
        Ok(files)
    }

    /// Build the image metadata for `listing`. File data is read from `root`
    /// joined with each listed path when a read reaches it.
    pub fn build(root: &Path, listing: &[ListedFile]) -> Result<Self> {
        Self::build_from(FileSource::Directory(root.to_path_buf()), listing)
    }

    /// Build the image metadata for `listing`, with file data from `source`.
    pub fn build_from(source: FileSource, listing: &[ListedFile]) -> Result<Self> {
        let mut dirs = vec![DirNode::default()];
        let mut by_path: BTreeMap<String, usize> = BTreeMap::new();
        let mut sorted = listing.to_vec();
        sorted.sort_by(|a, b| a.path.cmp(&b.path));
        for (index, file) in sorted.iter().enumerate() {
            let parts: Vec<&str> = file.path.split('/').collect();
            ensure!(
                parts
                    .iter()
                    .all(|part| !part.is_empty() && *part != "." && *part != ".."),
                "listing path {:?} is not a clean relative path",
                file.path
            );
            let mut current = 0;
            let mut prefix = String::new();
            for part in &parts[..parts.len() - 1] {
                if !prefix.is_empty() {
                    prefix.push('/');
                }
                prefix.push_str(part);
                current = match by_path.get(&prefix) {
                    Some(&id) => id,
                    None => {
                        let id = dirs.len();
                        dirs.push(DirNode {
                            name: (*part).to_string(),
                            parent: current,
                            ..DirNode::default()
                        });
                        dirs[current].children.push(id);
                        by_path.insert(prefix.clone(), id);
                        id
                    }
                };
            }
            dirs[current].files.push(index);
        }

        // Entry offsets: directories in creation order, files grouped by
        // directory in the same order.
        let mut dir_offsets = Vec::with_capacity(dirs.len());
        let mut cursor = 0u32;
        for dir in &dirs {
            dir_offsets.push(cursor);
            cursor += 0x18 + align(dir.name.len() as u64, 4) as u32;
        }
        let dir_table_size = cursor as usize;
        let mut file_offsets = vec![0u32; sorted.len()];
        let mut file_order = Vec::with_capacity(sorted.len());
        let mut cursor = 0u32;
        for dir in &dirs {
            for &file in &dir.files {
                file_offsets[file] = cursor;
                file_order.push(file);
                let name = sorted[file].path.rsplit('/').next().unwrap();
                cursor += 0x20 + align(name.len() as u64, 4) as u32;
            }
        }
        let file_table_size = cursor as usize;

        // File data placement.
        let mut data_offsets = vec![0u64; sorted.len()];
        let mut data_cursor = 0u64;
        for &file in &file_order {
            data_cursor = align(data_cursor, FILE_ALIGNMENT);
            data_offsets[file] = data_cursor;
            data_cursor += sorted[file].size;
        }
        let tables_offset = align(DATA_OFFSET + data_cursor, 4);

        let dir_buckets = bucket_count(dirs.len());
        let file_buckets = bucket_count(sorted.len());
        let mut dir_hash = vec![EMPTY; dir_buckets];
        let mut file_hash = vec![EMPTY; file_buckets];
        let mut dir_table = vec![0u8; dir_table_size];
        let mut file_table = vec![0u8; file_table_size];

        for (id, dir) in dirs.iter().enumerate() {
            let at = dir_offsets[id] as usize;
            let parent = dir_offsets[dir.parent];
            let next_sibling = if id == 0 {
                EMPTY
            } else {
                let siblings = &dirs[dir.parent].children;
                let position = siblings.iter().position(|&child| child == id).unwrap();
                siblings
                    .get(position + 1)
                    .map_or(EMPTY, |&next| dir_offsets[next])
            };
            let child = dir.children.first().map_or(EMPTY, |&c| dir_offsets[c]);
            let first_file = dir.files.first().map_or(EMPTY, |&f| file_offsets[f]);
            let bucket = name_hash(parent, dir.name.as_bytes()) as usize % dir_buckets;
            put_u32(&mut dir_table, at, parent);
            put_u32(&mut dir_table, at + 4, next_sibling);
            put_u32(&mut dir_table, at + 8, child);
            put_u32(&mut dir_table, at + 12, first_file);
            put_u32(&mut dir_table, at + 16, dir_hash[bucket]);
            put_u32(&mut dir_table, at + 20, dir.name.len() as u32);
            dir_table[at + 24..at + 24 + dir.name.len()].copy_from_slice(dir.name.as_bytes());
            dir_hash[bucket] = dir_offsets[id];
        }
        for (dir_id, dir) in dirs.iter().enumerate() {
            for (position, &file) in dir.files.iter().enumerate() {
                let at = file_offsets[file] as usize;
                let parent = dir_offsets[dir_id];
                let name = sorted[file].path.rsplit('/').next().unwrap();
                let next = dir
                    .files
                    .get(position + 1)
                    .map_or(EMPTY, |&next| file_offsets[next]);
                let bucket = name_hash(parent, name.as_bytes()) as usize % file_buckets;
                put_u32(&mut file_table, at, parent);
                put_u32(&mut file_table, at + 4, next);
                file_table[at + 8..at + 16].copy_from_slice(&data_offsets[file].to_le_bytes());
                file_table[at + 16..at + 24].copy_from_slice(&sorted[file].size.to_le_bytes());
                put_u32(&mut file_table, at + 24, file_hash[bucket]);
                put_u32(&mut file_table, at + 28, name.len() as u32);
                file_table[at + 32..at + 32 + name.len()].copy_from_slice(name.as_bytes());
                file_hash[bucket] = file_offsets[file];
            }
        }

        let mut tables = Vec::new();
        let dir_hash_offset = tables_offset;
        tables.extend(dir_hash.iter().flat_map(|value| value.to_le_bytes()));
        let dir_meta_offset = tables_offset + tables.len() as u64;
        tables.extend_from_slice(&dir_table);
        let file_hash_offset = tables_offset + tables.len() as u64;
        tables.extend(file_hash.iter().flat_map(|value| value.to_le_bytes()));
        let file_meta_offset = tables_offset + tables.len() as u64;
        tables.extend_from_slice(&file_table);
        let size = tables_offset + tables.len() as u64;

        let mut header = Vec::with_capacity(HEADER_SIZE as usize);
        for value in [
            HEADER_SIZE,
            dir_hash_offset,
            (dir_buckets * 4) as u64,
            dir_meta_offset,
            dir_table_size as u64,
            file_hash_offset,
            (file_buckets * 4) as u64,
            file_meta_offset,
            file_table_size as u64,
            DATA_OFFSET,
        ] {
            header.extend_from_slice(&value.to_le_bytes());
        }

        let mut files: Vec<ImageFile> = sorted
            .iter()
            .enumerate()
            .map(|(index, file)| ImageFile {
                path: file.path.clone(),
                offset: DATA_OFFSET + data_offsets[index],
                size: file.size,
            })
            .collect();
        files.sort_by_key(|file| file.offset);
        Ok(Self {
            source,
            header,
            tables,
            tables_offset,
            files,
            size,
        })
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    /// Read `length` bytes at `offset`. Padding reads as zero. A read that
    /// reaches a listed file whose data is not present in the root, or whose
    /// size differs from the listing, fails and names the file.
    pub fn read(&self, offset: u64, length: usize) -> Result<Vec<u8>> {
        let end = offset
            .checked_add(length as u64)
            .context("RomFS read range overflows")?;
        ensure!(end <= self.size, "RomFS read past the end of the image");
        let mut output = vec![0u8; length];
        let copy = |output: &mut [u8], source_start: u64, bytes: &[u8]| {
            let from = offset.max(source_start);
            let to = end.min(source_start + bytes.len() as u64);
            if from < to {
                output[(from - offset) as usize..(to - offset) as usize].copy_from_slice(
                    &bytes[(from - source_start) as usize..(to - source_start) as usize],
                );
            }
        };
        copy(&mut output, 0, &self.header);
        copy(&mut output, self.tables_offset, &self.tables);
        let first = self
            .files
            .partition_point(|file| file.offset + file.size <= offset);
        for file in &self.files[first..] {
            if file.offset >= end {
                break;
            }
            let from = offset.max(file.offset);
            let to = end.min(file.offset + file.size);
            if from >= to {
                continue;
            }
            self.source.read(
                &file.path,
                file.size,
                from - file.offset,
                &mut output[(from - offset) as usize..(to - offset) as usize],
            )?;
        }
        Ok(output)
    }
}

/// List every regular file below `root` (symlinks are rejected).
pub fn list_directory(root: &Path) -> Result<Vec<ListedFile>> {
    fn walk(root: &Path, relative: &str, out: &mut Vec<ListedFile>) -> Result<()> {
        let directory = if relative.is_empty() {
            root.to_path_buf()
        } else {
            root.join(relative)
        };
        for entry in std::fs::read_dir(&directory)
            .with_context(|| format!("list RomFS directory {}", directory.display()))?
        {
            let entry = entry?;
            let name = entry
                .file_name()
                .into_string()
                .map_err(|_| anyhow::anyhow!("RomFS name is not UTF-8"))?;
            let path = if relative.is_empty() {
                name
            } else {
                format!("{relative}/{name}")
            };
            let kind = entry.file_type()?;
            ensure!(!kind.is_symlink(), "RomFS contains a symlink: {path}");
            if kind.is_dir() {
                walk(root, &path, out)?;
            } else if kind.is_file() {
                out.push(ListedFile {
                    size: entry.metadata()?.len(),
                    path,
                });
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    walk(root, "", &mut files)?;
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    use echora_nup::writer as nup_writer;

    /// The same files served from a directory and from a NUP (raw and
    /// zstd-compressed) give byte-identical images.
    #[test]
    fn nup_source_matches_directory_source() {
        let big: Vec<u8> = (0..300_000u32).map(|i| (i % 7) as u8).collect();
        let files: Vec<(&'static str, Vec<u8>)> = vec![
            ("romfs/Pack/big.pack", big),
            ("romfs/System/a.txt", b"alpha".to_vec()),
            ("romfs/System/b.bin", (0..=255u8).collect()),
            ("romfs/top", b"top-level file".to_vec()),
        ];
        let dir = tempfile::tempdir().unwrap();
        for (path, data) in &files {
            let full = dir.path().join(path);
            std::fs::create_dir_all(full.parent().unwrap()).unwrap();
            std::fs::write(&full, data).unwrap();
        }
        let root = dir.path().join("romfs");
        let from_dir = VirtualRomfs::build(&root, &list_directory(&root).unwrap()).unwrap();

        let nup = nup_writer::write_temp(&nup_writer::build(
            &[nup_writer::PackageSpec::base(files)],
            true,
        ));
        let archive = std::sync::Arc::new(echora_nup::NupArchive::open(nup.path()).unwrap());
        let (source, listing) = NupFiles::open(archive, 0).unwrap();
        let from_nup = VirtualRomfs::build_from(FileSource::Nup(source), &listing).unwrap();

        assert_eq!(from_dir.size(), from_nup.size());
        let size = from_dir.size() as usize;
        assert_eq!(
            from_dir.read(0, size).unwrap(),
            from_nup.read(0, size).unwrap()
        );
        // Unaligned partial reads, twice (the second hits the decode cache).
        for _ in 0..2 {
            for (offset, length) in [(0x1f3, 0x40), (0x200, 0x10007), (size as u64 - 9, 9)] {
                assert_eq!(
                    from_dir.read(offset, length).unwrap(),
                    from_nup.read(offset, length).unwrap()
                );
            }
        }
    }

    fn read_u32(bytes: &[u8], at: u64) -> u32 {
        u32::from_le_bytes(bytes[at as usize..at as usize + 4].try_into().unwrap())
    }
    fn read_u64(bytes: &[u8], at: u64) -> u64 {
        u64::from_le_bytes(bytes[at as usize..at as usize + 8].try_into().unwrap())
    }

    /// Walk the synthesized image the way a RomFS reader does: hash lookup of
    /// each path component, then the file entry's data range.
    fn lookup(image: &[u8], path: &str) -> Option<(u64, u64)> {
        let header = |index: u64| read_u64(image, index * 8);
        let (dir_hash, dir_hash_size, dir_meta) = (header(1), header(2), header(3));
        let (file_hash, file_hash_size, file_meta) = (header(5), header(6), header(7));
        let data = header(9);
        let mut parent = 0u32;
        let parts: Vec<&str> = path.split('/').collect();
        for part in &parts[..parts.len() - 1] {
            let buckets = (dir_hash_size / 4) as u32;
            let mut entry = read_u32(
                image,
                dir_hash + u64::from(name_hash(parent, part.as_bytes()) % buckets) * 4,
            );
            loop {
                if entry == EMPTY {
                    return None;
                }
                let at = dir_meta + u64::from(entry);
                let length = read_u32(image, at + 20) as usize;
                let name = &image[(at + 24) as usize..(at + 24) as usize + length];
                if read_u32(image, at) == parent && name == part.as_bytes() {
                    parent = entry;
                    break;
                }
                entry = read_u32(image, at + 16);
            }
        }
        let name = parts.last().unwrap();
        let buckets = (file_hash_size / 4) as u32;
        let mut entry = read_u32(
            image,
            file_hash + u64::from(name_hash(parent, name.as_bytes()) % buckets) * 4,
        );
        while entry != EMPTY {
            let at = file_meta + u64::from(entry);
            let length = read_u32(image, at + 28) as usize;
            let found = &image[(at + 32) as usize..(at + 32) as usize + length];
            if read_u32(image, at) == parent && found == name.as_bytes() {
                return Some((data + read_u64(image, at + 8), read_u64(image, at + 16)));
            }
            entry = read_u32(image, at + 24);
        }
        None
    }

    #[test]
    fn synthesized_image_resolves_every_listed_file_and_serves_its_bytes() {
        let root = std::env::temp_dir().join(format!("romfs-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let files = [
            ("a.bin", b"alpha".to_vec()),
            ("dir/b.bin", vec![7u8; 33]),
            ("dir/sub/c.bin", b"gamma!".to_vec()),
            ("dir/d.bin", Vec::new()),
            ("z.txt", b"last".to_vec()),
        ];
        let mut listing = Vec::new();
        for (path, data) in &files {
            let full = root.join(path);
            std::fs::create_dir_all(full.parent().unwrap()).unwrap();
            std::fs::write(&full, data).unwrap();
            listing.push(ListedFile {
                path: (*path).to_string(),
                size: data.len() as u64,
            });
        }
        let romfs = VirtualRomfs::build(&root, &listing).unwrap();
        let image = romfs.read(0, romfs.size() as usize).unwrap();
        assert_eq!(read_u64(&image, 0), HEADER_SIZE);
        assert_eq!(read_u64(&image, 72), DATA_OFFSET);
        for (path, data) in &files {
            let (offset, size) = lookup(&image, path).unwrap_or_else(|| panic!("{path}"));
            assert_eq!(offset % FILE_ALIGNMENT, 0);
            assert_eq!(
                &image[offset as usize..(offset + size) as usize],
                data.as_slice()
            );
            // Partial reads straddling the file match too.
            let partial = romfs
                .read(offset.saturating_sub(3), size as usize + 3)
                .unwrap();
            assert_eq!(&partial[3..], data.as_slice());
        }
        assert!(lookup(&image, "dir/missing.bin").is_none());
        // A listed file that is not staged fails with its name.
        let mut absent = listing.clone();
        absent.push(ListedFile {
            path: "big/absent.bin".into(),
            size: 10,
        });
        let romfs = VirtualRomfs::build(&root, &absent).unwrap();
        let image = romfs.read(0, 0x50).unwrap();
        let (offset, _) = {
            let full = romfs.read(0, romfs.size() as usize);
            assert!(full.unwrap_err().to_string().contains("big/absent.bin"));
            let tables = romfs.read(romfs.tables_offset, romfs.tables.len()).unwrap();
            let mut combined = vec![0u8; romfs.size() as usize];
            combined[..0x50].copy_from_slice(&image);
            combined[romfs.tables_offset as usize..].copy_from_slice(&tables);
            lookup(&combined, "big/absent.bin").unwrap()
        };
        assert!(romfs.read(offset, 1).is_err());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn bucket_counts_follow_the_documented_rule() {
        assert_eq!(bucket_count(0), 3);
        assert_eq!(bucket_count(4), 5);
        assert_eq!(bucket_count(18), 19);
        assert_eq!(bucket_count(19), 19);
        assert_eq!(bucket_count(20), 23);
        assert_eq!(bucket_count(22), 23);
    }
}
