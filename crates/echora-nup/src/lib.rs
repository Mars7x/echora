//! Reader for NUP v1 containers.
//!
//! NUP stores the plaintext ExeFS and RomFS trees of base, update and add-on
//! packages in one file. The format and its Python reference implementation
//! are at <https://github.com/a55uka/nup>. This crate validates the whole
//! table of contents when an archive is opened, so every later lookup and read
//! works on bounds-checked records, and it streams file contents so that
//! multi-gigabyte RomFS files never need to be held in memory.

mod extract;
pub mod format;
mod meta;
mod read;
mod verify;
/// A minimal NUP writer for tests (the `writer` feature).
#[cfg(feature = "writer")]
#[allow(clippy::unwrap_used, missing_debug_implementations)]
pub mod writer;

use std::fs::File;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

pub use extract::ExtractStats;
pub use format::{Codec, PackageKind};
pub use meta::PackageMeta;
pub use read::FileReader;
pub use verify::{VerifyProblem, VerifyReport};

use format::Fields;

/// Upper bound on the table of contents. The largest known dump (about
/// 190,000 files) needs 21 MB; this leaves ample room while refusing to
/// allocate for a corrupt size field.
pub const MAX_TOC_SIZE: u64 = 512 << 20;
/// Upper bound on a package meta block (icons are the largest records).
pub const MAX_META_SIZE: u32 = 64 << 20;
/// Deepest directory nesting accepted in a package tree.
pub const MAX_TREE_DEPTH: usize = 64;

#[derive(Debug, thiserror::Error)]
pub enum NupError {
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("malformed NUP: {0}")]
    Format(String),
    #[error("integrity check failed: {0}")]
    Integrity(String),
}

pub type Result<T> = std::result::Result<T, NupError>;

fn malformed(msg: impl Into<String>) -> NupError {
    NupError::Format(msg.into())
}

/// Validated superblock fields.
#[derive(Debug, Clone)]
pub struct Superblock {
    pub header_size: u16,
    pub header_version: u16,
    pub toc_offset: u64,
    pub toc_size: u64,
    pub toc_sha256: [u8; 32],
    pub blob_offset: u64,
    pub file_size: u64,
    pub package_count: u32,
    pub uuid: [u8; 16],
}

/// Index of a directory record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DirId(pub u32);

/// Index of a file record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FileId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Node {
    Dir(DirId),
    File(FileId),
}

#[derive(Debug, Clone)]
pub struct Package {
    pub index: usize,
    pub title_id: u64,
    pub kind: PackageKind,
    pub version: u32,
    pub standalone_update: bool,
    pub root: DirId,
    pub exefs: Option<DirId>,
    pub romfs: Option<DirId>,
    meta_offset: u64,
    meta_size: u32,
}

#[derive(Debug, Clone)]
struct DirRecord {
    name: (u32, u32),
    first_child: u32,
    child_count: u32,
}

/// A validated file record.
#[derive(Debug, Clone)]
pub struct FileEntry {
    name: (u32, u32),
    pub size: u64,
    pub extent_offset: u64,
    pub extent_size: u64,
    pub sha256: [u8; 32],
    pub codec: Codec,
}

/// An opened, fully validated NUP archive.
#[derive(Debug)]
pub struct NupArchive {
    path: PathBuf,
    file: File,
    superblock: Superblock,
    packages: Vec<Package>,
    dirs: Vec<DirRecord>,
    files: Vec<FileEntry>,
    children: Vec<u32>,
    pool: Vec<u8>,
}

impl NupArchive {
    /// Opens `path`, verifies the superblock and TOC hash, and validates every
    /// record, name, extent and directory tree before returning.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let io_err = |source| NupError::Io {
            path: path.clone(),
            source,
        };
        let file = File::open(&path).map_err(io_err)?;
        let actual_len = file.metadata().map_err(io_err)?.len();

        let mut head = [0u8; format::SUPERBLOCK_SIZE];
        read::read_exact_at(&file, &mut head, 0).map_err(io_err)?;
        let superblock = parse_superblock(&head, actual_len)?;

        let toc_len = usize::try_from(superblock.toc_size)
            .map_err(|_| malformed("TOC size does not fit in memory"))?;
        let mut toc = vec![0u8; toc_len];
        read::read_exact_at(&file, &mut toc, superblock.toc_offset).map_err(io_err)?;
        let digest: [u8; 32] = Sha256::digest(&toc).into();
        if digest != superblock.toc_sha256 {
            return Err(NupError::Integrity(
                "TOC SHA-256 does not match superblock".into(),
            ));
        }

        let mut archive = parse_toc(path, file, superblock, &toc)?;
        archive.validate()?;
        Ok(archive)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn superblock(&self) -> &Superblock {
        &self.superblock
    }

    pub fn packages(&self) -> &[Package] {
        &self.packages
    }

    pub fn dir_count(&self) -> usize {
        self.dirs.len()
    }

    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    pub fn file(&self, id: FileId) -> &FileEntry {
        &self.files[id.0 as usize]
    }

    pub fn files(&self) -> impl ExactSizeIterator<Item = (FileId, &FileEntry)> {
        self.files
            .iter()
            .enumerate()
            .map(|(i, f)| (FileId(i as u32), f))
    }

    /// Name of a directory or file. Names were validated as UTF-8 path
    /// components when the archive was opened.
    pub fn name(&self, node: Node) -> &str {
        let (off, len) = match node {
            Node::Dir(d) => self.dirs[d.0 as usize].name,
            Node::File(f) => self.files[f.0 as usize].name,
        };
        let bytes = &self.pool[off as usize..(off + len) as usize];
        // Validated in `validate`.
        std::str::from_utf8(bytes).unwrap_or_default()
    }

    pub fn children(&self, dir: DirId) -> impl ExactSizeIterator<Item = Node> + '_ {
        let d = &self.dirs[dir.0 as usize];
        let range = d.first_child as usize..(d.first_child + d.child_count) as usize;
        self.children[range].iter().map(|&id| self.node(id))
    }

    /// Looks up a direct child by exact name. Children are sorted bytewise, so
    /// this is a binary search.
    pub fn lookup(&self, dir: DirId, name: &str) -> Option<Node> {
        let d = &self.dirs[dir.0 as usize];
        let slots =
            &self.children[d.first_child as usize..(d.first_child + d.child_count) as usize];
        slots
            .binary_search_by(|&id| self.name(self.node(id)).as_bytes().cmp(name.as_bytes()))
            .ok()
            .map(|i| self.node(slots[i]))
    }

    /// Resolves a `/`-separated path relative to `dir`. Empty components are
    /// ignored; `.` and `..` never match because they are rejected as names.
    pub fn resolve(&self, dir: DirId, path: &str) -> Option<Node> {
        let mut node = Node::Dir(dir);
        for part in path.split('/').filter(|p| !p.is_empty()) {
            let Node::Dir(d) = node else { return None };
            node = self.lookup(d, part)?;
        }
        Some(node)
    }

    /// All files below `dir`, with `/`-separated paths relative to it, in
    /// tree order.
    pub fn walk_files(&self, dir: DirId) -> Vec<(String, FileId)> {
        let mut out = Vec::new();
        let mut stack = vec![(dir, String::new())];
        while let Some((d, prefix)) = stack.pop() {
            let mut subdirs = Vec::new();
            for child in self.children(d) {
                let path = if prefix.is_empty() {
                    self.name(child).to_owned()
                } else {
                    format!("{prefix}/{}", self.name(child))
                };
                match child {
                    Node::File(f) => out.push((path, f)),
                    Node::Dir(sub) => subdirs.push((sub, path)),
                }
            }
            stack.extend(subdirs.into_iter().rev());
        }
        out
    }

    /// Reads and parses a package's meta block.
    pub fn package_meta(&self, package: &Package) -> Result<PackageMeta> {
        if package.meta_size == 0 {
            return Ok(PackageMeta::default());
        }
        let mut buf = vec![0u8; package.meta_size as usize];
        read::read_exact_at(&self.file, &mut buf, package.meta_offset).map_err(|source| {
            NupError::Io {
                path: self.path.clone(),
                source,
            }
        })?;
        meta::parse(&buf)
    }

    fn node(&self, id: u32) -> Node {
        let dirs = self.dirs.len() as u32;
        if id < dirs {
            Node::Dir(DirId(id))
        } else {
            Node::File(FileId(id - dirs))
        }
    }

    fn name_bytes(&self, (off, len): (u32, u32)) -> &[u8] {
        &self.pool[off as usize..(off + len) as usize]
    }

    fn validate(&mut self) -> Result<()> {
        let sb = &self.superblock;
        let toc_end = sb.toc_offset + sb.toc_size;

        for (i, d) in self.dirs.iter().enumerate() {
            check_name_ref(d.name, self.pool.len(), || format!("directory {i}"))?;
            let end = u64::from(d.first_child) + u64::from(d.child_count);
            if end > self.children.len() as u64 {
                return Err(malformed(format!(
                    "directory {i}: child range out of bounds"
                )));
            }
        }
        let used: u64 = self.dirs.iter().map(|d| u64::from(d.child_count)).sum();
        if used != self.children.len() as u64 {
            return Err(malformed(format!(
                "child slots used {used} != total {}",
                self.children.len()
            )));
        }
        let node_count = (self.dirs.len() + self.files.len()) as u64;
        if let Some(bad) = self.children.iter().find(|&&c| u64::from(c) >= node_count) {
            return Err(malformed(format!("child node id {bad} out of range")));
        }

        for (i, f) in self.files.iter().enumerate() {
            check_name_ref(f.name, self.pool.len(), || format!("file {i}"))?;
            if f.size == 0 {
                if f.extent_size != 0 || f.extent_offset != 0 || f.codec != Codec::Raw {
                    return Err(malformed(format!("file {i}: empty file with an extent")));
                }
                continue;
            }
            if f.codec == Codec::Raw && f.extent_size != f.size {
                return Err(malformed(format!("file {i}: raw extent size != file size")));
            }
            let end = f
                .extent_offset
                .checked_add(f.extent_size)
                .ok_or_else(|| malformed(format!("file {i}: extent overflows")))?;
            if f.extent_offset < sb.blob_offset || end > sb.file_size {
                return Err(malformed(format!("file {i}: extent out of blob region")));
            }
            if f.extent_offset % format::EXTENT_ALIGN != 0 {
                return Err(malformed(format!("file {i}: extent not 16-byte aligned")));
            }
        }
        self.check_extent_overlap()?;

        // Only names reached through a directory become path components;
        // package roots are unnamed in real archives.
        for &id in &self.children {
            let name = self.name_bytes(self.record_name(id));
            check_component(name).map_err(|why| malformed(format!("node {id}: {why}")))?;
        }

        for (i, d) in self.dirs.iter().enumerate() {
            let slots =
                &self.children[d.first_child as usize..(d.first_child + d.child_count) as usize];
            for pair in slots.windows(2) {
                let a = self.name_bytes(self.record_name(pair[0]));
                let b = self.name_bytes(self.record_name(pair[1]));
                if a >= b {
                    return Err(malformed(format!(
                        "directory {i}: children not strictly sorted by name"
                    )));
                }
            }
        }

        let has_base = self.packages.iter().any(|p| p.kind == PackageKind::Base);
        let dir_count = self.dirs.len() as u32;
        for p in &self.packages {
            let i = p.index;
            for d in [Some(p.root), p.exefs, p.romfs].into_iter().flatten() {
                if d.0 >= dir_count {
                    return Err(malformed(format!("package {i}: directory out of range")));
                }
            }
            if p.meta_size > 0 {
                let end = p.meta_offset.checked_add(u64::from(p.meta_size));
                if p.meta_offset < toc_end || end.is_none_or(|e| e > sb.file_size) {
                    return Err(malformed(format!("package {i}: meta block out of bounds")));
                }
                if p.meta_size > MAX_META_SIZE {
                    return Err(malformed(format!("package {i}: meta block too large")));
                }
            }
            if p.kind == PackageKind::Update && !has_base && !p.standalone_update {
                return Err(malformed(format!(
                    "package {i}: update without a base and without standalone_update"
                )));
            }
        }
        self.check_trees()
    }

    fn record_name(&self, id: u32) -> (u32, u32) {
        match self.node(id) {
            Node::Dir(d) => self.dirs[d.0 as usize].name,
            Node::File(f) => self.files[f.0 as usize].name,
        }
    }

    /// Extents may be shared only by identical records (SPEC §3.6).
    fn check_extent_overlap(&self) -> Result<()> {
        let mut extents: Vec<&FileEntry> = self.files.iter().filter(|f| f.size > 0).collect();
        extents.sort_unstable_by_key(|f| (f.extent_offset, f.extent_size));
        let mut prev: Option<&FileEntry> = None;
        for f in extents {
            if let Some(p) = prev {
                let shared = p.extent_offset == f.extent_offset
                    && p.extent_size == f.extent_size
                    && p.size == f.size
                    && p.codec == f.codec
                    && p.sha256 == f.sha256;
                if shared {
                    continue;
                }
                if f.extent_offset < p.extent_offset + p.extent_size {
                    return Err(malformed(format!(
                        "extents overlap at offset {:#x}",
                        f.extent_offset
                    )));
                }
            }
            prev = Some(f);
        }
        Ok(())
    }

    /// Every package tree must be acyclic and bounded in depth, and a
    /// directory may appear only once within a tree.
    fn check_trees(&self) -> Result<()> {
        let mut seen = vec![usize::MAX; self.dirs.len()];
        for p in &self.packages {
            let mut stack = vec![(p.root, 0usize)];
            while let Some((d, depth)) = stack.pop() {
                if depth > MAX_TREE_DEPTH {
                    return Err(malformed(format!("package {}: tree too deep", p.index)));
                }
                if seen[d.0 as usize] == p.index {
                    return Err(malformed(format!(
                        "package {}: directory {} reachable twice",
                        p.index, d.0
                    )));
                }
                seen[d.0 as usize] = p.index;
                for child in self.children(d) {
                    if let Node::Dir(sub) = child {
                        stack.push((sub, depth + 1));
                    }
                }
            }
            for (label, d) in [("exefs", p.exefs), ("romfs", p.romfs)] {
                if let Some(d) = d
                    && seen[d.0 as usize] != p.index
                {
                    return Err(malformed(format!(
                        "package {}: {label} directory is outside the package tree",
                        p.index
                    )));
                }
            }
        }
        Ok(())
    }
}

fn check_name_ref(
    (off, len): (u32, u32),
    pool_len: usize,
    what: impl FnOnce() -> String,
) -> Result<()> {
    if u64::from(off) + u64::from(len) > pool_len as u64 {
        return Err(malformed(format!("{}: name out of string pool", what())));
    }
    Ok(())
}

/// Names are labels, but they become path components when extracted, so only
/// plain UTF-8 components are accepted.
fn check_component(name: &[u8]) -> std::result::Result<(), &'static str> {
    let s = std::str::from_utf8(name).map_err(|_| "name is not UTF-8")?;
    if s.is_empty() {
        return Err("empty name");
    }
    if s == "." || s == ".." {
        return Err("name is a relative path component");
    }
    if s.contains(['/', '\\', '\0']) {
        return Err("name contains a path separator or NUL");
    }
    Ok(())
}

fn parse_superblock(head: &[u8; format::SUPERBLOCK_SIZE], actual_len: u64) -> Result<Superblock> {
    let f = Fields(head);
    if f.bytes::<4>(0) != format::MAGIC {
        return Err(malformed("not a NUP v1 file (bad magic)"));
    }
    let sb = Superblock {
        header_size: f.u16(0x04),
        header_version: f.u16(0x06),
        toc_offset: f.u64(0x08),
        toc_size: f.u64(0x10),
        toc_sha256: f.bytes(0x18),
        blob_offset: f.u64(0x38),
        file_size: f.u64(0x40),
        package_count: f.u32(0x48),
        uuid: f.bytes(0x50),
    };
    let flags = f.u32(0x4c);
    if (sb.header_size as usize) < format::SUPERBLOCK_SIZE {
        return Err(malformed("header_size smaller than 0x80"));
    }
    if sb.toc_offset != u64::from(sb.header_size) {
        return Err(malformed("toc_offset != header_size"));
    }
    if flags != 0 {
        return Err(malformed(format!("unknown superblock flags {flags:#x}")));
    }
    if sb.file_size != actual_len {
        return Err(malformed(format!(
            "file_size {} != actual length {actual_len} (truncated or extended file)",
            sb.file_size
        )));
    }
    if sb.toc_size > MAX_TOC_SIZE || (sb.toc_size as usize) < format::TOC_HEADER_SIZE {
        return Err(malformed(format!("unsupported TOC size {}", sb.toc_size)));
    }
    let toc_end = sb.toc_offset + sb.toc_size;
    if !sb.blob_offset.is_multiple_of(format::EXTENT_ALIGN)
        || sb.blob_offset < toc_end
        || sb.blob_offset > sb.file_size
    {
        return Err(malformed("blob_offset misplaced or misaligned"));
    }
    if sb.package_count == 0 {
        return Err(malformed("package_count is zero"));
    }
    Ok(sb)
}

fn parse_toc(path: PathBuf, file: File, superblock: Superblock, toc: &[u8]) -> Result<NupArchive> {
    let head = Fields(&toc[..format::TOC_HEADER_SIZE]);
    if head.bytes::<4>(0) != format::TOC_MAGIC {
        return Err(malformed("bad TOC magic"));
    }
    let package_count = head.u32(0x04);
    if package_count != superblock.package_count {
        return Err(malformed("TOC package_count != superblock package_count"));
    }
    let dir_count = head.u64(0x08);
    let file_count = head.u64(0x10);
    let pool_off = head.u64(0x18);
    let pool_size = head.u64(0x20);
    let toc_len = toc.len() as u64;

    let records_end = (|| {
        let mut end = format::TOC_HEADER_SIZE as u64;
        end =
            end.checked_add(u64::from(package_count).checked_mul(format::PACKAGE_SIZE as u64)?)?;
        end = end.checked_add(dir_count.checked_mul(format::DIR_SIZE as u64)?)?;
        end.checked_add(file_count.checked_mul(format::FILE_SIZE as u64)?)
    })()
    .ok_or_else(|| malformed("record counts overflow"))?;
    if records_end > toc_len || dir_count + file_count > u64::from(u32::MAX) {
        return Err(malformed("record arrays exceed the TOC"));
    }
    if pool_off < records_end || pool_off.checked_add(pool_size).is_none_or(|e| e > toc_len) {
        return Err(malformed("string pool out of TOC bounds"));
    }
    if (pool_off - records_end) % 4 != 0 {
        return Err(malformed("children array is not a whole number of slots"));
    }

    let mut pos = format::TOC_HEADER_SIZE;
    let mut packages = Vec::with_capacity(package_count as usize);
    for index in 0..package_count as usize {
        let r = Fields(&toc[pos..pos + format::PACKAGE_SIZE]);
        pos += format::PACKAGE_SIZE;
        let kind = PackageKind::from_raw(r.u8(0x18))
            .ok_or_else(|| malformed(format!("package {index}: unknown kind {}", r.u8(0x18))))?;
        let flags = r.u8(0x19);
        if flags & !format::PACKAGE_FLAG_STANDALONE_UPDATE != 0 {
            return Err(malformed(format!(
                "package {index}: unknown flags {flags:#x}"
            )));
        }
        let opt_dir = |raw: u32| (raw != format::NO_DIR).then_some(DirId(raw));
        packages.push(Package {
            index,
            title_id: r.u64(0x00),
            root: DirId(r.u32(0x08)),
            exefs: opt_dir(r.u32(0x0c)),
            romfs: opt_dir(r.u32(0x10)),
            version: r.u32(0x14),
            kind,
            standalone_update: flags & format::PACKAGE_FLAG_STANDALONE_UPDATE != 0,
            meta_offset: r.u64(0x20),
            meta_size: r.u32(0x28),
        });
    }

    let mut dirs = Vec::with_capacity(dir_count as usize);
    for _ in 0..dir_count {
        let r = Fields(&toc[pos..pos + format::DIR_SIZE]);
        pos += format::DIR_SIZE;
        dirs.push(DirRecord {
            name: (r.u32(0x00), r.u32(0x04)),
            first_child: r.u32(0x08),
            child_count: r.u32(0x0c),
        });
    }

    let mut files = Vec::with_capacity(file_count as usize);
    for i in 0..file_count {
        let r = Fields(&toc[pos..pos + format::FILE_SIZE]);
        pos += format::FILE_SIZE;
        let codec = match r.u8(0x40) {
            format::CODEC_RAW => Codec::Raw,
            format::CODEC_ZSTD => Codec::Zstd,
            other => return Err(malformed(format!("file {i}: unknown codec {other}"))),
        };
        files.push(FileEntry {
            name: (r.u32(0x00), r.u32(0x04)),
            size: r.u64(0x08),
            extent_offset: r.u64(0x10),
            extent_size: r.u64(0x18),
            sha256: r.bytes(0x20),
            codec,
        });
    }

    let children = toc[pos..pool_off as usize]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let pool = toc[pool_off as usize..(pool_off + pool_size) as usize].to_vec();

    let expected_uuid = content_uuid(&packages);
    if expected_uuid != superblock.uuid {
        return Err(NupError::Integrity(
            "superblock UUID does not match packages".into(),
        ));
    }

    Ok(NupArchive {
        path,
        file,
        superblock,
        packages,
        dirs,
        files,
        children,
        pool,
    })
}

/// SPEC §2.1: the first 16 bytes of SHA-256 over each package's title ID,
/// version and kind.
fn content_uuid(packages: &[Package]) -> [u8; 16] {
    let mut h = Sha256::new();
    for p in packages {
        h.update(p.title_id.to_le_bytes());
        h.update(p.version.to_le_bytes());
        h.update([p.kind.raw()]);
    }
    let digest: [u8; 32] = h.finalize().into();
    let mut uuid = [0; 16];
    uuid.copy_from_slice(&digest[..16]);
    uuid
}
