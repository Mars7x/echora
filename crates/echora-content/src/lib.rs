//! Game content input for Echora.
//!
//! A NUP container and an extracted dump directory are both turned into the
//! same [`ContentInventory`] and the same content-addressed ExeFS cache entry,
//! so everything downstream (executable analysis, translation caches, the
//! runtime's file systems) is independent of how the game was supplied.

pub mod exefs;

use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use echora_nup::{Node, NupArchive, PackageKind};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub use exefs::{NpdmInfo, NsoHeader};

/// ExeFS modules are small (tens of MB); anything larger is not a module.
pub const MAX_EXEFS_FILE: u64 = 1 << 30;
const MANIFEST: &str = "echora-exefs.json";

#[derive(Debug, thiserror::Error)]
pub enum ContentError {
    #[error(transparent)]
    Nup(#[from] echora_nup::NupError),
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("{0}")]
    Layout(String),
    #[error("integrity check failed: {0}")]
    Integrity(String),
}

pub type Result<T> = std::result::Result<T, ContentError>;

fn io_at(path: &Path) -> impl FnOnce(io::Error) -> ContentError + '_ {
    move |source| ContentError::Io {
        path: path.to_path_buf(),
        source,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContentKind {
    Base,
    Update,
    Addon,
    AddonUpdate,
    System,
}

impl From<PackageKind> for ContentKind {
    fn from(k: PackageKind) -> Self {
        match k {
            PackageKind::Base => Self::Base,
            PackageKind::Update => Self::Update,
            PackageKind::Addon => Self::Addon,
            PackageKind::AddonUpdate => Self::AddonUpdate,
            PackageKind::System => Self::System,
        }
    }
}

/// One package of an extracted dump directory.
#[derive(Debug, Clone)]
pub struct DirPackage {
    kind: ContentKind,
    title_id: Option<u64>,
    version: u32,
    exefs: Option<PathBuf>,
    romfs: Option<PathBuf>,
}

/// Where the game content comes from.
#[derive(Debug)]
pub enum ContentSource {
    Nup(Box<NupArchive>),
    Directory {
        root: PathBuf,
        packages: Vec<DirPackage>,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct ModuleInventory {
    pub name: String,
    pub size: u64,
    pub sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nso: Option<NsoHeader>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub npdm: Option<NpdmInfo>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RomfsInventory {
    pub files: u64,
    pub bytes: u64,
    /// True when every file has a recorded SHA-256 (NUP input); directory
    /// input is counted but not hashed during inventory.
    pub hashes_recorded: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct PackageInventory {
    pub index: usize,
    pub kind: ContentKind,
    pub title_id: Option<String>,
    /// NCM content version, verbatim.
    pub version: u32,
    pub standalone_update: bool,
    pub display_version: Option<String>,
    pub title_name: Option<String>,
    /// The package's icon image (JPEG or PNG as distributed), when the input
    /// carries one. Not included in serialized reports.
    #[serde(skip)]
    pub icon: Option<Vec<u8>>,
    pub exefs: Vec<ModuleInventory>,
    /// Content key of the ExeFS module set: SHA-256 over the sorted module
    /// names and their SHA-256 values.
    pub exefs_key: Option<String>,
    pub romfs: Option<RomfsInventory>,
}

impl PackageInventory {
    pub fn main_build_id(&self) -> Option<String> {
        self.exefs
            .iter()
            .find(|m| m.name == "main")
            .and_then(|m| m.nso.as_ref())
            .map(NsoHeader::build_id_hex)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ContentInventory {
    pub source: &'static str,
    pub path: PathBuf,
    pub packages: Vec<PackageInventory>,
    /// Package whose ExeFS is the executable set. Executables are never
    /// mixed across packages.
    pub program_package: Option<usize>,
    /// Observations a user or developer must know about this input.
    pub notes: Vec<String>,
}

impl ContentSource {
    /// Opens a NUP file or an extracted dump directory. Accepted directory
    /// layouts are the ones `nup pack` consumes: `exefs/`+`romfs/` at the
    /// root, or `base/`, `updates/<version>/` and `dlc/<title id>/<index>/`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let meta = fs::metadata(path).map_err(io_at(path))?;
        if meta.is_file() {
            return Ok(Self::Nup(Box::new(NupArchive::open(path)?)));
        }
        let packages = scan_directory(path)?;
        if packages.is_empty() {
            return Err(ContentError::Layout(format!(
                "{}: no exefs/ or romfs/ directory found (expected a NUP file or a dump \
                 directory with exefs/ and romfs/, or base/ and updates/<version>/)",
                path.display()
            )));
        }
        Ok(Self::Directory {
            root: path.to_path_buf(),
            packages,
        })
    }

    pub fn inventory(&self) -> Result<ContentInventory> {
        let mut packages = Vec::new();
        let mut notes = Vec::new();
        let (source, path) = match self {
            Self::Nup(a) => {
                for p in a.packages() {
                    let meta = a.package_meta(p)?;
                    let exefs = match p.exefs {
                        Some(d) => nup_modules(a, d)?,
                        None => Vec::new(),
                    };
                    let romfs = p.romfs.map(|d| {
                        let files = a.walk_files(d);
                        RomfsInventory {
                            files: files.len() as u64,
                            bytes: files.iter().map(|(_, f)| a.file(*f).size).sum(),
                            hashes_recorded: true,
                        }
                    });
                    if meta.build_id.is_none() && p.exefs.is_some() {
                        notes.push(format!(
                            "package {}: no build-ID meta record; the build ID is read from \
                             the main NSO header instead",
                            p.index
                        ));
                    }
                    packages.push(PackageInventory {
                        index: p.index,
                        kind: p.kind.into(),
                        title_id: Some(format!("{:016X}", p.title_id)),
                        version: p.version,
                        standalone_update: p.standalone_update,
                        display_version: meta.display_version,
                        title_name: meta.title_name,
                        icon: meta.icon,
                        exefs_key: exefs_key(&exefs),
                        exefs,
                        romfs,
                    });
                }
                ("nup", a.path().to_path_buf())
            }
            Self::Directory { root, packages: ps } => {
                for (index, p) in ps.iter().enumerate() {
                    let exefs = match &p.exefs {
                        Some(d) => dir_modules(d)?,
                        None => Vec::new(),
                    };
                    let npdm_id = exefs
                        .iter()
                        .find_map(|m| m.npdm.as_ref()?.program_id.clone());
                    let romfs = p.romfs.as_deref().map(count_tree).transpose()?;
                    packages.push(PackageInventory {
                        index,
                        kind: p.kind,
                        title_id: p.title_id.map(|t| format!("{t:016X}")).or(npdm_id),
                        version: p.version,
                        standalone_update: false,
                        display_version: None,
                        title_name: None,
                        icon: None,
                        exefs_key: exefs_key(&exefs),
                        exefs,
                        romfs,
                    });
                }
                ("directory", root.clone())
            }
        };

        let program_package = select_program_package(&packages);
        if let Some(i) = program_package {
            let p = &packages[i];
            if p.kind == ContentKind::Update
                && !packages.iter().any(|q| q.kind == ContentKind::Base)
            {
                notes.push(format!(
                    "package {i} is an update supplied without its base; its own RomFS is used \
                     as the complete game tree, which is not verified to be complete"
                ));
            }
            if let Some(nso) = p
                .exefs
                .iter()
                .filter_map(|m| m.nso.as_ref())
                .find(|n| n.unknown_flags != 0)
            {
                notes.push(format!(
                    "ExeFS contains NSO flags {:#x} beyond the classic compression/hash bits \
                     (0x80 marks ZBIC-compressed segments)",
                    nso.unknown_flags
                ));
            }
        }
        for p in &packages {
            let ids: Vec<_> = p
                .exefs
                .iter()
                .filter_map(|m| m.npdm.as_ref()?.program_id.clone())
                .collect();
            if let (Some(tid), Some(npdm)) = (&p.title_id, ids.first())
                && tid != npdm
            {
                notes.push(format!(
                    "package {}: title ID {tid} differs from main.npdm program ID {npdm}",
                    p.index
                ));
            }
        }
        Ok(ContentInventory {
            source,
            path,
            packages,
            program_package,
            notes,
        })
    }

    /// Reads one file from package `package`'s RomFS by its `/`-separated
    /// path inside the RomFS (e.g. `Event/Movie/Dm_Title_0045.webm`).
    /// `Ok(None)` when the package has no RomFS or no such file; files over
    /// `limit` bytes are an error.
    pub fn read_romfs_file(
        &self,
        package: usize,
        path: &str,
        limit: u64,
    ) -> Result<Option<Vec<u8>>> {
        match self {
            Self::Nup(archive) => {
                let Some(romfs) = archive.packages().get(package).and_then(|p| p.romfs) else {
                    return Ok(None);
                };
                match archive.resolve(romfs, path) {
                    Some(Node::File(id)) => Ok(Some(archive.read_file(id, limit)?)),
                    _ => Ok(None),
                }
            }
            Self::Directory { packages, .. } => {
                let Some(romfs) = packages.get(package).and_then(|p| p.romfs.as_ref()) else {
                    return Ok(None);
                };
                if path.split('/').any(|part| part == "..") {
                    return Ok(None);
                }
                let file = romfs.join(path);
                let size = match fs::metadata(&file) {
                    Ok(meta) if meta.is_file() => meta.len(),
                    Ok(_) => return Ok(None),
                    Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
                    Err(error) => return Err(io_at(&file)(error)),
                };
                if size > limit {
                    return Err(ContentError::Layout(format!(
                        "{}: {size} bytes exceeds the {limit}-byte read limit",
                        file.display()
                    )));
                }
                fs::read(&file).map(Some).map_err(io_at(&file))
            }
        }
    }

    /// Ensures the ExeFS of `package` exists as plain files under
    /// `cache_root/exefs/<exefs_key>/`, extracting and verifying it only if
    /// that entry is absent or incomplete. Returns the directory.
    pub fn materialize_exefs(
        &self,
        inventory: &ContentInventory,
        package: usize,
        cache_root: &Path,
    ) -> Result<PathBuf> {
        let inv = inventory
            .packages
            .get(package)
            .ok_or_else(|| ContentError::Layout(format!("no package {package}")))?;
        let key = inv
            .exefs_key
            .as_ref()
            .ok_or_else(|| ContentError::Layout(format!("package {package} has no ExeFS")))?;
        let parent = cache_root.join("exefs");
        let dest = parent.join(key);
        if cached_exefs_matches(&dest, inv) {
            return Ok(dest);
        }
        fs::create_dir_all(&parent).map_err(io_at(&parent))?;
        let staging = parent.join(format!(".staging-{key}-{}", std::process::id()));
        if staging.exists() {
            fs::remove_dir_all(&staging).map_err(io_at(&staging))?;
        }
        let result = self.write_exefs(package, inv, &staging);
        if let Err(e) = result {
            let _ = fs::remove_dir_all(&staging);
            return Err(e);
        }
        if dest.exists() {
            fs::remove_dir_all(&dest).map_err(io_at(&dest))?;
        }
        fs::rename(&staging, &dest).map_err(io_at(&dest))?;
        Ok(dest)
    }

    fn write_exefs(&self, package: usize, inv: &PackageInventory, staging: &Path) -> Result<()> {
        match self {
            Self::Nup(a) => {
                let dir = a.packages()[package]
                    .exefs
                    .ok_or_else(|| ContentError::Layout("package has no ExeFS".into()))?;
                a.extract_dir(dir, staging)?;
            }
            Self::Directory { packages, .. } => {
                let src = packages[package]
                    .exefs
                    .as_ref()
                    .ok_or_else(|| ContentError::Layout("package has no ExeFS".into()))?;
                fs::create_dir(staging).map_err(io_at(staging))?;
                for m in &inv.exefs {
                    let from = src.join(&m.name);
                    let to = staging.join(&m.name);
                    let digest = copy_hashed(&from, &to)?;
                    if digest != m.sha256 {
                        return Err(ContentError::Integrity(format!(
                            "{} changed while it was being imported",
                            from.display()
                        )));
                    }
                }
            }
        }
        let manifest = serde_json::to_vec_pretty(&inv.exefs)
            .map_err(|e| ContentError::Layout(format!("cannot serialize manifest: {e}")))?;
        let path = staging.join(MANIFEST);
        fs::write(&path, manifest).map_err(io_at(&path))
    }
}

/// A cache entry is reused when its manifest names the same modules and every
/// file has the recorded size. Content was hash-verified when it was written.
fn cached_exefs_matches(dest: &Path, inv: &PackageInventory) -> bool {
    let Ok(raw) = fs::read(dest.join(MANIFEST)) else {
        return false;
    };
    let Ok(cached) = serde_json::from_slice::<Vec<serde_json::Value>>(&raw) else {
        return false;
    };
    cached.len() == inv.exefs.len()
        && inv.exefs.iter().zip(&cached).all(|(m, c)| {
            c["name"] == m.name.as_str()
                && c["sha256"] == m.sha256.as_str()
                && fs::symlink_metadata(dest.join(&m.name))
                    .is_ok_and(|md| md.is_file() && md.len() == m.size)
        })
}

fn select_program_package(packages: &[PackageInventory]) -> Option<usize> {
    let with_exefs = |k: ContentKind| {
        packages
            .iter()
            .filter(move |p| p.kind == k && !p.exefs.is_empty())
    };
    with_exefs(ContentKind::Update)
        .max_by_key(|p| p.version)
        .or_else(|| with_exefs(ContentKind::Base).next())
        .map(|p| p.index)
}

fn exefs_key(modules: &[ModuleInventory]) -> Option<String> {
    if modules.is_empty() {
        return None;
    }
    let mut sorted: Vec<_> = modules.iter().collect();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));
    let mut h = Sha256::new();
    for m in sorted {
        h.update(m.name.as_bytes());
        h.update([0]);
        h.update(m.sha256.as_bytes());
        h.update([0]);
    }
    Some(hex::encode(h.finalize()))
}

fn identify(name: &str, head: &[u8]) -> (Option<NsoHeader>, Option<NpdmInfo>) {
    if name.ends_with(".npdm") {
        (None, NpdmInfo::parse(head))
    } else {
        (NsoHeader::parse(head), None)
    }
}

fn nup_modules(a: &NupArchive, dir: echora_nup::DirId) -> Result<Vec<ModuleInventory>> {
    let mut out = Vec::new();
    for child in a.children(dir) {
        let Node::File(f) = child else { continue };
        let entry = a.file(f);
        let name = a.name(child).to_owned();
        // Headers need at most 4 KiB (NPDM ACI0 lies within the first page in
        // practice); the full content hash is verified on materialization.
        let mut head = Vec::new();
        a.open_file(f, false)?
            .take(4096)
            .read_to_end(&mut head)
            .map_err(io_at(a.path()))?;
        let (nso, npdm) = identify(&name, &head);
        out.push(ModuleInventory {
            name,
            size: entry.size,
            sha256: hex::encode(entry.sha256),
            nso,
            npdm,
        });
    }
    Ok(out)
}

fn dir_modules(dir: &Path) -> Result<Vec<ModuleInventory>> {
    let mut entries: Vec<_> = fs::read_dir(dir)
        .map_err(io_at(dir))?
        .collect::<io::Result<_>>()
        .map_err(io_at(dir))?;
    entries.sort_by_key(|e| e.file_name());
    let mut out = Vec::new();
    for e in entries {
        let path = e.path();
        let md = fs::symlink_metadata(&path).map_err(io_at(&path))?;
        if !md.is_file() {
            continue;
        }
        let Some(name) = e.file_name().to_str().map(str::to_owned) else {
            return Err(ContentError::Layout(format!(
                "{}: non-UTF-8 name",
                path.display()
            )));
        };
        if md.len() > MAX_EXEFS_FILE {
            return Err(ContentError::Layout(format!(
                "{}: too large for ExeFS",
                path.display()
            )));
        }
        let data = fs::read(&path).map_err(io_at(&path))?;
        let (nso, npdm) = identify(&name, &data[..data.len().min(4096)]);
        out.push(ModuleInventory {
            name,
            size: data.len() as u64,
            sha256: hex::encode(Sha256::digest(&data)),
            nso,
            npdm,
        });
    }
    Ok(out)
}

/// Counts regular files below `dir` without following symlinks.
fn count_tree(dir: &Path) -> Result<RomfsInventory> {
    let mut inv = RomfsInventory {
        files: 0,
        bytes: 0,
        hashes_recorded: false,
    };
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in fs::read_dir(&d).map_err(io_at(&d))? {
            let e = e.map_err(io_at(&d))?;
            let md = fs::symlink_metadata(e.path()).map_err(io_at(&e.path()))?;
            if md.is_dir() {
                stack.push(e.path());
            } else if md.is_file() {
                inv.files += 1;
                inv.bytes += md.len();
            }
        }
    }
    Ok(inv)
}

fn copy_hashed(from: &Path, to: &Path) -> Result<String> {
    let mut src = fs::File::open(from).map_err(io_at(from))?;
    let mut dst = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(to)
        .map_err(io_at(to))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = src.read(&mut buf).map_err(io_at(from))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
        dst.write_all(&buf[..n]).map_err(io_at(to))?;
    }
    Ok(hex::encode(h.finalize()))
}

fn is_real_dir(p: &Path) -> bool {
    fs::symlink_metadata(p).is_ok_and(|m| m.is_dir())
}

fn content_dirs(root: &Path) -> (Option<PathBuf>, Option<PathBuf>) {
    let exefs = root.join("exefs");
    let romfs = root.join("romfs");
    (
        is_real_dir(&exefs).then_some(exefs),
        is_real_dir(&romfs).then_some(romfs),
    )
}

fn scan_directory(root: &Path) -> Result<Vec<DirPackage>> {
    let mut out = Vec::new();
    let base = root.join("base");
    let updates = root.join("updates");
    let dlc = root.join("dlc");
    if !is_real_dir(&base) && !is_real_dir(&updates) {
        let (exefs, romfs) = content_dirs(root);
        if exefs.is_some() || romfs.is_some() {
            out.push(DirPackage {
                kind: ContentKind::Base,
                title_id: None,
                version: 0,
                exefs,
                romfs,
            });
        }
        return Ok(out);
    }
    if is_real_dir(&base) {
        let (exefs, romfs) = content_dirs(&base);
        out.push(DirPackage {
            kind: ContentKind::Base,
            title_id: None,
            version: 0,
            exefs,
            romfs,
        });
    }
    let mut versions: Vec<_> = sorted_subdirs(&updates)?
        .into_iter()
        .filter_map(|(name, path)| Some((name.parse::<u32>().ok()?, path)))
        .collect();
    versions.sort();
    for (version, path) in versions {
        let (exefs, romfs) = content_dirs(&path);
        if exefs.is_some() || romfs.is_some() {
            out.push(DirPackage {
                kind: ContentKind::Update,
                title_id: None,
                version,
                exefs,
                romfs,
            });
        }
    }
    for (tid, path) in sorted_subdirs(&dlc)? {
        let Ok(title_id) = u64::from_str_radix(&tid, 16) else {
            continue;
        };
        if tid.len() != 16 {
            continue;
        }
        for (idx, sub) in sorted_subdirs(&path)? {
            let (exefs, romfs) = content_dirs(&sub);
            if (exefs.is_some() || romfs.is_some()) && idx.parse::<u32>().is_ok() {
                out.push(DirPackage {
                    kind: ContentKind::Addon,
                    title_id: Some(title_id),
                    version: 0,
                    exefs,
                    romfs,
                });
            }
        }
    }
    Ok(out)
}

fn sorted_subdirs(dir: &Path) -> Result<Vec<(String, PathBuf)>> {
    if !is_real_dir(dir) {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for e in fs::read_dir(dir).map_err(io_at(dir))? {
        let e = e.map_err(io_at(dir))?;
        if let (Some(name), true) = (e.file_name().to_str(), is_real_dir(&e.path())) {
            out.push((name.to_owned(), e.path()));
        }
    }
    out.sort();
    Ok(out)
}
