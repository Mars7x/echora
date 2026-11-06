//! Read-only mounts for explicitly declared local base, update, and DLC content.
//!
//! A selected update supplies one complete ExeFS set and one already-resolved
//! RomFS tree. Each DLC may declare a selected data patch; its one RomFS tree
//! must already include that patch. This module never overlays base files,
//! merges NCA patches, decrypts archives, or infers content identity from
//! filenames. Manifest identities and versions are user declarations, not
//! cryptographically verified CNMT facts.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, Metadata};
use std::io::{ErrorKind, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

pub const MAX_MANIFEST_SIZE: u64 = 1024 * 1024;
pub const MAX_READ_SIZE: usize = 16 * 1024 * 1024;
const MAX_DLC_ENTRIES: usize = 1024;
const MAX_GUEST_PATH_BYTES: usize = 4096;
const MAX_GUEST_PATH_COMPONENTS: usize = 256;

/// DLC mounts are identified by their explicitly declared 16-digit title ID.
/// Main content and every DLC have independent roots; there is no fallback.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ContentMount {
    Main,
    Dlc(String),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: u32,
    application_id: String,
    base: BaseDeclaration,
    selected_update: Option<UpdateDeclaration>,
    #[serde(default)]
    dlc: Vec<DlcDeclaration>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BaseDeclaration {
    exefs: Option<PathBuf>,
    romfs: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct UpdateDeclaration {
    content_version: u32,
    display_version: Option<String>,
    exefs: Option<PathBuf>,
    resolved_romfs: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DlcDeclaration {
    title_id: String,
    content_version: u32,
    label: Option<String>,
    romfs: Option<PathBuf>,
    selected_data_patch: Option<DataPatchDeclaration>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DataPatchDeclaration {
    title_id: String,
    content_version: u32,
}

#[derive(Clone, Debug)]
enum DirectoryState {
    NotDeclared,
    Missing,
    Available(PathBuf),
}

#[derive(Clone, Debug)]
struct ContentDirectory {
    declared: Option<PathBuf>,
    state: DirectoryState,
}

#[derive(Clone, Debug)]
struct BaseContent {
    exefs: ContentDirectory,
    romfs: ContentDirectory,
}

#[derive(Clone, Debug)]
struct UpdateContent {
    content_version: u32,
    display_version: Option<String>,
    exefs: ContentDirectory,
    resolved_romfs: ContentDirectory,
}

#[derive(Clone, Debug)]
struct DlcContent {
    content_version: u32,
    label: Option<String>,
    romfs: ContentDirectory,
    selected_data_patch: Option<SelectedDataPatch>,
}

#[derive(Clone, Debug)]
struct SelectedDataPatch {
    title_id: u64,
    content_version: u32,
}

/// Validated local directory declarations. Availability is observed while the
/// manifest is loaded; every file resolution checks the filesystem again.
#[derive(Clone, Debug)]
pub struct ContentSet {
    manifest_path: PathBuf,
    application_id: u64,
    base: BaseContent,
    selected_update: Option<UpdateContent>,
    dlc: BTreeMap<u64, DlcContent>,
}

fn title_id(text: &str, field: &str) -> Result<u64> {
    if text.len() != 16 || !text.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{field} must contain exactly 16 hexadecimal digits, without a 0x prefix");
    }
    u64::from_str_radix(text, 16).with_context(|| format!("invalid {field}"))
}

fn regular_metadata(path: &Path) -> Result<Metadata> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("inspect regular file {}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        bail!(
            "content must be a regular file, not a symlink or special file: {}",
            path.display()
        );
    }
    Ok(metadata)
}

fn same_file(before: &Metadata, after: &Metadata) -> bool {
    if !before.is_file() || !after.is_file() || before.len() != after.len() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if before.dev() != after.dev() || before.ino() != after.ino() {
            return false;
        }
    }
    true
}

fn read_manifest(path: &Path) -> Result<Vec<u8>> {
    let before = regular_metadata(path)?;
    if before.len() > MAX_MANIFEST_SIZE {
        bail!("content manifest exceeds the 1 MiB limit");
    }
    let mut file =
        File::open(path).with_context(|| format!("open content manifest {}", path.display()))?;
    if !same_file(&before, &file.metadata()?) {
        bail!("content manifest changed while opening");
    }
    let mut data = Vec::with_capacity(before.len() as usize);
    (&mut file)
        .take(MAX_MANIFEST_SIZE + 1)
        .read_to_end(&mut data)?;
    if data.len() as u64 != before.len() || !same_file(&before, &file.metadata()?) {
        bail!("content manifest changed while reading");
    }
    Ok(data)
}

impl ContentDirectory {
    fn load(manifest_dir: &Path, declared: Option<PathBuf>) -> Result<Self> {
        let Some(path) = declared.as_ref() else {
            return Ok(Self {
                declared,
                state: DirectoryState::NotDeclared,
            });
        };
        if path.as_os_str().is_empty() {
            bail!("content directory paths must not be empty; use null for unavailable content");
        }
        let candidate = if path.is_absolute() {
            path.clone()
        } else {
            manifest_dir.join(path)
        };
        // Components remove trailing slash/dot forms that could hide a final symlink.
        let candidate: PathBuf = candidate.components().collect();
        let state = match fs::symlink_metadata(&candidate) {
            Ok(metadata) => {
                if metadata.file_type().is_symlink() || !metadata.is_dir() {
                    bail!(
                        "declared content root must be a regular directory, not a symlink: {}",
                        candidate.display()
                    );
                }
                DirectoryState::Available(candidate.canonicalize().with_context(|| {
                    format!("resolve content directory {}", candidate.display())
                })?)
            }
            Err(error) if error.kind() == ErrorKind::NotFound => DirectoryState::Missing,
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("inspect content directory {}", candidate.display()));
            }
        };
        Ok(Self { declared, state })
    }

    fn available_path(&self) -> Option<&Path> {
        match &self.state {
            DirectoryState::Available(path) => Some(path),
            _ => None,
        }
    }

    fn report(&self) -> Value {
        let declared = self
            .declared
            .as_ref()
            .map(|path| path.to_string_lossy().into_owned());
        match &self.state {
            DirectoryState::Available(path) => json!({
                "available":true, "status":"available", "declared_path":declared,
                "path":path.to_string_lossy(),
            }),
            DirectoryState::NotDeclared | DirectoryState::Missing => json!({
                "available":false, "status":"not_available", "declared_path":declared,
                "path":Value::Null,
                "reason":if matches!(self.state,DirectoryState::NotDeclared) { "not_declared" } else { "directory_missing" },
            }),
        }
    }
}

impl ContentSet {
    /// Load a schema-1 JSON manifest. Relative paths use the manifest's directory;
    /// explicit absolute paths are also accepted. Missing directories are reported
    /// as unavailable, while existing unsafe file kinds and symlink roots fail.
    pub fn load_manifest(path: &Path) -> Result<Self> {
        let bytes = read_manifest(path)?;
        let manifest: Manifest =
            serde_json::from_slice(&bytes).context("parse content manifest")?;
        if manifest.schema != 1 {
            bail!(
                "unsupported content manifest schema {}; expected 1",
                manifest.schema
            );
        }
        let application_id = title_id(&manifest.application_id, "application_id")?;
        if manifest.dlc.len() > MAX_DLC_ENTRIES {
            bail!("content manifest contains too many DLC declarations");
        }
        let manifest_path = path
            .canonicalize()
            .context("resolve content manifest path")?;
        let directory = manifest_path
            .parent()
            .context("content manifest has no parent directory")?;
        let base = BaseContent {
            exefs: ContentDirectory::load(directory, manifest.base.exefs).context("base ExeFS")?,
            romfs: ContentDirectory::load(directory, manifest.base.romfs).context("base RomFS")?,
        };
        let selected_update = manifest
            .selected_update
            .map(|update| -> Result<UpdateContent> {
                Ok(UpdateContent {
                    content_version: update.content_version,
                    display_version: update.display_version,
                    exefs: ContentDirectory::load(directory, update.exefs)
                        .context("selected update ExeFS")?,
                    resolved_romfs: ContentDirectory::load(directory, update.resolved_romfs)
                        .context("selected update resolved RomFS")?,
                })
            })
            .transpose()?;
        let mut dlc = BTreeMap::new();
        let mut selected_data_patch_ids = BTreeSet::new();
        for declaration in manifest.dlc {
            let id = title_id(&declaration.title_id, "DLC title_id")?;
            if dlc.contains_key(&id) {
                bail!("duplicate DLC title_id {id:016x}");
            }
            let selected_data_patch = declaration
                .selected_data_patch
                .map(|patch| -> Result<SelectedDataPatch> {
                    let patch_id = title_id(&patch.title_id, "DLC selected_data_patch title_id")?;
                    if patch_id == id {
                        bail!(
                            "DLC {id:016x} and its selected data patch must have distinct title IDs"
                        );
                    }
                    if !selected_data_patch_ids.insert(patch_id) {
                        bail!("duplicate selected data patch title_id {patch_id:016x}");
                    }
                    Ok(SelectedDataPatch {
                        title_id: patch_id,
                        content_version: patch.content_version,
                    })
                })
                .transpose()?;
            dlc.insert(
                id,
                DlcContent {
                    content_version: declaration.content_version,
                    label: declaration.label,
                    romfs: ContentDirectory::load(directory, declaration.romfs)
                        .with_context(|| format!("DLC {id:016x} RomFS"))?,
                    selected_data_patch,
                },
            );
        }
        Ok(Self {
            manifest_path,
            application_id,
            base,
            selected_update,
            dlc,
        })
    }

    /// The declared parent application ID, for comparison with the selected NPDM.
    /// Matching those values does not authenticate CNMT or DLC relationships.
    pub fn application_id(&self) -> u64 {
        self.application_id
    }

    /// Existing roots referenced by this manifest, including unselected base
    /// content, for protecting every supplied content tree from output writes.
    pub fn available_roots(&self) -> Vec<&Path> {
        let mut roots = Vec::new();
        for directory in [&self.base.exefs, &self.base.romfs] {
            if let Some(path) = directory.available_path() {
                roots.push(path);
            }
        }
        if let Some(update) = &self.selected_update {
            for directory in [&update.exefs, &update.resolved_romfs] {
                if let Some(path) = directory.available_path() {
                    roots.push(path);
                }
            }
        }
        for dlc in self.dlc.values() {
            if let Some(path) = dlc.romfs.available_path() {
                roots.push(path);
            }
        }
        roots
    }

    /// A selected update supplies the entire ExeFS set. If it is unavailable,
    /// return None; never silently mix in or fall back to the base executable set.
    pub fn selected_exefs(&self) -> Option<&Path> {
        match &self.selected_update {
            Some(update) => update.exefs.available_path(),
            None => self.base.exefs.available_path(),
        }
    }

    fn selected_romfs(&self) -> &ContentDirectory {
        match &self.selected_update {
            Some(update) => &update.resolved_romfs,
            None => &self.base.romfs,
        }
    }

    /// Describe available declared directories without claiming completeness,
    /// publisher authenticity, an inferred display version, or all installed DLC.
    pub fn available_content_report(&self) -> Value {
        let selected_exefs = match &self.selected_update {
            Some(update) => &update.exefs,
            None => &self.base.exefs,
        };
        let update = self.selected_update.as_ref().map(|update| {
            json!({
                "content_version":update.content_version, "display_version":update.display_version,
                "exefs":update.exefs.report(), "resolved_romfs":update.resolved_romfs.report(),
                "romfs_requirement":"complete_already_resolved_tree",
            })
        });
        let dlc: Vec<_> = self.dlc.iter().map(|(id, content)| {
            let selected_data_patch = content.selected_data_patch.as_ref().map(|patch| json!({
                "title_id":format!("{:016x}",patch.title_id), "content_version":patch.content_version,
            }));
            json!({
                "title_id":format!("{id:016x}"), "content_version":content.content_version,
                "label":content.label, "mount":format!("dlc:{id:016x}"), "romfs":content.romfs.report(),
                "selected_data_patch":selected_data_patch,
                "romfs_requirement":"complete_already_resolved_tree",
                "romfs_selection":if content.selected_data_patch.is_some() { "add_on_with_selected_data_patch" } else { "add_on" },
            })
        }).collect();
        json!({
            "schema":1, "manifest":self.manifest_path.to_string_lossy(),
            "application_id":format!("{:016x}",self.application_id),
            "content_identity":"user_declared_not_cnmt_verified",
            "selection":if self.selected_update.is_some() { "update" } else { "base" },
            "base":{"exefs":self.base.exefs.report(),"romfs":self.base.romfs.report()},
            "selected_update":update,
            "selected":{"exefs":selected_exefs.report(),"main_romfs":self.selected_romfs().report()},
            "dlc":dlc, "declared_dlc_count":self.dlc.len(),
            "available_dlc_count":self.dlc.values().filter(|content|content.romfs.available_path().is_some()).count(),
            "dlc_inventory_completeness":"not_verified",
            "romfs_completeness_verified":false,
            "availability_definition":"declared directory existed when loaded; file reads revalidate paths",
            "patch_merging":"not_performed", "dlc_overlay":"not_performed",
            "maximum_read_bytes":MAX_READ_SIZE,
        })
    }

    fn mount_root(&self, mount: &ContentMount) -> Result<&Path> {
        match mount {
            ContentMount::Main => self
                .selected_romfs()
                .available_path()
                .context("main RomFS is not available for the selected content set"),
            ContentMount::Dlc(text) => {
                let id = title_id(text, "DLC mount title_id")?;
                self.dlc
                    .get(&id)
                    .with_context(|| format!("DLC {id:016x} is not declared"))?
                    .romfs
                    .available_path()
                    .with_context(|| format!("DLC {id:016x} RomFS is not available"))
            }
        }
    }

    /// Resolve one regular file within its mount. Guest paths are POSIX paths;
    /// a leading slash means the mount root, never the host root. Traversal,
    /// Windows drive/stream syntax, symlinks, and nonregular files are rejected.
    pub fn resolve(&self, mount: &ContentMount, guest_path: &str) -> Result<PathBuf> {
        if guest_path.len() > MAX_GUEST_PATH_BYTES || guest_path.contains(['\\', ':', '\0']) {
            bail!("guest path is too long or contains unsupported path syntax");
        }
        let parts: Vec<_> = guest_path
            .split('/')
            .filter(|part| !part.is_empty())
            .collect();
        if parts.is_empty()
            || parts.len() > MAX_GUEST_PATH_COMPONENTS
            || parts.iter().any(|part| *part == "." || *part == "..")
        {
            bail!("guest path must name a file without dot or parent traversal components");
        }
        let root = self.mount_root(mount)?;
        let metadata = fs::symlink_metadata(root).context("inspect mounted content root")?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() || root.canonicalize()? != root {
            bail!("mounted content root changed or is no longer a regular directory");
        }
        let mut path = root.to_path_buf();
        for (index, part) in parts.iter().enumerate() {
            path.push(part);
            let metadata = fs::symlink_metadata(&path)
                .with_context(|| format!("resolve guest content file {guest_path:?}"))?;
            if metadata.file_type().is_symlink() {
                bail!("symlinks are not permitted in guest content paths");
            }
            if index + 1 == parts.len() {
                if !metadata.is_file() {
                    bail!("guest content path is not a regular file");
                }
            } else if !metadata.is_dir() {
                bail!("guest content path has a non-directory component");
            }
        }
        let resolved = path.canonicalize().context("resolve guest file location")?;
        if !resolved.starts_with(root) {
            bail!("guest content path escaped its mount");
        }
        Ok(resolved)
    }

    /// Read an exact bounded range. EOF is not silently padded or truncated;
    /// offset + length must fit the file, and a single read is at most 16 MiB.
    pub fn read_range(
        &self,
        mount: &ContentMount,
        guest_path: &str,
        offset: u64,
        length: usize,
    ) -> Result<Vec<u8>> {
        if length > MAX_READ_SIZE {
            bail!("content read exceeds the 16 MiB per-call limit");
        }
        let end = offset
            .checked_add(length as u64)
            .context("content read range overflows")?;
        let path = self.resolve(mount, guest_path)?;
        let before = regular_metadata(&path)?;
        let mut file = File::open(&path).context("open mounted content file")?;
        let opened = file.metadata()?;
        if !same_file(&before, &opened) {
            bail!("content file changed while opening");
        }
        // Rewalk after open so a changed parent cannot silently redirect the read.
        if self.resolve(mount, guest_path)? != path
            || !same_file(&opened, &regular_metadata(&path)?)
        {
            bail!("content path changed while opening");
        }
        if end > opened.len() {
            bail!("content read range exceeds file size {}", opened.len());
        }
        file.seek(SeekFrom::Start(offset))?;
        let mut bytes = vec![0; length];
        file.read_exact(&mut bytes)
            .context("read mounted content range")?;
        let after = file.metadata()?;
        if !same_file(&opened, &after) || opened.modified().ok() != after.modified().ok() {
            bail!("content file changed while reading");
        }
        Ok(bytes)
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    static ID: AtomicU64 = AtomicU64::new(0);
    const APP: &str = "0400a3a01ee4c000";
    // Synthetic add-on identity, not a claim about any real DLC title ID.
    const DLC: &str = "040000000000f00a";
    const DATA_PATCH: &str = "040000000000ba5e";

    struct Fixture {
        root: PathBuf,
    }
    impl Fixture {
        fn new() -> Self {
            let id = ID.fetch_add(1, Ordering::Relaxed);
            let root = std::env::temp_dir()
                .join(format!("echora-content-test-{}-{id}", std::process::id()));
            fs::create_dir(&root).unwrap();
            Self { root }
        }
        fn directory(&self, name: &str) -> PathBuf {
            let path = self.root.join(name);
            fs::create_dir_all(&path).unwrap();
            path
        }
        fn load(&self, value: Value) -> Result<ContentSet> {
            let path = self.root.join("content.json");
            fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
            ContentSet::load_manifest(&path)
        }
        fn basic(&self) -> Value {
            json!({"schema":1,"application_id":APP,"base":{"exefs":null,"romfs":null},"dlc":[]})
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn selected_update_is_one_set_and_dlc_has_an_isolated_namespace() {
        let f = Fixture::new();
        let base_exefs = f.directory("base/exefs");
        let update_exefs = f.directory("update/exefs");
        let base = f.directory("base/romfs");
        let update = f.directory("update/romfs");
        let dlc = f.directory("dlc/romfs");
        fs::write(base_exefs.join("main"), "base module").unwrap();
        fs::write(update_exefs.join("main"), "update module").unwrap();
        fs::write(base.join("same.bin"), "base").unwrap();
        fs::write(base.join("base-only.bin"), "not an overlay").unwrap();
        fs::write(update.join("same.bin"), "update").unwrap();
        fs::write(dlc.join("same.bin"), "dlc").unwrap();
        fs::write(dlc.join("dlc-only.bin"), "isolated").unwrap();
        let set = f.load(json!({
            "schema":1,"application_id":APP,"base":{"exefs":"base/exefs","romfs":"base/romfs"},
            "selected_update":{"content_version":327680,"exefs":"update/exefs","resolved_romfs":"update/romfs"},
            "dlc":[{"title_id":DLC,"content_version":0,"romfs":"dlc/romfs"}]
        })).unwrap();
        assert_eq!(set.application_id(), 0x0400a3a01ee4c000);
        assert_eq!(
            set.selected_exefs(),
            Some(update_exefs.canonicalize().unwrap().as_path())
        );
        assert_eq!(
            set.read_range(&ContentMount::Main, "/same.bin", 0, 6)
                .unwrap(),
            b"update"
        );
        assert_eq!(
            set.read_range(&ContentMount::Dlc(DLC.to_uppercase()), "same.bin", 0, 3)
                .unwrap(),
            b"dlc"
        );
        assert!(set.resolve(&ContentMount::Main, "base-only.bin").is_err());
        assert!(set.resolve(&ContentMount::Main, "dlc-only.bin").is_err());
        let report = set.available_content_report();
        assert!(report["selected_update"]["display_version"].is_null());
        assert_eq!(report["available_dlc_count"], 1);
        assert_eq!(report["dlc_inventory_completeness"], "not_verified");
    }

    #[test]
    fn missing_update_and_dlc_are_unavailable_without_base_fallback() {
        let f = Fixture::new();
        f.directory("base/exefs");
        fs::write(f.directory("base/romfs").join("base.bin"), "base").unwrap();
        let set = f.load(json!({
            "schema":1,"application_id":APP,"base":{"exefs":"base/exefs","romfs":"base/romfs"},
            "selected_update":{"content_version":1,"exefs":"missing-update","resolved_romfs":null},
            "dlc":[{"title_id":DLC,"content_version":0,"romfs":"missing-dlc"}]
        })).unwrap();
        assert!(set.selected_exefs().is_none());
        assert!(set.resolve(&ContentMount::Main, "base.bin").is_err());
        assert!(
            set.resolve(&ContentMount::Dlc(DLC.into()), "base.bin")
                .is_err()
        );
        let report = set.available_content_report();
        assert_eq!(report["selected"]["exefs"]["reason"], "directory_missing");
        assert_eq!(report["selected"]["main_romfs"]["reason"], "not_declared");
        assert_eq!(report["declared_dlc_count"], 1);
        assert_eq!(report["available_dlc_count"], 0);
    }

    #[test]
    fn exefs_only_content_does_not_claim_romfs_or_all_dlc() {
        let f = Fixture::new();
        let exefs = f.directory("exefs");
        let mut value = f.basic();
        value["base"]["exefs"] = json!("exefs");
        let set = f.load(value).unwrap();
        assert_eq!(
            set.selected_exefs(),
            Some(exefs.canonicalize().unwrap().as_path())
        );
        let report = set.available_content_report();
        assert_eq!(report["selected"]["main_romfs"]["available"], false);
        assert_eq!(report["available_dlc_count"], 0);
        assert_eq!(report["dlc_inventory_completeness"], "not_verified");
        assert_eq!(
            report["content_identity"],
            "user_declared_not_cnmt_verified"
        );
    }

    #[test]
    fn identities_versions_and_schema_are_explicit_and_strict() {
        let f = Fixture::new();
        let mut duplicate = f.basic();
        duplicate["dlc"] = json!([
            {"title_id":DLC,"content_version":0},
            {"title_id":DLC.to_uppercase(),"content_version":1}
        ]);
        assert!(
            f.load(duplicate)
                .unwrap_err()
                .to_string()
                .contains("duplicate")
        );
        for invalid in [json!("327680"), json!(-1), json!(1.5), json!(4294967296u64)] {
            let mut value = f.basic();
            value["selected_update"] = json!({"content_version":invalid});
            assert!(f.load(value).is_err());
        }
        for invalid in ["0x0400a3a01ee4c000", "0400a3a01ee4c00z", "123"] {
            let mut value = f.basic();
            value["application_id"] = json!(invalid);
            assert!(f.load(value).is_err());
        }
        let mut wrong_schema = f.basic();
        wrong_schema["schema"] = json!(2);
        assert!(f.load(wrong_schema).is_err());
        let mut unknown_field = f.basic();
        unknown_field["update"] = json!({"content_version":1});
        assert!(f.load(unknown_field).is_err());
    }

    #[test]
    fn selected_data_patch_keeps_its_identity_and_uses_the_single_dlc_tree() {
        let f = Fixture::new();
        fs::write(f.directory("dlc/romfs").join("asset.bin"), b"resolved").unwrap();
        let mut value = f.basic();
        value["dlc"] = json!([{
            "title_id":DLC,"content_version":3,"romfs":"dlc/romfs",
            "selected_data_patch":{"title_id":DATA_PATCH.to_uppercase(),"content_version":65536}
        }]);
        let set = f.load(value.clone()).unwrap();
        assert_eq!(
            set.read_range(&ContentMount::Dlc(DLC.into()), "asset.bin", 0, 8)
                .unwrap(),
            b"resolved"
        );
        assert!(
            set.resolve(&ContentMount::Dlc(DATA_PATCH.into()), "asset.bin")
                .is_err()
        );
        let report = set.available_content_report();
        assert_eq!(report["dlc"][0]["content_version"], 3);
        assert_eq!(
            report["dlc"][0]["selected_data_patch"]["title_id"],
            DATA_PATCH
        );
        assert_eq!(
            report["dlc"][0]["selected_data_patch"]["content_version"],
            65536
        );
        assert_eq!(
            report["dlc"][0]["romfs_selection"],
            "add_on_with_selected_data_patch"
        );
        assert_eq!(report["available_dlc_count"], 1);
        assert_eq!(report["patch_merging"], "not_performed");
        assert_eq!(
            report["content_identity"],
            "user_declared_not_cnmt_verified"
        );

        // A declaration alone cannot make the combined DLC tree available.
        value["dlc"][0]["romfs"] = Value::Null;
        let report = f.load(value).unwrap().available_content_report();
        assert_eq!(report["available_dlc_count"], 0);
        assert_eq!(
            report["dlc"][0]["selected_data_patch"]["title_id"],
            DATA_PATCH
        );
        assert_eq!(report["dlc"][0]["romfs"]["available"], false);
    }

    #[test]
    fn selected_data_patch_rejects_invalid_duplicate_and_parent_ids() {
        let f = Fixture::new();
        let mut valid = f.basic();
        valid["dlc"] = json!([{
            "title_id":DLC,"content_version":0,
            "selected_data_patch":{"title_id":DATA_PATCH,"content_version":1}
        }]);
        for invalid in ["0x040000000000ba5e", "040000000000ba5z", "123", DLC] {
            let mut value = valid.clone();
            value["dlc"][0]["selected_data_patch"]["title_id"] = json!(invalid.to_uppercase());
            assert!(f.load(value).is_err(), "accepted patch ID {invalid}");
        }
        for invalid in [json!("1"), json!(-1), json!(1.5), json!(4294967296u64)] {
            let mut value = valid.clone();
            value["dlc"][0]["selected_data_patch"]["content_version"] = invalid;
            assert!(f.load(value).is_err());
        }
        let mut duplicate = valid.clone();
        duplicate["dlc"].as_array_mut().unwrap().push(json!({
            "title_id":"040000000000f00b","content_version":0,
            "selected_data_patch":{"title_id":DATA_PATCH.to_uppercase(),"content_version":2}
        }));
        assert!(
            f.load(duplicate)
                .unwrap_err()
                .to_string()
                .contains("duplicate selected data patch")
        );
        // A patch supplies identity only, never a separate overlay directory.
        valid["dlc"][0]["selected_data_patch"]["romfs"] = json!("patch-tree");
        assert!(f.load(valid).is_err());
    }

    #[test]
    fn guest_paths_cannot_traverse_or_use_host_path_syntax() {
        let f = Fixture::new();
        fs::write(f.directory("romfs").join("asset.bin"), b"asset").unwrap();
        fs::write(f.root.join("outside.bin"), b"outside").unwrap();
        let mut value = f.basic();
        value["base"]["romfs"] = json!("romfs");
        let set = f.load(value).unwrap();
        for path in [
            "../outside.bin",
            "sub/../../outside.bin",
            "..\\outside.bin",
            "C:/outside.bin",
            "/asset.bin:stream",
            "./asset.bin",
            "",
            "/",
        ] {
            assert!(set.resolve(&ContentMount::Main, path).is_err(), "{path}");
        }
        assert_eq!(
            set.read_range(&ContentMount::Main, "/asset.bin", 0, 5)
                .unwrap(),
            b"asset"
        );
        assert!(
            set.resolve(&ContentMount::Main, &"x".repeat(MAX_GUEST_PATH_BYTES + 1))
                .is_err()
        );
    }

    #[test]
    fn reads_are_exact_bounded_and_require_regular_files() {
        let f = Fixture::new();
        let root = f.directory("romfs");
        fs::write(root.join("asset.bin"), b"abcdef").unwrap();
        fs::create_dir(root.join("directory")).unwrap();
        let mut value = f.basic();
        value["base"]["romfs"] = json!("romfs");
        let set = f.load(value).unwrap();
        assert_eq!(
            set.read_range(&ContentMount::Main, "asset.bin", 2, 3)
                .unwrap(),
            b"cde"
        );
        assert!(
            set.read_range(&ContentMount::Main, "asset.bin", 6, 0)
                .unwrap()
                .is_empty()
        );
        for (offset, length) in [(7, 0), (5, 2), (u64::MAX, 1), (0, MAX_READ_SIZE + 1)] {
            assert!(
                set.read_range(&ContentMount::Main, "asset.bin", offset, length)
                    .is_err()
            );
        }
        assert!(
            set.read_range(&ContentMount::Main, "directory", 0, 0)
                .is_err()
        );
        let mut bad_root = f.basic();
        bad_root["base"]["romfs"] = json!("romfs/asset.bin");
        assert!(f.load(bad_root).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn symlink_files_directories_and_declared_roots_are_rejected() {
        use std::os::unix::fs::symlink;
        let f = Fixture::new();
        let root = f.directory("romfs");
        let outside = f.directory("outside");
        fs::write(outside.join("secret.bin"), b"outside").unwrap();
        symlink(outside.join("secret.bin"), root.join("link.bin")).unwrap();
        symlink(&outside, root.join("escape")).unwrap();
        let mut value = f.basic();
        value["base"]["romfs"] = json!("romfs");
        let set = f.load(value).unwrap();
        assert!(
            set.read_range(&ContentMount::Main, "link.bin", 0, 7)
                .is_err()
        );
        assert!(
            set.read_range(&ContentMount::Main, "escape/secret.bin", 0, 7)
                .is_err()
        );
        symlink(&root, f.root.join("root-alias")).unwrap();
        let mut bad_root = f.basic();
        bad_root["base"]["romfs"] = json!("root-alias/.");
        assert!(f.load(bad_root).is_err());
        symlink(
            f.root.join("content.json"),
            f.root.join("manifest-alias.json"),
        )
        .unwrap();
        assert!(ContentSet::load_manifest(&f.root.join("manifest-alias.json")).is_err());
    }
}
