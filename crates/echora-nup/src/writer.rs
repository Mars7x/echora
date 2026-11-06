// Minimal NUP v1 writer for synthetic test archives. It follows SPEC.md's
// producer rules: sorted children, 16-byte aligned extents, shared extents
// for identical content, and a content-derived UUID.

use std::collections::{BTreeMap, HashMap};

use sha2::{Digest, Sha256};

pub enum Entry {
    Dir(BTreeMap<String, Entry>),
    File(Vec<u8>),
}

pub struct PackageSpec {
    pub title_id: u64,
    pub kind: u8,
    pub version: u32,
    pub flags: u8,
    pub files: Vec<(&'static str, Vec<u8>)>,
    pub meta: Vec<(u16, Vec<u8>)>,
}

impl PackageSpec {
    pub fn base(files: Vec<(&'static str, Vec<u8>)>) -> Self {
        Self {
            title_id: 0x0400_0000_1234_0000,
            kind: 0,
            version: 0x10000,
            flags: 0,
            files,
            meta: Vec::new(),
        }
    }
}

fn tree(files: &[(&str, Vec<u8>)]) -> BTreeMap<String, Entry> {
    let mut root = BTreeMap::new();
    for (path, data) in files {
        let parts: Vec<&str> = path.split('/').collect();
        let mut cur = &mut root;
        for dir in &parts[..parts.len() - 1] {
            let e = cur
                .entry((*dir).to_owned())
                .or_insert_with(|| Entry::Dir(BTreeMap::new()));
            let Entry::Dir(m) = e else {
                panic!("file/dir clash")
            };
            cur = m;
        }
        cur.insert(parts[parts.len() - 1].to_owned(), Entry::File(data.clone()));
    }
    root
}

enum Child {
    Dir(usize),
    File(usize),
}

struct DirOut {
    name: String,
    children: Vec<Child>,
}

struct FileOut {
    name: String,
    data: Vec<u8>,
}

pub fn build(packages: &[PackageSpec], compress: bool) -> Vec<u8> {
    let mut dirs: Vec<DirOut> = Vec::new();
    let mut files: Vec<FileOut> = Vec::new();
    let mut roots = Vec::new();
    for p in packages {
        let root_tree = tree(&p.files);
        let root = dirs.len();
        roots.push(root);
        dirs.push(DirOut {
            name: format!("pkg{}", roots.len() - 1),
            children: Vec::new(),
        });
        let mut queue = vec![(root, root_tree)];
        while !queue.is_empty() {
            let (idx, map) = queue.remove(0);
            for (name, entry) in map {
                match entry {
                    Entry::Dir(sub) => {
                        let d = dirs.len();
                        dirs.push(DirOut {
                            name,
                            children: Vec::new(),
                        });
                        dirs[idx].children.push(Child::Dir(d));
                        queue.push((d, sub));
                    }
                    Entry::File(data) => {
                        files.push(FileOut { name, data });
                        dirs[idx].children.push(Child::File(files.len() - 1));
                    }
                }
            }
        }
    }

    let mut pool = Vec::new();
    let mut intern = |s: &str| {
        let off = pool.len() as u32;
        pool.extend_from_slice(s.as_bytes());
        (off, s.len() as u32)
    };
    let dir_names: Vec<_> = dirs.iter().map(|d| intern(&d.name)).collect();
    let file_names: Vec<_> = files.iter().map(|f| intern(&f.name)).collect();

    let dir_count = dirs.len();
    let mut children = Vec::new();
    let mut first_child = Vec::new();
    for d in &dirs {
        first_child.push(children.len() as u32);
        for c in &d.children {
            children.push(match c {
                Child::Dir(i) => *i as u32,
                Child::File(i) => (dir_count + i) as u32,
            });
        }
    }

    let records_end = 0x30 + packages.len() * 0x40 + dirs.len() * 0x18 + files.len() * 0x50;
    let pool_off = records_end + children.len() * 4;
    let toc_size = pool_off + pool.len();
    let toc_offset = 0x80u64;
    let blob_offset = align16(toc_offset + toc_size as u64);

    // Blob: extents first, then meta blocks.
    let mut blob: Vec<u8> = Vec::new();
    let mut extents: HashMap<[u8; 32], (u64, u64, u8)> = HashMap::new();
    let mut file_ext = Vec::new();
    for f in &files {
        let sha: [u8; 32] = Sha256::digest(&f.data).into();
        if f.data.is_empty() {
            file_ext.push((sha, 0u64, 0u64, 0u8));
            continue;
        }
        let (off, size, codec) = *extents.entry(sha).or_insert_with(|| {
            let (bytes, codec) = if compress {
                let c = zstd::bulk::compress(&f.data, 3).expect("zstd");
                if c.len() < f.data.len() {
                    (c, 1)
                } else {
                    (f.data.clone(), 0)
                }
            } else {
                (f.data.clone(), 0)
            };
            while !(blob_offset + blob.len() as u64).is_multiple_of(16) {
                blob.push(0);
            }
            let off = blob_offset + blob.len() as u64;
            blob.extend_from_slice(&bytes);
            (off, bytes.len() as u64, codec)
        });
        file_ext.push((sha, off, size, codec));
    }
    let mut meta_locs = Vec::new();
    for p in packages {
        if p.meta.is_empty() {
            meta_locs.push((0u64, 0u32));
            continue;
        }
        let off = blob_offset + blob.len() as u64;
        let start = blob.len();
        for (ty, payload) in &p.meta {
            blob.extend_from_slice(&ty.to_le_bytes());
            blob.extend_from_slice(&0u16.to_le_bytes());
            blob.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            blob.extend_from_slice(payload);
        }
        meta_locs.push((off, (blob.len() - start) as u32));
    }

    let mut toc = Vec::with_capacity(toc_size);
    toc.extend_from_slice(b"TOC\x01");
    toc.extend_from_slice(&(packages.len() as u32).to_le_bytes());
    toc.extend_from_slice(&(dirs.len() as u64).to_le_bytes());
    toc.extend_from_slice(&(files.len() as u64).to_le_bytes());
    toc.extend_from_slice(&(pool_off as u64).to_le_bytes());
    toc.extend_from_slice(&(pool.len() as u64).to_le_bytes());
    toc.extend_from_slice(&0u64.to_le_bytes());
    for (i, p) in packages.iter().enumerate() {
        let root = roots[i];
        let find = |name: &str| {
            dirs[root]
                .children
                .iter()
                .find_map(|c| match c {
                    Child::Dir(d) if dirs[*d].name == name => Some(*d as u32),
                    _ => None,
                })
                .unwrap_or(u32::MAX)
        };
        toc.extend_from_slice(&p.title_id.to_le_bytes());
        toc.extend_from_slice(&(root as u32).to_le_bytes());
        toc.extend_from_slice(&find("exefs").to_le_bytes());
        toc.extend_from_slice(&find("romfs").to_le_bytes());
        toc.extend_from_slice(&p.version.to_le_bytes());
        toc.push(p.kind);
        toc.push(p.flags);
        toc.extend_from_slice(&[0; 6]);
        toc.extend_from_slice(&meta_locs[i].0.to_le_bytes());
        toc.extend_from_slice(&meta_locs[i].1.to_le_bytes());
        toc.extend_from_slice(&[0; 20]);
    }
    for (i, d) in dirs.iter().enumerate() {
        toc.extend_from_slice(&dir_names[i].0.to_le_bytes());
        toc.extend_from_slice(&dir_names[i].1.to_le_bytes());
        toc.extend_from_slice(&first_child[i].to_le_bytes());
        toc.extend_from_slice(&(d.children.len() as u32).to_le_bytes());
        toc.extend_from_slice(&0u64.to_le_bytes());
    }
    for (i, f) in files.iter().enumerate() {
        let (sha, off, size, codec) = file_ext[i];
        toc.extend_from_slice(&file_names[i].0.to_le_bytes());
        toc.extend_from_slice(&file_names[i].1.to_le_bytes());
        toc.extend_from_slice(&(f.data.len() as u64).to_le_bytes());
        toc.extend_from_slice(&off.to_le_bytes());
        toc.extend_from_slice(&size.to_le_bytes());
        toc.extend_from_slice(&sha);
        toc.push(codec);
        toc.extend_from_slice(&[0; 15]);
    }
    for c in &children {
        toc.extend_from_slice(&c.to_le_bytes());
    }
    toc.extend_from_slice(&pool);
    assert_eq!(toc.len(), toc_size);

    let file_size = blob_offset + blob.len() as u64;
    let mut out = Vec::with_capacity(file_size as usize);
    out.extend_from_slice(b"NUP\x01");
    out.extend_from_slice(&0x80u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&toc_offset.to_le_bytes());
    out.extend_from_slice(&(toc_size as u64).to_le_bytes());
    out.extend_from_slice(&Sha256::digest(&toc));
    out.extend_from_slice(&blob_offset.to_le_bytes());
    out.extend_from_slice(&file_size.to_le_bytes());
    out.extend_from_slice(&(packages.len() as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    let mut h = Sha256::new();
    for p in packages {
        h.update(p.title_id.to_le_bytes());
        h.update(p.version.to_le_bytes());
        h.update([p.kind]);
    }
    out.extend_from_slice(&h.finalize()[..16]);
    out.extend_from_slice(&[0; 32]);
    out.extend_from_slice(&toc);
    out.resize(blob_offset as usize, 0);
    out.extend_from_slice(&blob);
    out
}

fn align16(v: u64) -> u64 {
    (v + 15) & !15
}

/// Recomputes the superblock's TOC hash after a test edits TOC bytes.
pub fn rehash_toc(bytes: &mut [u8]) {
    let toc_size = u64::from_le_bytes(bytes[0x10..0x18].try_into().unwrap()) as usize;
    let digest = Sha256::digest(&bytes[0x80..0x80 + toc_size]);
    bytes[0x18..0x38].copy_from_slice(&digest);
}

/// Offset of file record `index` within the archive bytes.
pub fn file_record_offset(bytes: &[u8], index: usize) -> usize {
    let toc = &bytes[0x80..];
    let pkgs = u32::from_le_bytes(toc[4..8].try_into().unwrap()) as usize;
    let dirs = u64::from_le_bytes(toc[8..16].try_into().unwrap()) as usize;
    0x80 + 0x30 + pkgs * 0x40 + dirs * 0x18 + index * 0x50
}

pub fn write_temp(bytes: &[u8]) -> tempfile::NamedTempFile {
    use std::io::Write;
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(bytes).unwrap();
    f.flush().unwrap();
    f
}
