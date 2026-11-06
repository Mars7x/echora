#![allow(clippy::unwrap_used)]

use std::fs;
use std::path::Path;

use echora_content::{ContentKind, ContentSource};

fn fake_nso(tag: u8) -> Vec<u8> {
    let mut h = vec![0u8; 0x180];
    h[..4].copy_from_slice(b"NSO0");
    h[0x40] = tag;
    h[0x41] = 0xab;
    h
}

fn fake_npdm(program_id: u64) -> Vec<u8> {
    let mut b = vec![0u8; 0x100];
    b[..4].copy_from_slice(b"META");
    b[0x70..0x74].copy_from_slice(&0x80u32.to_le_bytes());
    b[0x74..0x78].copy_from_slice(&0x40u32.to_le_bytes());
    b[0x80..0x84].copy_from_slice(b"ACI0");
    b[0x90..0x98].copy_from_slice(&program_id.to_le_bytes());
    b
}

fn write(path: &Path, data: &[u8]) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, data).unwrap();
}

#[test]
fn flat_dump_directory_inventory() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write(&root.join("exefs/main"), &fake_nso(0xe0));
    write(
        &root.join("exefs/main.npdm"),
        &fake_npdm(0x0400_4860_24DE_C000),
    );
    write(&root.join("romfs/System/a.bin"), b"abc");
    write(&root.join("romfs/b.bin"), b"de");

    let inv = ContentSource::open(root).unwrap().inventory().unwrap();
    assert_eq!(inv.source, "directory");
    assert_eq!(inv.packages.len(), 1);
    let p = &inv.packages[0];
    assert_eq!(p.kind, ContentKind::Base);
    assert_eq!(p.title_id.as_deref(), Some("0400486024DEC000"));
    assert_eq!(p.main_build_id().as_deref(), Some("e0ab000000000000"));
    let romfs = p.romfs.as_ref().unwrap();
    assert_eq!((romfs.files, romfs.bytes), (2, 5));
    assert_eq!(inv.program_package, Some(0));
}

#[test]
fn layered_dump_selects_newest_update_executables() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write(&root.join("base/exefs/main"), &fake_nso(1));
    write(&root.join("base/romfs/x"), b"x");
    write(&root.join("updates/65536/exefs/main"), &fake_nso(2));
    write(&root.join("updates/131072/exefs/main"), &fake_nso(3));
    write(&root.join("updates/notaversion/exefs/main"), &fake_nso(9));
    write(&root.join("dlc/0400486024DEC001/0/romfs/d"), b"d");

    let inv = ContentSource::open(root).unwrap().inventory().unwrap();
    let kinds: Vec<_> = inv.packages.iter().map(|p| (p.kind, p.version)).collect();
    assert_eq!(
        kinds,
        [
            (ContentKind::Base, 0),
            (ContentKind::Update, 65536),
            (ContentKind::Update, 131072),
            (ContentKind::Addon, 0),
        ]
    );
    let prog = &inv.packages[inv.program_package.unwrap()];
    assert_eq!(prog.version, 131072);
    assert_eq!(
        inv.packages[3].title_id.as_deref(),
        Some("0400486024DEC001")
    );
}

#[test]
fn exefs_materialization_is_content_addressed_and_reused() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("dump");
    write(&root.join("exefs/main"), &fake_nso(5));
    write(&root.join("exefs/rtld"), &fake_nso(6));
    let cache = dir.path().join("cache");

    let src = ContentSource::open(&root).unwrap();
    let inv = src.inventory().unwrap();
    let out = src.materialize_exefs(&inv, 0, &cache).unwrap();
    assert!(out.ends_with(inv.packages[0].exefs_key.as_ref().unwrap()));
    assert_eq!(fs::read(out.join("main")).unwrap(), fake_nso(5));
    let stamp = fs::metadata(out.join("main")).unwrap().modified().unwrap();

    // Unchanged content reuses the entry without rewriting it.
    let again = src.materialize_exefs(&inv, 0, &cache).unwrap();
    assert_eq!(again, out);
    assert_eq!(
        fs::metadata(out.join("main")).unwrap().modified().unwrap(),
        stamp
    );

    // A damaged entry is rebuilt.
    fs::write(out.join("rtld"), b"short").unwrap();
    src.materialize_exefs(&inv, 0, &cache).unwrap();
    assert_eq!(fs::read(out.join("rtld")).unwrap(), fake_nso(6));

    // Changed input content gets a different key.
    write(&root.join("exefs/main"), &fake_nso(7));
    let inv2 = src.inventory().unwrap();
    assert_ne!(inv2.packages[0].exefs_key, inv.packages[0].exefs_key);
}

#[test]
fn rejects_unrecognized_directory() {
    let dir = tempfile::tempdir().unwrap();
    fs::create_dir(dir.path().join("something")).unwrap();
    assert!(ContentSource::open(dir.path()).is_err());
}

/// NUP and extracted-directory input must normalize to the same ExeFS key:
/// `ECHORA_TEST_NUP=/path/game.nup cargo test -p echora-content -- --ignored`.
#[test]
#[ignore = "needs a user-supplied NUP in ECHORA_TEST_NUP"]
fn nup_and_directory_inputs_normalize_identically() {
    let nup = std::env::var("ECHORA_TEST_NUP").expect("ECHORA_TEST_NUP");
    let src = ContentSource::open(&nup).unwrap();
    let inv = src.inventory().unwrap();
    let prog = inv.program_package.unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let cached = src
        .materialize_exefs(&inv, prog, &tmp.path().join("cache"))
        .unwrap();

    let dump = tmp.path().join("dump/exefs");
    fs::create_dir_all(&dump).unwrap();
    for m in &inv.packages[prog].exefs {
        fs::copy(cached.join(&m.name), dump.join(&m.name)).unwrap();
    }
    let dir_inv = ContentSource::open(tmp.path().join("dump"))
        .unwrap()
        .inventory()
        .unwrap();
    assert_eq!(dir_inv.packages[0].exefs_key, inv.packages[prog].exefs_key);
    assert_eq!(
        dir_inv.packages[0].main_build_id(),
        inv.packages[prog].main_build_id()
    );
}
