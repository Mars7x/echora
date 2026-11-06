#![allow(clippy::unwrap_used)]

mod common;

use std::io::Read;

use common::{PackageSpec, build, file_record_offset, rehash_toc, write_temp};
use echora_nup::{Codec, Node, NupArchive, NupError, PackageKind};

fn sample(compress: bool) -> Vec<u8> {
    let big: Vec<u8> = (0..200_000u32).map(|i| (i % 251) as u8).collect();
    let mut base = PackageSpec::base(vec![
        ("exefs/main", b"NSO0 not really".to_vec()),
        ("exefs/main.npdm", b"META".to_vec()),
        ("romfs/System/big.bin", big),
        ("romfs/System/empty", Vec::new()),
        ("romfs/Pack/a.pack", b"shared".to_vec()),
        ("romfs/Pack/b.pack", b"shared".to_vec()),
    ]);
    base.meta = vec![
        (0x0004, b"1.0.0".to_vec()),
        (0x0005, b"Test Title".to_vec()),
        (0x0007, [7u8; 32].to_vec()),
        (0x8001, b"vendor".to_vec()),
    ];
    let update = PackageSpec {
        title_id: 0x0400_0000_1234_0800,
        kind: 1,
        version: 0x20000,
        flags: 0,
        files: vec![("exefs/main", b"newer main".to_vec())],
        meta: Vec::new(),
    };
    build(&[base, update], compress)
}

#[test]
fn opens_and_navigates() {
    for compress in [false, true] {
        let tmp = write_temp(&sample(compress));
        let a = NupArchive::open(tmp.path()).unwrap();
        assert_eq!(a.packages().len(), 2);
        let base = &a.packages()[0];
        assert_eq!(base.kind, PackageKind::Base);
        assert_eq!(a.packages()[1].kind, PackageKind::Update);
        assert_eq!(a.packages()[1].version, 0x20000);
        assert!(a.packages()[1].romfs.is_none());

        let exefs = base.exefs.unwrap();
        let names: Vec<_> = a.children(exefs).map(|n| a.name(n).to_owned()).collect();
        assert_eq!(names, ["main", "main.npdm"]);

        let romfs = base.romfs.unwrap();
        let Some(Node::File(big)) = a.resolve(romfs, "System/big.bin") else {
            panic!("big.bin not found")
        };
        let data = a.read_file(big, 1 << 20).unwrap();
        assert_eq!(data.len(), 200_000);
        assert_eq!(data[251], 0);
        if compress {
            assert_eq!(a.file(big).codec, Codec::Zstd);
            assert!(a.file(big).extent_size < 200_000);
        }
        assert!(a.resolve(romfs, "System/missing").is_none());
        assert!(a.resolve(romfs, "../romfs").is_none());

        let files: Vec<_> = a.walk_files(romfs).into_iter().map(|(p, _)| p).collect();
        assert_eq!(
            files,
            [
                "Pack/a.pack",
                "Pack/b.pack",
                "System/big.bin",
                "System/empty"
            ]
        );

        let meta = a.package_meta(base).unwrap();
        assert_eq!(meta.display_version.as_deref(), Some("1.0.0"));
        assert_eq!(meta.title_name.as_deref(), Some("Test Title"));
        assert_eq!(meta.build_id, Some([7; 32]));
        assert_eq!(meta.record_types, [0x0004, 0x0005, 0x0007, 0x8001]);
    }
}

#[test]
fn identical_files_share_one_extent() {
    let tmp = write_temp(&sample(false));
    let a = NupArchive::open(tmp.path()).unwrap();
    let romfs = a.packages()[0].romfs.unwrap();
    let (Some(Node::File(x)), Some(Node::File(y))) = (
        a.resolve(romfs, "Pack/a.pack"),
        a.resolve(romfs, "Pack/b.pack"),
    ) else {
        panic!()
    };
    assert_eq!(a.file(x).extent_offset, a.file(y).extent_offset);
    let report = a.verify_contents(4, &|_| {});
    assert!(report.ok(), "{:?}", report.problems);
    assert_eq!(report.files, report.extents + 1);
}

#[test]
fn verify_detects_corrupted_extent() {
    for compress in [false, true] {
        let mut bytes = sample(compress);
        let tmp = write_temp(&bytes);
        let a = NupArchive::open(tmp.path()).unwrap();
        let romfs = a.packages()[0].romfs.unwrap();
        let Some(Node::File(big)) = a.resolve(romfs, "System/big.bin") else {
            panic!()
        };
        let off = a.file(big).extent_offset as usize + 100;
        drop(a);
        bytes[off] ^= 0x55;
        let tmp = write_temp(&bytes);
        let a = NupArchive::open(tmp.path()).unwrap();
        let report = a.verify_contents(2, &|_| {});
        assert_eq!(report.problems.len(), 1);
        assert_eq!(report.problems[0].file, big);
        assert!(matches!(
            a.read_file(big, 1 << 20),
            Err(NupError::Integrity(_))
        ));
    }
}

#[test]
fn unverified_stream_still_bounds_length() {
    let tmp = write_temp(&sample(true));
    let a = NupArchive::open(tmp.path()).unwrap();
    let romfs = a.packages()[0].romfs.unwrap();
    let Some(Node::File(big)) = a.resolve(romfs, "System/big.bin") else {
        panic!()
    };
    let mut out = Vec::new();
    a.open_file(big, false)
        .unwrap()
        .read_to_end(&mut out)
        .unwrap();
    assert_eq!(out.len(), 200_000);
}

#[test]
fn rejects_tampered_toc() {
    let mut bytes = sample(false);
    bytes[0x80 + 0x30] ^= 1; // first package's title ID
    let tmp = write_temp(&bytes);
    assert!(matches!(
        NupArchive::open(tmp.path()),
        Err(NupError::Integrity(_))
    ));

    // With the TOC hash fixed up the UUID no longer matches the packages.
    rehash_toc(&mut bytes);
    let tmp = write_temp(&bytes);
    assert!(matches!(
        NupArchive::open(tmp.path()),
        Err(NupError::Integrity(_))
    ));
}

#[test]
fn rejects_truncated_and_extended_files() {
    let bytes = sample(false);
    let tmp = write_temp(&bytes[..bytes.len() - 1]);
    assert!(matches!(
        NupArchive::open(tmp.path()),
        Err(NupError::Format(_))
    ));
    let mut longer = bytes.clone();
    longer.push(0);
    let tmp = write_temp(&longer);
    assert!(matches!(
        NupArchive::open(tmp.path()),
        Err(NupError::Format(_))
    ));
    let tmp = write_temp(&bytes[..0x40]);
    assert!(NupArchive::open(tmp.path()).is_err());
    let tmp = write_temp(b"PFS0 something else entirely......");
    assert!(NupArchive::open(tmp.path()).is_err());
}

fn edit_file_record(edit: impl Fn(&mut [u8])) -> Result<NupArchive, NupError> {
    let mut bytes = sample(false);
    let at = file_record_offset(&bytes, 0);
    edit(&mut bytes[at..at + 0x50]);
    rehash_toc(&mut bytes);
    let tmp = write_temp(&bytes);
    NupArchive::open(tmp.path())
}

#[test]
fn rejects_bad_file_records() {
    // Unknown codec.
    assert!(edit_file_record(|r| r[0x40] = 9).is_err());
    // Extent past end of file.
    assert!(edit_file_record(|r| r[0x10..0x18].copy_from_slice(&u64::MAX.to_le_bytes())).is_err());
    // Misaligned extent.
    assert!(
        edit_file_record(|r| {
            let off = u64::from_le_bytes(r[0x10..0x18].try_into().unwrap()) + 1;
            r[0x10..0x18].copy_from_slice(&off.to_le_bytes());
        })
        .is_err()
    );
    // Name outside the string pool.
    assert!(edit_file_record(|r| r[0..4].copy_from_slice(&u32::MAX.to_le_bytes())).is_err());
    // Raw extent whose size disagrees with the file size.
    assert!(
        edit_file_record(|r| {
            let size = u64::from_le_bytes(r[0x08..0x10].try_into().unwrap()) + 16;
            r[0x08..0x10].copy_from_slice(&size.to_le_bytes());
        })
        .is_err()
    );
}

#[test]
fn rejects_path_traversal_names() {
    let bytes = build(
        &[PackageSpec::base(vec![("romfs/xx", b"x".to_vec())])],
        false,
    );
    let pos = bytes.windows(2).rposition(|w| w == b"xx").unwrap();
    for bad in [b"..".as_slice(), b"a/", b"a\\"] {
        let mut b = bytes.clone();
        b[pos..pos + 2].copy_from_slice(&bad[..2]);
        rehash_toc(&mut b);
        let tmp = write_temp(&b);
        assert!(
            matches!(NupArchive::open(tmp.path()), Err(NupError::Format(_))),
            "accepted name {bad:?}"
        );
    }
}

#[test]
fn rejects_update_without_base_unless_standalone() {
    let mut update = PackageSpec::base(vec![("exefs/main", b"m".to_vec())]);
    update.kind = 1;
    let tmp = write_temp(&build(std::slice::from_ref(&update), false));
    assert!(NupArchive::open(tmp.path()).is_err());
    update.flags = 1;
    let tmp = write_temp(&build(&[update], false));
    let a = NupArchive::open(tmp.path()).unwrap();
    assert!(a.packages()[0].standalone_update);
}

#[test]
fn extracts_subtree_with_verification() {
    let tmp = write_temp(&sample(true));
    let a = NupArchive::open(tmp.path()).unwrap();
    let out = tempfile::tempdir().unwrap();
    let dest = out.path().join("romfs");
    let stats = a
        .extract_dir(a.packages()[0].romfs.unwrap(), &dest)
        .unwrap();
    assert_eq!(stats.files, 4);
    assert_eq!(stats.dirs, 2);
    assert_eq!(std::fs::read(dest.join("Pack/b.pack")).unwrap(), b"shared");
    assert_eq!(
        std::fs::read(dest.join("System/big.bin")).unwrap().len(),
        200_000
    );
    assert!(std::fs::read(dest.join("System/empty")).unwrap().is_empty());

    // Refuses to merge into a non-empty destination.
    assert!(
        a.extract_dir(a.packages()[0].romfs.unwrap(), &dest)
            .is_err()
    );
}

/// Cross-checks the reader against a real archive when one is supplied:
/// `ECHORA_TEST_NUP=/path/game.nup cargo test -p echora-nup -- --ignored`.
#[test]
#[ignore = "needs a user-supplied NUP in ECHORA_TEST_NUP"]
fn real_archive_opens_and_exefs_verifies() {
    let path = std::env::var("ECHORA_TEST_NUP").expect("ECHORA_TEST_NUP");
    let a = NupArchive::open(path).unwrap();
    for p in a.packages() {
        let Some(exefs) = p.exefs else { continue };
        for (name, id) in a.walk_files(exefs) {
            let mut r = a.open_file(id, true).unwrap();
            std::io::copy(&mut r, &mut std::io::sink())
                .unwrap_or_else(|e| panic!("exefs/{name}: {e}"));
        }
    }
}
