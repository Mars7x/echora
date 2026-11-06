#![allow(clippy::unwrap_used)]

use std::fs;

use echora_loader::{Compression, LoaderError, exefs_load_order, load_exefs, nso};
use sha2::{Digest, Sha256};

/// Builds an NSO with LZ4-compressed, hash-checked segments.
fn build_nso(text: &[u8], rodata: &[u8], data: &[u8], bss: u32) -> Vec<u8> {
    let segs = [text, rodata, data];
    let mut out = vec![0u8; nso::HEADER_SIZE];
    out[..4].copy_from_slice(b"NSO0");
    out[0x0c..0x10].copy_from_slice(&0x3fu32.to_le_bytes());
    let mut mem = 0u32;
    for (i, s) in segs.iter().enumerate() {
        let packed = lz4_flex::block::compress(s);
        let h = 0x10 + i * 0x10;
        let file_offset = out.len() as u32;
        out[h..h + 4].copy_from_slice(&file_offset.to_le_bytes());
        out[h + 4..h + 8].copy_from_slice(&mem.to_le_bytes());
        out[h + 8..h + 12].copy_from_slice(&(s.len() as u32).to_le_bytes());
        out[0x60 + i * 4..0x64 + i * 4].copy_from_slice(&(packed.len() as u32).to_le_bytes());
        out[0xa0 + i * 0x20..0xc0 + i * 0x20].copy_from_slice(&Sha256::digest(s));
        out.extend_from_slice(&packed);
        mem = (mem + s.len() as u32 + 0xfff) & !0xfff;
    }
    out[0x3c..0x40].copy_from_slice(&bss.to_le_bytes());
    out[0x40..0x48].copy_from_slice(&[0xab; 8]);
    out
}

fn sample() -> Vec<u8> {
    let text: Vec<u8> = (0..0x2000u32)
        .flat_map(|i| (0xd503_201f ^ (i % 7)).to_le_bytes())
        .collect();
    build_nso(
        &text,
        b"read-only data, read-only data",
        &[1, 2, 3, 4],
        0x800,
    )
}

#[test]
fn parses_lz4_module() {
    let m = nso::parse("main", &sample(), nso::MAX_IMAGE_SIZE).unwrap();
    assert_eq!(m.text.bytes.len(), 0x8000);
    assert_eq!(m.text.compression, Compression::Lz4);
    assert!(m.segments().iter().all(|s| s.hash_verified));
    assert_eq!(m.rodata.memory_offset, 0x8000);
    assert_eq!(m.data.bytes, [1, 2, 3, 4]);
    assert_eq!(m.image_size, 0xa000);
    assert_eq!(m.build_id[..8], [0xab; 8]);
}

#[test]
fn detects_corrupt_segments_and_headers() {
    let good = sample();

    let mut b = good.clone();
    b[0xa0] ^= 1; // text hash
    assert!(matches!(
        nso::parse("m", &b, nso::MAX_IMAGE_SIZE),
        Err(LoaderError::Integrity(_))
    ));

    let mut b = good.clone();
    b[0x18..0x1c].copy_from_slice(&0x9000u32.to_le_bytes()); // wrong text size
    assert!(nso::parse("m", &b, nso::MAX_IMAGE_SIZE).is_err());

    let mut b = good.clone();
    b[0x24..0x28].copy_from_slice(&0x10u32.to_le_bytes()); // rodata overlaps text in memory
    assert!(nso::parse("m", &b, nso::MAX_IMAGE_SIZE).is_err());

    let mut b = good.clone();
    b[0x10..0x14].copy_from_slice(&0x10u32.to_le_bytes()); // text inside the header
    assert!(nso::parse("m", &b, nso::MAX_IMAGE_SIZE).is_err());

    assert!(nso::parse("m", &good[..good.len() - 1], nso::MAX_IMAGE_SIZE).is_err());
    assert!(nso::parse("m", &good, 0x4000).is_err()); // image budget
}

#[test]
fn flag_handling() {
    let mut b = sample();
    b[0x0c] = 0x3f | 0x40; // bit 6 alone is not understood
    assert!(nso::parse("m", &b, nso::MAX_IMAGE_SIZE).is_err());
    b[0x0d] = 0x01; // unknown high flag
    assert!(nso::parse("m", &b, nso::MAX_IMAGE_SIZE).is_err());
    // ZBIC flag on LZ4 data fails on the frame magic, not silently.
    let mut b = sample();
    b[0x0c] = 0xff;
    let e = nso::parse("m", &b, nso::MAX_IMAGE_SIZE)
        .unwrap_err()
        .to_string();
    assert!(e.contains("ZBIC"), "{e}");
}

#[test]
fn exefs_order_and_required_modules() {
    let dir = tempfile::tempdir().unwrap();
    for name in ["sdk", "main", "subsdk1", "subsdk0", "rtld", "main.npdm"] {
        fs::write(dir.path().join(name), sample()).unwrap();
    }
    assert_eq!(
        exefs_load_order(dir.path()).unwrap(),
        ["rtld", "main", "subsdk0", "subsdk1", "sdk"]
    );
    assert_eq!(load_exefs(dir.path()).unwrap().len(), 5);

    fs::write(dir.path().join("subsdk10"), sample()).unwrap();
    assert!(exefs_load_order(dir.path()).is_err());
    fs::remove_file(dir.path().join("subsdk10")).unwrap();
    fs::remove_file(dir.path().join("sdk")).unwrap();
    assert!(exefs_load_order(dir.path()).is_err());
}

/// Loads a real ExeFS, including ZBIC modules, checks every enabled segment
/// hash and requires MOD0 dynamic metadata with a counted symbol table: `ECHORA_TEST_EXEFS=/path/exefs cargo test -p echora-loader -- --ignored`.
#[test]
#[ignore = "needs a user-supplied ExeFS in ECHORA_TEST_EXEFS"]
fn real_exefs_loads_with_verified_hashes() {
    let dir = std::env::var("ECHORA_TEST_EXEFS").expect("ECHORA_TEST_EXEFS");
    let modules = load_exefs(std::path::Path::new(&dir)).unwrap();
    for m in &modules {
        for s in m.segments() {
            assert!(
                s.hash_verified || s.bytes.is_empty(),
                "{} not hash-checked",
                m.name
            );
        }
        let dynamic = m.mod0.as_ref().and_then(|i| i.dynamic.as_ref());
        let symbols = dynamic.and_then(|d| d.symbol_table.as_ref());
        assert!(
            symbols.and_then(|t| t.count).is_some(),
            "{}: no counted symbol table",
            m.name
        );
    }
}
