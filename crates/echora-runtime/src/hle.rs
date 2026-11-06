//! Host replacements for hot guest library functions.
//!
//! A replacement is installed only when the guest function's code bytes match
//! a recorded SHA-256, so it can never attach to different code. Set
//! `ECHORA_NO_HLE` to run the translated guest code instead.

use echora_guest::{Trap, memory::Memory};
use sha2::{Digest, Sha256};

use crate::loader::NsoModule;

/// The game's one-shot zlib wrapper in `main` (module text offset, the
/// SHA-256 of its 81 instructions) per known build, as disassembled from
/// updates 327680 and 655360:
/// `u64 uncompress(void* dst, u32 dst_size, const void* src, u64 src_size,
/// alloc_fn, free_fn)`. It runs `inflateInit_` on a zlib-format stream, feeds
/// the input in 64 KiB pieces to `inflate(Z_NO_FLUSH)`, and returns
/// `total_out` only when the stream ends (`Z_STREAM_END`) and `inflateEnd`
/// succeeds; input exhausted early, a full output buffer before the end, or
/// any zlib error returns 0. The allocator pair only backs zlib's internal
/// state, so skipping it has no other observable effect.
const ZLIB_UNCOMPRESS: [(u64, &str); 2] = [
    (
        0x103_4fe0,
        "19bf0380cf908860cbfb3c34715afd6e426566b50fe2bc8bf84f6fdb6870f003",
    ),
    (
        0x112_46e0,
        "78589139753c6687934449a35099348e25bb8daf71b46abd2af7bce7bcf8cb4d",
    ),
];
const ZLIB_UNCOMPRESS_WORDS: usize = 81;

/// zlib's `inflate` in `main` (offset, SHA-256 of its first 0x200 bytes) and
/// the `bl inflate` sites that pass Z_FINISH (`mov w1, #4` just before)
/// right after `inflateInit_`, so the stream is fresh, per known build.
/// Every site reads `total_out` only on Z_STREAM_END and then calls
/// `inflateEnd`.
struct InflateBuild {
    offset: u64,
    prefix_sha256: &'static str,
    finish_sites: [u64; 6],
}

const INFLATE: [InflateBuild; 2] = [
    InflateBuild {
        offset: 0x127_3880,
        prefix_sha256: "462b396a424f29f8d59d2881a76c82078b58cbeb317f1ecc452f2ea1515527f6",
        finish_sites: [
            0x4e_30f4, 0x50_fef8, 0x51_7970, 0x121_b7a0, 0x1f0_4a4c, 0x1f0_7860,
        ],
    },
    InflateBuild {
        offset: 0x136_31a0,
        prefix_sha256: "5678d0786550872b4b7c99d530cfea22ef53010cf4c1cd51fdb98e60e9c8e9b8",
        finish_sites: [
            0x54_77d4, 0x57_4f58, 0x57_cbd0, 0x130_b0c0, 0x1ff_458c, 0x1ff_73a0,
        ],
    },
];
const MOV_W1_4: u32 = 0x5280_0081;

const Z_STREAM_END: i32 = 1;
const Z_DATA_ERROR: i32 = -3;
const Z_BUF_ERROR: i32 = -5;

/// Largest compressed input read in one call (an explicit local bound).
const MAX_INPUT: u64 = 256 << 20;

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// The text offset of the zlib wrapper in `module`, if the exact code of a
/// known build is present there.
pub fn zlib_uncompress_offset(module: &NsoModule) -> Option<u64> {
    let text = &module.text;
    ZLIB_UNCOMPRESS.into_iter().find_map(|(offset, sha256)| {
        let start = offset.checked_sub(u64::from(text.memory_offset))? as usize;
        let bytes = text.bytes.get(start..start + ZLIB_UNCOMPRESS_WORDS * 4)?;
        (sha256_hex(bytes) == sha256).then_some(offset)
    })
}

/// The `bl inflate` / Z_FINISH call sites present with their exact code.
pub fn inflate_finish_sites(module: &NsoModule) -> Vec<u64> {
    let text = &module.text;
    let word = |offset: u64| -> Option<u32> {
        let start = offset.checked_sub(u64::from(text.memory_offset))? as usize;
        text.bytes
            .get(start..start + 4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
    };
    let matching = INFLATE.iter().find(|build| {
        build
            .offset
            .checked_sub(u64::from(text.memory_offset))
            .and_then(|start| text.bytes.get(start as usize..start as usize + 0x200))
            .is_some_and(|prefix| sha256_hex(prefix) == build.prefix_sha256)
    });
    let Some(build) = matching else {
        return Vec::new();
    };
    build
        .finish_sites
        .into_iter()
        .filter(|&site| {
            let bl =
                0x9400_0000 | (((build.offset.wrapping_sub(site) as i64 >> 2) as u32) & 0x3ff_ffff);
            word(site) == Some(bl) && word(site - 4) == Some(MOV_W1_4)
        })
        .collect()
}

/// zlib `inflate(strm, Z_FINISH)` on a freshly initialized zlib-format
/// stream: decode in one step and update the z_stream fields zlib updates
/// (next/avail/total in and out, and adler on success). Returns
/// Z_STREAM_END, Z_BUF_ERROR (input or output ran out) or Z_DATA_ERROR.
/// The decoder's private state is not touched; the caller only calls
/// `inflateEnd` next. A stream that already made progress stops explicitly.
pub fn inflate_finish(memory: &mut Memory, stream: u64, site: u64) -> Result<i32, Trap> {
    use miniz_oxide::inflate::TINFLStatus;
    use miniz_oxide::inflate::core::{DecompressorOxide, decompress, inflate_flags};
    let next_in = memory.read_u64(stream)?;
    let avail_in = u64::from(memory.read_u32(stream + 0x08)?);
    let total_in = memory.read_u64(stream + 0x10)?;
    let next_out = memory.read_u64(stream + 0x18)?;
    let avail_out = u64::from(memory.read_u32(stream + 0x20)?);
    let total_out = memory.read_u64(stream + 0x28)?;
    if total_in != 0 || total_out != 0 {
        return Err(Trap::InvalidState {
            message: format!(
                "inflate(Z_FINISH) at {site:#x} on a stream that already made progress"
            ),
        });
    }
    if avail_in > MAX_INPUT || avail_out > MAX_INPUT * 8 {
        return Err(Trap::InvalidState {
            message: format!("inflate buffers at {site:#x} exceed the local bound"),
        });
    }
    let mut input = vec![0u8; avail_in as usize];
    memory.read(next_in, &mut input)?;
    let mut output = vec![0u8; avail_out as usize];
    let mut decompressor = DecompressorOxide::new();
    let flags = inflate_flags::TINFL_FLAG_PARSE_ZLIB_HEADER
        | inflate_flags::TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF;
    let (status, consumed, written) = decompress(&mut decompressor, &input, &mut output, 0, flags);
    let result = match status {
        TINFLStatus::Done => Z_STREAM_END,
        TINFLStatus::NeedsMoreInput | TINFLStatus::HasMoreOutput => Z_BUF_ERROR,
        _ => Z_DATA_ERROR,
    };
    memory.write(next_out, &output[..written])?;
    memory.write_u64(stream, next_in + consumed as u64)?;
    memory.write_u32(stream + 0x08, (avail_in - consumed as u64) as u32)?;
    memory.write_u64(stream + 0x10, consumed as u64)?;
    memory.write_u64(stream + 0x18, next_out + written as u64)?;
    memory.write_u32(stream + 0x20, (avail_out - written as u64) as u32)?;
    memory.write_u64(stream + 0x28, written as u64)?;
    if result == Z_STREAM_END {
        let adler = decompressor.adler32().unwrap_or(1);
        memory.write_u64(stream + 0x60, u64::from(adler))?;
    }
    Ok(result)
}

/// Host equivalent of the wrapper: the decompressed size, or 0 on failure.
pub fn zlib_uncompress(
    memory: &mut Memory,
    destination: u64,
    capacity: u64,
    source: u64,
    size: u64,
) -> Result<u64, Trap> {
    if size > MAX_INPUT {
        return Err(Trap::InvalidState {
            message: format!("zlib input of {size:#x} bytes exceeds the local bound"),
        });
    }
    let mut input = vec![0u8; size as usize];
    memory.read(source, &mut input)?;
    match inflate_zlib(&input, capacity as usize) {
        Some(output) => {
            memory.write(destination, &output)?;
            Ok(output.len() as u64)
        }
        None => Ok(0),
    }
}

/// Decode one complete zlib stream (header, data, Adler-32) of at most
/// `capacity` bytes; trailing input after the stream end is ignored.
fn inflate_zlib(input: &[u8], capacity: usize) -> Option<Vec<u8>> {
    miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(input, capacity).ok()
}

#[cfg(all(test, any(target_os = "linux", windows, target_os = "macos")))]
mod tests {
    use super::*;

    /// "hello hello hello hello" compressed by zlib (level 6).
    const HELLO: &[u8] = &[
        0x78, 0x9c, 0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x57, 0xc8, 0x40, 0x27, 0x01, 0x68, 0x03, 0x08,
        0xb1,
    ];

    fn stream_memory(input: &[u8], capacity: u32) -> Memory {
        use echora_guest::memory::{READ, STATE_NORMAL, WRITE};
        let mut memory = Memory::new(0x80_0000_0000);
        memory
            .map(0x10000, 0x10000, READ | WRITE, STATE_NORMAL, "test")
            .unwrap();
        memory.write(0x11000, input).unwrap();
        // z_stream at 0x10000: next_in, avail_in, total_in, next_out,
        // avail_out, total_out (zero after inflateInit_).
        memory.write_u64(0x10000, 0x11000).unwrap();
        memory.write_u32(0x10008, input.len() as u32).unwrap();
        memory.write_u64(0x10018, 0x12000).unwrap();
        memory.write_u32(0x10020, capacity).unwrap();
        memory
    }

    #[test]
    fn inflate_finish_updates_the_stream_like_zlib() {
        let expected = b"hello hello hello hello";
        let mut memory = stream_memory(HELLO, 64);
        assert_eq!(
            inflate_finish(&mut memory, 0x10000, 0x1000).unwrap(),
            Z_STREAM_END
        );
        let mut output = [0u8; 23];
        memory.read(0x12000, &mut output).unwrap();
        assert_eq!(&output, expected);
        let len = HELLO.len() as u64;
        assert_eq!(memory.read_u64(0x10000).unwrap(), 0x11000 + len);
        assert_eq!(memory.read_u32(0x10008).unwrap(), 0);
        assert_eq!(memory.read_u64(0x10010).unwrap(), len);
        assert_eq!(memory.read_u64(0x10018).unwrap(), 0x12000 + 23);
        assert_eq!(memory.read_u32(0x10020).unwrap(), 64 - 23);
        assert_eq!(memory.read_u64(0x10028).unwrap(), 23);
        // Adler-32 of the output, as stored in the stream trailer.
        let trailer = u32::from_be_bytes(HELLO[len as usize - 4..].try_into().unwrap());
        assert_eq!(memory.read_u64(0x10060).unwrap(), u64::from(trailer));

        // Too little output space, and corrupt data.
        let mut memory = stream_memory(HELLO, 10);
        assert_eq!(
            inflate_finish(&mut memory, 0x10000, 0x1000).unwrap(),
            Z_BUF_ERROR
        );
        let mut corrupt = HELLO.to_vec();
        corrupt[1] ^= 0xff;
        let mut memory = stream_memory(&corrupt, 64);
        assert_eq!(
            inflate_finish(&mut memory, 0x10000, 0x1000).unwrap(),
            Z_DATA_ERROR
        );
        // A stream that already made progress is not handled here.
        let mut memory = stream_memory(HELLO, 64);
        memory.write_u64(0x10010, 1).unwrap();
        assert!(inflate_finish(&mut memory, 0x10000, 0x1000).is_err());
    }

    #[test]
    fn complete_streams_decode_and_failures_return_none() {
        let expected = b"hello hello hello hello";
        assert_eq!(inflate_zlib(HELLO, 64).as_deref(), Some(&expected[..]));
        // Exactly enough room, and trailing bytes after the stream end.
        let mut trailing = HELLO.to_vec();
        trailing.extend_from_slice(b"junk");
        assert_eq!(
            inflate_zlib(&trailing, expected.len()).as_deref(),
            Some(&expected[..])
        );
        // Too little room, a truncated stream and a corrupted checksum fail.
        assert_eq!(inflate_zlib(HELLO, expected.len() - 1), None);
        assert_eq!(inflate_zlib(&HELLO[..HELLO.len() - 3], 64), None);
        let mut corrupt = HELLO.to_vec();
        *corrupt.last_mut().unwrap() ^= 1;
        assert_eq!(inflate_zlib(&corrupt, 64), None);
    }
}
