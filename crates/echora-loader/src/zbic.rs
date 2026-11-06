//! ZBIC segment decompression.
//!
//! ZBIC, introduced with Switch firmware 22.0.0, is Zstandard with two
//! changes: the frame magic is `ZBIC` (`0x4349425A`) instead of
//! `0xFD2FB528`, and every FSE table description (normalized symbol counts)
//! is stored with binary interpolative coding instead of the standard
//! variable-length bit fields. Block structure, Huffman and FSE bitstreams,
//! sequences and checksums are unchanged.
//!
//! Rather than reimplementing a Zstandard decoder, this module transcodes a
//! ZBIC frame into a standard frame (re-encoding each table description and
//! adjusting the size fields that enclose it) and decompresses that with the
//! reference zstd library. The format knowledge comes from the public
//! description and behavior of `kinnay/zbic`; no code is taken from it.

use std::fmt;

pub const ZBIC_MAGIC: [u8; 4] = *b"ZBIC";
const ZSTD_MAGIC: [u8; 4] = 0xFD2F_B528u32.to_le_bytes();
/// Zstandard's maximum block content size (and compressed block size).
const BLOCK_SIZE_MAX: usize = 128 << 10;

// Maximum symbol values per table (RFC 8878 §3.1.1.3.2.1).
const MAX_LL: u32 = 35;
const MAX_OF: u32 = 31;
const MAX_ML: u32 = 52;
const MAX_HUF_WEIGHT_SYMBOL: u32 = 255;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZbicError(pub String);

impl fmt::Display for ZbicError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ZBIC: {}", self.0)
    }
}

impl std::error::Error for ZbicError {}

fn err<T>(msg: impl Into<String>) -> Result<T, ZbicError> {
    Err(ZbicError(msg.into()))
}

/// A decoded FSE table description.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizedCounts {
    /// Per-symbol normalized counts; `-1` marks a "less than one"
    /// probability. Absolute values sum to `1 << table_log`.
    pub counts: Vec<i16>,
    pub table_log: u32,
}

/// Counters from one transcoding pass, used to show that tests exercise the
/// table-bearing paths.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TranscodeStats {
    pub blocks: u32,
    pub huffman_fse_tables: u32,
    pub sequence_fse_tables: u32,
}

/// Decompresses one ZBIC frame whose content size must be `expected_size`.
pub fn decompress(src: &[u8], expected_size: usize) -> Result<Vec<u8>, ZbicError> {
    let (frame, _) = transcode(src, ZBIC_MAGIC, read_bic_counts)?;
    let mut out = vec![0u8; expected_size];
    let n = zstd::bulk::Decompressor::new()
        .and_then(|mut d| d.decompress_to_buffer(&frame, &mut out))
        .map_err(|e| ZbicError(format!("zstd rejected the transcoded frame: {e}")))?;
    if n != expected_size {
        return err(format!(
            "frame decoded to {n} bytes, expected {expected_size}"
        ));
    }
    Ok(out)
}

type CountReader = fn(&[u8], u32) -> Result<(NormalizedCounts, usize), ZbicError>;

/// Rewrites one frame with magic `magic`, decoding each table description with
/// `read` and writing it in standard form. The whole input must be one frame.
pub(crate) fn transcode(
    src: &[u8],
    magic: [u8; 4],
    read: CountReader,
) -> Result<(Vec<u8>, TranscodeStats), ZbicError> {
    let mut stats = TranscodeStats::default();
    if src.get(..4) != Some(&magic[..]) {
        return err("bad frame magic");
    }
    let mut out = Vec::with_capacity(src.len() + src.len() / 64 + 64);
    out.extend_from_slice(&ZSTD_MAGIC);

    let fhd = *src
        .get(4)
        .ok_or_else(|| ZbicError("truncated frame header".into()))?;
    if fhd & 0x08 != 0 {
        return err("reserved frame header bit set");
    }
    let single_segment = fhd & 0x20 != 0;
    let checksum = fhd & 0x04 != 0;
    let did_len = [0, 1, 2, 4][usize::from(fhd & 3)];
    let fcs_len = match fhd >> 6 {
        0 => usize::from(single_segment),
        1 => 2,
        2 => 4,
        _ => 8,
    };
    let header_end = 5 + usize::from(!single_segment) + did_len + fcs_len;
    let header = src
        .get(4..header_end)
        .ok_or_else(|| ZbicError("truncated frame header".into()))?;
    out.extend_from_slice(header);

    let mut pos = header_end;
    loop {
        let h = src
            .get(pos..pos + 3)
            .ok_or_else(|| ZbicError("truncated block header".into()))?;
        let raw_header = u32::from(h[0]) | u32::from(h[1]) << 8 | u32::from(h[2]) << 16;
        let last = raw_header & 1 != 0;
        let size = (raw_header >> 3) as usize;
        pos += 3;
        stats.blocks += 1;
        match (raw_header >> 1) & 3 {
            0 | 1 => {
                // Raw or RLE: copy through unchanged.
                let stored = if (raw_header >> 1) & 3 == 0 { size } else { 1 };
                let body = src
                    .get(pos..pos + stored)
                    .ok_or_else(|| ZbicError("truncated block".into()))?;
                out.extend_from_slice(&h[..3]);
                out.extend_from_slice(body);
                pos += stored;
            }
            2 => {
                let body = src
                    .get(pos..pos + size)
                    .ok_or_else(|| ZbicError("truncated block".into()))?;
                let block = transcode_block(body, read, &mut stats)?;
                if block.len() > BLOCK_SIZE_MAX {
                    return err("re-encoded block exceeds the maximum block size");
                }
                let new_header = (block.len() as u32) << 3 | 2 << 1 | u32::from(last);
                out.extend_from_slice(&new_header.to_le_bytes()[..3]);
                out.extend_from_slice(&block);
                pos += size;
            }
            _ => return err("reserved block type"),
        }
        if last {
            break;
        }
    }
    if checksum {
        let c = src
            .get(pos..pos + 4)
            .ok_or_else(|| ZbicError("truncated checksum".into()))?;
        out.extend_from_slice(c);
        pos += 4;
    }
    if pos != src.len() {
        return err(format!("{} bytes after the frame", src.len() - pos));
    }
    Ok((out, stats))
}

fn le_uint(b: &[u8]) -> u64 {
    b.iter().rev().fold(0, |acc, &x| acc << 8 | u64::from(x))
}

fn transcode_block(
    b: &[u8],
    read: CountReader,
    stats: &mut TranscodeStats,
) -> Result<Vec<u8>, ZbicError> {
    let trunc = || ZbicError("truncated compressed block".into());
    let mut out = Vec::with_capacity(b.len() + 16);
    let b0 = *b.first().ok_or_else(trunc)?;
    let lit_type = b0 & 3;
    let size_format = (b0 >> 2) & 3;

    let lit_end = match lit_type {
        0 | 1 => {
            let (hlen, regen) = match size_format {
                0 | 2 => (1, usize::from(b0 >> 3)),
                1 => (2, le_uint(b.get(..2).ok_or_else(trunc)?) as usize >> 4),
                _ => (3, le_uint(b.get(..3).ok_or_else(trunc)?) as usize >> 4),
            };
            let end = hlen + if lit_type == 0 { regen } else { 1 };
            out.extend_from_slice(b.get(..end).ok_or_else(trunc)?);
            end
        }
        _ => {
            let (hlen, bits, four_streams) = match size_format {
                0 => (3, 10, false),
                1 => (3, 10, true),
                2 => (4, 14, true),
                _ => (5, 18, true),
            };
            let v = le_uint(b.get(..hlen).ok_or_else(trunc)?);
            let mask = (1u64 << bits) - 1;
            let regen = (v >> 4) & mask;
            let comp = ((v >> (4 + bits)) & mask) as usize;
            let payload = b.get(hlen..hlen + comp).ok_or_else(trunc)?;
            let section_end = hlen + comp;

            let new_payload = if lit_type == 2 {
                let hb = usize::from(*payload.first().ok_or_else(trunc)?);
                if hb < 128 {
                    // FSE-compressed Huffman weights: table description then
                    // bitstream, `hb` bytes in total.
                    let region = payload.get(1..1 + hb).ok_or_else(trunc)?;
                    let (counts, used) = read(region, MAX_HUF_WEIGHT_SYMBOL)?;
                    let table = write_standard_counts(&counts)?;
                    let new_hb = table.len() + (hb - used);
                    if new_hb >= 128 {
                        return err("re-encoded Huffman weight description too large");
                    }
                    stats.huffman_fse_tables += 1;
                    let mut p = Vec::with_capacity(payload.len() + 8);
                    p.push(new_hb as u8);
                    p.extend_from_slice(&table);
                    p.extend_from_slice(&region[used..]);
                    p.extend_from_slice(&payload[1 + hb..]);
                    p
                } else {
                    payload.to_vec()
                }
            } else {
                payload.to_vec()
            };

            // Keep the size format unless the new size no longer fits; four
            // stream formats can grow to a wider one.
            let new_comp = new_payload.len() as u64;
            let (fmt, hlen, bits) = [(size_format, hlen, bits), (2, 4, 14), (3, 5, 18)]
                .into_iter()
                .find(|&(f, _, bits)| {
                    (f == size_format || (four_streams && f > size_format))
                        && new_comp < 1 << bits
                        && regen < 1 << bits
                })
                .ok_or_else(|| ZbicError("re-encoded literals do not fit their header".into()))?;
            let header =
                u64::from(lit_type) | u64::from(fmt) << 2 | regen << 4 | new_comp << (4 + bits);
            out.extend_from_slice(&header.to_le_bytes()[..hlen]);
            out.extend_from_slice(&new_payload);
            section_end
        }
    };

    let seq = &b[lit_end..];
    let n0 = *seq.first().ok_or_else(trunc)?;
    let count_len = match n0 {
        0 => {
            out.extend_from_slice(seq);
            return Ok(out);
        }
        1..=127 => 1,
        128..=254 => 2,
        255 => 3,
    };
    let modes = *seq.get(count_len).ok_or_else(trunc)?;
    if modes & 3 != 0 {
        return err("reserved sequence mode bits set");
    }
    out.extend_from_slice(&seq[..=count_len]);
    let mut pos = count_len + 1;
    for (shift, max) in [(6, MAX_LL), (4, MAX_OF), (2, MAX_ML)] {
        match (modes >> shift) & 3 {
            1 => {
                out.push(*seq.get(pos).ok_or_else(trunc)?);
                pos += 1;
            }
            2 => {
                let (counts, used) = read(seq.get(pos..).ok_or_else(trunc)?, max)?;
                out.extend_from_slice(&write_standard_counts(&counts)?);
                pos += used;
                stats.sequence_fse_tables += 1;
            }
            _ => {} // predefined or repeat: no table description
        }
    }
    out.extend_from_slice(seq.get(pos..).ok_or_else(trunc)?);
    Ok(out)
}

/// `(middle, first, last)` index triples in the order binary interpolative
/// coding visits them: a right-first preorder walk of the interval [0, 256],
/// followed by a terminal `(0, 0, 1)` step. A description with `n` (a power of
/// two) cumulative slots uses the last `n` steps, which walk [0, n].
fn bic_steps() -> &'static [(u16, u16, u16); 256] {
    static STEPS: std::sync::OnceLock<[(u16, u16, u16); 256]> = std::sync::OnceLock::new();
    STEPS.get_or_init(|| {
        let mut steps = [(0, 0, 0); 256];
        let mut n = 0;
        let mut stack = vec![(0u16, 256u16)];
        while let Some((lo, hi)) = stack.pop() {
            if hi - lo < 2 {
                continue;
            }
            let mid = (lo + hi) / 2;
            steps[n] = (mid, lo, hi);
            n += 1;
            // Visit (mid, hi) before (lo, mid).
            stack.push((lo, mid));
            stack.push((mid, hi));
        }
        steps[255] = (0, 0, 1);
        steps
    })
}

/// Pulls bytes from the end of the description toward its start until the
/// state holds at least 40 bits or the description is exhausted.
fn refill(mut x: u64, buf: &[u8], rem: &mut usize) -> u64 {
    while *rem != 0 && x >> 32 < 0x100 {
        x = u64::from(buf[*rem]) | x << 8;
        *rem -= 1;
    }
    x
}

/// Decodes a ZBIC (binary interpolative) table description. Returns the
/// counts and the number of bytes consumed.
pub fn read_bic_counts(
    buf: &[u8],
    max_symbol: u32,
) -> Result<(NormalizedCounts, usize), ZbicError> {
    let corrupt = |what: &str| ZbicError(format!("corrupt table description: {what}"));
    let b0 = *buf.first().ok_or_else(|| corrupt("empty"))?;
    let mut rem = usize::from(b0 & 0x7f);
    let low_prob = u64::from(b0 >> 7);
    let consumed = rem + 1;
    // The description is always followed by more data in a valid block.
    if consumed >= buf.len() {
        return Err(corrupt("overruns its section"));
    }

    let x = refill(0, buf, &mut rem);
    let max_sv = (x % 0x34) as u32 + 1;
    let x = refill(x / 0x34, buf, &mut rem);
    if max_sv > max_symbol {
        return Err(corrupt("symbol count exceeds the table's alphabet"));
    }
    let table_log = (x & 7) as u32 + 5;
    let x = refill(x >> 3, buf, &mut rem);
    let total = 1u64 << table_log;
    let last = if low_prob != 0 {
        u64::from(max_sv) + x % total + 2
    } else {
        x % total + 1
    };
    let mut state = refill(x / total, buf, &mut rem);

    // Cumulative (count + low_prob) values at slot boundaries 0..=slots,
    // where slots is the smallest power of two above the largest symbol.
    let slots = (max_sv + 1).next_power_of_two() as usize;
    if slots > 0xff {
        return Err(corrupt("too many symbols"));
    }
    let mut cum = [0u64; 257];
    cum[slots] = last;
    for &(mid, first, end) in &bic_steps()[256 - slots..] {
        let (lo, hi) = (cum[usize::from(first)], cum[usize::from(end)]);
        if lo == hi {
            for c in &mut cum[usize::from(first) + 1..usize::from(end)] {
                *c = lo;
            }
        } else {
            if hi < lo {
                return Err(corrupt("cumulative counts decrease"));
            }
            let range = hi - lo + 1;
            let entry = state % range;
            state = refill(state / range, buf, &mut rem);
            cum[usize::from(mid)] = entry + lo;
        }
    }
    if rem != 0 {
        return Err(corrupt("unused bytes"));
    }

    let mut counts = Vec::with_capacity(max_sv as usize + 1);
    let mut acc = 0i64;
    let mut weight = 0i64;
    for &c in &cum[1..=max_sv as usize + 1] {
        let dist = c as i64 - acc;
        acc = c as i64;
        let count = dist - low_prob as i64;
        if !(-1..=i64::from(i16::MAX)).contains(&count) {
            return Err(corrupt("count out of range"));
        }
        weight += count.abs();
        counts.push(count as i16);
    }
    if weight != total as i64 {
        return Err(corrupt("counts do not sum to the table size"));
    }
    Ok((NormalizedCounts { counts, table_log }, consumed))
}

/// Little-endian bit writer.
struct BitWriter {
    out: Vec<u8>,
    acc: u64,
    bits: u32,
}

impl BitWriter {
    fn put(&mut self, value: u32, n: u32) {
        self.acc |= u64::from(value) << self.bits;
        self.bits += n;
        while self.bits >= 8 {
            self.out.push(self.acc as u8);
            self.acc >>= 8;
            self.bits -= 8;
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.bits > 0 {
            self.out.push(self.acc as u8);
        }
        self.out
    }
}

/// Encodes counts as a standard Zstandard FSE table description
/// (RFC 8878 §4.1.1).
pub fn write_standard_counts(n: &NormalizedCounts) -> Result<Vec<u8>, ZbicError> {
    if !(5..=15).contains(&n.table_log) {
        return err("table log out of range");
    }
    let mut w = BitWriter {
        out: Vec::new(),
        acc: 0,
        bits: 0,
    };
    w.put(n.table_log - 5, 4);
    let table_size = 1i32 << n.table_log;
    let mut remaining = table_size + 1;
    let mut threshold = table_size;
    let mut nb_bits = n.table_log + 1;
    let mut symbol = 0usize;
    let mut previous_zero = false;
    while symbol < n.counts.len() && remaining > 1 {
        if previous_zero {
            let mut start = symbol;
            while symbol < n.counts.len() && n.counts[symbol] == 0 {
                symbol += 1;
            }
            if symbol == n.counts.len() {
                break;
            }
            while symbol >= start + 24 {
                start += 24;
                w.put(0xffff, 16);
            }
            while symbol >= start + 3 {
                start += 3;
                w.put(3, 2);
            }
            w.put((symbol - start) as u32, 2);
        }
        let mut count = i32::from(n.counts[symbol]);
        symbol += 1;
        let max = (2 * threshold - 1) - remaining;
        remaining -= count.abs();
        count += 1;
        if count >= threshold {
            count += max;
        }
        // Values below `max` are written with one bit fewer.
        w.put(
            count as u32,
            if count < max { nb_bits - 1 } else { nb_bits },
        );
        previous_zero = count == 1;
        if remaining < 1 {
            return err("counts exceed the table size");
        }
        while remaining < threshold {
            nb_bits -= 1;
            threshold >>= 1;
        }
    }
    if remaining != 1 {
        return err("counts do not sum to the table size");
    }
    Ok(w.finish())
}

/// Standard table-description reader (RFC 8878 §4.1.1), used to check the
/// transcoder against the reference library's own frames.
#[cfg(test)]
pub(crate) fn read_standard_counts(
    buf: &[u8],
    max_symbol: u32,
) -> Result<(NormalizedCounts, usize), ZbicError> {
    let bit =
        |pos: usize| -> u32 { u32::from(buf.get(pos / 8).copied().unwrap_or(0) >> (pos % 8) & 1) };
    let bits =
        |pos: usize, n: u32| -> u32 { (0..n).fold(0, |v, i| v | bit(pos + i as usize) << i) };
    let mut pos = 0usize;
    let table_log = bits(pos, 4) + 5;
    pos += 4;
    if table_log > 15 {
        return err("table log out of range");
    }
    let mut remaining = (1i32 << table_log) + 1;
    let mut threshold = 1i32 << table_log;
    let mut nb_bits = table_log + 1;
    let mut counts: Vec<i16> = Vec::new();
    let mut previous_zero = false;
    while remaining > 1 && counts.len() <= max_symbol as usize {
        if previous_zero {
            loop {
                let r = bits(pos, 2);
                pos += 2;
                counts.extend(std::iter::repeat_n(0, r as usize));
                if r != 3 {
                    break;
                }
            }
        }
        let max = (2 * threshold - 1) - remaining;
        let low = bits(pos, nb_bits - 1) as i32;
        let mut count = if low < max {
            pos += nb_bits as usize - 1;
            low
        } else {
            let mut c = bits(pos, nb_bits) as i32;
            pos += nb_bits as usize;
            if c >= threshold {
                c -= max;
            }
            c
        };
        count -= 1;
        remaining -= count.abs();
        counts.push(count as i16);
        previous_zero = count == 0;
        while remaining < threshold {
            nb_bits -= 1;
            threshold >>= 1;
        }
    }
    let used = pos.div_ceil(8);
    if remaining != 1 || used > buf.len() || counts.len() > max_symbol as usize + 1 {
        return err("corrupt standard table description");
    }
    Ok((NormalizedCounts { counts, table_log }, used))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_table_walks_each_power_of_two_interval() {
        let steps = bic_steps();
        assert_eq!(steps[0], (128, 0, 256));
        assert_eq!(steps[1], (192, 128, 256));
        assert_eq!(steps[7], (255, 254, 256));
        assert_eq!(steps[8], (253, 252, 254));
        assert_eq!(steps[255], (0, 0, 1));
        for slots in [2usize, 4, 8, 16, 32, 64, 128] {
            let tail = &steps[256 - slots..255];
            assert_eq!(tail[0], ((slots / 2) as u16, 0, slots as u16));
            let mut mids: Vec<_> = tail.iter().map(|s| s.0).collect();
            mids.sort_unstable();
            assert_eq!(mids, (1..slots as u16).collect::<Vec<_>>());
        }
    }

    fn corpus() -> Vec<Vec<u8>> {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut rng = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let words = [
            "link", "zelda", "ganon", "ocarina", "temple", "forest", "water", "time", "sheik",
        ];
        let text: Vec<u8> = (0..400_000)
            .flat_map(|_| {
                let w = words[(rng() % words.len() as u64) as usize];
                let mut v = w.as_bytes().to_vec();
                v.push(if rng() % 7 == 0 { b'\n' } else { b' ' });
                v
            })
            .collect();
        let random: Vec<u8> = (0..300_000).map(|_| rng() as u8).collect();
        let skewed: Vec<u8> = (0..300_000)
            .map(|_| (rng() % 1000 + 1).ilog2() as u8 * 17)
            .collect();
        let structured: Vec<u8> = (0..200_000u32)
            .flat_map(|i| (i.wrapping_mul(2_654_435_761) >> (i % 23)).to_le_bytes())
            .collect();
        vec![
            text,
            random,
            skewed,
            structured,
            vec![7; 100_000],
            b"short".to_vec(),
            Vec::new(),
        ]
    }

    /// With standard table coding on both sides the transcoder must reproduce
    /// the reference library's frames exactly, which checks the block,
    /// literals and sequences parsing and the standard table writer.
    #[test]
    fn standard_round_trip_is_byte_identical() {
        let mut total = TranscodeStats::default();
        for data in corpus() {
            for level in [1, 3, 9, 19] {
                let frame = zstd::bulk::compress(&data, level).unwrap();
                let (out, stats) = transcode(&frame, ZSTD_MAGIC, read_standard_counts).unwrap();
                assert_eq!(out, frame, "level {level}, {} bytes", data.len());
                total.blocks += stats.blocks;
                total.huffman_fse_tables += stats.huffman_fse_tables;
                total.sequence_fse_tables += stats.sequence_fse_tables;
            }
        }
        assert!(total.huffman_fse_tables > 10, "{total:?}");
        assert!(total.sequence_fse_tables > 10, "{total:?}");
    }

    #[test]
    fn rejects_malformed_frames() {
        let frame = zstd::bulk::compress(&corpus()[0][..50_000], 3).unwrap();
        assert!(transcode(&frame, ZBIC_MAGIC, read_bic_counts).is_err());
        for cut in [3, 5, 9, frame.len() / 2, frame.len() - 1] {
            assert!(transcode(&frame[..cut], ZSTD_MAGIC, read_standard_counts).is_err());
        }
        let mut longer = frame.clone();
        longer.push(0);
        assert!(transcode(&longer, ZSTD_MAGIC, read_standard_counts).is_err());
        assert!(decompress(b"ZBIC", 0).is_err());
    }

    /// Hand-derived descriptions. The digits (max symbol - 1, table log - 5,
    /// last cumulative value, then one digit per interpolation step) form a
    /// mixed-radix number whose bytes are read from the end of the field.
    #[test]
    fn bic_reader_decodes_hand_built_descriptions() {
        // Counts [16, 16], table log 5: digits 0 (radix 52), 0 (8), 31 (32),
        // 16 (33), 0 (17) -> 52 * 8 * (31 + 32 * 16) = 0x37260.
        let (n, used) = read_bic_counts(&[3, 0x60, 0x72, 0x03, 0xee], 35).unwrap();
        assert_eq!(
            n,
            NormalizedCounts {
                counts: vec![16, 16],
                table_log: 5
            }
        );
        assert_eq!(used, 4);

        // Counts [-1, 31] with the low-probability flag: last = 1 + 29 + 2,
        // one interpolation digit 0 -> 52 * 8 * 29 = 0x2f20.
        let (n, used) = read_bic_counts(&[0x82, 0x20, 0x2f, 0xee], 35).unwrap();
        assert_eq!(
            n,
            NormalizedCounts {
                counts: vec![-1, 31],
                table_log: 5
            }
        );
        assert_eq!(used, 3);

        // The symbol limit is enforced.
        assert!(read_bic_counts(&[3, 0x60, 0x72, 0x03, 0xee], 0).is_err());

        // Both re-encode to standard descriptions that read back unchanged.
        for counts in [vec![16, 16], vec![-1, 31], vec![0, 0, 7, -1, 0, 24]] {
            let n = NormalizedCounts {
                counts,
                table_log: 5,
            };
            let mut std = write_standard_counts(&n).unwrap();
            let len = std.len();
            std.push(0);
            assert_eq!(read_standard_counts(&std, 255).unwrap(), (n, len));
        }
    }

    #[test]
    fn bic_reader_rejects_bad_descriptions() {
        assert!(read_bic_counts(&[], 35).is_err());
        // Claims 5 payload bytes but the section has only 3.
        assert!(read_bic_counts(&[5, 1, 2], 35).is_err());
        // All-zero payload decodes to counts that do not fill the table.
        assert!(read_bic_counts(&[2, 0, 0, 0], 35).is_err());
    }
}
