//! De-tile a raw BC1-BC5 texture dump (ECHORA_GPU_RAW_DUMP) with each block
//! height and measure how well its 4x4 blocks join.
//!
//! `cargo run --release -p echora-runtime --example bc_layout -- RAW WIDTH
//! HEIGHT FORMAT [OUT_DIR]` with FORMAT one of bc1, bc2, bc3, bc4, bc5
//! prints, per block height (GOBs, log2), the mean brightness step across
//! block edges ("seam") beside the mean step inside blocks ("inner") for
//! level 0; with OUT_DIR each decode is written as `y<N>.pgm`. In the
//! console's layout neighbouring blocks continue each other, so the seam
//! step is close to the inner one; a wrong block height puts unrelated
//! blocks side by side and the seam step is several times larger.

use echora_runtime::nvn_format::{BlockShape, FormatInfo, block_linear_offset_in};

/// Reads one 8-byte block as 16 brightness values.
type Decode = fn(&[u8]) -> [u8; 16];

/// A BC4 block's 16 values (also BC3's and BC5's alpha blocks).
fn bc4(block: &[u8]) -> [u8; 16] {
    let (a, b) = (u32::from(block[0]), u32::from(block[1]));
    let mut palette = [a as u8, b as u8, 0, 0, 0, 0, 0, 0];
    if a > b {
        for step in 1..7 {
            palette[step as usize + 1] = (((7 - step) * a + step * b) / 7) as u8;
        }
    } else {
        for step in 1..5 {
            palette[step as usize + 1] = (((5 - step) * a + step * b) / 5) as u8;
        }
        palette[7] = 255;
    }
    let bits = block[2..8]
        .iter()
        .rev()
        .fold(0u64, |bits, &byte| bits << 8 | u64::from(byte));
    std::array::from_fn(|texel| palette[(bits >> (3 * texel) & 7) as usize])
}

/// A BC1 colour block's 16 texels as brightness (also BC2's and BC3's
/// colour blocks).
fn bc1(block: &[u8]) -> [u8; 16] {
    let colour = |at: usize| {
        let value = u32::from(u16::from_le_bytes([block[at], block[at + 1]]));
        // 5:6:5 to a brightness in 0..=255.
        let (r, g, b) = (value >> 11, value >> 5 & 0x3f, value & 0x1f);
        (r * 255 / 31 * 3 + g * 255 / 63 * 6 + b * 255 / 31) / 10
    };
    let (a, b) = (colour(0), colour(2));
    let ordered =
        u16::from_le_bytes([block[0], block[1]]) > u16::from_le_bytes([block[2], block[3]]);
    let palette = if ordered {
        [a, b, (2 * a + b) / 3, (a + 2 * b) / 3]
    } else {
        [a, b, (a + b) / 2, 0]
    };
    let bits = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    std::array::from_fn(|texel| palette[(bits >> (2 * texel) & 3) as usize] as u8)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [raw, width, height, kind] = &args[..4] else {
        panic!("usage: bc_layout RAW WIDTH HEIGHT bc1|bc2|bc3|bc4|bc5 [OUT_DIR]");
    };
    let raw = std::fs::read(raw).expect("read the dump");
    let number = |text: &String| text.parse::<u32>().expect("a number");
    let (width, height) = (number(width), number(height));
    // Bytes a block, and where its brightness block starts and how to read it.
    let (bytes, at, decode): (u32, usize, Decode) = match kind.as_str() {
        "bc1" => (8, 0, bc1),
        "bc2" | "bc3" => (16, 8, bc1),
        "bc4" => (8, 0, bc4),
        "bc5" => (16, 0, bc4),
        other => panic!("format {other}"),
    };
    let format = FormatInfo {
        name: "BC",
        block_width: 4,
        block_height: 4,
        bytes_per_block: bytes,
    };
    let (columns, rows) = (width.div_ceil(4), height.div_ceil(4));
    for y_shift in 0..=5 {
        let shape = BlockShape { y: y_shift, z: 0 };
        let (padded_width, padded_height) = ((columns * 4) as usize, (rows * 4) as usize);
        let mut image = vec![0u8; padded_width * padded_height];
        let mut outside = 0;
        for row in 0..rows {
            for column in 0..columns {
                let offset =
                    block_linear_offset_in(format, width, height, shape, column, row, 0) as usize;
                let Some(block) = raw.get(offset..offset + bytes as usize) else {
                    outside += 1;
                    continue;
                };
                let texels = decode(&block[at..at + 8]);
                for (index, texel) in texels.into_iter().enumerate() {
                    let (x, y) = (
                        column as usize * 4 + index % 4,
                        row as usize * 4 + index / 4,
                    );
                    image[y * padded_width + x] = texel;
                }
            }
        }
        // Steps between horizontal and vertical neighbours, split by
        // whether the pair straddles a block edge.
        let (mut seam, mut seams, mut inner, mut inners) = (0u64, 0u64, 0u64, 0u64);
        let step = |a: u8, b: u8| u64::from(a.abs_diff(b));
        for y in 0..padded_height {
            for x in 0..padded_width {
                let here = image[y * padded_width + x];
                if x + 1 < padded_width {
                    let value = step(here, image[y * padded_width + x + 1]);
                    if x % 4 == 3 {
                        seam += value;
                        seams += 1;
                    } else {
                        inner += value;
                        inners += 1;
                    }
                }
                if y + 1 < padded_height {
                    let value = step(here, image[(y + 1) * padded_width + x]);
                    if y % 4 == 3 {
                        seam += value;
                        seams += 1;
                    } else {
                        inner += value;
                        inners += 1;
                    }
                }
            }
        }
        println!(
            "block height {:2} GOBs: seam {:.2} inner {:.2} (ratio {:.2}), {outside} blocks outside the dump",
            1 << y_shift,
            seam as f64 / seams.max(1) as f64,
            inner as f64 / inners.max(1) as f64,
            (seam as f64 / seams.max(1) as f64) / (inner as f64 / inners.max(1) as f64).max(1e-9)
        );
        if let Some(directory) = args.get(4) {
            let mut pgm = format!("P5\n{padded_width} {padded_height}\n255\n").into_bytes();
            pgm.extend(&image);
            std::fs::write(format!("{directory}/y{y_shift}.pgm"), pgm).expect("write");
        }
    }
}
