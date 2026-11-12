//! De-tile a raw ASTC texture dump (ECHORA_GPU_RAW_DUMP) with each block
//! height and count the blocks that decode as errors.
//!
//! `cargo run --release -p echora-runtime --example astc_layout -- RAW
//! WIDTH HEIGHT BLOCK_W BLOCK_H [OUT_DIR]` prints, per block height (GOBs,
//! log2), the error texels of level 0; with OUT_DIR each decode is written
//! as `y<N>.ppm`. The console's layout leaves none.

use echora_runtime::nvn_format::{BlockShape, FormatInfo, block_linear_offset_in};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [raw, width, height, block_width, block_height] = &args[..5] else {
        panic!("usage: astc_layout RAW WIDTH HEIGHT BLOCK_W BLOCK_H [OUT_DIR]");
    };
    let raw = std::fs::read(raw).expect("read the dump");
    let number = |text: &String| text.parse::<u32>().expect("a number");
    let (width, height) = (number(width), number(height));
    let format = FormatInfo {
        name: "ASTC",
        block_width: number(block_width),
        block_height: number(block_height),
        bytes_per_block: 16,
    };
    let (columns, rows) = (
        width.div_ceil(format.block_width),
        height.div_ceil(format.block_height),
    );
    for y_shift in 0..=5 {
        let shape = BlockShape { y: y_shift, z: 0 };
        let mut packed = vec![0u8; (columns * rows * 16) as usize];
        let mut outside = 0;
        for row in 0..rows {
            for column in 0..columns {
                let offset =
                    block_linear_offset_in(format, width, height, shape, column, row, 0) as usize;
                let at = ((row * columns + column) * 16) as usize;
                match raw.get(offset..offset + 16) {
                    Some(block) => packed[at..at + 16].copy_from_slice(block),
                    None => outside += 1,
                }
            }
        }
        let mut rgba = vec![0u8; (width * height * 4) as usize];
        let footprint = astc_decode::Footprint::new(format.block_width, format.block_height);
        let _ = astc_decode::astc_decode(&packed[..], width, height, footprint, |x, y, texel| {
            let at = ((y * width + x) * 4) as usize;
            rgba[at..at + 4].copy_from_slice(&texel);
        });
        let errors = rgba
            .chunks(4)
            .filter(|texel| *texel == [255, 0, 255, 255])
            .count();
        println!(
            "block height {:2} GOBs: {errors} error texels ({:.2}%), {outside} blocks outside the dump",
            1 << y_shift,
            100.0 * errors as f64 / f64::from(width * height)
        );
        if let Some(directory) = args.get(5) {
            let mut ppm = format!("P6\n{width} {height}\n255\n").into_bytes();
            ppm.extend(rgba.chunks(4).flat_map(|texel| texel[..3].to_vec()));
            std::fs::write(format!("{directory}/y{y_shift}.ppm"), ppm).expect("write");
        }
    }
}
