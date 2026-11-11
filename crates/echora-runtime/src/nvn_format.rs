//! NVN texture formats and storage sizes.
//!
//! NVN's format numbering is not publicly documented. The application's
//! nn::gfx layer converts its own ImageFormat values to NVN formats through a
//! sorted table in the main module; nn::gfx ImageFormat values are
//! `(channel format << 8) | type format`, the encoding of BNTX texture files
//! documented by the Switch modding community. Each NVN number below is the
//! table's image of a known nn::gfx format, so its meaning is derived, not
//! guessed.
//!
//! Storage sizes follow the Tegra block-linear layout (64-byte x 8-row GOBs
//! stacked into blocks of up to 16 GOBs per mip level), as described by
//! NVIDIA's open-gpu-doc texture headers and implemented by public Switch
//! emulators. The application writes pre-tiled packaged texture data into
//! this storage, so sizes must match the hardware layout.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FormatInfo {
    pub name: &'static str,
    pub block_width: u32,
    pub block_height: u32,
    pub bytes_per_block: u32,
}

const fn f(name: &'static str, block_width: u32, block_height: u32, bytes: u32) -> FormatInfo {
    FormatInfo {
        name,
        block_width,
        block_height,
        bytes_per_block: bytes,
    }
}

/// ASTC block footprints for nn::gfx channel formats 0x2d..=0x3a, in order.
const ASTC: [(u32, u32); 14] = [
    (4, 4),
    (5, 4),
    (5, 5),
    (6, 5),
    (6, 6),
    (8, 5),
    (8, 6),
    (8, 8),
    (10, 5),
    (10, 6),
    (10, 8),
    (10, 10),
    (12, 10),
    (12, 12),
];

/// The NVN format number's layout, if it is one the application uses.
pub fn info(format: u32) -> Option<FormatInfo> {
    Some(match format {
        1 => f("R8", 1, 1, 1),
        2 => f("R8SN", 1, 1, 1),
        3 => f("R8UI", 1, 1, 1),
        4 => f("R8I", 1, 1, 1),
        5 => f("R16F", 1, 1, 2),
        6 => f("R16", 1, 1, 2),
        7 => f("R16SN", 1, 1, 2),
        8 => f("R16UI", 1, 1, 2),
        9 => f("R16I", 1, 1, 2),
        10 => f("R32F", 1, 1, 4),
        11 => f("R32UI", 1, 1, 4),
        12 => f("R32I", 1, 1, 4),
        13 => f("RG8", 1, 1, 2),
        14 => f("RG8SN", 1, 1, 2),
        15 => f("RG8UI", 1, 1, 2),
        16 => f("RG8I", 1, 1, 2),
        17 => f("RG16F", 1, 1, 4),
        18 => f("RG16", 1, 1, 4),
        19 => f("RG16SN", 1, 1, 4),
        20 => f("RG16UI", 1, 1, 4),
        21 => f("RG16I", 1, 1, 4),
        22 => f("RG32F", 1, 1, 8),
        23 => f("RG32UI", 1, 1, 8),
        24 => f("RG32I", 1, 1, 8),
        34 => f("RGB32F", 1, 1, 12),
        35 => f("RGB32UI", 1, 1, 12),
        36 => f("RGB32I", 1, 1, 12),
        37 => f("RGBA8", 1, 1, 4),
        38 => f("RGBA8SN", 1, 1, 4),
        39 => f("RGBA8UI", 1, 1, 4),
        40 => f("RGBA8I", 1, 1, 4),
        41 => f("RGBA16F", 1, 1, 8),
        42 => f("RGBA16", 1, 1, 8),
        43 => f("RGBA16SN", 1, 1, 8),
        44 => f("RGBA16UI", 1, 1, 8),
        45 => f("RGBA16I", 1, 1, 8),
        46 => f("RGBA32F", 1, 1, 16),
        47 => f("RGBA32UI", 1, 1, 16),
        48 => f("RGBA32I", 1, 1, 16),
        50 => f("DEPTH16", 1, 1, 2),
        52 => f("DEPTH32F", 1, 1, 4),
        53 => f("DEPTH24_STENCIL8", 1, 1, 4),
        54 => f("DEPTH32F_STENCIL8", 1, 1, 8),
        56 => f("RGBA8_SRGB", 1, 1, 4),
        57 => f("RGBA4", 1, 1, 2),
        59 => f("RGB5A1", 1, 1, 2),
        60 => f("RGB565", 1, 1, 2),
        61 => f("RGB10A2", 1, 1, 4),
        62 => f("RGB10A2UI", 1, 1, 4),
        63 => f("R11G11B10F", 1, 1, 4),
        64 => f("RGB9E5F", 1, 1, 4),
        66 => f("BC1_UNORM", 4, 4, 8),
        67 => f("BC2_UNORM", 4, 4, 16),
        68 => f("BC3_UNORM", 4, 4, 16),
        70 => f("BC1_SRGB", 4, 4, 8),
        71 => f("BC2_SRGB", 4, 4, 16),
        72 => f("BC3_SRGB", 4, 4, 16),
        73 => f("BC4_UNORM", 4, 4, 8),
        74 => f("BC4_SNORM", 4, 4, 8),
        75 => f("BC5_UNORM", 4, 4, 16),
        76 => f("BC5_SNORM", 4, 4, 16),
        77 => f("BC7_UNORM", 4, 4, 16),
        78 => f("BC7_SRGB", 4, 4, 16),
        79 => f("BC6H_SFLOAT", 4, 4, 16),
        80 => f("BC6H_UFLOAT", 4, 4, 16),
        // After the 24 UI2F/I2F formats (81-104); 107 and 108 are their
        // RGB10A2 forms, 109-120 the RGBX formats.
        105 => f("RGB10A2SN", 1, 1, 4),
        106 => f("RGB10A2I", 1, 1, 4),
        121..=134 => {
            let (w, h) = ASTC[(format - 121) as usize];
            f("ASTC", w, h, 16)
        }
        135..=148 => {
            let (w, h) = ASTC[(format - 135) as usize];
            f("ASTC_SRGB", w, h, 16)
        }
        149 => f("BGR565", 1, 1, 2),
        151 => f("BGR5A1", 1, 1, 2),
        152 => f("A1BGR5", 1, 1, 2),
        154 => f("BGRA8", 1, 1, 4),
        156 => f("BGRA8_SRGB", 1, 1, 4),
        _ => return None,
    })
}

const GOB_WIDTH: u64 = 64;
const GOB_HEIGHT: u64 = 8;
const GOB_SIZE: u64 = GOB_WIDTH * GOB_HEIGHT;
/// Shrink a block's GOB count (as log2 `shift`) while half of it still
/// covers `extent` (rows of blocks for height, slices for depth).
fn fit_block(mut shift: u32, unit: u64, extent: u64) -> u32 {
    while shift > 0 && (unit << (shift - 1)) >= extent {
        shift -= 1;
    }
    shift
}

fn align(value: u64, alignment: u64) -> u64 {
    value.div_ceil(alignment) * alignment
}

/// GOBs per block in y and z, as log2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockShape {
    pub y: u32,
    pub z: u32,
}

/// The block shape the console's NVN driver gives level 0 (no packaged
/// layout): 2D textures by 1.5 times their rows of format blocks against
/// the thresholds 16/32/64/128 GOB rows; 3D textures one GOB high and 16
/// deep, halved while the depth fits in half (rules from a public
/// recompilation of this build, checked there against the game's own
/// texture data).
pub fn base_block_shape(format: FormatInfo, height: u32, depth: u32) -> BlockShape {
    let rows = u64::from(height.max(1).div_ceil(format.block_height));
    if depth > 1 {
        let mut z = 4;
        while z > 0 && u64::from(depth) <= (1 << z) >> 1 {
            z -= 1;
        }
        return BlockShape { y: 0, z };
    }
    let scaled = rows + rows / 2;
    let y = match scaled {
        128.. => 4,
        64.. => 3,
        32.. => 2,
        16.. => 1,
        _ => 0,
    };
    BlockShape { y, z: 0 }
}

/// The block shape of mip `level`: a level's block height halves while its
/// rows fit in half the block; its depth halves once per level when its
/// slices fit in half.
pub fn level_block_shape(format: FormatInfo, height: u32, depth: u32, level: u32) -> BlockShape {
    let base = base_block_shape(format, height, depth);
    let mut z = base.z;
    for step in 1..=level {
        let d = u64::from((depth >> step).max(1));
        if z > 0 && d <= (1 << z) >> 1 {
            z -= 1;
        }
    }
    let rows = u64::from((height >> level).max(1).div_ceil(format.block_height));
    BlockShape {
        y: fit_block(base.y, GOB_HEIGHT, rows),
        z,
    }
}

/// Block-linear storage size of a texture with `layers` array layers (1 for
/// non-arrays), and the block-linear alignment (one GOB).
pub fn block_linear_size(
    format: FormatInfo,
    width: u32,
    height: u32,
    depth: u32,
    layers: u32,
    levels: u32,
) -> (u64, u64) {
    let blocks = |extent: u32, block: u32| u64::from(extent.max(1).div_ceil(block));
    let base = base_block_shape(format, height, depth);
    let (height_shift, depth_shift) = (base.y, base.z);
    let mut layer_size = 0;
    for level in 0..levels.max(1) {
        let w = (width >> level).max(1);
        let h = (height >> level).max(1);
        let d = (depth >> level).max(1);
        let rows = blocks(h, format.block_height);
        let shape = level_block_shape(format, height, depth, level);
        let (level_height_shift, level_depth_shift) = (shape.y, shape.z);
        let row_bytes = align(
            blocks(w, format.block_width) * u64::from(format.bytes_per_block),
            GOB_WIDTH,
        );
        let block_rows = GOB_HEIGHT << level_height_shift;
        let block_depth = 1u64 << level_depth_shift;
        layer_size += row_bytes * align(rows, block_rows) * align(u64::from(d), block_depth);
    }
    // Array layers start on a whole level-0 block.
    let block = GOB_SIZE << (height_shift + depth_shift);
    let layer_stride = if layers > 1 {
        align(layer_size, block)
    } else {
        layer_size
    };
    (layer_stride * u64::from(layers.max(1)), GOB_SIZE)
}

/// Byte offset of mip level `level` within a block-linear surface (levels
/// follow each other, each padded to its own block size).
pub fn block_linear_level_offset(
    format: FormatInfo,
    width: u32,
    height: u32,
    depth: u32,
    level: u32,
) -> u64 {
    if level == 0 {
        return 0;
    }
    block_linear_size(format, width, height, depth, 1, level).0
}

/// Byte offset of block (`x`, `y`, `z`) (in format blocks; `z` in slices)
/// within level 0 of a block-linear surface `width` x `height` x `depth`
/// texels: GOBs of 64 bytes x 8 rows stacked vertically, then in depth, into
/// blocks of the height and depth `block_linear_size` chooses; blocks laid
/// out left to right, then top to bottom, then front to back. Within a GOB,
/// bytes follow the Switch 2 GPU's swizzle: the Tegra X1 layout (NVIDIA
/// open-gpu-doc) with bits 5 and 6 of the in-GOB offset swapped (Eden-NS2
/// `NativeGobOffset`; confirmed here on the loading screen's BC7 sprite
/// sheet, which de-tiles cleanly only this way).
pub fn block_linear_offset(
    format: FormatInfo,
    width: u32,
    height: u32,
    depth: u32,
    x: u32,
    y: u32,
    z: u32,
) -> u64 {
    let shape = base_block_shape(format, height, depth);
    block_linear_offset_in(format, width, height, shape, x, y, z)
}

/// How many blocks from column `x` of a row lie at consecutive addresses:
/// within a GOB, 16-byte runs of a row are contiguous, so blocks whose size
/// divides 16 run to the next 16-byte boundary (at least one block).
pub fn block_linear_run(format: FormatInfo, x: u32) -> u32 {
    let bytes = format.bytes_per_block;
    if bytes == 0 || 16 % bytes != 0 {
        return 1;
    }
    (16 - (x * bytes) % 16) / bytes
}

/// As `block_linear_offset`, within a level `width` x `height` texels
/// whose blocks have `shape` (see `level_block_shape`).
pub fn block_linear_offset_in(
    format: FormatInfo,
    width: u32,
    height: u32,
    shape: BlockShape,
    x: u32,
    y: u32,
    z: u32,
) -> u64 {
    let rows = u64::from(height.max(1).div_ceil(format.block_height));
    let (gobs_y, gobs_z) = (1u64 << shape.y, 1u64 << shape.z);
    let block_size = GOB_SIZE * gobs_y * gobs_z;
    let row_bytes = align(
        u64::from(width.max(1).div_ceil(format.block_width)) * u64::from(format.bytes_per_block),
        GOB_WIDTH,
    );
    let blocks_x = row_bytes / GOB_WIDTH;
    let blocks_y = rows.div_ceil(GOB_HEIGHT * gobs_y);
    let byte_x = u64::from(x) * u64::from(format.bytes_per_block);
    let (y, z) = (u64::from(y), u64::from(z));
    let (gob_x, gob_y) = (byte_x / GOB_WIDTH, y / GOB_HEIGHT);
    let block = (z / gobs_z * blocks_y + gob_y / gobs_y) * blocks_x + gob_x;
    let gob_in_block = (z % gobs_z) * gobs_y + gob_y % gobs_y;
    let (bx, by) = (byte_x % GOB_WIDTH, y % GOB_HEIGHT);
    let within = (bx / 32) * 256
        + (by / 4) * 128
        + ((bx % 32) / 16) * 64
        + ((by % 4) / 2) * 32
        + (by % 2) * 16
        + bx % 16;
    block * block_size + gob_in_block * GOB_SIZE + within
}

/// The 16-byte runs of row `y` (in blocks) of slice `z`, as
/// (byte offset in the row, byte offset in the level), for formats whose
/// block size divides 16: the same offsets as `block_linear_offset_in` at
/// every 16th byte, with the row's block and GOB worked out once. Within a
/// GOB a row's four runs are 64 and 256 bytes apart (see
/// `block_linear_offset`).
pub fn block_linear_row_runs(
    format: FormatInfo,
    width: u32,
    height: u32,
    shape: BlockShape,
    y: u32,
    z: u32,
) -> impl Iterator<Item = (u64, u64)> {
    let rows = u64::from(height.max(1).div_ceil(format.block_height));
    let (gobs_y, gobs_z) = (1u64 << shape.y, 1u64 << shape.z);
    let block_size = GOB_SIZE * gobs_y * gobs_z;
    let used =
        u64::from(width.max(1).div_ceil(format.block_width)) * u64::from(format.bytes_per_block);
    let blocks_x = align(used, GOB_WIDTH) / GOB_WIDTH;
    let blocks_y = rows.div_ceil(GOB_HEIGHT * gobs_y);
    let (y, z) = (u64::from(y), u64::from(z));
    let gob_y = y / GOB_HEIGHT;
    let first_block = (z / gobs_z * blocks_y + gob_y / gobs_y) * blocks_x;
    let gob_in_block = (z % gobs_z) * gobs_y + gob_y % gobs_y;
    let by = y % GOB_HEIGHT;
    let row_in_gob = (by / 4) * 128 + ((by % 4) / 2) * 32 + (by % 2) * 16;
    let start = first_block * block_size + gob_in_block * GOB_SIZE + row_in_gob;
    (0..used.div_ceil(16)).map(move |run| {
        let gob = start + run / 4 * block_size;
        let within = (run % 4 / 2) * 256 + (run % 2) * 64;
        (run * 16, gob + within)
    })
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_runs_match_block_offsets() {
        // 1-, 4-, 8- and 16-byte blocks, uneven sizes, deep and tall blocks.
        for format in [1, 37, 46, 66].into_iter().filter_map(info) {
            for (width, height, depth) in [(100, 37, 1), (300, 200, 1), (64, 64, 9), (5, 3, 1)] {
                for level in 0..3 {
                    let (w, h, d) = (
                        (width >> level).max(1),
                        (height >> level).max(1),
                        (depth >> level).max(1),
                    );
                    let shape = level_block_shape(format, height, depth, level);
                    let rows = h.div_ceil(format.block_height);
                    for z in 0..d {
                        for y in 0..rows {
                            for (byte, offset) in block_linear_row_runs(format, w, h, shape, y, z) {
                                let x = (byte / u64::from(format.bytes_per_block)) as u32;
                                assert_eq!(
                                    offset,
                                    block_linear_offset_in(format, w, h, shape, x, y, z),
                                    "format {} {w}x{h}x{d} block ({x}, {y}, {z})",
                                    format.name
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn runs_are_contiguous_to_the_next_16_bytes() {
        let format = info(37).unwrap();
        let bytes = format.bytes_per_block;
        for x in 0..96u32 {
            let run = block_linear_run(format, x);
            let start = block_linear_offset(format, 96, 32, 1, x, 5, 0);
            for step in 0..run {
                assert_eq!(
                    block_linear_offset(format, 96, 32, 1, x + step, 5, 0),
                    start + u64::from(step * bytes)
                );
            }
        }
    }

    #[test]
    fn mip_levels_follow_each_other() {
        // BC7 1024x2048: level 0 is 256x512 blocks of 16 bytes, an exact
        // number of 16-GOB blocks; level 1 starts right after it.
        let format = FormatInfo {
            name: "BC7_UNORM",
            block_width: 4,
            block_height: 4,
            bytes_per_block: 16,
        };
        assert_eq!(block_linear_level_offset(format, 1024, 2048, 1, 0), 0);
        assert_eq!(
            block_linear_level_offset(format, 1024, 2048, 1, 1),
            256 * 16 * 512
        );
        let two = block_linear_size(format, 1024, 2048, 1, 1, 2).0;
        assert_eq!(block_linear_level_offset(format, 1024, 2048, 1, 2), two);
    }

    #[test]
    fn formats_follow_the_nn_gfx_table() {
        assert_eq!(info(37).unwrap().bytes_per_block, 4);
        assert_eq!(info(66).unwrap().bytes_per_block, 8);
        assert_eq!(info(128).unwrap().block_width, 8);
        assert_eq!(info(148).unwrap().block_height, 12);
        assert!(info(0).is_none());
    }

    #[test]
    fn block_linear_sizes() {
        let rgba8 = info(37).unwrap();
        // 1280x720 RGBA8: 5120-byte rows; 720 rows fit a 16-GOB (128-row)
        // block height, so rows round up to 768.
        assert_eq!(block_linear_size(rgba8, 1280, 720, 1, 1, 1).0, 5120 * 768);
        // A 4x4 texture still occupies one GOB (64 bytes x 8 rows).
        assert_eq!(block_linear_size(rgba8, 4, 4, 1, 1, 1).0, 512);
        // 16x16 BC1 is 4x4 blocks: one GOB.
        assert_eq!(block_linear_size(info(66).unwrap(), 16, 16, 1, 1, 1).0, 512);
        // Switch 2 swizzle: byte 16 of row 0 is 64 bytes in; row 1 starts
        // 16 bytes in, row 2 at 32 and row 4 at 128; byte 32 of row 0 opens
        // the second half-GOB.
        assert_eq!(block_linear_offset(rgba8, 64, 8, 1, 4, 0, 0), 64);
        assert_eq!(block_linear_offset(rgba8, 64, 8, 1, 0, 1, 0), 16);
        assert_eq!(block_linear_offset(rgba8, 64, 8, 1, 8, 0, 0), 256);
        assert_eq!(block_linear_offset(rgba8, 64, 8, 1, 0, 2, 0), 32);
        assert_eq!(block_linear_offset(rgba8, 64, 8, 1, 0, 4, 0), 128);
        assert_eq!(
            block_linear_offset(rgba8, 64, 8, 1, 12, 7, 0),
            256 + 128 + 64 + 32 + 16
        );
        // 64x16 RGBA8: 256-byte rows are four GOBs wide; the block holds two
        // GOBs vertically, so row 8 is the second GOB of the first block.
        assert_eq!(block_linear_offset(rgba8, 64, 16, 1, 0, 8, 0), 512);
        assert_eq!(block_linear_offset(rgba8, 64, 16, 1, 16, 0, 0), 1024);
        // 16x8x4 RGBA8: one GOB per slice, four slices per block (depth
        // shift 2), so slice 2 is the third GOB of the only block.
        assert_eq!(block_linear_offset(rgba8, 16, 8, 4, 0, 0, 2), 1024);
        assert_eq!(block_linear_size(rgba8, 16, 8, 4, 1, 1).0, 2048);
        // Two layers of a 4x4 texture: each layer starts on a GOB.
        assert_eq!(block_linear_size(rgba8, 4, 4, 1, 2, 1).0, 1024);
    }
}
