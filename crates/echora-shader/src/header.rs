//! The shader program header (SPH) in front of a graphics program: the
//! first 0x50 bytes of its 128-byte header, in the Maxwell layout. Field
//! positions are facts from the SPH encoder of Mesa's NAK (MIT,
//! `src/nouveau/compiler/nak/sph.rs` at `979c6d1`).

use echora_shader_abi::Stage;

/// Bytes of the header in front of a graphics program's code.
pub const SIZE: usize = 128;

/// How a fragment input component is interpolated (the SPH's pixel input
/// map, two bits per component).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interpolation {
    Unused,
    /// Flat: the provoking vertex's value.
    Constant,
    Perspective,
    /// Linear in screen space (no perspective correction).
    ScreenLinear,
}

/// What a translation needs from the header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub stage: Stage,
    /// Fragment: the input map, component `4 * vector + c` of the 32
    /// generic vectors.
    pub imap: [Interpolation; 128],
    /// Fragment: four component bits per render target (eight targets).
    pub targets: u32,
    pub writes_depth: bool,
    pub writes_sample_mask: bool,
    /// Local memory bytes per invocation: common word 1's low 24 bits
    /// (ShaderLocalMemoryLowSize); `None` when common word 2's high size
    /// (bits 32 and up of the size) is not zero.
    pub local_bytes: Option<u32>,
}

fn bits(bytes: &[u8], offset: usize, count: usize) -> u32 {
    let mut value = 0;
    for bit in 0..count {
        let at = offset + bit;
        if bytes[at / 8] >> (at % 8) & 1 != 0 {
            value |= 1 << bit;
        }
    }
    value
}

/// Parses the header at the start of `program`, or `None` when it is not
/// a VTG (type 1) or pixel (type 2) header with a known stage.
pub fn parse(program: &[u8]) -> Option<Header> {
    let bytes = program.get(..0x50)?;
    let sph_type = bits(bytes, 0, 5);
    let stage = match (sph_type, bits(bytes, 10, 4)) {
        (1, 1) => Stage::Vertex,
        (1, 2) => Stage::TessControl,
        (1, 3) => Stage::TessEval,
        (1, 4) => Stage::Geometry,
        (2, 5) => Stage::Fragment,
        _ => return None,
    };
    let mut imap = [Interpolation::Unused; 128];
    if stage == Stage::Fragment {
        for (n, entry) in imap.iter_mut().enumerate() {
            *entry = match bits(bytes, 192 + n * 2, 2) {
                0 => Interpolation::Unused,
                1 => Interpolation::Constant,
                2 => Interpolation::Perspective,
                _ => Interpolation::ScreenLinear,
            };
        }
    }
    let pixel = stage == Stage::Fragment;
    Some(Header {
        stage,
        imap,
        targets: if pixel { bits(bytes, 576, 32) } else { 0 },
        writes_sample_mask: pixel && bits(bytes, 608, 1) != 0,
        writes_depth: pixel && bits(bytes, 609, 1) != 0,
        local_bytes: (bits(bytes, 64, 24) == 0).then(|| bits(bytes, 32, 24)),
    })
}

/// A program's stage, by its first word: graphics programs start with a
/// header, compute programs with an instruction (whose guard bits, 12-14,
/// put 12 or more in the header's shader type field for a PT guard).
pub fn is_graphics(program: &[u8]) -> bool {
    parse(program).is_some()
}
