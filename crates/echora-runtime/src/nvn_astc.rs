//! Reference decoding of software sRGB ASTC uploads.

use ctt_astcenc as arm;
use std::cell::RefCell;
use std::sync::Mutex;

type Footprint = [u32; 2];
const MAX_IDLE_CONTEXTS: usize = 8;
static CONTEXTS: Mutex<Vec<(Footprint, arm::Context)>> = Mutex::new(Vec::new());

thread_local! {
    static COMPONENTS: RefCell<Vec<[u16; 4]>> = const { RefCell::new(Vec::new()) };
}

/// Recover a UNORM8 code from its half-float representation. All 256
/// quantized codes round back to themselves after conversion to half.
static UNORM8: [u8; 0x3c01] = {
    let mut table = [0; 0x3c01];
    let mut bits = 0x1800;
    while bits < table.len() {
        let shift = 25 - (bits >> 10);
        let mantissa = (bits & 0x03ff) + 0x0400;
        table[bits] = ((mantissa * 255 + (1 << (shift - 1))) >> shift) as u8;
        bits += 1;
    }
    table
};

#[cfg(target_arch = "x86_64")]
pub(super) fn available() -> bool {
    // The bindings choose AVX2 or SSE4.1 using the primary feature alone.
    // Their respective build variants also enable F16C and/or POPCNT.
    let popcnt = std::is_x86_feature_detected!("popcnt");
    if std::is_x86_feature_detected!("avx2") {
        popcnt && std::is_x86_feature_detected!("f16c")
    } else {
        !std::is_x86_feature_detected!("sse4.1") || popcnt
    }
}

#[cfg(target_arch = "aarch64")]
pub(super) fn available() -> bool {
    true
}

pub(super) fn decode(
    blocks: &[u8],
    output: &mut [u8],
    [width, height]: [u32; 2],
    footprint: Footprint,
) -> std::io::Result<()> {
    assert_eq!(output.len(), width as usize * height as usize * 4);
    // Reuse each footprint's tables across decoded layers, mip levels and
    // textures. Only take the lock for transfers of an idle context.
    let cached = {
        let mut idle = CONTEXTS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        idle.iter()
            .rposition(|(key, _)| *key == footprint)
            .map(|index| idle.remove(index).1)
    };
    let mut context = match cached {
        Some(context) => context,
        None => {
            let config = arm::config_init(
                arm::astcenc_profile_ASTCENC_PRF_LDR_SRGB,
                footprint[0],
                footprint[1],
                1,
                arm::ASTCENC_PRE_FASTEST,
                arm::ASTCENC_FLG_DECOMPRESS_ONLY,
            )
            .map_err(std::io::Error::other)?;
            arm::Context::new(&config).map_err(std::io::Error::other)?
        }
    };
    let rgba = arm::astcenc_swizzle {
        r: arm::astcenc_swz_ASTCENC_SWZ_R,
        g: arm::astcenc_swz_ASTCENC_SWZ_G,
        b: arm::astcenc_swz_ASTCENC_SWZ_B,
        a: arm::astcenc_swz_ASTCENC_SWZ_A,
    };
    let result = COMPONENTS.with_borrow_mut(|components| -> Result<(), arm::Error> {
        // This astcenc release's AVX2 UNORM8 packing misses NaN error
        // markers. The sRGB profile quantizes every valid channel to UNORM8
        // before storing it as a half and preserves error markers there.
        components.resize(output.len() / 4, [0; 4]);
        let mut slices = [components.as_mut_ptr().cast::<std::ffi::c_void>()];
        let mut image = arm::astcenc_image {
            dim_x: width,
            dim_y: height,
            dim_z: 1,
            data_type: arm::astcenc_type_ASTCENC_TYPE_F16,
            data: slices.as_mut_ptr(),
        };
        context.decompress(blocks, &mut image, &rgba)?;
        for (pixel, &[r, g, b, a]) in output
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(components.iter())
        {
            *pixel = if r & 0x7fff > 0x7c00 {
                [255, 0, 255, 255]
            } else {
                [
                    UNORM8[r as usize],
                    UNORM8[g as usize],
                    UNORM8[b as usize],
                    UNORM8[a as usize],
                ]
            };
        }
        Ok(())
    });
    context.decompress_reset().map_err(std::io::Error::other)?;
    let evicted = {
        let mut idle = CONTEXTS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let evicted = (idle.len() == MAX_IDLE_CONTEXTS).then(|| idle.remove(0));
        idle.push((footprint, context));
        evicted
    };
    drop(evicted);
    result.map_err(std::io::Error::other)
}
