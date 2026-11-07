//! AV1 decoding for the video decoder service, through rav1d (a Rust port
//! of dav1d) built for 8-bit content only. rav1d's public interface is
//! dav1d's C API, so this module is the one place that calls it: a decoder
//! takes one submission (a temporal unit of low-overhead OBUs) at a time and
//! returns the pictures it completes as tight 4:2:0 planes.
#![allow(unsafe_code)]

use rav1d::include::dav1d::data::Dav1dData;
use rav1d::include::dav1d::dav1d::{Dav1dContext, Dav1dSettings};
use rav1d::include::dav1d::headers::DAV1D_PIXEL_LAYOUT_I420;
use rav1d::include::dav1d::picture::Dav1dPicture;
use rav1d::src::lib::{
    dav1d_close, dav1d_data_create, dav1d_data_unref, dav1d_default_settings, dav1d_get_picture,
    dav1d_open, dav1d_picture_unref, dav1d_send_data,
};
use std::mem::MaybeUninit;
use std::ptr::NonNull;

/// dav1d's "try again" result (-EAGAIN, rav1d takes the platform's
/// value): input not consumed yet, or no picture ready.
#[cfg(any(
    target_os = "freebsd",
    target_os = "netbsd",
    target_os = "openbsd",
    target_os = "dragonfly"
))]
const AGAIN: i32 = -35;
#[cfg(not(any(
    target_os = "freebsd",
    target_os = "netbsd",
    target_os = "openbsd",
    target_os = "dragonfly"
)))]
const AGAIN: i32 = -11;

/// A decoded 8-bit 4:2:0 picture with rows packed tightly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Picture {
    pub width: u32,
    pub height: u32,
    /// width x height luma samples.
    pub y: Vec<u8>,
    /// ceil(width / 2) x ceil(height / 2) samples each.
    pub u: Vec<u8>,
    pub v: Vec<u8>,
    /// Full-range samples (the sequence header's color_range).
    pub full_range: bool,
    /// The sequence header's matrix coefficients (1 BT.709, 6 BT.601, ...).
    pub matrix: u32,
}

pub struct Decoder {
    context: Option<Dav1dContext>,
}

// SAFETY: the context is rav1d's Arc of its decoder state, whose mutable
// parts sit behind its own lock; this wrapper owns the only reference and
// reaches it through &mut self, so moving it to another thread is sound.
unsafe impl Send for Decoder {}

impl std::fmt::Debug for Decoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Decoder").finish_non_exhaustive()
    }
}

impl Decoder {
    /// A decoder with a frame delay of one, so each submission's picture
    /// comes out of the same `decode` call; up to four threads share each
    /// frame's tiles and rows (the guest runs on another host core).
    pub fn new() -> Result<Self, String> {
        let mut settings = MaybeUninit::<Dav1dSettings>::zeroed();
        // SAFETY: the pointer is to writable storage for a Dav1dSettings,
        // which dav1d_default_settings fills completely.
        let mut settings = unsafe {
            dav1d_default_settings(NonNull::from(&mut settings).cast());
            settings.assume_init()
        };
        settings.n_threads =
            std::thread::available_parallelism().map_or(1, |cores| cores.get().min(4) as i32);
        settings.max_frame_delay = 1;
        let mut context = None;
        // SAFETY: both pointers are to live, initialized locals.
        let result = unsafe {
            dav1d_open(
                Some(NonNull::from(&mut context)),
                Some(NonNull::from(&mut settings)),
            )
        };
        if result.0 != 0 || context.is_none() {
            return Err(format!("rav1d open failed ({})", result.0));
        }
        Ok(Self { context })
    }

    /// Feed one submission and return the pictures it completes.
    pub fn decode(&mut self, bytes: &[u8]) -> Result<Vec<Picture>, String> {
        let mut pictures = Vec::new();
        if bytes.is_empty() {
            return Ok(pictures);
        }
        let mut data = Dav1dData::default();
        // SAFETY: data is a live Dav1dData; dav1d_data_create gives it a
        // buffer of bytes.len() bytes and returns that buffer, which is then
        // filled from the slice.
        unsafe {
            let buffer = dav1d_data_create(Some(NonNull::from(&mut data)), bytes.len());
            if buffer.is_null() {
                return Err("rav1d could not allocate input".into());
            }
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer, bytes.len());
        }
        let outcome = (|| {
            loop {
                // SAFETY: the context is open (closed only in Drop) and data
                // holds the buffer created above.
                let sent = unsafe { dav1d_send_data(self.context, Some(NonNull::from(&mut data))) };
                if sent.0 != 0 && sent.0 != AGAIN {
                    return Err(format!("rav1d rejected input ({})", sent.0));
                }
                self.drain(&mut pictures)?;
                if data.sz == 0 {
                    return Ok(());
                }
                if sent.0 == AGAIN && pictures.is_empty() {
                    // Not consumed and nothing came out: stop instead of
                    // spinning.
                    return Err("rav1d did not consume input".into());
                }
            }
        })();
        if data.sz != 0 {
            // SAFETY: data still owns its buffer.
            unsafe { dav1d_data_unref(Some(NonNull::from(&mut data))) };
        }
        outcome.map(|()| pictures)
    }

    fn drain(&mut self, pictures: &mut Vec<Picture>) -> Result<(), String> {
        loop {
            let mut picture = Dav1dPicture::default();
            // SAFETY: the context is open and picture is a live local.
            let result =
                unsafe { dav1d_get_picture(self.context, Some(NonNull::from(&mut picture))) };
            if result.0 == AGAIN {
                return Ok(());
            }
            if result.0 != 0 {
                return Err(format!("rav1d decode failed ({})", result.0));
            }
            // SAFETY: a successful dav1d_get_picture filled picture with
            // planes of the given size and strides; it is released below.
            let copied = unsafe { copy_planes(&picture) };
            // SAFETY: picture holds references from dav1d_get_picture.
            unsafe { dav1d_picture_unref(Some(NonNull::from(&mut picture))) };
            pictures.push(copied?);
        }
    }
}

/// # Safety
///
/// `picture` must be a picture returned by dav1d_get_picture and not yet
/// released.
unsafe fn copy_planes(picture: &Dav1dPicture) -> Result<Picture, String> {
    let parameters = &picture.p;
    if parameters.layout != DAV1D_PIXEL_LAYOUT_I420 || parameters.bpc != 8 {
        return Err(format!(
            "rav1d picture layout {} at {} bits is not 8-bit 4:2:0",
            parameters.layout, parameters.bpc
        ));
    }
    let (width, height) = (parameters.w as usize, parameters.h as usize);
    let (chroma_width, chroma_height) = (width.div_ceil(2), height.div_ceil(2));
    let plane =
        |index: usize, stride: isize, columns: usize, rows: usize| -> Result<Vec<u8>, String> {
            let base = picture.data[index].ok_or("rav1d picture without a plane")?;
            let mut out = Vec::with_capacity(columns * rows);
            for row in 0..rows {
                // SAFETY: each of the plane's rows holds at least `columns`
                // samples `stride` bytes apart (the caller's contract).
                let line = unsafe {
                    std::slice::from_raw_parts(
                        (base.as_ptr() as *const u8).offset(row as isize * stride),
                        columns,
                    )
                };
                out.extend_from_slice(line);
            }
            Ok(out)
        };
    let (full_range, matrix) = match picture.seq_hdr {
        // SAFETY: the sequence header lives as long as the picture.
        Some(header) => unsafe {
            let header = header.as_ref();
            (header.color_range != 0, header.mtrx as u32)
        },
        None => (false, 1),
    };
    Ok(Picture {
        width: width as u32,
        height: height as u32,
        y: plane(0, picture.stride[0], width, height)?,
        u: plane(1, picture.stride[1], chroma_width, chroma_height)?,
        v: plane(2, picture.stride[1], chroma_width, chroma_height)?,
        full_range,
        matrix,
    })
}

impl Drop for Decoder {
    fn drop(&mut self) {
        // SAFETY: the context came from dav1d_open and is closed once.
        unsafe { dav1d_close(Some(NonNull::from(&mut self.context))) };
    }
}

impl Picture {
    /// RGBA8 (alpha 255) by the picture's matrix: BT.601 for matrix 5 or 6,
    /// BT.709 otherwise; limited or full range per the header.
    pub fn to_rgba(&self) -> Vec<u8> {
        let (kr, kb) = match self.matrix {
            5 | 6 => (0.299f32, 0.114f32),
            9 | 10 => (0.2627, 0.0593),
            _ => (0.2126, 0.0722),
        };
        let kg = 1.0 - kr - kb;
        let (width, height) = (self.width as usize, self.height as usize);
        let chroma_width = width.div_ceil(2);
        let mut out = vec![0u8; width * height * 4];
        for row in 0..height {
            for column in 0..width {
                let chroma = (row / 2) * chroma_width + column / 2;
                let (mut y, mut cb, mut cr) = (
                    f32::from(self.y[row * width + column]),
                    f32::from(self.u[chroma]) - 128.0,
                    f32::from(self.v[chroma]) - 128.0,
                );
                if !self.full_range {
                    y = (y - 16.0) * (255.0 / 219.0);
                    cb *= 255.0 / 224.0;
                    cr *= 255.0 / 224.0;
                }
                let r = y + 2.0 * (1.0 - kr) * cr;
                let b = y + 2.0 * (1.0 - kb) * cb;
                let g = (y - kr * r - kb * b) / kg;
                let at = (row * width + column) * 4;
                out[at] = r.round().clamp(0.0, 255.0) as u8;
                out[at + 1] = g.round().clamp(0.0, 255.0) as u8;
                out[at + 2] = b.round().clamp(0.0, 255.0) as u8;
                out[at + 3] = 255;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn garbage_is_an_error_not_a_picture() {
        let mut decoder = Decoder::new().unwrap();
        assert!(decoder.decode(&[]).unwrap().is_empty());
        // A temporal delimiter alone completes nothing.
        assert!(decoder.decode(&[0x12, 0x00]).unwrap().is_empty());
    }

    #[test]
    fn gray_converts_to_gray() {
        let picture = Picture {
            width: 2,
            height: 2,
            y: vec![126; 4],
            u: vec![128],
            v: vec![128],
            full_range: false,
            matrix: 1,
        };
        let rgba = picture.to_rgba();
        assert_eq!(&rgba[..4], &[128, 128, 128, 255]);
    }
}
