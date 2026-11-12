//! Opus decoding for the hardware Opus decoder service: libopus (as
//! `unsafe-libopus`, its transpilation to Rust) behind a safe interface.
//! One decoder holds one stream's state, or a multistream decoder's. The
//! library keeps C's interface, so calling it takes unsafe code.

#![allow(unsafe_code)]

use std::ptr::NonNull;

use unsafe_libopus::{
    OpusDecoder, opus_decode, opus_decoder_create, opus_decoder_destroy, opus_multistream_decode,
    opus_multistream_decoder_create, opus_multistream_decoder_destroy,
};

/// The longest Opus frame: 120 ms at 48 kHz, per channel.
const MAX_FRAME: usize = 5760;

enum State {
    Single(NonNull<OpusDecoder>),
    Multi(NonNull<unsafe_libopus::OpusMSDecoder>),
}

/// A stream's decoder at a fixed rate and channel count.
pub struct Decoder {
    state: State,
    channels: usize,
}

// SAFETY: the decoder state is plain heap memory owned by this value and
// only reached through `&mut self`.
unsafe impl Send for Decoder {}

/// A multistream layout: stream count, coupled (stereo) stream count and
/// the stream channel each output channel takes.
pub struct Layout<'a> {
    pub streams: u32,
    pub coupled: u32,
    pub mapping: &'a [u8],
}

impl Decoder {
    /// A decoder of `channels` channels at `sample_rate` (8, 12, 16, 24 or
    /// 48 kHz); with `layout`, a multistream one.
    pub fn new(
        sample_rate: u32,
        channels: u32,
        layout: Option<Layout<'_>>,
    ) -> Result<Self, String> {
        let mut error = 0i32;
        let state = match layout {
            None => {
                if !(1..=2).contains(&channels) {
                    return Err(format!("{channels} channels without a stream layout"));
                }
                // SAFETY: plain arguments and a valid error pointer.
                let pointer =
                    unsafe { opus_decoder_create(sample_rate as i32, channels as i32, &mut error) };
                NonNull::new(pointer).map(State::Single)
            }
            Some(layout) => {
                if layout.mapping.len() < channels as usize {
                    return Err("stream mapping shorter than the channel count".into());
                }
                // SAFETY: the mapping holds `channels` bytes (checked).
                let pointer = unsafe {
                    opus_multistream_decoder_create(
                        sample_rate as i32,
                        channels as i32,
                        layout.streams as i32,
                        layout.coupled as i32,
                        layout.mapping.as_ptr(),
                        &mut error,
                    )
                };
                NonNull::new(pointer).map(State::Multi)
            }
        };
        match state {
            Some(state) if error == 0 => Ok(Self {
                state,
                channels: channels as usize,
            }),
            _ => Err(format!("libopus refused the decoder (error {error})")),
        }
    }

    /// Decode one packet to interleaved 16-bit samples.
    pub fn decode(&mut self, packet: &[u8]) -> Result<Vec<i16>, String> {
        let mut pcm = vec![0i16; MAX_FRAME * self.channels];
        let length = i32::try_from(packet.len()).map_err(|_| "packet too long".to_string())?;
        // SAFETY: the state is live, the packet is `length` bytes and the
        // output holds MAX_FRAME samples per channel.
        let frames = unsafe {
            match self.state {
                State::Single(state) => opus_decode(
                    state.as_ptr(),
                    packet.as_ptr(),
                    length,
                    pcm.as_mut_ptr(),
                    MAX_FRAME as i32,
                    0,
                ),
                State::Multi(state) => opus_multistream_decode(
                    state.as_ptr(),
                    packet.as_ptr(),
                    length,
                    pcm.as_mut_ptr(),
                    MAX_FRAME as i32,
                    0,
                ),
            }
        };
        if frames < 0 {
            return Err(format!("libopus error {frames}"));
        }
        pcm.truncate(frames as usize * self.channels);
        Ok(pcm)
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        // SAFETY: created by the matching create call, destroyed once.
        unsafe {
            match self.state {
                State::Single(state) => opus_decoder_destroy(state.as_ptr()),
                State::Multi(state) => opus_multistream_decoder_destroy(state.as_ptr()),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_silent_frame_decodes_to_its_duration() {
        // TOC 0xf8: CELT fullband, 20 ms, mono, one frame; an empty frame
        // body is a lost frame, concealed as silence from a fresh state.
        let mut decoder = Decoder::new(48_000, 1, None).unwrap();
        let pcm = decoder.decode(&[0xf8]).unwrap();
        assert_eq!(pcm.len(), 960);
        assert!(pcm.iter().all(|&sample| sample == 0));
        // Stereo output of the same packet interleaves two channels.
        let mut stereo = Decoder::new(48_000, 2, None).unwrap();
        assert_eq!(stereo.decode(&[0xf8]).unwrap().len(), 1920);
        // Three channels need a stream layout.
        assert!(Decoder::new(48_000, 3, None).is_err());
        let layout = Layout {
            streams: 2,
            coupled: 1,
            mapping: &[0, 1, 2],
        };
        assert!(Decoder::new(48_000, 3, Some(layout)).is_ok());
    }
}
