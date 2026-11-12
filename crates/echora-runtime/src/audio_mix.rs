//! The audio renderer's signal processing: wave buffer decoding, pitch
//! resampling, biquad filters and ramped mixing, on 32-bit sample buffers
//! as the console's DSP holds them.
//!
//! The rules (sample formats, the fixed-point steps, which taps a ratio
//! selects) follow the same reference as `audio_renderer.rs`.

use crate::audio_tables::{HIGH_0, HIGH_1, HIGH_2, NORMAL_0, NORMAL_1, NORMAL_2};

/// Guest memory as the renderer reads it: false when the range is not
/// readable (the samples then decode as nothing).
pub trait Source {
    fn read(&self, address: u64, out: &mut [u8]) -> bool;

    /// Write `bytes` at `address` (the aux effect's buffers); false where
    /// the memory is not writable.
    fn write(&self, _address: u64, _bytes: &[u8]) -> bool {
        false
    }
}

/// NVN-independent sample formats of a voice's wave buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Format {
    #[default]
    Invalid,
    Pcm16,
    PcmFloat,
    Adpcm,
}

impl Format {
    pub fn from_raw(raw: u8) -> Self {
        match raw {
            2 => Self::Pcm16,
            5 => Self::PcmFloat,
            6 => Self::Adpcm,
            _ => Self::Invalid,
        }
    }
}

/// Resampler quality (SrcQuality): which filter a voice's pitch uses, and
/// how many history samples it keeps in front of the read position.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Quality {
    Low,
    #[default]
    Medium,
    High,
}

impl Quality {
    pub fn from_raw(raw: u8) -> Self {
        match raw {
            1 => Self::High,
            2 => Self::Low,
            _ => Self::Medium,
        }
    }

    /// History samples carried between frames (PitchBySrcQuality): the
    /// filter's taps.
    pub fn history(self) -> usize {
        match self {
            Self::High => 8,
            Self::Low | Self::Medium => 4,
        }
    }
}

/// The ADPCM decoder's running state (the wave buffer's context: header,
/// then the two previous samples).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AdpcmContext {
    pub header: u16,
    pub yn0: i16,
    pub yn1: i16,
}

impl AdpcmContext {
    pub fn parse(bytes: &[u8]) -> Self {
        let word = |at: usize| {
            bytes
                .get(at..at + 2)
                .map_or(0, |pair| u16::from_le_bytes([pair[0], pair[1]]))
        };
        Self {
            header: word(0),
            yn0: word(2) as i16,
            yn1: word(4) as i16,
        }
    }
}

const SAMPLES_PER_FRAME: u32 = 14;
const NIBBLES_PER_FRAME: u32 = 16;

/// Decode up to `wanted` samples of channel `channel` from interleaved
/// 16-bit (or 32-bit float) PCM at `buffer`: samples `start + offset` up
/// to `end` of the wave buffer. Returns how many were appended to `out`.
#[allow(clippy::too_many_arguments)]
pub fn decode_pcm(
    source: &dyn Source,
    float: bool,
    buffer: u64,
    channels: u32,
    channel: u32,
    start: u32,
    end: u32,
    offset: u32,
    wanted: u32,
    out: &mut Vec<i16>,
) -> u32 {
    if buffer == 0 || start >= end || channels == 0 {
        return 0;
    }
    let position = start.saturating_add(offset);
    let count = wanted.min(end.saturating_sub(position));
    if count == 0 {
        return 0;
    }
    let width = if float { 4 } else { 2 };
    let stride = width * channels as usize;
    let mut bytes = vec![0u8; count as usize * stride];
    if !source.read(buffer + u64::from(position) * stride as u64, &mut bytes) {
        return 0;
    }
    for frame in bytes.chunks_exact(stride) {
        let at = channel as usize * width;
        let sample = if float {
            let value =
                f32::from_le_bytes([frame[at], frame[at + 1], frame[at + 2], frame[at + 3]]);
            (value * 32768.0).clamp(-32768.0, 32767.0) as i16
        } else {
            i16::from_le_bytes([frame[at], frame[at + 1]])
        };
        out.push(sample);
    }
    count
}

/// Decode up to `wanted` samples of 4-bit ADPCM (14 samples to an 8-byte
/// frame: a header byte, then 14 nibbles) with predictor `coefficients`
/// (8 pairs), continuing `context`. Returns how many were appended.
#[allow(clippy::too_many_arguments)]
pub fn decode_adpcm(
    source: &dyn Source,
    buffer: u64,
    coefficients: &[i16; 16],
    context: &mut AdpcmContext,
    start: u32,
    end: u32,
    offset: u32,
    wanted: u32,
    out: &mut Vec<i16>,
) -> u32 {
    if buffer == 0 || start >= end {
        return 0;
    }
    let position = start.saturating_add(offset);
    let count = wanted.min(end.saturating_sub(position));
    if count == 0 {
        return 0;
    }
    let in_frame = position % SAMPLES_PER_FRAME;
    let mut nibble = (position / SAMPLES_PER_FRAME) * NIBBLES_PER_FRAME + in_frame;
    if in_frame != 0 {
        nibble += 2;
    }
    // The bytes the samples' nibbles and the frame headers on the way span.
    let first_byte = nibble / 2;
    let mut end_nibble = nibble;
    for _ in 0..count {
        if end_nibble % NIBBLES_PER_FRAME == 0 {
            end_nibble += 2;
        }
        end_nibble += 1;
    }
    let mut bytes = vec![0u8; (end_nibble.div_ceil(2) - first_byte) as usize];
    if !source.read(buffer + u64::from(first_byte), &mut bytes) {
        return 0;
    }
    let mut header = context.header as u8;
    let (mut yn0, mut yn1) = (i32::from(context.yn0), i32::from(context.yn1));
    let mut read = 0usize;
    let mut left = count;
    while left > 0 {
        if nibble % NIBBLES_PER_FRAME == 0 {
            header = bytes[read];
            read += 1;
            nibble += 2;
        }
        let index = usize::from(header >> 4) & 7;
        let scale = u32::from(header & 0xf);
        let (coeff0, coeff1) = (
            i32::from(coefficients[index * 2]),
            i32::from(coefficients[index * 2 + 1]),
        );
        let byte = bytes[read];
        let raw = if nibble & 1 != 0 {
            read += 1;
            byte & 0xf
        } else {
            byte >> 4
        };
        // Sign-extend the nibble.
        let code = (i32::from(raw) << 28) >> 28;
        let xn = code.wrapping_shl(scale);
        let sample = (xn.wrapping_shl(11))
            .wrapping_add(0x400)
            .wrapping_add(coeff0.wrapping_mul(yn0))
            .wrapping_add(coeff1.wrapping_mul(yn1))
            >> 11;
        let sample = sample.clamp(-0x8000, 0x7fff);
        yn1 = yn0;
        yn0 = sample;
        out.push(sample as i16);
        nibble += 1;
        left -= 1;
    }
    *context = AdpcmContext {
        header: u16::from(header),
        yn0: yn0 as i16,
        yn1: yn1 as i16,
    };
    count
}

/// Fixed-point with 15 fractional bits, as the resampler steps.
pub const FRACTION_ONE: u64 = 1 << 15;

/// A sample-rate ratio (source rate over output rate, times pitch) in
/// 15-bit fixed point.
pub fn ratio(source_rate: u32, target_rate: u32, pitch: f32) -> u64 {
    let value = source_rate as f32 / target_rate as f32 * pitch;
    if value.is_finite() && value > 0.0 {
        (value * FRACTION_ONE as f32) as u64
    } else {
        0
    }
}

/// Resample `input` (history first, then new samples) into `output` at
/// `ratio`, starting at phase `fraction` (both 15-bit fixed point);
/// `fraction` is left as the next frame's phase.
pub fn resample(
    output: &mut [i32],
    input: &[i16],
    ratio: u64,
    fraction: &mut u64,
    quality: Quality,
) {
    let sample = |index: usize| f32::from(input.get(index).copied().unwrap_or(0));
    let mut read = 0usize;
    match quality {
        Quality::Low => {
            if ratio == FRACTION_ONE {
                for (slot, &value) in output.iter_mut().zip(input) {
                    *slot = i32::from(value);
                }
                return;
            }
            for slot in output.iter_mut() {
                let at = read + usize::from(*fraction >= FRACTION_ONE / 2);
                *slot = i32::from(input.get(at).copied().unwrap_or(0));
                *fraction += ratio;
                read += (*fraction >> 15) as usize;
                *fraction &= FRACTION_ONE - 1;
            }
        }
        Quality::Medium | Quality::High => {
            let taps = if quality == Quality::High { 8 } else { 4 };
            // One table per ratio band: up to 1.0, below 1.3, above.
            let unity = FRACTION_ONE;
            let band = (FRACTION_ONE as f32 * 1.3) as u64;
            let table: &[f32] = match (quality, ratio) {
                (Quality::High, r) if r <= unity => &HIGH_2,
                (Quality::High, r) if r < band => &HIGH_1,
                (Quality::High, _) => &HIGH_0,
                (_, r) if r <= unity => &NORMAL_2,
                (_, r) if r < band => &NORMAL_1,
                _ => &NORMAL_0,
            };
            for slot in output.iter_mut() {
                let phase = ((*fraction & (FRACTION_ONE - 1)) >> 8) as usize * taps;
                // Each product keeps 8 fractional bits; the sum floors.
                let mut sum = 0i64;
                for tap in 0..taps {
                    sum += (sample(read + tap) * table[phase + tap] * 256.0) as i64;
                }
                *slot = (sum >> 8) as i32;
                *fraction += ratio;
                read += (*fraction >> 15) as usize;
                *fraction &= FRACTION_ONE - 1;
            }
        }
    }
}

/// A biquad's two delayed terms (transposed direct form II), kept per
/// voice channel and filter.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BiquadState {
    pub s0: f64,
    pub s1: f64,
}

/// Filter `samples` in place: y = b0 x + b1 x1 + b2 x2 + a1 y1 + a2 y2,
/// with the coefficients as floats (the 16-bit form is Q14).
pub fn biquad(samples: &mut [i32], b: [f32; 3], a: [f32; 2], state: &mut BiquadState) {
    let (b0, b1, b2) = (f64::from(b[0]), f64::from(b[1]), f64::from(b[2]));
    let (a1, a2) = (f64::from(a[0]), f64::from(a[1]));
    for sample in samples {
        let input = f64::from(*sample);
        let output = input * b0 + state.s0;
        state.s0 = input * b1 + output * a1 + state.s1;
        state.s1 = input * b2 + output * a2;
        *sample = output.clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32;
    }
}

/// Scale `samples` in place from `previous` to `volume` linearly over the
/// frame (a constant volume when they are equal).
pub fn volume_ramp(samples: &mut [i32], previous: f32, volume: f32) {
    if samples.is_empty() {
        return;
    }
    let step = (volume - previous) / samples.len() as f32;
    let mut current = previous;
    for sample in samples {
        *sample = (*sample as f32 * current) as i32;
        current += step;
    }
}

/// Add `input` scaled by a volume ramping from `previous` to `volume`
/// into `output`. Returns the last sample added (the depop value).
pub fn mix_ramp(output: &mut [i32], input: &[i32], previous: f32, volume: f32) -> i32 {
    if previous == 0.0 && volume == 0.0 {
        return 0;
    }
    let count = output.len().min(input.len());
    if count == 0 {
        return 0;
    }
    let step = (volume - previous) / count as f32;
    let mut current = previous;
    let mut last = 0;
    for (slot, &sample) in output.iter_mut().zip(input) {
        last = (sample as f32 * current) as i32;
        *slot = slot.wrapping_add(last);
        current += step;
    }
    last
}

/// Add `input` scaled by `volume` into `output`.
pub fn mix(output: &mut [i32], input: &[i32], volume: f32) {
    if volume == 0.0 {
        return;
    }
    for (slot, &sample) in output.iter_mut().zip(input) {
        *slot = slot.wrapping_add((sample as f32 * volume) as i32);
    }
}

/// Fade a stopped voice's last output out of `output` instead of cutting
/// it: `sample` decays by `decay` (Q15) each output sample. Returns what
/// is left for the next frame.
pub fn depop(output: &mut [i32], mut sample: i32, decay: i64) -> i32 {
    if sample == 0 {
        return 0;
    }
    for slot in output {
        sample = ((i64::from(sample) * decay) >> 15) as i32;
        *slot = slot.wrapping_add(sample);
    }
    sample
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Bytes(Vec<u8>);
    impl Source for Bytes {
        fn read(&self, address: u64, out: &mut [u8]) -> bool {
            let start = address as usize - 0x1000;
            match self.0.get(start..start + out.len()) {
                Some(bytes) => {
                    out.copy_from_slice(bytes);
                    true
                }
                None => false,
            }
        }
    }

    #[test]
    fn pcm16_takes_one_channel_of_the_buffer_range() {
        // Stereo: left counts up, right counts down.
        let mut bytes = Vec::new();
        for n in 0..16i16 {
            bytes.extend_from_slice(&n.to_le_bytes());
            bytes.extend_from_slice(&(-n).to_le_bytes());
        }
        let source = Bytes(bytes);
        let mut out = Vec::new();
        assert_eq!(
            decode_pcm(&source, false, 0x1000, 2, 1, 2, 10, 3, 100, &mut out),
            5
        );
        assert_eq!(out, [-5, -6, -7, -8, -9]);
        // Past the end of the range: nothing.
        assert_eq!(
            decode_pcm(&source, false, 0x1000, 2, 0, 2, 10, 8, 4, &mut out),
            0
        );
    }

    #[test]
    fn adpcm_predicts_from_the_two_previous_samples() {
        // One frame: predictor 0, scale 1; nibbles 1, 2, -1, then zeros.
        let frame = [0x01u8, 0x12, 0xf0, 0, 0, 0, 0, 0];
        let source = Bytes(frame.to_vec());
        let mut coefficients = [0i16; 16];
        // y = x + yn0 (coefficient 0x800 is 1.0 in Q11).
        coefficients[0] = 0x800;
        let mut context = AdpcmContext::default();
        let mut out = Vec::new();
        assert_eq!(
            decode_adpcm(
                &source,
                0x1000,
                &coefficients,
                &mut context,
                0,
                14,
                0,
                4,
                &mut out
            ),
            4
        );
        // Codes scaled by 2, each added to the previous output.
        assert_eq!(out, [2, 6, 4, 4]);
        assert_eq!((context.yn0, context.yn1), (4, 4));
        // Continuing mid-frame uses the kept header and history.
        assert_eq!(
            decode_adpcm(
                &source,
                0x1000,
                &coefficients,
                &mut context,
                0,
                14,
                4,
                2,
                &mut out
            ),
            2
        );
        assert_eq!(&out[4..], [4, 4]);
    }

    #[test]
    fn unity_resampling_passes_samples_through_the_filter_centre() {
        // A constant input stays constant through any filter phase.
        let input = [1000i16; 64];
        for quality in [Quality::Low, Quality::Medium, Quality::High] {
            let mut output = [0i32; 16];
            let mut fraction = 0;
            resample(&mut output, &input, FRACTION_ONE, &mut fraction, quality);
            for sample in output {
                assert!((sample - 1000).abs() <= 2, "{quality:?} gave {sample}");
            }
            assert_eq!(fraction, 0);
        }
    }

    #[test]
    fn resampling_advances_by_the_ratio() {
        // Twice the rate reads two input samples per output sample.
        let input: Vec<i16> = (0..64).map(|n| n * 100).collect();
        let mut output = [0i32; 8];
        let mut fraction = 0;
        resample(
            &mut output,
            &input,
            2 * FRACTION_ONE,
            &mut fraction,
            Quality::Low,
        );
        assert_eq!(output, [0, 200, 400, 600, 800, 1000, 1200, 1400]);
        // A half-rate ratio carries its phase between calls.
        let mut fraction = 0;
        let mut output = [0i32; 3];
        resample(
            &mut output,
            &input,
            FRACTION_ONE / 2,
            &mut fraction,
            Quality::Low,
        );
        assert_eq!(fraction, FRACTION_ONE / 2);
    }

    #[test]
    fn mixing_ramps_and_reports_the_last_sample() {
        let input = [1000i32; 4];
        let mut output = [1i32; 4];
        let last = mix_ramp(&mut output, &input, 0.0, 1.0);
        assert_eq!(output, [1, 251, 501, 751]);
        assert_eq!(last, 750);
        let mut samples = [1000i32; 2];
        volume_ramp(&mut samples, 0.5, 0.5);
        assert_eq!(samples, [500, 500]);
        // A biquad with b0 = 1 and nothing else is the identity.
        let mut samples = [3, -4, 5];
        biquad(
            &mut samples,
            [1.0, 0.0, 0.0],
            [0.0, 0.0],
            &mut BiquadState::default(),
        );
        assert_eq!(samples, [3, -4, 5]);
    }
}
