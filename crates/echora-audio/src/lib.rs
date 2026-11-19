//! Host audio output: the interleaved signed 16-bit PCM that the guest's
//! audio-out buffers carry, played on the default output device through
//! PipeWire on Linux and CPAL on other platforms. Opt-in with
//! `ECHORA_AUDIO=1`.
//!
//! A dedicated thread owns the stream (CPAL streams are not `Send` on every
//! platform) and moves submitted samples into a bounded queue that the
//! device callback drains; an empty queue plays silence. The queue holds at
//! most one second, dropping the oldest samples beyond that, so a guest
//! that runs ahead cannot build up unbounded latency. Buffer release timing
//! stays with the guest's audio service.

use std::collections::VecDeque;
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Mutex, PoisonError};

#[cfg(not(target_os = "linux"))]
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
#[cfg(target_os = "linux")]
mod pipewire;

/// Start a host stream of interleaved PCM16 at `sample_rate` with
/// `channels` channels, or `None` when host audio is off or no device can
/// play that format (the reason is printed).
pub fn open(sample_rate: u32, channels: u32) -> Option<Sender<Vec<u8>>> {
    std::env::var_os("ECHORA_AUDIO")?;
    open_device(sample_rate, channels)
}

/// As [`open`], without the `ECHORA_AUDIO` opt-in check.
pub fn open_device(sample_rate: u32, channels: u32) -> Option<Sender<Vec<u8>>> {
    let (sender, receiver) = channel::<Vec<u8>>();
    let (ready_tx, ready_rx) = channel::<Result<(), String>>();
    std::thread::Builder::new()
        .name("echora-audio".into())
        .spawn(move || {
            let queue = Arc::new(Mutex::new(VecDeque::<i16>::new()));
            #[cfg(target_os = "linux")]
            let stream = pipewire::Playback::start(sample_rate, channels, queue.clone());
            #[cfg(not(target_os = "linux"))]
            let stream = build_stream(sample_rate, channels, queue.clone());
            let stream = match stream {
                Ok(stream) => stream,
                Err(error) => {
                    let _ = ready_tx.send(Err(error));
                    return;
                }
            };
            let _ = ready_tx.send(Ok(()));
            let limit = sample_rate as usize * channels as usize;
            for pcm in receiver {
                let mut queue = queue.lock().unwrap_or_else(PoisonError::into_inner);
                push_pcm(&mut queue, &pcm, limit);
            }
            drop(stream);
        })
        .ok()?;
    match ready_rx.recv() {
        Ok(Ok(())) => Some(sender),
        Ok(Err(error)) => {
            eprintln!("audio: {error}");
            None
        }
        Err(_) => None,
    }
}

/// Appends PCM16 bytes as samples, keeping at most `limit` samples.
fn push_pcm(queue: &mut VecDeque<i16>, pcm: &[u8], limit: usize) {
    let (samples, _) = pcm.as_chunks::<2>();
    let samples = &samples[samples.len().saturating_sub(limit)..];
    let excess = queue.len().saturating_sub(limit - samples.len());
    if excess != 0 {
        queue.drain(..excess);
    }
    queue.extend(samples.iter().map(|pair| i16::from_le_bytes(*pair)));
}

/// Fills `out` from the queue, padding with silence.
fn pull<T: Copy>(
    queue: &Mutex<VecDeque<i16>>,
    out: &mut [T],
    convert: impl Fn(i16) -> T,
    silence: T,
) {
    let mut queue = queue.lock().unwrap_or_else(PoisonError::into_inner);
    let count = out.len().min(queue.len());
    let (ready, remaining) = out.split_at_mut(count);
    let (front, back) = queue.as_slices();
    let (first, second) = ready.split_at_mut(front.len().min(count));
    for (slot, &sample) in first.iter_mut().zip(front) {
        *slot = convert(sample);
    }
    for (slot, &sample) in second.iter_mut().zip(back) {
        *slot = convert(sample);
    }
    queue.drain(..count);
    drop(queue);
    remaining.fill(silence);
}

#[cfg(not(target_os = "linux"))]
fn build_stream(
    sample_rate: u32,
    channels: u32,
    queue: Arc<Mutex<VecDeque<i16>>>,
) -> Result<cpal::Stream, String> {
    let device = cpal::default_host()
        .default_output_device()
        .ok_or("no default output device")?;
    let channels_u16 = u16::try_from(channels).map_err(|_| format!("{channels} channels"))?;
    let rate = cpal::SampleRate(sample_rate);
    let supported = device
        .supported_output_configs()
        .map_err(|error| error.to_string())?
        .filter(|range| range.channels() == channels_u16)
        .filter(|range| range.min_sample_rate() <= rate && rate <= range.max_sample_rate())
        .map(|range| range.with_sample_rate(rate))
        .min_by_key(|config| match config.sample_format() {
            cpal::SampleFormat::I16 => 0,
            cpal::SampleFormat::F32 => 1,
            _ => 2,
        })
        .ok_or_else(|| {
            format!("the output device cannot play {channels} channels at {sample_rate} Hz")
        })?;
    let format = supported.sample_format();
    let config: cpal::StreamConfig = supported.into();
    let error = |error| eprintln!("audio: stream error: {error}");
    let stream = match format {
        cpal::SampleFormat::I16 => device.build_output_stream(
            &config,
            move |out: &mut [i16], _| pull(&queue, out, |s| s, 0),
            error,
            None,
        ),
        cpal::SampleFormat::F32 => device.build_output_stream(
            &config,
            move |out: &mut [f32], _| pull(&queue, out, |s| f32::from(s) / 32768.0, 0.0),
            error,
            None,
        ),
        other => return Err(format!("unsupported device sample format {other}")),
    }
    .map_err(|error| error.to_string())?;
    stream.play().map_err(|error| error.to_string())?;
    Ok(stream)
}

/// Scale interleaved PCM16 by `volume` (1.0 leaves it unchanged).
pub fn apply_volume(pcm: &mut [u8], volume: f32) {
    if volume == 1.0 {
        return;
    }
    for sample in pcm.as_chunks_mut::<2>().0 {
        let value = i16::from_le_bytes(*sample) as f32 * volume;
        let clamped = value.clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        sample.copy_from_slice(&clamped.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn volume_scales_and_clamps() {
        let mut pcm = [1000i16, -1000, 30000, -30000]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        apply_volume(&mut pcm, 2.0);
        let samples: Vec<i16> = pcm
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| i16::from_le_bytes(*pair))
            .collect();
        assert_eq!(samples, [2000, -2000, i16::MAX, i16::MIN]);
    }

    #[test]
    fn queue_is_bounded_and_underruns_play_silence() {
        let mut queue = VecDeque::new();
        let pcm: Vec<u8> = (1..=6i16).flat_map(i16::to_le_bytes).collect();
        push_pcm(&mut queue, &pcm, 4);
        assert_eq!(queue, [3, 4, 5, 6]);
        let queue = Mutex::new(queue);
        let mut out = [0f32; 6];
        pull(&queue, &mut out, f32::from, -1.0);
        assert_eq!(out, [3.0, 4.0, 5.0, 6.0, -1.0, -1.0]);
    }

    #[test]
    fn pull_preserves_wrapped_queue_order_across_callbacks() {
        let mut queue = VecDeque::with_capacity(4);
        let capacity = queue.capacity() as i16;
        queue.extend(0..capacity);
        queue.drain(..2);
        let pcm: Vec<_> = (capacity..capacity + 2)
            .flat_map(i16::to_le_bytes)
            .collect();
        push_pcm(&mut queue, &pcm, capacity as usize);

        let queue = Mutex::new(queue);
        let mut first = vec![0; capacity as usize - 1];
        pull(&queue, &mut first, |sample| sample, 0);
        assert_eq!(first, (2..capacity + 1).collect::<Vec<_>>());
        let mut last = [0; 2];
        pull(&queue, &mut last, |sample| sample, -1);
        assert_eq!(last, [capacity + 1, -1]);
        assert!(queue.into_inner().unwrap().is_empty());
    }
}
