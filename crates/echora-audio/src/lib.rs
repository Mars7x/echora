//! Host audio output: the interleaved signed 16-bit PCM that the guest's
//! audio-out buffers carry, played on the default output device through
//! PipeWire on Linux and CPAL on other platforms. Opt-in with
//! `ECHORA_AUDIO=1`.
//!
//! A dedicated thread owns the stream (CPAL streams are not `Send` on every
//! platform) and moves submitted samples into a queue that the device
//! callback drains; an empty queue plays silence. The guest renders on the
//! host's clock, as the console's DSP renders on its own, so the queue
//! stays short while both keep time. A stall that delays submission makes
//! the device play silence and then receive the late samples together;
//! keeping them would delay all later sound by the stall. Samples that
//! stayed queued through a whole window of device reads are such a
//! backlog and are dropped (`Queue::read`), so sound returns to its place
//! instead of drifting behind the picture. The queue never holds more than
//! one second. Buffer release timing stays with the guest's audio service.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
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
            let queue = Arc::new(Mutex::new(Queue::new(sample_rate, channels)));
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
            let mut stats = std::env::var_os("ECHORA_AUDIO_STATS").and_then(StatsLog::new);
            loop {
                let pcm = match stats {
                    Some(_) => match receiver.recv_timeout(StatsLog::PERIOD) {
                        Ok(pcm) => Some(pcm),
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => None,
                        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                    },
                    None => match receiver.recv() {
                        Ok(pcm) => Some(pcm),
                        Err(_) => break,
                    },
                };
                if let Some(pcm) = pcm {
                    let mut queue = queue.lock().unwrap_or_else(PoisonError::into_inner);
                    queue.push(&pcm);
                }
                if let Some(log) = &mut stats {
                    log.tick(&queue);
                }
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

/// Sample counts since the first stream opened, for measuring how the
/// device's playback keeps time with the guest's rendering.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// Samples submitted by the guest.
    pub submitted: u64,
    /// Submitted samples the device played.
    pub played: u64,
    /// Silence the device played for want of samples, once any arrived.
    pub silence: u64,
    /// Samples dropped as a backlog that stayed queued a whole window.
    pub trimmed: u64,
    /// Samples dropped beyond the one-second bound.
    pub overflowed: u64,
    /// Samples queued now.
    pub queued: u64,
}

static SUBMITTED: AtomicU64 = AtomicU64::new(0);
static PLAYED: AtomicU64 = AtomicU64::new(0);
static SILENCE: AtomicU64 = AtomicU64::new(0);
static TRIMMED: AtomicU64 = AtomicU64::new(0);
static OVERFLOWED: AtomicU64 = AtomicU64::new(0);
static QUEUED: AtomicU64 = AtomicU64::new(0);

/// The counts so far, over all streams.
pub fn stats() -> Stats {
    Stats {
        submitted: SUBMITTED.load(Ordering::Relaxed),
        played: PLAYED.load(Ordering::Relaxed),
        silence: SILENCE.load(Ordering::Relaxed),
        trimmed: TRIMMED.load(Ordering::Relaxed),
        overflowed: OVERFLOWED.load(Ordering::Relaxed),
        queued: QUEUED.load(Ordering::Relaxed),
    }
}

/// `ECHORA_AUDIO_STATS=<file>` (debugging): the counts every 100 ms, one
/// line each, with the host's monotonic time in microseconds.
struct StatsLog {
    file: std::fs::File,
    start: std::time::Instant,
    last: std::time::Instant,
}

impl StatsLog {
    const PERIOD: std::time::Duration = std::time::Duration::from_millis(100);

    fn new(path: std::ffi::OsString) -> Option<Self> {
        let file = std::fs::File::create(&path)
            .inspect_err(|error| eprintln!("audio: stats file: {error}"))
            .ok()?;
        let now = std::time::Instant::now();
        Some(Self {
            file,
            start: now,
            last: now,
        })
    }

    fn tick(&mut self, queue: &Mutex<Queue>) {
        use std::io::Write;
        let now = std::time::Instant::now();
        if now - self.last < Self::PERIOD {
            return;
        }
        self.last = now;
        let rate = queue.lock().unwrap_or_else(PoisonError::into_inner).rate;
        let s = stats();
        let _ = writeln!(
            self.file,
            "t_us {} rate {rate} submitted {} played {} silence {} trimmed {} overflowed {} queued {}",
            (now - self.start).as_micros(),
            s.submitted,
            s.played,
            s.silence,
            s.trimmed,
            s.overflowed,
            s.queued
        );
    }
}

/// Interleaved samples waiting for the device.
struct Queue {
    samples: VecDeque<i16>,
    channels: usize,
    /// Samples per second, over all channels.
    rate: usize,
    /// Whether any samples arrived: silence before that is not an underrun.
    primed: bool,
    /// Fewest samples left after a device read in the current window.
    low: usize,
    /// Samples the device read in the current window.
    read: usize,
}

impl Queue {
    fn new(sample_rate: u32, channels: u32) -> Self {
        let channels = channels.max(1) as usize;
        Self {
            samples: VecDeque::new(),
            channels,
            rate: sample_rate as usize * channels,
            primed: false,
            low: usize::MAX,
            read: 0,
        }
    }

    /// The most samples the queue holds: one second.
    fn limit(&self) -> usize {
        self.rate
    }

    /// The device reads this many samples between checks for a backlog: a
    /// quarter second spans several device periods and renderer frames.
    fn window(&self) -> usize {
        self.rate / 4
    }

    /// What may stay queued through a window without counting as a
    /// backlog: 10 ms, two renderer frames, absorbs submission jitter.
    fn margin(&self) -> usize {
        self.rate / 100 / self.channels * self.channels
    }

    fn clear(&mut self) {
        self.samples.clear();
        self.low = usize::MAX;
        self.read = 0;
        QUEUED.store(0, Ordering::Relaxed);
    }

    /// Appends PCM16 bytes as samples, keeping at most `limit` samples.
    fn push(&mut self, pcm: &[u8]) {
        let limit = self.limit();
        let (samples, _) = pcm.as_chunks::<2>();
        SUBMITTED.fetch_add(samples.len() as u64, Ordering::Relaxed);
        let kept = samples.len().min(limit) / self.channels * self.channels;
        let samples = &samples[samples.len() - kept..];
        let excess = self.samples.len().saturating_sub(limit - samples.len());
        if excess != 0 {
            self.samples.drain(..excess);
        }
        OVERFLOWED.fetch_add((excess + pcm.len() / 2 - kept) as u64, Ordering::Relaxed);
        self.samples
            .extend(samples.iter().map(|pair| i16::from_le_bytes(*pair)));
        self.primed |= !samples.is_empty();
        QUEUED.store(self.samples.len() as u64, Ordering::Relaxed);
    }

    /// Fills `out` from the queue, padding with silence, then drops a
    /// backlog: what stayed queued after every read of a whole window,
    /// beyond the margin, would otherwise delay all later sound.
    fn read<T: Copy>(&mut self, out: &mut [T], convert: impl Fn(i16) -> T, silence: T) {
        let count = out.len().min(self.samples.len());
        let (ready, remaining) = out.split_at_mut(count);
        let (front, back) = self.samples.as_slices();
        let (first, second) = ready.split_at_mut(front.len().min(count));
        for (slot, &sample) in first.iter_mut().zip(front) {
            *slot = convert(sample);
        }
        for (slot, &sample) in second.iter_mut().zip(back) {
            *slot = convert(sample);
        }
        self.samples.drain(..count);
        remaining.fill(silence);
        PLAYED.fetch_add(count as u64, Ordering::Relaxed);
        if self.primed {
            SILENCE.fetch_add(remaining.len() as u64, Ordering::Relaxed);
        }
        self.low = self.low.min(self.samples.len());
        self.read += out.len();
        if self.read >= self.window() {
            let backlog = self.low.saturating_sub(self.margin()) / self.channels * self.channels;
            if backlog != 0 {
                self.samples.drain(..backlog);
                TRIMMED.fetch_add(backlog as u64, Ordering::Relaxed);
            }
            self.low = usize::MAX;
            self.read = 0;
        }
        QUEUED.store(self.samples.len() as u64, Ordering::Relaxed);
    }
}

/// Fills `out` from the queue (`Queue::read`).
fn pull<T: Copy>(queue: &Mutex<Queue>, out: &mut [T], convert: impl Fn(i16) -> T, silence: T) {
    queue
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .read(out, convert, silence);
}

#[cfg(not(target_os = "linux"))]
fn build_stream(
    sample_rate: u32,
    channels: u32,
    queue: Arc<Mutex<Queue>>,
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

    fn pcm(samples: impl IntoIterator<Item = i16>) -> Vec<u8> {
        samples.into_iter().flat_map(i16::to_le_bytes).collect()
    }

    #[test]
    fn queue_is_bounded_and_underruns_play_silence() {
        // Four samples a second, one channel: the bound is four samples.
        let mut queue = Queue::new(4, 1);
        queue.push(&pcm(1..=6));
        assert_eq!(queue.samples, [3, 4, 5, 6]);
        let queue = Mutex::new(queue);
        let mut out = [0f32; 6];
        pull(&queue, &mut out, f32::from, -1.0);
        assert_eq!(out, [3.0, 4.0, 5.0, 6.0, -1.0, -1.0]);
    }

    #[test]
    fn pull_preserves_wrapped_queue_order_across_callbacks() {
        let mut queue = Queue::new(48_000, 1);
        queue.samples = VecDeque::with_capacity(4);
        let capacity = queue.samples.capacity() as i16;
        queue.samples.extend(0..capacity);
        queue.samples.drain(..2);
        queue.push(&pcm(capacity..capacity + 2));

        let queue = Mutex::new(queue);
        let mut first = vec![0; capacity as usize - 1];
        pull(&queue, &mut first, |sample| sample, 0);
        assert_eq!(first, (2..capacity + 1).collect::<Vec<_>>());
        let mut last = [0; 2];
        pull(&queue, &mut last, |sample| sample, -1);
        assert_eq!(last, [capacity + 1, -1]);
        assert!(queue.into_inner().unwrap().samples.is_empty());
    }

    /// 48 kHz stereo: renderer frames of 240 sample pairs every 5 ms and
    /// device reads of 1024 pairs as their time comes. Returns how many
    /// samples stay queued after the last read.
    fn play(queue: &mut Queue, stall_ms: u64, total_ms: u64) -> usize {
        let frame = vec![0u8; 240 * 2 * 2];
        let mut out = vec![0i16; 1024 * 2];
        let (mut rendered, mut read) = (0u64, 0u64);
        for ms in 0..total_ms {
            // The submitting thread stalls for the first `stall_ms` after
            // 100 ms, then catches up.
            let submitting = !(100..100 + stall_ms).contains(&ms);
            while submitting && rendered * 5 <= ms {
                queue.push(&frame);
                rendered += 1;
            }
            while (read + 1) * 1024 * 1000 <= ms * 48_000 {
                queue.read(&mut out, |sample| sample, 0);
                read += 1;
            }
        }
        queue.samples.len()
    }

    #[test]
    fn a_stall_backlog_is_dropped_instead_of_delaying_later_sound() {
        let mut steady = Queue::new(48_000, 2);
        let level = play(&mut steady, 0, 2000);
        let mut stalled = Queue::new(48_000, 2);
        let before = stats().trimmed;
        let after = play(&mut stalled, 200, 2000);
        // Without the trim, 200 ms (19,200 samples) would stay queued.
        assert!(
            after <= level + stalled.margin(),
            "{after} queued, steady {level}"
        );
        assert!(stats().trimmed - before >= 15_000);
        assert_eq!(after % 2, 0);
    }

    #[test]
    fn steady_playback_drops_nothing() {
        let mut queue = Queue::new(48_000, 2);
        play(&mut queue, 0, 3000);
        assert!(queue.samples.len() <= queue.margin() + 2 * 240 * 2);
        // A steady stream never keeps more than the margin through a window.
        let mut checked = Queue::new(48_000, 2);
        let frame = vec![0u8; 240 * 2 * 2];
        let mut out = vec![0i16; 240 * 2];
        for _ in 0..1000 {
            checked.push(&frame);
            checked.read(&mut out, |sample| sample, 0);
        }
        assert!(checked.samples.is_empty());
    }
}
