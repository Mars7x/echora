//! IAudioRenderer state: the update protocol and real-time voice progress.
//!
//! Layouts and rules follow Eden-NS2 1a6d7d4 (`audio_core/renderer`:
//! `behavior/info_updater.cpp`, `voice/voice_info.{h,cpp}`,
//! `memory/pool_mapper.cpp`, `effect/effect_info_base.h`). Nothing is mixed:
//! voices advance through their wave buffers at their pitch and sample rate
//! as rendered frames pass in real time, so the application sees buffers
//! consumed and samples played, but no sound is produced.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

/// UpdateDataHeader size; input and output both start with one.
const HEADER_SIZE: usize = 0x40;
const MEMORY_POOL_IN: usize = 0x20;
const MEMORY_POOL_OUT: usize = 0x10;
const VOICE_OUT: usize = 0x10;
const EFFECT_IN: usize = 0xc0;
const EFFECT_OUT_V1: usize = 0x10;
const EFFECT_OUT_V2: usize = 0x90;
const SINK_IN: usize = 0x140;
const SINK_OUT: usize = 0x20;
const PERFORMANCE_OUT: usize = 0x10;
const BEHAVIOR_OUT: usize = 0xb0;
const RENDER_INFO_OUT: usize = 0x10;
const MAX_WAVE_BUFFERS: usize = 4;
/// Frames rendered per `advance` at most, after a long stall.
const MAX_CATCH_UP_FRAMES: u64 = 200;

/// AudioRendererParameterInternal fields the renderer uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Parameters {
    pub sample_rate: u32,
    pub sample_count: u32,
    pub mixes: u32,
    pub voices: u32,
    pub sinks: u32,
    pub effects: u32,
    pub revision: u32,
}

impl Parameters {
    /// The 0x34-byte AudioRendererParameterInternal.
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 0x34 {
            return Err(format!("renderer parameters are {:#x} bytes", bytes.len()));
        }
        let word = |at: usize| crate::bytes::le_u32(&bytes[at..at + 4]);
        let parameters = Self {
            sample_rate: word(0),
            sample_count: word(4),
            mixes: word(8),
            voices: word(0x10),
            sinks: word(0x14),
            effects: word(0x18),
            revision: word(0x30),
        };
        if parameters.sample_rate == 0 || parameters.sample_count == 0 {
            return Err(format!("renderer parameters {parameters:?}"));
        }
        Ok(parameters)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct WaveBuffer {
    start: u32,
    end: u32,
    looped: bool,
    loop_start: u32,
    loop_end: u32,
    loop_count: i32,
    /// Copied from the client and not yet handed to rendering.
    pending: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
enum PlayState {
    Started,
    #[default]
    Stopped,
    Paused,
    RequestStop,
}

#[derive(Clone, Debug, Default)]
struct Voice {
    play_state: PlayState,
    sample_rate: u32,
    pitch: f32,
    buffers: [WaveBuffer; MAX_WAVE_BUFFERS],
    valid: [bool; MAX_WAVE_BUFFERS],
    index: usize,
    /// Samples played in the current pass over the current buffer.
    offset: u64,
    /// Whether the current buffer is replaying its loop range.
    in_loop: bool,
    /// Fractional input samples carried between frames.
    fraction: f64,
    played: u64,
    consumed: u32,
    flush: u32,
    /// Set by a new voice until the next frame renders; its status reads 0.
    new: bool,
}

impl Voice {
    /// Eden's UpdateParametersForCommandGeneration, then this frame's input
    /// samples for a started voice.
    fn render(&mut self, frame_seconds: f64) {
        let flush = std::mem::take(&mut self.flush);
        for _ in 0..flush {
            let index = self.index;
            self.buffers[index].pending = false;
            if self.valid[index] {
                self.valid[index] = false;
                self.index = (index + 1) % MAX_WAVE_BUFFERS;
                self.consumed = self.consumed.wrapping_add(1);
                self.offset = 0;
                self.in_loop = false;
            }
        }
        self.new = false;
        match self.play_state {
            PlayState::Started => {
                for (buffer, valid) in self.buffers.iter_mut().zip(&mut self.valid) {
                    if buffer.pending {
                        buffer.pending = false;
                        *valid = true;
                    }
                }
                self.advance(frame_seconds);
            }
            PlayState::RequestStop => {
                for index in 0..MAX_WAVE_BUFFERS {
                    self.buffers[index].pending = false;
                    if self.valid[index] {
                        self.valid[index] = false;
                        self.index = (self.index + 1) % MAX_WAVE_BUFFERS;
                        self.consumed = self.consumed.wrapping_add(1);
                    }
                }
                self.offset = 0;
                self.in_loop = false;
                self.played = 0;
                self.fraction = 0.0;
                self.play_state = PlayState::Stopped;
            }
            PlayState::Stopped | PlayState::Paused => {}
        }
    }

    /// Consume `frame_seconds` of input at the voice's rate from the
    /// valid buffers in order, looping and finishing buffers as the DSP does.
    fn advance(&mut self, frame_seconds: f64) {
        let rate = f64::from(self.sample_rate) * f64::from(self.pitch.max(0.0));
        let wanted = frame_seconds * rate + self.fraction;
        let mut remaining = wanted.floor() as u64;
        self.fraction = wanted - wanted.floor();
        // Each pass finishes, loops or partly plays a buffer; the bound keeps
        // zero-length loops finite.
        for _ in 0..64 {
            if remaining == 0 || !self.valid[self.index] {
                return;
            }
            let buffer = &mut self.buffers[self.index];
            let (start, end) = if self.in_loop && buffer.loop_end > buffer.loop_start {
                (buffer.loop_start, buffer.loop_end)
            } else {
                (buffer.start, buffer.end)
            };
            let length = u64::from(end.saturating_sub(start));
            let left = length.saturating_sub(self.offset);
            if remaining < left {
                self.offset += remaining;
                self.played += remaining;
                return;
            }
            remaining -= left;
            self.played += left;
            self.offset = 0;
            // A looping buffer replays its loop range (or whole range)
            // loop_count more times, or forever when it is negative.
            if buffer.looped && buffer.loop_count != 0 && length > 0 {
                if buffer.loop_count > 0 {
                    buffer.loop_count -= 1;
                }
                self.in_loop = true;
                continue;
            }
            self.in_loop = false;
            self.valid[self.index] = false;
            self.index = (self.index + 1) % MAX_WAVE_BUFFERS;
            self.consumed = self.consumed.wrapping_add(1);
            self.offset = 0;
        }
    }
}

/// One IAudioRenderer session.
#[derive(Clone, Debug)]
pub struct Renderer {
    pub parameters: Parameters,
    pub started: bool,
    pub frames: u64,
    pub updates: u64,
    next_frame: Option<Instant>,
    voices: BTreeMap<u32, Voice>,
    /// Usage of each effect slot: (type, new, enabled).
    effects: BTreeMap<usize, (u8, bool, bool)>,
    pub rendering_time_limit: u32,
    /// SetVoiceDropParameter's f32 bits (Eden-NS2's default 1.0). Voices
    /// are never dropped here, so it is only reported back.
    pub voice_drop_parameter: u32,
}

impl Renderer {
    pub fn new(parameters: Parameters) -> Self {
        Self {
            parameters,
            started: false,
            frames: 0,
            updates: 0,
            next_frame: None,
            voices: BTreeMap::new(),
            effects: BTreeMap::new(),
            rendering_time_limit: 100,
            voice_drop_parameter: 1.0f32.to_bits(),
        }
    }

    fn frame_period(&self) -> Duration {
        Duration::from_nanos(
            u64::from(self.parameters.sample_count) * 1_000_000_000
                / u64::from(self.parameters.sample_rate),
        )
    }

    pub fn set_started(&mut self, started: bool, now: Instant) {
        if started && !self.started {
            self.next_frame = Some(now + self.frame_period());
        }
        if !started {
            self.next_frame = None;
        }
        self.started = started;
    }

    /// When the next frame renders, while started.
    pub fn next_deadline(&self) -> Option<Instant> {
        self.next_frame
    }

    /// Render the frames due by `now`. Returns whether any rendered (the
    /// system event is then signaled).
    pub fn advance(&mut self, now: Instant) -> bool {
        let Some(mut next) = self.next_frame else {
            return false;
        };
        let period = self.frame_period();
        let mut rendered = 0;
        while next <= now && rendered < MAX_CATCH_UP_FRAMES {
            self.render_frame();
            next += period;
            rendered += 1;
        }
        if next <= now {
            // A long stall: drop the backlog rather than replaying it.
            next = now + period;
        }
        self.next_frame = Some(next);
        rendered > 0
    }

    fn render_frame(&mut self) {
        let frame_seconds =
            f64::from(self.parameters.sample_count) / f64::from(self.parameters.sample_rate);
        for voice in self.voices.values_mut() {
            voice.render(frame_seconds);
        }
        for usage in self.effects.values_mut() {
            usage.1 = false;
        }
        self.frames += 1;
    }

    /// RequestUpdate: apply `input` and build the `output_size`-byte
    /// output. The layout variant (effect status version, render info) is
    /// the one whose size equals the application's output buffer.
    pub fn update(&mut self, input: &[u8], output_size: usize) -> Result<Vec<u8>, String> {
        if input.len() < HEADER_SIZE {
            return Err(format!("update input is {:#x} bytes", input.len()));
        }
        let word = |at: usize| crate::bytes::le_u32(&input[at..at + 4]) as usize;
        let sizes = [
            word(0x04), // behavior
            word(0x08), // memory pools
            word(0x0c), // voices
            word(0x10), // voice channel resources
            word(0x14), // effects
            word(0x18), // mixes
            word(0x1c), // sinks
            word(0x20), // performance
            word(0x24), // splitter
            word(0x28), // render info
        ];
        let total: usize = HEADER_SIZE + sizes.iter().sum::<usize>();
        if total > input.len() {
            return Err(format!(
                "update sections ({total:#x} bytes) exceed the {:#x}-byte input",
                input.len()
            ));
        }
        let [
            behavior,
            pools,
            voices,
            resources,
            effects,
            _mixes,
            sinks,
            ..,
        ] = sizes;
        let voice_count = self.parameters.voices as usize;
        let effect_count = self.parameters.effects as usize;
        let sink_count = self.parameters.sinks as usize;
        if pools % MEMORY_POOL_IN != 0
            || voice_count == 0 && voices != 0
            || voice_count != 0 && !matches!(voices / voice_count, 0x170 | 0x188)
            || voice_count != 0 && voices % voice_count != 0
            || effects != effect_count * EFFECT_IN
            || sinks != sink_count * SINK_IN
        {
            return Err(format!(
                "update section sizes {sizes:x?} for {voice_count} voices, {effect_count} effects, {sink_count} sinks"
            ));
        }
        let pool_count = pools / MEMORY_POOL_IN;
        let fixed = HEADER_SIZE
            + pool_count * MEMORY_POOL_OUT
            + voice_count * VOICE_OUT
            + sink_count * SINK_OUT
            + PERFORMANCE_OUT
            + BEHAVIOR_OUT;
        let (effect_out, render_info) = [
            (EFFECT_OUT_V2, true),
            (EFFECT_OUT_V2, false),
            (EFFECT_OUT_V1, true),
            (EFFECT_OUT_V1, false),
        ]
        .into_iter()
        .find(|&(effect, info)| {
            fixed + effect_count * effect + if info { RENDER_INFO_OUT } else { 0 } == output_size
        })
        .ok_or_else(|| {
            format!("no update output layout is {output_size:#x} bytes (fixed part {fixed:#x})")
        })?;

        let mut output = vec![0u8; output_size];
        let mut out = HEADER_SIZE;
        let mut at = HEADER_SIZE + behavior;
        // Memory pools: RequestAttach (4) -> Attached (5), RequestDetach (2)
        // -> Detached (3), for page-aligned nonempty ranges.
        for index in 0..pool_count {
            let entry = &input[at + index * MEMORY_POOL_IN..];
            let address = crate::bytes::le_u64(&entry[..8]);
            let size = crate::bytes::le_u64(&entry[8..16]);
            let state = crate::bytes::le_u32(&entry[16..20]);
            let aligned = address != 0 && size != 0 && address % 0x1000 == 0 && size % 0x1000 == 0;
            let out_state = match state {
                4 if aligned => 5u32,
                2 if aligned => 3,
                _ => 0,
            };
            output[out + index * MEMORY_POOL_OUT..][..4].copy_from_slice(&out_state.to_le_bytes());
        }
        at += pools + resources;
        out += pool_count * MEMORY_POOL_OUT;
        // Voices.
        if voice_count != 0 {
            let stride = voices / voice_count;
            for index in 0..voice_count {
                let status =
                    self.update_voice(&input[at + index * stride..][..stride], stride == 0x188);
                if let Some((played, consumed)) = status {
                    let slot = &mut output[out + index * VOICE_OUT..][..VOICE_OUT];
                    slot[..8].copy_from_slice(&played.to_le_bytes());
                    slot[8..12].copy_from_slice(&consumed.to_le_bytes());
                }
            }
        }
        at += voices;
        out += voice_count * VOICE_OUT;
        // Effects: StoreStatus with the renderer's started state (OutStatus
        // Used 3, Removed 4).
        for index in 0..effect_count {
            let entry = &input[at + index * EFFECT_IN..];
            let (kind, new, enabled) = (entry[0], entry[1] != 0, entry[2] != 0);
            let usage = self.effects.entry(index).or_insert((kind, new, enabled));
            if usage.0 != kind || new {
                *usage = (kind, true, enabled);
            }
            usage.2 = enabled;
            let disabled = !usage.1 && !usage.2;
            let used = if self.started { !disabled } else { usage.1 };
            output[out + index * effect_out] = if used { 3 } else { 4 };
        }
        out += effect_count * effect_out;
        // Sinks report no circular-buffer progress.
        out += sink_count * SINK_OUT;
        // Performance and error info stay zero (no history, no errors).
        out += PERFORMANCE_OUT + BEHAVIOR_OUT;
        if render_info {
            output[out..out + 8].copy_from_slice(&self.frames.to_le_bytes());
            out += RENDER_INFO_OUT;
        }
        debug_assert_eq!(out, output_size);
        // Header: the client's revision, then each section's output size.
        output[..4].copy_from_slice(&input[..4]);
        let header = [
            (0x04, BEHAVIOR_OUT),
            (0x08, pool_count * MEMORY_POOL_OUT),
            (0x0c, voice_count * VOICE_OUT),
            (0x14, effect_count * effect_out),
            (0x1c, sink_count * SINK_OUT),
            (0x20, PERFORMANCE_OUT),
            (0x28, if render_info { RENDER_INFO_OUT } else { 0 }),
            (0x3c, output_size),
        ];
        for (offset, value) in header {
            output[offset..offset + 4].copy_from_slice(&(value as u32).to_le_bytes());
        }
        self.updates += 1;
        Ok(output)
    }

    /// One voice's input parameter (0x170, or 0x188 with float biquads).
    /// Returns its (played samples, consumed buffers) status, or None for
    /// a voice not in use.
    fn update_voice(&mut self, entry: &[u8], v2: bool) -> Option<(u64, u32)> {
        let word = |at: usize| crate::bytes::le_u32(&entry[at..at + 4]);
        let (id, is_new, in_use) = (word(0), entry[8] != 0, entry[9] != 0);
        if !in_use {
            return None;
        }
        let voice = self.voices.entry(id).or_default();
        if is_new {
            *voice = Voice {
                new: true,
                ..Voice::default()
            };
        }
        voice.sample_rate = word(0xc);
        voice.pitch = f32::from_bits(word(0x1c));
        let (buffers_at, flush_at) = if v2 { (0x78, 0x171) } else { (0x60, 0x159) };
        voice.flush += u32::from(entry[flush_at]);
        voice.play_state = match entry[0xa] {
            0 => PlayState::Started,
            1 if voice.play_state != PlayState::Stopped => PlayState::RequestStop,
            1 => PlayState::Stopped,
            2 => PlayState::Paused,
            _ => voice.play_state,
        };
        for index in 0..MAX_WAVE_BUFFERS {
            let buffer = &entry[buffers_at + index * 0x38..][..0x38];
            let sent = buffer[0x1a] != 0;
            if sent {
                continue;
            }
            let field = |at: usize| crate::bytes::le_u32(&buffer[at..at + 4]);
            voice.buffers[index] = WaveBuffer {
                start: field(0x10),
                end: field(0x14),
                looped: buffer[0x18] != 0,
                loop_count: field(0x1c) as i32,
                loop_start: field(0x30),
                loop_end: field(0x34),
                pending: true,
            };
        }
        Some(if voice.new {
            (0, 0)
        } else {
            (voice.played, voice.consumed)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parameters(voices: u32) -> Parameters {
        Parameters {
            sample_rate: 48_000,
            sample_count: 240,
            mixes: 2,
            voices,
            sinks: 1,
            effects: 0,
            revision: u32::from_le_bytes(*b"REv@"),
        }
    }

    /// An update input with one memory pool and one 0x188-byte voice.
    fn input(voice: &[u8; 0x188]) -> Vec<u8> {
        let mut input = vec![0u8; HEADER_SIZE];
        input[..4].copy_from_slice(b"REv@");
        let sizes = [(0x04, 0x10), (0x08, 0x20), (0x0c, 0x188), (0x1c, SINK_IN)];
        for (offset, size) in sizes {
            input[offset..offset + 4].copy_from_slice(&(size as u32).to_le_bytes());
        }
        input.extend([0u8; 0x10]); // behavior
        let mut pool = [0u8; 0x20];
        pool[..8].copy_from_slice(&0x10_0000u64.to_le_bytes());
        pool[8..16].copy_from_slice(&0x1000u64.to_le_bytes());
        pool[16..20].copy_from_slice(&4u32.to_le_bytes()); // RequestAttach
        input.extend(pool);
        input.extend(voice);
        input.extend([0u8; SINK_IN]);
        input
    }

    #[test]
    fn voices_consume_buffers_as_frames_render() {
        let mut renderer = Renderer::new(parameters(1));
        let mut voice = [0u8; 0x188];
        voice[8] = 1; // new
        voice[9] = 1; // in use
        voice[0xa] = 0; // started
        voice[0xc..0x10].copy_from_slice(&48_000u32.to_le_bytes());
        voice[0x1c..0x20].copy_from_slice(&1.0f32.to_bits().to_le_bytes());
        // One 480-sample buffer, not yet sent; the others already sent.
        for index in 0..4 {
            voice[0x78 + index * 0x38 + 0x1a] = 1;
        }
        voice[0x78 + 0x14..0x78 + 0x18].copy_from_slice(&480u32.to_le_bytes());
        voice[0x78 + 0x1a] = 0;
        let size =
            HEADER_SIZE + 0x10 + 0x10 + SINK_OUT + PERFORMANCE_OUT + BEHAVIOR_OUT + RENDER_INFO_OUT;
        let output = renderer.update(&input(&voice), size).unwrap();
        assert_eq!(crate::bytes::le_u32(&output[0x40..0x44]), 5); // Attached
        assert_eq!(crate::bytes::le_u32(&output[0x3c..0x40]) as usize, size);
        // Started: frames render every 5 ms; two frames play the buffer.
        let start = Instant::now();
        renderer.set_started(true, start);
        assert!(!renderer.advance(start));
        assert!(renderer.advance(start + Duration::from_millis(5)));
        voice[8] = 0;
        voice[0x78 + 0x1a] = 1; // the client marks it sent
        let output = renderer.update(&input(&voice), size).unwrap();
        let status = &output[0x50..0x60];
        assert_eq!(crate::bytes::le_u64(&status[..8]), 240);
        assert_eq!(crate::bytes::le_u32(&status[8..12]), 0);
        renderer.advance(start + Duration::from_millis(10));
        let output = renderer.update(&input(&voice), size).unwrap();
        let status = &output[0x50..0x60];
        assert_eq!(crate::bytes::le_u64(&status[..8]), 480);
        assert_eq!(crate::bytes::le_u32(&status[8..12]), 1);
        // Render info counts the frames.
        assert_eq!(crate::bytes::le_u64(&output[size - 0x10..size - 8]), 2);
    }

    #[test]
    fn output_sizes_must_match_a_layout() {
        let mut renderer = Renderer::new(parameters(1));
        assert!(renderer.update(&input(&[0u8; 0x188]), 0x1000).is_err());
    }
}
