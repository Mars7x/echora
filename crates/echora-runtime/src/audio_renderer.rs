//! IAudioRenderer state: the update protocol and real-time voice progress.
//!
//! Layouts and rules follow Eden-NS2 1a6d7d4 (`audio_core/renderer`:
//! `behavior/info_updater.cpp`, `voice/voice_info.{h,cpp}`,
//! `memory/pool_mapper.cpp`, `effect/effect_info_base.h`). Each rendered
//! frame decodes the started voices' wave buffers at their pitch, filters
//! and scales them, mixes them into their mixes' buffers, mixes those down
//! to the final mix and hands its buffers to the device sink
//! (`audio_mix` holds the signal processing). A mix's buffer mixer, Reverb,
//! I3DL2 reverb and aux effects run on its buffers in their processing
//! order (`audio_effects` holds the reverbs). Not yet processed: the delay,
//! biquad, limiter, capture and compressor effects, the depop fade of
//! stopped voices and the performance counters; the survey in the run
//! report counts what the application asks for.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use crate::audio_effects::{self as effects, I3dl2, I3dl2Parameters, Reverb, ReverbParameters};
use crate::audio_mix::{self as mix, AdpcmContext, BiquadState, Format, Quality, Source};

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
    address: u64,
    context: u64,
    context_size: u64,
    stream_ended: bool,
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
    format: Format,
    channels: u32,
    volume: f32,
    previous_volume: f32,
    mix: i32,
    splitter: i32,
    quality: Quality,
    /// Each of the two filters: enabled, numerator, denominator.
    biquads: [(bool, [f32; 3], [f32; 2]); 2],
    /// ADPCM predictor coefficients' address and size.
    coefficients: (u64, u64),
    /// The channel resource (mix volumes) of each channel.
    resources: [u32; 6],
    /// Times the current buffer has looped (the decoder's count).
    loops: u32,
    channel_state: Vec<ChannelState>,
}

/// One voice channel's decoder and filter state between frames.
#[derive(Clone, Debug, Default)]
struct ChannelState {
    history: Vec<i16>,
    fraction: u64,
    adpcm: AdpcmContext,
    biquads: [BiquadState; 2],
}

/// A voice channel's volumes into its mix's buffers, with last frame's
/// (the ramp's start).
#[derive(Clone, Copy, Debug, Default)]
struct ChannelResource {
    volumes: [f32; MAX_MIX_BUFFERS],
    previous: [f32; MAX_MIX_BUFFERS],
}

/// A mix as the application last set it.
#[derive(Clone, Debug)]
struct Mix {
    volume: f32,
    buffers: usize,
    destination: i32,
    splitter: i32,
    /// Volume from each of its buffers into each destination buffer.
    volumes: Vec<[f32; MAX_MIX_BUFFERS]>,
}

/// A splitter destination: a mix and the volume into each of its buffers
/// (`previous` is where this frame's ramp starts).
#[derive(Clone, Copy, Debug, Default)]
struct Destination {
    in_use: bool,
    mix: i32,
    volumes: [f32; MAX_MIX_BUFFERS],
    previous: [f32; MAX_MIX_BUFFERS],
    /// Mixed into this frame: `previous` then follows `volumes`.
    touched: bool,
}

/// A buffer mixer effect: (input, output, volume) per pair, on its mix.
#[derive(Clone, Debug)]
struct BufferMixer {
    mix: u32,
    order: u32,
    pairs: Vec<(usize, usize, f32)>,
}

/// An effect's ParameterState: the reverbs build their state on
/// Initialized, take parameters on Updating and then stand at Updated.
const STATE_INITIALIZED: u8 = 0;
const STATE_UPDATING: u8 = 1;
const STATE_UPDATED: u8 = 2;

/// The sizes of an aux buffer's two info blocks (the application's, then
/// the renderer's); the samples follow them.
const AUX_INFO_SIZE: u64 = 0x40;
const AUX_BUFFER_AT: u64 = 0x80;

/// An effect with signal processing of its own, on its mix.
#[derive(Clone, Debug)]
struct Effect {
    mix: u32,
    order: u32,
    enabled: bool,
    processor: Processor,
}

#[derive(Clone, Debug)]
enum Processor {
    Reverb {
        parameters: ReverbParameters,
        pending: u8,
        state: Option<Box<Reverb>>,
    },
    I3dl2 {
        parameters: I3dl2Parameters,
        pending: u8,
        state: Option<Box<I3dl2>>,
    },
    /// The application's own processing: each input goes out through the
    /// send buffer and each output comes back from the return buffer, both
    /// rings of `count_max` samples in its memory.
    Aux {
        pairs: Vec<(usize, usize)>,
        count_max: u32,
        send: u64,
        back: u64,
    },
}

impl Processor {
    /// The same effect type (its state carries over an update).
    fn kind(&self) -> u8 {
        match self {
            Self::Aux { .. } => 2,
            Self::Reverb { .. } => 4,
            Self::I3dl2 { .. } => 5,
        }
    }
}

/// Zero an aux buffer's read and write positions and its sample total.
fn reset_aux(memory: &dyn Source, info: u64) {
    let mut bytes = [0u8; 0x10];
    if info != 0 && memory.read(info, &mut bytes) {
        bytes[..8].fill(0);
        bytes[0xc..].fill(0);
        memory.write(info, &bytes);
    }
}

/// Write `input` into the send ring `offset` samples after its write
/// position, which then moves on by `update`.
fn write_aux(
    memory: &dyn Source,
    base: u64,
    count_max: u32,
    input: &[i32],
    offset: u32,
    update: u32,
) {
    let info = base + AUX_INFO_SIZE;
    let mut position = [0u8; 4];
    if count_max == 0 || input.len() > count_max as usize || !memory.read(info + 4, &mut position) {
        return;
    }
    let position = u32::from_le_bytes(position);
    let mut target = position.wrapping_add(offset);
    if target > count_max {
        return;
    }
    let mut done = 0;
    while done < input.len() {
        let run = ((count_max - target) as usize).min(input.len() - done);
        let bytes: Vec<u8> = input[done..done + run]
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect();
        memory.write(base + AUX_BUFFER_AT + u64::from(target) * 4, &bytes);
        target = (target + run as u32) % count_max;
        done += run;
    }
    if update != 0 {
        let next = position.wrapping_add(update) % count_max;
        memory.write(info + 4, &next.to_le_bytes());
    }
}

/// Fill `output` from the return ring `offset` samples after its read
/// position, which then moves on by `update`. Returns whether it was read.
fn read_aux(
    memory: &dyn Source,
    base: u64,
    count_max: u32,
    output: &mut [i32],
    offset: u32,
    update: u32,
) -> bool {
    let info = base + AUX_INFO_SIZE;
    let mut position = [0u8; 4];
    if count_max == 0 || output.len() > count_max as usize || !memory.read(info, &mut position) {
        return false;
    }
    let position = u32::from_le_bytes(position);
    let mut target = position.wrapping_add(offset);
    if target > count_max {
        return false;
    }
    let mut done = 0;
    while done < output.len() {
        let run = ((count_max - target) as usize).min(output.len() - done);
        let mut bytes = vec![0u8; run * 4];
        if !memory.read(base + AUX_BUFFER_AT + u64::from(target) * 4, &mut bytes) {
            bytes.fill(0);
        }
        for (sample, bytes) in output[done..done + run]
            .iter_mut()
            .zip(bytes.chunks_exact(4))
        {
            *sample = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        }
        target = (target + run as u32) % count_max;
        done += run;
    }
    if update != 0 {
        let next = position.wrapping_add(update) % count_max;
        memory.write(info, &next.to_le_bytes());
    }
    true
}

impl Effect {
    /// Run on its mix's buffers for one frame of `count` samples.
    fn process(
        &mut self,
        buffers: &mut [Vec<i32>],
        count: usize,
        memory: &dyn Source,
        long_pre_delay: bool,
    ) {
        match &mut self.processor {
            Processor::Reverb {
                parameters,
                pending,
                state,
            } => {
                if self.enabled {
                    match (*pending, state.as_deref_mut()) {
                        (STATE_UPDATING, Some(state)) => state.update(parameters),
                        (STATE_INITIALIZED, _) | (_, None) => {
                            *state = Some(Box::new(Reverb::new(parameters, long_pre_delay)));
                        }
                        _ => {}
                    }
                    if let Some(state) = state {
                        state.process(parameters, buffers, count);
                    }
                } else {
                    let channels = parameters.channels.min(6);
                    effects::bypass(
                        buffers,
                        &parameters.inputs[..channels],
                        &parameters.outputs[..channels],
                    );
                }
                *pending = STATE_UPDATED;
            }
            Processor::I3dl2 {
                parameters,
                pending,
                state,
            } => {
                if self.enabled {
                    match (*pending, state.as_deref_mut()) {
                        (STATE_UPDATING, Some(state)) => state.update(parameters),
                        (STATE_INITIALIZED, _) | (_, None) => {
                            *state = Some(Box::new(I3dl2::new(parameters)));
                        }
                        _ => {}
                    }
                    if let Some(state) = state {
                        state.process(parameters, buffers, count);
                    }
                } else {
                    let channels = parameters.channels.min(6);
                    effects::bypass(
                        buffers,
                        &parameters.inputs[..channels],
                        &parameters.outputs[..channels],
                    );
                }
                *pending = STATE_UPDATED;
            }
            Processor::Aux {
                pairs,
                count_max,
                send,
                back,
            } => {
                if *send == 0 || *back == 0 {
                    return;
                }
                if !self.enabled {
                    reset_aux(memory, *send + AUX_INFO_SIZE);
                    reset_aux(memory, *back + AUX_INFO_SIZE);
                    let (inputs, outputs): (Vec<usize>, Vec<usize>) = pairs.iter().copied().unzip();
                    effects::bypass(buffers, &inputs, &outputs);
                    return;
                }
                // Channel after channel in the rings; the last one moves
                // their positions past the whole frame.
                let total = (pairs.len() * count) as u32;
                for (channel, &(input, output)) in pairs.iter().enumerate() {
                    if input >= buffers.len() || output >= buffers.len() {
                        continue;
                    }
                    let offset = (channel * count) as u32;
                    let update = if channel + 1 == pairs.len() { total } else { 0 };
                    write_aux(
                        memory,
                        *send,
                        *count_max,
                        &buffers[input][..count],
                        offset,
                        update,
                    );
                    if !read_aux(
                        memory,
                        *back,
                        *count_max,
                        &mut buffers[output][..count],
                        offset,
                        update,
                    ) {
                        buffers[output][..count].fill(0);
                    }
                }
            }
        }
    }
}

const MAX_MIX_BUFFERS: usize = 24;
const FINAL_MIX: i32 = 0;
/// A voice or mix with no mix (or splitter) to go to.
const UNUSED_MIX: i32 = i32::MAX;
const UNUSED_SPLITTER: i32 = -1;
/// The decoder's scratch size: a frame never reads more input samples.
const TEMP_BUFFER_SIZE: u64 = 0x3f00;
/// At most this many interleaved samples wait for the host device.
const OUTPUT_LIMIT: usize = 48_000 * 6;

impl Voice {
    /// Eden's UpdateParametersForCommandGeneration, then this frame's input
    /// samples for a started voice.
    /// Returns whether the voice is playing this frame. `timed` advances
    /// it by time alone (no guest memory to decode from).
    fn render(&mut self, frame_seconds: f64, timed: bool) -> bool {
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
                if timed {
                    self.advance(frame_seconds);
                }
                return true;
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
                self.loops = 0;
                self.channel_state.clear();
                self.previous_volume = 0.0;
                self.play_state = PlayState::Stopped;
            }
            PlayState::Stopped | PlayState::Paused => {}
        }
        false
    }

    /// This frame's `count` output samples of channel `channel`: wave
    /// buffers decoded from the voice's position and resampled by its
    /// pitch. `commit` keeps the advanced position (the last channel's
    /// pass; every channel reads the same range).
    fn source(
        &mut self,
        memory: &dyn Source,
        channel: usize,
        target_rate: u32,
        count: usize,
        commit: bool,
    ) -> Vec<i32> {
        let mut output = vec![0i32; count];
        let ratio = mix::ratio(self.sample_rate, target_rate, self.pitch);
        let taps = self.quality.history();
        if self.channel_state.len() <= channel {
            self.channel_state
                .resize(channel + 1, ChannelState::default());
        }
        let mut state = std::mem::take(&mut self.channel_state[channel]);
        state.history.resize(taps, 0);
        let wanted = (state.fraction + count as u64 * ratio) >> 15;
        if ratio == 0 || taps as u64 + wanted > TEMP_BUFFER_SIZE {
            self.channel_state[channel] = state;
            return output;
        }
        let wanted = wanted as u32;
        let mut temp = state.history.clone();
        // The position, advanced on a copy unless this pass commits.
        let (mut index, mut offset, mut loops) = (self.index, self.offset as u32, self.loops);
        let (mut played, mut consumed) = (self.played, self.consumed);
        let mut valid = self.valid;
        let mut read = 0u32;
        while read < wanted {
            if !valid[index] {
                break;
            }
            let buffer = self.buffers[index];
            if offset == 0 && self.format == Format::Adpcm && buffer.context != 0 {
                let mut bytes = [0u8; 6];
                let length = (buffer.context_size as usize).min(6);
                if memory.read(buffer.context, &mut bytes[..length]) {
                    state.adpcm = AdpcmContext::parse(&bytes);
                }
            }
            let (start, end) = if buffer.looped && loops > 0 && buffer.loop_start <= buffer.loop_end
            {
                (buffer.loop_start, buffer.loop_end)
            } else {
                (buffer.start, buffer.end)
            };
            let decoded = match self.format {
                Format::Pcm16 | Format::PcmFloat => mix::decode_pcm(
                    memory,
                    self.format == Format::PcmFloat,
                    buffer.address,
                    self.channels.max(1),
                    channel as u32,
                    start,
                    end,
                    offset,
                    wanted - read,
                    &mut temp,
                ),
                Format::Adpcm => {
                    let mut coefficients = [0i16; 16];
                    let mut bytes = [0u8; 32];
                    let length = (self.coefficients.1 as usize).min(32);
                    if memory.read(self.coefficients.0, &mut bytes[..length]) {
                        for (slot, pair) in coefficients.iter_mut().zip(bytes.chunks_exact(2)) {
                            *slot = i16::from_le_bytes([pair[0], pair[1]]);
                        }
                    }
                    mix::decode_adpcm(
                        memory,
                        buffer.address,
                        &coefficients,
                        &mut state.adpcm,
                        start,
                        end,
                        offset,
                        wanted - read,
                        &mut temp,
                    )
                }
                Format::Invalid => 0,
            };
            played += u64::from(decoded);
            read += decoded;
            offset += decoded;
            if decoded != 0 && offset < end.saturating_sub(start) {
                continue;
            }
            offset = 0;
            // Whether the buffer ends here (a looping one after its count).
            let looping = buffer.looped
                && !(buffer.loop_count >= 0
                    && (loops + 1 > buffer.loop_count as u32 || decoded == 0));
            if looping {
                loops += 1;
            } else {
                valid[index] = false;
                loops = 0;
                if buffer.stream_ended {
                    played = 0;
                }
                index = (index + 1) % MAX_WAVE_BUFFERS;
                consumed = consumed.wrapping_add(1);
            }
            if buffer.looped && decoded == 0 {
                break;
            }
        }
        // Input the buffers could not supply reads as silence.
        temp.resize(taps + wanted as usize, 0);
        mix::resample(&mut output, &temp, ratio, &mut state.fraction, self.quality);
        state.history = temp[wanted as usize..].to_vec();
        self.channel_state[channel] = state;
        if commit {
            self.index = index;
            self.offset = u64::from(offset);
            self.loops = loops;
            self.played = played;
            self.consumed = consumed;
            self.valid = valid;
        }
        output
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
    /// What the application's updates ask for, counted by kind (voice
    /// formats and routing, mixes, effect and sink types): the survey the
    /// mixer is built from.
    pub survey: BTreeMap<String, u64>,
    resources: BTreeMap<u32, ChannelResource>,
    mixes: BTreeMap<i32, Mix>,
    mixers: BTreeMap<usize, BufferMixer>,
    /// The effect slots with signal processing of their own.
    processors: BTreeMap<usize, Effect>,
    /// Each splitter's destination ids, in order.
    splitters: BTreeMap<i32, Vec<i32>>,
    destinations: BTreeMap<i32, Destination>,
    /// The device sink's inputs: the final mix buffer of each channel.
    device: Option<Vec<usize>>,
    /// Interleaved 16-bit samples rendered for the device and not yet
    /// taken (`take_output`), with their channel count.
    output: Vec<i16>,
    pub output_channels: u32,
    /// Frames in which any voice was mixed.
    pub mixed_frames: u64,
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
            survey: BTreeMap::new(),
            resources: BTreeMap::new(),
            mixes: BTreeMap::new(),
            mixers: BTreeMap::new(),
            processors: BTreeMap::new(),
            splitters: BTreeMap::new(),
            destinations: BTreeMap::new(),
            device: None,
            output: Vec::new(),
            output_channels: 0,
            mixed_frames: 0,
        }
    }

    /// The run report's entry for this renderer.
    pub fn report(&self) -> serde_json::Value {
        serde_json::json!({
            "sample_rate": self.parameters.sample_rate,
            "sample_count": self.parameters.sample_count,
            "voices": self.parameters.voices,
            "mixes": self.parameters.mixes,
            "sinks": self.parameters.sinks,
            "effects": self.parameters.effects,
            "revision": format!("{:#x}", self.parameters.revision),
            "started": self.started,
            "frames_rendered": self.frames,
            "updates": self.updates,
            "survey": self.survey,
            "mixed_frames": self.mixed_frames,
            "output_channels": self.output_channels,
            "mixing_implemented": true,
        })
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
    pub fn advance(&mut self, now: Instant, memory: Option<&dyn Source>) -> bool {
        let Some(mut next) = self.next_frame else {
            return false;
        };
        let period = self.frame_period();
        let mut rendered = 0;
        while next <= now && rendered < MAX_CATCH_UP_FRAMES {
            self.render_frame(memory);
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

    /// The interleaved samples rendered for the device since the last call.
    pub fn take_output(&mut self) -> Vec<i16> {
        std::mem::take(&mut self.output)
    }

    fn render_frame(&mut self, memory: Option<&dyn Source>) {
        let frame_seconds =
            f64::from(self.parameters.sample_count) / f64::from(self.parameters.sample_rate);
        let count = self.parameters.sample_count as usize;
        let rate = self.parameters.sample_rate;
        // Every mix's buffers, silent.
        let mut buffers: BTreeMap<i32, Vec<Vec<i32>>> = self
            .mixes
            .iter()
            .map(|(&id, mix)| (id, vec![vec![0i32; count]; mix.buffers]))
            .collect();
        let mut mixed = false;
        for voice in self.voices.values_mut() {
            let playing = voice.render(frame_seconds, memory.is_none());
            let Some(memory) = memory else {
                continue;
            };
            if !playing {
                continue;
            }
            let channels = (voice.channels.max(1) as usize).min(voice.resources.len());
            for channel in 0..channels {
                let mut samples =
                    voice.source(memory, channel, rate, count, channel + 1 == channels);
                for filter in 0..2 {
                    let (enabled, b, a) = voice.biquads[filter];
                    if enabled {
                        let state = &mut voice.channel_state[channel].biquads[filter];
                        mix::biquad(&mut samples, b, a, state);
                    }
                }
                mix::volume_ramp(&mut samples, voice.previous_volume, voice.volume);
                if voice.mix != UNUSED_MIX {
                    let Some(resource) = self.resources.get_mut(&voice.resources[channel]) else {
                        continue;
                    };
                    if let Some(target) = buffers.get_mut(&voice.mix) {
                        for (index, buffer) in target.iter_mut().enumerate() {
                            mix::mix_ramp(
                                buffer,
                                &samples,
                                resource.previous[index],
                                resource.volumes[index],
                            );
                        }
                        mixed = true;
                    }
                    resource.previous = resource.volumes;
                } else if voice.splitter != UNUSED_SPLITTER {
                    // Channel c feeds the splitter's destinations c,
                    // c + channels, c + 2 channels, ...
                    let ids = self
                        .splitters
                        .get(&voice.splitter)
                        .cloned()
                        .unwrap_or_default();
                    for id in ids.iter().skip(channel).step_by(channels) {
                        let Some(destination) = self.destinations.get_mut(id) else {
                            continue;
                        };
                        if !destination.in_use || destination.mix == UNUSED_MIX {
                            continue;
                        }
                        if let Some(target) = buffers.get_mut(&destination.mix) {
                            for (index, buffer) in target.iter_mut().enumerate() {
                                mix::mix_ramp(
                                    buffer,
                                    &samples,
                                    destination.previous[index],
                                    destination.volumes[index],
                                );
                            }
                            destination.touched = true;
                            mixed = true;
                        }
                    }
                }
            }
            voice.previous_volume = voice.volume;
        }
        if let Some(memory) = memory {
            self.mix_down(&mut buffers, count, memory);
            if mixed {
                self.mixed_frames += 1;
            }
        }
        for usage in self.effects.values_mut() {
            usage.1 = false;
        }
        self.frames += 1;
    }

    /// Each mix into its destination, farthest from the final mix first;
    /// then the final mix's volume and the device sink.
    fn mix_down(
        &mut self,
        buffers: &mut BTreeMap<i32, Vec<Vec<i32>>>,
        count: usize,
        memory: &dyn Source,
    ) {
        // A mix's distance from the final mix (0); unconnected ones last.
        let distance = |mut id: i32| {
            let mut steps = 0;
            while id != FINAL_MIX {
                match self.mixes.get(&id) {
                    // Through a splitter: its first destination's mix.
                    Some(mix) if steps < 64 && mix.destination == UNUSED_MIX => {
                        let next = self
                            .splitters
                            .get(&mix.splitter)
                            .and_then(|ids| ids.first())
                            .and_then(|id| self.destinations.get(id))
                            .map(|destination| destination.mix);
                        match next {
                            Some(next) if next != UNUSED_MIX => id = next,
                            _ => return i32::MAX,
                        }
                    }
                    Some(mix) if steps < 64 => id = mix.destination,
                    _ => return i32::MAX,
                }
                steps += 1;
            }
            steps
        };
        let mut order: Vec<(i32, i32)> = self.mixes.keys().map(|&id| (distance(id), id)).collect();
        order.sort_unstable_by(|a, b| b.cmp(a));
        // Revision 3 lengthened the Reverb's pre-delay line.
        let long_pre_delay = (self.parameters.revision >> 24).saturating_sub(u32::from(b'0')) >= 3;
        for (steps, id) in order {
            // The mix's own effects, in their processing order.
            let mut slots: Vec<(u32, usize)> = self
                .mixers
                .iter()
                .filter(|(_, mixer)| mixer.mix as i32 == id)
                .map(|(&slot, mixer)| (mixer.order, slot))
                .chain(
                    self.processors
                        .iter()
                        .filter(|(_, effect)| effect.mix as i32 == id)
                        .map(|(&slot, effect)| (effect.order, slot)),
                )
                .collect();
            slots.sort_unstable();
            if let Some(own) = buffers.get_mut(&id) {
                for (_, slot) in slots {
                    if let Some(mixer) = self.mixers.get(&slot) {
                        for &(input, output, volume) in &mixer.pairs {
                            if input < own.len() && output < own.len() && input != output {
                                let source = own[input].clone();
                                mix::mix(&mut own[output], &source, volume);
                            }
                        }
                    } else if let Some(effect) = self.processors.get_mut(&slot) {
                        effect.process(own, count, memory, long_pre_delay);
                    }
                }
            }
            if id == FINAL_MIX || steps == i32::MAX {
                continue;
            }
            let mix = &self.mixes[&id];
            let Some(source) = buffers.get(&id).cloned() else {
                continue;
            };
            if mix.destination == UNUSED_MIX {
                // Through a splitter: destination d takes this mix's buffer
                // d modulo its buffer count.
                if mix.splitter == UNUSED_SPLITTER || source.is_empty() {
                    continue;
                }
                let ids = self
                    .splitters
                    .get(&mix.splitter)
                    .cloned()
                    .unwrap_or_default();
                for (index, destination_id) in ids.iter().enumerate() {
                    let Some(destination) = self.destinations.get(destination_id) else {
                        continue;
                    };
                    if !destination.in_use || destination.mix == UNUSED_MIX {
                        continue;
                    }
                    let samples = &source[index % source.len()];
                    if let Some(target) = buffers.get_mut(&destination.mix) {
                        for (to, buffer) in target.iter_mut().enumerate() {
                            mix::mix(buffer, samples, mix.volume * destination.volumes[to]);
                        }
                    }
                }
                continue;
            }
            let Some(target) = buffers.get_mut(&mix.destination) else {
                continue;
            };
            for (from, samples) in source.iter().enumerate() {
                for (to, buffer) in target.iter_mut().enumerate() {
                    mix::mix(buffer, samples, mix.volume * mix.volumes[from][to]);
                }
            }
        }
        // Destinations mixed into this frame start the next from here.
        for destination in self.destinations.values_mut() {
            if destination.in_use && std::mem::take(&mut destination.touched) {
                destination.previous = destination.volumes;
            }
        }
        let (Some(mix), Some(device)) = (self.mixes.get(&FINAL_MIX), self.device.as_ref()) else {
            return;
        };
        let Some(last) = buffers.get(&FINAL_MIX) else {
            return;
        };
        self.output_channels = device.len() as u32;
        for sample in 0..count {
            for &input in device {
                let value = last
                    .get(input)
                    .map_or(0.0, |buffer| buffer[sample] as f32 * mix.volume);
                self.output.push(value.clamp(-32768.0, 32767.0) as i16);
            }
        }
        if self.output.len() > OUTPUT_LIMIT {
            let excess = self.output.len() - OUTPUT_LIMIT;
            self.output.drain(..excess);
        }
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
        let splitter_size = sizes[8];
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
            mixes,
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
        // Voice channel resources (0x70 each): a channel's volume into each
        // buffer of its voice's mix.
        for entry in input[at + pools..][..resources].chunks_exact(0x70) {
            if entry[0x64] == 0 {
                continue;
            }
            let id = crate::bytes::le_u32(&entry[..4]);
            let resource = self.resources.entry(id).or_default();
            for (index, volume) in resource.volumes.iter_mut().enumerate() {
                *volume = f32::from_bits(crate::bytes::le_u32(&entry[4 + index * 4..][..4]));
            }
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
            if kind != 0 {
                let mix = crate::bytes::le_u32(&entry[4..8]);
                *self
                    .survey
                    .entry(format!("effect type {kind} mix {mix} enabled {enabled}"))
                    .or_default() += 1;
            }
            if kind == 1 && enabled {
                let specific = &entry[0x20..];
                let pairs = crate::bytes::le_u32(&specific[0x90..0x94]).min(24) as usize;
                self.mixers.insert(
                    index,
                    BufferMixer {
                        mix: crate::bytes::le_u32(&entry[4..8]),
                        order: crate::bytes::le_u32(&entry[0x18..0x1c]),
                        pairs: (0..pairs)
                            .map(|pair| {
                                (
                                    specific[pair] as usize,
                                    specific[0x18 + pair] as usize,
                                    f32::from_bits(crate::bytes::le_u32(
                                        &specific[0x30 + pair * 4..][..4],
                                    )),
                                )
                            })
                            .filter(|&(_, _, volume)| volume != 0.0)
                            .collect(),
                    },
                );
            } else {
                self.mixers.remove(&index);
            }
            let specific = &entry[0x20..EFFECT_IN];
            let kept = self
                .processors
                .get(&index)
                .is_some_and(|effect| effect.processor.kind() == kind && !new);
            // What the reverb's next frame does: build its state, or take
            // the parameters when the application marks them changed.
            let pending = |kept: Option<u8>, wanted: u8| match kept {
                Some(STATE_UPDATED) => wanted,
                Some(pending) => pending,
                None => STATE_INITIALIZED,
            };
            let processor = match (kind, self.processors.remove(&index).filter(|_| kept)) {
                (4, old) => {
                    let (was, state) = match old.map(|effect| effect.processor) {
                        Some(Processor::Reverb { pending, state, .. }) => (Some(pending), state),
                        _ => (None, None),
                    };
                    Some(Processor::Reverb {
                        parameters: ReverbParameters::parse(specific),
                        pending: pending(was, specific[0x40]),
                        state,
                    })
                }
                (5, old) => {
                    let (was, state) = match old.map(|effect| effect.processor) {
                        Some(Processor::I3dl2 { pending, state, .. }) => (Some(pending), state),
                        _ => (None, None),
                    };
                    Some(Processor::I3dl2 {
                        parameters: I3dl2Parameters::parse(specific),
                        pending: pending(was, specific[0x48]),
                        state,
                    })
                }
                (2, _) => {
                    let word = |at: usize| crate::bytes::le_u32(&specific[at..at + 4]);
                    let address = |at: usize| crate::bytes::le_u64(&specific[at..at + 8]);
                    Some(Processor::Aux {
                        pairs: (0..word(0x30).min(24) as usize)
                            .map(|pair| (specific[pair] as usize, specific[0x18 + pair] as usize))
                            .collect(),
                        count_max: word(0x38),
                        send: address(0x40),
                        back: address(0x50),
                    })
                }
                _ => None,
            };
            // The parameters' shape, for the run report.
            let shape = match &processor {
                Some(Processor::Reverb { parameters, .. }) => Some(format!(
                    "reverb channels {} rate {:#x} early mode {} late mode {} decay {:#x}",
                    parameters.channels,
                    parameters.sample_rate,
                    parameters.early_mode,
                    parameters.late_mode,
                    parameters.decay_time
                )),
                Some(Processor::I3dl2 { parameters, .. }) => Some(format!(
                    "I3DL2 reverb channels {} rate {} decay {} s",
                    parameters.channels, parameters.sample_rate, parameters.late_reverb_decay_time
                )),
                Some(Processor::Aux {
                    pairs,
                    count_max,
                    send,
                    back,
                }) => Some(format!(
                    "aux buffers {} ring {count_max} mapped {}",
                    pairs.len(),
                    *send != 0 && *back != 0
                )),
                None => None,
            };
            if let Some(shape) = shape.filter(|_| enabled) {
                *self.survey.entry(shape).or_default() += 1;
            }
            if let Some(processor) = processor {
                self.processors.insert(
                    index,
                    Effect {
                        mix: crate::bytes::le_u32(&entry[4..8]),
                        order: crate::bytes::le_u32(&entry[0x18..0x1c]),
                        enabled,
                        processor,
                    },
                );
            }
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
        // Mixes (0x930 each, after a 0x20 dirty-count header in later
        // revisions) and sinks, surveyed only.
        // The splitter section follows the effects: a header ("SNDH", info
        // count, destination count), each splitter ("SNDI", id, rate, its
        // destination ids) and each destination ("SNDD", id, mix volumes,
        // mix; 0x70 bytes, or 0x90 or 0xa8 with biquads).
        let splitter_at = at + effects;
        if splitter_size >= 0x20 {
            let section = &input[splitter_at..splitter_at + splitter_size];
            let word = |at: usize| {
                section
                    .get(at..at + 4)
                    .map_or(0, |bytes| crate::bytes::le_u32(bytes))
            };
            let (infos, destinations) = (word(4) as usize, word(8) as usize);
            let mut offset = 0x20;
            for _ in 0..infos.min(1024) {
                if word(offset) != u32::from_le_bytes(*b"SNDI") {
                    break;
                }
                let count = (word(offset + 0xc) as usize).min(256);
                let ids = (0..count)
                    .map(|index| word(offset + 0x10 + index * 4) as i32)
                    .collect();
                self.splitters.insert(word(offset + 4) as i32, ids);
                offset += 0x1c + count * 4;
            }
            // Each destination's size shows in where the next one starts
            // (the section may end in padding).
            let magic = u32::from_le_bytes(*b"SNDD");
            for _ in 0..destinations.min(4096) {
                if word(offset) != magic {
                    break;
                }
                let stride = [0x70usize, 0x90, 0xa8].into_iter().find(|size| {
                    let next = offset + size;
                    next == section.len() || (next + 4 <= section.len() && word(next) == magic)
                });
                let Some(stride) = stride.or_else(|| {
                    // The last one, before padding: the largest that fits.
                    [0xa8usize, 0x90, 0x70]
                        .into_iter()
                        .find(|size| offset + size <= section.len())
                }) else {
                    break;
                };
                let at = offset;
                let in_use = section[at + stride - if stride == 0x70 { 4 } else { 12 }] != 0;
                let destination = self.destinations.entry(word(at + 4) as i32).or_default();
                for (buffer, volume) in destination.volumes.iter_mut().enumerate() {
                    *volume = f32::from_bits(word(at + 8 + buffer * 4));
                }
                destination.mix = word(at + 0x68) as i32;
                if !destination.in_use && in_use {
                    destination.previous = destination.volumes;
                }
                destination.in_use = in_use;
                offset += stride;
            }
        }
        // Mixes (0x930 each). Later revisions send only the changed ones,
        // after a 0x20 header with their count; each names its own id.
        let mixes_at = splitter_at + splitter_size;
        let header = mixes % 0x930;
        let listed = if header == 0x20 {
            (crate::bytes::le_u32(&input[mixes_at + 4..][..4]) as usize).min(mixes / 0x930)
        } else {
            mixes / 0x930
        };
        for index in 0..listed {
            let entry = &input[mixes_at + header + index * 0x930..][..0x930];
            let field = |at: usize| crate::bytes::le_u32(&entry[at..at + 4]);
            let id = if header == 0x20 {
                field(0x10) as i32
            } else {
                index as i32
            };
            if entry[0xc] == 0 {
                self.mixes.remove(&id);
                continue;
            }
            let count = (field(8) as usize).min(MAX_MIX_BUFFERS);
            self.mixes.insert(
                id,
                Mix {
                    volume: f32::from_bits(field(0)),
                    buffers: count,
                    destination: field(0x924) as i32,
                    splitter: field(0x928) as i32,
                    volumes: (0..count)
                        .map(|from| {
                            std::array::from_fn(|to| {
                                f32::from_bits(field(0x24 + (from * MAX_MIX_BUFFERS + to) * 4))
                            })
                        })
                        .collect(),
                },
            );
            *self
                .survey
                .entry(format!(
                    "mix {} rate {} buffers {} effects {} volume {} to mix {} splitter {}",
                    field(0x10) as i32,
                    field(4),
                    field(8),
                    field(0x14),
                    f32::from_bits(field(0)),
                    field(0x924) as i32,
                    field(0x928) as i32
                ))
                .or_default() += 1;
        }
        let sinks_at = mixes_at + mixes;
        for index in 0..sink_count {
            let entry = &input[sinks_at + index * SINK_IN..][..SINK_IN];
            if entry[1] == 0 {
                continue;
            }
            let detail = match entry[0] {
                // Device: its name and input count.
                1 => {
                    let inputs = (crate::bytes::le_u32(&entry[0x120..0x124]) as usize).min(6);
                    self.device = Some(
                        entry[0x124..0x124 + inputs]
                            .iter()
                            .map(|&input| input as usize)
                            .collect(),
                    );
                    let name: String = entry[0x20..0x120]
                        .iter()
                        .take_while(|&&byte| byte != 0)
                        .map(|&byte| byte as char)
                        .collect();
                    format!(
                        "device {name:?} inputs {} downmix {}",
                        crate::bytes::le_u32(&entry[0x120..0x124]),
                        entry[0x12b]
                    )
                }
                // Circular buffer: size, inputs, sample count and format.
                2 => format!(
                    "circular buffer size {:#x} inputs {} samples {} format {}",
                    crate::bytes::le_u32(&entry[0x28..0x2c]),
                    crate::bytes::le_u32(&entry[0x2c..0x30]),
                    crate::bytes::le_u32(&entry[0x30..0x34]),
                    entry[0x38]
                ),
                other => format!("type {other}"),
            };
            *self.survey.entry(format!("sink {detail}")).or_default() += 1;
        }
        if splitter_size != 0 {
            *self
                .survey
                .entry(format!("splitter section {splitter_size:#x} bytes"))
                .or_default() += 1;
        }
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
        let (mix_at, biquad_stride) = if v2 { (0x70, 0x18) } else { (0x58, 0xc) };
        let survey = format!(
            "voice format {} channels {} rate {} to mix {} splitter {} biquads {}{} adpcm table {} quality {}",
            entry[0xb],
            word(0x18),
            word(0xc),
            word(mix_at) as i32,
            word(mix_at + 4) as i32,
            entry[0x24],
            entry[0x24 + biquad_stride],
            word(mix_at - 0x10) != 0 || word(mix_at - 0xc) != 0,
            entry[flush_at + 5]
        );
        voice.flush += u32::from(entry[flush_at]);
        *self.survey.entry(survey).or_default() += 1;
        voice.format = Format::from_raw(entry[0xb]);
        voice.channels = word(0x18).min(6);
        voice.volume = f32::from_bits(word(0x20));
        voice.mix = word(mix_at) as i32;
        voice.splitter = word(mix_at + 4) as i32;
        voice.quality = Quality::from_raw(entry[flush_at + 5]);
        voice.coefficients = (
            crate::bytes::le_u64(&entry[mix_at - 0x10..][..8]),
            crate::bytes::le_u64(&entry[mix_at - 8..][..8]),
        );
        for filter in 0..2 {
            let at = 0x24 + filter * biquad_stride;
            voice.biquads[filter] = if v2 {
                let float = |offset: usize| f32::from_bits(word(at + offset));
                (
                    entry[at] != 0,
                    [float(4), float(8), float(12)],
                    [float(16), float(20)],
                )
            } else {
                // Q14 coefficients.
                let fixed = |offset: usize| {
                    f32::from(i16::from_le_bytes([
                        entry[at + offset],
                        entry[at + offset + 1],
                    ])) / 16384.0
                };
                (
                    entry[at] != 0,
                    [fixed(2), fixed(4), fixed(6)],
                    [fixed(8), fixed(10)],
                )
            };
        }
        let resources_at = buffers_at + MAX_WAVE_BUFFERS * 0x38;
        for (channel, id) in voice.resources.iter_mut().enumerate() {
            *id = word(resources_at + channel * 4);
        }
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
                address: crate::bytes::le_u64(&buffer[..8]),
                context: crate::bytes::le_u64(&buffer[0x20..0x28]),
                context_size: crate::bytes::le_u64(&buffer[0x28..0x30]),
                stream_ended: buffer[0x19] != 0,
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
        assert!(!renderer.advance(start, None));
        assert!(renderer.advance(start + Duration::from_millis(5), None));
        voice[8] = 0;
        voice[0x78 + 0x1a] = 1; // the client marks it sent
        let output = renderer.update(&input(&voice), size).unwrap();
        let status = &output[0x50..0x60];
        assert_eq!(crate::bytes::le_u64(&status[..8]), 240);
        assert_eq!(crate::bytes::le_u32(&status[8..12]), 0);
        renderer.advance(start + Duration::from_millis(10), None);
        let output = renderer.update(&input(&voice), size).unwrap();
        let status = &output[0x50..0x60];
        assert_eq!(crate::bytes::le_u64(&status[..8]), 480);
        assert_eq!(crate::bytes::le_u32(&status[8..12]), 1);
        // Render info counts the frames.
        assert_eq!(crate::bytes::le_u64(&output[size - 0x10..size - 8]), 2);
    }

    struct Wave(Vec<u8>);
    impl Source for Wave {
        fn read(&self, address: u64, out: &mut [u8]) -> bool {
            let start = (address - 0x2000) as usize;
            match self.0.get(start..start + out.len()) {
                Some(bytes) => {
                    out.copy_from_slice(bytes);
                    true
                }
                None => false,
            }
        }
    }

    /// An update with one mono PCM16 voice (480 samples at 0x2000), a
    /// two-buffer final mix and a stereo device sink. The voice reaches
    /// the mix through channel resource 7 (volumes 1 and 0.5), or with
    /// `splitter` through splitter 3's destination 5 (volumes 1 and 0.25).
    fn scene(splitter: bool) -> Vec<u8> {
        let mut voice = [0u8; 0x188];
        voice[8] = 1;
        voice[9] = 1;
        voice[0xb] = 2; // PcmInt16
        voice[0xc..0x10].copy_from_slice(&48_000u32.to_le_bytes());
        voice[0x18..0x1c].copy_from_slice(&1u32.to_le_bytes());
        voice[0x1c..0x20].copy_from_slice(&1.0f32.to_bits().to_le_bytes());
        voice[0x20..0x24].copy_from_slice(&1.0f32.to_bits().to_le_bytes());
        let (mix, to) = if splitter { (i32::MAX, 3i32) } else { (0, -1) };
        voice[0x70..0x74].copy_from_slice(&mix.to_le_bytes());
        voice[0x74..0x78].copy_from_slice(&to.to_le_bytes());
        voice[0x158..0x15c].copy_from_slice(&7u32.to_le_bytes());
        voice[0x176] = 2; // low quality: samples pass through at pitch 1
        for index in 1..4 {
            voice[0x78 + index * 0x38 + 0x1a] = 1;
        }
        voice[0x78..0x80].copy_from_slice(&0x2000u64.to_le_bytes());
        voice[0x78 + 0x14..0x78 + 0x18].copy_from_slice(&480u32.to_le_bytes());
        let mut section = Vec::new();
        if splitter {
            section.extend(b"SNDH");
            section.extend(1u32.to_le_bytes()); // one splitter
            section.extend(1u32.to_le_bytes()); // one destination
            section.extend([0u8; 0x14]);
            section.extend(b"SNDI");
            section.extend(3u32.to_le_bytes());
            section.extend(48_000u32.to_le_bytes());
            section.extend(1u32.to_le_bytes());
            section.extend(5u32.to_le_bytes()); // its destination's id
            section.extend([0u8; 12]);
            let mut destination = [0u8; 0xa8];
            destination[..4].copy_from_slice(b"SNDD");
            destination[4..8].copy_from_slice(&5u32.to_le_bytes());
            destination[8..12].copy_from_slice(&1.0f32.to_bits().to_le_bytes());
            destination[12..16].copy_from_slice(&0.25f32.to_bits().to_le_bytes());
            destination[0x68..0x6c].copy_from_slice(&0u32.to_le_bytes());
            destination[0x9c] = 1;
            section.extend(destination);
        }
        let mut input = vec![0u8; HEADER_SIZE];
        input[..4].copy_from_slice(b"REv@");
        let sizes = [
            (0x04, 0x10),
            (0x0c, 0x188),
            (0x10, 0x70),
            (0x18, 0x930),
            (0x1c, SINK_IN),
            (0x24, section.len()),
        ];
        for (offset, size) in sizes {
            input[offset..offset + 4].copy_from_slice(&(size as u32).to_le_bytes());
        }
        input.extend([0u8; 0x10]); // behavior
        let mut resource = [0u8; 0x70];
        resource[..4].copy_from_slice(&7u32.to_le_bytes());
        resource[4..8].copy_from_slice(&1.0f32.to_bits().to_le_bytes());
        resource[8..12].copy_from_slice(&0.5f32.to_bits().to_le_bytes());
        resource[0x64] = 1;
        input.extend(resource);
        input.extend(voice);
        input.extend(section);
        let mut mix = vec![0u8; 0x930];
        mix[..4].copy_from_slice(&1.0f32.to_bits().to_le_bytes());
        mix[8..12].copy_from_slice(&2u32.to_le_bytes()); // two buffers
        mix[0xc] = 1;
        input.extend(mix);
        let mut sink = [0u8; SINK_IN];
        sink[0] = 1; // device
        sink[1] = 1;
        sink[0x120..0x124].copy_from_slice(&2u32.to_le_bytes());
        sink[0x124] = 0;
        sink[0x125] = 1;
        input.extend(sink);
        input
    }

    /// Two frames of `scene` over a wave counting up in tens.
    fn play(splitter: bool) -> (Renderer, Vec<i16>) {
        let mut renderer = Renderer::new(parameters(1));
        let size = HEADER_SIZE + 0x10 + SINK_OUT + PERFORMANCE_OUT + BEHAVIOR_OUT + RENDER_INFO_OUT;
        renderer.update(&scene(splitter), size).unwrap();
        let wave = Wave((0..480i16).flat_map(|n| (n * 10).to_le_bytes()).collect());
        let start = Instant::now();
        renderer.set_started(true, start);
        renderer.advance(start + Duration::from_millis(10), Some(&wave));
        let output = renderer.take_output();
        (renderer, output)
    }

    #[test]
    fn a_voice_is_decoded_mixed_and_sent_to_the_device() {
        let (renderer, output) = play(false);
        assert_eq!(renderer.output_channels, 2);
        assert_eq!(output.len(), 2 * 480);
        // The second frame plays at the settled volumes: left whole, right
        // half. The resampler's four history samples delay the input.
        for n in 240..480usize {
            let sample = (n as i16 - 4) * 10;
            assert_eq!(output[2 * n], sample, "left {n}");
            assert_eq!(output[2 * n + 1], sample / 2, "right {n}");
        }
        // The first frame ramps both volumes up from silence.
        assert_eq!(output[0], 0);
        assert!(output[2 * 120] > 0 && output[2 * 120] < 1200);
        assert_eq!(renderer.mixed_frames, 2);
    }

    #[test]
    fn a_voice_reaches_its_mix_through_a_splitter() {
        let (renderer, output) = play(true);
        assert_eq!(output.len(), 2 * 480);
        // A destination that comes into use starts at its volumes: no
        // ramp from silence, apart from the voice's own volume ramp.
        for n in 240..480usize {
            let sample = (n as i16 - 4) * 10;
            assert_eq!(output[2 * n], sample, "left {n}");
            assert_eq!(output[2 * n + 1], sample / 4, "right {n}");
        }
        assert_eq!(renderer.mixed_frames, 2);
    }

    #[test]
    fn output_sizes_must_match_a_layout() {
        let mut renderer = Renderer::new(parameters(1));
        assert!(renderer.update(&input(&[0u8; 0x188]), 0x1000).is_err());
    }

    /// Guest memory from address 0 for the aux effect's rings.
    struct Ram(std::cell::RefCell<Vec<u8>>);

    impl Source for Ram {
        fn read(&self, address: u64, out: &mut [u8]) -> bool {
            let ram = self.0.borrow();
            let Some(bytes) = ram.get(address as usize..address as usize + out.len()) else {
                return false;
            };
            out.copy_from_slice(bytes);
            true
        }

        fn write(&self, address: u64, bytes: &[u8]) -> bool {
            let mut ram = self.0.borrow_mut();
            let Some(to) = ram.get_mut(address as usize..address as usize + bytes.len()) else {
                return false;
            };
            to.copy_from_slice(bytes);
            true
        }
    }

    #[test]
    fn aux_sends_inputs_and_takes_the_return() {
        // Two rings of six samples: send at 0x100, return at 0x200.
        let ram = Ram(std::cell::RefCell::new(vec![0u8; 0x300]));
        let word = |address: u64| {
            let mut bytes = [0u8; 4];
            assert!(ram.read(address, &mut bytes));
            i32::from_le_bytes(bytes)
        };
        for (index, sample) in [10i32, 20, 30, 40, 50, 60].into_iter().enumerate() {
            ram.write(0x280 + index as u64 * 4, &sample.to_le_bytes());
        }
        let mut effect = Effect {
            mix: 0,
            order: 0,
            enabled: true,
            processor: Processor::Aux {
                pairs: vec![(0, 0), (1, 1)],
                count_max: 6,
                send: 0x100,
                back: 0x200,
            },
        };
        let mut buffers = vec![vec![1, 2], vec![3, 4]];
        effect.process(&mut buffers, 2, &ram, true);
        // Both channels went out one after the other and came back so.
        assert_eq!(
            [word(0x180), word(0x184), word(0x188), word(0x18c)],
            [1, 2, 3, 4]
        );
        assert_eq!(buffers, vec![vec![10, 20], vec![30, 40]]);
        // The renderer's write and read positions moved past the frame.
        assert_eq!((word(0x144), word(0x240)), (4, 4));
        // The next frame wraps around both rings.
        let mut buffers = vec![vec![5, 6], vec![7, 8]];
        effect.process(&mut buffers, 2, &ram, true);
        assert_eq!(
            [word(0x190), word(0x194), word(0x180), word(0x184)],
            [5, 6, 7, 8]
        );
        assert_eq!(buffers, vec![vec![50, 60], vec![10, 20]]);
        assert_eq!((word(0x144), word(0x240)), (2, 2));
        // Disabled, it passes through and rewinds the rings.
        effect.enabled = false;
        let mut buffers = vec![vec![9, 9], vec![8, 8]];
        effect.process(&mut buffers, 2, &ram, true);
        assert_eq!(buffers, vec![vec![9, 9], vec![8, 8]]);
        assert_eq!((word(0x144), word(0x240)), (0, 0));
    }
}
