//! The audio renderer's reverberation effects: the signal processing of the
//! Reverb and I3DL2 reverb effect types on a mix's buffers.
//!
//! Tables and structure follow Eden-NS2 1a6d7d4
//! (`audio_core/renderer/command/effect/reverb.cpp`, `i3dl2_reverb.cpp`,
//! `effect/reverb.h`, `effect/i3dl2.h`), with parameter changes applied at
//! once as the console does. Samples are a mix buffer's 32-bit values. The
//! Reverb works on them scaled by 64 and its delay lines hold whole numbers;
//! the I3DL2 reverb's lines hold the unscaled values exactly (whole numbers
//! there would leave a tail of rounding noise that never dies away).

/// The low-frequency channel of a six-channel effect (the centre is 2).
const LFE: usize = 3;

/// A parameter in the console's 18.14 fixed point.
fn q14(value: i32) -> f64 {
    f64::from(value) / 16384.0
}

/// Ten to the `value`, with the console's limits.
fn pow_10(value: f64) -> f64 {
    if value >= 0.0 {
        1.0
    } else if value <= -5.3 {
        0.0
    } else {
        10f64.powf(value)
    }
}

fn cos_degrees(degrees: f64) -> f64 {
    degrees.to_radians().cos()
}

fn sin_degrees(degrees: f64) -> f64 {
    degrees.to_radians().sin()
}

/// Copy each input to its output where they differ (a disabled effect).
pub fn bypass(buffers: &mut [Vec<i32>], inputs: &[usize], outputs: &[usize]) {
    for (&input, &output) in inputs.iter().zip(outputs) {
        if input != output && input < buffers.len() && output < buffers.len() {
            let source = buffers[input].clone();
            buffers[output].copy_from_slice(&source);
        }
    }
}

/// The channel counts the reverbs process; others pass through.
fn channels_processed(count: usize, inputs: &[usize], outputs: &[usize], buffers: usize) -> bool {
    matches!(count, 1 | 2 | 4 | 6)
        && inputs.len() >= count
        && outputs.len() >= count
        && inputs[..count]
            .iter()
            .chain(&outputs[..count])
            .all(|&index| index < buffers)
}

/// The Reverb effect's delay line: reads leave at `output`, writes enter at
/// `input`, both wrapping at `end`, the delay in use (the wrap `tap` reads
/// by).
#[derive(Clone, Debug, Default)]
struct ReverbLine {
    buffer: Vec<f64>,
    end: usize,
    input: usize,
    output: usize,
    count: i32,
    max: i32,
    decay: f64,
}

impl ReverbLine {
    fn new(delay: u32, decay: f64) -> Self {
        Self {
            buffer: vec![0.0; delay as usize + 1],
            end: delay as usize,
            input: 0,
            output: 0,
            count: delay as i32,
            max: delay as i32,
            decay,
        }
    }

    /// The delay in use, at most the line's length.
    fn set_count(&mut self, count: i32) {
        self.count = count.clamp(0, self.max);
        self.end = self.count as usize;
    }

    fn read(&self) -> f64 {
        self.buffer[self.output]
    }

    fn write(&mut self, sample: f64) {
        self.buffer[self.input] = sample.floor();
        self.input += 1;
        if self.input >= self.end {
            self.input = 0;
        }
    }

    fn tick(&mut self, sample: f64) -> f64 {
        let out = self.read();
        self.output += 1;
        if self.output >= self.end {
            self.output = 0;
        }
        self.write(sample);
        out
    }

    /// The sample written `index + 1` writes ago.
    fn tap(&self, index: i32) -> f64 {
        let mut at = self.input as i64 - (i64::from(index) + 1);
        if at < 0 {
            at += i64::from(self.count);
        }
        usize::try_from(at)
            .ok()
            .and_then(|at| self.buffer.get(at))
            .copied()
            .unwrap_or(0.0)
    }

    fn clear(&mut self) {
        self.buffer.fill(0.0);
    }
}

const REVERB_LINES: usize = 4;
const REVERB_TAPS: usize = 10;

/// Line lengths in milliseconds.
const FDN_MAX_TIMES: [f64; REVERB_LINES] = [
    53.9532470703125,
    79.19256591796875,
    116.23876953125,
    170.61529541015625,
];
const DECAY_MAX_TIMES: [f64; REVERB_LINES] = [7.0, 9.0, 13.0, 17.0];

/// Early reflection times in milliseconds per early mode; the eleventh is
/// the late reverb's.
const EARLY_TIMES: [[f64; REVERB_TAPS + 1]; 5] = [
    [
        0.000000, 3.500000, 2.799988, 3.899963, 2.699951, 13.399963, 7.899963, 8.399963, 9.899963,
        12.000000, 12.500000,
    ],
    [
        0.000000, 11.799988, 5.500000, 11.199951, 10.399963, 38.099976, 22.199951, 29.599976,
        21.199951, 24.799988, 40.000000,
    ],
    [
        0.000000, 41.500000, 20.500000, 41.299988, 0.000000, 29.500000, 33.799988, 45.199951,
        46.799988, 0.000000, 50.000000,
    ],
    [
        33.099976, 43.299988, 22.799988, 37.899963, 14.899963, 35.299988, 17.899963, 34.199951,
        0.000000, 43.299988, 50.000000,
    ],
    [0.0; REVERB_TAPS + 1],
];

const EARLY_GAINS: [[f64; REVERB_TAPS]; 5] = [
    [
        0.699951, 0.679993, 0.699951, 0.679993, 0.699951, 0.679993, 0.699951, 0.679993, 0.679993,
        0.679993,
    ],
    [
        0.699951, 0.679993, 0.699951, 0.679993, 0.699951, 0.679993, 0.679993, 0.679993, 0.679993,
        0.679993,
    ],
    [
        0.500000, 0.699951, 0.699951, 0.679993, 0.500000, 0.679993, 0.679993, 0.699951, 0.679993,
        0.000000,
    ],
    [
        0.929993, 0.919983, 0.869995, 0.859985, 0.939941, 0.809998, 0.799988, 0.769958, 0.759949,
        0.649963,
    ],
    [0.0; REVERB_TAPS],
];

const FDN_TIMES: [[f64; REVERB_LINES]; 5] = [
    [53.953247, 79.192566, 116.238770, 130.615295],
    [53.953247, 79.192566, 116.238770, 170.615295],
    [5.000000, 10.000000, 5.000000, 10.000000],
    [47.029968, 71.000000, 103.000000, 170.000000],
    [53.953247, 79.192566, 116.238770, 170.615295],
];

const DECAY_TIMES: [[f64; REVERB_LINES]; 5] = [
    [7.0, 9.0, 13.0, 17.0],
    [7.0, 9.0, 13.0, 17.0],
    [1.0, 1.0, 1.0, 1.0],
    [7.0, 7.0, 13.0, 9.0],
    [7.0, 9.0, 13.0, 17.0],
];

/// The Reverb effect's parameters as the game wrote them (gains, times and
/// the sample rate in kHz are 18.14 fixed point).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReverbParameters {
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
    pub channels: usize,
    pub sample_rate: i32,
    pub early_mode: usize,
    pub early_gain: i32,
    pub pre_delay: i32,
    pub late_mode: usize,
    pub late_gain: i32,
    pub decay_time: i32,
    pub high_frequency_decay_ratio: i32,
    pub colouration: i32,
    pub base_gain: i32,
    pub wet_gain: i32,
    pub dry_gain: i32,
}

impl ReverbParameters {
    /// From an effect's 0xa0 specific bytes.
    pub fn parse(bytes: &[u8]) -> Self {
        let word = |at: usize| crate::bytes::le_u32(&bytes[at..at + 4]) as i32;
        let half = |at: usize| usize::from(u16::from_le_bytes([bytes[at], bytes[at + 1]]));
        let buffers = |at: usize| (0..6).map(|index| usize::from(bytes[at + index])).collect();
        Self {
            inputs: buffers(0),
            outputs: buffers(6),
            channels: half(0xe),
            sample_rate: word(0x10),
            early_mode: (word(0x14) as usize).min(4),
            early_gain: word(0x18),
            pre_delay: word(0x1c),
            late_mode: (word(0x20) as usize).min(4),
            late_gain: word(0x24),
            decay_time: word(0x28),
            high_frequency_decay_ratio: word(0x2c),
            colouration: word(0x30),
            base_gain: word(0x34),
            wet_gain: word(0x38),
            dry_gain: word(0x3c),
        }
    }
}

/// The Reverb effect's state.
#[derive(Clone, Debug)]
pub struct Reverb {
    pre_delay_line: ReverbLine,
    center_delay_line: ReverbLine,
    early_delay_times: [i32; REVERB_TAPS],
    early_gains: [f64; REVERB_TAPS],
    pre_delay_time: i32,
    decay_delay_lines: [ReverbLine; REVERB_LINES],
    fdn_delay_lines: [ReverbLine; REVERB_LINES],
    hf_decay_gain: [f64; REVERB_LINES],
    hf_decay_prev_gain: [f64; REVERB_LINES],
    prev_feedback_output: [f64; REVERB_LINES],
}

impl Reverb {
    /// A silent reverb for `parameters`; `long_pre_delay` is the later
    /// revisions' 350 ms pre-delay line (150 ms before).
    pub fn new(parameters: &ReverbParameters, long_pre_delay: bool) -> Self {
        let rate = q14(parameters.sample_rate);
        let samples = |milliseconds: f64| (milliseconds * rate).max(0.0).floor() as u32;
        let mut reverb = Self {
            pre_delay_line: ReverbLine::new(
                samples(if long_pre_delay { 350.0 } else { 150.0 }),
                1.0,
            ),
            center_delay_line: ReverbLine::new(samples(5.0), 1.0),
            early_delay_times: [0; REVERB_TAPS],
            early_gains: [0.0; REVERB_TAPS],
            pre_delay_time: 0,
            decay_delay_lines: std::array::from_fn(|line| {
                ReverbLine::new(samples(DECAY_MAX_TIMES[line]), 0.0)
            }),
            fdn_delay_lines: std::array::from_fn(|line| {
                ReverbLine::new(samples(FDN_MAX_TIMES[line]), 1.0)
            }),
            hf_decay_gain: [0.0; REVERB_LINES],
            hf_decay_prev_gain: [0.0; REVERB_LINES],
            prev_feedback_output: [0.0; REVERB_LINES],
        };
        reverb.update(parameters);
        for line in reverb
            .fdn_delay_lines
            .iter_mut()
            .chain(&mut reverb.decay_delay_lines)
        {
            line.clear();
        }
        reverb.center_delay_line.clear();
        reverb.pre_delay_line.clear();
        reverb
    }

    /// Take changed parameters.
    pub fn update(&mut self, parameters: &ReverbParameters) {
        let rate = q14(parameters.sample_rate);
        let pre_delay = q14(parameters.pre_delay);
        let early_times = &EARLY_TIMES[parameters.early_mode];
        for tap in 0..REVERB_TAPS {
            let delay =
                (((pre_delay + early_times[tap]) * rate) as i32).min(self.pre_delay_line.max);
            self.early_delay_times[tap] = delay + 1;
            self.early_gains[tap] =
                q14(parameters.early_gain) * EARLY_GAINS[parameters.early_mode][tap];
        }
        if parameters.channels == 2 {
            self.early_gains[4] *= 0.5;
            self.early_gains[5] *= 0.5;
        }
        self.pre_delay_time =
            (((pre_delay + early_times[REVERB_TAPS]) * rate) as i32).min(self.pre_delay_line.max);
        let high_cut = cos_degrees(1280.0 / rate);
        for line in 0..REVERB_LINES {
            let fdn = &mut self.fdn_delay_lines[line];
            fdn.set_count((FDN_TIMES[parameters.late_mode][line] * rate) as i32);
            let decay = &mut self.decay_delay_lines[line];
            decay.set_count((DECAY_TIMES[parameters.late_mode][line] * rate) as i32);
            decay.decay = 0.5999755859375 * (1.0 - q14(parameters.colouration));
            // -60 dB after the decay time over the two lines' lengths.
            let a =
                f64::from(self.fdn_delay_lines[line].max + self.decay_delay_lines[line].max) * -3.0;
            let b = a / (q14(parameters.decay_time) * rate);
            let ratio = q14(parameters.high_frequency_decay_ratio);
            let (mut c, mut d) = (0.0, 1.0);
            if ratio <= 0.99493408203125 {
                let e = pow_10(((1.0 / ratio) - 1.0) * 2.0 / 100.0 * (b / 10.0));
                let f = 1.0 - e;
                let g = 2.0 - high_cut * e * 2.0;
                let h = (g * g - f * f * 4.0).sqrt();
                let low_pass = (g - h) / (f * 2.0);
                if low_pass.is_finite() {
                    c = low_pass;
                    d = 1.0 - low_pass;
                }
            }
            self.hf_decay_prev_gain[line] = c;
            let gain = pow_10(b / 1000.0) * d * 0.70709228515625;
            self.hf_decay_gain[line] = if gain.is_finite() { gain } else { 0.0 };
            self.prev_feedback_output[line] = 0.0;
        }
    }

    /// Reverberate `count` samples of the input buffers into the outputs.
    pub fn process(
        &mut self,
        parameters: &ReverbParameters,
        buffers: &mut [Vec<i32>],
        count: usize,
    ) {
        let channels = parameters.channels;
        let (inputs, outputs) = (&parameters.inputs, &parameters.outputs);
        if channels == 0 {
            return;
        }
        if !channels_processed(channels, inputs, outputs, buffers.len()) {
            bypass(buffers, inputs, outputs);
            return;
        }
        let taps: [usize; REVERB_TAPS] = match channels {
            1 => [0; REVERB_TAPS],
            2 => [0, 0, 1, 1, 0, 1, 0, 0, 1, 1],
            4 => [0, 0, 1, 1, 0, 1, 2, 2, 3, 3],
            _ => [0, 0, 1, 1, 2, 2, 4, 4, 5, 5],
        };
        let (base_gain, late_gain) = (q14(parameters.base_gain), q14(parameters.late_gain));
        let (dry_gain, wet_gain) = (q14(parameters.dry_gain), q14(parameters.wet_gain));
        for sample in 0..count {
            let mut early = [0.0f64; 6];
            for tap in 0..REVERB_TAPS {
                let value =
                    self.pre_delay_line.tap(self.early_delay_times[tap]) * self.early_gains[tap];
                early[taps[tap]] += value;
                if channels == 6 {
                    early[LFE] += value;
                }
            }
            if channels == 6 {
                early[LFE] *= 0.2;
            }
            let dry: [i32; 6] = std::array::from_fn(|channel| {
                if channel < channels {
                    buffers[inputs[channel]][sample]
                } else {
                    0
                }
            });
            let input: f64 = dry.iter().map(|&value| f64::from(value)).sum();
            self.pre_delay_line.write(input * 64.0 * base_gain);
            for line in 0..REVERB_LINES {
                self.prev_feedback_output[line] = self.prev_feedback_output[line]
                    * self.hf_decay_prev_gain[line]
                    + self.fdn_delay_lines[line].read() * self.hf_decay_gain[line];
            }
            let late = self.pre_delay_line.tap(self.pre_delay_time) * late_gain;
            let feedback = self.prev_feedback_output;
            let mixed = [
                feedback[2] + feedback[1] + late,
                -feedback[0] - feedback[3] + late,
                feedback[0] - feedback[3] + late,
                feedback[1] - feedback[2] + late,
            ];
            let mut all_pass = [0.0f64; REVERB_LINES];
            for line in 0..REVERB_LINES {
                let decay = &mut self.decay_delay_lines[line];
                let value = mixed[line] - decay.read() * decay.decay;
                let out = decay.tick(value) + value * decay.decay;
                self.fdn_delay_lines[line].tick(out);
                all_pass[line] = out;
            }
            let late_out: [f64; 6] = if channels == 6 {
                [
                    all_pass[0],
                    all_pass[1],
                    self.center_delay_line
                        .tick((all_pass[2] - all_pass[3]) * 0.5),
                    all_pass[3],
                    all_pass[2],
                    all_pass[3],
                ]
            } else {
                [all_pass[0], all_pass[1], all_pass[2], all_pass[3], 0.0, 0.0]
            };
            for channel in 0..channels {
                let wet = (early[channel] + late_out[channel]) * wet_gain / 64.0;
                buffers[outputs[channel]][sample] =
                    (f64::from(dry[channel]) * dry_gain + wet) as i32;
            }
        }
    }
}

/// The I3DL2 reverb's delay line: a write enters at `input`, `delay` ahead
/// of where reads leave, both wrapping over the whole buffer (the wrap `tap`
/// reads by).
#[derive(Clone, Debug, Default)]
struct I3dl2Line {
    buffer: Vec<f64>,
    end: usize,
    max: i32,
    input: usize,
    output: usize,
    delay: i32,
    wet_gain: f64,
}

impl I3dl2Line {
    fn new(delay: u32) -> Self {
        let mut line = Self {
            buffer: vec![0.0; delay as usize + 1],
            end: delay as usize + 1,
            max: delay as i32,
            ..Self::default()
        };
        line.set_delay(delay as i32);
        line
    }

    fn set_delay(&mut self, delay: i32) {
        if self.max < delay || delay < 0 {
            return;
        }
        self.delay = delay;
        self.input = (self.output + delay as usize) % (self.max as usize + 1);
    }

    fn read(&self) -> f64 {
        self.buffer[self.output]
    }

    fn write(&mut self, sample: f64) {
        self.buffer[self.input] = sample;
        self.input += 1;
        if self.input >= self.end {
            self.input = 0;
        }
    }

    fn tick(&mut self, sample: f64) -> f64 {
        self.write(sample);
        let out = self.read();
        self.output += 1;
        if self.output >= self.end {
            self.output = 0;
        }
        out
    }

    fn tap(&self, index: i32) -> f64 {
        let mut at = self.input as i64 - (i64::from(index) + 1);
        if at < 0 {
            at += i64::from(self.max) + 1;
        }
        usize::try_from(at)
            .ok()
            .and_then(|at| self.buffer.get(at))
            .copied()
            .unwrap_or(0.0)
    }
}

const I3DL2_LINES: usize = 4;
const I3DL2_TAPS: usize = 20;

/// Line lengths in milliseconds.
const I3DL2_MIN_TIMES: [f64; I3DL2_LINES] = [5.0, 6.0, 13.0, 14.0];
const I3DL2_MAX_TIMES: [f64; I3DL2_LINES] =
    [45.7042007446, 82.7817001343, 149.938293457, 271.575805664];
const I3DL2_DECAY0_TIMES: [f64; I3DL2_LINES] = [17.0, 13.0, 9.0, 7.0];
const I3DL2_DECAY1_TIMES: [f64; I3DL2_LINES] = [19.0, 11.0, 10.0, 6.0];

/// Early reflections: each tap's place in the reflection time and its gain.
const I3DL2_EARLY_TIMES: [f64; I3DL2_TAPS] = [
    0.0171360000968,
    0.0591540001333,
    0.161733001471,
    0.390186011791,
    0.425262004137,
    0.455410987139,
    0.689737021923,
    0.74590998888,
    0.833844006062,
    0.859502017498,
    0.0,
    0.0750240013003,
    0.168788000941,
    0.299901008606,
    0.337442994118,
    0.371903002262,
    0.599011003971,
    0.716741025448,
    0.817858994007,
    0.85166400671,
];
const I3DL2_EARLY_GAINS: [f64; I3DL2_TAPS] = [
    0.67096, 0.61027, 1.0, 0.3568, 0.68361, 0.65978, 0.51939, 0.24712, 0.45945, 0.45021, 0.64196,
    0.54879, 0.92925, 0.3827, 0.72867, 0.69794, 0.5464, 0.24563, 0.45214, 0.44042,
];

/// The I3DL2 reverb's parameters as the game wrote them (gains in
/// millibels, times in seconds).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct I3dl2Parameters {
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
    pub channels: usize,
    pub sample_rate: u32,
    pub room_hf_gain: f32,
    pub reference_hf: f32,
    pub late_reverb_decay_time: f32,
    pub late_reverb_hf_decay_ratio: f32,
    pub room_gain: f32,
    pub reflection_gain: f32,
    pub reverb_gain: f32,
    pub late_reverb_diffusion: f32,
    pub reflection_delay: f32,
    pub late_reverb_delay_time: f32,
    pub late_reverb_density: f32,
    pub dry_gain: f32,
}

impl I3dl2Parameters {
    /// From an effect's 0xa0 specific bytes.
    pub fn parse(bytes: &[u8]) -> Self {
        let word = |at: usize| crate::bytes::le_u32(&bytes[at..at + 4]);
        let float = |at: usize| f32::from_bits(word(at));
        let buffers = |at: usize| (0..6).map(|index| usize::from(bytes[at + index])).collect();
        Self {
            inputs: buffers(0),
            outputs: buffers(6),
            channels: usize::from(u16::from_le_bytes([bytes[0xe], bytes[0xf]])),
            sample_rate: word(0x14),
            room_hf_gain: float(0x18),
            reference_hf: float(0x1c),
            late_reverb_decay_time: float(0x20),
            late_reverb_hf_decay_ratio: float(0x24),
            room_gain: float(0x28),
            reflection_gain: float(0x2c),
            reverb_gain: float(0x30),
            late_reverb_diffusion: float(0x34),
            reflection_delay: float(0x38),
            late_reverb_delay_time: float(0x3c),
            late_reverb_density: float(0x40),
            dry_gain: float(0x44),
        }
    }
}

/// The I3DL2 reverb's state.
#[derive(Clone, Debug)]
pub struct I3dl2 {
    lowpass_0: f64,
    lowpass_1: f64,
    lowpass_2: f64,
    early_delay_line: I3dl2Line,
    early_tap_steps: [i32; I3DL2_TAPS],
    early_gain: f64,
    late_gain: f64,
    early_to_late_taps: i32,
    fdn_delay_lines: [I3dl2Line; I3DL2_LINES],
    decay_delay_lines0: [I3dl2Line; I3DL2_LINES],
    decay_delay_lines1: [I3dl2Line; I3DL2_LINES],
    center_delay_line: I3dl2Line,
    lowpass_coeff: [[f64; 3]; I3DL2_LINES],
    shelf_filter: [f64; I3DL2_LINES],
    dry_gain: f64,
}

impl I3dl2 {
    /// A silent reverb for `parameters`.
    pub fn new(parameters: &I3dl2Parameters) -> Self {
        let rate = f64::from(parameters.sample_rate) / 1000.0;
        let samples = |milliseconds: f64| (milliseconds * rate).max(0.0).floor() as u32;
        let lines = |times: [f64; I3DL2_LINES]| {
            std::array::from_fn(|line| I3dl2Line::new(samples(times[line])))
        };
        let mut reverb = Self {
            lowpass_0: 0.0,
            lowpass_1: 0.0,
            lowpass_2: 0.0,
            early_delay_line: I3dl2Line::new(samples(400.0)),
            early_tap_steps: [0; I3DL2_TAPS],
            early_gain: 0.0,
            late_gain: 0.0,
            early_to_late_taps: 0,
            fdn_delay_lines: lines(I3DL2_MAX_TIMES),
            decay_delay_lines0: lines(I3DL2_DECAY0_TIMES),
            decay_delay_lines1: lines(I3DL2_DECAY1_TIMES),
            center_delay_line: I3dl2Line::new(samples(5.0)),
            lowpass_coeff: [[0.0; 3]; I3DL2_LINES],
            shelf_filter: [0.0; I3DL2_LINES],
            dry_gain: 0.0,
        };
        reverb.update(parameters);
        reverb
    }

    /// Take changed parameters.
    pub fn update(&mut self, parameters: &I3dl2Parameters) {
        let value = |parameter: f32| f64::from(parameter);
        let sample_rate = f64::from(parameters.sample_rate);
        let rate = sample_rate / 1000.0;
        self.dry_gain = value(parameters.dry_gain);
        let room_gain = value(parameters.room_gain);
        self.early_gain =
            pow_10((room_gain + value(parameters.reflection_gain)).min(5000.0) / 2000.0);
        self.late_gain = pow_10((room_gain + value(parameters.reverb_gain)).min(5000.0) / 2000.0);
        let hf_gain = pow_10(value(parameters.room_hf_gain) / 2000.0);
        let reference_hf = value(parameters.reference_hf);
        if hf_gain >= 1.0 {
            self.lowpass_1 = 0.0;
            self.lowpass_2 = 1.0;
        } else {
            let a = 1.0 - hf_gain;
            let b = 2.0 - cos_degrees(reference_hf * 256.0 / sample_rate) * (hf_gain * 2.0);
            let c = (b * b - a * a * 4.0).sqrt();
            let low_pass = ((b - c) / (a * 2.0)).min(0.99723);
            self.lowpass_1 = if low_pass.is_finite() { low_pass } else { 0.0 };
            self.lowpass_2 = 1.0 - self.lowpass_1;
        }
        let reflection_delay = value(parameters.reflection_delay);
        let late_delay = value(parameters.late_reverb_delay_time);
        self.early_to_late_taps = ((reflection_delay + late_delay) * 1000.0 * rate) as i32;
        let echo = value(parameters.late_reverb_diffusion) * 0.6 * 0.01;
        for line in 0..I3DL2_LINES {
            let delay = (I3DL2_MIN_TIMES[line]
                + value(parameters.late_reverb_density) / 100.0
                    * (I3DL2_MAX_TIMES[line] - I3DL2_MIN_TIMES[line]))
                * rate;
            self.fdn_delay_lines[line].set_delay(delay as i32);
            let length = self.fdn_delay_lines[line].delay
                + self.decay_delay_lines0[line].delay
                + self.decay_delay_lines1[line].delay;
            // -60 dB after the decay time over the three lines' lengths.
            let a = f64::from(length) * -60.0
                / (value(parameters.late_reverb_decay_time) * sample_rate);
            let b = a / value(parameters.late_reverb_hf_decay_ratio);
            let angle = reference_hf * 0.5 * 128.0 / sample_rate;
            let c = cos_degrees(angle) / sin_degrees(angle);
            let d = pow_10((b - a) / 40.0);
            let e = pow_10((b + a) / 40.0) * 0.7071;
            let coefficients = [
                (c * d + 1.0) * e / (c + d),
                (1.0 - c * d) * e / (c + d),
                (c - d) / (c + d),
            ];
            self.lowpass_coeff[line] =
                coefficients.map(|value| if value.is_finite() { value } else { 0.0 });
            self.decay_delay_lines0[line].wet_gain = echo;
            self.decay_delay_lines1[line].wet_gain = echo * -0.9;
        }
        let reflection_time = (late_delay * 0.9998 + 0.02) * 1000.0;
        for tap in 0..I3DL2_TAPS {
            let length = ((reflection_delay * 1000.0 + reflection_time * I3DL2_EARLY_TIMES[tap])
                * rate) as i32;
            self.early_tap_steps[tap] = length.min(self.early_delay_line.max);
        }
    }

    /// Reverberate `count` samples of the input buffers into the outputs.
    pub fn process(
        &mut self,
        parameters: &I3dl2Parameters,
        buffers: &mut [Vec<i32>],
        count: usize,
    ) {
        let channels = parameters.channels;
        let (inputs, outputs) = (&parameters.inputs, &parameters.outputs);
        if channels == 0 {
            return;
        }
        if !channels_processed(channels, inputs, outputs, buffers.len()) {
            bypass(buffers, inputs, outputs);
            return;
        }
        let taps: [usize; I3DL2_TAPS] = match channels {
            1 => [0; I3DL2_TAPS],
            2 => [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
            4 => [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 3, 3, 3],
            _ => [2, 0, 0, 1, 1, 1, 1, 4, 4, 4, 1, 1, 1, 0, 0, 0, 0, 5, 5, 5],
        };
        for sample in 0..count {
            let late = self.early_delay_line.tap(self.early_to_late_taps) * self.late_gain;
            let mut early = [0.0f64; 6];
            for tap in 0..I3DL2_TAPS {
                let value =
                    self.early_delay_line.tap(self.early_tap_steps[tap]) * I3DL2_EARLY_GAINS[tap];
                early[taps[tap]] += value;
                if channels == 6 {
                    early[LFE] += value;
                }
            }
            let dry: [i32; 6] = std::array::from_fn(|channel| {
                if channel < channels {
                    buffers[inputs[channel]][sample]
                } else {
                    0
                }
            });
            let input: f64 = dry.iter().map(|&value| f64::from(value)).sum();
            self.lowpass_0 = input * self.lowpass_2 + self.lowpass_0 * self.lowpass_1;
            self.early_delay_line.tick(self.lowpass_0);
            for value in &mut early {
                *value *= self.early_gain;
            }
            let mut filtered = [0.0f64; I3DL2_LINES];
            for line in 0..I3DL2_LINES {
                let read = self.fdn_delay_lines[line].read();
                filtered[line] = read * self.lowpass_coeff[line][0] + self.shelf_filter[line];
                self.shelf_filter[line] = filtered[line] * self.lowpass_coeff[line][2]
                    + read * self.lowpass_coeff[line][1];
            }
            let mixed = [
                filtered[1] + filtered[2] + late,
                -filtered[0] - filtered[3] + late,
                filtered[0] - filtered[3] + late,
                filtered[1] - filtered[2] + late,
            ];
            let mut all_pass = [0.0f64; I3DL2_LINES];
            for line in 0..I3DL2_LINES {
                let decay = &mut self.decay_delay_lines0[line];
                let value = mixed[line] - decay.read() * decay.wet_gain;
                let out = decay.tick(value) + value * decay.wet_gain;
                let decay = &mut self.decay_delay_lines1[line];
                let value = out - decay.read() * decay.wet_gain;
                let out = decay.tick(value) + value * decay.wet_gain;
                self.fdn_delay_lines[line].tick(out);
                all_pass[line] = out;
            }
            let late_out: [f64; 6] = if channels == 6 {
                [
                    all_pass[0],
                    all_pass[1],
                    self.center_delay_line
                        .tick((all_pass[2] - all_pass[3]) * 0.5),
                    all_pass[3],
                    all_pass[2],
                    all_pass[3],
                ]
            } else {
                [all_pass[0], all_pass[1], all_pass[2], all_pass[3], 0.0, 0.0]
            };
            for channel in 0..channels {
                let out =
                    early[channel] + late_out[channel] + self.dry_gain * f64::from(dry[channel]);
                buffers[outputs[channel]][sample] = out.clamp(-8388600.0, 8388600.0) as i32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reverb_parameters(channels: usize) -> ReverbParameters {
        let fixed = |value: f64| (value * 16384.0) as i32;
        ReverbParameters {
            inputs: (0..6).collect(),
            outputs: (0..6).collect(),
            channels,
            sample_rate: fixed(48.0),
            early_mode: 1,
            early_gain: fixed(0.7),
            pre_delay: fixed(10.0),
            late_mode: 1,
            late_gain: fixed(0.7),
            decay_time: fixed(1.5),
            high_frequency_decay_ratio: fixed(0.6),
            colouration: fixed(0.4),
            base_gain: fixed(0.3),
            wet_gain: fixed(0.8),
            dry_gain: fixed(1.0),
        }
    }

    fn i3dl2_parameters(channels: usize) -> I3dl2Parameters {
        I3dl2Parameters {
            inputs: (0..6).collect(),
            outputs: (0..6).collect(),
            channels,
            sample_rate: 48_000,
            room_hf_gain: -1000.0,
            reference_hf: 5000.0,
            late_reverb_decay_time: 1.5,
            late_reverb_hf_decay_ratio: 0.6,
            room_gain: -1000.0,
            reflection_gain: -500.0,
            reverb_gain: 0.0,
            late_reverb_diffusion: 100.0,
            reflection_delay: 0.01,
            late_reverb_delay_time: 0.02,
            late_reverb_density: 100.0,
            dry_gain: 0.5,
        }
    }

    /// The energy of each of `frames` 240-sample frames after an impulse.
    fn tail(mut process: impl FnMut(&mut [Vec<i32>]), channels: usize, frames: usize) -> Vec<f64> {
        (0..frames)
            .map(|frame| {
                let mut buffers = vec![vec![0i32; 240]; 6];
                if frame == 0 {
                    for buffer in &mut buffers[..channels] {
                        buffer[0] = 20_000;
                    }
                }
                process(&mut buffers);
                buffers
                    .iter()
                    .flatten()
                    .map(|&value| f64::from(value) * f64::from(value))
                    .sum()
            })
            .collect()
    }

    #[test]
    fn reverb_rings_and_decays() {
        for channels in [1, 2, 4, 6] {
            let parameters = reverb_parameters(channels);
            let mut reverb = Reverb::new(&parameters, true);
            let energy = tail(
                |buffers| reverb.process(&parameters, buffers, 240),
                channels,
                2400,
            );
            // Early reflections and the late tail follow the impulse.
            assert!(
                energy[4..40].iter().sum::<f64>() > 1000.0,
                "{channels} channels"
            );
            // 12 s later (eight decay times) it is silent.
            assert!(
                energy[2300..].iter().sum::<f64>() < 1.0,
                "{channels} channels"
            );
            let early: f64 = energy[40..240].iter().sum();
            let late: f64 = energy[1000..1200].iter().sum();
            assert!(late < early, "{channels} channels");
        }
    }

    #[test]
    fn reverb_keeps_silence() {
        let parameters = reverb_parameters(2);
        let mut reverb = Reverb::new(&parameters, false);
        let mut buffers = vec![vec![0i32; 240]; 6];
        reverb.process(&parameters, &mut buffers, 240);
        assert!(buffers.iter().flatten().all(|&value| value == 0));
    }

    #[test]
    fn i3dl2_rings_and_decays() {
        for channels in [1, 2, 4, 6] {
            let parameters = i3dl2_parameters(channels);
            let mut reverb = I3dl2::new(&parameters);
            let energy = tail(
                |buffers| reverb.process(&parameters, buffers, 240),
                channels,
                2400,
            );
            assert!(
                energy[2..60].iter().sum::<f64>() > 1000.0,
                "{channels} channels"
            );
            assert!(
                energy[2300..].iter().sum::<f64>() < 1.0,
                "{channels} channels"
            );
            let early: f64 = energy[40..240].iter().sum();
            let late: f64 = energy[1000..1200].iter().sum();
            assert!(late < early, "{channels} channels");
        }
    }

    #[test]
    fn other_channel_counts_pass_through() {
        let mut parameters = reverb_parameters(3);
        parameters.outputs = vec![3, 4, 5, 0, 0, 0];
        let mut reverb = Reverb::new(&parameters, true);
        let mut buffers = vec![vec![0i32; 4]; 6];
        buffers[0] = vec![1, 2, 3, 4];
        reverb.process(&parameters, &mut buffers, 4);
        assert_eq!(buffers[3], vec![1, 2, 3, 4]);
    }
}
