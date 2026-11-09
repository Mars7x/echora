//! sequence_header_obu() of AV1 spec section 5.5.

use crate::Error;
use crate::bits::BitReader;

/// seq_force_screen_content_tools / seq_force_integer_mv value meaning
/// "chosen per frame".
pub const SELECT_SCREEN_CONTENT_TOOLS: u8 = 2;
pub const SELECT_INTEGER_MV: u8 = 2;

/// timing_info().
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TimingInfo {
    pub num_units_in_display_tick: u32,
    pub time_scale: u32,
    pub equal_picture_interval: bool,
    pub num_ticks_per_picture_minus_1: u32,
}

/// decoder_model_info().
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DecoderModelInfo {
    pub buffer_delay_length_minus_1: u8,
    pub num_units_in_decoding_tick: u32,
    pub buffer_removal_time_length_minus_1: u8,
    pub frame_presentation_time_length_minus_1: u8,
}

/// One operating point of the sequence header.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OperatingPoint {
    pub idc: u16,
    pub seq_level_idx: u8,
    pub seq_tier: u8,
    pub decoder_model_present: bool,
    /// operating_parameters_info(), when decoder_model_present.
    pub decoder_buffer_delay: u32,
    pub encoder_buffer_delay: u32,
    pub low_delay_mode_flag: bool,
    pub initial_display_delay_present: bool,
    pub initial_display_delay_minus_1: u8,
}

/// color_config().
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColorConfig {
    pub bit_depth: u8,
    pub mono_chrome: bool,
    pub color_description_present_flag: bool,
    pub color_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
    pub color_range: bool,
    pub subsampling_x: bool,
    pub subsampling_y: bool,
    pub chroma_sample_position: u8,
    pub separate_uv_delta_q: bool,
}

impl ColorConfig {
    /// NumPlanes.
    pub fn num_planes(&self) -> usize {
        if self.mono_chrome { 1 } else { 3 }
    }
}

/// A parsed sequence header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SequenceHeader {
    pub seq_profile: u8,
    pub still_picture: bool,
    pub reduced_still_picture_header: bool,
    pub timing_info: Option<TimingInfo>,
    pub decoder_model_info: Option<DecoderModelInfo>,
    pub initial_display_delay_present_flag: bool,
    pub operating_points: Vec<OperatingPoint>,
    pub frame_width_bits_minus_1: u8,
    pub frame_height_bits_minus_1: u8,
    pub max_frame_width_minus_1: u16,
    pub max_frame_height_minus_1: u16,
    pub frame_id_numbers_present_flag: bool,
    pub delta_frame_id_length_minus_2: u8,
    pub additional_frame_id_length_minus_1: u8,
    pub use_128x128_superblock: bool,
    pub enable_filter_intra: bool,
    pub enable_intra_edge_filter: bool,
    pub enable_interintra_compound: bool,
    pub enable_masked_compound: bool,
    pub enable_warped_motion: bool,
    pub enable_dual_filter: bool,
    pub enable_order_hint: bool,
    pub enable_jnt_comp: bool,
    pub enable_ref_frame_mvs: bool,
    /// 0, 1 or SELECT_SCREEN_CONTENT_TOOLS.
    pub seq_force_screen_content_tools: u8,
    /// 0, 1 or SELECT_INTEGER_MV.
    pub seq_force_integer_mv: u8,
    /// order_hint_bits_minus_1 + 1, or 0 without order hints.
    pub order_hint_bits: u8,
    pub enable_superres: bool,
    pub enable_cdef: bool,
    pub enable_restoration: bool,
    pub color_config: ColorConfig,
    pub film_grain_params_present: bool,
}

impl SequenceHeader {
    /// Parses the payload of a sequence header OBU.
    pub fn parse(payload: &[u8]) -> Result<Self, Error> {
        let mut r = BitReader::new(payload);
        Self::read(&mut r)
    }

    pub(crate) fn read(r: &mut BitReader<'_>) -> Result<Self, Error> {
        let seq_profile = r.f(3)? as u8;
        if seq_profile > 2 {
            return Err(Error::Unsupported("seq_profile above 2"));
        }
        let still_picture = r.flag()?;
        let reduced_still_picture_header = r.flag()?;
        let mut timing_info = None;
        let mut decoder_model_info = None;
        let mut initial_display_delay_present_flag = false;
        let mut operating_points = Vec::new();
        if reduced_still_picture_header {
            operating_points.push(OperatingPoint {
                seq_level_idx: r.f(5)? as u8,
                ..Default::default()
            });
        } else {
            if r.flag()? {
                let mut t = TimingInfo {
                    num_units_in_display_tick: r.f(32)?,
                    time_scale: r.f(32)?,
                    equal_picture_interval: r.flag()?,
                    num_ticks_per_picture_minus_1: 0,
                };
                if t.equal_picture_interval {
                    t.num_ticks_per_picture_minus_1 = r.uvlc()?;
                }
                timing_info = Some(t);
                if r.flag()? {
                    decoder_model_info = Some(DecoderModelInfo {
                        buffer_delay_length_minus_1: r.f(5)? as u8,
                        num_units_in_decoding_tick: r.f(32)?,
                        buffer_removal_time_length_minus_1: r.f(5)? as u8,
                        frame_presentation_time_length_minus_1: r.f(5)? as u8,
                    });
                }
            }
            initial_display_delay_present_flag = r.flag()?;
            let count = r.f(5)? + 1;
            for _ in 0..count {
                let mut op = OperatingPoint {
                    idc: r.f(12)? as u16,
                    seq_level_idx: r.f(5)? as u8,
                    ..Default::default()
                };
                if op.seq_level_idx > 7 {
                    op.seq_tier = r.f(1)? as u8;
                }
                if let Some(model) = &decoder_model_info {
                    op.decoder_model_present = r.flag()?;
                    if op.decoder_model_present {
                        let n = u32::from(model.buffer_delay_length_minus_1) + 1;
                        op.decoder_buffer_delay = r.f(n)?;
                        op.encoder_buffer_delay = r.f(n)?;
                        op.low_delay_mode_flag = r.flag()?;
                    }
                }
                if initial_display_delay_present_flag {
                    op.initial_display_delay_present = r.flag()?;
                    if op.initial_display_delay_present {
                        op.initial_display_delay_minus_1 = r.f(4)? as u8;
                    }
                }
                operating_points.push(op);
            }
        }
        let frame_width_bits_minus_1 = r.f(4)? as u8;
        let frame_height_bits_minus_1 = r.f(4)? as u8;
        let max_frame_width_minus_1 = r.f(u32::from(frame_width_bits_minus_1) + 1)? as u16;
        let max_frame_height_minus_1 = r.f(u32::from(frame_height_bits_minus_1) + 1)? as u16;
        let frame_id_numbers_present_flag = !reduced_still_picture_header && r.flag()?;
        let (mut delta_frame_id_length_minus_2, mut additional_frame_id_length_minus_1) = (0, 0);
        if frame_id_numbers_present_flag {
            delta_frame_id_length_minus_2 = r.f(4)? as u8;
            additional_frame_id_length_minus_1 = r.f(3)? as u8;
        }
        let use_128x128_superblock = r.flag()?;
        let enable_filter_intra = r.flag()?;
        let enable_intra_edge_filter = r.flag()?;
        let mut enable_interintra_compound = false;
        let mut enable_masked_compound = false;
        let mut enable_warped_motion = false;
        let mut enable_dual_filter = false;
        let mut enable_order_hint = false;
        let mut enable_jnt_comp = false;
        let mut enable_ref_frame_mvs = false;
        let mut seq_force_screen_content_tools = SELECT_SCREEN_CONTENT_TOOLS;
        let mut seq_force_integer_mv = SELECT_INTEGER_MV;
        let mut order_hint_bits = 0;
        if !reduced_still_picture_header {
            enable_interintra_compound = r.flag()?;
            enable_masked_compound = r.flag()?;
            enable_warped_motion = r.flag()?;
            enable_dual_filter = r.flag()?;
            enable_order_hint = r.flag()?;
            if enable_order_hint {
                enable_jnt_comp = r.flag()?;
                enable_ref_frame_mvs = r.flag()?;
            }
            let seq_choose_screen_content_tools = r.flag()?;
            if !seq_choose_screen_content_tools {
                seq_force_screen_content_tools = r.f(1)? as u8;
            }
            if seq_force_screen_content_tools > 0 {
                let seq_choose_integer_mv = r.flag()?;
                if !seq_choose_integer_mv {
                    seq_force_integer_mv = r.f(1)? as u8;
                }
            }
            if enable_order_hint {
                order_hint_bits = r.f(3)? as u8 + 1;
            }
        }
        let enable_superres = r.flag()?;
        let enable_cdef = r.flag()?;
        let enable_restoration = r.flag()?;
        let color_config = read_color_config(r, seq_profile)?;
        let film_grain_params_present = r.flag()?;
        Ok(Self {
            seq_profile,
            still_picture,
            reduced_still_picture_header,
            timing_info,
            decoder_model_info,
            initial_display_delay_present_flag,
            operating_points,
            frame_width_bits_minus_1,
            frame_height_bits_minus_1,
            max_frame_width_minus_1,
            max_frame_height_minus_1,
            frame_id_numbers_present_flag,
            delta_frame_id_length_minus_2,
            additional_frame_id_length_minus_1,
            use_128x128_superblock,
            enable_filter_intra,
            enable_intra_edge_filter,
            enable_interintra_compound,
            enable_masked_compound,
            enable_warped_motion,
            enable_dual_filter,
            enable_order_hint,
            enable_jnt_comp,
            enable_ref_frame_mvs,
            seq_force_screen_content_tools,
            seq_force_integer_mv,
            order_hint_bits,
            enable_superres,
            enable_cdef,
            enable_restoration,
            color_config,
            film_grain_params_present,
        })
    }

    /// idLen: the width of frame ids when frame_id_numbers_present_flag.
    pub fn frame_id_length(&self) -> u32 {
        u32::from(self.additional_frame_id_length_minus_1)
            + u32::from(self.delta_frame_id_length_minus_2)
            + 3
    }

    /// get_relative_dist(a, b) of section 7.12.1.
    pub fn relative_dist(&self, a: u32, b: u32) -> i32 {
        if !self.enable_order_hint {
            return 0;
        }
        let bits = u32::from(self.order_hint_bits);
        let diff = a as i32 - b as i32;
        let m = 1i32 << (bits - 1);
        (diff & (m - 1)) - (diff & m)
    }

    /// The sequence header semantics allow a repeated header only when it
    /// is identical; a different one starts a new coded video sequence.
    pub fn same_sequence(&self, other: &Self) -> bool {
        self == other
    }
}

fn read_color_config(r: &mut BitReader<'_>, seq_profile: u8) -> Result<ColorConfig, Error> {
    let high_bitdepth = r.flag()?;
    let bit_depth = if seq_profile == 2 && high_bitdepth {
        if r.flag()? { 12 } else { 10 }
    } else if high_bitdepth {
        10
    } else {
        8
    };
    let mono_chrome = seq_profile != 1 && r.flag()?;
    let color_description_present_flag = r.flag()?;
    // CP_UNSPECIFIED, TC_UNSPECIFIED, MC_UNSPECIFIED
    let (mut color_primaries, mut transfer_characteristics, mut matrix_coefficients) = (2, 2, 2);
    if color_description_present_flag {
        color_primaries = r.f(8)? as u8;
        transfer_characteristics = r.f(8)? as u8;
        matrix_coefficients = r.f(8)? as u8;
    }
    let mut c = ColorConfig {
        bit_depth,
        mono_chrome,
        color_description_present_flag,
        color_primaries,
        transfer_characteristics,
        matrix_coefficients,
        color_range: false,
        subsampling_x: true,
        subsampling_y: true,
        chroma_sample_position: 0, // CSP_UNKNOWN
        separate_uv_delta_q: false,
    };
    if mono_chrome {
        c.color_range = r.flag()?;
        return Ok(c);
    }
    // CP_BT_709, TC_SRGB, MC_IDENTITY
    if color_primaries == 1 && transfer_characteristics == 13 && matrix_coefficients == 0 {
        c.color_range = true;
        c.subsampling_x = false;
        c.subsampling_y = false;
    } else {
        c.color_range = r.flag()?;
        match seq_profile {
            0 => {}
            1 => {
                c.subsampling_x = false;
                c.subsampling_y = false;
            }
            _ => {
                if bit_depth == 12 {
                    c.subsampling_x = r.flag()?;
                    c.subsampling_y = c.subsampling_x && r.flag()?;
                } else {
                    c.subsampling_y = false;
                }
            }
        }
        if c.subsampling_x && c.subsampling_y {
            c.chroma_sample_position = r.f(2)? as u8;
        }
    }
    c.separate_uv_delta_q = r.flag()?;
    Ok(c)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::bits::BitWriter;

    /// Writes a plain profile-0 sequence header with order hints; the
    /// frame header tests build on it.
    pub(crate) fn write_basic(w: &mut BitWriter, width: u32, height: u32) {
        w.put(3, 0) // seq_profile
            .flag(false) // still_picture
            .flag(false) // reduced_still_picture_header
            .flag(false) // timing_info_present_flag
            .flag(false) // initial_display_delay_present_flag
            .put(5, 0) // operating_points_cnt_minus_1
            .put(12, 0) // operating_point_idc
            .put(5, 8) // seq_level_idx (4.0)
            .put(1, 0) // seq_tier
            .put(4, 15) // frame_width_bits_minus_1
            .put(4, 15)
            .put(16, width - 1)
            .put(16, height - 1)
            .flag(false) // frame_id_numbers_present_flag
            .flag(false) // use_128x128_superblock
            .flag(true) // enable_filter_intra
            .flag(true) // enable_intra_edge_filter
            .flag(true) // enable_interintra_compound
            .flag(true) // enable_masked_compound
            .flag(true) // enable_warped_motion
            .flag(true) // enable_dual_filter
            .flag(true) // enable_order_hint
            .flag(true) // enable_jnt_comp
            .flag(true) // enable_ref_frame_mvs
            .flag(true) // seq_choose_screen_content_tools
            .flag(true) // seq_choose_integer_mv
            .put(3, 6) // order_hint_bits_minus_1
            .flag(false) // enable_superres
            .flag(true) // enable_cdef
            .flag(true) // enable_restoration
            // color_config
            .flag(false) // high_bitdepth
            .flag(false) // mono_chrome
            .flag(true) // color_description_present_flag
            .put(8, 1)
            .put(8, 1)
            .put(8, 1)
            .flag(false) // color_range
            .put(2, 0) // chroma_sample_position
            .flag(false) // separate_uv_delta_q
            .flag(false); // film_grain_params_present
    }

    #[test]
    fn basic_header() {
        let mut w = BitWriter::default();
        write_basic(&mut w, 1920, 1080);
        let s = SequenceHeader::parse(&w.bytes).unwrap();
        assert_eq!(s.seq_profile, 0);
        assert_eq!(s.max_frame_width_minus_1, 1919);
        assert_eq!(s.max_frame_height_minus_1, 1079);
        assert_eq!(s.operating_points.len(), 1);
        assert_eq!(s.operating_points[0].seq_level_idx, 8);
        assert_eq!(s.operating_points[0].seq_tier, 0);
        assert!(s.enable_order_hint && s.enable_jnt_comp && s.enable_ref_frame_mvs);
        assert_eq!(s.order_hint_bits, 7);
        assert_eq!(
            s.seq_force_screen_content_tools,
            SELECT_SCREEN_CONTENT_TOOLS
        );
        assert_eq!(s.seq_force_integer_mv, SELECT_INTEGER_MV);
        assert_eq!(s.color_config.bit_depth, 8);
        assert_eq!(s.color_config.matrix_coefficients, 1);
        assert!(s.color_config.subsampling_x && s.color_config.subsampling_y);
        assert_eq!(s.color_config.num_planes(), 3);
        assert!(!s.film_grain_params_present);
    }

    #[test]
    fn reduced_still_picture() {
        let mut w = BitWriter::default();
        w.put(3, 0)
            .flag(true) // still_picture
            .flag(true) // reduced_still_picture_header
            .put(5, 1) // seq_level_idx
            .put(4, 7)
            .put(4, 6)
            .put(8, 99)
            .put(7, 49)
            .flag(true) // use_128x128_superblock
            .flag(false)
            .flag(false)
            .flag(false) // enable_superres
            .flag(false)
            .flag(false)
            .flag(false) // high_bitdepth
            .flag(true) // mono_chrome
            .flag(false) // color_description_present_flag
            .flag(true) // color_range
            .flag(false); // film_grain_params_present
        let s = SequenceHeader::parse(&w.bytes).unwrap();
        assert!(s.reduced_still_picture_header);
        assert_eq!(s.operating_points[0].seq_level_idx, 1);
        assert_eq!(s.max_frame_width_minus_1, 99);
        assert_eq!(s.max_frame_height_minus_1, 49);
        assert!(s.use_128x128_superblock);
        assert!(!s.enable_order_hint);
        assert_eq!(s.order_hint_bits, 0);
        assert!(s.color_config.mono_chrome && s.color_config.color_range);
        assert_eq!(s.color_config.num_planes(), 1);
        assert_eq!(s.color_config.color_primaries, 2);
    }

    #[test]
    fn timing_decoder_model_and_operating_points() {
        let mut w = BitWriter::default();
        w.put(3, 2) // seq_profile 2
            .flag(false)
            .flag(false)
            .flag(true) // timing_info_present_flag
            .put(32, 1001)
            .put(32, 60000)
            .flag(true) // equal_picture_interval
            .put(3, 0b010) // uvlc 1
            .flag(true) // decoder_model_info_present_flag
            .put(5, 9) // buffer_delay_length_minus_1
            .put(32, 1)
            .put(5, 4)
            .put(5, 6)
            .flag(true) // initial_display_delay_present_flag
            .put(5, 1) // two operating points
            .put(12, 0x103)
            .put(5, 9)
            .put(1, 1) // seq_tier
            .flag(true) // decoder_model_present_for_this_op
            .put(10, 500)
            .put(10, 300)
            .flag(true)
            .flag(true)
            .put(4, 3)
            .put(12, 0x101)
            .put(5, 4)
            .flag(false)
            .flag(false)
            .put(4, 10)
            .put(4, 10)
            .put(11, 1279)
            .put(11, 719)
            .flag(true) // frame_id_numbers_present_flag
            .put(4, 5)
            .put(3, 2)
            .flag(false)
            .flag(false)
            .flag(false)
            .put(5, 0) // the compound flags and enable_order_hint
            .flag(false) // seq_choose_screen_content_tools
            .put(1, 0) // seq_force_screen_content_tools
            .flag(true) // enable_superres
            .flag(false)
            .flag(false)
            .flag(true) // high_bitdepth
            .flag(true) // twelve_bit
            .flag(false) // mono_chrome
            .flag(false) // color_description_present_flag
            .flag(false) // color_range
            .flag(true) // subsampling_x
            .flag(false) // subsampling_y
            .flag(true) // separate_uv_delta_q
            .flag(true); // film_grain_params_present
        let s = SequenceHeader::parse(&w.bytes).unwrap();
        let t = s.timing_info.as_ref().unwrap();
        assert_eq!((t.num_units_in_display_tick, t.time_scale), (1001, 60000));
        assert_eq!(t.num_ticks_per_picture_minus_1, 1);
        let m = s.decoder_model_info.as_ref().unwrap();
        assert_eq!(m.buffer_removal_time_length_minus_1, 4);
        assert_eq!(m.frame_presentation_time_length_minus_1, 6);
        assert_eq!(s.operating_points.len(), 2);
        let op = &s.operating_points[0];
        assert_eq!((op.idc, op.seq_level_idx, op.seq_tier), (0x103, 9, 1));
        assert_eq!(
            (op.decoder_buffer_delay, op.encoder_buffer_delay),
            (500, 300)
        );
        assert!(op.low_delay_mode_flag);
        assert_eq!(op.initial_display_delay_minus_1, 3);
        assert!(!s.operating_points[1].decoder_model_present);
        assert_eq!(s.max_frame_width_minus_1, 1279);
        assert!(s.frame_id_numbers_present_flag);
        assert_eq!(s.frame_id_length(), 2 + 5 + 3);
        assert_eq!(s.seq_force_screen_content_tools, 0);
        assert_eq!(s.seq_force_integer_mv, SELECT_INTEGER_MV);
        assert!(s.enable_superres);
        assert_eq!(s.color_config.bit_depth, 12);
        assert!(s.color_config.subsampling_x && !s.color_config.subsampling_y);
        assert!(s.color_config.separate_uv_delta_q);
        assert!(s.film_grain_params_present);
    }

    #[test]
    fn srgb_is_444() {
        let mut w = BitWriter::default();
        w.put(3, 1)
            .flag(true)
            .flag(true)
            .put(5, 0)
            .put(4, 3)
            .put(4, 3)
            .put(4, 15)
            .put(4, 15)
            .put(3, 0)
            .put(3, 0)
            .flag(false) // high_bitdepth
            .flag(true) // color_description_present_flag
            .put(8, 1)
            .put(8, 13)
            .put(8, 0)
            .flag(false) // separate_uv_delta_q
            .flag(false);
        let s = SequenceHeader::parse(&w.bytes).unwrap();
        assert!(s.color_config.color_range);
        assert!(!s.color_config.subsampling_x && !s.color_config.subsampling_y);
    }

    #[test]
    fn relative_distance() {
        let mut w = BitWriter::default();
        write_basic(&mut w, 64, 64);
        let s = SequenceHeader::parse(&w.bytes).unwrap();
        // Seven bits of order hint wrap at 128.
        assert_eq!(s.relative_dist(5, 3), 2);
        assert_eq!(s.relative_dist(3, 5), -2);
        assert_eq!(s.relative_dist(1, 127), 2);
        assert_eq!(s.relative_dist(127, 1), -2);
        assert_eq!(s.relative_dist(64, 0), -64);
    }
}
