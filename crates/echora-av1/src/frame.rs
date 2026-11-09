//! The uncompressed frame header of AV1 spec section 5.9, with the
//! semantics of section 6.8 and the processes of section 7 that header
//! parsing depends on (set_frame_refs, superres, segmentation lookups).

use crate::Error;
use crate::bits::BitReader;
use crate::refs::{DEFAULT_LOOP_FILTER_REF_DELTAS, RefState};
use crate::sequence::{SELECT_INTEGER_MV, SELECT_SCREEN_CONTENT_TOOLS, SequenceHeader};

pub const NUM_REF_FRAMES: usize = 8;
pub const REFS_PER_FRAME: usize = 7;
pub const PRIMARY_REF_NONE: u8 = 7;
pub const MAX_SEGMENTS: usize = 8;
pub const SEG_LVL_MAX: usize = 8;
pub const SEG_LVL_ALT_Q: usize = 0;
pub const SEG_LVL_REF_FRAME: usize = 5;
pub const MAX_TILE_COLS: u32 = 64;
pub const MAX_TILE_ROWS: u32 = 64;
const MAX_TILE_WIDTH: u32 = 4096;
const MAX_TILE_AREA: u32 = 4096 * 2304;
const SUPERRES_NUM: u32 = 8;
const SUPERRES_DENOM_MIN: u32 = 9;
const SUPERRES_DENOM_BITS: u32 = 3;
const WARPEDMODEL_PREC_BITS: u32 = 16;
const GM_ABS_TRANS_BITS: u32 = 12;
const GM_ABS_TRANS_ONLY_BITS: u32 = 9;
const GM_ABS_ALPHA_BITS: u32 = 12;
const GM_ALPHA_PREC_BITS: u32 = 15;
const GM_TRANS_PREC_BITS: u32 = 6;
const GM_TRANS_ONLY_PREC_BITS: u32 = 3;
const RESTORATION_TILESIZE_MAX: u16 = 256;

/// Reference names (section 6.10.24's ref_frame values).
pub const INTRA_FRAME: usize = 0;
pub const LAST_FRAME: usize = 1;
pub const LAST2_FRAME: usize = 2;
pub const LAST3_FRAME: usize = 3;
pub const GOLDEN_FRAME: usize = 4;
pub const BWDREF_FRAME: usize = 5;
pub const ALTREF2_FRAME: usize = 6;
pub const ALTREF_FRAME: usize = 7;

const SEGMENTATION_FEATURE_BITS: [u32; SEG_LVL_MAX] = [8, 6, 6, 6, 6, 3, 0, 0];
const SEGMENTATION_FEATURE_SIGNED: [bool; SEG_LVL_MAX] =
    [true, true, true, true, true, false, false, false];
const SEGMENTATION_FEATURE_MAX: [i32; SEG_LVL_MAX] = [255, 63, 63, 63, 63, 7, 0, 0];

/// frame_type, with StdVideoAV1FrameType's values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FrameType {
    Key = 0,
    Inter = 1,
    IntraOnly = 2,
    Switch = 3,
}

impl FrameType {
    fn from_raw(v: u32) -> Self {
        match v & 3 {
            0 => Self::Key,
            1 => Self::Inter,
            2 => Self::IntraOnly,
            _ => Self::Switch,
        }
    }

    /// FrameIsIntra.
    pub fn is_intra(self) -> bool {
        matches!(self, Self::Key | Self::IntraOnly)
    }
}

/// interpolation_filter, with StdVideoAV1InterpolationFilter's values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum InterpolationFilter {
    #[default]
    EightTap = 0,
    EightTapSmooth = 1,
    EightTapSharp = 2,
    Bilinear = 3,
    Switchable = 4,
}

/// TxMode, with StdVideoAV1TxMode's values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum TxMode {
    #[default]
    Only4x4 = 0,
    Largest = 1,
    Select = 2,
}

/// FrameRestorationType, with StdVideoAV1FrameRestorationType's values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum RestorationType {
    #[default]
    None = 0,
    Wiener = 1,
    Sgrproj = 2,
    Switchable = 3,
}

/// GmType, with the spec's (and Vulkan's) values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum WarpModel {
    #[default]
    Identity = 0,
    Translation = 1,
    RotZoom = 2,
    Affine = 3,
}

/// frame_size(), superres_params(), render_size() and compute_image_size().
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FrameSize {
    /// FrameWidth: the coded width, after the superres downscale.
    pub frame_width: u32,
    pub frame_height: u32,
    /// UpscaledWidth.
    pub upscaled_width: u32,
    pub render_width: u32,
    pub render_height: u32,
    pub render_and_frame_size_different: bool,
    /// The size was copied from a reference (found_ref in
    /// frame_size_with_refs).
    pub found_ref: bool,
    pub use_superres: bool,
    /// coded_denom; SuperresDenom = coded_denom + 9 when use_superres.
    pub coded_denom: u8,
    pub superres_denom: u32,
    pub mi_cols: u32,
    pub mi_rows: u32,
}

/// tile_info().
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TileInfo {
    pub uniform_tile_spacing_flag: bool,
    pub tile_cols: u32,
    pub tile_rows: u32,
    pub tile_cols_log2: u32,
    pub tile_rows_log2: u32,
    /// MiColStarts, tile_cols + 1 entries (the last is MiCols).
    pub mi_col_starts: Vec<u32>,
    /// MiRowStarts, tile_rows + 1 entries (the last is MiRows).
    pub mi_row_starts: Vec<u32>,
    /// Tile widths and heights in superblocks, minus one.
    pub width_in_sbs_minus_1: Vec<u32>,
    pub height_in_sbs_minus_1: Vec<u32>,
    pub context_update_tile_id: u32,
    /// TileSizeBytes - 1; zero when there is a single tile.
    pub tile_size_bytes_minus_1: u8,
}

impl TileInfo {
    pub fn num_tiles(&self) -> u32 {
        self.tile_cols * self.tile_rows
    }
}

/// quantization_params().
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Quantization {
    pub base_q_idx: u8,
    pub delta_q_y_dc: i8,
    pub diff_uv_delta: bool,
    pub delta_q_u_dc: i8,
    pub delta_q_u_ac: i8,
    pub delta_q_v_dc: i8,
    pub delta_q_v_ac: i8,
    pub using_qmatrix: bool,
    pub qm_y: u8,
    pub qm_u: u8,
    pub qm_v: u8,
}

/// segmentation_params(), with the feature state loaded from the primary
/// reference when not updated.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Segmentation {
    pub enabled: bool,
    pub update_map: bool,
    pub temporal_update: bool,
    pub update_data: bool,
    pub feature_enabled: [[bool; SEG_LVL_MAX]; MAX_SEGMENTS],
    pub feature_data: [[i16; SEG_LVL_MAX]; MAX_SEGMENTS],
    pub seg_id_pre_skip: bool,
    pub last_active_seg_id: u8,
}

impl Segmentation {
    /// seg_feature_active_idx().
    pub fn feature_active(&self, segment: usize, feature: usize) -> bool {
        self.enabled && self.feature_enabled[segment][feature]
    }
}

/// delta_q_params() and delta_lf_params().
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DeltaParams {
    pub delta_q_present: bool,
    /// delta_q_res (log2 of the resolution).
    pub delta_q_res: u8,
    pub delta_lf_present: bool,
    pub delta_lf_res: u8,
    pub delta_lf_multi: bool,
}

/// loop_filter_params().
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoopFilter {
    /// loop_filter_level[0..4]: Y vertical, Y horizontal, U, V.
    pub level: [u8; 4],
    pub sharpness: u8,
    pub delta_enabled: bool,
    pub delta_update: bool,
    /// Bit i: update_ref_delta for reference i was set.
    pub update_ref_delta: u8,
    pub ref_deltas: [i8; NUM_REF_FRAMES],
    /// Bit i: update_mode_delta for mode i was set.
    pub update_mode_delta: u8,
    pub mode_deltas: [i8; 2],
}

impl Default for LoopFilter {
    fn default() -> Self {
        Self {
            level: [0; 4],
            sharpness: 0,
            delta_enabled: false,
            delta_update: false,
            update_ref_delta: 0,
            ref_deltas: DEFAULT_LOOP_FILTER_REF_DELTAS,
            update_mode_delta: 0,
            mode_deltas: [0; 2],
        }
    }
}

/// cdef_params(). Secondary strengths hold the spec's adjusted values
/// (a coded 3 reads as 4).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Cdef {
    pub damping_minus_3: u8,
    pub bits: u8,
    pub y_pri_strength: [u8; 8],
    pub y_sec_strength: [u8; 8],
    pub uv_pri_strength: [u8; 8],
    pub uv_sec_strength: [u8; 8],
}

/// lr_params().
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LoopRestoration {
    pub frame_restoration_type: [RestorationType; 3],
    /// LoopRestorationSize in samples; zero for every plane when unused.
    pub loop_restoration_size: [u16; 3],
    pub uses_lr: bool,
    pub uses_chroma_lr: bool,
}

/// global_motion_params(), indexed by reference name (0 unused).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GlobalMotion {
    pub gm_type: [WarpModel; NUM_REF_FRAMES],
    pub gm_params: [[i32; 6]; NUM_REF_FRAMES],
}

impl Default for GlobalMotion {
    fn default() -> Self {
        Self {
            gm_type: [WarpModel::Identity; NUM_REF_FRAMES],
            gm_params: default_gm_params(),
        }
    }
}

/// The identity warp for every reference.
pub fn default_gm_params() -> [[i32; 6]; NUM_REF_FRAMES] {
    let identity = [
        0,
        0,
        1 << WARPEDMODEL_PREC_BITS,
        0,
        0,
        1 << WARPEDMODEL_PREC_BITS,
    ];
    [identity; NUM_REF_FRAMES]
}

/// film_grain_params(). With update_grain clear, the parameters hold the
/// values loaded from film_grain_params_ref_idx (with this frame's
/// grain_seed), so they are always complete.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FilmGrainParams {
    pub apply_grain: bool,
    pub grain_seed: u16,
    pub update_grain: bool,
    pub film_grain_params_ref_idx: u8,
    pub num_y_points: u8,
    pub point_y_value: [u8; 14],
    pub point_y_scaling: [u8; 14],
    pub chroma_scaling_from_luma: bool,
    pub num_cb_points: u8,
    pub point_cb_value: [u8; 10],
    pub point_cb_scaling: [u8; 10],
    pub num_cr_points: u8,
    pub point_cr_value: [u8; 10],
    pub point_cr_scaling: [u8; 10],
    pub grain_scaling_minus_8: u8,
    pub ar_coeff_lag: u8,
    pub ar_coeffs_y_plus_128: [u8; 24],
    pub ar_coeffs_cb_plus_128: [u8; 25],
    pub ar_coeffs_cr_plus_128: [u8; 25],
    pub ar_coeff_shift_minus_6: u8,
    pub grain_scale_shift: u8,
    pub cb_mult: u8,
    pub cb_luma_mult: u8,
    pub cb_offset: u16,
    pub cr_mult: u8,
    pub cr_luma_mult: u8,
    pub cr_offset: u16,
    pub overlap_flag: bool,
    pub clip_to_restricted_range: bool,
}

impl Default for FilmGrainParams {
    /// reset_grain_params(): everything zero.
    fn default() -> Self {
        Self {
            apply_grain: false,
            grain_seed: 0,
            update_grain: false,
            film_grain_params_ref_idx: 0,
            num_y_points: 0,
            point_y_value: [0; 14],
            point_y_scaling: [0; 14],
            chroma_scaling_from_luma: false,
            num_cb_points: 0,
            point_cb_value: [0; 10],
            point_cb_scaling: [0; 10],
            num_cr_points: 0,
            point_cr_value: [0; 10],
            point_cr_scaling: [0; 10],
            grain_scaling_minus_8: 0,
            ar_coeff_lag: 0,
            ar_coeffs_y_plus_128: [0; 24],
            ar_coeffs_cb_plus_128: [0; 25],
            ar_coeffs_cr_plus_128: [0; 25],
            ar_coeff_shift_minus_6: 0,
            grain_scale_shift: 0,
            cb_mult: 0,
            cb_luma_mult: 0,
            cb_offset: 0,
            cr_mult: 0,
            cr_luma_mult: 0,
            cr_offset: 0,
            overlap_flag: false,
            clip_to_restricted_range: false,
        }
    }
}

/// A parsed uncompressed_header().
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameHeader {
    pub show_existing_frame: bool,
    pub frame_to_show_map_idx: u8,
    /// temporal_point_info(), when present.
    pub frame_presentation_time: Option<u32>,
    pub display_frame_id: u32,
    pub frame_type: FrameType,
    pub show_frame: bool,
    pub showable_frame: bool,
    pub error_resilient_mode: bool,
    pub disable_cdf_update: bool,
    pub allow_screen_content_tools: bool,
    pub force_integer_mv: bool,
    pub current_frame_id: u32,
    pub frame_size_override_flag: bool,
    /// OrderHint.
    pub order_hint: u8,
    pub primary_ref_frame: u8,
    pub buffer_removal_time_present_flag: bool,
    /// buffer_removal_time per operating point (zero where absent).
    pub buffer_removal_time: Vec<u32>,
    pub refresh_frame_flags: u8,
    /// ref_order_hint[], when coded (error resilient frames with order hints).
    pub ref_order_hint: Option<[u8; NUM_REF_FRAMES]>,
    pub allow_intrabc: bool,
    pub frame_refs_short_signaling: bool,
    pub last_frame_idx: u8,
    pub gold_frame_idx: u8,
    /// ref_frame_idx[]: the slot each of LAST..ALTREF uses. Meaningless on
    /// intra frames.
    pub ref_frame_idx: [u8; REFS_PER_FRAME],
    /// expectedFrameId[], when frame ids are present.
    pub expected_frame_id: [u32; REFS_PER_FRAME],
    pub size: FrameSize,
    pub allow_high_precision_mv: bool,
    pub is_filter_switchable: bool,
    pub interpolation_filter: InterpolationFilter,
    pub is_motion_mode_switchable: bool,
    pub use_ref_frame_mvs: bool,
    /// OrderHints[], indexed by reference name.
    pub order_hints: [u8; NUM_REF_FRAMES],
    /// RefFrameSignBias[] as a bit mask indexed by reference name.
    pub ref_frame_sign_bias: u8,
    pub disable_frame_end_update_cdf: bool,
    pub tile_info: TileInfo,
    pub quantization: Quantization,
    pub segmentation: Segmentation,
    pub delta: DeltaParams,
    pub coded_lossless: bool,
    pub all_lossless: bool,
    pub lossless_array: [bool; MAX_SEGMENTS],
    /// SegQMLevel[plane][segment], when using_qmatrix.
    pub seg_qm_level: [[u8; MAX_SEGMENTS]; 3],
    pub loop_filter: LoopFilter,
    pub cdef: Cdef,
    pub loop_restoration: LoopRestoration,
    pub tx_mode: TxMode,
    pub reference_select: bool,
    pub skip_mode_present: bool,
    /// SkipModeFrame[], reference names, when skip mode is allowed.
    pub skip_mode_frame: [u8; 2],
    pub allow_warped_motion: bool,
    pub reduced_tx_set: bool,
    pub global_motion: GlobalMotion,
    pub film_grain: FilmGrainParams,
    /// Bytes of the OBU payload the header took, up to (not including) the
    /// byte_alignment() that follows it in a frame OBU.
    pub header_bits: usize,
}

impl FrameHeader {
    /// FrameIsIntra.
    pub fn frame_is_intra(&self) -> bool {
        self.frame_type.is_intra()
    }

    /// get_qidx(1, segment): the segment's quantizer index without
    /// block-level deltas.
    pub fn qindex(&self, segment: usize) -> u8 {
        qidx(&self.quantization, &self.segmentation, segment)
    }

    fn empty(frame_type: FrameType) -> Self {
        Self {
            show_existing_frame: false,
            frame_to_show_map_idx: 0,
            frame_presentation_time: None,
            display_frame_id: 0,
            frame_type,
            show_frame: false,
            showable_frame: false,
            error_resilient_mode: false,
            disable_cdf_update: false,
            allow_screen_content_tools: false,
            force_integer_mv: false,
            current_frame_id: 0,
            frame_size_override_flag: false,
            order_hint: 0,
            primary_ref_frame: PRIMARY_REF_NONE,
            buffer_removal_time_present_flag: false,
            buffer_removal_time: Vec::new(),
            refresh_frame_flags: 0,
            ref_order_hint: None,
            allow_intrabc: false,
            frame_refs_short_signaling: false,
            last_frame_idx: 0,
            gold_frame_idx: 0,
            ref_frame_idx: [0; REFS_PER_FRAME],
            expected_frame_id: [0; REFS_PER_FRAME],
            size: FrameSize::default(),
            allow_high_precision_mv: false,
            is_filter_switchable: false,
            interpolation_filter: InterpolationFilter::EightTap,
            is_motion_mode_switchable: false,
            use_ref_frame_mvs: false,
            order_hints: [0; NUM_REF_FRAMES],
            ref_frame_sign_bias: 0,
            disable_frame_end_update_cdf: true,
            tile_info: TileInfo::default(),
            quantization: Quantization::default(),
            segmentation: Segmentation::default(),
            delta: DeltaParams::default(),
            coded_lossless: false,
            all_lossless: false,
            lossless_array: [false; MAX_SEGMENTS],
            seg_qm_level: [[0; MAX_SEGMENTS]; 3],
            loop_filter: LoopFilter::default(),
            cdef: Cdef::default(),
            loop_restoration: LoopRestoration::default(),
            tx_mode: TxMode::Only4x4,
            reference_select: false,
            skip_mode_present: false,
            skip_mode_frame: [0; 2],
            allow_warped_motion: false,
            reduced_tx_set: false,
            global_motion: GlobalMotion::default(),
            film_grain: FilmGrainParams::default(),
            header_bits: 0,
        }
    }

    /// Parses uncompressed_header() from `r`, reading and updating `state`
    /// as the spec does while parsing (RefValid and RefOrderHint resets,
    /// OrderHints, current_frame_id). The reference frame update process
    /// is left to the caller, which runs it once the frame is complete.
    pub fn parse(
        r: &mut BitReader<'_>,
        seq: &SequenceHeader,
        state: &mut RefState,
        temporal_id: u8,
        spatial_id: u8,
    ) -> Result<Self, Error> {
        let start = r.position();
        let mut h = parse_uncompressed_header(r, seq, state, temporal_id, spatial_id)?;
        h.header_bits = r.position() - start;
        Ok(h)
    }
}

fn qidx(q: &Quantization, seg: &Segmentation, segment: usize) -> u8 {
    if seg.feature_active(segment, SEG_LVL_ALT_Q) {
        let data = i32::from(seg.feature_data[segment][SEG_LVL_ALT_Q]);
        (i32::from(q.base_q_idx) + data).clamp(0, 255) as u8
    } else {
        q.base_q_idx
    }
}

fn parse_uncompressed_header(
    r: &mut BitReader<'_>,
    seq: &SequenceHeader,
    state: &mut RefState,
    temporal_id: u8,
    spatial_id: u8,
) -> Result<FrameHeader, Error> {
    let id_len = seq.frame_id_length();
    let all_frames = 0xffu8;
    let equal_picture_interval = seq
        .timing_info
        .as_ref()
        .is_some_and(|t| t.equal_picture_interval);
    let temporal_point_info = |r: &mut BitReader<'_>| -> Result<Option<u32>, Error> {
        match &seq.decoder_model_info {
            Some(m) if !equal_picture_interval => Ok(Some(
                r.f(u32::from(m.frame_presentation_time_length_minus_1) + 1)?,
            )),
            _ => Ok(None),
        }
    };

    let mut h;
    if seq.reduced_still_picture_header {
        h = FrameHeader::empty(FrameType::Key);
        h.show_frame = true;
    } else {
        if r.flag()? {
            // show_existing_frame
            let idx = r.f(3)? as u8;
            let slot = &state.slots[usize::from(idx)];
            if !slot.valid {
                return Err(Error::MissingReference(idx));
            }
            h = FrameHeader::empty(slot.frame_type);
            h.show_existing_frame = true;
            h.frame_to_show_map_idx = idx;
            h.frame_presentation_time = temporal_point_info(r)?;
            if seq.frame_id_numbers_present_flag {
                h.display_frame_id = r.f(id_len)?;
            }
            h.show_frame = true;
            h.current_frame_id = slot.frame_id;
            h.order_hint = slot.order_hint;
            h.order_hints = slot.saved_order_hints;
            h.size = FrameSize {
                frame_width: slot.frame_width,
                frame_height: slot.frame_height,
                upscaled_width: slot.upscaled_width,
                render_width: slot.render_width,
                render_height: slot.render_height,
                mi_cols: slot.mi_cols,
                mi_rows: slot.mi_rows,
                superres_denom: SUPERRES_NUM,
                ..FrameSize::default()
            };
            if h.frame_type == FrameType::Key {
                h.refresh_frame_flags = all_frames;
            }
            if seq.film_grain_params_present {
                h.film_grain = slot.film_grain.clone();
            }
            return Ok(h);
        }
        h = FrameHeader::empty(FrameType::from_raw(r.f(2)?));
        h.show_frame = r.flag()?;
        if h.show_frame {
            h.frame_presentation_time = temporal_point_info(r)?;
        }
        h.showable_frame = if h.show_frame {
            h.frame_type != FrameType::Key
        } else {
            r.flag()?
        };
        h.error_resilient_mode = if h.frame_type == FrameType::Switch
            || (h.frame_type == FrameType::Key && h.show_frame)
        {
            true
        } else {
            r.flag()?
        };
    }
    let frame_is_intra = h.frame_is_intra();

    if h.frame_type == FrameType::Key && h.show_frame {
        for slot in &mut state.slots {
            slot.valid = false;
            slot.order_hint = 0;
        }
        for hint in &mut state.order_hints[LAST_FRAME..] {
            *hint = 0;
        }
    }
    h.disable_cdf_update = r.flag()?;
    h.allow_screen_content_tools =
        if seq.seq_force_screen_content_tools == SELECT_SCREEN_CONTENT_TOOLS {
            r.flag()?
        } else {
            seq.seq_force_screen_content_tools != 0
        };
    h.force_integer_mv = if h.allow_screen_content_tools {
        if seq.seq_force_integer_mv == SELECT_INTEGER_MV {
            r.flag()?
        } else {
            seq.seq_force_integer_mv != 0
        }
    } else {
        false
    };
    if frame_is_intra {
        h.force_integer_mv = true;
    }
    if seq.frame_id_numbers_present_flag {
        h.current_frame_id = r.f(id_len)?;
        mark_ref_frames(seq, state, h.current_frame_id);
    }
    state.current_frame_id = h.current_frame_id;
    h.frame_size_override_flag = if h.frame_type == FrameType::Switch {
        true
    } else if seq.reduced_still_picture_header {
        false
    } else {
        r.flag()?
    };
    h.order_hint = r.f(u32::from(seq.order_hint_bits))? as u8;
    h.primary_ref_frame = if frame_is_intra || h.error_resilient_mode {
        PRIMARY_REF_NONE
    } else {
        r.f(3)? as u8
    };
    if let Some(model) = &seq.decoder_model_info {
        h.buffer_removal_time_present_flag = r.flag()?;
        if h.buffer_removal_time_present_flag {
            h.buffer_removal_time = vec![0; seq.operating_points.len()];
            for (op_num, op) in seq.operating_points.iter().enumerate() {
                if op.decoder_model_present {
                    let in_temporal = (op.idc >> temporal_id) & 1 != 0;
                    let in_spatial = (op.idc >> (spatial_id + 8)) & 1 != 0;
                    if op.idc == 0 || (in_temporal && in_spatial) {
                        let n = u32::from(model.buffer_removal_time_length_minus_1) + 1;
                        h.buffer_removal_time[op_num] = r.f(n)?;
                    }
                }
            }
        }
    }
    h.refresh_frame_flags =
        if h.frame_type == FrameType::Switch || (h.frame_type == FrameType::Key && h.show_frame) {
            all_frames
        } else {
            r.f(8)? as u8
        };
    if (!frame_is_intra || h.refresh_frame_flags != all_frames)
        && h.error_resilient_mode
        && seq.enable_order_hint
    {
        let mut hints = [0u8; NUM_REF_FRAMES];
        for (i, hint) in hints.iter_mut().enumerate() {
            *hint = r.f(u32::from(seq.order_hint_bits))? as u8;
            let slot = &mut state.slots[i];
            if *hint != slot.order_hint || !slot.valid {
                // The slot no longer holds the frame the encoder means;
                // the decoder substitutes a frame with this order hint.
                slot.valid = false;
                slot.order_hint = *hint;
            }
        }
        h.ref_order_hint = Some(hints);
    }

    if frame_is_intra {
        frame_size(r, seq, &mut h)?;
        render_size(r, &mut h)?;
        if h.allow_screen_content_tools && h.size.upscaled_width == h.size.frame_width {
            h.allow_intrabc = r.flag()?;
        }
    } else {
        if seq.enable_order_hint {
            h.frame_refs_short_signaling = r.flag()?;
            if h.frame_refs_short_signaling {
                h.last_frame_idx = r.f(3)? as u8;
                h.gold_frame_idx = r.f(3)? as u8;
                h.ref_frame_idx = set_frame_refs(seq, state, &h)?;
            }
        }
        for i in 0..REFS_PER_FRAME {
            if !h.frame_refs_short_signaling {
                h.ref_frame_idx[i] = r.f(3)? as u8;
            }
            if seq.frame_id_numbers_present_flag {
                let n = u32::from(seq.delta_frame_id_length_minus_2) + 2;
                let delta_frame_id = r.f(n)? + 1;
                h.expected_frame_id[i] =
                    (h.current_frame_id + (1 << id_len) - delta_frame_id) % (1 << id_len);
            }
        }
        for &idx in &h.ref_frame_idx {
            if !state.slots[usize::from(idx)].valid {
                return Err(Error::MissingReference(idx));
            }
        }
        if h.frame_size_override_flag && !h.error_resilient_mode {
            frame_size_with_refs(r, seq, state, &mut h)?;
        } else {
            frame_size(r, seq, &mut h)?;
            render_size(r, &mut h)?;
        }
        h.allow_high_precision_mv = if h.force_integer_mv { false } else { r.flag()? };
        h.is_filter_switchable = r.flag()?;
        h.interpolation_filter = if h.is_filter_switchable {
            InterpolationFilter::Switchable
        } else {
            match r.f(2)? {
                0 => InterpolationFilter::EightTap,
                1 => InterpolationFilter::EightTapSmooth,
                2 => InterpolationFilter::EightTapSharp,
                _ => InterpolationFilter::Bilinear,
            }
        };
        h.is_motion_mode_switchable = r.flag()?;
        h.use_ref_frame_mvs = if h.error_resilient_mode || !seq.enable_ref_frame_mvs {
            false
        } else {
            r.flag()?
        };
        for i in 0..REFS_PER_FRAME {
            let ref_frame = LAST_FRAME + i;
            let hint = state.slots[usize::from(h.ref_frame_idx[i])].order_hint;
            state.order_hints[ref_frame] = hint;
            let backward = seq.enable_order_hint
                && seq.relative_dist(u32::from(hint), u32::from(h.order_hint)) > 0;
            if backward {
                state.ref_frame_sign_bias |= 1 << ref_frame;
            } else {
                state.ref_frame_sign_bias &= !(1 << ref_frame);
            }
        }
    }
    h.order_hints = state.order_hints;
    h.ref_frame_sign_bias = state.ref_frame_sign_bias;

    h.disable_frame_end_update_cdf = if seq.reduced_still_picture_header || h.disable_cdf_update {
        true
    } else {
        r.flag()?
    };

    // load_previous() or setup_past_independence(): the state the
    // segmentation, loop filter and global motion syntax builds on.
    let mut prev_gm_params = default_gm_params();
    if h.primary_ref_frame == PRIMARY_REF_NONE {
        h.loop_filter.ref_deltas = DEFAULT_LOOP_FILTER_REF_DELTAS;
        h.loop_filter.mode_deltas = [0; 2];
    } else {
        let prev = &state.slots[usize::from(h.ref_frame_idx[usize::from(h.primary_ref_frame)])];
        prev_gm_params = prev.gm_params;
        h.loop_filter.ref_deltas = prev.loop_filter_ref_deltas;
        h.loop_filter.mode_deltas = prev.loop_filter_mode_deltas;
        h.segmentation.feature_enabled = prev.feature_enabled;
        h.segmentation.feature_data = prev.feature_data;
    }

    tile_info(r, seq, &mut h)?;
    quantization_params(r, seq, &mut h)?;
    segmentation_params(r, &mut h)?;
    delta_params(r, &mut h)?;

    h.coded_lossless = true;
    for segment in 0..MAX_SEGMENTS {
        let q = &h.quantization;
        let lossless = qidx(q, &h.segmentation, segment) == 0
            && q.delta_q_y_dc == 0
            && q.delta_q_u_ac == 0
            && q.delta_q_u_dc == 0
            && q.delta_q_v_ac == 0
            && q.delta_q_v_dc == 0;
        h.lossless_array[segment] = lossless;
        if !lossless {
            h.coded_lossless = false;
        }
        if q.using_qmatrix {
            let levels = if lossless {
                [15; 3]
            } else {
                [q.qm_y, q.qm_u, q.qm_v]
            };
            for (plane, level) in levels.into_iter().enumerate() {
                h.seg_qm_level[plane][segment] = level;
            }
        }
    }
    h.all_lossless = h.coded_lossless && h.size.frame_width == h.size.upscaled_width;

    loop_filter_params(r, seq, &mut h)?;
    cdef_params(r, seq, &mut h)?;
    lr_params(r, seq, &mut h)?;
    h.tx_mode = if h.coded_lossless {
        TxMode::Only4x4
    } else if r.flag()? {
        TxMode::Select
    } else {
        TxMode::Largest
    };
    h.reference_select = if frame_is_intra { false } else { r.flag()? };
    skip_mode_params(r, seq, state, &mut h)?;
    h.allow_warped_motion = if frame_is_intra || h.error_resilient_mode || !seq.enable_warped_motion
    {
        false
    } else {
        r.flag()?
    };
    h.reduced_tx_set = r.flag()?;
    global_motion_params(r, &mut h, &prev_gm_params)?;
    film_grain_params(r, seq, state, &mut h)?;
    Ok(h)
}

/// mark_ref_frames(idLen) of section 5.9.4.
fn mark_ref_frames(seq: &SequenceHeader, state: &mut RefState, current_frame_id: u32) {
    let id_len = seq.frame_id_length();
    let diff_len = u32::from(seq.delta_frame_id_length_minus_2) + 2;
    for slot in &mut state.slots {
        let id = slot.frame_id;
        if current_frame_id > (1 << diff_len) {
            if id > current_frame_id || id < current_frame_id - (1 << diff_len) {
                slot.valid = false;
            }
        } else if id > current_frame_id && id < (1 << id_len) + current_frame_id - (1 << diff_len) {
            slot.valid = false;
        }
    }
}

/// set_frame_refs() of section 7.8: the reference selection behind
/// frame_refs_short_signaling.
fn set_frame_refs(
    seq: &SequenceHeader,
    state: &RefState,
    h: &FrameHeader,
) -> Result<[u8; REFS_PER_FRAME], Error> {
    let mut ref_frame_idx = [-1i32; REFS_PER_FRAME];
    ref_frame_idx[0] = i32::from(h.last_frame_idx);
    ref_frame_idx[GOLDEN_FRAME - LAST_FRAME] = i32::from(h.gold_frame_idx);
    let mut used = [false; NUM_REF_FRAMES];
    used[usize::from(h.last_frame_idx)] = true;
    used[usize::from(h.gold_frame_idx)] = true;
    let cur_frame_hint = 1i32 << (seq.order_hint_bits - 1);
    let shifted: [i32; NUM_REF_FRAMES] = std::array::from_fn(|i| {
        cur_frame_hint
            + seq.relative_dist(
                u32::from(state.slots[i].order_hint),
                u32::from(h.order_hint),
            )
    });
    if shifted[usize::from(h.last_frame_idx)] >= cur_frame_hint
        || shifted[usize::from(h.gold_frame_idx)] >= cur_frame_hint
    {
        return Err(Error::Invalid(
            "short signaling LAST or GOLDEN is not a forward reference",
        ));
    }

    // find_latest_backward, find_earliest_backward, find_latest_forward.
    let find = |used: &[bool; NUM_REF_FRAMES], backward: bool, latest: bool| -> Option<usize> {
        let mut best: Option<(usize, i32)> = None;
        for (i, &hint) in shifted.iter().enumerate() {
            let side = if backward {
                hint >= cur_frame_hint
            } else {
                hint < cur_frame_hint
            };
            if used[i] || !side {
                continue;
            }
            let better = match best {
                None => true,
                Some((_, b)) => {
                    if latest {
                        hint >= b
                    } else {
                        hint < b
                    }
                }
            };
            if better {
                best = Some((i, hint));
            }
        }
        best.map(|(i, _)| i)
    };

    if let Some(i) = find(&used, true, true) {
        ref_frame_idx[ALTREF_FRAME - LAST_FRAME] = i as i32;
        used[i] = true;
    }
    if let Some(i) = find(&used, true, false) {
        ref_frame_idx[BWDREF_FRAME - LAST_FRAME] = i as i32;
        used[i] = true;
    }
    if let Some(i) = find(&used, true, false) {
        ref_frame_idx[ALTREF2_FRAME - LAST_FRAME] = i as i32;
        used[i] = true;
    }
    for ref_frame in [
        LAST2_FRAME,
        LAST3_FRAME,
        BWDREF_FRAME,
        ALTREF2_FRAME,
        ALTREF_FRAME,
    ] {
        if ref_frame_idx[ref_frame - LAST_FRAME] < 0
            && let Some(i) = find(&used, false, true)
        {
            ref_frame_idx[ref_frame - LAST_FRAME] = i as i32;
            used[i] = true;
        }
    }
    // Anything left gets the reference with the earliest order hint.
    let mut earliest: Option<(usize, i32)> = None;
    for (i, &hint) in shifted.iter().enumerate() {
        if earliest.is_none_or(|(_, e)| hint < e) {
            earliest = Some((i, hint));
        }
    }
    let earliest = earliest.map_or(0, |(i, _)| i as i32);
    Ok(ref_frame_idx.map(|i| if i < 0 { earliest } else { i } as u8))
}

/// frame_size() with superres_params() and compute_image_size().
fn frame_size(
    r: &mut BitReader<'_>,
    seq: &SequenceHeader,
    h: &mut FrameHeader,
) -> Result<(), Error> {
    if h.frame_size_override_flag {
        h.size.frame_width = r.f(u32::from(seq.frame_width_bits_minus_1) + 1)? + 1;
        h.size.frame_height = r.f(u32::from(seq.frame_height_bits_minus_1) + 1)? + 1;
    } else {
        h.size.frame_width = u32::from(seq.max_frame_width_minus_1) + 1;
        h.size.frame_height = u32::from(seq.max_frame_height_minus_1) + 1;
    }
    superres_params(r, seq, &mut h.size)
}

fn superres_params(
    r: &mut BitReader<'_>,
    seq: &SequenceHeader,
    s: &mut FrameSize,
) -> Result<(), Error> {
    s.use_superres = seq.enable_superres && r.flag()?;
    if s.use_superres {
        s.coded_denom = r.f(SUPERRES_DENOM_BITS)? as u8;
        s.superres_denom = u32::from(s.coded_denom) + SUPERRES_DENOM_MIN;
    } else {
        s.coded_denom = 0;
        s.superres_denom = SUPERRES_NUM;
    }
    s.upscaled_width = s.frame_width;
    s.frame_width = (s.upscaled_width * SUPERRES_NUM + s.superres_denom / 2) / s.superres_denom;
    s.mi_cols = 2 * ((s.frame_width + 7) >> 3);
    s.mi_rows = 2 * ((s.frame_height + 7) >> 3);
    Ok(())
}

fn render_size(r: &mut BitReader<'_>, h: &mut FrameHeader) -> Result<(), Error> {
    h.size.render_and_frame_size_different = r.flag()?;
    if h.size.render_and_frame_size_different {
        h.size.render_width = r.f(16)? + 1;
        h.size.render_height = r.f(16)? + 1;
    } else {
        h.size.render_width = h.size.upscaled_width;
        h.size.render_height = h.size.frame_height;
    }
    Ok(())
}

fn frame_size_with_refs(
    r: &mut BitReader<'_>,
    seq: &SequenceHeader,
    state: &RefState,
    h: &mut FrameHeader,
) -> Result<(), Error> {
    for i in 0..REFS_PER_FRAME {
        if r.flag()? {
            let slot = &state.slots[usize::from(h.ref_frame_idx[i])];
            h.size.found_ref = true;
            h.size.frame_width = slot.upscaled_width;
            h.size.frame_height = slot.frame_height;
            h.size.render_width = slot.render_width;
            h.size.render_height = slot.render_height;
            return superres_params(r, seq, &mut h.size);
        }
    }
    frame_size(r, seq, h)?;
    render_size(r, h)
}

fn tile_log2(blk_size: u32, target: u32) -> u32 {
    let mut k = 0;
    while (blk_size << k) < target {
        k += 1;
    }
    k
}

/// tile_info() of section 5.9.15.
fn tile_info(
    r: &mut BitReader<'_>,
    seq: &SequenceHeader,
    h: &mut FrameHeader,
) -> Result<(), Error> {
    let (mi_cols, mi_rows) = (h.size.mi_cols, h.size.mi_rows);
    let sb_shift = if seq.use_128x128_superblock { 5 } else { 4 };
    let sb_cols = (mi_cols + (1 << sb_shift) - 1) >> sb_shift;
    let sb_rows = (mi_rows + (1 << sb_shift) - 1) >> sb_shift;
    let sb_size = sb_shift + 2;
    let max_tile_width_sb = MAX_TILE_WIDTH >> sb_size;
    let mut max_tile_area_sb = MAX_TILE_AREA >> (2 * sb_size);
    let min_log2_tile_cols = tile_log2(max_tile_width_sb, sb_cols);
    let max_log2_tile_cols = tile_log2(1, sb_cols.min(MAX_TILE_COLS));
    let max_log2_tile_rows = tile_log2(1, sb_rows.min(MAX_TILE_ROWS));
    let min_log2_tiles = min_log2_tile_cols.max(tile_log2(max_tile_area_sb, sb_rows * sb_cols));

    let t = &mut h.tile_info;
    t.uniform_tile_spacing_flag = r.flag()?;
    // Superblock starts; MiColStarts are these shifted, with MiCols last.
    let mut col_starts_sb = Vec::new();
    let mut row_starts_sb = Vec::new();
    if t.uniform_tile_spacing_flag {
        t.tile_cols_log2 = min_log2_tile_cols;
        while t.tile_cols_log2 < max_log2_tile_cols && r.flag()? {
            t.tile_cols_log2 += 1;
        }
        let tile_width_sb = (sb_cols + (1 << t.tile_cols_log2) - 1) >> t.tile_cols_log2;
        col_starts_sb.extend((0..sb_cols).step_by(tile_width_sb as usize));
        let min_log2_tile_rows = min_log2_tiles.saturating_sub(t.tile_cols_log2);
        t.tile_rows_log2 = min_log2_tile_rows;
        while t.tile_rows_log2 < max_log2_tile_rows && r.flag()? {
            t.tile_rows_log2 += 1;
        }
        let tile_height_sb = (sb_rows + (1 << t.tile_rows_log2) - 1) >> t.tile_rows_log2;
        row_starts_sb.extend((0..sb_rows).step_by(tile_height_sb as usize));
    } else {
        let mut widest_tile_sb = 0;
        let mut start_sb = 0;
        while start_sb < sb_cols {
            if col_starts_sb.len() as u32 >= MAX_TILE_COLS {
                return Err(Error::Invalid("more than 64 tile columns"));
            }
            col_starts_sb.push(start_sb);
            let max_width = (sb_cols - start_sb).min(max_tile_width_sb);
            let size_sb = r.ns(max_width)? + 1;
            widest_tile_sb = widest_tile_sb.max(size_sb);
            start_sb += size_sb;
        }
        t.tile_cols_log2 = tile_log2(1, col_starts_sb.len() as u32);
        if min_log2_tiles > 0 {
            max_tile_area_sb = (sb_rows * sb_cols) >> (min_log2_tiles + 1);
        } else {
            max_tile_area_sb = sb_rows * sb_cols;
        }
        let max_tile_height_sb = (max_tile_area_sb / widest_tile_sb).max(1);
        start_sb = 0;
        while start_sb < sb_rows {
            if row_starts_sb.len() as u32 >= MAX_TILE_ROWS {
                return Err(Error::Invalid("more than 64 tile rows"));
            }
            row_starts_sb.push(start_sb);
            let max_height = (sb_rows - start_sb).min(max_tile_height_sb);
            start_sb += r.ns(max_height)? + 1;
        }
        t.tile_rows_log2 = tile_log2(1, row_starts_sb.len() as u32);
    }
    t.tile_cols = col_starts_sb.len() as u32;
    t.tile_rows = row_starts_sb.len() as u32;
    let sizes = |starts: &[u32], total: u32| -> Vec<u32> {
        (0..starts.len())
            .map(|i| starts.get(i + 1).copied().unwrap_or(total) - starts[i] - 1)
            .collect()
    };
    t.width_in_sbs_minus_1 = sizes(&col_starts_sb, sb_cols);
    t.height_in_sbs_minus_1 = sizes(&row_starts_sb, sb_rows);
    t.mi_col_starts = col_starts_sb
        .iter()
        .map(|s| s << sb_shift)
        .chain([mi_cols])
        .collect();
    t.mi_row_starts = row_starts_sb
        .iter()
        .map(|s| s << sb_shift)
        .chain([mi_rows])
        .collect();
    if t.tile_cols_log2 > 0 || t.tile_rows_log2 > 0 {
        t.context_update_tile_id = r.f(t.tile_rows_log2 + t.tile_cols_log2)?;
        if t.context_update_tile_id >= t.tile_cols * t.tile_rows {
            return Err(Error::Invalid("context_update_tile_id out of range"));
        }
        t.tile_size_bytes_minus_1 = r.f(2)? as u8;
    } else {
        t.context_update_tile_id = 0;
    }
    Ok(())
}

fn quantization_params(
    r: &mut BitReader<'_>,
    seq: &SequenceHeader,
    h: &mut FrameHeader,
) -> Result<(), Error> {
    let c = &seq.color_config;
    let q = &mut h.quantization;
    q.base_q_idx = r.f(8)? as u8;
    q.delta_q_y_dc = r.delta_q()?;
    if c.num_planes() > 1 {
        q.diff_uv_delta = c.separate_uv_delta_q && r.flag()?;
        q.delta_q_u_dc = r.delta_q()?;
        q.delta_q_u_ac = r.delta_q()?;
        if q.diff_uv_delta {
            q.delta_q_v_dc = r.delta_q()?;
            q.delta_q_v_ac = r.delta_q()?;
        } else {
            q.delta_q_v_dc = q.delta_q_u_dc;
            q.delta_q_v_ac = q.delta_q_u_ac;
        }
    }
    q.using_qmatrix = r.flag()?;
    if q.using_qmatrix {
        q.qm_y = r.f(4)? as u8;
        q.qm_u = r.f(4)? as u8;
        q.qm_v = if c.separate_uv_delta_q {
            r.f(4)? as u8
        } else {
            q.qm_u
        };
    }
    Ok(())
}

fn segmentation_params(r: &mut BitReader<'_>, h: &mut FrameHeader) -> Result<(), Error> {
    let primary_none = h.primary_ref_frame == PRIMARY_REF_NONE;
    let s = &mut h.segmentation;
    s.enabled = r.flag()?;
    if s.enabled {
        if primary_none {
            s.update_map = true;
            s.temporal_update = false;
            s.update_data = true;
        } else {
            s.update_map = r.flag()?;
            s.temporal_update = s.update_map && r.flag()?;
            s.update_data = r.flag()?;
        }
        if s.update_data {
            for i in 0..MAX_SEGMENTS {
                for j in 0..SEG_LVL_MAX {
                    let enabled = r.flag()?;
                    s.feature_enabled[i][j] = enabled;
                    let mut value = 0;
                    if enabled {
                        let bits = SEGMENTATION_FEATURE_BITS[j];
                        let limit = SEGMENTATION_FEATURE_MAX[j];
                        value = if SEGMENTATION_FEATURE_SIGNED[j] {
                            r.su(1 + bits)?.clamp(-limit, limit)
                        } else {
                            (r.f(bits)? as i32).clamp(0, limit)
                        };
                    }
                    s.feature_data[i][j] = value as i16;
                }
            }
        }
    } else {
        s.update_map = false;
        s.temporal_update = false;
        s.update_data = false;
        s.feature_enabled = [[false; SEG_LVL_MAX]; MAX_SEGMENTS];
        s.feature_data = [[0; SEG_LVL_MAX]; MAX_SEGMENTS];
    }
    s.seg_id_pre_skip = false;
    s.last_active_seg_id = 0;
    for i in 0..MAX_SEGMENTS {
        for j in 0..SEG_LVL_MAX {
            if s.feature_enabled[i][j] {
                s.last_active_seg_id = i as u8;
                if j >= SEG_LVL_REF_FRAME {
                    s.seg_id_pre_skip = true;
                }
            }
        }
    }
    Ok(())
}

fn delta_params(r: &mut BitReader<'_>, h: &mut FrameHeader) -> Result<(), Error> {
    let d = &mut h.delta;
    if h.quantization.base_q_idx > 0 {
        d.delta_q_present = r.flag()?;
    }
    if d.delta_q_present {
        d.delta_q_res = r.f(2)? as u8;
        if !h.allow_intrabc {
            d.delta_lf_present = r.flag()?;
        }
        if d.delta_lf_present {
            d.delta_lf_res = r.f(2)? as u8;
            d.delta_lf_multi = r.flag()?;
        }
    }
    Ok(())
}

fn loop_filter_params(
    r: &mut BitReader<'_>,
    seq: &SequenceHeader,
    h: &mut FrameHeader,
) -> Result<(), Error> {
    let lf = &mut h.loop_filter;
    if h.coded_lossless || h.allow_intrabc {
        lf.level[0] = 0;
        lf.level[1] = 0;
        lf.ref_deltas = DEFAULT_LOOP_FILTER_REF_DELTAS;
        lf.mode_deltas = [0; 2];
        return Ok(());
    }
    lf.level[0] = r.f(6)? as u8;
    lf.level[1] = r.f(6)? as u8;
    if seq.color_config.num_planes() > 1 && (lf.level[0] != 0 || lf.level[1] != 0) {
        lf.level[2] = r.f(6)? as u8;
        lf.level[3] = r.f(6)? as u8;
    }
    lf.sharpness = r.f(3)? as u8;
    lf.delta_enabled = r.flag()?;
    if lf.delta_enabled {
        lf.delta_update = r.flag()?;
        if lf.delta_update {
            for i in 0..NUM_REF_FRAMES {
                if r.flag()? {
                    lf.update_ref_delta |= 1 << i;
                    lf.ref_deltas[i] = r.su(7)? as i8;
                }
            }
            for i in 0..2 {
                if r.flag()? {
                    lf.update_mode_delta |= 1 << i;
                    lf.mode_deltas[i] = r.su(7)? as i8;
                }
            }
        }
    }
    Ok(())
}

fn cdef_params(
    r: &mut BitReader<'_>,
    seq: &SequenceHeader,
    h: &mut FrameHeader,
) -> Result<(), Error> {
    let c = &mut h.cdef;
    if h.coded_lossless || h.allow_intrabc || !seq.enable_cdef {
        // cdef_bits, the first strengths and CdefDamping - 3 are zero.
        *c = Cdef::default();
        return Ok(());
    }
    c.damping_minus_3 = r.f(2)? as u8;
    c.bits = r.f(2)? as u8;
    for i in 0..(1 << c.bits) {
        c.y_pri_strength[i] = r.f(4)? as u8;
        c.y_sec_strength[i] = r.f(2)? as u8;
        if c.y_sec_strength[i] == 3 {
            c.y_sec_strength[i] += 1;
        }
        if seq.color_config.num_planes() > 1 {
            c.uv_pri_strength[i] = r.f(4)? as u8;
            c.uv_sec_strength[i] = r.f(2)? as u8;
            if c.uv_sec_strength[i] == 3 {
                c.uv_sec_strength[i] += 1;
            }
        }
    }
    Ok(())
}

fn lr_params(
    r: &mut BitReader<'_>,
    seq: &SequenceHeader,
    h: &mut FrameHeader,
) -> Result<(), Error> {
    let lr = &mut h.loop_restoration;
    *lr = LoopRestoration::default();
    if h.all_lossless || h.allow_intrabc || !seq.enable_restoration {
        return Ok(());
    }
    let c = &seq.color_config;
    for i in 0..c.num_planes() {
        // Remap_Lr_Type
        lr.frame_restoration_type[i] = match r.f(2)? {
            0 => RestorationType::None,
            1 => RestorationType::Switchable,
            2 => RestorationType::Wiener,
            _ => RestorationType::Sgrproj,
        };
        if lr.frame_restoration_type[i] != RestorationType::None {
            lr.uses_lr = true;
            if i > 0 {
                lr.uses_chroma_lr = true;
            }
        }
    }
    if lr.uses_lr {
        let mut lr_unit_shift = r.f(1)?;
        if seq.use_128x128_superblock {
            lr_unit_shift += 1;
        } else if lr_unit_shift != 0 {
            lr_unit_shift += r.f(1)?;
        }
        let size0 = RESTORATION_TILESIZE_MAX >> (2 - lr_unit_shift);
        let lr_uv_shift = if c.subsampling_x && c.subsampling_y && lr.uses_chroma_lr {
            r.f(1)?
        } else {
            0
        };
        lr.loop_restoration_size = [size0, size0 >> lr_uv_shift, size0 >> lr_uv_shift];
    }
    Ok(())
}

/// skip_mode_params() of section 5.9.22.
fn skip_mode_params(
    r: &mut BitReader<'_>,
    seq: &SequenceHeader,
    state: &RefState,
    h: &mut FrameHeader,
) -> Result<(), Error> {
    let mut allowed = false;
    if !h.frame_is_intra() && h.reference_select && seq.enable_order_hint {
        let dist = |a: u8, b: u8| seq.relative_dist(u32::from(a), u32::from(b));
        let hint_of = |i: usize| state.slots[usize::from(h.ref_frame_idx[i])].order_hint;
        let mut forward: Option<(usize, u8)> = None;
        let mut backward: Option<(usize, u8)> = None;
        for i in 0..REFS_PER_FRAME {
            let ref_hint = hint_of(i);
            if dist(ref_hint, h.order_hint) < 0 {
                if forward.is_none_or(|(_, f)| dist(ref_hint, f) > 0) {
                    forward = Some((i, ref_hint));
                }
            } else if dist(ref_hint, h.order_hint) > 0
                && backward.is_none_or(|(_, b)| dist(ref_hint, b) < 0)
            {
                backward = Some((i, ref_hint));
            }
        }
        let pair = match (forward, backward) {
            (None, _) => None,
            (Some((f, _)), Some((b, _))) => Some((f, b)),
            (Some((f, forward_hint)), None) => {
                let mut second: Option<(usize, u8)> = None;
                for i in 0..REFS_PER_FRAME {
                    let ref_hint = hint_of(i);
                    if dist(ref_hint, forward_hint) < 0
                        && second.is_none_or(|(_, s)| dist(ref_hint, s) > 0)
                    {
                        second = Some((i, ref_hint));
                    }
                }
                second.map(|(s, _)| (f, s))
            }
        };
        if let Some((a, b)) = pair {
            allowed = true;
            h.skip_mode_frame = [(LAST_FRAME + a.min(b)) as u8, (LAST_FRAME + a.max(b)) as u8];
        }
    }
    h.skip_mode_present = allowed && r.flag()?;
    Ok(())
}

/// global_motion_params() of section 5.9.24.
fn global_motion_params(
    r: &mut BitReader<'_>,
    h: &mut FrameHeader,
    prev: &[[i32; 6]; NUM_REF_FRAMES],
) -> Result<(), Error> {
    h.global_motion = GlobalMotion::default();
    if h.frame_is_intra() {
        return Ok(());
    }
    for ref_frame in LAST_FRAME..=ALTREF_FRAME {
        let model = if r.flag()? {
            if r.flag()? {
                WarpModel::RotZoom
            } else if r.flag()? {
                WarpModel::Translation
            } else {
                WarpModel::Affine
            }
        } else {
            WarpModel::Identity
        };
        h.global_motion.gm_type[ref_frame] = model;
        let hp = h.allow_high_precision_mv;
        let mut read = |idx: usize, gm: &mut [[i32; 6]; NUM_REF_FRAMES]| -> Result<(), Error> {
            let (abs_bits, prec_bits) = if idx < 2 {
                if model == WarpModel::Translation {
                    (
                        GM_ABS_TRANS_ONLY_BITS - u32::from(!hp),
                        GM_TRANS_ONLY_PREC_BITS - u32::from(!hp),
                    )
                } else {
                    (GM_ABS_TRANS_BITS, GM_TRANS_PREC_BITS)
                }
            } else {
                (GM_ABS_ALPHA_BITS, GM_ALPHA_PREC_BITS)
            };
            let prec_diff = WARPEDMODEL_PREC_BITS - prec_bits;
            let round = if idx % 3 == 2 {
                1i32 << WARPEDMODEL_PREC_BITS
            } else {
                0
            };
            let sub = if idx % 3 == 2 { 1i32 << prec_bits } else { 0 };
            let mx = 1i32 << abs_bits;
            let reference = (prev[ref_frame][idx] >> prec_diff) - sub;
            let value = r.signed_subexp_with_ref(-mx, mx + 1, reference)?;
            gm[ref_frame][idx] = (value << prec_diff) + round;
            Ok(())
        };
        let gm = &mut h.global_motion.gm_params;
        if model >= WarpModel::RotZoom {
            read(2, gm)?;
            read(3, gm)?;
            if model == WarpModel::Affine {
                read(4, gm)?;
                read(5, gm)?;
            } else {
                gm[ref_frame][4] = -gm[ref_frame][3];
                gm[ref_frame][5] = gm[ref_frame][2];
            }
        }
        if model >= WarpModel::Translation {
            read(0, gm)?;
            read(1, gm)?;
        }
    }
    Ok(())
}

impl PartialOrd for WarpModel {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some((*self as u8).cmp(&(*other as u8)))
    }
}

/// film_grain_params() of section 5.9.30.
fn film_grain_params(
    r: &mut BitReader<'_>,
    seq: &SequenceHeader,
    state: &RefState,
    h: &mut FrameHeader,
) -> Result<(), Error> {
    h.film_grain = FilmGrainParams::default();
    if !seq.film_grain_params_present || (!h.show_frame && !h.showable_frame) {
        return Ok(());
    }
    if !r.flag()? {
        return Ok(());
    }
    let grain_seed = r.f(16)? as u16;
    let update_grain = if h.frame_type == FrameType::Inter {
        r.flag()?
    } else {
        true
    };
    if !update_grain {
        let idx = r.f(3)? as u8;
        if !h.ref_frame_idx.contains(&idx) {
            return Err(Error::Invalid(
                "film_grain_params_ref_idx is not a reference",
            ));
        }
        let mut g = state.slots[usize::from(idx)].film_grain.clone();
        g.apply_grain = true;
        g.grain_seed = grain_seed;
        g.update_grain = false;
        g.film_grain_params_ref_idx = idx;
        h.film_grain = g;
        return Ok(());
    }
    let c = &seq.color_config;
    let g = &mut h.film_grain;
    g.apply_grain = true;
    g.grain_seed = grain_seed;
    g.update_grain = true;
    g.num_y_points = r.f(4)? as u8;
    if g.num_y_points > 14 {
        return Err(Error::Invalid("num_y_points above 14"));
    }
    for i in 0..usize::from(g.num_y_points) {
        g.point_y_value[i] = r.f(8)? as u8;
        g.point_y_scaling[i] = r.f(8)? as u8;
    }
    g.chroma_scaling_from_luma = !c.mono_chrome && r.flag()?;
    if c.mono_chrome
        || g.chroma_scaling_from_luma
        || (c.subsampling_x && c.subsampling_y && g.num_y_points == 0)
    {
        g.num_cb_points = 0;
        g.num_cr_points = 0;
    } else {
        g.num_cb_points = r.f(4)? as u8;
        if g.num_cb_points > 10 {
            return Err(Error::Invalid("num_cb_points above 10"));
        }
        for i in 0..usize::from(g.num_cb_points) {
            g.point_cb_value[i] = r.f(8)? as u8;
            g.point_cb_scaling[i] = r.f(8)? as u8;
        }
        g.num_cr_points = r.f(4)? as u8;
        if g.num_cr_points > 10 {
            return Err(Error::Invalid("num_cr_points above 10"));
        }
        for i in 0..usize::from(g.num_cr_points) {
            g.point_cr_value[i] = r.f(8)? as u8;
            g.point_cr_scaling[i] = r.f(8)? as u8;
        }
    }
    g.grain_scaling_minus_8 = r.f(2)? as u8;
    g.ar_coeff_lag = r.f(2)? as u8;
    let lag = usize::from(g.ar_coeff_lag);
    let num_pos_luma = 2 * lag * (lag + 1);
    let num_pos_chroma = if g.num_y_points != 0 {
        for i in 0..num_pos_luma {
            g.ar_coeffs_y_plus_128[i] = r.f(8)? as u8;
        }
        num_pos_luma + 1
    } else {
        num_pos_luma
    };
    if g.chroma_scaling_from_luma || g.num_cb_points != 0 {
        for i in 0..num_pos_chroma {
            g.ar_coeffs_cb_plus_128[i] = r.f(8)? as u8;
        }
    }
    if g.chroma_scaling_from_luma || g.num_cr_points != 0 {
        for i in 0..num_pos_chroma {
            g.ar_coeffs_cr_plus_128[i] = r.f(8)? as u8;
        }
    }
    g.ar_coeff_shift_minus_6 = r.f(2)? as u8;
    g.grain_scale_shift = r.f(2)? as u8;
    if g.num_cb_points != 0 {
        g.cb_mult = r.f(8)? as u8;
        g.cb_luma_mult = r.f(8)? as u8;
        g.cb_offset = r.f(9)? as u16;
    }
    if g.num_cr_points != 0 {
        g.cr_mult = r.f(8)? as u8;
        g.cr_luma_mult = r.f(8)? as u8;
        g.cr_offset = r.f(9)? as u16;
    }
    g.overlap_flag = r.flag()?;
    g.clip_to_restricted_range = r.flag()?;
    Ok(())
}
