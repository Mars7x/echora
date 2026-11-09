//! Plain-data mirrors of the Vulkan Video AV1 decode structures
//! (vulkan_video_codec_av1std.h and vulkan_video_codec_av1std_decode.h).
//!
//! Field names follow the Std headers (in snake case where the header
//! mixes styles); bitfield flags are plain bools; pointers become owned
//! values. Enums are stored as their Std numeric values. Converting to
//! the ash types is left to the Vulkan side.

use crate::frame::{FilmGrainParams, FrameHeader, NUM_REF_FRAMES};
use crate::refs::RefSlot;
use crate::sequence::SequenceHeader as ParsedSequenceHeader;

/// StdVideoAV1ColorConfig.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ColorConfig {
    pub mono_chrome: bool,
    pub color_range: bool,
    pub separate_uv_delta_q: bool,
    pub color_description_present_flag: bool,
    pub bit_depth: u8,
    pub subsampling_x: u8,
    pub subsampling_y: u8,
    pub color_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
    pub chroma_sample_position: u8,
}

/// StdVideoAV1TimingInfo.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TimingInfo {
    pub equal_picture_interval: bool,
    pub num_units_in_display_tick: u32,
    pub time_scale: u32,
    pub num_ticks_per_picture_minus_1: u32,
}

/// StdVideoAV1SequenceHeader.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SequenceHeader {
    pub still_picture: bool,
    pub reduced_still_picture_header: bool,
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
    pub frame_id_numbers_present_flag: bool,
    pub enable_superres: bool,
    pub enable_cdef: bool,
    pub enable_restoration: bool,
    pub film_grain_params_present: bool,
    pub timing_info_present_flag: bool,
    pub initial_display_delay_present_flag: bool,
    /// StdVideoAV1Profile: 0 main, 1 high, 2 professional.
    pub seq_profile: u8,
    pub frame_width_bits_minus_1: u8,
    pub frame_height_bits_minus_1: u8,
    pub max_frame_width_minus_1: u16,
    pub max_frame_height_minus_1: u16,
    pub delta_frame_id_length_minus_2: u8,
    pub additional_frame_id_length_minus_1: u8,
    /// Zero when order hints are off (the Std field is then unused).
    pub order_hint_bits_minus_1: u8,
    pub seq_force_integer_mv: u8,
    pub seq_force_screen_content_tools: u8,
    pub color_config: ColorConfig,
    pub timing_info: Option<TimingInfo>,
}

impl SequenceHeader {
    pub fn new(s: &ParsedSequenceHeader) -> Self {
        let c = &s.color_config;
        Self {
            still_picture: s.still_picture,
            reduced_still_picture_header: s.reduced_still_picture_header,
            use_128x128_superblock: s.use_128x128_superblock,
            enable_filter_intra: s.enable_filter_intra,
            enable_intra_edge_filter: s.enable_intra_edge_filter,
            enable_interintra_compound: s.enable_interintra_compound,
            enable_masked_compound: s.enable_masked_compound,
            enable_warped_motion: s.enable_warped_motion,
            enable_dual_filter: s.enable_dual_filter,
            enable_order_hint: s.enable_order_hint,
            enable_jnt_comp: s.enable_jnt_comp,
            enable_ref_frame_mvs: s.enable_ref_frame_mvs,
            frame_id_numbers_present_flag: s.frame_id_numbers_present_flag,
            enable_superres: s.enable_superres,
            enable_cdef: s.enable_cdef,
            enable_restoration: s.enable_restoration,
            film_grain_params_present: s.film_grain_params_present,
            timing_info_present_flag: s.timing_info.is_some(),
            initial_display_delay_present_flag: s.initial_display_delay_present_flag,
            seq_profile: s.seq_profile,
            frame_width_bits_minus_1: s.frame_width_bits_minus_1,
            frame_height_bits_minus_1: s.frame_height_bits_minus_1,
            max_frame_width_minus_1: s.max_frame_width_minus_1,
            max_frame_height_minus_1: s.max_frame_height_minus_1,
            delta_frame_id_length_minus_2: s.delta_frame_id_length_minus_2,
            additional_frame_id_length_minus_1: s.additional_frame_id_length_minus_1,
            order_hint_bits_minus_1: s.order_hint_bits.saturating_sub(1),
            seq_force_integer_mv: s.seq_force_integer_mv,
            seq_force_screen_content_tools: s.seq_force_screen_content_tools,
            color_config: ColorConfig {
                mono_chrome: c.mono_chrome,
                color_range: c.color_range,
                separate_uv_delta_q: c.separate_uv_delta_q,
                color_description_present_flag: c.color_description_present_flag,
                bit_depth: c.bit_depth,
                subsampling_x: u8::from(c.subsampling_x),
                subsampling_y: u8::from(c.subsampling_y),
                color_primaries: c.color_primaries,
                transfer_characteristics: c.transfer_characteristics,
                matrix_coefficients: c.matrix_coefficients,
                chroma_sample_position: c.chroma_sample_position,
            },
            timing_info: s.timing_info.as_ref().map(|t| TimingInfo {
                equal_picture_interval: t.equal_picture_interval,
                num_units_in_display_tick: t.num_units_in_display_tick,
                time_scale: t.time_scale,
                num_ticks_per_picture_minus_1: t.num_ticks_per_picture_minus_1,
            }),
        }
    }
}

/// StdVideoAV1TileInfo with its four arrays owned.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TileInfo {
    pub uniform_tile_spacing_flag: bool,
    pub tile_cols: u8,
    pub tile_rows: u8,
    pub context_update_tile_id: u16,
    pub tile_size_bytes_minus_1: u8,
    /// pMiColStarts: tile_cols entries.
    pub mi_col_starts: Vec<u16>,
    /// pMiRowStarts: tile_rows entries.
    pub mi_row_starts: Vec<u16>,
    /// pWidthInSbsMinus1: tile_cols entries.
    pub width_in_sbs_minus_1: Vec<u16>,
    /// pHeightInSbsMinus1: tile_rows entries.
    pub height_in_sbs_minus_1: Vec<u16>,
}

/// StdVideoAV1Quantization.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Quantization {
    pub using_qmatrix: bool,
    pub diff_uv_delta: bool,
    pub base_q_idx: u8,
    pub delta_q_y_dc: i8,
    pub delta_q_u_dc: i8,
    pub delta_q_u_ac: i8,
    pub delta_q_v_dc: i8,
    pub delta_q_v_ac: i8,
    pub qm_y: u8,
    pub qm_u: u8,
    pub qm_v: u8,
}

/// StdVideoAV1Segmentation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Segmentation {
    /// Bit j of entry i: feature j enabled for segment i.
    pub feature_enabled: [u8; 8],
    pub feature_data: [[i16; 8]; 8],
}

/// StdVideoAV1LoopFilter.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LoopFilter {
    pub loop_filter_delta_enabled: bool,
    pub loop_filter_delta_update: bool,
    pub loop_filter_level: [u8; 4],
    pub loop_filter_sharpness: u8,
    /// Bit mask over the eight references.
    pub update_ref_delta: u8,
    pub loop_filter_ref_deltas: [i8; 8],
    /// Bit mask over the two modes.
    pub update_mode_delta: u8,
    pub loop_filter_mode_deltas: [i8; 2],
}

/// StdVideoAV1CDEF. Secondary strengths are the coded two-bit values
/// (0..=3), as drivers expect: RADV packs `(pri << 2) + sec` for the
/// firmware. The spec's adjusted value (a coded 3 means 4) is in
/// [`crate::frame::Cdef`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Cdef {
    pub cdef_damping_minus_3: u8,
    pub cdef_bits: u8,
    pub cdef_y_pri_strength: [u8; 8],
    pub cdef_y_sec_strength: [u8; 8],
    pub cdef_uv_pri_strength: [u8; 8],
    pub cdef_uv_sec_strength: [u8; 8],
}

/// StdVideoAV1LoopRestoration.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LoopRestoration {
    /// StdVideoAV1FrameRestorationType values: 0 none, 1 Wiener,
    /// 2 self-guided, 3 switchable.
    pub frame_restoration_type: [u8; 3],
    /// LoopRestorationSize as Vulkan takes it: log2 of the unit size
    /// minus 5 per plane (0..=3 for 32..=256 samples); zero when loop
    /// restoration is off. The size in samples is in
    /// [`crate::frame::LoopRestoration`].
    pub log2_restoration_size_minus_5: [u16; 3],
}

/// StdVideoAV1GlobalMotion, indexed by reference name (0 is intra).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GlobalMotion {
    pub gm_type: [u8; 8],
    pub gm_params: [[i32; 6]; 8],
}

/// StdVideoAV1FilmGrain. The ar_coeffs_*_plus_128 values are stored as
/// coded (0..=255); the Std header declares them int8_t, so reinterpret
/// the bits (`as i8`) when filling it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FilmGrain {
    pub chroma_scaling_from_luma: bool,
    pub overlap_flag: bool,
    pub clip_to_restricted_range: bool,
    pub update_grain: bool,
    pub grain_scaling_minus_8: u8,
    pub ar_coeff_lag: u8,
    pub ar_coeff_shift_minus_6: u8,
    pub grain_scale_shift: u8,
    pub grain_seed: u16,
    pub film_grain_params_ref_idx: u8,
    pub num_y_points: u8,
    pub point_y_value: [u8; 14],
    pub point_y_scaling: [u8; 14],
    pub num_cb_points: u8,
    pub point_cb_value: [u8; 10],
    pub point_cb_scaling: [u8; 10],
    pub num_cr_points: u8,
    pub point_cr_value: [u8; 10],
    pub point_cr_scaling: [u8; 10],
    pub ar_coeffs_y_plus_128: [u8; 24],
    pub ar_coeffs_cb_plus_128: [u8; 25],
    pub ar_coeffs_cr_plus_128: [u8; 25],
    pub cb_mult: u8,
    pub cb_luma_mult: u8,
    pub cb_offset: u16,
    pub cr_mult: u8,
    pub cr_luma_mult: u8,
    pub cr_offset: u16,
}

impl FilmGrain {
    /// The grain to apply, or None when apply_grain is clear.
    pub fn new(g: &FilmGrainParams) -> Option<Self> {
        if !g.apply_grain {
            return None;
        }
        Some(Self {
            chroma_scaling_from_luma: g.chroma_scaling_from_luma,
            overlap_flag: g.overlap_flag,
            clip_to_restricted_range: g.clip_to_restricted_range,
            update_grain: g.update_grain,
            grain_scaling_minus_8: g.grain_scaling_minus_8,
            ar_coeff_lag: g.ar_coeff_lag,
            ar_coeff_shift_minus_6: g.ar_coeff_shift_minus_6,
            grain_scale_shift: g.grain_scale_shift,
            grain_seed: g.grain_seed,
            film_grain_params_ref_idx: g.film_grain_params_ref_idx,
            num_y_points: g.num_y_points,
            point_y_value: g.point_y_value,
            point_y_scaling: g.point_y_scaling,
            num_cb_points: g.num_cb_points,
            point_cb_value: g.point_cb_value,
            point_cb_scaling: g.point_cb_scaling,
            num_cr_points: g.num_cr_points,
            point_cr_value: g.point_cr_value,
            point_cr_scaling: g.point_cr_scaling,
            ar_coeffs_y_plus_128: g.ar_coeffs_y_plus_128,
            ar_coeffs_cb_plus_128: g.ar_coeffs_cb_plus_128,
            ar_coeffs_cr_plus_128: g.ar_coeffs_cr_plus_128,
            cb_mult: g.cb_mult,
            cb_luma_mult: g.cb_luma_mult,
            cb_offset: g.cb_offset,
            cr_mult: g.cr_mult,
            cr_luma_mult: g.cr_luma_mult,
            cr_offset: g.cr_offset,
        })
    }
}

/// StdVideoDecodeAV1PictureInfo with its sub-structures owned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PictureInfo {
    // StdVideoDecodeAV1PictureInfoFlags
    pub error_resilient_mode: bool,
    pub disable_cdf_update: bool,
    pub use_superres: bool,
    pub render_and_frame_size_different: bool,
    pub allow_screen_content_tools: bool,
    pub is_filter_switchable: bool,
    pub force_integer_mv: bool,
    pub frame_size_override_flag: bool,
    pub buffer_removal_time_present_flag: bool,
    pub allow_intrabc: bool,
    pub frame_refs_short_signaling: bool,
    pub allow_high_precision_mv: bool,
    pub is_motion_mode_switchable: bool,
    pub use_ref_frame_mvs: bool,
    pub disable_frame_end_update_cdf: bool,
    pub allow_warped_motion: bool,
    pub reduced_tx_set: bool,
    pub reference_select: bool,
    pub skip_mode_present: bool,
    pub delta_q_present: bool,
    pub delta_lf_present: bool,
    pub delta_lf_multi: bool,
    pub segmentation_enabled: bool,
    pub segmentation_update_map: bool,
    pub segmentation_temporal_update: bool,
    pub segmentation_update_data: bool,
    pub uses_lr: bool,
    pub uses_chroma_lr: bool,
    pub apply_grain: bool,

    /// StdVideoAV1FrameType value.
    pub frame_type: u8,
    pub current_frame_id: u32,
    pub order_hint: u8,
    pub primary_ref_frame: u8,
    pub refresh_frame_flags: u8,
    /// StdVideoAV1InterpolationFilter value.
    pub interpolation_filter: u8,
    /// StdVideoAV1TxMode value.
    pub tx_mode: u8,
    pub delta_q_res: u8,
    pub delta_lf_res: u8,
    /// Reference names (LAST_FRAME = 1 ...), zero when skip mode is off.
    pub skip_mode_frame: [u8; 2],
    pub coded_denom: u8,
    /// OrderHints[], indexed by reference name (entry 0 unused).
    pub order_hints: [u8; 8],
    /// expectedFrameId[], indexed by reference name like order_hints
    /// (entry 0 unused); zero without frame ids.
    pub expected_frame_id: [u32; 8],

    pub tile_info: TileInfo,
    pub quantization: Quantization,
    pub segmentation: Segmentation,
    pub loop_filter: LoopFilter,
    pub cdef: Cdef,
    pub loop_restoration: LoopRestoration,
    pub global_motion: GlobalMotion,
    /// pFilmGrain; None when no grain is applied.
    pub film_grain: Option<FilmGrain>,
}

impl PictureInfo {
    pub fn new(h: &FrameHeader) -> Self {
        let t = &h.tile_info;
        let q = &h.quantization;
        let s = &h.segmentation;
        let lf = &h.loop_filter;
        let c = &h.cdef;
        let lr = &h.loop_restoration;
        let gm = &h.global_motion;
        let mut expected_frame_id = [0; 8];
        expected_frame_id[1..].copy_from_slice(&h.expected_frame_id);
        let n_cols = t.tile_cols as usize;
        let n_rows = t.tile_rows as usize;
        Self {
            error_resilient_mode: h.error_resilient_mode,
            disable_cdf_update: h.disable_cdf_update,
            use_superres: h.size.use_superres,
            render_and_frame_size_different: h.size.render_and_frame_size_different,
            allow_screen_content_tools: h.allow_screen_content_tools,
            is_filter_switchable: h.is_filter_switchable,
            force_integer_mv: h.force_integer_mv,
            frame_size_override_flag: h.frame_size_override_flag,
            buffer_removal_time_present_flag: h.buffer_removal_time_present_flag,
            allow_intrabc: h.allow_intrabc,
            frame_refs_short_signaling: h.frame_refs_short_signaling,
            allow_high_precision_mv: h.allow_high_precision_mv,
            is_motion_mode_switchable: h.is_motion_mode_switchable,
            use_ref_frame_mvs: h.use_ref_frame_mvs,
            disable_frame_end_update_cdf: h.disable_frame_end_update_cdf,
            allow_warped_motion: h.allow_warped_motion,
            reduced_tx_set: h.reduced_tx_set,
            reference_select: h.reference_select,
            skip_mode_present: h.skip_mode_present,
            delta_q_present: h.delta.delta_q_present,
            delta_lf_present: h.delta.delta_lf_present,
            delta_lf_multi: h.delta.delta_lf_multi,
            segmentation_enabled: s.enabled,
            segmentation_update_map: s.update_map,
            segmentation_temporal_update: s.temporal_update,
            segmentation_update_data: s.update_data,
            uses_lr: lr.uses_lr,
            uses_chroma_lr: lr.uses_chroma_lr,
            apply_grain: h.film_grain.apply_grain,
            frame_type: h.frame_type as u8,
            current_frame_id: h.current_frame_id,
            order_hint: h.order_hint,
            primary_ref_frame: h.primary_ref_frame,
            refresh_frame_flags: h.refresh_frame_flags,
            interpolation_filter: h.interpolation_filter as u8,
            tx_mode: h.tx_mode as u8,
            delta_q_res: h.delta.delta_q_res,
            delta_lf_res: h.delta.delta_lf_res,
            skip_mode_frame: if h.skip_mode_present {
                h.skip_mode_frame
            } else {
                [0; 2]
            },
            coded_denom: h.size.coded_denom,
            order_hints: h.order_hints,
            expected_frame_id,
            tile_info: TileInfo {
                uniform_tile_spacing_flag: t.uniform_tile_spacing_flag,
                tile_cols: t.tile_cols as u8,
                tile_rows: t.tile_rows as u8,
                context_update_tile_id: t.context_update_tile_id as u16,
                tile_size_bytes_minus_1: t.tile_size_bytes_minus_1,
                mi_col_starts: t.mi_col_starts[..n_cols]
                    .iter()
                    .map(|&v| v as u16)
                    .collect(),
                mi_row_starts: t.mi_row_starts[..n_rows]
                    .iter()
                    .map(|&v| v as u16)
                    .collect(),
                width_in_sbs_minus_1: t.width_in_sbs_minus_1.iter().map(|&v| v as u16).collect(),
                height_in_sbs_minus_1: t.height_in_sbs_minus_1.iter().map(|&v| v as u16).collect(),
            },
            quantization: Quantization {
                using_qmatrix: q.using_qmatrix,
                diff_uv_delta: q.diff_uv_delta,
                base_q_idx: q.base_q_idx,
                delta_q_y_dc: q.delta_q_y_dc,
                delta_q_u_dc: q.delta_q_u_dc,
                delta_q_u_ac: q.delta_q_u_ac,
                delta_q_v_dc: q.delta_q_v_dc,
                delta_q_v_ac: q.delta_q_v_ac,
                qm_y: q.qm_y,
                qm_u: q.qm_u,
                qm_v: q.qm_v,
            },
            segmentation: Segmentation {
                feature_enabled: std::array::from_fn(|i| {
                    s.feature_enabled[i]
                        .iter()
                        .enumerate()
                        .fold(0, |m, (j, &on)| m | (u8::from(on) << j))
                }),
                feature_data: s.feature_data,
            },
            loop_filter: LoopFilter {
                loop_filter_delta_enabled: lf.delta_enabled,
                loop_filter_delta_update: lf.delta_update,
                loop_filter_level: lf.level,
                loop_filter_sharpness: lf.sharpness,
                update_ref_delta: lf.update_ref_delta,
                loop_filter_ref_deltas: lf.ref_deltas,
                update_mode_delta: lf.update_mode_delta,
                loop_filter_mode_deltas: lf.mode_deltas,
            },
            cdef: Cdef {
                cdef_damping_minus_3: c.damping_minus_3,
                cdef_bits: c.bits,
                cdef_y_pri_strength: c.y_pri_strength,
                cdef_y_sec_strength: c.y_sec_strength.map(coded_sec_strength),
                cdef_uv_pri_strength: c.uv_pri_strength,
                cdef_uv_sec_strength: c.uv_sec_strength.map(coded_sec_strength),
            },
            loop_restoration: LoopRestoration {
                frame_restoration_type: lr.frame_restoration_type.map(|t| t as u8),
                log2_restoration_size_minus_5: lr
                    .loop_restoration_size
                    .map(|size| size.checked_ilog2().map_or(0, |log2| log2 as u16 - 5)),
            },
            global_motion: GlobalMotion {
                gm_type: gm.gm_type.map(|t| t as u8),
                gm_params: gm.gm_params,
            },
            film_grain: FilmGrain::new(&h.film_grain),
        }
    }
}

/// The coded cdef_*_sec_strength for the spec's value (4 was coded as 3).
fn coded_sec_strength(strength: u8) -> u8 {
    if strength == 4 { 3 } else { strength }
}

/// StdVideoDecodeAV1ReferenceInfo: what a DPB slot's picture carries.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReferenceInfo {
    pub disable_frame_end_update_cdf: bool,
    pub segmentation_enabled: bool,
    /// StdVideoAV1FrameType value.
    pub frame_type: u8,
    /// Bit i: RefFrameSignBias[i] of that picture's own decode.
    pub ref_frame_sign_bias: u8,
    pub order_hint: u8,
    /// SavedOrderHints, indexed by reference name.
    pub saved_order_hints: [u8; NUM_REF_FRAMES],
}

impl ReferenceInfo {
    /// The information of a reference slot.
    pub fn from_slot(slot: &RefSlot) -> Self {
        Self {
            disable_frame_end_update_cdf: slot.disable_frame_end_update_cdf,
            segmentation_enabled: slot.segmentation_enabled,
            frame_type: slot.frame_type as u8,
            ref_frame_sign_bias: slot.ref_frame_sign_bias,
            order_hint: slot.order_hint,
            saved_order_hints: slot.saved_order_hints,
        }
    }

    /// The information of the picture a header decodes, for the setup
    /// (reconstructed) slot.
    pub fn from_header(h: &FrameHeader) -> Self {
        Self {
            disable_frame_end_update_cdf: h.disable_frame_end_update_cdf,
            segmentation_enabled: h.segmentation.enabled,
            frame_type: h.frame_type as u8,
            ref_frame_sign_bias: h.ref_frame_sign_bias,
            order_hint: h.order_hint,
            saved_order_hints: h.order_hints,
        }
    }
}
