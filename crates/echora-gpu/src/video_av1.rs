//! AV1 movies on the video engine: `echora_av1`'s parsed pictures, turned
//! into the Vulkan Video standard structures and decoded by
//! `video::Av1Decoder`, with the DPB slots managed by picture.

use std::sync::Arc;

use ash::vk::native as std_video;
use echora_av1::{DecodePicture, Output, Parser, PictureId, video};

use crate::video::{Av1Decoder, Av1Frame, SLOTS, VideoDevice};

/// All-zero plain C data (no field of these structures needs another
/// value to be valid; pointers become null).
fn zeroed<T: Copy>() -> T {
    // SAFETY: only instantiated with the Std video structures below, which
    // are integers, fixed arrays, bindgen bitfield units and raw pointers.
    unsafe { std::mem::zeroed() }
}

/// Calls `f` with `sequence` as StdVideoAV1SequenceHeader (its pointers
/// valid during the call).
fn with_sequence<R>(
    sequence: &video::SequenceHeader,
    f: impl FnOnce(&std_video::StdVideoAV1SequenceHeader) -> R,
) -> R {
    let c = &sequence.color_config;
    let mut color: std_video::StdVideoAV1ColorConfig = zeroed();
    color.flags.set_mono_chrome(c.mono_chrome.into());
    color.flags.set_color_range(c.color_range.into());
    color
        .flags
        .set_separate_uv_delta_q(c.separate_uv_delta_q.into());
    color
        .flags
        .set_color_description_present_flag(c.color_description_present_flag.into());
    color.BitDepth = c.bit_depth;
    color.subsampling_x = c.subsampling_x;
    color.subsampling_y = c.subsampling_y;
    color.color_primaries = c.color_primaries.into();
    color.transfer_characteristics = c.transfer_characteristics.into();
    color.matrix_coefficients = c.matrix_coefficients.into();
    color.chroma_sample_position = c.chroma_sample_position.into();
    let mut timing: std_video::StdVideoAV1TimingInfo = zeroed();
    if let Some(t) = &sequence.timing_info {
        timing
            .flags
            .set_equal_picture_interval(t.equal_picture_interval.into());
        timing.num_units_in_display_tick = t.num_units_in_display_tick;
        timing.time_scale = t.time_scale;
        timing.num_ticks_per_picture_minus_1 = t.num_ticks_per_picture_minus_1;
    }
    let s = sequence;
    let mut header: std_video::StdVideoAV1SequenceHeader = zeroed();
    let flags = &mut header.flags;
    flags.set_still_picture(s.still_picture.into());
    flags.set_reduced_still_picture_header(s.reduced_still_picture_header.into());
    flags.set_use_128x128_superblock(s.use_128x128_superblock.into());
    flags.set_enable_filter_intra(s.enable_filter_intra.into());
    flags.set_enable_intra_edge_filter(s.enable_intra_edge_filter.into());
    flags.set_enable_interintra_compound(s.enable_interintra_compound.into());
    flags.set_enable_masked_compound(s.enable_masked_compound.into());
    flags.set_enable_warped_motion(s.enable_warped_motion.into());
    flags.set_enable_dual_filter(s.enable_dual_filter.into());
    flags.set_enable_order_hint(s.enable_order_hint.into());
    flags.set_enable_jnt_comp(s.enable_jnt_comp.into());
    flags.set_enable_ref_frame_mvs(s.enable_ref_frame_mvs.into());
    flags.set_frame_id_numbers_present_flag(s.frame_id_numbers_present_flag.into());
    flags.set_enable_superres(s.enable_superres.into());
    flags.set_enable_cdef(s.enable_cdef.into());
    flags.set_enable_restoration(s.enable_restoration.into());
    flags.set_film_grain_params_present(s.film_grain_params_present.into());
    flags.set_timing_info_present_flag(s.timing_info_present_flag.into());
    flags.set_initial_display_delay_present_flag(s.initial_display_delay_present_flag.into());
    header.seq_profile = s.seq_profile.into();
    header.frame_width_bits_minus_1 = s.frame_width_bits_minus_1;
    header.frame_height_bits_minus_1 = s.frame_height_bits_minus_1;
    header.max_frame_width_minus_1 = s.max_frame_width_minus_1;
    header.max_frame_height_minus_1 = s.max_frame_height_minus_1;
    header.delta_frame_id_length_minus_2 = s.delta_frame_id_length_minus_2;
    header.additional_frame_id_length_minus_1 = s.additional_frame_id_length_minus_1;
    header.order_hint_bits_minus_1 = s.order_hint_bits_minus_1;
    header.seq_force_integer_mv = s.seq_force_integer_mv;
    header.seq_force_screen_content_tools = s.seq_force_screen_content_tools;
    header.pColorConfig = &color;
    header.pTimingInfo = if s.timing_info.is_some() {
        &timing
    } else {
        std::ptr::null()
    };
    f(&header)
}

fn reference_info(info: &video::ReferenceInfo) -> std_video::StdVideoDecodeAV1ReferenceInfo {
    let mut out: std_video::StdVideoDecodeAV1ReferenceInfo = zeroed();
    out.flags
        .set_disable_frame_end_update_cdf(info.disable_frame_end_update_cdf.into());
    out.flags
        .set_segmentation_enabled(info.segmentation_enabled.into());
    out.frame_type = info.frame_type;
    out.RefFrameSignBias = info.ref_frame_sign_bias;
    out.OrderHint = info.order_hint;
    out.SavedOrderHints = info.saved_order_hints;
    out
}

/// Calls `f` with `picture` as StdVideoDecodeAV1PictureInfo (its pointers
/// valid during the call). Film grain is never passed: the session is
/// created without grain support.
fn with_picture<R>(
    picture: &video::PictureInfo,
    f: impl FnOnce(&std_video::StdVideoDecodeAV1PictureInfo) -> R,
) -> R {
    let p = picture;
    let t = &p.tile_info;
    let mut tile: std_video::StdVideoAV1TileInfo = zeroed();
    tile.flags
        .set_uniform_tile_spacing_flag(t.uniform_tile_spacing_flag.into());
    tile.TileCols = t.tile_cols;
    tile.TileRows = t.tile_rows;
    tile.context_update_tile_id = t.context_update_tile_id;
    tile.tile_size_bytes_minus_1 = t.tile_size_bytes_minus_1;
    tile.pMiColStarts = t.mi_col_starts.as_ptr();
    tile.pMiRowStarts = t.mi_row_starts.as_ptr();
    tile.pWidthInSbsMinus1 = t.width_in_sbs_minus_1.as_ptr();
    tile.pHeightInSbsMinus1 = t.height_in_sbs_minus_1.as_ptr();

    let q = &p.quantization;
    let mut quantization: std_video::StdVideoAV1Quantization = zeroed();
    quantization.flags.set_using_qmatrix(q.using_qmatrix.into());
    quantization.flags.set_diff_uv_delta(q.diff_uv_delta.into());
    quantization.base_q_idx = q.base_q_idx;
    quantization.DeltaQYDc = q.delta_q_y_dc;
    quantization.DeltaQUDc = q.delta_q_u_dc;
    quantization.DeltaQUAc = q.delta_q_u_ac;
    quantization.DeltaQVDc = q.delta_q_v_dc;
    quantization.DeltaQVAc = q.delta_q_v_ac;
    quantization.qm_y = q.qm_y;
    quantization.qm_u = q.qm_u;
    quantization.qm_v = q.qm_v;

    let mut segmentation: std_video::StdVideoAV1Segmentation = zeroed();
    segmentation.FeatureEnabled = p.segmentation.feature_enabled;
    segmentation.FeatureData = p.segmentation.feature_data;

    let l = &p.loop_filter;
    let mut loop_filter: std_video::StdVideoAV1LoopFilter = zeroed();
    loop_filter
        .flags
        .set_loop_filter_delta_enabled(l.loop_filter_delta_enabled.into());
    loop_filter
        .flags
        .set_loop_filter_delta_update(l.loop_filter_delta_update.into());
    loop_filter.loop_filter_level = l.loop_filter_level;
    loop_filter.loop_filter_sharpness = l.loop_filter_sharpness;
    loop_filter.update_ref_delta = l.update_ref_delta;
    loop_filter.loop_filter_ref_deltas = l.loop_filter_ref_deltas;
    loop_filter.update_mode_delta = l.update_mode_delta;
    loop_filter.loop_filter_mode_deltas = l.loop_filter_mode_deltas;

    let c = &p.cdef;
    let mut cdef: std_video::StdVideoAV1CDEF = zeroed();
    cdef.cdef_damping_minus_3 = c.cdef_damping_minus_3;
    cdef.cdef_bits = c.cdef_bits;
    cdef.cdef_y_pri_strength = c.cdef_y_pri_strength;
    cdef.cdef_y_sec_strength = c.cdef_y_sec_strength;
    cdef.cdef_uv_pri_strength = c.cdef_uv_pri_strength;
    cdef.cdef_uv_sec_strength = c.cdef_uv_sec_strength;

    let mut restoration: std_video::StdVideoAV1LoopRestoration = zeroed();
    restoration.FrameRestorationType = p
        .loop_restoration
        .frame_restoration_type
        .map(std_video::StdVideoAV1FrameRestorationType::from);
    // Vulkan takes each plane's unit size as log2(size) - 5 (the parser
    // gives samples; 0 when the plane is not restored).
    restoration.LoopRestorationSize = p.loop_restoration.loop_restoration_size.map(|size| {
        size.checked_ilog2()
            .map_or(0, |log2| log2.saturating_sub(5) as u16)
    });

    let mut motion: std_video::StdVideoAV1GlobalMotion = zeroed();
    motion.GmType = p.global_motion.gm_type;
    motion.gm_params = p.global_motion.gm_params;

    let mut info: std_video::StdVideoDecodeAV1PictureInfo = zeroed();
    let flags = &mut info.flags;
    flags.set_error_resilient_mode(p.error_resilient_mode.into());
    flags.set_disable_cdf_update(p.disable_cdf_update.into());
    flags.set_use_superres(p.use_superres.into());
    flags.set_render_and_frame_size_different(p.render_and_frame_size_different.into());
    flags.set_allow_screen_content_tools(p.allow_screen_content_tools.into());
    flags.set_is_filter_switchable(p.is_filter_switchable.into());
    flags.set_force_integer_mv(p.force_integer_mv.into());
    flags.set_frame_size_override_flag(p.frame_size_override_flag.into());
    flags.set_buffer_removal_time_present_flag(p.buffer_removal_time_present_flag.into());
    flags.set_allow_intrabc(p.allow_intrabc.into());
    flags.set_frame_refs_short_signaling(p.frame_refs_short_signaling.into());
    flags.set_allow_high_precision_mv(p.allow_high_precision_mv.into());
    flags.set_is_motion_mode_switchable(p.is_motion_mode_switchable.into());
    flags.set_use_ref_frame_mvs(p.use_ref_frame_mvs.into());
    flags.set_disable_frame_end_update_cdf(p.disable_frame_end_update_cdf.into());
    flags.set_allow_warped_motion(p.allow_warped_motion.into());
    flags.set_reduced_tx_set(p.reduced_tx_set.into());
    flags.set_reference_select(p.reference_select.into());
    flags.set_skip_mode_present(p.skip_mode_present.into());
    flags.set_delta_q_present(p.delta_q_present.into());
    flags.set_delta_lf_present(p.delta_lf_present.into());
    flags.set_delta_lf_multi(p.delta_lf_multi.into());
    flags.set_segmentation_enabled(p.segmentation_enabled.into());
    flags.set_segmentation_update_map(p.segmentation_update_map.into());
    flags.set_segmentation_temporal_update(p.segmentation_temporal_update.into());
    flags.set_segmentation_update_data(p.segmentation_update_data.into());
    flags.set_UsesLr(p.uses_lr.into());
    flags.set_usesChromaLr(p.uses_chroma_lr.into());
    info.frame_type = p.frame_type.into();
    info.current_frame_id = p.current_frame_id;
    info.OrderHint = p.order_hint;
    info.primary_ref_frame = p.primary_ref_frame;
    info.refresh_frame_flags = p.refresh_frame_flags;
    info.interpolation_filter = p.interpolation_filter.into();
    info.TxMode = p.tx_mode.into();
    info.delta_q_res = p.delta_q_res;
    info.delta_lf_res = p.delta_lf_res;
    info.SkipModeFrame = p.skip_mode_frame;
    info.coded_denom = p.coded_denom;
    info.OrderHints = p.order_hints;
    info.expectedFrameId = p.expected_frame_id;
    info.pTileInfo = &tile;
    info.pQuantization = &quantization;
    info.pSegmentation = &segmentation;
    info.pLoopFilter = &loop_filter;
    info.pCDEF = &cdef;
    info.pLoopRestoration = &restoration;
    info.pGlobalMotion = &motion;
    f(&info)
}

/// A picture to present: the DPB slot holding it and its size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shown {
    pub slot: u32,
    pub width: u32,
    pub height: u32,
}

/// One AV1 stream decoded on the video engine.
pub struct MovieDecoder {
    video: Arc<VideoDevice>,
    parser: Parser,
    decoder: Option<Av1Decoder>,
    /// The sequence the session parameters were made from.
    generation: Option<u64>,
    /// The picture each DPB slot holds.
    dpb: [Option<PictureId>; SLOTS as usize],
}

impl MovieDecoder {
    pub fn new(video: Arc<VideoDevice>) -> Self {
        Self {
            video,
            parser: Parser::new(),
            decoder: None,
            generation: None,
            dpb: [None; SLOTS as usize],
        }
    }

    /// Decodes one temporal unit (low-overhead OBUs) and returns the
    /// pictures it shows, in order. Errors for what the session does not
    /// support (film grain, superres, a size change), so the caller can
    /// fall back to the CPU.
    pub fn decode(&mut self, unit: &[u8]) -> Result<Vec<Shown>, String> {
        let outputs = self
            .parser
            .parse_temporal_unit(unit)
            .map_err(|error| format!("AV1 parse: {error:?}"))?;
        let mut shown = Vec::new();
        for output in outputs {
            match output {
                Output::Decode(picture) => {
                    let showing: Vec<u32> = shown.iter().map(|shown: &Shown| shown.slot).collect();
                    let slot = self.decode_picture(unit, &picture, &showing)?;
                    if picture.show_frame {
                        shown.push(Shown {
                            slot,
                            width: picture.upscaled_width,
                            height: picture.frame_height,
                        });
                    }
                }
                Output::ShowExisting(show) => {
                    if show.film_grain.is_some() {
                        return Err("AV1 film grain is not supported on the GPU".into());
                    }
                    let slot = self
                        .slot_of(show.id)
                        .ok_or("AV1 shows a picture no DPB slot holds")?;
                    let (width, height) = self
                        .decoder
                        .as_ref()
                        .and_then(|decoder| decoder.picture_size(slot))
                        .ok_or("AV1 shows a picture never decoded")?;
                    shown.push(Shown {
                        slot,
                        width,
                        height,
                    });
                }
                Output::Metadata(_) => {}
            }
        }
        Ok(shown)
    }

    /// Copies a shown picture into `out` as NV12 (`video::nv12_size`).
    pub fn read_nv12(&mut self, shown: Shown, pitch: u32, out: &mut [u8]) -> Result<(), String> {
        self.decoder
            .as_mut()
            .ok_or("no AV1 session")?
            .read_nv12(shown.slot, pitch, out)
    }

    fn slot_of(&self, id: PictureId) -> Option<u32> {
        self.dpb
            .iter()
            .position(|held| *held == Some(id))
            .map(|slot| slot as u32)
    }

    /// Decodes `picture` into a DPB slot that neither an AV1 slot nor
    /// `showing` (pictures this unit shows, not yet read) still needs.
    fn decode_picture(
        &mut self,
        unit: &[u8],
        picture: &DecodePicture,
        showing: &[u32],
    ) -> Result<u32, String> {
        if picture.std_picture.apply_grain {
            return Err("AV1 film grain is not supported on the GPU".into());
        }
        if picture.use_superres {
            return Err("AV1 superres is not supported on the GPU".into());
        }
        if self.decoder.is_none() {
            let sequence = &picture.std_sequence;
            let width = u32::from(sequence.max_frame_width_minus_1) + 1;
            let height = u32::from(sequence.max_frame_height_minus_1) + 1;
            self.decoder = Some(Av1Decoder::new(self.video.clone(), width, height)?);
        }
        let decoder = self.decoder.as_mut().ok_or("no AV1 session")?;
        if self.generation != Some(picture.sequence_generation) {
            with_sequence(&picture.std_sequence, |header| decoder.set_sequence(header))?;
            self.generation = Some(picture.sequence_generation);
        }
        // The DPB slot of each AV1 reference slot the picture reads.
        let mut names = [-1i32; 7];
        let mut references: Vec<(u32, std_video::StdVideoDecodeAV1ReferenceInfo)> = Vec::new();
        for (name, &av1_slot) in picture.reference_name_slot_indices.iter().enumerate() {
            if av1_slot < 0 {
                continue;
            }
            let av1_slot = av1_slot as usize;
            let id = picture.slot_pictures[av1_slot]
                .ok_or_else(|| format!("AV1 reference slot {av1_slot} is empty"))?;
            let slot = self
                .dpb
                .iter()
                .position(|held| *held == Some(id))
                .ok_or("AV1 reference was never decoded")? as u32;
            names[name] = slot as i32;
            if !references.iter().any(|(held, _)| *held == slot) {
                let info = picture.reference_infos[av1_slot]
                    .as_ref()
                    .ok_or_else(|| format!("AV1 reference slot {av1_slot} has no information"))?;
                references.push((slot, reference_info(info)));
            }
        }
        let live: Vec<PictureId> = picture.slot_pictures.iter().flatten().copied().collect();
        let target = (0..SLOTS)
            .filter(|slot| !showing.contains(slot))
            .find(|&slot| self.dpb[slot as usize].is_none_or(|held| !live.contains(&held)))
            .ok_or("no free AV1 DPB slot")?;
        let offsets = picture.tile_offsets();
        let sizes = picture.tile_sizes();
        let slot_info = reference_info(&picture.setup_reference_info);
        with_picture(&picture.std_picture, |info| {
            decoder.decode(&Av1Frame {
                bitstream: unit,
                frame_header_offset: picture.frame_header_offset,
                tile_offsets: &offsets,
                tile_sizes: &sizes,
                width: picture.frame_width,
                height: picture.frame_height,
                picture: info,
                reference_name_slot_indices: names,
                references: &references,
                slot: target,
                slot_info,
            })
        })?;
        self.dpb[target as usize] = Some(picture.id);
        Ok(target)
    }
}
