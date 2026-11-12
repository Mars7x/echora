//! Reference bookkeeping: the eight reference slots and the decoder state
//! that persists between frame headers (AV1 spec sections 7.20 and 7.21).

use crate::frame::{
    FilmGrainParams, FrameHeader, FrameType, NUM_REF_FRAMES, SEG_LVL_MAX, default_gm_params,
};

/// A decoded picture, unique within a [`crate::Parser`]'s lifetime.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PictureId(pub u64);

/// Loop filter deltas default to these after setup_past_independence().
pub const DEFAULT_LOOP_FILTER_REF_DELTAS: [i8; NUM_REF_FRAMES] = [1, 0, 0, 0, -1, 0, -1, -1];

/// What the spec saves for one reference slot, as far as parsing later
/// headers and describing references to Vulkan needs it. CDFs, motion
/// vectors and segment ids live in the decoder (the GPU), not here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RefSlot {
    /// RefValid.
    pub valid: bool,
    pub picture: Option<PictureId>,
    /// RefFrameId.
    pub frame_id: u32,
    pub upscaled_width: u32,
    pub frame_width: u32,
    pub frame_height: u32,
    pub render_width: u32,
    pub render_height: u32,
    pub mi_cols: u32,
    pub mi_rows: u32,
    /// RefFrameType.
    pub frame_type: FrameType,
    pub subsampling_x: bool,
    pub subsampling_y: bool,
    pub bit_depth: u8,
    /// RefOrderHint.
    pub order_hint: u8,
    /// SavedOrderHints, indexed by reference name (INTRA_FRAME..ALTREF_FRAME).
    pub saved_order_hints: [u8; NUM_REF_FRAMES],
    /// RefFrameSignBias of the frame when it was decoded, bit i for
    /// reference name i (Vulkan's StdVideoDecodeAV1ReferenceInfo).
    pub ref_frame_sign_bias: u8,
    /// SavedGmParams.
    pub gm_params: [[i32; 6]; NUM_REF_FRAMES],
    /// SavedLoopFilterRefDeltas and SavedLoopFilterModeDeltas.
    pub loop_filter_ref_deltas: [i8; NUM_REF_FRAMES],
    pub loop_filter_mode_deltas: [i8; 2],
    /// SavedFeatureEnabled and SavedFeatureData.
    pub feature_enabled: [[bool; SEG_LVL_MAX]; 8],
    pub feature_data: [[i16; SEG_LVL_MAX]; 8],
    /// Saved film grain parameters (load_grain_params).
    pub film_grain: FilmGrainParams,
    pub showable_frame: bool,
    pub show_frame: bool,
    /// Kept for Vulkan's StdVideoDecodeAV1ReferenceInfo flags.
    pub segmentation_enabled: bool,
    pub disable_frame_end_update_cdf: bool,
}

impl Default for RefSlot {
    fn default() -> Self {
        Self {
            valid: false,
            picture: None,
            frame_id: 0,
            upscaled_width: 0,
            frame_width: 0,
            frame_height: 0,
            render_width: 0,
            render_height: 0,
            mi_cols: 0,
            mi_rows: 0,
            frame_type: FrameType::Key,
            subsampling_x: false,
            subsampling_y: false,
            bit_depth: 0,
            order_hint: 0,
            saved_order_hints: [0; NUM_REF_FRAMES],
            ref_frame_sign_bias: 0,
            gm_params: default_gm_params(),
            loop_filter_ref_deltas: DEFAULT_LOOP_FILTER_REF_DELTAS,
            loop_filter_mode_deltas: [0; 2],
            feature_enabled: [[false; SEG_LVL_MAX]; 8],
            feature_data: [[0; SEG_LVL_MAX]; 8],
            film_grain: FilmGrainParams::default(),
            showable_frame: false,
            show_frame: false,
            segmentation_enabled: false,
            disable_frame_end_update_cdf: false,
        }
    }
}

/// The state frame header parsing reads and writes: the reference slots
/// and the few variables the spec keeps from one frame to the next.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RefState {
    pub slots: [RefSlot; NUM_REF_FRAMES],
    /// OrderHints[], indexed by reference name. Inter frames set it;
    /// a shown key frame clears it; other intra frames keep it.
    pub order_hints: [u8; NUM_REF_FRAMES],
    /// RefFrameSignBias[] as a bit mask, kept the same way.
    pub ref_frame_sign_bias: u8,
    /// current_frame_id of the previous frame header.
    pub current_frame_id: u32,
}

impl RefState {
    /// The reference frame update process (section 7.20) after a frame
    /// that was decoded as `picture`.
    pub fn update(&mut self, h: &FrameHeader, seq_color: (bool, bool, u8), picture: PictureId) {
        let (subsampling_x, subsampling_y, bit_depth) = seq_color;
        for (i, slot) in self.slots.iter_mut().enumerate().take(NUM_REF_FRAMES) {
            if h.refresh_frame_flags & (1 << i) == 0 {
                continue;
            }
            *slot = RefSlot {
                valid: true,
                picture: Some(picture),
                frame_id: h.current_frame_id,
                upscaled_width: h.size.upscaled_width,
                frame_width: h.size.frame_width,
                frame_height: h.size.frame_height,
                render_width: h.size.render_width,
                render_height: h.size.render_height,
                mi_cols: h.size.mi_cols,
                mi_rows: h.size.mi_rows,
                frame_type: h.frame_type,
                subsampling_x,
                subsampling_y,
                bit_depth,
                order_hint: h.order_hint,
                saved_order_hints: h.order_hints,
                ref_frame_sign_bias: h.ref_frame_sign_bias,
                gm_params: h.global_motion.gm_params,
                loop_filter_ref_deltas: h.loop_filter.ref_deltas,
                loop_filter_mode_deltas: h.loop_filter.mode_deltas,
                feature_enabled: h.segmentation.feature_enabled,
                feature_data: h.segmentation.feature_data,
                film_grain: h.film_grain.clone(),
                showable_frame: h.showable_frame,
                show_frame: h.show_frame,
                segmentation_enabled: h.segmentation.enabled,
                disable_frame_end_update_cdf: h.disable_frame_end_update_cdf,
            };
        }
    }

    /// show_existing_frame of a key frame: the reference frame loading
    /// process (section 7.21) followed by a refresh of every slot.
    pub fn show_existing_key_frame(&mut self, slot: usize) {
        let mut shown = self.slots[slot].clone();
        // The loaded frame is now shown, so it cannot be shown again.
        shown.showable_frame = false;
        self.current_frame_id = shown.frame_id;
        self.order_hints = shown.saved_order_hints;
        self.ref_frame_sign_bias = shown.ref_frame_sign_bias;
        self.slots = std::array::from_fn(|_| shown.clone());
    }

    /// The pictures the slots hold.
    pub fn pictures(&self) -> [Option<PictureId>; NUM_REF_FRAMES] {
        std::array::from_fn(|i| {
            if self.slots[i].valid {
                self.slots[i].picture
            } else {
                None
            }
        })
    }
}
