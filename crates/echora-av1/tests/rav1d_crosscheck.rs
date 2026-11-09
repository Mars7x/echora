//! Cross-checks the parsed headers against rav1d, which decodes the same
//! streams: every picture rav1d outputs (invisible frames included) must
//! match the next picture the parser describes, field by field.
//!
//! rav1d's interface is dav1d's C API, so calling it takes unsafe code;
//! this test is the only place in the crate that does.
#![allow(unsafe_code)]

mod common;

use std::collections::HashMap;
use std::mem::MaybeUninit;
use std::ptr::NonNull;
use std::sync::Arc;

use echora_av1::frame::{FrameHeader, FrameType, RestorationType};
use echora_av1::{Output, Parser};
use rav1d::include::dav1d::data::Dav1dData;
use rav1d::include::dav1d::dav1d::{DAV1D_INLOOPFILTER_NONE, Dav1dContext, Dav1dSettings};
use rav1d::include::dav1d::headers::Dav1dFrameHeader;
use rav1d::include::dav1d::picture::Dav1dPicture;
use rav1d::src::lib::{
    dav1d_close, dav1d_data_create, dav1d_data_unref, dav1d_default_settings, dav1d_get_picture,
    dav1d_open, dav1d_picture_unref, dav1d_send_data,
};

const AGAIN: i32 = -11;

/// rav1d set up to output every frame, shown or not, one at a time.
struct Rav1d {
    context: Option<Dav1dContext>,
}

impl Rav1d {
    fn new() -> Self {
        let mut settings = MaybeUninit::<Dav1dSettings>::zeroed();
        // SAFETY: dav1d_default_settings fills the whole struct.
        let mut settings = unsafe {
            dav1d_default_settings(NonNull::from(&mut settings).cast());
            settings.assume_init()
        };
        settings.n_threads = std::thread::available_parallelism().map_or(1, |n| n.get() as i32);
        settings.max_frame_delay = 1;
        settings.apply_grain = 0;
        settings.all_layers = 1;
        settings.output_invisible_frames = 1;
        // Only the headers are compared, so skip the in-loop filters.
        settings.inloop_filters = DAV1D_INLOOPFILTER_NONE;
        let mut context = None;
        // SAFETY: both pointers are to live locals.
        let result = unsafe {
            dav1d_open(
                Some(NonNull::from(&mut context)),
                Some(NonNull::from(&mut settings)),
            )
        };
        assert_eq!(result.0, 0, "dav1d_open");
        Self { context }
    }

    /// Decodes one temporal unit; calls `f` with the header of every
    /// picture that comes out.
    fn decode(&mut self, bytes: &[u8], f: &mut impl FnMut(&Dav1dFrameHeader)) {
        let mut data = Dav1dData::default();
        // SAFETY: dav1d_data_create returns a buffer of bytes.len() bytes.
        unsafe {
            let buffer = dav1d_data_create(Some(NonNull::from(&mut data)), bytes.len());
            assert!(!buffer.is_null());
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer, bytes.len());
        }
        loop {
            // SAFETY: the context is open and data owns its buffer.
            let sent = unsafe { dav1d_send_data(self.context, Some(NonNull::from(&mut data))) };
            assert!(
                sent.0 == 0 || sent.0 == AGAIN,
                "dav1d_send_data: {}",
                sent.0
            );
            self.drain(f);
            if data.sz == 0 {
                break;
            }
        }
        if data.sz != 0 {
            // SAFETY: data still owns its buffer.
            unsafe { dav1d_data_unref(Some(NonNull::from(&mut data))) };
        }
    }

    fn drain(&mut self, f: &mut impl FnMut(&Dav1dFrameHeader)) {
        loop {
            let mut picture = Dav1dPicture::default();
            // SAFETY: the context is open and picture is a live local.
            let result =
                unsafe { dav1d_get_picture(self.context, Some(NonNull::from(&mut picture))) };
            if result.0 == AGAIN {
                return;
            }
            assert_eq!(result.0, 0, "dav1d_get_picture");
            let header = picture.frame_hdr.expect("a picture has a frame header");
            // SAFETY: the header lives as long as the picture reference.
            f(unsafe { header.as_ref() });
            // SAFETY: picture holds references from dav1d_get_picture.
            unsafe { dav1d_picture_unref(Some(NonNull::from(&mut picture))) };
        }
    }
}

impl Drop for Rav1d {
    fn drop(&mut self) {
        // SAFETY: the context was opened in new.
        unsafe { dav1d_close(Some(NonNull::from(&mut self.context))) };
    }
}

/// Compares one parsed header with rav1d's. Returns the mismatches.
fn compare(h: &FrameHeader, d: &Dav1dFrameHeader, num_planes: usize) -> Vec<String> {
    let mut diffs = Vec::new();
    macro_rules! check {
        ($name:expr, $ours:expr, $theirs:expr) => {{
            let (ours, theirs) = ($ours, $theirs);
            if ours != theirs {
                diffs.push(format!("{}: parsed {:?}, rav1d {:?}", $name, ours, theirs));
            }
        }};
    }
    check!("frame_type", h.frame_type as u32, d.frame_type);
    check!("frame width", h.size.frame_width as i32, d.width[0]);
    check!("upscaled width", h.size.upscaled_width as i32, d.width[1]);
    check!("frame height", h.size.frame_height as i32, d.height);
    check!("render width", h.size.render_width as i32, d.render_width);
    check!(
        "render height",
        h.size.render_height as i32,
        d.render_height
    );
    check!("order hint", h.order_hint, d.frame_offset);
    check!("show_frame", u8::from(h.show_frame), d.show_frame);
    check!(
        "showable_frame",
        u8::from(h.showable_frame),
        d.showable_frame
    );
    check!(
        "refresh_frame_flags",
        h.refresh_frame_flags,
        d.refresh_frame_flags
    );
    check!(
        "error_resilient_mode",
        u8::from(h.error_resilient_mode),
        d.error_resilient_mode
    );
    check!(
        "disable_cdf_update",
        u8::from(h.disable_cdf_update),
        d.disable_cdf_update
    );
    check!(
        "allow_screen_content_tools",
        u8::from(h.allow_screen_content_tools),
        d.allow_screen_content_tools
    );
    check!(
        "force_integer_mv",
        u8::from(h.force_integer_mv),
        d.force_integer_mv
    );
    check!(
        "frame_size_override_flag",
        u8::from(h.frame_size_override_flag),
        d.frame_size_override
    );
    check!(
        "primary_ref_frame",
        h.primary_ref_frame,
        d.primary_ref_frame
    );
    check!("frame id", h.current_frame_id, d.frame_id);
    check!(
        "use_superres",
        u8::from(h.size.use_superres),
        d.super_res.enabled
    );
    if h.size.use_superres {
        check!(
            "superres denominator",
            h.size.superres_denom as u8,
            d.super_res.width_scale_denominator
        );
    }
    check!("allow_intrabc", u8::from(h.allow_intrabc), d.allow_intrabc);
    if !h.frame_is_intra() {
        check!(
            "frame_refs_short_signaling",
            u8::from(h.frame_refs_short_signaling),
            d.frame_ref_short_signaling
        );
        check!("ref_frame_idx", h.ref_frame_idx.map(|i| i as i8), d.refidx);
        check!(
            "allow_high_precision_mv",
            u8::from(h.allow_high_precision_mv),
            d.hp
        );
        check!(
            "interpolation_filter",
            h.interpolation_filter as u32,
            d.subpel_filter_mode as u32
        );
        check!(
            "is_motion_mode_switchable",
            u8::from(h.is_motion_mode_switchable),
            d.switchable_motion_mode
        );
        check!(
            "use_ref_frame_mvs",
            u8::from(h.use_ref_frame_mvs),
            d.use_ref_frame_mvs
        );
    }
    check!(
        "disable_frame_end_update_cdf",
        u8::from(!h.disable_frame_end_update_cdf),
        d.refresh_context
    );

    // Tiles.
    let t = &h.tile_info;
    let dt = &d.tiling;
    check!(
        "uniform_tile_spacing_flag",
        u8::from(t.uniform_tile_spacing_flag),
        dt.uniform
    );
    check!("tile cols", t.tile_cols as u8, dt.cols);
    check!("tile rows", t.tile_rows as u8, dt.rows);
    check!("tile cols log2", t.tile_cols_log2 as u8, dt.log2_cols);
    check!("tile rows log2", t.tile_rows_log2 as u8, dt.log2_rows);
    check!(
        "context_update_tile_id",
        t.context_update_tile_id as u16,
        dt.update
    );
    if t.tile_cols_log2 + t.tile_rows_log2 > 0 {
        check!("tile size bytes", t.tile_size_bytes_minus_1 + 1, dt.n_bytes);
    }
    let mut col_sb = 0u16;
    for (i, w) in t.width_in_sbs_minus_1.iter().enumerate() {
        check!(format!("tile col {i} start"), col_sb, dt.col_start_sb[i]);
        col_sb += *w as u16 + 1;
    }
    check!(
        "tile cols end",
        col_sb,
        dt.col_start_sb[t.tile_cols as usize]
    );
    let mut row_sb = 0u16;
    for (i, hgt) in t.height_in_sbs_minus_1.iter().enumerate() {
        check!(format!("tile row {i} start"), row_sb, dt.row_start_sb[i]);
        row_sb += *hgt as u16 + 1;
    }
    check!(
        "tile rows end",
        row_sb,
        dt.row_start_sb[t.tile_rows as usize]
    );

    // Quantization.
    let q = &h.quantization;
    let dq = &d.quant;
    check!("base_q_idx", q.base_q_idx, dq.yac);
    check!("DeltaQYDc", q.delta_q_y_dc, dq.ydc_delta);
    check!("DeltaQUDc", q.delta_q_u_dc, dq.udc_delta);
    check!("DeltaQUAc", q.delta_q_u_ac, dq.uac_delta);
    check!("DeltaQVDc", q.delta_q_v_dc, dq.vdc_delta);
    check!("DeltaQVAc", q.delta_q_v_ac, dq.vac_delta);
    check!("using_qmatrix", u8::from(q.using_qmatrix), dq.qm);
    if q.using_qmatrix {
        check!(
            "qm levels",
            (q.qm_y, q.qm_u, q.qm_v),
            (dq.qm_y, dq.qm_u, dq.qm_v)
        );
    }

    // Segmentation.
    let s = &h.segmentation;
    let ds = &d.segmentation;
    check!("segmentation_enabled", u8::from(s.enabled), ds.enabled);
    if s.enabled {
        check!(
            "segmentation_update_map",
            u8::from(s.update_map),
            ds.update_map
        );
        check!(
            "segmentation_temporal_update",
            u8::from(s.temporal_update),
            ds.temporal
        );
        check!(
            "segmentation_update_data",
            u8::from(s.update_data),
            ds.update_data
        );
        for i in 0..8 {
            let e = &s.feature_enabled[i];
            let v = &s.feature_data[i];
            let dd = &ds.seg_data.d[i];
            let ours = (
                if e[0] { v[0] } else { 0 },
                if e[1] { v[1] as i8 } else { 0 },
                if e[2] { v[2] as i8 } else { 0 },
                if e[3] { v[3] as i8 } else { 0 },
                if e[4] { v[4] as i8 } else { 0 },
                if e[5] { v[5] as i8 } else { -1 },
                u8::from(e[6]),
                u8::from(e[7]),
            );
            let theirs = (
                dd.delta_q,
                dd.delta_lf_y_v,
                dd.delta_lf_y_h,
                dd.delta_lf_u,
                dd.delta_lf_v,
                dd.r#ref,
                dd.skip,
                dd.globalmv,
            );
            check!(format!("segment {i} features"), ours, theirs);
        }
        check!(
            "SegIdPreSkip",
            u8::from(s.seg_id_pre_skip),
            ds.seg_data.preskip
        );
        let any = s.feature_enabled.iter().flatten().any(|&e| e);
        let last = if any { s.last_active_seg_id as i8 } else { -1 };
        check!("LastActiveSegId", last, ds.seg_data.last_active_segid);
    }
    for i in 0..8 {
        check!(
            format!("lossless {i}"),
            u8::from(h.lossless_array[i]),
            ds.lossless[i]
        );
        check!(format!("qidx {i}"), h.qindex(i), ds.qidx[i]);
    }
    check!("AllLossless", u8::from(h.all_lossless), d.all_lossless);

    // Delta q and delta lf.
    check!(
        "delta_q_present",
        u8::from(h.delta.delta_q_present),
        d.delta.q.present
    );
    check!("delta_q_res", h.delta.delta_q_res, d.delta.q.res_log2);
    check!(
        "delta_lf_present",
        u8::from(h.delta.delta_lf_present),
        d.delta.lf.present
    );
    check!("delta_lf_res", h.delta.delta_lf_res, d.delta.lf.res_log2);
    check!(
        "delta_lf_multi",
        u8::from(h.delta.delta_lf_multi),
        d.delta.lf.multi
    );

    // Loop filter.
    let lf = &h.loop_filter;
    let dl = &d.loopfilter;
    check!(
        "loop_filter_level y",
        [lf.level[0], lf.level[1]],
        dl.level_y
    );
    if num_planes > 1 {
        check!(
            "loop_filter_level uv",
            (lf.level[2], lf.level[3]),
            (dl.level_u, dl.level_v)
        );
    }
    if !h.coded_lossless && !h.allow_intrabc {
        check!("loop_filter_sharpness", lf.sharpness, dl.sharpness);
        check!(
            "loop_filter_delta_enabled",
            u8::from(lf.delta_enabled),
            dl.mode_ref_delta_enabled
        );
        check!(
            "loop_filter_delta_update",
            u8::from(lf.delta_update),
            dl.mode_ref_delta_update
        );
    }
    check!(
        "loop_filter_ref_deltas",
        lf.ref_deltas,
        dl.mode_ref_deltas.ref_delta
    );
    check!(
        "loop_filter_mode_deltas",
        lf.mode_deltas,
        dl.mode_ref_deltas.mode_delta
    );

    // CDEF: rav1d keeps the six coded bits, the secondary strength unadjusted.
    let c = &h.cdef;
    let dc = &d.cdef;
    // Where CDEF is not coded (lossless, intra block copy) the spec's
    // CdefDamping is 3 and rav1d leaves the field zero; it is unused.
    if !h.coded_lossless && !h.allow_intrabc {
        check!("cdef damping", c.damping_minus_3 + 3, dc.damping);
    }
    check!("cdef_bits", c.bits, dc.n_bits);
    let coded = |pri: u8, sec: u8| (pri << 2) | if sec == 4 { 3 } else { sec };
    for i in 0..(1usize << c.bits) {
        check!(
            format!("cdef y {i}"),
            coded(c.y_pri_strength[i], c.y_sec_strength[i]),
            dc.y_strength[i]
        );
        if num_planes > 1 {
            check!(
                format!("cdef uv {i}"),
                coded(c.uv_pri_strength[i], c.uv_sec_strength[i]),
                dc.uv_strength[i]
            );
        }
    }

    // Loop restoration: rav1d's types are in coded order, sizes in log2.
    let lr = &h.loop_restoration;
    let coded_type = |t: RestorationType| match t {
        RestorationType::None => 0u8,
        RestorationType::Switchable => 1,
        RestorationType::Wiener => 2,
        RestorationType::Sgrproj => 3,
    };
    for p in 0..num_planes {
        check!(
            format!("lr type {p}"),
            coded_type(lr.frame_restoration_type[p]),
            d.restoration.r#type[p]
        );
    }
    if lr.uses_lr {
        let log2 = |s: u16| s.trailing_zeros() as u8;
        check!(
            "lr unit size y",
            log2(lr.loop_restoration_size[0]),
            d.restoration.unit_size[0]
        );
        if lr.uses_chroma_lr {
            check!(
                "lr unit size uv",
                log2(lr.loop_restoration_size[1]),
                d.restoration.unit_size[1]
            );
        }
    }

    check!("TxMode", h.tx_mode as u32, d.txfm_mode);
    check!(
        "reference_select",
        u8::from(h.reference_select),
        d.switchable_comp_refs
    );
    check!(
        "skip_mode_present",
        u8::from(h.skip_mode_present),
        d.skip_mode_enabled
    );
    if h.skip_mode_present {
        check!(
            "SkipModeFrame",
            h.skip_mode_frame.map(|r| r as i8 - 1),
            d.skip_mode_refs
        );
    }
    check!(
        "allow_warped_motion",
        u8::from(h.allow_warped_motion),
        d.warp_motion
    );
    check!(
        "reduced_tx_set",
        u8::from(h.reduced_tx_set),
        d.reduced_txtp_set
    );

    // Global motion, LAST..ALTREF.
    if !h.frame_is_intra() {
        for i in 0..7 {
            let g = &d.gmv[i];
            check!(
                format!("GmType {i}"),
                h.global_motion.gm_type[i + 1] as u32,
                g.r#type
            );
            check!(
                format!("gm_params {i}"),
                h.global_motion.gm_params[i + 1],
                g.matrix
            );
        }
    }

    // Film grain.
    let g = &h.film_grain;
    let dg = &d.film_grain;
    check!("apply_grain", u8::from(g.apply_grain), dg.present);
    if g.apply_grain {
        let dd = &dg.data;
        check!("update_grain", u8::from(g.update_grain), dg.update);
        check!("grain_seed", u32::from(g.grain_seed), dd.seed);
        check!("num_y_points", i32::from(g.num_y_points), dd.num_y_points);
        for i in 0..usize::from(g.num_y_points) {
            check!(
                format!("y point {i}"),
                [g.point_y_value[i], g.point_y_scaling[i]],
                dd.y_points[i]
            );
        }
        check!(
            "chroma_scaling_from_luma",
            i32::from(g.chroma_scaling_from_luma),
            dd.chroma_scaling_from_luma
        );
        check!(
            "num_cb_points",
            i32::from(g.num_cb_points),
            dd.num_uv_points[0]
        );
        check!(
            "num_cr_points",
            i32::from(g.num_cr_points),
            dd.num_uv_points[1]
        );
        for i in 0..usize::from(g.num_cb_points) {
            check!(
                format!("cb point {i}"),
                [g.point_cb_value[i], g.point_cb_scaling[i]],
                dd.uv_points[0][i]
            );
        }
        for i in 0..usize::from(g.num_cr_points) {
            check!(
                format!("cr point {i}"),
                [g.point_cr_value[i], g.point_cr_scaling[i]],
                dd.uv_points[1][i]
            );
        }
        check!(
            "grain_scaling",
            i32::from(g.grain_scaling_minus_8) + 8,
            dd.scaling_shift
        );
        check!("ar_coeff_lag", i32::from(g.ar_coeff_lag), dd.ar_coeff_lag);
        let lag = usize::from(g.ar_coeff_lag);
        let num_pos_luma = 2 * lag * (lag + 1);
        if g.num_y_points > 0 {
            for i in 0..num_pos_luma {
                check!(
                    format!("ar y {i}"),
                    g.ar_coeffs_y_plus_128[i].wrapping_sub(128) as i8,
                    dd.ar_coeffs_y[i]
                );
            }
        }
        let num_pos_chroma = num_pos_luma + usize::from(g.num_y_points > 0);
        if g.chroma_scaling_from_luma || g.num_cb_points > 0 {
            for i in 0..num_pos_chroma {
                check!(
                    format!("ar cb {i}"),
                    g.ar_coeffs_cb_plus_128[i].wrapping_sub(128) as i8,
                    dd.ar_coeffs_uv[0][i]
                );
            }
        }
        if g.chroma_scaling_from_luma || g.num_cr_points > 0 {
            for i in 0..num_pos_chroma {
                check!(
                    format!("ar cr {i}"),
                    g.ar_coeffs_cr_plus_128[i].wrapping_sub(128) as i8,
                    dd.ar_coeffs_uv[1][i]
                );
            }
        }
        check!(
            "ar_coeff_shift",
            u64::from(g.ar_coeff_shift_minus_6) + 6,
            dd.ar_coeff_shift
        );
        check!(
            "grain_scale_shift",
            i32::from(g.grain_scale_shift),
            dd.grain_scale_shift
        );
        if g.num_cb_points > 0 {
            check!(
                "cb mult",
                (
                    i32::from(g.cb_mult) - 128,
                    i32::from(g.cb_luma_mult) - 128,
                    i32::from(g.cb_offset) - 256
                ),
                (dd.uv_mult[0], dd.uv_luma_mult[0], dd.uv_offset[0])
            );
        }
        if g.num_cr_points > 0 {
            check!(
                "cr mult",
                (
                    i32::from(g.cr_mult) - 128,
                    i32::from(g.cr_luma_mult) - 128,
                    i32::from(g.cr_offset) - 256
                ),
                (dd.uv_mult[1], dd.uv_luma_mult[1], dd.uv_offset[1])
            );
        }
        check!("overlap_flag", i32::from(g.overlap_flag), dd.overlap_flag);
        check!(
            "clip_to_restricted_range",
            i32::from(g.clip_to_restricted_range),
            dd.clip_to_restricted_range
        );
    }
    diffs
}

/// What the parser says the decoder outputs, in order: decoded pictures
/// (shown or not) and pictures shown again.
fn parsed_outputs(frames: &[&[u8]]) -> Vec<Vec<Arc<FrameHeader>>> {
    let mut parser = Parser::new();
    let mut by_id = HashMap::new();
    frames
        .iter()
        .enumerate()
        .map(|(n, tu)| {
            let outputs = parser
                .parse_temporal_unit(tu)
                .unwrap_or_else(|e| panic!("temporal unit {n}: {e}"));
            outputs
                .into_iter()
                .filter_map(|o| match o {
                    Output::Decode(p) => {
                        by_id.insert(p.id, p.header.clone());
                        Some(p.header.clone())
                    }
                    Output::ShowExisting(s) => Some(by_id[&s.id].clone()),
                    Output::Metadata(_) => None,
                })
                .collect()
        })
        .collect()
}

/// Unoptimised rav1d takes seconds per picture (its debug build checks
/// every buffer access), so a debug test run only decodes the smallest
/// streams with it; the parser still reads every stream to the end.
/// `cargo test --release -p echora-av1` cross-checks every picture.
fn decode_with_rav1d(width: u16, height: u16) -> bool {
    !cfg!(debug_assertions) || u32::from(width) * u32::from(height) <= 66 * 66
}

fn crosscheck(name: &str, file: &[u8]) -> usize {
    let (ivf, frames) = echora_av1::ivf::parse_ivf(file).expect("an IVF file");
    let units: Vec<&[u8]> = frames.iter().map(|f| f.data).collect();
    let expected = parsed_outputs(&units);
    let num_planes = {
        let mut p = Parser::new();
        p.parse_temporal_unit(units[0])
            .expect("a parsable first unit");
        p.sequence()
            .expect("a sequence header")
            .color_config
            .num_planes()
    };
    let mut rav1d = Rav1d::new();
    let mut checked = 0;
    let mut failures = Vec::new();
    if !decode_with_rav1d(ivf.width, ivf.height) {
        return 0;
    }
    for (n, tu) in units.iter().enumerate() {
        let mut ours = expected[n].iter();
        rav1d.decode(tu, &mut |d| {
            let Some(h) = ours.next() else {
                failures.push(format!(
                    "{name} unit {n}: rav1d output a picture the parser did not"
                ));
                return;
            };
            for diff in compare(h, d, num_planes) {
                failures.push(format!("{name} unit {n} picture {checked}: {diff}"));
            }
            checked += 1;
        });
        let missing = ours.count();
        if missing > 0 {
            failures.push(format!(
                "{name} unit {n}: {missing} parsed pictures rav1d did not output"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{}",
        failures[..failures.len().min(40)].join("\n")
    );
    assert!(checked > 0);
    checked
}

#[test]
fn checked_in_vectors_match_rav1d() {
    let mut total = 0;
    for (name, data) in common::checked_in_vectors() {
        let n = crosscheck(&name, &data);
        eprintln!("{name}: {n} pictures match");
        total += n;
    }
    assert!(total >= if cfg!(debug_assertions) { 4 } else { 80 });
}

/// Every .ivf file in ECHORA_AV1_VECTORS (see tests/fetch-vectors.sh).
#[test]
fn extra_vectors_match_rav1d() {
    for (name, data) in common::extra_vectors() {
        let n = crosscheck(&name, &data);
        eprintln!("{name}: {n} pictures match");
    }
}

#[test]
fn frame_types_are_seen() {
    // The vectors exercise key and inter frames, hidden frames and frames
    // shown again.
    let mut types = std::collections::HashSet::new();
    let mut hidden = 0;
    let mut shown_again = 0;
    for (_, data) in common::checked_in_vectors() {
        let (_, frames) = echora_av1::ivf::parse_ivf(&data).unwrap();
        let mut parser = Parser::new();
        for f in frames {
            for o in parser.parse_temporal_unit(f.data).unwrap() {
                match o {
                    Output::Decode(p) => {
                        types.insert(p.header.frame_type as u8);
                        hidden += usize::from(!p.show_frame);
                    }
                    Output::ShowExisting(_) => shown_again += 1,
                    Output::Metadata(_) => {}
                }
            }
        }
    }
    assert!(types.contains(&(FrameType::Key as u8)) && types.contains(&(FrameType::Inter as u8)));
    assert!(hidden > 0 && shown_again > 0);
}
