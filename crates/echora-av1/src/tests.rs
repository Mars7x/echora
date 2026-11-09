//! Header element tests on hand-built bitstreams: each test writes a
//! sequence header and frames bit by bit, as an encoder would, and checks
//! what the parser makes of them.

use crate::bits::BitWriter;
use crate::frame::*;
use crate::obu::{ObuIter, ObuType};
use crate::sequence::tests::write_basic;
use crate::{Error, Output, Parser};

/// Wraps a payload in an OBU with obu_size.
fn obu(obu_type: ObuType, payload: &[u8]) -> Vec<u8> {
    let mut out = vec![(obu_type.raw() << 3) | 0b010];
    let mut size = payload.len();
    loop {
        let byte = (size & 0x7f) as u8;
        size >>= 7;
        if size == 0 {
            out.push(byte);
            break;
        }
        out.push(byte | 0x80);
    }
    out.extend_from_slice(payload);
    out
}

fn sequence_obu(width: u32, height: u32) -> Vec<u8> {
    let mut w = BitWriter::default();
    write_basic(&mut w, width, height);
    w.align();
    obu(ObuType::SequenceHeader, &w.bytes)
}

/// Options for the frames the tests write. The defaults describe a shown
/// key frame of the basic sequence (64x64, one tile).
#[derive(Clone)]
struct FrameSpec {
    frame_type: u32,
    show_frame: bool,
    showable_frame: bool,
    error_resilient_mode: bool,
    order_hint: u32,
    primary_ref_frame: u32,
    refresh_frame_flags: u32,
    ref_frame_idx: [u32; 7],
    short_signaling: Option<(u32, u32)>,
    base_q_idx: u32,
    loop_filter_level: [u32; 4],
    write_loop_filter_deltas: bool,
    segmentation: bool,
    global_motion_last: bool,
    reference_select: bool,
    skip_mode_present: bool,
    cdef_bits: u32,
    lr_types: [u32; 3],
    tile_data: Vec<u8>,
}

impl Default for FrameSpec {
    fn default() -> Self {
        Self {
            frame_type: 0,
            show_frame: true,
            showable_frame: false,
            error_resilient_mode: false,
            order_hint: 0,
            primary_ref_frame: 7,
            refresh_frame_flags: 0xff,
            ref_frame_idx: [0; 7],
            short_signaling: None,
            base_q_idx: 100,
            loop_filter_level: [10, 12, 3, 4],
            write_loop_filter_deltas: false,
            segmentation: false,
            global_motion_last: false,
            reference_select: false,
            skip_mode_present: false,
            cdef_bits: 1,
            lr_types: [0, 0, 0],
            tile_data: vec![0xaa; 5],
        }
    }
}

/// Writes uncompressed_header() for the basic sequence.
fn write_header(w: &mut BitWriter, f: &FrameSpec) {
    let intra = f.frame_type == 0 || f.frame_type == 2;
    w.flag(false) // show_existing_frame
        .put(2, f.frame_type)
        .flag(f.show_frame);
    if !f.show_frame {
        w.flag(f.showable_frame);
    }
    let er_implied = f.frame_type == 3 || (f.frame_type == 0 && f.show_frame);
    if !er_implied {
        w.flag(f.error_resilient_mode);
    }
    let error_resilient = er_implied || f.error_resilient_mode;
    w.flag(false) // disable_cdf_update
        .flag(false) // allow_screen_content_tools
        .flag(false) // frame_size_override_flag
        .put(7, f.order_hint);
    if !intra && !error_resilient {
        w.put(3, f.primary_ref_frame);
    }
    if !(f.frame_type == 3 || (f.frame_type == 0 && f.show_frame)) {
        w.put(8, f.refresh_frame_flags);
    }
    if (!intra || f.refresh_frame_flags != 0xff) && error_resilient {
        for _ in 0..8 {
            w.put(7, 0); // ref_order_hint, matching the test's slots
        }
    }
    if intra {
        w.flag(false); // render_and_frame_size_different
    } else {
        w.flag(f.short_signaling.is_some());
        if let Some((last, gold)) = f.short_signaling {
            w.put(3, last).put(3, gold);
        } else {
            for idx in f.ref_frame_idx {
                w.put(3, idx);
            }
        }
        if error_resilient {
            w.flag(false); // render_and_frame_size_different
        } else {
            // frame_size_override_flag is 0, so frame_size() and render_size().
            w.flag(false);
        }
        w.flag(true) // allow_high_precision_mv
            .flag(false) // is_filter_switchable
            .put(2, 1) // EIGHTTAP_SMOOTH
            .flag(true); // is_motion_mode_switchable
        if !error_resilient {
            w.flag(true); // use_ref_frame_mvs
        }
    }
    w.flag(false); // disable_frame_end_update_cdf
    // tile_info: 64x64 is a single superblock.
    w.flag(true);
    // quantization_params
    w.put(8, f.base_q_idx)
        .flag(true) // DeltaQYDc coded
        .su(7, -3)
        .flag(false) // DeltaQUDc
        .flag(true)
        .su(7, 5) // DeltaQUAc
        .flag(false); // using_qmatrix
    // segmentation_params
    w.flag(f.segmentation);
    if f.segmentation {
        let primary_none = intra || error_resilient || f.primary_ref_frame == 7;
        if !primary_none {
            w.flag(true).flag(false).flag(true); // update_map, temporal, update_data
        }
        for segment in 0..8 {
            for feature in 0..8 {
                let on = segment == 1 && (feature == 0 || feature == 5);
                w.flag(on);
                if on && feature == 0 {
                    w.su(9, -40);
                } else if on {
                    w.put(3, 4);
                }
            }
        }
    }
    // delta_q_params
    if f.base_q_idx > 0 {
        w.flag(false);
    }
    // loop_filter_params (not coded lossless here)
    let lf = f.loop_filter_level;
    w.put(6, lf[0]).put(6, lf[1]);
    if lf[0] != 0 || lf[1] != 0 {
        w.put(6, lf[2]).put(6, lf[3]);
    }
    w.put(3, 2) // sharpness
        .flag(true); // loop_filter_delta_enabled
    w.flag(f.write_loop_filter_deltas);
    if f.write_loop_filter_deltas {
        for i in 0..8 {
            w.flag(i == 1);
            if i == 1 {
                w.su(7, -9);
            }
        }
        w.flag(false).flag(true).su(7, 6);
    }
    // cdef_params
    w.put(2, 1).put(2, f.cdef_bits);
    for i in 0..(1 << f.cdef_bits) {
        w.put(4, 5 + i).put(2, 3).put(4, 2).put(2, 1);
    }
    // lr_params
    for t in f.lr_types {
        w.put(2, t);
    }
    if f.lr_types.iter().any(|&t| t != 0) {
        w.put(1, 1).put(1, 0); // lr_unit_shift 1
        if f.lr_types[1] != 0 || f.lr_types[2] != 0 {
            w.put(1, 1); // lr_uv_shift
        }
    }
    w.flag(true); // tx_mode_select
    if !intra {
        w.flag(f.reference_select);
    }
    // skip_mode_params: the caller says whether skip mode is allowed by
    // setting skip_mode_present only then.
    if f.skip_mode_present {
        w.flag(true);
    }
    if !intra && !error_resilient {
        w.flag(false); // allow_warped_motion
    }
    w.flag(true); // reduced_tx_set
    if !intra {
        for r in 0..7 {
            let global = r == 0 && f.global_motion_last;
            w.flag(global);
            if global {
                // TRANSLATION: is_rot_zoom 0, is_translation 1, then two
                // parameters with allow_high_precision_mv set: absBits 9,
                // precBits 3, mx 512, reference r = 0 against [-512, 513).
                w.flag(false).flag(true);
                write_subexp_param(w, 1025, 512, 512 + 7);
                write_subexp_param(w, 1025, 512, 512 - 3);
            }
        }
    }
}

/// Writes decode_signed_subexp_with_ref's bits for a value in the range
/// [0, num_syms) (already offset by -low) against reference r, where
/// (r << 1) <= num_syms so the recentering is the plain one.
fn write_subexp_param(w: &mut BitWriter, num_syms: u32, r: u32, value: u32) {
    // recenter(r, value)
    let v = if value > 2 * r {
        value
    } else if value >= r {
        (value - r) << 1
    } else {
        ((r - value) << 1) - 1
    };
    // encode_subexp(v, num_syms) with k = 3.
    let mut i = 0;
    let mut mk = 0;
    loop {
        let b2 = if i != 0 { 3 + i - 1 } else { 3 };
        let a = 1 << b2;
        if num_syms <= mk + 3 * a {
            write_ns(w, num_syms - mk, v - mk);
            return;
        }
        if v >= mk + a {
            w.flag(true);
            i += 1;
            mk += a;
        } else {
            w.flag(false).put(b2, v - mk);
            return;
        }
    }
}

fn write_ns(w: &mut BitWriter, n: u32, v: u32) {
    let bits = 32 - n.leading_zeros();
    let m = (1 << bits) - n;
    if v < m {
        w.put(bits - 1, v);
    } else {
        let x = v + m;
        w.put(bits - 1, x >> 1).put(1, x & 1);
    }
}

/// A frame OBU: header, byte alignment, then a single tile.
fn frame_obu(f: &FrameSpec) -> Vec<u8> {
    let mut w = BitWriter::default();
    write_header(&mut w, f);
    w.align();
    let mut payload = w.bytes.clone();
    payload.extend_from_slice(&f.tile_data);
    obu(ObuType::Frame, &payload)
}

fn temporal_unit(parts: &[Vec<u8>]) -> Vec<u8> {
    let mut tu = obu(ObuType::TemporalDelimiter, &[]);
    for p in parts {
        tu.extend_from_slice(p);
    }
    tu
}

fn decode(out: Vec<Output>) -> Vec<crate::DecodePicture> {
    out.into_iter()
        .filter_map(|o| match o {
            Output::Decode(p) => Some(*p),
            _ => None,
        })
        .collect()
}

#[test]
fn obu_framing() {
    let tu = temporal_unit(&[sequence_obu(64, 64), obu(ObuType::Padding, &[1, 2, 3])]);
    let obus: Vec<_> = ObuIter::new(&tu).map(|o| o.unwrap()).collect();
    assert_eq!(obus.len(), 3);
    assert_eq!(obus[0].header.obu_type, ObuType::TemporalDelimiter);
    assert_eq!(obus[1].header.obu_type, ObuType::SequenceHeader);
    assert_eq!(obus[1].offset, 2);
    assert_eq!(obus[2].header.obu_type, ObuType::Padding);
    assert_eq!(obus[2].payload, &[1, 2, 3]);
    assert_eq!(obus[2].payload_offset + 3, tu.len());
}

#[test]
fn obu_extension_and_operating_point() {
    // A tile group OBU with temporal_id 1, spatial_id 1.
    let data = [(4 << 3) | 0b110, (1 << 5) | (1 << 3), 1, 0xee];
    let o = crate::obu::parse_obu(&data, 0, None).unwrap();
    assert!(o.header.extension_flag);
    assert_eq!((o.header.temporal_id, o.header.spatial_id), (1, 1));
    assert_eq!(o.payload, &[0xee]);
    assert!(o.header.in_operating_point(0));
    assert!(o.header.in_operating_point(0x203)); // temporal 0-1, spatial 1
    assert!(!o.header.in_operating_point(0x101)); // temporal 0, spatial 0
}

#[test]
fn annexb_framing() {
    // temporal_unit( frame_unit( obu_length, OBU without obu_size )... )
    let td = [2 << 3];
    let pad = [15 << 3, 9, 9];
    let mut frame_unit = vec![td.len() as u8];
    frame_unit.extend(td);
    frame_unit.push(pad.len() as u8);
    frame_unit.extend(pad);
    let mut tu = vec![frame_unit.len() as u8];
    tu.extend(&frame_unit);
    let obus = crate::obu::annexb_temporal_unit(&tu).unwrap();
    assert_eq!(obus.len(), 2);
    assert_eq!(obus[0].header.obu_type, ObuType::TemporalDelimiter);
    assert_eq!(obus[1].payload, &[9, 9]);
    let mut stream = vec![tu.len() as u8];
    stream.extend(&tu);
    assert_eq!(
        crate::obu::annexb_temporal_units(&stream).unwrap(),
        vec![1..stream.len()]
    );
}

#[test]
fn forbidden_bit_is_rejected() {
    assert_eq!(
        crate::obu::parse_obu(&[0x80 | (2 << 3) | 2, 0], 0, None).unwrap_err(),
        Error::Invalid("obu_forbidden_bit set")
    );
}

#[test]
fn frame_before_sequence_header() {
    let mut p = Parser::new();
    let tu = temporal_unit(&[frame_obu(&FrameSpec::default())]);
    assert_eq!(
        p.parse_temporal_unit(&tu).unwrap_err(),
        Error::MissingSequenceHeader
    );
}

#[test]
fn key_frame_header() {
    let mut p = Parser::new();
    let spec = FrameSpec {
        lr_types: [2, 0, 3],
        ..FrameSpec::default()
    };
    let tu = temporal_unit(&[sequence_obu(64, 64), frame_obu(&spec)]);
    let pics = decode(p.parse_temporal_unit(&tu).unwrap());
    assert_eq!(pics.len(), 1);
    let pic = &pics[0];
    let h = &pic.header;
    assert_eq!(h.frame_type, FrameType::Key);
    assert!(h.show_frame && !h.showable_frame && h.error_resilient_mode);
    assert_eq!(h.refresh_frame_flags, 0xff);
    assert_eq!(h.primary_ref_frame, PRIMARY_REF_NONE);
    assert_eq!((h.size.frame_width, h.size.frame_height), (64, 64));
    assert_eq!((h.size.render_width, h.size.render_height), (64, 64));
    assert_eq!((h.size.mi_cols, h.size.mi_rows), (16, 16));
    assert!(h.force_integer_mv);
    // Quantization.
    let q = &h.quantization;
    assert_eq!(q.base_q_idx, 100);
    assert_eq!((q.delta_q_y_dc, q.delta_q_u_dc, q.delta_q_u_ac), (-3, 0, 5));
    assert_eq!((q.delta_q_v_dc, q.delta_q_v_ac), (0, 5));
    assert!(!h.coded_lossless);
    // Loop filter, with the default deltas.
    assert_eq!(h.loop_filter.level, [10, 12, 3, 4]);
    assert_eq!(h.loop_filter.sharpness, 2);
    assert!(h.loop_filter.delta_enabled && !h.loop_filter.delta_update);
    assert_eq!(h.loop_filter.ref_deltas, [1, 0, 0, 0, -1, 0, -1, -1]);
    // CDEF: a coded secondary strength of 3 means 4.
    assert_eq!((h.cdef.damping_minus_3, h.cdef.bits), (1, 1));
    assert_eq!(h.cdef.y_pri_strength[..2], [5, 6]);
    assert_eq!(h.cdef.y_sec_strength[..2], [4, 4]);
    assert_eq!(h.cdef.uv_pri_strength[..2], [2, 2]);
    assert_eq!(h.cdef.uv_sec_strength[..2], [1, 1]);
    // Loop restoration: coded 2 is Wiener, 3 self-guided.
    let lr = &h.loop_restoration;
    assert_eq!(
        lr.frame_restoration_type,
        [
            RestorationType::Wiener,
            RestorationType::None,
            RestorationType::Sgrproj
        ]
    );
    assert!(lr.uses_lr && lr.uses_chroma_lr);
    assert_eq!(lr.loop_restoration_size, [128, 64, 64]);
    assert_eq!(h.tx_mode, TxMode::Select);
    assert!(h.reduced_tx_set);
    // Tiles: one, holding the bytes after the header.
    assert_eq!(h.tile_info.num_tiles(), 1);
    assert_eq!(h.tile_info.mi_col_starts, vec![0, 16]);
    assert_eq!(h.tile_info.width_in_sbs_minus_1, vec![0]);
    assert_eq!(pic.tiles.len(), 1);
    let tile = pic.tiles[0];
    assert_eq!(tile.size, 5);
    assert_eq!(&tu[tile.offset as usize..][..5], &[0xaa; 5]);
    // The frame header OBU follows the delimiter and sequence header.
    assert_eq!(
        tu[pic.frame_header_offset as usize] >> 3,
        ObuType::Frame.raw()
    );
    // Every slot now holds the key frame.
    assert_eq!(p.slot_pictures(), [Some(pic.id); 8]);
    assert_eq!(pic.reference_name_slot_indices, [-1; 7]);
    assert_eq!(pic.reference_infos, std::array::from_fn(|_| None));
    assert_eq!(pic.sequence_generation, 1);
}

#[test]
fn inter_frames_load_and_save_state() {
    let mut p = Parser::new();
    let key = FrameSpec {
        write_loop_filter_deltas: true,
        segmentation: true,
        ..FrameSpec::default()
    };
    p.parse_temporal_unit(&temporal_unit(&[sequence_obu(64, 64), frame_obu(&key)]))
        .unwrap();

    // A hidden inter frame (order hint 4) into slot 1, reading slot 0.
    let hidden = FrameSpec {
        frame_type: 1,
        show_frame: false,
        showable_frame: true,
        order_hint: 4,
        primary_ref_frame: 0,
        refresh_frame_flags: 0b10,
        global_motion_last: true,
        ..FrameSpec::default()
    };
    let pics = decode(
        p.parse_temporal_unit(&temporal_unit(&[frame_obu(&hidden)]))
            .unwrap(),
    );
    let h = &pics[0].header;
    assert_eq!(h.frame_type, FrameType::Inter);
    assert!(!h.show_frame && h.showable_frame);
    assert_eq!(h.ref_frame_idx, [0; 7]);
    assert_eq!(pics[0].reference_name_slot_indices, [0; 7]);
    assert_eq!(h.interpolation_filter, InterpolationFilter::EightTapSmooth);
    assert!(h.allow_high_precision_mv && h.use_ref_frame_mvs);
    // Loop filter deltas and segmentation come from the key frame.
    assert_eq!(h.loop_filter.ref_deltas, [1, -9, 0, 0, -1, 0, -1, -1]);
    assert_eq!(h.loop_filter.mode_deltas, [0, 6]);
    assert!(!h.segmentation.enabled);
    // Order hints: every reference is the key frame at 0, in the past.
    assert_eq!(h.order_hints, [0; 8]);
    assert_eq!(h.ref_frame_sign_bias, 0);
    // Global motion: LAST is a translation by (7, -3) in 1/8 units, the
    // others are the identity.
    assert_eq!(h.global_motion.gm_type[LAST_FRAME], WarpModel::Translation);
    assert_eq!(h.global_motion.gm_params[LAST_FRAME][0], 7 << 13);
    assert_eq!(h.global_motion.gm_params[LAST_FRAME][1], -3 << 13);
    assert_eq!(h.global_motion.gm_params[LAST_FRAME][2], 1 << 16);
    assert_eq!(h.global_motion.gm_type[GOLDEN_FRAME], WarpModel::Identity);
    let hidden_id = pics[0].id;
    let key_id = pics[0].reference_pictures[0].unwrap();
    assert_eq!(p.slot_pictures()[1], Some(hidden_id));
    assert_eq!(p.slot_pictures()[0], Some(key_id));

    // A shown inter frame at order hint 2 between the two, with LAST from
    // slot 0 and BWDREF from slot 1: skip mode is allowed.
    let mut refs = [0; 7];
    refs[BWDREF_FRAME - LAST_FRAME] = 1;
    let between = FrameSpec {
        frame_type: 1,
        order_hint: 2,
        primary_ref_frame: 4, // BWDREF: load from the hidden frame
        refresh_frame_flags: 0b100,
        ref_frame_idx: refs,
        reference_select: true,
        skip_mode_present: true,
        ..FrameSpec::default()
    };
    let pics = decode(
        p.parse_temporal_unit(&temporal_unit(&[frame_obu(&between)]))
            .unwrap(),
    );
    let h = &pics[0].header;
    assert_eq!(h.order_hints[BWDREF_FRAME], 4);
    assert_eq!(h.ref_frame_sign_bias, 1 << BWDREF_FRAME);
    assert!(h.skip_mode_present);
    assert_eq!(h.skip_mode_frame, [LAST_FRAME as u8, BWDREF_FRAME as u8]);
    assert_eq!(
        pics[0].reference_pictures[BWDREF_FRAME - LAST_FRAME],
        Some(hidden_id)
    );
    // The reference information describes slot 1 as the hidden frame.
    let info = pics[0].reference_infos[1].as_ref().unwrap();
    assert_eq!(
        (info.order_hint, info.frame_type),
        (4, FrameType::Inter as u8)
    );
    assert_eq!(pics[0].setup_reference_info.order_hint, 2);
    assert_eq!(
        pics[0].setup_reference_info.ref_frame_sign_bias,
        1 << BWDREF_FRAME
    );
    let std = &pics[0].std_picture;
    assert_eq!(std.order_hints[BWDREF_FRAME], 4);
    assert_eq!(std.skip_mode_frame, [1, 5]);
    // The global motion was loaded from the hidden frame (primary
    // reference) and coded as identity here.
    assert_eq!(h.global_motion.gm_type[LAST_FRAME], WarpModel::Identity);

    // show_existing_frame of the hidden frame.
    let mut w = BitWriter::default();
    w.flag(true).put(3, 1).align();
    let out = p
        .parse_temporal_unit(&temporal_unit(&[obu(ObuType::FrameHeader, &w.bytes)]))
        .unwrap();
    let Output::ShowExisting(show) = &out[0] else {
        panic!("expected show_existing_frame")
    };
    assert_eq!(show.id, hidden_id);
    assert_eq!(show.refresh_frame_flags, 0);
    assert_eq!(show.header.order_hint, 4);
}

#[test]
fn segmentation_features() {
    let mut p = Parser::new();
    let key = FrameSpec {
        segmentation: true,
        ..FrameSpec::default()
    };
    let pics = decode(
        p.parse_temporal_unit(&temporal_unit(&[sequence_obu(64, 64), frame_obu(&key)]))
            .unwrap(),
    );
    let s = &pics[0].header.segmentation;
    assert!(s.enabled && s.update_map && s.update_data && !s.temporal_update);
    assert!(s.feature_enabled[1][SEG_LVL_ALT_Q] && s.feature_enabled[1][SEG_LVL_REF_FRAME]);
    assert_eq!(s.feature_data[1][SEG_LVL_ALT_Q], -40);
    assert_eq!(s.feature_data[1][SEG_LVL_REF_FRAME], 4);
    assert_eq!(s.last_active_seg_id, 1);
    assert!(s.seg_id_pre_skip);
    assert_eq!(pics[0].header.qindex(0), 100);
    assert_eq!(pics[0].header.qindex(1), 60);
    let std = &pics[0].std_picture.segmentation;
    assert_eq!(std.feature_enabled[1], 0b10_0001);
    assert_eq!(std.feature_data[1][0], -40);
}

#[test]
fn lossless_frames_skip_the_filters() {
    let mut p = Parser::new();
    // base_q_idx 0 with zero deltas is lossless; the writer above always
    // codes DeltaQYDc -3, so build the header here.
    let mut w = BitWriter::default();
    w.flag(false)
        .put(2, 0)
        .flag(true) // shown key frame
        .flag(false)
        .flag(false)
        .flag(false)
        .put(7, 0) // cdf, sct, override, order hint
        .flag(false) // render_and_frame_size_different
        .flag(false) // disable_frame_end_update_cdf
        .flag(true) // uniform tiles
        .put(8, 0) // base_q_idx
        .flag(false)
        .flag(false)
        .flag(false) // no deltas
        .flag(false) // using_qmatrix
        .flag(false) // segmentation_enabled
        // delta_q_params: base_q_idx is 0, nothing coded.
        // loop filter, cdef: skipped for coded lossless.
        // lr: AllLossless, skipped. tx mode: ONLY_4X4, not coded.
        .flag(false); // reduced_tx_set
    w.align();
    let mut payload = w.bytes.clone();
    payload.push(0x55);
    let tu = temporal_unit(&[sequence_obu(64, 64), obu(ObuType::Frame, &payload)]);
    let pics = decode(p.parse_temporal_unit(&tu).unwrap());
    let h = &pics[0].header;
    assert!(h.coded_lossless && h.all_lossless);
    assert_eq!(h.lossless_array, [true; 8]);
    assert_eq!(h.tx_mode, TxMode::Only4x4);
    assert_eq!(h.loop_filter.level, [0; 4]);
    assert_eq!(h.cdef, Cdef::default());
    assert!(!h.loop_restoration.uses_lr);
    assert_eq!(pics[0].tiles[0].size, 1);
}

#[test]
fn short_signaling_selects_references() {
    let mut p = Parser::new();
    p.parse_temporal_unit(&temporal_unit(&[
        sequence_obu(64, 64),
        frame_obu(&FrameSpec::default()),
    ]))
    .unwrap();
    // Fill slots with order hints 1..=7 (slot i holds hint i; slot 0 the key frame at 0).
    for hint in 1..8u32 {
        let f = FrameSpec {
            frame_type: 1,
            show_frame: false,
            showable_frame: true,
            order_hint: hint,
            primary_ref_frame: 0,
            refresh_frame_flags: 1 << hint,
            ..FrameSpec::default()
        };
        p.parse_temporal_unit(&temporal_unit(&[frame_obu(&f)]))
            .unwrap();
    }
    // Now decode order hint 4 with LAST = slot 3 and GOLDEN = slot 1.
    let f = FrameSpec {
        frame_type: 1,
        order_hint: 4,
        primary_ref_frame: 0,
        refresh_frame_flags: 0,
        short_signaling: Some((3, 1)),
        ..FrameSpec::default()
    };
    let pics = decode(
        p.parse_temporal_unit(&temporal_unit(&[frame_obu(&f)]))
            .unwrap(),
    );
    let h = &pics[0].header;
    assert!(h.frame_refs_short_signaling);
    // set_frame_refs: ALTREF is the latest backward (7), BWDREF the
    // earliest backward (4: hint 4 is not in the past), ALTREF2 the next
    // (5); LAST2 and LAST3 the latest forward references left (2, then 0).
    assert_eq!(h.ref_frame_idx, [3, 2, 0, 1, 4, 5, 7]);
}

#[test]
fn tiles_in_tile_groups() {
    // A 1024x256 frame with uniform spacing and two tile columns, coded as
    // a frame header OBU and two tile group OBUs.
    let mut w = BitWriter::default();
    write_basic(&mut w, 1024, 256);
    w.align();
    let seq = obu(ObuType::SequenceHeader, &w.bytes);

    let mut w = BitWriter::default();
    w.flag(false)
        .put(2, 0)
        .flag(true)
        .flag(false)
        .flag(false)
        .flag(false)
        .put(7, 0);
    w.flag(false) // render size
        .flag(false) // disable_frame_end_update_cdf
        .flag(true) // uniform_tile_spacing_flag
        .flag(true) // increment_tile_cols_log2 -> 1
        .flag(false) // stop
        .flag(false) // increment_tile_rows_log2: stop at 0
        .put(1, 1) // context_update_tile_id
        .put(2, 1) // tile_size_bytes_minus_1: two bytes
        .put(8, 20) // base_q_idx
        .flag(false)
        .flag(false)
        .flag(false)
        .flag(false) // quant
        .flag(false) // segmentation
        .flag(false) // delta_q_present
        .put(6, 0)
        .put(6, 0) // loop filter levels zero
        .put(3, 0)
        .flag(false) // sharpness, delta disabled
        .put(2, 0)
        .put(2, 0)
        .put(4, 0)
        .put(2, 0)
        .put(4, 0)
        .put(2, 0) // cdef
        .put(2, 0)
        .put(2, 0)
        .put(2, 0) // lr none
        .flag(false) // tx mode largest
        .flag(false); // reduced_tx_set
    w.align();
    let header = obu(ObuType::FrameHeader, &w.bytes);
    // Tile group 1: tile_start_and_end_present_flag, tg_start 0, tg_end 0.
    let mut w = BitWriter::default();
    w.flag(true).put(1, 0).put(1, 0).align();
    let mut tg1 = w.bytes.clone();
    tg1.extend([0x11; 7]);
    // Tile group 2: tile 1 alone.
    let mut w = BitWriter::default();
    w.flag(true).put(1, 1).put(1, 1).align();
    let mut tg2 = w.bytes.clone();
    tg2.extend([0x22; 9]);
    let tu = temporal_unit(&[
        seq,
        header,
        obu(ObuType::TileGroup, &tg1),
        obu(ObuType::TileGroup, &tg2),
    ]);
    let mut p = Parser::new();
    let pics = decode(p.parse_temporal_unit(&tu).unwrap());
    let pic = &pics[0];
    let t = &pic.header.tile_info;
    assert_eq!((t.tile_cols, t.tile_rows), (2, 1));
    assert_eq!(t.mi_col_starts, vec![0, 128, 256]);
    assert_eq!(t.width_in_sbs_minus_1, vec![7, 7]);
    assert_eq!(t.height_in_sbs_minus_1, vec![3]);
    assert_eq!(
        (t.context_update_tile_id, t.tile_size_bytes_minus_1),
        (1, 1)
    );
    assert_eq!(pic.tiles.len(), 2);
    assert_eq!(
        &tu[pic.tiles[0].offset as usize..][..pic.tiles[0].size as usize],
        &[0x11; 7]
    );
    assert_eq!(
        &tu[pic.tiles[1].offset as usize..][..pic.tiles[1].size as usize],
        &[0x22; 9]
    );
    assert_eq!(pic.tile_offsets().len(), 2);
    assert_eq!(pic.std_picture.tile_info.mi_col_starts, vec![0, 128]);
}

#[test]
fn tile_sizes_inside_one_group() {
    // The same two-tile frame as one frame OBU: the first tile carries a
    // two-byte little-endian tile_size_minus_1.
    let mut w = BitWriter::default();
    write_basic(&mut w, 1024, 256);
    w.align();
    let seq = obu(ObuType::SequenceHeader, &w.bytes);
    let mut w = BitWriter::default();
    w.flag(false)
        .put(2, 0)
        .flag(true)
        .flag(false)
        .flag(false)
        .flag(false)
        .put(7, 0);
    w.flag(false)
        .flag(false)
        .flag(true)
        .flag(true)
        .flag(false)
        .flag(false)
        .put(1, 0)
        .put(2, 1);
    w.put(8, 20)
        .flag(false)
        .flag(false)
        .flag(false)
        .flag(false)
        .flag(false)
        .flag(false);
    w.put(6, 0).put(6, 0).put(3, 0).flag(false);
    w.put(2, 0)
        .put(2, 0)
        .put(4, 0)
        .put(2, 0)
        .put(4, 0)
        .put(2, 0);
    w.put(2, 0).put(2, 0).put(2, 0).flag(false).flag(false);
    w.align();
    // tile_group: no start/end flag (one bit 0), aligned.
    w.flag(false).align();
    let mut payload = w.bytes.clone();
    payload.extend([2, 0]); // tile_size_minus_1 = 2
    payload.extend([0x33; 3]);
    payload.extend([0x44; 4]);
    let tu = temporal_unit(&[seq, obu(ObuType::Frame, &payload)]);
    let pics = decode(Parser::new().parse_temporal_unit(&tu).unwrap());
    let tiles = &pics[0].tiles;
    assert_eq!(tiles.len(), 2);
    assert_eq!(
        &tu[tiles[0].offset as usize..][..tiles[0].size as usize],
        &[0x33; 3]
    );
    assert_eq!(
        &tu[tiles[1].offset as usize..][..tiles[1].size as usize],
        &[0x44; 4]
    );
}

#[test]
fn missing_reference_is_an_error_and_keeps_state() {
    let mut p = Parser::new();
    // An inter frame first: nothing to reference.
    let tu = temporal_unit(&[
        sequence_obu(64, 64),
        frame_obu(&FrameSpec {
            frame_type: 1,
            primary_ref_frame: 0,
            ..FrameSpec::default()
        }),
    ]);
    assert_eq!(
        p.parse_temporal_unit(&tu).unwrap_err(),
        Error::MissingReference(0)
    );
    assert_eq!(p.slot_pictures(), [None; 8]);
    // A key frame recovers.
    let pics = decode(
        p.parse_temporal_unit(&temporal_unit(&[frame_obu(&FrameSpec::default())]))
            .unwrap(),
    );
    assert_eq!(pics.len(), 1);
}

#[test]
fn truncated_frames_fail_cleanly() {
    let tu = temporal_unit(&[sequence_obu(64, 64), frame_obu(&FrameSpec::default())]);
    for len in 0..tu.len() {
        let mut p = Parser::new();
        // Any prefix may fail but must not panic.
        let _ = p.parse_temporal_unit(&tu[..len]);
    }
}
