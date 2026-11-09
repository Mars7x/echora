//! Prints the parsed headers of an AV1 stream, one line per picture, and
//! what changes from one picture to the next: a tool for comparing a
//! hardware decode against a software one.
//!
//! ```sh
//! cargo run -p echora-av1 --example av1_headers -- movie.ivf
//! cargo run -p echora-av1 --example av1_headers -- movie.obu   # raw low-overhead OBUs
//! ```

use std::collections::BTreeMap;

use echora_av1::{FrameHeader, Output, Parser};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: av1_headers FILE.ivf|FILE.obu")?;
    let data = std::fs::read(&path)?;
    let units: Vec<&[u8]> = if data.starts_with(b"DKIF") {
        echora_av1::ivf::parse_ivf(&data)?
            .1
            .iter()
            .map(|f| f.data)
            .collect()
    } else {
        vec![&data]
    };
    let mut parser = Parser::new();
    let mut previous: BTreeMap<&'static str, String> = BTreeMap::new();
    let mut picture = 0;
    for (unit, bytes) in units.iter().enumerate() {
        for output in parser.parse_temporal_unit(bytes)? {
            match output {
                Output::Decode(p) => {
                    let h = &p.header;
                    println!(
                        "picture {picture} (unit {unit}, id {}): {:?} show {} showable {} order_hint {} \
                         refresh {:#04x} primary_ref {} refs {:?} {}x{} tiles {}",
                        p.id.0,
                        h.frame_type,
                        h.show_frame,
                        h.showable_frame,
                        h.order_hint,
                        h.refresh_frame_flags,
                        h.primary_ref_frame,
                        p.reference_name_slot_indices,
                        p.frame_width,
                        p.frame_height,
                        p.tiles.len(),
                    );
                    let now = features(h);
                    for (name, value) in &now {
                        match previous.get(name) {
                            Some(old) if old == value => {}
                            Some(old) => println!("    {name}: {old} -> {value}"),
                            None => println!("    {name}: {value}"),
                        }
                    }
                    previous = now;
                    picture += 1;
                }
                Output::ShowExisting(s) => {
                    println!(
                        "show existing (unit {unit}): id {} from slot {} {:?} refresh {:#04x}",
                        s.id.0, s.slot, s.frame_type, s.refresh_frame_flags
                    );
                }
                Output::Metadata(m) => println!("metadata (unit {unit}): {m:?}"),
            }
        }
    }
    Ok(())
}

/// The header features that most affect reconstruction, as text.
fn features(h: &FrameHeader) -> BTreeMap<&'static str, String> {
    let q = &h.quantization;
    let s = &h.segmentation;
    let lf = &h.loop_filter;
    let c = &h.cdef;
    let lr = &h.loop_restoration;
    let n = 1usize << c.bits;
    let mut f = BTreeMap::new();
    f.insert(
        "quant",
        format!(
            "base {} dc/ac y {} u {}/{} v {}/{} qm {}",
            q.base_q_idx,
            q.delta_q_y_dc,
            q.delta_q_u_dc,
            q.delta_q_u_ac,
            q.delta_q_v_dc,
            q.delta_q_v_ac,
            q.using_qmatrix
        ),
    );
    f.insert(
        "segmentation",
        format!(
            "enabled {} update_map {} temporal {} update_data {} enabled_mask {:?} data {:?}",
            s.enabled,
            s.update_map,
            s.temporal_update,
            s.update_data,
            s.feature_enabled.map(|e| e
                .iter()
                .enumerate()
                .fold(0u8, |m, (j, &on)| m | (u8::from(on) << j))),
            s.feature_data
        ),
    );
    f.insert(
        "delta q/lf",
        format!(
            "q {} res {} lf {} res {} multi {}",
            h.delta.delta_q_present,
            h.delta.delta_q_res,
            h.delta.delta_lf_present,
            h.delta.delta_lf_res,
            h.delta.delta_lf_multi
        ),
    );
    f.insert("loop filter", format!(
        "levels {:?} sharpness {} delta_enabled {} delta_update {} ref_deltas {:?} mode_deltas {:?}",
        lf.level, lf.sharpness, lf.delta_enabled, lf.delta_update, lf.ref_deltas, lf.mode_deltas
    ));
    f.insert(
        "cdef",
        format!(
            "damping {} bits {} y pri {:?} sec {:?} uv pri {:?} sec {:?}",
            c.damping_minus_3 + 3,
            c.bits,
            &c.y_pri_strength[..n],
            &c.y_sec_strength[..n],
            &c.uv_pri_strength[..n],
            &c.uv_sec_strength[..n]
        ),
    );
    f.insert(
        "loop restoration",
        format!(
            "types {:?} sizes {:?}",
            lr.frame_restoration_type, lr.loop_restoration_size
        ),
    );
    f.insert(
        "modes",
        format!(
            "tx {:?} reference_select {} skip_mode {} {:?} warped {} reduced_tx_set {} \
         interp {:?} ref_mvs {} disable_frame_end_update_cdf {} lossless {}",
            h.tx_mode,
            h.reference_select,
            h.skip_mode_present,
            h.skip_mode_frame,
            h.allow_warped_motion,
            h.reduced_tx_set,
            h.interpolation_filter,
            h.use_ref_frame_mvs,
            h.disable_frame_end_update_cdf,
            h.coded_lossless
        ),
    );
    f.insert(
        "global motion",
        format!("{:?}", &h.global_motion.gm_type[1..]),
    );
    f
}
