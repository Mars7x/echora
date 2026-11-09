//! Decodes the first pictures of a raw AV1 stream (low-overhead OBUs, as
//! `ffmpeg -c:v copy -f obu` writes) on the GPU's video engine and with
//! rav1d, and compares the NV12 pictures.
//!   av1-gpu-check FILE.obu [PICTURES]
//! Exit status 1 when any picture differs or the GPU decode fails.

use std::collections::VecDeque;

use echora_av1::obu::{ObuIter, ObuType};
use echora_gpu::video::{VideoDevice, nv12_size};
use echora_gpu::video_av1::MovieDecoder;
use echora_runtime::av1::Decoder;

/// The stream split at its temporal delimiters.
fn temporal_units(data: &[u8]) -> Result<Vec<&[u8]>, String> {
    let mut starts = Vec::new();
    for obu in ObuIter::new(data) {
        let obu = obu.map_err(|error| format!("OBU stream: {error:?}"))?;
        if obu.header.obu_type == ObuType::TemporalDelimiter {
            starts.push(obu.offset);
        }
    }
    if starts.first() != Some(&0) {
        starts.insert(0, 0);
    }
    let ends = starts.iter().skip(1).copied().chain([data.len()]);
    Ok(starts
        .iter()
        .zip(ends)
        .map(|(&start, end)| &data[start..end])
        .collect())
}

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: av1-gpu-check FILE.obu [PICTURES]");
        std::process::exit(2);
    };
    let limit: usize = args.next().and_then(|n| n.parse().ok()).unwrap_or(8);
    let data = std::fs::read(&path).unwrap_or_else(|error| {
        eprintln!("{path}: {error}");
        std::process::exit(2);
    });
    let units = temporal_units(&data).unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(2);
    });
    let device = VideoDevice::new().unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(2);
    });
    println!(
        "{}: {} temporal units; device {}",
        path,
        units.len(),
        device.name
    );
    // AV1_CHECK_HIDDEN: compare the hidden pictures too.
    let hidden = std::env::var_os("AV1_CHECK_HIDDEN").is_some();
    let mut gpu = MovieDecoder::new(device);
    gpu.set_return_hidden(hidden);
    let mut cpu = Decoder::with_invisible_frames(hidden).unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(2);
    });
    let mut gpu_pictures: VecDeque<(u32, u32, Vec<u8>)> = VecDeque::new();
    let mut cpu_pictures = VecDeque::new();
    let (mut compared, mut differing) = (0usize, 0usize);
    let started = std::time::Instant::now();
    let mut gpu_time = std::time::Duration::ZERO;
    for (index, unit) in units.iter().enumerate() {
        if compared >= limit {
            break;
        }
        let begun = std::time::Instant::now();
        let shown = match gpu.decode(unit) {
            Ok(shown) => shown,
            Err(error) => {
                println!("unit {index}: GPU decode failed: {error}");
                std::process::exit(1);
            }
        };
        for picture in shown {
            let pitch = picture.width.next_multiple_of(2);
            let mut out = vec![0u8; nv12_size(picture.height, pitch)];
            if let Err(error) = gpu.read_nv12(picture, pitch, &mut out) {
                println!("unit {index}: GPU read-back failed: {error}");
                std::process::exit(1);
            }
            gpu_pictures.push_back((picture.width, picture.height, out));
        }
        gpu_time += begun.elapsed();
        match cpu.decode(unit) {
            Ok(pictures) => cpu_pictures.extend(pictures),
            Err(error) => {
                println!("unit {index}: rav1d failed: {error}");
                std::process::exit(1);
            }
        }
        while !gpu_pictures.is_empty() && !cpu_pictures.is_empty() {
            let (Some((width, height, ours)), Some(reference)) =
                (gpu_pictures.pop_front(), cpu_pictures.pop_front())
            else {
                break;
            };
            let expected = &reference.nv12;
            let luma = reference.chroma_offset();
            let differ = |range: std::ops::Range<usize>| {
                let (mut count, mut largest) = (0usize, 0u8);
                for (a, b) in ours[range.clone()].iter().zip(&expected[range]) {
                    if a != b {
                        count += 1;
                        largest = largest.max(a.abs_diff(*b));
                    }
                }
                (count, largest)
            };
            // AV1_CHECK_DUMP=DIR: both pictures as gpu-N.nv12 and
            // cpu-N.nv12, for inspection.
            if let Some(directory) = std::env::var_os("AV1_CHECK_DUMP") {
                let directory = std::path::Path::new(&directory);
                let _ = std::fs::write(directory.join(format!("gpu-{compared}.nv12")), &ours);
                let _ = std::fs::write(directory.join(format!("cpu-{compared}.nv12")), expected);
            }
            let size_ok = (width, height) == (reference.width, reference.height);
            let (y, y_max) = differ(0..luma);
            let (uv, uv_max) = differ(luma..ours.len().min(expected.len()));
            println!(
                "picture {compared}: {width}x{height}{} Y {y} differ (max {y_max}), UV {uv} differ (max {uv_max})",
                if size_ok {
                    String::new()
                } else {
                    format!(" (rav1d {}x{})", reference.width, reference.height)
                }
            );
            if !size_ok || y != 0 || uv != 0 {
                differing += 1;
            }
            compared += 1;
        }
    }
    println!(
        "{compared} pictures compared, {differing} differ; GPU {:.1} ms per picture, total {:.1} s",
        gpu_time.as_secs_f64() * 1000.0 / compared.max(1) as f64,
        started.elapsed().as_secs_f64()
    );
    std::process::exit(i32::from(differing != 0 || compared == 0));
}
