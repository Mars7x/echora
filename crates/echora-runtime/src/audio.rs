//! Host audio output: the PCM16 that audio-out buffers carry, streamed to
//! PipeWire's `pw-cat` (package pipewire-bin). Opt-in with ECHORA_AUDIO=1.
//!
//! A writer thread owns the child's stdin, so a full pipe never blocks the
//! guest; samples queue in memory until the host consumes them. Buffer
//! release timing stays with the service model (real time at the stream's
//! sample rate), so a guest that cannot keep up underruns rather than
//! drifting.

use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::mpsc::{Sender, channel};

/// Start a host stream of interleaved signed 16-bit samples, or `None` when
/// host audio is off or `pw-cat` cannot start.
pub fn open(sample_rate: u32, channels: u32) -> Option<Sender<Vec<u8>>> {
    std::env::var_os("ECHORA_AUDIO")?;
    let mut child = Command::new("pw-cat")
        .args(["--playback", "--format", "s16"])
        .args(["--rate", &sample_rate.to_string()])
        .args(["--channels", &channels.to_string()])
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| eprintln!("audio: pw-cat did not start: {error}"))
        .ok()?;
    let mut stdin = child.stdin.take()?;
    let (sender, receiver) = channel::<Vec<u8>>();
    std::thread::spawn(move || {
        for pcm in receiver {
            if stdin.write_all(&pcm).is_err() {
                break;
            }
        }
        drop(stdin);
        let _ = child.wait();
    });
    Some(sender)
}

/// Scale interleaved PCM16 by `volume` (1.0 leaves it unchanged).
pub fn apply_volume(pcm: &mut [u8], volume: f32) {
    if volume == 1.0 {
        return;
    }
    for sample in pcm.as_chunks_mut::<2>().0 {
        let value = i16::from_le_bytes(*sample) as f32 * volume;
        let clamped = value.clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        sample.copy_from_slice(&clamped.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn volume_scales_and_clamps() {
        let mut pcm = [1000i16, -1000, 30000, -30000]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        apply_volume(&mut pcm, 2.0);
        let samples: Vec<i16> = pcm
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| i16::from_le_bytes(*pair))
            .collect();
        assert_eq!(samples, [2000, -2000, i16::MAX, i16::MIN]);
    }
}
