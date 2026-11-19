use std::collections::VecDeque;
use std::io::Cursor;
use std::sync::{Arc, Mutex, mpsc};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use ::pipewire as pw;
use pw::spa::param::audio::{AudioFormat, AudioInfoRaw, MAX_CHANNELS};
use pw::spa::pod::{Object, Pod, Value, serialize::PodSerializer};
use pw::spa::{param::ParamType, sys, utils::SpaTypes};
use pw::stream::{StreamBox, StreamFlags, StreamState};

pub(super) struct Playback {
    shutdown: pw::channel::Sender<()>,
    worker: Option<JoinHandle<()>>,
}

impl Playback {
    pub(super) fn start(
        rate: u32,
        channels: u32,
        samples: Arc<Mutex<VecDeque<i16>>>,
    ) -> Result<Self, String> {
        let format = audio_format(rate, channels)?;
        let (shutdown, stop) = pw::channel::channel();
        let (ready, started) = mpsc::channel();
        let worker = thread::Builder::new()
            .name("echora-pipewire".into())
            .spawn(move || {
                if let Err(error) = run_stream(format, samples, stop, ready.clone()) {
                    let _ = ready.send(Err(error.to_string()));
                }
            })
            .map_err(|error| error.to_string())?;
        let playback = Self {
            shutdown,
            worker: Some(worker),
        };
        started
            .recv_timeout(Duration::from_secs(2))
            .map_err(|error| format!("starting PipeWire playback: {error}"))??;
        Ok(playback)
    }
}

impl Drop for Playback {
    fn drop(&mut self) {
        let _ = self.shutdown.send(());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn audio_format(rate: u32, channels: u32) -> Result<AudioInfoRaw, String> {
    if rate == 0 || channels == 0 || channels as usize > MAX_CHANNELS {
        return Err(format!(
            "invalid PipeWire format with {channels} channels at {rate} Hz"
        ));
    }
    let mut format = AudioInfoRaw::new();
    format.set_format(AudioFormat::S16LE);
    format.set_rate(rate);
    format.set_channels(channels);
    let speakers: &[u32] = match channels {
        1 => &[sys::SPA_AUDIO_CHANNEL_MONO],
        2 => &[sys::SPA_AUDIO_CHANNEL_FL, sys::SPA_AUDIO_CHANNEL_FR],
        6 => &[
            sys::SPA_AUDIO_CHANNEL_FL,
            sys::SPA_AUDIO_CHANNEL_FR,
            sys::SPA_AUDIO_CHANNEL_FC,
            sys::SPA_AUDIO_CHANNEL_LFE,
            sys::SPA_AUDIO_CHANNEL_RL,
            sys::SPA_AUDIO_CHANNEL_RR,
        ],
        _ => &[],
    };
    let mut positions = [0; MAX_CHANNELS];
    positions[..speakers.len()].copy_from_slice(speakers);
    format.set_position(positions);
    Ok(format)
}

fn run_stream(
    format: AudioInfoRaw,
    samples: Arc<Mutex<VecDeque<i16>>>,
    stop: pw::channel::Receiver<()>,
    ready: mpsc::Sender<Result<(), String>>,
) -> Result<(), Box<dyn std::error::Error>> {
    pw::init();
    let mainloop = pw::main_loop::MainLoopRc::new(None)?;
    let context = pw::context::ContextRc::new(&mainloop, None)?;
    let core = context.connect_rc(None)?;
    let stream = StreamBox::new(
        &core,
        "Echora",
        pw::properties::properties! {
            *pw::keys::MEDIA_TYPE => "Audio",
            *pw::keys::MEDIA_CATEGORY => "Playback",
            *pw::keys::MEDIA_ROLE => "Game",
        },
    )?;
    let loop_to_stop = mainloop.clone();
    let _shutdown = stop.attach(mainloop.loop_(), move |()| loop_to_stop.quit());
    let loop_on_error = mainloop.clone();
    let bytes_per_frame = format.channels() as usize * 2;
    let mut ready = Some(ready);
    let _listener = stream
        .add_local_listener_with_user_data(())
        .state_changed(move |_, _, _, state| match state {
            StreamState::Streaming => {
                if let Some(ready) = ready.take() {
                    let _ = ready.send(Ok(()));
                }
            }
            StreamState::Error(error) => {
                if let Some(ready) = ready.take() {
                    let _ = ready.send(Err(error));
                } else {
                    eprintln!("audio: PipeWire stream error: {error}");
                }
                loop_on_error.quit();
            }
            _ => {}
        })
        .process(move |stream, _| {
            let Some(mut buffer) = stream.dequeue_buffer() else {
                return;
            };
            let requested = buffer.requested();
            let Some(data) = buffer.datas_mut().first_mut() else {
                return;
            };
            let size = data.data().map_or(0, |bytes| {
                let capacity = bytes.len() / bytes_per_frame;
                let frames = if requested == 0 {
                    capacity
                } else {
                    capacity.min(usize::try_from(requested).unwrap_or(usize::MAX))
                };
                let size = frames * bytes_per_frame;
                super::pull(
                    &samples,
                    bytes[..size].as_chunks_mut::<2>().0,
                    i16::to_le_bytes,
                    [0; 2],
                );
                size
            });
            let chunk = data.chunk_mut();
            *chunk.offset_mut() = 0;
            *chunk.stride_mut() = bytes_per_frame as i32;
            *chunk.size_mut() = size as u32;
        })
        .register()?;
    let (serialized, _) = PodSerializer::serialize(
        Cursor::new(Vec::new()),
        &Value::Object(Object {
            type_: SpaTypes::ObjectParamFormat.as_raw(),
            id: ParamType::EnumFormat.as_raw(),
            properties: format.into(),
        }),
    )?;
    let serialized = serialized.into_inner();
    let format = Pod::from_bytes(&serialized).ok_or("invalid PipeWire format")?;
    stream.connect(
        pw::spa::utils::Direction::Output,
        None,
        StreamFlags::AUTOCONNECT | StreamFlags::MAP_BUFFERS,
        &mut [format],
    )?;
    mainloop.run();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surround_format_keeps_the_guest_speaker_order() {
        let format = audio_format(48_000, 6).unwrap();
        assert_eq!(
            &format.position()[..6],
            &[
                sys::SPA_AUDIO_CHANNEL_FL,
                sys::SPA_AUDIO_CHANNEL_FR,
                sys::SPA_AUDIO_CHANNEL_FC,
                sys::SPA_AUDIO_CHANNEL_LFE,
                sys::SPA_AUDIO_CHANNEL_RL,
                sys::SPA_AUDIO_CHANNEL_RR,
            ]
        );
    }
}
