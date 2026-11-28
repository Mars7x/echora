use std::cell::RefCell;
use std::io::Cursor;
use std::rc::Rc;
use std::sync::{Arc, Mutex, PoisonError, mpsc};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use ::pipewire as pw;
use pw::spa::param::audio::{AudioFormat, AudioInfoRaw, MAX_CHANNELS};
use pw::spa::pod::{Object, Pod, Value, serialize::PodSerializer};
use pw::spa::{param::ParamType, sys, utils::SpaTypes};
use pw::stream::{StreamBox, StreamFlags, StreamState};

/// How long `start` waits for the stream to connect. A suspended (or
/// Bluetooth) sink may take longer to start streaming, which is why a
/// connected but paused stream already counts as started.
const STARTUP: Duration = Duration::from_secs(2);

/// Consecutive attempts to recreate a lost stream before giving up on host
/// audio; a stream that starts resets the count.
const RETRIES: u32 = 8;

type Ready = mpsc::Sender<Result<(), String>>;

pub(super) struct Playback {
    shutdown: pw::channel::Sender<()>,
    /// Dropped to end a wait between reconnection attempts.
    stopping: Option<mpsc::Sender<()>>,
    worker: Option<JoinHandle<()>>,
}

impl Playback {
    pub(super) fn start(
        rate: u32,
        channels: u32,
        samples: Arc<Mutex<super::Queue>>,
    ) -> Result<Self, String> {
        let format = audio_format(rate, channels)?;
        let (shutdown, stop) = pw::channel::channel();
        let (stopping, stopped) = mpsc::channel();
        let (ready, started) = mpsc::channel();
        let worker = thread::Builder::new()
            .name("echora-pipewire".into())
            .spawn(move || run_playback(format, samples, stop, stopped, ready))
            .map_err(|error| error.to_string())?;
        let playback = Self {
            shutdown,
            stopping: Some(stopping),
            worker: Some(worker),
        };
        match started.recv_timeout(STARTUP) {
            Ok(started) => {
                started.map_err(|error| format!("starting PipeWire playback: {error}"))?;
            }
            // The worker keeps connecting and reports its errors itself.
            Err(mpsc::RecvTimeoutError::Timeout) => {
                eprintln!("audio: PipeWire is still connecting; playback starts when it does");
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err("starting PipeWire playback: the stream thread ended".into());
            }
        }
        Ok(playback)
    }
}

impl Drop for Playback {
    fn drop(&mut self) {
        self.stopping.take();
        let _ = self.shutdown.send(());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// What a stream state change means for playback.
#[derive(Debug, PartialEq)]
enum Change {
    /// Connected: paused until the sink runs, or streaming.
    Started,
    /// The stream failed or disconnected and must be recreated.
    Lost(String),
    Pending,
}

fn stream_change(state: StreamState) -> Change {
    match state {
        StreamState::Paused | StreamState::Streaming => Change::Started,
        StreamState::Error(error) => Change::Lost(error),
        StreamState::Unconnected => Change::Lost("the stream disconnected".into()),
        StreamState::Connecting => Change::Pending,
    }
}

/// The wait before reconnection attempt `failures + 1`, or `None` once the
/// attempts are used up.
fn retry_delay(failures: u32) -> Option<Duration> {
    (failures < RETRIES).then(|| Duration::from_millis(250 << failures.min(5)))
}

/// One stream's life, shared with its callbacks.
#[derive(Default)]
struct Session {
    started: bool,
    lost: Option<String>,
    stopped: bool,
}

/// Run streams until playback is dropped, recreating a lost stream with
/// backoff. Only a stream that fails before it starts, while `start` still
/// waits, ends playback.
fn run_playback(
    format: AudioInfoRaw,
    samples: Arc<Mutex<super::Queue>>,
    mut stop: pw::channel::Receiver<()>,
    stopping: mpsc::Receiver<()>,
    ready: Ready,
) {
    pw::init();
    let ready = Rc::new(RefCell::new(Some(ready)));
    let mut failures = 0;
    let mut retry = false;
    loop {
        let session = Rc::new(RefCell::new(Session::default()));
        let (result, returned) = run_stream(format, &samples, stop, &session, &ready, retry);
        stop = returned;
        let session = session.take();
        if session.stopped {
            return;
        }
        let error = match result {
            Err(error) => error.to_string(),
            Ok(()) => session.lost.unwrap_or_else(|| "the stream ended".into()),
        };
        if session.started {
            failures = 0;
        }
        // Not started yet: `start` reports the error, unless it stopped
        // waiting.
        let waiting = ready.borrow_mut().take();
        if waiting.is_some_and(|ready| ready.send(Err(error.clone())).is_ok()) {
            return;
        }
        let Some(delay) = retry_delay(failures) else {
            eprintln!("audio: giving up on PipeWire after {failures} attempts: {error}");
            return;
        };
        eprintln!("audio: PipeWire stream lost ({error}); reconnecting in {delay:?}");
        failures += 1;
        retry = true;
        if stopping.recv_timeout(delay) != Err(mpsc::RecvTimeoutError::Timeout) {
            return;
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

type StreamResult = Result<(), Box<dyn std::error::Error>>;

/// Run one stream until it is lost or playback stops, handing back the
/// shutdown receiver for the next stream.
fn run_stream(
    format: AudioInfoRaw,
    samples: &Arc<Mutex<super::Queue>>,
    stop: pw::channel::Receiver<()>,
    session: &Rc<RefCell<Session>>,
    ready: &Rc<RefCell<Option<Ready>>>,
    retry: bool,
) -> (StreamResult, pw::channel::Receiver<()>) {
    let mainloop = match pw::main_loop::MainLoopRc::new(None) {
        Ok(mainloop) => mainloop,
        Err(error) => return (Err(error.into()), stop),
    };
    let loop_to_stop = mainloop.clone();
    let stopped = session.clone();
    let attached = stop.attach(mainloop.loop_(), move |()| {
        stopped.borrow_mut().stopped = true;
        loop_to_stop.quit();
    });
    let result = stream_until_lost(&mainloop, format, samples, session, ready, retry);
    (result, attached.deattach())
}

fn stream_until_lost(
    mainloop: &pw::main_loop::MainLoopRc,
    format: AudioInfoRaw,
    samples: &Arc<Mutex<super::Queue>>,
    session: &Rc<RefCell<Session>>,
    ready: &Rc<RefCell<Option<Ready>>>,
    retry: bool,
) -> StreamResult {
    let context = pw::context::ContextRc::new(mainloop, None)?;
    let core = context.connect_rc(None)?;
    // A daemon that exits ends the connection with a core error.
    let loop_on_error = mainloop.clone();
    let lost = session.clone();
    let _core_listener = core
        .add_listener_local()
        .error(move |id, _, _, message| {
            if id == pw::core::PW_ID_CORE {
                lost.borrow_mut().lost.get_or_insert_with(|| message.into());
                loop_on_error.quit();
            }
        })
        .register();
    let stream = StreamBox::new(
        &core,
        "Echora",
        pw::properties::properties! {
            *pw::keys::MEDIA_TYPE => "Audio",
            *pw::keys::MEDIA_CATEGORY => "Playback",
            *pw::keys::MEDIA_ROLE => "Game",
        },
    )?;
    let loop_on_error = mainloop.clone();
    let bytes_per_frame = format.channels() as usize * 2;
    let changed = session.clone();
    let ready = ready.clone();
    let queue = samples.clone();
    let samples = samples.clone();
    let _listener = stream
        .add_local_listener_with_user_data(())
        .state_changed(move |_, _, _, state| match stream_change(state) {
            Change::Started => {
                let mut session = changed.borrow_mut();
                if retry && !session.started {
                    // What queued while reconnecting is up to a second old.
                    queue.lock().unwrap_or_else(PoisonError::into_inner).clear();
                }
                session.started = true;
                if let Some(ready) = ready.borrow_mut().take() {
                    let _ = ready.send(Ok(()));
                }
            }
            Change::Lost(error) => {
                changed.borrow_mut().lost.get_or_insert(error);
                loop_on_error.quit();
            }
            Change::Pending => {}
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
    fn a_paused_stream_counts_as_started() {
        assert_eq!(stream_change(StreamState::Paused), Change::Started);
        assert_eq!(stream_change(StreamState::Streaming), Change::Started);
        assert_eq!(stream_change(StreamState::Connecting), Change::Pending);
    }

    #[test]
    fn errors_and_disconnects_recreate_the_stream() {
        assert_eq!(
            stream_change(StreamState::Error("no sink".into())),
            Change::Lost("no sink".into())
        );
        assert!(matches!(
            stream_change(StreamState::Unconnected),
            Change::Lost(_)
        ));
    }

    #[test]
    fn reconnection_backs_off_then_gives_up() {
        let delays: Vec<_> = (0..)
            .map_while(retry_delay)
            .map(|delay| delay.as_millis())
            .collect();
        assert_eq!(delays, [250, 500, 1000, 2000, 4000, 8000, 8000, 8000]);
    }

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
