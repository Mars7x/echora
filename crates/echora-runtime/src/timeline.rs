//! ECHORA_TIMELINE=<file> (debugging): what each host and guest thread did
//! over a few presents, for laying one frame out: guest code runs, machine
//! lock boundaries by kind, NVN calls made with the state lock, the replay
//! thread's pieces and waits, and presents. Recording starts at present
//! ECHORA_TIMELINE_FROM (default 10400) and stops ECHORA_TIMELINE_PRESENTS
//! (default 10) presents later, when the file is written: one line per
//! span, "thread kind start_us end_us", microseconds since recording began.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

struct Recording {
    began: Instant,
    spans: Vec<(u64, String, u64, u64)>,
    until: u64,
}

static ON: AtomicBool = AtomicBool::new(false);
static RECORDING: Mutex<Option<Recording>> = Mutex::new(None);

fn file() -> Option<std::ffi::OsString> {
    static FILE: std::sync::OnceLock<Option<std::ffi::OsString>> = std::sync::OnceLock::new();
    FILE.get_or_init(|| std::env::var_os("ECHORA_TIMELINE"))
        .clone()
}

fn setting(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// Whether spans are being recorded (cheap: checked before each record).
#[inline]
pub(crate) fn on() -> bool {
    ON.load(Ordering::Relaxed)
}

/// Note `kind` on `thread` from `start` to `end`.
pub(crate) fn span(thread: u64, kind: &str, start: Instant, end: Instant) {
    if !on() {
        return;
    }
    let mut recording = RECORDING
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(recording) = recording.as_mut() {
        let at = |time: Instant| time.saturating_duration_since(recording.began).as_micros() as u64;
        let span = (thread, kind.to_string(), at(start), at(end));
        recording.spans.push(span);
    }
}

/// Present `count` happened: recording starts or ends here.
pub(crate) fn present(count: u64) {
    if file().is_none() {
        return;
    }
    let from = setting("ECHORA_TIMELINE_FROM", 10400);
    let mut recording = RECORDING
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if count == from && recording.is_none() {
        *recording = Some(Recording {
            began: Instant::now(),
            spans: Vec::new(),
            until: from + setting("ECHORA_TIMELINE_PRESENTS", 10),
        });
        ON.store(true, Ordering::Relaxed);
        return;
    }
    let Some(active) = recording.as_mut() else {
        return;
    };
    let now = Instant::now();
    let at = now.saturating_duration_since(active.began).as_micros() as u64;
    active.spans.push((0, format!("present {count}"), at, at));
    if count >= active.until {
        ON.store(false, Ordering::Relaxed);
        let text: String = active
            .spans
            .iter()
            .map(|(thread, kind, start, end)| {
                format!("{thread} {} {start} {end}\n", kind.replace(' ', "_"))
            })
            .collect();
        if let Some(path) = file()
            && let Err(error) = std::fs::write(&path, text)
        {
            eprintln!(
                "timeline {}: {error}",
                std::path::Path::new(&path).display()
            );
        }
        active.spans.clear();
        active.until = u64::MAX;
    }
}
