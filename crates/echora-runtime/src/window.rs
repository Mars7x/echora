//! Host window (ECHORA_WINDOW=1): a winit window on its own thread that
//! presented frames are shown in, and the keyboard state for the guest's
//! controller.
//!
//! The guest keeps running on the main thread; the window thread only owns
//! the window and its events. The GPU side builds its swapchain from the
//! native handles published here.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use echora_config::WindowMode;
use echora_gpu::present::WindowHandle;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Fullscreen, Window, WindowId};

/// Controller buttons, in the bit order of the HID npad button state.
/// Set when the user asked the run to end (the window was closed).
pub static STOP_REQUESTED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub mod button {
    pub const A: u64 = 1 << 0;
    pub const B: u64 = 1 << 1;
    pub const X: u64 = 1 << 2;
    pub const Y: u64 = 1 << 3;
    pub const L: u64 = 1 << 6;
    pub const R: u64 = 1 << 7;
    pub const ZL: u64 = 1 << 8;
    pub const ZR: u64 = 1 << 9;
    pub const PLUS: u64 = 1 << 10;
    pub const MINUS: u64 = 1 << 11;
    pub const LEFT: u64 = 1 << 12;
    pub const UP: u64 = 1 << 13;
    pub const RIGHT: u64 = 1 << 14;
    pub const DOWN: u64 = 1 << 15;
    pub const STICK_L_LEFT: u64 = 1 << 16;
    pub const STICK_L_UP: u64 = 1 << 17;
    pub const STICK_L_RIGHT: u64 = 1 << 18;
    pub const STICK_L_DOWN: u64 = 1 << 19;

    /// A button by name (A, B, UP, PLUS, ...), for scripted input.
    pub fn named(name: &str) -> Option<u64> {
        Some(match name.trim().to_ascii_uppercase().as_str() {
            "A" => A,
            "B" => B,
            "X" => X,
            "Y" => Y,
            "L" => L,
            "R" => R,
            "ZL" => ZL,
            "ZR" => ZR,
            "PLUS" => PLUS,
            "MINUS" => MINUS,
            "LEFT" => LEFT,
            "UP" => UP,
            "RIGHT" => RIGHT,
            "DOWN" => DOWN,
            _ => return None,
        })
    }
}

/// Keyboard layout: X/Z/S/A are A/B/X/Y, Q/W are L/R, 1/2 are ZL/ZR,
/// Enter and Backspace are Plus and Minus, the arrow keys the D-pad and
/// I/J/K/L the left stick.
fn button_for(key: KeyCode) -> u64 {
    use button::*;
    match key {
        KeyCode::KeyX => A,
        KeyCode::KeyZ => B,
        KeyCode::KeyS => X,
        KeyCode::KeyA => Y,
        KeyCode::KeyQ => L,
        KeyCode::KeyW => R,
        KeyCode::Digit1 => ZL,
        KeyCode::Digit2 => ZR,
        KeyCode::Enter => PLUS,
        KeyCode::Backspace => MINUS,
        KeyCode::ArrowLeft => LEFT,
        KeyCode::ArrowUp => UP,
        KeyCode::ArrowRight => RIGHT,
        KeyCode::ArrowDown => DOWN,
        KeyCode::KeyJ => STICK_L_LEFT,
        KeyCode::KeyI => STICK_L_UP,
        KeyCode::KeyL => STICK_L_RIGHT,
        KeyCode::KeyK => STICK_L_DOWN,
        _ => 0,
    }
}

/// State shared between the window thread and the guest.
pub struct Shared {
    handle: Mutex<Option<WindowHandle>>,
    /// Inner size in pixels: width << 32 | height.
    size: AtomicU64,
    /// Held buttons (see [`button`]).
    pub buttons: AtomicU64,
    failed: AtomicBool,
}

impl Shared {
    /// The native window, once the window thread has created it (waits for
    /// up to five seconds).
    pub fn handle(&self) -> Option<WindowHandle> {
        for _ in 0..500 {
            if let Some(handle) = *self.handle.lock().unwrap() {
                return Some(handle);
            }
            if self.failed.load(Ordering::Relaxed) {
                return None;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        None
    }

    pub fn size(&self) -> (u32, u32) {
        let size = self.size.load(Ordering::Relaxed);
        ((size >> 32) as u32, size as u32)
    }
}

static SHARED: OnceLock<&'static Shared> = OnceLock::new();

/// The window, opened on first use when ECHORA_WINDOW is set.
pub fn shared() -> Option<&'static Shared> {
    std::env::var_os("ECHORA_WINDOW")?;
    Some(SHARED.get_or_init(|| {
        let shared: &'static Shared = Box::leak(Box::new(Shared {
            handle: Mutex::new(None),
            size: AtomicU64::new({
                let (width, height) = window_size();
                u64::from(width) << 32 | u64::from(height)
            }),
            buttons: AtomicU64::new(0),
            failed: AtomicBool::new(false),
        }));
        std::thread::Builder::new()
            .name("window".into())
            .spawn(move || {
                if let Err(error) = run(shared) {
                    eprintln!("window: {error}");
                    shared.failed.store(true, Ordering::Relaxed);
                }
            })
            .expect("spawn the window thread");
        shared
    }))
}

fn run(shared: &'static Shared) -> Result<(), String> {
    let mut builder = EventLoop::builder();
    // The event loop runs on its own thread, which winit allows on Linux
    // and Windows; macOS requires the main thread (not supported yet).
    #[cfg(target_os = "linux")]
    {
        winit::platform::wayland::EventLoopBuilderExtWayland::with_any_thread(&mut builder, true);
        winit::platform::x11::EventLoopBuilderExtX11::with_any_thread(&mut builder, true);
    }
    #[cfg(target_os = "windows")]
    winit::platform::windows::EventLoopBuilderExtWindows::with_any_thread(&mut builder, true);
    let event_loop = builder.build().map_err(|error| error.to_string())?;
    let mut app = App {
        shared,
        window: None,
        alt_held: false,
    };
    event_loop
        .run_app(&mut app)
        .map_err(|error| error.to_string())
}

struct App {
    shared: &'static Shared,
    window: Option<Window>,
    alt_held: bool,
}

/// The window's initial size: ECHORA_WINDOW_SIZE=WIDTHxHEIGHT, else the
/// render resolution ECHORA_RESOLUTION, else the settings' window size.
fn window_size() -> (u32, u32) {
    let size = |name| {
        let value = std::env::var(name).ok()?;
        let (width, height) = value.trim().split_once(['x', 'X'])?;
        Some((width.trim().parse().ok()?, height.trim().parse().ok()?))
            .filter(|&(width, height): &(u32, u32)| width > 0 && height > 0)
    };
    size("ECHORA_WINDOW_SIZE")
        .or_else(|| size("ECHORA_RESOLUTION"))
        .unwrap_or_else(|| {
            let display = &crate::settings::get().display;
            (display.window_width, display.window_height)
        })
}

impl App {
    fn publish(&self, window: &Window) -> Result<(), String> {
        let display = window
            .display_handle()
            .map_err(|error| error.to_string())?
            .as_raw();
        let raw = window
            .window_handle()
            .map_err(|error| error.to_string())?
            .as_raw();
        let handle = match (display, raw) {
            (RawDisplayHandle::Wayland(display), RawWindowHandle::Wayland(window)) => {
                WindowHandle::Wayland {
                    display: display.display.as_ptr(),
                    surface: window.surface.as_ptr(),
                }
            }
            (RawDisplayHandle::Xlib(display), RawWindowHandle::Xlib(window)) => {
                WindowHandle::Xlib {
                    display: display.display.ok_or("no Xlib display")?.as_ptr(),
                    window: window.window,
                }
            }
            (RawDisplayHandle::Windows(_), RawWindowHandle::Win32(window)) => WindowHandle::Win32 {
                hinstance: window.hinstance.map_or(0, |h| h.get()),
                hwnd: window.hwnd.get(),
            },
            _ => return Err("unsupported window system".into()),
        };
        *self.shared.handle.lock().unwrap() = Some(handle);
        Ok(())
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let (width, height) = window_size();
        let mut attributes = Window::default_attributes()
            .with_title("Echora")
            .with_inner_size(winit::dpi::PhysicalSize::new(width, height));
        // ECHORA_FULLSCREEN=1 overrides the settings' window mode. Exclusive
        // fullscreen uses the monitor's current mode.
        let mode = match std::env::var_os("ECHORA_FULLSCREEN") {
            Some(value) if value != "0" => WindowMode::BorderlessFullscreen,
            Some(_) => WindowMode::Windowed,
            None => crate::settings::get().display.window_mode,
        };
        match mode {
            WindowMode::Windowed => {}
            WindowMode::BorderlessFullscreen => {
                attributes = attributes.with_fullscreen(Some(Fullscreen::Borderless(None)));
            }
            WindowMode::ExclusiveFullscreen => {
                let video = event_loop
                    .primary_monitor()
                    .and_then(|monitor| monitor.video_modes().next());
                attributes = attributes.with_fullscreen(Some(match video {
                    Some(video) => Fullscreen::Exclusive(video),
                    None => Fullscreen::Borderless(None),
                }));
            }
        }
        let window = match event_loop.create_window(attributes) {
            Ok(window) => window,
            Err(error) => {
                eprintln!("window: {error}");
                self.shared.failed.store(true, Ordering::Relaxed);
                event_loop.exit();
                return;
            }
        };
        if let Err(error) = self.publish(&window) {
            eprintln!("window: {error}");
            self.shared.failed.store(true, Ordering::Relaxed);
        }
        self.window = Some(window);
    }

    fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                // Closing the window ends the run at the next host boundary
                // (with its report). The window and its surface stay until
                // the process ends: a present may be using them right now,
                // and destroying them under it crashed the run.
                STOP_REQUESTED.store(true, Ordering::Relaxed);
            }
            WindowEvent::Resized(size) => {
                self.shared.size.store(
                    u64::from(size.width) << 32 | u64::from(size.height),
                    Ordering::Relaxed,
                );
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let PhysicalKey::Code(code) = event.physical_key else {
                    return;
                };
                // F11 (or Alt+Enter) switches between the window and
                // borderless fullscreen.
                let toggle = code == KeyCode::F11 || (code == KeyCode::Enter && self.alt_held);
                if matches!(code, KeyCode::AltLeft | KeyCode::AltRight) {
                    self.alt_held = event.state == ElementState::Pressed;
                }
                if toggle && event.state == ElementState::Pressed && !event.repeat {
                    if let Some(window) = &self.window {
                        let fullscreen = window.fullscreen().is_none();
                        window.set_fullscreen(fullscreen.then_some(Fullscreen::Borderless(None)));
                    }
                    return;
                }
                let bit = button_for(code);
                if event.state == ElementState::Pressed {
                    self.shared.buttons.fetch_or(bit, Ordering::Relaxed);
                } else {
                    self.shared.buttons.fetch_and(!bit, Ordering::Relaxed);
                }
            }
            _ => {}
        }
    }
}
