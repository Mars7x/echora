//! The user's settings (`echora-config`) as the runtime applies them.
//!
//! The front end passes the settings file in `ECHORA_SETTINGS`; without it,
//! or when the file is invalid, the defaults apply (the reason is printed).
//! Debug environment switches (`ECHORA_WINDOW_SIZE`, `ECHORA_FULLSCREEN`,
//! `ECHORA_RESOLUTION`, `ECHORA_RESOLUTION_SCALE`) take precedence.
//!
//! Applied so far: window size and mode, internal render scale, master
//! volume, and vibration on or off.

use std::sync::OnceLock;

use echora_config::{InternalResolutionSetting, Settings, WindowMode};

/// The settings, loaded on first use.
pub fn get() -> &'static Settings {
    static SETTINGS: OnceLock<Settings> = OnceLock::new();
    SETTINGS.get_or_init(|| {
        let Some(path) = std::env::var_os("ECHORA_SETTINGS") else {
            return Settings::default();
        };
        match Settings::load(std::path::Path::new(&path)) {
            Ok(settings) => settings,
            Err(error) => {
                eprintln!("settings: {error}; using defaults");
                Settings::default()
            }
        }
    })
}

/// Rows of the guest's render targets that the render scale is measured
/// against (the console's 1080-row output, as the renderer assumes).
const GUEST_ROWS: u32 = 1080;

/// The render scale (numerator, denominator) the settings ask for, if any.
pub fn render_scale() -> Option<(u32, u32)> {
    scale_for(&get().display)
}

fn scale_for(display: &echora_config::DisplaySettings) -> Option<(u32, u32)> {
    match display.internal_resolution {
        // Matching the output is known only for a window of a set size.
        InternalResolutionSetting::MatchOutput => (display.window_mode == WindowMode::Windowed)
            .then_some((display.window_height, GUEST_ROWS)),
        InternalResolutionSetting::Height { height } => Some((height, GUEST_ROWS)),
        InternalResolutionSetting::Scale { percent } => Some((percent, 100)),
    }
}

/// Master volume as a factor (0..=1).
pub fn master_volume() -> f32 {
    get().audio.volume.min(100) as f32 / 100.0
}

/// A vibration request, unless vibration is turned off.
pub fn rumble(device: u32, low: f32, high: f32) {
    if get().input.vibration {
        crate::input::rumble(device, low, high);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_scale_follows_the_internal_resolution_setting() {
        let mut display = echora_config::DisplaySettings::default();
        display.window_height = 1440;
        assert_eq!(scale_for(&display), Some((1440, 1080)));
        display.window_mode = WindowMode::BorderlessFullscreen;
        assert_eq!(scale_for(&display), None);
        display.internal_resolution = InternalResolutionSetting::Height { height: 2160 };
        assert_eq!(scale_for(&display), Some((2160, 1080)));
        display.internal_resolution = InternalResolutionSetting::Scale { percent: 150 };
        assert_eq!(scale_for(&display), Some((150, 100)));
    }
}
