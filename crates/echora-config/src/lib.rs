//! Echora's settings model, shared by the launcher and the runtime. Settings
//! are changed in the launcher; there is no in-game settings interface.
//!
//! Settings are stored as versioned TOML. What the user *requested* is kept
//! separate from what is *effective* on the current host: unsupported choices
//! are resolved to a working alternative with a recorded reason, never
//! silently reported as active. Every setting has an [`ApplyClass`] saying
//! what a live change costs.

use std::fmt;
use std::path::Path;

use echora_resolution::{AspectPolicy, Extent, InternalResolution, Ratio};
use serde::{Deserialize, Serialize};

pub const SCHEMA_VERSION: u32 = 2;

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("{path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("settings file is not valid: {0}")]
    Parse(String),
    #[error("settings schema {found} is newer than this build supports ({SCHEMA_VERSION})")]
    FutureSchema { found: u32 },
    #[error("invalid settings: {0}")]
    Invalid(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Settings {
    pub schema: u32,
    pub display: DisplaySettings,
    pub graphics: GraphicsSettings,
    pub frame: FrameSettings,
    pub audio: AudioSettings,
    pub input: InputSettings,
    pub launcher: LauncherSettings,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            schema: SCHEMA_VERSION,
            display: DisplaySettings::default(),
            graphics: GraphicsSettings::default(),
            frame: FrameSettings::default(),
            audio: AudioSettings::default(),
            input: InputSettings::default(),
            launcher: LauncherSettings::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WindowMode {
    Windowed,
    BorderlessFullscreen,
    ExclusiveFullscreen,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum InternalResolutionSetting {
    MatchOutput,
    Height {
        height: u32,
    },
    /// Render scale relative to the output, in percent.
    Scale {
        percent: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum AspectSetting {
    Original,
    Expand,
    Fixed { width: u32, height: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VsyncMode {
    On,
    Adaptive,
    Off,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DisplaySettings {
    /// Monitor name as reported by the windowing system; `None` = primary.
    pub monitor: Option<String>,
    pub window_mode: WindowMode,
    pub window_width: u32,
    pub window_height: u32,
    pub internal_resolution: InternalResolutionSetting,
    pub aspect: AspectSetting,
    pub vsync: VsyncMode,
}

impl Default for DisplaySettings {
    fn default() -> Self {
        Self {
            monitor: None,
            window_mode: WindowMode::Windowed,
            window_width: 1280,
            window_height: 720,
            internal_resolution: InternalResolutionSetting::MatchOutput,
            aspect: AspectSetting::Original,
            vsync: VsyncMode::On,
        }
    }
}

impl DisplaySettings {
    pub fn internal_resolution(&self) -> InternalResolution {
        match self.internal_resolution {
            InternalResolutionSetting::MatchOutput => InternalResolution::MatchOutput,
            InternalResolutionSetting::Height { height } => InternalResolution::Height(height),
            InternalResolutionSetting::Scale { percent } => {
                InternalResolution::Scale(Ratio::new(percent.max(1), 100))
            }
        }
    }

    pub fn aspect_policy(&self) -> AspectPolicy {
        match self.aspect {
            AspectSetting::Original => AspectPolicy::Original,
            AspectSetting::Expand => AspectPolicy::Expand,
            AspectSetting::Fixed { width, height } => {
                AspectPolicy::Fixed(Ratio::new(width.max(1), height.max(1)))
            }
        }
    }

    pub fn window_extent(&self) -> Extent {
        Extent::new(self.window_width, self.window_height)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphicsBackend {
    Vulkan,
    D3d12,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendPreference {
    Auto,
    Vulkan,
    D3d12,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GraphicsSettings {
    pub backend: BackendPreference,
    /// Physical device name; `None` lets the backend choose.
    pub device: Option<String>,
}

impl Default for GraphicsSettings {
    fn default() -> Self {
        Self {
            backend: BackendPreference::Auto,
            device: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum FrameLimit {
    /// The game's own pacing.
    Original,
    Fps {
        fps: u32,
    },
    Unlimited,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FrameSettings {
    pub limit: FrameLimit,
}

impl Default for FrameSettings {
    fn default() -> Self {
        Self {
            limit: FrameLimit::Original,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AudioSettings {
    /// Output device name; `None` follows the system default.
    pub device: Option<String>,
    /// 0-100.
    pub volume: u32,
}

impl Default for AudioSettings {
    fn default() -> Self {
        Self {
            device: None,
            volume: 100,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct InputSettings {
    pub vibration: bool,
}

impl Default for InputSettings {
    fn default() -> Self {
        Self { vibration: true }
    }
}

/// Launcher behaviour. These never affect a running game, so they have no
/// [`SettingKey`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LauncherSettings {
    /// Start the last chosen game straight away instead of showing the
    /// launcher window.
    pub skip: bool,
}

/// What applying a changed setting to a running game requires.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ApplyClass {
    Immediate,
    /// Recreate size-dependent GPU resources (swapchain, render targets).
    ResourceRecreation,
    /// Tear down and recreate the renderer; game state is preserved.
    RendererRestart,
    /// Takes effect the next time the game starts.
    GameRestart,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SettingKey {
    Monitor,
    WindowMode,
    WindowSize,
    InternalResolution,
    Aspect,
    Vsync,
    Backend,
    GraphicsDevice,
    FrameLimit,
    AudioDevice,
    Volume,
    Vibration,
}

impl SettingKey {
    pub fn apply_class(self) -> ApplyClass {
        use ApplyClass::*;
        match self {
            Self::FrameLimit | Self::Volume | Self::Vibration => Immediate,
            Self::WindowSize
            | Self::WindowMode
            | Self::Monitor
            | Self::InternalResolution
            | Self::Aspect
            | Self::Vsync
            | Self::AudioDevice => ResourceRecreation,
            // Guest GPU objects live in the backend; rebuilding them under a
            // different device is not supported, so these wait for a restart.
            Self::Backend | Self::GraphicsDevice => GameRestart,
        }
    }
}

impl fmt::Display for SettingKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

/// Host facts used to turn requested settings into effective ones.
#[derive(Debug, Clone)]
pub struct HostCapabilities {
    /// Backends compiled in and usable on this host, in preference order.
    pub backends: Vec<GraphicsBackend>,
    pub max_image_dimension: u32,
    pub supports_exclusive_fullscreen: bool,
}

/// A requested value that could not be honored, and what is used instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Adjustment {
    pub key: SettingKey,
    pub requested: String,
    pub effective: String,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct EffectiveSettings {
    pub settings: Settings,
    pub backend: Option<GraphicsBackend>,
    pub adjustments: Vec<Adjustment>,
}

impl Settings {
    /// Parses settings TOML, migrating older schema versions. Fields left out
    /// take their defaults; unknown fields are an error so a typo cannot
    /// silently change behavior.
    pub fn from_toml(text: &str) -> Result<Self, ConfigError> {
        let mut value: toml::Table =
            toml::from_str(text).map_err(|e| ConfigError::Parse(e.to_string()))?;
        let schema = match value.get("schema") {
            None => SCHEMA_VERSION,
            Some(v) => v
                .as_integer()
                .and_then(|i| u32::try_from(i).ok())
                .ok_or_else(|| ConfigError::Parse("schema must be a positive integer".into()))?,
        };
        if schema > SCHEMA_VERSION {
            return Err(ConfigError::FutureSchema { found: schema });
        }
        migrate(&mut value, schema)?;
        let settings: Self = value
            .try_into()
            .map_err(|e: toml::de::Error| ConfigError::Parse(e.to_string()))?;
        settings.validate()?;
        Ok(settings)
    }

    pub fn to_toml(&self) -> String {
        toml::to_string_pretty(self).unwrap_or_default()
    }

    /// Loads from `path`; a missing file yields defaults.
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        match std::fs::read_to_string(path) {
            Ok(text) => Self::from_toml(&text),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(source) => Err(ConfigError::Io {
                path: path.display().to_string(),
                source,
            }),
        }
    }

    /// Writes atomically: a temporary sibling file is renamed over `path`.
    pub fn save(&self, path: &Path) -> Result<(), ConfigError> {
        self.validate()?;
        let io = |source| ConfigError::Io {
            path: path.display().to_string(),
            source,
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(io)?;
        }
        let tmp = path.with_extension("toml.tmp");
        std::fs::write(&tmp, self.to_toml()).map_err(io)?;
        std::fs::rename(&tmp, path).map_err(io)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        let bad = |m: String| Err(ConfigError::Invalid(m));
        let d = &self.display;
        if !(1..=32768).contains(&d.window_width) || !(1..=32768).contains(&d.window_height) {
            return bad(format!(
                "window size {}x{}",
                d.window_width, d.window_height
            ));
        }
        match d.internal_resolution {
            InternalResolutionSetting::Height { height } if !(144..=16384).contains(&height) => {
                return bad(format!("internal resolution height {height} (144-16384)"));
            }
            InternalResolutionSetting::Scale { percent } if !(25..=400).contains(&percent) => {
                return bad(format!("render scale {percent}% (25-400)"));
            }
            _ => {}
        }
        if let AspectSetting::Fixed { width, height } = d.aspect
            && (width == 0 || height == 0 || width > 64 || height > 64)
        {
            return bad(format!("aspect ratio {width}:{height}"));
        }
        if let FrameLimit::Fps { fps } = self.frame.limit
            && !(10..=1000).contains(&fps)
        {
            return bad(format!("frame limit {fps} (10-1000)"));
        }
        if self.audio.volume > 100 {
            return bad(format!("volume {} (0-100)", self.audio.volume));
        }
        Ok(())
    }

    /// The settings that differ between `self` and `next`, each with its
    /// apply class. The highest class decides what a live change costs.
    pub fn changes(&self, next: &Self) -> Vec<(SettingKey, ApplyClass)> {
        let (a, b) = (self, next);
        let mut out = Vec::new();
        let mut diff = |changed: bool, key: SettingKey| {
            if changed {
                out.push((key, key.apply_class()));
            }
        };
        diff(a.display.monitor != b.display.monitor, SettingKey::Monitor);
        diff(
            a.display.window_mode != b.display.window_mode,
            SettingKey::WindowMode,
        );
        diff(
            a.display.window_extent() != b.display.window_extent(),
            SettingKey::WindowSize,
        );
        diff(
            a.display.internal_resolution != b.display.internal_resolution,
            SettingKey::InternalResolution,
        );
        diff(a.display.aspect != b.display.aspect, SettingKey::Aspect);
        diff(a.display.vsync != b.display.vsync, SettingKey::Vsync);
        diff(
            a.graphics.backend != b.graphics.backend,
            SettingKey::Backend,
        );
        diff(
            a.graphics.device != b.graphics.device,
            SettingKey::GraphicsDevice,
        );
        diff(a.frame.limit != b.frame.limit, SettingKey::FrameLimit);
        diff(a.audio.device != b.audio.device, SettingKey::AudioDevice);
        diff(a.audio.volume != b.audio.volume, SettingKey::Volume);
        diff(
            a.input.vibration != b.input.vibration,
            SettingKey::Vibration,
        );
        out
    }

    /// Resolves requested settings against what this host supports.
    pub fn resolve(&self, caps: &HostCapabilities) -> EffectiveSettings {
        let mut eff = self.clone();
        let mut adjustments = Vec::new();

        let wanted = match self.graphics.backend {
            BackendPreference::Auto => None,
            BackendPreference::Vulkan => Some(GraphicsBackend::Vulkan),
            BackendPreference::D3d12 => Some(GraphicsBackend::D3d12),
        };
        let backend = match wanted {
            Some(b) if caps.backends.contains(&b) => Some(b),
            Some(b) => {
                let fallback = caps.backends.first().copied();
                adjustments.push(Adjustment {
                    key: SettingKey::Backend,
                    requested: format!("{b:?}"),
                    effective: fallback.map_or("none".into(), |f| format!("{f:?}")),
                    reason: "backend is not available on this host".into(),
                });
                fallback
            }
            None => caps.backends.first().copied(),
        };

        if self.display.window_mode == WindowMode::ExclusiveFullscreen
            && !caps.supports_exclusive_fullscreen
        {
            eff.display.window_mode = WindowMode::BorderlessFullscreen;
            adjustments.push(Adjustment {
                key: SettingKey::WindowMode,
                requested: "exclusive_fullscreen".into(),
                effective: "borderless_fullscreen".into(),
                reason: "the windowing system does not offer exclusive fullscreen".into(),
            });
        }

        if let InternalResolutionSetting::Height { height } = self.display.internal_resolution
            && height > caps.max_image_dimension
        {
            eff.display.internal_resolution = InternalResolutionSetting::Height {
                height: caps.max_image_dimension,
            };
            adjustments.push(Adjustment {
                key: SettingKey::InternalResolution,
                requested: format!("{height}p"),
                effective: format!("{}p", caps.max_image_dimension),
                reason: "exceeds the GPU's maximum image dimension".into(),
            });
        }

        EffectiveSettings {
            settings: eff,
            backend,
            adjustments,
        }
    }
}

/// The settings file: `ECHORA_SETTINGS`, else `settings.toml` in the
/// platform's per-user configuration directory (`$XDG_CONFIG_HOME/echora`
/// or `~/.config/echora`, `%APPDATA%\\Echora`, or
/// `~/Library/Application Support/Echora`). `None` when neither is known.
pub fn default_path() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;
    if let Some(path) = std::env::var_os("ECHORA_SETTINGS") {
        return Some(PathBuf::from(path));
    }
    let env = |k| std::env::var_os(k).map(PathBuf::from);
    let dir = if cfg!(target_os = "windows") {
        env("APPDATA").map(|d| d.join("Echora"))
    } else {
        env("XDG_CONFIG_HOME")
            .or_else(|| env("HOME").map(|h| h.join(".config")))
            .map(|d| d.join("echora"))
    };
    dir.map(|d| d.join("settings.toml"))
}

/// Upgrades an older settings table in place to [`SCHEMA_VERSION`].
fn migrate(value: &mut toml::Table, from: u32) -> Result<(), ConfigError> {
    if from == 0 {
        return Err(ConfigError::Parse(
            "schema 0 is not a valid settings version".into(),
        ));
    }
    // Schema 2 dropped `input.settings_key`: there is no in-game settings
    // interface to open.
    if from < 2
        && let Some(toml::Value::Table(input)) = value.get_mut("input")
    {
        input.remove("settings_key");
    }
    value.insert(
        "schema".into(),
        toml::Value::Integer(i64::from(SCHEMA_VERSION)),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linux_caps() -> HostCapabilities {
        HostCapabilities {
            backends: vec![GraphicsBackend::Vulkan],
            max_image_dimension: 16384,
            supports_exclusive_fullscreen: false,
        }
    }

    #[test]
    fn defaults_round_trip() {
        let s = Settings::default();
        let text = s.to_toml();
        assert_eq!(Settings::from_toml(&text).unwrap(), s);
        assert_eq!(Settings::from_toml("").unwrap(), s);
    }

    #[test]
    fn partial_file_keeps_other_defaults() {
        let s = Settings::from_toml(
            r#"
            schema = 1
            [display]
            window_width = 3440
            window_height = 1440
            internal_resolution = { mode = "height", height = 1440 }
            aspect = { mode = "expand" }
            [frame]
            limit = { mode = "fps", fps = 120 }
            "#,
        )
        .unwrap();
        assert_eq!(s.display.window_extent(), Extent::new(3440, 1440));
        assert_eq!(
            s.display.internal_resolution(),
            InternalResolution::Height(1440)
        );
        assert_eq!(s.display.aspect_policy(), AspectPolicy::Expand);
        assert_eq!(s.frame.limit, FrameLimit::Fps { fps: 120 });
        assert_eq!(s.audio, AudioSettings::default());
    }

    #[test]
    fn rejects_unknown_fields_invalid_values_and_future_schema() {
        assert!(matches!(
            Settings::from_toml("[display]\nwindow_widht = 5"),
            Err(ConfigError::Parse(_))
        ));
        assert!(matches!(
            Settings::from_toml("[audio]\nvolume = 150"),
            Err(ConfigError::Invalid(_))
        ));
        assert!(matches!(
            Settings::from_toml(
                "[display]\ninternal_resolution = { mode = \"scale\", percent = 5 }"
            ),
            Err(ConfigError::Invalid(_))
        ));
        assert!(matches!(
            Settings::from_toml("schema = 99"),
            Err(ConfigError::FutureSchema { found: 99 })
        ));
    }

    #[test]
    fn change_classification() {
        let a = Settings::default();
        let mut b = a.clone();
        b.audio.volume = 40;
        b.display.internal_resolution = InternalResolutionSetting::Scale { percent: 150 };
        b.graphics.backend = BackendPreference::Vulkan;
        let changes = a.changes(&b);
        assert_eq!(
            changes,
            [
                (
                    SettingKey::InternalResolution,
                    ApplyClass::ResourceRecreation
                ),
                (SettingKey::Backend, ApplyClass::GameRestart),
                (SettingKey::Volume, ApplyClass::Immediate),
            ]
        );
        assert_eq!(
            changes.iter().map(|c| c.1).max(),
            Some(ApplyClass::GameRestart)
        );
        assert!(a.changes(&a.clone()).is_empty());
    }

    #[test]
    fn unsupported_requests_are_reported_not_hidden() {
        let mut s = Settings::default();
        s.graphics.backend = BackendPreference::D3d12;
        s.display.window_mode = WindowMode::ExclusiveFullscreen;
        s.display.internal_resolution = InternalResolutionSetting::Height { height: 16384 };
        let mut caps = linux_caps();
        caps.max_image_dimension = 8192;
        let eff = s.resolve(&caps);
        assert_eq!(eff.backend, Some(GraphicsBackend::Vulkan));
        assert_eq!(
            eff.settings.display.window_mode,
            WindowMode::BorderlessFullscreen
        );
        assert_eq!(
            eff.settings.display.internal_resolution,
            InternalResolutionSetting::Height { height: 8192 }
        );
        let keys: Vec<_> = eff.adjustments.iter().map(|a| a.key).collect();
        assert_eq!(
            keys,
            [
                SettingKey::Backend,
                SettingKey::WindowMode,
                SettingKey::InternalResolution
            ]
        );
        // The requested settings are untouched.
        assert_eq!(s.graphics.backend, BackendPreference::D3d12);
    }

    #[test]
    fn launcher_section_round_trips_and_defaults_off() {
        let settings = Settings::default();
        assert!(!settings.launcher.skip);
        let parsed: Settings = toml::from_str("[launcher]\nskip = true\n").unwrap();
        assert!(parsed.launcher.skip);
        assert!(Settings::default().changes(&parsed).is_empty());
    }

    #[test]
    fn schema_1_files_drop_the_settings_shortcut() {
        let old = "schema = 1\n[input]\nvibration = false\nsettings_key = \"F1\"\n";
        let settings = Settings::from_toml(old).unwrap();
        assert!(!settings.input.vibration);
        assert_eq!(settings.schema, SCHEMA_VERSION);
    }

    #[test]
    fn save_and_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested/settings.toml");
        assert_eq!(Settings::load(&path).unwrap(), Settings::default());
        let mut s = Settings::default();
        s.input.vibration = false;
        s.save(&path).unwrap();
        assert_eq!(Settings::load(&path).unwrap(), s);
    }
}
