//! The launcher window: the game's art fills the window and a single dock at
//! the bottom holds what the player needs, with settings in a sheet above it.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitCode};
use std::sync::Arc;
use std::time::Duration;

use echora_config::{InternalResolutionSetting, Settings, WindowMode};
use echora_content::ContentSource;
use gpui::{
    AnyElement, App, Application, Bounds, BoxShadow, Context, CursorStyle, Decorations,
    FocusHandle, FontWeight, HitboxBehavior, Hsla, Image, ImageFormat, KeyBinding, MouseButton,
    MouseMoveEvent, ObjectFit, PathPromptOptions, Pixels, RenderImage, ResizeEdge, SharedString,
    TitlebarOptions, Window, WindowBackgroundAppearance, WindowBounds, WindowDecorations,
    WindowOptions, actions, canvas, div, img, point, prelude::*, px, relative, rgba, size, svg,
};

use crate::chrome::{self, ButtonLayout, WindowButton};
use crate::install;

const APP_ID: &str = "io.github.mars7x.Echora";

/// Internal resolution choices offered in settings.
const RESOLUTIONS: [(&str, InternalResolutionSetting); 5] = [
    ("Window", InternalResolutionSetting::MatchOutput),
    ("720p", InternalResolutionSetting::Height { height: 720 }),
    ("1080p", InternalResolutionSetting::Height { height: 1080 }),
    ("1440p", InternalResolutionSetting::Height { height: 1440 }),
    ("2160p", InternalResolutionSetting::Height { height: 2160 }),
];

const WINDOW_MODES: [(&str, WindowMode); 3] = [
    ("Windowed", WindowMode::Windowed),
    ("Borderless", WindowMode::BorderlessFullscreen),
    ("Fullscreen", WindowMode::ExclusiveFullscreen),
];

actions!(echora, [Play, ToggleSettings, CloseSheet, ChooseGame, Quit]);

/// Colours as `0xRRGGBBAA`: near-black grays with white for text and the
/// primary action.
mod palette {
    pub const BACKDROP: u32 = 0x0d0d0dff;
    pub const DOCK: u32 = 0x121212e6;
    pub const SHEET: u32 = 0x141414ff;
    pub const DIVIDER: u32 = 0x262626ff;
    pub const CONTROL: u32 = 0x242424ff;
    pub const CONTROL_HOVER: u32 = 0x2e2e2eff;
    pub const SELECTED: u32 = 0xf5f5f5ff;
    pub const SELECTED_TEXT: u32 = 0x0d0d0dff;
    pub const TEXT: u32 = 0xf5f5f5ff;
    pub const TEXT_MUTED: u32 = 0x8f8f8fff;
    pub const PLAY: u32 = 0xffffffff;
    pub const PLAY_HOVER: u32 = 0xe0e0e0ff;
    pub const PLAY_TEXT: u32 = 0x0d0d0dff;
    pub const SCRIM: u32 = 0x00000080;
    pub const WINDOW_BUTTON: u32 = 0xffffff1f;
    pub const WINDOW_BUTTON_HOVER: u32 = 0xffffff33;
    pub const WINDOW_BORDER: u32 = 0xffffff14;
    pub const ART_SHADE: u32 = 0x00000033;
}

/// The panel shown over the launcher.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Sheet {
    Settings,
    About,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SettingsTab {
    Game,
    Display,
    Controls,
    Launcher,
}

impl SettingsTab {
    const ALL: [Self; 4] = [Self::Game, Self::Display, Self::Controls, Self::Launcher];

    fn label(self) -> &'static str {
        match self {
            Self::Display => "Display",
            Self::Controls => "Controls",
            Self::Game => "Game",
            Self::Launcher => "Launcher",
        }
    }
}

/// The game the launcher will start.
struct Game {
    path: PathBuf,
    title: SharedString,
    version: SharedString,
    icon: Option<Arc<Image>>,
}

/// The backdrop cropped to the window's shape, so it can be drawn at exactly
/// the window's size and its corners rounded with the window's.
struct Backdrop {
    crop: (u32, u32, u32, u32),
    image: Arc<RenderImage>,
}

struct Launcher {
    focus: FocusHandle,
    game: Option<Game>,
    settings: Settings,
    settings_path: Option<PathBuf>,
    sheet: Option<Sheet>,
    /// Window buttons the desktop asks for, when decorations are ours.
    buttons: ButtonLayout,
    /// Corner radius of the window this frame (zero when maximized or tiled).
    corner: Pixels,
    /// A title bar press that turns into a window move once the mouse drags.
    moving: bool,
    /// The frosted backdrop, decoded once (BGRA, as GPUI uploads it).
    art: Option<image::RgbaImage>,
    backdrop: Option<Backdrop>,
    settings_tab: SettingsTab,
    notice: Option<SharedString>,
    notice_generation: u64,
    running: bool,
    /// The game being installed and how far the copy has got.
    installing: Option<(SharedString, Arc<install::Progress>)>,
}

pub fn run() -> ExitCode {
    // `--show` opens the window even when the player chose to skip it.
    let show = std::env::args().skip(1).any(|arg| arg == "--show");
    if !show && let Some(code) = skip_to_game() {
        return code;
    }
    let buttons = ButtonLayout::from_desktop();
    Application::new()
        .with_assets(chrome::Icons)
        .run(move |cx: &mut App| {
            cx.bind_keys([
                KeyBinding::new("enter", Play, Some("Launcher")),
                KeyBinding::new("ctrl-,", ToggleSettings, Some("Launcher")),
                KeyBinding::new("cmd-,", ToggleSettings, Some("Launcher")),
                KeyBinding::new("escape", CloseSheet, Some("Launcher")),
                KeyBinding::new("ctrl-o", ChooseGame, Some("Launcher")),
                KeyBinding::new("cmd-o", ChooseGame, Some("Launcher")),
                KeyBinding::new("ctrl-q", Quit, None),
                KeyBinding::new("cmd-q", Quit, None),
            ]);
            cx.on_action(|_: &Quit, cx| cx.quit());
            let bounds = Bounds::centered(None, size(px(960.0), px(600.0)), cx);
            let options = WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                titlebar: Some(TitlebarOptions {
                    title: Some("Echora".into()),
                    appears_transparent: cfg!(target_os = "macos"),
                    traffic_light_position: Some(point(px(14.0), px(14.0))),
                }),
                app_id: Some(APP_ID.into()),
                window_min_size: Some(size(px(720.0), px(460.0))),
                window_decorations: Some(WindowDecorations::Client),
                // Transparent so client-side decorations can round the corners
                // and draw the shadow.
                window_background: WindowBackgroundAppearance::Transparent,
                ..Default::default()
            };
            let opened = cx.open_window(options, |window, cx| {
                cx.new(|cx| {
                    let launcher = Launcher::new(buttons, cx);
                    window.focus(&launcher.focus);
                    launcher
                })
            });
            if let Err(error) = opened {
                eprintln!("echora-launcher: could not open a window: {error}");
                cx.quit();
            }
            cx.activate(true);
        });
    ExitCode::SUCCESS
}

/// With "Skip launcher" on, starts the last game and waits for it. `None`
/// means the window should open instead.
fn skip_to_game() -> Option<ExitCode> {
    let (settings_path, settings) = load_settings();
    if !settings.launcher.skip {
        return None;
    }
    let mut command = game_command(&last_game()?, settings_path.as_deref())?;
    match command.status() {
        Ok(status) if status.success() => Some(ExitCode::SUCCESS),
        Ok(_) => Some(ExitCode::FAILURE),
        Err(error) => {
            eprintln!("echora-launcher: couldn't start the game: {error}");
            None
        }
    }
}

/// The settings file's location and its contents, or defaults.
fn load_settings() -> (Option<PathBuf>, Settings) {
    let path = echora_config::default_path();
    let settings = path
        .as_deref()
        .map(Settings::load)
        .transpose()
        .unwrap_or_else(|error| {
            eprintln!("settings: {error}; using defaults");
            None
        })
        .unwrap_or_default();
    (path, settings)
}

/// The game chosen last time, if it is still there.
fn last_game() -> Option<PathBuf> {
    last_game_file()
        .and_then(|file| std::fs::read_to_string(file).ok())
        .map(|text| PathBuf::from(text.trim()))
        .filter(|path| path.exists())
}

/// `echora run GAME` using the `echora` binary next to the launcher.
fn game_command(game: &Path, settings: Option<&Path>) -> Option<Command> {
    let echora = std::env::current_exe()
        .ok()
        .map(|exe| exe.with_file_name(format!("echora{}", std::env::consts::EXE_SUFFIX)))
        .filter(|path| path.is_file())?;
    let mut command = Command::new(echora);
    command.arg("run").arg(game);
    if let Some(settings) = settings {
        command.env("ECHORA_SETTINGS", settings);
    }
    Some(command)
}

impl Launcher {
    fn new(buttons: ButtonLayout, cx: &mut Context<Self>) -> Self {
        let (settings_path, settings) = load_settings();
        let mut launcher = Self {
            focus: cx.focus_handle(),
            game: None,
            settings,
            settings_path,
            sheet: None,
            buttons,
            corner: px(0.0),
            moving: false,
            art: decode_art(BACKDROP),
            backdrop: None,
            settings_tab: SettingsTab::Game,
            notice: None,
            notice_generation: 0,
            running: false,
            installing: None,
        };
        if let Some(last) = last_game() {
            launcher.select_game(last, cx);
        }
        launcher
    }

    /// Shows a short message above the dock for a few seconds.
    fn notify(&mut self, message: impl Into<SharedString>, cx: &mut Context<Self>) {
        self.notice = Some(message.into());
        self.notice_generation += 1;
        let generation = self.notice_generation;
        cx.spawn(async move |this, cx| {
            cx.background_executor().timer(Duration::from_secs(4)).await;
            let _ = this.update(cx, |this, cx| {
                if this.notice_generation == generation {
                    this.notice = None;
                    cx.notify();
                }
            });
        })
        .detach();
        cx.notify();
    }

    fn select_game(&mut self, path: PathBuf, cx: &mut Context<Self>) {
        match read_game(&path) {
            Ok(game) => {
                if let Some(file) = last_game_file() {
                    let _ = file
                        .parent()
                        .map(std::fs::create_dir_all)
                        .transpose()
                        .and_then(|_| std::fs::write(&file, path.display().to_string()));
                }
                self.game = Some(game);
            }
            Err(message) => self.notify(message, cx),
        }
        cx.notify();
    }

    fn choose_game(&mut self, _: &ChooseGame, _: &mut Window, cx: &mut Context<Self>) {
        // A NUP package is a file and a dump is a folder; the platform picker
        // offers both.
        let paths = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: true,
            multiple: false,
            prompt: Some("Choose".into()),
        });
        cx.spawn(async move |this, cx| {
            let Ok(Ok(Some(paths))) = paths.await else {
                return;
            };
            let Some(path) = paths.into_iter().next() else {
                return;
            };
            let _ = this.update(cx, |this, cx| this.install_game(path, cx));
        })
        .detach();
    }

    /// Copies the chosen NUP or dump into Echora's data folder on another
    /// thread, then switches to the installed copy.
    fn install_game(&mut self, source: PathBuf, cx: &mut Context<Self>) {
        if self.running || self.installing.is_some() {
            return;
        }
        let title = match read_game(&source) {
            Ok(game) => game.title,
            Err(message) => return self.notify(message, cx),
        };
        let Some(games) = install::data_dir().map(|dir| dir.join("game")) else {
            return self.notify("There's no data folder to install the game into", cx);
        };
        let progress = Arc::new(install::Progress::default());
        let worker = progress.clone();
        std::thread::spawn(move || install::install(&source, &games, &worker));
        self.installing = Some((title, progress.clone()));
        self.sheet = None;
        cx.notify();
        cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor()
                    .timer(Duration::from_millis(150))
                    .await;
                let result = progress.take_result();
                let done = result.is_some();
                let updated = this.update(cx, |this, cx| {
                    if let Some(result) = result {
                        this.installing = None;
                        match result {
                            Ok(installed) => this.select_game(installed, cx),
                            Err(error) => {
                                this.notify(format!("Couldn't install the game: {error}"), cx)
                            }
                        }
                    }
                    cx.notify();
                });
                if done || updated.is_err() {
                    break;
                }
            }
        })
        .detach();
    }

    fn play(&mut self, _: &Play, _: &mut Window, cx: &mut Context<Self>) {
        if self.running || self.sheet.is_some() {
            return;
        }
        let Some(game) = &self.game else {
            return;
        };
        let Some(mut command) = game_command(&game.path, self.settings_path.as_deref()) else {
            self.notify("The echora command wasn't found next to the launcher", cx);
            return;
        };
        let child = match command.spawn() {
            Ok(child) => child,
            Err(error) => {
                self.notify(format!("Couldn't start the game: {error}"), cx);
                return;
            }
        };
        self.running = true;
        self.notify("Starting…", cx);
        cx.spawn(async move |this, cx| {
            let succeeded = wait_for(child, cx).await;
            let _ = this.update(cx, |this, cx| {
                this.running = false;
                if !succeeded {
                    this.notify("The game stopped. The run report has the details.", cx);
                }
                cx.notify();
            });
        })
        .detach();
    }

    fn toggle_settings(&mut self, _: &ToggleSettings, _: &mut Window, cx: &mut Context<Self>) {
        self.sheet = match self.sheet {
            Some(Sheet::Settings) => None,
            _ => Some(Sheet::Settings),
        };
        cx.notify();
    }

    fn close_sheet(&mut self, _: &CloseSheet, _: &mut Window, cx: &mut Context<Self>) {
        self.sheet = None;
        cx.notify();
    }

    fn change(&mut self, cx: &mut Context<Self>, edit: impl FnOnce(&mut Settings)) {
        edit(&mut self.settings);
        match &self.settings_path {
            Some(path) => {
                if let Err(error) = self.settings.save(path) {
                    self.notify(format!("Settings not saved: {error}"), cx);
                }
            }
            None => self.notify("No configuration folder, so settings aren't saved", cx),
        }
        cx.notify();
    }
}

/// Polls the game process without blocking the window.
async fn wait_for(mut child: Child, cx: &mut gpui::AsyncApp) -> bool {
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return status.success(),
            Ok(None) => {
                cx.background_executor()
                    .timer(Duration::from_millis(500))
                    .await
            }
            Err(_) => return false,
        }
    }
}

fn read_game(path: &Path) -> Result<Game, String> {
    let inventory = ContentSource::open(path)
        .and_then(|source| source.inventory())
        .map_err(|error| format!("That isn't a usable game: {error}"))?;
    let package = inventory
        .program_package
        .map(|index| &inventory.packages[index])
        .ok_or("That selection has no game program in it")?;
    let version = match &package.display_version {
        Some(display) => format!("Version {display}"),
        None => format!(
            "Version {}.{}.{}",
            package.version >> 16,
            (package.version >> 8) & 0xff,
            package.version & 0xff
        ),
    };
    let icon = package.icon.as_ref().map(|bytes| {
        let format = if bytes.starts_with(b"\x89PNG") {
            ImageFormat::Png
        } else {
            ImageFormat::Jpeg
        };
        Arc::new(Image::from_bytes(format, bytes.clone()))
    });
    Ok(Game {
        path: path.to_path_buf(),
        title: package
            .title_name
            .clone()
            .unwrap_or_else(|| "Unknown game".into())
            .into(),
        version: version.into(),
        icon,
    })
}

/// The launcher backdrop: a heavily blurred frame from the game's
/// `Dm_Title_0045` title video (see NOTICE.md).
const BACKDROP: &[u8] = include_bytes!("../assets/backdrop.jpg");

/// Decodes an image into the BGRA layout GPUI uploads.
fn decode_art(bytes: &[u8]) -> Option<image::RgbaImage> {
    let mut art = image::load_from_memory(bytes).ok()?.into_rgba8();
    for pixel in art.as_chunks_mut::<4>().0 {
        pixel.swap(0, 2);
    }
    Some(art)
}

fn last_game_file() -> Option<PathBuf> {
    echora_config::default_path().map(|settings| settings.with_file_name("last-game"))
}

/// `0xRRGGBBAA`.
fn color(hex: u32) -> Hsla {
    rgba(hex).into()
}

/// A secondary button in the dock or the settings sheet.
fn button(id: &'static str, label: impl Into<SharedString>) -> gpui::Stateful<gpui::Div> {
    div()
        .id(id)
        .h(px(44.0))
        .px(px(16.0))
        .flex()
        .items_center()
        .justify_center()
        .rounded(px(10.0))
        .bg(color(palette::CONTROL))
        .hover(|style| style.bg(color(palette::CONTROL_HOVER)))
        .cursor_pointer()
        .text_sm()
        .text_color(color(palette::TEXT))
        .child(label.into())
}

fn about_button(cx: &mut Context<Launcher>) -> gpui::Stateful<gpui::Div> {
    button("about", "About").on_click(cx.listener(|this, _, _, cx| {
        this.sheet = Some(Sheet::About);
        cx.notify();
    }))
}

fn play_button(id: &'static str, label: &'static str) -> gpui::Stateful<gpui::Div> {
    div()
        .id(id)
        .h(px(44.0))
        .px(px(26.0))
        .flex()
        .items_center()
        .rounded(px(10.0))
        .bg(color(palette::PLAY))
        .hover(|style| style.bg(color(palette::PLAY_HOVER)))
        .cursor_pointer()
        .text_color(color(palette::PLAY_TEXT))
        .font_weight(FontWeight::SEMIBOLD)
        .child(label)
}

/// A round title bar button in the style GNOME uses.
fn window_button(button: WindowButton, maximized: bool) -> gpui::Stateful<gpui::Div> {
    div()
        .id(button.id())
        .size(px(24.0))
        .flex()
        .items_center()
        .justify_center()
        .rounded_full()
        .bg(color(palette::WINDOW_BUTTON))
        .hover(|style| style.bg(color(palette::WINDOW_BUTTON_HOVER)))
        .child(
            svg()
                .path(button.icon(maximized))
                .size(px(16.0))
                .text_color(color(palette::TEXT)),
        )
}

/// A row of mutually exclusive choices.
fn segmented<T: Copy + PartialEq + 'static>(
    id: &'static str,
    choices: &[(&'static str, T)],
    current: T,
    cx: &mut Context<Launcher>,
    apply: fn(&mut Settings, T),
) -> impl IntoElement {
    div()
        .flex()
        .p(px(3.0))
        .gap(px(2.0))
        .rounded(px(9.0))
        .bg(color(palette::CONTROL))
        .children(choices.iter().enumerate().map(|(index, &(label, value))| {
            let selected = value == current;
            div()
                .id((id, index))
                .px(px(12.0))
                .py(px(5.0))
                .rounded(px(7.0))
                .text_sm()
                .cursor_pointer()
                .when(selected, |el| {
                    el.bg(color(palette::SELECTED))
                        .text_color(color(palette::SELECTED_TEXT))
                })
                .when(!selected, |el| {
                    el.text_color(color(palette::TEXT_MUTED))
                        .hover(|style| style.text_color(color(palette::TEXT)))
                })
                .child(label)
                .on_click(cx.listener(move |this, _, _, cx| {
                    this.change(cx, |settings| apply(settings, value));
                }))
        }))
}

/// One setting: its name and what it does on the left, the control on the
/// right.
fn setting_row(
    title: &'static str,
    description: Option<&'static str>,
    control: impl IntoElement,
) -> impl IntoElement {
    div()
        .flex()
        .items_center()
        .justify_between()
        .gap(px(24.0))
        .py(px(14.0))
        .child(
            div()
                .flex()
                .flex_col()
                .flex_1()
                .min_w_0()
                .gap(px(2.0))
                .child(
                    div()
                        .text_sm()
                        .text_color(color(palette::TEXT))
                        .child(title),
                )
                .when_some(description, |el, description| {
                    el.child(
                        div()
                            .text_xs()
                            .text_color(color(palette::TEXT_MUTED))
                            .child(description),
                    )
                }),
        )
        .child(div().flex_none().child(control))
}

/// The people and projects Echora builds on, with what each contributed.
const CREDITS: [(&str, &str); 6] = [
    ("Andrew Slaiman (Mars7x)", "Creator of Echora"),
    ("PokeDinner, an Eglantine fork", "Shader recompiler"),
    ("sirit", "SPIR-V generation"),
    ("GPUI by Zed Industries", "Launcher interface"),
    ("a55uka/nup", "NUP format specification"),
    ("Switchbrew, libnx, Atmosphère", "Format documentation"),
];

fn about_page() -> Vec<AnyElement> {
    let mut page = vec![
        div()
            .pt(px(8.0))
            .pb(px(12.0))
            .text_sm()
            .text_color(color(palette::TEXT_MUTED))
            .child(
                "A native recompilation of The Legend of Zelda: Ocarina of Time for \
                 Nintendo Switch 2. You need your own copy of the game.",
            )
            .into_any_element(),
        divider().into_any_element(),
        div()
            .pt(px(12.0))
            .pb(px(2.0))
            .text_xs()
            .font_weight(FontWeight::MEDIUM)
            .text_color(color(palette::TEXT_MUTED))
            .child("CREDITS")
            .into_any_element(),
    ];
    page.extend(CREDITS.iter().map(|&(name, role)| {
        div()
            .flex()
            .justify_between()
            .gap(px(16.0))
            .py(px(3.0))
            .text_sm()
            .child(div().text_color(color(palette::TEXT)).child(name))
            .child(div().text_color(color(palette::TEXT_MUTED)).child(role))
            .into_any_element()
    }));
    page
}

/// Dims the launcher behind a sheet; clicking outside the sheet closes it.
fn scrim(
    sheet: gpui::Stateful<gpui::Div>,
    corner: Pixels,
    cx: &mut Context<Launcher>,
) -> impl IntoElement {
    div()
        .id("scrim")
        .absolute()
        .inset_0()
        .rounded(corner)
        // Keeps sheets inside the window on small or scaled displays.
        .p(px(24.0))
        .flex()
        .items_center()
        .justify_center()
        .bg(color(palette::SCRIM))
        .on_click(cx.listener(|this, _, window, cx| this.close_sheet(&CloseSheet, window, cx)))
        // Clicks inside the sheet must not reach the scrim behind it.
        .child(sheet.on_click(|_, _, cx| cx.stop_propagation()))
}

/// A thin line between settings rows.
fn divider() -> impl IntoElement {
    div().h(px(1.0)).bg(color(palette::DIVIDER))
}

impl Launcher {
    fn render_art(&mut self, window: &mut Window) -> AnyElement {
        let Some(art) = self.game.as_ref().and(self.art.as_ref()) else {
            return div()
                .size_full()
                .rounded(self.corner)
                .bg(color(palette::BACKDROP))
                .into_any_element();
        };
        // Fill the window's shape: trim the sides of the wide art when the
        // window is narrower, or its bottom when the window is wider.
        let inset = window.client_inset().unwrap_or(px(0.0)) * 2.0;
        let viewport = window.viewport_size();
        let aspect =
            f32::from(viewport.width - inset) / f32::from(viewport.height - inset).max(1.0);
        let (width, height) = art.dimensions();
        let crop = if aspect >= width as f32 / height as f32 {
            (
                0,
                0,
                width,
                ((width as f32 / aspect) as u32).clamp(1, height),
            )
        } else {
            let crop_width = ((height as f32 * aspect) as u32).clamp(1, width);
            ((width - crop_width) / 2, 0, crop_width, height)
        };
        if self
            .backdrop
            .as_ref()
            .is_none_or(|backdrop| backdrop.crop != crop)
        {
            let (x, y, w, h) = crop;
            let pixels = image::imageops::crop_imm(art, x, y, w, h).to_image();
            let image = Arc::new(RenderImage::new([image::Frame::new(pixels)]));
            if let Some(old) = self.backdrop.replace(Backdrop { crop, image }) {
                let _ = window.drop_image(old.image);
            }
        }
        let Some(backdrop) = &self.backdrop else {
            return div().into_any_element();
        };
        div()
            .size_full()
            .relative()
            .child(
                img(backdrop.image.clone())
                    .size_full()
                    .rounded(self.corner)
                    .object_fit(ObjectFit::Fill),
            )
            .child(
                div()
                    .absolute()
                    .inset_0()
                    .rounded(self.corner)
                    .bg(color(palette::ART_SHADE)),
            )
            .into_any_element()
    }

    fn render_top_bar(&self, window: &Window, cx: &mut Context<Self>) -> impl IntoElement {
        let client_side = matches!(window.window_decorations(), Decorations::Client { .. });
        let maximized = window.is_maximized();
        let buttons = |side: &[WindowButton]| {
            div()
                .flex()
                .gap(px(10.0))
                .children(side.iter().map(|&button| {
                    window_button(button, maximized).on_click(move |_, window, _| match button {
                        WindowButton::Minimize => window.minimize_window(),
                        WindowButton::Maximize => window.zoom_window(),
                        WindowButton::Close => window.remove_window(),
                    })
                }))
        };
        div()
            .id("title-bar")
            .absolute()
            .top_0()
            .left_0()
            .right_0()
            .h(px(52.0))
            .px(px(14.0))
            .flex()
            .items_center()
            .gap(px(12.0))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _, _, _| this.moving = true),
            )
            .on_mouse_up(
                MouseButton::Left,
                cx.listener(|this, _, _, _| this.moving = false),
            )
            .on_mouse_move(cx.listener(|this, event: &MouseMoveEvent, window, _| {
                if this.moving && event.dragging() {
                    this.moving = false;
                    window.start_window_move();
                }
            }))
            .on_click(|event, window, _| {
                if event.is_right_click() {
                    window.show_window_menu(event.position());
                } else if event.click_count() == 2 {
                    window.zoom_window();
                }
            })
            .when(client_side, |el| el.child(buttons(&self.buttons.left)))
            .child(
                div()
                    .flex_1()
                    .px(px(8.0))
                    .when(cfg!(target_os = "macos"), |el| el.pl(px(64.0)))
                    .text_xs()
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(color(palette::TEXT_MUTED))
                    // The empty screen already shows the name in large type.
                    .when(self.game.is_some(), |el| el.child("ECHORA")),
            )
            .when(client_side, |el| el.child(buttons(&self.buttons.right)))
    }

    fn render_dock(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let dock = div()
            .flex()
            .items_center()
            .gap(px(6.0))
            .p(px(8.0))
            .rounded(px(16.0))
            .bg(color(palette::DOCK));
        if let Some((title, progress)) = &self.installing {
            let fraction = progress.fraction().clamp(0.0, 1.0);
            return dock.px(px(18.0)).py(px(14.0)).child(
                div()
                    .w(px(360.0))
                    .flex()
                    .flex_col()
                    .gap(px(8.0))
                    .child(
                        div()
                            .flex()
                            .justify_between()
                            .text_sm()
                            .child(
                                div()
                                    .text_color(color(palette::TEXT))
                                    .truncate()
                                    .child(format!("Installing {title}")),
                            )
                            .child(
                                div()
                                    .text_color(color(palette::TEXT_MUTED))
                                    .child(format!("{}%", (fraction * 100.0) as u32)),
                            ),
                    )
                    .child(
                        div()
                            .h(px(4.0))
                            .rounded_full()
                            .bg(color(palette::CONTROL))
                            .child(
                                div()
                                    .h_full()
                                    .rounded_full()
                                    .bg(color(palette::TEXT))
                                    .w(relative(fraction)),
                            ),
                    ),
            );
        }
        let Some(game) = &self.game else {
            return dock
                .child(play_button("choose", "Install game").on_click(
                    cx.listener(|this, _, window, cx| this.choose_game(&ChooseGame, window, cx)),
                ))
                .child(button("settings", "Settings").on_click(cx.listener(
                    |this, _, window, cx| this.toggle_settings(&ToggleSettings, window, cx),
                )))
                .child(about_button(cx));
        };
        let thumbnail = match &game.icon {
            Some(icon) => img(icon.clone())
                .size(px(44.0))
                .rounded(px(10.0))
                .object_fit(ObjectFit::Cover)
                .into_any_element(),
            None => div()
                .size(px(44.0))
                .rounded(px(10.0))
                .bg(color(palette::CONTROL))
                .into_any_element(),
        };
        dock.child(thumbnail)
            .child(
                div()
                    .pl(px(6.0))
                    .pr(px(18.0))
                    .max_w(px(320.0))
                    .flex()
                    .flex_col()
                    .child(
                        div()
                            .text_color(color(palette::TEXT))
                            .font_weight(FontWeight::MEDIUM)
                            .truncate()
                            .child(game.title.clone()),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(color(palette::TEXT_MUTED))
                            .child(game.version.clone()),
                    ),
            )
            .child(
                play_button("play", if self.running { "Running" } else { "Play" })
                    .when(self.running, |el| el.opacity(0.6).cursor_default())
                    .on_click(cx.listener(|this, _, window, cx| this.play(&Play, window, cx))),
            )
            .child(button("settings", "Settings").on_click(
                cx.listener(|this, _, window, cx| {
                    this.toggle_settings(&ToggleSettings, window, cx)
                }),
            ))
            .child(about_button(cx))
    }

    fn render_settings(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let settings = &self.settings;
        let page: Vec<AnyElement> = match self.settings_tab {
            SettingsTab::Game => self.game_page(cx),
            SettingsTab::Display => vec![
                setting_row(
                    "Window mode",
                    None,
                    segmented(
                        "mode",
                        &WINDOW_MODES,
                        settings.display.window_mode,
                        cx,
                        |s, mode| s.display.window_mode = mode,
                    ),
                )
                .into_any_element(),
                divider().into_any_element(),
                setting_row(
                    "Internal resolution",
                    None,
                    segmented(
                        "resolution",
                        &RESOLUTIONS,
                        settings.display.internal_resolution,
                        cx,
                        |s, resolution| s.display.internal_resolution = resolution,
                    ),
                )
                .into_any_element(),
            ],
            SettingsTab::Controls => vec![
                setting_row(
                    "Vibration",
                    None,
                    segmented(
                        "vibration",
                        &[("Off", false), ("On", true)],
                        settings.input.vibration,
                        cx,
                        |s, on| s.input.vibration = on,
                    ),
                )
                .into_any_element(),
            ],
            SettingsTab::Launcher => vec![
                setting_row(
                    "Skip launcher",
                    Some("Run echora-launcher --show to come back here"),
                    segmented(
                        "skip",
                        &[("Off", false), ("On", true)],
                        settings.launcher.skip,
                        cx,
                        |s, on| s.launcher.skip = on,
                    ),
                )
                .into_any_element(),
            ],
        };
        let tabs = div()
            .w(px(150.0))
            .flex()
            .flex_col()
            .gap(px(2.0))
            .p(px(12.0))
            .border_r_1()
            .border_color(color(palette::DIVIDER))
            .child(
                div()
                    .px(px(10.0))
                    .pt(px(4.0))
                    .pb(px(14.0))
                    .text_lg()
                    .font_weight(FontWeight::MEDIUM)
                    .text_color(color(palette::TEXT))
                    .child("Settings"),
            )
            .children(SettingsTab::ALL.into_iter().map(|tab| {
                let selected = tab == self.settings_tab;
                div()
                    .id(tab.label())
                    .px(px(10.0))
                    .py(px(8.0))
                    .rounded(px(8.0))
                    .text_sm()
                    .cursor_pointer()
                    .when(selected, |el| {
                        el.bg(color(palette::CONTROL))
                            .text_color(color(palette::TEXT))
                    })
                    .when(!selected, |el| {
                        el.text_color(color(palette::TEXT_MUTED))
                            .hover(|style| style.text_color(color(palette::TEXT)))
                    })
                    .child(tab.label())
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.settings_tab = tab;
                        cx.notify();
                    }))
            }));
        let content = div()
            .id("settings-page")
            .overflow_y_scroll()
            .flex_1()
            .min_w_0()
            .flex()
            .flex_col()
            .px(px(24.0))
            .py(px(16.0))
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .pb(px(6.0))
                    .child(
                        div()
                            .font_weight(FontWeight::MEDIUM)
                            .text_color(color(palette::TEXT))
                            .child(self.settings_tab.label()),
                    )
                    .child(button("done", "Done").h(px(32.0)).on_click(cx.listener(
                        |this, _, window, cx| this.close_sheet(&CloseSheet, window, cx),
                    ))),
            )
            .children(page);
        let sheet = div()
            .id("sheet")
            .w_full()
            .max_w(px(760.0))
            .h(px(420.0))
            .max_h_full()
            .flex()
            .rounded(px(16.0))
            .border_1()
            .border_color(color(palette::DIVIDER))
            .bg(color(palette::SHEET))
            .child(tabs)
            .child(content);
        scrim(sheet, self.corner, cx)
    }

    fn render_about(&self, window: &Window, cx: &mut Context<Self>) -> impl IntoElement {
        // Room left for the credits once the window frame, the sheet's
        // margins, header and footer are taken out.
        let inset = window.client_inset().unwrap_or(px(0.0)) * 2.0;
        let body = (window.viewport_size().height - inset - px(190.0)).max(px(80.0));
        let sheet = div()
            .id("about")
            // A fixed width (the window is at least 720 wide) so wrapped
            // text is measured at its real width.
            .w(px(560.0))
            .flex()
            .flex_col()
            .px(px(24.0))
            .py(px(16.0))
            .rounded(px(16.0))
            .border_1()
            .border_color(color(palette::DIVIDER))
            .bg(color(palette::SHEET))
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .child(
                        div()
                            .flex()
                            .items_baseline()
                            .gap(px(8.0))
                            .child(
                                div()
                                    .font_weight(FontWeight::MEDIUM)
                                    .text_color(color(palette::TEXT))
                                    .child("Echora"),
                            )
                            .child(
                                div()
                                    .text_sm()
                                    .text_color(color(palette::TEXT_MUTED))
                                    .child(env!("CARGO_PKG_VERSION")),
                            ),
                    )
                    .child(button("done", "Done").h(px(32.0)).on_click(cx.listener(
                        |this, _, window, cx| this.close_sheet(&CloseSheet, window, cx),
                    ))),
            )
            .child(
                div()
                    .id("about-body")
                    // Wrapped text is measured late; never let the credits
                    // shrink to make room for it, only cap them below.
                    .flex_none()
                    .max_h(body)
                    .overflow_y_scroll()
                    .children(about_page()),
            )
            .child(
                div()
                    .pt(px(12.0))
                    .flex()
                    .items_center()
                    .justify_between()
                    .gap(px(16.0))
                    .child(
                        div()
                            .text_xs()
                            .text_color(color(palette::TEXT_MUTED))
                            .child("GPL-3.0-or-later. Not affiliated with Nintendo."),
                    )
                    .child(button("source", "Source code").h(px(32.0)).on_click(
                        cx.listener(|_, _, _, cx| cx.open_url("https://github.com/Mars7x/echora")),
                    )),
            );
        scrim(sheet, self.corner, cx)
    }

    /// The chosen game and the way to change it.
    fn game_page(&self, cx: &mut Context<Self>) -> Vec<AnyElement> {
        let summary = match &self.game {
            Some(game) => div()
                .flex()
                .items_center()
                .gap(px(14.0))
                .child(match &game.icon {
                    Some(icon) => img(icon.clone())
                        .size(px(56.0))
                        .rounded(px(12.0))
                        .object_fit(ObjectFit::Cover)
                        .into_any_element(),
                    None => div()
                        .size(px(56.0))
                        .rounded(px(12.0))
                        .bg(color(palette::CONTROL))
                        .into_any_element(),
                })
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .min_w_0()
                        .gap(px(2.0))
                        .child(
                            div()
                                .text_sm()
                                .text_color(color(palette::TEXT))
                                .child(game.title.clone()),
                        )
                        .child(
                            div()
                                .text_xs()
                                .text_color(color(palette::TEXT_MUTED))
                                .child(game.version.clone()),
                        ),
                ),
            None => div()
                .text_sm()
                .text_color(color(palette::TEXT_MUTED))
                .child("No game chosen yet"),
        };
        let installed_in = install::data_dir().map(|dir| dir.join("game"));
        let original = self
            .game
            .as_ref()
            .filter(|game| {
                installed_in
                    .as_ref()
                    .is_none_or(|dir| !game.path.starts_with(dir))
            })
            .map(|game| game.path.clone());
        let choose = cx.listener(|this: &mut Self, _: &gpui::ClickEvent, window, cx| {
            this.choose_game(&ChooseGame, window, cx)
        });
        let (note, action) = match (&self.game, original) {
            (None, _) => (None, button("install", "Install game").on_click(choose)),
            (Some(_), None) => (
                Some("Installed. Your original NUP or dump is no longer needed."),
                button("install", "Reinstall").on_click(choose),
            ),
            (Some(_), Some(path)) => (
                Some("Played from its original location. Install it to keep a copy in Echora."),
                button("install", "Install into Echora").on_click(
                    cx.listener(move |this, _, _, cx| this.install_game(path.clone(), cx)),
                ),
            ),
        };
        vec![
            div()
                .pt(px(10.0))
                .pb(px(12.0))
                .child(summary)
                .into_any_element(),
            div()
                .when_some(note, |el, note| {
                    el.pb(px(14.0))
                        .text_xs()
                        .text_color(color(palette::TEXT_MUTED))
                        .child(note)
                })
                .into_any_element(),
            div().flex().child(action.h(px(36.0))).into_any_element(),
        ]
    }
}

/// Space around the window for the shadow and the resize handles.
const WINDOW_INSET: f32 = 16.0;
const WINDOW_RADIUS: f32 = 14.0;

impl Render for Launcher {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let Decorations::Client { tiling } = window.window_decorations() else {
            self.corner = px(0.0);
            return self.render_content(window, cx).into_any_element();
        };
        let floating = !tiling.is_tiled() && !window.is_maximized();
        let inset = px(WINDOW_INSET);
        window.set_client_inset(if floating { inset } else { px(0.0) });
        self.corner = if floating { px(WINDOW_RADIUS) } else { px(0.0) };
        let content = self
            .render_content(window, cx)
            .rounded(self.corner)
            // The edge highlight sits on top of the art rather than as a
            // border, which would inset the art and show a dark frame.
            .when(floating, |el| {
                el.child(
                    div()
                        .absolute()
                        .inset_0()
                        .rounded(self.corner)
                        .border_1()
                        .border_color(color(palette::WINDOW_BORDER)),
                )
                // Faint, like the desktop's own window shadows: a soft
                // drop and a hairline that separates the edge.
                .shadow(vec![
                    BoxShadow {
                        color: Hsla::black().opacity(0.18),
                        blur_radius: px(10.0),
                        spread_radius: px(0.0),
                        offset: point(px(0.0), px(2.0)),
                    },
                    BoxShadow {
                        color: Hsla::black().opacity(0.12),
                        blur_radius: px(1.0),
                        spread_radius: px(0.0),
                        offset: point(px(0.0), px(0.0)),
                    },
                ])
            });
        div()
            .id("window")
            .size_full()
            .when(floating, |el| {
                el.p(inset)
                    .child(resize_cursor(inset))
                    .on_mouse_move(|_, window, _| window.refresh())
                    .on_mouse_down(MouseButton::Left, move |event, window, _| {
                        let size = window.window_bounds().get_bounds().size;
                        if let Some(edge) = chrome::resize_edge(event.position, inset, size) {
                            window.start_window_resize(edge);
                        }
                    })
            })
            .child(content)
            .into_any_element()
    }
}

/// Shows the resize cursor over the window's edges.
fn resize_cursor(inset: Pixels) -> impl IntoElement {
    canvas(
        |_, window, _| {
            window.insert_hitbox(
                Bounds::new(
                    point(px(0.0), px(0.0)),
                    window.window_bounds().get_bounds().size,
                ),
                HitboxBehavior::Normal,
            )
        },
        move |_, hitbox, window, _| {
            let size = window.window_bounds().get_bounds().size;
            let Some(edge) = chrome::resize_edge(window.mouse_position(), inset, size) else {
                return;
            };
            let cursor = match edge {
                ResizeEdge::Top | ResizeEdge::Bottom => CursorStyle::ResizeUpDown,
                ResizeEdge::Left | ResizeEdge::Right => CursorStyle::ResizeLeftRight,
                ResizeEdge::TopLeft | ResizeEdge::BottomRight => CursorStyle::ResizeUpLeftDownRight,
                ResizeEdge::TopRight | ResizeEdge::BottomLeft => CursorStyle::ResizeUpRightDownLeft,
            };
            window.set_cursor_style(cursor, &hitbox);
        },
    )
    .size_full()
    .absolute()
}

impl Launcher {
    fn render_content(&mut self, window: &mut Window, cx: &mut Context<Self>) -> gpui::Div {
        div()
            .key_context("Launcher")
            .track_focus(&self.focus)
            .on_action(cx.listener(Self::play))
            .on_action(cx.listener(Self::toggle_settings))
            .on_action(cx.listener(Self::close_sheet))
            .on_action(cx.listener(Self::choose_game))
            .size_full()
            .relative()
            .overflow_hidden()
            // No background of its own: the art layer paints the whole window,
            // and a second layer under its rounded corners would show at the
            // antialiased edge.
            .child(self.render_art(window))
            .when(self.game.is_none(), |el| {
                el.child(
                    div()
                        .absolute()
                        .inset_0()
                        .flex()
                        .flex_col()
                        .items_center()
                        .justify_center()
                        .pb(px(80.0))
                        .child(
                            div()
                                .text_3xl()
                                .font_weight(FontWeight::BOLD)
                                .text_color(color(palette::TEXT))
                                .child("Echora"),
                        )
                        .child(
                            div()
                                .pt(px(6.0))
                                .text_color(color(palette::TEXT_MUTED))
                                .child(
                                    "Choose your NUP package or dump folder to install the game",
                                ),
                        ),
                )
            })
            .child(self.render_top_bar(window, cx))
            .child(
                div()
                    .absolute()
                    .left_0()
                    .right_0()
                    .bottom(px(24.0))
                    .flex()
                    .flex_col()
                    .items_center()
                    .gap(px(10.0))
                    .when_some(self.notice.clone(), |el, notice| {
                        el.child(
                            div()
                                .px(px(14.0))
                                .py(px(7.0))
                                .rounded(px(10.0))
                                .bg(color(palette::DOCK))
                                .text_sm()
                                .text_color(color(palette::TEXT))
                                .child(notice),
                        )
                    })
                    .child(self.render_dock(cx)),
            )
            .when_some(self.sheet, |el, sheet| {
                el.child(match sheet {
                    Sheet::Settings => self.render_settings(cx).into_any_element(),
                    Sheet::About => self.render_about(window, cx).into_any_element(),
                })
            })
    }
}
