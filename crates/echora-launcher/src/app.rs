//! The launcher window: the game's art fills the window and a single dock at
//! the bottom holds what the player needs, with settings in a sheet above it.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitCode};
use std::sync::Arc;
use std::time::Duration;

use echora_config::{InternalResolutionSetting, Settings, WindowMode};
use echora_content::ContentSource;
use gpui::{
    AnyElement, App, Application, Bounds, Context, Decorations, FocusHandle, FontWeight, Hsla,
    Image, ImageFormat, KeyBinding, MouseButton, ObjectFit, PathPromptOptions, SharedString,
    TitlebarOptions, Window, WindowBounds, WindowDecorations, WindowOptions, actions, div, img,
    point, prelude::*, px, rgba, size,
};

const APP_ID: &str = "io.github.mars7x.Echora";

/// Internal resolution choices offered in settings.
const RESOLUTIONS: [(&str, InternalResolutionSetting); 5] = [
    ("Window", InternalResolutionSetting::MatchOutput),
    ("720p", InternalResolutionSetting::Height { height: 720 }),
    ("1080p", InternalResolutionSetting::Height { height: 1080 }),
    ("1440p", InternalResolutionSetting::Height { height: 1440 }),
    ("4K", InternalResolutionSetting::Height { height: 2160 }),
];

const WINDOW_MODES: [(&str, WindowMode); 3] = [
    ("Windowed", WindowMode::Windowed),
    ("Borderless", WindowMode::BorderlessFullscreen),
    ("Fullscreen", WindowMode::ExclusiveFullscreen),
];

actions!(
    echora,
    [Play, ToggleSettings, CloseSettings, ChooseGame, Quit]
);

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
    pub const SCRIM: u32 = 0x000000a6;
    pub const ART_SHADE: u32 = 0x0000004d;
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SettingsTab {
    Display,
    Controls,
    Launcher,
}

impl SettingsTab {
    const ALL: [Self; 3] = [Self::Display, Self::Controls, Self::Launcher];

    fn label(self) -> &'static str {
        match self {
            Self::Display => "Display",
            Self::Controls => "Controls",
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

struct Launcher {
    focus: FocusHandle,
    game: Option<Game>,
    settings: Settings,
    settings_path: Option<PathBuf>,
    settings_open: bool,
    settings_tab: SettingsTab,
    notice: Option<SharedString>,
    notice_generation: u64,
    running: bool,
}

pub fn run() -> ExitCode {
    // `--show` opens the window even when the player chose to skip it.
    let show = std::env::args().skip(1).any(|arg| arg == "--show");
    if !show && let Some(code) = skip_to_game() {
        return code;
    }
    Application::new().run(|cx: &mut App| {
        cx.bind_keys([
            KeyBinding::new("enter", Play, Some("Launcher")),
            KeyBinding::new("ctrl-,", ToggleSettings, Some("Launcher")),
            KeyBinding::new("cmd-,", ToggleSettings, Some("Launcher")),
            KeyBinding::new("escape", CloseSettings, Some("Launcher")),
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
            ..Default::default()
        };
        let opened = cx.open_window(options, |window, cx| {
            cx.new(|cx| {
                let launcher = Launcher::new(cx);
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
    fn new(cx: &mut Context<Self>) -> Self {
        let (settings_path, settings) = load_settings();
        let mut launcher = Self {
            focus: cx.focus_handle(),
            game: None,
            settings,
            settings_path,
            settings_open: false,
            settings_tab: SettingsTab::Display,
            notice: None,
            notice_generation: 0,
            running: false,
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
            let _ = this.update(cx, |this, cx| this.select_game(path, cx));
        })
        .detach();
    }

    fn play(&mut self, _: &Play, _: &mut Window, cx: &mut Context<Self>) {
        if self.running || self.settings_open {
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
        self.settings_open = !self.settings_open;
        cx.notify();
    }

    fn close_settings(&mut self, _: &CloseSettings, _: &mut Window, cx: &mut Context<Self>) {
        self.settings_open = false;
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

fn window_control(id: &'static str, glyph: &'static str) -> gpui::Stateful<gpui::Div> {
    div()
        .id(id)
        .size(px(26.0))
        .flex()
        .items_center()
        .justify_center()
        .rounded_full()
        .bg(color(palette::DOCK))
        .hover(|style| style.bg(color(palette::CONTROL_HOVER)))
        .cursor_pointer()
        .text_sm()
        .text_color(color(palette::TEXT))
        .child(glyph)
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
    description: &'static str,
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
                .child(
                    div()
                        .text_xs()
                        .text_color(color(palette::TEXT_MUTED))
                        .child(description),
                ),
        )
        .child(div().flex_none().child(control))
}

/// A thin line between settings rows.
fn divider() -> impl IntoElement {
    div().h(px(1.0)).bg(color(palette::DIVIDER))
}

impl Launcher {
    fn render_art(&self, window: &Window) -> AnyElement {
        let Some(icon) = self.game.as_ref().and_then(|game| game.icon.clone()) else {
            return div()
                .size_full()
                .bg(color(palette::BACKDROP))
                .into_any_element();
        };
        div()
            .size_full()
            .relative()
            // The icon is square: scale it to the window's width and keep its
            // top, where the title artwork sits, in view.
            .child(
                img(icon)
                    .absolute()
                    .top_0()
                    .left_0()
                    .w_full()
                    .h(window.viewport_size().width)
                    .object_fit(ObjectFit::Cover),
            )
            .child(div().absolute().inset_0().bg(color(palette::ART_SHADE)))
            .into_any_element()
    }

    fn render_top_bar(&self, window: &Window) -> impl IntoElement {
        let client_side = matches!(window.window_decorations(), Decorations::Client { .. });
        div()
            .absolute()
            .top_0()
            .left_0()
            .right_0()
            .h(px(52.0))
            .px(px(22.0))
            .flex()
            .items_center()
            .justify_between()
            .on_mouse_down(MouseButton::Left, |_, window, _| window.start_window_move())
            .child(
                div()
                    .when(cfg!(target_os = "macos"), |el| el.pl(px(64.0)))
                    .text_xs()
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(color(palette::TEXT_MUTED))
                    .child("ECHORA"),
            )
            .when(client_side && !cfg!(target_os = "macos"), |el| {
                el.child(
                    div()
                        .flex()
                        .gap(px(8.0))
                        .child(
                            window_control("minimize", "–")
                                .on_click(|_, window, _| window.minimize_window()),
                        )
                        .child(
                            window_control("close", "×")
                                .on_click(|_, window, _| window.remove_window()),
                        ),
                )
            })
    }

    fn render_dock(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let dock = div()
            .flex()
            .items_center()
            .gap(px(6.0))
            .p(px(8.0))
            .rounded(px(16.0))
            .bg(color(palette::DOCK));
        let Some(game) = &self.game else {
            return dock
                .child(play_button("choose", "Choose game").on_click(
                    cx.listener(|this, _, window, cx| this.choose_game(&ChooseGame, window, cx)),
                ))
                .child(button("settings", "Settings").on_click(cx.listener(
                    |this, _, window, cx| this.toggle_settings(&ToggleSettings, window, cx),
                )));
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
            .child(button("change", "Change game").on_click(
                cx.listener(|this, _, window, cx| this.choose_game(&ChooseGame, window, cx)),
            ))
    }

    fn render_settings(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let settings = &self.settings;
        let page: Vec<AnyElement> = match self.settings_tab {
            SettingsTab::Display => vec![
                setting_row(
                    "Window mode",
                    "How the game's window is shown",
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
                    "The resolution scenes are drawn at",
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
                    "Rumble on controllers that support it",
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
                    "Start the game straight away. Run echora-launcher --show to come back here.",
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
            }))
            .child(div().flex_1())
            .child(
                div()
                    .px(px(10.0))
                    .text_xs()
                    .text_color(color(palette::TEXT_MUTED))
                    .child(format!("Echora {}", env!("CARGO_PKG_VERSION"))),
            );
        let content = div()
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
                            .text_xs()
                            .font_weight(FontWeight::MEDIUM)
                            .text_color(color(palette::TEXT_MUTED))
                            .child(self.settings_tab.label().to_uppercase()),
                    )
                    .child(button("done", "Done").h(px(32.0)).on_click(cx.listener(
                        |this, _, window, cx| this.close_settings(&CloseSettings, window, cx),
                    ))),
            )
            .children(page)
            .child(div().flex_1())
            .when(self.settings_tab != SettingsTab::Launcher, |el| {
                el.child(
                    div()
                        .text_xs()
                        .text_color(color(palette::TEXT_MUTED))
                        .child("Changes apply the next time the game starts."),
                )
            });
        let sheet = div()
            .id("sheet")
            .w_full()
            .max_w(px(760.0))
            .mx(px(24.0))
            .h(px(380.0))
            .flex()
            .rounded(px(16.0))
            .border_1()
            .border_color(color(palette::DIVIDER))
            .bg(color(palette::SHEET))
            // Clicks inside the sheet must not reach the scrim behind it.
            .on_click(|_, _, cx| cx.stop_propagation())
            .child(tabs)
            .child(content);
        div()
            .id("scrim")
            .absolute()
            .inset_0()
            .flex()
            .items_center()
            .justify_center()
            .bg(color(palette::SCRIM))
            .on_click(
                cx.listener(|this, _, window, cx| this.close_settings(&CloseSettings, window, cx)),
            )
            .child(sheet)
    }
}

impl Render for Launcher {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .key_context("Launcher")
            .track_focus(&self.focus)
            .on_action(cx.listener(Self::play))
            .on_action(cx.listener(Self::toggle_settings))
            .on_action(cx.listener(Self::close_settings))
            .on_action(cx.listener(Self::choose_game))
            .size_full()
            .relative()
            .overflow_hidden()
            .bg(color(palette::BACKDROP))
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
                                .child("Choose your NUP package or dump folder to begin"),
                        ),
                )
            })
            .child(self.render_top_bar(window))
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
            .when(self.settings_open, |el| el.child(self.render_settings(cx)))
    }
}
