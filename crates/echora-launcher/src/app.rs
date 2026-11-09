//! The launcher window: the game's title large in the bottom left with Play,
//! Settings and About under it, and settings and about in sheets above.
//! Everything visual comes from the design system in [`crate::ui`].

use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitCode};
use std::sync::Arc;
use std::time::{Duration, Instant};

use echora_config::{InternalResolutionSetting, Settings, WindowMode, Workers};
use echora_content::ContentSource;
use echora_input::button;
use gpui::{
    Action, Animation, AnimationExt as _, AnyElement, App, Application, Bounds, BoxShadow, Context,
    CursorStyle, Decorations, Div, ElementId, FocusHandle, FontWeight, HitboxBehavior, Hsla, Image,
    ImageFormat, KeyBinding, KeyDownEvent, MouseButton, MouseDownEvent, MouseMoveEvent, ObjectFit,
    PathPromptOptions, Pixels, ResizeEdge, SharedString, Stateful, TitlebarOptions, Window,
    WindowBackgroundAppearance, WindowBounds, WindowDecorations, WindowOptions, actions, anchored,
    canvas, deferred, div, img, point, prelude::*, px, size,
};

use crate::chrome::{self, ButtonLayout, WindowButton};
use crate::install;
use crate::nav::{Activate, Direction, Press as _, Stops};
use crate::prepare;
use crate::ui::{self, Kind, Size, color, palette, radius, space, text, u};

const APP_ID: &str = "io.github.mars7x.Echora";

/// Internal resolution choices offered in settings.
const RESOLUTIONS: [(&str, InternalResolutionSetting); 5] = [
    ("Window", InternalResolutionSetting::MatchOutput),
    ("720p", InternalResolutionSetting::Height { height: 720 }),
    ("1080p", InternalResolutionSetting::Height { height: 1080 }),
    ("1440p", InternalResolutionSetting::Height { height: 1440 }),
    ("2160p", InternalResolutionSetting::Height { height: 2160 }),
];

/// How many CPU cores run the game: automatic (up to six), or a count.
const CPU_CORES: [(&str, Workers); 7] = [
    ("Auto", Workers::Auto),
    ("1", Workers::Count(1)),
    ("2", Workers::Count(2)),
    ("3", Workers::Count(3)),
    ("4", Workers::Count(4)),
    ("5", Workers::Count(5)),
    ("6", Workers::Count(6)),
];

const WINDOW_MODES: [(&str, WindowMode); 3] = [
    ("Windowed", WindowMode::Windowed),
    ("Borderless", WindowMode::BorderlessFullscreen),
    ("Fullscreen", WindowMode::ExclusiveFullscreen),
];

actions!(
    echora,
    [
        Play,
        ToggleSettings,
        CloseSheet,
        ChooseGame,
        FocusNext,
        FocusPrev,
        FocusLeft,
        FocusUp,
        FocusRight,
        FocusDown,
        NextTab,
        PrevTab,
        Quit
    ]
);

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

    /// The tab `step` places after this one, wrapping around.
    fn step(self, step: isize) -> Self {
        let index = Self::ALL.iter().position(|&tab| tab == self).unwrap_or(0) as isize;
        Self::ALL[(index + step).rem_euclid(Self::ALL.len() as isize) as usize]
    }

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
    /// Title id and version of the program package.
    build: Option<(String, u32)>,
    title: SharedString,
    version: SharedString,
    icon: Option<Arc<Image>>,
}

struct Launcher {
    /// The launcher as a whole, holding focus when no control does, so its
    /// shortcuts keep working.
    focus: FocusHandle,
    /// The controls the keyboard and controllers reach.
    stops: Stops,
    /// Focus moves to the default control on the next frame, after the one
    /// that held it went away.
    refocus: bool,
    /// The control focused when a sheet opened, which gets focus back when
    /// it closes.
    return_focus: Option<FocusHandle>,
    /// The keyboard or a controller is being used to move focus, so focus
    /// rings show.
    keyboard: bool,
    /// The desktop allows animations.
    motion: bool,
    font: SharedString,
    game: Option<Game>,
    settings: Settings,
    settings_path: Option<PathBuf>,
    sheet: Option<Sheet>,
    /// Holds focus while a sheet is open and none of its controls does, so
    /// Escape always reaches it.
    sheet_focus: FocusHandle,
    /// A sheet fading out after being closed.
    closing: Option<Sheet>,
    /// The settings pop-up menu that is open, by id.
    open_menu: Option<&'static str>,
    /// A menu folding away after being closed, and which closing it is.
    closing_menu: Option<(&'static str, u64)>,
    menu_closes: u64,
    /// The menu closed by the current mouse press, so the release on its
    /// own button doesn't open it again.
    menu_dismissed: Option<&'static str>,
    /// The switch just flipped; only it animates.
    toggled: Option<&'static str>,
    /// Window buttons the desktop asks for, when decorations are ours.
    buttons: ButtonLayout,
    /// Corner radius of the window this frame (zero when maximized or tiled).
    corner: Pixels,
    /// Where a header press began, which turns into a window move once the
    /// pointer drags past [`DRAG_THRESHOLD`].
    press: Option<gpui::Point<Pixels>>,
    /// The desktop is moving the window. It may report the window inactive
    /// meanwhile, which mustn't dim the window buttons.
    dragging: bool,
    /// The resize edge under the pointer, if any.
    resize_edge: Option<ResizeEdge>,
    /// The remembered game is still being read.
    loading: bool,
    settings_tab: SettingsTab,
    notice: Option<SharedString>,
    notice_generation: u64,
    running: bool,
    /// The game being installed and how far the copy has got.
    installing: Option<(SharedString, Arc<install::Progress>)>,
    /// The game being prepared to run (translated and built), before its
    /// first start.
    preparing: Option<Arc<prepare::Preparation>>,
    /// Whether the selected game starts right away; `None` until checked.
    prepared: Option<bool>,
}

pub fn run() -> ExitCode {
    // `--show` opens the window even when the player chose to skip it.
    let show = std::env::args().skip(1).any(|arg| arg == "--show");
    if !show && let Some(code) = skip_to_game() {
        return code;
    }
    let buttons = ButtonLayout::from_desktop();
    let motion = ui::motion_enabled();
    Application::new()
        .with_assets(chrome::Icons)
        .run(move |cx: &mut App| {
            cx.bind_keys([
                KeyBinding::new("tab", FocusNext, Some("Launcher")),
                KeyBinding::new("shift-tab", FocusPrev, Some("Launcher")),
                KeyBinding::new("left", FocusLeft, Some("Launcher")),
                KeyBinding::new("up", FocusUp, Some("Launcher")),
                KeyBinding::new("right", FocusRight, Some("Launcher")),
                KeyBinding::new("down", FocusDown, Some("Launcher")),
                KeyBinding::new("ctrl-tab", NextTab, Some("Launcher")),
                KeyBinding::new("ctrl-shift-tab", PrevTab, Some("Launcher")),
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
                    appears_transparent: false,
                    traffic_light_position: None,
                }),
                app_id: Some(APP_ID.into()),
                window_min_size: Some(size(px(720.0), px(460.0))),
                window_decorations: Some(WindowDecorations::Client),
                // Transparent so client-side decorations can round the corners
                // and draw the shadow.
                window_background: WindowBackgroundAppearance::Transparent,
                ..Default::default()
            };
            let font = ui::load_font(cx);
            let opened = cx.open_window(options, |window, cx| {
                cx.new(|cx| {
                    let launcher = Launcher::new(buttons, motion, font, window, cx);
                    window.focus(&launcher.default_focus(cx));
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
/// means the window should open instead: also when the game fails, so the
/// player isn't left with nothing on screen.
fn skip_to_game() -> Option<ExitCode> {
    let (settings_path, settings) = load_settings();
    if !settings.launcher.skip {
        return None;
    }
    let game = last_game()?;
    // A game that still needs building opens the launcher, which shows it.
    if !prepare::is_prepared(prepare::command(&echora_binary()?, &game, true)) {
        return None;
    }
    let mut command = game_command(&game, settings_path.as_deref())?;
    match command.status() {
        Ok(status) if status.success() => Some(ExitCode::SUCCESS),
        Ok(status) => {
            eprintln!("echora-launcher: the game stopped ({status}); opening the launcher");
            None
        }
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

/// The `echora` binary next to the launcher.
fn echora_binary() -> Option<PathBuf> {
    std::env::current_exe()
        .ok()
        .map(|exe| exe.with_file_name(format!("echora{}", std::env::consts::EXE_SUFFIX)))
        .filter(|path| path.is_file())
}

/// `echora run GAME` using the `echora` binary next to the launcher, with
/// no instruction budget: `run` stops test runs after a billion guest
/// instructions by default (a few seconds), but a player's game runs until
/// they close it.
fn game_command(game: &Path, settings: Option<&Path>) -> Option<Command> {
    let mut command = Command::new(echora_binary()?);
    command.args(run_arguments(game));
    game_environment(&mut command, settings, |name| {
        std::env::var_os(name).is_some()
    });
    Some(command)
}

/// `run GAME --budget MAX`: the largest budget never runs out.
fn run_arguments(game: &Path) -> [std::ffi::OsString; 4] {
    [
        "run".into(),
        game.into(),
        "--budget".into(),
        u64::MAX.to_string().into(),
    ]
}

/// The game's window and sound are on, as a player starting it expects,
/// unless the launcher's own environment already chooses (`is_set`); the
/// runtime leaves both off without them, for headless runs.
fn game_environment(command: &mut Command, settings: Option<&Path>, is_set: impl Fn(&str) -> bool) {
    if let Some(settings) = settings {
        command.env("ECHORA_SETTINGS", settings);
    }
    for name in ["ECHORA_WINDOW", "ECHORA_AUDIO"] {
        if !is_set(name) {
            command.env(name, "1");
        }
    }
}

impl Launcher {
    fn new(
        buttons: ButtonLayout,
        motion: bool,
        font: SharedString,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let (settings_path, settings) = load_settings();
        let mut launcher = Self {
            focus: cx.focus_handle(),
            stops: Stops::default(),
            refocus: false,
            return_focus: None,
            keyboard: false,
            motion,
            font,
            game: None,
            settings,
            settings_path,
            sheet: None,
            sheet_focus: cx.focus_handle(),
            closing: None,
            open_menu: None,
            closing_menu: None,
            menu_closes: 0,
            menu_dismissed: None,
            toggled: None,
            buttons,
            corner: px(0.0),
            press: None,
            dragging: false,
            resize_edge: None,
            loading: false,
            settings_tab: SettingsTab::Game,
            notice: None,
            notice_generation: 0,
            running: false,
            installing: None,
            preparing: None,
            prepared: None,
        };
        // Read the remembered game off the UI thread so the window opens at
        // once; the title fills in when the game has been read.
        if let Some(last) = last_game() {
            launcher.loading = true;
            let task = cx
                .background_executor()
                .spawn(async move { read_game(&last) });
            cx.spawn_in(window, async move |this, cx| {
                let game = task.await;
                let _ = this.update_in(cx, |this, window, cx| {
                    this.loading = false;
                    match game {
                        Ok(game) => {
                            this.game = Some(game);
                            this.check_prepared(cx);
                        }
                        Err(message) => this.notify(message, cx),
                    }
                    if this.sheet.is_none() {
                        window.focus(&this.default_focus(cx));
                    }
                    cx.notify();
                });
            })
            .detach();
        }
        watch_controllers(window, cx);
        cx.observe_window_activation(window, |this, window, cx| {
            if window.is_window_active() {
                this.dragging = false;
            }
            cx.notify();
        })
        .detach();
        launcher
    }

    /// The control focus rests on: Play, or Choose game on the empty
    /// screen, so Enter (or A) does the main thing at once. While a sheet is
    /// open, the sheet; otherwise the launcher itself.
    fn default_focus(&self, cx: &App) -> FocusHandle {
        if self.sheet.is_some() {
            self.sheet_focus.clone()
        } else if self.installing.is_some() || self.preparing.is_some() || self.loading {
            self.focus.clone()
        } else if self.game.is_some() {
            self.stops.handle("play", cx)
        } else {
            self.stops.handle("choose", cx)
        }
    }

    /// Makes `el`, a button, a control the keyboard and controllers reach
    /// while `live`. Controls behind a sheet or an open menu aren't live.
    fn stop(&self, el: Stateful<Div>, live: bool, cx: &App) -> Stateful<Div> {
        self.stop_rounded(el, radius::CONTROL, live, cx)
    }

    /// [`Self::stop`] for a control rounded other than a button is, which
    /// the focus ring follows.
    fn stop_rounded(&self, el: Stateful<Div>, corner: f32, live: bool, cx: &App) -> Stateful<Div> {
        if !live {
            return el;
        }
        self.stops.add(el, corner, cx)
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
                self.refocus = self.sheet.is_none();
                self.check_prepared(cx);
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
        if self.running || self.installing.is_some() || self.preparing.is_some() {
            return;
        }
        let title = match read_game(&source).and_then(supported_build) {
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
        // The sheet's controls go with it, so focus moves to the launcher.
        self.sheet = None;
        self.open_menu = None;
        self.return_focus = None;
        self.refocus = true;
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
                        this.refocus = this.sheet.is_none();
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
        if self.running || self.sheet.is_some() || self.preparing.is_some() {
            return;
        }
        let Some(game) = &self.game else {
            return;
        };
        let Some(echora) = echora_binary() else {
            self.notify("The echora command wasn't found next to the launcher", cx);
            return;
        };
        if !prepare::is_prepared(prepare::command(&echora, &game.path, true)) {
            self.prepare_game(&echora, cx);
        } else {
            self.start_game(cx);
        }
    }

    /// Asks in the background whether the selected game starts right away,
    /// so its first start can say it will build first.
    fn check_prepared(&mut self, cx: &mut Context<Self>) {
        self.prepared = None;
        let (Some(game), Some(echora)) = (&self.game, echora_binary()) else {
            return;
        };
        let command = prepare::command(&echora, &game.path, true);
        let path = game.path.clone();
        let task = cx
            .background_executor()
            .spawn(async move { prepare::is_prepared(command) });
        cx.spawn(async move |this, cx| {
            let prepared = task.await;
            let _ = this.update(cx, |this, cx| {
                // Only if the same game is still selected.
                if this.game.as_ref().is_some_and(|game| game.path == path) {
                    this.prepared = Some(prepared);
                    cx.notify();
                }
            });
        })
        .detach();
    }

    /// Translates and builds the selected game with `echora prepare`,
    /// showing its progress, then starts it.
    fn prepare_game(&mut self, echora: &Path, cx: &mut Context<Self>) {
        let Some(game) = &self.game else {
            return;
        };
        let preparation = Arc::new(prepare::Preparation::default());
        if let Err(error) = preparation.start(prepare::command(echora, &game.path, false)) {
            self.notify(format!("Couldn't prepare the game: {error}"), cx);
            return;
        }
        self.preparing = Some(preparation.clone());
        // Play goes away while it builds, so focus moves to the launcher.
        self.open_menu = None;
        self.refocus = true;
        cx.notify();
        cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor()
                    .timer(Duration::from_millis(250))
                    .await;
                let result = preparation.take_result();
                let done = result.is_some();
                let updated = this.update(cx, |this, cx| {
                    if let Some(result) = result {
                        this.preparing = None;
                        this.refocus = this.sheet.is_none();
                        match result {
                            Ok(()) => {
                                this.prepared = Some(true);
                                this.start_game(cx);
                            }
                            Err(error) => {
                                this.prepared = Some(false);
                                this.notify(format!("Couldn't prepare the game: {error}"), cx);
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

    fn start_game(&mut self, cx: &mut Context<Self>) {
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

    fn toggle_settings(&mut self, _: &ToggleSettings, window: &mut Window, cx: &mut Context<Self>) {
        if self.sheet == Some(Sheet::Settings) {
            self.close_sheet(&CloseSheet, window, cx);
        } else {
            self.open_sheet(Sheet::Settings, window, cx);
        }
    }

    fn open_sheet(&mut self, sheet: Sheet, window: &mut Window, cx: &mut Context<Self>) {
        if self.sheet.is_none() {
            self.return_focus = window.focused(cx);
        }
        self.sheet = Some(sheet);
        self.closing = None;
        self.open_menu = None;
        self.toggled = None;
        // Focus goes straight to the sheet's first control: the open
        // settings tab, or About's one button.
        let first = match sheet {
            Sheet::Settings => self.stops.handle(self.settings_tab.label(), cx),
            Sheet::About => self.stops.handle("source", cx),
        };
        window.focus(&first);
        cx.notify();
    }

    /// Closes the open menu, which folds away, and returns its id.
    fn close_menu(&mut self, cx: &mut Context<Self>) -> Option<&'static str> {
        let menu = self.open_menu.take()?;
        if self.motion {
            self.menu_closes += 1;
            let close = (menu, self.menu_closes);
            self.closing_menu = Some(close);
            cx.spawn(async move |this, cx| {
                cx.background_executor().timer(ui::MENU_OUT).await;
                let _ = this.update(cx, |this, cx| {
                    if this.closing_menu == Some(close) {
                        this.closing_menu = None;
                        cx.notify();
                    }
                });
            })
            .detach();
        }
        cx.notify();
        Some(menu)
    }

    /// Escape, B and the close buttons: closes an open menu first, then the
    /// sheet, which fades out. Focus goes back to what opened it.
    fn close_sheet(&mut self, _: &CloseSheet, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(menu) = self.close_menu(cx) {
            window.focus(&self.stops.handle((menu, 0usize), cx));
            cx.notify();
            return;
        }
        let Some(sheet) = self.sheet.take() else {
            return;
        };
        if self.motion {
            self.closing = Some(sheet);
            cx.spawn(async move |this, cx| {
                cx.background_executor().timer(SHEET_OUT).await;
                let _ = this.update(cx, |this, cx| {
                    if this.closing == Some(sheet) {
                        this.closing = None;
                        cx.notify();
                    }
                });
            })
            .detach();
        }
        let target = self
            .return_focus
            .take()
            .unwrap_or_else(|| self.default_focus(cx));
        window.focus(&target);
        cx.notify();
    }

    fn focus_next(&mut self, _: &FocusNext, window: &mut Window, cx: &mut Context<Self>) {
        self.keyboard = true;
        window.focus_next();
        cx.notify();
    }

    fn focus_prev(&mut self, _: &FocusPrev, window: &mut Window, cx: &mut Context<Self>) {
        self.keyboard = true;
        window.focus_prev();
        cx.notify();
    }

    /// The arrow keys and a controller's D-pad and stick: focus moves to
    /// the nearest control that way.
    fn move_focus(&mut self, direction: Direction, window: &mut Window, cx: &mut Context<Self>) {
        self.keyboard = true;
        if let Some(target) = self.stops.toward(window.focused(cx).as_ref(), direction) {
            window.focus(&target);
        }
        cx.notify();
    }

    /// Ctrl+Tab and the shoulder buttons: the next or previous settings
    /// tab, while Settings is open.
    fn step_tab(&mut self, step: isize, window: &mut Window, cx: &mut Context<Self>) {
        if self.sheet != Some(Sheet::Settings) {
            return;
        }
        self.settings_tab = self.settings_tab.step(step);
        self.open_menu = None;
        self.toggled = None;
        // The focused control may have been on the old page.
        window.focus(&self.stops.handle(self.settings_tab.label(), cx));
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

/// How often the controllers are read.
const PAD_POLL: Duration = Duration::from_millis(16);
/// A held direction repeats after this long, then every [`PAD_REPEAT`].
const PAD_REPEAT_DELAY: Duration = Duration::from_millis(400);
const PAD_REPEAT: Duration = Duration::from_millis(110);

/// The D-pad and the left stick, as directions.
const PAD_DIRECTIONS: [(u64, Direction); 8] = [
    (button::LEFT, Direction::Left),
    (button::UP, Direction::Up),
    (button::RIGHT, Direction::Right),
    (button::DOWN, Direction::Down),
    (button::STICK_L_LEFT, Direction::Left),
    (button::STICK_L_LEFT << 1, Direction::Up),
    (button::STICK_L_LEFT << 2, Direction::Right),
    (button::STICK_L_LEFT << 3, Direction::Down),
];

/// Drives the launcher from any connected controller, as the keyboard
/// does, with the Switch's layout the game uses: the D-pad or left stick
/// moves focus, A presses the focused control (as Enter), B goes back (as
/// Escape), + opens Settings, and L and R (or ZL and ZR) change settings
/// tab. Controllers are ignored while the game runs or another window is
/// active.
fn watch_controllers(window: &mut Window, cx: &mut Context<Launcher>) {
    let pads = echora_input::gamepads();
    cx.spawn_in(window, async move |this, cx| {
        let mut held = 0u64;
        // The direction being held, and when it next repeats.
        let mut repeat: Option<(Direction, Instant)> = None;
        loop {
            cx.background_executor().timer(PAD_POLL).await;
            let Ok(listening) = this.update_in(cx, |this, window, _| {
                !this.running && window.is_window_active()
            }) else {
                break;
            };
            // Only presses count, so a button still held from the game
            // does nothing when the launcher comes back.
            let buttons = pads.state().buttons;
            let pressed = if listening { buttons & !held } else { 0 };
            held = buttons;
            if !listening {
                repeat = None;
                continue;
            }

            let now = Instant::now();
            let mut moves = Vec::new();
            if let Some(&(_, direction)) = PAD_DIRECTIONS.iter().find(|(bit, _)| pressed & bit != 0)
            {
                moves.push(direction);
                repeat = Some((direction, now + PAD_REPEAT_DELAY));
            } else if let Some((direction, at)) = repeat {
                let still_held = PAD_DIRECTIONS
                    .iter()
                    .any(|&(bit, d)| d == direction && held & bit != 0);
                if !still_held {
                    repeat = None;
                } else if now >= at {
                    moves.push(direction);
                    repeat = Some((direction, now + PAD_REPEAT));
                }
            }
            let mut actions: Vec<Box<dyn Action>> = moves
                .into_iter()
                .map(|direction| -> Box<dyn Action> {
                    match direction {
                        Direction::Left => Box::new(FocusLeft),
                        Direction::Up => Box::new(FocusUp),
                        Direction::Right => Box::new(FocusRight),
                        Direction::Down => Box::new(FocusDown),
                    }
                })
                .collect();
            if pressed & button::B != 0 {
                actions.push(Box::new(CloseSheet));
            }
            if pressed & button::PLUS != 0 {
                actions.push(Box::new(ToggleSettings));
            }
            if pressed & (button::L | button::ZL) != 0 {
                actions.push(Box::new(PrevTab));
            }
            if pressed & (button::R | button::ZR) != 0 {
                actions.push(Box::new(NextTab));
            }
            if pressed & button::A != 0 {
                actions.push(Box::new(Activate));
            }
            if actions.is_empty() {
                continue;
            }
            // A controller in use shows the focus ring, as keys do.
            let shown = this.update(cx, |this, cx| {
                if !this.keyboard {
                    this.keyboard = true;
                    cx.notify();
                }
            });
            if shown.is_err() {
                break;
            }
            // Dispatched outside the launcher's own update, as key presses
            // are, so the controls' handlers can update it.
            let dispatched = cx.update(|window, cx| {
                for action in actions {
                    window.dispatch_action(action, cx);
                }
            });
            if dispatched.is_err() {
                break;
            }
        }
    })
    .detach();
}

/// The one game build Echora translates: Ocarina of Time's update 1.1.1.
const SUPPORTED_TITLE: &str = "0400486024DEC000";
const SUPPORTED_VERSION: u32 = 131072;

/// `game`, if it is the build Echora supports; other games and versions are
/// refused before anything is installed.
fn supported_build(game: Game) -> Result<Game, String> {
    match &game.build {
        Some((title, version)) if title == SUPPORTED_TITLE && *version == SUPPORTED_VERSION => {
            Ok(game)
        }
        Some((title, _)) if title == SUPPORTED_TITLE => Err(format!(
            "This is {} of the game; Echora needs version 1.1.1",
            game.version.to_lowercase()
        )),
        _ => Err(
            "Echora supports The Legend of Zelda: Ocarina of Time for Nintendo Switch 2 only"
                .into(),
        ),
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
        build: package
            .title_id
            .clone()
            .map(|title| (title.to_ascii_uppercase(), package.version)),
        title: package
            .title_name
            .clone()
            .unwrap_or_else(|| "Unknown game".into())
            .into(),
        version: version.into(),
        icon,
    })
}

/// Splits a title into its series and the game's own name, for titles such
/// as "The Legend of Zelda Ocarina of Time" or "Series: Name".
fn split_title(title: &str) -> (Option<SharedString>, SharedString) {
    const SERIES: &str = "The Legend of Zelda";
    if let Some(rest) = title.strip_prefix(SERIES) {
        let rest = rest.trim_start_matches([':', ' ']);
        if !rest.is_empty() {
            return (Some(SERIES.into()), rest.to_owned().into());
        }
    }
    match title.split_once(": ") {
        Some((series, name)) => (Some(series.to_owned().into()), name.to_owned().into()),
        None => (None, title.to_owned().into()),
    }
}

fn last_game_file() -> Option<PathBuf> {
    echora_config::default_path().map(|settings| settings.with_file_name("last-game"))
}

/// The people and projects Echora builds on, with what each contributed.
const CREDITS: [(&str, &str); 6] = [
    ("Andrew Slaiman (Mars7x)", "Creator of Echora"),
    ("Lichii, an Eglantine fork", "Shader recompiler"),
    ("sirit", "SPIR-V generation"),
    ("GPUI by Zed Industries", "Launcher interface"),
    ("a55uka/nup", "NUP format specification"),
    ("Switchbrew, libnx, Atmosphère", "Format documentation"),
];

/// Space around the window for the shadow and the resize handles.
const WINDOW_INSET: f32 = 16.0;
/// Margin between a sheet and the window's edges.
const SHEET_MARGIN: f32 = 24.0;

/// How long a closing sheet takes to fade out.
const SHEET_OUT: Duration = Duration::from_millis(180);

/// Dims the launcher behind a sheet; clicking outside the sheet closes it.
/// A closing sheet fades out instead of in and takes no input.
fn scrim(
    sheet: gpui::Stateful<gpui::Div>,
    corner: Pixels,
    motion: bool,
    closing: bool,
    cx: &mut Context<Launcher>,
) -> AnyElement {
    let sheet = sheet.on_click(|_, _, cx| cx.stop_propagation());
    let backdrop = div()
        .id(if closing { "scrim-closing" } else { "scrim" })
        .absolute()
        .inset_0()
        .rounded(corner)
        .flex()
        .items_center()
        .justify_center()
        .bg(color(palette::SCRIM))
        .when(!closing, |el| {
            el.on_click(
                cx.listener(|this, _, window, cx| this.close_sheet(&CloseSheet, window, cx)),
            )
        })
        // Clicks inside the sheet must not reach the scrim behind it.
        .child(if motion {
            fade(sheet, closing)
        } else {
            sheet.into_any_element()
        });
    if !motion {
        return backdrop.into_any_element();
    }
    if closing {
        return backdrop
            .with_animation(
                "scrim-out",
                Animation::new(SHEET_OUT),
                // Leaving needs no bounce: a quick fade.
                |el, t| el.opacity(1.0 - ui::ease_out(t)),
            )
            .into_any_element();
    }
    backdrop
        .with_animation(
            "scrim-in",
            Animation::new(Duration::from_millis(220)),
            |el, t| el.opacity(ui::ease_out(t)),
        )
        .into_any_element()
}

/// Opening fades the sheet in with the dim behind it; closing fades it out
/// a little faster. Neither scales: GPUI can't scale an element without its
/// text and images snapping to the pixel grid frame by frame.
fn fade(sheet: gpui::Stateful<gpui::Div>, closing: bool) -> AnyElement {
    let duration = if closing {
        SHEET_OUT
    } else {
        Duration::from_millis(200)
    };
    sheet
        .with_animation(
            if closing { "sheet-out" } else { "sheet-in" },
            Animation::new(duration),
            move |sheet, t| {
                sheet.opacity(if closing {
                    1.0 - ui::ease_out(t)
                } else {
                    ui::ease_out(t)
                })
            },
        )
        .into_any_element()
}

/// Window pixels as rems, so a sheet sized from the window scales with the
/// rest of its contents.
fn to_rems(pixels: Pixels, rem: Pixels) -> gpui::Rems {
    gpui::rems(f32::from(pixels) / f32::from(rem))
}

/// A sheet's surface: the window's own background, at a fixed size so
/// wrapped text is measured at its real width.
fn sheet(
    id: &'static str,
    width: Pixels,
    height: Option<Pixels>,
    surface: u32,
    rem: Pixels,
) -> gpui::Stateful<gpui::Div> {
    div()
        .id(id)
        .w(to_rems(width, rem))
        .when_some(height, |el, height| el.h(to_rems(height, rem)))
        .flex()
        .rounded(u(radius::SHEET))
        .border_1()
        .border_color(color(palette::WINDOW_EDGE))
        .bg(color(surface))
}

impl Launcher {
    /// Room for a sheet inside the window.
    fn sheet_room(&self, window: &Window) -> (Pixels, Pixels) {
        let frame = window.client_inset().unwrap_or(px(0.0)) * 2.0 + px(SHEET_MARGIN * 2.0);
        let viewport = window.viewport_size();
        (viewport.width - frame, viewport.height - frame)
    }

    /// A sheet's close button. The mouse's way out: it takes no focus, as
    /// Escape and B close the sheet from anywhere in it.
    fn close_button(&self, cx: &mut Context<Self>) -> gpui::Stateful<gpui::Div> {
        ui::round_button("close-sheet", WindowButton::Close.icon(false), true)
            // Its click area reaches into the header's padding.
            .mr(px(-ui::ROUND_BUTTON_REACH))
            .on_click(cx.listener(|this, _, window, cx| this.close_sheet(&CloseSheet, window, cx)))
    }

    /// The window's plain background, rounded with the window.
    fn render_background(&self) -> impl IntoElement {
        div()
            .size_full()
            .rounded(self.corner)
            .bg(color(palette::BACKDROP))
    }

    fn render_title_bar(&self, window: &Window, cx: &mut Context<Self>) -> impl IntoElement {
        let client_side = matches!(window.window_decorations(), Decorations::Client { .. });
        let maximized = window.is_maximized();
        let active = window.is_window_active() || self.dragging;
        let buttons = |side: &[WindowButton]| {
            div()
                .flex()
                // The buttons' click areas fill the gap between them.
                .gap(px(space::M - 2.0 * ui::ROUND_BUTTON_REACH))
                .children(side.iter().map(|&button| {
                    ui::round_button(button.id(), button.icon(maximized), active)
                        // A press on a button must not start a window drag,
                        // nor a double-click maximize the window.
                        .on_mouse_down(MouseButton::Left, |_, _, cx| cx.stop_propagation())
                        .on_click(move |_, window, cx| {
                            cx.stop_propagation();
                            match button {
                                WindowButton::Minimize => window.minimize_window(),
                                WindowButton::Maximize => window.zoom_window(),
                                WindowButton::Close => window.remove_window(),
                            }
                        })
                }))
        };
        self.drag_window(
            div()
                .absolute()
                .top_0()
                .left_0()
                .right_0()
                .h(px(48.0))
                .px(px(space::M - ui::ROUND_BUTTON_REACH))
                .flex()
                .items_center()
                .justify_between()
                // The app's name, centred as a header bar's title is, where the
                // launcher draws its own header bar.
                .when(client_side, |el| {
                    el.child(
                        div()
                            .absolute()
                            .inset_0()
                            .flex()
                            .items_center()
                            .justify_center()
                            .font_family(ui::TITLE_FONT)
                            .text_size(u(text::HEADLINE))
                            .font_weight(FontWeight::BOLD)
                            .text_color(color(palette::LABEL))
                            .child("Echora"),
                    )
                })
                .child(div().when(client_side, |el| el.child(buttons(&self.buttons.left))))
                .child(div().when(client_side, |el| el.child(buttons(&self.buttons.right)))),
            cx,
        )
    }

    /// No game yet: a status page with the one thing to do.
    fn render_welcome(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let choose = self
            .stop(
                ui::button("choose", "Choose game…", Kind::Primary, Size::Large),
                self.sheet.is_none(),
                cx,
            )
            .on_press(cx.listener(|this, _, window, cx| this.choose_game(&ChooseGame, window, cx)));
        div()
            .absolute()
            .inset_0()
            .flex()
            .flex_col()
            .items_center()
            .justify_center()
            .gap(u(space::S))
            .pb(u(72.0))
            .child(
                div()
                    .text_size(u(text::LARGE_TITLE))
                    .font_weight(FontWeight::BOLD)
                    .text_color(color(palette::LABEL))
                    .child("Echora"),
            )
            .child(
                div()
                    .text_size(u(text::HEADLINE))
                    .text_color(color(palette::SECONDARY_LABEL))
                    .child("Choose your NUP package or dump folder to install the game."),
            )
            .child(div().pt(u(space::L)).child(choose))
    }

    /// Settings and About; the whole bottom bar on the empty screen, and
    /// beside Play on the game screen.
    fn secondary_actions(&self, cx: &mut Context<Self>) -> [gpui::Stateful<gpui::Div>; 2] {
        let active = self.sheet.is_none();
        let settings = self
            .stop(
                ui::button("settings", "Settings", Kind::Secondary, Size::Large).w(u(DOCK_BUTTON)),
                active,
                cx,
            )
            .on_press(
                cx.listener(|this, _, window, cx| {
                    this.toggle_settings(&ToggleSettings, window, cx)
                }),
            );
        let about = self
            .stop(
                ui::button("about", "About", Kind::Secondary, Size::Large).w(u(DOCK_BUTTON)),
                active,
                cx,
            )
            .on_press(cx.listener(|this, _, window, cx| this.open_sheet(Sheet::About, window, cx)));
        [settings, about]
    }

    /// The empty screen's bar: Settings and About.
    fn render_dock(&self, cx: &mut Context<Self>) -> AnyElement {
        div()
            .flex()
            .items_center()
            .gap(u(space::S))
            .p(u(space::S))
            .rounded(u(radius::DOCK))
            .border_1()
            .border_color(color(palette::BORDER))
            .bg(color(palette::MATERIAL))
            .children(self.secondary_actions(cx))
            .into_any_element()
    }

    /// The game screen: the title large in the bottom left, with Play,
    /// Settings and About under it (or the install's progress).
    fn render_hero(&self, cx: &mut Context<Self>) -> AnyElement {
        let (title, detail) = match (&self.installing, &self.game) {
            (Some((title, progress)), _) => (
                title.clone(),
                format!(
                    "Installing, {}%",
                    (progress.fraction().clamp(0.0, 1.0) * 100.0) as u32
                ),
            ),
            (None, Some(game)) => match (&self.preparing, self.prepared) {
                (Some(preparation), _) => (game.title.clone(), preparation.step().describe()),
                (None, Some(false)) => (
                    game.title.clone(),
                    format!(
                        "{} · Builds before its first start, about an hour",
                        game.version
                    ),
                ),
                _ => (game.title.clone(), game.version.to_string()),
            },
            (None, None) => return div().into_any_element(),
        };
        let (series, name) = split_title(&title);
        let progress = match (&self.installing, &self.preparing) {
            (Some((_, progress)), _) => Some(progress.fraction()),
            (None, Some(preparation)) => Some(preparation.step().fraction()),
            (None, None) => None,
        };
        let actions = if let Some(fraction) = progress {
            div()
                .w(u(320.0))
                .child(ui::progress(fraction))
                .into_any_element()
        } else {
            let active = self.sheet.is_none();
            let play = self
                .stop(
                    ui::button(
                        "play",
                        if self.running { "Running" } else { "Play" },
                        Kind::Primary,
                        Size::Large,
                    )
                    .w(u(DOCK_BUTTON)),
                    active,
                    cx,
                )
                .when(self.running, |el| el.opacity(0.6))
                .on_press(cx.listener(|this, _, window, cx| this.play(&Play, window, cx)));
            div()
                .flex()
                .gap(u(space::S))
                .child(play)
                .children(self.secondary_actions(cx))
                .into_any_element()
        };
        div()
            .flex()
            .flex_col()
            .when_some(series, |el, series| {
                el.child(
                    div()
                        .font_family(ui::TITLE_FONT)
                        .text_size(u(text::HEADLINE))
                        .text_color(color(palette::SECONDARY_LABEL))
                        .child(series),
                )
            })
            .child(
                div()
                    .font_family(ui::TITLE_FONT)
                    .text_size(u(text::HERO))
                    .line_height(gpui::relative(1.1))
                    .font_weight(FontWeight::BOLD)
                    .text_color(color(palette::LABEL))
                    .child(name),
            )
            .child(
                div()
                    .pt(u(space::XS + 2.0))
                    .text_size(u(text::CALLOUT))
                    .text_color(color(palette::SECONDARY_LABEL))
                    .child(detail),
            )
            .child(div().pt(u(space::XL)).child(actions))
            .into_any_element()
    }

    /// Makes `el` behave as a header bar does: dragging it moves the
    /// window, a double-click maximizes or restores it, and a right-click
    /// opens the window menu.
    fn drag_window(&self, el: gpui::Div, cx: &mut Context<Self>) -> gpui::Div {
        el.on_mouse_down(
            MouseButton::Left,
            cx.listener(|this, event: &MouseDownEvent, window, _| {
                if event.click_count == 2 {
                    window.zoom_window();
                } else {
                    this.press = Some(event.position);
                }
            }),
        )
        .on_mouse_down(MouseButton::Right, |event, window, _| {
            window.show_window_menu(event.position);
        })
        .on_mouse_up(
            MouseButton::Left,
            cx.listener(|this, _, _, _| this.press = None),
        )
        .on_mouse_move(cx.listener(|this, event: &MouseMoveEvent, window, _| {
            let Some(press) = this.press else {
                return;
            };
            if !event.dragging() {
                this.press = None;
            } else if (event.position - press).magnitude() >= f64::from(DRAG_THRESHOLD) {
                this.press = None;
                this.dragging = true;
                window.start_window_move();
            }
        }))
    }

    /// The settings sheet. A `closing` sheet is drawn for its fade-out only.
    fn render_settings(
        &self,
        window: &Window,
        closing: bool,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let (room_w, room_h) = self.sheet_room(window);
        let scale = window.rem_size() / px(16.0);
        let width = room_w.min(px(760.0) * scale);
        let height = room_h.min(px(460.0) * scale);
        // An open menu keeps focus to itself.
        let live = !closing && self.open_menu.is_none();
        let title = self.drag_window(
            div()
                .flex_none()
                .h(u(48.0))
                .px(u(space::L + space::XS))
                .flex()
                .items_center()
                .text_size(u(text::HEADLINE))
                .font_weight(FontWeight::SEMIBOLD)
                .text_color(color(palette::LABEL))
                .child("Settings"),
            cx,
        );
        // The sidebar's own colour, rounded on the sheet's left corners:
        // children are not clipped to the sheet's rounding.
        let tabs = div()
            .flex_none()
            .w(u(180.0))
            .flex()
            .flex_col()
            .rounded_l(u(radius::SHEET))
            .bg(color(palette::SIDEBAR))
            .child(title)
            .child(
                div()
                    .px(u(space::S))
                    .flex()
                    .flex_col()
                    .gap(u(2.0))
                    .children(SettingsTab::ALL.into_iter().map(|tab| {
                        let selected = tab == self.settings_tab;
                        self.stop(
                            div()
                                .id(tab.label())
                                .h(u(36.0))
                                .px(u(space::M))
                                .flex()
                                .items_center()
                                .rounded(u(radius::CONTROL))
                                .text_size(u(text::BODY))
                                .when(selected, |el| {
                                    el.bg(color(palette::SELECTED))
                                        .text_color(color(palette::LABEL))
                                })
                                .when(!selected, |el| {
                                    el.text_color(color(palette::SECONDARY_LABEL))
                                        .hover(|style| {
                                            style
                                                .bg(color(palette::FILL))
                                                .text_color(color(palette::LABEL))
                                        })
                                })
                                .child(tab.label()),
                            live,
                            cx,
                        )
                        .on_press(cx.listener(move |this, _, _, cx| {
                            this.settings_tab = tab;
                            this.open_menu = None;
                            this.toggled = None;
                            cx.notify();
                        }))
                    })),
            );
        let header = self
            .drag_window(
                div()
                    .flex_none()
                    .h(u(48.0))
                    .pl(u(space::XL))
                    .pr(u(space::M))
                    .flex()
                    .items_center()
                    .justify_between()
                    .child(
                        div()
                            .text_size(u(text::HEADLINE))
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(color(palette::LABEL))
                            .child(self.settings_tab.label()),
                    ),
                cx,
            )
            .child(self.close_button(cx));
        let page = div()
            .flex()
            .flex_col()
            .gap(u(space::XL))
            .children(self.settings_page(window, !closing, cx));
        let content = div()
            .flex_1()
            .min_w_0()
            .flex()
            .flex_col()
            .child(header)
            .child(
                div()
                    .id("settings-page")
                    .flex_1()
                    .min_h_0()
                    .overflow_y_scroll()
                    .px(u(space::XL))
                    .pt(u(space::XS))
                    .pb(u(space::XL))
                    .child(page),
            );
        let surface = palette::VIEW;
        let sheet = sheet("settings", width, Some(height), surface, window.rem_size())
            .when(!closing, |el| el.track_focus(&self.sheet_focus))
            .child(tabs)
            .child(content);
        scrim(sheet, self.corner, self.motion, closing, cx)
    }

    fn settings_page(
        &self,
        window: &Window,
        live: bool,
        cx: &mut Context<Self>,
    ) -> Vec<AnyElement> {
        let settings = &self.settings;
        let motion = self.motion;
        let toggled = self.toggled;
        // A setting that is on or off: the whole row flips it.
        let switch = |id: &'static str,
                      titles: gpui::Div,
                      on: bool,
                      flip: fn(&mut Settings),
                      cx: &mut Context<Self>| {
            self.stop_rounded(
                ui::switch_row(id, titles, on, motion && toggled == Some(id)),
                radius::CARD,
                live,
                cx,
            )
            .on_press(cx.listener(move |this, _, _, cx| {
                this.toggled = Some(id);
                this.change(cx, flip);
            }))
            .into_any_element()
        };
        match self.settings_tab {
            SettingsTab::Game => self.game_page(live, cx),
            SettingsTab::Display => vec![
                ui::group(
                    None,
                    vec![
                        ui::row(
                            None,
                            ui::titles("Window mode", None),
                            Some(self.popup(
                                "mode",
                                &WINDOW_MODES,
                                settings.display.window_mode,
                                window,
                                live,
                                cx,
                                |s, v| s.display.window_mode = v,
                            )),
                        )
                        .into_any_element(),
                        ui::row(
                            None,
                            ui::titles("Internal resolution", None),
                            Some(self.popup(
                                "resolution",
                                &RESOLUTIONS,
                                settings.display.internal_resolution,
                                window,
                                live,
                                cx,
                                |s, v| s.display.internal_resolution = v,
                            )),
                        )
                        .into_any_element(),
                        ui::row(
                            None,
                            ui::titles(
                                "CPU cores",
                                Some(
                                    "Auto uses up to six, as many as the console gives the game."
                                        .into(),
                                ),
                            ),
                            Some(self.popup(
                                "cores",
                                &CPU_CORES,
                                settings.cpu.workers,
                                window,
                                live,
                                cx,
                                |s, v| s.cpu.workers = v,
                            )),
                        )
                        .into_any_element(),
                    ],
                )
                .into_any_element(),
            ],
            SettingsTab::Controls => vec![
                ui::group(
                    None,
                    vec![switch(
                        "vibration",
                        ui::titles("Vibration", None),
                        settings.input.vibration,
                        |s| s.input.vibration = !s.input.vibration,
                        cx,
                    )],
                )
                .into_any_element(),
            ],
            SettingsTab::Launcher => vec![
                ui::group(
                    None,
                    vec![switch(
                        "skip",
                        ui::titles(
                            "Skip launcher",
                            Some(
                                "Start the game directly. Choose Open launcher from \
                                     Echora's app menu to come back."
                                    .into(),
                            ),
                        ),
                        settings.launcher.skip,
                        |s| s.launcher.skip = !s.launcher.skip,
                        cx,
                    )],
                )
                .into_any_element(),
            ],
        }
    }

    /// A pop-up button over `choices` that opens a menu of them; picking
    /// one saves it.
    #[allow(clippy::too_many_arguments)]
    fn popup<T: Copy + PartialEq + 'static>(
        &self,
        id: &'static str,
        choices: &'static [(&'static str, T)],
        current: T,
        window: &Window,
        live: bool,
        cx: &mut Context<Self>,
        apply: fn(&mut Settings, T),
    ) -> AnyElement {
        let open = live && self.open_menu == Some(id);
        let selected = choices
            .iter()
            .position(|(_, value)| *value == current)
            .unwrap_or(0);
        let label = choices[selected].0;
        let button = self
            .stop(
                ui::popup_button((id, 0usize), label),
                live && self.open_menu.is_none(),
                cx,
            )
            .on_press(cx.listener(move |this, _, window, cx| {
                if this.menu_dismissed.take() == Some(id) {
                    return;
                }
                if this.open_menu == Some(id) {
                    this.close_menu(cx);
                    window.focus(&this.stops.handle((id, 0usize), cx));
                } else {
                    // Focus starts on the current choice.
                    this.open_menu = Some(id);
                    window.focus(&this.stops.handle((id, selected + 1), cx));
                }
                cx.notify();
            }));
        // Where the menu hangs, below the button's right edge, above
        // everything else.
        let hang = |menu: AnyElement| {
            div().absolute().top(u(36.0)).right_0().child(
                deferred(
                    anchored()
                        .anchor(gpui::Corner::TopRight)
                        .snap_to_window_with_margin(px(space::S))
                        .child(menu),
                )
                .with_priority(2),
            )
        };
        if !open {
            // A menu just closed folds away, no longer taking input.
            let closing = self
                .closing_menu
                .filter(|&(menu, _)| live && menu == id)
                .map(|(_, close)| {
                    let menu =
                        ui::menu(choices.iter().enumerate().map(|(index, &(label, value))| {
                            ui::menu_item((id, index + 1), label, value == current)
                        }));
                    hang(ui::menu_closing(
                        menu,
                        choices.len(),
                        ElementId::Name(format!("{id}-close-{close}").into()),
                    ))
                });
            return div()
                .relative()
                .flex_none()
                .child(button)
                .children(closing)
                .into_any_element();
        }
        let menu = ui::menu(choices.iter().enumerate().map(|(index, &(label, value))| {
            self.stop_rounded(
                ui::menu_item((id, index + 1), label, value == current),
                radius::INNER,
                true,
                cx,
            )
            .on_press(cx.listener(move |this, _, window, cx| {
                this.close_menu(cx);
                window.focus(&this.stops.handle((id, 0usize), cx));
                this.change(cx, |settings| apply(settings, value));
            }))
        }));
        let viewport = window.viewport_size();
        div()
            .relative()
            .flex_none()
            .child(button)
            // A click anywhere outside the menu closes it.
            .child(
                deferred(
                    anchored().position(point(px(0.0), px(0.0))).child(
                        div()
                            .id((id, 100usize))
                            .w(viewport.width)
                            .h(viewport.height)
                            .on_mouse_down(
                                MouseButton::Left,
                                cx.listener(move |this, _, window, cx| {
                                    this.close_menu(cx);
                                    this.menu_dismissed = Some(id);
                                    window.focus(&this.stops.handle((id, 0usize), cx));
                                    cx.notify();
                                }),
                            ),
                    ),
                )
                .with_priority(1),
            )
            .child(hang(ui::menu_opening(
                menu,
                choices.len(),
                (id, 101usize),
                self.motion,
            )))
            .into_any_element()
    }

    /// The installed game, and installing or reinstalling it.
    fn game_page(&self, live: bool, cx: &mut Context<Self>) -> Vec<AnyElement> {
        let choose = |label: &'static str, cx: &mut Context<Self>| {
            self.stop(
                ui::button("install", label, Kind::Secondary, Size::Regular),
                live,
                cx,
            )
            .on_press(cx.listener(|this, _, window, cx| this.choose_game(&ChooseGame, window, cx)))
            .into_any_element()
        };
        let Some(game) = &self.game else {
            return vec![
                ui::group(
                    None,
                    vec![
                        ui::row(
                            None,
                            ui::titles(
                                "No game",
                                Some("Choose your NUP package or dump folder.".into()),
                            ),
                            Some(choose("Choose game…", cx)),
                        )
                        .into_any_element(),
                    ],
                )
                .into_any_element(),
            ];
        };
        let icon = match &game.icon {
            Some(icon) => img(icon.clone())
                .flex_none()
                .size(u(40.0))
                .rounded(u(radius::CONTROL))
                .object_fit(ObjectFit::Cover)
                .into_any_element(),
            None => div()
                .flex_none()
                .size(u(40.0))
                .rounded(u(radius::CONTROL))
                .bg(color(palette::FILL))
                .into_any_element(),
        };
        let installed = install::data_dir()
            .map(|dir| dir.join("game"))
            .is_some_and(|dir| game.path.starts_with(dir));
        let install_row = if installed {
            ui::row(
                None,
                ui::titles(
                    "Installed",
                    Some("Your original NUP or dump is no longer needed.".into()),
                ),
                Some(choose("Reinstall…", cx)),
            )
        } else {
            let path = game.path.clone();
            ui::row(
                None,
                ui::titles(
                    "Not installed",
                    Some("Played from its original location.".into()),
                ),
                Some(
                    self.stop(
                        ui::button(
                            "install",
                            "Install into Echora",
                            Kind::Secondary,
                            Size::Regular,
                        ),
                        live,
                        cx,
                    )
                    .on_press(
                        cx.listener(move |this, _, _, cx| this.install_game(path.clone(), cx)),
                    )
                    .into_any_element(),
                ),
            )
        };
        vec![
            ui::group(
                None,
                vec![
                    ui::row(
                        Some(icon),
                        ui::titles(game.title.clone(), Some(game.version.clone())),
                        None,
                    )
                    .into_any_element(),
                    install_row.into_any_element(),
                ],
            )
            .into_any_element(),
        ]
    }

    /// About: name, version and description, the credits in two columns,
    /// and the licence with a link to the source, all without scrolling.
    fn render_about(&self, window: &Window, closing: bool, cx: &mut Context<Self>) -> AnyElement {
        let (room_w, _) = self.sheet_room(window);
        let width = room_w.min(px(560.0) * (window.rem_size() / px(16.0)));
        let source = self
            .stop(
                ui::button("source", "Source code", Kind::Secondary, Size::Regular),
                !closing,
                cx,
            )
            .on_press(cx.listener(|_, _, _, cx| cx.open_url("https://github.com/Mars7x/echora")));
        let credit = |&(name, role): &(&'static str, &'static str)| {
            div()
                .flex_1()
                .min_w_0()
                .flex()
                .flex_col()
                .gap(u(2.0))
                .child(
                    div()
                        .text_size(u(text::BODY))
                        .text_color(color(palette::LABEL))
                        .truncate()
                        .child(name),
                )
                .child(
                    div()
                        .text_size(u(text::CAPTION))
                        .text_color(color(palette::SECONDARY_LABEL))
                        .child(role),
                )
        };
        let credits = div()
            .p(u(space::L))
            .flex()
            .flex_col()
            .gap(u(space::M))
            .rounded(u(radius::CARD))
            .bg(color(palette::CARD))
            .children(CREDITS.chunks(2).map(|pair| {
                div()
                    .flex()
                    .gap(u(space::L))
                    .children(pair.iter().map(credit))
            }));
        let top = self
            .drag_window(
                div()
                    .h(u(48.0))
                    .px(u(space::M))
                    .flex()
                    .items_center()
                    .justify_end(),
                cx,
            )
            .child(self.close_button(cx));
        let surface = palette::SURFACE;
        let sheet = sheet("about", width, None, surface, window.rem_size())
            .flex_col()
            .when(!closing, |el| el.track_focus(&self.sheet_focus))
            .child(top)
            .child(
                div()
                    .px(u(space::XL + space::XS))
                    .flex()
                    .flex_col()
                    .gap(u(space::XS + 2.0))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(u(space::S + 2.0))
                            .child(
                                div()
                                    .text_size(u(text::LARGE_TITLE))
                                    .font_weight(FontWeight::BOLD)
                                    .text_color(color(palette::LABEL))
                                    .child("Echora"),
                            )
                            .child(
                                div()
                                    .px(u(space::S))
                                    .py(u(2.0))
                                    .rounded_full()
                                    .bg(color(palette::FILL))
                                    .text_size(u(text::CAPTION))
                                    .text_color(color(palette::LABEL))
                                    .child(env!("CARGO_PKG_VERSION")),
                            ),
                    )
                    .child(
                        div()
                            .text_size(u(text::BODY))
                            .text_color(color(palette::SECONDARY_LABEL))
                            .child(
                                "A native recompilation of The Legend of Zelda: Ocarina of \
                                 Time for Nintendo Switch 2. You need your own copy of the game.",
                            ),
                    ),
            )
            .child(
                div()
                    .px(u(space::XL))
                    .pt(u(space::L + space::XS))
                    .flex()
                    .flex_col()
                    .gap(u(space::S))
                    .child(
                        div()
                            .px(u(space::XS))
                            .text_size(u(text::CALLOUT))
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(color(palette::SECONDARY_LABEL))
                            .child("Credits"),
                    )
                    .child(credits),
            )
            .child(
                div()
                    .px(u(space::XL + space::XS))
                    .py(u(space::L))
                    .flex()
                    .items_center()
                    .justify_between()
                    .gap(u(space::L))
                    .child(
                        div()
                            .text_size(u(text::CAPTION))
                            .text_color(color(palette::SECONDARY_LABEL))
                            .child("GPL-3.0-or-later. Not affiliated with Nintendo."),
                    )
                    .child(source),
            );
        scrim(sheet, self.corner, self.motion, closing, cx)
    }
}

/// How far, in pixels, a header press must travel before it moves the
/// window, as GTK waits, so a click that wobbles stays a click.
const DRAG_THRESHOLD: f32 = 4.0;

/// The width Play, Settings and About share, so the row reads as one set.
const DOCK_BUTTON: f32 = 112.0;

/// The artwork tile beside the title, and the narrowest window (in design
/// pixels) that shows it.
const ART_SIZE: f32 = 220.0;
const ART_MIN_WIDTH: f32 = 860.0;

/// The window size the design is drawn for; larger windows scale it up.
const DESIGN_SIZE: (f32, f32) = (960.0, 600.0);
/// The most the UI grows on large windows.
const MAX_SCALE: f32 = 1.6;

impl Render for Launcher {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // Everything sized in rems grows with the window: 16 px per rem at
        // the design size, up to 1.6 times that on large windows.
        let content =
            window.viewport_size() - gpui::size(px(WINDOW_INSET * 2.0), px(WINDOW_INSET * 2.0));
        let scale = (f32::from(content.width) / DESIGN_SIZE.0)
            .min(f32::from(content.height) / DESIGN_SIZE.1)
            .clamp(1.0, MAX_SCALE);
        window.set_rem_size(px(16.0 * scale));
        let Decorations::Client { tiling } = window.window_decorations() else {
            self.corner = px(0.0);
            return self.render_content(window, cx).into_any_element();
        };
        let floating = !tiling.is_tiled() && !window.is_maximized();
        let inset = px(WINDOW_INSET);
        window.set_client_inset(if floating { inset } else { px(0.0) });
        self.corner = if floating {
            px(radius::WINDOW)
        } else {
            px(0.0)
        };
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
                        .border_color(color(palette::WINDOW_EDGE)),
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
                    // Redraw only when the pointer enters or leaves a resize
                    // edge, so the cursor updates without a frame per move.
                    .on_mouse_move(cx.listener(move |this, event: &MouseMoveEvent, window, _| {
                        let size = window.window_bounds().get_bounds().size;
                        let edge = chrome::resize_edge(event.position, inset, size);
                        if edge != this.resize_edge {
                            this.resize_edge = edge;
                            window.refresh();
                        }
                    }))
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
        let welcome = self.game.is_none() && !self.loading && self.installing.is_none();
        self.stops.begin_frame();
        if std::mem::take(&mut self.refocus) {
            window.focus(&self.default_focus(cx));
        }
        div()
            .key_context("Launcher")
            .track_focus(&self.focus)
            .on_action(cx.listener(Self::play))
            .on_action(cx.listener(Self::toggle_settings))
            .on_action(cx.listener(Self::close_sheet))
            .on_action(cx.listener(Self::choose_game))
            .on_action(cx.listener(Self::focus_next))
            .on_action(cx.listener(Self::focus_prev))
            .on_action(cx.listener(|this, _: &FocusLeft, window, cx| {
                this.move_focus(Direction::Left, window, cx)
            }))
            .on_action(cx.listener(|this, _: &FocusUp, window, cx| {
                this.move_focus(Direction::Up, window, cx)
            }))
            .on_action(cx.listener(|this, _: &FocusRight, window, cx| {
                this.move_focus(Direction::Right, window, cx)
            }))
            .on_action(cx.listener(|this, _: &FocusDown, window, cx| {
                this.move_focus(Direction::Down, window, cx)
            }))
            .on_action(cx.listener(|this, _: &NextTab, window, cx| this.step_tab(1, window, cx)))
            .on_action(cx.listener(|this, _: &PrevTab, window, cx| this.step_tab(-1, window, cx)))
            // Any key press starts keyboard navigation, showing the focus
            // ring, and any mouse press ends it.
            .capture_key_down(cx.listener(|this, _: &KeyDownEvent, _, cx| {
                if !this.keyboard {
                    this.keyboard = true;
                    cx.notify();
                }
            }))
            // The pointer is back with the window, button released, once a
            // move has ended.
            .on_mouse_move(cx.listener(|this, event: &MouseMoveEvent, _, cx| {
                if this.dragging && event.pressed_button.is_none() {
                    this.dragging = false;
                    cx.notify();
                }
            }))
            .capture_any_mouse_down(cx.listener(|this, _, window, cx| {
                this.menu_dismissed = None;
                if this.keyboard {
                    this.keyboard = false;
                    cx.notify();
                }
                // Keep focus inside the launcher, so its shortcuts (Escape
                // above all) keep working after clicks on plain areas.
                if !this.focus.contains_focused(window, cx) {
                    let target = if this.sheet.is_some() {
                        &this.sheet_focus
                    } else {
                        &this.focus
                    };
                    window.focus(target);
                }
            }))
            .size_full()
            .relative()
            .overflow_hidden()
            .font_family(self.font.clone())
            .text_size(u(text::BODY))
            .child(self.render_background())
            .when(welcome, |el| el.child(self.render_welcome(cx)))
            .child(self.render_title_bar(window, cx))
            .map(|el| {
                let toast = self.notice.clone().map(|notice| {
                    ui::appear(
                        ui::toast(notice),
                        ElementId::Name(format!("toast-{}", self.notice_generation).into()),
                        space::S,
                        self.motion,
                    )
                });
                if self.loading {
                    return el;
                }
                // The game's artwork balances the title on the right, where
                // the window is wide enough for both.
                let wide = f32::from(window.viewport_size().width)
                    / (f32::from(window.rem_size()) / 16.0)
                    >= ART_MIN_WIDTH;
                let art = self
                    .game
                    .as_ref()
                    .and_then(|game| game.icon.clone())
                    .filter(|_| wide)
                    .map(|icon| {
                        img(icon)
                            .flex_none()
                            .size(u(ART_SIZE))
                            .rounded(u(radius::DOCK))
                            .border_1()
                            .border_color(color(palette::WINDOW_EDGE))
                            .object_fit(ObjectFit::Cover)
                    });
                if self.game.is_some() || self.installing.is_some() {
                    // The title hero sits in the bottom left; messages
                    // appear above it.
                    el.child(
                        div()
                            .absolute()
                            .left(u(space::XL + space::L))
                            .right(u(space::XL + space::L))
                            .bottom(u(space::XL + space::L))
                            .flex()
                            .flex_col()
                            .items_start()
                            .gap(u(space::L))
                            .children(toast)
                            .child(
                                div()
                                    .w_full()
                                    .flex()
                                    .items_end()
                                    .justify_between()
                                    .gap(u(space::XL))
                                    .child(ui::appear(
                                        div().child(self.render_hero(cx)),
                                        "hero-in",
                                        space::L,
                                        self.motion,
                                    ))
                                    .children(art.map(|art| {
                                        ui::appear(art, "art-in", space::L, self.motion)
                                    })),
                            ),
                    )
                } else {
                    el.child(
                        div()
                            .absolute()
                            .left_0()
                            .right_0()
                            .bottom(u(space::XL))
                            .flex()
                            .flex_col()
                            .items_center()
                            .gap(u(space::M))
                            .children(toast)
                            .child(ui::appear(
                                div().child(self.render_dock(cx)),
                                "dock-in",
                                space::L,
                                self.motion,
                            )),
                    )
                }
            })
            .when_some(self.closing, |el, sheet| {
                el.child(match sheet {
                    Sheet::Settings => self.render_settings(window, true, cx),
                    Sheet::About => self.render_about(window, true, cx),
                })
            })
            .when_some(self.sheet, |el, sheet| {
                el.child(match sheet {
                    Sheet::Settings => self.render_settings(window, false, cx),
                    Sheet::About => self.render_about(window, false, cx),
                })
            })
            .child(self.stops.ring(self.keyboard, self.motion))
    }
}

#[cfg(test)]
mod tests {
    use super::{Game, game_environment, run_arguments, split_title, supported_build};

    fn game(build: Option<(&str, u32)>, version: &str) -> Game {
        Game {
            path: "game.nup".into(),
            build: build.map(|(title, version)| (title.to_owned(), version)),
            title: "The Legend of Zelda: Ocarina of Time".into(),
            version: version.to_owned().into(),
            icon: None,
        }
    }

    #[test]
    fn only_the_supported_build_installs() {
        assert!(supported_build(game(Some(("0400486024DEC000", 131072)), "Version 1.1.1")).is_ok());
        let older = supported_build(game(Some(("0400486024DEC000", 65536)), "Version 1.0.0"));
        assert_eq!(
            older.err().unwrap(),
            "This is version 1.0.0 of the game; Echora needs version 1.1.1"
        );
        assert!(supported_build(game(Some(("0100000000010000", 0)), "Version 1.0")).is_err());
        assert!(supported_build(game(None, "Version 1.0")).is_err());
    }

    #[test]
    fn the_game_runs_without_a_budget() {
        let arguments = run_arguments(std::path::Path::new("game.nup"));
        assert_eq!(arguments[0], "run");
        assert_eq!(arguments[1], "game.nup");
        assert_eq!(arguments[2], "--budget");
        assert_eq!(arguments[3], u64::MAX.to_string().as_str());
    }

    /// The variables `command` sets for the game, by name.
    fn environment(command: &std::process::Command) -> Vec<(String, Option<String>)> {
        command
            .get_envs()
            .map(|(name, value)| {
                (
                    name.to_string_lossy().into_owned(),
                    value.map(|value| value.to_string_lossy().into_owned()),
                )
            })
            .collect()
    }

    #[test]
    fn the_game_starts_with_its_window_and_sound() {
        let mut command = std::process::Command::new("echora");
        game_environment(&mut command, Some(std::path::Path::new("s.toml")), |_| {
            false
        });
        let env = environment(&command);
        for (name, value) in [
            ("ECHORA_SETTINGS", "s.toml"),
            ("ECHORA_WINDOW", "1"),
            ("ECHORA_AUDIO", "1"),
        ] {
            assert!(
                env.contains(&(name.into(), Some(value.into()))),
                "{name} in {env:?}"
            );
        }
    }

    #[test]
    fn a_window_or_sound_choice_in_the_environment_is_kept() {
        let mut command = std::process::Command::new("echora");
        game_environment(&mut command, None, |name| name == "ECHORA_AUDIO");
        let env = environment(&command);
        assert!(env.contains(&("ECHORA_WINDOW".into(), Some("1".into()))));
        assert!(
            !env.iter()
                .any(|(name, _)| name == "ECHORA_AUDIO" || name == "ECHORA_SETTINGS")
        );
    }

    #[test]
    fn splits_series_from_the_title() {
        let (series, name) = split_title("The Legend of Zelda Ocarina of Time");
        assert_eq!(
            series.map(|s| s.to_string()).as_deref(),
            Some("The Legend of Zelda")
        );
        assert_eq!(name.to_string(), "Ocarina of Time");
        let (series, name) = split_title("The Legend of Zelda: Ocarina of Time");
        assert_eq!(
            series.map(|s| s.to_string()).as_deref(),
            Some("The Legend of Zelda")
        );
        assert_eq!(name.to_string(), "Ocarina of Time");
        let (series, name) = split_title("Some Game");
        assert_eq!(series, None);
        assert_eq!(name.to_string(), "Some Game");
    }
}
