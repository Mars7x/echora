//! The launcher window: the game's title large in the bottom left with Play,
//! Settings and About under it, and settings and about in sheets above.
//! Everything visual comes from the design system in [`crate::ui`].

use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitCode};
use std::sync::Arc;
use std::time::Duration;

use echora_config::{InternalResolutionSetting, Settings, WindowMode};
use echora_content::ContentSource;
use gpui::{
    Animation, AnimationExt as _, AnyElement, App, Application, Bounds, BoxShadow, Context,
    CursorStyle, Decorations, ElementId, FocusHandle, FontWeight, HitboxBehavior, Hsla, Image,
    ImageFormat, KeyBinding, MouseButton, MouseMoveEvent, ObjectFit, PathPromptOptions, Pixels,
    ResizeEdge, SharedString, TitlebarOptions, Window, WindowBackgroundAppearance, WindowBounds,
    WindowDecorations, WindowOptions, actions, anchored, canvas, deferred, div, img, point,
    prelude::*, px, size,
};

use crate::chrome::{self, ButtonLayout, WindowButton};
use crate::install;
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

struct Launcher {
    focus: FocusHandle,
    /// Play is the default button: it holds focus, so Enter or Space starts
    /// the game.
    play_focus: FocusHandle,
    /// The keyboard is being used to move focus, so focus rings show.
    keyboard: bool,
    /// The desktop allows animations.
    motion: bool,
    font: SharedString,
    game: Option<Game>,
    settings: Settings,
    settings_path: Option<PathBuf>,
    sheet: Option<Sheet>,
    /// Holds focus while a sheet is open, so Escape always reaches it.
    sheet_focus: FocusHandle,
    /// A sheet fading out after being closed.
    closing: Option<Sheet>,
    /// The settings pop-up menu that is open, by id.
    open_menu: Option<&'static str>,
    /// The menu closed by the current mouse press, so the release on its
    /// own button doesn't open it again.
    menu_dismissed: Option<&'static str>,
    /// The switch just flipped; only it animates.
    toggled: Option<&'static str>,
    /// Window buttons the desktop asks for, when decorations are ours.
    buttons: ButtonLayout,
    /// Corner radius of the window this frame (zero when maximized or tiled).
    corner: Pixels,
    /// A title bar press that turns into a window move once the mouse drags.
    moving: bool,
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
            let font = ui::load_font(cx);
            let opened = cx.open_window(options, |window, cx| {
                cx.new(|cx| {
                    let launcher = Launcher::new(buttons, motion, font, window, cx);
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
/// means the window should open instead: also when the game fails, so the
/// player isn't left with nothing on screen.
fn skip_to_game() -> Option<ExitCode> {
    let (settings_path, settings) = load_settings();
    if !settings.launcher.skip {
        return None;
    }
    let mut command = game_command(&last_game()?, settings_path.as_deref())?;
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
            play_focus: cx.focus_handle(),
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
            menu_dismissed: None,
            toggled: None,
            buttons,
            corner: px(0.0),
            moving: false,
            resize_edge: None,
            loading: false,
            settings_tab: SettingsTab::Game,
            notice: None,
            notice_generation: 0,
            running: false,
            installing: None,
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
                            window.focus(&this.play_focus);
                        }
                        Err(message) => this.notify(message, cx),
                    }
                    cx.notify();
                });
            })
            .detach();
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

    fn toggle_settings(&mut self, _: &ToggleSettings, window: &mut Window, cx: &mut Context<Self>) {
        if self.sheet == Some(Sheet::Settings) {
            self.close_sheet(&CloseSheet, window, cx);
        } else {
            self.open_sheet(Sheet::Settings, window, cx);
        }
    }

    fn open_sheet(&mut self, sheet: Sheet, window: &mut Window, cx: &mut Context<Self>) {
        self.sheet = Some(sheet);
        self.closing = None;
        self.open_menu = None;
        self.toggled = None;
        window.focus(&self.sheet_focus);
        cx.notify();
    }

    /// Escape and the close buttons: closes an open menu first, then the
    /// sheet, which fades out.
    fn close_sheet(&mut self, _: &CloseSheet, window: &mut Window, cx: &mut Context<Self>) {
        if self.open_menu.take().is_some() {
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
        if self.game.is_some() {
            window.focus(&self.play_focus);
        } else {
            window.focus(&self.focus);
        }
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
    ("PokeDinner, an Eglantine fork", "Shader recompiler"),
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
fn to_rems(pixels: Pixels) -> gpui::Rems {
    gpui::rems(f32::from(pixels) / 16.0)
}

/// A sheet's surface: the window's own background, at a fixed size so
/// wrapped text is measured at its real width.
fn sheet(
    id: &'static str,
    width: Pixels,
    height: Option<Pixels>,
    surface: u32,
) -> gpui::Stateful<gpui::Div> {
    div()
        .id(id)
        .w(to_rems(width))
        .when_some(height, |el, height| el.h(to_rems(height)))
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

    fn close_button(&self, live: bool, cx: &mut Context<Self>) -> gpui::Stateful<gpui::Div> {
        ui::focusable(
            ui::icon_button("close-sheet", WindowButton::Close.icon(false)),
            live,
            self.keyboard,
        )
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
        let buttons = |side: &[WindowButton]| {
            div()
                .flex()
                .gap(px(space::M))
                .children(side.iter().map(|&button| {
                    ui::icon_button(button.id(), button.icon(maximized)).on_click(
                        move |_, window, _| match button {
                            WindowButton::Minimize => window.minimize_window(),
                            WindowButton::Maximize => window.zoom_window(),
                            WindowButton::Close => window.remove_window(),
                        },
                    )
                }))
        };
        div()
            .id("title-bar")
            .absolute()
            .top_0()
            .left_0()
            .right_0()
            .h(px(48.0))
            .px(px(space::M))
            .flex()
            .items_center()
            .justify_between()
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
            .child(div().when(client_side, |el| el.child(buttons(&self.buttons.left))))
            .child(div().when(client_side, |el| el.child(buttons(&self.buttons.right))))
    }

    /// No game yet: a status page with the one thing to do.
    fn render_welcome(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let choose = ui::focusable(
            ui::button("choose", "Choose game…", Kind::Primary, Size::Large),
            self.sheet.is_none(),
            self.keyboard,
        )
        .on_click(cx.listener(|this, _, window, cx| this.choose_game(&ChooseGame, window, cx)));
        div()
            .absolute()
            .inset_0()
            .flex()
            .flex_col()
            .items_center()
            .justify_center()
            .gap(px(space::S))
            .pb(px(72.0))
            .child(
                div()
                    .text_size(px(text::LARGE_TITLE))
                    .font_weight(FontWeight::BOLD)
                    .text_color(color(palette::LABEL))
                    .child("Echora"),
            )
            .child(
                div()
                    .text_size(px(text::HEADLINE))
                    .text_color(color(palette::SECONDARY_LABEL))
                    .child("Choose your NUP package or dump folder to install the game."),
            )
            .child(div().pt(px(space::L)).child(choose))
    }

    /// Settings and About; the whole bottom bar on the empty screen, and
    /// beside Play on the game screen.
    fn secondary_actions(&self, cx: &mut Context<Self>) -> [gpui::Stateful<gpui::Div>; 2] {
        let active = self.sheet.is_none();
        let settings = ui::focusable(
            ui::button("settings", "Settings", Kind::Secondary, Size::Large),
            active,
            self.keyboard,
        )
        .on_click(
            cx.listener(|this, _, window, cx| this.toggle_settings(&ToggleSettings, window, cx)),
        );
        let about = ui::focusable(
            ui::button("about", "About", Kind::Secondary, Size::Large),
            active,
            self.keyboard,
        )
        .on_click(cx.listener(|this, _, window, cx| this.open_sheet(Sheet::About, window, cx)));
        [settings, about]
    }

    /// The empty screen's bar: Settings and About.
    fn render_dock(&self, cx: &mut Context<Self>) -> AnyElement {
        div()
            .flex()
            .items_center()
            .gap(px(space::S))
            .p(px(space::S))
            .rounded(px(radius::DOCK))
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
            (None, Some(game)) => (game.title.clone(), game.version.to_string()),
            (None, None) => return div().into_any_element(),
        };
        let (series, name) = split_title(&title);
        let actions = if let Some((_, progress)) = &self.installing {
            div()
                .w(px(320.0))
                .child(ui::progress(progress.fraction()))
                .into_any_element()
        } else {
            let active = self.sheet.is_none();
            let play = ui::focusable(
                ui::button(
                    "play",
                    if self.running { "Running" } else { "Play" },
                    Kind::Primary,
                    Size::Large,
                )
                .px(px(36.0)),
                active,
                self.keyboard,
            )
            .when(active, |el| el.track_focus(&self.play_focus))
            .when(self.running, |el| el.opacity(0.6).cursor_default())
            .on_click(cx.listener(|this, _, window, cx| this.play(&Play, window, cx)));
            div()
                .flex()
                .gap(px(space::S))
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
                        .text_size(px(text::HEADLINE))
                        .text_color(color(palette::SECONDARY_LABEL))
                        .child(series),
                )
            })
            .child(
                div()
                    .font_family(ui::TITLE_FONT)
                    .text_size(px(text::HERO))
                    .line_height(gpui::relative(1.1))
                    .font_weight(FontWeight::BOLD)
                    .text_color(color(palette::LABEL))
                    .child(name),
            )
            .child(
                div()
                    .pt(px(space::XS + 2.0))
                    .text_size(px(text::CALLOUT))
                    .text_color(color(palette::SECONDARY_LABEL))
                    .child(detail),
            )
            .child(div().pt(px(space::XL)).child(actions))
            .into_any_element()
    }

    /// Lets `el` drag the window, like the title bar.
    fn drag_window(&self, el: gpui::Div, cx: &mut Context<Self>) -> gpui::Div {
        el.on_mouse_down(
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
    }

    /// The settings sheet. A `closing` sheet is drawn for its fade-out only.
    fn render_settings(
        &self,
        window: &Window,
        closing: bool,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let (room_w, room_h) = self.sheet_room(window);
        let width = room_w.min(px(760.0));
        let height = room_h.min(px(460.0));
        let keyboard = self.keyboard;
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
            .border_r_1()
            .border_color(color(palette::BORDER))
            .child(title)
            .child(
                div()
                    .px(u(space::S))
                    .flex()
                    .flex_col()
                    .gap(u(2.0))
                    .children(SettingsTab::ALL.into_iter().map(|tab| {
                        let selected = tab == self.settings_tab;
                        ui::focusable(
                            div()
                                .id(tab.label())
                                .h(u(36.0))
                                .px(u(space::M))
                                .flex()
                                .items_center()
                                .rounded(u(radius::CONTROL))
                                .text_size(u(text::BODY))
                                .cursor_pointer()
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
                            !closing,
                            keyboard,
                        )
                        .on_click(cx.listener(move |this, _, _, cx| {
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
            .child(self.close_button(!closing, cx));
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
        let sheet = sheet("settings", width, Some(height), surface)
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
        let keyboard = self.keyboard;
        let motion = self.motion;
        let toggled = self.toggled;
        let switch =
            |id: &'static str, on: bool, flip: fn(&mut Settings), cx: &mut Context<Self>| {
                ui::focusable(
                    ui::switch(id, on, motion && toggled == Some(id)),
                    live,
                    keyboard,
                )
                .on_click(cx.listener(move |this, _, _, cx| {
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
                    ],
                )
                .into_any_element(),
            ],
            SettingsTab::Controls => vec![
                ui::group(
                    None,
                    vec![
                        ui::row(
                            None,
                            ui::titles("Vibration", None),
                            Some(switch(
                                "vibration",
                                settings.input.vibration,
                                |s| s.input.vibration = !s.input.vibration,
                                cx,
                            )),
                        )
                        .into_any_element(),
                    ],
                )
                .into_any_element(),
            ],
            SettingsTab::Launcher => vec![
                ui::group(
                    None,
                    vec![
                        ui::row(
                            None,
                            ui::titles(
                                "Skip launcher",
                                Some(
                                    "Start the game directly. Choose Open launcher from \
                                     Echora's app menu to come back."
                                        .into(),
                                ),
                            ),
                            Some(switch(
                                "skip",
                                settings.launcher.skip,
                                |s| s.launcher.skip = !s.launcher.skip,
                                cx,
                            )),
                        )
                        .into_any_element(),
                    ],
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
        let keyboard = self.keyboard;
        let open = live && self.open_menu == Some(id);
        let label = choices
            .iter()
            .find(|(_, value)| *value == current)
            .map_or(choices[0].0, |(label, _)| label);
        let button = ui::focusable(ui::popup_button((id, 0usize), label), live, keyboard).on_click(
            cx.listener(move |this, _, _, cx| {
                if this.menu_dismissed.take() == Some(id) {
                    return;
                }
                this.open_menu = if this.open_menu == Some(id) {
                    None
                } else {
                    Some(id)
                };
                cx.notify();
            }),
        );
        if !open {
            return div().flex_none().child(button).into_any_element();
        }
        let menu = ui::menu(choices.iter().enumerate().map(|(index, &(label, value))| {
            ui::focusable(
                ui::menu_item((id, index + 1), label, value == current),
                true,
                keyboard,
            )
            .on_click(cx.listener(move |this, _, _, cx| {
                this.open_menu = None;
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
                                cx.listener(move |this, _, _, cx| {
                                    this.open_menu = None;
                                    this.menu_dismissed = Some(id);
                                    cx.notify();
                                }),
                            ),
                    ),
                )
                .with_priority(1),
            )
            .child(
                div().absolute().top(u(36.0)).right_0().child(
                    deferred(
                        anchored()
                            .anchor(gpui::Corner::TopRight)
                            .snap_to_window_with_margin(px(space::S))
                            .child(ui::appear(menu, (id, 101usize), -6.0, self.motion)),
                    )
                    .with_priority(2),
                ),
            )
            .into_any_element()
    }

    /// The installed game, and installing or reinstalling it.
    fn game_page(&self, live: bool, cx: &mut Context<Self>) -> Vec<AnyElement> {
        let keyboard = self.keyboard;
        let choose = |label: &'static str, cx: &mut Context<Self>| {
            ui::focusable(
                ui::button("install", label, Kind::Secondary, Size::Regular),
                live,
                keyboard,
            )
            .on_click(cx.listener(|this, _, window, cx| this.choose_game(&ChooseGame, window, cx)))
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
                    ui::focusable(
                        ui::button(
                            "install",
                            "Install into Echora",
                            Kind::Secondary,
                            Size::Regular,
                        ),
                        live,
                        keyboard,
                    )
                    .on_click(
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
        let width = room_w.min(px(560.0));
        let source = ui::focusable(
            ui::button("source", "Source code", Kind::Secondary, Size::Regular),
            !closing,
            self.keyboard,
        )
        .on_click(cx.listener(|_, _, _, cx| cx.open_url("https://github.com/Mars7x/echora")));
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
                    .h(u(44.0))
                    .px(u(space::M))
                    .flex()
                    .items_center()
                    .justify_end(),
                cx,
            )
            .child(self.close_button(!closing, cx));
        let surface = palette::SURFACE;
        let sheet = sheet("about", width, None, surface)
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

impl Render for Launcher {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
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
        div()
            .key_context("Launcher")
            .track_focus(&self.focus)
            .on_action(cx.listener(Self::play))
            .on_action(cx.listener(Self::toggle_settings))
            .on_action(cx.listener(Self::close_sheet))
            .on_action(cx.listener(Self::choose_game))
            .on_action(cx.listener(Self::focus_next))
            .on_action(cx.listener(Self::focus_prev))
            // Any mouse press ends keyboard navigation, hiding focus rings.
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
            .text_size(px(text::BODY))
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
                if self.game.is_some() || self.installing.is_some() {
                    // The title hero sits in the bottom left; messages
                    // appear above it.
                    el.child(
                        div()
                            .absolute()
                            .left(px(space::XL + space::L))
                            .right(px(space::XL + space::L))
                            .bottom(px(space::XL + space::L))
                            .flex()
                            .flex_col()
                            .items_start()
                            .gap(px(space::L))
                            .children(toast)
                            .child(ui::appear(
                                div().child(self.render_hero(cx)),
                                "hero-in",
                                space::L,
                                self.motion,
                            )),
                    )
                } else {
                    el.child(
                        div()
                            .absolute()
                            .left_0()
                            .right_0()
                            .bottom(px(space::XL))
                            .flex()
                            .flex_col()
                            .items_center()
                            .gap(px(space::M))
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
    }
}

#[cfg(test)]
mod tests {
    use super::split_title;

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
