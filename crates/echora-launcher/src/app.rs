//! The GTK4/libadwaita launcher window.

use std::cell::RefCell;
use std::path::PathBuf;
use std::process::{Child, Command, ExitCode};
use std::rc::Rc;
use std::time::Duration;

use adw::prelude::*;
use echora_config::{InternalResolutionSetting, Settings, WindowMode};
use echora_content::ContentSource;
use gtk::{gdk, gio, glib};

const APP_ID: &str = "io.github.mars7x.Echora";

/// Internal resolution choices offered in preferences.
const RESOLUTIONS: [(&str, InternalResolutionSetting); 5] = [
    ("Match Window", InternalResolutionSetting::MatchOutput),
    ("720p", InternalResolutionSetting::Height { height: 720 }),
    ("1080p", InternalResolutionSetting::Height { height: 1080 }),
    ("1440p", InternalResolutionSetting::Height { height: 1440 }),
    ("2160p", InternalResolutionSetting::Height { height: 2160 }),
];

const WINDOW_MODES: [(&str, WindowMode); 3] = [
    ("Windowed", WindowMode::Windowed),
    ("Borderless Fullscreen", WindowMode::BorderlessFullscreen),
    ("Exclusive Fullscreen", WindowMode::ExclusiveFullscreen),
];

/// A menu action.
type Action = fn(&Rc<State>);

/// The launcher's look: flat surfaces, the game's icon, clear type and one
/// accent colour for the primary action.
const STYLE: &str = r#"
window.echora {
    background: #121417;
    color: #e6e8eb;
}
window.echora headerbar {
    background: transparent;
    box-shadow: none;
}
.echora-wordmark {
    font-size: 36pt;
    font-weight: 800;
    letter-spacing: 6px;
}
.echora-tagline {
    font-size: 11pt;
    color: alpha(#e6e8eb, 0.6);
}
.echora-hint {
    font-size: 9pt;
    color: alpha(#e6e8eb, 0.4);
}
.echora-icon {
    border-radius: 20px;
}
.echora-icon-placeholder {
    background: #1b1e22;
    border-radius: 20px;
}
.echora-tag {
    font-size: 9pt;
    font-weight: 700;
    letter-spacing: 0.5px;
    color: #4cc9a0;
}
.echora-title {
    font-size: 24pt;
    font-weight: 800;
}
.echora-subtitle {
    font-size: 11pt;
    color: alpha(#e6e8eb, 0.6);
}
button.echora-play {
    background: #4cc9a0;
    color: #0b1a14;
    font-size: 12pt;
    font-weight: 700;
    border-radius: 10px;
    padding: 8px 32px;
}
button.echora-play:hover {
    background: #5fd4ad;
}
button.echora-play:disabled {
    background: #24282d;
    color: alpha(#e6e8eb, 0.45);
}
button.echora-round {
    background: #1f2328;
    border-radius: 10px;
    min-width: 40px;
    min-height: 40px;
}
button.echora-round:hover {
    background: #292e34;
}
.echora-card {
    background: #1a1d21;
    border-radius: 14px;
    padding: 16px 20px;
}
.echora-key {
    font-size: 9pt;
    font-weight: 700;
    letter-spacing: 0.5px;
    color: alpha(#e6e8eb, 0.45);
}
.echora-value {
    font-family: monospace;
    font-size: 10pt;
}
"#;

struct State {
    window: adw::ApplicationWindow,
    toasts: adw::ToastOverlay,
    stack: gtk::Stack,
    icon: gtk::Picture,
    icon_placeholder: gtk::Box,
    tag: gtk::Label,
    title: gtk::Label,
    subtitle: gtk::Label,
    build: gtk::Label,
    version: gtk::Label,
    path: gtk::Label,
    launch: gtk::Button,
    game: RefCell<Option<PathBuf>>,
    settings_path: Option<PathBuf>,
    settings: RefCell<Settings>,
    child: RefCell<Option<Child>>,
}

pub fn run() -> ExitCode {
    let app = adw::Application::builder().application_id(APP_ID).build();
    app.connect_startup(|_| {
        adw::StyleManager::default().set_color_scheme(adw::ColorScheme::ForceDark);
        let css = gtk::CssProvider::new();
        css.load_from_string(STYLE);
        if let Some(display) = gdk::Display::default() {
            gtk::style_context_add_provider_for_display(
                &display,
                &css,
                gtk::STYLE_PROVIDER_PRIORITY_APPLICATION,
            );
        }
    });
    app.connect_activate(build_window);
    if app.run() == glib::ExitCode::SUCCESS {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn label(text: &str, class: &str) -> gtk::Label {
    let label = gtk::Label::new(Some(text));
    label.add_css_class(class);
    label.set_xalign(0.0);
    label
}

fn round_button(icon: &str, tooltip: &str) -> gtk::Button {
    let button = gtk::Button::from_icon_name(icon);
    button.add_css_class("echora-round");
    button.set_tooltip_text(Some(tooltip));
    button.set_valign(gtk::Align::Center);
    button
}

/// One labelled value in the details card.
fn detail(grid: &gtk::Grid, row: i32, key: &str) -> gtk::Label {
    let name = label(key, "echora-key");
    name.set_valign(gtk::Align::Start);
    let value = label("", "echora-value");
    value.set_selectable(true);
    value.set_wrap(true);
    value.set_wrap_mode(gtk::pango::WrapMode::Char);
    value.set_hexpand(true);
    grid.attach(&name, 0, row, 1, 1);
    grid.attach(&value, 1, row, 1, 1);
    value
}

fn build_window(app: &adw::Application) {
    let settings_path = echora_config::default_path();
    let settings = settings_path
        .as_deref()
        .map(Settings::load)
        .transpose()
        .unwrap_or_else(|error| {
            eprintln!("settings: {error}; using defaults");
            None
        })
        .unwrap_or_default();

    // Empty state: the wordmark and one action.
    let choose = gtk::Button::with_label("Choose Game");
    choose.add_css_class("echora-play");
    choose.set_halign(gtk::Align::Center);
    choose.set_margin_top(28);
    let empty = gtk::Box::new(gtk::Orientation::Vertical, 6);
    empty.set_valign(gtk::Align::Center);
    empty.set_halign(gtk::Align::Center);
    let wordmark = gtk::Label::new(Some("ECHORA"));
    wordmark.add_css_class("echora-wordmark");
    let tagline = gtk::Label::new(Some("Ocarina of Time, recompiled to run natively"));
    tagline.add_css_class("echora-tagline");
    let hint = gtk::Label::new(Some("Select your NUP package or dump folder to begin"));
    hint.add_css_class("echora-hint");
    hint.set_margin_top(10);
    empty.append(&wordmark);
    empty.append(&tagline);
    empty.append(&choose);
    empty.append(&hint);

    // Game page: icon, title and actions, then the details card.
    let icon = gtk::Picture::new();
    icon.set_size_request(184, 184);
    icon.set_can_shrink(true);
    icon.set_content_fit(gtk::ContentFit::Cover);
    icon.add_css_class("echora-icon");
    icon.set_overflow(gtk::Overflow::Hidden);
    let icon_placeholder = gtk::Box::new(gtk::Orientation::Vertical, 0);
    icon_placeholder.set_size_request(184, 184);
    icon_placeholder.add_css_class("echora-icon-placeholder");
    let placeholder_image = gtk::Image::from_icon_name("applications-games-symbolic");
    placeholder_image.set_pixel_size(72);
    placeholder_image.set_vexpand(true);
    placeholder_image.set_opacity(0.4);
    icon_placeholder.append(&placeholder_image);
    let icon_slot = gtk::Box::new(gtk::Orientation::Vertical, 0);
    icon_slot.set_valign(gtk::Align::Center);
    icon_slot.append(&icon);
    icon_slot.append(&icon_placeholder);

    let tag = label("", "echora-tag");
    tag.set_halign(gtk::Align::Start);
    let title = label("", "echora-title");
    title.set_wrap(true);
    let subtitle = label("", "echora-subtitle");
    let launch = gtk::Button::with_label("Play");
    launch.add_css_class("echora-play");
    let settings_button = round_button("emblem-system-symbolic", "Settings");
    let change = round_button("folder-open-symbolic", "Choose Another Game");
    let actions = gtk::Box::new(gtk::Orientation::Horizontal, 10);
    actions.set_margin_top(18);
    actions.append(&launch);
    actions.append(&settings_button);
    actions.append(&change);
    let heading = gtk::Box::new(gtk::Orientation::Vertical, 8);
    heading.set_valign(gtk::Align::Center);
    heading.set_hexpand(true);
    heading.append(&tag);
    heading.append(&title);
    heading.append(&subtitle);
    heading.append(&actions);
    let hero = gtk::Box::new(gtk::Orientation::Horizontal, 32);
    hero.append(&icon_slot);
    hero.append(&heading);

    let grid = gtk::Grid::new();
    grid.set_row_spacing(10);
    grid.set_column_spacing(24);
    let build = detail(&grid, 0, "BUILD ID");
    let version = detail(&grid, 1, "CONTENT VERSION");
    let path = detail(&grid, 2, "LOCATION");
    let card = gtk::Box::new(gtk::Orientation::Vertical, 0);
    card.add_css_class("echora-card");
    card.append(&grid);

    let page = gtk::Box::new(gtk::Orientation::Vertical, 32);
    page.set_margin_top(12);
    page.set_margin_bottom(36);
    page.set_margin_start(40);
    page.set_margin_end(40);
    page.set_valign(gtk::Align::Center);
    page.append(&hero);
    page.append(&card);
    let game_page = adw::Clamp::builder().maximum_size(760).child(&page).build();

    let stack = gtk::Stack::new();
    stack.set_transition_type(gtk::StackTransitionType::Crossfade);
    stack.add_named(&empty, Some("empty"));
    stack.add_named(&game_page, Some("game"));

    let menu = gio::Menu::new();
    menu.append(Some("Choose Game…"), Some("app.choose"));
    menu.append(Some("Choose Dump Folder…"), Some("app.folder"));
    menu.append(Some("Settings"), Some("app.preferences"));
    menu.append(Some("About Echora"), Some("app.about"));
    let menu_button = gtk::MenuButton::builder()
        .icon_name("open-menu-symbolic")
        .menu_model(&menu)
        .primary(true)
        .tooltip_text("Main Menu")
        .build();
    let header = adw::HeaderBar::new();
    header.set_show_title(false);
    header.pack_end(&menu_button);

    let toasts = adw::ToastOverlay::new();
    toasts.set_child(Some(&stack));
    let toolbar = adw::ToolbarView::new();
    toolbar.add_top_bar(&header);
    toolbar.set_content(Some(&toasts));
    let window = adw::ApplicationWindow::builder()
        .application(app)
        .title("Echora")
        .default_width(860)
        .default_height(560)
        .content(&toolbar)
        .build();
    window.add_css_class("echora");

    let state = Rc::new(State {
        window: window.clone(),
        toasts,
        stack,
        icon,
        icon_placeholder,
        tag,
        title,
        subtitle,
        build,
        version,
        path,
        launch: launch.clone(),
        game: RefCell::new(None),
        settings_path,
        settings: RefCell::new(settings),
        child: RefCell::new(None),
    });

    {
        let state = state.clone();
        choose.connect_clicked(move |_| choose_package(&state));
    }
    {
        let state = state.clone();
        change.connect_clicked(move |_| choose_package(&state));
    }
    {
        let state = state.clone();
        settings_button.connect_clicked(move |_| show_preferences(&state));
    }
    {
        let state = state.clone();
        launch.connect_clicked(move |_| launch_game(&state));
    }
    let actions: [(&str, Action); 4] = [
        ("choose", choose_package),
        ("folder", choose_folder),
        ("preferences", show_preferences),
        ("about", show_about),
    ];
    for (name, run) in actions {
        let entry = gio::SimpleAction::new(name, None);
        let state = state.clone();
        entry.connect_activate(move |_, _| run(&state));
        app.add_action(&entry);
    }
    app.set_accels_for_action("app.preferences", &["<Control>comma"]);
    app.set_accels_for_action("app.choose", &["<Control>o"]);

    if let Some(last) = last_game_file()
        .and_then(|file| std::fs::read_to_string(file).ok())
        .map(|text| PathBuf::from(text.trim()))
        .filter(|path| path.exists())
    {
        select_game(&state, last);
    }
    window.present();
}

fn toast(state: &State, message: impl AsRef<str>) {
    state.toasts.add_toast(adw::Toast::new(message.as_ref()));
}

fn last_game_file() -> Option<PathBuf> {
    echora_config::default_path().map(|settings| settings.with_file_name("last-game"))
}

fn choose_package(state: &Rc<State>) {
    let filter = gtk::FileFilter::new();
    filter.set_name(Some("NUP Packages"));
    filter.add_pattern("*.nup");
    let filters = gio::ListStore::new::<gtk::FileFilter>();
    filters.append(&filter);
    let dialog = gtk::FileDialog::builder()
        .title("Choose Game")
        .filters(&filters)
        .build();
    let chosen = state.clone();
    dialog.open(Some(&state.window), gio::Cancellable::NONE, move |result| {
        if let Some(path) = result.ok().and_then(|file| file.path()) {
            select_game(&chosen, path);
        }
    });
}

fn choose_folder(state: &Rc<State>) {
    let dialog = gtk::FileDialog::builder()
        .title("Choose Dump Folder")
        .build();
    let chosen = state.clone();
    dialog.select_folder(Some(&state.window), gio::Cancellable::NONE, move |result| {
        if let Some(path) = result.ok().and_then(|file| file.path()) {
            select_game(&chosen, path);
        }
    });
}

fn select_game(state: &Rc<State>, path: PathBuf) {
    let inventory = match ContentSource::open(&path).and_then(|source| source.inventory()) {
        Ok(inventory) => inventory,
        Err(error) => {
            toast(state, format!("Not a usable game: {error}"));
            return;
        }
    };
    let Some(package) = inventory.program_package.map(|i| &inventory.packages[i]) else {
        toast(state, "The selection contains no game program");
        return;
    };
    state
        .title
        .set_label(package.title_name.as_deref().unwrap_or("Unknown title"));
    let numeric = format!(
        "{}.{}.{}",
        package.version >> 16,
        (package.version >> 8) & 0xff,
        package.version & 0xff
    );
    state.tag.set_label(&match &package.display_version {
        Some(display) => format!("VERSION {display}"),
        None => format!("VERSION {numeric}"),
    });
    state.subtitle.set_label(match inventory.source {
        "nup" => "NUP package, read in place",
        _ => "Extracted dump folder",
    });
    state
        .build
        .set_label(package.main_build_id().as_deref().unwrap_or("Unknown"));
    state
        .version
        .set_label(&format!("{numeric} ({})", package.version));
    state.path.set_label(&path.display().to_string());
    let texture = package
        .icon
        .as_deref()
        .and_then(|bytes| gdk::Texture::from_bytes(&glib::Bytes::from(bytes)).ok());
    state.icon.set_paintable(texture.as_ref());
    state.icon.set_visible(texture.is_some());
    state.icon_placeholder.set_visible(texture.is_none());
    if let Some(file) = last_game_file() {
        let _ = file
            .parent()
            .map(std::fs::create_dir_all)
            .transpose()
            .and_then(|_| std::fs::write(&file, path.display().to_string()));
    }
    *state.game.borrow_mut() = Some(path);
    state.stack.set_visible_child_name("game");
}

fn launch_game(state: &Rc<State>) {
    let Some(game) = state.game.borrow().clone() else {
        return;
    };
    let echora = std::env::current_exe()
        .ok()
        .map(|exe| exe.with_file_name("echora"))
        .filter(|path| path.is_file());
    let Some(echora) = echora else {
        toast(
            state,
            "The echora command was not found next to the launcher",
        );
        return;
    };
    let mut command = Command::new(echora);
    command.arg("run").arg(&game);
    if let Some(settings) = &state.settings_path {
        command.env("ECHORA_SETTINGS", settings);
    }
    match command.spawn() {
        Ok(child) => {
            *state.child.borrow_mut() = Some(child);
            state.launch.set_sensitive(false);
            toast(state, "Starting Ocarina of Time…");
            let watched = state.clone();
            glib::timeout_add_local(Duration::from_millis(500), move || {
                let status = watched
                    .child
                    .borrow_mut()
                    .as_mut()
                    .and_then(|child| child.try_wait().ok().flatten());
                let Some(status) = status else {
                    return glib::ControlFlow::Continue;
                };
                *watched.child.borrow_mut() = None;
                watched.launch.set_sensitive(true);
                if !status.success() {
                    toast(&watched, "The game stopped; see the run report for details");
                }
                glib::ControlFlow::Break
            });
        }
        Err(error) => toast(state, format!("Could not start echora: {error}")),
    }
}

fn save(state: &State) {
    let Some(path) = &state.settings_path else {
        toast(state, "No configuration directory; settings are not saved");
        return;
    };
    if let Err(error) = state.settings.borrow().save(path) {
        toast(state, format!("Settings not saved: {error}"));
    }
}

fn combo(title: &str, labels: &[&str], selected: usize) -> adw::ComboRow {
    let row = adw::ComboRow::builder()
        .title(title)
        .model(&gtk::StringList::new(labels))
        .build();
    row.set_selected(selected as u32);
    row
}

fn spin(title: &str, lower: f64, upper: f64, value: u32) -> adw::SpinRow {
    let row = adw::SpinRow::with_range(lower, upper, 1.0);
    row.set_title(title);
    row.set_value(f64::from(value));
    row
}

fn show_preferences(state: &Rc<State>) {
    let current = state.settings.borrow().clone();
    let dialog = adw::PreferencesDialog::new();

    let display = adw::PreferencesPage::builder()
        .title("Display")
        .icon_name("video-display-symbolic")
        .build();
    let window_group = adw::PreferencesGroup::builder()
        .title("Window")
        .description("Applied when the game starts.")
        .build();
    let mode_index = WINDOW_MODES
        .iter()
        .position(|(_, mode)| *mode == current.display.window_mode)
        .unwrap_or(0);
    let mode = combo("Mode", &WINDOW_MODES.map(|(label, _)| label), mode_index);
    let width = spin("Width", 320.0, 7680.0, current.display.window_width);
    let height = spin("Height", 240.0, 4320.0, current.display.window_height);
    window_group.add(&mode);
    window_group.add(&width);
    window_group.add(&height);
    let render_group = adw::PreferencesGroup::builder().title("Rendering").build();
    let resolution_index = RESOLUTIONS
        .iter()
        .position(|(_, setting)| *setting == current.display.internal_resolution)
        .unwrap_or(0);
    let resolution = combo(
        "Internal Resolution",
        &RESOLUTIONS.map(|(label, _)| label),
        resolution_index,
    );
    resolution.set_subtitle("The resolution the game's scenes are drawn at");
    render_group.add(&resolution);
    display.add(&window_group);
    display.add(&render_group);

    let audio = adw::PreferencesPage::builder()
        .title("Audio")
        .icon_name("audio-speakers-symbolic")
        .build();
    let audio_group = adw::PreferencesGroup::new();
    let volume = spin("Volume", 0.0, 100.0, current.audio.volume);
    audio_group.add(&volume);
    audio.add(&audio_group);

    let input = adw::PreferencesPage::builder()
        .title("Controls")
        .icon_name("input-gaming-symbolic")
        .build();
    let input_group = adw::PreferencesGroup::new();
    let vibration = adw::SwitchRow::builder()
        .title("Vibration")
        .active(current.input.vibration)
        .build();
    input_group.add(&vibration);
    input.add(&input_group);

    for page in [&display, &audio, &input] {
        dialog.add(page);
    }

    let update = {
        let state = state.clone();
        move |change: &dyn Fn(&mut Settings)| {
            change(&mut state.settings.borrow_mut());
            save(&state);
        }
    };
    let on = Rc::new(update);
    {
        let on = on.clone();
        mode.connect_selected_notify(move |row| {
            let mode = WINDOW_MODES[row.selected() as usize].1;
            on(&|s| s.display.window_mode = mode);
        });
    }
    {
        let on = on.clone();
        width.connect_value_notify(move |row| {
            let value = row.value() as u32;
            on(&|s| s.display.window_width = value);
        });
    }
    {
        let on = on.clone();
        height.connect_value_notify(move |row| {
            let value = row.value() as u32;
            on(&|s| s.display.window_height = value);
        });
    }
    {
        let on = on.clone();
        resolution.connect_selected_notify(move |row| {
            let setting = RESOLUTIONS[row.selected() as usize].1;
            on(&|s| s.display.internal_resolution = setting);
        });
    }
    {
        let on = on.clone();
        volume.connect_value_notify(move |row| {
            let value = row.value() as u32;
            on(&|s| s.audio.volume = value);
        });
    }
    vibration.connect_active_notify(move |row| {
        let active = row.is_active();
        on(&|s| s.input.vibration = active);
    });
    dialog.present(Some(&state.window));
}

fn show_about(state: &Rc<State>) {
    let about = adw::AboutDialog::builder()
        .application_name("Echora")
        .application_icon("applications-games")
        .version(env!("CARGO_PKG_VERSION"))
        .developer_name("Echora contributors")
        .license_type(gtk::License::Gpl30)
        .website("https://github.com/Mars7x/echora")
        .comments(
            "An experimental native recompilation of The Legend of Zelda: Ocarina of \
             Time for Nintendo Switch 2. Requires your own legally obtained copy of the game.",
        )
        .build();
    about.present(Some(&state.window));
}
