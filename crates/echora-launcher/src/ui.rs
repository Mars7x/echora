//! The launcher's design system: one set of tokens and a few standard
//! controls, following what Apple's Human Interface Guidelines and the GNOME
//! HIG agree on.
//!
//! - A 4 px spacing grid; nested corner radii shrink by the padding between
//!   them (window 12, sheet 14, card 10, control 8, inner 6).
//! - One type scale (44/26/15/14/13/12) in Inter (the game title in
//!   Cinzel), bundled so every
//!   platform shows the same type with real weights; sentence case for
//!   every label.
//! - Semantic dark colours from the author's GTK theme (label, secondary
//!   label, separator, fills), a lighter neutral for selection, text
//!   contrast of at least 4.5:1, and the light foreground as the accent.
//! - Controls at least 32 px tall (40 px in the dock), each with hover,
//!   pressed and keyboard-focus states. The focus ring shows only while the
//!   keyboard is being used, as both guidelines ask.
//! - Short motion (120-200 ms, easing out) for things that appear or
//!   change, skipped when the desktop asks for reduced motion.
//! - Switches for on/off settings, pop-up menus for choices, boxed lists
//!   for settings rows.

use std::time::Duration;

use gpui::{
    Animation, AnimationExt as _, AnyElement, BoxShadow, Div, ElementId, FontWeight, Hsla,
    SharedString, Stateful, div, ease_out_quint, prelude::*, px, rgba,
};

/// Spacing on a 4 px grid.
pub mod space {
    pub const XS: f32 = 4.0;
    pub const S: f32 = 8.0;
    pub const M: f32 = 12.0;
    pub const L: f32 = 16.0;
    pub const XL: f32 = 24.0;
}

/// Corner radii, nested so each is the outer radius less the padding.
pub mod radius {
    pub const WINDOW: f32 = 12.0;
    pub const SHEET: f32 = 14.0;
    pub const DOCK: f32 = 16.0;
    pub const CARD: f32 = 10.0;
    pub const CONTROL: f32 = 8.0;
    pub const INNER: f32 = 6.0;
}

/// The type scale, in pixels.
pub mod text {
    pub const HERO: f32 = 44.0;
    pub const LARGE_TITLE: f32 = 26.0;
    pub const HEADLINE: f32 = 15.0;
    pub const BODY: f32 = 14.0;
    pub const CALLOUT: f32 = 13.0;
    pub const CAPTION: f32 = 12.0;
}

/// Colours as `0xRRGGBBAA`, from the author's GTK theme
/// (github.com/Mars7x/mygtktheme), keeping its neutrals and selection
/// colour; the accent is its light foreground rather than the teal.
pub mod palette {
    /// The window, headers and sheets' content.
    pub const BACKDROP: u32 = 0x101010ff;
    /// Sheets, as the theme's dialogs.
    pub const SURFACE: u32 = 0x191919ff;
    pub const SIDEBAR: u32 = 0x171717ff;
    /// Floating bars and toasts.
    pub const MATERIAL: u32 = 0x252525f0;
    /// Boxed lists.
    pub const CARD: u32 = 0x1c1c1cff;
    pub const SEPARATOR: u32 = 0x262626ff;
    pub const BORDER: u32 = 0x303030ff;
    pub const LABEL: u32 = 0xf0f0f0ff;
    /// 7:1 on the window background.
    pub const SECONDARY_LABEL: u32 = 0x9a9a9aff;
    pub const FILL: u32 = 0x242424ff;
    pub const FILL_HOVER: u32 = 0x2b2b2bff;
    pub const FILL_PRESSED: u32 = 0x303030ff;
    /// Selected items: a lighter neutral, no colour.
    pub const SELECTED: u32 = 0x333333ff;
    /// The thin edge around the window.
    pub const WINDOW_EDGE: u32 = 0xffffff0f;
    pub const ACCENT: u32 = 0xf0f0f0ff;
    pub const ACCENT_HOVER: u32 = 0xddddddff;
    pub const ACCENT_PRESSED: u32 = 0xc8c8c8ff;
    pub const ON_ACCENT: u32 = 0x101010ff;
    pub const FOCUS: u32 = 0xf0f0f0a6;
    pub const SCRIM: u32 = 0x00000099;
    /// Pop-up menus, as the theme's popovers.
    pub const MENU: u32 = 0x191919ff;
    pub const SWITCH_OFF: u32 = 0x3a3a3aff;
}

/// `0xRRGGBBAA`.
pub fn color(hex: u32) -> Hsla {
    rgba(hex).into()
}

/// The ring drawn around the focused control during keyboard navigation.
fn focus_ring() -> Vec<BoxShadow> {
    vec![BoxShadow {
        color: color(palette::FOCUS),
        blur_radius: px(0.0),
        spread_radius: px(2.0),
        offset: gpui::point(px(0.0), px(0.0)),
    }]
}

/// Makes `el` a keyboard stop, with a focus ring while `keyboard` is on.
/// Elements behind an open sheet pass `focusable: false`.
pub fn focusable(el: Stateful<Div>, focusable: bool, keyboard: bool) -> Stateful<Div> {
    if !focusable {
        return el;
    }
    el.tab_index(0)
        .when(keyboard, |el| el.focus(|style| style.shadow(focus_ring())))
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// The one main action of a view.
    Primary,
    Secondary,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Size {
    /// 32 px, in sheets and lists.
    Regular,
    /// 40 px, in the dock.
    Large,
}

/// A text button.
pub fn button(
    id: impl Into<gpui::ElementId>,
    label: impl Into<SharedString>,
    kind: Kind,
    size: Size,
) -> Stateful<Div> {
    let (fill, hover, pressed, label_color) = match kind {
        Kind::Primary => (
            palette::ACCENT,
            palette::ACCENT_HOVER,
            palette::ACCENT_PRESSED,
            palette::ON_ACCENT,
        ),
        Kind::Secondary => (
            palette::FILL,
            palette::FILL_HOVER,
            palette::FILL_PRESSED,
            palette::LABEL,
        ),
    };
    let (height, padding, font) = match size {
        Size::Regular => (32.0, space::M, text::BODY),
        Size::Large => (40.0, space::L + space::XS, text::HEADLINE),
    };
    div()
        .id(id)
        .flex_none()
        .h(px(height))
        .px(px(padding))
        .flex()
        .items_center()
        .justify_center()
        .rounded(px(radius::CONTROL))
        .bg(color(fill))
        .hover(|style| style.bg(color(hover)))
        .active(|style| style.bg(color(pressed)))
        .cursor_pointer()
        .text_size(px(font))
        .font_weight(FontWeight::SEMIBOLD)
        .text_color(color(label_color))
        .child(label.into())
}

/// A round 24 px icon button, like the window's own buttons.
pub fn icon_button(id: impl Into<gpui::ElementId>, icon: &'static str) -> Stateful<Div> {
    div()
        .id(id)
        .flex_none()
        .size(px(24.0))
        .flex()
        .items_center()
        .justify_center()
        .rounded_full()
        .bg(color(palette::FILL))
        .hover(|style| style.bg(color(palette::FILL_HOVER)))
        .active(|style| style.bg(color(palette::FILL_PRESSED)))
        .child(
            gpui::svg()
                .path(icon)
                .size(px(16.0))
                .text_color(color(palette::LABEL)),
        )
}

/// An on/off switch, 40 by 24 px: the same dark knob in both states, on a
/// white track when on and a gray one when off. `animate` slides the knob,
/// for the switch that was just flipped.
pub fn switch(id: &'static str, on: bool, animate: bool) -> Stateful<Div> {
    // Knob travel: from 2 px in to 18 px in.
    let at = move |t: f32| px(2.0 + 16.0 * if on { t } else { 1.0 - t });
    let knob = div()
        .absolute()
        .top(px(2.0))
        .size(px(20.0))
        .rounded_full()
        .bg(color(palette::BACKDROP));
    let knob = if animate {
        knob.with_animation(
            ElementId::Name(format!("{id}-{on}").into()),
            Animation::new(Duration::from_millis(140)).with_easing(ease_out_quint()),
            move |knob, t| knob.left(at(t)),
        )
        .into_any_element()
    } else {
        knob.left(at(1.0)).into_any_element()
    };
    div()
        .id(id)
        .relative()
        .flex_none()
        .w(px(40.0))
        .h(px(24.0))
        .rounded_full()
        .cursor_pointer()
        .bg(color(if on {
            palette::LABEL
        } else {
            palette::SWITCH_OFF
        }))
        .child(knob)
}

/// Fades `el` in while it rises `rise` px into place, once per `id`.
pub fn appear<E>(el: E, id: impl Into<ElementId>, rise: f32, motion: bool) -> AnyElement
where
    E: Styled + IntoElement + 'static,
{
    if !motion {
        return el.into_any_element();
    }
    el.with_animation(
        id,
        Animation::new(Duration::from_millis(180)).with_easing(ease_out_quint()),
        move |el, t| el.opacity(t).relative().top(px(rise * (1.0 - t))),
    )
    .into_any_element()
}

/// Whether the desktop allows animations (GNOME's `enable-animations`).
pub fn motion_enabled() -> bool {
    #[cfg(target_os = "linux")]
    {
        let setting = std::process::Command::new("gsettings")
            .args(["get", "org.gnome.desktop.interface", "enable-animations"])
            .output();
        if let Ok(output) = setting
            && output.status.success()
        {
            return String::from_utf8_lossy(&output.stdout).trim() != "false";
        }
    }
    true
}

/// A pop-up button: the current choice and a chevron; it opens a [`menu`].
pub fn popup_button(id: impl Into<ElementId>, label: &'static str) -> Stateful<Div> {
    div()
        .id(id)
        .flex_none()
        .h(px(32.0))
        .pl(px(space::M))
        .pr(px(space::S))
        .flex()
        .items_center()
        .gap(px(space::XS + 2.0))
        .rounded(px(radius::CONTROL))
        .bg(color(palette::FILL))
        .hover(|style| style.bg(color(palette::FILL_HOVER)))
        .active(|style| style.bg(color(palette::FILL_PRESSED)))
        .cursor_pointer()
        .text_size(px(text::BODY))
        .text_color(color(palette::LABEL))
        .child(label)
        .child(
            gpui::svg()
                .path("icons/chevron-down.svg")
                .size(px(16.0))
                .text_color(color(palette::SECONDARY_LABEL)),
        )
}

/// A pop-up menu's panel; `items` come from [`menu_item`].
pub fn menu(items: impl IntoIterator<Item = Stateful<Div>>) -> Div {
    div()
        .w(px(200.0))
        .p(px(space::XS))
        .flex()
        .flex_col()
        .gap(px(2.0))
        .rounded(px(radius::CARD))
        .border_1()
        .border_color(color(palette::BORDER))
        .bg(color(palette::MENU))
        .shadow(vec![BoxShadow {
            color: Hsla::black().opacity(0.35),
            blur_radius: px(16.0),
            spread_radius: px(0.0),
            offset: gpui::point(px(0.0), px(6.0)),
        }])
        .children(items)
}

/// One choice in a pop-up menu, with a check mark when it is selected.
pub fn menu_item(id: impl Into<ElementId>, label: &'static str, selected: bool) -> Stateful<Div> {
    div()
        .id(id)
        .h(px(32.0))
        .px(px(space::S))
        .flex()
        .items_center()
        .gap(px(space::S))
        .rounded(px(radius::INNER))
        .cursor_pointer()
        .hover(|style| style.bg(color(palette::FILL)))
        .active(|style| style.bg(color(palette::FILL_HOVER)))
        .text_size(px(text::BODY))
        .text_color(color(palette::LABEL))
        .child(div().size(px(16.0)).when(selected, |el| {
            el.child(
                gpui::svg()
                    .path("icons/check.svg")
                    .size(px(16.0))
                    .text_color(color(palette::LABEL)),
            )
        }))
        .child(label)
}

/// A boxed list: rows on a card, separated by thin lines, with an optional
/// group title above.
pub fn group(title: Option<&'static str>, rows: Vec<AnyElement>) -> Div {
    let count = rows.len();
    let mut card = div()
        .flex()
        .flex_col()
        .rounded(px(radius::CARD))
        .bg(color(palette::CARD));
    for (index, row) in rows.into_iter().enumerate() {
        card = card.child(row);
        if index + 1 < count {
            card = card.child(
                div()
                    .mx(px(space::L))
                    .h(px(1.0))
                    .bg(color(palette::SEPARATOR)),
            );
        }
    }
    div()
        .flex()
        .flex_col()
        .gap(px(space::S))
        .when_some(title, |el, title| {
            el.child(
                div()
                    .px(px(space::XS))
                    .text_size(px(text::CALLOUT))
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(color(palette::SECONDARY_LABEL))
                    .child(title),
            )
        })
        .child(card)
}

/// A row's title with an optional subtitle under it.
pub fn titles(title: impl Into<SharedString>, subtitle: Option<SharedString>) -> Div {
    div()
        .flex()
        .flex_col()
        .flex_1()
        .min_w_0()
        .gap(px(2.0))
        .child(
            div()
                .text_size(px(text::BODY))
                .text_color(color(palette::LABEL))
                .truncate()
                .child(title.into()),
        )
        .when_some(subtitle, |el, subtitle| {
            el.child(
                div()
                    .text_size(px(text::CALLOUT))
                    .text_color(color(palette::SECONDARY_LABEL))
                    .child(subtitle),
            )
        })
}

/// A list row: optional leading content, titles, and a trailing control.
pub fn row(leading: Option<AnyElement>, titles: Div, trailing: Option<AnyElement>) -> Div {
    div()
        .min_h(px(52.0))
        .px(px(space::L))
        .py(px(space::S + 2.0))
        .flex()
        .items_center()
        .gap(px(space::M))
        .children(leading)
        .child(titles)
        .children(trailing)
}

/// A thin progress bar; `fraction` runs from 0 to 1.
pub fn progress(fraction: f32) -> Div {
    div()
        .h(px(4.0))
        .w_full()
        .rounded_full()
        .bg(color(palette::FILL))
        .child(
            div()
                .h_full()
                .rounded_full()
                .bg(color(palette::ACCENT))
                .w(gpui::relative(fraction.clamp(0.0, 1.0))),
        )
}

/// A short message floating above the dock.
pub fn toast(message: SharedString) -> Div {
    div()
        .px(px(space::L))
        .py(px(space::S + 2.0))
        .rounded_full()
        .bg(color(palette::MATERIAL))
        .border_1()
        .border_color(color(palette::BORDER))
        .text_size(px(text::BODY))
        .text_color(color(palette::LABEL))
        .child(message)
}

/// The game title's typeface: engraved Roman capitals in the spirit of the
/// series' logo.
pub const TITLE_FONT: &str = "Cinzel";

/// Inter for everything, in the three weights the launcher uses, and Cinzel
/// for the game title; both SIL Open Font License (see `assets/fonts`).
const FONTS: [&[u8]; 5] = [
    include_bytes!("../assets/fonts/Inter-Regular.ttf"),
    include_bytes!("../assets/fonts/Inter-SemiBold.ttf"),
    include_bytes!("../assets/fonts/Inter-Bold.ttf"),
    include_bytes!("../assets/fonts/Cinzel-Regular.ttf"),
    include_bytes!("../assets/fonts/Cinzel-Bold.ttf"),
];

/// Registers the bundled font and returns its family, or the platform's UI
/// font if registering fails.
pub fn load_font(cx: &gpui::App) -> SharedString {
    let fonts = FONTS
        .iter()
        .map(|font| std::borrow::Cow::Borrowed(*font))
        .collect();
    match cx.text_system().add_fonts(fonts) {
        Ok(()) => "Inter".into(),
        Err(error) => {
            eprintln!("echora-launcher: bundled font not loaded: {error}");
            ".SystemUIFont".into()
        }
    }
}
