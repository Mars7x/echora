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
//!   contrast of at least 4.5:1, and Link's forest green as the one accent.
//! - Controls at least 32 px tall (40 px in the dock), each with hover,
//!   pressed and keyboard-focus states. The focus ring shows only while the
//!   keyboard is being used, as both guidelines ask.
//! - Spring motion, as libadwaita uses (a slight overshoot that settles),
//!   for things that appear or change; skipped when the desktop asks for
//!   reduced motion.
//! - Switches for on/off settings, pop-up menus for choices, boxed lists
//!   for settings rows.

use std::time::Duration;

use gpui::{
    Animation, AnimationExt as _, AnyElement, BorderStyle, Bounds, BoxShadow, Div, ElementId,
    FontWeight, Hsla, Pixels, Rems, SharedString, Stateful, Window, div, prelude::*, px, rems,
    rgba,
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
    /// Fully round: pills and circles.
    pub const FULL: f32 = f32::INFINITY;
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

/// Colours as `0xRRGGBBAA`, fixed to the author's GTK theme on every system
/// (github.com/Mars7x/mygtktheme): its window, view, sidebar, dialog,
/// popover, card and border colours. Selection is a lighter neutral and the
/// accent is the forest green of Link's tunic, not the theme's teal.
pub mod palette {
    /// The window, headers and sheets' content.
    pub const BACKDROP: u32 = 0x101010ff;
    /// The main view inside a sheet with a sidebar (Settings), as the
    /// theme's views.
    pub const VIEW: u32 = 0x101010ff;
    /// Plain sheets (About), as the theme's dialogs.
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
    /// Play, and the main action of the empty screen. 4.9:1 under its
    /// label.
    pub const ACCENT: u32 = 0x2f7a34ff;
    pub const ACCENT_HOVER: u32 = 0x36853bff;
    pub const ACCENT_PRESSED: u32 = 0x29692dff;
    pub const ON_ACCENT: u32 = 0xf5f5f5ff;
    pub const FOCUS: u32 = 0xf0f0f0d9;
    pub const SCRIM: u32 = 0x00000099;
    /// Pop-up menus, as the theme's popovers.
    pub const MENU: u32 = 0x191919ff;
    pub const SWITCH_OFF: u32 = 0x3a3a3aff;
}

/// A length in the design system's pixels, as rems (1 rem = 16 px).
pub fn u(pixels: f32) -> Rems {
    rems(pixels / 16.0)
}

/// `0xRRGGBBAA`.
pub fn color(hex: u32) -> Hsla {
    rgba(hex).into()
}

/// The focus ring: the same for every control, 2 px wide around its edge
/// and following its rounding. It reaches 1 px over the edge too, since two
/// smoothed edges meeting on the same pixels leave a faint dark seam.
/// Painted rather than a box shadow, which GPUI draws nothing for without
/// blur.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct RingShape {
    bounds: Bounds<Pixels>,
    radius: Pixels,
    opacity: f32,
}

impl RingShape {
    const WIDTH: f32 = 2.0;
    /// How far the ring reaches over the control's edge.
    const OVERLAP: f32 = 1.0;
    /// How much larger the ring starts when it appears, as it springs in.
    const APPEAR_GROWTH: f32 = 6.0;

    /// The ring around a control laid out in `bounds` with corners rounded
    /// `corner` design pixels ([`radius::FULL`] for pills and circles).
    pub fn around(bounds: Bounds<Pixels>, corner: f32, window: &Window) -> Self {
        let scale = window.rem_size() / px(16.0);
        let corner = (px(corner) * scale).min(bounds.size.height.min(bounds.size.width) / 2.0);
        let outset = px(Self::WIDTH);
        Self {
            bounds: bounds.dilate(outset),
            radius: corner + outset,
            opacity: 1.0,
        }
    }

    /// Where the ring starts when it appears: a little larger and clear.
    pub fn appearing(&self) -> Self {
        let growth = px(Self::APPEAR_GROWTH);
        Self {
            bounds: self.bounds.dilate(growth),
            radius: self.radius + growth,
            opacity: 0.0,
        }
    }

    /// The shape `motion` of the way from `self` to `to` (beyond 1
    /// overshoots), faded `fade` of the way, which must not.
    pub fn lerp(&self, to: &Self, motion: f32, fade: f32) -> Self {
        let mix = |a: Pixels, b: Pixels| a + (b - a) * motion;
        Self {
            bounds: Bounds {
                origin: gpui::point(
                    mix(self.bounds.origin.x, to.bounds.origin.x),
                    mix(self.bounds.origin.y, to.bounds.origin.y),
                ),
                size: gpui::size(
                    mix(self.bounds.size.width, to.bounds.size.width),
                    mix(self.bounds.size.height, to.bounds.size.height),
                ),
            },
            radius: mix(self.radius, to.radius),
            opacity: self.opacity + (to.opacity - self.opacity) * fade.clamp(0.0, 1.0),
        }
    }

    pub fn paint(&self, window: &mut Window) {
        window.paint_quad(gpui::quad(
            self.bounds,
            self.radius,
            gpui::transparent_black(),
            px(Self::WIDTH + Self::OVERLAP),
            color(palette::FOCUS).opacity(self.opacity),
            BorderStyle::Solid,
        ));
    }
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
        .h(u(height))
        .px(u(padding))
        .flex()
        .items_center()
        .justify_center()
        .rounded(u(radius::CONTROL))
        .bg(color(fill))
        .hover(|style| style.bg(color(hover)))
        .active(|style| style.bg(color(pressed)))
        .text_size(u(font))
        .font_weight(FontWeight::SEMIBOLD)
        .text_color(color(label_color))
        .child(label.into())
}

/// A round 24 px button holding a 16 px symbolic icon: the window controls
/// (close, minimize, maximize) and a sheet's close button. It stays 24 px at
/// every UI scale, as the desktop's own controls do, and its icon dims while
/// the window is in the background.
///
/// The circle is an SVG like the icon. GPUI snaps each SVG to whole pixels
/// from its centre, so two SVGs sharing a centre land on the same pixels; a
/// circle drawn as a rounded box would sit at the exact position instead,
/// which on a sheet centred to half a pixel puts the glyph off-centre.
pub fn round_button(id: impl Into<ElementId>, icon: &'static str, active: bool) -> Stateful<Div> {
    div()
        .id(id)
        .group(ROUND_BUTTON)
        .relative()
        .flex_none()
        .size(px(24.0))
        .flex()
        .items_center()
        .justify_center()
        .rounded_full()
        .child(
            gpui::svg()
                .path(CIRCLE)
                .id("circle")
                .absolute()
                .inset_0()
                .text_color(color(palette::FILL))
                .group_hover(ROUND_BUTTON, |style| {
                    style.text_color(color(palette::FILL_HOVER))
                })
                .group_active(ROUND_BUTTON, |style| {
                    style.text_color(color(palette::FILL_PRESSED))
                }),
        )
        .child(
            gpui::svg()
                .path(icon)
                .size(px(16.0))
                .text_color(color(if active {
                    palette::LABEL
                } else {
                    palette::SECONDARY_LABEL
                })),
        )
}

const ROUND_BUTTON: &str = "round-button";
const CIRCLE: &str = "icons/circle.svg";

/// An on/off switch, 40 by 24 px: the same dark knob in both states, on a
/// white track when on and a gray one when off. `animate` slides the knob,
/// for the switch that was just flipped.
pub fn switch(id: &'static str, on: bool, animate: bool) -> Stateful<Div> {
    // Knob travel: from 2 px in to 18 px in.
    let at = move |t: f32| u(2.0 + 16.0 * if on { t } else { 1.0 - t });
    let knob = div()
        .absolute()
        .top(u(2.0))
        .size(u(20.0))
        .rounded_full()
        .bg(color(palette::BACKDROP));
    let knob = if animate {
        knob.with_animation(
            ElementId::Name(format!("{id}-{on}").into()),
            Animation::new(Duration::from_millis(320)),
            move |knob, t| knob.left(at(spring(t))),
        )
        .into_any_element()
    } else {
        knob.left(at(1.0)).into_any_element()
    };
    div()
        .id(id)
        .relative()
        .flex_none()
        .w(u(40.0))
        .h(u(24.0))
        .rounded_full()
        .bg(color(if on {
            palette::LABEL
        } else {
            palette::SWITCH_OFF
        }))
        .child(knob)
}

/// A slightly underdamped spring, as libadwaita animates: position over
/// normalized time, overshooting by about 5% and settling by `t = 1`.
pub fn spring(t: f32) -> f32 {
    // Damping ratio 0.7 and a natural frequency that settles within the
    // animation's duration.
    const DAMPING: f32 = 0.7;
    const FREQUENCY: f32 = 10.0;
    if t >= 1.0 {
        return 1.0;
    }
    let damped = FREQUENCY * (1.0 - DAMPING * DAMPING).sqrt();
    let decay = (-DAMPING * FREQUENCY * t).exp();
    1.0 - decay * ((damped * t).cos() + DAMPING * FREQUENCY / damped * (damped * t).sin())
}

/// Ease-out for opacity, which must not overshoot.
pub fn ease_out(t: f32) -> f32 {
    1.0 - (1.0 - t.clamp(0.0, 1.0)).powi(3)
}

/// How long an element takes to spring into place.
pub const SPRING: Duration = Duration::from_millis(380);

/// Fades `el` in while it springs `rise` px into place, once per `id`.
pub fn appear<E>(el: E, id: impl Into<ElementId>, rise: f32, motion: bool) -> AnyElement
where
    E: Styled + IntoElement + 'static,
{
    if !motion {
        return el.into_any_element();
    }
    el.with_animation(id, Animation::new(SPRING), move |el, t| {
        // Fully opaque by 40% of the way, while the spring still settles.
        el.opacity(ease_out(t / 0.4))
            .relative()
            .top(u(rise * (1.0 - spring(t))))
    })
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
        .h(u(32.0))
        .pl(u(space::M))
        .pr(u(space::S))
        .flex()
        .items_center()
        .gap(u(space::XS + 2.0))
        .rounded(u(radius::CONTROL))
        .bg(color(palette::FILL))
        .hover(|style| style.bg(color(palette::FILL_HOVER)))
        .active(|style| style.bg(color(palette::FILL_PRESSED)))
        .text_size(u(text::BODY))
        .text_color(color(palette::LABEL))
        .child(label)
        .child(
            gpui::svg()
                .path("icons/chevron-down.svg")
                .size(u(16.0))
                .text_color(color(palette::SECONDARY_LABEL)),
        )
}

/// A pop-up menu's panel; `items` come from [`menu_item`].
pub fn menu(items: impl IntoIterator<Item = Stateful<Div>>) -> Div {
    div()
        .w(u(200.0))
        .p(u(space::XS))
        .flex()
        .flex_col()
        .gap(u(2.0))
        .rounded(u(radius::CARD))
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
        .h(u(32.0))
        .px(u(space::S))
        .flex()
        .items_center()
        .gap(u(space::S))
        .rounded(u(radius::INNER))
        .hover(|style| style.bg(color(palette::FILL)))
        .active(|style| style.bg(color(palette::FILL_HOVER)))
        .text_size(u(text::BODY))
        .text_color(color(palette::LABEL))
        .child(div().size(u(16.0)).when(selected, |el| {
            el.child(
                gpui::svg()
                    .path("icons/check.svg")
                    .size(u(16.0))
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
        .rounded(u(radius::CARD))
        .bg(color(palette::CARD));
    for (index, row) in rows.into_iter().enumerate() {
        card = card.child(row);
        if index + 1 < count {
            card = card.child(
                div()
                    .mx(u(space::L))
                    .h(u(1.0))
                    .bg(color(palette::SEPARATOR)),
            );
        }
    }
    div()
        .flex()
        .flex_col()
        .gap(u(space::S))
        .when_some(title, |el, title| {
            el.child(
                div()
                    .px(u(space::XS))
                    .text_size(u(text::CALLOUT))
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
        .gap(u(2.0))
        .child(
            div()
                .text_size(u(text::BODY))
                .text_color(color(palette::LABEL))
                .truncate()
                .child(title.into()),
        )
        .when_some(subtitle, |el, subtitle| {
            el.child(
                div()
                    .text_size(u(text::CALLOUT))
                    .text_color(color(palette::SECONDARY_LABEL))
                    .child(subtitle),
            )
        })
}

/// A list row: optional leading content, titles, and a trailing control.
pub fn row(leading: Option<AnyElement>, titles: Div, trailing: Option<AnyElement>) -> Div {
    div()
        .min_h(u(52.0))
        .px(u(space::L))
        .py(u(space::S + 2.0))
        .flex()
        .items_center()
        .gap(u(space::M))
        .children(leading)
        .child(titles)
        .children(trailing)
}

/// A thin progress bar; `fraction` runs from 0 to 1.
pub fn progress(fraction: f32) -> Div {
    div()
        .h(u(4.0))
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
        .px(u(space::L))
        .py(u(space::S + 2.0))
        .rounded_full()
        .bg(color(palette::MATERIAL))
        .border_1()
        .border_color(color(palette::BORDER))
        .text_size(u(text::BODY))
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

#[cfg(test)]
mod tests {
    use super::{ease_out, spring};

    #[test]
    fn the_spring_overshoots_a_little_and_settles() {
        assert!(spring(0.0).abs() < 1e-6);
        assert_eq!(spring(1.0), 1.0);
        let peak = (0..=100)
            .map(|i| spring(i as f32 / 100.0))
            .fold(0.0, f32::max);
        assert!(peak > 1.02 && peak < 1.1, "overshoot {peak}");
        assert!((spring(0.95) - 1.0).abs() < 0.01, "not settled");
        assert_eq!(ease_out(2.0), 1.0);
    }
}
