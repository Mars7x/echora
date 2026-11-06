//! Client-side window decorations: which window buttons the desktop wants
//! and where, their icons, and the resize edges around the window.

use std::borrow::Cow;

use gpui::{AssetSource, Pixels, Point, ResizeEdge, SharedString, Size};

/// A window button.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowButton {
    Minimize,
    Maximize,
    Close,
}

impl WindowButton {
    pub fn icon(self, maximized: bool) -> &'static str {
        match self {
            Self::Minimize => "icons/minimize.svg",
            Self::Maximize if maximized => "icons/restore.svg",
            Self::Maximize => "icons/maximize.svg",
            Self::Close => "icons/close.svg",
        }
    }

    pub fn id(self) -> &'static str {
        match self {
            Self::Minimize => "window-minimize",
            Self::Maximize => "window-maximize",
            Self::Close => "window-close",
        }
    }
}

/// Which buttons sit on each side of the title bar.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ButtonLayout {
    pub left: Vec<WindowButton>,
    pub right: Vec<WindowButton>,
}

impl ButtonLayout {
    /// The desktop's layout, or GNOME's default (close only, on the right)
    /// when it can't be read.
    pub fn from_desktop() -> Self {
        desktop_layout()
            .map(|layout| Self::parse(&layout))
            .unwrap_or_else(|| Self::parse("appmenu:close"))
    }

    /// Parses the GNOME `button-layout` format, e.g. `"close:minimize,maximize"`
    /// (left of the colon goes left). Unknown entries such as `appmenu` and
    /// `icon` are ignored.
    pub fn parse(layout: &str) -> Self {
        let side = |part: &str| {
            part.split(',')
                .filter_map(|name| match name.trim() {
                    "minimize" => Some(WindowButton::Minimize),
                    "maximize" => Some(WindowButton::Maximize),
                    "close" => Some(WindowButton::Close),
                    _ => None,
                })
                .collect()
        };
        let layout = layout.trim().trim_matches('\'');
        let (left, right) = layout.split_once(':').unwrap_or(("", layout));
        Self {
            left: side(left),
            right: side(right),
        }
    }
}

/// GNOME's setting; other desktops that draw client-side decorations
/// (Phosh, Budgie, Pantheon) honour the same key.
#[cfg(target_os = "linux")]
fn desktop_layout() -> Option<String> {
    let output = std::process::Command::new("gsettings")
        .args(["get", "org.gnome.desktop.wm.preferences", "button-layout"])
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).into_owned())
}

#[cfg(not(target_os = "linux"))]
fn desktop_layout() -> Option<String> {
    None
}

/// The window edge or corner under `position`, if it is within `inset` of
/// the window's outer bounds.
pub fn resize_edge(
    position: Point<Pixels>,
    inset: Pixels,
    size: Size<Pixels>,
) -> Option<ResizeEdge> {
    let top = position.y < inset;
    let bottom = position.y > size.height - inset;
    let left = position.x < inset;
    let right = position.x > size.width - inset;
    Some(match (top, bottom, left, right) {
        (true, _, true, _) => ResizeEdge::TopLeft,
        (true, _, _, true) => ResizeEdge::TopRight,
        (_, true, true, _) => ResizeEdge::BottomLeft,
        (_, true, _, true) => ResizeEdge::BottomRight,
        (true, ..) => ResizeEdge::Top,
        (_, true, ..) => ResizeEdge::Bottom,
        (_, _, true, _) => ResizeEdge::Left,
        (_, _, _, true) => ResizeEdge::Right,
        _ => return None,
    })
}

/// The launcher's built-in icons, drawn as 16-pixel symbolic glyphs.
pub struct Icons;

const ICONS: [(&str, &str); 4] = [
    (
        "icons/close.svg",
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"><path d="M4.5 4.5l7 7M11.5 4.5l-7 7" stroke="#000" stroke-width="1.6" stroke-linecap="round" fill="none"/></svg>"##,
    ),
    (
        "icons/minimize.svg",
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"><path d="M4.5 10.5h7" stroke="#000" stroke-width="1.6" stroke-linecap="round" fill="none"/></svg>"##,
    ),
    (
        "icons/maximize.svg",
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"><rect x="4.5" y="4.5" width="7" height="7" rx="1" stroke="#000" stroke-width="1.5" fill="none"/></svg>"##,
    ),
    (
        "icons/restore.svg",
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"><rect x="4" y="6" width="6" height="6" rx="1" stroke="#000" stroke-width="1.4" fill="none"/><path d="M6.5 4h4.5a1 1 0 0 1 1 1v4.5" stroke="#000" stroke-width="1.4" fill="none"/></svg>"##,
    ),
];

impl AssetSource for Icons {
    fn load(&self, path: &str) -> gpui::Result<Option<Cow<'static, [u8]>>> {
        Ok(ICONS
            .iter()
            .find(|(name, _)| *name == path)
            .map(|(_, svg)| Cow::Borrowed(svg.as_bytes())))
    }

    fn list(&self, path: &str) -> gpui::Result<Vec<SharedString>> {
        Ok(ICONS
            .iter()
            .filter(|(name, _)| name.starts_with(path))
            .map(|(name, _)| SharedString::from(*name))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use WindowButton::*;

    #[test]
    fn parses_gnome_layouts() {
        assert_eq!(
            ButtonLayout::parse("'appmenu:close'\n"),
            ButtonLayout {
                left: vec![],
                right: vec![Close],
            }
        );
        assert_eq!(
            ButtonLayout::parse("close,minimize,maximize:"),
            ButtonLayout {
                left: vec![Close, Minimize, Maximize],
                right: vec![],
            }
        );
        assert_eq!(
            ButtonLayout::parse("icon:minimize,maximize,close"),
            ButtonLayout {
                left: vec![],
                right: vec![Minimize, Maximize, Close],
            }
        );
    }

    #[test]
    fn finds_resize_edges() {
        let size = Size {
            width: Pixels::from(100.0),
            height: Pixels::from(80.0),
        };
        let at = |x: f32, y: f32| resize_edge(gpui::point(x.into(), y.into()), 10.0.into(), size);
        assert_eq!(at(2.0, 2.0), Some(ResizeEdge::TopLeft));
        assert_eq!(at(98.0, 78.0), Some(ResizeEdge::BottomRight));
        assert_eq!(at(50.0, 2.0), Some(ResizeEdge::Top));
        assert_eq!(at(98.0, 40.0), Some(ResizeEdge::Right));
        assert_eq!(at(50.0, 40.0), None);
    }
}
