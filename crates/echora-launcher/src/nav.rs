//! Focus by direction: arrow keys and a controller's D-pad move to the
//! nearest control that way, as on a console, while Tab keeps the reading
//! order.
//!
//! Every control the keyboard can reach is a [`Stops`] entry: it gets a
//! focus handle that lasts as long as the launcher, keyed by its element id,
//! so a control can be focused before it is drawn (the chosen item of a menu
//! that is just opening) and focus can return to it later. Each frame the
//! controls record where they were laid out, and a move picks among those.
//!
//! One focus ring serves every control, drawn above everything (menus
//! included). It springs in when it appears and from the control focus left
//! to the one it went to.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::time::{Duration, Instant};

use gpui::prelude::*;
use gpui::{
    App, Bounds, Div, Element, ElementId, FocusHandle, KeyDownEvent, Pixels, Stateful, Window,
    actions, canvas, deferred,
};

use crate::ui;

actions!(
    nav,
    [
        /// Presses the focused control, as a controller's A button does.
        Activate
    ]
);

/// Controls that a click, [`Activate`], or Enter or Space while focused
/// presses.
pub trait Press {
    fn on_press(self, listener: impl Fn(&(), &mut Window, &mut App) + 'static) -> Self;
}

impl Press for Stateful<Div> {
    fn on_press(self, listener: impl Fn(&(), &mut Window, &mut App) + 'static) -> Self {
        let listener = Rc::new(listener);
        let activate = listener.clone();
        let key = listener.clone();
        // GPUI clicks a focused control when Enter or Space is released;
        // pressing on the key going down instead feels immediate. Held keys
        // don't repeat the press.
        self.on_click(move |event, window, cx| {
            if !event.is_keyboard() {
                listener(&(), window, cx);
            }
        })
        .on_key_down(move |event: &KeyDownEvent, window, cx| {
            let stroke = &event.keystroke;
            if matches!(stroke.key.as_str(), "enter" | "space") && !stroke.modifiers.modified() {
                cx.stop_propagation();
                if !event.is_held {
                    key(&(), window, cx);
                }
            }
        })
        // Actions go to the focused element first, so only the focused
        // control answers Activate.
        .on_action(move |_: &Activate, window, cx| activate(&(), window, cx))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Direction {
    Left,
    Up,
    Right,
    Down,
}

impl Direction {
    fn reverse(self) -> Self {
        match self {
            Self::Left => Self::Right,
            Self::Up => Self::Down,
            Self::Right => Self::Left,
            Self::Down => Self::Up,
        }
    }
}

/// A move of focus, remembered so that moving straight back returns to the
/// control it left, even where another control is nearer that way (from a
/// sidebar tab to the button level with a lower tab, and back).
struct Move<T> {
    from: T,
    direction: Direction,
    to: T,
}

impl<T: PartialEq> Move<T> {
    /// The control to return to when moving `direction` from `at`.
    fn back(&self, at: &T, direction: Direction) -> Option<&T> {
        (self.to == *at && direction == self.direction.reverse()).then_some(&self.from)
    }
}

#[derive(Clone, Default)]
pub struct Stops(Rc<RefCell<Inner>>);

/// A control drawn this frame.
struct Stop {
    bounds: Bounds<Pixels>,
    /// Its corner radius in design pixels, which the ring follows.
    corner: f32,
    handle: FocusHandle,
}

#[derive(Default)]
struct Inner {
    handles: HashMap<ElementId, FocusHandle>,
    /// The controls drawn this frame, in paint order.
    frame: Vec<Stop>,
    /// The focus ring, while one shows.
    ring: Option<Glide>,
    last_move: Option<Move<FocusHandle>>,
}

/// How long the ring takes to spring to a newly focused control.
const GLIDE: Duration = Duration::from_millis(300);

/// The focus ring springing from one shape to the control it now marks.
struct Glide {
    handle: FocusHandle,
    from: ui::RingShape,
    to: ui::RingShape,
    start: Instant,
}

impl Glide {
    fn at(&self, now: Instant) -> ui::RingShape {
        let t = now.saturating_duration_since(self.start).as_secs_f32() / GLIDE.as_secs_f32();
        // Fully opaque well before the spring settles, as things appear.
        self.from
            .lerp(&self.to, ui::spring(t), ui::ease_out(t / 0.4))
    }
}

impl Stops {
    /// Forgets last frame's layout; the controls drawn next record theirs.
    pub fn begin_frame(&self) {
        self.0.borrow_mut().frame.clear();
    }

    /// The focus handle of the control `id`.
    pub fn handle(&self, id: impl Into<ElementId>, cx: &App) -> FocusHandle {
        self.0
            .borrow_mut()
            .handles
            .entry(id.into())
            .or_insert_with(|| cx.focus_handle().tab_stop(true))
            .clone()
    }

    /// Makes `el`, whose corners are rounded `corner` design pixels, a
    /// stop. `el` must have an id.
    pub fn add(&self, el: Stateful<Div>, corner: f32, cx: &App) -> Stateful<Div> {
        let id = Element::id(&el).expect("a focus stop needs an element id");
        let handle = self.handle(id, cx);
        let stops = self.clone();
        el.track_focus(&handle.clone()).child(
            canvas(
                move |bounds, _, _| {
                    stops.0.borrow_mut().frame.push(Stop {
                        bounds,
                        corner,
                        handle: handle.clone(),
                    })
                },
                |_, _, _, _| {},
            )
            .absolute()
            .inset_0(),
        )
    }

    /// The focus ring, above everything else on screen: around the focused
    /// control while `show` (the keyboard or a controller is in use),
    /// springing there from the last control it marked when `motion`
    /// allows.
    pub fn ring(&self, show: bool, motion: bool) -> impl IntoElement {
        let stops = self.clone();
        deferred(
            canvas(
                |_, _, _| {},
                move |_, _, window, _| stops.paint_ring(show, motion, window),
            )
            .absolute()
            .inset_0(),
        )
        .with_priority(3)
    }

    fn paint_ring(&self, show: bool, motion: bool, window: &mut Window) {
        let mut inner = self.0.borrow_mut();
        let Inner { frame, ring, .. } = &mut *inner;
        let focused = frame.iter().find(|stop| stop.handle.is_focused(window));
        let (Some(stop), true) = (focused, show) else {
            *ring = None;
            return;
        };
        let target = ui::RingShape::around(stop.bounds, stop.corner, window);
        let now = Instant::now();
        match ring {
            // The same control: follow it as it moves (a sheet springing in).
            Some(glide) if glide.handle == stop.handle => glide.to = target,
            Some(glide) if motion => {
                *glide = Glide {
                    handle: stop.handle.clone(),
                    from: glide.at(now),
                    to: target,
                    start: now,
                }
            }
            // Appearing: spring in from a little larger while fading in.
            _ => {
                *ring = Some(Glide {
                    handle: stop.handle.clone(),
                    from: if motion { target.appearing() } else { target },
                    to: target,
                    start: now,
                })
            }
        }
        let glide = ring.as_ref().expect("the ring was just set");
        glide.at(now).paint(window);
        if now < glide.start + GLIDE {
            window.request_animation_frame();
        }
    }

    /// The control to focus when moving `direction` from `focused`: back to
    /// the control the last move left when this undoes it, otherwise the
    /// nearest one that way, or the first control when focus is on none of
    /// them. `None` leaves focus where it is.
    pub fn toward(
        &self,
        focused: Option<&FocusHandle>,
        direction: Direction,
    ) -> Option<FocusHandle> {
        let mut inner = self.0.borrow_mut();
        let Inner {
            frame, last_move, ..
        } = &mut *inner;
        let from = focused.and_then(|focused| frame.iter().find(|stop| stop.handle == *focused));
        let Some(from) = from else {
            return frame.first().map(|stop| stop.handle.clone());
        };
        let drawn = |handle: &FocusHandle| frame.iter().any(|stop| stop.handle == *handle);
        let back = last_move
            .as_ref()
            .and_then(|last| last.back(&from.handle, direction))
            .filter(|handle| drawn(handle))
            .cloned();
        let rect = Rect::from(from.bounds);
        let to = back.or_else(|| {
            frame
                .iter()
                .filter_map(|stop| {
                    Some((
                        rect.distance(&Rect::from(stop.bounds), direction)?,
                        &stop.handle,
                    ))
                })
                .min_by(|a, b| a.0.total_cmp(&b.0))
                .map(|(_, handle)| handle.clone())
        })?;
        *last_move = Some(Move {
            from: from.handle.clone(),
            direction,
            to: to.clone(),
        });
        Some(to)
    }
}

/// Bounds as plain numbers.
#[derive(Clone, Copy, Debug)]
struct Rect {
    left: f32,
    top: f32,
    right: f32,
    bottom: f32,
}

impl From<Bounds<Pixels>> for Rect {
    fn from(bounds: Bounds<Pixels>) -> Self {
        Self {
            left: bounds.left().into(),
            top: bounds.top().into(),
            right: bounds.right().into(),
            bottom: bounds.bottom().into(),
        }
    }
}

/// How far a control may overlap the one focused and still count as being
/// beside it, for controls whose rings or rounding touch.
const OVERLAP: f32 = 2.0;

/// Sideways distance counts this many times the distance ahead, so a move
/// prefers the control in line over a nearer one off to the side.
const SIDEWAYS: f32 = 3.0;

impl Rect {
    /// How far `to` is from `self` going `direction`, or `None` when it
    /// isn't that way.
    fn distance(&self, to: &Rect, direction: Direction) -> Option<f32> {
        let ahead = match direction {
            Direction::Left => self.left - to.right,
            Direction::Right => to.left - self.right,
            Direction::Up => self.top - to.bottom,
            Direction::Down => to.top - self.bottom,
        };
        if ahead < -OVERLAP {
            return None;
        }
        // The gap between the two across the move, zero where they overlap.
        let gap = |a0: f32, a1: f32, b0: f32, b1: f32| (b0 - a1).max(a0 - b1).max(0.0);
        let sideways = match direction {
            Direction::Left | Direction::Right => gap(self.top, self.bottom, to.top, to.bottom),
            Direction::Up | Direction::Down => gap(self.left, self.right, to.left, to.right),
        };
        Some(ahead.max(0.0) + sideways * SIDEWAYS)
    }
}

#[cfg(test)]
mod tests {
    use super::{Direction, Move, Rect};

    fn rect(left: f32, top: f32, width: f32, height: f32) -> Rect {
        Rect {
            left,
            top,
            right: left + width,
            bottom: top + height,
        }
    }

    #[test]
    fn moves_only_toward_controls_that_way() {
        let play = rect(40.0, 500.0, 120.0, 40.0);
        let settings = rect(168.0, 500.0, 100.0, 40.0);
        assert!(play.distance(&settings, Direction::Right).is_some());
        assert!(play.distance(&settings, Direction::Left).is_none());
        assert!(play.distance(&settings, Direction::Up).is_none());
        assert!(settings.distance(&play, Direction::Left).is_some());
    }

    #[test]
    fn prefers_the_control_in_line() {
        // A sidebar tab, the control level with it further right, and a
        // nearer one well below.
        let tab = rect(10.0, 100.0, 160.0, 36.0);
        let level = rect(600.0, 96.0, 120.0, 32.0);
        let below = rect(220.0, 300.0, 120.0, 32.0);
        let to_level = tab.distance(&level, Direction::Right).unwrap();
        let to_below = tab.distance(&below, Direction::Right).unwrap();
        assert!(to_level < to_below);
    }

    #[test]
    fn moving_straight_back_returns_where_focus_came_from() {
        let last = Move {
            from: "game",
            direction: Direction::Right,
            to: "reinstall",
        };
        assert_eq!(last.back(&"reinstall", Direction::Left), Some(&"game"));
        assert_eq!(last.back(&"reinstall", Direction::Up), None);
        assert_eq!(last.back(&"display", Direction::Left), None);
    }
}
