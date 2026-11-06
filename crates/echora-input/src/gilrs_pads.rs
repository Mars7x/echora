//! Windows and macOS backend: gamepads through gilrs (XInput and Windows
//! Gaming Input, or GameController). Buttons are mapped by position as on
//! Linux: the bottom face button is B, the right one A, the top one X and
//! the left one Y. Several pads at once act as one controller.
//!
//! One thread owns the gilrs context, polls it at 250 Hz and plays rumble
//! requests (low band on the strong motor, high band on the weak one) as
//! short force-feedback effects that the next request replaces.

use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::Duration;

use gilrs::ff::{BaseEffect, BaseEffectType, EffectBuilder, Replay, Ticks};
use gilrs::{Axis, Button, Gilrs};

use crate::pad::{PRESS, Pad, button, directions, stick};

#[derive(Debug, Default)]
pub struct Gamepads {
    pad: Mutex<Pad>,
    connected: Mutex<bool>,
    rumble: Mutex<Option<Sender<(f32, f32)>>>,
}

impl Gamepads {
    /// The merged state of every connected pad.
    pub fn state(&self) -> Pad {
        *self.pad.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Whether any gamepad is connected.
    pub fn connected(&self) -> bool {
        *self
            .connected
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

/// The gamepads, polled from the first call on.
pub fn gamepads() -> &'static Gamepads {
    static GAMEPADS: OnceLock<&'static Gamepads> = OnceLock::new();
    GAMEPADS.get_or_init(|| {
        let gamepads: &'static Gamepads = Box::leak(Box::default());
        let (sender, receiver) = channel();
        *gamepads
            .rumble
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some(sender);
        let _ = std::thread::Builder::new()
            .name("echora-gamepads".into())
            .spawn(move || poll(gamepads, receiver));
        gamepads
    })
}

/// A vibration request (each band 0..=1). Every pad with force feedback
/// plays it; `device` names the guest's vibration device and is not needed
/// when all pads act as one controller.
pub fn rumble(_device: u32, low: f32, high: f32) {
    let sender = gamepads()
        .rumble
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone();
    if let Some(sender) = sender {
        let _ = sender.send((low, high));
    }
}

fn poll(gamepads: &'static Gamepads, rumble: Receiver<(f32, f32)>) {
    let mut gilrs = match Gilrs::new() {
        Ok(gilrs) => gilrs,
        Err(error) => {
            eprintln!("gamepads: {error}");
            return;
        }
    };
    // The playing effect; dropping it stops the motors.
    let mut _effect = None;
    loop {
        while gilrs.next_event().is_some() {}
        while let Ok((low, high)) = rumble.try_recv() {
            _effect = play_rumble(&mut gilrs, low, high);
        }
        let mut merged = Pad::default();
        let mut any = false;
        for (_, pad) in gilrs.gamepads() {
            any = true;
            merged = merged.merge(read(&pad));
        }
        *gamepads.pad.lock().unwrap_or_else(PoisonError::into_inner) = merged;
        *gamepads
            .connected
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = any;
        std::thread::sleep(Duration::from_millis(4));
    }
}

fn play_rumble(gilrs: &mut Gilrs, low: f32, high: f32) -> Option<gilrs::ff::Effect> {
    let ids: Vec<_> = gilrs
        .gamepads()
        .filter(|(_, pad)| pad.is_ff_supported())
        .map(|(id, _)| id)
        .collect();
    if ids.is_empty() || (low <= 0.0 && high <= 0.0) {
        return None;
    }
    let magnitude = |band: f32| (band.clamp(0.0, 1.0) * f32::from(u16::MAX)) as u16;
    let scheduling = Replay {
        play_for: Ticks::from_ms(200),
        ..Default::default()
    };
    let effect = EffectBuilder::new()
        .add_effect(BaseEffect {
            kind: BaseEffectType::Strong {
                magnitude: magnitude(low),
            },
            scheduling,
            ..Default::default()
        })
        .add_effect(BaseEffect {
            kind: BaseEffectType::Weak {
                magnitude: magnitude(high),
            },
            scheduling,
            ..Default::default()
        })
        .gamepads(&ids)
        .finish(gilrs)
        .ok()?;
    effect.play().ok()?;
    Some(effect)
}

fn read(pad: &gilrs::Gamepad<'_>) -> Pad {
    use button::*;
    let mut buttons = 0;
    for (from, to) in [
        (Button::East, A),
        (Button::South, B),
        (Button::North, X),
        (Button::West, Y),
        (Button::LeftTrigger, L),
        (Button::RightTrigger, R),
        (Button::Start, PLUS),
        (Button::Select, MINUS),
        (Button::LeftThumb, STICK_L),
        (Button::RightThumb, STICK_R),
        (Button::DPadLeft, LEFT),
        (Button::DPadUp, UP),
        (Button::DPadRight, RIGHT),
        (Button::DPadDown, DOWN),
    ] {
        if pad.is_pressed(from) {
            buttons |= to;
        }
    }
    // Analog triggers press ZL/ZR past the same threshold as on Linux.
    for (from, to) in [(Button::LeftTrigger2, ZL), (Button::RightTrigger2, ZR)] {
        let value = pad.button_data(from).map_or(0.0, |data| data.value());
        if pad.is_pressed(from) || value > PRESS {
            buttons |= to;
        }
    }
    // gilrs reports y up as positive, as the Switch does.
    let left = stick((pad.value(Axis::LeftStickX), pad.value(Axis::LeftStickY)));
    let right = stick((pad.value(Axis::RightStickX), pad.value(Axis::RightStickY)));
    buttons |= directions(left, STICK_L_LEFT) | directions(right, STICK_R_LEFT);
    Pad {
        buttons,
        left,
        right,
    }
}
