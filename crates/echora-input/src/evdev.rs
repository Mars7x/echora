//! Linux backend: evdev gamepads.
//!
//! Every gamepad node under /dev/input (an evdev device reporting
//! BTN_SOUTH) is read on its own thread; new ones are picked up while the
//! game runs, and unplugged ones drop out. The buttons follow the kernel's
//! gamepad conventions (Documentation/input/gamepad.rst), mapped by
//! position the way the Switch lays them out: the bottom face button is B,
//! the right one A, the top one X and the left one Y.
//! ECHORA_GAMEPAD_LAYOUT=labels maps by label instead (an Xbox pad's A is
//! the Switch's A). Several pads at once act as one controller.
//!
//! Pads with rumble motors (FF_RUMBLE) play what [`rumble`] asks for: the
//! low band on the strong motor and the high band on the weak one.
//!
//! Only reading the device, asking for its axis ranges and uploading the
//! rumble effect needs the kernel; capabilities, names and drivers come
//! from sysfs.
//!
//! Adapted from pokopia-recomp `crates/input` (same author, GPL-3.0-or-later).

// The kernel interface (ioctl on device nodes) needs unsafe calls.
#![allow(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::Duration;

use crate::pad::{PRESS, Pad, button, directions, stick};
#[cfg(test)]
use crate::pad::STICK_MAX;

const EV_SYN: u16 = 0;
const EV_KEY: u16 = 1;
const EV_ABS: u16 = 3;
const EV_FF: u16 = 0x15;
const FF_RUMBLE: u16 = 0x50;
const SYN_REPORT: u16 = 0;

const BTN_SOUTH: u16 = 0x130;
const BTN_EAST: u16 = 0x131;
const BTN_NORTH: u16 = 0x133;
const BTN_WEST: u16 = 0x134;
const BTN_TL: u16 = 0x136;
const BTN_TR: u16 = 0x137;
const BTN_TL2: u16 = 0x138;
const BTN_TR2: u16 = 0x139;
const BTN_SELECT: u16 = 0x13a;
const BTN_START: u16 = 0x13b;
const BTN_THUMBL: u16 = 0x13d;
const BTN_THUMBR: u16 = 0x13e;
const BTN_DPAD_UP: u16 = 0x220;
const BTN_DPAD_DOWN: u16 = 0x221;
const BTN_DPAD_LEFT: u16 = 0x222;
const BTN_DPAD_RIGHT: u16 = 0x223;

const ABS_X: u16 = 0x00;
const ABS_Y: u16 = 0x01;
const ABS_Z: u16 = 0x02;
const ABS_RX: u16 = 0x03;
const ABS_RY: u16 = 0x04;
const ABS_RZ: u16 = 0x05;
const ABS_HAT0X: u16 = 0x10;
const ABS_HAT0Y: u16 = 0x11;
const ABS_COUNT: usize = 0x40;

/// How a device's axes are used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AxisRole {
    LeftX,
    LeftY,
    RightX,
    RightY,
    TriggerLeft,
    TriggerRight,
    HatX,
    HatY,
}

/// An axis's range, from EVIOCGABS.
#[derive(Clone, Copy, Debug, Default)]
struct Range {
    minimum: i32,
    maximum: i32,
}

impl Range {
    /// `value` within -1..=1 (the centre of the range at 0).
    fn centred(self, value: i32) -> f32 {
        let span = self.maximum as f32 - self.minimum as f32;
        if span <= 0.0 {
            return 0.0;
        }
        ((value as f32 - self.minimum as f32) / span * 2.0 - 1.0).clamp(-1.0, 1.0)
    }

    /// `value` within 0..=1 (the minimum at 0), for triggers.
    fn pressed_fraction(self, value: i32) -> f32 {
        (self.centred(value) + 1.0) / 2.0
    }
}

/// One gamepad's layout and its state as events arrive.
#[derive(Clone, Debug)]
struct Device {
    /// Axis code -> role and range.
    axes: BTreeMap<u16, (AxisRole, Range)>,
    /// Key code -> NpadButton bit.
    keys: BTreeMap<u16, u64>,
    held: u64,
    values: BTreeMap<u16, i32>,
}

impl Device {
    /// The layout of a device with `keys` and `axes` (capability bits,
    /// with the ranges `range` reports for each axis). `xpad` swaps the
    /// north and west codes: that driver reports an Xbox pad's X (on the
    /// left) as BTN_X, which is BTN_NORTH.
    fn new(
        keys: &BTreeSet<u16>,
        axes: &BTreeSet<u16>,
        range: impl Fn(u16) -> Range,
        xpad: bool,
        by_label: bool,
    ) -> Device {
        use button::*;
        let (north, west) = if xpad {
            (BTN_WEST, BTN_NORTH)
        } else {
            (BTN_NORTH, BTN_WEST)
        };
        let (south_button, east_button, north_button, west_button) =
            if by_label { (A, B, Y, X) } else { (B, A, X, Y) };
        let mut map = BTreeMap::new();
        for (code, bit) in [
            (BTN_SOUTH, south_button),
            (BTN_EAST, east_button),
            (north, north_button),
            (west, west_button),
            (BTN_TL, L),
            (BTN_TR, R),
            (BTN_TL2, ZL),
            (BTN_TR2, ZR),
            (BTN_SELECT, MINUS),
            (BTN_START, PLUS),
            (BTN_THUMBL, STICK_L),
            (BTN_THUMBR, STICK_R),
            (BTN_DPAD_UP, UP),
            (BTN_DPAD_DOWN, DOWN),
            (BTN_DPAD_LEFT, LEFT),
            (BTN_DPAD_RIGHT, RIGHT),
        ] {
            if keys.contains(&code) {
                map.insert(code, bit);
            }
        }
        // ABS_RX/RY are the right stick when present, and ABS_Z/RZ then
        // the analog triggers (Xbox, PlayStation). Without them, a generic
        // HID pad reports its right stick as ABS_Z/RZ.
        let mut roles = vec![
            (ABS_X, AxisRole::LeftX),
            (ABS_Y, AxisRole::LeftY),
            (ABS_HAT0X, AxisRole::HatX),
            (ABS_HAT0Y, AxisRole::HatY),
        ];
        if axes.contains(&ABS_RX) {
            roles.extend([(ABS_RX, AxisRole::RightX), (ABS_RY, AxisRole::RightY)]);
            if !keys.contains(&BTN_TL2) {
                roles.push((ABS_Z, AxisRole::TriggerLeft));
            }
            if !keys.contains(&BTN_TR2) {
                roles.push((ABS_RZ, AxisRole::TriggerRight));
            }
        } else {
            roles.extend([(ABS_Z, AxisRole::RightX), (ABS_RZ, AxisRole::RightY)]);
        }
        let axes = roles
            .into_iter()
            .filter(|(code, _)| axes.contains(code))
            .map(|(code, role)| (code, (role, range(code))))
            .collect();
        Device {
            axes,
            keys: map,
            held: 0,
            values: BTreeMap::new(),
        }
    }

    fn event(&mut self, kind: u16, code: u16, value: i32) {
        match kind {
            EV_KEY => {
                if let Some(&bit) = self.keys.get(&code) {
                    if value != 0 {
                        self.held |= bit;
                    } else {
                        self.held &= !bit;
                    }
                }
            }
            EV_ABS => {
                self.values.insert(code, value);
            }
            _ => {}
        }
    }

    /// The state the events so far describe.
    fn pad(&self) -> Pad {
        use button::*;
        let mut buttons = self.held;
        let mut left = (0.0, 0.0);
        let mut right = (0.0, 0.0);
        for (code, &(role, range)) in &self.axes {
            let Some(&value) = self.values.get(code) else {
                continue;
            };
            let centred = range.centred(value);
            match role {
                // Linux reports down as positive; the Switch, up.
                AxisRole::LeftX => left.0 = centred,
                AxisRole::LeftY => left.1 = -centred,
                AxisRole::RightX => right.0 = centred,
                AxisRole::RightY => right.1 = -centred,
                AxisRole::TriggerLeft | AxisRole::TriggerRight => {
                    if range.pressed_fraction(value) > PRESS {
                        buttons |= if role == AxisRole::TriggerLeft {
                            ZL
                        } else {
                            ZR
                        };
                    }
                }
                AxisRole::HatX => {
                    buttons |= match value.signum() {
                        -1 => LEFT,
                        1 => RIGHT,
                        _ => 0,
                    }
                }
                AxisRole::HatY => {
                    buttons |= match value.signum() {
                        -1 => UP,
                        1 => DOWN,
                        _ => 0,
                    }
                }
            }
        }
        let left = stick(left);
        let right = stick(right);
        buttons |= directions(left, STICK_L_LEFT) | directions(right, STICK_R_LEFT);
        Pad {
            buttons,
            left,
            right,
        }
    }
}

/// A pad's rumble effect: the device opened for writing and the effect's
/// id once uploaded.
struct Rumbler {
    file: File,
    id: i16,
    playing: (u16, u16),
}

impl Rumbler {
    /// Play `strong` and `weak` motor strengths until changed (both zero
    /// stops).
    fn set(&mut self, strong: u16, weak: u16) -> bool {
        if (strong >> 8, weak >> 8) == (self.playing.0 >> 8, self.playing.1 >> 8) {
            return true;
        }
        if strong == 0 && weak == 0 {
            self.playing = (0, 0);
            return self.id < 0 || self.write_event(EV_FF, self.id as u16, 0);
        }
        let Some(id) = upload_rumble(self.file.as_raw_fd(), self.id, strong, weak) else {
            return false;
        };
        self.id = id;
        self.playing = (strong, weak);
        self.write_event(EV_FF, id as u16, 1)
    }

    fn write_event(&mut self, kind: u16, code: u16, value: i32) -> bool {
        use std::io::Write;
        let mut event = vec![0u8; std::mem::size_of::<libc::timeval>()];
        event.extend_from_slice(&kind.to_ne_bytes());
        event.extend_from_slice(&code.to_ne_bytes());
        event.extend_from_slice(&value.to_ne_bytes());
        self.file.write_all(&event).is_ok()
    }
}

/// The connected gamepads.
#[allow(missing_debug_implementations)]
pub struct Gamepads {
    pads: Mutex<BTreeMap<PathBuf, Pad>>,
    open: Mutex<BTreeSet<PathBuf>>,
    rumblers: Mutex<BTreeMap<PathBuf, Rumbler>>,
    /// Requested low- and high-band amplitude (0..=1) per vibration
    /// device (a controller has one per side).
    motors: Mutex<BTreeMap<u32, (f32, f32)>>,
}

impl Gamepads {
    /// All connected pads as one controller (nothing held without any).
    pub fn state(&self) -> Pad {
        self.pads
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .values()
            .fold(Pad::default(), |merged, pad| merged.merge(*pad))
    }

    /// Whether any gamepad is connected.
    pub fn connected(&self) -> bool {
        !self
            .pads
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_empty()
    }
}

static GAMEPADS: OnceLock<&'static Gamepads> = OnceLock::new();

/// Vibrate the pads: vibration device `device` (one per side of a
/// controller) asks for `low` and `high` band amplitudes (0..=1). The pads
/// play the strongest request of all devices. Nothing happens before
/// [`gamepads`] has started watching.
pub fn rumble(device: u32, low: f32, high: f32) {
    let Some(gamepads) = GAMEPADS.get() else {
        return;
    };
    let (strong, weak) = {
        let mut motors = gamepads
            .motors
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        motors.insert(device, (low.clamp(0.0, 1.0), high.clamp(0.0, 1.0)));
        motors
            .values()
            .fold((0.0f32, 0.0f32), |(a, b), &(low, high)| {
                (a.max(low), b.max(high))
            })
    };
    let strong = (strong * f32::from(u16::MAX)) as u16;
    let weak = (weak * f32::from(u16::MAX)) as u16;
    gamepads
        .rumblers
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .retain(|_, rumbler| rumbler.set(strong, weak));
}

/// The gamepads, watched from the first call on.
pub fn gamepads() -> &'static Gamepads {
    GAMEPADS.get_or_init(|| {
        let gamepads: &'static Gamepads = Box::leak(Box::new(Gamepads {
            pads: Mutex::new(BTreeMap::new()),
            open: Mutex::new(BTreeSet::new()),
            rumblers: Mutex::new(BTreeMap::new()),
            motors: Mutex::new(BTreeMap::new()),
        }));
        std::thread::Builder::new()
            .name("gamepads".into())
            .spawn(move || scan(gamepads))
            .expect("spawn the gamepad thread");
        gamepads
    })
}

/// Look for new gamepads every second.
fn scan(gamepads: &'static Gamepads) {
    let by_label = std::env::var("ECHORA_GAMEPAD_LAYOUT").is_ok_and(|layout| layout == "labels");
    let mut refused = BTreeSet::new();
    loop {
        let mut nodes: Vec<PathBuf> = std::fs::read_dir("/dev/input")
            .into_iter()
            .flatten()
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("event"))
            })
            .collect();
        nodes.sort();
        for node in nodes {
            if gamepads
                .open
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .contains(&node)
            {
                continue;
            }
            let sys = Path::new("/sys/class/input")
                .join(node.file_name().expect("device nodes have a file name"));
            let keys = capabilities(&sys.join("device/capabilities/key"));
            if !keys.contains(&BTN_SOUTH) {
                continue;
            }
            let file = match File::open(&node) {
                Ok(file) => file,
                Err(error) => {
                    if refused.insert(node.clone()) {
                        eprintln!("gamepad {}: {error}", node.display());
                    }
                    continue;
                }
            };
            refused.remove(&node);
            let axes = capabilities(&sys.join("device/capabilities/abs"));
            let name = std::fs::read_to_string(sys.join("device/name"))
                .map(|name| name.trim().to_string())
                .unwrap_or_default();
            let xpad = std::fs::read_link(sys.join("device/device/driver"))
                .ok()
                .and_then(|driver| driver.file_name().map(|name| name == "xpad"))
                .unwrap_or(false);
            let fd = file.as_raw_fd();
            let mut device = Device::new(
                &keys,
                &axes,
                |code| absolute_range(fd, code),
                xpad,
                by_label,
            );
            for code in device.axes.keys().copied().collect::<Vec<_>>() {
                if let Some(value) = absolute_value(fd, code) {
                    device.values.insert(code, value);
                }
            }
            let rumbles = capabilities(&sys.join("device/capabilities/ff")).contains(&FF_RUMBLE);
            let rumbler = rumbles
                .then(|| {
                    std::fs::OpenOptions::new()
                        .read(true)
                        .write(true)
                        .open(&node)
                        .ok()
                })
                .flatten();
            eprintln!(
                "gamepad: {name} ({}){}",
                node.display(),
                match (rumbles, rumbler.is_some()) {
                    (true, true) => ", rumble",
                    (true, false) => ", rumble not writable",
                    _ => "",
                }
            );
            if let Some(file) = rumbler {
                gamepads
                    .rumblers
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .insert(
                        node.clone(),
                        Rumbler {
                            file,
                            id: -1,
                            playing: (0, 0),
                        },
                    );
            }
            gamepads
                .open
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .insert(node.clone());
            gamepads
                .pads
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .insert(node.clone(), device.pad());
            let path = node.clone();
            let spawned = std::thread::Builder::new()
                .name("gamepad".into())
                .spawn(move || read(gamepads, path, file, device));
            if spawned.is_err() {
                gamepads
                    .open
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .remove(&node);
                gamepads
                    .pads
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .remove(&node);
            }
        }
        std::thread::sleep(Duration::from_secs(1));
    }
}

/// Read one device's events until it goes away.
fn read(gamepads: &'static Gamepads, path: PathBuf, mut file: File, mut device: Device) {
    // struct input_event: a timeval (two longs), type, code and value.
    const SIZE: usize = std::mem::size_of::<libc::timeval>() + 8;
    let mut event = [0u8; SIZE];
    while file.read_exact(&mut event).is_ok() {
        let at = SIZE - 8;
        let kind = u16::from_ne_bytes([event[at], event[at + 1]]);
        let code = u16::from_ne_bytes([event[at + 2], event[at + 3]]);
        let value = i32::from_ne_bytes(
            event[at + 4..]
                .try_into()
                .expect("event values are 4 bytes"),
        );
        if kind == EV_SYN && code == SYN_REPORT {
            gamepads
                .pads
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .insert(path.clone(), device.pad());
        } else {
            device.event(kind, code, value);
        }
    }
    eprintln!("gamepad: {} disconnected", path.display());
    gamepads
        .pads
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .remove(&path);
    gamepads
        .rumblers
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .remove(&path);
    gamepads
        .open
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .remove(&path);
}

/// A sysfs capability bitmap (hex words, most significant first) as the
/// set codes.
fn capabilities(path: &Path) -> BTreeSet<u16> {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    parse_capabilities(&text)
}

fn parse_capabilities(text: &str) -> BTreeSet<u16> {
    let bits = usize::BITS as usize;
    let mut set = BTreeSet::new();
    for (index, word) in text.split_whitespace().rev().enumerate() {
        let Ok(word) = usize::from_str_radix(word, 16) else {
            continue;
        };
        for bit in 0..bits {
            if word >> bit & 1 != 0
                && let Ok(code) = u16::try_from(index * bits + bit)
            {
                set.insert(code);
            }
        }
    }
    set
}

/// struct input_absinfo: value, minimum, maximum, fuzz, flat, resolution.
fn absolute_info(fd: i32, code: u16) -> Option<[i32; 6]> {
    if usize::from(code) >= ABS_COUNT {
        return None;
    }
    let mut info = [0i32; 6];
    // EVIOCGABS(code) = _IOR('E', 0x40 + code, struct input_absinfo).
    let request = (2u64 << 30) | (24 << 16) | (u64::from(b'E') << 8) | (0x40 + u64::from(code));
    // SAFETY: `fd` is an open evdev node and the request writes exactly
    // one 24-byte input_absinfo into `info`.
    let result = unsafe { libc::ioctl(fd, request as _, info.as_mut_ptr()) };
    (result >= 0).then_some(info)
}

/// Upload (or with `id` from an earlier upload, update) a rumble effect
/// that plays until stopped; the effect's id.
fn upload_rumble(fd: i32, id: i16, strong: u16, weak: u16) -> Option<i16> {
    // struct ff_effect (48 bytes on 64-bit): type, id, direction, trigger
    // (button, interval), replay (length 0 = until stopped, delay), then
    // the union at 16, whose rumble member is strong and weak magnitudes.
    let mut effect = [0u8; 48];
    effect[0..2].copy_from_slice(&FF_RUMBLE.to_ne_bytes());
    effect[2..4].copy_from_slice(&id.to_ne_bytes());
    effect[16..18].copy_from_slice(&strong.to_ne_bytes());
    effect[18..20].copy_from_slice(&weak.to_ne_bytes());
    // EVIOCSFF = _IOW('E', 0x80, struct ff_effect).
    let request = (1u64 << 30) | (48 << 16) | (u64::from(b'E') << 8) | 0x80;
    // SAFETY: `fd` is an open evdev node and the kernel reads one
    // ff_effect from `effect` and writes back its id.
    let result = unsafe { libc::ioctl(fd, request as _, effect.as_mut_ptr()) };
    (result >= 0).then(|| i16::from_ne_bytes([effect[2], effect[3]]))
}

fn absolute_range(fd: i32, code: u16) -> Range {
    let info = absolute_info(fd, code).unwrap_or([0, -1, 1, 0, 0, 0]);
    Range {
        minimum: info[1],
        maximum: info[2],
    }
}

fn absolute_value(fd: i32, code: u16) -> Option<i32> {
    absolute_info(fd, code).map(|info| info[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xbox() -> Device {
        let keys = [
            BTN_SOUTH, BTN_EAST, BTN_NORTH, BTN_WEST, BTN_TL, BTN_TR, BTN_SELECT, BTN_START,
            BTN_THUMBL, BTN_THUMBR,
        ]
        .into_iter()
        .collect();
        let axes = [
            ABS_X, ABS_Y, ABS_Z, ABS_RX, ABS_RY, ABS_RZ, ABS_HAT0X, ABS_HAT0Y,
        ]
        .into_iter()
        .collect();
        Device::new(
            &keys,
            &axes,
            |code| match code {
                ABS_Z | ABS_RZ => Range {
                    minimum: 0,
                    maximum: 1023,
                },
                ABS_HAT0X | ABS_HAT0Y => Range {
                    minimum: -1,
                    maximum: 1,
                },
                _ => Range {
                    minimum: -32768,
                    maximum: 32767,
                },
            },
            true,
            false,
        )
    }

    #[test]
    fn capability_words_count_from_the_last() {
        let set = parse_capabilities("1 0 3\n");
        let bits = usize::BITS as u16;
        assert_eq!(set.into_iter().collect::<Vec<_>>(), vec![0, 1, 2 * bits]);
    }

    #[test]
    fn buttons_map_by_position_and_xpad_north_west_swap() {
        let mut pad = xbox();
        pad.event(EV_KEY, BTN_SOUTH, 1);
        assert_eq!(pad.pad().buttons, button::B);
        pad.event(EV_KEY, BTN_SOUTH, 0);
        // xpad's BTN_X (= BTN_NORTH) is the left face button: Switch Y.
        pad.event(EV_KEY, BTN_NORTH, 1);
        assert_eq!(pad.pad().buttons, button::Y);
        pad.event(EV_KEY, BTN_NORTH, 0);
        pad.event(EV_KEY, BTN_START, 1);
        assert_eq!(pad.pad().buttons, button::PLUS);
    }

    #[test]
    fn triggers_hat_and_sticks() {
        let mut pad = xbox();
        pad.event(EV_ABS, ABS_Z, 1000);
        pad.event(EV_ABS, ABS_HAT0Y, -1);
        assert_eq!(pad.pad().buttons, button::ZL | button::UP);
        pad.event(EV_ABS, ABS_Z, 0);
        pad.event(EV_ABS, ABS_HAT0Y, 0);
        // Full left-down and full up on the right stick.
        pad.event(EV_ABS, ABS_X, -32768);
        pad.event(EV_ABS, ABS_RY, -32768);
        let state = pad.pad();
        assert_eq!(state.left, (-STICK_MAX, 0));
        assert_eq!(state.right, (0, STICK_MAX));
        assert_eq!(
            state.buttons,
            button::STICK_L_LEFT | button::STICK_R_LEFT << 1
        );
        // Small deflections are inside the dead zone.
        pad.event(EV_ABS, ABS_X, 2000);
        pad.event(EV_ABS, ABS_RY, 0);
        assert_eq!(pad.pad(), Pad::default());
    }

    #[test]
    fn diagonal_stays_within_the_circle() {
        let (x, y) = stick((1.0, 1.0));
        assert!(i64::from(x).pow(2) + i64::from(y).pow(2) <= i64::from(STICK_MAX).pow(2) + 2);
        assert!(x > 0 && y > 0);
    }
}
