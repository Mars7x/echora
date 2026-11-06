//! Host NVN entry points: the guest-facing half of high-level NVN.
//!
//! The application looks up every NVN entry point by name: it imports
//! `nvnBootstrapLoader` from the sdk module, asks it for
//! `nvnDeviceGetProcAddress`, and stores each returned function pointer. The
//! runtime replaces that export with a host function, so every pointer the
//! application receives is a host thunk and the console's driver never runs.
//!
//! This module keeps the entry-point table (hooks, thunks, call counts).
//! The NVN implementation itself (objects, command buffers, the graphics
//! backend) has not been ported yet, so `call` implements no entry point and
//! the first call stops with the function's name. A lookup alone has no
//! observable effect.
//!
//! Adapted from pokopia-recomp `src/nvn.rs`.

use echora_guest::Trap;
use echora_guest::cpu::Cpu;
use echora_guest::memory::Memory;

/// What an NVN call sees of the guest: the calling thread's registers, the
/// full memory with its checked host-side accesses, and the system counter
/// at the call.
pub struct Guest<'a> {
    pub cpu: &'a mut Cpu,
    pub memory: GuestMemory<'a>,
    pub counter: u64,
}

/// Guest memory as an NVN call reaches it: the machine's checked memory
/// (under the machine lock), or the calling worker's flat view while a
/// queue submission runs without the lock (the view other guest threads
/// already use concurrently; it faults only outside the address space).
pub enum GuestMemory<'a> {
    Checked(&'a mut Memory),
    View(&'a mut echora_guest::view::GuestView),
}

impl GuestMemory<'_> {
    pub fn read(&self, address: u64, out: &mut [u8]) -> Result<(), Trap> {
        match self {
            Self::Checked(memory) => memory.read(address, out),
            Self::View(view) => view.read(address, out),
        }
    }

    pub fn write(&mut self, address: u64, bytes: &[u8]) -> Result<(), Trap> {
        match self {
            Self::Checked(memory) => memory.write(address, bytes),
            Self::View(view) => view.write(address, bytes),
        }
    }

    pub fn read_u32(&self, address: u64) -> Result<u32, Trap> {
        let mut bytes = [0; 4];
        self.read(address, &mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }

    pub fn read_u64(&self, address: u64) -> Result<u64, Trap> {
        let mut bytes = [0; 8];
        self.read(address, &mut bytes)?;
        Ok(u64::from_le_bytes(bytes))
    }

    pub fn write_u32(&mut self, address: u64, value: u32) -> Result<(), Trap> {
        self.write(address, &value.to_le_bytes())
    }

    pub fn write_u64(&mut self, address: u64, value: u64) -> Result<(), Trap> {
        self.write(address, &value.to_le_bytes())
    }

    /// Whether the bytes at `address` equal `expected`, compared in place.
    pub fn equals(&self, address: u64, expected: &[u8]) -> Result<bool, Trap> {
        match self {
            Self::Checked(memory) => memory.equals(address, expected),
            Self::View(view) => {
                let mut chunk = [0u8; 4096];
                for (index, part) in expected.chunks(chunk.len()).enumerate() {
                    let out = &mut chunk[..part.len()];
                    view.read(address + (index * 4096) as u64, out)?;
                    if out != part {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
        }
    }
}
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};

/// Exports replaced by host functions, resolved by symbol name in whichever
/// loaded module defines them. Shared by the lifter (which leaves these
/// addresses untranslated) and the runtime (which handles them).
pub const HOOKED_EXPORTS: &[&str] = &["nvnBootstrapLoader"];

/// Thunk address range, left unmapped. Nothing is ever executed there: the
/// translated code reaches a thunk only by branching to it, and the host loop
/// handles that branch before any instruction would run.
pub const THUNK_BASE: u64 = 0x7e_0000_0000;
pub const THUNK_SIZE: u64 = 0x10000;
const THUNK_STRIDE: u64 = 16;

/// Longest entry-point name accepted from guest memory.
pub const MAX_NAME: usize = 256;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum HostEntry {
    /// The replaced sdk export `nvnBootstrapLoader(const char* name)`.
    BootstrapLoader,
    /// A thunk handed to the application for one NVN entry point.
    Proc(String),
    /// A guest function replaced by an equivalent host implementation
    /// (`crate::hle`), installed only for its exact code bytes.
    ZlibUncompress,
    /// A `bl inflate` call site with flush Z_FINISH (`crate::hle`): the
    /// call is performed on the host and execution continues after it.
    InflateFinishCall,
}

/// The ECHORA_* environment, read once.
fn debug_env() -> &'static HashMap<String, std::ffi::OsString> {
    static ENV: std::sync::OnceLock<HashMap<String, std::ffi::OsString>> =
        std::sync::OnceLock::new();
    ENV.get_or_init(|| {
        std::env::vars_os()
            .filter_map(|(name, value)| Some((name.into_string().ok()?, value)))
            .filter(|(name, _)| name.starts_with("ECHORA_"))
            .collect()
    })
}

/// `std::env::var_os` for an ECHORA_* switch, from the snapshot.
pub(crate) fn debug_var_os(name: &str) -> Option<&'static std::ffi::OsStr> {
    debug_env().get(name).map(|value| value.as_os_str())
}

/// `std::env::var` for an ECHORA_* switch, from the snapshot.
pub(crate) fn debug_var(name: &str) -> Result<String, std::env::VarError> {
    let value = debug_var_os(name).ok_or(std::env::VarError::NotPresent)?;
    value
        .to_str()
        .map(str::to_owned)
        .ok_or_else(|| std::env::VarError::NotUnicode(value.to_owned()))
}

#[derive(Debug, Default, Serialize)]
pub struct Nvn {
    /// Guest addresses of replaced exports.
    hooks: BTreeMap<u64, HostEntry>,
    /// Thunk index -> entry-point name.
    thunks: Vec<String>,
    #[serde(skip)]
    by_name: BTreeMap<String, u64>,
    /// Calls per entry point that reached the host.
    pub calls: BTreeMap<String, u64>,
    /// Host microseconds spent in each entry point.
    pub call_microseconds: BTreeMap<String, u64>,
    pub presented_frames: u64,
}

impl Nvn {
    pub fn add_hook(&mut self, address: u64, entry: HostEntry) {
        self.hooks.insert(address, entry);
    }

    pub fn hook_addresses(&self) -> impl Iterator<Item = u64> + '_ {
        self.hooks.keys().copied()
    }

    /// Addresses where translated code must resume after a hooked call site
    /// (the instruction after the replaced `bl`).
    pub fn hook_continuations(&self) -> impl Iterator<Item = u64> + '_ {
        self.hooks
            .iter()
            .filter(|(_, entry)| **entry == HostEntry::InflateFinishCall)
            .map(|(address, _)| address + 4)
    }

    /// The host entry a branch to `pc` reaches, if any.
    pub fn entry(&self, pc: u64) -> Option<HostEntry> {
        if let Some(entry) = self.hooks.get(&pc) {
            return Some(entry.clone());
        }
        let offset = pc.checked_sub(THUNK_BASE)?;
        if offset >= THUNK_SIZE || !offset.is_multiple_of(THUNK_STRIDE) {
            return None;
        }
        self.thunks
            .get((offset / THUNK_STRIDE) as usize)
            .map(|name| HostEntry::Proc(name.clone()))
    }

    /// Whether `pc` is a host entry (no allocation, unlike `entry`).
    pub fn has_entry(&self, pc: u64) -> bool {
        self.hooks.contains_key(&pc) || self.thunk_name(pc).is_some()
    }

    /// Whether `pc` is the thunk of entry point `name`.
    pub fn entry_is(&self, pc: u64, name: &str) -> bool {
        self.thunk_name(pc) == Some(name)
    }

    fn thunk_name(&self, pc: u64) -> Option<&str> {
        let offset = pc.checked_sub(THUNK_BASE)?;
        if offset >= THUNK_SIZE || !offset.is_multiple_of(THUNK_STRIDE) {
            return None;
        }
        self.thunks
            .get((offset / THUNK_STRIDE) as usize)
            .map(String::as_str)
    }

    /// A stand-in holding only the entry-point table, while a submission
    /// running without the machine lock has the real state.
    pub fn entries_only(&self) -> Self {
        Self {
            hooks: self.hooks.clone(),
            thunks: self.thunks.clone(),
            by_name: self.by_name.clone(),
            ..Self::default()
        }
    }

    /// The thunk address for an NVN entry point, allocated on first lookup.
    pub fn thunk(&mut self, name: &str) -> Result<u64, Trap> {
        if let Some(&address) = self.by_name.get(name) {
            return Ok(address);
        }
        let address = THUNK_BASE + self.thunks.len() as u64 * THUNK_STRIDE;
        if address + THUNK_STRIDE > THUNK_BASE + THUNK_SIZE {
            return Err(Trap::InvalidState {
                message: "NVN thunk range exhausted".into(),
            });
        }
        self.thunks.push(name.to_owned());
        self.by_name.insert(name.to_owned(), address);
        Ok(address)
    }

    /// Run an implemented entry point. Arguments follow the AArch64 calling
    /// convention (X0..X7, V0..V7); a result goes to X0 (or V0). Returns
    /// Ok(false) when `name` is not implemented, so the caller stops with it.

    /// Run an implemented entry point. Returns Ok(false) when `name` is not
    /// implemented, so the caller stops with it. No entry point is
    /// implemented until the NVN layer is ported.
    pub fn call(&mut self, _name: &str, _core: &mut Guest) -> Result<bool, Trap> {
        Ok(false)
    }

    pub fn report(&self) -> serde_json::Value {
        serde_json::json!({
            "model": "host_nvn_entry_points",
            "implementation": "entry-point table only; no NVN function implemented yet",
            "hooked_exports": self.hooks.values().collect::<Vec<_>>(),
            "entry_points_looked_up": self.thunks.len(),
            "calls": self.calls,
            "presented_frames": self.presented_frames,
        })
    }
}

/// Whether `name` is a plausible NVN entry-point name (ASCII identifier with
/// the `nvn` prefix). Lookups of anything else stop explicitly.
pub fn is_entry_point_name(name: &str) -> bool {
    name.starts_with("nvn")
        && name.len() > 3
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}
