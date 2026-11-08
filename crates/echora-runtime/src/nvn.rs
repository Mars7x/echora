//! High-level NVN: the game's graphics API calls run as host code.
//!
//! The application looks up every NVN entry point by name: it imports
//! `nvnBootstrapLoader` from the sdk module, asks it for
//! `nvnDeviceGetProcAddress`, and stores each returned function pointer in a
//! global (`nvnLoadCProcs`). The runtime replaces the sdk export with a host
//! function, so every pointer the application receives is a host thunk and the
//! NVIDIA driver in the sdk module never runs.
//!
//! NVN is not publicly documented. Entry-point names come from the binary;
//! argument layouts are inferred from the application's own calls and are
//! labeled where implemented. A thunk whose function is not implemented stops
//! with the function's name when called, never earlier: a lookup alone has no
//! observable effect.

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
    /// Host address of `length` guest bytes, when guest memory is one
    /// contiguous host mapping (for sharing it with the GPU in place).
    pub fn host_span(&self, address: u64, length: usize) -> Option<*mut u8> {
        match self {
            Self::Checked(memory) => memory.view().host_span(address, length),
            Self::View(view) => view.host_span(address, length),
        }
    }

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
    /// The game's zstd wrapper, likewise (`crate::hle`).
    ZstdDecompress,
    /// A `bl inflate` call site with flush Z_FINISH (`crate::hle`): the
    /// call is performed on the host and execution continues after it.
    InflateFinishCall,
    /// The zstd frame decoder (`crate::hle`), likewise.
    ZstdFrames,
    /// The streaming zstd reader's call site (`crate::hle`): performed on
    /// the host when it can be, else by the guest decoder it calls.
    ZstdStreamCall,
}

#[path = "nvn_gpu.rs"]
mod gpu;

/// Queued work of one queue (see `Nvn::pending`).
#[derive(Debug)]
enum QueueItem {
    /// Commands of a list from `next` on.
    List {
        commands: std::sync::Arc<Vec<Command>>,
        next: usize,
    },
    /// A FenceSync: the sync signals once earlier work has run.
    Fence { sync: u64, args: Args },
    /// A WaitSync: later work waits for the sync.
    WaitSync { sync: u64 },
}

/// The ECHORA_* environment, read once: the draw path consulted debug
/// switches several times per draw, and each lookup locks and scans the
/// environment.
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

/// `std::env::var_os` for a ECHORA_* switch, from the snapshot.
pub(crate) fn debug_var_os(name: &str) -> Option<&'static std::ffi::OsStr> {
    // The draw path asks per texture binding; names are literals, so the
    // answer is cached by the name's address (an integer lookup instead
    // of hashing the string into the environment snapshot).
    thread_local! {
        static CACHE: std::cell::RefCell<Vec<(usize, usize, Option<&'static std::ffi::OsStr>)>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }
    let key = (name.as_ptr() as usize, name.len());
    CACHE.with(|cache| {
        if let Some(&(_, _, value)) = cache
            .borrow()
            .iter()
            .find(|(address, length, _)| (*address, *length) == key)
        {
            return value;
        }
        let value = debug_env().get(name).map(|value| value.as_os_str());
        cache.borrow_mut().push((key.0, key.1, value));
        value
    })
}

/// `std::env::var` for a ECHORA_* switch, from the snapshot.
pub(crate) fn debug_var(name: &str) -> Result<String, std::env::VarError> {
    let value = debug_var_os(name).ok_or(std::env::VarError::NotPresent)?;
    value
        .to_str()
        .map(str::to_owned)
        .ok_or_else(|| std::env::VarError::NotUnicode(value.to_owned()))
}

/// A small, fast hasher for host-side maps looked up many times per draw
/// (multiply-rotate over 8-byte words, as rustc's FxHasher; not
/// collision-resistant, which these keys need not be).
#[derive(Clone, Copy, Default)]
pub struct FastHasher(u64);

impl std::hash::Hasher for FastHasher {
    fn write(&mut self, bytes: &[u8]) {
        let (words, tail) = bytes.as_chunks::<8>();
        for word in words {
            self.add(u64::from_le_bytes(*word));
        }
        if !tail.is_empty() {
            let mut last = [0u8; 8];
            last[..tail.len()].copy_from_slice(tail);
            self.add(u64::from_le_bytes(last));
        }
    }

    fn write_u64(&mut self, value: u64) {
        self.add(value);
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

impl FastHasher {
    fn add(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
}

/// A HashMap with [`FastHasher`].
pub type FastMap<K, V> = HashMap<K, V, std::hash::BuildHasherDefault<FastHasher>>;

/// A texture handle resolved for a draw: what it samples and how.
type ResolvedBinding = (echora_gpu::draw::TextureSource, echora_gpu::draw::Sampler);

/// A render target's width, height, format, mip levels and shape.
type TargetShape = (
    u32,
    u32,
    echora_gpu::ash::vk::Format,
    u32,
    echora_gpu::draw::ImageShape,
);

/// A stage's code window and its program key.
type StageCode = (std::sync::Arc<Vec<u8>>, u64);

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
    /// ECHORA_NVN_PROFILE: host microseconds spent in each entry point.
    pub call_microseconds: BTreeMap<String, u64>,
    /// Property bags of builders and builder-like structures (anything set up
    /// with `nvn<Kind>SetDefaults`), keyed by guest address. Their layout is
    /// host-private: the application touches them only through NVN calls.
    bags: BTreeMap<u64, Bag>,
    /// Initialized objects by guest address.
    objects: BTreeMap<u64, Object>,
    /// Next free host GPU virtual address for memory pools.
    next_gpu_address: u64,
    /// Stage code windows by GPU address (see `stage_code`).
    #[serde(skip)]
    stage_cache: std::cell::RefCell<HashMap<u64, StageCode>>,
    /// GPU ranges as (pool object, pool GPU address, range GPU address,
    /// size, guest address): a pool's storage, or one mapping of a virtual
    /// pool. Rebuilt when a lookup misses (see `pool_for`).
    #[serde(skip)]
    pool_ranges: std::cell::RefCell<Vec<[u64; 5]>>,
    /// Commands recorded by each command buffer that is recording.
    #[serde(skip)]
    recording: BTreeMap<u64, Vec<Command>>,
    /// Finished command lists by NVNcommandHandle, with the order of their
    /// last recording or submission.
    #[serde(skip)]
    command_lists: BTreeMap<u64, (std::sync::Arc<Vec<Command>>, u64, u32)>,
    /// Command handles by that order (least recently used first).
    #[serde(skip)]
    command_list_use: BTreeMap<u64, u64>,
    command_list_clock: u64,
    /// Commands in `command_lists`.
    command_list_commands: usize,
    /// Submissions of command handles no longer kept (skipped).
    pub stale_command_submissions: u64,
    next_command_handle: u64,
    /// Host images of textures that have been rendered to.
    #[serde(skip)]
    images: BTreeMap<u64, Image>,
    /// Storage of finalized textures, by structure address: a texture
    /// initialized there later with other storage is another texture.
    #[serde(skip)]
    finalized_storage: HashMap<u64, u64>,
    /// Texture views registered with texture pool descriptors, by (pool, id).
    #[serde(skip)]
    views: BTreeMap<(u64, u32), Bag>,
    /// Render state the host executor tracks between submitted lists.
    #[serde(skip)]
    render: RenderState,
    pub presented_frames: u64,
    /// Nesting of command lists running through CallCommands.
    #[serde(skip)]
    call_depth: u32,
    /// Work each queue has been given but not run yet, in order: a queue
    /// stops at a WaitEvent whose condition does not hold (or a WaitSync on
    /// a sync not yet signaled) and resumes when another queue or the CPU
    /// changes that (see `pump_queues`).
    #[serde(skip)]
    pending: BTreeMap<u64, std::collections::VecDeque<QueueItem>>,
    /// Waits that never held when work had to run anyway (a present,
    /// Finish or blocking CPU wait drained the queue), by kind.
    pub forced_waits: BTreeMap<String, u64>,
    /// ECHORA_NVN_TEXTURE_LOG: each texture initialized (format, target,
    /// size, stride and guest storage address), the first 512; a hex
    /// `low-high` value keeps those with storage in that range.
    texture_log: Vec<String>,
    /// Order of flow notes in the sequenced frame (see `sequence_note`).
    pub sequence: u64,
    /// The present ECHORA_GPU_SEQUENCE_AFTER chose to sequence.
    #[serde(skip)]
    sequence_chosen: Option<u64>,
    /// Texture objects initialized, finalized or moved: a storage index of
    /// render targets is current while this holds.
    #[serde(skip)]
    texture_epoch: u64,
    /// Render-target shapes by texture (width, height, format, levels,
    /// shape), valid for the texture epoch they were derived in: draws
    /// look them up for every color and depth target.
    #[serde(skip)]
    target_shapes: (u64, HashMap<u64, TargetShape>),
    /// Each queue submission, compute dispatch, texture or sampler pool
    /// change and CPU invalidation of texture memory: texture bindings
    /// resolved before may differ after.
    #[serde(skip)]
    binding_serial: u64,
    /// Texture bindings resolved during the current stamp (texture epoch,
    /// render-target changes, binding serial, present): a submission's
    /// draws share most handles, and resolving one walks several maps per
    /// draw. Registering a descriptor evicts the handles that name it.
    #[serde(skip)]
    binding_memo: ((u64, u64, u64, u64), FastMap<u64, ResolvedBinding>),
    /// nvnSyncWait calls that timed out (never fenced), by sync address.
    pub sync_timeouts: BTreeMap<u64, u64>,
    /// Hash of the last frame written to ECHORA_FRAME_DIR; identical
    /// frames are not written again.
    #[serde(skip)]
    last_frame_hash: Option<u64>,
    /// Work commands skipped under ECHORA_NVN_SKIP_WORK, by name.
    pub skipped_work: BTreeMap<String, u64>,
    /// Host GPU execution (ECHORA_GPU), created on first use.
    #[serde(skip)]
    gpu: Option<gpu::Backend>,
    #[serde(skip)]
    gpu_checked: bool,
}

/// One recorded command: its entry point, argument registers and any data
/// NVN reads through pointer arguments at record time.
#[derive(Clone, Debug)]
pub struct Command {
    pub name: String,
    pub x: [u64; 8],
    pub v: [u64; 8],
    pub captured: Vec<u64>,
    /// Snapshots of the state structures a Bind*State command points to,
    /// taken at record time: the application may reuse that memory before
    /// the list executes. Shared: draws read them without copying.
    pub states: std::sync::Arc<Vec<BTreeMap<String, Args>>>,
}

/// A host image: RGBA, one f32 per channel.
#[derive(Clone, Debug, Default)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub texels: Vec<[f32; 4]>,
}

#[derive(Clone, Debug, Default)]
pub struct RenderState {
    pub colors: Vec<u64>,
    /// The mip level each color target renders into (low 32 bits) and
    /// the array layer or cube face (high 32 bits), from its view.
    pub color_levels: Vec<u64>,
    pub depth: u64,
    /// The array layer the depth target's view selects.
    pub depth_layer: u32,
    pub scissor: Option<[i32; 4]>,
    /// Root tables written by Push{Graphics,Compute}Constants: [graphics,
    /// compute] x 8 tables of 64 words (256 bytes each).
    pub roots: [Vec<[u32; 64]>; 2],
    /// The latest submitted state command of each kind (viewports, bound
    /// programs, state objects, pools, resources). Kept for the host
    /// renderer; nothing consumes it until draws are implemented.
    pub state: FastMap<String, Command>,
}

/// Recorded commands that only change render state (SetRenderTargets,
/// SetScissor and ClearColor have their own host execution). They execute by
/// updating `RenderState::state`.
fn is_state_command(rest: &str) -> bool {
    const PREFIXES: &[&str] = &["Set", "Bind", "Push", "Pop", "Insert", "Enable"];
    const OTHERS: &[&str] = &["Barrier", "ShadingRateImagePalette", "OverrideSubtileSize"];
    PREFIXES.iter().any(|prefix| rest.starts_with(prefix)) || OTHERS.contains(&rest)
}

/// Commands that produce GPU work. Without a host rasterizer they stop by
/// name, unless ECHORA_NVN_SKIP_WORK is set: then they are counted and
/// skipped, so CPU-side progress can continue past them. A skipped command
/// leaves its targets unchanged; frames rendered that way are incomplete.
fn is_work_command(rest: &str) -> bool {
    const PREFIXES: &[&str] = &[
        "Draw",
        "MultiDraw",
        "Dispatch",
        "Copy",
        "Clear",
        "Downsample",
        "TiledDownsample",
        "Resolve",
        "Discard",
        "Build",
        "Compact",
        "Relocate",
        "EvaluateDlss",
        "ComputeDirty",
        "UpdateUniformBuffer",
        "ReportCounter",
        "ReportValue",
        "ResetCounter",
        "SaveZCull",
        "RestoreZCull",
    ];
    PREFIXES.iter().any(|prefix| rest.starts_with(prefix))
}

/// Command lists kept for resubmission (least recently used dropped first).
const COMMAND_LIST_LIMIT: usize = 8192;
/// Commands those lists may hold in all.
const COMMAND_LIMIT: usize = 500_000;

/// Command-buffer management calls (everything else on a recording command
/// buffer is recorded).
const COMMAND_BUFFER_MANAGEMENT: &[&str] = &[
    "Initialize",
    "Finalize",
    "AddCommandMemory",
    "AddControlMemory",
    "BeginRecording",
    "EndRecording",
    "GetCommandMemoryUsed",
    "GetControlMemoryUsed",
    "GetCommandMemorySize",
    "GetControlMemorySize",
    "GetCommandMemoryFree",
    "GetControlMemoryFree",
    "SetMemoryCallback",
    "SetMemoryCallbackData",
    "GetMemoryCallback",
    "GetMemoryCallbackData",
    "SetCommandMemoryCallbackEnabled",
    "SetControlMemoryCallbackEnabled",
    "IsRecording",
    "SetDebugLabel",
    "SetRecordingMode",
];

/// Command buffer memory the host counts as used (GetCommandMemoryUsed,
/// GetControlMemoryUsed): per recorded command, and per non-empty list for
/// its handle. Host lists do not live in that memory, but the application
/// reads a list with no used memory as empty and never submits it (main
/// 0x7100666970 keeps that flag beside each worker list's handle).
const COMMAND_MEMORY_PER_COMMAND: u64 = 16;
const CONTROL_MEMORY_PER_LIST: u64 = 16;

/// First GPU virtual address handed to a memory pool, and pool alignment.
const GPU_ADDRESS_BASE: u64 = 0x10_0000_0000;
const GPU_ADDRESS_ALIGNMENT: u64 = 0x10000;

/// Raw argument registers after the object pointer (X1..X4, V0..V1 low 64
/// bits). Setters store them; matching getters return the first of each.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Args {
    pub x: [u64; 4],
    pub v: [u64; 2],
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Bag {
    pub kind: String,
    pub props: BTreeMap<String, Args>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Object {
    pub kind: String,
    /// Builder properties at initialization.
    pub builder: BTreeMap<String, Args>,
    /// Properties set on the object afterwards (nvn<Kind>Set<Prop>).
    pub state: BTreeMap<String, Args>,
}

/// Key of the copy kept of a texture whose structure was finalized or
/// reused, by its storage: pool descriptors registered before still
/// describe it (guest addresses never set bit 63).
pub(crate) fn retired_texture(storage: u64) -> u64 {
    1 << 63 | storage
}

/// NVN object kinds, longest first so prefixes resolve uniquely.
const KINDS: &[&str] = &[
    "TextureBuilder",
    "TextureView",
    "TexturePool",
    "TextureHandle",
    "Texture",
    "SamplerBuilder",
    "SamplerPool",
    "Sampler",
    "BufferBuilder",
    "Buffer",
    "MemoryPoolBuilder",
    "MemoryPool",
    "QueueBuilder",
    "Queue",
    "CommandBuffer",
    "DeviceBuilder",
    "Device",
    "WindowBuilder",
    "Window",
    "Sync",
    "ProgramBuilder",
    "Program",
    "BlendState",
    "ChannelMaskState",
    "ColorState",
    "DepthStencilState",
    "MultisampleState",
    "PolygonState",
    "VertexAttribState",
    "VertexStreamState",
    "EventBuilder",
    "Event",
];

/// Split "nvn<Kind><Rest>" into (kind, rest).
fn split_name(name: &str) -> Option<(&'static str, &str)> {
    let body = name.strip_prefix("nvn")?;
    KINDS
        .iter()
        .find(|kind| body.starts_with(**kind))
        .map(|kind| (*kind, &body[kind.len()..]))
}

impl Nvn {
    /// Count `bytes` of a command buffer's command or control memory as
    /// used, up to the memory it was given.
    fn use_command_memory(&mut self, buffer: u64, key: &str, bytes: u64) {
        if let Some(object) = self.objects.get_mut(&buffer) {
            let size = object.state.get(key).map_or(0, |value| value.x[0]);
            let used = object.state.entry(format!("{key}Used")).or_default();
            used.x[0] = (used.x[0] + bytes).min(size);
        }
    }

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
    pub fn call(&mut self, name: &str, core: &mut Guest) -> Result<bool, Trap> {
        // Where the CPU may look at what the GPU wrote (after waiting for
        // it, or invalidating its view of a mapped range), deferred
        // dispatch results are stored first; between those points they
        // stay with the GPU (one wait for many dispatches).
        if matches!(
            name,
            "nvnSyncWait"
                | "nvnQueueFinish"
                | "nvnBufferInvalidateMappedRange"
                | "nvnTextureReadTexels"
                | "nvnQueuePresentTexture"
        ) {
            if let Err(message) = self.resolve_storage(core, None) {
                return Err(Trap::InvalidState {
                    message: format!("{name}: dispatch results: {message}"),
                });
            }
        }
        let x = core.cpu.regs;
        let args = Args {
            x: [x[1], x[2], x[3], x[4]],
            v: [core.cpu.vectors[0] as u64, core.cpu.vectors[1] as u64],
        };
        let invalid = |message: String| Trap::InvalidState {
            message: format!("{name}: {message}"),
        };
        let pc = core.cpu.pc;
        let stop = move |message: String| Trap::UnsupportedService {
            pc,
            service: "nvn".into(),
            command_id: None,
            message: format!("{name}: {message}"),
        };
        let this = x[0];
        if this == 0 {
            return Err(invalid("null object".into()));
        }
        match name {
            // nvnDeviceInitialize(NVNdevice*, const NVNdeviceBuilder*) ->
            // NVNboolean. Builder flags (debug-layer levels) are recorded; no
            // debug layer exists, so they have no further effect.
            "nvnDeviceInitialize" => {
                let builder = self.bag(x[1], "DeviceBuilder").map_err(invalid)?;
                self.initialize(this, "Device", builder).map_err(invalid)?;
                core.cpu.regs[0] = 1;
                return Ok(true);
            }
            // nvnDeviceGetInteger(const NVNdevice*, NVNdeviceInfo, int* out).
            // Device properties are added one by one as the application asks;
            // an unknown property stops with its number.
            "nvnDeviceGetInteger" => {
                self.object(this, "Device").map_err(invalid)?;
                let property = x[1] as u32;
                let Some(value) = device_integer(property) else {
                    return Err(Trap::UnsupportedService {
                        pc: core.cpu.pc,
                        service: "nvn".into(),
                        command_id: Some(property),
                        message: format!(
                            "{name} property {property} ({property:#x}) is not modeled"
                        ),
                    });
                };
                if x[2] == 0 {
                    return Err(invalid("null output".into()));
                }
                core.memory.write_u32(x[2], value as u32)?;
                return Ok(true);
            }
            // nvnDeviceRegisterFastClearColor(device, const float* rgba,
            // NVNformat): registers a clear color for compressed fast
            // clears; the application ignores the result (main+0xb44c4c).
            // The host renderer has no fast-clear path, so nothing changes.
            "nvnDeviceRegisterFastClearColor" => {
                self.object(this, "Device").map_err(invalid)?;
                core.cpu.regs[0] = 1;
                return Ok(true);
            }
            // nvnDeviceGetTimestampInNanoseconds(device, const
            // NVNcounterData*) -> u64 (NVN1's signature): the report's GPU
            // timestamp (offset 8, written by ReportCounter as the physical
            // counter) in nanoseconds.
            "nvnDeviceGetTimestampInNanoseconds" => {
                self.object(this, "Device").map_err(invalid)?;
                let ticks = core.memory.read_u64(x[1] + 8)?;
                let nanoseconds = u128::from(ticks) * 1_000_000_000
                    / u128::from(echora_guest::PHYSICAL_COUNTER_FREQUENCY_HZ);
                core.cpu.regs[0] = nanoseconds as u64;
                return Ok(true);
            }
            // nvnQueueBuilderGetQueueMemorySize(builder) -> size: the
            // memory the application must allocate and pass back with
            // SetQueueMemory (main+0xf49150 allocates it 4 KiB aligned).
            // The host queue keeps nothing there; the answer is the sum of
            // the command, compute and control sizes the builder set.
            "nvnQueueBuilderGetQueueMemorySize" => {
                let bag = self
                    .bags
                    .get(&this)
                    .filter(|bag| bag.kind == "QueueBuilder")
                    .ok_or_else(|| invalid(format!("no QueueBuilder at {this:#x}")))?;
                let mut size = 0u64;
                for property in ["CommandMemorySize", "ComputeMemorySize", "ControlMemorySize"] {
                    let Some(value) = bag.props.get(property) else {
                        return Err(stop(format!("{property} was never set")));
                    };
                    size += value.x[0] & 0xffff_ffff;
                }
                core.cpu.regs[0] = size;
                return Ok(true);
            }
            // nvnDeviceSetWindowOriginMode / nvnDeviceSetDepthMode (device,
            // enum). The application passes 1 for both; recorded on the
            // device. The renderer draws with an upper-left window origin
            // and [0, 1] depth, the modes 1 select (see nvn_gpu's viewport
            // and depth range for the evidence).
            "nvnDeviceSetWindowOriginMode" | "nvnDeviceSetDepthMode" => {
                let mode = x[1] as u32;
                if mode > 1 {
                    return Err(stop(format!("mode {mode} is not modeled")));
                }
                let device = self.objects.get_mut(&this).filter(|o| o.kind == "Device");
                let device = device.ok_or_else(|| invalid("device is not initialized".into()))?;
                let property = name.trim_start_matches("nvnDeviceSet");
                device.state.insert(property.into(), args);
                return Ok(true);
            }
            // nvnQueueInitialize(NVNqueue*, const NVNqueueBuilder*) ->
            // NVNboolean. The host queue executes submitted work in order at
            // submission; the builder's memory sizes and flush threshold are
            // recorded and need no reservation.
            "nvnQueueInitialize" => {
                let builder = self.bag(x[1], "QueueBuilder").map_err(invalid)?;
                self.require_device(&builder).map_err(invalid)?;
                self.initialize(this, "Queue", builder).map_err(invalid)?;
                core.cpu.regs[0] = 1;
                return Ok(true);
            }
            // nvnSyncInitialize(NVNsync*, NVNdevice*) -> NVNboolean. A new
            // sync object is unsignaled until a queue fence signals it.
            "nvnSyncInitialize" => {
                self.object(x[1], "Device").map_err(invalid)?;
                self.initialize(this, "Sync", BTreeMap::new())
                    .map_err(invalid)?;
                core.cpu.regs[0] = 1;
                return Ok(true);
            }
            // nvnSyncWait(const NVNsync*, u64 timeout_ns) -> NVNsyncWaitResult.
            // The application treats 0 as "already signaled" and anything else
            // as not yet (inferred from its polling loop with timeout 0, which
            // flushes the queue and retries). Host work completes at
            // submission, so a fenced sync is always signaled; an unfenced one
            // can only time out: 2 (inferred "timeout expired") for a zero
            // timeout, and an explicit stop for a blocking wait that would
            // never finish.
            "nvnSyncWait" => {
                self.object(this, "Sync").map_err(invalid)?;
                // A blocking wait on a sync a blocked queue will signal: the
                // queues are drained (the CPU would wait forever otherwise).
                if x[1] != 0 && self.sync_fence_pending(this) {
                    self.drain_queues(core)?;
                }
                let sync = self.object(this, "Sync").map_err(invalid)?;
                let signaled = sync.state.contains_key("Signaled");
                core.cpu.regs[0] = if signaled {
                    0
                } else if x[1] == 0 {
                    *self.sync_timeouts.entry(this).or_default() += 1;
                    2
                } else {
                    return Err(stop(
                        "blocking wait on a sync that no queue will signal".into(),
                    ));
                };
                return Ok(true);
            }
            // nvnMemoryPoolInitialize(NVNmemoryPool*, const NVNmemoryPoolBuilder*)
            // -> NVNboolean. Storage is the application's own guest memory
            // (SetStorage: pointer, size); the pool gets a host GPU address
            // range of the same size. Host rendering reads guest memory
            // directly, so pools are coherent.
            "nvnMemoryPoolInitialize" => {
                let builder = self.bag(x[1], "MemoryPoolBuilder").map_err(invalid)?;
                self.require_device(&builder).map_err(invalid)?;
                let storage = builder
                    .get("Storage")
                    .copied()
                    .ok_or_else(|| invalid("memory pool storage was never set".into()))?;
                let (memory, size) = (storage.x[0], storage.x[1]);
                if size == 0 || size.checked_add(memory).is_none() {
                    return Err(invalid(format!(
                        "memory pool storage {memory:#x}+{size:#x}"
                    )));
                }
                if self.next_gpu_address == 0 {
                    self.next_gpu_address = GPU_ADDRESS_BASE;
                }
                let gpu = self.next_gpu_address;
                self.next_gpu_address = (gpu + size)
                    .checked_next_multiple_of(GPU_ADDRESS_ALIGNMENT)
                    .ok_or_else(|| invalid("GPU address space exhausted".into()))?;
                self.initialize(this, "MemoryPool", builder)
                    .map_err(invalid)?;
                let pool = self.objects.get_mut(&this).unwrap();
                pool.state.insert(
                    "GpuAddress".into(),
                    Args {
                        x: [gpu, 0, 0, 0],
                        v: [0; 2],
                    },
                );
                core.cpu.regs[0] = 1;
                return Ok(true);
            }
            // nvnMemoryPoolGetBufferAddress(const NVNmemoryPool*) -> u64.
            // A pool not yet initialized answers 0: the application tests
            // for it before initializing a pool lazily (main+0xb45f44), where
            // the driver reads its still-zeroed structure.
            "nvnMemoryPoolGetBufferAddress" => {
                core.cpu.regs[0] = match self.objects.get(&this) {
                    None => 0,
                    Some(_) => {
                        let pool = self.object(this, "MemoryPool").map_err(invalid)?;
                        pool.state["GpuAddress"].x[0]
                    }
                };
                return Ok(true);
            }
            // nvnMemoryPoolMapVirtual(NVNmemoryPool* virtual, int count,
            // const NVNmappingRequest*) -> NVNboolean. A request (0x28
            // bytes, from the application's call at main+0x11a9d40): u64
            // physical pool, s64 physical offset, s64 virtual offset, u64
            // size, u16 storage class. A null physical pool unmaps. The
            // virtual pool (no storage) keeps each mapping; its GPU range
            // resolves through it to the physical pool's storage.
            "nvnMemoryPoolMapVirtual" => {
                let pool = self.object(this, "MemoryPool").map_err(invalid)?;
                let storage = pool.builder["Storage"];
                if storage.x[0] != 0 {
                    return Err(stop("MapVirtual on a pool with storage".into()));
                }
                let pool_size = storage.x[1];
                let count = x[1] as i32;
                if !(0..=4096).contains(&count) {
                    return Err(invalid(format!("{count} mapping requests")));
                }
                let mut mappings = Vec::new();
                for index in 0..count as u64 {
                    let at = x[2] + index * 0x28;
                    let word = |offset: u64| core.memory.read_u64(at + offset);
                    let request = [word(0)?, word(8)?, word(0x10)?, word(0x18)?];
                    mappings.push(request);
                }
                for [physical, physical_offset, offset, size] in mappings {
                    if size == 0 || offset.checked_add(size).is_none_or(|end| end > pool_size) {
                        return Err(invalid(format!(
                            "virtual mapping {offset:#x}+{size:#x} outside the pool"
                        )));
                    }
                    if physical != 0 {
                        let target = self.object(physical, "MemoryPool").map_err(invalid)?;
                        let target = target.builder["Storage"];
                        if target.x[0] == 0
                            || physical_offset
                                .checked_add(size)
                                .is_none_or(|end| end > target.x[1])
                        {
                            return Err(invalid(format!(
                                "physical range {physical_offset:#x}+{size:#x} of pool {physical:#x}"
                            )));
                        }
                    }
                    let pool = self.objects.get_mut(&this).unwrap();
                    let overlapping: Vec<(String, u64, u64)> = pool
                        .state
                        .iter()
                        .filter_map(|(name, mapping)| {
                            let start = u64::from_str_radix(name.strip_prefix(VIRTUAL_MAPPING)?, 16)
                                .ok()?;
                            let end = start + mapping.x[2];
                            (start < offset + size && offset < end)
                                .then(|| (name.clone(), start, end))
                        })
                        .collect();
                    for (name, start, end) in overlapping {
                        if start < offset || end > offset + size {
                            return Err(stop(format!(
                                "virtual mapping {offset:#x}+{size:#x} splits {start:#x}..{end:#x}"
                            )));
                        }
                        pool.state.remove(&name);
                    }
                    if physical != 0 {
                        pool.state.insert(
                            format!("{VIRTUAL_MAPPING}{offset:016x}"),
                            Args {
                                x: [physical, physical_offset, size, 0],
                                v: [0; 2],
                            },
                        );
                    }
                }
                self.pool_ranges.borrow_mut().clear();
                core.cpu.regs[0] = 1;
                return Ok(true);
            }
            // nvnMemoryPoolGetSize(const NVNmemoryPool*) -> size_t: the
            // size the builder's SetStorage gave.
            "nvnMemoryPoolGetSize" => {
                let pool = self.object(this, "MemoryPool").map_err(invalid)?;
                core.cpu.regs[0] = pool.builder["Storage"].x[1];
                return Ok(true);
            }
            // nvnMemoryPoolMap(const NVNmemoryPool*) -> void*: the storage.
            "nvnMemoryPoolMap" => {
                let pool = self.object(this, "MemoryPool").map_err(invalid)?;
                core.cpu.regs[0] = pool.builder["Storage"].x[0];
                return Ok(true);
            }
            // nvnMemoryPoolFlushMappedRange / InvalidateMappedRange (pool,
            // offset, size): pools are coherent with host rendering. A flush
            // follows CPU writes, so cached textures in the range are stale.
            "nvnMemoryPoolFlushMappedRange" | "nvnMemoryPoolInvalidateMappedRange" => {
                let pool = self.object(this, "MemoryPool").map_err(invalid)?;
                let (base, size) = (pool.builder["Storage"].x[0], pool.builder["Storage"].x[1]);
                if x[1].checked_add(x[2]).is_none_or(|end| end > size) {
                    return Err(invalid(format!(
                        "range {:#x}+{:#x} outside the pool",
                        x[1], x[2]
                    )));
                }
                let start = base + x[1];
                if name == "nvnMemoryPoolFlushMappedRange" {
                    self.gpu_invalidate(start..start + x[2], None);
                } else if let Err(message) = self.resolve_storage(core, Some(start..start + x[2])) {
                    // The CPU is about to read the range: GPU results in it
                    // come back first.
                    return Err(Trap::InvalidState {
                        message: format!("{name}: dispatch results: {message}"),
                    });
                }
                return Ok(true);
            }
            // nvnBufferInitialize(NVNbuffer*, const NVNbufferBuilder*) ->
            // NVNboolean. SetStorage is (pool, offset, size) within an
            // initialized pool.
            "nvnBufferInitialize" => {
                let builder = self.bag(x[1], "BufferBuilder").map_err(invalid)?;
                self.require_device(&builder).map_err(invalid)?;
                let storage = builder
                    .get("Storage")
                    .copied()
                    .ok_or_else(|| invalid("buffer storage was never set".into()))?;
                let (pool, offset, size) = (storage.x[0], storage.x[1], storage.x[2]);
                let pool_size =
                    self.object(pool, "MemoryPool").map_err(invalid)?.builder["Storage"].x[1];
                if offset.checked_add(size).is_none_or(|end| end > pool_size) {
                    return Err(invalid(format!(
                        "buffer {offset:#x}+{size:#x} outside its pool"
                    )));
                }
                self.initialize(this, "Buffer", builder).map_err(invalid)?;
                core.cpu.regs[0] = 1;
                return Ok(true);
            }
            // nvnBufferGetAddress / nvnBufferMap / nvnBufferGetSize.
            "nvnBufferGetAddress" | "nvnBufferMap" | "nvnBufferGetSize" => {
                let buffer = self.object(this, "Buffer").map_err(invalid)?;
                let storage = buffer.builder["Storage"];
                let pool = &self.objects[&storage.x[0]];
                core.cpu.regs[0] = match name {
                    "nvnBufferGetAddress" => pool.state["GpuAddress"].x[0] + storage.x[1],
                    "nvnBufferMap" => pool.builder["Storage"].x[0] + storage.x[1],
                    _ => storage.x[2],
                };
                return Ok(true);
            }
            // nvnTextureGetTextureAddress -> the GPU address of the
            // texture's storage: its pool's GPU address plus the offset its
            // builder's SetStorage gave, as for buffers.
            "nvnTextureGetTextureAddress" => {
                let texture = self.object(this, "Texture").map_err(invalid)?;
                let storage = *texture
                    .builder
                    .get("Storage")
                    .ok_or_else(|| stop("Texture has no storage".into()))?;
                let pool = self.object(storage.x[0], "MemoryPool").map_err(invalid)?;
                let base = pool
                    .state
                    .get("GpuAddress")
                    .ok_or_else(|| invalid("memory pool has no GPU address".into()))?
                    .x[0];
                core.cpu.regs[0] = base + storage.x[1];
                return Ok(true);
            }
            // nvn{Buffer,Texture}GetMemoryPool / GetMemoryOffset: the pool and
            // offset given to the builder's SetStorage(pool, offset[, size]).
            "nvnBufferGetMemoryPool"
            | "nvnBufferGetMemoryOffset"
            | "nvnTextureGetMemoryPool"
            | "nvnTextureGetMemoryOffset" => {
                let kind = if name.starts_with("nvnBuffer") { "Buffer" } else { "Texture" };
                let object = self.object(this, kind).map_err(invalid)?;
                let storage = object
                    .builder
                    .get("Storage")
                    .ok_or_else(|| stop(format!("{kind} has no storage")))?;
                core.cpu.regs[0] = if name.ends_with("Pool") {
                    storage.x[0]
                } else {
                    storage.x[1]
                };
                return Ok(true);
            }
            // nvnQueueFenceSync(NVNqueue*, NVNsync*, condition, flags): the
            // sync signals when prior work completes, which on the host is at
            // submission, so it is signaled now.
            "nvnQueueFenceSync" => {
                self.object(this, "Queue").map_err(invalid)?;
                if debug_var_os("ECHORA_GPU_FLOW").is_some() {
                    let note = format!("flow queue {this:x} fences sync {:x}", x[1]);
                    self.gpu_note(note);
                }
                self.object(x[1], "Sync").map_err(invalid)?;
                let sync = self.objects.get_mut(&x[1]).unwrap();
                if self.pending.get(&this).is_some_and(|work| !work.is_empty()) {
                    // Signaled when the queue's earlier work has run.
                    sync.state.remove("Signaled");
                    self.pending
                        .get_mut(&this)
                        .unwrap()
                        .push_back(QueueItem::Fence { sync: x[1], args });
                } else {
                    sync.state.insert("Signaled".into(), args);
                }
                return Ok(true);
            }
            // nvnQueueFlush / nvnQueueFinish: queued work runs as far as its
            // waits allow; Finish waits for all of it, so a queue still
            // blocked then is drained.
            "nvnQueueFlush" | "nvnQueueFinish" => {
                self.object(this, "Queue").map_err(invalid)?;
                self.pump_queues(core)?;
                if name == "nvnQueueFinish" {
                    self.drain_queues(core)?;
                }
                return Ok(true);
            }
            // nvnQueueWaitSync(NVNqueue*, const NVNsync*) -> NVNboolean: later
            // work waits for the sync. A fenced sync is already signaled; an
            // unfenced one would block the queue forever, so it stops.
            "nvnQueueWaitSync" => {
                self.object(this, "Queue").map_err(invalid)?;
                if debug_var_os("ECHORA_GPU_FLOW").is_some() {
                    let note = format!("flow queue {this:x} waits sync {:x}", x[1]);
                    self.gpu_note(note);
                }
                self.object(x[1], "Sync").map_err(invalid)?;
                self.pending
                    .entry(this)
                    .or_default()
                    .push_back(QueueItem::WaitSync { sync: x[1] });
                self.pump_queues(core)?;
                core.cpu.regs[0] = 1;
                return Ok(true);
            }
            // nvnCommandBufferInitialize(NVNcommandBuffer*, NVNdevice*) ->
            // NVNboolean. Host command buffers record calls, not GPU words,
            // so the command and control memory the application adds is
            // tracked for its size queries but never consumed.
            "nvnCommandBufferInitialize" => {
                self.object(x[1], "Device").map_err(invalid)?;
                self.initialize(this, "CommandBuffer", BTreeMap::new())
                    .map_err(invalid)?;
                core.cpu.regs[0] = 1;
                return Ok(true);
            }
            // AddCommandMemory(cmd, pool, offset, size) and
            // AddControlMemory(cmd, void* memory, size).
            "nvnCommandBufferAddCommandMemory" | "nvnCommandBufferAddControlMemory" => {
                self.object(this, "CommandBuffer").map_err(invalid)?;
                let (key, size) = if name.ends_with("CommandMemory") {
                    self.object(x[1], "MemoryPool").map_err(invalid)?;
                    ("CommandMemory", x[3])
                } else {
                    ("ControlMemory", x[2])
                };
                let object = self.objects.get_mut(&this).unwrap();
                let total = object.state.entry(key.into()).or_default();
                total.x[0] = size;
                // New memory starts unused.
                object.state.remove(&format!("{key}Used"));
                return Ok(true);
            }
            "nvnCommandBufferGetCommandMemorySize"
            | "nvnCommandBufferGetControlMemorySize"
            | "nvnCommandBufferGetCommandMemoryFree"
            | "nvnCommandBufferGetControlMemoryFree"
            | "nvnCommandBufferGetCommandMemoryUsed"
            | "nvnCommandBufferGetControlMemoryUsed" => {
                let object = self.object(this, "CommandBuffer").map_err(invalid)?;
                let key = if name.contains("Command") && name.matches("Command").count() == 2 {
                    "CommandMemory"
                } else {
                    "ControlMemory"
                };
                let size = object.state.get(key).map_or(0, |value| value.x[0]);
                let used = object
                    .state
                    .get(&format!("{key}Used"))
                    .map_or(0, |value| value.x[0]);
                core.cpu.regs[0] = if name.ends_with("Used") {
                    used
                } else if name.ends_with("Free") {
                    size - used
                } else {
                    size
                };
                return Ok(true);
            }
            // Memory callbacks are recorded; the host never runs out of
            // command memory, so they are never invoked.
            "nvnCommandBufferSetMemoryCallback"
            | "nvnCommandBufferSetMemoryCallbackData"
            | "nvnCommandBufferSetCommandMemoryCallbackEnabled"
            | "nvnCommandBufferSetControlMemoryCallbackEnabled"
            | "nvnCommandBufferSetDebugLabel"
            // SetRecordingMode(cmd, mode): how the driver writes command
            // memory (inferred). Host command buffers record calls either way.
            | "nvnCommandBufferSetRecordingMode" => {
                self.object(this, "CommandBuffer").map_err(invalid)?;
                let property = name.trim_start_matches("nvnCommandBufferSet");
                let object = self.objects.get_mut(&this).unwrap();
                object.state.insert(property.into(), args);
                return Ok(true);
            }
            "nvnCommandBufferBeginRecording" => {
                self.object(this, "CommandBuffer").map_err(invalid)?;
                if self.recording.insert(this, Vec::new()).is_some() {
                    return Err(invalid("command buffer is already recording".into()));
                }
                return Ok(true);
            }
            "nvnCommandBufferIsRecording" => {
                self.object(this, "CommandBuffer").map_err(invalid)?;
                core.cpu.regs[0] = u64::from(self.recording.contains_key(&this));
                return Ok(true);
            }
            // EndRecording(cmd) -> NVNcommandHandle: a nonzero host handle.
            "nvnCommandBufferEndRecording" => {
                let commands = self
                    .recording
                    .remove(&this)
                    .ok_or_else(|| invalid("command buffer is not recording".into()))?;
                if !commands.is_empty() {
                    self.use_command_memory(this, "ControlMemory", CONTROL_MEMORY_PER_LIST);
                }
                self.next_command_handle += 1;
                let handle = 0x4e56_0000_0000_0000 | self.next_command_handle;
                // Only in a sequenced present: lists are recorded by the
                // dozen per frame.
                if debug_var_os("ECHORA_GPU_FLOW").is_some() && self.sequencing() {
                    let mut names: BTreeMap<&str, usize> = BTreeMap::new();
                    for command in &commands {
                        *names
                            .entry(command.name.trim_start_matches("nvnCommandBuffer"))
                            .or_default() += 1;
                    }
                    let work: Vec<String> = names
                        .iter()
                        .filter(|(name, _)| {
                            is_work_command(name) || name.starts_with("Call") || name.ends_with("Event")
                        })
                        .map(|(name, count)| format!("{name} {count}"))
                        .collect();
                    let note = format!(
                        "flow recorded {this:x} list {handle:x} from lr {:#x} commands {} [{}]",
                        core.cpu.regs[30],
                        commands.len(),
                        work.join(", ")
                    );
                    self.gpu_note(note);
                }
                self.command_list_clock += 1;
                self.command_list_commands += commands.len();
                self.command_lists.insert(
                    handle,
                    (std::sync::Arc::new(commands), self.command_list_clock, 0),
                );
                self.command_list_use.insert(self.command_list_clock, handle);
                // Command memory is reused by later recordings, and the
                // application rarely finalizes handles: keep the most
                // recently recorded or submitted lists. Each holds its
                // commands' state snapshots; gameplay lists hold ~500
                // commands, and 65536 (later 8192) lists grew to gigabytes.
                while self.command_lists.len() > COMMAND_LIST_LIMIT
                    || self.command_list_commands > COMMAND_LIMIT
                {
                    let Some((_, oldest)) = self.command_list_use.pop_first() else {
                        break;
                    };
                    // A list submitted more than once is reused by the
                    // application: it gets a second chance (its count
                    // restarts) before one-shot lists behind it go.
                    if let Some(entry) = self.command_lists.get_mut(&oldest) {
                        if entry.2 > 1 {
                            entry.2 = 1;
                            self.command_list_clock += 1;
                            entry.1 = self.command_list_clock;
                            self.command_list_use.insert(self.command_list_clock, oldest);
                            continue;
                        }
                    }
                    if let Some((list, ..)) = self.command_lists.remove(&oldest) {
                        self.command_list_commands -= list.len();
                    }
                }
                core.cpu.regs[0] = handle;
                return Ok(true);
            }
            // nvnDeviceFinalizeCommandHandle(device, NVNcommandHandle): the
            // handle's command list is released; it cannot be submitted again.
            "nvnDeviceFinalizeCommandHandle" => {
                self.object(this, "Device").map_err(invalid)?;
                let Some((list, used, _)) = self.command_lists.remove(&x[1]) else {
                    return Err(invalid(format!("unknown command handle {:#x}", x[1])));
                };
                self.command_list_commands -= list.len();
                self.command_list_use.remove(&used);
                return Ok(true);
            }
            // nvnQueueSubmitCommands(NVNqueue*, int count, const
            // NVNcommandHandle* handles): executed in order at submission.
            "nvnQueueSubmitCommands" => {
                self.object(this, "Queue").map_err(invalid)?;
                // Bindings are resolved anew per submission: keeping them
                // for a whole present measured no faster.
                self.binding_serial += 1;
                let count = x[1] as u32;
                if count > 4096 {
                    return Err(invalid(format!("{count} command handles")));
                }
                for index in 0..u64::from(count) {
                    let handle = core.memory.read_u64(x[2] + index * 8)?;
                    let Some((list, used, submits)) = self.command_lists.get_mut(&handle) else {
                        // A list dropped from the kept window (the
                        // application seldom finalizes handles, so lists
                        // are kept by recent use): skipped, counted.
                        if handle >> 48 == 0x4e56 {
                            self.stale_command_submissions += 1;
                            continue;
                        }
                        return Err(invalid(format!("unknown command handle {handle:#x}")));
                    };
                    *submits = submits.saturating_add(1);
                    let list = list.clone();
                    // Submitted again: most recently used.
                    self.command_list_clock += 1;
                    let previous = std::mem::replace(used, self.command_list_clock);
                    self.command_list_use.remove(&previous);
                    self.command_list_use.insert(self.command_list_clock, handle);
                    if debug_var_os("ECHORA_GPU_FLOW").is_some() {
                        let work = list
                            .iter()
                            .filter(|command| command.name.contains("Draw") || command.name.contains("Dispatch"))
                            .count();
                        let waits = list.iter().filter(|command| command.name.ends_with("WaitSync")).count();
                        let note = format!(
                            "flow submit queue {this:x} list {handle:x} from lr {:#x} commands {} work {work} waits {waits}",
                            core.cpu.regs[30],
                            list.len()
                        );
                        self.gpu_note(note);
                    }
                    self.pending
                        .entry(this)
                        .or_default()
                        .push_back(QueueItem::List {
                            commands: list,
                            next: 0,
                        });
                }
                self.pump_queues(core)?;
                return Ok(true);
            }
            // nvnTextureGetSwizzle(texture, int* r, int* g, int* b, int* a):
            // the builder's SetSwizzle, or the identity (R, G, B, A: 2, 3, 4,
            // 5 in the numbering nvn_gpu decodes) when never set.
            "nvnTextureGetSwizzle" => {
                let texture = self.object(this, "Texture").map_err(invalid)?;
                let swizzle = texture.builder.get("Swizzle").map_or([2, 3, 4, 5], |value| value.x);
                for (index, component) in swizzle.iter().enumerate() {
                    let out = x[1 + index];
                    if out == 0 {
                        return Err(invalid("null swizzle output".into()));
                    }
                    core.memory.write_u32(out, *component as u32)?;
                }
                return Ok(true);
            }
            // nvnTextureGetSamples(texture) -> int: the builder's SetSamples,
            // 0 (single-sampled) when never set, as SetDefaults leaves it.
            "nvnTextureGetSamples" => {
                let texture = self.object(this, "Texture").map_err(invalid)?;
                core.cpu.regs[0] = texture.builder.get("Samples").map_or(0, |value| value.x[0]);
                return Ok(true);
            }
            // nvnTextureBuilderGetStorageClass / nvnTextureGetStorageClass:
            // the application stores the class as a u16 in its texture info
            // (main+0xc2ce4c) and nothing nearby interprets it. Host textures
            // have one storage class, answered as 1. Provisional.
            "nvnTextureBuilderGetStorageClass" | "nvnTextureGetStorageClass" => {
                if name.starts_with("nvnTextureBuilder") {
                    self.bag(this, "TextureBuilder").map_err(invalid)?;
                } else {
                    self.object(this, "Texture").map_err(invalid)?;
                }
                core.cpu.regs[0] = 1;
                return Ok(true);
            }
            // nvnTextureBuilderGetZCullStorageSize (builder) -> size: the
            // memory that saves and restores the described depth target's
            // culling state; the application keeps it beside device
            // integer 54, the alignment of that memory (main+0x244db4).
            // The host renderer keeps no depth culling state: 0 bytes.
            // Provisional.
            "nvnTextureBuilderGetZCullStorageSize" => {
                self.bag(this, "TextureBuilder").map_err(invalid)?;
                core.cpu.regs[0] = 0;
                return Ok(true);
            }
            // nvnTextureBuilderGetStorageSize / GetStorageAlignment (builder)
            // and nvnTextureGetStorageSize (texture): the block-linear size of
            // the described texture (see nvn_format).
            "nvnTextureBuilderGetStorageSize"
            | "nvnTextureBuilderGetStorageAlignment"
            | "nvnTextureGetStorageSize" => {
                let props = if name.starts_with("nvnTextureBuilder") {
                    self.bag(this, "TextureBuilder").map_err(invalid)?
                } else {
                    self.object(this, "Texture")
                        .map_err(invalid)?
                        .builder
                        .clone()
                };
                let (size, alignment) = texture_storage(&props).map_err(stop)?;
                core.cpu.regs[0] = if name.ends_with("Alignment") {
                    alignment
                } else {
                    size
                };
                return Ok(true);
            }
            // nvnSamplerInitialize(NVNsampler*, const NVNsamplerBuilder*) ->
            // NVNboolean.
            // nvnEventSignal(NVNevent*, mode, u32 value) from the CPU and
            // nvnEventGetValue(const NVNevent*) -> u32: the event is the u32
            // at its storage (memory pool, offset). Mode 0 writes the value
            // and 1 adds it (inferred; the application signals mode 0 with
            // value 0 at main+0x2cb41c).
            "nvnEventSignal" | "nvnEventGetValue" => {
                let address = self.event_address(this).map_err(invalid)?;
                let flow = debug_var_os("ECHORA_GPU_FLOW").is_some() && self.sequencing();
                if name == "nvnEventGetValue" {
                    let value = core.memory.read_u32(address)?;
                    if flow {
                        let note = format!(
                            "flow event {this:x} read {value} from lr {:#x}",
                            core.cpu.regs[30]
                        );
                        self.gpu_note(note);
                    }
                    core.cpu.regs[0] = u64::from(value);
                    return Ok(true);
                }
                let (mode, value) = (x[1] as u32, x[2] as u32);
                if flow {
                    let note = format!(
                        "flow event {this:x} cpu signal mode {mode} value {value} from lr {:#x}",
                        core.cpu.regs[30]
                    );
                    self.gpu_note(note);
                }
                let new = match mode {
                    0 => value,
                    1 => core.memory.read_u32(address)?.wrapping_add(value),
                    _ => return Err(stop(format!("event signal mode {mode} is not modeled"))),
                };
                core.memory.write_u32(address, new)?;
                self.pump_queues(core)?;
                return Ok(true);
            }
            // nvnEventInitialize(NVNevent*, const NVNeventBuilder*) ->
            // NVNboolean. The builder's storage (memory pool, offset) is
            // recorded; signal and wait calls are modeled as they appear.
            "nvnEventInitialize" => {
                let builder = self.bag(x[1], "EventBuilder").map_err(invalid)?;
                self.initialize(this, "Event", builder).map_err(invalid)?;
                core.cpu.regs[0] = 1;
                return Ok(true);
            }
            "nvnSamplerInitialize" => {
                let builder = self.bag(x[1], "SamplerBuilder").map_err(invalid)?;
                self.require_device(&builder).map_err(invalid)?;
                self.initialize(this, "Sampler", builder).map_err(invalid)?;
                core.cpu.regs[0] = 1;
                return Ok(true);
            }
            "nvnSamplerFinalize" | "nvnTexturePoolFinalize" | "nvnSamplerPoolFinalize" => {
                let kind = name.trim_start_matches("nvn").trim_end_matches("Finalize");
                self.object(this, kind).map_err(invalid)?;
                self.objects.remove(&this);
                return Ok(true);
            }
            // nvnTexturePoolRegisterTexture / RegisterImage(pool, int id, const
            // NVNtexture*, const NVNtextureView*) and
            // nvnSamplerPoolRegisterSampler(pool, int id, const NVNsampler*) /
            // RegisterSamplerBuilder(pool, int id, const NVNsamplerBuilder*):
            // the descriptor at `id` now describes that object. The host keeps
            // the registration (with the view's properties, read now) instead
            // of writing hardware descriptors into the pool memory.
            "nvnTexturePoolRegisterTexture"
            | "nvnTexturePoolRegisterImage"
            | "nvnSamplerPoolRegisterSampler"
            | "nvnSamplerPoolRegisterSamplerBuilder" => {
                let (kind, rest) = split_name(name).unwrap();
                let count = self.object(this, kind).map_err(invalid)?.builder["Storage"].x[2] as u32;
                let id = x[1] as u32;
                if id >= count {
                    return Err(invalid(format!("descriptor {id} outside a pool of {count}")));
                }
                // Handles naming this descriptor resolve anew (texture ids
                // are bits 0..20 of a handle, sampler ids bits 20..32).
                let sampler = rest.starts_with("RegisterSampler");
                self.binding_memo.1.retain(|handle, _| {
                    let named = if sampler {
                        handle >> 20 & 0xfff
                    } else {
                        handle & 0xf_ffff
                    };
                    named != u64::from(id)
                });
                let mut entry = Args::default();
                match rest {
                    "RegisterSamplerBuilder" => {
                        self.bag(x[2], "SamplerBuilder").map_err(invalid)?;
                    }
                    "RegisterSampler" => {
                        self.object(x[2], "Sampler").map_err(invalid)?;
                    }
                    _ => {
                        self.object(x[2], "Texture").map_err(invalid)?;
                        if x[3] != 0 {
                            self.bag(x[3], "TextureView").map_err(invalid)?;
                            let view = self.bags[&x[3]].clone();
                            self.views.insert((this, id), view);
                        } else {
                            self.views.remove(&(this, id));
                        }
                    }
                }
                entry.x[0] = x[2];
                entry.x[1] = u64::from(rest == "RegisterImage");
                // A hardware descriptor captures the texture as it is now:
                // its storage tells a later reuse of the structure apart.
                entry.x[2] = self.texture_storage(x[2]).unwrap_or(0);
                let pool = self.objects.get_mut(&this).unwrap();
                pool.state.insert(format!("Descriptor{id}"), entry);
                return Ok(true);
            }
            // nvn{Texture,Sampler}PoolGetMemoryPool / GetMemoryOffset / GetSize:
            // the initialization arguments (pool, offset, descriptor count).
            "nvnTexturePoolGetMemoryPool"
            | "nvnTexturePoolGetMemoryOffset"
            | "nvnTexturePoolGetSize"
            | "nvnSamplerPoolGetMemoryPool"
            | "nvnSamplerPoolGetMemoryOffset"
            | "nvnSamplerPoolGetSize" => {
                let (kind, rest) = split_name(name).unwrap();
                let storage = self.object(this, kind).map_err(invalid)?.builder["Storage"];
                core.cpu.regs[0] = match rest {
                    "GetMemoryPool" => storage.x[0],
                    "GetMemoryOffset" => storage.x[1],
                    _ => storage.x[2] as u32 as u64,
                };
                return Ok(true);
            }
            // nvnDeviceGet{Texture,TexelFetch,Image,SeparateTexture,
            // SeparateSampler}Handle(device, ...) -> u64 handles shaders use
            // to address descriptors. Packed as Maxwell bindless handles
            // (texture index in bits 0..19, sampler index in bits 20..31;
            // inferred), since translated shaders will decode them that way.
            "nvnDeviceGetTextureHandle" => {
                self.object(this, "Device").map_err(invalid)?;
                let (texture, sampler) = (x[1] as u32, x[2] as u32);
                if texture >= 1 << 20 || sampler >= 1 << 12 {
                    return Err(invalid(format!("handle for {texture}/{sampler}")));
                }
                core.cpu.regs[0] = u64::from(texture) | (u64::from(sampler) << 20);
                return Ok(true);
            }
            "nvnDeviceGetTexelFetchHandle"
            | "nvnDeviceGetImageHandle"
            | "nvnDeviceGetSeparateTextureHandle" => {
                self.object(this, "Device").map_err(invalid)?;
                let texture = x[1] as u32;
                if texture >= 1 << 20 {
                    return Err(invalid(format!("handle for texture {texture}")));
                }
                core.cpu.regs[0] = u64::from(texture);
                return Ok(true);
            }
            "nvnDeviceGetSeparateSamplerHandle" => {
                self.object(this, "Device").map_err(invalid)?;
                let sampler = x[1] as u32;
                if sampler >= 1 << 12 {
                    return Err(invalid(format!("handle for sampler {sampler}")));
                }
                core.cpu.regs[0] = u64::from(sampler) << 20;
                return Ok(true);
            }
            // nvnTexture{Flush,Invalidate}Texels(texture, view, region) and
            // nvnBuffer{Flush,Invalidate}MappedRange(buffer, offset, size):
            // CPU cache maintenance. Host storage is coherent.
            "nvnTextureFlushTexels" | "nvnTextureInvalidateTexels" => {
                self.object(this, "Texture").map_err(invalid)?;
                if name == "nvnTextureFlushTexels" {
                    self.gpu_invalidate(0..0, Some(this));
                }
                return Ok(true);
            }
            "nvnBufferFlushMappedRange" | "nvnBufferInvalidateMappedRange" => {
                let size = self.object(this, "Buffer").map_err(invalid)?.builder["Storage"].x[2];
                if x[1].checked_add(x[2]).is_none_or(|end| end > size) {
                    return Err(invalid(format!("range {:#x}+{:#x} outside the buffer", x[1], x[2])));
                }
                return Ok(true);
            }
            // nvnTextureWriteTexels(const NVNtexture*, const NVNtextureView*,
            // const NVNcopyRegion*, const void* data): the CPU writes tightly
            // packed texels (format blocks, row after row) into the texture's
            // storage. NVNcopyRegion is six ints: x, y, z offsets and width,
            // height, depth (NVN's public layout). Storage is block-linear
            // (buffer textures are linear), so the write is swizzled the way
            // the GPU reads it. Level 0 of a single-layer 1D/2D/rectangle or
            // buffer texture is modeled.
            // nvnTextureReadTexels has the same arguments and copies the
            // other way, into the CPU buffer: from the GPU target when the
            // texture was rendered to, else from guest memory.
            "nvnTextureWriteTexels" | "nvnTextureReadTexels" => {
                let view = if x[1] != 0 {
                    Some(self.bag(x[1], "TextureView").map_err(invalid)?)
                } else {
                    None
                };
                let region = read_region(core, x[2])?;
                let to_texture = name == "nvnTextureWriteTexels";
                let base_view = view.as_ref().is_none_or(|view| {
                    ["Levels", "Layers"]
                        .iter()
                        .all(|key| view.get(*key).is_none_or(|value| value.x[0] == 0))
                });
                if to_texture && debug_var_os("ECHORA_GPU_FLOW").is_some() {
                    let mut sample = [0u8; 16];
                    let _ = core.memory.read(x[3], &mut sample);
                    let note = format!(
                        "flow write texels {this:x} region {region:?} from lr {:#x} first {sample:02x?}",
                        core.cpu.regs[30]
                    );
                    self.gpu_note(note);
                }
                if !to_texture && base_view {
                    match self.gpu_read_texels(core, this, region, x[3]) {
                        Ok(true) => return Ok(true),
                        Ok(false) => {}
                        Err(error) => self.gpu_note(format!("texel read: {error}")),
                    }
                }
                self.copy_texels(core, this, view.as_ref(), region, x[3], to_texture)
                    .map_err(|error| error.into_trap(name, core.cpu.pc))?;
                if to_texture {
                    self.gpu_invalidate(0..0, Some(this));
                }
                return Ok(true);
            }
            // nvnTexturePoolInitialize / nvnSamplerPoolInitialize(pool, const
            // NVNmemoryPool*, ptrdiff_t offset, int count) -> NVNboolean: the
            // descriptor table lives in the memory pool at offset (each entry
            // is the descriptor size from GetInteger 15/16).
            "nvnTexturePoolInitialize" | "nvnSamplerPoolInitialize" => {
                let pool_size =
                    self.object(x[1], "MemoryPool").map_err(invalid)?.builder["Storage"].x[1];
                let count = x[3] as u32;
                let end = x[2].checked_add(u64::from(count) * 32);
                if count == 0 || end.is_none_or(|end| end > pool_size) {
                    return Err(invalid(format!(
                        "{count} descriptors at {:#x} outside the pool",
                        x[2]
                    )));
                }
                let kind = name
                    .trim_start_matches("nvn")
                    .trim_end_matches("Initialize");
                let mut builder = BTreeMap::new();
                builder.insert("Storage".to_string(), args);
                self.initialize(this, kind, builder).map_err(invalid)?;
                core.cpu.regs[0] = 1;
                return Ok(true);
            }
            // nvnTextureInitialize(NVNtexture*, const NVNtextureBuilder*) ->
            // NVNboolean. Builder properties are kept for rendering; the
            // texture's storage stays the application's pool memory.
            "nvnTextureInitialize" => {
                self.texture_epoch += 1;
                let builder = self.bag(x[1], "TextureBuilder").map_err(invalid)?;
                self.require_device(&builder).map_err(invalid)?;
                let previous = self
                    .texture_storage(this)
                    .or_else(|| self.finalized_storage.remove(&this));
                if let Some(object) = self.objects.get(&this).filter(|o| o.kind == "Texture") {
                    if let Some(storage) = self.texture_storage(this) {
                        self.objects.insert(retired_texture(storage), object.clone());
                    }
                }
                self.initialize(this, "Texture", builder).map_err(invalid)?;
                // A texture structure reused for other storage: what the GPU
                // kept under its address describes the old texture.
                let storage = self.texture_storage(this);
                // ECHORA_NVN_TEXTURE_LOG=<low>-<high> (hex) logs only textures
                // whose storage starts in that range.
                let logged = std::env::var("ECHORA_NVN_TEXTURE_LOG").ok().is_some_and(|range| {
                    let bounds = range.split_once('-').and_then(|(low, high)| {
                        let parse = |text: &str| u64::from_str_radix(text.trim_start_matches("0x"), 16).ok();
                        Some((parse(low)?, parse(high)?))
                    });
                    match (bounds, storage) {
                        (Some((low, high)), Some(storage)) => (low..high).contains(&storage),
                        (Some(_), None) => false,
                        (None, _) => true,
                    }
                });
                if self.texture_log.len() < 512 && logged {
                    let get = |key: &str| self.objects[&this].builder.get(key).map_or(0, |v| v.x[0]);
                    self.texture_log.push(format!(
                        "format {} target {} {}x{}x{} levels {} stride {:#x} storage {:#x} after {} frames",
                        get("Format"),
                        get("Target"),
                        get("Width"),
                        get("Height"),
                        get("Depth"),
                        get("Levels"),
                        get("Stride"),
                        storage.unwrap_or(0),
                        self.presented_frames,
                    ));
                }
                if let Some(previous) = previous.filter(|previous| Some(*previous) != storage) {
                    self.gpu_move_texture(this, previous, storage);
                }
                core.cpu.regs[0] = 1;
                return Ok(true);
            }
            "nvnTextureFinalize" | "nvnWindowFinalize" | "nvnCommandBufferFinalize" => {
                self.texture_epoch += 1;
                let kind = name.trim_start_matches("nvn").trim_end_matches("Finalize");
                self.object(this, kind).map_err(invalid)?;
                if let Some(storage) = self.texture_storage(this) {
                    self.finalized_storage.insert(this, storage);
                    let object = self.objects[&this].clone();
                    self.objects.insert(retired_texture(storage), object);
                }
                if kind == "Texture" {
                    self.gpu_forget_texture(this);
                }
                self.objects.remove(&this);
                self.images.remove(&this);
                self.recording.remove(&this);
                return Ok(true);
            }
            // nvnWindowInitialize(NVNwindow*, const NVNwindowBuilder*) ->
            // NVNboolean. SetTextures(builder, count, const NVNtexture* const*)
            // supplies the presentable textures; their pointers are read now.
            "nvnWindowInitialize" => {
                let builder = self.bag(x[1], "WindowBuilder").map_err(invalid)?;
                self.require_device(&builder).map_err(invalid)?;
                let textures = builder
                    .get("Textures")
                    .copied()
                    .ok_or_else(|| invalid("window textures were never set".into()))?;
                let count = textures.x[0];
                if count == 0 || count > 16 {
                    return Err(invalid(format!("{count} window textures")));
                }
                self.initialize(this, "Window", builder).map_err(invalid)?;
                let mut pointers = Vec::new();
                for index in 0..count {
                    pointers.push(core.memory.read_u64(textures.x[1] + index * 8)?);
                }
                let window = self.objects.get_mut(&this).unwrap();
                for (index, texture) in pointers.iter().enumerate() {
                    let mut entry = Args::default();
                    entry.x[0] = *texture;
                    window.state.insert(format!("Texture{index}"), entry);
                }
                window.state.insert("NextTexture".into(), Args::default());
                core.cpu.regs[0] = 1;
                return Ok(true);
            }
            // nvnWindowAcquireTexture(NVNwindow*, NVNsync*, int* index) ->
            // NVNwindowAcquireTextureResult: textures are handed out in order;
            // the host display never holds one, so the sync is signaled now
            // and the result is 0 (success, inferred from the application
            // treating 0 as success).
            "nvnWindowAcquireTexture" => {
                self.object(x[1], "Sync").map_err(invalid)?;
                let window = self
                    .objects
                    .get_mut(&this)
                    .filter(|object| object.kind == "Window")
                    .ok_or_else(|| invalid("window is not initialized".into()))?;
                let count = window.builder["Textures"].x[0];
                let next = window.state["NextTexture"].x[0];
                window.state.get_mut("NextTexture").unwrap().x[0] = (next + 1) % count;
                core.memory.write_u32(x[2], next as u32)?;
                self.objects
                    .get_mut(&x[1])
                    .unwrap()
                    .state
                    .insert("Signaled".into(), args);
                core.cpu.regs[0] = 0;
                return Ok(true);
            }
            // nvnQueuePresentTexture(NVNqueue*, NVNwindow*, int index): the
            // presented texture is the frame. With ECHORA_FRAME_DIR set, each
            // presented host image that differs from the last one written is
            // written there as a PPM (frame-<present count>.ppm); GPU frames
            // are sampled (ECHORA_FRAME_EVERY).
            "nvnQueuePresentTexture" => {
                self.object(this, "Queue").map_err(invalid)?;
                // The presented image needs the work queued before it.
                self.pump_queues(core)?;
                self.drain_queues(core)?;
                let window = self.object(x[1], "Window").map_err(invalid)?;
                let index = x[2] as u32;
                let texture = window
                    .state
                    .get(&format!("Texture{index}"))
                    .ok_or_else(|| invalid(format!("window texture {index} does not exist")))?
                    .x[0];
                if debug_var_os("ECHORA_GPU_FLOW").is_some() {
                    let note = format!("flow present queue {this:x} texture {texture:x}");
                    self.gpu_note(note);
                }
                self.presented_frames += 1;
                if self.gpu().is_some() {
                    self.gpu_present_window(texture).map_err(stop)?;
                    // A frame's work is submitted by its present, windowed or
                    // not: headless runs otherwise gathered hundreds of
                    // frames into one GPU job, which the kernel's 10 s job
                    // limit ended with a whole-GPU reset.
                    self.gpu_submit().map_err(stop)?;
                }
                // ECHORA_CAPTURE_EVERY=N with ECHORA_CAPTURE_DIR (echora run
                // sets it beside the report): the first and every N-th
                // presented image as frame-<present>.png, unchanged frames
                // included, so a run shows what was on screen. N=0 is off.
                if let (Ok(every), Some(directory)) = (
                    debug_var("ECHORA_CAPTURE_EVERY"),
                    debug_var_os("ECHORA_CAPTURE_DIR"),
                ) {
                    let every = every.parse::<u64>().unwrap_or(0);
                    if every != 0
                        && (self.presented_frames == 1 || self.presented_frames.is_multiple_of(every))
                    {
                        let image = if self.gpu().is_some() {
                            self.gpu_read_back(texture).map_err(stop)?
                        } else {
                            self.images.get(&texture).cloned()
                        };
                        if let Some(image) = image {
                            let directory = std::path::Path::new(&directory);
                            std::fs::create_dir_all(directory).map_err(|error| {
                                invalid(format!("create {}: {error}", directory.display()))
                            })?;
                            let path =
                                directory.join(format!("frame-{:05}.png", self.presented_frames));
                            std::fs::write(&path, image.png()).map_err(|error| {
                                invalid(format!("write {}: {error}", path.display()))
                            })?;
                        }
                    }
                }
                if self.gpu().is_some() {
                    self.gpu_dump_targets().map_err(stop)?;
                }
                if let Some(directory) = debug_var_os("ECHORA_FRAME_DIR") {
                    if self.gpu().is_some() {
                        self.log_cache_sizes();
                    }
                    // Reading back, encoding and hashing a frame costs more
                    // than rendering it: only every ECHORA_FRAME_EVERY-th
                    // present (default 60) and the first are written.
                    let every = debug_var("ECHORA_FRAME_EVERY")
                        .ok()
                        .and_then(|value| value.parse::<u64>().ok())
                        .unwrap_or(60)
                        .max(1);
                    if self.presented_frames != 1 && !self.presented_frames.is_multiple_of(every) {
                        return Ok(true);
                    }
                    if self.gpu().is_some() {
                        if let Some(image) = self.gpu_read_back(texture).map_err(stop)? {
                            self.images.insert(texture, image);
                        }
                    }
                    let image = self.images.get(&texture).ok_or_else(|| {
                        stop(format!("presented texture {texture:#x} was never rendered"))
                    })?;
                    let directory = std::path::Path::new(&directory);
                    std::fs::create_dir_all(directory).map_err(|error| {
                        invalid(format!("create {}: {error}", directory.display()))
                    })?;
                    let ppm = image.ppm();
                    let hash = {
                        use std::hash::{Hash, Hasher};
                        let mut hasher = std::collections::hash_map::DefaultHasher::new();
                        ppm.hash(&mut hasher);
                        hasher.finish()
                    };
                    if self.last_frame_hash == Some(hash) {
                        return Ok(true);
                    }
                    self.last_frame_hash = Some(hash);
                    let path = directory.join(format!("frame-{:05}.ppm", self.presented_frames));
                    std::fs::write(&path, ppm)
                        .map_err(|error| invalid(format!("write {}: {error}", path.display())))?;
                }
                return Ok(true);
            }
            // State-only setters on initialized objects: recorded for the
            // presenter and matching getters. nvnWindowSetCrop(window, int x,
            // int y, int width, int height) selects the presented rectangle;
            // nvnWindowSetPresentInterval(window, int) the vsync interval.
            "nvnWindowSetCrop"
            | "nvnWindowSetPresentInterval"
            // SetNumActiveTextures(window, int): how many of the window's
            // textures the application cycles through (its builder's
            // SetNumActiveTextures at run time); the host presents what is
            // presented, so it is recorded only.
            | "nvnWindowSetNumActiveTextures"
            | "nvnTextureSetDebugLabel"
            | "nvnBufferSetDebugLabel"
            | "nvnMemoryPoolSetDebugLabel"
            | "nvnQueueSetDebugLabel"
            | "nvnSyncSetDebugLabel"
            | "nvnWindowSetDebugLabel"
            | "nvnSamplerSetDebugLabel"
            | "nvnTexturePoolSetDebugLabel"
            | "nvnSamplerPoolSetDebugLabel" => {
                let (kind, rest) = split_name(name).unwrap();
                let property = rest.trim_start_matches("Set");
                let object = self
                    .objects
                    .get_mut(&this)
                    .filter(|object| object.kind == kind)
                    .ok_or_else(|| invalid(format!("{kind} is not initialized")))?;
                object.state.insert(property.into(), args);
                return Ok(true);
            }
            // nvnWindowGetCrop(const NVNwindow*, NVNrectangle* out): four ints
            // (x, y, width, height); the full first texture until SetCrop.
            "nvnWindowGetCrop" => {
                let window = self.object(this, "Window").map_err(invalid)?;
                let crop = match window.state.get("Crop") {
                    Some(crop) => [crop.x[0], crop.x[1], crop.x[2], crop.x[3]].map(|v| v as u32),
                    None => {
                        let texture = window.state["Texture0"].x[0];
                        let (width, height) = self.texture_extent(texture).map_err(stop)?;
                        [0, 0, width, height]
                    }
                };
                for (index, value) in crop.iter().enumerate() {
                    core.memory.write_u32(x[1] + index as u64 * 4, *value)?;
                }
                return Ok(true);
            }
            // nvnProgramInitialize(NVNprogram*, NVNdevice*) -> NVNboolean.
            "nvnProgramInitialize" => {
                self.object(x[1], "Device").map_err(invalid)?;
                self.initialize(this, "Program", BTreeMap::new())
                    .map_err(invalid)?;
                core.cpu.regs[0] = 1;
                return Ok(true);
            }
            // nvnProgramSetShaders(NVNprogram*, int count, const
            // NVNshaderData*) -> NVNboolean. Each 16-byte entry is the GPU
            // address of a compiled stage and a pointer to its control
            // section (inferred from the 16-byte stride at the call sites).
            // The stages are kept for shader translation; nothing is
            // compiled yet, so the program always links.
            // nvnProgramSetShadersExt: the same with 24-byte entries, whose
            // first two fields are SetShaders' and the third a further u64
            // the application copies from its shader record (main+0x12a4798).
            "nvnProgramSetShaders" | "nvnProgramSetShadersExt" => {
                self.object(this, "Program").map_err(invalid)?;
                let stride = if name.ends_with("Ext") { 24 } else { 16 };
                self.set_program_shaders(core, this, x[1], x[2], stride)?;
                core.cpu.regs[0] = 1;
                return Ok(true);
            }
            // nvnProgramInitializeFromBuilder(NVNprogram*, const
            // NVNprogramBuilder*) -> NVNboolean: Initialize with the
            // builder's device, then its SetShaders(count, data), read now
            // (the application calls it right after setting them).
            "nvnProgramInitializeFromBuilder" => {
                let builder = self.bag(x[1], "ProgramBuilder").map_err(invalid)?;
                self.require_device(&builder).map_err(invalid)?;
                let shaders = builder
                    .get("Shaders")
                    .copied()
                    .ok_or_else(|| stop("program builder has no shaders".into()))?;
                self.initialize(this, "Program", builder).map_err(invalid)?;
                self.set_program_shaders(core, this, shaders.x[0], shaders.x[1], 16)?;
                core.cpu.regs[0] = 1;
                return Ok(true);
            }
            "nvnProgramSetDebugLabel" => {
                let program = self
                    .objects
                    .get_mut(&this)
                    .filter(|object| object.kind == "Program")
                    .ok_or_else(|| invalid("program is not initialized".into()))?;
                program.state.insert("DebugLabel".into(), args);
                return Ok(true);
            }
            "nvnBufferFinalize"
            | "nvnProgramFinalize"
            | "nvnSyncFinalize"
            | "nvnQueueFinalize"
            | "nvnMemoryPoolFinalize" => {
                let kind = name.trim_start_matches("nvn").trim_end_matches("Finalize");
                // Destructors finalize embedded objects that may never have
                // been initialized (main+0x3b61ab4 finalizes two buffers of
                // every instance); the driver tolerates that, so it is a
                // counted no-op.
                if !self.objects.contains_key(&this) {
                    *self.skipped_work.entry(format!("{kind} finalized uninitialized")).or_default() += 1;
                    return Ok(true);
                }
                self.object(this, kind).map_err(invalid)?;
                if kind == "MemoryPool"
                    && self.objects.values().any(|object| {
                        object.kind == "Buffer" && object.builder["Storage"].x[0] == this
                    })
                {
                    return Err(stop("memory pool finalized while buffers use it".into()));
                }
                if kind == "MemoryPool" {
                    let base = self.objects[&this].builder.get("Storage").map_or(0, |storage| storage.x[0]);
                    self.gpu_forget_pool(base);
                }
                self.objects.remove(&this);
                return Ok(true);
            }
            _ => {}
        }
        // Calls on a recording command buffer are recorded for execution at
        // submission, with any data NVN reads through pointers captured now.
        if let Some(rest) = name.strip_prefix("nvnCommandBuffer") {
            if !COMMAND_BUFFER_MANAGEMENT.contains(&rest) && self.recording.contains_key(&this) {
                let mut command = Command {
                    name: name.into(),
                    x: std::array::from_fn(|index| x[index]),
                    v: std::array::from_fn(|index| core.cpu.vectors[index] as u64),
                    captured: Vec::new(),
                    states: Default::default(),
                };
                capture(&mut command, core)?;
                // CallCommands / CopyCommands(cmd, int count, const
                // NVNcommandHandle* handles): the handles are read now.
                if matches!(rest, "CallCommands" | "CopyCommands") {
                    let count = x[1] as u32;
                    if count > 4096 {
                        return Err(invalid(format!("{count} called command handles")));
                    }
                    for index in 0..u64::from(count) {
                        command
                            .captured
                            .push(core.memory.read_u64(x[2] + index * 8)?);
                    }
                }
                self.snapshot_states(&mut command);
                if command.name == "nvnCommandBufferCopyBufferToTexture" && command.x[3] != 0 {
                    let view = self.bag(command.x[3], "TextureView").map_err(invalid)?;
                    for (index, key) in ["Levels", "Layers"].iter().enumerate() {
                        command.captured[7 + index] = view.get(*key).map_or(0, |value| value.x[0]);
                    }
                }
                if command.name == "nvnCommandBufferClearTexture" {
                    let bases = if command.x[2] == 0 {
                        [0, 0]
                    } else {
                        let view = self.bag(command.x[2], "TextureView").map_err(invalid)?;
                        ["Levels", "Layers"].map(|key| view.get(key).map_or(0, |value| value.x[0]))
                    };
                    command.captured.extend(bases);
                }
                if command.name == "nvnCommandBufferCopyTextureToTexture" {
                    // After both regions: each view's level and layer base
                    // (0 without a view), read now as NVN does.
                    for view in [command.x[2], command.x[5]] {
                        let bases = if view == 0 {
                            [0, 0]
                        } else {
                            let view = self.bag(view, "TextureView").map_err(invalid)?;
                            ["Levels", "Layers"]
                                .map(|key| view.get(key).map_or(0, |value| value.x[0]))
                        };
                        command.captured.extend(bases);
                    }
                }
                if command.name == "nvnCommandBufferSetRenderTargets" {
                    // Views are resolved now (NVN reads them at record time)
                    // to the mip level (low 32 bits) and base layer (high
                    // 32 bits) they select (0 without a view). The depth
                    // view (X5) is appended after the color views.
                    let count = command.x[1] as usize;
                    let mut views: Vec<u64> = command.captured[count..].to_vec();
                    views.push(command.x[5]);
                    for (index, view) in views.iter_mut().enumerate() {
                        if *view != 0 {
                            let bag = self.bag(*view, "TextureView").map_err(invalid)?;
                            if let (Some(format), true) =
                                (bag.get("Format"), debug_var_os("ECHORA_GPU_FLOW").is_some())
                            {
                                let texture = command.captured.get(index).copied().unwrap_or(0);
                                let note = format!(
                                    "flow target view format {} on {texture:x}",
                                    format.x[0]
                                );
                                self.gpu_note(note);
                            }
                            let level = bag.get("Levels").map_or(0, |value| value.x[0]);
                            let layer = bag.get("Layers").map_or(0, |value| value.x[0]);
                            *view = level & 0xffff_ffff | layer << 32;
                        }
                    }
                    command.captured.truncate(count);
                    command.captured.extend(views);
                }
                self.recording.get_mut(&this).unwrap().push(command);
                self.use_command_memory(this, "CommandMemory", COMMAND_MEMORY_PER_COMMAND);
                return Ok(true);
            }
        }
        // Generic builder-style property bags: SetDefaults / Set<Prop> /
        // Get<Prop> on structures the application only reaches through these
        // calls. Getters return what the matching setter stored; a property
        // that was never set stops explicitly (defaults are not modeled).
        let Some((kind, rest)) = split_name(name) else {
            return Ok(false);
        };
        // Sampler getters that write through output pointers (X1.. in
        // order): nvnSampler[Builder]GetLodClamp(s, float* min, float* max),
        // GetMinMagFilter(s, int* min, int* mag), GetWrapMode(s, int* s,
        // int* t, int* r) and GetCompare(s, int* mode, int* function). They
        // return what the matching setter stored (floats in V0/V1).
        if matches!(kind, "Sampler" | "SamplerBuilder") {
            // SetBorderColor[i|ui](builder, const T rgba[4]): the four words
            // are captured, since the application may reuse the array.
            // Get* writes them to the array at X1 (default transparent
            // black, OpenGL's).
            if let Some(suffix) = rest
                .strip_prefix("SetBorderColor")
                .filter(|_| kind == "SamplerBuilder")
            {
                let mut words = [0u64; 4];
                for (index, word) in words.iter_mut().enumerate() {
                    *word = u64::from(core.memory.read_u32(x[1] + index as u64 * 4)?);
                }
                let bag = self.bags.entry(this).or_insert_with(|| Bag {
                    kind: kind.into(),
                    props: BTreeMap::new(),
                });
                if bag.kind != kind {
                    return Err(invalid(format!("no SamplerBuilder at {this:#x}")));
                }
                bag.props.insert(
                    format!("BorderColor{suffix}"),
                    Args {
                        x: words,
                        v: [0; 2],
                    },
                );
                return Ok(true);
            }
            if let Some(suffix) = rest.strip_prefix("GetBorderColor") {
                let props = if kind == "Sampler" {
                    self.object(this, "Sampler")
                        .map_err(invalid)?
                        .builder
                        .clone()
                } else {
                    self.bag(this, "SamplerBuilder").map_err(invalid)?
                };
                let words = props
                    .get(&format!("BorderColor{suffix}"))
                    .map_or([0; 4], |value| value.x);
                for (index, word) in words.iter().enumerate() {
                    core.memory
                        .write_u32(x[1] + index as u64 * 4, *word as u32)?;
                }
                return Ok(true);
            }
            let outputs = match rest {
                "GetLodClamp" => Some((2, true)),
                "GetMinMagFilter" | "GetCompare" => Some((2, false)),
                "GetWrapMode" => Some((3, false)),
                _ => None,
            };
            if let Some((count, floats)) = outputs {
                let property = &rest[3..];
                let props = if kind == "Sampler" {
                    self.object(this, "Sampler")
                        .map_err(invalid)?
                        .builder
                        .clone()
                } else {
                    self.bag(this, "SamplerBuilder").map_err(invalid)?
                };
                let Some(value) = props.get(property).copied().or(sampler_default(property)) else {
                    return Err(stop(format!("{property} was never set")));
                };
                for index in 0..count {
                    let out = x[1 + index];
                    if out == 0 {
                        return Err(invalid("null output".into()));
                    }
                    let word = if floats {
                        value.v[index] as u32
                    } else {
                        value.x[index] as u32
                    };
                    core.memory.write_u32(out, word)?;
                }
                return Ok(true);
            }
        }
        // Fixed-function state objects (nvn<Kind>State) are plain structures
        // the application configures with SetDefaults/Set and binds by
        // pointer, like builders.
        let bag_like =
            kind.ends_with("Builder") || kind == "TextureView" || kind.ends_with("State");
        if bag_like && rest == "SetDefaults" {
            // Builders keep their device across SetDefaults: the application
            // sets the device first for some kinds (QueueBuilder).
            let device = self
                .bags
                .get(&this)
                .and_then(|bag| bag.props.get("Device").copied());
            let mut bag = Bag {
                kind: kind.into(),
                props: BTreeMap::new(),
            };
            if let Some(device) = device {
                bag.props.insert("Device".into(), device);
            }
            self.bags.insert(this, bag);
            return Ok(true);
        }
        if bag_like {
            if let Some(property) = rest.strip_prefix("Set") {
                // Structures live in application memory (often the stack):
                // a setter of another kind at the same address means the old
                // structure's lifetime ended, so a new one starts there.
                let bag = self
                    .bags
                    .entry(this)
                    .and_modify(|bag| {
                        if bag.kind != kind {
                            *bag = Bag {
                                kind: kind.into(),
                                props: BTreeMap::new(),
                            };
                        }
                    })
                    .or_insert_with(|| Bag {
                        kind: kind.into(),
                        props: BTreeMap::new(),
                    });
                // SetSize1D/2D/3D(builder, width[, height[, depth]]) are the
                // combined forms of SetWidth/SetHeight/SetDepth.
                if let Some(dimensions) = property
                    .strip_prefix("Size")
                    .and_then(|rest| rest.strip_suffix('D'))
                    .and_then(|count| count.parse::<usize>().ok())
                    .filter(|count| (1..=3).contains(count))
                {
                    for (index, key) in ["Width", "Height", "Depth"].iter().enumerate() {
                        let value = if index < dimensions { args.x[index] } else { 1 };
                        bag.props.insert(
                            (*key).into(),
                            Args {
                                x: [value, 0, 0, 0],
                                v: [0; 2],
                            },
                        );
                    }
                    return Ok(true);
                }
                // Per-target setters (SetBlendEnable(state, index, enable),
                // SetChannelMask(state, index, r, g, b, a)) also keep each
                // index's latest call.
                if matches!(property, "BlendEnable" | "ChannelMask") {
                    let mut args = args;
                    // The mask's alpha flag is the fifth argument (X5),
                    // kept where integer setters leave V1 unused.
                    if property == "ChannelMask" {
                        args.v[1] = x[5];
                    }
                    bag.props.insert(format!("{property}/{}", args.x[0]), args);
                }
                bag.props.insert(property.into(), args);
                return Ok(true);
            }
            if let Some(property) = rest.strip_prefix("Get") {
                let bag = self
                    .bags
                    .get(&this)
                    .filter(|bag| bag.kind == kind)
                    .ok_or_else(|| invalid(format!("no {kind} at {this:#x}")))?;
                let Some(value) = bag.props.get(property).copied() else {
                    return Err(stop(format!("{property} was never set")));
                };
                core.cpu.regs[0] = value.x[0];
                core.cpu.vectors[0] = u128::from(value.v[0]);
                return Ok(true);
            }
        }
        // Generic getters on initialized objects for properties their
        // builder set (nvn<Kind>Get<Prop> after nvn<Kind>BuilderSet<Prop>).
        if let (Some(property), Some(object)) = (rest.strip_prefix("Get"), self.objects.get(&this))
        {
            if object.kind == kind {
                let default = match kind {
                    "Sampler" => sampler_default(property),
                    "Texture" => texture_default(property),
                    _ => None,
                };
                if let Some(value) = object
                    .state
                    .get(property)
                    .or(object.builder.get(property))
                    .or(default.as_ref())
                {
                    core.cpu.regs[0] = value.x[0];
                    core.cpu.vectors[0] = u128::from(value.v[0]);
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    /// Run queued work on every queue until none can advance: a queue stops
    /// at a wait that does not hold, and work another queue runs (a
    /// SignalEvent, a fence) may let it continue.
    fn pump_queues(&mut self, core: &mut Guest) -> Result<(), Trap> {
        loop {
            let mut progressed = false;
            let queues: Vec<u64> = self.pending.keys().copied().collect();
            for queue in queues {
                progressed |= self.run_queue(queue, core, false)?;
            }
            if !progressed {
                self.pending.retain(|_, work| !work.is_empty());
                return Ok(());
            }
        }
    }

    /// Run all queued work, waits that do not hold included (counted in
    /// `forced_waits`): the CPU needs the results now.
    fn drain_queues(&mut self, core: &mut Guest) -> Result<(), Trap> {
        while let Some(&queue) = self
            .pending
            .iter()
            .find(|(_, work)| !work.is_empty())
            .map(|(queue, _)| queue)
        {
            // Queues that can run first, so the forced wait is the last
            // resort.
            self.pump_queues(core)?;
            if self
                .pending
                .get(&queue)
                .is_some_and(|work| !work.is_empty())
            {
                self.run_queue(queue, core, true)?;
            }
        }
        self.pending.clear();
        Ok(())
    }

    /// Whether `sync` has a fence queued behind blocked work.
    fn sync_fence_pending(&self, sync: u64) -> bool {
        self.pending
            .values()
            .flatten()
            .any(|item| matches!(item, QueueItem::Fence { sync: fenced, .. } if *fenced == sync))
    }

    /// Whether a recorded WaitEvent's condition holds now.
    fn event_wait_holds(&self, command: &Command, core: &Guest) -> Result<bool, Trap> {
        let address = self
            .event_address(command.x[1])
            .map_err(|message| Trap::InvalidState {
                message: format!("WaitEvent: {message}"),
            })?;
        let value = core.memory.read_u32(address)?;
        let target = command.x[3] as u32;
        Ok(match command.x[2] as u32 {
            1 => value >= target,
            _ => value == target,
        })
    }

    /// Run `queue`'s work in order until it is done or blocked (`force`
    /// runs through waits). Returns whether anything ran.
    fn run_queue(&mut self, queue: u64, core: &mut Guest, force: bool) -> Result<bool, Trap> {
        let mut progressed = false;
        loop {
            let Some(item) = self
                .pending
                .get_mut(&queue)
                .and_then(|work| work.front_mut())
            else {
                return Ok(progressed);
            };
            match item {
                QueueItem::List { commands, next } => {
                    if commands.get(*next).is_none() {
                        self.pending.get_mut(&queue).unwrap().pop_front();
                        continue;
                    }
                    let commands = commands.clone();
                    let index = *next;
                    let command = &commands[index];
                    let rest = command.name.trim_start_matches("nvnCommandBuffer");
                    if rest == "WaitEvent" && !self.event_wait_holds(command, core)? {
                        if !force {
                            return Ok(progressed);
                        }
                        *self.forced_waits.entry("WaitEvent".into()).or_default() += 1;
                        *self
                            .skipped_work
                            .entry("WaitEvent unsatisfied".into())
                            .or_default() += 1;
                    }
                    if let QueueItem::List { next, .. } =
                        self.pending.get_mut(&queue).unwrap().front_mut().unwrap()
                    {
                        *next += 1;
                    }
                    progressed = true;
                    if rest == "WaitEvent" {
                        continue;
                    }
                    if matches!(rest, "CallCommands" | "CopyCommands") {
                        // The called lists run next, in order, ahead of
                        // the rest of this one (their waits block too).
                        let work = self.pending.get_mut(&queue).unwrap();
                        for handle in command.captured.iter().rev() {
                            match self.command_lists.get(handle) {
                                Some((list, ..)) => work.push_front(QueueItem::List {
                                    commands: list.clone(),
                                    next: 0,
                                }),
                                None => self.stale_command_submissions += 1,
                            }
                        }
                        continue;
                    }
                    self.execute(command, core)?;
                }
                QueueItem::Fence { sync, args } => {
                    let (sync, args) = (*sync, *args);
                    self.pending.get_mut(&queue).unwrap().pop_front();
                    if let Some(object) = self.objects.get_mut(&sync) {
                        object.state.insert("Signaled".into(), args);
                    }
                    progressed = true;
                }
                QueueItem::WaitSync { sync } => {
                    let sync = *sync;
                    let signaled = self
                        .objects
                        .get(&sync)
                        .is_some_and(|object| object.state.contains_key("Signaled"));
                    if !signaled && !self.sync_fence_pending(sync) {
                        // No queue will signal it: waiting would never end.
                        return Err(Trap::UnsupportedService {
                            pc: core.cpu.pc,
                            service: "nvn".into(),
                            command_id: None,
                            message: "queue waits on a sync that is never signaled".into(),
                        });
                    }
                    if !signaled && !force {
                        return Ok(progressed);
                    }
                    if !signaled {
                        *self.forced_waits.entry("WaitSync".into()).or_default() += 1;
                    }
                    self.pending.get_mut(&queue).unwrap().pop_front();
                    progressed = true;
                }
            }
        }
    }

    /// Execute one recorded command on the host. Commands that are not
    /// modeled stop with their name.
    fn execute(&mut self, command: &Command, core: &mut Guest) -> Result<(), Trap> {
        // A guest memory fault names the command that made it.
        self.execute_command(command, core)
            .map_err(|trap| match trap {
                Trap::MemoryFault { address, access } => Trap::UnsupportedService {
                    pc: core.cpu.pc,
                    service: "nvn".into(),
                    command_id: None,
                    message: format!(
                        "executing {}: {access} fault at {address:#x} (x {:x?})",
                        command.name,
                        &command.x[..5]
                    ),
                },
                other => other,
            })
    }

    fn execute_command(&mut self, command: &Command, core: &mut Guest) -> Result<(), Trap> {
        // Commands that read or write guest memory themselves see earlier
        // dispatches' results in place (draws and dispatches resolve what
        // they read).
        if matches!(
            command.name.as_str(),
            "nvnCommandBufferCopyBufferToTexture"
                | "nvnCommandBufferCopyTextureToTexture"
                | "nvnCommandBufferReportCounter"
                | "nvnCommandBufferClearBuffer"
        ) {
            if let Err(message) = self.resolve_storage(core, None) {
                return Err(Trap::UnsupportedService {
                    pc: core.cpu.pc,
                    service: "nvn".into(),
                    command_id: None,
                    message: format!("executing {}: {message}", command.name),
                });
            }
        }
        let pc = core.cpu.pc;
        let stop = |message: String| Trap::UnsupportedService {
            pc,
            service: "nvn".into(),
            command_id: None,
            message: format!("executing {}: {message}", command.name),
        };
        match command.name.as_str() {
            // Push{Graphics,Compute}Constants: words into root table `table`
            // at `offset` (bytes, word aligned, within the 256-byte table).
            "nvnCommandBufferPushGraphicsConstants" | "nvnCommandBufferPushComputeConstants" => {
                let pipeline = usize::from(command.name.ends_with("ComputeConstants"));
                let table = (command.x[1] & 7) as usize;
                let offset = command.x[2] as usize;
                if offset % 4 != 0 || offset + command.captured.len() * 4 > 0x100 {
                    self.gpu_note(format!(
                        "push constants at {offset:#x} outside the root table"
                    ));
                    return Ok(());
                }
                let roots = &mut self.render.roots[pipeline];
                if roots.len() < 8 {
                    roots.resize(8, [0; 64]);
                }
                for (index, word) in command.captured.iter().enumerate() {
                    roots[table][offset / 4 + index] = *word as u32;
                }
                Ok(())
            }
            // CallCommands / CopyCommands: the called lists run in place, in
            // order (a copy runs the same commands when it is submitted).
            // Lists no longer kept are skipped and counted, as at
            // submission.
            "nvnCommandBufferCallCommands" | "nvnCommandBufferCopyCommands" => {
                if self.call_depth >= 8 {
                    return Err(stop("command lists call each other too deeply".into()));
                }
                for handle in command.captured.clone() {
                    let Some((list, ..)) = self.command_lists.get(&handle) else {
                        self.stale_command_submissions += 1;
                        continue;
                    };
                    let list = list.clone();
                    self.call_depth += 1;
                    let result = list
                        .iter()
                        .try_for_each(|called| self.execute(called, core));
                    self.call_depth -= 1;
                    result?;
                }
                Ok(())
            }
            // SetRenderTargets(cmd, int colors, const NVNtexture* const*,
            // const NVNtextureView* const*, depth, depthView). Captured:
            // color texture pointers, then color view pointers.
            "nvnCommandBufferSetRenderTargets" => {
                let count = command.x[1] as usize;
                let colors = command.captured[..count].to_vec();
                let views = &command.captured[count..];
                // Each color target's mip level and layer, resolved at
                // record time; the depth view's follows the colors'.
                self.render.color_levels = (0..count)
                    .map(|index| views.get(index).copied().unwrap_or(0))
                    .collect();
                self.render.depth_layer = views.get(count).map_or(0, |view| (view >> 32) as u32);
                // The depth-stencil target (X4) is validated and kept; depth
                // testing needs draws, which are not rendered yet.
                if command.x[4] != 0 {
                    self.object(command.x[4], "Texture").map_err(stop)?;
                }
                self.render.depth = command.x[4];
                for texture in &colors {
                    if *texture != 0 {
                        self.object(*texture, "Texture").map_err(stop)?;
                    }
                }
                self.render.colors = colors;
                Ok(())
            }
            // SetViewports(cmd, first, count, const float* {x, y, w, h}...)
            // and SetScissors(cmd, first, count, const int* {x, y, w, h}...)
            // (GL-style, inferred): the first array entry, read at record
            // time, sets viewport (or scissor) 0, as SetViewport and
            // SetScissor do.
            "nvnCommandBufferSetViewports" | "nvnCommandBufferSetScissors" => {
                if command.x[1] != 0 || command.captured.len() < 4 {
                    return Ok(());
                }
                let rect: [u64; 4] = std::array::from_fn(|index| command.captured[index]);
                if command.name.ends_with("Viewports") {
                    let mut viewport = command.clone();
                    viewport.name = "nvnCommandBufferSetViewport".into();
                    for (index, value) in rect.iter().enumerate() {
                        viewport.x[1 + index] = f32::from_bits(*value as u32) as i32 as u64;
                    }
                    self.render.state.insert("SetViewport".into(), viewport);
                } else {
                    self.render.scissor = Some(rect.map(|value| value as u32 as i32));
                }
                Ok(())
            }
            // SetDepthRanges(cmd, first, count, const float* {near, far}...)
            // (inferred like SetViewports): the first entry, read at record
            // time, sets viewport 0's window depth range.
            "nvnCommandBufferSetDepthRanges" => {
                if command.x[1] != 0 || command.captured.len() < 2 {
                    return Ok(());
                }
                let mut range = command.clone();
                range.name = "nvnCommandBufferSetDepthRange".into();
                range.v[0] = command.captured[0];
                range.v[1] = command.captured[1];
                self.render.state.insert("SetDepthRange".into(), range);
                Ok(())
            }
            // SetScissor(cmd, x, y, width, height).
            "nvnCommandBufferSetScissor" => {
                self.render.scissor = Some([
                    command.x[1] as i32,
                    command.x[2] as i32,
                    command.x[3] as i32,
                    command.x[4] as i32,
                ]);
                Ok(())
            }
            // ClearTexture: as ClearColor on the texture's level and layer
            // (the view's bases), whole level; a region smaller than the
            // level and partial masks are noted, not modeled.
            "nvnCommandBufferClearTexture" => {
                let texture = command.x[1];
                self.object(texture, "Texture").map_err(stop)?;
                let captured = &command.captured;
                let (level, layer) = (
                    captured[captured.len() - 2] as u32,
                    captured[captured.len() - 1] as u32,
                );
                let color: [u32; 4] = std::array::from_fn(|channel| captured[channel] as u32);
                if command.x[5] as u32 & 0xf != 0xf {
                    self.gpu_note("masked texture clears are not modeled");
                }
                if captured.len() == 12 {
                    let (width, height) = self.texture_extent(texture).map_err(stop)?;
                    let region = &captured[4..10];
                    let whole = region[0] == 0
                        && region[1] == 0
                        && region[3] as u32 >= width >> level
                        && region[4] as u32 >= height >> level;
                    if !whole {
                        self.gpu_note(
                            "partial texture clears are not modeled (whole level cleared)",
                        );
                    }
                }
                if level != 0 && debug_var_os("ECHORA_GPU_MIPS").is_none() {
                    return Ok(());
                }
                if self.gpu().is_some() {
                    return self
                        .gpu_clear_color(texture, level, layer, color)
                        .map_err(stop);
                }
                self.gpu_note("texture clears without the GPU are not modeled");
                Ok(())
            }
            // ClearColor(cmd, int index, const float color[4], int mask):
            // fills the bound color target (inferred GL semantics: the
            // scissor rectangle limits the clear; mask bit i enables channel
            // i).
            "nvnCommandBufferClearColor" => {
                let index = command.x[1] as usize;
                let texture = *self
                    .render
                    .colors
                    .get(index)
                    .filter(|texture| **texture != 0)
                    .ok_or_else(|| stop(format!("no color target {index}")))?;
                let view = self.render.color_levels.get(index).copied().unwrap_or(0);
                let (level, layer) = (view & 0xffff_ffff, (view >> 32) as u32);
                if debug_var_os("ECHORA_GPU_FLOW").is_some() {
                    let note = format!(
                        "flow clear {texture:x} index {index} level {level} layer {layer} mask {:#x} scissor {:?} color {:?}",
                        command.x[3],
                        self.render.scissor,
                        command.captured[..command.captured.len().min(4)]
                            .iter()
                            .map(|word| f32::from_bits(*word as u32))
                            .collect::<Vec<_>>()
                    );
                    self.gpu_note(note);
                }
                if level != 0 && debug_var_os("ECHORA_GPU_MIPS").is_none() {
                    // Without mip-level targets, only level 0 exists.
                    return Ok(());
                }
                let mask = command.x[3] as u32;
                if self.gpu().is_some() {
                    if mask & 0xf != 0xf {
                        self.gpu_note("masked color clears are not modeled on the GPU");
                    }
                    let color = std::array::from_fn(|channel| command.captured[channel] as u32);
                    return self
                        .gpu_clear_color(texture, level as u32, layer, color)
                        .map_err(stop);
                }
                let (width, height) = self.texture_extent(texture).map_err(stop)?;
                let color: [f32; 4] =
                    std::array::from_fn(|channel| f32::from_bits(command.captured[channel] as u32));
                let scissor = self
                    .render
                    .scissor
                    .unwrap_or([0, 0, width as i32, height as i32]);
                let image = self.images.entry(texture).or_insert_with(|| Image {
                    width,
                    height,
                    texels: vec![[0.0; 4]; width as usize * height as usize],
                });
                let x0 = scissor[0].clamp(0, width as i32) as u32;
                let y0 = scissor[1].clamp(0, height as i32) as u32;
                let x1 = (scissor[0] + scissor[2]).clamp(0, width as i32) as u32;
                let y1 = (scissor[1] + scissor[3]).clamp(0, height as i32) as u32;
                for y in y0..y1 {
                    for x in x0..x1 {
                        let texel = &mut image.texels[(y * width + x) as usize];
                        for channel in 0..4 {
                            if mask & (1 << channel) != 0 {
                                texel[channel] = color[channel];
                            }
                        }
                    }
                }
                Ok(())
            }
            // CopyBufferToBuffer(cmd, NVNbufferAddress src, NVNbufferAddress
            // dst, size, int flags): a byte copy between pool memory.
            "nvnCommandBufferCopyBufferToBuffer" => {
                let (source, destination, size) = (command.x[1], command.x[2], command.x[3]);
                if size > 1 << 32 {
                    return Err(stop(format!("{size:#x}-byte copy")));
                }
                let from = self.gpu_to_guest(source, size).map_err(stop)?;
                let to = self.gpu_to_guest(destination, size).map_err(stop)?;
                let mut bytes = vec![0u8; size as usize];
                core.memory.read(from, &mut bytes)?;
                core.memory.write(to, &bytes)?;
                Ok(())
            }
            // FenceSync(cmd, NVNsync*, condition, flags) in a command buffer:
            // the sync signals when execution reaches it, which on the host is
            // now. WaitSync(cmd, const NVNsync*): later commands wait for it;
            // an unsignaled sync would block the queue forever, so it stops.
            "nvnCommandBufferFenceSync" => {
                let sync = self
                    .objects
                    .get_mut(&command.x[1])
                    .filter(|object| object.kind == "Sync")
                    .ok_or_else(|| stop("fenced sync is not initialized".into()))?;
                let mut args = Args::default();
                args.x[..3].copy_from_slice(&command.x[1..4]);
                sync.state.insert("Signaled".into(), args);
                Ok(())
            }
            "nvnCommandBufferWaitSync" => {
                let sync = self.object(command.x[1], "Sync").map_err(stop)?;
                if !sync.state.contains_key("Signaled") {
                    return Err(stop(
                        "command buffer waits on a sync that is never signaled".into(),
                    ));
                }
                Ok(())
            }
            // ClearDepthStencil(cmd, float depth, NVNboolean depthMask, int
            // stencil, int stencilMask) on the bound depth target. No depth
            // buffer is read until draws are rendered, so the cleared values
            // are recorded as the depth target's contents.
            "nvnCommandBufferClearDepthStencil" => {
                if self.render.depth == 0 {
                    return Err(stop("no depth-stencil target is bound".into()));
                }
                self.render
                    .state
                    .insert("ClearDepthStencil".into(), command.clone());
                if self.gpu().is_some() {
                    // The depth value is the first float argument (V0).
                    let depth = f32::from_bits(command.v[0] as u32);
                    let (depth_mask, stencil, stencil_mask) =
                        (command.x[1] != 0, command.x[2] as u32, command.x[3] as u32);
                    let texture = self.render.depth;
                    if debug_var_os("ECHORA_GPU_FLOW").is_some() {
                        let note = format!(
                            "flow clear depth {texture:x} value {depth} write {depth_mask} stencil {stencil} mask {stencil_mask:#x}"
                        );
                        self.gpu_note(note);
                    }
                    return self
                        .gpu_clear_depth(
                            texture,
                            depth_mask.then_some(depth),
                            (stencil_mask != 0).then_some(stencil),
                        )
                        .map_err(stop);
                }
                Ok(())
            }
            // CopyBufferToTexture(cmd, NVNbufferAddress src, const NVNtexture*
            // dst, const NVNtextureView*, const NVNcopyRegion*, int flags):
            // packed texels from pool memory into the texture (as
            // WriteTexels). Captured: the region, then whether a view was
            // given and the view's level and layer bases.
            "nvnCommandBufferCopyBufferToTexture" => {
                let region: [u32; 6] = std::array::from_fn(|index| command.captured[index] as u32);
                let mut view = BTreeMap::new();
                if command.captured[6] != 0 {
                    for (key, base) in [
                        ("Levels", command.captured[7]),
                        ("Layers", command.captured[8]),
                    ] {
                        view.insert(
                            key.to_string(),
                            Args {
                                x: [base, 0, 0, 0],
                                v: [0; 2],
                            },
                        );
                    }
                }
                let size = {
                    let texture = self.object(command.x[2], "Texture").map_err(stop)?;
                    let format = texture
                        .builder
                        .get("Format")
                        .map_or(0, |value| value.x[0] as u32);
                    let info = crate::nvn_format::info(format)
                        .ok_or_else(|| stop(format!("texture format {format} is not modeled")))?;
                    u64::from(region[3].div_ceil(info.block_width))
                        * u64::from(region[4].div_ceil(info.block_height))
                        * u64::from(region[5])
                        * u64::from(info.bytes_per_block)
                };
                let source = self.gpu_to_guest(command.x[1], size).map_err(stop)?;
                if debug_var_os("ECHORA_GPU_FLOW").is_some() {
                    let mut sample = vec![0u8; size.min(1 << 20) as usize];
                    core.memory.read(source, &mut sample)?;
                    let nonzero = sample.iter().any(|byte| *byte != 0);
                    self.gpu_note(format!(
                        "copy to texture {:x}: source {}",
                        command.x[2],
                        if nonzero { "has data" } else { "is zero" }
                    ));
                }
                let view = (command.captured[6] != 0).then_some(&view);
                self.copy_texels(core, command.x[2], view, region, source, true)
                    .map_err(|error| error.into_trap(&command.name, core.cpu.pc))?;
                // As WriteTexels: the texture's contents changed (new glyphs
                // in a font atlas, say), so its cached upload is dropped.
                self.gpu_invalidate(0..0, Some(command.x[2]));
                Ok(())
            }
            // ReportCounter(cmd, NVNcounterType, NVNbufferAddress): the GPU
            // writes a 16-byte report, a u64 counter then a u64 GPU
            // timestamp (NVNcounterData; layout inferred from NVN1). Counts
            // are not modeled (0); the timestamp is the physical counter,
            // the domain nvnDeviceGetCurrentTimestampInNanoseconds converts.
            "nvnCommandBufferReportCounter" => {
                if debug_var_os("ECHORA_GPU_FLOW").is_some() {
                    self.gpu_note(format!("counter report of type {}", command.x[1]));
                }
                let to = self.gpu_to_guest(command.x[2], 16).map_err(stop)?;
                let timestamp = core.counter;
                let mut report = [0u8; 16];
                report[8..].copy_from_slice(&timestamp.to_le_bytes());
                core.memory.write(to, &report)?;
                Ok(())
            }
            // ResetCounter(cmd, NVNcounterType): sets that counter to 0. The
            // counts ReportCounter writes are not modeled and always 0, so
            // there is nothing to reset.
            "nvnCommandBufferResetCounter" => Ok(()),
            // ClearBuffer(cmd, NVNbufferAddress, size, u32 value): fills with
            // the 32-bit value (size is a multiple of 4).
            "nvnCommandBufferClearBuffer" => {
                let (address, size, value) = (command.x[1], command.x[2], command.x[3] as u32);
                if size % 4 != 0 || size > 1 << 32 {
                    return Err(stop(format!("{size:#x}-byte clear")));
                }
                let to = self.gpu_to_guest(address, size).map_err(stop)?;
                let bytes: Vec<u8> = std::iter::repeat_n(value.to_le_bytes(), size as usize / 4)
                    .flatten()
                    .collect();
                core.memory.write(to, &bytes)?;
                Ok(())
            }
            name => {
                let rest = name.trim_start_matches("nvnCommandBuffer");
                if is_state_command(rest) {
                    // Per-slot bindings are kept per stage and slot; the
                    // others keep their latest command.
                    let key = match rest {
                        "BindTexture"
                        | "BindUniformBuffer"
                        | "BindStorageBuffer"
                        | "BindImage"
                        | "BindSeparateTexture"
                        | "BindSeparateSampler" => {
                            format!("{rest}/{}/{}", command.x[1], command.x[2])
                        }
                        "BindVertexBuffer" => format!("{rest}/{}", command.x[1]),
                        // A blend state holds one target's factors (its
                        // BlendTarget, read at record time): each target
                        // keeps the latest state bound for it. The decal
                        // pass binds one per G-buffer target; keeping only
                        // the last left target 0 overwritten, not blended.
                        "BindBlendState" => {
                            let target = command
                                .states
                                .first()
                                .and_then(|state| state.get("BlendTarget"))
                                .map_or(0, |value| value.x[0]);
                            format!("{rest}/{target}")
                        }
                        // BindProgram(cmd, program, stage bits): a program
                        // replaces the stages its bits name (inferred).
                        "BindProgram" => {
                            for bit in 0..8 {
                                if command.x[2] >> bit & 1 != 0 {
                                    self.render
                                        .state
                                        .insert(format!("{rest}@{bit}"), command.clone());
                                }
                            }
                            return Ok(());
                        }
                        // Another pool: every handle may name other entries.
                        "SetTexturePool" | "SetSamplerPool" => {
                            self.binding_serial += 1;
                            rest.into()
                        }
                        _ => rest.into(),
                    };
                    self.render.state.insert(key, command.clone());
                    return Ok(());
                }
                if rest.starts_with("Draw") || rest.starts_with("Dispatch") {
                    // Opt-in only: a line per draw grew to gigabytes and
                    // took most of the run time in long runs.
                    // ECHORA_DRAW_LOG_FROM=N starts at present N.
                    if let Some(path) = debug_var_os("ECHORA_DRAW_LOG") {
                        let from = debug_var("ECHORA_DRAW_LOG_FROM")
                            .ok()
                            .and_then(|value| value.parse::<u64>().ok())
                            .unwrap_or(0);
                        if self.presented_frames >= from {
                            self.log_draw(core, command, path);
                        }
                    }
                }
                if rest == "CopyTextureToTexture" && self.gpu().is_some() {
                    if let Err(error) = self.gpu_copy_texture(core, command) {
                        self.gpu_note(format!("texture copy skipped: {error}"));
                        *self.skipped_work.entry(rest.into()).or_default() += 1;
                    }
                    return Ok(());
                }
                // DispatchComputeIndirect(cmd, NVNbufferAddress): the group
                // counts are three u32 at the address when the GPU reaches
                // the command (a dispatch before it may have written them,
                // so its results are resolved first); it runs as
                // DispatchCompute with them.
                if rest == "DispatchComputeIndirect" {
                    if let Err(message) = self.resolve_storage(core, None) {
                        return Err(stop(format!("dispatch results: {message}")));
                    }
                    let at = self.gpu_to_guest(command.x[1], 12).map_err(stop)?;
                    let mut direct = command.clone();
                    direct.name = "nvnCommandBufferDispatchCompute".into();
                    for (index, slot) in direct.x[1..4].iter_mut().enumerate() {
                        *slot = u64::from(core.memory.read_u32(at + index as u64 * 4)?);
                    }
                    return self.execute_command(&direct, core);
                }
                if rest == "DispatchCompute" && self.gpu().is_some() {
                    let start = std::time::Instant::now();
                    let result = self.gpu_dispatch(core, command);
                    let backend = self.gpu().unwrap();
                    backend.stat("stat: dispatches", 1);
                    backend.stat(
                        "stat: dispatch microseconds",
                        start.elapsed().as_micros() as u64,
                    );
                    if let Err(error) = result {
                        self.gpu_note(format!("dispatch skipped: {error}"));
                    }
                    return Ok(());
                }
                if rest.starts_with("Draw") && self.gpu().is_some() {
                    // GPU execution is best effort: a draw it cannot run is
                    // counted by reason and skipped.
                    let start = std::time::Instant::now();
                    let result = self.gpu_draw(core, command);
                    let backend = self.gpu().unwrap();
                    backend.stat("stat: draws", 1);
                    backend.stat(
                        "stat: draw microseconds",
                        start.elapsed().as_micros() as u64,
                    );
                    if let Err(error) = result {
                        self.gpu_note(format!("draw skipped: {error}"));
                    }
                    return Ok(());
                }
                // WaitEvent(event, mode, u32 value): the GPU waits until the
                // event satisfies the comparison (the application waits mode
                // 0 for 1 on an event it set to 0, frame pacing). Queues run
                // at submission and cannot stall, so the wait is evaluated
                // (mode 0 as equal, 1 as at least, inferred) and execution
                // continues; unsatisfied waits are counted in the report.
                // SignalEvent(event, mode, location, flags, u32 value): the
                // GPU writes (mode 0) or adds (1) the value when it reaches
                // the command; location (which pipeline stage) has no host
                // meaning since earlier commands have already run.
                // UpdateUniformBuffer: the data captured at record time is
                // written into the buffer when the command runs.
                if rest == "UpdateUniformBuffer" {
                    let (offset, size) = (command.x[3], command.x[4]);
                    let guest = self
                        .gpu_to_guest(command.x[1] + offset, size)
                        .map_err(stop)?;
                    let bytes: Vec<u8> = command
                        .captured
                        .iter()
                        .flat_map(|word| word.to_le_bytes())
                        .take(size as usize)
                        .collect();
                    core.memory.write(guest, &bytes)?;
                    self.gpu_invalidate(guest..guest + size, None);
                    return Ok(());
                }
                if rest == "SignalEvent" {
                    let address = self.event_address(command.x[1]).map_err(stop)?;
                    let value = command.x[5] as u32;
                    let new = match command.x[2] as u32 {
                        0 => value,
                        1 => core.memory.read_u32(address)?.wrapping_add(value),
                        mode => {
                            return Err(stop(format!("event signal mode {mode} is not modeled")));
                        }
                    };
                    core.memory.write_u32(address, new)?;
                    return Ok(());
                }
                if rest == "WaitEvent" {
                    let address = self.event_address(command.x[1]).map_err(stop)?;
                    let (mode, target) = (command.x[2] as u32, command.x[3] as u32);
                    let value = core.memory.read_u32(address)?;
                    let satisfied = match mode {
                        0 => value == target,
                        1 => value >= target,
                        _ => return Err(stop(format!("event wait mode {mode} is not modeled"))),
                    };
                    if !satisfied {
                        *self
                            .skipped_work
                            .entry("WaitEvent unsatisfied".into())
                            .or_default() += 1;
                    }
                    return Ok(());
                }
                if is_work_command(rest) && debug_var_os("ECHORA_NVN_SKIP_WORK").is_some() {
                    *self.skipped_work.entry(rest.into()).or_default() += 1;
                    return Ok(());
                }
                Err(stop(format!(
                    "command is not modeled (x {:x?})",
                    &command.x[..5]
                )))
            }
        }
    }

    /// Write tightly packed texels (format blocks, row after row, slice after
    /// slice) from guest memory at `source` into `region` (x, y, z, width,
    /// height, depth) of `texture`'s level 0. Storage is block-linear (buffer
    /// textures are linear), so the write is swizzled the way the GPU reads
    /// it. 1D, 2D, 3D, rectangle and buffer textures are modeled; views of
    /// other levels or layers are not.
    /// Copy tightly packed texels between CPU memory at `source` and the
    /// texture's storage: into the texture when `to_texture`, else out.
    fn copy_texels(
        &self,
        core: &mut Guest,
        texture: u64,
        view: Option<&BTreeMap<String, Args>>,
        region: [u32; 6],
        source: u64,
        to_texture: bool,
    ) -> Result<(), TexelError> {
        let texture_object = texture;
        let texture = self
            .object(texture, "Texture")
            .map_err(TexelError::Invalid)?;
        if let Some(view) = view {
            let base = |key: &str| view.get(key).map_or(0, |value| value.x[0]);
            if base("Levels") != 0 || base("Layers") != 0 {
                return Err(TexelError::Stop(
                    "texture views of other levels or layers".into(),
                ));
            }
        }
        let get = |key: &str| texture.builder.get(key).map(|value| value.x[0] as u32);
        let format = get("Format").unwrap_or(0);
        let info = crate::nvn_format::info(format)
            .ok_or_else(|| TexelError::Stop(format!("texture format {format} is not modeled")))?;
        let target = get("Target").unwrap_or(1);
        let (width, mut height) = (get("Width").unwrap_or(1), get("Height").unwrap_or(1));
        let depth = if target == 2 {
            get("Depth").unwrap_or(1)
        } else {
            1
        };
        // Array and cube targets keep their layers (faces) in Depth; the
        // region's z selects them. Each layer is a whole mip chain at a fixed
        // stride (the per-layer size block_linear_size uses). A 1D array
        // (target 3) keeps its layers in Height and selects them with the
        // region's y: each layer is one row, the region's y/height become
        // its z/depth.
        let mut region = region;
        let layers = match target {
            4 | 6 | 8 | 9 => get("Depth").unwrap_or(1),
            3 => {
                let [rx, ry, _, rw, rh, _] = region;
                region = [rx, 0, ry, rw, 1, rh];
                std::mem::replace(&mut height, 1)
            }
            _ => 1,
        };
        let layer_stride = if layers > 1 {
            let levels = get("Levels").unwrap_or(1);
            crate::nvn_format::block_linear_size(info, width, height, 1, 2, levels).0 / 2
        } else {
            0
        };
        let [rx, ry, rz, rw, rh, rd] = region;
        let stride = texture.builder.get("Stride").map_or(0, |value| value.x[0]);
        if !matches!(target, 0..=4 | 7..=10)
            || rz.checked_add(rd).is_none_or(|end| end > depth.max(layers))
        {
            return Err(TexelError::Stop(format!(
                "texel write to target {target} at depth {rz}+{rd} of {depth}"
            )));
        }
        if rx.checked_add(rw).is_none_or(|end| end > width)
            || ry.checked_add(rh).is_none_or(|end| end > height.max(1))
        {
            return Err(TexelError::Invalid(format!(
                "region {rx},{ry} {rw}x{rh} outside {width}x{height}"
            )));
        }
        let storage = texture
            .builder
            .get("Storage")
            .ok_or_else(|| TexelError::Stop("texture has no storage".into()))?;
        self.object(storage.x[0], "MemoryPool")
            .map_err(TexelError::Invalid)?;
        let base = self
            .texture_storage(texture_object)
            .ok_or_else(|| TexelError::Stop("texture storage is in no mapped memory".into()))?;
        let (bw, bh) = (info.block_width, info.block_height);
        if rx % bw != 0 || ry % bh != 0 {
            return Err(TexelError::Invalid("region is not block aligned".into()));
        }
        let bytes = u64::from(info.bytes_per_block);
        let (columns, rows) = (rw.div_ceil(bw), rh.div_ceil(bh));
        let row_size = u64::from(columns) * bytes;
        let mut line = vec![0u8; row_size as usize];
        // Move `length` bytes between the CPU line at `start` and texture
        // storage at `address`, in the copy's direction.
        let transfer = |core: &mut Guest, line: &mut [u8], address: u64| {
            if to_texture {
                core.memory.write(address, line)
            } else {
                core.memory.read(address, line)
            }
            .map_err(TexelError::Trap)
        };
        for slice in 0..rd {
            for row in 0..rows {
                let from =
                    source + (u64::from(slice) * u64::from(rows) + u64::from(row)) * row_size;
                if to_texture {
                    core.memory
                        .read(from, &mut line)
                        .map_err(TexelError::Trap)?;
                }
                if target == 10 {
                    // Buffer textures have one linear row.
                    transfer(core, &mut line, base + u64::from(rx) * bytes)?;
                } else if stride != 0 {
                    let offset = u64::from(ry / bh + row) * stride + u64::from(rx / bw) * bytes;
                    transfer(core, &mut line, base + offset)?;
                } else {
                    let (z, layer) = if layers > 1 {
                        (0, u64::from(rz + slice))
                    } else {
                        (rz + slice, 0)
                    };
                    // A run of blocks at consecutive addresses at a time.
                    let mut column = 0;
                    while column < columns {
                        let x = rx / bw + column;
                        let run =
                            crate::nvn_format::block_linear_run(info, x).min(columns - column);
                        let offset = layer * layer_stride
                            + crate::nvn_format::block_linear_offset(
                                info,
                                width,
                                height,
                                depth,
                                x,
                                ry / bh + row,
                                z,
                            );
                        let start = (u64::from(column) * bytes) as usize;
                        transfer(
                            core,
                            &mut line[start..start + (u64::from(run) * bytes) as usize],
                            base + offset,
                        )?;
                        column += run;
                    }
                }
                if !to_texture {
                    core.memory.write(from, &line).map_err(TexelError::Trap)?;
                }
            }
        }
        Ok(())
    }

    /// Fill `command.states` for the Bind*State commands. Single states are
    /// passed by pointer (X1); vertex attribute and stream states as
    /// (count X1, array X2). Array elements are the bags of that kind at or
    /// above the array, in address order (the element size is NVN-private).
    fn snapshot_states(&self, command: &mut Command) {
        let Some(kind) = command
            .name
            .strip_prefix("nvnCommandBufferBind")
            .filter(|kind| kind.ends_with("State"))
        else {
            return;
        };
        let (count, base) = match kind {
            "VertexAttribState" | "VertexStreamState" => (command.x[1] as usize, command.x[2]),
            _ => (1, command.x[1]),
        };
        if count > 32 || base == 0 {
            return;
        }
        command.states = std::sync::Arc::new(
            self.bags
                .range(base..)
                .filter(|(_, bag)| bag.kind == kind)
                .take(count)
                .map(|(address, bag)| {
                    let mut props = bag.props.clone();
                    // Element offset from the array start, as evidence of the layout.
                    props.insert(
                        "@Offset".into(),
                        Args {
                            x: [address - base, 0, 0, 0],
                            v: [0; 2],
                        },
                    );
                    props
                })
                .collect(),
        );
    }

    /// The code window of the stage at GPU address `gpu` (up to 256 KiB:
    /// two gameplay programs run past 64 KiB; clipped to its memory pool
    /// and to readable memory) and its key: an
    /// FNV-1a hash of the program up to the end `sass::program_length` finds.
    /// Windows are cached by address and reread only when their first 512
    /// bytes (the header and first instructions) change.
    fn stage_code(&self, core: &Guest, gpu: u64) -> Option<(std::sync::Arc<Vec<u8>>, u64)> {
        let (guest, remaining) = self.pool_for(gpu)?;
        let length = remaining.min(0x40000);
        let prefix = read_prefix(core, guest, length.min(512));
        if let Some((code, key)) = self.stage_cache.borrow().get(&gpu) {
            if code.len() as u64 == length && code.starts_with(&prefix) {
                return Some((code.clone(), *key));
            }
        }
        let code = read_prefix(core, guest, length);
        let end = crate::sass::program_length(&code).map_or(code.len(), |count| count * 16);
        let key = code[..end]
            .iter()
            .fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
                (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
            });
        let code = std::sync::Arc::new(code);
        self.stage_cache
            .borrow_mut()
            .insert(gpu, (code.clone(), key));
        Some((code, key))
    }

    /// With ECHORA_DRAW_LOG set, append one JSON line per executed draw or
    /// dispatch:
    /// the bound program's stage keys (as in ECHORA_SHADER_DIR) and the
    /// per-slot bindings, the evidence for how NVN bindings reach the
    /// constant buffers translated shaders read.
    fn log_draw(&self, core: &Guest, command: &Command, path: &std::ffi::OsStr) {
        // Stage keys by stage bit, from each bound program's stages.
        let stages: Vec<serde_json::Value> = self
            .render
            .state
            .iter()
            .filter_map(|(key, bind)| Some((key.strip_prefix("BindProgram@")?, bind)))
            .map(|(bit, bind)| {
                let keys: Vec<String> = self
                    .objects
                    .get(&bind.x[1])
                    .map(|program| {
                        (0..6)
                            .filter_map(|index| program.state.get(&format!("Stage{index}")))
                            .filter_map(|stage| self.stage_code(core, stage.x[0]))
                            .map(|(_, key)| format!("{key:016x}"))
                            .collect()
                    })
                    .unwrap_or_default();
                serde_json::json!([bit, format!("{:#x}", bind.x[1]), keys])
            })
            .collect();
        let binds: Vec<serde_json::Value> = self
            .render
            .state
            .iter()
            .filter(|(key, _)| key.contains('/'))
            .map(|(key, bind)| serde_json::json!([key, bind.x[1..6]]))
            .collect();
        // Fixed-function state: each property's arguments.
        let states: serde_json::Map<String, serde_json::Value> = self
            .render
            .state
            .iter()
            .filter(|(key, bind)| key.ends_with("State") && !bind.states.is_empty())
            .map(|(key, bind)| {
                let elements: Vec<BTreeMap<&String, [u64; 4]>> = bind
                    .states
                    .iter()
                    .map(|props| props.iter().map(|(name, args)| (name, args.x)).collect())
                    .collect();
                (key.clone(), serde_json::json!(elements))
            })
            .collect();
        let line = serde_json::json!({
            "present": self.presented_frames,
            "colors": self.render.colors.iter().map(|texture| format!("{texture:x}")).collect::<Vec<_>>(),
            "depth": format!("{:x}", self.render.depth),
            "draw": command.name,
            "args": command.x[1..6],
            "stages": stages,
            "binds": binds,
            "states": states,
        });
        use std::io::Write;
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            let _ = writeln!(file, "{line}");
        }
    }

    /// Debug aid for shader translation: with ECHORA_SHADER_DIR set, write
    /// each stage's code (up to 64 KiB, clipped to its memory pool) and
    /// control section (first 4 KiB) as stage-<key>.{code,control}. The code
    /// window is kept whole: library subroutines after the main EXIT belong
    /// to the program, and the translator finds the end itself. The key
    /// hashes the code up to the end `sass::program_length` finds, so a
    /// program shared by many NVN programs is written once.
    fn dump_shader_stages(
        &self,
        core: &Guest,
        stages: &[Args],
        directory: &std::path::Path,
    ) -> Result<(), String> {
        std::fs::create_dir_all(directory).map_err(|error| error.to_string())?;
        for stage in stages {
            let (gpu, control) = (stage.x[0], stage.x[1]);
            let Some((code, key)) = self.stage_code(core, gpu) else {
                continue;
            };
            let path = directory.join(format!("stage-{key:016x}.code"));
            if path.exists() {
                continue;
            }
            std::fs::write(
                directory.join(format!("stage-{key:016x}.control")),
                read_prefix(core, control, 0x1000),
            )
            .map_err(|error| error.to_string())?;
            std::fs::write(&path, &code[..]).map_err(|error| error.to_string())?;
        }
        Ok(())
    }

    /// Guest address of `size` bytes at GPU address `address`: the memory
    /// pool whose GPU range holds them, at the same offset in its storage.
    fn gpu_to_guest(&self, address: u64, size: u64) -> Result<u64, String> {
        self.pool_for(address)
            .filter(|(_, remaining)| size <= *remaining)
            .map(|(guest, _)| guest)
            .ok_or_else(|| format!("GPU range {address:#x}+{size:#x} is in no memory pool"))
    }

    /// The guest address of GPU address `address` and the bytes left in its
    /// memory pool. Pool ranges are cached; a range whose pool object is
    /// gone, or a miss, rebuilds the cache from the live pools.
    fn pool_for(&self, address: u64) -> Option<(u64, u64)> {
        let find = |ranges: &[[u64; 5]]| {
            ranges
                .iter()
                .find_map(|&[key, pool_gpu, gpu, size, memory]| {
                    let offset = address.checked_sub(gpu).filter(|offset| *offset < size)?;
                    let live = self.objects.get(&key).is_some_and(|pool| {
                        pool.kind == "MemoryPool"
                            && pool.state.get("GpuAddress").map(|args| args.x[0]) == Some(pool_gpu)
                    });
                    live.then_some((memory + offset, size - offset))
                })
        };
        if let Some(found) = find(&self.pool_ranges.borrow()) {
            return Some(found);
        }
        let mut ranges: Vec<[u64; 5]> = Vec::new();
        for (key, pool) in &self.objects {
            if pool.kind != "MemoryPool" {
                continue;
            }
            let Some(gpu) = pool.state.get("GpuAddress").map(|args| args.x[0]) else {
                continue;
            };
            let storage = &pool.builder["Storage"];
            if storage.x[0] != 0 {
                ranges.push([*key, gpu, gpu, storage.x[1], storage.x[0]]);
                continue;
            }
            // A virtual pool: each mapping reaches into a physical pool.
            for (name, mapping) in &pool.state {
                let Some(offset) = name
                    .strip_prefix(VIRTUAL_MAPPING)
                    .and_then(|hex| u64::from_str_radix(hex, 16).ok())
                else {
                    continue;
                };
                let [physical, physical_offset, size, _] = mapping.x;
                let Some(storage) = self
                    .objects
                    .get(&physical)
                    .filter(|object| object.kind == "MemoryPool")
                    .map(|object| object.builder["Storage"].x[0])
                else {
                    continue;
                };
                ranges.push([*key, gpu, gpu + offset, size, storage + physical_offset]);
            }
        }
        let found = find(&ranges);
        *self.pool_ranges.borrow_mut() = ranges;
        found
    }

    fn texture_extent(&self, texture: u64) -> Result<(u32, u32), String> {
        let object = self.object(texture, "Texture")?;
        let width = object.builder.get("Width").map(|value| value.x[0] as u32);
        let height = object.builder.get("Height").map(|value| value.x[0] as u32);
        match (width, height) {
            (Some(width), Some(height))
                if width > 0 && height > 0 && width <= 16384 && height <= 16384 =>
            {
                Ok((width, height))
            }
            _ => Err(format!("texture {texture:#x} has no usable size")),
        }
    }

    fn bag(&self, address: u64, kind: &str) -> Result<BTreeMap<String, Args>, String> {
        self.bags
            .get(&address)
            .filter(|bag| bag.kind == kind)
            .map(|bag| bag.props.clone())
            .ok_or_else(|| format!("no {kind} at {address:#x}"))
    }

    fn object(&self, address: u64, kind: &str) -> Result<&Object, String> {
        self.objects
            .get(&address)
            .filter(|object| object.kind == kind)
            .ok_or_else(|| format!("{kind} {address:#x} is not initialized"))
    }

    fn require_device(&self, builder: &BTreeMap<String, Args>) -> Result<(), String> {
        let device = builder
            .get("Device")
            .ok_or_else(|| "builder device was never set".to_string())?;
        self.object(device.x[0], "Device").map(|_| ())
    }

    /// Guest address of an initialized event's u32: its builder's storage
    /// (memory pool, offset) inside the pool's storage.
    fn event_address(&self, event: u64) -> Result<u64, String> {
        let event = self.object(event, "Event")?;
        let storage = event
            .builder
            .get("Storage")
            .copied()
            .ok_or("event has no storage")?;
        let pool = self.object(storage.x[0], "MemoryPool")?;
        let pool_storage = pool.builder["Storage"];
        if pool_storage.x[0] == 0 || storage.x[1].saturating_add(4) > pool_storage.x[1] {
            return Err(format!(
                "event storage {:#x}+{:#x} outside its pool",
                storage.x[0], storage.x[1]
            ));
        }
        Ok(pool_storage.x[0] + storage.x[1])
    }

    /// SetShaders(count, const NVNshaderData*) on an initialized program.
    /// Each entry (`stride` bytes) starts with the GPU address of a compiled
    /// stage and a
    /// pointer to its control section (inferred from the 16-byte stride at
    /// the call sites). The stages are kept for shader translation; nothing
    /// is compiled yet, so the program always links.
    fn set_program_shaders(
        &mut self,
        core: &mut Guest,
        program: u64,
        count: u64,
        data: u64,
        stride: u64,
    ) -> Result<(), Trap> {
        let invalid = |message: String| Trap::InvalidState {
            message: format!("nvnProgram shaders: {message}"),
        };
        let count = count as u32;
        if count == 0 || count > 6 {
            return Err(invalid(format!("{count} shader stages")));
        }
        let mut stages = Vec::new();
        for index in 0..u64::from(count) {
            let entry = data + index * stride;
            stages.push(Args {
                x: [
                    core.memory.read_u64(entry)?,
                    core.memory.read_u64(entry + 8)?,
                    0,
                    0,
                ],
                v: [0; 2],
            });
        }
        if let Some(directory) = debug_var_os("ECHORA_SHADER_DIR") {
            self.dump_shader_stages(core, &stages, std::path::Path::new(&directory))
                .map_err(|error| invalid(format!("shader dump: {error}")))?;
        }
        let program = self
            .objects
            .get_mut(&program)
            .filter(|object| object.kind == "Program")
            .ok_or_else(|| invalid("program is not initialized".into()))?;
        program.state.retain(|key, _| !key.starts_with("Stage"));
        for (index, stage) in stages.into_iter().enumerate() {
            program.state.insert(format!("Stage{index}"), stage);
        }
        Ok(())
    }

    fn initialize(
        &mut self,
        address: u64,
        kind: &str,
        builder: BTreeMap<String, Args>,
    ) -> Result<(), String> {
        if self.objects.contains_key(&address) {
            return Err(format!("{kind} {address:#x} is already initialized"));
        }
        self.objects.insert(
            address,
            Object {
                kind: kind.into(),
                builder,
                state: BTreeMap::new(),
            },
        );
        Ok(())
    }

    pub fn report(&self) -> serde_json::Value {
        serde_json::json!({
            "model": "host_nvn_entry_points",
            "objects": self.objects.len(),
            "devices": self.objects.values().filter(|o| o.kind == "Device").collect::<Vec<_>>(),
            "hooked_exports": self.hooks.values().collect::<Vec<_>>(),
            "entry_points_looked_up": self.thunks.len(),
            "calls": self.calls,
            "call_microseconds": self.call_microseconds,
            "presented_frames": self.presented_frames,
            "texture_log": self.texture_log,
            "stale_command_submissions": self.stale_command_submissions,
            "skipped_work": self.skipped_work,
            "forced_waits": self.forced_waits,
            "sync_timeouts": self.sync_timeouts.iter().map(|(sync, count)| (format!("{sync:#x}"), count)).collect::<BTreeMap<_, _>>(),
            "gpu_notes": self.gpu.as_ref().map(|backend| &backend.notes),
            "gpu_flushes": self.gpu.as_ref().map(|backend| {
                [
                    backend.renderer.flushes,
                    backend.renderer.flush_wait_microseconds,
                    backend.renderer.targets_created,
                    backend.renderer.targets_recreated,
                    backend.renderer.partial_clears,
                    backend.renderer.descaled,
                ]
            }),
        })
    }
}

/// Storage size and alignment of a texture described by builder properties.
/// Targets (inferred from the application's nn::gfx conversion, which picks
/// them by dimension and sample count, and consistent with NVN's public enum
/// order): 0 1D, 1 2D, 2 3D, 3 1D array (layers in Height), 4 2D array, 5 2D
/// multisample, 6 2D multisample array, 7 rectangle, 8 cube map (depth 6), 9
/// cube map array (depth 6 x layers), 10 buffer (the application sizes it
/// with SetSize1D in texels of its format). Array and cube targets carry their
/// layers in Depth. Buffer textures are linear: width times the texel size,
/// with the 256-byte alignment used for every buffer.
/// Up to `length` bytes at `address`, shrunk to the readable prefix when the
/// window crosses unmapped memory.
fn read_prefix(core: &Guest, address: u64, length: u64) -> Vec<u8> {
    let mut bytes = vec![0u8; length as usize];
    let mut readable = bytes.len();
    while readable > 0 && core.memory.read(address, &mut bytes[..readable]).is_err() {
        readable /= 2;
    }
    bytes.truncate(readable);
    bytes
}

fn texture_storage(props: &BTreeMap<String, Args>) -> Result<(u64, u64), String> {
    let get = |key: &str| props.get(key).map(|value| value.x[0] as u32);
    if props.contains_key("PackagedTextureLayout") {
        return Err("packaged texture layouts are not modeled".into());
    }
    let format = get("Format").ok_or("texture format was never set")?;
    let info = crate::nvn_format::info(format)
        .ok_or_else(|| format!("texture format {format} is not modeled"))?;
    let target = get("Target").ok_or("texture target was never set")?;
    let width = get("Width").ok_or("texture width was never set")?;
    let height = get("Height").unwrap_or(1);
    let depth = get("Depth").unwrap_or(1);
    let levels = get("Levels").unwrap_or(1);
    let samples = get("Samples").unwrap_or(0).max(1);
    let max_width = if target == 10 { 1 << 27 } else { 32768 };
    if width == 0
        || width > max_width
        || height == 0
        || height > 32768
        || levels == 0
        || levels > 16
    {
        return Err(format!("texture {width}x{height} with {levels} levels"));
    }
    // Linear (pitch) textures: SetStride gives the row pitch in bytes.
    let stride = props.get("Stride").map_or(0, |value| value.x[0]);
    if stride != 0 && target != 10 {
        if levels != 1 || !matches!(target, 1 | 7) {
            return Err(format!(
                "linear texture target {target} with {levels} levels"
            ));
        }
        let rows = u64::from(height.div_ceil(info.block_height));
        return Ok((stride * rows, 256));
    }
    if target == 10 {
        if info.block_width != 1 || info.block_height != 1 || levels != 1 {
            return Err(format!(
                "buffer texture of format {format} with {levels} levels"
            ));
        }
        return Ok((u64::from(width) * u64::from(info.bytes_per_block), 256));
    }
    let (height, depth, layers) = match target {
        0 => (1, 1, 1),
        1 | 5 | 7 => (height, 1, 1),
        2 => (height, depth, 1),
        3 => (1, 1, height),
        4 | 6 | 8 | 9 => (height, 1, depth),
        _ => return Err(format!("texture target {target} is not modeled")),
    };
    // Multisampled surfaces store their samples as a larger grid.
    let (sample_x, sample_y) = match samples {
        1 => (1, 1),
        2 => (2, 1),
        4 => (2, 2),
        8 => (4, 2),
        _ => return Err(format!("{samples} samples")),
    };
    Ok(crate::nvn_format::block_linear_size(
        info,
        width * sample_x,
        height * sample_y,
        depth,
        layers,
        levels,
    ))
}

/// Why a texel transfer could not run.
enum TexelError {
    Stop(String),
    Invalid(String),
    Trap(Trap),
}

impl TexelError {
    fn into_trap(self, name: &str, pc: u64) -> Trap {
        match self {
            Self::Stop(message) => Trap::UnsupportedService {
                pc,
                service: "nvn".into(),
                command_id: None,
                message: format!("{name}: {message}"),
            },
            Self::Invalid(message) => Trap::InvalidState {
                message: format!("{name}: {message}"),
            },
            Self::Trap(trap) => trap,
        }
    }
}

/// NVNcopyRegion: six ints (x, y, z offsets; width, height, depth), NVN's
/// public layout.
fn read_region(core: &Guest, address: u64) -> Result<[u32; 6], Trap> {
    let mut region = [0u32; 6];
    for (index, value) in region.iter_mut().enumerate() {
        *value = core.memory.read_u32(address + index as u64 * 4)?;
    }
    Ok(region)
}

/// Data NVN reads through pointer arguments when a command is recorded.
fn capture(command: &mut Command, core: &Guest) -> Result<(), Trap> {
    let read_u64s = |address: u64, count: usize| -> Result<Vec<u64>, Trap> {
        (0..count as u64)
            .map(|index| {
                if address == 0 {
                    Ok(0)
                } else {
                    core.memory.read_u64(address + index * 8)
                }
            })
            .collect()
    };
    match command.name.as_str() {
        "nvnCommandBufferSetRenderTargets" => {
            let count = command.x[1] as usize;
            if count > 8 {
                return Err(Trap::InvalidState {
                    message: format!("{count} render targets"),
                });
            }
            command.captured = read_u64s(command.x[2], count)?;
            command.captured.extend(read_u64s(command.x[3], count)?);
        }
        "nvnCommandBufferCopyBufferToTexture" => {
            command.captured = read_region(core, command.x[4])?
                .iter()
                .map(|value| u64::from(*value))
                .collect();
            // View presence, then its level and layer bases (filled in by
            // Nvn::call, which holds the view's property bag).
            command
                .captured
                .extend([u64::from(command.x[3] != 0), 0, 0]);
        }
        // CopyTextureToTexture(cmd, src texture, src view, src region,
        // dst texture, dst view, dst region, flags) (inferred from the
        // other copies): both regions, read at record time.
        "nvnCommandBufferCopyTextureToTexture" => {
            command.captured = read_region(core, command.x[3])?
                .iter()
                .chain(read_region(core, command.x[6])?.iter())
                .map(|value| u64::from(*value))
                .collect();
        }
        "nvnCommandBufferSetDepthRanges" => {
            // The first entry's near and far (floats).
            if command.x[2] != 0 && command.x[3] != 0 {
                command.captured = (0..2)
                    .map(|word| core.memory.read_u32(command.x[3] + word * 4).map(u64::from))
                    .collect::<Result<_, _>>()?;
            }
        }
        "nvnCommandBufferSetViewports" | "nvnCommandBufferSetScissors" => {
            // The first entry's four 32-bit words (floats or ints).
            if command.x[2] != 0 && command.x[3] != 0 {
                command.captured = (0..4)
                    .map(|word| core.memory.read_u32(command.x[3] + word * 4).map(u64::from))
                    .collect::<Result<_, _>>()?;
            }
        }
        // UpdateUniformBuffer(cmd, buffer address, buffer size, offset,
        // size, const void* data): the data, in u64 words (the last one
        // zero-padded).
        "nvnCommandBufferUpdateUniformBuffer" => {
            let size = command.x[4];
            if size > 0x1_0000
                || command.x[3]
                    .checked_add(size)
                    .is_none_or(|end| end > command.x[2])
            {
                return Err(Trap::InvalidState {
                    message: format!(
                        "uniform update {:#x}+{size:#x} of a {:#x}-byte buffer",
                        command.x[3], command.x[2]
                    ),
                });
            }
            let mut bytes = vec![0u8; size.next_multiple_of(8) as usize];
            core.memory
                .read(command.x[5], &mut bytes[..size as usize])?;
            command.captured = bytes
                .chunks(8)
                .map(|word| u64::from_le_bytes(word.try_into().unwrap_or_default()))
                .collect();
        }
        // Push{Graphics,Compute}Constants(cmd, int table, int offset, int
        // bytes, const void* data): the words now (the driver clamps to the
        // 256-byte table and drops pushes under 4 bytes).
        "nvnCommandBufferPushGraphicsConstants" | "nvnCommandBufferPushComputeConstants" => {
            let bytes = (command.x[3] as u32).min(0x100);
            command.captured = (0..u64::from(bytes / 4))
                .map(|word| core.memory.read_u32(command.x[4] + word * 4).map(u64::from))
                .collect::<Result<_, _>>()?;
        }
        "nvnCommandBufferClearColor" => {
            command.captured = (0..4)
                .map(|channel| {
                    core.memory
                        .read_u32(command.x[2] + channel * 4)
                        .map(u64::from)
                })
                .collect::<Result<_, _>>()?;
        }
        // ClearTexture(cmd, const NVNtexture*, const NVNtextureView*, const
        // NVNcopyRegion*, const float color[4], int mask): the four color
        // words, then the region's six words (none for a null region); the
        // view's level and layer bases follow at record time.
        "nvnCommandBufferClearTexture" => {
            let mut captured: Vec<u64> = (0..4)
                .map(|channel| {
                    core.memory
                        .read_u32(command.x[4] + channel * 4)
                        .map(u64::from)
                })
                .collect::<Result<_, _>>()?;
            if command.x[3] != 0 {
                for word in 0..6 {
                    captured.push(u64::from(core.memory.read_u32(command.x[3] + word * 4)?));
                }
            }
            command.captured = captured;
        }
        _ => {}
    }
    Ok(())
}

impl Image {
    /// Binary PPM (P6), 8 bits per channel, channels clamped to [0, 1].
    /// The frame as written: a binary PPM of 8-bit RGB.
    pub fn ppm(&self) -> Vec<u8> {
        let mut bytes = format!("P6\n{} {}\n255\n", self.width, self.height).into_bytes();
        for texel in &self.texels {
            for channel in &texel[..3] {
                bytes.push((channel.clamp(0.0, 1.0) * 255.0).round() as u8);
            }
        }
        bytes
    }

    /// PNG of 8-bit RGB (channels clamped to [0, 1]), for frame captures.
    pub fn png(&self) -> Vec<u8> {
        let mut raw = Vec::with_capacity(self.texels.len() * 3 + self.height as usize);
        for row in self.texels.chunks(self.width.max(1) as usize) {
            raw.push(0); // filter: none
            for texel in row {
                for channel in &texel[..3] {
                    raw.push((channel.clamp(0.0, 1.0) * 255.0).round() as u8);
                }
            }
        }
        let mut header = Vec::with_capacity(13);
        header.extend_from_slice(&self.width.to_be_bytes());
        header.extend_from_slice(&self.height.to_be_bytes());
        header.extend_from_slice(&[8, 2, 0, 0, 0]); // 8-bit RGB
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        for (kind, data) in [
            (b"IHDR", header),
            (b"IDAT", miniz_oxide::deflate::compress_to_vec_zlib(&raw, 6)),
            (b"IEND", Vec::new()),
        ] {
            png.extend_from_slice(&(data.len() as u32).to_be_bytes());
            let start = png.len();
            png.extend_from_slice(kind);
            png.extend_from_slice(&data);
            let crc = crc32(&png[start..]);
            png.extend_from_slice(&crc.to_be_bytes());
        }
        png
    }
}

/// CRC-32 (IEEE, as PNG chunks use).
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xedb8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// Modeled nvnDeviceGetInteger properties. NVN is undocumented, so each value
/// is justified by how the application uses the answer; the values describe
/// this host implementation, not NVIDIA's driver:
/// - 42, 90: folded as `value != 0` into two optional-feature bits, and both
///   states are handled. The host device offers neither optional feature.
/// - 4, 10, 11, 12, 13, 53, 111, 124: used only as alignment requirements (the
///   application takes the maximum with its own power-of-two alignment) for
///   buffers selected by usage flags. Host memory pools are plain guest memory,
///   so one 256-byte alignment (Vulkan's largest common offset requirement)
///   serves every usage.
/// - 15, 16: the per-slot increment of texture and sampler descriptor pools
///   (slot count times this value sizes the pool). 32 bytes, the hardware
///   texture-header and sampler entry size (NVIDIA open-gpu-doc).
/// - 17, 18: queried next to 15 and 16 when sizing those pools and used as
///   their alignment: 256 like the other alignments.
/// - 19, 20: reserved texture and sampler descriptor counts. The host reserves
///   no descriptors.
/// - 56, 57: the row-pitch alignment of linear textures (57 when the
///   texture's render-target usage bits 0x180 are set); the application rounds
///   the pitch up to it before setting the stride (main+0x16b9268). 256, a
///   multiple of every pitch-linear alignment and the host's buffer alignment.
/// - 36, 37: queried next to 17 and 18 when nn::gfx initializes a texture or
///   sampler descriptor pool (main+0x105cb1c); the answer is stored to a
///   stack slot that is never read. 0.
/// - 0, 1: the API major and minor version. The application requires major
///   231 and minor 40 or later (main+0x346c10c) before it creates its
///   graphics system; the sdk's own NVN reports 231.40 (sdk+0x69f000).
/// - 24, 25, 26: the application multiplies 24 by a count of its own and
///   requires the product to be a multiple of 25, of 26 and of 4 KiB before
///   allocating it (main+0xf4962c), the shape of shader scratch memory
///   (factor, alignment, granularity). The host needs no scratch memory, so
///   each is 4096, the smallest answer every check accepts.
/// - 81, 85, 91: copied into the size fields of a default configuration
///   structure next to fixed 4 MiB and 1 MiB sizes (main+0xf26c70).
///   Host command buffers record into host memory and need none of it, so
///   each is 4096, valid as a size and as a power-of-two alignment.
///   Provisional.
/// - 39, 78, 103: cached with 15..18, 25, 36, 37 in the application's
///   device-info structure (main+0x20e8540). With 103 at 0 a buffer offset
///   computed from it becomes -1 (rounding up to a multiple of 0), and with
///   39 at 0 a flushed range falls outside its buffer, so both act as
///   alignments: 256, like the other alignments. 78 showed no effect: 0.
///   Provisional.
/// - 54: queried when the application creates a render target after the
///   opening movie (main+0x244bdc) and kept with it (+0xe4) as the
///   alignment of the memory that saves and restores the target's depth
///   culling state. The host keeps no such state: 256, like the other
///   alignments. Provisional.
fn device_integer(property: u32) -> Option<i32> {
    match property {
        39 => Some(256),
        78 => Some(0),
        103 => Some(256),
        81 | 85 | 91 => Some(4096),
        24..=26 => Some(4096),
        0 => Some(231),
        1 => Some(40),
        36 | 37 => Some(0),
        56 | 57 => Some(256),
        42 | 90 => Some(0),
        4 | 10 | 11 | 12 | 13 | 53 | 54 | 111 | 124 | 17 | 18 => Some(256),
        15 | 16 => Some(32),
        19 | 20 => Some(0),
        _ => None,
    }
}

/// A sampler property SetDefaults sets, for getters on samplers whose
/// builder never set it: OpenGL's defaults, which NVN sampler state mirrors
/// (LOD clamp -1000..1000, LOD bias 0, no depth compare with function
/// LEQUAL, anisotropy 1, min filter nearest-mipmap-linear and mag linear,
/// repeat wrapping, average reduction 0, LOD snap 0), in the enum numbering `nvn_gpu` decodes: compare
/// LEQUAL 4, filters 4 and 1, wrap 1. Floats are in `v`. Provisional.
fn sampler_default(property: &str) -> Option<Args> {
    let float = |value: f32| u64::from(value.to_bits());
    let (x, v) = match property {
        "LodClamp" => ([0; 4], [float(-1000.0), float(1000.0)]),
        "LodBias" => ([0; 4], [float(0.0), 0]),
        "Compare" => ([0, 4, 0, 0], [0; 2]),
        "MaxAnisotropy" => ([0; 4], [float(1.0), 0]),
        "MinMagFilter" => ([4, 1, 0, 0], [0; 2]),
        "WrapMode" => ([1, 1, 1, 0], [0; 2]),
        "ReductionFilter" => ([0; 4], [0; 2]),
        "LodSnap" => ([0; 4], [float(0.0), 0]),
        _ => return None,
    };
    Some(Args { x, v })
}

/// A texture property SetDefaults sets, for getters on textures whose
/// builder never set it: no stride (block-linear), single-sampled, depth
/// read mode 0, no flags, one level, depth 1.
fn texture_default(property: &str) -> Option<Args> {
    let value = match property {
        "Stride" | "Samples" | "DepthStencilMode" | "Flags" => 0,
        "Levels" | "Depth" => 1,
        _ => return None,
    };
    Some(Args {
        x: [value, 0, 0, 0],
        v: [0; 2],
    })
}

/// State-key prefix of a virtual pool's mapping (followed by the virtual
/// offset in hex): physical pool, physical offset, size.
const VIRTUAL_MAPPING: &str = "VirtualMapping/";

/// Whether `name` is a plausible NVN entry-point name (ASCII identifier with
/// the `nvn` prefix). Lookups of anything else stop explicitly.
pub fn is_entry_point_name(name: &str) -> bool {
    name.starts_with("nvn")
        && name.len() > 3
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    use echora_guest::memory::{Memory, READ, STATE_NORMAL, WRITE};

    fn call(nvn: &mut Nvn, core: &mut Guest, name: &str, args: &[u64]) -> u64 {
        core.cpu.regs[..args.len()].copy_from_slice(args);
        assert!(nvn.call(name, core).unwrap(), "{name} is not implemented");
        core.cpu.regs[0]
    }

    #[test]
    fn virtual_pool_mappings_resolve_to_physical_storage() {
        let mut memory = Memory::new(1 << 39);
        memory
            .map(0x10_0000, 0x10_0000, READ | WRITE, STATE_NORMAL, "test")
            .unwrap();
        let mut cpu = Cpu::default();
        let mut core = Guest {
            cpu: &mut cpu,
            memory: GuestMemory::Checked(&mut memory),
            counter: 0,
        };
        let mut nvn = Nvn::default();
        let (device_builder, device) = (0x10_0000, 0x10_1000);
        call(
            &mut nvn,
            &mut core,
            "nvnDeviceBuilderSetDefaults",
            &[device_builder],
        );
        call(
            &mut nvn,
            &mut core,
            "nvnDeviceInitialize",
            &[device, device_builder],
        );
        let pool = |nvn: &mut Nvn, core: &mut Guest, builder, object, storage, size| {
            call(nvn, core, "nvnMemoryPoolBuilderSetDefaults", &[builder]);
            call(
                nvn,
                core,
                "nvnMemoryPoolBuilderSetDevice",
                &[builder, device],
            );
            call(
                nvn,
                core,
                "nvnMemoryPoolBuilderSetStorage",
                &[builder, storage, size],
            );
            assert_eq!(
                call(nvn, core, "nvnMemoryPoolInitialize", &[object, builder]),
                1
            );
            call(nvn, core, "nvnMemoryPoolGetBufferAddress", &[object])
        };
        pool(&mut nvn, &mut core, 0x10_2000, 0x10_3000, 0x10_8000, 0x4000);
        let virtual_gpu = pool(&mut nvn, &mut core, 0x10_2000, 0x10_3100, 0, 0x8000);
        // Physical offset 0x1000 of the first pool at virtual offset 0x2000.
        let request = 0x10_5000;
        for (index, word) in [0x10_3000u64, 0x1000, 0x2000, 0x2000].iter().enumerate() {
            core.memory
                .write(request + index as u64 * 8, &word.to_le_bytes())
                .unwrap();
        }
        let map = [0x10_3100, 1, request];
        assert_eq!(
            call(&mut nvn, &mut core, "nvnMemoryPoolMapVirtual", &map),
            1
        );
        assert_eq!(nvn.gpu_to_guest(virtual_gpu + 0x2010, 0x10), Ok(0x10_9010));
        assert!(nvn.gpu_to_guest(virtual_gpu + 0x3ff0, 0x20).is_err());
        assert!(nvn.gpu_to_guest(virtual_gpu, 4).is_err());
        // A null physical pool unmaps the range.
        core.memory.write(request, &0u64.to_le_bytes()).unwrap();
        assert_eq!(
            call(&mut nvn, &mut core, "nvnMemoryPoolMapVirtual", &map),
            1
        );
        assert!(nvn.gpu_to_guest(virtual_gpu + 0x2010, 0x10).is_err());
    }

    #[test]
    fn queue_memory_size_sums_the_builder_sizes() {
        let mut memory = Memory::new(1 << 39);
        let mut cpu = Cpu::default();
        let mut core = Guest {
            cpu: &mut cpu,
            memory: GuestMemory::Checked(&mut memory),
            counter: 0,
        };
        let mut nvn = Nvn::default();
        let builder = 0x10_0000;
        call(
            &mut nvn,
            &mut core,
            "nvnQueueBuilderSetDefaults",
            &[builder],
        );
        for (name, size) in [("Command", 0x1_0000), ("Compute", 0x4000)] {
            let setter = format!("nvnQueueBuilderSet{name}MemorySize");
            call(&mut nvn, &mut core, &setter, &[builder, size]);
        }
        let getter = "nvnQueueBuilderGetQueueMemorySize";
        assert!(nvn.call(getter, &mut core).is_err());
        call(
            &mut nvn,
            &mut core,
            "nvnQueueBuilderSetControlMemorySize",
            &[builder, 0x1000],
        );
        assert_eq!(call(&mut nvn, &mut core, getter, &[builder]), 0x1_5000);
    }

    #[test]
    fn clear_and_present_writes_the_frame() {
        let mut memory = Memory::new(1 << 39);
        memory
            .map(0x10_0000, 0x10_0000, READ | WRITE, STATE_NORMAL, "test")
            .unwrap();
        let mut cpu = Cpu::default();
        let mut core = Guest {
            cpu: &mut cpu,
            memory: GuestMemory::Checked(&mut memory),
            counter: 0,
        };
        // The CPU reference path: no GPU backend, so the images hold texels.
        let mut nvn = Nvn {
            gpu_checked: true,
            ..Nvn::default()
        };
        let (device_builder, device) = (0x10_0000, 0x10_1000);
        call(
            &mut nvn,
            &mut core,
            "nvnDeviceBuilderSetDefaults",
            &[device_builder],
        );
        assert_eq!(
            call(
                &mut nvn,
                &mut core,
                "nvnDeviceInitialize",
                &[device, device_builder]
            ),
            1
        );
        let (texture_builder, texture) = (0x10_2000, 0x10_3000);
        call(
            &mut nvn,
            &mut core,
            "nvnTextureBuilderSetDevice",
            &[texture_builder, device],
        );
        call(
            &mut nvn,
            &mut core,
            "nvnTextureBuilderSetDefaults",
            &[texture_builder],
        );
        call(
            &mut nvn,
            &mut core,
            "nvnTextureBuilderSetWidth",
            &[texture_builder, 4],
        );
        call(
            &mut nvn,
            &mut core,
            "nvnTextureBuilderSetHeight",
            &[texture_builder, 2],
        );
        call(
            &mut nvn,
            &mut core,
            "nvnTextureBuilderSetFormat",
            &[texture_builder, 37], // RGBA8
        );
        assert_eq!(
            call(
                &mut nvn,
                &mut core,
                "nvnTextureBuilderGetWidth",
                &[texture_builder]
            ),
            4
        );
        call(
            &mut nvn,
            &mut core,
            "nvnTextureInitialize",
            &[texture, texture_builder],
        );
        // Window with one texture: the pointer array lives at 0x10_4000.
        core.memory.write_u64(0x10_4000, texture).unwrap();
        let (window_builder, window, sync, queue_builder, queue) =
            (0x10_5000, 0x10_6000, 0x10_7000, 0x10_8000, 0x10_9000);
        call(
            &mut nvn,
            &mut core,
            "nvnWindowBuilderSetDevice",
            &[window_builder, device],
        );
        call(
            &mut nvn,
            &mut core,
            "nvnWindowBuilderSetDefaults",
            &[window_builder],
        );
        call(
            &mut nvn,
            &mut core,
            "nvnWindowBuilderSetTextures",
            &[window_builder, 1, 0x10_4000],
        );
        call(
            &mut nvn,
            &mut core,
            "nvnWindowInitialize",
            &[window, window_builder],
        );
        call(
            &mut nvn,
            &mut core,
            "nvnQueueBuilderSetDevice",
            &[queue_builder, device],
        );
        call(
            &mut nvn,
            &mut core,
            "nvnQueueBuilderSetDefaults",
            &[queue_builder],
        );
        call(
            &mut nvn,
            &mut core,
            "nvnQueueInitialize",
            &[queue, queue_builder],
        );
        call(&mut nvn, &mut core, "nvnSyncInitialize", &[sync, device]);
        assert_eq!(call(&mut nvn, &mut core, "nvnSyncWait", &[sync, 0]), 2);
        call(
            &mut nvn,
            &mut core,
            "nvnWindowAcquireTexture",
            &[window, sync, 0x10_4100],
        );
        assert_eq!(core.memory.read_u32(0x10_4100).unwrap(), 0);
        assert_eq!(call(&mut nvn, &mut core, "nvnSyncWait", &[sync, 0]), 0);
        // Record SetRenderTargets + ClearColor(0.5, 0.25, 1.0, 1.0).
        let command_buffer = 0x10_a000;
        call(
            &mut nvn,
            &mut core,
            "nvnCommandBufferInitialize",
            &[command_buffer, device],
        );
        call(
            &mut nvn,
            &mut core,
            "nvnCommandBufferBeginRecording",
            &[command_buffer],
        );
        core.memory.write_u64(0x10_4200, texture).unwrap();
        call(
            &mut nvn,
            &mut core,
            "nvnCommandBufferSetRenderTargets",
            &[command_buffer, 1, 0x10_4200, 0, 0, 0],
        );
        for (index, value) in [0.5f32, 0.25, 1.0, 1.0].iter().enumerate() {
            core.memory
                .write_u32(0x10_4300 + index as u64 * 4, value.to_bits())
                .unwrap();
        }
        call(
            &mut nvn,
            &mut core,
            "nvnCommandBufferClearColor",
            &[command_buffer, 0, 0x10_4300, 0xf],
        );
        // The color is captured at record time.
        core.memory.write_u32(0x10_4300, 0).unwrap();
        let handle = call(
            &mut nvn,
            &mut core,
            "nvnCommandBufferEndRecording",
            &[command_buffer],
        );
        core.memory.write_u64(0x10_4400, handle).unwrap();
        call(
            &mut nvn,
            &mut core,
            "nvnQueueSubmitCommands",
            &[queue, 1, 0x10_4400],
        );
        let image = &nvn.images[&texture];
        assert_eq!((image.width, image.height), (4, 2));
        assert!(
            image
                .texels
                .iter()
                .all(|texel| *texel == [0.5, 0.25, 1.0, 1.0])
        );
        call(
            &mut nvn,
            &mut core,
            "nvnQueuePresentTexture",
            &[queue, window, 0],
        );
        assert_eq!(nvn.presented_frames, 1);
    }

    #[test]
    fn thunks_are_stable_and_hooks_resolve() {
        let mut nvn = Nvn::default();
        nvn.add_hook(0x7104_0000, HostEntry::BootstrapLoader);
        let first = nvn.thunk("nvnDeviceGetProcAddress").unwrap();
        let second = nvn.thunk("nvnQueueInitialize").unwrap();
        assert_eq!(first, THUNK_BASE);
        assert_eq!(second, THUNK_BASE + THUNK_STRIDE);
        assert_eq!(nvn.thunk("nvnDeviceGetProcAddress").unwrap(), first);
        assert_eq!(
            nvn.entry(second),
            Some(HostEntry::Proc("nvnQueueInitialize".into()))
        );
        assert_eq!(nvn.entry(second + 4), None);
        assert_eq!(nvn.entry(THUNK_BASE + 0x100), None);
        assert_eq!(nvn.entry(0x7104_0000), Some(HostEntry::BootstrapLoader));
        assert!(is_entry_point_name("nvnQueueFinish"));
        assert!(!is_entry_point_name("nvn"));
        assert!(!is_entry_point_name("glClear"));
        assert!(!is_entry_point_name("nvn Queue"));
        assert_eq!(
            split_name("nvnTextureViewSetLevels"),
            Some(("TextureView", "SetLevels"))
        );
        assert_eq!(
            split_name("nvnTextureBuilderGetHeight"),
            Some(("TextureBuilder", "GetHeight"))
        );
        assert_eq!(split_name("nvnQueueFinish"), Some(("Queue", "Finish")));
        assert_eq!(
            split_name("nvnSamplerPoolGetSize"),
            Some(("SamplerPool", "GetSize"))
        );
        assert_eq!(
            split_name("nvnProgramBuilderSetShaders"),
            Some(("ProgramBuilder", "SetShaders"))
        );
    }
}
