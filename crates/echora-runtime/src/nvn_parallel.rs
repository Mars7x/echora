//! Parallel draw planning (ECHORA_PARALLEL_DRAWS=N, off by default).
//!
//! The replay records draws one at a time on its own thread, and most of a
//! draw's host time is planning: finding its bound program, resolving each
//! binding (uniform and storage buffers, driver constants, textures) from
//! the render state. That part reads state and changes none of it in the
//! common case, so it can run ahead of the replay on several threads:
//!
//! 1. Look-ahead (`prepare_parallel_draws`): at a draw no window covers,
//!    the replay looks down its command list, into the lists it calls and
//!    on through the lists queued after it, for a window of commands:
//!    everything up to a command that changes the render state another way
//!    (a pool set naming another pool, a deferred bind whose array GPU work
//!    may still write), a guest wait (WaitEvent, WaitSync: what follows may
//!    read what other work has yet to produce) or a queue item that is not
//!    a list (a present, a sync wait). A window's length adapts: halved
//!    after one whose plans were mostly unused, doubled after one whose
//!    were not, and planning pauses after several wasted in a row. Deferred binds' arrays are read for the window, and checked
//!    when they run (`check_parallel_deferred`). Other work in between
//!    (clears, copies, dispatches) leaves the render state as it is and is
//!    passed over. Plans outlive the replay's pieces. Why each window ended
//!    is counted ("parallel draw windows ended by ...").
//! 2. Plan (`plan_draw`, on `N` worker threads and the replay thread): the
//!    window's draws are split into one run per thread. Each thread starts
//!    from the render state at the window's start, applies the state
//!    commands before its draws as the replay does (`RenderState::
//!    apply_state`, shared with it), and plans each of its draws from the
//!    state that draw sees. The replay waits for all of them: while they
//!    run nothing changes the NVN state they read. Draws whose program the
//!    replay has not resolved yet get it resolved then, in draw order, from
//!    the state each draw meets (`resolve_ahead`), and are planned again.
//!    Textures are never resolved ahead: resolving one may upload it or
//!    create its image, which stays at its draw, in order.
//! 3. Commit (the replay, in order): each command runs as before; a draw
//!    with a plan takes its bindings from it (`take_planned_bindings`) in
//!    place of resolving them. What changes shared state stays in order:
//!    bindings that import shared spans or may wait for GPU work (storage
//!    buffers, guest uniform buffers) are resolved there by
//!    the replay's own code (`Deferred`), and programs, targets and
//!    recording are untouched.
//!
//! A plan holds only when what it read is unchanged at its commit. The
//! replay follows the window command by command (`follow`: each run of a
//! list is a segment of its own, planned from the state that run
//! inherits), and leaving it drops the window. The program bound must have
//! the plan's stages, each planned texture binding must still be the
//! memo's (checked as the binding loop checks it), and the window's stamp
//! (pools, present) must hold; uniform and storage buffers, which GPU work
//! between the plan and the draw may write, are resolved at the commit. A
//! draw that fails any of these, or that the planner could not handle (a
//! program never bound before, a texture to resolve anew),
//! is planned by the replay as before. ECHORA_PARALLEL_DRAWS_VERIFY=1
//! (debugging) plans every draw both ways and reports any difference.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use echora_gpu::draw::{Sampler, TextureSource};
use echora_gpu::shader::{self, DescriptorKind};

use super::super::{
    Command, Guest, GuestMemory, Nvn, PoolRange, QueueItem, RenderState, ResolvedBinding,
    StateSnapshots, is_state_command, is_work_command, plural_bind, pool_lookup, program_key,
    without_prefix,
};
use super::{
    BoundProgram, GuestFileSpan, bind_key, c0_storage_table, draw_switches_set, empty_draw,
    generic_input_types, nvn_stage, overlapping_spans_walked, shared_write_serial_in,
    typed_sampled_texture,
};

/// Threads planning draws besides the replay's (ECHORA_PARALLEL_DRAWS=N;
/// unset or 0: draws are planned by the replay alone, as before).
pub(in crate::nvn) fn threads() -> usize {
    #[cfg(test)]
    if let Some(threads) = testing::THREADS.with(std::cell::Cell::get) {
        return threads;
    }
    static THREADS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *THREADS.get_or_init(|| {
        std::env::var("ECHORA_PARALLEL_DRAWS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(0)
            .min(32)
    })
}

/// Whether draws are planned in parallel (`placement` leaves threads to
/// plan on), unless planning failed (`FAILED`).
pub(in crate::nvn) fn enabled() -> bool {
    placement().0 > 0 && !FAILED.load(Ordering::Relaxed)
}

/// The planning threads to start and the CPUs they may run on: only CPUs
/// the guest threads leave idle (`planner_placement`), whatever
/// ECHORA_PARALLEL_DRAWS asks for.
fn placement() -> (usize, Option<Vec<usize>>) {
    #[cfg(test)]
    if let Some(threads) = testing::THREADS.with(std::cell::Cell::get) {
        return (threads, None);
    }
    static PLACEMENT: std::sync::OnceLock<(usize, Option<Vec<usize>>)> = std::sync::OnceLock::new();
    PLACEMENT
        .get_or_init(|| {
            let requested = threads();
            if requested == 0 {
                return (0, None);
            }
            let reserved = crate::host_cores::reserved();
            let placed = planner_placement(
                requested,
                reserved
                    .as_ref()
                    .map(|reserved| (&reserved.kept[..], &reserved.others[..])),
                crate::bootstrap::worker_count(),
                std::thread::available_parallelism().map_or(1, usize::from),
            );
            eprintln!(
                "parallel draws: {} planning threads of {requested} asked for{}",
                placed.0,
                placed
                    .1
                    .as_ref()
                    .map(|cpus| format!(" on CPUs {cpus:?}"))
                    .unwrap_or_default()
            );
            placed
        })
        .clone()
}

/// Where planning threads go: the replay's own core (`host_cores`: its
/// sibling hardware threads are idle while it waits for its planners) and
/// the CPUs beyond those the `workers` guest threads take; without a
/// reserved core, the CPUs beyond the workers and the replay. The replay
/// plans a share itself, so one CPU of the set is its own.
fn planner_placement(
    requested: usize,
    reserved: Option<(&[usize], &[usize])>,
    workers: usize,
    available: usize,
) -> (usize, Option<Vec<usize>>) {
    match reserved {
        Some((kept, others)) => {
            let idle = others.len().saturating_sub(workers);
            let mut cpus = kept.to_vec();
            cpus.extend(others.iter().rev().take(idle));
            cpus.sort_unstable();
            let count = requested.min(cpus.len().saturating_sub(1));
            (count, (count > 0).then_some(cpus))
        }
        None => (requested.min(available.saturating_sub(workers + 1)), None),
    }
}

/// Queued lists the replay has yet to run past which new windows are made
/// (ECHORA_PARALLEL_DRAWS_BACKLOG, 4): with fewer it keeps up with the
/// game, and planning would only take CPU time from the game's threads.
fn backlog() -> usize {
    #[cfg(test)]
    {
        testing::BACKLOG.with(std::cell::Cell::get).unwrap_or(0)
    }
    #[cfg(not(test))]
    {
        static BACKLOG: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *BACKLOG.get_or_init(|| {
            std::env::var("ECHORA_PARALLEL_DRAWS_BACKLOG")
                .ok()
                .and_then(|value| value.trim().parse::<usize>().ok())
                .unwrap_or(4)
        })
    }
}

/// Set when a planning thread panicked: the replay then plans every draw
/// itself, as with the planner off.
static FAILED: AtomicBool = AtomicBool::new(false);

/// What planner code the replay is in (`Phase`), and since when (ms after
/// `phase_clock`): a monitor thread reports one that lasts.
static PHASE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
static PHASE_SINCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn phase_clock() -> u64 {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    START
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_millis() as u64
}

/// Planner code the replay runs.
#[derive(Clone, Copy)]
#[repr(u8)]
enum Phase {
    LookAhead = 1,
    Planning = 2,
    ResolvingAhead = 3,
    CheckingTextures = 4,
    ResolvingAtCommit = 5,
}

const PHASE_NAMES: [&str; 6] = [
    "",
    "the look-ahead",
    "planning (waiting for the planning threads)",
    "resolving programs ahead",
    "checking planned texture bindings at a draw",
    "resolving deferred bindings at a draw",
];

/// The replay is in `phase` until this is dropped (back to the phase it
/// was in).
struct InPhase(u8);

impl InPhase {
    fn enter(phase: Phase) -> Self {
        let previous = PHASE.swap(phase as u8, Ordering::AcqRel);
        PHASE_SINCE.store(phase_clock(), Ordering::Release);
        Self(previous)
    }
}

impl Drop for InPhase {
    fn drop(&mut self) {
        PHASE.store(self.0, Ordering::Release);
        PHASE_SINCE.store(phase_clock(), Ordering::Release);
    }
}

/// Report planner code the replay stays in for over 2 s, every 10 s while
/// it does (a stall's report says whether the planner holds the replay).
fn start_phase_monitor() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new()
            .name("nvn-plan-watch".into())
            .spawn(|| {
                let mut reported = (0, 0);
                loop {
                    std::thread::sleep(std::time::Duration::from_secs(1));
                    let phase = PHASE.load(Ordering::Acquire);
                    let since = PHASE_SINCE.load(Ordering::Acquire);
                    let held = phase_clock().saturating_sub(since);
                    if phase == 0 || held < 2000 {
                        continue;
                    }
                    if reported.0 == since && held < reported.1 + 10_000 {
                        continue;
                    }
                    reported = (since, held);
                    eprintln!(
                        "parallel draws: the replay has been in {} for {:.1} s",
                        PHASE_NAMES[usize::from(phase).min(PHASE_NAMES.len() - 1)],
                        held as f64 / 1000.0
                    );
                }
            });
    });
}

/// ECHORA_PARALLEL_DRAWS_VERIFY=1 (debugging): each planned draw is also
/// planned by the replay, which uses its own result and reports a plan
/// that differs.
pub(in crate::nvn) fn verify() -> bool {
    #[cfg(test)]
    if testing::VERIFY.with(std::cell::Cell::get) {
        return true;
    }
    static VERIFY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *VERIFY.get_or_init(|| {
        std::env::var_os("ECHORA_PARALLEL_DRAWS_VERIFY").is_some_and(|value| value != "0")
    })
}

/// The most commands one look-ahead window spans
/// (ECHORA_PARALLEL_DRAWS_WINDOW, 4096).
fn window_commands() -> usize {
    static WINDOW: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *WINDOW.get_or_init(|| {
        std::env::var("ECHORA_PARALLEL_DRAWS_WINDOW")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|&value| value >= 2)
            .unwrap_or(4096)
    })
}

/// Fewer draws than this in a window are not worth the hand-over.
const FEWEST_DRAWS: usize = 4;

/// Spare uniform buffers a draw keeps for the next: planned draws bring
/// their own, which would otherwise pile up (`gpu_draw`).
pub(super) const SPARE_UNIFORMS: usize = 256;

/// How the look-ahead treats a command.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Step {
    /// A draw the planner plans.
    Draw,
    /// Changes the render state as `RenderState::apply_state` does.
    State,
    /// A plural bind with its array captured.
    Plural(&'static str, usize),
    /// A plural bind with its array in guest memory, read when the window
    /// is made (`Window::deferred`) and checked when it runs.
    Deferred(&'static str, usize),
    /// Push{Graphics,Compute}Constants.
    Constants,
    /// SetTexturePool or SetSamplerPool: in a window only while it sets
    /// the pool already set (another moves every memoized binding).
    Pool,
    /// SetRenderTargets (the program's output types follow them).
    Targets,
    /// CallCommands or CopyCommands: the called lists join the window.
    Call,
    /// Changes nothing planning reads (other draws, barriers, viewports,
    /// render enables, other work): passed over.
    Pass,
    /// Anything else: the window ends before it.
    End,
}

/// Draws whose bindings the planner resolves (`gpu_draw` with no mesh).
fn planned_draw(rest: &str) -> bool {
    matches!(
        rest,
        "DrawArrays"
            | "DrawArraysInstanced"
            | "DrawElements"
            | "DrawElementsBaseVertex"
            | "DrawElementsInstanced"
    )
}

fn step(command: &Command) -> Step {
    let rest = without_prefix(command.name, "nvnCommandBuffer");
    if planned_draw(rest) {
        return if empty_draw(command) {
            Step::Pass
        } else {
            Step::Draw
        };
    }
    match rest {
        "SetRenderTargets" => return Step::Targets,
        // Handled by `execute_command`'s own arms, none of whose effects
        // planning reads.
        "SetViewports"
        | "SetScissors"
        | "SetScissor"
        | "SetDepthRanges"
        | "Barrier"
        | "SetRenderEnable"
        | "SetRenderEnableConditional" => return Step::Pass,
        "PushGraphicsConstants" | "PushComputeConstants" => return Step::Constants,
        "SetTexturePool" | "SetSamplerPool" => return Step::Pool,
        "CallCommands" | "CopyCommands" => return Step::Call,
        _ => {}
    }
    if rest.starts_with("Draw") || rest.starts_with("MultiDraw") {
        return Step::Pass;
    }
    if let Some((single, words)) = plural_bind(rest) {
        return if rest.ends_with("Deferred") {
            Step::Deferred(single, words)
        } else {
            Step::Plural(single, words)
        };
    }
    if is_state_command(rest) {
        return Step::State;
    }
    // Other work leaves the render state as it is; what it changes besides
    // (targets, GPU writes, waits) the commit's stamp check sees.
    if is_work_command(rest) || rest == "FenceSync" {
        return Step::Pass;
    }
    // A guest wait (WaitEvent, WaitSync) ends the window: what comes after
    // it may read what other work has yet to produce.
    Step::End
}

/// The render state `plan_draw` reads: programs, vertex attributes,
/// uniform, storage and texture bindings, the graphics root table.
fn plan_state(render: &RenderState) -> RenderState {
    let mut state = RenderState::default();
    for (key, command) in &render.state {
        if plan_key(key) {
            state.state.insert(key, command.clone());
        }
    }
    state.bound_uniforms = render.bound_uniforms.clone();
    state.bound_textures = render.bound_textures.clone();
    state.roots[0] = render.roots[0].clone();
    state
}

/// The render state keys `plan_draw` reads.
fn plan_key(key: &str) -> bool {
    key.starts_with("BindProgram@")
        || key == "BindVertexAttribState"
        || key.starts_with("BindStorageBuffer/")
}

/// Whether a look-ahead step changes what `plan_state` keeps.
fn plan_reads(command: &Command, step: Step) -> bool {
    let rest = without_prefix(command.name, "nvnCommandBuffer");
    match step {
        Step::State => matches!(
            rest,
            "BindProgram"
                | "BindVertexAttribState"
                | "BindUniformBuffer"
                | "BindTexture"
                | "BindStorageBuffer"
        ),
        Step::Plural(single, _) | Step::Deferred(single, _) => single != "BindImage",
        Step::Constants => rest == "PushGraphicsConstants",
        _ => false,
    }
}

/// Apply a look-ahead step's change to the render state (`deferred`: a
/// deferred bind's array, as the window read it).
fn apply(render: &mut RenderState, command: &Command, step: Step, deferred: Option<&[u64]>) {
    match step {
        Step::State | Step::Pool => {
            let rest = without_prefix(command.name, "nvnCommandBuffer");
            render.apply_state(rest, command);
        }
        Step::Plural(single, words) => {
            render.apply_plural_bind(single, words, command, &command.captured);
        }
        Step::Deferred(single, words) => {
            if let Some(values) = deferred {
                render.apply_plural_bind(single, words, command, values);
            }
        }
        Step::Constants => {
            render.push_constants(command);
        }
        Step::Targets => render.set_render_targets(command),
        Step::Draw | Step::Call | Step::Pass | Step::End => {}
    }
}

/// The state a window's plans read, as it was when they were made: a plan
/// is used only while its draw meets the same.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Stamp {
    /// A present ends a window anyway; plans read the memo of one present.
    presented_frames: u64,
    /// Memory pools (guest addresses of GPU ones).
    pool_epoch: u64,
}

/// A binding the commit resolves, in descriptor order, with the replay's
/// own code: what imports shared spans (which must stay in order) or may
/// wait for GPU work.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Deferred {
    /// A guest uniform buffer (`resolve_guest_uniform`: read in place, or
    /// made current after GPU work that writes it). `direct`: when planned,
    /// no GPU work could write it, so while the GPU writes recorded are
    /// those the plan saw (`WriteGeneration`) it is copied as it is with
    /// nothing to resolve.
    Uniform {
        key: (usize, u32),
        guest: u64,
        size: u64,
        direct: bool,
    },
    /// A storage buffer (`resolve_draw_storage`: shared in place or
    /// copied).
    Storage {
        key: (usize, u32),
        guest: u64,
        base: u64,
        length: u64,
        read_only: bool,
    },
}

impl Deferred {
    fn key(&self) -> (usize, u32) {
        match self {
            Self::Uniform { key, .. } | Self::Storage { key, .. } => *key,
        }
    }
}

/// A draw's bindings, resolved by the planner as the replay's binding
/// loop in `gpu_draw_with_bindings` resolves them.
#[derive(Debug)]
pub(super) struct PlannedBindings {
    pub program: Arc<BoundProgram>,
    pub uniforms: echora_gpu::FastMap<(usize, u32), Vec<u8>>,
    pub guest_uniforms: echora_gpu::FastMap<(usize, u32), (u64, usize)>,
    pub textures: echora_gpu::FastMap<(usize, u32), (TextureSource, Sampler)>,
    pub sampled_textures: [Vec<shader::SampledTexture>; 3],
    pub fragment_constants: Option<(usize, u32, usize)>,
    pub deferred: Vec<Deferred>,
    pub storage_written: super::super::FastMap<(usize, u32), (u64, u64)>,
    pub storage_entries: Vec<(usize, u32, u64, u64)>,
    pub uniform_bytes: u64,
    pub memo_hits: u64,
    /// The texture bindings planned: handle, binding and sample type, each
    /// checked against the memo when the draw takes the plan.
    pub texture_handles: Vec<(u64, (usize, u32), shader::SampleType)>,
    /// The guest memory flushes the plan saw (`Nvn::flush_epoch`): after
    /// another, the stage constants are read again (`stage_constants`).
    pub flush_epoch: u64,
    /// The stage constant buffers (c1) read, with their stage index.
    pub stage_constants: Vec<((usize, u32), usize)>,
    /// The GPU writes recorded when the plan was made (`Deferred::Uniform`).
    pub write_generation: WriteGeneration,
}

/// What `resolve_guest_uniform` and `resolve_storage` check of the GPU's
/// recorded writes: unchanged, a range they found nothing to wait for or
/// bring back still has nothing (finished work only shrinks them).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct WriteGeneration {
    gpu_writes: u64,
    in_place_writes: (usize, u64, u64),
    pending_storage: (usize, u64, u64),
    resident: (u64, usize),
    page_table: (u64, u64, usize, u64),
}

/// What a planner thread reads: shared views of the NVN state, which
/// nothing changes while the threads run.
struct View<'a> {
    memory: &'a echora_guest::view::GuestView,
    programs: Option<
        &'a super::super::FastMap<
            (u64, u64, [u8; super::super::MAX_COLOR_TARGETS]),
            (StateSnapshots, Arc<BoundProgram>),
        >,
    >,
    recent: &'a super::super::FastMap<ProgramKey, (StateSnapshots, Arc<BoundProgram>)>,
    typed: &'a super::super::FastMap<TypedKey, Arc<BoundProgram>>,
    pools: &'a [PoolRange],
    memo: &'a super::super::FastMap<u64, ResolvedBinding>,
    block_aliases: &'a super::super::FastMap<u64, u64>,
    presented_frames: u64,
    flush_epoch: u64,
    gpu_writes: u64,
    fragment_jump_stages: &'a super::super::FastSet<usize>,
    in_place_writes: &'a [std::ops::Range<u64>],
    pending_storage: &'a [(u64, std::ops::Range<u64>, u32, u64, u64)],
    resident_dirty: &'a [std::ops::Range<u64>],
    /// Page-table writes not yet finished (`draw_input_write_serial`).
    table_writes: Option<TableWrites<'a>>,
    write_generation: WriteGeneration,
}

struct TableWrites<'a> {
    spans: &'a [GuestFileSpan],
    shared_writes: &'a VecDeque<(u64, super::BoundedWrites)>,
    unbounded_write_serial: u64,
    completed: u64,
}

impl View<'_> {
    fn gpu_to_guest(&self, address: u64, size: u64) -> Option<u64> {
        pool_lookup(self.pools, address)
            .filter(|(_, remaining)| size <= *remaining)
            .map(|(guest, _)| guest)
    }

    /// `Nvn::gpu_writes_pending`: what `resolve_storage` would bring back.
    fn gpu_writes_pending(&self, range: &std::ops::Range<u64>) -> bool {
        let overlaps =
            |written: &std::ops::Range<u64>| written.start < range.end && range.start < written.end;
        self.in_place_writes.iter().any(overlaps)
            || self
                .pending_storage
                .iter()
                .any(|(_, guest, ..)| overlaps(guest))
            || super::ranges_overlap(self.resident_dirty, range)
    }

    /// Whether GPU work not yet finished may store into `range` through
    /// the page table (`draw_input_write_serial` above the completed
    /// serial): the replay then reads the input in place or waits for it.
    fn table_write_pending(&self, range: std::ops::Range<u64>) -> bool {
        let Some(writes) = &self.table_writes else {
            return false;
        };
        overlapping_spans_walked(writes.spans, range, || {})
            .map(|span| {
                shared_write_serial_in(
                    writes.shared_writes,
                    writes.unbounded_write_serial,
                    writes.completed,
                    span,
                )
            })
            .max()
            .is_some_and(|serial| serial > writes.completed)
    }

    /// The bound program as `Nvn::bound_program` memoized it for this
    /// state. Several entries for the same stages (other target formats)
    /// are not told apart here: the commit checks the program it binds.
    fn program(&self, render: &RenderState) -> Result<Arc<BoundProgram>, &'static str> {
        let (key, snapshots) = program_key_of(render).ok_or("no program bound")?;
        let (vertex, fragment, attributes) = key;
        // The program the same binds bound last, with these attributes or
        // others giving the same input types.
        if let Some((_, bound)) = self.recent.get(&key) {
            return Ok(bound.clone());
        }
        if let Some(bound) = self.typed.get(&typed_key(key, &snapshots)) {
            return Ok(bound.clone());
        }
        let programs = self.programs.ok_or("program not memoized")?;
        let mut found: Option<&Arc<BoundProgram>> = None;
        for (key, (memo_attributes, bound)) in programs {
            let memo_address = memo_attributes
                .0
                .as_ref()
                .map_or(0, |snapshots| Arc::as_ptr(snapshots) as usize);
            if key.0 != vertex || key.1 != fragment || memo_address != attributes {
                continue;
            }
            if found.is_some_and(|known| !Arc::ptr_eq(known, bound)) {
                return Err("program memoized for several target formats");
            }
            found = Some(bound);
        }
        found.cloned().ok_or("program not memoized")
    }

    /// `with_memoized_binding` for an entry from another present or GPU
    /// write generation: whether it holds (without marking it checked).
    fn memo_still_holds(&self, entry: &ResolvedBinding) -> bool {
        if entry.present != self.presented_frames {
            let current = match &entry.source {
                TextureSource::Target(key) | TextureSource::TargetView { key, .. } => {
                    !self.block_aliases.contains_key(key)
                }
                _ => false,
            };
            if !current {
                return false;
            }
        }
        !(entry.gpu_writes != self.gpu_writes
            && entry.backing.as_ref().is_some_and(|backing| {
                backing
                    .pieces()
                    .iter()
                    .any(|(_, range)| self.gpu_writes_pending(range))
            }))
    }

    /// `Nvn::read_stage_constants`.
    fn stage_constants(&self, stage: super::super::Args, bytes: &mut Vec<u8>) -> Option<()> {
        let (code, control) = (stage.x[0], stage.x[1]);
        let size = self.memory.read_u32(control + 0x3c).ok()?;
        let offset = self.memory.read_u32(control + 0x40).ok()?;
        if size == 0 || size > 0x10000 {
            return None;
        }
        let guest = self.gpu_to_guest(code + u64::from(offset), u64::from(size))?;
        bytes.resize(size as usize, 0);
        self.memory.read(guest, bytes).ok()
    }
}

/// Plan a draw's bindings from the render state it meets, as the binding
/// loop of `gpu_draw_with_bindings` resolves them; `Err` names why the
/// replay must resolve them itself.
fn plan_draw(view: &View<'_>, render: &RenderState) -> Result<PlannedBindings, &'static str> {
    let program = view.program(render)?;
    let mut plan = PlannedBindings {
        program: program.clone(),
        uniforms: Default::default(),
        guest_uniforms: Default::default(),
        textures: Default::default(),
        sampled_textures: Default::default(),
        fragment_constants: None,
        deferred: Vec::new(),
        storage_written: Default::default(),
        storage_entries: Vec::new(),
        uniform_bytes: 0,
        memo_hits: 0,
        texture_handles: Vec::new(),
        flush_epoch: view.flush_epoch,
        stage_constants: Vec::new(),
        write_generation: view.write_generation,
    };
    plan.uniforms.reserve(program.uniforms);
    plan.guest_uniforms.reserve(program.uniforms);
    plan.textures.reserve(program.textures);
    for (index, stage) in program.stages.iter().enumerate() {
        let nvn = nvn_stage(stage.stage);
        for descriptor in &stage.descriptors {
            let key = (index, descriptor.binding);
            match descriptor.kind() {
                DescriptorKind::UniformBuffer if descriptor.index == 24 => {
                    let words = render.roots[0].first().copied().unwrap_or([0; 64]);
                    let bytes = words.iter().flat_map(|word| word.to_le_bytes()).collect();
                    plan.uniforms.insert(key, bytes);
                }
                DescriptorKind::UniformBuffer if descriptor.index >= 3 => {
                    let slot = descriptor.index - 3;
                    let bind = render
                        .bound_uniforms
                        .get(u64::from(nvn), u64::from(slot))
                        .ok_or("uniform buffer not bound")?;
                    let used = u64::from(stage.cbuf_used_sizes[descriptor.index as usize]);
                    let size = bind.x[4].min(used.max(16)).min(64 << 10);
                    let guest = view
                        .gpu_to_guest(bind.x[3], size)
                        .ok_or("uniform buffer in no pool")?;
                    // Nothing to resolve when no recorded GPU work may write
                    // it (`resolve_guest_uniform` would copy it as it is);
                    // the commit checks no GPU writes were recorded since.
                    let direct = guest.checked_add(size).is_some_and(|end| {
                        size == 0
                            || !view.table_write_pending(guest..end)
                                && !view.gpu_writes_pending(&(guest..end))
                    });
                    plan.deferred.push(Deferred::Uniform {
                        key,
                        guest,
                        size,
                        direct,
                    });
                }
                DescriptorKind::UniformBuffer if descriptor.index == 1 => {
                    let mut bytes = Vec::new();
                    if view
                        .stage_constants(program.entries[index].0, &mut bytes)
                        .is_none()
                    {
                        bytes.clear();
                    }
                    if stage.stage == shader::Stage::Fragment
                        && view
                            .fragment_jump_stages
                            .contains(&(Arc::as_ptr(stage) as usize))
                    {
                        plan.fragment_constants = Some((index, descriptor.binding, bytes.len()));
                    }
                    bytes.resize(bytes.len().max(16).next_multiple_of(16), 0);
                    plan.uniforms.insert(key, bytes);
                    plan.stage_constants.push((key, index));
                }
                DescriptorKind::UniformBuffer => {
                    let used = stage.cbuf_used_sizes[descriptor.index as usize].max(16);
                    plan.uniforms.insert(key, vec![0; used as usize]);
                }
                DescriptorKind::StorageBuffer
                    if descriptor.cbuf_index == 0
                        && descriptor.cbuf_offset >= c0_storage_table(nvn)
                        && descriptor.cbuf_offset < c0_storage_table(nvn) + 0x100
                        && descriptor.cbuf_offset % 16 == 0 =>
                {
                    let slot = (descriptor.cbuf_offset - c0_storage_table(nvn)) / 16;
                    let bind = render
                        .state
                        .get(bind_key("BindStorageBuffer", Some(nvn as u64), slot as u64))
                        .ok_or("storage buffer not bound")?;
                    let (gpu, size) = (bind.x[3], bind.x[4]);
                    let base = gpu & !15;
                    let length = gpu - base + size;
                    let guest = view
                        .gpu_to_guest(base, length)
                        .ok_or("storage buffer in no pool")?;
                    if descriptor.flags & 1 != 0 {
                        plan.storage_written.insert(key, (guest, base));
                    }
                    plan.storage_entries
                        .push((index, descriptor.cbuf_offset, gpu, size));
                    plan.deferred.push(Deferred::Storage {
                        key,
                        guest,
                        base,
                        length,
                        read_only: descriptor.flags & 1 == 0,
                    });
                }
                DescriptorKind::Texture | DescriptorKind::TextureBuffer
                    if descriptor.cbuf_index == 2 && descriptor.cbuf_offset >= 0x20 =>
                {
                    let slot = (descriptor.cbuf_offset - 0x20) / 8;
                    let &handle = render
                        .bound_textures
                        .get(u64::from(nvn), u64::from(slot))
                        .ok_or("texture not bound")?;
                    // The memo as `with_memoized_binding` reads it: an entry
                    // checked at this present and GPU write generation, or
                    // one that still holds; others the replay resolves.
                    let entry = view
                        .memo
                        .get(&handle)
                        .ok_or("texture binding not memoized")?;
                    if entry.present != view.presented_frames || entry.gpu_writes != view.gpu_writes
                    {
                        // Checked again as `with_memoized_binding` does;
                        // the commit marks it checked (`install`).
                        if !view.memo_still_holds(entry) {
                            return Err("texture binding to resolve again");
                        }
                    }
                    plan.memo_hits += 1;
                    plan.texture_handles.push((handle, key, entry.sample_type));
                    if let Some(sampled) = typed_sampled_texture(descriptor, entry.sample_type) {
                        plan.sampled_textures[index].push(sampled);
                    }
                    plan.textures
                        .insert(key, (entry.source.clone(), entry.sampler));
                }
                _ => return Err("binding the replay reports"),
            }
        }
    }
    Ok(plan)
}

type PlanResult = Result<PlannedBindings, &'static str>;

/// The parallel planner's state in `Nvn`.
#[derive(Default)]
pub(in crate::nvn) struct ParallelDraws {
    pool: Option<Pool>,
    /// The command being executed: its list (by address) and index.
    pub position: Option<(usize, usize)>,
    /// Its place in the window (`follow`), when it is in it.
    current: Option<(usize, usize)>,
    /// The queue whose lists the plans cover.
    queue: Option<u64>,
    /// The lists the plans cover, in replay order, each with the command
    /// range (end exclusive). Held, so no other list takes their address.
    segments: Vec<(Arc<Vec<Command>>, usize, usize)>,
    /// Deferred binds' arrays as the window read them, by (segment,
    /// index): their commit checks it reads the same.
    deferred: super::super::FastMap<(usize, usize), Vec<u64>>,
    /// The window's command the replay is at, by (segment, index): the
    /// replay follows the window's segments one command after another
    /// (each run of a list its own segment).
    cursor: Option<(usize, usize)>,
    stamp: Option<Stamp>,
    /// Plans by (segment, command index), in replay order.
    plans: VecDeque<((usize, usize), PlanResult)>,
    /// The program each draw bound, by its stages' binds and vertex
    /// attribute snapshots (`ProgramKey`), and by those binds and the
    /// attributes' input types (`TypedKey`): what later plans of the same
    /// binds plan with (the commit checks it is still the one bound).
    programs: super::super::FastMap<ProgramKey, (StateSnapshots, Arc<BoundProgram>)>,
    typed_programs: super::super::FastMap<TypedKey, Arc<BoundProgram>>,
    /// The current window's size limit (`window_commands` at most): halved
    /// after a window whose plans were mostly not used, doubled after one
    /// whose were.
    limit: usize,
    /// Plans used in the current window, and how many it made.
    used: usize,
    planned: usize,
    /// Windows in a row whose plans were mostly not used; after
    /// `WASTED_WINDOWS` planning pauses for `PAUSE_DRAWS` draws.
    wasted: usize,
    paused: usize,
    /// Where the last window started, and how many windows in a row
    /// started there (a window made again and again at one draw).
    start: Option<(usize, usize)>,
    restarts: usize,
    /// Plans the last window used (`restarts` counts windows that used
    /// none).
    last_used: usize,
}

/// Wasted windows in a row before planning pauses.
const WASTED_WINDOWS: usize = 3;
/// Draws the replay plans alone after `WASTED_WINDOWS`.
const PAUSE_DRAWS: usize = 256;
/// The smallest window limit, in commands.
const SMALLEST_WINDOW: usize = 64;
/// Windows started at the same draw before that draw is left unplanned.
const RESTARTS: usize = 2;
/// Draws one window resolves ahead (`resolve_ahead`) at most.
const RESOLVED_AHEAD: usize = 64;
/// A window taking this long is reported.
const SLOW_WINDOW: std::time::Duration = std::time::Duration::from_millis(50);

/// The vertex and fragment programs bound and the vertex attribute
/// snapshots' address (kept alive beside the key).
type ProgramKey = (u64, u64, usize);

/// The vertex and fragment programs bound and the generic input types
/// their attributes give (what the program's translation depends on).
type TypedKey = (u64, u64, [u8; 32]);

/// Programs kept for planning before they are forgotten all at once.
const PROGRAMS_KEPT: usize = 4096;

fn program_key_of(render: &RenderState) -> Option<(ProgramKey, StateSnapshots)> {
    let bind = |bit: u32| {
        render
            .state
            .get(program_key(bit))
            .map(|command| command.x[1])
    };
    let attributes = render
        .state
        .get("BindVertexAttribState")
        .map(|command| command.states.clone())
        .unwrap_or_default();
    let address = attributes
        .0
        .as_ref()
        .map_or(0, |snapshots| Arc::as_ptr(snapshots) as usize);
    Some(((bind(0)?, bind(1)?, address), attributes))
}

/// `TypedKey` of a `ProgramKey`'s binds and attributes.
fn typed_key((vertex, fragment, _): ProgramKey, attributes: &StateSnapshots) -> TypedKey {
    (vertex, fragment, generic_input_types(attributes))
}

/// Whether bindings planned for `planned` hold for `bound`: the same
/// stages and stage entries (what the binding loop reads).
fn same_program(planned: &BoundProgram, bound: &BoundProgram) -> bool {
    std::ptr::eq(planned, bound)
        || planned.keys == bound.keys
            && planned.stages.len() == bound.stages.len()
            && planned
                .stages
                .iter()
                .zip(bound.stages.iter())
                .all(|(a, b)| Arc::ptr_eq(a, b))
            && planned.entries.len() == bound.entries.len()
            && planned
                .entries
                .iter()
                .zip(bound.entries.iter())
                .all(|(a, b)| a.0 == b.0)
}

impl std::fmt::Debug for ParallelDraws {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ParallelDraws")
            .field("segments", &self.segments.len())
            .field("plans", &self.plans.len())
            .finish()
    }
}

impl ParallelDraws {
    /// Drop every plan; how many were never used. The window's use sets
    /// the next one's limit (`limit`, `wasted`, `paused`).
    pub fn forget(&mut self) -> usize {
        let unused = self.plans.len();
        if self.planned > 0 {
            let most = window_commands();
            if self.limit == 0 {
                self.limit = most;
            }
            if self.used * 4 < self.planned {
                self.limit = (self.limit / 2).max(SMALLEST_WINDOW.min(most));
                self.wasted += 1;
                if self.wasted >= WASTED_WINDOWS {
                    self.wasted = 0;
                    self.paused = PAUSE_DRAWS;
                }
            } else {
                self.limit = (self.limit * 2).min(most);
                self.wasted = 0;
            }
        }
        self.last_used = self.used;
        self.used = 0;
        self.planned = 0;
        self.queue = None;
        self.segments.clear();
        self.deferred.clear();
        self.cursor = None;
        self.stamp = None;
        self.plans.clear();
        self.position = None;
        self.current = None;
        unused
    }

    /// Follow the replay to `list[index]` (its address): the window's
    /// command it is at, when it is the one at the cursor (again: a command
    /// begun once more) or the next.
    fn follow(&mut self, address: usize, index: usize) -> Option<(usize, usize)> {
        let (segment, at) = self.cursor?;
        let is = |segment: usize, at: usize| {
            self.segments
                .get(segment)
                .is_some_and(|(list, start, end)| {
                    Arc::as_ptr(list) as usize == address
                        && (*start..*end).contains(&at)
                        && at == index
                })
        };
        if is(segment, at) {
            return Some((segment, at));
        }
        let next = if at + 1 < self.segments[segment].2 {
            (segment, at + 1)
        } else {
            (segment + 1, self.segments.get(segment + 1)?.1)
        };
        if is(next.0, next.1) {
            self.cursor = Some(next);
            return Some(next);
        }
        None
    }

    /// Pop the plans before `key` (draws not drawn: render disabled,
    /// stopped); whether the next one is `key`'s, and how many were popped.
    fn reach(&mut self, key: (usize, usize)) -> (bool, u64) {
        let mut popped = 0;
        while self
            .plans
            .front()
            .is_some_and(|(planned, _)| *planned < key)
        {
            self.plans.pop_front();
            popped += 1;
        }
        let found = self
            .plans
            .front()
            .is_some_and(|(planned, _)| *planned == key);
        (found, popped)
    }

    /// Remember the program a draw bound.
    fn remember_program(&mut self, render: &RenderState, program: &Arc<BoundProgram>) {
        // Vertex and fragment programs only (a mesh draw binds others).
        if program
            .stages
            .first()
            .is_none_or(|stage| stage.stage != shader::Stage::Vertex)
        {
            return;
        }
        let Some((key, attributes)) = program_key_of(render) else {
            return;
        };
        if self
            .programs
            .get(&key)
            .is_some_and(|(_, known)| Arc::ptr_eq(known, program))
        {
            return;
        }
        if self.programs.len() >= PROGRAMS_KEPT {
            self.programs.clear();
            self.typed_programs.clear();
        }
        self.typed_programs
            .insert(typed_key(key, &attributes), program.clone());
        self.programs.insert(key, (attributes, program.clone()));
    }
}

/// Why a look-ahead window ended (counted per reason).
fn window_end_stat(reason: &str) -> &'static str {
    thread_local! {
        static NAMES: std::cell::RefCell<super::super::FastMap<String, &'static str>> =
            std::cell::RefCell::new(Default::default());
    }
    NAMES.with(|names| {
        let mut names = names.borrow_mut();
        if let Some(known) = names.get(reason) {
            return *known;
        }
        let name: &'static str =
            Box::leak(format!("stat: parallel draw windows ended by {reason}").into_boxed_str());
        names.insert(reason.to_string(), name);
        name
    })
}

/// A look-ahead window as it is made.
#[derive(Default)]
struct Window {
    segments: Vec<(Arc<Vec<Command>>, usize, usize)>,
    draws: Vec<(usize, usize)>,
    deferred: super::super::FastMap<(usize, usize), Vec<u64>>,
    walked: usize,
    /// The texture and sampler pools set (`Step::Pool`).
    pools: [Option<u64>; 2],
    /// Why it ended (`None`: it may go on).
    ended: Option<String>,
    /// The most commands it may span (`ParallelDraws::limit`).
    limit: usize,
}

impl Nvn {
    /// The GPU writes recorded so far, as `WriteGeneration` compares them.
    fn write_generation(&self) -> WriteGeneration {
        let Some(backend) = self.gpu.as_ref() else {
            return WriteGeneration::default();
        };
        let ends = |ranges: &mut dyn Iterator<Item = &std::ops::Range<u64>>| {
            let mut count = 0;
            let mut first = 0;
            let mut last = 0;
            for range in ranges {
                if count == 0 {
                    first = range.start;
                }
                last = range.end;
                count += 1;
            }
            (count, first, last)
        };
        WriteGeneration {
            gpu_writes: backend.gpu_writes,
            in_place_writes: ends(&mut backend.in_place_writes.iter()),
            pending_storage: ends(&mut backend.pending_storage.iter().map(|(_, range, ..)| range)),
            resident: (
                backend.dirty_removals,
                backend
                    .resident
                    .values()
                    .map(|pool| pool.gpu_dirty.len())
                    .sum(),
            ),
            page_table: (
                backend.page_table_write_serial,
                backend.unbounded_write_serial,
                backend.shared_writes.len(),
                backend
                    .shared_writes
                    .back()
                    .map_or(0, |(serial, _)| *serial),
            ),
        }
    }

    /// The renderer's texture uploads, images and targets made so far.
    fn renderer_made(&self) -> [u64; 3] {
        self.gpu.as_ref().map_or([0; 3], |backend| {
            let renderer = &backend.renderer;
            [
                renderer.texture_uploads,
                renderer.images_created.get(),
                renderer.targets_created,
            ]
        })
    }

    /// What plans depend on besides what the commit checks itself
    /// (programs, texture bindings, and the bindings it resolves).
    fn parallel_stamp(&self) -> Option<Stamp> {
        self.gpu.as_ref()?;
        Some(Stamp {
            presented_frames: self.presented_frames,
            pool_epoch: self.pool_epoch,
        })
    }

    /// Before the replay executes `list[index]` of `queue`: when it is a
    /// draw no window covers, make the look-ahead window from there and
    /// plan its draws on other threads. The window runs on through the
    /// lists it calls and the queue's next lists, up to a command that
    /// changes the render state another way (`Step::End`, a pool change, a
    /// deferred bind GPU work writes), a queue item other than a list or
    /// fence, or `window_commands`.
    pub(in crate::nvn) fn prepare_parallel_draws(
        &mut self,
        queue: u64,
        list: &Arc<Vec<Command>>,
        index: usize,
        core: &mut Guest,
    ) {
        let address = Arc::as_ptr(list) as usize;
        self.parallel.position = Some((address, index));
        self.parallel.current = None;
        if self.parallel.queue == Some(queue) && self.parallel.cursor.is_some() {
            if let Some(key) = self.parallel.follow(address, index) {
                // On through the window (planned or too short to plan).
                self.parallel.current = Some(key);
                return;
            }
            // Off the window: past its end, or the replay went another way.
            let finished = self.parallel.cursor.is_some_and(|(segment, at)| {
                segment + 1 == self.parallel.segments.len()
                    && at + 1 == self.parallel.segments[segment].2
            });
            let unused = self.parallel.forget() as u64;
            self.parallel.position = Some((address, index));
            if let Some(backend) = self.gpu.as_mut() {
                backend.stat(
                    if finished {
                        "stat: parallel draw windows run to their end"
                    } else {
                        "stat: parallel draw windows left by the replay"
                    },
                    1,
                );
                if unused > 0 {
                    backend.stat("stat: parallel plans dropped unused", unused);
                }
            }
        }
        let Some(command) = list.get(index) else {
            return;
        };
        if step(command) != Step::Draw {
            return;
        }
        let paused_before = self.parallel.paused;
        let unused = self.parallel.forget();
        self.parallel.position = Some((address, index));
        if self.parallel.paused > 0 {
            self.parallel.paused -= 1;
            if paused_before == 0
                && let Some(backend) = self.gpu.as_mut()
            {
                backend.stat("stat: parallel planning pauses after wasted windows", 1);
            }
            return;
        }
        // A window made again and again at the same draw (its plans
        // dropped at once each time) is not made again.
        let start = Some((address, index));
        if self.parallel.start == start && self.parallel.last_used == 0 {
            self.parallel.restarts += 1;
            if self.parallel.restarts >= RESTARTS {
                // Planned by the replay this time; tried again after.
                self.parallel.restarts = 0;
                if let Some(backend) = self.gpu.as_mut() {
                    backend.stat("stat: parallel draw windows not made again at a draw", 1);
                }
                return;
            }
        } else {
            self.parallel.start = start;
            self.parallel.restarts = 0;
        }
        if self.gpu.is_none()
            || draw_switches_set()
            || matches!(core.memory, GuestMemory::Checked(_))
        {
            return;
        }
        if unused > 0 {
            self.gpu
                .as_mut()
                .unwrap()
                .stat("stat: parallel plans dropped unused", unused as u64);
        }
        // Other modes of these change which bindings are read in place.
        static PLANNABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*PLANNABLE.get_or_init(|| {
            super::super::debug_var_os("ECHORA_UNIFORMS_IN_PLACE").is_none_or(|mode| mode != "all")
        }) {
            return;
        }
        start_phase_monitor();
        let walk_started = std::time::Instant::now();
        let made_before = self.renderer_made();
        let look_ahead = InPhase::enter(Phase::LookAhead);
        // Only while the replay is behind the game (work queued for it).
        let queued: usize = self
            .pending
            .values()
            .map(|items| {
                items
                    .iter()
                    .filter(|item| matches!(item, QueueItem::List { .. }))
                    .count()
            })
            .sum();
        if queued < backlog() {
            if let Some(backend) = self.gpu.as_mut() {
                backend.stat(
                    "stat: parallel draw windows skipped: the replay keeps up",
                    1,
                );
            }
            return;
        }
        let pool_set = |name| self.render.state.get(name).map(|set| set.x[1]);
        if self.parallel.limit == 0 {
            self.parallel.limit = window_commands();
        }
        let mut window = Window {
            pools: [pool_set("SetTexturePool"), pool_set("SetSamplerPool")],
            limit: self.parallel.limit,
            ..Window::default()
        };
        if self.walk_window(&mut window, list, index, 0, core) {
            // The first item is the list being replayed.
            for item in self.pending.get(&queue).into_iter().flatten().skip(1) {
                match item {
                    QueueItem::List { commands, next, .. } => {
                        if *next >= commands.len() {
                            continue;
                        }
                        if !self.walk_window(&mut window, commands, *next, 0, core) {
                            break;
                        }
                    }
                    QueueItem::Fence { .. } => {}
                    QueueItem::Present { .. } => {
                        window.ended = Some("a present".into());
                        break;
                    }
                    QueueItem::WaitSync { .. } => {
                        window.ended = Some("a sync wait".into());
                        break;
                    }
                }
            }
        }
        let reason = window
            .ended
            .take()
            .unwrap_or_else(|| "the end of the queued work".into());
        let Window {
            mut segments,
            draws,
            deferred,
            ..
        } = window;
        if segments.is_empty() {
            segments.push((list.clone(), index, index + 1));
        }
        self.parallel.queue = Some(queue);
        self.parallel.segments = segments;
        self.parallel.deferred = deferred;
        self.parallel.cursor = Some((0, index));
        self.parallel.current = Some((0, index));
        let backend = self.gpu.as_mut().unwrap();
        backend.stat(window_end_stat(&reason), 1);
        if draws.len() < FEWEST_DRAWS {
            backend.stat("stat: parallel draw windows too short", 1);
            backend.stat(
                "stat: parallel draws in windows too short",
                draws.len() as u64,
            );
            return;
        }
        drop(look_ahead);
        let started = std::time::Instant::now();
        let walk_time = started - walk_started;
        // What the binding loop would do at its first texture first: the
        // planners then read the memo as the replay would.
        self.refresh_binding_memo();
        if let Some(backend) = self.gpu.as_ref() {
            // Built now: the planners read it as it is.
            backend.resident_writes_pending(&(0..0));
        }
        let mut stamp = self.parallel_stamp();
        let pool = self
            .parallel
            .pool
            .take()
            .unwrap_or_else(|| Pool::new(placement()));
        let mut plans = self.plan_window(&pool, &draws, core);
        let planned_at = std::time::Instant::now();
        let mut resolve_time = std::time::Duration::ZERO;
        let mut replan_time = std::time::Duration::ZERO;
        // Draws whose program the replay has not resolved yet (a program
        // new since the last present or flush): the replay resolves them
        // now, in draw order, as their draws would, and they are planned
        // again.
        let unresolved: Vec<(usize, usize)> = plans
            .iter()
            .filter(|(_, plan)| matches!(plan, Err("program not memoized")))
            .map(|(key, _)| *key)
            .collect();
        let mut unresolved = unresolved;
        if unresolved.len() > RESOLVED_AHEAD {
            // The rest the replay resolves at their draws.
            unresolved.truncate(RESOLVED_AHEAD);
            self.gpu
                .as_mut()
                .unwrap()
                .stat("stat: parallel draw windows with more to resolve ahead", 1);
        }
        if !unresolved.is_empty() {
            let resolving = std::time::Instant::now();
            self.resolve_ahead(&unresolved, core);
            resolve_time = resolving.elapsed();
            let replanning = std::time::Instant::now();
            self.refresh_binding_memo();
            let now = self.parallel_stamp();
            let again = if now == stamp {
                self.plan_window(&pool, &unresolved, core)
            } else {
                // Resolving changed what the other plans read: all are
                // planned once more (never again for this window).
                stamp = now;
                self.gpu
                    .as_mut()
                    .unwrap()
                    .stat("stat: parallel draw windows planned again whole", 1);
                self.plan_window(&pool, &draws, core)
            };
            replan_time = replanning.elapsed();
            let mut again = again.into_iter().peekable();
            for (key, plan) in &mut plans {
                if again.peek().is_some_and(|(planned, _)| planned == key) {
                    *plan = again.next().unwrap().1;
                }
            }
            self.gpu.as_mut().unwrap().stat(
                "stat: parallel draws resolved ahead",
                unresolved.len() as u64,
            );
        }
        self.parallel.pool = Some(pool);
        self.parallel.planned = plans.len();
        self.parallel.plans = plans.into();
        self.parallel.stamp = stamp;
        let took = started.elapsed() + walk_time;
        let plan_time = planned_at - started;
        // Uploads and images made while the window was made: none are
        // meant to be (each stays at its draw, in order).
        let made_after = self.renderer_made();
        let made = [
            made_after[0].saturating_sub(made_before[0]),
            made_after[1].saturating_sub(made_before[1]),
            made_after[2].saturating_sub(made_before[2]),
        ];
        if took >= SLOW_WINDOW || made.iter().any(|&count| count > 0) {
            static REPORTED: AtomicUsize = AtomicUsize::new(0);
            if REPORTED.fetch_add(1, Ordering::Relaxed) < 20 {
                eprintln!(
                    "parallel draws: a window of {} draws in {} lists ({reason}) took {took:?}: look-ahead {walk_time:?}, planning {plan_time:?}, resolving {} programs ahead {resolve_time:?}, planning again {replan_time:?}; texture uploads {}, images {}, targets {} made meanwhile",
                    draws.len(),
                    self.parallel.segments.len(),
                    unresolved.len(),
                    made[0],
                    made[1],
                    made[2]
                );
            }
        }
        let backend = self.gpu.as_mut().unwrap();
        backend.stat("stat: parallel texture uploads while planning", made[0]);
        backend.stat("stat: parallel images created while planning", made[1]);
        backend.stat("stat: parallel targets created while planning", made[2]);
        backend.stat(
            "stat: parallel draw look-ahead us",
            walk_time.as_micros() as u64,
        );
        backend.stat(
            "stat: parallel draw resolving ahead us",
            resolve_time.as_micros() as u64,
        );
        backend.stat("stat: parallel draw windows", 1);
        backend.stat("stat: parallel draws planned", draws.len() as u64);
        backend.stat("stat: parallel draw planning us", took.as_micros() as u64);
    }

    /// Add `list` from `from` (and the lists it calls, in place) to
    /// `window`; false once the window ends.
    fn walk_window(
        &self,
        window: &mut Window,
        list: &Arc<Vec<Command>>,
        from: usize,
        depth: usize,
        core: &Guest,
    ) -> bool {
        let most = window.limit;
        let mut start = from;
        let mut at = from;
        while at < list.len() {
            if window.walked >= most {
                window.ended = Some("the window's length".into());
                break;
            }
            let command = &list[at];
            match step(command) {
                Step::End => {
                    window.ended = Some(without_prefix(command.name, "nvnCommandBuffer").into());
                    break;
                }
                Step::Draw => window.draws.push((window.segments.len(), at)),
                Step::Pool => {
                    let slot = usize::from(command.name.ends_with("SamplerPool"));
                    if window.pools[slot] != Some(command.x[1]) {
                        window.ended = Some("a pool change".into());
                        break;
                    }
                }
                Step::Deferred(_, words) => match self.deferred_values(command, words, core) {
                    Some(values) => {
                        window.deferred.insert((window.segments.len(), at), values);
                    }
                    None => {
                        window.ended = Some("a deferred bind GPU work writes".into());
                        break;
                    }
                },
                Step::Call => {
                    if depth >= 8 {
                        window.ended = Some("calls nested too deeply".into());
                        break;
                    }
                    // The calling list up to the call, the called lists, then
                    // the rest (the replay queues them that way; it runs the
                    // call itself before it follows the window, so the call
                    // is in no segment).
                    if at > start {
                        window.segments.push((list.clone(), start, at));
                    }
                    window.walked += 1;
                    at += 1;
                    start = at;
                    for handle in &command.captured {
                        let Some(called) = self
                            .command_lists
                            .get(handle)
                            .map(|called| called.commands.clone())
                        else {
                            window.ended = Some("a called list not at hand".into());
                            return false;
                        };
                        if !self.walk_window(window, &called, 0, depth + 1, core) {
                            return false;
                        }
                    }
                    continue;
                }
                _ => {}
            }
            window.walked += 1;
            at += 1;
        }
        if at > start {
            window.segments.push((list.clone(), start, at));
        }
        window.ended.is_none()
    }

    /// A deferred bind's array as its execution would read it, when no
    /// GPU work may still write it.
    fn deferred_values(&self, command: &Command, words: usize, core: &Guest) -> Option<Vec<u64>> {
        let count = (command.x[3] as usize).min(32);
        let address = command.x[4];
        let length = (count * words * 8) as u64;
        let end = address.checked_add(length)?;
        if length > 0 {
            let backend = self.gpu.as_ref()?;
            if self.gpu_writes_pending(&(address..end))
                || self
                    .draw_input_write_serial(address..end)
                    .is_some_and(|serial| serial > backend.renderer.completed_serial())
            {
                return None;
            }
        }
        (0..(count * words) as u64)
            .map(|index| core.memory.read_u64(address + index * 8).ok())
            .collect()
    }

    /// Plan the window's `wanted` draws (in replay order) on the pool's
    /// threads and this one.
    fn plan_window(
        &self,
        pool: &Pool,
        wanted: &[(usize, usize)],
        core: &Guest,
    ) -> Vec<((usize, usize), PlanResult)> {
        let threads = pool.threads.len();
        // A few draws a thread at least: each starts from the window's
        // state and applies what comes before its own draws.
        let chunks = (threads + 1)
            .min(wanted.len().div_ceil(FEWEST_DRAWS))
            .max(1);
        let results: Vec<Mutex<Vec<((usize, usize), PlanResult)>>> =
            (0..chunks).map(|_| Mutex::new(Vec::new())).collect();
        let program_stamp = (self.presented_frames, self.flush_epoch);
        let pools = self.indexed_pool_ranges();
        let backend = self.gpu.as_ref().unwrap();
        let dirty = backend.dirty_index.borrow();
        let completed = backend.renderer.completed_serial();
        let index_held = (backend.page_table_writes && backend.page_table_write_serial > completed)
            .then(|| self.shared_guest_index())
            .flatten();
        let memory = match &core.memory {
            GuestMemory::View(view) => &**view,
            // Checked reads report faults the view's do not.
            GuestMemory::Checked(_) => unreachable!("checked memory is not planned"),
        };
        let view = View {
            memory,
            programs: (self.program_memo.0 == program_stamp).then_some(&self.program_memo.1),
            recent: &self.parallel.programs,
            typed: &self.parallel.typed_programs,
            pools: &pools.entries,
            memo: &self.binding_memo.1,
            block_aliases: &self.block_aliases,
            presented_frames: self.presented_frames,
            flush_epoch: self.flush_epoch,
            gpu_writes: backend.gpu_writes,
            fragment_jump_stages: &backend.fragment_jump_stages,
            in_place_writes: &backend.in_place_writes,
            pending_storage: &backend.pending_storage,
            resident_dirty: if backend.resident.is_empty() {
                &[]
            } else {
                &dirty.1
            },
            table_writes: index_held.as_ref().map(|index| TableWrites {
                spans: &index.spans,
                shared_writes: &backend.shared_writes,
                unbounded_write_serial: backend.unbounded_write_serial,
                completed,
            }),
            write_generation: self.write_generation(),
        };
        // Only what planning reads of the render state, so each thread's
        // copy and the commands it applies are few.
        let render = &plan_state(&self.render);
        let results_ref = &results;
        let segments = &self.parallel.segments;
        let deferred = &self.parallel.deferred;
        let planning = InPhase::enter(Phase::Planning);
        let ran = pool.run(chunks, &|chunk| {
            // An even share of the draws, each from its own state.
            let from = wanted.len() * chunk / chunks;
            let to = wanted.len() * (chunk + 1) / chunks;
            let last = wanted[to - 1];
            let mut state = render.clone();
            let mut out = Vec::with_capacity(to - from);
            let mut next_draw = from;
            'segments: for (segment, (list, start, end)) in segments.iter().enumerate() {
                for at in *start..*end {
                    let key = (segment, at);
                    if key > last {
                        break 'segments;
                    }
                    let command = &list[at];
                    let kind = step(command);
                    if kind == Step::Draw {
                        if next_draw < to && wanted[next_draw] == key {
                            out.push((key, plan_draw(&view, &state)));
                            next_draw += 1;
                        }
                        continue;
                    }
                    if plan_reads(command, kind) {
                        apply(
                            &mut state,
                            command,
                            kind,
                            deferred.get(&key).map(Vec::as_slice),
                        );
                    }
                }
            }
            *results_ref[chunk]
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = out;
        });
        drop(planning);
        drop(view);
        if let Err(error) = ran {
            // A planner bug: the replay plans every draw itself from now on.
            if !FAILED.swap(true, Ordering::AcqRel) {
                eprintln!("parallel draws: {error}; draws are planned by the replay alone");
            }
            return Vec::new();
        }
        results
            .into_iter()
            .flat_map(|result| {
                result
                    .into_inner()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
            })
            .collect()
    }

    /// Resolve the programs and textures the window's `draws` (in replay
    /// order) bind, from the render state each meets, as their binding
    /// loops would.
    fn resolve_ahead(&mut self, draws: &[(usize, usize)], core: &mut Guest) {
        let _phase = InPhase::enter(Phase::ResolvingAhead);
        let mut render = self.render.clone();
        let segments = self.parallel.segments.clone();
        let mut draws = draws.iter().copied().peekable();
        'segments: for (segment, (list, start, end)) in segments.iter().enumerate() {
            for at in *start..*end {
                let Some(&next) = draws.peek() else {
                    break 'segments;
                };
                let key = (segment, at);
                let command = &list[at];
                let kind = step(command);
                if kind == Step::Draw {
                    if key == next {
                        draws.next();
                        std::mem::swap(&mut self.render, &mut render);
                        self.resolve_draw_ahead(core);
                        std::mem::swap(&mut self.render, &mut render);
                    }
                    continue;
                }
                let values = self.parallel.deferred.get(&key).cloned();
                apply(&mut render, command, kind, values.as_deref());
            }
        }
    }

    /// `resolve_ahead` for the draw the render state is at: its program
    /// only (translation on the CPU, as its draw would make it). Textures
    /// are left to the draw: resolving one may upload it or create its
    /// image, which must stay in draw order. Errors are left to the draw.
    fn resolve_draw_ahead(&mut self, core: &mut Guest) {
        if let Ok(program) = self.bound_program(core) {
            self.parallel.remember_program(&self.render, &program);
        }
    }

    /// The plan for the draw being executed, when one was made and the
    /// state it read still holds. Remembers the draw's program for later
    /// plans.
    pub(super) fn take_planned_bindings(
        &mut self,
        program: &Arc<BoundProgram>,
    ) -> Option<PlannedBindings> {
        if !enabled() {
            return None;
        }
        let started = super::draw_phase_timing().then(std::time::Instant::now);
        let plan = self.take_planned_bindings_untimed(program);
        if let Some(started) = started
            && plan.is_some()
        {
            self.gpu.as_mut().unwrap().stat(
                "stat: parallel draw commit ns",
                started.elapsed().as_nanos() as u64,
            );
        }
        plan
    }

    fn take_planned_bindings_untimed(
        &mut self,
        program: &Arc<BoundProgram>,
    ) -> Option<PlannedBindings> {
        self.parallel.remember_program(&self.render, program);
        if self.parallel.queue != self.render_queue {
            return None;
        }
        let key = self.parallel.current?;
        let (found, popped) = self.parallel.reach(key);
        if popped > 0 {
            self.gpu
                .as_mut()
                .unwrap()
                .stat("stat: parallel plans of draws not drawn", popped);
        }
        if !found {
            return None;
        }
        let (_, plan) = self.parallel.plans.pop_front().unwrap();
        let now = self.parallel_stamp();
        if let (Some(was), Some(now)) = (self.parallel.stamp, now)
            && was != now
        {
            // Something the plans read changed: the rest are made again.
            let dropped = 1 + self.parallel.forget() as u64;
            let backend = self.gpu.as_mut().unwrap();
            backend.stat("stat: parallel draw windows dropped by a state change", 1);
            backend.stat(
                "stat: parallel draw plans dropped by a state change",
                dropped,
            );
            if was.presented_frames != now.presented_frames {
                backend.stat("stat: parallel draw windows dropped by a present", 1);
            }
            if was.pool_epoch != now.pool_epoch {
                backend.stat("stat: parallel draw windows dropped by a pool change", 1);
            }
            return None;
        }
        // Each texture binding planned must still be the memo's, checked
        // as the binding loop checks it (marking it checked as it would).
        if let Ok(plan) = &plan
            && !plan.texture_handles.is_empty()
        {
            let _phase = InPhase::enter(Phase::CheckingTextures);
            self.refresh_binding_memo();
            for &(handle, key, sample_type) in &plan.texture_handles {
                let Some((source, sampler)) = plan.textures.get(&key) else {
                    continue;
                };
                let holds = self.with_memoized_binding(handle, |entry| {
                    entry.source.same_as(source)
                        && entry.sampler == *sampler
                        && entry.sample_type == sample_type
                });
                if holds != Some(true) {
                    self.gpu.as_mut().unwrap().stat(
                        "stat: parallel draws replanned: a texture binding changed",
                        1,
                    );
                    return None;
                }
            }
        }
        let backend = self.gpu.as_mut().unwrap();
        match plan {
            Ok(plan) if same_program(&plan.program, program) => {
                backend.stat("stat: parallel draws used", 1);
                self.parallel.used += 1;
                Some(plan)
            }
            Ok(_) => {
                backend.stat("stat: parallel draws replanned: another program", 1);
                None
            }
            Err(reason) => {
                backend.stat(fallback_stat(reason), 1);
                None
            }
        }
    }

    /// A deferred bind (at the replay's position) read `values`: plans
    /// made with others are dropped.
    pub(in crate::nvn) fn check_parallel_deferred(&mut self, values: &[u64]) {
        let Some(key) = self.parallel.current else {
            return;
        };
        if self
            .parallel
            .deferred
            .get(&key)
            .is_some_and(|expected| expected.as_slice() != values)
        {
            let dropped = self.parallel.forget() as u64;
            if let Some(backend) = self.gpu.as_mut() {
                backend.stat("stat: parallel draw windows dropped by a deferred bind", 1);
                backend.stat(
                    "stat: parallel draw plans dropped by a deferred bind",
                    dropped,
                );
            }
        }
    }

    /// Install a draw's planned bindings into the draw's maps, as the
    /// binding loop would have left them. The storage buffers' sharing is
    /// decided here, in order.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn install_planned_bindings(
        &mut self,
        core: &mut Guest,
        plan: PlannedBindings,
        out: InstallTo<'_>,
    ) -> Result<(), String> {
        let started = super::draw_phase_timing().then(std::time::Instant::now);
        let result = self.install_planned_bindings_untimed(core, plan, out);
        if let Some(started) = started {
            let backend = self.gpu.as_mut().unwrap();
            backend.stat(
                "stat: parallel draw commit ns",
                started.elapsed().as_nanos() as u64,
            );
            backend.stat("stat: parallel draws timed at commit", 1);
        }
        result
    }

    fn install_planned_bindings_untimed(
        &mut self,
        core: &mut Guest,
        plan: PlannedBindings,
        out: InstallTo<'_>,
    ) -> Result<(), String> {
        let PlannedBindings {
            program,
            uniforms,
            guest_uniforms,
            textures,
            sampled_textures,
            fragment_constants,
            deferred,
            storage_written,
            storage_entries,
            uniform_bytes,
            memo_hits,
            texture_handles: _,
            flush_epoch,
            stage_constants,
            write_generation,
        } = plan;
        let mut uniforms = uniforms;
        let mut fragment_constants = fragment_constants;
        if flush_epoch != self.flush_epoch {
            // Guest memory flushed since: the constants as they are now,
            // read as the binding loop reads them.
            for (key, index) in stage_constants {
                let mut bytes = uniforms.remove(&key).unwrap_or_default();
                bytes.clear();
                if self
                    .read_stage_constants(core, program.entries[index].0, &mut bytes)
                    .is_none()
                {
                    bytes.clear();
                }
                if let Some((stage, binding, size)) = fragment_constants.as_mut()
                    && (*stage, *binding) == key
                {
                    *size = bytes.len();
                }
                bytes.resize(bytes.len().max(16).next_multiple_of(16), 0);
                uniforms.insert(key, bytes);
            }
            self.gpu
                .as_mut()
                .unwrap()
                .stat("stat: parallel draw stage constants read again", 1);
        }
        out.uniforms.extend(uniforms);
        out.guest_uniforms.extend(guest_uniforms);
        out.textures.extend(textures);
        *out.sampled_textures = sampled_textures;
        *out.fragment_constants = fragment_constants;
        out.storage_written.extend(storage_written);
        out.storage_entries.extend(storage_entries);
        *out.uniform_bytes += uniform_bytes;
        *out.memo_hits += memo_hits;
        let _phase = InPhase::enter(Phase::ResolvingAtCommit);
        let unchanged = self.write_generation() == write_generation;
        if !unchanged {
            self.gpu.as_mut().unwrap().stat(
                "stat: parallel draws with GPU writes recorded since planned",
                1,
            );
        }
        for binding in deferred {
            match binding {
                Deferred::Uniform {
                    key,
                    guest,
                    size,
                    direct,
                } => {
                    if direct && unchanged {
                        *out.uniform_bytes += size;
                        out.guest_uniforms.insert(key, (guest, size as usize));
                        continue;
                    }
                    match self.resolve_guest_uniform(core, guest, size)? {
                        Some(in_place) => {
                            out.uniforms_in_place.insert(key, in_place);
                        }
                        None => {
                            *out.uniform_bytes += size;
                            out.guest_uniforms.insert(key, (guest, size as usize));
                        }
                    }
                }
                Deferred::Storage {
                    key,
                    guest,
                    base,
                    length,
                    read_only,
                } => {
                    match self.resolve_draw_storage(core, guest, base, length, read_only, false)? {
                        Ok(in_place) => {
                            out.storage_in_place.insert(key, in_place);
                        }
                        Err(bytes) => {
                            out.storage.insert(key, bytes);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// ECHORA_PARALLEL_DRAWS_VERIFY: compare a plan with what the replay's
    /// own binding loop resolved, reporting the first differences. Values
    /// are compared as they are, never formatted: an upload's source holds
    /// its texels (hundreds of MiB for the forest's texture arrays).
    pub(super) fn verify_planned_bindings(&mut self, plan: &PlannedBindings, got: Resolved<'_>) {
        let mut differences = Vec::new();
        if plan.uniforms.len() != got.uniforms.len()
            || plan
                .uniforms
                .iter()
                .any(|(key, bytes)| got.uniforms.get(key) != Some(bytes))
        {
            differences.push("uniforms");
        }
        // Uniform buffers the commit resolves are read in place or copied.
        let deferred_key = |key: &(usize, u32)| plan.deferred.iter().any(|item| item.key() == *key);
        let direct = got
            .guest_uniforms
            .iter()
            .filter(|(key, _)| !deferred_key(key));
        let mut deferred_uniforms: Vec<(usize, u32)> = plan
            .deferred
            .iter()
            .filter(|item| matches!(item, Deferred::Uniform { .. }))
            .map(Deferred::key)
            .collect();
        deferred_uniforms.sort_unstable();
        let mut got_deferred: Vec<(usize, u32)> = got
            .uniforms_in_place
            .keys()
            .copied()
            .chain(got.guest_uniforms.keys().copied().filter(deferred_key))
            .collect();
        got_deferred.sort_unstable();
        if direct.clone().count() != plan.guest_uniforms.len()
            || direct
                .into_iter()
                .any(|(key, value)| plan.guest_uniforms.get(key) != Some(value))
            || deferred_uniforms != got_deferred
        {
            differences.push("guest uniforms");
        }
        if plan.textures.len() != got.textures.len()
            || plan.textures.iter().any(|(key, (source, sampler))| {
                got.textures
                    .get(key)
                    .is_none_or(|(got_source, got_sampler)| {
                        !source.same_as(got_source) || sampler != got_sampler
                    })
            })
        {
            differences.push("textures");
        }
        if plan.sampled_textures != *got.sampled_textures {
            differences.push("sampled texture types");
        }
        if plan.fragment_constants != got.fragment_constants {
            differences.push("fragment constants");
        }
        let mut storage_keys: Vec<(usize, u32)> = got
            .storage
            .keys()
            .chain(got.storage_in_place.keys())
            .copied()
            .collect();
        storage_keys.sort_unstable();
        let mut planned_keys: Vec<(usize, u32)> = plan
            .deferred
            .iter()
            .filter(|item| matches!(item, Deferred::Storage { .. }))
            .map(Deferred::key)
            .collect();
        planned_keys.sort_unstable();
        if storage_keys != planned_keys {
            differences.push("storage buffers");
        }
        if plan.storage_written.len() != got.storage_written.len()
            || plan
                .storage_written
                .iter()
                .any(|(key, value)| got.storage_written.get(key) != Some(value))
        {
            differences.push("written storage buffers");
        }
        if plan.storage_entries != *got.storage_entries {
            differences.push("driver storage entries");
        }
        let deferred_bytes: u64 = plan
            .deferred
            .iter()
            .filter_map(|item| match item {
                Deferred::Uniform { key, size, .. } if got.guest_uniforms.contains_key(key) => {
                    Some(*size)
                }
                _ => None,
            })
            .sum();
        if plan.uniform_bytes + deferred_bytes != got.uniform_bytes
            || plan.memo_hits != got.memo_hits
        {
            differences.push("counts");
        }
        let backend = self.gpu.as_mut().unwrap();
        backend.stat("stat: parallel draws verified", 1);
        if !differences.is_empty() {
            backend.stat("stat: parallel draws that differ", 1);
            static REPORTED: AtomicUsize = AtomicUsize::new(0);
            if REPORTED.fetch_add(1, Ordering::Relaxed) < 20 {
                eprintln!(
                    "parallel draw plan differs at present {} program {:016x}/{:016x}: {}",
                    self.presented_frames,
                    plan.program.keys.0,
                    plan.program.keys.1,
                    differences.join(", ")
                );
            }
        }
    }
}

/// The binding loop's outputs a plan is installed into.
pub(super) struct InstallTo<'a> {
    pub uniforms: &'a mut echora_gpu::FastMap<(usize, u32), Vec<u8>>,
    pub guest_uniforms: &'a mut echora_gpu::FastMap<(usize, u32), (u64, usize)>,
    pub textures: &'a mut echora_gpu::FastMap<(usize, u32), (TextureSource, Sampler)>,
    pub sampled_textures: &'a mut [Vec<shader::SampledTexture>; 3],
    pub fragment_constants: &'a mut Option<(usize, u32, usize)>,
    pub storage: &'a mut echora_gpu::FastMap<(usize, u32), Vec<u8>>,
    pub uniforms_in_place:
        &'a mut echora_gpu::FastMap<(usize, u32), (echora_gpu::ash::vk::Buffer, u64, u64)>,
    pub storage_in_place:
        &'a mut echora_gpu::FastMap<(usize, u32), (echora_gpu::ash::vk::Buffer, u64, u64)>,
    pub storage_written: &'a mut super::super::FastMap<(usize, u32), (u64, u64)>,
    pub storage_entries: &'a mut Vec<(usize, u32, u64, u64)>,
    pub uniform_bytes: &'a mut u64,
    pub memo_hits: &'a mut u64,
}

/// What the binding loop resolved, for `verify_planned_bindings`.
pub(super) struct Resolved<'a> {
    pub uniforms: &'a echora_gpu::FastMap<(usize, u32), Vec<u8>>,
    pub guest_uniforms: &'a echora_gpu::FastMap<(usize, u32), (u64, usize)>,
    pub textures: &'a echora_gpu::FastMap<(usize, u32), (TextureSource, Sampler)>,
    pub sampled_textures: &'a [Vec<shader::SampledTexture>; 3],
    pub fragment_constants: Option<(usize, u32, usize)>,
    pub storage: &'a echora_gpu::FastMap<(usize, u32), Vec<u8>>,
    pub uniforms_in_place:
        &'a echora_gpu::FastMap<(usize, u32), (echora_gpu::ash::vk::Buffer, u64, u64)>,
    pub storage_in_place:
        &'a echora_gpu::FastMap<(usize, u32), (echora_gpu::ash::vk::Buffer, u64, u64)>,
    pub storage_written: &'a super::super::FastMap<(usize, u32), (u64, u64)>,
    pub storage_entries: &'a Vec<(usize, u32, u64, u64)>,
    pub uniform_bytes: u64,
    pub memo_hits: u64,
}

/// The statistic a fallback's reason is counted under.
fn fallback_stat(reason: &'static str) -> &'static str {
    match reason {
        "program from another present" | "program memo from another present" => {
            "stat: parallel draw fallbacks: program memo from another present"
        }
        "no program bound" => "stat: parallel draw fallbacks: no program bound",
        "program memoized for several target formats" => {
            "stat: parallel draw fallbacks: program memoized for several target formats"
        }
        "program not memoized" => "stat: parallel draw fallbacks: program not memoized",
        "uniform buffer not bound" => "stat: parallel draw fallbacks: uniform buffer not bound",
        "uniform buffer in no pool" => "stat: parallel draw fallbacks: uniform buffer in no pool",
        "uniform buffer overflows" => "stat: parallel draw fallbacks: uniform buffer overflows",
        "storage buffer not bound" => "stat: parallel draw fallbacks: storage buffer not bound",
        "storage buffer in no pool" => "stat: parallel draw fallbacks: storage buffer in no pool",
        "texture not bound" => "stat: parallel draw fallbacks: texture not bound",
        "texture binding not memoized" => {
            "stat: parallel draw fallbacks: texture binding not memoized"
        }
        "texture binding to resolve again" => {
            "stat: parallel draw fallbacks: texture binding to resolve again"
        }
        _ => "stat: parallel draw fallbacks: other bindings",
    }
}

/// A job for every thread of a `Pool`: called with each chunk index once.
type Job<'a> = dyn Fn(usize) + Sync + 'a;

struct PoolShared {
    slot: Mutex<Slot>,
    wake: Condvar,
    done: Condvar,
    /// The next chunk to take, of the current job.
    next: AtomicUsize,
    panicked: AtomicBool,
    /// `Slot::generation`, read without the lock by waiting threads.
    generation: std::sync::atomic::AtomicU64,
}

#[derive(Default)]
struct Slot {
    generation: u64,
    /// The current job with its lifetime erased: `Pool::run` returns only
    /// once no thread holds it.
    job: Option<(usize, *const Job<'static>)>,
    /// Threads still inside the current job.
    busy: usize,
    stop: bool,
}

// SAFETY: the job pointer is shared only while `Pool::run` keeps the job
// alive and waits for every thread to leave it; the job is `Sync`.
#[allow(unsafe_code)]
unsafe impl Send for Slot {}

/// How long a planning thread watches for the next job before sleeping.
const SPIN: std::time::Duration = std::time::Duration::from_micros(50);

/// Threads that plan draws with the replay thread.
struct Pool {
    shared: Arc<PoolShared>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

impl Pool {
    fn new((threads, cpus): (usize, Option<Vec<usize>>)) -> Self {
        let shared = Arc::new(PoolShared {
            slot: Mutex::new(Slot::default()),
            wake: Condvar::new(),
            done: Condvar::new(),
            next: AtomicUsize::new(0),
            panicked: AtomicBool::new(false),
            generation: std::sync::atomic::AtomicU64::new(0),
        });
        let threads = (0..threads)
            .map(|index| {
                let shared = shared.clone();
                let cpus = cpus.clone();
                std::thread::Builder::new()
                    .name(format!("nvn-plan-{index}"))
                    .spawn(move || {
                        // Not the replay's placement (inherited): the CPUs
                        // the guest threads leave idle.
                        if let Some(cpus) = cpus {
                            crate::host_cores::run_on(&cpus);
                        }
                        Self::work(&shared)
                    })
                    .expect("draw planning thread")
            })
            .collect();
        Self { shared, threads }
    }

    #[allow(unsafe_code)]
    fn work(shared: &PoolShared) {
        let mut seen = 0;
        loop {
            // Windows come close together: watch for the next a while
            // before sleeping (a wake-up costs tens of microseconds).
            let until = std::time::Instant::now() + SPIN;
            while shared.generation.load(Ordering::Acquire) == seen
                && std::time::Instant::now() < until
            {
                std::hint::spin_loop();
            }
            let (chunks, job) = {
                let mut slot = shared
                    .slot
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                loop {
                    if slot.stop {
                        return;
                    }
                    if slot.generation != seen {
                        seen = slot.generation;
                        // A job withdrawn already (its chunks all taken).
                        if let Some(job) = slot.job {
                            slot.busy += 1;
                            break job;
                        }
                    }
                    slot = shared
                        .wake
                        .wait(slot)
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                }
            };
            // SAFETY: `run` keeps the job alive until `busy` is back to 0.
            let job = unsafe { &*job };
            Self::take_chunks(shared, chunks, job);
            let mut slot = shared
                .slot
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            slot.busy -= 1;
            if slot.busy == 0 {
                shared.done.notify_all();
            }
        }
    }

    fn take_chunks(shared: &PoolShared, chunks: usize, job: &Job<'_>) {
        loop {
            let chunk = shared.next.fetch_add(1, Ordering::AcqRel);
            if chunk >= chunks {
                return;
            }
            if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job(chunk))).is_err() {
                shared.panicked.store(true, Ordering::Release);
            }
        }
    }

    /// Run `job` for chunks 0..chunks on the pool's threads and this one,
    /// returning when all are done.
    #[allow(unsafe_code)]
    fn run(&self, chunks: usize, job: &Job<'_>) -> Result<(), &'static str> {
        let shared = &*self.shared;
        shared.next.store(0, Ordering::Release);
        shared.panicked.store(false, Ordering::Release);
        // SAFETY: the lifetime is erased only for the threads, which leave
        // the job before this function returns (it waits for `busy`).
        let erased: *const Job<'static> =
            unsafe { std::mem::transmute::<*const Job<'_>, *const Job<'static>>(job) };
        {
            let mut slot = shared
                .slot
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            slot.generation += 1;
            slot.job = Some((chunks, erased));
            shared.generation.store(slot.generation, Ordering::Release);
        }
        shared.wake.notify_all();
        Self::take_chunks(shared, chunks, job);
        let mut slot = shared
            .slot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // No thread takes the job once it is withdrawn; those inside it
        // finish first. (They hold the job's borrows: this cannot return
        // before they do, so a long wait is reported, not abandoned.)
        slot.job = None;
        let started = std::time::Instant::now();
        let mut reported = 0;
        while slot.busy > 0 {
            let (next, timeout) = shared
                .done
                .wait_timeout(slot, std::time::Duration::from_secs(1))
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            slot = next;
            if timeout.timed_out() && slot.busy > 0 {
                reported += 1;
                if reported <= 10 || reported % 60 == 0 {
                    eprintln!(
                        "parallel draws: {} planning threads still busy after {:?} ({chunks} chunks)",
                        slot.busy,
                        started.elapsed()
                    );
                }
            }
        }
        drop(slot);
        if shared.panicked.load(Ordering::Acquire) {
            return Err("a draw planning thread panicked");
        }
        Ok(())
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.slot.lock().unwrap().stop = true;
        self.shared.wake.notify_all();
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}

/// Test switches: this thread's planner settings, and a record of each
/// draw's resolved bindings (the draw then stops there).
#[cfg(test)]
pub(super) mod testing {
    use std::cell::{Cell, RefCell};
    thread_local! {
        pub static THREADS: Cell<Option<usize>> = const { Cell::new(None) };
        pub static VERIFY: Cell<bool> = const { Cell::new(false) };
        pub static BACKLOG: Cell<Option<usize>> = const { Cell::new(None) };
        pub static RECORD: RefCell<Option<Vec<String>>> = const { RefCell::new(None) };
    }

    /// Record a draw's bindings (`describe`) when recording; whether the
    /// draw stops after them.
    pub fn record_bindings(describe: impl FnOnce() -> String) -> bool {
        RECORD.with(|record| {
            let mut record = record.borrow_mut();
            let Some(record) = record.as_mut() else {
                return false;
            };
            record.push(describe());
            true
        })
    }
}

/// A texture source without its texels (for comparing resolved bindings).
#[cfg(test)]
pub(super) fn source_summary(source: &TextureSource) -> String {
    match source {
        TextureSource::Upload {
            key,
            generation,
            width,
            height,
            format,
            ..
        } => format!("upload {key:#x}/{generation} {width}x{height} {format:?}"),
        other => format!("{other:?}"),
    }
}

/// Sorted debug lines of a map (for comparing resolved bindings).
#[cfg(test)]
pub(super) fn sorted_entries<K: std::fmt::Debug, V: std::fmt::Debug>(
    entries: impl IntoIterator<Item = (K, V)>,
) -> Vec<String> {
    let mut lines: Vec<String> = entries
        .into_iter()
        .map(|(key, value)| format!("{key:?}={value:?}"))
        .collect();
    lines.sort();
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_runs_every_chunk_once_and_can_be_reused() {
        let pool = Pool::new((3, None));
        for round in 0..50 {
            let chunks = 1 + round % 7;
            let counts: Vec<AtomicUsize> = (0..chunks).map(|_| AtomicUsize::new(0)).collect();
            pool.run(chunks, &|chunk| {
                counts[chunk].fetch_add(1, Ordering::Relaxed);
            })
            .unwrap();
            assert!(
                counts
                    .iter()
                    .all(|count| count.load(Ordering::Relaxed) == 1)
            );
        }
    }

    #[test]
    fn look_ahead_ends_at_work_and_passes_over_render_enables() {
        let command = |name: &'static str, x3: u64| Command {
            name,
            x: [0, 0, 0, x3, 0, 0, 0, 0],
            v: [0; 8],
            captured: Vec::new(),
            states: Default::default(),
        };
        assert_eq!(step(&command("nvnCommandBufferDrawArrays", 3)), Step::Draw);
        assert_eq!(step(&command("nvnCommandBufferDrawArrays", 0)), Step::Pass);
        assert_eq!(
            step(&command("nvnCommandBufferSetRenderEnableConditional", 0)),
            Step::Pass
        );
        assert_eq!(
            step(&command("nvnCommandBufferBindTexture", 0)),
            Step::State
        );
        assert_eq!(
            step(&command("nvnCommandBufferBindTextures", 0)),
            Step::Plural("BindTexture", 1)
        );
        assert_eq!(
            step(&command("nvnCommandBufferBindTexturesDeferred", 0)),
            Step::Deferred("BindTexture", 1)
        );
        assert_eq!(
            step(&command("nvnCommandBufferSetTexturePool", 0)),
            Step::Pool
        );
        assert_eq!(step(&command("nvnCommandBufferClearColor", 0)), Step::Pass);
        assert_eq!(
            step(&command("nvnCommandBufferDispatchCompute", 0)),
            Step::Pass
        );
        assert_eq!(
            step(&command("nvnCommandBufferCallCommands", 0)),
            Step::Call
        );
        assert_eq!(step(&command("nvnCommandBufferWaitEvent", 0)), Step::End);
        assert_eq!(step(&command("nvnCommandBufferWaitSync", 0)), Step::End);
        assert_eq!(
            step(&command("nvnCommandBufferCopyCommands", 0)),
            Step::Call
        );
        assert_eq!(step(&command("nvnCommandBufferFooBar", 0)), Step::End);
    }
}

/// A recorded batch replayed with its bindings resolved by the replay
/// alone and by the planner, on lavapipe or whatever device is present.
#[cfg(test)]
mod replay_tests {
    use super::super::super::{Args, QueueItem};
    use super::super::{Backend, GraphicsStageLayout, TextureBacking};
    use super::*;
    use echora_gpu::shader::{Descriptor, Stage};
    use echora_guest::{cpu::Cpu, memory::Memory};
    use echora_shader_abi::ABI_VERSION;
    use std::collections::BTreeMap;

    const GUEST: u64 = 0x10_0000;
    const GPU: u64 = 0x20_0000;
    const QUEUE: u64 = 5;
    const CONTROL: u64 = GUEST + 0xf000;

    /// Guest memory with the pool's bytes and the stages' control blocks.
    fn memory() -> Memory {
        let mut memory = Memory::new(1 << 39);
        use echora_guest::memory::{READ, STATE_NORMAL, WRITE};
        memory
            .map(GUEST, 0x10000, READ | WRITE, STATE_NORMAL, "pool")
            .unwrap();
        let bytes: Vec<u8> = (0..0x10000u32)
            .map(|at| (at * 7 + at / 251) as u8)
            .collect();
        memory.write(GUEST, &bytes).unwrap();
        for (control, size, offset) in
            [(CONTROL, 0x24u32, 0x400u32), (CONTROL + 0x100, 0x10, 0x480)]
        {
            memory.write(control + 0x3c, &size.to_le_bytes()).unwrap();
            memory.write(control + 0x40, &offset.to_le_bytes()).unwrap();
        }
        memory
    }

    fn command(name: &'static str, x: [u64; 8], captured: Vec<u64>) -> Command {
        Command {
            name,
            x,
            v: [0; 8],
            captured,
            states: Default::default(),
        }
    }

    fn descriptor(kind: DescriptorKind, binding: u32, index: u32) -> Descriptor {
        Descriptor {
            kind: kind as u32,
            binding,
            index,
            ..Default::default()
        }
    }

    fn stage(stage: Stage, descriptors: Vec<Descriptor>) -> Arc<Translated> {
        let mut cbuf_used_sizes = [0; 18];
        cbuf_used_sizes[0] = 0x700;
        cbuf_used_sizes[3] = 0x100;
        cbuf_used_sizes[4] = 0x40;
        cbuf_used_sizes[6] = 0x20;
        Arc::new(Translated {
            abi_version: ABI_VERSION,
            stage,
            spirv: Vec::new(),
            descriptors,
            loaded_generics: 0,
            stored_generics: 0,
            stores: [0; 8],
            frag_colors: 0,
            cbuf_used_sizes,
            spec_constants: Vec::new(),
            push_fields: 0,
            texture_pool_bindings: 0,
        })
    }

    use echora_gpu::shader::Translated;

    /// An NVN state with a GPU, one memory pool, a bound program pair
    /// whose stages read every kind of binding the planner handles, and
    /// memoized texture bindings.
    fn fixture() -> Option<Nvn> {
        let backend = Backend::create(None).ok()??;
        let mut nvn = Nvn {
            gpu: Some(backend),
            gpu_checked: true,
            ..Nvn::default()
        };
        let args = |x| Args {
            x,
            ..Default::default()
        };
        nvn.initialize(
            1,
            "MemoryPool",
            BTreeMap::from([("Storage".into(), args([GUEST, 0x10000, 0, 0]))]),
        )
        .unwrap();
        nvn.objects
            .get_mut(&1)
            .unwrap()
            .state
            .insert("GpuAddress".into(), args([GPU, 0, 0, 0]));
        let mut storage = descriptor(DescriptorKind::StorageBuffer, 3, 0);
        storage.cbuf_offset = c0_storage_table(0);
        let mut written = descriptor(DescriptorKind::StorageBuffer, 5, 0);
        written.cbuf_offset = c0_storage_table(0) + 16;
        written.flags = 1;
        let vertex = stage(
            Stage::Vertex,
            vec![
                descriptor(DescriptorKind::UniformBuffer, 0, 24),
                descriptor(DescriptorKind::UniformBuffer, 1, 3),
                descriptor(DescriptorKind::UniformBuffer, 2, 0),
                storage,
                descriptor(DescriptorKind::UniformBuffer, 4, 1),
                written,
            ],
        );
        let texture = |binding, slot: u32| {
            let mut texture = descriptor(DescriptorKind::Texture, binding, 0);
            texture.cbuf_index = 2;
            texture.cbuf_offset = 0x20 + 8 * slot;
            texture
        };
        let fragment = stage(
            Stage::Fragment,
            vec![
                texture(0, 0),
                texture(1, 1),
                descriptor(DescriptorKind::UniformBuffer, 2, 4),
                descriptor(DescriptorKind::UniformBuffer, 3, 6),
            ],
        );
        let key = (2, 3, nvn.fragment_output_types());
        // Each stage's code at the pool's start, its control block (the
        // constants' size and offset at 0x3c and 0x40) near its end.
        let entries = [CONTROL, CONTROL + 0x100]
            .map(|control| (args([GPU, control, 0, 0]), GraphicsStageLayout::default()));
        nvn.program_memo.1.insert(
            key,
            (
                Default::default(),
                Arc::new(BoundProgram::new(
                    (2, 3),
                    [vertex, fragment].into(),
                    entries.into(),
                )),
            ),
        );
        // Handle 22 comes from an earlier present: a render target's
        // binding that still holds (checked again, then used).
        nvn.presented_frames = 1;
        nvn.program_memo.0 = (1, 0);
        for handle in 19..23 {
            nvn.binding_memo.1.insert(
                handle,
                ResolvedBinding {
                    source: TextureSource::Target(handle + 100),
                    sampler: Sampler::default(),
                    sample_type: shader::SampleType::Float,
                    backing: Some(TextureBacking::linear(GUEST + 0x8000, 0x1000)),
                    gpu_writes: 0,
                    present: u64::from(handle != 22),
                    texture: handle + 100,
                },
            );
        }
        nvn.render_queue = Some(QUEUE);
        Some(nvn)
    }

    /// Binds, constants and draws the planner looks ahead through,
    /// changing every kind of binding between draws.
    fn batch() -> Vec<Command> {
        let draw = command(
            "nvnCommandBufferDrawArrays",
            [0, 4, 0, 3, 0, 0, 0, 0],
            vec![],
        );
        let uniform = |stage, slot, offset, size| {
            command(
                "nvnCommandBufferBindUniformBuffer",
                [0, stage, slot, GPU + offset, size, 0, 0, 0],
                vec![],
            )
        };
        let texture = |slot, handle| {
            command(
                "nvnCommandBufferBindTexture",
                [0, 1, slot, handle, 0, 0, 0, 0],
                vec![],
            )
        };
        let storage = |slot, offset, size| {
            command(
                "nvnCommandBufferBindStorageBuffer",
                [0, 0, slot, GPU + offset, size, 0, 0, 0],
                vec![],
            )
        };
        let push = |offset, words: Vec<u64>| {
            command(
                "nvnCommandBufferPushGraphicsConstants",
                [0, 0, offset, 0, 0, 0, 0, 0],
                words,
            )
        };
        vec![
            command(
                "nvnCommandBufferBindProgram",
                [0, 2, 1, 0, 0, 0, 0, 0],
                vec![],
            ),
            command(
                "nvnCommandBufferBindProgram",
                [0, 3, 2, 0, 0, 0, 0, 0],
                vec![],
            ),
            uniform(0, 0, 0x100, 0x100),
            uniform(1, 1, 0x300, 0x40),
            uniform(1, 3, 0x340, 0x10),
            storage(0, 0x803, 0x40),
            storage(1, 0x900, 0x20),
            texture(0, 19),
            texture(1, 20),
            push(0, vec![1, 2, 3]),
            draw.clone(),
            texture(1, 21),
            command("nvnCommandBufferBarrier", [0, 0, 0, 0, 0, 0, 0, 0], vec![]),
            push(16, vec![9]),
            draw.clone(),
            command(
                "nvnCommandBufferBindTextures",
                [0, 1, 0, 2, 0, 0, 0, 0],
                vec![22, 19],
            ),
            command(
                "nvnCommandBufferBindUniformBuffers",
                [0, 0, 0, 1, 0, 0, 0, 0],
                vec![GPU + 0x200, 0x80],
            ),
            command(
                "nvnCommandBufferDrawArraysInstanced",
                [0, 4, 0, 3, 0, 2, 0, 0],
                vec![],
            ),
            command(
                "nvnCommandBufferDrawArrays",
                [0, 4, 0, 0, 0, 0, 0, 0],
                vec![],
            ),
            storage(0, 0xa00, 0x10),
            command(
                "nvnCommandBufferSetScissor",
                [0, 0, 0, 16, 16, 0, 0, 0],
                vec![],
            ),
            draw.clone(),
            uniform(0, 0, 0x180, 0x20),
            draw.clone(),
            push(0x40, vec![7, 7]),
            texture(0, 20),
            draw,
        ]
    }

    /// Each draw's resolved bindings after replaying `batch` with `threads`
    /// planner threads (0: the replay alone), and the NVN state's stats.
    fn replay(threads: usize, verify: bool) -> Option<(Vec<String>, Nvn)> {
        replay_with(threads, verify, &[batch()], None, |_| {}, |_| {})
    }

    /// `replay` of `lists`, queued in order, in pieces of `budget`
    /// commands, after `setup`.
    fn replay_with(
        threads: usize,
        verify: bool,
        lists: &[Vec<Command>],
        budget: Option<u32>,
        setup: impl Fn(&mut Nvn),
        fill: fn(&mut Memory),
    ) -> Option<(Vec<String>, Nvn)> {
        let mut nvn = fixture()?;
        setup(&mut nvn);
        let mut memory = memory();
        fill(&mut memory);
        let mut view = memory.view();
        let mut cpu = Cpu::default();
        let mut core = Guest {
            cpu: &mut cpu,
            memory: GuestMemory::View(&mut view),
            counter: Some(0),
        };
        // The queue's own state already (no switch, which would swap the
        // memo out), seeing the list's descriptors.
        nvn.render_queue = Some(QUEUE);
        nvn.render.descriptor_epoch = Some(0);
        for list in lists {
            nvn.pending
                .entry(QUEUE)
                .or_default()
                .push_back(QueueItem::List {
                    commands: Arc::new(list.clone()),
                    next: 0,
                    descriptor_epoch: 0,
                });
        }
        testing::THREADS.with(|cell| cell.set(Some(threads)));
        testing::VERIFY.with(|cell| cell.set(verify));
        testing::RECORD.with(|record| *record.borrow_mut() = Some(Vec::new()));
        let mut result = Ok(true);
        for _ in 0..1000 {
            nvn.replay_budget = budget;
            nvn.replay_paused = false;
            result = nvn.run_queue(QUEUE, &mut core, true);
            if result.is_err() || !nvn.replay_paused {
                break;
            }
        }
        nvn.replay_budget = None;
        let record = testing::RECORD.with(|record| record.borrow_mut().take().unwrap());
        testing::THREADS.with(|cell| cell.set(None));
        testing::VERIFY.with(|cell| cell.set(false));
        result.unwrap();
        Some((record, nvn))
    }

    fn stat(nvn: &Nvn, name: &str) -> u64 {
        nvn.gpu.as_ref().unwrap().stat_value(name)
    }

    #[test]
    fn planned_bindings_match_the_replay_s_own() {
        let Some((serial, serial_nvn)) = replay(0, false) else {
            return;
        };
        assert_eq!(serial.len(), 6, "{serial:#?}");
        assert_eq!(stat(&serial_nvn, "stat: parallel draws used"), 0);
        for threads in [1, 3] {
            let (parallel, nvn) = replay(threads, false).unwrap();
            assert_eq!(stat(&nvn, "stat: parallel draws used"), 6);
            assert_eq!(parallel, serial, "{threads} planner threads");
            for name in [
                "stat: draw storage shared",
                "stat: draw storage bytes",
                "stat: texture binding memo hits",
                "stat: draw uniform bytes",
            ] {
                assert_eq!(stat(&nvn, name), stat(&serial_nvn, name), "{name}");
            }
        }
    }

    #[test]
    fn verification_finds_no_difference() {
        let Some((record, nvn)) = replay(2, true) else {
            return;
        };
        assert_eq!(record.len(), 6);
        assert_eq!(stat(&nvn, "stat: parallel draws verified"), 6);
        assert_eq!(stat(&nvn, "stat: parallel draws that differ"), 0);
    }

    #[test]
    fn a_state_change_drops_the_remaining_plans() {
        let Some(mut nvn) = fixture() else {
            return;
        };
        let memory = memory();
        let mut view = memory.view();
        let mut cpu = Cpu::default();
        let mut core = Guest {
            cpu: &mut cpu,
            memory: GuestMemory::View(&mut view),
            counter: Some(0),
        };
        let list = Arc::new(batch());
        testing::THREADS.with(|cell| cell.set(Some(2)));
        // The render state at the first draw, as the replay would have it.
        let first = list
            .iter()
            .position(|command| step(command) == Step::Draw)
            .unwrap();
        for command in &list[..first] {
            apply(&mut nvn.render, command, step(command), None);
        }
        nvn.prepare_parallel_draws(QUEUE, &list, first, &mut core);
        assert_eq!(stat(&nvn, "stat: parallel draws planned"), 6);
        let program = nvn.program_memo.1.values().next().unwrap().1.clone();
        assert!(nvn.take_planned_bindings(&program).is_some());
        // Memory pools changed since: guest addresses may have moved.
        nvn.pool_epoch += 1;
        let second = first
            + 1
            + list[first + 1..]
                .iter()
                .position(|command| step(command) == Step::Draw)
                .unwrap();
        // The replay follows the window to the next draw.
        for at in first + 1..=second {
            nvn.prepare_parallel_draws(QUEUE, &list, at, &mut core);
        }
        assert!(nvn.take_planned_bindings(&program).is_none());
        assert_eq!(
            stat(&nvn, "stat: parallel draw windows dropped by a pool change"),
            1
        );
        assert_eq!(
            stat(
                &nvn,
                "stat: parallel draw windows dropped by a state change"
            ),
            1
        );
        // That draw's plan and the four after it.
        assert_eq!(
            stat(&nvn, "stat: parallel draw plans dropped by a state change"),
            5
        );
        testing::THREADS.with(|cell| cell.set(None));
    }

    #[test]
    fn a_binding_the_planner_cannot_resolve_falls_back_to_the_replay() {
        let Some(mut nvn) = fixture() else {
            return;
        };
        nvn.binding_memo.1.remove(&21);
        let memory = memory();
        let mut view = memory.view();
        let mut cpu = Cpu::default();
        let mut core = Guest {
            cpu: &mut cpu,
            memory: GuestMemory::View(&mut view),
            counter: Some(0),
        };
        let list = Arc::new(batch());
        testing::THREADS.with(|cell| cell.set(Some(1)));
        let first = list
            .iter()
            .position(|command| step(command) == Step::Draw)
            .unwrap();
        for command in &list[..first] {
            apply(&mut nvn.render, command, step(command), None);
        }
        nvn.prepare_parallel_draws(QUEUE, &list, first, &mut core);
        testing::THREADS.with(|cell| cell.set(None));
        let reasons: Vec<Result<(), &str>> = nvn
            .parallel
            .plans
            .iter()
            .map(|(_, plan)| plan.as_ref().map(|_| ()).map_err(|reason| *reason))
            .collect();
        // Handle 21 is bound at the second draw and replaced by the third.
        assert_eq!(
            reasons,
            [
                Ok(()),
                Err("texture binding not memoized"),
                Ok(()),
                Ok(()),
                Ok(()),
                Ok(())
            ]
        );
    }
    #[test]
    fn a_window_runs_on_through_pieces_and_the_next_lists() {
        let mut second = batch();
        // Its programs and bindings carry over from the first list.
        second.drain(..2);
        let lists = [batch(), second];
        let Some((serial, _)) = replay_with(0, false, &lists, Some(7), |_| {}, |_| {}) else {
            return;
        };
        assert_eq!(serial.len(), 12);
        let (parallel, nvn) = replay_with(2, false, &lists, Some(7), |_| {}, |_| {}).unwrap();
        assert_eq!(parallel, serial);
        assert_eq!(stat(&nvn, "stat: parallel draw windows"), 1);
        assert_eq!(stat(&nvn, "stat: parallel draws used"), 12);
    }

    #[test]
    fn uniforms_with_gpu_results_pending_are_resolved_at_the_commit() {
        // GPU work stored into the first uniform buffer: the draw resolves
        // it (bringing the results back changes what later plans read).
        let pending = |nvn: &mut Nvn| {
            nvn.gpu
                .as_mut()
                .unwrap()
                .in_place_writes
                .push(GUEST + 0x100..GUEST + 0x104);
        };
        let Some((serial, _)) = replay_with(0, false, &[batch()], None, pending, |_| {}) else {
            return;
        };
        let (parallel, nvn) = replay_with(3, false, &[batch()], None, pending, |_| {}).unwrap();
        assert_eq!(parallel, serial);
        assert!(stat(&nvn, "stat: parallel draws used") >= 1);
        let (_, verified) = replay_with(3, true, &[batch()], None, pending, |_| {}).unwrap();
        assert_eq!(stat(&verified, "stat: parallel draws that differ"), 0);
    }

    #[test]
    fn plans_use_the_program_the_same_binds_bound_last() {
        let Some(mut nvn) = fixture() else {
            return;
        };
        let memory = memory();
        let mut view = memory.view();
        let mut cpu = Cpu::default();
        let mut core = Guest {
            cpu: &mut cpu,
            memory: GuestMemory::View(&mut view),
            counter: Some(0),
        };
        let list = Arc::new(batch());
        let first = list
            .iter()
            .position(|command| step(command) == Step::Draw)
            .unwrap();
        for command in &list[..first] {
            apply(&mut nvn.render, command, step(command), None);
        }
        // A flush since: the present's memo is for another flush.
        let program = nvn.program_memo.1.values().next().unwrap().1.clone();
        nvn.flush_epoch += 1;
        testing::THREADS.with(|cell| cell.set(Some(2)));
        nvn.prepare_parallel_draws(QUEUE, &list, first, &mut core);
        assert!(nvn.parallel.plans.iter().all(|(_, plan)| plan.is_err()));
        // A draw that bound it (the replay's) is remembered.
        nvn.parallel.forget();
        assert!(nvn.take_planned_bindings(&program).is_none());
        nvn.prepare_parallel_draws(QUEUE, &list, first, &mut core);
        assert!(!nvn.parallel.plans.is_empty());
        assert!(nvn.parallel.plans.iter().all(|(_, plan)| plan.is_ok()));
        let taken = nvn.take_planned_bindings(&program).is_some();
        testing::THREADS.with(|cell| cell.set(None));
        assert!(taken);
    }

    #[test]
    fn called_lists_and_deferred_binds_join_the_window() {
        const CALLED: u64 = 0x4000;
        const ARRAY: u64 = GUEST + 0xe000;
        let mut called = batch();
        called.drain(..2);
        // A texture array the GPU reads from guest memory.
        let deferred = command(
            "nvnCommandBufferBindTexturesDeferred",
            [0, 1, 0, 2, ARRAY, 0, 0, 0],
            vec![],
        );
        let mut list = batch();
        let at = list.len() - 3;
        list.insert(at, deferred);
        list.insert(
            3,
            command(
                "nvnCommandBufferCallCommands",
                [0, 1, 0, 0, 0, 0, 0, 0],
                vec![CALLED],
            ),
        );
        let setup = |nvn: &mut Nvn| {
            nvn.command_lists.insert(
                CALLED,
                super::super::super::CommandList::new(called.clone(), 0),
            );
        };
        let write_array: fn(&mut Memory) = |memory| {
            for (index, handle) in [21u64, 22].into_iter().enumerate() {
                memory
                    .write(ARRAY + index as u64 * 8, &handle.to_le_bytes())
                    .unwrap();
            }
        };
        let lists = [list];
        let Some((serial, _)) = replay_with(0, false, &lists, None, setup, write_array) else {
            return;
        };
        assert_eq!(serial.len(), 12);
        let (parallel, nvn) = replay_with(3, false, &lists, None, setup, write_array).unwrap();
        assert_eq!(parallel, serial);
        assert_eq!(stat(&nvn, "stat: parallel draw windows"), 1);
        assert_eq!(stat(&nvn, "stat: parallel draws used"), 12);
    }

    #[test]
    fn a_window_too_short_to_plan_is_counted_once() {
        // Two draws, then the end of the queued work.
        let list: Vec<Command> = batch().into_iter().take(15).collect();
        let Some((record, nvn)) = replay_with(2, false, &[list], None, |_| {}, |_| {}) else {
            return;
        };
        assert_eq!(record.len(), 2);
        assert_eq!(stat(&nvn, "stat: parallel draw windows too short"), 1);
        assert_eq!(stat(&nvn, "stat: parallel draws in windows too short"), 2);
        assert_eq!(
            stat(
                &nvn,
                "stat: parallel draw windows ended by the end of the queued work"
            ),
            1
        );
    }

    fn call(handles: Vec<u64>) -> Command {
        command(
            "nvnCommandBufferCallCommands",
            [0, handles.len() as u64, 0, 0, 0, 0, 0, 0],
            handles,
        )
    }

    #[test]
    fn lists_that_call_each_other_stop_at_the_nesting_limit() {
        const FIRST: u64 = 0x5000;
        const SECOND: u64 = 0x5001;
        let Some(mut nvn) = fixture() else {
            return;
        };
        let draw = command(
            "nvnCommandBufferDrawArrays",
            [0, 4, 0, 3, 0, 0, 0, 0],
            vec![],
        );
        let first = vec![draw.clone(), call(vec![SECOND]), draw.clone()];
        let second = vec![draw.clone(), call(vec![FIRST]), draw];
        nvn.command_lists.insert(
            FIRST,
            super::super::super::CommandList::new(first.clone(), 0),
        );
        nvn.command_lists
            .insert(SECOND, super::super::super::CommandList::new(second, 0));
        let memory = memory();
        let view = memory.view();
        let mut cpu = Cpu::default();
        let mut view = view;
        let core = Guest {
            cpu: &mut cpu,
            memory: GuestMemory::View(&mut view),
            counter: Some(0),
        };
        let list = nvn.command_lists[&FIRST].commands.clone();
        let mut window = Window {
            limit: window_commands(),
            ..Window::default()
        };
        assert!(!nvn.walk_window(&mut window, &list, 0, 0, &core));
        // Each run of a list is a segment of its own, up to the nesting
        // limit: the first list's start, the second's, the first's...
        assert_eq!(window.ended.as_deref(), Some("calls nested too deeply"));
        assert_eq!(window.segments.len(), 9);
        assert_eq!(
            window.draws,
            (0..9).map(|segment| (segment, 0)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_list_called_twice_replays_as_the_replay_alone_does() {
        const CALLED: u64 = 0x6000;
        let draw = command(
            "nvnCommandBufferDrawArrays",
            [0, 4, 0, 3, 0, 0, 0, 0],
            vec![],
        );
        let texture = |slot, handle| {
            command(
                "nvnCommandBufferBindTexture",
                [0, 1, slot, handle, 0, 0, 0, 0],
                vec![],
            )
        };
        // The called list rebinds a texture and draws twice.
        let called = vec![texture(1, 22), draw.clone(), texture(1, 20), draw.clone()];
        let mut list: Vec<Command> = batch().into_iter().take(11).collect();
        list.extend([
            call(vec![CALLED]),
            texture(0, 21),
            draw.clone(),
            call(vec![CALLED, CALLED]),
            draw.clone(),
            draw,
        ]);
        let setup = |nvn: &mut Nvn| {
            nvn.command_lists.insert(
                CALLED,
                super::super::super::CommandList::new(called.clone(), 0),
            );
        };
        let lists = [list];
        let Some((serial, _)) = replay_with(0, false, &lists, None, setup, |_| {}) else {
            return;
        };
        // 1 + 2 + 1 + 2 + 2 + 2.
        assert_eq!(serial.len(), 10);
        for threads in [1, 3] {
            let (parallel, nvn) = replay_with(threads, false, &lists, None, setup, |_| {}).unwrap();
            assert_eq!(parallel, serial, "{threads} planner threads");
            // One window, each run of the called list planned from the
            // state that run inherits.
            assert_eq!(stat(&nvn, "stat: parallel draw windows"), 1);
            assert_eq!(stat(&nvn, "stat: parallel draws used"), 10);
            let (_, verified) = replay_with(threads, true, &lists, None, setup, |_| {}).unwrap();
            assert_eq!(stat(&verified, "stat: parallel draws that differ"), 0);
        }
    }

    #[test]
    fn a_texture_to_resolve_is_left_to_its_draw() {
        let Some(mut nvn) = fixture() else {
            return;
        };
        // Handle 21 has no memoized binding: resolving it may upload it.
        nvn.binding_memo.1.remove(&21);
        let memory = memory();
        let mut view = memory.view();
        let mut cpu = Cpu::default();
        let mut core = Guest {
            cpu: &mut cpu,
            memory: GuestMemory::View(&mut view),
            counter: Some(0),
        };
        let list = Arc::new(batch());
        let first = list
            .iter()
            .position(|command| step(command) == Step::Draw)
            .unwrap();
        for command in &list[..first] {
            apply(&mut nvn.render, command, step(command), None);
        }
        let made = nvn.renderer_made();
        testing::THREADS.with(|cell| cell.set(Some(2)));
        nvn.prepare_parallel_draws(QUEUE, &list, first, &mut core);
        testing::THREADS.with(|cell| cell.set(None));
        assert_eq!(nvn.renderer_made(), made);
        assert_eq!(stat(&nvn, "stat: parallel draws resolved ahead"), 0);
        assert_eq!(
            stat(&nvn, "stat: parallel texture uploads while planning"),
            0
        );
        assert_eq!(
            stat(&nvn, "stat: parallel images created while planning"),
            0
        );
        // Its draw resolves it itself, in order.
        assert!(
            nvn.parallel
                .plans
                .iter()
                .any(|(_, plan)| { matches!(plan, Err("texture binding not memoized")) })
        );
        assert!(!nvn.binding_memo.1.contains_key(&21));
    }

    #[test]
    fn a_texture_binding_changed_since_the_plan_is_planned_by_the_replay() {
        let Some(mut nvn) = fixture() else {
            return;
        };
        let memory = memory();
        let mut view = memory.view();
        let mut cpu = Cpu::default();
        let mut core = Guest {
            cpu: &mut cpu,
            memory: GuestMemory::View(&mut view),
            counter: Some(0),
        };
        let list = Arc::new(batch());
        let first = list
            .iter()
            .position(|command| step(command) == Step::Draw)
            .unwrap();
        for command in &list[..first] {
            apply(&mut nvn.render, command, step(command), None);
        }
        testing::THREADS.with(|cell| cell.set(Some(2)));
        nvn.prepare_parallel_draws(QUEUE, &list, first, &mut core);
        // The memo resolves texture 19 to another source meanwhile.
        nvn.binding_memo.1.get_mut(&19).unwrap().source = TextureSource::Target(999);
        let program = nvn.program_memo.1.values().next().unwrap().1.clone();
        let taken = nvn.take_planned_bindings(&program);
        testing::THREADS.with(|cell| cell.set(None));
        assert!(taken.is_none());
        assert_eq!(
            stat(
                &nvn,
                "stat: parallel draws replanned: a texture binding changed"
            ),
            1
        );
    }

    #[test]
    fn a_commit_wait_on_unsubmitted_work_submits_it() {
        let Some(mut nvn) = fixture() else {
            return;
        };
        // GPU work recorded and not yet submitted (an upload into a
        // resident pool), as a draw's commit may wait for.
        let renderer = &mut nvn.gpu.as_mut().unwrap().renderer;
        renderer.resident_buffer(GUEST, 4096).unwrap();
        renderer.upload_resident(GUEST, 0, &[7; 64]).unwrap();
        let serial = renderer.recorded_serial();
        assert!(serial > renderer.completed_serial());
        let (done, finished) = std::sync::mpsc::channel();
        std::thread::scope(|scope| {
            scope.spawn(|| {
                let _phase = InPhase::enter(Phase::ResolvingAtCommit);
                renderer.wait_for(serial).unwrap();
                done.send(()).unwrap();
            });
            assert!(
                finished
                    .recv_timeout(std::time::Duration::from_secs(10))
                    .is_ok(),
                "waiting for recorded work blocked"
            );
        });
        assert!(nvn.gpu.as_ref().unwrap().renderer.completed_serial() >= serial);
    }

    #[test]
    fn verifying_a_draw_does_not_format_an_upload_s_texels() {
        // A memoized upload as large as a forest texture array's share.
        let big = |nvn: &mut Nvn| {
            nvn.binding_memo.1.get_mut(&19).unwrap().source = TextureSource::Upload {
                key: 0x19,
                generation: 1,
                width: 1365,
                height: 1365,
                shape: echora_gpu::draw::ImageShape::D2 { layers: 9 },
                format: echora_gpu::ash::vk::Format::R8G8B8A8_UNORM,
                texels: Arc::new(vec![0; 64 << 20]),
                levels: Arc::new([]),
            };
        };
        let started = std::time::Instant::now();
        let Some((_, nvn)) = replay_with(2, true, &[batch()], None, big, |_| {}) else {
            return;
        };
        assert_eq!(stat(&nvn, "stat: parallel draws verified"), 6);
        assert_eq!(stat(&nvn, "stat: parallel draws that differ"), 0);
        // Formatting the texels took seconds a draw.
        assert!(
            started.elapsed() < std::time::Duration::from_secs(5),
            "verifying took {:?}",
            started.elapsed()
        );
    }

    #[test]
    fn a_gpu_write_recorded_after_planning_resolves_the_uniforms_at_the_commit() {
        let Some(mut nvn) = fixture() else {
            return;
        };
        let memory = memory();
        let mut view = memory.view();
        let mut cpu = Cpu::default();
        let mut core = Guest {
            cpu: &mut cpu,
            memory: GuestMemory::View(&mut view),
            counter: Some(0),
        };
        let list = Arc::new(batch());
        let first = list
            .iter()
            .position(|command| step(command) == Step::Draw)
            .unwrap();
        for command in &list[..first] {
            apply(&mut nvn.render, command, step(command), None);
        }
        testing::THREADS.with(|cell| cell.set(Some(2)));
        nvn.prepare_parallel_draws(QUEUE, &list, first, &mut core);
        let program = nvn.program_memo.1.values().next().unwrap().1.clone();
        let plan = nvn.take_planned_bindings(&program).unwrap();
        testing::THREADS.with(|cell| cell.set(None));
        assert!(
            plan.deferred
                .iter()
                .any(|item| matches!(item, Deferred::Uniform { direct: true, .. }))
        );
        // GPU work recorded since stores into the first uniform buffer.
        nvn.gpu
            .as_mut()
            .unwrap()
            .in_place_writes
            .push(GUEST + 0x100..GUEST + 0x104);
        let (mut uniforms, mut guest_uniforms, mut textures) =
            (Default::default(), Default::default(), Default::default());
        let (mut storage, mut uniforms_in_place, mut storage_in_place) =
            (Default::default(), Default::default(), Default::default());
        let (mut storage_written, mut storage_entries) = (Default::default(), Vec::new());
        let (mut sampled, mut fragment, mut bytes, mut hits) = (Default::default(), None, 0, 0);
        nvn.install_planned_bindings(
            &mut core,
            plan,
            InstallTo {
                uniforms: &mut uniforms,
                guest_uniforms: &mut guest_uniforms,
                textures: &mut textures,
                sampled_textures: &mut sampled,
                fragment_constants: &mut fragment,
                storage: &mut storage,
                uniforms_in_place: &mut uniforms_in_place,
                storage_in_place: &mut storage_in_place,
                storage_written: &mut storage_written,
                storage_entries: &mut storage_entries,
                uniform_bytes: &mut bytes,
                memo_hits: &mut hits,
            },
        )
        .unwrap();
        assert_eq!(
            stat(
                &nvn,
                "stat: parallel draws with GPU writes recorded since planned"
            ),
            1
        );
        // The replay's own resolution brought the write back first.
        assert!(nvn.gpu.as_ref().unwrap().in_place_writes.is_empty());
        assert!(guest_uniforms.contains_key(&(0, 1)));
    }

    #[test]
    fn planners_take_only_the_cpus_the_guest_threads_leave() {
        // Eight CPUs, the replay's core 6-7, five guest workers: 7 and
        // the one CPU the workers leave (5).
        let (count, cpus) = planner_placement(4, Some((&[6, 7], &[0, 1, 2, 3, 4, 5])), 5, 8);
        assert_eq!((count, cpus.as_deref()), (2, Some(&[5, 6, 7][..])));
        // Six workers on six CPUs: the replay's sibling only.
        let (count, cpus) = planner_placement(4, Some((&[6, 7], &[0, 1, 2, 3, 4, 5])), 6, 8);
        assert_eq!((count, cpus.as_deref()), (1, Some(&[6, 7][..])));
        // No sibling and nothing idle: no planning threads.
        assert_eq!(
            planner_placement(4, Some((&[7], &[0, 1, 2, 3, 4, 5, 6])), 7, 8),
            (0, None)
        );
        // Without a reserved core: what the workers and the replay leave.
        assert_eq!(planner_placement(4, None, 6, 8), (1, None));
        assert_eq!(planner_placement(2, None, 2, 16), (2, None));
    }

    #[test]
    fn windows_wait_for_a_backlog() {
        let Some(mut nvn) = fixture() else {
            return;
        };
        let memory = memory();
        let mut view = memory.view();
        let mut cpu = Cpu::default();
        let mut core = Guest {
            cpu: &mut cpu,
            memory: GuestMemory::View(&mut view),
            counter: Some(0),
        };
        let list = Arc::new(batch());
        let first = list
            .iter()
            .position(|command| step(command) == Step::Draw)
            .unwrap();
        testing::THREADS.with(|cell| cell.set(Some(2)));
        testing::BACKLOG.with(|cell| cell.set(Some(2)));
        nvn.prepare_parallel_draws(QUEUE, &list, first, &mut core);
        testing::BACKLOG.with(|cell| cell.set(None));
        testing::THREADS.with(|cell| cell.set(None));
        assert_eq!(stat(&nvn, "stat: parallel draw windows"), 0);
        assert_eq!(
            stat(
                &nvn,
                "stat: parallel draw windows skipped: the replay keeps up"
            ),
            1
        );
    }
}
