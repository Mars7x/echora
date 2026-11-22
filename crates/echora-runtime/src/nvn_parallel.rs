//! Parallel draw planning (ECHORA_PARALLEL_DRAWS=N, off by default).
//!
//! The replay records draws one at a time on its own thread, and most of a
//! draw's host time is planning: finding its bound program, resolving each
//! binding (uniform and storage buffers, driver constants, textures) from
//! the render state. That part reads state and changes none of it in the
//! common case, so it can run ahead of the replay on several threads:
//!
//! 1. Look-ahead (`prepare_parallel_draws`): at a draw, the replay looks
//!    down its command list for a window of commands that only change the
//!    render state (binds, constants, state objects) or draw, up to the
//!    next command that does other work (copies, clears, dispatches, waits,
//!    calls) or the end of the replay's piece.
//! 2. Plan (`plan_draw`, on `N` worker threads and the replay thread): the
//!    window's draws are split into one run per thread. Each thread starts
//!    from the render state at the window's start, applies the state
//!    commands before its draws as the replay does (`RenderState::
//!    apply_state`, shared with it), and plans each of its draws from the
//!    state that draw sees. The replay waits for all of them: while they
//!    run nothing changes the NVN state they read.
//! 3. Commit (the replay, in order): each command runs as before; a draw
//!    with a plan takes its bindings from it (`take_planned_bindings`) in
//!    place of resolving them, and everything that changes shared state
//!    (imports of shared spans, programs, targets, recording) stays where it
//!    was, in order.
//!
//! A plan holds only when the state it read is unchanged at its commit:
//! each window records a stamp of that state (memo generations, GPU write
//! tracking, pools, targets), and a draw whose stamp no longer matches, or
//! that the planner could not handle (a binding that would wait for GPU
//! work, import a span, or resolve a texture anew), is planned by the
//! replay as before. ECHORA_PARALLEL_DRAWS_VERIFY=1 (debugging) plans
//! every draw both ways and reports any difference.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use echora_gpu::draw::{Sampler, TextureSource};
use echora_gpu::shader::{self, DescriptorKind};

use super::super::{
    Command, Guest, GuestMemory, Nvn, PoolRange, RenderState, ResolvedBinding, StateSnapshots,
    is_state_command, plural_bind, pool_lookup, program_key, without_prefix,
};
use super::{
    BoundProgram, GuestFileSpan, bind_key, c0_storage_table, draw_switches_set, empty_draw,
    nvn_stage, overlapping_spans, shared_write_serial_in, typed_sampled_texture,
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

/// Whether draws are planned in parallel (`threads`).
pub(in crate::nvn) fn enabled() -> bool {
    threads() > 0
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
/// (ECHORA_PARALLEL_DRAWS_WINDOW, 1024).
fn window_commands() -> usize {
    static WINDOW: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *WINDOW.get_or_init(|| {
        std::env::var("ECHORA_PARALLEL_DRAWS_WINDOW")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|&value| value >= 2)
            .unwrap_or(1024)
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
    /// Push{Graphics,Compute}Constants.
    Constants,
    /// Changes nothing planning reads (other draws, barriers, targets,
    /// viewports, render enables): passed over.
    Pass,
    /// Other work: the window ends before it.
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
        // Handled by `execute_command`'s own arms, none of whose effects
        // planning reads (targets enter only through the program, which the
        // commit checks).
        "SetRenderTargets"
        | "SetViewports"
        | "SetScissors"
        | "SetScissor"
        | "SetDepthRanges"
        | "Barrier"
        | "SetRenderEnable"
        | "SetRenderEnableConditional" => return Step::Pass,
        "PushGraphicsConstants" | "PushComputeConstants" => return Step::Constants,
        // A pool set moves the binding serial (every memoized binding).
        "SetTexturePool" | "SetSamplerPool" => return Step::End,
        _ => {}
    }
    if rest.starts_with("Draw") || rest.starts_with("MultiDraw") {
        return Step::Pass;
    }
    if let Some((single, words)) = plural_bind(rest) {
        // Deferred arrays are read from guest memory in queue order.
        return if rest.ends_with("Deferred") {
            Step::End
        } else {
            Step::Plural(single, words)
        };
    }
    if is_state_command(rest) {
        return Step::State;
    }
    Step::End
}

/// Apply a look-ahead step's change to the render state.
fn apply(render: &mut RenderState, command: &Command, step: Step) {
    match step {
        Step::State => {
            let rest = without_prefix(command.name, "nvnCommandBuffer");
            render.apply_state(rest, command);
        }
        Step::Plural(single, words) => {
            render.apply_plural_bind(single, words, command, &command.captured);
        }
        Step::Constants => {
            render.push_constants(command);
        }
        Step::Draw | Step::Pass | Step::End => {}
    }
}

/// The state a window's plans read, as it was when they were made: a plan
/// is used only while its draw meets the same.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Stamp {
    presented_frames: u64,
    flush_epoch: u64,
    program_memo: (u64, u64),
    binding_memo: (u64, u64, u64, u64),
    binding_epoch: u64,
    binding_serial: u64,
    binding_evictions: usize,
    pool_inputs: u64,
    descriptor_epoch: Option<u64>,
    pool_epoch: u64,
    gpu_writes: u64,
    target_changes: u64,
    share_generation: usize,
    in_place_writes: usize,
    pending_storage: usize,
    dirty_removals: u64,
    resident_dirty: usize,
    page_table_writes: bool,
    page_table_write_serial: u64,
    unbounded_write_serial: u64,
    shared_writes: (usize, u64),
}

/// A storage buffer binding as planned: whether it is shared in place or
/// copied is decided at its commit (`shared_binding` imports spans, which
/// must stay in order).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct PlannedStorage {
    pub key: (usize, u32),
    pub guest: u64,
    pub base: u64,
    pub length: u64,
    /// The stage only reads it (it may be shared in place).
    pub read_only: bool,
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
    pub storage: Vec<PlannedStorage>,
    pub storage_written: super::super::FastMap<(usize, u32), (u64, u64)>,
    pub storage_entries: Vec<(usize, u32, u64, u64)>,
    pub uniform_bytes: u64,
    pub memo_hits: u64,
    /// Memoized texture bindings from another present or GPU write
    /// generation that still hold: marked checked at the commit.
    pub memo_checks: Vec<u64>,
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
    pools: &'a [PoolRange],
    memo: &'a super::super::FastMap<u64, ResolvedBinding>,
    block_aliases: &'a super::super::FastMap<u64, u64>,
    presented_frames: u64,
    gpu_writes: u64,
    fragment_jump_stages: &'a super::super::FastSet<usize>,
    in_place_writes: &'a [std::ops::Range<u64>],
    pending_storage: &'a [(u64, std::ops::Range<u64>, u32, u64, u64)],
    resident_dirty: &'a [std::ops::Range<u64>],
    /// Page-table writes not yet finished (`draw_input_write_serial`).
    table_writes: Option<TableWrites<'a>>,
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
        overlapping_spans(writes.spans, range)
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
        let programs = self.programs.ok_or("program memo from another present")?;
        let bind = |bit: u32| {
            render
                .state
                .get(program_key(bit))
                .map(|command| command.x[1])
        };
        let (Some(vertex), Some(fragment)) = (bind(0), bind(1)) else {
            return Err("no program bound");
        };
        let attributes = render
            .state
            .get("BindVertexAttribState")
            .map(|command| command.states.clone())
            .unwrap_or_default();
        let mut found: Option<&Arc<BoundProgram>> = None;
        for (key, (memo_attributes, bound)) in programs {
            if key.0 != vertex || key.1 != fragment || !attributes.ptr_eq(memo_attributes) {
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
        storage: Vec::new(),
        storage_written: Default::default(),
        storage_entries: Vec::new(),
        uniform_bytes: 0,
        memo_hits: 0,
        memo_checks: Vec::new(),
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
                    if size != 0 {
                        let end = guest.checked_add(size).ok_or("uniform buffer overflows")?;
                        // Read in place or after a wait by the replay.
                        if view.table_write_pending(guest..end) {
                            return Err("uniform buffer written by unfinished GPU work");
                        }
                        if view.gpu_writes_pending(&(guest..end)) {
                            return Err("uniform buffer holds GPU results to store");
                        }
                    }
                    plan.uniform_bytes += size;
                    plan.guest_uniforms.insert(key, (guest, size as usize));
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
                    if view.gpu_writes_pending(&(guest..guest + length)) {
                        return Err("storage buffer holds GPU results to store");
                    }
                    plan.storage.push(PlannedStorage {
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
                        plan.memo_checks.push(handle);
                    }
                    plan.memo_hits += 1;
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
    /// The list and command range the plans cover (end exclusive).
    covered: Option<(usize, usize, usize)>,
    stamp: Option<Stamp>,
    plans: VecDeque<(usize, PlanResult)>,
}

impl std::fmt::Debug for ParallelDraws {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ParallelDraws")
            .field("covered", &self.covered)
            .field("plans", &self.plans.len())
            .finish()
    }
}

impl ParallelDraws {
    /// Drop every plan (the replay gave up the NVN state, which others may
    /// now change).
    pub fn forget(&mut self) {
        self.covered = None;
        self.stamp = None;
        self.plans.clear();
        self.position = None;
    }
}

impl Nvn {
    fn parallel_stamp(&self) -> Option<Stamp> {
        let backend = self.gpu.as_ref()?;
        Some(Stamp {
            presented_frames: self.presented_frames,
            flush_epoch: self.flush_epoch,
            program_memo: self.program_memo.0,
            binding_memo: self.binding_memo.0,
            binding_epoch: self.binding_epoch,
            binding_serial: self.binding_serial,
            binding_evictions: self.binding_evictions.len(),
            pool_inputs: self.pool_inputs,
            descriptor_epoch: self.render.descriptor_epoch,
            pool_epoch: self.pool_epoch,
            gpu_writes: backend.gpu_writes,
            target_changes: backend.renderer.target_changes,
            share_generation: backend.share_generation,
            in_place_writes: backend.in_place_writes.len(),
            pending_storage: backend.pending_storage.len(),
            dirty_removals: backend.dirty_removals,
            resident_dirty: backend
                .resident
                .values()
                .map(|pool| pool.gpu_dirty.len())
                .sum(),
            page_table_writes: backend.page_table_writes,
            page_table_write_serial: backend.page_table_write_serial,
            unbounded_write_serial: backend.unbounded_write_serial,
            shared_writes: (
                backend.shared_writes.len(),
                backend
                    .shared_writes
                    .back()
                    .map_or(0, |(serial, _)| *serial),
            ),
        })
    }

    /// Before the replay executes `list[index]`: plan the draws of the
    /// look-ahead window from there on other threads, unless plans already
    /// cover it. `budget` bounds the window (the replay's piece).
    pub(in crate::nvn) fn prepare_parallel_draws(
        &mut self,
        list: &Arc<Vec<Command>>,
        index: usize,
        budget: Option<usize>,
        core: &Guest,
    ) {
        let address = Arc::as_ptr(list) as usize;
        self.parallel.position = Some((address, index));
        if self.parallel.covered.is_some_and(|(covered, start, end)| {
            covered == address && (start..end).contains(&index)
        }) {
            return;
        }
        let Some(command) = list.get(index) else {
            return;
        };
        if step(command) != Step::Draw
            || self.gpu.is_none()
            || draw_switches_set()
            || matches!(core.memory, GuestMemory::Checked(_))
        {
            return;
        }
        // Other modes of these change which bindings are read in place.
        static PLANNABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*PLANNABLE.get_or_init(|| {
            super::super::debug_var_os("ECHORA_UNIFORMS_IN_PLACE").is_none_or(|mode| mode != "all")
        }) {
            return;
        }
        // The window: state changes and draws up to other work.
        let most = window_commands().min(budget.unwrap_or(usize::MAX).max(1));
        let mut end = index;
        let mut draws = Vec::new();
        while end < list.len() && end - index < most {
            match step(&list[end]) {
                Step::End => break,
                Step::Draw => draws.push(end),
                _ => {}
            }
            end += 1;
        }
        self.parallel.plans.clear();
        self.parallel.covered = Some((address, index, end.max(index + 1)));
        self.parallel.stamp = None;
        if draws.len() < FEWEST_DRAWS {
            self.gpu
                .as_mut()
                .unwrap()
                .stat("stat: parallel draw windows too short", 1);
            return;
        }
        let started = std::time::Instant::now();
        // What the binding loop would do at its first texture first: the
        // planners then read the memo as the replay would.
        self.refresh_binding_memo();
        let program_stamp = (self.presented_frames, self.flush_epoch);
        if let Some(backend) = self.gpu.as_ref() {
            // Built now: the planners read it as it is.
            backend.resident_writes_pending(&(0..0));
        }
        let stamp = self.parallel_stamp();
        let threads = threads();
        let pool = self
            .parallel
            .pool
            .take()
            .unwrap_or_else(|| Pool::new(threads));
        let chunks = (threads + 1).min(draws.len());
        let results: Vec<Mutex<Vec<(usize, PlanResult)>>> =
            (0..chunks).map(|_| Mutex::new(Vec::new())).collect();
        {
            let pools = self.indexed_pool_ranges();
            let backend = self.gpu.as_ref().unwrap();
            let dirty = backend.dirty_index.borrow();
            let completed = backend.renderer.completed_serial();
            let index_held = (backend.page_table_writes
                && backend.page_table_write_serial > completed)
                .then(|| self.shared_guest_index())
                .flatten();
            let memory = match &core.memory {
                GuestMemory::View(view) => &**view,
                // Checked reads report faults the view's do not.
                GuestMemory::Checked(_) => unreachable!("checked above"),
            };
            let view = View {
                memory,
                programs: (self.program_memo.0 == program_stamp).then_some(&self.program_memo.1),
                pools: &pools.entries,
                memo: &self.binding_memo.1,
                block_aliases: &self.block_aliases,
                presented_frames: self.presented_frames,
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
            };
            let render = &self.render;
            let draws = &draws;
            let results = &results;
            let commands: &[Command] = list;
            let first = index;
            pool.run(chunks, &|chunk| {
                // An even share of the draws, each from its own state.
                let from = draws.len() * chunk / chunks;
                let to = draws.len() * (chunk + 1) / chunks;
                let mut state = render.clone();
                let mut out = Vec::with_capacity(to - from);
                let mut next_draw = from;
                for (at, command) in commands
                    .iter()
                    .enumerate()
                    .take(draws[to - 1] + 1)
                    .skip(first)
                {
                    let kind = step(command);
                    if kind == Step::Draw {
                        if next_draw < to && draws[next_draw] == at {
                            out.push((at, plan_draw(&view, &state)));
                            next_draw += 1;
                        }
                        continue;
                    }
                    apply(&mut state, command, kind);
                }
                *results[chunk].lock().unwrap() = out;
            });
        }
        self.parallel.pool = Some(pool);
        for result in results {
            self.parallel
                .plans
                .extend(result.into_inner().unwrap_or_default());
        }
        self.parallel.stamp = stamp;
        let backend = self.gpu.as_mut().unwrap();
        backend.stat("stat: parallel draw windows", 1);
        backend.stat("stat: parallel draws planned", draws.len() as u64);
        backend.stat(
            "stat: parallel draw planning us",
            started.elapsed().as_micros() as u64,
        );
    }

    /// The plan for the draw being executed, when one was made and the
    /// state it read still holds.
    pub(super) fn take_planned_bindings(
        &mut self,
        program: &Arc<BoundProgram>,
    ) -> Option<PlannedBindings> {
        if !enabled() {
            return None;
        }
        let (address, index) = self.parallel.position?;
        let (covered, start, end) = self.parallel.covered?;
        if covered != address || !(start..end).contains(&index) {
            return None;
        }
        // Plans of draws passed over (not drawn: render disabled, skipped).
        while self
            .parallel
            .plans
            .front()
            .is_some_and(|(planned, _)| *planned < index)
        {
            self.parallel.plans.pop_front();
        }
        if self
            .parallel
            .plans
            .front()
            .is_none_or(|(planned, _)| *planned != index)
        {
            return None;
        }
        let (_, plan) = self.parallel.plans.pop_front().unwrap();
        if self.parallel.stamp.is_none() || self.parallel_stamp() != self.parallel.stamp {
            // Something the plans read changed: the rest are made again.
            self.parallel.plans.clear();
            self.parallel.covered = None;
            self.gpu
                .as_mut()
                .unwrap()
                .stat("stat: parallel draw plans dropped by a state change", 1);
            return None;
        }
        let backend = self.gpu.as_mut().unwrap();
        match plan {
            Ok(plan) if Arc::ptr_eq(&plan.program, program) => {
                backend.stat("stat: parallel draws used", 1);
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
        let PlannedBindings {
            program: _,
            uniforms,
            guest_uniforms,
            textures,
            sampled_textures,
            fragment_constants,
            storage,
            storage_written,
            storage_entries,
            uniform_bytes,
            memo_hits,
            memo_checks,
        } = plan;
        // As the binding loop's lookups would have (the same check, under
        // the same stamp).
        for handle in memo_checks {
            if self.with_memoized_binding(handle, |_| ()).is_none() {
                return Err(format!(
                    "planned texture binding {handle:#x} no longer holds"
                ));
            }
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
        for binding in storage {
            if binding.read_only
                && let Some((buffer, offset)) =
                    self.shared_binding(binding.guest, binding.length)?
            {
                out.storage_in_place
                    .insert(binding.key, (buffer, offset, binding.length));
                self.gpu
                    .as_mut()
                    .unwrap()
                    .stat("stat: draw storage shared", 1);
                continue;
            }
            self.gpu
                .as_mut()
                .unwrap()
                .stat("stat: draw storage bytes", binding.length);
            out.storage.insert(
                binding.key,
                self.gpu_bytes(core, binding.base, binding.length)?,
            );
        }
        Ok(())
    }

    /// ECHORA_PARALLEL_DRAWS_VERIFY: compare a plan with what the replay's
    /// own binding loop resolved, reporting the first differences.
    pub(super) fn verify_planned_bindings(&mut self, plan: &PlannedBindings, got: Resolved<'_>) {
        let mut differences = Vec::new();
        let sorted = |map: Vec<String>| {
            let mut map = map;
            map.sort();
            map
        };
        if sorted(
            plan.uniforms
                .iter()
                .map(|(k, v)| format!("{k:?}{v:?}"))
                .collect(),
        ) != sorted(
            got.uniforms
                .iter()
                .map(|(k, v)| format!("{k:?}{v:?}"))
                .collect(),
        ) {
            differences.push("uniforms");
        }
        if sorted(
            plan.guest_uniforms
                .iter()
                .map(|e| format!("{e:?}"))
                .collect(),
        ) != sorted(
            got.guest_uniforms
                .iter()
                .map(|e| format!("{e:?}"))
                .collect(),
        ) {
            differences.push("guest uniforms");
        }
        if sorted(plan.textures.iter().map(|e| format!("{e:?}")).collect())
            != sorted(got.textures.iter().map(|e| format!("{e:?}")).collect())
        {
            differences.push("textures");
        }
        if format!("{:?}", plan.sampled_textures) != format!("{:?}", got.sampled_textures) {
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
        let mut planned_keys: Vec<(usize, u32)> = plan.storage.iter().map(|s| s.key).collect();
        planned_keys.sort_unstable();
        if storage_keys != planned_keys {
            differences.push("storage buffers");
        }
        if sorted(
            plan.storage_written
                .iter()
                .map(|e| format!("{e:?}"))
                .collect(),
        ) != sorted(
            got.storage_written
                .iter()
                .map(|e| format!("{e:?}"))
                .collect(),
        ) {
            differences.push("written storage buffers");
        }
        if plan.storage_entries != *got.storage_entries {
            differences.push("driver storage entries");
        }
        if plan.uniform_bytes != got.uniform_bytes || plan.memo_hits != got.memo_hits {
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
        "uniform buffer written by unfinished GPU work" => {
            "stat: parallel draw fallbacks: uniform buffer written by unfinished GPU work"
        }
        "uniform buffer holds GPU results to store" => {
            "stat: parallel draw fallbacks: uniform buffer holds GPU results to store"
        }
        "storage buffer not bound" => "stat: parallel draw fallbacks: storage buffer not bound",
        "storage buffer in no pool" => "stat: parallel draw fallbacks: storage buffer in no pool",
        "storage buffer holds GPU results to store" => {
            "stat: parallel draw fallbacks: storage buffer holds GPU results to store"
        }
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

/// Threads that plan draws with the replay thread.
struct Pool {
    shared: Arc<PoolShared>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

impl Pool {
    fn new(threads: usize) -> Self {
        let shared = Arc::new(PoolShared {
            slot: Mutex::new(Slot::default()),
            wake: Condvar::new(),
            done: Condvar::new(),
            next: AtomicUsize::new(0),
            panicked: AtomicBool::new(false),
        });
        let threads = (0..threads)
            .map(|index| {
                let shared = shared.clone();
                std::thread::Builder::new()
                    .name(format!("nvn-plan-{index}"))
                    .spawn(move || Self::work(&shared))
                    .expect("draw planning thread")
            })
            .collect();
        Self { shared, threads }
    }

    #[allow(unsafe_code)]
    fn work(shared: &PoolShared) {
        let mut seen = 0;
        loop {
            let (chunks, job) = {
                let mut slot = shared.slot.lock().unwrap();
                loop {
                    if slot.stop {
                        return;
                    }
                    if slot.generation != seen
                        && let Some(job) = slot.job
                    {
                        seen = slot.generation;
                        slot.busy += 1;
                        break job;
                    }
                    slot = shared.wake.wait(slot).unwrap();
                }
            };
            // SAFETY: `run` keeps the job alive until `busy` is back to 0.
            let job = unsafe { &*job };
            Self::take_chunks(shared, chunks, job);
            let mut slot = shared.slot.lock().unwrap();
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
    fn run(&self, chunks: usize, job: &Job<'_>) {
        let shared = &*self.shared;
        shared.next.store(0, Ordering::Release);
        shared.panicked.store(false, Ordering::Release);
        // SAFETY: the lifetime is erased only for the threads, which leave
        // the job before this function returns (it waits for `busy`).
        let erased: *const Job<'static> =
            unsafe { std::mem::transmute::<*const Job<'_>, *const Job<'static>>(job) };
        {
            let mut slot = shared.slot.lock().unwrap();
            slot.generation += 1;
            slot.job = Some((chunks, erased));
        }
        shared.wake.notify_all();
        Self::take_chunks(shared, chunks, job);
        let mut slot = shared.slot.lock().unwrap();
        // No thread takes the job once it is withdrawn; those inside it
        // finish first.
        slot.job = None;
        while slot.busy > 0 {
            slot = shared.done.wait(slot).unwrap();
        }
        drop(slot);
        if shared.panicked.load(Ordering::Acquire) {
            panic!("a draw planning thread panicked");
        }
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
        let pool = Pool::new(3);
        for round in 0..50 {
            let chunks = 1 + round % 7;
            let counts: Vec<AtomicUsize> = (0..chunks).map(|_| AtomicUsize::new(0)).collect();
            pool.run(chunks, &|chunk| {
                counts[chunk].fetch_add(1, Ordering::Relaxed);
            });
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
            Step::End
        );
        assert_eq!(
            step(&command("nvnCommandBufferSetTexturePool", 0)),
            Step::End
        );
        assert_eq!(step(&command("nvnCommandBufferClearColor", 0)), Step::End);
        assert_eq!(
            step(&command("nvnCommandBufferDispatchCompute", 0)),
            Step::End
        );
        assert_eq!(step(&command("nvnCommandBufferCallCommands", 0)), Step::End);
        assert_eq!(step(&command("nvnCommandBufferWaitEvent", 0)), Step::End);
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
        let mut nvn = fixture()?;
        let memory = memory();
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
        nvn.pending
            .entry(QUEUE)
            .or_default()
            .push_back(QueueItem::List {
                commands: Arc::new(batch()),
                next: 0,
                descriptor_epoch: 0,
            });
        testing::THREADS.with(|cell| cell.set(Some(threads)));
        testing::VERIFY.with(|cell| cell.set(verify));
        testing::RECORD.with(|record| *record.borrow_mut() = Some(Vec::new()));
        let result = nvn.run_queue(QUEUE, &mut core, true);
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
        let core = Guest {
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
            apply(&mut nvn.render, command, step(command));
        }
        nvn.prepare_parallel_draws(&list, first, None, &core);
        assert_eq!(stat(&nvn, "stat: parallel draws planned"), 6);
        let program = nvn.program_memo.1.values().next().unwrap().1.clone();
        assert!(nvn.take_planned_bindings(&program).is_some());
        // GPU work recorded since may have written what the next plans read.
        nvn.gpu.as_mut().unwrap().gpu_writes += 1;
        let second = first
            + 1
            + list[first + 1..]
                .iter()
                .position(|command| step(command) == Step::Draw)
                .unwrap();
        nvn.parallel.position = Some((Arc::as_ptr(&list) as usize, second));
        assert!(nvn.take_planned_bindings(&program).is_none());
        assert_eq!(
            stat(&nvn, "stat: parallel draw plans dropped by a state change"),
            1
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
        let core = Guest {
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
            apply(&mut nvn.render, command, step(command));
        }
        nvn.prepare_parallel_draws(&list, first, None, &core);
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
}
