//! Guest state shared by the runtime and generated code.
//!
//! Generated region crates depend only on this crate, so changes to services,
//! the scheduler or the kernel model never force them to recompile. Generated
//! code runs against [`Core`]; anything that needs the host (an SVC, or a block
//! boundary while scheduling or exit work is pending) returns a control
//! transfer through [`Trap`] to the runtime's host loop.

// Unsafe code is denied except where a module allows it with a stated
// invariant (the TLB's direct page references in `memory`).
#![deny(unsafe_code)]
// Machine state types are large and not meaningfully printable.
#![allow(missing_debug_implementations)]

pub mod cpu;
mod host_memory;
pub mod memory;
pub mod simd;
mod trap;
pub mod view;

use serde::Serialize;
use std::time::{Duration, Instant};

pub use trap::Trap;

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct ThreadPc {
    pub thread_id: u64,
    pub pc: u64,
}

const RECENT_PCS: usize = 64;

/// Bit position of the instruction count in `Core::block_tag`.
const BLOCK_TAG_SHIFT: u64 = 40;

/// Trace hits kept in full; later hits are only counted.
pub const TRACE_LIMIT: usize = 1 << 20;

/// Entry into a block that the lifter was asked to trace: the thread, its
/// argument registers and link register, and the instruction count.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct TraceHit {
    pub thread_id: u64,
    pub pc: u64,
    pub steps: u64,
    pub x: [u64; 4],
    pub lr: u64,
}

/// CNTPCT_EL0 frequency: 31.25 MHz on the Switch 2 (Eden-NS2's guest
/// counter frequency). The SDK converts ticks with this constant, so the
/// Switch 1 value (19.2 MHz) ran the game's clock at 61 %.
pub const PHYSICAL_COUNTER_FREQUENCY_HZ: u64 = 31_250_000;

pub fn physical_ticks_for_elapsed(elapsed: Duration) -> u64 {
    // Even Duration::MAX * this fixed frequency fits u128. Round down before
    // the intentional modulo-2^64 cast; neither floats nor guest steps enter.
    (elapsed.as_nanos() * u128::from(PHYSICAL_COUNTER_FREQUENCY_HZ) / 1_000_000_000) as u64
}

#[derive(Clone)]
pub struct PhysicalCounter {
    epoch: Instant,
    maximum_elapsed: Duration,
    pub read_count: u64,
    pub last_value: Option<u64>,
    pub backward_samples_clamped: u64,
    /// When set, guest time follows executed instructions at this rate
    /// instead of host time, so a slower host does not look like a slower
    /// console to code that budgets work by time.
    pub instructions_per_second: Option<u64>,
}

impl Default for PhysicalCounter {
    fn default() -> Self {
        Self {
            epoch: Instant::now(),
            maximum_elapsed: Duration::ZERO,
            read_count: 0,
            last_value: None,
            backward_samples_clamped: 0,
            instructions_per_second: None,
        }
    }
}

impl PhysicalCounter {
    pub fn sample_elapsed(&mut self, elapsed: Duration) -> u64 {
        if elapsed < self.maximum_elapsed {
            self.backward_samples_clamped = self.backward_samples_clamped.saturating_add(1);
        } else {
            self.maximum_elapsed = elapsed;
        }
        // Clamp unwrapped elapsed time, not the final register value: a real
        // modulo wrap must still be possible. Equal samples remain equal.
        let ticks = physical_ticks_for_elapsed(self.maximum_elapsed);
        self.read_count = self.read_count.saturating_add(1);
        self.last_value = Some(ticks);
        ticks
    }

    /// The counter after `steps` guest instructions in all.
    pub fn read(&mut self, steps: u64) -> u64 {
        let elapsed = match self.instructions_per_second {
            Some(rate) => Duration::from_nanos(
                (u128::from(steps) * 1_000_000_000 / u128::from(rate.max(1))) as u64,
            ),
            None => self.epoch.elapsed(),
        };
        self.sample_elapsed(elapsed)
    }
}

/// Spin polls a thread makes before it yields (see [`Core::spin_hint`]).
pub const SPIN_POLL_LIMIT: u32 = 32;

/// A translated window: runs blocks from the PC until it leaves the window.
pub type WindowFn = fn(&mut Core) -> Result<(), Trap>;

/// Every window function by window index (guest address >> `shift`), for
/// direct calls between windows (`Core::call`). Installed once by the
/// generated dispatcher.
struct WindowTable {
    shift: u32,
    first: u64,
    functions: Vec<Option<WindowFn>>,
}

static WINDOWS: std::sync::OnceLock<WindowTable> = std::sync::OnceLock::new();

/// Install the window functions (window index, function) of `shift`-bit
/// windows. Later calls are ignored.
pub fn install_windows(shift: u32, windows: &[(u64, WindowFn)]) {
    WINDOWS.get_or_init(|| {
        let first = windows.iter().map(|(index, _)| *index).min().unwrap_or(0);
        let last = windows.iter().map(|(index, _)| *index).max().unwrap_or(0);
        let mut functions = vec![None; (last - first + 1) as usize];
        for (index, function) in windows {
            functions[(index - first) as usize] = Some(*function);
        }
        WindowTable {
            shift,
            first,
            functions,
        }
    });
}

/// Nesting limit of `Core::call` (host stack frames per guest call).
const MAX_CALL_DEPTH: u32 = 48;

pub struct Core {
    pub cpu: cpu::Cpu,
    pub memory: view::GuestView,
    pub steps: u64,
    pub max_steps: u64,
    pub current_thread_id: u64,
    /// `steps` when the host last credited instructions to the active
    /// thread's record; the difference is what it has started since.
    pub thread_steps_base: u64,
    pub completed_blocks: u64,
    /// Set by the host when the next block boundary needs host work (it
    /// then also sets `host_check_at` to 0).
    pub attention: bool,
    /// Step count at which the next block boundary returns to the host:
    /// slices, timed waits, `attention`. Starts at 0 so the first boundary
    /// reaches the host.
    pub host_check_at: u64,
    /// Set by another host thread when a better-priority guest thread waits
    /// for this core: the next block boundary returns to the host, which
    /// reschedules (the host clears it there).
    pub preempt: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Whether other host threads run guest code too (only then can
    /// `preempt` be set, so single-threaded runs skip the shared load).
    pub parallel: bool,
    /// Spin-poll hints (YIELD, WFE, a compare-and-swap that cannot change
    /// memory) since the host last scheduled; past SPIN_POLL_LIMIT the
    /// thread gives up the processor at its next block boundary.
    pub spin_polls: u32,
    /// `steps` when the host last scheduled this core (its slice start).
    pub slice_start: u64,
    /// Ring of the last RECENT_PCS instructions; `recent_next` counts every
    /// instruction ever recorded.
    recent: [ThreadPc; RECENT_PCS],
    recent_next: usize,
    /// The block charged by the latest `tick_block`, so a trap inside it can
    /// refund the unexecuted tail: its last instruction's address in the
    /// low 40 bits (guest addresses fit) and its instruction count above;
    /// 0 when no block is charged. One store per block.
    block_tag: u64,
    pub physical_counter: PhysicalCounter,
    /// Hits of traced blocks (see [`Core::trace_hit`]), the first
    /// TRACE_LIMIT in full; `trace_count` counts them all.
    pub trace: Vec<TraceHit>,
    pub trace_count: u64,
    /// Guest calls running in nested host frames (`call`).
    call_depth: u32,
}

impl Core {
    pub fn new(memory: view::GuestView) -> Self {
        Self {
            cpu: cpu::Cpu::default(),
            memory,
            steps: 0,
            max_steps: 0,
            current_thread_id: 0,
            thread_steps_base: 0,
            completed_blocks: 0,
            attention: true,
            host_check_at: 0,
            preempt: Default::default(),
            parallel: false,
            spin_polls: 0,
            slice_start: 0,
            recent: [ThreadPc::default(); RECENT_PCS],
            recent_next: 0,
            block_tag: 0,
            physical_counter: PhysicalCounter::default(),
            trace: Vec::new(),
            trace_count: 0,
            call_depth: 0,
        }
    }

    /// A guest call (BL/BLR) that just set the PC to its callee and the
    /// link register to `return_to`: run the callee's windows in nested
    /// host frames until it returns there, instead of returning to the
    /// dispatcher at every window exit. Only an optimization: the PC alone
    /// says where execution continues, so returning early (no window for
    /// the PC, the nesting limit, a trap unwinding to the host) leaves the
    /// rest to the dispatcher.
    #[inline]
    pub fn call(&mut self, return_to: u64) -> Result<(), Trap> {
        let Some(table) = WINDOWS.get() else {
            return Ok(());
        };
        if self.call_depth >= MAX_CALL_DEPTH {
            return Ok(());
        }
        self.call_depth += 1;
        let result = loop {
            let pc = self.cpu.pc;
            if pc == return_to {
                break Ok(());
            }
            let function = (pc >> table.shift)
                .checked_sub(table.first)
                .and_then(|index| table.functions.get(index as usize).copied().flatten());
            let Some(function) = function else {
                break Ok(());
            };
            // Each window runs at least one block (or traps), and every
            // block boundary checks for host work: this loop yields.
            if let Err(trap) = function(self) {
                break Err(trap);
            }
        };
        self.call_depth -= 1;
        result
    }

    /// Per-instruction accounting. Thread-state checks happen in the host at
    /// block boundaries, which is the only place that state can change. The
    /// common path is a budget compare, an alignment check and a ring store.
    /// Execute permission is checked by the host where it enters generated
    /// code (see `enter_window`).
    #[inline(always)]
    pub fn tick(&mut self) -> Result<(), Trap> {
        let pc = self.cpu.pc;
        if self.steps >= self.max_steps || pc & 3 != 0 {
            return self.tick_slow();
        }
        self.count_instruction(pc);
        Ok(())
    }

    /// Entry into translated code at the current PC: alignment and the
    /// recent-PC ring. Split window functions call this once on entry;
    /// single dispatch functions call it for every block. The execute
    /// permission of the PC's page is the host's to check before it enters
    /// generated code: translated blocks only branch to other translated
    /// code, which the loader maps executable, and permissions change only
    /// in SVCs, which return to the host.
    #[inline(always)]
    pub fn enter_window(&mut self) -> Result<(), Trap> {
        let pc = self.cpu.pc;
        if pc & 3 != 0 {
            return self.enter_window_slow();
        }
        self.record_recent(pc);
        Ok(())
    }

    #[cold]
    #[inline(never)]
    fn enter_window_slow(&mut self) -> Result<(), Trap> {
        let pc = self.cpu.pc;
        if !pc.is_multiple_of(4) {
            return Err(Trap::InvalidState {
                message: format!("unaligned instruction address {pc:#x}"),
            });
        }
        self.record_recent(pc);
        Ok(())
    }

    #[inline(always)]
    fn record_recent(&mut self, pc: u64) {
        self.recent[self.recent_next % RECENT_PCS] = ThreadPc {
            thread_id: self.current_thread_id,
            pc,
        };
        self.recent_next = self.recent_next.wrapping_add(1);
    }

    /// Accounting for a straight-line block of `count` instructions starting
    /// at the current PC, charged once on entry. Generated code enters blocks
    /// only at their start, and SVCs and host boundaries end blocks, so budget
    /// and preemption keep their meaning at block granularity. Execute checks
    /// happen in `enter_window`.
    #[inline(always)]
    pub fn tick_block(&mut self, count: u64) -> Result<(), Trap> {
        self.tick_block_at(self.cpu.pc, count)
    }

    /// `tick_block` for a block at `pc`, which must be the current PC;
    /// translated code passes it as a constant, saving the PC's reload.
    #[inline(always)]
    pub fn tick_block_at(&mut self, pc: u64, count: u64) -> Result<(), Trap> {
        if self.steps + count > self.max_steps {
            return self.block_budget_exhausted();
        }
        self.steps += count;
        self.block_tag = (pc + 4 * (count - 1)) | count << BLOCK_TAG_SHIFT;
        Ok(())
    }

    #[cold]
    #[inline(never)]
    fn block_budget_exhausted(&mut self) -> Result<(), Trap> {
        // Nothing of this block is charged.
        self.block_tag = 0;
        Err(Trap::Budget {
            pc: self.cpu.pc,
            steps: self.steps,
        })
    }

    #[inline(always)]
    fn count_instruction(&mut self, pc: u64) {
        self.steps += 1;
        self.recent[self.recent_next % RECENT_PCS] = ThreadPc {
            thread_id: self.current_thread_id,
            pc,
        };
        self.recent_next = self.recent_next.wrapping_add(1);
    }

    /// Budget and alignment.
    #[cold]
    #[inline(never)]
    fn tick_slow(&mut self) -> Result<(), Trap> {
        let pc = self.cpu.pc;
        if self.steps >= self.max_steps {
            return Err(Trap::Budget {
                pc,
                steps: self.steps,
            });
        }
        if !pc.is_multiple_of(4) {
            return Err(Trap::InvalidState {
                message: format!("unaligned instruction address {pc:#x}"),
            });
        }
        self.count_instruction(pc);
        Ok(())
    }

    /// Called by the host when a trap leaves generated code: an instruction
    /// that trapped inside the current block keeps its own count, and the
    /// instructions after it, charged on block entry, are refunded. The PC of
    /// a trapping instruction is exact, so counts after faults and SVCs match
    /// per-instruction accounting.
    pub fn settle_block_trap(&mut self) {
        let pc = self.cpu.pc;
        let end = self.block_tag & ((1 << BLOCK_TAG_SHIFT) - 1);
        let count = self.block_tag >> BLOCK_TAG_SHIFT;
        if count != 0 && (end - 4 * (count - 1)..=end).contains(&pc) {
            let unexecuted = (end - pc) / 4;
            self.steps -= unexecuted;
        }
        self.block_tag = 0;
    }

    /// The most recent instructions or block starts, oldest first (any thread).
    pub fn recent_thread_pcs(&self) -> Vec<ThreadPc> {
        let count = self.recent_next.min(RECENT_PCS);
        (self.recent_next - count..self.recent_next)
            .map(|index| self.recent[index % RECENT_PCS])
            .collect()
    }

    /// The most recent instruction addresses, oldest first (any thread).
    pub fn recent_pcs(&self) -> impl Iterator<Item = u64> + '_ {
        self.recent_thread_pcs().into_iter().map(|entry| entry.pc)
    }

    /// SVC: the host performs the call, sets the PC to the next instruction
    /// and completes the block.
    #[inline]
    pub fn svc(&mut self, number: u16) -> Result<(), Trap> {
        Err(Trap::SupervisorCall { number })
    }

    /// Read the modeled physical counter once (MRS CNTPCT_EL0).
    #[inline]
    pub fn physical_counter_ticks(&mut self) -> u64 {
        self.physical_counter.read(self.steps)
    }

    /// A spin-wait poll: a thread polling memory another thread must change
    /// gets nowhere on this single host processor, so after a few polls its
    /// slice ends at the next block boundary.
    #[inline(always)]
    pub fn spin_hint(&mut self) {
        self.spin_polls += 1;
        if self.spin_polls >= SPIN_POLL_LIMIT {
            self.host_check_at = self.host_check_at.min(self.steps);
        }
    }

    /// Record entry into a traced block. The lifter emits this call only at
    /// the block starts it was asked to trace.
    #[cold]
    #[inline(never)]
    pub fn trace_hit(&mut self) {
        self.trace_count += 1;
        if self.trace.len() < TRACE_LIMIT {
            let regs = &self.cpu.regs;
            self.trace.push(TraceHit {
                thread_id: self.current_thread_id,
                pc: self.cpu.pc,
                steps: self.steps,
                x: [regs[0], regs[1], regs[2], regs[3]],
                lr: regs[30],
            });
        }
    }

    #[inline]
    pub fn finish_block(&mut self) -> Result<(), Trap> {
        if self.steps >= self.host_check_at
            || (self.parallel && self.preempt.load(std::sync::atomic::Ordering::Relaxed))
        {
            return Err(Trap::HostBoundary);
        }
        Ok(())
    }
}
