use crate::cpu::Cpu;
use crate::memory::{
    MapMemoryError, Memory, PAGE_SIZE, READ, STATE_CODE_DATA, STATE_FREE, STATE_NORMAL, WRITE,
};
use crate::services::Services;
use crate::synchronization::{MutexWaits, ThreadPriority, ZeroHandleWaits};
use crate::threads::{ThreadCollection, ThreadCreate, ThreadState};
pub use echora_guest::Core;
use serde::Serialize;
use std::collections::BTreeMap;
#[cfg(all(test, any(target_os = "linux", windows)))]
use std::time::Duration;

pub use echora_guest::Trap;

/// An explicit prototype choice, not a measured Switch 2 address-space layout.
/// No remote address range is reserved and no remote mapping operation exists.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RemoteMemoryModel {
    #[default]
    Unsupported,
    AbsentRegionCompatibility,
}

/// Horizon kernel results (module 1) returned by the physical memory SVCs.
const RESULT_INVALID_SIZE: u64 = 0xca01;
const RESULT_INVALID_ADDRESS: u64 = 0xcc01;
const RESULT_INVALID_MEMORY_REGION: u64 = 0xd401;
const RESULT_LIMIT_REACHED: u64 = 0x10801;
/// Label of alias-region mappings made by MapPhysicalMemory.
const PHYSICAL_LABEL: &str = "physical memory";

#[derive(Clone, Debug, Serialize)]
pub struct Layout {
    pub code_base: u64,
    pub address_space_end: u64,
    pub heap_base: u64,
    pub heap_capacity: u64,
    pub alias_base: u64,
    pub alias_size: u64,
    /// Extra alias capacity is absent from the prototype's fixed layout.
    pub alias_extra_size: u64,
    pub stack_base: u64,
    pub stack_size: u64,
    pub stack_region_base: u64,
    pub stack_region_size: u64,
    pub tls_base: u64,
    pub total_memory: u64,
    pub initial_used_memory: u64,
    /// NPDM SystemResourceSize: the process's private kernel resource pool
    /// (secure memory for kernel objects such as page tables). It counts as
    /// used user memory but not as non-system memory.
    pub system_resource_size: u64,
    pub main_thread_handle: u64,
    pub main_thread_priority: u8,
    /// Stable logical identity in this single-process model, separate from its handle.
    pub main_thread_id: u64,
    /// Logical runtime identity, not the NPDM application/program identifier.
    pub process_id: u64,
    pub processor_id: u8,
    /// Process default virtual core, independent of the executing thread's core.
    pub process_ideal_core_id: u8,
    /// Allowed virtual cores from validated ACI0 KernelFlags; never host CPU count.
    /// None means no permissions have been supplied to this synthetic machine.
    pub process_core_mask: Option<u64>,
    /// Allowed priorities from the same validated ACI0 KernelFlags grant.
    pub process_priority_mask: Option<u64>,
    pub remote_memory_model: RemoteMemoryModel,
    /// Instructions a thread runs before another runnable thread gets the
    /// processor; 0 switches at every block boundary.
    #[serde(skip)]
    pub scheduler_slice: u64,
    /// `scheduler_slice` with several workers.
    #[serde(skip)]
    pub parallel_slice: u64,
    /// Host threads running guest threads at once; 1 runs every guest
    /// thread on the calling thread.
    #[serde(skip)]
    pub workers: usize,
}

impl Default for Layout {
    fn default() -> Self {
        Self {
            code_base: 0x7100000000,
            address_space_end: 1 << 39,
            heap_base: 0x100000000,
            // Must cover total memory minus initial use: applications size
            // their heap as GetInfo total - used (observed: 0x2fb000000).
            // Guest pages are sparse, so unused capacity costs nothing.
            heap_capacity: 12 << 30,
            alias_base: 0x1000000000,
            alias_size: 0x800000000,
            alias_extra_size: 0,
            stack_base: 0x7ff0000000 - 0x200000,
            stack_size: 0x200000,
            stack_region_base: (1 << 39) - (2 << 30),
            stack_region_size: 2 << 30,
            tls_base: 0x7000000000,
            total_memory: 12 << 30,
            initial_used_memory: 0,
            system_resource_size: 0,
            main_thread_handle: 0x100,
            main_thread_priority: 44,
            main_thread_id: 1,
            process_id: 1,
            processor_id: 0,
            process_ideal_core_id: 0,
            process_core_mask: None,
            process_priority_mask: None,
            remote_memory_model: RemoteMemoryModel::Unsupported,
            scheduler_slice: 0,
            parallel_slice: 0,
            workers: 1,
        }
    }
}

impl Layout {
    /// The process local region page, directly below the main thread's TLS.
    pub fn process_local_region(&self) -> u64 {
        self.tls_base - PAGE_SIZE
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct SvcEvent {
    pub thread_id: u64,
    pub pc: u64,
    /// X30 at the SVC: in the SDK's leaf wrappers, the caller's return.
    pub lr: u64,
    pub number: u16,
    pub arguments: [u64; 4],
    /// CreateThread also consumes W4/W5. Retain the raw X-register values so
    /// diagnostics can distinguish the ABI's low-32-bit interpretation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments_4_5: Option<[u64; 2]>,
    pub detail: Option<String>,
}

pub use echora_guest::ThreadPc;

#[derive(Clone, Debug, Serialize)]
struct ContextSwitch {
    from_thread_id: u64,
    to_thread_id: u64,
    instruction_count: u64,
    source_pc: u64,
    destination_pc: u64,
    reason: &'static str,
}

// Pinned Eden CNTFREQ and libnx's 12/625 ns-to-tick conversion agree on
// this compatibility frequency. It is not a measured Switch 2 clock rate.

// Pinned Atmosphere's kernel interval is checked before alignment. The top
// 2 MiB is outside that half-open interval; ordinary userspace access checks
// still reject it afterwards. This prototype has no shadow-stack region.
const KERNEL_VIRTUAL_ADDRESS_BASE: u64 = 0xffff_ff80_0000_0000;
const KERNEL_VIRTUAL_ADDRESS_END: u64 = 0xffff_ffff_ffe0_0000;
const MUTEX_WAIT_MASK: u32 = 0x4000_0000;
/// The Switch 2's six application cores (affinity mask bits 0-5).
const APPLICATION_CORES: u64 = 0x3f;
/// Thread-local region fields after the IPC message buffer.
const TLS_DISABLE_COUNT: u64 = 0x100;
const TLS_INTERRUPT_FLAG: u64 = 0x102;

pub use echora_guest::{
    PHYSICAL_COUNTER_FREQUENCY_HZ, PhysicalCounter, WindowFn, install_windows,
    physical_ticks_for_elapsed,
};

fn physical_counter_report(counter: &PhysicalCounter) -> serde_json::Value {
    // Reporting must not read the clock or change the guest-visible count.
    serde_json::json!({
        "register":"CNTPCT_EL0",
        "frequency_hz":PHYSICAL_COUNTER_FREQUENCY_HZ,
        "frequency_policy":"pinned_eden_libnx_compatibility_not_hardware_measurement",
        "source":"std_time_Instant_monotonic_elapsed",
        "epoch":"machine_creation_not_console_or_utc_boot",
        "width_bits":64,
        "rounding":"integer_floor",
        "wrap":"modulo_2_64",
        "backward_sample_policy":"retain_maximum_unwrapped_elapsed",
        "host_suspend_policy":"host_Instant_semantics_not_console_verified",
        "deterministic":false,
        "uses_guest_instruction_count":false,
        "guest_el0_read_permitted":true,
        "access_control_and_virtualization_implemented":false,
        "timer_interrupts_implemented":false,
        "virtual_counter_implemented":false,
        "frequency_register_implemented":false,
        "read_count":counter.read_count,
        "last_value":counter.last_value,
        "backward_samples_clamped":counter.backward_samples_clamped,
    })
}

/// Guest instructions between host checks of timed waits while other
/// threads run.
const HOST_TIMER_CHECK_INSTRUCTIONS: u64 = 100_000;
/// Minimum instructions between rescans of parked waits (see host_ready).
const WAKE_CHECK_INSTRUCTIONS: u64 = 5_000;
/// How long (in instructions of any thread) a thread that yielded while
/// spin-polling is passed over in favor of other runnable threads.
const SPIN_BACKOFF_INSTRUCTIONS: u64 = 200_000;
/// Polls at least this dense (one per this many instructions) mark a spin.
const SPIN_DENSITY_INSTRUCTIONS: u64 = 64;

pub struct Machine {
    /// CPU, memory and per-instruction accounting used by generated code.
    pub core: Core,
    /// Mappings, permissions and checked host-side access to guest memory;
    /// `core.memory` is the generated code's view of the same bytes.
    pub memory: Memory,
    pub layout: Layout,
    pub heap_size: u64,
    /// Bytes mapped into the alias region by MapPhysicalMemory.
    pub physical_size: u64,
    pub svc_trace: Vec<SvcEvent>,
    /// The most recent SVCs (`svc_trace` keeps only the first 4096).
    pub recent_svcs: std::collections::VecDeque<SvcEvent>,
    pub debug_messages: Vec<String>,
    pub content: Option<crate::content::ContentSet>,
    pub services: Services,
    threads: ThreadCollection,
    /// Mutable virtual affinity of the sole executing thread. The process's
    /// default ideal core and current logical processor remain independent.
    main_thread_affinity: (i32, u64),
    main_thread_handle_open: bool,
    main_thread_state: ThreadState,
    /// SetThreadPriority changes this; the layout keeps the creation value.
    main_thread_base_priority: u8,
    main_thread_effective_priority: u8,
    main_mutex_owner_references: u32,
    main_wait_cancelled: bool,
    main_saved_cpu: Option<Cpu>,
    main_thread_tls_owned: bool,
    main_instructions_started: u64,
    pending_thread_exit: Option<u64>,
    pending_mutex_wait: Option<u64>,
    pending_synchronization_wait: Option<u64>,
    mutex_waits: MutexWaits,
    synchronization_waits: ZeroHandleWaits,
    mutex_waits_started: u64,
    mutex_waits_woken: u64,
    mutex_waits_returned: u64,
    mutex_raced_lock_returns: u64,
    mutex_unlock_calls: u64,
    synchronization_waits_started: u64,
    synchronization_waits_cancelled: u64,
    condition_waits_started: u64,
    condition_signals: u64,
    synchronization_waits_returned: u64,
    synchronization_timeout_zero_returns: u64,
    synchronization_pending_cancel_returns: u64,
    synchronization_cancel_requests: u64,
    synchronization_latched_cancel_requests: u64,
    process_exit_requested: bool,
    process_exit_reason: Option<&'static str>,
    context_switches: u64,
    /// ECHORA_CHECK_CONTEXTS (diagnostic): per thread, the sequence numbers
    /// of its last context save and last dispatch. A dispatch whose saved
    /// context is older than its previous dispatch resumes a stale snapshot.
    context_sequence: std::collections::HashMap<u64, (u64, u64)>,
    context_counter: u64,
    thread_exits: u64,
    last_context_switch: Option<ContextSwitch>,
    random_entropy: Option<[u64; 4]>,
    /// High-level NVN entry points (host functions).
    pub nvn: crate::nvn::Nvn,
    /// Where the hid shared memory is mapped, and controller samples
    /// published to it (ECHORA_WINDOW).
    hid_mapping: Option<u64>,
    hid_samples: u64,
    /// (thread, PC) at each host scheduling boundary, counted: a coarse
    /// guest profile (`pc_samples` in the report).
    pc_samples: std::collections::HashMap<(u64, u64), u64>,
    /// With `ECHORA_PC_STACKS=LO-HI` (guest addresses, hex): frame-pointer
    /// return chains of the samples whose PC is in that range.
    pc_stacks: std::collections::HashMap<Vec<u64>, u64>,
    /// Boundaries that kept a thread running because it had user
    /// preemption disabled.
    user_preemption_deferrals: u64,
    /// Host streams of the game's streaming zstd reader.
    zstd_streams: crate::hle::ZstdStreams,
    /// Where a host call continues instead of its link register (a hooked
    /// call site the guest performs after all).
    host_call_continue: Option<u64>,
    pc_sample_tick: u32,
    /// Threads that gave up the processor spinning, with the step count
    /// until which other runnable threads go first.
    spin_backoff: std::collections::BTreeMap<u64, u64>,
    /// The thread each guest core last ran, by guest core, with several
    /// cores folded onto a worker: equal priorities on a core take turns
    /// after it (see `next_runnable_thread`).
    core_last_thread: [u64; 64],
    /// By guest core folded onto a worker: the thread with dispatch
    /// disabled (TLS DisableCount) that gave the worker to another core's
    /// turn. It keeps its core: nothing else runs there, and it resumes on
    /// no other, until it ends a slice with dispatch enabled.
    core_hold: [Option<u64>; 64],
    /// The thread the worker holding the machine goes on running after a
    /// boundary (set by `request_preemptions`, cleared when the worker
    /// idles): `parked_workers_with_work` wakes no one for it.
    keeps_running: Option<u64>,
    /// SignalToAddress calls of type 0 or 1 that found no waiter.
    signal_to_address_without_waiters: u64,
    /// WaitForAddress calls that returned at once (the value did not match,
    /// or timeout 0).
    wait_for_address_immediate_returns: u64,
    /// Step count of the last scan of parked waits.
    last_wake_check: u64,
    /// With several workers: the threads other workers are running now
    /// (never the thread of the core swapped into `core`).
    running: std::collections::BTreeSet<u64>,
    /// With several workers: the modelled core (0-2) of the worker whose
    /// core is swapped in. Threads are picked for it by affinity and
    /// priority (see `next_runnable_thread`).
    worker_core: Option<u8>,
    /// Times a worker took a thread another worker was running (a
    /// scheduler error; see `Worker::swap_out`).
    double_dispatches: u64,
    /// With several workers: the thread each core runs (by core) and the
    /// flags that make a core's next block boundary reschedule.
    core_threads: Vec<Option<u64>>,
    preempt_flags: Vec<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// Preemptions requested (a better-priority thread waited for a core).
    preemptions: u64,
    /// Several workers run: an idle boundary returns instead of sleeping.
    parallel: bool,
    /// Workers parked without a thread, waiting for one to become runnable.
    parked_workers: usize,
    /// Parallel boundaries: count, microseconds waiting for the machine
    /// lock, microseconds holding it.
    lock_stats: [u64; 3],
    /// Boundaries by kind: SVC numbers 0..0x80, then NVN/host calls,
    /// host boundaries, time-slice ends and other traps.
    boundary_kinds: Vec<u64>,
    /// Parallel runs: microseconds the machine lock was held for each kind
    /// of boundary (indexed as `boundary_kinds`; NVN calls that run without
    /// the lock count only their locked parts).
    boundary_held: Vec<u64>,
    /// NVN calls workers made without the machine lock (see
    /// `Worker::held_nvn_call`).
    lock_free_nvn_calls: u64,
    /// Of those, calls recorded on command buffers without even the NVN
    /// state's lock (see `Worker::recorded_nvn_call`).
    recorded_nvn_calls: u64,
    /// Microseconds those calls waited for the NVN state's own lock, and
    /// held it.
    nvn_lock_stats: [u64; 2],
    /// Microseconds each guest thread's NVN calls waited for that lock.
    nvn_thread_wait: BTreeMap<u64, u64>,
    /// Parallel runs: host microseconds per guest thread running guest
    /// code, waiting for the machine lock and holding it (for finding the
    /// thread that bounds the frame rate).
    thread_time: BTreeMap<u64, [u64; 3]>,
    /// Boundaries since the run started, for the stop-request check.
    stop_check_tick: u64,
}

impl Machine {
    pub fn new(layout: Layout, max_steps: u64) -> Self {
        let threads = ThreadCollection::new(layout.main_thread_id);
        let main_thread_affinity = (
            i32::from(layout.processor_id),
            1u64.checked_shl(u32::from(layout.processor_id))
                .unwrap_or(0),
        );
        let current_thread_id = layout.main_thread_id;
        let main_thread_effective_priority = layout.main_thread_priority;
        let memory = Memory::new(layout.address_space_end);
        let mut core = Core::new(memory.view());
        core.max_steps = max_steps;
        core.current_thread_id = current_thread_id;
        Self {
            core,
            memory,
            layout,
            heap_size: 0,
            physical_size: 0,
            svc_trace: Vec::new(),
            recent_svcs: Default::default(),
            debug_messages: Vec::new(),
            content: None,
            services: Services::default(),
            threads,
            main_thread_affinity,
            main_thread_handle_open: true,
            main_thread_state: ThreadState::Runnable,
            main_thread_base_priority: main_thread_effective_priority,
            main_thread_effective_priority,
            main_mutex_owner_references: 0,
            main_wait_cancelled: false,
            main_saved_cpu: None,
            main_thread_tls_owned: false,
            main_instructions_started: 0,
            pending_thread_exit: None,
            pending_mutex_wait: None,
            pending_synchronization_wait: None,
            mutex_waits: MutexWaits::default(),
            synchronization_waits: ZeroHandleWaits::default(),
            mutex_waits_started: 0,
            mutex_waits_woken: 0,
            mutex_waits_returned: 0,
            mutex_raced_lock_returns: 0,
            mutex_unlock_calls: 0,
            synchronization_waits_started: 0,
            synchronization_waits_cancelled: 0,
            condition_waits_started: 0,
            condition_signals: 0,
            synchronization_waits_returned: 0,
            synchronization_timeout_zero_returns: 0,
            synchronization_pending_cancel_returns: 0,
            synchronization_cancel_requests: 0,
            synchronization_latched_cancel_requests: 0,
            process_exit_requested: false,
            process_exit_reason: None,
            context_switches: 0,
            context_sequence: Default::default(),
            context_counter: 0,
            thread_exits: 0,
            last_context_switch: None,
            random_entropy: None,
            nvn: Default::default(),
            hid_mapping: None,
            hid_samples: 0,
            pc_samples: Default::default(),
            pc_stacks: Default::default(),
            user_preemption_deferrals: 0,
            zstd_streams: Default::default(),
            host_call_continue: None,
            pc_sample_tick: 0,
            spin_backoff: Default::default(),
            core_last_thread: [0; 64],
            core_hold: [None; 64],
            keeps_running: None,
            signal_to_address_without_waiters: 0,
            wait_for_address_immediate_returns: 0,
            last_wake_check: 0,
            running: Default::default(),
            worker_core: None,
            double_dispatches: 0,
            core_threads: Vec::new(),
            preempt_flags: Vec::new(),
            preemptions: 0,
            parallel: false,
            parked_workers: 0,
            lock_stats: [0; 3],
            boundary_kinds: vec![0; 0x84],
            boundary_held: vec![0; 0x84],
            lock_free_nvn_calls: 0,
            recorded_nvn_calls: 0,
            nvn_lock_stats: [0; 2],
            nvn_thread_wait: BTreeMap::new(),
            thread_time: BTreeMap::new(),
            stop_check_tick: 0,
        }
    }

    /// Horizon exposes four stable process entropy words. Draw them once from
    /// the host OS source; never substitute fixed words on an I/O failure.
    pub fn initialize_entropy_from_os(&mut self) -> std::io::Result<()> {
        let mut bytes = [0u8; 32];
        getrandom::fill(&mut bytes)?;
        self.random_entropy = Some(std::array::from_fn(|index| {
            crate::bytes::le_u64(&bytes[index * 8..index * 8 + 8])
        }));
        Ok(())
    }

    /// Read the modeled physical counter once. The AOT instruction performs its
    /// ordinary execution/budget checks first. The clock retains its epoch when
    /// execution resumes on this machine; constructing another machine resets it.
    pub fn physical_counter_ticks(&mut self) -> u64 {
        self.core.physical_counter.read(self.core.steps)
    }

    pub fn current_thread_id(&self) -> u64 {
        self.core.current_thread_id
    }

    /// Bootstrap calls this after allocating and initializing the exact main
    /// TLS page. Synthetic machines with no such allocation own no main TLS.
    pub fn register_main_thread_tls(&mut self) -> Result<(), Trap> {
        let mapping = self.memory.query(self.layout.tls_base);
        if self.main_thread_tls_owned
            || mapping.base != self.layout.tls_base
            || mapping.size != PAGE_SIZE
            || mapping.state != crate::memory::STATE_TLS
        {
            return Err(Trap::InvalidState {
                message: "main thread TLS registration requires its newly owned exact page".into(),
            });
        }
        self.main_thread_tls_owned = true;
        Ok(())
    }

    fn validate_main_thread_tls_release(&self) -> Result<(), Trap> {
        if self.main_thread_tls_owned {
            let mapping = self.memory.query(self.layout.tls_base);
            if mapping.base != self.layout.tls_base
                || mapping.size != PAGE_SIZE
                || mapping.state != crate::memory::STATE_TLS
            {
                return Err(Trap::InvalidState {
                    message: "main thread no longer owns its exact TLS mapping".into(),
                });
            }
            let base = self.layout.tls_base;
            let end = base + PAGE_SIZE;
            if self
                .memory
                .stack_aliases()
                .any(|(destination, source, size)| {
                    (base < destination + size && destination < end)
                        || (base < source + size && source < end)
                })
            {
                return Err(Trap::InvalidState {
                    message: "main TLS cleanup overlaps an active stack alias".into(),
                });
            }
        }
        Ok(())
    }

    fn release_main_thread_tls(&mut self) -> Result<(), Trap> {
        self.validate_main_thread_tls_release()?;
        if self.main_thread_tls_owned {
            self.memory.unmap_exact(self.layout.tls_base, PAGE_SIZE)?;
            self.main_thread_tls_owned = false;
        }
        Ok(())
    }

    fn thread_id_for_handle(&self, handle: u32) -> Option<u64> {
        if handle == 0xffff8000 {
            Some(self.core.current_thread_id)
        } else if self.main_thread_handle_open && handle == self.layout.main_thread_handle as u32 {
            Some(self.layout.main_thread_id)
        } else {
            self.services.thread_id(handle)
        }
    }

    /// CreateTransferMemory (Atmosphere 6b0c077 kern_svc_transfer_memory /
    /// LockForTransferMemory): page-aligned nonempty range in the address
    /// space, permission None/R/RW, every page transferable (heap or static
    /// data) with user RW and no attributes. The owner's permission becomes
    /// the requested one and the range is Locked until the handle closes.
    fn create_transfer_memory(
        &mut self,
        address: u64,
        size: u64,
        permission: u32,
    ) -> Result<(u32, u32), Trap> {
        if !address.is_multiple_of(PAGE_SIZE) {
            return Ok((0xcc01, 0));
        }
        if !size.is_multiple_of(PAGE_SIZE) || size == 0 {
            return Ok((0xca01, 0));
        }
        if address.checked_add(size).is_none() {
            return Ok((0xd401, 0));
        }
        if !matches!(permission, 0 | 1 | 3) {
            return Ok((0xd801, 0));
        }
        if address + size > self.layout.address_space_end {
            return Ok((0xd401, 0));
        }
        let locked = self.memory.lock_range(
            address,
            size,
            &[STATE_NORMAL, STATE_CODE_DATA],
            READ | WRITE,
            permission as u8,
            crate::memory::ATTRIBUTE_LOCKED,
        )?;
        if !locked {
            return Ok((0xd401, 0));
        }
        let handle = self
            .services
            .insert_transfer_memory(crate::services::TransferMemory {
                address,
                size,
                permission: permission as u8,
            });
        match handle {
            Ok(handle) => Ok((0, handle)),
            Err(error) => {
                self.memory.unlock_range(
                    address,
                    size,
                    crate::memory::ATTRIBUTE_LOCKED,
                    READ | WRITE,
                )?;
                Err(error)
            }
        }
    }

    /// Signal state of one waitable handle: a readable event is signaled while
    /// its backing event is; a thread is signaled once terminated. `None` is
    /// not a synchronization object (Horizon InvalidHandle).
    fn synchronization_object_signaled(&self, handle: u32) -> Result<Option<bool>, Trap> {
        if let Some(state) = self.services.readable_event_signaled(handle)? {
            return Ok(Some(state));
        }
        let id = if handle == 0xffff8000 {
            Some(self.core.current_thread_id)
        } else {
            self.real_thread_id_for_handle(handle)
        };
        Ok(id.map(|id| matches!(self.thread_state(id), None | Some(ThreadState::Terminated))))
    }

    fn real_thread_id_for_handle(&self, handle: u32) -> Option<u64> {
        // ArbitrateLock uses GetObjectWithoutPseudoHandle<KThread>. In
        // particular, the current-thread pseudo-handle is not a mutex owner;
        // the real table rejects both reserved high handle bits.
        if handle & 0xc000_0000 != 0 {
            None
        } else {
            self.thread_id_for_handle(handle)
        }
    }

    fn main_thread_alive(&self) -> bool {
        self.main_thread_handle_open
            || self.main_thread_state.holds_running_reference()
            || self.main_mutex_owner_references != 0
    }

    fn thread_state(&self, id: u64) -> Option<ThreadState> {
        if id == self.layout.main_thread_id {
            self.main_thread_alive().then_some(self.main_thread_state)
        } else {
            self.threads.state(id)
        }
    }

    fn waiting_thread_count(&self) -> usize {
        usize::from(self.main_thread_state == ThreadState::Waiting) + self.threads.waiting_count()
    }

    fn started_thread_count(&self) -> usize {
        usize::from(self.main_thread_state.holds_running_reference()) + self.threads.started_count()
    }

    fn thread_wait_cancelled(&self, id: u64) -> Result<bool, Trap> {
        let cancelled = if id == self.layout.main_thread_id {
            self.main_thread_alive().then_some(self.main_wait_cancelled)
        } else {
            self.threads.wait_cancelled(id)
        };
        cancelled.ok_or_else(|| Trap::InvalidState {
            message: "thread has no synchronization cancellation backing".into(),
        })
    }

    fn set_thread_wait_cancelled(&mut self, id: u64, cancelled: bool) -> Result<(), Trap> {
        if id == self.layout.main_thread_id {
            if !self.main_thread_alive() {
                return Err(Trap::InvalidState {
                    message: "main cancellation target no longer exists".into(),
                });
            }
            self.main_wait_cancelled = cancelled;
            Ok(())
        } else {
            self.threads.set_wait_cancelled(id, cancelled)
        }
    }

    /// The reached subset waits on no objects. Incoming X0/X1 therefore do not
    /// cause a userspace memory access. A nonzero count still requires future
    /// object subscriptions/references, even if some handles might be signaled.
    fn wait_synchronization_result(
        &mut self,
        handles_address: u64,
        count: i32,
        timeout_ns: i64,
    ) -> Result<Option<(u32, i32)>, Trap> {
        if !(0..=64).contains(&count) {
            return Ok(Some((0xee01, 0)));
        }
        // Kernel order: copy and resolve every handle (InvalidPointer /
        // InvalidHandle), then termination, then the first signaled object in
        // array order, then timeout zero, then the cancellation latch.
        let mut signaled = Vec::with_capacity(count as usize);
        let mut handles = Vec::with_capacity(count as usize);
        for index in 0..count as u64 {
            let Ok(handle) = self.memory.read_u32(handles_address + index * 4) else {
                return Ok(Some((0xcc01, 0)));
            };
            match self.synchronization_object_signaled(handle)? {
                Some(state) => signaled.push(state),
                None => return Ok(Some((0xe401, 0))),
            }
            handles.push(handle);
        }
        if self.process_exit_requested
            || self.pending_thread_exit == Some(self.core.current_thread_id)
        {
            return Ok(Some((0x7601, 0)));
        }
        if let Some(index) = signaled.iter().position(|state| *state) {
            return Ok(Some((0, index as i32)));
        }
        if self.thread_state(self.core.current_thread_id) != Some(ThreadState::Runnable)
            || self.pending_mutex_wait.is_some()
            || self.pending_synchronization_wait.is_some()
        {
            return Err(Trap::InvalidState {
                message: "WaitSynchronization requires the active runnable context without a pending wait".into(),
            });
        }
        // No signaled-object scan exists for count zero. Timeout zero precedes
        // the cancellation latch and must not consume a previously queued cancel.
        if timeout_ns == 0 {
            self.synchronization_timeout_zero_returns =
                self.synchronization_timeout_zero_returns.saturating_add(1);
            // A zero-timeout poll that finds nothing signaled is a spin-wait
            // (update 655360 polls one event this way in a loop): other
            // runnable threads go first for a while, as after dense polling.
            self.spin_backoff.insert(
                self.core.current_thread_id,
                self.core.steps.saturating_add(SPIN_BACKOFF_INSTRUCTIONS),
            );
            return Ok(Some((0xea01, 0)));
        }
        if self.thread_wait_cancelled(self.core.current_thread_id)? {
            self.set_thread_wait_cancelled(self.core.current_thread_id, false)?;
            self.synchronization_pending_cancel_returns = self
                .synchronization_pending_cancel_returns
                .saturating_add(1);
            return Ok(Some((0xec01, 0)));
        }
        if count == 0 && timeout_ns < 0 {
            self.synchronization_waits
                .begin(self.core.current_thread_id, timeout_ns)?;
        } else {
            // Unsignaled objects and/or a positive timeout: park until a
            // waited object is signaled or the host-monotonic deadline passes.
            let deadline = (timeout_ns > 0).then(|| {
                std::time::Instant::now() + std::time::Duration::from_nanos(timeout_ns as u64)
            });
            self.synchronization_waits.begin_objects(
                self.core.current_thread_id,
                timeout_ns,
                handles,
                deadline,
            )?;
        }
        self.pending_synchronization_wait = Some(self.core.current_thread_id);
        // None defers BOTH result registers until cancellation wakes this wait.
        Ok(None)
    }

    /// Complete parked waits whose objects are now signaled (first signaled
    /// handle in array order) or whose deadline has passed (TimedOut, -1).
    /// Returns whether any wait completed.
    fn wake_ready_waiters(&mut self) -> Result<bool, Trap> {
        // Servers with shared-memory request queues run before waits are
        // re-evaluated, so a waiter sees their completion events.
        self.services.refresh_syncpoint_cache(&mut self.memory)?;
        let now = std::time::Instant::now();
        self.services.advance_audio(now, &mut self.memory);
        let mut woke = self.time_out_mutex_waits(now)?;
        for (id, record) in self.synchronization_waits.waiters() {
            let mut completion = None;
            for (index, &handle) in record.handles.iter().enumerate() {
                if self.synchronization_object_signaled(handle)? == Some(true) {
                    completion = Some((0, index as i32));
                    break;
                }
            }
            if completion.is_none() && record.deadline.is_some_and(|deadline| now >= deadline) {
                // A finished SleepThread returns no result (W0 is written 0).
                completion = Some(if record.sleep { (0, -1) } else { (0xea01, -1) });
                if !record.sleep && self.debug_messages.len() < 256 {
                    let objects: Vec<String> = record
                        .handles
                        .iter()
                        .map(|&handle| match self.services.readable_event_kind(handle) {
                            Some(kind) => format!("{handle:#x} {kind}"),
                            None => format!("{handle:#x}"),
                        })
                        .collect();
                    self.debug_messages.push(format!(
                        "wait timed out: thread {id}, objects [{}]",
                        objects.join(", ")
                    ));
                }
            }
            let Some((result, index)) = completion else {
                continue;
            };
            if self.thread_state(id) != Some(ThreadState::Waiting) {
                return Err(Trap::InvalidState {
                    message: "a parked synchronization wait belongs to a non-waiting thread".into(),
                });
            }
            let completed = self.synchronization_waits.complete(id, result, index)?;
            if !completed.writes_index {
                self.complete_w0_wait(id, completed.result)?;
            } else if id == self.layout.main_thread_id {
                let cpu = self
                    .main_saved_cpu
                    .as_mut()
                    .ok_or_else(|| Trap::InvalidState {
                        message: "main wait has no saved continuation".into(),
                    })?;
                cpu.regs[0] = u64::from(completed.result);
                cpu.regs[1] = u64::from(completed.index as u32);
                self.main_thread_state = ThreadState::Runnable;
            } else {
                self.threads.complete_synchronization_wait(
                    id,
                    completed.result,
                    completed.index,
                )?;
            }
            self.synchronization_waits_returned =
                self.synchronization_waits_returned.saturating_add(1);
            woke = true;
        }
        Ok(woke)
    }

    /// Signaled condition waiters still waiting for their mutex whose
    /// deadline has passed: TimedOut, without the mutex (see
    /// `MutexWaits::prepare_timeout`). Returns whether any timed out.
    fn time_out_mutex_waits(&mut self, now: std::time::Instant) -> Result<bool, Trap> {
        let expired = self.mutex_waits.expired(now);
        for &id in &expired {
            if self.thread_state(id) != Some(ThreadState::Waiting) {
                return Err(Trap::InvalidState {
                    message: "a timed-out mutex waiter is not parked".into(),
                });
            }
            let priorities = self.with_mutex_waits(|waits, lookup| waits.time_out(id, lookup))?;
            self.validate_thread_priorities(&priorities)?;
            self.apply_thread_priorities(&priorities)?;
            if id == self.layout.main_thread_id {
                let cpu = self
                    .main_saved_cpu
                    .as_mut()
                    .ok_or_else(|| Trap::InvalidState {
                        message: "waiting main context disappeared at its mutex timeout".into(),
                    })?;
                cpu.regs[0] = 0xea01;
                self.main_thread_state = ThreadState::Runnable;
            } else {
                self.threads.complete_wait(id, 0xea01)?;
            }
        }
        Ok(!expired.is_empty())
    }

    /// No thread can run: wake ready waits, sleeping on the host until the
    /// earliest wait deadline when nothing is ready yet. Dispatches the next
    /// runnable thread and returns true, or false when nothing can ever wake
    /// (no deadline and no signal source outside the guest).
    fn idle_until_runnable(&mut self) -> Result<bool, Trap> {
        // This worker runs nothing now: its last thread, if runnable, is
        // anyone's (it may have moved to another worker's core).
        self.keeps_running = None;
        loop {
            self.wake_ready_waiters()?;
            let current = self.core.current_thread_id;
            if let Some(next) = self.next_runnable_thread(current, None) {
                // The active context was parked (its CPU saved) when no thread
                // could run, so restore the woken continuation even if it is
                // the same thread.
                let cpu = self.prepare_dispatch_cpu(next)?;
                self.commit_context_switch(next, cpu, "idle_wake");
                return Ok(true);
            }
            let deadline = [
                self.synchronization_waits.earliest_deadline(),
                self.mutex_waits.earliest_deadline(),
                self.services.next_audio_deadline(),
            ]
            .into_iter()
            .flatten()
            .min();
            let Some(deadline) = deadline else {
                return Ok(false);
            };
            if self.parallel {
                // The worker parks with the machine unlocked instead.
                return Ok(false);
            }
            let now = std::time::Instant::now();
            if idle_wall_limit_passed() {
                return Err(Trap::Budget {
                    pc: self.core.cpu.pc,
                    steps: self.core.steps,
                });
            }
            if deadline > now {
                std::thread::sleep(deadline - now);
            }
        }
    }

    fn cancel_synchronization_result(&mut self, handle: u32) -> Result<u32, Trap> {
        if self.thread_state(self.core.current_thread_id) != Some(ThreadState::Runnable)
            || self.pending_mutex_wait.is_some()
            || self.pending_synchronization_wait.is_some()
            || self.pending_thread_exit.is_some()
        {
            return Err(Trap::InvalidState {
                message: "CancelSynchronization requires an active runnable caller at a completed prior SVC boundary".into(),
            });
        }
        let id = if handle == 0xffff8000 {
            Some(self.core.current_thread_id)
        } else {
            self.real_thread_id_for_handle(handle)
        };
        let Some(id) = id else {
            return Ok(0xe401);
        };
        let state = self.thread_state(id).ok_or_else(|| Trap::InvalidState {
            message: "typed cancellation handle has no thread backing".into(),
        })?;
        if state == ThreadState::Waiting && self.synchronization_waits.is_waiting(id) {
            if id == self.layout.main_thread_id && self.main_saved_cpu.is_none() {
                return Err(Trap::InvalidState {
                    message: "cancellable main wait has no saved continuation".into(),
                });
            }
            let completed = self.synchronization_waits.cancel(id)?;
            if id == self.layout.main_thread_id {
                let cpu = self
                    .main_saved_cpu
                    .as_mut()
                    .ok_or_else(|| Trap::InvalidState {
                        message:
                            "main continuation disappeared during synchronization cancellation"
                                .into(),
                    })?;
                cpu.regs[0] = u64::from(completed.result);
                cpu.regs[1] = u64::from(completed.index as u32);
                self.main_thread_state = ThreadState::Runnable;
            } else {
                self.threads.complete_synchronization_wait(
                    id,
                    completed.result,
                    completed.index,
                )?;
            }
            self.set_thread_wait_cancelled(id, false)?;
            self.synchronization_waits_cancelled =
                self.synchronization_waits_cancelled.saturating_add(1);
        } else {
            // WaitCancel latches on Runnable, Initialized, Terminated and
            // non-cancellable mutex waits. It must not wake a mutex waiter.
            self.set_thread_wait_cancelled(id, true)?;
            self.synchronization_latched_cancel_requests = self
                .synchronization_latched_cancel_requests
                .saturating_add(1);
        }
        self.synchronization_cancel_requests =
            self.synchronization_cancel_requests.saturating_add(1);
        Ok(0)
    }

    /// SetThreadPriority (Atmosphere 6b0c077 kern_svc_thread.cpp): range,
    /// then process grant (InvalidPriority), then handle (InvalidHandle).
    /// SetBasePriority then restores effective priorities up the mutex
    /// owner chain; the serialized scheduler sees them at its next pick.
    fn set_thread_priority(&mut self, handle: u32, priority: i32) -> Result<u32, Trap> {
        let Ok(priority) = u8::try_from(priority) else {
            return Ok(0xe001);
        };
        if priority >= 64 {
            return Ok(0xe001);
        }
        let priority_mask = self
            .layout
            .process_priority_mask
            .filter(|mask| *mask != 0)
            .ok_or_else(|| Trap::InvalidState {
                message:
                    "SetThreadPriority requires validated nonzero process priority permissions"
                        .into(),
            })?;
        if priority_mask & (1u64 << priority) == 0 {
            return Ok(0xe001);
        }
        let Some(id) = self.thread_id_for_handle(handle) else {
            return Ok(0xe401);
        };
        let priorities =
            self.with_mutex_waits(|waits, lookup| waits.set_base_priority(id, priority, lookup))?;
        if id == self.layout.main_thread_id {
            self.main_thread_base_priority = priority;
        } else {
            self.threads.set_base_priority(id, priority)?;
        }
        self.validate_thread_priorities(&priorities)?;
        self.apply_thread_priorities(&priorities)?;
        Ok(0)
    }

    /// SetGpuMemoryAttribute: only the GpuSharable bit (0x20) may be set or
    /// cleared, on page-aligned normal or code/data memory, with the
    /// SetMemoryAttribute result codes. Switchbrew documents only the
    /// signature; CreateGpuMemory requires its range to carry 0x20. The
    /// ranges are recorded here; QueryMemory does not report the bit yet.
    fn set_gpu_memory_attribute(
        &mut self,
        address: u64,
        size: u64,
        mask: u32,
        attr: u32,
    ) -> Result<u32, Trap> {
        const GPU_SHARABLE: u32 = 0x20;
        if !address.is_multiple_of(PAGE_SIZE) {
            return Ok(0xcc01);
        }
        if size == 0 || !size.is_multiple_of(PAGE_SIZE) {
            return Ok(0xca01);
        }
        if mask != GPU_SHARABLE || attr & !mask != 0 {
            return Ok(0xe801);
        }
        let Some(end) = address.checked_add(size) else {
            return Ok(0xcc01);
        };
        let mut cursor = address;
        while cursor < end {
            let mapping = self.memory.query(cursor);
            if !matches!(mapping.state, STATE_CODE_DATA | STATE_NORMAL) {
                return Ok(0xd401);
            }
            cursor = mapping.base.saturating_add(mapping.size).min(end);
        }
        // Remove the range from every recorded run, then add it back if set.
        let overlapping: Vec<(u64, u64)> = self
            .services
            .gpu_sharable
            .range(..end)
            .filter(|&(_, &run_end)| run_end > address)
            .map(|(&start, &run_end)| (start, run_end))
            .collect();
        for (start, run_end) in overlapping {
            self.services.gpu_sharable.remove(&start);
            if start < address {
                self.services.gpu_sharable.insert(start, address);
            }
            if run_end > end {
                self.services.gpu_sharable.insert(end, run_end);
            }
        }
        if attr == GPU_SHARABLE {
            self.services.gpu_sharable.insert(address, end);
        }
        Ok(0)
    }

    /// One thread's base and effective priority (none for a thread that is
    /// gone): what mutex planning looks up, thread by thread.
    fn thread_priority(&self, id: u64) -> Option<ThreadPriority> {
        if id == self.layout.main_thread_id {
            return self.main_thread_alive().then_some(ThreadPriority {
                base: self.main_thread_base_priority,
                effective: self.main_thread_effective_priority,
            });
        }
        Some(ThreadPriority {
            base: self.threads.base_priority(id)?,
            effective: self.threads.priority(id)?,
        })
    }

    /// Run `change` on the mutex queue with each thread's priority looked
    /// up as needed (the queue is moved out for the call, then back).
    fn with_mutex_waits<T>(
        &mut self,
        change: impl FnOnce(
            &mut crate::synchronization::MutexWaits,
            &dyn Fn(u64) -> Option<ThreadPriority>,
        ) -> Result<T, Trap>,
    ) -> Result<T, Trap> {
        let mut waits = std::mem::take(&mut self.mutex_waits);
        let result = change(&mut waits, &|id| self.thread_priority(id));
        self.mutex_waits = waits;
        result
    }

    fn validate_thread_priorities(&self, priorities: &BTreeMap<u64, u8>) -> Result<(), Trap> {
        for (&id, &priority) in priorities {
            let base = if id == self.layout.main_thread_id {
                self.main_thread_alive()
                    .then_some(self.main_thread_base_priority)
            } else {
                self.threads.base_priority(id)
            };
            if !base.is_some_and(|base| base < 64 && priority <= base) {
                return Err(Trap::InvalidState {
                    message: "mutex donation targets an absent thread or invalid priority".into(),
                });
            }
        }
        Ok(())
    }

    /// The caller validates every update before retaining objects or publishing
    /// a prepared queue. Nothing can change these records within this host phase.
    fn apply_thread_priorities(&mut self, priorities: &BTreeMap<u64, u8>) -> Result<(), Trap> {
        for (&id, &priority) in priorities {
            if id == self.layout.main_thread_id {
                self.main_thread_effective_priority = priority;
            } else {
                self.threads.set_effective_priority(id, priority)?;
            }
        }
        Ok(())
    }

    fn retain_mutex_owner(&mut self, id: u64) -> Result<(), Trap> {
        if id == self.layout.main_thread_id {
            if !self.main_thread_alive() {
                return Err(Trap::InvalidState {
                    message: "mutex owner main thread no longer exists".into(),
                });
            }
            self.main_mutex_owner_references = self
                .main_mutex_owner_references
                .checked_add(1)
                .ok_or_else(|| Trap::InvalidState {
                    message: "main mutex-owner reference count exhausted".into(),
                })?;
            Ok(())
        } else {
            self.threads.retain_mutex_owner_reference(id)
        }
    }

    fn release_mutex_owner(&mut self, id: u64) -> Result<(), Trap> {
        if id == self.layout.main_thread_id {
            if self.main_mutex_owner_references == 0 {
                return Err(Trap::InvalidState {
                    message: "main mutex-owner reference count is already zero".into(),
                });
            }
            if self.main_mutex_owner_references == 1
                && !self.main_thread_handle_open
                && !self.main_thread_state.holds_running_reference()
            {
                // A cleanup failure preserves the reference and the pending
                // return that owns it, just like the child collection.
                self.release_main_thread_tls()?;
            }
            self.main_mutex_owner_references -= 1;
            Ok(())
        } else {
            self.threads
                .release_mutex_owner_reference(&mut self.memory, id)
        }
    }

    fn mutex_address_result(address: u64) -> u32 {
        if (KERNEL_VIRTUAL_ADDRESS_BASE..KERNEL_VIRTUAL_ADDRESS_END).contains(&address) {
            0xd401
        } else if !address.is_multiple_of(4) {
            0xcc01
        } else {
            0
        }
    }

    /// None means the caller has begun a deferred SVC. Its complete post-SVC
    /// continuation is captured only after the generated next-PC epilogue.
    fn arbitrate_lock_result(
        &mut self,
        owner_handle: u32,
        address: u64,
        tag: u32,
    ) -> Result<Option<u32>, Trap> {
        let address_result = Self::mutex_address_result(address);
        if address_result != 0 {
            return Ok(Some(address_result));
        }
        if self.process_exit_requested
            || self.pending_thread_exit == Some(self.core.current_thread_id)
        {
            return Ok(Some(0x7601));
        }
        if self.thread_state(self.core.current_thread_id) != Some(ThreadState::Runnable)
            || self.pending_mutex_wait.is_some()
            || self.pending_synchronization_wait.is_some()
        {
            return Err(Trap::InvalidState {
                message: "ArbitrateLock requires an active runnable context without a pending wait"
                    .into(),
            });
        }
        let word = match self.memory.read_u32(address) {
            Ok(word) => word,
            Err(Trap::MemoryFault { .. }) => return Ok(Some(0xd401)),
            Err(trap) => return Err(trap),
        };
        if word != (owner_handle | MUTEX_WAIT_MASK) {
            self.mutex_raced_lock_returns = self.mutex_raced_lock_returns.saturating_add(1);
            return Ok(Some(0));
        }
        let Some(owner) = self.real_thread_id_for_handle(owner_handle) else {
            return Ok(Some(0xe401));
        };
        let current = self.core.current_thread_id;
        let priorities = self
            .with_mutex_waits(|waits, lookup| waits.wait(current, owner, address, tag, lookup))?;
        self.validate_thread_priorities(&priorities)?;
        self.retain_mutex_owner(owner)?;
        self.apply_thread_priorities(&priorities)?;
        self.pending_mutex_wait = Some(self.core.current_thread_id);
        // Lock never writes or CASes the userspace word and does not write a
        // provisional success Result into the parked continuation.
        Ok(None)
    }

    /// Wake a parked waiter whose wait returns only W0.
    fn complete_w0_wait(&mut self, id: u64, result: u32) -> Result<(), Trap> {
        if id == self.layout.main_thread_id {
            let cpu = self
                .main_saved_cpu
                .as_mut()
                .ok_or_else(|| Trap::InvalidState {
                    message: "waiting main context has no saved continuation".into(),
                })?;
            cpu.regs[0] = u64::from(result);
            self.main_thread_state = ThreadState::Runnable;
            Ok(())
        } else {
            self.threads.complete_wait(id, result)
        }
    }

    /// WaitProcessWideKeyAtomic (Atmosphere 6b0c077 KConditionVariable::Wait):
    /// hand the mutex at `address` to its next waiter (or clear it), set the
    /// condition key to 1, then time out immediately for a zero timeout or
    /// park on `key` until signaled or the deadline passes.
    fn wait_condition_result(
        &mut self,
        address: u64,
        key: u64,
        tag: u32,
        timeout_ns: i64,
    ) -> Result<Option<u32>, Trap> {
        let address_result = Self::mutex_address_result(address);
        if address_result != 0 {
            return Ok(Some(address_result));
        }
        if (KERNEL_VIRTUAL_ADDRESS_BASE..KERNEL_VIRTUAL_ADDRESS_END).contains(&key) {
            return Ok(Some(0xd401));
        }
        let current = self.core.current_thread_id;
        if self.process_exit_requested || self.pending_thread_exit == Some(current) {
            return Ok(Some(0x7601));
        }
        if self.thread_state(current) != Some(ThreadState::Runnable)
            || self.pending_mutex_wait.is_some()
            || self.pending_synchronization_wait.is_some()
        {
            return Err(Trap::InvalidState {
                message:
                    "condition wait requires an active runnable context without a pending wait"
                        .into(),
            });
        }
        let plan = self.with_mutex_waits(|waits, lookup| waits.unlock(current, address, lookup))?;
        self.validate_thread_priorities(&plan.priorities)?;
        if let Some(id) = plan.selected {
            if self.thread_state(id) != Some(ThreadState::Waiting) {
                return Err(Trap::InvalidState {
                    message: "mutex queue selected a thread without a waiting continuation".into(),
                });
            }
        }
        self.apply_thread_priorities(&plan.priorities)?;
        if let Some(id) = plan.selected {
            self.mutex_waits.set_wait_result(id, 0)?;
            self.complete_w0_wait(id, 0)?;
            self.mutex_waits_woken = self.mutex_waits_woken.saturating_add(1);
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        for (target, value) in [(key, 1), (address, plan.next_word)] {
            match self.memory.check_write(target, 4) {
                Ok(()) => self.guest_store_u32(target, value)?,
                Err(Trap::MemoryFault { .. }) => return Ok(Some(0xd401)),
                Err(trap) => return Err(trap),
            }
        }
        if timeout_ns == 0 {
            return Ok(Some(0xea01));
        }
        let deadline = (timeout_ns > 0).then(|| {
            std::time::Instant::now() + std::time::Duration::from_nanos(timeout_ns as u64)
        });
        self.synchronization_waits
            .begin_condition(current, timeout_ns, deadline, key, address, tag)?;
        self.pending_synchronization_wait = Some(current);
        self.condition_waits_started = self.condition_waits_started.saturating_add(1);
        Ok(None)
    }

    /// SignalProcessWideKey (Atmosphere 6b0c077 KConditionVariable::Signal):
    /// wake up to `count` waiters on `key` in priority order. Each reacquires
    /// its mutex if free, or becomes a waiter of the current owner. Once no
    /// waiter remains on `key`, the key word is cleared.
    fn signal_condition(&mut self, key: u64, count: i32) -> Result<(), Trap> {
        // Only this key's waiters are looked up and sorted.
        let mut waiters = self.synchronization_waits.condition_waiters(key);
        waiters.sort_by_cached_key(|(id, wait)| {
            (
                self.thread_priority(*id)
                    .map_or(u8::MAX, |priority| priority.effective),
                wait.order,
            )
        });
        let limit = if count <= 0 {
            waiters.len()
        } else {
            count as usize
        };
        for (id, wait) in waiters.into_iter().take(limit) {
            if self.thread_state(id) != Some(ThreadState::Waiting) {
                return Err(Trap::InvalidState {
                    message: "condition waiter is not parked".into(),
                });
            }
            let word = match self.update_lock_atomic(wait.mutex, wait.tag) {
                Ok(word) => Some(word),
                Err(Trap::MemoryFault { .. }) => None,
                Err(trap) => return Err(trap),
            };
            match word {
                None => {
                    self.synchronization_waits.complete(id, 0xd401, 0)?;
                    self.complete_w0_wait(id, 0xd401)?;
                }
                Some(0) => {
                    self.synchronization_waits.complete(id, 0, 0)?;
                    self.complete_w0_wait(id, 0)?;
                }
                Some(word) => {
                    let owner_handle = word & !MUTEX_WAIT_MASK;
                    let Some(owner) = self.real_thread_id_for_handle(owner_handle) else {
                        self.synchronization_waits.complete(id, 0xfa01, 0)?;
                        self.complete_w0_wait(id, 0xfa01)?;
                        continue;
                    };
                    let priorities = self.with_mutex_waits(|waits, lookup| {
                        waits.wait(id, owner, wait.mutex, wait.tag, lookup)
                    })?;
                    self.validate_thread_priorities(&priorities)?;
                    self.retain_mutex_owner(owner)?;
                    self.apply_thread_priorities(&priorities)?;
                    // Its timeout still runs while it waits for the mutex.
                    let deadline = self.synchronization_waits.transfer_condition(id)?;
                    self.mutex_waits.set_deadline(id, deadline)?;
                }
            }
            self.condition_signals = self.condition_signals.saturating_add(1);
        }
        if self.synchronization_waits.condition_waiters(key).is_empty() {
            if let Ok(()) = self.memory.check_write(key, 4) {
                self.memory.write_u32(key, 0)?;
            }
        }
        Ok(())
    }

    /// Whether the active thread has user preemption disabled (TLS
    /// DisableCount at +0x100); if so its InterruptFlag (+0x102) is set.
    fn user_preemption_disabled(&mut self) -> bool {
        if crate::nvn::debug_var_os("ECHORA_IGNORE_DISABLE_COUNT").is_some() {
            return false;
        }
        let tls = self.core.cpu.tpidrro_el0;
        if tls == 0
            || !matches!(self.memory.read_u16(tls + TLS_DISABLE_COUNT), Ok(count) if count != 0)
        {
            return false;
        }
        let _ = self.memory.write_u16(tls + TLS_INTERRUPT_FLAG, 1);
        self.user_preemption_deferrals = self.user_preemption_deferrals.saturating_add(1);
        true
    }

    /// Compare-and-swap on a guest word, returning the value seen. The
    /// kernel updates user memory with LDXR/STXR loops because other cores
    /// run user code meanwhile; with several workers so do these, so a
    /// read-then-write here can overwrite a guest lock or unlock.
    fn guest_cas_u32(&mut self, address: u64, expected: u32, desired: u32) -> Result<u32, Trap> {
        self.memory.check_write(address, 4)?;
        let seen = self.core.memory.compare_exchange_single_thread(
            address,
            4,
            u64::from(expected),
            u64::from(desired),
            true,
            true,
        )?;
        Ok(seen as u32)
    }

    /// Store a guest word atomically: with several workers, guest code on
    /// other cores may compare-and-swap it at the same time.
    fn guest_store_u32(&mut self, address: u64, value: u32) -> Result<(), Trap> {
        self.memory.check_write(address, 4)?;
        self.core.memory.atomic_rmw_single_thread(
            address,
            4,
            echora_guest::memory::AtomicOp::Swap,
            u64::from(value),
            false,
            true,
        )?;
        Ok(())
    }

    /// Atmosphere UpdateLockAtomic: take the mutex word for `tag` if it is
    /// free, else mark the current owner as having waiters. The previous
    /// word is returned (0: `tag` now owns the mutex).
    fn update_lock_atomic(&mut self, address: u64, tag: u32) -> Result<u32, Trap> {
        let mut word = self.memory.read_u32(address)?;
        loop {
            let next = if word == 0 {
                tag
            } else {
                word | MUTEX_WAIT_MASK
            };
            let seen = self.guest_cas_u32(address, word, next)?;
            if seen == word {
                return Ok(word);
            }
            word = seen;
        }
    }

    /// Result for an arbitrated address that cannot be used: 0xcc01
    /// (InvalidAddress) when unaligned, 0xd401 (InvalidCurrentMemory) in the
    /// kernel range.
    fn arbiter_address_result(address: u64) -> u32 {
        if address & 3 != 0 {
            0xcc01
        } else if (KERNEL_VIRTUAL_ADDRESS_BASE..KERNEL_VIRTUAL_ADDRESS_END).contains(&address) {
            0xd401
        } else {
            0
        }
    }

    /// WaitForAddress (Atmosphere KAddressArbiter::WaitIfLessThan /
    /// WaitIfEqual / WaitIfEqual64): type 0 waits while the s32 at `address`
    /// is less than `value`, type 1 also decrements it first, type 2 waits
    /// while it equals `value`, and type 3 waits while the 8-byte-aligned u64
    /// at `address` equals the full 64-bit `raw_value` (X2). A failed
    /// comparison returns InvalidState (0xfa01) without waiting; a zero
    /// timeout returns TimedOut (0xea01) after the update.
    fn wait_for_address_result(
        &mut self,
        address: u64,
        kind: u32,
        raw_value: u64,
        timeout_ns: i64,
    ) -> Result<Option<u32>, Trap> {
        let value = raw_value as u32 as i32;
        let address_result = Self::arbiter_address_result(address);
        if address_result != 0 {
            return Ok(Some(address_result));
        }
        if kind > 3 {
            return Ok(Some(0xf001));
        }
        if kind == 3 && address & 7 != 0 {
            return Ok(Some(0xcc01));
        }
        let current = self.core.current_thread_id;
        if self.process_exit_requested || self.pending_thread_exit == Some(current) {
            return Ok(Some(0x7601));
        }
        if self.thread_state(current) != Some(ThreadState::Runnable)
            || self.pending_mutex_wait.is_some()
            || self.pending_synchronization_wait.is_some()
        {
            return Err(Trap::InvalidState {
                message: "address wait requires an active runnable context without a pending wait"
                    .into(),
            });
        }
        let read = if kind == 3 {
            self.memory.read_u64(address)
        } else {
            self.memory.read_u32(address).map(u64::from)
        };
        let current_raw = match read {
            Ok(word) => word,
            Err(Trap::MemoryFault { .. }) => return Ok(Some(0xd401)),
            Err(trap) => return Err(trap),
        };
        let current_value = current_raw as u32 as i32;
        let waits = match kind {
            0 | 1 => current_value < value,
            2 => current_value == value,
            _ => current_raw == raw_value,
        };
        if !waits {
            self.wait_for_address_immediate_returns += 1;
            return Ok(Some(0xfa01));
        }
        if kind == 1 {
            // DecrementIfLessThan, atomically against other cores.
            let mut seen = current_raw as u32;
            loop {
                let decremented = (seen as i32).wrapping_sub(1) as u32;
                match self.guest_cas_u32(address, seen, decremented) {
                    Ok(now) if now == seen => break,
                    Ok(now) if (now as i32) < value => seen = now,
                    Ok(_) => {
                        self.wait_for_address_immediate_returns += 1;
                        return Ok(Some(0xfa01));
                    }
                    Err(Trap::MemoryFault { .. }) => return Ok(Some(0xd401)),
                    Err(trap) => return Err(trap),
                }
            }
        }
        if timeout_ns == 0 {
            self.wait_for_address_immediate_returns += 1;
            return Ok(Some(0xea01));
        }
        let deadline = (timeout_ns > 0).then(|| {
            std::time::Instant::now() + std::time::Duration::from_nanos(timeout_ns as u64)
        });
        self.synchronization_waits
            .begin_arbiter(current, timeout_ns, deadline, address)?;
        self.pending_synchronization_wait = Some(current);
        Ok(None)
    }

    /// SignalToAddress (Atmosphere KAddressArbiter::Signal /
    /// SignalAndIncrementIfEqual / SignalAndModifyByWaitingCountIfEqual):
    /// wake up to `count` waiters on `address` in priority, then FIFO order.
    /// Type 1 first requires the s32 at `address` to equal `value` and
    /// increments it; type 2 requires equality and stores `value + 1` with no
    /// waiters, `value - 2` when waking all of them, `value - 1` when `count`
    /// covers every waiter and `value` otherwise. A failed comparison returns
    /// InvalidState (0xfa01) and wakes nothing.
    fn signal_to_address_result(
        &mut self,
        address: u64,
        kind: u32,
        value: i32,
        count: i32,
    ) -> Result<u32, Trap> {
        let address_result = Self::arbiter_address_result(address);
        if address_result != 0 {
            return Ok(address_result);
        }
        if kind > 2 {
            return Ok(0xf001);
        }
        // Only this address's waiters are looked up and sorted.
        let mut waiters = self.synchronization_waits.arbiter_waiters(address);
        // A signal (or increment) nobody waits for changes no thread: the
        // case a lock-free path could take.
        if waiters.is_empty() && kind < 2 {
            self.signal_to_address_without_waiters += 1;
        }
        waiters.sort_by_cached_key(|(id, wait)| {
            (
                self.thread_priority(*id)
                    .map_or(u8::MAX, |priority| priority.effective),
                wait.order,
            )
        });
        if kind != 0 {
            let current_value = match self.memory.read_u32(address) {
                Ok(word) => word as i32,
                Err(Trap::MemoryFault { .. }) => return Ok(0xd401),
                Err(trap) => return Err(trap),
            };
            if current_value != value {
                return Ok(0xfa01);
            }
            let new_value = if kind == 1 || waiters.is_empty() {
                value.wrapping_add(1)
            } else if count <= 0 {
                value.wrapping_sub(2)
            } else if waiters.len() <= count as usize {
                value.wrapping_sub(1)
            } else {
                value
            };
            if new_value != value {
                // UpdateIfEqual, atomically against other cores.
                match self.guest_cas_u32(address, value as u32, new_value as u32) {
                    Ok(seen) if seen == value as u32 => {}
                    Ok(_) => return Ok(0xfa01),
                    Err(Trap::MemoryFault { .. }) => return Ok(0xd401),
                    Err(trap) => return Err(trap),
                }
            }
        }
        let limit = if count <= 0 {
            waiters.len()
        } else {
            count as usize
        };
        for (id, _) in waiters.into_iter().take(limit) {
            if self.thread_state(id) != Some(ThreadState::Waiting) {
                return Err(Trap::InvalidState {
                    message: "address waiter is not parked".into(),
                });
            }
            self.synchronization_waits.complete(id, 0, 0)?;
            self.complete_w0_wait(id, 0)?;
        }
        Ok(0)
    }

    fn arbitrate_unlock_result(&mut self, address: u64) -> Result<u32, Trap> {
        let address_result = Self::mutex_address_result(address);
        if address_result != 0 {
            return Ok(address_result);
        }
        if self.thread_state(self.core.current_thread_id) != Some(ThreadState::Runnable)
            || self.pending_mutex_wait.is_some()
            || self.pending_synchronization_wait.is_some()
            || self.pending_thread_exit.is_some()
        {
            return Err(Trap::InvalidState {
                message: "ArbitrateUnlock requires the active runnable context".into(),
            });
        }
        // A real userspace write fault must still transfer the queue and wake
        // the selected waiter. A prototype-only self-modifying-code trap is
        // rejected before mutation. This preflight does not read the word.
        let writable = match self.memory.check_write(address, 4) {
            Ok(()) => true,
            Err(Trap::MemoryFault { .. }) => false,
            Err(trap) => return Err(trap),
        };
        let current = self.core.current_thread_id;
        let plan = self.with_mutex_waits(|waits, lookup| waits.unlock(current, address, lookup))?;
        self.validate_thread_priorities(&plan.priorities)?;
        if let Some(id) = plan.selected {
            if self.thread_state(id) != Some(ThreadState::Waiting)
                || (id == self.layout.main_thread_id && self.main_saved_cpu.is_none())
            {
                return Err(Trap::InvalidState {
                    message: "mutex queue selected a thread without a waiting continuation".into(),
                });
            }
        }
        self.apply_thread_priorities(&plan.priorities)?;
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        let result = if writable {
            self.memory.write_u32(address, plan.next_word)?;
            0
        } else {
            0xd401
        };
        if let Some(id) = plan.selected {
            self.mutex_waits.set_wait_result(id, result)?;
            if id == self.layout.main_thread_id {
                let cpu = self
                    .main_saved_cpu
                    .as_mut()
                    .ok_or_else(|| Trap::InvalidState {
                        message: "waiting main context disappeared during unlock".into(),
                    })?;
                cpu.regs[0] = u64::from(result);
                self.main_thread_state = ThreadState::Runnable;
            } else {
                self.threads.complete_wait(id, result)?;
            }
            self.mutex_waits_woken = self.mutex_waits_woken.saturating_add(1);
        }
        self.mutex_unlock_calls = self.mutex_unlock_calls.saturating_add(1);
        Ok(result)
    }

    fn thread_processor_id(&self, id: u64) -> Result<u8, Trap> {
        let processor = if id == self.layout.main_thread_id {
            Some(self.layout.processor_id)
        } else {
            self.threads.processor_id(id)
        };
        processor
            .filter(|core| *core < 64)
            .ok_or_else(|| Trap::InvalidState {
                message: "runnable thread has no valid logical processor".into(),
            })
    }

    fn start_thread_result(&mut self, handle: u32) -> Result<u32, Trap> {
        let Some(id) = self.thread_id_for_handle(handle) else {
            return Ok(0xe401);
        };
        let state = self.thread_state(id).ok_or_else(|| Trap::InvalidState {
            message: "typed thread handle has no lifecycle backing".into(),
        })?;
        if state == ThreadState::Terminated
            || self.process_exit_requested
            || self.pending_thread_exit == Some(id)
            || self.pending_thread_exit == Some(self.core.current_thread_id)
        {
            return Ok(0x7601); // Target/caller TerminationRequested precedes state.
        }
        if state != ThreadState::Initialized {
            return Ok(0xfa01); // InvalidState, including main/current Runnable.
        }
        self.threads.start(id)?;
        Ok(0)
    }

    /// The scheduling diagnostics: ECHORA_ANY_CORE, ECHORA_FLAT_PRIORITY,
    /// ECHORA_ONE_CORE and ECHORA_PARALLEL_THREADS (see
    /// `next_runnable_thread`). Read once: they apply at every scheduling
    /// decision.
    fn scheduler_flags() -> &'static (bool, bool, bool, Option<Vec<u64>>) {
        static FLAGS: std::sync::LazyLock<(bool, bool, bool, Option<Vec<u64>>)> =
            std::sync::LazyLock::new(|| {
                let set = |name| std::env::var_os(name).is_some();
                let split = std::env::var("ECHORA_PARALLEL_THREADS").ok().map(|list| {
                    list.split(',')
                        .filter_map(|item| item.trim().parse().ok())
                        .collect()
                });
                (
                    set("ECHORA_ANY_CORE"),
                    set("ECHORA_FLAT_PRIORITY"),
                    set("ECHORA_ONE_CORE"),
                    split,
                )
            });
        &FLAGS
    }

    /// The guest cores this worker schedules one by one: its cores, unless
    /// there is no worker or a diagnostic merges them into one group.
    fn scheduled_guest_cores(&self) -> u64 {
        let (any_core, _, one_core, split) = Self::scheduler_flags();
        match self.worker_core {
            Some(core) if !any_core && !one_core && split.is_none() => {
                self.worker_guest_cores(core)
            }
            _ => 0,
        }
    }

    /// Each runnable thread this worker may run now (not `exclude`, not
    /// running elsewhere), with its home core here: `visit(id, home)` until
    /// it returns false. The worker, its cores and the holds are worked out
    /// once per call, not per thread.
    fn for_each_schedulable(&self, exclude: Option<u64>, visit: &mut dyn FnMut(u64, u8) -> bool) {
        // With several workers each runs a fixed set of the application
        // cores (see `worker_guest_cores`), as on Horizon: only threads
        // whose affinity mask names one of them. A thread pinned to one
        // core then always runs on one worker and never migrates, which the
        // game's per-core data relies on. A mask naming no application core
        // is allowed anywhere.
        // ECHORA_ANY_CORE (diagnostic): schedule as one worker does,
        // ignoring affinity. ECHORA_ONE_CORE: every thread on worker 0.
        // ECHORA_PARALLEL_THREADS=ID,... (diagnostic): only those threads
        // run on worker 1, all others on worker 0.
        let (any_core, _, one_core, split) = Self::scheduler_flags();
        let worker_cores = self.worker_core.map(|core| self.worker_guest_cores(core));
        let mask_of = |id: u64| -> u64 {
            if id == self.layout.main_thread_id {
                Some(self.main_thread_affinity.1)
            } else {
                self.threads.affinity(id).map(|(_, mask)| mask)
            }
            .unwrap_or(u64::MAX)
        };
        let allowed = |id: u64| -> bool {
            if *one_core {
                return self.worker_core.is_none_or(|core| core == 0);
            }
            if let (Some(list), Some(worker)) = (split, self.worker_core) {
                return list.contains(&id) == (worker == 1);
            }
            let Some(cores) = worker_cores.filter(|_| !any_core) else {
                return true;
            };
            let mask = mask_of(id);
            mask & APPLICATION_CORES == 0 || mask & cores != 0
        };
        // A worker serving several guest cores runs each core's best
        // priority in turn (see `next_runnable_thread`); the diagnostics
        // above keep one group.
        let cores = self.scheduled_guest_cores();
        let folded = cores.count_ones() > 1;
        // A thread holding a core (dispatch disabled) runs only there, and
        // nothing else runs on a core held by a runnable holder.
        let any_hold = self.core_hold.iter().any(Option::is_some);
        let held_at = |id: u64| -> Option<u8> {
            if !any_hold {
                return None;
            }
            (0..64u8).find(|&core| self.core_hold[usize::from(core)] == Some(id))
        };
        // As `guest_core_on_worker`, with this worker's cores worked out.
        let home = |id: u64, held: Option<u8>| -> u8 {
            if !folded {
                return 0;
            }
            held.unwrap_or_else(|| {
                let mask = mask_of(id);
                let usable = if mask & cores != 0 {
                    mask & cores
                } else if mask & APPLICATION_CORES != 0 {
                    mask & APPLICATION_CORES
                } else {
                    cores
                };
                usable.trailing_zeros() as u8
            })
        };
        let holding = |id: u64| -> bool {
            self.thread_state(id) == Some(ThreadState::Runnable) && !self.running.contains(&id)
        };
        let main =
            (self.main_thread_state == ThreadState::Runnable).then_some(self.layout.main_thread_id);
        for id in main.into_iter().chain(self.threads.runnable_ids()) {
            if exclude == Some(id) || self.running.contains(&id) {
                continue;
            }
            let held = held_at(id);
            // A core's holder runs there even if its mask has moved on
            // since: it migrates once it ends a slice with dispatch enabled.
            let held_here = held.is_some_and(|held| cores & 1 << held != 0);
            if !(allowed(id) || held_here) {
                continue;
            }
            let core = home(id, held);
            let hold_allows = !any_hold
                || if held.is_some() {
                    held_here
                } else {
                    !self.core_hold[usize::from(core) % 64].is_some_and(|holder| holding(holder))
                };
            if hold_allows && !visit(id, core) {
                return;
            }
        }
    }

    /// The runnable thread after `after` in id order (wrapping). Threads
    /// backing off from a spin-wait are passed over while any other
    /// thread is runnable.
    fn next_runnable_thread(&self, after: u64, exclude: Option<u64>) -> Option<u64> {
        // The best (lowest) priority first, per guest core.
        // ECHORA_FLAT_PRIORITY (diagnostic): schedule as one worker does,
        // ignoring priority.
        let (_, flat_priority, _, _) = Self::scheduler_flags();
        let priority = |id: u64| -> u8 {
            if self.worker_core.is_none() || *flat_priority {
                return 0;
            }
            if id == self.layout.main_thread_id {
                self.main_thread_effective_priority
            } else {
                self.threads.priority(id).unwrap_or(u8::MAX)
            }
        };
        // A worker serving several guest cores runs each core's best
        // priority in turn, as Horizon's cores would side by side: with
        // strict priority across them a busy thread on one core would starve
        // a worse-priority thread on another forever. Slice ends rotate to
        // the next core with work after `after`'s. With one core per worker
        // (W=6) every thread shares the worker's core and this is strict
        // priority as before.
        // (id, home core, priority, backing off from a spin)
        let mut candidates: Vec<(u64, u8, u8, bool)> = Vec::new();
        self.for_each_schedulable(exclude, &mut |id, core| {
            let spinning = self
                .spin_backoff
                .get(&id)
                .is_some_and(|until| *until > self.core.steps);
            candidates.push((id, core, priority(id), spinning));
            true
        });
        if candidates.is_empty() {
            return None;
        }
        let cores = self.scheduled_guest_cores();
        let home = |id: u64| -> u8 {
            match self.worker_core {
                Some(core) if cores.count_ones() > 1 => self
                    .core_hold
                    .iter()
                    .position(|hold| *hold == Some(id))
                    .map_or_else(|| self.guest_core_on_worker(id, core), |held| held as u8),
                _ => 0,
            }
        };
        let mut best = [u8::MAX; 64];
        for &(_, core, priority, _) in &candidates {
            let slot = &mut best[usize::from(core) % 64];
            *slot = (*slot).min(priority);
        }
        // Cores in turn after `after`'s, ending with its own.
        let start = home(after);
        let turn = |core: u8| (u32::from(core) + 63 - u32::from(start)) % 64;
        let pick = |skip_spinning: bool| {
            let mut first: Option<u32> = None;
            let mut next: Option<(u32, u64)> = None;
            let mut wrapped: Option<(u32, u64)> = None;
            for &(id, core, priority, spinning) in &candidates {
                if priority != best[usize::from(core) % 64] || skip_spinning && spinning {
                    continue;
                }
                let turn = turn(core);
                // Equal priorities on a core take turns: after `after` on
                // its own core (the last turn), after the thread a core
                // last ran on the others.
                let key = (turn, id);
                let previous = if turn == 63 {
                    after
                } else {
                    self.core_last_thread[usize::from(core) % 64]
                };
                // The first core in turn decides; within it, the next id
                // after `previous`, else (wrapping) the lowest.
                let slot = if id > previous {
                    &mut next
                } else {
                    &mut wrapped
                };
                *slot = Some(slot.map_or(key, |old| old.min(key)));
                first = Some(first.map_or(turn, |old: u32| old.min(turn)));
            }
            let first = first?;
            [next, wrapped]
                .into_iter()
                .flatten()
                .find(|&(turn, _)| turn == first)
                .map(|(_, id)| id)
        };
        pick(true).or_else(|| pick(false))
    }

    /// Horizon preemption across workers: a runnable thread no worker runs
    /// takes a core whose running thread has a worse priority, at that
    /// core's next block boundary. The game relies on it (a better-priority
    /// thread it wakes runs before the waker's core goes on).
    fn request_preemptions(&mut self) {
        self.note_core_thread();
        self.keeps_running = Some(self.core.current_thread_id);
        let priority = |machine: &Machine, id: u64| -> u8 {
            if id == machine.layout.main_thread_id {
                machine.main_thread_effective_priority
            } else {
                machine.threads.priority(id).unwrap_or(u8::MAX)
            }
        };
        let mask = |machine: &Machine, id: u64| -> u64 {
            if id == machine.layout.main_thread_id {
                machine.main_thread_affinity.1
            } else {
                machine
                    .threads
                    .affinity(id)
                    .map_or(u64::MAX, |(_, mask)| mask)
            }
        };
        let main =
            (self.main_thread_state == ThreadState::Runnable).then_some(self.layout.main_thread_id);
        // The best priority waiting for each application core, in one pass
        // over the threads no worker runs (a mask naming no application core
        // waits for every one of them).
        let mut best = [u8::MAX; 64];
        for id in main.into_iter().chain(self.threads.runnable_ids()) {
            if self.running.contains(&id) || id == self.core.current_thread_id {
                continue;
            }
            let (waiting, mask) = (priority(self, id), mask(self, id));
            let cores = if mask & APPLICATION_CORES == 0 {
                APPLICATION_CORES
            } else {
                mask & APPLICATION_CORES
            };
            let mut bits = cores;
            while bits != 0 {
                let core = bits.trailing_zeros() as usize;
                best[core] = best[core].min(waiting);
                bits &= bits - 1;
            }
        }
        for core in 0..self.core_threads.len() {
            // This worker's own core was just scheduled.
            if self.worker_core == Some(core as u8) {
                continue;
            }
            let Some(thread) = self.core_threads[core] else {
                continue;
            };
            let running = priority(self, thread);
            // A better priority waits for one of the worker's cores.
            let mut bits = self.worker_guest_cores(core as u8);
            let mut wanted = false;
            while bits != 0 {
                let guest = bits.trailing_zeros() as usize;
                wanted |= best[guest] < running;
                bits &= bits - 1;
            }
            if wanted {
                if let Some(flag) = self.preempt_flags.get(core) {
                    if !flag.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        self.preemptions += 1;
                    }
                }
            }
        }
    }

    /// Record the thread this worker runs as its guest core's last, for the
    /// turns of equal priorities on folded cores.
    fn note_core_thread(&mut self) {
        let Some(worker) = self.worker_core else {
            return;
        };
        let current = self.core.current_thread_id;
        let cores = self.worker_guest_cores(worker);
        if cores.count_ones() > 1 {
            // Not a core of another worker's (a mask moved while dispatch
            // is disabled leaves the thread here, on the core it ran on).
            let core = self.guest_core_on_worker(current, worker);
            if cores & 1 << core != 0 {
                self.core_last_thread[usize::from(core) % 64] = current;
            }
        }
    }

    /// At a slice end with dispatch disabled: with several guest cores on
    /// this worker, hold the current thread's core and return the thread
    /// of the worker's next core in turn, if another core has one. With
    /// one core per worker (or one worker) there is no other turn: None.
    fn dispatch_disabled_turn(&mut self) -> Option<u64> {
        let worker = self.worker_core?;
        if self.scheduled_guest_cores().count_ones() < 2 {
            return None;
        }
        let current = self.core.current_thread_id;
        // The core it runs on: one it already holds, else its home here,
        // else (its mask has moved to another worker's cores since it was
        // dispatched) the core of this worker it last ran on.
        let cores = self.worker_guest_cores(worker);
        let home = usize::from(self.guest_core_on_worker(current, worker)) % 64;
        let Some(core) = (0..64)
            .find(|&core| self.core_hold[core] == Some(current))
            .or_else(|| (cores & 1 << home != 0).then_some(home))
            .or_else(|| {
                (0..64)
                    .find(|&core| cores & 1 << core != 0 && self.core_last_thread[core] == current)
            })
        else {
            return None;
        };
        self.core_hold[core] = Some(current);
        let next = self.next_runnable_thread(current, None);
        if next.is_none_or(|next| next == current) {
            self.core_hold[core] = None;
            return None;
        }
        next
    }

    /// The current thread ended a slice with dispatch enabled: it holds no
    /// core any more.
    fn release_core_hold(&mut self) {
        let current = self.core.current_thread_id;
        for hold in &mut self.core_hold {
            if *hold == Some(current) {
                *hold = None;
            }
        }
    }

    /// The application cores worker `worker` runs: core `c` belongs to
    /// worker `c % workers`, so each core has exactly one worker.
    fn worker_guest_cores(&self, worker: u8) -> u64 {
        let workers = self.core_threads.len().max(1) as u64;
        // ECHORA_CORE_MAP=W0,W1,... (diagnostic): core c belongs to the
        // c-th listed worker instead.
        let map: Option<Vec<u64>> = crate::nvn::debug_var_os("ECHORA_CORE_MAP").map(|map| {
            map.to_string_lossy()
                .split(',')
                .filter_map(|item| item.trim().parse().ok())
                .collect()
        });
        (0..u64::from(APPLICATION_CORES.count_ones()))
            .filter(|&core| match &map {
                Some(map) => map.get(core as usize).copied().unwrap_or(0) == u64::from(worker),
                None => core % workers == u64::from(worker),
            })
            .fold(0, |mask, core| mask | 1 << core)
    }

    /// The core thread `id` runs on while on worker `worker`: the first of
    /// that worker's cores its affinity mask names, else the first core of
    /// its mask (a diagnostic placement), else the worker's first core. The
    /// game indexes per-core data by this number, so it is always a core
    /// the thread may run on.
    fn guest_core_on_worker(&self, id: u64, worker: u8) -> u8 {
        let mask = if id == self.layout.main_thread_id {
            self.main_thread_affinity.1
        } else {
            self.threads.affinity(id).map_or(u64::MAX, |(_, mask)| mask)
        };
        let cores = self.worker_guest_cores(worker);
        let usable = if mask & cores != 0 {
            mask & cores
        } else if mask & APPLICATION_CORES != 0 {
            mask & APPLICATION_CORES
        } else {
            cores
        };
        usable.trailing_zeros() as u8
    }

    /// The parked workers (bit per worker) whose cores allow a thread no
    /// worker runs: one parked worker finding nothing for its own core is
    /// not a deadlock, and these are the workers worth waking. A thread only
    /// a busy worker may run is not counted: waking parked workers for it
    /// would have them wake each other without end, starving the busy
    /// workers of the machine lock.
    fn parked_workers_with_work(&mut self) -> u64 {
        let core = self.worker_core;
        let current = self.core.current_thread_id;
        let mut workers = 0;
        for worker in 0..self.core_threads.len() {
            if self.core_threads[worker].is_some() {
                continue;
            }
            self.worker_core = Some(worker as u8);
            let exclude = self.keeps_running.filter(|&id| id == current);
            // Whether next_runnable_thread would find one: any will do.
            let mut found = false;
            self.for_each_schedulable(exclude, &mut |_, _| {
                found = true;
                false
            });
            if found {
                workers |= 1 << worker;
            }
        }
        self.worker_core = core;
        workers
    }

    fn saved_thread_cpu(&self, id: u64) -> Result<Cpu, Trap> {
        self.thread_processor_id(id)?;
        if id == self.layout.main_thread_id {
            if self.main_thread_state != ThreadState::Runnable {
                return Err(Trap::InvalidState {
                    message: "cannot dispatch a non-runnable main thread".into(),
                });
            }
            self.main_saved_cpu
                .clone()
                .ok_or_else(|| Trap::InvalidState {
                    message: "main thread has not saved a dispatch context".into(),
                })
        } else {
            self.threads.saved_cpu(id)
        }
    }

    fn note_context_saved(&mut self, id: u64) {
        if crate::nvn::debug_var_os("ECHORA_CHECK_CONTEXTS").is_some() {
            self.context_counter += 1;
            self.context_sequence.entry(id).or_default().0 = self.context_counter;
        }
    }

    fn note_context_dispatched(&mut self, id: u64) {
        if crate::nvn::debug_var_os("ECHORA_CHECK_CONTEXTS").is_none() {
            return;
        }
        self.context_counter += 1;
        let counter = self.context_counter;
        let entry = self.context_sequence.entry(id).or_default();
        if entry.1 > entry.0 {
            eprintln!(
                "stale dispatch: thread {id} saved at {} but last dispatched at {} (worker {:?}, from thread {}, pc {:#x}, steps {})",
                entry.0,
                entry.1,
                self.worker_core,
                self.core.current_thread_id,
                self.core.cpu.pc,
                self.core.steps
            );
        }
        entry.1 = counter;
    }

    fn prepare_dispatch_cpu(&mut self, id: u64) -> Result<Cpu, Trap> {
        let cpu = self.saved_thread_cpu(id)?;
        self.note_context_dispatched(id);
        let synchronization_return = self.synchronization_waits.pending_return(id);
        if let Some(completed) = synchronization_return {
            if self.mutex_waits.pending_return(id).is_some()
                || cpu.regs[0] != u64::from(completed.result)
                || (completed.writes_index && cpu.regs[1] != u64::from(completed.index as u32))
            {
                return Err(Trap::InvalidState {
                    message: "synchronization return does not match its unique saved continuation"
                        .into(),
                });
            }
        }
        if let Some(pending) = self.mutex_waits.pending_return(id) {
            let result = pending.result.ok_or_else(|| Trap::InvalidState {
                message: "mutex waiter cannot resume before its userspace store result".into(),
            })?;
            if cpu.regs[0] != u64::from(result) {
                return Err(Trap::InvalidState {
                    message: "resumed mutex Result differs from its saved continuation".into(),
                });
            }
            // WaitForAddress retains its ORIGINAL owner until the blocked SVC
            // returns, even if unlock transferred the remaining queue elsewhere.
            self.release_mutex_owner(pending.original_owner)?;
            self.mutex_waits.remove_pending_return(id)?;
            self.mutex_waits_returned = self.mutex_waits_returned.saturating_add(1);
        }
        if synchronization_return.is_some() {
            self.synchronization_waits.remove_pending_return(id)?;
            self.synchronization_waits_returned =
                self.synchronization_waits_returned.saturating_add(1);
        }
        Ok(cpu)
    }

    fn commit_context_switch(&mut self, next: u64, cpu: Cpu, reason: &'static str) {
        self.last_context_switch = Some(ContextSwitch {
            from_thread_id: self.core.current_thread_id,
            to_thread_id: next,
            instruction_count: self.core.steps,
            source_pc: self.core.cpu.pc,
            destination_pc: cpu.pc,
            reason,
        });
        self.core.cpu = cpu;
        self.core.current_thread_id = next;
        // A context switch is an exception return on the switching PE, which
        // clears the local exclusive monitor (Arm DDI0487 B2.9). Without this,
        // another guest thread's STXR could consume this thread's reservation.
        self.core.memory.clear_exclusive();
        self.context_switches = self.context_switches.saturating_add(1);
    }

    /// Called only after a completed generated basic block, including the SVC
    /// next-PC epilogue. One-block, cyclic-ID round robin is a declared prototype
    /// policy, not Horizon priority/preemption/multicore scheduling. Ordinary
    /// traps and mid-block budget stops never enter this method.
    pub fn finish_block(&mut self) -> Result<(), Trap> {
        if self.process_exit_requested {
            return Err(Trap::Exit { code: 0 });
        }
        self.core.completed_blocks = self.core.completed_blocks.saturating_add(1);
        if let Some(waiting) = self.pending_synchronization_wait {
            if waiting != self.core.current_thread_id
                || self.pending_thread_exit.is_some()
                || self.pending_mutex_wait.is_some()
                || self.thread_state(waiting) != Some(ThreadState::Runnable)
                || !self.synchronization_waits.is_parked(waiting)
            {
                return Err(Trap::InvalidState {
                    message: "pending synchronization wait does not belong to the active runnable context".into(),
                });
            }
            let incoming = self
                .next_runnable_thread(waiting, Some(waiting))
                .map(|id| self.prepare_dispatch_cpu(id).map(|cpu| (id, cpu)))
                .transpose()?;
            if waiting == self.layout.main_thread_id {
                self.main_saved_cpu = Some(self.core.cpu.clone());
                self.note_context_saved(self.layout.main_thread_id);
                self.main_thread_state = ThreadState::Waiting;
            } else {
                self.threads.block_wait(waiting, &self.core.cpu)?;
                self.note_context_saved(waiting);
            }
            self.pending_synchronization_wait = None;
            self.synchronization_waits_started =
                self.synchronization_waits_started.saturating_add(1);
            if let Some((id, cpu)) = incoming {
                self.commit_context_switch(id, cpu, "zero_handle_synchronization_wait");
                return Ok(());
            }
            return Err(Trap::NoRunnableThreads {
                waiting_threads: self.waiting_thread_count(),
            });
        }
        if let Some(waiting) = self.pending_mutex_wait {
            if waiting != self.core.current_thread_id
                || self.pending_thread_exit.is_some()
                || self.thread_state(waiting) != Some(ThreadState::Runnable)
            {
                return Err(Trap::InvalidState {
                    message: "pending mutex wait does not belong to the active runnable context"
                        .into(),
                });
            }
            let incoming = self
                .next_runnable_thread(waiting, Some(waiting))
                .map(|id| self.prepare_dispatch_cpu(id).map(|cpu| (id, cpu)))
                .transpose()?;
            if waiting == self.layout.main_thread_id {
                self.main_saved_cpu = Some(self.core.cpu.clone());
                self.note_context_saved(self.layout.main_thread_id);
                self.main_thread_state = ThreadState::Waiting;
            } else {
                self.threads.block_wait(waiting, &self.core.cpu)?;
                self.note_context_saved(waiting);
            }
            self.pending_mutex_wait = None;
            self.mutex_waits_started = self.mutex_waits_started.saturating_add(1);
            if let Some((id, cpu)) = incoming {
                self.commit_context_switch(id, cpu, "mutex_wait");
                return Ok(());
            }
            // There is no host event loop, timeout or fabricated acquisition.
            // Keep all parked contexts and references alive at this stop.
            return Err(Trap::NoRunnableThreads {
                waiting_threads: self.waiting_thread_count(),
            });
        }
        if let Some(exiting) = self.pending_thread_exit {
            if exiting != self.core.current_thread_id {
                return Err(Trap::InvalidState {
                    message: "pending thread exit does not belong to the active context".into(),
                });
            }
            let next = self.next_runnable_thread(exiting, Some(exiting));
            let released_on_dispatch = u32::from(
                next.and_then(|id| self.mutex_waits.pending_return(id))
                    .is_some_and(|pending| pending.original_owner == exiting),
            );
            // Validate outgoing cleanup BEFORE consuming an incoming blocked
            // SVC's owner reference. The incoming return may own the exiting
            // thread's last reference; no dispatch occurs if this check fails.
            if exiting == self.layout.main_thread_id {
                if self.main_thread_state != ThreadState::Runnable {
                    return Err(Trap::InvalidState {
                        message: "main thread is not runnable at its exit boundary".into(),
                    });
                }
                let remaining = self
                    .main_mutex_owner_references
                    .checked_sub(released_on_dispatch)
                    .ok_or_else(|| Trap::InvalidState {
                        message: "incoming mutex return has no exiting main-owner reference".into(),
                    })?;
                if !self.main_thread_handle_open && remaining == 0 {
                    self.validate_main_thread_tls_release()?;
                }
            } else {
                self.threads
                    .preflight_terminate(&self.memory, exiting, released_on_dispatch)?;
            }
            let incoming = next
                .map(|id| self.prepare_dispatch_cpu(id).map(|cpu| (id, cpu)))
                .transpose()?;
            // The outgoing block is complete: no guest instruction can touch
            // this context/TLS while the host finishes termination below.
            if exiting == self.layout.main_thread_id {
                if self.main_thread_state != ThreadState::Runnable {
                    return Err(Trap::InvalidState {
                        message: "main thread is not runnable at its exit boundary".into(),
                    });
                }
                if !self.main_thread_handle_open && self.main_mutex_owner_references == 0 {
                    self.release_main_thread_tls()?;
                }
                self.main_saved_cpu = Some(self.core.cpu.clone());
                self.note_context_saved(self.layout.main_thread_id);
                self.main_thread_state = ThreadState::Terminated;
            } else {
                self.threads
                    .terminate(&mut self.memory, exiting, &self.core.cpu)?;
            }
            self.pending_thread_exit = None;
            self.thread_exits = self.thread_exits.saturating_add(1);
            if let Some((id, cpu)) = incoming {
                self.commit_context_switch(id, cpu, "thread_exit");
                return Ok(());
            }
            if self.started_thread_count() != 0 {
                return Err(Trap::NoRunnableThreads {
                    waiting_threads: self.waiting_thread_count(),
                });
            }
            self.process_exit_requested = true;
            self.process_exit_reason = Some("last_started_thread_exit");
            return Err(Trap::Exit { code: 0 });
        }
        // Horizon never preempts or migrates a thread whose thread-local
        // DisableCount is set; it raises the InterruptFlag instead and the
        // thread calls SynchronizePreemptionState once it re-enables. The
        // game's per-core data (indexed by GetCurrentProcessorNumber) relies
        // on it.
        // With several guest cores folded onto this worker the thread holds
        // only its own core: the others still take their turns, and the
        // thread resumes on its core with nothing run there in between.
        if self.user_preemption_disabled() {
            let Some(next) = self.dispatch_disabled_turn() else {
                return Ok(());
            };
            return self.switch_at_slice_end(next, "dispatch_disabled_turn");
        }
        self.release_core_hold();
        // A thread that ended its slice spin-polling waits for memory other
        // threads must change: let them run first for a while. Only dense
        // polling counts (a poll per SPIN_DENSITY_INSTRUCTIONS or more):
        // this game compiles ordinary atomic loads as compare-and-swaps, so
        // working threads also reach the poll limit, just more slowly.
        let slice_length = self.core.steps.saturating_sub(self.core.slice_start);
        if self.core.spin_polls >= echora_guest::SPIN_POLL_LIMIT
            && slice_length
                <= u64::from(self.core.spin_polls).saturating_mul(SPIN_DENSITY_INSTRUCTIONS)
        {
            self.spin_backoff.insert(
                self.core.current_thread_id,
                self.core.steps.saturating_add(SPIN_BACKOFF_INSTRUCTIONS),
            );
        }
        let Some(next) = self.next_runnable_thread(self.core.current_thread_id, None) else {
            // With several workers the running thread itself can be
            // ineligible here (its affinity mask moved it to another core's
            // worker). It stays runnable, so another worker resumes it from
            // its saved context: save the live one first, or that worker
            // runs it from a stale snapshot.
            if self.thread_state(self.core.current_thread_id) == Some(ThreadState::Runnable) {
                if self.core.current_thread_id == self.layout.main_thread_id {
                    self.main_saved_cpu = Some(self.core.cpu.clone());
                    self.note_context_saved(self.layout.main_thread_id);
                } else {
                    self.threads
                        .save_cpu(self.core.current_thread_id, &self.core.cpu)?;
                    self.note_context_saved(self.core.current_thread_id);
                }
            }
            return Err(Trap::NoRunnableThreads {
                waiting_threads: self.waiting_thread_count(),
            });
        };
        if next == self.core.current_thread_id {
            return Ok(()); // Never rewind the active CPU to its saved snapshot.
        }
        // ECHORA_NO_TIMESLICE (experiment): as on Horizon, a thread keeps
        // its core until it blocks, yields or a better priority preempts it;
        // only priority 59 rotates.
        if self.parallel && crate::nvn::debug_var_os("ECHORA_NO_TIMESLICE").is_some() {
            let current = self.core.current_thread_id;
            let priority = |id: u64| {
                if id == self.layout.main_thread_id {
                    self.main_thread_effective_priority
                } else {
                    self.threads.priority(id).unwrap_or(u8::MAX)
                }
            };
            if self.thread_state(current) == Some(ThreadState::Runnable)
                && priority(current) != 59
                && priority(next) >= priority(current)
            {
                return Ok(());
            }
        }
        self.switch_at_slice_end(next, "completed_block_round_robin")
    }

    /// Save the current thread's context and dispatch `next`.
    fn switch_at_slice_end(&mut self, next: u64, reason: &'static str) -> Result<(), Trap> {
        let cpu = self.prepare_dispatch_cpu(next)?;
        if self.core.current_thread_id == self.layout.main_thread_id {
            self.main_saved_cpu = Some(self.core.cpu.clone());
            self.note_context_saved(self.layout.main_thread_id);
        } else {
            self.threads
                .save_cpu(self.core.current_thread_id, &self.core.cpu)?;
            self.note_context_saved(self.core.current_thread_id);
        }
        self.commit_context_switch(next, cpu, reason);
        Ok(())
    }

    /// One instruction outside generated code: the host readiness checks,
    /// then the same accounting generated code performs through [`Core`].
    pub fn tick(&mut self) -> Result<(), Trap> {
        self.host_ready()?;
        self.core.tick()?;
        self.flush_core_counts()
    }

    /// Thread and process state that must hold before the next instruction.
    /// It only changes inside SVCs and block-boundary scheduling, so generated
    /// code relies on these checks having run at the last host boundary.
    fn host_ready(&mut self) -> Result<(), Trap> {
        // Parked waits are rescanned at most every WAKE_CHECK_INSTRUCTIONS
        // while the current thread can run (time-sliced runs only): a woken
        // thread gets the processor at a slice end anyway, and the scan over
        // every wait was a measurable share of run time.
        let current_runnable =
            self.thread_state(self.core.current_thread_id) == Some(ThreadState::Runnable);
        if self.layout.scheduler_slice == 0
            || !current_runnable
            || self.core.steps.wrapping_sub(self.last_wake_check) >= WAKE_CHECK_INSTRUCTIONS
        {
            self.last_wake_check = self.core.steps;
            self.wake_ready_waiters()?;
        }
        // A stop request (window closed, or ECHORA_STOP_FILE created) ends
        // the run like a process exit, so the report is written. The file
        // is checked about once a second.
        self.stop_check_tick = self.stop_check_tick.wrapping_add(1);
        if self.stop_check_tick.is_multiple_of(1 << 16)
            && (crate::window::STOP_REQUESTED.load(std::sync::atomic::Ordering::Relaxed)
                || std::env::var_os("ECHORA_STOP_FILE")
                    .is_some_and(|path| std::path::Path::new(&path).exists()))
        {
            self.process_exit_requested = true;
            self.process_exit_reason = Some("host_stop_request");
        }
        if self.process_exit_requested {
            return Err(Trap::Exit { code: 0 });
        }
        if self.pending_thread_exit.is_some() {
            return Err(Trap::InvalidState {
                message: "thread exit must finish at its block boundary before another instruction"
                    .into(),
            });
        }
        if self.pending_mutex_wait.is_some() {
            return Err(Trap::InvalidState {
                message: "mutex wait must finish at its block boundary before another instruction"
                    .into(),
            });
        }
        if self.pending_synchronization_wait.is_some() {
            return Err(Trap::InvalidState {
                message: "synchronization wait must finish at its block boundary before another instruction".into(),
            });
        }
        if self.thread_state(self.core.current_thread_id) != Some(ThreadState::Runnable) {
            if self
                .next_runnable_thread(self.core.current_thread_id, None)
                .is_none()
                && self.waiting_thread_count() != 0
            {
                return Err(Trap::NoRunnableThreads {
                    waiting_threads: self.waiting_thread_count(),
                });
            }
            return Err(Trap::InvalidState {
                message: "the active context is not runnable".into(),
            });
        }
        if self.core.current_thread_id == self.layout.main_thread_id
            && self.main_thread_state != ThreadState::Runnable
        {
            return Err(Trap::InvalidState {
                message: "terminated main thread cannot execute".into(),
            });
        }
        // Every boundary needs the host while any exit/wait is pending.
        self.core.attention = self.process_exit_requested
            || self.pending_thread_exit.is_some()
            || self.pending_mutex_wait.is_some()
            || self.pending_synchronization_wait.is_some();
        // Timed waits are checked at block boundaries at least this often,
        // and another runnable thread takes the processor after a slice.
        let mut check_at = u64::MAX;
        if self.synchronization_waits.has_deadline()
            || self.mutex_waits.earliest_deadline().is_some()
            || self.services.next_audio_deadline().is_some()
        {
            check_at = self
                .core
                .steps
                .saturating_add(HOST_TIMER_CHECK_INSTRUCTIONS);
        }
        // Any other runnable thread, backing off from a spin or not, gets the
        // processor back at slice ends.
        let current = self.core.current_thread_id;
        let others = (self.main_thread_state == ThreadState::Runnable
            && self.layout.main_thread_id != current
            && !self.running.contains(&self.layout.main_thread_id))
            || self
                .threads
                .runnable_ids()
                .any(|id| id != current && !self.running.contains(&id));
        // Other workers' stops and wakeups are seen at boundaries.
        if others || self.parallel {
            let slice = if self.parallel {
                self.layout.parallel_slice
            } else {
                self.layout.scheduler_slice
            };
            check_at = check_at.min(self.core.steps.saturating_add(slice.max(1)));
        }
        // `attention` forces the next boundary too: one compare per block.
        self.core.host_check_at = if self.core.attention { 0 } else { check_at };
        self.core.spin_polls = 0;
        self.core.slice_start = self.core.steps;
        // Every 16th boundary: the profile keeps its shape at a fraction of
        // the hashing cost. ECHORA_PC_SAMPLES_FROM=N samples from present N
        // on (one phase of a long run, such as after a movie).
        self.pc_sample_tick = self.pc_sample_tick.wrapping_add(1);
        if self.pc_sample_tick.is_multiple_of(16)
            && crate::nvn::debug_var("ECHORA_PC_SAMPLES_FROM")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .is_none_or(|from| self.nvn.presented_frames >= from)
        {
            *self
                .pc_samples
                .entry((self.core.current_thread_id, self.core.cpu.pc))
                .or_default() += 16;
            if let Some(range) = crate::nvn::debug_var_os("ECHORA_PC_STACKS") {
                self.sample_pc_stack(range);
            }
        }
        Ok(())
    }

    /// Run a boundary step; when it finds no runnable thread but a wait can
    /// still complete (signal already pending or a deadline), idle on the
    /// host until a thread is runnable and continue there.
    fn resume_or_idle(
        &mut self,
        step: impl FnOnce(&mut Self) -> Result<(), Trap>,
    ) -> Result<(), Trap> {
        match step(self) {
            Err(Trap::NoRunnableThreads { waiting_threads }) => {
                if self.idle_until_runnable()? {
                    self.host_ready()
                } else {
                    Err(Trap::NoRunnableThreads { waiting_threads })
                }
            }
            other => other,
        }
    }

    /// Credit instructions counted by [`Core::tick`] to the active thread.
    fn flush_core_counts(&mut self) -> Result<(), Trap> {
        let count = self.core.steps - self.core.thread_steps_base;
        self.core.thread_steps_base = self.core.steps;
        if count == 0 {
            return Ok(());
        }
        if self.core.current_thread_id == self.layout.main_thread_id {
            self.main_instructions_started = self
                .main_instructions_started
                .checked_add(count)
                .ok_or_else(|| Trap::InvalidState {
                    message: "main instruction counter exhausted".into(),
                })?;
            Ok(())
        } else {
            self.threads
                .note_instructions(self.core.current_thread_id, count)
        }
    }

    /// Host loop for generated code. `dispatch` runs translated blocks on
    /// [`Core`] until they leave their window or need the host: an SVC is
    /// performed here, followed by the PC update and block completion that
    /// generated code used to do inline. With several workers
    /// (`Layout::workers`), each runs guest threads on its own host thread.
    pub fn run_generated(
        &mut self,
        dispatch: impl Fn(&mut Core) -> Result<(), Trap> + Sync,
    ) -> Result<(), Trap> {
        self.resume_or_idle(|machine| machine.host_ready())?;
        self.check_entry_pc()?;
        let _profile = crate::host_profile::start();
        // ECHORA_PARALLEL_PATH (diagnostic): the worker machinery even for
        // one worker.
        if self.layout.workers > 1 || crate::nvn::debug_var_os("ECHORA_PARALLEL_PATH").is_some() {
            return self.run_parallel(&dispatch);
        }
        let _watch = crate::fault::watch(&self.core);
        loop {
            let result = dispatch(&mut self.core);
            if result.is_ok() {
                continue;
            }
            self.boundary(result)?;
        }
    }

    /// Host work after generated code returned `result` (not `Ok`): the
    /// SVC, host call or scheduling it asks for, then the readiness checks
    /// for the next instruction. Generated code branches only to translated
    /// code, so host-chosen PCs are where execute permission is checked.
    fn boundary(&mut self, result: Result<(), Trap>) -> Result<(), Trap> {
        self.core.settle_block_trap();
        // A requested preemption is this boundary's scheduling decision.
        self.core
            .preempt
            .store(false, std::sync::atomic::Ordering::Relaxed);
        self.boundary_kinds[boundary_kind(&result)] += 1;
        match result {
            Ok(()) => return Ok(()),
            Err(Trap::SupervisorCall { number }) => {
                self.flush_core_counts()?;
                let next = self.core.cpu.pc.wrapping_add(4);
                self.svc(number)?;
                self.core.cpu.pc = next;
            }
            Err(Trap::HostBoundary) => self.flush_core_counts()?,
            Err(Trap::MissingBlock { pc }) if self.nvn.has_entry(pc) => {
                // A branch to a host function: run it with the AArch64
                // calling convention, then return to the link register.
                self.flush_core_counts()?;
                let entry = self.nvn.entry(pc).expect("has_entry(pc) was checked above");
                self.call_nvn(entry)?;
                return self.finish_host_call();
            }
            Err(other) => {
                self.flush_core_counts()?;
                return Err(other);
            }
        }
        self.resume_or_idle(|machine| {
            machine.finish_block()?;
            machine.host_ready()
        })?;
        self.check_entry_pc()
    }

    /// After a host function: return to the link register and continue (or
    /// switch threads) as after any boundary.
    fn finish_host_call(&mut self) -> Result<(), Trap> {
        self.core.cpu.pc = self
            .host_call_continue
            .take()
            .unwrap_or(self.core.cpu.regs[30]);
        self.resume_or_idle(|machine| {
            machine.finish_block()?;
            machine.host_ready()
        })?;
        self.check_entry_pc()
    }

    /// Execute permission at a PC the host is about to enter. Host function
    /// thunks (NVN) have no guest memory: dispatch turns them into host
    /// calls.
    fn check_entry_pc(&self) -> Result<(), Trap> {
        let pc = self.core.cpu.pc;
        if self.nvn.has_entry(pc) {
            return Ok(());
        }
        self.memory.check_execute(pc, 4)
    }

    /// Several workers: each runs its guest thread's generated code without
    /// the machine lock and takes it for host work, swapping its core into
    /// `core` so the host code sees the thread it serves. A worker without a
    /// runnable thread parks until one is.
    fn run_parallel(
        &mut self,
        dispatch: &(impl Fn(&mut Core) -> Result<(), Trap> + Sync),
    ) -> Result<(), Trap> {
        // Start ECHORA_STOP_SECONDS's clock with the workers.
        stop_seconds_passed();
        // One worker per application core.
        let workers = self
            .layout
            .workers
            .min(APPLICATION_CORES.count_ones() as usize);
        let global = self.core.steps;
        let idle = |machine: &Machine| {
            let mut core = Core::new(machine.memory.view());
            core.steps = global;
            core.thread_steps_base = global;
            core.max_steps = machine.core.max_steps;
            core.physical_counter = machine.core.physical_counter.clone();
            core
        };
        let placeholder = idle(self);
        let mut cores = vec![std::mem::replace(&mut self.core, placeholder)];
        cores.extend((1..workers).map(|_| idle(self)));
        for core in &mut cores {
            core.parallel = true;
        }
        self.preempt_flags = cores.iter().map(|core| core.preempt.clone()).collect();
        self.core_threads = vec![None; workers];
        self.core_threads[0] = Some(cores[0].current_thread_id);
        self.running.insert(cores[0].current_thread_id);
        self.parallel = true;
        let placeholder = self.nvn.entries_only();
        let recorder = self.nvn.recorder();
        let thunks = std::sync::RwLock::new(self.nvn.thunk_names());
        let nvn = std::sync::Mutex::new(std::mem::replace(&mut self.nvn, placeholder));
        let shared = Shared {
            machine: std::sync::Mutex::new(&mut *self),
            wake: (0..workers).map(|_| std::sync::Condvar::new()).collect(),
            nvn,
            recorder,
            thunks,
            stop: std::sync::atomic::AtomicBool::new(false),
            serial: crate::nvn::debug_var_os("ECHORA_SERIAL_GUEST")
                .map(|_| std::sync::Mutex::new(())),
            world: crate::nvn::debug_var_os("ECHORA_STOP_WORLD")
                .map(|_| std::sync::RwLock::new(())),
            exclusive: crate::nvn::debug_var_os("ECHORA_EXCLUSIVE_THREADS").map(|list| {
                let ids = list
                    .to_string_lossy()
                    .split(',')
                    .filter_map(|id| id.trim().parse().ok())
                    .collect();
                (std::sync::RwLock::new(()), ids)
            }),
        };
        let finished: Vec<(Core, Option<Trap>)> = std::thread::scope(|scope| {
            let handles: Vec<_> = cores
                .into_iter()
                .enumerate()
                .map(|(index, core)| {
                    let shared = &shared;
                    let worker = Worker {
                        core,
                        synced: global,
                        active: index == 0,
                        index: index as u8,
                        held_failed: None,
                        lock_free_calls: 0,
                        recorded_calls: 0,
                        nvn_lock_micros: [0; 2],
                    };
                    std::thread::Builder::new()
                        .name(format!("guest-worker-{index}"))
                        .stack_size(64 << 20)
                        .spawn_scoped(scope, move || worker.run(shared, dispatch))
                        .expect("spawn guest worker")
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("guest worker panicked"))
                .collect()
        });
        self.nvn = shared
            .nvn
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.parallel = false;
        self.running.clear();
        self.worker_core = None;
        self.core_threads.clear();
        self.preempt_flags.clear();
        self.parked_workers = 0;
        // The first stop is the run's result; its core is the one reports see.
        let global = self.core.steps;
        let mut stop = None;
        for (mut core, trap) in finished {
            if stop.is_none() {
                if let Some(trap) = trap {
                    core.steps = core.steps.max(global);
                    self.core = core;
                    stop = Some(trap);
                }
            }
        }
        Err(stop.unwrap_or(Trap::InvalidState {
            message: "guest workers stopped without a result".into(),
        }))
    }

    /// The earliest host-time event a parked wait can complete at.
    fn next_wait_deadline(&self) -> Option<std::time::Instant> {
        [
            self.synchronization_waits.earliest_deadline(),
            self.mutex_waits.earliest_deadline(),
            self.services.next_audio_deadline(),
        ]
        .into_iter()
        .flatten()
        .min()
    }

    /// One high-level NVN call. Arguments are in X0..X7 (and V0..V7); the
    /// result goes to X0.
    fn call_nvn(&mut self, entry: crate::nvn::HostEntry) -> Result<(), Trap> {
        use crate::nvn::HostEntry;
        let name = match &entry {
            HostEntry::BootstrapLoader => "nvnBootstrapLoader".to_string(),
            HostEntry::Proc(name) => name.clone(),
            HostEntry::ZlibUncompress => "hle:zlib_uncompress".to_string(),
            HostEntry::ZstdDecompress => "hle:zstd_decompress".to_string(),
            HostEntry::InflateFinishCall => "hle:inflate_finish".to_string(),
            HostEntry::ZstdStreamCall => "hle:zstd_stream".to_string(),
            HostEntry::ZstdFrames => "hle:zstd_frames".to_string(),
        };
        if entry == HostEntry::ZstdFrames {
            *self.nvn.calls.entry(name).or_default() += 1;
            let r = self.core.cpu.regs;
            self.core.cpu.regs[0] = crate::hle::zstd_decompress_frames(
                &mut self.memory,
                r[1],
                r[2],
                r[3],
                r[4],
                r[5],
                r[6],
                r[7],
            )?;
            return Ok(());
        }
        if entry == HostEntry::ZstdStreamCall {
            // The replaced `bl` would leave LR at the next instruction, where
            // the host call returns; when the host cannot perform it, the
            // guest decoder runs as the `bl` would have run it.
            let site = self.core.cpu.pc;
            self.core.cpu.regs[30] = site + 4;
            let regs = self.core.cpu.regs;
            let fed = crate::hle::zstd_stream_feed(
                &mut self.zstd_streams,
                &mut self.memory,
                regs[0],
                regs[1],
                regs[2] as i32,
                regs[3],
            )?;
            match fed {
                Some(result) => {
                    *self.nvn.calls.entry(name).or_default() += 1;
                    self.core.cpu.regs[0] = u64::from(result);
                }
                None => {
                    *self
                        .nvn
                        .calls
                        .entry("hle:zstd_stream (guest)".into())
                        .or_default() += 1;
                    let decoder = (site as i64
                        + (((self.memory.read_u32(site)? << 6) as i32 >> 4) as i64))
                        as u64;
                    self.host_call_continue = Some(decoder);
                }
            }
            return Ok(());
        }
        if entry == HostEntry::InflateFinishCall {
            *self.nvn.calls.entry(name).or_default() += 1;
            // The replaced `bl` would leave LR at the next instruction; the
            // dispatcher continues at LR after this host call.
            let site = self.core.cpu.pc;
            self.core.cpu.regs[30] = site + 4;
            let stream = self.core.cpu.regs[0];
            self.core.cpu.regs[0] =
                u64::from(crate::hle::inflate_finish(&mut self.memory, stream, site)? as u32);
            return Ok(());
        }
        if entry == HostEntry::ZlibUncompress {
            *self.nvn.calls.entry(name).or_default() += 1;
            let regs = &self.core.cpu.regs;
            let (destination, capacity, source, size) =
                (regs[0], regs[1] & 0xffff_ffff, regs[2], regs[3]);
            self.core.cpu.regs[0] =
                crate::hle::zlib_uncompress(&mut self.memory, destination, capacity, source, size)?;
            return Ok(());
        }
        if entry == HostEntry::ZstdDecompress {
            *self.nvn.calls.entry(name).or_default() += 1;
            let regs = &self.core.cpu.regs;
            let (destination, capacity, source, size, workspace, ddict) =
                (regs[0], regs[1], regs[2], regs[3], regs[4], regs[5]);
            self.core.cpu.regs[0] = crate::hle::zstd_decompress(
                &mut self.memory,
                destination,
                capacity,
                source,
                size,
                workspace,
                ddict,
            )?;
            return Ok(());
        }
        match self.nvn.calls.get_mut(&name) {
            Some(count) => *count += 1,
            None => {
                self.nvn.calls.insert(name.clone(), 1);
            }
        }
        let lookup = match entry {
            // nvnBootstrapLoader(const char* name) -> PFNC.
            HostEntry::BootstrapLoader => Some(self.core.cpu.regs[0]),
            // nvnDeviceGetProcAddress(const NVNdevice*, const char* name) -> PFNC.
            HostEntry::Proc(ref proc) if proc == "nvnDeviceGetProcAddress" => {
                Some(self.core.cpu.regs[1])
            }
            HostEntry::Proc(_)
            | HostEntry::ZlibUncompress
            | HostEntry::ZstdDecompress
            | HostEntry::InflateFinishCall
            | HostEntry::ZstdStreamCall
            | HostEntry::ZstdFrames => None,
        };
        if let Some(address) = lookup {
            let requested = self.read_string(address, crate::nvn::MAX_NAME)?;
            if !crate::nvn::is_entry_point_name(&requested) {
                return Err(Trap::UnsupportedService {
                    pc: self.core.cpu.pc,
                    service: "nvn".into(),
                    command_id: None,
                    message: format!("{name} lookup of non-NVN name {requested:?}"),
                });
            }
            self.core.cpu.regs[0] = self.nvn.thunk(&requested)?;
            return Ok(());
        }
        let counter = self.core.physical_counter_ticks();
        let mut guest = crate::nvn::Guest {
            cpu: &mut self.core.cpu,
            memory: crate::nvn::GuestMemory::Checked(&mut self.memory),
            counter,
        };
        let profile = crate::nvn::debug_var_os("ECHORA_NVN_PROFILE")
            .is_some()
            .then(std::time::Instant::now);
        let handled = self.nvn.call(&name, &mut guest)?;
        if let Some(started) = profile {
            *self.nvn.call_microseconds.entry(name.clone()).or_default() +=
                started.elapsed().as_micros() as u64;
        }
        if handled {
            if name == "nvnQueuePresentTexture" {
                self.publish_controller()?;
                // ECHORA_STOP_PRESENT=N: end the run (with its report) at
                // present N, so two runs measure the frames between them.
                if crate::nvn::debug_var("ECHORA_STOP_PRESENT")
                    .ok()
                    .and_then(|value| value.parse::<u64>().ok())
                    .is_some_and(|stop| self.nvn.presented_frames >= stop)
                {
                    self.process_exit_requested = true;
                    self.process_exit_reason = Some("host_stop_request");
                }
                // ECHORA_PROFILE_FROM=N: the lock, boundary, thread, IPC and
                // NVN timings and the host sampler (ECHORA_HOST_PROFILE)
                // restart at present N, so the report profiles
                // one phase of a long run (such as after the opening movie).
                if crate::nvn::debug_var("ECHORA_PROFILE_FROM")
                    .ok()
                    .and_then(|value| value.parse::<u64>().ok())
                    .is_some_and(|from| self.nvn.presented_frames == from)
                {
                    self.lock_stats = [0; 3];
                    self.lock_free_nvn_calls = 0;
                    self.recorded_nvn_calls = 0;
                    self.nvn_lock_stats = [0; 2];
                    self.nvn_thread_wait.clear();
                    self.boundary_kinds.fill(0);
                    self.boundary_held.fill(0);
                    self.thread_time.clear();
                    self.pc_samples.clear();
                    self.services.reset_profile();
                    self.nvn.calls.clear();
                    self.nvn.call_microseconds.clear();
                    self.nvn.reset_gpu_stats();
                    crate::host_profile::restart();
                }
            }
            return Ok(());
        }
        Err(Trap::UnsupportedService {
            pc: self.core.cpu.pc,
            service: "nvn".into(),
            command_id: None,
            message: format!("{name} is not implemented"),
        })
    }

    /// Once per presented frame, the window's keyboard state becomes a
    /// controller sample in the hid shared memory (see crate::hid).
    /// Publish one controller sample per present: the window's keyboard,
    /// or without a window, ECHORA_AUTO_PRESS=N holds a button for 6
    /// samples (ECHORA_AUTO_HOLD) every N (for headless runs through
    /// menus): the next one of
    /// ECHORA_AUTO_KEYS (comma-separated names such as A,UP,A), then A,
    /// from sample ECHORA_AUTO_START until ECHORA_AUTO_END; otherwise an
    /// idle, connected controller.
    fn publish_controller(&mut self) -> Result<(), Trap> {
        let Some(base) = self.hid_mapping else {
            return Ok(());
        };
        let buttons = if let Some(window) = crate::window::shared() {
            // The keyboard and any gamepads (ECHORA_GAMEPAD=0 ignores
            // them); an analog stick off centre wins over the keys.
            let keys = window.buttons.load(std::sync::atomic::Ordering::Relaxed);
            let pad = if std::env::var_os("ECHORA_GAMEPAD").is_some_and(|value| value == "0") {
                crate::input::Pad::default()
            } else {
                crate::input::gamepads().state()
            };
            let left = if pad.left != (0, 0) {
                pad.left
            } else {
                crate::hid::digital_stick(keys)
            };
            self.hid_samples += 1;
            return crate::hid::write_controller(
                &mut self.memory,
                base,
                self.hid_samples,
                keys | pad.buttons,
                left,
                pad.right,
            );
        } else if let Some(every) = std::env::var("ECHORA_AUTO_PRESS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|every| *every > 6)
        {
            let hold = std::env::var("ECHORA_AUTO_HOLD")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(6)
                .clamp(1, every - 1);
            // ECHORA_AUTO_START=S: no presses before sample S (loading
            // screens take a nearly fixed number of presents).
            let start = std::env::var("ECHORA_AUTO_START")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(0);
            // ECHORA_AUTO_END=E: no presses from sample E on (benchmarks
            // stop pressing once gameplay is reached).
            let end = std::env::var("ECHORA_AUTO_END")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(u64::MAX);
            let Some(sample) = self
                .hid_samples
                .checked_sub(start)
                .filter(|_| self.hid_samples < end)
            else {
                self.hid_samples += 1;
                return crate::hid::write_controller(
                    &mut self.memory,
                    base,
                    self.hid_samples,
                    0,
                    (0, 0),
                    (0, 0),
                );
            };
            let press = sample / every;
            let key = std::env::var("ECHORA_AUTO_KEYS")
                .ok()
                .and_then(|keys| {
                    keys.split(',')
                        .nth(press as usize)
                        .and_then(crate::window::button::named)
                })
                .unwrap_or(crate::window::button::A);
            u64::from(sample % every < hold) * key
        } else {
            // No window and no scripted presses: the controller is still
            // connected, with nothing pressed.
            0
        };
        self.hid_samples += 1;
        crate::hid::write_controller(
            &mut self.memory,
            base,
            self.hid_samples,
            buttons,
            crate::hid::digital_stick(buttons),
            (0, 0),
        )
    }

    fn read_string(&self, address: u64, limit: usize) -> Result<String, Trap> {
        let mut bytes = Vec::new();
        for index in 0..limit {
            let value = self
                .memory
                .read_u8(address.checked_add(index as u64).ok_or_else(|| {
                    Trap::InvalidState {
                        message: "string address overflow".into(),
                    }
                })?)?;
            if value == 0 {
                return String::from_utf8(bytes).map_err(|_| Trap::InvalidState {
                    message: "named port is not valid UTF-8".into(),
                });
            }
            bytes.push(value);
        }
        Err(Trap::InvalidState {
            message: format!("named port has no terminator within {limit} bytes"),
        })
    }

    fn map_memory_result(&mut self, destination: u64, source: u64, size: u64) -> u32 {
        // Atmosphere 6b0c077 kern_svc_memory.cpp supplies this validation order
        // and Result ABI. Pinned Eden differs for an unaligned source and omits
        // CanContain(Stack); neither reference proves a measured console layout.
        const INVALID_ADDRESS: u32 = 0xcc01;
        const INVALID_SIZE: u32 = 0xca01;
        const INVALID_CURRENT_MEMORY: u32 = 0xd401;
        const INVALID_MEMORY_REGION: u32 = 0xdc01;
        if !destination.is_multiple_of(PAGE_SIZE) || !source.is_multiple_of(PAGE_SIZE) {
            return INVALID_ADDRESS;
        }
        if size == 0 || !size.is_multiple_of(PAGE_SIZE) {
            return INVALID_SIZE;
        }
        let Some(source_end) = source.checked_add(size) else {
            return INVALID_CURRENT_MEMORY;
        };
        let Some(destination_end) = destination.checked_add(size) else {
            return INVALID_CURRENT_MEMORY;
        };
        if source_end > self.memory.address_space_end() {
            return INVALID_CURRENT_MEMORY;
        }
        let stack_end = self
            .layout
            .stack_region_base
            .checked_add(self.layout.stack_region_size);
        if self.layout.stack_region_size == 0
            || destination < self.layout.stack_region_base
            || stack_end.is_none_or(|end| destination_end > end)
            || destination_end > self.memory.address_space_end()
        {
            return INVALID_MEMORY_REGION;
        }
        // CanContain(Stack) excludes nonempty Heap and Alias ranges. There is
        // no ShadowStack region in this declared prototype address-space model.
        for (base, capacity) in [
            (self.layout.heap_base, self.layout.heap_capacity),
            (self.layout.alias_base, self.layout.alias_size),
        ] {
            if capacity != 0 {
                let Some(end) = base.checked_add(capacity) else {
                    return INVALID_MEMORY_REGION;
                };
                if destination < end && base < destination_end {
                    return INVALID_MEMORY_REGION;
                }
            }
        }
        match self.memory.map_stack_alias(destination, source, size) {
            Ok(()) => 0,
            Err(MapMemoryError::InvalidAddress) => INVALID_ADDRESS,
            Err(MapMemoryError::InvalidSize) => INVALID_SIZE,
            Err(MapMemoryError::InvalidCurrentMemory) => INVALID_CURRENT_MEMORY,
        }
    }

    /// UnmapMemory with Atmosphere's validation order: alignment, size,
    /// address-space containment, the destination within the stack region,
    /// then the page-table state (an exact MapMemory alias of the source).
    fn unmap_memory_result(&mut self, destination: u64, source: u64, size: u64) -> u32 {
        const INVALID_ADDRESS: u32 = 0xcc01;
        const INVALID_SIZE: u32 = 0xca01;
        const INVALID_CURRENT_MEMORY: u32 = 0xd401;
        const INVALID_MEMORY_REGION: u32 = 0xdc01;
        if !destination.is_multiple_of(PAGE_SIZE) || !source.is_multiple_of(PAGE_SIZE) {
            return INVALID_ADDRESS;
        }
        if size == 0 || !size.is_multiple_of(PAGE_SIZE) {
            return INVALID_SIZE;
        }
        let end = self.memory.address_space_end();
        if source
            .checked_add(size)
            .is_none_or(|source_end| source_end > end)
        {
            return INVALID_CURRENT_MEMORY;
        }
        let stack_end = self
            .layout
            .stack_region_base
            .checked_add(self.layout.stack_region_size);
        if destination < self.layout.stack_region_base
            || destination
                .checked_add(size)
                .zip(stack_end)
                .is_none_or(|(destination_end, stack_end)| destination_end > stack_end)
        {
            return INVALID_MEMORY_REGION;
        }
        match self.memory.unmap_stack_alias(destination, source, size) {
            Ok(()) => 0,
            Err(MapMemoryError::InvalidAddress) => INVALID_ADDRESS,
            Err(MapMemoryError::InvalidSize) => INVALID_SIZE,
            Err(MapMemoryError::InvalidCurrentMemory) => INVALID_CURRENT_MEMORY,
        }
    }

    fn create_thread_result(
        &mut self,
        entry: u64,
        argument: u64,
        stack_top: u64,
        priority: i32,
        core_selector: i32,
    ) -> Result<(u32, u32), Trap> {
        // Pinned Atmosphere validates virtual core (including its permission
        // bit) before priority. Only -2 selects the process's default core.
        let core = if core_selector == -2 {
            i32::from(self.layout.process_ideal_core_id)
        } else {
            core_selector
        };
        if !(0..64).contains(&core) {
            return Ok((0xe201, 0)); // InvalidCoreId.
        }
        let core_mask = self
            .layout
            .process_core_mask
            .filter(|mask| *mask != 0)
            .ok_or_else(|| Trap::InvalidState {
                message: "CreateThread requires validated nonzero process core permissions".into(),
            })?;
        if core_mask & (1u64 << core) == 0 {
            return Ok((0xe201, 0));
        }
        if !(0..64).contains(&priority) {
            return Ok((0xe001, 0)); // InvalidPriority.
        }
        let priority_mask = self
            .layout
            .process_priority_mask
            .filter(|mask| *mask != 0)
            .ok_or_else(|| Trap::InvalidState {
                message: "CreateThread requires validated nonzero process priority permissions"
                    .into(),
            })?;
        if priority_mask & (1u64 << priority) == 0 {
            return Ok((0xe001, 0));
        }

        let id = self.threads.next_id()?;
        // The guard owns an exclusive borrow of the dynamic allocator, but
        // consumes nothing unless committed. No IPC can interleave with this
        // single-thread creation path while TLS/context preparation may fail.
        let reservation = self.services.reserve_thread_handle(id)?;
        let handle = reservation.handle();
        if handle == self.layout.main_thread_handle as u32 {
            return Err(Trap::InvalidState {
                message: "configured main handle collides with the dynamic handle namespace".into(),
            });
        }
        let mut random = [0u8; 8];
        getrandom::fill(&mut random).map_err(|error| Trap::InvalidState {
            message: format!("initialize thread X18 entropy from the host: {error}"),
        })?;
        self.threads.create(
            &mut self.memory,
            self.layout.tls_base,
            ThreadCreate {
                id,
                handle,
                entry,
                argument,
                stack_top,
                priority: priority as u8,
                core: core as u8,
                x18: u64::from_le_bytes(random) | 1,
                fpcr: self.core.cpu.fpcr,
                fpsr: self.core.cpu.fpsr,
            },
        )?;
        debug_assert!(self.threads.contains(id));
        // All fallible work is finished. This publishes the typed handle for
        // the prepared Initialized thread; it does not start guest execution.
        Ok((0, reservation.commit()))
    }

    fn set_thread_core_mask_result(
        &mut self,
        handle: u32,
        core_selector: i32,
        requested_mask: u64,
    ) -> Result<u32, Trap> {
        // Pinned Atmosphere resolves the typed, current-process-owned thread
        // BEFORE validating the selector/mask. Eden uses a different order.
        // All thread records in this bounded model belong to this one process.
        let Some(id) = self.thread_id_for_handle(handle) else {
            return Ok(0xe401); // InvalidHandle.
        };
        let target = (id != self.layout.main_thread_id).then_some(id);
        let old_affinity = if let Some(id) = target {
            self.threads
                .affinity(id)
                .ok_or_else(|| Trap::InvalidState {
                    message: "thread handle has no initialized affinity backing".into(),
                })?
        } else {
            self.main_thread_affinity
        };
        let process_mask = self
            .layout
            .process_core_mask
            .filter(|mask| *mask != 0)
            .ok_or_else(|| Trap::InvalidState {
                message: "SetThreadCoreMask requires validated nonzero process core permissions"
                    .into(),
            })?;
        let (ideal_core, affinity_mask) = if core_selector == -2 {
            // UseProcessValue replaces BOTH arguments, ignoring the caller's
            // X2 entirely. Bootstrap validates this default against the grant.
            let core = self.layout.process_ideal_core_id;
            if core >= 64 || process_mask & (1u64 << core) == 0 {
                return Err(Trap::InvalidState {
                    message: "SetThreadCoreMask process default is not a permitted virtual core"
                        .into(),
                });
            }
            (i32::from(core), 1u64 << core)
        } else {
            if requested_mask & !process_mask != 0 {
                return Ok(0xe201); // InvalidCoreId, before all other mask checks.
            }
            if requested_mask == 0 {
                return Ok(0xe801); // InvalidCombination.
            }
            if (0..64).contains(&core_selector) {
                if requested_mask & (1u64 << core_selector) == 0 {
                    return Ok(0xe801);
                }
            } else if core_selector != -1 && core_selector != -3 {
                return Ok(0xe201);
            }
            let core = if core_selector == -3 {
                // The pinned C++ references shift by the retained signed ideal
                // without defining the -1 edge. Do not invent bit-63 semantics
                // or a kernel Result for NoUpdate after DontCare.
                if !(0..64).contains(&old_affinity.0) {
                    return Err(Trap::InvalidState {
                        message: "SetThreadCoreMask NoUpdate after DontCare is not established"
                            .into(),
                    });
                }
                if requested_mask & (1u64 << old_affinity.0) == 0 {
                    return Ok(0xe801);
                }
                old_affinity.0
            } else {
                core_selector
            };
            (core, requested_mask)
        };

        if let Some(id) = target {
            // The collection retains a started child's selected logical core;
            // A started child migrates (see threads::spread_processor).
            self.threads.set_affinity(id, ideal_core, affinity_mask)?;
        } else {
            let current_core = self.layout.processor_id;
            if self.main_thread_state.holds_running_reference()
                && (current_core >= 64 || old_affinity.1 & (1u64 << current_core) == 0)
            {
                return Err(Trap::InvalidState {
                    message: "main thread's current logical core is inconsistent with its affinity"
                        .into(),
                });
            }
            if self.main_thread_state.holds_running_reference()
                && affinity_mask & (1u64 << current_core) == 0
            {
                return Err(Trap::InvalidState {
                    message: "SetThreadCoreMask main-thread migration is not implemented".into(),
                });
            }
            self.main_thread_affinity = (ideal_core, affinity_mask);
        }
        Ok(0)
    }

    /// MapSharedMemory (SVC 0x13), following pinned Atmosphere 6b0c077
    /// kern_svc_shared_memory.cpp / KSharedMemory::Map: page alignment
    /// (InvalidAddress / InvalidSize), no wrap (InvalidCurrentMemory), Read or
    /// ReadWrite (InvalidNewMemoryPermission), a shared-memory handle
    /// (InvalidHandle), a Shared-capable region outside heap and alias
    /// (InvalidMemoryRegion), matching size (InvalidSize), the exact granted
    /// permission (InvalidNewMemoryPermission) and a free destination
    /// (InvalidCurrentMemory). The object's contents are placed in the pages.
    fn map_shared_memory(
        &mut self,
        handle: u32,
        address: u64,
        size: u64,
        permission: u32,
    ) -> Result<u32, Trap> {
        const PAGE: u64 = crate::memory::PAGE_SIZE;
        if !address.is_multiple_of(PAGE) {
            return Ok(0xcc01);
        }
        if !size.is_multiple_of(PAGE) || size == 0 {
            return Ok(0xca01);
        }
        let Some(end) = address.checked_add(size) else {
            return Ok(0xd401);
        };
        if !matches!(permission, 1 | 3) {
            return Ok(0xd801);
        }
        let Some(object) = self.services.shared_memory(handle).cloned() else {
            return Ok(0xe401);
        };
        let layout = &self.layout;
        let overlaps = |base: u64, length: u64| address < base.saturating_add(length) && base < end;
        if end > layout.address_space_end
            || overlaps(layout.heap_base, layout.heap_capacity)
            || overlaps(layout.alias_base, layout.alias_size)
        {
            return Ok(0xdc01);
        }
        if object.size != size {
            return Ok(0xca01);
        }
        if u32::from(object.permission) != permission {
            return Ok(0xd801);
        }
        let mut cursor = address;
        while cursor < end {
            let mapping = self.memory.query(cursor);
            if mapping.state != crate::memory::STATE_FREE {
                return Ok(0xd401);
            }
            cursor = (mapping.base + mapping.size).min(end);
        }
        self.memory.map(
            address,
            size,
            permission as u8,
            crate::memory::STATE_SHARED,
            object.label,
        )?;
        self.memory.initialize(address, &object.contents)?;
        if object.label == "hid" {
            self.hid_mapping = Some(address);
        }
        Ok(0)
    }

    pub fn svc(&mut self, number: u16) -> Result<(), Trap> {
        // An exception return clears the local exclusive monitor (Arm DDI0487
        // B2.9). Clearing on entry is equivalent here: no guest code runs
        // between this SVC's entry and its return to the caller.
        self.core.memory.clear_exclusive();
        let args = [
            self.core.cpu.regs[0],
            self.core.cpu.regs[1],
            self.core.cpu.regs[2],
            self.core.cpu.regs[3],
        ];
        let detail = if number == 0x1f {
            self.read_string(args[1], 12).ok()
        } else {
            None
        };
        let event = SvcEvent {
            thread_id: self.core.current_thread_id,
            pc: self.core.cpu.pc,
            lr: self.core.cpu.regs[30],
            number,
            arguments: args,
            arguments_4_5: (number == 0x08)
                .then_some([self.core.cpu.regs[4], self.core.cpu.regs[5]]),
            detail,
        };
        if self.recent_svcs.len() == 2048 {
            self.recent_svcs.pop_front();
        }
        self.recent_svcs.push_back(event.clone());
        if self.svc_trace.len() < 4096 {
            self.svc_trace.push(event);
        }
        match number {
            0x01 => {
                // Raw Horizon SVC ABI: size is X1; result/base return in X0/X1.
                let size = args[1];
                if !size.is_multiple_of(0x200000) || size > self.layout.heap_capacity {
                    return Err(Trap::InvalidState {
                        message: format!("unsupported heap size {size:#x}"),
                    });
                }
                if self.heap_size == 0 {
                    if size != 0 {
                        self.memory.map(
                            self.layout.heap_base,
                            size,
                            READ | WRITE,
                            STATE_NORMAL,
                            "heap",
                        )?;
                    }
                } else if size != self.heap_size {
                    self.memory
                        .resize_heap(self.layout.heap_base, self.heap_size, size)?;
                }
                self.heap_size = size;
                self.core.cpu.regs[0] = 0;
                self.core.cpu.regs[1] = self.layout.heap_base;
                Ok(())
            }
            0x02 => {
                let perm = args[2] as u32;
                if !matches!(perm, 0 | 1 | 3) {
                    return Err(Trap::InvalidState {
                        message: "SetMemoryPermission accepts None, R or RW".into(),
                    });
                }
                let end = args[0]
                    .checked_add(args[1])
                    .ok_or_else(|| Trap::InvalidState {
                        message: "permission range overflow".into(),
                    })?;
                let first = self.memory.query(args[0]);
                let mut cursor = args[0];
                while cursor < end {
                    let mapping = self.memory.query(cursor);
                    if !matches!(mapping.state, STATE_CODE_DATA | STATE_NORMAL)
                        || mapping.attributes != 0
                        || mapping.state != first.state
                        || mapping.permissions != first.permissions
                    {
                        return Err(Trap::InvalidState {message:"SetMemoryPermission requires a uniform re-protectable mapping without attributes".into()});
                    }
                    cursor = mapping
                        .base
                        .checked_add(mapping.size)
                        .ok_or_else(|| Trap::InvalidState {
                            message: "permission range leaves address space".into(),
                        })?
                        .min(end);
                }
                self.memory.set_permissions(args[0], args[1], perm as u8)?;
                self.core.cpu.regs[0] = 0;
                Ok(())
            }
            0x03 => {
                let mask = args[2] as u32;
                let attr = args[3] as u32;
                let end = args[0]
                    .checked_add(args[1])
                    .ok_or_else(|| Trap::InvalidState {
                        message: "attribute range overflow".into(),
                    })?;
                let mut cursor = args[0];
                while cursor < end {
                    let mapping = self.memory.query(cursor);
                    if !matches!(mapping.state, STATE_CODE_DATA | STATE_NORMAL) {
                        return Err(Trap::InvalidState {
                            message: "SetMemoryAttribute requires a changeable memory state".into(),
                        });
                    }
                    cursor = mapping
                        .base
                        .checked_add(mapping.size)
                        .ok_or_else(|| Trap::InvalidState {
                            message: "attribute range leaves address space".into(),
                        })?
                        .min(end);
                }
                self.memory.set_attributes(args[0], args[1], mask, attr)?;
                self.core.cpu.regs[0] = 0;
                Ok(())
            }
            0x80 => {
                // SetGpuMemoryAttribute (Switchbrew SVC, [S2]): X0 address,
                // X1 size, W2 mask, W3 attr; W0 result.
                self.core.cpu.regs[0] = u64::from(self.set_gpu_memory_attribute(
                    args[0],
                    args[1],
                    args[2] as u32,
                    args[3] as u32,
                )?);
                Ok(())
            }
            0x04 => {
                // Raw A64 ABI: X0=destination, X1=source, X2=size. X3 is not an
                // argument. Only W0 is returned, zero-extended into X0.
                self.core.cpu.regs[0] =
                    u64::from(self.map_memory_result(args[0], args[1], args[2]));
                Ok(())
            }
            0x05 => {
                // UnmapMemory: X0=destination, X1=source, X2=size; W0 result.
                self.core.cpu.regs[0] =
                    u64::from(self.unmap_memory_result(args[0], args[1], args[2]));
                Ok(())
            }
            0x06 => {
                // X0 points to MemoryInfo, X2 is the query address; W1 is an output.
                let mapping = self.memory.query(args[2]);
                let mut info = [0u8; 0x28];
                info[0..8].copy_from_slice(&mapping.base.to_le_bytes());
                info[8..16].copy_from_slice(&mapping.size.to_le_bytes());
                info[16..20].copy_from_slice(&mapping.state.to_le_bytes());
                info[20..24].copy_from_slice(&mapping.attributes.to_le_bytes());
                info[24..28].copy_from_slice(&(mapping.permissions as u32).to_le_bytes());
                self.memory.write(args[0], &info)?;
                self.core.cpu.regs[0] = 0;
                self.core.cpu.regs[1] = 0;
                Ok(())
            }
            0x07 => {
                self.process_exit_requested = true;
                self.process_exit_reason = Some("exit_process_svc");
                Err(Trap::Exit { code: 0 })
            }
            0x08 => {
                // X0 is the C wrapper's output pointer, not a raw SVC input.
                // W4/W5 are signed low-32-bit fields; return only W0/W1.
                let (result, handle) = self.create_thread_result(
                    args[1],
                    args[2],
                    args[3],
                    self.core.cpu.regs[4] as u32 as i32,
                    self.core.cpu.regs[5] as u32 as i32,
                )?;
                self.core.cpu.regs[0] = u64::from(result);
                // Error-output zero follows pinned Eden/prototype policy; it
                // is not asserted as a universal Horizon unused-output value.
                self.core.cpu.regs[1] = u64::from(handle);
                Ok(())
            }
            0x09 => {
                let result = self.start_thread_result(args[0] as u32)?;
                self.core.cpu.regs[0] = u64::from(result);
                Ok(())
            }
            0x0a => {
                if self.thread_state(self.core.current_thread_id) != Some(ThreadState::Runnable)
                    || self.pending_thread_exit.is_some()
                    || self.pending_mutex_wait.is_some()
                    || self.pending_synchronization_wait.is_some()
                {
                    return Err(Trap::InvalidState {
                        message: "ExitThread requires the active runnable context".into(),
                    });
                }
                if self.mutex_waits.has_waiters(self.core.current_thread_id) {
                    return Err(Trap::InvalidState {
                        message: "ExitThread with owned mutex waiters requires unsupported cancellation/finalization".into(),
                    });
                }
                // No register Result and no context swap inside the SVC. The
                // generated epilogue completes, then finish_block deschedules
                // and terminates this thread without executing its next PC.
                self.pending_thread_exit = Some(self.core.current_thread_id);
                Ok(())
            }
            0x0c | 0x25 => {
                // Raw ABI: W1 is the input handle; X1/W1 returns ID/priority. The
                // guest wrapper, not the SVC, writes the C output pointer.
                let handle = args[1] as u32;
                if let Some(id) = self.thread_id_for_handle(handle) {
                    let priority = if id == self.layout.main_thread_id {
                        self.main_thread_effective_priority
                    } else {
                        self.threads
                            .priority(id)
                            .ok_or_else(|| Trap::InvalidState {
                                message: "thread handle has no owned thread backing".into(),
                            })?
                    };
                    self.core.cpu.regs[0] = 0;
                    self.core.cpu.regs[1] = if number == 0x0c {
                        u64::from(priority)
                    } else {
                        id
                    };
                } else {
                    self.core.cpu.regs[0] = 0xe401;
                    self.core.cpu.regs[1] = 0;
                }
                Ok(())
            }
            0x0d => {
                // Raw ABI: W0 handle, signed W1 priority; only W0 is an output.
                let result = self.set_thread_priority(args[0] as u32, args[1] as u32 as i32)?;
                self.core.cpu.regs[0] = u64::from(result);
                Ok(())
            }
            0x0e => {
                // GetThreadCoreMask raw ABI: W2 handle in; W0 result, W1 ideal
                // core and X2 affinity mask out (Horizon; the outputs keep
                // their old values on InvalidHandle).
                let handle = args[2] as u32;
                let affinity = match self.thread_id_for_handle(handle) {
                    Some(id) if id == self.layout.main_thread_id => Some(self.main_thread_affinity),
                    Some(id) => {
                        Some(
                            self.threads
                                .affinity(id)
                                .ok_or_else(|| Trap::InvalidState {
                                    message: "thread handle has no initialized affinity backing"
                                        .into(),
                                })?,
                        )
                    }
                    None => None,
                };
                match affinity {
                    Some((core, mask)) => {
                        self.core.cpu.regs[0] = 0;
                        self.core.cpu.regs[1] = u64::from(core as u32);
                        self.core.cpu.regs[2] = mask;
                    }
                    None => self.core.cpu.regs[0] = 0xe401,
                }
                Ok(())
            }
            0x0f => {
                // Raw ABI: W0 handle, signed W1 selector, full-width X2 mask;
                // only W0 is an output. X3 and every other register survive.
                let result = self.set_thread_core_mask_result(
                    args[0] as u32,
                    args[1] as u32 as i32,
                    args[2],
                )?;
                self.core.cpu.regs[0] = u64::from(result);
                Ok(())
            }
            0x10 => {
                // With several workers, the core the thread runs on now: the
                // game indexes per-core data by it without locks, so two
                // threads must never get the same number while running at
                // once.
                let processor = match self.worker_core {
                    Some(worker) => self.guest_core_on_worker(self.core.current_thread_id, worker),
                    None => self.thread_processor_id(self.core.current_thread_id)?,
                };
                self.core.cpu.regs[0] = u64::from(processor);
                Ok(())
            }
            0x16 => {
                let handle = args[0] as u32;
                let closed = if self.main_thread_handle_open
                    && handle == self.layout.main_thread_handle as u32
                {
                    // Closing a handle does not terminate its thread. The
                    // current-thread pseudo-handle remains valid.
                    if !self.main_thread_state.holds_running_reference()
                        && self.main_mutex_owner_references == 0
                    {
                        self.release_main_thread_tls()?;
                    }
                    self.main_thread_handle_open = false;
                    true
                } else if let Some(id) = self.services.thread_id(handle) {
                    // A running reference keeps started context/TLS alive.
                    // Otherwise release only its owned TLS, never the supplied
                    // stack. Do not consume a handle when cleanup fails.
                    self.threads.close(&mut self.memory, id)?;
                    let removed = self.services.release_thread_handle(handle);
                    debug_assert_eq!(removed, Some(id));
                    true
                } else if self.services.transfer_memory(handle).is_some() {
                    // Destroying the last reference unlocks the owner's range
                    // back to read/write; a server reference keeps it locked.
                    if let Some(transfer) = self.services.remove_transfer_memory(handle) {
                        self.memory.unlock_range(
                            transfer.address,
                            transfer.size,
                            crate::memory::ATTRIBUTE_LOCKED,
                            READ | WRITE,
                        )?;
                    }
                    true
                } else {
                    self.services.close(handle)
                };
                self.core.cpu.regs[0] = if closed { 0 } else { 0xe401 };
                Ok(())
            }
            0x0b => {
                // SleepThread(s64 ns) (Switchbrew "SVC"): a positive duration
                // parks the thread until the host-monotonic deadline; 0, -1
                // and -2 are yields, modeled as a zero-length sleep so other
                // runnable threads run first. It returns nothing.
                let ns = args[0] as i64;
                if ns < -2 {
                    return Err(Trap::InvalidState {
                        message: format!("SleepThread with undefined value {ns}"),
                    });
                }
                if self.pending_synchronization_wait.is_some()
                    || self.pending_mutex_wait.is_some()
                    || self.pending_thread_exit.is_some()
                {
                    return Err(Trap::InvalidState {
                        message: "SleepThread with a pending wait or exit".into(),
                    });
                }
                let now = std::time::Instant::now();
                let deadline = if ns > 0 {
                    now + std::time::Duration::from_nanos(ns as u64)
                } else {
                    now
                };
                self.synchronization_waits.begin_sleep(
                    self.core.current_thread_id,
                    ns,
                    deadline,
                )?;
                self.pending_synchronization_wait = Some(self.core.current_thread_id);
                Ok(())
            }
            0x18 => {
                // X0 is the client wrapper's output pointer, not a kernel
                // memory output. Count zero also ignores the X1 array pointer.
                if let Some((result, index)) = self.wait_synchronization_result(
                    args[1],
                    args[2] as u32 as i32,
                    args[3] as i64,
                )? {
                    self.core.cpu.regs[0] = u64::from(result);
                    // Early errors use Eden's zero-index compatibility policy;
                    // a cancelled parked wait instead supplies low W1 = -1.
                    self.core.cpu.regs[1] = u64::from(index as u32);
                }
                Ok(())
            }
            0x19 => {
                let result = self.cancel_synchronization_result(args[0] as u32)?;
                self.core.cpu.regs[0] = u64::from(result);
                Ok(())
            }
            0x1a => {
                // Only the low W0/W2 fields are handles/tags. X1 is the full
                // process virtual address. A blocked call has no Result yet.
                if let Some(result) =
                    self.arbitrate_lock_result(args[0] as u32, args[1], args[2] as u32)?
                {
                    self.core.cpu.regs[0] = u64::from(result);
                }
                Ok(())
            }
            0x1b => {
                let result = self.arbitrate_unlock_result(args[0])?;
                self.core.cpu.regs[0] = u64::from(result);
                Ok(())
            }
            0x1c => {
                // WaitProcessWideKeyAtomic: X0 mutex, X1 key, W2 tag, X3
                // timeout. A parked wait defers W0; X1 is never an output.
                if let Some(result) =
                    self.wait_condition_result(args[0], args[1], args[2] as u32, args[3] as i64)?
                {
                    self.core.cpu.regs[0] = u64::from(result);
                }
                Ok(())
            }
            0x1d => {
                // SignalProcessWideKey: X0 key, W1 count (<= 0 wakes all).
                // It returns no result.
                self.signal_condition(args[0], args[1] as u32 as i32)?;
                Ok(())
            }
            0x34 => {
                // WaitForAddress: X0 address, W1 ArbitrationType, W2 value
                // (X2 for the 64-bit type),
                // X3 timeout. A parked wait defers W0.
                if let Some(result) =
                    self.wait_for_address_result(args[0], args[1] as u32, args[2], args[3] as i64)?
                {
                    self.core.cpu.regs[0] = u64::from(result);
                }
                Ok(())
            }
            0x36 => {
                // SynchronizePreemptionState: the thread re-enabled
                // preemption after a deferred one; clear the InterruptFlag
                // (the next boundary may switch it out).
                let tls = self.core.cpu.tpidrro_el0;
                if tls != 0 {
                    let _ = self.memory.write_u16(tls + TLS_INTERRUPT_FLAG, 0);
                }
                Ok(())
            }
            0x35 => {
                // SignalToAddress: X0 address, W1 SignalType, W2 value,
                // W3 count (<= 0 wakes all).
                let result = self.signal_to_address_result(
                    args[0],
                    args[1] as u32,
                    args[2] as u32 as i32,
                    args[3] as u32 as i32,
                )?;
                self.core.cpu.regs[0] = u64::from(result);
                Ok(())
            }
            0x11 => {
                // SignalEvent: only the writable side. A signaled syncpoint
                // cache request is served at once by its server.
                let result = if self.services.signal_event(args[0] as u32).is_some() {
                    self.services.refresh_syncpoint_cache(&mut self.memory)?;
                    0
                } else {
                    0xe401
                };
                self.core.cpu.regs[0] = result;
                Ok(())
            }
            0x12 => {
                // ClearEvent: either side.
                self.core.cpu.regs[0] = if self.services.clear_event(args[0] as u32).is_some() {
                    0
                } else {
                    0xe401
                };
                Ok(())
            }
            0x17 => {
                // ResetSignal: readable side; InvalidState when not signaled.
                self.core.cpu.regs[0] = match self.services.reset_signal(args[0] as u32) {
                    Some(true) => 0,
                    Some(false) => 0xfa01,
                    None => 0xe401,
                };
                Ok(())
            }
            0x15 => {
                // CreateTransferMemory: X1 address, X2 size, W3 permission;
                // W1 receives the handle.
                let (result, handle) =
                    self.create_transfer_memory(args[1], args[2], args[3] as u32)?;
                self.core.cpu.regs[0] = u64::from(result);
                self.core.cpu.regs[1] = u64::from(handle);
                Ok(())
            }
            0x13 => {
                // MapSharedMemory: W0 handle, X1 address, X2 size, W3 perm.
                let result =
                    self.map_shared_memory(args[0] as u32, args[1], args[2], args[3] as u32)?;
                self.core.cpu.regs[0] = u64::from(result);
                Ok(())
            }
            0x1f => {
                // X1 is the name pointer. The SVC returns W0=result, W1=handle;
                // the guest wrapper writes the pointer originally passed in X0.
                let name = self.read_string(args[1], 12)?;
                let handle = self.services.connect_named_port(&name, self.core.cpu.pc)?;
                self.core.cpu.regs[0] = 0;
                self.core.cpu.regs[1] = handle as u64;
                Ok(())
            }
            0x21 => {
                let handle = args[0] as u32;
                if !self.services.contains(handle) {
                    self.core.cpu.regs[0] = 0xe401;
                    return Ok(());
                }
                let sent = self.services.send_sync_request(
                    &mut self.memory,
                    self.core.cpu.tpidrro_el0,
                    handle,
                    self.layout.process_id,
                    self.core.cpu.pc,
                );
                if let Err(Trap::UnsupportedService { .. }) = &sent {
                    let frames = self.frame_backtrace();
                    self.debug_messages
                        .push(format!("unsupported request backtrace: {frames}"));
                }
                sent?;
                if let Some(handle) = self.services.take_blocked_request() {
                    // The request parked its thread (a bsd:u Poll with
                    // nothing ready): wait without timeout on its internal
                    // event; the service writes the reply before signaling.
                    if self.thread_state(self.core.current_thread_id) != Some(ThreadState::Runnable)
                        || self.pending_mutex_wait.is_some()
                        || self.pending_synchronization_wait.is_some()
                    {
                        return Err(Trap::InvalidState {
                            message: "a blocking request requires the active runnable context"
                                .into(),
                        });
                    }
                    self.synchronization_waits.begin_objects(
                        self.core.current_thread_id,
                        -1,
                        vec![handle],
                        None,
                    )?;
                    self.pending_synchronization_wait = Some(self.core.current_thread_id);
                    return Ok(());
                }
                // The transport result is separate from the service result in
                // the TLS reply. No reply or success is fabricated on a trap.
                self.core.cpu.regs[0] =
                    u64::from(self.services.take_transport_result().unwrap_or(0));
                Ok(())
            }
            0x26 if args[0] as u32 & 0x80000000 != 0 => Ok(()),
            0x26 => {
                // The payload (for an SDK abort, the failing Result) goes
                // to the report's debug messages when it is small and
                // readable.
                if (1..=0x100).contains(&args[2]) {
                    let mut payload = vec![0u8; args[2] as usize];
                    if self.memory.read(args[1], &mut payload).is_ok() {
                        let hex: String =
                            payload.iter().map(|byte| format!("{byte:02x}")).collect();
                        self.debug_messages.push(format!("break payload: {hex}"));
                    }
                }
                let frames = self.frame_backtrace();
                self.debug_messages
                    .push(format!("break backtrace: {frames}"));
                Err(Trap::GuestBreak {
                    pc: self.core.cpu.pc,
                    reason: args[0],
                    address: args[1],
                    size: args[2],
                })
            }
            0x27 => {
                let size = usize::try_from(args[1])
                    .ok()
                    .filter(|n| *n <= 65536)
                    .ok_or_else(|| Trap::InvalidState {
                        message: "debug output exceeds 64 KiB limit".into(),
                    })?;
                let mut text = vec![0; size];
                self.memory.read(args[0], &mut text)?;
                if self.debug_messages.len() < 256 {
                    self.debug_messages
                        .push(String::from_utf8_lossy(&text).into_owned());
                }
                self.core.cpu.regs[0] = 0;
                Ok(())
            }
            0x29 => {
                let kind = args[1] as u32;
                if kind == 0 {
                    // Atmosphere 6b0c077b4c608a3a2ee712b8dc35fee87ceb62e8,
                    // libraries/libmesosphere/source/svc/kern_svc_info.cpp:
                    // CoreMask validates the full-width subtype before looking
                    // up the W2 process handle. Only the current-process pseudo
                    // handle names a KProcess in this single-process prototype.
                    let error = if args[3] != 0 {
                        0xe801 // InvalidCombination
                    } else if args[2] as u32 != 0xffff8001 {
                        0xe401 // InvalidHandle
                    } else {
                        0
                    };
                    if error != 0 {
                        self.core.cpu.regs[0] = error;
                        // Deterministic prototype output on errors; no claim
                        // that the kernel ABI specifies the unused X1 value.
                        self.core.cpu.regs[1] = 0;
                        return Ok(());
                    }
                    let mask = self
                        .layout
                        .process_core_mask
                        .filter(|mask| *mask != 0)
                        .ok_or_else(|| Trap::InvalidState {
                            message:
                                "GetInfo CoreMask requires nonzero validated NPDM core permissions"
                                    .into(),
                        })?;
                    // Raw SVC returns W0/X1. The user wrapper, not the kernel,
                    // stores X1 through the original X0 output pointer.
                    self.core.cpu.regs[0] = 0;
                    self.core.cpu.regs[1] = mask;
                    return Ok(());
                }
                if (29..=33).contains(&kind) {
                    // Match the supplied Eden patch's extended-ACID policy:
                    // a deliberately zero-capacity remote region, no mappings,
                    // and ordinary 4 KiB process pages. This does not establish
                    // the real console's region topology or support remote SVCs.
                    let error = if args[3] != 0 {
                        0xf001
                    } else if args[2] as u32 != 0xffff8001 {
                        0xe401
                    } else if self.layout.remote_memory_model == RemoteMemoryModel::Unsupported {
                        0xf001
                    } else {
                        0
                    };
                    self.core.cpu.regs[0] = error;
                    self.core.cpu.regs[1] = if error == 0 && kind == 33 {
                        PAGE_SIZE
                    } else {
                        0
                    };
                    return Ok(());
                }
                if kind == 11 {
                    // RandomEntropy uses a null handle and four process-local
                    // words, unlike the current-process address-space queries.
                    let error = if args[2] as u32 != 0 {
                        0xe401
                    } else if args[3] >= 4 {
                        0xe801
                    } else {
                        0
                    };
                    if error != 0 {
                        self.core.cpu.regs[0] = error;
                        self.core.cpu.regs[1] = 0;
                        return Ok(());
                    }
                    let entropy =
                        self.random_entropy
                            .as_ref()
                            .ok_or_else(|| Trap::InvalidState {
                                message: "process entropy was not initialized".into(),
                            })?;
                    self.core.cpu.regs[0] = 0;
                    self.core.cpu.regs[1] = entropy[args[3] as usize];
                    return Ok(());
                }
                if args[2] as u32 != 0xffff8001 {
                    return Err(Trap::InvalidState {
                        message: format!(
                            "GetInfo supports only the current-process handle, got {:#x}",
                            args[2] as u32
                        ),
                    });
                }
                if args[3] != 0 {
                    return Err(Trap::InvalidState {
                        message: format!("unsupported GetInfo subtype {} for type {kind}", args[3]),
                    });
                }
                let value = match kind {
                    2 => self.layout.alias_base,
                    3 => self.layout.alias_size,
                    4 => self.layout.heap_base,
                    5 => self.layout.heap_capacity,
                    // Atmosphere KProcess: the private kernel resource pool is
                    // part of total and used user memory but excluded from the
                    // non-system figures (kern_k_process.cpp
                    // GetTotal/UsedNonSystemUserPhysicalMemorySize).
                    6 => self.layout.total_memory,
                    21 => self.layout.total_memory - self.layout.system_resource_size,
                    7 => self.used_memory() + self.layout.system_resource_size,
                    22 => self.used_memory(),
                    // SystemResourceSizeTotal / SystemResourceSizeUsed.
                    16 => self.layout.system_resource_size,
                    17 if self.layout.system_resource_size == 0 => 0,
                    17 => {
                        return Err(Trap::InvalidState {
                            message: "GetInfo SystemResourceSizeUsed (17) depends on kernel \
                                      page-table allocations, which are not modeled"
                                .into(),
                        });
                    }
                    12 => 0x8000000,
                    13 => self.layout.address_space_end - 0x8000000,
                    14 => self.layout.stack_region_base,
                    15 => self.layout.stack_region_size,
                    // UserExceptionContextAddress: the process local region
                    // (Atmosphere kern_svc_info.cpp).
                    20 => self.layout.process_local_region(),
                    28 => self.layout.alias_extra_size,
                    _ => {
                        return Err(Trap::InvalidState {
                            message: format!("GetInfo type {kind} is not implemented"),
                        });
                    }
                };
                self.core.cpu.regs[0] = 0;
                self.core.cpu.regs[1] = value;
                Ok(())
            }
            0x2c => {
                let result = self.map_physical_memory(args[0], args[1])?;
                self.core.cpu.regs[0] = result;
                Ok(())
            }
            0x2d => {
                let result = self.unmap_physical_memory(args[0], args[1])?;
                self.core.cpu.regs[0] = result;
                Ok(())
            }
            _ => Err(Trap::UnsupportedSvc {
                pc: self.core.cpu.pc,
                number,
            }),
        }
    }

    /// User memory in use: the process image, the heap and physical memory
    /// mapped into the alias region.
    fn used_memory(&self) -> u64 {
        self.layout.initial_used_memory + self.heap_size + self.physical_size
    }

    /// Validates a Map/UnmapPhysicalMemory range: page-aligned, non-empty and
    /// inside the alias region. Returns the Horizon result for a bad range.
    fn physical_range_error(&self, address: u64, size: u64) -> Option<u64> {
        if !address.is_multiple_of(PAGE_SIZE) {
            return Some(RESULT_INVALID_ADDRESS);
        }
        if size == 0 || !size.is_multiple_of(PAGE_SIZE) {
            return Some(RESULT_INVALID_SIZE);
        }
        let alias_end = self.layout.alias_base + self.layout.alias_size;
        let inside = address >= self.layout.alias_base
            && address
                .checked_add(size)
                .is_some_and(|end| end <= alias_end);
        (!inside).then_some(RESULT_INVALID_MEMORY_REGION)
    }

    /// SVC 0x2C MapPhysicalMemory: backs the unmapped pages of an alias-region
    /// range with zeroed read/write Normal memory, skipping pages already
    /// mapped, and counts them as used memory (Atmosphere
    /// KPageTable::MapPhysicalMemory).
    fn map_physical_memory(&mut self, address: u64, size: u64) -> Result<u64, Trap> {
        if let Some(error) = self.physical_range_error(address, size) {
            return Ok(error);
        }
        let end = address + size;
        let mut gaps = Vec::new();
        let mut cursor = address;
        while cursor < end {
            let mapping = self.memory.query(cursor);
            let next = mapping.base.saturating_add(mapping.size).min(end);
            if mapping.state == STATE_FREE {
                gaps.push((cursor, next - cursor));
            }
            cursor = next;
        }
        let needed: u64 = gaps.iter().map(|(_, size)| size).sum();
        if self.used_memory() + self.layout.system_resource_size + needed > self.layout.total_memory
        {
            return Ok(RESULT_LIMIT_REACHED);
        }
        for (base, size) in gaps {
            self.memory
                .map(base, size, READ | WRITE, STATE_NORMAL, PHYSICAL_LABEL)?;
            self.physical_size += size;
        }
        Ok(0)
    }

    /// SVC 0x2D UnmapPhysicalMemory: releases physical memory mapped in an
    /// alias-region range, including the covered part of a mapping that
    /// extends past it.
    /// The current thread's return addresses: X30, then the frame-record
    /// chain (x29 -> [previous fp, lr]), as hex, for stop diagnostics.
    fn frame_backtrace(&self) -> String {
        let mut frames = vec![format!("{:#x}", self.core.cpu.regs[30])];
        let mut fp = self.core.cpu.regs[29];
        while frames.len() < 24 && fp != 0 && fp % 16 == 0 {
            let mut record = [0u8; 16];
            if self.memory.read(fp, &mut record).is_err() {
                break;
            }
            let next = crate::bytes::le_u64(&record[..8]);
            frames.push(format!("{:#x}", crate::bytes::le_u64(&record[8..])));
            if next <= fp {
                break;
            }
            fp = next;
        }
        frames.join(" ")
    }

    fn unmap_physical_memory(&mut self, address: u64, size: u64) -> Result<u64, Trap> {
        if let Some(error) = self.physical_range_error(address, size) {
            return Ok(error);
        }
        let end = address + size;
        let mut cursor = address;
        while cursor < end {
            let mapping = self.memory.query(cursor);
            let next = mapping.base.saturating_add(mapping.size);
            if mapping.state != STATE_FREE {
                if mapping.label != PHYSICAL_LABEL
                    || ((mapping.base < address || next > end)
                        && (mapping.permissions != READ | WRITE || mapping.attributes != 0))
                {
                    return Err(Trap::InvalidState {
                        message: format!(
                            "UnmapPhysicalMemory of {address:#x}+{size:#x} covers part of \
                             {:?} at {:#x}+{:#x} (permissions {:#x}, attributes {:#x}); \
                             this partial unmap is not modeled",
                            mapping.label,
                            mapping.base,
                            mapping.size,
                            mapping.permissions,
                            mapping.attributes
                        ),
                    });
                }
                // Only the covered part goes. Changing its permissions splits
                // it from the parts outside the range without touching their
                // host mappings: other threads keep using those while this
                // runs (copying them out and mapping them again lost their
                // writes and faulted their reads meanwhile).
                let start = mapping.base.max(address);
                let stop = next.min(end);
                if start != mapping.base || stop != next {
                    self.memory.set_permissions(start, stop - start, READ)?;
                }
                self.memory.unmap_exact(start, stop - start)?;
                self.physical_size -= stop - start;
            }
            cursor = next.min(end);
        }
        Ok(0)
    }

    fn memory_model_report(&self) -> serde_json::Value {
        let aliases: Vec<_> = self
            .memory
            .stack_aliases()
            .map(|(destination, source, size)| {
                let source_mapping = self.memory.query(source);
                let destination_mapping = self.memory.query(destination);
                serde_json::json!({
                    "destination":destination,"source":source,"size":size,
                    "source_state":source_mapping.state,
                    "source_permissions":source_mapping.permissions,
                    "source_attributes":source_mapping.attributes,
                    "destination_state":destination_mapping.state,
                    "destination_permissions":destination_mapping.permissions,
                    "destination_attributes":destination_mapping.attributes,
                })
            })
            .collect();
        serde_json::json!({
            "map_memory_implemented":true,
            "unmap_memory_implemented":true,
            "map_memory_validation_policy":"atmosphere_6b0c077_with_declared_prototype_layout",
            "layout_console_verified":false,
            "page_size":PAGE_SIZE,
            "backing_model":"canonical_sparse_source_pages_with_stack_alias_ranges",
            "source_states":[STATE_CODE_DATA,STATE_NORMAL,crate::memory::STATE_ALIAS_CODE_DATA],
            "required_source_permissions":READ|WRITE,
            "required_source_attributes":0,
            "locked_source_permissions":0,
            "locked_source_attributes":crate::memory::ATTRIBUTE_LOCKED,
            "destination_state":crate::memory::STATE_STACK,
            "destination_permissions":READ|WRITE,
            "destination_attributes":0,
            "stack_region":{"base":self.layout.stack_region_base,"size":self.layout.stack_region_size},
            "excluded_heap_region":{"base":self.layout.heap_base,"size":self.layout.heap_capacity},
            "excluded_alias_region":{"base":self.layout.alias_base,"size":self.layout.alias_size},
            "shadow_stack_region_implemented":false,
            "alias_teardown_implemented":false,
            "kernel_mapping_metadata_resource_accounting_implemented":false,
            "map_memory_allocates_user_pages":false,
            "map_memory_changes_committed_user_memory":false,
            "active_stack_alias_count":aliases.len(),
            "stack_aliases":aliases,
        })
    }

    fn thread_model_report(&self) -> serde_json::Value {
        let mut model = self.threads.report();
        model["set_thread_core_mask_implemented"] = true.into();
        model["get_thread_core_mask_implemented"] = false.into();
        model["thread_core_mask_model"] =
            "mutable_virtual_affinity_with_fixed_started_logical_processors".into();
        model["core_mask_validation_policy"] =
            "atmosphere_6b0c077_handle_before_selector_and_mask".into();
        model["core_mask_supported_threads"] = serde_json::json!([
            "initialized",
            "runnable_with_migration",
            "waiting_with_migration",
            "terminated"
        ]);
        model["no_update_after_dont_care_policy"] = "unsupported_before_mutation".into();
        model["main_thread_migration_implemented"] = false.into();
        model["physical_core_topology_implemented"] = false.into();
        model["thread_pinning_implemented"] = false.into();
        model["scheduler_affinity_notifications_implemented"] = false.into();
        model["preemptions"] = serde_json::json!(self.preemptions);
        model["lock_free_sync_candidates"] = serde_json::json!({"signal_to_address_without_waiters": self.signal_to_address_without_waiters, "wait_for_address_immediate_returns": self.wait_for_address_immediate_returns});
        model["double_dispatches"] = serde_json::json!(self.double_dispatches);
        model["main_thread_affinity"] = serde_json::json!({
            "thread_id": self.layout.main_thread_id,
            "ideal_core": self.main_thread_affinity.0,
            "affinity_mask": self.main_thread_affinity.1,
            "current_processor_id": self.layout.processor_id,
        });
        model["main_thread_record"] = serde_json::json!({
            "id":self.layout.main_thread_id,
            "state":self.main_thread_state,
            "handle_open":self.main_thread_handle_open,
            "object_alive":self.main_thread_alive(),
            "running_reference":self.main_thread_state.holds_running_reference(),
            "mutex_owner_references":self.main_mutex_owner_references,
            "wait_cancelled":self.main_wait_cancelled,
            "creation_priority":self.layout.main_thread_priority,
            "base_priority":self.main_thread_base_priority,
            "effective_priority":self.main_thread_effective_priority,
            "signaled":self.main_thread_state == ThreadState::Terminated,
            "tls_allocation_owned":self.main_thread_tls_owned,
            "saved_dispatch_context_present":self.main_saved_cpu.is_some(),
            "saved_cpu_context":self.main_saved_cpu.as_ref().map(crate::threads::cpu_report),
            "instructions_started":self.main_instructions_started,
        });
        let backtrace = |context: &serde_json::Value| -> Vec<u64> {
            self.frame_chain(context["registers"][29].as_u64().unwrap_or(0))
        };
        let main_frames = backtrace(&model["main_thread_record"]["saved_cpu_context"]);
        model["main_thread_record"]["backtrace"] = main_frames.into();
        if let Some(threads) = model["threads"].as_array_mut() {
            for thread in threads {
                let frames = backtrace(&thread["saved_cpu_context"]);
                thread["backtrace"] = frames.into();
            }
        }
        model["thread_scheduler_implemented"] = true.into();
        model["exit_thread_implemented"] = true.into();
        model["thread_wait_implemented"] = true.into();
        model["thread_wait_scope"] = "mutex_and_zero_handle_synchronization".into();
        model["cancel_synchronization_implemented"] = true.into();
        model["wait_cancelled_field_semantics"] =
            "pending_synchronization_cancellation_latch".into();
        let mut mutexes = self.mutex_waits.report();
        mutexes["waits_started"] = self.mutex_waits_started.into();
        mutexes["waits_woken"] = self.mutex_waits_woken.into();
        mutexes["waits_returned"] = self.mutex_waits_returned.into();
        mutexes["raced_lock_returns"] = self.mutex_raced_lock_returns.into();
        mutexes["unlock_calls"] = self.mutex_unlock_calls.into();
        mutexes["waiting_context_count"] = (self.mutex_waits.waiting_count()
            - usize::from(self.pending_mutex_wait.is_some()))
        .into();
        mutexes["queued_waiter_count"] = self.mutex_waits.waiting_count().into();
        let owner_references = u64::from(self.main_mutex_owner_references)
            + self
                .threads
                .ids()
                .filter_map(|id| self.threads.mutex_owner_references(id))
                .map(u64::from)
                .sum::<u64>();
        mutexes["retained_owner_reference_count"] = owner_references.into();
        mutexes["arbitrate_lock_implemented"] = true.into();
        mutexes["arbitrate_unlock_implemented"] = true.into();
        mutexes["address_validation_policy"] =
            "pinned_kernel_interval_then_alignment_then_userspace_access".into();
        mutexes["kernel_interval_base"] = KERNEL_VIRTUAL_ADDRESS_BASE.into();
        mutexes["kernel_interval_end_exclusive"] = KERNEL_VIRTUAL_ADDRESS_END.into();
        mutexes["shadow_stack_region_implemented"] = false.into();
        mutexes["owner_reference_release"] = "original_owner_when_awakened_svc_resumes".into();
        mutexes["owner_exit_with_waiters_policy"] = "unsupported_before_exit_mutation".into();
        mutexes["no_runnable_policy"] =
            "typed_stop_preserving_waiting_contexts_and_references".into();
        model["mutex_waits"] = mutexes;
        let mut synchronization = self.synchronization_waits.report();
        synchronization["wait_synchronization_implemented"] = true.into();
        synchronization["wait_synchronization_scope"] =
            "zero_handles_untimed_or_immediate_return_only".into();
        synchronization["cancel_synchronization_implemented"] = true.into();
        synchronization["waits_started"] = self.synchronization_waits_started.into();
        synchronization["waits_cancelled"] = self.synchronization_waits_cancelled.into();
        synchronization["condition_waits_started"] = self.condition_waits_started.into();
        synchronization["condition_signals"] = self.condition_signals.into();
        synchronization["waits_returned"] = self.synchronization_waits_returned.into();
        synchronization["timeout_zero_returns"] = self.synchronization_timeout_zero_returns.into();
        synchronization["pending_cancel_returns"] =
            self.synchronization_pending_cancel_returns.into();
        synchronization["cancel_requests"] = self.synchronization_cancel_requests.into();
        synchronization["latched_cancel_requests"] =
            self.synchronization_latched_cancel_requests.into();
        synchronization["waiting_context_count"] = (self.synchronization_waits.waiting_count()
            - usize::from(self.pending_synchronization_wait.is_some()))
        .into();
        synchronization["zero_count_pointer_policy"] =
            "no_memory_access_for_incoming_X0_or_X1".into();
        synchronization["nonzero_handle_count_policy"] =
            "unsupported_before_pointer_access_or_state_mutation".into();
        synchronization["uncancelled_positive_timeout_policy"] =
            "unsupported_before_wait_registration".into();
        synchronization["early_error_index_policy"] =
            "zero_eden_compatibility_not_universal_kernel_value".into();
        synchronization["blocked_cancel_index"] = (-1).into();
        synchronization["register_output_policy"] = "zero_extend_W0_W1_only".into();
        synchronization["mutex_waits_cancellable"] = false.into();
        synchronization["cancellation_latch_policy"] =
            "boolean_consumed_after_termination_and_zero_timeout_checks".into();
        model["synchronization_waits"] = synchronization;
        model["scheduler"] = serde_json::json!({
            "policy":"cyclic_thread_id_after_each_completed_generated_block",
            "host_execution_threads":1,
            "current_thread_id":self.core.current_thread_id,
            "completed_blocks":self.core.completed_blocks,
            "context_switches":self.context_switches,
            "steps":self.core.steps,
            "host_check_at":self.core.host_check_at,
            "attention":self.core.attention,
            "parallel_lock":{"boundaries":self.lock_stats[0],"wait_us":self.lock_stats[1],"held_us":self.lock_stats[2],"lock_free_nvn_calls":self.lock_free_nvn_calls,"recorded_nvn_calls":self.recorded_nvn_calls,"nvn_wait_us":self.nvn_lock_stats[0],"nvn_held_us":self.nvn_lock_stats[1],"nvn_thread_wait_us":self.nvn_thread_wait.iter().map(|(thread, wait)| (thread.to_string(), *wait)).collect::<BTreeMap<_, _>>()},
            "thread_time_us":self.thread_time.iter().map(|(thread, [run, wait, held])| (thread.to_string(), serde_json::json!({"guest": run, "lock_wait": wait, "lock_held": held}))).collect::<BTreeMap<_, _>>(),
            "boundary_kinds":self.boundary_kinds.iter().enumerate().filter(|(_, count)| **count != 0).map(|(kind, count)| {
                (boundary_kind_name(kind), *count)
            }).collect::<BTreeMap<_, _>>(),
            "boundary_held_us":self.boundary_held.iter().enumerate().filter(|(_, held)| **held != 0).map(|(kind, held)| {
                (boundary_kind_name(kind), *held)
            }).collect::<BTreeMap<_, _>>(),
            "earliest_deadline_in_ns":self.synchronization_waits.earliest_deadline().map(|deadline| {
                let now = std::time::Instant::now();
                if deadline >= now {
                    (deadline - now).as_nanos() as i128
                } else {
                    -((now - deadline).as_nanos() as i128)
                }
            }).map(|ns| ns as i64),
            "thread_exits":self.thread_exits,
            "last_context_switch":self.last_context_switch,
            "active_context_owner":"machine_cpu",
            "saved_context_scope":"last_dispatch_boundary_not_active_live_registers",
            "schedule_depends_on_generated_block_partition":true,
            "budget_is_scheduling_quantum":false,
            "priority_scheduling_implemented":false,
            "instruction_preemption_implemented":false,
            "multicore_parallelism_implemented":false,
            "thread_wait_implemented":true,
            "thread_wait_scope":"mutex_and_zero_handle_synchronization",
            "termination_waiter_notification_implemented":false,
            "process_exit_cleanup_implemented":false,
            "process_exit_requested":self.process_exit_requested,
            "process_exit_reason":self.process_exit_reason,
            "pending_thread_exit":self.pending_thread_exit,
            "pending_mutex_wait":self.pending_mutex_wait,
            "pending_synchronization_wait":self.pending_synchronization_wait,
        });
        model
    }

    /// The most frequent (thread, PC) host-boundary samples, most first.
    fn sample_pc_stack(&mut self, range: &std::ffi::OsStr) {
        let parse = |text: &str| u64::from_str_radix(text.trim_start_matches("0x"), 16).ok();
        let Some((lo, hi)) = range
            .to_str()
            .and_then(|text| text.split_once('-'))
            .and_then(|(lo, hi)| Some((parse(lo)?, parse(hi)?)))
        else {
            return;
        };
        let pc = self.core.cpu.pc;
        if pc < lo || pc >= hi {
            return;
        }
        let mut stack = vec![pc, self.core.cpu.regs[30]];
        let mut frame = self.core.cpu.regs[29];
        while stack.len() < 16 && frame != 0 && frame % 8 == 0 {
            let (Ok(next), Ok(ret)) =
                (self.memory.read_u64(frame), self.memory.read_u64(frame + 8))
            else {
                break;
            };
            stack.push(ret);
            if next <= frame {
                break;
            }
            frame = next;
        }
        *self.pc_stacks.entry(stack).or_default() += 16;
    }

    pub fn top_pc_stacks(&self, count: usize) -> Vec<(Vec<u64>, u64)> {
        let mut stacks: Vec<_> = self
            .pc_stacks
            .iter()
            .map(|(s, &h)| (s.clone(), h))
            .collect();
        stacks.sort_unstable_by_key(|stack| std::cmp::Reverse(stack.1));
        stacks.truncate(count);
        stacks
    }

    pub fn top_pc_samples(&self, count: usize) -> Vec<(u64, u64, u64)> {
        let mut samples: Vec<(u64, u64, u64)> = self
            .pc_samples
            .iter()
            .map(|(&(thread, pc), &hits)| (thread, pc, hits))
            .collect();
        samples.sort_unstable_by_key(|sample| std::cmp::Reverse(sample.2));
        samples.truncate(count);
        samples
    }

    /// Debug aid: return addresses from a frame-pointer chain starting at
    /// `frame` (x29 -> {previous x29, x30}), up to 24 frames. A frame that
    /// cannot be read ends the chain.
    fn frame_chain(&self, mut frame: u64) -> Vec<u64> {
        let mut frames = Vec::new();
        while frame != 0 && frames.len() < 24 {
            let (Ok(previous), Ok(lr)) = (
                self.memory.read_u64(frame),
                self.memory.read_u64(frame.wrapping_add(8)),
            ) else {
                break;
            };
            frames.push(lr);
            if previous <= frame {
                break;
            }
            frame = previous;
        }
        frames
    }

    pub fn report(&self, stop: &Trap) -> serde_json::Value {
        serde_json::json!({
            "stop":stop,"guest_instructions":self.core.steps,"pc":self.core.cpu.pc,
            "user_preemption_deferrals":self.user_preemption_deferrals,
            "current_thread_id":self.core.current_thread_id,
            "current_processor_id":self.thread_processor_id(self.core.current_thread_id).ok(),
            "registers":self.core.cpu.regs,"sp":self.core.cpu.sp,"nzcv":self.core.cpu.nzcv,
            "backtrace":self.frame_chain(self.core.cpu.regs[29]),
            "fpcr":self.core.cpu.fpcr,"fpsr":self.core.cpu.fpsr,
            "vector_registers_hex":self.core.cpu.vectors.iter().map(|value|format!("{value:032x}")).collect::<Vec<_>>(),
            "svc_trace":self.svc_trace,"recent_svcs":self.recent_svcs,"nvn":self.nvn.report(),"debug_messages":self.debug_messages,
            "recent_pcs":self.core.recent_pcs().collect::<Vec<_>>(),"recent_thread_pcs":self.core.recent_thread_pcs(),
            "resident_guest_bytes":self.memory.resident_bytes(),
            "heap_size":self.heap_size,"layout":self.layout,
            "memory_model":self.memory_model_report(),
            "thread_model":self.thread_model_report(),
            "cpu_model":{
                "pointer_authentication":self.core.cpu.pointer_authentication_policy(),
                "top_byte_ignore":crate::cpu::PAUTH_TOP_BYTE_IGNORE,
                "guest_address_bits":crate::cpu::PAUTH_ADDRESS_BITS,
                "guest_threads":usize::from(self.main_thread_alive())+self.threads.len(),
                "executing_guest_threads":usize::from(!self.process_exit_requested && self.thread_state(self.core.current_thread_id) == Some(ThreadState::Runnable)),
                "initialized_guest_threads":self.threads.initialized_count(),
                "runnable_guest_threads":usize::from(self.main_thread_state == ThreadState::Runnable)+self.threads.runnable_count(),
                "waiting_guest_threads":self.waiting_thread_count(),
                "started_guest_threads":self.started_thread_count(),
                "terminated_guest_threads":usize::from(self.main_thread_alive() && self.main_thread_state == ThreadState::Terminated)+self.threads.terminated_count(),
                "atomic_memory_model":"cooperative_guest_threads_on_one_host_with_conservative_ordering",
                "exclusive_monitor":"single_pe_local_monitor_exact_address_and_size_with_conservative_clearing",
                "fp_status_model":"raw_FPCR_FPSR_storage_and_thread_inheritance_without_fp_arithmetic",
                "ordered_load_alignment":"natural_without_lse2_or_naa_relaxation",
                "ordered_store_alignment":"natural_without_lse2_or_naa_relaxation",
                "physical_counter":physical_counter_report(&self.core.physical_counter)
            },
            "process_entropy_initialized":self.random_entropy.is_some(),
            "main_thread_handle_open":self.main_thread_handle_open,
            "services":self.services.report(),
            "content":self.content.as_ref().map(crate::content::ContentSet::available_content_report),
        })
    }
}

/// ECHORA_WALL_SECONDS=N: an idle guest (every thread waiting, only timers
/// left) ends the run as a budget stop once N host seconds have passed since
/// the guest first went idle, so the report shows what it waits on.
fn idle_wall_limit_passed() -> bool {
    static WALL_LIMIT: std::sync::OnceLock<Option<std::time::Instant>> = std::sync::OnceLock::new();
    let now = std::time::Instant::now();
    WALL_LIMIT
        .get_or_init(|| {
            std::env::var("ECHORA_WALL_SECONDS")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .map(|seconds| now + std::time::Duration::from_secs(seconds))
        })
        .is_some_and(|limit| now >= limit)
}

/// ECHORA_STOP_SECONDS=N (diagnostic): a parallel run ends as a budget stop
/// at the first worker boundary or park after N host seconds, busy or not,
/// so a run that stops making progress still writes its report.
fn stop_seconds_passed() -> bool {
    static STOP_AT: std::sync::OnceLock<Option<std::time::Instant>> = std::sync::OnceLock::new();
    STOP_AT
        .get_or_init(|| {
            std::env::var("ECHORA_STOP_SECONDS")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .map(|seconds| std::time::Instant::now() + std::time::Duration::from_secs(seconds))
        })
        .is_some_and(|at| std::time::Instant::now() >= at)
}

/// What the workers of a parallel run share.
struct Shared<'a> {
    machine: std::sync::Mutex<&'a mut Machine>,
    /// The NVN state while workers run: an NVN call locks it alone (see
    /// `Worker::held_nvn_call`), and a boundary that needs it under the
    /// machine lock (present, lookups, host functions) swaps it into the
    /// machine, whose `nvn` holds only the entry points meanwhile. Taken
    /// after the machine lock, never before it.
    nvn: std::sync::Mutex<crate::nvn::Nvn>,
    /// Records calls on recording command buffers without the NVN state's
    /// lock (see `Worker::recorded_nvn_call`).
    recorder: crate::nvn::Recorder,
    /// The NVN thunks' entry point names, for recognizing those calls
    /// without the NVN state's lock; refreshed with the machine's table when
    /// lookups add thunks.
    thunks: std::sync::RwLock<Vec<String>>,
    /// ECHORA_EXCLUSIVE_THREADS=ID,... (diagnostic): those guest threads run
    /// only while no other worker runs guest code.
    exclusive: Option<(std::sync::RwLock<()>, Vec<u64>)>,
    /// ECHORA_STOP_WORLD (diagnostic): no guest code runs on any worker
    /// while one handles a boundary.
    world: Option<std::sync::RwLock<()>>,
    /// Per worker: signaled when a thread its cores allow may have become
    /// runnable, or the run stops.
    wake: Vec<std::sync::Condvar>,
    stop: std::sync::atomic::AtomicBool,
    /// Held while a worker runs guest code when ECHORA_SERIAL_GUEST is set:
    /// threads keep their cores but never execute at the same time, which
    /// separates scheduling faults from memory-ordering ones.
    serial: Option<std::sync::Mutex<()>>,
}

impl Shared<'_> {
    /// Wake the parked workers named by `workers` (bit per worker).
    fn wake_workers(&self, workers: u64) {
        for (index, wake) in self.wake.iter().enumerate() {
            if workers & (1 << index) != 0 {
                wake.notify_one();
            }
        }
    }
}

/// One host thread running guest threads.
struct Worker {
    core: Core,
    /// The machine's step count when this core last left it; the core's
    /// own count past that is what it executed since.
    synced: u64,
    /// Whether the core holds a thread (false while parked).
    active: bool,
    /// The modelled core this worker is.
    index: u8,
    /// A held call failed: its error to report, or none for a call the
    /// held path cannot make (it is then made under the lock).
    held_failed: Option<Option<Trap>>,
    /// Held calls since this worker last took the machine lock.
    lock_free_calls: u64,
    /// Of those, calls recorded without the NVN state's lock.
    recorded_calls: u64,
    /// Their microseconds waiting for the NVN state's lock and holding it.
    nvn_lock_micros: [u64; 2],
}

/// NVN entry points that stay under the machine lock: present publishes
/// the controller sample and may end the run, and proc-address lookups add
/// entry points. Every other NVN call can run on a worker holding the NVN
/// state (see `Worker::held_nvn_call`).
fn locked_nvn_call(name: &str) -> bool {
    matches!(
        name,
        "nvnQueuePresentTexture" | "nvnDeviceGetProcAddress" | "nvnBootstrapLoader"
    )
}

/// The `boundary_kinds` slot of a block's result: SVC numbers 0..0x80,
/// then host calls (NVN, HLE), host boundaries, time-slice ends and other
/// traps.
fn boundary_kind(result: &Result<(), Trap>) -> usize {
    match result {
        Err(Trap::SupervisorCall { number }) => (*number as usize).min(0x7f),
        Err(Trap::MissingBlock { .. }) => 0x80,
        Err(Trap::HostBoundary) => 0x81,
        Ok(()) => 0x82,
        Err(_) => 0x83,
    }
}

/// The report's name for a `boundary_kinds` slot.
fn boundary_kind_name(kind: usize) -> String {
    match kind {
        0x80 => "host call (NVN, HLE)".into(),
        0x81 => "host boundary".into(),
        0x82 => "time slice".into(),
        0x83 => "other trap".into(),
        svc => format!("svc {svc:#04x}"),
    }
}

/// Parked workers recheck waits and deadlines at least this often.
const PARKED_RECHECK: std::time::Duration = std::time::Duration::from_millis(2);

impl Worker {
    fn run(
        mut self,
        shared: &Shared,
        dispatch: &(impl Fn(&mut Core) -> Result<(), Trap> + Sync),
    ) -> (Core, Option<Trap>) {
        use std::sync::atomic::Ordering;
        let _watch = crate::fault::watch(&self.core);
        loop {
            if self.active {
                let ran = std::time::Instant::now();
                let serial = shared.serial.as_ref().map(|serial| {
                    serial
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                });
                let thread = self.core.current_thread_id;
                let (exclusive_write, exclusive_read) = match &shared.exclusive {
                    Some((lock, ids)) if ids.contains(&thread) => (
                        Some(
                            lock.write()
                                .unwrap_or_else(std::sync::PoisonError::into_inner),
                        ),
                        None,
                    ),
                    Some((lock, _)) => (
                        None,
                        Some(
                            lock.read()
                                .unwrap_or_else(std::sync::PoisonError::into_inner),
                        ),
                    ),
                    None => (None, None),
                };
                let world_read = shared.world.as_ref().map(|world| {
                    world
                        .read()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                });
                let result = loop {
                    match dispatch(&mut self.core) {
                        Ok(()) => {}
                        Err(Trap::MissingBlock { pc })
                            if self.recorded_nvn_call(shared, pc)
                                || self.held_nvn_call(shared, pc) => {}
                        result => break result,
                    }
                };
                drop(serial);
                drop((exclusive_write, exclusive_read, world_read));
                let _world_write = shared.world.as_ref().map(|world| {
                    world
                        .write()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                });
                let waited = std::time::Instant::now();
                let mut machine = shared
                    .machine
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let held = std::time::Instant::now();
                machine.lock_free_nvn_calls += std::mem::take(&mut self.lock_free_calls);
                machine.recorded_nvn_calls += std::mem::take(&mut self.recorded_calls);
                let [nvn_waited, nvn_held] = std::mem::take(&mut self.nvn_lock_micros);
                machine.nvn_lock_stats[0] += nvn_waited;
                if nvn_waited != 0 {
                    *machine.nvn_thread_wait.entry(thread).or_default() += nvn_waited;
                }
                machine.nvn_lock_stats[1] += nvn_held;
                if shared.stop.load(Ordering::Relaxed) {
                    return (self.core, None);
                }
                // A held call's error ends the run here; a call the held path
                // could not make is made under the lock below.
                let held_failed = self.held_failed.take();
                if let Some(Some(trap)) = held_failed {
                    self.swap_in(&mut machine);
                    return self.stop(shared, machine, trap);
                }
                if stop_seconds_passed() {
                    self.swap_in(&mut machine);
                    machine.core.settle_block_trap();
                    let trap = Trap::Budget {
                        pc: machine.core.cpu.pc,
                        steps: machine.core.steps,
                    };
                    return self.stop(shared, machine, trap);
                }
                machine.lock_stats[0] += 1;
                machine.lock_stats[1] += (held - waited).as_micros() as u64;
                {
                    let time = machine.thread_time.entry(thread).or_default();
                    time[0] += (waited - ran).as_micros() as u64;
                    time[1] += (held - waited).as_micros() as u64;
                }
                let kind = boundary_kind(&result);
                let host_call = match &result {
                    Err(Trap::MissingBlock { pc }) if machine.nvn.has_entry(*pc) => Some(*pc),
                    _ => None,
                };
                // Host calls (a failed held call among them) swap the NVN
                // state in for the boundary.
                self.swap_in(&mut machine);
                let outcome = if host_call.is_some() {
                    let mut nvn = shared
                        .nvn
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    std::mem::swap(&mut machine.nvn, &mut *nvn);
                    let outcome = machine.boundary(result);
                    std::mem::swap(&mut machine.nvn, &mut *nvn);
                    // A lookup added entry points: the machine's table must
                    // know their thunks are host calls.
                    if nvn.thunk_count() != machine.nvn.thunk_count() {
                        machine.nvn = nvn.entries_only();
                        *shared
                            .thunks
                            .write()
                            .unwrap_or_else(std::sync::PoisonError::into_inner) = nvn.thunk_names();
                    }
                    outcome
                } else {
                    machine.boundary(result)
                };
                machine.lock_stats[2] += held.elapsed().as_micros() as u64;
                machine.thread_time.entry(thread).or_default()[2] +=
                    held.elapsed().as_micros() as u64;
                machine.boundary_held[kind] += held.elapsed().as_micros() as u64;
                match outcome {
                    Ok(()) => {
                        machine.request_preemptions();
                        let spare = if machine.parked_workers != 0 {
                            machine.parked_workers_with_work()
                        } else {
                            0
                        };
                        self.swap_out(&mut machine, true);
                        drop(machine);
                        shared.wake_workers(spare);
                        continue;
                    }
                    Err(Trap::NoRunnableThreads { .. }) => {
                        // The thread blocked and nothing else can run here.
                        self.swap_out(&mut machine, false);
                    }
                    Err(trap) => return self.stop(shared, machine, trap),
                }
            }
            // Parked: wait for a runnable thread no other worker runs.
            let mut machine = shared
                .machine
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            machine.parked_workers += 1;
            // Parked workers' own lock use, reported as thread 10000 + index
            // (guest: rechecks; lock_held: microseconds holding the lock).
            let mut held_since = std::time::Instant::now();
            loop {
                if shared.stop.load(Ordering::Relaxed) {
                    machine.parked_workers -= 1;
                    return (self.core, None);
                }
                self.swap_in(&mut machine);
                if stop_seconds_passed() {
                    machine.parked_workers -= 1;
                    let trap = Trap::Budget {
                        pc: machine.core.cpu.pc,
                        steps: machine.core.steps,
                    };
                    return self.stop(shared, machine, trap);
                }
                let resumed = match machine.idle_until_runnable() {
                    Ok(true) => machine.host_ready().and_then(|()| machine.check_entry_pc()),
                    Ok(false) => Err(Trap::NoRunnableThreads {
                        waiting_threads: machine.waiting_thread_count(),
                    }),
                    Err(trap) => Err(trap),
                };
                match resumed {
                    Ok(()) => {
                        machine.parked_workers -= 1;
                        if crate::nvn::debug_var_os("ECHORA_TRACE_WORKERS").is_some() {
                            eprintln!(
                                "worker {} resumes thread {}",
                                self.index, machine.core.current_thread_id
                            );
                        }
                        self.swap_out(&mut machine, true);
                        break;
                    }
                    Err(Trap::NoRunnableThreads { waiting_threads }) => {
                        let deadline = machine.next_wait_deadline();
                        let everyone_parked = machine.running.is_empty();
                        // Nothing for this core may still be something for
                        // another parked worker's core.
                        let others = machine.parked_workers_with_work() & !(1 << self.index);
                        let elsewhere = others != 0;
                        shared.wake_workers(others);
                        if everyone_parked && deadline.is_none() && !elsewhere {
                            machine.parked_workers -= 1;
                            return self.stop(
                                shared,
                                machine,
                                Trap::NoRunnableThreads { waiting_threads },
                            );
                        }
                        if everyone_parked && !elsewhere && idle_wall_limit_passed() {
                            machine.parked_workers -= 1;
                            let trap = Trap::Budget {
                                pc: machine.core.cpu.pc,
                                steps: machine.core.steps,
                            };
                            return self.stop(shared, machine, trap);
                        }
                        self.swap_out(&mut machine, false);
                        let wait = deadline
                            .map(|at| at.saturating_duration_since(std::time::Instant::now()))
                            .unwrap_or(PARKED_RECHECK)
                            .min(PARKED_RECHECK);
                        {
                            let time = machine
                                .thread_time
                                .entry(10_000 + u64::from(self.index))
                                .or_default();
                            time[0] += 1;
                            time[2] += held_since.elapsed().as_micros() as u64;
                        }
                        machine = shared.wake[usize::from(self.index)]
                            .wait_timeout(machine, wait)
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .0;
                        held_since = std::time::Instant::now();
                    }
                    Err(trap) => {
                        machine.parked_workers -= 1;
                        return self.stop(shared, machine, trap);
                    }
                }
            }
        }
    }

    /// Record NVN call `pc` on a recording command buffer without any lock
    /// but that buffer's own (see `nvn::Recorder`): the game records command
    /// buffers on several cores at once, and through the NVN state's lock
    /// they waited for one another and for submitted work. The call returns
    /// to the link register as `held_nvn_call`'s do. False when `pc` is no
    /// such call (it is then made with the NVN state) or recording failed
    /// (reported at the next boundary, `held_failed`).
    fn recorded_nvn_call(&mut self, shared: &Shared, pc: u64) -> bool {
        if self.held_failed.is_some()
            || crate::nvn::debug_var_os("ECHORA_LOCKED_NVN").is_some()
            || crate::nvn::debug_var_os("ECHORA_LOCKED_SUBMIT").is_some()
            || crate::nvn::debug_var_os("ECHORA_GPU_FLOW").is_some()
        {
            return false;
        }
        let name = {
            let thunks = shared
                .thunks
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            match crate::nvn::thunk_in(&thunks, pc) {
                Some(name)
                    if name.starts_with("nvnCommandBuffer") || crate::nvn::front_call(name) =>
                {
                    name.to_owned()
                }
                _ => return false,
            }
        };
        self.core.settle_block_trap();
        let counter = self.core.physical_counter_ticks();
        let recorded = {
            let mut guest = crate::nvn::Guest {
                cpu: &mut self.core.cpu,
                memory: crate::nvn::GuestMemory::View(&mut self.core.memory),
                counter,
            };
            if name.starts_with("nvnCommandBuffer") {
                shared.recorder.record(&name, &mut guest)
            } else {
                shared.recorder.front(&name, &mut guest)
            }
        };
        match recorded {
            Ok(true) => {
                self.core.cpu.pc = self.core.cpu.regs[30];
                self.lock_free_calls += 1;
                self.recorded_calls += 1;
                true
            }
            Ok(false) => false,
            Err(trap) => {
                self.held_failed = Some(Some(trap));
                false
            }
        }
    }

    /// Make NVN call `pc` holding only the NVN state's own lock, not the
    /// machine's: recording commands is most of the game's host calls, and
    /// executing submitted work most of their time, while other guest
    /// threads need the machine. Guest memory is reached through this
    /// core's flat view, as guest code reaches it, and the call returns to
    /// the link register as `finish_host_call` does; the thread stays this
    /// worker's (an NVN call never blocks it). False when `pc` is not such
    /// a call or the call failed: the boundary then takes the machine lock,
    /// where a failed call's error is reported, or a call this path cannot
    /// make is made (`held_failed`). ECHORA_LOCKED_NVN (or its older name
    /// ECHORA_LOCKED_SUBMIT; diagnostic) makes every call under the lock.
    fn held_nvn_call(&mut self, shared: &Shared, pc: u64) -> bool {
        if self.held_failed.is_some()
            || crate::nvn::debug_var_os("ECHORA_LOCKED_NVN").is_some()
            || crate::nvn::debug_var_os("ECHORA_LOCKED_SUBMIT").is_some()
        {
            return false;
        }
        let asked = std::time::Instant::now();
        let mut nvn = shared
            .nvn
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let locked = std::time::Instant::now();
        let made = self.held_nvn_call_locked(&mut nvn, pc);
        drop(nvn);
        self.nvn_lock_micros[0] += (locked - asked).as_micros() as u64;
        self.nvn_lock_micros[1] += locked.elapsed().as_micros() as u64;
        made
    }

    /// `held_nvn_call` with the NVN state locked.
    fn held_nvn_call_locked(&mut self, nvn: &mut crate::nvn::Nvn, pc: u64) -> bool {
        let Some(crate::nvn::HostEntry::Proc(name)) = nvn.entry(pc) else {
            return false;
        };
        if locked_nvn_call(&name) {
            return false;
        }
        self.core.settle_block_trap();
        let counter = self.core.physical_counter_ticks();
        let started = std::time::Instant::now();
        let outcome = {
            let mut guest = crate::nvn::Guest {
                cpu: &mut self.core.cpu,
                memory: crate::nvn::GuestMemory::View(&mut self.core.memory),
                counter,
            };
            nvn.call(&name, &mut guest)
        };
        match outcome {
            Ok(true) => {
                *nvn.calls.entry(name.clone()).or_default() += 1;
                if crate::nvn::debug_var_os("ECHORA_NVN_PROFILE").is_some() {
                    *nvn.call_microseconds.entry(name).or_default() +=
                        started.elapsed().as_micros() as u64;
                }
                self.core.cpu.pc = self.core.cpu.regs[30];
                self.lock_free_calls += 1;
                true
            }
            Ok(false) => {
                self.held_failed = Some(None);
                false
            }
            Err(trap) => {
                self.held_failed = Some(Some(trap));
                false
            }
        }
    }

    /// Put this worker's core into the machine for host work, crediting
    /// the steps it ran since it last left to the machine's count.
    fn swap_in(&mut self, machine: &mut Machine) {
        let shift = machine.core.steps.wrapping_sub(self.synced);
        self.core.steps = self.core.steps.wrapping_add(shift);
        self.core.thread_steps_base = self.core.thread_steps_base.wrapping_add(shift);
        self.core.slice_start = self.core.slice_start.wrapping_add(shift);
        std::mem::swap(&mut machine.core, &mut self.core);
        machine.worker_core = Some(self.index);
        if self.active {
            machine.running.remove(&machine.core.current_thread_id);
        }
    }

    fn swap_out(&mut self, machine: &mut Machine, active: bool) {
        if active && !machine.running.insert(machine.core.current_thread_id) {
            // Two workers running one guest thread from the same saved
            // context corrupt it: report it where it happens.
            eprintln!(
                "worker {}: thread {} is already running on another worker (pc {:#x})",
                self.index, machine.core.current_thread_id, machine.core.cpu.pc
            );
            machine.double_dispatches += 1;
        }
        if let Some(slot) = machine.core_threads.get_mut(usize::from(self.index)) {
            *slot = active.then_some(machine.core.current_thread_id);
        }
        std::mem::swap(&mut machine.core, &mut self.core);
        machine.core.steps = self.core.steps;
        self.synced = self.core.steps;
        self.active = active;
    }

    /// End the run for every worker with `trap` as this worker's result.
    fn stop(
        mut self,
        shared: &Shared,
        mut machine: std::sync::MutexGuard<&mut Machine>,
        trap: Trap,
    ) -> (Core, Option<Trap>) {
        shared
            .stop
            .store(true, std::sync::atomic::Ordering::Relaxed);
        self.swap_out(&mut machine, false);
        drop(machine);
        shared.wake_workers(u64::MAX);
        (self.core, Some(trap))
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(all(test, any(target_os = "linux", windows)))]
mod tests {
    use super::*;
    use crate::memory::{
        ATTRIBUTE_LOCKED, EXECUTE, STATE_ALIAS_CODE_DATA, STATE_CODE, STATE_STACK, STATE_TLS,
    };

    fn thread_machine() -> Machine {
        let layout = Layout {
            process_core_mask: Some(0x3f),
            process_priority_mask: Some(((1u64 << 60) - 1) & !((1u64 << 28) - 1)),
            process_ideal_core_id: 5,
            initial_used_memory: 0x4000,
            ..Layout::default()
        };
        let mut m = Machine::new(layout, 1);
        m.memory
            .map(
                m.layout.tls_base,
                PAGE_SIZE,
                READ | WRITE,
                STATE_TLS,
                "main TLS",
            )
            .unwrap();
        m.memory
            .write_u64(m.layout.tls_base + 0x120, 0xfeedface)
            .unwrap();
        m.register_main_thread_tls().unwrap();
        m.memory
            .map(
                0x8000,
                PAGE_SIZE,
                READ | WRITE,
                STATE_CODE_DATA,
                "stack backing",
            )
            .unwrap();
        m.memory.write_u64(0x8000, 0x0123456789abcdef).unwrap();
        m.memory
            .map_stack_alias(m.layout.stack_region_base, 0x8000, PAGE_SIZE)
            .unwrap();
        m.core.cpu.regs = std::array::from_fn(|index| 0xfedcba9800000000 + index as u64);
        m.core.cpu.pc = 0x1000;
        m.core.cpu.sp = 0xdead;
        m.core.cpu.nzcv = 0xb;
        m.core.cpu.vectors[3] = u128::MAX;
        m.core.cpu.tpidr_el0 = 0x1234;
        m.core.cpu.tpidrro_el0 = m.layout.tls_base;
        m.core.cpu.fpcr = 0x00400000;
        m.core.cpu.fpsr = 1;
        m
    }

    fn create_thread_args(m: &mut Machine, priority: u64, core: u64) {
        // X0 and the opaque argument need not be mapped. Entry intentionally
        // retains bit 1 after bit 0 is cleared; creation does not execute it.
        m.core.cpu.regs[..6].copy_from_slice(&[
            u64::MAX,
            0x1234567,
            u64::MAX - 8,
            m.layout.stack_region_base + PAGE_SIZE - 5,
            priority,
            core,
        ]);
    }

    fn mutex_test_machine() -> Machine {
        let mut m = thread_machine();
        m.memory
            .map(0xa000, PAGE_SIZE, READ | WRITE, STATE_NORMAL, "mutex words")
            .unwrap();
        m
    }

    fn mutex_test_child(m: &mut Machine, priority: u64, start: bool) -> u32 {
        create_thread_args(m, priority, 5);
        m.svc(8).unwrap();
        let handle = m.core.cpu.regs[1] as u32;
        if start {
            m.core.cpu.regs[0] = u64::from(handle);
            m.svc(9).unwrap();
            assert_eq!(m.core.cpu.regs[0], 0);
        }
        handle
    }

    fn mutex_test_lock(m: &mut Machine, owner: u32, address: u64, tag: u32) {
        m.core.cpu.regs[..3].copy_from_slice(&[u64::from(owner), address, u64::from(tag)]);
        m.svc(0x1a).unwrap();
        m.core.cpu.pc += 4;
        m.finish_block().unwrap();
    }

    fn zero_handle_wait_args(m: &mut Machine, timeout_ns: i64) {
        m.core.cpu.regs[..4].copy_from_slice(&[
            u64::MAX,
            u64::MAX - 2,
            0xabcdef1200000000,
            timeout_ns as u64,
        ]);
    }

    #[test]
    fn zero_handle_wait_checks_signed_count_and_early_results_without_pointer_access() {
        for (count, timeout, terminating, cancelled, expected, remaining_cancel) in [
            (0, 0, false, false, Some(0xea01), false),
            (0xabcdef1200000000, 0, false, true, Some(0xea01), true),
            (0, -1, false, true, Some(0xec01), false),
            (0, 1, false, true, Some(0xec01), false),
            (0, 1, false, false, None, false),
            (0, i64::MAX, false, false, None, false),
            (0, 0, true, true, Some(0x7601), true),
            (0, -1, true, true, Some(0x7601), true),
            (0xabcdef12ffffffff, -1, true, true, Some(0xee01), true),
            (65, -1, true, true, Some(0xee01), true),
            (0xabcdef1280000000, 0, false, true, Some(0xee01), true),
            // Nonzero counts copy the handle array first; this fixture's X1
            // is not readable guest memory.
            (1, -1, false, true, Some(0xcc01), true),
            (64, 0, false, true, Some(0xcc01), true),
        ] {
            let mut m = thread_machine();
            zero_handle_wait_args(&mut m, timeout);
            m.core.cpu.regs[2] = count;
            m.main_wait_cancelled = cancelled;
            m.pending_thread_exit = terminating.then_some(m.current_thread_id());
            let before_cpu = m.core.cpu.clone();
            let before_model = m.thread_model_report();
            let resident = m.memory.resident_bytes();
            let result = m.svc(0x18);
            if let Some(result_code) = expected {
                result.unwrap();
                let mut expected_cpu = before_cpu;
                expected_cpu.regs[0] = result_code;
                expected_cpu.regs[1] = 0; // Declared early-error output policy.
                assert_eq!(m.core.cpu, expected_cpu);
            } else {
                // A positive timeout parks a timed wait: both result
                // registers stay deferred until it times out or is cancelled.
                result.unwrap();
                assert_eq!(m.core.cpu, before_cpu);
                assert_ne!(m.thread_model_report(), before_model);
            }
            let parked = expected.is_none();
            assert_eq!(m.main_wait_cancelled, remaining_cancel);
            assert_eq!(m.main_thread_state, ThreadState::Runnable);
            assert_eq!(
                m.pending_synchronization_wait,
                parked.then_some(m.current_thread_id())
            );
            assert_eq!(m.synchronization_waits.waiting_count(), usize::from(parked));
            assert_eq!(m.memory.resident_bytes(), resident);
        }
    }

    #[test]
    fn condition_wait_releases_the_mutex_sets_the_key_and_times_out() {
        let mut m = thread_machine();
        let base = 0x1_0000_0000;
        m.memory
            .map(base, 0x1000, READ | WRITE, STATE_NORMAL, "heap")
            .unwrap();
        let (mutex, key) = (base, base + 8);
        let tag = m.layout.main_thread_handle as u32;
        m.memory.write_u32(mutex, tag).unwrap();
        // Zero timeout: unlock and key update happen, then TimedOut.
        m.core.cpu.regs[..4].copy_from_slice(&[mutex, key, u64::from(tag), 0]);
        m.core.cpu.regs[1] = key;
        let x1 = m.core.cpu.regs[1];
        m.svc(0x1c).unwrap();
        assert_eq!(m.core.cpu.regs[0], 0xea01);
        assert_eq!(m.core.cpu.regs[1], x1);
        assert_eq!(m.memory.read_u32(mutex).unwrap(), 0);
        assert_eq!(m.memory.read_u32(key).unwrap(), 1);
        // A timed wait parks, then idles to its deadline and returns TimedOut
        // in W0 only.
        m.memory.write_u32(mutex, tag).unwrap();
        m.core.cpu.regs[..4].copy_from_slice(&[mutex, key, u64::from(tag), 1_000_000]);
        m.svc(0x1c).unwrap();
        assert_eq!(m.pending_synchronization_wait, Some(m.current_thread_id()));
        assert!(matches!(
            m.finish_block(),
            Err(Trap::NoRunnableThreads { .. })
        ));
        assert!(m.idle_until_runnable().unwrap());
        assert_eq!(m.core.cpu.regs[0], 0xea01);
        assert_eq!(m.core.cpu.regs[1], key);
        // Signaling with no waiters clears the key.
        m.core.cpu.regs[..2].copy_from_slice(&[key, 1]);
        m.svc(0x1d).unwrap();
        assert_eq!(m.memory.read_u32(key).unwrap(), 0);
    }

    #[test]
    fn address_arbiter_compares_updates_and_times_out() {
        let mut m = thread_machine();
        let base = 0x1_0000_0000;
        m.memory
            .map(base, 0x1000, READ | WRITE, STATE_NORMAL, "heap")
            .unwrap();
        let word = base + 0x10;
        let call = |m: &mut Machine, svc: u16, args: [u64; 4]| {
            m.core.cpu.regs[..4].copy_from_slice(&args);
            m.svc(svc).unwrap();
            m.core.cpu.regs[0]
        };
        // Unaligned address, unknown type, failed comparisons.
        assert_eq!(call(&mut m, 0x34, [word + 1, 2, 0, 0]), 0xcc01);
        assert_eq!(call(&mut m, 0x34, [word, 4, 0, 0]), 0xf001);
        assert_eq!(call(&mut m, 0x35, [word, 3, 0, 0]), 0xf001);
        m.memory.write_u32(word, 5).unwrap();
        assert_eq!(call(&mut m, 0x34, [word, 2, 4, u64::MAX]), 0xfa01);
        assert_eq!(call(&mut m, 0x34, [word, 0, 5, u64::MAX]), 0xfa01);
        // DecrementAndWaitIfLessThan with a zero timeout: 5 < 6, so the word
        // is decremented, then TimedOut.
        assert_eq!(call(&mut m, 0x34, [word, 1, 6, 0]), 0xea01);
        assert_eq!(m.memory.read_u32(word).unwrap(), 4);
        // A timed WaitIfEqual parks and times out with W0 only.
        m.core.cpu.regs[..4].copy_from_slice(&[word, 2, 4, 1_000_000]);
        m.svc(0x34).unwrap();
        assert_eq!(m.pending_synchronization_wait, Some(m.current_thread_id()));
        assert!(matches!(
            m.finish_block(),
            Err(Trap::NoRunnableThreads { .. })
        ));
        assert!(m.idle_until_runnable().unwrap());
        assert_eq!(m.core.cpu.regs[0], 0xea01);
        assert_eq!(m.core.cpu.regs[1], 2);
        // Signals: plain, increment-if-equal, modify-by-waiting-count with no
        // waiters (value + 1), and a failed comparison.
        assert_eq!(call(&mut m, 0x35, [word, 0, 0, 1]), 0);
        assert_eq!(call(&mut m, 0x35, [word, 1, 4, 1]), 0);
        assert_eq!(m.memory.read_u32(word).unwrap(), 5);
        assert_eq!(call(&mut m, 0x35, [word, 2, 5, 1]), 0);
        assert_eq!(m.memory.read_u32(word).unwrap(), 6);
        assert_eq!(call(&mut m, 0x35, [word, 1, 5, 1]), 0xfa01);
        assert_eq!(m.memory.read_u32(word).unwrap(), 6);
    }

    #[test]
    fn wait_if_equal_64_compares_the_full_register() {
        let mut m = thread_machine();
        let base = 0x1_0000_0000;
        m.memory
            .map(base, 0x1000, READ | WRITE, STATE_NORMAL, "heap")
            .unwrap();
        let word = base + 0x20;
        let call = |m: &mut Machine, args: [u64; 4]| {
            m.core.cpu.regs[..4].copy_from_slice(&args);
            m.svc(0x34).unwrap();
            m.core.cpu.regs[0]
        };
        m.memory.write_u64(word, 0x1_0000_0005).unwrap();
        // 4-byte but not 8-byte aligned: InvalidAddress.
        assert_eq!(call(&mut m, [word + 4, 3, 5, 0]), 0xcc01);
        // Only the low halves match: no wait.
        assert_eq!(call(&mut m, [word, 3, 5, 0]), 0xfa01);
        // Equal 64-bit values with a zero timeout: TimedOut, memory unchanged.
        assert_eq!(call(&mut m, [word, 3, 0x1_0000_0005, 0]), 0xea01);
        assert_eq!(m.memory.read_u64(word).unwrap(), 0x1_0000_0005);
        // A timed wait parks, then times out.
        m.core.cpu.regs[..4].copy_from_slice(&[word, 3, 0x1_0000_0005, 1_000_000]);
        m.svc(0x34).unwrap();
        assert_eq!(m.pending_synchronization_wait, Some(m.current_thread_id()));
        assert!(matches!(
            m.finish_block(),
            Err(Trap::NoRunnableThreads { .. })
        ));
        assert!(m.idle_until_runnable().unwrap());
        assert_eq!(m.core.cpu.regs[0], 0xea01);
    }

    #[test]
    fn transfer_memory_locks_its_range_until_the_handle_closes() {
        let mut m = thread_machine();
        let base = 0x1_0000_0000;
        m.memory
            .map(base, 0x3000, READ | WRITE, STATE_NORMAL, "heap")
            .unwrap();
        // Misaligned address, zero size and an invalid permission fail early.
        for (address, size, permission, expected) in [
            (base + 1, 0x1000, 1, 0xcc01),
            (base, 0, 1, 0xca01),
            (base, 0x1000, 2, 0xd801),
            (base, 0x8000, 1, 0xd401),
        ] {
            m.core.cpu.regs[1] = address;
            m.core.cpu.regs[2] = size;
            m.core.cpu.regs[3] = permission;
            m.svc(0x15).unwrap();
            assert_eq!(m.core.cpu.regs[0], expected);
        }
        m.core.cpu.regs[1] = base + 0x1000;
        m.core.cpu.regs[2] = 0x2000;
        m.core.cpu.regs[3] = 1;
        m.svc(0x15).unwrap();
        assert_eq!(m.core.cpu.regs[0], 0);
        let handle = m.core.cpu.regs[1];
        let locked = m.memory.query(base + 0x1000);
        assert_eq!(
            (locked.permissions, locked.attributes),
            (READ, crate::memory::ATTRIBUTE_LOCKED)
        );
        assert_eq!(m.memory.query(base).permissions, READ | WRITE);
        // A second lock of an already locked page fails without a handle.
        m.core.cpu.regs[1] = base + 0x2000;
        m.core.cpu.regs[2] = 0x1000;
        m.svc(0x15).unwrap();
        assert_eq!(m.core.cpu.regs[0], 0xd401);
        m.core.cpu.regs[0] = handle;
        m.svc(0x16).unwrap();
        assert_eq!(m.core.cpu.regs[0], 0);
        let unlocked = m.memory.query(base + 0x1000);
        assert_eq!(
            (unlocked.permissions, unlocked.attributes),
            (READ | WRITE, 0)
        );
    }

    #[test]
    fn timed_wait_parks_then_idles_until_its_deadline_and_returns_timed_out() {
        let mut m = thread_machine();
        zero_handle_wait_args(&mut m, 1_000_000); // 1 ms, no objects
        m.core.cpu.regs[2] = 0;
        let continuation = m.core.cpu.clone();
        m.svc(0x18).unwrap();
        assert_eq!(m.pending_synchronization_wait, Some(m.current_thread_id()));
        assert!(matches!(
            m.finish_block(),
            Err(Trap::NoRunnableThreads { .. })
        ));
        assert_eq!(m.main_thread_state, ThreadState::Waiting);
        let started = std::time::Instant::now();
        assert!(m.idle_until_runnable().unwrap());
        assert!(started.elapsed() >= std::time::Duration::from_millis(1));
        assert_eq!(m.main_thread_state, ThreadState::Runnable);
        assert_eq!(m.core.cpu.regs[0], 0xea01);
        assert_eq!(m.core.cpu.regs[1], u64::from(u32::MAX));
        assert_eq!(m.core.cpu.pc, continuation.pc);
        assert_eq!(m.synchronization_waits.waiting_count(), 0);
    }

    #[test]
    fn cancel_synchronization_checks_typed_handles_and_retains_latch_across_states() {
        let mut m = mutex_test_machine();
        let child = mutex_test_child(&mut m, 28, false);
        m.core.cpu.regs[0] = 0xabcdef1200000000 | u64::from(child);
        let mut expected = m.core.cpu.clone();
        expected.regs[0] = 0;
        m.svc(0x19).unwrap();
        assert_eq!(m.core.cpu, expected);
        assert_eq!(m.threads.wait_cancelled(2), Some(true));
        assert_eq!(m.threads.state(2), Some(ThreadState::Initialized));
        let session = m.services.connect_named_port("sm:", 0x1000).unwrap();
        for handle in [0, 0xdead, 0xffff8001, child | MUTEX_WAIT_MASK, session] {
            m.core.cpu.regs[0] = u64::from(handle);
            let mut expected = m.core.cpu.clone();
            expected.regs[0] = 0xe401;
            let model = m.thread_model_report();
            m.svc(0x19).unwrap();
            assert_eq!(m.core.cpu, expected);
            assert_eq!(m.thread_model_report(), model);
        }
        m.core.cpu.regs[0] = u64::from(child);
        m.svc(9).unwrap();
        assert_eq!(m.threads.wait_cancelled(2), Some(true));
        m.finish_block().unwrap();
        assert_eq!(m.current_thread_id(), 2);
        zero_handle_wait_args(&mut m, -1);
        m.svc(0x18).unwrap();
        assert_eq!((m.core.cpu.regs[0], m.core.cpu.regs[1]), (0xec01, 0));
        assert_eq!(m.threads.wait_cancelled(2), Some(false));
        m.svc(0xa).unwrap();
        m.core.cpu.pc += 4;
        m.finish_block().unwrap();
        assert_eq!(m.current_thread_id(), 1);
        assert_eq!(m.threads.state(2), Some(ThreadState::Terminated));
        m.core.cpu.regs[0] = u64::from(child);
        m.svc(0x19).unwrap();
        assert_eq!(m.core.cpu.regs[0], 0);
        assert_eq!(m.threads.wait_cancelled(2), Some(true));
        assert_eq!(m.threads.state(2), Some(ThreadState::Terminated));
        m.core.cpu.regs[0] = u64::from(child);
        m.svc(0x16).unwrap();
        m.core.cpu.regs[0] = u64::from(child);
        m.svc(0x19).unwrap();
        assert_eq!(m.core.cpu.regs[0], 0xe401);
        assert_eq!(m.threads.wait_cancelled(2), None);
        m.core.cpu.regs[0] = m.layout.main_thread_handle;
        m.svc(0x16).unwrap();
        m.core.cpu.regs[0] = m.layout.main_thread_handle;
        m.svc(0x19).unwrap();
        assert_eq!(m.core.cpu.regs[0], 0xe401);
        for _ in 0..2 {
            m.core.cpu.regs[0] = 0xabcdef12ffff8000;
            m.svc(0x19).unwrap();
            assert_eq!(m.core.cpu.regs[0], 0);
            assert!(m.main_wait_cancelled);
        }
        zero_handle_wait_args(&mut m, 0);
        m.svc(0x18).unwrap();
        assert_eq!(m.core.cpu.regs[0], 0xea01);
        assert!(m.main_wait_cancelled);
        zero_handle_wait_args(&mut m, -1);
        m.svc(0x18).unwrap();
        assert_eq!((m.core.cpu.regs[0], m.core.cpu.regs[1]), (0xec01, 0));
        assert!(!m.main_wait_cancelled); // Repeated requests coalesced to one latch.
        // A positive timeout now parks a timed wait.
        zero_handle_wait_args(&mut m, 1);
        m.svc(0x18).unwrap();
        assert_eq!(m.pending_synchronization_wait, Some(m.current_thread_id()));
    }

    #[test]
    fn zero_handle_wait_parks_full_continuation_and_cancellation_race_preserves_next_latch() {
        for waiting_main in [false, true] {
            let mut m = mutex_test_machine();
            let child = mutex_test_child(&mut m, 28, true);
            if !waiting_main {
                m.finish_block().unwrap();
            }
            let waiting = m.current_thread_id();
            let handle = if waiting_main { 0x100 } else { child };
            m.core.cpu.regs = std::array::from_fn(|index| 0x9988776600000000 + index as u64);
            m.core.cpu.vectors = std::array::from_fn(|index| u128::MAX - index as u128);
            m.core.cpu.nzcv = 0xe;
            m.core.cpu.fpcr = 0x00800000;
            m.core.cpu.fpsr = 0x10;
            m.core.cpu.tpidr_el0 = 0x123456789abcdef0;
            zero_handle_wait_args(&mut m, i64::MIN);
            let before = m.core.cpu.clone();
            m.svc(0x18).unwrap();
            assert_eq!(m.core.cpu, before);
            assert_eq!(m.pending_synchronization_wait, Some(waiting));
            assert_eq!(m.thread_state(waiting), Some(ThreadState::Runnable));
            assert!(matches!(m.tick(), Err(Trap::InvalidState { .. })));
            assert!(matches!(
                m.cancel_synchronization_result(0xffff8000),
                Err(Trap::InvalidState { .. })
            ));
            assert_eq!(m.core.cpu, before);
            m.core.cpu.pc += 4;
            let mut resumed = m.core.cpu.clone();
            resumed.regs[0] = 0xec01;
            resumed.regs[1] = u64::from(u32::MAX);
            m.finish_block().unwrap();
            assert_ne!(m.current_thread_id(), waiting);
            assert_eq!(m.thread_state(waiting), Some(ThreadState::Waiting));
            assert_eq!(m.waiting_thread_count(), 1);
            let model = m.thread_model_report();
            assert_eq!(model["mutex_waits"]["waiting_context_count"], 0);
            assert_eq!(model["synchronization_waits"]["waiting_context_count"], 1);
            assert_eq!(
                model["synchronization_waits"]["retained_object_reference_count"],
                0
            );
            m.core.cpu.regs[0] = u64::from(handle);
            let mut cancelling_cpu = m.core.cpu.clone();
            cancelling_cpu.regs[0] = 0;
            m.svc(0x19).unwrap();
            assert_eq!(m.core.cpu, cancelling_cpu);
            assert_eq!(m.thread_state(waiting), Some(ThreadState::Runnable));
            assert!(!m.thread_wait_cancelled(waiting).unwrap());
            assert!(m.synchronization_waits.pending_return(waiting).is_some());
            assert_eq!(m.saved_thread_cpu(waiting).unwrap(), resumed);
            // A second cancellation after wake, before actual guest resumption,
            // queues the NEXT cancellation without altering this returned Result.
            m.core.cpu.regs[0] = u64::from(handle);
            m.svc(0x19).unwrap();
            assert!(m.thread_wait_cancelled(waiting).unwrap());
            assert_eq!(m.saved_thread_cpu(waiting).unwrap(), resumed);
            m.core.cpu.pc += 4;
            m.finish_block().unwrap();
            assert_eq!(m.current_thread_id(), waiting);
            assert_eq!(m.core.cpu, resumed);
            assert!(m.synchronization_waits.pending_return(waiting).is_none());
            assert_eq!(m.synchronization_waits_returned, 1);
            assert!(m.thread_wait_cancelled(waiting).unwrap());
            zero_handle_wait_args(&mut m, -2);
            m.svc(0x18).unwrap();
            assert_eq!((m.core.cpu.regs[0], m.core.cpu.regs[1]), (0xec01, 0));
            assert!(!m.thread_wait_cancelled(waiting).unwrap());
            assert_eq!(m.synchronization_waits_started, 1);
            assert_eq!(m.synchronization_waits_cancelled, 1);
            assert_eq!(m.synchronization_pending_cancel_returns, 1);
        }
    }

    #[test]
    fn cancelling_mutex_waiter_only_latches_until_a_later_synchronization_wait() {
        let mut m = mutex_test_machine();
        let child = mutex_test_child(&mut m, 28, true);
        m.memory.write_u32(0xa000, 0x40000100).unwrap();
        m.finish_block().unwrap();
        mutex_test_lock(&mut m, 0x100, 0xa000, child);
        let mutex_model = m.thread_model_report()["mutex_waits"].clone();
        let saved = m.threads.report()["threads"][0]["saved_cpu_context"].clone();
        m.core.cpu.regs[0] = u64::from(child);
        m.svc(0x19).unwrap();
        assert_eq!(m.core.cpu.regs[0], 0);
        assert_eq!(m.threads.state(2), Some(ThreadState::Waiting));
        assert_eq!(m.threads.wait_cancelled(2), Some(true));
        assert_eq!(m.thread_model_report()["mutex_waits"], mutex_model);
        assert_eq!(m.threads.report()["threads"][0]["saved_cpu_context"], saved);
        assert_eq!(m.main_mutex_owner_references, 1);
        m.core.cpu.regs[0] = 0xa000;
        m.svc(0x1b).unwrap();
        m.core.cpu.pc += 4;
        m.finish_block().unwrap();
        assert_eq!(m.current_thread_id(), 2);
        assert_eq!(m.core.cpu.regs[0], 0); // Original mutex wait was not cancelled.
        assert_eq!(m.main_mutex_owner_references, 0);
        assert!(m.thread_wait_cancelled(2).unwrap());
        zero_handle_wait_args(&mut m, 0);
        m.svc(0x18).unwrap();
        assert_eq!(m.core.cpu.regs[0], 0xea01);
        assert!(m.thread_wait_cancelled(2).unwrap());
        zero_handle_wait_args(&mut m, -1);
        m.svc(0x18).unwrap();
        assert_eq!(m.core.cpu.regs[0], 0xec01);
        assert!(!m.thread_wait_cancelled(2).unwrap());
        assert_eq!(m.synchronization_waits_started, 0);
        assert_eq!(m.synchronization_waits_cancelled, 0);
    }

    fn set_priority(m: &mut Machine, handle: u64, priority: u64) -> u64 {
        m.core.cpu.regs[..2].copy_from_slice(&[handle, priority]);
        m.svc(0x0d).unwrap();
        m.core.cpu.regs[0]
    }

    #[test]
    fn get_thread_core_mask_reports_the_set_affinity() {
        let mut m = thread_machine();
        let get = |m: &mut Machine, handle: u64| {
            m.core.cpu.regs[2] = handle;
            m.svc(0x0e).unwrap();
            (m.core.cpu.regs[0], m.core.cpu.regs[1], m.core.cpu.regs[2])
        };
        assert_eq!(get(&mut m, 0x1234).0, 0xe401);
        let (result, _, mask) = get(&mut m, 0xffff8000);
        assert_eq!(result, 0);
        assert_ne!(mask, 0);
        m.core.cpu.regs[..3].copy_from_slice(&[0xffff8000, 0, 1]);
        m.svc(0x0f).unwrap();
        assert_eq!(m.core.cpu.regs[0], 0);
        assert_eq!(get(&mut m, 0xffff8000), (0, 0, 1));
    }

    #[test]
    fn gpu_memory_attribute_marks_and_clears_sharable_pages() {
        let mut m = mutex_test_machine();
        let svc = |m: &mut Machine, address: u64, size: u64, mask: u64, attr: u64| {
            m.core.cpu.regs[..4].copy_from_slice(&[address, size, mask, attr]);
            m.svc(0x80).unwrap();
            m.core.cpu.regs[0]
        };
        assert_eq!(svc(&mut m, 0xa001, PAGE_SIZE, 0x20, 0x20), 0xcc01);
        assert_eq!(svc(&mut m, 0xa000, 0, 0x20, 0x20), 0xca01);
        assert_eq!(svc(&mut m, 0xa000, PAGE_SIZE, 0x8, 0x8), 0xe801);
        assert_eq!(svc(&mut m, 0xa000, PAGE_SIZE, 0x20, 0x21), 0xe801);
        assert_eq!(svc(&mut m, 0x7000_0000, PAGE_SIZE, 0x20, 0x20), 0xd401);
        assert!(!m.services.is_gpu_sharable(0xa000, PAGE_SIZE));
        assert_eq!(svc(&mut m, 0xa000, PAGE_SIZE, 0x20, 0x20), 0);
        assert!(m.services.is_gpu_sharable(0xa000, PAGE_SIZE));
        assert!(!m.services.is_gpu_sharable(0xa000, 2 * PAGE_SIZE));
        assert_eq!(svc(&mut m, 0xa000, PAGE_SIZE, 0x20, 0), 0);
        assert!(!m.services.is_gpu_sharable(0xa000, PAGE_SIZE));
    }

    #[test]
    fn set_thread_priority_checks_range_grant_then_handle() {
        let mut m = thread_machine();
        let base = m.main_thread_base_priority;
        // Range and grant come before the handle lookup.
        for priority in [64, u64::from(u32::MAX), 5] {
            assert_eq!(set_priority(&mut m, 0x1234, priority), 0xe001);
        }
        assert_eq!(set_priority(&mut m, 0x1234, 40), 0xe401);
        assert_eq!(m.main_thread_base_priority, base);
        m.core.cpu.regs[1] = 0x9999;
        assert_eq!(set_priority(&mut m, 0xffff8000, 40), 0);
        assert_eq!(m.core.cpu.regs[1], 40); // W1 is input only.
        assert_eq!(m.main_thread_base_priority, 40);
        assert_eq!(m.main_thread_effective_priority, 40);
        m.core.cpu.regs[1] = 0xffff8000;
        m.svc(0x0c).unwrap();
        assert_eq!(m.core.cpu.regs[..2], [0, 40]);
        let report = m.report(&Trap::MissingBlock { pc: 0 });
        assert_eq!(
            report["thread_model"]["main_thread_record"]["base_priority"],
            40
        );
        assert_eq!(
            report["thread_model"]["main_thread_record"]["creation_priority"],
            base
        );
    }

    #[test]
    fn set_thread_priority_keeps_donation_and_restores_the_owner_chain() {
        let mut m = mutex_test_machine();
        let child = mutex_test_child(&mut m, 30, true);
        m.memory.write_u32(0xa000, 0x40000100).unwrap();
        m.finish_block().unwrap();
        mutex_test_lock(&mut m, 0x100, 0xa000, child);
        assert_eq!(m.main_thread_effective_priority, 30);
        // A lower base on the owner keeps the waiter's donation.
        assert_eq!(set_priority(&mut m, 0x100, 50), 0);
        assert_eq!(m.main_thread_base_priority, 50);
        assert_eq!(m.main_thread_effective_priority, 30);
        // Raising the waiter propagates to its owner.
        assert_eq!(set_priority(&mut m, u64::from(child), 29), 0);
        assert_eq!(m.threads.base_priority(2), Some(29));
        assert_eq!(m.threads.priority(2), Some(29));
        assert_eq!(m.main_thread_effective_priority, 29);
        // Lowering it again restores the owner to the waiter's new priority.
        assert_eq!(set_priority(&mut m, u64::from(child), 45), 0);
        assert_eq!(m.main_thread_effective_priority, 45);
        assert_eq!(set_priority(&mut m, u64::from(child), 55), 0);
        assert_eq!(m.main_thread_effective_priority, 50);
    }

    #[test]
    fn mixed_wait_kinds_keep_scoped_counts_and_mutex_references_when_no_context_can_run() {
        let mut m = mutex_test_machine();
        let child = mutex_test_child(&mut m, 28, true);
        m.memory.write_u32(0xa000, 0x40000100).unwrap();
        m.finish_block().unwrap();
        mutex_test_lock(&mut m, 0x100, 0xa000, child);
        zero_handle_wait_args(&mut m, -1);
        m.svc(0x18).unwrap();
        m.core.cpu.pc += 4;
        let stop = m.finish_block().unwrap_err();
        assert!(matches!(
            stop,
            Trap::NoRunnableThreads { waiting_threads: 2 }
        ));
        let report = m.report(&stop);
        assert_eq!(report["cpu_model"]["executing_guest_threads"], 0);
        assert_eq!(report["cpu_model"]["waiting_guest_threads"], 2);
        assert_eq!(report["cpu_model"]["started_guest_threads"], 2);
        assert_eq!(
            report["thread_model"]["mutex_waits"]["waiting_context_count"],
            1
        );
        assert_eq!(
            report["thread_model"]["synchronization_waits"]["waiting_context_count"],
            1
        );
        assert_eq!(m.main_mutex_owner_references, 1);
        assert_eq!(m.main_thread_effective_priority, 28);
        assert_eq!(m.memory.read_u32(0xa000).unwrap(), 0x40000100);
        assert!(!m.process_exit_requested);
    }

    #[test]
    fn closed_zero_handle_waiter_survives_last_runnable_exit_without_fake_process_exit() {
        let mut m = mutex_test_machine();
        let child = mutex_test_child(&mut m, 28, true);
        m.finish_block().unwrap();
        zero_handle_wait_args(&mut m, -1);
        m.svc(0x18).unwrap();
        m.core.cpu.pc += 4;
        m.finish_block().unwrap();
        assert_eq!(m.current_thread_id(), 1);
        m.core.cpu.regs[0] = u64::from(child);
        m.svc(0x16).unwrap();
        m.core.cpu.regs[0] = u64::from(child);
        m.svc(0x19).unwrap();
        assert_eq!(m.core.cpu.regs[0], 0xe401);
        assert_eq!(m.threads.state(2), Some(ThreadState::Waiting));
        assert_eq!(m.threads.wait_cancelled(2), Some(false));
        assert_eq!(
            m.memory
                .read_u32(m.layout.tls_base + PAGE_SIZE + 0x110)
                .unwrap(),
            child
        );
        m.core.cpu.regs[0] = m.layout.main_thread_handle;
        m.svc(0x16).unwrap();
        m.svc(0xa).unwrap();
        m.core.cpu.pc += 4;
        let stop = m.finish_block().unwrap_err();
        assert!(matches!(
            stop,
            Trap::NoRunnableThreads { waiting_threads: 1 }
        ));
        assert!(!m.main_thread_alive());
        assert!(!m.process_exit_requested);
        assert_eq!(
            m.memory.query(m.layout.tls_base).state,
            crate::memory::STATE_FREE
        );
        assert_eq!(
            m.memory.query(m.layout.tls_base + PAGE_SIZE).state,
            STATE_TLS
        );
        assert_eq!(m.synchronization_waits.waiting_count(), 1);
        assert_eq!(m.synchronization_waits_cancelled, 0);
        let report = m.report(&stop);
        assert_eq!(report["cpu_model"]["guest_threads"], 1);
        assert_eq!(report["cpu_model"]["executing_guest_threads"], 0);
        assert_eq!(report["cpu_model"]["started_guest_threads"], 1);
    }

    #[test]
    fn outgoing_cleanup_failure_does_not_retire_incoming_synchronization_return() {
        let mut m = mutex_test_machine();
        let child = mutex_test_child(&mut m, 28, true);
        m.finish_block().unwrap();
        zero_handle_wait_args(&mut m, -1);
        m.svc(0x18).unwrap();
        m.core.cpu.pc += 4;
        m.finish_block().unwrap();
        m.core.cpu.regs[0] = u64::from(child);
        m.svc(0x19).unwrap();
        m.core.cpu.regs[0] = m.layout.main_thread_handle;
        m.svc(0x16).unwrap();
        m.svc(0xa).unwrap();
        m.core.cpu.pc += 4;
        m.memory.unmap_exact(m.layout.tls_base, PAGE_SIZE).unwrap();
        m.memory
            .map(
                m.layout.tls_base,
                PAGE_SIZE,
                READ | WRITE,
                STATE_NORMAL,
                "wrong outgoing TLS",
            )
            .unwrap();
        let cpu = m.core.cpu.clone();
        let before = m.thread_model_report();
        assert!(matches!(m.finish_block(), Err(Trap::InvalidState { .. })));
        assert_eq!(m.core.cpu, cpu);
        assert_eq!(m.current_thread_id(), 1);
        assert_eq!(m.pending_thread_exit, Some(1));
        assert!(m.synchronization_waits.pending_return(2).is_some());
        assert_eq!(m.synchronization_waits_returned, 0);
        let mut after = m.thread_model_report();
        after["scheduler"]["completed_blocks"] = before["scheduler"]["completed_blocks"].clone();
        assert_eq!(after, before);
    }

    #[test]
    fn arbitrate_lock_checks_address_read_and_real_owner_in_reference_order() {
        // A full-width address plus high garbage in W-register inputs catches
        // accidental X1 truncation and X0/X2 high-half validation.
        const ADDRESS: u64 = 0x1_0000_a000;
        for (address, permissions, owner, word, expected) in [
            (ADDRESS + 1, READ, 0xdeadu32, 0u32, 0xcc01),
            (KERNEL_VIRTUAL_ADDRESS_BASE + 1, READ, 0xdead, 0, 0xd401),
            (KERNEL_VIRTUAL_ADDRESS_END + 1, READ, 0xdead, 0, 0xcc01),
            (KERNEL_VIRTUAL_ADDRESS_END, READ, 0xdead, 0, 0xd401),
            ((1 << 39) + 1, READ, 0xdead, 0, 0xcc01),
            (1 << 39, READ, 0xdead, 0, 0xd401),
            (u64::MAX, READ, 0xdead, 0, 0xcc01),
            (ADDRESS + PAGE_SIZE, READ, 0xdead, 0, 0xd401),
            (ADDRESS, WRITE, 0xdead, 0, 0xd401),
            (ADDRESS, READ, 0xdead, 0, 0),
            (ADDRESS, READ, 0xdead, 0x4000dead, 0xe401),
            (ADDRESS, READ, 0xffff8000, 0xffff8000, 0xe401),
            (ADDRESS, READ, 0xffff8001, 0xffff8001, 0xe401),
        ] {
            let mut m = thread_machine();
            m.memory
                .map(ADDRESS, PAGE_SIZE, permissions, STATE_NORMAL, "ABI mutex")
                .unwrap();
            m.memory.initialize(ADDRESS, &word.to_le_bytes()).unwrap();
            m.core.cpu.regs[..4].copy_from_slice(&[
                0xabcdef1200000000 | u64::from(owner),
                address,
                0x87654321f0000005,
                u64::MAX,
            ]);
            let mut expected_cpu = m.core.cpu.clone();
            expected_cpu.regs[0] = expected;
            let before = m.thread_model_report();
            m.svc(0x1a).unwrap();
            assert_eq!(
                m.core.cpu, expected_cpu,
                "address={address:#x}, owner={owner:#x}"
            );
            assert_eq!(m.mutex_waits.waiting_count(), 0);
            assert_eq!(m.main_mutex_owner_references, 0);
            assert_eq!(m.pending_mutex_wait, None);
            let mut after = m.thread_model_report();
            after["mutex_waits"]["raced_lock_returns"] =
                before["mutex_waits"]["raced_lock_returns"].clone();
            assert_eq!(after, before);
        }
        let mut m = mutex_test_machine();
        let session = m.services.connect_named_port("sm:", 0x1000).unwrap();
        m.memory
            .write_u32(0xa000, session | MUTEX_WAIT_MASK)
            .unwrap();
        m.core.cpu.regs[..3].copy_from_slice(&[u64::from(session), 0xa000, 0]);
        m.svc(0x1a).unwrap();
        assert_eq!(m.core.cpu.regs[0], 0xe401); // Valid session, wrong object type.
        m.layout.main_thread_handle = 0x40000100;
        for (word, expected) in [(0x40000100, 0xe401), (0, 0)] {
            m.memory.write_u32(0xa000, word).unwrap();
            m.core.cpu.regs[..3].copy_from_slice(&[m.layout.main_thread_handle, 0xa000, 0]);
            m.svc(0x1a).unwrap();
            assert_eq!(m.core.cpu.regs[0], expected); // Reserved bits checked only after matching.
        }
        m.pending_thread_exit = Some(m.current_thread_id());
        for (address, expected) in [(0xb001, 0xcc01), (0xb000, 0x7601)] {
            m.core.cpu.regs[..3].copy_from_slice(&[0xdead, address, 0]);
            m.svc(0x1a).unwrap();
            assert_eq!(m.core.cpu.regs[0], expected);
        }
    }

    #[test]
    fn mutex_wait_saves_post_svc_context_and_retains_closed_handles_until_return() {
        let mut m = mutex_test_machine();
        let handle = mutex_test_child(&mut m, 28, true);
        let main_handle = m.layout.main_thread_handle as u32;
        m.memory
            .write_u32(0xa000, main_handle | MUTEX_WAIT_MASK)
            .unwrap();
        m.core.cpu.pc = 0x1004;
        let main_cpu = m.core.cpu.clone();
        m.finish_block().unwrap();
        assert_eq!(m.current_thread_id(), 2);
        m.core.cpu.regs = std::array::from_fn(|index| 0x1234567800000000 + index as u64);
        m.core.cpu.vectors = std::array::from_fn(|index| u128::MAX - index as u128);
        m.core.cpu.nzcv = 0xd;
        m.core.cpu.tpidr_el0 = 0x99887766;
        m.core.cpu.fpcr = 0x00800000;
        m.core.cpu.fpsr = 0x10;
        m.core.cpu.regs[..3].copy_from_slice(&[
            0xabcdef1200000000 | u64::from(main_handle),
            0xa000,
            0x87654321a0000005,
        ]);
        let before = m.core.cpu.clone();
        m.svc(0x1a).unwrap();
        assert_eq!(m.core.cpu, before); // No provisional success or premature swap.
        assert_eq!(m.current_thread_id(), 2);
        assert_eq!(m.threads.state(2), Some(ThreadState::Runnable));
        assert_eq!(m.pending_mutex_wait, Some(2));
        assert!(matches!(m.tick(), Err(Trap::InvalidState { .. })));
        m.core.cpu.pc += 4;
        let mut resumed = m.core.cpu.clone();
        resumed.regs[0] = 0;
        m.finish_block().unwrap();
        assert_eq!(m.current_thread_id(), 1);
        assert_eq!(m.core.cpu, main_cpu);
        assert_eq!(m.threads.state(2), Some(ThreadState::Waiting));
        assert_eq!(m.main_thread_effective_priority, 28);
        assert_eq!(m.main_mutex_owner_references, 1);
        assert_eq!(
            m.memory.read_u32(0xa000).unwrap(),
            main_handle | MUTEX_WAIT_MASK
        );
        m.core.cpu.regs[1] = u64::from(main_handle);
        m.svc(0x0c).unwrap();
        assert_eq!(m.core.cpu.regs[1], 28);
        for closed in [handle, main_handle] {
            m.core.cpu.regs[0] = u64::from(closed);
            m.svc(0x16).unwrap();
            assert_eq!(m.core.cpu.regs[0], 0);
        }
        assert_eq!(
            m.memory
                .read_u32(m.layout.tls_base + PAGE_SIZE + 0x110)
                .unwrap(),
            handle
        );
        assert_eq!(m.memory.query(m.layout.tls_base).state, STATE_TLS);
        let before_exit = m.thread_model_report();
        assert!(matches!(m.svc(0xa), Err(Trap::InvalidState { .. })));
        assert_eq!(m.thread_model_report(), before_exit);
        m.core.cpu.regs[0] = 0xa000;
        m.svc(0x1b).unwrap();
        assert_eq!(m.core.cpu.regs[0], 0);
        assert_eq!(m.memory.read_u32(0xa000).unwrap(), 0xa0000005);
        assert_eq!(m.threads.state(2), Some(ThreadState::Runnable));
        assert_eq!(m.main_thread_effective_priority, 44);
        assert_eq!(m.main_mutex_owner_references, 1); // Not released at unlock.
        m.core.cpu.pc += 4;
        let main_after_unlock = m.core.cpu.clone();
        m.finish_block().unwrap();
        assert_eq!(m.current_thread_id(), 2);
        assert_eq!(m.core.cpu, resumed);
        assert_eq!(m.main_mutex_owner_references, 0);
        assert!(m.mutex_waits.pending_return(2).is_none());
        assert_eq!(m.mutex_waits_returned, 1);
        m.svc(0xa).unwrap();
        m.core.cpu.pc += 4;
        m.finish_block().unwrap();
        assert_eq!(m.core.cpu, main_after_unlock);
        assert!(!m.threads.contains(2));
        assert_eq!(
            m.memory.query(m.layout.tls_base + PAGE_SIZE).state,
            crate::memory::STATE_FREE
        );
        m.svc(0xa).unwrap();
        m.core.cpu.pc += 4;
        assert!(matches!(m.finish_block(), Err(Trap::Exit { code: 0 })));
        assert_eq!(
            m.memory.query(m.layout.tls_base).state,
            crate::memory::STATE_FREE
        );
    }

    #[test]
    fn unlock_uses_write_permission_without_read_or_word_ownership_validation() {
        for permissions in [WRITE, READ] {
            let mut m = thread_machine();
            m.memory
                .map(
                    0xa000,
                    PAGE_SIZE,
                    permissions,
                    STATE_NORMAL,
                    "permission mutex",
                )
                .unwrap();
            m.memory
                .initialize(0xa000, &0xdeadbeefu32.to_le_bytes())
                .unwrap();
            m.core.cpu.regs[0] = 0xa000;
            let mut expected = m.core.cpu.clone();
            expected.regs[0] = if permissions == WRITE { 0 } else { 0xd401 };
            m.svc(0x1b).unwrap();
            assert_eq!(m.core.cpu, expected);
            if permissions == WRITE {
                m.memory.set_permissions(0xa000, PAGE_SIZE, READ).unwrap();
                assert_eq!(m.memory.read_u32(0xa000).unwrap(), 0);
            } else {
                assert_eq!(m.memory.read_u32(0xa000).unwrap(), 0xdeadbeef);
            }
        }
        // An unrelated virtual address cannot select a waiter's queue, even
        // though unlock still writes zero there without comparing owner bits.
        let mut m = mutex_test_machine();
        mutex_test_child(&mut m, 28, true);
        m.memory.write_u32(0xa000, 0x40000100).unwrap();
        m.finish_block().unwrap();
        mutex_test_lock(&mut m, 0x100, 0xa000, 0);
        m.core.cpu.regs[0] = 0xa004;
        m.svc(0x1b).unwrap();
        assert_eq!(m.threads.state(2), Some(ThreadState::Waiting));
        assert_eq!(m.memory.read_u32(0xa000).unwrap(), 0x40000100);
        m.core.cpu.regs[0] = 0xa000;
        m.svc(0x1b).unwrap();
        assert_eq!(m.memory.read_u32(0xa000).unwrap(), 0); // Raw zero tag is legal.
        assert_eq!(m.threads.state(2), Some(ThreadState::Runnable));
    }

    #[test]
    fn unlock_failed_store_wakes_with_error_but_prototype_write_trap_preserves_queue() {
        for protect_translation in [false, true] {
            let mut m = mutex_test_machine();
            mutex_test_child(&mut m, 28, true);
            m.memory.write_u32(0xa000, 0x40000100).unwrap();
            if protect_translation {
                m.memory.protect_translated_code(0xa000, PAGE_SIZE).unwrap();
            } else {
                m.memory.set_permissions(0xa000, PAGE_SIZE, READ).unwrap();
            }
            m.finish_block().unwrap();
            mutex_test_lock(&mut m, 0x100, 0xa000, 0x200);
            assert_eq!(m.threads.state(2), Some(ThreadState::Waiting));
            m.core.cpu.regs[0] = 0xa000;
            let before_cpu = m.core.cpu.clone();
            let before_model = m.thread_model_report();
            let result = m.svc(0x1b);
            if protect_translation {
                assert!(matches!(result, Err(Trap::InvalidState { .. })));
                assert_eq!(m.core.cpu, before_cpu);
                assert_eq!(m.thread_model_report(), before_model);
            } else {
                result.unwrap();
                assert_eq!(m.core.cpu.regs[0], 0xd401);
                assert_eq!(m.threads.state(2), Some(ThreadState::Runnable));
                assert_eq!(m.threads.saved_cpu(2).unwrap().regs[0], 0xd401);
                assert_eq!(m.main_thread_effective_priority, 44);
                assert_eq!(m.main_mutex_owner_references, 1);
                assert_eq!(m.mutex_waits.waiting_count(), 0);
                m.core.cpu.pc += 4;
                m.finish_block().unwrap();
                assert_eq!(m.current_thread_id(), 2);
                assert_eq!(m.core.cpu.regs[0], 0xd401);
                assert_eq!(m.main_mutex_owner_references, 0);
            }
            assert_eq!(m.memory.read_u32(0xa000).unwrap(), 0x40000100);
        }
    }

    #[test]
    fn last_runnable_exit_keeps_waiting_and_reference_retained_initialized_objects_alive() {
        let mut m = mutex_test_machine();
        let dormant = mutex_test_child(&mut m, 28, false);
        mutex_test_child(&mut m, 30, true);
        m.memory
            .write_u32(0xa000, dormant | MUTEX_WAIT_MASK)
            .unwrap();
        mutex_test_lock(&mut m, dormant, 0xa000, 0x100);
        assert_eq!(m.current_thread_id(), 3);
        assert_eq!(m.main_thread_state, ThreadState::Waiting);
        for handle in [dormant, m.layout.main_thread_handle as u32] {
            m.core.cpu.regs[0] = u64::from(handle);
            m.svc(0x16).unwrap();
        }
        assert_eq!(m.threads.mutex_owner_references(2), Some(1));
        assert_eq!(
            m.memory.query(m.layout.tls_base + PAGE_SIZE).state,
            STATE_TLS
        );
        m.svc(0xa).unwrap();
        m.core.cpu.pc += 4;
        let stop = m.finish_block().unwrap_err();
        assert!(matches!(
            stop,
            Trap::NoRunnableThreads { waiting_threads: 1 }
        ));
        assert!(!m.process_exit_requested);
        assert_eq!(m.pending_thread_exit, None);
        assert_eq!(m.thread_exits, 1);
        assert_eq!(m.main_thread_state, ThreadState::Waiting);
        assert_eq!(m.threads.state(2), Some(ThreadState::Initialized));
        assert_eq!(m.threads.state(3), Some(ThreadState::Terminated));
        let report = m.report(&stop);
        assert_eq!(report["cpu_model"]["executing_guest_threads"], 0);
        assert_eq!(report["cpu_model"]["waiting_guest_threads"], 1);
        assert_eq!(report["cpu_model"]["started_guest_threads"], 1);
        assert_eq!(report["cpu_model"]["guest_threads"], 3);
        assert_eq!(
            report["thread_model"]["main_thread_record"]["running_reference"],
            true
        );
        assert_eq!(
            report["thread_model"]["main_thread_record"]["signaled"],
            false
        );
        assert!(matches!(
            m.tick(),
            Err(Trap::NoRunnableThreads { waiting_threads: 1 })
        ));
    }

    #[test]
    fn mutex_handoff_retains_original_owner_through_exit_and_failed_final_cleanup() {
        for corrupt_main_tls in [false, true] {
            let mut m = mutex_test_machine();
            let first = mutex_test_child(&mut m, 28, true);
            let second = mutex_test_child(&mut m, 30, true);
            m.memory.write_u32(0xa000, 0x40000100).unwrap();
            m.finish_block().unwrap();
            mutex_test_lock(&mut m, 0x100, 0xa000, first);
            assert_eq!(m.current_thread_id(), 3);
            mutex_test_lock(&mut m, 0x100, 0xa000, second);
            assert_eq!(m.current_thread_id(), 1);
            assert_eq!(m.main_mutex_owner_references, 2);
            m.core.cpu.regs[0] = m.layout.main_thread_handle;
            m.svc(0x16).unwrap();
            m.core.cpu.regs[0] = 0xa000;
            m.svc(0x1b).unwrap();
            assert_eq!(m.memory.read_u32(0xa000).unwrap(), first | MUTEX_WAIT_MASK);
            assert!(!m.mutex_waits.has_waiters(1));
            assert!(m.mutex_waits.has_waiters(2));
            assert_eq!(m.main_mutex_owner_references, 2);
            m.svc(0xa).unwrap(); // Queue moved away; ordinary exit is now supported.
            m.core.cpu.pc += 4;
            m.finish_block().unwrap();
            assert_eq!(m.current_thread_id(), 2);
            assert_eq!(m.main_thread_state, ThreadState::Terminated);
            assert_eq!(m.main_mutex_owner_references, 1);
            assert!(m.main_thread_alive());
            assert_eq!(m.memory.query(m.layout.tls_base).state, STATE_TLS);
            assert!(matches!(m.svc(0xa), Err(Trap::InvalidState { .. })));
            m.core.cpu.regs[0] = 0xa000;
            m.svc(0x1b).unwrap();
            assert_eq!(m.memory.read_u32(0xa000).unwrap(), second);
            assert_eq!(m.mutex_waits.pending_return(3).unwrap().original_owner, 1);
            assert_eq!(m.main_mutex_owner_references, 1);
            if corrupt_main_tls {
                m.memory.unmap_exact(m.layout.tls_base, PAGE_SIZE).unwrap();
                m.memory
                    .map(
                        m.layout.tls_base,
                        PAGE_SIZE,
                        READ | WRITE,
                        STATE_NORMAL,
                        "wrong TLS owner",
                    )
                    .unwrap();
            }
            m.core.cpu.pc += 4;
            let before_cpu = m.core.cpu.clone();
            let switches = m.context_switches;
            if corrupt_main_tls {
                assert!(matches!(m.finish_block(), Err(Trap::InvalidState { .. })));
                assert_eq!(m.core.cpu, before_cpu);
                assert_eq!(m.current_thread_id(), 2);
                assert_eq!(m.context_switches, switches);
                assert_eq!(m.main_mutex_owner_references, 1);
                assert!(m.mutex_waits.pending_return(3).is_some());
                assert!(m.main_thread_alive());
            } else {
                m.finish_block().unwrap();
                assert_eq!(m.current_thread_id(), 3);
                assert_eq!(m.core.cpu.regs[0], 0);
                assert_eq!(m.main_mutex_owner_references, 0);
                assert!(!m.main_thread_alive());
                assert_eq!(
                    m.memory.query(m.layout.tls_base).state,
                    crate::memory::STATE_FREE
                );
                assert!(m.mutex_waits.pending_return(3).is_none());
            }
        }
    }

    #[test]
    fn unsupported_self_and_cyclic_mutex_waits_do_not_publish_or_retain() {
        for cycle in [false, true] {
            let mut m = mutex_test_machine();
            m.memory.write_u32(0xa000, 0x40000100).unwrap();
            if cycle {
                let child = mutex_test_child(&mut m, 28, true);
                m.memory.write_u32(0xa004, child | MUTEX_WAIT_MASK).unwrap();
                mutex_test_lock(&mut m, child, 0xa004, 0x100);
                assert_eq!(m.current_thread_id(), 2);
            }
            m.core.cpu.regs[..3].copy_from_slice(&[0x100, 0xa000, 0xdead]);
            let cpu = m.core.cpu.clone();
            let model = m.thread_model_report();
            assert!(matches!(m.svc(0x1a), Err(Trap::InvalidState { .. })));
            assert_eq!(m.core.cpu, cpu);
            assert_eq!(m.thread_model_report(), model);
            assert_eq!(m.pending_mutex_wait, None);
            assert_eq!(m.main_mutex_owner_references, 0);
        }
    }

    #[test]
    fn outgoing_exit_cleanup_failure_preserves_incoming_mutex_return_reference() {
        for main_owner in [false, true] {
            let mut m = mutex_test_machine();
            let child = mutex_test_child(&mut m, 28, true);
            let (owner, waiter, owner_handle, tls) = if main_owner {
                m.memory.write_u32(0xa000, 0x40000100).unwrap();
                m.finish_block().unwrap();
                mutex_test_lock(&mut m, 0x100, 0xa000, child);
                (1, 2, 0x100, m.layout.tls_base)
            } else {
                m.memory.write_u32(0xa000, child | MUTEX_WAIT_MASK).unwrap();
                mutex_test_lock(&mut m, child, 0xa000, 0x100);
                (2, 1, child, m.layout.tls_base + PAGE_SIZE)
            };
            assert_eq!(m.current_thread_id(), owner);
            m.core.cpu.regs[0] = u64::from(owner_handle);
            m.svc(0x16).unwrap();
            m.core.cpu.regs[0] = 0xa000;
            m.svc(0x1b).unwrap();
            assert_eq!(
                m.mutex_waits.pending_return(waiter).unwrap().original_owner,
                owner
            );
            m.svc(0xa).unwrap();
            m.core.cpu.pc += 4;
            m.memory.unmap_exact(tls, PAGE_SIZE).unwrap();
            m.memory
                .map(
                    tls,
                    PAGE_SIZE,
                    READ | WRITE,
                    STATE_NORMAL,
                    "invalid outgoing TLS",
                )
                .unwrap();
            let cpu = m.core.cpu.clone();
            let before = m.thread_model_report();
            assert!(matches!(m.finish_block(), Err(Trap::InvalidState { .. })));
            assert_eq!(m.core.cpu, cpu);
            assert_eq!(m.current_thread_id(), owner);
            assert_eq!(m.thread_state(owner), Some(ThreadState::Runnable));
            assert_eq!(m.pending_thread_exit, Some(owner));
            assert_eq!(
                m.mutex_waits.pending_return(waiter).unwrap().result,
                Some(0)
            );
            let mut after = m.thread_model_report();
            after["scheduler"]["completed_blocks"] =
                before["scheduler"]["completed_blocks"].clone();
            assert_eq!(after, before);
        }
    }

    #[test]
    fn create_thread_prepares_independent_context_tls_and_raw_output_without_running() {
        let mut m = thread_machine();
        create_thread_args(&mut m, 0xabcdef120000001c, 0xabcdef12fffffffe);
        let before = m.core.cpu.clone();
        let resident = m.memory.resident_bytes();
        m.svc(0x08).unwrap();
        let handle = m.core.cpu.regs[1] as u32;
        assert_eq!(handle, 0x200);
        let mut expected = before.clone();
        expected.regs[0] = 0;
        expected.regs[1] = u64::from(handle);
        assert_eq!(m.core.cpu, expected);
        assert_eq!(m.core.steps, 0);
        assert_eq!(m.services.thread_id(handle), Some(2));
        assert_eq!(m.threads.priority(2), Some(28));
        assert!(!m.services.contains(handle));
        let tls = m.layout.tls_base + PAGE_SIZE;
        let mapping = m.memory.query(tls);
        assert_eq!(
            (mapping.state, mapping.permissions, mapping.attributes),
            (STATE_TLS, READ | WRITE, 0)
        );
        let mut bytes = [0xff; 0x200];
        let mut expected_tls = [0; 0x200];
        expected_tls[0x110..0x114].copy_from_slice(&handle.to_le_bytes());
        m.memory.read(tls, &mut bytes).unwrap();
        assert_eq!(bytes, expected_tls);
        assert_eq!(m.memory.resident_bytes(), resident + PAGE_SIZE);
        assert_eq!(
            m.memory.read_u64(m.layout.tls_base + 0x120).unwrap(),
            0xfeedface
        );
        assert_eq!(
            m.memory.read_u64(m.layout.stack_region_base).unwrap(),
            0x0123456789abcdef
        );
        assert_eq!(m.svc_trace[0].arguments, before.regs[..4]);
        assert_eq!(
            m.svc_trace[0].arguments_4_5,
            Some([before.regs[4], before.regs[5]])
        );
        let report = m.report(&Trap::Exit { code: 0 });
        let thread = &report["thread_model"]["threads"][0];
        let context = &thread["initial_cpu_context"];
        assert_eq!(context["pc"], 0x1234566);
        assert_eq!(context["sp"], before.regs[3] & !15);
        assert_eq!(context["registers"][0], before.regs[2]);
        assert_eq!(context["registers"][18].as_u64().unwrap() & 1, 1);
        assert_eq!(context["fpcr"], before.fpcr);
        assert_eq!(context["fpsr"], before.fpsr);
        assert_eq!(context["tpidr_el0"], 0);
        assert_eq!(context["tpidrro_el0"], tls);
        assert_eq!(report["cpu_model"]["guest_threads"], 2);
        assert_eq!(report["cpu_model"]["executing_guest_threads"], 1);
        for kind in [7, 22] {
            m.core.cpu.regs[..4].copy_from_slice(&[0xdead, kind, 0xffff8001, 0]);
            m.svc(0x29).unwrap();
            assert_eq!((m.core.cpu.regs[0], m.core.cpu.regs[1]), (0, 0x4000));
        }
        assert_eq!(m.threads.state(2), Some(ThreadState::Initialized));
        assert_eq!(m.threads.runnable_count(), 0);
    }

    #[test]
    fn start_thread_returns_only_w0_and_checks_handle_termination_then_state() {
        let mut m = thread_machine();
        create_thread_args(&mut m, 28, 5);
        m.svc(8).unwrap();
        let handle = m.core.cpu.regs[1];
        let session = u64::from(m.services.connect_named_port("sm:", 0).unwrap());
        for (target, result) in [
            (0xdead, 0xe401),
            (session, 0xe401),
            (0xffff8001, 0xe401),
            (m.layout.main_thread_handle, 0xfa01),
            (0xffff8000, 0xfa01),
        ] {
            m.core.cpu.regs[0] = 0xabcdef1200000000 | target;
            let mut expected = m.core.cpu.clone();
            expected.regs[0] = result;
            let threads = m.threads.report();
            m.svc(9).unwrap();
            assert_eq!(m.core.cpu, expected);
            assert_eq!(m.threads.report(), threads);
        }
        m.core.cpu.regs[0] = 0xabcdef1200000000 | handle;
        let mut expected = m.core.cpu.clone();
        expected.regs[0] = 0;
        m.svc(9).unwrap();
        assert_eq!(m.core.cpu, expected);
        assert_eq!(m.current_thread_id(), 1);
        assert_eq!(m.threads.state(2), Some(ThreadState::Runnable));
        assert_eq!(m.context_switches, 0);
        assert_eq!(m.core.steps, 0);
        m.core.cpu.regs[0] = handle;
        m.svc(9).unwrap();
        assert_eq!(m.core.cpu.regs[0], 0xfa01);
        let child = m.threads.saved_cpu(2).unwrap();
        m.threads.terminate(&mut m.memory, 2, &child).unwrap();
        m.core.cpu.regs[0] = handle;
        m.svc(9).unwrap();
        assert_eq!(m.core.cpu.regs[0], 0x7601);
        m.core.cpu.regs[0] = handle;
        m.svc(0x16).unwrap();
        m.core.cpu.regs[0] = handle;
        m.svc(9).unwrap();
        assert_eq!(m.core.cpu.regs[0], 0xe401);
        create_thread_args(&mut m, 28, 5);
        m.svc(8).unwrap();
        let unstarted = m.core.cpu.regs[1];
        m.svc(0x0a).unwrap();
        m.core.cpu.regs[0] = unstarted;
        m.svc(9).unwrap();
        assert_eq!(m.core.cpu.regs[0], 0x7601);
        assert_eq!(m.threads.state(3), Some(ThreadState::Initialized));
    }

    #[test]
    fn block_dispatch_preserves_full_context_and_current_thread_identity() {
        let mut m = thread_machine();
        create_thread_args(&mut m, 28, 5);
        m.svc(8).unwrap();
        let handle = m.core.cpu.regs[1];
        m.core.cpu.regs[0] = handle;
        m.svc(9).unwrap();
        let initial_child = m.threads.saved_cpu(2).unwrap();
        // Reproduce the generated SVC epilogue before invoking the hook.
        m.core.cpu.pc = 0x1004;
        let parent = m.core.cpu.clone();
        let counter = m.physical_counter_ticks();
        m.finish_block().unwrap();
        assert_eq!(m.current_thread_id(), 2);
        assert_eq!(m.core.cpu, initial_child);
        assert_eq!(m.main_saved_cpu.as_ref(), Some(&parent));
        assert!(m.physical_counter_ticks() >= counter);
        for (number, result) in [(0x0c, 28), (0x25, 2)] {
            m.core.cpu.regs[1] = 0xabcdef12ffff8000;
            m.svc(number).unwrap();
            assert_eq!((m.core.cpu.regs[0], m.core.cpu.regs[1]), (0, result));
            assert_eq!(m.svc_trace.last().unwrap().thread_id, 2);
        }
        m.svc(0x10).unwrap();
        assert_eq!(m.core.cpu.regs[0], 5);
        m.core.cpu.regs[0] = handle;
        m.svc(0x16).unwrap();
        assert_eq!(m.services.thread_id(handle as u32), None);
        assert_eq!(
            m.memory.read_u32(m.core.cpu.tpidrro_el0 + 0x110).unwrap(),
            handle as u32
        );
        m.core.cpu.regs[1] = 0xffff8000;
        m.svc(0x25).unwrap();
        assert_eq!(m.core.cpu.regs[1], 2);
        m.core.cpu.regs[..3].copy_from_slice(&[0xffff8000, 4, 0x30]);
        m.svc(0x0f).unwrap();
        assert_eq!(m.threads.affinity(2), Some((4, 0x30)));
        m.svc(0x10).unwrap();
        assert_eq!(m.core.cpu.regs[0], 4); // Migrated to its ideal core.
        m.core.cpu.regs[..3].copy_from_slice(&[0xffff8000, 4, 0x10]);
        m.svc(0x0f).unwrap();
        assert_eq!(m.threads.affinity(2), Some((4, 0x10)));
        m.core.cpu.regs = std::array::from_fn(|i| 0xaabbccdd00000000 + i as u64);
        m.core.cpu.vectors = std::array::from_fn(|i| u128::MAX - i as u128);
        m.core.cpu.pc = 0x2004;
        m.core.cpu.sp = 0x8ef0;
        m.core.cpu.nzcv = 6;
        m.core.cpu.tpidr_el0 = 0xfeedface;
        m.core.cpu.fpcr = 0x00800000;
        m.core.cpu.fpsr = 0x10;
        let child = m.core.cpu.clone();
        m.finish_block().unwrap();
        assert_eq!(m.current_thread_id(), 1);
        assert_eq!(m.core.cpu, parent);
        assert_eq!(m.threads.saved_cpu(2).unwrap(), child);
        m.finish_block().unwrap();
        assert_eq!(m.core.cpu, child);
        assert_eq!(m.current_thread_id(), 2);
        assert_eq!(m.core.steps, 0);
        assert_eq!(m.context_switches, 3);
        // Child CreateThread inherits the active FP state. -2 still selects
        // the process default rather than this child's mutable ideal core4.
        create_thread_args(&mut m, 28, 0xfffffffe);
        m.svc(8).unwrap();
        let report = m.threads.report();
        assert_eq!(report["threads"][1]["creation_core"], 5);
        assert_eq!(
            report["threads"][1]["initial_cpu_context"]["fpcr"],
            0x00800000
        );
        assert_eq!(report["threads"][1]["initial_cpu_context"]["fpsr"], 0x10);
        assert_eq!(
            report["threads"][0]["initial_cpu_context"]["fpcr"],
            initial_child.fpcr
        );
    }

    #[test]
    fn child_exit_releases_running_reference_and_preserves_retained_handles() {
        for close_before_exit in [false, true] {
            let mut m = thread_machine();
            create_thread_args(&mut m, 28, 5);
            m.svc(8).unwrap();
            let handle = m.core.cpu.regs[1];
            let tls = m.layout.tls_base + PAGE_SIZE;
            m.core.cpu.regs[0] = handle;
            m.svc(9).unwrap();
            if close_before_exit {
                m.core.cpu.regs[0] = handle;
                m.svc(0x16).unwrap();
                assert_eq!(m.threads.state(2), Some(ThreadState::Runnable));
                assert_eq!(m.memory.read_u32(tls + 0x110).unwrap(), handle as u32);
            }
            m.core.cpu.pc = 0x1004;
            let parent = m.core.cpu.clone();
            m.finish_block().unwrap();
            let before = m.core.cpu.clone();
            m.svc(0x0a).unwrap();
            assert_eq!(m.core.cpu, before);
            assert_eq!(m.current_thread_id(), 2);
            assert_eq!(m.threads.state(2), Some(ThreadState::Runnable));
            m.core.cpu.pc += 4;
            m.finish_block().unwrap();
            assert_eq!(m.current_thread_id(), 1);
            assert_eq!(m.core.cpu, parent);
            assert_eq!(m.thread_exits, 1);
            assert!(!m.process_exit_requested);
            if close_before_exit {
                assert!(!m.threads.contains(2));
                assert_eq!(m.memory.query(tls).state, crate::memory::STATE_FREE);
            } else {
                assert_eq!(m.threads.state(2), Some(ThreadState::Terminated));
                assert_eq!(m.threads.report()["threads"][0]["signaled"], true);
                assert_eq!(m.memory.query(tls).state, STATE_TLS);
                m.core.cpu.regs[1] = handle;
                m.svc(0x25).unwrap();
                assert_eq!((m.core.cpu.regs[0], m.core.cpu.regs[1]), (0, 2));
                m.core.cpu.regs[0] = handle;
                m.svc(9).unwrap();
                assert_eq!(m.core.cpu.regs[0], 0x7601);
                m.core.cpu.regs[0] = handle;
                m.svc(0x16).unwrap();
                assert!(!m.threads.contains(2));
                assert_eq!(m.memory.query(tls).state, crate::memory::STATE_FREE);
            }
            assert_eq!(m.memory.query(m.layout.tls_base).state, STATE_TLS);
            assert_eq!(m.memory.stack_aliases().count(), 1);
            assert_eq!(
                m.memory.read_u64(m.layout.stack_region_base).unwrap(),
                0x0123456789abcdef
            );
        }
    }

    #[test]
    fn main_exit_and_last_started_exit_do_not_wait_for_unstarted_objects() {
        for close_main_before_exit in [false, true] {
            let mut m = thread_machine();
            create_thread_args(&mut m, 28, 5);
            m.svc(8).unwrap();
            let child_handle = m.core.cpu.regs[1];
            create_thread_args(&mut m, 28, 5);
            m.svc(8).unwrap(); // This initialized object never starts.
            m.core.cpu.regs[0] = child_handle;
            m.svc(9).unwrap();
            m.core.cpu.pc = 0x1004;
            m.finish_block().unwrap();
            m.finish_block().unwrap();
            if close_main_before_exit {
                m.core.cpu.regs[0] = m.layout.main_thread_handle;
                m.svc(0x16).unwrap();
                assert_eq!(m.memory.query(m.layout.tls_base).state, STATE_TLS);
            }
            m.svc(0x0a).unwrap();
            m.core.cpu.pc += 4;
            m.finish_block().unwrap();
            assert_eq!(m.current_thread_id(), 2);
            assert_eq!(m.main_thread_state, ThreadState::Terminated);
            assert!(!m.process_exit_requested);
            if close_main_before_exit {
                assert_eq!(
                    m.memory.query(m.layout.tls_base).state,
                    crate::memory::STATE_FREE
                );
            } else {
                assert_eq!(m.memory.query(m.layout.tls_base).state, STATE_TLS);
                m.core.cpu.regs[1] = m.layout.main_thread_handle;
                m.svc(0x25).unwrap();
                assert_eq!(m.core.cpu.regs[1], 1);
                m.core.cpu.regs[0] = m.layout.main_thread_handle;
                m.svc(9).unwrap();
                assert_eq!(m.core.cpu.regs[0], 0x7601);
                m.core.cpu.regs[0] = m.layout.main_thread_handle;
                m.svc(0x16).unwrap();
                assert_eq!(
                    m.memory.query(m.layout.tls_base).state,
                    crate::memory::STATE_FREE
                );
            }
            m.svc(0x0a).unwrap();
            m.core.cpu.pc += 4;
            assert!(matches!(m.finish_block(), Err(Trap::Exit { code: 0 })));
            assert_eq!(m.process_exit_reason, Some("last_started_thread_exit"));
            assert_eq!(m.threads.state(3), Some(ThreadState::Initialized));
            assert_eq!(m.threads.state(2), Some(ThreadState::Terminated));
            assert_eq!(m.threads.runnable_count(), 0);
            assert!(matches!(m.tick(), Err(Trap::Exit { code: 0 })));
            assert_eq!(
                m.report(&Trap::Exit { code: 0 })["cpu_model"]["executing_guest_threads"],
                0
            );
            assert_eq!(m.memory.stack_aliases().count(), 1);
        }
    }

    #[test]
    fn failed_main_tls_cleanup_does_not_commit_a_thread_exit_or_context_switch() {
        let mut m = thread_machine();
        create_thread_args(&mut m, 28, 5);
        m.svc(8).unwrap();
        let handle = m.core.cpu.regs[1];
        m.core.cpu.regs[0] = handle;
        m.svc(9).unwrap();
        m.core.cpu.regs[0] = m.layout.main_thread_handle;
        m.svc(0x16).unwrap();
        m.memory.unmap_exact(m.layout.tls_base, PAGE_SIZE).unwrap();
        m.memory
            .map(
                m.layout.tls_base,
                PAGE_SIZE,
                READ | WRITE,
                STATE_NORMAL,
                "foreign replacement",
            )
            .unwrap();
        m.memory.write_u64(m.layout.tls_base, 0xfeedface).unwrap();
        m.svc(0x0a).unwrap();
        m.core.cpu.pc += 4;
        let before = m.core.cpu.clone();
        let child = m.threads.report();
        assert!(matches!(m.finish_block(), Err(Trap::InvalidState { .. })));
        assert_eq!(m.core.cpu, before);
        assert_eq!(m.threads.report(), child);
        assert_eq!(m.current_thread_id(), 1);
        assert_eq!(m.main_thread_state, ThreadState::Runnable);
        assert_eq!(m.pending_thread_exit, Some(1));
        assert_eq!(m.context_switches, 0);
        assert_eq!(m.thread_exits, 0);
        assert_eq!(m.memory.read_u64(m.layout.tls_base).unwrap(), 0xfeedface);
    }

    #[test]
    fn thread_core_mask_updates_current_affinity_with_raw_abi_and_preserved_creation() {
        let mut m = thread_machine();
        create_thread_args(&mut m, 28, 5);
        m.svc(0x08).unwrap();
        let handle = m.core.cpu.regs[1];
        let initial = m.threads.report()["threads"][0].clone();
        let services = m.services.report();
        let mappings = m.memory.mappings().cloned().collect::<Vec<_>>();
        let resident = m.memory.resident_bytes();
        let tls = m.layout.tls_base + PAGE_SIZE;
        let mut initial_tls = [0; 0x200];
        m.memory.read(tls, &mut initial_tls).unwrap();
        for (selector, mask, expected_ideal, expected_mask) in [
            (0xabcdef1200000003, 0x18, 3, 0x18),
            (0xabcdef12fffffffd, 0x28, 3, 0x28),
            (0xabcdef12ffffffff, 0x11, -1, 0x11),
            (0xabcdef12fffffffe, u64::MAX, 5, 0x20),
            (0xabcdef12fffffffe, 0, 5, 0x20),
            (5, 0x20, 5, 0x20),
        ] {
            m.core.cpu.regs[..4].copy_from_slice(&[
                0xfedcba9800000000 | handle,
                selector,
                mask,
                u64::MAX,
            ]);
            let before = m.core.cpu.clone();
            m.svc(0x0f).unwrap();
            let mut expected = before.clone();
            expected.regs[0] = 0;
            assert_eq!(m.core.cpu, expected);
            assert_eq!(m.core.steps, 0);
            assert_eq!(m.threads.affinity(2), Some((expected_ideal, expected_mask)));
            let mut expected_thread = initial.clone();
            expected_thread["ideal_core"] = expected_ideal.into();
            expected_thread["affinity_mask"] = expected_mask.into();
            assert_eq!(m.threads.report()["threads"][0], expected_thread);
            assert_eq!(m.services.report(), services);
            assert_eq!(m.memory.mappings().cloned().collect::<Vec<_>>(), mappings);
            assert_eq!(m.memory.resident_bytes(), resident);
            let mut bytes = [0; 0x200];
            m.memory.read(tls, &mut bytes).unwrap();
            assert_eq!(bytes, initial_tls);
            assert_eq!(m.svc_trace.last().unwrap().arguments, before.regs[..4]);
            assert_eq!(m.svc_trace.last().unwrap().arguments_4_5, None);
            assert_eq!(m.layout.process_ideal_core_id, 5);
            assert_eq!(m.layout.process_core_mask, Some(0x3f));
        }
        // No 32-bit narrowing of X2, even though the actual profile grants six
        // cores. Synthetic validated metadata can grant virtual core 63.
        m.layout.process_core_mask = Some(u64::MAX);
        m.core.cpu.regs[..3].copy_from_slice(&[handle, 0xabcdef120000003f, 1u64 << 63]);
        m.svc(0x0f).unwrap();
        assert_eq!(m.threads.affinity(2), Some((63, 1u64 << 63)));
        m.core.cpu.regs[..3].copy_from_slice(&[handle, 0xfffffffd, (1u64 << 63) | 1]);
        m.svc(0x0f).unwrap();
        assert_eq!(m.threads.affinity(2), Some((63, (1u64 << 63) | 1)));
        let report = m.report(&Trap::Exit { code: 0 });
        assert_eq!(report["thread_model"]["threads"][0]["creation_core"], 5);
        assert_eq!(
            report["thread_model"]["threads"][0]["creation_affinity_mask"],
            0x20
        );
        assert_eq!(
            report["thread_model"]["set_thread_core_mask_implemented"],
            true
        );
        assert_eq!(report["thread_model"]["thread_scheduler_implemented"], true);
        assert_eq!(report["cpu_model"]["executing_guest_threads"], 1);
    }

    #[test]
    fn thread_core_mask_checks_typed_handle_then_mask_then_selector_atomically() {
        let mut m = thread_machine();
        create_thread_args(&mut m, 28, 5);
        m.svc(0x08).unwrap();
        let handle = m.core.cpu.regs[1];
        let session = u64::from(m.services.connect_named_port("sm:", 0).unwrap());
        let threads = m.thread_model_report();
        let services = m.services.report();
        let mappings = m.memory.mappings().cloned().collect::<Vec<_>>();
        for (target, selector, mask, result) in [
            (0xdead, 64, 0, 0xe401),
            (session, u64::MAX, u64::MAX, 0xe401),
            (0xffff8001, 64, 0, 0xe401),
            (handle, 64, 0, 0xe801),
            (handle, 5, 0, 0xe801),
            (handle, 5, 0x40, 0xe201),
            (handle, 5, 1, 0xe801),
            (handle, 64, 1, 0xe201),
            (handle, 0xfffffffc, 1, 0xe201),
            (handle, 0xfffffffd, 1, 0xe801),
            (handle, 63, 1u64 << 63, 0xe201),
        ] {
            m.core.cpu.regs[..4].copy_from_slice(&[
                0xabcdef1200000000 | target,
                selector,
                mask,
                0xfeedface,
            ]);
            let mut expected = m.core.cpu.clone();
            expected.regs[0] = result;
            m.svc(0x0f).unwrap();
            assert_eq!(m.core.cpu, expected);
            assert_eq!(m.thread_model_report(), threads);
            assert_eq!(m.services.report(), services);
            assert_eq!(m.memory.mappings().cloned().collect::<Vec<_>>(), mappings);
        }
        m.core.cpu.regs[0] = handle;
        m.svc(0x16).unwrap();
        m.core.cpu.regs[..3].copy_from_slice(&[handle, 64, 0]);
        m.svc(0x0f).unwrap();
        assert_eq!(m.core.cpu.regs[0], 0xe401);
        assert_eq!(m.threads.len(), 0);
        // Missing permission metadata must not supersede typed-handle errors.
        m.layout.process_core_mask = None;
        m.core.cpu.regs[..3].copy_from_slice(&[handle, 0xfffffffe, 0]);
        m.svc(0x0f).unwrap();
        assert_eq!(m.core.cpu.regs[0], 0xe401);
    }

    #[test]
    fn thread_core_mask_unsupported_or_missing_metadata_preserves_state() {
        let mut m = thread_machine();
        create_thread_args(&mut m, 28, 5);
        m.svc(0x08).unwrap();
        let handle = m.core.cpu.regs[1];
        for (grant, default, selector, mask) in [
            (None, 5, 5, 0x20),
            (Some(0), 5, 0xfffffffe, 0),
            (Some(0x3f), 64, 0xfffffffe, u64::MAX),
            (Some(1), 5, 0xfffffffe, 1),
        ] {
            m.layout.process_core_mask = grant;
            m.layout.process_ideal_core_id = default;
            m.core.cpu.regs[..3].copy_from_slice(&[handle, selector, mask]);
            let before = m.core.cpu.clone();
            let threads = m.thread_model_report();
            let services = m.services.report();
            assert!(matches!(m.svc(0x0f), Err(Trap::InvalidState { .. })));
            assert_eq!(m.core.cpu, before);
            assert_eq!(m.thread_model_report(), threads);
            assert_eq!(m.services.report(), services);
        }
        m.layout.process_core_mask = Some(0x3f);
        m.layout.process_ideal_core_id = 5;
        m.core.cpu.regs[..3].copy_from_slice(&[handle, 0xffffffff, 3]);
        m.svc(0x0f).unwrap();
        assert_eq!(m.threads.affinity(2), Some((-1, 3)));
        m.core.cpu.regs[..3].copy_from_slice(&[handle, 0xfffffffd, 3]);
        let before = m.core.cpu.clone();
        let threads = m.thread_model_report();
        assert!(matches!(m.svc(0x0f), Err(Trap::InvalidState { .. })));
        assert_eq!(m.core.cpu, before);
        assert_eq!(m.thread_model_report(), threads);
        // Outer mask validation still precedes the ambiguous retained-ideal
        // edge. A later UseProcessValue can recover from DontCare normally.
        m.core.cpu.regs[2] = 0;
        m.svc(0x0f).unwrap();
        assert_eq!(m.core.cpu.regs[0], 0xe801);
        m.core.cpu.regs[..3].copy_from_slice(&[handle, 0xfffffffe, 0]);
        m.svc(0x0f).unwrap();
        assert_eq!(m.threads.affinity(2), Some((5, 0x20)));
    }

    #[test]
    fn main_thread_affinity_preserves_process_default_and_stops_before_migration() {
        let layout = Layout {
            process_core_mask: Some(0x3f),
            ..Layout::default()
        };
        let mut m = Machine::new(layout, 1);
        let handle = m.layout.main_thread_handle;
        m.core.cpu.regs[..4].copy_from_slice(&[handle, 1, 3, u64::MAX]);
        m.svc(0x0f).unwrap();
        assert_eq!(m.main_thread_affinity, (1, 3));
        assert_eq!(m.layout.process_ideal_core_id, 0);
        m.svc(0x10).unwrap();
        assert_eq!(m.core.cpu.regs[0], 0);
        m.core.cpu.regs[..3].copy_from_slice(&[0xffff8000, 2, 4]);
        let before = m.core.cpu.clone();
        let threads = m.thread_model_report();
        assert!(matches!(m.svc(0x0f), Err(Trap::InvalidState { .. })));
        assert_eq!(m.core.cpu, before);
        assert_eq!(m.thread_model_report(), threads);
        assert_eq!(m.layout.processor_id, 0);
        m.core.cpu.regs[..3].copy_from_slice(&[0xffff8000, 0xfffffffd, 1]);
        m.svc(0x0f).unwrap();
        assert_eq!(m.core.cpu.regs[0], 0xe801);
        assert_eq!(m.main_thread_affinity, (1, 3));
        m.core.cpu.regs[0] = handle;
        m.svc(0x16).unwrap();
        m.core.cpu.regs[..3].copy_from_slice(&[handle, 64, 0]);
        m.svc(0x0f).unwrap();
        assert_eq!(m.core.cpu.regs[0], 0xe401);
        m.core.cpu.regs[..3].copy_from_slice(&[0xabcdef12ffff8000, 0xabcdef12fffffffe, u64::MAX]);
        m.svc(0x0f).unwrap();
        assert_eq!(m.core.cpu.regs[0], 0);
        assert_eq!(m.main_thread_affinity, (0, 1));
        assert!(!m.main_thread_handle_open);
        let model = m.thread_model_report();
        assert_eq!(
            model["main_thread_affinity"],
            serde_json::json!({
                "thread_id":1,"ideal_core":0,"affinity_mask":1,"current_processor_id":0
            })
        );
        assert_eq!(model["main_thread_migration_implemented"], false);
        assert_eq!(model["physical_core_topology_implemented"], false);
    }

    #[test]
    fn create_thread_validates_signed_core_before_priority_and_requires_real_masks() {
        for (priority, core, result) in [
            (u64::MAX, u64::MAX, 0xe201),
            (64, 0xffff_fffd, 0xe201),
            (28, 64, 0xe201),
            (27, 6, 0xe201),
            (27, 5, 0xe001),
            (60, 5, 0xe001),
            (64, 5, 0xe001),
            (u64::MAX, 5, 0xe001),
        ] {
            let mut m = thread_machine();
            create_thread_args(&mut m, priority, core);
            let before = m.core.cpu.clone();
            let services = m.services.report();
            let threads = m.threads.report();
            let mappings = m.memory.mappings().cloned().collect::<Vec<_>>();
            let resident = m.memory.resident_bytes();
            m.svc(0x08).unwrap();
            let mut expected = before;
            expected.regs[0] = result;
            expected.regs[1] = 0;
            assert_eq!(
                m.core.cpu, expected,
                "priority {priority:#x}, core {core:#x}"
            );
            assert_eq!(m.services.report(), services);
            assert_eq!(m.threads.report(), threads);
            assert_eq!(m.memory.mappings().cloned().collect::<Vec<_>>(), mappings);
            assert_eq!(m.memory.resident_bytes(), resident);
            assert_eq!(m.threads.next_id().unwrap(), 2);
        }
        for (core_mask, priority_mask) in [
            (None, Some(1 << 28)),
            (Some(0), Some(1 << 28)),
            (Some(0x3f), None),
            (Some(0x3f), Some(0)),
        ] {
            let mut m = thread_machine();
            m.layout.process_core_mask = core_mask;
            m.layout.process_priority_mask = priority_mask;
            create_thread_args(&mut m, 28, 5);
            let before = m.core.cpu.clone();
            assert!(matches!(m.svc(0x08), Err(Trap::InvalidState { .. })));
            assert_eq!(m.core.cpu, before);
            assert_eq!(m.threads.len(), 0);
            assert_eq!(
                m.memory.query(m.layout.tls_base + PAGE_SIZE).state,
                crate::memory::STATE_FREE
            );
        }
        let mut m = thread_machine();
        m.layout.process_core_mask = Some(1 << 63);
        m.layout.process_priority_mask = Some(1 << 63);
        m.layout.process_ideal_core_id = 63;
        create_thread_args(&mut m, 0xffff_ffff_0000_003f, 0xffff_fffe);
        m.svc(0x08).unwrap();
        assert_eq!(m.core.cpu.regs[0], 0);
        assert_eq!(m.threads.priority(2), Some(63));
    }

    #[test]
    fn initialized_thread_handles_are_typed_and_close_releases_only_owned_tls() {
        let mut m = thread_machine();
        create_thread_args(&mut m, 28, 5);
        m.svc(0x08).unwrap();
        let handle = m.core.cpu.regs[1];
        let tls = m.layout.tls_base + PAGE_SIZE;
        let session = m.services.connect_named_port("sm:", 0).unwrap();
        for (number, expected) in [(0x0c, 28), (0x25, 2)] {
            m.core.cpu.regs[..2].copy_from_slice(&[u64::MAX, 0xabcdef1200000000 | handle]);
            m.svc(number).unwrap();
            assert_eq!((m.core.cpu.regs[0], m.core.cpu.regs[1]), (0, expected));
            m.core.cpu.regs[1] = u64::from(session);
            m.svc(number).unwrap();
            assert_eq!((m.core.cpu.regs[0], m.core.cpu.regs[1]), (0xe401, 0));
        }
        // Typed rejection must happen before any access to this invalid TLS.
        m.core.cpu.tpidrro_el0 = u64::MAX;
        m.core.cpu.regs[0] = handle;
        m.svc(0x21).unwrap();
        assert_eq!(m.core.cpu.regs[0], 0xe401);
        m.core.cpu.regs[0] = m.layout.main_thread_handle;
        m.svc(0x16).unwrap();
        m.core.cpu.regs[1] = 0xffff8000;
        m.svc(0x25).unwrap();
        assert_eq!((m.core.cpu.regs[0], m.core.cpu.regs[1]), (0, 1));
        assert_eq!(m.threads.len(), 1);
        m.core.cpu.regs[0] = handle;
        m.svc(0x16).unwrap();
        assert_eq!(m.core.cpu.regs[0], 0);
        assert_eq!(m.threads.len(), 0);
        assert_eq!(m.services.thread_id(handle as u32), None);
        assert_eq!(m.memory.query(tls).state, crate::memory::STATE_FREE);
        assert_eq!(m.memory.stack_aliases().count(), 1);
        assert_eq!(
            m.memory.read_u64(m.layout.stack_region_base).unwrap(),
            0x0123456789abcdef
        );
        assert_eq!(
            m.memory.read_u64(m.layout.tls_base + 0x120).unwrap(),
            0xfeedface
        );
        m.core.cpu.regs[0] = handle;
        m.svc(0x16).unwrap();
        assert_eq!(m.core.cpu.regs[0], 0xe401);
        for number in [0x0c, 0x25] {
            m.core.cpu.regs[1] = handle;
            m.svc(number).unwrap();
            assert_eq!((m.core.cpu.regs[0], m.core.cpu.regs[1]), (0xe401, 0));
        }
        create_thread_args(&mut m, 28, 5);
        m.svc(0x08).unwrap();
        assert_eq!(m.core.cpu.regs[1], 0x202);
        assert_eq!(m.services.thread_id(0x202), Some(3));
        assert_eq!(m.memory.read_u32(tls + 0x110).unwrap(), 0x202);
    }

    #[test]
    fn create_thread_tls_and_shared_handle_failures_publish_nothing_and_can_retry() {
        let mut m = thread_machine();
        let tls = m.layout.tls_base + PAGE_SIZE;
        m.memory
            .map(tls, PAGE_SIZE, READ | WRITE, STATE_NORMAL, "collision")
            .unwrap();
        m.memory.write_u64(tls, 0xfeed).unwrap();
        create_thread_args(&mut m, 28, 5);
        let before = m.core.cpu.clone();
        let services = m.services.report();
        let mappings = m.memory.mappings().cloned().collect::<Vec<_>>();
        assert!(matches!(m.svc(0x08), Err(Trap::InvalidState { .. })));
        assert_eq!(m.core.cpu, before);
        assert_eq!(m.services.report(), services);
        assert_eq!(m.memory.mappings().cloned().collect::<Vec<_>>(), mappings);
        assert_eq!(m.memory.read_u64(tls).unwrap(), 0xfeed);
        assert_eq!(m.threads.next_id().unwrap(), 2);
        m.memory.unmap_exact(tls, PAGE_SIZE).unwrap();
        m.svc(0x08).unwrap();
        assert_eq!(m.core.cpu.regs[1], 0x200);
        let mut m = thread_machine();
        let mut sessions = Vec::new();
        for _ in 0..1024 {
            sessions.push(m.services.connect_named_port("sm:", 0).unwrap());
        }
        create_thread_args(&mut m, 28, 5);
        let before = m.core.cpu.clone();
        let services = m.services.report();
        assert!(matches!(m.svc(0x08), Err(Trap::InvalidState { .. })));
        assert_eq!(m.core.cpu, before);
        assert_eq!(m.services.report(), services);
        assert_eq!(m.threads.len(), 0);
        assert_eq!(m.threads.next_id().unwrap(), 2);
        assert_eq!(
            m.memory.query(m.layout.tls_base + PAGE_SIZE).state,
            crate::memory::STATE_FREE
        );
        assert!(m.services.close(sessions[0]));
        m.svc(0x08).unwrap();
        assert_eq!(m.core.cpu.regs[1], 0x600);
        assert_eq!(m.services.thread_id(0x600), Some(2));
    }

    fn map_memory_machine(state: u32) -> Machine {
        let layout = Layout {
            address_space_end: 0x100000,
            heap_base: 0x10000,
            heap_capacity: 0x10000,
            alias_base: 0x30000,
            alias_size: 0x10000,
            stack_region_base: 0x20000,
            stack_region_size: 0x50000,
            initial_used_memory: 0x4000,
            ..Layout::default()
        };
        let mut m = Machine::new(layout, 1);
        m.memory
            .map(0x8000, 3 * PAGE_SIZE, READ | WRITE, state, "source")
            .unwrap();
        m.memory.write_u64(0x9000, 0x0123456789abcdef).unwrap();
        m.core.cpu.regs = std::array::from_fn(|index| 0xfedcba9800000000 + index as u64);
        m.core.cpu.nzcv = 0xb;
        m.core.cpu.sp = 0xdead;
        m
    }

    #[test]
    fn map_memory_returns_raw_result_and_exposes_locked_source_and_stack_alias() {
        for state in [STATE_CODE_DATA, STATE_NORMAL, STATE_ALIAS_CODE_DATA] {
            let mut m = map_memory_machine(state);
            let resident = m.memory.resident_bytes();
            m.core.cpu.regs[..3].copy_from_slice(&[0x20000, 0x9000, PAGE_SIZE]);
            let before = m.core.cpu.regs;
            m.svc(0x04).unwrap();
            assert_eq!(m.core.cpu.regs[0], 0);
            assert_eq!(m.core.cpu.regs[1..], before[1..]);
            assert_eq!((m.core.cpu.sp, m.core.cpu.nzcv), (0xdead, 0xb));
            assert_eq!(m.memory.resident_bytes(), resident);
            assert_eq!(m.memory.read_u64(0x20000).unwrap(), 0x0123456789abcdef);
            assert!(matches!(
                m.memory.read_u64(0x9000),
                Err(Trap::MemoryFault {
                    address: 0x9000,
                    ..
                })
            ));
            assert!(m.memory.write_u64(0x9000, 0).is_err());
            assert!(m.memory.check_execute(0x20000, 4).is_err());
            assert_eq!(m.memory.query(0x8000).permissions, READ | WRITE);
            assert_eq!(m.memory.query(0xa000).permissions, READ | WRITE);

            // QueryMemory is a guest ABI check, including zero IPC/device refs,
            // rather than relying only on the internal mapping metadata.
            m.memory
                .map(0x1000, PAGE_SIZE, READ | WRITE, STATE_TLS, "info")
                .unwrap();
            for (address, expected_state, attributes, permissions) in [
                (0x9000, state, ATTRIBUTE_LOCKED, 0),
                (0x20000, STATE_STACK, 0, u32::from(READ | WRITE)),
            ] {
                m.memory.write(0x1000, &[0xff; 0x28]).unwrap();
                m.core.cpu.regs[..4].copy_from_slice(&[0x1000, u64::MAX, address, 0x1234]);
                m.svc(0x06).unwrap();
                assert_eq!((m.core.cpu.regs[0], m.core.cpu.regs[1]), (0, 0));
                assert_eq!(m.core.cpu.regs[2..4], [address, 0x1234]);
                assert_eq!(m.memory.read_u64(0x1000).unwrap(), address);
                assert_eq!(m.memory.read_u64(0x1008).unwrap(), PAGE_SIZE);
                assert_eq!(m.memory.read_u32(0x1010).unwrap(), expected_state);
                assert_eq!(m.memory.read_u32(0x1014).unwrap(), attributes);
                assert_eq!(m.memory.read_u32(0x1018).unwrap(), permissions);
                let mut refs_and_padding = [0xff; 12];
                m.memory.read(0x101c, &mut refs_and_padding).unwrap();
                assert_eq!(refs_and_padding, [0; 12]);
            }
            for kind in [7, 22] {
                m.core.cpu.regs[..4].copy_from_slice(&[0xdead, kind, 0xffff8001, 0]);
                m.svc(0x29).unwrap();
                assert_eq!((m.core.cpu.regs[0], m.core.cpu.regs[1]), (0, 0x4000));
            }
            let report = m.report(&Trap::Exit { code: 0 });
            let model = &report["memory_model"];
            assert_eq!(model["active_stack_alias_count"], 1);
            assert_eq!(
                model["stack_aliases"],
                serde_json::json!([{
                    "destination":0x20000,"source":0x9000,"size":PAGE_SIZE,
                    "source_state":state,"source_permissions":0,"source_attributes":1,
                    "destination_state":11,"destination_permissions":3,"destination_attributes":0,
                }])
            );
            assert_eq!(model["map_memory_allocates_user_pages"], false);
            assert_eq!(model["map_memory_changes_committed_user_memory"], false);
            assert_eq!(
                model["kernel_mapping_metadata_resource_accounting_implemented"],
                false
            );
            assert_eq!(model["layout_console_verified"], false);
            assert_eq!(model["shadow_stack_region_implemented"], false);
            assert_eq!(model["unmap_memory_implemented"], true);
            assert_eq!(&m.report(&Trap::Exit { code: 0 })["memory_model"], model);
            // UnmapMemory with the same arguments removes the alias and
            // unlocks the source.
            m.core.cpu.regs[..3].copy_from_slice(&[0x20000, 0x9000, PAGE_SIZE]);
            m.svc(0x05).unwrap();
            assert_eq!(m.core.cpu.regs[0], 0);
            assert_eq!(m.memory.stack_aliases().count(), 0);
            let source = m.memory.query(0x9000);
            assert_eq!((source.permissions, source.attributes), (READ | WRITE, 0));
        }
    }

    #[test]
    fn map_memory_error_precedence_returns_zero_extended_results_without_mutation() {
        // Deliberately combine invalid fields to verify the public validation
        // order. Expected Result numbers are independent literal ABI fixtures.
        for (destination, source, size, result) in [
            (0x20001, 0x9001, 0, 0xcc01),
            (0x20000, 0x9001, 0, 0xcc01),
            (0x80000, 0x100000, 0, 0xca01),
            (0x80000, 0x100000, PAGE_SIZE + 1, 0xca01),
            (0x80000, u64::MAX - PAGE_SIZE + 1, PAGE_SIZE, 0xd401),
            (u64::MAX - PAGE_SIZE + 1, 0x9000, PAGE_SIZE, 0xd401),
            (0x80000, 0x100000, PAGE_SIZE, 0xd401),
            (0x80000, 0, PAGE_SIZE, 0xdc01),
            (0x20000, 0, PAGE_SIZE, 0xd401),
            (0x20000, 0xa000, 2 * PAGE_SIZE, 0xd401),
        ] {
            let mut m = map_memory_machine(STATE_CODE_DATA);
            let mappings: Vec<_> = m.memory.mappings().cloned().collect();
            let model = m.memory_model_report();
            let resident = m.memory.resident_bytes();
            m.core.cpu.regs[..3].copy_from_slice(&[destination, source, size]);
            let before = m.core.cpu.regs;
            m.svc(0x04).unwrap();
            assert_eq!(
                m.core.cpu.regs[0], result,
                "{destination:#x}/{source:#x}/{size:#x}"
            );
            assert_eq!(m.core.cpu.regs[1..], before[1..]);
            assert_eq!((m.core.cpu.sp, m.core.cpu.nzcv), (0xdead, 0xb));
            assert_eq!(m.memory.mappings().cloned().collect::<Vec<_>>(), mappings);
            assert_eq!(m.memory_model_report(), model);
            assert_eq!(m.memory.resident_bytes(), resident);
            assert_eq!(m.memory.read_u64(0x9000).unwrap(), 0x0123456789abcdef);
        }
        // An otherwise valid occupied destination is a memory-state error.
        let mut m = map_memory_machine(STATE_CODE_DATA);
        m.memory
            .map(0x20000, PAGE_SIZE, READ | WRITE, STATE_STACK, "existing")
            .unwrap();
        m.core.cpu.regs[..3].copy_from_slice(&[0x20000, 0x9000, PAGE_SIZE]);
        m.svc(0x04).unwrap();
        assert_eq!(m.core.cpu.regs[0], 0xd401);
        assert_eq!(m.memory.query(0x9000).permissions, READ | WRITE);
        assert_eq!(m.memory.stack_aliases().count(), 0);
    }

    #[test]
    fn map_memory_stack_containment_excludes_nonempty_heap_and_alias_regions() {
        for (destination, size, result) in [
            (0x1f000, 2 * PAGE_SIZE, 0xdc01), // Starts before Stack.
            (0x6f000, 2 * PAGE_SIZE, 0xdc01), // Ends after Stack.
            (0x6f000, PAGE_SIZE, 0),          // Exact Stack end is valid.
            (0x2f000, 2 * PAGE_SIZE, 0xdc01), // Crosses into Alias.
            (0x30000, PAGE_SIZE, 0xdc01),     // Wholly within Alias.
            (0x2f000, PAGE_SIZE, 0),          // Adjacent to Alias is valid.
            (0x40000, PAGE_SIZE, 0),
        ] {
            let mut m = map_memory_machine(STATE_NORMAL);
            m.core.cpu.regs[..3].copy_from_slice(&[destination, 0x8000, size]);
            m.svc(0x04).unwrap();
            assert_eq!(m.core.cpu.regs[0], result, "{destination:#x}/{size:#x}");
        }
        for variant in 0..6 {
            let mut m = map_memory_machine(STATE_NORMAL);
            let expected = match variant {
                0 => {
                    m.layout.stack_region_size = 0;
                    0xdc01
                }
                1 => {
                    m.layout.stack_region_base = u64::MAX - PAGE_SIZE + 1;
                    0xdc01
                }
                2 => {
                    m.layout.stack_region_size = u64::MAX;
                    0xdc01
                }
                3 => {
                    m.layout.heap_base = 0x20000;
                    0xdc01
                }
                4 => {
                    m.layout.heap_base = 0x20000;
                    m.layout.heap_capacity = 0;
                    0
                }
                5 => {
                    m.layout.alias_base = 0x20000;
                    m.layout.alias_size = 0;
                    0
                }
                _ => unreachable!(),
            };
            m.core.cpu.regs[..3].copy_from_slice(&[0x20000, 0x8000, PAGE_SIZE]);
            m.svc(0x04).unwrap();
            assert_eq!(m.core.cpu.regs[0], expected, "layout case {variant}");
        }
    }

    #[test]
    fn physical_counter_conversion_uses_integer_time_units_and_full_width_wrap() {
        // Independent decimal fixtures include fractional tick boundaries,
        // values beyond f64's exact integer range and the 64-bit wrap boundary.
        for (seconds, nanos, expected) in [
            (0, 0, 0),
            (0, 31, 0),
            (0, 32, 1),
            (0, 625, 19),
            (1, 0, 31_250_000),
            (1_000_000_000, 123_456_789, 31_250_000_003_858_024),
            (590_295_810_358, 705_651_711, u64::MAX),
            (590_295_810_358, 705_651_712, 0),
            (590_295_810_358, 705_651_744, 1),
            (u64::MAX, 999_999_999, u64::MAX),
        ] {
            assert_eq!(
                physical_ticks_for_elapsed(Duration::new(seconds, nanos)),
                expected
            );
        }
    }

    #[test]
    fn virtual_clock_follows_instructions() {
        let mut counter = PhysicalCounter::default();
        counter.instructions_per_second = Some(31_250_000); // One tick each.
        assert_eq!(counter.read(0), 0);
        // One millisecond, then never backwards, even for an earlier count.
        assert_eq!(counter.read(31_250), 31_250);
        assert_eq!(counter.read(10), 31_250);
    }

    #[test]
    fn physical_counter_keeps_elapsed_monotonic_without_forcing_tick_advancement() {
        let mut counter = PhysicalCounter::default();
        assert_eq!(physical_counter_report(&counter)["read_count"], 0);
        assert!(physical_counter_report(&counter)["last_value"].is_null());
        assert_eq!(counter.sample_elapsed(Duration::from_nanos(625)), 19);
        assert_eq!(counter.sample_elapsed(Duration::from_nanos(625)), 19);
        assert_eq!(counter.sample_elapsed(Duration::from_nanos(624)), 19);
        assert_eq!(counter.sample_elapsed(Duration::from_nanos(1_250)), 39);
        assert_eq!(
            counter.sample_elapsed(Duration::new(590_295_810_358, 705_651_711)),
            u64::MAX
        );
        assert_eq!(
            counter.sample_elapsed(Duration::new(590_295_810_358, 705_651_712)),
            0
        );
        let report = physical_counter_report(&counter);
        assert_eq!(report["read_count"], 6);
        assert_eq!(report["last_value"], 0);
        assert_eq!(report["backward_samples_clamped"], 1);
        assert_eq!(report["frequency_hz"], 31_250_000);
        assert_eq!(report["deterministic"], false);
        assert_eq!(report["uses_guest_instruction_count"], false);
        assert_eq!(report["timer_interrupts_implemented"], false);
        assert_eq!(physical_counter_report(&counter), report);
    }

    #[test]
    fn named_port_and_ipc_use_raw_register_abi_and_tls_then_close_invalidates_handle() {
        let mut m = Machine::new(Layout::default(), 1);
        m.memory
            .map(0x8000, PAGE_SIZE, READ | WRITE, STATE_TLS, "TLS")
            .unwrap();
        m.core.cpu.tpidrro_el0 = 0x8000;
        m.memory.write(0x8200, b"sm:\0").unwrap();
        // X0 is deliberately unmapped; the C pointer is the wrapper's concern.
        m.core.cpu.regs[..3].copy_from_slice(&[0xdead, 0x8200, 0x12345678]);
        m.svc(0x1f).unwrap();
        let handle = m.core.cpu.regs[1];
        assert_eq!(m.core.cpu.regs[0], 0);
        assert!(m.services.contains(handle as u32));
        assert_eq!(m.core.cpu.regs[2], 0x12345678);
        let mut request = [0; crate::ipc::TLS_MESSAGE_BUFFER_SIZE];
        request[..4].copy_from_slice(&0x10u32.to_le_bytes());
        request[4..8].copy_from_slice(&0x80000000u32.to_le_bytes());
        request[8..12].copy_from_slice(&1u32.to_le_bytes());
        request[12..20].copy_from_slice(&999u64.to_le_bytes());
        m.memory.write(0x8000, &request).unwrap();
        // Only W0 carries the session handle, even if X0's high half is set.
        m.core.cpu.regs[0] = handle | 0xabcdef0100000000;
        m.core.cpu.regs[1] = 0xfeed;
        m.svc(0x21).unwrap();
        assert_eq!((m.core.cpu.regs[0], m.core.cpu.regs[1]), (0, 0xfeed));
        assert_eq!(m.memory.read_u32(0x8000).unwrap(), 0x10);
        assert_eq!(m.memory.read_u32(0x8004).unwrap(), 1);
        assert_eq!(m.memory.read_u32(0x8008).unwrap(), 0);
        assert_eq!(m.services.report()["trace"][0]["completed"], true);
        m.core.cpu.regs[0] = handle;
        m.svc(0x16).unwrap();
        assert_eq!(m.core.cpu.regs[0], 0);
        assert!(!m.services.contains(handle as u32));
        let mut before = [0; crate::ipc::TLS_MESSAGE_BUFFER_SIZE];
        m.memory.read(0x8000, &mut before).unwrap();
        m.core.cpu.regs[0] = handle;
        m.svc(0x21).unwrap();
        assert_eq!(m.core.cpu.regs[0], 0xe401);
        let mut after = [0; crate::ipc::TLS_MESSAGE_BUFFER_SIZE];
        m.memory.read(0x8000, &mut after).unwrap();
        assert_eq!(after, before);
        m.core.cpu.regs[0] = handle;
        m.svc(0x16).unwrap();
        assert_eq!(m.core.cpu.regs[0], 0xe401);
    }

    #[test]
    fn closing_thread_handle_keeps_current_thread_pseudo_handle_alive() {
        let mut m = Machine::new(Layout::default(), 1);
        m.core.cpu.regs[0] = m.layout.main_thread_handle;
        m.svc(0x16).unwrap();
        assert_eq!(m.core.cpu.regs[0], 0);
        for handle in [m.layout.main_thread_handle, 0xffff8000, 0xffff8001] {
            m.core.cpu.regs[0] = handle;
            m.svc(0x16).unwrap();
            assert_eq!(m.core.cpu.regs[0], 0xe401);
        }
        for number in [0x0c, 0x25] {
            m.core.cpu.regs[1] = m.layout.main_thread_handle;
            m.svc(number).unwrap();
            assert_eq!((m.core.cpu.regs[0], m.core.cpu.regs[1]), (0xe401, 0));
            m.core.cpu.regs[1] = 0xffff8000;
            m.svc(number).unwrap();
            assert_eq!(m.core.cpu.regs[0], 0);
        }
        assert!(
            !m.report(&Trap::Exit { code: 0 })["main_thread_handle_open"]
                .as_bool()
                .unwrap()
        );
    }

    #[test]
    fn copied_readable_event_handles_close_independently_and_cannot_receive_ipc() {
        use crate::ipc::TLS_MESSAGE_BUFFER_SIZE;

        fn packet(words: &[(usize, u32)]) -> [u8; TLS_MESSAGE_BUFFER_SIZE] {
            let mut bytes = [0; TLS_MESSAGE_BUFFER_SIZE];
            for &(offset, value) in words {
                bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            }
            bytes
        }
        fn transact(
            m: &mut Machine,
            handle: u32,
            request: &[u8; TLS_MESSAGE_BUFFER_SIZE],
        ) -> [u8; TLS_MESSAGE_BUFFER_SIZE] {
            m.memory.write(0x8000, request).unwrap();
            m.core.cpu.regs[0] = u64::from(handle);
            m.svc(0x21).unwrap();
            assert_eq!(m.core.cpu.regs[0], 0);
            let mut reply = [0; TLS_MESSAGE_BUFFER_SIZE];
            m.memory.read(0x8000, &mut reply).unwrap();
            reply
        }
        fn word(bytes: &[u8], offset: usize) -> u32 {
            crate::bytes::le_u32(&bytes[offset..offset + 4])
        }
        fn command(id: u32) -> [u8; TLS_MESSAGE_BUFFER_SIZE] {
            packet(&[(0, 4), (4, 8), (16, u32::from_le_bytes(*b"SFCI")), (24, id)])
        }

        let mut m = Machine::new(Layout::default(), 1);
        m.memory
            .map(0x8000, PAGE_SIZE, READ | WRITE, STATE_TLS, "TLS")
            .unwrap();
        m.core.cpu.tpidrro_el0 = 0x8000;
        m.services
            .initialize_main_application(m.layout.process_id, 0x0100000000123000, &"ab".repeat(32))
            .unwrap();
        let sm = m.services.connect_named_port("sm:", 0).unwrap();
        transact(
            &mut m,
            sm,
            &packet(&[(0, 0x10), (4, 0x80000000), (8, 1), (12, 999)]),
        );
        let reply = transact(
            &mut m,
            sm,
            &packet(&[
                (0, 0x11),
                (4, 2),
                (8, u32::from_le_bytes(*b"appl")),
                (12, u32::from_le_bytes(*b"etOE")),
            ]),
        );
        let applet = word(&reply, 12);
        // Open the application proxy with a borrowed current-process handle.
        // The PID placeholder differs from the trusted runtime identity.
        let reply = transact(
            &mut m,
            applet,
            &packet(&[
                (0, 4),
                (4, 0x8000000c),
                (8, 3),
                (12, 999),
                (20, 0xffff8001),
                (32, u32::from_le_bytes(*b"SFCI")),
                (40, 10),
            ]),
        );
        let proxy = word(&reply, 12);
        let controller = word(&transact(&mut m, proxy, &command(1)), 12);
        let functions = word(&transact(&mut m, proxy, &command(20)), 12);
        let first = transact(&mut m, controller, &command(91));
        let second = transact(&mut m, controller, &command(91));
        let gpu = transact(&mut m, functions, &command(130));
        assert_eq!(
            (word(&first, 8), word(&second, 8), word(&gpu, 8)),
            (2, 2, 2)
        );
        assert_eq!(word(&first, 28), 0x56ef_eaa1);
        assert_eq!(word(&gpu, 28), 0x21f9_610e);
        let aliases = [word(&first, 12), word(&second, 12), word(&gpu, 12)];
        assert_ne!(aliases[0], aliases[1]);
        assert_ne!(aliases[0], aliases[2]);
        assert_ne!(aliases[1], aliases[2]);
        assert!(aliases.iter().all(|handle| !m.services.contains(*handle)));

        for handle in [functions, controller, proxy, applet, sm] {
            m.core.cpu.regs[0] = u64::from(handle);
            m.svc(0x16).unwrap();
            assert_eq!(m.core.cpu.regs[0], 0);
        }
        let ipc_count = m.services.report()["trace"].as_array().unwrap().len();
        // An event handle must fail the session type check before TLS is read.
        // Closing a parent or another alias must not invalidate this handle.
        m.core.cpu.tpidrro_el0 = 0xdead0000;
        for handle in aliases {
            m.core.cpu.regs[0] = u64::from(handle) | 0xabcdef0100000000;
            let mut expected_start = m.core.cpu.clone();
            expected_start.regs[0] = 0xe401;
            m.svc(9).unwrap();
            assert_eq!(m.core.cpu, expected_start);
            m.core.cpu.regs[..3].copy_from_slice(&[u64::from(handle) | 0xabcdef0100000000, 64, 0]);
            let mut expected = m.core.cpu.clone();
            expected.regs[0] = 0xe401;
            m.svc(0x0f).unwrap();
            assert_eq!(m.core.cpu, expected);
            m.core.cpu.regs[..2].copy_from_slice(&[u64::from(handle) | 0xabcdef0100000000, 0xfeed]);
            m.svc(0x21).unwrap();
            assert_eq!((m.core.cpu.regs[0], m.core.cpu.regs[1]), (0xe401, 0xfeed));
            m.core.cpu.regs[0] = u64::from(handle) | 0xabcdef0100000000;
            m.svc(0x16).unwrap();
            assert_eq!(m.core.cpu.regs[0], 0);
            m.core.cpu.regs[0] = u64::from(handle);
            m.svc(0x16).unwrap();
            assert_eq!(m.core.cpu.regs[0], 0xe401);
        }
        assert_eq!(
            m.services.report()["trace"].as_array().unwrap().len(),
            ipc_count
        );
        let mut after = [0; TLS_MESSAGE_BUFFER_SIZE];
        m.memory.read(0x8000, &mut after).unwrap();
        assert_eq!(after, gpu);
    }

    #[test]
    fn ipc_transport_success_preserves_a_nonzero_service_result_in_the_reply() {
        let mut m = Machine::new(Layout::default(), 1);
        m.memory
            .map(0x8000, PAGE_SIZE, READ | WRITE, STATE_TLS, "TLS")
            .unwrap();
        m.core.cpu.tpidrro_el0 = 0x8000;
        let handle = m.services.connect_named_port("sm:", 0).unwrap();
        // GetServiceHandle on an unregistered session returns InvalidClient.
        m.memory.write_u32(0x8000, 0x11).unwrap();
        m.memory.write_u32(0x8004, 2).unwrap();
        m.core.cpu.regs[0] = handle as u64;
        m.svc(0x21).unwrap();
        assert_eq!(m.core.cpu.regs[0], 0);
        assert_eq!(m.memory.read_u32(0x800c).unwrap(), 0);
        assert_eq!(m.memory.read_u32(0x8010).unwrap(), 0x415);
        assert!(m.services.contains(handle));
        assert_eq!(m.services.report()["trace"][0]["service_result"], 0x415);
    }

    #[test]
    fn unsupported_port_and_invalid_name_do_not_allocate_a_session_or_return_success() {
        let mut m = Machine::new(Layout::default(), 1);
        m.memory
            .map(0x8000, PAGE_SIZE, READ | WRITE, STATE_TLS, "name")
            .unwrap();
        m.memory.write(0x8000, b"nope\0").unwrap();
        m.core.cpu.regs[..2].copy_from_slice(&[0xdead, 0x8000]);
        assert!(matches!(
            m.svc(0x1f),
            Err(Trap::UnsupportedService {
                command_id: None,
                ..
            })
        ));
        assert_eq!((m.core.cpu.regs[0], m.core.cpu.regs[1]), (0xdead, 0x8000));
        m.memory.write(0x8000, &[b'a'; 12]).unwrap();
        assert!(matches!(m.svc(0x1f), Err(Trap::InvalidState { .. })));
        assert_eq!(
            m.services.report()["sessions"].as_object().unwrap().len(),
            0
        );
    }

    #[test]
    fn non_system_accounting_tracks_committed_heap_with_no_private_kernel_pool() {
        let layout = Layout {
            initial_used_memory: 0x5000,
            ..Layout::default()
        };
        let mut m = Machine::new(layout, 1);
        for size in [0x200000, 0] {
            m.core.cpu.regs[1] = size;
            m.svc(1).unwrap();
            for kind in [7, 22] {
                m.core.cpu.regs[..4].copy_from_slice(&[0xdead, kind, 0xffff8001, 0]);
                m.svc(0x29).unwrap();
                assert_eq!((m.core.cpu.regs[0], m.core.cpu.regs[1]), (0, 0x5000 + size));
            }
        }
        for (kind, expected) in [
            (6, m.layout.total_memory),
            (21, m.layout.total_memory),
            (16, 0),
            (17, 0),
        ] {
            m.core.cpu.regs[..4].copy_from_slice(&[0xdead, kind, 0xffff8001, 0]);
            m.svc(0x29).unwrap();
            assert_eq!((m.core.cpu.regs[0], m.core.cpu.regs[1]), (0, expected));
        }
    }

    #[test]
    fn private_kernel_pool_counts_as_used_but_not_non_system_memory() {
        let layout = Layout {
            initial_used_memory: 0x5000,
            system_resource_size: 0x300_0000,
            ..Layout::default()
        };
        let mut m = Machine::new(layout, 1);
        m.core.cpu.regs[1] = 0x200000;
        m.svc(1).unwrap();
        let total = m.layout.total_memory;
        for (kind, expected) in [
            (6, total),
            (21, total - 0x300_0000),
            (7, 0x5000 + 0x200000 + 0x300_0000),
            (22, 0x5000 + 0x200000),
            (16, 0x300_0000),
        ] {
            m.core.cpu.regs[..4].copy_from_slice(&[0xdead, kind, 0xffff8001, 0]);
            m.svc(0x29).unwrap();
            assert_eq!(
                (m.core.cpu.regs[0], m.core.cpu.regs[1]),
                (0, expected),
                "type {kind}"
            );
        }
        m.core.cpu.regs[..4].copy_from_slice(&[0xdead, 17, 0xffff8001, 0]);
        assert!(matches!(m.svc(0x29), Err(Trap::InvalidState { .. })));
    }

    #[test]
    fn alias_queries_report_the_fixed_layout_without_creating_memory() {
        let mut m = Machine::new(Layout::default(), 1);
        for (kind, value) in [
            (2, m.layout.alias_base),
            (3, m.layout.alias_size),
            (28, m.layout.alias_extra_size),
        ] {
            m.core.cpu.regs[..4].copy_from_slice(&[0xdead, kind, 0xffff8001, 0]);
            m.svc(0x29).unwrap();
            assert_eq!((m.core.cpu.regs[0], m.core.cpu.regs[1]), (0, value));
        }
        assert_eq!(m.layout.alias_extra_size, 0);
        assert_eq!(m.memory.mappings().count(), 0);
    }

    #[test]
    fn thread_queries_use_process_state_and_check_thread_handle() {
        let layout = Layout {
            main_thread_priority: 31,
            ..Layout::default()
        };
        let mut m = Machine::new(layout, 1);
        for (number, expected) in [(0x0c, 31), (0x25, 1)] {
            for handle in [0x100, 0xffff8000] {
                m.core.cpu.regs[..2].copy_from_slice(&[0xdead, handle]);
                m.svc(number).unwrap();
                assert_eq!((m.core.cpu.regs[0], m.core.cpu.regs[1]), (0, expected));
            }
            m.core.cpu.regs[..2].copy_from_slice(&[0xdead, 0xffff8001]);
            m.svc(number).unwrap();
            assert_eq!((m.core.cpu.regs[0], m.core.cpu.regs[1]), (0xe401, 0));
        }
    }

    #[test]
    fn physical_memory_maps_alias_pages_once_and_counts_them() {
        let mut m = Machine::new(Layout::default(), 1);
        let alias = m.layout.alias_base;
        let used = |m: &mut Machine| {
            m.core.cpu.regs[..4].copy_from_slice(&[0xdead, 22, 0xffff8001, 0]);
            m.svc(0x29).unwrap();
            m.core.cpu.regs[1]
        };
        let before = used(&mut m);
        let call = |m: &mut Machine, number: u16, address: u64, size: u64| {
            m.core.cpu.regs[..2].copy_from_slice(&[address, size]);
            m.svc(number).unwrap();
            m.core.cpu.regs[0]
        };
        assert_eq!(call(&mut m, 0x2c, alias, PAGE_SIZE), 0);
        m.memory.write(alias, &[7; 8]).unwrap();
        // Mapping an overlapping range keeps existing pages and adds the rest.
        assert_eq!(call(&mut m, 0x2c, alias, 3 * PAGE_SIZE), 0);
        assert_eq!(m.memory.read_u64(alias).unwrap(), 0x0707070707070707);
        assert_eq!(m.memory.read_u64(alias + PAGE_SIZE).unwrap(), 0);
        assert_eq!(used(&mut m), before + 3 * PAGE_SIZE);
        assert_eq!(call(&mut m, 0x2c, alias + 1, PAGE_SIZE), 0xcc01);
        assert_eq!(call(&mut m, 0x2c, alias, 0), 0xca01);
        // Unmapping the middle page keeps both neighbours and their data.
        m.memory.write(alias + 2 * PAGE_SIZE, &[9; 8]).unwrap();
        assert_eq!(call(&mut m, 0x2d, alias + PAGE_SIZE, PAGE_SIZE), 0);
        assert!(m.memory.read_u64(alias + PAGE_SIZE).is_err());
        assert_eq!(m.memory.read_u64(alias).unwrap(), 0x0707070707070707);
        assert_eq!(
            m.memory.read_u64(alias + 2 * PAGE_SIZE).unwrap(),
            0x0909090909090909
        );
        // The neighbours stay read/write: only the covered page is split off.
        for page in [alias, alias + 2 * PAGE_SIZE] {
            assert_eq!(m.memory.query(page).permissions, READ | WRITE);
        }
        assert_eq!(used(&mut m), before + 2 * PAGE_SIZE);
        assert_eq!(call(&mut m, 0x2d, alias, 3 * PAGE_SIZE), 0);
        assert!(m.memory.read_u64(alias).is_err());
        assert_eq!(used(&mut m), before);
    }

    #[test]
    fn remote_region_policy_is_explicit_and_does_not_create_mappings() {
        let layout = Layout {
            remote_memory_model: RemoteMemoryModel::AbsentRegionCompatibility,
            ..Layout::default()
        };
        let mut m = Machine::new(layout, 1);
        for (kind, expected) in [(29, 0), (30, 0), (31, 0), (32, 0), (33, PAGE_SIZE)] {
            // X0 is the C wrapper's pointer, not a raw-SVC output pointer.
            m.core.cpu.regs[..4].copy_from_slice(&[0xdead, kind, 0xffff8001, 0]);
            m.svc(0x29).unwrap();
            assert_eq!((m.core.cpu.regs[0], m.core.cpu.regs[1]), (0, expected));
        }
        assert_eq!(m.memory.mappings().count(), 0);
        assert!(m.memory.read_u64(0).is_err());
        // MapPhysicalMemory outside the alias region is refused, mapping nothing.
        m.core.cpu.regs[..2].copy_from_slice(&[0, PAGE_SIZE]);
        m.svc(0x2c).unwrap();
        assert_eq!(m.core.cpu.regs[0], 0xd401);
        assert_eq!(m.memory.mappings().count(), 0);
        let report = m.report(&Trap::Exit { code: 0 });
        assert_eq!(
            report["layout"]["remote_memory_model"],
            "absent_region_compatibility"
        );
        for (handle, subtype, error) in [(0, 1, 0xf001), (0, 0, 0xe401)] {
            m.core.cpu.regs[..4].copy_from_slice(&[0xdead, 29, handle, subtype]);
            m.svc(0x29).unwrap();
            assert_eq!((m.core.cpu.regs[0], m.core.cpu.regs[1]), (error, 0));
        }
        m.layout.remote_memory_model = RemoteMemoryModel::Unsupported;
        m.core.cpu.regs[..4].copy_from_slice(&[0xdead, 29, 0xffff8001, 0]);
        m.svc(0x29).unwrap();
        assert_eq!((m.core.cpu.regs[0], m.core.cpu.regs[1]), (0xf001, 0));
        m.core.cpu.regs[..4].copy_from_slice(&[0xdead, 34, 0xffff8001, 0]);
        assert!(matches!(m.svc(0x29), Err(Trap::InvalidState { .. })));
    }

    #[test]
    fn random_entropy_has_stable_process_words_and_exact_error_abi() {
        let mut m = Machine::new(Layout::default(), 1);
        m.random_entropy = Some([0x11, 0x22, 0x33, 0x44]);
        for _ in 0..2 {
            m.core.cpu.regs[0] = 0xdead;
            m.core.cpu.regs[1] = 11;
            m.core.cpu.regs[2] = 0;
            m.core.cpu.regs[3] = 2;
            m.svc(0x29).unwrap();
            assert_eq!((m.core.cpu.regs[0], m.core.cpu.regs[1]), (0, 0x33));
        }
        for (handle, index, error) in [(1, 9, 0xe401), (0, 4, 0xe801), (0, u64::MAX, 0xe801)] {
            m.core.cpu.regs[1] = 11;
            m.core.cpu.regs[2] = handle;
            m.core.cpu.regs[3] = index;
            m.svc(0x29).unwrap();
            assert_eq!((m.core.cpu.regs[0], m.core.cpu.regs[1]), (error, 0));
        }
    }

    #[test]
    fn core_mask_uses_validated_process_permissions_and_raw_w_register_abi() {
        let layout = Layout {
            process_core_mask: Some(0x3f),
            ..Layout::default()
        };
        let mut m = Machine::new(layout, 1);
        m.core.cpu.nzcv = 0xa;
        m.core.cpu.regs[4] = 0x123456789abcdef0;
        for mask in [0x3f, 1 << 63] {
            m.layout.process_core_mask = Some(mask);
            // Upper halves of X1/X2 are ignored as W1/W2 inputs. X0 is an
            // unmapped wrapper pointer and must not be dereferenced by SVC.
            m.core.cpu.regs[..4].copy_from_slice(&[0xdead, 0x123400000000, 0x5678ffff8001, 0]);
            m.svc(0x29).unwrap();
            assert_eq!((m.core.cpu.regs[0], m.core.cpu.regs[1]), (0, mask));
            assert_eq!(m.core.cpu.regs[2], 0x5678ffff8001);
            assert_eq!(m.core.cpu.regs[3], 0);
            assert_eq!(m.core.cpu.regs[4], 0x123456789abcdef0);
            assert_eq!(m.core.cpu.nzcv, 0xa);
        }
        assert_eq!(m.memory.mappings().count(), 0);
        assert_eq!(m.memory.resident_bytes(), 0);
        assert_eq!(
            m.report(&Trap::Exit { code: 0 })["layout"]["process_core_mask"],
            serde_json::json!(1u64 << 63)
        );
    }

    #[test]
    fn core_mask_errors_validate_full_subtype_before_process_handle() {
        let mut m = Machine::new(Layout::default(), 1);
        // Invalid calls return their kernel-style error even when permissions
        // are absent. X1=0 on error is our explicit deterministic policy.
        for (handle, subtype, error) in [
            (0, 1, 0xe801),
            (0xffff8001, 1 << 32, 0xe801),
            (0xffff8000, u64::MAX, 0xe801),
            (0, 0, 0xe401),
            (0xffff8000, 0, 0xe401),
            (0x100, 0, 0xe401),
            (0x200, 0, 0xe401),
        ] {
            m.core.cpu.regs[..4].copy_from_slice(&[0xdead, 0, handle, subtype]);
            m.core.cpu.nzcv = 0x6;
            m.svc(0x29).unwrap();
            assert_eq!((m.core.cpu.regs[0], m.core.cpu.regs[1]), (error, 0));
            assert_eq!((m.core.cpu.regs[2], m.core.cpu.regs[3]), (handle, subtype));
            assert_eq!(m.core.cpu.nzcv, 0x6);
        }
    }

    #[test]
    fn core_mask_cannot_succeed_without_nonzero_permissions_or_implement_other_queries() {
        for mask in [None, Some(0)] {
            let mut m = Machine::new(
                Layout {
                    process_core_mask: mask,
                    ..Layout::default()
                },
                1,
            );
            m.core.cpu.regs[..4].copy_from_slice(&[0xdead, 0, 0xffff8001, 0]);
            let before = m.core.cpu.regs;
            assert!(matches!(m.svc(0x29), Err(Trap::InvalidState { .. })));
            assert_eq!(m.core.cpu.regs, before);
        }
        let mut m = Machine::new(
            Layout {
                process_core_mask: Some(0x3f),
                ..Layout::default()
            },
            1,
        );
        m.core.cpu.regs[..4].copy_from_slice(&[0xdead, 1, 0xffff8001, 0]);
        let before = m.core.cpu.regs;
        assert!(matches!(m.svc(0x29), Err(Trap::InvalidState { .. })));
        assert_eq!(m.core.cpu.regs, before);
    }

    #[test]
    fn query_memory_uses_raw_svc_abi_and_complete_structure() {
        let mut m = Machine::new(Layout::default(), 100);
        m.memory
            .map(0x1000, 0x1000, READ | WRITE, STATE_STACK, "output")
            .unwrap();
        m.memory
            .map(0x5000, 0x2000, READ | EXECUTE, STATE_CODE, "text")
            .unwrap();
        m.core.cpu.regs[0] = 0x1000;
        m.core.cpu.regs[1] = 0xffffffffffffffff;
        m.core.cpu.regs[2] = 0x6000;
        m.svc(6).unwrap();
        assert_eq!(m.memory.read_u64(0x1000).unwrap(), 0x5000);
        assert_eq!(m.memory.read_u64(0x1008).unwrap(), 0x2000);
        assert_eq!(m.memory.read_u32(0x1010).unwrap(), STATE_CODE);
        assert_eq!(m.memory.read_u32(0x1018).unwrap(), 5);
        assert_eq!((m.core.cpu.regs[0], m.core.cpu.regs[1]), (0, 0));
    }

    #[test]
    fn heap_resize_preserves_contents_and_zeroes_new_pages() {
        let mut m = Machine::new(Layout::default(), 100);
        m.core.cpu.regs[0] = 999;
        m.core.cpu.regs[1] = 0x200000;
        m.svc(1).unwrap();
        let heap = m.core.cpu.regs[1];
        m.memory.write_u64(heap, 42).unwrap();
        m.core.cpu.regs[0] = heap;
        m.core.cpu.regs[1] = 0x1000;
        m.core.cpu.regs[2] = 1;
        m.svc(2).unwrap();
        m.core.cpu.regs[1] = 0x400000;
        m.svc(1).unwrap();
        assert_eq!(m.memory.read_u64(heap).unwrap(), 42);
        assert!(m.memory.write_u64(heap, 43).is_err());
        m.memory.write_u64(heap + 0x200000, 7).unwrap();
        assert_eq!(m.memory.read_u64(heap + 0x300000).unwrap(), 0);
        m.core.cpu.regs[1] = 0;
        m.svc(1).unwrap();
        assert!(m.memory.read_u64(heap).is_err());
    }

    #[test]
    fn unsupported_svc_and_budget_do_not_fake_success() {
        let mut m = Machine::new(Layout::default(), 1);
        m.memory
            .map(0x1000, 0x1000, READ | EXECUTE, STATE_CODE, "text")
            .unwrap();
        m.core.cpu.pc = 0x1000;
        m.core.cpu.regs[0] = 123;
        assert!(matches!(
            m.svc(0x22),
            Err(Trap::UnsupportedSvc { number: 0x22, .. })
        ));
        assert_eq!(m.core.cpu.regs[0], 123);
        m.tick().unwrap();
        assert!(matches!(m.tick(), Err(Trap::Budget { steps: 1, .. })));
    }

    #[test]
    fn processor_number_is_not_thread_exit() {
        let layout = Layout {
            processor_id: 2,
            ..Layout::default()
        };
        let mut m = Machine::new(layout, 1);
        m.svc(0x10).unwrap();
        assert_eq!(m.core.cpu.regs[0], 2);
        m.svc(0x0a).unwrap();
        assert!(matches!(m.finish_block(), Err(Trap::Exit { code: 0 })));
    }

    #[test]
    fn loader_break_notifications_are_distinct_from_fatal_breaks() {
        let mut m = Machine::new(Layout::default(), 1);
        m.core.cpu.regs[0] = 0x80000003;
        m.svc(0x26).unwrap();
        assert_eq!(m.core.cpu.regs[0], 0x80000003);
        m.core.cpu.regs[0] = 3;
        assert!(matches!(
            m.svc(0x26),
            Err(Trap::GuestBreak { reason: 3, .. })
        ));
    }
}
