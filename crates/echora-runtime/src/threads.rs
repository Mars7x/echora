//! Bounded child-thread lifetimes and full CPU snapshots for runtime dispatch.
//!
//! The architectural initialization follows pinned Atmosphere 6b0c077b4c608a3a2
//! (kern_k_thread_context.cpp / kern_svc_thread.cpp): X0 is the argument,
//! X18 is a nonzero odd random value supplied by the caller, PC clears bit 0,
//! SP rounds down to 16 bytes, and FPCR/FPSR come from the creating thread.
//! The thread-local region is 0x200 bytes with its handle at offset 0x110.
//!
//! One dedicated 4 KiB TLS page per owned child and a 128-thread limit are
//! explicit prototype resource policies, not the kernel's packed TLS allocator
//! or a console ThreadCountMax value. The caller supplies and owns the user
//! stack; creation and destruction never allocate, clear or unmap that stack.

use crate::cpu::Cpu;
use crate::memory::{Memory, PAGE_SIZE, READ, STATE_TLS, WRITE};
use crate::runtime::Trap;
use serde::Serialize;
use std::collections::BTreeMap;

const MAX_OWNED_THREADS: usize = 128;
const TLS_REGION_SIZE: usize = 0x200;
const TLS_HANDLE_OFFSET: usize = 0x110;

/// The SVC layer validates signed core/priority arguments against the process
/// grants and supplies OS randomness/current FPU status before calling create.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ThreadCreate {
    pub(crate) id: u64,
    pub(crate) handle: u32,
    pub(crate) entry: u64,
    pub(crate) argument: u64,
    pub(crate) stack_top: u64,
    pub(crate) priority: u8,
    pub(crate) core: u8,
    pub(crate) x18: u64,
    pub(crate) fpcr: u32,
    pub(crate) fpsr: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ThreadState {
    Initialized,
    Runnable,
    Waiting,
    Terminated,
}

impl ThreadState {
    pub(crate) fn holds_running_reference(self) -> bool {
        matches!(self, Self::Runnable | Self::Waiting)
    }
}

struct InitializedThread {
    // Preserve creation provenance independently of mutable scheduling metadata.
    request: ThreadCreate,
    /// SetThreadPriority changes this; creation priority stays in `request`.
    base_priority: u8,
    effective_priority: u8,
    ideal_core: i32,
    affinity_mask: u64,
    tls_slot: u64,
    tls_address: u64,
    state: ThreadState,
    signaled: bool,
    wait_cancelled: bool,
    handle_open: bool,
    mutex_owner_references: u32,
    logical_processor_id: Option<u8>,
    instructions_started: u64,
    initial_cpu: Cpu,
    // Last dispatch snapshot. Machine.cpu is canonical while this child runs.
    cpu: Cpu,
}

pub(crate) struct ThreadCollection {
    main_thread_id: u64,
    next_id: Option<u64>,
    tls_anchor: Option<u64>,
    threads: BTreeMap<u64, InitializedThread>,
    /// IDs of the Runnable threads, kept with every state change so the
    /// scheduler does not scan every thread on each pick.
    runnable: std::collections::BTreeSet<u64>,
}

/// The logical core a thread runs on: its ideal core, else one of its
/// permitted cores chosen by thread ID. Horizon's scheduler migrates
/// threads without an ideal core onto idle permitted cores; spreading them
/// keeps GetCurrentProcessorNumber distinct among the worker threads that a
/// console would run in parallel (inferred; the console's choice is dynamic).
fn spread_processor(id: u64, ideal_core: i32, affinity_mask: u64) -> u8 {
    if (0..64).contains(&ideal_core) {
        return ideal_core as u8;
    }
    let mut mask = affinity_mask;
    for _ in 0..id % u64::from(affinity_mask.count_ones().max(1)) {
        mask &= mask - 1;
    }
    mask.trailing_zeros() as u8
}

fn invalid(message: impl Into<String>) -> Trap {
    Trap::InvalidState {
        message: message.into(),
    }
}

pub(crate) fn cpu_report(cpu: &Cpu) -> serde_json::Value {
    serde_json::json!({
        "registers":cpu.regs,
        "vector_registers_hex":cpu.vectors.iter().map(|value| format!("{value:032x}")).collect::<Vec<_>>(),
        "pc":cpu.pc, "sp":cpu.sp, "nzcv":cpu.nzcv,
        "tpidr_el0":cpu.tpidr_el0, "tpidrro_el0":cpu.tpidrro_el0,
        "fpcr":cpu.fpcr, "fpsr":cpu.fpsr,
    })
}

impl ThreadCollection {
    pub(crate) fn new(main_id: u64) -> Self {
        Self {
            main_thread_id: main_id,
            next_id: main_id.checked_add(1).filter(|_| main_id != 0),
            tls_anchor: None,
            threads: BTreeMap::new(),
            runnable: Default::default(),
        }
    }

    pub(crate) fn next_id(&self) -> Result<u64, Trap> {
        self.next_id
            .ok_or_else(|| invalid("initialized-thread ID range is unavailable or exhausted"))
    }

    pub(crate) fn len(&self) -> usize {
        self.threads.len()
    }

    pub(crate) fn ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.threads.keys().copied()
    }

    pub(crate) fn state(&self, id: u64) -> Option<ThreadState> {
        self.threads.get(&id).map(|thread| thread.state)
    }

    pub(crate) fn wait_cancelled(&self, id: u64) -> Option<bool> {
        self.threads.get(&id).map(|thread| thread.wait_cancelled)
    }

    /// The cancellation latch belongs to the thread object in every state.
    /// Runtime decides whether cancellation is latched for a future wait or completes
    /// a currently cancellable synchronization wait; mutex waits are separate.
    pub(crate) fn set_wait_cancelled(&mut self, id: u64, cancelled: bool) -> Result<(), Trap> {
        let thread = self
            .threads
            .get_mut(&id)
            .ok_or_else(|| invalid("owned child thread ID does not exist"))?;
        thread.wait_cancelled = cancelled;
        Ok(())
    }

    pub(crate) fn initialized_count(&self) -> usize {
        self.threads
            .values()
            .filter(|thread| thread.state == ThreadState::Initialized)
            .count()
    }

    pub(crate) fn runnable_count(&self) -> usize {
        self.runnable_ids().count()
    }

    pub(crate) fn waiting_count(&self) -> usize {
        self.threads
            .values()
            .filter(|thread| thread.state == ThreadState::Waiting)
            .count()
    }

    pub(crate) fn started_count(&self) -> usize {
        self.threads
            .values()
            .filter(|thread| thread.state.holds_running_reference())
            .count()
    }

    pub(crate) fn terminated_count(&self) -> usize {
        self.threads
            .values()
            .filter(|thread| thread.state == ThreadState::Terminated)
            .count()
    }

    pub(crate) fn runnable_ids(&self) -> impl Iterator<Item = u64> + '_ {
        debug_assert!(self.runnable.iter().all(|id| {
            self.threads
                .get(id)
                .is_some_and(|thread| thread.state == ThreadState::Runnable)
        }));
        self.runnable.iter().copied()
    }

    pub(crate) fn processor_id(&self, id: u64) -> Option<u8> {
        self.threads
            .get(&id)
            .and_then(|thread| thread.logical_processor_id)
    }

    pub(crate) fn contains(&self, id: u64) -> bool {
        self.threads.contains_key(&id)
    }

    pub(crate) fn priority(&self, id: u64) -> Option<u8> {
        self.threads
            .get(&id)
            .map(|thread| thread.effective_priority)
    }

    pub(crate) fn base_priority(&self, id: u64) -> Option<u8> {
        self.threads.get(&id).map(|thread| thread.base_priority)
    }

    /// The caller recomputes and applies effective priorities afterwards.
    pub(crate) fn set_base_priority(&mut self, id: u64, priority: u8) -> Result<(), Trap> {
        let thread = self
            .threads
            .get_mut(&id)
            .ok_or_else(|| invalid("owned child thread ID does not exist"))?;
        if priority >= 64 {
            return Err(invalid(
                "base priority must be a validated virtual priority",
            ));
        }
        thread.base_priority = priority;
        Ok(())
    }

    /// Runtime owns mutex donation and queue ordering. The collection stores
    /// the resulting effective priority without changing creation provenance.
    pub(crate) fn set_effective_priority(&mut self, id: u64, priority: u8) -> Result<(), Trap> {
        let thread = self
            .threads
            .get_mut(&id)
            .ok_or_else(|| invalid("owned child thread ID does not exist"))?;
        if priority > thread.base_priority {
            return Err(invalid(
                "effective priority cannot be lower than the base priority",
            ));
        }
        thread.effective_priority = priority;
        Ok(())
    }

    pub(crate) fn affinity(&self, id: u64) -> Option<(i32, u64)> {
        self.threads
            .get(&id)
            .map(|thread| (thread.ideal_core, thread.affinity_mask))
    }

    /// Store resolved virtual affinity for an owned thread. The SVC layer
    /// owns process grants, special selectors and kernel Result precedence.
    /// A started child moves to its new ideal or permitted core;
    /// this updates no CPU/TLS/lifetime state.
    pub(crate) fn set_affinity(
        &mut self,
        id: u64,
        ideal_core: i32,
        affinity_mask: u64,
    ) -> Result<(), Trap> {
        if affinity_mask == 0 || (ideal_core != -1 && !(0..64).contains(&ideal_core)) {
            return Err(invalid(
                "initialized thread affinity requires a resolved ideal and nonzero mask",
            ));
        }
        if ideal_core >= 0 && affinity_mask & (1u64 << ideal_core) == 0 {
            return Err(invalid(
                "initialized thread ideal core is outside its affinity",
            ));
        }
        let thread = self
            .threads
            .get_mut(&id)
            .ok_or_else(|| invalid("owned child thread ID does not exist"))?;
        if thread.state.holds_running_reference() {
            thread
                .logical_processor_id
                .filter(|processor| *processor < 64)
                .ok_or_else(|| invalid("started child has no valid logical processor"))?;
            // A started thread migrates to its new ideal core, or onto one of
            // its permitted cores (see `spread_processor`).
            thread.logical_processor_id = Some(spread_processor(id, ideal_core, affinity_mask));
        }
        thread.ideal_core = ideal_core;
        thread.affinity_mask = affinity_mask;
        Ok(())
    }

    /// Acquire the modeled running reference without executing guest code.
    /// Runtime owns raw SVC error precedence and schedules only after the
    /// generated caller has stored its post-SVC PC. The logical-core choice is
    /// an explicit virtual compatibility policy, not physical CPU placement.
    pub(crate) fn start(&mut self, id: u64) -> Result<(), Trap> {
        let thread = self
            .threads
            .get_mut(&id)
            .ok_or_else(|| invalid("owned child thread ID does not exist"))?;
        if thread.state != ThreadState::Initialized || !thread.handle_open {
            return Err(invalid("only an open initialized child can be started"));
        }
        if !(0..64).contains(&thread.ideal_core)
            && (thread.ideal_core != -1 || thread.affinity_mask == 0)
        {
            return Err(invalid("child start has invalid resolved affinity"));
        }
        let processor = spread_processor(id, thread.ideal_core, thread.affinity_mask);
        if thread.affinity_mask & (1u64 << processor) == 0 {
            return Err(invalid("child start ideal is outside its affinity"));
        }
        thread.logical_processor_id = Some(processor);
        thread.state = ThreadState::Runnable;
        self.runnable.insert(id);
        Ok(())
    }

    pub(crate) fn saved_cpu(&self, id: u64) -> Result<Cpu, Trap> {
        let thread = self
            .threads
            .get(&id)
            .filter(|thread| thread.state == ThreadState::Runnable)
            .ok_or_else(|| invalid("CPU dispatch requires a runnable child"))?;
        Ok(thread.cpu.clone())
    }

    pub(crate) fn save_cpu(&mut self, id: u64, cpu: &Cpu) -> Result<(), Trap> {
        let thread = self
            .threads
            .get_mut(&id)
            .filter(|thread| thread.state == ThreadState::Runnable)
            .ok_or_else(|| invalid("CPU snapshot requires a runnable child"))?;
        thread.cpu = cpu.clone();
        Ok(())
    }

    /// Runtime invokes this only after the SVC's generated next-PC epilogue.
    /// Waiting retains the started reference and a full continuation snapshot.
    pub(crate) fn block_wait(&mut self, id: u64, post_svc_cpu: &Cpu) -> Result<(), Trap> {
        let thread = self
            .threads
            .get_mut(&id)
            .filter(|thread| thread.state == ThreadState::Runnable)
            .ok_or_else(|| invalid("blocking requires a runnable child"))?;
        thread.cpu = post_svc_cpu.clone();
        thread.state = ThreadState::Waiting;
        self.runnable.remove(&id);
        Ok(())
    }

    /// Complete the blocked SVC without executing or dispatching the thread.
    /// W0 is its only return register; all other continuation state is retained.
    pub(crate) fn complete_wait(&mut self, id: u64, result: u32) -> Result<(), Trap> {
        let thread = self
            .threads
            .get_mut(&id)
            .filter(|thread| thread.state == ThreadState::Waiting)
            .ok_or_else(|| invalid("wait completion requires a waiting child"))?;
        thread.cpu.regs[0] = u64::from(result);
        thread.state = ThreadState::Runnable;
        self.runnable.insert(id);
        Ok(())
    }

    /// Complete WaitSynchronization's two-register return without dispatching.
    /// Runtime validates the wait kind and owns cancellation-latch consumption;
    /// this preserves both that latch and all other continuation state.
    pub(crate) fn complete_synchronization_wait(
        &mut self,
        id: u64,
        result: u32,
        index: i32,
    ) -> Result<(), Trap> {
        let thread = self
            .threads
            .get_mut(&id)
            .filter(|thread| thread.state == ThreadState::Waiting)
            .ok_or_else(|| invalid("synchronization completion requires a waiting child"))?;
        thread.cpu.regs[0] = u64::from(result);
        thread.cpu.regs[1] = u64::from(index as u32);
        thread.state = ThreadState::Runnable;
        self.runnable.insert(id);
        Ok(())
    }

    /// Count the same pre-execution instruction boundary as Machine::tick.
    /// Runtime preserves the process-wide budget and commits its count only
    /// after this fallible child accounting succeeds.
    pub(crate) fn note_instructions(&mut self, id: u64, count: u64) -> Result<(), Trap> {
        let thread = self
            .threads
            .get_mut(&id)
            .filter(|thread| thread.state == ThreadState::Runnable)
            .ok_or_else(|| invalid("instruction accounting requires a runnable child"))?;
        let next = thread
            .instructions_started
            .checked_add(count)
            .ok_or_else(|| invalid("child instruction counter exhausted"))?;
        thread.instructions_started = next;
        Ok(())
    }

    /// Prepare everything before publishing the record/ID. Runtime holds the
    /// shared handle reservation until this succeeds, then commits that handle.
    pub(crate) fn create(
        &mut self,
        memory: &mut Memory,
        tls_anchor: u64,
        request: ThreadCreate,
    ) -> Result<(), Trap> {
        if request.id != self.next_id()? || self.threads.contains_key(&request.id) {
            return Err(invalid(
                "thread creation requires the next unused logical ID",
            ));
        }
        if request.handle == 0
            || self
                .threads
                .values()
                .any(|thread| thread.request.handle == request.handle)
        {
            return Err(invalid(
                "thread creation requires a nonzero unique reserved handle",
            ));
        }
        if request.priority >= 64 || request.core >= 64 {
            return Err(invalid(
                "thread core/priority must already be validated virtual IDs",
            ));
        }
        if self.threads.len() >= MAX_OWNED_THREADS {
            return Err(invalid("prototype owned-child-thread capacity exhausted"));
        }
        if !tls_anchor.is_multiple_of(PAGE_SIZE)
            || self.tls_anchor.is_some_and(|anchor| anchor != tls_anchor)
        {
            return Err(invalid(
                "thread TLS requires one stable page-aligned anchor",
            ));
        }
        // Slot zero is the main thread's page. Reuse only this collection's
        // released slots, never an externally occupied page or the caller stack.
        let tls_slot = (1..=MAX_OWNED_THREADS as u64)
            .find(|slot| self.threads.values().all(|thread| thread.tls_slot != *slot))
            .ok_or_else(|| invalid("prototype owned-child TLS slots exhausted"))?;
        let tls_address = tls_anchor
            .checked_add(tls_slot * PAGE_SIZE)
            .ok_or_else(|| invalid("thread TLS address overflows"))?;
        let tls_end = tls_address
            .checked_add(PAGE_SIZE)
            .filter(|end| *end <= memory.address_space_end())
            .ok_or_else(|| invalid("thread TLS page exceeds the guest address space"))?;
        debug_assert!(tls_address < tls_end);

        let mut cpu = Cpu {
            pc: request.entry & !1,
            sp: request.stack_top & !15,
            tpidrro_el0: tls_address,
            fpcr: request.fpcr,
            fpsr: request.fpsr,
            ..Cpu::default()
        };
        cpu.regs[0] = request.argument;
        cpu.regs[18] = request.x18 | 1;
        let record = InitializedThread {
            request,
            base_priority: request.priority,
            effective_priority: request.priority,
            ideal_core: i32::from(request.core),
            affinity_mask: 1u64 << request.core,
            tls_slot,
            tls_address,
            state: ThreadState::Initialized,
            signaled: false,
            wait_cancelled: false,
            handle_open: true,
            mutex_owner_references: 0,
            logical_processor_id: None,
            instructions_started: 0,
            initial_cpu: cpu.clone(),
            cpu,
        };
        let mut local_region = [0; TLS_REGION_SIZE];
        local_region[TLS_HANDLE_OFFSET..TLS_HANDLE_OFFSET + 4]
            .copy_from_slice(&request.handle.to_le_bytes());
        let label = format!("initialized thread {} TLS", request.id);
        let next_id = request.id.checked_add(1);

        // A page collision is an explicit local resource failure. No bytes are
        // copied from the user stack or a previous thread's released TLS page.
        memory.map(tls_address, PAGE_SIZE, READ | WRITE, STATE_TLS, label)?;
        if let Err(error) = memory.write(tls_address, &local_region) {
            // The new exact mapping cannot have gained aliases in this exclusive
            // operation. Roll it back before exposing the initialization error.
            memory.unmap_exact(tls_address, PAGE_SIZE)?;
            return Err(error);
        }
        self.threads.insert(request.id, record);
        self.next_id = next_id;
        self.tls_anchor = Some(tls_anchor);
        Ok(())
    }

    fn validate_owned_tls(memory: &Memory, address: u64) -> Result<(), Trap> {
        let mapping = memory.query(address);
        if mapping.base != address || mapping.size != PAGE_SIZE || mapping.state != STATE_TLS {
            return Err(invalid("child no longer owns its exact TLS mapping"));
        }
        let end = address
            .checked_add(PAGE_SIZE)
            .ok_or_else(|| invalid("owned child TLS range overflows"))?;
        for (destination, source, size) in memory.stack_aliases() {
            let destination_end = destination
                .checked_add(size)
                .ok_or_else(|| invalid("active alias destination range overflows"))?;
            let source_end = source
                .checked_add(size)
                .ok_or_else(|| invalid("active alias source range overflows"))?;
            if (address < destination_end && destination < end)
                || (address < source_end && source < end)
            {
                return Err(invalid("owned child TLS overlaps an active memory alias"));
            }
        }
        Ok(())
    }

    fn unmap_owned_tls(memory: &mut Memory, address: u64) -> Result<(), Trap> {
        Self::validate_owned_tls(memory, address)?;
        memory.unmap_exact(address, PAGE_SIZE)
    }

    pub(crate) fn mutex_owner_references(&self, id: u64) -> Option<u32> {
        self.threads
            .get(&id)
            .map(|thread| thread.mutex_owner_references)
    }

    /// WaitForAddress retains the original owner until its blocked SVC resumes,
    /// independently of any later queue-ownership transfer. Runtime owns those
    /// associations and calls these methods once for each retained reference.
    pub(crate) fn retain_mutex_owner_reference(&mut self, id: u64) -> Result<(), Trap> {
        let thread = self
            .threads
            .get_mut(&id)
            .ok_or_else(|| invalid("owned child thread ID does not exist"))?;
        let next = thread
            .mutex_owner_references
            .checked_add(1)
            .ok_or_else(|| invalid("mutex owner reference count exhausted"))?;
        thread.mutex_owner_references = next;
        Ok(())
    }

    pub(crate) fn release_mutex_owner_reference(
        &mut self,
        memory: &mut Memory,
        id: u64,
    ) -> Result<(), Trap> {
        let thread = self
            .threads
            .get_mut(&id)
            .ok_or_else(|| invalid("owned child thread ID does not exist"))?;
        if thread.mutex_owner_references == 0 {
            return Err(invalid("child has no retained mutex owner reference"));
        }
        if thread.mutex_owner_references == 1
            && !thread.handle_open
            && !thread.state.holds_running_reference()
        {
            // Validate and release storage before consuming the last reference.
            // Failure leaves both the record and its count intact for diagnosis.
            Self::unmap_owned_tls(memory, thread.tls_address)?;
            self.threads.remove(&id);
            self.runnable.remove(&id);
        } else {
            thread.mutex_owner_references -= 1;
        }
        Ok(())
    }

    /// Release the child handle before runtime removes its typed table entry.
    /// Runnable/Waiting records and mutex-referenced owners retain context and
    /// the unchanged TLS handle field. Cleanup requires every reference gone.
    /// Failed unmapping preserves the record and its open-handle state.
    pub(crate) fn close(&mut self, memory: &mut Memory, id: u64) -> Result<(), Trap> {
        let thread = self
            .threads
            .get_mut(&id)
            .ok_or_else(|| invalid("owned child thread ID does not exist"))?;
        if !thread.handle_open {
            return Err(invalid("child thread handle is already closed"));
        }
        if thread.state.holds_running_reference() || thread.mutex_owner_references != 0 {
            thread.handle_open = false;
            return Ok(());
        }
        Self::unmap_owned_tls(memory, thread.tls_address)?;
        self.threads.remove(&id);
        self.runnable.remove(&id);
        Ok(())
    }

    /// Validate outgoing termination before an incoming waiter consumes its
    /// original-owner reference. Runtime must not change mappings or unrelated
    /// ownership between this preflight and the boundary's final commit.
    pub(crate) fn preflight_terminate(
        &self,
        memory: &Memory,
        id: u64,
        releasing_mutex_references: u32,
    ) -> Result<(), Trap> {
        let thread = self
            .threads
            .get(&id)
            .filter(|thread| thread.state == ThreadState::Runnable)
            .ok_or_else(|| invalid("termination preflight requires a runnable child"))?;
        let remaining = thread
            .mutex_owner_references
            .checked_sub(releasing_mutex_references)
            .ok_or_else(|| {
                invalid("termination would release unavailable mutex owner references")
            })?;
        if !thread.handle_open && remaining == 0 {
            Self::validate_owned_tls(memory, thread.tls_address)?;
        }
        Ok(())
    }

    /// Complete termination only after the runtime has descheduled this guest
    /// context at a completed-block host boundary. A surviving handle or mutex
    /// owner reference retains the signaled terminated object/TLS; otherwise
    /// cleanup precedes removal. Runtime owns queue restrictions/cancellation.
    pub(crate) fn terminate(
        &mut self,
        memory: &mut Memory,
        id: u64,
        cpu: &Cpu,
    ) -> Result<(), Trap> {
        let thread = self
            .threads
            .get_mut(&id)
            .filter(|thread| thread.state == ThreadState::Runnable)
            .ok_or_else(|| invalid("termination requires a runnable child"))?;
        if thread.handle_open || thread.mutex_owner_references != 0 {
            thread.cpu = cpu.clone();
            thread.state = ThreadState::Terminated;
            self.runnable.remove(&id);
            thread.signaled = true;
            return Ok(());
        }
        Self::unmap_owned_tls(memory, thread.tls_address)?;
        self.threads.remove(&id);
        self.runnable.remove(&id);
        Ok(())
    }

    pub(crate) fn report(&self) -> serde_json::Value {
        let threads: Vec<_> = self
            .threads
            .values()
            .map(|thread| {
                // This full local report contains private initial CPU state.
                // Public checkpoints must select numeric metadata explicitly.
                let initial_cpu_context = cpu_report(&thread.initial_cpu);
                let saved_cpu_context = cpu_report(&thread.cpu);
                serde_json::json!({
                    "id":thread.request.id, "handle":thread.request.handle,
                    "entry":thread.request.entry, "argument":thread.request.argument,
                    "stack_top":thread.request.stack_top,
                    "priority":thread.effective_priority, "core":thread.request.core,
                    "creation_priority":thread.request.priority,
                    "base_priority":thread.base_priority,
                    "effective_priority":thread.effective_priority,
                    "creation_core":thread.request.core,
                    "creation_affinity_mask":1u64 << thread.request.core,
                    "ideal_core":thread.ideal_core,
                    "affinity_mask":thread.affinity_mask,
                    "state":thread.state, "signaled":thread.signaled,
                    "wait_cancelled":thread.wait_cancelled,
                    "handle_open":thread.handle_open,
                    "running_reference":thread.state.holds_running_reference(),
                    "mutex_owner_references":thread.mutex_owner_references,
                    "logical_processor_id":thread.logical_processor_id,
                    "instructions_started":thread.instructions_started,
                    "tls_address":thread.tls_address, "tls_slot":thread.tls_slot,
                    "initial_cpu_context":initial_cpu_context,
                    "saved_cpu_context":saved_cpu_context,
                })
            })
            .collect();
        let mut report = serde_json::json!({
            "model":"owned_child_threads_with_dispatch_snapshots",
            "main_thread_id":self.main_thread_id, "next_thread_id":self.next_id,
            "initialized_thread_limit":MAX_OWNED_THREADS,
            "initialized_thread_limit_scope":"shared_with_all_owned_child_states",
            "owned_thread_limit":MAX_OWNED_THREADS,
            "owned_thread_count":self.threads.len(),
            "initialized_thread_count":self.initialized_count(),
            "runnable_thread_count":self.runnable_count(),
            "waiting_thread_count":self.waiting_count(),
            "started_thread_count":self.started_count(),
            "terminated_thread_count":self.terminated_count(),
            "owned_tls_bytes":self.threads.len() as u64 * PAGE_SIZE,
            "tls_anchor":self.tls_anchor, "tls_region_size":TLS_REGION_SIZE,
            "tls_page_size":PAGE_SIZE,
            "tls_allocation_model":"one_owned_page_per_child_until_last_reference",
            "resource_limit_model":"explicit_prototype_bounds_not_console_thread_count_limit",
            "thread_id_policy":"monotonic_without_reuse",
            "threads":threads,
        });
        let lifecycle_scope = serde_json::json!({
            "tls_close_policy":"retain_started_and_mutex_owner_references_until_last_reference",
            "start_thread_implemented":true, "thread_wait_implemented":false,
            "thread_scheduler_implemented":false, "thread_suspension_implemented":false,
            "logical_processor_selection_policy":"ideal_core_else_lowest_set_virtual_bit_on_start",
            "collection_scheduler_role":"context_and_lifetime_storage_runtime_owns_dispatch",
            "private_context_field":"initial_cpu_context",
            "private_context_fields":["initial_cpu_context","saved_cpu_context"],
            "saved_cpu_context_semantics":"last_dispatch_snapshot_not_active_canonical",
        });
        let resource_scope = serde_json::json!({
            "core_field_semantics":"creation_virtual_core",
            "priority_field_semantics":"current_effective_priority",
            "create_thread_implemented":true,
            "kernel_thread_count_quota_implemented":false,
            "resource_reservation_wait_implemented":false,
            "kernel_thread_stack_implemented":false,
            "tls_charged_to_used_user_memory":false,
        });
        for part in [lifecycle_scope, resource_scope] {
            let serde_json::Value::Object(fields) = part else {
                unreachable!("thread report fragments are object literals");
            };
            report
                .as_object_mut()
                .expect("the report is built as a JSON object")
                .extend(fields);
        }
        report
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(all(test, any(target_os = "linux", windows, target_os = "macos")))]
mod tests {
    use super::*;
    use crate::memory::{ATTRIBUTE_LOCKED, STATE_CODE_DATA, STATE_FREE, STATE_NORMAL, STATE_STACK};
    use crate::services::Services;

    const TLS_ANCHOR: u64 = 0x10000;
    const STACK_SOURCE: u64 = 0x180000;
    const STACK_ALIAS: u64 = 0x190000;

    fn fixture() -> (Memory, ThreadCollection, Services) {
        let mut memory = Memory::new(0x200000);
        memory
            .map(TLS_ANCHOR, PAGE_SIZE, READ | WRITE, STATE_TLS, "main TLS")
            .unwrap();
        memory
            .write(TLS_ANCHOR, &[0xa5; PAGE_SIZE as usize])
            .unwrap();
        memory
            .map(
                STACK_SOURCE,
                PAGE_SIZE,
                READ | WRITE,
                STATE_CODE_DATA,
                "caller stack backing",
            )
            .unwrap();
        memory
            .write(STACK_SOURCE, &[0x6d; PAGE_SIZE as usize])
            .unwrap();
        memory
            .map_stack_alias(STACK_ALIAS, STACK_SOURCE, PAGE_SIZE)
            .unwrap();
        (memory, ThreadCollection::new(1), Services::default())
    }

    fn request(id: u64, handle: u32) -> ThreadCreate {
        ThreadCreate {
            id,
            handle,
            entry: 0x12347,
            argument: 0xfedc_ba98_7654_3210,
            stack_top: 0x190ffb,
            priority: 28,
            core: 5,
            x18: 0x0123_4567_89ab_cde0,
            fpcr: 0x07c0_0000,
            fpsr: 0x0800_009f,
        }
    }

    fn read_page(memory: &Memory, address: u64) -> Vec<u8> {
        let mut bytes = vec![0; PAGE_SIZE as usize];
        memory.read(address, &mut bytes).unwrap();
        bytes
    }

    fn create_reserved(
        threads: &mut ThreadCollection,
        memory: &mut Memory,
        services: &mut Services,
    ) -> (u64, u32) {
        let id = threads.next_id().unwrap();
        let reserved = services.reserve_thread_handle(id).unwrap();
        let handle = reserved.handle();
        threads
            .create(memory, TLS_ANCHOR, request(id, handle))
            .unwrap();
        assert_eq!(reserved.commit(), handle);
        (id, handle)
    }

    #[test]
    fn cancellation_latch_follows_owned_thread_lifetime_in_every_state() {
        let (mut memory, mut threads, mut services) = fixture();
        let (id, handle) = create_reserved(&mut threads, &mut memory, &mut services);
        let (other, _) = create_reserved(&mut threads, &mut memory, &mut services);
        let initial = threads.threads[&id].initial_cpu.clone();
        let creation = threads.threads[&id].request;
        let tls = read_page(&memory, 0x11000);
        assert_eq!(threads.wait_cancelled(id), Some(false));
        assert_eq!(threads.wait_cancelled(other), Some(false));

        for state in [
            ThreadState::Initialized,
            ThreadState::Runnable,
            ThreadState::Waiting,
            ThreadState::Terminated,
        ] {
            match state {
                ThreadState::Initialized => {}
                ThreadState::Runnable => {
                    threads.start(id).unwrap();
                    assert_eq!(threads.wait_cancelled(id), Some(true));
                }
                ThreadState::Waiting => {
                    let cpu = threads.saved_cpu(id).unwrap();
                    threads.block_wait(id, &cpu).unwrap();
                    assert_eq!(threads.wait_cancelled(id), Some(true));
                }
                ThreadState::Terminated => {
                    threads.retain_mutex_owner_reference(id).unwrap();
                    threads.close(&mut memory, id).unwrap();
                    assert_eq!(services.release_thread_handle(handle), Some(id));
                    assert_eq!(threads.wait_cancelled(id), Some(true));
                    threads.complete_wait(id, 0).unwrap();
                    assert_eq!(threads.wait_cancelled(id), Some(true));
                    let cpu = threads.saved_cpu(id).unwrap();
                    threads.terminate(&mut memory, id, &cpu).unwrap();
                    assert_eq!(threads.wait_cancelled(id), Some(true));
                }
            }
            let before = threads.report();
            for cancelled in [false, true, true] {
                threads.set_wait_cancelled(id, cancelled).unwrap();
                let mut expected = before.clone();
                expected["threads"][0]["wait_cancelled"] = serde_json::json!(cancelled);
                assert_eq!(threads.report(), expected);
                assert_eq!(threads.wait_cancelled(id), Some(cancelled));
                assert_eq!(threads.wait_cancelled(other), Some(false));
                assert_eq!(threads.state(id), Some(state));
                assert_eq!(threads.threads[&id].initial_cpu, initial);
                assert_eq!(threads.threads[&id].request, creation);
                assert_eq!(read_page(&memory, 0x11000), tls);
            }
            let before_invalid = threads.report();
            for unknown in [0, 1, 99] {
                assert_eq!(threads.wait_cancelled(unknown), None);
                assert!(threads.set_wait_cancelled(unknown, true).is_err());
                assert!(threads.set_wait_cancelled(unknown, false).is_err());
                assert_eq!(threads.report(), before_invalid);
            }
        }
        assert!(!threads.threads[&id].handle_open);
        assert_eq!(threads.mutex_owner_references(id), Some(1));
        threads
            .release_mutex_owner_reference(&mut memory, id)
            .unwrap();
        assert_eq!(threads.wait_cancelled(id), None);
        assert_eq!(memory.query(0x11000).state, STATE_FREE);
        let (new_id, _) = create_reserved(&mut threads, &mut memory, &mut services);
        assert_eq!(new_id, 4);
        assert_eq!(threads.threads[&new_id].tls_address, 0x11000);
        assert_eq!(threads.wait_cancelled(new_id), Some(false));
        assert_eq!(threads.wait_cancelled(other), Some(false));
        assert_eq!(read_page(&memory, TLS_ANCHOR), vec![0xa5; 0x1000]);
        assert_eq!(read_page(&memory, STACK_ALIAS), vec![0x6d; 0x1000]);
    }

    #[test]
    fn synchronization_completion_changes_only_w0_w1_and_mutex_completion_keeps_x1() {
        let (mut memory, mut threads, mut services) = fixture();
        let (id, _) = create_reserved(&mut threads, &mut memory, &mut services);
        let initialized = threads.report();
        assert!(
            threads
                .complete_synchronization_wait(id, 0xec01, -1)
                .is_err()
        );
        assert_eq!(threads.report(), initialized);
        threads.start(id).unwrap();
        let initial = threads.threads[&id].initial_cpu.clone();
        let tls = read_page(&memory, 0x11000);
        let continuation = Cpu {
            regs: std::array::from_fn(|index| 0xfedc_ba98_7654_0000 | index as u64),
            vectors: std::array::from_fn(|index| u128::MAX - index as u128),
            pc: 0x334454,
            sp: 0x190ed0,
            nzcv: 0xd,
            tpidr_el0: 0x1122_3344_5566_7788,
            tpidrro_el0: 0x11000,
            fpcr: 0x0040_0000,
            fpsr: 0x17,
        };
        threads.set_wait_cancelled(id, true).unwrap();
        for (result, index, expected_index) in
            [(0xec01, -1, 0xffff_ffff), (u32::MAX, i32::MIN, 0x8000_0000)]
        {
            threads.block_wait(id, &continuation).unwrap();
            let waiting = threads.report();
            assert!(
                threads
                    .complete_synchronization_wait(99, result, index)
                    .is_err()
            );
            assert_eq!(threads.report(), waiting);
            threads
                .complete_synchronization_wait(id, result, index)
                .unwrap();
            let mut expected = continuation.clone();
            expected.regs[0] = u64::from(result);
            expected.regs[1] = expected_index;
            assert_eq!(threads.saved_cpu(id).unwrap(), expected);
            assert_eq!(threads.state(id), Some(ThreadState::Runnable));
            assert_eq!(threads.wait_cancelled(id), Some(true));
            assert_eq!(threads.threads[&id].initial_cpu, initial);
            assert_eq!(threads.threads[&id].instructions_started, 0);
            let completed = threads.report();
            assert!(
                threads
                    .complete_synchronization_wait(id, result, index)
                    .is_err()
            );
            assert_eq!(threads.report(), completed);
        }
        threads.block_wait(id, &continuation).unwrap();
        threads.complete_wait(id, 0xd401).unwrap();
        let mut expected = continuation;
        expected.regs[0] = 0xd401;
        assert_eq!(expected.regs[1], 0xfedc_ba98_7654_0001);
        assert_eq!(threads.saved_cpu(id).unwrap(), expected);
        assert_eq!(threads.wait_cancelled(id), Some(true));
        assert_eq!(threads.threads[&id].initial_cpu, initial);
        assert_eq!(read_page(&memory, 0x11000), tls);
        assert_eq!(read_page(&memory, TLS_ANCHOR), vec![0xa5; 0x1000]);
        assert_eq!(read_page(&memory, STACK_ALIAS), vec![0x6d; 0x1000]);
    }

    #[test]
    fn waiting_preserves_full_continuation_and_closed_handle_until_result_completion() {
        let (mut memory, mut threads, mut services) = fixture();
        let (id, handle) = create_reserved(&mut threads, &mut memory, &mut services);
        let (other, _) = create_reserved(&mut threads, &mut memory, &mut services);
        threads.start(id).unwrap();
        threads.start(other).unwrap();
        let initial = threads.threads[&id].initial_cpu.clone();
        let initial_request = threads.threads[&id].request;
        let tls = read_page(&memory, 0x11000);
        let mut continuation = initial.clone();
        continuation.regs = std::array::from_fn(|index| 0xaabb_ccdd_0000_0000 | index as u64);
        continuation.vectors = std::array::from_fn(|index| u128::MAX - index as u128);
        continuation.sp = 0x190ed0;
        continuation.nzcv = 0xd;
        continuation.tpidr_el0 = 0x1122_3344_5566_7788;
        continuation.fpcr = 0x0040_0000;
        continuation.fpsr = 0x17;

        for (index, result) in [0, 0xd401, u32::MAX].into_iter().enumerate() {
            continuation.pc = 0x222004 + index as u64 * 8;
            threads.block_wait(id, &continuation).unwrap();
            assert_eq!(threads.state(id), Some(ThreadState::Waiting));
            assert_eq!(threads.waiting_count(), 1);
            assert_eq!(threads.started_count(), 2);
            assert_eq!(threads.runnable_ids().collect::<Vec<_>>(), vec![other]);
            assert_eq!(threads.processor_id(id), Some(5));
            assert_eq!(threads.threads[&id].cpu, continuation);
            let waiting = threads.report();
            assert_eq!(waiting["threads"][0]["state"], "waiting");
            assert_eq!(waiting["threads"][0]["running_reference"], true);
            assert_eq!(waiting["threads"][0]["signaled"], false);
            assert!(threads.saved_cpu(id).is_err());
            assert!(threads.save_cpu(id, &initial).is_err());
            assert!(threads.note_instructions(id, 1).is_err());
            assert!(threads.block_wait(id, &initial).is_err());
            assert!(threads.terminate(&mut memory, id, &initial).is_err());
            assert!(threads.complete_wait(other, 0).is_err());
            assert!(threads.complete_wait(99, 0).is_err());
            assert_eq!(threads.report(), waiting);

            if index == 0 {
                threads.close(&mut memory, id).unwrap();
                assert_eq!(services.release_thread_handle(handle), Some(id));
            }
            assert!(!threads.threads[&id].handle_open);
            assert_eq!(memory.read_u32(0x11110).unwrap(), handle);
            assert_eq!(read_page(&memory, 0x11000), tls);
            threads.complete_wait(id, result).unwrap();
            let mut expected = continuation.clone();
            expected.regs[0] = u64::from(result);
            assert_eq!(threads.saved_cpu(id).unwrap(), expected);
            assert_eq!(threads.state(id), Some(ThreadState::Runnable));
            assert_eq!(threads.waiting_count(), 0);
            assert_eq!(threads.started_count(), 2);
            assert_eq!(threads.runnable_ids().collect::<Vec<_>>(), vec![id, other]);
            assert_eq!(threads.threads[&id].initial_cpu, initial);
            assert_eq!(threads.threads[&id].request, initial_request);
            assert_eq!(threads.threads[&id].instructions_started, 0);
            let completed = threads.report();
            assert!(threads.complete_wait(id, result).is_err());
            assert_eq!(threads.report(), completed);
        }
        assert_eq!(read_page(&memory, TLS_ANCHOR), vec![0xa5; 0x1000]);
        assert_eq!(read_page(&memory, STACK_ALIAS), vec![0x6d; 0x1000]);
    }

    #[test]
    fn mutex_owner_references_retain_all_states_independently_of_handles_and_running() {
        let (mut memory, mut threads, mut services) = fixture();
        let mut records = Vec::new();
        for state in [
            ThreadState::Initialized,
            ThreadState::Runnable,
            ThreadState::Waiting,
            ThreadState::Terminated,
        ] {
            let (id, handle) = create_reserved(&mut threads, &mut memory, &mut services);
            if state != ThreadState::Initialized {
                threads.start(id).unwrap();
                let cpu = threads.saved_cpu(id).unwrap();
                if state == ThreadState::Waiting {
                    threads.block_wait(id, &cpu).unwrap();
                } else if state == ThreadState::Terminated {
                    threads.terminate(&mut memory, id, &cpu).unwrap();
                }
            }
            let address = threads.threads[&id].tls_address;
            let tls = read_page(&memory, address);
            threads.retain_mutex_owner_reference(id).unwrap();
            threads.retain_mutex_owner_reference(id).unwrap();
            threads.close(&mut memory, id).unwrap();
            assert_eq!(services.release_thread_handle(handle), Some(id));
            assert_eq!(threads.state(id), Some(state));
            assert_eq!(threads.mutex_owner_references(id), Some(2));
            assert_eq!(read_page(&memory, address), tls);
            assert!(!threads.threads[&id].handle_open);
            let before = threads.report();
            assert!(threads.close(&mut memory, id).is_err());
            assert!(threads.start(id).is_err());
            assert_eq!(threads.report(), before);
            threads
                .release_mutex_owner_reference(&mut memory, id)
                .unwrap();
            assert_eq!(threads.mutex_owner_references(id), Some(1));
            assert_eq!(read_page(&memory, address), tls);
            records.push((id, state, address, tls));
        }
        assert_eq!(
            (
                threads.len(),
                threads.started_count(),
                threads.waiting_count()
            ),
            (4, 2, 1)
        );
        for (id, state, address, tls) in records {
            threads
                .release_mutex_owner_reference(&mut memory, id)
                .unwrap();
            if state.holds_running_reference() {
                assert_eq!(threads.mutex_owner_references(id), Some(0));
                assert_eq!(read_page(&memory, address), tls);
                if state == ThreadState::Waiting {
                    threads.complete_wait(id, 0).unwrap();
                }
                let cpu = threads.saved_cpu(id).unwrap();
                threads.terminate(&mut memory, id, &cpu).unwrap();
            }
            assert!(!threads.contains(id));
            assert_eq!(memory.query(address).state, STATE_FREE);
        }
        assert_eq!(
            (
                threads.len(),
                threads.started_count(),
                threads.waiting_count()
            ),
            (0, 0, 0)
        );
        assert_eq!(read_page(&memory, TLS_ANCHOR), vec![0xa5; 0x1000]);
        assert_eq!(read_page(&memory, STACK_ALIAS), vec![0x6d; 0x1000]);
    }

    #[test]
    fn mutex_owner_reference_bounds_and_failed_final_cleanup_preserve_ownership() {
        for terminate_owner in [false, true] {
            let (mut memory, mut threads, mut services) = fixture();
            let (id, handle) = create_reserved(&mut threads, &mut memory, &mut services);
            let before = threads.report();
            assert!(
                threads
                    .release_mutex_owner_reference(&mut memory, id)
                    .is_err()
            );
            assert!(threads.retain_mutex_owner_reference(99).is_err());
            assert!(
                threads
                    .release_mutex_owner_reference(&mut memory, 99)
                    .is_err()
            );
            assert_eq!(threads.mutex_owner_references(99), None);
            assert_eq!(threads.report(), before);
            threads.threads.get_mut(&id).unwrap().mutex_owner_references = u32::MAX;
            let saturated = threads.report();
            assert!(threads.retain_mutex_owner_reference(id).is_err());
            assert_eq!(threads.report(), saturated);
            threads.threads.get_mut(&id).unwrap().mutex_owner_references = 0;
            // Releasing a reference while a handle remains must retain TLS.
            threads.retain_mutex_owner_reference(id).unwrap();
            threads
                .release_mutex_owner_reference(&mut memory, id)
                .unwrap();
            assert_eq!(threads.report(), before);
            threads.retain_mutex_owner_reference(id).unwrap();
            if terminate_owner {
                threads.start(id).unwrap();
            }
            threads.close(&mut memory, id).unwrap();
            assert_eq!(services.release_thread_handle(handle), Some(id));
            if terminate_owner {
                let mut final_cpu = threads.saved_cpu(id).unwrap();
                final_cpu.pc = 0x333304;
                threads.terminate(&mut memory, id, &final_cpu).unwrap();
                assert_eq!(threads.state(id), Some(ThreadState::Terminated));
                assert_eq!(threads.threads[&id].cpu, final_cpu);
                assert!(threads.threads[&id].signaled);
            }
            assert_eq!(threads.mutex_owner_references(id), Some(1));
            let tls = read_page(&memory, 0x11000);
            memory.unmap_exact(0x11000, PAGE_SIZE).unwrap();
            memory
                .map(
                    0x11000,
                    PAGE_SIZE,
                    READ | WRITE,
                    STATE_NORMAL,
                    "unrelated replacement",
                )
                .unwrap();
            memory.write_u64(0x11000, 0x7654_3210_fedc_ba98).unwrap();
            let before = threads.report();
            let mappings = memory.mappings().cloned().collect::<Vec<_>>();
            assert!(
                threads
                    .release_mutex_owner_reference(&mut memory, id)
                    .is_err()
            );
            assert_eq!(threads.report(), before);
            assert_eq!(threads.mutex_owner_references(id), Some(1));
            assert_eq!(memory.mappings().cloned().collect::<Vec<_>>(), mappings);
            assert_eq!(memory.read_u64(0x11000).unwrap(), 0x7654_3210_fedc_ba98);
            memory.unmap_exact(0x11000, PAGE_SIZE).unwrap();
            memory
                .map(
                    0x11000,
                    PAGE_SIZE,
                    READ | WRITE,
                    STATE_TLS,
                    "restored owned TLS",
                )
                .unwrap();
            memory.write(0x11000, &tls).unwrap();
            threads
                .release_mutex_owner_reference(&mut memory, id)
                .unwrap();
            assert!(!threads.contains(id));
            assert_eq!(memory.query(0x11000).state, STATE_FREE);
            assert_eq!(threads.next_id().unwrap(), 3);
            assert_eq!(read_page(&memory, TLS_ANCHOR), vec![0xa5; 0x1000]);
            assert_eq!(read_page(&memory, STACK_ALIAS), vec![0x6d; 0x1000]);
        }
    }

    #[test]
    fn termination_preflight_accounts_for_incoming_owner_release_without_mutation() {
        let (mut memory, mut threads, mut services) = fixture();
        let (id, handle) = create_reserved(&mut threads, &mut memory, &mut services);
        let initialized = threads.report();
        assert!(threads.preflight_terminate(&memory, id, 0).is_err());
        assert_eq!(threads.report(), initialized);
        threads.start(id).unwrap();
        let final_cpu = threads.saved_cpu(id).unwrap();
        threads.retain_mutex_owner_reference(id).unwrap();
        let open = threads.report();
        threads.preflight_terminate(&memory, id, 0).unwrap();
        threads.preflight_terminate(&memory, id, 1).unwrap();
        assert!(threads.preflight_terminate(&memory, id, 2).is_err());
        assert!(threads.preflight_terminate(&memory, 99, 0).is_err());
        assert_eq!(threads.report(), open);
        threads.close(&mut memory, id).unwrap();
        assert_eq!(services.release_thread_handle(handle), Some(id));
        let tls = read_page(&memory, 0x11000);
        let before = threads.report();
        threads.preflight_terminate(&memory, id, 0).unwrap();
        threads.preflight_terminate(&memory, id, 1).unwrap();
        assert_eq!(threads.report(), before);
        assert_eq!(read_page(&memory, 0x11000), tls);

        memory.unmap_exact(0x11000, PAGE_SIZE).unwrap();
        memory
            .map(
                0x11000,
                PAGE_SIZE,
                READ | WRITE,
                STATE_NORMAL,
                "unrelated replacement",
            )
            .unwrap();
        memory.write_u64(0x11000, 0x1234_5678_9abc_def0).unwrap();
        let mappings = memory.mappings().cloned().collect::<Vec<_>>();
        let aliases = memory.stack_aliases().collect::<Vec<_>>();
        let before = threads.report();
        // Keeping the owner reference needs no cleanup, but dispatching the
        // waiter would release it. Detect that final cleanup failure first.
        threads.preflight_terminate(&memory, id, 0).unwrap();
        assert!(threads.preflight_terminate(&memory, id, 1).is_err());
        assert_eq!(threads.report(), before);
        assert_eq!(threads.mutex_owner_references(id), Some(1));
        assert_eq!(threads.saved_cpu(id).unwrap(), final_cpu);
        assert_eq!(threads.ids().collect::<Vec<_>>(), vec![id]);
        assert_eq!(memory.mappings().cloned().collect::<Vec<_>>(), mappings);
        assert_eq!(memory.stack_aliases().collect::<Vec<_>>(), aliases);
        assert_eq!(memory.read_u64(0x11000).unwrap(), 0x1234_5678_9abc_def0);
        threads.retain_mutex_owner_reference(id).unwrap();
        let retained = threads.report();
        threads.preflight_terminate(&memory, id, 1).unwrap();
        assert!(threads.preflight_terminate(&memory, id, 2).is_err());
        assert!(threads.preflight_terminate(&memory, id, 3).is_err());
        assert_eq!(threads.report(), retained);
        threads
            .release_mutex_owner_reference(&mut memory, id)
            .unwrap();

        memory.unmap_exact(0x11000, PAGE_SIZE).unwrap();
        memory
            .map(
                0x11000,
                PAGE_SIZE,
                READ | WRITE,
                STATE_TLS,
                "restored owned TLS",
            )
            .unwrap();
        memory.write(0x11000, &tls).unwrap();
        let before = threads.report();
        threads.preflight_terminate(&memory, id, 1).unwrap();
        assert_eq!(threads.report(), before);
        threads
            .release_mutex_owner_reference(&mut memory, id)
            .unwrap();
        threads.terminate(&mut memory, id, &final_cpu).unwrap();
        assert!(threads.ids().next().is_none());
        assert_eq!(memory.query(0x11000).state, STATE_FREE);
        assert_eq!(read_page(&memory, TLS_ANCHOR), vec![0xa5; 0x1000]);
        assert_eq!(read_page(&memory, STACK_ALIAS), vec![0x6d; 0x1000]);
    }

    #[test]
    fn effective_priority_is_separate_from_creation_and_updates_no_context_or_lifetime() {
        let (mut memory, mut threads, mut services) = fixture();
        let (id, _) = create_reserved(&mut threads, &mut memory, &mut services);
        let (other, _) = create_reserved(&mut threads, &mut memory, &mut services);
        let initial = threads.threads[&id].initial_cpu.clone();
        let request = threads.threads[&id].request;
        let tls = read_page(&memory, 0x11000);
        assert_eq!(threads.base_priority(99), None);
        for state in [
            ThreadState::Initialized,
            ThreadState::Runnable,
            ThreadState::Waiting,
            ThreadState::Terminated,
        ] {
            match state {
                ThreadState::Initialized => {}
                ThreadState::Runnable => threads.start(id).unwrap(),
                ThreadState::Waiting => {
                    let cpu = threads.saved_cpu(id).unwrap();
                    threads.block_wait(id, &cpu).unwrap();
                }
                ThreadState::Terminated => {
                    threads.complete_wait(id, 0).unwrap();
                    let cpu = threads.saved_cpu(id).unwrap();
                    threads.terminate(&mut memory, id, &cpu).unwrap();
                }
            }
            let before = threads.report();
            for priority in [0, 7, 28] {
                threads.set_effective_priority(id, priority).unwrap();
                assert_eq!(threads.priority(id), Some(priority));
                assert_eq!(threads.base_priority(id), Some(28));
                assert_eq!(threads.priority(other), Some(28));
                assert_eq!(threads.state(id), Some(state));
                let mut expected = before.clone();
                expected["threads"][0]["priority"] = serde_json::json!(priority);
                expected["threads"][0]["effective_priority"] = serde_json::json!(priority);
                assert_eq!(threads.report(), expected);
                assert_eq!(threads.threads[&id].request, request);
                assert_eq!(threads.threads[&id].initial_cpu, initial);
                assert_eq!(read_page(&memory, 0x11000), tls);
            }
            let before_invalid = threads.report();
            for priority in [29, 63, 64, 255] {
                assert!(threads.set_effective_priority(id, priority).is_err());
                assert_eq!(threads.report(), before_invalid);
            }
            assert!(threads.set_effective_priority(99, 0).is_err());
            assert_eq!(threads.report(), before_invalid);
        }
        let report = threads.report();
        assert_eq!(
            report["priority_field_semantics"],
            "current_effective_priority"
        );
        assert_eq!(report["threads"][0]["creation_priority"], 28);
        assert_eq!(report["threads"][0]["base_priority"], 28);
        assert_eq!(report["threads"][0]["effective_priority"], 28);
    }

    #[test]
    fn start_selects_virtual_processor_and_full_snapshots_preserve_initial_context() {
        let (mut memory, mut threads, mut services) = fixture();
        let (first, _) = create_reserved(&mut threads, &mut memory, &mut services);
        let (second, _) = create_reserved(&mut threads, &mut memory, &mut services);
        let initial = threads.threads[&first].initial_cpu.clone();
        let initial_report = threads.report()["threads"][0]["initial_cpu_context"].clone();
        let first_tls = read_page(&memory, 0x11000);
        let second_tls = read_page(&memory, 0x12000);
        assert_eq!(threads.state(first), Some(ThreadState::Initialized));
        assert_eq!(threads.state(1), None);
        assert_eq!(threads.processor_id(first), None);
        assert_eq!(
            (
                threads.initialized_count(),
                threads.runnable_count(),
                threads.terminated_count()
            ),
            (2, 0, 0)
        );
        let before = threads.report();
        assert!(threads.saved_cpu(first).is_err());
        assert!(threads.save_cpu(first, &initial).is_err());
        assert!(threads.note_instructions(first, 1).is_err());
        assert!(threads.terminate(&mut memory, first, &initial).is_err());
        assert!(threads.start(99).is_err());
        assert_eq!(threads.report(), before);

        threads
            .set_affinity(first, -1, (1u64 << 5) | (1u64 << 63))
            .unwrap();
        threads.set_affinity(second, 63, 1 | (1u64 << 63)).unwrap();
        // Iteration order is by logical ID, independent of start order.
        threads.start(second).unwrap();
        threads.start(first).unwrap();
        assert_eq!(threads.runnable_ids().collect::<Vec<_>>(), vec![2, 3]);
        assert_eq!(
            (
                threads.initialized_count(),
                threads.runnable_count(),
                threads.terminated_count()
            ),
            (0, 2, 0)
        );
        assert_eq!(threads.processor_id(first), Some(5));
        assert_eq!(threads.processor_id(second), Some(63));
        assert_eq!(threads.saved_cpu(first).unwrap(), initial);
        let started = threads.report();
        assert!(threads.start(first).is_err());
        assert_eq!(threads.report(), started);

        let mut advanced = initial.clone();
        advanced.regs = std::array::from_fn(|index| 0xfeed_0000_0000_0000 | index as u64);
        advanced.vectors =
            std::array::from_fn(|index| (1u128 << (96 + index)) | (index as u128 + 7));
        advanced.pc = 0x223344;
        advanced.sp = 0x190f00;
        advanced.nzcv = 0xb;
        advanced.tpidr_el0 = 0x1234_5678_9abc_def0;
        advanced.tpidrro_el0 = 0x11000;
        advanced.fpcr = 0x0040_0000;
        advanced.fpsr = 0x15;
        threads.save_cpu(first, &advanced).unwrap();
        threads.note_instructions(first, 1).unwrap();
        threads.note_instructions(first, 1).unwrap();
        assert_eq!(threads.saved_cpu(first).unwrap(), advanced);
        let mut detached = threads.saved_cpu(first).unwrap();
        detached.regs[0] ^= 1;
        assert_ne!(threads.saved_cpu(first).unwrap(), detached);
        assert_eq!(threads.threads[&first].initial_cpu, initial);
        assert_eq!(
            threads.threads[&second].cpu,
            threads.threads[&second].initial_cpu
        );
        let report = threads.report();
        assert_eq!(report["threads"][0]["initial_cpu_context"], initial_report);
        assert_eq!(report["threads"][0]["saved_cpu_context"]["pc"], 0x223344);
        assert_eq!(report["threads"][0]["instructions_started"], 2);
        assert_eq!(report["threads"][0]["running_reference"], true);
        assert_eq!(report["threads"][0]["signaled"], false);
        assert_eq!(
            report["saved_cpu_context_semantics"],
            "last_dispatch_snapshot_not_active_canonical"
        );
        assert_eq!(read_page(&memory, 0x11000), first_tls);
        assert_eq!(read_page(&memory, 0x12000), second_tls);
    }

    #[test]
    fn runnable_affinity_migrates_and_terminated_affinity_preserves_final_context() {
        let (mut memory, mut threads, mut services) = fixture();
        let (id, _) = create_reserved(&mut threads, &mut memory, &mut services);
        threads.start(id).unwrap();
        let initial = threads.saved_cpu(id).unwrap();
        threads.set_affinity(id, 2, (1 << 2) | (1 << 5)).unwrap();
        assert_eq!(threads.processor_id(id), Some(2));
        assert_eq!(threads.affinity(id), Some((2, 0x24)));
        // Without an ideal core the thread lands on a permitted core.
        threads.set_affinity(id, -1, 1 << 5).unwrap();
        assert_eq!(threads.processor_id(id), Some(5));
        assert_eq!(threads.saved_cpu(id).unwrap(), initial);

        let mut final_cpu = initial.clone();
        final_cpu.pc = 0x334450;
        final_cpu.regs[8] = 0xcafe;
        threads.terminate(&mut memory, id, &final_cpu).unwrap();
        threads.set_affinity(id, 2, 1 << 2).unwrap();
        assert_eq!(threads.state(id), Some(ThreadState::Terminated));
        assert_eq!(threads.affinity(id), Some((2, 4)));
        assert_eq!(threads.processor_id(id), Some(5)); // Last selected, not executing.
        assert_eq!(threads.threads[&id].cpu, final_cpu);
        assert_eq!(threads.threads[&id].initial_cpu, initial);
        assert!(threads.threads[&id].signaled);
    }

    #[test]
    fn running_reference_survives_handle_close_until_descheduled_termination() {
        let (mut memory, mut threads, mut services) = fixture();
        let (id, handle) = create_reserved(&mut threads, &mut memory, &mut services);
        let (other, _) = create_reserved(&mut threads, &mut memory, &mut services);
        threads.start(id).unwrap();
        let mut cpu = threads.saved_cpu(id).unwrap();
        cpu.pc = 0x445560;
        cpu.regs[2] = 0x9876;
        threads.save_cpu(id, &cpu).unwrap();
        threads.note_instructions(id, 1).unwrap();
        memory.write_u64(0x11300, 0x7654_3210_fedc_ba98).unwrap();
        let tls = read_page(&memory, 0x11000);
        let other_tls = read_page(&memory, 0x12000);
        threads.close(&mut memory, id).unwrap();
        assert_eq!(services.release_thread_handle(handle), Some(id));
        assert_eq!(services.thread_id(handle), None);
        assert_eq!(threads.state(id), Some(ThreadState::Runnable));
        assert_eq!(threads.saved_cpu(id).unwrap(), cpu);
        assert_eq!(threads.priority(id), Some(28));
        assert_eq!(threads.runnable_ids().collect::<Vec<_>>(), vec![id]);
        assert!(!threads.threads[&id].handle_open);
        assert!(!threads.threads[&id].signaled);
        assert_eq!(read_page(&memory, 0x11000), tls);
        assert_eq!(memory.read_u32(0x11110).unwrap(), handle);
        let before = threads.report();
        assert!(threads.close(&mut memory, id).is_err());
        assert_eq!(threads.report(), before);
        threads.note_instructions(id, 1).unwrap();
        assert_eq!(threads.threads[&id].instructions_started, 2);

        cpu.pc = 0x445580;
        threads.terminate(&mut memory, id, &cpu).unwrap();
        assert!(!threads.contains(id));
        assert_eq!(threads.runnable_count(), 0);
        assert_eq!(threads.initialized_count(), 1);
        assert_eq!(threads.state(other), Some(ThreadState::Initialized));
        assert_eq!(memory.query(0x11000).state, STATE_FREE);
        assert_eq!(read_page(&memory, 0x12000), other_tls);
        let (next_id, next_handle) = create_reserved(&mut threads, &mut memory, &mut services);
        assert_eq!((next_id, next_handle), (4, 0x202));
        assert_eq!(threads.threads[&next_id].tls_address, 0x11000);
        let mut cleared = vec![0; 0x1000];
        cleared[0x110..0x114].copy_from_slice(&next_handle.to_le_bytes());
        assert_eq!(read_page(&memory, 0x11000), cleared);
        assert_eq!(read_page(&memory, TLS_ANCHOR), vec![0xa5; 0x1000]);
        assert_eq!(read_page(&memory, STACK_ALIAS), vec![0x6d; 0x1000]);
    }

    #[test]
    fn terminated_handle_retains_identity_tls_and_signal_until_successful_final_close() {
        let (mut memory, mut threads, mut services) = fixture();
        let (id, handle) = create_reserved(&mut threads, &mut memory, &mut services);
        threads.start(id).unwrap();
        let initial = threads.threads[&id].initial_cpu.clone();
        let mut final_cpu = threads.saved_cpu(id).unwrap();
        final_cpu.pc = 0x556670;
        final_cpu.nzcv = 6;
        let tls = read_page(&memory, 0x11000);
        threads.terminate(&mut memory, id, &final_cpu).unwrap();
        assert_eq!(
            (
                threads.initialized_count(),
                threads.runnable_count(),
                threads.terminated_count()
            ),
            (0, 0, 1)
        );
        assert_eq!(services.thread_id(handle), Some(id));
        assert_eq!(threads.priority(id), Some(28));
        assert_eq!(threads.threads[&id].cpu, final_cpu);
        assert_eq!(threads.threads[&id].initial_cpu, initial);
        assert!(threads.threads[&id].handle_open);
        assert!(threads.threads[&id].signaled);
        assert_eq!(read_page(&memory, 0x11000), tls);
        let before = threads.report();
        assert_eq!(before["threads"][0]["running_reference"], false);
        assert!(threads.start(id).is_err());
        assert!(threads.saved_cpu(id).is_err());
        assert!(threads.save_cpu(id, &initial).is_err());
        assert!(threads.note_instructions(id, 1).is_err());
        assert!(threads.terminate(&mut memory, id, &initial).is_err());
        assert_eq!(threads.report(), before);

        memory.unmap_exact(0x11000, PAGE_SIZE).unwrap();
        memory
            .map(
                0x11000,
                PAGE_SIZE,
                READ | WRITE,
                STATE_NORMAL,
                "unrelated replacement",
            )
            .unwrap();
        memory.write_u64(0x11000, 0x1111_2222_3333_4444).unwrap();
        assert!(threads.close(&mut memory, id).is_err());
        assert_eq!(threads.report(), before);
        assert_eq!(services.thread_id(handle), Some(id));
        assert_eq!(memory.read_u64(0x11000).unwrap(), 0x1111_2222_3333_4444);
        memory.unmap_exact(0x11000, PAGE_SIZE).unwrap();
        memory
            .map(
                0x11000,
                PAGE_SIZE,
                READ | WRITE,
                STATE_TLS,
                "restored owned TLS",
            )
            .unwrap();
        memory.write(0x11000, &tls).unwrap();
        threads.close(&mut memory, id).unwrap();
        assert_eq!(services.release_thread_handle(handle), Some(id));
        assert_eq!(threads.len(), 0);
        assert_eq!(threads.terminated_count(), 0);
        assert_eq!(memory.query(0x11000).state, STATE_FREE);
        assert_eq!(read_page(&memory, TLS_ANCHOR), vec![0xa5; 0x1000]);
        assert_eq!(read_page(&memory, STACK_ALIAS), vec![0x6d; 0x1000]);
    }

    #[test]
    fn accounting_overflow_and_handleless_termination_cleanup_failure_are_atomic() {
        let (mut memory, mut threads, mut services) = fixture();
        let (id, handle) = create_reserved(&mut threads, &mut memory, &mut services);
        threads.start(id).unwrap();
        threads.threads.get_mut(&id).unwrap().instructions_started = u64::MAX;
        let before = threads.report();
        assert!(threads.note_instructions(id, 1).is_err());
        assert_eq!(threads.report(), before);
        threads.close(&mut memory, id).unwrap();
        assert_eq!(services.release_thread_handle(handle), Some(id));
        let original_cpu = threads.saved_cpu(id).unwrap();
        let mut final_cpu = original_cpu.clone();
        final_cpu.pc = 0x667780;
        let tls = read_page(&memory, 0x11000);
        memory.unmap_exact(0x11000, PAGE_SIZE).unwrap();
        memory
            .map(
                0x11000,
                PAGE_SIZE,
                READ | WRITE,
                STATE_NORMAL,
                "unrelated replacement",
            )
            .unwrap();
        memory.write_u64(0x11000, 0xaaaa_bbbb_cccc_dddd).unwrap();
        let before = threads.report();
        let mappings = memory.mappings().cloned().collect::<Vec<_>>();
        assert!(threads.terminate(&mut memory, id, &final_cpu).is_err());
        assert_eq!(threads.report(), before);
        assert_eq!(threads.saved_cpu(id).unwrap(), original_cpu);
        assert_eq!(memory.mappings().cloned().collect::<Vec<_>>(), mappings);
        assert_eq!(memory.read_u64(0x11000).unwrap(), 0xaaaa_bbbb_cccc_dddd);
        memory.unmap_exact(0x11000, PAGE_SIZE).unwrap();
        memory
            .map(
                0x11000,
                PAGE_SIZE,
                READ | WRITE,
                STATE_TLS,
                "restored owned TLS",
            )
            .unwrap();
        memory.write(0x11000, &tls).unwrap();
        threads.terminate(&mut memory, id, &final_cpu).unwrap();
        assert_eq!(threads.len(), 0);
        assert_eq!(threads.runnable_count(), 0);
        assert_eq!(memory.query(0x11000).state, STATE_FREE);
        assert_eq!(read_page(&memory, TLS_ANCHOR), vec![0xa5; 0x1000]);
        assert_eq!(read_page(&memory, STACK_ALIAS), vec![0x6d; 0x1000]);
    }

    #[test]
    fn affinity_updates_preserve_creation_context_tls_and_other_thread_state() {
        let (mut memory, mut threads, mut services) = fixture();
        let (id, handle) = create_reserved(&mut threads, &mut memory, &mut services);
        let (other_id, _) = create_reserved(&mut threads, &mut memory, &mut services);
        memory.write_u64(0x11300, 0x1020_3040_5060_7080).unwrap();
        let initial_request = threads.threads[&id].request;
        let initial_cpu = threads.threads[&id].cpu.clone();
        let initial_report = threads.report();
        let initial_services = services.report();
        let mappings = memory.mappings().cloned().collect::<Vec<_>>();
        let tls = read_page(&memory, 0x11000);
        let other_tls = read_page(&memory, 0x12000);
        assert_eq!(threads.affinity(id), Some((5, 0x20)));
        assert_eq!(threads.affinity(1), None);

        // Include a nontrivial mask, signed DontCare, and bit 63. The helper
        // deliberately does not enforce process grants owned by the SVC layer.
        for (ideal, mask) in [
            (2, 0x2c),
            (-1, 0x8000_0000_0000_0001),
            (63, 0x8000_0000_0000_0000),
            (0, u64::MAX),
        ] {
            threads.set_affinity(id, ideal, mask).unwrap();
            assert_eq!(threads.affinity(id), Some((ideal, mask)));
            threads.set_affinity(id, ideal, mask).unwrap();
            let mut expected = initial_report.clone();
            expected["threads"][0]["ideal_core"] = serde_json::json!(ideal);
            expected["threads"][0]["affinity_mask"] = serde_json::json!(mask);
            assert_eq!(threads.report(), expected);
            let thread = &threads.threads[&id];
            assert_eq!(thread.request, initial_request);
            assert_eq!(thread.cpu, initial_cpu);
            assert_eq!(threads.priority(id), Some(28));
            assert_eq!(threads.affinity(other_id), Some((5, 0x20)));
            assert_eq!(threads.next_id().unwrap(), 4);
            assert_eq!(services.thread_id(handle), Some(id));
            assert_eq!(services.report(), initial_services);
            assert_eq!(memory.mappings().cloned().collect::<Vec<_>>(), mappings);
            assert_eq!(read_page(&memory, 0x11000), tls);
            assert_eq!(read_page(&memory, 0x12000), other_tls);
            assert_eq!(read_page(&memory, TLS_ANCHOR), vec![0xa5; 0x1000]);
            assert_eq!(read_page(&memory, STACK_ALIAS), vec![0x6d; 0x1000]);
        }
        let report = threads.report();
        assert_eq!(report["core_field_semantics"], "creation_virtual_core");
        assert_eq!(report["threads"][0]["core"], 5);
        assert_eq!(report["threads"][0]["creation_core"], 5);
        assert_eq!(report["threads"][0]["creation_affinity_mask"], 32);
    }

    #[test]
    fn invalid_or_unknown_thread_affinity_updates_are_atomic() {
        let (mut memory, mut threads, mut services) = fixture();
        let (id, _) = create_reserved(&mut threads, &mut memory, &mut services);
        threads.set_affinity(id, 2, 0x14).unwrap();
        let before = threads.report();
        let before_services = services.report();
        let request = threads.threads[&id].request;
        let cpu = threads.threads[&id].cpu.clone();
        let tls = read_page(&memory, 0x11000);
        let mappings = memory.mappings().cloned().collect::<Vec<_>>();
        for (target, ideal, mask) in [
            (id, -2, 4),
            (id, -3, 4),
            (id, -4, 4),
            (id, i32::MIN, 4),
            (id, i32::MAX, u64::MAX),
            (id, 64, u64::MAX),
            (id, -1, 0),
            (id, 0, 0),
            (id, 2, 8),
            (id, 63, 1 << 31),
            (1, 2, 4),
            (99, 2, 4),
        ] {
            assert!(matches!(
                threads.set_affinity(target, ideal, mask),
                Err(Trap::InvalidState { .. })
            ));
            assert_eq!(threads.report(), before);
            assert_eq!(threads.affinity(id), Some((2, 0x14)));
            assert_eq!(threads.threads[&id].request, request);
            assert_eq!(threads.threads[&id].cpu, cpu);
            assert_eq!(services.report(), before_services);
            assert_eq!(memory.mappings().cloned().collect::<Vec<_>>(), mappings);
            assert_eq!(read_page(&memory, 0x11000), tls);
        }
        assert_eq!(threads.affinity(99), None);
    }

    #[test]
    fn creation_initializes_separate_context_and_exact_tls_without_touching_user_stack() {
        let (mut memory, mut threads, mut services) = fixture();
        let resident = memory.resident_bytes();
        let (id, handle) = create_reserved(&mut threads, &mut memory, &mut services);
        assert_eq!((id, handle), (2, 0x200));
        assert_eq!(threads.len(), 1);
        assert_eq!(threads.next_id().unwrap(), 3);
        assert_eq!(threads.priority(id), Some(28));
        assert_eq!(threads.priority(1), None);
        assert!(!threads.contains(1));
        assert!(threads.contains(id));
        let thread = &threads.threads[&id];
        let mut expected = Cpu {
            pc: 0x12346, // Clear only bit 0; bit 1 is retained.
            sp: 0x190ff0,
            tpidrro_el0: 0x11000,
            fpcr: 0x07c0_0000,
            fpsr: 0x0800_009f,
            ..Cpu::default()
        };
        expected.regs[0] = 0xfedc_ba98_7654_3210;
        expected.regs[18] = 0x0123_4567_89ab_cde1;
        assert_eq!(thread.cpu, expected);
        assert!(!thread.signaled);
        let mapping = memory.query(0x11000);
        assert_eq!((mapping.base, mapping.size), (0x11000, 0x1000));
        assert_eq!(
            (mapping.state, mapping.permissions, mapping.attributes),
            (STATE_TLS, 3, 0)
        );
        let mut expected_page = vec![0; 0x1000];
        expected_page[0x110..0x114].copy_from_slice(&0x200u32.to_le_bytes());
        assert_eq!(read_page(&memory, 0x11000), expected_page);
        assert_eq!(read_page(&memory, TLS_ANCHOR), vec![0xa5; 0x1000]);
        assert_eq!(read_page(&memory, STACK_ALIAS), vec![0x6d; 0x1000]);
        assert_eq!(memory.query(STACK_SOURCE).attributes, ATTRIBUTE_LOCKED);
        assert_eq!(memory.query(STACK_ALIAS).state, STATE_STACK);
        assert_eq!(memory.resident_bytes(), resident + PAGE_SIZE);
        let report = threads.report();
        assert_eq!(report["threads"][0]["state"], "initialized");
        assert_eq!(report["threads"][0]["signaled"], false);
        assert_eq!(report["threads"][0]["affinity_mask"], 32);
        assert_eq!(
            report["threads"][0]["initial_cpu_context"]["registers"][18],
            0x0123_4567_89ab_cde1u64
        );
        assert_eq!(report["owned_tls_bytes"], 4096);
        assert_eq!(report["tls_region_size"], 512);
        assert_eq!(report["start_thread_implemented"], true);
        assert_eq!(report["thread_scheduler_implemented"], false);
        assert_eq!(report["create_thread_implemented"], true);
        assert_eq!(report["kernel_thread_count_quota_implemented"], false);
        assert_eq!(report["tls_charged_to_used_user_memory"], false);
    }

    #[test]
    fn dormant_close_reuses_only_cleared_tls_slots_and_keeps_other_contexts_and_stacks() {
        let (mut memory, mut threads, mut services) = fixture();
        let (first, first_handle) = create_reserved(&mut threads, &mut memory, &mut services);
        let (second, second_handle) = create_reserved(&mut threads, &mut memory, &mut services);
        assert_eq!((first, second), (2, 3));
        assert_eq!(threads.threads[&first].tls_address, 0x11000);
        assert_eq!(threads.threads[&second].tls_address, 0x12000);
        memory.write_u64(0x11300, 0xdead_beef_cafe_f00d).unwrap();
        memory.write_u64(0x12300, 0x1122_3344_5566_7788).unwrap();
        let second_cpu = threads.threads[&second].cpu.clone();
        let second_tls = read_page(&memory, 0x12000);

        threads.close(&mut memory, first).unwrap();
        assert_eq!(services.release_thread_handle(first_handle), Some(first));
        assert_eq!(memory.query(0x11000).state, STATE_FREE);
        assert!(threads.close(&mut memory, first).is_err());
        assert!(threads.contains(second));
        assert_eq!(services.thread_id(second_handle), Some(second));
        assert_eq!(threads.threads[&second].cpu, second_cpu);
        assert_eq!(read_page(&memory, 0x12000), second_tls);

        let (third, third_handle) = create_reserved(&mut threads, &mut memory, &mut services);
        assert_eq!((third, third_handle), (4, 0x202));
        assert_eq!(threads.threads[&third].tls_address, 0x11000);
        let mut expected_page = vec![0; 0x1000];
        expected_page[0x110..0x114].copy_from_slice(&0x202u32.to_le_bytes());
        assert_eq!(read_page(&memory, 0x11000), expected_page);
        assert_eq!(read_page(&memory, 0x12000), second_tls);
        assert_eq!(read_page(&memory, TLS_ANCHOR), vec![0xa5; 0x1000]);
        assert_eq!(read_page(&memory, STACK_ALIAS), vec![0x6d; 0x1000]);
        assert_eq!(
            memory.stack_aliases().collect::<Vec<_>>(),
            vec![(STACK_ALIAS, STACK_SOURCE, PAGE_SIZE)]
        );
    }

    #[test]
    fn failed_creation_rolls_back_tls_and_consumes_neither_identity_nor_handle() {
        for case in 0..9 {
            let (mut memory, mut threads, mut services) = fixture();
            let mut candidate = request(2, 0x200);
            let mut anchor = TLS_ANCHOR;
            match case {
                0 => candidate.id = 3,
                1 => candidate.handle = 0,
                2 => candidate.priority = 64,
                3 => candidate.core = 64,
                4 => anchor += 1,
                5 => anchor = !(PAGE_SIZE - 1),
                6 => anchor = 0x200000 - PAGE_SIZE,
                7 => {
                    memory
                        .map(
                            0x11000,
                            PAGE_SIZE,
                            READ | WRITE,
                            STATE_NORMAL,
                            "external occupant",
                        )
                        .unwrap();
                    memory.write(0x11000, &[0x9c; PAGE_SIZE as usize]).unwrap();
                }
                8 => {
                    // Force the fallible initialization after map succeeds.
                    memory.protect_translated_code(0x11000, PAGE_SIZE).unwrap();
                }
                _ => unreachable!(),
            }
            let before_threads = threads.report();
            let before_services = services.report();
            let before_mappings = memory.mappings().cloned().collect::<Vec<_>>();
            let before_resident = memory.resident_bytes();
            {
                let reserved = services.reserve_thread_handle(2).unwrap();
                assert_eq!(reserved.handle(), 0x200);
                assert!(
                    matches!(
                        threads.create(&mut memory, anchor, candidate),
                        Err(Trap::InvalidState { .. })
                    ),
                    "case {case}"
                );
            }
            assert_eq!(threads.report(), before_threads, "case {case}");
            assert_eq!(services.report(), before_services, "case {case}");
            assert_eq!(
                memory.mappings().cloned().collect::<Vec<_>>(),
                before_mappings,
                "case {case}"
            );
            assert_eq!(memory.resident_bytes(), before_resident, "case {case}");
            assert_eq!(read_page(&memory, TLS_ANCHOR), vec![0xa5; 0x1000]);
            assert_eq!(read_page(&memory, STACK_ALIAS), vec![0x6d; 0x1000]);
            if case == 7 {
                assert_eq!(read_page(&memory, 0x11000), vec![0x9c; 0x1000]);
            } else {
                assert_eq!(memory.query(0x11000).state, STATE_FREE);
            }
            assert_eq!(threads.next_id().unwrap(), 2);
            assert_eq!(services.reserve_thread_handle(2).unwrap().handle(), 0x200);
        }
    }

    #[test]
    fn duplicate_identity_anchor_changes_and_failed_close_preserve_live_ownership() {
        let (mut memory, mut threads, mut services) = fixture();
        let (id, handle) = create_reserved(&mut threads, &mut memory, &mut services);
        for (anchor, candidate) in [
            (TLS_ANCHOR, request(2, 0x201)),
            (TLS_ANCHOR, request(3, handle)),
            (TLS_ANCHOR + PAGE_SIZE, request(3, 0x201)),
        ] {
            let before = threads.report();
            let mappings = memory.mappings().cloned().collect::<Vec<_>>();
            assert!(threads.create(&mut memory, anchor, candidate).is_err());
            assert_eq!(threads.report(), before);
            assert_eq!(memory.mappings().cloned().collect::<Vec<_>>(), mappings);
        }

        // Simulate an invalid host-side mapping replacement. Destruction must
        // not remove an unrelated mapping or forget the still-owned record.
        memory.unmap_exact(0x11000, PAGE_SIZE).unwrap();
        memory
            .map(
                0x11000,
                PAGE_SIZE,
                READ | WRITE,
                STATE_NORMAL,
                "replacement",
            )
            .unwrap();
        memory.write_u64(0x11000, 0x5566_7788_99aa_bbcc).unwrap();
        let before = threads.report();
        let mappings = memory.mappings().cloned().collect::<Vec<_>>();
        assert!(threads.close(&mut memory, id).is_err());
        assert_eq!(threads.report(), before);
        assert_eq!(memory.mappings().cloned().collect::<Vec<_>>(), mappings);
        assert_eq!(memory.read_u64(0x11000).unwrap(), 0x5566_7788_99aa_bbcc);
        assert_eq!(services.thread_id(handle), Some(id));
        assert!(!services.close(handle));

        memory.unmap_exact(0x11000, PAGE_SIZE).unwrap();
        memory
            .map(
                0x11000,
                PAGE_SIZE,
                READ | WRITE,
                STATE_TLS,
                "restored owned TLS",
            )
            .unwrap();
        threads.close(&mut memory, id).unwrap();
        assert_eq!(services.release_thread_handle(handle), Some(id));
        assert_eq!(threads.len(), 0);
        assert_eq!(threads.next_id().unwrap(), 3);
        assert_eq!(read_page(&memory, TLS_ANCHOR), vec![0xa5; 0x1000]);
        assert_eq!(read_page(&memory, STACK_ALIAS), vec![0x6d; 0x1000]);
    }

    #[test]
    fn owned_thread_limit_includes_waiting_runnable_and_retained_terminated_children() {
        let (mut memory, mut threads, _) = fixture();
        for index in 0..128 {
            threads
                .create(
                    &mut memory,
                    TLS_ANCHOR,
                    request(index + 2, 0x200 + index as u32),
                )
                .unwrap();
        }
        assert_eq!(threads.len(), 128);
        assert_eq!(threads.next_id().unwrap(), 130);
        assert_eq!(threads.report()["owned_tls_bytes"], 128 * 4096);
        threads.start(2).unwrap();
        let final_cpu = threads.saved_cpu(2).unwrap();
        threads.terminate(&mut memory, 2, &final_cpu).unwrap();
        threads.start(3).unwrap();
        threads.close(&mut memory, 3).unwrap();
        threads.start(4).unwrap();
        let waiting_cpu = threads.saved_cpu(4).unwrap();
        threads.block_wait(4, &waiting_cpu).unwrap();
        threads.close(&mut memory, 4).unwrap();
        assert_eq!(
            (
                threads.initialized_count(),
                threads.runnable_count(),
                threads.waiting_count(),
                threads.terminated_count(),
            ),
            (125, 1, 1, 1)
        );
        assert_eq!(threads.started_count(), 2);
        assert!(!threads.threads[&3].handle_open);
        assert!(!threads.threads[&4].handle_open);
        assert!(threads.threads[&2].handle_open);
        assert_eq!(threads.len(), 128);
        let before = threads.report();
        let mappings = memory.mappings().cloned().collect::<Vec<_>>();
        let resident = memory.resident_bytes();
        assert!(
            threads
                .create(&mut memory, TLS_ANCHOR, request(130, 0x280))
                .is_err()
        );
        assert_eq!(threads.report(), before);
        assert_eq!(memory.mappings().cloned().collect::<Vec<_>>(), mappings);
        assert_eq!(memory.resident_bytes(), resident);
        threads.close(&mut memory, 65).unwrap();
        assert_eq!(memory.query(0x50000).state, STATE_FREE);
        threads
            .create(&mut memory, TLS_ANCHOR, request(130, 0x280))
            .unwrap();
        assert_eq!(threads.threads[&130].tls_address, 0x50000);
        assert!(!threads.contains(65));
        assert_eq!(threads.next_id().unwrap(), 131);
        assert_eq!(threads.len(), 128);
        assert_eq!(
            (
                threads.initialized_count(),
                threads.runnable_count(),
                threads.waiting_count(),
                threads.terminated_count(),
            ),
            (125, 1, 1, 1)
        );
        assert_eq!(threads.started_count(), 2);
    }

    #[test]
    fn thread_id_exhaustion_never_wraps_and_entry_stack_mapping_is_not_checked() {
        assert!(ThreadCollection::new(0).next_id().is_err());
        assert!(ThreadCollection::new(u64::MAX).next_id().is_err());
        let (mut memory, _, mut services) = fixture();
        let mut threads = ThreadCollection::new(u64::MAX - 1);
        assert_eq!(threads.next_id().unwrap(), u64::MAX);
        let reserved = services.reserve_thread_handle(u64::MAX).unwrap();
        let mut candidate = request(u64::MAX, reserved.handle());
        candidate.entry = u64::MAX;
        candidate.stack_top = u64::MAX;
        candidate.x18 = 0;
        threads.create(&mut memory, TLS_ANCHOR, candidate).unwrap();
        let handle = reserved.commit();
        let context = &threads.threads[&u64::MAX].cpu;
        assert_eq!(context.pc, u64::MAX - 1);
        assert_eq!(context.sp, u64::MAX - 15);
        assert_eq!(context.regs[18], 1);
        assert!(threads.next_id().is_err());
        let before = threads.report();
        assert!(
            threads
                .create(&mut memory, TLS_ANCHOR, request(0, handle + 1))
                .is_err()
        );
        assert_eq!(threads.report(), before);
        threads.close(&mut memory, u64::MAX).unwrap();
        assert_eq!(services.release_thread_handle(handle), Some(u64::MAX));
        assert!(threads.next_id().is_err());
        assert_eq!(threads.len(), 0);
        assert_eq!(read_page(&memory, TLS_ANCHOR), vec![0xa5; 0x1000]);
        assert_eq!(read_page(&memory, STACK_ALIAS), vec![0x6d; 0x1000]);
    }
}
