# Multi-core review: echora-runtime workers and synchronisation

A read-only review of how `echora-runtime` runs guest threads on several
workers (`ECHORA_WORKERS`), looking for causes of the stall with two
workers and the guest memory faults with three. Line numbers are for
`crates/echora-runtime/src/runtime.rs` at the commit that adds this file.

What is already ruled out: `tests/stress.rs` (aot-tests) runs CAS, LL/SC
and LDADD claims, a ticket lock, a seqlock and message passing (STLR/LDAR
and DMB) on two to four Cores sharing one Memory on host threads,
preempted at random points, for millions of iterations, in every build
configuration. The generated code and echora-guest's memory primitives
are correct for those patterns.

## The design, briefly

Each worker runs its guest thread's generated code without a lock and
takes the one machine mutex for everything else (`Worker::run`, from
line 4393). The worker's `Core` is swapped into `machine.core` for host
work (`swap_in` / `swap_out`, 4668). Every SVC, its wait and the parking
of the waiter happen inside one `boundary` call under that mutex
(`boundary`, 2324: `svc` and then `finish_block` at 2361). Only guest
code and `nvnQueueSubmitCommands` (`submit_unlocked`, 4603) run outside
it.

## Findings, most likely first

### 1. With fewer than six workers, threads on different guest cores share one strict-priority run queue (likely the two-worker stall)

Worker `w` serves every guest core `c` with `c % W == w`
(`worker_guest_cores`, 1778, 1791). With two workers, cores 0, 2 and 4
are worker 0's. `next_runnable_thread` (1643) then chooses among *all*
threads allowed on those cores by strict priority: `best` (1692) is the
minimum priority over every allowed runnable thread, the current one
included, and only threads at that priority are picked.
`request_preemptions` (1728, 1762) only preempts for a strictly better
priority.

On Horizon each of those cores has its own run queue, so a priority-44
thread pinned to core 2 runs while a priority-28 thread pinned to core 0
spins or works. Here they share worker 0. As long as the core-0 thread
stays runnable (busy, or spinning on a flag the core-2 thread is to set),
the core-2 thread never runs: a livelock on the guest's terms, and a
stall with fewer guest instructions retired by useful threads. With one
worker, `priority()` returns 0 for every thread (1683: `worker_core` is
None), so scheduling is round-robin and nothing starves, which matches
"one worker boots, two stall".

Spin back-off cannot help: `pick(true)` skips spinning threads only
among those at `best` priority.

The same collapse also turns preemption-disabled spins into deadlocks.
A thread whose TLS DisableCount is set is never preempted at a slice end
(`user_preemption_disabled`, 1316, honoured at 2047). If it waits, with
preemption disabled, for a thread pinned to another guest core of the
same worker, that thread can never run.

How to confirm: run with `ECHORA_WORKERS=6`. Each worker then serves
exactly one guest core, and per-worker strict priority is Horizon's
per-core scheduling. If two workers stall and six do not, this is it.
`ECHORA_FLAT_PRIORITY=1` with two workers should also unstick it, since
it is round-robin within the worker.

Fix options:
- always run six workers (one per application core) and let the host
  time-slice them when it has fewer CPUs;
- or keep a run queue per guest core inside each worker: at each slice
  end, rotate among the worker's guest cores and run the best-priority
  runnable thread of the next one. Only threads of the *same* guest core
  compete by priority, and DisableCount holds the worker only for its own
  core's slice.

### 2. Per-core data collisions are guarded; keep them so

`GetCurrentProcessorNumber` (SVC 0x10, 3412) answers
`guest_core_on_worker`, a core of the running worker that the thread's
mask names. Every thread that is told core `c` runs on `c`'s worker, so
two threads told the same core never run at once. The migration path
saves the live context before another worker may resume the thread
(2069), and `swap_out` counts any double dispatch (4676). No defect found.

Two cases still deserve an assertion, since the game indexes per-core
data by the answer without locks:
- a thread whose mask names no core of its current worker (it is told
  the first core of its mask, which belongs to another worker, while it
  runs here);
- a thread told core `c` that keeps running on the same worker after
  SetThreadCoreMask removes `c`, until its next boundary.

If finding 1 is fixed by running six workers, both cases go away.

### 3. Unlocked NVN submission reads guest memory while other workers change mappings (candidate for the three-worker faults)

`submit_unlocked` (4603) executes command buffers through the
submitting worker's guest view with the machine unlocked, while other
workers' SVCs may unmap, shrink or reprotect memory (UnmapMemory,
SetHeapSize, transfer and code memory). If the game frees or reuses
memory a submission still reads, synchronised by something the emulator
completes before the submit finishes, the host read faults. The fault
is then reported as a guest memory fault on the submitting worker.

How to confirm: rerun the three-worker case with `ECHORA_LOCKED_SUBMIT=1`
(the submit then runs under the machine lock). If the faults stop, this
is it. The faulting address, compared with the last unmap or permission
change of that page (trace them), would show which.

### 4. Kernel writes of the mutex and condition key words are plain

`wait_condition_result` (1223) writes the condition key (1) and the
released mutex word with `Memory::write_u32`, a plain copy through the
physical view, while other workers may run LDAXR/STLXR or CAS on those
words. Atmosphere also writes them with plain stores, and an emulated
STXR fails whenever the value changed (its CAS compares with the value
LDXR saw). So this is correct for every value change. Prefer
`guest_cas_u32` (1335) or an atomic store anyway, for data-race freedom:
a plain copy racing a host atomic on the same word is undefined in
Rust's model.

## Checked and found sound

- Check-then-sleep: WaitForAddress (types 0 to 3, the 64-bit type 3
  included), WaitProcessWideKeyAtomic and ArbitrateLock read the guest
  word, decide, register the waiter and park it in one `boundary` under
  the mutex (2324, 2361, 1910). SignalToAddress, SignalProcessWideKey and
  ArbitrateUnlock take the same mutex, so a signal is never lost between
  a waiter's check and its sleep. Guest stores by other workers are
  ordered by the guest protocol (store, then signal SVC), as on
  hardware. The decrement and update forms use `guest_cas_u32`.
- Lost wake-ups across workers: a wake that makes a thread runnable
  while workers are parked notifies them (`spare`, 4503). Parked workers
  also recheck every 2 ms (`PARKED_RECHECK`, 4405). That is a latency
  bound, not a loss.
- Shared state: scheduler queues, handle tables, thread lists, TLS, the
  services and the NVN state are all inside the machine and reached
  under its mutex. During an unlocked submission the NVN state is moved
  out, and other NVN calls wait for it (`nvn_back`); `has_entry` keeps
  working on the placeholder (4617).
- The exclusive monitor is cleared on every context switch and SVC, as
  Arm requires.
- echora-guest's process globals: the `MONITOR` buckets are atomic
  (spurious STXR failures only), `WINDOWS` is a `OnceLock`, and the
  fault handler's view globals are set once.

## Part 2: what differs from one worker when guest code is serialised

The main session's runs crash the same way with `ECHORA_SERIAL_GUEST=1`
(only one worker runs guest code at a time), also with
`ECHORA_FLAT_PRIORITY` and `ECHORA_ANY_CORE`, so simultaneous execution
is not needed. The aot-tests stress suite, now with a CASPAL Treiber
stack and torn-half checks of 128-bit LDP/STP and LDR/STR Q, also rules
out the 128-bit forms: CASP is a real 128-bit atomic
(`GuestView::compare_exchange_pair_single_thread`, portable-atomic's
cmpxchg16b). LDXP and STXP are not decoded at all, so they would stop the
run instead of tearing.

Line numbers below are for `runtime.rs` at the commit that adds this
part.

### What is the same as with one worker (checked)

- The context save and restore covers the whole `Cpu`: general and
  vector registers, SP, PC, NZCV, TPIDR_EL0/TPIDRRO_EL0, FPCR and FPSR.
  It is saved at every switch away (round robin 2129-2134, waits
  1938-1943 and 1969-1973, an ineligible current thread before its
  worker parks 2069-2081), and `next == current` never rewinds (2085).
  Split-window register locals are stored back into the `Core` before
  any trap leaves generated code, so the saved `Cpu` is complete.
- SVC results reach the right thread. `boundary` (2341-2344) runs the
  SVC and sets PC + 4 on the core that issued it. Context switches only
  happen afterwards, in `finish_block` and `idle_until_runnable`
  (`commit_context_switch` is called only at 801, 1948, 1978, 2043 and
  2125). Wake-ups write results into saved contexts only (`complete_wait`
  in threads.rs:361, `wake_ready_waiters` 720), never into a worker's
  live core, even when a parked worker's stale core is swapped in.
- `host_call_continue` is set and taken within one boundary under the
  lock (2573, 2384).
- The exclusive monitor is cleared at every switch (`commit_context_switch`
  1896), on whichever worker's core is swapped in.
- The physical counter: every worker's copy has the same epoch and
  clamps to its own maximum, so host-time CNTVCT stays monotonic when a
  thread moves between workers. (With `ECHORA_VIRTUAL_CLOCK` it follows
  each worker's step count, which `swap_in` keeps in step.)

### What still differs, most likely first

**A. Unlocked NVN submission (W>1 only, unaffected by the serial
switches).** With more than one worker every `nvnQueueSubmitCommands`
runs in `submit_unlocked` (4603), not in `call_nvn`. That means:
- through the worker's flat guest view, `GuestMemory::View`
  (4697), instead of `GuestMemory::Checked(&mut self.memory)` (2650). The
  view bypasses `Memory`'s permission checks and its resident-page
  bookkeeping, so a submission reading or writing memory the game has
  just unmapped or reprotected does not fail cleanly; it faults or writes
  freed pages;
- with the machine unlocked and the serial guest lock not held. Other
  threads run guest code and SVCs (unmap, heap, transfer memory) during
  the submission, which `ECHORA_SERIAL_GUEST` does not prevent;
- with the NVN state moved out (4617). Guest threads that don't call NVN
  but read what a submission writes (semaphore releases, query results,
  fence values in guest memory) see it written at a different point than
  with one worker.
A submission that writes a semaphore or query result through a stale
address, or into memory another thread has reused, corrupts objects the
way the crash looks: a pointer with a wrong high half in an object being
destroyed.
Test: `ECHORA_LOCKED_SUBMIT=1` with `ECHORA_SERIAL_GUEST=1` and long
slices.

**B. GetCurrentProcessorNumber depends on the worker, even with
`ECHORA_ANY_CORE`.** SVC 0x10 (3412) answers `guest_core_on_worker` as
soon as there are workers. `ECHORA_ANY_CORE` only changes which threads
a worker may pick (1671); it doesn't change this answer. With one worker
it is `thread_processor_id` (1606), the thread's own fixed logical core.
So with several workers, one thread's core number changes as it moves
between workers, and threads that would each always report their own
core now report the worker's. Per-core data the game indexes by this
number (thread-local caches, per-core free lists and job queues, which
it touches without locks because "only the thread on core c uses slot
c") then gets used by different threads in a different order. It also
gets used by one thread under two indices across a migration. A per-core
free list or object pool with an entry pushed to one core's slot and
popped from another's, interleaved by a slice end, gives exactly
double-freed or half-initialised objects in a destructor path.
Test: answer SVC 0x10 with `thread_processor_id` also when parallel
(add a diagnostic switch), with `ECHORA_SERIAL_GUEST=1`. A cleaner fix
keeps a thread on the worker that owns its recorded core.

**C. One `Core` per worker.** Some state lives in the worker's Core, not
the thread's `Cpu`, and stays with the worker when the thread moves:
`spin_polls`, `slice_start`, `host_check_at`, `attention`, `call_depth`
and `block_tag`. All of them are reset or recomputed at each boundary
(`finish_block` 2222-2228; `call_depth` unwinds with its frames, and
`block_tag` is settled at every boundary). None of it is guest-visible.
Listed for completeness; unlikely.

### A bisection ladder

Each step removes one W>1-only difference. The first step at which the
crash stops names the cause:

1. `ECHORA_WORKERS=2 ECHORA_SERIAL_GUEST=1 ECHORA_ONE_CORE=1`: every
   thread on worker 0 (1661), so the processor numbers are worker 0's,
   and worker 1 only parks. What still differs from W=1: unlocked submit
   and per-worker cores.
2. Add `ECHORA_LOCKED_SUBMIT=1`: removes A.
3. If 1 already passes, remove `ECHORA_ONE_CORE` and keep
   `ECHORA_LOCKED_SUBMIT=1`. If it then crashes, B (the processor
   numbers) or migration is the cause.

## Part 3: cutting time under the machine lock

At W=6, host work under the one machine mutex is about 40% of wall time.
The main session gave only that total. The shares below are estimates
from what each path does and how often it runs in a rendering game, so
step 0 is to measure them. Line numbers are for `runtime.rs` at the
commit that adds this part.

### 0. Measure it (an hour, no behaviour change)

`Worker::run` already times each hold (`lock_stats`, 4552;
`thread_time`, 4555). Add the held time per boundary kind, the index
`boundary` already computes for `boundary_kinds` (SVC number,
0x80 NVN host call, 0x81 slice end), and per NVN entry name for kind
0x80. Then the plan below can be ranked by data.

### What takes the lock, and what could take less

Most likely first.

**1. NVN host calls other than submit: est. 40-60% of lock time.**
Every NVN entry (command-buffer recording, state setters, texture and
sampler descriptors, queries, present) is a MissingBlock at a thunk,
handled in `boundary` → `call_nvn` (2568) under the machine lock, through
`GuestMemory::Checked(&mut self.memory)` (2700). A frame records
thousands of these. `submit_unlocked` (4707) already shows the state can
leave the machine. Plan: keep `nvn` behind its own mutex in `Shared`, and
run every NVN call under that lock only, reaching guest memory through
the worker's guest view, as submit does, plus a read lock on the memory
layout (see "must stay serialised"). The few NVN paths that touch other
machine state, such as present's `publish_controller` and the
display/vsync events, take the machine lock briefly for that part only.
Texture uploads and shader translation then run beside other threads'
SVCs instead of in front of them.

**2. Slice ends: est. 10-25%.**
Every host boundary takes the lock even when nothing changes: a slice
end, a spin-hint end, or a preemption flag. Under it the worker:
- runs `flush_core_counts` (2336) and `host_ready` (2189), which every
  `WAKE_CHECK_INSTRUCTIONS` rescans every wait, refreshes the syncpoint
  cache and advances audio;
- runs `finish_block`'s scheduler, with several passes over all
  runnable threads: `next_runnable_thread` twice (best, then pick),
  `request_preemptions` (4586, allocating a Vec at 1759 and cloning
  `core_threads` at 1765), and `runnable_on_any_core` (one more pass);
- checks the entry PC against the mapping tree (2447);
- updates `thread_time`, a hash-map lookup per hold.
Plan: a lock-free fast path. Keep an atomic scheduler epoch, bumped by
anything that can change the decision (a thread made runnable or
blocked, a priority or affinity change, a new deadline, a stop request).
At a slice end, a worker whose epoch is unchanged, whose thread has no
pending exit or wait, and whose next deadline has not passed extends
`host_check_at` and continues without the lock. Keep the wait rescan on
its own timer outside the per-boundary path. Spinning threads, which end
slices every 32 polls, then cost nothing until another thread changes
something.

**3. Synchronisation SVCs: est. 10-20%.**
These are WaitForAddress and SignalToAddress (0x34/0x35),
ArbitrateLock/Unlock (0x1a/0x1b), WaitProcessWideKeyAtomic and
SignalProcessWideKey (0x1c/0x1d), WaitSynchronization, and
Signal/Clear/ResetEvent (0x11/0x12/0x17). They need the thread states,
the wait sets and the mutex queues, and nothing else. Plan: a scheduler
lock (threads, `synchronization_waits`, `mutex_waits`, priorities,
`running`, `core_threads`) separate from services, NVN and memory. A
signal by one thread then no longer waits behind another thread's IPC
or NVN call.

**4. IPC, SendSyncRequest (0x21, 3701): est. 5-15%.**
Services (hid, audio, vi, fs, time) are reached under the machine lock.
Plan: the services behind their own mutex, or one per service. Most
requests touch only their service, the handle table and guest memory
for the message buffer.

**5. Per-SVC and per-boundary bookkeeping: est. ~5%.**
- every SVC clones its `SvcEvent` into `recent_svcs` and pushes it to
  `svc_trace` (3225);
- `thread_time` and `lock_stats` are updated per hold;
- the PC sample every 16th boundary (2299) allocates a String through
  `debug_var`.
Plan: per-worker rings and counters, merged into the machine at report
time.

**6. Thread-local and read-only SVCs: est. small.**
GetCurrentProcessorNumber (0x10), GetThreadId, OutputDebugString
(0x27) and the read-only GetInfo (0x29) kinds need no shared mutable
state. Plan: answer them from per-worker data, or from atomics and an
RwLock read, without the machine lock.

### What must stay serialised

- **Memory layout changes**: Map/UnmapMemory, Map/UnmapPhysicalMemory,
  SetHeapSize, SetMemoryPermission/Attribute, transfer, shared and code
  memory. One writer at a time. And because guest code on other workers
  never takes a lock, every page must go from its old state to its new
  one in a single host step. A changed page must never pass through
  "unmapped" or "copied out", which was today's bug: the old
  UnmapPhysicalMemory copied the kept parts out, unmapped and remapped.
  The fixed version (4020) changes only the covered part. `sync_guest_view`
  maps each run with one `mmap(MAP_FIXED)`, which is atomic per page; keep
  it that way.
  Host code that reads or writes guest memory (NVN, services) should
  hold a read lock on the layout. A layout change then takes it for
  writing, and a host read never races a change of the pages it is
  using, as the NVN submission could.
- **Thread states and wait queues** (the scheduler lock in 3): the
  check-then-sleep atomicity of every wait SVC relies on one lock
  covering the word read, the decision and the parking (part 1).
- **The handle table**: an RwLock suffices. Lookups are reads;
  create, duplicate and close are writes.

Lock order, to avoid deadlocks: scheduler → handles → services → NVN →
memory layout (read). Memory layout writes take nothing else while held.

### A model of the effect

`cargo bench -p echora-aot-tests --bench lock` models this without the
game. Workers on host threads run the arith kernel and, after each
50,000-instruction slice, do 10 us of host work (about 40% of one
worker's time), split by the main session's measurements at W=6: NVN
62%, slice ends 1.3%, the rest SVCs and IPC. On four host CPUs:

| workers | NVN under the global lock | NVN under its own lock |
|---|---|---|
| 1 | 1.00x, 43% holding | 1.00x, 17% holding |
| 2 | 1.34x | 1.69x |
| 4 | 1.47x | 1.77x |
| 6 | 1.45x | 1.92x |

Moving NVN to its own lock cuts global hold time to 0.40 of what it was,
as measured in the game (78.3 s to 32.5 s, 0.42 of it, with dea1c7f).
The model then shows the next limit: workers wait on the NVN lock itself
(56% of their time at six). So the next gain is NVN work that does not
serialise all workers: per-object locks, or recording that only takes a
lock at submit. Slice ends (`--fast`) matter little at 1.3%.

### Expected effect

If step 0 confirms the estimates, items 1 and 2 remove roughly half to
three quarters of today's lock time, and items 3 and 4 most of the rest.
At W=6 that would bring host work under the lock from about 40% of wall
time to perhaps 10-15%. Item 2 is the smallest change. Item 1 is the
largest gain, and submit already shows the way.
