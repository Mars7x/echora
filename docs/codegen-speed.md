# Speed of translated code: what costs, and the planned changes

Measured on the benchmark kernels (`benches/kernels.rs`, x86-64, release),
with the generated machine code read from the bench binary. A typical
integer loop (`ldst`: LDR, ADD, STR, SUBS, B.NE) cost about 53 host
instructions per 5 guest instructions:

| Part | Host instructions | Share |
|---|---|---|
| Guest loads and stores | ~10 | 19% |
| NZCV for the SUBS (only the B.NE reads it) | 15 | 28% |
| Per-block bookkeeping | ~20 | 38% |
| The arithmetic itself and the rest | ~8 | 15% |

The flags are now handled by `EmitOptions::fused_flags` (`ldst` +50%,
`memset` +40-50%), opt-in until the next full rebuild of a game
translation. This note designs the two remaining parts. Both need changes in
`echora-guest`, and any change there rebuilds every region of the game's
translation (about an hour). So they are to be done together, in one
rebuild, with `fused-flags` and `inline-vector` turned on at the same
time.

## Guest loads and stores

A load or store is one host `mov` after a range test. The relaxed atomics
(69e76d6) cost nothing: LLVM merges their aligned and unaligned paths into
the same `mov` and leaves one dead `test`. Plain accesses measured up to 8%
faster on `ldst`, but are not correct (a plain load can be hoisted out of a
polling loop; `tests/concurrency.rs`), and a branchless inline-asm `mov` was
no faster than today.

What does cost is the view's base and mask: they are fields of the `Core`,
and a guest store through a raw pointer may alias the `Core`, so LLVM
reloads both after every store. Neither ever changes while a window runs
(a mapping changes page protections, not the base or size of the
reservation).

**`echora-guest` API.** A copy of the access path that the generated code
can keep in registers:

```rust
// view.rs
#[derive(Clone, Copy)]
pub struct WindowMemory(GuestAccess);

impl GuestView {
    /// The view's base and range check, for a window to keep in locals.
    /// Valid while the view lives; the base and size never change.
    #[inline(always)]
    pub fn window_memory(&self) -> WindowMemory;
}

impl WindowMemory {
    // Exactly GuestView's plain accesses (same relaxed atomics, same
    // range test, same faults):
    pub fn load_u8(self, address: u64) -> u8;   // also u16, u32, u64
    pub fn store_u8(self, address: u64, value: u8); // also u16, u32, u64
}
```

Exclusive, atomic and vector accesses stay on `m.memory`: they need the
monitor or are rarer.

**Emission** (`EmitOptions::window_memory`, opt-in): the window's prologue
gets `let mem = m.memory.window_memory();`, and `m.memory.load_uN(` /
`m.memory.store_uN(` become `mem.load_uN(` / `mem.store_uN(`. Expected gain:
a few percent on memory-heavy code (two loads per block after a store).

## Per-block bookkeeping

Each block now pays:

1. `tick_block_at`: load `steps`, add, compare with `max_steps` (a load),
   branch, store `steps`, store `block_tag`.
2. `finish_block`: compare `steps` with `host_check_at` (a load), then if
   `parallel` (a load) load the `preempt` pointer and its flag.
3. A PC store, and (in flat emission) the back edge of a loop goes through
   the window's PC `match` instead of a jump.

**`echora-guest` changes.**

- **One limit.** Replace the budget and host checks with one shared limit:

  ```rust
  pub struct Core {
      /// min(max_steps, host_check_at), or 0 to stop at the next block:
      /// another worker lowers it to preempt this core.
      pub stop: Arc<AtomicU64>, // read Relaxed
      ...
  }
  ```

  `tick_block_at(pc, count)` becomes one check at the block's start:
  `if steps + count > stop.load(Relaxed) { return self.block_limit(pc, count) }`.
  The cold `block_limit` returns `Trap::Budget` if `steps + count >
  max_steps`, else `Trap::HostBoundary` before the block runs (nothing
  charged, `cpu.pc = pc`), so the host boundary moves from the end of a
  block to the start of the next, which resumption already handles (the
  boundary's PC is a block start either way). `finish_block` then needs no
  check at all. This removes about eight host instructions per block: the
  `max_steps` and `host_check_at` loads and compares, the `parallel` byte and
  the preempt pointer and flag.
- **`block_tag` only where a block can stop inside.** It exists to refund
  the unexecuted tail of a block that traps midway. Generated code would
  store it (through a new `Core::set_block_tag(pc, count)`) only in blocks
  with an instruction that can return an error to the host (a helper with
  `?`, an SVC, a call); blocks of plain arithmetic, loads and stores (a
  memory fault ends the process) skip the store.
- **Steps in a local** (emission, with the above): the window keeps
  `steps` in `r.steps` like the registers, stored back before calls,
  helpers that read it and every exit.

**Runtime changes** (`runtime.rs`, the preempt path):

- Where the host sets `core.host_check_at` or `core.max_steps` (the
  boundary's `check_at`, the attention case setting 0), it stores
  `min(max_steps, host_check_at)` into `stop`.
- `request_preemptions` stores 0 into the target core's `stop` (kept as
  `Vec<Arc<AtomicU64>>` like `preempt_flags` today) instead of setting its
  `preempt` flag; the target's boundary restores `stop` when it
  reschedules. `Core::preempt` and `Core::parallel` go.
- `settle_block_trap` stays: a boundary before a block charges nothing, so
  there is nothing to refund.

**Tests.** The block-accounting tests in `echora-guest` (a boundary before
a block charges and refunds nothing; a budget trap at a block start), the
runtime's `folded_cores`, `sync_stress` and `wait_timeouts` (preemption
through `stop`), and every aot-tests suite sliced.

## Tried: vector registers in window locals

Keeping the vector registers an inline vector path uses in window locals
(stored and reloaded only around helpers and calls, as the general
registers are) passed every suite but measured within noise of
`inline-vector` alone on the `vector` and `scalar` kernels (three
alternating runs each), so it was reverted. Their cost is elsewhere: each
inline FP operation checks every lane for a normal or zero value before its
fast path, and reloads `m.cpu.fpcr` (and `fpsr`) after every guest store,
which may alias the `Cpu`.

## Whole-vector checks (opt-in)

An inlined vector float operation (FADD, FSUB, FMUL, FMLA, FMLS) runs
natively only when every operand lane is normal or zero and every result
lane is normal or an exact zero. Checked lane by lane, that was half the
`vector` kernel's time (skipping it, incorrectly, doubled the rate).
`EmitOptions::whole_vector_checks` tests each operand vector at once: the
exponent and mantissa fields of every lane are below its top bit, so adding
all-ones below the top bit sets that bit exactly when the field is
nonzero, without a carry into the next lane. A result that is not all
normal (a zero, say) falls back to the exact per-lane test. Three
alternating runs: `vector` 199/226/199 to 284/262/259 Minstr/s (best),
`scalar` (no vector float operations) unchanged. Every aot suite passes
with it; accepting every operand fails the FP semantics suite.

On x86-64, `mul_add` (FMLA, FMADD) is a call to `compiler_builtins`'
`fmaf`/`fma`, which dispatches again at run time: building the translation
with `-C target-feature=+fma` inlines it (`scalar` +12-15%), but needs a
host CPU with FMA (x86-64-v3); a decision for the game build's flags.

## FPCR and FPSR in window locals (opt-in)

`EmitOptions::fp_status_locals` keeps FPCR and FPSR in the window's
locals, stored and reloaded only around helpers and calls, instead of
reloading them after every guest store (the inline float paths test them
on every operation). Three alternating runs on top of `inline-vector` and
whole-vector checks: `scalar` 552/556/561 to 573/569/574 Minstr/s (+2-4%,
every round), `vector` 217/220/219 to 220/222/220. Small but consistent,
so it stays opt-in for the rebuild. `tests/fp_status.rs` compares it with
the plain path around MSR, MRS and float helpers.

## PLT stubs inlined at their callers (default)

The game's hottest translated window was its PLT: stubs of
`ADRP X16, page; LDR X17, [X16, #off]; ADD X16, X16, #off; BR X17`
(`codegen::plt_stubs`). Each call through one ran the stub's window (entry,
locals, a dispatch match, one block, the exit) and then the window table
again for the target. Split emission now runs the stub in place at every
BL or B from another window: the caller charges the stub's block with the
same `tick_block_at` (a host boundary or the budget there leaves the PC at
the stub, and the host resumes it through the stub's own window), runs the
ADRP, LDR and ADD, and branches to the loaded X17. The load is the guest's
GOT word, so a rewritten or not-yet-bound slot (a lazy JUMP_SLOT holds its
PLT0 value) goes where the word says.

`echora lift` fills `Translation::got_words` with the words rtld is
expected to leave in each relocated slot (`process::predicted_got_words`:
relative relocations, and JUMP_SLOT, GLOB_DAT and ABS64 bound to the
symbol's first definition in load order after rtld; that search order is
inferred, not measured). When a call's predicted target is a translated
entry in the caller's region, the call is
`if x17 == PREDICTED { call_window(next, run_window_T) } else { call(next) }`:
the window function directly, skipping the table lookup and the indirect
call. Window functions of other regions are not nameable from a region
crate, so their calls keep the table. The guard only saves work:
`call_window` with a window that does not hold the PC returns at once and
continues through the table.

Stubs only reached by B, stubs a trace names, and branches within the
stub's own window keep the window path; `stub-windows` in
`ECHORA_LIFT_OPTIONS` (`EmitOptions::stub_windows`) keeps it everywhere,
for comparison. `echora lift` prints the stubs found, predicted and
inlined, and the BL (direct) and B sites rewritten. The recent-PC ring no
longer records an inlined stub's entry (diagnostics only).
`tests/plt.rs` runs a synthetic PLT with the predicted words, rewritten
words, and the budget and a host boundary falling on an inlined stub.

## Stores before calls: only what changed (default)

A split window keeps guest registers, SP and NZCV in locals and used to
store every local the window writes before each call or register helper
(the callee and the dispatcher read the `Core`). `RegisterLocals::rewrite`
now stores only the locals possibly written since they last equalled the
`Core`: a forward walk through each match arm, each arm entered with the
union of the states its predecessors hand it (a constant `next_pc`, or
either side of a two-way one; any other successor may be any arm; entering
the window leaves all equal), to a fixed point. A store or reload at an
arm's top level leaves every local equal on all paths past it; one in a
nested branch changes nothing. NZCV counts as written by flag operations
but not by an ADD/SUB or add-with-carry whose `set flags` argument is
`false`. Arms with their own loops or jumps (structured emission) store
everything, as before. The window's final store is unchanged. A PC store
that another follows with only constants moved into locals between is
dropped. Over the aot-tests programs (650 windows, 1,123 calls) the stores
before a call fell from 28.3 to 9.7 on average. `tests/spills.rs`: a
callee reading registers its callers wrote before the call, after it in a
loop's previous pass, through a conditional successor, and never.

## Float runs (opt-in)

`EmitOptions::float_runs` (`float-runs`): a run of at least eight scalar
single-precision FADD, FSUB, FMUL, FNMUL, FDIV, FMADD, FMSUB, FNMADD,
FNMSUB or FSQRT (with FMOV, FABS, FNEG between them; nothing that stores,
branches, calls or touches other state, and neither its block's first nor
last instruction) is emitted twice. The fast copy computes on host f32
locals and clears one `good` flag where an operand from before the run or
any result is anything but normal or zero, or a zero product, quotient or
fused result might be an underflow. It commits to the vector registers
when FPCR is in the inline paths' mode, FPSR already holds the sticky IXC
and `good` holds: the values are then the IEEE ones, which Arm gives too,
and nothing beyond IXC would be raised. Otherwise the exact copy (each
operation as emitted alone) runs from the run's start. No MXCSR state is
read: the per-result class checks are a few integer operations, the run
needs one FPCR and FPSR test, and operands from inside the run are not
checked again. Kernels (game set vs game set plus float-runs, two rounds
of seven runs): the new `floats` kernel (twelve operations per element)
861/860 Minstr/s against 407/396 (2.1x); `scalar` (two operations per loop,
no run) and the rest within noise. `tests/float_runs.rs` compares it with
the exact emission over edge values, FZ, DN and a directed rounding mode.

### Loads in float runs (opt-in)

`EmitOptions::float_run_loads` (`float-run-loads`, with float runs and
inline vector operations): runs may also hold LDR, LDUR and LDP of S
registers (immediate and register offsets, pre- and post-indexed). The
fast copy loads in program order through the same inline loads and checks
each loaded value as an operand from before the run; a base update is a
local committed with the run. A guest load the guest may not make ends
the process, so the fast copy dies at the same load the exact copy would
(only the registers it leaves differ: the run's start). An SP base not
16-byte aligned skips the load and declines, so the exact copy traps as
before. Kernels (game set plus float-runs, with and without loads, two
rounds of seven runs): `floatld` (four loads among twelve operations)
+41%/+41%, `floats` -3%/-4% (its loads now in the run, checked), the rest
within noise.

## Deferred block charges (opt-in)

`EmitOptions::deferred_charges` (`deferred-charges`, with split ticks): a
block that cannot trap, call, read the counter or poll, and is not a block
execution can resume at or a backward branch's target (`resume_blocks`),
adds its instruction count to a window local instead of charging the
`Core` (a relaxed load of `stop`, a compare, a store). Every other charge
and every call first adds that count to `steps`, and the window adds what
is left when its body returns, before the host settles a trap or checks
anything; every loop still passes a checked charge. No echora-guest change:
`Core::steps` is public. A host boundary may come one arm later than
before (the arm's `finish_block` sees the steps without the pending
count), as a slice boundary moving by a few blocks. Kernels (game set vs
game set plus deferred-charges, two rounds of seven runs): memset
+34%/+30%, calls +4%/+7%, the rest within noise. `tests/deferred_charges.rs`
checks registers and steps against per-block charging, with a host
boundary at every step of the run.

Flushes: a charge adds the pending count first only where a block that
may defer reaches it by a constant successor, or one in its window leaves
by a computed successor (then every block of the window does), or in a
lifted function (every charge); calls never do, as they follow their
block's charge. In game regions every charge and call used to flush
(gen-r-710510: 2918 charges, 4485 flushes, 1184 deferred blocks), and the
translated code took about 20% more samples in the forest; on the
aot-tests programs the flushes fall from 127,032 to 9,849 (125,427
charges, 1,741 deferred blocks).

## Lifted functions (design, opt-in prototype)

Today a guest call (BL) stores the caller window's written locals, runs
`Core::call` (the window table, an indirect call into the callee's 1 KiB
window, whose `match next_pc` finds the entry block and whose entry loads
its own locals from the `Core`), and reloads the caller's locals after.
A guest function longer than its window leaves through the dispatcher.

**Finding functions** (`codegen::functions`). Every BL target that starts
a block is a function entry. Its body is the blocks reachable from it in
its region through branches, both sides of conditional ones,
fallthroughs and the return point after a call, without entering another
function's entry (a B there is a tail call: it leaves). A return, BR (jump
tables included: the target leaves), BLR's continuation, an SVC or
another trapping end, and a branch out of the region leave the function.
A block can belong to several functions (shared tails, multiple entries
into one guest routine each get their own copy). Bodies are capped at
`FUNCTION_BLOCK_LIMIT` (256) blocks; larger ones keep the window path. The
lift prints the BL sites, how many target the caller's region, how many
target a function entry, and how many of those are in the caller's region.

**Emission** (`EmitOptions::functions`). Each function is a host function
`run_function_<entry>(m) -> Result<(), Trap>`, the `WindowFn` shape: the
window machinery over the function's blocks instead of a window's, with
its register locals for the whole function, starting at its entry block
(a constant, so no first dispatch), and returning when the next PC is none
of its blocks. The windows stay: everything not reached through a call,
and every resumption, runs there.

**Calls.** A BL to a function calls `Core::call_window(return, f)`: its
depth limit (`MAX_CALL_DEPTH`) bounds host stack use for deep or recursive
guest chains, past which the call continues through the window table, and
a callee that leaves before returning (a tail call, an indirect branch, a
host function) continues through the window table until the PC is the
return address, exactly as `Core::call` does. Within a region the call is
direct (`run_function_<entry>`). Region crates cannot name each other's
functions, so with `EmitOptions::function_table` every region holds a
copy of a table of all functions' pointers, which the root's `install`
fills (`install_functions`); a BL in any region then calls the target's
entry by its constant index (`function_at(i)`: one OnceLock load, an
indexed load and an indirect call, against `Core::call`'s table lookup,
loop and return-address checks, and the callee's window dispatch). Main's
relift12 counts 292,105 BL sites of which 15,530 call an entered block of
the caller's 64 KiB region, so the table is the path most calls would
take; it is part of the design, not a later step.

**The calling convention is not assumed.** At a call the `Core` stays the
truth: the caller stores the locals possibly written since they last
equalled it (`RegisterLocals` dirty tracking) and reloads the ones it uses
after, whatever the callee touched; the callee loads what it uses on entry
and stores what it wrote on return. A guest that breaks the AArch64
convention (a callee writing a callee-saved register, a caller reading a
clobbered one) runs as it does today. Narrowing the reload to what the
callee may write needs a transitive, conservative effect summary over the
callee and everything it reaches (indirect calls: everything); a later
step, measured on its own.

**Exits and resumption.** A trap, an SVC, a host boundary (slice end,
preemption, attention) or the budget leaves the function with the PC
where execution continues, after the function's final store, and every
caller up the host stack returns the trap the same way, storing its own
locals; the host resumes at the PC through the dispatcher and the
windows, as after a trap in nested `Core::call` frames today. Block
charges, host boundaries (`resume_blocks`), deferred charges (the
function body has its own pending count, added at its end) and step
counts are those of the windows' blocks, which the function emits with
the same code.

**Recursion and stack.** Each call is one host frame (the function's
locals); `call_window`'s depth limit caps the chain at `MAX_CALL_DEPTH`,
beyond which calls run through the window table in the same frame loop as
today.

**Prototype measurements** (kernels lifted with 64 KiB regions, as the
game's regions hold code, game option set, two rounds of seven runs,
best): calls kernel, against window emission: direct-calls -3.6%/-5.4%,
functions +6.1%/+7.0%; calls through the function table against direct
function calls -6.4%/-3.0% (698/716 to 653/695 Minstr/s, against window
emission's 612/612 in the earlier rounds: still ahead, across builds). Main's earlier direct-calls figure (-21%) was from kernels lifted
with 1 KiB regions, where the calls kernel's caller and leaves are in
different regions and its emission does not change: layout noise.
`tests/calls.rs` runs its callers against the reference interpreter with
functions in one region, directly and through the table.

### Call summaries for leaf functions (opt-in)

`EmitOptions::call_summaries` (`call-summaries`, with functions): a leaf
function (no BL or BLR in it, so nothing else runs while it does) gets an
effect summary from `crate::defuse` over all its instructions: the
general registers, SP and NZCV it may read or write, and those it may
write (`codegen::leaf_summaries`). A call of a leaf stores only the dirty
locals it may read or write (a register it may write is stored too: the
write may not happen, and the reload must find the caller's value), runs
the leaf's host function directly (one frame: no other code can run in
it, so no depth limit is needed), and reloads only what it may write; the
caller's other locals stay in host registers. FPCR and FPSR are stored and
reloaded as before.

Exits: a trap, SVC, host boundary or the budget comes back as an error,
on which the caller's final store writes every local it holds before the
host sees the `Core`; a leaf that leaves before returning (its PC not the
return address: a branch out of the function or region, BR) has the
caller store everything it holds before continuing through the window
table, so no other code ever sees a stale `Core`. Non-leaf callees keep
the full store and reload; summarizing them needs every callee's
summary and a guarantee that none returns early (else the callee's own
fallback runs foreign code while the caller holds locals), a later step.

Coverage: the kernels' two lifted functions are both leaves (2 of 2 BL
sites); the call test programs have 5 leaves among 8 functions, 239 of
402 BL sites. Calls kernel (64 KiB regions, game set, two rounds of seven
runs, best): functions against window emission +11.5%/+22.6%, call
summaries against functions +49.5%/+40.2% (974/997 against window
emission's 585/580 Minstr/s, about 1.7x). `tests/calls.rs` runs its callers
with summarized calls against the reference interpreter; a summary
without a leaf's writes, or without its reads, fails it. Through the function table
(`function-table`, the path 84% of the game's BL sites need: relift14
counts 796,334 BL sites, 125,946 of them to the caller's region) a
summarized call is `function_at(i)(m)`, the same otherwise: calls kernel,
functions and the table with summaries against without +37.9%/+32.5%
(699/721 to 964/956 Minstr/s). `tests/early_exits.rs` has leaves that
leave early, through a branch out of their region to code reading a
register the caller held, and through an SVC; dropping the caller's store
of held locals there fails it.

Transitive summaries (`transitive-summaries`, with call summaries): a
function that calls others gets a summary when every call in it is of a
lifted function with one that the call reaches as a function (its region,
or any through the table), none is a PLT stub, it has no BLR, and it is on
no cycle of calls (so direct calls nest a bounded depth); the summary is
its own effects and its callees', to a fixpoint (`transitive_summaries`).
No other code runs while it does: in a lifted function a summarized call
whose callee returns early returns from the function too (its final store
writes every local), up to the caller that holds nothing (a call through
`Core::call_window`, which continues through the window table) or the
window, which stores what it holds and continues with `Core::call`. A
trap, an SVC, a host boundary (a slice end, a preemption) or the budget
in a summarized callee comes back as an error, on which every caller's
final store writes its locals. Coverage: the call test programs' 3
functions that call others get summaries (163 BL sites; with the 239 to
leaves, all 402); the kernels' calls are to leaves. New `calls2` kernel
(a call of a function that keeps a frame and calls both leaves; game set
plus functions, function-table and call-summaries, two rounds, best):
+7.1%/+7.7% with transitive summaries. `tests/early_exits.rs` has two
levels (window, function, leaf) where the leaf leaves early into code
reading registers both callers held; without the propagation, or with
summaries missing a callee's effects, it fails.

Only a function that leaves solely by its own return (RET, RETAA, RETAB),
with every branch, fallthrough and call return point in its body and no
block ending in an SVC, a BR or another end, gets a summary (leaf or
transitive). An SVC wrapper (`SVC`; the RET after it is a resume point,
outside the body) gets none: it could end by running on outside its body,
where code relies on registers in the `Core`. With call-summaries the game
(spd162) hung in a lock loop calling such wrappers through the table;
the wrapper path as reproduced here (`tests/early_exits.rs`: a caller
loop around an SDK-style SVC wrapper, the host completing the SVC and
resuming at the RET) stores the caller's X30 before the SVC trap reaches
the host, so the cause there is not yet seen, and the rule removes such
calls from summarized ones whatever it was.

## The step count in a window local (opt-in)

`EmitOptions::steps_local` (`steps-local`, with split ticks, not with
deferred charges): a split window or lifted function keeps the step count
in a local (a parameter of its inlined body). An untagged block's charge
is `if *st + n <= stop { *st += n } else { exact checks }` with the same
relaxed load of the stop word as `Core::charge_block_at`, so a preemption
is seen as soon as before; a tagged block's charge goes through the
`Core` (its tag is private to echora-guest) with the count stored before
and reloaded after. The count goes to the `Core` before calls (summarized
ones too), the exact checks, a host check, counter reads, spin hints and
traces, comes back after calls, and goes back when the window's body
returns (every trap, SVC and exit passes there). `Core::finish_block` at
an arm's end reads no step count (the charges take host boundaries), so it
needs no store of the local. No echora-guest change.

`tests/steps_local.rs` compares final registers and steps, every host
boundary's PC and step count (the host check due at every step) and the
budget's trap (every budget) with per-block charging; `steps_local_calls.rs`
does the same with the window table installed, so calls run in the
caller's frame. Dropping the write-back after the body, or the reload
after a call, fails them.

Static counts on the aot-tests programs: 4,682 charges local, 125,959
tagged (through the `Core`, with a store and reload around each: their
blocks can trap after the charge). Kernels (game set vs game set plus
steps-local, each kernel run alone, alternating, best of seven): memset
2539 to 4124 Minstr/s (+62%), ldst 4435 to 5844 (+32%), calls2 flat, calls
-2% to -4% (the full suite's runs showed -10% to -20%, mostly noise of
the machine).

The calls loss is not the syncs. At a window call the store before the
tagged charge is the only extra instruction (LLVM drops the explicit one
before the call, the charge having just synced the count); the callee
loads the count on entry and stores it on leaving, as base's charges do.
Testing the local first and charging through the `Core` only when it
passes keeps that store (the relaxed load of the stop word lies between)
and adds a second load: no gain. A public `Core::tag_block` (an
echora-guest change, tried locally) makes the tagged fast path
`*st += n` and the tag alone, the caller's code then matching base
instruction for instruction, and calls measured the same: the rest is
layout. Lifted functions are called through `WindowFn`, so passing the
local to them would need a second entry per function; not done.

## Vector float unary operations inlined (opt-in)

`EmitOptions::inline_vector_unary` (`inline-vector-unary`, with
`inline-vector`): vector FSQRT, FCVTZS/FCVTZU (to integers, toward zero),
FRSQRTE, FRECPE and FRINT* (4S, 2S, 2D, and the scalar forms that exist)
no longer call `vec_fp_unary` for the whole vector. Under the FPCR the
helper accepts (round to nearest, no AH or FZ16; it traps otherwise, and so
does the inlined code, through the helper), each lane runs natively when
that is the helper's result and the helper raises nothing for it:

- FSQRT: a positive normal or a zero.
- FCVTZS/FCVTZU: a normal or a zero whose truncation fits the integer
  (the helper raises IOC otherwise; it models no IXC for conversions).
- FRINT*: a normal or a zero, with the host's rounding of the same
  direction (ties to even, up, down, toward zero, ties away).
- FRECPE: a normal whose reciprocal is normal (biased exponent below
  2 * bias - 1): the 8-bit estimate from the top fraction bits, the
  exponent 2 * bias - 1 - e, the sign kept.
- FRSQRTE: a positive normal: the helper's table (its low bytes, a
  384-byte constant in the emitted code) indexed by the exponent's parity
  and top fraction bits, the exponent (3 * bias - 1 - e) / 2.

Any other lane (NaN, infinity, denormal, negative square root, out of
range conversion, an estimate with a denormal result) goes through the
helper's scalar form alone, in Vd, which is written last; the helper
raises what that lane raises (IOC, IDC, DZC, UFC), and FZ and DN apply as
in the helper. No echora-guest change.

`tests/vector_unary.rs` runs every form of every operation from V1 into V0
and into V1 itself, from edge lanes and random bits, under FZ and DN on
and off and random FPSR flags, against the same instructions lifted with
the helper for every lane: vector registers, FPCR, FPSR and the trap must
agree, also under a rounding mode the helpers trap on. FRECPE (positive and
negative) and FRSQRTE are compared for every fraction of an even and an
odd exponent. Letting a negative square root, 2^31 for FCVTZS, -1 for
FCVTZU, a denormal-result FRECPE or a denormal FRINT through natively, or
an off-by-one estimate, fails them.

Kernels (game set vs game set plus `inline-vector-unary`, two rounds of
seven runs, best): `convert` (FMUL, FCVTZS, FCVTZU) 80.8 to 363.3 and
81.6 to 365.3 Minstr/s (+349%, +348%). `vnorm` (FMUL, FRSQRTE, FMUL,
FRSQRTS, FMUL, FMUL), each binary alone, alternating, three times: 151 to
278-286 Minstr/s (+84% to +89%); with the estimate computed from a
correctly rounded square root instead of the table (exact as well) it was
3% slower than the helper, the division and square root per lane costing
more than the call. The other kernels have
no such operation: memset (+88%, +89%) and switch (+32%, +29%) moved with
code layout, the rest within noise.

## Vector integer operations, element moves and float pairwise (opt-in)

What an `inline-vector` lift still sent to helpers (every decodable SIMD
word sampled, `simd::emit_inline` asked of each): every vector integer
operation but ADD and SUB, every element move (DUP, INS, UMOV/SMOV, MOVI,
ORR/BIC immediate), the pairwise and across-lanes forms, and among floats
FADDP/FMAXP/FMINP/FMAXNMP/FMINNMP, FMAXV and the like, FABD, FMULX,
conversions other than those of `inline-vector-unary`, and the scalar
FSQRT and FRINT*; also TBL/TBX, SDOT/UDOT, PMULL and the crypto
instructions.

`EmitOptions::inline_vector_int` (`inline-vector-int`, with
`inline-vector`) writes out the helpers' own lane functions for the
operation's lane width: three-same (`int_lane`: compares, MUL/MLA/MLS,
min/max, absolute differences, halving and rounding-halving adds,
saturating adds and subtracts, SQDMULH/SQRDMULH, the register shifts
SSHL/USHL/SRSHL/URSHL/SQSHL/UQSHL/SQRSHL/UQRSHL), by element, pairwise
(ADDP, SMAXP, ...), unary (ABS, NEG, NOT, CNT, CLZ, CLS, RBIT, REV*,
compares against zero), shifts by immediate (SHL, SSHR, USHR, SSRA, USRA,
SRSHR, URSHR, SRSRA, URSRA, SQSHL, UQSHL, SQSHLU, SHRN), SSHLL/USHLL, the
widening operations (SADDL, SSUBW, SMULL, SMLAL, SABAL, ...), the
narrowing ones (XTN, SQXTN, UQXTN, SQXTUN, with their shifting and rounding
forms), SADDLP/UADDLP/SADALP/UADALP, ADDV/SMAXV/UMINV/SADDLV/UADDLV, and
the element moves. The same arithmetic (i128 where the helper widens), the
same saturation and FPSR.QC, the same zeroing of what Vd's form does not
write; nothing traps. `EmitOptions::inline_vector_pairwise`
(`inline-vector-pairwise`): FADDP, FMAXP, FMINP, FMAXNMP and FMINNMP per
result lane natively when the pair is normal or zero (and for FADDP the sum
normal or zero, under FPSR.IXC already set, as the other inlined float
operations), any other lane through the exact single-lane helper.

`tests/vector_int.rs` samples random words of every such operation and
form (4,612 forms, up to two words each), lifted with the options and
without, and runs each from vector registers of edge lanes (0, 1, all
ones, the sign bit, the largest positive, float edges, shift counts at
every lane width in every byte) and random bits, with FZ, DN, QC and IXC
varied and a rounding mode the float helpers trap on: every register, FPSR
and the trap must agree. The lift with the options must keep no integer or
move helper call. Dropping QC, reversing REV's order, misplacing the upper
half of a narrow, not clearing a W register's upper half (UMOV), letting a
NaN pair through FADDP, replicating DUP over 16 bytes for a 64-bit vector,
an off-by-one CLS or SRSHR rounding constant, or SSHL's shift of exactly
the lane width each fail it. Its lifts, like `vector_unary`'s, keep FPCR
and FPSR in window locals as game builds do.

Kernels (game set vs game set plus both options, two rounds of seven
runs, best): `intvec` (halfword SQADD, SSHR, MUL, CMGT, SQXTN, USHLL,
UMAXV, UMOV, DUP, AND, ADD) 55.8 to 484.2 and 56.6 to 483.0 Minstr/s
(+768%, +754%); `fpairs` (FADDP, FMAXP, FMINNMP, scalar FADDP) 72.5 to
449.7 and 72.4 to 468.9 (+520%, +548%). Kernels without these
operations: memset (-47%, -47%) and switch (-18%, -14%) moved with code
layout (memset has measured both about 1,850 and 3,500 Minstr/s from
unchanged code in these runs), the rest within noise.

## Scalar float moves and conversions (opt-in)

Static helper calls in the game's 100 hottest regions (88% of samples)
count the fallbacks of already inlined code with the main-path calls: the
large ones (`vec_fp_lane` 38k, `fp_binary` 34k, `vec_fp` 29k, `fp_ternary`
14k, `fp_compare` 13k, `vec_fp_element` 10k) are the exact helper behind
inlined FADD/FMUL/FMADD/FCMP and their vector forms, run only for lanes
that are not normal or zero. What `emit_inline` never takes (every
decodable word asked): FMOV between general and FP registers
(`fp_move_from_general` 4.5k, `fp_move_to_general` 1.2k), the scalar
conversions (`int_to_fp` 1.3k, `fp_convert` 1.0k, `fp_to_int` 1.0k,
`simd_int_to_fp` 0.8k), scalar FSQRT and FRINT* (`fp_unary` 1.5k) and
FCCMP (0.9k); then TBL (1.1k) and FMAXV and the like.

`EmitOptions::inline_fp_scalar` (`inline-fp-scalar`, with `inline-vector`)
takes that group (`crate::simd_scalar`):

- FMOV to and from general registers, upper halves included: a bit move,
  no helper.
- SCVTF/UCVTF from a general or an FP register: the helper's own host
  conversion once the FPCR is one it accepts (it checks nothing else).
- FCVT{N,P,M,Z,A}{S,U} to a general register: a normal or zero operand
  whose rounded value fits (the helper raises IOC otherwise, and models no
  IXC for conversions).
- FCVT between single and double: a normal or zero operand whose result is
  normal or zero (no flush, no overflow, no NaN conversion).
- FSQRT, FRINT* and FRINTX: as their vector forms (`inline-vector-unary`).
- FCCMP/FCCMPE: the immediate flags when the condition fails, else the
  compare as FCMP is inlined.

Everything else calls the helper, on a line of its own: register locals are
stored before and reloaded after a line that calls a helper, which undid a
native write to the destination on the same line (the first version of
FCVTZS lost its result that way; the test caught it).

`tests/fp_scalar.rs` samples random words of the group (every operation
and form, up to four words each, FCCMP's every condition and immediate),
lifted with the option and without, and runs each from float and integer
edge values and random bits, random NZCV, under FZ and DN on and off and a
rounding mode the helpers trap on: every register, NZCV, FPSR and the trap
must agree; the lift with the option keeps no FMOV helper call. FCVTZS
admitting 2^31, FCVT without the result check, FSQRT of negatives
natively, FMOV to an upper half clearing the lower, FCCMP's immediate
flags dropped, UCVTF of a W register signed, or FRINTX rounding ties away
each fail it.

Kernels (game set vs game set plus `inline-fp-scalar`, two rounds of seven
runs, best): `fpscalar` (FMOV both ways, SCVTF, FCVTZS, FCVT both ways,
FSQRT, FRINTM, FCCMP, FADD) 369.8 to 846.5 and 366.2 to 892.6 Minstr/s
(+129%, +144%); the other kernels have none of these operations and are
within noise.

## The remaining vector forms (opt-in)

`EmitOptions::inline_vector_misc` (`inline-vector-misc`, with
`inline-vector`), `crate::simd_misc`:

- TBL/TBX: the byte lookup over the table registers, out-of-range indices
  zero (TBL) or Vd's byte (TBX).
- LD1-LD4/ST1-ST4 (single structure, to a lane), LD1R-LD4R and
  LD1-LD4/ST1-ST4 (multiple structures): the helper's own single
  `GuestView::read` or `write` of the whole transfer, into a buffer on the
  stack instead of a heap vector, then the lanes placed as the helper
  places them (all registers of a load written after the read). The same
  access, the same fault (a guest access outside mapped memory ends the
  process either way) and nothing written before it.
- FABD and FMULX (vector, scalar, by element): per lane, natively for
  normal or zero operands whose result is normal or an exact zero, under
  FPSR.IXC already set, as the inlined arithmetic (FMULX of such operands
  is FMUL); any other lane through the exact single-lane helper.
- FMAXV/FMINV/FMAXNMV/FMINNMV: the helper's tree natively when all four
  lanes are normal or zero (the forms agree then and raise nothing), else
  the helper.
- FCVT*S/U and SCVTF/UCVTF with fraction bits (scalar, to and from general
  registers), the vector FCVT*S/U in a directed rounding or with fraction
  bits (per lane, a normal or zero lane whose scaled, rounded value fits;
  any other lane through the helper's scalar form) and vector SCVTF/UCVTF
  (the helper's own arithmetic once the FPCR is accepted).

INS and DUP from a general register are `inline-vector-int`'s already.

`tests/vector_misc.rs` samples random words of these (one of each form,
120 of each kind), lifted with the option and without, and runs each from
vectors of float edges, table indices around the tables' sizes and random
bits, general registers and SP inside the data (for the other operations
every other run edge values), data of random bytes, FZ and DN on and off
and a rounding mode the float helpers trap on: every register, the data,
FPSR and the trap must agree; the lift with the option keeps no TBL or
structure transfer helper call.

## Counting helper fallbacks (debugging, opt-in)

`EmitOptions::count_fallbacks` (`count-fallbacks`): every helper call inside
inlined code (the exact helper an inlined operation falls back to) first
adds one to its counter in the region's `FALLBACKS` (relaxed atomics, one
per helper in `FALLBACK_HELPERS`, the last for any other). The root's
`fallback_counts()` sums them over the regions and
`print_fallback_counts()` prints them, most first; with the option the
runner prints them when the run returns, and every
`ECHORA_FALLBACK_COUNTS_EVERY` seconds when that is set (a run that is
killed returns nothing). Without the option nothing in the emission or the
runner changes. `tests/fallback_counts.rs`: operands the inlined FADD,
FSQRT and FMLA take natively count nothing, a NaN lane counts one call of
the helper each falls back to.

Kernels (game set vs game set plus `inline-vector-misc`, two rounds of
seven runs, best): `misc` (LD1R, a lane load and store, TBL, FABD, FMULX,
FMAXV, a fixed-point SCVTF) 51.7 to 294.4 and 52.8 to 294.5 Minstr/s
(+469%, +458%). arith (-17.5%, -10.7%) and ldst (-33.5%, -19.3%) fell in
both rounds and alone, alternating (ldst about 5,400 against 4,300): their
generated code is the same in both builds, and their windows' machine code
is the same instruction for instruction, at another address (ldst's window
at 0x3d020 against 0x3cd50): alignment, from the added code before it.

## Layout sensitivity and aligned functions (opt-in build flag)

Adding code to the kernels once cost arith 11-18% and ldst 19-34% in both
rounds, with their generated and machine code unchanged and only at
another address. Measured on this machine (Intel Xeon, family 6 model
207, Emerald Rapids; the JCC erratum of Skylake-era cores does not apply):
`ECHORA_KERNEL_PAD=n` (aot-tests `build.rs`) puts a function of n kept
additions before the kernels' windows, moving the same code by 0, 64, 80,
96, 128 and 144 bytes (pads 0, 1, 2, 3, 5, 8: shifts of 0, 16 and 32
modulo 64). Each binary ran each kernel six times alternating with the
others, best of seven each time, pinned to one CPU.

Without alignment, the best rate follows the hot loop's place in its
64-byte lines: memset 2,868, 3,120 and 3,704 Minstr/s at shift 0, 1,792 to
1,915 at shift 32; ldst 4,577 to 5,853 at shift 0, 3,482 to 4,162 at 32.
The hot loops are 63 to 118 bytes and always cross a line (ldst's: 113
bytes, eight guest-level loads, stores and checks); where they start
decides how many 64-byte fetch windows of the decoded-instruction cache
they take. Spread of the best rate over the six layouts: memset 75%, ldst
50%, switch 43%, arith 21%.

`-C llvm-args=-align-loops=64` does not reach these loops: their headers
have a fallthrough predecessor, and LLVM leaves such a header unaligned.
`-C llvm-args=-align-all-functions=6` (every function 64-byte aligned) does
fix the place of every loop relative to the lines: it then depends only on
its own function's code. Spread with it: memset 22%, ldst 27%, switch 23%,
arith 23% — the noise floor, as three binaries with every window at the
same address differ as much (medians ldst 2,469 to 3,394, arith 2,618 to
3,921). The mean moves either way, as each loop lands where its function
puts it: arith +10%, switch +15%, ldst -18%, memset -20%. It makes speed
reproducible, not higher.

`EmitOptions::align_windows` (`align-windows`) is that flag: `echora lift`
adds it to the translation's `.cargo/config.toml` rustflags; nothing
generated changes (it applies to every crate of the build, listed regions
or not). Use it on both builds of a comparison, so that code added in one
place does not move the loops of every other region.

Comparing two game builds: on this machine one best-of-seven kernel rate
varies by 18% (coefficient of variation, quartiles 12-24%). Measure the
noise of the game's own metric first (two runs of the same build, several
times), then, with CV the run-to-run variation and d the change to see,
about 16 * (CV / d)^2 runs of each build, alternating, compared by median
(a 5% change at CV 10% needs 64 runs a side; at CV 18%, 200). Without
`align-windows` on both, also relink each build at three or more layouts
(any unrelated code change moves everything after it) and count the
spread between them as noise: a single relink can move a hot loop's speed
by a third.

## Where emitted code spends its operations (census)

`echora_lift::census` counts, per window or lifted function of emitted
Rust, guest loads and stores by width (direct through the window's view
of memory, or checked: `read`/`write`, LDP/STP of Q registers, atomics
and exclusives), calls and indirect branches, flag computations kept and
fused, stores and reloads of register locals inside the body (NZCV, FPCR
and FPSR among them), block charges, helper calls by helper (fallbacks
inside inlined code included) and inlined operations by the option that
inlined them (`EmitOptions::census` marks them with comments; nothing
compiled changes). aot-tests' `kernel_code::KERNEL_CENSUS` is the table
for the kernels as built; `cargo test --release -p echora-aot-tests
--features <options> --test census -- --nocapture` prints it. With the
game set and `inline-vector-unary`, `-int`, `-pairwise`,
`inline-fp-scalar` and `inline-vector-misc`:

| function | loads 1/2/4/8 | stores 1/2/4/8 | checked | calls | indirect | flags kept/fused | spills (status) | reloads (status) | charges | helper calls | inlined |
|---|---|---|---|---|---|---|---|---|---|---|---|
| arith | 0/0/0/0 | 0/0/0/0 | 0 | 0 | 0 | 1/0 | 0 (0) | 0 (0) | 4 | - | - |
| memcpy | 0/0/0/0 | 0/0/0/0 | 4 | 0 | 0 | 1/1 | 0 (0) | 0 (0) | 5 | - | - |
| memset | 0/0/0/0 | 0/0/0/2 | 0 | 0 | 0 | 1/1 | 0 (0) | 0 (0) | 5 | - | - |
| calls | 0/0/0/0 | 0/0/0/0 | 0 | 2 | 0 | 1/0 | 4 (1) | 6 (2) | 5 | - | - |
| (leaf functions) | 0/0/0/0 | 0/0/0/0 | 0 | 0 | 2 | 0/0 | 0 (0) | 0 (0) | 2 | - | - |
| switch | 0/1/0/0 | 0/0/0/0 | 0 | 0 | 1 | 1/0 | 0 (0) | 0 (0) | 12 | - | - |
| vector | 0/0/0/4 | 0/0/0/0 | 0 | 0 | 0 | 1/1 | 6 (6) | 12 (12) | 5 | vec_fp 3, vec_fp_lane 3 | vector 3 |
| poll | 0/0/0/0 | 0/0/0/0 | 1 | 0 | 0 | 0/0 | 0 (0) | 0 (0) | 3 | - | - |
| scalar | 0/0/1/0 | 0/0/1/0 | 0 | 0 | 0 | 1/1 | 1 (1) | 7 (7) | 5 | fp_binary 1, fp_compare 1, fp_ternary 1 | vector 5 |
| ldst | 0/0/0/1 | 0/0/0/1 | 0 | 0 | 0 | 1/1 | 0 (0) | 0 (0) | 5 | - | - |
| floats | 0/0/3/0 | 0/0/1/0 | 0 | 0 | 0 | 1/1 | 0 (0) | 24 (24) | 5 | fp_binary 9, fp_ternary 2, fp_unary 1 | fp-scalar 1, vector 11 |
| floatld | 0/0/4/0 | 0/0/1/0 | 0 | 0 | 0 | 1/1 | 0 (0) | 24 (24) | 5 | fp_binary 9, fp_ternary 2, fp_unary 1 | fp-scalar 1, vector 11 |
| calls2 | 0/0/0/2 | 0/0/0/2 | 0 | 3 | 1 | 1/0 | 11 (1) | 15 (3) | 7 | - | - |
| vnorm | 0/0/0/2 | 0/0/0/2 | 0 | 0 | 0 | 1/1 | 12 (12) | 24 (24) | 5 | vec_fp 5, vec_fp_lane 5, vec_fp_unary 2 | unary 1, vector 5 |
| convert | 0/0/0/2 | 0/0/0/2 | 0 | 0 | 0 | 1/1 | 6 (6) | 12 (12) | 6 | vec_fp 1, vec_fp_lane 1, vec_fp_unary 4 | int 1, unary 2, vector 3 |
| intvec | 0/0/0/4 | 0/0/0/2 | 0 | 0 | 0 | 1/1 | 0 (0) | 0 (0) | 5 | - | int 9, vector 2 |
| fpairs | 0/0/0/4 | 0/0/1/2 | 0 | 0 | 0 | 1/1 | 0 (0) | 34 (34) | 5 | vec_fp_lane 13, vec_fp_pairwise 4 | pairwise 4 |
| fpscalar | 0/0/1/0 | 0/0/1/0 | 0 | 0 | 0 | 2/1 | 15 (3) | 29 (20) | 5 | fp_binary 1, fp_compare 1, fp_convert 2, fp_to_int 1, fp_unary 2, int_to_fp 1 | fp-scalar 9, vector 2 |
| misc | 0/0/0/2 | 0/0/0/0 | 3 | 0 | 0 | 1/1 | 8 (8) | 16 (16) | 6 | vec_fp 3, vec_fp_across 1, vec_fp_lane 3, vec_fp_unary 1 | int 1, misc 8, vector 1 |

The slowest kernels per guest instruction are the vector float ones
(vector, vnorm, misc, fpairs: 280 to 470 Minstr/s against 1,600 to 5,400
for the integer ones): none of their hot paths calls a helper (every
helper call left is a fallback, and `count-fallbacks` counts none in any
kernel) or spills, so what they spend is the inlined float operations
themselves, 3 to 11 a loop, each testing its operands and results and
the FPCR. memcpy moves Q register pairs through the checked
`Cpu::load_q_pair`/`store_q_pair` (4 checked accesses a loop, 1,590
Minstr/s against ldst's 5,400 with direct ones): a direct path for LDP/STP
of Q registers, as LDR/STR of Q already have, is the next candidate the
numbers point at. The FPCR and FPSR reloads (up to 34 a window) are
around those fallback calls, off the hot paths; switch spends its
operations on block charges (12 for 11 blocks).

## Options on some regions only

`ECHORA_LIFT_OPTIONS_REGIONS` (a comma-separated list of region ids as in
the crate names, `gen-r-<id>`, as the value or in the file it names): the
listed regions get the default options plus `ECHORA_LIFT_OPTIONS`, every
other region the defaults, byte for byte as without the variable, so a
relift rebuilds only the listed crates. Without it `ECHORA_LIFT_OPTIONS`
replaces the defaults everywhere, as before. The lift prints the listed
regions it emitted with those options and any listed ids that match no
region.

`codegen::emit_rust_split_to_regions` runs the analyses (resume blocks,
PLT stubs, liveness, structured trees, functions, summaries) once and
switches each region's settings when its first window starts. Lifted
functions exist only in regions whose options have `functions`: a BL in
another region keeps `Core::call`, a BL to such a region's code from a
region with functions is an ordinary call too (the function set holds
only emitted functions), and the function table lists only emitted
functions, installed only into regions that use it. A function gets a
transitive summary only when its own region propagates early returns.
With an empty list the emission is the single-option emission, byte for
byte (the golden hashes do not move). `tests/calls.rs` lifts its callers
with every other region holding functions, the table and transitive
summaries and the rest without, so calls cross both ways, against the
reference interpreter; the unlisted regions must name no function.

## Recommended set for the next rebuild (measured 2026-11)

**Turn on `inline-vector,fused-flags,whole-vector-checks,fp-status-locals`
together; leave `structured` off.** `ECHORA_LIFT_OPTIONS` accepts all five
names.

Correctness: every `echora-aot-tests` suite (22 test binaries, 87 tests:
the integer, SIMD, FP and SIMD-memory semantic suites against the
reference interpreters, sequences, loops, switches, calls, exclusives,
spin, stress, concurrency, structure, `fused_flags`, `fp_status`, golden)
passes with all four on at once:

```
cargo test -p echora-aot-tests --features inline-vector,fused-flags,whole-vector-checks,fp-status-locals
```

(The release profile of this crate, LTO with one codegen unit, needs more
than 15 GB with two rustc jobs; `-j 1` fits.)

Speed, from `cargo bench -p echora-aot-tests --bench kernels`, best
Minstr/s against the default (flat) emission, on a shared 4-core x86-64
container. Two runs of the unchanged flat build differed by up to 21% on
the integer kernels, so only changes well past that, or repeated in every
run, count:

| Kernel | flat | inline-vector | recommended (2 runs) | without fused-flags | + structured |
|---|---|---|---|---|---|
| vector | 63 | 194 (3.1x) | 333, 340 (5.3x) | 285 (4.5x) | +25% |
| scalar | 345 | 680 (2.0x) | 753, 907 (2.2-2.6x) | 729 (2.1x) | -4% |
| memset | 1891 | 1457 | 2186, 2314 (+16-22%) | 1559 (-18%) | -12% |
| ldst | 3002 | 2467 | 3361, 4264 (+12-42%) | 2577 (-14%) | -13% |
| arith, memcpy, calls, switch, poll | | within noise | within noise | within noise | -14% to +48% (switch) |

- `inline-vector` with `whole-vector-checks` is the large win (vector
  kernels about 5x, scalar FP about 2x).
- `fused-flags` is what moves `memset` and `ldst` (both runs with it are
  up, the run without it is down), as measured when it was added.
- `fp-status-locals` adds a little on `scalar`, consistent with its
  earlier +2-4%.
- `structured` is mixed and costs size, so not now.

Size and compile time, from `cargo bench -p echora-aot-tests --features
codesize --bench codesize` (every split test program, 60,728 guest
instructions, compiled as a game region at `opt-level` 1):

| Mode | Rust bytes/ins | rustc s | `.text` bytes/ins |
|---|---|---|---|
| flat | 156.7 | 54.9 | 157.1 |
| structured | 193.0 | 59.0 | 164.7 |
| inline_vector | 166.3 | 57.7 | 161.8 |
| fused_flags | 156.7 | 51.9 | 157.1 |
| whole_vector | 169.8 | 58.1 | 162.0 |
| fp_locals | 174.0 | 61.6 | 162.1 |
| recommended | 174.0 | 57.4 | 162.1 |

The recommended set costs 3% more machine code and about 5% more compile
time than flat, so the rebuild takes about as long as today's.
`fused_flags` changes size little on these random programs (only 357 of
3,757 flag writes in them are dead); in game code compares before a branch
are the common case.

The `window-memory` and block bookkeeping changes above need
`echora-guest` and runtime work first. They are not in this set.

## Order for the rebuild

1. Land the `echora-guest` API and runtime changes with today's emission
   (the generated code still calls `tick_block_at` / `finish_block`, now
   cheaper).
2. Turn on, in `ECHORA_LIFT_OPTIONS`: `inline-vector`, `fused-flags`,
   `whole-vector-checks`, `fp-status-locals` (measured above), and, once
   they exist, `window-memory` and the block bookkeeping emission. Without
   the `echora-guest` changes, the four measured options can go in a
   rebuild on their own.
3. One full rebuild of the game's translation; compare frames and speed
   with the previous build.
