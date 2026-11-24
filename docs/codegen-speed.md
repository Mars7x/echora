# Speed of translated code: what costs, and the planned changes

The sections below record experiments and the state of each option when
measured. See [the development guide](development.md#switches-for-test-runs)
for the CLI's current defaults.

## Code discovery

Discovery retains decoded instructions and formats disassembly only when it
appears in a diagnostic. Straight-line instructions queue their next PC
without allocating a successor list. Both changes apply to every guest
program.

On a Ryzen 7 5800X3D with Rust 1.99.1, three alternating release runs of
`echora analyze --limit 20000000 --report report.json game.nup` on the
supported 1.1.1 dump had a median end-to-end time of 19.74 s before and
11.13 s after. Peak RSS fell from 6.05 to 3.21 GiB. Each run reported
17,633,790 instructions in 3,778,014 blocks and produced the same JSON
report byte for byte. This measures discovery, which runs before emitting
Rust and building the native translation.

The register-liveness pass summarizes each basic block once and propagates
register masks by block index. A comparison of both solvers on 3,779,116
blocks discovered in the supported ExeFS, with host entries included, ran
three times on the same 5800X3D. The median for liveness alone fell from
29.25 s to 1.36 s, and every live-out mask matched in each comparison.

## Rust emission

The register rewrite scans register references once per line and reuses
the original text when the register, flag or stack rewrites make no
change. A benchmark using 61,662 lines from hot translated windows
took a median of 212 ms with the new rewrite and 733 ms with the previous
rewrite, with identical text and annotations. A full lift of the same
game and options emitted byte-identical Rust for all 1,111 regions. The
two full lift runs took 181 s and 296 s, respectively.

Region emission shares the translation analyses and writes completed
regions in address order. With four logical CPUs available to the lift,
emitting up to four regions at once took 85 s against 181 s for serial
emission. All 1,111 generated region sources matched byte for byte.

## Translated execution

The NEON widening shifts use fixed lane widths, and integer reductions
read the lanes directly from the register. On the Ryzen 7 5800X3D, the
median widening-shift time fell from 23.15 ns to 5.62 ns and the median
reduction time fell from 25.66 ns to 13.81 ns. Each instruction family
matched the previous implementation on 32,768 inputs.

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

On x86-64 with SSE2, generated code classifies packed f32 lanes using
integer vector comparisons. These distinguish signed zeros, subnormals,
finite normals, infinities and NaNs without consulting the host FP mode.
The portable bit checks handle f64 and other targets. On a Ryzen 7
5800X3D, four alternating runs pinned to one core and built with aligned
functions, `target-cpu=native`, and `vector-float-runs` gave these medians
in million guest instructions per second:

| Kernel | Portable checks | Packed checks | Packed checks and arithmetic |
|---|---:|---:|---:|
| `vector` | 417.8 | 573.1 | 935.2 |
| `vnorm` | 555.9 | 662.4 | 832.4 |
| `convert` | 523.3 | 588.9 | 665.7 |
| `fpairs` | 704.9 | 705.0 | 703.9 |
| `ldst` | 4387.6 | 4387.2 | 4387.3 |

Packed arithmetic computes the same candidate values in one vector
instruction, using FMA only when enabled by the host target. Five more
alternating runs measured the last column. The generated code still
checks operands, results and FP status before committing them.

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

### Toward the game: audit, jump tables, depth, tests and measurements

**Audit** (`EmitOptions::call_summaries_audit`, `call-summaries-audit`,
with functions and call summaries). Every summarized call of a lifted
function (from a window or a function, directly or through the table)
runs as `window-calls-audit` runs window calls: all locals stored, the
registers, SP and NZCV outside the callee's summary filled with markers,
and when the callee comes back (or traps) a `Trap::InvalidState` ("summary
audit: the call at <site> (<callee>) changed <register>, outside its
summary") if one outside what it may write changed; then the caller's
values go back. For one checked game run before trusting the summaries.

**Jump tables.** A recovered jump table's BR (`Translation::jump_table_cases`,
its case targets by BR address) no longer ends a function: its cases are
part of the body, and a function whose every case is in its body still
leaves only by its return, so it (and its callers) can get a summary. A
target the table does not hold at run time is not in the body: the
function leaves, and its caller stores what it holds before going on
through the window table, as for any early return.

**Depth.** Summarized calls take no `Core::call` depth, so transitive
summaries now stop at `WINDOW_CALL_DEPTH_LIMIT` (16) nested levels, as
window calls do: a function that would nest deeper has no summary and is
called through `Core::call_window` (with its depth limit).

**Lift report.** `echora lift` also prints the BL targets over
`FUNCTION_BLOCK_LIMIT`, the functions holding a jump table, and the
instructions in function bodies with their share of the translation
(each body is a copy of window code that compiles again as a function).

**Tests.** `tests/window_calls.rs`'s game-shaped program now has a
switch through a halfword jump table in the code (a leaf, and a frame
calling it), a function of 600 blocks across three windows (over the
block limit), and a game-shaped function (a frame saving X19/X20, six
rounds of random bodies with CBZ skips and calls of the switch and its
frame, in an eight-round loop). Besides its window-call lifts it lifts
functions with the table, call and transitive summaries in one 64 KiB
region, in 1 KiB regions (bodies cut at region edges), audited, with
direct calls, with deferred charges and with the step count in a local,
each against the function-free lift (whole, sliced and budgeted runs, in
child processes: every general and vector register, SP, NZCV, FPCR,
FPSR, PC, step and boundary count, trap, data and stack), and the
integer twin's 64 KiB and 1 KiB function lifts against the reference
interpreter (which reads the table from the code): 792 runs each, none
skipped. `lifted_functions_cover_the_game_s_cases` checks the switch, its
frame and the game function are lifted with summaries and the long one is
not. The lifter's unit tests cover a function's jump-table cases and its
summary, the depth limit and the audit's text. The whole
`echora-aot-tests` suite passes with the game set plus `functions`,
`function-table`, `call-summaries` and `transitive-summaries`.

**Measurements.** Kernels lifted in 64 KiB regions on a shared 4-core
container, median Minstr/s of six interleaved rounds of seven runs
(ranges): game set + window-calls (wc), + window-calls, long-windows and
known-returns (wcl), and game set + functions, function-table,
call-summaries and transitive-summaries (fn). `gamefn` is a new kernel: an
update loop calling, per object, a function of about 70 instructions
(a frame saving X19-X22, loads, compares and branches, a four-way switch
through a jump table, calls of the clamp and flag leaves, stores back).

| kernel | wc | wcl | fn | fn against wc |
|---|---|---|---|---|
| calls | 1098 [903-1189] | 1021 [980-1029] | 1066 [635-1102] | -3% |
| calls2 | 965 [731-1000] | 947 [619-979] | 760 [512-787] | -21% |
| gamecalls | 1498 [1289-1580] | 1480 [1219-1574] | 1646 [888-1715] | +10% |
| crossing | 2113 [2063-2155] | 4778 [3216-4836] | 5251 [5151-5292] | +149% |
| gamefn | 2041 [1535-2094] | 2008 [1180-2031] | 2095 [1071-2152] | +3% |

Census (stores and reloads of register locals; calls are emitted call
sites):

| window or function | wc calls/spills/reloads | fn calls/spills/reloads |
|---|---|---|
| calls window | 2 / 2 / 0 | 4 / 6 / 6 |
| calls2 window | 3 / 5 / 3 | 6 / 15 / 18 |
| gamecalls window | 2 / 8 / 9 | 4 / 14 / 31 |
| gamecalls update (window; fn: `run_function_105000`) | 2 / 6 / 3 | 2 / 6 / 3 |
| crossing window | 1 / 9 / 11 | 2 / 9 / 14 |
| gamefn window | 1 / 9 / 11 | 2 / 9 / 22 |
| gamefn function (window; fn: `run_function_106800`) | 2 / 6 / 3 | 2 / 6 / 3 |

A function's body has the stores and reloads of the window it copies
(a summarized call keeps the rest in host registers either way); with
functions the windows hold a summarized and an unsummarized path per call
site, which is where their larger counts come from. On these kernels the
call boundary is no longer the cost: functions with summaries are about
level with window calls, ahead where a function runs across a window
edge (crossing, which long windows also cover), and behind on calls2
(within this container's noise of about ±30% on the other kernels).
Making the lifted code itself several times faster will take changes
inside blocks (the per-block charge and PC bookkeeping, guest memory
access, flags), not at calls.

## Window calls (opt-in)

`EmitOptions::window_calls` (`window-calls` in `ECHORA_LIFT_OPTIONS`; split
windows only, ignored with `structured` or `functions`) makes calls
between windows cheaper without lifting functions: no new copies of guest
code, only a summary per call target and a second name for the windows
calls enter. Nothing in `echora-guest` changes.

Without it a BL stores every dirty local, runs `Core::call` (window table
lookup, depth check, loop, indirect call), the callee's window records
the entry (`enter_window`: alignment check and the recent-PC ring),
dispatches the target and loads its locals, and the caller reloads every
local it holds afterwards. With it, a BL to a target T that has a summary
is a summarized call (the same rewrite as `call_summaries`): the caller
stores only the dirty locals T may read or write, calls T's window
directly (`enter_window_<w>` in its own region, `window_call_at(i)` of a
per-region copy of the root's window-call table elsewhere, as
`function-table` does for functions), and reloads only what T may write.
The other locals stay in host registers across the call. An entered
window's body is compiled once, in `enter_window_<w>` (direct calls, which
skip `enter_window()`'s diagnostics: a constant, aligned target);
`run_window_<w>` (the table and dispatcher path) runs the diagnostics and
calls it, one more call on that path (`Result<(), Trap>` returns through
memory, so it is no tail jump). The body was once inlined into both, which
doubled every entered window's machine code: on the game, bound by
instruction fetch, 213 MB of copies with window calls (379 MB with long
windows) in 2.7 GB of `.text`. The root's `install` fills every region's table; its
`window_calls_installed()` says whether it did, and emission asserts the
line it adds that to.

**Summaries** (`window_call_summaries`). A window entered at T runs its
blocks until its next PC leaves it. In this mode a RET (and BR X30, RETAA,
RETAB) leaves the window as such an exit does (`m.cpu.pc = next_pc;
return m.finish_block();`, so a RET into the same window goes through the
dispatcher or `Core::call`, which return at once when it is the return
address). T's blocks are those reached from T inside its window through
branches, both sides of conditional ones, fallthroughs and call return
points; any other BR, or a B to a PLT stub (run in place), may reach any
block of the window, so they all count. T gets a summary when those
blocks hold no BLR, SVC, UDF, unsupported instruction or traced block,
every BL in them is itself to a target with a summary, the calls form no
cycle and nest at most `WINDOW_CALL_DEPTH_LIMIT` (16) deep (summarized
calls take no `Core::call` depth, so the host stack is bounded here),
and T is no PLT stub and has its own dispatch arm. Unless
`window_call_early_exits` (`window-call-early-exits`) is on, those blocks
may also leave the window only by a return: no branch or fallthrough out
of the window, no other indirect branch, no call whose return point lies
outside it. Any other exit unwinds every summarized caller up to the
window the dispatcher runs (a cost), and early exits are what the game's
unexplained hang with call summaries had in common. The summary is the
union of `crate::defuse` effects (general registers, SP, NZCV; FPCR and
FPSR are stored and reloaded around every call as before) over those
blocks and the callees' summaries.

`echora lift` prints, with window calls on, the summarized targets,
their windows and BL sites, those of them that leave early, and how many
more early exits would add. `window_calls_audit` (`window-calls-audit`)
checks every summary while the game runs: each summarized call stores all
locals, fills the registers, SP and NZCV outside what the callee may
touch with markers, and when the callee returns (or traps) raises
`Trap::InvalidState` naming the call site and the register if one
outside what it may write changed, then puts the caller's values back.
Its emission is slower (every call stores and reloads everything); it is
for one run before trusting the real one.

**Why nothing else sees a stale `Core`.** While a summarized window runs,
only its reached blocks and summarized callees run: any exit returns
from it. When it comes back with the PC not at the return address (an
early return elsewhere, a branch out of the window, a call inside it that
came back early), the caller returns from its own window too (its final
store writes every local it holds), up to the window run by the
dispatcher or `Core::call`, which continues from the PC. A trap, an SVC,
a host boundary, preemption or the budget inside the callee comes back as
an error, after which every caller reloads what the callee may have
written and its final store writes the rest. Block charges, tags,
boundaries and step counts are those of the same blocks, unchanged. The
callee's window loads all its locals on entry and stores all it writes at
exit, also those its reached blocks never touch: those hold the `Core`'s
(possibly stale) value and write it back unchanged, and the caller, which
holds the true value, never reloads them.

**Coverage and gains** (`echora lift` prints the BL targets with
summaries, their windows and the BL sites calling them). Kernels lifted
in one 64 KiB region with the game's default set, against the same set
plus `window-calls`, on a shared 4-core x86-64 container, five rounds
(three of the call kernels alone, seven runs each; two of every kernel,
five runs each), median Minstr/s per round:

| Kernel | default set | + window-calls | change per round |
|---|---|---|---|
| calls (two leaves per iteration) | 580-614 | 833-889 | +38% to +46% |
| calls2 (a frame function calling both leaves) | 317-342 | 702-787 | 2.1x to 2.5x |
| gamecalls (game-shaped update loop) | 920-1198 | 1259-1614 | +8.5% to +71% (middle +37%) |

After both entries got the inlined body (and with early exits behind
their own option, which none of these kernels' targets need), three more
rounds of the call kernels gave medians of 419-610 against 746-791
(calls), 279-325 against 733-842 (calls2) and 868-984 against 1259-1388
(gamecalls). The second copy of the four entered windows grew the bench
binary's `.text` by 5.3 KB (1.0%); a game build pays it for every window
a summarized call enters, which `echora lift` counts.

Two full runs over every kernel moved only the call kernels' code (the
other kernels' emitted Rust is identical between the builds); their
changes there, -27% to +22% in both directions between rounds, are
layout noise.

Census (`echora_lift::census`, stores and reloads of locals inside window
bodies) of the kernels' windows, default set against + window-calls:

| window | spills | reloads |
|---|---|---|
| calls | 4 -> 2 | 6 -> 0 |
| calls2 | 11 -> 5 | 15 -> 3 |
| gamecalls | 12 -> 8 | 22 -> 9 |
| gamecalls update, hash | 15 -> 6 | 16 -> 3 |

and of the game-shaped random program of `tests/window_calls.rs` (7
windows, 272 call sites, random bodies using every free register, so
large summaries): per window 640 -> 538 spills and 1,249 -> 763 reloads
(-16%, -39%).

`tests/window_calls.rs` runs that program (frames, BL and BLR, B tail
calls, a jump through BR, BR X30 and RETAA returns, early returns to
another window, SVCs, recursion, UDF faults, PLT stub calls and tail
calls, scalar and vector float and SIMD instructions and moves between
general and vector registers, a function falling through into the next
window, a jump into a block's middle, and a caller holding a register its
callee's callee's early return reads) in pairs without and with window
calls: the build's options (in one region; through the table across
1 KiB regions; in every other region; with early exits; audited) and
those options with deferred charges, the step count in a local, dense
dispatch, direct calls and stub windows. 63 callers from 12 states each,
whole, sliced into host boundaries and against a budget; every general
and vector register, SP, NZCV, FPCR, FPSR, PC, step count, boundary
count, trap, the data memory and the stack must equal the lift without.
Each lift runs in its own process (the window table `Core::call` uses is
one per process); the table lift's recent-PC ring must equal the direct
lift's and differ from the same lift without `install` (whose table calls
return early), and every window-call lift must report its table
installed. An integer twin of the program is checked against the
reference interpreter without and with window calls (every run). A
summary without the callee's writes (the audit names the call and
register), RETs that do not leave the window, a BR not reaching the
whole window or a callee's early return continuing in place each fail
it. `tests/calls.rs` runs its callers with window calls against the
reference interpreter, each lift in a child process.

Summaries are only as good as `crate::defuse`. The semantic suites
(`semantics`, `simd_semantics`, `fp_semantics`, `simd_memory_semantics`)
check it against the lifted code of every instruction they sample
(`tests/common/effects.rs`, 52,408 checks over 13,102 instructions): each
runs from a state and again with every general register, SP and NZCV it
is said not to read changed (pointers into the data and small indices to
other in-range values); the trap, PC, vectors, FPCR, FPSR and memory must
come out the same, every register it may write the same in both runs and
every other one unchanged. That check does not reach system register
moves, PAuth helpers or paths its states never take; the lifter's
`defuse_effects_cover_every_emitted_register_access` covers them
statically: it decodes about 4 million words (random low bits under
every value of the top 16 and top 21 bits, every operand field of the
system, one-source and branch-to-register spaces, and random words; every
`Instruction` and `SimdOp` kind occurs), emits each one's Rust in both PC
forms and requires every `m.cpu.regs[n]`, `m.cpu.sp` and `m.cpu.nzcv`
access, flag operation and register helper argument in it to lie inside
its effects (an index that is not a literal, or the `Cpu` handed to a
helper not known to leave these alone, fails it too). It found no
mismatch. The whole `echora-aot-tests` suite passes with
every program lifted with the game set and `window-calls`.

## What a guest instruction costs inside a block (2026-11)

**Method.** The game compiles its region crates at `opt-level` 1 with one
codegen unit and no LTO (`echora`'s generated workspace), echora-guest at
3, for the native CPU; the kernel benchmark builds at 3 with LTO. So the
kernels were built the game's way (`--config
profile.bench.package.echora-aot-tests.opt-level=1`, codegen-units 1, no
LTO, echora-guest at 3; `-C target-cpu=x86-64-v3`, as valgrind decodes
no AVX-512) and counted under valgrind's cachegrind (host instructions
and its simulated branch mispredictions), the runs at `--scale 1` and
`--scale 2` differenced so setup cancels: deterministic, unlike timings
on this shared container (where removing work measured up to 49% slower,
layout and the host dominating). `small-address-space` (aot-tests
feature) makes the guest reservation fit valgrind. Each cost class was
removed from the kernels' emitted Rust by `ECHORA_KERNEL_ABLATE` (a
measuring-only rewrite in `build.rs`: the code is no longer faithful):
`charges` (no block charges), `pc` (no PC stores only a trap or boundary
reads), `memory` (accesses without the reservation and alignment tests).
Game set plus window-calls; host instructions per guest instruction
(mispredictions per 100 guest instructions):

| kernel | game set | no charges | no PC stores | o3 | structured | deferred-charges | steps-local |
|---|---|---|---|---|---|---|---|
| gamecalls | 14.95 (6.1) | 11.76 | 14.36 | 14.79 | 21.68 | 15.76 | 15.30 |
| gamefn | 12.41 (15.0) | 9.95 | 12.23 | 11.94 | 15.22 | 12.14 | 12.76 |
| crossing | 10.95 (10.2) | 9.66 | 10.72 | 9.99 | 10.96 | 10.51 | 10.77 |
| calls2 | 23.70 (38.5) | 16.00 | 23.23 | 23.00 | 45.96 | 23.70 | 24.54 |
| vector | 28.14 | 27.71 | 27.86 | 23.86 | 29.00 | 28.14 | 28.71 |
| floats | 17.11 | 16.67 | 17.06 | 16.84 | 17.45 | 17.11 | 17.28 |
| scalar | 147.88 | 145.87 | 147.50 | 144.25 | 149.75 | 148.00 | 148.38 |
| ldst | 5.20 | 4.20 | 5.00 | 5.60 | 6.40 | 5.60 | 5.40 |
| arith | 5.29 | 1.61 | 5.15 | 5.29 | 5.43 | 5.29 | 5.15 |

- **Block charges** are the largest single class: 20-21% of gamecalls'
  and gamefn's instructions, 12% of crossing's, 32% of calls2's, 70% of
  arith's (two-instruction blocks), 2-3% of the float kernels'. Each is
  seven host instructions: the stop word's pointer and then the word (two
  dependent loads), the steps loaded, added, compared, stored.
- **PC stores** cost 1-4%.
- **Memory checks**: the ablation's stand-in accessor made the code
  larger at `opt-level` 1 (it cannot be priced that way); statically each
  access adds a mask test and an alignment test, both predictable. In the
  machine code there is less: the aligned path (a relaxed atomic) and the
  misaligned one (`read_unaligned`) are the same x86-64 `mov`, which LLVM
  merges, so a plain access is `andn mask, address; jne cold` and one
  `mov` at the view's base plus the address (a dead `test $7` is left of
  the alignment test). Page protections are the guest's permissions, with
  no per-access lookup. See "Plain guest memory access" below.
- **Dispatch**: a block ending in `next_pc = <constant>` goes back
  through the window's `match` (a compare tree and a jump table; LLVM does
  not thread the constant through the loop header at any level). In the
  hot gamefn window a two-instruction block costs one useful instruction,
  seven for its charge and about eleven for the dispatch.
- **Register locals**: 22% of the hot gamefn window's instructions (265
  of 1,173) touch stack slots: its 16 locals and temporaries do not fit
  x86-64's registers.
- **`opt-level` 3** for the regions changes 0-15% (vector -15%): not the
  lever it might seem.
- `deferred-charges`, `steps-local` and `structured` give nothing at
  `opt-level` 1, or lose.

**Tried on this basis** (each opt-in, measured the same way):

- `threaded-successors`: a block's constant forward successor that keeps
  its own arm runs as a copy in the arm instead of going back through the
  `match` (no loop heads, no SVC or trap ends, `THREADED_LINE_BUDGET`
  lines of copies per arm; each block keeps its arm, so resuming is
  unchanged). With a 96-line budget and every successor: gamefn -7%
  (mispredictions 15.0 to 8.2), gamecalls -6%, crossing -6%, but arith
  +27% and ldst +12% (copying a loop's head into the outer arm changes
  the registers of the whole loop), and the aot-tests programs' Rust 9.7x
  larger. Without loop heads: gamefn -4%, the loops unchanged, 2.8x. With
  24 lines: gamefn -3%, the rest unchanged, 1.6x the Rust. Not worth its
  compile time on the game.
- `cached-stop`: the stop word in a window local (loaded on entry, at
  every backward dispatch and when an untagged charge does not clear it),
  so an untagged block's charge is a compare and an add without the two
  dependent loads; steps stay in the `Core`. +3-4% host instructions on
  the game-shaped kernels and +13% on arith: under x86-64's register
  pressure the local lives on the stack, and the backward-dispatch test
  adds work. `tests/steps_local.rs` checks it charges and stops exactly as
  per-block charging (every host check point and budget).
- `known-returns` (earlier): no change at `opt-level` 1.
- `arm-charges`: an arm (no dispatch inside, so its blocks form a tree)
  tests once that all its charges fit under the stop word, and its
  untagged blocks then only add their counts, with the exact checks where
  the test failed; `tests/steps_local.rs` checks it stops exactly as
  per-block charging. No gain (gamefn 12.31 to 12.33, ldst 5.20 to 5.60):
  few arms hold more than one untagged block, and the flag costs a
  register.
- One body per entered window (`enter_window_<w>`, above): the kernels'
  `.text` 550,947 to 544,371 bytes, executed instructions the same or
  fewer (gamecalls 14.95 to 14.73); the codesize programs with window
  calls 206.1 to 204.4 `.text` bytes per guest instruction (they enter
  few windows; the game's copies were 8% of its `.text`).
- The unaligned-SP trap built in a cold helper instead of an inline
  `format!` at every stack access: no measurable `.text` change (LLVM
  already merged those identical cold tails).
- `long-windows` (earlier), counted the same way: crossing 10.95 to 4.90
  host instructions per guest instruction (mispredictions 10.2 to 3.4
  per 100), every other kernel unchanged, with or without
  `known-returns`; the aot-tests programs' Rust 44% larger (75 to 109
  MB). The game measured about 8% slower with window-calls, long-windows
  and known-returns than plain (forest, three rounds): on the kernels
  neither option adds instructions, so the extra code (instruction cache,
  predictor capacity, which kernels this small cannot show) is the
  candidate; a profile of the two translations decides it.

**Where that leaves the cost.** On game-shaped code about 12-15 host
instructions per guest instruction: roughly 2.5-3 for the charges, 1-1.5
for the dispatch, a fraction for PC stores and memory checks, and the
rest the instructions' own work plus the stack traffic of locals that do
not fit the host's registers. No single change inside blocks found here
is worth more than a few percent faithfully; the next candidates are
fewer, larger blocks (charges and dispatch scale with blocks, not
instructions), and fewer live locals per window (smaller windows or
liveness-based locals) to cut the stack traffic.

## Long windows and known returns (opt-in)

A profile of the game's worker threads (forest scene, main) put 69% of
their CPU in translated windows, 80% of window entries in direct
window-to-window calls, and of those 21% to the neighbouring 1 KiB window:
`Core::call` is inlined at each BL site, so when a callee runs past the
end of its window, that window stores its locals and returns, and the
call loop in the caller's frame enters the next window, which loads them
again (and its `match next_pc` dispatches the PC afresh). A straight run
never crosses a window edge (`discover` splits blocks there), so every
function or loop laid across an edge pays this at each crossing. The
window table (`echora_guest::install_windows`, 1 KiB windows) is
unchanged.

**Long windows** (`EmitOptions::long_windows`, `long-windows`; split
windows, not structured, nor with lifted functions or deferred charges).
Each window's `match` also has arms for copies of the blocks its code
reaches outside the window (`window_blocks`): breadth first through
branches, both sides of conditional ones, fallthroughs and call return
points (no BL targets, no PLT stubs, no indirect branches), within the
same region, each block with those inlined into it, up to
`WINDOW_EXTENSION_LIMIT` extra instructions per window, and of those only
the ones from which the window's own blocks are reached again (loops
across its edge; a straight run out of the window leaves it once either
way). Window calls keep the targets plain windows give
(`window_call_targets`), each with its long window's summary. The first
version copied every reached block up to 256 instructions and let long
windows add targets: about 8% slower on the game (forest), its `.text`
4.4 GB against 2.7, fetch-bound; now 64 instructions and loops only. Every block
keeps its arm in its own window, which the dispatcher, `Core::call` and
resuming after a trap, boundary or budget use as before; a copy charges,
tags and stores the PC exactly as the original, so nothing but the number
of window entries changes (the recent-PC ring records fewer). Reaching a
block a window has no arm for leaves the window as before. Window-call
summaries (`window_call_summaries_in`) count a window's copies as inside
it, so a callee that runs across an edge and returns gets a summary
without early exits. `echora lift` prints how many windows hold copies
and their share of the translation: that much more Rust compiles (at most
one window's worth per window).

**Known returns** (`EmitOptions::known_returns`, `known-returns`). After a
call in a split window (through `Core::call`, `call_window` or a
summarized window call) the window continued with `next_pc = m.cpu.pc`,
a value the host compiler cannot see, so every return went through the
window's `match`. With this option it is `next_pc = <return address>;
if m.cpu.pc != next_pc { return m.finish_block(); }`: a constant on the
common path, which the compiler branches to directly, and on any other PC
(an early return, the call depth limit) the window leaves and its caller
or the dispatcher continues from the PC, as with any exit. Dense
dispatch (`dense_dispatch`, a jump table for the `match`) was measured
slower before (memset -18%, calls -4%) and is not changed.

Kernels lifted with the game's set plus `window-calls` (base), with
`long-windows`, `known-returns` and both, on a shared 4-core container,
median Minstr/s of six interleaved rounds of seven runs (ranges in
brackets):

| kernel | base | + long-windows | + known-returns | both |
|---|---|---|---|---|
| crossing (a function across a window edge, 4-round loop over it) | 2193 [1302-2209] | 4916 [2850-5176] (+124%) | 2066 [1340-2118] | 5003 [2954-5153] (+128%) |
| calls | 794 [688-1132] | 1067 [686-1095] | 950 [683-981] | 922 [702-1100] |
| gamecalls | 1189 [817-1574] | 1453 [850-1580] | 1449 [936-1539] | 1265 [830-1559] |

Long windows change only the crossing kernel's three windows (the other
kernels' emitted Rust is identical), so its calls and gamecalls columns
show this container's noise (±30%); known returns change the call
kernels' windows, and their gains there are inside that noise. A full
three-round run over every kernel moved only these kernels' code; the
rest moved -35% to +77% in both directions between builds of identical
code (layout and the shared host). Census of crossing's windows (default
set + window-calls, against + long-windows): the caller's window 9 -> 3
spills and 11 -> 3 reloads (its BL now has a window-call summary), the
function's first window holding the tail's stores and back-edge, every
other kernel's row unchanged.

The whole `echora-aot-tests` suite passes with the game set,
`window-calls`, `long-windows` and `known-returns` (with `long-windows`
its split programs lift in 64 KiB regions, the PLT program still in 1 KiB
ones); `tests/kernel_calls.rs` runs the call kernels (calls, calls2,
gamecalls, crossing) against the reference interpreter, whole and sliced,
under whatever options the build has; the lifter's unit tests cover the
copies, the limit, dense arms outside the window, the summaries and the
known-return text.

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
- LD2-LD4/ST2-ST4 (single structure, to a lane), LD1R-LD4R and
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

LD1 and ST1 of a single lane always use a checked scalar guest-memory
access at the lane width. Loads preserve the other lanes, and post-indexing
updates the base after a successful transfer.

`tests/vector_misc.rs` samples random words of these (one of each form,
120 of each kind), lifted with the option and without, and runs each from
vectors of float edges, table indices around the tables' sizes and random
bits, general registers and SP inside the data (for the other operations
every other run edge values), data of random bytes, FZ and DN on and off
and a rounding mode the float helpers trap on: every register, the data,
FPSR and the trap must agree; the lift with the option keeps no TBL or
structure transfer helper call.

## Vector float groups checked by the host's flags (opt-in)

**The game's hottest structured functions** (the regalloc candidate on the
forest) are 4S multiplies, adds and subtracts: their special-value guards
took about a quarter of their cycles, 2.3 times the arithmetic they
protect, and their slow paths almost never ran.

**`mxcsr-groups` (`EmitOptions::mxcsr_groups`, with `inline-vector`, on an
x86-64 host).** A group (the consecutive operations `vector_runs_in` takes,
or a single FADD, FSUB, FMUL, FDIV, FMLA or FMLS: `vector_groups_in`)
whose operations are all vector FADD, FSUB, FMUL, FDIV, FMLA or FMLS
(whole vectors or by element; 4S, 2S, 2D) computes natively with no
operand tests (`crate::simd::vector_group_mxcsr`):

- one test of the FPCR mode and the sticky IXC for the group;
- `mxcsr_clear` clears MXCSR's flags, then the operands load (the clear
  may write memory, so no load moves above it);
- each operation's result lanes add to `bad` where they are a NaN, a
  denormal or exactly the smallest normal (`mxcsr_suspect_4s`: SSE2
  compares and a movemask; doubles lane by lane);
- `mxcsr_read(bad)` reads MXCSR (with `bad` as an operand, so every result
  is computed before it); the group commits when IE, DE, ZE, OE and UE are
  clear and no lane was bad, else the exact copy reruns it from the
  unchanged registers.

Why that is exact: a denormal operand raises DE (the host's DAZ and FTZ
are off), an invalid operation or a signalling NaN IE, a division by zero
ZE, an overflow OE, a tiny inexact result UE; infinities and zeros
otherwise give the IEEE results Arm gives and raise nothing; a quiet NaN
operand gives a NaN result, which the lane test sees (so do DN's default
NaNs). The lane test covers what the flags miss: x86 detects tininess
after rounding and raises nothing for an exact denormal result, where Arm
detects it before rounding and, under FZ, flushes and raises UFC, also for
a value overwritten later in the group (every result is tested, not only
the final ones). Groups with FMLA or FMLS need the host's FMA (`fmaf` in
software rounds through double): without `target_feature = "fma"` they
always take the exact copy. Groups with other operations keep the runs'
own fast copy.

`tests/mxcsr_groups.rs` lifts each form alone and a chain reusing
registers, with the option, against the helpers alone: 60,000 runs from
lanes drawn from normal, inexact, denormal, smallest normal, values
rounding to it, zeros, infinities, quiet and signalling NaNs and values
that overflow or underflow, under FZ, DN, round toward minus infinity and
FPSR with and without IXC; vectors, FPCR, FPSR and the trap must agree
(with and without FMA, and with regalloc). Dropping the lane test,
ignoring the flags, or ignoring UE alone each fail it.

Counted as above (`x86-64-v3`, so with FMA), the game set plus
`window-calls` ("game") and the regalloc set, each with
`vector-float-runs`, against the same with `mxcsr-groups`. `vchain` is a
new kernel shaped like the game's functions: two vectors in, twelve 4S
multiplies, adds and subtracts reusing V2-V5, one vector out:

| kernel | game | + mxcsr-groups | regalloc | + mxcsr-groups |
|---|---|---|---|---|
| vector | 21.72 | 12.86 (-41%) | 20.94 | 12.27 (-41%) |
| vchain | 13.24 | 10.24 (-23%) | 12.87 | 10.01 (-22%) |
| vnorm, fpairs, misc, floats, gamefn | | unchanged | | unchanged |

vnorm, fpairs and misc hold forms the groups do not take (FRSQRTE,
pairwise operations, FABD, FMAXV), so their runs keep their own fast copy.
The kernel binary's `.text` shrank 4.5 KB (game) and 8.9 KB (regalloc).
The golden hashes moved by the new lifts alone (the `vchain` kernel and
`mxcsr_group_code`'s programs; with both removed they are as before):
`0x279a_09f4_0a57_d429` to `0xa3b3_121e_c0e4_5cd6`, and
`0x153e_e794_08c5_1d4b` to `0xfdab_2aea_d9c7_830f`.

A first version tested the result lanes in a `for` loop over the `u128`:
at `opt-level` 1 the host compiler neither unrolls nor vectorizes it, and
vchain went from 13.2 to 30.8.

**`__m128` values and one test per group.** A group's 4S values are now
`__m128` locals (`mx_singles` at the first use, `mx_bits` at the commit;
2S and 2D lanes stay `u128`), and its arithmetic the SSE intrinsics
themselves (`mx_add`, `mx_mul_add`, `mx_lane` for by element). That alone
changed no machine code: LLVM had already kept the values in SSE
registers through the `u128` transmutes. What the disassembly showed
instead was the lane test: seven instructions after each one-instruction
operation. A group now tracks its 4S results' extremes (`mx_track`: the
unsigned minimum of each lane's magnitude minus one, a zero wrapping to
the top, and the unsigned maximum of its magnitude; four instructions per
result) and tests them once at its end (`mx_track_bad`: the same
conditions as two SSE4.1 unsigned compares and one movemask). Its first
form reduced the extremes lane by lane in scalar code, and the vector
kernel's three-operation groups lost 18% to it.

| kernel | before mxcsr-groups | per-result tests | one test per group |
|---|---|---|---|
| vchain (game) | 13.24 | 10.24 | 8.18 (-38% in all) |
| vchain (regalloc) | 12.87 | 10.01 | 7.95 (-38%) |
| vector (game) | 21.72 | 12.86 | 11.72 (-46%) |
| vector (regalloc) | 20.94 | 12.27 | 11.13 (-47%) |

vnorm is unchanged; `.text` within 200 bytes. `tests/mxcsr_groups.rs`
fails with the minimum's half or the maximum's half of the tracking
removed, or with the NaN half of the final test.

## Step accounting: what needs exact counts (2026-11)

**Who uses the step count.** Generated code charges each block's
instructions to `Core::steps`; with `regalloc` a structured function counts
them in a local and tests the stop word (`steps + n + ahead > stop`) at its
entries, loop heads and call returns, leaving to the windows, which take
budgets and host boundaries on the same block as window code. The users:

- **Guest-visible values: none by default.** `MRS CNTPCT`/`CNTVCT`
  (`physical_counter_ticks`) reads host time (`PhysicalCounter` with no
  rate: `epoch.elapsed()`); steps feed it only with the debug variable
  `ECHORA_VIRTUAL_CLOCK`, where only monotonic and roughly proportional
  values matter. NVN timestamps come from the same counter.
  `GetSystemTick` (SVC 0x1E) is not implemented; `SleepThread` and timed
  waits use host `Instant` deadlines. No SVC returns a value derived from
  steps.
- **Approximate progress accounting:**
  - scheduler slices (1,000 steps, 50,000 with parallel workers), the
    timed-wait and audio polling cadence (`HOST_TIMER_CHECK_INSTRUCTIONS`,
    100,000);
  - wake-scan throttling (5,000), spin backoff and density (200,000 and
    64), `spin_hint`'s slice end;
  - per-thread instruction credits, which are only reported.
  Cross-worker preemption is driven by priority, not steps (another thread
  sets `preempt` and stores 0 into the stop word), and the run report
  already states that switch points depend on the generated block
  partition, not exact instruction positions.
- **Budget:** `u64::MAX` for players (the launcher); 1e9 by default for
  CLI test runs, and synthesised from wall-clock limits. A test stop, not
  semantics.
- **Exactness only for tests and diagnostics:** budget traps at exact
  steps and PCs, every host boundary's PC and step count compared between
  lift options (`steps_local`, `counter_reads`, `kernel_calls`,
  `window_calls`, the runtime's unit tests), the per-instruction refund of
  a trapping block (`settle_block_trap`), trace hits.

The one hard requirement is where execution stops, not the count: a host
boundary or budget must leave the PC at a block start that can be resumed,
with the state consistent at SVCs and traps.

**What it costs with regalloc.** Kernels lifted with the regalloc set and
`regalloc-loop-inlining`, counted as above; "no stop tests" removes the
structured functions' stop word tests and "no step adds" also their
per-block `*n += count` (`ECHORA_KERNEL_ABLATE=stop` and `stop,steps`,
measuring only: neither is faithful):

| kernel | regalloc | no stop tests | no stop tests, no step adds |
|---|---|---|---|
| gamefn | 2.67 | 2.63 (-1%) | 2.60 (-3%) |
| gamecalls | 2.12 | 2.24 (+6%) | 2.05 (-3%) |
| crossing | 2.30 | 1.88 (-18%) | 1.68 (-27%) |
| calls2 | 1.39 | 1.31 (-6%) | 1.23 (-12%) |
| calls | 1.23 | 0.36 (-71%) | 0.36 |
| vcalls | 11.81 | 9.43 (-20%) | 8.90 (-25%) |
| arith | 2.43 | 1.39 (-43%) | 1.39 |
| ldst | 4.86 | 2.60 (-46%) | 2.60 |
| memset | 7.21 | 5.67 (-21%) | 4.00 (-45%) |
| switch | 2.55 | 2.18 (-15%) | 1.91 (-25%) |

The game-shaped kernels (gamefn, gamecalls) move by 3%. The large figures
are tight loops that, with no test left, have no side effect in the loop,
so the host compiler collapses, unrolls or vectorizes them (calls -71%,
arith -43%); a thread that must stay preemptible cannot have that. What a
test actually costs shows in the machine code: in the ldst loop the stop
word's pointer and the step count are already in registers (LLVM hoisted
the `Arc` load), and the test is `mov 0x10(%r14), %rax; add $5, %rdx; cmp;
ja`: one relaxed load and two or three instructions per iteration, plus
the add of the count. Testing the stop word through a pointer kept in a
local (tried as an option: the window body's `let stop_word = &*m.stop`)
changed no kernel (gamefn 2.67 to 2.71, the rest identical) and was
dropped.

**The poll-flag design, and why it is not built.** A loop head testing
only a per-thread flag (one relaxed load and a branch, about one
instruction less than today's test) would end a slice only when something
set the flag: slices are measured in steps (`host_check_at`), and nothing
but the steps reaching it ends one today. It needs a host-side source of
slice ends, such as a timer thread storing 0 into each running core's
stop word every slice (the cross-worker preemption path already does that
store), with steps then charged coarsely for the reports and the wait and
spin heuristics (per call and per loop iteration, an estimate). That is
an echora-runtime change, and it would make every exact-step test
compare differently (each would need the exact accounting kept as a
mode). The gain it can buy is the one instruction per loop iteration
above, plus the per-block adds (memset's 21% to 45% is the most; the
game-shaped kernels 1% to 3%). Not worth it before the game shows tight
loops of this kind dominating.

**A finding on the way: the alignment branch under regalloc.** In the
structured ldst loop, unlike window code, LLVM did not merge the view's
aligned (relaxed atomic) and misaligned (`read_unaligned`) paths: each
access is `test $7, %reg; je` over two identical `mov`s. That is the
branch the one-instruction access lowering ("Plain guest memory access"
below) removed, and why it gained 7.4% on ldst with regalloc while losing
on the window kernels. An inline accessor in `echora-guest` choosing the
same instruction for both (or a single `asm!` access, as that lowering
did) would remove it for regalloc; that is a guest change, for the batch.

## Floating point on the regalloc set (2026-11)

Counted as above (the kernels built the game's way, cachegrind, scale 2
minus scale 1), with the game set plus window-calls ("game") and the
regalloc set ("regalloc"); host instructions per guest instruction.

**`float-run-loads` takes only loads between a run's operations.** A
run (`float_runs_in`) no longer starts or ends with S loads: a load that
leads or ends it runs before or after it as it would alone. The floats
kernel's block starts with its three loads, which the run held; floatld's
loads come between its operations.

| kernel | no loads | loads before | loads now |
|---|---|---|---|
| floats (game) | 17.11 | 17.89 (+4.5%) | 17.11 |
| floatld (game) | 26.00 | 18.37 (-29.3%) | 18.63 (-28.3%) |
| floats (regalloc) | 17.05 | 17.89 (+4.9%) | 17.05 |
| floatld (regalloc) | 25.98 | 18.36 (-29.3%) | 18.41 (-29.1%) |

`.text` of the kernel binary: 538,851 to 538,707 bytes (game), 616,115 to
616,019 (regalloc). The golden hashes moved with it (default
`0xbf59_98a9_1cf0_05b2` to `0x14cd_5b18_6741_5324`, inline-vector
`0xcf9f_4071_0fe5_727c` to `0xdd3b_fcd4_ea31_f1ae`), only through
`float_run_load_code`, whose programs lift with float-run-loads on in every
mode; with the trim undone both hashes are as before. Merged with
`claude/vigilant-dirac-e24yg5` at 11d52d9 (whose own hashes are
`0xc1cb_bbf7_f3dc_8b4b` and `0xefbe_5e0a_726e_0329`) they are
`0x73bb_c562_6345_cce9` and `0x0946_818b_2c33_349f`, again its own with the
trim undone; with the indirect call sites' statics named in upper-case hex
(`CALL_SITE_{pc:X}`, so clippy's `non_upper_case_globals` passes on the
lifted test programs) `0x9047_f551_4f90_3869` and
`0xe217_ba2b_c432_2edf`. Merged with 5db8463 (its own hashes
`0x5903_28ef_ed10_45e4` and `0x00d8_4736_2f4d_2c4a`, which this tree
reproduces with the trim and the upper-case names undone):
`0xa480_c0ef_6f90_8bd2` and `0xb832_7cb2_b09e_bf28`. Merged with 20bc158
(its own `0x9e84_3648_63e5_219a` and `0xc443_a540_49d7_6f08`, reproduced the same way): `0x2b43_4d11_deee_88f8`
and `0xedda_60da_f011_faae`. Merged with d463ae0 (which names call-site
caches in upper case too; its own `0x7cf4_cca0_c48c_5dc6` and
`0x3974_9197_5f6e_b628`, reproduced with the trim undone):
`0xf2d8_5f49_31a5_18a0` and `0x3cbf_5c35_44e7_725a`. Merged with d85a654
(its own `0x6692_75bb_232c_5e13` and `0x1a35_630d_2c0b_b08d`, reproduced
with the trim undone): `0x279a_09f4_0a57_d429` and
`0x153e_e794_08c5_1d4b`. `codegen::tests::float_runs_hold_only_loads_between_their_operations`
checks the run's bounds; `tests/float_runs.rs` the semantics.

**`cold-fp-exact` (opt-in, `EmitOptions::cold_fp_exact`).** A float run's
exact copy is marked cold (`core::hint::cold_path`), and the exact scalar
helpers the inline paths fall back to (`fp_binary`, `fp_ternary`,
`fp_compare`) are called through one `#[cold] #[inline(never)]` wrapper
per region. With float-run-loads, against the same without:

| kernel | game | + cold-fp-exact | regalloc | + cold-fp-exact |
|---|---|---|---|---|
| floats | 17.11 | 17.06 (-0.3%) | 17.05 | 16.92 (-0.8%) |
| floatld | 18.63 | 18.21 (-2.3%) | 18.41 | 18.28 (-0.7%) |
| fpscalar | 66.34 | 66.34 | 65.50 | 65.43 (-0.1%) |
| scalar | 147.88 | 148.13 (+0.2%) | 146.50 | 146.65 (+0.1%) |

vector, vnorm, misc, gamefn and gamecalls are unchanged. The hot float
windows' machine code moves by -3% to +4% (floats' function 7,440 to
7,236 bytes; floatld's window 7,495 to 7,766 without regalloc). The exact
helpers were already out of line (echora-simd's bodies are separate
functions; the wrappers are 38-byte calls), so their call sites were
small already: the hot float code is the inline fast paths' own checks
(operands and results normal or zero, the FPCR mode and IXC test per
operation outside runs). Not worth turning on for its gain alone.

**The subtraction V flag.** Fused flags (`fused_condition`) already lower
a subtraction's GE, LT, GT and LE to one signed host compare, as regalloc's
`LazyFlags` do. VS and VC were the bit formula `((a ^ b) & (a ^ v)) >>
63`, six host instructions; they are now the signed subtraction's
overflow (`overflowing_sub`, `cmp` and `seto`, or one `jo` in a branch),
and an addition's V (CMN, ADDS) its `overflowing_add`, a CMN's GE six
host instructions to four. No kernel branches on VS or VC; the fused
emission changes only there (not in the golden modes), and
`tests/fused_flags.rs` checks every condition after every setter against
the reference. Conditions that do not fuse in window code (a CSEL, a
condition in another block) read the flags `echora-guest`'s `Flags`
computed, which compute V by the formula and GE as `n == v`.

**`hoist-fp-mode` (opt-in, `EmitOptions::hoist_fp_mode`).** Two or
more consecutive inlined float operations in a block, outside float runs,
test the inline paths' FPCR mode and sticky IXC once, before the first
(`let fp_mode_<pc> = (mode && IXC, FPCR != 0 && mode && IXC)`), and each
operation reads that instead of testing again; each keeps its operand and
result tests and its exact fallback. No helper writes FPCR or clears an
FPSR flag (echora-simd's writes of either are in its tests), so a test
true before the group stays true, and one false stays false: the group's
operations then take their exact helpers, as each would until IXC was
raised. The fallback needs no restart. An MSR, or any instruction without
the test, ends a group. Groups are per block (`emit_block_body`, which
windows and regalloc's structured functions share); a structured
function's straight-line stretch across blocks would need regalloc's own
emission.

`tests/fp_mode_groups.rs` lifts groups of FMUL, FADD, FDIV, FMADD and
FSUB (singles and doubles, alone and with an MSR FPSR, which can clear
IXC, or an MSR FPCR between them, and with FCMP and FMAX among them) with
every inline path and the option, against the helpers alone: 7,120 runs
from normal, inexact, denormal, zero, infinite, NaN and overflowing
operands under FZ, DN, round-toward-minus-infinity, FPSR with and without
IXC and each value the MSRs write; vectors, FPCR, FPSR, NZCV and the trap
must agree, with and without regalloc. A hoisted test forced true fails
`tests/float_runs.rs`; groups running past an instruction without the test
(an MSR) fail it with 8 mismatches.

| kernel | game | + hoist-fp-mode | regalloc | + hoist-fp-mode |
|---|---|---|---|---|
| vnorm | 93.79 | 93.49 (-0.3%) | 94.11 | 93.33 (-0.8%) |
| fpscalar, floats, floatld, misc, gamefn | | unchanged | | unchanged |

With `inline-fp-scalar` too (fpscalar 19.74, or 18.11 with regalloc,
against 66.34 without it) nothing more moves: its conversions, FSQRT and
FRINT test FPCR without IXC, and FMAX and FCMP do not use this test, so
they form no group. The kernels hold three groups (vector, scalar, vnorm);
the kernels' window and function code grows 203 bytes without regalloc
and not at all with it. The per-operation mode test is not where the
float kernels spend: it is a few predictable instructions on locals, and
the operand and result tests dominate. Not worth turning on.

## Plain guest memory access (2026-11)

A plain load or store (`wmem.load_u*`/`store_u*`, `GuestAccess` in
`echora-guest`) is already the view's host base plus the guest address,
after one test against the reservation's power-of-two mask (an address
outside it panics). The view's page protections hold the guest's
permissions: an access the guest may not make faults in the host, whose
handler names the guest address before the process ends (no guest data
abort is delivered). Exclusive and ordered accesses check natural
alignment, LSE atomics a 16-byte block, SP-based accesses SP's alignment,
each a trap. Nothing on this path consults MMIO or GPU tracking.

An opt-in lowering was tried (not kept): each plain access as one `asm!`
`mov` (`ldr`/`str` on AArch64) at the base plus the address, after the same
range test, to drop the alignment branch. Counted as above (game set plus
window-calls, and the regalloc set), host instructions per guest
instruction:

| kernel | game set | + one-instruction access | regalloc | + one-instruction access |
|---|---|---|---|---|
| ldst | 5.20 | 5.60 | 4.86 | 4.50 |
| memset | 8.34 | 8.34 | 7.21 | 7.50 |
| gamefn | 12.49 | 12.44 | 4.68 | 5.14 |
| gamecalls | 14.98 | 15.41 | 4.66 | 4.74 |

The alignment branch was already gone from the machine code (above), so
the lowering saved one dead instruction per access and lost it to
register allocation: the asm cannot fold into memory operands, and its
mask lived in a stack slot. With the base taken through
`GuestView::host_span` (a call at every window entry) gamecalls was +20%;
the figures above read it through inline accessors. What remains per
access is the range test (two instructions and a register). Removing it
faithfully needs either a constant base and mask (a fixed-address
reservation: one guest memory per process) or proof that an address lies
within the reservation's 1 MiB inaccessible tail of one already tested
(the second half of a Q or pair access, a base used again with a small
offset), which turns the panic at the space's end into a host fault.

`tests/memory_faults.rs` checks the faults and traps of every plain,
exclusive, ordered, atomic and SP-based form against a model of what each
must do: completed accesses against the reference interpreter, traps by
kind and address with nothing changed, the panic outside the space, and,
in child processes, accesses to read-only, unmapped, translated-code and
past-the-end bytes ending the process by SIGSEGV with the handler naming
one of those bytes.

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
`inline-fp-scalar`, `inline-vector-misc`, `direct-q-pairs` and
`vector-float-runs` (the
checked column gives in brackets the accesses made by an out-of-line call,
`Cpu::load_q_pair`/`store_q_pair`: 4 of memcpy's 4 without
`direct-q-pairs`):

| function | loads 1/2/4/8 | stores 1/2/4/8 | checked (out of line) | calls | indirect | flags kept/fused | spills (status) | reloads (status) | charges | helper calls | inlined | float tests (mode) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| arith | 0/0/0/0 | 0/0/0/0 | 0 (0) | 0 | 0 | 1/0 | 0 (0) | 0 (0) | 4 | - | - | 0 (0) |
| memcpy | 0/0/0/0 | 0/0/0/0 | 4 (4) | 0 | 0 | 1/1 | 0 (0) | 0 (0) | 5 | - | - | 0 (0) |
| memset | 0/0/0/0 | 0/0/0/2 | 0 (0) | 0 | 0 | 1/1 | 0 (0) | 0 (0) | 5 | - | - | 0 (0) |
| calls | 0/0/0/0 | 0/0/0/0 | 0 (0) | 2 | 0 | 1/0 | 4 (1) | 6 (2) | 5 | - | - | 0 (0) |
| (leaf functions) | 0/0/0/0 | 0/0/0/0 | 0 (0) | 0 | 2 | 0/0 | 0 (0) | 0 (0) | 2 | - | - | 0 (0) |
| switch | 0/1/0/0 | 0/0/0/0 | 0 (0) | 0 | 1 | 1/0 | 0 (0) | 0 (0) | 12 | - | - | 0 (0) |
| vector | 0/0/0/4 | 0/0/0/0 | 0 (0) | 0 | 0 | 1/1 | 6 (6) | 12 (12) | 5 | vec_fp 3, vec_fp_lane 3 | vector 3 | 13 (1) |
| poll | 0/0/0/0 | 0/0/0/0 | 1 (0) | 0 | 0 | 0/0 | 0 (0) | 0 (0) | 3 | - | - | 0 (0) |
| scalar | 0/0/1/0 | 0/0/1/0 | 0 (0) | 0 | 0 | 1/1 | 1 (1) | 7 (7) | 5 | fp_binary 1, fp_compare 1, fp_ternary 1 | vector 5 | 7 (3) |
| ldst | 0/0/0/1 | 0/0/0/1 | 0 (0) | 0 | 0 | 1/1 | 0 (0) | 0 (0) | 5 | - | - | 0 (0) |
| floats | 0/0/3/0 | 0/0/1/0 | 0 (0) | 0 | 0 | 1/1 | 0 (0) | 24 (24) | 5 | fp_binary 9, fp_ternary 2, fp_unary 1 | fp-scalar 1, vector 11 | 15 (1) |
| floatld | 0/0/4/0 | 0/0/1/0 | 0 (0) | 0 | 0 | 1/1 | 0 (0) | 24 (24) | 5 | fp_binary 9, fp_ternary 2, fp_unary 1 | fp-scalar 1, vector 11 | 27 (5) |
| calls2 | 0/0/0/2 | 0/0/0/2 | 0 (0) | 3 | 1 | 1/0 | 11 (1) | 15 (3) | 7 | - | - | 0 (0) |
| vnorm | 0/0/0/2 | 0/0/0/2 | 0 (0) | 0 | 0 | 1/1 | 12 (12) | 24 (24) | 5 | vec_fp 5, vec_fp_lane 5, vec_fp_unary 2 | unary 1, vector 5 | 14 (1) |
| convert | 0/0/0/2 | 0/0/0/2 | 0 (0) | 0 | 0 | 1/1 | 6 (6) | 12 (12) | 6 | vec_fp 1, vec_fp_lane 1, vec_fp_unary 4 | int 1, unary 2, vector 3 | 6 (1) |
| intvec | 0/0/0/4 | 0/0/0/2 | 0 (0) | 0 | 0 | 1/1 | 0 (0) | 0 (0) | 5 | - | int 9, vector 2 | 0 (0) |
| fpairs | 0/0/0/4 | 0/0/1/2 | 0 (0) | 0 | 0 | 1/1 | 0 (0) | 34 (34) | 5 | vec_fp_lane 13, vec_fp_pairwise 4 | pairwise 4 | 21 (1) |
| fpscalar | 0/0/1/0 | 0/0/1/0 | 0 (0) | 0 | 0 | 2/1 | 15 (3) | 29 (20) | 5 | fp_binary 1, fp_compare 1, fp_convert 2, fp_to_int 1, fp_unary 2, int_to_fp 1 | fp-scalar 9, vector 2 | 8 (8) |
| misc | 0/0/0/2 | 0/0/0/0 | 3 (0) | 0 | 0 | 1/1 | 8 (8) | 16 (16) | 6 | vec_fp 3, vec_fp_across 1, vec_fp_lane 3, vec_fp_unary 1 | int 1, misc 8, vector 1 | 8 (1) |

The slowest kernels per guest instruction are the vector float ones
(vector, vnorm, misc, fpairs: 280 to 470 Minstr/s against 1,600 to 5,400
for the integer ones): none of their hot paths calls a helper (every
helper call left is a fallback, and `count-fallbacks` counts none in any
kernel) or spills, so what they spend is the inlined float operations
themselves, 3 to 11 a loop, each testing its operands and results and
the FPCR. memcpy moved its Q register pairs through the out-of-line
`Cpu::load_q_pair`/`store_q_pair` (4 a loop, 1,590 Minstr/s against
ldst's 5,400 with direct accesses); `direct-q-pairs` writes their body out
(below: +7.9%), and its accesses stay range-checked copies. The FPCR and FPSR reloads (up to 34 a window) are
around those fallback calls, off the hot paths; switch spends its
operations on block charges (12 for 11 blocks).

## LDP/STP of Q registers written out (opt-in)

`Cpu::load_q_pair`/`store_q_pair` (out of line in echora-guest) make one
`GuestView::read`/`write` of the 32 bytes: a range test against the guest
address space (a memory fault trap, nothing moved, when it fails) and one
copy; inside the space, page permissions fault in the host and end the
process, as for every other access. `EmitOptions::direct_q_pairs`
(`direct-q-pairs`) writes that body out in place of the call: the same
`read`/`write`, both registers written after the read, the base's
writeback after the access as before. The semantics are the call's by
construction.

`tests/q_pairs.rs` runs LDP and STP in every addressing form at offsets
-1024, -16, 0, 16 and 1008, with an ordinary base or SP, and a store then
an overlapping load, lifted with the option and without: inside mapped
memory (pairs across a page boundary among them) registers, SP, data and
the trap agree; past the end of the address space (straddling it, at it,
far past, wrapping) both trap alike with nothing moved and no writeback;
on unmapped memory inside the space (straddling the data's end, a page
past it) each run is a child process, and the fault message and signal
agree. (LDP with both registers the same does not decode.) Swapping the
halves, or dropping the range test, fails it.

memcpy, every function 64-byte aligned (`align-windows`' flag), 30
alternating rounds of seven runs, pinned: median 1,636 Minstr/s without,
1,765 with (+7.9%, a 95% interval of +5.7% to +11.6% by resampling: not
noise). The census shows the Q pairs' 4 checked accesses no longer out of
line. A first version that read and wrote the pairs as eight direct
8-byte accesses (after the same range test) was 27% slower (1,582 to
1,149, interval -27.9% to -26.8%): each 8-byte guest access is a relaxed
atomic, so the compiler cannot merge them into the 16-byte moves of the
call's copy, and echora-guest has no 16-byte direct access.

## Vector float runs (opt-in)

The vector float kernels spend their operations in the inlined float
operations' tests: forcing every operation's fast path (wrong, as a bound)
made vector 89% and vnorm 158% faster. `EmitOptions::vector_float_runs`
(`vector-float-runs`, with `inline-vector`) treats two or more consecutive
vector float arithmetic operations of a block (the forms `vec_fp_arith`
inlines: FADD, FSUB, FMUL, FMLA, FMLS, FDIV, FRECPS, FRSQRTS and the
by-element forms; not a block's first or last instruction) as one run,
emitted twice:

- the fast copy computes every operation on locals, a result feeding the
  operations after it without going through the register file; it tests
  the FPCR mode and FPSR.IXC once for the run, each operand vector from
  before the run once with the whole-vector tests (an operand that is an
  earlier result of the run, of the same lane width and lanes, is not
  tested again: it was tested as a result), and each result with
  `normal_all`, lane by lane only when that fails (a zero must be exact);
  an overflow is not taken (it raises OFC). It writes nothing until every
  test has passed, and then raises nothing;
- otherwise the exact copy runs: the operations as each is emitted alone,
  from the run's start, so every fallback (the single-lane helper, the
  whole helper under another FPCR) raises its flags in today's order.

Nothing between a run's operations writes FPCR, calls a helper or leaves
the block: they are consecutive (an MSR FPCR ends a run). The census
(above, with the option) counts the tests of the path taken when the fast
copies commit: vector 23 to 13 test sites (FPCR mode tests 3 to 1), vnorm
36 to 21 (6 to 3); misc and fpairs hold no such run (17 and 31, unchanged).

`tests/vector_runs.rs` lifts 600 programs of two to six such operations on
V0 to V3 (results feed later operations, registers alias; a quarter write
FPCR from X5 in the middle), with and without the option, and runs each
from mostly ordinary lanes with float edges (NaN payloads, signalling NaNs,
denormals, infinities, zeros, the extremes) among them, so an exceptional
lane falls mid-run, under the three FPCR and FPSR settings, FPSR without
IXC and a trapping rounding mode, X5 any of them: vectors, FPCR, FPSR and
the trap agree. Dropping the result test, treating every operand as
tested, committing only the first register written or dropping the IXC
condition fails it; a divisor tested as usable instead of normal does not
(a zero divisor's result is infinite or a NaN, which the result test
rejects).

Kernels, every function 64-byte aligned, 20 alternating rounds of seven
runs, pinned, medians with a 95% interval by resampling: vector 261 to 304
Minstr/s (+16.8%, +10.1% to +23.7%), vnorm 239 to 318 (+33.5%, +30.9% to
+35.8%); misc and fpairs (no runs) +0.4% and +0.1%, within noise. A first
version that tested every lane of every operand and result one at a time
had fewer test sites but was slower (vector -8.5%, vnorm -4.3%): the
whole-vector tests cost less than per-lane ones.

### More forms in a run

A run also holds the forms below, each under the conditions its own inline
path takes a lane in (stated where `crate::simd::run_other` emits it), so
vnorm's run is no longer cut in two by its FRSQRTE, and misc's and
fpairs' float operations form runs of their own:

- FMAX/FMIN/FMAXNM/FMINNM: both lanes normal or zero (then the
  NaN-handling forms agree and nothing is raised); the result, one of
  them (+0 above -0), is normal or zero.
- FCMEQ/FCMGE/FCMGT/FACGE/FACGT and the compares against zero: lanes
  normal or zero; the result is an integer mask, never taken as a tested
  float operand by a later operation.
- FABD, FMULX (and by element): as the inlined arithmetic.
- FABS/FNEG: the sign bits, nothing tested; the result as tested as its
  operand.
- FSQRT, FRINT*, FRECPE, FRSQRTE, FCVT*S/U toward zero: each lane under
  `inline-vector-unary`'s own condition (a positive normal for the
  estimates, whose results are then normal; a fitting truncation for the
  conversions, whose results are integers).
- SCVTF/UCVTF: nothing tested (an integer converts to a normal or a zero).
- FADDP/FMAXP/FMINP/FMAXNMP/FMINNMP: each pair normal or zero, and an
  FADDP sum normal or zero.
- FMAXV/FMINV/FMAXNMV/FMINNMV: all four lanes normal or zero.

`tests/vector_runs.rs` samples all of them (197 forms), with adjacent
lanes whose pairwise sum overflows, is denormal or meets a NaN added to
its inputs. A mutation of each new form's rule fails it: FMAX/FMIN,
FABD/FMULX, the compares against zero or FMAXV without their operand
tests, a compare's mask or an FCVT's integer taken as a tested float, FABS
or FNEG passing on a test its operand did not have, the unary forms
without their native condition, FADDP without its result test, FABD
without clearing the sign.

The census, with the option, counts the committing path's float tests
(FPCR mode tests): vnorm 36 (6) to 14 (1), misc 17 (5) to 8 (1), fpairs 31
(4) to 21 (1), vector 23 (3) to 13 (1). Kernels, every function 64-byte
aligned, 20 alternating rounds of seven runs, pinned, medians with a 95%
interval by resampling: vector 265 to 304 Minstr/s (+14.7%, +13.4% to
+18.2%), vnorm 239 to 393 (+64.2%, +61.0% to +71.6%), misc 246 to 286
(+16.2%, +14.3% to +18.0%), fpairs 377 to 515 (+36.8%, +32.4% to +41.4%):
none within noise.

In a gameplay comparison on the Ryzen 7 5800X3D and RX 9070 XT, enabling
`vector-float-runs` for the profiled regions changed 71 of 1,111 sources.
Two runs per configuration averaged 16.19 FPS with the control and 16.13
FPS with the option, measured over the same 200 forest presents. Each
run played the full opening movie.

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

## Measurements behind the initial default set (2026-11)

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

## Original rollout plan

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
