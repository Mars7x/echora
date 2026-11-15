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
without a leaf's writes, or without its reads, fails it.

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
