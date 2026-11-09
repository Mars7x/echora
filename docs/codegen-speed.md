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
(f87afd9) cost nothing: LLVM merges their aligned and unaligned paths into
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

## Order for the rebuild

1. Land the `echora-guest` API and runtime changes with today's emission
   (the generated code still calls `tick_block_at` / `finish_block`, now
   cheaper).
2. Turn on, in `ECHORA_LIFT_OPTIONS`: `inline-vector`, `fused-flags`,
   `whole-vector-checks`, `window-memory`, and the block bookkeeping
   emission.
3. One full rebuild of the game's translation; compare frames and speed
   with the previous build.
