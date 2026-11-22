# Lifter redesign: structured functions (`regalloc`)

## Status

**Done** (opt-in lift option `regalloc`, with `functions`; the default
output of every existing lift is byte-identical to the base branch's,
checked by diffing the emitted modules; the golden hashes move only
because the new regalloc lifts are part of the hashed programs, and the
base branch's checked-in constants were already stale):

1. Each lifted function whose blocks form a reducible graph is one host
   function with real control flow (loops, ifs, labelled blocks from
   `structure::function_structure`), not a `match next_pc` loop.
2. Step accounting per function: a local count, flushed before calls and
   at the end; the stop word tested only at entries, loop heads and call
   return points, with a look-ahead, leaving to the windows (which then
   take every budget and host boundary on the same block as window code)
   where the look-ahead could pass the stop word.
3. Callees of at most 24 instructions, with what they call, run inlined;
   calls within a region are direct.
4. Translation entries are function entries too, functions can be entered
   at their outermost loop heads, and the window arm of every such block
   runs the structured function first.
5. SP alignment checked at entries and call returns (leaving to the
   windows if it is off) when every SP write keeps it.
6. Every exit stores only the locals possibly written on some path to it,
   so registers written before any exit are not loaded.
7. NZCV as `cpu::LazyFlags`: the last flag setter's operands, the flags
   computed whole only where stored; conditions are plain compares.

**Numbers** (cachegrind, game-shaped build: region code at opt-level 1,
one codegen unit, no LTO, `x86-64-v3`; host instructions per guest
instruction, mispredictions per 100 guest instructions in brackets; base
is the game set plus `window-calls`, regalloc the game set plus
`functions,function-table,call-summaries,transitive-summaries,regalloc`):

| kernel | base | structured (1-2) | + inlining (3) | + entries (4) | + SP, exits (5, 6) | + lazy flags, nested inlining (7) |
|---|---|---|---|---|---|---|
| gamefn | 12.31 (12.8) | 9.05 | 6.54 | 6.43 | 5.43 | 4.67 (0.1) |
| gamecalls | 14.73 (6.1) | 12.27 | 8.30 | 5.92 | 4.95 | 4.55 (0) |
| calls2 | 23.46 (46.2) | 17.61 | 11.08 | 9.62 | 7.39 | 1.62 (0) |
| crossing | 10.95 (10.2) | 3.21 | 3.21 | 3.51 | 3.28 | 3.21 (0) |
| calls | 19.11 (33.3) | | | | | 1.34 (0) |
| vcalls | 13.87 (3.3) | | | | | 11.27 (10.0) |
| arith | 5.28 | 5.28 | | 3.29 | 3.29 | 3.14 |
| ldst | 5.20 | 5.20 | | 5.09 | | 5.09 |
| memset | 8.33 | 8.33 | | 8.41 | | 8.41 |
| switch | 7.18 (27.3) | | | | | 2.73 (9.1) |

(`calls` was measured with a substring filter before, which also ran
calls2 and gamecalls; the benchmark now takes `=NAME` for one kernel, and
the row shows the kernel alone. `vcalls`, new: an update loop calling
one of two small functions per object through a table of function
pointers; BLR in a structured function goes through
`Core::call_indirect`, below.)

Kernel machine code (`kernel_code::*` symbols): base 86,363 bytes,
functions 103,776, regalloc 162,985 (windows 95,732 of it, structured
functions 66,320): the windows stay (resumption at any block), and the
structured functions are further copies. Expect region crates to compile
correspondingly longer, and more `.text`, most of it cold window code.

Every split test program compiled as a game region
(`cargo bench -p echora-aot-tests --features codesize --bench codesize -- game`,
90,317 guest instructions, opt-level 1):

| mode | Rust bytes/ins | rustc s | `.text` bytes/ins |
|---|---|---|---|
| game | 334.7 | 194 | 210.6 |
| game_window_calls | 339.2 | 200 | 215.2 |
| game_functions | 343.4 | 201 | 217.0 |
| game_regalloc | 720.5 | 365 | 351.7 |

The structured functions' Rust is about as large as the windows' (30 MB
against 34 MB here: every program entry is a function), and compiles to
about two thirds of the windows' machine code. A window whose every block
a structured function holds runs only next to budgets and host
boundaries, so it is `#[cold]` (out of the hot code's way, in
`.text.unlikely`); the windows cannot go, since the per-block charges
that place every boundary exactly live there. For the game, enabling
regalloc on the hottest regions only (`ECHORA_LIFT_OPTIONS_REGIONS`)
keeps the growth to those.

Review notes (read-only faithfulness review, no correctness findings):
a trap's refund (`tail`) takes the block of the function holding the
trapping PC, as the block tag would name it (blocks do not overlap); a
preemption another thread requests is seen up to the look-ahead later;
a `?` in a string no longer turns precise exit stores off.

**Next**: fewer locals live across non-inlined calls (stores and reloads
only where summaries prove them needed), the call boundary itself
(results through memory, the callee's prologue), code size (inlining and
copies only where hot), and the remaining in-function costs (the guest
memory range test per access).

## Where the cost was

Measured before this work (`docs/codegen-speed.md`, "What a guest
instruction costs inside a block"): 12-15 host instructions per guest
instruction on game-shaped code; per block a charge of seven host
instructions (two dependent loads for the stop word, the steps loaded,
added, compared and stored); a dispatch back through the window's
`match` after every block (1-1.5 host instructions per guest instruction
and most mispredictions); about a fifth of the instructions stack-slot
traffic for register locals; every call storing and reloading the
caller's locals and the callee loading and storing its own.

## The lowering

### Functions with real control flow

`codegen::functions` recovers function bodies (BL targets, and with
`regalloc` also the translation's entries). `regalloc.rs` builds each
body's graph (jump tables: their cases, then any other target as an
exit), and `structure::function_structure` gives Ramsey's structured tree
for it from the entry (and the outermost loop heads, below), or None for
an irreducible graph: then the function keeps the existing window-shaped
function form. Constant successors are plain jumps; a conditional branch
branches on its own condition; leaving the body (a return, a branch out,
BR to an unknown target) stores the PC and returns.

Functions holding spin-wait heads, traced blocks, exclusive accesses or
anything reading the step count (counter reads, spin hints) keep the old
form: their per-block boundaries matter.

### Steps

Each block adds its instruction count to the function's local `n`
(constants the host compiler folds along a path). The count is exact
wherever it can be observed:

- before every call that is not inlined, `n` goes into `Core::steps`
  (the callee's own charges and tests see exact steps);
- when the function returns, `n` goes into the steps;
- on a trap from one of the function's own instructions, the steps get
  `n` less the instructions after the trapping one in its block (`tail`,
  as `Core::settle_block_trap` would), and the block tag is cleared
  (`Core::clear_block_tag`, new) so the host refunds nothing more.

Budgets and host boundaries (slice ends, timed waits, attention,
preemption) are taken at block starts by the windows' per-block charges.
A structured function tests the stop word only at its entries, its loop
heads and its call return points: `steps + n + L > stop`, with L the
most instructions any path from there runs before the next test, a call
or leaving (every cycle passes a loop head in a reducible graph, so L is
finite). If the test fails the function leaves at that block (each such
block keeps its window arm) and the windows run on from it with exact
per-block charges; so every budget trap and host boundary falls on the
block it falls on in window code, with the same steps and state. If the
test passes, no block before the next test could have failed its own
charge. Only a preemption that another thread requests (lowering the stop
word while the function runs) is seen at the next test instead of the
next block, as with any timing between threads.

### Inlined callees

A BL to a lifted function that runs at most `INLINE_LIMIT` instructions
with everything it calls inlined (no BLR, no recursion, plain as above)
runs inlined: the callee's blocks are copied
into the caller's graph under their own keys, the BL sets X30 and goes
to the copy, and the callee's return goes on in the caller when the
target is the return address (any other target leaves, as the RET would).
Counts, traps and tests are those of the copied blocks.

### Indirect calls

A BLR in a structured function calls `Core::call_indirect` with a static
two-entry cache for its site: each entry the target PC and the index of
its lifted function (or none) in one word, so threads sharing the site
never pair a PC with another target's function. A hit runs the function
as a direct call would (`Core::call_window`); a miss searches the
entries the root installed (`install_function_entries`, with regalloc
and the function table) and replaces the older entry; a target with no
function continues through the window table as before. The caller stores
and reloads every local it holds around it (the callee is unknown).

### Entering at loop heads and from windows

After a host boundary execution resumes at a block start through the
windows. A structured function is therefore also entered at the heads of
its outermost loops (a dispatch on the PC at the top; inner heads would
make the outer loop irreducible), and the window arm of each entry and
such head first runs the function (`run_entry`, the window's locals
stored and reloaded around it as for a call). If the function comes back
at that same block (its first test failed), the arm runs the block itself
with its per-block charge, so this never loops.

### Locals

Guest registers stay in the function's locals (`RegisterLocals`); inside
the structured body the host compiler sees real control flow, so
registers live only where they are used. Calls keep the existing
summaries (stores of what the callee may touch, reloads of what it may
write). When no `?` is left in the body, every exit is a `return` the
emitter can see: each gets the stores of the locals possibly written on
some path to it (a fixpoint over the blocks, by line within a block),
and stores before calls and helpers skip locals nothing wrote (they still
equal the `Core`); the entry loads of registers written before any exit
are then dead and the host compiler drops them.

### SP alignment

A stack access traps when SP is not 16-byte aligned. Where every SP
write of a function and its inlined callees adds or subtracts a multiple
of 16 (pre- and post-indexed accesses, ADD/SUB of an immediate), SP keeps
its alignment from the entry, so the per-access checks go and the
entries and call returns test SP once, leaving to the windows (which
trap at the exact instruction) when it is not aligned.

### Flags

The NZCV local of a structured function is a `cpu::LazyFlags`: the
operands of the last flag-setting ADD/SUB or logical operation, NZCV
computed only where stored (calls, exits); a condition reads the operands
(a subtraction's GE is a signed compare), which the host compiler folds
into the branch. ADCS/SBCS and MSR keep the flags themselves. A unit test
in `echora-guest` checks every operation and condition against
`cpu::Flags` on edge and random operands.

## Tests

`tests/window_calls.rs` lifts its game-shaped programs (random bodies,
frames, jump tables, calls, PLT stubs, functions over the block limit,
FP and integer twins) also with `regalloc` in one 64 KiB region, in
1 KiB regions and without the function table, and compares every run
against the function-free lift: whole runs, sliced runs (a host boundary
after every few instructions) and budgeted runs, every general and vector
register, SP, NZCV, FPCR, FPSR, PC, steps and boundary count, trap, data
and stack; the integer twin's regalloc lifts also run against the
reference interpreter. `tests/calls.rs` runs its callers with structured
functions against the reference interpreter, and `tests/early_exits.rs`
runs its early-exit, two-level and SVC-wrapper programs with structured
functions too. The whole `echora-aot-tests` suite passes with the game
set plus `functions,function-table,call-summaries,transitive-summaries,
regalloc` (golden aside, as above).

## Measuring

As in `docs/codegen-speed.md`: the kernels built the game's way and
counted under cachegrind at `--scale 1` and `--scale 2`, differenced:

```bash
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo bench --no-run -p echora-aot-tests --bench kernels --features inline-vector,fused-flags,whole-vector-checks,fp-status-locals,window-memory,untagged-blocks,split-ticks,float-runs,functions,function-table,call-summaries,transitive-summaries,regalloc,small-address-space --config 'profile.bench.lto=false' --config 'profile.bench.package.echora-aot-tests.opt-level=1' --config 'profile.bench.codegen-units=1'
```

For the game (`ECHORA_LIFT_OPTIONS` replaces the defaults, so they are
listed too):

```bash
ECHORA_LIFT_OPTIONS=inline-vector,fused-flags,whole-vector-checks,fp-status-locals,window-memory,untagged-blocks,split-ticks,float-runs,functions,function-table,call-summaries,transitive-summaries,regalloc
```
