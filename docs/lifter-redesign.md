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

With `regalloc-loop-inlining` as well (callees of up to 96 instructions,
with what they call, inlined where they are called inside a loop):
gamefn 2.57, gamecalls 2.12, crossing 2.30, the rest unchanged; kernel
machine code 170,134 against 170,759 bytes (an inlined body replaces
the call's own code). Its size on the game depends on how many loop call
sites there are; opt-in so that the forest decides.

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
| game_regalloc (later run, more lifts) | 732.2 | 391 | 357.2 |
| game_regalloc_loops (same run) | 732.4 | 414 | 357.3 |

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

**Opt-in on top** (both measured below): `regalloc-loop-inlining`
(gamefn 2.57, gamecalls 2.12, crossing 2.30; size flat on the kernels)
and `regalloc-resume-loops` (ldst 4.80; kernel code +28%). Dropping the
windows of covered blocks for `.text` waits on the step-accounting
decision ("Which windows stay").

**Build cost** (below, "Compile time and memory"): `regalloc-crates`
emits each region as crates of at most 10 MB of Rust
(`REGALLOC_CRATE_BYTES`, or `ECHORA_REGALLOC_CRATE_BYTES`): its windows,
then its lifted functions, packed in order by emitted size, so the game's
largest region (83 MB of Rust, past 25 GB of rustc) becomes about nine
crates that cargo builds in parallel; a structured function over 1.5 MB
of Rust with its inlined callees is emitted without inlining; functions with compare-and-swap
are now structured too (they fell back to the window-shaped form, which
compiles slower).

**First game result** (coordinator, 20bc158 on 87 of the 89 hottest
regions): forest 24.4/24.4/23.8 fps against 24.1: no gain. In those
regions 24.5% of guest cycles still ran in windows and 16.6% in
structured functions (44.4% in windows before), total instructions -3%;
lifted functions held 39% of the translation's instructions. So coverage
first, then the gain where covered (about 1.2x). Since then:

- Exclusive accesses no longer keep a function in the window form (game
  atomics are load/store exclusive pairs).
- Lifted functions under regalloc may hold 1,024 blocks
  (`REGALLOC_BLOCK_LIMIT`, 256 before); the size caps bound what they
  emit (instructions cap raised to 4,000).
- Every `?` in a structured body is written out as a `match` returning
  the error, so exact exit stores (and the stores before calls filtered
  to what was written) apply to every structured function, not only to
  those without a fallible helper (FP fallbacks, CAS, `.at(m, pc)?`); in
  the window-call test programs all 119 structured functions qualify now.

Coverage table from the game (coordinator, guest cycles still in
windows inside the regalloc regions) and what answers each:

| share | cause | answer |
|---|---|---|
| 9.91% | function dropped for the 256-block limit | `REGALLOC_BLOCK_LIMIT` 1,024 under regalloc (6db3c3a), the size caps the bound |
| 5.90% | reached only from another region (bodies stopped at 64 KiB edges) | `regalloc-span-regions` (opt-in, with `regalloc-crates`): bodies run on past their region's edge |
| 4.94% + 1.84% | inside a structured function but resumed mid-function or handed back | next: resume entries at call returns and window-entered blocks |
| 1.52% | reached only by indirect branches | next: indirect targets as entries |

A capped function's stub cannot run the function through the window
table to its return instead of handing back (see `capped_stub`).

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
spin hints keep the old form: their per-block boundaries matter. A
counter read (MRS CNTPCT, which follows the steps with
`ECHORA_VIRTUAL_CLOCK`) has the count flushed into the steps right
before it, so it reads what it reads in window code (a charged block
counts whole); `tests/counter_reads.rs` compares registers, steps and
every host boundary with the window form, whole and in every slice up
to 23 steps.

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

With `EmitOptions::regalloc_loop_inlining` (`regalloc-loop-inlining`),
a call site on a cycle of the caller's blocks inlines callees of up to
`LOOP_INLINE_LIMIT` (96) instructions with what they call; every call
inside an inlined callee runs inlined too. Tests: `regalloc_loops` in
`tests/window_calls.rs` and `call_regalloc_loops_code` in
`tests/calls.rs`.

With `EmitOptions::regalloc_resume_loops` (`regalloc-resume-loops`),
each inner loop of at most `RESUME_LIMIT` (64) instructions up to the
function's next entry gets a resume function, `run_function_<head>`, which
its head's window arm runs: after a host boundary at an inner loop's head
execution is structured again at once, not at the loop's end. ldst (a
loop around a word loop, a boundary every 100,000 instructions): 5.09 to
4.80, the window share gone (20% of the run now in the resume function);
but kernel machine code 170,759 to 218,885 bytes, most of it the vector
kernels' inner loops, so opt-in.

### Which windows stay

The windows cannot shrink to the blocks execution resumes at from
outside (SVC and trap returns, indirect and table targets, entries) while
host boundaries and budgets stay exact: after a structured function
leaves at a test that the look-ahead failed, the next blocks run with
per-block charges until the boundary falls where window code takes it,
and that can be any block of the next `L` instructions, so every block
needs code that charges it alone. Two ways to drop the windows of the
blocks structured functions hold:

- Take the boundary at the test itself: a test that fails because the
  boundary is near returns `Trap::HostBoundary` (or the budget's trap)
  there, before the block, at most `L` instructions earlier than window
  code would. The guest sees nothing (no instruction runs differently);
  the host's slices end a few instructions sooner, and a budgeted run
  stops before its exact limit. Every test point must then be enterable
  by structured code: entries and outermost loop heads are, call return
  points and inner loop heads would need resume functions (above). The
  windows keep only the arms of blocks no structured function holds and
  the delegating arms.
- Count steps differently altogether (back-edge polls instead of exact
  counts), as the other lifter session is studying: the same
  consequence for windows.

Either is a change of what the runtime may assume about where boundaries
fall (tests compare boundary PCs and steps with window code), so it waits
for that design note; with it, the windows of covered blocks go and
`.text` falls back to about the functions' own size.

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

## Compile time and memory

Measured with rustc as for a game region (opt-level 1, one codegen unit,
`x86-64-v3`), peak RSS of the rustc process, on one game-shaped function
at the block limit from the codesize build (`big_function`: 240 blocks,
1,001 instructions, loads, stores, CAS, forward skips, one-block loops,
calls of two leaves and of a 40-instruction function, BLRs):

| lift of that function's region | rustc s | peak RSS | `.text` |
|---|---|---|---|
| game set | 2.5 | 187 MB | 182,570 |
| regalloc, one crate | 5.1 | 284 MB | 361,769 |
| regalloc-crates: the windows' crate | 3.0 | 198 MB | 216,369 |
| regalloc-crates: the functions' crate | 2.6 | 220 MB | 147,783 |

Smaller versions of the same function (60, 120 blocks) cost the game set
0.5 and 1.1 s and regalloc 0.8 and 1.9 s: about linear. LLVM's passes are
nearly all of it (`-Z time-llvm-passes`: greedy register allocation, the
register coalescer, InstCombine and instruction selection lead); the
structured function is one large host function holding every guest
register as a value, with a summarized call's stores and reloads at each
call (28 and 17 here).

Before CAS was handled, every function holding one (its emitted code may
call `m.spin_hint()`) fell back to the window-shaped function form, which
compiled slower than the structured one (4.8 s against 3.3 s for this
function without its calls): game code with atomics took that path. A
spin hint now has the count flushed into the steps first, and the stop
word it may lower is seen at the function's next test, as a preemption.

With `regalloc-crates` a region is emitted as crates packed by size: the
region's own id (`gen-r-<region>`) for the first, `gen-r-<region | k <<
40>` (k from 1) for the others, each at most `REGALLOC_CRATE_BYTES` (10
MB; `EmitOptions::regalloc_crate_bytes`, from `ECHORA_REGALLOC_CRATE_BYTES`
in `echora lift`) of Rust but for a single larger window or function: the
region's windows in order, then its lifted functions in order. Nothing in
one crate names a window or function of another: the root's window table
and function table name each one's crate; windows enter structured
functions and every function calls the others through the function table
(`function_at`, `run_entry_at`); an inlined PLT stub's predicted call goes
through the window table instead of naming its window; and the root
installs the function table into every crate. Resume functions are not
emitted then. The sink gets the extra crates as regions, so the game's
workspace and bundles take them as they are. Independently, a structured
function whose Rust with its inlined callees passes
`regalloc::FUNCTION_BYTES` (1.5 MB) is emitted again without inlining.
Builds on `5db8463` failed on the game here: a structured function's
call through an inlined PLT stub named the predicted target's window
(`m.call_window(.., run_window_<w>)`), which another crate held; from
`5fba752` that call goes through the window table when the crates are
split, and so does a direct window call (`direct-calls`).
`tests/plt.rs` runs every PLT case also with structured functions in one
crate per region and in crates of one window or function each (with
direct calls), and asserts that the former names windows from functions
and the latter does not; without the fix it fails to compile as the game
did. Packing cannot split one function: 7101b9 held a structured function
(`run_function_7101b9935c`) of 7.75 MB of Rust, past 22 GB of rustc.
Structured functions are therefore capped
(`regalloc::FUNCTION_INSTRUCTIONS`, 2,000 guest instructions;
`regalloc::FUNCTION_BYTES`, 500 KB of Rust, or
`EmitOptions::regalloc_function_bytes`): over either with its inlined
callees, it is emitted again without inlining; its own instructions over
the first, or its Rust without inlining over the second, and it is left to
the windows: its entry's host function is a stub that returns at once
(its PC still the entry), so a caller goes on through the windows as
after any early exit, and no window arm delegates to it. `echora lift`
prints the functions the caps left to the windows
(`codegen::take_capped_functions`). The 1,001-instruction function above
is 236 KB of structured Rust (about 3 s and 300 MB of rustc for its
region), so 500 KB keeps such a function well under the superlinear
range. `tests/window_calls.rs`'s `regalloc_capped` lift caps at 4 KB
(52 functions left to the windows) and runs against the function-free
lift like the others.

`tests/window_calls.rs`'s `regalloc_crates` lift packs into crates of
200 KB (ten crates: five of windows, five of functions) and runs against
the function-free lift like the others.

At the game's rate (7101b9: 83 MB of Rust, past 25 GB of rustc RSS, about
0.3 GB per MB) a 10 MB crate should stay near 3 GB; lower
`ECHORA_REGALLOC_CRATE_BYTES` if not.

The game's regions of 10-20 minutes and up to 12.8 GB are not reproduced
by these programs (the whole set of test programs, about six regions'
worth in one crate, takes 338 s and 6.2 GB with the game set). To find
what one of those regions holds, the size of its largest functions:

```bash
python3 -c 'import re,sys; t=open(sys.argv[1]).read(); ends=[m.start() for m in re.finditer(r"\npub fn ", t)]+[len(t)]; fns=[(e-m.start(), m.group(1)) for m in re.finditer(r"\npub fn (\w+)\(", t) for e in [min(x for x in ends if x > m.start())]]; print(*sorted(fns, reverse=True)[:15], sep=chr(10))' regions/7101c7/src/lib.rs
```

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
ECHORA_LIFT_OPTIONS=inline-vector,fused-flags,whole-vector-checks,fp-status-locals,window-memory,untagged-blocks,split-ticks,float-runs,functions,function-table,call-summaries,transitive-summaries,regalloc,regalloc-crates
```

(`regalloc-loop-inlining` and `regalloc-resume-loops` are further
opt-ins; resume functions are off with `regalloc-crates`.)
