# Opt-in lift options: readiness for the next relift

What every opt-in lift option does, how it is tested, what it measured,
what it costs in generated code and compile time, and whether it should
go into the next relift of the game. Measured 2026-11-22 on main (466d858)
on the aot-tests kernels and programs; the game was not run.

## Recommendation

Relift with `align-windows` and these options on:

- **Everywhere** (cheap: at most 0.1% more Rust on the test programs):
  `direct-q-pairs`, `vector-float-runs`.
- **On the hot regions only** (`ECHORA_LIFT_OPTIONS_REGIONS`, the ~100
  regions holding ~88% of lifted-code samples):
  - `inline-vector-unary`, `inline-vector-int`, `inline-vector-pairwise`,
    `inline-fp-scalar`, `inline-vector-misc` and `steps-local`. Each costs
    up to 12% more Rust and 14% more rustc time per region, and together
    32% more rustc time and 29% more memory. They gain only where their
    forms run.
  - `window-calls`, `long-windows` and `known-returns`, the call path:
    calls +71%, calls2 +124%, gamecalls +38%, crossing +85% with the set.
    Run the forest once with `window-calls-audit` added (it checks every
    summary as the game runs), then once without, before trusting them.
    Leave `window-call-early-exits` off: early exits are what the game's
    hang with call summaries had in common, and it gained nothing
    measurable on the kernels over `window-calls` alone.

The resulting `ECHORA_LIFT_OPTIONS`, with `ECHORA_LIFT_OPTIONS_REGIONS`
naming the hot regions (the listed regions get the defaults plus these):

```
ECHORA_LIFT_OPTIONS=align-windows,direct-q-pairs,vector-float-runs,inline-vector-unary,inline-vector-int,inline-vector-pairwise,inline-fp-scalar,inline-vector-misc,steps-local,window-calls,long-windows,known-returns
```

`align-windows` is a build flag and applies to every crate of the build
whether or not it is listed. With no region list, `ECHORA_LIFT_OPTIONS`
replaces the defaults everywhere, so the defaults must be named too.

`direct-q-pairs` and `vector-float-runs` also belong outside the hot
regions. That needs one lift option list for every region and another for
the listed ones, which `echora lift` does not have. Until it does, put
them in the hot regions' list with the rest; they gain nothing in cold
code anyway.

Changed since the draft on the old base:
- **`float-run-loads` moves to keep off.** It costs the floats kernel
  -15.4% (-25.8% to -5.4%) alone and -14% to -16% in every set holding
  it, against floatld +31%. On the old base the loss was -7.8%. The
  floats kernel's block starts with its three loads, which the run then
  holds; floatld's loads come between its operations. Why the longer run
  is slower there is not yet known. Taking only loads between a run's
  operations, not those that lead it, may keep the gain without the
  loss; that is a lifter change to measure later.
- **The lifted-function family moves to keep off.** The window-call
  options take the same calls, without new copies of guest code and
  without a hang in their history, and measure the same: with the set,
  window calls give calls 553 Minstr/s and calls2 399 against the
  functions' 471 and 417, gamecalls 648 against 597, crossing 1870
  against 2112, every pair within the other's interval.
- **The vector kernel's loss with the set is gone**: +11.7% (+1.4% to
  +23.6%) with the recommended set, against -3.1% before.

Keep off: `float-run-loads`, the function family (`functions`,
`function-table`, `call-summaries`, `transitive-summaries`),
`deferred-charges`, `direct-calls`, `dense-dispatch`, `structured`,
`stub-windows`, `count-fallbacks` and `window-call-early-exits`
(reasons in the table). `dense-dispatch` measured well again (memset
+66%, ldst +20%) but has no dedicated test and has not been measured with
the set; it waits for the relift after this one.

The game's own numbers suggest that the per-instruction options move the
frame by little. Side's census of the hot windows finds the vector float
operations mostly apart, not in runs (`vector-float-runs` changed the hot
regions by 0.5% of lines). The windows spend their operations on register
spills and reloads around calls and window entries, which is what the
window-call options address.

## A bug found on the way: results at the smallest normal

Running the float-run programs with inline options in alternate regions
(below) found a bug in `inline-vector` itself, which is in the game's
default set. Arm detects underflow before rounding: a product, quotient,
fused result or narrowed value whose exact magnitude is just below 2^-126
(2^-1022) and rounds up to it raises UFC, and under FZ it is a zero. The
inline paths took any result whose exponent field is nonzero, so such a
result was committed as the smallest normal, with no UFC and no flush
under FZ. The echora-simd helpers had the same fault at that boundary.

Main's a3e6260 ("Take the exact path for products that round up to the
smallest normal") fixed it, in the inline paths and in the helpers, with
`tests/rounded_up.rs` and its own golden change; my own fix was dropped in
the rebase. `tests/smallest_normal.rs`, added here, checks the same
boundary from more sides:
- the multiplies, divides, fused forms, FMULX and the narrowing FCVT,
  alone and in a vector and a scalar run, with every inline and run
  option on, against the helpers alone;
- operands at the boundary and operands far above it whose product
  rounds to it;
- FZ on and off, DN, and FPSR with and without IXC and UFC;
- the vector fused forms against a model of Arm's rounding
  (`arm_fused`), independently of the helpers.

It passes on main and fails with a3e6260 reverted.

## The options

Kernel effects use the task-18 recipe. Every function is 64-byte aligned.
The 26 builds ran alternately, each kernel seven times per run, pinned to
one CPU, for 12 rounds. Each figure is the median of the runs' medians
against the game set's, with a 95% interval by resampling.

Only changes whose interval excludes zero are listed; every other kernel
of the 20 is within noise. This run was noisier than the old base's:
- intervals of ±10% to ±15% are common for the kernels a build does not
  change;
- `float-run-loads` shows vector -8.2% (-14.3% to -0.5%), though it
  changes nothing in that kernel (Q loads, no scalar run);
- `count-fallbacks` shows vector -13.5%.

So changes under about 10% are within noise here, even where their
interval excludes zero.

Baselines, with the game set alone, in Minstr/s (about half the old
base's on kernels no change touched, such as arith 3919 to 1923, so only
ratios within one run compare):

| arith | memcpy | memset | calls | switch | vector | poll | scalar | ldst | floats |
|---|---|---|---|---|---|---|---|---|---|
| 1923 | 647 | 807 | 323 | 1016 | 195 | 217 | 402 | 1740 | 490 |

| floatld | calls2 | vnorm | convert | intvec | fpairs | fpscalar | misc | gamecalls | crossing |
|---|---|---|---|---|---|---|---|---|---|
| 354 | 178 | 78 | 39 | 30 | 145 | 159 | 48 | 471 | 1010 |

Sizes are of the generated Rust against the game set, measured two ways:
- for the 20 kernels (`kernel_code`, one 64 KiB region);
- for every split test program as one crate (86,559 guest instructions;
  `cargo bench -p echora-aot-tests --features codesize --bench codesize --
  game`), with its rustc time and peak memory at `opt-level` 1 and 16
  codegen units, as a game region builds.

Each crate compiled once; differences in rustc time under about 5% are
noise. The test programs lift in 1 KiB regions, where `long-windows` has
nothing to copy, so its kernel figure is the one that counts.

"Differential" means the suite compares the option's lift with the same
programs lifted without it (or against a reference interpreter or an Arm
model); "edges" means float or memory edge inputs; "FPCR/FPSR" means FZ,
DN, sticky flags and a rounding mode the helpers trap on are varied.

| option | what it changes | tests | kernels (12 rounds) | Rust: kernels; programs (rustc s, MiB) | left on the old path | recommendation |
|---|---|---|---|---|---|---|
| `inline-vector-unary` | Vector FSQRT, FCVTZS/U, FRECPE, FRSQRTE, FRINT* natively per lane where exact | `vector_unary` (differential, edges, every FRECPE/FRSQRTE fraction, FPCR/FPSR); `smallest_normal`; `vector_runs` and `fallback_counts` in part | vnorm +122.5% (+93.1% to +146.4%), convert +332.7% (+304.9% to +386.0%) | +3.4%; +1.2% (−0%, +1%) | Lanes not normal or zero, out-of-range conversions, estimates with denormal results; conversions with fraction bits or directed rounding (`inline-vector-misc`'s); half precision; any FPCR the helpers trap on | Hot regions |
| `inline-vector-int` | Vector integer operations and element moves written out | `vector_int` (4,612 forms; differential, edges, shift counts, QC, FPCR/FPSR) | intvec +589.2% (+551.7% to +712.5%) | +1.9% (−0.4% lines); +4.7% (+11%, +3%) | SDOT/UDOT, PMULL, the crypto instructions | Hot regions |
| `inline-vector-pairwise` | FADDP, FMAXP, FMINP, FMAXNMP, FMINNMP per lane | `vector_int` (as above) | fpairs +39.2% (+26.1% to +60.4%) | +6.1%; +0.7% (+3%, +0%) | Pairs not normal or zero, sums not normal or zero | Hot regions |
| `inline-fp-scalar` | Scalar FMOV to/from general registers, SCVTF/UCVTF, FCVT*S/U, FCVT S↔D, FSQRT, FRINT*, FCCMP | `fp_scalar` (differential, edges, NZCV, FPCR/FPSR); `float_runs` (alternate regions); `smallest_normal` | fpscalar +126.9% (+116.1% to +136.9%) | +4.5%; +5.5% (+11%, +5%) | Half precision; conversions that do not fit or flush; FPCR the helpers trap on | Hot regions |
| `inline-vector-misc` | TBL/TBX, structure loads and stores, FABD, FMULX, FMAXV and kin, conversions with fraction bits or directed rounding | `vector_misc` (differential, edges, table indices, memory data, FPCR/FPSR); `smallest_normal` | misc +191.6% (+175.0% to +206.1%) | +2.7%; +8.9% (+14%, +8%) | Lanes not normal or zero (FABD, FMULX), FMAXV unless all lanes are; FPCR the helpers trap on | Hot regions |
| `direct-q-pairs` | LDP/STP of Q registers: the helper's body written out | `q_pairs` (differential, every addressing form, faults in child processes) | memcpy +9.0% (−1.9% to +19.5%), within noise this time (+15.6% on the old base) | +0.5%; +0.1% (−2%, −0%) | Nothing | Everywhere |
| `vector-float-runs` | Two or more consecutive vector float operations checked once on a fast copy | `vector_runs` (197 forms; differential, edges, pairwise edges, FPCR/FPSR, MSR FPCR inside; every per-instruction option in alternate regions); `smallest_normal` | vnorm +190.8% (+143.2% to +193.9%), convert +549.6% (+526.5% to +605.9%), fpairs +81.9% (+71.9% to +116.2%), misc +160.3% (+150.8% to +202.0%) | +17.1%; +0.1% (−1%, +0%) | Single operations, a block's first and last instruction (see task 26's report) | Everywhere |
| `float-run-loads` | Scalar float runs may hold LDR/LDUR/LDP of S | `float_runs` (differential, edges, FPCR/FPSR) | floatld +31.1% (+6.2% to +59.4%), **floats −15.4% (−25.8% to −5.4%)**; −14% to −16% in every set with it | +0.8%; +0.0% (+0%, +0%) | D and Q loads; an SP base not 16-byte aligned | Keep off: the floats loss, twice measured |
| `steps-local` | The step count in a window local; untagged charges without the `Core` | `steps_local`, `steps_local_calls` (differential: registers, steps, every host boundary, every budget) | memset +101.4% (+85.6% to +132.2%), ldst +43.7% (+35.2% to +52.0%), switch +32.5% (+15.5% to +47.3%) | +8.1%; +12.0% (+9%, +10%) | Tagged charges (blocks that can trap after the charge: most in the test programs) still go through the `Core`; not with `deferred-charges` | Hot regions |
| `deferred-charges` | Blocks that cannot trap add their count to a pending sum | `deferred_charges` (differential, a host boundary at every step) | within noise everywhere (calls +10.1%, vector +7.8%) | +1.2% | Blocks that may trap, call or be resumed at | Keep off: `steps-local` takes the same charges, wins where both are on (below) and gains more |
| `functions` (+ `function-table`, `call-summaries`, `transitive-summaries`) | Lifted functions as host functions, called directly or through a table; summarized calls store and reload only what the callee may touch | `calls` (vs the reference interpreter, also with functions in alternate regions), `early_exits` (callees leaving early, two levels, SVC wrappers), `spills` | functions: calls +11.1%, calls2 +81.1%, crossing +88.9%; table: calls +9.0%, calls2 +78.6%, crossing +79.1%; summaries: calls +40.7%, calls2 +105.6%, gamecalls +37.1%, crossing +84.2%; transitive: calls +44.1% (+31.7% to +56.9%), calls2 +139.6% (+116.3% to +175.5%), gamecalls +44.6%, crossing +91.4% | +13.5% to +15.0% (+22% to +23% lines); all four +2.9% (+3%, +1%) | Bodies over 256 blocks; BLR; summaries only for functions leaving by their own return | Keep off: the window-call options take the same calls as well, with no copies of guest code and no hang in their history |
| `window-calls` | A BL to a target with a summary calls its window directly, storing and reloading only what the callee may touch | `window_calls` (a game-shaped random program in pairs without and with, 63 callers from 12 states, whole, sliced and budgeted, every register, memory and stack; an integer twin against the reference interpreter; with every other call and charge option), `calls`, `kernel_calls` (vs the reference interpreter); `defuse` effects checked by the semantic suites and statically | calls +64.6% (+52.3% to +83.7%), calls2 +134.7% (+114.9% to +170.8%), gamecalls +44.7% (+28.8% to +59.4%) | +0.9% (+3.2% lines); +1.5% (−0%, +1%), `.text` +2.9% (a second copy of each entered window) | BLR; targets whose window can be left other than by a return (`window-call-early-exits`'s); calls nested over 16 deep; ignored with `functions` or `structured` | Hot regions, after one forest run with `window-calls-audit` and one without |
| `window-call-early-exits` | With `window-calls`: also targets whose window can be left early | `window_calls` (with early exits) | calls +72.8%, calls2 +133.2%, gamecalls +49.8%: the same as `window-calls` within noise | +0.7% | — | Keep off: no gain here, and early exits were common to the game's hang |
| `window-calls-audit` | With `window-calls`: every summarized call checks its summary as it runs and traps naming the call and register | `window_calls` (audited; a summary missing a write is named) | not measured: it stores and reloads everything at every call | — | — | One forest run before trusting `window-calls` |
| `long-windows` | Each window also runs copies of the blocks it reaches in neighbouring windows of its region (up to 256 instructions) | `window_calls`, `kernel_calls` (crossing, whole and sliced, vs the reference interpreter), the lifter's unit tests | crossing +83.5% (+58.7% to +111.8%) | +3.1% (+5.1% lines); +0.0% (1 KiB regions; at most one window's worth more per window in the game's) | Indirect branches, BL targets, PLT stubs, other regions; ignored with `functions`, `structured` or `deferred-charges` | Hot regions |
| `known-returns` | After a call the return address is a constant successor; any other PC leaves the window | `window_calls`, `kernel_calls`, unit tests | calls +9.6% (+2.7% to +32.1%), at the edge of noise | +0.4%; −2.5% (−6%, −2%), **`.text` +12.4%** | — | Hot regions, with the window calls it serves |
| `direct-calls` | A BL to a block of the caller's region calls its window directly | none of its own; the whole suite with the feature | calls2 +39.8% (+25.8% to +46.9%) | +0.2% | — | Keep off: `functions` and `window-calls` take its calls first (below) and are tested |
| `dense-dispatch` | Dispatch on the block's index in the window, not its PC | none of its own; the whole suite with the feature | memset +65.5% (+53.3% to +97.1%), ldst +20.3% (+1.4% to +42.7%) | +0.7% | — | Keep off for this relift: a measured loss before (unaligned), never measured with the set, no dedicated test. Measure with the set next |
| `structured` | Windows as structured loops and ifs | `structure_semantics`; the whole suite with the feature | memset +97.0%, switch +66.4%; floats −14.3%, floatld −17.1%, vector −6.8% | +8.4% (+29.8% lines) | Irreducible parts as local dispatch | Keep off: mixed, larger, slow to build |
| `stub-windows` | PLT stubs in their own windows (the old path) | `plt` (with it off) | not built for the kernels (no PLT) | 0% | — | Keep off (it is the old behaviour) |
| `count-fallbacks` | Counts helper fallbacks (debugging) | `fallback_counts` | vector −13.5% (layout; see above) | +3.0% | — | Keep off (debugging) |
| `align-windows` | Build flag: every function 64-byte aligned; nothing emitted changes | none needed | Speed reproducible, not higher (task 18) | 0 | — | On: comparisons of later builds stay valid |
| `census` | Comments marking inlined code (`EmitOptions` only, not an `echora lift` option) | `census` and unit tests | — | comments | — | Not an `echora lift` option |

`known-returns` emits less Rust and compiles faster, but its machine
code is 12% larger on the test programs. Sizes are deterministic, so that
figure is not noise. A likely cause, not checked: with a constant
successor the compiler lays out the code after each call in place, where
it used to jump to the window's `match`.

The sets together, against the game set (c1 still held `float-run-loads`;
it changes only the scalar float runs with S loads, so leaving it out
changes only floats and floatld):

| set | significant changes | Rust: kernels; programs (rustc s, MiB) |
|---|---|---|
| c1: the per-instruction options, `direct-q-pairs`, `vector-float-runs`, `float-run-loads`, `steps-local` | memcpy +18.3%, memset +126.6%, switch +37.7%, vector +11.7%, ldst +43.8%, floatld +23.9%, vnorm +175.4%, convert +539.4%, intvec +589.4%, fpairs +77.3%, fpscalar +124.8%, misc +213.2%; **floats −14.0%** | without `float-run-loads` (the recommended set but windows): +44.1%; +33.9% (+32% rustc s, +29% MiB), `.text` +16.5% |
| c2: c1 + the function family | as c1, and calls +45.9%, calls2 +134.7%, gamecalls +26.7%, crossing +109.0% | functions add about 3% |
| c3: c1 + `window-calls`, `long-windows`, `known-returns` | as c1, and calls +71.4% (+52.0% to +81.6%), calls2 +124.4% (+101.1% to +159.8%), gamecalls +37.6% (+24.2% to +47.5%), crossing +85.1% (+61.1% to +122.1%); floats −16.0% | without `float-run-loads` (the recommended set): +48.8%; +31.7% (+38% rustc s, +27% MiB), `.text` +29.0% |

## Every option together: the whole aot-tests suite

`cargo test -p echora-aot-tests --features <game set>,<recommended
options>` (the per-instruction options, `direct-q-pairs`,
`vector-float-runs`, `steps-local`, `window-calls`, `long-windows`,
`known-returns`) ran the whole suite on main: 41 test binaries and 132
tests, all passing. They cover:
- the semantic suites against the reference interpreters and the
  Arm-pseudocode float model;
- the option suites, `window_calls`, `kernel_calls`, `smallest_normal`
  and `rounded_up`;
- the stress and concurrency suites, and golden.

With `long-windows` the split programs lift in 64 KiB regions, as the
game's do.

On the old base the same suite ran with the old recommended set and with
the function family added. Interactions found there:

1. The smallest-normal bug above. The float-run programs showed it once
   inline options were in alternate regions; with every program lifted
   with the same options, both sides of each comparison had it.
2. Three tests assumed the default features, not the emission:
   - `calls`' mixed-region lift used the feature options as its "without
     functions" base;
   - `plt` counted every `call_window` as a direct window call, which with
     `functions` also covers function calls;
   - `census` expected two calls for the calls kernel, where a summarized
     call is the call and its fallback.

   Each now states what it means and passes with the features on and off.

The run on main found no new interaction.

## Overlaps: which option wins

- **The per-instruction options never take the same form.**
  - With every inline option on, no SIMD word is taken by two of int,
    pairwise, fp-scalar, misc, the base forms with unary, and the
    structure transfers (`simd::overlap_tests`, 4 million sampled words;
    letting misc also take FCVTZS toward zero fails it).
  - So turning several on changes no instruction's emission from what
    each option alone gives it.
- **`vector-float-runs` and the per-instruction options** (FRSQRTE,
  FADDP, FABD, FMULX, FMAXV and others): the run's fast copy runs. When
  it declines, the exact copy runs, which is the per-instruction option's
  code where that option is on.
  - `vector_runs` lifts the run programs with every per-instruction
    option in alternate 1 KiB regions, so both kinds of exact copy are
    compared with the helpers.
- **`float-runs` and `inline-fp-scalar`** (scalar FSQRT in a run): the
  same, and `float_runs` lifts with `inline-vector` and `inline-fp-scalar`
  in alternate regions.
- **Charges and calls**, each pair checked by
  `codegen::tests::overlapping_options_emit_the_winners_code`: the
  emission with both equals the winner's alone, and the loser alone
  changes it.
  - `deferred-charges` over `steps-local`: `steps-local` is off where both
    are set.
  - `functions` over `direct-calls`: a BL to a lifted function calls it
    whatever `direct-calls` says.
  - `functions` over `window-calls`: window calls are off with lifted
    functions.
  - `window-calls` over `direct-calls`: a BL to a summarized target is a
    window call.
  - `functions` and `deferred-charges` over `long-windows`: no copies are
    made with either (checked on a loop across a window edge).
  - Making `steps-local` ignore `deferred-charges`, `direct-calls` take a
    call first, window calls ignore `functions` or long windows ignore
    `deferred-charges` fails it.

## Side's questions about the game

**(1) What `vector-float-runs` does to float tests, helper sites, spills
and reloads.** Census per kernel window on main, in three columns:
- game set;
- game set plus `vector-float-runs`;
- the recommended set.

| kernel | float tests (mode) | helper call sites | spills | reloads |
|---|---|---|---|---|
| vector | 27 (3) / 15 (1) / 15 (1) | 6 / 6 / 6 | 6 / 6 / 6 | 12 / 12 / 12 |
| vnorm | 43 (5) / 18 (1) / 18 (1) | 11 / 11 / 12 | 11 / 11 / 12 | 22 / 22 / 24 |
| misc | 7 (1) / 9 (1) / 9 (1) | 8 / 8 / 8 | 8 / 8 / 8 | 16 / 16 / 16 |
| fpairs | 0 (0) / 21 (1) / 21 (1) | 4 / 4 / 17 | 0 / 0 / 0 | 0 / 8 / 34 |

Where the operations are consecutive, a run cuts the tests on the
committing path by 44% to 58% and the FPCR mode tests to one; misc,
whose float operations are mostly not run forms, gains two. It removes
no helper call site, spill or reload: those are counted statically and
belong to the exact copy, which stays as the run's fallback. So in the
game's census:
- the 194 float tests per hot window would fall only for operations that
  are in runs;
- the 67 helper sites and 544 spills and reloads would not move.

fpairs has no float test with the game set because its pairwise
operations are helper calls there; the run tests their lanes instead.
Task 26's tool measures how many of the game's operations are in runs.

**(2) Scalar `fp_binary`, `fp_compare`, `fp_ternary`.**
- `inline-vector` (in the default set) already inlines scalar FADD,
  FSUB, FMUL, FNMUL, FDIV, FMADD and its kin, FCMP, FMAX and FMIN.
- The inlined forms run only under a nonzero FPCR (they test
  `m.cpu.fpcr != 0` first, because the helpers' own fast path takes FPCR
  0). The game runs with FZ set (FPCR 0x0100_0000), so they apply there.
- The helper sites are their fallbacks, one per inlined operation, taken
  for operands or results that are not normal or zero. They are code that
  is present, not calls that are made.
- `float-runs` merges the tests only for runs of at least eight
  single-precision operations. Nothing merges the tests of shorter scalar
  sequences or of doubles: each operation tests its operands, result and
  the FPCR mode itself. That gap could be measured with a report like
  task 26's for scalar operations.

**(3) Size, compile time and memory per region.** From the codesize
crate (every test program as one region-sized crate, `opt-level` 1, 16
codegen units):

| options | Rust bytes | rustc s | peak MiB | `.text` |
|---|---|---|---|---|
| game set | 26.4 MB | 145.2 | 4,498 | 202 B/ins |
| + unary | +1.2% | 144.9 | 4,536 | +0.8% |
| + int | +4.7% | 161.6 | 4,653 | +1.3% |
| + pairwise | +0.7% | 150.1 | 4,508 | +0.4% |
| + fp-scalar | +5.5% | 161.3 | 4,707 | +3.9% |
| + misc | +8.9% | 165.7 | 4,874 | +8.1% |
| + q-pairs | +0.1% | 142.8 | 4,495 | +0.3% |
| + vector-float-runs | +0.1% | 143.8 | 4,503 | +0.0% |
| + float-run-loads | +0.0% | 145.8 | 4,517 | +0.0% |
| + steps-local | +12.0% | 158.6 | 4,947 | +1.4% |
| + the function family | +2.9% | 149.5 | 4,564 | +3.4% |
| + window-calls | +1.5% | 145.1 | 4,536 | +2.9% |
| + long-windows | +0.0% (1 KiB regions) | 141.7 | 4,459 | +0.0% |
| + known-returns | −2.5% | 136.4 | 4,403 | +12.4% |
| recommended set but windows | +33.9% | 191.0 | 5,784 | +16.5% |
| recommended set | +31.7% | 199.7 | 5,689 | +29.0% |

Each figure is one compile; differences under about 5% in rustc time are
noise. The random test programs hold few runs, so `vector-float-runs`
costs little here and +17% on the kernels, where every float operation is
in one. The game's region 0x7101c7 grew 13% in lines with it. The
programs lift in 1 KiB regions, so `long-windows` copies nothing there;
on the kernels (one 64 KiB region) it adds 3.1%, and in the game it adds
at most one window's worth of blocks (256 instructions) per window, which
`echora lift` prints. The recommended set emits less Rust than the set
without the windows because of `known-returns`, but compiles 5% slower and
has 11% more machine code.

The recommended set on every region would cost about 38% more rustc time
and 27% more memory per region: a 40-to-120-minute rebuild becomes about
55 to 165 minutes, and rustc jobs need 27% more memory. On the ~100 hot
regions only, about 9% of them, it adds about 3% to 4% to the whole
rebuild. The per-instruction options, `steps-local` and the window calls
go to the hot regions for that reason.

## Method

- Speed: `crates/echora-aot-tests/benches/kernels.rs`, built with
  `RUSTFLAGS="-C llvm-args=-align-all-functions=6"` and `--features
  <game set>,<option>`.
  - Each binary ran `--runs 7 --bench`, alternating with the others,
    pinned with `taskset -c 2`.
  - Each figure is the median of the per-run medians against the game
    set's, with a 4,000-sample bootstrap interval of the ratio of medians.
- Sizes: `build.rs`'s `kernel_sizes` (the kernels' module text per
  option) and `codesize` modes, printed by `benches/codesize.rs`
  (in `build.rs`, `recommended_set` is the recommended set without the
  window calls and `recommended_windows_set` the recommended set).
- Census: `cargo test -p echora-aot-tests --features <options> --test
  census -- --nocapture`.
