# Testing

`cargo test --workspace` needs no game files. It covers:

| Crate | Coverage |
|---|---|
| `echora-nup` | Synthetic archives built by a test-only writer (`tests/common`): raw and zstd extents, shared extents, navigation and lookup, meta records, extraction, parallel verification; rejection of a tampered TOC, UUID mismatch, truncated or extended files, unknown codecs, out-of-bounds or misaligned extents, names outside the pool, path-traversal names, and an update without a base. Corrupted extent bytes are detected by both verification and reads. |
| `echora-content` | NSO and NPDM header parsing; flat and layered dump directories; numeric update ordering; program-package selection; content-addressed ExeFS caching, reuse, repair of a damaged entry and re-keying on changed input. |
| `echora-loader` | ZBIC step-table generation; hand-derived BIC descriptions (including low-probability counts); standard table descriptions read back unchanged; byte-identical re-transcoding of libzstd frames at four levels over seven inputs (raw, RLE, compressed and treeless literals, FSE, RLE, predefined and repeat sequence modes); truncated and trailing-data frames. NSO parsing: LZ4 modules, hash mismatch, bad sizes, overlapping segments, header overlap, image budget, flag handling, ExeFS load order and required modules. MOD0: signed relative offsets, unterminated dynamic tables, out-of-range relocation tables, bounded GNU-hash chain walks, GNU-only symbol tables with RELR and exported-symbol lookup, unknown tags. |
| `echora-guest` | CPU register/flag helpers, PAuth stripping, SIMD/FP semantics and fast paths, guest memory mappings, permissions, aliases, atomics and the guest view (memory-backed tests on Linux and Windows); on Linux every guest view mapping is checked to be excluded from core dumps; on Windows, page-granular view splits, aliases and chunk release are checked. |
| `echora-lift` | Decoding, discovery, jump tables, block construction and emitted Rust for each instruction family; register def/use per instruction; spin-loop detection, call targets and structured control flow on assembled programs; a differential test of the decoder against yaxpeax-arm (see [Differential and semantic tests](#differential-and-semantic-tests)). |
| `echora-aot-tests` | Programs lifted at build time and executed natively: CRC32/CRC32C against zlib-derived values, MRS/MSR NZCV, FP across-lane reductions with NaNs and signed zeros, CASP (X and W forms, aliasing, misalignment), SABAL/UABDL, UDOT/SDOT against independently computed results, integer and flag results, call/return with and without PAuth hints, step budget, unsupported system registers, undiscovered indirect targets, calls across dispatch windows and holes inside them, CAS/LDADD at every width including register aliasing and SP forms, load/store-exclusive loops and monitor clearing, LDAR/LDAPR/STLR, alignment faults, and a store to an unmapped page ending the process with the guest address (in a child process with core dumps disabled). Also the semantic differential tests below. |
| `echora-resolution` | Non-integer scales, internal resolution independent of output, exact-size aspect checks, ultrawide pillarboxing and view expansion (21:9, 32:9), 16:10, seamless viewport tiling at 1366x768, fixed versus screen-relative targets, change impact, degenerate sizes. |
| `echora-config` | Round trip, partial files, unknown-field and range rejection, future schema rejection, change classification, requested-versus-effective resolution, atomic save. |
| `echora-runtime` | 265 tests: IPC framing and the system services, threads, scheduling and synchronization, NPDM capabilities, content metadata and RomFS, file systems, display and HID, the NVIDIA driver, NVN formats and SASS program lengths, the audio renderer, AV1 decoding, logging. |
| `echora-gpu` | Vulkan on the host's driver: a triangle draw, a compute dispatch into a storage buffer, whole and partial clears read back; the shader library's ABI layout and short-program rejection; loop guards on SPIR-V; rational resolution scales and aspect-preserving presentation. |
| `echora-launcher` | Installing a file or dump and reinstalling in place; reading `echora prepare`'s progress lines and the progress bar's weighting; GNOME button layouts; resize edges; directional focus (in-line preference, returning where a move came from); the spring curve; series titles. |
| `echora` | `prepare`: counting built region crates from cargo's messages, when a translation counts as prepared (its runner, and no other build's stamp), build jobs. |
| `echora-input` | evdev capability words, button mapping by position (with the xpad north/west swap), triggers, hats and sticks within the circle. |
| `echora-audio` | Volume scaling and clamping; a bounded queue that plays silence on underrun. |

Ignored tests, run with `ECHORA_TEST_NUP` set, open the real archive, verify
every ExeFS file, and check that NUP and directory inputs produce the same
ExeFS key and build ID. With `ECHORA_TEST_EXEFS` set to a cached ExeFS
directory, the loader test decompresses every module, requires every
segment to be hash-checked and every module to carry MOD0 dynamic metadata
with a counted symbol table.

Shader translation is checked offline against the stages a run dumps
(`ECHORA_SHADER_DIR`): `translate-dump` in `echora-gpu` translates each one
and groups the failures by reason, and `shader-check` also compiles each
translation on the host's Vulkan driver.

The running game is checked from its run reports and from presented frames
saved with `ECHORA_CAPTURE_EVERY` ([development](development.md)). These
checks need the game and are not part of CI.

## Differential and semantic tests

These need no game and run with `cargo test --workspace`. They compare the
decoder with an independent one, and lifted code with reference
interpreters written from the Arm pseudocode in the tests themselves. Each
test runs on its own with `cargo test -p <crate> --test <name>`; add
`-- --nocapture` to see the counts it logs. Every one was checked by
breaking its reference on purpose and seeing it fail.

| Test | Checks |
|---|---|
| `echora-lift` `decoder_differential` | Random operand fields of every encoding class the decoder accepts, decoded by it and by yaxpeax-arm: opcode, registers (SP versus zero register), immediates, targets, addressing, system registers, and for SIMD/FP the scalar sizes, arrangements, lanes and register groups. A million random words: every accepted word must be valid. Known name differences are listed in the test. |
| `echora-aot-tests` `semantics` | About 3,000 random encodings of the integer classes (ALU in every operand form, flags, bitfield, multiply, divide, shifts, bit operations, loads and stores in every addressing mode, pairs, conditional branches, CRC32, CAS/CASP and the other LSE atomics, acquire/release, exclusives, MRS/MSR NZCV), each from 24 random states: registers, SP, NZCV, memory and branch outcome. |
| `simd_semantics` | About 3,900 integer SIMD encodings: vector and scalar arithmetic, compares, shifts (saturating and rounding included), pairwise and across-lanes, widening and narrowing, dot products, PMULL, permutes, EXT, TBL/TBX, DUP, INS, UMOV/SMOV, MOVI, vector logic, AES and SHA-1/SHA-256. |
| `fp_semantics` | About 5,300 scalar and vector FP encodings, sampled per operation, on inputs with NaNs, infinities, zeros and denormals, with FPCR.FZ and DN at random: NaN propagation, default NaN, flush to zero, fused forms, FMAX/FMIN and NM, conversions in every rounding mode, half precision, FRECPE/FRSQRTE and the step forms, and the FPSR flags IOC, DZC, OFC and IDC. A known runtime difference (the sign of a NaN through FNMUL, FRECPS and FRSQRTS) is accepted and counted. |
| `simd_memory_semantics` | SIMD loads and stores: LDR/STR B to Q in every addressing mode, LDP/STP S/D/Q, LD1-4/ST1-4 (multiple and single), LD1R-4R, post-indexing. |
| `sequences` | 600 random sequences of 4 to 12 integer, flag and memory instructions with forward branches: interactions inside a block, straight and with one- to three-instruction slices (resuming through the dispatcher after host boundaries). |
| `loops` | Counted loops with mid-loop entries, ADR + BR to blocks reached only through a register, ADRP + ADD + BLR into leaves mid-window and mid-block; straight and sliced. |
| `switches` | 200 jump tables (2 to 8 ways, halfword and byte entries) recovered by discovery; straight and sliced. |
| `calls` | 300 callers using BL and BLR into leaf functions in other windows, some nesting calls behind STP/LDP of X29/X30: registers, memory and stack. |
| `exclusives` | LDXR/STXR pairs on one context, sliced, and on two contexts interleaved on one memory, against the runtime's local and global monitor model (a successful STXR on one context fails the other's pending pair). |
| `spin_kernels`, `spin_semantics` | The spin-loop detector on the benchmark kernels (only the polling loop spins) and on 400 random loop candidates and every generated program: each reported loop, run from random states, never exits once it has come back to its head, and every pure poll is found. |
| `structure_semantics` | Every window's structured tree from `echora_lift::structure` (blocks, loops, ifs, dispatch, irreducible regions and their switches) run on the reference, against plain execution of the window: straight from every entry, and sliced, with every block an entry and the tree left and re-entered after a third of the blocks. |

The generated programs come from `build.rs` in `echora-aot-tests`, with
fixed seeds. Memory instructions in them use X28 as their base and X27 as
their index, which nothing else writes, so every access stays inside the
data area: an unmapped guest access ends the process by design.

## Codegen speed without the game

`echora-aot-tests` has a benchmark of lifted code. It needs no game and runs
in seconds, so codegen changes can be compared before a game run:

```
cargo bench -p echora-aot-tests --bench kernels
cargo bench -p echora-aot-tests --bench kernels -- calls --scale 4 --runs 5
```

`build.rs` lifts seven kernels from `crates/echora-aot-tests/benches/kernels.s`
through split emission, the layout real builds use, and the benchmark runs
them natively, reporting guest instructions per second (best and median of
`--runs`):

| Kernel | What it exercises |
|---|---|
| `arith` | A tight integer loop: add, shifted EOR, MADD, ROR, flags and a back-edge. |
| `memcpy` | 64 KiB copies with LDP/STP of Q registers and post-indexing. |
| `memset` | 64 KiB fills with STP of X registers. |
| `calls` | Two BLs per iteration to leaf functions in another dispatch window. |
| `switch` | An eight-way switch through a halfword jump table (BR). |
| `vector` | FMLA, FADD and FMUL on 4S vectors loaded with post-indexing. |
| `poll` | An LDAR/CBZ polling loop; the host ends it after 500 slices of 100,000 instructions, as another thread setting the flag would. |

A positional argument filters kernels by name, and `--scale` multiplies each
kernel's work. The kernels are original code. To change them, edit
`kernels.s` and run `benches/assemble_kernels.py` (it needs llvm-mc,
llvm-objcopy and llvm-objdump) to regenerate `src/kernels.rs`. The words are
checked in, so building needs no assembler. `cargo test` builds the kernels
but does not run the benchmark.

CI runs formatting, clippy with warnings denied, and the tests on Linux
x86-64 and Windows x86-64. Passing CI means the code compiles
and the tests pass on that host; it says nothing about running the game.
