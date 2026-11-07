# Testing

`cargo test --workspace` needs no game files. It covers:

| Crate | Coverage |
|---|---|
| `echora-nup` | Synthetic archives built by a test-only writer (`tests/common`): raw and zstd extents, shared extents, navigation and lookup, meta records, extraction, parallel verification; rejection of a tampered TOC, UUID mismatch, truncated or extended files, unknown codecs, out-of-bounds or misaligned extents, names outside the pool, path-traversal names, and an update without a base. Corrupted extent bytes are detected by both verification and reads. |
| `echora-content` | NSO and NPDM header parsing; flat and layered dump directories; numeric update ordering; program-package selection; content-addressed ExeFS caching, reuse, repair of a damaged entry and re-keying on changed input. |
| `echora-loader` | ZBIC step-table generation; hand-derived BIC descriptions (including low-probability counts); standard table descriptions read back unchanged; byte-identical re-transcoding of libzstd frames at four levels over seven inputs (raw, RLE, compressed and treeless literals, FSE, RLE, predefined and repeat sequence modes); truncated and trailing-data frames. NSO parsing: LZ4 modules, hash mismatch, bad sizes, overlapping segments, header overlap, image budget, flag handling, ExeFS load order and required modules. MOD0: signed relative offsets, unterminated dynamic tables, out-of-range relocation tables, bounded GNU-hash chain walks, GNU-only symbol tables with RELR and exported-symbol lookup, unknown tags. |
| `echora-guest` | CPU register/flag helpers, PAuth stripping, SIMD/FP semantics and fast paths, guest memory mappings, permissions, aliases, atomics and the guest view (memory-backed tests on Linux and Windows); on Linux every guest view mapping is checked to be excluded from core dumps; on Windows, page-granular view splits, aliases and chunk release are checked. |
| `echora-lift` | Decoding, discovery, jump tables, block construction and emitted Rust for each instruction family; register def/use per instruction; spin-loop detection, call targets and structured control flow on assembled programs. A differential test decodes random operand fields of every accepted encoding class with both the decoder and yaxpeax-arm and compares opcode, registers (SP versus zero register), immediates, targets and addressing, and does the same for every SIMD and floating-point operation (scalar sizes, arrangements, lanes, register groups); known differences are named in the test, and a million random words check that every accepted word is valid. |
| `echora-aot-tests` | Programs lifted at build time and executed natively: CRC32/CRC32C against zlib-derived values, MRS/MSR NZCV, FP across-lane reductions with NaNs and signed zeros, CASP (X and W forms, aliasing, misalignment), SABAL/UABDL, UDOT/SDOT against independently computed results, integer and flag results, call/return with and without PAuth hints, step budget, unsupported system registers, undiscovered indirect targets, calls across dispatch windows and holes inside them, CAS/LDADD at every width including register aliasing and SP forms, load/store-exclusive loops and monitor clearing, LDAR/LDAPR/STLR, alignment faults, and a store to an unmapped page ending the process with the guest address (in a child process with core dumps disabled). A semantic differential runs about 1,900 random encodings of the integer classes (add/subtract in every operand form, carries, logical, move-wide, bitfield, EXTR, conditional compare and select, multiplies, divides, variable shifts, bit counts and reversals, loads and stores in every addressing mode, pairs) natively from 24 random register, SP, flag and memory states each, and compares the results with a reference interpreter written from the Arm pseudocode. The spin-loop detector is run on the benchmark kernels (only the polling loop spins). |
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
