# Research notes and reuse audit

Audit performed 2026-11-06. Paths and commits are recorded so the work is not
repeated.

## Checked input

| Item | Value |
|---|---|
| Container | NUP v1, one package, 25,020,741,061 bytes |
| Package | `update`, `standalone_update` flag set (no base in container) |
| Title ID | `0400486024DEC000` |
| Version | 131072 (`0x20000`); display version TLV `1.1.1` |
| Meta records | NPDM, icon (JPEG), display version, title name, source dump name; no build-ID record |
| `main` build ID | `e0f3ccfaf50065ae4ea01eb5aeb13b1a45da3348` |
| ExeFS SHA-256 | `main` `7db9ddfd…`, `main.npdm` `94c154d8…`, `rtld` `219f25ef…`, `sdk` `3ae3cd4c…`, `subsdk0` `0d32c4bd…` |
| `main` NSO | flags `0xff`; text `0x3c461d0`, rodata `0xc05080`, data `0x4f9b80`, bss `0x159480` (memory sizes) |
| Other NSOs | flags `0x3f` (classic LZ4 + hash) |
| RomFS | 186,587 files, 24.43 GiB; top-level directories include `Phive`, `Bake`, `Banc`, `AISchedule`, `RSDB`, `agl`, `gsys` |

The RomFS layout is that of Nintendo EPD's recent in-house engine, not the
N64 codebase, so N64 decompilation projects contribute game knowledge only.

The ExeFS hashes above were cross-checked against files extracted by the
NUP reference tool (`a55uka/nup` `d40d9cf`).

## Reuse matrix

| Source | License | What exists | Decision |
|---|---|---|---|
| [a55uka/nup](https://github.com/a55uka/nup) `d40d9cf` | MIT | Format spec and Python reference (pack, info, extract, verify). | Used for format rules and cross-validation. A Rust reader was still needed: the runtime must read RomFS files in place (24 GiB) without extraction or a Python dependency. |
| [ly05010419/MacNativeGames](https://github.com/ly05010419/MacNativeGames) issue #1773 | MIT | OoT bring-up on the same build (v131072) on a direct-execution Apple-silicon runtime. Branch `MD5Visual/MacNativeGames` `feat/oot-boot` (`2e7e6036`) reaches the first presented frame, then stalls before frame 2; a later report aborts in `InitializeAv1DecoderByTransferMemory`. | Checklist of required behavior (below). Their architecture runs guest ARM64 code directly and is a reference for a future ARM64-host backend, not for the x86-64 path. |
| [seedds/nxnative](https://github.com/seedds/nxnative) | MIT | NSO rewriter, Horizon HLE host, NVN→Metal, AV1 backend; PRs #599 (clock_gettime ifunc), #1283 (socket event counters). | Reference for SDK 22 behavior and the Metal backend. |
| [LNPANDY/NX2Apple](https://github.com/LNPANDY/NX2Apple) | MIT | Packaging front end for nxnative-style builds. | Not needed now. |
| [Djihads80/Eglantine](https://github.com/Djihads80/Eglantine), [zortos293/Lichii](https://github.com/zortos293/Lichii) | GPL-3.0 | Switch 2 emulators; OoT issues Lichii #87, #112, Eglantine #567 cover WaitForAddress type 3, mesh/task pipelines, Ampere shader forms (`TLD4.SCR`). | License-compatible; consult for shader and SVC semantics when reached. |
| [kinnay/zbic](https://github.com/kinnay/zbic) `11b08f2` | GPLv2-only | ZBIC compression (zstd variant introduced with Switch firmware 22.0.0), bundling Atmosphère's zstd sources. | Used only to learn the format; no code taken. Echora's decoder is an independent transcoder (see below). |
| zeldaret/oot, Shipwright, Zelda64Recomp | various | N64 decompilation and ports. | Gameplay knowledge only; no code or execution design transfers. |

## OoT requirements reported by other projects

From MacNativeGames #1773 (game-agnostic fixes that title forced), to be
confirmed against Echora's own runs rather than assumed:

- ZBIC-compressed NSO segments in `main`.
- Dynamic symbol tables with `DT_GNU_HASH` only (no `DT_HASH`).
- Thread ID seeded at TLR+0x110 at thread creation.
- SDK 22 `clock_gettime` bound to `nn::time` shared memory.
- `IAudioDevice` commands 15–18, `IAudioRenderer` 12/13, `auddmg:u`.
- RomFS directory listing and multiple mount aliases (`content`, `_npln`).
- `nvnDeviceGetInteger` reporting minor version ≥ 40.
- `nn::socket` event counters; gRPC aborts at boot without them.
- SVC 0x34/0x35 WaitForAddress/SignalToAddress arbitration type 3 (64-bit).
- NVN2 mesh/task pipelines (`DrawMeshTasksIndirect` in the opening), AV1 and
  NDVE movie decoding, resolution-dependent `TLD4.SCR` gathers.

Confirmed in Echora so far: ZBIC segments in `main` (see below), symbol
tables reached only through `DT_GNU_HASH`, and AV1 for the intro movie
(decoded with rav1d). The mesh and task pipelines are translated by the
shader recompiler but not yet drawn.

## ZBIC format

Studied from `kinnay/zbic` (`src/zstd.c`, the code under
`ZSTD_ZBIC_SUPPORT`). ZBIC differs from Zstandard in two places only:

- The frame magic is `ZBIC` (`0x4349425A`).
- FSE table descriptions (Huffman weight tables and the literal-length,
  offset and match-length tables) use binary interpolative coding. Byte 0
  holds the description length minus one (low 7 bits) and a
  low-probability flag (bit 7). The remaining bytes, read from the end
  toward the start into a state kept at 40 bits or more, encode mixed-radix
  digits: max symbol minus one (radix 52), table log minus 5 (radix 8), the
  final cumulative value, then one digit per step of an interpolative walk
  over cumulative counts. The walk order is a right-first preorder traversal
  of [0, 256] plus a terminal step; Echora generates it instead of storing
  the reference's table.

Echora decodes each description, re-encodes it in standard form, adjusts the
enclosing Huffman header, literals size and block size, and decompresses the
result with libzstd. With standard coding on both sides the same transcoder
reproduces libzstd's own frames byte for byte. On the checked input all three
`main` segments match their header SHA-256 values.

`main` has NSO flags `0xff`: bits 0-5 as usual, bit 7 ZBIC, bit 6 unknown.

## Module metadata

From `echora load` on the checked input. Every module has MOD0 and a dynamic
section, and every symbol table is indexed by `DT_GNU_HASH` only (no
`DT_HASH`). Relative relocations are packed as RELR.

| Module | soname / needs | Symbols (defined/undefined) | RELA (ABS64, GLOB_DAT) | RELR | JUMP_SLOT | init_array |
|---|---|---|---|---|---|---|
| `rtld` | `nnrtld.nss` | 3 (0/2) | 1 (0, 1) | 2 | 1 | 0 |
| `main` | needs `movie.nss`, `nnSdk.nss` | 1,249 (65/1,183) | 2,051 (2,004, 47) | 10,138 | 1,110 | 9,783 |
| `subsdk0` | `movie.nss` | 1,083 (824/258) | 462 (420, 42) | 16 | 506 | 0 |
| `sdk` | `nnSdk.nss` | 28,707 (28,698/8) | 10,857 (9,414, 1,443) | 531 | 10,082 | 96 |

The guest rtld applies these relocations at startup; the host only validates
and reports them.

## Code discovery on the checked input

`echora analyze game.nup --limit 20000000` (2026-11-06, release build,
Ryzen 7 5800X3D). Modules are placed from `0x7100000000`; roots are the
rtld entry and every module's `init_array` targets; deferred roots are
relative-relocation targets (RELA and RELR) and `PACIBSP` prologues; jump
tables are followed through rodata.

| Module | Constructors | Relocation targets | Prologues | Discovered / text words |
|---|---|---|---|---|
| `rtld` | 0 | 5 | 18 | 2,336 / 2,512 (93.0%) |
| `main` | 9,783 | 151,750 | 161,622 | 15,256,834 / 15,800,436 (96.6%) |
| `subsdk0` | 0 | 67 | 722 | 129,184 / 135,644 (95.2%) |
| `sdk` | 96 | 7,633 | 20,524 | 1,955,604 / 2,226,708 (87.8%) |

17,343,958 instructions in 3,727,283 blocks, 1,255 jump tables (37,327
targets), no unresolved indirect targets, 13.6 s for discovery and a peak
of 6.2 GB resident (about 340 bytes per discovered instruction).

The first run found 19 instruction forms the ported lifter did not
support. Implemented since, each with native execution tests:

| Instruction | Sites | Implementation |
|---|---|---|
| `fmaxv`, `fminv`, `fmaxnmv`, `fminnmv` (4S) | 601 | tree reduction over the exact FP lane operations |
| `crc32b/h/w/x`, `crc32cb/cx` | 325 | reflected bitwise CRC32 / CRC32C |
| `mrs Xt, NZCV`, `msr NZCV, Xt` | 16 + 16 | flags in bits 31:28 |
| `casp`, `caspa`, `caspl`, `caspal` | 6 | 64-bit or 128-bit host compare-and-swap (`portable-atomic`) |
| `sabal` (and `sabdl`/`uabal`/`uabdl`) | 2 | widening absolute difference |
| `udot` (and `sdot`), vector form | 1 | 32-bit lane dot products |
| `udf` | 8,515 | `Trap::UndefinedInstruction`; execution does not continue past it |

Still unsupported: one `stg` (memory tagging). Without MTE it is an
undefined instruction; it stays an explicit stop until a path that reaches
it is found.

## Translation size and compile cost

`echora lift game.nup` (2026-11-06) writes the translation to
`<cache>/translations/<ExeFS key>/`: a dispatch crate and one crate per
64 KiB guest region (1 KiB dispatch windows inside each).

| Measure | Value |
|---|---|
| Instructions | 17,343,958 |
| Region crates | 1,111 |
| Generated Rust | 4,220 MiB (0.57 MB to 5.3 MB per region, median 4.0 MB) |
| Emission time / peak memory | 197 s / 6.2 GB (discovery dominates; regions stream to disk) |

Compiling single regions in release mode (`opt-level = 1`, one job):

| Region | Source | Time | Peak memory |
|---|---|---|---|
| `710000` (rtld) | 0.57 MB | 2.4 s | 0.24 GB |
| `710335` (median) | 4.0 MB | 14.9 s | 0.81 GB |
| `71050a` (largest) | 5.3 MB | 24.2 s | 1.12 GB |

`cargo check --release -j 6` over the whole translation (1,111 regions
plus the dispatch crate) passes with no errors or warnings in 15 min 6 s;
the largest single process peaks at 3.8 GB. Every emitted instruction form
therefore type-checks against `echora-guest`.

Extrapolated, a full build is about 4.5 CPU-hours of region compilation;
at six parallel jobs it needs roughly 7 GB of memory. Generated output
belongs on disk: `/tmp` is a RAM-backed tmpfs on the development machine.

## Mesh and task shaders

The game's NVN table includes the whole `DrawMeshTasks` family, and the OoT
bring-up notes say the opening uses `DrawMeshTasksIndirect`. The shader
recompiler was first pinned to PokeDinner, which has no task or mesh
stages. Eglantine (GPL-3.0, branch `nx2`) added them in `310dd59`
("translate Switch 2 task and mesh programs", for VK_EXT_mesh_shader): task
programs write a payload, mesh programs write vertex and primitive groups,
and the two mesh instructions sirit lacks are patched in after assembly.
The RX 9070 XT (RADV) exposes task and mesh stages with subgroup size
control 32-64.

## Recompiler pin: Lichii

The pin moved from PokeDinner (`c3ce9bc`) to Lichii (`809fb3a`), an
Eglantine fork that carries the task and mesh work and lowers more of OoT's
Ampere code. The two diverge from a common Eglantine base of 2026-10-30.
Measured with `translate-dump` and `shader-check` (translate, then compile
on RADV) over every stage OoT reached in a 3001-frame run through the title
and intro movie (6074 stages, `EDEN_NX2_TENSOR=1`):

| Pin | Translated | Compiled on the driver |
|---|---|---|
| PokeDinner `c3ce9bc` + 9 patches | 4472 | 4472 |
| Lichii `809fb3a` + 5 patches | 5960 | 5960 |

No stage translates with PokeDinner and fails with Lichii. The compute
stages PokeDinner skipped in that run (LEA op 291, op 4a4, a USHF form
op 899) all translate with Lichii. Over a further 10,044 captured Switch 2
programs from another title, Lichii translates and compiles 8485 against
PokeDinner's 7830, again with none lost.

Five of PokeDinner's patches are upstream in Lichii in more general forms
(TXQ second destination, TLDS x/z pairs, FSETP with a uniform source,
undefined registers as zero, UIMAD.WIDE with an immediate) and were
dropped. Patch `0010` keeps the uniform clamp idiom (UISETP then USEL)
foldable to IMNMX through any UPn, as PokeDinner did; Lichii accepted only
UP0, so indirect-branch tables guarded through another UPn failed to track.
The 87 stage-buffer reads were mesh (81) and task (6) programs translated
as vertex programs: classified as at draw time (`translate-dump` does), all
translate. The remaining 27 failures are gather forms (op b63, 15), op c08
(6) and register shadowing modes (6).

## Licensing decision

Echora is GPL-3.0-or-later, compatible with the GPL-3.0 emulators and MIT/0BSD/Apache dependencies.
GPLv2-only sources are not incorporated.
