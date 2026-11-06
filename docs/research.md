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
| [Mars7x/pokopia-recomp](https://github.com/Mars7x/pokopia-recomp) `8a308dd` | GPL-3.0-or-later | AArch64 lifter emitting Rust (`src/aarch64.rs`, `src/codegen.rs`, `src/simd.rs`); guest CPU/memory/SIMD crate (`crates/guest`); NSO/NPDM/MOD0 loader (`src/loader/`); kernel SVCs, threads, synchronization, IPC, ~11k lines of services (`src/runtime.rs`, `src/threads.rs`, `src/synchronization.rs`, `src/ipc.rs`, `src/services.rs`); high-level NVN on Vulkan (`src/nvn*.rs`, `crates/gpu`) with Eden-derived shader translation; evdev input; PipeWire audio. Reaches gameplay in another Switch 2 title. | Primary base. Port the loader, lifter and guest crate first, then runtime pieces as OoT reaches them. Keep Pokopia's repository untouched; Echora carries adapted copies with provenance notes until shared crates are agreed. |
| Pokopia, known limits | — | NSO loader rejects ZBIC; `crates/guest/src/host_memory.rs` uses Linux `mmap` directly; input is evdev-only; audio is PipeWire-only; every NVN draw runs under a global machine lock (its README names this as the main frame-rate limit); resolution scaling is an env var (`4ac0962`). | Each becomes an Echora boundary: ZBIC decoder in the loader, host-memory backend trait, portable input/audio backends, NVN execution designed without a global lock, resolution through `echora-resolution`. |
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
- Thread ID seeded at TLR+0x110 at thread creation (Pokopia already does this).
- SDK 22 `clock_gettime` bound to `nn::time` shared memory.
- `IAudioDevice` commands 15–18, `IAudioRenderer` 12/13, `auddmg:u`.
- RomFS directory listing and multiple mount aliases (`content`, `_npln`).
- `nvnDeviceGetInteger` reporting minor version ≥ 40.
- `nn::socket` event counters; gRPC aborts at boot without them.
- SVC 0x34/0x35 WaitForAddress/SignalToAddress arbitration type 3 (64-bit).
- NVN2 mesh/task pipelines (`DrawMeshTasksIndirect` in the opening), AV1 and
  NDVE movie decoding, resolution-dependent `TLD4.SCR` gathers.

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

## Licensing decision

Echora is GPL-3.0-or-later, matching Pokopia so its code can be carried over,
and compatible with the GPL-3.0 emulators and MIT/0BSD/Apache dependencies.
GPLv2-only sources are not incorporated.
