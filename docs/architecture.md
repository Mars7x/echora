# Architecture

Echora is a portable core with host-specific backends. The core never calls
operating-system, windowing or graphics APIs directly; those live in backend
crates selected per host.

## Crates

| Crate | Layer | Responsibility | State |
|---|---|---|---|
| `echora-nup` | core | NUP v1 reader: superblock/TOC validation, tree lookup, streaming zstd reads with SHA-256 checks, safe extraction, parallel verification. | Implemented, tested on synthetic and real input. |
| `echora-content` | core | One content model for NUP and dump directories: packages, versions, ExeFS module identification (NSO/NPDM headers), program-package selection, content-addressed ExeFS cache. | Implemented, tested. |
| `echora-loader` | core | NSO parsing with bounds checks, LZ4 and ZBIC segment decompression, segment hash verification, ExeFS load order and image budget, process layout and discovery roots (prologues, RELA/RELR relative targets, `init_array`). ZBIC frames are transcoded to standard Zstandard and decoded by libzstd. MOD0 and dynamic-section analysis (symbols via SysV or GNU hash, RELA/REL/RELR/JMPREL tables, init/fini arrays, exported-symbol lookup), read-only. | Implemented, verified on the real modules. |
| `echora-guest` | core + host backend | Guest CPU state, guest memory (mappings, permissions, TLB, exclusive monitor), SIMD/FP semantics, traps, and the `Core` that generated code runs against. Guest memory has a per-host backend (`host_memory/`): Linux maps one memfd twice so generated code accesses guest memory directly. | Ported from Pokopia; 57 tests. Linux backend only. |
| `echora-lift` | core | AArch64 decoding (scalar and SIMD/FP), code discovery from entries, prologues, relocation targets and jump tables, basic blocks, and Rust emission. Generated code depends only on `echora-guest` and exposes `dispatch(&mut Core)`. | Ported from Pokopia; 53 tests. |
| `echora-aot-tests` | test | Synthetic AArch64 programs lifted by `build.rs` and run natively on a minimal host loop (integer, flags, calls, PAuth hints, budgets, window dispatch, missing blocks, CAS/LDADD/exclusives, acquire/release). | 11 execution tests (Linux). |
| `echora-runtime` | core (+ host stand-ins) | Process bootstrap (module mapping, stack, TLS, entry registers, NPDM capabilities including the private kernel resource pool), scheduler and threads, synchronization, kernel SVCs, IPC framing and the system services, file systems and RomFS images, HID shared memory. NVN keeps only its entry-point table; window, input and audio are stand-ins with no host device. | Ported from Pokopia; 231 tests (Linux). Boots the real game's process; not yet run. |
| `echora-resolution` | core | Resolution planning: guest logical size, guest view, internal scene size, presentation rectangle, output size; target/viewport mapping; change impact. | Implemented, tested. Not yet used by a renderer. |
| `echora-config` | core | Versioned TOML settings, validation, migration, requested vs effective settings, apply classes for live changes. | Implemented, tested. Not yet used by a runtime or launcher. |
| `echora` | front end | Command-line tool; owns host paths (cache location). | `inspect`, `verify`, `import`, `load`, `analyze`, `lift`, `extract`. |

Planned crates, in roadmap order: `echora-gpu` (portable graphics translation) with
`echora-gpu-vulkan`, and host crates for windowing, input and audio. The
GTK4/libadwaita launcher will be a separate binary crate that depends on
`echora-config` only, never the reverse.

## Boundaries

- **Portable core:** content, loader, lifter, IR, guest semantics, runtime
  service logic, resolution planning, configuration, diagnostics. These
  crates use `std` only for files and threads. The one host-specific detail
  so far, positional file reads, is a narrow `cfg` in `echora-nup/src/read.rs`.
- **Host backends:** executable memory, windowing, graphics API, audio,
  input, timing, native dialogs. Each sits behind a narrow interface owned by
  the core; static dispatch is preferred where only one backend can exist per
  build.
- **Front ends:** the CLI and launcher decide host paths and present
  settings; they do not contain game or runtime logic.

## Execution model (planned)

The design follows pokopia-recomp, which already runs another Switch 2 game
through gameplay on Linux:

1. Load `rtld`, `main`, `subsdk*`, `sdk` at deterministic bases; the guest's
   own rtld performs relocation.
2. Discover code from function prologues, relocation targets and jump
   tables; translate ahead of time; report every unresolved indirect target
   as a deterministic missing-block stop.
3. Emit generated code that the host toolchain compiles natively. Pokopia
   emits Rust and relies on LLVM for x86-64 code; the same path reaches ARM64
   hosts without a second backend. The IR boundary is kept so that a direct
   machine-code backend can be added if measurements justify it.
4. Each unimplemented instruction, SVC or service stops the run with a
   precise report rather than returning invented success.

## Guest memory backends

Generated code loads and stores through a host view of guest memory, so the
backend decides both speed and portability. The Linux backend maps one sparse
memfd twice (a host-side physical view and a guest view with per-page guest
permissions); a forbidden guest access faults in the host and ends the
process with the guest address reported. It needs 4 KiB host pages. Both
views are excluded from core dumps: dumping a shared mapping of the
address-space-sized file would fault in every page. Windows (64 KiB view
granularity) and 16 KiB-page hosts such as Apple silicon need a different
strategy and currently have no backend; `echora-guest` compiles there and
fails clearly if guest memory is created.

## Resolution model

`echora-resolution` distinguishes:

- **guest logical** — the size the game renders for;
- **guest view** — the guest-space region shown (wider than logical when the
  aspect policy expands the view, which also needs projection and UI
  adjustments in game-specific code);
- **scene** — the internal render size, chosen as match-output, a height, a
  fractional scale, or an exact size;
- **presentation** — where the scene is placed in the output surface;
- **output** — the swapchain size in physical pixels.

Guest render targets are classified as screen-relative (scaled by the exact
scene/view ratio per axis) or fixed (shadow maps, lookup tables). Viewports
and scissors map by rounding both edges, so tiled rectangles stay seamless at
non-integer scales. A change between plans reports whether it needs nothing,
a presentation update, or scene-target recreation.
