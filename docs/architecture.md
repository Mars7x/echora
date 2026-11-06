# Architecture

Echora is a portable core with host-specific backends. The core never calls
operating-system, windowing or graphics APIs directly; those live in backend
crates selected per host.

## Crates

| Crate | Layer | Responsibility | State |
|---|---|---|---|
| `echora-nup` | core | NUP v1 reader: superblock/TOC validation, tree lookup, streaming zstd reads with SHA-256 checks, safe extraction, parallel verification. | Implemented, tested on synthetic and real input. |
| `echora-content` | core | One content model for NUP and dump directories: packages, versions, ExeFS module identification (NSO/NPDM headers), program-package selection, content-addressed ExeFS cache. | Implemented, tested. |
| `echora-loader` | core | NSO parsing with bounds checks, LZ4 and ZBIC segment decompression, segment hash verification, ExeFS load order and image budget. ZBIC frames are transcoded to standard Zstandard and decoded by libzstd. | MOD0 and dynamic-section analysis (symbols via SysV or GNU hash, RELA/REL/RELR/JMPREL tables, init/fini arrays, exported-symbol lookup), read-only. | Implemented, verified on the real modules. |
| `echora-resolution` | core | Resolution planning: guest logical size, guest view, internal scene size, presentation rectangle, output size; target/viewport mapping; change impact. | Implemented, tested. Not yet used by a renderer. |
| `echora-config` | core | Versioned TOML settings, validation, migration, requested vs effective settings, apply classes for live changes. | Implemented, tested. Not yet used by a runtime or launcher. |
| `echora` | front end | Command-line tool; owns host paths (cache location). | `inspect`, `verify`, `import`, `load`, `extract`. |

Planned crates, in roadmap order: `echora-lift` (AArch64 decoding, discovery,
IR, code generation), `echora-guest` (guest CPU state, memory, SIMD/FP
semantics shared with generated code), `echora-runtime` (scheduler, SVCs,
IPC, services), `echora-gpu` (portable graphics translation) with
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
