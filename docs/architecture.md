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
| `echora-guest` | core + host backend | Guest CPU state, guest memory (mappings, permissions, TLB, exclusive monitor), SIMD/FP semantics, traps, and the `Core` that generated code runs against. Guest memory has a per-host backend (`host_memory/`): Linux maps one memfd twice, Windows maps pagefile-backed sections into placeholder reservations, so generated code accesses guest memory directly; macOS translates each access through a per-guest-page table. | Ported from Pokopia. Linux, Windows and macOS backends. |
| `echora-lift` | core | AArch64 decoding (scalar and SIMD/FP), code discovery from entries, prologues, relocation targets and jump tables, basic blocks, and Rust emission. Generated code depends only on `echora-guest` and exposes `dispatch(&mut Core)`. | Ported from Pokopia; 53 tests. |
| `echora-aot-tests` | test | Synthetic AArch64 programs lifted by `build.rs` and run natively on a minimal host loop (integer, flags, calls, PAuth hints, budgets, window dispatch, missing blocks, CAS/LDADD/exclusives, acquire/release). | Execution tests (Linux, Windows and macOS). |
| `echora-runtime` | core (+ host stand-ins) | Process bootstrap (module mapping, stack, TLS, entry registers, NPDM capabilities including the private kernel resource pool), scheduler and threads, synchronization, kernel SVCs, IPC framing and the system services, file systems and RomFS images, HID shared memory, and high-level NVN (objects, command buffers, execution on `echora-gpu`, presentation to a winit window with `ECHORA_WINDOW=1`). Gamepads come from `echora-input`; audio plays through PipeWire's `pw-cat` on Linux with `ECHORA_AUDIO=1`. | Ported from Pokopia; tests on Linux, Windows and macOS. Boots the real game's process; not yet run. NVN calls the Vulkan backend directly rather than through a backend trait. |
| `echora-gpu` | host backend | Vulkan device, renderer, draws, compute, presentation, and shader translation through the PokeDinner recompiler (C ABI shim built by `build.rs` when `ECHORA_SHADER_SRC` is set). | Ported from Pokopia; 9 tests. Used by the runtime's NVN layer. |
| `echora-input` | host backend | Gamepads as a Switch controller state behind a platform-neutral interface: Linux reads evdev devices (hotplug, axis ranges, rumble); other platforms report no pads. | Ported from Pokopia; 4 tests. |
| `echora-resolution` | core | Resolution planning: guest logical size, guest view, internal scene size, presentation rectangle, output size; target/viewport mapping; change impact. | Implemented, tested. Not yet used by a renderer. |
| `echora-config` | core | Versioned TOML settings, validation, migration, requested vs effective settings, apply classes for live changes. | Implemented, tested. Not yet used by a runtime or launcher. |
| `echora` | front end | Command-line tool; owns host paths (cache location). | `inspect`, `verify`, `import`, `load`, `analyze`, `lift`, `extract`. |

Planned crates, in roadmap order: the NVN layer in the runtime on top of
`echora-gpu`, and host crates for windowing, input and audio. The
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
address-space-sized file would fault in every page.

The Windows backend (Windows 10 1803 or later) splits the memory file into
2 MiB pagefile-backed sections, created on first write or guest mapping, so
commit charge follows memory in use. Both views are placeholder reservations;
replacing a placeholder with `MapViewOfFile3` maps views at page rather than
64 KiB granularity. Changing part of a view unmaps it whole and remaps the
untouched remainders. Section pages cannot be decommitted, so releasing a
whole chunk drops its section while partial releases are zeroed in place. A
vectored exception handler reports forbidden guest accesses as on Linux.

Apple silicon's 16 KiB pages cannot give each 4 KiB guest page its own
backing and permissions, so macOS uses the checked backend: a translation
table holds, per guest page, the host address of its backing page plus
READ/WRITE bits. Each access looks its page up and checks the permission;
a forbidden access prints the same report and aborts. Accesses that cross a
guest page are split, since neighbouring guest pages need not have
neighbouring backing. Linux and Windows reach memory directly; on macOS every
access pays the lookup. `--cfg echora_checked_memory` selects this backend on
Linux for testing.

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
