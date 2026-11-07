# Architecture

Echora is a portable core with host-specific backends. The core never calls
operating-system, windowing or graphics APIs directly; those live in backend
crates selected per host.

## Crates

| Crate | Layer | Responsibility | State |
|---|---|---|---|
| `echora-launcher` | front end | GPUI launcher (Linux, Windows): installs the chosen NUP or dump into Echora's data folder (staged copy with progress, swapped in when complete) so the original is no longer needed; the game's artwork fills the window, with a dock holding the title, version, Play, Settings and About; a tabbed settings sheet (Game: the installed game, with Install or Reinstall; Display: window mode and internal resolution; Controls: vibration; Launcher: skip the launcher and start the last game directly, with `--show` to reopen it) and an About sheet with credits, saved to the shared settings file. Play prepares the game the first time (`echora prepare`: fetching the shader recompiler, translating and building, with its progress shown) and then launches it through `echora run`. Every control is reached by mouse, keyboard (Tab, arrows, Enter) and controller (D-pad or stick, A, B, +, L/R) with one focus ring that moves between controls; pop-up menus and sheets keep focus inside them. | Implemented; checked on Linux under headless sway with lavapipe, including a virtual gamepad. Not yet run on Windows. Starts the game with `ECHORA_WINDOW` and `ECHORA_AUDIO` on unless its environment sets them. |
| `echora-nup` | core | NUP v1 reader: superblock/TOC validation, tree lookup, streaming zstd reads with SHA-256 checks, safe extraction, parallel verification. | Implemented, tested on synthetic and real input. |
| `echora-content` | core | One content model for NUP and dump directories: packages, versions, ExeFS module identification (NSO/NPDM headers), program-package selection, content-addressed ExeFS cache. | Implemented, tested. |
| `echora-loader` | core | NSO parsing with bounds checks, LZ4 and ZBIC segment decompression, segment hash verification, ExeFS load order and image budget, process layout and discovery roots (prologues, RELA/RELR relative targets, `init_array`). ZBIC frames are transcoded to standard Zstandard and decoded by libzstd. MOD0 and dynamic-section analysis (symbols via SysV or GNU hash, RELA/REL/RELR/JMPREL tables, init/fini arrays, exported-symbol lookup), read-only. | Implemented, verified on the real modules. |
| `echora-guest` | core + host backend | Guest CPU state, guest memory (mappings, permissions, TLB, exclusive monitor), SIMD/FP semantics, traps, and the `Core` that generated code runs against. Guest memory has a per-host backend (`host_memory/`): Linux maps one memfd twice, Windows maps pagefile-backed sections into placeholder reservations, so generated code accesses guest memory directly. | Linux and Windows backends. |
| `echora-lift` | core | AArch64 decoding (scalar and SIMD/FP), code discovery from entries, prologues, relocation targets and jump tables, basic blocks, and Rust emission. Generated code depends only on `echora-guest` and exposes `dispatch(&mut Core)`. | 62 tests. Translates the whole game (17.3M instructions, 1,111 region crates) on Linux. |
| `echora-aot-tests` | test | Synthetic AArch64 programs lifted by `build.rs` and run natively on a minimal host loop (integer, flags, calls, PAuth hints, budgets, window dispatch, missing blocks, CAS/LDADD/exclusives, acquire/release). | Execution tests (Linux and Windows). |
| `echora-runtime` | core (+ host stand-ins) | Process bootstrap (module mapping, stack, TLS, entry registers, NPDM capabilities including the private kernel resource pool), scheduler and threads, synchronization, kernel SVCs, IPC framing and the system services, file systems and RomFS images, HID shared memory, and high-level NVN (objects, command buffers, execution on `echora-gpu`, presentation to a winit window with `ECHORA_WINDOW=1`). Gamepads come from `echora-input`; audio plays through `echora-audio` with `ECHORA_AUDIO=1`. | 265 tests on Linux and Windows. Runs the translated game on Linux through the title and intro movie (AV1 decoded with rav1d on up to four threads); guest threads run on one host thread by default, or on up to six workers modelling the console's application cores (not yet stable); the game's zlib and zstd wrappers are replaced by host code; stops with a JSON report at the first unimplemented operation; can save presented frames as PNGs (`ECHORA_CAPTURE_EVERY`). NVN calls the Vulkan backend directly rather than through a backend trait. |
| `echora-gpu` | host backend | Vulkan device, renderer, draws, compute, presentation, and shader translation through the Lichii recompiler (C ABI shim built by `build.rs` when `ECHORA_SHADER_SRC` is set). | 9 tests. Used by the runtime's NVN layer. Compute storage is bound over guest memory imported into Vulkan (`VK_EXT_external_memory_host`) where the driver allows, otherwise kept in device buffers that mirror the game's memory pools and are copied only when the CPU side touches them. `translate-dump` and `shader-check` translate dumped shader stages (`ECHORA_SHADER_DIR`) offline and compile them on the driver. |
| `echora-input` | host backend | Gamepads as a Switch controller state behind a platform-neutral interface: Linux reads evdev devices (hotplug, axis ranges, rumble); Windows uses gilrs with the same layout, dead zone and rumble bands. | 4 tests. gilrs backend compile-checked for Windows, not yet run on hardware. |
| `echora-resolution` | core | Resolution planning: guest logical size, guest view, internal scene size, presentation rectangle, output size; target/viewport mapping; change impact. | Implemented, tested. Not yet connected to the renderer (the runtime applies an internal render scale itself). |
| `echora-audio` | host backend | The guest's PCM16 audio-out streams on the default output device through CPAL (ALSA/PipeWire, WASAPI), fed through a bounded queue by a stream-owning thread. | 2 tests; plays on the development machine. |
| `echora-config` | core | Versioned TOML settings, validation, migration, requested vs effective settings, apply classes for live changes. | Implemented, tested. Edited by the launcher; the runtime applies window size and mode, internal resolution, volume and vibration. |
| `echora` | front end | Command-line tool; owns host paths (cache location). | `inspect`, `verify`, `import`, `load`, `analyze`, `boot`, `lift`, `prepare` (fetch, lift and build in one step, reporting progress for the launcher), `run`, `extract`. |

The launcher depends on `echora-config` and `echora-content`, never the
reverse. Windowing is a module of the runtime (winit, on its own thread);
input and audio are their own host crates.

## Front ends

The launcher uses GPUI, the pure-Rust, GPU-rendered toolkit from Zed, so one
codebase serves Linux and Windows with its own look rather than a
desktop's native widgets. The cost is a young, fast-moving API with few
ready-made controls, so the launcher draws its own (buttons, segmented
choices). It draws client-side window controls on Linux compositors that
require them and keeps the native title bar on Windows. Settings are changed only in the launcher; there is no
in-game settings interface.

GPUI 0.2.2 reaches `xattr` 0.2.3 through its HTTP client, and that `xattr`
needs `libc::ENOATTR`, which `libc` 0.2.190 removed on Linux. `Cargo.lock`
therefore pins `libc` 0.2.189 until GPUI moves to a fixed `xattr`.

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

## Execution model

The model has four parts, all in use with the real game on Linux:

1. Load `rtld`, `main`, `subsdk*`, `sdk` at deterministic bases; the guest's
   own rtld performs relocation.
2. Discover code from function prologues, relocation targets and jump
   tables; translate ahead of time; report every unresolved indirect target
   as a deterministic missing-block stop.
3. Emit generated code that the host toolchain compiles natively. Echora
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
