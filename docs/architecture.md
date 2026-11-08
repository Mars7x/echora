# Architecture

Echora is a portable core with host-specific backends. The core never calls
operating-system, windowing or graphics APIs directly; those live in backend
crates selected per host.

## Crates

| Crate | Layer | Responsibility | State |
|---|---|---|---|
| `echora-launcher` | front end | GPUI launcher (Linux, Windows): checks that the chosen NUP or dump is the supported build, then installs it into Echora's data folder as a copy-on-write clone (Btrfs, XFS) or a hard link where the disk allows, both instant and taking no extra space, else as a copy after a free-space check (staged, swapped in when complete) so the original is no longer needed; the game's artwork fills the window, with a dock holding the title, version, Play, Settings and About; a tabbed settings sheet (Game: the installed game, with Install or Reinstall; Display: window mode and internal resolution; Controls: vibration; Launcher: skip the launcher and start the last game directly, with `--show` to reopen it) and an About sheet with credits, saved to the shared settings file. Play prepares the game the first time (`echora prepare`: fetching the shader recompiler, translating and building, with its progress shown) and then launches it through `echora run`. Every control is reached by mouse, keyboard (Tab, arrows, Enter) and controller (D-pad or stick, A, B, +, L/R) with one focus ring that moves between controls; pop-up menus and sheets keep focus inside them. | Implemented; checked on Linux under headless sway with lavapipe, including a virtual gamepad. Not yet run on Windows. Starts the game with `ECHORA_WINDOW` and `ECHORA_AUDIO` on unless its environment sets them. |
| `echora-nup` | core | NUP v1 reader: superblock/TOC validation, tree lookup, streaming zstd reads with SHA-256 checks, safe extraction, parallel verification. | Implemented, tested on synthetic and real input. |
| `echora-content` | core | One content model for NUP and dump directories: packages, versions, ExeFS module identification (NSO/NPDM headers), program-package selection, content-addressed ExeFS cache. | Implemented, tested. |
| `echora-loader` | core | NSO parsing with bounds checks, LZ4 and ZBIC segment decompression, segment hash verification, ExeFS load order and image budget, process layout and discovery roots (prologues, RELA/RELR relative targets, `init_array`). ZBIC frames are transcoded to standard Zstandard and decoded by libzstd. MOD0 and dynamic-section analysis (symbols via SysV or GNU hash, RELA/REL/RELR/JMPREL tables, init/fini arrays, exported-symbol lookup), read-only. | Implemented, verified on the real modules. |
| `echora-guest` | core + host backend | Guest CPU state, guest memory (mappings, permissions, TLB, exclusive monitor), SIMD/FP semantics, traps, and the `Core` that generated code runs against. Guest memory has a per-host backend (`host_memory/`): Linux maps one memfd twice, Windows maps pagefile-backed sections into placeholder reservations, so generated code accesses guest memory directly. | Linux and Windows backends. |
| `echora-simd` | core | The SIMD/FP operations themselves. `echora_guest::simd` keeps their types and declares each one; generated code calls those declarations, which reach this crate's exported symbols through the shared guest library, so changing an operation does not recompile the regions. | Their tests. |
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

### Calls between windows

Split emission puts each 1 KiB guest window in a function `run_window_W`.
Windows are grouped into regions of `1 << region_bits` guest bytes, one
crate each (`gen_r_R`), so an unchanged region does not recompile. The
build links the regions into a few shared libraries ("bundles"), and only
the dispatcher crate sees every region. A guest BL runs its callee in
nested host frames:

- `Core::call(return_to)` looks the callee's window up in the
  process-wide window table (`install_windows`): a shift, a subtraction, a
  bounds check, a load of an `Option<WindowFn>` and an indirect call. Then
  it loops the same way until the PC is back at `return_to`.
- With `EmitOptions::direct_calls`, a BL whose target is in the caller's
  own region calls `Core::call_window(return_to, run_window_W)` instead.
  The first window is a direct call to a function the crate can name, and
  the rest is as in `Core::call`.
- A BL to a block in another region goes through a call slot (option 2
  below). The region source holds `pub static CALLS:
  [OnceLock<WindowFn>; N]`, one slot per window it calls in other
  regions. The root's generated `install()` fills every slot, so no build
  change is needed. Until then, the call falls back to `Core::call`.
  BLRs and calls to addresses with no translated block keep `Core::call`.

What a direct call avoids is the table lookup for the callee's first
window, plus an indirect call whose target the compiler cannot see. The
table holds 8 bytes per window: for tens of megabytes of guest code, that
is hundreds of kilobytes indexed by callee address, so a cold call is
likely to miss L1 there. On the benchmark's calls kernel (two BLs to
two-instruction leaves per four-instruction iteration, all in one region),
direct calls give +7% through a stand-in for `call_window`. In real code,
the gain scales with the share of BLs that cross a window boundary and
reach a cold table entry. A count of `echora_lift::structure::call_sites`
over a translation, split by whether the target shares the caller's
region, would size what cross-region direct calls could add before any
build work. Calls within a window need no lookup in either case.

Calling across regions directly could be done four ways. Each changes the
build in `crates/echora/src/main.rs`, and none is implemented:

1. **Exported window symbols.** Each window function gets a stable
   unmangled name (`#[unsafe(no_mangle)]`, e.g. `echora_window_7105_75cc`
   from its window index), and a calling region declares the callees it
   needs in an `unsafe extern "Rust"` block. Names come from guest
   addresses, so they stay stable across rebuilds: a caller region does
   not recompile when the callee's region changes. Every BL target is a
   block of the same translation, so the symbol exists. The cost is in
   linking. Bundles would reference one another, possibly in both
   directions, so there is no Cargo dependency order. On Linux an ELF
   shared library may keep undefined symbols, which the loader resolves
   from libraries already loaded (they must load with global visibility,
   and lazy binding must not leave a missing name to fail at first call).
   On Windows a DLL must import each symbol from a named DLL at link time,
   which needs import libraries made from `.def` files before the bundles
   link, or one image per process.
2. **A slot table per region.** Each region crate holds a static array of
   function pointers, one per distinct out-of-region callee, which the
   dispatcher fills when it installs the window table. A call loads its
   slot (a constant offset in the caller's own crate, close to its code)
   and calls through it. It needs no cross-library symbols and works the
   same on every host. The call stays indirect, but the lookup is one
   load from a small, local table instead of a computed index into the
   process-wide one, and each call site's branch target is stable.
3. **Regions grouped by the call graph.** Assign windows to regions so
   that hot callers and callees share one, and direct calls stay within a
   region. This keeps today's linking, but region contents then depend on
   the call graph, not only on addresses. One changed function can move
   windows between regions and rebuild more than its own region.
4. **One crate per bundle.** Make each bundle a single crate with the
   regions as modules, so any window in the bundle can be named. Calls
   between bundles still go through the table. This gives up the
   per-region crate that makes rebuilds after a small change cheap.

Option 2 is implemented, behind `EmitOptions::direct_calls`. The slots
add about 1.4 bytes of Rust per guest instruction on the aot-tests
programs. In the game build, before slots, 125,929 static call sites
were direct (same region) and 853,320 went through `Core::call`. Option
1 would give true direct calls, but depends on the host's loader.

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
