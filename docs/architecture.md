# Architecture

Echora is a portable core with host-specific backends. The core never calls
operating-system, windowing or graphics APIs directly; those live in backend
crates chosen for each host.

## Crates

| Crate | Role |
|---|---|
| `echora-nup` | Reads NUP containers in place: validates the table of contents, looks up files, streams compressed data with SHA-256 checks, and extracts safely. |
| `echora-content` | Gives NUP files and dump folders one content model: packages, versions, executable modules and the content-keyed ExeFS cache. |
| `echora-loader` | Parses and decompresses NSO modules (LZ4 and ZBIC), verifies their segment hashes, lays out the process and finds where code starts. |
| `echora-lift` | Decodes AArch64, discovers the game's code and emits it as Rust. Also holds the analyses the emitter uses: register use, spin loops, call targets and structured control flow. |
| `echora-guest` | The guest CPU state, guest memory and the `Core` that generated code runs against, with a memory backend for each host. |
| `echora-simd` | The vector and floating-point operations. Generated code reaches them through declarations in `echora-guest`, so changing one does not recompile the game's code. |
| `echora-runtime` | Boots the game and runs it: threads and scheduling, kernel calls, IPC and system services, file systems, input, audio, movies, and the NVN graphics layer. |
| `echora-gpu` | The Vulkan device, renderer and presentation, plus shader translation: the Rust translator (`echora-shader`) and an optional external plugin (docs/shader-plugin.md). |
| `echora-input` | Gamepads as a Switch controller: evdev on Linux, gilrs on Windows. |
| `echora-audio` | Plays the game's audio on the default output device through CPAL. |
| `echora-resolution` | Plans render and output sizes. Not yet connected to the renderer. |
| `echora-config` | Versioned settings with validation and migration, shared by the launcher and the runtime. |
| `echora-launcher` | The GPUI launcher: installs the game, edits settings, prepares the translation and starts the game. |
| `echora` | The command-line tool, including `prepare`, `lift` and `run`. |
| `echora-aot-tests` | Synthetic AArch64 programs lifted at build time and run natively, with semantic tests and benchmarks. |

The launcher depends on `echora-config` and `echora-content`, never the
other way round. Windowing is part of the runtime (winit, on its own thread);
input and audio are host crates of their own.

### Notes on individual crates

- **Launcher.** Before installing, it checks that the game is the supported
  build. The install is a copy-on-write clone (Btrfs, XFS) or a hard link
  where the disk allows, both instant and taking no extra space, and
  otherwise a copy after a free-space check. Either way the original can be
  moved or deleted afterwards. On first Play it runs `echora prepare`
  (fetching the shader recompiler, translating and building) with its
  progress shown. Every control works with mouse, keyboard and controller.
- **Runtime.** Guest threads run on one host thread by default, or on up to
  six workers that model the console's application cores (not yet stable).
  The game's zlib and zstd routines run as host code with identical results.
  Movies are AV1, decoded on the GPU's video engine (Vulkan Video) or with
  rav1d where the GPU cannot. Every unimplemented operation stops
  the run with a JSON report.
- **GPU.** Guest memory is imported into Vulkan where the driver allows
  (`VK_EXT_external_memory_host`); otherwise the game's memory pools are
  mirrored in device memory and copied only when the CPU touches them. Mesh
  and task pipelines are drawn with `VK_EXT_mesh_shader`. The
  `translate-dump` and `shader-check` tools translate dumped shader stages
  offline.
- **Generated code.** Three code-generation options can be turned on for a
  lift with `ECHORA_LIFT_OPTIONS` (`inline-vector`, `structured`,
  `fused-flags`). All are off by default until they are checked in game.

## Front ends

The launcher uses GPUI, the pure-Rust, GPU-rendered toolkit from Zed, so one
codebase serves Linux and Windows with its own look rather than a
desktop's native widgets. The cost is a young, fast-moving API with few
ready-made controls, so the launcher draws its own (buttons, segmented
choices). It draws client-side window controls on Linux compositors that
require them and keeps the native title bar on Windows. Settings are changed
only in the launcher; there is no in-game settings interface.

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
Windows are grouped into regions of `1 << region_bits` guest bytes, one crate
each (`gen_r_R`), so an unchanged region does not recompile. The build links
the regions into a few shared libraries ("bundles"), and only the dispatcher
crate sees every region. A guest BL runs its callee in nested host frames: `Core::call(return_to)`
looks the callee's window up in the process-wide window table and calls it
indirectly, repeating until the PC is back at `return_to`.

Calling a window function by name instead was tried and retired. In the
game's translation only about 13% of BL sites call within their own region
(126k against 853k sites, counted statically); the rest would need a call
slot per target filled at `install()`, which is still an indirect call. On
the benchmark's calls kernel the direct form was about 15% slower than
`Core::call`, and no change in emission could make the cross-region call
anything but indirect. Truly direct calls across regions would need
exported window symbols resolved by the loader (bundles would then depend
on each other, and Windows would need import libraries), grouping regions
by the call graph (rebuilds become less local), or one crate per bundle
(losing the per-region crates that keep small rebuilds cheap).

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
