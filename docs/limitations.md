# Known limitations

Echora is not ready for stable gameplay. Runs can hang, crash or render
incorrectly, and saves are untested. The points below describe what is known
to be missing or incomplete.

## Gameplay and graphics

- The game gets past the New Game movie and draws its first 3D scene, but
  the scene is heavily overexposed and no gameplay has been reached yet.
  Every unimplemented operation still stops the run with a report.
- Mesh and task pipelines (`DrawMeshTasks`) are drawn through
  `VK_EXT_mesh_shader`, but have not yet been checked against the console's
  output.
- NVN calls the Vulkan backend directly; a backend trait (for Direct3D 12) is
  still to come.
- The Rust shader translator does not translate programs yet. Without an
  external translator plugin ([shader-plugin.md](shader-plugin.md)) the
  run stops at the first program it does not handle, naming the reason.

## Performance

- Runs are far below full speed. The console runs the game at 60 frames per
  second; here the opening movie plays at about 22 to 29 pictures per second
  and the first 3D scene at about 6 frames per second.
- Guest threads run on up to six host threads by default, one per
  application core (`ECHORA_WORKERS`, or the launcher's CPU cores setting).
  On six cores the first 3D scene runs about 2.9 times as fast as on one, and
  three cores get most of that. More cores help little for now, because the
  thread that turns the game's GPU commands into Vulkan works alone.
- The faster code-generation options (`inline-vector`, `structured`,
  `fused-flags`) are off by default until they are checked in game.

## Platforms

- The game's translation has only been built and run on Linux. CI builds and
  tests everything else on Linux x86-64 and Windows x86-64.
- On Windows a Rust dylib may export at most 65,535 symbols, which the
  translation's shared-library bundles (192 regions, about 500 MB of code
  each) will exceed. Building the translation there will need smaller bundles
  or fewer exports.
- Guest memory maps directly into host memory on Linux and Windows (10 1803
  or later), which needs 4 KiB host pages. On Windows, releasing part of a
  2 MiB memory chunk zeroes it but keeps its storage committed until the whole
  chunk is released.
- Gamepads on Windows (gilrs) have not been tested on hardware. Audio output
  does not resample, so the device must accept the game's rate and channel
  count.

## Content input

- Update RomFS is not overlaid on base RomFS. The supported input is a
  standalone update whose RomFS is used whole; its completeness is not checked
  against a base.
- RomFS data from a NUP is read in place. Raw files are not hash-checked on
  every read (`echora verify` checks every file); compressed files are checked
  when decoded.
- Dump folders hash their ExeFS modules but only count RomFS files.
- NSO flag bit 6 (`0x40`) is set in `main` alongside the ZBIC bit, and its
  meaning is unknown. It is accepted only together with ZBIC, and segment
  hashes confirm the decoded bytes. A ZBIC frame whose re-encoded tables no
  longer fit their size fields would be rejected rather than decoded; this has
  not happened with the real modules.

## Runtime

- A guest access to unmapped or protected memory ends the process, with the
  guest address reported, instead of raising a recoverable trap.
- The ported NVN modules still use `unwrap()` after presence checks; the rest
  of the runtime states its invariants or recovers from poisoned locks.
- The runtime applies window size and mode, internal render scale (measured
  against an assumed 1080-row guest output), volume and vibration from the
  settings. `echora-resolution` is not yet connected to the renderer, and
  widening the view (`AspectPolicy::Expand`) would need game-specific camera
  and interface changes.
