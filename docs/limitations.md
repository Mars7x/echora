# Known limitations

Echora is not ready for stable gameplay. Runs can hang, crash or render
incorrectly, and saves are untested. The points below describe what is known
to be missing or incomplete.

## Gameplay and graphics

- The game gets past the New Game movie and into the first scene's code, but
  no gameplay scene has been shown yet. Every unimplemented operation still
  stops the run with a report.
- Mesh and task pipelines (`DrawMeshTasks`) are drawn through
  `VK_EXT_mesh_shader`, but have not yet been checked against the console's
  output.
- Of the 6,074 shader stages the game has reached so far, 16 do not translate
  yet: 15 texture gathers whose offsets are not provably constant, and one
  return linkage mode. Draws that need them are skipped.
- NVN calls the Vulkan backend directly; a backend trait (for Direct3D 12) is
  still to come.
- Shader translation needs the fetched recompiler sources
  (`ECHORA_SHADER_SRC`). Binaries built with them include AGPL-3.0 code from
  sirit.

## Performance

- Runs are far below full speed. The console runs the game at 60 frames per
  second; here the opening movie decodes at about 20 pictures per second.
- By default every guest thread runs on one host thread, which does most of
  the work. `ECHORA_WORKERS` spreads them over up to six host threads, one per
  application core, but those runs still crash within about half a minute.
- The faster code-generation options (`inline-vector`, `direct-calls`,
  `structured`) are off by default until they are checked in game.

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
