# Known limitations

- The game runs as far as the intro movie; gameplay has not been reached.
  Each unimplemented operation still stops the run with a report.
- **Echora is not ready for stable gameplay.** Runs can hang, crash or
  render incorrectly, and saves are untested.
- Runs are far below full speed: the game runs at 60 frames per second on
  the console, and the intro movie decodes here at about 20 pictures per
  second. By default every guest thread runs on one host thread, which does
  most of the work. `ECHORA_WORKERS` runs
  them on up to six host threads, one per application core, but those runs
  still crash within about half a minute.
- Of the game's own library functions, only the zlib and zstd wrappers are
  replaced with host code; its streaming zstd decoder (about an eighth of
  run time) still runs translated.
- Guest memory reaches host memory directly on Linux and Windows (10 1803 or
  later), which need 4 KiB host pages. macOS is not supported.
- On Windows, releasing part of a 2 MiB memory chunk zeroes it but keeps its
  storage committed until the whole chunk is released.
- A guest access to unmapped or protected memory ends the process (with the
  guest address reported) instead of returning a recoverable trap.
- NSO flag bit 6 (`0x40`) is set in `main` alongside the ZBIC bit; its meaning
  is unknown. It is accepted only together with ZBIC, and segment hashes
  confirm the decoded bytes. A ZBIC frame whose re-encoded tables no longer
  fit their size fields is rejected rather than decoded; this has not occurred
  with the real modules.
- Mesh and task pipelines (`DrawMeshTasks`) are drawn through
  `VK_EXT_mesh_shader` but not yet checked against the console's output;
  after the New Game movie the presented frames are still black. Of the
  6,074 shader stages OoT reached in a test run, 27 do not translate yet
  (gather forms, opcode c08 and register shadowing modes); draws that need
  them are skipped.
  NVN calls the Vulkan backend directly; a backend trait (for Direct3D
  12) is still to be introduced. Gamepads on Windows (gilrs) are untested
  on hardware, and audio output does not resample (the device must accept
  the guest's rate and channel count).
- Shader translation needs the fetched recompiler sources
  (`ECHORA_SHADER_SRC`); binaries built with them include AGPL-3.0 code
  (sirit).
- RomFS data from a NUP is read in place; raw files are not hash-checked on
  each read (`echora verify` checks every file), compressed files are
  verified when decoded.
- The ported NVN modules still use `unwrap()` after presence checks; the
  rest of the runtime states its invariants or recovers from poisoned locks.
- Update RomFS is not overlaid on base RomFS. The checked OoT input is a
  standalone update whose RomFS is used whole; its completeness is not
  verified against a base.
- Directory input hashes ExeFS modules but only counts RomFS files.
- The runtime applies window size and mode, internal render scale (measured
  against an assumed 1080-row guest output), volume and vibration from
  `echora-config`; `echora-resolution` is not yet connected to the renderer.
  View expansion (`AspectPolicy::Expand`) will need
  game-specific projection and UI changes before it can be offered.
- CI covers Linux x86-64 and Windows x86-64 compilation and tests only.
- The translation links as shared-library bundles (192 regions, about
  500 MB of code each). On Windows a Rust dylib may export at most 65,535
  symbols, which bundles of this size will exceed; building the
  translation on Windows will need smaller bundles or exports limited to
  each bundle's dispatch entry points. Only Linux builds of the translation
  have been run.
