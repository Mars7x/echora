# Known limitations

Echora is not ready for stable gameplay. Runs can hang, crash or render
incorrectly, and saves are untested. The points below describe what is known
to be missing or incomplete.

## Gameplay and graphics

- The game gets past the New Game movie into its first 3D scene, which is
  drawn with its ground cover, plants, fireflies, mist, canopy and the Great
  Deku Tree. The opening subtitles render, and full-opening tests at
  2160p show fog and scenery across the frame after fixes to render-target
  scaling. In one sampled forest frame, the 854×480 guest scene targets
  were rendered at 1281×720 before the final 3840×2160 output. The water
  is too red.
  No gameplay has been reached yet. Every unimplemented operation still
  stops the run with a report.
- Mesh and task pipelines (`DrawMeshTasks`), occlusion queries and
  conditional rendering run on the GPU (`VK_EXT_mesh_shader`,
  `VK_EXT_conditional_rendering`), but have not yet been checked against the
  console's output.
- NVN calls the Vulkan backend directly; a backend trait (for Direct3D 12) is
  still to come.
- The Rust shader translator translates all 6,215 shader stages the game
  has used in test runs. A few instruction forms no program has used yet
  are still refused, naming the reason
  ([shader-translator.md](shader-translator.md)).
- Blending with two source colours (NVN's SRC1 factors) is not wired up in
  the renderer: such a target is drawn unblended, with a note. No program
  has asked for it so far.
- A sampler's border colour is mapped to the nearest of Vulkan's built-in
  ones (transparent black, opaque black, opaque white).

## Sound

- The game's audio renderer is mixed and played: wave buffers (PCM16, float,
  ADPCM) are decoded and resampled, voices and mixes are routed through
  splitters, and reverb, I3DL2 reverb and aux sends run on their mixes. Output
  is opt-in from the command line (`ECHORA_AUDIO=1`); the launcher turns it
  on. Depop, the performance counters, and the delay, biquad filter,
  limiter, capture and compressor effects are not implemented yet.
- On Linux, PipeWire handles sample-rate conversion and mixes six channels
  down to stereo for stereo outputs. Windows playback requires a device
  that supports the game's sample rate and mixes six channels down to
  stereo where the device cannot play six.
- The guest sometimes sets voice gains to zero during the opening. In an
  isolated 2160p Wayland run, the mixer and host recording stayed aligned
  across sampled movie and forest audio. Display-to-speaker synchronization
  still needs direct measurement.

## Performance

- The console runs the game at 60 frames per second. The opening movie
  reaches that speed in Echora. Recent full-opening runs on a Ryzen 7
  5800X3D and Radeon RX 9070 XT averaged about 20 FPS over 600 frames
  in the first 3D scene at both 2560×1440 and 3840×2160. A longer 2160p
  Wayland run averaged about 16 FPS over 1,600 frames deeper into the forest.
  The camera and draw workload can vary between runs.
- Guest threads run on one host thread per CPU by default, at most six, one
  per application core (`ECHORA_WORKERS`, or the launcher's CPU cores setting).
  The game's submitted GPU work is replayed into Vulkan on a thread of its
  own (`ECHORA_REPLAY_INLINE=1` replays it on the submitting thread, as
  before). In a 600-frame forest profile, the Vulkan batches occupied
  about a third of wall time. CPU work includes guest simulation, command
  recording, and preparing texture bindings for replay.
- The CLI enables the measured vector, floating-point and block
  optimizations by default. Structured control flow and several newer
  [code generation options](codegen-speed.md) are still experimental.

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
- Gamepads on Windows (gilrs) have not been tested on hardware.

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
- The runtime applies window size and mode, internal render scale, volume
  and vibration from the settings. Absolute resolution uses the first guest
  window texture's height when available at renderer startup, with a
  1080-row fallback. Match Output uses the host window's physical height at
  startup. `echora-resolution` is not yet connected to the renderer, and
  widening the view (`AspectPolicy::Expand`) would need game-specific camera
  and interface changes.
- Ocarina of Time's Dynamic resolution add-on is off by default and uses
  the game's fixed-resolution path. Enable it under Game → Add-ons to
  restore the game's adaptive 3D resolution. The launcher prepares the
  updated translation on the next launch. The build and branch match the
  [Lichii Max Resolution add-on](https://github.com/zortos293/Lichii/blob/17baed7/src/core/file_sys/patch_manager.cpp).
- Vertical sync uses tear-free mailbox presentation when available. Adaptive
  sync prefers relaxed FIFO. Disabling sync prefers immediate presentation,
  then mailbox. The fallback is FIFO.
