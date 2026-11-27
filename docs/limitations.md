# Known limitations

Echora is in development. Runs can hang, crash or render incorrectly.
Local Linux tests load a cloned save and reach Kokiri Forest. CI uses
synthetic fixtures and runs without the game files.

## Gameplay and graphics

- The game gets past the New Game movie into its first 3D scene, which is
  drawn with its ground cover, plants, fireflies, mist, canopy and the Great
  Deku Tree. The opening subtitles render, and full-opening tests at
  2160p show fog and scenery across the frame after fixes to render-target
  scaling. With Dynamic resolution disabled, a 1440p test used 2560×1440
  scene targets. The adaptive path used 854×480 targets in the sampled scene.
  A traced water pixel in the opening forest retained more than 99.99% of
  its base color at the blend before the volume passes in twelve live
  samples. In another live capture the same pixel chose the positive Z
  cubemap face in 16 successive samples, at mip levels 0.39–0.40. The
  cubemap returned more red than green each time. The final water color
  still differs from the matching console footage. Six consecutive 1440p
  captures of the opening forest (presents
  10,496–10,501) showed no black frame. After compensating for camera
  motion, no matched 40×40-pixel patch on the four interior frames had a
  brightness pulse above 12/255 relative to both neighbors. The check
  covered about 96% of the measured tiles in that shot. Six later live frames
  (10,580–10,585) showed fine shimmer on the foreground bank. In one
  motion-aligned patch, 3.56% of pixel samples reversed brightness by more
  than 20/255 across three frames. None did in the matching compressed
  console footage. No 4×4 block in either patch showed a reversal above
  12/255 after averaging its pixels.
  In another nine-frame sample, the dark foreground edge also appeared
  in the reference video. A broad-leaf clump on the lower bank was visible
  in the reference and missing in Echora at two camera positions. The
  adjacent rock aligned in both comparisons. The affected draw is under
  investigation.
  The missing floor has yet to be reproduced in these frames. Unimplemented
  operations can stop the run and produce a report.
- The reserved zero texture in a guest texture pool samples as transparent
  black and reports zero dimensions on hosts supporting Vulkan null
  descriptors. Two vertex programs previously skipped 35 depth draws
  when they used this slot. A Radeon run through present 10,600 logged
  zero draw errors after the fix. The effect on the visible gaps still
  needs comparison at the affected camera poses.
- Sampled views inherit the texture’s channel mapping until the guest sets
  an explicit mapping on the view. Rebinding a bindless texture slot clears
  image views left by the previous texture’s shape or numeric type. A Vulkan
  regression checks a float-to-integer replacement and the neighboring slot.
  A reset sampler builder uses linear minification and magnification, repeat
  addressing and LOD limits of 0 to 1000. The guest getters and registered
  Vulkan samplers share defaults verified against the guest SDK. Float and
  integer border-color getters read the same four words. Replaying a captured
  draw or dispatch retains the sampler settings visible to that submission.
  A queue released by an explicit guest wait can see registrations made
  while it was blocked.
- Min/max filtering uses the host's Vulkan sampler reduction support. On
  Mesa 25.2.8 llvmpipe, a test sampling 0.75 and 0.125 in adjacent mip
  levels at LOD 0.5 returned 0.4375 for minimum reduction. The expected
  value is 0.125. Cross-mip behavior on Radeon still needs verification.
  See [Vulkan mipmap filtering](https://docs.vulkan.org/spec/latest/chapters/textures.html#textures-texel-mip-filtering).
- Finalizing a texture removes its texel buffer and alternate 1D, 3D and
  array images from the lookup caches. Pending GPU commands retain their
  resources until completion. Vulkan tests cover both retirement and a
  reused handle reading new buffer contents. After a full flush, resources
  retired between submissions are freed even if no further work was recorded.
  Rendering into a compressed
  texture through an integer view loads and writes back the selected mip and
  layer. Color clears through a block view also preserve the scissor and
  channel mask. A Vulkan test checks that partial writes preserve neighboring
  blocks and that recycling the scratch attachment preserves each texture's
  contents. A clear on a compressed image without a compatible block view is
  reported as unsupported before the renderer records a Vulkan clear.
- Inactive targets can share a compressed host archive. Restoring or replacing
  the final target using one releases its backing allocation immediately.
  `ECHORA_CACHE_SIZES` reports Vulkan allocation bytes for render targets,
  uploads and images still held by command buffers. The effect on memory
  growth during the flight through Kokiri Forest still needs measurement.
- An instrumented Radeon run reached present 10,700 and exercised 128
  compressed texture writebacks. Forest entry allocated about 3.7 GiB
  of additional live Vulkan images between presents 10,200 and 10,300.
  Across the next four samples (10,400–10,700), targets and uploads stayed
  within 16 MiB of their initial total and held no retired image bytes
  at the sampling points. In a separate interactive run from a saved game,
  resident memory reached 15,078 MiB shortly after present 6,100. The test
  watchdog terminated the process at its 14,336 MiB limit. Its junction
  temperature was 38°C. The earlier black-screen failure remains open.
- Large single-plane depth textures stream through the upload ring. On a
  saved Kokiri Forest route, the 12,288×12,288 D16 shadow texture used
  bounded transfers for all 288 MiB. After this and the texture-cache
  lifetime fixes, peak resident memory through the matching 1440p capture
  at present 3,200 was 11,789 MiB, down from 12,290 MiB in the control.
  The patched headless run exited normally at present 6,200, peaked at
  12,594 MiB and averaged 14.0 FPS from presents 3,200 to 6,200. The
  scripted inputs cover a different path from the interactive session.
- Two subsequent saved-game runs at 2560×1440 also exited normally at
  present 6,200. The packed vertex and channel-mask build averaged 12.53 FPS
  over presents 3,200–6,200 and peaked at 12,885 MiB resident. Allowing GPU
  replay during CPU decompression measured 12.59 FPS and 12,369 MiB on the
  same scripted route. The two captures reached the same room at different
  animation phases. This pair establishes no repeatable FPS gain.
  Six-second CPU samples in both runs attributed 6.1–6.8% of process
  user-mode cycles to one translated guest search function. Those samples
  measure CPU execution across threads, separately from GPU frame time.
  In a further experiment the compiler optimized that one region more
  aggressively. The matched control and candidate measured 12.38 and
  12.56 FPS respectively over 3,000 presents. A single pair overlaps
  earlier run-to-run variation. The candidate continued through present
  8,200 with a peak resident size of 12,026 MiB and averaged 11.68 FPS
  over its final 2,000 presents.
- Mesh and task pipelines (`DrawMeshTasks`), occlusion queries and
  conditional rendering run on the GPU (`VK_EXT_mesh_shader`,
  `VK_EXT_conditional_rendering`), but have not yet been checked against the
  console's output. The guest's 3D render-enable state gates draws and
  clears. Compute dispatches still execute under a false render condition.
- NVN calls the Vulkan backend directly; a backend trait (for Direct3D 12) is
  still to come.
- The Rust shader translator translates all 6,215 shader stages the game
  has used in test runs. A few instruction forms no program has used yet
  are still refused, naming the reason
  ([shader-translator.md](shader-translator.md)).
- Vertex attribute arrays, vertex stream layouts, color state and channel
  masks are read from guest bytes when a command is recorded. Replaying a
  command preserves those values if the game has since overwritten the
  source. The renderer applies blend enables and RGBA write masks to up to
  eight color attachments. Nondefault alpha tests and logic operations
  still need renderer support.
- In a capture near the Great Deku Tree, 18 visible foliage fragments
  matched both the depth prepass and the color stored in the render target.
  On the middle bank the material retained about 96% of the sampled leaf
  luminance before lighting. The median after lighting was 6.7% of that
  value. Two upper-bank fragments became brighter. The lighting calculation
  still needs comparison with console output at the same fragments.
- Blending with two source colours (NVN's SRC1 factors) is not wired up in
  the renderer: such a target is drawn unblended, with a note. No program
  has asked for it so far.
- Samplers support transparent black, opaque black and opaque white borders
  for floating-point and integer textures. Other colours use an approximation.

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

- The console runs the game at 60 frames per second. A Radeon run with
  the candidate 12 translation and current renderer processed 9,660 movie
  pictures at 71.1 pictures/s. In the forest, presents 10,500–10,700
  took 8.8 seconds (22.7 FPS). The archived renderer took 7.8 seconds
  (25.6 FPS) with the same seven guest bundles and shared guest library.
  Both runs used the same save, fixed resolution, native render scale,
  audio and no screenshots. The current renderer logged zero skipped
  draws over the opening. The archived renderer logged 35. The cost of
  those draws within the measured window has not been isolated. A repeat
  with timing probes enabled reached 24.7 FPS on current and 24.5 FPS
  on the archive. The size of the difference varies between runs.
- A separate graphics test stage used an older six-bundle translation
  built without `target-cpu=native`. Its patched run reached 16.0 FPS
  in the same forest window, and its archived runner reached 21.1 FPS.
  These headless timings measure throughput before window scanout.
- In an earlier translation benchmark of the first 3D scene
  (the Great Deku Tree, presents 10,500 to 10,700 of a scripted run from a
  new game, Ryzen 7 5800X3D, Radeon RX 9070 XT, Mesa 26.2.7), the default
  translation ran at about 23.4 frames per second, and a translation with
  the opt-in structured-function and floating-point group options on the
  148 hottest regions at about 26.3 (November 2026). Earlier route tests at
  2560×1440 measured about 11 FPS on a saved forest route. Camera position,
  effects and scene loading change the workload.
- In that scene about 61% of CPU time is the game's own threads, mostly
  translated game code (window code 27%, structured functions 16%), and
  about 14% is the thread that replays the game's GPU work into Vulkan.
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
  updated translation on the next launch.
  SMAA is also off by default. Enable it under Game → Add-ons to use the
  game's edge blending. Sharpening is on by default. Turn it off under
  Game → Add-ons to zero the game's sharpening strength in both the
  fixed and adaptive NDSR resolves. The game's reconstruction and
  presentation still run. Both resolve shaders scale fixed filter weights
  by the sharpening strength. AMD's [FSR 1 RCAS filter](https://gpuopen.com/fidelityfx-superresolution/)
  calculates sharpening weights from nearby pixel values.
- Vertical sync uses tear-free mailbox presentation when available. Adaptive
  sync prefers relaxed FIFO. Disabling sync prefers immediate presentation,
  then mailbox. The fallback is FIFO.
