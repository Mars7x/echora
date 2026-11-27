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
  In Link's room at present 3,500, the fixed NDSR resolve read a 2560×1440
  image and wrote a 2560×1440 image with a sharpening strength of 2.5.
  All 815 logged graphics draws had multisampling disabled. The guest also
  used smaller viewports for intermediate passes.
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
  A separate Radeon comparison captured six frames with and without SMAA
  at camera positions within two source pixels. With fixed 1440p rendering
  and sharpening enabled, neither sample showed a black flash. Motion-aligned
  4×4 brightness reversals above 12/255 fell from 2.11% to 1.80% with SMAA.
  At native pixel scale, foreground reversals above 20/255 changed from
  0.17% to 0.25%. These short samples show mixed behavior.
  In a six-present sharpening comparison at the same camera poses, three
  rock landmarks aligned. Screening for motion alignment in both runs
  left 89 foreground patches and 51,264 samples per run. Disabling
  sharpening reduced brightness reversals above 20/255 from 203 to 23 in
  those patches. The full images differed, including outside the measured
  foreground.
  A same-input Radeon test of the fixed 1440p resolve reproduced every
  texel of the live forest output on a separate render target. Changing
  only the sharpening strength from 2.5 to zero changed display luma
  by more than 20/255 in 160,498 pixels (4.4% of the frame). In nine
  matched patches, the median RMS luma gradient was 29.5 with the
  game's strength, 12.2 with zero strength, and 9.0 in compressed console
  footage. The test isolates sharpening for one frame. The compressed
  footage provides a reference for detail levels across those patches.
  In another nine-frame sample, the dark foreground edge also appeared
  in the reference video. A broad-leaf clump on the lower bank was visible
  in the reference and missing in Echora at two camera positions. The
  adjacent rock aligned in both comparisons. In one capture the paired
  foliage depth and material draws each emitted 48,415 triangles. Both
  covered nearby leaves, but none of their triangles covered the center
  of the missing clump. A separate trace of 33 indexed, alpha-tested
  depth draws recovered all 934 triangles. None intersected the 96×64
  patch around that clump before face culling. A camera-matched trace
  identified four plant instances in the patch. All four survived initial
  culling. Their generator then read an empty plant definition because the
  shader translator treated a negative constant-buffer displacement as
  positive. Correcting the address selected the four expected definitions
  and enabled leaf emission in a replay of the captured inputs. A Radeon
  test triggered by camera position then showed the missing clump across
  five consecutive frames. In a 96×64 patch matched to the same console
  frame, green pixels rose from 0.85% to 30.75%. The console measured
  78.91%. The restored leaves still render visibly darker.
  At that camera pose, 100 matched leaf-interior pixels had median
  display luma 22.1 in Echora and 49.1 in the console footage on an
  8-bit scale. The sampled bark, roots, and haze differed by at most
  four luma levels.
  The missing floor has yet to be reproduced in these frames. Unimplemented
  operations can stop the run and produce a report.
- The fairy beside the forest sign had a saturated red core in a player
  capture. A fragment trace found a nonfinite red value after a read from
  a varying the preceding stage had left unwritten. With corrected shader
  stage linkage, the median core color in an archived capture was
  RGB (235, 238, 229). The corresponding console frame measured
  (229, 231, 226), and the player capture measured (255, 5, 1).
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
  levels at LOD 0.5 returned 0.4375 for both minimum and maximum reduction.
  The same headless test on Radeon with the local RADV 26.2.7 build
  returned the expected 0.125 and 0.75 respectively. Both drivers passed
  the same-level min/max checks.
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
- A later build with renderer lock, GPU page-table and shader comparison
  changes averaged 13.11 FPS over saved-game presents 3,200–6,200 against
  12.38 FPS in the matched control. Peak resident memory was 12,488 MiB
  and 12,270 MiB respectively. Over opening forest presents 10,400–10,500,
  the same builds averaged 26.3 and 23.8 FPS. Both comparisons used the
  same translated guest libraries on Radeon. These single pairs do not
  isolate the effect of any one change.
  Moving sampler-builder setters off the renderer mutex then reduced the
  busiest guest thread's lock wait from 15.04 to 8.48 ms per frame in two
  18-frame traces. A 3,000-present run with that change averaged 12.21 FPS.
  The controls before and after it averaged 13.11 and 12.87 FPS, so the
  default uses the original sampler path. A separate route with scripted
  stick movement stayed inside Link's room through present 6,200, averaged
  10.55 FPS over its final 3,000 presents and peaked at 13,015 MiB resident.
- Workers record commands on each command buffer independently. Publishing
  the finished list takes the NVN lock shared with replay. A Radeon comparison
  measured 14.08 and 14.53 FPS with locked publication on either side of a
  13.12 FPS run that published under the command-list registry lock alone.
  Each run used the same saved room and translated game libraries. Across
  presents 4,200–5,200 at 2560×1440, all three averaged 721–723 draws
  per frame. The busiest guest thread executed 44.5–45.0 million
  instructions per frame with locked publication and 61.9 million with
  the registry lock alone. The reason
  the guest issued different amounts of work is still under investigation.
- On a route that opens the door and walks outside, 900 presents
  (3,800–4,700) averaged 8.19 FPS at `5b572a3`, 7.78 at `02514f3`
  and 8.63 at `cdf94fd`. The forest title appeared at the same checkpoints.
  The last build reached present 6,200, retired nine device buffers
  totaling 383 MiB, and peaked at 13,741 MiB process RSS. The longer
  run did not reproduce the reported interactive crash.
  A repeat at `bdb1c86` averaged 8.43 FPS in the same outdoor window,
  exited cleanly at present 6,200 and peaked at 13,289 MiB process RSS.
  Distinct compressed target archives grew from 225 MiB at present
  3,800 to 1,952 MiB at present 6,200. Restoring 416 archives took
  1.39 seconds of cumulative host time across the full run, including
  1.09 seconds in decompression.
  A run at `7dd318b` with an archive inventory written after the final
  present averaged 8.34 FPS in that window and peaked at 13,558 MiB.
  All 2,210 retained archives at present 6,200 had storage addresses
  inside the live guest heap. Their 1,928 distinct compressed payloads
  occupied 2,004 MiB.
  A profiling run at `caea754` averaged 8.39 FPS over presents
  3,800–4,700, compared with 8.32 at `d405cfa`. Replay-worker pieces
  fell from 854,018 to 799,988, and the worker's cumulative wait for
  the NVN lock fell from 28.64 to 24.59 seconds. Both runs exited cleanly
  and reached the outdoor title at the same checkpoints. This pair shows
  a small throughput difference across builds with several changes.
  At `8b9a219`, the same outdoor window averaged 8.58 FPS. Cumulative
  render-target index time dropped from 2.27 to 1.14 seconds, and
  full rebuilds fell from 1,518 to zero. The run exited cleanly at
  present 4,700 and the cloned save digest matched.
  In a separate profiled outdoor run at `2a1156e`, the host spent
  14.49 seconds feeding streaming decompression and 6 milliseconds
  decoding standalone frame requests. In a 12-present timeline, the
  busiest guest thread's NVN lock waits overlapped replay for 44.87 ms
  per present.
  Two same-executable comparisons ran streaming decompression under and
  outside the machine lock. Outdoor throughput improved from 8.06 to
  8.88 FPS, then from 8.34 to 8.93 FPS in reverse run order. The faster
  runs processed about 6–7% less compressed input. Aggregate host-call
  lock time in the 900-present window fell from 14.6 seconds to 6.0 seconds.
  With parallel decoding enabled by default, another run averaged
  9.22 FPS over the same window, peaked at 13,675 MiB resident and reached
  the same forest checkpoints with an unchanged cloned save. A guarded
  opening run exited at present 10,586 and peaked at 10,123 MiB resident.
  The startup movie reported 75.5 pictures per second, followed by 24.1
  presents per second in the gameplay sample ending at present 10,500.
  A private trial used the same executable for two guarded runs with
  plain and losslessly filtered archives. At present 6,200 the filter
  reduced retained payloads from 1,812 to 1,003 MiB. Peak process RSS
  was 13,268 and 13,327 MiB respectively, with 21.67 and 23.03 seconds
  of archive encoding CPU time. The filtered run averaged 7.83 FPS
  outdoors compared with 8.60 FPS for the plain run. One 100-present
  interval accounted for the slowdown. Both runs exited cleanly.
- Releasing CPU copies after their texture uploads reached the GPU reduced
  retained CPU texels from 138 to 5 MiB at saved-room presents 3,200 and
  6,100. With native blend state and corrected primitive restart, the room
  averaged 11.62 FPS over presents 3,200–6,200 and peaked at 12,511 MiB.
  A repeat with the same executable averaged 12.00 FPS and peaked at
  12,258 MiB. In the two runs the renderer allocated 1,592 and 1,397
  page-table buffers. Opening forest throughput ranged from 25.6 to 26.3
  FPS over presents 10,400–10,500. The hardware effect of reusing page-table
  buffers across extent changes still needs a matched build comparison.
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
  eight color attachments. On Vulkan devices with logic operations, an
  active operation replaces blending and combines the source and destination
  bits of integer and normalized fixed-point attachments. Floating-point
  and sRGB attachments pass through source values. Nondefault alpha tests
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

- With the same Radeon build and saved forest route at 2560×1440, retaining
  a bounded GPU page-table extent averaged 9.66 FPS over presents
  4,200–5,100, compared with 8.90 FPS with exact resizing. The renderer
  allocated 61 page-table buffers instead of 680. Both runs reached the
  forest title, completed normally and preserved the cloned checkpoint.
  A later trial remapped completed page tables in place 753 times. Outdoor
  throughput across presents 4,600–5,100 measured 9.84 FPS versus 9.42 FPS
  in the control. Across the 900-present profiling window, guest instruction
  count fell by about 3% and page-table allocations rose from 19 to 129.
  Peak resident memory for the full runs rose from 13,343 to 13,854 MiB.
  The trial was kept local.
- Reading texture-builder storage properties through the shared builder
  state avoided 80,805 calls under the NVN lock in a 900-present saved-forest
  run. Aggregate guest NVN lock wait fell from 85.5 to 82.1 seconds.
  The matched pair averaged 8.82 and 9.05 FPS across presents 4,200–5,100
  and peaked at 13,639 and 13,780 MiB resident. Both runs reached the same
  forest checkpoints and preserved the cloned save.
- Indexing storage buffers by stage and slot reduced driver constant-table
  preparation from 2.03 to 0.43 ms per present in a 12-present profile.
  The paired 1440p saved-forest runs averaged 9.11 and 9.23 FPS over
  presents 4,200–5,100 and 10.00 and 10.50 FPS over 4,600–5,100.
  The candidate issued about 2% fewer draws over the longer interval.
  Both runs reached the same outdoor checkpoints and preserved the save.
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
