# Development

Notes for working on Echora from the command line. To build Echora and start
the game, see the [README](../README.md#building).

## Command line

The launcher's steps are commands of their own:

```sh
target/release/echora prepare /path/to/game.nup   # translate and build (-j N, --force, --check)
target/release/echora inspect /path/to/game.nup   # packages, versions, build ID
target/release/echora verify /path/to/game.nup    # hash-check every file
target/release/echora import /path/to/game.nup    # validate and cache the program's ExeFS
target/release/echora load /path/to/game.nup      # decompress and verify every module
target/release/echora analyze /path/to/game.nup --limit 20000000   # code discovery (~6 GB RAM)
target/release/echora boot /path/to/game.nup      # create the game process without running it
target/release/echora lift /path/to/game.nup      # only translate (several GiB of Rust)
target/release/echora extract /path/to/game.nup out --subtree romfs
target/release/echora shaders /path/to/game.nup   # count every GPU program the game ships (--dump DIR writes them)
```

See [content input](content.md) for the formats Echora reads.

`prepare` reuses a runner after this build of Echora stamps it. It
rebuilds unstamped runners so the launcher picks up changes to generated
code. To manage a translation by hand, use `echora lift` and then
`cargo build --release -p echora-run` in the translation folder it prints.

`echora run` starts the built translation directly, which is how the game
is tested:

```sh
ECHORA_WINDOW=1 ECHORA_AUDIO=1 target/release/echora run /path/to/game.nup
```

Without `ECHORA_WINDOW=1` it runs headless (still drawing on the GPU);
without `ECHORA_AUDIO=1` it is silent. It stops after `--budget` guest
instructions, a billion by default (a few seconds of play), so give longer
test runs a larger budget; the launcher passes no limit. Settings are read
from `~/.config/echora/settings.toml` (`ECHORA_SETTINGS` overrides the
path), `ECHORA_CACHE_DIR` moves the cache and `ECHORA_LOG=debug` adds
logging.

`cargo test --workspace` runs the tests, which need no game files
([testing](testing.md)). Tests that need the real game are ignored by
default:

```sh
ECHORA_TEST_NUP=/path/to/game.nup cargo test --workspace --release -- --include-ignored
```

## Notes

- The translation is built against the clone's sources and links to the
  Rust toolchain that built it: after changing either, run
  `echora prepare --force`. Any change to `echora-guest` recompiles all
  1,111 region crates; a change to how a SIMD/FP operation works belongs in
  `echora-simd`, which rebuilds alone (generated code reaches it through
  `echora_guest::simd`'s declarations) before the bundles relink.
- `prepare` runs as many build jobs as the processors and memory allow
  (about 1.5 GB each); `-j N` overrides it.
- Keep the cache on disk: on systems where `/tmp` is a RAM-backed tmpfs,
  do not point `ECHORA_CACHE_DIR` there.
- `ECHORA_GAMEPAD_LAYOUT=labels` maps face buttons by label instead of
  position. Saves go to `~/.local/share/echora/save` (`ECHORA_SAVE_DIR`
  overrides it).

## Switches for test runs

These environment variables are read by `echora run`, or by `lift` where noted. The
others in the source are narrow debugging aids, described where they are
read.

**Running**

| Variable | Effect |
|---|---|
| `ECHORA_WINDOW=1`, `ECHORA_AUDIO=1` | Show the game in a window, and play its sound. |
| `ECHORA_STOP_PRESENT=N` | End the run, with its report, at presented frame N. |
| `ECHORA_WALL_SECONDS=N` | End an idle run after N host seconds, so the report shows what every thread waits on. |
| `ECHORA_AUTO_PRESS=N`, `ECHORA_AUTO_KEYS=A,PLUS,...` | Without a window, press the listed buttons in turn, one every N frames. |
| `ECHORA_GPU=0` | Run without the GPU backend. |
| `ECHORA_ALLOW_OLD_RADV=1` | Run on a Mesa RADV driver older than 26.1.4, which Echora otherwise refuses at start: those versions hang the GPU on the game's task shader draws, and after the kernel resets the GPU every later run hangs until a reboot. For debugging the driver only. |
| `ECHORA_UNBOUNDED_WRITE_WAITS=1` | CPU reads of guest memory wait for every pending shader store through the GPU page table, not only for the dispatches whose analysed write ranges overlap the read. For checking whether a stale read comes from those ranges. |
| `ECHORA_UNIFORMS_IN_PLACE=0` or `=all` | Draws read uniform buffers that earlier GPU work may still write directly from shared guest memory, in queue order, instead of waiting for that work and copying them. `0` copies every uniform buffer (waiting), `all` reads every shared one in place. For comparing the two. |
| `ECHORA_REPLAY_CORE=0` | On Linux with eight or more logical CPUs, one core (both of its hardware threads) is kept for the GPU replay thread and the guest workers run on the others. This leaves thread placement to the scheduler instead. |
| `ECHORA_AUDIO_THREAD=0` | Mix the game's sound on the game's own threads under the machine lock, as before, instead of on a host thread of its own. For comparing the two. |
| `ECHORA_NO_DMA_BUF=1` | Copy draws' storage buffers for the GPU instead of sharing guest memory with it through udmabuf (Linux). Sharing is used when `/dev/udmabuf` and the driver allow it. |
| `ECHORA_PACK_DMA_SPANS=0` | Keep individual Linux DMA-BUF imports when rebuilding the GPU page table. Packing discontiguous ranges into fewer imports is enabled by default when the kernel and driver accept them. |
| `ECHORA_DYNAMIC_UNIFORMS=0` | Use ordinary uniform descriptors for graphics. By default, graphics programs within the Vulkan device's dynamic uniform limit can reuse a descriptor set while changing uniform offsets. |
| `ECHORA_SUBMIT_ITEMS=N` | Change the maximum draws and dispatches per Vulkan submission. The default is 100 so the GPU can start processing work before the CPU finishes recording the frame. |
| `ECHORA_REPLAY_INLINE=1` | Replay submitted GPU work on the submitting guest thread instead of a thread of its own. |
| `ECHORA_REPLAY_CHUNK=N` | Let other threads access NVN state after at most N replayed commands. The default is 96, and 0 runs the replay without yielding between commands. |
| `ECHORA_CPU_INDIRECT=1` | Read indirect draw and dispatch arguments back on the CPU instead of letting the GPU read them (debugging). |
| `ECHORA_PAGE_TABLE=0` | Give translated programs an empty GPU page table, so their global memory reads return zeros (debugging). The page table is on by default. |

**Translation**

| Variable | Effect |
|---|---|
| `ECHORA_LIFT_OPTIONS=a,b` | Replace the CLI's default set for `lift`: `inline-vector,fused-flags,whole-vector-checks,fp-status-locals,window-memory,untagged-blocks,split-ticks,float-runs`. Use `flat` to turn off the set. See [codegen speed](codegen-speed.md) for measured options. |
| `ECHORA_LIFT_OPTIONS_REGIONS=regions` | Comma-separated region IDs (the hex suffix of `gen-r-<id>`) or a file listing them. Apply `ECHORA_LIFT_OPTIONS` on top of the defaults for those regions only. |
| `ECHORA_SHADER_CACHE=dir` | Keep the driver's compiled shader binaries in this folder between runs, so later runs skip compiling them. `echora run` sets it to `shader-cache` in the translation folder. |
| `ECHORA_SHADER_DIR=dir` | Dump every shader stage the game creates, for the `survey` example in `echora-shader` and `shader-check` in `echora-gpu`. |
| `ECHORA_SHADER_TRANSLATOR=plugin\|compare` | Use an installed shader plugin instead of Echora's own translator, or compare the two. `ECHORA_SHADER_PLUGIN=path` names the plugin; see [shader plugins](shader-plugin.md). |
| `ECHORA_HMMA=exact\|subgroup\|coop16` | How the matrix multiplies translate. By default cooperative matrices with 32-bit fragment staging where the device has them, else the subgroup path. |

**Measuring**

| Variable | Effect |
|---|---|
| `ECHORA_VIDEO_DECODE=cpu` | Decode the movies with rav1d on the CPU (`rav1d` works too). By default they decode on the GPU's video engine (Vulkan Video), and a movie the GPU cannot decode continues on rav1d. See [av1-decode.md](av1-decode.md). |
| `ECHORA_MOVIE_RATE=1` | Every 60 movie pictures, print how many have been decoded, the rate per second and the time since boot. This is the speed measure for the movie; presented frames per second are not comparable there, because loading presents black frames at a load-dependent rate. In the 3D scene, use `ECHORA_PRESENT_RATE`. |
| `ECHORA_PRESENT_RATE=N` | Every N presents, print the count and the rate over those N. |
| `ECHORA_TIMELINE=file` | Write each thread's guest runs, lock waits, NVN calls and replay pieces as spans over a few presents (debugging). |
| `ECHORA_GPU_TIME=1` | Compare time covered by GPU submission intervals with wall time and CPU fence waits. Include per-phase draw recording timings in the run report. |
| `ECHORA_GPU_TIME=items` | Also rank draw and dispatch timestamp intervals. Intervals can overlap and include waits for earlier commands. |
| `ECHORA_PROFILE_FROM=N` | Reset runtime and GPU statistics at present N to measure a specific part of a run. |
| `ECHORA_HOST_PROFILE=file` | Sample the host instruction pointer of every runtime thread and write the samples with the process's memory map, for offline symbolization (no `perf` needed). |
| `ECHORA_PC_STACKS=LO-HI` | Record the guest return chain of samples whose PC is in that range (`pc_stacks` in the report). |
| `ECHORA_PC_SAMPLES_FROM=N` | Sample guest PCs only from presented frame N on. |

For a before-and-after graphics measurement, use the same game data,
driver, settings and warmed shader cache. Run each build through the same
scene, alternating builds on the same boot. For the forest in the supported
1.1.1 dump, automated input with `ECHORA_AUTO_PRESS=120` and
`ECHORA_AUTO_KEYS=A,PLUS,A,PLUS,A,B,A` has reached the scene by present
10,500. Use `ECHORA_PROFILE_FROM=10500`, `ECHORA_STOP_PRESENT=10700` and
`ECHORA_PRESENT_RATE=100`, and confirm each run reached the scene. Compare
the present rate and divide `nvn.gpu_notes` timing totals by their draw
count to check changes in CPU work per draw. Movie decoding is measured
separately with `ECHORA_MOVIE_RATE`.

On a Radeon RX 9070 XT with RADV 26.2.7, two runs in each mode averaged
16.67 FPS with individual DMA-BUF imports and 16.95 FPS with packed imports.
The number of shared buffers per submission fell from 846–849 to 97–99.
Draw counts varied from 201,301 to 208,081 across the four runs.

**Capturing**

| Variable | Effect |
|---|---|
| `ECHORA_CAPTURE_EVERY=N` | Save the first and every N-th presented frame as PNGs in `run-<time>-frames/` beside the report (`0` turns it off). |
| `ECHORA_GPU_DUMP=dir`, `ECHORA_GPU_DUMP_FROM=N` | Every 300th present, write each colour target as an image, starting at present N. |
| `ECHORA_DRAW_LOG=file`, `ECHORA_DRAW_LOG_FROM=N` | Append a JSON line for every draw, with its targets, to the file, starting at present N. |

**Threads and scheduling**

| Variable | Effect |
|---|---|
| `ECHORA_WORKERS=N` or `auto` | Run guest threads on N host threads; `auto` (the default) is one per host CPU, at most 6, and `ECHORA_WORKERS=1` forces one worker. Overrides the settings' `[cpu] workers` (`"auto"` or 1-6, default `"auto"`: `echora_config::DEFAULT_WORKERS`). Each worker owns the application cores `c` with `c % N` equal to its index. |
| `ECHORA_PARALLEL_SLICE=N` | Guest instructions per slice when several workers run (default 50,000). Each slice end takes the global machine lock, so short slices make workers wait for each other. |
| `ECHORA_SCHEDULER_SLICE=N` | Guest instructions per scheduling slice (default 1,000). |
| `ECHORA_SERIAL_GUEST`, `ECHORA_ONE_CORE`, `ECHORA_ANY_CORE`, `ECHORA_FLAT_PRIORITY`, `ECHORA_PARALLEL_THREADS`, `ECHORA_EXCLUSIVE_THREADS`, `ECHORA_NO_TIMESLICE`, `ECHORA_TRACE_WORKERS`, `ECHORA_CORE_MAP` | Diagnostics for narrowing down multi-worker faults. |
| `ECHORA_VIRTUAL_CLOCK` | A timing experiment. |
| `ECHORA_NO_HLE` | When set during `lift`, keeps the game's zlib and zstd routines translated instead of replacing them with host code. |

Keep game files out of the repository: `.gitignore` excludes `*.nup`, NSO
and NPDM files, and `exefs/` and `romfs/` at the root.
