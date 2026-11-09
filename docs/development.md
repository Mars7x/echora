# Development

Notes for working on Echora from the command line. To build Echora and start
the game, see the [README](../README.md#building).

## Command line

The launcher's steps are commands of their own:

```sh
target/release/echora prepare /path/to/game.nup   # fetch, translate and build (-j N, --force, --check)
target/release/echora inspect /path/to/game.nup   # packages, versions, build ID
target/release/echora verify /path/to/game.nup    # hash-check every file
target/release/echora import /path/to/game.nup    # validate and cache the program's ExeFS
target/release/echora load /path/to/game.nup      # decompress and verify every module
target/release/echora analyze /path/to/game.nup --limit 20000000   # code discovery (~6 GB RAM)
target/release/echora boot /path/to/game.nup      # create the game process without running it
target/release/echora lift /path/to/game.nup      # only translate (several GiB of Rust)
target/release/echora extract /path/to/game.nup out --subtree romfs
```

See [content input](content.md) for the formats Echora reads.

`prepare` does nothing when the game is already prepared. The steps it
runs can also be run one at a time: `scripts/fetch-shader-sources.sh`
(before lifting, since `lift` records where the shader sources are),
`echora lift`, then `cargo build --release -p echora-run` in the
translation folder it prints. A translation built by hand this way is
used as it is.

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
| `ECHORA_NO_DMA_BUF=1` | Copy draws' storage buffers for the GPU instead of sharing guest memory with it through udmabuf (Linux). Sharing is used when `/dev/udmabuf` and the driver allow it. |

**Translation**

| Variable | Effect |
|---|---|
| `ECHORA_LIFT_OPTIONS=a,b` | Code generation options for `lift`: `inline-vector`, `structured`. Both are off by default. |
| `ECHORA_SHADER_CACHE=dir` | Keep the driver's compiled shader binaries in this folder between runs, so later runs skip compiling them. `echora run` sets it to `shader-cache` in the translation folder. |
| `ECHORA_SHADER_DIR=dir` | Dump every shader stage the game creates, for `translate-dump` and `shader-check` in `echora-gpu`. |

**Measuring**

| Variable | Effect |
|---|---|
| `ECHORA_MOVIE_RATE=1` | Every 60 movie pictures, print how many have been decoded, the rate per second and the time since boot. This is the speed measure to compare runs with; presented frames per second are not comparable, because loading presents black frames at a load-dependent rate. |
| `ECHORA_GPU_TIME=1` | Every few seconds, print how long the GPU was busy, against wall time and against the time the CPU waited for it. |
| `ECHORA_HOST_PROFILE=file` | Sample the host instruction pointer of every runtime thread and write the samples with the process's memory map, for offline symbolization (no `perf` needed). |
| `ECHORA_PC_STACKS=LO-HI` | Record the guest return chain of samples whose PC is in that range (`pc_stacks` in the report). |
| `ECHORA_PC_SAMPLES_FROM=N` | Sample guest PCs only from presented frame N on. |

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
