# Development

Notes for working on Echora from the command line. To build and play,
see the [README](../README.md#building).

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
target/release/echora lift /path/to/game.nup      # only translate (4.2 GiB of Rust)
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
  1,111 region crates.
- `prepare` runs as many build jobs as the processors and memory allow
  (about 1.5 GB each); `-j N` overrides it.
- Keep the cache on disk: on systems where `/tmp` is a RAM-backed tmpfs,
  do not point `ECHORA_CACHE_DIR` there.
- `ECHORA_GAMEPAD_LAYOUT=labels` maps face buttons by label instead of
  position. Saves go to `~/.local/share/echora/save` (`ECHORA_SAVE_DIR`
  overrides it).

## Switches for test runs

| Variable | Effect |
|---|---|
| `ECHORA_CAPTURE_EVERY=N` | Save the first and every N-th presented frame as PNGs in `run-<time>-frames/` beside the report (`0` turns it off). |
| `ECHORA_STOP_PRESENT=N` | End the run, with its report, at presented frame N. |
| `ECHORA_WALL_SECONDS=N` | End an idle run after N host seconds, so the report shows what every thread waits on. |
| `ECHORA_AUTO_PRESS=N`, `ECHORA_AUTO_KEYS=A,PLUS,...` | Without a window, press the listed buttons in turn, one every N frames (headless runs through menus). |
| `ECHORA_SHADER_DIR=dir` | Dump every shader stage the game creates, for `translate-dump` and `shader-check` in `echora-gpu`. |
| `ECHORA_WORKERS=N` | Run guest threads on N host threads, up to 6 (default 1). Each worker owns the application cores `c` with `c % N` equal to its index. Not yet stable: runs crash within about half a minute. |
| `ECHORA_GPU=0` | Run without the GPU backend. |
| `ECHORA_HOST_PROFILE=file` | Sample the host instruction pointer of every runtime thread on a user-time timer and write the samples with the process's memory map, for offline symbolization (no `perf` needed). |
| `ECHORA_PC_STACKS=LO-HI` | Record the guest return chain of samples whose PC is in that range (`pc_stacks` in the report). |
| `ECHORA_SERIAL_GUEST`, `ECHORA_ONE_CORE`, `ECHORA_ANY_CORE`, `ECHORA_FLAT_PRIORITY`, `ECHORA_PARALLEL_THREADS`, `ECHORA_EXCLUSIVE_THREADS`, `ECHORA_NO_TIMESLICE`, `ECHORA_TRACE_WORKERS` | Diagnostics for bisecting multi-worker faults. |
| `ECHORA_SCHEDULER_SLICE`, `ECHORA_VIRTUAL_CLOCK`, `ECHORA_NO_HLE` | Scheduling and timing experiments. |

Keep game files out of the repository; `.gitignore` excludes `*.nup`, NSO and
NPDM files and `exefs/`/`romfs/` at the root.
