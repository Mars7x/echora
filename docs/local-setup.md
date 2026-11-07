# Local setup

## Requirements

- Rust 1.88 or newer (`rustup` stable).
- A C compiler: the `zstd` crate builds the reference zstd library.
- Your own NUP file or extracted dump of the game.
- On Linux: the ALSA, udev, xkbcommon, Wayland, fontconfig, X11-xcb and
  Vulkan development packages (Ubuntu: `libasound2-dev libudev-dev
  libxkbcommon-x11-dev libwayland-dev libfontconfig-dev libx11-xcb-dev
  libvulkan-dev`).

Development is done on Fedora (Linux x86-64, AMD RADV). The current tools
also build on Windows and macOS.

## Build and test

```sh
cargo build --release
cargo test --workspace
```

Tests that need the real game are ignored by default. Run them with:

```sh
ECHORA_TEST_NUP=/path/to/game.nup cargo test --workspace --release -- --include-ignored
```

## Working with the game

```sh
target/release/echora inspect /path/to/game.nup
target/release/echora verify /path/to/game.nup
target/release/echora import /path/to/game.nup
```

`import` prints the cached ExeFS directory. Set `ECHORA_CACHE_DIR` to put the
cache elsewhere and `ECHORA_LOG=debug` for more logging.

## Translating and running

Fetch the shader recompiler sources once (they are not stored in this
repository), then translate, build and run:

```sh
scripts/fetch-shader-sources.sh          # Lichii, sirit, fmt, SPIRV-Headers, Boost headers
target/release/echora lift /path/to/game.nup
(cd ~/.cache/echora/translations/<ExeFS key> && cargo build --release -j 6 -p echora-run)
ECHORA_WINDOW=1 ECHORA_AUDIO=1 target/release/echora run /path/to/game.nup
```

`fetch-shader-sources.sh` puts the sources in `~/.cache/echora/shader-src`
(`ECHORA_SHADER_SRC` overrides it); the generated workspace builds the
shader library from there. `lift` needs about 6 GB of memory and writes
4.2 GiB of Rust. The first build compiles 1,111 region crates (about an hour
on a Ryzen 7 5800X3D at six jobs, up to about 1.2 GB per compiler process);
later builds recompile only changed regions, but any change to
`echora-guest` recompiles them all. Keep the cache on disk: on systems where
`/tmp` is a RAM-backed tmpfs, do not point `ECHORA_CACHE_DIR` there.

`run` draws on the GPU (Vulkan) by default. `ECHORA_WINDOW=1` shows the
game in a window (keyboard and gamepads drive it; `ECHORA_GAMEPAD_LAYOUT=labels`
maps face buttons by label instead of position) and `ECHORA_AUDIO=1` plays
its sound. Saves go to `save/` in Echora's data folder
(`~/.local/share/echora` on Linux; `ECHORA_SAVE_DIR` overrides it).

Settings are read from `~/.config/echora/settings.toml` (the platform's
configuration directory; `ECHORA_SETTINGS` overrides the path). The runtime
applies the window size and mode, internal resolution, volume and
vibration so far.

`run` stops at the first operation that is not implemented and writes a
JSON report (`run-<time>.json`) next to the translation. Useful switches:

| Variable | Effect |
|---|---|
| `ECHORA_CAPTURE_EVERY=N` | Save the first and every N-th presented frame as PNGs in `run-<time>-frames/` beside the report. |
| `ECHORA_STOP_PRESENT=N` | End the run, with its report, at presented frame N. |
| `ECHORA_WALL_SECONDS=N` | End an idle run after N host seconds, so the report shows what every thread waits on. |
| `ECHORA_AUTO_PRESS=N`, `ECHORA_AUTO_KEYS=A,PLUS,...` | Without a window, press the listed buttons in turn, one every N frames (headless runs through menus). |
| `ECHORA_SHADER_DIR=dir` | Dump every shader stage the game creates, for `translate-dump` and `shader-check` in `echora-gpu`. |
| `ECHORA_WORKERS=N` | Run guest threads on N host threads (default 1). |
| `ECHORA_SCHEDULER_SLICE`, `ECHORA_VIRTUAL_CLOCK`, `ECHORA_NO_HLE` | Scheduling and timing experiments. |

Keep game files out of the repository; `.gitignore` excludes `*.nup`, NSO and
NPDM files and `exefs/`/`romfs/` at the root.
