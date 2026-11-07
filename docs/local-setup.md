# Local setup

This guide goes from a fresh clone to the game starting from the launcher.
Expect it to take a couple of hours, most of it one long build. Only Linux
can build and run the game's translation today; the tools and launcher
also build on Windows and macOS.

> Even when every step works, the game only reaches the intro movie, at
> about 6 frames per second. See the [known limitations](limitations.md).

## What you need

- **Linux x86-64 with a Vulkan GPU and driver.** Development is on Fedora
  with an AMD RX 9070 XT (RADV).
- **Your own copy of the game:** the Switch 2 update `0400486024DEC000`,
  version 1.1.1 (v131072), as a NUP file or an extracted dump directory.
  The translation is made from that exact build.
- **Rust 1.88 or newer** through `rustup` (stable), plus `git`.
- **C and C++ compilers** with C++20 support (GCC or Clang): the bundled
  zstd is C and the shader recompiler is C++.
- **Development packages** for ALSA, udev, xkbcommon, Wayland, fontconfig,
  X11-xcb and Vulkan:
  - Ubuntu: `sudo apt install build-essential libasound2-dev libudev-dev
    libxkbcommon-x11-dev libwayland-dev libfontconfig-dev libx11-xcb-dev
    libvulkan-dev`
  - Fedora: `sudo dnf install gcc gcc-c++ alsa-lib-devel systemd-devel
    libxkbcommon-x11-devel wayland-devel fontconfig-devel libxcb-devel
    vulkan-loader-devel`
- **Memory and disk:** about 16 GB of RAM (generating the translation
  needs about 6 GB, and each compiler process up to about 1.2 GB) and
  about 30 GB free for the generated code and its build.

## 1. Build Echora

```sh
git clone https://github.com/Mars7x/echora.git
cd echora
cargo build --release
```

This builds the `echora` command-line tool and `echora-launcher` into
`target/release/`. Keep the two side by side: the launcher starts the game
through the `echora` binary next to it.

To check your copy of the game before going further:

```sh
target/release/echora inspect /path/to/game.nup   # packages, versions, build ID
target/release/echora verify /path/to/game.nup    # hash-check every file
```

## 2. Fetch the shader recompiler

```sh
scripts/fetch-shader-sources.sh
```

The console's GPU programs are translated by the Lichii shader recompiler,
which is not stored in this repository. The script downloads it at a pinned
commit with its dependencies (sirit, fmt, SPIRV-Headers, Boost headers)
into `~/.cache/echora/shader-src` (`ECHORA_SHADER_SRC` overrides it), and
applies Echora's patches.

**Do this before step 3.** `lift` records where the shader sources are;
a translation made without them builds, but draws nothing.

## 3. Translate the game

```sh
target/release/echora lift /path/to/game.nup
```

This writes the game's code as Rust (4.2 GiB in 1,111 crates) and ends by
printing the translation folder:

```
translation: /home/you/.cache/echora/translations/<key>
```

The key comes from the game's content, so a NUP and an extracted dump of
the same build share one translation.

## 4. Build the translation

```sh
cd ~/.cache/echora/translations/<key>
cargo build --release -j 6 -p echora-run
```

The first build takes about an hour on a Ryzen 7 5800X3D. Lower `-j` if
you are short of memory: each job can use about 1.2 GB.

- Later builds recompile only what changed, but any change to
  `echora-guest` recompiles every region crate, so after pulling new
  commits, run `echora lift` again and rebuild.
- The translation links to the Rust toolchain that built it. If you
  update or remove that toolchain, rebuild the translation.
- Keep the cache on disk. On systems where `/tmp` is a RAM-backed tmpfs,
  do not point `ECHORA_CACHE_DIR` there.

## 5. Start the game from the launcher

```sh
target/release/echora-launcher
```

1. Choose your NUP file or dump. The launcher copies the game into
   Echora's data folder (`~/.local/share/echora/game`), after which the
   original is no longer needed.
2. Press **Play**. The game starts in its own window with sound and no
   time limit.

The launcher works with the mouse, the keyboard (Tab, arrows, Enter) and
controllers (D-pad or stick, A, B). Its settings (window mode, internal
resolution, vibration) are saved to `~/.config/echora/settings.toml` and
applied by the runtime. With **Skip launcher** on, opening the launcher
starts the game straight away; `echora-launcher --show` opens the launcher
window again.

In the game, keyboard and gamepads drive the controls;
`ECHORA_GAMEPAD_LAYOUT=labels` maps face buttons by label instead of
position. Saves go to `~/.local/share/echora/save` (`ECHORA_SAVE_DIR`
overrides it).

## Troubleshooting

- **"no built translation at …"**: steps 3 and 4 have not been done for
  this build of the game, or the build failed. Run them again.
- **"the ExeFS does not match this translation's build"**: the game
  you installed is a different version from the one translated. Lift and
  build again from the installed game.
- **The window stays black**: the shader sources were missing when you
  ran `lift`. Do step 2, then steps 3 and 4 again.
- **The game stops with a report**: every operation Echora does not
  implement yet ends the run and writes `run-<time>.json` in the
  translation folder. This is expected at the current stage.

## Running from the command line

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

Useful switches for test runs:

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
