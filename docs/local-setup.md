# Local setup

This guide goes from a fresh clone to the game starting from the launcher.
You build Echora once; the launcher does the rest. Expect the first start
to take about an hour, most of it building the game. Only Linux can build
and run the game's translation today; the tools and launcher also build on
Windows and macOS.

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
`target/release/`. Keep the two side by side and keep the clone where it
is: the launcher starts the game through the `echora` binary next to it,
and the game's translation is built against the clone's sources.

## 2. Start the launcher

```sh
target/release/echora-launcher
```

1. **Choose your game.** Pick your NUP file or dump. The launcher copies
   it into Echora's data folder (`~/.local/share/echora/game`), after which
   the original is no longer needed.
2. **Press Play.** The first time, the launcher prepares the game before
   starting it, showing each step and a progress bar:
   - *Getting the shader recompiler*: downloads the Lichii shader
     recompiler, which translates the console's GPU programs, at a pinned
     commit with its dependencies, into `~/.cache/echora/shader-src`, and
     applies Echora's patches.
   - *Translating the game*: writes the game's code as Rust (4.2 GiB in
     1,111 crates) into `~/.cache/echora/translations/<key>`. The key
     comes from the game's content, so a NUP and a dump of the same build
     share one translation.
   - *Building the game*: compiles that translation, counting the 1,111
     crates. This takes about an hour on a Ryzen 7 5800X3D. The launcher
     runs as many compiler jobs at once as your processors and memory
     allow, leaving about 1.5 GB for each.

   When it finishes, the game starts in its own window with sound and no
   time limit. Later starts are immediate.

Closing the launcher while it prepares stops the preparation; the next
Play continues where the build left off. After you pull new commits and
build Echora again, the next Play prepares the game again, which is
quicker when little has changed.

The launcher works with the mouse, the keyboard (Tab, arrows, Enter) and
controllers (D-pad or stick, A, B). Its settings (window mode, internal
resolution, vibration) are saved to `~/.config/echora/settings.toml` and
applied by the runtime. With **Skip launcher** on, opening the launcher
starts the game straight away (or shows the launcher when the game still
needs preparing); `echora-launcher --show` opens the launcher window
again.

In the game, keyboard and gamepads drive the controls;
`ECHORA_GAMEPAD_LAYOUT=labels` maps face buttons by label instead of
position. Saves go to `~/.local/share/echora/save` (`ECHORA_SAVE_DIR`
overrides it).

Keep the cache on disk: on systems where `/tmp` is a RAM-backed tmpfs, do
not point `ECHORA_CACHE_DIR` there. The translation links to the Rust
toolchain that built it, so if you update or remove that toolchain,
prepare the game again (`echora prepare --force`).

## Troubleshooting

- **"Couldn't prepare the game: …"**: the message names the step that
  failed. Fetching needs `git` and a network connection; building needs
  Rust (`cargo` on your `PATH` or in `~/.cargo/bin`), C and C++ compilers
  and the packages above. To see the full output, run the same step in a
  terminal: `target/release/echora prepare ~/.local/share/echora/game/game.nup`.
- **The build runs out of memory**: run `echora prepare` yourself with
  fewer jobs, for example `-j 2`.
- **"the ExeFS does not match this translation's build"**: the installed
  game is a different version from the one translated. Run
  `echora prepare --force` on it.
- **The game stops with a report**: every operation Echora does not
  implement yet ends the run and writes `run-<time>.json` in the
  translation folder. This is expected at the current stage.

## Running from the command line

The launcher's steps are commands of their own:

```sh
target/release/echora prepare /path/to/game.nup   # fetch, translate and build (-j N, --force, --check)
target/release/echora inspect /path/to/game.nup   # packages, versions, build ID
target/release/echora verify /path/to/game.nup    # hash-check every file
```

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
