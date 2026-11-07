# Echora

An experimental native recompilation of the Nintendo Switch 2 release of
*The Legend of Zelda: Ocarina of Time*. The goal is to translate the game's
AArch64 code ahead of time into native host code and run it on a portable
reimplementation of the console services it uses, with native graphics,
display, input and audio, rather than emulating the console.

> [!CAUTION]
> **Echora is NOT ready for stable gameplay, and it is not a way to play
> Ocarina of Time today.** It does not reach gameplay, runs far below full
> speed, is missing large parts of the graphics path, and can hang, crash
> or render incorrectly at any point. Saves are untested and may be lost
> or corrupted. It needs much more work, with no timeline, so do not report
> gameplay problems as bugs yet.

> Echora contains no game code, assets or keys, and it cannot run anything
> without a legally obtained copy of the game that you dump and supply
> yourself.

> **Note:** Echora is a personal project, made for my own enjoyment. AI
> tools were used in developing it.

## Status

Latest measured state (2026-11-07, update `0400486024DEC000` v131072,
display version 1.1.1, Linux x86-64 with an AMD RX 9070 XT):

- **The game runs.** The whole of the game's code is translated ahead of
  time (17.3 million instructions, 4.2 GiB of generated Rust in 1,111
  crates), compiled natively, and run on Echora's runtime. It boots, draws
  on the host GPU through Vulkan and presents frames: test runs go through
  3,000 presented frames, show the title logo and play the intro movie
  (AV1, decoded on the host). The movie renders in colour.
- **Not playable yet.** Runs have been seen as far as the intro movie;
  gameplay has not been reached. The mesh and task shaders that gameplay
  scenes need have no draw path in the runtime yet.
- **Far below full speed.** The game runs at 60 frames per second on the
  console; here the intro movie plays at about 6. The first 1,200 frames take about 165 s,
  down from 323 s. Every guest thread runs on one host thread by default,
  and that thread does most of the work, so faster translated code alone
  cannot reach full speed. Running guest threads on several host threads
  (`ECHORA_WORKERS`, up to the console's six application cores) was 1.9
  times faster in a partial test, but full parallel runs still crash
  within half a minute, so it stays off by default.
- **Shaders.** The console's GPU programs are translated to SPIR-V by the
  Lichii shader recompiler with Echora's patches. Of the 6,074 shader stages
  OoT reached in a test run, 5,960 translate and compile on the driver.
- **Content input works.** Echora opens the game's NUP container directly,
  validates its whole table of contents, and identifies the executable set.
  A full content check of the 25 GB package (186,592 files, 23.5 GiB after
  deduplication) passes in 12.6 s on a Ryzen 7 5800X3D. NUP and
  extracted-directory input produce the same cache key.
- **Executable loading works.** All four modules (`rtld`, `main`,
  `subsdk0`, `sdk`) are decompressed and every segment matches the SHA-256
  stored in its NSO header. `main` uses ZBIC, a Zstandard variant from
  Switch firmware 22, which Echora decodes by transcoding to standard
  Zstandard.
- **Launcher.** A GPUI launcher installs the game, edits the settings the
  runtime applies and starts it. It works with the mouse, the keyboard and
  controllers.
- **Platforms:** the workspace builds and its tests pass in CI on Linux
  x86-64 and Windows x86-64. The game's translation has only been built
  and run on Linux. macOS is not supported.

See the [roadmap](docs/roadmap.md) and [known limitations](docs/limitations.md).

## How it works

1. **Content.** `echora-nup` reads [NUP](https://github.com/a55uka/nup)
   containers in place: it verifies the TOC hash and UUID, bounds-checks every
   record, rejects unsafe names, and streams zstd extents with per-file SHA-256
   checks. `echora-content` turns a NUP or an extracted `exefs/`+`romfs/`
   dump into one inventory (packages, versions, build IDs, module hashes) and
   materializes the program's ExeFS in a cache keyed by its content.
   `echora-loader` decompresses NSO modules (LZ4 or ZBIC) and verifies each
   segment hash.
2. **Translation.** `echora-lift` discovers the game's code from its entry
   points, function prologues, relocation targets and jump tables, and emits
   it as Rust that the host toolchain compiles to native code. Each 64 KiB
   guest region becomes its own crate; the crates link into shared-library
   bundles.
3. **Runtime.** `echora-runtime` provides guest memory, threads and
   scheduling, synchronization, kernel SVCs, IPC and the system services the
   game reaches, implemented as the executable requires them. An operation
   that is not implemented stops the run with a report rather than
   returning invented success.
4. **Graphics.** NVN calls are handled on the host and executed on Vulkan
   (`echora-gpu`); shader programs are translated to SPIR-V. Movies are
   decoded on the host.
5. **Host integration.** Windowing, keyboard and gamepad input, audio, and a
   GPUI launcher on Linux and Windows, where all settings are changed.

See [architecture](docs/architecture.md) for the crate boundaries.

## Building

The game runs only on Linux for now. You need:

- your own dump of the game: the Switch 2 update 1.1.1, as a NUP file or
  an extracted dump;
- a Vulkan GPU, about 16 GB of RAM and about 30 GB of free disk;
- Rust 1.88 or newer, `git`, C and C++ compilers, and the system
  libraries:

```sh
# Ubuntu
sudo apt install build-essential libasound2-dev libudev-dev libxkbcommon-x11-dev libwayland-dev libfontconfig-dev libx11-xcb-dev libvulkan-dev
# Fedora
sudo dnf install gcc gcc-c++ alsa-lib-devel systemd-devel libxkbcommon-x11-devel wayland-devel fontconfig-devel libxcb-devel vulkan-loader-devel
```

Then build Echora and open the launcher:

```sh
git clone https://github.com/Mars7x/echora.git
cd echora
cargo build --release
target/release/echora-launcher
```

Choose your game, then press Play. The first start translates and builds
the game, which takes about an hour; later starts are immediate.

## Documentation

- [Architecture](docs/architecture.md)
- [Development](docs/development.md)
- [Content input](docs/content.md)
- [Roadmap](docs/roadmap.md)
- [Research notes and reuse audit](docs/research.md)
- [Testing](docs/testing.md)
- [Known limitations](docs/limitations.md)

## Legal

This is an independent project, not affiliated with or endorsed by Nintendo
or NVIDIA. The repository contains only original source code, tests and
documentation. It does not include or download game files, keys or firmware.
Do not open issues or pull requests containing game files, keys, or code
generated from them; generated translations stay on your machine.

Licensed under GPL-3.0-or-later. See [LICENSE](LICENSE) and
[NOTICE.md](NOTICE.md).
