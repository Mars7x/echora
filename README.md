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

> **Note:** AI tools were used in developing this project.

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
- **Far below full speed.** The intro movie runs at about 6 frames per
  second against the game's 30; the first 1,200 frames take about 165 s,
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
  x86-64, Windows x86-64 and macOS ARM64. The game's translation has only
  been built and run on Linux.

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
   GPUI launcher on Linux, Windows and macOS, where all settings are
   changed.

See [architecture](docs/architecture.md) for the crate boundaries.

## Building and running

You need Rust 1.88 or newer and a C compiler (for the bundled zstd).

```sh
cargo build --release
cargo test --workspace
```

The current commands work on your NUP file or extracted dump directory:

```sh
echora inspect game.nup          # packages, versions, build ID, module hashes
echora verify game.nup           # decode and hash-check every file
echora import game.nup           # validate and cache the program's ExeFS
echora load game.nup             # decompress and verify every module
echora analyze game.nup --limit 20000000   # code discovery (needs ~6 GB RAM)
echora boot game.nup             # create the game process without running it
echora lift game.nup             # write the translation (4.2 GiB of Rust) to the cache
echora run game.nup              # run the built translation (see docs/local-setup.md)
echora extract game.nup out --subtree romfs
```

Running the game needs the translation built first and the shader
recompiler sources fetched; [local setup](docs/local-setup.md) has the
steps. `ECHORA_WINDOW=1` opens the game's window and `ECHORA_AUDIO=1`
plays its sound.

`echora-launcher` (GPUI) installs the game from your NUP or dump into
Echora's data folder (`~/.local/share/echora/game` on Linux), after which
the original is no longer needed, edits the settings the runtime applies
and starts it. It runs on Linux, Windows and macOS; place it next to
the `echora` binary. With "Skip launcher" on it starts the game directly;
`echora-launcher --show` opens the window again. The launcher starts the
game with its window and sound on.

See [local setup](docs/local-setup.md) and [content input](docs/content.md).

## Documentation

- [Architecture](docs/architecture.md)
- [Local setup](docs/local-setup.md)
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
