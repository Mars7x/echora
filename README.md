# Echora

An experimental native recompilation of the Nintendo Switch 2 release of
*The Legend of Zelda: Ocarina of Time*. The goal is to translate the game's
AArch64 code ahead of time into native host code and run it on a portable
reimplementation of the console services it uses, with native graphics,
display, input and audio, rather than emulating the console.

> **This is an early development project, not a playable port.** It contains
> no game code, assets or keys, and it cannot run anything without a legally
> obtained copy of the game that you dump and supply yourself.

> **Note:** AI tools were used in developing this project.

## Status

Latest measured state (2026-11-06, update `0400486024DEC000` v131072,
display version 1.1.1):

- **Content input works.** Echora opens the game's NUP container directly,
  validates its whole table of contents, and identifies the executable set.
  A full content check of the 25 GB package (186,592 files, 23.5 GiB after
  deduplication) passes in 12.6 s on a Ryzen 7 5800X3D. The ExeFS is placed
  in a content-addressed cache; NUP and extracted-directory input produce the
  same cache key.
- **Resolution and settings foundations** exist as tested libraries: a
  resolution planner that keeps guest, internal scene, UI and output sizes
  separate (arbitrary sizes, non-integer ratios, ultrawide), and a versioned
  settings model with requested-versus-effective resolution and live-change
  classes. Neither is connected to a renderer yet.
- **Executable loading works.** `echora load` decompresses all four modules
  (`rtld`, `main`, `subsdk0`, `sdk`; `main` alone is 81 MB of text, rodata and
  data)
  and every segment matches the SHA-256 stored in its NSO header. `main` uses
  ZBIC, a Zstandard variant from Switch firmware 22, which Echora decodes by
  transcoding to standard Zstandard. MOD0 and dynamic-section metadata
  (GNU-hash symbol tables, RELA/RELR/PLT relocations, init arrays) are
  validated for every module. Loading takes about 0.2 s.
- **Native execution of synthetic code works.** The AArch64 lifter and guest
  CPU/memory model (ported from pokopia-recomp) translate test programs to
  Rust at build time; they run natively and pass integer, call, atomic,
  exclusive and acquire/release semantics checks. Guest memory works on
  Linux, Windows (10 1803 or later) and macOS.
- **Code discovery on the real game:** `echora analyze` finds 17.3 million
  instructions, 96.6% of `main`'s text, in 14 s. Every discovered
  instruction form is translated except a single `stg` (memory tagging);
  see the [research notes](docs/research.md#code-discovery-on-the-checked-input).
- **Translation:** `echora lift` writes the whole game as 4.2 GiB of Rust in
  1,111 crates; all of it type-checks. A full native build has not been
  run yet (estimated at about 4.5 CPU-hours).
- **No game code runs yet.** Completing the lifter and the runtime are
  next; see the [roadmap](docs/roadmap.md).
- **Platforms:** the workspace builds and its tests pass in CI on Linux
  x86-64, Windows x86-64 and macOS ARM64. Synthetic programs run natively on
  all three; no game code runs on any platform.

## How it works

1. **Content.** `echora-nup` reads [NUP](https://github.com/a55uka/nup)
   containers in place: it verifies the TOC hash and UUID, bounds-checks every
   record, rejects unsafe names, and streams zstd extents with per-file SHA-256
   checks. `echora-content` turns a NUP or an extracted `exefs/`+`romfs/`
   dump into one inventory (packages, versions, build IDs, module hashes) and
   materializes the program's ExeFS in a cache keyed by its content.
   `echora-loader` decompresses NSO modules (LZ4 or ZBIC) and verifies each
   segment hash.
2. **Translation** (planned). Executable modules are decoded and translated
   ahead of time into native code through an architecture-neutral
   representation, following the approach proven in
   [pokopia-recomp](https://github.com/Mars7x/pokopia-recomp).
3. **Runtime** (planned). Guest memory, threads, synchronization, IPC and the
   system services the game reaches, implemented as the executable requires
   them.
4. **Graphics** (planned). NVN calls are handled on the host and executed on
   Vulkan first, behind a backend boundary that also admits Metal and
   Direct3D 12. Render targets are sized by `echora-resolution`, so the scene
   renders at the selected internal resolution instead of being stretched.
5. **Host integration** (planned). Native windowing, input, audio and a
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
echora lift game.nup             # write the translation (4.2 GiB of Rust) to the cache
echora run game.nup              # run the built translation (see docs/local-setup.md)
echora extract game.nup out --subtree romfs
```

`echora-launcher` (GPUI) installs the game from your NUP or dump into
Echora's data folder (`~/.local/share/echora/game` on Linux), after which
the original is no longer needed, edits the settings the runtime applies
and starts it. It runs on Linux, Windows and macOS; place it next to
the `echora` binary. With "Skip launcher" on it starts the game directly;
`echora-launcher --show` opens the window again.

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
