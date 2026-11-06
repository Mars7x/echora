# Local setup

## Requirements

- Rust 1.88 or newer (`rustup` stable).
- A C compiler: the `zstd` crate builds the reference zstd library.
- Your own NUP file or extracted dump of the game.

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

```sh
target/release/echora lift /path/to/game.nup
(cd ~/.cache/echora/translations/<ExeFS key> && cargo build --release -j 6 -p echora-run)
target/release/echora run /path/to/game.nup
```

`lift` needs about 6 GB of memory and writes 4.2 GiB of Rust. The first
build compiles 1,111 region crates (about an hour on a Ryzen 7 5800X3D at
six jobs, up to about 1.2 GB per compiler process); later builds recompile
only changed regions, but any change to `echora-guest` recompiles them all.
Keep the cache on disk: on systems where `/tmp` is a RAM-backed tmpfs, do
not point `ECHORA_CACHE_DIR` there.

`run` stops at the first operation that is not implemented and writes a
JSON report (`run-<time>.json`) next to the translation. Useful switches:
`ECHORA_WORKERS` (host threads for guest threads), `ECHORA_SCHEDULER_SLICE`,
`ECHORA_VIRTUAL_CLOCK`, `ECHORA_NO_HLE`.

Keep game files out of the repository; `.gitignore` excludes `*.nup`, NSO and
NPDM files and `exefs/`/`romfs/` at the root.
