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

Keep game files out of the repository; `.gitignore` excludes `*.nup`, NSO and
NPDM files and `exefs/`/`romfs/` at the root.
