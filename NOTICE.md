# Notices

## Project source

Echora is licensed under GPL-3.0-or-later; see [LICENSE](LICENSE). The
repository contains original source, tests and documentation. Game files and
anything generated from them are not part of it.

## Direct Rust dependencies

Versions as resolved in `Cargo.lock` at the initial commit. Cargo fetches
these separately; preserve their notices when redistributing builds.

| Package | Version | License |
|---|---|---|
| `anyhow` | 1.0.104 | MIT OR Apache-2.0 |
| `ash` | 0.38.0 | MIT OR Apache-2.0 |
| `astc-decode` | 0.3.1 | MIT OR Apache-2.0 |
| `cc` (build) | 1.11.0 | MIT OR Apache-2.0 |
| `clap` | 4.6.7 | MIT OR Apache-2.0 |
| `cpal` | 0.15.3 | Apache-2.0 |
| `hex` | 0.4.3 | MIT OR Apache-2.0 |
| `jiff` | 0.2.39 | Unlicense OR MIT |
| `libc` | 0.2.191 | MIT OR Apache-2.0 |
| `miniz_oxide` | 0.8.9 | MIT OR Zlib OR Apache-2.0 |
| `portable-atomic` | 1.15.0 | Apache-2.0 OR MIT |
| `raw-window-handle` | 0.6.2 | MIT OR Apache-2.0 OR Zlib |
| `lz4_flex` | 0.11.6 | MIT |
| `serde` | 1.0.229 | MIT OR Apache-2.0 |
| `serde_json` | 1.0.154 | MIT OR Apache-2.0 |
| `sha2` | 0.10.9 | MIT OR Apache-2.0 |
| `thiserror` | 2.0.21 | MIT OR Apache-2.0 |
| `toml` | 1.2.0 | MIT OR Apache-2.0 |
| `tracing` | 0.1.44 | MIT |
| `tracing-subscriber` | 0.3.23 | MIT |
| `winit` | 0.30.13 | Apache-2.0 |
| `yaxpeax-arch` | 0.3.3 | 0BSD |
| `yaxpeax-arm` | 0.4.0 | 0BSD |
| `zstd` | 0.13.4 | MIT (bundles zstd, BSD-3-Clause OR GPL-2.0) |

## Fetched build inputs

When `ECHORA_SHADER_SRC` is set, `echora-gpu` compiles sources fetched by
`scripts/fetch-shader-sources.sh`; none are stored in this repository:

| Source | Pinned | License |
|---|---|---|
| [PokeDinner](https://git.eden-emu.dev/Franseven/PokeDinner-a-Eglantine-Fork) shader recompiler | `c3ce9bc` plus `crates/echora-gpu/native/shader/patches` | GPL-3.0-or-later |
| [sirit](https://github.com/eden-emulator/sirit) | v1.0.5 | AGPL-3.0 |
| [fmt](https://github.com/fmtlib/fmt) | 12.1.0 | MIT |
| [SPIRV-Headers](https://github.com/KhronosGroup/SPIRV-Headers) | vulkan-sdk-1.4.341.0 | MIT-style (Khronos) |
| Boost headers | 1.91.0 | BSL-1.0 |

GPL-3.0 section 13 permits combining with AGPL-3.0 code. A binary built
with the shader library carries the AGPL terms for the sirit part,
including its network-interaction clause.

## References

- NUP format specification and reference implementation:
  [a55uka/nup](https://github.com/a55uka/nup) (MIT). `echora-nup` is an
  independent Rust implementation of that specification.
- [pokopia-recomp](https://github.com/Mars7x/pokopia-recomp) (GPL-3.0-or-later),
  the design basis for Echora's recompiler and runtime.
  `crates/echora-loader/src/nso.rs` and `mod0.rs` are adapted from its
  `src/loader/nso.rs` and `src/loader/mod0.rs`; `crates/echora-guest` from
  its `crates/guest`; `crates/echora-lift` from its `src/aarch64.rs`,
  `src/simd.rs` and `src/codegen.rs`; `crates/echora-runtime` from its
  runtime, threads, synchronization, IPC, services, logging, file system,
  RomFS, HID, display, content, HLE, audio, bootstrap, NVN, SASS, format
  and window modules; `crates/echora-gpu` from its `crates/gpu` and
  `native/shader`; `crates/echora-input` from its `crates/input`; and the
  synthetic programs and
  atomics assertions in `crates/echora-aot-tests` from its
  `tests/aot_execution.rs`.
- [kinnay/zbic](https://github.com/kinnay/zbic) (GPLv2) was studied to learn
  the ZBIC format; Echora's decoder contains no code from it.
- Zstandard format: RFC 8878.
- Switchbrew, libnx and Atmosphère documentation for executable formats.
- Projects consulted for game-specific behavior are listed in
  [docs/research.md](docs/research.md).
