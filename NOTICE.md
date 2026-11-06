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
| `clap` | 4.6.7 | MIT OR Apache-2.0 |
| `hex` | 0.4.3 | MIT OR Apache-2.0 |
| `libc` | 0.2.191 | MIT OR Apache-2.0 |
| `lz4_flex` | 0.11.6 | MIT |
| `serde` | 1.0.229 | MIT OR Apache-2.0 |
| `serde_json` | 1.0.154 | MIT OR Apache-2.0 |
| `sha2` | 0.10.9 | MIT OR Apache-2.0 |
| `thiserror` | 2.0.21 | MIT OR Apache-2.0 |
| `toml` | 1.2.0 | MIT OR Apache-2.0 |
| `tracing` | 0.1.44 | MIT |
| `tracing-subscriber` | 0.3.23 | MIT |
| `yaxpeax-arch` | 0.3.3 | 0BSD |
| `yaxpeax-arm` | 0.4.0 | 0BSD |
| `zstd` | 0.13.4 | MIT (bundles zstd, BSD-3-Clause OR GPL-2.0) |

## References

- NUP format specification and reference implementation:
  [a55uka/nup](https://github.com/a55uka/nup) (MIT). `echora-nup` is an
  independent Rust implementation of that specification.
- [pokopia-recomp](https://github.com/Mars7x/pokopia-recomp) (GPL-3.0-or-later),
  the design basis for Echora's recompiler and runtime.
  `crates/echora-loader/src/nso.rs` and `mod0.rs` are adapted from its
  `src/loader/nso.rs` and `src/loader/mod0.rs`; `crates/echora-guest` from
  its `crates/guest`; `crates/echora-lift` from its `src/aarch64.rs`,
  `src/simd.rs` and `src/codegen.rs`; and the synthetic programs and
  atomics assertions in `crates/echora-aot-tests` from its
  `tests/aot_execution.rs`.
- [kinnay/zbic](https://github.com/kinnay/zbic) (GPLv2) was studied to learn
  the ZBIC format; Echora's decoder contains no code from it.
- Zstandard format: RFC 8878.
- Switchbrew, libnx and Atmosphère documentation for executable formats.
- Projects consulted for game-specific behavior are listed in
  [docs/research.md](docs/research.md).
