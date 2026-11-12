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
| `gilrs` | 0.11.2 | Apache-2.0 OR MIT |
| `gpui` | 0.2.2 | Apache-2.0 |
| `hex` | 0.4.3 | MIT OR Apache-2.0 |
| `jiff` | 0.2.39 | Unlicense OR MIT |
| `libc` | 0.2.189 | MIT OR Apache-2.0 |
| `miniz_oxide` | 0.8.9 | MIT OR Zlib OR Apache-2.0 |
| `portable-atomic` | 1.15.0 | Apache-2.0 OR MIT |
| `rav1d` | 1.1.0 | BSD-2-Clause |
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
| `unsafe-libopus` | 0.2.0 | BSD-3-Clause (libopus, transpiled to Rust) |
| `yaxpeax-arch` | 0.3.3 | 0BSD |
| `yaxpeax-arm` | 0.4.0 | 0BSD |
| `zstd` | 0.13.4 | MIT (bundles zstd, BSD-3-Clause OR GPL-2.0) |

## Shader translator plugins

Echora can load an external shader translator at run time
([docs/shader-plugin.md](docs/shader-plugin.md)). None is stored in or
built by this repository; a plugin is distributed under its own license.

## Reference implementations

Echora's behaviour was checked against these projects. Layouts, constants
and rules were learned from them and written anew in Rust; the pinned
revisions are the ones the source comments cite.

| Project | Used for | License |
|---|---|---|
| [Eden](https://git.eden-emu.dev/eden-emu/eden), with the Switch 2 patch set the source comments call Eden-NS2 (`1a6d7d4`) | System service behaviour, kernel checks, the graphics driver interface, display, input, and the audio renderer's layouts and rules | GPL-3.0-or-later |
| yuzu (as carried in Eden) | The audio renderer's processing rules and resampling filter taps, file system result numbering, shader input and output conventions | GPL-2.0-or-later |
| [Lichii](https://github.com/zortos293/Lichii) `c215653`, `src/shader_recompiler/frontend/ampere` | Ampere instruction encodings and semantics | GPL-3.0-or-later |
| Mesa's NAK `979c6d1`, `src/nouveau/compiler/nak/sm70_encode.rs` | Ampere bit layouts (cross-checks) | MIT |
| [Atmosphère](https://github.com/Atmosphere-NX/Atmosphere) `6b0c077` | Kernel behaviour: memory mapping checks, IPC and synchronization | GPL-2.0 |

## Test data

`crates/echora-av1/tests/vectors` holds AV1 conformance test vectors from
the Alliance for Open Media's libaom test data
(`https://storage.googleapis.com/aom-test-data/`, listed in libaom's
`test/test_vectors.cc`), unchanged. They are Copyright (c) the Alliance
for Open Media, under the BSD 2-Clause license and the Alliance for Open
Media Patent License 1.0 (<https://aomedia.org/license/>).
`crates/echora-av1/tests/fetch-vectors.sh` downloads more of them at test
time; they are not stored in the repository.

| Files | Source | License |
|---|---|---|
| `av1-1-b8-01-size-*.ivf`, `av1-1-b8-04-cdfupdate.ivf`, `av1-1-b8-05-mv.ivf`, `av1-1-b8-06-mfmv.ivf`, `av1-1-b8-22-svc-L1T2.ivf`, `av1-1-b8-23-film_grain-50.ivf`, `av1-1-b10-23-film_grain-50.ivf`, `av1-1-b8-24-monochrome.ivf`, `av1-1-b10-24-monochrome.ivf` | libaom test vectors | BSD-2-Clause, AOM Patent License 1.0 |

## Bundled fonts

`crates/echora-launcher/assets/fonts` holds Inter 4.1 (Regular, SemiBold,
Bold) by The Inter Project Authors and Cinzel (Regular, Bold) by The Cinzel
Project Authors, both under the SIL Open Font License 1.1 (`Inter-OFL.txt`
and `Cinzel-OFL.txt` alongside). They are embedded in the launcher binary.

## References

- NUP format specification and reference implementation:
  [a55uka/nup](https://github.com/a55uka/nup) (MIT). `echora-nup` is an
  independent Rust implementation of that specification.
- [kinnay/zbic](https://github.com/kinnay/zbic) (GPLv2) was studied to learn
  the ZBIC format; Echora's decoder contains no code from it.
- Zstandard format: RFC 8878.
- Switchbrew, libnx and Atmosphère documentation for executable formats.
- Projects consulted for game-specific behavior are listed in
  [docs/research.md](docs/research.md).
