# Shader translation library

`echora_shader.{h,cpp}` and `echora_common.cpp` are a C ABI over the
Lichii shader recompiler (an Eglantine/Eden fork), which translates the console's
GPU machine code to SPIR-V. The recompiler is not part of this repository.

```sh
scripts/fetch-shader-sources.sh          # pinned sources into ~/.cache/echora/shader-src
ECHORA_SHADER_SRC=~/.cache/echora/shader-src cargo build -p echora-gpu
```

`fetch-shader-sources.sh` checks out Lichii at a pinned commit, applies
`patches/` (Echora's lowering extensions), and
clones sirit, fmt, SPIRV-Headers and the needed Boost headers at pinned
tags. `build.rs` compiles them with the `cc` crate when `ECHORA_SHADER_SRC`
is set; otherwise shader translation reports that the library is absent.

Licenses of the fetched sources: Lichii GPL-3.0-or-later, sirit
(eden-emulator fork) AGPL-3.0, fmt MIT, SPIRV-Headers MIT-style (Khronos),
Boost BSL-1.0. GPL-3.0 permits combining with AGPL-3.0 code; a binary built
with this library carries the AGPL terms for that part, including its
network-interaction clause.
