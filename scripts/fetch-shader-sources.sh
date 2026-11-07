#!/usr/bin/env bash
# Fetch the pinned shader recompiler sources (GPL-3.0-or-later, PokeDinner,
# an Eden fork) and its header dependencies into $ECHORA_SHADER_SRC
# (default ~/.cache/echora/shader-src).
# Nothing here enters Git.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${ECHORA_SHADER_SRC:-$HOME/.cache/echora/shader-src}"
mkdir -p "$ROOT"
cd "$ROOT"

POKEDINNER_URL=https://git.eden-emu.dev/Franseven/PokeDinner-a-Eglantine-Fork.git
POKEDINNER_COMMIT=c3ce9bcd5b04e5a1480ebf764ad7f1dd47b1adaa
if [ ! -d pokedinner/.git ]; then
    git init -q pokedinner
    git -C pokedinner remote add origin "$POKEDINNER_URL"
    git -C pokedinner sparse-checkout set --no-cone /src/shader_recompiler /src/common /LICENSE.txt
fi
git -C pokedinner fetch -q --depth 1 --filter=blob:none origin "$POKEDINNER_COMMIT"
git -C pokedinner -c advice.detachedHead=false checkout -q --force FETCH_HEAD
# Our extensions to the lowering, kept as patches against the pin.
for patch in "$REPO"/crates/echora-gpu/native/shader/patches/*.patch; do
    git -C pokedinner apply "$patch"
done

clone() { # repo tag dir
    [ -d "$3" ] || git clone -q --depth 1 --branch "$2" "https://github.com/$1.git" "$3"
}
clone fmtlib/fmt 12.1.0 fmt
clone eden-emulator/sirit v1.0.5 sirit
clone KhronosGroup/SPIRV-Headers vulkan-sdk-1.4.341.0 SPIRV-Headers
mkdir -p boost boost-include
for lib in container intrusive move config assert core static_assert type_traits \
           throw_exception predef unordered container_hash mp11 describe utility; do
    clone "boostorg/$lib" boost-1.91.0 "boost/$lib"
    cp -r "boost/$lib/include/boost" boost-include/
done
echo "shader sources in $ROOT"
