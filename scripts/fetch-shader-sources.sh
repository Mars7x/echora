#!/usr/bin/env bash
# Fetch the pinned shader recompiler sources (GPL-3.0-or-later, Lichii,
# an Eglantine/Eden fork) and its header dependencies into $ECHORA_SHADER_SRC
# (default ~/.cache/echora/shader-src).
# Nothing here enters Git.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${ECHORA_SHADER_SRC:-$HOME/.cache/echora/shader-src}"
mkdir -p "$ROOT"
cd "$ROOT"

LICHII_URL=https://github.com/zortos293/Lichii.git
LICHII_COMMIT=809fb3aa35f4ca5973ceb2f0c3de85e4e0c624d7
if [ ! -d lichii/.git ]; then
    git init -q lichii
    git -C lichii remote add origin "$LICHII_URL"
    git -C lichii sparse-checkout set --no-cone /src/shader_recompiler /src/common /LICENSE.txt
fi
git -C lichii fetch -q --depth 1 --filter=blob:none origin "$LICHII_COMMIT"
git -C lichii -c advice.detachedHead=false checkout -q --force FETCH_HEAD
# Our extensions to the lowering, kept as patches against the pin.
for patch in "$REPO"/crates/echora-gpu/native/shader/patches/*.patch; do
    git -C lichii apply "$patch"
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
