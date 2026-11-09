# Echora

[![CI](https://img.shields.io/github/actions/workflow/status/Mars7x/echora/ci.yml?branch=main&style=flat-square&label=CI)](https://github.com/Mars7x/echora/actions/workflows/ci.yml)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue?style=flat-square)](LICENSE)
![Written in pure Rust](https://img.shields.io/badge/written%20in-pure%20Rust-orange?style=flat-square)
![Status: experimental](https://img.shields.io/badge/status-experimental-red?style=flat-square)

Echora is a native recompilation of the Nintendo Switch 2 release of *The
Legend of Zelda: Ocarina of Time*. Rather than emulating the console, it
translates the game's ARM code into Rust ahead of time, compiles it to native
code, and runs it on a reimplementation of the Switch's operating system and
graphics.

> [!CAUTION]
> Echora cannot be used to play Ocarina of Time yet. It does not reach
> gameplay, runs well below full speed, and can hang, crash or render
> incorrectly at any point. Saves are untested. Please do not report gameplay
> problems as bugs for now.

Echora is a personal project that I build for my own enjoyment. It doesn't
include any game code, assets or keys, so you'll need your own copy of the
game.

> [!IMPORTANT]
> Echora is made with the help of AI.

## Status

In short: Echora runs the game's code, but on its own it can't draw anything
yet, because I'm in the middle of replacing its shader translator.

What works today:

- Echora reads and checks your NUP file or dump, and translates the whole
  game, 17.3 million instructions, into native code. The first start takes
  about an hour; after that it starts straight away.
- The translated game runs: it boots through the system services, plays its
  sound, takes input, and gets through New Game and the opening movie into
  its first 3D scene.
- The game's threads run on up to six CPU cores by default.
- The launcher installs the game, changes settings and starts it, with a
  mouse, keyboard or controller.

What doesn't work yet:

- Drawing. Echora's own shader translator, written in Rust, handles more than
  half of the game's compute shaders so far and none of its drawing shaders,
  so a run stops at the first shader it can't translate and says which one.
  With the old shader recompiler the title screen and the opening movie
  rendered correctly; I still load it on my own machine as a plugin to test
  against. The section below explains why I'm replacing it.
- Gameplay. The first 3D scene after the movie isn't drawn correctly, and
  nothing past it has been reached.
- Speed. The console runs the game at 60 frames per second. Here the opening
  movie plays at about 22 to 29 pictures per second, limited by video
  decoding, and the first 3D scene at about 6.
- Platforms. The game only runs on Linux at the moment. The rest of the
  project builds and passes its tests on Windows too.

These figures come from update 1.1.1 (`0400486024DEC000` v131072) on an AMD
RX 9070 XT. The [roadmap](docs/roadmap.md) and
[known limitations](docs/limitations.md) go into more detail.

## What I'm working on

The game's GPU programs are written for the Switch 2's NVIDIA GPU and have to
be translated into shaders your PC's GPU can run. Until now Echora used a C++
shader recompiler from the Switch emulator world. It got the title screen and
the opening movie on screen, but after the movie it turned out to be the
source of the hardest bugs so far: it guessed where shaders keep their data,
and the wrong guesses fed the game's own programs bad data and hung the GPU.
So I'm writing Echora's translator in Rust, for four reasons:

- Shaders reach the game's real memory through a page table, the way the
  console's GPU sees it, instead of guesses.
- Every shader can be translated without running the game, so they can all
  be prepared before it starts and nothing stutters while you play.
- The PC's own hardware, such as the matrix units the game's upscaler needs,
  can be used directly instead of being emulated step by step.
- The whole project shares one language and one design.

It is a new design, with the details of each instruction checked against
public references and every translation compared with the old recompiler on
my own machine. The old recompiler is no longer part of Echora, but it can
still be loaded as a separate plugin for testing.

Once the title screen, the movie and the first 3D scene render through the new
translator, the next steps are getting that scene right, reaching gameplay,
and making each CPU thread do less work than it does now.

## Approach

Echora is not an emulator. The game's ARM code is translated into native code
before it runs, so nothing interprets or recompiles instructions while you
play. The console's hardware is not simulated either: Echora implements the
operating system and graphics API the game calls, and draws with your PC's own
GPU through Vulkan.

I want Echora to behave like the console rather than work around it. It
reimplements the Switch's kernel, system services and NVN graphics API and
leaves the game's own code alone. When the game asks for something that is not
supported yet, the run stops and reports exactly what was missing instead of
pretending it worked. The translated code is tested against a reference
interpreter written from Arm's specification, and the game is read straight
from your NUP file or dump without unpacking it.

Echora is written in pure Rust. Like any PC program it uses the system's own
libraries for graphics, sound, input and windows, and its AV1 video decoder
keeps hand-written assembly for speed. Anything that goes beyond what the
console does, such as frame rates above 60 or ultrawide, will be kept in a
separate layer of enhancements that can be turned off.

## How it works

1. Echora reads your NUP file or dump, verifies it and loads the game's
   executables.
2. It discovers the game's ARM code and emits it as Rust, one crate per 64 KiB
   region, which is then compiled to native code.
3. The running game talks to Echora's implementation of the Switch kernel and
   system services.
4. NVN graphics calls are replayed on Vulkan, with the console's shaders
   translated to SPIR-V by Echora's own translator, and movies are decoded on
   the PC.

See the [architecture notes](docs/architecture.md) for how the crates fit
together.

## Building

### Requirements

- Linux
- Your own dump of the game (update 1.1.1), as a NUP file or a folder
- A Vulkan GPU, about 16 GB of RAM and about 30 GB of free disk space
- Rust 1.88 or newer, `git`, C and C++ compilers, NASM and these system
  libraries:

```sh
# Ubuntu
sudo apt install build-essential nasm libasound2-dev libudev-dev libxkbcommon-x11-dev libwayland-dev libfontconfig-dev libx11-xcb-dev libvulkan-dev
# Fedora
sudo dnf install gcc gcc-c++ nasm alsa-lib-devel systemd-devel libxkbcommon-x11-devel wayland-devel fontconfig-devel libxcb-devel vulkan-loader-devel
```

### Getting started

```sh
git clone https://github.com/Mars7x/echora.git
cd echora
cargo build --release
target/release/echora-launcher
```

Choose your game and press Play. The first start translates and builds the
whole game, which takes about an hour. After that it starts straight away.

## Documentation

- [Architecture](docs/architecture.md): the crates and how they fit together
- [Development](docs/development.md): command-line tools and test runs
- [Content input](docs/content.md): NUP files and dumps
- [Roadmap](docs/roadmap.md): milestones and next steps
- [Research notes](docs/research.md): prior work Echora builds on
- [Testing](docs/testing.md): the test suites
- [Known limitations](docs/limitations.md): what does not work yet

## Legal

Echora is an independent project and is not affiliated with or endorsed by
Nintendo or NVIDIA. This repository contains only original source code, tests
and documentation. It does not include or download game files, keys or
firmware, so please do not open issues or pull requests that contain them or
code generated from them. Translated game code stays on your machine.

Echora is licensed under the GNU General Public License, version 3 or later.
See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).
