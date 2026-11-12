# Roadmap

Each phase counts as done only when the evidence named for it exists, not
when the code for it does.

| Phase | Done when | State |
|---|---|---|
| 0. Reuse audit | The reuse matrix and architecture decision are written ([research](research.md)). | Done 2026-11-06. |
| 1. Portable foundation | The workspace builds and its tests pass in CI on Linux and Windows. | Done 2026-11-06. |
| 2. Content processing | The NUP is validated and inventoried, and its ExeFS cached, without manual extraction; NUP and folder input give the same result. | Done for the supported input. Overlaying update RomFS on a base is not needed yet. |
| 3. Compiler foundation | Synthetic programs are translated and run natively with verified results, and missing blocks are reported. | Done 2026-11-06. The whole game (17.3 million instructions) is translated and runs on Linux. |
| 4. Guest runtime | Translated startup reaches a recorded milestone. | Done 2026-11-06. Startup runs through boot, the system services and GPU setup to presented frames. |
| 5. Graphics and resolution | Synthetic workloads run at arbitrary sizes and resize safely. | In progress. NVN runs on Vulkan with Echora's own Rust shader translator, and the internal render scale is applied; the resolution planner is not connected yet. |
| 6. First game frame | A correct frame drawn by the game. | Done 2026-11-06 with the earlier C++ shader recompiler, and again on 2026-11-12 with Echora's own shader translator alone. |
| 7. Interactive gameplay | A playable scene with graphics, input and audio. | In progress. Input reaches the game; it gets past the New Game movie and draws its first 3D scene close to the console's, with a few faults left ([limitations](limitations.md)). There is no sound yet. |
| 8. Native resolution | The internal render size is verified at several resolutions. | Not started. |
| 9. Launcher | The launcher configures and starts Echora, and its settings work. | In progress. It installs the game (as a clone or hard link where possible), edits and saves settings, prepares the translation and starts the game, with mouse, keyboard and controller. |
| 10. Full-game compatibility | The game can be completed without known major failures. | Not started. |
| 11. High frame rates | Frame rates above 60 with no timing, physics or animation problems. | Not started. |
| 12. Performance | Reproducible measurements show gains without regressions. | In progress. With six cores and the movies decoded on the GPU, the opening movie averages about 30 pictures per second (`ECHORA_MOVIE_RATE`) and the first 3D scene runs at about 2 frames per second now that all of it is drawn. |
| 13. Other platforms | Real game code runs on another OS or CPU without rewriting the core. | Not started. |
| 14. Release quality | A clean build, setup and launch with user-supplied data. | Not started. |

## Next steps

1. Finish the first scene after the New Game movie: the placeholder pattern
   on the tree's roots, the tree's brightness, subtitles, and the frames
   that come out black. Then carry on to gameplay.
2. Sound: mix and play the game's audio renderer (voices, mixes, effects
   and the output device).
3. Speed, until the opening movie and the first scene run at the console's
   60 frames per second: fewer and cheaper GPU submissions, less work per
   NVN command, then the cost of the translated code.
4. Turn on the faster code generation in game builds once it is checked in
   game: inline vector arithmetic, branches that skip unused flags and
   floating-point status kept in locals, then structured control flow once
   it measures faster. They go in with the next full rebuild.
5. Shorten the first build from about an hour: leaner generated code (about
   a third smaller in tests so far), full optimisation only for the hot code a
   profile finds, and more parallel compiler jobs.
6. Keep compressed textures compressed on the GPU. The game's ASTC textures
   are unpacked to plain RGBA8 because PC GPUs mostly can't sample ASTC, which
   uses four to eight times the video memory (one 90-layer array after the
   opening movie becomes 852 MiB). Re-encoding them as BC7, cached on disk like
   the shader binaries, would bring that close to the console's footprint.
7. Translate and compile every shader before the game starts, so nothing
   stutters while it runs. `echora shaders` already lists all 34,275 unique
   shader stages the game ships.
8. Run the opening movie's neural-network filter on the GPU's matrix units
   (cooperative matrices), its largest remaining cost per picture.
