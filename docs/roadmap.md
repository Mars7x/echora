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
| 5. Graphics and resolution | Synthetic workloads run at arbitrary sizes and resize safely. | In progress. NVN runs on Vulkan with translated shaders, and the internal render scale is applied; the resolution planner is not connected yet. |
| 6. First game frame | A correct frame drawn by the game. | Done 2026-11-06. The title screen and the opening movie render correctly. |
| 7. Interactive gameplay | A playable scene with graphics, input and audio. | In progress. Input reaches the game and its audio plays; the game gets past the New Game movie and draws its first 3D scene, still overexposed. |
| 8. Native resolution | The internal render size is verified at several resolutions. | Not started. |
| 9. Launcher | The launcher configures and starts Echora, and its settings work. | In progress. It installs the game (as a clone or hard link where possible), edits and saves settings, prepares the translation and starts the game, with mouse, keyboard and controller. |
| 10. In-game settings | Dropped: settings are changed in the launcher only. | — |
| 11. Full-game compatibility | The game can be completed without known major failures. | Not started. |
| 12. High frame rates | Frame rates above 60 with no timing, physics or animation problems. | Not started. |
| 13. Performance | Reproducible measurements show gains without regressions. | In progress. The opening movie decodes at about 22 to 24 pictures per second (`ECHORA_MOVIE_RATE`), with the main CPU thread as the limit. |
| 14. Other platforms | Real game code runs on another OS or CPU without rewriting the core. | Not started. |
| 15. Release quality | A clean build, setup and launch with user-supplied data. | Not started. |

## Next steps

1. Correct the overexposed first scene after the New Game movie, check what
   it draws (including the mesh and task pipelines) and carry on to gameplay.
2. Translate the last 16 failing shader stages.
3. Finish multi-core (`ECHORA_WORKERS`): runs no longer crash, so check long
   runs against single-core ones, cut the time workers spend waiting for the
   global machine lock, and give each guest core its own run queue inside a
   worker. Then make it the default.
4. Turn on the faster code generation in game builds once it is checked in
   game: inline vector arithmetic first, then structured control flow and
   direct calls once they measure faster.
5. Shorten the first build from about an hour: leaner generated code (about
   a third smaller in tests so far), full optimisation only for the hot code a
   profile finds, and more parallel compiler jobs.
6. Keep compressed textures compressed on the GPU. The game's ASTC textures
   are unpacked to plain RGBA8 because PC GPUs mostly can't sample ASTC, which
   uses four to eight times the video memory (one 90-layer array after the
   opening movie becomes 852 MiB). Re-encoding them as BC7, cached on disk like
   the shader binaries, would bring that close to the console's footprint.
