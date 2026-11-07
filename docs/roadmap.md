# Roadmap

Each phase completes only on the evidence named here, not when code exists.

| Phase | Completion evidence | State |
|---|---|---|
| 0. Reuse audit | Reuse matrix and architecture decision ([research](research.md)). | Done 2026-11-06. |
| 1. Repository and portable foundation | Workspace builds; tests pass on Linux and Windows CI. | Done 2026-11-06. |
| 2. Content processing | The supplied NUP is validated, inventoried and its ExeFS cached without manual extraction; NUP and directory input normalize identically. | Done for the checked input. RomFS base/update overlay not yet needed. |
| 3. Compiler foundation | Synthetic guest functions translated and executed natively with verified semantics; missing-block reports. | Done 2026-11-06. Synthetic programs run natively on Linux and Windows; the whole game (17.3M instructions, 96.6% of `main`'s text discovered) is translated and runs on Linux. |
| 4. Guest runtime | Translated startup reaches a recorded runtime milestone. | Done 2026-11-06. Translated startup runs through boot, the system services and GPU setup to presented frames; runs end with a JSON report. |
| 5. Graphics and resolution foundations | Synthetic workloads at arbitrary sizes; safe resizes. | In progress. NVN runs on Vulkan with shaders translated to SPIR-V; the internal render scale from the settings is applied. The resolution planner is not yet connected to the renderer. |
| 6. First real game frame | A correct game-generated frame. | In progress. The title logo and the intro movie render; frames are checked from captures (`ECHORA_CAPTURE_EVERY`). |
| 7. Input, audio, interactive gameplay | Interactive scene with graphics, input and audio. | In progress. Keyboard and gamepad input reach the game and its audio plays (`ECHORA_AUDIO=1`); gameplay has not been reached. |
| 8. Native resolution support | Internal render size verified at several arbitrary resolutions. | — |
| 9. Launcher (GPUI) | Launcher configures and starts Echora; exposed settings work. | In progress. Installs the game, edits and saves the settings, starts the game, and works with mouse, keyboard and controllers. Play starts the game with its window and sound. |
| 10. In-game settings | Dropped: settings are changed in the launcher, with no in-game interface. | — |
| 11. Full-game compatibility | The game can be completed without known major failures. | — |
| 12. High frame rates | No timing, physics or animation regressions. | — |
| 13. Performance | Reproducible benchmarks show gains without regressions. | In progress. Speed is measured by movie pictures decoded per second (`ECHORA_MOVIE_RATE`), about 17 once the movie plays, with frames checked against baseline captures. Done so far: contiguous texture swizzling, compute storage on the GPU, guest memory imported into Vulkan, unwaited frame submits, host zstd, shorter scheduler slices, spin-wait handoff, vector FP fast paths. Code-generation benchmarks without the game: `cargo bench -p echora-aot-tests --bench kernels`. `ECHORA_GPU_TIME=1` prints GPU busy time against wall time: during the movie the GPU is busy about 40% of the time and the CPU's main thread is the limit. Several workers are not yet stable. |
| 14. Other platforms | Real game code on another OS or CPU without core rewrites. | — |
| 15. Release quality | Clean build, configure, launch with user-supplied data. | — |

## Next steps

1. Make several workers stable (`ECHORA_WORKERS`): runs crash once the
   game's per-core job threads run in parallel, which looks like one job
   claimed by two threads. This is the largest speed gain available.
2. Add the mesh and task draw path (`DrawMeshTasks`, `VK_EXT_mesh_shader`)
   that gameplay scenes need, and advance to gameplay.
3. Translate the remaining failing shader stages (stage-buffer reads
   outside task and mesh programs, gather forms).
4. Reduce the cost of dispatching between translated blocks (structured
   code generation), the largest remaining single-thread cost.
