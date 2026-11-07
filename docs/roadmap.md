# Roadmap

Each phase completes only on the evidence named here, not when code exists.

| Phase | Completion evidence | State |
|---|---|---|
| 0. Reuse audit | Reuse matrix and architecture decision ([research](research.md)). | Done 2026-11-06. |
| 1. Repository and portable foundation | Workspace builds; tests pass on Linux, Windows and macOS CI. | Done 2026-11-06. |
| 2. Content processing | The supplied NUP is validated, inventoried and its ExeFS cached without manual extraction; NUP and directory input normalize identically. | Done for the checked input. RomFS base/update overlay not yet needed. |
| 3. Compiler foundation | Synthetic guest functions translated and executed natively with verified semantics; missing-block reports. | Done 2026-11-06. Synthetic programs run natively on Linux, Windows and macOS; the whole game (17.3M instructions, 96.6% of `main`'s text discovered) is translated and runs on Linux. |
| 4. Guest runtime | Translated startup reaches a recorded runtime milestone. | Done 2026-11-06. Translated startup runs through boot, the system services and GPU setup to presented frames; runs end with a JSON report. |
| 5. Graphics and resolution foundations | Synthetic workloads at arbitrary sizes; safe resizes. | In progress. NVN runs on Vulkan with shaders translated to SPIR-V; the internal render scale from the settings is applied. The resolution planner is not yet connected to the renderer. |
| 6. First real game frame | A correct game-generated frame. | In progress. The title logo and the intro movie render; frames are checked from captures (`ECHORA_CAPTURE_EVERY`). |
| 7. Input, audio, interactive gameplay | Interactive scene with graphics, input and audio. | In progress. Keyboard and gamepad input reach the game and its audio plays (`ECHORA_AUDIO=1`); gameplay has not been reached. |
| 8. Native resolution support | Internal render size verified at several arbitrary resolutions. | — |
| 9. Launcher (GPUI) | Launcher configures and starts Echora; exposed settings work. | In progress. Installs the game, edits and saves the settings, starts the game, and works with mouse, keyboard and controllers. Play starts the game with its window and sound. |
| 10. In-game settings | Dropped: settings are changed in the launcher, with no in-game interface. | — |
| 11. Full-game compatibility | The game can be completed without known major failures. | — |
| 12. High frame rates | No timing, physics or animation regressions. | — |
| 13. Performance | Reproducible benchmarks show gains without regressions. | — |
| 14. Other platforms | Real game code on another OS or CPU without core rewrites. | — |
| 15. Release quality | Clean build, configure, launch with user-supplied data. | — |

## Next steps

1. Confirm with frame captures that the Lichii shader recompiler, which
   translates the compute stages PokeDinner skipped, renders the intro
   movie's colours and the menus correctly.
2. Translate the remaining failing shader stages (stage-buffer reads
   outside task and mesh programs, gather forms).
3. Add the mesh and task draw path (`DrawMeshTasks`, `VK_EXT_mesh_shader`)
   that gameplay scenes need, and advance to gameplay.
4. Measure and raise the frame rate, starting with guest threads on several
   host threads (`ECHORA_WORKERS`).
