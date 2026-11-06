# Roadmap

Each phase completes only on the evidence named here, not when code exists.

| Phase | Completion evidence | State |
|---|---|---|
| 0. Reuse audit | Reuse matrix and architecture decision ([research](research.md)). | Done 2026-11-06. |
| 1. Repository and portable foundation | Workspace builds; tests pass on Linux, Windows and macOS CI. | Done 2026-11-06. |
| 2. Content processing | The supplied NUP is validated, inventoried and its ExeFS cached without manual extraction; NUP and directory input normalize identically. | Done for the checked input. RomFS base/update overlay not yet needed. |
| 3. Compiler foundation | Synthetic guest functions translated and executed natively with verified semantics; missing-block reports. | In progress. Loader (ZBIC, MOD0) verified on all modules. Guest crate and lifter ported; synthetic programs (integer, flags, calls, atomics, exclusives, acquire/release, window dispatch, missing blocks) run natively on Linux and Windows. Discovery on the real modules reaches 96.6% of `main`'s text (17.3M instructions); every discovered form except one `stg` is supported. |
| 4. Guest runtime | Translated startup reaches a recorded runtime milestone. | In progress. Runtime core ported (231 tests); the real process boots (modules, stack, TLS, NPDM capabilities). Running translated startup is next. |
| 5. Graphics and resolution foundations | Synthetic workloads at arbitrary sizes; safe resizes. | Resolution planner exists. |
| 6. First real game frame | A correct game-generated frame. | — |
| 7. Input, audio, interactive gameplay | Interactive scene with graphics, input and audio. | — |
| 8. Native resolution support | Internal render size verified at several arbitrary resolutions. | — |
| 9. Launcher (GTK4/libadwaita) | Launcher configures and starts Echora; exposed settings work. | Settings model exists. |
| 10. In-game settings | Supported settings change during play without corrupting state. | Apply classes defined. |
| 11. Full-game compatibility | The game can be completed without known major failures. | — |
| 12. High frame rates | No timing, physics or animation regressions. | — |
| 13. Performance | Reproducible benchmarks show gains without regressions. | — |
| 14. Other platforms | Real game code on another OS or CPU without core rewrites. | — |
| 15. Release quality | Clean build, configure, launch with user-supplied data. | — |

## Next steps

1. Generate a runner with each translation, build the full translation, and
   run translated rtld from the process entry until its first stop.
2. Advance startup through each reported stop.
