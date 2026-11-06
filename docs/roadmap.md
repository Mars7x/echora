# Roadmap

Each phase completes only on the evidence named here, not when code exists.

| Phase | Completion evidence | State |
|---|---|---|
| 0. Reuse audit | Reuse matrix and architecture decision ([research](research.md)). | Done 2026-11-06. |
| 1. Repository and portable foundation | Workspace builds; tests pass on Linux, Windows and macOS CI. | Done 2026-11-06. |
| 2. Content processing | The supplied NUP is validated, inventoried and its ExeFS cached without manual extraction; NUP and directory input normalize identically. | Done for the checked input. RomFS base/update overlay not yet needed. |
| 3. Compiler foundation | Synthetic guest functions translated and executed natively with verified semantics; missing-block reports. | In progress. NSO loader with ZBIC done and verified on all modules; MOD0 and the lifter are next. |
| 4. Guest runtime | Translated startup reaches a recorded runtime milestone. | — |
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

1. Port MOD0 and dynamic-section analysis from Pokopia into `echora-loader`,
   including `DT_GNU_HASH`-only symbol tables.
2. Port Pokopia's AArch64 decoder, lifter and guest crate into Echora with
   their synthetic tests, splitting Linux-specific host memory behind a
   backend trait.
3. Measure translation and build cost for OoT's 63 MB `main` text before
   committing to the generated-crate layout.
