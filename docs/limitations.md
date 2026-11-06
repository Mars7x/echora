# Known limitations

- Game code is loaded and verified but not yet translated or executed.
  Synthetic AArch64 programs are lifted and executed natively.
- Guest memory works on Linux with 4 KiB pages only. Windows and macOS build
  and run the non-memory tests but cannot create guest memory yet.
- A guest access to unmapped or protected memory ends the process (with the
  guest address reported) instead of returning a recoverable trap.
- NSO flag bit 6 (`0x40`) is set in `main` alongside the ZBIC bit; its meaning
  is unknown. It is accepted only together with ZBIC, and segment hashes
  confirm the decoded bytes. A ZBIC frame whose re-encoded tables no longer
  fit their size fields is rejected rather than decoded; this has not occurred
  with the real modules.
- Update RomFS is not overlaid on base RomFS. The checked OoT input is a
  standalone update whose RomFS is used whole; its completeness is not
  verified against a base.
- Directory input hashes ExeFS modules but only counts RomFS files.
- `echora-resolution` and `echora-config` are not connected to a renderer,
  runtime or launcher. View expansion (`AspectPolicy::Expand`) will need
  game-specific projection and UI changes before it can be offered.
- CI covers Linux x86-64, Windows x86-64 and macOS ARM64 compilation and
  tests only.
