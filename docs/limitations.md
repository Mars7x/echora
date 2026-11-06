# Known limitations

- No game code is loaded, translated or executed yet.
- The NSO loader does not exist yet; `main` uses ZBIC segments that need a
  new decoder (the only public implementation is GPLv2-only).
- Update RomFS is not overlaid on base RomFS. The checked OoT input is a
  standalone update whose RomFS is used whole; its completeness is not
  verified against a base.
- Directory input hashes ExeFS modules but only counts RomFS files.
- `echora-resolution` and `echora-config` are not connected to a renderer,
  runtime or launcher. View expansion (`AspectPolicy::Expand`) will need
  game-specific projection and UI changes before it can be offered.
- CI covers Linux x86-64, Windows x86-64 and macOS ARM64 compilation and
  tests only.
