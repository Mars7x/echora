# Content input

Echora accepts the game as a NUP file or as an extracted dump directory.
Both go through `echora-content` and produce the same inventory and the same
ExeFS cache entry, so nothing downstream depends on the input form.

## NUP

[NUP](https://github.com/a55uka/nup) is a container of plaintext ExeFS and
RomFS trees for base, update and add-on packages. No keys are involved.
Opening a NUP:

- checks the superblock, the TOC SHA-256 and the content-derived UUID;
- bounds-checks every package, directory and file record, string reference
  and extent, and rejects overlapping extents unless they are identical;
- accepts only plain UTF-8 path components as names (no `.`, `..`, `/`,
  `\` or NUL), so extraction cannot leave its destination;
- requires strictly sorted children and acyclic package trees;
- rejects unknown codecs, kinds and flags.

Files are read in place. zstd extents are streamed; a reader fails if the
decoded length or SHA-256 differs from the record. The TOC is capped at
512 MiB and meta blocks at 64 MiB.

## Directories

Accepted layouts match the ones `nup pack` consumes:

```
dump/exefs/ dump/romfs/                     a single base package
dump/base/{exefs,romfs}/                    base
dump/updates/<decimal version>/{exefs,romfs}/
dump/dlc/<16-hex title id>/<index>/{exefs,romfs}/
```

Symlinks are not followed. ExeFS modules are hashed during inventory; RomFS
files are counted but not hashed.

## Inventory

`echora inspect` reports each package's kind, title ID, version, display
version, ExeFS modules (size, SHA-256, NSO segment sizes and flags, NPDM
program ID) and RomFS size. It also selects the **program package**: the
newest update with an ExeFS, else the base. Executables are never mixed
across packages. Notes flag conditions that matter, such as an update
supplied without its base or NSO flags beyond the classic ones.

## Cache

`echora import` places the program package's ExeFS in
`<cache>/exefs/<key>/`, where `key` is SHA-256 over the sorted module names
and their SHA-256 values. An existing entry is reused when its manifest and
file sizes match; otherwise it is rebuilt in a staging directory and renamed
into place. The cache root is `ECHORA_CACHE_DIR`, else
`$XDG_CACHE_HOME/echora` (Linux) or
`%LOCALAPPDATA%\Echora\cache` (Windows). It is never inside the repository.

## Updates and DLC

Layering is deterministic: an update's ExeFS replaces the base's entirely,
and an update's RomFS is expected to be a complete tree unless a base is
present. Overlaying update RomFS on base RomFS is not implemented yet; the
checked OoT input is a standalone update whose RomFS is used as is.
