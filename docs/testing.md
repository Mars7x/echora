# Testing

`cargo test --workspace` needs no game files. It covers:

| Crate | Coverage |
|---|---|
| `echora-nup` | Synthetic archives built by a test-only writer (`tests/common`): raw and zstd extents, shared extents, navigation and lookup, meta records, extraction, parallel verification; rejection of a tampered TOC, UUID mismatch, truncated or extended files, unknown codecs, out-of-bounds or misaligned extents, names outside the pool, path-traversal names, and an update without a base. Corrupted extent bytes are detected by both verification and reads. |
| `echora-content` | NSO and NPDM header parsing; flat and layered dump directories; numeric update ordering; program-package selection; content-addressed ExeFS caching, reuse, repair of a damaged entry and re-keying on changed input. |
| `echora-loader` | ZBIC step-table generation; hand-derived BIC descriptions (including low-probability counts); standard table descriptions read back unchanged; byte-identical re-transcoding of libzstd frames at four levels over seven inputs (raw, RLE, compressed and treeless literals, FSE, RLE, predefined and repeat sequence modes); truncated and trailing-data frames. NSO parsing: LZ4 modules, hash mismatch, bad sizes, overlapping segments, header overlap, image budget, flag handling, ExeFS load order and required modules. MOD0: signed relative offsets, unterminated dynamic tables, out-of-range relocation tables, bounded GNU-hash chain walks, GNU-only symbol tables with RELR and exported-symbol lookup, unknown tags. |
| `echora-guest` | Pokopia's guest suite: CPU register/flag helpers, PAuth stripping, SIMD/FP semantics and fast paths, guest memory mappings, permissions, aliases, atomics and the guest view (memory-backed tests on Linux only); every guest view mapping is checked to be excluded from core dumps. |
| `echora-lift` | Pokopia's lifter suite: decoding, discovery, jump tables, block construction and emitted Rust for each instruction family. |
| `echora-aot-tests` | Programs lifted at build time and executed natively: integer and flag results, call/return with and without PAuth hints, step budget, unsupported system registers, undiscovered indirect targets, calls across dispatch windows and holes inside them, CAS/LDADD at every width including register aliasing and SP forms, load/store-exclusive loops and monitor clearing, LDAR/LDAPR/STLR, alignment faults, and a store to an unmapped page ending the process with the guest address (in a child process with core dumps disabled). |
| `echora-resolution` | Non-integer scales, internal resolution independent of output, exact-size aspect checks, ultrawide pillarboxing and view expansion (21:9, 32:9), 16:10, seamless viewport tiling at 1366x768, fixed versus screen-relative targets, change impact, degenerate sizes. |
| `echora-config` | Round trip, partial files, unknown-field and range rejection, future schema rejection, change classification, requested-versus-effective resolution, atomic save. |

Ignored tests, run with `ECHORA_TEST_NUP` set, open the real archive, verify
every ExeFS file, and check that NUP and directory inputs produce the same
ExeFS key and build ID. With `ECHORA_TEST_EXEFS` set to a cached ExeFS
directory, the loader test decompresses every module, requires every
segment to be hash-checked and every module to carry MOD0 dynamic metadata
with a counted symbol table.

CI runs formatting, clippy with warnings denied, and the tests on Linux
x86-64, Windows x86-64 and macOS ARM64. Passing CI means the code compiles
and the tests pass on that host; it says nothing about running the game.
