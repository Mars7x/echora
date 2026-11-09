# AV1 decoding on the GPU

The game's movies are AV1. Echora decodes them on the CPU with rav1d, and is
moving them to the GPU's video engine through Vulkan Video
(`VK_KHR_video_decode_av1`), with rav1d kept as the fallback. Vulkan Video
decodes tiles, but the application parses the headers and keeps track of
references. `crates/echora-av1` is that half: plain Rust, no GPU, no
`unsafe`.

## Design

```text
submission bytes ──► Parser::parse_temporal_unit ──► Vec<Output>
                         │                              ├─ Decode(DecodePicture)
                         │ OBUs (obu.rs)                ├─ ShowExisting(ShowExisting)
                         │ sequence header (sequence.rs)└─ Metadata(Metadata)
                         │ frame header (frame.rs)
                         │ tile groups (parser.rs)
                         └ eight reference slots (refs.rs)
```

- `bits.rs`: the spec's descriptors (f(n), su(n), ns(n), le(n), leb128(),
  uvlc(), the sub-exponential codes of global motion).
- `obu.rs`: OBU headers and extension headers, `obu_size`, the operating
  point's layer selection, and Annex B length-delimited streams.
- `sequence.rs`: `sequence_header_obu()` in full (section 5.5): timing and
  decoder model info, operating points, the tool flags, color config.
- `frame.rs`: `uncompressed_header()` in full (section 5.9) with the
  processes it depends on: `set_frame_refs()` (short reference signaling),
  superres and image size, `frame_size_with_refs()`, `mark_ref_frames()`,
  tile info, quantisation, segmentation (with `get_qidx()`, lossless and
  qmatrix levels), delta q and delta lf, loop filter, CDEF, loop
  restoration, tx mode, reference mode, skip mode, global motion and film
  grain.
- `refs.rs`: the eight reference slots and the state the spec carries
  between frame headers: RefValid, RefFrameId, RefFrameType, sizes,
  RefOrderHint and SavedOrderHints, saved global motion, loop filter
  deltas, segmentation features and film grain, plus OrderHints and
  RefFrameSignBias. The reference frame update process (7.20) runs when a
  frame's last tile arrives; `show_existing_frame` of a key frame runs the
  loading process (7.21) and refreshes every slot.
- `parser.rs`: the streaming entry point. A frame may arrive as one frame
  OBU or as a frame header OBU and several tile group OBUs. Each tile's
  data is located (offset and size, without its `tile_size_minus_1` field)
  but not decoded. An error drops the frame in progress and leaves the
  reference state as it was, so a later key frame recovers.
- `video.rs`: plain-data mirrors of the `StdVideoAV1*` and
  `StdVideoDecodeAV1*` structures. Bitfield flags are plain bools and
  pointers are owned values; enums carry their Std numeric values.
- `ivf.rs`: an IVF reader, for the test streams.
- `examples/av1_headers.rs`: prints every picture's header and what
  changes from one picture to the next
  (`cargo run -p echora-av1 --example av1_headers -- FILE.ivf|FILE.obu`).

CDFs, motion vectors and segment maps are decoder state that lives on the
GPU; the parser never needs them.

## Using it

```rust
let mut parser = echora_av1::Parser::new();
for output in parser.parse_temporal_unit(&submission)? {
    match output {
        Output::Decode(picture) => { /* record a decode */ }
        Output::ShowExisting(show) => { /* present show.id again */ }
        Output::Metadata(_) => {}
    }
}
```

Every byte offset in the output is into the buffer that was passed in, so
the simplest bitstream buffer is the whole submission.

## Vulkan Video mapping

| Vulkan | echora-av1 |
|---|---|
| `StdVideoAV1SequenceHeader` (+ `ColorConfig`, `TimingInfo`) | `DecodePicture::std_sequence` (`video::SequenceHeader`); recreate the session parameters when `sequence_generation` changes |
| `StdVideoDecodeAV1PictureInfo` and its sub-structures | `DecodePicture::std_picture` (`video::PictureInfo`) |
| `pTileInfo` arrays | `tile_info.{mi_col_starts, mi_row_starts, width_in_sbs_minus_1, height_in_sbs_minus_1}` |
| `pFilmGrain` | `std_picture.film_grain` (`None` when no grain applies) |
| `VkVideoDecodeAV1PictureInfoKHR::referenceNameSlotIndices` | `reference_name_slot_indices` gives the AV1 slot (0..8) for LAST..ALTREF, -1 on intra frames; map slots to DPB slots through `reference_pictures` / `slot_pictures` |
| `frameHeaderOffset` | `frame_header_offset` (the OBU that holds the frame header) |
| `tileCount`, `pTileOffsets`, `pTileSizes` | `tiles`, `tile_offsets()`, `tile_sizes()` |
| `StdVideoDecodeAV1ReferenceInfo` of each active reference | `reference_infos[slot]` |
| `StdVideoDecodeAV1ReferenceInfo` of the setup slot | `setup_reference_info` |
| `codedExtent` | `frame_width` x `frame_height` (`upscaled_width` is the width after superres) |

Notes on values:

- `order_hints`, `expected_frame_id`, `gm_type` and `gm_params` are indexed
  by reference name: entry 0 (INTRA_FRAME) is unused and LAST_FRAME is 1.
  The Std headers size them 8 for that reason. Check `expected_frame_id`'s
  indexing against the driver; it only matters with frame ids, which the
  game's movies do not use.
- `video::Cdef` carries the coded secondary strengths (0..=3), as drivers
  expect (RADV packs `(pri << 2) + sec`); `frame::Cdef` has the spec's
  values, where a coded 3 means 4.
- `video::LoopRestoration::log2_restoration_size_minus_5` is Vulkan's
  `LoopRestorationSize`: log2 of the unit size minus 5 (0..=3), zero when
  loop restoration is off. `frame::LoopRestoration` has the size in
  samples.
- `pMiColStarts` and `pMiRowStarts` are in 4x4 (MI) units, as the Vulkan
  spec and FFmpeg have them. RADV 25.2 copies them into fields named for
  superblock units, so multi-tile streams on RADV deserve a check.
- `ar_coeffs_*_plus_128` are stored as coded (0..=255); the Std header
  declares them `int8_t`, so copy the bits.
- With `update_grain` clear the film grain parameters are already loaded
  from `film_grain_params_ref_idx`, with this frame's `grain_seed`.
- `ReferenceInfo::ref_frame_sign_bias` is a mask (bit i for reference
  name i) of the reference's own RefFrameSignBias.
- An error-resilient frame may declare a slot's order hint different from
  what the slot holds (`ref_order_hint`); the slot then becomes invalid.
  The spec has the decoder substitute a frame; the parser reports the slot
  as empty.

## The Vulkan side

`crates/echora-gpu/src/video.rs` and `video_av1.rs`:

- `VideoDevice` is a Vulkan device of its own with a decode queue and a
  copy queue, so movie decoding never shares a queue with the renderer.
  On RADV the decode family cannot copy.
- The session is AV1 Main, 8-bit 4:2:0, without film grain. The DPB is
  nine images, one per slot, which also hold the output pictures
  (`DPB_AND_OUTPUT_COINCIDE`). An array image would be one allocation,
  but RADV 25.2 misplaces the chroma plane of layers past the first.
- `MovieDecoder` keeps the DPB slots by `PictureId`, maps the AV1 slots
  of `reference_name_slot_indices` to DPB slots, and reads shown pictures
  back as NV12 at the guest's pitch. It refuses film grain and superres,
  so the caller falls back to rav1d.
- `ECHORA_VIDEO_DECODE=gpu` makes the game's movies decode this way
  (rav1d stays the default). If the device has no AV1 decode, or a decode
  fails, that movie continues on rav1d.
- `av1-gpu-check FILE.obu [PICTURES]` (in `echora-runtime`) decodes a raw
  stream (`ffmpeg -c:v copy -an -f obu`) both ways and compares the NV12
  planes; `AV1_CHECK_HIDDEN=1` compares hidden frames too (rav1d's
  invisible-frame output). All 26 of the game's movies, 83,449 pictures,
  decode identically, at 2 to 4 ms per 1080p picture on an RX 9070 XT.

## Tests

`cargo test -p echora-av1` runs:

- unit tests per header element on bitstreams written bit by bit in the
  tests (`src/tests.rs`, and the tests in each module);
- a cross-check against rav1d (`tests/rav1d_crosscheck.rs`): every
  picture rav1d outputs, hidden frames included, must match the parsed
  header in frame size, frame type, show flags, refresh flags, order hint,
  reference indices, tiles, quantisation, segmentation, delta q/lf, loop
  filter, CDEF, loop restoration, tx mode, skip mode, global motion and
  film grain. The AOM conformance vectors in `tests/vectors` are checked
  in (see NOTICE.md); `tests/fetch-vectors.sh DIR` downloads more, and
  `ECHORA_AV1_VECTORS=DIR` adds them to the run.

Unoptimised rav1d takes seconds per picture, so a debug `cargo test`
decodes only the smallest vectors with it (the parser still reads every
stream). Run the full cross-check with
`cargo test --release -p echora-av1`; it takes under a second and, with
the downloaded vectors, compares about 180 pictures from 31 streams.
