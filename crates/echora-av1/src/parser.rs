//! The streaming parser: OBUs in, per-picture decode descriptions out.

use std::sync::Arc;

use crate::Error;
use crate::bits::BitReader;
use crate::frame::{FrameHeader, FrameType, NUM_REF_FRAMES, REFS_PER_FRAME};
use crate::metadata::Metadata;
use crate::obu::{Obu, ObuIter, ObuType, annexb_temporal_unit};
use crate::refs::{PictureId, RefState};
use crate::sequence::SequenceHeader;
use crate::video;

/// What a temporal unit asks of the decoder, in bitstream order.
#[derive(Clone, Debug)]
pub enum Output {
    /// Decode a picture (shown now if `show_frame`).
    Decode(Box<DecodePicture>),
    /// Show a picture decoded earlier, without decoding anything.
    ShowExisting(ShowExisting),
    /// A metadata OBU.
    Metadata(Metadata),
}

/// One coded tile inside the submitted buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tile {
    /// Byte offset of the tile data (after its tile_size_minus_1 field).
    pub offset: u32,
    pub size: u32,
}

/// Everything needed to decode one picture.
#[derive(Clone, Debug)]
pub struct DecodePicture {
    pub id: PictureId,
    /// Bumped whenever a sequence header different from the previous one
    /// arrives; recreate session parameters when it changes.
    pub sequence_generation: u64,
    pub sequence: Arc<SequenceHeader>,
    pub std_sequence: Arc<video::SequenceHeader>,
    pub header: Arc<FrameHeader>,
    pub std_picture: video::PictureInfo,
    /// FrameWidth: the coded width (before the superres upscale).
    pub frame_width: u32,
    /// FrameHeight.
    pub frame_height: u32,
    /// UpscaledWidth: the width of the reconstructed picture. Equal to
    /// `frame_width` unless `use_superres`.
    pub upscaled_width: u32,
    pub use_superres: bool,
    pub render_width: u32,
    pub render_height: u32,
    /// The reference slot (0..8) each of LAST..ALTREF reads; -1 on intra
    /// frames (Vulkan's referenceNameSlotIndices, in AV1 slot numbers).
    pub reference_name_slot_indices: [i32; REFS_PER_FRAME],
    /// The picture each of LAST..ALTREF reads.
    pub reference_pictures: [Option<PictureId>; REFS_PER_FRAME],
    /// The reference information of every valid slot before this picture.
    pub reference_infos: [Option<video::ReferenceInfo>; NUM_REF_FRAMES],
    /// The pictures the slots hold before this picture.
    pub slot_pictures: [Option<PictureId>; NUM_REF_FRAMES],
    /// The reference information of this picture once decoded.
    pub setup_reference_info: video::ReferenceInfo,
    /// Slots that hold this picture afterwards.
    pub refresh_frame_flags: u8,
    pub show_frame: bool,
    pub showable_frame: bool,
    /// Byte offset of the OBU holding the frame header.
    pub frame_header_offset: u32,
    /// The tiles in tile order (row by row).
    pub tiles: Vec<Tile>,
    /// OBU extension ids of the frame header.
    pub temporal_id: u8,
    pub spatial_id: u8,
}

impl DecodePicture {
    /// pTileOffsets.
    pub fn tile_offsets(&self) -> Vec<u32> {
        self.tiles.iter().map(|t| t.offset).collect()
    }

    /// pTileSizes.
    pub fn tile_sizes(&self) -> Vec<u32> {
        self.tiles.iter().map(|t| t.size).collect()
    }
}

/// show_existing_frame.
#[derive(Clone, Debug)]
pub struct ShowExisting {
    pub id: PictureId,
    /// frame_to_show_map_idx.
    pub slot: u8,
    pub frame_type: FrameType,
    /// 0, or 0xff when a key frame is shown again: every slot then holds
    /// `id`.
    pub refresh_frame_flags: u8,
    /// The grain to apply to this showing, if any.
    pub film_grain: Option<video::FilmGrain>,
    pub header: Arc<FrameHeader>,
}

/// A frame whose header has been seen but not all of its tiles.
#[derive(Debug)]
struct PendingFrame {
    header: Arc<FrameHeader>,
    /// The reference state as the header left it, committed when the
    /// frame completes.
    state: RefState,
    /// The reference state before the header (for reference_infos).
    before: RefState,
    tiles: Vec<Tile>,
    frame_header_offset: u32,
    temporal_id: u8,
    spatial_id: u8,
}

/// The AV1 header parser and reference tracker.
#[derive(Debug)]
pub struct Parser {
    operating_point: usize,
    operating_point_idc: u16,
    sequence: Option<Arc<SequenceHeader>>,
    std_sequence: Option<Arc<video::SequenceHeader>>,
    sequence_generation: u64,
    refs: RefState,
    pending: Option<PendingFrame>,
    next_picture: u64,
}

impl Default for Parser {
    fn default() -> Self {
        Self::new()
    }
}

impl Parser {
    /// A parser for operating point 0 (all layers of the best quality).
    pub fn new() -> Self {
        Self::with_operating_point(0)
    }

    /// A parser for the given operating point (choose_operating_point()).
    pub fn with_operating_point(operating_point: usize) -> Self {
        Self {
            operating_point,
            operating_point_idc: 0,
            sequence: None,
            std_sequence: None,
            sequence_generation: 0,
            refs: RefState::default(),
            pending: None,
            next_picture: 0,
        }
    }

    /// The current sequence header.
    pub fn sequence(&self) -> Option<&Arc<SequenceHeader>> {
        self.sequence.as_ref()
    }

    /// The reference slots.
    pub fn refs(&self) -> &RefState {
        &self.refs
    }

    /// The pictures the eight slots hold. A picture no slot holds (and that
    /// is not waiting to be shown) can be released.
    pub fn slot_pictures(&self) -> [Option<PictureId>; NUM_REF_FRAMES] {
        self.refs.pictures()
    }

    /// Forgets every reference, as after a seek; the next frame must be a
    /// key frame. The sequence header is kept.
    pub fn reset(&mut self) {
        self.refs = RefState::default();
        self.pending = None;
    }

    /// Parses one temporal unit (or any run of whole OBUs) in the
    /// low-overhead format, where every OBU has obu_size. Offsets in the
    /// output are into `data`.
    pub fn parse_temporal_unit(&mut self, data: &[u8]) -> Result<Vec<Output>, Error> {
        let mut out = Vec::new();
        for obu in ObuIter::new(data) {
            self.parse_obu(&obu?, &mut out)?;
        }
        Ok(out)
    }

    /// Parses one Annex B temporal_unit() (without its temporal_unit_size).
    pub fn parse_annexb_temporal_unit(&mut self, data: &[u8]) -> Result<Vec<Output>, Error> {
        let mut out = Vec::new();
        for obu in annexb_temporal_unit(data)? {
            self.parse_obu(&obu, &mut out)?;
        }
        Ok(out)
    }

    /// Parses one OBU, appending to `out` what it completes. On an error the
    /// partly parsed frame is dropped and the reference state is unchanged.
    pub fn parse_obu(&mut self, obu: &Obu<'_>, out: &mut Vec<Output>) -> Result<(), Error> {
        let result = self.parse_obu_inner(obu, out);
        if result.is_err() {
            self.pending = None;
        }
        result
    }

    fn parse_obu_inner(&mut self, obu: &Obu<'_>, out: &mut Vec<Output>) -> Result<(), Error> {
        if !obu.header.in_operating_point(self.operating_point_idc) {
            return Ok(());
        }
        match obu.header.obu_type {
            ObuType::TemporalDelimiter => {
                if self.pending.is_some() {
                    self.pending = None;
                    return Err(Error::Invalid("temporal unit ended inside a frame"));
                }
            }
            ObuType::SequenceHeader => self.sequence_header(obu.payload)?,
            ObuType::FrameHeader | ObuType::RedundantFrameHeader => {
                // A copy of the header of the frame in progress is skipped.
                if self.pending.is_none() {
                    if obu.header.obu_type == ObuType::RedundantFrameHeader {
                        return Ok(());
                    }
                    self.frame_header(obu, false, out)?;
                }
            }
            ObuType::Frame => {
                if self.pending.is_some() {
                    return Err(Error::Invalid("frame OBU inside another frame"));
                }
                self.frame_header(obu, true, out)?;
            }
            ObuType::TileGroup => {
                let Some(pending) = self.pending.as_mut() else {
                    return Err(Error::Invalid("tile group without a frame header"));
                };
                if tile_group(obu.payload, obu.payload_offset, pending)? {
                    self.finish_frame(out);
                }
            }
            ObuType::Metadata => out.push(Output::Metadata(Metadata::parse(obu.payload)?)),
            ObuType::TileList => {
                return Err(Error::Unsupported("large scale tile (tile list OBU)"));
            }
            ObuType::Padding | ObuType::Reserved(_) => {}
        }
        Ok(())
    }

    fn sequence_header(&mut self, payload: &[u8]) -> Result<(), Error> {
        let seq = SequenceHeader::parse(payload)?;
        if self.sequence.as_deref() != Some(&seq) {
            let op = self.operating_point.min(seq.operating_points.len() - 1);
            self.operating_point_idc = seq.operating_points[op].idc;
            self.std_sequence = Some(Arc::new(video::SequenceHeader::new(&seq)));
            self.sequence = Some(Arc::new(seq));
            self.sequence_generation += 1;
        }
        Ok(())
    }

    fn frame_header(
        &mut self,
        obu: &Obu<'_>,
        is_frame: bool,
        out: &mut Vec<Output>,
    ) -> Result<(), Error> {
        let seq = self.sequence.clone().ok_or(Error::MissingSequenceHeader)?;
        let mut state = self.refs.clone();
        let mut r = BitReader::new(obu.payload);
        let (temporal_id, spatial_id) = (obu.header.temporal_id, obu.header.spatial_id);
        let header = FrameHeader::parse(&mut r, &seq, &mut state, temporal_id, spatial_id)?;
        if header.show_existing_frame {
            if is_frame {
                return Err(Error::Invalid("show_existing_frame in a frame OBU"));
            }
            let slot = usize::from(header.frame_to_show_map_idx);
            let id = self.refs.slots[slot]
                .picture
                .ok_or(Error::MissingReference(slot as u8))?;
            if header.frame_type == FrameType::Key {
                self.refs.show_existing_key_frame(slot);
            }
            out.push(Output::ShowExisting(ShowExisting {
                id,
                slot: slot as u8,
                frame_type: header.frame_type,
                refresh_frame_flags: header.refresh_frame_flags,
                film_grain: video::FilmGrain::new(&header.film_grain),
                header: Arc::new(header),
            }));
            return Ok(());
        }
        let mut pending = PendingFrame {
            header: Arc::new(header),
            state,
            before: self.refs.clone(),
            tiles: Vec::new(),
            frame_header_offset: obu.offset as u32,
            temporal_id,
            spatial_id,
        };
        if is_frame {
            r.byte_alignment();
            let header_bytes = r.position() / 8;
            let payload = obu.payload.get(header_bytes..).ok_or(Error::EndOfData)?;
            if !tile_group(payload, obu.payload_offset + header_bytes, &mut pending)? {
                return Err(Error::Invalid("frame OBU without its last tile"));
            }
            self.pending = Some(pending);
            self.finish_frame(out);
        } else {
            self.pending = Some(pending);
        }
        Ok(())
    }

    /// decode_frame_wrapup() as far as headers go: the reference update.
    fn finish_frame(&mut self, out: &mut Vec<Output>) {
        let Some(p) = self.pending.take() else { return };
        let seq = self
            .sequence
            .clone()
            .expect("a frame needs a sequence header");
        let std_sequence = self
            .std_sequence
            .clone()
            .expect("set with the sequence header");
        let id = PictureId(self.next_picture);
        self.next_picture += 1;
        let h = &p.header;
        let intra = h.frame_is_intra();
        let reference_name_slot_indices = std::array::from_fn(|i| {
            if intra {
                -1
            } else {
                i32::from(h.ref_frame_idx[i])
            }
        });
        let reference_pictures = std::array::from_fn(|i| {
            if intra {
                None
            } else {
                p.state.slots[usize::from(h.ref_frame_idx[i])].picture
            }
        });
        let reference_infos = std::array::from_fn(|i| {
            let slot = &p.state.slots[i];
            slot.valid.then(|| video::ReferenceInfo::from_slot(slot))
        });
        let slot_pictures = p.before.pictures();
        let c = &seq.color_config;
        self.refs = p.state;
        self.refs
            .update(h, (c.subsampling_x, c.subsampling_y, c.bit_depth), id);
        out.push(Output::Decode(Box::new(DecodePicture {
            id,
            sequence_generation: self.sequence_generation,
            sequence: seq.clone(),
            std_sequence,
            std_picture: video::PictureInfo::new(h),
            frame_width: h.size.frame_width,
            frame_height: h.size.frame_height,
            upscaled_width: h.size.upscaled_width,
            use_superres: h.size.use_superres,
            render_width: h.size.render_width,
            render_height: h.size.render_height,
            reference_name_slot_indices,
            reference_pictures,
            reference_infos,
            slot_pictures,
            setup_reference_info: video::ReferenceInfo::from_header(h),
            refresh_frame_flags: h.refresh_frame_flags,
            show_frame: h.show_frame,
            showable_frame: h.showable_frame,
            frame_header_offset: p.frame_header_offset,
            tiles: p.tiles,
            temporal_id: p.temporal_id,
            spatial_id: p.spatial_id,
            header: p.header,
        })));
    }
}

/// tile_group_obu(sz) of section 5.11.1: records the tiles and returns
/// whether the frame's last tile was in this group. `base` is the offset of
/// `payload` in the submitted buffer.
fn tile_group(payload: &[u8], base: usize, p: &mut PendingFrame) -> Result<bool, Error> {
    let t = &p.header.tile_info;
    let num_tiles = t.num_tiles();
    let mut r = BitReader::new(payload);
    let mut tg_start = 0;
    let mut tg_end = num_tiles - 1;
    if num_tiles > 1 && r.flag()? {
        let tile_bits = t.tile_cols_log2 + t.tile_rows_log2;
        tg_start = r.f(tile_bits)?;
        tg_end = r.f(tile_bits)?;
    }
    r.byte_alignment();
    if tg_start != p.tiles.len() as u32 || tg_end < tg_start || tg_end >= num_tiles {
        return Err(Error::Invalid("tile group out of order"));
    }
    let tile_size_bytes = usize::from(t.tile_size_bytes_minus_1) + 1;
    let mut pos = r.position() / 8;
    for tile_num in tg_start..=tg_end {
        let size = if tile_num == tg_end {
            payload.len().checked_sub(pos).ok_or(Error::EndOfData)?
        } else {
            let bytes = payload
                .get(pos..pos + tile_size_bytes)
                .ok_or(Error::EndOfData)?;
            pos += tile_size_bytes;
            bytes
                .iter()
                .rev()
                .fold(0usize, |v, &b| (v << 8) | usize::from(b))
                + 1
        };
        if pos + size > payload.len() || size == 0 {
            return Err(Error::EndOfData);
        }
        p.tiles.push(Tile {
            offset: (base + pos) as u32,
            size: size as u32,
        });
        pos += size;
    }
    Ok(tg_end == num_tiles - 1)
}
