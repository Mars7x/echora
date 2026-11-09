//! OBU framing (AV1 spec sections 5.3 and 5.4, and Annex B).

use crate::Error;
use crate::bits::BitReader;

/// obu_type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ObuType {
    SequenceHeader,
    TemporalDelimiter,
    FrameHeader,
    TileGroup,
    Metadata,
    Frame,
    RedundantFrameHeader,
    TileList,
    Padding,
    /// 0 and 9..=14.
    Reserved(u8),
}

impl ObuType {
    pub fn from_raw(value: u8) -> Self {
        match value {
            1 => Self::SequenceHeader,
            2 => Self::TemporalDelimiter,
            3 => Self::FrameHeader,
            4 => Self::TileGroup,
            5 => Self::Metadata,
            6 => Self::Frame,
            7 => Self::RedundantFrameHeader,
            8 => Self::TileList,
            15 => Self::Padding,
            other => Self::Reserved(other),
        }
    }

    pub fn raw(self) -> u8 {
        match self {
            Self::SequenceHeader => 1,
            Self::TemporalDelimiter => 2,
            Self::FrameHeader => 3,
            Self::TileGroup => 4,
            Self::Metadata => 5,
            Self::Frame => 6,
            Self::RedundantFrameHeader => 7,
            Self::TileList => 8,
            Self::Padding => 15,
            Self::Reserved(other) => other,
        }
    }
}

/// obu_header() with its optional obu_extension_header().
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObuHeader {
    pub obu_type: ObuType,
    pub extension_flag: bool,
    pub has_size_field: bool,
    /// temporal_id and spatial_id; zero without an extension header.
    pub temporal_id: u8,
    pub spatial_id: u8,
}

impl ObuHeader {
    /// Header bytes: one, or two with the extension.
    pub fn len(&self) -> usize {
        1 + usize::from(self.extension_flag)
    }

    /// Never empty; present for clippy's len_without_is_empty.
    pub fn is_empty(&self) -> bool {
        false
    }

    pub fn parse(r: &mut BitReader<'_>) -> Result<Self, Error> {
        if r.flag()? {
            return Err(Error::Invalid("obu_forbidden_bit set"));
        }
        let obu_type = ObuType::from_raw(r.f(4)? as u8);
        let extension_flag = r.flag()?;
        let has_size_field = r.flag()?;
        let _reserved = r.flag()?;
        let (mut temporal_id, mut spatial_id) = (0, 0);
        if extension_flag {
            temporal_id = r.f(3)? as u8;
            spatial_id = r.f(2)? as u8;
            let _reserved = r.f(3)?;
        }
        Ok(Self {
            obu_type,
            extension_flag,
            has_size_field,
            temporal_id,
            spatial_id,
        })
    }

    /// Whether the operating point selected by `operating_point_idc` keeps
    /// this OBU (section 7.5's drop_obu rule).
    pub fn in_operating_point(&self, operating_point_idc: u16) -> bool {
        if matches!(
            self.obu_type,
            ObuType::SequenceHeader | ObuType::TemporalDelimiter
        ) || operating_point_idc == 0
            || !self.extension_flag
        {
            return true;
        }
        let in_temporal = (operating_point_idc >> self.temporal_id) & 1 != 0;
        let in_spatial = (operating_point_idc >> (self.spatial_id + 8)) & 1 != 0;
        in_temporal && in_spatial
    }
}

/// One OBU inside a buffer.
#[derive(Clone, Copy, Debug)]
pub struct Obu<'a> {
    pub header: ObuHeader,
    /// Byte offset of the OBU's first header byte in the buffer it came from.
    pub offset: usize,
    /// Byte offset of the payload in that buffer.
    pub payload_offset: usize,
    pub payload: &'a [u8],
}

impl Obu<'_> {
    /// Total size including header and size field.
    pub fn total_len(&self) -> usize {
        self.payload_offset + self.payload.len() - self.offset
    }
}

/// Parses one OBU at `offset` of `data`. `sz` is the OBU's length when the
/// container gives it (Annex B obu_length); otherwise the OBU must carry
/// obu_size.
pub fn parse_obu(data: &[u8], offset: usize, sz: Option<usize>) -> Result<Obu<'_>, Error> {
    let bytes = data.get(offset..).ok_or(Error::EndOfData)?;
    let mut r = BitReader::new(bytes);
    let header = ObuHeader::parse(&mut r)?;
    let obu_size = if header.has_size_field {
        r.leb128()? as usize
    } else {
        let sz = sz.ok_or(Error::Invalid("OBU without obu_size outside Annex B"))?;
        sz.checked_sub(header.len())
            .ok_or(Error::Invalid("obu_length shorter than its header"))?
    };
    let payload_offset = offset + r.position() / 8;
    let end = payload_offset
        .checked_add(obu_size)
        .ok_or(Error::EndOfData)?;
    if end > data.len() {
        return Err(Error::EndOfData);
    }
    if let Some(sz) = sz
        && end - offset > sz
    {
        return Err(Error::Invalid("obu_size larger than obu_length"));
    }
    Ok(Obu {
        header,
        offset,
        payload_offset,
        payload: &data[payload_offset..end],
    })
}

/// Iterates over the OBUs of a low-overhead bitstream (section 5.2), as in
/// IVF, MP4 and Matroska samples: every OBU carries obu_size.
#[derive(Clone, Debug)]
pub struct ObuIter<'a> {
    data: &'a [u8],
    offset: usize,
    failed: bool,
}

impl<'a> ObuIter<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            offset: 0,
            failed: false,
        }
    }
}

impl<'a> Iterator for ObuIter<'a> {
    type Item = Result<Obu<'a>, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed || self.offset >= self.data.len() {
            return None;
        }
        match parse_obu(self.data, self.offset, None) {
            Ok(obu) => {
                self.offset = obu.payload_offset + obu.payload.len();
                Some(Ok(obu))
            }
            Err(e) => {
                self.failed = true;
                Some(Err(e))
            }
        }
    }
}

/// Splits one Annex B temporal_unit() (section B.2: the temporal_unit_size
/// already removed, as when reading it from a container) into its OBUs.
pub fn annexb_temporal_unit(data: &[u8]) -> Result<Vec<Obu<'_>>, Error> {
    let mut obus = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        let (frame_unit_size, len) = read_leb128(&data[pos..])?;
        let frame_start = pos + len;
        let frame_end = frame_start
            .checked_add(frame_unit_size)
            .ok_or(Error::EndOfData)?;
        if frame_end > data.len() {
            return Err(Error::EndOfData);
        }
        let mut obu_pos = frame_start;
        while obu_pos < frame_end {
            let (obu_length, len) = read_leb128(&data[obu_pos..frame_end])?;
            let obu_start = obu_pos + len;
            let obu_end = obu_start.checked_add(obu_length).ok_or(Error::EndOfData)?;
            if obu_end > frame_end {
                return Err(Error::EndOfData);
            }
            obus.push(parse_obu(&data[..obu_end], obu_start, Some(obu_length))?);
            obu_pos = obu_end;
        }
        pos = frame_end;
    }
    Ok(obus)
}

/// Splits an Annex B stream (a sequence of temporal_unit(temporal_unit_size))
/// into the byte ranges of its temporal units, without their size fields.
pub fn annexb_temporal_units(data: &[u8]) -> Result<Vec<std::ops::Range<usize>>, Error> {
    let mut units = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        let (size, len) = read_leb128(&data[pos..])?;
        let start = pos + len;
        let end = start.checked_add(size).ok_or(Error::EndOfData)?;
        if end > data.len() {
            return Err(Error::EndOfData);
        }
        units.push(start..end);
        pos = end;
    }
    Ok(units)
}

/// A leb128() value and its length in bytes.
fn read_leb128(data: &[u8]) -> Result<(usize, usize), Error> {
    let mut r = BitReader::new(data);
    let value = r.leb128()? as usize;
    Ok((value, r.byte_position()))
}
