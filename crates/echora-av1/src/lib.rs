//! AV1 bitstream parsing for hardware (Vulkan Video) decoding.
//!
//! Vulkan Video's AV1 decode leaves header parsing and reference
//! bookkeeping to the application. This crate does that half, in plain
//! Rust: it splits OBUs, parses the sequence header and the uncompressed
//! frame header (AV1 spec sections 5.5 and 5.9), tracks the eight
//! reference slots, finds each tile's bytes, and describes every picture
//! with plain-data mirrors of the `StdVideoAV1*` structures ([`video`]).
//! Tile data itself is not decoded.
//!
//! ```no_run
//! # fn submit(_: &echora_av1::DecodePicture) {}
//! let mut parser = echora_av1::Parser::new();
//! # let temporal_unit: &[u8] = &[];
//! for output in parser.parse_temporal_unit(temporal_unit)? {
//!     match output {
//!         echora_av1::Output::Decode(picture) => submit(&picture),
//!         echora_av1::Output::ShowExisting(show) => println!("show {:?}", show.id),
//!         echora_av1::Output::Metadata(_) => {}
//!     }
//! }
//! # Ok::<(), echora_av1::Error>(())
//! ```

pub mod bits;
pub mod frame;
pub mod ivf;
pub mod metadata;
pub mod obu;
mod parser;
pub mod refs;
pub mod sequence;
pub mod video;

pub use frame::FrameHeader;
pub use parser::{DecodePicture, Output, Parser, ShowExisting, Tile};
pub use refs::PictureId;
pub use sequence::SequenceHeader;

/// A parse failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// The data ended inside a syntax element or an OBU.
    EndOfData,
    /// The bitstream breaks a rule of the specification.
    Invalid(&'static str),
    /// Valid AV1 this crate does not handle.
    Unsupported(&'static str),
    /// A frame header arrived before any sequence header.
    MissingSequenceHeader,
    /// A frame uses a reference slot that holds no valid frame.
    MissingReference(u8),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EndOfData => write!(f, "AV1 data ended early"),
            Self::Invalid(what) => write!(f, "invalid AV1 bitstream: {what}"),
            Self::Unsupported(what) => write!(f, "unsupported AV1 feature: {what}"),
            Self::MissingSequenceHeader => write!(f, "AV1 frame before a sequence header"),
            Self::MissingReference(slot) => write!(f, "AV1 reference slot {slot} is empty"),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests;
