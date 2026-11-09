//! metadata_obu() of AV1 spec section 5.8. HDR metadata is parsed; other
//! types keep their payload.

use crate::Error;
use crate::bits::BitReader;

/// A metadata OBU.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Metadata {
    /// METADATA_TYPE_HDR_CLL.
    ContentLightLevel { max_cll: u16, max_fall: u16 },
    /// METADATA_TYPE_HDR_MDCV: chromaticities in 0.16 fixed point,
    /// luminance in 24.8 (max) and 18.14 (min).
    MasteringDisplay {
        primary_chromaticity_x: [u16; 3],
        primary_chromaticity_y: [u16; 3],
        white_point_chromaticity_x: u16,
        white_point_chromaticity_y: u16,
        luminance_max: u32,
        luminance_min: u32,
    },
    /// Any other type (scalability, ITU-T T.35, timecode, unregistered),
    /// with the payload after metadata_type.
    Other {
        metadata_type: u64,
        payload: Vec<u8>,
    },
}

impl Metadata {
    pub fn parse(payload: &[u8]) -> Result<Self, Error> {
        let mut r = BitReader::new(payload);
        let metadata_type = r.leb128()?;
        Ok(match metadata_type {
            1 => Self::ContentLightLevel {
                max_cll: r.f(16)? as u16,
                max_fall: r.f(16)? as u16,
            },
            2 => {
                let mut x = [0; 3];
                let mut y = [0; 3];
                for i in 0..3 {
                    x[i] = r.f(16)? as u16;
                    y[i] = r.f(16)? as u16;
                }
                Self::MasteringDisplay {
                    primary_chromaticity_x: x,
                    primary_chromaticity_y: y,
                    white_point_chromaticity_x: r.f(16)? as u16,
                    white_point_chromaticity_y: r.f(16)? as u16,
                    luminance_max: r.f(32)?,
                    luminance_min: r.f(32)?,
                }
            }
            _ => Self::Other {
                metadata_type,
                payload: payload[r.position() / 8..].to_vec(),
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_light_level() {
        let m = Metadata::parse(&[1, 0x03, 0xe8, 0x01, 0x90]).unwrap();
        assert_eq!(
            m,
            Metadata::ContentLightLevel {
                max_cll: 1000,
                max_fall: 400
            }
        );
    }

    #[test]
    fn mastering_display() {
        let mut p = vec![2];
        for v in [1u16, 2, 3, 4, 5, 6, 7, 8] {
            p.extend(v.to_be_bytes());
        }
        p.extend(1000u32.to_be_bytes());
        p.extend(5u32.to_be_bytes());
        let Metadata::MasteringDisplay {
            primary_chromaticity_x,
            primary_chromaticity_y,
            white_point_chromaticity_y,
            luminance_max,
            luminance_min,
            ..
        } = Metadata::parse(&p).unwrap()
        else {
            panic!("not MDCV");
        };
        assert_eq!(primary_chromaticity_x, [1, 3, 5]);
        assert_eq!(primary_chromaticity_y, [2, 4, 6]);
        assert_eq!(white_point_chromaticity_y, 8);
        assert_eq!((luminance_max, luminance_min), (1000, 5));
    }

    #[test]
    fn other_types_keep_their_payload() {
        let m = Metadata::parse(&[4, 0xb5, 0x00, 0x31]).unwrap();
        assert_eq!(
            m,
            Metadata::Other {
                metadata_type: 4,
                payload: vec![0xb5, 0x00, 0x31]
            }
        );
    }
}
