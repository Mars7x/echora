//! The bit reader of AV1 spec section 4.10 and 8.1: most significant bit
//! first, with the descriptors f(n), su(n), ns(n), le(n), leb128() and
//! uvlc().

use crate::Error;

/// Reads bits from a byte slice, most significant bit first.
#[derive(Clone, Debug)]
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Position in bits from the start of `data`.
    pos: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// get_position(): bits read so far.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Bits left in the slice.
    pub fn remaining(&self) -> usize {
        self.data.len() * 8 - self.pos
    }

    /// Bytes consumed, counting a partly read byte as consumed.
    pub fn byte_position(&self) -> usize {
        self.pos.div_ceil(8)
    }

    pub fn bit(&mut self) -> Result<bool, Error> {
        let byte = *self.data.get(self.pos / 8).ok_or(Error::EndOfData)?;
        let bit = (byte >> (7 - self.pos % 8)) & 1;
        self.pos += 1;
        Ok(bit != 0)
    }

    /// f(n), n up to 32.
    pub fn f(&mut self, n: u32) -> Result<u32, Error> {
        debug_assert!(n <= 32);
        if n as usize > self.remaining() {
            return Err(Error::EndOfData);
        }
        let mut x = 0u64;
        for _ in 0..n {
            x = (x << 1) | u64::from(self.bit()?);
        }
        Ok(x as u32)
    }

    /// f(1) as a flag.
    pub fn flag(&mut self) -> Result<bool, Error> {
        self.bit()
    }

    /// su(n): an n-bit two's complement value.
    pub fn su(&mut self, n: u32) -> Result<i32, Error> {
        let value = self.f(n)? as i64;
        let sign_mask = 1i64 << (n - 1);
        Ok((if value & sign_mask != 0 {
            value - 2 * sign_mask
        } else {
            value
        }) as i32)
    }

    /// ns(n): a value in 0..n with a non-symmetric unsigned encoding.
    pub fn ns(&mut self, n: u32) -> Result<u32, Error> {
        if n <= 1 {
            return Ok(0);
        }
        let w = 32 - n.leading_zeros(); // FloorLog2(n) + 1
        let m = (1u32 << w) - n;
        let v = self.f(w - 1)?;
        if v < m {
            return Ok(v);
        }
        let extra_bit = self.f(1)?;
        Ok((v << 1) - m + extra_bit)
    }

    /// le(n): n little-endian bytes. Byte aligned by definition.
    pub fn le(&mut self, n: u32) -> Result<u32, Error> {
        let mut t = 0u32;
        for i in 0..n {
            t |= self.f(8)? << (i * 8);
        }
        Ok(t)
    }

    /// leb128(): up to eight bytes, each carrying seven bits.
    pub fn leb128(&mut self) -> Result<u64, Error> {
        let mut value = 0u64;
        for i in 0..8 {
            let byte = self.f(8)?;
            value |= u64::from(byte & 0x7f) << (i * 7);
            if byte & 0x80 == 0 {
                return if value > u64::from(u32::MAX) {
                    Err(Error::Invalid("leb128 value over 2^32 - 1"))
                } else {
                    Ok(value)
                };
            }
        }
        Err(Error::Invalid("leb128 longer than 8 bytes"))
    }

    /// uvlc(): an Exp-Golomb style value.
    pub fn uvlc(&mut self) -> Result<u32, Error> {
        let mut leading_zeros = 0u32;
        while !self.bit()? {
            leading_zeros += 1;
        }
        if leading_zeros >= 32 {
            return Ok(u32::MAX);
        }
        let value = self.f(leading_zeros)?;
        Ok(value + ((1u32 << leading_zeros) - 1))
    }

    /// delta_q via read_delta_q(): a flag and then su(1 + 6).
    pub fn delta_q(&mut self) -> Result<i8, Error> {
        if self.flag()? {
            Ok(self.su(7)? as i8)
        } else {
            Ok(0)
        }
    }

    /// byte_alignment(): skip to the next byte boundary. The spec requires
    /// the skipped bits to be zero; they are not checked, as decoders do.
    pub fn byte_alignment(&mut self) {
        self.pos = self.pos.div_ceil(8) * 8;
    }

    /// decode_signed_subexp_with_ref(low, high, r) of section 5.9.26.
    pub fn signed_subexp_with_ref(&mut self, low: i32, high: i32, r: i32) -> Result<i32, Error> {
        let x = self.unsigned_subexp_with_ref((high - low) as u32, (r - low) as u32)?;
        Ok(x as i32 + low)
    }

    fn unsigned_subexp_with_ref(&mut self, mx: u32, r: u32) -> Result<u32, Error> {
        let v = self.subexp(mx)?;
        Ok(if (r << 1) <= mx {
            inverse_recenter(r, v)
        } else {
            mx - 1 - inverse_recenter(mx - 1 - r, v)
        })
    }

    fn subexp(&mut self, num_syms: u32) -> Result<u32, Error> {
        let mut i = 0u32;
        let mut mk = 0u32;
        let k = 3u32;
        loop {
            let b2 = if i != 0 { k + i - 1 } else { k };
            let a = 1u32 << b2;
            if num_syms <= mk + 3 * a {
                return Ok(self.ns(num_syms - mk)? + mk);
            }
            if self.flag()? {
                i += 1;
                mk += a;
            } else {
                return Ok(self.f(b2)? + mk);
            }
        }
    }
}

fn inverse_recenter(r: u32, v: u32) -> u32 {
    if v > 2 * r {
        v
    } else if v & 1 != 0 {
        r - ((v + 1) >> 1)
    } else {
        r + (v >> 1)
    }
}

/// Writes bits most significant first; the tests build headers with it.
#[cfg(test)]
#[derive(Debug, Default)]
pub(crate) struct BitWriter {
    pub bytes: Vec<u8>,
    bits: usize,
}

#[cfg(test)]
impl BitWriter {
    pub fn put(&mut self, n: u32, value: u32) -> &mut Self {
        for i in (0..n).rev() {
            if self.bits.is_multiple_of(8) {
                self.bytes.push(0);
            }
            if (value >> i) & 1 != 0 {
                let last = self.bytes.len() - 1;
                self.bytes[last] |= 0x80 >> (self.bits % 8);
            }
            self.bits += 1;
        }
        self
    }

    pub fn flag(&mut self, b: bool) -> &mut Self {
        self.put(1, u32::from(b))
    }

    pub fn su(&mut self, n: u32, value: i32) -> &mut Self {
        self.put(n, (value as u32) & ((1u64 << n) - 1) as u32)
    }

    pub fn align(&mut self) -> &mut Self {
        while !self.bits.is_multiple_of(8) {
            self.put(1, 0);
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_width_reads() {
        let mut r = BitReader::new(&[0b1010_1100, 0xff, 0x01]);
        assert_eq!(r.f(1).unwrap(), 1);
        assert_eq!(r.f(3).unwrap(), 0b010);
        assert_eq!(r.f(8).unwrap(), 0b1100_1111);
        assert_eq!(r.position(), 12);
        assert_eq!(r.f(12).unwrap(), 0b1111_0000_0001);
        assert!(matches!(r.f(1), Err(Error::EndOfData)));
    }

    #[test]
    fn signed_reads() {
        let mut w = BitWriter::default();
        w.su(7, -64).su(7, 63).su(7, -1).su(4, 0);
        let mut r = BitReader::new(&w.bytes);
        assert_eq!(r.su(7).unwrap(), -64);
        assert_eq!(r.su(7).unwrap(), 63);
        assert_eq!(r.su(7).unwrap(), -1);
        assert_eq!(r.su(4).unwrap(), 0);
    }

    #[test]
    fn leb128_values() {
        assert_eq!(BitReader::new(&[0x05]).leb128().unwrap(), 5);
        assert_eq!(
            BitReader::new(&[0xe5, 0x8e, 0x26]).leb128().unwrap(),
            624485
        );
        // A padded encoding is allowed.
        assert_eq!(BitReader::new(&[0x81, 0x80, 0x00]).leb128().unwrap(), 1);
        assert!(BitReader::new(&[0x80; 9]).leb128().is_err());
        assert!(
            BitReader::new(&[0xff, 0xff, 0xff, 0xff, 0x7f])
                .leb128()
                .is_err()
        );
    }

    #[test]
    fn uvlc_values() {
        // 1 -> 0, 010 -> 1, 011 -> 2, 00100 -> 3
        let mut w = BitWriter::default();
        w.put(1, 1).put(3, 0b010).put(3, 0b011).put(5, 0b00100);
        let mut r = BitReader::new(&w.bytes);
        assert_eq!(r.uvlc().unwrap(), 0);
        assert_eq!(r.uvlc().unwrap(), 1);
        assert_eq!(r.uvlc().unwrap(), 2);
        assert_eq!(r.uvlc().unwrap(), 3);
    }

    #[test]
    fn ns_values() {
        // n = 5: w = 3, m = 3. Values 0..3 take two bits, 3 and 4 take three.
        let mut w = BitWriter::default();
        w.put(2, 0).put(2, 2).put(3, 0b110).put(3, 0b111);
        let mut r = BitReader::new(&w.bytes);
        assert_eq!(r.ns(5).unwrap(), 0);
        assert_eq!(r.ns(5).unwrap(), 2);
        assert_eq!(r.ns(5).unwrap(), 3);
        assert_eq!(r.ns(5).unwrap(), 4);
        assert_eq!(r.ns(1).unwrap(), 0);
        assert_eq!(r.position(), 10);
    }

    #[test]
    fn le_and_alignment() {
        let mut r = BitReader::new(&[0x80, 0x34, 0x12]);
        r.f(1).unwrap();
        r.byte_alignment();
        assert_eq!(r.position(), 8);
        assert_eq!(r.le(2).unwrap(), 0x1234);
        r.byte_alignment();
        assert_eq!(r.position(), 24);
    }

    #[test]
    fn subexp_round_trip_small() {
        // decode_subexp(numSyms = 10): numSyms <= 0 + 3 * 8, so ns(10).
        // ns(10): w = 4, m = 6; v = 5 in three bits is below m.
        let mut w = BitWriter::default();
        w.put(3, 5);
        let mut r = BitReader::new(&w.bytes);
        // low = 0, high = 10, r = 0: (0 << 1) <= 10, inverse_recenter(0, 5) = 5.
        assert_eq!(r.signed_subexp_with_ref(0, 10, 0).unwrap(), 5);
    }

    #[test]
    fn inverse_recenter_values() {
        assert_eq!(inverse_recenter(4, 9), 9);
        assert_eq!(inverse_recenter(4, 0), 4);
        assert_eq!(inverse_recenter(4, 1), 3);
        assert_eq!(inverse_recenter(4, 2), 5);
        assert_eq!(inverse_recenter(4, 3), 2);
    }
}
