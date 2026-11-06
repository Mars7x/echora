//! Fixed-width little- and big-endian reads from byte slices.
//!
//! Callers pass slices whose length was already validated (by the request
//! shape, a record size or a bounds check); a short slice is a runtime bug,
//! not a guest error, and panics with that explanation.

macro_rules! reader {
    ($name:ident, $ty:ty, $from:ident) => {
        pub(crate) fn $name(bytes: &[u8]) -> $ty {
            <$ty>::$from(
                *bytes
                    .first_chunk()
                    .expect("byte length validated by the caller"),
            )
        }
    };
}

reader!(le_u16, u16, from_le_bytes);
reader!(le_u32, u32, from_le_bytes);
reader!(le_u64, u64, from_le_bytes);
reader!(le_i64, i64, from_le_bytes);
reader!(le_u128, u128, from_le_bytes);
reader!(le_f32, f32, from_le_bytes);
reader!(be_u32, u32, from_be_bytes);
