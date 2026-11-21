//! A small, fast hasher for the renderer's maps looked up several times per
//! draw (keyed by Vulkan handles and target keys).

/// Multiply-rotate over 8-byte words, as rustc's FxHasher; not
/// collision-resistant, which these keys need not be.
#[derive(Clone, Copy, Default)]
pub struct FastHasher(u64);

impl std::hash::Hasher for FastHasher {
    fn write(&mut self, bytes: &[u8]) {
        let (words, tail) = bytes.as_chunks::<8>();
        for word in words {
            self.add(u64::from_le_bytes(*word));
        }
        if !tail.is_empty() {
            let mut last = [0u8; 8];
            last[..tail.len()].copy_from_slice(tail);
            self.add(u64::from_le_bytes(last));
        }
    }

    fn write_u64(&mut self, value: u64) {
        self.add(value);
    }

    // Narrower integers are one word each, not a padded byte slice.
    fn write_u8(&mut self, value: u8) {
        self.add(u64::from(value));
    }

    fn write_u16(&mut self, value: u16) {
        self.add(u64::from(value));
    }

    fn write_u32(&mut self, value: u32) {
        self.add(u64::from(value));
    }

    fn write_usize(&mut self, value: usize) {
        self.add(value as u64);
    }

    fn write_i32(&mut self, value: i32) {
        self.add(u64::from(value as u32));
    }

    fn finish(&self) -> u64 {
        // The well-mixed high bits rotated down: the maps index buckets by
        // the low bits, which a multiply leaves zero for aligned keys (64-
        // byte aligned addresses used 64 of 4096 buckets).
        self.0.rotate_left(26)
    }
}

impl FastHasher {
    fn add(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
}

/// A HashMap with [`FastHasher`].
pub type FastMap<K, V> = std::collections::HashMap<K, V, std::hash::BuildHasherDefault<FastHasher>>;
/// A HashSet with [`FastHasher`].
pub type FastSet<K> = std::collections::HashSet<K, std::hash::BuildHasherDefault<FastHasher>>;
