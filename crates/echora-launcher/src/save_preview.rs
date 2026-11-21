//! The Ocarina of Time Load screen's screenshot and health for one save.

use std::fs::{self, File};
use std::io::{Cursor, Read};
use std::path::Path;
use std::sync::Arc;

use gpui::RenderImage;
use image::{Frame, ImageFormat, ImageReader, Limits};

const TITLE_ID: &str = "0400486024DEC000";
const MAX_CAPTION_BYTES: u64 = 256 * 1024;
const TABLE_START: usize = 40;
const CAPTION_KEYS: [u32; 6] = [
    0x9811_a3f7,
    0xdfd8_40d3,
    0xbd46_f485,
    0x23f3_d75e,
    0x2785_3bf7,
    0x5ee5_0bed,
];
// Health is in quarter hearts. Both values are from caption.sav, as is the
// length-prefixed JPEG.
const CURRENT_LIFE: u32 = 0x1e55_ecc5;
const MAX_LIFE: u32 = 0x74bd_05d0;
const THUMBNAIL: u32 = 0x6369_6a32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Health {
    pub quarters: u8,
    pub containers: u8,
}

pub struct SavePreview {
    pub image: Arc<RenderImage>,
    pub health: Option<Health>,
}

pub fn load(title: &str) -> Option<SavePreview> {
    load_from(&echora_config::save_dir()?, title)
}

fn load_from(root: &Path, title: &str) -> Option<SavePreview> {
    if !title.eq_ignore_ascii_case(TITLE_ID) {
        return None;
    }
    let account = u128::from_le_bytes(echora_config::LOCAL_USER_ID);
    let profile = root
        .join(TITLE_ID.to_ascii_lowercase())
        .join(format!("1-{account:032x}-0"));
    let (_, _, _, latest) = (0..6)
        .filter_map(|slot| {
            let dir = profile.join(format!("slot_{slot:02}"));
            let progress = fs::metadata(dir.join("progress.sav")).ok()?;
            if !progress.is_file() || progress.len() == 0 {
                return None;
            }
            let file = File::open(dir.join("caption.sav")).ok()?;
            let mut bytes = Vec::new();
            file.take(MAX_CAPTION_BYTES + 1)
                .read_to_end(&mut bytes)
                .ok()?;
            let stamp = Caption::new(&bytes)?.saved_at();
            Some((stamp, progress.modified().ok(), slot, bytes))
        })
        .max_by_key(|(stamp, modified, slot, _)| (*stamp, *modified, *slot))?;
    let caption = Caption::new(&latest)?;
    Some(SavePreview {
        image: decode_image(caption.jpeg()?)?,
        health: caption.health(),
    })
}

struct Caption<'a> {
    bytes: &'a [u8],
    fields: &'a [u8],
}

impl<'a> Caption<'a> {
    fn new(bytes: &'a [u8]) -> Option<Self> {
        if bytes.len() as u64 > MAX_CAPTION_BYTES || word(bytes, 0)? != 0x0102_0304 {
            return None;
        }
        let end = word(bytes, 8)? as usize;
        let fields = bytes.get(TABLE_START..end)?;
        fields
            .len()
            .is_multiple_of(8)
            .then_some(Self { bytes, fields })
    }

    fn field(&self, key: u32) -> Option<u32> {
        self.fields.as_chunks::<8>().0.iter().find_map(|entry| {
            if word(entry, 0)? == key {
                word(entry, 4)
            } else {
                None
            }
        })
    }

    fn saved_at(&self) -> Option<[u32; 6]> {
        let [year, month, day, hour, minute, second] = CAPTION_KEYS.map(|key| self.field(key));
        let stamp = [year?, month?, day?, hour?, minute?, second.unwrap_or(0)];
        ((1980..=2199).contains(&stamp[0])
            && (1..=12).contains(&stamp[1])
            && (1..=31).contains(&stamp[2])
            && stamp[3] < 24
            && stamp[4] < 60
            && stamp[5] < 60)
            .then_some(stamp)
    }

    fn health(&self) -> Option<Health> {
        let current = self.field(CURRENT_LIFE)?;
        let maximum = self.field(MAX_LIFE)?;
        if !(12..=80).contains(&maximum) || !maximum.is_multiple_of(4) || current > maximum {
            return None;
        }
        Some(Health {
            quarters: current as u8,
            containers: (maximum / 4) as u8,
        })
    }

    fn jpeg(&self) -> Option<&'a [u8]> {
        let offset = self.field(THUMBNAIL)? as usize;
        if offset < TABLE_START + self.fields.len() {
            return None;
        }
        let record = self.bytes.get(offset..)?;
        let length = word(record, 0)? as usize;
        let jpeg = record.get(4..)?.get(..length)?;
        (jpeg.starts_with(b"\xff\xd8\xff") && jpeg.ends_with(b"\xff\xd9")).then_some(jpeg)
    }
}

fn word(bytes: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        bytes.get(offset..)?.get(..4)?.try_into().ok()?,
    ))
}

fn decode_image(jpeg: &[u8]) -> Option<Arc<RenderImage>> {
    let mut reader = ImageReader::with_format(Cursor::new(jpeg), ImageFormat::Jpeg);
    let mut limits = Limits::default();
    limits.max_image_width = Some(1024);
    limits.max_image_height = Some(1024);
    limits.max_alloc = Some(8 * 1024 * 1024);
    reader.limits(limits);
    let mut pixels = reader.decode().ok()?.into_rgba8();
    // GPUI expects BGRA pixels.
    for pixel in pixels.as_chunks_mut::<4>().0 {
        pixel.swap(0, 2);
    }
    Some(Arc::new(RenderImage::new([Frame::new(pixels)])))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ExtendedColorType, codecs::jpeg::JpegEncoder};

    fn caption(second: u32, health: [u32; 2], rgb: [u8; 3]) -> Vec<u8> {
        let mut jpeg = Vec::new();
        JpegEncoder::new(&mut jpeg)
            .encode(&rgb.repeat(4), 2, 2, ExtendedColorType::Rgb8)
            .unwrap();
        let end = TABLE_START + 9 * 8;
        let mut data = vec![0; TABLE_START];
        data[..4].copy_from_slice(&0x0102_0304_u32.to_le_bytes());
        data[8..12].copy_from_slice(&(end as u32).to_le_bytes());
        let time = CAPTION_KEYS.into_iter().zip([2026, 11, 19, 20, 40, second]);
        for (key, value) in time.chain([
            (CURRENT_LIFE, health[0]),
            (MAX_LIFE, health[1]),
            (THUMBNAIL, end as u32),
        ]) {
            data.extend(key.to_le_bytes());
            data.extend(value.to_le_bytes());
        }
        data.extend((jpeg.len() as u32).to_le_bytes());
        data.extend(jpeg);
        data
    }

    fn slot(root: &Path, index: u8, caption: &[u8]) -> std::path::PathBuf {
        let dir = root
            .join("0400486024dec000/1-0100656d69746e75722d61726f686365-0")
            .join(format!("slot_{index:02}"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("caption.sav"), caption).unwrap();
        let progress = dir.join("progress.sav");
        fs::write(&progress, b"complete").unwrap();
        progress
    }

    #[test]
    fn picture_and_health_both_come_from_the_latest_completed_save() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let older = slot(root, 1, &caption(20, [12, 12], [250, 0, 0]));
        let latest = slot(root, 0, &caption(35, [7, 16], [0, 250, 0]));
        let later = slot(root, 2, &caption(45, [16, 16], [0, 0, 250]));
        // A restored backup can give the older progress file a newer mtime.
        File::options()
            .write(true)
            .open(older)
            .unwrap()
            .set_modified(
                std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(2_000_000_000),
            )
            .unwrap();
        File::options()
            .write(true)
            .open(latest)
            .unwrap()
            .set_modified(
                std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_900_000_000),
            )
            .unwrap();
        fs::remove_file(later).unwrap();
        let preview = load_from(root, TITLE_ID).unwrap();
        assert_eq!(
            preview.health,
            Some(Health {
                quarters: 7,
                containers: 4
            })
        );
        let pixel = &preview.image.as_bytes(0).unwrap()[..4];
        assert!(pixel[1] > 240 && pixel[0] < 10 && pixel[2] < 10);
        assert!(load_from(root, "0100000000000000").is_none());
    }

    #[test]
    fn invalid_health_is_omitted_and_an_invalid_thumbnail_uses_cover_art() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let mut data = caption(40, [18, 12], [0, 0, 250]);
        slot(root, 0, &data);
        assert_eq!(load_from(root, TITLE_ID).unwrap().health, None);
        data.pop();
        slot(root, 0, &data);
        assert!(load_from(root, TITLE_ID).is_none());
    }
}
