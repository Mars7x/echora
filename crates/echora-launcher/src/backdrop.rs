//! The launcher's backdrop: a frame from one of the game's own title-screen
//! videos, taken from the player's copy and kept in Echora's data folder.
//! Nothing from the game ships with Echora.

use std::path::{Path, PathBuf};
use std::process::Command;

use echora_content::ContentSource;

use crate::install;

/// The volcano over the misty field, halfway through this title video.
const MOVIE: &str = "Event/Movie/Dm_Title_0045.webm";
const AT_SECONDS: &str = "15";
const MOVIE_LIMIT: u64 = 256 << 20;

/// Where the extracted frame is kept.
pub fn path() -> Option<PathBuf> {
    install::data_dir().map(|dir| dir.join("backdrop.png"))
}

/// Takes the frame from `game` (a NUP or dump) and saves it at [`path`].
/// Decoding uses `ffmpeg`, which must be installed.
pub fn extract(game: &Path) -> Result<PathBuf, String> {
    let target = path().ok_or("there's no data folder")?;
    let dir = target.parent().ok_or("there's no data folder")?;
    let source = ContentSource::open(game).map_err(|error| error.to_string())?;
    let package = source
        .inventory()
        .map_err(|error| error.to_string())?
        .program_package
        .ok_or("the game has no program")?;
    let movie = source
        .read_romfs_file(package, MOVIE, MOVIE_LIMIT)
        .map_err(|error| error.to_string())?
        .ok_or("the title video isn't in this copy of the game")?;
    std::fs::create_dir_all(dir).map_err(|error| error.to_string())?;
    let video = dir.join(".backdrop.webm");
    let frame = dir.join(".backdrop.png");
    std::fs::write(&video, movie).map_err(|error| error.to_string())?;
    let decoded = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-ss", AT_SECONDS, "-i"])
        .arg(&video)
        .args(["-frames:v", "1"])
        .arg(&frame)
        .status();
    let _ = std::fs::remove_file(&video);
    match decoded {
        Ok(status) if status.success() => {}
        Ok(status) => return Err(format!("ffmpeg failed ({status})")),
        Err(_) => return Err("ffmpeg isn't installed".into()),
    }
    std::fs::rename(&frame, &target).map_err(|error| error.to_string())?;
    Ok(target)
}
