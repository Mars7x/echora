//! Opens the default output device at 48 kHz stereo and plays a quarter
//! second of silence: checks that host audio works without making a sound.

fn main() {
    match echora_audio::open_device(48_000, 2) {
        Some(sender) => {
            let silence = vec![0u8; 48_000 / 4 * 2 * 2];
            sender.send(silence).expect("audio thread running");
            std::thread::sleep(std::time::Duration::from_millis(300));
            println!("audio output: ok (48 kHz stereo)");
        }
        None => println!("audio output: unavailable"),
    }
}
