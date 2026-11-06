//! The Echora launcher: choose the game, adjust the settings the runtime
//! applies (`echora-config`), and start it.
//!
//! Built with GPUI, so the same front end runs on Linux, Windows and macOS.
//! `--show` opens the window even when "Skip launcher" is on.

mod app;

fn main() -> std::process::ExitCode {
    app::run()
}
