//! The Echora launcher: choose the game, see what it is, adjust the
//! settings the runtime applies, and start it.
//!
//! GTK4 and libadwaita on Linux. Other platforms will get their own front
//! end on the same settings model (`echora-config`).

#[cfg(target_os = "linux")]
mod app;

#[cfg(target_os = "linux")]
fn main() -> std::process::ExitCode {
    app::run()
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("The Echora launcher is available on Linux only; use the echora command.");
    std::process::exit(1);
}
