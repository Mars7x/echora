//! Preparing the installed game to run: `echora prepare` fetches the shader
//! recompiler, translates the game and builds the translation, which takes
//! about an hour the first time. It runs as a child process; its progress
//! lines (`echora-prepare: ...`) are followed here for the launcher to show.

use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

const PREFIX: &str = "echora-prepare: ";

/// What the preparation is doing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Step {
    #[default]
    Starting,
    Fetch,
    Lift,
    /// Region crates built of the total.
    Build(u32, u32),
    Ready,
}

impl Step {
    /// What the player reads under the title.
    pub fn describe(self) -> String {
        match self {
            Self::Starting | Self::Fetch => "Getting the shader recompiler".into(),
            Self::Lift => "Translating the game".into(),
            Self::Build(built, total) => {
                format!(
                    "Building the game, {} of {}",
                    grouped(built),
                    grouped(total)
                )
            }
            Self::Ready => "Ready".into(),
        }
    }

    /// Overall progress from 0 to 1. Building takes nearly all the time.
    pub fn fraction(self) -> f32 {
        match self {
            Self::Starting => 0.0,
            Self::Fetch => 0.01,
            Self::Lift => 0.03,
            Self::Build(_, 0) => 0.1,
            Self::Build(built, total) => 0.1 + 0.9 * (built as f32 / total as f32).min(1.0),
            Self::Ready => 1.0,
        }
    }
}

/// `n` with thousands separated: 1,111.
fn grouped(n: u32) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, digit) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(digit);
    }
    out
}

/// The step a line of `echora prepare`'s output announces, if any.
pub fn parse(line: &str) -> Option<Step> {
    let event = line.strip_prefix(PREFIX)?;
    match event.split_whitespace().collect::<Vec<_>>()[..] {
        ["step", "fetch"] => Some(Step::Fetch),
        ["step", "lift"] => Some(Step::Lift),
        ["step", "build"] => Some(Step::Build(0, 0)),
        ["built", built, "of", total] => {
            Some(Step::Build(built.parse().ok()?, total.parse().ok()?))
        }
        ["ready"] => Some(Step::Ready),
        _ => None,
    }
}

/// `echora prepare GAME --check`: whether the game can start right away.
/// When the check can't be started at all, the answer is yes, and starting
/// the game reports the real problem.
pub fn is_prepared(mut command: Command) -> bool {
    command.stdout(Stdio::null()).stderr(Stdio::null());
    match command.status() {
        Ok(status) => status.success(),
        Err(_) => true,
    }
}

/// A preparation running in the background.
#[derive(Default)]
pub struct Preparation {
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    step: Step,
    /// The last line of error output, for a failure's message.
    last_error: Option<String>,
    result: Option<Result<(), String>>,
    child: Option<Child>,
}

impl Preparation {
    pub fn step(&self) -> Step {
        self.state
            .lock()
            .map_or_else(|_| Step::default(), |state| state.step)
    }

    /// Success or the error, once the preparation has ended.
    pub fn take_result(&self) -> Option<Result<(), String>> {
        self.state.lock().ok()?.result.take()
    }

    /// Starts `command` (`echora prepare GAME`) and follows it on other
    /// threads.
    pub fn start(self: &std::sync::Arc<Self>, mut command: Command) -> std::io::Result<()> {
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        if let Ok(mut state) = self.state.lock() {
            state.child = Some(child);
        }
        let errors = stderr.map(|stderr| {
            let this = self.clone();
            std::thread::spawn(move || {
                for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                    let line = line.trim();
                    if !line.is_empty()
                        && let Ok(mut state) = this.state.lock()
                    {
                        state.last_error = Some(line.to_owned());
                    }
                }
            })
        });
        let this = self.clone();
        std::thread::spawn(move || {
            if let Some(stdout) = stdout {
                for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                    if let Some(step) = parse(&line)
                        && let Ok(mut state) = this.state.lock()
                    {
                        state.step = step;
                    }
                }
            }
            // The error output's last line names what went wrong.
            if let Some(errors) = errors {
                let _ = errors.join();
            }
            this.finish();
        });
        Ok(())
    }

    fn finish(&self) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        let status = state.child.take().map(|mut child| child.wait());
        state.result = Some(match status {
            Some(Ok(status)) if status.success() => Ok(()),
            _ => Err(state
                .last_error
                .take()
                .map(|line| line.trim_start_matches("error: ").to_owned())
                .unwrap_or_else(|| "the preparation stopped".into())),
        });
    }
}

impl Drop for Preparation {
    /// Closing the launcher stops a preparation still running.
    fn drop(&mut self) {
        if let Ok(state) = self.state.get_mut()
            && let Some(child) = state.child.as_mut()
        {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// `echora prepare GAME` (with `--check`, only asks).
pub fn command(echora: &Path, game: &Path, check: bool) -> Command {
    let mut command = Command::new(echora);
    command.arg("prepare").arg(game);
    if check {
        command.arg("--check");
    } else {
        // The launcher holds its standard input open, so the preparation
        // ends with the launcher.
        command.arg("--stop-with-stdin");
    }
    command
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_lines_become_steps() {
        assert_eq!(parse("echora-prepare: step fetch"), Some(Step::Fetch));
        assert_eq!(parse("echora-prepare: step lift"), Some(Step::Lift));
        assert_eq!(
            parse("echora-prepare: built 340 of 1111"),
            Some(Step::Build(340, 1111))
        );
        assert_eq!(parse("echora-prepare: ready"), Some(Step::Ready));
        assert_eq!(parse("translation: /somewhere"), None);
        assert_eq!(parse("echora-prepare: built many of 1111"), None);
    }

    #[test]
    fn counts_are_grouped() {
        assert_eq!(
            Step::Build(340, 1111).describe(),
            "Building the game, 340 of 1,111"
        );
        assert_eq!(grouped(0), "0");
        assert_eq!(grouped(1_234_567), "1,234,567");
    }

    #[test]
    fn building_takes_most_of_the_bar() {
        assert!(Step::Lift.fraction() < Step::Build(0, 1111).fraction());
        assert_eq!(Step::Build(1111, 1111).fraction(), 1.0);
        assert!((Step::Build(555, 1110).fraction() - 0.55).abs() < 1e-6);
    }
}
