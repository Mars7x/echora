//! Preparing the installed game to run: `echora prepare` translates the game
//! and builds the translation, which takes about an hour the first time. It runs as a child process; its progress
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
    Analyzing,
    Lift(u32, u32),
    /// Region crates built of the total.
    Build(u32, u32),
    Ready,
}

impl Step {
    pub fn status(self) -> (&'static str, Option<u8>) {
        match self {
            Self::Starting => ("Checking game files", None),
            Self::Analyzing => ("Analyzing game code", None),
            Self::Lift(done, total) => ("Translating game", percent(done, total)),
            Self::Build(done, total) if total != 0 && done >= total => ("Finishing build", None),
            Self::Build(done, total) => ("Building game", percent(done, total)),
            Self::Ready => ("Ready to launch", None),
        }
    }
}

fn percent(done: u32, total: u32) -> Option<u8> {
    let completed = u64::from(done.min(total)) * 100;
    Some(completed.checked_div(u64::from(total))? as u8)
}

/// The step a line of `echora prepare`'s output announces, if any.
pub fn parse(line: &str) -> Option<Step> {
    let event = line.strip_prefix(PREFIX)?;
    match event.split_whitespace().collect::<Vec<_>>()[..] {
        ["step", "lift"] => Some(Step::Analyzing),
        ["lifted", done, "of", total] => Some(Step::Lift(done.parse().ok()?, total.parse().ok()?)),
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
        let child = self
            .state
            .lock()
            .ok()
            .and_then(|mut state| state.child.take());
        let status = child.map(|mut child| child.wait());
        let Ok(mut state) = self.state.lock() else {
            return;
        };
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
        assert_eq!(parse("echora-prepare: step lift"), Some(Step::Analyzing));
        assert_eq!(
            parse("echora-prepare: lifted 340 of 1111"),
            Some(Step::Lift(340, 1111))
        );
        assert_eq!(
            parse("echora-prepare: built 340 of 1111"),
            Some(Step::Build(340, 1111))
        );
        assert_eq!(parse("echora-prepare: ready"), Some(Step::Ready));
        assert_eq!(parse("translation: /somewhere"), None);
        assert_eq!(parse("echora-prepare: built many of 1111"), None);
    }

    #[test]
    fn progress_measures_current_stage_and_names_the_remaining_work() {
        assert_eq!(Step::Analyzing.status().1, None);
        assert_eq!(
            Step::Lift(555, 1110).status(),
            ("Translating game", Some(50))
        );
        assert_eq!(Step::Lift(1110, 1110).status().1, Some(100));
        assert_eq!(Step::Build(0, 1110).status(), ("Building game", Some(0)));
        assert_eq!(Step::Build(1110, 1110).status(), ("Finishing build", None));
    }
}
