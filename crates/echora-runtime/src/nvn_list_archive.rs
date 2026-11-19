//! Compact host storage for recorded lists outside the replay cache.

use super::{Command, StateSnapshots};
use std::mem::{size_of, size_of_val};

#[derive(Debug)]
struct Header {
    name: &'static str,
    capture_count: usize,
    registers: u16,
    has_state: bool,
}

#[derive(Debug)]
pub(super) struct ArchivedList {
    headers: Box<[Header]>,
    words: Box<[u64]>,
    snapshots: Box<[StateSnapshots]>,
}

impl ArchivedList {
    pub(super) fn pack(commands: &[Command]) -> Self {
        let mut headers = Vec::with_capacity(commands.len());
        let mut words = Vec::new();
        let mut snapshots = Vec::new();
        for command in commands {
            let mut registers = 0;
            for (index, value) in command.x.into_iter().chain(command.v).enumerate() {
                if value != 0 {
                    registers |= 1u16 << index;
                    words.push(value);
                }
            }
            words.extend_from_slice(&command.captured);
            let has_state = command.states.0.is_some();
            if has_state {
                snapshots.push(command.states.clone());
            }
            headers.push(Header {
                name: command.name,
                capture_count: command.captured.len(),
                registers,
                has_state,
            });
        }
        Self {
            headers: headers.into_boxed_slice(),
            words: words.into_boxed_slice(),
            snapshots: snapshots.into_boxed_slice(),
        }
    }

    pub(super) fn unpack(self) -> Vec<Command> {
        let mut commands = Vec::with_capacity(self.headers.len());
        let mut next = 0;
        let mut snapshots = self.snapshots.into_iter();
        for entry in self.headers {
            let mut registers = [0; 16];
            let mut mask = entry.registers;
            while mask != 0 {
                registers[mask.trailing_zeros() as usize] = self.words[next];
                next += 1;
                mask &= mask - 1;
            }
            let (x, v) = registers.split_at(8);
            let end = next + entry.capture_count;
            commands.push(Command {
                name: entry.name,
                x: x.try_into().expect("eight integer registers"),
                v: v.try_into().expect("eight vector registers"),
                captured: self.words[next..end].to_vec(),
                states: if entry.has_state {
                    snapshots.next().expect("state recorded for this command")
                } else {
                    StateSnapshots::default()
                },
            });
            next = end;
        }
        debug_assert_eq!(next, self.words.len());
        debug_assert!(snapshots.next().is_none());
        commands
    }

    pub(super) fn bytes(&self) -> usize {
        size_of::<Self>()
            + size_of_val(&*self.headers)
            + size_of_val(&*self.words)
            + size_of_val(&*self.snapshots)
    }
}
