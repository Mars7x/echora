use serde::Serialize;
use std::fmt;

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind")]
pub enum Trap {
    UnsupportedInstruction {
        pc: u64,
        word: u32,
    },
    MissingBlock {
        pc: u64,
    },
    Budget {
        pc: u64,
        steps: u64,
    },
    MemoryFault {
        address: u64,
        access: String,
    },
    UnsupportedSvc {
        pc: u64,
        number: u16,
    },
    UnsupportedService {
        pc: u64,
        service: String,
        command_id: Option<u32>,
        message: String,
    },
    Exit {
        code: u64,
    },
    InvalidState {
        message: String,
    },
    NoRunnableThreads {
        waiting_threads: usize,
    },
    /// Control transfer, not an error: generated code reached SVC #number
    /// and returns to the host loop, which performs it.
    SupervisorCall {
        number: u16,
    },
    /// Control transfer, not an error: a block completed while the host has
    /// pending scheduling or exit work for this boundary.
    HostBoundary,
    GuestBreak {
        pc: u64,
        reason: u64,
        address: u64,
        size: u64,
    },
}

impl fmt::Display for Trap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for Trap {}
