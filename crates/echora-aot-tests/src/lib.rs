//! A minimal host for running lifted synthetic programs natively.
//!
//! It provides only what CPU-level tests need: guest memory, a core, and a
//! host loop in which SVC #7 (ExitProcess) ends the run and any other SVC is
//! reported as unsupported. Kernel and service behavior belongs to the
//! runtime and is tested there.

#![allow(
    missing_debug_implementations,
    clippy::all,
    unused_imports,
    unused_parens,
    unreachable_code
)]

pub use echora_guest::cpu::Cpu;
pub use echora_guest::memory::{EXECUTE, Memory, READ, STATE_CODE, STATE_NORMAL, WRITE};
pub use echora_guest::{Core, Trap};
// The SIMD/FP operations generated code calls (see `echora_guest::simd`).
extern crate echora_simd;

mod programs_list {
    include!("programs.rs");
}
pub use programs_list::{CODE_BASE, PROGRAMS};

/// The benchmark kernels (`benches/kernels.rs`), lifted as `kernel_code`.
pub mod kernels {
    include!("kernels.rs");
}

/// A stand-in for `Core::call_window` until echora-guest has it, so direct
/// calls (`--features direct-calls`) can be run: the callee's window
/// function, then on as `Core::call` goes. It nests at most as deep as
/// `Core::call` does; deeper, the window is left to `Core::call` (its
/// table finds the same function). An inherent `call_window` takes
/// precedence over this trait's.
pub trait CallWindow {
    fn call_window(&mut self, return_to: u64, function: echora_guest::WindowFn)
    -> Result<(), Trap>;
}

thread_local! {
    static SHIM_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

impl CallWindow for Core {
    fn call_window(
        &mut self,
        return_to: u64,
        function: echora_guest::WindowFn,
    ) -> Result<(), Trap> {
        let depth = SHIM_DEPTH.get();
        if depth < 48 && self.cpu.pc != return_to {
            SHIM_DEPTH.set(depth + 1);
            let result = function(self);
            SHIM_DEPTH.set(depth);
            result?;
        }
        self.call(return_to)
    }
}

pub struct Machine {
    pub memory: Memory,
    pub core: Core,
}

/// The process's 39-bit guest address space (the model PAuth stripping
/// depends on). The host reservation is sparse.
pub const ADDRESS_SPACE: u64 = 1 << 39;

impl Machine {
    /// Code mapped at 0x1000 (one page, read/execute) and data at 0x8000
    /// (two pages, read/write); PC at the code base; a 100-instruction budget.
    pub fn new() -> Self {
        let mut memory = Memory::new(ADDRESS_SPACE);
        memory
            .map(0x1000, 0x1000, READ | EXECUTE, STATE_CODE, "synthetic code")
            .expect("map code");
        memory
            .map(0x8000, 0x2000, READ | WRITE, STATE_NORMAL, "data")
            .expect("map data");
        let mut core = Core::new(memory.view());
        core.cpu.pc = CODE_BASE;
        core.max_steps = 100;
        core.host_check_at = u64::MAX;
        Self { memory, core }
    }

    /// Code of `code_size` bytes from `code_base` holding `code`
    /// (read/execute), and `data_size` bytes of zeroed data from `data_base`
    /// (read/write). No instruction budget and no host checks.
    pub fn with_layout(
        code_base: u64,
        code: &[u8],
        code_size: u64,
        data_base: u64,
        data_size: u64,
    ) -> Self {
        let mut memory = Memory::new(ADDRESS_SPACE);
        memory
            .map(code_base, code_size, READ | EXECUTE, STATE_CODE, "code")
            .expect("map code");
        memory.initialize(code_base, code).expect("load code");
        memory
            .map(data_base, data_size, READ | WRITE, STATE_NORMAL, "data")
            .expect("map data");
        let mut core = Core::new(memory.view());
        core.cpu.pc = code_base;
        core.max_steps = u64::MAX;
        core.host_check_at = u64::MAX;
        Self { memory, core }
    }

    /// Runs `dispatch` until a trap ends the program.
    pub fn run(&mut self, dispatch: fn(&mut Core) -> Result<(), Trap>) -> Result<(), Trap> {
        loop {
            let trap = match dispatch(&mut self.core) {
                Ok(()) => continue,
                Err(trap) => trap,
            };
            self.core.settle_block_trap();
            if matches!(trap, Trap::SupervisorCall { .. }) {
                // Exception entry and return clear the local monitor.
                self.core.memory.clear_exclusive();
            }
            return Err(match trap {
                Trap::SupervisorCall { number: 7 } => Trap::Exit { code: 0 },
                Trap::SupervisorCall { number } => Trap::UnsupportedSvc {
                    pc: self.core.cpu.pc,
                    number,
                },
                other => other,
            });
        }
    }
}

impl Default for Machine {
    fn default() -> Self {
        Self::new()
    }
}

include!(concat!(env!("OUT_DIR"), "/programs.rs"));
