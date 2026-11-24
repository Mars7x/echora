//! The call-shaped benchmark kernels (`calls`, `calls2`, `gamecalls`,
//! `crossing`, `gamefn`) as `build.rs` lifts them for the benchmark (the build's
//! options, 64 KiB regions), run at small iteration counts against the
//! shared reference interpreter extended with BL and RET: every register,
//! SP, NZCV and the data memory (objects and stack) must match at the
//! final SVC, run whole and sliced into host boundaries. With
//! `long-windows` this covers windows running a function across a window
//! edge (`crossing`).
#![cfg(any(target_os = "linux", windows))]
#![allow(clippy::unwrap_used)]

mod common;

use common::*;
use echora_aot_tests::kernels::{KERNEL_BASE, KERNEL_ENTRIES, KERNEL_SIZE, image};
use echora_aot_tests::{Machine, Trap, kernel_code};
use echora_lift::aarch64::{Address, Instruction, MemoryOp, decode};

const SVC_EXIT: u32 = 0xd400_00e1;

fn word_at(image: &[u8], pc: u64) -> Option<u32> {
    let offset = pc.checked_sub(KERNEL_BASE)? as usize;
    Some(u32::from_le_bytes(
        image.get(offset..offset + 4)?.try_into().ok()?,
    ))
}

/// The reference run from `entry` to its SVC #7, or None.
fn reference(image: &[u8], entry: u64, state: &mut State) -> Option<u64> {
    let mut pc = entry;
    for steps in 0..5_000_000u64 {
        let word = word_at(image, pc)?;
        if word == SVC_EXIT {
            return Some(steps);
        }
        let instruction = decode(word, pc);
        pc = match instruction {
            Instruction::Branch { target, link } => {
                if link {
                    state.x[30] = pc + 4;
                }
                target
            }
            Instruction::Adr { rd, address } => {
                set(state, rd, address);
                pc + 4
            }
            // A load from the code (a jump table's halfwords).
            Instruction::LoadStore {
                rt,
                rn,
                bytes,
                op: op @ MemoryOp::Load { .. },
                address: Address::Register { rm, option, shift },
            } if word_at(
                image,
                get_sp(state, rn).wrapping_add(extend(get(state, rm), option, shift, 64)) & !3,
            )
            .is_some() =>
            {
                let at = get_sp(state, rn).wrapping_add(extend(get(state, rm), option, shift, 64));
                let mut value = 0u64;
                for byte in 0..u64::from(bytes) {
                    let word = word_at(image, (at + byte) & !3)?;
                    value |= u64::from((word >> (8 * ((at + byte) & 3))) as u8) << (8 * byte);
                }
                set(state, rt, loaded(value, u64::from(bytes), op));
                pc + 4
            }
            Instruction::BranchReg { rn, link } => {
                let target = get(state, rn);
                if link {
                    state.x[30] = pc + 4;
                }
                target
            }
            _ => {
                let taken = interpret(&instruction, state)?;
                match instruction {
                    Instruction::BranchCond { target, .. }
                    | Instruction::CompareBranch { target, .. }
                    | Instruction::TestBranch { target, .. }
                        if taken =>
                    {
                        target
                    }
                    _ => pc + 4,
                }
            }
        };
    }
    None
}

/// The lifted run of the same: the end state, or the trap.
fn lifted(m: &mut Machine, entry: u64, start: &State, slice: u64) -> Result<State, Trap> {
    m.core.cpu.regs = start.x;
    m.core.cpu.sp = start.sp;
    m.core.cpu.nzcv = start.nzcv;
    m.core.cpu.pc = entry;
    m.memory.write(DATA, &start.memory).expect("data is mapped");
    m.core.max_steps = u64::MAX;
    m.core.host_check_at = m.core.steps + slice;
    m.core.rearm();
    let trap = loop {
        match kernel_code::dispatch(&mut m.core) {
            Ok(()) => continue,
            Err(Trap::HostBoundary) => {
                m.core.host_check_at = m.core.steps + slice;
                m.core.rearm();
            }
            Err(trap) => break trap,
        }
    };
    m.core.settle_block_trap();
    if !matches!(trap, Trap::SupervisorCall { number: 7 }) {
        return Err(trap);
    }
    let mut memory = vec![0; DATA_SIZE as usize];
    m.memory.read(DATA, &mut memory).expect("data is mapped");
    Ok(State {
        x: m.core.cpu.regs,
        sp: m.core.cpu.sp,
        nzcv: m.core.cpu.nzcv,
        memory,
    })
}

#[test]
fn call_kernels_match_the_reference() {
    kernel_code::install();
    let image = image();
    let mut m = Machine::with_layout(
        KERNEL_BASE,
        &image,
        KERNEL_SIZE.next_multiple_of(0x1000),
        DATA,
        DATA_SIZE,
    );
    let mut random = Random(0x5eed_c0de_1234_5678);
    let mut runs = 0;
    for &(name, offset) in KERNEL_ENTRIES {
        let iterations = match name {
            "calls" | "calls2" => 300,
            "gamecalls" | "crossing" | "gamefn" | "vcalls" => 200,
            _ => continue,
        };
        for _ in 0..3 {
            let mut start = State {
                x: std::array::from_fn(|_| random.next()),
                sp: DATA + DATA_SIZE,
                nzcv: (random.next() & 15) as u8,
                memory: (0..DATA_SIZE).map(|_| random.next() as u8).collect(),
            };
            start.x[0] = iterations;
            // The objects (64 of 64 bytes) at the data's start; the stack
            // grows down from its end.
            start.x[1] = DATA;
            let mut expected = start.clone();
            let steps = reference(&image, KERNEL_BASE + offset, &mut expected)
                .unwrap_or_else(|| panic!("{name}: the reference did not finish"));
            for slice in [u64::MAX / 4, 7, 61] {
                let actual = lifted(&mut m, KERNEL_BASE + offset, &start, slice)
                    .unwrap_or_else(|trap| panic!("{name} (slice {slice}): {trap:?}"));
                assert_eq!(actual.x, expected.x, "{name} (slice {slice}): registers");
                assert_eq!(actual.sp, expected.sp, "{name} (slice {slice}): SP");
                assert_eq!(actual.nzcv, expected.nzcv, "{name} (slice {slice}): NZCV");
                assert!(
                    actual.memory == expected.memory,
                    "{name} (slice {slice}): memory"
                );
                runs += 1;
            }
            assert!(steps > iterations, "{name}");
        }
    }
    assert_eq!(runs, 6 * 3 * 3);
}
