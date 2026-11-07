//! Semantic differential test for SIMD loads and stores: lifted LDR/STR
//! (B, H, S, D, Q in every addressing mode), LDP/STP (S, D, Q, including
//! the non-temporal forms), LD1-LD4/ST1-ST4 (multiple structures and single
//! lanes) and LD1R-LD4R, with immediate and register post-indexing,
//! against a reference written from the Arm pseudocode's element loops.
//! Vector registers, general registers, SP and the data memory are
//! compared.
//!
//! Every generated access stays inside the data area: an unmapped guest
//! access ends the host process by design (see `tests/cpu.rs`), so the
//! harness never relies on a fault.
#![cfg(any(target_os = "linux", windows))]
#![allow(clippy::unwrap_used)]

use echora_aot_tests::{Machine, Trap, simd_memory_code};
use echora_lift::aarch64::{Address, Instruction, Writeback, decode};
use echora_lift::simd::SimdOp;

const DATA: u64 = 0x80_0000;
/// Every access stays inside: offsets reach from -1024 (LDP Q pre-index)
/// to 65,520 + 16 (LDR Q, unsigned offset) around the base. An unmapped
/// guest access ends the process by design.
const DATA_SIZE: u64 = 0x2_0000;
const BASE_ADDRESS: u64 = DATA + 0x2000;
const STATES_PER_WORD: usize = 8;
const STRIDE: u64 = 12;

#[derive(Clone, PartialEq, Eq)]
struct State {
    x: [u64; 31],
    sp: u64,
    v: [u128; 32],
    memory: Vec<u8>,
}

impl std::fmt::Debug for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "State {{ x: {:x?}, sp: {:#x}, ", self.x, self.sp)?;
        for (n, value) in self.v.iter().enumerate() {
            write!(f, "v{n}: {value:#034x}, ")?;
        }
        write!(f, ".. }}")
    }
}

struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn read(s: &State, address: u64, bytes: u64) -> Option<u128> {
    let offset = address.checked_sub(DATA)?;
    if offset + bytes > DATA_SIZE {
        return None;
    }
    let mut value = 0;
    for i in 0..bytes {
        value |= u128::from(s.memory[(offset + i) as usize]) << (8 * i);
    }
    Some(value)
}

fn write(s: &mut State, address: u64, bytes: u64, value: u128) -> Option<()> {
    let offset = address.checked_sub(DATA)?;
    if offset + bytes > DATA_SIZE {
        return None;
    }
    for i in 0..bytes {
        s.memory[(offset + i) as usize] = (value >> (8 * i)) as u8;
    }
    Some(())
}

fn mask(bits: u32) -> u128 {
    if bits >= 128 {
        u128::MAX
    } else {
        (1 << bits) - 1
    }
}

fn lane(v: u128, bits: u32, index: u32) -> u128 {
    (v >> (bits * index)) & mask(bits)
}

fn with_lane(v: u128, bits: u32, index: u32, value: u128) -> u128 {
    let at = bits * index;
    (v & !(mask(bits) << at)) | ((value & mask(bits)) << at)
}

fn x(s: &State, n: u8) -> u64 {
    if n == 31 { 0 } else { s.x[n as usize] }
}

fn base(s: &State, n: u8) -> u64 {
    if n == 31 { s.sp } else { s.x[n as usize] }
}

fn set_base(s: &mut State, n: u8, value: u64) {
    if n == 31 {
        s.sp = value;
    } else {
        s.x[n as usize] = value;
    }
}

fn extend(value: u64, option: u8, shift: u8) -> u64 {
    let width = 8 << (option & 3);
    let value = if width == 64 {
        value
    } else {
        value & ((1 << width) - 1)
    };
    let value = if option & 4 != 0 && width < 64 {
        (((value << (64 - width)) as i64) >> (64 - width)) as u64
    } else {
        value
    };
    value << shift
}

/// The access address and the base after writeback.
fn address(s: &State, rn: u8, address: Address) -> (u64, Option<u64>) {
    let b = base(s, rn);
    match address {
        Address::Immediate { offset, writeback } => {
            let moved = b.wrapping_add(offset as u64);
            match writeback {
                Writeback::None => (moved, None),
                Writeback::Pre => (moved, Some(moved)),
                Writeback::Post => (b, Some(moved)),
            }
        }
        Address::Register { rm, option, shift } => {
            (b.wrapping_add(extend(x(s, rm), option, shift)), None)
        }
    }
}

/// A pair transfer of `bytes`-sized registers.
#[allow(clippy::too_many_arguments)]
fn pair(
    s: &mut State,
    rt: u8,
    rt2: u8,
    rn: u8,
    bytes: u64,
    load: bool,
    offset: i64,
    writeback: Writeback,
) -> Option<()> {
    let b = base(s, rn);
    let moved = b.wrapping_add(offset as u64);
    let at = if writeback == Writeback::Post {
        b
    } else {
        moved
    };
    if load {
        let first = read(s, at, bytes)?;
        let second = read(s, at + bytes, bytes)?;
        s.v[rt as usize] = first;
        s.v[rt2 as usize] = second;
    } else {
        let (first, second) = (s.v[rt as usize], s.v[rt2 as usize]);
        write(s, at, bytes, first)?;
        write(s, at + bytes, bytes, second)?;
    }
    if writeback != Writeback::None {
        set_base(s, rn, moved);
    }
    Some(())
}

/// The post-index amount: the transfer size for register 31, else Xm.
fn post_amount(s: &State, post: Option<u8>, size: u64) -> Option<u64> {
    post.map(|rm| if rm == 31 { size } else { x(s, rm) })
}

fn interpret(instruction: &Instruction, s: &mut State) -> Option<()> {
    match *instruction {
        Instruction::LoadStoreVectorPair {
            rt,
            rt2,
            rn,
            load,
            offset,
            writeback,
        } => pair(s, rt, rt2, rn, 16, load, offset, writeback),
        Instruction::Simd(ref op) => simd(op, s),
        ref other => panic!("no reference for {other:?}"),
    }
}

fn simd(op: &SimdOp, s: &mut State) -> Option<()> {
    match *op {
        SimdOp::LoadStore {
            rt,
            rn,
            bytes,
            load,
            address: mode,
        } => {
            let bytes = u64::from(bytes);
            let (at, written_back) = address(s, rn, mode);
            if load {
                s.v[rt as usize] = read(s, at, bytes)?;
            } else {
                let value = s.v[rt as usize];
                write(s, at, bytes, value)?;
            }
            if let Some(value) = written_back {
                set_base(s, rn, value);
            }
        }
        SimdOp::LoadStorePair {
            rt,
            rt2,
            rn,
            bytes,
            load,
            offset,
            writeback,
        } => pair(s, rt, rt2, rn, u64::from(bytes), load, offset, writeback)?,
        SimdOp::TransferMultiple {
            rt,
            rn,
            registers,
            structure,
            bits,
            q,
            load,
            post,
        } => {
            let bits = u32::from(bits);
            let ebytes = u64::from(bits / 8);
            let elements = if q { 128 / bits } else { 64 / bits };
            let (repeat, per_element) = if structure == 1 {
                (u32::from(registers), 1)
            } else {
                (1, u32::from(structure))
            };
            let at = base(s, rn);
            let mut offset = 0;
            // Loads build each register from a datasize-wide read of it:
            // a 64-bit transfer clears the upper half.
            if load {
                for r in 0..u32::from(registers) {
                    let t = (u32::from(rt) + r) % 32;
                    if !q {
                        s.v[t as usize] &= mask(64);
                    }
                }
            }
            for r in 0..repeat {
                for e in 0..elements {
                    let mut t = (u32::from(rt) + r) % 32;
                    for _ in 0..per_element {
                        if load {
                            let value = read(s, at.wrapping_add(offset), ebytes)?;
                            s.v[t as usize] = with_lane(s.v[t as usize], bits, e, value);
                        } else {
                            let value = lane(s.v[t as usize], bits, e);
                            write(s, at.wrapping_add(offset), ebytes, value)?;
                        }
                        offset += ebytes;
                        t = (t + 1) % 32;
                    }
                }
            }
            if let Some(amount) = post_amount(s, post, offset) {
                set_base(s, rn, at.wrapping_add(amount));
            }
        }
        SimdOp::TransferSingle {
            rt,
            rn,
            registers,
            bits,
            index,
            load,
            post,
        } => {
            let bits = u32::from(bits);
            let ebytes = u64::from(bits / 8);
            let at = base(s, rn);
            let mut offset = 0;
            for r in 0..u32::from(registers) {
                let t = ((u32::from(rt) + r) % 32) as usize;
                if load {
                    let value = read(s, at.wrapping_add(offset), ebytes)?;
                    s.v[t] = with_lane(s.v[t], bits, u32::from(index), value);
                } else {
                    let value = lane(s.v[t], bits, u32::from(index));
                    write(s, at.wrapping_add(offset), ebytes, value)?;
                }
                offset += ebytes;
            }
            if let Some(amount) = post_amount(s, post, offset) {
                set_base(s, rn, at.wrapping_add(amount));
            }
        }
        SimdOp::LoadReplicate {
            rt,
            rn,
            registers,
            bits,
            q,
            post,
        } => {
            let bits = u32::from(bits);
            let ebytes = u64::from(bits / 8);
            let elements = if q { 128 / bits } else { 64 / bits };
            let at = base(s, rn);
            let mut offset = 0;
            for r in 0..u32::from(registers) {
                let t = ((u32::from(rt) + r) % 32) as usize;
                let value = read(s, at.wrapping_add(offset), ebytes)?;
                s.v[t] = (0..elements).fold(0, |v, e| with_lane(v, bits, e, value));
                offset += ebytes;
            }
            if let Some(amount) = post_amount(s, post, offset) {
                set_base(s, rn, at.wrapping_add(amount));
            }
        }
        ref other => panic!("no reference for {other:?}"),
    }
    Some(())
}

/// The base register and any index or post-index register.
fn address_registers(instruction: &Instruction) -> (u8, Option<u8>) {
    match *instruction {
        Instruction::LoadStoreVectorPair { rn, .. } => (rn, None),
        Instruction::Simd(ref op) => match *op {
            SimdOp::LoadStore { rn, address, .. } => (
                rn,
                match address {
                    Address::Register { rm, .. } => Some(rm),
                    Address::Immediate { .. } => None,
                },
            ),
            SimdOp::LoadStorePair { rn, .. } => (rn, None),
            SimdOp::TransferMultiple { rn, post, .. }
            | SimdOp::TransferSingle { rn, post, .. }
            | SimdOp::LoadReplicate { rn, post, .. } => (rn, post.filter(|&rm| rm != 31)),
            ref other => panic!("no addressing for {other:?}"),
        },
        ref other => panic!("no addressing for {other:?}"),
    }
}

fn run_lifted(m: &mut Machine, entry: u64, state: &State) -> Result<State, Trap> {
    m.core.cpu.regs = state.x;
    m.core.cpu.sp = state.sp;
    m.core.cpu.vectors = state.v;
    m.core.cpu.pc = entry;
    m.memory.write(DATA, &state.memory).expect("data is mapped");
    let trap = loop {
        match simd_memory_code::dispatch(&mut m.core) {
            Ok(()) => continue,
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
        v: m.core.cpu.vectors,
        memory,
    })
}

fn difference(expected: &State, actual: &State) -> String {
    let mut parts = Vec::new();
    for n in 0..32 {
        if expected.v[n] != actual.v[n] {
            parts.push(format!(
                "v{n}: expected {:#034x}, lifted {:#034x}",
                expected.v[n], actual.v[n]
            ));
        }
    }
    for n in 0..31 {
        if expected.x[n] != actual.x[n] {
            parts.push(format!(
                "x{n}: expected {:#x}, lifted {:#x}",
                expected.x[n], actual.x[n]
            ));
        }
    }
    if expected.sp != actual.sp {
        parts.push(format!(
            "sp: expected {:#x}, lifted {:#x}",
            expected.sp, actual.sp
        ));
    }
    if let Some(at) = (0..expected.memory.len()).find(|&i| expected.memory[i] != actual.memory[i]) {
        parts.push(format!(
            "memory at {:#x}: expected {:#04x}, lifted {:#04x}",
            DATA + at as u64,
            expected.memory[at],
            actual.memory[at]
        ));
    }
    parts.join("; ")
}

#[test]
fn lifted_simd_loads_and_stores_match_the_reference() {
    let words = simd_memory_code::WORDS;
    let code_size = (words.len() as u64 * STRIDE).next_multiple_of(0x1000);
    let mut image = Vec::new();
    for word in words {
        image.extend(word.to_le_bytes());
        image.extend(0xd400_00e1_u32.to_le_bytes());
        image.extend(0xd400_0101_u32.to_le_bytes());
    }
    let mut m = Machine::with_layout(simd_memory_code::BASE, &image, code_size, DATA, DATA_SIZE);
    simd_memory_code::install();
    let mut random = Random(0xbb67_ae85_84ca_a73b);
    let memory: Vec<u8> = (0..DATA_SIZE).map(|_| random.next() as u8).collect();
    let mut failures = Vec::new();
    let mut runs = 0;
    for (index, &word) in words.iter().enumerate() {
        let entry = simd_memory_code::BASE + STRIDE * index as u64;
        let instruction = decode(word, entry);
        let (rn, index_register) = address_registers(&instruction);
        for _ in 0..STATES_PER_WORD {
            let mut before = State {
                x: std::array::from_fn(|_| random.next()),
                sp: random.next(),
                v: std::array::from_fn(|_| {
                    (u128::from(random.next()) << 64) | u128::from(random.next())
                }),
                memory: memory.clone(),
            };
            // The index first, so a base that is also the index wins.
            if let Some(rm) = index_register.filter(|&rm| rm != 31) {
                before.x[rm as usize] = random.next() % 0x100;
            }
            // SP-based accesses check SP's 16-byte alignment (SCTLR_EL1.SA0,
            // as the guest OS sets it), so an SP base stays aligned.
            let offset = random.next() % 64;
            let offset = if rn == 31 { offset & !15 } else { offset };
            set_base(&mut before, rn, BASE_ADDRESS + offset);
            let mut expected = before.clone();
            let reference = interpret(&instruction, &mut expected);
            runs += 1;
            let problem = match (reference, run_lifted(&mut m, entry, &before)) {
                (Some(()), Ok(actual)) if actual == expected => continue,
                (Some(()), Ok(actual)) => difference(&expected, &actual),
                (Some(()), Err(trap)) => format!("lifted trapped: {trap:?}"),
                (None, Ok(_)) => "the reference access left the data".to_owned(),
                (None, Err(_)) => continue,
            };
            failures.push(format!(
                "{word:#010x} {instruction:?}: {problem}\n    from {before:?}"
            ));
            break;
        }
    }
    println!(
        "{} instructions, {runs} runs, {} mismatching instructions",
        words.len(),
        failures.len()
    );
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures
            .iter()
            .take(30)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}
