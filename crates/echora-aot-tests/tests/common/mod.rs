//! The integer reference interpreter shared by the semantic tests: Arm
//! pseudocode for the integer, flag, branch, CRC32, atomic and memory
//! instructions over a register, SP, NZCV and data-memory state.
#![allow(dead_code)]

use echora_lift::aarch64::{
    Address, AtomicOp, BitfieldOp, CompareOperand, Instruction, LogicalOp, MemoryOp, WideOp,
    Writeback,
};

pub const DATA: u64 = 0x80_0000;
pub const DATA_SIZE: u64 = 0x1_0000;
/// Base registers point here, so every immediate offset stays mapped.
pub const BASE_ADDRESS: u64 = DATA + DATA_SIZE / 2;

#[derive(Clone, PartialEq, Eq)]
pub struct State {
    pub x: [u64; 31],
    pub sp: u64,
    pub nzcv: u8,
    pub memory: Vec<u8>,
}

impl std::fmt::Debug for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("State")
            .field("x", &self.x)
            .field("sp", &format_args!("{:#x}", self.sp))
            .field("nzcv", &self.nzcv)
            .finish_non_exhaustive()
    }
}

pub struct Random(pub u64);

impl Random {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Mostly random, sometimes an edge value.
    pub fn value(&mut self) -> u64 {
        const EDGES: [u64; 10] = [
            0,
            1,
            u64::MAX,
            0x7fff_ffff,
            0x8000_0000,
            0xffff_ffff,
            0x1_0000_0000,
            i64::MAX as u64,
            i64::MIN as u64,
            0xffff_ffff_8000_0000,
        ];
        let pick = self.next();
        if pick.is_multiple_of(4) {
            EDGES[(pick >> 8) as usize % EDGES.len()]
        } else {
            self.next()
        }
    }
}

// Reference semantics (Arm A64 pseudocode).

pub fn mask(bits: u32) -> u64 {
    if bits == 64 {
        u64::MAX
    } else {
        (1 << bits) - 1
    }
}

pub fn sign_extend(value: u64, bits: u32) -> u64 {
    if bits >= 64 {
        value
    } else {
        (((value << (64 - bits)) as i64) >> (64 - bits)) as u64
    }
}

/// AddWithCarry: the result and NZCV.
pub fn add_with_carry(x: u64, y: u64, carry: bool, bits: u32) -> (u64, u8) {
    let m = mask(bits);
    let (x, y) = (x & m, y & m);
    let unsigned = u128::from(x) + u128::from(y) + u128::from(carry);
    let result = (unsigned as u64) & m;
    let signed = i128::from(sign_extend(x, bits) as i64)
        + i128::from(sign_extend(y, bits) as i64)
        + i128::from(carry);
    let n = (result >> (bits - 1)) & 1 == 1;
    let z = result == 0;
    let c = unsigned != u128::from(result);
    let v = signed != i128::from(sign_extend(result, bits) as i64);
    (result, nzcv(n, z, c, v))
}

pub fn nzcv(n: bool, z: bool, c: bool, v: bool) -> u8 {
    (u8::from(n) << 3) | (u8::from(z) << 2) | (u8::from(c) << 1) | u8::from(v)
}

pub fn condition_holds(condition: u8, flags: u8) -> bool {
    let (n, z, c, v) = (
        flags & 8 != 0,
        flags & 4 != 0,
        flags & 2 != 0,
        flags & 1 != 0,
    );
    let result = match condition >> 1 {
        0 => z,
        1 => c,
        2 => n,
        3 => v,
        4 => c && !z,
        5 => n == v,
        6 => n == v && !z,
        _ => true,
    };
    if condition & 1 == 1 && condition != 15 {
        !result
    } else {
        result
    }
}

pub fn shift(value: u64, kind: u8, amount: u32, bits: u32) -> u64 {
    let value = value & mask(bits);
    let amount = amount % bits;
    let result = match kind {
        0 => value << amount,
        1 => value >> amount,
        2 => (sign_extend(value, bits) as i64 >> amount) as u64,
        _ => {
            if amount == 0 {
                value
            } else {
                (value >> amount) | (value << (bits - amount))
            }
        }
    };
    result & mask(bits)
}

/// ExtendReg: UXTB..SXTX (option 0..7), then a left shift.
pub fn extend(value: u64, option: u8, amount: u8, bits: u32) -> u64 {
    let width = 8 << (option & 3);
    let value = value & mask(width);
    let value = if option & 4 != 0 {
        sign_extend(value, width)
    } else {
        value
    };
    (value << amount) & mask(bits)
}

/// DecodeBitMasks (wmask, tmask) for the bitfield instructions, written
/// independently of the decoder's.
pub fn bit_masks(n: u32, imms: u32, immr: u32, bits: u32) -> (u64, u64) {
    let len = 31 - ((n << 6) | (!imms & 0x3f)).leading_zeros();
    let size = 1 << len;
    let levels = size - 1;
    let s = imms & levels;
    let r = immr & levels;
    let diff = s.wrapping_sub(r) & levels;
    let welem = mask(s + 1);
    let telem = mask(diff + 1);
    let rotate = |value: u64| {
        if r == 0 {
            value
        } else {
            ((value >> r) | (value << (size - r))) & mask(size)
        }
    };
    let replicate = |element: u64| {
        let mut out = 0;
        let mut at = 0;
        while at < bits {
            out |= element << at;
            at += size;
        }
        out & mask(bits)
    };
    (replicate(rotate(welem)), replicate(telem))
}

pub fn read(state: &State, address: u64, bytes: u64) -> Option<u64> {
    let offset = address.checked_sub(DATA)?;
    if offset + bytes > DATA_SIZE {
        return None;
    }
    let mut value = 0;
    for i in 0..bytes {
        value |= u64::from(state.memory[(offset + i) as usize]) << (8 * i);
    }
    Some(value)
}

pub fn write(state: &mut State, address: u64, bytes: u64, value: u64) -> Option<()> {
    let offset = address.checked_sub(DATA)?;
    if offset + bytes > DATA_SIZE {
        return None;
    }
    for i in 0..bytes {
        state.memory[(offset + i) as usize] = (value >> (8 * i)) as u8;
    }
    Some(())
}

/// Register 31 as the zero register.
pub fn get(state: &State, n: u8) -> u64 {
    if n == 31 { 0 } else { state.x[n as usize] }
}

/// Register 31 as SP.
pub fn get_sp(state: &State, n: u8) -> u64 {
    if n == 31 {
        state.sp
    } else {
        state.x[n as usize]
    }
}

pub fn set(state: &mut State, n: u8, value: u64) {
    if n != 31 {
        state.x[n as usize] = value;
    }
}

pub fn set_sp(state: &mut State, n: u8, value: u64) {
    if n == 31 {
        state.sp = value;
    } else {
        state.x[n as usize] = value;
    }
}

/// The value of a load of `bytes` with `op`'s extension.
pub fn loaded(value: u64, bytes: u64, op: MemoryOp) -> u64 {
    match op {
        MemoryOp::Load { signed: true, bits } => {
            sign_extend(value, 8 * bytes as u32) & mask(u32::from(bits))
        }
        _ => value,
    }
}

/// Runs `instruction` on `state`: whether it branched, or `None` for an
/// access outside the data.
pub fn interpret(instruction: &Instruction, state: &mut State) -> Option<bool> {
    use Instruction::*;
    match *instruction {
        BranchCond { condition, .. } => return Some(condition_holds(condition, state.nzcv)),
        CompareBranch {
            rt, bits, nonzero, ..
        } => {
            let zero = get(state, rt) & mask(u32::from(bits)) == 0;
            return Some(zero != nonzero);
        }
        TestBranch {
            rt, bit, nonzero, ..
        } => {
            let set = (get(state, rt) >> bit) & 1 == 1;
            return Some(set == nonzero);
        }
        Crc32 {
            rd,
            rn,
            rm,
            bytes,
            castagnoli,
        } => {
            let polynomial: u32 = if castagnoli { 0x82f6_3b78 } else { 0xedb8_8320 };
            let mut crc = get(state, rn) as u32;
            let data = get(state, rm);
            for i in 0..bytes {
                crc ^= u32::from((data >> (8 * i)) as u8);
                for _ in 0..8 {
                    crc = (crc >> 1) ^ (polynomial & (crc & 1).wrapping_neg());
                }
            }
            set(state, rd, u64::from(crc));
        }
        CompareAndSwap {
            rs, rt, rn, bytes, ..
        } => {
            let bytes = u64::from(bytes);
            let address = get_sp(state, rn);
            let old = read(state, address, bytes)?;
            if old == get(state, rs) & mask(8 * bytes as u32) {
                write(state, address, bytes, get(state, rt))?;
            }
            set(state, rs, old);
        }
        CompareAndSwapPair {
            rs, rt, rn, bytes, ..
        } => {
            let bytes = u64::from(bytes);
            let m = mask(8 * bytes as u32);
            let address = get_sp(state, rn);
            let first = read(state, address, bytes)?;
            let second = read(state, address + bytes, bytes)?;
            if first == get(state, rs) & m && second == get(state, rs + 1) & m {
                let (new_first, new_second) = (get(state, rt), get(state, rt + 1));
                write(state, address, bytes, new_first)?;
                write(state, address + bytes, bytes, new_second)?;
            }
            set(state, rs, first);
            set(state, rs + 1, second);
        }
        AtomicAdd {
            rs, rt, rn, bytes, ..
        } => {
            atomic(state, AtomicOp::Swap, rs, rt, rn, bytes, true)?;
        }
        AtomicMemory {
            op,
            rs,
            rt,
            rn,
            bytes,
            ..
        } => {
            atomic(state, op, rs, rt, rn, bytes, false)?;
        }
        LoadAcquireRcpc { rt, rn, bytes }
        | LoadAcquireRcsc { rt, rn, bytes }
        | LoadExclusive { rt, rn, bytes, .. } => {
            let value = read(state, get_sp(state, rn), u64::from(bytes))?;
            set(state, rt, value);
        }
        StoreRelease { rt, rn, bytes } => {
            let value = get(state, rt);
            write(state, get_sp(state, rn), u64::from(bytes), value)?;
        }
        // With the monitor open (every run starts so) the store fails:
        // status 1 and memory unchanged.
        StoreExclusive { rs, .. } => set(state, rs, 1),
        ReadNzcv { rt } => set(state, rt, u64::from(state.nzcv) << 28),
        WriteNzcv { rt } => state.nzcv = ((get(state, rt) >> 28) & 15) as u8,
        AddSubImmediate {
            rd,
            rn,
            bits,
            immediate,
            subtract,
            flags,
        } => {
            let bits = u32::from(bits);
            let x = get_sp(state, rn);
            let (result, f) = if subtract {
                add_with_carry(x, !immediate, true, bits)
            } else {
                add_with_carry(x, immediate, false, bits)
            };
            if flags {
                state.nzcv = f;
                set(state, rd, result);
            } else {
                set_sp(state, rd, result);
            }
        }
        AddSubShifted {
            rd,
            rn,
            rm,
            bits,
            shift: kind,
            amount,
            subtract,
            flags,
        } => {
            let bits = u32::from(bits);
            let y = shift(get(state, rm), kind, u32::from(amount), bits);
            let x = get(state, rn);
            let (result, f) = if subtract {
                add_with_carry(x, !y, true, bits)
            } else {
                add_with_carry(x, y, false, bits)
            };
            if flags {
                state.nzcv = f;
            }
            set(state, rd, result);
        }
        AddSubExtended {
            rd,
            rn,
            rm,
            bits,
            option,
            amount,
            subtract,
            flags,
        } => {
            let bits = u32::from(bits);
            let y = extend(get(state, rm), option, amount, bits);
            let x = get_sp(state, rn);
            let (result, f) = if subtract {
                add_with_carry(x, !y, true, bits)
            } else {
                add_with_carry(x, y, false, bits)
            };
            if flags {
                state.nzcv = f;
                set(state, rd, result);
            } else {
                set_sp(state, rd, result);
            }
        }
        AddCarry {
            rd,
            rn,
            rm,
            bits,
            subtract,
            flags,
        } => {
            let bits = u32::from(bits);
            let carry = state.nzcv & 2 != 0;
            let y = get(state, rm);
            let y = if subtract { !y } else { y };
            let (result, f) = add_with_carry(get(state, rn), y, carry, bits);
            if flags {
                state.nzcv = f;
            }
            set(state, rd, result);
        }
        LogicalShifted {
            rd,
            rn,
            rm,
            bits,
            op,
            invert,
            shift: kind,
            amount,
        } => {
            let bits = u32::from(bits);
            let mut y = shift(get(state, rm), kind, u32::from(amount), bits);
            if invert {
                y = !y & mask(bits);
            }
            let result = logical(state, op, get(state, rn), y, bits);
            set(state, rd, result);
        }
        LogicalImmediate {
            rd,
            rn,
            bits,
            op,
            mask: immediate,
        } => {
            let bits = u32::from(bits);
            let result = logical(state, op, get(state, rn), immediate, bits);
            if op == LogicalOp::Ands {
                set(state, rd, result);
            } else {
                set_sp(state, rd, result);
            }
        }
        MoveWide {
            rd,
            bits,
            op,
            immediate,
            shift,
        } => {
            let bits = u32::from(bits);
            let value = u64::from(immediate) << shift;
            let result = match op {
                WideOp::Zero => value,
                WideOp::Not => !value & mask(bits),
                WideOp::Keep => (get(state, rd) & !(0xffff << shift) | value) & mask(bits),
            };
            set(state, rd, result);
        }
        Bitfield {
            rd,
            rn,
            bits,
            op,
            rotation,
            sign_bit,
            ..
        } => {
            let bits = u32::from(bits);
            let n = u32::from(bits == 64);
            let (r, s) = (u32::from(rotation), u32::from(sign_bit));
            let (wmask, tmask) = bit_masks(n, s, r, bits);
            let source = get(state, rn) & mask(bits);
            let rotated = shift(source, 3, r, bits);
            let destination = if op == BitfieldOp::Merge {
                get(state, rd) & mask(bits)
            } else {
                0
            };
            let bot = (destination & !wmask) | (rotated & wmask);
            let top = if op == BitfieldOp::Signed {
                if (source >> s) & 1 == 1 {
                    mask(bits)
                } else {
                    0
                }
            } else {
                destination
            };
            let result = ((top & !tmask) | (bot & tmask)) & mask(bits);
            set(state, rd, result);
        }
        Extract {
            rd,
            rn,
            rm,
            bits,
            lsb,
        } => {
            let bits = u32::from(bits);
            let concat = (u128::from(get(state, rn) & mask(bits)) << bits)
                | u128::from(get(state, rm) & mask(bits));
            let result = ((concat >> lsb) as u64) & mask(bits);
            set(state, rd, result);
        }
        Select {
            rd,
            rn,
            rm,
            bits,
            condition,
            operation,
        } => {
            let bits = u32::from(bits);
            let result = if condition_holds(condition, state.nzcv) {
                get(state, rn)
            } else {
                let m = get(state, rm);
                match operation {
                    0 => m,
                    1 => m.wrapping_add(1),
                    2 => !m,
                    _ => (!m).wrapping_add(1),
                }
            };
            set(state, rd, result & mask(bits));
        }
        ConditionalCompare {
            rn,
            bits,
            rhs,
            condition,
            nzcv: fallback,
        }
        | ConditionalCompareNegative {
            rn,
            bits,
            rhs,
            condition,
            nzcv: fallback,
        } => {
            let bits = u32::from(bits);
            state.nzcv = if condition_holds(condition, state.nzcv) {
                let y = match rhs {
                    CompareOperand::Register(rm) => get(state, rm),
                    CompareOperand::Immediate(value) => u64::from(value),
                };
                let negative = matches!(instruction, ConditionalCompareNegative { .. });
                if negative {
                    add_with_carry(get(state, rn), y, false, bits).1
                } else {
                    add_with_carry(get(state, rn), !y, true, bits).1
                }
            } else {
                fallback
            };
        }
        MultiplyAdd {
            rd,
            rn,
            rm,
            ra,
            bits,
            subtract,
        } => {
            let product = get(state, rn).wrapping_mul(get(state, rm));
            let a = get(state, ra);
            let result = if subtract {
                a.wrapping_sub(product)
            } else {
                a.wrapping_add(product)
            };
            set(state, rd, result & mask(u32::from(bits)));
        }
        UnsignedMultiplyAddLong { rd, rn, rm, ra } => {
            let product = (get(state, rn) & 0xffff_ffff) * (get(state, rm) & 0xffff_ffff);
            set(state, rd, get(state, ra).wrapping_add(product));
        }
        MultiplyAddLong {
            rd,
            rn,
            rm,
            ra,
            signed,
            subtract,
        } => {
            let operand = |value: u64| {
                if signed {
                    sign_extend(value & 0xffff_ffff, 32)
                } else {
                    value & 0xffff_ffff
                }
            };
            let product = operand(get(state, rn)).wrapping_mul(operand(get(state, rm)));
            let a = get(state, ra);
            let result = if subtract {
                a.wrapping_sub(product)
            } else {
                a.wrapping_add(product)
            };
            set(state, rd, result);
        }
        MultiplyHigh { rd, rn, rm, signed } => {
            let (n, m) = (get(state, rn), get(state, rm));
            let high = if signed {
                ((i128::from(n as i64) * i128::from(m as i64)) >> 64) as u64
            } else {
                ((u128::from(n) * u128::from(m)) >> 64) as u64
            };
            set(state, rd, high);
        }
        Divide {
            rd,
            rn,
            rm,
            bits,
            signed,
        } => {
            let bits = u32::from(bits);
            let (n, m) = (get(state, rn) & mask(bits), get(state, rm) & mask(bits));
            let result = if m == 0 {
                0
            } else if signed {
                let (n, m) = (sign_extend(n, bits) as i64, sign_extend(m, bits) as i64);
                // Truncating division; MIN / -1 wraps to MIN.
                (i128::from(n) / i128::from(m)) as u64
            } else {
                n / m
            };
            set(state, rd, result & mask(bits));
        }
        VariableShift {
            rd,
            rn,
            rm,
            bits,
            shift: kind,
        } => {
            let bits = u32::from(bits);
            let amount = (get(state, rm) % u64::from(bits)) as u32;
            set(state, rd, shift(get(state, rn), kind, amount, bits));
        }
        CountLeading { rd, rn, bits, sign } => {
            let bits = u32::from(bits);
            let value = get(state, rn) & mask(bits);
            let count = if sign {
                // Leading bits equal to the sign bit, excluding it.
                let x = (value ^ (value >> 1)) & mask(bits - 1);
                leading_zeros(x, bits - 1)
            } else {
                leading_zeros(value, bits)
            };
            set(state, rd, u64::from(count));
        }
        ReverseBits { rd, rn, bits } => {
            let bits = u32::from(bits);
            let value = get(state, rn) & mask(bits);
            set(state, rd, value.reverse_bits() >> (64 - bits));
        }
        ReverseBytes {
            rd,
            rn,
            bits,
            group_bits,
        } => {
            let value = get(state, rn) & mask(u32::from(bits));
            let mut result = 0;
            let mut at = 0;
            while at < u32::from(bits) {
                let group = (value >> at) & mask(u32::from(group_bits));
                let bytes = u32::from(group_bits) / 8;
                let mut reversed = 0;
                for i in 0..bytes {
                    reversed |= ((group >> (8 * i)) & 0xff) << (8 * (bytes - 1 - i));
                }
                result |= reversed << at;
                at += u32::from(group_bits);
            }
            set(state, rd, result);
        }
        LoadStore {
            rt,
            rn,
            bytes,
            op,
            address,
        } => {
            let bytes = u64::from(bytes);
            let base = get_sp(state, rn);
            let (target, written_back) = match address {
                Address::Immediate { offset, writeback } => {
                    let moved = base.wrapping_add(offset as u64);
                    match writeback {
                        Writeback::None => (moved, None),
                        Writeback::Pre => (moved, Some(moved)),
                        Writeback::Post => (base, Some(moved)),
                    }
                }
                Address::Register { rm, option, shift } => {
                    let index = extend(get(state, rm), option, shift, 64);
                    (base.wrapping_add(index), None)
                }
            };
            match op {
                MemoryOp::Store => write(state, target, bytes, get(state, rt))?,
                MemoryOp::Load { .. } => {
                    let value = loaded(read(state, target, bytes)?, bytes, op);
                    set(state, rt, value);
                }
            }
            if let Some(value) = written_back {
                set_sp(state, rn, value);
            }
        }
        LoadStorePair {
            rt,
            rt2,
            rn,
            bytes,
            op,
            offset,
            writeback,
        } => {
            let bytes = u64::from(bytes);
            let base = get_sp(state, rn);
            let moved = base.wrapping_add(offset as u64);
            let target = if writeback == Writeback::Post {
                base
            } else {
                moved
            };
            match op {
                MemoryOp::Store => {
                    let (first, second) = (get(state, rt), get(state, rt2));
                    write(state, target, bytes, first)?;
                    write(state, target + bytes, bytes, second)?;
                }
                MemoryOp::Load { .. } => {
                    let first = loaded(read(state, target, bytes)?, bytes, op);
                    let second = loaded(read(state, target + bytes, bytes)?, bytes, op);
                    set(state, rt, first);
                    set(state, rt2, second);
                }
            }
            if writeback != Writeback::None {
                set_sp(state, rn, moved);
            }
        }
        ref other => panic!("no reference semantics for {other:?}"),
    }
    Some(false)
}

/// LD<op>, ST<op> and SWP; `add` for LDADD (where `op` is unused).
pub fn atomic(
    state: &mut State,
    op: AtomicOp,
    rs: u8,
    rt: u8,
    rn: u8,
    bytes: u8,
    add: bool,
) -> Option<()> {
    let bits = 8 * u32::from(bytes);
    let m = mask(bits);
    let address = get_sp(state, rn);
    let old = read(state, address, u64::from(bytes))?;
    let operand = get(state, rs) & m;
    let signed = |value: u64| sign_extend(value, bits) as i64;
    let new = if add {
        old.wrapping_add(operand)
    } else {
        match op {
            AtomicOp::Clear => old & !operand,
            AtomicOp::ExclusiveOr => old ^ operand,
            AtomicOp::Set => old | operand,
            AtomicOp::SignedMax => {
                if signed(old) >= signed(operand) {
                    old
                } else {
                    operand
                }
            }
            AtomicOp::SignedMin => {
                if signed(old) <= signed(operand) {
                    old
                } else {
                    operand
                }
            }
            AtomicOp::UnsignedMax => old.max(operand),
            AtomicOp::UnsignedMin => old.min(operand),
            AtomicOp::Swap => operand,
        }
    };
    write(state, address, u64::from(bytes), new & m)?;
    set(state, rt, old);
    Some(())
}

pub fn leading_zeros(value: u64, bits: u32) -> u32 {
    if value == 0 {
        bits
    } else {
        value.leading_zeros() - (64 - bits)
    }
}

pub fn logical(state: &mut State, op: LogicalOp, x: u64, y: u64, bits: u32) -> u64 {
    let result = match op {
        LogicalOp::And | LogicalOp::Ands => x & y,
        LogicalOp::Orr => x | y,
        LogicalOp::Eor => x ^ y,
    } & mask(bits);
    if op == LogicalOp::Ands {
        state.nzcv = nzcv((result >> (bits - 1)) & 1 == 1, result == 0, false, false);
    }
    result
}

/// The first difference between two states, described.
pub fn difference(expected: &State, actual: &State) -> String {
    let mut parts = Vec::new();
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
    if expected.nzcv != actual.nzcv {
        parts.push(format!(
            "nzcv: expected {:#06b}, lifted {:#06b}",
            expected.nzcv, actual.nzcv
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
