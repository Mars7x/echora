//! Differential test of the AArch64 decoder against yaxpeax-arm.
//!
//! For every encoding class `aarch64::decode` accepts, random operand fields
//! are decoded by both and the operands compared: opcode, register numbers
//! and widths (including whether register 31 is SP or the zero register),
//! immediates, branch targets and addressing modes. Over random words, every
//! word the decoder accepts must also be valid for yaxpeax.
//!
//! Known, explained differences are listed in `explained`. Hints,
//! barriers, PAuth forms with disabled keys, prefetches (decoded as NOP),
//! cache maintenance and SIMD are checked for acceptance only. Run with
//! `--nocapture` to see the classes checked and their counts.

use echora_lift::aarch64::{
    Address, BitfieldOp, CompareOperand, Instruction, LogicalOp, MemoryOp, WideOp, Writeback,
    decode,
};
use yaxpeax_arch::{Decoder, U8Reader};
use yaxpeax_arm::armv8::a64::{InstDecoder, Operand, ShiftStyle, SizeCode};

const PC: u64 = 0x7100_0000;

fn theirs(word: u32) -> Option<yaxpeax_arm::armv8::a64::Instruction> {
    let bytes = word.to_le_bytes();
    let mut reader = U8Reader::new(&bytes);
    InstDecoder::default().decode(&mut reader).ok()
}

/// An operand in a form both decoders can be put into.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Op {
    /// A general register with 31 as the zero register; `true` for X.
    Zr(bool, u8),
    /// A general register with 31 as SP.
    Sp(bool, u8),
    Imm(u64),
    Target(u64),
    Cond(u8),
    /// Base, offset and writeback (`None`, pre, post).
    Mem(u8, i64, Writeback),
    /// Base, index, index width X, extend option, shift amount.
    MemReg(u8, u8, bool, u8, u8),
    /// Register (zero-register form), width X, shift kind, amount.
    Shifted(u8, bool, u8, u8),
    /// Register, width X, extend option (`UXTB`..`SXTX` as 0..7), amount.
    Extended(u8, bool, u8, u8),
    Sys(u16),
    /// A SIMD register used whole: size letter and number.
    Vector(String, u8),
    Other(String),
}

fn shift_code(style: ShiftStyle) -> Option<u8> {
    Some(match style {
        ShiftStyle::LSL => 0,
        ShiftStyle::LSR => 1,
        ShiftStyle::ASR => 2,
        ShiftStyle::ROR => 3,
        _ => return None,
    })
}

fn extend_code(style: ShiftStyle) -> Option<u8> {
    Some(match style {
        ShiftStyle::UXTB => 0,
        ShiftStyle::UXTH => 1,
        ShiftStyle::UXTW => 2,
        ShiftStyle::UXTX => 3,
        ShiftStyle::SXTB => 4,
        ShiftStyle::SXTH => 5,
        ShiftStyle::SXTW => 6,
        ShiftStyle::SXTX => 7,
        _ => return None,
    })
}

fn normalize(operand: &Operand) -> Option<Op> {
    Some(match *operand {
        Operand::Nothing => return None,
        Operand::Register(size, n) => Op::Zr(size == SizeCode::X, n as u8),
        Operand::RegisterOrSP(size, n) => Op::Sp(size == SizeCode::X, n as u8),
        Operand::Immediate(value) => Op::Imm(u64::from(value)),
        Operand::Imm64(value) => Op::Imm(value),
        Operand::Imm16(value) => Op::Imm(u64::from(value)),
        Operand::ImmShift(value, shift) => Op::Imm(u64::from(value) << shift),
        Operand::PCOffset(offset) => Op::Target(PC.wrapping_add(offset as u64)),
        Operand::ConditionCode(condition) => Op::Cond(condition),
        Operand::RegPreIndex(base, offset, writeback) => Op::Mem(
            base as u8,
            i64::from(offset),
            if writeback {
                Writeback::Pre
            } else {
                Writeback::None
            },
        ),
        Operand::RegPostIndex(base, offset) => {
            Op::Mem(base as u8, i64::from(offset), Writeback::Post)
        }
        Operand::RegRegOffset(base, index, size, style, amount) => {
            let option = match style {
                ShiftStyle::LSL => 3,
                other => extend_code(other)?,
            };
            Op::MemReg(base as u8, index as u8, size == SizeCode::X, option, amount)
        }
        Operand::RegShift(style, amount, size, n) => match shift_code(style) {
            Some(kind) => Op::Shifted(n as u8, size == SizeCode::X, kind, amount),
            None => Op::Extended(n as u8, size == SizeCode::X, extend_code(style)?, amount),
        },
        Operand::SystemReg(n) => Op::Sys(n),
        Operand::SIMDRegister(size, n) => Op::Vector(format!("{size:?}"), n as u8),
        ref other => Op::Other(format!("{other:?}")),
    })
}

fn x(bits: u8) -> bool {
    bits == 64
}

fn target(address: u64) -> Op {
    Op::Target(address)
}

fn condition_name(condition: u8) -> String {
    format!("Bcc({condition})")
}

/// The opcode yaxpeax should report and the operands, from our decoding.
/// `None` for variants checked only for acceptance.
fn expected(instruction: &Instruction) -> Option<(String, Vec<Op>)> {
    use Instruction::*;
    use Op::*;
    let name = |s: &str| s.to_owned();
    Some(match *instruction {
        Branch { target: t, link } => (name(if link { "BL" } else { "B" }), vec![target(t)]),
        BranchCond {
            target: t,
            condition,
        } => (condition_name(condition), vec![target(t)]),
        CompareBranch {
            rt,
            bits,
            nonzero,
            target: t,
        } => (
            name(if nonzero { "CBNZ" } else { "CBZ" }),
            vec![Zr(x(bits), rt), target(t)],
        ),
        TestBranch {
            rt,
            bit,
            nonzero,
            target: t,
        } => (
            name(if nonzero { "TBNZ" } else { "TBZ" }),
            vec![Zr(bit >= 32, rt), Imm(u64::from(bit)), target(t)],
        ),
        BranchReg { rn, link } => {
            let word_name = if link { "BLR" } else { "BR" };
            (name(word_name), vec![Zr(true, rn)])
        }
        Svc { number } => (name("SVC"), vec![Imm(u64::from(number))]),
        Adr { rd, address } => (name("ADR"), vec![Zr(true, rd), target(address)]),
        MoveWide {
            rd,
            bits,
            op,
            immediate,
            shift,
        } => (
            name(match op {
                WideOp::Not => "MOVN",
                WideOp::Zero => "MOVZ",
                WideOp::Keep => "MOVK",
            }),
            vec![Zr(x(bits), rd), Imm(u64::from(immediate) << shift)],
        ),
        AddSubImmediate {
            rd,
            rn,
            bits,
            immediate,
            subtract,
            flags,
        } => (
            add_sub_name(subtract, flags),
            vec![
                if flags {
                    Zr(x(bits), rd)
                } else {
                    Sp(x(bits), rd)
                },
                Sp(x(bits), rn),
                Imm(immediate),
            ],
        ),
        AddSubShifted {
            rd,
            rn,
            rm,
            bits,
            shift,
            amount,
            subtract,
            flags,
        } => (
            add_sub_name(subtract, flags),
            vec![
                Zr(x(bits), rd),
                Zr(x(bits), rn),
                Shifted(rm, x(bits), shift, amount),
            ],
        ),
        AddSubExtended {
            rd,
            rn,
            rm,
            bits,
            option,
            amount,
            subtract,
            flags,
        } => (
            add_sub_name(subtract, flags),
            vec![
                if flags {
                    Zr(x(bits), rd)
                } else {
                    Sp(x(bits), rd)
                },
                Sp(x(bits), rn),
                Extended(rm, x(bits) && option & 3 == 3, option, amount),
            ],
        ),
        AddCarry {
            rd,
            rn,
            rm,
            bits,
            subtract,
            flags,
        } => (
            format!(
                "{}{}",
                if subtract { "SBC" } else { "ADC" },
                if flags { "S" } else { "" }
            ),
            vec![Zr(x(bits), rd), Zr(x(bits), rn), Zr(x(bits), rm)],
        ),
        LogicalShifted {
            rd,
            rn,
            rm,
            bits,
            op,
            invert,
            shift,
            amount,
        } => (
            name(match (op, invert) {
                (LogicalOp::And, false) => "AND",
                (LogicalOp::And, true) => "BIC",
                (LogicalOp::Orr, false) => "ORR",
                (LogicalOp::Orr, true) => "ORN",
                (LogicalOp::Eor, false) => "EOR",
                (LogicalOp::Eor, true) => "EON",
                (LogicalOp::Ands, false) => "ANDS",
                (LogicalOp::Ands, true) => "BICS",
            }),
            vec![
                Zr(x(bits), rd),
                Zr(x(bits), rn),
                Shifted(rm, x(bits), shift, amount),
            ],
        ),
        LogicalImmediate {
            rd,
            rn,
            bits,
            op,
            mask,
        } => (
            name(match op {
                LogicalOp::And => "AND",
                LogicalOp::Orr => "ORR",
                LogicalOp::Eor => "EOR",
                LogicalOp::Ands => "ANDS",
            }),
            vec![
                if op == LogicalOp::Ands {
                    Zr(x(bits), rd)
                } else {
                    Sp(x(bits), rd)
                },
                Zr(x(bits), rn),
                Imm(mask),
            ],
        ),
        Bitfield {
            rd,
            rn,
            bits,
            op,
            rotation,
            sign_bit,
            ..
        } => (
            name(match op {
                BitfieldOp::Signed => "SBFM",
                BitfieldOp::Merge => "BFM",
                BitfieldOp::Unsigned => "UBFM",
            }),
            vec![
                Zr(x(bits), rd),
                Zr(x(bits), rn),
                Imm(u64::from(rotation)),
                Imm(u64::from(sign_bit)),
            ],
        ),
        Extract {
            rd,
            rn,
            rm,
            bits,
            lsb,
        } => (
            name("EXTR"),
            vec![
                Zr(x(bits), rd),
                Zr(x(bits), rn),
                Zr(x(bits), rm),
                Imm(u64::from(lsb)),
            ],
        ),
        Select {
            rd,
            rn,
            rm,
            bits,
            condition,
            operation,
        } => (
            name(["CSEL", "CSINC", "CSINV", "CSNEG"][operation as usize]),
            vec![
                Zr(x(bits), rd),
                Zr(x(bits), rn),
                Zr(x(bits), rm),
                Cond(condition),
            ],
        ),
        ConditionalCompare {
            rn,
            bits,
            rhs,
            condition,
            nzcv,
        }
        | ConditionalCompareNegative {
            rn,
            bits,
            rhs,
            condition,
            nzcv,
        } => (
            name(if matches!(instruction, ConditionalCompare { .. }) {
                "CCMP"
            } else {
                "CCMN"
            }),
            vec![
                Zr(x(bits), rn),
                match rhs {
                    CompareOperand::Register(rm) => Zr(x(bits), rm),
                    CompareOperand::Immediate(value) => Imm(u64::from(value)),
                },
                Imm(u64::from(nzcv)),
                Cond(condition),
            ],
        ),
        MultiplyAdd {
            rd,
            rn,
            rm,
            ra,
            bits,
            subtract,
        } => (
            name(if subtract { "MSUB" } else { "MADD" }),
            vec![
                Zr(x(bits), rd),
                Zr(x(bits), rn),
                Zr(x(bits), rm),
                Zr(x(bits), ra),
            ],
        ),
        UnsignedMultiplyAddLong { rd, rn, rm, ra } => (
            name("UMADDL"),
            vec![Zr(true, rd), Zr(false, rn), Zr(false, rm), Zr(true, ra)],
        ),
        MultiplyAddLong {
            rd,
            rn,
            rm,
            ra,
            signed,
            subtract,
        } => (
            format!(
                "{}M{}L",
                if signed { "S" } else { "U" },
                if subtract { "SUB" } else { "ADD" }
            ),
            vec![Zr(true, rd), Zr(false, rn), Zr(false, rm), Zr(true, ra)],
        ),
        MultiplyHigh { rd, rn, rm, signed } => (
            name(if signed { "SMULH" } else { "UMULH" }),
            vec![Zr(true, rd), Zr(true, rn), Zr(true, rm)],
        ),
        Divide {
            rd,
            rn,
            rm,
            bits,
            signed,
        } => (
            name(if signed { "SDIV" } else { "UDIV" }),
            vec![Zr(x(bits), rd), Zr(x(bits), rn), Zr(x(bits), rm)],
        ),
        VariableShift {
            rd,
            rn,
            rm,
            bits,
            shift,
        } => (
            name(["LSLV", "LSRV", "ASRV", "RORV"][shift as usize]),
            vec![Zr(x(bits), rd), Zr(x(bits), rn), Zr(x(bits), rm)],
        ),
        CountLeading { rd, rn, bits, sign } => (
            name(if sign { "CLS" } else { "CLZ" }),
            vec![Zr(x(bits), rd), Zr(x(bits), rn)],
        ),
        ReverseBits { rd, rn, bits } => (name("RBIT"), vec![Zr(x(bits), rd), Zr(x(bits), rn)]),
        ReverseBytes {
            rd,
            rn,
            bits,
            group_bits,
        } => (
            name(match (bits, group_bits) {
                (_, 16) => "REV16",
                (64, 32) => "REV32",
                _ => "REV",
            }),
            vec![Zr(x(bits), rd), Zr(x(bits), rn)],
        ),
        Crc32 {
            rd,
            rn,
            rm,
            bytes,
            castagnoli,
        } => (
            format!(
                "CRC32{}{}",
                if castagnoli { "C" } else { "" },
                match bytes {
                    1 => "B",
                    2 => "H",
                    4 => "W",
                    _ => "X",
                }
            ),
            vec![Zr(false, rd), Zr(false, rn), Zr(bytes == 8, rm)],
        ),
        CompareAndSwap {
            rs,
            rt,
            rn,
            bytes,
            acquire,
            release,
        } => (
            format!("CAS{}({})", size_suffix(bytes), ordering(acquire, release)),
            vec![
                Zr(bytes == 8, rs),
                Zr(bytes == 8, rt),
                Mem(rn, 0, Writeback::None),
            ],
        ),
        CompareAndSwapPair {
            rs,
            rt,
            rn,
            bytes,
            acquire,
            release,
        } => (
            format!("CASP({})", ordering(acquire, release)),
            vec![
                Other(format!("RegisterPair({:?}, {rs})", size_code(bytes))),
                Other(format!("RegisterPair({:?}, {rt})", size_code(bytes))),
                Mem(rn, 0, Writeback::None),
            ],
        ),
        AtomicAdd {
            rs,
            rt,
            rn,
            bytes,
            acquire,
            release,
        } => (
            format!(
                "LDADD{}({})",
                size_suffix(bytes),
                ordering(acquire, release)
            ),
            vec![
                Zr(bytes == 8, rs),
                Zr(bytes == 8, rt),
                Mem(rn, 0, Writeback::None),
            ],
        ),
        AtomicMemory {
            op,
            rs,
            rt,
            rn,
            bytes,
            acquire,
            release,
        } => (
            format!(
                "{}{}({})",
                match format!("{op:?}").as_str() {
                    "Clear" => "LDCLR",
                    "ExclusiveOr" => "LDEOR",
                    "Set" => "LDSET",
                    "SignedMax" => "LDSMAX",
                    "SignedMin" => "LDSMIN",
                    "UnsignedMax" => "LDUMAX",
                    "UnsignedMin" => "LDUMIN",
                    _ => "SWP",
                },
                size_suffix(bytes),
                ordering(acquire, release)
            ),
            vec![
                Zr(bytes == 8, rs),
                Zr(bytes == 8, rt),
                Mem(rn, 0, Writeback::None),
            ],
        ),
        LoadAcquireRcpc { rt, rn, bytes } => (
            format!("LDAPR{}", size_suffix(bytes)),
            vec![Zr(bytes == 8, rt), Mem(rn, 0, Writeback::None)],
        ),
        LoadAcquireRcsc { rt, rn, bytes } => (
            format!("LDAR{}", size_suffix(bytes)),
            vec![Zr(bytes == 8, rt), Mem(rn, 0, Writeback::None)],
        ),
        StoreRelease { rt, rn, bytes } => (
            format!("STLR{}", size_suffix(bytes)),
            vec![Zr(bytes == 8, rt), Mem(rn, 0, Writeback::None)],
        ),
        LoadExclusive {
            rt,
            rn,
            bytes,
            acquire,
        } => (
            format!(
                "LD{}XR{}",
                if acquire { "A" } else { "" },
                size_suffix(bytes)
            ),
            vec![Zr(bytes == 8, rt), Mem(rn, 0, Writeback::None)],
        ),
        StoreExclusive {
            rs,
            rt,
            rn,
            bytes,
            release,
        } => (
            format!(
                "ST{}XR{}",
                if release { "L" } else { "" },
                size_suffix(bytes)
            ),
            vec![
                Zr(false, rs),
                Zr(bytes == 8, rt),
                Mem(rn, 0, Writeback::None),
            ],
        ),
        LoadStore {
            rt,
            rn,
            bytes,
            op,
            address,
        } => {
            let unscaled = matches!(
                address,
                Address::Immediate {
                    writeback: Writeback::None,
                    offset,
                } if offset < 0 || offset % i64::from(bytes) != 0
            );
            (
                load_store_name(op, bytes, unscaled),
                vec![
                    Zr(register_is_x(op, bytes), rt),
                    match address {
                        Address::Immediate { offset, writeback } => Mem(rn, offset, writeback),
                        Address::Register { rm, option, shift } => {
                            MemReg(rn, rm, option & 1 == 1, option, shift)
                        }
                    },
                ],
            )
        }
        LoadStorePair {
            rt,
            rt2,
            rn,
            bytes,
            op,
            offset,
            writeback,
        } => (
            name(match op {
                MemoryOp::Store => "STP",
                MemoryOp::Load { signed: true, .. } => "LDPSW",
                MemoryOp::Load { .. } => "LDP",
            }),
            vec![
                Zr(register_is_x(op, bytes), rt),
                Zr(register_is_x(op, bytes), rt2),
                Mem(rn, offset, writeback),
            ],
        ),
        LoadLiteral {
            rt,
            signed,
            bits,
            address,
            ..
        } => (
            name(if signed { "LDRSW" } else { "LDR" }),
            vec![Zr(x(bits), rt), target(address)],
        ),
        LoadStoreVectorPair {
            rt,
            rt2,
            rn,
            load,
            offset,
            writeback,
        } => (
            name(if load { "LDP" } else { "STP" }),
            vec![
                Vector("Q".into(), rt),
                Vector("Q".into(), rt2),
                Mem(rn, offset, writeback),
            ],
        ),
        ReadPhysicalCounter { rt }
        | ReadThreadPointer { rt, .. }
        | ReadFpControl { rt, .. }
        | ReadCacheType { rt }
        | ReadNzcv { rt } => (
            name("MRS"),
            vec![Zr(true, rt), system_register(instruction)?],
        ),
        WriteThreadPointer { rt } | WriteFpControl { rt, .. } | WriteNzcv { rt } => (
            name("MSR"),
            vec![system_register(instruction)?, Zr(true, rt)],
        ),
        StripPac { rd } => (name("XPAC"), vec![Zr(true, rd)]),
        PacGeneric { rd, rn, rm } => (
            name("PACGA"),
            vec![Zr(true, rd), Zr(true, rn), Sp(true, rm)],
        ),
        Undefined => (name("UDF"), vec![Other(String::new())]),
        // Checked for acceptance only: hints, barriers, PAuth forms with
        // disabled keys, cache maintenance and SIMD.
        _ => return None,
    })
}

/// The system register field (op0:op1:CRn:CRm:op2, bits 19:5) of the
/// encoding each MRS/MSR variant stands for.
fn system_register(instruction: &Instruction) -> Option<Op> {
    use Instruction::*;
    let word: u32 = match *instruction {
        ReadPhysicalCounter { .. } => 0xd53b_e020,
        ReadNzcv { .. } => 0xd53b_4200,
        WriteNzcv { .. } => 0xd51b_4200,
        ReadThreadPointer {
            read_only: false, ..
        } => 0xd53b_d040,
        ReadThreadPointer {
            read_only: true, ..
        } => 0xd53b_d060,
        WriteThreadPointer { .. } => 0xd51b_d040,
        ReadFpControl { status: false, .. } => 0xd53b_4400,
        ReadFpControl { status: true, .. } => 0xd53b_4420,
        WriteFpControl { status: false, .. } => 0xd51b_4400,
        WriteFpControl { status: true, .. } => 0xd51b_4420,
        ReadCacheType { .. } => 0xd53b_0020,
        _ => return None,
    };
    Some(Op::Sys(((word >> 5) & 0x7fff) as u16))
}

fn add_sub_name(subtract: bool, flags: bool) -> String {
    format!(
        "{}{}",
        if subtract { "SUB" } else { "ADD" },
        if flags { "S" } else { "" }
    )
}

fn size_suffix(bytes: u8) -> &'static str {
    match bytes {
        1 => "B",
        2 => "H",
        _ => "",
    }
}

fn size_code(bytes: u8) -> SizeCode {
    if bytes == 8 { SizeCode::X } else { SizeCode::W }
}

/// yaxpeax's ordering field: acquire in bit 1, release in bit 0.
fn ordering(acquire: bool, release: bool) -> u8 {
    (u8::from(acquire) << 1) | u8::from(release)
}

fn register_is_x(op: MemoryOp, bytes: u8) -> bool {
    match op {
        MemoryOp::Store => bytes == 8,
        MemoryOp::Load { bits, .. } => bits == 64,
    }
}

fn load_store_name(op: MemoryOp, bytes: u8, unscaled: bool) -> String {
    let (prefix, signed) = match op {
        MemoryOp::Store => ("ST", false),
        MemoryOp::Load { signed, .. } => ("LD", signed),
    };
    let size = match (bytes, signed) {
        (1, _) => "B",
        (2, _) => "H",
        (4, true) => "W",
        _ => "",
    };
    format!(
        "{prefix}{}{}{size}",
        if unscaled { "UR" } else { "R" },
        if signed { "S" } else { "" }
    )
}

/// The opcode name and operands yaxpeax reports, normalized.
fn reported(instruction: &yaxpeax_arm::armv8::a64::Instruction) -> (String, Vec<Op>) {
    (
        format!("{:?}", instruction.opcode),
        instruction.operands.iter().filter_map(normalize).collect(),
    )
}

/// An extend that leaves the register unchanged (UXTW of a W register,
/// UXTX of an X register) is LSL; yaxpeax always reports it as LSL.
fn identity_extend(op: &Op) -> Op {
    match *op {
        Op::Extended(rm, true, 3, amount) => Op::Shifted(rm, true, 0, amount),
        Op::Extended(rm, false, 2, amount) => Op::Shifted(rm, false, 0, amount),
        ref other => other.clone(),
    }
}

/// Placeholder operands (`Other("")`) match anything of their position.
fn matches(expected: &(String, Vec<Op>), actual: &(String, Vec<Op>)) -> bool {
    let name_ok = if expected.0 == "XPAC" {
        actual.0 == "XPACI" || actual.0 == "XPACD"
    } else {
        expected.0 == actual.0
    };
    name_ok
        && expected.1.len() == actual.1.len()
        && expected
            .1
            .iter()
            .zip(&actual.1)
            .all(|(e, a)| e == a || identity_extend(e) == *a || *e == Op::Other(String::new()))
}

/// A deterministic xorshift generator.
struct Random(u64);

impl Random {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }
}

/// An encoding class: fixed bits (`mask`, `value`); the rest are random.
struct Class {
    name: &'static str,
    mask: u32,
    value: u32,
}

const fn class(name: &'static str, mask: u32, value: u32) -> Class {
    Class { name, mask, value }
}

/// Every class `decode` accepts, by the fixed bits it matches.
const CLASSES: &[Class] = &[
    class("B/BL", 0x7c00_0000, 0x1400_0000),
    class("B.cond", 0xff00_0010, 0x5400_0000),
    class("CBZ/CBNZ", 0x7e00_0000, 0x3400_0000),
    class("TBZ/TBNZ", 0x7e00_0000, 0x3600_0000),
    class("BR/BLR/RET", 0xff9f_fc1f, 0xd61f_0000),
    class("SVC", 0xffe0_001f, 0xd400_0001),
    class("ADR/ADRP", 0x1f00_0000, 0x1000_0000),
    class("MOVN/MOVZ/MOVK", 0x1f80_0000, 0x1280_0000),
    class("ADD/SUB (immediate)", 0x1f80_0000, 0x1100_0000),
    class("ADD/SUB (shifted register)", 0x1f20_0000, 0x0b00_0000),
    class("ADD/SUB (extended register)", 0x1fe0_0000, 0x0b20_0000),
    class("ADC/SBC", 0x1fe0_fc00, 0x1a00_0000),
    class("logical (shifted register)", 0x1f00_0000, 0x0a00_0000),
    class("logical (immediate)", 0x1f80_0000, 0x1200_0000),
    class("bitfield", 0x1f80_0000, 0x1300_0000),
    class("EXTR", 0x7fa0_0000, 0x1380_0000),
    class("conditional compare", 0x3fe0_0410, 0x3a40_0000),
    class("conditional select", 0x3fe0_0800, 0x1a80_0000),
    class("MADD/MSUB", 0x7fe0_0000, 0x1b00_0000),
    class("long multiply-add", 0xff60_0000, 0x9b20_0000),
    class("SMULH/UMULH", 0xff60_fc00, 0x9b40_7c00),
    class("data processing (2 source)", 0x7fe0_c000, 0x1ac0_0000),
    class("CRC32", 0x7fe0_e000, 0x1ac0_4000),
    class("PACGA", 0xffe0_fc00, 0x9ac0_3000),
    class("data processing (1 source)", 0x7fff_f000, 0x5ac0_0000),
    class("PAC/AUT/XPAC", 0xffff_c000, 0xdac1_0000),
    class("CAS", 0x3fa0_7c00, 0x08a0_7c00),
    class("CASP", 0xbfa0_7c00, 0x0820_7c00),
    class("LSE atomics", 0x3f20_0c00, 0x3820_0000),
    class("load/store exclusive", 0x3f80_7c00, 0x0800_7c00),
    class("LDAPR", 0x3fff_fc00, 0x38bf_c000),
    class("LDAR/STLR", 0x3f7f_fc00, 0x089f_fc00),
    class("load/store (unsigned offset)", 0x3f00_0000, 0x3900_0000),
    class("load/store (unscaled, pre, post)", 0x3f20_0000, 0x3800_0000),
    class("load/store (register offset)", 0x3f20_0c00, 0x3820_0800),
    class("load/store pair", 0x3e00_0000, 0x2800_0000),
    class("load/store pair (Q)", 0xfe00_0000, 0xac00_0000),
    class("load literal", 0x3f00_0000, 0x1800_0000),
    class("MRS CNTPCT_EL0", 0xffff_ffe0, 0xd53b_e020),
    class("MRS NZCV", 0xffff_ffe0, 0xd53b_4200),
    class("MSR NZCV", 0xffff_ffe0, 0xd51b_4200),
    class("MRS TPIDR_EL0", 0xffff_ffe0, 0xd53b_d040),
    class("MRS TPIDRRO_EL0", 0xffff_ffe0, 0xd53b_d060),
    class("MSR TPIDR_EL0", 0xffff_ffe0, 0xd51b_d040),
    class("MRS FPCR", 0xffff_ffe0, 0xd53b_4400),
    class("MRS FPSR", 0xffff_ffe0, 0xd53b_4420),
    class("MSR FPCR", 0xffff_ffe0, 0xd51b_4400),
    class("MSR FPSR", 0xffff_ffe0, 0xd51b_4420),
    class("MRS CTR_EL0", 0xffff_ffe0, 0xd53b_0020),
    class("DC CVAC", 0xffff_ffe0, 0xd50b_7a20),
    class("DC CVAU", 0xffff_ffe0, 0xd50b_7b20),
    class("DC CIVAC", 0xffff_ffe0, 0xd50b_7e20),
    class("IC IVAU", 0xffff_ffe0, 0xd50b_7520),
    class("RETAA/RETAB", 0xffff_fbff, 0xd65f_0bff),
    class("hints and barriers", 0xffff_f01f, 0xd503_201f),
    class("barriers", 0xffff_f01f, 0xd503_301f),
    class("UDF", 0xffff_0000, 0x0000_0000),
];

/// Opcode names that differ from ours only in a hint the decoder does not
/// keep, with the reason. The operands must still match.
const SAME_OPERATION: &[(&str, &str, &str)] = &[
    (
        "BR",
        "RET",
        "RET Xn branches to Xn like BR Xn; only the return-prediction hint differs",
    ),
    (
        "ADR",
        "ADRP",
        "ADRP is decoded into Adr with the page address already resolved",
    ),
    ("STP", "STNP", "STNP is STP with a non-temporal cache hint"),
    ("LDP", "LDNP", "LDNP is LDP with a non-temporal cache hint"),
];

/// Why a disagreement is not a decoder bug, if it is a known one.
fn explained(expected: &(String, Vec<Op>), actual: &(String, Vec<Op>)) -> Option<&'static str> {
    let same_operands = || matches(&(actual.0.clone(), expected.1.clone()), actual);
    // LDUR/STUR with an offset the scaled form could also encode: both
    // decode to the same access, and the decoder does not keep which.
    let unscaled = expected
        .0
        .strip_prefix("LDR")
        .map(|rest| format!("LDUR{rest}"))
        .or_else(|| {
            expected
                .0
                .strip_prefix("STR")
                .map(|rest| format!("STUR{rest}"))
        });
    if unscaled.as_deref() == Some(actual.0.as_str()) && same_operands() {
        return Some("LDUR/STUR with an offset LDR/STR could encode is the same access");
    }
    // yaxpeax-arm 0.4 takes MSR's Rt from bits 3:0 only (`word & 0b1111`
    // in its system-instruction decoding); llvm-objdump reads all five bits,
    // as the decoder does (`msr NZCV, x16` is 0xd51b4210).
    if let (Some(Op::Zr(true, ours)), Some(Op::Zr(true, theirs))) =
        (expected.1.get(1), actual.1.get(1))
        && expected.0 == "MSR"
        && actual.0 == "MSR"
        && expected.1[0] == actual.1[0]
        && *ours >= 16
        && ours & 15 == *theirs
    {
        return Some("yaxpeax-arm 0.4 decodes MSR's Rt from four bits (bug)");
    }
    SAME_OPERATION
        .iter()
        .find(|(ours, theirs, _)| expected.0 == *ours && actual.0 == *theirs)
        .filter(|_| same_operands())
        .map(|(_, _, reason)| *reason)
}

#[test]
fn decoder_agrees_with_yaxpeax_by_class() {
    let mut random = Random(0x9e37_79b9_7f4a_7c15);
    let mut failures = Vec::new();
    let mut reasons = std::collections::BTreeSet::new();
    for class in CLASSES {
        let (mut accepted, mut compared, mut explained_count) = (0, 0, 0);
        for _ in 0..20_000 {
            let word = (random.next() & !class.mask) | class.value;
            let ours = decode(word, PC);
            if matches!(ours, Instruction::Unsupported) {
                continue;
            }
            accepted += 1;
            let Some(theirs) = theirs(word) else {
                failures.push(format!(
                    "{}: {word:#010x} accepted as {ours:?}, invalid for yaxpeax",
                    class.name
                ));
                continue;
            };
            let Some(expected) = expected(&ours) else {
                continue;
            };
            compared += 1;
            let actual = reported(&theirs);
            if !matches(&expected, &actual) {
                if let Some(reason) = explained(&expected, &actual) {
                    explained_count += 1;
                    reasons.insert(reason);
                } else {
                    failures.push(format!(
                        "{}: {word:#010x} ours {ours:?} expects {expected:?}, yaxpeax {actual:?}",
                        class.name
                    ));
                }
            }
        }
        println!(
            "{:<36} accepted {accepted:>6} compared {compared:>6} explained {explained_count:>5}",
            class.name
        );
        assert!(accepted > 0, "{}: no accepted encodings", class.name);
    }
    for reason in &reasons {
        println!("explained: {reason}");
    }
    failures.sort();
    failures.dedup();
    // A few of each class.
    let mut shown: Vec<&String> = Vec::new();
    for class in CLASSES {
        let prefix = format!("{}: ", class.name);
        shown.extend(failures.iter().filter(|f| f.starts_with(&prefix)).take(4));
    }
    assert!(
        failures.is_empty(),
        "{} disagreements:\n{}",
        failures.len(),
        shown
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn every_accepted_random_word_is_valid() {
    let mut random = Random(0x2545_f491_4f6c_dd1d);
    let mut failures = Vec::new();
    let (mut accepted, mut simd) = (0, 0);
    for _ in 0..1_000_000 {
        let word = random.next();
        let ours = decode(word, PC);
        if matches!(ours, Instruction::Unsupported) {
            continue;
        }
        accepted += 1;
        if matches!(ours, Instruction::Simd(_)) {
            simd += 1;
        }
        if theirs(word).is_none() {
            failures.push(format!("{word:#010x} accepted as {ours:?}"));
        }
    }
    println!("random words accepted {accepted} (SIMD {simd})");
    assert!(
        failures.is_empty(),
        "{} accepted words invalid for yaxpeax:\n{}",
        failures.len(),
        failures
            .iter()
            .take(60)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}
