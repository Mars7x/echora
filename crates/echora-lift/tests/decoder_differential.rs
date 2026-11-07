//! Differential test of the AArch64 decoder against yaxpeax-arm.
//!
//! For every encoding class `aarch64::decode` accepts, random operand fields
//! are decoded by both and the operands compared: opcode, register numbers
//! and widths (including whether register 31 is SP or the zero register),
//! immediates, branch targets and addressing modes. Over random words, every
//! word the decoder accepts must also be valid for yaxpeax.
//!
//! SIMD and floating-point words (random fields, biased to that encoding
//! space) are compared the same way per `SimdOp` variant: registers with
//! their scalar size, vector arrangement, lane or register group, general
//! registers, addressing, immediates and, where our operation maps to one
//! mnemonic, the opcode. The immediates of MOVI/MVNI/ORR/BIC (vector) and
//! the SHA forms compare register numbers only.
//!
//! Known, explained differences are listed in `explained` and
//! `simd_same_operation`. Hints, barriers, PAuth forms with disabled keys,
//! prefetches (decoded as NOP) and cache maintenance are checked for
//! acceptance only. Run with `--nocapture` to see the classes checked and
//! their counts.

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

// SIMD and floating point.

use echora_lift::simd::{
    FpBinaryOp, FpPrecision, FpSize, FpTernaryOp, FpUnaryOp, Rounding, SimdOp, VecAcrossOp,
    VecFpOp, VecFpUnaryOp, VecIntOp, VecLongOp, VecShiftOp, VecUnaryOp,
};
use yaxpeax_arm::armv8::a64::SIMDSizeCode;

/// A SIMD operand in a form both decoders can be put into.
#[derive(Clone, Debug, PartialEq, Eq)]
enum V {
    /// A whole register as a scalar: size in bits, number.
    Scalar(u32, u8),
    /// A vector: number, Q (128-bit), element bits.
    Vector(u8, bool, u32),
    /// One lane: number, element bits, index.
    Lane(u8, u32, u8),
    /// Consecutive registers: first, Q, element bits, count.
    Group(u8, bool, u32, u8),
    /// One lane of consecutive registers: first, element bits, count, index.
    GroupLane(u8, u32, u8, u8),
    /// A general operand, as in the scalar comparison.
    General(Op),
    /// A floating-point immediate, as bits of an f64.
    Float(u64),
}

fn simd_bits(size: SIMDSizeCode) -> u32 {
    match size {
        SIMDSizeCode::B => 8,
        SIMDSizeCode::H => 16,
        SIMDSizeCode::S => 32,
        SIMDSizeCode::D => 64,
        SIMDSizeCode::Q => 128,
    }
}

fn normalize_simd(operand: &Operand) -> Option<V> {
    Some(match *operand {
        Operand::SIMDRegister(size, n) => V::Scalar(simd_bits(size), n as u8),
        Operand::SIMDRegisterElements(vector, n, element) => {
            V::Vector(n as u8, vector == SIMDSizeCode::Q, simd_bits(element))
        }
        Operand::SIMDRegisterElementsLane(_, n, element, lane) => {
            V::Lane(n as u8, simd_bits(element), lane)
        }
        Operand::SIMDRegisterGroup(vector, n, element, count) => V::Group(
            n as u8,
            vector == SIMDSizeCode::Q,
            simd_bits(element),
            count,
        ),
        Operand::SIMDRegisterGroupLane(n, element, count, lane) => {
            V::GroupLane(n as u8, simd_bits(element), count, lane)
        }
        Operand::ImmediateDouble(value) => V::Float(value.to_bits()),
        ref other => V::General(normalize(other)?),
    })
}

fn fp_bits(size: FpSize) -> u32 {
    match size {
        FpSize::S => 32,
        FpSize::D => 64,
    }
}

fn precision_bits(precision: FpPrecision) -> u32 {
    match precision {
        FpPrecision::H => 16,
        FpPrecision::S => 32,
        FpPrecision::D => 64,
    }
}

fn rounding_letter(rounding: Rounding) -> &'static str {
    match rounding {
        Rounding::Nearest => "N",
        Rounding::Plus => "P",
        Rounding::Minus => "M",
        Rounding::Zero => "Z",
        Rounding::Away => "A",
    }
}

fn vec_fp_name(op: VecFpOp) -> String {
    match op {
        VecFpOp::MaxNum => "FMAXNM".into(),
        VecFpOp::MinNum => "FMINNM".into(),
        VecFpOp::AbsDiff => "FABD".into(),
        VecFpOp::MulExtended => "FMULX".into(),
        VecFpOp::MulAdd => "FMLA".into(),
        VecFpOp::MulSub => "FMLS".into(),
        VecFpOp::RecipStep => "FRECPS".into(),
        VecFpOp::RSqrtStep => "FRSQRTS".into(),
        other => format!("F{}", format!("{other:?}").to_uppercase()),
    }
}

fn vec_int_name(op: VecIntOp) -> String {
    format!("{op:?}").to_uppercase()
}

/// Operands of a SIMD register of `bits` used as a scalar or, unless
/// `scalar`, as a vector of `element`-bit lanes.
fn reg(n: u8, scalar: bool, q: bool, element: u32) -> V {
    if scalar {
        V::Scalar(element, n)
    } else {
        V::Vector(n, q, element)
    }
}

/// The opcode yaxpeax should report (None: not compared) and the operands.
fn simd_expected(op: &SimdOp) -> Option<(Option<String>, Vec<V>)> {
    use SimdOp::*;
    let s = |name: &str| Some(name.to_owned());
    let gpr = |bits: u8, n: u8| V::General(Op::Zr(bits == 64, n));
    let mem = |rn: u8, address: Address| {
        V::General(match address {
            Address::Immediate { offset, writeback } => Op::Mem(rn, offset, writeback),
            Address::Register { rm, option, shift } => {
                Op::MemReg(rn, rm, option & 1 == 1, option, shift)
            }
        })
    };
    Some(match *op {
        LoadStore {
            rt,
            rn,
            bytes,
            address,
            ..
        } => (
            None,
            vec![V::Scalar(u32::from(bytes) * 8, rt), mem(rn, address)],
        ),
        LoadStorePair {
            rt,
            rt2,
            rn,
            bytes,
            offset,
            writeback,
            ..
        } => (
            None,
            vec![
                V::Scalar(u32::from(bytes) * 8, rt),
                V::Scalar(u32::from(bytes) * 8, rt2),
                V::General(Op::Mem(rn, offset, writeback)),
            ],
        ),
        LoadLiteral { rt, bytes, address } => (
            s("LDR"),
            vec![
                V::Scalar(u32::from(bytes) * 8, rt),
                V::General(Op::Target(address)),
            ],
        ),
        FpUnary { op, size, rd, rn } => (
            match op {
                FpUnaryOp::Mov => s("FMOV"),
                FpUnaryOp::Abs => s("FABS"),
                FpUnaryOp::Neg => s("FNEG"),
                FpUnaryOp::Sqrt => s("FSQRT"),
                FpUnaryOp::Round(r) => Some(format!(
                    "FRINT{}",
                    match r {
                        Rounding::Nearest => "N",
                        Rounding::Plus => "P",
                        Rounding::Minus => "M",
                        Rounding::Zero => "Z",
                        Rounding::Away => "A",
                    }
                )),
                FpUnaryOp::RoundExact => s("FRINTX"),
            },
            vec![V::Scalar(fp_bits(size), rd), V::Scalar(fp_bits(size), rn)],
        ),
        FpConvert { rd, rn, from, to } => (
            s("FCVT"),
            vec![V::Scalar(fp_bits(to), rd), V::Scalar(fp_bits(from), rn)],
        ),
        FpConvertPrecision { rd, rn, from, to } => (
            s("FCVT"),
            vec![
                V::Scalar(precision_bits(to), rd),
                V::Scalar(precision_bits(from), rn),
            ],
        ),
        FpBinary {
            op,
            size,
            rd,
            rn,
            rm,
        } => (
            Some(match op {
                FpBinaryOp::MaxNum => "FMAXNM".into(),
                FpBinaryOp::MinNum => "FMINNM".into(),
                FpBinaryOp::NegMul => "FNMUL".into(),
                other => format!("F{}", format!("{other:?}").to_uppercase()),
            }),
            vec![
                V::Scalar(fp_bits(size), rd),
                V::Scalar(fp_bits(size), rn),
                V::Scalar(fp_bits(size), rm),
            ],
        ),
        FpTernary {
            op,
            size,
            rd,
            rn,
            rm,
            ra,
        } => (
            s(match op {
                FpTernaryOp::MulAdd => "FMADD",
                FpTernaryOp::MulSub => "FMSUB",
                FpTernaryOp::NegMulAdd => "FNMADD",
                FpTernaryOp::NegMulSub => "FNMSUB",
            }),
            vec![
                V::Scalar(fp_bits(size), rd),
                V::Scalar(fp_bits(size), rn),
                V::Scalar(fp_bits(size), rm),
                V::Scalar(fp_bits(size), ra),
            ],
        ),
        FpCompare {
            size,
            rn,
            rm,
            signaling,
        } => (
            s(if signaling { "FCMPE" } else { "FCMP" }),
            vec![
                V::Scalar(fp_bits(size), rn),
                match rm {
                    Some(rm) => V::Scalar(fp_bits(size), rm),
                    None => V::Float(0.0_f64.to_bits()),
                },
            ],
        ),
        FpConditionalCompare {
            size,
            rn,
            rm,
            condition,
            nzcv,
            signaling,
        } => (
            s(if signaling { "FCCMPE" } else { "FCCMP" }),
            vec![
                V::Scalar(fp_bits(size), rn),
                V::Scalar(fp_bits(size), rm),
                V::General(Op::Imm(u64::from(nzcv))),
                V::General(Op::Cond(condition)),
            ],
        ),
        FpSelect {
            size,
            rd,
            rn,
            rm,
            condition,
        } => (
            s("FCSEL"),
            vec![
                V::Scalar(fp_bits(size), rd),
                V::Scalar(fp_bits(size), rn),
                V::Scalar(fp_bits(size), rm),
                V::General(Op::Cond(condition)),
            ],
        ),
        FpMoveImmediate { size, rd, bits } => {
            let value = match size {
                FpSize::S => f64::from(f32::from_bits(bits as u32)),
                FpSize::D => f64::from_bits(bits),
            };
            (
                s("FMOV"),
                vec![V::Scalar(fp_bits(size), rd), V::Float(value.to_bits())],
            )
        }
        FpMoveToGeneral {
            rd,
            rn,
            bits,
            upper,
        } => (
            s("FMOV"),
            vec![
                gpr(bits, rd),
                if upper {
                    V::Lane(rn, 64, 1)
                } else {
                    V::Scalar(u32::from(bits), rn)
                },
            ],
        ),
        FpMoveFromGeneral {
            rd,
            rn,
            bits,
            upper,
        } => (
            s("FMOV"),
            vec![
                if upper {
                    V::Lane(rd, 64, 1)
                } else {
                    V::Scalar(u32::from(bits), rd)
                },
                gpr(bits, rn),
            ],
        ),
        IntToFp {
            rd,
            rn,
            gp_bits,
            size,
            signed,
        } => (
            s(if signed { "SCVTF" } else { "UCVTF" }),
            vec![V::Scalar(fp_bits(size), rd), gpr(gp_bits, rn)],
        ),
        IntToFpFixed {
            rd,
            rn,
            gp_bits,
            size,
            signed,
            fbits,
        } => (
            s(if signed { "SCVTF" } else { "UCVTF" }),
            vec![
                V::Scalar(fp_bits(size), rd),
                gpr(gp_bits, rn),
                V::General(Op::Imm(u64::from(fbits))),
            ],
        ),
        SimdIntToFp {
            rd,
            rn,
            size,
            signed,
        } => (
            s(if signed { "SCVTF" } else { "UCVTF" }),
            vec![V::Scalar(fp_bits(size), rd), V::Scalar(fp_bits(size), rn)],
        ),
        FpToInt {
            rd,
            rn,
            gp_bits,
            size,
            signed,
            rounding,
        } => (
            Some(format!(
                "FCVT{}{}",
                rounding_letter(rounding),
                if signed { "S" } else { "U" }
            )),
            vec![gpr(gp_bits, rd), V::Scalar(fp_bits(size), rn)],
        ),
        FpToFixed {
            rd,
            rn,
            gp_bits,
            size,
            signed,
            fbits,
        } => (
            s(if signed { "FCVTZS" } else { "FCVTZU" }),
            vec![
                gpr(gp_bits, rd),
                V::Scalar(fp_bits(size), rn),
                V::General(Op::Imm(u64::from(fbits))),
            ],
        ),
        VectorLogical { op, rd, rn, rm, q } => (
            Some(format!("{op:?}").to_uppercase()),
            vec![
                V::Vector(rd, q, 8),
                V::Vector(rn, q, 8),
                V::Vector(rm, q, 8),
            ],
        ),
        InsertElement {
            rd,
            rn,
            bytes,
            dst,
            src,
        } => (
            s("INS"),
            vec![
                V::Lane(rd, u32::from(bytes) * 8, dst),
                V::Lane(rn, u32::from(bytes) * 8, src),
            ],
        ),
        InsertGeneral {
            rd,
            rn,
            bytes,
            index,
        } => (
            s("INS"),
            vec![
                V::Lane(rd, u32::from(bytes) * 8, index),
                gpr(if bytes == 8 { 64 } else { 32 }, rn),
            ],
        ),
        MoveToGeneral {
            rd,
            rn,
            bytes,
            index,
            signed,
            gp_bits,
        } => (
            s(if signed { "SMOV" } else { "UMOV" }),
            vec![gpr(gp_bits, rd), V::Lane(rn, u32::from(bytes) * 8, index)],
        ),
        DupElement {
            rd,
            rn,
            bytes,
            index,
            q,
            scalar,
        } => (
            None,
            vec![
                reg(rd, scalar, q, u32::from(bytes) * 8),
                V::Lane(rn, u32::from(bytes) * 8, index),
            ],
        ),
        DupGeneral { rd, rn, bytes, q } => (
            s("DUP"),
            vec![
                V::Vector(rd, q, u32::from(bytes) * 8),
                gpr(if bytes == 8 { 64 } else { 32 }, rn),
            ],
        ),
        VecFp {
            op,
            size,
            q,
            scalar,
            rd,
            rn,
            rm,
        } => (
            Some(vec_fp_name(op)),
            vec![
                reg(rd, scalar, q, fp_bits(size)),
                reg(rn, scalar, q, fp_bits(size)),
                reg(rm, scalar, q, fp_bits(size)),
            ],
        ),
        VecInt {
            op,
            bits,
            q,
            scalar,
            rd,
            rn,
            rm,
        } => (
            Some(vec_int_name(op)),
            vec![
                reg(rd, scalar, q, u32::from(bits)),
                reg(rn, scalar, q, u32::from(bits)),
                reg(rm, scalar, q, u32::from(bits)),
            ],
        ),
        VecUnary {
            op,
            bits,
            q,
            scalar,
            rd,
            rn,
        } => (
            match op {
                VecUnaryOp::CmEqZero => s("CMEQ"),
                VecUnaryOp::CmGeZero => s("CMGE"),
                VecUnaryOp::CmGtZero => s("CMGT"),
                VecUnaryOp::CmLeZero => s("CMLE"),
                VecUnaryOp::CmLtZero => s("CMLT"),
                other => Some(format!("{other:?}").to_uppercase()),
            },
            {
                let mut operands = vec![
                    reg(rd, scalar, q, u32::from(bits)),
                    reg(rn, scalar, q, u32::from(bits)),
                ];
                if matches!(
                    op,
                    VecUnaryOp::CmEqZero
                        | VecUnaryOp::CmGeZero
                        | VecUnaryOp::CmGtZero
                        | VecUnaryOp::CmLeZero
                        | VecUnaryOp::CmLtZero
                ) {
                    operands.push(V::General(Op::Imm(0)));
                }
                operands
            },
        ),
        VecPermute {
            op,
            bits,
            q,
            rd,
            rn,
            rm,
        } => (
            Some(format!("{op:?}").to_uppercase()),
            vec![
                V::Vector(rd, q, u32::from(bits)),
                V::Vector(rn, q, u32::from(bits)),
                V::Vector(rm, q, u32::from(bits)),
            ],
        ),
        VecExt {
            q,
            rd,
            rn,
            rm,
            index,
        } => (
            s("EXT"),
            vec![
                V::Vector(rd, q, 8),
                V::Vector(rn, q, 8),
                V::Vector(rm, q, 8),
                V::General(Op::Imm(u64::from(index))),
            ],
        ),
        VecTable {
            q,
            extend,
            rd,
            rn,
            regs,
            rm,
        } => (
            s(if extend { "TBX" } else { "TBL" }),
            vec![
                V::Vector(rd, q, 8),
                V::Group(rn, true, 8, regs),
                V::Vector(rm, q, 8),
            ],
        ),
        TransferMultiple {
            rt,
            rn,
            registers,
            structure,
            bits,
            q,
            load,
            post,
        } => (
            Some(format!("{}{structure}", if load { "LD" } else { "ST" })),
            vec![
                V::Group(rt, q, u32::from(bits), registers),
                post_address(rn, post, u64::from(registers) * if q { 16 } else { 8 }),
            ],
        ),
        TransferSingle {
            rt,
            rn,
            registers,
            bits,
            index,
            load,
            post,
        } => (
            Some(format!("{}{registers}", if load { "LD" } else { "ST" })),
            vec![
                V::GroupLane(rt, u32::from(bits), registers, index),
                post_address(rn, post, u64::from(registers) * u64::from(bits) / 8),
            ],
        ),
        LoadReplicate {
            rt,
            rn,
            registers,
            bits,
            q,
            post,
        } => (
            Some(format!("LD{registers}R")),
            vec![
                V::Group(rt, q, u32::from(bits), registers),
                post_address(rn, post, u64::from(registers) * u64::from(bits) / 8),
            ],
        ),
        VecIntElement {
            op,
            bits,
            q,
            scalar,
            rd,
            rn,
            rm,
            index,
        } => (
            Some(vec_int_name(op)),
            vec![
                reg(rd, scalar, q, u32::from(bits)),
                reg(rn, scalar, q, u32::from(bits)),
                V::Lane(rm, u32::from(bits), index),
            ],
        ),
        VecFpElement {
            op,
            size,
            q,
            scalar,
            rd,
            rn,
            rm,
            index,
        } => (
            Some(vec_fp_name(op)),
            vec![
                reg(rd, scalar, q, fp_bits(size)),
                reg(rn, scalar, q, fp_bits(size)),
                V::Lane(rm, fp_bits(size), index),
            ],
        ),
        VecFpPairwise {
            op,
            size,
            q,
            scalar,
            rd,
            rn,
            rm,
        } => (
            Some(format!("{}P", vec_fp_name(op))),
            if scalar {
                vec![
                    V::Scalar(fp_bits(size), rd),
                    V::Vector(rn, size == FpSize::D, fp_bits(size)),
                ]
            } else {
                vec![
                    V::Vector(rd, q, fp_bits(size)),
                    V::Vector(rn, q, fp_bits(size)),
                    V::Vector(rm, q, fp_bits(size)),
                ]
            },
        ),
        VecIntPairwise {
            op,
            bits,
            q,
            scalar,
            rd,
            rn,
            rm,
        } => (
            Some(format!("{}P", vec_int_name(op))),
            if scalar {
                vec![V::Scalar(64, rd), V::Vector(rn, true, 64)]
            } else {
                vec![
                    V::Vector(rd, q, u32::from(bits)),
                    V::Vector(rn, q, u32::from(bits)),
                    V::Vector(rm, q, u32::from(bits)),
                ]
            },
        ),
        VecLong {
            op,
            signed,
            bits,
            upper,
            rd,
            rn,
            rm,
        } => {
            let bits = u32::from(bits);
            let wide_source = matches!(op, VecLongOp::Addw | VecLongOp::Subw);
            (
                Some(format!(
                    "{}{}{}",
                    if signed { "S" } else { "U" },
                    format!("{op:?}").to_uppercase(),
                    if upper { "2" } else { "" }
                )),
                vec![
                    V::Vector(rd, true, 2 * bits),
                    if wide_source {
                        V::Vector(rn, true, 2 * bits)
                    } else {
                        V::Vector(rn, upper, bits)
                    },
                    V::Vector(rm, upper, bits),
                ],
            )
        }
        VecLongElement {
            op,
            signed,
            bits,
            upper,
            rd,
            rn,
            rm,
            index,
        } => (
            Some(format!(
                "{}{}{}",
                if signed { "S" } else { "U" },
                format!("{op:?}").to_uppercase(),
                if upper { "2" } else { "" }
            )),
            vec![
                V::Vector(rd, true, 2 * u32::from(bits)),
                V::Vector(rn, upper, u32::from(bits)),
                V::Lane(rm, u32::from(bits), index),
            ],
        ),
        VecNarrow {
            bits,
            upper,
            scalar,
            rd,
            rn,
            shift,
            ..
        } => {
            let bits = u32::from(bits);
            let mut operands = if scalar {
                vec![V::Scalar(bits / 2, rd), V::Scalar(bits, rn)]
            } else {
                vec![V::Vector(rd, upper, bits / 2), V::Vector(rn, true, bits)]
            };
            if shift != 0 {
                operands.push(V::General(Op::Imm(u64::from(shift))));
            }
            (None, operands)
        }
        VecPolyMulLong {
            bits,
            upper,
            rd,
            rn,
            rm,
        } => (
            s(if upper { "PMULL2" } else { "PMULL" }),
            vec![
                V::Vector(rd, true, 2 * u32::from(bits)),
                V::Vector(rn, upper, u32::from(bits)),
                V::Vector(rm, upper, u32::from(bits)),
            ],
        ),
        VecShift {
            op,
            bits,
            q,
            scalar,
            rd,
            rn,
            shift,
        } => (
            None,
            vec![
                if op == VecShiftOp::Shrn {
                    V::Vector(rd, q, u32::from(bits) / 2)
                } else {
                    reg(rd, scalar, q, u32::from(bits))
                },
                if op == VecShiftOp::Shrn {
                    V::Vector(rn, true, u32::from(bits))
                } else {
                    reg(rn, scalar, q, u32::from(bits))
                },
                V::General(Op::Imm(u64::from(shift))),
            ],
        ),
        VecShiftLong {
            signed,
            bits,
            upper,
            rd,
            rn,
            shift,
        } => (
            Some(format!(
                "{}SHLL{}",
                if signed { "S" } else { "U" },
                if upper { "2" } else { "" }
            )),
            vec![
                V::Vector(rd, true, 2 * u32::from(bits)),
                V::Vector(rn, upper, u32::from(bits)),
                V::General(Op::Imm(u64::from(shift))),
            ],
        ),
        VecPairwiseLong {
            signed,
            accumulate,
            bits,
            q,
            rd,
            rn,
        } => (
            Some(format!(
                "{}{}",
                if signed { "S" } else { "U" },
                if accumulate { "ADALP" } else { "ADDLP" }
            )),
            vec![
                V::Vector(rd, q, 2 * u32::from(bits)),
                V::Vector(rn, q, u32::from(bits)),
            ],
        ),
        VecAcross {
            op,
            bits,
            q,
            rd,
            rn,
        } => {
            let widening = matches!(op, VecAcrossOp::SAddLV | VecAcrossOp::UAddLV);
            (
                Some(format!("{op:?}").to_uppercase()),
                vec![
                    V::Scalar(u32::from(bits) * if widening { 2 } else { 1 }, rd),
                    V::Vector(rn, q, u32::from(bits)),
                ],
            )
        }
        VecFpAcross { op, rd, rn } => (
            Some(format!("{}V", vec_fp_name(op))),
            vec![V::Scalar(32, rd), V::Vector(rn, true, 32)],
        ),
        VecFpConvert {
            from,
            to,
            upper,
            rd,
            rn,
        } => {
            let (from, to) = (precision_bits(from), precision_bits(to));
            let widening = to > from;
            (
                Some(format!(
                    "FCVT{}{}",
                    if widening { "L" } else { "N" },
                    if upper { "2" } else { "" }
                )),
                if widening {
                    vec![V::Vector(rd, true, to), V::Vector(rn, upper, from)]
                } else {
                    vec![V::Vector(rd, upper, to), V::Vector(rn, true, from)]
                },
            )
        }
        VecFpUnary {
            op,
            size,
            q,
            scalar,
            rd,
            rn,
            fbits,
        } => {
            let mut operands = vec![
                reg(rd, scalar, q, fp_bits(size)),
                reg(rn, scalar, q, fp_bits(size)),
            ];
            match op {
                VecFpUnaryOp::CmEqZero
                | VecFpUnaryOp::CmGeZero
                | VecFpUnaryOp::CmGtZero
                | VecFpUnaryOp::CmLeZero
                | VecFpUnaryOp::CmLtZero => operands.push(V::Float(0.0_f64.to_bits())),
                _ if fbits != 0 => operands.push(V::General(Op::Imm(u64::from(fbits)))),
                _ => {}
            }
            (None, operands)
        }
        VecDot {
            signed,
            q,
            rd,
            rn,
            rm,
        } => (
            s(if signed { "SDOT" } else { "UDOT" }),
            vec![
                V::Vector(rd, q, 32),
                V::Vector(rn, q, 8),
                V::Vector(rm, q, 8),
            ],
        ),
        // Immediates and the mixed scalar/vector SHA forms: register
        // numbers only (`simd_register_numbers`).
        Crypto { .. } | MoveImmediate { .. } | ModifyImmediate { .. } => return None,
    })
}

/// The SIMD register numbers of the shapes `simd_expected` leaves out, in
/// operand order.
fn simd_register_numbers(op: &SimdOp) -> Vec<u8> {
    match *op {
        SimdOp::Crypto { op, rd, rn, rm } => {
            let three = !matches!(
                format!("{op:?}").as_str(),
                "Aese" | "Aesd" | "Aesmc" | "Aesimc" | "Sha1h" | "Sha1su1" | "Sha256su0"
            );
            if three {
                vec![rd, rn, rm]
            } else {
                vec![rd, rn]
            }
        }
        SimdOp::MoveImmediate { rd, .. } | SimdOp::ModifyImmediate { rd, .. } => vec![rd],
        _ => Vec::new(),
    }
}

/// SIMD opcode names that are the same operation as ours.
fn simd_same_operation(ours: &str, theirs: &str) -> bool {
    // FRINTI rounds in the FPCR mode, which the runtime holds at
    // round-to-nearest-even (simd.rs: other modes stop explicitly).
    let frinti = ours == "FRINTN" && theirs == "FRINTI";
    // SHLL shifts by the element size; USHLL/SSHLL by that amount is the
    // same result (the extension is shifted out), and the operands, shift
    // included, are compared as usual.
    let shll = matches!(
        (ours, theirs),
        ("USHLL" | "SSHLL", "SHLL") | ("USHLL2" | "SSHLL2", "SHLL2")
    );
    frinti || shll
}

/// The address of a structure transfer: post-indexed by a register, or by
/// the transfer size when `post` is 31.
fn post_address(rn: u8, post: Option<u8>, size: u64) -> V {
    V::General(match post {
        None => Op::Mem(rn, 0, Writeback::None),
        Some(31) => Op::Mem(rn, size as i64, Writeback::Post),
        Some(rm) => Op::Other(format!("RegPostIndexReg({rn}, {rm})")),
    })
}

/// The register numbers a SIMD operation names, in operand order: the
/// fallback comparison for shapes `simd_expected` leaves out.
fn simd_registers(operands: &[V]) -> Vec<u8> {
    operands
        .iter()
        .filter_map(|operand| match *operand {
            V::Scalar(_, n)
            | V::Vector(n, _, _)
            | V::Lane(n, _, _)
            | V::Group(n, _, _, _)
            | V::GroupLane(n, _, _, _) => Some(n),
            V::General(Op::Zr(_, n) | Op::Sp(_, n)) => Some(n),
            _ => None,
        })
        .collect()
}

/// Words in the SIMD and floating-point encoding space: data processing
/// (bits 27:25 = 111) and, more densely, its scalar floating-point,
/// scalar SIMD and two-register/across-lanes parts; and loads and stores of
/// SIMD registers (bits 27:25 = 110).
fn simd_word(random: &mut Random) -> u32 {
    let word = random.next();
    let (mask, value) = match random.next() % 5 {
        0 => (0x0e00_0000, 0x0e00_0000),
        1 => (0x5f20_0000, 0x1e20_0000),
        2 => (0xdf00_0000, 0x5e00_0000),
        3 => (0x9f20_0c00, 0x0e20_0800),
        _ => (0x0e00_0000, 0x0c00_0000),
    };
    (word & !mask) | value
}

fn variant_name(op: &SimdOp) -> String {
    let text = format!("{op:?}");
    text.split([' ', '(', '{'])
        .next()
        .unwrap_or_default()
        .to_owned()
}

#[test]
fn simd_operands_agree_with_yaxpeax() {
    let mut random = Random(0x0123_4567_89ab_cdef);
    let mut counts: std::collections::BTreeMap<String, (u32, u32)> = Default::default();
    let mut failures: std::collections::BTreeMap<String, Vec<String>> = Default::default();
    for _ in 0..4_000_000 {
        let word = simd_word(&mut random);
        let Instruction::Simd(op) = decode(word, PC) else {
            continue;
        };
        let variant = variant_name(&op);
        let count = counts.entry(variant.clone()).or_default();
        count.0 += 1;
        let Some(instruction) = theirs(word) else {
            failures
                .entry(variant)
                .or_default()
                .push(format!("{word:#010x} {op:?}: invalid for yaxpeax"));
            continue;
        };
        let actual: Vec<V> = instruction
            .operands
            .iter()
            .filter_map(normalize_simd)
            .collect();
        let name = format!("{:?}", instruction.opcode);
        count.1 += 1;
        let problem = match simd_expected(&op) {
            Some((expected_name, expected)) => {
                let name_ok = expected_name.as_ref().is_none_or(|expected| {
                    *expected == name || simd_same_operation(expected, &name)
                });
                (!name_ok || expected != actual)
                    .then(|| format!("expects {expected_name:?} {expected:?}"))
            }
            None => {
                let expected = simd_register_numbers(&op);
                (expected != simd_registers(&actual))
                    .then(|| format!("expects registers {expected:?}"))
            }
        };
        if let Some(problem) = problem {
            failures.entry(variant).or_default().push(format!(
                "{word:#010x} {op:?}: {problem}, yaxpeax {name} {actual:?}"
            ));
        }
    }
    for (variant, (accepted, compared)) in &counts {
        println!("{variant:<24} accepted {accepted:>6} compared {compared:>6}");
    }
    let total: usize = failures.values().map(Vec::len).sum();
    let shown: Vec<String> = failures
        .values()
        .flat_map(|list| list.iter().take(3).cloned())
        .collect();
    assert!(
        failures.is_empty(),
        "{total} SIMD disagreements in {} variants:\n{}",
        failures.len(),
        shown.join("\n")
    );
}
