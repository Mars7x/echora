//! Conservative semantic decoder. Unsupported encodings remain explicit traps.

use echora_guest::cpu::{InstructionAuthKey, PAuthHint};
use serde::Serialize;
use yaxpeax_arch::{Decoder, U8Reader};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum LogicalOp {
    And,
    Orr,
    Eor,
    Ands,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum WideOp {
    Not,
    Zero,
    Keep,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum BitfieldOp {
    Signed,
    Merge,
    Unsigned,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Writeback {
    None,
    Pre,
    Post,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Address {
    Immediate { offset: i64, writeback: Writeback },
    Register { rm: u8, option: u8, shift: u8 },
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum MemoryOp {
    Store,
    Load { signed: bool, bits: u8 },
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum CompareOperand {
    Register(u8),
    Immediate(u8),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum Instruction {
    Nop,
    /// YIELD or WFE: a spin-wait hint (see `Core::spin_hint`).
    SpinHint,
    PAuthDisabledHint {
        hint: PAuthHint,
    },
    PAuthDisabledReturn {
        key: InstructionAuthKey,
    },
    StripLrInstructionPac,
    Barrier,
    /// CLREX #imm: return the local exclusive monitor to Open Access.
    ClearExclusive,
    Adr {
        rd: u8,
        address: u64,
    },
    Branch {
        target: u64,
        link: bool,
    },
    BranchCond {
        target: u64,
        condition: u8,
    },
    CompareBranch {
        rt: u8,
        bits: u8,
        nonzero: bool,
        target: u64,
    },
    TestBranch {
        rt: u8,
        bit: u8,
        nonzero: bool,
        target: u64,
    },
    BranchReg {
        rn: u8,
        link: bool,
    },
    Svc {
        number: u16,
    },
    MoveWide {
        rd: u8,
        bits: u8,
        op: WideOp,
        immediate: u16,
        shift: u8,
    },
    AddSubImmediate {
        rd: u8,
        rn: u8,
        bits: u8,
        immediate: u64,
        subtract: bool,
        flags: bool,
    },
    AddSubShifted {
        rd: u8,
        rn: u8,
        rm: u8,
        bits: u8,
        shift: u8,
        amount: u8,
        subtract: bool,
        flags: bool,
    },
    AddSubExtended {
        rd: u8,
        rn: u8,
        rm: u8,
        bits: u8,
        option: u8,
        amount: u8,
        subtract: bool,
        flags: bool,
    },
    AddCarry {
        rd: u8,
        rn: u8,
        rm: u8,
        bits: u8,
        subtract: bool,
        flags: bool,
    },
    LogicalShifted {
        rd: u8,
        rn: u8,
        rm: u8,
        bits: u8,
        op: LogicalOp,
        invert: bool,
        shift: u8,
        amount: u8,
    },
    LogicalImmediate {
        rd: u8,
        rn: u8,
        bits: u8,
        op: LogicalOp,
        mask: u64,
    },
    Bitfield {
        rd: u8,
        rn: u8,
        bits: u8,
        op: BitfieldOp,
        rotation: u8,
        sign_bit: u8,
        write_mask: u64,
        top_mask: u64,
    },
    Extract {
        rd: u8,
        rn: u8,
        rm: u8,
        bits: u8,
        lsb: u8,
    },
    Select {
        rd: u8,
        rn: u8,
        rm: u8,
        bits: u8,
        condition: u8,
        operation: u8,
    },
    ConditionalCompare {
        rn: u8,
        bits: u8,
        rhs: CompareOperand,
        condition: u8,
        nzcv: u8,
    },
    MultiplyAdd {
        rd: u8,
        rn: u8,
        rm: u8,
        ra: u8,
        bits: u8,
        subtract: bool,
    },
    UnsignedMultiplyAddLong {
        rd: u8,
        rn: u8,
        rm: u8,
        ra: u8,
    },
    MultiplyHigh {
        rd: u8,
        rn: u8,
        rm: u8,
        signed: bool,
    },
    Divide {
        rd: u8,
        rn: u8,
        rm: u8,
        bits: u8,
        signed: bool,
    },
    VariableShift {
        rd: u8,
        rn: u8,
        rm: u8,
        bits: u8,
        shift: u8,
    },
    CountLeading {
        rd: u8,
        rn: u8,
        bits: u8,
        sign: bool,
    },
    ReverseBits {
        rd: u8,
        rn: u8,
        bits: u8,
    },
    ReverseBytes {
        rd: u8,
        rn: u8,
        bits: u8,
        group_bits: u8,
    },
    /// Single-register LSE CAS. Ordering fields retain the encoded A/L bits.
    CompareAndSwap {
        rs: u8,
        rt: u8,
        rn: u8,
        bytes: u8,
        acquire: bool,
        release: bool,
    },
    /// LSE CASP{A,L,AL}: pair compare-and-swap on even register pairs;
    /// `bytes` is the element size (4 or 8).
    CompareAndSwapPair {
        rs: u8,
        rt: u8,
        rn: u8,
        bytes: u8,
        acquire: bool,
        release: bool,
    },
    /// LSE LDADD/B/H, including STADD aliases. Ordering fields retain encoded
    /// A/R bits; effective acquire additionally requires Rt != 31.
    AtomicAdd {
        rs: u8,
        rt: u8,
        rn: u8,
        bytes: u8,
        acquire: bool,
        release: bool,
    },
    /// Canonical no-offset LDAPR/B/H, with conservative single-thread ordering.
    LoadAcquireRcpc {
        rt: u8,
        rn: u8,
        bytes: u8,
    },
    /// Canonical no-offset LDAR/B/H: RCsc acquire, distinct from LDAPR's RCpc.
    LoadAcquireRcsc {
        rt: u8,
        rn: u8,
        bytes: u8,
    },
    /// LDXR/LDAXR B/H/W/X: zero-extending read that marks the local monitor.
    LoadExclusive {
        rt: u8,
        rn: u8,
        bytes: u8,
        acquire: bool,
    },
    /// STXR/STLXR B/H/W/X: Ws receives 0 when stored and 1 when not.
    StoreExclusive {
        rs: u8,
        rt: u8,
        rn: u8,
        bytes: u8,
        release: bool,
    },
    /// Canonical no-offset STLR/B/H, with single-thread release ordering.
    StoreRelease {
        rt: u8,
        rn: u8,
        bytes: u8,
    },
    LoadStore {
        rt: u8,
        rn: u8,
        bytes: u8,
        op: MemoryOp,
        address: Address,
    },
    LoadStorePair {
        rt: u8,
        rt2: u8,
        rn: u8,
        bytes: u8,
        op: MemoryOp,
        offset: i64,
        writeback: Writeback,
    },
    /// A pair of raw 128-bit Q registers; no floating-point arithmetic.
    LoadStoreVectorPair {
        rt: u8,
        rt2: u8,
        rn: u8,
        load: bool,
        offset: i64,
        writeback: Writeback,
    },
    LoadLiteral {
        rt: u8,
        bytes: u8,
        signed: bool,
        bits: u8,
        address: u64,
    },
    /// MRS CNTPCT_EL0 under the runtime's enabled EL0 access model. The 64-bit
    /// counter is sampled during execution, never folded during translation.
    ReadPhysicalCounter {
        rt: u8,
    },
    ReadThreadPointer {
        rt: u8,
        read_only: bool,
    },
    WriteThreadPointer {
        rt: u8,
    },
    /// CCMN: as CCMP, but compares Rn with the negated operand (Rn + op).
    ConditionalCompareNegative {
        rn: u8,
        bits: u8,
        rhs: CompareOperand,
        condition: u8,
        nzcv: u8,
    },
    /// SMADDL/SMSUBL/UMSUBL (UMADDL keeps its own variant).
    MultiplyAddLong {
        rd: u8,
        rn: u8,
        rm: u8,
        ra: u8,
        signed: bool,
        subtract: bool,
    },
    /// LSE LD<op>/ST<op> other than ADD, and SWP. Ordering fields keep the
    /// encoded A/R bits; effective acquire additionally requires Rt != 31
    /// for the LD<op> forms (SWP always acquires when A is set).
    AtomicMemory {
        op: AtomicOp,
        rs: u8,
        rt: u8,
        rn: u8,
        bytes: u8,
        acquire: bool,
        release: bool,
    },
    /// MRS Xt, FPCR/FPSR.
    ReadFpControl {
        rt: u8,
        status: bool,
    },
    /// MSR FPCR/FPSR, Xt.
    WriteFpControl {
        rt: u8,
        status: bool,
    },
    Simd(crate::simd::SimdOp),
    /// XPACI/XPACD Xd: strip a pointer authentication code without checking.
    StripPac {
        rd: u8,
    },
    /// PACGA Xd, Xn, Xm|SP: a generic authentication code in Xd's upper
    /// 32 bits (lower 32 zero). Only equality between computations matters
    /// (setjmp signs registers that longjmp signs again and compares), so
    /// any deterministic function of the operands serves as the key.
    PacGeneric {
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// PAC*/AUT* register forms with every key disabled: the register is
    /// unchanged (authentication neither adds nor checks a code).
    PAuthDisabledData,
    /// MRS Xt, CTR_EL0.
    ReadCacheType {
        rt: u8,
    },
    /// MRS Xt, NZCV: the flags in bits 31:28, every other bit zero.
    ReadNzcv {
        rt: u8,
    },
    /// MSR NZCV, Xt: the flags from bits 31:28; other bits are ignored.
    WriteNzcv {
        rt: u8,
    },
    /// CRC32{B,H,W,X} and CRC32C{B,H,W,X}: `Wd = crc(Wn, low bytes of Rm)`,
    /// reflected bit order, no initial or final inversion.
    Crc32 {
        rd: u8,
        rn: u8,
        rm: u8,
        bytes: u8,
        castagnoli: bool,
    },
    /// DC CVAC/CVAU/CIVAC and IC IVAU. Guest memory is coherent in this
    /// model and translated code cannot change without trapping, so cache
    /// maintenance has no observable effect.
    CacheMaintenance,
    /// UDF #imm16: permanently undefined.
    Undefined,
    Unsupported,
}

pub use echora_guest::memory::AtomicOp;

fn signed(value: u32, bits: u8) -> i64 {
    ((value as i64) << (64 - bits)) >> (64 - bits)
}
fn target(pc: u64, immediate: u32, bits: u8) -> u64 {
    pc.wrapping_add((signed(immediate, bits) << 2) as u64)
}
fn logical_op(word: u32) -> LogicalOp {
    match (word >> 29) & 3 {
        0 => LogicalOp::And,
        1 => LogicalOp::Orr,
        2 => LogicalOp::Eor,
        _ => LogicalOp::Ands,
    }
}
fn ones(bits: u8) -> u64 {
    if bits == 64 {
        u64::MAX
    } else {
        (1_u64 << bits) - 1
    }
}

/// ARM DecodeBitMasks, restricted to architectural 32- and 64-bit operands.
pub fn decode_bit_masks(
    n: u8,
    imms: u8,
    immr: u8,
    immediate: bool,
    bits: u8,
) -> Option<(u64, u64)> {
    if !matches!(bits, 32 | 64) || n > 1 || imms > 63 || immr > 63 {
        return None;
    }
    let combined = ((n as u32) << 6) | ((!imms as u32) & 63);
    if combined == 0 {
        return None;
    }
    let len = 31 - combined.leading_zeros();
    if len < 1 {
        return None;
    }
    let element = 1_u8 << len;
    if element > bits {
        return None;
    }
    let levels = element - 1;
    let s = imms & levels;
    let r = immr & levels;
    if immediate && s == levels {
        return None;
    }
    let difference = s.wrapping_sub(r) & levels;
    let initial = ones(s + 1);
    let rotated = if r == 0 {
        initial
    } else {
        ((initial >> r) | (initial << (element - r))) & ones(element)
    };
    let mut write_mask = 0;
    let mut top_mask = 0;
    let mut offset = 0;
    while offset < bits {
        write_mask |= rotated << offset;
        top_mask |= ones(difference + 1) << offset;
        offset += element;
    }
    Some((write_mask, top_mask))
}

fn memory_op(word: u32, size: u8) -> Option<MemoryOp> {
    match (word >> 22) & 3 {
        0 => Some(MemoryOp::Store),
        1 => Some(MemoryOp::Load {
            signed: false,
            bits: if size == 3 { 64 } else { 32 },
        }),
        2 if size < 3 => Some(MemoryOp::Load {
            signed: true,
            bits: 64,
        }),
        3 if size < 2 => Some(MemoryOp::Load {
            signed: true,
            bits: 32,
        }),
        _ => None,
    }
}

pub fn decode(word: u32, pc: u64) -> Instruction {
    use Instruction::*;
    let rd = (word & 31) as u8;
    let rn = ((word >> 5) & 31) as u8;
    let rm = ((word >> 16) & 31) as u8;
    let bits = if word >> 31 != 0 { 64 } else { 32 };
    let subtract = word & (1 << 30) != 0;
    if word >> 16 == 0 {
        return Undefined;
    }
    let flags = word & (1 << 29) != 0;
    // NOP, YIELD, WFE, WFI, SEV and SEVL: on this single PE the event
    // register needs no other observer, and WFE/WFI may complete at once.
    // YIELD and WFE mark spin-waits, which hand the processor on.
    if matches!(word, 0xd503_203f | 0xd503_205f) {
        return SpinHint;
    }
    if matches!(word, 0xd503_201f | 0xd503_207f | 0xd503_209f | 0xd503_20bf) {
        return Nop;
    }
    // PRFM (immediate, literal, register) and PRFUM: prefetch hints have no
    // architectural effect, including for unmapped addresses.
    if word & 0xffc0_0000 == 0xf980_0000
        || word & 0xff00_0000 == 0xd800_0000
        || word & 0xffe0_0c00 == 0xf8a0_0800
        || word & 0xffe0_0c00 == 0xf880_0000
    {
        return Nop;
    }
    match word & 0xffff_ffe0 {
        0xdac1_43e0 | 0xdac1_47e0 => return StripPac { rd },
        // CTR_EL0 is S3_3_C0_C0_1.
        0xd53b_0020 => return ReadCacheType { rt: rd },
        // DC CVAC, DC CVAU, DC CIVAC, IC IVAU.
        0xd50b_7a20 | 0xd50b_7b20 | 0xd50b_7e20 | 0xd50b_7520 => return CacheMaintenance,
        _ => {}
    }
    // PACIA/PACIB/PACDA/PACDB/AUTIA/AUTIB/AUTDA/AUTDB Xd, Xn|SP, and the Z
    // forms with Rn = 31.
    if word & 0xffff_e000 == 0xdac1_0000 || (word & 0xffff_e000 == 0xdac1_2000 && rn == 31) {
        return PAuthDisabledData;
    }
    // Explicit disabled-key semantics; do not accept all HINT encodings as NOP.
    // In particular XPACLRI strips PAC bits independently of these key enables.
    match word {
        0xd503_20ff => return StripLrInstructionPac,
        0xd503_233f => {
            return PAuthDisabledHint {
                hint: PAuthHint::PaciaSp,
            };
        }
        0xd503_237f => {
            return PAuthDisabledHint {
                hint: PAuthHint::PacibSp,
            };
        }
        0xd503_23bf => {
            return PAuthDisabledHint {
                hint: PAuthHint::AutiaSp,
            };
        }
        0xd503_23ff => {
            return PAuthDisabledHint {
                hint: PAuthHint::AutibSp,
            };
        }
        0xd503_211f => {
            return PAuthDisabledHint {
                hint: PAuthHint::Pacia1716,
            };
        }
        0xd503_215f => {
            return PAuthDisabledHint {
                hint: PAuthHint::Pacib1716,
            };
        }
        0xd503_219f => {
            return PAuthDisabledHint {
                hint: PAuthHint::Autia1716,
            };
        }
        0xd503_21df => {
            return PAuthDisabledHint {
                hint: PAuthHint::Autib1716,
            };
        }
        0xd65f_0bff => {
            return PAuthDisabledReturn {
                key: InstructionAuthKey::A,
            };
        }
        0xd65f_0fff => {
            return PAuthDisabledReturn {
                key: InstructionAuthKey::B,
            };
        }
        _ => {}
    }
    let barrier = word & 0xffff_f0ff;
    if ((barrier == 0xd503_309f || barrier == 0xd503_30bf) && (word >> 8) & 3 != 0)
        || (word == 0xd503_3fdf)
    {
        return Barrier;
    }
    // CLREX: CRm[11:8] is ignored by the architecture.
    if word & 0xffff_f0ff == 0xd503_305f {
        return ClearExclusive;
    }
    if word & 0x7c00_0000 == 0x1400_0000 {
        return Branch {
            target: target(pc, word & 0x03ff_ffff, 26),
            link: word >> 31 != 0,
        };
    }
    if word & 0xff00_0010 == 0x5400_0000 {
        return BranchCond {
            target: target(pc, (word >> 5) & 0x7ffff, 19),
            condition: (word & 15) as u8,
        };
    }
    if word & 0x7e00_0000 == 0x3400_0000 {
        return CompareBranch {
            rt: rd,
            bits,
            nonzero: word & (1 << 24) != 0,
            target: target(pc, (word >> 5) & 0x7ffff, 19),
        };
    }
    if word & 0x7e00_0000 == 0x3600_0000 {
        return TestBranch {
            rt: rd,
            bit: (((word >> 26) & 32) | ((word >> 19) & 31)) as u8,
            nonzero: word & (1 << 24) != 0,
            target: target(pc, (word >> 5) & 0x3fff, 14),
        };
    }
    match word & 0xffff_fc1f {
        0xd61f_0000 | 0xd65f_0000 => return BranchReg { rn, link: false },
        0xd63f_0000 => return BranchReg { rn, link: true },
        _ => {}
    }
    if word & 0xffe0_001f == 0xd400_0001 {
        return Svc {
            number: ((word >> 5) & 0xffff) as u16,
        };
    }
    if word & 0x1f00_0000 == 0x1000_0000 {
        let immediate = ((word >> 29) & 3) | (((word >> 5) & 0x7ffff) << 2);
        let offset = signed(immediate, 21);
        let address = if word >> 31 != 0 {
            (pc & !0xfff).wrapping_add((offset << 12) as u64)
        } else {
            pc.wrapping_add(offset as u64)
        };
        return Adr { rd, address };
    }
    if word & 0x1f80_0000 == 0x1280_0000 {
        let shift = (((word >> 21) & 3) * 16) as u8;
        if shift >= bits {
            return Unsupported;
        }
        let op = match (word >> 29) & 3 {
            0 => WideOp::Not,
            2 => WideOp::Zero,
            3 => WideOp::Keep,
            _ => return Unsupported,
        };
        return MoveWide {
            rd,
            bits,
            op,
            immediate: ((word >> 5) & 0xffff) as u16,
            shift,
        };
    }
    if word & 0x1f80_0000 == 0x1100_0000 {
        return AddSubImmediate {
            rd,
            rn,
            bits,
            immediate: (((word >> 10) & 0xfff) as u64) << (((word >> 22) & 1) * 12),
            subtract,
            flags,
        };
    }
    if word & 0x1f20_0000 == 0x0b00_0000 {
        let shift = ((word >> 22) & 3) as u8;
        let amount = ((word >> 10) & 63) as u8;
        if shift == 3 || amount >= bits {
            return Unsupported;
        }
        return AddSubShifted {
            rd,
            rn,
            rm,
            bits,
            shift,
            amount,
            subtract,
            flags,
        };
    }
    if word & 0x1fe0_0000 == 0x0b20_0000 {
        let amount = ((word >> 10) & 7) as u8;
        if amount > 4 {
            return Unsupported;
        }
        return AddSubExtended {
            rd,
            rn,
            rm,
            bits,
            option: ((word >> 13) & 7) as u8,
            amount,
            subtract,
            flags,
        };
    }
    if word & 0x1fe0_fc00 == 0x1a00_0000 {
        return AddCarry {
            rd,
            rn,
            rm,
            bits,
            subtract,
            flags,
        };
    }
    if word & 0x1f00_0000 == 0x0a00_0000 {
        let amount = ((word >> 10) & 63) as u8;
        if amount >= bits {
            return Unsupported;
        }
        return LogicalShifted {
            rd,
            rn,
            rm,
            bits,
            op: logical_op(word),
            invert: word & (1 << 21) != 0,
            shift: ((word >> 22) & 3) as u8,
            amount,
        };
    }
    if word & 0x1f80_0000 == 0x1200_0000 {
        let n = ((word >> 22) & 1) as u8;
        if bits == 32 && n != 0 {
            return Unsupported;
        }
        if let Some((mask, _)) = decode_bit_masks(
            n,
            ((word >> 10) & 63) as u8,
            ((word >> 16) & 63) as u8,
            true,
            bits,
        ) {
            return LogicalImmediate {
                rd,
                rn,
                bits,
                op: logical_op(word),
                mask,
            };
        }
        return Unsupported;
    }
    if word & 0x1f80_0000 == 0x1300_0000 {
        let n = ((word >> 22) & 1) as u8;
        let rotation = ((word >> 16) & 63) as u8;
        let sign_bit = ((word >> 10) & 63) as u8;
        if n != u8::from(bits == 64) || rotation >= bits || sign_bit >= bits {
            return Unsupported;
        }
        let op = match (word >> 29) & 3 {
            0 => BitfieldOp::Signed,
            1 => BitfieldOp::Merge,
            2 => BitfieldOp::Unsigned,
            _ => return Unsupported,
        };
        let (write_mask, top_mask) = decode_bit_masks(n, sign_bit, rotation, false, bits)
            .expect("validated bitfield encoding");
        return Bitfield {
            rd,
            rn,
            bits,
            op,
            rotation,
            sign_bit,
            write_mask,
            top_mask,
        };
    }
    if word & 0x7fa0_0000 == 0x1380_0000 {
        let n = ((word >> 22) & 1) as u8;
        let lsb = ((word >> 10) & 63) as u8;
        if n != u8::from(bits == 64) || lsb >= bits {
            return Unsupported;
        }
        return Extract {
            rd,
            rn,
            rm,
            bits,
            lsb,
        };
    }
    // CCMN: op=0, S=1, o2=o3=0.
    if word & 0x7fe0_0410 == 0x3a40_0000 {
        return ConditionalCompareNegative {
            rn,
            bits,
            rhs: if word & (1 << 11) == 0 {
                CompareOperand::Register(rm)
            } else {
                CompareOperand::Immediate(rm)
            },
            condition: ((word >> 12) & 15) as u8,
            nzcv: (word & 15) as u8,
        };
    }
    // CCMP only: op=1, S=1, o2=o3=0. Bit 11 selects imm5 or Rm.
    if word & 0x7fe0_0410 == 0x7a40_0000 {
        return ConditionalCompare {
            rn,
            bits,
            rhs: if word & (1 << 11) == 0 {
                CompareOperand::Register(rm)
            } else {
                CompareOperand::Immediate(rm)
            },
            condition: ((word >> 12) & 15) as u8,
            nzcv: (word & 15) as u8,
        };
    }
    if word & 0x3fe0_0800 == 0x1a80_0000 {
        return Select {
            rd,
            rn,
            rm,
            bits,
            condition: ((word >> 12) & 15) as u8,
            operation: (((word >> 29) & 2) | ((word >> 10) & 1)) as u8,
        };
    }
    if word & 0x7fe0_0000 == 0x1b00_0000 {
        return MultiplyAdd {
            rd,
            rn,
            rm,
            ra: ((word >> 10) & 31) as u8,
            bits,
            subtract: word & (1 << 15) != 0,
        };
    }
    // UMADDL (including its UMULL alias). Signed and subtract-long forms stay unsupported.
    if word & 0xffe0_8000 == 0x9ba0_0000 {
        return UnsignedMultiplyAddLong {
            rd,
            rn,
            rm,
            ra: ((word >> 10) & 31) as u8,
        };
    }
    // SMADDL/SMSUBL/UMSUBL: 1 00 11011 U 01 Rm o0 Ra Rn Rd.
    if word & 0xff60_0000 == 0x9b20_0000 {
        return MultiplyAddLong {
            rd,
            rn,
            rm,
            ra: ((word >> 10) & 31) as u8,
            signed: word & (1 << 23) == 0,
            subtract: word & (1 << 15) != 0,
        };
    }
    match word & 0xffe0_fc00 {
        0x9b40_7c00 => {
            return MultiplyHigh {
                rd,
                rn,
                rm,
                signed: true,
            };
        }
        0x9bc0_7c00 => {
            return MultiplyHigh {
                rd,
                rn,
                rm,
                signed: false,
            };
        }
        _ => {}
    }
    if word & 0xffe0_fc00 == 0x9ac0_3000 {
        return PacGeneric { rd, rn, rm };
    }
    // CRC32/CRC32C: sz selects the data width; only the X form uses sf=1.
    if word & 0x7fe0_e000 == 0x1ac0_4000 {
        let sz = (word >> 10) & 3;
        if (word >> 31 == 1) == (sz == 3) {
            return Crc32 {
                rd,
                rn,
                rm,
                bytes: 1 << sz,
                castagnoli: (word >> 12) & 1 != 0,
            };
        }
    }
    match word & 0x7fe0_fc00 {
        0x1ac0_0800 => {
            return Divide {
                rd,
                rn,
                rm,
                bits,
                signed: false,
            };
        }
        0x1ac0_0c00 => {
            return Divide {
                rd,
                rn,
                rm,
                bits,
                signed: true,
            };
        }
        0x1ac0_2000 | 0x1ac0_2400 | 0x1ac0_2800 | 0x1ac0_2c00 => {
            return VariableShift {
                rd,
                rn,
                rm,
                bits,
                shift: ((word >> 10) & 3) as u8,
            };
        }
        _ => {}
    }
    match word & 0x7fff_fc00 {
        0x5ac0_1000 => {
            return CountLeading {
                rd,
                rn,
                bits,
                sign: false,
            };
        }
        0x5ac0_1400 => {
            return CountLeading {
                rd,
                rn,
                bits,
                sign: true,
            };
        }
        0x5ac0_0000 => return ReverseBits { rd, rn, bits },
        0x5ac0_0400 => {
            return ReverseBytes {
                rd,
                rn,
                bits,
                group_bits: 16,
            };
        }
        0x5ac0_0800 => {
            return ReverseBytes {
                rd,
                rn,
                bits,
                group_bits: 32,
            };
        }
        0x5ac0_0c00 if bits == 64 => {
            return ReverseBytes {
                rd,
                rn,
                bits,
                group_bits: 64,
            };
        }
        _ => {}
    }
    // CAS{A,L,AL}{B,H}, W and X: 1/2/4/8-byte single-register operations.
    // CASP and the load/store-exclusive classes have different fixed bits.
    // No Rs/Rt/Rn overlap restriction applies; the emitter snapshots sources.
    if word & 0x3fa0_7c00 == 0x08a0_7c00 {
        return CompareAndSwap {
            rs: rm,
            rt: rd,
            rn,
            bytes: 1 << (word >> 30),
            acquire: word & (1 << 22) != 0,
            release: word & (1 << 15) != 0,
        };
    }
    // CASP{A,L,AL}: sz[30], L[22], Rs[20:16], o0[15], Rn[9:5], Rt[4:0].
    // Rs and Rt name even-numbered pairs; odd numbers are not decoded.
    if word & 0xbfa0_7c00 == 0x0820_7c00 && rm.is_multiple_of(2) && rd.is_multiple_of(2) {
        return CompareAndSwapPair {
            rs: rm,
            rt: rd,
            rn,
            bytes: if word & (1 << 30) != 0 { 8 } else { 4 },
            acquire: word & (1 << 22) != 0,
            release: word & (1 << 15) != 0,
        };
    }
    // Arm DDI0557A.b B11.1.5-.7: size[31:30], A[23], R[22], Rs[20:16],
    // Rn[9:5], Rt[4:0]; the six bits [15:10] are zero for ADD only.
    // Other LSE arithmetic/bitwise selectors and SWP remain unsupported.
    // https://documentation-service.arm.com/static/5fb7d32fd77dd807b9a80c61
    if word & 0x3f20_fc00 == 0x3820_0000 {
        return AtomicAdd {
            rs: rm,
            rt: rd,
            rn,
            bytes: 1 << (word >> 30),
            acquire: word & (1 << 23) != 0,
            release: word & (1 << 22) != 0,
        };
    }
    // LSE LD<op> (o3=0, opc!=000) and SWP (o3=1, opc=000):
    // size 111000 A R 1 Rs o3 opc 00 Rn Rt.
    if word & 0x3f20_0c00 == 0x3820_0000 {
        let op = match ((word >> 15) & 1, (word >> 12) & 7) {
            (0, 1) => Some(AtomicOp::Clear),
            (0, 2) => Some(AtomicOp::ExclusiveOr),
            (0, 3) => Some(AtomicOp::Set),
            (0, 4) => Some(AtomicOp::SignedMax),
            (0, 5) => Some(AtomicOp::SignedMin),
            (0, 6) => Some(AtomicOp::UnsignedMax),
            (0, 7) => Some(AtomicOp::UnsignedMin),
            (1, 0) => Some(AtomicOp::Swap),
            _ => None,
        };
        if let Some(op) = op {
            return AtomicMemory {
                op,
                rs: rm,
                rt: rd,
                rn,
                bytes: 1 << (word >> 30),
                acquire: word & (1 << 23) != 0,
                release: word & (1 << 22) != 0,
            };
        }
    }
    // Arm DDI0487 C4.1 load/store exclusive register: size[31:30] 001000
    // o2[23]=0 L[22] o1[21]=0 Rs[20:16] o0[15] Rt2[14:10]=11111 Rn Rt.
    // Pairs (o1=1) and the Rs/Rt2 should-be-one violations stay unsupported.
    if word & 0x3fff_7c00 == 0x085f_7c00 {
        return LoadExclusive {
            rt: rd,
            rn,
            bytes: 1 << (word >> 30),
            acquire: word & (1 << 15) != 0,
        };
    }
    if word & 0x3fe0_7c00 == 0x0800_7c00 {
        // Ws == Wt, or Ws == Xn with Rn != SP, is CONSTRAINED UNPREDICTABLE.
        if rm == rd || (rm == rn && rn != 31) {
            return Unsupported;
        }
        return StoreExclusive {
            rs: rm,
            rt: rd,
            rn,
            bytes: 1 << (word >> 30),
            release: word & (1 << 15) != 0,
        };
    }
    // Only the FEAT_LRCPC no-offset class. LDAPUR, post-index forms and
    // exclusive operations are separate encodings and stay unsupported.
    if word & 0x3fff_fc00 == 0x38bf_c000 {
        return LoadAcquireRcpc {
            rt: rd,
            rn,
            bytes: 1 << (word >> 30),
        };
    }
    // Canonical RCsc LDAR/B/H only. Limited-order LDLAR and exclusive loads
    // have different fixed bits and require separate semantics.
    if word & 0x3fff_fc00 == 0x08df_fc00 {
        return LoadAcquireRcsc {
            rt: rd,
            rn,
            bytes: 1 << (word >> 30),
        };
    }
    // Canonical STLR/B/H only. Limited-order STLLR and exclusive or
    // other-addressing-mode stores require separate semantics.
    if word & 0x3fff_fc00 == 0x089f_fc00 {
        return StoreRelease {
            rt: rd,
            rn,
            bytes: 1 << (word >> 30),
        };
    }
    // Q128 pairs only: separate vector and GPR banks mean that Rn == Rt is
    // valid, including writeback. Duplicate load destinations are constrained
    // unpredictable and rejected; non-temporal, S, and D forms remain unsupported.
    if word & 0xfe00_0000 == 0xac00_0000 {
        let writeback = match (word >> 23) & 3 {
            1 => Writeback::Post,
            2 => Writeback::None,
            3 => Writeback::Pre,
            // Non-temporal LDNP/STNP: the SIMD decoder handles every pair size.
            _ => return crate::simd::decode(word, pc).map_or(Unsupported, Simd),
        };
        let rt2 = ((word >> 10) & 31) as u8;
        let load = word & (1 << 22) != 0;
        if load && rd == rt2 {
            return Unsupported;
        }
        return LoadStoreVectorPair {
            rt: rd,
            rt2,
            rn,
            load,
            offset: signed((word >> 15) & 0x7f, 7) * 16,
            writeback,
        };
    }
    if word & 0x0400_0000 == 0 {
        let size = (word >> 30) as u8;
        if word & 0x3b00_0000 == 0x3900_0000 {
            if let Some(op) = memory_op(word, size) {
                return LoadStore {
                    rt: rd,
                    rn,
                    bytes: 1 << size,
                    op,
                    address: Address::Immediate {
                        offset: (((word >> 10) & 0xfff) << size) as i64,
                        writeback: Writeback::None,
                    },
                };
            }
            return Unsupported;
        }
        if word & 0x3b20_0000 == 0x3800_0000 {
            let writeback = match (word >> 10) & 3 {
                0 => Writeback::None,
                1 => Writeback::Post,
                3 => Writeback::Pre,
                _ => return Unsupported,
            };
            if writeback != Writeback::None && rn == rd && rn != 31 {
                return Unsupported;
            }
            if let Some(op) = memory_op(word, size) {
                return LoadStore {
                    rt: rd,
                    rn,
                    bytes: 1 << size,
                    op,
                    address: Address::Immediate {
                        offset: signed((word >> 12) & 0x1ff, 9),
                        writeback,
                    },
                };
            }
            return Unsupported;
        }
        if word & 0x3b20_0c00 == 0x3820_0800 {
            let option = ((word >> 13) & 7) as u8;
            if option & 2 == 0 {
                return Unsupported;
            }
            if let Some(op) = memory_op(word, size) {
                return LoadStore {
                    rt: rd,
                    rn,
                    bytes: 1 << size,
                    op,
                    address: Address::Register {
                        rm,
                        option,
                        shift: if word & (1 << 12) == 0 { 0 } else { size },
                    },
                };
            }
            return Unsupported;
        }
        if word & 0x3a00_0000 == 0x2800_0000 {
            let load = word & (1 << 22) != 0;
            let (bytes, op) = match ((word >> 30) & 3, load) {
                (0, false) => (4, MemoryOp::Store),
                (0, true) => (
                    4,
                    MemoryOp::Load {
                        signed: false,
                        bits: 32,
                    },
                ),
                // LDPSW has post-index, offset and pre-index forms, but no
                // non-temporal (mode 00) form. That encoding is reserved.
                (1, true) if (word >> 23) & 3 != 0 => (
                    4,
                    MemoryOp::Load {
                        signed: true,
                        bits: 64,
                    },
                ),
                (2, false) => (8, MemoryOp::Store),
                (2, true) => (
                    8,
                    MemoryOp::Load {
                        signed: false,
                        bits: 64,
                    },
                ),
                _ => return Unsupported,
            };
            let rt2 = ((word >> 10) & 31) as u8;
            let writeback = match (word >> 23) & 3 {
                1 => Writeback::Post,
                3 => Writeback::Pre,
                _ => Writeback::None,
            };
            if (load && rd == rt2)
                || (writeback != Writeback::None && rn != 31 && (rn == rd || rn == rt2))
            {
                return Unsupported;
            }
            return LoadStorePair {
                rt: rd,
                rt2,
                rn,
                bytes,
                op,
                offset: signed((word >> 15) & 0x7f, 7) * bytes as i64,
                writeback,
            };
        }
        if word & 0x3b00_0000 == 0x1800_0000 {
            let (bytes, signed, bits) = match word >> 30 {
                0 => (4, false, 32),
                1 => (8, false, 64),
                2 => (4, true, 64),
                _ => return Unsupported,
            };
            return LoadLiteral {
                rt: rd,
                bytes,
                signed,
                bits,
                address: target(pc, (word >> 5) & 0x7ffff, 19),
            };
        }
    }
    match word & 0xffff_ffe0 {
        // Arm Cortex-A78AE TRM 101779_0002_07_en, Table 3-3: architectural
        // S3_3_C14_C0_1, width 64. This identifies the register encoding only,
        // not guest hardware or clock frequency; neighboring timers stay out.
        0xd53b_e020 => ReadPhysicalCounter { rt: rd },
        0xd53b_4200 => ReadNzcv { rt: rd },
        0xd51b_4200 => WriteNzcv { rt: rd },
        0xd53b_d040 => ReadThreadPointer {
            rt: rd,
            read_only: false,
        },
        0xd53b_d060 => ReadThreadPointer {
            rt: rd,
            read_only: true,
        },
        0xd51b_d040 => WriteThreadPointer { rt: rd },
        // FPCR is S3_3_C4_C4_0 and FPSR is S3_3_C4_C4_1.
        0xd53b_4400 => ReadFpControl {
            rt: rd,
            status: false,
        },
        0xd53b_4420 => ReadFpControl {
            rt: rd,
            status: true,
        },
        0xd51b_4400 => WriteFpControl {
            rt: rd,
            status: false,
        },
        0xd51b_4420 => WriteFpControl {
            rt: rd,
            status: true,
        },
        _ => crate::simd::decode(word, pc).map_or(Unsupported, Simd),
    }
}

impl Instruction {
    pub fn successors(&self, pc: u64) -> Vec<u64> {
        let next = pc.wrapping_add(4);
        match self {
            Self::Branch {
                target,
                link: false,
            } => vec![*target],
            Self::Branch { target, link: true }
            | Self::BranchCond { target, .. }
            | Self::CompareBranch { target, .. }
            | Self::TestBranch { target, .. } => vec![*target, next],
            Self::BranchReg { link: false, .. }
            | Self::PAuthDisabledReturn { .. }
            | Self::Undefined
            | Self::Unsupported => vec![],
            _ => vec![next],
        }
    }

    pub fn ends_block(&self) -> bool {
        matches!(
            self,
            Self::Branch { .. }
                | Self::BranchCond { .. }
                | Self::CompareBranch { .. }
                | Self::TestBranch { .. }
                | Self::BranchReg { .. }
                | Self::PAuthDisabledReturn { .. }
                | Self::Svc { .. }
                | Self::Undefined
                | Self::Unsupported
        )
    }
}

pub fn disassemble(word: u32, _pc: u64) -> String {
    let bytes = word.to_le_bytes();
    let mut reader = U8Reader::new(&bytes);
    match yaxpeax_arm::armv8::a64::InstDecoder::default().decode(&mut reader) {
        Ok(instruction) => instruction.to_string(),
        Err(_) => format!(".word 0x{word:08x}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_counter_read_uses_only_the_exact_cntpct_encoding() {
        for rt in 0..32u8 {
            let instruction = decode(0xd53b_e020 | u32::from(rt), 0x4000);
            assert_eq!(instruction, Instruction::ReadPhysicalCounter { rt });
            assert_eq!(instruction.successors(0x4000), vec![0x4004]);
            assert!(!instruction.ends_block());
        }
        // Frequency, virtual/self-synchronized counters, timer control,
        // different privilege selector and the write encoding are not modeled.
        for word in [
            0xd53b_e000,
            0xd53b_e040,
            0xd53b_e0a0,
            0xd53b_e0c0,
            0xd53b_e220,
            0xd538_e020,
            0xd51b_e020,
        ] {
            assert_eq!(decode(word, 0x4000), Instruction::Unsupported, "{word:#x}");
        }
    }

    #[test]
    fn casp_decodes_pairs_and_matches_the_disassembler() {
        for (word, text, rs, rt, bytes, acquire, release) in [
            (
                0x4822_7c24,
                "casp x2, x3, x4, x5, [x1]",
                2,
                4,
                8,
                false,
                false,
            ),
            (
                0x4862_fc24,
                "caspal x2, x3, x4, x5, [x1]",
                2,
                4,
                8,
                true,
                true,
            ),
            (
                0x0822_7c24,
                "casp w2, w3, w4, w5, [x1]",
                2,
                4,
                4,
                false,
                false,
            ),
            (
                0x4862_7c22,
                "caspa x2, x3, x2, x3, [x1]",
                2,
                2,
                8,
                true,
                false,
            ),
            (
                0x4822_fc24,
                "caspl x2, x3, x4, x5, [x1]",
                2,
                4,
                8,
                false,
                true,
            ),
        ] {
            assert_eq!(disassemble(word, 0), text);
            assert_eq!(
                decode(word, 0),
                Instruction::CompareAndSwapPair {
                    rs,
                    rt,
                    rn: 1,
                    bytes,
                    acquire,
                    release
                }
            );
        }
        // Odd register numbers do not name a pair.
        assert!(!matches!(
            decode(0x4823_7c24, 0),
            Instruction::CompareAndSwapPair { .. }
        ));
        assert!(!matches!(
            decode(0x4822_7c25, 0),
            Instruction::CompareAndSwapPair { .. }
        ));
    }

    #[test]
    fn cas_decodes_all_element_widths_orderings_and_register_fields() {
        for size in 0..4 {
            for acquire in [false, true] {
                for release in [false, true] {
                    let word = 0x08a1_7c02
                        | (size << 30)
                        | (u32::from(acquire) << 22)
                        | (u32::from(release) << 15);
                    let instruction = decode(word, 0x1000);
                    assert_eq!(
                        instruction,
                        Instruction::CompareAndSwap {
                            rs: 1,
                            rt: 2,
                            rn: 0,
                            bytes: 1 << size,
                            acquire,
                            release,
                        },
                        "{word:#x}"
                    );
                    assert_eq!(instruction.successors(0x1000), vec![0x1004]);
                    assert!(!instruction.ends_block());
                }
            }
        }
        assert_eq!(
            decode(0x88e8_7c14, 0),
            Instruction::CompareAndSwap {
                rs: 8,
                rt: 20,
                rn: 0,
                bytes: 4,
                acquire: true,
                release: false,
            }
        );
        assert_eq!(
            decode(0xc8ff_ffff, 0),
            Instruction::CompareAndSwap {
                rs: 31,
                rt: 31,
                rn: 31,
                bytes: 8,
                acquire: true,
                release: true,
            }
        );
        // CASP is a separate pair operation, not CAS.
        assert!(matches!(
            decode(0x4820_7c82, 0),
            Instruction::CompareAndSwapPair { .. }
        ));
        // Exclusive pairs (LDXP/STXP) need their own semantics.
        for word in [0x887f_0801, 0x8823_0801] {
            assert_eq!(decode(word, 0), Instruction::Unsupported, "{word:#x}");
        }
        // LDCLR and SWP are their own atomic operations, not CAS.
        assert!(matches!(
            decode(0xb820_1001, 0),
            Instruction::AtomicMemory {
                op: AtomicOp::Clear,
                ..
            }
        ));
        assert!(matches!(
            decode(0xb820_8001, 0),
            Instruction::AtomicMemory {
                op: AtomicOp::Swap,
                ..
            }
        ));
    }

    #[test]
    fn ldadd_decodes_all_widths_orderings_and_aliases_without_other_lse_selectors() {
        for size in 0..4 {
            for acquire in [false, true] {
                for release in [false, true] {
                    let word = 0x3821_0002
                        | (size << 30)
                        | (u32::from(acquire) << 23)
                        | (u32::from(release) << 22);
                    let instruction = decode(word, 0x1000);
                    assert_eq!(
                        instruction,
                        Instruction::AtomicAdd {
                            rs: 1,
                            rt: 2,
                            rn: 0,
                            bytes: 1 << size,
                            acquire,
                            release,
                        },
                        "{word:#x}"
                    );
                    assert_eq!(instruction.successors(0x1000), vec![0x1004]);
                    assert!(!instruction.ends_block());
                }
            }
        }
        for (word, rs, rt, rn, bytes, acquire, release) in [
            (0xb8e8_0008, 8, 8, 0, 4, true, true),
            (0xf821_0000, 1, 0, 0, 8, false, false),
            (0xf820_0002, 0, 2, 0, 8, false, false),
            (0xf8ff_03ff, 31, 31, 31, 8, true, true),
            (0xf861_03ff, 1, 31, 31, 8, false, true),
        ] {
            assert_eq!(
                decode(word, 0),
                Instruction::AtomicAdd {
                    rs,
                    rt,
                    rn,
                    bytes,
                    acquire,
                    release,
                }
            );
        }
        // LDCLR, LDEOR, LDSET, signed/unsigned min/max and SWP are not ADD.
        let ops = [
            AtomicOp::Clear,
            AtomicOp::ExclusiveOr,
            AtomicOp::Set,
            AtomicOp::SignedMax,
            AtomicOp::SignedMin,
            AtomicOp::UnsignedMax,
            AtomicOp::UnsignedMin,
            AtomicOp::Swap,
        ];
        for (selector, op) in (1..=8).zip(ops) {
            let word = 0xb8e1_0002 | (selector << 12);
            assert_eq!(
                decode(word, 0),
                Instruction::AtomicMemory {
                    op,
                    rs: 1,
                    rt: 2,
                    rn: 0,
                    bytes: 4,
                    acquire: true,
                    release: true,
                },
                "{word:#x}"
            );
        }
    }

    #[test]
    fn ldapr_decodes_only_canonical_rcpc_loads_with_all_widths_and_aliases() {
        for size in 0..4 {
            let instruction = decode(0x38bf_c002 | (size << 30), 0x1000);
            assert_eq!(
                instruction,
                Instruction::LoadAcquireRcpc {
                    rt: 2,
                    rn: 0,
                    bytes: 1 << size,
                }
            );
            assert_eq!(instruction.successors(0x1000), vec![0x1004]);
            assert!(!instruction.ends_block());
        }
        assert_eq!(
            decode(0x38bf_c108, 0),
            Instruction::LoadAcquireRcpc {
                rt: 8,
                rn: 8,
                bytes: 1,
            }
        );
        assert_eq!(
            decode(0xf8bf_c3ff, 0),
            Instruction::LoadAcquireRcpc {
                rt: 31,
                rn: 31,
                bytes: 8,
            }
        );
        // LDLAR, STLLR, LDAPUR and a neighboring fixed-bit encoding.
        for word in [0xc8df_7c20, 0xc89f_7c20, 0x9940_0020, 0x38bf_c400] {
            assert_eq!(decode(word, 0), Instruction::Unsupported, "{word:#x}");
        }
    }

    #[test]
    fn exclusive_register_forms_decode_all_widths_orderings_and_register_roles() {
        let names = ["b", "h", "", ""];
        for size in 0..4u32 {
            let bytes = 1 << size;
            for (acquire, name) in [(false, "ldxr"), (true, "ldaxr")] {
                let word = 0x085f_7c25 | (size << 30) | (u32::from(acquire) << 15);
                let instruction = decode(word, 0x1000);
                assert_eq!(
                    instruction,
                    Instruction::LoadExclusive {
                        rt: 5,
                        rn: 1,
                        bytes,
                        acquire,
                    }
                );
                assert_eq!(instruction.successors(0x1000), vec![0x1004]);
                assert!(!instruction.ends_block());
                // Independent cross-check against the yaxpeax-arm decoder.
                let text = disassemble(word, 0);
                assert!(
                    text.starts_with(&format!("{name}{} ", names[size as usize])),
                    "{text}"
                );
            }
            for (release, name) in [(false, "stxr"), (true, "stlxr")] {
                let word = 0x0803_7c25 | (size << 30) | (u32::from(release) << 15);
                assert_eq!(
                    decode(word, 0),
                    Instruction::StoreExclusive {
                        rs: 3,
                        rt: 5,
                        rn: 1,
                        bytes,
                        release,
                    }
                );
                let text = disassemble(word, 0);
                assert!(
                    text.starts_with(&format!("{name}{} ", names[size as usize])),
                    "{text}"
                );
            }
        }
        // Rt == Rn, SP bases and zero-register data/status are all valid.
        assert_eq!(
            decode(0xc85f_7c21, 0),
            Instruction::LoadExclusive {
                rt: 1,
                rn: 1,
                bytes: 8,
                acquire: false
            }
        );
        assert_eq!(
            decode(0x881f_ffe2, 0),
            Instruction::StoreExclusive {
                rs: 31,
                rt: 2,
                rn: 31,
                bytes: 4,
                release: true
            }
        );
        assert_eq!(
            decode(0xc800_7fe1, 0),
            Instruction::StoreExclusive {
                rs: 0,
                rt: 1,
                rn: 31,
                bytes: 8,
                release: false
            }
        );
        // Ws == Wt and Ws == Xn (non-SP) are CONSTRAINED UNPREDICTABLE;
        // Rt2/Rs should-be-one violations and exclusive pairs stay unsupported.
        for word in [
            0xc805_7c25,
            0xc801_7c25,
            0x881f_ffff,
            0xc85f_7825,
            0xc85e_7c25,
            0xc803_7825,
            0xc87f_0c25,
            0xc823_0c25,
        ] {
            assert_eq!(decode(word, 0), Instruction::Unsupported, "{word:#x}");
        }
        for word in [0xd503_305f, 0xd503_3f5f, 0xd503_355f] {
            assert_eq!(decode(word, 0), Instruction::ClearExclusive, "{word:#x}");
            assert!(disassemble(word, 0).starts_with("clrex"), "{word:#x}");
        }
    }

    #[test]
    fn ldar_decodes_only_canonical_rcsc_loads_with_all_widths_and_register_roles() {
        for size in 0..4 {
            let instruction = decode(0x08df_fc25 | (size << 30), 0x1000);
            assert_eq!(
                instruction,
                Instruction::LoadAcquireRcsc {
                    rt: 5,
                    rn: 1,
                    bytes: 1 << size,
                }
            );
            assert_eq!(instruction.successors(0x1000), vec![0x1004]);
            assert!(!instruction.ends_block());
        }
        for (word, rt, rn, bytes) in [
            (0xc8df_fd00, 0, 8, 8),
            (0x08df_fc63, 3, 3, 1),
            (0x88df_fc3f, 31, 1, 4),
            (0xc8df_ffe5, 5, 31, 8),
        ] {
            assert_eq!(
                decode(word, 0),
                Instruction::LoadAcquireRcsc { rt, rn, bytes }
            );
        }
        // LDLAR and a neighboring reserved fixed-bit encoding.
        for word in [0xc8df_7c25, 0xc8df_f825] {
            assert_eq!(decode(word, 0), Instruction::Unsupported, "{word:#x}");
        }
        assert!(matches!(
            decode(0xf8bf_c025, 0),
            Instruction::LoadAcquireRcpc { .. }
        ));
    }

    #[test]
    fn stlr_decodes_only_canonical_release_stores_with_all_widths_and_register_roles() {
        for size in 0..4 {
            let instruction = decode(0x089f_fc02 | (size << 30), 0x1000);
            assert_eq!(
                instruction,
                Instruction::StoreRelease {
                    rt: 2,
                    rn: 0,
                    bytes: 1 << size,
                }
            );
            assert_eq!(instruction.successors(0x1000), vec![0x1004]);
            assert!(!instruction.ends_block());
        }
        assert_eq!(
            decode(0x089f_fc15, 0),
            Instruction::StoreRelease {
                rt: 21,
                rn: 0,
                bytes: 1,
            }
        );
        for (word, rt, rn) in [(0xc89f_fc00, 0, 0), (0xc89f_ffff, 31, 31)] {
            assert_eq!(
                decode(word, 0),
                Instruction::StoreRelease { rt, rn, bytes: 8 }
            );
        }
        // LDLAR, STLLR and unscaled-immediate STLUR stay unsupported.
        for word in [0xc8df_7c20, 0xc89f_7c20, 0xd900_0020] {
            assert_eq!(decode(word, 0), Instruction::Unsupported, "{word:#x}");
        }
    }

    #[test]
    fn q_pair_decoder_distinguishes_vector_bank_scaling_and_writeback() {
        assert_eq!(
            decode(0xadbf_1fe6, 0),
            Instruction::LoadStoreVectorPair {
                rt: 6,
                rt2: 7,
                rn: 31,
                load: false,
                offset: -32,
                writeback: Writeback::Pre,
            }
        );
        assert_eq!(
            decode(0xacc1_07e0, 0),
            Instruction::LoadStoreVectorPair {
                rt: 0,
                rt2: 1,
                rn: 31,
                load: true,
                offset: 32,
                writeback: Writeback::Post,
            }
        );
        assert_eq!(
            decode(0xad01_07e0, 0),
            Instruction::LoadStoreVectorPair {
                rt: 0,
                rt2: 1,
                rn: 31,
                load: false,
                offset: 32,
                writeback: Writeback::None,
            }
        );
        assert_eq!(
            decode(0xacc1_0400, 0),
            Instruction::LoadStoreVectorPair {
                rt: 0,
                rt2: 1,
                rn: 0,
                load: true,
                offset: 32,
                writeback: Writeback::Post,
            }
        );
        assert_eq!(
            decode(0xacc1_03ff, 0),
            Instruction::LoadStoreVectorPair {
                rt: 31,
                rt2: 0,
                rn: 31,
                load: true,
                offset: 32,
                writeback: Writeback::Post,
            }
        );
        // Duplicate load destinations and the reserved size stay unsupported.
        for word in [0xacc1_03e0, 0xedbf_07e0] {
            assert_eq!(decode(word, 0), Instruction::Unsupported, "{word:#x}");
        }
        // Non-temporal Q pairs and S/D pairs are decoded by the SIMD decoder.
        for word in [0xac01_07e0, 0x6dbf_07e0, 0x2dbf_07e0] {
            assert!(
                matches!(
                    decode(word, 0),
                    Instruction::Simd(crate::simd::SimdOp::LoadStorePair { .. })
                ),
                "{word:#x}"
            );
        }
    }

    #[test]
    fn xpaclri_is_distinct_from_disabled_authentication_hints() {
        let instruction = decode(0xd503_20ff, 0x1000);
        assert_eq!(instruction, Instruction::StripLrInstructionPac);
        assert_eq!(instruction.successors(0x1000), vec![0x1004]);
        assert!(!instruction.ends_block());
    }

    #[test]
    fn unsigned_multiply_add_long_decodes_distinct_w_inputs_and_x_accumulator() {
        assert_eq!(
            decode(0x9bb5_2508, 0),
            Instruction::UnsignedMultiplyAddLong {
                rd: 8,
                rn: 8,
                rm: 21,
                ra: 9
            }
        );
        assert_eq!(
            decode(0x9ba8_6037, 0),
            Instruction::UnsignedMultiplyAddLong {
                rd: 23,
                rn: 1,
                rm: 8,
                ra: 24
            }
        );
        assert_eq!(
            decode(0x9ba2_7c20, 0),
            Instruction::UnsignedMultiplyAddLong {
                rd: 0,
                rn: 1,
                rm: 2,
                ra: 31
            }
        );
        // UMSUBL and SMADDL are distinct long forms; sf=0 is not a long form.
        assert_eq!(
            decode(0x9bb5_a508, 0),
            Instruction::MultiplyAddLong {
                rd: 8,
                rn: 8,
                rm: 21,
                ra: 9,
                signed: false,
                subtract: true,
            }
        );
        assert_eq!(
            decode(0x9b35_2508, 0),
            Instruction::MultiplyAddLong {
                rd: 8,
                rn: 8,
                rm: 21,
                ra: 9,
                signed: true,
                subtract: false,
            }
        );
        assert_eq!(decode(0x1bb5_2508, 0), Instruction::Unsupported);
    }

    #[test]
    fn conditional_compare_decodes_register_immediate_width_and_literal_flags() {
        assert_eq!(
            decode(0x7a48_1924, 0),
            Instruction::ConditionalCompare {
                rn: 9,
                bits: 32,
                rhs: CompareOperand::Immediate(8),
                condition: 1,
                nzcv: 4,
            }
        );
        assert_eq!(
            decode(0x7a58_1104, 0),
            Instruction::ConditionalCompare {
                rn: 8,
                bits: 32,
                rhs: CompareOperand::Register(24),
                condition: 1,
                nzcv: 4,
            }
        );
        assert_eq!(
            decode(0xfa49_1100, 0),
            Instruction::ConditionalCompare {
                rn: 8,
                bits: 64,
                rhs: CompareOperand::Register(9),
                condition: 1,
                nzcv: 0,
            }
        );
        assert_eq!(
            decode(0xfa5f_ebef, 0),
            Instruction::ConditionalCompare {
                rn: 31,
                bits: 64,
                rhs: CompareOperand::Immediate(31),
                condition: 14,
                nzcv: 15,
            }
        );
        // CCMN is its own operation; reserved o2/o3/S encodings stay unsupported.
        assert_eq!(
            decode(0x3a48_1924, 0),
            Instruction::ConditionalCompareNegative {
                rn: 9,
                bits: 32,
                rhs: CompareOperand::Immediate(8),
                condition: 1,
                nzcv: 4,
            }
        );
        for word in [0x7a48_1d24, 0x7a48_1934, 0x5a48_1924] {
            assert_eq!(decode(word, 0), Instruction::Unsupported);
        }
    }

    #[test]
    fn branch_offsets_link_and_register_targets_are_decoded_exactly() {
        assert_eq!(
            decode(0x9400_0002, 0x1000),
            Instruction::Branch {
                target: 0x1008,
                link: true
            }
        );
        assert_eq!(
            decode(0x17ff_ffff, 0x1000),
            Instruction::Branch {
                target: 0xffc,
                link: false
            }
        );
        assert_eq!(
            decode(0xd65f_03c0, 0),
            Instruction::BranchReg {
                rn: 30,
                link: false
            }
        );
        assert_eq!(
            decode(0xd63f_00a0, 0),
            Instruction::BranchReg { rn: 5, link: true }
        );
        assert_eq!(
            decode(0xb7f8_0020, 0x1000),
            Instruction::TestBranch {
                rt: 0,
                bit: 63,
                nonzero: true,
                target: 0x1004
            }
        );
    }

    #[test]
    fn immediate_masks_cover_replication_rotation_and_reserved_encoding() {
        assert_eq!(decode_bit_masks(1, 7, 0, true, 64), Some((0xff, 0xff)));
        assert_eq!(
            decode_bit_masks(0, 39, 0, true, 64).map(|x| x.0),
            Some(0x00ff_00ff_00ff_00ff)
        );
        assert_eq!(
            decode_bit_masks(1, 0, 1, true, 64).map(|x| x.0),
            Some(0x8000_0000_0000_0000)
        );
        assert!(decode_bit_masks(1, 63, 0, true, 64).is_none());
        assert!(decode_bit_masks(1, 1, 0, true, 32).is_none());
    }

    #[test]
    fn load_store_sign_extension_and_writeback_are_explicit() {
        assert_eq!(
            decode(0x3980_0020, 0),
            Instruction::LoadStore {
                rt: 0,
                rn: 1,
                bytes: 1,
                op: MemoryOp::Load {
                    signed: true,
                    bits: 64
                },
                address: Address::Immediate {
                    offset: 0,
                    writeback: Writeback::None
                }
            }
        );
        assert_eq!(
            decode(0xf81f_0fe0, 0),
            Instruction::LoadStore {
                rt: 0,
                rn: 31,
                bytes: 8,
                op: MemoryOp::Store,
                address: Address::Immediate {
                    offset: -16,
                    writeback: Writeback::Pre
                }
            }
        );
        assert_eq!(
            decode(0xf840_8420, 0),
            Instruction::LoadStore {
                rt: 0,
                rn: 1,
                bytes: 8,
                op: MemoryOp::Load {
                    signed: false,
                    bits: 64
                },
                address: Address::Immediate {
                    offset: 8,
                    writeback: Writeback::Post
                }
            }
        );
        assert_eq!(
            decode(0xa9bf_7bfd, 0),
            Instruction::LoadStorePair {
                rt: 29,
                rt2: 30,
                rn: 31,
                bytes: 8,
                op: MemoryOp::Store,
                offset: -16,
                writeback: Writeback::Pre
            }
        );
    }

    #[test]
    fn signed_pair_load_rejects_reserved_non_temporal_mode() {
        assert_eq!(decode(0x6840_0440, 0), Instruction::Unsupported);
        for (word, writeback) in [
            (0x68c0_0440, Writeback::Post),
            (0x6940_0440, Writeback::None),
            (0x69c0_0440, Writeback::Pre),
        ] {
            assert_eq!(
                decode(word, 0),
                Instruction::LoadStorePair {
                    rt: 0,
                    rt2: 1,
                    rn: 2,
                    bytes: 4,
                    op: MemoryOp::Load {
                        signed: true,
                        bits: 64,
                    },
                    offset: 0,
                    writeback,
                }
            );
        }
    }

    #[test]
    fn disabled_pauth_hints_and_returns_have_explicit_decodings_and_control_flow() {
        for (word, hint) in [
            (0xd503_233f, PAuthHint::PaciaSp),
            (0xd503_237f, PAuthHint::PacibSp),
            (0xd503_23bf, PAuthHint::AutiaSp),
            (0xd503_23ff, PAuthHint::AutibSp),
            (0xd503_211f, PAuthHint::Pacia1716),
            (0xd503_215f, PAuthHint::Pacib1716),
            (0xd503_219f, PAuthHint::Autia1716),
            (0xd503_21df, PAuthHint::Autib1716),
        ] {
            let instruction = decode(word, 0x1000);
            assert_eq!(instruction, Instruction::PAuthDisabledHint { hint });
            assert_eq!(instruction.successors(0x1000), vec![0x1004]);
            assert!(!instruction.ends_block());
        }
        for (word, key) in [
            (0xd65f_0bff, InstructionAuthKey::A),
            (0xd65f_0fff, InstructionAuthKey::B),
        ] {
            let instruction = decode(word, 0x1000);
            assert_eq!(instruction, Instruction::PAuthDisabledReturn { key });
            assert!(instruction.successors(0x1000).is_empty());
            assert!(instruction.ends_block());
        }
    }

    #[test]
    fn unsupported_atomic_and_unmodeled_pauth_do_not_become_nops() {
        assert_eq!(decode(0xc87f_0820, 0), Instruction::Unsupported);
        // Word 0 is UDF #0: an explicit undefined-instruction stop.
        assert_eq!(decode(0, 0), Instruction::Undefined);
        assert_eq!(decode(0xd503_201f, 0), Instruction::Nop);
        // PACIA1716 uses X17/X16 implicitly; with keys disabled it is the
        // explicit disabled-key hint, not a generic NOP.
        assert_eq!(
            decode(0xd503_211f, 0),
            Instruction::PAuthDisabledHint {
                hint: PAuthHint::Pacia1716
            }
        );
    }

    #[test]
    fn pointer_strip_cache_and_event_hints() {
        assert_eq!(
            decode(0x9ad1_3262, 0),
            Instruction::PacGeneric {
                rd: 2,
                rn: 19,
                rm: 17
            }
        );
        assert_eq!(decode(0xdac1_43e3, 0), Instruction::StripPac { rd: 3 });
        assert_eq!(decode(0xdac1_47e0, 0), Instruction::StripPac { rd: 0 });
        // PACIA X0, X1; AUTDB X2, SP; PACIZA X5.
        for word in [0xdac1_0020, 0xdac1_1fe2, 0xdac1_23e5] {
            assert_eq!(
                decode(word, 0),
                Instruction::PAuthDisabledData,
                "{word:08x}"
            );
        }
        // PACIZA with Rn != 31 is unallocated.
        assert_eq!(decode(0xdac1_2025, 0), Instruction::Unsupported);
        assert_eq!(decode(0xd53b_0029, 0), Instruction::ReadCacheType { rt: 9 });
        assert_eq!(decode(0xd53b_4204, 0), Instruction::ReadNzcv { rt: 4 });
        assert_eq!(decode(0xd51b_4206, 0), Instruction::WriteNzcv { rt: 6 });
        assert_eq!(disassemble(0xd51b_4206, 0), "msr s3_3_c4_c2_0, x6");
        assert_eq!(
            decode(0x9ac3_5c42, 0),
            Instruction::Crc32 {
                rd: 2,
                rn: 2,
                rm: 3,
                bytes: 8,
                castagnoli: true
            }
        );
        // sf must be set exactly for the 64-bit data form.
        assert!(!matches!(decode(0x9ac1_4000, 0), Instruction::Crc32 { .. }));
        assert!(!matches!(decode(0x1ac1_4c00, 0), Instruction::Crc32 { .. }));
        for word in [0xd50b_7e21, 0xd50b_7522, 0xd50b_7a20, 0xd50b_7b20] {
            assert_eq!(decode(word, 0), Instruction::CacheMaintenance, "{word:08x}");
        }
        for word in [0xd503_20bf, 0xd503_209f, 0xd503_207f] {
            assert_eq!(decode(word, 0), Instruction::Nop, "{word:08x}");
        }
        for word in [0xd503_203f, 0xd503_205f] {
            assert_eq!(decode(word, 0), Instruction::SpinHint, "{word:08x}");
        }
    }

    #[test]
    fn malformed_32_bit_shift_and_mov_wide_are_rejected() {
        assert_eq!(decode(0x52c0_0000, 0), Instruction::Unsupported);
        assert_eq!(decode(0x0b00_fc00, 0), Instruction::Unsupported);
    }
}
