//! Scalar floating-point and the reached subset of Advanced SIMD.
//!
//! Decoding produces a fully resolved [`SimdOp`]; translated code calls the
//! matching function here with constant operands, so no instruction word is
//! decoded at run time. Arithmetic follows the Arm A64 pseudocode for the
//! behaviours a single-PE user process can observe:
//!
//! * NaN propagation (FPProcessNaNs/FPProcessNaNs3 order), default NaN for
//!   invalid operations, FPCR.DN and FPCR.FZ (single and double precision).
//! * FPSR cumulative IOC, DZC, OFC and IDC. Inexact (IXC) and underflow (UFC)
//!   accumulation is not modeled; FPCR trap-enable bits are stored only.
//! * FPCR.RMode other than round-to-nearest-even, FPCR.AH, FPCR.FZ16 and
//!   FPCR.AHP stop explicitly instead of producing differently rounded
//!   results. Half precision is supported only by the FCVT conversions.
//! * The saturating integer instructions accumulate FPSR.QC.

use serde::Serialize;

use crate::aarch64::{Address, Writeback};
pub use echora_guest::simd::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum SimdOp {
    /// LDR/STR/LDUR/STUR (B/H/S/D/Q). Loads zero the rest of the register.
    LoadStore {
        rt: u8,
        rn: u8,
        bytes: u8,
        load: bool,
        address: Address,
    },
    /// LDP/STP/LDNP/STNP of S, D or Q registers.
    LoadStorePair {
        rt: u8,
        rt2: u8,
        rn: u8,
        bytes: u8,
        load: bool,
        offset: i64,
        writeback: Writeback,
    },
    /// LDR (literal) of S, D or Q.
    LoadLiteral { rt: u8, bytes: u8, address: u64 },
    FpUnary {
        op: FpUnaryOp,
        size: FpSize,
        rd: u8,
        rn: u8,
    },
    /// FCVT between single and double precision.
    FpConvert {
        rd: u8,
        rn: u8,
        from: FpSize,
        to: FpSize,
    },
    FpBinary {
        op: FpBinaryOp,
        size: FpSize,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    FpTernary {
        op: FpTernaryOp,
        size: FpSize,
        rd: u8,
        rn: u8,
        rm: u8,
        ra: u8,
    },
    /// FCMP/FCMPE; `rm` None compares with +0.0.
    FpCompare {
        size: FpSize,
        rn: u8,
        rm: Option<u8>,
        signaling: bool,
    },
    FpConditionalCompare {
        size: FpSize,
        rn: u8,
        rm: u8,
        condition: u8,
        nzcv: u8,
        signaling: bool,
    },
    FpSelect {
        size: FpSize,
        rd: u8,
        rn: u8,
        rm: u8,
        condition: u8,
    },
    /// FMOV (scalar, immediate) with the expanded bit pattern.
    FpMoveImmediate { size: FpSize, rd: u8, bits: u64 },
    /// FMOV Wd,Sn / Xd,Dn / Xd,Vn.D[1].
    FpMoveToGeneral {
        rd: u8,
        rn: u8,
        bits: u8,
        upper: bool,
    },
    /// FMOV Sd,Wn / Dd,Xn / Vd.D[1],Xn.
    FpMoveFromGeneral {
        rd: u8,
        rn: u8,
        bits: u8,
        upper: bool,
    },
    /// SCVTF/UCVTF from a general register.
    IntToFp {
        rd: u8,
        rn: u8,
        gp_bits: u8,
        size: FpSize,
        signed: bool,
    },
    /// SCVTF/UCVTF (scalar, integer in a SIMD register of the same size).
    SimdIntToFp {
        rd: u8,
        rn: u8,
        size: FpSize,
        signed: bool,
    },
    /// FCVT{N,P,M,Z,A}{S,U} to a general register.
    FpToInt {
        rd: u8,
        rn: u8,
        gp_bits: u8,
        size: FpSize,
        signed: bool,
        rounding: Rounding,
    },
    /// MOVI/MVNI/FMOV (vector, immediate): the complete register result.
    MoveImmediate { rd: u8, value: u128 },
    /// ORR/BIC (vector, immediate) applied to the existing register.
    ModifyImmediate {
        rd: u8,
        value: u128,
        clear: bool,
        q: bool,
    },
    VectorLogical {
        op: VectorLogicalOp,
        rd: u8,
        rn: u8,
        rm: u8,
        q: bool,
    },
    /// INS (element): Vd.T[dst] = Vn.T[src], other elements preserved.
    InsertElement {
        rd: u8,
        rn: u8,
        bytes: u8,
        dst: u8,
        src: u8,
    },
    /// INS (general): Vd.T[index] = Rn.
    InsertGeneral {
        rd: u8,
        rn: u8,
        bytes: u8,
        index: u8,
    },
    /// UMOV/SMOV: Rd = Vn.T[index], zero- or sign-extended.
    MoveToGeneral {
        rd: u8,
        rn: u8,
        bytes: u8,
        index: u8,
        signed: bool,
        gp_bits: u8,
    },
    /// DUP (element): every lane (or the scalar result) = Vn.T[index].
    DupElement {
        rd: u8,
        rn: u8,
        bytes: u8,
        index: u8,
        q: bool,
        scalar: bool,
    },
    /// DUP (general): every lane = low bits of Rn.
    DupGeneral { rd: u8, rn: u8, bytes: u8, q: bool },
    VecFp {
        op: VecFpOp,
        size: FpSize,
        q: bool,
        scalar: bool,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// FADDP/FMAXP/FMINP/FMAXNMP/FMINNMP: adjacent pairs of the
    /// concatenation Vm:Vn (vector), or the two lanes of Vn (scalar).
    VecFpPairwise {
        op: VecFpOp,
        size: FpSize,
        q: bool,
        scalar: bool,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    VecIntPairwise {
        op: VecIntOp,
        bits: u8,
        q: bool,
        scalar: bool,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    VecIntElement {
        op: VecIntOp,
        bits: u8,
        q: bool,
        scalar: bool,
        rd: u8,
        rn: u8,
        rm: u8,
        index: u8,
    },
    VecLongElement {
        op: VecLongOp,
        signed: bool,
        bits: u8,
        upper: bool,
        rd: u8,
        rn: u8,
        rm: u8,
        index: u8,
    },
    /// XTN/SQXTN/UQXTN/SQXTUN and the (rounding) shift-right-narrow family;
    /// `bits` is the source lane width.
    VecNarrow {
        op: VecNarrowOp,
        round: bool,
        bits: u8,
        upper: bool,
        scalar: bool,
        rd: u8,
        rn: u8,
        shift: u8,
    },
    /// PMULL/PMULL2 on 8-bit or 64-bit elements.
    VecPolyMulLong {
        bits: u8,
        upper: bool,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// AES, SHA-1 and SHA-256.
    Crypto {
        op: CryptoOp,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// TBL/TBX over `regs` table registers starting at Vn.
    VecTable {
        q: bool,
        extend: bool,
        rd: u8,
        rn: u8,
        regs: u8,
        rm: u8,
    },
    /// FCVTL/FCVTL2/FCVTN/FCVTN2.
    VecFpConvert {
        from: FpPrecision,
        to: FpPrecision,
        upper: bool,
        rd: u8,
        rn: u8,
    },
    /// Scalar FCVT to or from half precision.
    FpConvertPrecision {
        rd: u8,
        rn: u8,
        from: FpPrecision,
        to: FpPrecision,
    },
    VecFpElement {
        op: VecFpOp,
        size: FpSize,
        q: bool,
        scalar: bool,
        rd: u8,
        rn: u8,
        rm: u8,
        index: u8,
    },
    VecInt {
        op: VecIntOp,
        bits: u8,
        q: bool,
        scalar: bool,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    VecUnary {
        op: VecUnaryOp,
        bits: u8,
        q: bool,
        scalar: bool,
        rd: u8,
        rn: u8,
    },
    VecFpUnary {
        op: VecFpUnaryOp,
        size: FpSize,
        q: bool,
        scalar: bool,
        rd: u8,
        rn: u8,
        fbits: u8,
    },
    /// FCVTZS/FCVTZU (scalar, fixed-point) to a general register.
    FpToFixed {
        rd: u8,
        rn: u8,
        gp_bits: u8,
        size: FpSize,
        signed: bool,
        fbits: u8,
    },
    /// SCVTF/UCVTF (scalar, fixed-point) from a general register.
    IntToFpFixed {
        rd: u8,
        rn: u8,
        gp_bits: u8,
        size: FpSize,
        signed: bool,
        fbits: u8,
    },
    VecAcross {
        op: VecAcrossOp,
        bits: u8,
        q: bool,
        rd: u8,
        rn: u8,
    },
    VecShift {
        op: VecShiftOp,
        bits: u8,
        q: bool,
        scalar: bool,
        rd: u8,
        rn: u8,
        shift: u8,
    },
    VecShiftLong {
        signed: bool,
        bits: u8,
        upper: bool,
        rd: u8,
        rn: u8,
        shift: u8,
    },
    VecLong {
        op: VecLongOp,
        signed: bool,
        bits: u8,
        upper: bool,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    VecPairwiseLong {
        signed: bool,
        accumulate: bool,
        bits: u8,
        q: bool,
        rd: u8,
        rn: u8,
    },
    VecPermute {
        op: VecPermuteOp,
        bits: u8,
        q: bool,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    VecExt {
        q: bool,
        rd: u8,
        rn: u8,
        rm: u8,
        index: u8,
    },
    /// LD1-LD4/ST1-ST4 (multiple structures); `post` is the post-index
    /// register (31 = immediate, the transfer size).
    TransferMultiple {
        rt: u8,
        rn: u8,
        registers: u8,
        structure: u8,
        bits: u8,
        q: bool,
        load: bool,
        post: Option<u8>,
    },
    TransferSingle {
        rt: u8,
        rn: u8,
        registers: u8,
        bits: u8,
        index: u8,
        load: bool,
        post: Option<u8>,
    },
    LoadReplicate {
        rt: u8,
        rn: u8,
        registers: u8,
        bits: u8,
        q: bool,
        post: Option<u8>,
    },
}

fn signed_field(value: u32, bits: u8) -> i64 {
    ((value as i64) << (64 - bits)) >> (64 - bits)
}

fn writeback_mode(bits: u32) -> Option<Writeback> {
    match bits {
        0 => Some(Writeback::None),
        1 => Some(Writeback::Post),
        3 => Some(Writeback::Pre),
        _ => None,
    }
}

/// VFPExpandImm for single (32) and double (64) precision.
pub fn expand_fp_immediate(imm8: u8, size: FpSize) -> u64 {
    let sign = u64::from(imm8 >> 7);
    let b6 = u64::from((imm8 >> 6) & 1);
    let low = u64::from(imm8 & 0x3f);
    match size {
        FpSize::S => {
            let exponent_top = (1 - b6) << 7 | (if b6 == 1 { 0x1f } else { 0 }) << 2;
            (sign << 31) | (exponent_top << 23) | (low << 19)
        }
        FpSize::D => {
            let exponent_top = (1 - b6) << 10 | (if b6 == 1 { 0xff } else { 0 }) << 2;
            (sign << 63) | (exponent_top << 52) | (low << 48)
        }
    }
}

/// AdvSIMDExpandImm for the MOVI/MVNI/ORR/BIC/FMOV (vector) class.
fn expand_simd_immediate(op: bool, cmode: u32, imm8: u8) -> Option<u64> {
    let imm = u64::from(imm8);
    let replicate32 = |value: u64| value | (value << 32);
    let replicate16 = |value: u64| value * 0x0001_0001_0001_0001;
    Some(match cmode >> 1 {
        0 => replicate32(imm),
        1 => replicate32(imm << 8),
        2 => replicate32(imm << 16),
        3 => replicate32(imm << 24),
        4 => replicate16(imm),
        5 => replicate16(imm << 8),
        6 => {
            if cmode & 1 == 0 {
                replicate32((imm << 8) | 0xff)
            } else {
                replicate32((imm << 16) | 0xffff)
            }
        }
        _ => {
            if cmode & 1 == 0 {
                if op {
                    let mut value = 0;
                    for bit in 0..8 {
                        if imm8 & (1 << bit) != 0 {
                            value |= 0xff << (bit * 8);
                        }
                    }
                    value
                } else {
                    imm * 0x0101_0101_0101_0101
                }
            } else if op {
                expand_fp_immediate(imm8, FpSize::D)
            } else {
                replicate32(expand_fp_immediate(imm8, FpSize::S))
            }
        }
    })
}

fn fp_size(ftype: u32) -> Option<FpSize> {
    match ftype {
        0 => Some(FpSize::S),
        1 => Some(FpSize::D),
        _ => None,
    }
}

/// Decode SIMD&FP load/store and data-processing encodings. `None` means the
/// word is outside these classes or uses a form not implemented here.
pub fn decode(word: u32, pc: u64) -> Option<SimdOp> {
    let rd = (word & 31) as u8;
    let rn = ((word >> 5) & 31) as u8;
    let rm = ((word >> 16) & 31) as u8;
    let op0 = (word >> 25) & 15;
    if op0 & 0b0101 == 0b0100 && word & (1 << 26) != 0 {
        return decode_memory(word, pc, rd, rn, rm);
    }
    if op0 & 0b0111 != 0b0111 {
        return None;
    }
    // Conversion between floating-point and fixed-point (bit 21 clear).
    if word & 0x7f20_0000 == 0x1e00_0000 {
        let size = fp_size((word >> 22) & 3)?;
        let sf = word >> 31;
        let scale = (word >> 10) & 63;
        if sf == 0 && scale < 32 {
            return None;
        }
        let fbits = (64 - scale) as u8;
        let gp_bits = if sf == 1 { 64 } else { 32 };
        let signed = (word >> 16) & 1 == 0;
        return match ((word >> 19) & 3, (word >> 16) & 7) {
            (3, 0) | (3, 1) => Some(SimdOp::FpToFixed {
                rd,
                rn,
                gp_bits,
                size,
                signed,
                fbits,
            }),
            (0, 2) | (0, 3) => Some(SimdOp::IntToFpFixed {
                rd,
                rn,
                gp_bits,
                size,
                signed,
                fbits,
            }),
            _ => None,
        };
    }
    // Scalar floating-point (M=0, S=0, bit 24 selects the 3-source class).
    if word & 0x5f00_0000 == 0x1e00_0000 && word & (1 << 21) != 0 {
        return decode_fp_scalar(word, rd, rn, rm);
    }
    if word & 0xff00_0000 == 0x1f00_0000 {
        let size = fp_size((word >> 22) & 3)?;
        let op = match ((word >> 21) & 1, (word >> 15) & 1) {
            (0, 0) => FpTernaryOp::MulAdd,
            (0, _) => FpTernaryOp::MulSub,
            (_, 0) => FpTernaryOp::NegMulAdd,
            _ => FpTernaryOp::NegMulSub,
        };
        return Some(SimdOp::FpTernary {
            op,
            size,
            rd,
            rn,
            rm,
            ra: ((word >> 10) & 31) as u8,
        });
    }
    // SCVTF/UCVTF (vector scalar, integer).
    if word & 0xdfbf_fc00 == 0x5e21_d800 {
        return Some(SimdOp::SimdIntToFp {
            rd,
            rn,
            size: if word & (1 << 22) == 0 {
                FpSize::S
            } else {
                FpSize::D
            },
            signed: word & (1 << 29) == 0,
        });
    }
    let q = word & (1 << 30) != 0;
    // Advanced SIMD modified immediate.
    if word & 0x9ff8_0400 == 0x0f00_0400 {
        if word & (1 << 11) != 0 {
            return None; // FMOV (vector, half precision)
        }
        let op = word & (1 << 29) != 0;
        let cmode = (word >> 12) & 15;
        let imm8 = ((((word >> 16) & 7) << 5) | ((word >> 5) & 31)) as u8;
        if cmode == 15 && op && !q {
            return None;
        }
        let value = expand_simd_immediate(op, cmode, imm8)?;
        let modify = cmode & 1 == 1 && cmode < 12;
        let value = if !modify && op && cmode < 14 {
            !value
        } else {
            value
        };
        let full = if q {
            (u128::from(value) << 64) | u128::from(value)
        } else {
            u128::from(value)
        };
        return Some(if modify {
            SimdOp::ModifyImmediate {
                rd,
                value: full,
                clear: op,
                q,
            }
        } else {
            SimdOp::MoveImmediate { rd, value: full }
        });
    }
    // Three-same logical.
    if word & 0x9f20_fc00 == 0x0e20_1c00 {
        let op = match ((word >> 29) & 1, (word >> 22) & 3) {
            (0, 0) => VectorLogicalOp::And,
            (0, 1) => VectorLogicalOp::Bic,
            (0, 2) => VectorLogicalOp::Orr,
            (0, _) => VectorLogicalOp::Orn,
            (_, 0) => VectorLogicalOp::Eor,
            (_, 1) => VectorLogicalOp::Bsl,
            (_, 2) => VectorLogicalOp::Bit,
            _ => VectorLogicalOp::Bif,
        };
        return Some(SimdOp::VectorLogical { op, rd, rn, rm, q });
    }
    // Copy class: imm5 selects the element size and index.
    if word & 0x9fe0_8400 == 0x0e00_0400 {
        let imm5 = (word >> 16) & 31;
        if imm5 & 15 == 0 {
            return None;
        }
        let shift = imm5.trailing_zeros();
        let bytes = 1u8 << shift;
        let index = (imm5 >> (shift + 1)) as u8;
        let imm4 = (word >> 11) & 15;
        let op = (word >> 29) & 1;
        return match (op, imm4) {
            (1, _) if q => Some(SimdOp::InsertElement {
                rd,
                rn,
                bytes,
                dst: index,
                src: (imm4 >> shift) as u8,
            }),
            (0, 0) if bytes < 8 || q => Some(SimdOp::DupElement {
                rd,
                rn,
                bytes,
                index,
                q,
                scalar: false,
            }),
            (0, 1) if bytes < 8 || q => Some(SimdOp::DupGeneral { rd, rn, bytes, q }),
            (0, 3) if q => Some(SimdOp::InsertGeneral {
                rd,
                rn,
                bytes,
                index,
            }),
            (0, 7) if (bytes == 8) == q && bytes >= 4 => Some(SimdOp::MoveToGeneral {
                rd,
                rn,
                bytes,
                index,
                signed: false,
                gp_bits: if q { 64 } else { 32 },
            }),
            (0, 7) if !q && bytes < 4 => Some(SimdOp::MoveToGeneral {
                rd,
                rn,
                bytes,
                index,
                signed: false,
                gp_bits: 32,
            }),
            (0, 5) if bytes < 4 || (bytes == 4 && q) => Some(SimdOp::MoveToGeneral {
                rd,
                rn,
                bytes,
                index,
                signed: true,
                gp_bits: if q { 64 } else { 32 },
            }),
            _ => None,
        };
    }
    // DUP (element), scalar form: also the MOV (scalar) alias.
    if word & 0xffe0_fc00 == 0x5e00_0400 {
        let imm5 = (word >> 16) & 31;
        if imm5 & 15 == 0 {
            return None;
        }
        let shift = imm5.trailing_zeros();
        return Some(SimdOp::DupElement {
            rd,
            rn,
            bytes: 1 << shift,
            index: (imm5 >> (shift + 1)) as u8,
            q: false,
            scalar: true,
        });
    }
    decode_vector(word, rd, rn, rm)
}

fn fp_size_bit(bit: u32) -> FpSize {
    if bit == 0 { FpSize::S } else { FpSize::D }
}

/// Advanced SIMD three-same, two-register misc, across-lanes, by-element,
/// shift-by-immediate, permute, EXT and three-different classes.
fn decode_vector(word: u32, rd: u8, rn: u8, rm: u8) -> Option<SimdOp> {
    // Cryptographic AES, three-register SHA and two-register SHA.
    let crypto = match word & 0xffff_fc00 {
        0x4e28_4800 => Some(CryptoOp::Aese),
        0x4e28_5800 => Some(CryptoOp::Aesd),
        0x4e28_6800 => Some(CryptoOp::Aesmc),
        0x4e28_7800 => Some(CryptoOp::Aesimc),
        0x5e28_0800 => Some(CryptoOp::Sha1h),
        0x5e28_1800 => Some(CryptoOp::Sha1su1),
        0x5e28_2800 => Some(CryptoOp::Sha256su0),
        _ => None,
    }
    .or_else(|| {
        if word & 0xffe0_8c00 != 0x5e00_0000 {
            return None;
        }
        Some(match (word >> 12) & 7 {
            0 => CryptoOp::Sha1c,
            1 => CryptoOp::Sha1p,
            2 => CryptoOp::Sha1m,
            3 => CryptoOp::Sha1su0,
            4 => CryptoOp::Sha256h,
            5 => CryptoOp::Sha256h2,
            6 => CryptoOp::Sha256su1,
            _ => return None,
        })
    });
    if let Some(op) = crypto {
        return Some(SimdOp::Crypto { op, rd, rn, rm });
    }
    let q = word & (1 << 30) != 0;
    let u = (word >> 29) & 1;
    let size = (word >> 22) & 3;
    // Three same (vector 0x0e20_0400, scalar 0x5e20_0400).
    let three_vector = word & 0x9f20_0400 == 0x0e20_0400;
    let three_scalar = word & 0xdf20_0400 == 0x5e20_0400;
    if three_vector || three_scalar {
        let scalar = three_scalar;
        let opcode = (word >> 11) & 31;
        if opcode >= 0x18 {
            let a = size >> 1;
            let fp = fp_size_bit(size & 1);
            if fp == FpSize::D && !q && !scalar {
                return None;
            }
            let pairwise = match (u, a, opcode) {
                (1, 0, 0x18) => Some(VecFpOp::MaxNum),
                (1, 1, 0x18) => Some(VecFpOp::MinNum),
                (1, 0, 0x1a) => Some(VecFpOp::Add),
                (1, 0, 0x1e) => Some(VecFpOp::Max),
                (1, 1, 0x1e) => Some(VecFpOp::Min),
                _ => None,
            };
            if let Some(op) = pairwise {
                if scalar {
                    return None;
                }
                return Some(SimdOp::VecFpPairwise {
                    op,
                    size: fp,
                    q,
                    scalar: false,
                    rd,
                    rn,
                    rm,
                });
            }
            let op = match (u, a, opcode) {
                (0, 0, 0x18) => VecFpOp::MaxNum,
                (0, 1, 0x18) => VecFpOp::MinNum,
                (0, 0, 0x19) => VecFpOp::MulAdd,
                (0, 1, 0x19) => VecFpOp::MulSub,
                (0, 0, 0x1a) => VecFpOp::Add,
                (0, 1, 0x1a) => VecFpOp::Sub,
                (1, 1, 0x1a) => VecFpOp::AbsDiff,
                (0, 0, 0x1b) => VecFpOp::MulExtended,
                (1, 0, 0x1b) => VecFpOp::Mul,
                (0, 0, 0x1c) => VecFpOp::CmEq,
                (1, 0, 0x1c) => VecFpOp::CmGe,
                (1, 1, 0x1c) => VecFpOp::CmGt,
                (1, 0, 0x1d) => VecFpOp::AcGe,
                (1, 1, 0x1d) => VecFpOp::AcGt,
                (0, 0, 0x1e) => VecFpOp::Max,
                (0, 1, 0x1e) => VecFpOp::Min,
                (1, 0, 0x1f) => VecFpOp::Div,
                (0, 0, 0x1f) => VecFpOp::RecipStep,
                (0, 1, 0x1f) => VecFpOp::RSqrtStep,
                _ => return None,
            };
            let scalar_allowed = matches!(
                op,
                VecFpOp::RecipStep
                    | VecFpOp::RSqrtStep
                    | VecFpOp::AbsDiff
                    | VecFpOp::MulExtended
                    | VecFpOp::CmEq
                    | VecFpOp::CmGe
                    | VecFpOp::CmGt
                    | VecFpOp::AcGe
                    | VecFpOp::AcGt
            );
            if scalar && !scalar_allowed {
                return None;
            }
            return Some(SimdOp::VecFp {
                op,
                size: fp,
                q,
                scalar,
                rd,
                rn,
                rm,
            });
        }
        let pairwise = match (u, opcode) {
            (0, 0x17) => Some(VecIntOp::Add),
            (0, 0x14) => Some(VecIntOp::Smax),
            (1, 0x14) => Some(VecIntOp::Umax),
            (0, 0x15) => Some(VecIntOp::Smin),
            (1, 0x15) => Some(VecIntOp::Umin),
            _ => None,
        };
        if let Some(op) = pairwise {
            if scalar || (size == 3 && (!q || op != VecIntOp::Add)) {
                return None;
            }
            return Some(SimdOp::VecIntPairwise {
                op,
                bits: 8 << size,
                q,
                scalar: false,
                rd,
                rn,
                rm,
            });
        }
        let op = match (u, opcode) {
            (0, 0x00) => VecIntOp::Shadd,
            (1, 0x00) => VecIntOp::Uhadd,
            (0, 0x01) => VecIntOp::Sqadd,
            (1, 0x01) => VecIntOp::Uqadd,
            (0, 0x02) => VecIntOp::Srhadd,
            (1, 0x02) => VecIntOp::Urhadd,
            (0, 0x04) => VecIntOp::Shsub,
            (1, 0x04) => VecIntOp::Uhsub,
            (0, 0x05) => VecIntOp::Sqsub,
            (1, 0x05) => VecIntOp::Uqsub,
            (0, 0x12) => VecIntOp::Mla,
            (1, 0x12) => VecIntOp::Mls,
            (0, 0x09) => VecIntOp::Sqshl,
            (1, 0x09) => VecIntOp::Uqshl,
            (0, 0x0a) => VecIntOp::Srshl,
            (1, 0x0a) => VecIntOp::Urshl,
            (0, 0x0b) => VecIntOp::Sqrshl,
            (1, 0x0b) => VecIntOp::Uqrshl,
            (0, 0x16) => VecIntOp::Sqdmulh,
            (1, 0x16) => VecIntOp::Sqrdmulh,
            (0, 0x10) => VecIntOp::Add,
            (1, 0x10) => VecIntOp::Sub,
            (0, 0x13) if size != 3 => VecIntOp::Mul,
            (0, 0x11) => VecIntOp::CmTst,
            (1, 0x11) => VecIntOp::CmEq,
            (0, 0x06) => VecIntOp::CmGt,
            (1, 0x06) => VecIntOp::CmHi,
            (0, 0x07) => VecIntOp::CmGe,
            (1, 0x07) => VecIntOp::CmHs,
            (0, 0x08) => VecIntOp::Sshl,
            (1, 0x08) => VecIntOp::Ushl,
            (0, 0x0c) => VecIntOp::Smax,
            (1, 0x0c) => VecIntOp::Umax,
            (0, 0x0d) => VecIntOp::Smin,
            (1, 0x0d) => VecIntOp::Umin,
            (0, 0x0e) => VecIntOp::Sabd,
            (1, 0x0e) => VecIntOp::Uabd,
            _ => return None,
        };
        let minmax = matches!(
            op,
            VecIntOp::Smax
                | VecIntOp::Smin
                | VecIntOp::Umax
                | VecIntOp::Umin
                | VecIntOp::Sabd
                | VecIntOp::Uabd
        );
        let saturating = matches!(
            op,
            VecIntOp::Sqadd
                | VecIntOp::Uqadd
                | VecIntOp::Sqsub
                | VecIntOp::Uqsub
                | VecIntOp::Sqshl
                | VecIntOp::Uqshl
                | VecIntOp::Sqrshl
                | VecIntOp::Uqrshl
        );
        let doubling = matches!(op, VecIntOp::Sqdmulh | VecIntOp::Sqrdmulh);
        let no_doubleword = minmax
            || matches!(
                op,
                VecIntOp::Mul
                    | VecIntOp::Mla
                    | VecIntOp::Mls
                    | VecIntOp::Shadd
                    | VecIntOp::Uhadd
                    | VecIntOp::Shsub
                    | VecIntOp::Uhsub
                    | VecIntOp::Srhadd
                    | VecIntOp::Urhadd
            );
        let scalar_allowed = saturating || doubling || (size == 3 && !no_doubleword);
        if (size == 3 && !q && !scalar)
            || (scalar && !scalar_allowed)
            || (no_doubleword && size == 3)
            || (doubling && !matches!(size, 1 | 2))
        {
            return None;
        }
        return Some(SimdOp::VecInt {
            op,
            bits: 8 << size,
            q,
            scalar,
            rd,
            rn,
            rm,
        });
    }
    // Two-register misc (vector 0x0e20_0800, scalar 0x5e20_0800).
    let misc_vector = word & 0x9f3e_0c00 == 0x0e20_0800;
    let misc_scalar = word & 0xdf3e_0c00 == 0x5e20_0800;
    if misc_vector || misc_scalar {
        let scalar = misc_scalar;
        let opcode = (word >> 12) & 31;
        let a = size >> 1;
        let fp = fp_size_bit(size & 1);
        let fp_op = match (u, a, opcode) {
            (0, 1, 0x0f) => Some(VecFpUnaryOp::Abs),
            (1, 1, 0x0f) => Some(VecFpUnaryOp::Neg),
            (1, 1, 0x1f) => Some(VecFpUnaryOp::Sqrt),
            (0, 0, 0x1a) => Some(VecFpUnaryOp::ToSigned(Rounding::Nearest)),
            (1, 0, 0x1a) => Some(VecFpUnaryOp::ToUnsigned(Rounding::Nearest)),
            (0, 1, 0x1a) => Some(VecFpUnaryOp::ToSigned(Rounding::Plus)),
            (1, 1, 0x1a) => Some(VecFpUnaryOp::ToUnsigned(Rounding::Plus)),
            (0, 0, 0x1b) => Some(VecFpUnaryOp::ToSigned(Rounding::Minus)),
            (1, 0, 0x1b) => Some(VecFpUnaryOp::ToUnsigned(Rounding::Minus)),
            (0, 1, 0x1b) => Some(VecFpUnaryOp::ToSigned(Rounding::Zero)),
            (1, 1, 0x1b) => Some(VecFpUnaryOp::ToUnsigned(Rounding::Zero)),
            (0, 0, 0x1c) => Some(VecFpUnaryOp::ToSigned(Rounding::Away)),
            (1, 0, 0x1c) => Some(VecFpUnaryOp::ToUnsigned(Rounding::Away)),
            (0, 0, 0x1d) => Some(VecFpUnaryOp::FromSigned),
            (1, 0, 0x1d) => Some(VecFpUnaryOp::FromUnsigned),
            (0, 0, 0x18) => Some(VecFpUnaryOp::Round(Rounding::Nearest)),
            (0, 1, 0x18) => Some(VecFpUnaryOp::Round(Rounding::Plus)),
            (0, 0, 0x19) => Some(VecFpUnaryOp::Round(Rounding::Minus)),
            (0, 1, 0x19) => Some(VecFpUnaryOp::Round(Rounding::Zero)),
            (1, 0, 0x18) => Some(VecFpUnaryOp::Round(Rounding::Away)),
            (0, 1, 0x0c) => Some(VecFpUnaryOp::CmGtZero),
            (1, 1, 0x0c) => Some(VecFpUnaryOp::CmGeZero),
            (0, 1, 0x0d) => Some(VecFpUnaryOp::CmEqZero),
            (1, 1, 0x0d) => Some(VecFpUnaryOp::CmLeZero),
            (0, 1, 0x0e) => Some(VecFpUnaryOp::CmLtZero),
            (0, 1, 0x1d) => Some(VecFpUnaryOp::RecipEstimate),
            (1, 1, 0x1d) => Some(VecFpUnaryOp::RSqrtEstimate),
            _ => None,
        };
        if let Some(op) = fp_op {
            if fp == FpSize::D && !q && !scalar {
                return None;
            }
            if scalar
                && matches!(
                    op,
                    VecFpUnaryOp::Abs
                        | VecFpUnaryOp::Neg
                        | VecFpUnaryOp::Sqrt
                        | VecFpUnaryOp::Round(_)
                )
            {
                return None;
            }
            return Some(SimdOp::VecFpUnary {
                op,
                size: fp,
                q,
                scalar,
                rd,
                rn,
                fbits: 0,
            });
        }
        if !scalar && matches!(opcode, 0x02 | 0x06) && size != 3 {
            return Some(SimdOp::VecPairwiseLong {
                signed: u == 0,
                accumulate: opcode == 0x06,
                bits: 8 << size,
                q,
                rd,
                rn,
            });
        }
        let narrow = match (u, opcode) {
            (0, 0x12) => Some(VecNarrowOp::Truncate),
            (1, 0x12) => Some(VecNarrowOp::SignedToUnsigned),
            (0, 0x14) => Some(VecNarrowOp::Signed),
            (1, 0x14) => Some(VecNarrowOp::Unsigned),
            _ => None,
        };
        if let Some(op) = narrow {
            if size == 3 || (scalar && op == VecNarrowOp::Truncate) {
                return None;
            }
            return Some(SimdOp::VecNarrow {
                op,
                round: false,
                bits: 16 << size,
                upper: q && !scalar,
                scalar,
                rd,
                rn,
                shift: 0,
            });
        }
        if !scalar && u == 1 && opcode == 0x13 && size != 3 {
            // SHLL/SHLL2: widen and shift left by the source element size.
            return Some(SimdOp::VecShiftLong {
                signed: false,
                bits: 8 << size,
                upper: q,
                rd,
                rn,
                shift: 8 << size,
            });
        }
        if !scalar && u == 0 && a == 0 && matches!(opcode, 0x16 | 0x17) {
            let (wide, small) = if size & 1 == 0 {
                (FpPrecision::S, FpPrecision::H)
            } else {
                (FpPrecision::D, FpPrecision::S)
            };
            let (from, to) = if opcode == 0x16 {
                (wide, small)
            } else {
                (small, wide)
            };
            return Some(SimdOp::VecFpConvert {
                from,
                to,
                upper: q,
                rd,
                rn,
            });
        }
        let op = match (u, opcode) {
            (0, 0x04) if size < 3 => VecUnaryOp::Cls,
            (1, 0x04) if size < 3 => VecUnaryOp::Clz,
            (0, 0x00) if size < 3 => VecUnaryOp::Rev64,
            (1, 0x00) if size < 2 => VecUnaryOp::Rev32,
            (0, 0x01) if size == 0 => VecUnaryOp::Rev16,
            (0, 0x05) if size == 0 => VecUnaryOp::Cnt,
            (1, 0x05) if size == 0 => VecUnaryOp::Not,
            (1, 0x05) if size == 1 => VecUnaryOp::Rbit,
            (0, 0x08) => VecUnaryOp::CmGtZero,
            (1, 0x08) => VecUnaryOp::CmGeZero,
            (0, 0x09) => VecUnaryOp::CmEqZero,
            (1, 0x09) => VecUnaryOp::CmLeZero,
            (0, 0x0a) => VecUnaryOp::CmLtZero,
            (0, 0x0b) => VecUnaryOp::Abs,
            (1, 0x0b) => VecUnaryOp::Neg,
            _ => return None,
        };
        let reversal = matches!(
            op,
            VecUnaryOp::Rev64 | VecUnaryOp::Rev32 | VecUnaryOp::Rev16
        );
        if (size == 3 && !q && !scalar) || (scalar && (size != 3 || reversal)) {
            return None;
        }
        // RBIT and NOT operate on bytes regardless of the size field.
        let bits = if matches!(op, VecUnaryOp::Rbit | VecUnaryOp::Not) {
            8
        } else {
            8 << size
        };
        return Some(SimdOp::VecUnary {
            op,
            bits,
            q,
            scalar,
            rd,
            rn,
        });
    }
    // Across lanes.
    // Scalar ADDP Dd, Vn.2D.
    if word & 0xff3e_0c00 == 0x5e30_0800 && (word >> 12) & 31 == 0x1b && size == 3 {
        return Some(SimdOp::VecIntPairwise {
            op: VecIntOp::Add,
            bits: 64,
            q: true,
            scalar: true,
            rd,
            rn,
            rm,
        });
    }
    // Scalar pairwise FP (U=1, single/double): FADDP, FMAXNMP, FMINNMP,
    // FMAXP, FMINP on the two lanes of Vn.
    if word & 0xff3e_0c00 == 0x7e30_0800 {
        let o1 = (word >> 23) & 1;
        let op = match (o1, (word >> 12) & 31) {
            (0, 0x0c) => VecFpOp::MaxNum,
            (1, 0x0c) => VecFpOp::MinNum,
            (0, 0x0d) => VecFpOp::Add,
            (0, 0x0f) => VecFpOp::Max,
            (1, 0x0f) => VecFpOp::Min,
            _ => return None,
        };
        return Some(SimdOp::VecFpPairwise {
            op,
            size: fp_size_bit(size & 1),
            q: false,
            scalar: true,
            rd,
            rn,
            rm,
        });
    }
    if word & 0x9f3e_0c00 == 0x0e30_0800 {
        let opcode = (word >> 12) & 31;
        let op = match (u, opcode) {
            (0, 0x1b) => VecAcrossOp::AddV,
            (0, 0x0a) => VecAcrossOp::SMaxV,
            (1, 0x0a) => VecAcrossOp::UMaxV,
            (0, 0x1a) => VecAcrossOp::SMinV,
            (1, 0x1a) => VecAcrossOp::UMinV,
            (0, 0x03) => VecAcrossOp::SAddLV,
            (1, 0x03) => VecAcrossOp::UAddLV,
            _ => return None,
        };
        if size == 3 || (size == 2 && !q) {
            return None;
        }
        return Some(SimdOp::VecAcross {
            op,
            bits: 8 << size,
            q,
            rd,
            rn,
        });
    }
    // By element (vector 0x0f00_0000, scalar 0x5f00_0000; bit 10 clear).
    let element_vector = word & 0x9f00_0400 == 0x0f00_0000;
    let element_scalar = word & 0xdf00_0400 == 0x5f00_0000;
    if (element_vector || element_scalar) && matches!(size, 1 | 2) {
        let scalar = element_scalar;
        let opcode = (word >> 12) & 15;
        let (h, l, m) = ((word >> 11) & 1, (word >> 21) & 1, (word >> 20) & 1);
        let (index, rm) = if size == 1 {
            (((h << 2) | (l << 1) | m) as u8, rm & 15)
        } else {
            (((h << 1) | l) as u8, rm)
        };
        let bits = 8u8 << size;
        let long = match opcode {
            0x2 => Some(VecLongOp::Mlal),
            0x6 => Some(VecLongOp::Mlsl),
            0xa => Some(VecLongOp::Mull),
            _ => None,
        };
        if let Some(op) = long {
            if scalar {
                return None;
            }
            return Some(SimdOp::VecLongElement {
                op,
                signed: u == 0,
                bits,
                upper: q,
                rd,
                rn,
                rm,
                index,
            });
        }
        let op = match (u, opcode) {
            (0, 0x8) => Some(VecIntOp::Mul),
            (1, 0x0) => Some(VecIntOp::Mla),
            (1, 0x4) => Some(VecIntOp::Mls),
            (0, 0xc) => Some(VecIntOp::Sqdmulh),
            (0, 0xd) => Some(VecIntOp::Sqrdmulh),
            _ => None,
        };
        if let Some(op) = op {
            if scalar && !matches!(op, VecIntOp::Sqdmulh | VecIntOp::Sqrdmulh) {
                return None;
            }
            return Some(SimdOp::VecIntElement {
                op,
                bits,
                q,
                scalar,
                rd,
                rn,
                rm,
                index,
            });
        }
    }
    if (element_vector || element_scalar) && size >= 2 {
        let scalar = element_scalar;
        let opcode = (word >> 12) & 15;
        let op = match (u, opcode) {
            (0, 0x1) => VecFpOp::MulAdd,
            (0, 0x5) => VecFpOp::MulSub,
            (0, 0x9) => VecFpOp::Mul,
            (1, 0x9) => VecFpOp::MulExtended,
            _ => return None,
        };
        let fp = fp_size_bit(size & 1);
        let (h, l) = ((word >> 11) & 1, (word >> 21) & 1);
        let index = match fp {
            FpSize::S => (h << 1) | l,
            FpSize::D => {
                if l != 0 || (!q && !scalar) {
                    return None;
                }
                h
            }
        };
        return Some(SimdOp::VecFpElement {
            op,
            size: fp,
            q,
            scalar,
            rd,
            rn,
            rm,
            index: index as u8,
        });
    }
    // Shift by immediate (vector 0x0f00_0400, scalar 0x5f00_0400; immh != 0).
    let shift_vector = word & 0x9f80_0400 == 0x0f00_0400;
    let shift_scalar = word & 0xdf80_0400 == 0x5f00_0400;
    if (shift_vector || shift_scalar) && (word >> 19) & 15 != 0 {
        let scalar = shift_scalar;
        let immh = (word >> 19) & 15;
        let immhb = (word >> 16) & 0x7f;
        let esize = 8u32 << (31 - immh.leading_zeros());
        let opcode = (word >> 11) & 31;
        if esize == 64 && !q && !scalar {
            return None;
        }
        let right = (2 * esize - immhb) as u8;
        let left = (immhb - esize) as u8;
        return match (u, opcode) {
            (_, 0x00) | (_, 0x02) => {
                if scalar && esize != 64 {
                    return None;
                }
                let op = match (u, opcode) {
                    (0, 0x00) => VecShiftOp::Sshr,
                    (1, 0x00) => VecShiftOp::Ushr,
                    (0, _) => VecShiftOp::Ssra,
                    _ => VecShiftOp::Usra,
                };
                Some(SimdOp::VecShift {
                    op,
                    bits: esize as u8,
                    q,
                    scalar,
                    rd,
                    rn,
                    shift: right,
                })
            }
            (0, 0x0a) => {
                if scalar && esize != 64 {
                    return None;
                }
                Some(SimdOp::VecShift {
                    op: VecShiftOp::Shl,
                    bits: esize as u8,
                    q,
                    scalar,
                    rd,
                    rn,
                    shift: left,
                })
            }
            (_, 0x04) | (_, 0x06) => {
                if scalar && esize != 64 {
                    return None;
                }
                let op = match (u, opcode) {
                    (0, 0x04) => VecShiftOp::Srshr,
                    (1, 0x04) => VecShiftOp::Urshr,
                    (0, _) => VecShiftOp::Srsra,
                    _ => VecShiftOp::Ursra,
                };
                Some(SimdOp::VecShift {
                    op,
                    bits: esize as u8,
                    q,
                    scalar,
                    rd,
                    rn,
                    shift: right,
                })
            }
            (0, 0x0e) | (1, 0x0e) | (1, 0x0c) => {
                let op = match (u, opcode) {
                    (0, _) => VecShiftOp::Sqshl,
                    (1, 0x0e) => VecShiftOp::Uqshl,
                    _ => VecShiftOp::Sqshlu,
                };
                Some(SimdOp::VecShift {
                    op,
                    bits: esize as u8,
                    q,
                    scalar,
                    rd,
                    rn,
                    shift: left,
                })
            }
            (0, 0x11) | (1, 0x10) | (1, 0x11) | (_, 0x12) | (_, 0x13) if esize < 64 => {
                let (op, round) = match (u, opcode) {
                    (0, 0x11) => (VecNarrowOp::Truncate, true),
                    (1, 0x10) => (VecNarrowOp::SignedToUnsigned, false),
                    (1, 0x11) => (VecNarrowOp::SignedToUnsigned, true),
                    (0, 0x12) => (VecNarrowOp::Signed, false),
                    (1, 0x12) => (VecNarrowOp::Unsigned, false),
                    (0, _) => (VecNarrowOp::Signed, true),
                    _ => (VecNarrowOp::Unsigned, true),
                };
                if scalar && op == VecNarrowOp::Truncate {
                    return None;
                }
                Some(SimdOp::VecNarrow {
                    op,
                    round,
                    bits: (esize * 2) as u8,
                    upper: q && !scalar,
                    scalar,
                    rd,
                    rn,
                    shift: right,
                })
            }
            (_, 0x14) if !scalar && esize < 64 => Some(SimdOp::VecShiftLong {
                signed: u == 0,
                bits: esize as u8,
                upper: q,
                rd,
                rn,
                shift: left,
            }),
            (0, 0x10) if !scalar && esize < 64 => Some(SimdOp::VecShift {
                op: VecShiftOp::Shrn,
                bits: (esize * 2) as u8,
                q,
                scalar: false,
                rd,
                rn,
                shift: right,
            }),
            (_, 0x1c) | (_, 0x1f) if esize >= 32 => {
                let size = if esize == 32 { FpSize::S } else { FpSize::D };
                let op = match (u, opcode) {
                    (0, 0x1c) => VecFpUnaryOp::FromSigned,
                    (1, 0x1c) => VecFpUnaryOp::FromUnsigned,
                    (0, _) => VecFpUnaryOp::ToSigned(Rounding::Zero),
                    _ => VecFpUnaryOp::ToUnsigned(Rounding::Zero),
                };
                Some(SimdOp::VecFpUnary {
                    op,
                    size,
                    q,
                    scalar,
                    rd,
                    rn,
                    fbits: right,
                })
            }
            _ => None,
        };
    }
    // TBL/TBX.
    if word & 0xbfe0_8c00 == 0x0e00_0000 {
        return Some(SimdOp::VecTable {
            q,
            extend: word & (1 << 12) != 0,
            rd,
            rn,
            regs: ((word >> 13) & 3) as u8 + 1,
            rm,
        });
    }
    // Permute (UZP/TRN/ZIP).
    if word & 0xbf20_8c00 == 0x0e00_0800 {
        let op = match (word >> 12) & 7 {
            1 => VecPermuteOp::Uzp1,
            2 => VecPermuteOp::Trn1,
            3 => VecPermuteOp::Zip1,
            5 => VecPermuteOp::Uzp2,
            6 => VecPermuteOp::Trn2,
            7 => VecPermuteOp::Zip2,
            _ => return None,
        };
        if size == 3 && !q {
            return None;
        }
        return Some(SimdOp::VecPermute {
            op,
            bits: 8 << size,
            q,
            rd,
            rn,
            rm,
        });
    }
    // EXT.
    if word & 0xbfe0_8400 == 0x2e00_0000 {
        let index = ((word >> 11) & 15) as u8;
        if !q && index >= 8 {
            return None;
        }
        return Some(SimdOp::VecExt {
            q,
            rd,
            rn,
            rm,
            index,
        });
    }
    // PMULL/PMULL2 (8-bit, and 64-bit into 128-bit).
    if word & 0xbf20_fc00 == 0x0e20_e000 && matches!(size, 0 | 3) {
        return Some(SimdOp::VecPolyMulLong {
            bits: if size == 0 { 8 } else { 64 },
            upper: q,
            rd,
            rn,
            rm,
        });
    }
    // Three different (widening).
    if word & 0x9f20_0c00 == 0x0e20_0000 && size != 3 {
        let op = match (word >> 12) & 15 {
            0x0 => VecLongOp::Addl,
            0x1 => VecLongOp::Addw,
            0x2 => VecLongOp::Subl,
            0x3 => VecLongOp::Subw,
            0x8 => VecLongOp::Mlal,
            0xa => VecLongOp::Mlsl,
            0xc => VecLongOp::Mull,
            _ => return None,
        };
        return Some(SimdOp::VecLong {
            op,
            signed: u == 0,
            bits: 8 << size,
            upper: q,
            rd,
            rn,
            rm,
        });
    }
    None
}

fn decode_fp_scalar(word: u32, rd: u8, rn: u8, rm: u8) -> Option<SimdOp> {
    let ftype = (word >> 22) & 3;
    // Conversion between floating-point and integer (bits 15:10 zero).
    if word & 0x0000_fc00 == 0 && word & (1 << 24) == 0 && word & (1 << 29) == 0 {
        let sf = word >> 31;
        let gp_bits = if sf == 1 { 64 } else { 32 };
        let rmode = (word >> 19) & 3;
        let opcode = (word >> 16) & 7;
        return match (rmode, opcode) {
            (_, 0 | 1) | (0, 4 | 5) => {
                let size = fp_size(ftype)?;
                let rounding = if opcode >= 4 {
                    Rounding::Away
                } else {
                    match rmode {
                        0 => Rounding::Nearest,
                        1 => Rounding::Plus,
                        2 => Rounding::Minus,
                        _ => Rounding::Zero,
                    }
                };
                Some(SimdOp::FpToInt {
                    rd,
                    rn,
                    gp_bits,
                    size,
                    signed: opcode & 1 == 0,
                    rounding,
                })
            }
            (0, 2 | 3) => Some(SimdOp::IntToFp {
                rd,
                rn,
                gp_bits,
                size: fp_size(ftype)?,
                signed: opcode == 2,
            }),
            (0, 6 | 7) if (sf, ftype) == (0, 0) || (sf, ftype) == (1, 1) => Some(if opcode == 6 {
                SimdOp::FpMoveToGeneral {
                    rd,
                    rn,
                    bits: gp_bits,
                    upper: false,
                }
            } else {
                SimdOp::FpMoveFromGeneral {
                    rd,
                    rn,
                    bits: gp_bits,
                    upper: false,
                }
            }),
            (1, 6 | 7) if (sf, ftype) == (1, 2) => Some(if opcode == 6 {
                SimdOp::FpMoveToGeneral {
                    rd,
                    rn,
                    bits: 64,
                    upper: true,
                }
            } else {
                SimdOp::FpMoveFromGeneral {
                    rd,
                    rn,
                    bits: 64,
                    upper: true,
                }
            }),
            _ => None,
        };
    }
    if word & (1 << 31) != 0 || word & (1 << 29) != 0 {
        return None;
    }
    let size = fp_size(ftype);
    // Data-processing, 1 source.
    if (word >> 10) & 0x1f == 0x10 {
        let opcode = (word >> 15) & 0x3f;
        if (4..=7).contains(&opcode) {
            let precision = |value: u32| match value {
                0 => Some(FpPrecision::S),
                1 => Some(FpPrecision::D),
                3 => Some(FpPrecision::H),
                _ => None,
            };
            let from = precision(ftype)?;
            let to = match opcode {
                4 => FpPrecision::S,
                5 => FpPrecision::D,
                7 => FpPrecision::H,
                _ => return None,
            };
            if from == to {
                return None;
            }
            if from == FpPrecision::H || to == FpPrecision::H {
                return Some(SimdOp::FpConvertPrecision { rd, rn, from, to });
            }
            let size = |value: FpPrecision| {
                if value == FpPrecision::S {
                    FpSize::S
                } else {
                    FpSize::D
                }
            };
            return Some(SimdOp::FpConvert {
                rd,
                rn,
                from: size(from),
                to: size(to),
            });
        }
        let op = match opcode {
            0 => FpUnaryOp::Mov,
            1 => FpUnaryOp::Abs,
            2 => FpUnaryOp::Neg,
            3 => FpUnaryOp::Sqrt,
            8 => FpUnaryOp::Round(Rounding::Nearest),
            9 => FpUnaryOp::Round(Rounding::Plus),
            10 => FpUnaryOp::Round(Rounding::Minus),
            11 => FpUnaryOp::Round(Rounding::Zero),
            12 => FpUnaryOp::Round(Rounding::Away),
            14 => FpUnaryOp::RoundExact,
            15 => FpUnaryOp::Round(Rounding::Nearest),
            _ => return None,
        };
        let op = if opcode == 15 {
            // FRINTI uses the FPCR mode, which is checked to be RN.
            FpUnaryOp::Round(Rounding::Nearest)
        } else {
            op
        };
        return Some(SimdOp::FpUnary {
            op,
            size: size?,
            rd,
            rn,
        });
    }
    // Compare.
    if word & 0x0000_fc07 == 0x0000_2000 {
        let opcode2 = (word >> 3) & 3;
        let zero = opcode2 & 1 != 0;
        if zero && rm != 0 {
            return None;
        }
        return Some(SimdOp::FpCompare {
            size: size?,
            rn,
            rm: (!zero).then_some(rm),
            signaling: opcode2 & 2 != 0,
        });
    }
    // Immediate.
    if word & 0x0000_1fe0 == 0x0000_1000 {
        let size = size?;
        let imm8 = ((word >> 13) & 0xff) as u8;
        return Some(SimdOp::FpMoveImmediate {
            size,
            rd,
            bits: expand_fp_immediate(imm8, size),
        });
    }
    match (word >> 10) & 3 {
        1 => Some(SimdOp::FpConditionalCompare {
            size: size?,
            rn,
            rm,
            condition: ((word >> 12) & 15) as u8,
            nzcv: (word & 15) as u8,
            signaling: word & (1 << 4) != 0,
        }),
        2 => {
            let op = match (word >> 12) & 15 {
                0 => FpBinaryOp::Mul,
                1 => FpBinaryOp::Div,
                2 => FpBinaryOp::Add,
                3 => FpBinaryOp::Sub,
                4 => FpBinaryOp::Max,
                5 => FpBinaryOp::Min,
                6 => FpBinaryOp::MaxNum,
                7 => FpBinaryOp::MinNum,
                8 => FpBinaryOp::NegMul,
                _ => return None,
            };
            Some(SimdOp::FpBinary {
                op,
                size: size?,
                rd,
                rn,
                rm,
            })
        }
        3 => Some(SimdOp::FpSelect {
            size: size?,
            rd,
            rn,
            rm,
            condition: ((word >> 12) & 15) as u8,
        }),
        _ => None,
    }
}

fn decode_memory(word: u32, pc: u64, rt: u8, rn: u8, rm: u8) -> Option<SimdOp> {
    if word & 0x3e00_0000 == 0x0c00_0000 {
        return decode_structure(word, rt, rn, rm);
    }
    let size = word >> 30;
    let opc = (word >> 22) & 3;
    // Literal.
    if word & 0x3f00_0000 == 0x1c00_0000 {
        let bytes = match size {
            0 => 4,
            1 => 8,
            2 => 16,
            _ => return None,
        };
        let offset = signed_field((word >> 5) & 0x7_ffff, 19) << 2;
        return Some(SimdOp::LoadLiteral {
            rt,
            bytes,
            address: pc.wrapping_add(offset as u64),
        });
    }
    // Pairs (Q offset/pre/post forms are decoded by the core decoder first).
    if word & 0x3c00_0000 == 0x2c00_0000 {
        let bytes = match size {
            0 => 4,
            1 => 8,
            2 => 16,
            _ => return None,
        };
        let writeback = match (word >> 23) & 3 {
            0 | 2 => Writeback::None,
            1 => Writeback::Post,
            _ => Writeback::Pre,
        };
        let load = word & (1 << 22) != 0;
        let rt2 = ((word >> 10) & 31) as u8;
        if load && rt == rt2 {
            return None;
        }
        return Some(SimdOp::LoadStorePair {
            rt,
            rt2,
            rn,
            bytes,
            load,
            offset: signed_field((word >> 15) & 0x7f, 7) * i64::from(bytes),
            writeback,
        });
    }
    if word & 0x3b00_0000 != 0x3900_0000 && word & 0x3b00_0000 != 0x3800_0000 {
        return None;
    }
    let (bytes, load) = match (size, opc) {
        (_, 0) => (1u8 << size, false),
        (_, 1) => (1u8 << size, true),
        (0, 2) => (16, false),
        (0, 3) => (16, true),
        _ => return None,
    };
    let scale = bytes.trailing_zeros();
    let address = if word & (1 << 24) != 0 {
        Address::Immediate {
            offset: (((word >> 10) & 0xfff) << scale) as i64,
            writeback: Writeback::None,
        }
    } else if word & (1 << 21) == 0 {
        Address::Immediate {
            offset: signed_field((word >> 12) & 0x1ff, 9),
            writeback: writeback_mode((word >> 10) & 3)?,
        }
    } else if (word >> 10) & 3 == 2 {
        let option = ((word >> 13) & 7) as u8;
        if option & 2 == 0 {
            return None;
        }
        Address::Register {
            rm,
            option,
            shift: if word & (1 << 12) == 0 {
                0
            } else {
                scale as u8
            },
        }
    } else {
        return None;
    };
    Some(SimdOp::LoadStore {
        rt,
        rn,
        bytes,
        load,
        address,
    })
}

/// LD1-LD4/ST1-ST4 (multiple and single structures) and LD1R-LD4R.
fn decode_structure(word: u32, rt: u8, rn: u8, rm: u8) -> Option<SimdOp> {
    let q = word & (1 << 30) != 0;
    let load = word & (1 << 22) != 0;
    let size = (word >> 10) & 3;
    let post_index = word & (1 << 23) != 0;
    let post = post_index.then_some(rm);
    if word & 0xbfbf_0000 == 0x0c00_0000 || word & 0xbfa0_0000 == 0x0c80_0000 {
        let (registers, structure) = match (word >> 12) & 15 {
            0x0 => (4, 4),
            0x2 => (4, 1),
            0x4 => (3, 3),
            0x6 => (3, 1),
            0x7 => (1, 1),
            0x8 => (2, 2),
            0xa => (2, 1),
            _ => return None,
        };
        if size == 3 && !q && structure > 1 {
            return None;
        }
        return Some(SimdOp::TransferMultiple {
            rt,
            rn,
            registers,
            structure,
            bits: 8 << size,
            q,
            load,
            post,
        });
    }
    if word & 0xbf9f_0000 == 0x0d00_0000 || word & 0xbf80_0000 == 0x0d80_0000 {
        let opcode = (word >> 13) & 7;
        let s = (word >> 12) & 1;
        let r = (word >> 21) & 1;
        let registers = (((opcode & 1) << 1) | r) as u8 + 1;
        let qbit = u32::from(q);
        let (bits, index) = match opcode >> 1 {
            0 => (8, (qbit << 3) | (s << 2) | size),
            1 if size & 1 == 0 => (16, (qbit << 2) | (s << 1) | (size >> 1)),
            2 if size == 0 => (32, (qbit << 1) | s),
            2 if size == 1 && s == 0 => (64, qbit),
            3 if load && s == 0 => {
                return Some(SimdOp::LoadReplicate {
                    rt,
                    rn,
                    registers,
                    bits: 8 << size,
                    q,
                    post,
                });
            }
            _ => return None,
        };
        return Some(SimdOp::TransferSingle {
            rt,
            rn,
            registers,
            bits,
            index: index as u8,
            load,
            post,
        });
    }
    None
}

/// The generated-code call for a non-memory operation.
pub fn emit_call(op: &SimdOp) -> Option<String> {
    let p = "echora_guest::simd";
    Some(match *op {
        SimdOp::FpUnary { op, size, rd, rn } => {
            format!(
                "{p}::fp_unary(&mut m.cpu, {p}::{}, {p}::FpSize::{size:?}, {rd}, {rn})?;",
                unary_path(op)
            )
        }
        SimdOp::FpConvert { rd, rn, from, to } => format!(
            "{p}::fp_convert(&mut m.cpu, {rd}, {rn}, {p}::FpSize::{from:?}, {p}::FpSize::{to:?})?;"
        ),
        SimdOp::FpBinary {
            op,
            size,
            rd,
            rn,
            rm,
        } => format!(
            "{p}::fp_binary(&mut m.cpu, {p}::FpBinaryOp::{op:?}, {p}::FpSize::{size:?}, {rd}, {rn}, {rm})?;"
        ),
        SimdOp::FpTernary {
            op,
            size,
            rd,
            rn,
            rm,
            ra,
        } => format!(
            "{p}::fp_ternary(&mut m.cpu, {p}::FpTernaryOp::{op:?}, {p}::FpSize::{size:?}, {rd}, {rn}, {rm}, {ra})?;"
        ),
        SimdOp::FpCompare {
            size,
            rn,
            rm,
            signaling,
        } => format!(
            "{p}::fp_compare(&mut m.cpu, {p}::FpSize::{size:?}, {rn}, {rm:?}, {signaling})?;"
        ),
        SimdOp::FpConditionalCompare {
            size,
            rn,
            rm,
            condition,
            nzcv,
            signaling,
        } => format!(
            "{p}::fp_conditional_compare(&mut m.cpu, {p}::FpSize::{size:?}, {rn}, {rm}, {condition}, {nzcv}, {signaling})?;"
        ),
        SimdOp::FpSelect {
            size,
            rd,
            rn,
            rm,
            condition,
        } => format!(
            "{p}::fp_select(&mut m.cpu, {p}::FpSize::{size:?}, {rd}, {rn}, {rm}, {condition});"
        ),
        SimdOp::FpMoveImmediate { size, rd, bits } => {
            format!("{p}::fp_move_immediate(&mut m.cpu, {p}::FpSize::{size:?}, {rd}, {bits:#x});")
        }
        SimdOp::FpMoveToGeneral {
            rd,
            rn,
            bits,
            upper,
        } => format!("{p}::fp_move_to_general(&mut m.cpu, {rd}, {rn}, {bits}, {upper});"),
        SimdOp::FpMoveFromGeneral {
            rd,
            rn,
            bits,
            upper,
        } => format!("{p}::fp_move_from_general(&mut m.cpu, {rd}, {rn}, {bits}, {upper});"),
        SimdOp::IntToFp {
            rd,
            rn,
            gp_bits,
            size,
            signed,
        } => format!(
            "{p}::int_to_fp(&mut m.cpu, {rd}, {rn}, {gp_bits}, {p}::FpSize::{size:?}, {signed})?;"
        ),
        SimdOp::SimdIntToFp {
            rd,
            rn,
            size,
            signed,
        } => format!(
            "{p}::simd_int_to_fp(&mut m.cpu, {rd}, {rn}, {p}::FpSize::{size:?}, {signed})?;"
        ),
        SimdOp::FpToInt {
            rd,
            rn,
            gp_bits,
            size,
            signed,
            rounding,
        } => format!(
            "{p}::fp_to_int(&mut m.cpu, {rd}, {rn}, {gp_bits}, {p}::FpSize::{size:?}, {signed}, {p}::Rounding::{rounding:?})?;"
        ),
        SimdOp::MoveImmediate { rd, value } => {
            format!("{p}::move_immediate(&mut m.cpu, {rd}, {value:#x});")
        }
        SimdOp::ModifyImmediate {
            rd,
            value,
            clear,
            q,
        } => {
            format!("{p}::modify_immediate(&mut m.cpu, {rd}, {value:#x}, {clear}, {q});")
        }
        SimdOp::VectorLogical { op, rd, rn, rm, q } => format!(
            "{p}::vector_logical(&mut m.cpu, {p}::VectorLogicalOp::{op:?}, {rd}, {rn}, {rm}, {q});"
        ),
        SimdOp::InsertElement {
            rd,
            rn,
            bytes,
            dst,
            src,
        } => format!("{p}::insert_element(&mut m.cpu, {rd}, {rn}, {bytes}, {dst}, {src});"),
        SimdOp::InsertGeneral {
            rd,
            rn,
            bytes,
            index,
        } => format!("{p}::insert_general(&mut m.cpu, {rd}, {rn}, {bytes}, {index});"),
        SimdOp::MoveToGeneral {
            rd,
            rn,
            bytes,
            index,
            signed,
            gp_bits,
        } => format!(
            "{p}::move_to_general(&mut m.cpu, {rd}, {rn}, {bytes}, {index}, {signed}, {gp_bits});"
        ),
        SimdOp::DupElement {
            rd,
            rn,
            bytes,
            index,
            q,
            scalar,
        } => format!("{p}::dup_element(&mut m.cpu, {rd}, {rn}, {bytes}, {index}, {q}, {scalar});"),
        SimdOp::DupGeneral { rd, rn, bytes, q } => {
            format!("{p}::dup_general(&mut m.cpu, {rd}, {rn}, {bytes}, {q});")
        }
        SimdOp::VecFp {
            op,
            size,
            q,
            scalar,
            rd,
            rn,
            rm,
        } => format!(
            "{p}::vec_fp(&mut m.cpu, {p}::VecFpOp::{op:?}, {p}::FpSize::{size:?}, {q}, {scalar}, {rd}, {rn}, {rm})?;"
        ),
        SimdOp::VecIntPairwise {
            op,
            bits,
            q,
            scalar,
            rd,
            rn,
            rm,
        } => format!(
            "{p}::vec_int_pairwise(&mut m.cpu, {p}::VecIntOp::{op:?}, {bits}, {q}, {scalar}, {rd}, {rn}, {rm});"
        ),
        SimdOp::VecIntElement {
            op,
            bits,
            q,
            scalar,
            rd,
            rn,
            rm,
            index,
        } => format!(
            "{p}::vec_int_element(&mut m.cpu, {p}::VecIntOp::{op:?}, {bits}, {q}, {scalar}, {rd}, {rn}, {rm}, {index});"
        ),
        SimdOp::VecLongElement {
            op,
            signed,
            bits,
            upper,
            rd,
            rn,
            rm,
            index,
        } => format!(
            "{p}::vec_long_element(&mut m.cpu, {p}::VecLongOp::{op:?}, {signed}, {bits}, {upper}, {rd}, {rn}, {rm}, {index});"
        ),
        SimdOp::VecNarrow {
            op,
            round,
            bits,
            upper,
            scalar,
            rd,
            rn,
            shift,
        } => format!(
            "{p}::vec_narrow(&mut m.cpu, {p}::VecNarrowOp::{op:?}, {round}, {bits}, {upper}, {scalar}, {rd}, {rn}, {shift});"
        ),
        SimdOp::VecPolyMulLong {
            bits,
            upper,
            rd,
            rn,
            rm,
        } => format!("{p}::vec_poly_mul_long(&mut m.cpu, {bits}, {upper}, {rd}, {rn}, {rm});"),
        SimdOp::Crypto { op, rd, rn, rm } => {
            format!("{p}::crypto(&mut m.cpu, {p}::CryptoOp::{op:?}, {rd}, {rn}, {rm});")
        }
        SimdOp::VecTable {
            q,
            extend,
            rd,
            rn,
            regs,
            rm,
        } => format!("{p}::vec_table(&mut m.cpu, {q}, {extend}, {rd}, {rn}, {regs}, {rm});"),
        SimdOp::VecFpConvert {
            from,
            to,
            upper,
            rd,
            rn,
        } => format!(
            "{p}::vec_fp_convert(&mut m.cpu, {p}::FpPrecision::{from:?}, {p}::FpPrecision::{to:?}, {upper}, {rd}, {rn})?;"
        ),
        SimdOp::FpConvertPrecision { rd, rn, from, to } => format!(
            "{p}::fp_convert_precision(&mut m.cpu, {rd}, {rn}, {p}::FpPrecision::{from:?}, {p}::FpPrecision::{to:?})?;"
        ),
        SimdOp::VecFpPairwise {
            op,
            size,
            q,
            scalar,
            rd,
            rn,
            rm,
        } => format!(
            "{p}::vec_fp_pairwise(&mut m.cpu, {p}::VecFpOp::{op:?}, {p}::FpSize::{size:?}, {q}, {scalar}, {rd}, {rn}, {rm})?;"
        ),
        SimdOp::VecFpElement {
            op,
            size,
            q,
            scalar,
            rd,
            rn,
            rm,
            index,
        } => format!(
            "{p}::vec_fp_element(&mut m.cpu, {p}::VecFpOp::{op:?}, {p}::FpSize::{size:?}, {q}, {scalar}, {rd}, {rn}, {rm}, {index})?;"
        ),
        SimdOp::VecInt {
            op,
            bits,
            q,
            scalar,
            rd,
            rn,
            rm,
        } => format!(
            "{p}::vec_int(&mut m.cpu, {p}::VecIntOp::{op:?}, {bits}, {q}, {scalar}, {rd}, {rn}, {rm});"
        ),
        SimdOp::VecUnary {
            op,
            bits,
            q,
            scalar,
            rd,
            rn,
        } => format!(
            "{p}::vec_unary(&mut m.cpu, {p}::VecUnaryOp::{op:?}, {bits}, {q}, {scalar}, {rd}, {rn});"
        ),
        SimdOp::VecFpUnary {
            op,
            size,
            q,
            scalar,
            rd,
            rn,
            fbits,
        } => format!(
            "{p}::vec_fp_unary(&mut m.cpu, {p}::{}, {p}::FpSize::{size:?}, {q}, {scalar}, {rd}, {rn}, {fbits})?;",
            vec_fp_unary_path(op)
        ),
        SimdOp::FpToFixed {
            rd,
            rn,
            gp_bits,
            size,
            signed,
            fbits,
        } => format!(
            "{p}::fp_to_fixed(&mut m.cpu, {rd}, {rn}, {gp_bits}, {p}::FpSize::{size:?}, {signed}, {fbits})?;"
        ),
        SimdOp::IntToFpFixed {
            rd,
            rn,
            gp_bits,
            size,
            signed,
            fbits,
        } => format!(
            "{p}::int_to_fp_fixed(&mut m.cpu, {rd}, {rn}, {gp_bits}, {p}::FpSize::{size:?}, {signed}, {fbits})?;"
        ),
        SimdOp::VecAcross {
            op,
            bits,
            q,
            rd,
            rn,
        } => format!(
            "{p}::vec_across(&mut m.cpu, {p}::VecAcrossOp::{op:?}, {bits}, {q}, {rd}, {rn});"
        ),
        SimdOp::VecShift {
            op,
            bits,
            q,
            scalar,
            rd,
            rn,
            shift,
        } => format!(
            "{p}::vec_shift(&mut m.cpu, {p}::VecShiftOp::{op:?}, {bits}, {q}, {scalar}, {rd}, {rn}, {shift});"
        ),
        SimdOp::VecShiftLong {
            signed,
            bits,
            upper,
            rd,
            rn,
            shift,
        } => format!(
            "{p}::vec_shift_long(&mut m.cpu, {signed}, {bits}, {upper}, {rd}, {rn}, {shift});"
        ),
        SimdOp::VecLong {
            op,
            signed,
            bits,
            upper,
            rd,
            rn,
            rm,
        } => format!(
            "{p}::vec_long(&mut m.cpu, {p}::VecLongOp::{op:?}, {signed}, {bits}, {upper}, {rd}, {rn}, {rm});"
        ),
        SimdOp::VecPairwiseLong {
            signed,
            accumulate,
            bits,
            q,
            rd,
            rn,
        } => format!(
            "{p}::vec_pairwise_long(&mut m.cpu, {signed}, {accumulate}, {bits}, {q}, {rd}, {rn});"
        ),
        SimdOp::VecPermute {
            op,
            bits,
            q,
            rd,
            rn,
            rm,
        } => format!(
            "{p}::vec_permute(&mut m.cpu, {p}::VecPermuteOp::{op:?}, {bits}, {q}, {rd}, {rn}, {rm});"
        ),
        SimdOp::VecExt {
            q,
            rd,
            rn,
            rm,
            index,
        } => format!("{p}::vec_ext(&mut m.cpu, {q}, {rd}, {rn}, {rm}, {index});"),
        SimdOp::LoadStore { .. }
        | SimdOp::LoadStorePair { .. }
        | SimdOp::LoadLiteral { .. }
        | SimdOp::TransferMultiple { .. }
        | SimdOp::TransferSingle { .. }
        | SimdOp::LoadReplicate { .. } => return None,
    })
}

fn vec_fp_unary_path(op: VecFpUnaryOp) -> String {
    match op {
        VecFpUnaryOp::Round(r) => {
            format!("VecFpUnaryOp::Round(echora_guest::simd::Rounding::{r:?})")
        }
        VecFpUnaryOp::ToSigned(r) => {
            format!("VecFpUnaryOp::ToSigned(echora_guest::simd::Rounding::{r:?})")
        }
        VecFpUnaryOp::ToUnsigned(r) => {
            format!("VecFpUnaryOp::ToUnsigned(echora_guest::simd::Rounding::{r:?})")
        }
        other => format!("VecFpUnaryOp::{other:?}"),
    }
}

fn unary_path(op: FpUnaryOp) -> String {
    match op {
        FpUnaryOp::Round(r) => format!("FpUnaryOp::Round(echora_guest::simd::Rounding::{r:?})"),
        other => format!("FpUnaryOp::{other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aarch64::disassemble;
    use echora_guest::cpu::Cpu;

    fn cpu() -> Cpu {
        Cpu::default()
    }

    fn s(cpu: &mut Cpu, index: u8, value: f32) {
        cpu.vectors[index as usize] = u128::from(value.to_bits()) | (0xdead_u128 << 64);
    }
    fn d(cpu: &mut Cpu, index: u8, value: f64) {
        cpu.vectors[index as usize] = u128::from(value.to_bits()) | (0xdead_u128 << 64);
    }
    fn get_s(cpu: &Cpu, index: u8) -> f32 {
        f32::from_bits(cpu.vectors[index as usize] as u32)
    }

    /// Every decoded word disassembles (via the independent yaxpeax decoder)
    /// to a mnemonic consistent with the decoded operation.
    #[test]
    fn decoded_classes_agree_with_an_independent_disassembler() {
        let cases: &[(u32, &str)] = &[
            (0x1e220800, "fmul"),
            (0x1e621800, "fdiv"),
            (0x1e222800, "fadd"),
            (0x1e623800, "fsub"),
            (0x1e224800, "fmax"),
            (0x1e225800, "fmin"),
            (0x1e226800, "fmaxnm"),
            (0x1e227800, "fminnm"),
            (0x1e228800, "fnmul"),
            (0x1f020c20, "fmadd"),
            (0x1f028c20, "fmsub"),
            (0x1f220c20, "fnmadd"),
            (0x1f228c20, "fnmsub"),
            (0x1e204020, "fmov"),
            (0x1e20c020, "fabs"),
            (0x1e214020, "fneg"),
            (0x1e21c020, "fsqrt"),
            (0x1e22c020, "fcvt"),
            (0x1e624020, "fcvt"),
            (0x1e264020, "frinta"),
            (0x1e254020, "frintm"),
            (0x1e202000, "fcmp"),
            (0x1e202008, "fcmp"),
            (0x1e202010, "fcmpe"),
            (0x1e221c20, "fcsel"),
            (0x1e201000, "fmov"),
            (0x1e260020, "fmov"),
            (0x9e670020, "fmov"),
            (0x9eae0020, "fmov"),
            (0x1e230020, "ucvtf"),
            (0x9e620020, "scvtf"),
            (0x1e380020, "fcvtzs"),
            (0x9e790020, "fcvtzu"),
            (0x1e300020, "fcvtms"),
            (0x1e240020, "fcvtas"),
            (0x1e200020, "fcvtns"),
            (0x7e21d800, "ucvtf"),
            (0x2f00e400, "movi"),
            (0x6f00e400, "movi"),
            (0x4f03f600, "fmov"),
            (0x4ea11c20, "mov"),
            (0x6e0c0420, "mov"),
            (0x4e0c1c20, "mov"),
            (0x0e0c3c20, "mov"),
            (0x4e040420, "mov"), // yaxpeax prints DUP (element) as mov
            (0x5e0c0420, "mov"),
            (0xbd400020, "ldr"),
            (0xfd000020, "str"),
            (0x3dc00020, "ldr"),
            (0xbc404020, "ldur"),
            (0xfc1f0fe0, "str"),
            (0xbc617820, "ldr"),
            (0x2d410420, "ldp"),
            (0x6d810fe0, "stp"),
            (0x1c000040, "ldr"),
        ];
        for &(word, mnemonic) in cases {
            let decoded = decode(word, 0x1000);
            assert!(decoded.is_some(), "{word:#010x} {}", disassemble(word, 0));
            let text = disassemble(word, 0);
            assert_eq!(
                text.split_whitespace().next(),
                Some(mnemonic),
                "{word:#010x} {text}"
            );
        }
    }

    /// Advanced SIMD and structure-memory words encoded by an independent
    /// assembler (llvm-mc), decoded here and disassembled by yaxpeax.
    #[test]
    fn vector_words_from_an_independent_assembler_decode() {
        let cases: &[(u32, &str)] = &[
            (0x4e22d420, "fadd"),
            (0x2e37d6f7, "faddp"),
            (0x4ea3fc43, "frsqrts"),
            (0x6ea1d822, "frsqrte"),
            (0x0ea1d831, "frecpe"),
            (0x0e070298, "tbl"),
            (0x4e0662b5, "tbl"),
            (0x0e614800, "sqxtn"),
            (0x0e612800, "xtn"),
            (0x0f41a042, "smull"),
            (0x0f80a044, "smull"),
            (0x6ea0b425, "sqrdmulh"),
            (0x4f83c232, "sqdmulh"),
            (0x0e60b6f7, "sqdmulh"),
            (0x5ef1b800, "addp"),
            (0x4ea2bc42, "addp"),
            (0x5ea10c00, "sqadd"),
            (0x4ea00c24, "sqadd"),
            (0x5ea12c00, "sqsub"),
            (0x6e622c63, "uqsub"),
            (0x6ea048e7, "clz"),
            (0x0e720458, "shadd"),
            (0x4e642403, "shsub"),
            (0x4f2876d6, "sqshl"),
            (0x0f149c00, "sqrshrn"),
            (0x0f168c84, "rshrn"),
            (0x5f712421, "srshr"),
            (0x6eb097fe, "mls"),
            (0x0f422020, "smlal"),
            (0x0ea0a400, "smaxp"),
            (0x0e617821, "fcvtl"),
            (0x4e6178b4, "fcvtl2"),
            (0x2e601040, "uaddw"),
            (0x2ea13825, "shll"),
            (0x1ee24008, "fcvt"),
            (0x5e020016, "sha1c"),
            (0x5e021016, "sha1p"),
            (0x5e022016, "sha1m"),
            (0x5e023016, "sha1su0"),
            (0x5e024016, "sha256h"),
            (0x5e025016, "sha256h2"),
            (0x5e026016, "sha256su1"),
            (0x5e280820, "sha1h"),
            (0x5e281820, "sha1su1"),
            (0x5e282820, "sha256su0"),
            (0x4e284813, "aese"),
            (0x4e285a56, "aesd"),
            (0x4e286821, "aesmc"),
            (0x4e287821, "aesimc"),
            (0x0ee1e004, "pmull"),
            (0x4e21e004, "pmull2"),
            (0x4ea154c4, "srshl"),
            (0x4ee354e7, "srshl"),
            (0x6ea154c4, "urshl"),
            (0x4ea14cc4, "sqshl"),
            (0x6ea15cc4, "uqrshl"),
            (0x6e22d420, "faddp"),
            (0x6e22f420, "fmaxp"),
            (0x6ea2f420, "fminp"),
            (0x6e22c420, "fmaxnmp"),
            (0x7e30d820, "faddp"),
            (0x7e30c820, "fmaxnmp"),
            (0x7ef0f820, "fminp"),
            (0x4ee2d420, "fsub"),
            (0x6e22dc20, "fmul"),
            (0x4fa29020, "fmul"),
            (0x4e22cc20, "fmla"),
            (0x4fa21820, "fmla"),
            (0x7ea2d420, "fabd"),
            (0x5ea1b820, "fcvtzs"),
            (0x1e18e020, "fcvtzs"),
            (0x1e02f020, "scvtf"),
            (0x0e205820, "cnt"),
            (0x6ea28c20, "cmeq"),
            (0x4e209820, "cmeq"),
            (0x2f08a420, "ushll"),
            (0x4f12a420, "sshll2"),
            (0x0e628020, "smlal"),
            (0x2ea2c020, "umull"),
            (0x4ea00820, "rev64"),
            (0x6ea0b820, "neg"),
            (0x6e022020, "ext"),
            (0x4e827820, "zip2"),
            (0x4e421820, "uzp1"),
            (0x4eb1b820, "addv"),
            (0x6e30a820, "umaxv"),
            (0x4e602820, "saddlp"),
            (0x4ea28420, "add"),
            (0x6ee28420, "sub"),
            (0x4e221c20, "and"),
            (0x6e621c20, "bsl"),
            (0x6f3d0420, "ushr"),
            (0x4f455420, "shl"),
            (0x0f108420, "shrn"),
            (0x4ea1b820, "fcvtzs"),
            (0x4e21d820, "scvtf"),
            (0x6ea0f820, "fneg"),
            (0x4ea0c820, "fcmgt"),
            (0x4c407020, "ld1"),
            (0x4cdfa820, "ld1"),
            (0x4c408820, "ld2"),
            (0x4c828820, "st2"),
            (0x0d009020, "st1"),
            (0x4ddf9020, "ld1"),
            (0x4d40c820, "ld1r"),
            (0x4c400820, "ld4"),
            (0x4c9f2c00, "st1"),
        ];
        for &(word, mnemonic) in cases {
            assert!(decode(word, 0x1000).is_some(), "{word:#010x} {mnemonic}");
            let text = disassemble(word, 0);
            assert_eq!(
                text.split_whitespace().next(),
                Some(mnemonic),
                "{word:#010x} {text}"
            );
        }
    }

    #[test]
    fn decoded_operands_for_representative_words() {
        assert_eq!(
            decode(0x1e222841, 0),
            Some(SimdOp::FpBinary {
                op: FpBinaryOp::Add,
                size: FpSize::S,
                rd: 1,
                rn: 2,
                rm: 2
            })
        );
        assert_eq!(
            decode(0x1e2e1000, 0),
            Some(SimdOp::FpMoveImmediate {
                size: FpSize::S,
                rd: 0,
                bits: u64::from(1.0f32.to_bits())
            })
        );
        assert_eq!(
            decode(0x1e6e1000, 0),
            Some(SimdOp::FpMoveImmediate {
                size: FpSize::D,
                rd: 0,
                bits: 1.0f64.to_bits()
            })
        );
        // fmov v0.4s, #1.0
        assert_eq!(
            decode(0x4f03f600, 0),
            Some(SimdOp::MoveImmediate {
                rd: 0,
                value: u128::from(1.0f32.to_bits()) * 0x0000_0001_0000_0001_0000_0001_0000_0001
            })
        );
        // movi d0, #0xff00ff00ff00ff00 (op=1, cmode=1110, imm8=0xaa)
        assert_eq!(
            decode(0x2f05e540, 0),
            Some(SimdOp::MoveImmediate {
                rd: 0,
                value: 0xff00_ff00_ff00_ff00
            })
        );
        // ldr s0, [x1, #4]
        assert_eq!(
            decode(0xbd400420, 0),
            Some(SimdOp::LoadStore {
                rt: 0,
                rn: 1,
                bytes: 4,
                load: true,
                address: Address::Immediate {
                    offset: 4,
                    writeback: Writeback::None
                }
            })
        );
        // str d0, [sp, #-16]!
        assert_eq!(
            decode(0xfc1f0fe0, 0),
            Some(SimdOp::LoadStore {
                rt: 0,
                rn: 31,
                bytes: 8,
                load: false,
                address: Address::Immediate {
                    offset: -16,
                    writeback: Writeback::Pre
                }
            })
        );
        // Unallocated / unimplemented forms stay None.
        assert_eq!(decode(0x1ee22800, 0), None); // half precision fadd
        assert_eq!(decode(0x1e2d4020, 0), None); // FP 1-source opcode 26 (unallocated)
    }

    #[test]
    fn arithmetic_matches_ieee_and_zeroes_upper_bits() {
        let mut cpu = cpu();
        s(&mut cpu, 1, 1.5);
        s(&mut cpu, 2, 2.25);
        fp_binary(&mut cpu, FpBinaryOp::Mul, FpSize::S, 0, 1, 2).unwrap();
        assert_eq!(cpu.vectors[0], u128::from(3.375f32.to_bits()));
        fp_binary(&mut cpu, FpBinaryOp::NegMul, FpSize::S, 0, 1, 2).unwrap();
        assert_eq!(get_s(&cpu, 0), -3.375);
        d(&mut cpu, 3, 1.0);
        d(&mut cpu, 4, 3.0);
        fp_binary(&mut cpu, FpBinaryOp::Div, FpSize::D, 5, 3, 4).unwrap();
        assert_eq!(cpu.vectors[5], u128::from((1.0f64 / 3.0).to_bits()));
        // Fused multiply-add rounds once.
        d(&mut cpu, 6, 0.1);
        d(&mut cpu, 7, 10.0);
        d(&mut cpu, 8, -1.0);
        fp_ternary(&mut cpu, FpTernaryOp::MulAdd, FpSize::D, 9, 6, 7, 8).unwrap();
        assert_eq!(
            f64::from_bits(cpu.vectors[9] as u64),
            0.1f64.mul_add(10.0, -1.0)
        );
        fp_ternary(&mut cpu, FpTernaryOp::MulSub, FpSize::D, 9, 6, 7, 8).unwrap();
        assert_eq!(
            f64::from_bits(cpu.vectors[9] as u64),
            (-0.1f64).mul_add(10.0, -1.0)
        );
        assert_eq!(cpu.fpsr, 0);
    }

    #[test]
    fn invalid_operations_produce_the_positive_default_nan_and_set_ioc() {
        let mut cpu = cpu();
        s(&mut cpu, 1, f32::INFINITY);
        s(&mut cpu, 2, f32::INFINITY);
        fp_binary(&mut cpu, FpBinaryOp::Sub, FpSize::S, 0, 1, 2).unwrap();
        assert_eq!(cpu.vectors[0] as u32, 0x7fc0_0000);
        assert_eq!(cpu.fpsr & FPSR_IOC, FPSR_IOC);
        // Signaling NaN in the second operand is quieted with its payload.
        cpu.fpsr = 0;
        cpu.vectors[3] = 0xffa0_0001;
        s(&mut cpu, 4, 1.0);
        fp_binary(&mut cpu, FpBinaryOp::Add, FpSize::S, 0, 4, 3).unwrap();
        assert_eq!(cpu.vectors[0] as u32, 0xffe0_0001);
        assert_eq!(cpu.fpsr, FPSR_IOC);
        // DN forces the default NaN even for propagated NaNs.
        cpu.fpcr = FPCR_DN;
        fp_binary(&mut cpu, FpBinaryOp::Add, FpSize::S, 0, 4, 3).unwrap();
        assert_eq!(cpu.vectors[0] as u32, 0x7fc0_0000);
        // Division by zero.
        cpu.fpcr = 0;
        cpu.fpsr = 0;
        s(&mut cpu, 5, 0.0);
        fp_binary(&mut cpu, FpBinaryOp::Div, FpSize::S, 0, 4, 5).unwrap();
        assert_eq!(get_s(&cpu, 0), f32::INFINITY);
        assert_eq!(cpu.fpsr, FPSR_DZC);
        // Non-nearest rounding stops explicitly.
        cpu.fpcr = 1 << 22;
        assert!(fp_binary(&mut cpu, FpBinaryOp::Add, FpSize::S, 0, 4, 5).is_err());
    }

    #[test]
    fn flush_to_zero_flushes_inputs_and_outputs() {
        let mut cpu = cpu();
        cpu.fpcr = FPCR_FZ;
        cpu.vectors[1] = 1; // smallest positive single denormal
        s(&mut cpu, 2, 1.0);
        fp_binary(&mut cpu, FpBinaryOp::Add, FpSize::S, 0, 1, 2).unwrap();
        assert_eq!(get_s(&cpu, 0), 1.0);
        assert_eq!(cpu.fpsr & FPSR_IDC, FPSR_IDC);
        s(&mut cpu, 3, f32::MIN_POSITIVE);
        s(&mut cpu, 4, 0.5);
        fp_binary(&mut cpu, FpBinaryOp::Mul, FpSize::S, 0, 3, 4).unwrap();
        assert_eq!(cpu.vectors[0], 0);
    }

    #[test]
    fn min_max_order_signed_zeros_and_num_variants_ignore_one_quiet_nan() {
        let mut cpu = cpu();
        s(&mut cpu, 1, 0.0);
        s(&mut cpu, 2, -0.0);
        fp_binary(&mut cpu, FpBinaryOp::Max, FpSize::S, 0, 1, 2).unwrap();
        assert_eq!(cpu.vectors[0] as u32, 0);
        fp_binary(&mut cpu, FpBinaryOp::Min, FpSize::S, 0, 1, 2).unwrap();
        assert_eq!(cpu.vectors[0] as u32, 0x8000_0000);
        cpu.vectors[3] = 0x7fc0_0000;
        s(&mut cpu, 4, 5.0);
        fp_binary(&mut cpu, FpBinaryOp::MaxNum, FpSize::S, 0, 3, 4).unwrap();
        assert_eq!(get_s(&cpu, 0), 5.0);
        fp_binary(&mut cpu, FpBinaryOp::Max, FpSize::S, 0, 3, 4).unwrap();
        assert_eq!(cpu.vectors[0] as u32, 0x7fc0_0000);
    }

    #[test]
    fn compares_set_arm_flags() {
        let mut cpu = cpu();
        s(&mut cpu, 1, 1.0);
        s(&mut cpu, 2, 2.0);
        fp_compare(&mut cpu, FpSize::S, 1, Some(2), false).unwrap();
        assert_eq!(cpu.nzcv, 0b1000);
        fp_compare(&mut cpu, FpSize::S, 2, Some(1), false).unwrap();
        assert_eq!(cpu.nzcv, 0b0010);
        fp_compare(&mut cpu, FpSize::S, 1, Some(1), false).unwrap();
        assert_eq!(cpu.nzcv, 0b0110);
        cpu.vectors[3] = 0x7fc0_0000;
        fp_compare(&mut cpu, FpSize::S, 3, None, false).unwrap();
        assert_eq!((cpu.nzcv, cpu.fpsr), (0b0011, 0));
        fp_compare(&mut cpu, FpSize::S, 3, None, true).unwrap();
        assert_eq!((cpu.nzcv, cpu.fpsr), (0b0011, FPSR_IOC));
    }

    #[test]
    fn conversions_saturate_round_and_handle_nan() {
        let mut cpu = cpu();
        s(&mut cpu, 1, -2.5);
        for (rounding, expected) in [
            (Rounding::Nearest, -2i64),
            (Rounding::Plus, -2),
            (Rounding::Minus, -3),
            (Rounding::Zero, -2),
            (Rounding::Away, -3),
        ] {
            fp_to_int(&mut cpu, 0, 1, 64, FpSize::S, true, rounding).unwrap();
            assert_eq!(cpu.regs[0] as i64, expected, "{rounding:?}");
        }
        assert_eq!(cpu.fpsr, 0);
        fp_to_int(&mut cpu, 0, 1, 32, FpSize::S, false, Rounding::Zero).unwrap();
        assert_eq!((cpu.regs[0], cpu.fpsr), (0, FPSR_IOC));
        cpu.fpsr = 0;
        d(&mut cpu, 2, 1e300);
        fp_to_int(&mut cpu, 0, 2, 32, FpSize::D, true, Rounding::Zero).unwrap();
        assert_eq!((cpu.regs[0], cpu.fpsr), (i32::MAX as u32 as u64, FPSR_IOC));
        cpu.regs[3] = u64::MAX;
        int_to_fp(&mut cpu, 0, 3, 64, FpSize::S, false).unwrap();
        assert_eq!(get_s(&cpu, 0), 18_446_744_073_709_551_616.0);
        int_to_fp(&mut cpu, 0, 3, 32, FpSize::D, true).unwrap();
        assert_eq!(f64::from_bits(cpu.vectors[0] as u64), -1.0);
        // FCVT keeps NaN payload bits and quiets.
        cpu.vectors[4] = 0x7fa0_0000;
        fp_convert(&mut cpu, 0, 4, FpSize::S, FpSize::D).unwrap();
        assert_eq!(cpu.vectors[0] as u64, 0x7ffc_0000_0000_0000);
        d(&mut cpu, 5, 1e300);
        fp_convert(&mut cpu, 0, 5, FpSize::D, FpSize::S).unwrap();
        assert_eq!(get_s(&cpu, 0), f32::INFINITY);
    }

    #[test]
    fn moves_and_lane_operations() {
        let mut cpu = cpu();
        cpu.regs[1] = 0x1122_3344_5566_7788;
        fp_move_from_general(&mut cpu, 0, 1, 64, false);
        assert_eq!(cpu.vectors[0], 0x1122_3344_5566_7788);
        fp_move_from_general(&mut cpu, 0, 1, 64, true);
        assert_eq!(cpu.vectors[0], 0x1122_3344_5566_7788_1122_3344_5566_7788);
        fp_move_to_general(&mut cpu, 2, 0, 32, false);
        assert_eq!(cpu.regs[2], 0x5566_7788);
        insert_general(&mut cpu, 0, 31, 4, 1);
        assert_eq!(cpu.vectors[0], 0x1122_3344_5566_7788_0000_0000_5566_7788);
        insert_element(&mut cpu, 0, 0, 4, 3, 0);
        assert_eq!(cpu.vectors[0], 0x5566_7788_5566_7788_0000_0000_5566_7788);
        dup_element(&mut cpu, 3, 0, 4, 0, false, false);
        assert_eq!(cpu.vectors[3], 0x5566_7788_5566_7788);
        dup_general(&mut cpu, 3, 1, 2, true);
        assert_eq!(cpu.vectors[3], 0x7788_7788_7788_7788_7788_7788_7788_7788);
        move_to_general(&mut cpu, 4, 3, 2, 0, true, 64);
        assert_eq!(cpu.regs[4], 0x7788);
        cpu.vectors[5] = 0x8000;
        move_to_general(&mut cpu, 4, 5, 2, 0, true, 64);
        assert_eq!(cpu.regs[4], 0xffff_ffff_ffff_8000);
        vector_logical(&mut cpu, VectorLogicalOp::Orr, 6, 3, 3, false);
        assert_eq!(cpu.vectors[6], 0x7788_7788_7788_7788);
        modify_immediate(&mut cpu, 6, 0xff, true, false);
        assert_eq!(cpu.vectors[6], 0x7788_7788_7788_7700);
    }
}
