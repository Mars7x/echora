//! Scalar floating-point and Advanced SIMD execution helpers called by
//! generated code. Decoding lives in the recompiler crate.
//!
//! The operations themselves live in `echora-simd`, which generated code does
//! not depend on: each is declared here and reached through its exported
//! symbol, so changing how an operation works rebuilds that crate only.
#![allow(unsafe_code)]

use serde::Serialize;

use crate::Trap;
use crate::cpu::Cpu;
use crate::view::GuestView;

pub const FPCR_AH: u32 = 1 << 1;

pub const FPCR_FZ16: u32 = 1 << 19;

pub const FPCR_RMODE: u32 = 3 << 22;

pub const FPCR_FZ: u32 = 1 << 24;

pub const FPCR_DN: u32 = 1 << 25;

pub const FPCR_AHP: u32 = 1 << 26;

pub const FPSR_IOC: u32 = 1 << 0;

pub const FPSR_DZC: u32 = 1 << 1;

pub const FPSR_OFC: u32 = 1 << 2;

pub const FPSR_UFC: u32 = 1 << 3;

pub const FPSR_IXC: u32 = 1 << 4;

pub const FPSR_IDC: u32 = 1 << 7;

/// Cumulative saturation of the Advanced SIMD saturating instructions.
pub const FPSR_QC: u32 = 1 << 27;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum FpSize {
    S,
    D,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum FpUnaryOp {
    Mov,
    Abs,
    Neg,
    Sqrt,
    /// FRINT<r>: round to integral in floating-point format.
    Round(Rounding),
    /// FRINTX: as FRINTI, but signals inexact (not accumulated here).
    RoundExact,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum FpBinaryOp {
    Mul,
    Div,
    Add,
    Sub,
    Max,
    Min,
    MaxNum,
    MinNum,
    NegMul,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum FpTernaryOp {
    /// Rd = Ra + Rn*Rm
    MulAdd,
    /// Rd = Ra - Rn*Rm
    MulSub,
    /// Rd = -Ra - Rn*Rm
    NegMulAdd,
    /// Rd = -Ra + Rn*Rm
    NegMulSub,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Rounding {
    /// Ties to even.
    Nearest,
    /// Towards +infinity.
    Plus,
    /// Towards -infinity.
    Minus,
    /// Towards zero.
    Zero,
    /// Ties away from zero.
    Away,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum VectorLogicalOp {
    And,
    Bic,
    Orr,
    Orn,
    Eor,
    Bsl,
    Bit,
    Bif,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum VecFpOp {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
    MaxNum,
    MinNum,
    /// |a - b|
    AbsDiff,
    MulExtended,
    CmEq,
    CmGe,
    CmGt,
    AcGe,
    AcGt,
    /// d + n*m (fused)
    MulAdd,
    /// d - n*m (fused)
    MulSub,
    /// FRECPS: 2 - n*m (fused).
    RecipStep,
    /// FRSQRTS: (3 - n*m) / 2 (fused).
    RSqrtStep,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum VecIntOp {
    Add,
    Sub,
    Mul,
    CmEq,
    CmTst,
    CmGt,
    CmGe,
    CmHi,
    CmHs,
    Sshl,
    Ushl,
    Smax,
    Smin,
    Umax,
    Umin,
    /// Unsigned/signed absolute difference.
    Uabd,
    Sabd,
    /// Saturating add/subtract (FPSR.QC on saturation).
    Sqadd,
    Uqadd,
    Sqsub,
    Uqsub,
    /// Halving add/subtract, and rounding halving add.
    Shadd,
    Uhadd,
    Shsub,
    Uhsub,
    Srhadd,
    Urhadd,
    /// Saturating (rounding) doubling multiply returning the high half.
    Sqdmulh,
    Sqrdmulh,
    /// d + n*m and d - n*m.
    Mla,
    Mls,
    /// Register shifts by the signed low byte of Vm: rounding, saturating,
    /// and saturating rounding.
    Srshl,
    Urshl,
    Sqshl,
    Uqshl,
    Sqrshl,
    Uqrshl,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum VecUnaryOp {
    Cnt,
    Not,
    Rbit,
    Neg,
    Abs,
    /// Reverse lanes within each 64-, 32- or 16-bit container.
    Rev64,
    Rev32,
    Rev16,
    CmEqZero,
    CmGeZero,
    CmGtZero,
    CmLeZero,
    CmLtZero,
    /// Count leading zero bits / leading bits equal to the sign bit.
    Clz,
    Cls,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum VecFpUnaryOp {
    Abs,
    Neg,
    Sqrt,
    Round(Rounding),
    /// FCVT<r>S / FCVT<r>U with `fbits` fraction bits (0 for integer).
    ToSigned(Rounding),
    ToUnsigned(Rounding),
    FromSigned,
    FromUnsigned,
    CmEqZero,
    CmGeZero,
    CmGtZero,
    CmLeZero,
    CmLtZero,
    /// FRECPE and FRSQRTE.
    RecipEstimate,
    RSqrtEstimate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum VecAcrossOp {
    AddV,
    SMaxV,
    SMinV,
    UMaxV,
    UMinV,
    /// Widening sums: the result is twice the lane width.
    SAddLV,
    UAddLV,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum VecShiftOp {
    Shl,
    Sshr,
    Ushr,
    Ssra,
    Usra,
    /// Shift right and narrow to half-width lanes (SHRN/SHRN2).
    Shrn,
    /// Rounding shift right (and accumulate).
    Srshr,
    Urshr,
    Srsra,
    Ursra,
    /// Saturating shift left: signed, unsigned, signed to unsigned.
    Sqshl,
    Uqshl,
    Sqshlu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum VecLongOp {
    /// d += n*m (widening)
    Mlal,
    /// d -= n*m (widening)
    Mlsl,
    Mull,
    Addl,
    Subl,
    /// SADDW/UADDW, SSUBW/USUBW: Vn is already wide.
    Addw,
    Subw,
    /// SABAL/UABAL: d += |n - m| (widening)
    Abal,
    /// SABDL/UABDL: |n - m| (widening)
    Abdl,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum VecNarrowOp {
    /// Truncate (XTN, SHRN, RSHRN).
    Truncate,
    /// Signed to signed saturating (SQXTN, SQSHRN, SQRSHRN).
    Signed,
    /// Unsigned to unsigned saturating (UQXTN, UQSHRN, UQRSHRN).
    Unsigned,
    /// Signed to unsigned saturating (SQXTUN, SQSHRUN, SQRSHRUN).
    SignedToUnsigned,
}

/// Floating-point precisions for conversions that include half precision.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum FpPrecision {
    H,
    S,
    D,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum CryptoOp {
    Aese,
    Aesd,
    Aesmc,
    Aesimc,
    Sha1c,
    Sha1p,
    Sha1m,
    Sha1h,
    Sha1su0,
    Sha1su1,
    Sha256h,
    Sha256h2,
    Sha256su0,
    Sha256su1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum VecPermuteOp {
    Uzp1,
    Uzp2,
    Trn1,
    Trn2,
    Zip1,
    Zip2,
}

unsafe extern "Rust" {
    safe fn __echora_simd_fp_unary(
        cpu: &mut Cpu,
        op: FpUnaryOp,
        size: FpSize,
        rd: u8,
        rn: u8,
    ) -> Result<(), Trap>;
    safe fn __echora_simd_fp_convert(
        cpu: &mut Cpu,
        rd: u8,
        rn: u8,
        from: FpSize,
        to: FpSize,
    ) -> Result<(), Trap>;
    safe fn __echora_simd_fp_binary(
        cpu: &mut Cpu,
        op: FpBinaryOp,
        size: FpSize,
        rd: u8,
        rn: u8,
        rm: u8,
    ) -> Result<(), Trap>;
    #[allow(clippy::too_many_arguments)]
    safe fn __echora_simd_fp_ternary(
        cpu: &mut Cpu,
        op: FpTernaryOp,
        size: FpSize,
        rd: u8,
        rn: u8,
        rm: u8,
        ra: u8,
    ) -> Result<(), Trap>;
    safe fn __echora_simd_fp_compare(
        cpu: &mut Cpu,
        size: FpSize,
        rn: u8,
        rm: Option<u8>,
        signaling: bool,
    ) -> Result<(), Trap>;
    safe fn __echora_simd_fp_conditional_compare(
        cpu: &mut Cpu,
        size: FpSize,
        rn: u8,
        rm: u8,
        condition: u8,
        nzcv: u8,
        signaling: bool,
    ) -> Result<(), Trap>;
    safe fn __echora_simd_fp_select(
        cpu: &mut Cpu,
        size: FpSize,
        rd: u8,
        rn: u8,
        rm: u8,
        condition: u8,
    );
    safe fn __echora_simd_fp_move_immediate(cpu: &mut Cpu, size: FpSize, rd: u8, bits: u64);
    safe fn __echora_simd_fp_move_to_general(cpu: &mut Cpu, rd: u8, rn: u8, bits: u8, upper: bool);
    safe fn __echora_simd_fp_move_from_general(
        cpu: &mut Cpu,
        rd: u8,
        rn: u8,
        bits: u8,
        upper: bool,
    );
    safe fn __echora_simd_int_to_fp(
        cpu: &mut Cpu,
        rd: u8,
        rn: u8,
        gp_bits: u8,
        size: FpSize,
        signed: bool,
    ) -> Result<(), Trap>;
    safe fn __echora_simd_simd_int_to_fp(
        cpu: &mut Cpu,
        rd: u8,
        rn: u8,
        size: FpSize,
        signed: bool,
    ) -> Result<(), Trap>;
    #[allow(clippy::too_many_arguments)]
    safe fn __echora_simd_fp_to_int(
        cpu: &mut Cpu,
        rd: u8,
        rn: u8,
        gp_bits: u8,
        size: FpSize,
        signed: bool,
        rounding: Rounding,
    ) -> Result<(), Trap>;
    safe fn __echora_simd_move_immediate(cpu: &mut Cpu, rd: u8, value: u128);
    safe fn __echora_simd_modify_immediate(
        cpu: &mut Cpu,
        rd: u8,
        value: u128,
        clear: bool,
        q: bool,
    );
    safe fn __echora_simd_vector_logical(
        cpu: &mut Cpu,
        op: VectorLogicalOp,
        rd: u8,
        rn: u8,
        rm: u8,
        q: bool,
    );
    safe fn __echora_simd_insert_element(
        cpu: &mut Cpu,
        rd: u8,
        rn: u8,
        bytes: u8,
        dst: u8,
        src: u8,
    );
    safe fn __echora_simd_insert_general(cpu: &mut Cpu, rd: u8, rn: u8, bytes: u8, index: u8);
    safe fn __echora_simd_move_to_general(
        cpu: &mut Cpu,
        rd: u8,
        rn: u8,
        bytes: u8,
        index: u8,
        signed: bool,
        gp_bits: u8,
    );
    safe fn __echora_simd_dup_element(
        cpu: &mut Cpu,
        rd: u8,
        rn: u8,
        bytes: u8,
        index: u8,
        q: bool,
        scalar: bool,
    );
    safe fn __echora_simd_dup_general(cpu: &mut Cpu, rd: u8, rn: u8, bytes: u8, q: bool);
    safe fn __echora_simd_load(
        cpu: &mut Cpu,
        memory: &GuestView,
        address: u64,
        rt: u8,
        bytes: u8,
    ) -> Result<(), Trap>;
    safe fn __echora_simd_store(
        cpu: &Cpu,
        memory: &mut GuestView,
        address: u64,
        rt: u8,
        bytes: u8,
    ) -> Result<(), Trap>;
    safe fn __echora_simd_load_pair(
        cpu: &mut Cpu,
        memory: &GuestView,
        address: u64,
        rt: u8,
        rt2: u8,
        bytes: u8,
    ) -> Result<(), Trap>;
    safe fn __echora_simd_store_pair(
        cpu: &Cpu,
        memory: &mut GuestView,
        address: u64,
        rt: u8,
        rt2: u8,
        bytes: u8,
    ) -> Result<(), Trap>;
    #[allow(clippy::too_many_arguments)]
    safe fn __echora_simd_vec_fp(
        cpu: &mut Cpu,
        op: VecFpOp,
        size: FpSize,
        q: bool,
        scalar: bool,
        rd: u8,
        rn: u8,
        rm: u8,
    ) -> Result<(), Trap>;
    #[allow(clippy::too_many_arguments)]
    safe fn __echora_simd_vec_fp_pairwise(
        cpu: &mut Cpu,
        op: VecFpOp,
        size: FpSize,
        q: bool,
        scalar: bool,
        rd: u8,
        rn: u8,
        rm: u8,
    ) -> Result<(), Trap>;
    safe fn __echora_simd_vec_fp_across(
        cpu: &mut Cpu,
        op: VecFpOp,
        rd: u8,
        rn: u8,
    ) -> Result<(), Trap>;
    #[allow(clippy::too_many_arguments)]
    safe fn __echora_simd_vec_fp_element(
        cpu: &mut Cpu,
        op: VecFpOp,
        size: FpSize,
        q: bool,
        scalar: bool,
        rd: u8,
        rn: u8,
        rm: u8,
        index: u8,
    ) -> Result<(), Trap>;
    #[allow(clippy::too_many_arguments)]
    safe fn __echora_simd_vec_int_element(
        cpu: &mut Cpu,
        op: VecIntOp,
        bits: u8,
        q: bool,
        scalar: bool,
        rd: u8,
        rn: u8,
        rm: u8,
        index: u8,
    );
    #[allow(clippy::too_many_arguments)]
    safe fn __echora_simd_vec_int_pairwise(
        cpu: &mut Cpu,
        op: VecIntOp,
        bits: u8,
        q: bool,
        scalar: bool,
        rd: u8,
        rn: u8,
        rm: u8,
    );
    #[allow(clippy::too_many_arguments)]
    safe fn __echora_simd_vec_int(
        cpu: &mut Cpu,
        op: VecIntOp,
        bits: u8,
        q: bool,
        scalar: bool,
        rd: u8,
        rn: u8,
        rm: u8,
    );
    safe fn __echora_simd_vec_unary(
        cpu: &mut Cpu,
        op: VecUnaryOp,
        bits: u8,
        q: bool,
        scalar: bool,
        rd: u8,
        rn: u8,
    );
    #[allow(clippy::too_many_arguments)]
    safe fn __echora_simd_vec_fp_unary(
        cpu: &mut Cpu,
        op: VecFpUnaryOp,
        size: FpSize,
        q: bool,
        scalar: bool,
        rd: u8,
        rn: u8,
        fbits: u8,
    ) -> Result<(), Trap>;
    #[allow(clippy::too_many_arguments)]
    safe fn __echora_simd_fp_to_fixed(
        cpu: &mut Cpu,
        rd: u8,
        rn: u8,
        gp_bits: u8,
        size: FpSize,
        signed: bool,
        fbits: u8,
    ) -> Result<(), Trap>;
    safe fn __echora_simd_int_to_fp_fixed(
        cpu: &mut Cpu,
        rd: u8,
        rn: u8,
        gp_bits: u8,
        size: FpSize,
        signed: bool,
        fbits: u8,
    ) -> Result<(), Trap>;
    safe fn __echora_simd_vec_across(
        cpu: &mut Cpu,
        op: VecAcrossOp,
        bits: u8,
        q: bool,
        rd: u8,
        rn: u8,
    );
    #[allow(clippy::too_many_arguments)]
    safe fn __echora_simd_vec_shift(
        cpu: &mut Cpu,
        op: VecShiftOp,
        bits: u8,
        q: bool,
        scalar: bool,
        rd: u8,
        rn: u8,
        shift: u8,
    );
    safe fn __echora_simd_vec_shift_long(
        cpu: &mut Cpu,
        signed: bool,
        bits: u8,
        upper: bool,
        rd: u8,
        rn: u8,
        shift: u8,
    );
    #[allow(clippy::too_many_arguments)]
    safe fn __echora_simd_vec_long(
        cpu: &mut Cpu,
        op: VecLongOp,
        signed: bool,
        bits: u8,
        upper: bool,
        rd: u8,
        rn: u8,
        rm: u8,
    );
    #[allow(clippy::too_many_arguments)]
    safe fn __echora_simd_vec_long_element(
        cpu: &mut Cpu,
        op: VecLongOp,
        signed: bool,
        bits: u8,
        upper: bool,
        rd: u8,
        rn: u8,
        rm: u8,
        index: u8,
    );
    safe fn __echora_simd_vec_dot(cpu: &mut Cpu, signed: bool, q: bool, rd: u8, rn: u8, rm: u8);
    safe fn __echora_simd_vec_pairwise_long(
        cpu: &mut Cpu,
        signed: bool,
        accumulate: bool,
        bits: u8,
        q: bool,
        rd: u8,
        rn: u8,
    );
    #[allow(clippy::too_many_arguments)]
    safe fn __echora_simd_vec_narrow(
        cpu: &mut Cpu,
        op: VecNarrowOp,
        round: bool,
        bits: u8,
        upper: bool,
        scalar: bool,
        rd: u8,
        rn: u8,
        shift: u8,
    );
    safe fn __echora_simd_vec_table(
        cpu: &mut Cpu,
        q: bool,
        extend: bool,
        rd: u8,
        rn: u8,
        regs: u8,
        rm: u8,
    );
    safe fn __echora_simd_fp_convert_precision(
        cpu: &mut Cpu,
        rd: u8,
        rn: u8,
        from: FpPrecision,
        to: FpPrecision,
    ) -> Result<(), Trap>;
    safe fn __echora_simd_vec_fp_convert(
        cpu: &mut Cpu,
        from: FpPrecision,
        to: FpPrecision,
        upper: bool,
        rd: u8,
        rn: u8,
    ) -> Result<(), Trap>;
    safe fn __echora_simd_vec_poly_mul_long(
        cpu: &mut Cpu,
        bits: u8,
        upper: bool,
        rd: u8,
        rn: u8,
        rm: u8,
    );
    safe fn __echora_simd_crypto(cpu: &mut Cpu, op: CryptoOp, rd: u8, rn: u8, rm: u8);
    safe fn __echora_simd_vec_permute(
        cpu: &mut Cpu,
        op: VecPermuteOp,
        bits: u8,
        q: bool,
        rd: u8,
        rn: u8,
        rm: u8,
    );
    safe fn __echora_simd_vec_ext(cpu: &mut Cpu, q: bool, rd: u8, rn: u8, rm: u8, index: u8);
    #[allow(clippy::too_many_arguments)]
    safe fn __echora_simd_transfer_multiple(
        cpu: &mut Cpu,
        memory: &mut GuestView,
        address: u64,
        rt: u8,
        registers: u8,
        structure: u8,
        bits: u8,
        q: bool,
        load: bool,
    ) -> Result<u64, Trap>;
    #[allow(clippy::too_many_arguments)]
    safe fn __echora_simd_transfer_single(
        cpu: &mut Cpu,
        memory: &mut GuestView,
        address: u64,
        rt: u8,
        registers: u8,
        bits: u8,
        index: u8,
        load: bool,
    ) -> Result<u64, Trap>;
    #[allow(clippy::too_many_arguments)]
    safe fn __echora_simd_load_replicate(
        cpu: &mut Cpu,
        memory: &GuestView,
        address: u64,
        rt: u8,
        registers: u8,
        bits: u8,
        q: bool,
    ) -> Result<u64, Trap>;
}

#[inline]
pub fn fp_unary(cpu: &mut Cpu, op: FpUnaryOp, size: FpSize, rd: u8, rn: u8) -> Result<(), Trap> {
    __echora_simd_fp_unary(cpu, op, size, rd, rn)
}

#[inline]
pub fn fp_convert(cpu: &mut Cpu, rd: u8, rn: u8, from: FpSize, to: FpSize) -> Result<(), Trap> {
    __echora_simd_fp_convert(cpu, rd, rn, from, to)
}

#[inline]
pub fn fp_binary(
    cpu: &mut Cpu,
    op: FpBinaryOp,
    size: FpSize,
    rd: u8,
    rn: u8,
    rm: u8,
) -> Result<(), Trap> {
    __echora_simd_fp_binary(cpu, op, size, rd, rn, rm)
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub fn fp_ternary(
    cpu: &mut Cpu,
    op: FpTernaryOp,
    size: FpSize,
    rd: u8,
    rn: u8,
    rm: u8,
    ra: u8,
) -> Result<(), Trap> {
    __echora_simd_fp_ternary(cpu, op, size, rd, rn, rm, ra)
}

#[inline]
pub fn fp_compare(
    cpu: &mut Cpu,
    size: FpSize,
    rn: u8,
    rm: Option<u8>,
    signaling: bool,
) -> Result<(), Trap> {
    __echora_simd_fp_compare(cpu, size, rn, rm, signaling)
}

#[inline]
pub fn fp_conditional_compare(
    cpu: &mut Cpu,
    size: FpSize,
    rn: u8,
    rm: u8,
    condition: u8,
    nzcv: u8,
    signaling: bool,
) -> Result<(), Trap> {
    __echora_simd_fp_conditional_compare(cpu, size, rn, rm, condition, nzcv, signaling)
}

#[inline]
pub fn fp_select(cpu: &mut Cpu, size: FpSize, rd: u8, rn: u8, rm: u8, condition: u8) {
    __echora_simd_fp_select(cpu, size, rd, rn, rm, condition)
}

#[inline]
pub fn fp_move_immediate(cpu: &mut Cpu, size: FpSize, rd: u8, bits: u64) {
    __echora_simd_fp_move_immediate(cpu, size, rd, bits)
}

#[inline]
pub fn fp_move_to_general(cpu: &mut Cpu, rd: u8, rn: u8, bits: u8, upper: bool) {
    __echora_simd_fp_move_to_general(cpu, rd, rn, bits, upper)
}

#[inline]
pub fn fp_move_from_general(cpu: &mut Cpu, rd: u8, rn: u8, bits: u8, upper: bool) {
    __echora_simd_fp_move_from_general(cpu, rd, rn, bits, upper)
}

#[inline]
pub fn int_to_fp(
    cpu: &mut Cpu,
    rd: u8,
    rn: u8,
    gp_bits: u8,
    size: FpSize,
    signed: bool,
) -> Result<(), Trap> {
    __echora_simd_int_to_fp(cpu, rd, rn, gp_bits, size, signed)
}

#[inline]
pub fn simd_int_to_fp(
    cpu: &mut Cpu,
    rd: u8,
    rn: u8,
    size: FpSize,
    signed: bool,
) -> Result<(), Trap> {
    __echora_simd_simd_int_to_fp(cpu, rd, rn, size, signed)
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub fn fp_to_int(
    cpu: &mut Cpu,
    rd: u8,
    rn: u8,
    gp_bits: u8,
    size: FpSize,
    signed: bool,
    rounding: Rounding,
) -> Result<(), Trap> {
    __echora_simd_fp_to_int(cpu, rd, rn, gp_bits, size, signed, rounding)
}

#[inline]
pub fn move_immediate(cpu: &mut Cpu, rd: u8, value: u128) {
    __echora_simd_move_immediate(cpu, rd, value)
}

#[inline]
pub fn modify_immediate(cpu: &mut Cpu, rd: u8, value: u128, clear: bool, q: bool) {
    __echora_simd_modify_immediate(cpu, rd, value, clear, q)
}

#[inline]
pub fn vector_logical(cpu: &mut Cpu, op: VectorLogicalOp, rd: u8, rn: u8, rm: u8, q: bool) {
    __echora_simd_vector_logical(cpu, op, rd, rn, rm, q)
}

#[inline]
pub fn insert_element(cpu: &mut Cpu, rd: u8, rn: u8, bytes: u8, dst: u8, src: u8) {
    __echora_simd_insert_element(cpu, rd, rn, bytes, dst, src)
}

#[inline]
pub fn insert_general(cpu: &mut Cpu, rd: u8, rn: u8, bytes: u8, index: u8) {
    __echora_simd_insert_general(cpu, rd, rn, bytes, index)
}

#[inline]
pub fn move_to_general(
    cpu: &mut Cpu,
    rd: u8,
    rn: u8,
    bytes: u8,
    index: u8,
    signed: bool,
    gp_bits: u8,
) {
    __echora_simd_move_to_general(cpu, rd, rn, bytes, index, signed, gp_bits)
}

#[inline]
pub fn dup_element(cpu: &mut Cpu, rd: u8, rn: u8, bytes: u8, index: u8, q: bool, scalar: bool) {
    __echora_simd_dup_element(cpu, rd, rn, bytes, index, q, scalar)
}

#[inline]
pub fn dup_general(cpu: &mut Cpu, rd: u8, rn: u8, bytes: u8, q: bool) {
    __echora_simd_dup_general(cpu, rd, rn, bytes, q)
}

#[inline]
pub fn load(
    cpu: &mut Cpu,
    memory: &GuestView,
    address: u64,
    rt: u8,
    bytes: u8,
) -> Result<(), Trap> {
    __echora_simd_load(cpu, memory, address, rt, bytes)
}

#[inline]
pub fn store(
    cpu: &Cpu,
    memory: &mut GuestView,
    address: u64,
    rt: u8,
    bytes: u8,
) -> Result<(), Trap> {
    __echora_simd_store(cpu, memory, address, rt, bytes)
}

#[inline]
pub fn load_pair(
    cpu: &mut Cpu,
    memory: &GuestView,
    address: u64,
    rt: u8,
    rt2: u8,
    bytes: u8,
) -> Result<(), Trap> {
    __echora_simd_load_pair(cpu, memory, address, rt, rt2, bytes)
}

#[inline]
pub fn store_pair(
    cpu: &Cpu,
    memory: &mut GuestView,
    address: u64,
    rt: u8,
    rt2: u8,
    bytes: u8,
) -> Result<(), Trap> {
    __echora_simd_store_pair(cpu, memory, address, rt, rt2, bytes)
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub fn vec_fp(
    cpu: &mut Cpu,
    op: VecFpOp,
    size: FpSize,
    q: bool,
    scalar: bool,
    rd: u8,
    rn: u8,
    rm: u8,
) -> Result<(), Trap> {
    __echora_simd_vec_fp(cpu, op, size, q, scalar, rd, rn, rm)
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub fn vec_fp_pairwise(
    cpu: &mut Cpu,
    op: VecFpOp,
    size: FpSize,
    q: bool,
    scalar: bool,
    rd: u8,
    rn: u8,
    rm: u8,
) -> Result<(), Trap> {
    __echora_simd_vec_fp_pairwise(cpu, op, size, q, scalar, rd, rn, rm)
}

#[inline]
pub fn vec_fp_across(cpu: &mut Cpu, op: VecFpOp, rd: u8, rn: u8) -> Result<(), Trap> {
    __echora_simd_vec_fp_across(cpu, op, rd, rn)
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub fn vec_fp_element(
    cpu: &mut Cpu,
    op: VecFpOp,
    size: FpSize,
    q: bool,
    scalar: bool,
    rd: u8,
    rn: u8,
    rm: u8,
    index: u8,
) -> Result<(), Trap> {
    __echora_simd_vec_fp_element(cpu, op, size, q, scalar, rd, rn, rm, index)
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub fn vec_int_element(
    cpu: &mut Cpu,
    op: VecIntOp,
    bits: u8,
    q: bool,
    scalar: bool,
    rd: u8,
    rn: u8,
    rm: u8,
    index: u8,
) {
    __echora_simd_vec_int_element(cpu, op, bits, q, scalar, rd, rn, rm, index)
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub fn vec_int_pairwise(
    cpu: &mut Cpu,
    op: VecIntOp,
    bits: u8,
    q: bool,
    scalar: bool,
    rd: u8,
    rn: u8,
    rm: u8,
) {
    __echora_simd_vec_int_pairwise(cpu, op, bits, q, scalar, rd, rn, rm)
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub fn vec_int(
    cpu: &mut Cpu,
    op: VecIntOp,
    bits: u8,
    q: bool,
    scalar: bool,
    rd: u8,
    rn: u8,
    rm: u8,
) {
    __echora_simd_vec_int(cpu, op, bits, q, scalar, rd, rn, rm)
}

#[inline]
pub fn vec_unary(cpu: &mut Cpu, op: VecUnaryOp, bits: u8, q: bool, scalar: bool, rd: u8, rn: u8) {
    __echora_simd_vec_unary(cpu, op, bits, q, scalar, rd, rn)
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub fn vec_fp_unary(
    cpu: &mut Cpu,
    op: VecFpUnaryOp,
    size: FpSize,
    q: bool,
    scalar: bool,
    rd: u8,
    rn: u8,
    fbits: u8,
) -> Result<(), Trap> {
    __echora_simd_vec_fp_unary(cpu, op, size, q, scalar, rd, rn, fbits)
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub fn fp_to_fixed(
    cpu: &mut Cpu,
    rd: u8,
    rn: u8,
    gp_bits: u8,
    size: FpSize,
    signed: bool,
    fbits: u8,
) -> Result<(), Trap> {
    __echora_simd_fp_to_fixed(cpu, rd, rn, gp_bits, size, signed, fbits)
}

#[inline]
pub fn int_to_fp_fixed(
    cpu: &mut Cpu,
    rd: u8,
    rn: u8,
    gp_bits: u8,
    size: FpSize,
    signed: bool,
    fbits: u8,
) -> Result<(), Trap> {
    __echora_simd_int_to_fp_fixed(cpu, rd, rn, gp_bits, size, signed, fbits)
}

#[inline]
pub fn vec_across(cpu: &mut Cpu, op: VecAcrossOp, bits: u8, q: bool, rd: u8, rn: u8) {
    __echora_simd_vec_across(cpu, op, bits, q, rd, rn)
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub fn vec_shift(
    cpu: &mut Cpu,
    op: VecShiftOp,
    bits: u8,
    q: bool,
    scalar: bool,
    rd: u8,
    rn: u8,
    shift: u8,
) {
    __echora_simd_vec_shift(cpu, op, bits, q, scalar, rd, rn, shift)
}

#[inline]
pub fn vec_shift_long(
    cpu: &mut Cpu,
    signed: bool,
    bits: u8,
    upper: bool,
    rd: u8,
    rn: u8,
    shift: u8,
) {
    __echora_simd_vec_shift_long(cpu, signed, bits, upper, rd, rn, shift)
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub fn vec_long(
    cpu: &mut Cpu,
    op: VecLongOp,
    signed: bool,
    bits: u8,
    upper: bool,
    rd: u8,
    rn: u8,
    rm: u8,
) {
    __echora_simd_vec_long(cpu, op, signed, bits, upper, rd, rn, rm)
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub fn vec_long_element(
    cpu: &mut Cpu,
    op: VecLongOp,
    signed: bool,
    bits: u8,
    upper: bool,
    rd: u8,
    rn: u8,
    rm: u8,
    index: u8,
) {
    __echora_simd_vec_long_element(cpu, op, signed, bits, upper, rd, rn, rm, index)
}

#[inline]
pub fn vec_dot(cpu: &mut Cpu, signed: bool, q: bool, rd: u8, rn: u8, rm: u8) {
    __echora_simd_vec_dot(cpu, signed, q, rd, rn, rm)
}

#[inline]
pub fn vec_pairwise_long(
    cpu: &mut Cpu,
    signed: bool,
    accumulate: bool,
    bits: u8,
    q: bool,
    rd: u8,
    rn: u8,
) {
    __echora_simd_vec_pairwise_long(cpu, signed, accumulate, bits, q, rd, rn)
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub fn vec_narrow(
    cpu: &mut Cpu,
    op: VecNarrowOp,
    round: bool,
    bits: u8,
    upper: bool,
    scalar: bool,
    rd: u8,
    rn: u8,
    shift: u8,
) {
    __echora_simd_vec_narrow(cpu, op, round, bits, upper, scalar, rd, rn, shift)
}

#[inline]
pub fn vec_table(cpu: &mut Cpu, q: bool, extend: bool, rd: u8, rn: u8, regs: u8, rm: u8) {
    __echora_simd_vec_table(cpu, q, extend, rd, rn, regs, rm)
}

#[inline]
pub fn fp_convert_precision(
    cpu: &mut Cpu,
    rd: u8,
    rn: u8,
    from: FpPrecision,
    to: FpPrecision,
) -> Result<(), Trap> {
    __echora_simd_fp_convert_precision(cpu, rd, rn, from, to)
}

#[inline]
pub fn vec_fp_convert(
    cpu: &mut Cpu,
    from: FpPrecision,
    to: FpPrecision,
    upper: bool,
    rd: u8,
    rn: u8,
) -> Result<(), Trap> {
    __echora_simd_vec_fp_convert(cpu, from, to, upper, rd, rn)
}

#[inline]
pub fn vec_poly_mul_long(cpu: &mut Cpu, bits: u8, upper: bool, rd: u8, rn: u8, rm: u8) {
    __echora_simd_vec_poly_mul_long(cpu, bits, upper, rd, rn, rm)
}

#[inline]
pub fn crypto(cpu: &mut Cpu, op: CryptoOp, rd: u8, rn: u8, rm: u8) {
    __echora_simd_crypto(cpu, op, rd, rn, rm)
}

#[inline]
pub fn vec_permute(cpu: &mut Cpu, op: VecPermuteOp, bits: u8, q: bool, rd: u8, rn: u8, rm: u8) {
    __echora_simd_vec_permute(cpu, op, bits, q, rd, rn, rm)
}

#[inline]
pub fn vec_ext(cpu: &mut Cpu, q: bool, rd: u8, rn: u8, rm: u8, index: u8) {
    __echora_simd_vec_ext(cpu, q, rd, rn, rm, index)
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub fn transfer_multiple(
    cpu: &mut Cpu,
    memory: &mut GuestView,
    address: u64,
    rt: u8,
    registers: u8,
    structure: u8,
    bits: u8,
    q: bool,
    load: bool,
) -> Result<u64, Trap> {
    __echora_simd_transfer_multiple(
        cpu, memory, address, rt, registers, structure, bits, q, load,
    )
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub fn transfer_single(
    cpu: &mut Cpu,
    memory: &mut GuestView,
    address: u64,
    rt: u8,
    registers: u8,
    bits: u8,
    index: u8,
    load: bool,
) -> Result<u64, Trap> {
    __echora_simd_transfer_single(cpu, memory, address, rt, registers, bits, index, load)
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub fn load_replicate(
    cpu: &mut Cpu,
    memory: &GuestView,
    address: u64,
    rt: u8,
    registers: u8,
    bits: u8,
    q: bool,
) -> Result<u64, Trap> {
    __echora_simd_load_replicate(cpu, memory, address, rt, registers, bits, q)
}
