//! Register def/use and side effects of decoded instructions.
//!
//! [`effects`] records, for every [`Instruction`], which general registers,
//! SP, NZCV and SIMD registers it reads and writes, and whether it loads,
//! stores, calls, makes an SVC, touches a system register, is a barrier or
//! waits on events. Both matches are exhaustive: a new decoder variant fails
//! to compile here until its effects are spelled out.
//!
//! The sets are exact for general registers, SP and NZCV. SIMD register sets
//! are conservative: an instruction that writes part of a vector register,
//! or might accumulate into it, also reads it. Implicit FPCR reads and FPSR
//! cumulative-flag writes of floating-point arithmetic are not recorded.

use crate::aarch64::{Address, BitfieldOp, CompareOperand, Instruction, LogicalOp, MemoryOp};
use crate::aarch64::{WideOp, Writeback};
use crate::simd::{CryptoOp, SimdOp};

/// A set of guest registers. Register number 31 is SP or the zero register
/// depending on the operand; the zero register is never recorded.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Regs {
    /// Bit `n` is X`n` (W`n`), for `n` in 0..31.
    pub gpr: u32,
    pub sp: bool,
    pub nzcv: bool,
    /// Bit `n` is V`n` (Q/D/S/H/B`n`).
    pub vector: u32,
}

impl Regs {
    pub const NONE: Self = Self {
        gpr: 0,
        sp: false,
        nzcv: false,
        vector: 0,
    };
    pub const ALL: Self = Self {
        gpr: 0x7fff_ffff,
        sp: true,
        nzcv: true,
        vector: u32::MAX,
    };

    pub fn is_empty(self) -> bool {
        self == Self::NONE
    }

    pub fn union(self, other: Self) -> Self {
        Self {
            gpr: self.gpr | other.gpr,
            sp: self.sp | other.sp,
            nzcv: self.nzcv | other.nzcv,
            vector: self.vector | other.vector,
        }
    }

    pub fn intersection(self, other: Self) -> Self {
        Self {
            gpr: self.gpr & other.gpr,
            sp: self.sp & other.sp,
            nzcv: self.nzcv & other.nzcv,
            vector: self.vector & other.vector,
        }
    }

    pub fn difference(self, other: Self) -> Self {
        Self {
            gpr: self.gpr & !other.gpr,
            sp: self.sp & !other.sp,
            nzcv: self.nzcv & !other.nzcv,
            vector: self.vector & !other.vector,
        }
    }

    pub fn intersects(self, other: Self) -> bool {
        !self.intersection(other).is_empty()
    }

    /// Whether X`n` is in the set (`n` = 31 is the zero register: never).
    pub fn has_gpr(self, n: u8) -> bool {
        n < 31 && self.gpr & (1 << n) != 0
    }

    /// Whether register operand `n` of a base-address or other SP-capable
    /// field is in the set: 31 means SP.
    pub fn has_gpr_or_sp(self, n: u8) -> bool {
        if n == 31 { self.sp } else { self.has_gpr(n) }
    }

    /// Whether V`n` is in the set.
    pub fn has_vector(self, n: u8) -> bool {
        self.vector & (1 << (n & 31)) != 0
    }

    /// X`n`, with 31 as the zero register (nothing).
    pub fn gpr(n: u8) -> Self {
        let mut regs = Self::NONE;
        regs.add_gpr(n);
        regs
    }

    /// X`n`, with 31 as SP.
    pub fn gpr_or_sp(n: u8) -> Self {
        let mut regs = Self::NONE;
        regs.add_gpr_or_sp(n);
        regs
    }

    fn add_gpr(&mut self, n: u8) {
        if n < 31 {
            self.gpr |= 1 << n;
        }
    }

    fn add_gpr_or_sp(&mut self, n: u8) {
        if n == 31 {
            self.sp = true;
        } else {
            self.add_gpr(n);
        }
    }

    fn add_vector(&mut self, n: u8) {
        self.vector |= 1 << (n & 31);
    }

    /// `count` consecutive vector registers from V`first`, wrapping at 32.
    fn add_vectors(&mut self, first: u8, count: u8) {
        for i in 0..count {
            self.add_vector(first.wrapping_add(i));
        }
    }
}

impl std::ops::BitOr for Regs {
    type Output = Self;
    fn bitor(self, other: Self) -> Self {
        self.union(other)
    }
}

/// What one instruction reads, writes and does besides register arithmetic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Effects {
    pub reads: Regs,
    pub writes: Regs,
    /// Reads guest memory.
    pub loads: bool,
    /// Writes guest memory (including exclusive and atomic stores).
    pub stores: bool,
    /// A single-instruction read-modify-write of memory (LSE CAS, LD<op>,
    /// ST<op>, SWP).
    pub atomic: bool,
    /// Uses the exclusive monitor (LDXR, STXR, CLREX).
    pub exclusive: bool,
    /// May transfer control somewhere other than the next instruction.
    pub branch: bool,
    /// BL or BLR.
    pub calls: bool,
    pub svc: bool,
    /// Reads or writes a system register, or performs a system operation
    /// (cache maintenance). MRS/MSR NZCV only touch NZCV and are not counted.
    pub system: bool,
    /// DMB, DSB or ISB.
    pub barrier: bool,
    /// YIELD or WFE.
    pub event_wait: bool,
    /// Stops translated execution (UDF or an unsupported encoding).
    pub traps: bool,
}

impl Effects {
    fn read(&mut self, n: u8) {
        self.reads.add_gpr(n);
    }
    fn read_sp(&mut self, n: u8) {
        self.reads.add_gpr_or_sp(n);
    }
    fn write(&mut self, n: u8) {
        self.writes.add_gpr(n);
    }
    fn write_sp(&mut self, n: u8) {
        self.writes.add_gpr_or_sp(n);
    }
    fn read_vector(&mut self, n: u8) {
        self.reads.add_vector(n);
    }
    fn write_vector(&mut self, n: u8) {
        self.writes.add_vector(n);
    }

    /// The base register (SP-capable), any writeback and any offset register.
    fn address(&mut self, rn: u8, address: Address) {
        self.read_sp(rn);
        match address {
            Address::Immediate { writeback, .. } => self.writeback(rn, writeback),
            Address::Register { rm, .. } => self.read(rm),
        }
    }

    fn writeback(&mut self, rn: u8, writeback: Writeback) {
        if writeback != Writeback::None {
            self.write_sp(rn);
        }
    }

    /// A post-indexed structure transfer: Rm (31 = immediate) advances Xn|SP.
    fn post_index(&mut self, rn: u8, post: Option<u8>) {
        if let Some(rm) = post {
            self.write_sp(rn);
            self.read(rm);
        }
    }

    fn memory(&mut self, load: bool) {
        if load {
            self.loads = true;
        } else {
            self.stores = true;
        }
    }

    /// A vector instruction that may read Vd (merging or accumulating).
    fn vector_op(&mut self, rd: u8, sources: &[u8]) {
        for &source in sources {
            self.read_vector(source);
        }
        self.read_vector(rd);
        self.write_vector(rd);
    }
}

/// The register def/use and side effects of `instruction`.
pub fn effects(instruction: &Instruction) -> Effects {
    use Instruction::*;
    let mut e = Effects::default();
    match *instruction {
        Nop => {}
        SpinHint => e.event_wait = true,
        // Every key is disabled: these leave registers unchanged.
        PAuthDisabledHint { .. } | PAuthDisabledData => {}
        PAuthDisabledReturn { .. } => {
            e.branch = true;
            e.read(30);
            e.reads.sp = true;
        }
        StripLrInstructionPac => {
            e.read(30);
            e.write(30);
        }
        Barrier => e.barrier = true,
        ClearExclusive => e.exclusive = true,
        Adr { rd, .. } => e.write(rd),
        Branch { link, .. } => {
            e.branch = true;
            if link {
                e.calls = true;
                e.write(30);
            }
        }
        BranchCond { .. } => {
            e.branch = true;
            e.reads.nzcv = true;
        }
        CompareBranch { rt, .. } | TestBranch { rt, .. } => {
            e.branch = true;
            e.read(rt);
        }
        BranchReg { rn, link } => {
            e.branch = true;
            e.read(rn);
            if link {
                e.calls = true;
                e.write(30);
            }
        }
        Svc { .. } => {
            // Supervisor calls take arguments in and return results in X0-X7.
            e.svc = true;
            for n in 0..8 {
                e.read(n);
                e.write(n);
            }
        }
        MoveWide { rd, op, .. } => {
            if op == WideOp::Keep {
                e.read(rd);
            }
            e.write(rd);
        }
        AddSubImmediate { rd, rn, flags, .. } => {
            e.read_sp(rn);
            if flags {
                e.write(rd);
                e.writes.nzcv = true;
            } else {
                e.write_sp(rd);
            }
        }
        AddSubShifted {
            rd, rn, rm, flags, ..
        } => {
            e.read(rn);
            e.read(rm);
            e.write(rd);
            e.writes.nzcv = flags;
        }
        AddSubExtended {
            rd, rn, rm, flags, ..
        } => {
            e.read_sp(rn);
            e.read(rm);
            if flags {
                e.write(rd);
                e.writes.nzcv = true;
            } else {
                e.write_sp(rd);
            }
        }
        AddCarry {
            rd, rn, rm, flags, ..
        } => {
            e.read(rn);
            e.read(rm);
            e.reads.nzcv = true;
            e.write(rd);
            e.writes.nzcv = flags;
        }
        LogicalShifted { rd, rn, rm, op, .. } => {
            e.read(rn);
            e.read(rm);
            e.write(rd);
            e.writes.nzcv = op == LogicalOp::Ands;
        }
        LogicalImmediate { rd, rn, op, .. } => {
            e.read(rn);
            if op == LogicalOp::Ands {
                e.write(rd);
                e.writes.nzcv = true;
            } else {
                e.write_sp(rd);
            }
        }
        Bitfield { rd, rn, op, .. } => {
            e.read(rn);
            if op == BitfieldOp::Merge {
                e.read(rd);
            }
            e.write(rd);
        }
        Extract { rd, rn, rm, .. }
        | MultiplyHigh { rd, rn, rm, .. }
        | Divide { rd, rn, rm, .. }
        | VariableShift { rd, rn, rm, .. }
        | Crc32 { rd, rn, rm, .. } => {
            e.read(rn);
            e.read(rm);
            e.write(rd);
        }
        Select { rd, rn, rm, .. } => {
            e.read(rn);
            e.read(rm);
            e.reads.nzcv = true;
            e.write(rd);
        }
        ConditionalCompare { rn, rhs, .. } | ConditionalCompareNegative { rn, rhs, .. } => {
            e.read(rn);
            if let CompareOperand::Register(rm) = rhs {
                e.read(rm);
            }
            e.reads.nzcv = true;
            e.writes.nzcv = true;
        }
        MultiplyAdd { rd, rn, rm, ra, .. }
        | UnsignedMultiplyAddLong { rd, rn, rm, ra }
        | MultiplyAddLong { rd, rn, rm, ra, .. } => {
            e.read(rn);
            e.read(rm);
            e.read(ra);
            e.write(rd);
        }
        CountLeading { rd, rn, .. } | ReverseBits { rd, rn, .. } | ReverseBytes { rd, rn, .. } => {
            e.read(rn);
            e.write(rd);
        }
        CompareAndSwap { rs, rt, rn, .. } => {
            e.read(rs);
            e.read(rt);
            e.read_sp(rn);
            e.write(rs);
            e.loads = true;
            e.stores = true;
            e.atomic = true;
        }
        CompareAndSwapPair { rs, rt, rn, .. } => {
            e.read(rs);
            e.read(rs + 1);
            e.read(rt);
            e.read(rt + 1);
            e.read_sp(rn);
            e.write(rs);
            e.write(rs + 1);
            e.loads = true;
            e.stores = true;
            e.atomic = true;
        }
        AtomicAdd { rs, rt, rn, .. } | AtomicMemory { rs, rt, rn, .. } => {
            e.read(rs);
            e.read_sp(rn);
            e.write(rt);
            e.loads = true;
            e.stores = true;
            e.atomic = true;
        }
        LoadAcquireRcpc { rt, rn, .. } | LoadAcquireRcsc { rt, rn, .. } => {
            e.read_sp(rn);
            e.write(rt);
            e.loads = true;
        }
        LoadExclusive { rt, rn, .. } => {
            e.read_sp(rn);
            e.write(rt);
            e.loads = true;
            e.exclusive = true;
        }
        StoreExclusive { rs, rt, rn, .. } => {
            e.read(rt);
            e.read_sp(rn);
            e.write(rs);
            e.stores = true;
            e.exclusive = true;
        }
        StoreRelease { rt, rn, .. } => {
            e.read(rt);
            e.read_sp(rn);
            e.stores = true;
        }
        LoadStore {
            rt,
            rn,
            op,
            address,
            ..
        } => {
            e.address(rn, address);
            match op {
                MemoryOp::Store => {
                    e.read(rt);
                    e.stores = true;
                }
                MemoryOp::Load { .. } => {
                    e.write(rt);
                    e.loads = true;
                }
            }
        }
        LoadStorePair {
            rt,
            rt2,
            rn,
            op,
            writeback,
            ..
        } => {
            e.read_sp(rn);
            e.writeback(rn, writeback);
            match op {
                MemoryOp::Store => {
                    e.read(rt);
                    e.read(rt2);
                    e.stores = true;
                }
                MemoryOp::Load { .. } => {
                    e.write(rt);
                    e.write(rt2);
                    e.loads = true;
                }
            }
        }
        LoadStoreVectorPair {
            rt,
            rt2,
            rn,
            load,
            writeback,
            ..
        } => {
            e.read_sp(rn);
            e.writeback(rn, writeback);
            vector_pair(&mut e, rt, rt2, load);
        }
        LoadLiteral { rt, .. } => {
            e.write(rt);
            e.loads = true;
        }
        ReadPhysicalCounter { rt }
        | ReadThreadPointer { rt, .. }
        | ReadFpControl { rt, .. }
        | ReadCacheType { rt } => {
            e.write(rt);
            e.system = true;
        }
        WriteThreadPointer { rt } | WriteFpControl { rt, .. } => {
            e.read(rt);
            e.system = true;
        }
        Simd(ref op) => simd(&mut e, op),
        StripPac { rd } => {
            e.read(rd);
            e.write(rd);
        }
        PacGeneric { rd, rn, rm } => {
            e.read(rn);
            e.read_sp(rm);
            e.write(rd);
        }
        ReadNzcv { rt } => {
            e.reads.nzcv = true;
            e.write(rt);
        }
        WriteNzcv { rt } => {
            e.read(rt);
            e.writes.nzcv = true;
        }
        // DC/IC take an address register the decoder does not keep.
        CacheMaintenance => e.system = true,
        Undefined => {
            e.traps = true;
            e.reads = Regs::ALL;
        }
        Unsupported => {
            e.traps = true;
            e.reads = Regs::ALL;
            e.writes = Regs::ALL;
        }
    }
    e
}

fn vector_pair(e: &mut Effects, rt: u8, rt2: u8, load: bool) {
    if load {
        e.write_vector(rt);
        e.write_vector(rt2);
    } else {
        e.read_vector(rt);
        e.read_vector(rt2);
    }
    e.memory(load);
}

fn simd(e: &mut Effects, op: &SimdOp) {
    use SimdOp::*;
    match *op {
        LoadStore {
            rt,
            rn,
            load,
            address,
            ..
        } => {
            e.address(rn, address);
            if load {
                e.write_vector(rt);
            } else {
                e.read_vector(rt);
            }
            e.memory(load);
        }
        LoadStorePair {
            rt,
            rt2,
            rn,
            load,
            writeback,
            ..
        } => {
            e.read_sp(rn);
            e.writeback(rn, writeback);
            vector_pair(e, rt, rt2, load);
        }
        LoadLiteral { rt, .. } => {
            e.write_vector(rt);
            e.loads = true;
        }
        // Scalar results zero the rest of Vd.
        FpUnary { rd, rn, .. }
        | FpConvert { rd, rn, .. }
        | SimdIntToFp { rd, rn, .. }
        | DupElement { rd, rn, .. } => {
            e.read_vector(rn);
            e.write_vector(rd);
        }
        FpBinary { rd, rn, rm, .. } => {
            e.read_vector(rn);
            e.read_vector(rm);
            e.write_vector(rd);
        }
        FpTernary { rd, rn, rm, ra, .. } => {
            e.read_vector(rn);
            e.read_vector(rm);
            e.read_vector(ra);
            e.write_vector(rd);
        }
        FpCompare { rn, rm, .. } => {
            e.read_vector(rn);
            if let Some(rm) = rm {
                e.read_vector(rm);
            }
            e.writes.nzcv = true;
        }
        FpConditionalCompare { rn, rm, .. } => {
            e.read_vector(rn);
            e.read_vector(rm);
            e.reads.nzcv = true;
            e.writes.nzcv = true;
        }
        FpSelect { rd, rn, rm, .. } => {
            e.read_vector(rn);
            e.read_vector(rm);
            e.reads.nzcv = true;
            e.write_vector(rd);
        }
        FpMoveImmediate { rd, .. } | MoveImmediate { rd, .. } => e.write_vector(rd),
        FpMoveToGeneral { rd, rn, .. }
        | FpToInt { rd, rn, .. }
        | FpToFixed { rd, rn, .. }
        | MoveToGeneral { rd, rn, .. } => {
            e.read_vector(rn);
            e.write(rd);
        }
        FpMoveFromGeneral { rd, rn, upper, .. } => {
            e.read(rn);
            if upper {
                e.read_vector(rd);
            }
            e.write_vector(rd);
        }
        IntToFp { rd, rn, .. } | IntToFpFixed { rd, rn, .. } | DupGeneral { rd, rn, .. } => {
            e.read(rn);
            e.write_vector(rd);
        }
        ModifyImmediate { rd, .. } => e.vector_op(rd, &[]),
        InsertGeneral { rd, rn, .. } => {
            e.read(rn);
            e.vector_op(rd, &[]);
        }
        VectorLogical { rd, rn, rm, .. }
        | VecFp { rd, rn, rm, .. }
        | VecFpPairwise { rd, rn, rm, .. }
        | VecIntPairwise { rd, rn, rm, .. }
        | VecIntElement { rd, rn, rm, .. }
        | VecLongElement { rd, rn, rm, .. }
        | VecPolyMulLong { rd, rn, rm, .. }
        | VecFpElement { rd, rn, rm, .. }
        | VecInt { rd, rn, rm, .. }
        | VecDot { rd, rn, rm, .. }
        | VecLong { rd, rn, rm, .. }
        | VecPermute { rd, rn, rm, .. }
        | VecExt { rd, rn, rm, .. } => e.vector_op(rd, &[rn, rm]),
        Crypto { op, rd, rn, rm } => {
            // The two-register forms leave the Rm field unused.
            let three_registers = match op {
                CryptoOp::Aese
                | CryptoOp::Aesd
                | CryptoOp::Aesmc
                | CryptoOp::Aesimc
                | CryptoOp::Sha1h
                | CryptoOp::Sha1su1
                | CryptoOp::Sha256su0 => false,
                CryptoOp::Sha1c
                | CryptoOp::Sha1p
                | CryptoOp::Sha1m
                | CryptoOp::Sha1su0
                | CryptoOp::Sha256h
                | CryptoOp::Sha256h2
                | CryptoOp::Sha256su1 => true,
            };
            if three_registers {
                e.vector_op(rd, &[rn, rm]);
            } else {
                e.vector_op(rd, &[rn]);
            }
        }
        InsertElement { rd, rn, .. }
        | VecNarrow { rd, rn, .. }
        | VecFpConvert { rd, rn, .. }
        | FpConvertPrecision { rd, rn, .. }
        | VecUnary { rd, rn, .. }
        | VecFpUnary { rd, rn, .. }
        | VecAcross { rd, rn, .. }
        | VecFpAcross { rd, rn, .. }
        | VecShift { rd, rn, .. }
        | VecShiftLong { rd, rn, .. }
        | VecPairwiseLong { rd, rn, .. } => e.vector_op(rd, &[rn]),
        VecTable {
            rd, rn, regs, rm, ..
        } => {
            e.reads.add_vectors(rn, regs);
            e.vector_op(rd, &[rm]);
        }
        TransferMultiple {
            rt,
            rn,
            registers,
            load,
            post,
            ..
        } => {
            e.read_sp(rn);
            e.post_index(rn, post);
            if load {
                e.writes.add_vectors(rt, registers);
            } else {
                e.reads.add_vectors(rt, registers);
            }
            e.memory(load);
        }
        TransferSingle {
            rt,
            rn,
            registers,
            load,
            post,
            ..
        } => {
            e.read_sp(rn);
            e.post_index(rn, post);
            // A single lane: loads keep the other lanes.
            e.reads.add_vectors(rt, registers);
            if load {
                e.writes.add_vectors(rt, registers);
            }
            e.memory(load);
        }
        LoadReplicate {
            rt,
            rn,
            registers,
            post,
            ..
        } => {
            e.read_sp(rn);
            e.post_index(rn, post);
            e.writes.add_vectors(rt, registers);
            e.loads = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aarch64::decode;

    const SP: Regs = Regs {
        sp: true,
        ..Regs::NONE
    };
    const NZCV: Regs = Regs {
        nzcv: true,
        ..Regs::NONE
    };
    const N: Regs = Regs::NONE;

    fn x(registers: &[u8]) -> Regs {
        registers
            .iter()
            .fold(Regs::NONE, |set, &n| set | Regs::gpr(n))
    }

    fn v(registers: &[u8]) -> Regs {
        let vector = registers.iter().fold(0, |set, &n| set | 1 << n);
        Regs {
            vector,
            ..Regs::NONE
        }
    }

    fn rw(reads: Regs, writes: Regs) -> Effects {
        Effects {
            reads,
            writes,
            ..Effects::default()
        }
    }

    fn load(reads: Regs, writes: Regs) -> Effects {
        Effects {
            loads: true,
            ..rw(reads, writes)
        }
    }

    fn store(reads: Regs, writes: Regs) -> Effects {
        Effects {
            stores: true,
            ..rw(reads, writes)
        }
    }

    fn branch(reads: Regs) -> Effects {
        Effects {
            branch: true,
            ..rw(reads, N)
        }
    }

    fn system(reads: Regs, writes: Regs) -> Effects {
        Effects {
            system: true,
            ..rw(reads, writes)
        }
    }

    /// Each word must decode to a supported instruction with these effects.
    fn check(cases: &[(u32, Effects)]) {
        for &(word, expected) in cases {
            let instruction = decode(word, 0x1000);
            assert_ne!(instruction, Instruction::Unsupported, "{word:#010x}");
            assert_eq!(
                effects(&instruction),
                expected,
                "{word:#010x} {instruction:?}"
            );
        }
    }

    #[test]
    fn regs_sets() {
        assert!(Regs::gpr(31).is_empty());
        assert_eq!(Regs::gpr_or_sp(31), SP);
        let set = x(&[1, 30]) | SP | v(&[0, 31]);
        assert!(set.has_gpr(1) && set.has_gpr(30) && !set.has_gpr(31));
        assert!(set.has_gpr_or_sp(31) && set.has_vector(31) && !set.has_vector(1));
        assert_eq!(
            set.difference(x(&[1])).intersection(set),
            x(&[30]) | SP | v(&[0, 31])
        );
        assert!(set.intersects(SP) && !set.intersects(NZCV));
        assert_eq!(Regs::ALL.difference(Regs::ALL), Regs::NONE);
    }

    #[test]
    fn hints_barriers_and_pointer_authentication() {
        check(&[
            (0xd503_201f, rw(N, N)), // nop
            (
                0xd503_203f, // yield
                Effects {
                    event_wait: true,
                    ..rw(N, N)
                },
            ),
            (
                0xd503_205f, // wfe
                Effects {
                    event_wait: true,
                    ..rw(N, N)
                },
            ),
            (0xd503_233f, rw(N, N)),               // paciasp
            (0xd65f_0bff, branch(x(&[30]) | SP)),  // retaa
            (0xd503_20ff, rw(x(&[30]), x(&[30]))), // xpaclri
            (
                0xd503_3bbf, // dmb ish
                Effects {
                    barrier: true,
                    ..rw(N, N)
                },
            ),
            (
                0xd503_3f5f, // clrex
                Effects {
                    exclusive: true,
                    ..rw(N, N)
                },
            ),
            (0xdac1_43e1, rw(x(&[1]), x(&[1]))),      // xpaci x1
            (0x9adf_3041, rw(x(&[2]) | SP, x(&[1]))), // pacga x1, x2, sp
            (0xdac1_0041, rw(N, N)),                  // pacia x1, x2
        ]);
    }

    #[test]
    fn branches_and_supervisor_calls() {
        let call = |reads| Effects {
            calls: true,
            writes: x(&[30]),
            ..branch(reads)
        };
        check(&[
            (0x1400_0000, branch(N)),        // b .
            (0x9400_0000, call(N)),          // bl .
            (0x5400_0001, branch(NZCV)),     // b.ne .
            (0x3400_0004, branch(x(&[4]))),  // cbz w4, .
            (0x3718_0005, branch(x(&[5]))),  // tbnz w5, #3, .
            (0xd61f_00c0, branch(x(&[6]))),  // br x6
            (0xd63f_00e0, call(x(&[7]))),    // blr x7
            (0xd65f_03c0, branch(x(&[30]))), // ret
            (
                0xd400_0021, // svc #1
                Effects {
                    svc: true,
                    ..rw(x(&[0, 1, 2, 3, 4, 5, 6, 7]), x(&[0, 1, 2, 3, 4, 5, 6, 7]))
                },
            ),
        ]);
    }

    #[test]
    fn moves_and_addresses() {
        check(&[
            (0x1000_0003, rw(N, x(&[3]))),       // adr x3, .
            (0x5280_00a1, rw(N, x(&[1]))),       // movz w1, #5
            (0xf2a0_0022, rw(x(&[2]), x(&[2]))), // movk x2, #1, lsl #16
        ]);
    }

    #[test]
    fn add_and_subtract() {
        check(&[
            (0x9100_43ff, rw(SP, SP)),                            // add sp, sp, #16
            (0xb100_43e1, rw(SP, x(&[1]) | NZCV)),                // adds x1, sp, #16
            (0x8b03_0841, rw(x(&[2, 3]), x(&[1]))),               // add x1, x2, x3, lsl #2
            (0x6b03_0041, rw(x(&[2, 3]), x(&[1]) | NZCV)),        // subs w1, w2, w3
            (0x8b22_43e1, rw(SP | x(&[2]), x(&[1]))),             // add x1, sp, w2, uxtw
            (0xeb23_63ff, rw(SP | x(&[3]), NZCV)),                // cmp sp, x3
            (0xba03_0041, rw(x(&[2, 3]) | NZCV, x(&[1]) | NZCV)), // adcs x1, x2, x3
        ]);
    }

    #[test]
    fn logical_bitfield_and_select() {
        check(&[
            (0x8a03_0041, rw(x(&[2, 3]), x(&[1]))), // and x1, x2, x3
            (0x6a03_0041, rw(x(&[2, 3]), x(&[1]) | NZCV)), // ands w1, w2, w3
            (0xb240_1c5f, rw(x(&[2]), SP)),         // orr sp, x2, #0xff
            (0xf240_1c41, rw(x(&[2]), x(&[1]) | NZCV)), // ands x1, x2, #0xff
            (0xb37c_1c41, rw(x(&[1, 2]), x(&[1]))), // bfi x1, x2, #4, #8
            (0xd344_2c41, rw(x(&[2]), x(&[1]))),    // ubfx x1, x2, #4, #8
            (0x93c3_1441, rw(x(&[2, 3]), x(&[1]))), // extr x1, x2, x3, #5
            (0x9a83_0041, rw(x(&[2, 3]) | NZCV, x(&[1]))), // csel x1, x2, x3, eq
            (0xfa43_1040, rw(x(&[2, 3]) | NZCV, NZCV)), // ccmp x2, x3, #0, ne
            (0x3a44_0842, rw(x(&[2]) | NZCV, NZCV)), // ccmn w2, #4, #2, eq
            (0xd53b_4201, rw(NZCV, x(&[1]))),       // mrs x1, nzcv
            (0xd51b_4201, rw(x(&[1]), NZCV)),       // msr nzcv, x1
        ]);
    }

    #[test]
    fn multiply_divide_shift_and_bit_operations() {
        check(&[
            (0x9b03_1041, rw(x(&[2, 3, 4]), x(&[1]))), // madd x1, x2, x3, x4
            (0x9ba3_1041, rw(x(&[2, 3, 4]), x(&[1]))), // umaddl x1, w2, w3, x4
            (0x9b23_1041, rw(x(&[2, 3, 4]), x(&[1]))), // smaddl x1, w2, w3, x4
            (0x9bc3_7c41, rw(x(&[2, 3]), x(&[1]))),    // umulh x1, x2, x3
            (0x1ac3_0c41, rw(x(&[2, 3]), x(&[1]))),    // sdiv w1, w2, w3
            (0x9ac3_2041, rw(x(&[2, 3]), x(&[1]))),    // lslv x1, x2, x3
            (0x9ac3_5c41, rw(x(&[2, 3]), x(&[1]))),    // crc32cx w1, w2, x3
            (0xdac0_1041, rw(x(&[2]), x(&[1]))),       // clz x1, x2
            (0x5ac0_0041, rw(x(&[2]), x(&[1]))),       // rbit w1, w2
            (0xdac0_0c41, rw(x(&[2]), x(&[1]))),       // rev x1, x2
        ]);
    }

    #[test]
    fn atomics() {
        let atomic = |reads, writes| Effects {
            atomic: true,
            loads: true,
            stores: true,
            ..rw(reads, writes)
        };
        check(&[
            (0x88e1_fc62, atomic(x(&[1, 2, 3]), x(&[1]))), // casal w1, w2, [x3]
            (0x4824_7fe6, atomic(x(&[4, 5, 6, 7]) | SP, x(&[4, 5]))), // casp x4, x5, x6, x7, [sp]
            (0xb821_0062, atomic(x(&[1, 3]), x(&[2]))),    // ldadd w1, w2, [x3]
            (0xb821_007f, atomic(x(&[1, 3]), N)),          // stadd w1, [x3]
            (0xf8e1_8062, atomic(x(&[1, 3]), x(&[2]))),    // swpal x1, x2, [x3]
        ]);
    }

    #[test]
    fn ordered_and_exclusive_accesses() {
        check(&[
            (0x38bf_c041, load(x(&[2]), x(&[1]))), // ldaprb w1, [x2]
            (0xc8df_ffe1, load(SP, x(&[1]))),      // ldar x1, [sp]
            (
                0x885f_fc41, // ldaxr w1, [x2]
                Effects {
                    exclusive: true,
                    ..load(x(&[2]), x(&[1]))
                },
            ),
            (
                0x8801_fc62, // stlxr w1, w2, [x3]
                Effects {
                    exclusive: true,
                    ..store(x(&[2, 3]), x(&[1]))
                },
            ),
            (0xc89f_fc41, store(x(&[1, 2]), N)), // stlr x1, [x2]
        ]);
    }

    #[test]
    fn loads_and_stores() {
        check(&[
            (0xf940_0441, load(x(&[2]), x(&[1]))),       // ldr x1, [x2, #8]
            (0xf800_8c41, store(x(&[1, 2]), x(&[2]))),   // str x1, [x2, #8]!
            (0x3840_1441, load(x(&[2]), x(&[1, 2]))),    // ldrb w1, [x2], #1
            (0xb863_7841, load(x(&[2, 3]), x(&[1]))),    // ldr w1, [x2, x3, lsl #2]
            (0x7900_07ff, store(SP, N)),                 // strh wzr, [sp, #2]
            (0xa8c1_0be1, load(SP, x(&[1, 2]) | SP)),    // ldp x1, x2, [sp], #16
            (0xa9bf_7bfd, store(x(&[29, 30]) | SP, SP)), // stp x29, x30, [sp, #-16]!
            (0x6940_0861, load(x(&[3]), x(&[1, 2]))),    // ldpsw x1, x2, [x3]
            (0xad41_0861, load(x(&[3]), v(&[1, 2]))),    // ldp q1, q2, [x3, #32]
            (0xad81_0861, store(x(&[3]) | v(&[1, 2]), x(&[3]))), // stp q1, q2, [x3, #32]!
            (0x5800_0001, load(N, x(&[1]))),             // ldr x1, .
        ]);
    }

    #[test]
    fn system_registers_and_cache_maintenance() {
        check(&[
            (0xd53b_e021, system(N, x(&[1]))), // mrs x1, cntpct_el0
            (0xd53b_d041, system(N, x(&[1]))), // mrs x1, tpidr_el0
            (0xd53b_d061, system(N, x(&[1]))), // mrs x1, tpidrro_el0
            (0xd51b_d041, system(x(&[1]), N)), // msr tpidr_el0, x1
            (0xd53b_4401, system(N, x(&[1]))), // mrs x1, fpcr
            (0xd51b_4421, system(x(&[1]), N)), // msr fpsr, x1
            (0xd53b_0021, system(N, x(&[1]))), // mrs x1, ctr_el0
            (0xd50b_7e21, system(N, N)),       // dc civac, x1
        ]);
    }

    #[test]
    fn traps_read_everything() {
        let udf = decode(0x0000_0000, 0x1000);
        assert_eq!(udf, Instruction::Undefined);
        assert_eq!(
            effects(&udf),
            Effects {
                traps: true,
                ..rw(Regs::ALL, N)
            }
        );
        let hvc = decode(0xd400_0002, 0x1000);
        assert_eq!(hvc, Instruction::Unsupported);
        assert_eq!(
            effects(&hvc),
            Effects {
                traps: true,
                ..rw(Regs::ALL, Regs::ALL)
            }
        );
    }

    #[test]
    fn simd_loads_and_stores() {
        check(&[
            (0x3dc0_0441, load(x(&[2]), v(&[1]))), // ldr q1, [x2, #16]
            (0xbc24_7be3, store(SP | x(&[4]) | v(&[3]), N)), // str s3, [sp, x4, lsl #2]
            (0xfc40_84c5, load(x(&[6]), x(&[6]) | v(&[5]))), // ldr d5, [x6], #8
            (0x6d40_0861, load(x(&[3]), v(&[1, 2]))), // ldp d1, d2, [x3]
            (0xadbf_0be1, store(SP | v(&[1, 2]), SP)), // stp q1, q2, [sp, #-32]!
            (0x1c00_0004, load(N, v(&[4]))),       // ldr s4, .
        ]);
    }

    #[test]
    fn simd_structure_transfers() {
        check(&[
            // ld1 {v31.16b, v0.16b}, [x2], x3
            (0x4cc3_a05f, load(x(&[2, 3]), x(&[2]) | v(&[31, 0]))),
            (0x4c00_8be1, store(SP | v(&[1, 2]), N)), // st2 {v1.4s, v2.4s}, [sp]
            // ld1 {v1.s}[1], [x2], #4
            (0x0ddf_9041, load(x(&[2]) | v(&[1]), x(&[2]) | v(&[1]))),
            (0x4d40_c841, load(x(&[2]), v(&[1]))), // ld1r {v1.4s}, [x2]
        ]);
    }

    #[test]
    fn scalar_floating_point() {
        check(&[
            (0x1e61_c041, rw(v(&[2]), v(&[1]))),           // fsqrt d1, d2
            (0x1e22_c041, rw(v(&[2]), v(&[1]))),           // fcvt d1, s2
            (0x1e23_c041, rw(v(&[1, 2]), v(&[1]))),        // fcvt h1, s2
            (0x1e23_2841, rw(v(&[2, 3]), v(&[1]))),        // fadd s1, s2, s3
            (0x1f43_1041, rw(v(&[2, 3, 4]), v(&[1]))),     // fmadd d1, d2, d3, d4
            (0x1e20_2028, rw(v(&[1]), NZCV)),              // fcmp s1, #0.0
            (0x1e62_2020, rw(v(&[1, 2]), NZCV)),           // fcmp d1, d2
            (0x1e22_1420, rw(v(&[1, 2]) | NZCV, NZCV)),    // fccmp s1, s2, #0, ne
            (0x1e63_cc41, rw(v(&[2, 3]) | NZCV, v(&[1]))), // fcsel d1, d2, d3, gt
            (0x1e2e_1001, rw(N, v(&[1]))),                 // fmov s1, #1.0
            (0x5e21_d841, rw(v(&[2]), v(&[1]))),           // scvtf s1, s2
        ]);
    }

    #[test]
    fn simd_general_register_transfers() {
        check(&[
            (0x1e26_0041, rw(v(&[2]), x(&[1]))),           // fmov w1, s2
            (0x9eae_0041, rw(v(&[2]), x(&[1]))),           // fmov x1, v2.d[1]
            (0x9e67_0041, rw(x(&[2]), v(&[1]))),           // fmov d1, x2
            (0x9eaf_0041, rw(x(&[2]) | v(&[1]), v(&[1]))), // fmov v1.d[1], x2
            (0x1e62_0041, rw(x(&[2]), v(&[1]))),           // scvtf d1, w2
            (0x9e42_f041, rw(x(&[2]), v(&[1]))),           // scvtf d1, x2, #4
            (0x1e78_0041, rw(v(&[2]), x(&[1]))),           // fcvtzs w1, d2
            (0x9e58_f041, rw(v(&[2]), x(&[1]))),           // fcvtzs x1, d2, #4
            (0x4e0c_1c41, rw(x(&[2]) | v(&[1]), v(&[1]))), // mov v1.s[1], w2
            (0x0e07_3c41, rw(v(&[2]), x(&[1]))),           // umov w1, v2.b[3]
            (0x4e04_0c41, rw(x(&[2]), v(&[1]))),           // dup v1.4s, w2
        ]);
    }

    #[test]
    fn simd_vector_operations() {
        let unary = rw(v(&[1, 2]), v(&[1]));
        let binary = rw(v(&[1, 2, 3]), v(&[1]));
        check(&[
            (0x4f03_e7e1, rw(N, v(&[1]))),       // movi v1.16b, #0x7f
            (0x4f07_17e1, rw(v(&[1]), v(&[1]))), // orr v1.4s, #0xff
            (0x4e23_1c41, binary),               // and v1.16b, v2.16b, v3.16b
            (0x6e0c_0441, unary),                // mov v1.s[1], v2.s[0]
            (0x4e0c_0441, rw(v(&[2]), v(&[1]))), // dup v1.4s, v2.s[1]
            (0x4e23_d441, binary),               // fadd v1.4s, v2.4s, v3.4s
            (0x6e23_d441, binary),               // faddp v1.4s, v2.4s, v3.4s
            (0x4ea3_bc41, binary),               // addp v1.4s, v2.4s, v3.4s
            (0x4fa3_8041, binary),               // mul v1.4s, v2.4s, v3.s[1]
            (0x0f53_a041, binary),               // smull v1.4s, v2.4h, v3.h[1]
            (0x0e61_2841, unary),                // xtn v1.4h, v2.4s
            (0x0e23_e041, binary),               // pmull v1.8h, v2.8b, v3.8b
            (0x4e28_4841, unary),                // aese v1.16b, v2.16b
            (0x5e03_0041, binary),               // sha1c q1, s2, v3.4s
            // tbl v1.16b, {v30.16b, v31.16b, v0.16b}, v4.16b
            (0x4e04_43c1, rw(v(&[1, 30, 31, 0, 4]), v(&[1]))),
            (0x0e61_7841, unary),  // fcvtl v1.2d, v2.2s
            (0x4f83_9841, binary), // fmul v1.4s, v2.4s, v3.s[2]
            (0x4ea3_8441, binary), // add v1.4s, v2.4s, v3.4s
            (0x4ea0_b841, unary),  // abs v1.4s, v2.4s
            (0x4e21_9841, unary),  // frintm v1.4s, v2.4s
            (0x4eb1_b841, unary),  // addv s1, v2.4s
            (0x4e83_9441, binary), // sdot v1.4s, v2.16b, v3.16b
            (0x6e30_f841, unary),  // fmaxv s1, v2.4s
            (0x4f23_5441, unary),  // shl v1.4s, v2.4s, #3
            (0x0f21_a441, unary),  // sshll v1.2d, v2.2s, #1
            (0x2e63_0041, binary), // uaddl v1.4s, v2.4h, v3.4h
            (0x4e60_6841, unary),  // sadalp v1.4s, v2.8h
            (0x4e83_3841, binary), // zip1 v1.4s, v2.4s, v3.4s
            (0x6e03_2041, binary), // ext v1.16b, v2.16b, v3.16b, #4
        ]);
    }
}
