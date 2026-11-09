//! Decoding of 128-bit Ampere (SM 8.x) instruction words.
//!
//! The encodings here were learned from the Ampere front end of Lichii
//! (<https://github.com/zortos293/Lichii>, GPL-3.0-or-later,
//! `src/shader_recompiler/frontend/ampere/lower_to_maxwell.h` at `c215653`):
//! field positions and meanings only, cross-checked against the SM70+
//! encoder of Mesa's NAK (MIT, `src/nouveau/compiler/nak/sm70_encode.rs` at
//! `979c6d1`). The code and its structure are written for Echora. See
//! NOTICE.md and docs/research.md.
//!
//! An instruction is one little-endian 128-bit word. Bits 0-8 select the
//! operation and bits 9-11 the form of its operands (register, immediate,
//! constant buffer or uniform register in the second or third source); bits
//! 12-15 are the guard predicate (bit 15 negates it; predicate 7 is PT);
//! bits 16-23 the destination register (255 is RZ); bits 105-127 the
//! scheduling controls, which decoding ignores. A form or operation this
//! decoder does not know is [`Op::Unknown`], never a panic.

/// A general register; 255 is RZ (reads zero, writes are dropped).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Reg(pub u8);

impl Reg {
    pub const ZERO: Self = Self(255);
}

/// A uniform register; 63 is URZ.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct UniformReg(pub u8);

impl UniformReg {
    pub const ZERO: Self = Self(63);
}

/// A predicate register, possibly negated; 7 is PT (true).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Pred {
    pub index: u8,
    pub negated: bool,
}

impl Pred {
    pub const TRUE: Self = Self {
        index: 7,
        negated: false,
    };

    /// Whether the predicate is always true (PT).
    pub fn is_true(self) -> bool {
        self.index == 7 && !self.negated
    }
}

/// Where a source operand comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Src {
    Reg(Reg),
    Uniform(UniformReg),
    /// A 32-bit immediate (for float operations, the float's bits).
    Imm(u32),
    /// A word of constant buffer `bank` at byte `offset`.
    Const {
        bank: u8,
        offset: u16,
    },
    /// A word at byte `offset` of the constant buffer whose handle is in
    /// uniform register `handle` (bit 91 set in a constant form).
    BindlessConst {
        handle: UniformReg,
        offset: u16,
    },
}

/// A source operand with its float modifiers (ignored by integer
/// operations, whose encodings use the same bits otherwise or not at all).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Operand {
    pub src: Src,
    pub abs: bool,
    pub neg: bool,
}

/// How a predicate result is combined with an accumulated predicate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Combine {
    And,
    Or,
    Xor,
}

/// Where a load or store goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Space {
    /// Global memory, through the GPU page table.
    Global,
    /// The workgroup's shared memory.
    Shared,
    /// The invocation's local memory.
    Local,
}

/// A memory address: a register (a pair when `wide`), scaled by `scale`
/// (shared memory only), plus a uniform register (a pair when its flag is
/// set) and a signed byte offset.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Address {
    pub reg: Reg,
    pub wide: bool,
    pub scale: u8,
    pub uniform: Option<(UniformReg, bool)>,
    pub offset: i32,
}

/// A decoded instruction: its guard and operation, and the word it came
/// from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Instruction {
    pub raw: u128,
    pub guard: Pred,
    pub op: Op,
    /// A uniform-datapath instruction: its registers are uniform
    /// registers (63 is URZ) and its predicates, guard included, uniform
    /// predicates (7 is UPT). Its operation is decoded as the per-thread
    /// form it mirrors.
    pub uniform: bool,
}

/// The operations decoded so far. Field names follow the SASS operands:
/// `d` the destination, `a`, `b`, `c` the sources in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// MOV d, b.
    Mov {
        d: Reg,
        src: Src,
    },
    /// IADD3 d, a, b, c, with up to two carry outputs. Without `.X`
    /// (`extended`, bit 74) a negated source is its two's complement; with
    /// it, its bitwise inverse, and the two carry inputs are added.
    Iadd3 {
        d: Reg,
        a: Operand,
        b: Operand,
        c: Operand,
        carry_out: [Pred; 2],
        carry_in: [Pred; 2],
        extended: bool,
    },
    /// IMAD d, a, b, c: the low word of a * b + c.
    Imad {
        d: Reg,
        a: Operand,
        b: Operand,
        c: Operand,
    },
    /// IMAD.WIDE d:d+1, a, b, c:c+1: the 64-bit a * b + c, the factors
    /// `signed` (bit 73) or not; IMAD.HI (`high`) writes only its high
    /// word to d. `carry_out` (81-83) receives the 64-bit sum's carry.
    ImadWide {
        d: Reg,
        a: Operand,
        b: Operand,
        c: Operand,
        signed: bool,
        high: bool,
        carry_out: u8,
    },
    /// POPC / FLO / BREV d, src (inverted first with `invert`, bit 63):
    /// the set-bit count, the most significant set bit (`signed`: the most
    /// significant bit unlike the sign; `shift_amount`: as 31 minus the
    /// position), or the bits reversed.
    BitCount {
        d: Reg,
        src: Operand,
        invert: bool,
    },
    Flo {
        d: Reg,
        src: Operand,
        invert: bool,
        signed: bool,
        shift_amount: bool,
    },
    Brev {
        d: Reg,
        src: Operand,
    },
    /// SHFL d, p, a, lane, c: a from another lane of the subgroup: `mode`
    /// (bits 58-59) 0 index, 1 up, 2 down, 3 xor; c holds the clamp
    /// (bits 0-4) and segment mask (8-12); `p` receives whether the lane
    /// was in range.
    Shfl {
        d: Reg,
        p: u8,
        a: Reg,
        lane: Src,
        c: Src,
        mode: u8,
    },
    /// VOTE d, p, condition: `mode` (bits 72-73) 0 all, 1 any, 2 all
    /// equal into p; d receives the ballot of the condition.
    Vote {
        d: Reg,
        p: u8,
        condition: Pred,
        mode: u8,
        /// VOTEU: the ballot goes to a uniform register and the vote to a
        /// uniform predicate.
        uniform_out: bool,
    },
    /// REDUX d, src: the subgroup's and, or, xor, sum, min or max
    /// (`operation`, bits 78-80; min and max `signed` at 73) of src, into
    /// a uniform register.
    Redux {
        d: UniformReg,
        src: Reg,
        operation: u8,
        signed: bool,
    },
    /// IABS d, src: the absolute value of a signed word.
    Iabs {
        d: Reg,
        src: Operand,
    },
    /// P2R d, a, mask: a with byte `byte` (bits 76-77) replaced, where
    /// `mask` is set, by the predicates P0-P6 as bits.
    P2r {
        d: Reg,
        a: Reg,
        mask: u32,
        byte: u8,
    },
    /// R2P a, mask: predicate Pi, where bit i of `mask` is set, from bit i
    /// of byte `byte` of a.
    R2p {
        a: Reg,
        mask: u32,
        byte: u8,
    },
    /// LEA d, a, b (and c with `.HI`): (a << shift) + b; with `high`
    /// (`.HI`, bit 80) the shifted value is the high word of c:a << shift.
    /// `negate_shifted` (bit 72) negates the shifted value (its inverse
    /// with `.X`); `.X` (`extended`, bit 74) adds `carry_in`; `overflow`
    /// receives the carry out.
    Lea {
        d: Reg,
        a: Reg,
        b: Operand,
        c: Operand,
        shift: u8,
        high: bool,
        negate_shifted: bool,
        extended: bool,
        overflow: u8,
        carry_in: Pred,
    },
    /// SHF d, low, shift, high: the 64-bit funnel high:low shifted left or
    /// `right`, its low word or, with `high_word`, its high word. `kind`
    /// (bits 73-74): 0 signed 64-bit, 1 unsigned 64-bit, 2 signed 32-bit,
    /// 3 unsigned 32-bit; the shift wraps (`wrap`) or clamps at that width.
    Shf {
        d: Reg,
        low: Reg,
        shift: Operand,
        high: Operand,
        kind: u8,
        wrap: bool,
        right: bool,
        high_word: bool,
    },
    /// PRMT d, a, selector, c: bytes of c:a picked by the selector's
    /// nibbles (bit 3 of a nibble replicates the byte's sign); `mode` (bits
    /// 72-74) 0 is that indexed form.
    Prmt {
        d: Reg,
        a: Reg,
        selector: Operand,
        c: Operand,
        mode: u8,
    },
    /// SGXT d, a, bits: the low `bits` bits of a, sign-extended if
    /// `signed`.
    Sgxt {
        d: Reg,
        a: Reg,
        bits: Operand,
        signed: bool,
        wrap: bool,
    },
    /// SEL / FSEL d, a, b, condition: a if the condition holds, else b
    /// (FSEL, `float`, applies the float modifiers first).
    Sel {
        d: Reg,
        a: Operand,
        b: Operand,
        condition: Pred,
        float: bool,
    },
    /// FMNMX d, a, b, min: the minimum if `min` holds, else the maximum (a
    /// NaN loses to a number).
    Fmnmx {
        d: Reg,
        a: Operand,
        b: Operand,
        min: Pred,
    },
    /// IMNMX d, a, b, min, as FMNMX for integers.
    Imnmx {
        d: Reg,
        a: Operand,
        b: Operand,
        min: Pred,
        signed: bool,
    },
    /// MUFU d, src: `function` (bits 74-79) 0 cos, 1 sin (of turns), 2
    /// exp2, 3 log2, 4 reciprocal, 5 reciprocal square root, 6/7 their
    /// 64-bit high-word forms, 8 square root, 9 tanh.
    Mufu {
        d: Reg,
        function: u8,
        src: Operand,
    },
    /// F2I d, src: float to integer of `dst_size` (bits 75-76, log2 bytes)
    /// from `src_size` (84-85), `signed` (72), rounded by `rounding`.
    F2i {
        d: Reg,
        src: Operand,
        signed: bool,
        dst_size: u8,
        src_size: u8,
        rounding: u8,
    },
    /// I2F / I2FP.F32 d, src: integer (`signed`, bit 74) to float, sizes
    /// as F2I.
    I2f {
        d: Reg,
        src: Operand,
        signed: bool,
        dst_size: u8,
        src_size: u8,
        rounding: u8,
    },
    /// FRND d, src: rounded to an integral float by `rounding`.
    Frnd {
        d: Reg,
        src: Operand,
        rounding: u8,
        dst_size: u8,
        src_size: u8,
    },
    /// FSET.BF d, a, b, acc: 1.0 if (a cmp b) combine acc, else 0.0.
    Fset {
        d: Reg,
        a: Operand,
        b: Operand,
        compare: FloatCompare,
        combine: Combine,
        accumulate: Pred,
    },
    /// PLOP3.LUT p, q, a, b, c: two functions of three predicates.
    Plop3 {
        p: u8,
        q: u8,
        srcs: [Pred; 3],
        luts: [u8; 2],
    },
    /// CS2R d, special (`wide`: a register pair); 255 is SRZ (zero).
    Cs2r {
        d: Reg,
        special: u8,
        wide: bool,
    },
    /// LDC d, c[bank][index + offset], `size` the memory type (bits 73-75:
    /// 4 a word, 5 a pair), `mode` (78-79) 0 for one bank.
    Ldc {
        d: Reg,
        index: Reg,
        bank: u8,
        offset: u16,
        size: u8,
        mode: u8,
    },
    /// ULDC d, constant: a uniform register (pair with `size` 5).
    Uldc {
        d: UniformReg,
        src: Src,
        size: u8,
    },
    /// LDG / LDS / LDL d, [address]: `size` the memory type (bits 73-75: 0
    /// U8, 1 S8, 2 U16, 3 S16, 4 a word, 5 a pair, 6 four words); a global
    /// load reads zero where `condition` fails.
    Load {
        d: Reg,
        space: Space,
        address: Address,
        size: u8,
        condition: Pred,
    },
    /// STG / STS / STL [address], data.
    Store {
        data: Reg,
        space: Space,
        address: Address,
        size: u8,
    },
    /// RED / ATOMG / ATOMS d, [address], data: `operation` (bits 87-90) 0
    /// add, 1 min, 2 max, 3 increment, 4 decrement, 5 and, 6 or, 7 xor, 8
    /// exchange; with `compare` (CAS) a compare-and-swap. `kind` (73-75):
    /// 0 U32, 1 S32, 2 U64, 3 F32, 4 F16x2, 5 S64, 6 F64. RED has no
    /// result (RZ).
    Atomic {
        d: Reg,
        space: Space,
        address: Address,
        data: Reg,
        compare: Option<Reg>,
        operation: u8,
        kind: u8,
    },
    /// BAR.SYNC (`mode` 0, bits 77-78) and the other barrier modes.
    Bar {
        mode: u8,
    },
    /// BSSY and BSYNC (convergence barriers) and DEPBAR (scoreboard
    /// waits): nothing for a translation that runs each invocation on its
    /// own.
    Bssy,
    Bsync,
    Depbar,
    /// LOP3.LUT d, a, b, c, lut: any function of three inputs by its truth
    /// table; `p`, unless PT, receives whether the result is nonzero.
    Lop3 {
        d: Reg,
        a: Reg,
        b: Operand,
        c: Operand,
        lut: u8,
        p: Pred,
        /// `.PAND` (bit 80), as nvdisasm prints it: the predicate output's
        /// mode, decoded but not yet interpreted.
        pand: bool,
    },
    /// FFMA d, a, b, c (a * b + c, fused).
    Ffma {
        d: Reg,
        a: Operand,
        b: Operand,
        c: Operand,
        modes: FloatModes,
        /// `.FMZ` (bit 76): a denormal or infinite factor times zero is
        /// zero.
        dnz: bool,
    },
    /// FMUL d, a, b, with a power-of-two `scale` of the result (bits
    /// 84-86; 4 is none).
    Fmul {
        d: Reg,
        a: Operand,
        b: Operand,
        modes: FloatModes,
        scale: u8,
        /// As FFMA's.
        dnz: bool,
    },
    /// FADD d, a, b.
    Fadd {
        d: Reg,
        a: Operand,
        b: Operand,
        modes: FloatModes,
    },
    /// FSETP p, q, a, b, acc: p = (a cmp b) combine acc, q = !(a cmp b)
    /// combine acc.
    Fsetp {
        p: u8,
        q: u8,
        a: Operand,
        b: Operand,
        compare: FloatCompare,
        flush_to_zero: bool,
        combine: Combine,
        accumulate: Pred,
    },
    /// ISETP p, q, a, b, acc, as FSETP for integers (`signed` or not).
    /// `extended` (`.EX`) compares the high words of a 64-bit pair, with
    /// the low words' result in `low`.
    Isetp {
        p: u8,
        q: u8,
        a: Operand,
        b: Operand,
        compare: IntCompare,
        signed: bool,
        combine: Combine,
        accumulate: Pred,
        extended: Option<Pred>,
    },
    /// R2UR d, src: a register's value (the first active invocation's)
    /// into a uniform register.
    R2ur {
        d: UniformReg,
        src: Reg,
    },
    /// S2R d, special register (S2UR: into a uniform register) (the Maxwell numbering: 0 lane id, 33-35
    /// thread id x/y/z, 37-39 workgroup id x/y/z, ...).
    S2r {
        d: Reg,
        special: u8,
    },
    /// BRA to `offset` bytes from the next instruction, if `condition`.
    /// `mode` is the convergence mode in the offset's low bits (0 plain,
    /// 2 `.DIV`, 3 `.CONV`).
    Bra {
        offset: i64,
        mode: u8,
        condition: Pred,
    },
    Exit,
    Nop,
    /// An operation or form not decoded (yet); `opcode` is bits 0-11.
    Unknown {
        opcode: u16,
    },
}

/// Float result modes: saturate to [0, 1] (bit 77), rounding (bits 78-79:
/// 0 nearest even, 1 down, 2 up, 3 toward zero) and flushing denormals to
/// zero (bit 80).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FloatModes {
    pub saturate: bool,
    pub rounding: u8,
    pub flush_to_zero: bool,
}

/// FSETP comparisons (bits 76-79): ordered then unordered forms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FloatCompare {
    False,
    Lt,
    Eq,
    Le,
    Gt,
    Ne,
    Ge,
    /// Neither is NaN.
    Num,
    /// Either is NaN.
    Nan,
    Ltu,
    Equ,
    Leu,
    Gtu,
    Neu,
    Geu,
    True,
}

/// ISETP comparisons (bits 76-78).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IntCompare {
    False,
    Lt,
    Eq,
    Le,
    Gt,
    Ne,
    Ge,
    True,
}

/// The bits of an instruction word.
#[derive(Clone, Copy)]
struct Word(u128);

impl Word {
    fn bits(self, offset: u32, count: u32) -> u64 {
        ((self.0 >> offset) & ((1u128 << count) - 1)) as u64
    }

    fn bit(self, offset: u32) -> bool {
        self.bits(offset, 1) != 0
    }

    fn reg(self, offset: u32) -> Reg {
        Reg(self.bits(offset, 8) as u8)
    }

    /// A predicate in bits `offset`..+3, negated by bit `negate`.
    fn pred(self, offset: u32, negate: u32) -> Pred {
        Pred {
            index: self.bits(offset, 3) as u8,
            negated: self.bit(negate),
        }
    }

    fn form(self) -> u8 {
        self.bits(9, 3) as u8
    }

    /// A constant-buffer operand: bank 54-58 and byte offset 38-53, or
    /// with bit 91 the buffer's handle in the uniform register at 32-37.
    fn constant(self) -> Src {
        let offset = self.bits(38, 16) as u16;
        if self.bit(91) {
            Src::BindlessConst {
                handle: UniformReg(self.bits(32, 6) as u8),
                offset,
            }
        } else {
            Src::Const {
                bank: self.bits(54, 5) as u8,
                offset,
            }
        }
    }

    /// The first source: a register (bits 24-31), with float abs (73) and
    /// negate (72).
    fn a(self) -> Operand {
        Operand {
            src: Src::Reg(self.reg(24)),
            abs: self.bit(73),
            neg: self.bit(72),
        }
    }

    /// The second (`which` 1) or third (2) source, by the form: forms 4, 5
    /// and 6 put an immediate, constant or uniform register in the second
    /// source, forms 2, 3 and 7 in the third; the register sources then sit
    /// in bits 64-71 (abs 74, negate 75), or the second in bits 32-39 (abs
    /// 62, negate 63) in form 1. `None` for a form that has none.
    fn source(self, which: u32) -> Option<Operand> {
        let form = self.form();
        let wide = if which == 1 {
            matches!(form, 4..=6)
        } else {
            matches!(form, 2 | 3 | 7)
        };
        if !(1..=7).contains(&form) {
            return None;
        }
        Some(if wide {
            let src = match form {
                2 | 4 => Src::Imm(self.bits(32, 32) as u32),
                3 | 5 => self.constant(),
                _ => Src::Uniform(UniformReg(self.bits(32, 6) as u8)),
            };
            let immediate = matches!(src, Src::Imm(_));
            Operand {
                src,
                abs: !immediate && self.bit(62),
                neg: !immediate && self.bit(63),
            }
        } else {
            let upper = which == 2 || matches!(form, 2 | 3 | 7);
            let (offset, abs, neg) = if upper { (64, 74, 75) } else { (32, 62, 63) };
            Operand {
                src: Src::Reg(self.reg(offset)),
                abs: self.bit(abs),
                neg: self.bit(neg),
            }
        })
    }

    /// A memory access's address: the register at 24, a pair when
    /// `wide_bit` is set; the uniform register at `uniform_at` when bit 91
    /// is set, a pair when `uniform_wide_bit` is; a signed offset at 40-63;
    /// and for `scaled` (shared) accesses the register's scale at 78-79
    /// (1, 4, 8 or 16).
    fn address(
        self,
        wide_bit: Option<u32>,
        uniform_at: Option<u32>,
        uniform_wide_bit: Option<u32>,
        scaled: bool,
    ) -> Address {
        let uniform = uniform_at.filter(|_| self.bit(91)).map(|at| {
            (
                UniformReg(self.bits(at, 6) as u8),
                uniform_wide_bit.is_some_and(|bit| self.bit(bit)),
            )
        });
        Address {
            reg: self.reg(24),
            wide: wide_bit.is_some_and(|bit| self.bit(bit)),
            scale: if scaled {
                [1, 4, 8, 16][self.bits(78, 2) as usize]
            } else {
                1
            },
            uniform,
            offset: ((self.bits(40, 24) as u32) << 8) as i32 >> 8,
        }
    }

    fn float_modes(self) -> FloatModes {
        FloatModes {
            saturate: self.bit(77),
            rounding: self.bits(78, 2) as u8,
            flush_to_zero: self.bit(80),
        }
    }

    fn combine(self) -> Option<Combine> {
        Some(match self.bits(74, 2) {
            0 => Combine::And,
            1 => Combine::Or,
            2 => Combine::Xor,
            _ => return None,
        })
    }
}

/// Decodes one instruction word (little-endian: the first byte of the
/// instruction is bits 0-7).
pub fn decode(raw: u128) -> Instruction {
    let word = Word(raw);
    let opcode = word.bits(0, 12) as u16;
    let (op, uniform) = match uniform_mirror(opcode) {
        Some(mirror) => (decode_op(word, mirror), true),
        None => (
            decode_op(word, opcode),
            matches!(opcode, 0xab9 | 0xabb | 0x9c3),
        ),
    };
    Instruction {
        raw,
        guard: word.pred(12, 15),
        op: op.unwrap_or(Op::Unknown { opcode }),
        uniform,
    }
}

/// The per-thread opcode a uniform-datapath opcode mirrors, for the forms
/// that operate on uniform registers alone (register, and an immediate in
/// the second or third source; UMOV also from a uniform register).
fn uniform_mirror(opcode: u16) -> Option<u16> {
    let form = opcode >> 9;
    let operation = opcode & 0x1ff;
    let mirrored = match operation {
        0x082 => matches!(form, 4 | 6),
        0x087 | 0x08c | 0x090 | 0x091 | 0x092 | 0x096 | 0x099 | 0x09a | 0x0a4 | 0x0a5 => {
            matches!(form, 1 | 2 | 4)
        }
        _ => opcode == 0x89c,
    };
    mirrored.then(|| {
        if opcode == 0x89c {
            0x81c
        } else {
            form << 9 | (operation - 0x080)
        }
    })
}

/// Decodes the instructions of a program (16 bytes each; a trailing
/// partial instruction is ignored).
pub fn decode_program(bytes: &[u8]) -> Vec<Instruction> {
    let (words, _) = bytes.as_chunks::<16>();
    words
        .iter()
        .map(|word| decode(u128::from_le_bytes(*word)))
        .collect()
}

fn decode_op(word: Word, opcode: u16) -> Option<Op> {
    let d = word.reg(16);
    let form = word.form();
    // Operations whose bits 9-11 are not an ALU form match all 12 bits.
    match opcode {
        0x389 | 0x589 | 0x989 | 0xf89 => {
            let lane = if opcode & 0x800 != 0 {
                Src::Imm(word.bits(53, 5) as u32)
            } else {
                Src::Reg(word.reg(32))
            };
            let c = if matches!(opcode, 0x589 | 0xf89) {
                Src::Imm(word.bits(40, 13) as u32)
            } else {
                Src::Reg(word.reg(64))
            };
            return Some(Op::Shfl {
                d,
                p: word.bits(81, 3) as u8,
                a: word.reg(24),
                lane,
                c,
                mode: word.bits(58, 2) as u8,
            });
        }
        0x806 | 0x886 => {
            return Some(Op::Vote {
                d,
                p: word.bits(81, 3) as u8,
                condition: word.pred(87, 90),
                mode: word.bits(72, 2) as u8,
                uniform_out: opcode == 0x886,
            });
        }
        0x3c4 => {
            return Some(Op::Redux {
                d: UniformReg(word.bits(16, 6) as u8),
                src: word.reg(24),
                operation: word.bits(78, 3) as u8,
                signed: word.bit(73),
            });
        }
        // BREAK (a convergence barrier) and WARPSYNC: nothing per
        // invocation.
        0x942 | 0x948 => return Some(Op::Bsync),
        0x803 => {
            return Some(Op::P2r {
                d,
                a: word.reg(24),
                mask: word.bits(32, 32) as u32,
                byte: word.bits(76, 2) as u8,
            });
        }
        0x804 => {
            return Some(Op::R2p {
                a: word.reg(24),
                mask: word.bits(32, 32) as u32,
                byte: word.bits(76, 2) as u8,
            });
        }
        0x3c2 => {
            return Some(Op::R2ur {
                d: UniformReg(word.bits(16, 6) as u8),
                src: word.reg(24),
            });
        }
        0x919 | 0x9c3 => {
            return Some(Op::S2r {
                d,
                special: word.bits(72, 8) as u8,
            });
        }
        0x947 => {
            // A 50-bit byte offset in bits 32-81 whose two low bits are
            // the convergence mode.
            let field = word.bits(32, 50);
            let offset = ((field & !3) << 14) as i64 >> 14;
            return Some(Op::Bra {
                offset,
                mode: (field & 3) as u8,
                condition: word.pred(87, 90),
            });
        }
        0x94d => return Some(Op::Exit),
        0x918 => return Some(Op::Nop),
        0x945 => return Some(Op::Bssy),
        0x941 => return Some(Op::Bsync),
        0x91a => return Some(Op::Depbar),
        0x81c => {
            // The second function's table sits where a destination would.
            return Some(Op::Plop3 {
                p: word.bits(81, 3) as u8,
                q: word.bits(84, 3) as u8,
                srcs: [word.pred(87, 90), word.pred(77, 80), word.pred(68, 71)],
                luts: [
                    (word.bits(64, 3) | word.bits(72, 5) << 3) as u8,
                    word.bits(16, 8) as u8,
                ],
            });
        }
        0x805 => {
            return Some(Op::Cs2r {
                d,
                special: word.bits(72, 8) as u8,
                wide: word.bit(80),
            });
        }
        0x381 | 0x981 | 0x984 | 0x983 => {
            let space = match opcode {
                0x381 | 0x981 => Space::Global,
                0x984 => Space::Shared,
                _ => Space::Local,
            };
            let condition = if opcode == 0x981 {
                // Stored reversed: 7 - the predicate.
                Pred {
                    index: 7 - word.bits(64, 3) as u8,
                    negated: word.bit(67),
                }
            } else {
                Pred::TRUE
            };
            let address = match opcode {
                0x381 => word.address(Some(72), None, None, false),
                0x981 => word.address(Some(90), Some(32), Some(72), false),
                0x984 => word.address(None, Some(32), None, true),
                _ => word.address(None, Some(32), None, false),
            };
            return Some(Op::Load {
                d,
                space,
                address,
                size: word.bits(73, 3) as u8,
                condition,
            });
        }
        0x386 | 0x986 | 0x388 | 0x988 | 0x387 | 0x987 => {
            let space = match opcode & 0xff {
                0x86 => Space::Global,
                0x88 => Space::Shared,
                _ => Space::Local,
            };
            let address = match opcode {
                0x386 => word.address(Some(72), None, None, false),
                0x986 => word.address(Some(90), Some(64), Some(72), false),
                0x388 => word.address(None, None, None, true),
                0x988 => word.address(None, Some(64), None, true),
                0x387 => word.address(None, None, None, false),
                _ => word.address(None, Some(64), None, false),
            };
            return Some(Op::Store {
                data: word.reg(32),
                space,
                address,
                size: word.bits(73, 3) as u8,
            });
        }
        0x98e | 0x9a8 | 0x3a8 | 0x3a9 | 0x38c | 0x98c | 0x38d => {
            let (space, address) = match opcode {
                0x98e => (
                    Space::Global,
                    word.address(Some(90), Some(64), Some(72), false),
                ),
                0x9a8 => (
                    Space::Global,
                    word.address(Some(70), Some(64), Some(72), false),
                ),
                0x3a8 | 0x3a9 => (Space::Global, word.address(Some(72), None, None, false)),
                0x98c => (Space::Shared, word.address(None, Some(64), None, true)),
                _ => (Space::Shared, word.address(None, None, None, true)),
            };
            let exchange = matches!(opcode, 0x3a9 | 0x38d);
            return Some(Op::Atomic {
                d: if opcode == 0x98e { Reg::ZERO } else { d },
                space,
                address,
                data: word.reg(if exchange { 64 } else { 32 }),
                compare: exchange.then(|| word.reg(32)),
                operation: if exchange {
                    0
                } else {
                    word.bits(87, if opcode == 0x98e { 3 } else { 4 }) as u8
                },
                kind: word.bits(73, 3) as u8,
            });
        }
        0xb1d => {
            return Some(Op::Bar {
                mode: word.bits(77, 2) as u8,
            });
        }
        // LDC, and 0xabb: ULDC indexed by the uniform register at 24 (a
        // uniform instruction, so its registers are uniform ones).
        0xb82 | 0xabb => {
            return Some(Op::Ldc {
                d,
                index: word.reg(24),
                bank: word.bits(54, 5) as u8,
                offset: word.bits(38, 16) as u16,
                size: word.bits(73, 3) as u8,
                mode: word.bits(78, 2) as u8,
            });
        }
        0xab9 => {
            let offset = word.bits(38, 16) as u16;
            // Bindless: the handle in the uniform register at 24.
            let src = if word.bit(91) {
                Src::BindlessConst {
                    handle: UniformReg(word.bits(24, 6) as u8),
                    offset,
                }
            } else {
                Src::Const {
                    bank: word.bits(54, 5) as u8,
                    offset,
                }
            };
            return Some(Op::Uldc {
                d: UniformReg(word.bits(16, 6) as u8),
                src,
                size: word.bits(73, 3) as u8,
            });
        }
        _ => {}
    }
    Some(match opcode & 0x1ff {
        0x002 if matches!(form, 1 | 4 | 5 | 6) => Op::Mov {
            d,
            src: word.source(1)?.src,
        },
        0x010 if (1..=7).contains(&form) => Op::Iadd3 {
            d,
            a: word.a(),
            b: word.source(1)?,
            c: word.source(2)?,
            carry_out: [word.pred(81, 84), word.pred(84, 87)].map(|pred| Pred {
                negated: false,
                ..pred
            }),
            carry_in: [word.pred(87, 90), word.pred(77, 80)],
            extended: word.bit(74),
        },
        0x024 if (1..=7).contains(&form) => Op::Imad {
            d,
            a: word.a(),
            b: word.source(1)?,
            c: word.source(2)?,
        },
        0x012 if (1..=7).contains(&form) => Op::Lop3 {
            d,
            a: word.reg(24),
            b: word.source(1)?,
            c: plain(word.source(2)?),
            lut: word.bits(72, 8) as u8,
            p: Pred {
                index: word.bits(81, 3) as u8,
                negated: false,
            },
            pand: word.bit(80),
        },
        0x023 if (1..=7).contains(&form) => Op::Ffma {
            d,
            a: word.a(),
            b: word.source(1)?,
            c: word.source(2)?,
            modes: word.float_modes(),
            dnz: word.bit(76),
        },
        0x020 if matches!(form, 1 | 4 | 5 | 6) => Op::Fmul {
            d,
            a: word.a(),
            b: word.source(1)?,
            modes: word.float_modes(),
            scale: word.bits(84, 3) as u8,
            dnz: word.bit(76),
        },
        // FADD's second source is the third slot in the forms that put a
        // wide operand there.
        0x021 if (1..=7).contains(&form) => Op::Fadd {
            d,
            a: word.a(),
            b: word.source(if matches!(form, 2 | 3 | 7) { 2 } else { 1 })?,
            modes: word.float_modes(),
        },
        0x00b if matches!(form, 1 | 4 | 5 | 6) => Op::Fsetp {
            p: word.bits(81, 3) as u8,
            q: word.bits(84, 3) as u8,
            a: word.a(),
            b: word.source(1)?,
            compare: float_compare(word.bits(76, 4)),
            flush_to_zero: word.bit(80),
            combine: word.combine()?,
            accumulate: word.pred(87, 90),
        },
        0x00c if matches!(form, 1 | 4 | 5 | 6) => Op::Isetp {
            p: word.bits(81, 3) as u8,
            q: word.bits(84, 3) as u8,
            a: word.a(),
            b: word.source(1)?,
            compare: int_compare(word.bits(76, 3)),
            signed: word.bit(73),
            combine: word.combine()?,
            accumulate: word.pred(87, 90),
            extended: word.bit(72).then(|| word.pred(68, 71)),
        },
        0x025 | 0x027 if (1..=7).contains(&form) => Op::ImadWide {
            d,
            a: word.a(),
            b: word.source(1)?,
            c: word.source(2)?,
            signed: word.bit(73),
            high: opcode & 0x1ff == 0x027,
            carry_out: word.bits(81, 3) as u8,
        },
        0x109 if matches!(form, 1 | 4 | 5 | 6) => Op::BitCount {
            d,
            src: word.source(1)?,
            invert: form == 1 && word.bit(63),
        },
        0x100 if matches!(form, 1 | 4 | 5 | 6) => Op::Flo {
            d,
            src: word.source(1)?,
            invert: form == 1 && word.bit(63),
            signed: word.bit(73),
            shift_amount: word.bit(74),
        },
        0x101 if matches!(form, 1 | 4 | 5 | 6) => Op::Brev {
            d,
            src: word.source(1)?,
        },
        0x013 if matches!(form, 1 | 4 | 5 | 6) => Op::Iabs {
            d,
            src: word.source(1)?,
        },
        0x011 if (1..=7).contains(&form) => {
            let mut b = word.source(1)?;
            // Bits 74-79 are .X and the shift, not the C slot's modifiers
            // (where B sits in the C slot).
            if matches!(form, 2 | 3 | 7) {
                b.neg = false;
            }
            // Bits 74 and 75 are .X and the shift, not C's modifiers.
            let c = plain(word.source(2)?);
            Op::Lea {
                d,
                a: word.reg(24),
                b,
                c,
                shift: word.bits(75, 5) as u8,
                high: word.bit(80),
                negate_shifted: word.bit(72),
                extended: word.bit(74),
                overflow: word.bits(81, 3) as u8,
                carry_in: word.pred(87, 90),
            }
        }
        0x019 if (1..=7).contains(&form) => Op::Shf {
            d,
            low: word.reg(24),
            shift: word.source(1)?,
            high: plain(word.source(2)?),
            kind: word.bits(73, 2) as u8,
            wrap: word.bit(75),
            right: word.bit(76),
            high_word: word.bit(80),
        },
        0x016 if (1..=7).contains(&form) => Op::Prmt {
            d,
            a: word.reg(24),
            selector: word.source(1)?,
            c: plain(word.source(2)?),
            mode: word.bits(72, 3) as u8,
        },
        0x01a if matches!(form, 1 | 4 | 5 | 6) => Op::Sgxt {
            d,
            a: word.reg(24),
            bits: word.source(1)?,
            signed: word.bit(73),
            wrap: word.bit(75),
        },
        0x007 | 0x008 if matches!(form, 1 | 4 | 5 | 6) => Op::Sel {
            d,
            a: word.a(),
            b: word.source(1)?,
            condition: word.pred(87, 90),
            float: opcode & 0x1ff == 0x008,
        },
        0x009 if matches!(form, 1 | 4 | 5 | 6) => Op::Fmnmx {
            d,
            a: word.a(),
            b: word.source(1)?,
            min: word.pred(87, 90),
        },
        0x017 if matches!(form, 1 | 4 | 5 | 6) => Op::Imnmx {
            d,
            a: word.a(),
            b: word.source(1)?,
            min: word.pred(87, 90),
            signed: word.bit(73),
        },
        // Bit 73 selects half precision, not translated yet.
        0x108 if matches!(form, 1 | 4 | 5 | 6) && !word.bit(73) => Op::Mufu {
            d,
            function: word.bits(74, 6) as u8,
            src: word.source(1)?,
        },
        0x105 if matches!(form, 1 | 4 | 5 | 6) => Op::F2i {
            d,
            src: word.source(1)?,
            signed: word.bit(72),
            dst_size: word.bits(75, 2) as u8,
            src_size: word.bits(84, 2) as u8,
            rounding: word.bits(78, 2) as u8,
        },
        0x106 | 0x045 if matches!(form, 1 | 4 | 5 | 6) => Op::I2f {
            d,
            src: word.source(1)?,
            signed: word.bit(74),
            dst_size: word.bits(75, 2) as u8,
            src_size: word.bits(84, 2) as u8,
            rounding: word.bits(78, 2) as u8,
        },
        0x107 if matches!(form, 1 | 4 | 5 | 6) => Op::Frnd {
            d,
            src: word.source(1)?,
            rounding: word.bits(78, 2) as u8,
            dst_size: word.bits(75, 2) as u8,
            src_size: word.bits(84, 2) as u8,
        },
        0x00a if matches!(form, 1 | 4 | 5 | 6) => Op::Fset {
            d,
            a: word.a(),
            b: word.source(1)?,
            compare: float_compare(word.bits(76, 4)),
            combine: word.combine()?,
            accumulate: word.pred(87, 90),
        },
        _ => return None,
    })
}

/// An operand without float modifiers, for operations whose modifier bits
/// mean something else.
fn plain(operand: Operand) -> Operand {
    Operand {
        abs: false,
        neg: false,
        ..operand
    }
}

fn float_compare(bits: u64) -> FloatCompare {
    use FloatCompare::*;
    [
        False, Lt, Eq, Le, Gt, Ne, Ge, Num, Nan, Ltu, Equ, Leu, Gtu, Neu, Geu, True,
    ][bits as usize & 15]
}

fn int_compare(bits: u64) -> IntCompare {
    use IntCompare::*;
    [False, Lt, Eq, Le, Gt, Ne, Ge, True][bits as usize & 7]
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// A word with `fields` (bit offset, width, value) set and a PT guard.
    fn word(fields: &[(u32, u32, u128)]) -> u128 {
        let mut raw = 7u128 << 12;
        for &(offset, width, value) in fields {
            assert!(value < 1u128 << width, "field at {offset} too wide");
            raw &= !(((1u128 << width) - 1) << offset);
            raw |= value << offset;
        }
        raw
    }

    fn reg(n: u8) -> Operand {
        Operand {
            src: Src::Reg(Reg(n)),
            abs: false,
            neg: false,
        }
    }

    #[test]
    fn mov_in_each_operand_form() {
        let base = |opcode: u128| [(0, 12, opcode), (16, 8, 4)];
        let cases = [
            (
                word(&[base(0x202)[0], base(0x202)[1], (32, 8, 9)]),
                Src::Reg(Reg(9)),
            ),
            (
                word(&[base(0x802)[0], base(0x802)[1], (32, 32, 0x3f80_0000)]),
                Src::Imm(0x3f80_0000),
            ),
            (
                word(&[base(0xa02)[0], base(0xa02)[1], (54, 5, 3), (38, 16, 0x1c)]),
                Src::Const {
                    bank: 3,
                    offset: 0x1c,
                },
            ),
            (
                word(&[base(0xc02)[0], base(0xc02)[1], (32, 6, 17)]),
                Src::Uniform(UniformReg(17)),
            ),
        ];
        for (raw, src) in cases {
            let instruction = decode(raw);
            assert_eq!(instruction.op, Op::Mov { d: Reg(4), src }, "{raw:#034x}");
            assert!(instruction.guard.is_true());
        }
    }

    #[test]
    fn a_constant_form_with_bit_91_takes_its_buffer_from_a_uniform_register() {
        let raw = word(&[
            (0, 12, 0xa02),
            (16, 8, 1),
            (91, 1, 1),
            (32, 6, 5),
            (38, 16, 0x40),
        ]);
        assert_eq!(
            decode(raw).op,
            Op::Mov {
                d: Reg(1),
                src: Src::BindlessConst {
                    handle: UniformReg(5),
                    offset: 0x40
                }
            }
        );
    }

    #[test]
    fn three_source_forms_place_the_wide_operand() {
        // IMAD form 1: registers in 24, 32, 64; form 2: immediate third,
        // register second in 64; form 6: uniform second.
        let imad = |form: u128, fields: &[(u32, u32, u128)]| {
            let mut all = vec![(0, 12, 0x024 | form << 9), (16, 8, 1), (24, 8, 2)];
            all.extend_from_slice(fields);
            decode(word(&all)).op
        };
        assert_eq!(
            imad(1, &[(32, 8, 3), (64, 8, 5)]),
            Op::Imad {
                d: Reg(1),
                a: reg(2),
                b: reg(3),
                c: reg(5),
            }
        );
        assert_eq!(
            imad(2, &[(32, 32, 0x10), (64, 8, 6)]),
            Op::Imad {
                d: Reg(1),
                a: reg(2),
                b: reg(6),
                c: Operand {
                    src: Src::Imm(0x10),
                    abs: false,
                    neg: false
                },
            }
        );
        assert_eq!(
            imad(6, &[(32, 6, 4), (64, 8, 7)]),
            Op::Imad {
                d: Reg(1),
                a: reg(2),
                b: Operand {
                    src: Src::Uniform(UniformReg(4)),
                    abs: false,
                    neg: false
                },
                c: reg(7),
            }
        );
    }

    #[test]
    fn float_operations_with_modifiers_and_modes() {
        // FFMA R0, -|R1|, R2, R3 with saturate, round toward zero, FTZ.
        let raw = word(&[
            (0, 12, 0x223),
            (16, 8, 0),
            (24, 8, 1),
            (72, 1, 1),
            (73, 1, 1),
            (32, 8, 2),
            (64, 8, 3),
            (77, 1, 1),
            (78, 2, 3),
            (80, 1, 1),
        ]);
        assert_eq!(
            decode(raw).op,
            Op::Ffma {
                d: Reg(0),
                a: Operand {
                    src: Src::Reg(Reg(1)),
                    abs: true,
                    neg: true
                },
                b: reg(2),
                c: reg(3),
                modes: FloatModes {
                    saturate: true,
                    rounding: 3,
                    flush_to_zero: true
                },
                dnz: false,
            }
        );
        // FADD form 2 takes its second source from the third slot.
        let raw = word(&[
            (0, 12, 0x421),
            (16, 8, 8),
            (24, 8, 9),
            (32, 32, 0x4000_0000),
        ]);
        let Op::Fadd { b, .. } = decode(raw).op else {
            panic!("FADD")
        };
        assert_eq!(b.src, Src::Imm(0x4000_0000));
    }

    #[test]
    fn address_shift_and_predicate_logic_fields() {
        // LEA.HI.X R6, R1, R4, R2, 0x2, P2
        let raw = word(&[
            (0, 12, 0x211),
            (16, 8, 6),
            (24, 8, 1),
            (32, 8, 4),
            (64, 8, 2),
            (74, 1, 1),
            (75, 5, 2),
            (80, 1, 1),
            (81, 3, 7),
            (87, 3, 2),
        ]);
        let Op::Lea {
            d,
            a,
            b,
            c,
            shift,
            high,
            extended,
            overflow,
            carry_in,
            ..
        } = decode(raw).op
        else {
            panic!("LEA")
        };
        assert_eq!((d, a, b, c), (Reg(6), Reg(1), reg(4), reg(2)));
        assert_eq!((shift, high, extended, overflow), (2, true, true, 7));
        assert_eq!(carry_in.index, 2);
        // SHF.R.S32.HI R12, RZ, 0x1f, R1
        let raw = word(&[
            (0, 12, 0x819),
            (16, 8, 12),
            (24, 8, 255),
            (32, 32, 0x1f),
            (64, 8, 1),
            (73, 2, 2),
            (76, 1, 1),
            (80, 1, 1),
        ]);
        let Op::Shf {
            low,
            shift,
            high,
            kind,
            right,
            high_word,
            wrap,
            ..
        } = decode(raw).op
        else {
            panic!("SHF")
        };
        assert_eq!((low, shift.src, high), (Reg::ZERO, Src::Imm(0x1f), reg(1)));
        assert_eq!((kind, right, high_word, wrap), (2, true, true, false));
        // PLOP3.LUT P3, P4, P0, P1, PT, 0x80, 0x8: the first table split
        // over bits 64-66 and 72-76, the second in the destination field.
        let raw = word(&[
            (0, 12, 0x81c),
            (16, 8, 0x08),
            (64, 3, 0),
            (72, 5, 0x10),
            (81, 3, 3),
            (84, 3, 4),
            (87, 3, 0),
            (77, 3, 1),
            (68, 3, 7),
        ]);
        let Op::Plop3 { p, q, srcs, luts } = decode(raw).op else {
            panic!("PLOP3")
        };
        assert_eq!((p, q, luts), (3, 4, [0x80, 0x08]));
        assert_eq!(srcs.map(|pred| pred.index), [0, 1, 7]);
        // ULDC.64 UR4, c[0x0][0x118]
        let raw = word(&[
            (0, 12, 0xab9),
            (16, 6, 4),
            (54, 5, 0),
            (38, 16, 0x118),
            (73, 3, 5),
        ]);
        assert_eq!(
            decode(raw).op,
            Op::Uldc {
                d: UniformReg(4),
                src: Src::Const {
                    bank: 0,
                    offset: 0x118
                },
                size: 5,
            }
        );
        // I2FP.F32.U32 R9, R2 decodes as I2F.
        let raw = word(&[
            (0, 12, 0x245),
            (16, 8, 9),
            (32, 8, 2),
            (75, 2, 2),
            (84, 2, 2),
        ]);
        assert!(matches!(
            decode(raw).op,
            Op::I2f {
                signed: false,
                dst_size: 2,
                src_size: 2,
                ..
            }
        ));
    }

    #[test]
    fn predicate_setters_branches_and_guards() {
        // @!P2 ISETP.GE.U32.OR P1, P6, R4, 0x7, !P3
        let raw = word(&[
            (0, 12, 0x80c),
            (12, 4, 0xa),
            (81, 3, 1),
            (84, 3, 6),
            (24, 8, 4),
            (32, 32, 7),
            (76, 3, 6),
            (74, 2, 1),
            (87, 3, 3),
            (90, 1, 1),
        ]);
        let instruction = decode(raw);
        assert_eq!(
            instruction.guard,
            Pred {
                index: 2,
                negated: true
            }
        );
        assert_eq!(
            instruction.op,
            Op::Isetp {
                p: 1,
                q: 6,
                a: reg(4),
                b: Operand {
                    src: Src::Imm(7),
                    abs: false,
                    neg: false
                },
                compare: IntCompare::Ge,
                signed: false,
                combine: Combine::Or,
                accumulate: Pred {
                    index: 3,
                    negated: true
                },
                extended: None,
            }
        );
        // BRA -0x20 (back two instructions from the next one).
        let field = ((-0x20i64 as u64) & ((1 << 50) - 1)) as u128;
        let raw = word(&[(0, 12, 0x947), (32, 50, field), (87, 3, 7)]);
        assert_eq!(
            decode(raw).op,
            Op::Bra {
                offset: -0x20,
                mode: 0,
                condition: Pred::TRUE
            }
        );
        assert_eq!(decode(word(&[(0, 12, 0x94d)])).op, Op::Exit);
    }

    #[test]
    fn unknown_forms_keep_their_bits_and_nothing_panics() {
        // MOV has no form 2; opcode 0x1ff is not decoded.
        for raw in [word(&[(0, 12, 0x402)]), word(&[(0, 12, 0x1ff)])] {
            let instruction = decode(raw);
            assert!(matches!(instruction.op, Op::Unknown { .. }), "{raw:#x}");
            assert_eq!(instruction.raw, raw);
        }
        // Every opcode and form, with arbitrary other bits.
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for opcode in 0..0x1000u128 {
            for _ in 0..16 {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let raw =
                    (u128::from(state) << 64 | u128::from(state.rotate_left(29))) & !0xfff | opcode;
                assert_eq!(decode(raw).raw, raw);
            }
        }
        let program: Vec<u8> = (0..40u8).collect();
        assert_eq!(decode_program(&program).len(), 2);
    }
}
