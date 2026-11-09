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

/// A decoded instruction: its guard and operation, and the word it came
/// from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Instruction {
    pub raw: u128,
    pub guard: Pred,
    pub op: Op,
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
    /// IADD3 d, a, b, c, with up to two carry outputs and two carry
    /// inputs (`.X`).
    Iadd3 {
        d: Reg,
        a: Operand,
        b: Operand,
        c: Operand,
        carry_out: [Pred; 2],
        carry_in: [Pred; 2],
    },
    /// IMAD d, a, b, c: the low word of a * b + c.
    Imad {
        d: Reg,
        a: Operand,
        b: Operand,
        c: Operand,
    },
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
    /// S2R d, special register (the Maxwell numbering: 0 lane id, 33-35
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
    let op = decode_op(word, opcode).unwrap_or(Op::Unknown { opcode });
    Instruction {
        raw,
        guard: word.pred(12, 15),
        op,
    }
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
        0x919 => {
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
            c: word.source(2)?,
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
        _ => return None,
    })
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
