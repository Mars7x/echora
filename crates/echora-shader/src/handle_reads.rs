//! Find bindless handles derived from draw-uniform inputs. The renderer
//! uses these lookups to populate descriptors before draws that read their
//! handles from a buffer. Unrecognized instructions end the analysis.

use crate::decode::{
    Address, Combine, Instruction, IntCompare, Op, Operand, Pred, Reg, Space, Src, UniformReg,
};

type Id = usize;
type Value = Option<Id>;

/// A shader word the renderer can read from the same inputs as the draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Input {
    Constant { bank: u8, offset: u16 },
    GlobalWord { address: u64, word: u8, words: u8 },
}

#[derive(Clone, Copy, Debug)]
enum Expr {
    Literal(u32),
    Constant {
        bank: u8,
        offset: u16,
    },
    Load {
        base: [Id; 2],
        offset: [Id; 2],
        byte_offset: i32,
        word: u8,
        words: u8,
    },
    Add(Id, Id),
    Multiply(Id, Id),
    MultiplyAddWide {
        a: Id,
        b: Id,
        c: [Id; 2],
        high: bool,
    },
    LocalId(u8),
    GroupId(u8),
    And(Id, Id),
    ShiftLeft(Id, u8),
    NotEqual(Id, Id),
    Not(Id),
    Select(Id, Id, Id),
}

/// Expressions for handles read in a straight-line shader prefix, with
/// each register value derived from constants or uniform global addresses.
#[derive(Debug)]
pub struct HandleReads {
    expressions: Vec<Expr>,
    handles: Vec<Id>,
}

impl HandleReads {
    /// Reads from the native graphics or compute program's instruction
    /// stream. Stages without analyzable handles return no plan.
    pub fn from_program(program: &[u8]) -> Option<Self> {
        let code = if crate::is_graphics(program) {
            program.get(crate::GRAPHICS_HEADER_SIZE..)?
        } else {
            program
        };
        let compiled = Self::compile(&crate::program_code(code).ok()?);
        (!compiled.is_empty()).then_some(compiled)
    }

    /// Compile the part of the shader that can be interpreted without
    /// executing any per-invocation work or modeling control flow.
    pub fn compile(code: &[Instruction]) -> Self {
        let mut compiler = Compiler::new();
        // Limit both analysis and expression evaluation on large programs.
        for &instruction in code.iter().take(512) {
            if !compiler.instruction(instruction) {
                break;
            }
        }
        Self {
            expressions: compiler.expressions,
            handles: compiler.handles,
        }
    }

    /// Whether any of the shader's bindless handles could be expressed.
    pub fn is_empty(&self) -> bool {
        self.handles.is_empty()
    }

    /// Report the handles whose inputs were available for this draw.
    pub fn evaluate(
        &self,
        mut read: impl FnMut(Input) -> Option<u32>,
        mut handle: impl FnMut(u32),
    ) {
        let mut local = [None; 128];
        let mut allocated;
        let evaluated = if self.expressions.len() <= local.len() {
            &mut local[..self.expressions.len()]
        } else {
            allocated = vec![None; self.expressions.len()];
            &mut allocated
        };
        for &id in &self.handles {
            if let Some(word) = self.word(id, &mut read, evaluated) {
                handle(word);
            }
        }
    }

    fn word(
        &self,
        id: Id,
        read: &mut impl FnMut(Input) -> Option<u32>,
        evaluated: &mut [Option<Option<u32>>],
    ) -> Option<u32> {
        if let Some(result) = evaluated[id] {
            return result;
        }
        let result = self.evaluate_expr(self.expressions[id], read, evaluated);
        evaluated[id] = Some(result);
        result
    }

    fn evaluate_expr(
        &self,
        expression: Expr,
        read: &mut impl FnMut(Input) -> Option<u32>,
        evaluated: &mut [Option<Option<u32>>],
    ) -> Option<u32> {
        let mut get = |id| self.word(id, read, evaluated);
        Some(match expression {
            Expr::Literal(word) => word,
            Expr::Constant { bank, offset } => return read(Input::Constant { bank, offset }),
            Expr::Load {
                base: [lo, hi],
                offset: [off_lo, off_hi],
                byte_offset,
                word,
                words,
            } => {
                let base = u64::from(get(lo)?) | u64::from(get(hi)?) << 32;
                let offset = u64::from(get(off_lo)?) | u64::from(get(off_hi)?) << 32;
                let address = base
                    .wrapping_add(offset)
                    .wrapping_add_signed(i64::from(byte_offset));
                return read(Input::GlobalWord {
                    address,
                    word,
                    words,
                });
            }
            Expr::Add(a, b) => get(a)?.wrapping_add(get(b)?),
            Expr::Multiply(a, b) => get(a)?.wrapping_mul(get(b)?),
            Expr::MultiplyAddWide {
                a,
                b,
                c: [lo, hi],
                high,
            } => wide_word(get(a)?, get(b)?, get(lo)?, get(hi)?, high),

            Expr::LocalId(_) | Expr::GroupId(_) => return None,
            Expr::And(a, b) => get(a)? & get(b)?,
            Expr::ShiftLeft(id, shift) => get(id)?.wrapping_shl(u32::from(shift)),
            Expr::NotEqual(a, b) => u32::from(get(a)? != get(b)?),
            Expr::Not(id) => u32::from(get(id)? == 0),
            Expr::Select(condition, a, b) => {
                let selected = if get(condition)? != 0 { a } else { b };
                return get(selected);
            }
        })
    }
}

fn wide_word(a: u32, b: u32, lo: u32, hi: u32, high: bool) -> u32 {
    let c = u64::from(lo) | u64::from(hi) << 32;
    let value = (u64::from(a) * u64::from(b)).wrapping_add(c);
    (value >> if high { 32 } else { 0 }) as u32
}

#[derive(Clone, Copy, Debug)]
struct GlobalAddress {
    base: [Id; 2],
    offset: [Id; 2],
    byte_offset: i32,
}

/// A conservative address plan for the possible global stores and atomics
/// of a compute shader. If any write cannot be bounded, the plan fails.
#[derive(Debug)]
pub struct GlobalWrites {
    expressions: Vec<Expr>,
    writes: Vec<GlobalAddress>,
}

impl GlobalWrites {
    /// Build a plan for a straight-line compute program.
    pub fn from_program(program: &[u8]) -> Option<Self> {
        if crate::is_graphics(program) {
            return None;
        }
        let code = crate::program_code(program).ok()?;
        if code.len() > 512 {
            return None;
        }
        Self::compile(&code).or_else(|| Self::compile_uniform_stores(&code))
    }

    fn compile(code: &[Instruction]) -> Option<Self> {
        if code.len() > 512 {
            return None;
        }
        let mut state = Compiler::new();
        let mut writes = Vec::new();
        for &instruction in code {
            match instruction.op {
                Op::Exit { condition } if condition.is_true() && instruction.guard.is_true() => {
                    break;
                }
                Op::Exit { .. } => continue,
                Op::Store {
                    space: Space::Global,
                    address,
                    ..
                }
                | Op::Atomic {
                    space: Space::Global,
                    address,
                    ..
                } => {
                    writes.push(state.global_address(address)?);
                    if let Op::Atomic { d, .. } = instruction.op {
                        // The old value is unknown. Invalidate the pair
                        // as well, for instructions with a wide result.
                        state.set_register(d, false, None);
                        state.set_register(Reg(d.0.saturating_add(1)), false, None);
                    }
                }
                _ if !state.instruction(instruction) => return None,
                _ => {}
            }
        }
        (!writes.is_empty()).then_some(Self {
            expressions: state.expressions,
            writes,
        })
    }

    /// Global operations with zero general offset can use a uniform
    /// pointer established before divergent control flow. Use it only
    /// when every path to a store executes the whole initialization and
    /// nothing after it changes a uniform register.
    fn compile_uniform_stores(code: &[Instruction]) -> Option<Self> {
        let writes: Vec<_> = code
            .iter()
            .enumerate()
            .filter_map(|(index, instruction)| match instruction.op {
                Op::Store {
                    space: Space::Global,
                    address,
                    ..
                }
                | Op::Atomic {
                    space: Space::Global,
                    address,
                    ..
                } => Some((index, address)),
                _ => None,
            })
            .collect();
        let &(first, _) = writes.first()?;
        let mut pointer_words = 0u64;
        for &(_, address) in &writes {
            if address.reg != Reg::ZERO {
                return None;
            }
            if let Some((reg, wide)) = address.uniform {
                pointer_words |= 1u64 << reg.0;
                if wide {
                    pointer_words |= 1u64 << reg.0.saturating_add(1).min(63);
                }
            }
        }
        let changes_uniforms = |instruction: &Instruction| {
            instruction.uniform
                || matches!(
                    instruction.op,
                    Op::Uldc { .. }
                        | Op::R2ur { .. }
                        | Op::Redux { .. }
                        | Op::Vote {
                            uniform_out: true,
                            ..
                        }
                )
        };
        let changes_pointer = |instruction: &Instruction| match instruction.op {
            Op::R2ur { d, .. } => pointer_words & (1u64 << d.0.min(63)) != 0,
            Op::Redux { d, .. } => pointer_words & (1u64 << d.0.min(63)) != 0,
            Op::Vote {
                d,
                uniform_out: true,
                ..
            } => pointer_words & (1u64 << d.0.min(63)) != 0,
            _ => changes_uniforms(instruction),
        };
        let last = code[..first].iter().rposition(changes_pointer)?;
        if code[last + 1..].iter().any(changes_pointer) {
            return None;
        }
        let mut edges = Vec::new();
        let mut start = 0;
        for (source, instruction) in code.iter().enumerate() {
            match instruction.op {
                Op::Call { .. } | Op::Ret { .. } | Op::Brx { .. } => return None,
                Op::Bra {
                    offset, mode: 0, ..
                } if offset % 16 == 0 => {
                    let target = usize::try_from(source as i64 + 1 + offset / 16).ok()?;
                    if target >= code.len() {
                        return None;
                    }
                    if target <= last {
                        start = start.max(target);
                    }
                    if source < last {
                        start = start.max(source + 1);
                    }
                    edges.push((source, target));
                }
                Op::Bra { .. } => return None,
                _ => {}
            }
        }
        if edges
            .into_iter()
            .any(|(source, target)| source < start && target > last)
        {
            return None;
        }
        let mut state = Compiler::new();
        for &instruction in &code[start..=last] {
            if changes_uniforms(&instruction) && !state.instruction(instruction) {
                return None;
            }
        }
        let writes = writes
            .into_iter()
            .map(|(_, address)| state.global_address(address))
            .collect::<Option<Vec<_>>>()?;
        Some(Self {
            expressions: state.expressions,
            writes,
        })
    }

    /// Byte intervals covering every invocation's possible writes. Each
    /// count comes from the actual host dispatch and shader local size.
    pub fn intervals(
        &self,
        groups: [u32; 3],
        local_size: [u32; 3],
        mut constant: impl FnMut(u8, u16) -> Option<u32>,
    ) -> Option<Vec<std::ops::Range<u64>>> {
        if groups.contains(&0) {
            return Some(Vec::new());
        }
        let interval = |kind: u8, axis: u8| {
            let count = if kind == 0 { local_size } else { groups };
            Some([0, count[usize::from(axis)].checked_sub(1)?])
        };
        let mut cache = vec![None; self.expressions.len()];
        let mut word = |id| self.word_bounds(id, &mut constant, interval, &mut cache);
        let mut ranges = Vec::with_capacity(self.writes.len());
        for &address in &self.writes {
            let mut pair = |[lo, hi]: [Id; 2]| -> Option<[u64; 2]> {
                let low = word(lo)?;
                let high = word(hi)?;
                Some(std::array::from_fn(|n| {
                    u64::from(low[n]) | u64::from(high[n]) << 32
                }))
            };
            let [base_lo, base_hi] = pair(address.base)?;
            let [offset_lo, offset_hi] = pair(address.offset)?;
            let offset = i64::from(address.byte_offset);
            let first = base_lo.checked_add(offset_lo)?.checked_add_signed(offset)?;
            let last = base_hi.checked_add(offset_hi)?.checked_add_signed(offset)?;
            // Subword stores can atomically update their aligned word.
            // At most 16 bytes are written by one translated instruction.
            ranges.push((first & !3)..last.checked_add(16)?);
        }
        Some(ranges)
    }

    fn word_bounds(
        &self,
        id: Id,
        constant: &mut impl FnMut(u8, u16) -> Option<u32>,
        invocation: impl Fn(u8, u8) -> Option<[u32; 2]> + Copy,
        known: &mut [Option<[u32; 2]>],
    ) -> Option<[u32; 2]> {
        if let Some(bounds) = known[id] {
            return Some(bounds);
        }
        let mut get = |id| self.word_bounds(id, constant, invocation, known);
        let bounds = match self.expressions[id] {
            Expr::Literal(value) => [value; 2],
            Expr::Constant { bank, offset } => [constant(bank, offset)?; 2],
            Expr::LocalId(axis) => invocation(0, axis)?,
            Expr::GroupId(axis) => invocation(1, axis)?,
            Expr::Add(a, b) => {
                let [a_min, a_max] = get(a)?;
                let [b_min, b_max] = get(b)?;
                [a_min.checked_add(b_min)?, a_max.checked_add(b_max)?]
            }
            Expr::Multiply(a, b) => {
                let [a_min, a_max] = get(a)?;
                let [b_min, b_max] = get(b)?;
                [a_min.checked_mul(b_min)?, a_max.checked_mul(b_max)?]
            }
            Expr::MultiplyAddWide {
                a,
                b,
                c: [lo, hi],
                high,
            } => {
                let mut exact = |id| {
                    let [min, max] = get(id)?;
                    (min == max).then_some(min)
                };
                [wide_word(exact(a)?, exact(b)?, exact(lo)?, exact(hi)?, high); 2]
            }
            Expr::And(a, b) => [0, get(a)?[1].min(get(b)?[1])],
            Expr::ShiftLeft(id, shift) => {
                let [min, max] = get(id)?;
                let multiplier = 1u32 << shift;
                [min.checked_mul(multiplier)?, max.checked_mul(multiplier)?]
            }
            Expr::NotEqual(..) | Expr::Not(..) => [0, 1],
            Expr::Select(_, a, b) => {
                let [a_min, a_max] = get(a)?;
                let [b_min, b_max] = get(b)?;
                [a_min.min(b_min), a_max.max(b_max)]
            }
            Expr::Load { .. } => return None,
        };
        known[id] = Some(bounds);
        Some(bounds)
    }
}

struct Compiler {
    expressions: Vec<Expr>,
    handles: Vec<Id>,
    registers: [Value; 256],
    uniforms: [Value; 64],
    predicates: [Value; 8],
}

impl Compiler {
    fn new() -> Self {
        let mut registers = [None; 256];
        registers[255] = Some(0);
        let mut uniforms = [None; 64];
        uniforms[63] = Some(0);
        let mut predicates = [None; 8];
        predicates[7] = Some(1);
        Self {
            expressions: vec![Expr::Literal(0), Expr::Literal(1)],
            handles: Vec::new(),
            registers,
            uniforms,
            predicates,
        }
    }

    fn push(&mut self, value: Expr) -> Id {
        let id = self.expressions.len();
        self.expressions.push(value);
        id
    }

    fn source(&mut self, source: Src, uniform: bool) -> Value {
        Some(match source {
            Src::Reg(reg) => self.register(reg, uniform)?,
            Src::Uniform(reg) => self.uniforms.get(reg.0 as usize).copied().flatten()?,
            Src::Imm(0) => 0,
            Src::Imm(1) => 1,
            Src::Imm(value) => self.push(Expr::Literal(value)),
            Src::Const { bank, offset } => self.push(Expr::Constant { bank, offset }),
            Src::BindlessConst { .. } => return None,
        })
    }

    fn operand(&mut self, value: Operand, uniform: bool) -> Value {
        if value.neg || value.abs {
            return None;
        }
        self.source(value.src, uniform)
    }

    fn register(&self, reg: Reg, uniform: bool) -> Value {
        if uniform {
            return self.uniforms.get(reg.0 as usize).copied().flatten();
        }
        self.registers[reg.0 as usize]
    }

    fn set_register(&mut self, reg: Reg, uniform: bool, value: Value) {
        if uniform {
            if reg.0 < UniformReg::ZERO.0 {
                self.uniforms[reg.0 as usize] = value;
            }
        } else if reg != Reg::ZERO {
            self.registers[reg.0 as usize] = value;
        }
    }

    fn predicate(&mut self, pred: Pred) -> Value {
        let id = self
            .predicates
            .get(pred.index as usize)
            .copied()
            .flatten()?;
        Some(if pred.negated {
            self.push(Expr::Not(id))
        } else {
            id
        })
    }

    fn binary(&mut self, a: Value, b: Value, operation: fn(Id, Id) -> Expr) -> Value {
        Some(self.push(operation(a?, b?)))
    }

    fn select(&mut self, pred: Pred, a: Value, b: Value) -> Value {
        if pred.is_true() {
            return a;
        }
        if pred.index == 7 {
            return b;
        }
        if a == b {
            return a;
        }
        let pred = self.predicate(pred)?;
        Some(self.push(Expr::Select(pred, a?, b?)))
    }

    fn global_address(&self, address: Address) -> Option<GlobalAddress> {
        let low = self.register(address.reg, false)?;
        let high = if address.wide {
            self.register(Reg(address.reg.0.saturating_add(1)), false)?
        } else {
            0
        };
        let offset = match address.uniform {
            Some((reg, wide)) => {
                let low = self.uniforms.get(reg.0 as usize).copied().flatten()?;
                let high = if wide {
                    self.uniforms
                        .get(reg.0.saturating_add(1).min(63) as usize)
                        .copied()
                        .flatten()?
                } else {
                    0
                };
                [low, high]
            }
            None => [0, 0],
        };
        Some(GlobalAddress {
            base: [low, high],
            offset,
            byte_offset: address.offset,
        })
    }

    fn load(&mut self, address: Address, word: u8, words: u8) -> Value {
        let address = self.global_address(address)?;
        Some(self.push(Expr::Load {
            base: address.base,
            offset: address.offset,
            byte_offset: address.byte_offset,
            word,
            words,
        }))
    }

    fn instruction(
        &mut self,
        Instruction {
            guard, op, uniform, ..
        }: Instruction,
    ) -> bool {
        let (dst, value) = match op {
            Op::Mov { d, src, lanes: 15 } => (d, self.source(src, uniform)),
            Op::Mov { d, .. } => (d, None),
            Op::Uldc {
                d,
                src: Src::Const { bank, offset },
                size: size @ (4 | 5),
            } if guard.is_true() => {
                let lo = self.source(Src::Const { bank, offset }, true);
                self.set_register(Reg(d.0), true, lo);
                if size == 5 {
                    let hi = self.source(
                        Src::Const {
                            bank,
                            offset: offset.wrapping_add(4),
                        },
                        true,
                    );
                    self.set_register(Reg(d.0.saturating_add(1)), true, hi);
                }
                return true;
            }
            Op::R2ur { d, src } => {
                let value = guard.is_true().then(|| self.register(src, false)).flatten();
                self.set_register(Reg(d.0), true, value);
                return true;
            }
            Op::Lop3 {
                d,
                a,
                b,
                lut,
                p,
                pand: false,
                ..
            } if p.is_true() && lut == 0xc0 => {
                let lower = self.register(a, uniform);
                let upper = self.operand(b, uniform);
                (d, self.binary(lower, upper, Expr::And))
            }
            Op::Lop3 {
                d,
                p: Pred { index: 7, .. },
                ..
            } => (d, None),
            Op::Iadd3 {
                d,
                a,
                b,
                c,
                carry_out,
                extended: false,
                ..
            } if carry_out.iter().all(|p| p.index == 7) => {
                let a = self.operand(a, uniform);
                let b = self.operand(b, uniform);
                let c = self.operand(c, uniform);
                let sum = self.binary(a, b, Expr::Add);
                (
                    d,
                    if c == Some(0) {
                        sum
                    } else {
                        self.binary(sum, c, Expr::Add)
                    },
                )
            }
            Op::Imad {
                d,
                a,
                b,
                c,
                carry_in: None,
            } => {
                // The shader's integer multiply reads a and b as raw
                // words. Only the third term has a negate modifier.
                let a = self.source(a.src, uniform);
                let b = self.source(b.src, uniform);
                let c = if c.neg && !matches!(c.src, Src::Imm(_)) {
                    None
                } else {
                    self.source(c.src, uniform)
                };
                let product = self.binary(a, b, Expr::Multiply);
                (d, self.binary(product, c, Expr::Add))
            }
            Op::Imad { d, .. } => (d, None),
            Op::ImadWide {
                d,
                a,
                b,
                c:
                    Operand {
                        src: Src::Reg(addend),
                        neg: false,
                        abs: false,
                    },
                signed: false,
                high: false,
                carry_out: 7,
                carry_in: None,
            } if guard.is_true() => {
                let make = |state: &mut Self| -> Option<(Id, Id)> {
                    let a = state.source(a.src, uniform)?;
                    let b = state.source(b.src, uniform)?;
                    let lo = state.register(addend, uniform)?;
                    let hi = if lo == 0 && addend.0 == if uniform { 63 } else { 255 } {
                        0
                    } else {
                        state.register(Reg(addend.0.saturating_add(1)), uniform)?
                    };
                    let c = [lo, hi];
                    let low = state.push(Expr::MultiplyAddWide {
                        a,
                        b,
                        c,
                        high: false,
                    });
                    let high = state.push(Expr::MultiplyAddWide {
                        a,
                        b,
                        c,
                        high: true,
                    });
                    Some((low, high))
                };
                let (lower, upper) =
                    make(self).map_or((None, None), |(lo, hi)| (Some(lo), Some(hi)));
                self.set_register(d, uniform, lower);
                self.set_register(Reg(d.0.saturating_add(1)), uniform, upper);
                return true;
            }
            Op::S2r { d, special } if !uniform => {
                let expression = match special {
                    0x21..=0x23 => Some(Expr::LocalId(special - 0x21)),
                    0x25..=0x27 => Some(Expr::GroupId(special - 0x25)),
                    _ => None,
                };
                (d, expression.map(|expression| self.push(expression)))
            }
            Op::Lea {
                d,
                a,
                b,
                shift,
                high: false,
                negate_shifted: false,
                extended: false,
                overflow: 7,
                ..
            } if shift < 32 => {
                let a = self
                    .register(a, uniform)
                    .map(|id| self.push(Expr::ShiftLeft(id, shift)));
                let b = self.operand(b, uniform);
                (d, self.binary(a, b, Expr::Add))
            }
            Op::Shf {
                d,
                low,
                high,
                shift:
                    Operand {
                        src: Src::Imm(shift),
                        abs: false,
                        neg: false,
                    },
                kind: 2 | 3,
                wrap: false,
                right: false,
                high_word,
                ..
            } if shift < 32 && (!high_word || self.register(low, uniform) == Some(0)) => {
                let value = if high_word {
                    self.operand(high, uniform)
                } else {
                    self.register(low, uniform)
                };
                (
                    d,
                    value.map(|id| self.push(Expr::ShiftLeft(id, shift as u8))),
                )
            }
            Op::Shf { d, .. } => (d, None),
            Op::Sel {
                d,
                a,
                b,
                condition,
                float: false,
            } if !uniform => {
                let a = self.operand(a, false);
                let b = self.operand(b, false);
                (d, self.select(condition, a, b))
            }
            Op::Isetp {
                p,
                q,
                a,
                b,
                compare,
                combine,
                accumulate,
                extended,
                ..
            } if !uniform => {
                let result = if guard.is_true()
                    && compare == IntCompare::Ne
                    && combine == Combine::And
                    && accumulate.is_true()
                    && extended.is_none()
                {
                    let a = self.operand(a, false);
                    let b = self.operand(b, false);
                    self.binary(a, b, Expr::NotEqual)
                } else {
                    None
                };
                for (index, value) in [(p, result), (q, result.map(|id| self.push(Expr::Not(id))))]
                {
                    if index < 7 {
                        self.predicates[index as usize] = value;
                    }
                }
                return true;
            }
            Op::Fsetp { p, q, .. } if !uniform => {
                for index in [p, q] {
                    if index < 7 {
                        self.predicates[index as usize] = None;
                    }
                }
                return true;
            }
            Op::Load {
                d,
                space: Space::Global,
                address,
                size: size @ (4 | 5),
                condition,
                strong: false,
            } if !uniform && guard.is_true() && condition.is_true() => {
                let words = size - 3;
                let low = self.load(address, 0, words);
                let high = (words == 2).then(|| self.load(address, 1, words));
                self.set_register(d, false, low);
                if let Some(high) = high {
                    self.set_register(Reg(d.0.saturating_add(1)), false, high);
                }
                return true;
            }
            Op::Tex(texture) => {
                if let Some(reg) = texture.handle
                    && let Some(value) = self.uniforms.get(reg.0 as usize).copied().flatten()
                    && !self.handles.contains(&value)
                {
                    self.handles.push(value);
                }
                for reg in texture.output_registers() {
                    self.set_register(reg, false, None);
                }
                return true;
            }
            Op::Cs2r { d, wide, .. } if !uniform => {
                if wide {
                    self.set_register(Reg(d.0.saturating_add(1)), false, None);
                }
                (d, None)
            }
            Op::Fadd { d, .. }
            | Op::Fmul { d, .. }
            | Op::Ffma { d, .. }
            | Op::Fmnmx { d, .. }
            | Op::Imnmx { d, .. }
            | Op::Fset { d, .. }
            | Op::F2i {
                d, dst_size: 0..=2, ..
            }
            | Op::I2f { d, dst_size: 2, .. }
            | Op::Sel { d, .. }
            | Op::Mufu { d, .. }
            | Op::Ipa { d, .. } => (d, None),
            Op::Depbar => return true,
            _ => return false,
        };
        let value = if uniform && !guard.is_true() {
            None
        } else {
            self.select(guard, value, self.register(dst, uniform))
        };
        self.set_register(dst, uniform, value);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::{Texture, TextureOp};

    fn op(op: Op) -> Instruction {
        Instruction {
            raw: 0,
            guard: Pred::TRUE,
            op,
            uniform: false,
        }
    }

    #[test]
    fn global_write_bounds_cover_dispatch_coordinates_and_reject_wrap() {
        let base = op(Op::Uldc {
            d: UniformReg(4),
            src: Src::Const {
                bank: 0,
                offset: 32,
            },
            size: 5,
        });
        let group = op(Op::S2r {
            d: Reg(1),
            special: 0x25,
        });
        let local = op(Op::S2r {
            d: Reg(2),
            special: 0x21,
        });
        let operand = |src| Operand {
            src,
            abs: false,
            neg: false,
        };
        let offset = op(Op::Imad {
            d: Reg(3),
            a: operand(Src::Reg(Reg(1))),
            b: operand(Src::Imm(64)),
            c: operand(Src::Reg(Reg(2))),
            carry_in: None,
        });
        let store = op(Op::Store {
            data: Reg(0),
            space: Space::Global,
            address: Address {
                reg: Reg(3),
                wide: false,
                scale: 1,
                uniform: Some((UniformReg(4), true)),
                offset: -16,
            },
            size: 4,
            strong: false,
        });
        let exit = op(Op::Exit {
            condition: Pred::TRUE,
        });
        let code = [base, group, local, offset, store, exit];
        let plan = GlobalWrites::compile(&code).unwrap();
        let constant = |_, offset| match offset {
            32 => Some(0x2000),
            36 => Some(0x10),
            _ => None,
        };
        assert_eq!(
            plan.intervals([3, 1, 1], [8, 1, 1], constant).unwrap(),
            Vec::from_iter(std::iter::once(0x10_0000_1ff0..0x10_0000_2087))
        );
        assert!(
            plan.intervals([u32::MAX, 1, 1], [8, 1, 1], constant)
                .is_none()
        );
        assert!(
            plan.intervals([0, 1, 1], [8, 1, 1], constant)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn uniform_write_bounds_require_every_branch_to_initialize_the_pointer() {
        let base = op(Op::Uldc {
            d: UniformReg(4),
            src: Src::Const {
                bank: 0,
                offset: 32,
            },
            size: 5,
        });
        let branch = op(Op::Bra {
            offset: 16,
            mode: 0,
            condition: Pred {
                index: 0,
                negated: false,
            },
        });
        let vote = |d| {
            op(Op::Vote {
                d: Reg(d),
                p: 7,
                condition: Pred::TRUE,
                mode: 0,
                uniform_out: true,
            })
        };
        let store = op(Op::Store {
            data: Reg(0),
            space: Space::Global,
            address: Address {
                reg: Reg::ZERO,
                wide: true,
                scale: 1,
                uniform: Some((UniformReg(4), true)),
                offset: 20,
            },
            size: 4,
            strong: false,
        });
        let exit = op(Op::Exit {
            condition: Pred::TRUE,
        });
        let plan =
            GlobalWrites::compile_uniform_stores(&[base, branch, vote(6), store, exit]).unwrap();
        let constant = |_, offset| match offset {
            32 => Some(0x1000),
            36 => Some(0),
            _ => None,
        };
        assert_eq!(
            plan.intervals([1; 3], [1; 3], constant).unwrap(),
            Vec::from_iter(std::iter::once(0x1014..0x1024))
        );
        assert!(GlobalWrites::compile_uniform_stores(&[branch, base, store, exit]).is_none());
        assert!(
            GlobalWrites::compile_uniform_stores(&[base, branch, vote(4), store, exit]).is_none()
        );
    }

    fn sample() -> Instruction {
        op(Op::Tex(Texture {
            op: TextureOp::Sample,
            d: Reg(8),
            d2: Reg::ZERO,
            a: Reg(5),
            b: Reg::ZERO,
            bank: 0,
            handle_offset: 0,
            handle: Some(UniformReg(2)),
            handle_value: false,
            packed: true,
            dimension: 1,
            lod: 0,
            offsets: false,
            per_tap: false,
            depth_compare: false,
            mask: 15,
            half: false,
        }))
    }

    fn program() -> [Instruction; 6] {
        [
            Instruction {
                uniform: true,
                ..op(Op::Uldc {
                    d: UniformReg(4),
                    src: Src::Const {
                        bank: 0,
                        offset: 0x110,
                    },
                    size: 5,
                })
            },
            op(Op::Mov {
                d: Reg(0),
                src: Src::Const {
                    bank: 3,
                    offset: 16,
                },
                lanes: 15,
            }),
            op(Op::Lea {
                d: Reg(0),
                a: Reg(0),
                b: Operand {
                    src: Src::Imm(8),
                    abs: false,
                    neg: false,
                },
                c: Operand {
                    src: Src::Reg(Reg::ZERO),
                    abs: false,
                    neg: false,
                },
                shift: 4,
                high: false,
                negate_shifted: false,
                extended: false,
                overflow: 7,
                carry_in: Pred {
                    index: 7,
                    negated: true,
                },
            }),
            op(Op::Load {
                d: Reg(10),
                space: Space::Global,
                address: Address {
                    reg: Reg(0),
                    wide: false,
                    scale: 1,
                    uniform: Some((UniformReg(4), true)),
                    offset: -8,
                },
                size: 5,
                condition: Pred::TRUE,
                strong: false,
            }),
            op(Op::R2ur {
                d: UniformReg(2),
                src: Reg(11),
            }),
            sample(),
        ]
    }

    #[test]
    fn bindless_handle_reads_the_selected_word_of_uniform_storage() {
        let plan = HandleReads::compile(&program());
        let mut handles = Vec::new();
        let mut loads = Vec::new();
        let mut evaluate = |material, handle| {
            plan.evaluate(
                |input| match input {
                    Input::Constant {
                        bank: 0,
                        offset: 0x110,
                    } => Some(0x1000),
                    Input::Constant {
                        bank: 0,
                        offset: 0x114,
                    } => Some(2),
                    Input::Constant {
                        bank: 3,
                        offset: 16,
                    } => Some(material),
                    Input::GlobalWord {
                        address,
                        word,
                        words,
                    } => {
                        loads.push((address, word, words));
                        handle
                    }
                    _ => None,
                },
                |handle| handles.push(handle),
            );
        };
        evaluate(7, Some(0x3210_0004u32));
        evaluate(8, Some(0x3210_0009u32));
        evaluate(9, None);
        assert_eq!(handles, [0x3210_0004, 0x3210_0009]);
        assert_eq!(
            loads,
            [
                (0x2_0000_1070, 1, 2),
                (0x2_0000_1080, 1, 2),
                (0x2_0000_1090, 1, 2)
            ]
        );
    }

    #[test]
    fn ignores_handles_derived_from_invocations_or_written_storage() {
        let mut code = program();
        code[1] = op(Op::Ipa {
            d: Reg(0),
            address: 0,
            location: 0,
            mode: 0,
            offset: Reg::ZERO,
            index: None,
        });
        assert!(HandleReads::compile(&code).is_empty());

        let mut code = program();
        code[4] = op(Op::Store {
            data: Reg(4),
            space: Space::Global,
            address: Address {
                reg: Reg(0),
                wide: false,
                scale: 1,
                uniform: None,
                offset: 0,
            },
            size: 5,
            strong: false,
        });
        assert!(HandleReads::compile(&code).is_empty());
    }
}

/// All bytes a compute program can write through pointers derived from
/// uniform constant-bank loads, independent of control flow.
#[derive(Debug)]
pub struct UniformWriteRanges(Vec<UniformPointer>);

#[derive(Debug)]
struct UniformPointer {
    offsets: std::ops::RangeInclusive<i64>,
    constant: (u8, u16),
    low_unknown: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct UniformAddress {
    constant: (u8, u16),
    /// Inclusive upper bound for the unsigned additions to the base.
    max_offset: i64,
}

/// A carry from adding an unsigned word to the low word of a pointer.
#[derive(Clone, Copy, PartialEq, Eq)]
struct PointerCarry {
    base: UniformAddress,
    high_input: u8,
    low_result: u8,
}

#[derive(Clone, Copy)]
struct UniformDefinitions {
    pointers: [Option<UniformAddress>; 64],
    /// Pointer high words can survive an unrelated change to the low word
    /// or be copied into a register pair with a different low word.
    high_words: [Option<UniformAddress>; 64],
    carries: [Option<PointerCarry>; 7],
}

impl UniformDefinitions {
    const UNKNOWN: Self = Self {
        pointers: [None; 64],
        high_words: [None; 64],
        carries: [None; 7],
    };

    fn overwrite(&mut self, first: u8, last: u8) {
        let last = last.min(62);
        if first <= last {
            self.pointers[usize::from(first.saturating_sub(1))..=usize::from(last)].fill(None);
            self.high_words[usize::from(first)..=usize::from(last)].fill(None);
            let words = first..=last;
            for carry in &mut self.carries {
                if carry.is_some_and(|carry| {
                    words.contains(&carry.low_result) || words.contains(&carry.high_input)
                }) {
                    *carry = None;
                }
            }
        }
    }

    fn set_carry(&mut self, predicate: u8, carry: Option<PointerCarry>) {
        if let Some(output) = self.carries.get_mut(usize::from(predicate)) {
            *output = carry;
        }
    }

    fn pointer_word(operand: Operand) -> Option<u8> {
        match operand {
            Operand {
                src: Src::Reg(Reg(reg)),
                abs: false,
                neg: false,
            } if reg < 63 => Some(reg),
            _ => None,
        }
    }

    fn zero(operand: Operand) -> bool {
        !operand.neg && matches!(operand.src, Src::Reg(Reg(63)) | Src::Imm(0))
    }

    fn low_sum(&self, d: Reg, base: Operand) -> Option<PointerCarry> {
        if d.0 >= 62 {
            return None;
        }
        let pointer = Self::pointer_word(base)?;
        let base = self.pointers[usize::from(pointer)]?;
        Some(PointerCarry {
            base,
            high_input: pointer + 1,
            low_result: d.0,
        })
    }

    fn high_sum(
        &self,
        d: Reg,
        a: Operand,
        b: Operand,
        c: Operand,
        carry_in: [Pred; 2],
    ) -> Option<(usize, UniformAddress)> {
        if carry_in[0].negated
            || carry_in[1]
                != (Pred {
                    index: 7,
                    negated: true,
                })
            || !Self::zero(c)
        {
            return None;
        }
        let carry = self
            .carries
            .get(usize::from(carry_in[0].index))
            .copied()
            .flatten()?;
        if d.0 != carry.low_result + 1
            || ![(a, b), (b, a)].into_iter().any(|(high, zero)| {
                Self::pointer_word(high) == Some(carry.high_input) && Self::zero(zero)
            })
        {
            return None;
        }
        let mut address = carry.base;
        address.max_offset = address.max_offset.checked_add(i64::from(u32::MAX))?;
        Some((usize::from(carry.low_result), address))
    }

    fn advance(&mut self, instruction: &Instruction) {
        if instruction.guard.index == 7 && instruction.guard.negated {
            return;
        }
        let unconditional = instruction.guard.is_true();
        match instruction.op {
            Op::Uldc { d, src, size } => {
                // Sizes 0-4 load one word, 5 two and 6 four; any other is
                // taken as four, so no stale fact survives in d+1..d+3.
                let words: u8 = match size {
                    0..=4 => 1,
                    5 => 2,
                    _ => 4,
                };
                self.overwrite(d.0, d.0.saturating_add(words - 1));
                if unconditional
                    && size == 5
                    && d.0 < 62
                    && let Src::Const { bank, offset } = src
                {
                    let address = UniformAddress {
                        constant: (bank, offset & !3),
                        max_offset: 0,
                    };
                    self.pointers[usize::from(d.0)] = Some(address);
                    self.high_words[usize::from(d.0 + 1)] = Some(address);
                }
            }
            Op::Iadd3 {
                d,
                a,
                b,
                c,
                carry_out,
                carry_in,
                extended,
            } if instruction.uniform => {
                let completed = (unconditional && extended)
                    .then(|| self.high_sum(d, a, b, c, carry_in))
                    .flatten();
                let carry = (unconditional
                    && !extended
                    && !a.neg
                    && !b.neg
                    && !carry_out[0].negated
                    && Self::zero(c))
                .then(|| self.low_sum(d, a).or_else(|| self.low_sum(d, b)))
                .flatten();
                self.overwrite(d.0, d.0);
                self.set_carry(carry_out[0].index, carry);
                self.set_carry(carry_out[1].index, None);
                if let Some((low, address)) = completed {
                    self.pointers[low] = Some(address);
                    self.high_words[low + 1] = Some(address);
                }
            }
            Op::Lea {
                d,
                b,
                high,
                negate_shifted,
                extended,
                overflow,
                ..
            } if instruction.uniform => {
                // LEA's shifted term is a u32, so its addition has the
                // same carry and bound as an unsigned low-word sum.
                let carry = (unconditional && !high && !negate_shifted && !extended)
                    .then(|| self.low_sum(d, b))
                    .flatten();
                self.overwrite(d.0, d.0);
                self.set_carry(overflow, carry);
            }
            Op::Isetp { p, q, .. } if instruction.uniform => {
                self.set_carry(p, None);
                self.set_carry(q, None);
            }
            Op::Mov { d, src, lanes } if instruction.uniform => {
                let source = match src {
                    Src::Uniform(UniformReg(word)) | Src::Reg(Reg(word)) if word < 63 => {
                        self.high_words[usize::from(word)]
                    }
                    _ => None,
                };
                self.overwrite(d.0, d.0);
                if unconditional && lanes == 15 && d.0 < 63 {
                    self.high_words[usize::from(d.0)] = source;
                }
            }
            Op::Imad { d, .. } | Op::Shf { d, .. } if instruction.uniform => {
                self.overwrite(d.0, d.0);
            }
            Op::ImadWide { d, carry_out, .. } if instruction.uniform => {
                self.overwrite(d.0, d.0.saturating_add(1));
                self.set_carry(carry_out, None);
            }
            Op::Bra { .. } | Op::Call { .. } | Op::Exit { .. } => {}
            Op::R2ur { d, .. } | Op::Redux { d, .. } => self.overwrite(d.0, d.0),
            Op::Vote {
                d,
                p,
                uniform_out: true,
                ..
            } => {
                self.overwrite(d.0, d.0);
                self.set_carry(p, None);
            }
            _ if instruction.uniform => *self = Self::UNKNOWN,
            _ => {}
        }
    }

    fn intersect(&mut self, other: Self) -> bool {
        fn keep_matching<T: PartialEq>(
            current: &mut [Option<T>],
            other: impl IntoIterator<Item = Option<T>>,
        ) -> bool {
            let mut changed = false;
            for (old, new) in current.iter_mut().zip(other) {
                if old.is_some() && *old != new {
                    *old = None;
                    changed = true;
                }
            }
            changed
        }
        keep_matching(&mut self.pointers, other.pointers)
            | keep_matching(&mut self.high_words, other.high_words)
            | keep_matching(&mut self.carries, other.carries)
    }
}

impl UniformWriteRanges {
    pub fn from_program(program: &[u8]) -> Option<Self> {
        if crate::is_graphics(program) {
            return None;
        }
        let code = crate::program_code(program).ok()?;
        Self::compile_reaching(&code)
    }

    /// Constant-bank pointer definitions that reach every path to a write.
    /// Any incoming path with a different or unknown definition invalidates
    /// the pair until a later instruction establishes a new definition.
    fn compile_reaching(code: &[Instruction]) -> Option<Self> {
        let mut successors = Vec::with_capacity(code.len());
        let mut writes = Vec::new();
        for (index, instruction) in code.iter().enumerate() {
            let next = (index + 1 < code.len()).then_some(index + 1);
            successors.push(match instruction.op {
                Op::Bra {
                    mode: 0,
                    offset,
                    condition,
                } => {
                    let target = crate::compute::relative_target(index, offset, code.len())?;
                    let fallthrough = if instruction.guard.is_true() && condition.is_true() {
                        None
                    } else {
                        next
                    };
                    [Some(target), fallthrough]
                }
                Op::Exit { condition } if condition.is_true() && instruction.guard.is_true() => {
                    [None, None]
                }
                Op::Call { offset } => {
                    let target = crate::compute::relative_target(index, offset, code.len())?;
                    [
                        Some(target),
                        if instruction.guard.is_true() {
                            None
                        } else {
                            next
                        },
                    ]
                }
                Op::Bra { .. } | Op::Brx { .. } | Op::Ret { .. } | Op::Unknown { .. } => {
                    return None;
                }
                Op::Store {
                    space: Space::Global,
                    address,
                    ..
                }
                | Op::Atomic {
                    space: Space::Global,
                    address,
                    ..
                } => {
                    writes.push((index, address));
                    [next, None]
                }
                _ => [next, None],
            });
        }
        if writes.is_empty() {
            return None;
        }

        let mut inputs = vec![None; code.len()];
        inputs[0] = Some(UniformDefinitions::UNKNOWN);
        let mut pending = std::collections::VecDeque::from([0]);
        let mut queued = vec![false; code.len()];
        while let Some(at) = pending.pop_front() {
            queued[at] = false;
            let mut output = inputs[at]?;
            output.advance(&code[at]);
            for next in successors[at].into_iter().flatten() {
                let changed = if let Some(input) = &mut inputs[next] {
                    input.intersect(output)
                } else {
                    inputs[next] = Some(output);
                    true
                };
                if changed && !queued[next] {
                    pending.push_back(next);
                    queued[next] = true;
                }
            }
        }

        let mut ranges = std::collections::BTreeMap::new();
        for (index, address) in writes {
            let Some(state) = &inputs[index] else {
                continue;
            };
            let (pointer, true) = address.uniform? else {
                return None;
            };
            let dynamic = address.reg != Reg::ZERO;
            if address.wide && dynamic {
                return None;
            }
            let word = usize::from(pointer.0);
            let (source, low_unknown) = match state.pointers[word] {
                Some(source) => (source, false),
                None => (state.high_words.get(word + 1).copied().flatten()?, true),
            };
            let first = i64::from(address.offset);
            let last = first
                .checked_add(source.max_offset)?
                .checked_add(i64::from(u32::MAX) * (i64::from(dynamic) + i64::from(low_unknown)))?;
            ranges
                .entry((source.constant, low_unknown))
                .and_modify(|(lo, hi): &mut (i64, i64)| {
                    *lo = (*lo).min(first);
                    *hi = (*hi).max(last);
                })
                .or_insert((first, last));
        }
        (!ranges.is_empty()).then(|| {
            Self(
                ranges
                    .into_iter()
                    .map(|((source, low_unknown), (lo, hi))| UniformPointer {
                        offsets: lo..=hi,
                        constant: source,
                        low_unknown,
                    })
                    .collect(),
            )
        })
    }

    pub fn intervals(
        &self,
        mut constant: impl FnMut(u8, u16) -> Option<u32>,
    ) -> Option<Vec<std::ops::Range<u64>>> {
        let mut ranges = Vec::new();
        for pointer in &self.0 {
            let (bank, offset) = pointer.constant;
            let base = u64::from(constant(bank, offset)?)
                | u64::from(constant(bank, offset.wrapping_add(4))?) << 32;
            // Subword stores can update a whole aligned word. One translated
            // instruction writes at most sixteen bytes.
            // With only the high word known, any low word can appear.
            let first_base = if pointer.low_unknown {
                base & !u64::from(u32::MAX)
            } else {
                base
            };
            let first = first_base.checked_add_signed(*pointer.offsets.start())? & !3;
            let end = base
                .checked_add_signed(*pointer.offsets.end())?
                .checked_add(16)?;
            ranges.push(first..end);
        }
        Some(ranges)
    }
}

#[cfg(test)]
mod uniform_range_tests {
    use super::*;

    fn ins(op: Op) -> Instruction {
        Instruction {
            raw: 0,
            guard: Pred::TRUE,
            uniform: false,
            op,
        }
    }

    fn load(d: u8, offset: u16, size: u8) -> Instruction {
        let mut load = ins(Op::Uldc {
            d: UniformReg(d),
            src: Src::Const { bank: 0, offset },
            size,
        });
        load.guard = Pred {
            index: 0,
            negated: true,
        };
        load.uniform = true;
        load
    }

    fn store(reg: Reg, wide: bool, offset: i32) -> Instruction {
        ins(Op::Store {
            address: Address {
                reg,
                wide,
                offset,
                scale: 1,
                uniform: Some((UniformReg(4), true)),
            },
            data: Reg(3),
            space: Space::Global,
            size: 6,
            strong: false,
        })
    }

    #[test]
    fn complete_pair_definitions_after_partial_writes_bound_straight_line_stores() {
        let mut pair = load(4, 32, 5);
        pair.guard = Pred::TRUE;
        let partial = load(5, 64, 4);
        let write = store(Reg(2), false, 0);
        let plan = UniformWriteRanges::compile_reaching(&[partial, pair, write]).unwrap();
        let constants = |_, at| match at {
            32 => Some(0x100),
            36 => Some(8),
            _ => None,
        };
        assert_eq!(
            plan.intervals(constants).unwrap()[0],
            0x8_0000_0100..0x9_0000_010f
        );
        assert!(UniformWriteRanges::compile_reaching(&[pair, partial, write]).is_none());
    }

    #[test]
    fn uniform_pointer_additions_follow_their_carry_through_control_flow() {
        let reg = |n| Operand {
            src: Src::Reg(Reg(n)),
            abs: false,
            neg: false,
        };
        let uniform = |op| Instruction {
            uniform: true,
            ..ins(op)
        };
        let never = Pred {
            index: 7,
            negated: true,
        };
        let carry = Pred {
            index: 2,
            negated: false,
        };
        let call = |offset| Instruction {
            guard: carry,
            ..ins(Op::Call { offset })
        };
        let exit = ins(Op::Exit {
            condition: Pred::TRUE,
        });
        let write = store(Reg(2), false, 0);
        for base_register in [4, 8] {
            let mut base = load(base_register, 32, 5);
            base.guard = Pred::TRUE;
            let low = uniform(Op::Iadd3 {
                d: Reg(4),
                a: reg(7),
                b: reg(base_register),
                c: reg(63),
                carry_out: [carry, Pred::TRUE],
                carry_in: [never; 2],
                extended: false,
            });
            let shifted_low = uniform(Op::Lea {
                d: Reg(4),
                a: Reg(7),
                b: reg(base_register),
                c: reg(63),
                shift: 1,
                high: false,
                negate_shifted: false,
                extended: false,
                overflow: 2,
                carry_in: never,
            });
            let high = uniform(Op::Iadd3 {
                d: Reg(5),
                a: reg(63),
                b: reg(base_register + 1),
                c: reg(63),
                carry_out: [Pred::TRUE; 2],
                carry_in: [carry, never],
                extended: true,
            });
            for low in [low, shifted_low] {
                // The call goes straight to the exit on lanes that take it.
                let plan =
                    UniformWriteRanges::compile_reaching(&[base, low, call(32), high, write, exit])
                        .unwrap();
                let ranges = plan
                    .intervals(|_, word| match word {
                        32 => Some(0x100),
                        36 => Some(8),
                        _ => None,
                    })
                    .unwrap();
                assert_eq!(ranges[0], 0x8_0000_0100..0xa_0000_010e);
                // A path entering the store without its high word has no bound.
                assert!(
                    UniformWriteRanges::compile_reaching(&[base, low, call(16), high, write, exit])
                        .is_none()
                );
                // Even a conditional overwrite of the base invalidates the carry.
                assert!(
                    UniformWriteRanges::compile_reaching(&[
                        base,
                        low,
                        load(base_register + 1, 64, 4),
                        high,
                        write
                    ])
                    .is_none()
                );
            }
        }
    }

    #[test]
    fn a_forwarded_high_word_bounds_unknown_low_words() {
        let mut base = load(8, 32, 5);
        base.guard = Pred::TRUE;
        let copy_high = Instruction {
            uniform: true,
            ..ins(Op::Mov {
                d: Reg(5),
                src: Src::Uniform(UniformReg(9)),
                lanes: 15,
            })
        };
        let write = store(Reg::ZERO, false, -4);
        let plan = UniformWriteRanges::compile_reaching(&[base, copy_high, write]).unwrap();
        let ranges = plan
            .intervals(|_, word| match word {
                32 => Some(0x100),
                36 => Some(8),
                _ => None,
            })
            .unwrap();
        assert_eq!(ranges[0], 0x7_ffff_fffc..0x9_0000_010b);
        assert!(
            UniformWriteRanges::compile_reaching(&[base, load(9, 64, 4), copy_high, write])
                .is_none()
        );
    }

    fn branch(offset: i64, unconditional: bool) -> Instruction {
        let mut branch = ins(Op::Bra {
            offset,
            mode: 0,
            condition: Pred::TRUE,
        });
        if !unconditional {
            branch.guard = Pred {
                index: 0,
                negated: false,
            };
        }
        branch
    }

    #[test]
    fn reaching_definitions_survive_only_when_every_path_agrees() {
        let pair = |offset| {
            let mut pair = load(4, offset, 5);
            pair.guard = Pred::TRUE;
            pair
        };
        let write = store(Reg(2), false, 0);
        let mut code = [
            branch(32, false),
            pair(32),
            branch(16, true),
            pair(32),
            write,
        ];
        let matching = UniformWriteRanges::compile_reaching(&code).unwrap();
        assert_eq!(matching.intervals(|_, _| Some(0)).unwrap().len(), 1);
        code[3] = pair(64);
        assert!(UniformWriteRanges::compile_reaching(&code).is_none());
    }

    #[test]
    fn back_edges_can_invalidate_a_definition_seen_on_the_first_iteration() {
        let mut pair = load(4, 32, 5);
        pair.guard = Pred::TRUE;
        let partial = load(5, 64, 4);
        let write = store(Reg(2), false, 0);
        assert!(
            UniformWriteRanges::compile_reaching(&[pair, write, partial, branch(-48, false)])
                .is_none()
        );
        assert!(
            UniformWriteRanges::compile_reaching(&[pair, write, partial, branch(-64, false)])
                .is_some()
        );
    }

    #[test]
    fn branch_around_a_load_cannot_define_the_pointer() {
        let mut pair = load(4, 32, 5);
        pair.guard = Pred::TRUE;
        let write = store(Reg(2), false, 0);
        assert!(UniformWriteRanges::compile_reaching(&[branch(16, false), pair, write]).is_none());
        assert!(
            UniformWriteRanges::compile_reaching(&[pair, branch(16, true), load(5, 64, 4), write])
                .is_some()
        );
    }

    #[test]
    fn a_conditional_exit_keeps_the_fallthrough_write_reachable() {
        let exit = ins(Op::Exit {
            condition: Pred {
                index: 0,
                negated: false,
            },
        });
        let mut pair = load(4, 32, 5);
        pair.guard = Pred::TRUE;
        let write = store(Reg::ZERO, false, 4);
        let plan = UniformWriteRanges::compile_reaching(&[exit, pair, write]).unwrap();
        let ranges = plan.intervals(|_, _| Some(0)).unwrap();
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0], 4..20);
        let exit = ins(Op::Exit {
            condition: Pred::TRUE,
        });
        assert!(UniformWriteRanges::compile_reaching(&[exit, pair, write]).is_none());
    }

    #[test]
    fn uniform_add_only_invalidates_the_word_it_overwrites() {
        let mut pair = load(4, 32, 5);
        pair.guard = Pred::TRUE;
        let value = Operand {
            src: Src::Imm(1),
            abs: false,
            neg: false,
        };
        let add = |d| {
            let mut instruction = ins(Op::Iadd3 {
                d: Reg(d),
                a: value,
                b: value,
                c: value,
                carry_out: [Pred::TRUE; 2],
                carry_in: [Pred::TRUE; 2],
                extended: false,
            });
            instruction.uniform = true;
            instruction
        };
        let write = store(Reg(2), false, 0);
        for d in [0, 3, 6, 62, 63] {
            let plan = UniformWriteRanges::compile_reaching(&[pair, add(d), write]).unwrap();
            assert!(!plan.0[0].low_unknown);
        }
        let plan = UniformWriteRanges::compile_reaching(&[pair, add(4), write]).unwrap();
        assert!(plan.0[0].low_unknown);
        assert!(UniformWriteRanges::compile_reaching(&[pair, add(5), write]).is_none());
    }

    #[test]
    fn declines_wrapping_addresses() {
        let mut pair = load(4, 32, 5);
        pair.guard = Pred::TRUE;
        let plan = UniformWriteRanges::compile_reaching(&[pair, store(Reg(2), false, 8)]).unwrap();
        assert!(plan.intervals(|_, _| Some(u32::MAX)).is_none());
        let plan =
            UniformWriteRanges::compile_reaching(&[pair, store(Reg::ZERO, true, -8)]).unwrap();
        assert!(plan.intervals(|_, _| Some(0)).is_none());
    }
}
