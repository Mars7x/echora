//! Translation of a decoded program to a SPIR-V compute shader.
//!
//! The program's general registers and predicates are function variables
//! (the driver's compiler promotes them to SSA values); RZ reads zero and
//! drops writes, PT is true. Control flow runs through one structured
//! dispatch loop: a `switch` on the index of the block to run next, so any
//! control-flow graph is valid SPIR-V, irreducible ones included. Every
//! iteration spends one unit of the invocation's loop budget, so no
//! program can hang the GPU: when the budget runs out the invocation
//! stops. Constant bank `n` is a uniform buffer (64 KiB of `uvec4`),
//! reported as a descriptor.

use std::collections::{BTreeMap, BTreeSet};

use echora_shader_abi::{ABI_VERSION, Descriptor, Stage, Translated, Unsupported};

use crate::decode::{
    Combine, FloatCompare, FloatModes, Instruction, IntCompare, Op, Operand, Pred, Src,
};
use crate::spirv::{
    self, Function, Module, Type, capability, decoration, execution, glsl, op, storage,
};

/// Block transitions an invocation may make before it is stopped: far
/// beyond any finite loop a game runs per invocation.
pub const LOOP_BUDGET: u32 = 1 << 20;

/// How a block ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum End {
    /// Into the next block.
    Fall,
    /// To block `target` (an instruction index) when `condition` holds,
    /// else into the next block.
    Branch { target: usize, condition: Guard },
    /// The invocation ends when `condition` holds, else falls through.
    Exit { condition: Guard },
}

/// An instruction's guard and a branch condition together.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Guard {
    guard: Pred,
    extra: Pred,
}

impl Guard {
    fn always(self) -> bool {
        self.guard.is_true() && self.extra.is_true()
    }
}

/// A basic block: instructions `start..end` (exclusive of the terminator,
/// which is the last instruction when it branches or exits).
#[derive(Clone, Debug)]
struct Block {
    start: usize,
    body_end: usize,
    end: End,
}

fn unsupported(index: usize, reason: impl Into<String>) -> Unsupported {
    Unsupported {
        reason: reason.into(),
        offset: Some(u32::try_from(index * 16).unwrap_or(u32::MAX)),
    }
}

/// Splits `code` into blocks.
fn blocks(code: &[Instruction]) -> Result<Vec<Block>, Unsupported> {
    let mut leaders = BTreeSet::from([0]);
    for (index, instruction) in code.iter().enumerate() {
        match instruction.op {
            Op::Bra { offset, mode, .. } => {
                if mode != 0 {
                    return Err(unsupported(index, "branch with a convergence mode"));
                }
                let target = i64::try_from(index + 1)
                    .ok()
                    .and_then(|next| next.checked_mul(16)?.checked_add(offset))
                    .filter(|target| *target >= 0 && target % 16 == 0)
                    .map(|target| (target / 16) as usize)
                    .filter(|target| *target < code.len())
                    .ok_or_else(|| unsupported(index, "branch outside the program"))?;
                leaders.insert(target);
                leaders.insert(index + 1);
            }
            Op::Exit => {
                leaders.insert(index + 1);
            }
            _ => {}
        }
    }
    leaders.retain(|&leader| leader < code.len());
    let starts: Vec<usize> = leaders.into_iter().collect();
    let mut blocks = Vec::new();
    for (position, &start) in starts.iter().enumerate() {
        let next = starts.get(position + 1).copied().unwrap_or(code.len());
        let last = &code[next - 1];
        let (body_end, end) = match last.op {
            Op::Bra {
                offset, condition, ..
            } => {
                let target = ((next as i64) * 16 + offset) as usize / 16;
                (
                    next - 1,
                    End::Branch {
                        target,
                        condition: Guard {
                            guard: last.guard,
                            extra: condition,
                        },
                    },
                )
            }
            Op::Exit => (
                next - 1,
                End::Exit {
                    condition: Guard {
                        guard: last.guard,
                        extra: Pred::TRUE,
                    },
                },
            ),
            _ => (next, End::Fall),
        };
        blocks.push(Block {
            start,
            body_end,
            end,
        });
    }
    Ok(blocks)
}

/// Translates a compute program (its instructions from the first one) with
/// workgroup size `workgroup`.
pub fn translate(code: &[Instruction], workgroup: [u32; 3]) -> Result<Translated, Unsupported> {
    if code.is_empty() {
        return Err(unsupported(0, "empty program"));
    }
    let blocks = blocks(code)?;
    let mut emitter = Emitter::new(workgroup);
    emitter.program(code, &blocks)?;
    Ok(emitter.finish())
}

/// Bytes of a constant bank.
const BANK_SIZE: u32 = 0x10000;

/// SPIR-V types and the module-level objects the program uses.
struct Emitter {
    module: Module,
    function: Function,
    entry: u32,
    void: u32,
    bool: u32,
    u32: u32,
    f32: u32,
    /// Function-variable pointer types.
    u32_ptr: u32,
    bool_ptr: u32,
    /// Variables of general registers and predicates, by number.
    registers: BTreeMap<u8, u32>,
    predicates: BTreeMap<u8, u32>,
    /// Uniform buffer of each constant bank used, by bank.
    banks: BTreeMap<u8, u32>,
    /// The block type every bank's buffer shares, once declared.
    bank_block: Option<u32>,
    /// Built-in input variables (local invocation id, workgroup id).
    builtins: BTreeMap<u32, u32>,
    interface: Vec<u32>,
    descriptors: Vec<Descriptor>,
    cbuf_used_sizes: [u32; 18],
}

impl Emitter {
    fn new(workgroup: [u32; 3]) -> Self {
        let mut module = Module::new();
        module.capability(capability::SHADER);
        module.memory_model(0, 1); // Logical, GLSL450
        let void = module.void();
        let bool = module.bool();
        let u32 = module.u32();
        let f32 = module.f32();
        let u32_ptr = module.pointer(storage::FUNCTION, u32);
        let bool_ptr = module.pointer(storage::FUNCTION, bool);
        let function_type = module.ty(Type::Function {
            result: void,
            parameters: vec![],
        });
        let function = Function::new(&mut module, void, function_type);
        let entry = function.id;
        module.execution_mode(entry, execution::LOCAL_SIZE, &workgroup);
        Self {
            module,
            function,
            entry,
            void,
            bool,
            u32,
            f32,
            u32_ptr,
            bool_ptr,
            registers: BTreeMap::new(),
            predicates: BTreeMap::new(),
            banks: BTreeMap::new(),
            bank_block: None,
            builtins: BTreeMap::new(),
            interface: Vec::new(),
            descriptors: Vec::new(),
            cbuf_used_sizes: [0; 18],
        }
    }

    fn finish(mut self) -> Translated {
        let _ = self.void;
        self.function.finish(&mut self.module);
        self.module
            .entry_point(execution::GL_COMPUTE, self.entry, "main", &self.interface);
        Translated {
            abi_version: ABI_VERSION,
            stage: Stage::Compute,
            spirv: self.module.finish(),
            descriptors: self.descriptors,
            loaded_generics: 0,
            stored_generics: 0,
            stores: [0; 8],
            frag_colors: 0,
            cbuf_used_sizes: self.cbuf_used_sizes,
            spec_constants: Vec::new(),
            push_fields: 0,
        }
    }

    fn value(&mut self, opcode: u16, ty: u32, operands: &[u32]) -> u32 {
        self.function.value(&mut self.module, opcode, ty, operands)
    }

    fn constant(&mut self, value: u32) -> u32 {
        self.module.u32_constant(value)
    }

    /// The variable of general register `n` (declared, zero, on first use).
    fn register(&mut self, n: u8) -> u32 {
        if let Some(&variable) = self.registers.get(&n) {
            return variable;
        }
        let zero = self.constant(0);
        let variable = self
            .function
            .variable(&mut self.module, self.u32_ptr, Some(zero));
        self.registers.insert(n, variable);
        variable
    }

    fn predicate_variable(&mut self, n: u8) -> u32 {
        if let Some(&variable) = self.predicates.get(&n) {
            return variable;
        }
        let no = self.module.bool_constant(false);
        let variable = self
            .function
            .variable(&mut self.module, self.bool_ptr, Some(no));
        self.predicates.insert(n, variable);
        variable
    }

    fn read_register(&mut self, n: u8) -> u32 {
        if n == 255 {
            return self.constant(0);
        }
        let variable = self.register(n);
        self.value(op::LOAD, self.u32, &[variable])
    }

    fn write_register(&mut self, n: u8, value: u32) {
        if n != 255 {
            let variable = self.register(n);
            self.function.emit(op::STORE, &[variable, value]);
        }
    }

    fn read_predicate(&mut self, pred: Pred) -> u32 {
        let value = if pred.index == 7 {
            self.module.bool_constant(true)
        } else {
            let variable = self.predicate_variable(pred.index);
            self.value(op::LOAD, self.bool, &[variable])
        };
        if pred.negated {
            self.value(op::LOGICAL_NOT, self.bool, &[value])
        } else {
            value
        }
    }

    fn write_predicate(&mut self, n: u8, value: u32) {
        if n != 7 {
            let variable = self.predicate_variable(n);
            self.function.emit(op::STORE, &[variable, value]);
        }
    }

    /// The uniform buffer of constant bank `bank`: a 64 KiB block of
    /// `uvec4` (std140), at the next set-0 binding, declared and reported
    /// as a descriptor on first use.
    fn bank(&mut self, bank: u8) -> Result<u32, String> {
        if let Some(&variable) = self.banks.get(&bank) {
            return Ok(variable);
        }
        if usize::from(bank) >= self.cbuf_used_sizes.len() {
            return Err(format!("constant bank {bank}"));
        }
        let block = match self.bank_block {
            Some(block) => block,
            None => {
                let uvec4 = self.module.ty(Type::Vector {
                    component: self.u32,
                    count: 4,
                });
                let length = self.constant(BANK_SIZE / 16);
                let array = self.module.ty(Type::Array {
                    element: uvec4,
                    length,
                });
                self.module.decorate(array, decoration::ARRAY_STRIDE, &[16]);
                let block = self.module.ty(Type::Struct {
                    members: vec![array],
                });
                self.module.decorate(block, decoration::BLOCK, &[]);
                self.module
                    .member_decorate(block, 0, decoration::OFFSET, &[0]);
                self.bank_block = Some(block);
                block
            }
        };
        let pointer = self.module.pointer(storage::UNIFORM, block);
        let variable = self.module.global_variable(pointer, storage::UNIFORM);
        let binding = self.descriptors.len() as u32;
        self.module
            .decorate(variable, decoration::DESCRIPTOR_SET, &[0]);
        self.module
            .decorate(variable, decoration::BINDING, &[binding]);
        self.descriptors.push(Descriptor {
            kind: 0,
            binding,
            count: 1,
            index: u32::from(bank),
            ..Descriptor::default()
        });
        self.interface.push(variable);
        self.banks.insert(bank, variable);
        Ok(variable)
    }

    /// The word at byte `offset` of constant bank `bank`.
    fn constant_word(&mut self, bank: u8, offset: u16) -> Result<u32, String> {
        if !offset.is_multiple_of(4) {
            return Err(format!("unaligned constant c[{bank}][{offset:#x}]"));
        }
        let variable = self.bank(bank)?;
        let used = &mut self.cbuf_used_sizes[usize::from(bank)];
        *used = (*used).max(u32::from(offset) + 4);
        let pointer = self.module.pointer(storage::UNIFORM, self.u32);
        let member = self.constant(0);
        let vector = self.constant(u32::from(offset) / 16);
        let component = self.constant(u32::from(offset) / 4 % 4);
        let element = self.value(
            op::ACCESS_CHAIN,
            pointer,
            &[variable, member, vector, component],
        );
        Ok(self.value(op::LOAD, self.u32, &[element]))
    }

    /// A 32-bit source operand's value (before float modifiers).
    fn source(&mut self, index: usize, src: Src) -> Result<u32, Unsupported> {
        match src {
            Src::Reg(reg) => Ok(self.read_register(reg.0)),
            Src::Imm(value) => Ok(self.constant(value)),
            Src::Const { bank, offset } => self
                .constant_word(bank, offset)
                .map_err(|reason| unsupported(index, reason)),
            Src::Uniform(_) => Err(unsupported(index, "uniform register operand")),
            Src::BindlessConst { .. } => Err(unsupported(index, "bindless constant buffer")),
        }
    }

    /// An operand as an integer, ignoring float modifiers.
    fn integer(&mut self, index: usize, operand: Operand) -> Result<u32, Unsupported> {
        self.source(index, operand.src)
    }

    /// An operand as a float, with its absolute value and negation.
    fn float(&mut self, index: usize, operand: Operand) -> Result<u32, Unsupported> {
        let bits = self.source(index, operand.src)?;
        let mut value = self.value(op::BITCAST, self.f32, &[bits]);
        if operand.abs {
            let glsl = self.module.glsl();
            value = self.value(op::EXT_INST, self.f32, &[glsl, glsl::F_ABS, value]);
        }
        if operand.neg {
            value = self.value(op::F_NEGATE, self.f32, &[value]);
        }
        Ok(value)
    }

    /// A float result with the saturate mode, written to `d`. Rounding
    /// other than to nearest even is not translated.
    fn float_result(
        &mut self,
        index: usize,
        d: u8,
        value: u32,
        modes: FloatModes,
    ) -> Result<(), Unsupported> {
        if modes.rounding != 0 {
            return Err(unsupported(index, "float rounding mode"));
        }
        let mut value = value;
        if modes.saturate {
            let glsl = self.module.glsl();
            let zero = self.module.constant(self.f32, 0);
            let one = self.module.constant(self.f32, 1.0f32.to_bits());
            value = self.value(
                op::EXT_INST,
                self.f32,
                &[glsl, glsl::F_CLAMP, value, zero, one],
            );
        }
        let bits = self.value(op::BITCAST, self.u32, &[value]);
        self.write_register(d, bits);
        Ok(())
    }

    /// A built-in `uvec3` input's component `component`.
    fn builtin(&mut self, builtin: u32, component: u32) -> u32 {
        let variable = match self.builtins.get(&builtin) {
            Some(&variable) => variable,
            None => {
                let uvec3 = self.module.ty(Type::Vector {
                    component: self.u32,
                    count: 3,
                });
                let pointer = self.module.pointer(storage::INPUT, uvec3);
                let variable = self.module.global_variable(pointer, storage::INPUT);
                self.module
                    .decorate(variable, decoration::BUILT_IN, &[builtin]);
                self.builtins.insert(builtin, variable);
                self.interface.push(variable);
                variable
            }
        };
        let uvec3 = self.module.ty(Type::Vector {
            component: self.u32,
            count: 3,
        });
        let vector = self.value(op::LOAD, uvec3, &[variable]);
        self.value(op::COMPOSITE_EXTRACT, self.u32, &[vector, component])
    }

    /// Runs `body` only when `guard` holds (a selection construct), or
    /// unconditionally for PT.
    fn guarded(
        &mut self,
        guard: Pred,
        body: impl FnOnce(&mut Self) -> Result<(), Unsupported>,
    ) -> Result<(), Unsupported> {
        if guard.is_true() {
            return body(self);
        }
        let condition = self.read_predicate(guard);
        let then = self.module.id();
        let merge = self.module.id();
        self.function
            .emit(op::SELECTION_MERGE, &[merge, spirv::NONE]);
        self.function
            .emit(op::BRANCH_CONDITIONAL, &[condition, then, merge]);
        self.function.label(then);
        body(self)?;
        self.function.emit(op::BRANCH, &[merge]);
        self.function.label(merge);
        Ok(())
    }

    fn instruction(&mut self, index: usize, instruction: &Instruction) -> Result<(), Unsupported> {
        let op = instruction.op;
        self.guarded(instruction.guard, |e| e.operation(index, op))
    }

    fn operation(&mut self, index: usize, op: Op) -> Result<(), Unsupported> {
        match op {
            Op::Nop => {}
            Op::Mov { d, src } => {
                let value = self.source(index, src)?;
                self.write_register(d.0, value);
            }
            Op::Iadd3 {
                d,
                a,
                b,
                c,
                carry_out,
                carry_in,
            } => {
                if carry_out
                    .iter()
                    .chain(&carry_in)
                    .any(|pred| pred.index != 7)
                {
                    return Err(unsupported(index, "IADD3 with carries"));
                }
                let (a, b, c) = (
                    self.signed_integer(index, a)?,
                    self.signed_integer(index, b)?,
                    self.signed_integer(index, c)?,
                );
                let sum = self.value(op::I_ADD, self.u32, &[a, b]);
                let sum = self.value(op::I_ADD, self.u32, &[sum, c]);
                self.write_register(d.0, sum);
            }
            Op::Imad { d, a, b, c } => {
                let (a, b) = (self.integer(index, a)?, self.integer(index, b)?);
                let c = self.signed_integer(index, c)?;
                let product = self.value(op::I_MUL, self.u32, &[a, b]);
                let sum = self.value(op::I_ADD, self.u32, &[product, c]);
                self.write_register(d.0, sum);
            }
            Op::Lop3 {
                d, a, b, c, lut, p, ..
            } => {
                let a = self.read_register(a.0);
                let (b, c) = (self.integer(index, b)?, self.integer(index, c)?);
                let result = self.lut(lut, a, b, c);
                self.write_register(d.0, result);
                if p.index != 7 {
                    let zero = self.constant(0);
                    let nonzero = self.value(op::I_NOT_EQUAL, self.bool, &[result, zero]);
                    self.write_predicate(p.index, nonzero);
                }
            }
            Op::Ffma {
                d, a, b, c, modes, ..
            } => {
                let (a, b, c) = (
                    self.float(index, a)?,
                    self.float(index, b)?,
                    self.float(index, c)?,
                );
                let glsl = self.module.glsl();
                let value = self.value(op::EXT_INST, self.f32, &[glsl, glsl::FMA, a, b, c]);
                self.float_result(index, d.0, value, modes)?;
            }
            Op::Fmul {
                d,
                a,
                b,
                modes,
                scale,
                ..
            } => {
                if scale != 4 {
                    return Err(unsupported(index, "FMUL with a result scale"));
                }
                let (a, b) = (self.float(index, a)?, self.float(index, b)?);
                let value = self.value(op::F_MUL, self.f32, &[a, b]);
                self.float_result(index, d.0, value, modes)?;
            }
            Op::Fadd { d, a, b, modes } => {
                let (a, b) = (self.float(index, a)?, self.float(index, b)?);
                let value = self.value(op::F_ADD, self.f32, &[a, b]);
                self.float_result(index, d.0, value, modes)?;
            }
            Op::Fsetp {
                p,
                q,
                a,
                b,
                compare,
                combine,
                accumulate,
                ..
            } => {
                let (a, b) = (self.float(index, a)?, self.float(index, b)?);
                let result = self.float_compare(compare, a, b);
                self.set_predicates(p, q, result, combine, accumulate);
            }
            Op::Isetp {
                p,
                q,
                a,
                b,
                compare,
                signed,
                combine,
                accumulate,
                extended,
            } => {
                if extended.is_some() {
                    return Err(unsupported(index, "64-bit ISETP (.EX)"));
                }
                let (a, b) = (self.integer(index, a)?, self.integer(index, b)?);
                let result = self.int_compare(compare, signed, a, b);
                self.set_predicates(p, q, result, combine, accumulate);
            }
            Op::S2r { d, special } => {
                // Thread id (33-35) and workgroup id (37-39), as on Maxwell.
                let (builtin, component) = match special {
                    33..=35 => (27, u32::from(special - 33)), // LocalInvocationId
                    37..=39 => (26, u32::from(special - 37)), // WorkgroupId
                    _ => return Err(unsupported(index, format!("special register {special}"))),
                };
                let value = self.builtin(builtin, component);
                self.write_register(d.0, value);
            }
            Op::Bra { .. } | Op::Exit => {
                return Err(unsupported(index, "control flow inside a block"));
            }
            Op::Unknown { opcode } => {
                return Err(unsupported(index, format!("instruction {opcode:#05x}")));
            }
        }
        Ok(())
    }

    /// An integer operand with its negation (IADD3 and IMAD's addend use
    /// the negate bit as two's complement).
    fn signed_integer(&mut self, index: usize, operand: Operand) -> Result<u32, Unsupported> {
        let value = self.integer(index, operand)?;
        Ok(if operand.neg && !matches!(operand.src, Src::Imm(_)) {
            self.value(op::S_NEGATE, self.u32, &[value])
        } else {
            value
        })
    }

    /// LOP3's function of `a`, `b`, `c` by truth table: bit `4a + 2b + c`
    /// of `lut` is the result for those input bits. Summed over the
    /// table's set entries.
    fn lut(&mut self, lut: u8, a: u32, b: u32, c: u32) -> u32 {
        let not = |e: &mut Self, value: u32| e.value(op::NOT, e.u32, &[value]);
        let (na, nb, nc) = (not(self, a), not(self, b), not(self, c));
        let mut result = self.constant(0);
        for entry in 0..8u8 {
            if lut >> entry & 1 == 0 {
                continue;
            }
            let x = if entry & 4 != 0 { a } else { na };
            let y = if entry & 2 != 0 { b } else { nb };
            let z = if entry & 1 != 0 { c } else { nc };
            let term = self.value(op::BITWISE_AND, self.u32, &[x, y]);
            let term = self.value(op::BITWISE_AND, self.u32, &[term, z]);
            result = self.value(op::BITWISE_OR, self.u32, &[result, term]);
        }
        result
    }

    fn float_compare(&mut self, compare: FloatCompare, a: u32, b: u32) -> u32 {
        use FloatCompare::*;
        let opcode = match compare {
            False => return self.module.bool_constant(false),
            True => return self.module.bool_constant(true),
            Lt => op::F_ORD_LESS_THAN,
            Eq => op::F_ORD_EQUAL,
            Le => op::F_ORD_LESS_THAN_EQUAL,
            Gt => op::F_ORD_GREATER_THAN,
            Ne => op::F_ORD_NOT_EQUAL,
            Ge => op::F_ORD_GREATER_THAN_EQUAL,
            Ltu => op::F_UNORD_LESS_THAN,
            Equ => op::F_UNORD_EQUAL,
            Leu => op::F_UNORD_LESS_THAN_EQUAL,
            Gtu => op::F_UNORD_GREATER_THAN,
            Neu => op::F_UNORD_NOT_EQUAL,
            Geu => op::F_UNORD_GREATER_THAN_EQUAL,
            Num | Nan => {
                let a_nan = self.value(op::IS_NAN, self.bool, &[a]);
                let b_nan = self.value(op::IS_NAN, self.bool, &[b]);
                let either = self.value(op::LOGICAL_OR, self.bool, &[a_nan, b_nan]);
                return if compare == Nan {
                    either
                } else {
                    self.value(op::LOGICAL_NOT, self.bool, &[either])
                };
            }
        };
        self.value(opcode, self.bool, &[a, b])
    }

    fn int_compare(&mut self, compare: IntCompare, signed: bool, a: u32, b: u32) -> u32 {
        use IntCompare::*;
        let opcode = match (compare, signed) {
            (False, _) => return self.module.bool_constant(false),
            (True, _) => return self.module.bool_constant(true),
            (Eq, _) => op::I_EQUAL,
            (Ne, _) => op::I_NOT_EQUAL,
            (Lt, false) => op::U_LESS_THAN,
            (Lt, true) => op::S_LESS_THAN,
            (Le, false) => op::U_LESS_THAN_EQUAL,
            (Le, true) => op::S_LESS_THAN_EQUAL,
            (Gt, false) => op::U_GREATER_THAN,
            (Gt, true) => op::S_GREATER_THAN,
            (Ge, false) => op::U_GREATER_THAN_EQUAL,
            (Ge, true) => op::S_GREATER_THAN_EQUAL,
        };
        self.value(opcode, self.bool, &[a, b])
    }

    /// FSETP/ISETP outputs: `p = result combine accumulate`, `q = !result
    /// combine accumulate`.
    fn set_predicates(&mut self, p: u8, q: u8, result: u32, combine: Combine, accumulate: Pred) {
        let accumulate = self.read_predicate(accumulate);
        let inverse = self.value(op::LOGICAL_NOT, self.bool, &[result]);
        let opcode = match combine {
            Combine::And => op::LOGICAL_AND,
            Combine::Or => op::LOGICAL_OR,
            Combine::Xor => op::LOGICAL_NOT_EQUAL,
        };
        let first = self.value(opcode, self.bool, &[result, accumulate]);
        let second = self.value(opcode, self.bool, &[inverse, accumulate]);
        self.write_predicate(p, first);
        self.write_predicate(q, second);
    }

    /// The dispatch loop over `blocks`.
    fn program(&mut self, code: &[Instruction], blocks: &[Block]) -> Result<(), Unsupported> {
        let block_of: BTreeMap<usize, u32> = blocks
            .iter()
            .enumerate()
            .map(|(number, block)| (block.start, number as u32))
            .collect();
        let u32_ptr = self.u32_ptr;
        let bool_ptr = self.bool_ptr;
        let first = self.constant(0);
        let next = self
            .function
            .variable(&mut self.module, u32_ptr, Some(first));
        let budget_value = self.constant(LOOP_BUDGET);
        let budget = self
            .function
            .variable(&mut self.module, u32_ptr, Some(budget_value));
        let yes = self.module.bool_constant(true);
        let running = self
            .function
            .variable(&mut self.module, bool_ptr, Some(yes));
        let (header, body, continuing, merge) = (
            self.module.id(),
            self.module.id(),
            self.module.id(),
            self.module.id(),
        );
        self.function.emit(op::BRANCH, &[header]);
        // Header: run while not exited and within budget.
        self.function.label(header);
        let still = self.value(op::LOAD, self.bool, &[running]);
        let left = self.value(op::LOAD, self.u32, &[budget]);
        let zero = self.constant(0);
        let some_left = self.value(op::U_GREATER_THAN, self.bool, &[left, zero]);
        let go = self.value(op::LOGICAL_AND, self.bool, &[still, some_left]);
        self.function
            .emit(op::LOOP_MERGE, &[merge, continuing, spirv::NONE]);
        self.function
            .emit(op::BRANCH_CONDITIONAL, &[go, body, merge]);
        // Body: the switch over blocks.
        self.function.label(body);
        let selector = self.value(op::LOAD, self.u32, &[next]);
        let switch_merge = self.module.id();
        let labels: Vec<u32> = blocks.iter().map(|_| self.module.id()).collect();
        // An index past the blocks stops the invocation.
        let stop = self.module.id();
        let mut switch = vec![selector, stop];
        for (number, label) in labels.iter().enumerate() {
            switch.extend([number as u32, *label]);
        }
        self.function
            .emit(op::SELECTION_MERGE, &[switch_merge, spirv::NONE]);
        self.function.emit(op::SWITCH, &switch);
        self.function.label(stop);
        let no = self.module.bool_constant(false);
        self.function.emit(op::STORE, &[running, no]);
        self.function.emit(op::BRANCH, &[switch_merge]);
        for (number, block) in blocks.iter().enumerate() {
            self.function.label(labels[number]);
            for (index, instruction) in code
                .iter()
                .enumerate()
                .take(block.body_end)
                .skip(block.start)
            {
                self.instruction(index, instruction)?;
            }
            let fall = block_of
                .get(&block.end_index(block.body_end))
                .copied()
                .unwrap_or(u32::MAX);
            match block.end {
                End::Fall => {
                    let fall = self.constant(fall);
                    self.function.emit(op::STORE, &[next, fall]);
                }
                End::Branch { target, condition } => {
                    let target = self.constant(block_of[&target]);
                    let fall = self.constant(fall);
                    let taken = self.condition(condition);
                    let chosen = self.value(op::SELECT, self.u32, &[taken, target, fall]);
                    self.function.emit(op::STORE, &[next, chosen]);
                }
                End::Exit { condition } => {
                    let fall = self.constant(fall);
                    self.function.emit(op::STORE, &[next, fall]);
                    let exits = self.condition(condition);
                    let still = self.value(op::LOGICAL_NOT, self.bool, &[exits]);
                    self.function.emit(op::STORE, &[running, still]);
                }
            }
            self.function.emit(op::BRANCH, &[switch_merge]);
        }
        self.function.label(switch_merge);
        self.function.emit(op::BRANCH, &[continuing]);
        // Continue: spend one unit of the budget.
        self.function.label(continuing);
        let left = self.value(op::LOAD, self.u32, &[budget]);
        let one = self.constant(1);
        let less = self.value(op::I_SUB, self.u32, &[left, one]);
        self.function.emit(op::STORE, &[budget, less]);
        self.function.emit(op::BRANCH, &[header]);
        self.function.label(merge);
        self.function.emit(op::RETURN, &[]);
        Ok(())
    }

    fn condition(&mut self, condition: Guard) -> u32 {
        if condition.always() {
            return self.module.bool_constant(true);
        }
        let guard = self.read_predicate(condition.guard);
        let extra = self.read_predicate(condition.extra);
        self.value(op::LOGICAL_AND, self.bool, &[guard, extra])
    }
}

impl Block {
    /// The instruction after the block (its fall-through successor).
    fn end_index(&self, body_end: usize) -> usize {
        match self.end {
            End::Fall => body_end,
            _ => body_end + 1,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::decode::decode;
    use crate::spirv::tests::validate;

    /// A word with `fields` (bit offset, width, value) and a PT guard.
    fn word(fields: &[(u32, u32, u128)]) -> u128 {
        let mut raw = 7u128 << 12;
        for &(offset, width, value) in fields {
            raw &= !(((1u128 << width) - 1) << offset);
            raw |= value << offset;
        }
        raw
    }

    const EXIT: u128 = 0x94d | 7 << 12;

    fn translated(words: &[u128]) -> Result<Translated, Unsupported> {
        let code: Vec<Instruction> = words.iter().map(|&raw| decode(raw)).collect();
        translate(&code, [64, 1, 1])
    }

    fn translated_err(words: &[u128]) -> Unsupported {
        translated(words).unwrap_err()
    }

    fn valid(words: &[u128]) -> Translated {
        let translated = translated(words).unwrap();
        if let Some(problem) = validate(&translated.spirv) {
            panic!("{problem}");
        }
        translated
    }

    fn bra(from: usize, to: usize, guard: u128) -> u128 {
        let offset = (to as i64 - from as i64 - 1) * 16;
        let field = (offset as u64 & ((1 << 50) - 1)) as u128;
        word(&[(0, 12, 0x947), (12, 4, guard), (32, 50, field), (87, 3, 7)])
    }

    #[test]
    fn integer_and_logic_families_validate() {
        valid(&[
            // MOV R1, 0x5; IADD3 R2, R1, -R1, 0x7 (form 4: imm b);
            word(&[(0, 12, 0x802), (16, 8, 1), (32, 32, 5)]),
            word(&[
                (0, 12, 0x210),
                (16, 8, 2),
                (24, 8, 1),
                (32, 8, 1),
                (63, 1, 1),
                (64, 8, 255),
                (77, 3, 7),
                (81, 3, 7),
                (84, 3, 7),
                (87, 3, 7),
            ]),
            // IMAD R3, R2, R1, R2; LOP3.LUT R4, R1, R2, R3, 0x96, P0
            word(&[
                (0, 12, 0x224),
                (16, 8, 3),
                (24, 8, 2),
                (32, 8, 1),
                (64, 8, 2),
            ]),
            word(&[
                (0, 12, 0x212),
                (16, 8, 4),
                (24, 8, 1),
                (32, 8, 2),
                (64, 8, 3),
                (72, 8, 0x96),
                (81, 3, 0),
            ]),
            EXIT,
        ]);
    }

    #[test]
    fn constant_bank_operands_become_uniform_buffers() {
        // MOV R1, c[0x3][0x14]; IADD3 R2, R1, c[0x0][0x8], RZ;
        // FADD R3, R1, c[0x3][0x20]
        let translated = valid(&[
            word(&[(0, 12, 0xa02), (16, 8, 1), (54, 5, 3), (38, 16, 0x14)]),
            word(&[
                (0, 12, 0xa10),
                (16, 8, 2),
                (24, 8, 1),
                (54, 5, 0),
                (38, 16, 0x8),
                (64, 8, 255),
                (77, 3, 7),
                (81, 3, 7),
                (84, 3, 7),
                (87, 3, 7),
            ]),
            word(&[
                (0, 12, 0xa21),
                (16, 8, 3),
                (24, 8, 1),
                (54, 5, 3),
                (38, 16, 0x20),
            ]),
            EXIT,
        ]);
        let banks: Vec<(u32, u32)> = translated
            .descriptors
            .iter()
            .map(|descriptor| (descriptor.binding, descriptor.index))
            .collect();
        assert_eq!(banks, [(0, 3), (1, 0)]);
        assert_eq!(translated.cbuf_used_sizes[3], 0x24);
        assert_eq!(translated.cbuf_used_sizes[0], 0xc);
        let unaligned = translated_err(&[
            word(&[(0, 12, 0xa02), (16, 8, 1), (54, 5, 3), (38, 16, 0x2)]),
            EXIT,
        ]);
        assert!(unaligned.reason.contains("unaligned"));
    }

    #[test]
    fn float_families_validate() {
        valid(&[
            // FFMA.SAT R0, -|R1|, R2, R3; FMUL R4, R0, 2.0; FADD R5, R4, R0
            word(&[
                (0, 12, 0x223),
                (24, 8, 1),
                (72, 1, 1),
                (73, 1, 1),
                (32, 8, 2),
                (64, 8, 3),
                (77, 1, 1),
            ]),
            word(&[
                (0, 12, 0x820),
                (16, 8, 4),
                (32, 32, 0x4000_0000),
                (84, 3, 4),
            ]),
            word(&[(0, 12, 0x221), (16, 8, 5), (24, 8, 4), (32, 8, 0)]),
            // FSETP.GTU.AND P1, P2, R5, R0, PT; FSETP.NUM.XOR P3, PT, R5, R5, !P1
            word(&[
                (0, 12, 0x20b),
                (81, 3, 1),
                (84, 3, 2),
                (24, 8, 5),
                (32, 8, 0),
                (76, 4, 12),
                (87, 3, 7),
            ]),
            word(&[
                (0, 12, 0x20b),
                (81, 3, 3),
                (84, 3, 7),
                (24, 8, 5),
                (32, 8, 5),
                (76, 4, 7),
                (74, 2, 2),
                (87, 3, 1),
                (90, 1, 1),
            ]),
            EXIT,
        ]);
    }

    #[test]
    fn a_counted_loop_with_guards_and_ids_validates() {
        // S2R R0, SR_TID.X; MOV R1, 0;
        // loop: IADD3 R1, R1, 1, RZ; ISETP.LT.U32 P0, PT, R1, 10, PT;
        //       @P0 BRA loop; @!P0 EXIT (guarded exit, then a final EXIT).
        let words = [
            word(&[(0, 12, 0x919), (72, 8, 33)]),
            word(&[(0, 12, 0x802), (16, 8, 1), (32, 32, 0)]),
            word(&[
                (0, 12, 0x810),
                (16, 8, 1),
                (24, 8, 1),
                (32, 32, 1),
                (64, 8, 255),
                (77, 3, 7),
                (81, 3, 7),
                (84, 3, 7),
                (87, 3, 7),
            ]),
            word(&[
                (0, 12, 0x80c),
                (81, 3, 0),
                (84, 3, 7),
                (24, 8, 1),
                (32, 32, 10),
                (76, 3, 1),
                (87, 3, 7),
            ]),
            bra(4, 2, 0),
            word(&[(0, 12, 0x94d), (12, 4, 8)]),
            EXIT,
        ];
        let translated = valid(&words);
        assert_eq!(translated.stage, Stage::Compute);
        assert_eq!(translated.abi_version, ABI_VERSION);
    }

    #[test]
    fn unsupported_programs_name_the_instruction_and_its_offset() {
        let unknown = translated(&[word(&[(0, 12, 0x1ff)]), EXIT]).unwrap_err();
        assert_eq!(unknown.offset, Some(0));
        assert!(unknown.reason.contains("0x1ff"), "{}", unknown.reason);
        let rounding = translated(&[EXIT, word(&[(0, 12, 0x221), (78, 2, 3)]), EXIT]).unwrap_err();
        assert_eq!(rounding.offset, Some(16));
        let outside = translated(&[bra(0, 50, 7), EXIT]).unwrap_err();
        assert!(outside.reason.contains("outside"), "{}", outside.reason);
    }
}
