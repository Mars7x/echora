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
pub fn translate(code: &[Instruction], layout: &Layout) -> Result<Translated, Unsupported> {
    if code.is_empty() {
        return Err(unsupported(0, "empty program"));
    }
    let blocks = blocks(code)?;
    let mut emitter = Emitter::new(layout);
    emitter.program(code, &blocks)?;
    Ok(emitter.finish())
}

mod memory;

/// What a compute program needs beyond its code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub workgroup: [u32; 3],
    /// Bytes of shared memory, when known.
    pub shared_bytes: Option<u32>,
    /// Bytes of local memory per invocation, when known.
    pub local_bytes: Option<u32>,
}

impl Layout {
    /// A layout with only a workgroup size.
    pub fn workgroup(workgroup: [u32; 3]) -> Self {
        Self {
            workgroup,
            shared_bytes: None,
            local_bytes: None,
        }
    }
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
    uniform_registers: BTreeMap<u8, u32>,
    /// Whether the instruction being translated is a uniform one.
    uniform: bool,
    /// The 64-bit integer type, declared (with the Int64 capability) on
    /// first use.
    u64: Option<u32>,
    layout: Layout,
    memory: memory::Memory,
    push_fields: u32,
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
    fn new(layout: &Layout) -> Self {
        let workgroup = layout.workgroup;
        let mut module = Module::new();
        module.capability(capability::SHADER);
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
            uniform_registers: BTreeMap::new(),
            uniform: false,
            u64: None,
            layout: *layout,
            memory: memory::Memory::default(),
            push_fields: 0,
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
        // GLSL450 memory model, with physical pointers when global memory
        // is used.
        let addressing = if self.memory.physical {
            capability::ADDRESSING_PHYSICAL_STORAGE_BUFFER64
        } else {
            0
        };
        self.module.memory_model(addressing, 1);
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
            push_fields: self.push_fields,
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

    /// The variable of predicate `n`, in the uniform file (kept as 8-15)
    /// for a uniform instruction.
    fn predicate_variable(&mut self, n: u8) -> u32 {
        let n = if self.uniform { n + 8 } else { n };
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

    /// Whether register `n` is the zero register (RZ, or URZ in a uniform
    /// instruction).
    fn zero_register(&self, n: u8) -> bool {
        if self.uniform { n >= 63 } else { n == 255 }
    }

    /// Register `n`: a general register, or a uniform register in a
    /// uniform instruction.
    fn read_register(&mut self, n: u8) -> u32 {
        if self.uniform {
            return self.read_uniform(n.min(63));
        }
        if n == 255 {
            return self.constant(0);
        }
        let variable = self.register(n);
        self.value(op::LOAD, self.u32, &[variable])
    }

    fn write_register(&mut self, n: u8, value: u32) {
        if self.uniform {
            self.write_uniform(n.min(63), value);
        } else if n != 255 {
            let variable = self.register(n);
            self.function.emit(op::STORE, &[variable, value]);
        }
    }

    fn read_predicate(&mut self, pred: Pred) -> u32 {
        if pred.index == 7 {
            return self.module.bool_constant(!pred.negated);
        }
        let value = {
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

    /// The variable of uniform register `n` (declared, zero, on first
    /// use).
    fn uniform_variable(&mut self, n: u8) -> u32 {
        if let Some(&variable) = self.uniform_registers.get(&n) {
            return variable;
        }
        let zero = self.constant(0);
        let variable = self
            .function
            .variable(&mut self.module, self.u32_ptr, Some(zero));
        self.uniform_registers.insert(n, variable);
        variable
    }

    fn read_uniform(&mut self, n: u8) -> u32 {
        if n == 63 {
            return self.constant(0);
        }
        let variable = self.uniform_variable(n);
        self.value(op::LOAD, self.u32, &[variable])
    }

    fn write_uniform(&mut self, n: u8, value: u32) {
        if n != 63 {
            let variable = self.uniform_variable(n);
            self.function.emit(op::STORE, &[variable, value]);
        }
    }

    /// The 64-bit unsigned integer type.
    fn u64(&mut self) -> u32 {
        if let Some(ty) = self.u64 {
            return ty;
        }
        self.module.capability(capability::INT64);
        let ty = self.module.u64();
        self.u64 = Some(ty);
        ty
    }

    fn constant64(&mut self, value: u64) -> u32 {
        let ty = self.u64();
        self.module.constant64(ty, value)
    }

    /// A word zero- or sign-extended to 64 bits.
    fn widen(&mut self, value: u32, signed: bool) -> u32 {
        let ty = self.u64();
        let opcode = if signed { op::S_CONVERT } else { op::U_CONVERT };
        self.value(opcode, ty, &[value])
    }

    /// high:low as one 64-bit value.
    fn join(&mut self, low: u32, high: u32) -> u32 {
        let ty = self.u64();
        let low = self.widen(low, false);
        let high = self.widen(high, false);
        let thirty_two = self.constant(32);
        let high = self.value(op::SHIFT_LEFT_LOGICAL, ty, &[high, thirty_two]);
        self.value(op::BITWISE_OR, ty, &[high, low])
    }

    /// The low and high words of a 64-bit value.
    fn split(&mut self, value: u32) -> (u32, u32) {
        let ty = self.u64();
        let low = self.value(op::U_CONVERT, self.u32, &[value]);
        let thirty_two = self.constant(32);
        let high = self.value(op::SHIFT_RIGHT_LOGICAL, ty, &[value, thirty_two]);
        let high = self.value(op::U_CONVERT, self.u32, &[high]);
        (low, high)
    }

    /// Registers n:n+1 as a 64-bit value (RZ reads zero).
    fn read_pair(&mut self, n: u8) -> u32 {
        if self.zero_register(n) {
            return self.constant64(0);
        }
        let low = self.read_register(n);
        let high = self.read_register(n.saturating_add(1));
        self.join(low, high)
    }

    fn write_pair(&mut self, n: u8, value: u32) {
        if self.zero_register(n) {
            return;
        }
        let (low, high) = self.split(value);
        self.write_register(n, low);
        self.write_register(n.saturating_add(1), high);
    }

    /// A 64-bit operand pair: registers, or RZ.
    fn pair_operand(&mut self, index: usize, operand: Operand) -> Result<u32, Unsupported> {
        match operand.src {
            Src::Reg(reg) => Ok(self.read_pair(reg.0)),
            _ => Err(unsupported(
                index,
                "64-bit operand that is not a register pair",
            )),
        }
    }

    /// One added term of a carry chain: the value, or with `negate` its
    /// two's complement (`extended`: its bitwise inverse), widened.
    fn term(&mut self, value: u32, negate: bool, extended: bool) -> u32 {
        if !negate {
            return self.widen(value, false);
        }
        let inverse = self.value(op::NOT, self.u32, &[value]);
        let wide = self.widen(inverse, false);
        if extended {
            return wide;
        }
        let ty = self.u64();
        let one = self.constant64(1);
        self.value(op::I_ADD, ty, &[wide, one])
    }

    /// A predicate as a 64-bit 0 or 1.
    fn carry(&mut self, pred: Pred) -> u32 {
        let ty = self.u64();
        let condition = self.read_predicate(pred);
        let (one, zero) = (self.constant64(1), self.constant64(0));
        self.value(op::SELECT, ty, &[condition, one, zero])
    }

    /// The sum of 64-bit terms; its low word goes to `d` and to each
    /// carry output `outputs[i]` whether the sum reached `(i + 1) << 32`.
    fn carry_sum(&mut self, d: u8, terms: &[u32], outputs: &[u8]) {
        let ty = self.u64();
        let mut sum = terms[0];
        for &term in &terms[1..] {
            sum = self.value(op::I_ADD, ty, &[sum, term]);
        }
        let (low, high) = self.split(sum);
        self.write_register(d, low);
        for (i, &output) in outputs.iter().enumerate() {
            if output != 7 {
                let bound = self.constant(i as u32 + 1);
                let reached = self.value(op::U_GREATER_THAN_EQUAL, self.bool, &[high, bound]);
                self.write_predicate(output, reached);
            }
        }
    }

    fn glsl(&mut self, ty: u32, function: u32, operands: &[u32]) -> u32 {
        let glsl = self.module.glsl();
        let mut all = vec![glsl, function];
        all.extend_from_slice(operands);
        self.value(op::EXT_INST, ty, &all)
    }

    fn float_constant(&mut self, value: f32) -> u32 {
        self.module.constant(self.f32, value.to_bits())
    }

    fn write_float(&mut self, d: u8, value: u32) {
        let bits = self.value(op::BITCAST, self.u32, &[value]);
        self.write_register(d, bits);
    }

    /// `result combine accumulate`.
    fn combined(&mut self, result: u32, combine: Combine, accumulate: Pred) -> u32 {
        let accumulate = self.read_predicate(accumulate);
        let opcode = match combine {
            Combine::And => op::LOGICAL_AND,
            Combine::Or => op::LOGICAL_OR,
            Combine::Xor => op::LOGICAL_NOT_EQUAL,
        };
        self.value(opcode, self.bool, &[result, accumulate])
    }

    /// The word at byte address `address` (a value) of a bank's buffer,
    /// zero outside the bank; `words` 1 or 2.
    fn bank_words(&mut self, variable: u32, address: u32, words: u32) -> Vec<u32> {
        let bytes = words * 4;
        let limit = self.constant(BANK_SIZE - bytes);
        let inside = self.value(op::U_LESS_THAN_EQUAL, self.bool, &[address, limit]);
        let clamped = self.glsl(self.u32, glsl::U_MIN, &[address, limit]);
        let pointer = self.module.pointer(storage::UNIFORM, self.u32);
        let (member, zero) = (self.constant(0), self.constant(0));
        let (two, four, three) = (self.constant(2), self.constant(4), self.constant(3));
        let mut values = Vec::new();
        for word in 0..words {
            let offset = self.constant(word * 4);
            let at = self.value(op::I_ADD, self.u32, &[clamped, offset]);
            let vector = self.value(op::SHIFT_RIGHT_LOGICAL, self.u32, &[at, four]);
            let component = self.value(op::SHIFT_RIGHT_LOGICAL, self.u32, &[at, two]);
            let component = self.value(op::BITWISE_AND, self.u32, &[component, three]);
            let element = self.value(
                op::ACCESS_CHAIN,
                pointer,
                &[variable, member, vector, component],
            );
            let value = self.value(op::LOAD, self.u32, &[element]);
            values.push(self.value(op::SELECT, self.u32, &[inside, value, zero]));
        }
        values
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

    /// The word at byte `offset` of constant bank `bank`. Banks past the
    /// 18 a program can bind are never bound and read zero.
    fn constant_word(&mut self, bank: u8, offset: u16) -> Result<u32, String> {
        if !offset.is_multiple_of(4) {
            return Err(format!("unaligned constant c[{bank}][{offset:#x}]"));
        }
        if usize::from(bank) >= self.cbuf_used_sizes.len() {
            return Ok(self.constant(0));
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
            Src::Uniform(reg) => Ok(self.read_uniform(reg.0)),
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
        self.uniform = instruction.uniform;
        let result = self.guarded(instruction.guard, |e| e.operation(index, op));
        self.uniform = false;
        result
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
                extended,
            } => {
                let outputs = carry_out.map(|pred| pred.index);
                if outputs == [7, 7] && !extended {
                    let (a, b, c) = (
                        self.signed_integer(index, a)?,
                        self.signed_integer(index, b)?,
                        self.signed_integer(index, c)?,
                    );
                    let sum = self.value(op::I_ADD, self.u32, &[a, b]);
                    let sum = self.value(op::I_ADD, self.u32, &[sum, c]);
                    self.write_register(d.0, sum);
                } else {
                    let mut terms = Vec::new();
                    for operand in [a, b, c] {
                        let value = self.integer(index, operand)?;
                        terms.push(self.term(value, operand.neg, extended));
                    }
                    if extended {
                        for pred in carry_in {
                            terms.push(self.carry(pred));
                        }
                    }
                    self.carry_sum(d.0, &terms, &outputs);
                }
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
                let (a, b) = (self.integer(index, a)?, self.integer(index, b)?);
                let mut result = self.int_compare(compare, signed, a, b);
                // .EX: the high words decide unless equal, then the low
                // words' result does.
                if let Some(low) = extended
                    && !matches!(compare, IntCompare::False | IntCompare::True)
                {
                    let equal = self.value(op::I_EQUAL, self.bool, &[a, b]);
                    let low = self.read_predicate(low);
                    result = self.value(op::SELECT, self.bool, &[equal, low, result]);
                }
                self.set_predicates(p, q, result, combine, accumulate);
            }
            Op::R2ur { d, src } => {
                let value = self.read_register(src.0);
                self.write_uniform(d.0.min(63), value);
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
            Op::ImadWide { d, a, b, c, signed } => {
                let ty = self.u64();
                let (a, b) = (self.integer(index, a)?, self.integer(index, b)?);
                let c = self.pair_operand(index, c)?;
                let (a, b) = (self.widen(a, signed), self.widen(b, signed));
                let product = self.value(op::I_MUL, ty, &[a, b]);
                let sum = self.value(op::I_ADD, ty, &[product, c]);
                self.write_pair(d.0, sum);
            }
            Op::Lea {
                d,
                a,
                b,
                c,
                shift,
                high,
                negate_shifted,
                extended,
                overflow,
                carry_in,
            } => {
                let a = self.read_register(a.0);
                let amount = self.constant(u32::from(shift));
                let shifted = if high {
                    let c = self.integer(index, c)?;
                    let ty = self.u64();
                    let joined = self.join(a, c);
                    let amount = self.constant(32 - u32::from(shift).min(32));
                    let wide = self.value(op::SHIFT_RIGHT_LOGICAL, ty, &[joined, amount]);
                    self.value(op::U_CONVERT, self.u32, &[wide])
                } else {
                    self.value(op::SHIFT_LEFT_LOGICAL, self.u32, &[a, amount])
                };
                let b_value = self.integer(index, b)?;
                if overflow == 7 && !extended && !negate_shifted && !b.neg {
                    let sum = self.value(op::I_ADD, self.u32, &[shifted, b_value]);
                    self.write_register(d.0, sum);
                } else {
                    let mut terms = vec![
                        self.term(shifted, negate_shifted, extended),
                        self.term(b_value, b.neg, extended),
                    ];
                    if extended {
                        terms.push(self.carry(carry_in));
                    }
                    self.carry_sum(d.0, &terms, &[overflow]);
                }
            }
            Op::Shf {
                d,
                low,
                shift,
                high,
                kind,
                wrap,
                right,
                high_word,
            } => {
                let ty = self.u64();
                let width = if kind < 2 { 64 } else { 32 };
                let signed = kind & 1 == 0;
                let low = self.read_register(low.0);
                let high = self.integer(index, high)?;
                let amount = self.integer(index, shift)?;
                let amount = if wrap {
                    let mask = self.constant(width - 1);
                    self.value(op::BITWISE_AND, self.u32, &[amount, mask])
                } else {
                    let width = self.constant(width);
                    self.glsl(self.u32, glsl::U_MIN, &[amount, width])
                };
                // A shift by 64 (a clamped 64-bit shift) is done as 63 and
                // then fixed: a signed right shift fills with the sign
                // either way, anything else gives zero.
                let sixty_three = self.constant(63);
                let safe = self.glsl(self.u32, glsl::U_MIN, &[amount, sixty_three]);
                let value = self.join(low, high);
                let opcode = match (right, signed) {
                    (true, true) => op::SHIFT_RIGHT_ARITHMETIC,
                    (true, false) => op::SHIFT_RIGHT_LOGICAL,
                    (false, _) => op::SHIFT_LEFT_LOGICAL,
                };
                let mut shifted = self.value(opcode, ty, &[value, safe]);
                if width == 64 && !wrap && !(right && signed) {
                    let sixty_four = self.constant(64);
                    let all = self.value(op::I_EQUAL, self.bool, &[amount, sixty_four]);
                    let zero = self.constant64(0);
                    shifted = self.value(op::SELECT, ty, &[all, zero, shifted]);
                }
                let (low, high) = self.split(shifted);
                self.write_register(d.0, if high_word { high } else { low });
            }
            Op::Prmt {
                d,
                a,
                selector,
                c,
                mode,
            } => {
                if mode != 0 {
                    return Err(unsupported(index, format!("PRMT mode {mode}")));
                }
                let a = self.read_register(a.0);
                let selector = self.integer(index, selector)?;
                let c = self.integer(index, c)?;
                let (three, four, seven, eight) = (
                    self.constant(3),
                    self.constant(4),
                    self.constant(7),
                    self.constant(8),
                );
                let (byte_mask, sign_bit, zero) =
                    (self.constant(0xff), self.constant(0x80), self.constant(0));
                let mut result = self.constant(0);
                for byte in 0..4u32 {
                    let at = self.constant(byte * 4);
                    let nibble = self.value(op::SHIFT_RIGHT_LOGICAL, self.u32, &[selector, at]);
                    let which = self.value(op::BITWISE_AND, self.u32, &[nibble, seven]);
                    let from_c = self.value(op::U_GREATER_THAN_EQUAL, self.bool, &[which, four]);
                    let source = self.value(op::SELECT, self.u32, &[from_c, c, a]);
                    let lane = self.value(op::BITWISE_AND, self.u32, &[which, three]);
                    let bit = self.value(op::I_MUL, self.u32, &[lane, eight]);
                    let value = self.value(op::SHIFT_RIGHT_LOGICAL, self.u32, &[source, bit]);
                    let value = self.value(op::BITWISE_AND, self.u32, &[value, byte_mask]);
                    // Nibble bit 3: the byte's sign bit, replicated.
                    let replicate = self.value(op::BITWISE_AND, self.u32, &[nibble, eight]);
                    let replicate = self.value(op::I_NOT_EQUAL, self.bool, &[replicate, zero]);
                    let sign = self.value(op::BITWISE_AND, self.u32, &[value, sign_bit]);
                    let negative = self.value(op::I_NOT_EQUAL, self.bool, &[sign, zero]);
                    let fill = self.value(op::SELECT, self.u32, &[negative, byte_mask, zero]);
                    let value = self.value(op::SELECT, self.u32, &[replicate, fill, value]);
                    let place = self.constant(byte * 8);
                    let value = self.value(op::SHIFT_LEFT_LOGICAL, self.u32, &[value, place]);
                    result = self.value(op::BITWISE_OR, self.u32, &[result, value]);
                }
                self.write_register(d.0, result);
            }
            Op::Sgxt {
                d,
                a,
                bits,
                signed,
                wrap,
            } => {
                if wrap {
                    return Err(unsupported(index, "SGXT.W"));
                }
                let a = self.read_register(a.0);
                let bits = self.integer(index, bits)?;
                let thirty_two = self.constant(32);
                let count = self.glsl(self.u32, glsl::U_MIN, &[bits, thirty_two]);
                let zero = self.constant(0);
                let opcode = if signed {
                    op::BIT_FIELD_S_EXTRACT
                } else {
                    op::BIT_FIELD_U_EXTRACT
                };
                let value = self.value(opcode, self.u32, &[a, zero, count]);
                self.write_register(d.0, value);
            }
            Op::Sel {
                d,
                a,
                b,
                condition,
                float,
            } => {
                let (a, b) = if float {
                    let (a, b) = (self.float(index, a)?, self.float(index, b)?);
                    (
                        self.value(op::BITCAST, self.u32, &[a]),
                        self.value(op::BITCAST, self.u32, &[b]),
                    )
                } else {
                    (self.integer(index, a)?, self.integer(index, b)?)
                };
                let condition = self.read_predicate(condition);
                let value = self.value(op::SELECT, self.u32, &[condition, a, b]);
                self.write_register(d.0, value);
            }
            Op::Fmnmx { d, a, b, min } => {
                let (a, b) = (self.float(index, a)?, self.float(index, b)?);
                let value = self.min_max(self.f32, min, glsl::N_MIN, glsl::N_MAX, a, b);
                self.write_float(d.0, value);
            }
            Op::Imnmx {
                d,
                a,
                b,
                min,
                signed,
            } => {
                let (a, b) = (self.integer(index, a)?, self.integer(index, b)?);
                let (smaller, larger) = if signed {
                    (glsl::S_MIN, glsl::S_MAX)
                } else {
                    (glsl::U_MIN, glsl::U_MAX)
                };
                let value = self.min_max(self.u32, min, smaller, larger, a, b);
                self.write_register(d.0, value);
            }
            Op::Mufu { d, function, src } => {
                let x = self.float(index, src)?;
                let value = match function {
                    // SIN and COS take turns, not radians.
                    0 | 1 => {
                        let turn = self.float_constant(std::f32::consts::TAU);
                        let radians = self.value(op::F_MUL, self.f32, &[x, turn]);
                        let f = if function == 0 { glsl::COS } else { glsl::SIN };
                        self.glsl(self.f32, f, &[radians])
                    }
                    2 => self.glsl(self.f32, glsl::EXP2, &[x]),
                    3 => self.glsl(self.f32, glsl::LOG2, &[x]),
                    4 => {
                        let one = self.float_constant(1.0);
                        self.value(op::F_DIV, self.f32, &[one, x])
                    }
                    5 => self.glsl(self.f32, glsl::INVERSE_SQRT, &[x]),
                    8 => self.glsl(self.f32, glsl::SQRT, &[x]),
                    9 => self.glsl(self.f32, glsl::TANH, &[x]),
                    _ => return Err(unsupported(index, format!("MUFU function {function}"))),
                };
                self.write_float(d.0, value);
            }
            Op::F2i {
                d,
                src,
                signed,
                dst_size,
                src_size,
                rounding,
            } => {
                if (dst_size, src_size) != (2, 2) {
                    return Err(unsupported(index, "F2I other than 32-bit"));
                }
                let x = self.float(index, src)?;
                let rounded = self.rounded(x, rounding);
                let value = self.saturating_integer(rounded, signed);
                self.write_register(d.0, value);
            }
            Op::I2f {
                d,
                src,
                signed,
                dst_size,
                src_size,
                rounding,
            } => {
                if (dst_size, src_size) != (2, 2) {
                    return Err(unsupported(index, "I2F other than 32-bit"));
                }
                if rounding != 0 {
                    return Err(unsupported(index, "I2F rounding mode"));
                }
                let x = self.integer(index, src)?;
                let opcode = if signed {
                    op::CONVERT_S_TO_F
                } else {
                    op::CONVERT_U_TO_F
                };
                let value = self.value(opcode, self.f32, &[x]);
                self.write_float(d.0, value);
            }
            Op::Frnd {
                d,
                src,
                rounding,
                dst_size,
                src_size,
            } => {
                if (dst_size, src_size) != (2, 2) {
                    return Err(unsupported(index, "FRND other than 32-bit"));
                }
                let x = self.float(index, src)?;
                let value = self.rounded(x, rounding);
                self.write_float(d.0, value);
            }
            Op::Fset {
                d,
                a,
                b,
                compare,
                combine,
                accumulate,
            } => {
                let (a, b) = (self.float(index, a)?, self.float(index, b)?);
                let result = self.float_compare(compare, a, b);
                let result = self.combined(result, combine, accumulate);
                let (one, zero) = (self.float_constant(1.0), self.float_constant(0.0));
                let value = self.value(op::SELECT, self.f32, &[result, one, zero]);
                self.write_float(d.0, value);
            }
            Op::Plop3 { p, q, srcs, luts } => {
                let inputs = srcs.map(|pred| self.read_predicate(pred));
                let zero = self.constant(0);
                let mut entry = zero;
                for (input, weight) in inputs.into_iter().zip([4, 2, 1]) {
                    let weight = self.constant(weight);
                    let term = self.value(op::SELECT, self.u32, &[input, weight, zero]);
                    entry = self.value(op::BITWISE_OR, self.u32, &[entry, term]);
                }
                let one = self.constant(1);
                for (output, lut) in [(p, luts[0]), (q, luts[1])] {
                    if output == 7 {
                        continue;
                    }
                    let table = self.constant(u32::from(lut));
                    let bit = self.value(op::SHIFT_RIGHT_LOGICAL, self.u32, &[table, entry]);
                    let bit = self.value(op::BITWISE_AND, self.u32, &[bit, one]);
                    let set = self.value(op::I_NOT_EQUAL, self.bool, &[bit, zero]);
                    self.write_predicate(output, set);
                }
            }
            Op::Cs2r { d, special, wide } => {
                if special != 255 {
                    return Err(unsupported(
                        index,
                        format!("CS2R special register {special}"),
                    ));
                }
                let zero = self.constant(0);
                self.write_register(d.0, zero);
                if wide && !self.zero_register(d.0) {
                    self.write_register(d.0.saturating_add(1), zero);
                }
            }
            Op::Ldc {
                d,
                index: base,
                bank,
                offset,
                size,
                mode,
            } => {
                if mode != 0 {
                    return Err(unsupported(index, format!("LDC mode {mode}")));
                }
                let words = match size {
                    4 => 1,
                    5 => 2,
                    _ => return Err(unsupported(index, format!("LDC size {size}"))),
                };
                if usize::from(bank) >= self.cbuf_used_sizes.len() {
                    // A bank that is never bound reads zero.
                    let zero = self.constant(0);
                    for n in 0..words {
                        if !self.zero_register(d.0) {
                            self.write_register(d.0.saturating_add(n as u8), zero);
                        }
                    }
                    return Ok(());
                }
                let variable = self
                    .bank(bank)
                    .map_err(|reason| unsupported(index, reason))?;
                if let Some(used) = self.cbuf_used_sizes.get_mut(usize::from(bank)) {
                    *used = BANK_SIZE;
                }
                let base = self.read_register(base.0);
                let offset = self.constant(u32::from(offset));
                let address = self.value(op::I_ADD, self.u32, &[base, offset]);
                let values = self.bank_words(variable, address, words);
                for (n, value) in values.into_iter().enumerate() {
                    if !self.zero_register(d.0) {
                        self.write_register(d.0.saturating_add(n as u8), value);
                    }
                }
            }
            Op::Uldc { d, src, size } => {
                let words = match size {
                    4 => 1,
                    5 => 2,
                    _ => return Err(unsupported(index, format!("ULDC size {size}"))),
                };
                let Src::Const { bank, offset } = src else {
                    return Err(unsupported(index, "bindless ULDC"));
                };
                for n in 0..words {
                    let value = self
                        .constant_word(bank, offset.wrapping_add(n * 4))
                        .map_err(|reason| unsupported(index, reason))?;
                    if d.0 != 63 {
                        self.write_uniform(d.0.saturating_add(n as u8), value);
                    }
                }
            }
            Op::Load {
                d,
                space,
                address,
                size,
                condition,
            } => self.load(index, d.0, space, address, size, condition)?,
            Op::Store {
                data,
                space,
                address,
                size,
            } => self.store(index, data.0, space, address, size)?,
            Op::Bar { mode } => {
                if mode != 0 {
                    return Err(unsupported(index, format!("BAR mode {mode}")));
                }
                self.barrier();
            }
            Op::Bssy | Op::Bsync | Op::Depbar => {}
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

    /// The minimum of a and b where `min` holds, else the maximum.
    fn min_max(&mut self, ty: u32, min: Pred, smaller: u32, larger: u32, a: u32, b: u32) -> u32 {
        if min.index == 7 {
            let f = if min.negated { larger } else { smaller };
            return self.glsl(ty, f, &[a, b]);
        }
        let condition = self.read_predicate(min);
        let low = self.glsl(ty, smaller, &[a, b]);
        let high = self.glsl(ty, larger, &[a, b]);
        self.value(op::SELECT, ty, &[condition, low, high])
    }

    /// A float rounded to an integral value: 0 to nearest even, 1 down, 2
    /// up, 3 toward zero.
    fn rounded(&mut self, x: u32, rounding: u8) -> u32 {
        let f = match rounding {
            0 => glsl::ROUND_EVEN,
            1 => glsl::FLOOR,
            2 => glsl::CEIL,
            _ => glsl::TRUNC,
        };
        self.glsl(self.f32, f, &[x])
    }

    /// An integral float converted as the hardware does: saturated to the
    /// integer's range, NaN to zero.
    fn saturating_integer(&mut self, x: u32, signed: bool) -> u32 {
        // The largest float below the range's end, and the end itself.
        let (low, high, end, top) = if signed {
            (
                -2_147_483_648.0f32,
                2_147_483_520.0f32,
                2_147_483_648.0f32,
                0x7fff_ffff,
            )
        } else {
            (0.0, 4_294_967_040.0, 4_294_967_296.0, u32::MAX)
        };
        let (low, high, end) = (
            self.float_constant(low),
            self.float_constant(high),
            self.float_constant(end),
        );
        let clamped = self.glsl(self.f32, glsl::F_CLAMP, &[x, low, high]);
        let opcode = if signed {
            op::CONVERT_F_TO_S
        } else {
            op::CONVERT_F_TO_U
        };
        let value = self.value(opcode, self.u32, &[clamped]);
        let over = self.value(op::F_ORD_GREATER_THAN_EQUAL, self.bool, &[x, end]);
        let top = self.constant(top);
        let value = self.value(op::SELECT, self.u32, &[over, top, value]);
        let nan = self.value(op::IS_NAN, self.bool, &[x]);
        let zero = self.constant(0);
        self.value(op::SELECT, self.u32, &[nan, zero, value])
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
        translate(&code, &Layout::workgroup([64, 1, 1]))
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
    fn carries_wide_multiplies_shifts_and_permutes_validate() {
        valid(&[
            // IADD3 R2, P0, P1, R1, R3, RZ; IADD3.X R4, R1, R3, RZ, P0, P1
            word(&[
                (0, 12, 0x210),
                (16, 8, 2),
                (24, 8, 1),
                (32, 8, 3),
                (64, 8, 255),
                (81, 3, 0),
                (84, 3, 1),
                (87, 3, 7),
                (77, 3, 7),
            ]),
            word(&[
                (0, 12, 0x210),
                (16, 8, 4),
                (24, 8, 1),
                (32, 8, 3),
                (63, 1, 1),
                (64, 8, 255),
                (74, 1, 1),
                (81, 3, 7),
                (84, 3, 7),
                (87, 3, 0),
                (77, 3, 1),
            ]),
            // LEA R5, P2, R1, R3, 0x2; LEA.HI.X R6, R1, R4, R2, 0x2, P2
            word(&[
                (0, 12, 0x211),
                (16, 8, 5),
                (24, 8, 1),
                (32, 8, 3),
                (64, 8, 255),
                (75, 5, 2),
                (81, 3, 2),
                (87, 3, 7),
            ]),
            word(&[
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
            ]),
            // IMAD.WIDE.U32 R8, R1, 0x10, R2; IMAD.WIDE R10, R1, R3, RZ
            word(&[
                (0, 12, 0x825),
                (16, 8, 8),
                (24, 8, 1),
                (32, 32, 0x10),
                (64, 8, 2),
                (81, 3, 7),
            ]),
            word(&[
                (0, 12, 0x225),
                (16, 8, 10),
                (24, 8, 1),
                (32, 8, 3),
                (64, 8, 255),
                (73, 1, 1),
                (81, 3, 7),
            ]),
            // SHF.R.S32.HI R12, RZ, 0x1f, R1; SHF.L.U64 R13, R1, R3, R2;
            // SHF.R.W.S64 R14, R1, R3, R2
            word(&[
                (0, 12, 0x819),
                (16, 8, 12),
                (24, 8, 255),
                (32, 32, 0x1f),
                (64, 8, 1),
                (73, 2, 2),
                (76, 1, 1),
                (80, 1, 1),
            ]),
            word(&[
                (0, 12, 0x219),
                (16, 8, 13),
                (24, 8, 1),
                (32, 8, 3),
                (64, 8, 2),
                (73, 2, 1),
            ]),
            word(&[
                (0, 12, 0x219),
                (16, 8, 14),
                (24, 8, 1),
                (32, 8, 3),
                (64, 8, 2),
                (73, 2, 0),
                (75, 1, 1),
                (76, 1, 1),
            ]),
            // PRMT R15, R1, 0x7650, R3; PRMT R16, R1, R2, R3;
            // SGXT R17, R1, 0x8
            word(&[
                (0, 12, 0x816),
                (16, 8, 15),
                (24, 8, 1),
                (32, 32, 0x7650),
                (64, 8, 3),
            ]),
            word(&[
                (0, 12, 0x216),
                (16, 8, 16),
                (24, 8, 1),
                (32, 8, 2),
                (64, 8, 3),
            ]),
            word(&[
                (0, 12, 0x81a),
                (16, 8, 17),
                (24, 8, 1),
                (32, 32, 8),
                (73, 1, 1),
            ]),
            // IMNMX R18, R1, R2, !P0; PLOP3.LUT P3, P4, P0, P1, PT, 0x80, 0x8
            word(&[
                (0, 12, 0x217),
                (16, 8, 18),
                (24, 8, 1),
                (32, 8, 2),
                (73, 1, 1),
                (87, 3, 0),
                (90, 1, 1),
            ]),
            word(&[
                (0, 12, 0x81c),
                (16, 8, 0x08),
                (64, 3, 0),
                (72, 5, 0x10),
                (81, 3, 3),
                (84, 3, 4),
                (87, 3, 0),
                (77, 3, 1),
                (68, 3, 7),
            ]),
            // CS2R R20, SRZ (.64); BSSY B0; BSYNC B0; DEPBAR
            word(&[(0, 12, 0x805), (16, 8, 20), (72, 8, 255), (80, 1, 1)]),
            word(&[(0, 12, 0x945)]),
            word(&[(0, 12, 0x941)]),
            word(&[(0, 12, 0x91a)]),
            EXIT,
        ]);
    }

    #[test]
    fn selects_conversions_and_special_functions_validate() {
        let mut words = vec![
            // SEL R2, R1, 0x5, P0; FSEL R3, -R1, 1.0, !P1
            word(&[
                (0, 12, 0x807),
                (16, 8, 2),
                (24, 8, 1),
                (32, 32, 5),
                (87, 3, 0),
            ]),
            word(&[
                (0, 12, 0x808),
                (16, 8, 3),
                (24, 8, 1),
                (72, 1, 1),
                (32, 32, 0x3f80_0000),
                (87, 3, 1),
                (90, 1, 1),
            ]),
            // FMNMX R4, R1, R3, PT; FMNMX R5, R1, R3, P2
            word(&[
                (0, 12, 0x209),
                (16, 8, 4),
                (24, 8, 1),
                (32, 8, 3),
                (87, 3, 7),
            ]),
            word(&[
                (0, 12, 0x209),
                (16, 8, 5),
                (24, 8, 1),
                (32, 8, 3),
                (87, 3, 2),
            ]),
            // F2I.TRUNC.NTZ R6, R1; F2I.U32 R7, |R1|
            word(&[
                (0, 12, 0x305),
                (16, 8, 6),
                (32, 8, 1),
                (72, 1, 1),
                (75, 2, 2),
                (78, 2, 3),
                (84, 2, 2),
            ]),
            word(&[
                (0, 12, 0x305),
                (16, 8, 7),
                (32, 8, 1),
                (62, 1, 1),
                (75, 2, 2),
                (84, 2, 2),
            ]),
            // I2F R8, R2 (signed); I2FP.F32.U32 R9, R2; FRND.FLOOR R10, R1
            word(&[
                (0, 12, 0x306),
                (16, 8, 8),
                (32, 8, 2),
                (74, 1, 1),
                (75, 2, 2),
                (84, 2, 2),
            ]),
            word(&[
                (0, 12, 0x245),
                (16, 8, 9),
                (32, 8, 2),
                (75, 2, 2),
                (84, 2, 2),
            ]),
            word(&[
                (0, 12, 0x307),
                (16, 8, 10),
                (32, 8, 1),
                (75, 2, 2),
                (78, 2, 1),
                (84, 2, 2),
            ]),
            // FSET.BF.GT.AND R11, R1, R3, PT
            word(&[
                (0, 12, 0x20a),
                (16, 8, 11),
                (24, 8, 1),
                (32, 8, 3),
                (76, 4, 4),
                (87, 3, 7),
            ]),
        ];
        // MUFU of each translated function.
        for function in [0, 1, 2, 3, 4, 5, 8, 9] {
            words.push(word(&[
                (0, 12, 0x308),
                (16, 8, 12),
                (32, 8, 1),
                (74, 6, function),
            ]));
        }
        words.push(EXIT);
        valid(&words);
        let rcp64 = translated_err(&[word(&[(0, 12, 0x308), (16, 8, 2), (74, 6, 6)]), EXIT]);
        assert!(rcp64.reason.contains("MUFU"));
    }

    #[test]
    fn indexed_and_uniform_constant_loads_validate() {
        let translated = valid(&[
            // LDC R2, c[0x3][R1+0x10]; LDC.64 R4, c[0x3][R1+0x20]
            word(&[
                (0, 12, 0xb82),
                (16, 8, 2),
                (24, 8, 1),
                (54, 5, 3),
                (38, 16, 0x10),
                (73, 3, 4),
            ]),
            word(&[
                (0, 12, 0xb82),
                (16, 8, 4),
                (24, 8, 1),
                (54, 5, 3),
                (38, 16, 0x20),
                (73, 3, 5),
            ]),
            // ULDC.64 UR4, c[0x0][0x118]; MOV R6, UR5; IADD3 R7, R6, UR4, RZ
            word(&[
                (0, 12, 0xab9),
                (16, 6, 4),
                (54, 5, 0),
                (38, 16, 0x118),
                (73, 3, 5),
            ]),
            word(&[(0, 12, 0xc02), (16, 8, 6), (32, 6, 5)]),
            word(&[
                (0, 12, 0xc10),
                (16, 8, 7),
                (24, 8, 6),
                (32, 6, 4),
                (64, 8, 255),
                (77, 3, 7),
                (81, 3, 7),
                (84, 3, 7),
                (87, 3, 7),
            ]),
            EXIT,
        ]);
        let banks: Vec<u32> = translated
            .descriptors
            .iter()
            .map(|descriptor| descriptor.index)
            .collect();
        assert_eq!(banks, [3, 0]);
        // An indexed load may read anywhere in its bank.
        assert_eq!(translated.cbuf_used_sizes[3], 0x10000);
        assert_eq!(translated.cbuf_used_sizes[0], 0x120);
    }

    fn valid_with(words: &[u128], layout: &Layout) -> Translated {
        let code: Vec<Instruction> = words.iter().map(|&raw| decode(raw)).collect();
        let translated = translate(&code, layout).unwrap();
        if let Some(problem) = validate(&translated.spirv) {
            panic!("{problem}");
        }
        translated
    }

    #[test]
    fn global_memory_goes_through_the_page_table() {
        let translated = valid(&[
            // LDG.E.64 R2, [R4.64+0x10] (no uniform form)
            word(&[
                (0, 12, 0x381),
                (16, 8, 2),
                (24, 8, 4),
                (40, 24, 0x10),
                (72, 1, 1),
                (73, 3, 5),
            ]),
            // @P0-gated LDG.E.S8 R6, [R1+UR4.64-0x4] (P0 stored reversed)
            word(&[
                (0, 12, 0x981),
                (16, 8, 6),
                (24, 8, 1),
                (32, 6, 4),
                (40, 24, 0xff_fffc),
                (64, 3, 7),
                (72, 1, 1),
                (73, 3, 1),
                (91, 1, 1),
            ]),
            // LDG.E.128 R8, [R4.64]; STG.E.128 [R4.64+0x20], R8
            word(&[
                (0, 12, 0x981),
                (16, 8, 8),
                (24, 8, 4),
                (32, 6, 63),
                (72, 1, 1),
                (73, 3, 6),
                (90, 1, 1),
                (91, 1, 1),
            ]),
            word(&[
                (0, 12, 0x986),
                (24, 8, 4),
                (32, 8, 8),
                (40, 24, 0x20),
                (64, 6, 63),
                (72, 1, 1),
                (73, 3, 6),
                (90, 1, 1),
                (91, 1, 1),
            ]),
            // STG.E [R4.64], R2
            word(&[
                (0, 12, 0x386),
                (24, 8, 4),
                (32, 8, 2),
                (72, 1, 1),
                (73, 3, 4),
            ]),
            EXIT,
        ]);
        assert_eq!(translated.push_fields, echora_shader_abi::PUSH_PAGE_TABLE);
    }

    #[test]
    fn shared_and_local_memory_are_arrays() {
        let words = [
            // STS [R1.X4+0x8], R2; BAR.SYNC; LDS.64 R4, [R1+UR5+0x4];
            // LDS.U16 R6, [R1+0x2]
            word(&[
                (0, 12, 0x388),
                (24, 8, 1),
                (32, 8, 2),
                (40, 24, 8),
                (73, 3, 4),
                (78, 2, 1),
            ]),
            word(&[(0, 12, 0xb1d)]),
            word(&[
                (0, 12, 0x984),
                (16, 8, 4),
                (24, 8, 1),
                (32, 6, 5),
                (40, 24, 4),
                (73, 3, 5),
                (91, 1, 1),
            ]),
            word(&[
                (0, 12, 0x984),
                (16, 8, 6),
                (24, 8, 1),
                (40, 24, 2),
                (73, 3, 2),
            ]),
            // STL [R1+0x10], R6; LDL R7, [R1+0x10]
            word(&[
                (0, 12, 0x387),
                (24, 8, 1),
                (32, 8, 6),
                (40, 24, 0x10),
                (73, 3, 4),
            ]),
            word(&[
                (0, 12, 0x983),
                (16, 8, 7),
                (24, 8, 1),
                (40, 24, 0x10),
                (73, 3, 4),
            ]),
            EXIT,
        ];
        let layout = Layout {
            workgroup: [64, 1, 1],
            shared_bytes: Some(0x1000),
            local_bytes: Some(0x40),
        };
        let translated = valid_with(&words, &layout);
        assert_eq!(translated.push_fields, 0);
        // Without sizes, the accesses are unsupported.
        let code: Vec<Instruction> = words.iter().map(|&raw| decode(raw)).collect();
        let missing = translate(&code, &Layout::workgroup([64, 1, 1])).unwrap_err();
        assert!(missing.reason.contains("shared memory size"));
    }

    #[test]
    fn uniform_instructions_use_the_uniform_files() {
        let words = [
            // S2UR UR4, SR_CTAID.X; UIADD3 UR5, UR4, 0x10, URZ
            word(&[(0, 12, 0x9c3), (16, 8, 4), (72, 8, 37)]),
            word(&[
                (0, 12, 0x890),
                (16, 8, 5),
                (24, 8, 4),
                (32, 32, 0x10),
                (64, 8, 63),
                (77, 3, 7),
                (81, 3, 7),
                (84, 3, 7),
                (87, 3, 7),
                (91, 1, 1),
            ]),
            // UISETP.GE.U32.AND UP0, UPT, UR5, 0x20, UPT
            word(&[
                (0, 12, 0x88c),
                (24, 8, 5),
                (32, 32, 0x20),
                (76, 3, 6),
                (81, 3, 0),
                (84, 3, 7),
                (87, 3, 7),
                (91, 1, 1),
            ]),
            // @UP0 UMOV UR6, 0x1; USEL UR7, UR6, UR5, UP0
            word(&[(0, 12, 0x882), (12, 4, 0), (16, 8, 6), (32, 32, 1)]),
            word(&[
                (0, 12, 0x287),
                (16, 8, 7),
                (24, 8, 6),
                (32, 8, 5),
                (87, 3, 0),
                (91, 1, 1),
            ]),
            // R2UR UR8, R1; MOV R2, UR7; MOV R3, UR8
            word(&[(0, 12, 0x3c2), (16, 8, 8), (24, 8, 1)]),
            word(&[(0, 12, 0xc02), (16, 8, 2), (32, 6, 7)]),
            word(&[(0, 12, 0xc02), (16, 8, 3), (32, 6, 8)]),
            // ISETP.GT.U32.EX P1, PT, R3, R2, PT, P0 (low result in P0)
            word(&[
                (0, 12, 0x20c),
                (24, 8, 3),
                (32, 8, 2),
                (68, 3, 0),
                (72, 1, 1),
                (76, 3, 4),
                (81, 3, 1),
                (84, 3, 7),
                (87, 3, 7),
            ]),
            // MOV R4, c[0x18][0x0]: bank 24 is never bound and reads zero.
            word(&[(0, 12, 0xa02), (16, 8, 4), (54, 5, 24)]),
            EXIT,
        ];
        let code: Vec<Instruction> = words.iter().map(|&raw| decode(raw)).collect();
        assert!(code[0].uniform && code[1].uniform && code[3].uniform);
        assert!(!code[5].uniform && !code[6].uniform);
        let translated = valid(&words);
        assert!(translated.descriptors.is_empty());
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
