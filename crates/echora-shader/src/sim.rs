//! A SPIR-V simulator for translated compute programs: it runs every
//! invocation of a workgroup on the CPU, over the subset of SPIR-V the
//! translator emits, and reports what would go wrong on a GPU: loops that
//! run away, barriers some invocations miss or that invocations reach at
//! different places (a deadlock), reads and writes of device memory no
//! dumped region covers (a fault), indices past an array, shuffles from
//! lanes that are not running there, and anything SPIR-V leaves undefined
//! that a program uses (a shift by the width or more, a float conversion
//! out of range). It also gives every invocation's final variables, which
//! the tests use as the registers.
//!
//! Invocations run one at a time until each reaches a barrier or a
//! subgroup operation; a subgroup operation completes when every running
//! invocation of its subgroup is at it, a barrier when every running
//! invocation of its scope is. Global memory is the dumped regions of
//! device memory, addressed through physical pointers as the program
//! computes them (its page table is in those regions too).

// The unordered comparisons are the negated ordered ones: true on NaN.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use std::collections::{BTreeMap, HashMap};

use crate::spirv::{glsl, op, storage};

/// Opcodes not in `op`.
const F_CONVERT: u16 = 115;
const VECTOR_SHUFFLE: u16 = 79;
const I_ADD_CARRY: u16 = 149;
const I_SUB_BORROW: u16 = 150;
const U_MUL_EXTENDED: u16 = 151;
const S_MUL_EXTENDED: u16 = 152;

/// Decorations read here.
const DECORATION_ARRAY_STRIDE: u32 = 6;
const DECORATION_BUILT_IN: u32 = 11;
const DECORATION_BINDING: u32 = 33;
const DECORATION_DESCRIPTOR_SET: u32 = 34;
const DECORATION_OFFSET: u32 = 35;

/// Built-ins the simulator provides.
const BUILTIN_NUM_WORKGROUPS: u32 = 24;
const BUILTIN_WORKGROUP_ID: u32 = 26;
const BUILTIN_LOCAL_INVOCATION_ID: u32 = 27;
const BUILTIN_GLOBAL_INVOCATION_ID: u32 = 28;
const BUILTIN_LOCAL_INVOCATION_INDEX: u32 = 29;
const BUILTIN_SUBGROUP_SIZE: u32 = 36;
const BUILTIN_NUM_SUBGROUPS: u32 = 38;
const BUILTIN_SUBGROUP_ID: u32 = 40;
const BUILTIN_SUBGROUP_LOCAL_INVOCATION_ID: u32 = 41;
const BUILTIN_SUBGROUP_EQ_MASK: u32 = 4416;

/// The execution mode LocalSize.
const EXECUTION_LOCAL_SIZE: u32 = 17;

#[derive(Clone, Debug, PartialEq, Eq)]
enum Ty {
    Void,
    Bool,
    Int { width: u32, signed: bool },
    Float { width: u32 },
    Vector { component: u32, count: u32 },
    Array { element: u32, length: u32 },
    RuntimeArray { element: u32 },
    Struct { members: Vec<u32> },
    Pointer { pointee: u32 },
    Other,
}

/// A value: one 64-bit slot per scalar (its bits), flattened.
pub type Value = Vec<u64>;

/// A pointer: slots of a variable, or bytes of device memory.
#[derive(Clone, Copy, Debug)]
enum Pointer {
    Logical {
        variable: u32,
        offset: usize,
        pointee: u32,
    },
    Physical {
        address: u64,
        pointee: u32,
    },
}

#[derive(Debug)]
struct Instruction {
    opcode: u16,
    operands: Vec<u32>,
}

/// What a workgroup runs with.
#[derive(Clone, Debug, Default)]
pub struct Inputs {
    /// The workgroup's id and the dispatch's workgroup counts.
    pub workgroup_id: [u32; 3],
    pub workgroup_count: [u32; 3],
    /// Invocations per subgroup (32 when 0).
    pub subgroup_size: u32,
    /// The push constants' bytes.
    pub push: Vec<u8>,
    /// Uniform buffers' bytes by (descriptor set, binding).
    pub uniforms: HashMap<(u32, u32), Vec<u8>>,
    /// Device memory: regions by their first address.
    pub memory: BTreeMap<u64, Vec<u8>>,
    /// Iterations of one loop by one invocation that count as running
    /// away (1 << 16 when 0).
    pub loop_limit: u32,
}

/// What a workgroup's run left.
#[derive(Debug)]
pub struct Report {
    /// Each invocation's variables, by id.
    pub invocations: Vec<HashMap<u32, Value>>,
    /// The workgroup barriers passed: the instruction's index in the entry
    /// function and the invocations still running.
    pub barriers: Vec<(usize, usize)>,
    /// What went wrong, in order.
    pub problems: Vec<String>,
    /// What is worth knowing but works on the hardware (a barrier met
    /// after whole subgroups returned).
    pub notes: Vec<String>,
    /// Device memory after the run.
    pub memory: BTreeMap<u64, Vec<u8>>,
}

/// A loaded module.
#[derive(Debug)]
pub struct Machine {
    types: HashMap<u32, Ty>,
    /// The result type of each constant.
    result_types: HashMap<u32, u32>,
    constants: HashMap<u32, Value>,
    /// Decorations: built-ins, descriptor sets and bindings by variable;
    /// array strides by type; member offsets by (struct, member).
    builtins: HashMap<u32, u32>,
    sets: HashMap<u32, u32>,
    bindings: HashMap<u32, u32>,
    strides: HashMap<u32, u32>,
    offsets: HashMap<(u32, u32), u32>,
    /// Module-level variables: (id, pointer type, storage class).
    globals: Vec<(u32, u32, u32)>,
    /// The entry function's instructions and its labels' places.
    body: Vec<Instruction>,
    labels: HashMap<u32, usize>,
    glsl: u32,
    /// The workgroup size (LocalSize).
    pub local_size: [u32; 3],
    cooperative: bool,
}

/// One invocation's state.
struct Invocation {
    at: usize,
    values: HashMap<u32, Value>,
    value_types: HashMap<u32, u32>,
    pointers: HashMap<u32, Pointer>,
    /// Its own variables (function, private, input, output).
    variables: HashMap<u32, Value>,
    /// Loop headers passed, by instruction.
    loops: HashMap<usize, u32>,
    done: bool,
    steps: u64,
}

/// What a workgroup shares: its shared, uniform and push-constant
/// variables, device memory and the problems found.
struct Shared {
    variables: HashMap<u32, Value>,
    memory: BTreeMap<u64, Vec<u8>>,
    problems: Vec<String>,
    notes: Vec<String>,
    loop_limit: u32,
}

impl Shared {
    fn problem(&mut self, problem: String) {
        if self.problems.len() < 64 && !self.problems.contains(&problem) {
            self.problems.push(problem);
        }
    }

    /// The region holding `count` bytes at `address`, and the offset there.
    fn region(&mut self, address: u64, count: usize) -> Option<(u64, usize)> {
        let (&base, bytes) = self.memory.range(..=address).next_back()?;
        let offset = (address - base) as usize;
        (offset + count <= bytes.len()).then_some((base, offset))
    }

    fn read(&mut self, address: u64, count: usize) -> Vec<u8> {
        match self.region(address, count) {
            Some((base, offset)) => self.memory[&base][offset..offset + count].to_vec(),
            None => {
                self.problem(format!(
                    "read of {count} bytes at {address:#x}, outside the dumped memory (a fault)"
                ));
                vec![0; count]
            }
        }
    }

    fn write(&mut self, address: u64, bytes: &[u8]) {
        match self.region(address, bytes.len()) {
            Some((base, offset)) => {
                if let Some(region) = self.memory.get_mut(&base) {
                    region[offset..offset + bytes.len()].copy_from_slice(bytes);
                }
            }
            None => self.problem(format!(
                "write of {} bytes at {address:#x}, outside the dumped memory (a fault)",
                bytes.len()
            )),
        }
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exponent = u32::from(bits >> 10 & 0x1f);
    let fraction = u32::from(bits & 0x3ff);
    let value = match (exponent, fraction) {
        (0, 0) => sign,
        (0, _) => {
            // Subnormal: normalise.
            let mut e = 127 - 15 + 1;
            let mut f = fraction;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            sign | (e as u32) << 23 | (f & 0x3ff) << 13
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, _) => sign | 0x7fc0_0000 | fraction << 13,
        _ => sign | (exponent + 127 - 15) << 23 | fraction << 13,
    };
    f32::from_bits(value)
}

/// `value` rounded once to the nearest half (ties to even).
pub(crate) fn f64_to_f16(value: f64) -> u16 {
    let sign: u16 = if value.is_sign_negative() { 0x8000 } else { 0 };
    if value.is_nan() {
        return sign | 0x7e00;
    }
    let magnitude = value.abs();
    // The largest finite half rounds up to infinity from 65520.
    if magnitude >= 65520.0 {
        return sign | 0x7c00;
    }
    // The largest half not above the magnitude (halves order as their
    // bits), then the nearer of it and the next, ties to the even one.
    let (mut low, mut high) = (0u16, 0x7bff);
    while low < high {
        let middle = (low + high).div_ceil(2);
        if f64::from(f16_to_f32(middle)) <= magnitude {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    let below = f64::from(f16_to_f32(low));
    let above = f64::from(f16_to_f32(low + 1));
    let chosen = if magnitude - below < above - magnitude
        || (magnitude - below == above - magnitude && low & 1 == 0)
    {
        low
    } else {
        low + 1
    };
    sign | chosen
}

/// Round to nearest even, as Vulkan's float conversions do by default.
fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 16 & 0x8000) as u16;
    if value.is_nan() {
        return sign | 0x7e00;
    }
    let magnitude = value.abs();
    if magnitude >= 65520.0 {
        return sign | 0x7c00;
    }
    if magnitude < f32::from_bits(0x3880_0000) {
        // Below the smallest normal half: a multiple of 2^-24, rounded.
        let scaled = magnitude * 16_777_216.0;
        let rounded = scaled.round_ties_even() as u16;
        return sign | rounded;
    }
    let exponent = (bits >> 23 & 0xff) as i32 - 127 + 15;
    let fraction = bits & 0x7f_ffff;
    let mut half = (exponent as u32) << 10 | fraction >> 13;
    let rest = fraction & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && half & 1 == 1) {
        half += 1;
    }
    sign | half as u16
}

impl Machine {
    pub fn new(words: &[u32]) -> Self {
        let mut machine = Self {
            types: HashMap::new(),
            result_types: HashMap::new(),
            constants: HashMap::new(),
            builtins: HashMap::new(),
            sets: HashMap::new(),
            bindings: HashMap::new(),
            strides: HashMap::new(),
            offsets: HashMap::new(),
            globals: Vec::new(),
            body: Vec::new(),
            labels: HashMap::new(),
            glsl: 0,
            local_size: [1, 1, 1],
            cooperative: false,
        };
        let mut entry = 0;
        let mut functions: Vec<(u32, Vec<Instruction>)> = Vec::new();
        let mut at = 5;
        while at < words.len() {
            let opcode = (words[at] & 0xffff) as u16;
            let count = ((words[at] >> 16) as usize).max(1);
            let operands = words[at + 1..(at + count).min(words.len())].to_vec();
            at += count;
            match opcode {
                op::EXT_INST_IMPORT => machine.glsl = operands[0],
                op::ENTRY_POINT => entry = operands[1],
                op::EXECUTION_MODE if operands[1] == EXECUTION_LOCAL_SIZE => {
                    machine.local_size = [operands[2], operands[3], operands[4]];
                }
                op::DECORATE => {
                    let (target, value) = (operands[0], operands.get(2).copied().unwrap_or(0));
                    match operands[1] {
                        DECORATION_BUILT_IN => {
                            machine.builtins.insert(target, value);
                        }
                        DECORATION_DESCRIPTOR_SET => {
                            machine.sets.insert(target, value);
                        }
                        DECORATION_BINDING => {
                            machine.bindings.insert(target, value);
                        }
                        DECORATION_ARRAY_STRIDE => {
                            machine.strides.insert(target, value);
                        }
                        _ => {}
                    }
                }
                op::MEMBER_DECORATE if operands[2] == DECORATION_OFFSET => {
                    machine
                        .offsets
                        .insert((operands[0], operands[1]), operands[3]);
                }
                op::TYPE_VOID => {
                    machine.types.insert(operands[0], Ty::Void);
                }
                op::TYPE_BOOL => {
                    machine.types.insert(operands[0], Ty::Bool);
                }
                op::TYPE_INT => {
                    machine.types.insert(
                        operands[0],
                        Ty::Int {
                            width: operands[1],
                            signed: operands[2] != 0,
                        },
                    );
                }
                op::TYPE_FLOAT => {
                    machine
                        .types
                        .insert(operands[0], Ty::Float { width: operands[1] });
                }
                op::TYPE_VECTOR => {
                    machine.types.insert(
                        operands[0],
                        Ty::Vector {
                            component: operands[1],
                            count: operands[2],
                        },
                    );
                }
                op::TYPE_ARRAY => {
                    let length = machine.constants[&operands[2]][0] as u32;
                    machine.types.insert(
                        operands[0],
                        Ty::Array {
                            element: operands[1],
                            length,
                        },
                    );
                }
                op::TYPE_RUNTIME_ARRAY => {
                    machine.types.insert(
                        operands[0],
                        Ty::RuntimeArray {
                            element: operands[1],
                        },
                    );
                }
                op::TYPE_STRUCT => {
                    machine.types.insert(
                        operands[0],
                        Ty::Struct {
                            members: operands[1..].to_vec(),
                        },
                    );
                }
                op::TYPE_POINTER => {
                    machine.types.insert(
                        operands[0],
                        Ty::Pointer {
                            pointee: operands[2],
                        },
                    );
                }
                op::TYPE_COOPERATIVE_MATRIX => {
                    machine.cooperative = true;
                    machine.types.insert(operands[0], Ty::Other);
                }
                op::TYPE_FUNCTION | op::TYPE_IMAGE | op::TYPE_SAMPLER | op::TYPE_SAMPLED_IMAGE => {
                    machine.types.insert(operands[0], Ty::Other);
                }
                op::CONSTANT | op::SPEC_CONSTANT => {
                    let value = operands[2..]
                        .iter()
                        .enumerate()
                        .fold(0u64, |value, (n, &word)| {
                            value | u64::from(word) << (32 * n)
                        });
                    machine.constants.insert(operands[1], vec![value]);
                    machine.result_types.insert(operands[1], operands[0]);
                }
                op::CONSTANT_TRUE | op::CONSTANT_FALSE | op::SPEC_CONSTANT_FALSE => {
                    let value = u64::from(opcode == op::CONSTANT_TRUE);
                    machine.constants.insert(operands[1], vec![value]);
                    machine.result_types.insert(operands[1], operands[0]);
                }
                op::CONSTANT_COMPOSITE => {
                    let value = operands[2..]
                        .iter()
                        .flat_map(|id| machine.constants[id].clone())
                        .collect();
                    machine.constants.insert(operands[1], value);
                    machine.result_types.insert(operands[1], operands[0]);
                }
                op::VARIABLE if functions.is_empty() => {
                    machine
                        .globals
                        .push((operands[1], operands[0], operands[2]));
                }
                op::FUNCTION => functions.push((operands[1], Vec::new())),
                _ => {
                    if let Some((_, body)) = functions.last_mut() {
                        body.push(Instruction { opcode, operands });
                    }
                }
            }
        }
        let (_, body) = functions
            .into_iter()
            .find(|(id, _)| *id == entry)
            .expect("the entry function");
        for (n, instruction) in body.iter().enumerate() {
            if instruction.opcode == op::LABEL {
                machine.labels.insert(instruction.operands[0], n);
            }
        }
        machine.body = body;
        machine
    }

    /// Whether the module has cooperative matrix types (not simulated).
    pub fn uses_cooperative_matrices(&self) -> bool {
        self.cooperative
    }

    fn ty(&self, id: u32) -> Ty {
        self.types.get(&id).cloned().unwrap_or(Ty::Other)
    }

    /// The scalar type of a value of type `ty` (its components').
    fn scalar(&self, ty: u32) -> Ty {
        match self.ty(ty) {
            Ty::Vector { component, .. } => self.ty(component),
            other => other,
        }
    }

    /// The slots a value of type `ty` takes (0 for a runtime array).
    fn size(&self, ty: u32) -> usize {
        match self.ty(ty) {
            Ty::Vector { count, .. } => count as usize,
            Ty::Array { element, length } => self.size(element) * length as usize,
            Ty::RuntimeArray { .. } => 0,
            Ty::Struct { members } => members.iter().map(|&member| self.size(member)).sum(),
            _ => 1,
        }
    }

    /// Bytes of a scalar of type `ty`.
    fn scalar_bytes(&self, ty: u32) -> usize {
        match self.ty(ty) {
            Ty::Int { width, .. } | Ty::Float { width } => width as usize / 8,
            _ => 4,
        }
    }

    /// The byte size of a value of type `ty` laid out in memory.
    fn byte_size(&self, ty: u32) -> usize {
        match self.ty(ty) {
            Ty::Vector { component, count } => self.scalar_bytes(component) * count as usize,
            Ty::Array { element, length } => self.stride(ty, element) * length as usize,
            Ty::Struct { members } => members
                .iter()
                .enumerate()
                .map(|(n, &member)| self.member_offset(ty, n as u32) + self.byte_size(member))
                .max()
                .unwrap_or(0),
            Ty::RuntimeArray { .. } => 0,
            _ => self.scalar_bytes(ty),
        }
    }

    fn stride(&self, array: u32, element: u32) -> usize {
        self.strides
            .get(&array)
            .map_or_else(|| self.byte_size(element), |&stride| stride as usize)
    }

    fn member_offset(&self, ty: u32, member: u32) -> usize {
        self.offsets
            .get(&(ty, member))
            .map_or(0, |&offset| offset as usize)
    }

    /// The slots of a value of type `ty` read from `bytes` at `at` (zeros
    /// past their end).
    fn fill(&self, ty: u32, bytes: &[u8], at: usize, slots: &mut Value) {
        match self.ty(ty) {
            Ty::Vector { component, count } => {
                let size = self.scalar_bytes(component);
                for n in 0..count as usize {
                    self.fill(component, bytes, at + n * size, slots);
                }
            }
            Ty::Array { element, length } => {
                let stride = self.stride(ty, element);
                for n in 0..length as usize {
                    self.fill(element, bytes, at + n * stride, slots);
                }
            }
            Ty::Struct { members } => {
                for (n, &member) in members.iter().enumerate() {
                    self.fill(member, bytes, at + self.member_offset(ty, n as u32), slots);
                }
            }
            Ty::RuntimeArray { .. } => {}
            _ => {
                let size = self.scalar_bytes(ty);
                let mut value = 0u64;
                for n in 0..size {
                    value |= u64::from(*bytes.get(at + n).unwrap_or(&0)) << (8 * n);
                }
                slots.push(value);
            }
        }
    }

    /// A value of type `ty` as bytes in memory (scalars and vectors).
    fn bytes_of(&self, ty: u32, value: &Value) -> Vec<u8> {
        let size = self.scalar_bytes(match self.ty(ty) {
            Ty::Vector { component, .. } => component,
            _ => ty,
        });
        value
            .iter()
            .flat_map(|&slot| slot.to_le_bytes()[..size].to_vec())
            .collect()
    }

    /// Runs the entry point alone with built-in inputs `builtins`: its
    /// variables.
    #[cfg(test)]
    fn run(&self, builtins: &HashMap<u32, Value>) -> HashMap<u32, Value> {
        let builtins = builtins.clone();
        let mut report = self.run_workgroup(1, 1, |_| builtins.clone());
        assert!(report.problems.is_empty(), "{:?}", report.problems);
        report.invocations.remove(0)
    }

    /// Runs `count` invocations of a workgroup in subgroups of `subgroup`,
    /// invocation n with built-ins `builtins(n)`, nothing bound.
    pub fn run_workgroup(
        &self,
        count: usize,
        subgroup: usize,
        builtins: impl Fn(usize) -> HashMap<u32, Value>,
    ) -> Report {
        self.run_group(count, subgroup, builtins, &Inputs::default())
    }

    /// Runs workgroup `inputs.workgroup_id` of a dispatch.
    pub fn simulate(&self, inputs: &Inputs) -> Report {
        let [x, y, z] = self.local_size;
        let count = (x * y * z) as usize;
        let subgroup = if inputs.subgroup_size == 0 {
            32
        } else {
            inputs.subgroup_size as usize
        };
        let id = inputs.workgroup_id;
        let groups = inputs.workgroup_count;
        self.run_group(
            count,
            subgroup,
            |n| {
                let n32 = n as u32;
                let local = [n32 % x, n32 / x % y, n32 / (x * y)];
                let global: Vec<u64> = (0..3)
                    .map(|axis| u64::from(id[axis] * self.local_size[axis] + local[axis]))
                    .collect();
                let lane = (n % subgroup) as u64;
                let mut builtins = HashMap::from([
                    (
                        BUILTIN_LOCAL_INVOCATION_ID,
                        local.iter().map(|&v| u64::from(v)).collect(),
                    ),
                    (BUILTIN_LOCAL_INVOCATION_INDEX, vec![n as u64]),
                    (
                        BUILTIN_WORKGROUP_ID,
                        id.iter().map(|&v| u64::from(v)).collect(),
                    ),
                    (
                        BUILTIN_NUM_WORKGROUPS,
                        groups.iter().map(|&v| u64::from(v)).collect(),
                    ),
                    (BUILTIN_GLOBAL_INVOCATION_ID, global),
                    (BUILTIN_SUBGROUP_SIZE, vec![subgroup as u64]),
                    (BUILTIN_NUM_SUBGROUPS, vec![count.div_ceil(subgroup) as u64]),
                    (BUILTIN_SUBGROUP_ID, vec![(n / subgroup) as u64]),
                    (BUILTIN_SUBGROUP_LOCAL_INVOCATION_ID, vec![lane]),
                ]);
                // The eq, ge, gt, le and lt masks.
                for (k, mask) in [
                    1u128 << lane,
                    !0u128 << lane,
                    !0u128 << lane << 1,
                    !(!0u128 << lane << 1),
                    !(!0u128 << lane),
                ]
                .into_iter()
                .enumerate()
                {
                    let mask = mask
                        & if subgroup >= 128 {
                            !0
                        } else {
                            (1u128 << subgroup) - 1
                        };
                    let words = (0..4).map(|w| (mask >> (32 * w)) as u32 as u64).collect();
                    builtins.insert(BUILTIN_SUBGROUP_EQ_MASK + k as u32, words);
                }
                builtins
            },
            inputs,
        )
    }

    fn run_group(
        &self,
        count: usize,
        subgroup: usize,
        builtins: impl Fn(usize) -> HashMap<u32, Value>,
        inputs: &Inputs,
    ) -> Report {
        let mut shared = Shared {
            variables: HashMap::new(),
            memory: inputs.memory.clone(),
            problems: Vec::new(),
            notes: Vec::new(),
            loop_limit: if inputs.loop_limit == 0 {
                1 << 16
            } else {
                inputs.loop_limit
            },
        };
        // The workgroup's variables.
        for &(variable, pointer, class) in &self.globals {
            let Ty::Pointer { pointee } = self.ty(pointer) else {
                continue;
            };
            let bytes = match class {
                storage::UNIFORM | storage::STORAGE_BUFFER => {
                    let key = (
                        self.sets.get(&variable).copied().unwrap_or(0),
                        self.bindings.get(&variable).copied().unwrap_or(0),
                    );
                    inputs.uniforms.get(&key).cloned().unwrap_or_default()
                }
                storage::PUSH_CONSTANT => inputs.push.clone(),
                storage::WORKGROUP => Vec::new(),
                _ => continue,
            };
            let mut slots = Vec::new();
            self.fill(pointee, &bytes, 0, &mut slots);
            shared.variables.insert(variable, slots);
        }
        let mut invocations: Vec<Invocation> =
            (0..count).map(|n| self.start(&builtins(n))).collect();
        let mut barriers = Vec::new();
        loop {
            for invocation in invocations.iter_mut() {
                if !invocation.done {
                    self.advance(invocation, &mut shared);
                }
            }
            if invocations.iter().all(|invocation| invocation.done) {
                break;
            }
            let mut progress = false;
            // Subgroup operations: every running invocation of the
            // subgroup at the same one.
            for first in (0..count).step_by(subgroup) {
                let members: Vec<usize> = (first..(first + subgroup).min(count))
                    .filter(|&n| !invocations[n].done)
                    .collect();
                let Some(&lead) = members.first() else {
                    continue;
                };
                let at = invocations[lead].at;
                if self.body[at].opcode == op::CONTROL_BARRIER
                    || members.iter().any(|&n| invocations[n].at != at)
                {
                    continue;
                }
                self.subgroup_operation(&mut invocations, first, &members, &mut shared);
                progress = true;
            }
            if progress {
                continue;
            }
            // Barriers: every running invocation (of the subgroup for a
            // subgroup barrier) at the same one.
            let running: Vec<usize> = (0..count).filter(|&n| !invocations[n].done).collect();
            let at = invocations[running[0]].at;
            let is_barrier = self.body[at].opcode == op::CONTROL_BARRIER;
            let scope = if is_barrier {
                let scope_id = self.body[at].operands[0];
                self.constants.get(&scope_id).map_or(2, |value| value[0])
            } else {
                2
            };
            if is_barrier && scope == 3 {
                let members: Vec<usize> = running
                    .iter()
                    .copied()
                    .filter(|&n| n / subgroup == running[0] / subgroup)
                    .collect();
                if members.iter().all(|&n| invocations[n].at == at) {
                    for &n in &members {
                        invocations[n].at += 1;
                    }
                    continue;
                }
            } else if is_barrier && running.iter().all(|&n| invocations[n].at == at) {
                if running.len() < count && shared.notes.len() < 64 {
                    shared.notes.push(format!(
                        "barrier at instruction {at} reached with {} of {count} invocations \
                         still running (the others returned)",
                        running.len()
                    ));
                }
                barriers.push((at, running.len()));
                for &n in &running {
                    invocations[n].at += 1;
                }
                continue;
            }
            let mut places: Vec<(usize, usize)> =
                running.iter().map(|&n| (n, invocations[n].at)).collect();
            places.dedup_by_key(|place| place.1);
            shared.problem(format!(
                "deadlock: running invocations wait at different barriers or subgroup \
                 operations; (first invocation, instruction): {places:?}"
            ));
            break;
        }
        Report {
            invocations: invocations
                .into_iter()
                .map(|invocation| invocation.variables)
                .collect(),
            barriers,
            problems: shared.problems,
            notes: shared.notes,
            memory: shared.memory,
        }
    }

    /// An invocation about to start, with its built-in inputs.
    fn start(&self, builtins: &HashMap<u32, Value>) -> Invocation {
        let mut variables = HashMap::new();
        let mut pointers = HashMap::new();
        for &(variable, pointer, class) in &self.globals {
            let Ty::Pointer { pointee } = self.ty(pointer) else {
                continue;
            };
            pointers.insert(
                variable,
                Pointer::Logical {
                    variable,
                    offset: 0,
                    pointee,
                },
            );
            if matches!(class, storage::INPUT | storage::OUTPUT | storage::PRIVATE) {
                let value = self
                    .builtins
                    .get(&variable)
                    .and_then(|builtin| builtins.get(builtin))
                    .cloned()
                    .unwrap_or_else(|| vec![0; self.size(pointee)]);
                variables.insert(variable, value);
            }
        }
        Invocation {
            at: 0,
            values: self.constants.clone(),
            value_types: self.result_types.clone(),
            pointers,
            variables,
            loops: HashMap::new(),
            done: false,
            steps: 0,
        }
    }

    /// Whether `opcode` needs the other invocations (a barrier, or a
    /// subgroup operation).
    fn collective(opcode: u16) -> bool {
        opcode == op::CONTROL_BARRIER || (333..=366).contains(&opcode)
    }

    /// The subgroup operation `members` (running invocations of the
    /// subgroup starting at `first`) have all reached.
    fn subgroup_operation(
        &self,
        invocations: &mut [Invocation],
        first: usize,
        members: &[usize],
        shared: &mut Shared,
    ) {
        let at = invocations[members[0]].at;
        let instruction = &self.body[at];
        let (opcode, operands) = (instruction.opcode, &instruction.operands);
        let (ty, id) = (operands[0], operands[1]);
        let input = |invocation: &Invocation, n: usize| invocation.values[&operands[n]].clone();
        let all = |n: usize| -> Vec<Value> {
            members.iter().map(|&m| input(&invocations[m], n)).collect()
        };
        let same = |value: Value| vec![value; members.len()];
        let results: Vec<Value> = match opcode {
            op::GROUP_NON_UNIFORM_ALL => same(vec![u64::from(all(3).iter().all(|v| v[0] != 0))]),
            op::GROUP_NON_UNIFORM_ANY => same(vec![u64::from(all(3).iter().any(|v| v[0] != 0))]),
            op::GROUP_NON_UNIFORM_ALL_EQUAL => {
                let values = all(3);
                same(vec![u64::from(values.iter().all(|v| *v == values[0]))])
            }
            op::GROUP_NON_UNIFORM_BALLOT => {
                let mut mask = 0u128;
                for (&m, vote) in members.iter().zip(all(3)) {
                    if vote[0] != 0 {
                        mask |= 1 << (m - first);
                    }
                }
                same((0..4).map(|w| (mask >> (32 * w)) as u32 as u64).collect())
            }
            op::GROUP_NON_UNIFORM_SHUFFLE => members
                .iter()
                .map(|&n| {
                    let lane = input(&invocations[n], 4)[0] as usize;
                    match members.iter().find(|&&m| m == first + lane) {
                        Some(&source) => input(&invocations[source], 3),
                        None => {
                            shared.problem(format!(
                                "instruction {at}: invocation {n} shuffles from lane {lane}, \
                                 which is not running there"
                            ));
                            vec![0; input(&invocations[n], 3).len()]
                        }
                    }
                })
                .collect(),
            op::GROUP_NON_UNIFORM_I_ADD
            | op::GROUP_NON_UNIFORM_U_MIN
            | op::GROUP_NON_UNIFORM_U_MAX
            | op::GROUP_NON_UNIFORM_S_MIN
            | op::GROUP_NON_UNIFORM_S_MAX
            | op::GROUP_NON_UNIFORM_BITWISE_AND
            | op::GROUP_NON_UNIFORM_BITWISE_OR
            | op::GROUP_NON_UNIFORM_BITWISE_XOR => {
                assert_eq!(operands[3], 0, "only reductions are simulated");
                let values: Vec<u32> = all(4).iter().map(|v| v[0] as u32).collect();
                let signed = |v: u32| v as i32;
                let result = values
                    .iter()
                    .copied()
                    .reduce(|x, y| match opcode {
                        op::GROUP_NON_UNIFORM_I_ADD => x.wrapping_add(y),
                        op::GROUP_NON_UNIFORM_U_MIN => x.min(y),
                        op::GROUP_NON_UNIFORM_U_MAX => x.max(y),
                        op::GROUP_NON_UNIFORM_S_MIN => signed(x).min(signed(y)) as u32,
                        op::GROUP_NON_UNIFORM_S_MAX => signed(x).max(signed(y)) as u32,
                        op::GROUP_NON_UNIFORM_BITWISE_AND => x & y,
                        op::GROUP_NON_UNIFORM_BITWISE_OR => x | y,
                        _ => x ^ y,
                    })
                    .unwrap_or(0);
                same(vec![u64::from(result)])
            }
            _ => panic!("the simulator has no subgroup opcode {opcode}"),
        };
        for (&n, result) in members.iter().zip(results) {
            let invocation = &mut invocations[n];
            invocation.values.insert(id, result);
            invocation.value_types.insert(id, ty);
            invocation.at += 1;
        }
    }

    /// The value at `pointer`.
    fn load(&self, invocation: &Invocation, shared: &mut Shared, pointer: Pointer) -> Value {
        match pointer {
            Pointer::Logical {
                variable,
                offset,
                pointee,
            } => {
                let size = self.size(pointee);
                let slots = invocation
                    .variables
                    .get(&variable)
                    .or_else(|| shared.variables.get(&variable))
                    .unwrap_or_else(|| panic!("a load of %{variable}, which holds nothing"));
                match slots.get(offset..offset + size) {
                    Some(slots) => slots.to_vec(),
                    None => {
                        shared.problem(format!(
                            "read past %{variable} ({} slots) at slot {offset}",
                            slots.len()
                        ));
                        vec![0; size]
                    }
                }
            }
            Pointer::Physical { address, pointee } => {
                let bytes = shared.read(address, self.byte_size(pointee));
                let mut slots = Vec::new();
                self.fill(pointee, &bytes, 0, &mut slots);
                slots
            }
        }
    }

    fn store(
        &self,
        invocation: &mut Invocation,
        shared: &mut Shared,
        pointer: Pointer,
        value: &Value,
    ) {
        match pointer {
            Pointer::Logical {
                variable, offset, ..
            } => {
                let slots = match invocation.variables.get_mut(&variable) {
                    Some(slots) => slots,
                    None => shared
                        .variables
                        .get_mut(&variable)
                        .unwrap_or_else(|| panic!("a store to %{variable}, which holds nothing")),
                };
                let length = slots.len();
                match slots.get_mut(offset..offset + value.len()) {
                    Some(target) => target.copy_from_slice(value),
                    None => shared.problem(format!(
                        "write past %{variable} ({length} slots) at slot {offset}"
                    )),
                }
            }
            Pointer::Physical { address, pointee } => {
                let bytes = self.bytes_of(pointee, value);
                shared.write(address, &bytes);
            }
        }
    }

    /// The pointer `base` indexed by `indices` (values).
    fn chain(&self, base: Pointer, indices: &[u64], shared: &mut Shared) -> Pointer {
        let mut pointer = base;
        for &index in indices {
            pointer = match pointer {
                Pointer::Logical {
                    variable,
                    offset,
                    pointee,
                } => {
                    let (step, element, length) = match self.ty(pointee) {
                        Ty::Struct { members } => {
                            let skip: usize = members[..index as usize]
                                .iter()
                                .map(|&member| self.size(member))
                                .sum();
                            (skip, members[index as usize], None)
                        }
                        Ty::Array { element, length } => {
                            (self.size(element) * index as usize, element, Some(length))
                        }
                        Ty::RuntimeArray { element } => {
                            (self.size(element) * index as usize, element, None)
                        }
                        Ty::Vector { component, count } => (index as usize, component, Some(count)),
                        other => panic!("an access chain into {other:?}"),
                    };
                    if length.is_some_and(|length| index >= u64::from(length)) {
                        shared.problem(format!(
                            "index {index} past an array of {} in %{variable}",
                            length.unwrap_or(0)
                        ));
                    }
                    Pointer::Logical {
                        variable,
                        offset: offset + step,
                        pointee: element,
                    }
                }
                Pointer::Physical { address, pointee } => {
                    let (step, element) = match self.ty(pointee) {
                        Ty::Struct { members } => (
                            self.member_offset(pointee, index as u32) as u64,
                            members[index as usize],
                        ),
                        Ty::Array { element, .. } | Ty::RuntimeArray { element } => {
                            (self.stride(pointee, element) as u64 * index, element)
                        }
                        Ty::Vector { component, .. } => {
                            (self.scalar_bytes(component) as u64 * index, component)
                        }
                        other => panic!("an access chain into {other:?}"),
                    };
                    Pointer::Physical {
                        address: address.wrapping_add(step),
                        pointee: element,
                    }
                }
            };
        }
        pointer
    }

    /// Runs `invocation` until it ends or reaches a barrier or subgroup
    /// operation (left at it).
    fn advance(&self, invocation: &mut Invocation, shared: &mut Shared) {
        loop {
            invocation.steps += 1;
            if invocation.steps > 1 << 28 {
                shared.problem("an invocation ran 2^28 instructions".to_string());
                invocation.done = true;
                return;
            }
            let at = invocation.at;
            let instruction = &self.body[at];
            if Self::collective(instruction.opcode) {
                return;
            }
            invocation.at += 1;
            let operands = &instruction.operands;
            let opcode = instruction.opcode;
            match opcode {
                op::LABEL | op::SELECTION_MERGE | op::MEMORY_BARRIER => continue,
                op::LOOP_MERGE => {
                    let passes = invocation.loops.entry(at).or_insert(0);
                    *passes += 1;
                    if *passes > shared.loop_limit {
                        let label = self.body[at - 1].operands[0];
                        shared.problem(format!(
                            "loop with header %{label} (instruction {at}) ran {} times in one \
                             invocation (runaway)",
                            shared.loop_limit
                        ));
                        invocation.done = true;
                        return;
                    }
                    continue;
                }
                op::VARIABLE => {
                    let Ty::Pointer { pointee } = self.ty(operands[0]) else {
                        panic!("a variable of a non-pointer type");
                    };
                    let value = operands.get(3).map_or_else(
                        || vec![0; self.size(pointee)],
                        |id| invocation.values[id].clone(),
                    );
                    invocation.variables.insert(operands[1], value);
                    invocation.pointers.insert(
                        operands[1],
                        Pointer::Logical {
                            variable: operands[1],
                            offset: 0,
                            pointee,
                        },
                    );
                    continue;
                }
                op::BRANCH => {
                    invocation.at = self.labels[&operands[0]];
                    continue;
                }
                op::BRANCH_CONDITIONAL => {
                    let taken = invocation.values[&operands[0]][0] != 0;
                    invocation.at = self.labels[&operands[if taken { 1 } else { 2 }]];
                    continue;
                }
                op::SWITCH => {
                    let selector = invocation.values[&operands[0]][0] as u32;
                    let mut target = operands[1];
                    for pair in operands[2..].chunks(2) {
                        if pair[0] == selector {
                            target = pair[1];
                        }
                    }
                    invocation.at = self.labels[&target];
                    continue;
                }
                op::RETURN => {
                    invocation.done = true;
                    return;
                }
                op::UNREACHABLE => {
                    shared.problem(format!("OpUnreachable reached at instruction {at}"));
                    invocation.done = true;
                    return;
                }
                op::STORE => {
                    let pointer = invocation.pointers[&operands[0]];
                    let value = invocation.values[&operands[1]].clone();
                    self.store(invocation, shared, pointer, &value);
                    continue;
                }
                op::FUNCTION_CALL => panic!("the simulator does not call functions"),
                _ => {}
            }
            // Instructions with a result.
            let (ty, id, args) = (operands[0], operands[1], &operands[2..]);
            let values = &invocation.values;
            let value = |n: usize| values[&args[n]].clone();
            let result: Value = match opcode {
                op::LOAD => {
                    let pointer = invocation.pointers[&args[0]];
                    self.load(invocation, shared, pointer)
                }
                op::ACCESS_CHAIN => {
                    let base = invocation.pointers[&args[0]];
                    let indices: Vec<u64> = args[1..].iter().map(|a| values[a][0]).collect();
                    let pointer = self.chain(base, &indices, shared);
                    invocation.pointers.insert(id, pointer);
                    continue;
                }
                op::CONVERT_U_TO_PTR => {
                    let Ty::Pointer { pointee } = self.ty(ty) else {
                        panic!("ConvertUToPtr to a non-pointer type");
                    };
                    invocation.pointers.insert(
                        id,
                        Pointer::Physical {
                            address: value(0)[0],
                            pointee,
                        },
                    );
                    continue;
                }
                op::ATOMIC_I_ADD
                | op::ATOMIC_U_MIN
                | op::ATOMIC_U_MAX
                | op::ATOMIC_S_MIN
                | op::ATOMIC_S_MAX
                | op::ATOMIC_AND
                | op::ATOMIC_OR
                | op::ATOMIC_XOR
                | op::ATOMIC_EXCHANGE
                | op::ATOMIC_COMPARE_EXCHANGE => {
                    let pointer = invocation.pointers[&args[0]];
                    let old = self.load(invocation, shared, pointer)[0] as u32;
                    let operand = |n: usize| invocation.values[&args[n]][0] as u32;
                    let new = match opcode {
                        op::ATOMIC_I_ADD => old.wrapping_add(operand(3)),
                        op::ATOMIC_U_MIN => old.min(operand(3)),
                        op::ATOMIC_U_MAX => old.max(operand(3)),
                        op::ATOMIC_S_MIN => (old as i32).min(operand(3) as i32) as u32,
                        op::ATOMIC_S_MAX => (old as i32).max(operand(3) as i32) as u32,
                        op::ATOMIC_AND => old & operand(3),
                        op::ATOMIC_OR => old | operand(3),
                        op::ATOMIC_XOR => old ^ operand(3),
                        op::ATOMIC_EXCHANGE => operand(3),
                        _ => {
                            // Value, then comparator, after two semantics.
                            if old == operand(5) { operand(4) } else { old }
                        }
                    };
                    self.store(invocation, shared, pointer, &vec![u64::from(new)]);
                    vec![u64::from(old)]
                }
                // The clock as a uvec2 (or u64): the steps run so far.
                op::READ_CLOCK => match self.ty(ty) {
                    Ty::Vector { .. } => {
                        vec![invocation.steps & 0xffff_ffff, invocation.steps >> 32]
                    }
                    _ => vec![invocation.steps],
                },
                // Components of the two vectors by literal index (0xffffffff:
                // undefined, here 0).
                VECTOR_SHUFFLE => {
                    let both: Value = [value(0), value(1)].concat();
                    args[2..]
                        .iter()
                        .map(|&n| both.get(n as usize).copied().unwrap_or(0))
                        .collect()
                }
                op::BITCAST => self.bitcast(invocation.value_types[&args[0]], ty, &value(0)),
                op::COMPOSITE_CONSTRUCT => args.iter().flat_map(|a| values[a].clone()).collect(),
                op::COMPOSITE_EXTRACT => {
                    // Flat: the slot of the index path (vectors and flat
                    // structs of scalars).
                    vec![value(0)[args[1] as usize]]
                }
                op::EXT_INST => {
                    assert_eq!(args[0], self.glsl, "an unknown extended instruction set");
                    let inputs: Vec<Value> = args[2..].iter().map(|a| values[a].clone()).collect();
                    self.extended(ty, args[1], &inputs)
                }
                // Two results in a struct: the low word or sum, then the high
                // word or carry (or borrow).
                I_ADD_CARRY | I_SUB_BORROW | U_MUL_EXTENDED | S_MUL_EXTENDED => {
                    let (x, y) = (value(0)[0] as u32, value(1)[0] as u32);
                    let (low, high) = match opcode {
                        I_ADD_CARRY => {
                            let (sum, carry) = x.overflowing_add(y);
                            (sum, u32::from(carry))
                        }
                        I_SUB_BORROW => {
                            let (difference, borrow) = x.overflowing_sub(y);
                            (difference, u32::from(borrow))
                        }
                        U_MUL_EXTENDED => {
                            let product = u64::from(x) * u64::from(y);
                            (product as u32, (product >> 32) as u32)
                        }
                        _ => {
                            let product = i64::from(x as i32) * i64::from(y as i32);
                            (product as u32, (product >> 32) as u32)
                        }
                    };
                    vec![u64::from(low), u64::from(high)]
                }
                _ => {
                    if let Some(missing) = args.iter().find(|a| !values.contains_key(a)) {
                        panic!(
                            "the simulator does not run opcode {opcode} (instruction {at}: operand \
                             {missing} is no value)"
                        );
                    }
                    let inputs: Vec<Value> = args.iter().map(|a| values[a].clone()).collect();
                    let operand_type = args
                        .iter()
                        .rev()
                        .find_map(|a| invocation.value_types.get(a).copied())
                        .unwrap_or(ty);
                    self.arithmetic(opcode, ty, operand_type, &inputs)
                }
            };
            self.check(ty, &result);
            invocation.values.insert(id, result);
            invocation.value_types.insert(id, ty);
        }
    }

    /// The bits of `value` (of type `from`) as type `to`.
    fn bitcast(&self, from: u32, to: u32, value: &Value) -> Value {
        let width = |ty: u32| match self.scalar(ty) {
            Ty::Int { width, .. } | Ty::Float { width } => width,
            _ => 32,
        };
        let (from_width, to_width) = (width(from), width(to));
        let mut bits: u128 = 0;
        for (n, &component) in value.iter().enumerate() {
            bits |= u128::from(component) << (n as u32 * from_width);
        }
        let total = value.len() as u32 * from_width;
        (0..total / to_width)
            .map(|n| (bits >> (n * to_width) & ((1u128 << to_width) - 1)) as u64)
            .collect()
    }

    /// A result's components fit its type.
    fn check(&self, ty: u32, value: &Value) {
        let width = match self.scalar(ty) {
            Ty::Bool => 1,
            Ty::Int { width, .. } | Ty::Float { width } => width,
            _ => return,
        };
        for &component in value {
            assert!(
                width == 64 || component >> width == 0,
                "a {width}-bit result with bits beyond it"
            );
        }
    }

    fn arithmetic(&self, opcode: u16, ty: u32, operand: u32, inputs: &[Value]) -> Value {
        let scalar = self.scalar(ty);
        let width = match scalar {
            Ty::Int { width, .. } | Ty::Float { width } => width,
            _ => 32,
        };
        // Half operands: computed exactly in f64, rounded once.
        if matches!(self.scalar(operand), Ty::Float { width: 16 })
            && !matches!(opcode, F_CONVERT | op::SELECT)
        {
            return self.half_arithmetic(opcode, ty, inputs);
        }
        let mask = if width == 64 {
            u64::MAX
        } else {
            (1u64 << width) - 1
        };
        let lanes = inputs.first().map_or(1, Vec::len);
        let mut result = Vec::with_capacity(lanes);
        for lane in 0..lanes {
            let x = |n: usize| inputs[n][lane.min(inputs[n].len() - 1)];
            let f = |n: usize| f32::from_bits(x(n) as u32);
            let i = |n: usize| x(n) as u32 as i32;
            let fb = |v: f32| u64::from(v.to_bits());
            let b = |v: bool| u64::from(v);
            let value = match opcode {
                op::I_ADD => x(0).wrapping_add(x(1)) & mask,
                op::I_SUB => x(0).wrapping_sub(x(1)) & mask,
                op::I_MUL => x(0).wrapping_mul(x(1)) & mask,
                op::S_NEGATE => x(0).wrapping_neg() & mask,
                op::BITWISE_AND => x(0) & x(1),
                op::BITWISE_OR => x(0) | x(1),
                op::BITWISE_XOR => x(0) ^ x(1),
                op::NOT => !x(0) & mask,
                op::SHIFT_LEFT_LOGICAL => {
                    assert!(
                        x(1) < u64::from(width),
                        "a shift by {} of {width} bits",
                        x(1)
                    );
                    (x(0) << x(1)) & mask
                }
                op::SHIFT_RIGHT_LOGICAL => {
                    assert!(
                        x(1) < u64::from(width),
                        "a shift by {} of {width} bits",
                        x(1)
                    );
                    x(0) >> x(1)
                }
                op::SHIFT_RIGHT_ARITHMETIC => {
                    assert!(
                        x(1) < u64::from(width),
                        "a shift by {} of {width} bits",
                        x(1)
                    );
                    let signed = ((x(0) << (64 - width)) as i64) >> (64 - width);
                    ((signed >> x(1)) as u64) & mask
                }
                op::BIT_REVERSE => u64::from((x(0) as u32).reverse_bits()),
                op::BIT_COUNT => u64::from((x(0) as u32).count_ones()),
                op::BIT_FIELD_U_EXTRACT | op::BIT_FIELD_S_EXTRACT => {
                    let (offset, count) = (x(1), x(2));
                    assert!(offset + count <= 32, "a bit field past the word");
                    if count == 0 {
                        0
                    } else {
                        let field = (x(0) >> offset) & ((1u64 << count) - 1);
                        if opcode == op::BIT_FIELD_S_EXTRACT && field >> (count - 1) & 1 == 1 {
                            (field | !((1u64 << count) - 1)) & mask
                        } else {
                            field
                        }
                    }
                }
                201 => {
                    let (offset, count) = (x(2), x(3));
                    assert!(offset + count <= 32, "a bit field past the word");
                    let field = ((1u64 << count) - 1) << offset;
                    (x(0) & !field | (x(1) << offset) & field) & mask
                }
                op::SELECT => {
                    if inputs[0][lane.min(inputs[0].len() - 1)] != 0 {
                        x(1)
                    } else {
                        x(2)
                    }
                }
                op::I_EQUAL | op::LOGICAL_EQUAL => b(x(0) == x(1)),
                op::I_NOT_EQUAL | op::LOGICAL_NOT_EQUAL => b(x(0) != x(1)),
                op::U_LESS_THAN => b(x(0) < x(1)),
                op::U_LESS_THAN_EQUAL => b(x(0) <= x(1)),
                op::U_GREATER_THAN => b(x(0) > x(1)),
                op::U_GREATER_THAN_EQUAL => b(x(0) >= x(1)),
                op::S_LESS_THAN => b(self.signed(x(0)) < self.signed(x(1))),
                op::S_LESS_THAN_EQUAL => b(self.signed(x(0)) <= self.signed(x(1))),
                op::S_GREATER_THAN => b(self.signed(x(0)) > self.signed(x(1))),
                op::S_GREATER_THAN_EQUAL => b(self.signed(x(0)) >= self.signed(x(1))),
                op::LOGICAL_AND => b(x(0) != 0 && x(1) != 0),
                op::LOGICAL_OR => b(x(0) != 0 || x(1) != 0),
                op::LOGICAL_NOT => b(x(0) == 0),
                op::BITCAST => x(0),
                op::U_CONVERT => x(0) & mask,
                op::S_CONVERT => {
                    let signed = (x(0) as u32 as i32) as i64 as u64;
                    signed & mask
                }
                op::F_ADD => fb(f(0) + f(1)),
                op::F_SUB => fb(f(0) - f(1)),
                op::F_MUL => fb(f(0) * f(1)),
                op::F_DIV => fb(f(0) / f(1)),
                op::F_NEGATE => fb(-f(0)),
                op::IS_NAN => b(f(0).is_nan()),
                op::IS_INF => b(f(0).is_infinite()),
                op::F_ORD_EQUAL => b(f(0) == f(1)),
                op::F_ORD_NOT_EQUAL => b(!f(0).is_nan() && !f(1).is_nan() && f(0) != f(1)),
                op::F_ORD_LESS_THAN => b(f(0) < f(1)),
                op::F_ORD_LESS_THAN_EQUAL => b(f(0) <= f(1)),
                op::F_ORD_GREATER_THAN => b(f(0) > f(1)),
                op::F_ORD_GREATER_THAN_EQUAL => b(f(0) >= f(1)),
                op::F_UNORD_EQUAL => b(!(f(0) != f(1))),
                op::F_UNORD_NOT_EQUAL => b(f(0) != f(1)),
                op::F_UNORD_LESS_THAN => b(!(f(0) >= f(1))),
                op::F_UNORD_LESS_THAN_EQUAL => b(!(f(0) > f(1))),
                op::F_UNORD_GREATER_THAN => b(!(f(0) <= f(1))),
                op::F_UNORD_GREATER_THAN_EQUAL => b(!(f(0) < f(1))),
                op::CONVERT_F_TO_U => {
                    let v = f(0);
                    assert!(
                        v.is_finite() && v > -1.0 && v < 4_294_967_296.0,
                        "ConvertFToU of {v}"
                    );
                    u64::from(v as u32)
                }
                op::CONVERT_F_TO_S => {
                    let v = f(0);
                    assert!(
                        v.is_finite() && v > -2_147_483_649.0 && v < 2_147_483_648.0,
                        "ConvertFToS of {v}"
                    );
                    u64::from(v as i32 as u32)
                }
                op::CONVERT_U_TO_F => fb(x(0) as u32 as f32),
                op::CONVERT_S_TO_F => fb(i(0) as f32),
                F_CONVERT => match width {
                    16 => u64::from(f32_to_f16(f(0))),
                    _ => fb(f16_to_f32(x(0) as u16)),
                },
                _ => panic!("the interpreter has no opcode {opcode}"),
            };
            result.push(value);
        }
        if matches!(opcode, op::SELECT) && inputs[0].len() == 1 {
            // A scalar condition picks whole vectors.
            return if inputs[0][0] != 0 {
                inputs[1].clone()
            } else {
                inputs[2].clone()
            };
        }
        result
    }

    fn half_arithmetic(&self, opcode: u16, ty: u32, inputs: &[Value]) -> Value {
        let lanes = inputs[0].len();
        let half_result = matches!(self.scalar(ty), Ty::Float { width: 16 });
        (0..lanes)
            .map(|lane| {
                let h = |n: usize| f64::from(f16_to_f32(inputs[n][lane] as u16));
                let b = |v: bool| u64::from(v);
                let r = |v: f64| u64::from(f64_to_f16(v));
                let value = match opcode {
                    op::F_ADD => r(h(0) + h(1)),
                    op::F_SUB => r(h(0) - h(1)),
                    op::F_MUL => r(h(0) * h(1)),
                    op::F_NEGATE => r(-h(0)),
                    op::IS_NAN => b(h(0).is_nan()),
                    op::IS_INF => b(h(0).is_infinite()),
                    op::F_ORD_EQUAL => b(h(0) == h(1)),
                    op::F_ORD_LESS_THAN => b(h(0) < h(1)),
                    op::F_ORD_LESS_THAN_EQUAL => b(h(0) <= h(1)),
                    op::F_ORD_GREATER_THAN => b(h(0) > h(1)),
                    op::F_ORD_GREATER_THAN_EQUAL => b(h(0) >= h(1)),
                    op::F_ORD_NOT_EQUAL => b(!h(0).is_nan() && !h(1).is_nan() && h(0) != h(1)),
                    op::F_UNORD_NOT_EQUAL => b(h(0) != h(1)),
                    op::F_UNORD_LESS_THAN => b(!(h(0) >= h(1))),
                    op::F_UNORD_GREATER_THAN => b(!(h(0) <= h(1))),
                    op::F_UNORD_EQUAL => b(!(h(0) != h(1))),
                    op::F_UNORD_LESS_THAN_EQUAL => b(!(h(0) > h(1))),
                    op::F_UNORD_GREATER_THAN_EQUAL => b(!(h(0) < h(1))),
                    op::SELECT => unreachable!("handled with whole values"),
                    _ => panic!("the interpreter has no half opcode {opcode}"),
                };
                let _ = half_result;
                value
            })
            .collect()
    }

    fn signed(&self, value: u64) -> i64 {
        value as u32 as i32 as i64
    }

    fn extended(&self, ty: u32, instruction: u32, inputs: &[Value]) -> Value {
        let lanes = inputs[0].len();
        let half = matches!(self.scalar(ty), Ty::Float { width: 16 });
        let mut result = Vec::new();
        if instruction == glsl::UNPACK_HALF_2X16 {
            let word = inputs[0][0] as u32;
            return vec![
                u64::from(f16_to_f32(word as u16).to_bits()),
                u64::from(f16_to_f32((word >> 16) as u16).to_bits()),
            ];
        }
        if instruction == glsl::PACK_HALF_2X16 {
            let low = f32_to_f16(f32::from_bits(inputs[0][0] as u32));
            let high = f32_to_f16(f32::from_bits(inputs[0][1] as u32));
            return vec![u64::from(low) | u64::from(high) << 16];
        }
        if half {
            return (0..lanes)
                .map(|lane| {
                    let h = |n: usize| f64::from(f16_to_f32(inputs[n][lane] as u16));
                    let r = |v: f64| u64::from(f64_to_f16(v));
                    match instruction {
                        // Fused: the exact a * b + c, rounded once.
                        glsl::FMA => r(h(0) * h(1) + h(2)),
                        glsl::F_ABS => r(h(0).abs()),
                        glsl::N_MIN => r(h(0).min(h(1))),
                        glsl::N_MAX => r(h(0).max(h(1))),
                        glsl::F_CLAMP | glsl::F_MIN | glsl::F_MAX => {
                            assert!(
                                (0..inputs.len()).all(|n| !h(n).is_nan()),
                                "FMin/FMax/FClamp of NaN (undefined)"
                            );
                            match instruction {
                                glsl::F_MIN => r(h(0).min(h(1))),
                                glsl::F_MAX => r(h(0).max(h(1))),
                                _ => r(h(0).max(h(1)).min(h(2))),
                            }
                        }
                        _ => panic!(
                            "the interpreter has no half GLSL.std.450 instruction {instruction}"
                        ),
                    }
                })
                .collect();
        }
        for lane in 0..lanes {
            let x = |n: usize| inputs[n][lane.min(inputs[n].len() - 1)] as u32;
            let f = |n: usize| f32::from_bits(x(n));
            let fb = |v: f32| u64::from(v.to_bits());
            let value = match instruction {
                glsl::ROUND_EVEN => fb(f(0).round_ties_even()),
                glsl::TRUNC => fb(f(0).trunc()),
                glsl::FLOOR => fb(f(0).floor()),
                glsl::CEIL => fb(f(0).ceil()),
                glsl::F_ABS => fb(f(0).abs()),
                glsl::S_ABS => u64::from((x(0) as i32).wrapping_abs() as u32),
                glsl::FMA => fb(f(0).mul_add(f(1), f(2))),
                glsl::SQRT => fb(f(0).sqrt()),
                glsl::INVERSE_SQRT => fb(1.0 / f(0).sqrt()),
                glsl::EXP2 => fb(f(0).exp2()),
                glsl::SIN => fb(f(0).sin()),
                glsl::COS => fb(f(0).cos()),
                glsl::TANH => fb(f(0).tanh()),
                glsl::LOG2 => fb(f(0).log2()),
                glsl::U_MIN => u64::from(x(0).min(x(1))),
                glsl::U_MAX => u64::from(x(0).max(x(1))),
                glsl::S_MIN => u64::from((x(0) as i32).min(x(1) as i32) as u32),
                glsl::S_MAX => u64::from((x(0) as i32).max(x(1) as i32) as u32),
                glsl::F_MIN | glsl::F_MAX | glsl::F_CLAMP => {
                    assert!(
                        (0..inputs.len()).all(|n| !f(n).is_nan()),
                        "FMin/FMax/FClamp of NaN (undefined)"
                    );
                    match instruction {
                        glsl::F_MIN => fb(f(0).min(f(1))),
                        glsl::F_MAX => fb(f(0).max(f(1))),
                        _ => fb(f(0).max(f(1)).min(f(2))),
                    }
                }
                glsl::N_MIN => fb(f(0).min(f(1))),
                glsl::N_MAX => fb(f(0).max(f(1))),
                glsl::FIND_U_MSB => u64::from(if x(0) == 0 {
                    u32::MAX
                } else {
                    31 - x(0).leading_zeros()
                }),
                glsl::FIND_S_MSB => {
                    let v = x(0) as i32;
                    let v = if v < 0 { !v } else { v } as u32;
                    u64::from(if v == 0 {
                        u32::MAX
                    } else {
                        31 - v.leading_zeros()
                    })
                }
                _ => panic!("the simulator does not run GLSL.std.450 instruction {instruction}"),
            };
            result.push(value);
        }
        result
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::compute::{Layout, translate_with_registers};
    use crate::decode::decode;

    const EXIT: u128 = 0x94d | 7 << 12;

    /// A word with `fields` (bit offset, width, value) set and a PT guard.
    fn word(fields: &[(u32, u32, u128)]) -> u128 {
        let mut raw = 7u128 << 12;
        for &(offset, _, value) in fields {
            raw |= value << offset;
        }
        raw
    }

    /// MOV Rd, value.
    fn mov(d: u128, value: u32) -> u128 {
        word(&[
            (0, 12, 0x802),
            (16, 8, d),
            (32, 32, u128::from(value)),
            (72, 4, 0xf),
        ])
    }

    /// Runs `words` (after MOVs of `inputs` into R0..): the general
    /// registers' and predicates' final values.
    fn run(inputs: &[u32], words: &[u128]) -> (HashMap<u8, u32>, HashMap<u8, bool>) {
        let mut program: Vec<u128> = inputs
            .iter()
            .enumerate()
            .map(|(n, &value)| mov(n as u128, value))
            .collect();
        program.extend_from_slice(words);
        program.push(EXIT);
        let code: Vec<_> = program.iter().map(|&raw| decode(raw)).collect();
        let (translated, registers, predicates) =
            translate_with_registers(&code, &Layout::workgroup([1, 1, 1])).unwrap();
        let variables = Machine::new(&translated.spirv).run(&HashMap::new());
        let registers = registers
            .iter()
            .map(|(&n, variable)| (n, variables[variable][0] as u32))
            .collect();
        let predicates = predicates
            .iter()
            .map(|(&n, variable)| (n, variables[variable][0] != 0))
            .collect();
        (registers, predicates)
    }

    /// Deterministic test values: edge cases, then a linear congruential
    /// sequence.
    fn values(count: usize) -> Vec<u32> {
        let mut list = vec![0, 1, 2, 0x7fff_ffff, 0x8000_0000, 0xffff_ffff, 0x1234_5678];
        let mut state = 0x9e37_79b9u32;
        while list.len() < count {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            list.push(state);
        }
        list
    }

    #[test]
    fn integer_adds_multiplies_and_logic_match_the_hardware() {
        let list = values(24);
        for window in list.windows(3) {
            let (a, b, c) = (window[0], window[1], window[2]);
            // IADD3 R3, R0, -R1, R2
            let (r, _) = run(
                &[a, b, c],
                &[word(&[
                    (0, 12, 0x210),
                    (16, 8, 3),
                    (24, 8, 0),
                    (32, 8, 1),
                    (63, 1, 1),
                    (64, 8, 2),
                    (77, 3, 7),
                    (81, 3, 7),
                    (84, 3, 7),
                    (87, 3, 7),
                ])],
            );
            assert_eq!(
                r[&3],
                a.wrapping_sub(b).wrapping_add(c),
                "IADD3 {a:#x} {b:#x} {c:#x}"
            );
            // IMAD R3, R0, R1, R2
            let (r, _) = run(
                &[a, b, c],
                &[word(&[
                    (0, 12, 0x224),
                    (16, 8, 3),
                    (24, 8, 0),
                    (32, 8, 1),
                    (64, 8, 2),
                ])],
            );
            assert_eq!(r[&3], a.wrapping_mul(b).wrapping_add(c), "IMAD");
            // LOP3.LUT R3, R0, R1, R2 for some tables (PT result)
            for lut in [0x96u32, 0xe8, 0xca, 0x3c, 0x80, 0xfe, 0x01, 0x5a] {
                let (r, _) = run(
                    &[a, b, c],
                    &[word(&[
                        (0, 12, 0x212),
                        (16, 8, 3),
                        (24, 8, 0),
                        (32, 8, 1),
                        (64, 8, 2),
                        (72, 8, u128::from(lut)),
                        (81, 3, 7),
                    ])],
                );
                let expected = (0..32).fold(0u32, |out, k| {
                    let index = (a >> k & 1) << 2 | (b >> k & 1) << 1 | (c >> k & 1);
                    out | (lut >> index & 1) << k
                });
                assert_eq!(r[&3], expected, "LOP3 {lut:#x}");
            }
            // ISETP.{LT,GE}.{U32,S32}.AND P0, PT, R0, R1, PT
            for (compare, signed) in [(1u128, false), (6, false), (1, true), (6, true)] {
                let (_, p) = run(
                    &[a, b],
                    &[word(&[
                        (0, 12, 0x20c),
                        (24, 8, 0),
                        (32, 8, 1),
                        (73, 1, u128::from(signed)),
                        (76, 3, compare),
                        (81, 3, 0),
                        (84, 3, 7),
                        (87, 3, 7),
                    ])],
                );
                let less = if signed {
                    (a as i32) < (b as i32)
                } else {
                    a < b
                };
                let expected = if compare == 1 { less } else { !less };
                assert_eq!(p[&0], expected, "ISETP {compare} {signed} {a:#x} {b:#x}");
            }
        }
    }

    #[test]
    fn float_adds_multiplies_and_fused_multiply_adds_round_once() {
        let floats = [
            1.0f32,
            -2.5,
            3.0e-39,
            1.0e30,
            0.1,
            -0.0,
            65504.0,
            1.0 + f32::EPSILON,
        ];
        for &a in &floats {
            for &b in &floats {
                for &c in &floats[..3] {
                    let bits = [a.to_bits(), b.to_bits(), c.to_bits()];
                    // FFMA R3, R0, R1, R2; FADD R4, R0, R1; FMUL R5, R0, R1
                    let (r, _) = run(
                        &bits,
                        &[
                            word(&[
                                (0, 12, 0x223),
                                (16, 8, 3),
                                (24, 8, 0),
                                (32, 8, 1),
                                (64, 8, 2),
                            ]),
                            word(&[(0, 12, 0x221), (16, 8, 4), (24, 8, 0), (32, 8, 1)]),
                            word(&[
                                (0, 12, 0x220),
                                (16, 8, 5),
                                (24, 8, 0),
                                (32, 8, 1),
                                (84, 3, 4),
                            ]),
                        ],
                    );
                    // The hardware flushes no denormals without .FTZ.
                    assert_eq!(r[&3], a.mul_add(b, c).to_bits(), "FFMA {a} {b} {c}");
                    assert_eq!(r[&4], (a + b).to_bits(), "FADD {a} {b}");
                    assert_eq!(r[&5], (a * b).to_bits(), "FMUL {a} {b}");
                }
            }
        }
    }

    #[test]
    fn half_fused_multiply_adds_round_once_and_relu_clamps() {
        let halves: Vec<u16> = vec![
            0x3c00, 0xbc00, 0x3555, 0x0001, 0x8001, 0x7bff, 0x4000, 0xc400, 0x0000, 0x8000, 0x3e00,
            0x2e66, 0xb800, 0x5640,
        ];
        let reference = |a: u16, b: u16, c: u16, relu: bool| {
            let h = |x: u16| f64::from(f16_to_f32(x));
            let result = f64_to_f16(h(a) * h(b) + h(c));
            if !relu {
                result
            } else if result & 0x7fff > 0x7c00 {
                0x7fff
            } else if result & 0x8000 != 0 {
                0
            } else {
                result
            }
        };
        for (n, &a0) in halves.iter().enumerate() {
            for (m, &b0) in halves.iter().enumerate() {
                let (a1, b1) = (
                    halves[(n + 3) % halves.len()],
                    halves[(m + 5) % halves.len()],
                );
                let (c0, c1) = (
                    halves[(n + m) % halves.len()],
                    halves[(n * m + 1) % halves.len()],
                );
                let pack = |low: u16, high: u16| u32::from(low) | u32::from(high) << 16;
                for relu in [false, true] {
                    // HFMA2(.RELU) R3, R0, R1, R2
                    let mut fields = vec![
                        (0, 12, 0x231),
                        (16, 8, 3),
                        (24, 8, 0),
                        (32, 8, 1),
                        (64, 8, 2),
                    ];
                    if relu {
                        fields.extend([(79, 1, 1), (87, 3, 7)]);
                    }
                    let (r, _) = run(
                        &[pack(a0, a1), pack(b0, b1), pack(c0, c1)],
                        &[word(&fields)],
                    );
                    let expected = pack(reference(a0, b0, c0, relu), reference(a1, b1, c1, relu));
                    assert_eq!(
                        r[&3], expected,
                        "HFMA2 relu {relu}: {a0:#x}*{b0:#x}+{c0:#x}, {a1:#x}*{b1:#x}+{c1:#x}"
                    );
                }
            }
        }
    }

    /// BRA from instruction `from` to `to` (indices in the whole program,
    /// MOVs included) under `guard` (index, plus 8 when negated).
    fn bra(from: usize, to: usize, guard: u128) -> u128 {
        let offset = (to as i64 - from as i64 - 1) * 16;
        let field = (offset as u64 & ((1 << 50) - 1)) as u128;
        let mut raw = word(&[(0, 12, 0x947), (32, 50, field), (87, 3, 7)]);
        raw &= !(0xf << 12);
        raw | guard << 12
    }

    /// IADD3 Rd, Ra, imm, RZ.
    fn add(d: u128, a: u128, imm: u32) -> u128 {
        word(&[
            (0, 12, 0x810),
            (16, 8, d),
            (24, 8, a),
            (32, 32, u128::from(imm)),
            (64, 8, 255),
            (77, 3, 7),
            (81, 3, 7),
            (84, 3, 7),
            (87, 3, 7),
        ])
    }

    /// ISETP.LT.U32.AND Pp, PT, Ra, Rb, PT.
    fn less(p: u128, a: u128, b: u128) -> u128 {
        word(&[
            (0, 12, 0x20c),
            (24, 8, a),
            (32, 8, b),
            (76, 3, 1),
            (81, 3, p),
            (84, 3, 7),
            (87, 3, 7),
        ])
    }

    const BAR: u128 = 0xb1d | 7 << 12;

    #[test]
    fn loops_run_their_iterations_and_stop() {
        // R0 = 0, R1 = 10 (the MOVs are instructions 0 and 1).
        // 2: IADD3 R0, R0, 1; 3: BAR; 4: ISETP.LT P0, R0, R1; 5: @P0 BRA 2
        let (r, _) = run(&[0, 10], &[add(0, 0, 1), BAR, less(0, 0, 1), bra(5, 2, 0)]);
        assert_eq!(r[&0], 10);
        // An EXIT inside the loop: R2 counts, leaving at R2 == 3.
        // 3: IADD3 R2, R2, 1; 4: ISETP.LT P1, R2, R1 (R1 = 3);
        // 5: @!P1 EXIT; 6: IADD3 R0, R0, 1; 7: BRA 3
        let exit = |guard: u128| (0x94d | 7 << 12) & !(0xf << 12) | guard << 12;
        let (r, _) = run(
            &[0, 3, 0],
            &[
                add(2, 2, 1),
                less(1, 2, 1),
                exit(1 | 8),
                add(0, 0, 1),
                bra(7, 3, 7),
            ],
        );
        assert_eq!((r[&2], r[&0]), (3, 2));
        // Nested: the inner loop 4 times for each of 3 outer iterations.
        // R0 outer, R1 inner, R2 total; R3 = 3, R4 = 4.
        // 5: MOV R1, 0; 6: IADD3 R1, R1, 1; 7: IADD3 R2, R2, 1;
        // 8: ISETP.LT P0, R1, R4; 9: @P0 BRA 6; 10: IADD3 R0, R0, 1;
        // 11: ISETP.LT P1, R0, R3; 12: @P1 BRA 5
        let (r, _) = run(
            &[0, 0, 0, 3, 4],
            &[
                mov(1, 0),
                add(1, 1, 1),
                add(2, 2, 1),
                less(0, 1, 4),
                bra(9, 6, 0),
                add(0, 0, 1),
                less(1, 0, 3),
                bra(12, 5, 1),
            ],
        );
        assert_eq!((r[&0], r[&2]), (3, 12));
    }

    /// Each invocation's general registers, the barriers passed and the
    /// problems found.
    type GroupRun = (Vec<HashMap<u8, u32>>, Vec<(usize, usize)>, Vec<String>);

    /// Runs `words` (after MOVs of `inputs` into R0..) as a workgroup of
    /// `count` invocations in subgroups of 32 (invocation n has local id
    /// (n, 0, 0)): each invocation's general registers, the barriers passed
    /// and the problems found.
    fn run_group(count: usize, inputs: &[u32], words: &[u128]) -> GroupRun {
        let mut program: Vec<u128> = inputs
            .iter()
            .enumerate()
            .map(|(n, &value)| mov(n as u128, value))
            .collect();
        program.extend_from_slice(words);
        program.push(EXIT);
        let code: Vec<_> = program.iter().map(|&raw| decode(raw)).collect();
        let layout = Layout::workgroup([count as u32, 1, 1]);
        let (translated, registers, _) = translate_with_registers(&code, &layout).unwrap();
        let machine = Machine::new(&translated.spirv);
        let group = machine.run_workgroup(count, 32, |n| {
            HashMap::from([(27, vec![n as u64, 0, 0]), (41, vec![(n % 32) as u64])])
        });
        let registers = group
            .invocations
            .iter()
            .map(|variables| {
                registers
                    .iter()
                    .map(|(&n, variable)| (n, variables[variable][0] as u32))
                    .collect()
            })
            .collect();
        (registers, group.barriers, group.problems)
    }

    #[test]
    fn invocations_that_exit_early_still_meet_the_barriers() {
        // R1 = the threshold; S2R R0, SR_TID.X; ISETP.GE.U32 P0, R0, R1;
        // @P0 EXIT; IADD3 R2, R2, 1; BAR; IADD3 R2, R2, 1; BAR
        let s2r = word(&[(0, 12, 0x919), (16, 8, 0), (72, 8, 0x21)]);
        let at_least = word(&[
            (0, 12, 0x20c),
            (24, 8, 0),
            (32, 8, 1),
            (76, 3, 6),
            (81, 3, 0),
            (84, 3, 7),
            (87, 3, 7),
        ]);
        let exit = (0x94d | 7 << 12) & !(0xf << 12);
        let body = [s2r, at_least, exit, add(2, 2, 1), BAR, add(2, 2, 1), BAR];
        // Invocations 40-63 exit: the second subgroup goes on partly, its
        // exited invocations reaching the barriers and doing nothing else.
        let (registers, barriers, problems) = run_group(64, &[0, 40, 0], &body);
        assert!(problems.is_empty(), "{problems:?}");
        assert_eq!(barriers.len(), 2);
        assert!(barriers.iter().all(|&(_, running)| running == 64));
        for (n, registers) in registers.iter().enumerate() {
            assert_eq!(registers[&2], if n < 40 { 2 } else { 0 }, "invocation {n}");
        }
        // Invocations 32-63 exit: the second subgroup has all exited and
        // returns; the first meets the barriers alone.
        let (registers, barriers, problems) = run_group(64, &[0, 32, 0], &body);
        assert!(problems.is_empty(), "{problems:?}");
        assert!(barriers.iter().all(|&(_, running)| running == 32));
        assert_eq!(registers[31][&2], 2);
        assert_eq!(registers[32][&2], 0);
    }

    #[test]
    fn workgroups_share_memory_across_barriers_and_read_global_memory_through_the_page_table() {
        use echora_shader_abi::{PAGE_SHIFT, PUSH_PAGE_TABLE_OFFSET};
        let s2r = word(&[(0, 12, 0x919), (16, 8, 0), (72, 8, 0x21)]);
        // IMAD R1, R0, 4, RZ (a byte offset); IADD3 R4, R1, 0x10000, RZ
        let scale = word(&[
            (0, 12, 0x824),
            (16, 8, 1),
            (24, 8, 0),
            (32, 32, 4),
            (64, 8, 255),
        ]);
        let address = add(4, 1, 0x10000);
        // LDG.E R2, [R4.64]; IADD3 R2, R2, 1; STS [R1], R2; BAR
        let load = word(&[
            (0, 12, 0x381),
            (16, 8, 2),
            (24, 8, 4),
            (72, 1, 1),
            (73, 3, 4),
        ]);
        let store_shared = word(&[(0, 12, 0x388), (24, 8, 1), (32, 8, 2), (73, 3, 4)]);
        // LOP3.LUT R6, R1, 0x80, RZ, 0x3c (xor: the other subgroup's
        // word); LDS R3, [R6]; STG.E [R4.64], R3
        let other = word(&[
            (0, 12, 0x812),
            (16, 8, 6),
            (24, 8, 1),
            (32, 32, 0x80),
            (64, 8, 255),
            (72, 8, 0x3c),
            (81, 3, 7),
        ]);
        let load_shared = word(&[(0, 12, 0x984), (16, 8, 3), (24, 8, 6), (73, 3, 4)]);
        let store = word(&[
            (0, 12, 0x386),
            (24, 8, 4),
            (32, 8, 3),
            (72, 1, 1),
            (73, 3, 4),
        ]);
        let program = [
            mov(5, 0),
            s2r,
            scale,
            address,
            load,
            add(2, 2, 1),
            store_shared,
            BAR,
            other,
            load_shared,
            store,
            EXIT,
        ];
        let code: Vec<_> = program.iter().map(|&raw| decode(raw)).collect();
        let layout = Layout {
            shared_bytes: Some(256),
            ..Layout::workgroup([64, 1, 1])
        };
        let translated = crate::compute::translate(&code, &layout).unwrap();
        // Guest page 0x10 at device 0x3000_0000; the table at 0x2000_0000.
        let (table, page) = (0x2000_0000u64, 0x3000_0000u64);
        let words: Vec<u8> = (0..1024u32).flat_map(u32::to_le_bytes).collect();
        let mut push = vec![0u8; 128];
        let at = PUSH_PAGE_TABLE_OFFSET as usize;
        push[at..at + 8].copy_from_slice(&table.to_le_bytes());
        push[at + 8..at + 12].copy_from_slice(&(0x10000u32 >> PAGE_SHIFT).to_le_bytes());
        push[at + 12..at + 16].copy_from_slice(&1u32.to_le_bytes());
        let inputs = Inputs {
            workgroup_count: [1, 1, 1],
            push,
            memory: BTreeMap::from([(table, page.to_le_bytes().to_vec()), (page, words)]),
            ..Inputs::default()
        };
        let machine = Machine::new(&translated.spirv);
        assert_eq!(machine.local_size, [64, 1, 1]);
        let report = machine.simulate(&inputs);
        assert!(report.problems.is_empty(), "{:?}", report.problems);
        assert_eq!(report.barriers.len(), 1);
        let memory = &report.memory[&page];
        for n in 0..64usize {
            let word = u32::from_le_bytes(memory[4 * n..4 * n + 4].try_into().unwrap());
            assert_eq!(word, (n as u32 ^ 32) + 1, "invocation {n}");
        }
        // Outside the dumped memory: a fault, reported.
        let mut missing = inputs.clone();
        missing.memory.remove(&page);
        let report = machine.simulate(&missing);
        assert!(
            report
                .problems
                .iter()
                .any(|problem| problem.contains("outside the dumped memory"))
        );
    }

    #[test]
    fn runaway_loops_are_reported() {
        // 1: IADD3 R0, R0, 1; 2: BRA 1 (forever)
        let code: Vec<_> = [mov(0, 0), add(0, 0, 1), bra(2, 1, 7), EXIT]
            .iter()
            .map(|&raw| decode(raw))
            .collect();
        let translated = crate::compute::translate(&code, &Layout::workgroup([32, 1, 1])).unwrap();
        let report = Machine::new(&translated.spirv).simulate(&Inputs {
            loop_limit: 100,
            ..Inputs::default()
        });
        assert!(
            report
                .problems
                .iter()
                .any(|problem| problem.contains("runaway")),
            "{:?}",
            report.problems
        );
    }
}
