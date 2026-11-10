//! A SPIR-V interpreter for tests: it runs a translated program's entry
//! point for one invocation, over the scalar and vector arithmetic the
//! translator emits for ALU instructions, and reports its function
//! variables' final values (the registers). Anything outside that subset,
//! and anything SPIR-V leaves undefined that the program then uses (a
//! shift by the width or more, a float conversion out of range), panics,
//! so a test sees it rather than a guessed value.

// The unordered comparisons are the negated ordered ones: true on NaN.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use std::collections::HashMap;

use crate::spirv::{glsl, op};

/// OpFConvert, OpQuantizeToF16 are not in `op`.
const F_CONVERT: u16 = 115;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ty {
    Void,
    Bool,
    Int { width: u32, signed: bool },
    Float { width: u32 },
    Vector { component: u32, count: u32 },
    Pointer { pointee: u32 },
    Other,
}

/// A value: one 64-bit slot per component (bits of the component's type).
type Value = Vec<u64>;

/// A pointer: a variable and, for a component of a vector variable, its
/// index.
#[derive(Clone, Copy, Debug)]
struct Pointer {
    variable: u32,
    component: Option<usize>,
}

struct Instruction {
    opcode: u16,
    operands: Vec<u32>,
}

pub(crate) struct Machine {
    types: HashMap<u32, Ty>,
    /// The result type of each id that has one.
    result_types: HashMap<u32, u32>,
    constants: HashMap<u32, Value>,
    /// Built-in input variables by id: their decoration, for `inputs`.
    builtins: HashMap<u32, u32>,
    /// The entry function's instructions.
    body: Vec<Instruction>,
    labels: HashMap<u32, usize>,
    glsl: u32,
}

/// What a run left: the entry function's variables, by id.
pub(crate) struct Finished {
    pub(crate) variables: HashMap<u32, Value>,
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
    pub(crate) fn new(words: &[u32]) -> Self {
        let mut machine = Self {
            types: HashMap::new(),
            result_types: HashMap::new(),
            constants: HashMap::new(),
            builtins: HashMap::new(),
            body: Vec::new(),
            labels: HashMap::new(),
            glsl: 0,
        };
        let mut entry = 0;
        let mut functions: Vec<(u32, Vec<Instruction>)> = Vec::new();
        let mut at = 5;
        while at < words.len() {
            let opcode = (words[at] & 0xffff) as u16;
            let count = (words[at] >> 16) as usize;
            let operands = words[at + 1..at + count].to_vec();
            at += count;
            match opcode {
                op::EXT_INST_IMPORT => machine.glsl = operands[0],
                op::ENTRY_POINT => entry = operands[1],
                op::DECORATE if operands[1] == 11 => {
                    machine.builtins.insert(operands[0], operands[2]);
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
                op::TYPE_POINTER => {
                    machine.types.insert(
                        operands[0],
                        Ty::Pointer {
                            pointee: operands[2],
                        },
                    );
                }
                op::TYPE_ARRAY
                | op::TYPE_RUNTIME_ARRAY
                | op::TYPE_STRUCT
                | op::TYPE_FUNCTION
                | op::TYPE_IMAGE
                | op::TYPE_SAMPLER
                | op::TYPE_SAMPLED_IMAGE
                | op::TYPE_COOPERATIVE_MATRIX => {
                    machine.types.insert(operands[0], Ty::Other);
                }
                op::CONSTANT | op::SPEC_CONSTANT => {
                    let words = &operands[2..];
                    let value = words.iter().enumerate().fold(0u64, |value, (n, &word)| {
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

    fn ty(&self, id: u32) -> Ty {
        *self.types.get(&id).unwrap_or(&Ty::Other)
    }

    /// The scalar type of a value of type `ty` (its components').
    fn scalar(&self, ty: u32) -> Ty {
        match self.ty(ty) {
            Ty::Vector { component, .. } => self.ty(component),
            other => other,
        }
    }

    /// Runs the entry point with built-in inputs `inputs` (by built-in).
    pub(crate) fn run(&self, inputs: &HashMap<u32, Value>) -> Finished {
        let mut values: HashMap<u32, Value> = self.constants.clone();
        let mut value_types: HashMap<u32, u32> = self.result_types.clone();
        let mut pointers: HashMap<u32, Pointer> = HashMap::new();
        let mut variables: HashMap<u32, Value> = HashMap::new();
        let mut global_values: HashMap<u32, Value> = HashMap::new();
        for (&variable, builtin) in &self.builtins {
            if let Some(value) = inputs.get(builtin) {
                global_values.insert(variable, value.clone());
            }
        }
        let mut at = 0;
        let mut steps = 0u32;
        loop {
            steps += 1;
            assert!(steps < 1_000_000, "the program runs too long");
            let instruction = &self.body[at];
            at += 1;
            let operands = &instruction.operands;
            let opcode = instruction.opcode;
            match opcode {
                op::LABEL | op::SELECTION_MERGE | op::LOOP_MERGE => continue,
                op::VARIABLE => {
                    let pointee = match self.ty(operands[0]) {
                        Ty::Pointer { pointee } => pointee,
                        _ => panic!("a variable of a non-pointer type"),
                    };
                    let width = match self.ty(pointee) {
                        Ty::Vector { count, .. } => count as usize,
                        _ => 1,
                    };
                    let value = operands
                        .get(3)
                        .map_or_else(|| vec![0; width], |id| values[id].clone());
                    variables.insert(operands[1], value);
                    pointers.insert(
                        operands[1],
                        Pointer {
                            variable: operands[1],
                            component: None,
                        },
                    );
                    continue;
                }
                op::BRANCH => {
                    at = self.labels[&operands[0]];
                    continue;
                }
                op::BRANCH_CONDITIONAL => {
                    let taken = values[&operands[0]][0] != 0;
                    at = self.labels[&operands[if taken { 1 } else { 2 }]];
                    continue;
                }
                op::SWITCH => {
                    let selector = values[&operands[0]][0] as u32;
                    let mut target = operands[1];
                    for pair in operands[2..].chunks(2) {
                        if pair[0] == selector {
                            target = pair[1];
                        }
                    }
                    at = self.labels[&target];
                    continue;
                }
                op::RETURN => break,
                op::UNREACHABLE => panic!("OpUnreachable reached"),
                op::STORE => {
                    let pointer = pointers[&operands[0]];
                    let value = values[&operands[1]].clone();
                    let target = variables
                        .get_mut(&pointer.variable)
                        .expect("a store to a function variable");
                    match pointer.component {
                        Some(n) => target[n] = value[0],
                        None => *target = value,
                    }
                    continue;
                }
                _ => {}
            }
            // Instructions with a result.
            let (ty, id, args) = (operands[0], operands[1], &operands[2..]);
            let value = |n: usize| values[&args[n]].clone();
            let result: Value = match opcode {
                op::LOAD => {
                    if let Some(global) = global_values.get(&args[0]) {
                        global.clone()
                    } else {
                        let pointer = pointers
                            .get(&args[0])
                            .unwrap_or_else(|| panic!("a load of %{} (not an input)", args[0]));
                        let whole = &variables[&pointer.variable];
                        match pointer.component {
                            Some(n) => vec![whole[n]],
                            None => whole.clone(),
                        }
                    }
                }
                op::ACCESS_CHAIN => {
                    let base = pointers[&args[0]];
                    let component = values[&args[1]][0] as usize;
                    assert!(args.len() == 2 && base.component.is_none());
                    pointers.insert(
                        id,
                        Pointer {
                            variable: base.variable,
                            component: Some(component),
                        },
                    );
                    continue;
                }
                op::BITCAST => self.bitcast(value_types[&args[0]], ty, &value(0)),
                op::COMPOSITE_CONSTRUCT => args.iter().flat_map(|a| values[a].clone()).collect(),
                op::COMPOSITE_EXTRACT => vec![value(0)[args[1] as usize]],
                op::EXT_INST => {
                    assert_eq!(args[0], self.glsl, "an unknown extended instruction set");
                    let inputs: Vec<Value> = args[2..].iter().map(|a| values[a].clone()).collect();
                    self.extended(ty, args[1], &inputs)
                }
                _ => {
                    let inputs: Vec<Value> = args.iter().map(|a| values[a].clone()).collect();
                    let operand_type = args
                        .iter()
                        .rev()
                        .find_map(|a| value_types.get(a).copied())
                        .unwrap_or(ty);
                    self.arithmetic(opcode, ty, operand_type, &inputs)
                }
            };
            self.check(ty, &result);
            values.insert(id, result);
            value_types.insert(id, ty);
        }
        Finished { variables }
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
        if matches!(self.scalar(operand), Ty::Float { width: 16 }) && opcode != F_CONVERT {
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
                _ => panic!("the interpreter has no GLSL.std.450 instruction {instruction}"),
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
        let finished = Machine::new(&translated.spirv).run(&HashMap::new());
        let registers = registers
            .iter()
            .map(|(&n, variable)| (n, finished.variables[variable][0] as u32))
            .collect();
        let predicates = predicates
            .iter()
            .map(|(&n, variable)| (n, finished.variables[variable][0] != 0))
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
}
