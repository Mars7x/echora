//! A small SPIR-V module writer: the binary layout, unique types and
//! constants, and helpers for the instructions the translator emits.
//! Opcode and enumerant numbers are from the SPIR-V 1.6 specification
//! (Khronos) and its GLSL.std.450 extended instruction set.

use std::collections::HashMap;

/// Opcodes used by the translator.
#[allow(missing_docs)]
pub mod op {
    pub const NAME: u16 = 5;
    pub const EXTENSION: u16 = 10;
    pub const EXT_INST_IMPORT: u16 = 11;
    pub const EXT_INST: u16 = 12;
    pub const MEMORY_MODEL: u16 = 14;
    pub const ENTRY_POINT: u16 = 15;
    pub const EXECUTION_MODE: u16 = 16;
    pub const CAPABILITY: u16 = 17;
    pub const TYPE_VOID: u16 = 19;
    pub const TYPE_BOOL: u16 = 20;
    pub const TYPE_INT: u16 = 21;
    pub const TYPE_FLOAT: u16 = 22;
    pub const TYPE_VECTOR: u16 = 23;
    pub const TYPE_ARRAY: u16 = 28;
    pub const TYPE_RUNTIME_ARRAY: u16 = 29;
    pub const TYPE_STRUCT: u16 = 30;
    pub const TYPE_POINTER: u16 = 32;
    pub const TYPE_FUNCTION: u16 = 33;
    pub const CONSTANT_TRUE: u16 = 41;
    pub const CONSTANT_FALSE: u16 = 42;
    pub const CONSTANT: u16 = 43;
    pub const FUNCTION: u16 = 54;
    pub const FUNCTION_END: u16 = 56;
    pub const VARIABLE: u16 = 59;
    pub const LOAD: u16 = 61;
    pub const STORE: u16 = 62;
    pub const ACCESS_CHAIN: u16 = 65;
    pub const DECORATE: u16 = 71;
    pub const MEMBER_DECORATE: u16 = 72;
    pub const COMPOSITE_CONSTRUCT: u16 = 80;
    pub const COMPOSITE_EXTRACT: u16 = 81;
    pub const CONVERT_F_TO_U: u16 = 109;
    pub const CONVERT_F_TO_S: u16 = 110;
    pub const CONVERT_S_TO_F: u16 = 111;
    pub const CONVERT_U_TO_F: u16 = 112;
    pub const U_CONVERT: u16 = 113;
    pub const S_CONVERT: u16 = 114;
    pub const CONVERT_U_TO_PTR: u16 = 120;
    pub const BITCAST: u16 = 124;
    pub const S_NEGATE: u16 = 126;
    pub const F_NEGATE: u16 = 127;
    pub const I_ADD: u16 = 128;
    pub const F_ADD: u16 = 129;
    pub const I_SUB: u16 = 130;
    pub const F_SUB: u16 = 131;
    pub const I_MUL: u16 = 132;
    pub const F_MUL: u16 = 133;
    pub const F_DIV: u16 = 136;
    pub const I_ADD_CARRY: u16 = 149;
    pub const IS_NAN: u16 = 156;
    pub const LOGICAL_EQUAL: u16 = 164;
    pub const LOGICAL_NOT_EQUAL: u16 = 165;
    pub const LOGICAL_OR: u16 = 166;
    pub const LOGICAL_AND: u16 = 167;
    pub const LOGICAL_NOT: u16 = 168;
    pub const SELECT: u16 = 169;
    pub const I_EQUAL: u16 = 170;
    pub const I_NOT_EQUAL: u16 = 171;
    pub const U_GREATER_THAN: u16 = 172;
    pub const S_GREATER_THAN: u16 = 173;
    pub const U_GREATER_THAN_EQUAL: u16 = 174;
    pub const S_GREATER_THAN_EQUAL: u16 = 175;
    pub const U_LESS_THAN: u16 = 176;
    pub const S_LESS_THAN: u16 = 177;
    pub const U_LESS_THAN_EQUAL: u16 = 178;
    pub const S_LESS_THAN_EQUAL: u16 = 179;
    pub const F_ORD_EQUAL: u16 = 180;
    pub const F_UNORD_EQUAL: u16 = 181;
    pub const F_ORD_NOT_EQUAL: u16 = 182;
    pub const F_UNORD_NOT_EQUAL: u16 = 183;
    pub const F_ORD_LESS_THAN: u16 = 184;
    pub const F_UNORD_LESS_THAN: u16 = 185;
    pub const F_ORD_GREATER_THAN: u16 = 186;
    pub const F_UNORD_GREATER_THAN: u16 = 187;
    pub const F_ORD_LESS_THAN_EQUAL: u16 = 188;
    pub const F_UNORD_LESS_THAN_EQUAL: u16 = 189;
    pub const F_ORD_GREATER_THAN_EQUAL: u16 = 190;
    pub const F_UNORD_GREATER_THAN_EQUAL: u16 = 191;
    pub const SHIFT_RIGHT_LOGICAL: u16 = 194;
    pub const SHIFT_RIGHT_ARITHMETIC: u16 = 195;
    pub const SHIFT_LEFT_LOGICAL: u16 = 196;
    pub const BITWISE_OR: u16 = 197;
    pub const BITWISE_XOR: u16 = 198;
    pub const BITWISE_AND: u16 = 199;
    pub const NOT: u16 = 200;
    pub const BIT_FIELD_S_EXTRACT: u16 = 202;
    pub const BIT_FIELD_U_EXTRACT: u16 = 203;
    pub const CONTROL_BARRIER: u16 = 224;
    pub const ATOMIC_EXCHANGE: u16 = 229;
    pub const ATOMIC_COMPARE_EXCHANGE: u16 = 230;
    pub const ATOMIC_I_ADD: u16 = 234;
    pub const ATOMIC_S_MIN: u16 = 236;
    pub const ATOMIC_U_MIN: u16 = 237;
    pub const ATOMIC_S_MAX: u16 = 238;
    pub const ATOMIC_U_MAX: u16 = 239;
    pub const ATOMIC_AND: u16 = 240;
    pub const ATOMIC_OR: u16 = 241;
    pub const ATOMIC_XOR: u16 = 242;
    pub const ATOMIC_F_ADD: u16 = 6035;
    pub const LOOP_MERGE: u16 = 246;
    pub const SELECTION_MERGE: u16 = 247;
    pub const LABEL: u16 = 248;
    pub const BRANCH: u16 = 249;
    pub const BRANCH_CONDITIONAL: u16 = 250;
    pub const SWITCH: u16 = 251;
    pub const RETURN: u16 = 253;
}

/// Storage classes.
#[allow(missing_docs)]
pub mod storage {
    pub const UNIFORM_CONSTANT: u32 = 0;
    pub const INPUT: u32 = 1;
    pub const UNIFORM: u32 = 2;
    pub const OUTPUT: u32 = 3;
    pub const WORKGROUP: u32 = 4;
    pub const PRIVATE: u32 = 6;
    pub const FUNCTION: u32 = 7;
    pub const PUSH_CONSTANT: u32 = 9;
    pub const STORAGE_BUFFER: u32 = 12;
    pub const PHYSICAL_STORAGE_BUFFER: u32 = 5349;
}

/// Capabilities.
#[allow(missing_docs)]
pub mod capability {
    pub const SHADER: u32 = 1;
    pub const INT64: u32 = 11;
    pub const ATOMIC_FLOAT32_ADD: u32 = 6033;
    pub const PHYSICAL_STORAGE_BUFFER_ADDRESSES: u32 = 5347;
    /// The addressing model with physical storage buffer pointers.
    pub const ADDRESSING_PHYSICAL_STORAGE_BUFFER64: u32 = 5348;
}

/// Decorations.
#[allow(missing_docs)]
pub mod decoration {
    pub const BLOCK: u32 = 2;
    pub const ARRAY_STRIDE: u32 = 6;
    pub const BUILT_IN: u32 = 11;
    pub const BINDING: u32 = 33;
    pub const DESCRIPTOR_SET: u32 = 34;
    pub const OFFSET: u32 = 35;
}

/// Execution models and modes.
#[allow(missing_docs)]
pub mod execution {
    pub const VERTEX: u32 = 0;
    pub const FRAGMENT: u32 = 4;
    pub const GL_COMPUTE: u32 = 5;
    pub const LOCAL_SIZE: u32 = 17;
}

/// GLSL.std.450 extended instructions.
#[allow(missing_docs)]
pub mod glsl {
    pub const ROUND_EVEN: u32 = 2;
    pub const TRUNC: u32 = 3;
    pub const F_ABS: u32 = 4;
    pub const FLOOR: u32 = 8;
    pub const CEIL: u32 = 9;
    pub const SIN: u32 = 13;
    pub const COS: u32 = 14;
    pub const TANH: u32 = 21;
    pub const EXP2: u32 = 29;
    pub const LOG2: u32 = 30;
    pub const SQRT: u32 = 31;
    pub const INVERSE_SQRT: u32 = 32;
    pub const U_MIN: u32 = 38;
    pub const S_MIN: u32 = 39;
    pub const U_MAX: u32 = 41;
    pub const S_MAX: u32 = 42;
    pub const F_MIN: u32 = 37;
    pub const F_MAX: u32 = 40;
    pub const F_CLAMP: u32 = 43;
    pub const FMA: u32 = 50;
    pub const N_MIN: u32 = 79;
    pub const N_MAX: u32 = 80;
}

/// Function control, selection control and loop control: none.
pub const NONE: u32 = 0;

/// Types the module declares once each.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Type {
    Void,
    Bool,
    Int { width: u32, signed: bool },
    Float { width: u32 },
    Vector { component: u32, count: u32 },
    Array { element: u32, length: u32 },
    RuntimeArray { element: u32 },
    Struct { members: Vec<u32> },
    Pointer { storage: u32, pointee: u32 },
    Function { result: u32, parameters: Vec<u32> },
}

/// A module being built. Sections are kept apart and joined in the order
/// the specification requires.
#[derive(Debug, Default)]
pub struct Module {
    next_id: u32,
    capabilities: Vec<u32>,
    extensions: Vec<u32>,
    imports: Vec<u32>,
    memory_model: Vec<u32>,
    entry_points: Vec<u32>,
    execution_modes: Vec<u32>,
    names: Vec<u32>,
    annotations: Vec<u32>,
    globals: Vec<u32>,
    /// Function bodies: the current function's variables (which must open
    /// its first block) and code are joined at its end.
    functions: Vec<u32>,
    types: HashMap<Type, u32>,
    constants: HashMap<(u32, u32), u32>,
    constants64: HashMap<(u32, u64), u32>,
    glsl: Option<u32>,
}

/// One instruction's words: word count and opcode, then the operands.
fn instruction(code: &mut Vec<u32>, opcode: u16, operands: &[u32]) {
    let count = u32::try_from(operands.len() + 1).expect("an instruction's length fits");
    code.push(count << 16 | u32::from(opcode));
    code.extend_from_slice(operands);
}

/// A string as SPIR-V literal words (nul-terminated, zero-padded).
pub fn string(text: &str) -> Vec<u32> {
    let mut bytes = text.as_bytes().to_vec();
    bytes.push(0);
    bytes.resize(bytes.len().next_multiple_of(4), 0);
    let (words, _) = bytes.as_chunks::<4>();
    words.iter().map(|word| u32::from_le_bytes(*word)).collect()
}

impl Module {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            ..Self::default()
        }
    }

    pub fn id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Declares a capability (once).
    pub fn capability(&mut self, capability: u32) {
        let (declared, _) = self.capabilities.as_chunks::<2>();
        if declared.iter().all(|&[_, existing]| existing != capability) {
            instruction(&mut self.capabilities, op::CAPABILITY, &[capability]);
        }
    }

    /// Declares an extension (once).
    pub fn extension(&mut self, name: &str) {
        let mut words = Vec::new();
        instruction(&mut words, op::EXTENSION, &string(name));
        if !self
            .extensions
            .windows(words.len())
            .any(|window| window == words.as_slice())
        {
            self.extensions.extend(words);
        }
    }

    /// The GLSL.std.450 import (declared once).
    pub fn glsl(&mut self) -> u32 {
        if let Some(id) = self.glsl {
            return id;
        }
        let id = self.id();
        let mut operands = vec![id];
        operands.extend(string("GLSL.std.450"));
        instruction(&mut self.imports, op::EXT_INST_IMPORT, &operands);
        self.glsl = Some(id);
        id
    }

    pub fn memory_model(&mut self, addressing: u32, memory: u32) {
        instruction(
            &mut self.memory_model,
            op::MEMORY_MODEL,
            &[addressing, memory],
        );
    }

    pub fn entry_point(&mut self, model: u32, function: u32, name: &str, interface: &[u32]) {
        let mut operands = vec![model, function];
        operands.extend(string(name));
        operands.extend_from_slice(interface);
        instruction(&mut self.entry_points, op::ENTRY_POINT, &operands);
    }

    pub fn execution_mode(&mut self, function: u32, mode: u32, literals: &[u32]) {
        let mut operands = vec![function, mode];
        operands.extend_from_slice(literals);
        instruction(&mut self.execution_modes, op::EXECUTION_MODE, &operands);
    }

    pub fn name(&mut self, target: u32, name: &str) {
        let mut operands = vec![target];
        operands.extend(string(name));
        instruction(&mut self.names, op::NAME, &operands);
    }

    pub fn decorate(&mut self, target: u32, decoration: u32, literals: &[u32]) {
        let mut operands = vec![target, decoration];
        operands.extend_from_slice(literals);
        instruction(&mut self.annotations, op::DECORATE, &operands);
    }

    pub fn member_decorate(&mut self, target: u32, member: u32, decoration: u32, literals: &[u32]) {
        let mut operands = vec![target, member, decoration];
        operands.extend_from_slice(literals);
        instruction(&mut self.annotations, op::MEMBER_DECORATE, &operands);
    }

    /// The id of `ty`, declaring it the first time. Struct types are
    /// shared by member list too, so decorate them only where the members
    /// identify a single use.
    pub fn ty(&mut self, ty: Type) -> u32 {
        if let Some(&id) = self.types.get(&ty) {
            return id;
        }
        let id = self.id();
        let (opcode, operands) = match &ty {
            Type::Void => (op::TYPE_VOID, vec![id]),
            Type::Bool => (op::TYPE_BOOL, vec![id]),
            Type::Int { width, signed } => (op::TYPE_INT, vec![id, *width, u32::from(*signed)]),
            Type::Float { width } => (op::TYPE_FLOAT, vec![id, *width]),
            Type::Vector { component, count } => (op::TYPE_VECTOR, vec![id, *component, *count]),
            Type::Array { element, length } => (op::TYPE_ARRAY, vec![id, *element, *length]),
            Type::RuntimeArray { element } => (op::TYPE_RUNTIME_ARRAY, vec![id, *element]),
            Type::Struct { members } => {
                let mut operands = vec![id];
                operands.extend_from_slice(members);
                (op::TYPE_STRUCT, operands)
            }
            Type::Pointer { storage, pointee } => (op::TYPE_POINTER, vec![id, *storage, *pointee]),
            Type::Function { result, parameters } => {
                let mut operands = vec![id, *result];
                operands.extend_from_slice(parameters);
                (op::TYPE_FUNCTION, operands)
            }
        };
        instruction(&mut self.globals, opcode, &operands);
        self.types.insert(ty, id);
        id
    }

    pub fn void(&mut self) -> u32 {
        self.ty(Type::Void)
    }

    pub fn bool(&mut self) -> u32 {
        self.ty(Type::Bool)
    }

    pub fn u32(&mut self) -> u32 {
        self.ty(Type::Int {
            width: 32,
            signed: false,
        })
    }

    pub fn u64(&mut self) -> u32 {
        self.ty(Type::Int {
            width: 64,
            signed: false,
        })
    }

    pub fn f32(&mut self) -> u32 {
        self.ty(Type::Float { width: 32 })
    }

    pub fn pointer(&mut self, storage: u32, pointee: u32) -> u32 {
        self.ty(Type::Pointer { storage, pointee })
    }

    /// A 32-bit (or narrower) constant of type `ty`, declared once.
    pub fn constant(&mut self, ty: u32, value: u32) -> u32 {
        if let Some(&id) = self.constants.get(&(ty, value)) {
            return id;
        }
        let id = self.id();
        instruction(&mut self.globals, op::CONSTANT, &[ty, id, value]);
        self.constants.insert((ty, value), id);
        id
    }

    /// A 64-bit constant of type `ty`, declared once.
    pub fn constant64(&mut self, ty: u32, value: u64) -> u32 {
        if let Some(&id) = self.constants64.get(&(ty, value)) {
            return id;
        }
        let id = self.id();
        instruction(
            &mut self.globals,
            op::CONSTANT,
            &[ty, id, value as u32, (value >> 32) as u32],
        );
        self.constants64.insert((ty, value), id);
        id
    }

    pub fn u32_constant(&mut self, value: u32) -> u32 {
        let ty = self.u32();
        self.constant(ty, value)
    }

    /// `true` or `false`, declared once each.
    pub fn bool_constant(&mut self, value: bool) -> u32 {
        let ty = self.bool();
        // Keyed apart from integer constants by an impossible value.
        let key = (ty, u32::from(value));
        if let Some(&id) = self.constants.get(&key) {
            return id;
        }
        let id = self.id();
        let opcode = if value {
            op::CONSTANT_TRUE
        } else {
            op::CONSTANT_FALSE
        };
        instruction(&mut self.globals, opcode, &[ty, id]);
        self.constants.insert(key, id);
        id
    }

    /// A module-scope variable (not `Function` storage).
    pub fn global_variable(&mut self, pointer_type: u32, storage: u32) -> u32 {
        let id = self.id();
        instruction(
            &mut self.globals,
            op::VARIABLE,
            &[pointer_type, id, storage],
        );
        id
    }

    /// The module's words: the header (SPIR-V 1.6, generator 0, the id
    /// bound) and every section in order.
    pub fn finish(self) -> Vec<u32> {
        let mut words = vec![0x0723_0203, 0x0001_0600, 0, self.next_id, 0];
        for section in [
            self.capabilities,
            self.extensions,
            self.imports,
            self.memory_model,
            self.entry_points,
            self.execution_modes,
            self.names,
            self.annotations,
            self.globals,
            self.functions,
        ] {
            words.extend(section);
        }
        words
    }
}

/// A function being built: its variables are collected apart, because
/// SPIR-V requires them at the start of the first block.
#[derive(Debug)]
pub struct Function {
    pub id: u32,
    header: Vec<u32>,
    variables: Vec<u32>,
    code: Vec<u32>,
}

impl Function {
    /// Begins a function returning `result` of type `function_type`, and
    /// its first block.
    pub fn new(module: &mut Module, result: u32, function_type: u32) -> Self {
        let id = module.id();
        let mut header = Vec::new();
        instruction(
            &mut header,
            op::FUNCTION,
            &[result, id, NONE, function_type],
        );
        let label = module.id();
        instruction(&mut header, op::LABEL, &[label]);
        Self {
            id,
            header,
            variables: Vec::new(),
            code: Vec::new(),
        }
    }

    /// A `Function`-storage variable of pointer type `pointer_type`, with
    /// an optional initializer.
    pub fn variable(
        &mut self,
        module: &mut Module,
        pointer_type: u32,
        initializer: Option<u32>,
    ) -> u32 {
        let id = module.id();
        let mut operands = vec![pointer_type, id, storage::FUNCTION];
        operands.extend(initializer);
        instruction(&mut self.variables, op::VARIABLE, &operands);
        id
    }

    /// An instruction without a result.
    pub fn emit(&mut self, opcode: u16, operands: &[u32]) {
        instruction(&mut self.code, opcode, operands);
    }

    /// An instruction with a result of type `ty`; returns the result id.
    pub fn value(&mut self, module: &mut Module, opcode: u16, ty: u32, operands: &[u32]) -> u32 {
        let id = module.id();
        let mut all = vec![ty, id];
        all.extend_from_slice(operands);
        instruction(&mut self.code, opcode, &all);
        id
    }

    /// A new block's label (to be placed with [`Function::label`]).
    pub fn label(&mut self, label: u32) {
        instruction(&mut self.code, op::LABEL, &[label]);
    }

    /// Ends the function and adds it to `module`.
    pub fn finish(self, module: &mut Module) {
        module.functions.extend(self.header);
        module.functions.extend(self.variables);
        module.functions.extend(self.code);
        instruction(&mut module.functions, op::FUNCTION_END, &[]);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Runs `spirv-val` (SPIRV-Tools) on `words` for Vulkan 1.3 if it is
    /// installed; returns its complaint, or `None` when valid or when the
    /// tool is missing (then the caller's other checks stand alone).
    pub(crate) fn validate(words: &[u32]) -> Option<String> {
        let dir = std::env::temp_dir().join(format!(
            "echora-shader-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).ok()?;
        let path = dir.join("module.spv");
        let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        std::fs::write(&path, bytes).ok()?;
        let output = std::process::Command::new("spirv-val")
            .args(["--target-env", "vulkan1.3"])
            .arg(&path)
            .output()
            .ok()?;
        let _ = std::fs::remove_dir_all(&dir);
        (!output.status.success()).then(|| {
            String::from_utf8_lossy(&output.stdout).into_owned()
                + &String::from_utf8_lossy(&output.stderr)
        })
    }

    #[test]
    fn strings_are_nul_terminated_words() {
        assert_eq!(string("abc"), vec![u32::from_le_bytes(*b"abc\0")]);
        assert_eq!(string("abcd"), vec![u32::from_le_bytes(*b"abcd"), 0]);
    }

    #[test]
    fn an_empty_compute_shader_is_valid() {
        let mut module = Module::new();
        module.capability(capability::SHADER);
        module.memory_model(0, 1); // Logical, GLSL450
        let void = module.void();
        let function_type = module.ty(Type::Function {
            result: void,
            parameters: vec![],
        });
        let mut function = Function::new(&mut module, void, function_type);
        function.emit(op::RETURN, &[]);
        let id = function.id;
        function.finish(&mut module);
        module.entry_point(execution::GL_COMPUTE, id, "main", &[]);
        module.execution_mode(id, execution::LOCAL_SIZE, &[1, 1, 1]);
        let words = module.finish();
        assert_eq!(words[0], 0x0723_0203);
        if let Some(problem) = validate(&words) {
            panic!("{problem}");
        }
    }
}
