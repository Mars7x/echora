//! Translation of a decoded program to a SPIR-V compute shader.
//!
//! The program's general registers and predicates are function variables
//! (the driver's compiler promotes them to SSA values); RZ reads zero and
//! drops writes, PT is true. Control flow is structured (`structure`):
//! nested blocks, loops and if/else. A graph that cannot be (irreducible,
//! or with computed branches) runs through one dispatch loop instead: a
//! `switch` on the index of the block to run next, so any control-flow
//! graph is valid SPIR-V. The subgroup runs the lowest pending block of
//! its invocations each time round (lowest-PC-first, so they reconverge
//! at joins as a warp's threads do before barriers and warp-wide
//! operations). Every
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

/// The loop budget translations use: LOOP_BUDGET, or ECHORA_LOOP_BUDGET
/// (debugging: a small budget turns a loop that never ends into a
/// finished, wrong dispatch).
pub fn loop_budget() -> u32 {
    static BUDGET: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *BUDGET.get_or_init(|| {
        std::env::var("ECHORA_LOOP_BUDGET")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(LOOP_BUDGET)
    })
}

/// ECHORA_GUEST_LINES (debugging): the translation marks each guest
/// instruction with an OpLine (file "step", line its byte offset), which
/// the simulator uses to name the first instruction giving an invocation a
/// non-finite float. No cost on the GPU; off by default.
pub fn guest_lines() -> bool {
    static LINES: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *LINES.get_or_init(|| std::env::var_os("ECHORA_GUEST_LINES").is_some())
}

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
    /// To the instruction at the byte offset in `target` when `condition`
    /// holds, else into the next block.
    Indirect { target: Src, condition: Guard },
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
            Op::Exit | Op::Brx { .. } | Op::Ret { .. } => {
                leaders.insert(index + 1);
            }
            Op::Call { offset } => {
                let target = i64::try_from(index + 1)
                    .ok()
                    .and_then(|next| next.checked_mul(16)?.checked_add(offset))
                    .filter(|target| *target >= 0 && target % 16 == 0)
                    .map(|target| (target / 16) as usize)
                    .filter(|target| *target < code.len())
                    .ok_or_else(|| unsupported(index, "call outside the program"))?;
                leaders.insert(target);
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
            // A call goes to its subroutine; the return address is already
            // in a register, which RET branches to.
            Op::Call { offset } => (
                next - 1,
                End::Branch {
                    target: ((next as i64) * 16 + offset) as usize / 16,
                    condition: Guard {
                        guard: last.guard,
                        extra: Pred::TRUE,
                    },
                },
            ),
            Op::Ret { target } => (
                next - 1,
                End::Indirect {
                    target: Src::Reg(target),
                    condition: Guard {
                        guard: last.guard,
                        extra: Pred::TRUE,
                    },
                },
            ),
            Op::Brx { target } => (
                next - 1,
                End::Indirect {
                    target,
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
    Ok(emit(code, layout)?.finish())
}

/// What a translation needs of the device beyond core Vulkan, before the
/// requirements report reads it: each capability with the guest
/// instruction (byte offset in the code) first needing it, and the first
/// buffer or image store or atomic.
#[derive(Clone, Debug, Default)]
pub struct Needs {
    pub capabilities: Vec<(u32, Option<u32>)>,
    pub store: Option<Option<u32>>,
}

/// `translate`, with what the program needs (`Needs`).
pub fn translate_with_needs(
    code: &[Instruction],
    layout: &Layout,
) -> Result<(Translated, Needs), Unsupported> {
    let emitter = emit(code, layout)?;
    let needs = Needs {
        capabilities: emitter.module.capability_sites().to_vec(),
        store: emitter.module.store_site(),
    };
    Ok((emitter.finish(), needs))
}

/// A translation, the variables of its general registers and those of its
/// predicates (for tests that run it).
#[cfg(test)]
pub(crate) type WithRegisters = (Translated, BTreeMap<u8, u32>, BTreeMap<u8, u32>);

#[cfg(test)]
pub(crate) fn translate_with_registers(
    code: &[Instruction],
    layout: &Layout,
) -> Result<WithRegisters, Unsupported> {
    let emitter = emit(code, layout)?;
    let (registers, predicates) = (emitter.registers.clone(), emitter.predicates.clone());
    Ok((emitter.finish(), registers, predicates))
}

fn emit(code: &[Instruction], layout: &Layout) -> Result<Emitter, Unsupported> {
    if code.is_empty() {
        return Err(unsupported(0, "empty program"));
    }
    let blocks = blocks(code)?;
    let mut emitter = Emitter::new(layout);
    emitter.hmma_pairs = matrix::pairs(code, &blocks);
    emitter.mesh_begin();
    emitter.handle_words = handle_words(code);
    // Structured control flow where the graph allows, else the dispatch
    // loop.
    if !emitter.structured(code, &blocks)? {
        emitter.program(code, &blocks)?;
    }
    emitter.mesh_end()?;
    Ok(emitter)
}

mod denormal;
mod fetch;
mod half;
mod image;
mod io;
pub(crate) mod matrix;
mod memory;
mod mesh;
mod structure;
mod texture;
mod vertex_trace;

/// The uniform registers that only ever hold one constant-bank word (each
/// write a ULDC of that same word), by register: the bank and offset. A
/// bindless texture whose handle is in one binds like a bound one.
fn handle_words(code: &[Instruction]) -> BTreeMap<u8, (u8, u16)> {
    let mut words: BTreeMap<u8, Option<(u8, u16)>> = BTreeMap::new();
    let mut note = |register: u8, word: Option<(u8, u16)>| {
        let entry = words.entry(register).or_insert(word);
        if *entry != word {
            *entry = None;
        }
    };
    for instruction in code {
        let destination = (instruction.raw >> 16) as u8 & 0x3f;
        match instruction.op {
            Op::Uldc {
                d,
                src: Src::Const { bank, offset },
                size,
            } => {
                note(d.0, Some((bank, offset)));
                if size == 5 {
                    note(d.0.saturating_add(1), Some((bank, offset.wrapping_add(4))));
                }
            }
            Op::R2ur { .. }
            | Op::Redux { .. }
            | Op::Vote {
                uniform_out: true, ..
            } => {
                note(destination, None);
            }
            // Any other uniform instruction may write its destination and
            // the registers after it.
            _ if instruction.uniform => {
                for n in 0..4 {
                    note(destination.saturating_add(n), None);
                }
            }
            _ => {}
        }
    }
    words
        .into_iter()
        .filter_map(|(register, word)| Some((register, word?)))
        .collect()
}

/// What a program needs beyond its code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub stage: Stage,
    /// A graphics program's header.
    pub header: Option<crate::header::Header>,
    pub workgroup: [u32; 3],
    /// Bytes of shared memory, when known.
    pub shared_bytes: Option<u32>,
    /// Bytes of local memory per invocation, when known.
    pub local_bytes: Option<u32>,
    /// How HMMA is translated.
    pub matrices: MatrixPath,
    /// Task and mesh programs: their pipeline state.
    pub mesh: Option<MeshLayout>,
    /// Debugging: each structured loop counts its iterations in a buffer
    /// at the device address the specialization constants
    /// `SpecMeaning::LoopTraceLow` and `LoopTraceHigh` give (nothing while
    /// it is 0). Loop k (in the order the program's loops are emitted)
    /// owns words 4k to 4k + 3: the iterations of all invocations, the most
    /// iterations of one invocation in one entry to the loop, the times
    /// the loop budget ran out there, and the loop header's byte offset in
    /// the program plus one (0: no such loop).
    pub loop_trace: bool,
    /// Debugging: vertex and mesh programs record their first invocations'
    /// positions and texture results in a buffer at the device address
    /// `SpecMeaning::VertexTraceLow` and `VertexTraceHigh` give; see
    /// `vertex_trace` for the layout.
    pub vertex_trace: bool,
    /// The device runs 64-bit floats (shaderFloat64): non-`.FTZ` FADD,
    /// FMUL and FFMA keep denormals exactly through f64 (see `denormal`),
    /// whatever the host does with 32-bit denormals.
    pub float64: bool,
    /// Fragment programs: colour output 1 is render target 0's second blend
    /// source (Location 0, Index 1; `Options::dual_source_blend`).
    pub dual_source_blend: bool,
}

/// A task or mesh program's pipeline state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshLayout {
    /// The task program (else the mesh program).
    pub task: bool,
    /// A task program runs before the mesh program.
    pub has_task: bool,
    /// Threads per workgroup (at most 32).
    pub threads: u32,
    pub max_vertices: u32,
    pub max_primitives: u32,
    /// 1 points, 2 lines, 3 triangles.
    pub vertices_per_primitive: u8,
    /// Words of the task payload (the task's output attributes).
    pub payload_words: u32,
    /// Words of the mesh program's local stage buffer, as the pipeline
    /// sizes it (beyond the lines the outputs need).
    pub local_words: u32,
    /// The mesh program's per-vertex outputs (its header's output map,
    /// `header::output_map`).
    pub vertex_outputs: [u32; 5],
    /// The per-primitive outputs (the primitive header's output map).
    pub primitive_outputs: Option<[u32; 5]>,
}

/// HMMA's translation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MatrixPath {
    /// Subgroup shuffles and 32-bit sums (any device).
    #[default]
    Subgroup,
    /// Cooperative matrices (VK_KHR_cooperative_matrix, 16x16x16 at
    /// subgroup scope) for HMMA.16816 and HMMA.1688 (K padded by zeros).
    Cooperative,
    /// As `Cooperative`, the fragments staged as whole registers (half
    /// matrices loaded through word pointers).
    Cooperative32,
    /// The subgroup path with the tensor cores' published arithmetic
    /// (block sums aligned and truncated; see `compute::matrix`).
    Exact,
}

impl Layout {
    /// A layout with only a workgroup size.
    pub fn workgroup(workgroup: [u32; 3]) -> Self {
        Self {
            stage: Stage::Compute,
            header: None,
            workgroup,
            shared_bytes: None,
            local_bytes: None,
            matrices: MatrixPath::Subgroup,
            mesh: None,
            loop_trace: false,
            vertex_trace: false,
            float64: false,
            dual_source_blend: false,
        }
    }
}

/// SPIR-V built-ins.
const BUILTIN_WORKGROUP_ID: u32 = 26;
const BUILTIN_LOCAL_INVOCATION_ID: u32 = 27;
const BUILTIN_SUBGROUP_LOCAL_INVOCATION_ID: u32 = 41;
/// SubgroupEqMask; Ge, Gt, Le and Lt follow, as S2R's 0x38-0x3c do.
const BUILTIN_SUBGROUP_EQ_MASK: u32 = 4416;

/// Descriptors per stage: a stage's bindings start at its slot times this.
pub const MAX_DESCRIPTORS: u32 = 64;

/// The subgroup scope, for subgroup operations.
const SCOPE_SUBGROUP: u32 = 3;

/// Bytes of a constant bank.
const BANK_SIZE: u32 = 0x10000;

/// What a value in `Emitter::known` is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Known {
    /// A register, predicate or uniform register variable's value.
    Variable(u32),
    /// A texture pool entry: binding, element type, index value, and
    /// whether the index is uniform.
    PoolEntry(u32, u32, u32, bool),
    /// Whether the pool entry at an index value is integer.
    PoolFlag(u32),
    /// A pool handle value's texture index (zero past the pool), whether
    /// it is inside the pool, and its sampler index.
    PoolIndex(u32),
    PoolInside(u32),
    PoolSampler(u32),
    /// A lane's index into the matrix scratch (`matrix::MatrixLane`).
    MatrixLane(u8),
    /// A value (the second) reinterpreted as a type (the first).
    Bitcast(u32, u32),
}

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
    stage: Stage,
    header: Option<crate::header::Header>,
    memory: memory::Memory,
    io: io::Io,
    /// Vertex inputs the prologue fetched, by location.
    fetched: BTreeMap<u32, u32>,
    /// Uniform registers holding a fixed constant-bank word.
    handle_words: BTreeMap<u8, (u8, u16)>,
    frag_colors: u32,
    push_fields: u32,
    /// Uniform buffer of each constant bank used, by bank.
    banks: BTreeMap<u8, u32>,
    /// The block type every bank's buffer shares, once declared.
    bank_block: Option<u32>,
    /// Built-in input variables (local invocation id, workgroup id).
    builtins: BTreeMap<u32, u32>,
    interface: Vec<u32>,
    descriptors: Vec<Descriptor>,
    /// Each descriptor's variable.
    descriptor_variables: Vec<u32>,
    /// Combined image samplers by handle bank, offset, texture type and
    /// depth comparison.
    textures: BTreeMap<(u8, u16, u32, bool), u32>,
    /// Cooperative-matrix scratch by component type, and whether any
    /// cooperative matrix is used.
    matrix_scratch: BTreeMap<u32, u32>,
    matrices_used: bool,
    /// Texture pool arrays by binding, and the bindings used (bit n).
    pool_arrays: BTreeMap<u32, u32>,
    pool_bindings: u32,
    /// The texture pool's capacity (a specialization constant), once
    /// declared.
    pool_capacity: Option<u32>,
    /// Whether the texture-pool handles the current texture instruction
    /// uses are within the pool (its results are zeros where not).
    pool_inside: Option<u32>,
    /// For the current texture instruction's pool texture with a uint
    /// view: its views and whether the entry is integer.
    pool_integer: Option<texture::PoolViews>,
    /// The texture pool's integer flags buffer, once declared.
    pool_flags: Option<u32>,
    /// Storage images by handle bank, offset and dimension.
    images: BTreeMap<(u8, u16, u8), image::ImageViews>,
    /// R32_UINT image descriptors for SURED (by handle and dimension).
    atomic_images: BTreeMap<(u8, u16, u8), u32>,
    /// Task and mesh programs' variables.
    mesh: Option<mesh::MeshState>,
    /// Values known in a block, and the block they were made in (see
    /// `known`).
    known: BTreeMap<Known, ((u32, u32), u32)>,
    /// The blocks that dominate the current one through the selections
    /// `choose` is emitting.
    dominators: Vec<(u32, u32)>,
    /// The OpString naming guest code in exit OpLines.
    guest_file: Option<u32>,
    /// Vertex tracing: the buffer address's two specialization constants,
    /// the invocation's record in the current function (a u64 value) and
    /// the texture instructions with entries (`vertex_trace`).
    vertex_trace_constants: Option<(u32, u32)>,
    vertex_trace_record: Option<u32>,
    vertex_trace_entries: Vec<usize>,
    /// The OpString naming guest instructions (`guest_lines`).
    step_file: Option<u32>,
    /// Cooperative HMMA pairs: the first's index to the second's (d, b,
    /// c), and the seconds' indices (emitted with their first).
    hmma_pairs: matrix::Pairs,
    /// More variables at a descriptor's binding (by descriptor index).
    descriptor_aliases: Vec<(usize, u32)>,
    /// Each image descriptor's integer-format spec constant id.
    image_spec_ids: Vec<(usize, u32)>,
    spec_constants: Vec<echora_shader_abi::SpecConstant>,
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
        match layout.stage {
            Stage::Compute => module.execution_mode(entry, execution::LOCAL_SIZE, &workgroup),
            Stage::Fragment => {
                module.execution_mode(entry, execution::ORIGIN_UPPER_LEFT, &[]);
                if layout.header.is_some_and(|header| header.writes_depth) {
                    module.execution_mode(entry, execution::DEPTH_REPLACING, &[]);
                }
            }
            _ => {}
        }
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
            stage: layout.stage,
            header: layout.header,
            memory: memory::Memory::default(),
            io: io::Io::default(),
            fetched: BTreeMap::new(),
            handle_words: BTreeMap::new(),
            frag_colors: 0,
            push_fields: 0,
            banks: BTreeMap::new(),
            bank_block: None,
            builtins: BTreeMap::new(),
            interface: Vec::new(),
            descriptors: Vec::new(),
            descriptor_variables: Vec::new(),
            textures: BTreeMap::new(),
            images: BTreeMap::new(),
            atomic_images: BTreeMap::new(),
            mesh: None,
            known: BTreeMap::new(),
            dominators: Vec::new(),
            guest_file: None,
            vertex_trace_constants: None,
            vertex_trace_record: None,
            vertex_trace_entries: Vec::new(),
            step_file: None,
            hmma_pairs: matrix::Pairs::default(),
            matrix_scratch: BTreeMap::new(),
            matrices_used: false,
            pool_arrays: BTreeMap::new(),
            pool_bindings: 0,
            pool_capacity: None,
            pool_inside: None,
            pool_integer: None,
            pool_flags: None,
            descriptor_aliases: Vec::new(),
            image_spec_ids: Vec::new(),
            spec_constants: Vec::new(),
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
        // Each stage's bindings start at its slot times MAX_DESCRIPTORS;
        // compute programs are alone in their pipeline.
        let slot = match self.stage {
            Stage::Vertex | Stage::Compute => 0,
            Stage::TessControl => 1,
            Stage::TessEval => 2,
            Stage::Geometry => 3,
            Stage::Fragment => 4,
            Stage::Task => 5,
            Stage::Mesh => 6,
        };
        self.assign_bindings(slot * MAX_DESCRIPTORS);
        self.function.finish(&mut self.module);
        let model = match self.stage {
            Stage::Vertex => execution::VERTEX,
            Stage::Fragment => execution::FRAGMENT,
            stage => Self::mesh_model(stage).unwrap_or(execution::GL_COMPUTE),
        };
        self.module
            .entry_point(model, self.entry, "main", &self.interface);
        Translated {
            abi_version: ABI_VERSION,
            stage: self.stage,
            spirv: self.module.finish(),
            descriptors: self.descriptors,
            loaded_generics: self.io.loaded_generics,
            stored_generics: self.io.stored_generics,
            stores: self.io.stores,
            frag_colors: self.frag_colors,
            cbuf_used_sizes: self.cbuf_used_sizes,
            spec_constants: self.spec_constants,
            push_fields: self.push_fields,
            texture_pool_bindings: self.pool_bindings,
        }
    }

    fn value(&mut self, opcode: u16, ty: u32, operands: &[u32]) -> u32 {
        // A value reinterpreted once serves wherever it reaches.
        if let (op::BITCAST, &[source]) = (opcode, operands) {
            let key = Known::Bitcast(ty, source);
            if let Some(value) = self.known(key) {
                return value;
            }
            let value = self.function.value(&mut self.module, opcode, ty, operands);
            self.remember(key, value);
            return value;
        }
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
        // Named for debugging (the simulator prints registers by name).
        self.module.name(variable, &format!("R{n}"));
        self.registers.insert(n, variable);
        variable
    }

    /// The variable of predicate `n`, in the uniform file (kept as 8-15)
    /// for a uniform instruction.
    fn predicate_variable(&mut self, n: u8) -> u32 {
        // Uniform predicates are 8-15 (a uniform instruction's 0-7).
        let n = if n < 8 && self.uniform { n + 8 } else { n };
        if let Some(&variable) = self.predicates.get(&n) {
            return variable;
        }
        let no = self.module.bool_constant(false);
        let variable = self
            .function
            .variable(&mut self.module, self.bool_ptr, Some(no));
        let name = if n < 8 {
            format!("P{n}")
        } else {
            format!("UP{}", n - 8)
        };
        self.module.name(variable, &name);
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
        self.load_variable(variable, self.u32)
    }

    fn write_register(&mut self, n: u8, value: u32) {
        if self.uniform {
            self.write_uniform(n.min(63), value);
        } else if n != 255 {
            let variable = self.register(n);
            self.store_variable(variable, value);
        }
    }

    /// A register-like variable's value: the one stored or loaded earlier
    /// in the same block, else a load.
    /// The value known for `key`, when it was made in the current block or
    /// one that dominates it.
    fn known(&self, key: Known) -> Option<u32> {
        let &(block, value) = self.known.get(&key)?;
        (block == self.function.block() || self.dominators.contains(&block)).then_some(value)
    }

    /// Carries the values known in block `from` on into the current block,
    /// which `from` dominates.
    fn carry_known(&mut self, from: (u32, u32)) {
        let to = self.function.block();
        for (block, _) in self.known.values_mut() {
            if *block == from {
                *block = to;
            }
        }
    }

    /// Records `value` for `key`, made in the current block.
    fn remember(&mut self, key: Known, value: u32) {
        self.known.insert(key, (self.function.block(), value));
    }

    fn load_variable(&mut self, variable: u32, ty: u32) -> u32 {
        if let Some(value) = self.known(Known::Variable(variable)) {
            return value;
        }
        let value = self.value(op::LOAD, ty, &[variable]);
        self.remember(Known::Variable(variable), value);
        value
    }

    fn store_variable(&mut self, variable: u32, value: u32) {
        self.function.emit(op::STORE, &[variable, value]);
        self.remember(Known::Variable(variable), value);
    }

    /// Predicate `pred.index` (8-15: uniform predicate index - 8, as a
    /// per-invocation instruction reads one; 7 and 15 are true).
    fn read_predicate(&mut self, pred: Pred) -> u32 {
        if pred.index == 7 || pred.index == 15 {
            return self.module.bool_constant(!pred.negated);
        }
        let value = {
            let variable = self.predicate_variable(pred.index);
            self.load_variable(variable, self.bool)
        };
        if pred.negated {
            self.value(op::LOGICAL_NOT, self.bool, &[value])
        } else {
            value
        }
    }

    fn write_predicate(&mut self, n: u8, value: u32) {
        if n != 7 && n != 15 {
            let variable = self.predicate_variable(n);
            self.store_variable(variable, value);
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
        self.module.name(variable, &format!("UR{n}"));
        self.uniform_registers.insert(n, variable);
        variable
    }

    fn read_uniform(&mut self, n: u8) -> u32 {
        if n == 63 {
            return self.constant(0);
        }
        let variable = self.uniform_variable(n);
        self.load_variable(variable, self.u32)
    }

    fn write_uniform(&mut self, n: u8, value: u32) {
        if n != 63 {
            let variable = self.uniform_variable(n);
            self.store_variable(variable, value);
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
        // Read back as a float, the bits are `value` again.
        self.remember(Known::Bitcast(self.f32, bits), value);
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
        self.add_descriptor(
            variable,
            Descriptor {
                kind: 0,
                count: 1,
                index: u32::from(bank),
                ..Descriptor::default()
            },
        );
        self.banks.insert(bank, variable);
        Ok(variable)
    }

    /// Records a set-0 resource; its binding is assigned at the end.
    fn add_descriptor(&mut self, variable: u32, descriptor: Descriptor) {
        self.descriptors.push(descriptor);
        self.descriptor_variables.push(variable);
        self.interface.push(variable);
    }

    /// Numbers the descriptors: the stage's base (its slot times
    /// `MAX_DESCRIPTORS`), then uniform buffers, storage buffers, texel
    /// buffers, image buffers, textures and images, each in the order
    /// first used.
    fn assign_bindings(&mut self, base: u32) {
        let mut order: Vec<usize> = (0..self.descriptors.len()).collect();
        order.sort_by_key(|&n| self.descriptors[n].kind);
        let mut sorted = Vec::new();
        for (binding, &n) in order.iter().enumerate() {
            let mut descriptor = self.descriptors[n];
            descriptor.binding = base + binding as u32;
            let aliases = self
                .descriptor_aliases
                .iter()
                .filter(|&&(index, _)| index == n)
                .map(|&(_, variable)| variable);
            let variables: Vec<u32> = std::iter::once(self.descriptor_variables[n])
                .chain(aliases)
                .collect();
            for variable in variables {
                self.module
                    .decorate(variable, decoration::DESCRIPTOR_SET, &[0]);
                self.module
                    .decorate(variable, decoration::BINDING, &[descriptor.binding]);
            }
            for &(index, id) in &self.image_spec_ids {
                if index == n {
                    self.spec_constants.push(echora_shader_abi::SpecConstant {
                        id,
                        meaning: echora_shader_abi::SpecMeaning::IntegerFormat(descriptor.binding),
                    });
                }
            }
            sorted.push(descriptor);
        }
        self.descriptors = sorted;
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

    /// `.SAT`: `value` (of type `ty`, a float or a vector of them, with
    /// `bool_type` its comparison type) clamped to [`zero`, `one`], a NaN
    /// (and -0) giving +0 as the hardware does (FClamp leaves NaN
    /// undefined).
    fn saturate(&mut self, ty: u32, bool_type: u32, value: u32, zero: u32, one: u32) -> u32 {
        let positive = self.value(op::F_ORD_GREATER_THAN, bool_type, &[value, zero]);
        let capped = self.glsl(ty, glsl::N_MIN, &[value, one]);
        self.value(op::SELECT, ty, &[positive, capped, zero])
    }

    /// Whether `a` or `b` is a zero factor for a `.DNZ` multiply (NAK's
    /// `fmulz`): ±0, or with `flush` a subnormal too (flushed first).
    fn zero_factor(&mut self, a: u32, b: u32, flush: bool) -> u32 {
        let zero = |e: &mut Self, x: u32| {
            if flush {
                let magnitude = e.glsl(e.f32, glsl::F_ABS, &[x]);
                let smallest = e.float_constant(f32::MIN_POSITIVE);
                e.value(op::F_ORD_LESS_THAN, e.bool, &[magnitude, smallest])
            } else {
                let zero = e.float_constant(0.0);
                e.value(op::F_ORD_EQUAL, e.bool, &[x, zero])
            }
        };
        let (a, b) = (zero(self, a), zero(self, b));
        self.value(op::LOGICAL_OR, self.bool, &[a, b])
    }

    /// A float operation the driver must not fuse with another
    /// (NoContraction): the hardware rounds each one.
    fn exact(&mut self, opcode: u16, operands: &[u32]) -> u32 {
        let value = self.value(opcode, self.f32, operands);
        self.module.decorate(value, decoration::NO_CONTRACTION, &[]);
        value
    }

    /// `value`, the nearest-even result of an operation whose exact
    /// result is `value + residual` (by sign), rounded instead by
    /// `rounding`: 1 down, 2 up, 3 toward zero. Infinities and NaN stay.
    fn directed(&mut self, value: u32, residual: u32, rounding: u8) -> u32 {
        let zero_f = self.float_constant(0.0);
        let below = self.value(op::F_ORD_LESS_THAN, self.bool, &[residual, zero_f]);
        let above = self.value(op::F_ORD_GREATER_THAN, self.bool, &[residual, zero_f]);
        let no = self.module.bool_constant(false);
        let (down, up) = match rounding {
            1 => (below, no),
            2 => (no, above),
            _ => {
                let positive = self.value(op::F_ORD_GREATER_THAN, self.bool, &[value, zero_f]);
                let negative = self.value(op::F_ORD_LESS_THAN, self.bool, &[value, zero_f]);
                (
                    self.value(op::LOGICAL_AND, self.bool, &[positive, below]),
                    self.value(op::LOGICAL_AND, self.bool, &[negative, above]),
                )
            }
        };
        let bits = self.value(op::BITCAST, self.u32, &[value]);
        let (zero, sign, one) = (
            self.constant(0),
            self.constant(0x8000_0000),
            self.constant(1),
        );
        // The next float up: -0 counts as +0.
        let is_negative_zero = self.value(op::I_EQUAL, self.bool, &[bits, sign]);
        let b = self.value(op::SELECT, self.u32, &[is_negative_zero, zero, bits]);
        let sign_bit = self.value(op::BITWISE_AND, self.u32, &[b, sign]);
        let negative = self.value(op::I_NOT_EQUAL, self.bool, &[sign_bit, zero]);
        let minus = self.value(op::I_SUB, self.u32, &[b, one]);
        let plus = self.value(op::I_ADD, self.u32, &[b, one]);
        let next_up = self.value(op::SELECT, self.u32, &[negative, minus, plus]);
        // The next float down: +0 counts as -0.
        let is_zero = self.value(op::I_EQUAL, self.bool, &[bits, zero]);
        let b = self.value(op::SELECT, self.u32, &[is_zero, sign, bits]);
        let sign_bit = self.value(op::BITWISE_AND, self.u32, &[b, sign]);
        let negative = self.value(op::I_NOT_EQUAL, self.bool, &[sign_bit, zero]);
        let minus = self.value(op::I_SUB, self.u32, &[b, one]);
        let plus = self.value(op::I_ADD, self.u32, &[b, one]);
        let next_down = self.value(op::SELECT, self.u32, &[negative, plus, minus]);
        let result = self.value(op::SELECT, self.u32, &[down, next_down, bits]);
        let result = self.value(op::SELECT, self.u32, &[up, next_up, result]);
        let infinite = self.value(op::IS_INF, self.bool, &[value]);
        let nan = self.value(op::IS_NAN, self.bool, &[value]);
        let special = self.value(op::LOGICAL_OR, self.bool, &[infinite, nan]);
        let result = self.value(op::SELECT, self.u32, &[special, bits, result]);
        self.value(op::BITCAST, self.f32, &[result])
    }

    /// A float result with its rounding (`residual`: the sign of the
    /// exact result's excess over `value`, needed for directed rounding)
    /// and saturate modes, written to `d`.
    fn float_result(
        &mut self,
        d: u8,
        value: u32,
        residual: Option<u32>,
        modes: FloatModes,
    ) -> Result<(), Unsupported> {
        let mut value = match residual {
            Some(residual) if modes.rounding != 0 => self.directed(value, residual, modes.rounding),
            _ => value,
        };
        if modes.saturate {
            let (zero, one) = (self.float_constant(0.0), self.float_constant(1.0));
            value = self.saturate(self.f32, self.bool, value, zero, one);
        }
        let bits = self.value(op::BITCAST, self.u32, &[value]);
        self.write_register(d, bits);
        Ok(())
    }

    /// Special register `special` (S2R's numbering): the lane id, thread
    /// and workgroup ids, lane masks (subgroups of 32 lanes) and the clock.
    fn special_register(&mut self, index: usize, special: u8) -> Result<u32, Unsupported> {
        Ok(match special {
            0 => {
                self.module.capability(capability::GROUP_NON_UNIFORM);
                self.scalar_builtin(BUILTIN_SUBGROUP_LOCAL_INVOCATION_ID)
            }
            0x21..=0x23 => self.builtin(BUILTIN_LOCAL_INVOCATION_ID, u32::from(special - 0x21)),
            0x25..=0x27 => self.builtin(BUILTIN_WORKGROUP_ID, u32::from(special - 0x25)),
            // SR_EQMASK, SR_LTMASK, SR_LEMASK, SR_GTMASK, SR_GEMASK (as Mesa's
            // NAK numbers them); SPIR-V orders its masks eq, ge, gt, le, lt.
            0x38..=0x3c => {
                self.module.capability(capability::GROUP_NON_UNIFORM_BALLOT);
                let builtin =
                    BUILTIN_SUBGROUP_EQ_MASK + [0, 4, 3, 2, 1][usize::from(special - 0x38)];
                let uvec4 = self.module.ty(Type::Vector {
                    component: self.u32,
                    count: 4,
                });
                let variable = self.builtin_variable(builtin, uvec4);
                let vector = self.value(op::LOAD, uvec4, &[variable]);
                self.value(op::COMPOSITE_EXTRACT, self.u32, &[vector, 0])
            }
            // THREAD_KILL: whether the invocation is a helper (or was
            // demoted by KILL), as 1 or 0.
            0x13 => {
                self.module
                    .capability(capability::DEMOTE_TO_HELPER_INVOCATION);
                let helper = self.value(op::IS_HELPER_INVOCATION, self.bool, &[]);
                let (one, zero) = (self.constant(1), self.constant(0));
                self.value(op::SELECT, self.u32, &[helper, one, zero])
            }
            // VARIABLE_RATE: the fragment shading rate, X in bits 8-15 and Y
            // in 16-23. Rendering never enables variable-rate shading, so
            // every fragment is 1x1.
            0x84 => self.constant(1 << 8 | 1 << 16),
            0x50 | 0x51 => {
                self.module.capability(capability::SHADER_CLOCK);
                self.module.extension("SPV_KHR_shader_clock");
                let uvec2 = self.module.ty(Type::Vector {
                    component: self.u32,
                    count: 2,
                });
                let subgroup = self.constant(3);
                let clock = self.value(op::READ_CLOCK, uvec2, &[subgroup]);
                self.value(
                    op::COMPOSITE_EXTRACT,
                    self.u32,
                    &[clock, u32::from(special - 0x50)],
                )
            }
            _ => return Err(unsupported(index, format!("special register {special:#x}"))),
        })
    }

    /// A built-in input variable of type `ty`, declared once.
    fn builtin_variable(&mut self, builtin: u32, ty: u32) -> u32 {
        if let Some(&variable) = self.builtins.get(&builtin) {
            return variable;
        }
        let pointer = self.module.pointer(storage::INPUT, ty);
        let variable = self.module.global_variable(pointer, storage::INPUT);
        self.module
            .decorate(variable, decoration::BUILT_IN, &[builtin]);
        self.builtins.insert(builtin, variable);
        self.interface.push(variable);
        variable
    }

    /// A scalar `u32` built-in input.
    fn scalar_builtin(&mut self, builtin: u32) -> u32 {
        let variable = self.builtin_variable(builtin, self.u32);
        self.value(op::LOAD, self.u32, &[variable])
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
        if guest_lines() {
            let file = match self.step_file {
                Some(file) => file,
                None => {
                    let file = self.module.debug_string("step");
                    self.step_file = Some(file);
                    file
                }
            };
            self.function.emit(op::LINE, &[file, index as u32 * 16, 0]);
        }
        let op = instruction.op;
        self.uniform = instruction.uniform;
        self.module.site = Some(u32::try_from(index * 16).unwrap_or(u32::MAX));
        let result = self.guarded(instruction.guard, |e| e.operation(index, op));
        self.module.site = None;
        self.uniform = false;
        result
    }

    fn operation(&mut self, index: usize, op: Op) -> Result<(), Unsupported> {
        match op {
            Op::Nop => {}
            Op::Mov { d, src, lanes } => {
                let mut value = self.source(index, src)?;
                if lanes != 0xf {
                    // Only the quad lanes in the mask take the value.
                    let lane = self.special_register(index, 0)?;
                    let three = self.constant(3);
                    let position = self.value(op::BITWISE_AND, self.u32, &[lane, three]);
                    let mask = self.constant(u32::from(lanes));
                    let bit = self.value(op::SHIFT_RIGHT_LOGICAL, self.u32, &[mask, position]);
                    let one = self.constant(1);
                    let bit = self.value(op::BITWISE_AND, self.u32, &[bit, one]);
                    let zero = self.constant(0);
                    let written = self.value(op::I_NOT_EQUAL, self.bool, &[bit, zero]);
                    let old = self.read_register(d.0);
                    value = self.value(op::SELECT, self.u32, &[written, value, old]);
                }
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
                d,
                a,
                b,
                c,
                modes,
                dnz,
            } if !dnz && self.keeps_denormals(modes) => {
                self.keeping_denormals(index, d.0, denormal::Fused::Fma, &[a, b, c], modes)?
            }
            Op::Ffma {
                d,
                a,
                b,
                c,
                modes,
                dnz,
            } => {
                let (a, b, c) = (
                    self.float(index, a)?,
                    self.float(index, b)?,
                    self.float(index, c)?,
                );
                let mut value = self.glsl(self.f32, glsl::FMA, &[a, b, c]);
                // a*b - value, then + c: the excess's sign.
                let mut residual = (modes.rounding != 0).then(|| {
                    let negated = self.value(op::F_NEGATE, self.f32, &[value]);
                    let part = self.glsl(self.f32, glsl::FMA, &[a, b, negated]);
                    self.exact(op::F_ADD, &[part, c])
                });
                if dnz {
                    // A zero factor: +0 + c, exactly.
                    let zero = self.zero_factor(a, b, modes.flush_to_zero);
                    let plus_zero = self.float_constant(0.0);
                    let sum = self.exact(op::F_ADD, &[plus_zero, c]);
                    value = self.value(op::SELECT, self.f32, &[zero, sum, value]);
                    residual =
                        residual.map(|r| self.value(op::SELECT, self.f32, &[zero, plus_zero, r]));
                }
                self.float_result(d.0, value, residual, modes)?;
            }
            Op::Fmul {
                d,
                a,
                b,
                modes,
                scale,
                dnz,
            } if scale == 4 && !dnz && self.keeps_denormals(modes) => {
                self.keeping_denormals(index, d.0, denormal::Fused::Mul, &[a, b], modes)?
            }
            Op::Fmul {
                d,
                a,
                b,
                modes,
                scale,
                dnz,
            } => {
                // The scale: 1-3 divide by 8, 4, 2; 5-7 multiply by 2, 4, 8.
                if scale == 0 || (scale != 4 && modes.rounding != 0) {
                    return Err(unsupported(index, format!("FMUL scale {scale}")));
                }
                let (a, b) = (self.float(index, a)?, self.float(index, b)?);
                let mut value = self.exact(op::F_MUL, &[a, b]);
                if scale != 4 {
                    let factor = self.float_constant(2f32.powi(i32::from(scale) - 4));
                    value = self.exact(op::F_MUL, &[value, factor]);
                }
                // The product's error, exactly: a*b - value.
                let mut residual = (modes.rounding != 0).then(|| {
                    let negated = self.value(op::F_NEGATE, self.f32, &[value]);
                    self.glsl(self.f32, glsl::FMA, &[a, b, negated])
                });
                if dnz {
                    // A zero factor: +0, whatever the other (Inf, NaN).
                    let zero = self.zero_factor(a, b, modes.flush_to_zero);
                    let plus_zero = self.float_constant(0.0);
                    value = self.value(op::SELECT, self.f32, &[zero, plus_zero, value]);
                    residual =
                        residual.map(|r| self.value(op::SELECT, self.f32, &[zero, plus_zero, r]));
                }
                self.float_result(d.0, value, residual, modes)?;
            }
            Op::Fadd { d, a, b, modes } if self.keeps_denormals(modes) => {
                self.keeping_denormals(index, d.0, denormal::Fused::Add, &[a, b], modes)?
            }
            Op::Fadd { d, a, b, modes } => {
                let (a, b) = (self.float(index, a)?, self.float(index, b)?);
                let value = self.exact(op::F_ADD, &[a, b]);
                // The sum's error, exactly (two-sum).
                let residual = (modes.rounding != 0).then(|| {
                    let b_part = self.exact(op::F_SUB, &[value, a]);
                    let a_part = self.exact(op::F_SUB, &[value, b_part]);
                    let a_error = self.exact(op::F_SUB, &[a, a_part]);
                    let b_error = self.exact(op::F_SUB, &[b, b_part]);
                    self.exact(op::F_ADD, &[a_error, b_error])
                });
                self.float_result(d.0, value, residual, modes)?;
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
                // The first active invocation's value, the same for the
                // whole warp (each invocation's own would make the uniform
                // register, and the branches on it, differ across lanes).
                self.module.capability(capability::GROUP_NON_UNIFORM_BALLOT);
                let value = self.read_register(src.0);
                let subgroup = self.constant(SCOPE_SUBGROUP);
                let first = self.value(
                    op::GROUP_NON_UNIFORM_BROADCAST_FIRST,
                    self.u32,
                    &[subgroup, value],
                );
                self.write_uniform(d.0.min(63), first);
            }
            Op::S2r { d, special } => {
                // Thread id (33-35) and workgroup id (37-39), as on Maxwell.
                let value = self.special_register(index, special)?;
                self.write_register(d.0, value);
            }
            Op::ImadWide {
                d,
                a,
                b,
                c,
                signed,
                high,
                carry_out,
            } => {
                if carry_out != 7 {
                    return Err(unsupported(index, "IMAD.WIDE carry output"));
                }
                let ty = self.u64();
                let (a, b) = (self.integer(index, a)?, self.integer(index, b)?);
                let c = self.pair_operand(index, c)?;
                let (a, b) = (self.widen(a, signed), self.widen(b, signed));
                let product = self.value(op::I_MUL, ty, &[a, b]);
                let sum = self.value(op::I_ADD, ty, &[product, c]);
                if high {
                    let (_, high_word) = self.split(sum);
                    self.write_register(d.0, high_word);
                } else {
                    self.write_pair(d.0, sum);
                }
            }
            Op::BitCount { d, src, invert } => {
                let mut x = self.integer(index, src)?;
                if invert {
                    x = self.value(op::NOT, self.u32, &[x]);
                }
                let value = self.value(op::BIT_COUNT, self.u32, &[x]);
                self.write_register(d.0, value);
            }
            Op::Flo {
                d,
                src,
                invert,
                signed,
                shift_amount,
            } => {
                let mut x = self.integer(index, src)?;
                if invert {
                    x = self.value(op::NOT, self.u32, &[x]);
                }
                // Both give -1 (all ones) where no bit is found.
                let f = if signed {
                    glsl::FIND_S_MSB
                } else {
                    glsl::FIND_U_MSB
                };
                let mut value = self.glsl(self.u32, f, &[x]);
                if shift_amount {
                    let (none, thirty_one) = (self.constant(u32::MAX), self.constant(31));
                    let missing = self.value(op::I_EQUAL, self.bool, &[value, none]);
                    let distance = self.value(op::I_SUB, self.u32, &[thirty_one, value]);
                    value = self.value(op::SELECT, self.u32, &[missing, none, distance]);
                }
                self.write_register(d.0, value);
            }
            Op::Brev { d, src } => {
                let x = self.integer(index, src)?;
                let value = self.value(op::BIT_REVERSE, self.u32, &[x]);
                self.write_register(d.0, value);
            }
            Op::Shfl {
                d,
                p,
                a,
                lane,
                c,
                mode,
            } => {
                self.module
                    .capability(capability::GROUP_NON_UNIFORM_SHUFFLE);
                let value = self.read_register(a.0);
                let b = self.source(index, lane)?;
                let c = self.source(index, c)?;
                let me = self.special_register(index, 0)?;
                let (mask5, eight) = (self.constant(0x1f), self.constant(8));
                let b = self.value(op::BITWISE_AND, self.u32, &[b, mask5]);
                let clamp = self.value(op::BITWISE_AND, self.u32, &[c, mask5]);
                let segment = self.value(op::SHIFT_RIGHT_LOGICAL, self.u32, &[c, eight]);
                let segment = self.value(op::BITWISE_AND, self.u32, &[segment, mask5]);
                let outside = self.value(op::NOT, self.u32, &[segment]);
                let min_lane = self.value(op::BITWISE_AND, self.u32, &[me, segment]);
                let high = self.value(op::BITWISE_AND, self.u32, &[clamp, outside]);
                let max_lane = self.value(op::BITWISE_OR, self.u32, &[min_lane, high]);
                let (source, in_range) = match mode {
                    0 => {
                        let free = self.value(op::BITWISE_AND, self.u32, &[b, outside]);
                        let j = self.value(op::BITWISE_OR, self.u32, &[min_lane, free]);
                        (
                            j,
                            self.value(op::U_LESS_THAN_EQUAL, self.bool, &[j, max_lane]),
                        )
                    }
                    1 => {
                        let j = self.value(op::I_SUB, self.u32, &[me, b]);
                        (
                            j,
                            self.value(op::S_GREATER_THAN_EQUAL, self.bool, &[j, max_lane]),
                        )
                    }
                    2 => {
                        let j = self.value(op::I_ADD, self.u32, &[me, b]);
                        (
                            j,
                            self.value(op::U_LESS_THAN_EQUAL, self.bool, &[j, max_lane]),
                        )
                    }
                    _ => {
                        let j = self.value(op::BITWISE_XOR, self.u32, &[me, b]);
                        (
                            j,
                            self.value(op::U_LESS_THAN_EQUAL, self.bool, &[j, max_lane]),
                        )
                    }
                };
                let source = self.value(op::SELECT, self.u32, &[in_range, source, me]);
                let subgroup = self.constant(SCOPE_SUBGROUP);
                let shuffled = self.value(
                    op::GROUP_NON_UNIFORM_SHUFFLE,
                    self.u32,
                    &[subgroup, value, source],
                );
                self.write_register(d.0, shuffled);
                self.write_predicate(p, in_range);
            }
            Op::Redux {
                d,
                src,
                operation,
                signed,
            } => {
                self.module
                    .capability(capability::GROUP_NON_UNIFORM_ARITHMETIC);
                let opcode = match (operation, signed) {
                    (0, _) => op::GROUP_NON_UNIFORM_BITWISE_AND,
                    (1, _) => op::GROUP_NON_UNIFORM_BITWISE_OR,
                    (2, _) => op::GROUP_NON_UNIFORM_BITWISE_XOR,
                    (3, _) => op::GROUP_NON_UNIFORM_I_ADD,
                    (4, false) => op::GROUP_NON_UNIFORM_U_MIN,
                    (4, true) => op::GROUP_NON_UNIFORM_S_MIN,
                    (5, false) => op::GROUP_NON_UNIFORM_U_MAX,
                    (5, true) => op::GROUP_NON_UNIFORM_S_MAX,
                    _ => return Err(unsupported(index, format!("REDUX operation {operation}"))),
                };
                let value = self.read_register(src.0);
                let subgroup = self.constant(SCOPE_SUBGROUP);
                // Group operation 0: reduce.
                let reduced = self.value(opcode, self.u32, &[subgroup, 0, value]);
                self.write_uniform(d.0.min(63), reduced);
            }
            Op::Vote {
                d,
                p,
                condition,
                mode,
                uniform_out,
            } => {
                self.module.capability(capability::GROUP_NON_UNIFORM_VOTE);
                self.module.capability(capability::GROUP_NON_UNIFORM_BALLOT);
                let condition = self.read_predicate(condition);
                let subgroup = self.constant(SCOPE_SUBGROUP);
                let uvec4 = self.module.ty(Type::Vector {
                    component: self.u32,
                    count: 4,
                });
                let ballot =
                    self.value(op::GROUP_NON_UNIFORM_BALLOT, uvec4, &[subgroup, condition]);
                let ballot = self.value(op::COMPOSITE_EXTRACT, self.u32, &[ballot, 0]);
                if uniform_out {
                    self.write_uniform(d.0.min(63), ballot);
                } else {
                    self.write_register(d.0, ballot);
                }
                let opcode = match mode {
                    0 => op::GROUP_NON_UNIFORM_ALL,
                    1 => op::GROUP_NON_UNIFORM_ANY,
                    2 => op::GROUP_NON_UNIFORM_ALL_EQUAL,
                    _ => return Err(unsupported(index, "VOTE mode 3")),
                };
                if p != 7 {
                    let vote = self.value(opcode, self.bool, &[subgroup, condition]);
                    // VOTEU's vote goes to the uniform predicates.
                    self.uniform = uniform_out;
                    self.write_predicate(p, vote);
                    self.uniform = false;
                }
            }
            Op::Fswzadd { d, a, b, pattern } => {
                // The lane's operation: bits 6 - 2n of the pattern for quad
                // lane n (NAK's SM70 encoding, the reverse of SM50's, which
                // yuzu reads), 0 a + b, 1 a - b, 2 b - a, 3 b.
                let a = self.read_register(a.0);
                let b = self.read_register(b.0);
                let (a, b) = (
                    self.value(op::BITCAST, self.f32, &[a]),
                    self.value(op::BITCAST, self.f32, &[b]),
                );
                let lane = self.special_register(index, 0)?;
                let three = self.constant(3);
                let quad = self.value(op::BITWISE_AND, self.u32, &[lane, three]);
                let two = self.constant(2);
                let reversed = self.value(op::BITWISE_XOR, self.u32, &[quad, three]);
                let shift = self.value(op::I_MUL, self.u32, &[reversed, two]);
                let pattern = self.constant(u32::from(pattern));
                let code = self.value(op::SHIFT_RIGHT_LOGICAL, self.u32, &[pattern, shift]);
                let code = self.value(op::BITWISE_AND, self.u32, &[code, three]);
                let sum = self.exact(op::F_ADD, &[a, b]);
                let difference = self.exact(op::F_SUB, &[a, b]);
                let reverse = self.exact(op::F_SUB, &[b, a]);
                let mut value = b;
                for (n, candidate) in [(0, sum), (1, difference), (2, reverse)] {
                    let n = self.constant(n);
                    let matches = self.value(op::I_EQUAL, self.bool, &[code, n]);
                    value = self.value(op::SELECT, self.f32, &[matches, candidate, value]);
                }
                self.write_float(d.0, value);
            }
            Op::Half {
                op,
                d,
                a,
                b,
                saturate,
                single: true,
                ..
            } => self.half_single(index, op, d, a, b, saturate)?,
            Op::Half {
                op,
                d,
                a,
                b,
                c,
                saturate,
                relu,
                flush,
                dnz,
                ..
            } => self.half(index, op, d, a, b, c, saturate, relu, flush, dnz)?,
            Op::Hsetp2 {
                p,
                q,
                a,
                b,
                compare,
                combine,
                accumulate,
            } => self.half_set_predicates(index, p, q, a, b, compare, combine, accumulate)?,
            Op::F2fp {
                d,
                a,
                b,
                c,
                merge,
                rounding,
            } => self.pack_halves(index, d, a, b, c, merge, rounding)?,
            Op::Iabs { d, src } => {
                let x = self.integer(index, src)?;
                let value = self.glsl(self.u32, glsl::S_ABS, &[x]);
                self.write_register(d.0, value);
            }
            Op::P2r { d, a, mask, byte } => {
                let a = self.read_register(a.0);
                let zero = self.constant(0);
                let mut bits = zero;
                for n in 0..7u8 {
                    if mask >> n & 1 == 0 {
                        continue;
                    }
                    let predicate = self.read_predicate(Pred {
                        index: n,
                        negated: false,
                    });
                    let bit = self.constant(1 << n);
                    let value = self.value(op::SELECT, self.u32, &[predicate, bit, zero]);
                    bits = self.value(op::BITWISE_OR, self.u32, &[bits, value]);
                }
                let shift = u32::from(byte) * 8;
                let place = self.constant(shift);
                let bits = self.value(op::SHIFT_LEFT_LOGICAL, self.u32, &[bits, place]);
                let keep = self.constant(!((mask & 0x7f) << shift));
                let kept = self.value(op::BITWISE_AND, self.u32, &[a, keep]);
                let value = self.value(op::BITWISE_OR, self.u32, &[kept, bits]);
                self.write_register(d.0, value);
            }
            Op::R2p { a, mask, byte } => {
                let a = self.read_register(a.0);
                let zero = self.constant(0);
                for n in 0..7u8 {
                    if mask >> n & 1 == 0 {
                        continue;
                    }
                    let bit = self.constant(1 << (u32::from(n) + u32::from(byte) * 8));
                    let value = self.value(op::BITWISE_AND, self.u32, &[a, bit]);
                    let set = self.value(op::I_NOT_EQUAL, self.bool, &[value, zero]);
                    self.write_predicate(n, set);
                }
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
                // The other modes pick by the selector's low two bits a
                // fixed byte order of c:a (PTX's prmt .f4e, .b4e, .rc8,
                // .ecl, .ecr and .rc16, in NAK's encoding order; the
                // orders as GPGPU-Sim's `read_byte` tables give them), so
                // they are the indexed form with that order's selector.
                let orders: [[u32; 4]; 4] = match mode {
                    0 => [[0; 4]; 4],
                    1 => [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]],
                    2 => [[0, 7, 6, 5], [1, 0, 7, 6], [2, 1, 0, 7], [3, 2, 1, 0]],
                    3 => [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]],
                    4 => [[0, 1, 2, 3], [1, 1, 2, 3], [2, 2, 2, 3], [3, 3, 3, 3]],
                    5 => [[0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 2, 2], [0, 1, 2, 3]],
                    6 => [[0, 1, 0, 1], [2, 3, 2, 3], [0, 1, 0, 1], [2, 3, 2, 3]],
                    _ => return Err(unsupported(index, format!("PRMT mode {mode}"))),
                };
                let a = self.read_register(a.0);
                let selector = self.integer(index, selector)?;
                let selector = if mode == 0 {
                    selector
                } else {
                    let rows = orders.map(|row| {
                        self.constant(row.iter().rev().fold(0, |word, &byte| word << 4 | byte))
                    });
                    let three = self.constant(3);
                    let low = self.value(op::BITWISE_AND, self.u32, &[selector, three]);
                    let mut chosen = rows[0];
                    for (n, &row) in rows.iter().enumerate().skip(1) {
                        let n = self.constant(n as u32);
                        let is = self.value(op::I_EQUAL, self.bool, &[low, n]);
                        chosen = self.value(op::SELECT, self.u32, &[is, row, chosen]);
                    }
                    chosen
                };
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
            Op::Mufu {
                d,
                function,
                src,
                half,
            } => {
                let x = match half {
                    None => self.float(index, src)?,
                    // A half, exactly as a float, with its modifiers.
                    Some(lane) => {
                        let Src::Reg(register) = src.src else {
                            return Err(unsupported(index, "MUFU.F16 of a non-register"));
                        };
                        let bits = self.read_register(register.0);
                        let x = self.half_to_float(bits, lane);
                        let x = if src.abs {
                            self.glsl(self.f32, glsl::F_ABS, &[x])
                        } else {
                            x
                        };
                        if src.neg {
                            self.value(op::F_NEGATE, self.f32, &[x])
                        } else {
                            x
                        }
                    }
                };
                // MUFU flushes a subnormal input to a zero of its sign.
                let x = {
                    let magnitude = self.glsl(self.f32, glsl::F_ABS, &[x]);
                    let smallest = self.float_constant(f32::MIN_POSITIVE);
                    let tiny = self.value(op::F_ORD_LESS_THAN, self.bool, &[magnitude, smallest]);
                    let bits = self.value(op::BITCAST, self.u32, &[x]);
                    let sign = self.constant(0x8000_0000);
                    let signed_zero = self.value(op::BITWISE_AND, self.u32, &[bits, sign]);
                    let signed_zero = self.value(op::BITCAST, self.f32, &[signed_zero]);
                    self.value(op::SELECT, self.f32, &[tiny, signed_zero, x])
                };
                let value = match function {
                    // SIN and COS take turns, not radians.
                    0 | 1 => {
                        let turn = self.float_constant(std::f32::consts::TAU);
                        let radians = self.value(op::F_MUL, self.f32, &[x, turn]);
                        let f = if function == 0 { glsl::COS } else { glsl::SIN };
                        self.glsl(self.f32, f, &[radians])
                    }
                    2 => self.glsl(self.f32, glsl::EXP2, &[x]),
                    // LG2(±0) is -Inf (GLSL leaves Log2(0) undefined).
                    3 => {
                        let log = self.glsl(self.f32, glsl::LOG2, &[x]);
                        let zero = self.float_constant(0.0);
                        let is_zero = self.value(op::F_ORD_EQUAL, self.bool, &[x, zero]);
                        let minus_infinity = self.float_constant(f32::NEG_INFINITY);
                        self.value(op::SELECT, self.f32, &[is_zero, minus_infinity, log])
                    }
                    4 => {
                        let one = self.float_constant(1.0);
                        self.value(op::F_DIV, self.f32, &[one, x])
                    }
                    5 => self.glsl(self.f32, glsl::INVERSE_SQRT, &[x]),
                    8 => self.glsl(self.f32, glsl::SQRT, &[x]),
                    9 => self.glsl(self.f32, glsl::TANH, &[x]),
                    _ => return Err(unsupported(index, format!("MUFU function {function}"))),
                };
                match half {
                    // .F16: the float result rounded toward zero to a half,
                    // as NAK's hardware test finds it (test_op_mufu_f16_down:
                    // F2F.F16.RZ of the 32-bit MUFU of the widened half, but
                    // for Exp2 and Rcp results next to infinity, where the
                    // hardware may give infinity).
                    Some(_) => {
                        let bits = self.round_to_half(value, 3);
                        self.write_register(d.0, bits);
                    }
                    None => self.write_float(d.0, value),
                }
            }
            Op::F2i {
                d,
                src,
                signed,
                dst_size,
                src_size,
                rounding,
            } => {
                let x = match src_size {
                    2 => self.float(index, src)?,
                    // A half in the source's low bits.
                    1 => {
                        if src.abs || src.neg {
                            return Err(unsupported(index, "F2I.F16 with modifiers"));
                        }
                        let bits = self.integer(index, src)?;
                        let vec2 = self.module.ty(Type::Vector {
                            component: self.f32,
                            count: 2,
                        });
                        let pair = self.glsl(vec2, glsl::UNPACK_HALF_2X16, &[bits]);
                        self.value(op::COMPOSITE_EXTRACT, self.f32, &[pair, 0])
                    }
                    _ => return Err(unsupported(index, "F2I from 64-bit")),
                };
                let rounded = self.rounded(x, rounding);
                if dst_size == 3 {
                    let value = self.saturating_integer64(rounded, signed);
                    self.write_pair(d.0, value);
                } else {
                    let value = self.saturating_integer(rounded, signed, 8 << dst_size);
                    self.write_register(d.0, value);
                }
            }
            Op::F2ip { d, a, b, rounding } => {
                let mut bytes = Vec::new();
                for source in [b, a] {
                    let bits = self.read_register(source.0);
                    let x = self.value(op::BITCAST, self.f32, &[bits]);
                    let rounded = self.rounded(x, rounding);
                    bytes.push(self.saturating_integer(rounded, false, 8));
                }
                let eight = self.constant(8);
                let high = self.value(op::SHIFT_LEFT_LOGICAL, self.u32, &[bytes[1], eight]);
                let value = self.value(op::BITWISE_OR, self.u32, &[high, bytes[0]]);
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
                if dst_size != 2 {
                    return Err(unsupported(index, "I2F to other than 32-bit"));
                }
                let opcode = if signed {
                    op::CONVERT_S_TO_F
                } else {
                    op::CONVERT_U_TO_F
                };
                let value = match src_size {
                    // A byte or half from the source's low bits, exactly.
                    0 | 1 => {
                        let x = self.integer(index, src)?;
                        let zero = self.constant(0);
                        let bits = self.constant(if src_size == 0 { 8 } else { 16 });
                        let extract = if signed {
                            op::BIT_FIELD_S_EXTRACT
                        } else {
                            op::BIT_FIELD_U_EXTRACT
                        };
                        let x = self.value(extract, self.u32, &[x, zero, bits]);
                        self.value(opcode, self.f32, &[x])
                    }
                    2 => {
                        let x = self.integer(index, src)?;
                        let value = self.value(opcode, self.f32, &[x]);
                        if rounding == 0 {
                            value
                        } else {
                            // The excess's sign: x against the result
                            // converted back (in 64 bits, where it fits).
                            let ty = self.u64();
                            let wide = self.widen(x, signed);
                            let back_opcode = if signed {
                                op::CONVERT_F_TO_S
                            } else {
                                op::CONVERT_F_TO_U
                            };
                            let back = self.value(back_opcode, ty, &[value]);
                            let (greater, less) = if signed {
                                (op::S_GREATER_THAN, op::S_LESS_THAN)
                            } else {
                                (op::U_GREATER_THAN, op::U_LESS_THAN)
                            };
                            let above = self.value(greater, self.bool, &[wide, back]);
                            let below = self.value(less, self.bool, &[wide, back]);
                            let (one, minus_one, zero) = (
                                self.float_constant(1.0),
                                self.float_constant(-1.0),
                                self.float_constant(0.0),
                            );
                            let residual =
                                self.value(op::SELECT, self.f32, &[below, minus_one, zero]);
                            let residual =
                                self.value(op::SELECT, self.f32, &[above, one, residual]);
                            self.directed(value, residual, rounding)
                        }
                    }
                    _ => {
                        if rounding != 0 {
                            return Err(unsupported(index, "I2F of 64 bits with a rounding mode"));
                        }
                        let x = self.pair_operand(index, src)?;
                        self.value(opcode, self.f32, &[x])
                    }
                };
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
                let (low, high) = match special {
                    255 => (self.constant(0), self.constant(0)),
                    _ => (
                        self.special_register(index, special)?,
                        self.special_register(index, special.wrapping_add(1))?,
                    ),
                };
                self.write_register(d.0, low);
                if wide && !self.zero_register(d.0) {
                    self.write_register(d.0.saturating_add(1), high);
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
                // A byte or half word comes from the word that holds it.
                let words = match size {
                    0..=4 => 1,
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
                let (three, not_three) = (self.constant(3), self.constant(!3));
                let aligned = self.value(op::BITWISE_AND, self.u32, &[address, not_three]);
                let mut values = self.bank_words(variable, aligned, words);
                if size < 4 {
                    let low_bits = self.value(op::BITWISE_AND, self.u32, &[address, three]);
                    values[0] = self.sub_word(values[0], low_bits, size);
                }
                for (n, value) in values.into_iter().enumerate() {
                    if !self.zero_register(d.0) {
                        self.write_register(d.0.saturating_add(n as u8), value);
                    }
                }
            }
            Op::Uldc { d, src, size } => {
                let words = match size {
                    0..=4 => 1,
                    5 => 2,
                    _ => return Err(unsupported(index, format!("ULDC size {size}"))),
                };
                let Src::Const { bank, offset } = src else {
                    return Err(unsupported(index, "bindless ULDC"));
                };
                for n in 0..words {
                    let at = (offset & !3).wrapping_add(n * 4);
                    let mut value = self
                        .constant_word(bank, at)
                        .map_err(|reason| unsupported(index, reason))?;
                    if size < 4 {
                        let low_bits = self.constant(u32::from(offset & 3));
                        value = self.sub_word(value, low_bits, size);
                    }
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
                strong,
            } => self.load(index, d.0, space, address, size, condition, strong)?,
            Op::Store {
                data,
                space,
                address,
                size,
                strong,
            } => self.store(index, data.0, space, address, size, strong)?,
            Op::Atomic {
                d,
                space,
                address,
                data,
                compare,
                operation,
                kind,
            } => self.atomic(
                index,
                d.0,
                space,
                address,
                data.0,
                compare.map(|reg| reg.0),
                operation,
                kind,
            )?,
            Op::Tex(texture) => self.texture(index, &texture)?,
            Op::Surface {
                store,
                d,
                a,
                data,
                bank,
                handle_offset,
                dimension,
                mask,
            } => self.surface(
                index,
                store,
                d.0,
                a.0,
                data.0,
                bank,
                handle_offset,
                dimension,
                mask,
            )?,
            Op::SurfaceReduce {
                a,
                data,
                bank,
                handle_offset,
                dimension,
                operation,
            } => self.surface_reduce(
                index,
                a.0,
                data.0,
                bank,
                handle_offset,
                dimension,
                operation,
            )?,
            Op::Ald {
                d,
                offset,
                vertex,
                address,
                count,
                patch,
                output,
            } => self.attribute_load(
                index,
                d.0,
                offset.0,
                vertex.0,
                address,
                count,
                patch || output,
            )?,
            Op::Ast {
                data,
                offset,
                vertex,
                address,
                count,
                patch,
            } => self.attribute_store(index, data.0, offset.0, vertex.0, address, count, patch)?,
            Op::Ipa {
                d,
                address,
                location,
                mode,
                index: attribute_index,
                ..
            } => match attribute_index {
                // Frequency 1 reads the provoking vertex's value, which
                // only a flat input holds; 2 (state) is not known.
                _ if mode == 2 => return Err(unsupported(index, "IPA.STATE")),
                _ if mode == 1 && !self.flat_input(address) => {
                    return Err(unsupported(index, "IPA.CONSTANT of an interpolated input"));
                }
                Some(register) if register.0 != 63 => {
                    self.interpolate_indexed(index, d.0, address, location, register.0)?
                }
                _ => self.interpolate(index, d.0, address, location)?,
            },
            Op::Kill { condition } => {
                if self.stage != Stage::Fragment {
                    return Err(unsupported(index, "KILL outside a fragment program"));
                }
                self.kill(condition)?;
            }
            Op::Hmma {
                d,
                a,
                b,
                c,
                k16,
                single,
            } => self.matrix_multiply(index, d.0, a.0, b.0, c.0, k16, single)?,
            Op::Isbe {
                write,
                value,
                a,
                offset,
                size,
                region,
                output,
                ..
            } => self.stage_buffer(index, write, value.0, a.0, offset, size, region, output)?,
            Op::Movm { d, a } => self.matrix_transpose(index, d.0, a.0)?,
            Op::Ldsm {
                d,
                a,
                uniform,
                offset,
                count,
                transpose,
            } => self.matrix_load(
                index,
                d.0,
                a.0,
                uniform.map(|register| register.0),
                offset,
                count,
                transpose,
            )?,
            Op::Bar { mode } => {
                if mode != 0 {
                    return Err(unsupported(index, format!("BAR mode {mode}")));
                }
                self.barrier();
            }
            Op::Membar { scope } => self.memory_barrier(scope),
            Op::Ldgsts {
                shared,
                shared_offset,
                global,
                global_offset,
                enable,
            } => self.copy_to_shared(
                index,
                shared.0,
                shared_offset,
                global.0,
                global_offset,
                enable,
            )?,
            Op::Bssy | Op::Bsync | Op::Depbar | Op::Errbar | Op::Ldgdepbar => {}
            Op::Bra { .. } | Op::Exit | Op::Brx { .. } | Op::Call { .. } | Op::Ret { .. } => {
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
    fn saturating_integer(&mut self, x: u32, signed: bool, bits: u32) -> u32 {
        if bits < 32 {
            // Narrow ranges are exact in a float: clamp, then convert.
            let (low, high) = if signed {
                (
                    -(2f32.powi(bits as i32 - 1)),
                    2f32.powi(bits as i32 - 1) - 1.0,
                )
            } else {
                (0.0, 2f32.powi(bits as i32) - 1.0)
            };
            let (low, high) = (self.float_constant(low), self.float_constant(high));
            // NMax/NMin: a NaN gives a bound (FClamp leaves it undefined);
            // it reads 0 below.
            let raised = self.glsl(self.f32, glsl::N_MAX, &[x, low]);
            let clamped = self.glsl(self.f32, glsl::N_MIN, &[raised, high]);
            let opcode = if signed {
                op::CONVERT_F_TO_S
            } else {
                op::CONVERT_F_TO_U
            };
            let value = self.value(opcode, self.u32, &[clamped]);
            let nan = self.value(op::IS_NAN, self.bool, &[x]);
            let zero = self.constant(0);
            return self.value(op::SELECT, self.u32, &[nan, zero, value]);
        }
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
        let raised = self.glsl(self.f32, glsl::N_MAX, &[x, low]);
        let clamped = self.glsl(self.f32, glsl::N_MIN, &[raised, high]);
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

    /// As `saturating_integer`, to a 64-bit integer.
    fn saturating_integer64(&mut self, x: u32, signed: bool) -> u32 {
        let ty = self.u64();
        let (low, high, end, top) = if signed {
            (
                -(2f32.powi(63)),
                2f32.powi(63) - 2f32.powi(39),
                2f32.powi(63),
                i64::MAX as u64,
            )
        } else {
            (0.0, 2f32.powi(64) - 2f32.powi(40), 2f32.powi(64), u64::MAX)
        };
        let (low, high, end) = (
            self.float_constant(low),
            self.float_constant(high),
            self.float_constant(end),
        );
        let raised = self.glsl(self.f32, glsl::N_MAX, &[x, low]);
        let clamped = self.glsl(self.f32, glsl::N_MIN, &[raised, high]);
        let opcode = if signed {
            op::CONVERT_F_TO_S
        } else {
            op::CONVERT_F_TO_U
        };
        let value = self.value(opcode, ty, &[clamped]);
        let over = self.value(op::F_ORD_GREATER_THAN_EQUAL, self.bool, &[x, end]);
        let top = self.constant64(top);
        let value = self.value(op::SELECT, ty, &[over, top, value]);
        let nan = self.value(op::IS_NAN, self.bool, &[x]);
        let zero = self.constant64(0);
        self.value(op::SELECT, ty, &[nan, zero, value])
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

    /// The dispatch loop over `blocks`, selecting each by the index of its
    /// first instruction (so a computed branch's target selects its block
    /// directly; a target that starts no block stops the invocation).
    fn program(&mut self, code: &[Instruction], blocks: &[Block]) -> Result<(), Unsupported> {
        self.vertex_prologue(code);
        self.vertex_trace_begin();
        let starts: BTreeSet<usize> = blocks.iter().map(|block| block.start).collect();
        let u32_ptr = self.u32_ptr;
        let bool_ptr = self.bool_ptr;
        let first = self.constant(0);
        let next = self
            .function
            .variable(&mut self.module, u32_ptr, Some(first));
        let budget_value = self.constant(loop_budget());
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
        // Header: the subgroup runs the lowest next block among its running
        // invocations (so they reconverge where the warp's threads would:
        // a barrier's block runs once for all of them, and subgroup
        // operations see them together), while any runs and within budget
        // (both the same across the subgroup).
        self.function.label(header);
        self.module
            .capability(capability::GROUP_NON_UNIFORM_ARITHMETIC);
        let still = self.value(op::LOAD, self.bool, &[running]);
        let own = self.value(op::LOAD, self.u32, &[next]);
        let none = self.constant(u32::MAX);
        let pending = self.value(op::SELECT, self.u32, &[still, own, none]);
        let subgroup = self.constant(SCOPE_SUBGROUP);
        // Group operation 0: reduce.
        let selector = self.value(
            op::GROUP_NON_UNIFORM_U_MIN,
            self.u32,
            &[subgroup, 0, pending],
        );
        let any = self.value(op::I_NOT_EQUAL, self.bool, &[selector, none]);
        let left = self.value(op::LOAD, self.u32, &[budget]);
        let zero = self.constant(0);
        let some_left = self.value(op::U_GREATER_THAN, self.bool, &[left, zero]);
        let go = self.value(op::LOGICAL_AND, self.bool, &[any, some_left]);
        self.function
            .emit(op::LOOP_MERGE, &[merge, continuing, spirv::NONE]);
        self.function
            .emit(op::BRANCH_CONDITIONAL, &[go, body, merge]);
        // Body: the invocations at that block run it (the switch over
        // blocks); the others wait.
        self.function.label(body);
        let here = self.value(op::I_EQUAL, self.bool, &[own, selector]);
        let chosen = self.value(op::LOGICAL_AND, self.bool, &[still, here]);
        let (run, waiting) = (self.module.id(), self.module.id());
        self.function
            .emit(op::SELECTION_MERGE, &[waiting, spirv::NONE]);
        self.function
            .emit(op::BRANCH_CONDITIONAL, &[chosen, run, waiting]);
        self.function.label(run);
        let switch_merge = self.module.id();
        let labels: Vec<u32> = blocks.iter().map(|_| self.module.id()).collect();
        // An index past the blocks stops the invocation.
        let stop = self.module.id();
        let mut switch = vec![selector, stop];
        for (block, label) in blocks.iter().zip(&labels) {
            switch.extend([block.start as u32, *label]);
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
            let fall = Some(block.end_index(block.body_end))
                .filter(|index| starts.contains(index))
                .map_or(u32::MAX, |index| index as u32);
            match block.end {
                End::Fall => {
                    let fall = self.constant(fall);
                    self.function.emit(op::STORE, &[next, fall]);
                }
                End::Branch { target, condition } => {
                    let target = self.constant(target as u32);
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
                End::Indirect { target, condition } => {
                    let value = self
                        .source(block.body_end, target)
                        .map_err(|error| unsupported(block.body_end, error.reason))?;
                    // A byte offset that is not an instruction's selects
                    // nothing, and stops.
                    let (four, fifteen, zero, none) = (
                        self.constant(4),
                        self.constant(15),
                        self.constant(0),
                        self.constant(u32::MAX),
                    );
                    let low = self.value(op::BITWISE_AND, self.u32, &[value, fifteen]);
                    let aligned = self.value(op::I_EQUAL, self.bool, &[low, zero]);
                    let index = self.value(op::SHIFT_RIGHT_LOGICAL, self.u32, &[value, four]);
                    let index = self.value(op::SELECT, self.u32, &[aligned, index, none]);
                    let fall = self.constant(fall);
                    let taken = self.condition(condition);
                    let chosen = self.value(op::SELECT, self.u32, &[taken, index, fall]);
                    self.function.emit(op::STORE, &[next, chosen]);
                }
            }
            self.function.emit(op::BRANCH, &[switch_merge]);
        }
        self.function.label(switch_merge);
        self.function.emit(op::BRANCH, &[waiting]);
        self.function.label(waiting);
        self.function.emit(op::BRANCH, &[continuing]);
        // Continue: spend one unit of the budget.
        self.function.label(continuing);
        let left = self.value(op::LOAD, self.u32, &[budget]);
        let one = self.constant(1);
        let less = self.value(op::I_SUB, self.u32, &[left, one]);
        self.function.emit(op::STORE, &[budget, less]);
        self.function.emit(op::BRANCH, &[header]);
        self.function.label(merge);
        self.vertex_trace_end();
        self.fragment_outputs();
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
        simulates(&translated);
        translated
    }

    /// Runs a compute program in the simulator, nothing bound: every
    /// instruction it emits must be one the simulator runs (what it then
    /// reports, with no memory or images behind it, does not matter here).
    fn simulates(translated: &Translated) {
        if translated.stage != Stage::Compute {
            return;
        }
        let machine = crate::sim::Machine::new(&translated.spirv);
        machine.simulate(&crate::sim::Inputs {
            workgroup_count: [1, 1, 1],
            loop_limit: 64,
            ..crate::sim::Inputs::default()
        });
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
            word(&[(0, 12, 0x802), (72, 4, 0xf), (16, 8, 1), (32, 32, 5)]),
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
            word(&[
                (0, 12, 0xa02),
                (72, 4, 0xf),
                (16, 8, 1),
                (54, 5, 3),
                (38, 16, 0x14),
            ]),
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
            word(&[
                (0, 12, 0xa02),
                (72, 4, 0xf),
                (16, 8, 1),
                (54, 5, 3),
                (38, 16, 0x2),
            ]),
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
            word(&[(0, 12, 0xc02), (72, 4, 0xf), (16, 8, 6), (32, 6, 5)]),
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
        simulates(&translated);
        translated
    }

    #[test]
    fn strong_global_accesses_are_volatile() {
        // LDG.E.STRONG.GPU R2, [R4.64]; STG.E.STRONG.SYS [R4.64], R2;
        // LDG.E.STRONG.SM R3, [R4.64] (the usual order: not volatile)
        let load = |order: u128, d: u128| {
            word(&[
                (0, 12, 0x381),
                (16, 8, d),
                (24, 8, 4),
                (72, 1, 1),
                (73, 3, 4),
                (77, 4, order),
            ])
        };
        let store = word(&[
            (0, 12, 0x386),
            (24, 8, 4),
            (32, 8, 2),
            (72, 1, 1),
            (73, 3, 4),
            (77, 4, 0xa),
        ]);
        let words = [load(7, 2), store, load(5, 3), EXIT];
        assert!(matches!(decode(words[0]).op, Op::Load { strong: true, .. }));
        assert!(matches!(
            decode(words[1]).op,
            Op::Store { strong: true, .. }
        ));
        assert!(matches!(
            decode(words[2]).op,
            Op::Load { strong: false, .. }
        ));
        let spirv = valid(&words).spirv;
        // OpLoad and OpStore with Volatile | Aligned (3), once each.
        let accesses = |opcode: u32, length: usize| {
            let mut found = 0;
            let mut at = 5;
            while at < spirv.len() {
                let count = (spirv[at] >> 16) as usize;
                if spirv[at] & 0xffff == opcode && count == length && spirv[at + length - 2] == 3 {
                    found += 1;
                }
                at += count.max(1);
            }
            found
        };
        assert_eq!(accesses(61, 6), 1);
        assert_eq!(accesses(62, 5), 1);
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
            stage: Stage::Compute,
            header: None,
            workgroup: [64, 1, 1],
            shared_bytes: Some(0x1000),
            local_bytes: Some(0x40),
            matrices: MatrixPath::Subgroup,
            mesh: None,
            loop_trace: false,
            vertex_trace: false,
            float64: false,
            dual_source_blend: false,
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
            word(&[(0, 12, 0xc02), (72, 4, 0xf), (16, 8, 2), (32, 6, 7)]),
            word(&[(0, 12, 0xc02), (72, 4, 0xf), (16, 8, 3), (32, 6, 8)]),
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
            word(&[(0, 12, 0xa02), (72, 4, 0xf), (16, 8, 4), (54, 5, 24)]),
            EXIT,
        ];
        let code: Vec<Instruction> = words.iter().map(|&raw| decode(raw)).collect();
        assert!(code[0].uniform && code[1].uniform && code[3].uniform);
        assert!(!code[5].uniform && !code[6].uniform);
        let translated = valid(&words);
        assert!(translated.descriptors.is_empty());
    }

    #[test]
    fn uniform_predicates_pack_into_a_uniform_register() {
        // UP2UR UR4, UPR, URZ, 0x7f
        let packed = word(&[
            (0, 12, 0x883),
            (16, 8, 4),
            (24, 8, 63),
            (32, 32, 0x7f),
            (91, 1, 1),
        ]);
        let instruction = decode(packed);
        assert!(instruction.uniform);
        assert!(matches!(instruction.op, Op::P2r { mask: 0x7f, .. }));
        valid(&[packed, EXIT]);
    }

    #[test]
    fn bindless_size_queries_read_the_pool() {
        // TXQ.B R4, R0 (handle), R1 (level), width and height.
        let query = word(&[
            (0, 12, 0x370),
            (16, 8, 4),
            (24, 8, 0),
            (59, 1, 1),
            (64, 8, 255),
            (72, 4, 3),
        ]);
        assert!(matches!(
            decode(query).op,
            Op::Tex(crate::decode::Texture {
                op: crate::decode::TextureOp::Query,
                handle_value: true,
                ..
            })
        ));
        let translated = valid(&[query, EXIT]);
        use echora_shader_abi::texture_pool::{SAMPLED_2D, SAMPLERS};
        assert_ne!(translated.texture_pool_bindings & 1 << SAMPLED_2D, 0);
        assert_ne!(translated.texture_pool_bindings & 1 << SAMPLERS, 0);
        // A query kind past the dimensions is refused.
        assert!(matches!(decode(query | 1 << 80).op, Op::Unknown { .. }));
        // So is TXQ's (0xb6f).
        let bound = word(&[(0, 12, 0xb6f), (16, 8, 4), (24, 8, 0), (72, 4, 3)]);
        assert!(!matches!(decode(bound).op, Op::Unknown { .. }));
        assert!(matches!(decode(bound | 1 << 80).op, Op::Unknown { .. }));
    }

    #[test]
    fn global_to_shared_copies_validate() {
        // LDGSTS.E.BYPASS.128 [R1+0x40], [R2.64+0x10], P0 (the first form),
        // then with a descriptor (UR4) and !P1; LDGDEPBAR.
        let promoted = word(&[
            (0, 12, 0xfae),
            (16, 8, 1),
            (24, 8, 2),
            (32, 12, 0x10),
            (48, 16, 4),
            (64, 32, 0x0810_1d40),
        ]);
        let described = word(&[
            (0, 12, 0xfae),
            (16, 8, 1),
            (24, 8, 2),
            (64, 32, 0x0c10_1c00 | 4 | 1 << 6 | 1 << 23),
        ]);
        assert!(matches!(
            decode(promoted).op,
            Op::Ldgsts {
                shared_offset: 0x40,
                global_offset: 0x10,
                enable: Pred {
                    index: 0,
                    negated: false
                },
                ..
            }
        ));
        assert!(matches!(
            decode(described).op,
            Op::Ldgsts {
                enable: Pred {
                    index: 1,
                    negated: true
                },
                ..
            }
        ));
        let layout = Layout {
            shared_bytes: Some(0x1000),
            ..Layout::workgroup([32, 1, 1])
        };
        valid_with(
            &[promoted, described, word(&[(0, 12, 0x9af)]), EXIT],
            &layout,
        );
        // An odd global register is refused.
        assert!(matches!(decode(promoted | 1 << 24).op, Op::Unknown { .. }));
    }

    #[test]
    fn memory_barriers_validate() {
        // MEMBAR.SC.CTA; MEMBAR.SC.GPU; MEMBAR.SC.SYS; ERRBAR
        let membar = |scope: u128| word(&[(0, 12, 0x992), (76, 3, scope)]);
        valid(&[
            membar(0),
            membar(2),
            membar(3),
            word(&[(0, 12, 0x9ab)]),
            EXIT,
        ]);
        // .MMIO is refused.
        assert!(matches!(decode(membar(2) | 1 << 72).op, Op::Unknown { .. }));
    }

    #[test]
    fn atomics_validate() {
        let words = [
            // RED.E.ADD.F32 [R2.64+UR4], R1 (0x0c10e7 in bits 72-95)
            word(&[
                (0, 12, 0x98e),
                (24, 8, 2),
                (32, 8, 1),
                (64, 6, 4),
                (72, 24, 0x0c10e7),
            ]),
            // RED.E.MIN.S32 [R2.64], R1
            word(&[
                (0, 12, 0x98e),
                (24, 8, 2),
                (32, 8, 1),
                (64, 6, 63),
                (72, 24, 0x0c90e3),
            ]),
            // ATOMG.E.ADD R3, [R2.64+UR4], R1 (pair flag at 70)
            word(&[
                (0, 12, 0x9a8),
                (16, 8, 3),
                (24, 8, 2),
                (32, 8, 1),
                (64, 6, 4),
                (70, 1, 1),
                (72, 24, 0x081ee1),
            ]),
            // ATOMG.E.CAS R5, [R2.64], R3, R4
            word(&[
                (0, 12, 0x3a9),
                (16, 8, 5),
                (24, 8, 2),
                (32, 8, 3),
                (64, 8, 4),
                (72, 1, 1),
            ]),
            // ATOMS.EXCH R6, [R1+0x8], R2; ATOMS.CAS R7, [R1], R2, R3
            word(&[
                (0, 12, 0x38c),
                (16, 8, 6),
                (24, 8, 1),
                (32, 8, 2),
                (40, 24, 8),
                (87, 4, 8),
            ]),
            word(&[
                (0, 12, 0x38d),
                (16, 8, 7),
                (24, 8, 1),
                (32, 8, 2),
                (64, 8, 3),
            ]),
            EXIT,
        ];
        let layout = Layout {
            stage: Stage::Compute,
            header: None,
            workgroup: [64, 1, 1],
            shared_bytes: Some(0x100),
            local_bytes: None,
            matrices: MatrixPath::Subgroup,
            mesh: None,
            loop_trace: false,
            vertex_trace: false,
            float64: false,
            dual_source_blend: false,
        };
        valid_with(&words, &layout);
        let code: Vec<Instruction> = words.iter().map(|&raw| decode(raw)).collect();
        let Op::Atomic {
            address,
            operation,
            kind,
            ..
        } = code[2].op
        else {
            panic!("ATOMG")
        };
        assert!(address.wide && address.uniform.is_some_and(|(_, wide)| wide));
        assert_eq!((operation, kind), (0, 0));
        // Increment wraps at a bound, which is not translated yet.
        let inc = translated_err(&[
            word(&[(0, 12, 0x98e), (24, 8, 2), (32, 8, 1), (87, 3, 3)]),
            EXIT,
        ]);
        assert!(inc.reason.contains("atomic operation 3"));
    }

    #[test]
    fn rounding_modes_conversions_and_special_registers_validate() {
        valid(&[
            // FADD.RZ R2, R0, R1; FMUL.RM R3, R0, R1; FFMA.RP R4, R0, R1, R2
            word(&[
                (0, 12, 0x221),
                (16, 8, 2),
                (24, 8, 0),
                (32, 8, 1),
                (78, 2, 3),
            ]),
            word(&[
                (0, 12, 0x220),
                (16, 8, 3),
                (24, 8, 0),
                (32, 8, 1),
                (78, 2, 1),
                (84, 3, 4),
            ]),
            word(&[
                (0, 12, 0x223),
                (16, 8, 4),
                (24, 8, 0),
                (32, 8, 1),
                (64, 8, 2),
                (78, 2, 2),
            ]),
            // I2F.RZ R5, R1; I2F.S16 R6, R1; I2F.S64 R7, R8
            word(&[
                (0, 12, 0x306),
                (16, 8, 5),
                (32, 8, 1),
                (75, 2, 2),
                (78, 2, 3),
                (84, 2, 2),
            ]),
            word(&[
                (0, 12, 0x306),
                (16, 8, 6),
                (32, 8, 1),
                (74, 1, 1),
                (75, 2, 2),
                (84, 2, 1),
            ]),
            word(&[
                (0, 12, 0x306),
                (16, 8, 7),
                (32, 8, 8),
                (74, 1, 1),
                (75, 2, 2),
                (84, 2, 3),
            ]),
            // IMAD.HI.U32 R9, R1, 0x55555556, RZ; IABS R10, R1
            word(&[
                (0, 12, 0x827),
                (16, 8, 9),
                (24, 8, 1),
                (32, 32, 0x5555_5556),
                (64, 8, 255),
                (81, 3, 7),
            ]),
            word(&[(0, 12, 0x213), (16, 8, 10), (32, 8, 1)]),
            // P2R R11, PR, R1, 0x7f; R2P PR, R11, 0x3
            word(&[(0, 12, 0x803), (16, 8, 11), (24, 8, 1), (32, 32, 0x7f)]),
            word(&[(0, 12, 0x804), (24, 8, 11), (32, 32, 0x3)]),
            // S2R R12, SR_LANEID; S2R R13, SR_LTMASK; CS2R R14, SR_CLOCKLO
            word(&[(0, 12, 0x919), (16, 8, 12), (72, 8, 0)]),
            word(&[(0, 12, 0x919), (16, 8, 13), (72, 8, 0x39)]),
            word(&[(0, 12, 0x805), (16, 8, 14), (72, 8, 0x50), (80, 1, 1)]),
            // LDC.U8 R16, c[0x3][R1+0x3]; ULDC.S16 UR4, c[0x3][0x6];
            // ULDC (indexed) UR5, c[0x3][UR4+0x10]
            word(&[
                (0, 12, 0xb82),
                (16, 8, 16),
                (24, 8, 1),
                (54, 5, 3),
                (38, 16, 3),
                (73, 3, 0),
            ]),
            word(&[
                (0, 12, 0xab9),
                (16, 6, 4),
                (54, 5, 3),
                (38, 16, 6),
                (73, 3, 3),
            ]),
            word(&[
                (0, 12, 0xabb),
                (16, 6, 5),
                (24, 6, 4),
                (54, 5, 3),
                (38, 16, 0x10),
                (73, 3, 4),
                (91, 1, 1),
            ]),
            EXIT,
        ]);
    }

    /// The directed-rounding steps the translation emits, in Rust: the
    /// nearest result, its residual's sign, and one step up or down.
    fn directed_model(value: f32, residual: f32, rounding: u8) -> f32 {
        let (down, up) = match rounding {
            1 => (residual < 0.0, false),
            2 => (false, residual > 0.0),
            _ => (value > 0.0 && residual < 0.0, value < 0.0 && residual > 0.0),
        };
        if !value.is_finite() {
            return value;
        }
        let bits = value.to_bits();
        let b = if bits == 0x8000_0000 { 0 } else { bits };
        let next_up = if b & 0x8000_0000 != 0 { b - 1 } else { b + 1 };
        let b = if bits == 0 { 0x8000_0000 } else { bits };
        let next_down = if b & 0x8000_0000 != 0 { b + 1 } else { b - 1 };
        f32::from_bits(if up {
            next_up
        } else if down {
            next_down
        } else {
            bits
        })
    }

    /// `exact` rounded to f32 by `rounding`, from f64 (exact for these
    /// operations' results on f32 inputs, sums and products).
    fn rounded_from(exact: f64, rounding: u8) -> f32 {
        let nearest = exact as f32;
        let n = f64::from(nearest);
        let step = |x: f32, up: bool| {
            let b = x.to_bits();
            f32::from_bits(match (x >= 0.0, up) {
                (_, _) if x == 0.0 => {
                    if up {
                        1
                    } else {
                        0x8000_0001
                    }
                }
                (true, true) | (false, false) => b + 1,
                _ => b - 1,
            })
        };
        match rounding {
            1 if n > exact => step(nearest, false),
            2 if n < exact => step(nearest, true),
            3 if n.abs() > exact.abs() => step(nearest, exact < 0.0),
            _ => nearest,
        }
    }

    #[test]
    fn directed_rounding_steps_match_exact_rounding() {
        let mut state = 0x1234_5678u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            // Moderate exponents, so f64 holds sums and products exactly.
            let mantissa = state & 0x007f_ffff;
            let exponent = 100 + (state >> 24) % 50;
            let sign = state & 0x8000_0000;
            f32::from_bits(sign | exponent << 23 | mantissa)
        };
        for _ in 0..20_000 {
            let (a, b) = (next(), next());
            for rounding in 1..=3 {
                // FADD: two-sum's error.
                let sum = a + b;
                let b_part = sum - a;
                let error = (a - (sum - b_part)) + (b - b_part);
                let got = directed_model(sum, error, rounding);
                let want = rounded_from(f64::from(a) + f64::from(b), rounding);
                assert_eq!(got.to_bits(), want.to_bits(), "{a} + {b} by {rounding}");
                // FMUL: the product's error by FMA.
                let product = a * b;
                let error = a.mul_add(b, -product);
                let got = directed_model(product, error, rounding);
                let want = rounded_from(f64::from(a) * f64::from(b), rounding);
                assert_eq!(got.to_bits(), want.to_bits(), "{a} * {b} by {rounding}");
            }
        }
    }

    #[test]
    fn warp_and_bit_operations_validate() {
        valid(&[
            // SHFL.BFLY PT, R2, R1, 0x1, 0x1f; SHFL.IDX P0, R3, R1, R4, R5
            word(&[
                (0, 12, 0xf89),
                (16, 8, 2),
                (24, 8, 1),
                (40, 13, 0x1f),
                (53, 5, 1),
                (58, 2, 3),
                (81, 3, 7),
            ]),
            word(&[
                (0, 12, 0x389),
                (16, 8, 3),
                (24, 8, 1),
                (32, 8, 4),
                (64, 8, 5),
                (81, 3, 0),
            ]),
            // SHFL.UP R6, R1, 0x2, RZ; SHFL.DOWN R7, R1, R4, 0x1f
            word(&[
                (0, 12, 0x989),
                (16, 8, 6),
                (24, 8, 1),
                (53, 5, 2),
                (58, 2, 1),
                (64, 8, 255),
                (81, 3, 7),
            ]),
            word(&[
                (0, 12, 0x589),
                (16, 8, 7),
                (24, 8, 1),
                (32, 8, 4),
                (40, 13, 0x1f),
                (58, 2, 2),
                (81, 3, 7),
            ]),
            // VOTE.ANY R8, P1, P0; VOTE.EQ RZ, P2, !P0
            word(&[
                (0, 12, 0x806),
                (16, 8, 8),
                (72, 2, 1),
                (81, 3, 1),
                (87, 3, 0),
            ]),
            word(&[
                (0, 12, 0x806),
                (16, 8, 255),
                (72, 2, 2),
                (81, 3, 2),
                (87, 3, 0),
                (90, 1, 1),
            ]),
            // POPC R9, ~R1; FLO.U32.SH R10, R1; FLO.S32 R11, R1; BREV R12, R1
            word(&[(0, 12, 0x309), (16, 8, 9), (32, 8, 1), (63, 1, 1)]),
            word(&[(0, 12, 0x300), (16, 8, 10), (32, 8, 1), (74, 1, 1)]),
            word(&[(0, 12, 0x300), (16, 8, 11), (32, 8, 1), (73, 1, 1)]),
            word(&[(0, 12, 0x301), (16, 8, 12), (32, 8, 1)]),
            EXIT,
        ]);
    }

    #[test]
    fn scales_wide_conversions_and_subgroup_reductions_validate() {
        valid(&[
            // FMUL.FTZ.D2 R2, R0, R1; FMUL.M8 R3, R0, 0.5
            word(&[
                (0, 12, 0x220),
                (16, 8, 2),
                (24, 8, 0),
                (32, 8, 1),
                (80, 1, 1),
                (84, 3, 3),
            ]),
            word(&[
                (0, 12, 0x820),
                (16, 8, 3),
                (24, 8, 0),
                (32, 32, 0x3f00_0000),
                (84, 3, 7),
            ]),
            // F2I.S64.TRUNC R4, R0; F2I.U16 R6, R0; F2I.S32.F16 R7, R1
            word(&[
                (0, 12, 0x305),
                (16, 8, 4),
                (32, 8, 0),
                (72, 1, 1),
                (75, 2, 3),
                (78, 2, 3),
                (84, 2, 2),
            ]),
            word(&[
                (0, 12, 0x305),
                (16, 8, 6),
                (32, 8, 0),
                (75, 2, 1),
                (84, 2, 2),
            ]),
            word(&[
                (0, 12, 0x305),
                (16, 8, 7),
                (32, 8, 1),
                (72, 1, 1),
                (75, 2, 2),
                (84, 2, 1),
            ]),
            // REDUX.SUM UR4, R1; REDUX.MAX.S32 UR5, R1
            word(&[(0, 12, 0x3c4), (16, 6, 4), (24, 8, 1), (78, 3, 3)]),
            word(&[
                (0, 12, 0x3c4),
                (16, 6, 5),
                (24, 8, 1),
                (73, 1, 1),
                (78, 3, 5),
            ]),
            // VOTEU.ANY UR6, UP1, P0; BREAK; WARPSYNC
            word(&[
                (0, 12, 0x886),
                (16, 8, 6),
                (72, 2, 1),
                (81, 3, 1),
                (87, 3, 0),
            ]),
            word(&[(0, 12, 0x942)]),
            word(&[(0, 12, 0x948)]),
            EXIT,
        ]);
    }

    /// A texture instruction word with a handle at c[0][handle_offset].
    fn tex(opcode: u128, fields: &[(u32, u32, u128)]) -> u128 {
        let mut all = vec![(0, 12, opcode), (54, 5, 0), (81, 3, 7)];
        all.extend_from_slice(fields);
        word(&all)
    }

    #[test]
    fn textures_are_combined_samplers_found_by_their_handle_word() {
        let translated = valid(&[
            // MOV R0, c[0x3][0x0] (a uniform buffer first in use)
            word(&[(0, 12, 0xa02), (72, 4, 0xf), (16, 8, 0), (54, 5, 3)]),
            // TEX.LZ R4, R6, R0, R1, 2D, 0xf (c[0][0x50]); split result
            tex(
                0xb60,
                &[
                    (16, 8, 4),
                    (24, 8, 0),
                    (32, 8, 1),
                    (40, 14, 0x14),
                    (61, 3, 1),
                    (64, 8, 6),
                    (72, 4, 0xf),
                    (87, 3, 1),
                ],
            ),
            // TEX.LL.DC R8, R0, R2, 2D, 0x1: (x, y | level, reference)
            tex(
                0xb60,
                &[
                    (16, 8, 8),
                    (24, 8, 0),
                    (32, 8, 2),
                    (40, 14, 0x14),
                    (61, 3, 1),
                    (64, 8, 255),
                    (72, 4, 1),
                    (78, 1, 1),
                    (87, 3, 3),
                ],
            ),
            // TEX R9, R0, R1, ARRAY_2D, 0x3 (layer, x | y); TEX CUBE.F16 R10
            tex(
                0xb60,
                &[
                    (16, 8, 9),
                    (24, 8, 0),
                    (32, 8, 1),
                    (40, 14, 0x16),
                    (61, 3, 5),
                    (64, 8, 255),
                    (72, 4, 3),
                ],
            ),
            tex(
                0xb60,
                &[
                    (16, 8, 10),
                    (24, 8, 0),
                    (32, 8, 2),
                    (40, 14, 0x18),
                    (61, 3, 3),
                    (64, 8, 255),
                    (72, 4, 0xf),
                    (79, 1, 1),
                ],
            ),
            // TLD.SCR.LL.AOFFI R12, R0, R2, 2D, 0xf: (x, y | level, offsets)
            tex(
                0xb66,
                &[
                    (16, 8, 12),
                    (24, 8, 0),
                    (32, 8, 2),
                    (40, 14, 0x14),
                    (61, 3, 1),
                    (64, 8, 14),
                    (72, 4, 0xf),
                    (76, 1, 1),
                    (87, 3, 3),
                ],
            ),
            // TLD4.G.AOFFI R16, R0, R2, 2D; TLD4.DC R20, R0, R2, 2D
            tex(
                0xb63,
                &[
                    (16, 8, 16),
                    (24, 8, 0),
                    (32, 8, 2),
                    (40, 14, 0x14),
                    (61, 3, 1),
                    (64, 8, 18),
                    (72, 4, 0xf),
                    (76, 1, 1),
                    (87, 2, 1),
                ],
            ),
            tex(
                0xb63,
                &[
                    (16, 8, 20),
                    (24, 8, 0),
                    (32, 8, 2),
                    (40, 14, 0x14),
                    (61, 3, 1),
                    (64, 8, 22),
                    (72, 4, 0xf),
                    (78, 1, 1),
                ],
            ),
            // TEX.LZ.AOFFI R28, R0, R2, 2D, 0x1: (x, y | offsets)
            tex(
                0xb60,
                &[
                    (16, 8, 28),
                    (24, 8, 0),
                    (32, 8, 2),
                    (40, 14, 0x14),
                    (61, 3, 1),
                    (64, 8, 255),
                    (72, 4, 1),
                    (76, 1, 1),
                    (87, 3, 1),
                ],
            ),
            // TLD4.PTP.G.F16 R30, R0, R2, 2D (bits 72-95 0x009eaf, as games
            // use it: per-tap offsets in R2, R3, half results)
            tex(
                0xb63,
                &[
                    (16, 8, 30),
                    (24, 8, 0),
                    (32, 8, 2),
                    (40, 14, 0x14),
                    (61, 3, 1),
                    (64, 8, 31),
                    (72, 24, 0x009eaf),
                ],
            ),
            // TXD R32, R0 (x, y), R8 (gradients), 2D, 0xf, second
            // destination R36
            tex(
                0xb6c,
                &[
                    (16, 8, 32),
                    (24, 8, 0),
                    (32, 8, 8),
                    (40, 14, 0x14),
                    (61, 3, 1),
                    (64, 8, 36),
                    (72, 4, 0xf),
                ],
            ),
            // TXQ R24, R1 (level), 0xf, c[0][0x50] (no fault predicate)
            word(&[
                (0, 12, 0xb6f),
                (16, 8, 24),
                (24, 8, 1),
                (40, 14, 0x14),
                (64, 8, 26),
                (72, 4, 0xf),
            ]),
            EXIT,
        ]);
        let summary: Vec<(u32, u32, u32, u32, u32)> = translated
            .descriptors
            .iter()
            .map(|d| {
                (
                    d.kind,
                    d.binding,
                    d.cbuf_offset,
                    d.texture_type,
                    d.flags & 8,
                )
            })
            .collect();
        assert_eq!(
            summary,
            [
                (0, 0, 0, 0, 0),
                (4, 1, 0x50, 2, 0),
                (4, 2, 0x50, 2, 8),
                (4, 3, 0x58, 3, 0),
                (4, 4, 0x60, 5, 0),
            ]
        );
    }

    #[test]
    fn computed_branches_and_swizzled_adds_validate() {
        valid(&[
            // MOV R0, 0x30; BRX R0 (to instruction 3); EXIT
            word(&[(0, 12, 0x802), (72, 4, 0xf), (16, 8, 0), (32, 32, 0x30)]),
            word(&[(0, 12, 0x949), (24, 8, 0)]),
            EXIT,
            // FSWZADD R2, R0, R1, 0x99 (lanes: a+b... per pattern); EXIT
            word(&[
                (0, 12, 0x822),
                (16, 8, 2),
                (24, 8, 0),
                (32, 8, 0x99),
                (64, 8, 1),
            ]),
            EXIT,
        ]);
    }

    #[test]
    fn bindless_handles_loaded_from_one_constant_word_bind_like_bound_ones() {
        let uldc = word(&[
            (0, 12, 0xab9),
            (16, 6, 4),
            (54, 5, 2),
            (38, 16, 0x28),
            (73, 3, 4),
        ]);
        // TEX R0, R2, R3, UR4, 2D, 0x1 (bindless: handle register 40-45)
        let bindless = word(&[
            (0, 12, 0xf60),
            (16, 8, 0),
            (24, 8, 2),
            (32, 8, 3),
            (40, 6, 4),
            (61, 3, 1),
            (64, 8, 255),
            (72, 4, 1),
            (81, 3, 7),
            (91, 1, 1),
        ]);
        let translated = valid(&[uldc, bindless, EXIT]);
        // The ULDC's bank is a uniform buffer; the texture follows it.
        let texture = translated.descriptors[1];
        assert_eq!(
            (texture.kind, texture.cbuf_index, texture.cbuf_offset),
            (4, 2, 0x28)
        );
        // A handle also written by other instructions is computed.
        let uiadd3 = word(&[
            (0, 12, 0x890),
            (16, 8, 4),
            (24, 8, 4),
            (32, 32, 8),
            (64, 8, 63),
            (91, 1, 1),
        ]);
        // A handle also written by other instructions is computed at run
        // time: it indexes the texture pool (set 1) instead.
        let computed = valid(&[uldc, uiadd3, bindless, EXIT]);
        assert!(computed.descriptors.iter().all(|d| d.kind != 4));
        // The 2D images (float and uint views), the samplers and the
        // integer flags.
        use echora_shader_abi::texture_pool::{FLAGS, SAMPLED_2D, SAMPLERS, UINT_2D};
        assert_eq!(
            computed.texture_pool_bindings,
            1 << SAMPLERS | 1 << SAMPLED_2D | 1 << UINT_2D | 1 << FLAGS
        );
        // The pool's capacity is a specialization constant.
        let constants: Vec<_> = computed
            .spec_constants
            .iter()
            .map(|constant| (constant.id, constant.meaning))
            .collect();
        assert_eq!(
            constants,
            [(
                echora_shader_abi::TEXTURE_POOL_CAPACITY_SPEC_ID,
                echora_shader_abi::SpecMeaning::TexturePoolCapacity
            )]
        );
    }

    /// Whether the module has an OpSwitch (the dispatch loop's).
    fn dispatches(spirv: &[u32]) -> bool {
        let mut at = 5;
        while at < spirv.len() {
            if spirv[at] & 0xffff == 251 {
                return true;
            }
            at += (spirv[at] >> 16).max(1) as usize;
        }
        false
    }

    /// ISETP.GT/LT.U32 p, PT, R(a), immediate, PT.
    fn isetp(p: u128, a: u128, compare: u128, immediate: u128) -> u128 {
        word(&[
            (0, 12, 0x80c),
            (81, 3, p),
            (84, 3, 7),
            (24, 8, a),
            (32, 32, immediate),
            (76, 3, compare),
            (87, 3, 7),
        ])
    }

    #[test]
    fn reducible_programs_are_structured() {
        let nop = word(&[(0, 12, 0x918)]);
        let mov = |d: u128, value: u128| {
            word(&[(0, 12, 0x802), (72, 4, 0xf), (16, 8, d), (32, 32, value)])
        };
        // A diamond: @P0 BRA 3; MOV; BRA 4; MOV; EXIT.
        let diamond = valid(&[bra(0, 3, 0), mov(1, 1), bra(2, 4, 7), mov(1, 2), EXIT]);
        assert!(!dispatches(&diamond.spirv));
        // Nested loops, the inner one leaving both: outer at 1, inner at 2,
        // @P0 BRA 9 out of both, @P1 BRA 2 (inner), @P2 BRA 9 (the outer
        // one leaving too, so 9 is a merge outside the inner loop), BRA 1.
        let iadd = word(&[
            (0, 12, 0x810),
            (16, 8, 1),
            (24, 8, 1),
            (32, 32, 1),
            (64, 8, 255),
            (77, 3, 7),
            (81, 3, 7),
            (84, 3, 7),
            (87, 3, 7),
        ]);
        let nested = valid(&[
            mov(0, 0),
            mov(1, 0),
            iadd,
            isetp(0, 1, 4, 5),
            bra(4, 9, 0),
            isetp(1, 1, 1, 3),
            bra(6, 2, 1),
            bra(7, 9, 2),
            bra(8, 1, 7),
            mov(2, 7),
            EXIT,
        ]);
        assert!(!dispatches(&nested.spirv));
        // Two entries into a loop: irreducible, so the dispatch loop.
        let irreducible = valid(&[bra(0, 2, 0), nop, bra(2, 1, 1), EXIT]);
        assert!(dispatches(&irreducible.spirv));
    }

    #[test]
    fn half_pair_operations_validate() {
        valid(&[
            // HADD2 R2, R0.H1_H0, -R1.H0_H0; HADD2.SAT R3, R0, 1.0 (form 2)
            word(&[
                (0, 12, 0x230),
                (16, 8, 2),
                (24, 8, 0),
                (32, 8, 1),
                (60, 2, 2),
                (63, 1, 1),
            ]),
            word(&[
                (0, 12, 0x430),
                (16, 8, 3),
                (24, 8, 0),
                (32, 32, 0x3c00_3c00),
                (77, 1, 1),
            ]),
            // HMUL2 R4, |R0|.H1_H1, R1; HFMA2 R5, R0, R1, R2.F32
            word(&[
                (0, 12, 0x232),
                (16, 8, 4),
                (24, 8, 0),
                (73, 1, 1),
                (74, 2, 3),
                (32, 8, 1),
            ]),
            word(&[
                (0, 12, 0x231),
                (16, 8, 5),
                (24, 8, 0),
                (32, 8, 1),
                (64, 8, 2),
                (81, 2, 1),
            ]),
            // HFMA2.RELU R10, R0, (1.0, 1.0), R2 (immediate B, always) and
            // with the clamp only where P1 holds
            word(&[
                (0, 12, 0x831),
                (16, 8, 10),
                (24, 8, 0),
                (32, 32, 0x3c00_3c00),
                (64, 8, 2),
                (79, 1, 1),
                (87, 3, 7),
            ]),
            word(&[
                (0, 12, 0x831),
                (16, 8, 11),
                (24, 8, 0),
                (32, 32, 0x3c00_3c00),
                (64, 8, 2),
                (79, 1, 1),
                (87, 3, 1),
            ]),
            // HMNMX2 R6, R0, R1, P0; F2FP.PACK_AB R7, R0, R1;
            // F2FP.MERGE_C R8, R1, R2
            word(&[
                (0, 12, 0x240),
                (16, 8, 6),
                (24, 8, 0),
                (32, 8, 1),
                (87, 3, 0),
            ]),
            // HADD2.F32 R9, -RZ, R1.H1_H1 (a half to a float)
            word(&[
                (0, 12, 0x230),
                (16, 8, 9),
                (24, 8, 255),
                (32, 8, 1),
                (60, 2, 3),
                (72, 1, 1),
                (78, 1, 1),
            ]),
            // HSETP2.GT.AND P1, P2, R0, R1.H1_H1, PT
            word(&[
                (0, 12, 0x234),
                (24, 8, 0),
                (32, 8, 1),
                (60, 2, 3),
                (76, 4, 4),
                (81, 3, 1),
                (84, 3, 2),
                (87, 3, 7),
            ]),
            // UBREV UR4, UR5; UFLO.U32 UR6, UR5; UPOPC UR7, UR5
            word(&[(0, 12, 0x2be), (16, 8, 4), (32, 8, 5), (91, 1, 1)]),
            word(&[(0, 12, 0x2bd), (16, 8, 6), (32, 8, 5), (91, 1, 1)]),
            word(&[(0, 12, 0x2bf), (16, 8, 7), (32, 8, 5), (91, 1, 1)]),
            word(&[(0, 12, 0x23e), (16, 8, 7), (24, 8, 0), (32, 8, 1)]),
            word(&[
                (0, 12, 0x23e),
                (16, 8, 8),
                (32, 8, 1),
                (64, 8, 2),
                (78, 1, 1),
            ]),
            EXIT,
        ]);
    }

    #[test]
    fn surface_reductions_use_an_r32ui_view() {
        let sured = |dimension: u128, operation: u128| {
            word(&[
                (0, 12, 0x59f),
                (16, 8, 255),
                (24, 8, 0),
                (32, 8, 4),
                (40, 14, 0x18),
                (54, 5, 0),
                (61, 3, dimension),
                (77, 3, 7),
                (84, 1, 1),
                (87, 3, operation),
            ])
        };
        // SURED.D.ADD.U32 into a 2D image twice (one R32ui view), and
        // SURED.D.MAX.U32 into a buffer.
        let translated = valid(&[sured(3, 0), sured(3, 6), sured(1, 2), EXIT]);
        let flags: Vec<u32> = translated.descriptors.iter().map(|d| d.flags).collect();
        assert_eq!(flags, [0x503, 0x503]);
        assert!(translated.spec_constants.is_empty());
        // The wrapping increment is refused, as is an unknown mode bit.
        translated_err(&[sured(3, 3), EXIT]);
        assert!(matches!(
            crate::decode::decode(sured(3, 0) | 1 << 72).op,
            crate::decode::Op::Unknown { .. }
        ));
    }

    #[test]
    fn storage_images_have_float_and_unsigned_views_picked_by_a_constant() {
        let translated = valid(&[
            // SULD.P.2D R4, [R0], 0x3, c[0][0x60]; SUST.P.2D [R0], R4, 0xf
            word(&[
                (0, 12, 0xb97),
                (16, 8, 4),
                (24, 8, 0),
                (40, 14, 0x18),
                (54, 5, 0),
                (61, 3, 3),
                (72, 4, 3),
                (81, 3, 7),
            ]),
            word(&[
                (0, 12, 0xb9b),
                (24, 8, 0),
                (32, 8, 4),
                (40, 14, 0x18),
                (54, 5, 0),
                (61, 3, 3),
                (72, 4, 0xf),
            ]),
            // SULD.P.BUFFER R8, [R1], 0x1, c[0][0x68]
            word(&[
                (0, 12, 0xb97),
                (16, 8, 8),
                (24, 8, 1),
                (40, 14, 0x1a),
                (54, 5, 0),
                (61, 3, 1),
                (72, 4, 1),
                (81, 3, 7),
            ]),
            EXIT,
        ]);
        let summary: Vec<(u32, u32, u32, u32, u32)> = translated
            .descriptors
            .iter()
            .map(|d| (d.kind, d.binding, d.cbuf_offset, d.texture_type, d.flags))
            .collect();
        // The image buffer (kind 3) numbers before the image (kind 5).
        assert_eq!(summary, [(3, 0, 0x68, 7, 2), (5, 1, 0x60, 2, 3)]);
        // The 2D image's constant (id 32) at binding 1, the buffer's (33)
        // at binding 0.
        let constants: Vec<_> = translated
            .spec_constants
            .iter()
            .map(|constant| (constant.id, constant.meaning))
            .collect();
        use echora_shader_abi::SpecMeaning::IntegerFormat;
        assert_eq!(constants, [(33, IntegerFormat(0)), (32, IntegerFormat(1))]);
    }

    #[test]
    fn matrix_loads_shuffle_row_addresses_and_read_shared_memory() {
        let words = [
            // LDSM.16.M88.4 R4, [R0+UR4+0x10]; LDSM.16.MT88.2 R8, [R1]
            word(&[
                (0, 12, 0x83b),
                (16, 8, 4),
                (24, 8, 0),
                (32, 6, 4),
                (40, 24, 0x10),
                (72, 2, 2),
                (91, 1, 1),
            ]),
            word(&[
                (0, 12, 0x83b),
                (16, 8, 8),
                (24, 8, 1),
                (72, 2, 1),
                (78, 1, 1),
            ]),
            EXIT,
        ];
        let layout = Layout {
            stage: Stage::Compute,
            header: None,
            workgroup: [32, 1, 1],
            shared_bytes: Some(0x2000),
            local_bytes: None,
            matrices: MatrixPath::Subgroup,
            mesh: None,
            loop_trace: false,
            vertex_trace: false,
            float64: false,
            dual_source_blend: false,
        };
        valid_with(&words, &layout);
    }

    #[test]
    fn packed_float_to_byte_conversions_validate() {
        // F2IP.U8.F32 R2, R5, R6, RZ (nearest); F2IP.U8.F32.TRUNC R3, RZ, R6, RZ
        valid(&[
            word(&[
                (0, 12, 0x243),
                (16, 8, 2),
                (24, 8, 5),
                (32, 8, 6),
                (64, 8, 255),
                (74, 1, 1),
            ]),
            word(&[
                (0, 12, 0x243),
                (16, 8, 3),
                (24, 8, 255),
                (32, 8, 6),
                (64, 8, 255),
                (74, 1, 1),
                (78, 2, 3),
            ]),
            EXIT,
        ]);
        // A third source is refused.
        assert!(matches!(
            crate::decode::decode(word(&[(0, 12, 0x243), (32, 8, 6), (64, 8, 7), (74, 1, 1)])).op,
            crate::decode::Op::Unknown { .. }
        ));
    }

    #[test]
    fn matrix_transposes_validate() {
        // MOVM.16.MT88 R2, R5; MOVM.16.MT88 RZ, R1
        valid(&[
            word(&[(0, 12, 0x23a), (16, 8, 2), (24, 8, 5)]),
            word(&[(0, 12, 0x23a), (16, 8, 255), (24, 8, 1)]),
            EXIT,
        ]);
        // Another mode bit is refused.
        assert!(matches!(
            crate::decode::decode(word(&[(0, 12, 0x23a), (24, 8, 5), (72, 1, 1)])).op,
            crate::decode::Op::Unknown { .. }
        ));
    }

    #[test]
    fn warp_matrix_multiplies_validate() {
        let words = warp_matrix_multiplies();
        valid(&words);
        let exact = Layout {
            matrices: MatrixPath::Exact,
            ..Layout::workgroup([64, 1, 1])
        };
        valid_with(&words, &exact);
    }

    fn warp_matrix_multiplies() -> [u128; 4] {
        [
            // HMMA.16816.F32 R8, R0, R4, R8; HMMA.16816.F16 R12, R0, R4, R14;
            // HMMA.1688.F32 R16, R0, R4, RZ
            word(&[
                (0, 12, 0x23c),
                (16, 8, 8),
                (24, 8, 0),
                (32, 8, 4),
                (64, 8, 8),
                (75, 1, 1),
                (76, 1, 1),
            ]),
            word(&[
                (0, 12, 0x23c),
                (16, 8, 12),
                (24, 8, 0),
                (32, 8, 4),
                (64, 8, 14),
                (75, 1, 1),
            ]),
            word(&[
                (0, 12, 0x23c),
                (16, 8, 16),
                (24, 8, 0),
                (32, 8, 4),
                (64, 8, 255),
                (76, 1, 1),
            ]),
            EXIT,
        ]
    }

    #[test]
    fn warp_matrix_multiplies_use_cooperative_matrices_when_chosen() {
        let words = [
            // HMMA.16816.F16 R8, R0, R4, R8 then HMMA.16816.F32 R12, R0, R4, R12
            word(&[
                (0, 12, 0x23c),
                (16, 8, 8),
                (24, 8, 0),
                (32, 8, 4),
                (64, 8, 8),
                (75, 1, 1),
            ]),
            word(&[
                (0, 12, 0x23c),
                (16, 8, 12),
                (24, 8, 0),
                (32, 8, 4),
                (64, 8, 12),
                (75, 1, 1),
                (76, 1, 1),
            ]),
            EXIT,
        ];
        for matrices in [MatrixPath::Cooperative, MatrixPath::Cooperative32] {
            let layout = Layout {
                workgroup: [32, 2, 1],
                matrices,
                mesh: None,
                ..Layout::workgroup([32, 2, 1])
            };
            valid_with(&words, &layout);
        }
    }

    #[test]
    fn cooperative_matrix_multiplies_pair_hmmas_that_share_a() {
        // HMMA.16816.F32 d, R0, b, c: words with those registers.
        let hmma = |d: u128, b: u128, c: u128| {
            word(&[
                (0, 12, 0x23c),
                (16, 8, d),
                (24, 8, 0),
                (32, 8, b),
                (64, 8, c),
                (75, 1, 1),
                (76, 1, 1),
            ])
        };
        for matrices in [MatrixPath::Cooperative, MatrixPath::Cooperative32] {
            let layout = Layout {
                matrices,
                ..Layout::workgroup([32, 1, 1])
            };
            let multiplies = |words: &[u128]| {
                let spirv = valid_with(words, &layout).spirv;
                let mut count = 0;
                let mut at = 5;
                while at < spirv.len() {
                    if spirv[at] & 0xffff == u32::from(op::COOPERATIVE_MATRIX_MUL_ADD) {
                        count += 1;
                    }
                    at += ((spirv[at] >> 16) as usize).max(1);
                }
                count
            };
            // Columns 0-7 and 8-15 of one product: one multiply.
            assert_eq!(multiplies(&[hmma(8, 4, 8), hmma(12, 6, 12), EXIT]), 1);
            // The second reads the first's result: two.
            assert_eq!(multiplies(&[hmma(8, 4, 8), hmma(16, 8, 16), EXIT]), 2);
        }
    }

    #[test]
    fn a_masked_move_writes_only_its_quad_lanes() {
        // MOV R1, 0x7 to quad lanes 0 and 2 only.
        let masked = word(&[(0, 12, 0x802), (16, 8, 1), (32, 32, 7), (72, 4, 0x5)]);
        assert!(matches!(decode(masked).op, Op::Mov { lanes: 0x5, .. }));
        valid(&[masked, EXIT]);
    }

    #[test]
    fn bindless_textures_with_a_register_handle_index_the_pool_non_uniformly() {
        // TEX.B R4, R0 (x, y), R2 (handle), 2D, 0xf; unpacked (bit 60 clear)
        let translated = valid(&[
            word(&[
                (0, 12, 0x361),
                (16, 8, 4),
                (24, 8, 0),
                (32, 8, 2),
                (59, 1, 1),
                (61, 3, 1),
                (64, 8, 255),
                (72, 4, 0xf),
                (81, 3, 7),
            ]),
            EXIT,
        ]);
        use echora_shader_abi::texture_pool::{FLAGS, SAMPLED_2D, SAMPLERS, UINT_2D};
        assert_eq!(
            translated.texture_pool_bindings,
            1 << SAMPLERS | 1 << SAMPLED_2D | 1 << UINT_2D | 1 << FLAGS
        );
        // NonUniform (5300) decorates the register handle's accesses.
        assert!(
            translated
                .spirv
                .windows(3)
                .any(|w| w[0] == (3 << 16 | 71) && w[2] == 5300)
        );
    }

    #[test]
    fn byte_and_half_word_stores_keep_their_neighbours() {
        let words = [
            // STG.E.U16 [R4.64+0x2], R2; STS.U8 [R1+0x3], R2; STL.U16 [R1], R2
            word(&[
                (0, 12, 0x386),
                (24, 8, 4),
                (32, 8, 2),
                (40, 24, 2),
                (72, 1, 1),
                (73, 3, 2),
            ]),
            word(&[
                (0, 12, 0x388),
                (24, 8, 1),
                (32, 8, 2),
                (40, 24, 3),
                (73, 3, 0),
            ]),
            word(&[(0, 12, 0x387), (24, 8, 1), (32, 8, 2), (73, 3, 2)]),
            EXIT,
        ];
        let layout = Layout {
            shared_bytes: Some(0x100),
            local_bytes: Some(0x40),
            ..Layout::workgroup([64, 1, 1])
        };
        valid_with(&words, &layout);
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
            word(&[(0, 12, 0x802), (72, 4, 0xf), (16, 8, 1), (32, 32, 0)]),
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
        // FMUL's result scale 0 is invalid.
        let nop = word(&[(0, 12, 0x918)]);
        let scaled = translated(&[nop, word(&[(0, 12, 0x220), (84, 3, 0)]), EXIT]).unwrap_err();
        assert_eq!(scaled.offset, Some(16));
        let outside = translated(&[bra(0, 50, 7), EXIT]).unwrap_err();
        assert!(outside.reason.contains("outside"), "{}", outside.reason);
    }
}
