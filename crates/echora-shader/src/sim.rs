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

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::spirv::{execution, glsl, op, storage};

/// Opcodes not in `op`.
const F_CONVERT: u16 = 115;
const VECTOR_SHUFFLE: u16 = 79;
const I_ADD_CARRY: u16 = 149;
const I_SUB_BORROW: u16 = 150;
const U_MUL_EXTENDED: u16 = 151;
const S_MUL_EXTENDED: u16 = 152;

/// Decorations read here.
const DECORATION_SPEC_ID: u32 = 1;
const DECORATION_ARRAY_STRIDE: u32 = 6;
const DECORATION_BUILT_IN: u32 = 11;
const DECORATION_LOCATION: u32 = 30;
/// OpCopyObject (inlined calls deliver their results with it).
const COPY_OBJECT: u16 = 83;
const DECORATION_COMPONENT: u32 = 31;
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
    Int {
        width: u32,
        signed: bool,
    },
    Float {
        width: u32,
    },
    Vector {
        component: u32,
        count: u32,
    },
    Array {
        element: u32,
        length: u32,
    },
    RuntimeArray {
        element: u32,
    },
    Struct {
        members: Vec<u32>,
    },
    Pointer {
        pointee: u32,
    },
    /// A cooperative matrix: its component type, rows and columns.
    Matrix {
        component: u32,
        rows: u32,
        columns: u32,
    },
    /// An image (dim 0 1D, 1 2D, 2 3D, 3 cube, 5 buffer), a sampled image
    /// of one, a sampler.
    Image {
        dim: u32,
        arrayed: bool,
    },
    SampledImage {
        image: u32,
    },
    Sampler,
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

#[derive(Clone, Debug)]
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
    /// Bound textures and images by (descriptor set, binding).
    pub images: HashMap<(u32, u32), Image>,
    /// Elements of image arrays (the texture pool) by (descriptor set,
    /// binding, element), sampled and queried only.
    pub image_arrays: HashMap<(u32, u32, u32), Image>,
    /// Specialization constants' values by SpecId (else their defaults).
    pub specialization: HashMap<u32, u32>,
    /// Iterations of one loop by one invocation that count as running
    /// away (1 << 16 when 0).
    pub loop_limit: u32,
    /// More built-in inputs, by BuiltIn (e.g. 15, FragCoord, as f32 bits),
    /// for every invocation: what a fragment program reads of its pixel.
    pub builtins: HashMap<u32, Vec<u64>>,
    /// Input variables' values by Location (a fragment program's
    /// interpolated attributes, as 32-bit words).
    pub locations: HashMap<u32, Vec<u64>>,
    /// A mesh program's task payload (bytes) from its task program.
    pub payload: Vec<u8>,
    /// Bound textures' levels past 0 (1, 2, ...), by (descriptor set,
    /// binding): explicit, gradient and fetch levels read them.
    pub mips: HashMap<(u32, u32), Vec<Image>>,
    /// Depth comparisons' VkCompareOp (the sampler's; 0 never, 1 less, 2
    /// equal, 3 less or equal, 4 greater, 5 not equal, 6 greater or equal,
    /// 7 always): `None` takes less or equal, with a note.
    pub compare_op: Option<u32>,
    /// Address ranges (device addresses, or guest addresses through the
    /// page table) whose stores and atomics `Report::writes` lists.
    pub watch: Vec<std::ops::Range<u64>>,
    /// An invocation (local index) and a variable (a register's id) whose
    /// every store `Report::traced` lists.
    pub trace: Option<(usize, u32)>,
}

/// A store or atomic into a watched range.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Write {
    /// The invocation (local index) and the instruction (index in the
    /// entry function, as `Report::ends`) that wrote, and its opcode.
    pub invocation: usize,
    pub instruction: usize,
    pub opcode: u16,
    /// The device address, and the guest address when the page table
    /// maps the page.
    pub device: u64,
    pub guest: Option<u64>,
    /// The bytes before and after.
    pub old: Vec<u8>,
    pub new: Vec<u8>,
}

/// Writes past this many are not listed (a note says so).
const WRITE_LIMIT: usize = 1 << 20;

/// A texture's or image's level 0: `layers` layers (or a 3D image's
/// `depth` slices) of `width` x `height` texels, each the four components
/// as the shader reads them (float bits, or integers).
#[derive(Clone, Debug, Default)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub layers: u32,
    pub texels: Vec<[u32; 4]>,
}

impl Image {
    fn slices(&self) -> u32 {
        self.depth.max(1) * self.layers.max(1)
    }

    fn index(&self, x: usize, y: usize, z: usize) -> usize {
        (z * self.height.max(1) as usize + y) * self.width.max(1) as usize + x
    }

    /// The texel at (x, y, z), each clamped to the image.
    fn texel(&self, x: i64, y: i64, z: i64) -> [u32; 4] {
        let clamp = |value: i64, size: u32| value.clamp(0, i64::from(size.max(1)) - 1) as usize;
        let (x, y, z) = (
            clamp(x, self.width),
            clamp(y, self.height),
            clamp(z, self.slices()),
        );
        self.texels
            .get(self.index(x, y, z))
            .copied()
            .unwrap_or([0, 0, 0, 0])
    }

    fn texel_mut(&mut self, x: i64, y: i64, z: i64) -> Option<&mut [u32; 4]> {
        let inside = |value: i64, size: u32| (0..i64::from(size.max(1))).contains(&value);
        if !(inside(x, self.width) && inside(y, self.height) && inside(z, self.slices())) {
            return None;
        }
        let at = self.index(x as usize, y as usize, z as usize);
        self.texels.get_mut(at)
    }
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
    /// The images after the run.
    pub images: HashMap<(u32, u32), Image>,
    /// The uniform buffer bytes the run read: (descriptor set, binding,
    /// byte offset), each a 4-byte word.
    pub uniform_reads: std::collections::BTreeSet<(u32, u32, u32)>,
    /// Guest pages accessed that the page table does not map (outside it,
    /// or a zero entry), with how many accesses: the translation reads 0
    /// from them and drops stores, as on the GPU.
    pub unmapped_pages: BTreeMap<u64, u32>,
    /// Where each invocation ended: the index of the instruction (in the
    /// entry function) it returned at, or stopped at on a problem.
    pub ends: Vec<usize>,
    /// The stores and atomics into `Inputs::watch`, in the order they ran.
    pub writes: Vec<Write>,
    /// A mesh program's OpSetMeshOutputsEXT (vertices, primitives), and a
    /// task program's OpEmitMeshTasksEXT group counts.
    pub mesh_outputs: Option<(u32, u32)>,
    pub mesh_tasks: Option<[u32; 3]>,
    /// The workgroup's own outputs after the run, by variable id: a mesh
    /// program's output arrays (shared by the workgroup), a task
    /// program's payload.
    pub workgroup_outputs: HashMap<u32, Value>,
    /// Each invocation's guest EXIT (byte offset), the first it took; none
    /// when it ran off the end or stopped on a problem.
    pub exits: Vec<Option<u32>>,
    /// Each invocation's first NaN float result: the guest instruction
    /// (byte offset) it came from, when the translation marks them
    /// (ECHORA_GUEST_LINES), the SPIR-V opcode and the instruction's index
    /// in the entry function.
    pub first_nan: Vec<Option<(Option<u32>, u16, usize)>>,
    /// Likewise its first infinite one (often legitimate: a reciprocal of
    /// zero later selected away or multiplied by zero with .DNZ).
    pub first_infinite: Vec<Option<(Option<u32>, u16, usize)>>,
    /// The stores `Inputs::trace` names, in order: the guest instruction
    /// (byte offset, when marked) and the value stored.
    pub traced: Vec<(Option<u32>, Value)>,
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
    /// SpecIds by constant.
    spec_ids: HashMap<u32, u32>,
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
    /// Debug names (OpName) and the ids they name: the translation names
    /// its register variables R0.., P0.., UR0.., UP0...
    pub names: BTreeMap<String, u32>,
    /// Input and output variables' Locations (a fragment program's
    /// attributes and colours), by variable id.
    pub locations: HashMap<u32, u32>,
    /// The entry point's execution model.
    model: u32,
    /// Their first Components (a scalar input of one component of a
    /// location), by variable id.
    components: HashMap<u32, u32>,
    /// The OpString of the translation's guest instruction lines.
    step_file: Option<u32>,
}

/// One invocation's state.
struct Invocation {
    /// Its local index.
    number: usize,
    at: usize,
    /// The instruction running (`at` moves on before it runs).
    current: usize,
    /// The guest EXIT it took first (an OpLine's line), if any.
    exit: Option<u32>,
    /// The guest instruction running (its line), when marked.
    step: Option<u32>,
    /// The first NaN and infinite float results (see
    /// `Report::first_nan`).
    first_nan: Option<(Option<u32>, u16, usize)>,
    first_infinite: Option<(Option<u32>, u16, usize)>,
    values: HashMap<u32, Value>,
    value_types: HashMap<u32, u32>,
    pointers: HashMap<u32, Pointer>,
    /// Its own variables (function, private, input, output).
    variables: HashMap<u32, Value>,
    /// Loop headers passed, by instruction.
    loops: HashMap<usize, u32>,
    /// Values that are the page table's page count (for page misses).
    page_counts: HashSet<u32>,
    /// Values, and variables holding them, that come from a shuffle of a
    /// lane not running there (undefined on a GPU): a branch on one is
    /// reported.
    tainted: HashSet<u32>,
    tainted_variables: HashSet<u32>,
    /// Starting values of function variables, replacing their
    /// initializers.
    initial: HashMap<u32, Value>,
    done: bool,
    steps: u64,
}

/// What a workgroup shares: its shared, uniform and push-constant
/// variables, device memory and the problems found.
struct Shared {
    variables: HashMap<u32, Value>,
    memory: BTreeMap<u64, Vec<u8>>,
    images: HashMap<(u32, u32), Image>,
    image_arrays: HashMap<(u32, u32, u32), Image>,
    problems: Vec<String>,
    notes: Vec<String>,
    loop_limit: u32,
    /// Uniform buffer words read: (set, binding, byte offset).
    uniform_reads: std::collections::BTreeSet<(u32, u32, u32)>,
    /// The uniform buffer variables, with their set and binding.
    uniforms: HashMap<u32, (u32, u32)>,
    /// The page table from the push constants: its address, first page and
    /// page count.
    page_table: Option<(u64, u64, u64)>,
    unmapped_pages: BTreeMap<u64, u32>,
    compare_op: Option<u32>,
    mips: HashMap<(u32, u32), Vec<Image>>,
    watch: Vec<std::ops::Range<u64>>,
    trace: Option<(usize, u32)>,
    traced: Vec<(Option<u32>, Value)>,
    /// Guest pages by the device page they map to (when watching).
    guest_pages: HashMap<u64, u64>,
    writes: Vec<Write>,
    mesh_outputs: Option<(u32, u32)>,
    mesh_tasks: Option<[u32; 3]>,
}

impl Shared {
    fn note(&mut self, note: &str) {
        if !self.notes.iter().any(|known| known == note) {
            self.notes.push(note.to_string());
        }
    }

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

    /// The guest address `address` (a device address) is mapped from.
    fn guest(&self, address: u64) -> Option<u64> {
        use echora_shader_abi::PAGE_SHIFT;
        let page = self.guest_pages.get(&(address >> PAGE_SHIFT))?;
        Some(page << PAGE_SHIFT | address & ((1 << PAGE_SHIFT) - 1))
    }

    /// Lists a write of `new` at `address` when it is watched.
    fn watch_write(&mut self, at: (usize, usize, u16), address: u64, new: &[u8]) {
        let guest = self.guest(address);
        let end = new.len() as u64;
        let watched = |start: u64| {
            self.watch
                .iter()
                .any(|range| start < range.end && range.start < start + end)
        };
        if !(watched(address) || guest.is_some_and(watched)) {
            return;
        }
        if self.writes.len() == WRITE_LIMIT {
            self.note("more watched writes than are listed");
            return;
        }
        let old = match self.region(address, new.len()) {
            Some((base, offset)) => self.memory[&base][offset..offset + new.len()].to_vec(),
            None => Vec::new(),
        };
        self.writes.push(Write {
            invocation: at.0,
            instruction: at.1,
            opcode: at.2,
            device: address,
            guest,
            old,
            new: new.to_vec(),
        });
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
            spec_ids: HashMap::new(),
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
            names: BTreeMap::new(),
            locations: HashMap::new(),
            components: HashMap::new(),
            model: 0,
            step_file: None,
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
                op::STRING if operands.get(1) == Some(&u32::from_le_bytes(*b"step")) => {
                    machine.step_file = Some(operands[0]);
                }
                op::NAME if !operands.is_empty() => {
                    let bytes: Vec<u8> = operands[1..]
                        .iter()
                        .flat_map(|word| word.to_le_bytes())
                        .take_while(|&byte| byte != 0)
                        .collect();
                    machine
                        .names
                        .insert(String::from_utf8_lossy(&bytes).into_owned(), operands[0]);
                }
                op::ENTRY_POINT => {
                    machine.model = operands[0];
                    entry = operands[1];
                }
                op::EXECUTION_MODE if operands[1] == EXECUTION_LOCAL_SIZE => {
                    machine.local_size = [operands[2], operands[3], operands[4]];
                }
                op::DECORATE => {
                    let (target, value) = (operands[0], operands.get(2).copied().unwrap_or(0));
                    match operands[1] {
                        DECORATION_BUILT_IN => {
                            machine.builtins.insert(target, value);
                        }
                        DECORATION_SPEC_ID => {
                            machine.spec_ids.insert(target, value);
                        }
                        DECORATION_DESCRIPTOR_SET => {
                            machine.sets.insert(target, value);
                        }
                        DECORATION_BINDING => {
                            machine.bindings.insert(target, value);
                        }
                        DECORATION_LOCATION => {
                            machine.locations.insert(target, value);
                        }
                        DECORATION_COMPONENT => {
                            machine.components.insert(target, value);
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
                    let rows = machine.constants[&operands[3]][0] as u32;
                    let columns = machine.constants[&operands[4]][0] as u32;
                    machine.types.insert(
                        operands[0],
                        Ty::Matrix {
                            component: operands[1],
                            rows,
                            columns,
                        },
                    );
                }
                op::TYPE_IMAGE => {
                    machine.types.insert(
                        operands[0],
                        Ty::Image {
                            dim: operands[2],
                            arrayed: operands[4] != 0,
                        },
                    );
                }
                op::TYPE_SAMPLED_IMAGE => {
                    machine
                        .types
                        .insert(operands[0], Ty::SampledImage { image: operands[1] });
                }
                op::TYPE_SAMPLER => {
                    machine.types.insert(operands[0], Ty::Sampler);
                }
                op::TYPE_FUNCTION => {
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
        // Calls run inline: the callee's instructions in place of the call
        // (its labels renamed per copy, its parameters replaced by the
        // arguments), each return a branch to a label after them (a
        // returned value copied into the call's result first).
        let callees: HashMap<u32, Vec<Instruction>> = functions
            .iter()
            .map(|(id, body)| (*id, body.clone()))
            .collect();
        let (_, mut body) = functions
            .into_iter()
            .find(|(id, _)| *id == entry)
            .expect("the entry function");
        let mut next_label = u32::MAX;
        for _ in 0..8 {
            if !body.iter().any(|i| i.opcode == op::FUNCTION_CALL) {
                break;
            }
            let mut inlined = Vec::with_capacity(body.len());
            for instruction in body {
                let callee = (instruction.opcode == op::FUNCTION_CALL)
                    .then(|| callees.get(&instruction.operands[2]))
                    .flatten();
                let Some(callee) = callee else {
                    inlined.push(instruction);
                    continue;
                };
                let (result_type, result) = (instruction.operands[0], instruction.operands[1]);
                let arguments = &instruction.operands[3..];
                let after = next_label;
                next_label -= 1;
                // This copy's names: its labels fresh, its parameters the
                // arguments.
                let mut names: HashMap<u32, u32> = HashMap::new();
                let mut parameter = 0;
                for inner in callee {
                    match inner.opcode {
                        op::LABEL => {
                            names.insert(inner.operands[0], next_label);
                            next_label -= 1;
                        }
                        op::FUNCTION_PARAMETER => {
                            names.insert(inner.operands[1], arguments[parameter]);
                            parameter += 1;
                        }
                        _ => {}
                    }
                }
                let rename = |id: u32| names.get(&id).copied().unwrap_or(id);
                for inner in callee {
                    let operands: Vec<u32> = match inner.opcode {
                        op::FUNCTION_PARAMETER => continue,
                        // Labels and ids: every operand renamed (labels and
                        // parameters are ids no literal here takes).
                        _ => inner.operands.iter().map(|&o| rename(o)).collect(),
                    };
                    match inner.opcode {
                        op::RETURN => inlined.push(Instruction {
                            opcode: op::BRANCH,
                            operands: vec![after],
                        }),
                        op::RETURN_VALUE => {
                            inlined.push(Instruction {
                                opcode: COPY_OBJECT,
                                operands: vec![result_type, result, operands[0]],
                            });
                            inlined.push(Instruction {
                                opcode: op::BRANCH,
                                operands: vec![after],
                            });
                        }
                        opcode => inlined.push(Instruction { opcode, operands }),
                    }
                }
                inlined.push(Instruction {
                    opcode: op::LABEL,
                    operands: vec![after],
                });
            }
            body = inlined;
        }
        for (n, instruction) in body.iter().enumerate() {
            if instruction.opcode == op::LABEL {
                machine.labels.insert(instruction.operands[0], n);
            }
        }
        machine.body = body;
        machine
    }

    /// The level an image access reads: a query's or a fetch's level, a
    /// sample's explicit Lod or the level its Grad operands give (log2 of
    /// the larger gradient's length in level-0 texels), rounded to the
    /// nearest (mipmapMode nearest); 0 otherwise, an implicit level
    /// included (its derivatives are not simulated).
    fn image_level(
        &self,
        opcode: u16,
        invocation: &Invocation,
        args: &[u32],
        base: &Image,
    ) -> usize {
        let value = |n: usize| args.get(n).and_then(|id| invocation.values.get(id));
        let float = |words: &Value, n: usize| {
            f64::from(f32::from_bits(words.get(n).copied().unwrap_or(0) as u32))
        };
        let from_lambda = |lambda: f64| (lambda + 0.5).floor().max(0.0) as usize;
        match opcode {
            op::IMAGE_QUERY_SIZE_LOD => value(1).map_or(0, |words| words[0] as usize),
            op::IMAGE_FETCH | op::IMAGE_SAMPLE_EXPLICIT_LOD if args.len() > 2 => {
                let mask = args[2];
                let mut at = 3;
                if mask & 1 != 0 {
                    at += 1;
                }
                if mask & 2 != 0 {
                    let Some(lod) = value(at) else { return 0 };
                    return if opcode == op::IMAGE_FETCH {
                        lod[0] as usize
                    } else {
                        from_lambda(float(lod, 0))
                    };
                }
                if mask & 4 != 0
                    && let (Some(dx), Some(dy)) = (value(at), value(at + 1))
                {
                    let sizes = [f64::from(base.width.max(1)), f64::from(base.height.max(1))];
                    let length = |gradient: &Value| {
                        (0..gradient.len().min(2))
                            .map(|n| (float(gradient, n) * sizes[n]).powi(2))
                            .sum::<f64>()
                            .sqrt()
                    };
                    let rho = length(dx).max(length(dy));
                    return if rho > 0.0 {
                        from_lambda(rho.log2())
                    } else {
                        0
                    };
                }
                0
            }
            _ => 0,
        }
    }

    /// A gather's texel offset: its image operands after the component or
    /// reference (Bias, Lod, Grad (two), then ConstOffset or Offset).
    fn gather_offset(&self, invocation: &Invocation, args: &[u32]) -> [i64; 2] {
        let mut offset = [0i64; 2];
        if args.len() > 3 {
            let mask = args[3];
            let mut at = 4;
            for (bit, count) in [(1u32, 1usize), (2, 1), (4, 2)] {
                if mask & bit != 0 {
                    at += count;
                }
            }
            if mask & (0x8 | 0x10) != 0
                && let Some(words) = args.get(at).and_then(|id| invocation.values.get(id))
            {
                for (n, slot) in offset.iter_mut().enumerate() {
                    *slot = i64::from(words.get(n).copied().unwrap_or(0) as u32 as i32);
                }
            }
        }
        offset
    }

    /// The BuiltIn a variable is decorated with.
    pub fn builtin(&self, variable: u32) -> Option<u32> {
        self.builtins.get(&variable).copied()
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
            Ty::Matrix { rows, columns, .. } => (rows * columns) as usize,
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
    /// past their end; a runtime array as long as they reach).
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
            // As many elements as the bytes hold.
            Ty::RuntimeArray { element } => {
                let stride = self.stride(ty, element).max(1);
                for n in 0..bytes.len().saturating_sub(at).div_ceil(stride) {
                    self.fill(element, bytes, at + n * stride, slots);
                }
            }
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
        self.run_group(
            count,
            subgroup,
            builtins,
            |_| HashMap::new(),
            &Inputs::default(),
        )
    }

    /// As `run_workgroup`, invocation n's function variables starting
    /// with the values `initial(n)` gives (by variable id) instead of
    /// their initializers.
    #[cfg(test)]
    fn run_workgroup_with(
        &self,
        count: usize,
        subgroup: usize,
        builtins: impl Fn(usize) -> HashMap<u32, Value>,
        initial: impl Fn(usize) -> HashMap<u32, Value>,
    ) -> Report {
        self.run_group(count, subgroup, builtins, initial, &Inputs::default())
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
                builtins.extend(inputs.builtins.clone());
                builtins
            },
            |_| HashMap::new(),
            inputs,
        )
    }

    fn run_group(
        &self,
        count: usize,
        subgroup: usize,
        builtins: impl Fn(usize) -> HashMap<u32, Value>,
        initial: impl Fn(usize) -> HashMap<u32, Value>,
        inputs: &Inputs,
    ) -> Report {
        let mut shared = Shared {
            variables: HashMap::new(),
            memory: inputs.memory.clone(),
            images: inputs.images.clone(),
            image_arrays: inputs.image_arrays.clone(),
            problems: Vec::new(),
            notes: Vec::new(),
            uniform_reads: std::collections::BTreeSet::new(),
            page_table: {
                let at = echora_shader_abi::PUSH_PAGE_TABLE_OFFSET as usize;
                let word = |n: usize| {
                    inputs.push.get(at + 4 * n..at + 4 * n + 4).map(|bytes| {
                        u64::from(u32::from_le_bytes(bytes.try_into().expect("4 bytes")))
                    })
                };
                match (word(0), word(1), word(2), word(3)) {
                    (Some(low), Some(high), Some(first), Some(count)) if low | high != 0 => {
                        Some((low | high << 32, first, count))
                    }
                    _ => None,
                }
            },
            unmapped_pages: BTreeMap::new(),
            compare_op: inputs.compare_op,
            mips: inputs.mips.clone(),
            watch: inputs.watch.clone(),
            trace: inputs.trace,
            traced: Vec::new(),
            guest_pages: HashMap::new(),
            writes: Vec::new(),
            mesh_outputs: None,
            mesh_tasks: None,
            uniforms: self
                .globals
                .iter()
                .filter(|&&(_, _, class)| class == storage::UNIFORM)
                .map(|&(variable, _, _)| {
                    (
                        variable,
                        (
                            self.sets.get(&variable).copied().unwrap_or(0),
                            self.bindings.get(&variable).copied().unwrap_or(0),
                        ),
                    )
                })
                .collect(),
            loop_limit: if inputs.loop_limit == 0 {
                1 << 16
            } else {
                inputs.loop_limit
            },
        };
        // Guest pages by device page, for the watched writes' guest
        // addresses.
        if let (false, Some((table, first, count))) = (shared.watch.is_empty(), shared.page_table) {
            use echora_shader_abi::PAGE_SHIFT;
            for n in 0..count {
                if let Some((base, offset)) = shared.region(table + 8 * n, 8) {
                    let bytes = &shared.memory[&base][offset..offset + 8];
                    let entry = u64::from_le_bytes(bytes.try_into().expect("8 bytes"));
                    if entry != 0 {
                        shared.guest_pages.insert(entry >> PAGE_SHIFT, first + n);
                    }
                }
            }
        }
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
                storage::TASK_PAYLOAD_WORKGROUP_EXT => inputs.payload.clone(),
                // A mesh program's outputs belong to the workgroup.
                storage::OUTPUT if self.model == execution::MESH_EXT => Vec::new(),
                _ => continue,
            };
            let mut slots = Vec::new();
            self.fill(pointee, &bytes, 0, &mut slots);
            shared.variables.insert(variable, slots);
        }
        let mut invocations: Vec<Invocation> = (0..count)
            .map(|n| {
                let mut invocation = self.start(&builtins(n), &inputs.locations);
                invocation.number = n;
                invocation.initial = initial(n);
                for (&constant, spec_id) in &self.spec_ids {
                    if let Some(&value) = inputs.specialization.get(spec_id) {
                        invocation.values.insert(constant, vec![u64::from(value)]);
                    }
                }
                invocation
            })
            .collect();
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
            // Subgroup barriers: every subgroup whose running invocations
            // all wait at one passes it, all such subgroups in the same
            // round (their warps run at once on the hardware, so one
            // warp's memory traffic interleaves with another's).
            let scope_of = |at: usize| {
                let scope_id = self.body[at].operands[0];
                self.constants.get(&scope_id).map_or(2, |value| value[0])
            };
            for first in (0..count).step_by(subgroup) {
                let members: Vec<usize> = (first..(first + subgroup).min(count))
                    .filter(|&n| !invocations[n].done)
                    .collect();
                let Some(&lead) = members.first() else {
                    continue;
                };
                let at = invocations[lead].at;
                if self.body[at].opcode == op::CONTROL_BARRIER
                    && scope_of(at) == 3
                    && members.iter().all(|&n| invocations[n].at == at)
                {
                    for &n in &members {
                        invocations[n].at += 1;
                    }
                    progress = true;
                }
            }
            if progress {
                continue;
            }
            // Workgroup barriers: every running invocation at the same one.
            let running: Vec<usize> = (0..count).filter(|&n| !invocations[n].done).collect();
            let at = invocations[running[0]].at;
            let is_barrier = self.body[at].opcode == op::CONTROL_BARRIER;
            if is_barrier && scope_of(at) != 3 && running.iter().all(|&n| invocations[n].at == at) {
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
        let ends = invocations.iter().map(|invocation| invocation.at).collect();
        let exits = invocations
            .iter()
            .map(|invocation| invocation.exit)
            .collect();
        let first_nan = invocations
            .iter()
            .map(|invocation| invocation.first_nan)
            .collect();
        let first_infinite = invocations
            .iter()
            .map(|invocation| invocation.first_infinite)
            .collect();
        Report {
            ends,
            exits,
            first_nan,
            first_infinite,
            writes: shared.writes,
            traced: shared.traced,
            invocations: invocations
                .into_iter()
                .map(|invocation| invocation.variables)
                .collect(),
            barriers,
            problems: shared.problems,
            notes: shared.notes,
            memory: shared.memory,
            images: shared.images,
            uniform_reads: shared.uniform_reads,
            unmapped_pages: shared.unmapped_pages,
            mesh_outputs: shared.mesh_outputs,
            mesh_tasks: shared.mesh_tasks,
            workgroup_outputs: self
                .globals
                .iter()
                .filter(|&&(_, _, class)| {
                    class == storage::TASK_PAYLOAD_WORKGROUP_EXT
                        || (class == storage::OUTPUT && self.model == execution::MESH_EXT)
                })
                .filter_map(|&(variable, _, _)| {
                    shared
                        .variables
                        .get(&variable)
                        .map(|value| (variable, value.clone()))
                })
                .collect(),
        }
    }

    /// An invocation about to start, with its built-in inputs.
    fn start(&self, builtins: &HashMap<u32, Value>, locations: &HashMap<u32, Value>) -> Invocation {
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
            let shared_output = class == storage::OUTPUT && self.model == execution::MESH_EXT;
            if matches!(class, storage::INPUT | storage::OUTPUT | storage::PRIVATE)
                && !shared_output
            {
                let located = (class == storage::INPUT)
                    .then(|| self.locations.get(&variable))
                    .flatten()
                    .and_then(|location| locations.get(location))
                    .map(|value| {
                        let first = self.components.get(&variable).copied().unwrap_or(0);
                        value.get(first as usize..).unwrap_or_default().to_vec()
                    });
                let mut value = self
                    .builtins
                    .get(&variable)
                    .and_then(|builtin| builtins.get(builtin))
                    .cloned()
                    .or(located)
                    .unwrap_or_else(|| vec![0; self.size(pointee)]);
                value.resize(self.size(pointee), 0);
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
            tainted: HashSet::new(),
            tainted_variables: HashSet::new(),
            page_counts: HashSet::new(),
            initial: HashMap::new(),
            number: 0,
            current: 0,
            exit: None,
            step: None,
            first_nan: None,
            first_infinite: None,
            done: false,
            steps: 0,
        }
    }

    /// Whether `opcode` needs the other invocations (a barrier, or a
    /// subgroup operation).
    fn collective(opcode: u16) -> bool {
        opcode == op::CONTROL_BARRIER
            || (333..=366).contains(&opcode)
            || matches!(
                opcode,
                op::COOPERATIVE_MATRIX_LOAD
                    | op::COOPERATIVE_MATRIX_STORE
                    | op::COOPERATIVE_MATRIX_MUL_ADD
            )
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
        // Cooperative matrix operations run once for the subgroup (each
        // invocation's operands are the same), as on the hardware.
        if matches!(
            opcode,
            op::COOPERATIVE_MATRIX_LOAD
                | op::COOPERATIVE_MATRIX_STORE
                | op::COOPERATIVE_MATRIX_MUL_ADD
        ) {
            let lead = members[0];
            invocations[lead].current = at;
            let result = self.matrix_operation(&mut invocations[lead], shared);
            for &n in members {
                let invocation = &mut invocations[n];
                if let Some((id, ty, value)) = &result {
                    invocation.values.insert(*id, value.clone());
                    invocation.value_types.insert(*id, *ty);
                }
                invocation.at += 1;
            }
            return;
        }
        let (ty, id) = (operands[0], operands[1]);
        let input = |invocation: &Invocation, n: usize| invocation.values[&operands[n]].clone();
        let all = |n: usize| -> Vec<Value> {
            members.iter().map(|&m| input(&invocations[m], n)).collect()
        };
        let same = |value: Value| vec![value; members.len()];
        // A result is undefined where it reads an undefined value, or (a
        // shuffle) a lane not running there.
        let any_undefined = members.iter().any(|&m| {
            operands[2..]
                .iter()
                .any(|a| invocations[m].tainted.contains(a))
        });
        let mut undefined: Vec<bool> = vec![any_undefined; members.len()];
        let results: Vec<Value> = match opcode {
            op::GROUP_NON_UNIFORM_ALL => same(vec![u64::from(all(3).iter().all(|v| v[0] != 0))]),
            op::GROUP_NON_UNIFORM_ANY => same(vec![u64::from(all(3).iter().any(|v| v[0] != 0))]),
            op::GROUP_NON_UNIFORM_BROADCAST_FIRST => same(all(3)[0].clone()),
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
                .enumerate()
                .map(|(k, &n)| {
                    let lane = input(&invocations[n], 4)[0] as usize;
                    match members.iter().find(|&&m| m == first + lane) {
                        Some(&source) => {
                            undefined[k] = invocations[source].tainted.contains(&operands[3])
                                || invocations[n].tainted.contains(&operands[4]);
                            input(&invocations[source], 3)
                        }
                        None => {
                            undefined[k] = true;
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
        for ((&n, result), undefined) in members.iter().zip(results).zip(undefined) {
            let invocation = &mut invocations[n];
            invocation.values.insert(id, result);
            invocation.value_types.insert(id, ty);
            if undefined {
                invocation.tainted.insert(id);
            } else {
                invocation.tainted.remove(&id);
            }
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
                if let Some(&(set, binding)) = shared.uniforms.get(&variable) {
                    // Uniform blocks hold 32-bit scalars: a slot is 4 bytes.
                    for slot in offset..offset + size {
                        shared.uniform_reads.insert((set, binding, slot as u32 * 4));
                    }
                }
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
                // A page table entry of 0: an unmapped guest page.
                if let Some((table, first, count)) = shared.page_table {
                    let inside = address >= table && address < table + 8 * count;
                    if inside && bytes.len() == 8 && slots == [0] {
                        let page = first + (address - table) / 8;
                        *shared.unmapped_pages.entry(page).or_insert(0) += 1;
                    }
                }
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
                if !shared.watch.is_empty() {
                    let at = invocation.current;
                    let opcode = self
                        .body
                        .get(at)
                        .map_or(0, |instruction| instruction.opcode);
                    shared.watch_write((invocation.number, at, opcode), address, &bytes);
                }
                shared.write(address, &bytes);
            }
        }
    }

    /// Where element `n` (row-major order) of a matrix of type `ty` lies
    /// for a load or store at `pointer` with `layout` (0 row-major, 1
    /// column-major) and `stride` (in the pointer's elements, which may
    /// hold several of the matrix's): the element's pointer, and the
    /// element's bit shift and width within it.
    fn matrix_element(
        &self,
        pointer: Pointer,
        ty: u32,
        layout: u64,
        stride: usize,
        n: usize,
    ) -> (Pointer, u64, u64) {
        let Ty::Matrix {
            columns, component, ..
        } = self.ty(ty)
        else {
            panic!("not a matrix: {ty}");
        };
        let pointee = match pointer {
            Pointer::Logical { pointee, .. } | Pointer::Physical { pointee, .. } => pointee,
        };
        let bits = 8 * self.scalar_bytes(component);
        let per = 8 * self.scalar_bytes(pointee) / bits;
        let (row, column) = (n / columns as usize, n % columns as usize);
        let (line, along) = match layout {
            0 => (row, column),
            1 => (column, row),
            other => panic!("cooperative matrix layout {other}"),
        };
        let at = line * stride * per + along;
        (
            self.offset(pointer, at / per),
            (at % per * bits) as u64,
            bits as u64,
        )
    }

    /// A cooperative matrix operation, run by `invocation` for its
    /// subgroup: the result (id, type, value), if it has one.
    fn matrix_operation(
        &self,
        invocation: &mut Invocation,
        shared: &mut Shared,
    ) -> Option<(u32, u32, Value)> {
        let instruction = &self.body[invocation.at];
        let (opcode, operands) = (instruction.opcode, &instruction.operands);
        if opcode == op::COOPERATIVE_MATRIX_STORE {
            let pointer = invocation.pointers[&operands[0]];
            let matrix = invocation.values[&operands[1]].clone();
            let ty = invocation.value_types[&operands[1]];
            let layout = invocation.values[&operands[2]][0];
            let stride = invocation.values[&operands[3]][0] as usize;
            for (n, &element) in matrix.iter().enumerate() {
                let (at, shift, bits) = self.matrix_element(pointer, ty, layout, stride, n);
                let mask = (1u64 << bits) - 1;
                let old = self.load(invocation, shared, at)[0];
                let new = old & !(mask << shift) | (element & mask) << shift;
                self.store(invocation, shared, at, &vec![new]);
            }
            return None;
        }
        let (ty, id, args) = (operands[0], operands[1], &operands[2..]);
        let value = |n: usize| invocation.values[&args[n]].clone();
        let result = if opcode == op::COOPERATIVE_MATRIX_LOAD {
            let pointer = invocation.pointers[&args[0]];
            let layout = invocation.values[&args[1]][0];
            let stride = invocation.values[&args[2]][0] as usize;
            let (rows, columns) = match self.ty(ty) {
                Ty::Matrix { rows, columns, .. } => (rows as usize, columns as usize),
                other => panic!("a matrix load of {other:?}"),
            };
            (0..rows * columns)
                .map(|n| {
                    let (at, shift, bits) = self.matrix_element(pointer, ty, layout, stride, n);
                    self.load(invocation, shared, at)[0] >> shift & ((1u64 << bits) - 1)
                })
                .collect()
        } else {
            // A x B + C, each element's sum exact, rounded once to the
            // result's component type.
            let (a, b, c) = (value(0), value(1), value(2));
            let a_type = invocation.value_types[&args[0]];
            let k = self.matrix_columns(a_type);
            let columns = self.matrix_columns(ty);
            let rows = a.len() / k;
            let element = |matrix: u32| match self.ty(matrix) {
                Ty::Matrix { component, .. } => component,
                _ => matrix,
            };
            let read = |bits: u64, component: u32| match self.ty(component) {
                Ty::Float { width: 16 } => f64::from(f16_to_f32(bits as u16)),
                _ => f64::from(f32::from_bits(bits as u32)),
            };
            let (a_component, b_component) =
                (element(a_type), element(invocation.value_types[&args[1]]));
            let c_component = element(invocation.value_types[&args[2]]);
            let result_component = element(ty);
            (0..rows * columns)
                .map(|n| {
                    let (row, column) = (n / columns, n % columns);
                    let sum = (0..k).fold(read(c[n], c_component), |sum, index| {
                        sum + read(a[row * k + index], a_component)
                            * read(b[index * columns + column], b_component)
                    });
                    match self.ty(result_component) {
                        Ty::Float { width: 16 } => u64::from(f64_to_f16(sum)),
                        _ => u64::from((sum as f32).to_bits()),
                    }
                })
                .collect()
        };
        Some((id, ty, result))
    }

    /// The byte offset of slot `offset` of push-constant variable
    /// `variable` (a block of scalars), if it is one.
    fn push_word(&self, variable: u32, offset: usize) -> Option<u32> {
        let &(_, pointer, _) = self
            .globals
            .iter()
            .find(|&&(id, _, class)| id == variable && class == storage::PUSH_CONSTANT)?;
        let Ty::Pointer { pointee } = self.ty(pointer) else {
            return None;
        };
        let Ty::Struct { members } = self.ty(pointee) else {
            return None;
        };
        // Slots of scalar members, in order.
        let mut slot = 0;
        for (n, &member) in members.iter().enumerate() {
            if slot == offset {
                return Some(self.member_offset(pointee, n as u32) as u32);
            }
            slot += self.size(member);
        }
        None
    }

    /// The (descriptor set, binding) of a handle's bound texture or image
    /// (not a texture pool entry).
    fn bound_image(&self, handle: &Value) -> Option<(u32, u32)> {
        let variable = *handle.first()? as u32;
        let is_array = self.globals.iter().any(|&(id, pointer, _)| {
            id == variable
                && matches!(self.ty(pointer), Ty::Pointer { pointee }
                    if matches!(self.ty(pointee), Ty::RuntimeArray { .. } | Ty::Array { .. }))
        });
        if is_array {
            return None;
        }
        Some((
            self.sets.get(&variable).copied().unwrap_or(0),
            *self.bindings.get(&variable)?,
        ))
    }

    /// The (set, binding, element) of an image array's element: its
    /// handle is the array variable and the element's index.
    fn array_element(&self, handle: &Value) -> Option<(u32, u32, u32)> {
        let variable = *handle.first()? as u32;
        let element = *handle.get(1)? as u32;
        Some((
            self.sets.get(&variable).copied().unwrap_or(0),
            *self.bindings.get(&variable)?,
            element,
        ))
    }

    /// An image type's dimension and arrayed flag (through a sampled
    /// image type).
    fn image_shape(&self, ty: u32) -> (u32, bool) {
        match self.ty(ty) {
            Ty::Image { dim, arrayed } => (dim, arrayed),
            Ty::SampledImage { image } => self.image_shape(image),
            _ => (1, false),
        }
    }

    /// Integer coordinates (x, y, then z or the layer).
    fn integer_coordinates(&self, image_type: u32, coordinate: &Value) -> [i64; 3] {
        let get = |n: usize| coordinate.get(n).map_or(0, |&v| i64::from(v as u32 as i32));
        match self.image_shape(image_type) {
            (0 | 5, false) => [get(0), 0, 0],
            (0, true) => [get(0), 0, get(1)],
            (1, false) => [get(0), get(1), 0],
            _ => [get(0), get(1), get(2)],
        }
    }

    /// A read of `image`: a fetch, a nearest sample (level 0, clamped to
    /// the edge), a gather of `component`, a storage read, or a query.
    #[allow(clippy::too_many_arguments)]
    fn image_read(
        &self,
        opcode: u16,
        image_type: u32,
        image: &Image,
        coordinate: &Value,
        component: usize,
        offset: [i64; 2],
        ty: u32,
    ) -> Value {
        let (dim, arrayed) = self.image_shape(image_type);
        let size = self.size(ty);
        let words = |texel: [u32; 4]| texel.iter().take(size).map(|&w| u64::from(w)).collect();
        let float = |n: usize| {
            f64::from(f32::from_bits(
                coordinate.get(n).copied().unwrap_or(0) as u32
            ))
        };
        match opcode {
            op::IMAGE_QUERY_SIZE_LOD => {
                let mut sizes = vec![u64::from(image.width)];
                if dim != 0 && dim != 5 {
                    sizes.push(u64::from(image.height));
                }
                if dim == 2 {
                    sizes.push(u64::from(image.depth));
                }
                if arrayed {
                    sizes.push(u64::from(image.layers));
                }
                sizes.truncate(size.max(1));
                sizes
            }
            op::IMAGE_QUERY_LEVELS => vec![1],
            op::IMAGE_FETCH | op::IMAGE_READ => {
                let [x, y, z] = self.integer_coordinates(image_type, coordinate);
                words(image.texel(x, y, z))
            }
            _ => {
                // Normalized coordinates; an array's layer is a float index.
                let scaled = |n: usize, extent: u32| float(n) * f64::from(extent.max(1));
                let (u, v) = (scaled(0, image.width), scaled(1, image.height));
                let z = match (dim, arrayed) {
                    (1, true) => float(2).round_ties_even() as i64,
                    (0, true) => float(1).round_ties_even() as i64,
                    (2, _) => scaled(2, image.depth).floor() as i64,
                    _ => 0,
                };
                if opcode == op::IMAGE_GATHER {
                    let (i0, j0) = (
                        (u - 0.5).floor() as i64 + offset[0],
                        (v - 0.5).floor() as i64 + offset[1],
                    );
                    [(i0, j0 + 1), (i0 + 1, j0 + 1), (i0 + 1, j0), (i0, j0)]
                        .iter()
                        .map(|&(i, j)| u64::from(image.texel(i, j, z)[component.min(3)]))
                        .collect()
                } else {
                    words(image.texel(u.floor() as i64, v.floor() as i64, z))
                }
            }
        }
    }

    fn matrix_columns(&self, ty: u32) -> usize {
        match self.ty(ty) {
            Ty::Matrix { columns, .. } => columns as usize,
            other => panic!("not a matrix: {other:?}"),
        }
    }

    /// `pointer` (to an array element) moved on by `elements` elements.
    fn offset(&self, pointer: Pointer, elements: usize) -> Pointer {
        match pointer {
            Pointer::Logical {
                variable,
                offset,
                pointee,
            } => Pointer::Logical {
                variable,
                offset: offset + elements * self.size(pointee),
                pointee,
            },
            Pointer::Physical { address, pointee } => Pointer::Physical {
                address: address + (elements * self.byte_size(pointee)) as u64,
                pointee,
            },
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
            invocation.current = at;
            invocation.at += 1;
            let operands = &instruction.operands;
            let opcode = instruction.opcode;
            match opcode {
                op::LABEL | op::SELECTION_MERGE | op::MEMORY_BARRIER => continue,
                op::SET_MESH_OUTPUTS_EXT => {
                    let count = |n: usize| invocation.values[&operands[n]][0] as u32;
                    shared.mesh_outputs = Some((count(0), count(1)));
                    continue;
                }
                op::EMIT_MESH_TASKS_EXT => {
                    let count = |n: usize| invocation.values[&operands[n]][0] as u32;
                    shared.mesh_tasks = Some([count(0), count(1), count(2)]);
                    invocation.done = true;
                    return;
                }
                // The translation marks each guest EXIT taken with a line.
                op::LINE => {
                    if self.step_file.is_some() && operands.first() == self.step_file.as_ref() {
                        invocation.step = operands.get(1).copied();
                    } else if invocation.exit.is_none() {
                        invocation.exit = operands.get(1).copied();
                    }
                    continue;
                }
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
                    let value = match invocation.initial.get(&operands[1]) {
                        Some(value) => value.clone(),
                        None => operands.get(3).map_or_else(
                            || vec![0; self.size(pointee)],
                            |id| invocation.values[id].clone(),
                        ),
                    };
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
                    if invocation.tainted.contains(&operands[0]) {
                        shared.problem(format!(
                            "instruction {at}: a branch depends on a shuffle from a lane not \
                             running there (undefined on a GPU)"
                        ));
                    }
                    let taken = invocation.values[&operands[0]][0] != 0;
                    invocation.at = self.labels[&operands[if taken { 1 } else { 2 }]];
                    continue;
                }
                op::SWITCH => {
                    if invocation.tainted.contains(&operands[0]) {
                        shared.problem(format!(
                            "instruction {at}: a switch depends on a shuffle from a lane not \
                             running there (undefined on a GPU)"
                        ));
                    }
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
                    if let Pointer::Logical { variable, .. } = pointer {
                        if shared.trace == Some((invocation.number, variable)) {
                            shared.traced.push((invocation.step, value.clone()));
                        }
                        if invocation.tainted.contains(&operands[1]) {
                            invocation.tainted_variables.insert(variable);
                        } else if invocation.variables.get(&variable).map(Vec::len)
                            == Some(value.len())
                        {
                            invocation.tainted_variables.remove(&variable);
                        }
                    }
                    self.store(invocation, shared, pointer, &value);
                    continue;
                }
                op::FUNCTION_CALL => panic!("the simulator does not call functions"),
                op::IMAGE_WRITE => {
                    let coordinate = invocation.values[&operands[1]].clone();
                    let texel = invocation.values[&operands[2]].clone();
                    let image_type = invocation.value_types[&operands[0]];
                    let handle = invocation.values[&operands[0]].clone();
                    match self.bound_image(&handle) {
                        Some(key) if shared.images.contains_key(&key) => {
                            let [x, y, z] = self.integer_coordinates(image_type, &coordinate);
                            let image = shared.images.get_mut(&key).expect("checked");
                            match image.texel_mut(x, y, z) {
                                Some(target) => {
                                    for (n, slot) in target.iter_mut().enumerate() {
                                        *slot = texel.get(n).copied().unwrap_or(0) as u32;
                                    }
                                }
                                None => {
                                    let guest =
                                        invocation.step.map_or_else(String::new, |offset| {
                                            format!(" (guest {offset:#06x})")
                                        });
                                    shared.problem(format!(
                                        "image write at ({x}, {y}, {z}) outside the image at \
                                         {key:?}{guest}"
                                    ))
                                }
                            }
                        }
                        _ => shared.note("writes to images without dumped data are dropped"),
                    }
                    continue;
                }
                _ => {}
            }
            // Instructions with a result.
            let (ty, id, args) = (operands[0], operands[1], &operands[2..]);
            let values = &invocation.values;
            let value = |n: usize| values[&args[n]].clone();
            let result: Value = match opcode {
                // Images, samplers and sampled images: a handle, the
                // variable and the element (a texture pool's entry).
                op::LOAD
                    if matches!(
                        self.ty(ty),
                        Ty::Image { .. } | Ty::SampledImage { .. } | Ty::Sampler | Ty::Other
                    ) =>
                {
                    match invocation.pointers[&args[0]] {
                        Pointer::Logical {
                            variable, offset, ..
                        } => vec![u64::from(variable), offset as u64],
                        Pointer::Physical { .. } => vec![0, 0],
                    }
                }
                op::SAMPLED_IMAGE | op::IMAGE | COPY_OBJECT => value(0),
                op::IMAGE_SAMPLE_IMPLICIT_LOD
                | op::IMAGE_SAMPLE_EXPLICIT_LOD
                | op::IMAGE_FETCH
                | op::IMAGE_GATHER
                | op::IMAGE_READ
                | op::IMAGE_QUERY_SIZE_LOD
                | op::IMAGE_QUERY_LEVELS => {
                    let image_type = invocation.value_types[&args[0]];
                    let handle = value(0);
                    let key = self.bound_image(&handle);
                    let image = match key {
                        Some(key) => shared.images.get(&key).cloned(),
                        None => self
                            .array_element(&handle)
                            .and_then(|key| shared.image_arrays.get(&key).cloned()),
                    };
                    // Its level chain: level 0, then `Inputs::mips`.
                    let chain: Vec<Image> = image
                        .iter()
                        .cloned()
                        .chain(
                            key.and_then(|key| shared.mips.get(&key))
                                .into_iter()
                                .flatten()
                                .cloned(),
                        )
                        .collect();
                    if opcode == op::IMAGE_QUERY_LEVELS && !chain.is_empty() {
                        vec![chain.len() as u64]
                    } else {
                        match image {
                            Some(base) => {
                                let level = self.image_level(opcode, invocation, args, &base);
                                let image = chain[level.min(chain.len() - 1)].clone();
                                let coordinate = if args.len() > 1 { value(1) } else { Vec::new() };
                                let component = if opcode == op::IMAGE_GATHER {
                                    value(2)[0] as usize
                                } else {
                                    0
                                };
                                let offset = if opcode == op::IMAGE_GATHER {
                                    self.gather_offset(invocation, args)
                                } else {
                                    [0; 2]
                                };
                                self.image_read(
                                    opcode,
                                    image_type,
                                    &image,
                                    &coordinate,
                                    component,
                                    offset,
                                    ty,
                                )
                            }
                            None => {
                                shared.note("reads of textures without dumped data give zeros");
                                vec![0; self.size(ty)]
                            }
                        }
                    }
                }
                // Depth comparisons: the reference against component 0 of
                // the nearest texel (a sample) or of the four a gather takes,
                // by the sampler's compare op: 1.0 where it holds.
                op::IMAGE_SAMPLE_DREF_IMPLICIT_LOD
                | op::IMAGE_SAMPLE_DREF_EXPLICIT_LOD
                | op::IMAGE_DREF_GATHER => {
                    let image_type = invocation.value_types[&args[0]];
                    let handle = value(0);
                    let image = match self.bound_image(&handle) {
                        Some(key) => shared.images.get(&key).cloned(),
                        None => self
                            .array_element(&handle)
                            .and_then(|key| shared.image_arrays.get(&key).cloned()),
                    };
                    let compare = match shared.compare_op {
                        Some(op) => op,
                        None => {
                            shared.note(
                                "depth comparisons without Inputs::compare_op: less or equal",
                            );
                            3
                        }
                    };
                    match image {
                        Some(image) => {
                            let coordinate = value(1);
                            let reference = f32::from_bits(value(2)[0] as u32);
                            let gather = opcode == op::IMAGE_DREF_GATHER;
                            let offset = if gather {
                                self.gather_offset(invocation, args)
                            } else {
                                [0; 2]
                            };
                            let read = if gather {
                                op::IMAGE_GATHER
                            } else {
                                op::IMAGE_SAMPLE_EXPLICIT_LOD
                            };
                            let texels = self.image_read(
                                read,
                                image_type,
                                &image,
                                &coordinate,
                                0,
                                offset,
                                ty,
                            );
                            let depths: Vec<u64> = if gather {
                                texels
                            } else {
                                texels.into_iter().take(1).collect()
                            };
                            depths
                                .iter()
                                .map(|&bits| {
                                    let depth = f32::from_bits(bits as u32);
                                    let holds = match compare {
                                        0 => false,
                                        1 => reference < depth,
                                        2 => reference == depth,
                                        3 => reference <= depth,
                                        4 => reference > depth,
                                        5 => reference != depth,
                                        6 => reference >= depth,
                                        _ => true,
                                    };
                                    u64::from(if holds { 1.0f32 } else { 0.0 }.to_bits())
                                })
                                .collect()
                        }
                        None => {
                            shared.note("reads of textures without dumped data give zeros");
                            vec![0; self.size(ty)]
                        }
                    }
                }
                op::IMAGE_QUERY_LOD => {
                    shared.note("level queries give zeros");
                    vec![0; self.size(ty)]
                }
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
                // An image texel for an atomic: a word of its own (no image
                // data), starting at zero.
                op::IMAGE_TEXEL_POINTER => {
                    shared.note("image atomics act on scratch words (no image data)");
                    let Ty::Pointer { pointee } = self.ty(ty) else {
                        panic!("ImageTexelPointer to a non-pointer type");
                    };
                    invocation.variables.insert(id, vec![0]);
                    invocation.pointers.insert(
                        id,
                        Pointer::Logical {
                            variable: id,
                            offset: 0,
                            pointee,
                        },
                    );
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
                | op::ATOMIC_F_ADD
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
                        op::ATOMIC_F_ADD => {
                            (f32::from_bits(old) + f32::from_bits(operand(3))).to_bits()
                        }
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
            // The page count, and the comparison with it that finds a page
            // outside the table.
            match opcode {
                op::LOAD => {
                    if let Some(Pointer::Logical {
                        variable, offset, ..
                    }) = invocation.pointers.get(&args[0])
                        && self.push_word(*variable, *offset)
                            == Some(echora_shader_abi::PUSH_PAGE_TABLE_OFFSET + 12)
                    {
                        invocation.page_counts.insert(id);
                    }
                }
                op::U_CONVERT if invocation.page_counts.contains(&args[0]) => {
                    invocation.page_counts.insert(id);
                }
                op::U_GREATER_THAN
                    if invocation.page_counts.contains(&args[0]) && result == [0] =>
                {
                    if let Some((_, first, _)) = shared.page_table {
                        let relative = invocation.values[&args[1]][0];
                        let page = relative.wrapping_add(first);
                        *shared.unmapped_pages.entry(page).or_insert(0) += 1;
                    }
                }
                _ => {}
            }
            let undefined = if opcode == op::LOAD {
                matches!(invocation.pointers.get(&args[0]), Some(Pointer::Logical { variable, .. })
                    if invocation.tainted_variables.contains(variable))
            } else {
                args.iter().any(|a| invocation.tainted.contains(a))
            };
            if undefined {
                invocation.tainted.insert(id);
            } else {
                invocation.tainted.remove(&id);
            }
            let site = Some((invocation.step, opcode, invocation.current));
            let (nan, infinite) = self.nonfinite(ty, &result);
            if nan && invocation.first_nan.is_none() {
                invocation.first_nan = site;
            }
            if infinite && invocation.first_infinite.is_none() {
                invocation.first_infinite = site;
            }
            invocation.values.insert(id, result);
            invocation.value_types.insert(id, ty);
        }
    }

    /// Whether a float result holds a NaN, and whether an Inf.
    fn nonfinite(&self, ty: u32, value: &Value) -> (bool, bool) {
        let (exponent, shift) = match self.scalar(ty) {
            Ty::Float { width: 16 } => (0x1f, 10),
            Ty::Float { width: 32 } => (0xff, 23),
            Ty::Float { width: 64 } => (0x7ff, 52),
            _ => return (false, false),
        };
        let mantissa = (1u64 << shift) - 1;
        let special = |component: u64| component >> shift & exponent == exponent;
        (
            value
                .iter()
                .any(|&component| special(component) && component & mantissa != 0),
            value
                .iter()
                .any(|&component| special(component) && component & mantissa == 0),
        )
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
        let operand_width = match self.scalar(operand) {
            Ty::Int { width, .. } | Ty::Float { width } => width,
            _ => 32,
        };
        let lanes = inputs.first().map_or(1, Vec::len);
        let mut result = Vec::with_capacity(lanes);
        for lane in 0..lanes {
            let x = |n: usize| inputs[n][lane.min(inputs[n].len() - 1)];
            let f = |n: usize| f32::from_bits(x(n) as u32);
            let i = |n: usize| x(n) as u32 as i32;
            // An operand as a signed integer of its own width.
            let signed = |value: u64| -> i64 {
                if operand_width == 64 {
                    value as i64
                } else {
                    i64::from(value as u32 as i32)
                }
            };
            let fb = |v: f32| u64::from(v.to_bits());
            let b = |v: bool| u64::from(v);
            let value = match opcode {
                op::I_ADD => x(0).wrapping_add(x(1)) & mask,
                op::I_SUB => x(0).wrapping_sub(x(1)) & mask,
                op::I_MUL => x(0).wrapping_mul(x(1)) & mask,
                op::U_DIV => {
                    assert!(x(1) != 0, "UDiv by zero (undefined)");
                    x(0) / x(1)
                }
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
                op::S_LESS_THAN => b(signed(x(0)) < signed(x(1))),
                op::S_LESS_THAN_EQUAL => b(signed(x(0)) <= signed(x(1))),
                op::S_GREATER_THAN => b(signed(x(0)) > signed(x(1))),
                op::S_GREATER_THAN_EQUAL => b(signed(x(0)) >= signed(x(1))),
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
                // Unordered (either NaN) or equal.
                op::F_UNORD_EQUAL => b(f(0).is_nan() || f(1).is_nan() || f(0) == f(1)),
                op::F_UNORD_NOT_EQUAL => b(f(0) != f(1)),
                op::F_UNORD_LESS_THAN => b(!(f(0) >= f(1))),
                op::F_UNORD_LESS_THAN_EQUAL => b(!(f(0) > f(1))),
                op::F_UNORD_GREATER_THAN => b(!(f(0) <= f(1))),
                op::F_UNORD_GREATER_THAN_EQUAL => b(!(f(0) < f(1))),
                // To the result's width (32 or 64 bits); out of range is
                // undefined in SPIR-V.
                op::CONVERT_F_TO_U => {
                    let v = f(0);
                    let end = if width == 64 {
                        18_446_744_073_709_551_616.0
                    } else {
                        4_294_967_296.0
                    };
                    assert!(v.is_finite() && v > -1.0 && v < end, "ConvertFToU of {v}");
                    if width == 64 {
                        v as u64
                    } else {
                        u64::from(v as u32)
                    }
                }
                op::CONVERT_F_TO_S => {
                    let v = f(0);
                    let end = if width == 64 {
                        9_223_372_036_854_775_808.0
                    } else {
                        2_147_483_648.0
                    };
                    assert!(
                        v.is_finite() && (-end..end).contains(&v),
                        "ConvertFToS of {v}"
                    );
                    if width == 64 {
                        v as i64 as u64
                    } else {
                        u64::from(v as i32 as u32)
                    }
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
                    op::F_UNORD_EQUAL => b(h(0).is_nan() || h(1).is_nan() || h(0) == h(1)),
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
            let lane = (n % 32) as u32;
            // The eq, ge, gt, le and lt masks (SPIR-V's order).
            let masks = [
                1u32 << lane,
                !0u32 << lane,
                (!0u32 << lane) << 1,
                !((!0u32 << lane) << 1),
                !(!0u32 << lane),
            ];
            let mut builtins =
                HashMap::from([(27, vec![n as u64, 0, 0]), (41, vec![u64::from(lane)])]);
            for (k, mask) in masks.into_iter().enumerate() {
                builtins.insert(4416 + k as u32, vec![u64::from(mask), 0, 0, 0]);
            }
            builtins
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
            // Guest words 2-3, and device word 40.
            watch: vec![0x10008..0x10010, page + 160..page + 161],
            ..Inputs::default()
        };
        let machine = Machine::new(&translated.spirv);
        assert_eq!(machine.local_size, [64, 1, 1]);
        let report = machine.simulate(&inputs);
        assert!(report.problems.is_empty(), "{:?}", report.problems);
        assert_eq!(report.barriers.len(), 1);
        // The watched stores: invocations 2, 3 and 40, each its word.
        let listed: Vec<_> = report
            .writes
            .iter()
            .map(|w| {
                (
                    w.invocation,
                    w.device,
                    w.guest,
                    w.old.clone(),
                    w.new.clone(),
                )
            })
            .collect();
        let expected: Vec<_> = [2u32, 3, 40]
            .iter()
            .map(|&n| {
                let at = 4 * u64::from(n);
                (
                    n as usize,
                    page + at,
                    Some(0x10000 + at),
                    n.to_le_bytes().to_vec(),
                    ((n ^ 32) + 1).to_le_bytes().to_vec(),
                )
            })
            .collect();
        assert_eq!(listed, expected);
        assert!(report.writes.iter().all(|w| w.opcode == op::STORE));
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

    #[test]
    fn warp_matrix_multiplies_match_the_ptx_layout_on_every_path() {
        use crate::compute::MatrixPath;
        // A (16x16) shared by two HMMA.16816.F32s with their own B (16x8)
        // and C (16x8): small integers, exact in halves and in any sum.
        let a = |row: usize, k: usize, warp: usize| ((row * 3 + k * 5 + warp) % 7) as f64 - 3.0;
        let b = |n: usize, k: usize, column: usize| ((k * 2 + column * 3 + n) % 5) as f64 - 2.0;
        // Warp w's C is 100 w more (two warps: their scratch must not
        // overlap).
        let c = |n: usize, row: usize, column: usize, warp: usize| {
            (row + 2 * column + n + 100 * warp) as f64
        };
        let half = |value: f64| u64::from(f64_to_f16(value));
        let pair = |low: f64, high: f64| half(low) | half(high) << 16;
        for k16 in [true, false] {
            let depth = if k16 { 16 } else { 8 };
            let hmma = |d: u128, b: u128, c: u128| {
                word(&[
                    (0, 12, 0x23c),
                    (16, 8, d),
                    (24, 8, 0),
                    (32, 8, b),
                    (64, 8, c),
                    (75, 1, u128::from(k16)),
                    (76, 1, 1),
                ])
            };
            // A in R0-R3; B in R4-R5 and R6-R7; C in R8-R11 and R12-R15
            // (HMMA.1688: A in R0-R1, B in R4 and R6).
            let program = [hmma(16, 4, 8), hmma(20, 6, 12), EXIT];
            let code: Vec<_> = program.iter().map(|&raw| decode(raw)).collect();
            for matrices in [
                MatrixPath::Subgroup,
                MatrixPath::Cooperative,
                MatrixPath::Cooperative32,
                MatrixPath::Exact,
            ] {
                let layout = Layout {
                    matrices,
                    ..Layout::workgroup([64, 1, 1])
                };
                let (translated, registers, _) = translate_with_registers(&code, &layout).unwrap();
                let machine = Machine::new(&translated.spirv);
                // Lane l's fragments, by the PTX layout (g = l / 4, t = l % 4).
                let initial = |invocation: usize| {
                    let (lane, warp) = (invocation % 32, invocation / 32);
                    let (g, t) = (lane / 4, lane % 4);
                    let mut values: HashMap<u8, u64> = HashMap::new();
                    for j in 0..depth / 4 {
                        let row = g + 8 * (j & 1);
                        let k = 2 * t + 8 * (j >> 1);
                        values.insert(j as u8, pair(a(row, k, warp), a(row, k + 1, warp)));
                    }
                    for n in 0..2 {
                        for j in 0..depth / 8 {
                            let k = 2 * t + 8 * j;
                            values.insert((4 + 2 * n + j) as u8, pair(b(n, k, g), b(n, k + 1, g)));
                        }
                        for i in 0..4 {
                            let (row, column) = (g + 8 * (i >> 1), 2 * t + (i & 1));
                            let value = (c(n, row, column, warp) as f32).to_bits();
                            values.insert((8 + 4 * n + i) as u8, u64::from(value));
                        }
                    }
                    values
                        .into_iter()
                        .filter_map(|(register, value)| {
                            registers
                                .get(&register)
                                .map(|&variable| (variable, vec![value]))
                        })
                        .collect()
                };
                let report = machine.run_workgroup_with(
                    64,
                    32,
                    |n| {
                        HashMap::from([
                            (27, vec![n as u64, 0, 0]),
                            (29, vec![n as u64]),
                            (40, vec![n as u64 / 32]),
                            (41, vec![n as u64 % 32]),
                        ])
                    },
                    initial,
                );
                assert!(
                    report.problems.is_empty(),
                    "{matrices:?}: {:?}",
                    report.problems
                );
                for (invocation, variables) in report.invocations.iter().enumerate() {
                    let (lane, warp) = (invocation % 32, invocation / 32);
                    let (g, t) = (lane / 4, lane % 4);
                    for n in 0..2 {
                        for i in 0..4 {
                            let (row, column) = (g + 8 * (i >> 1), 2 * t + (i & 1));
                            let expected = (0..depth).fold(c(n, row, column, warp), |sum, k| {
                                sum + a(row, k, warp) * b(n, k, column)
                            }) as f32;
                            let register = (16 + 4 * n + i) as u8;
                            let got = f32::from_bits(variables[&registers[&register]][0] as u32);
                            assert_eq!(
                                got, expected,
                                "{matrices:?} k16 {k16} warp {warp} lane {lane} D{n}[{i}]"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn storage_images_read_and_write_the_dumped_texels() {
        // SULD.P.2D R4, [R0] (x = 1, y = 2), 0xf; MOV R0, 3;
        // SUST.P.2D [R0], R4, 0xf: texel (1, 2) copied to (3, 2).
        let surface = |store: bool, data: u128| {
            word(&[
                (0, 12, if store { 0xb9b } else { 0xb97 }),
                (16, 8, if store { 0 } else { data }),
                (24, 8, 0),
                (32, 8, if store { data } else { 0 }),
                (40, 14, 0x18),
                (54, 5, 0),
                (61, 3, 3),
                (72, 4, 0xf),
                (81, 3, if store { 0 } else { 7 }),
            ])
        };
        let program = [
            mov(0, 1),
            mov(1, 2),
            surface(false, 4),
            mov(0, 3),
            surface(true, 4),
            EXIT,
        ];
        let code: Vec<_> = program.iter().map(|&raw| decode(raw)).collect();
        let translated = crate::compute::translate(&code, &Layout::workgroup([1, 1, 1])).unwrap();
        let image = translated.descriptors.iter().find(|d| d.kind == 5).unwrap();
        let texels = (0..16u32).map(|n| [n, n + 100, n + 200, n + 300]).collect();
        let inputs = Inputs {
            workgroup_count: [1, 1, 1],
            images: HashMap::from([(
                (0, image.binding),
                Image {
                    width: 4,
                    height: 4,
                    depth: 1,
                    layers: 1,
                    texels,
                },
            )]),
            ..Inputs::default()
        };
        let report = Machine::new(&translated.spirv).simulate(&inputs);
        assert!(report.problems.is_empty(), "{:?}", report.problems);
        let result = &report.images[&(0, image.binding)];
        assert_eq!(result.texels[2 * 4 + 3], [9, 109, 209, 309]);
    }

    #[test]
    fn uniform_registers_take_the_first_active_lane_s_value() {
        // S2R R0, SR_LANEID; R2UR UR4, R0; MOV R1, UR4
        let lane = word(&[(0, 12, 0x919), (16, 8, 0), (72, 8, 0)]);
        let to_uniform = word(&[(0, 12, 0x3c2), (16, 6, 4), (24, 8, 0)]);
        let back = word(&[(0, 12, 0xc02), (72, 4, 0xf), (16, 8, 1), (32, 6, 4)]);
        let (registers, _, problems) = run_group(32, &[], &[lane, to_uniform, back]);
        assert!(problems.is_empty(), "{problems:?}");
        assert!(registers.iter().all(|registers| registers[&1] == 0));
    }

    #[test]
    fn branches_on_shuffles_from_lanes_not_running_are_reported() {
        // S2R R0, SR_LANEID; ISETP.GE P0, R0, R1 (R1 = 16); @P0 EXIT;
        // SHFL.IDX R2, R0, R3 (R3 = 20), 0x1f; ISETP.LT P1, R2, R1;
        // @P1 BRA past the next instruction
        let lane = word(&[(0, 12, 0x919), (16, 8, 0), (72, 8, 0)]);
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
        let shuffle = word(&[
            (0, 12, 0x589),
            (16, 8, 2),
            (24, 8, 0),
            (32, 8, 3),
            (40, 13, 0x1f),
            (81, 3, 7),
        ]);
        let below = word(&[
            (0, 12, 0x20c),
            (24, 8, 2),
            (32, 8, 1),
            (76, 3, 1),
            (81, 3, 1),
            (84, 3, 7),
            (87, 3, 7),
        ]);
        // MOVs of R0, R1, R2, R3 are instructions 0-3.
        let body = [
            lane,
            at_least,
            exit,
            shuffle,
            below,
            bra(9, 11, 1),
            add(4, 4, 1),
        ];
        let (_, _, problems) = run_group(32, &[0, 16, 0, 20], &body);
        assert!(
            problems
                .iter()
                .any(|problem| problem.contains("a branch depends on a shuffle")),
            "{problems:?}"
        );
        // From a running lane: nothing to report.
        let (_, _, problems) = run_group(32, &[0, 16, 0, 5], &body);
        assert!(problems.is_empty(), "{problems:?}");
    }

    #[test]
    fn traced_loops_count_their_iterations_in_the_trace_buffer() {
        use echora_shader_abi::{LOOP_TRACE_HIGH_SPEC_ID, LOOP_TRACE_LOW_SPEC_ID};
        // R1 = 10; 2: IADD3 R0, R0, 1; 3: ISETP.LT P0, R0, R1; 4: @P0 BRA 2
        let program = [
            mov(0, 0),
            mov(1, 10),
            add(0, 0, 1),
            less(0, 0, 1),
            bra(4, 2, 0),
            EXIT,
        ];
        let code: Vec<_> = program.iter().map(|&raw| decode(raw)).collect();
        let layout = Layout {
            loop_trace: true,
            ..Layout::workgroup([32, 1, 1])
        };
        let translated = crate::compute::translate(&code, &layout).unwrap();
        if let Some(problem) = crate::spirv::tests::validate(&translated.spirv) {
            panic!("{problem}");
        }
        let buffer = 0x5000u64;
        let inputs = Inputs {
            workgroup_count: [1, 1, 1],
            memory: BTreeMap::from([(buffer, vec![0; 64])]),
            specialization: HashMap::from([
                (LOOP_TRACE_LOW_SPEC_ID, buffer as u32),
                (LOOP_TRACE_HIGH_SPEC_ID, 0),
            ]),
            ..Inputs::default()
        };
        let report = Machine::new(&translated.spirv).simulate(&inputs);
        assert!(report.problems.is_empty(), "{:?}", report.problems);
        let words: Vec<u32> = report.memory[&buffer]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&word| u32::from_le_bytes(word))
            .collect();
        // 32 invocations x 10 iterations, 10 at most, the budget never
        // out, the header at instruction 2 (byte 0x20).
        assert_eq!(words[..4], [320, 10, 0, 0x21]);
        // Untraced (address 0): nothing written, nothing read.
        let report = Machine::new(&translated.spirv).simulate(&Inputs {
            workgroup_count: [1, 1, 1],
            ..Inputs::default()
        });
        assert!(report.problems.is_empty(), "{:?}", report.problems);
    }

    #[test]
    fn predicate_logic_reads_a_uniform_predicate_with_bit_67() {
        // UPLOP3.LUT UP0, UPT, UPT, UPT, UPT, 0xff, 0x0 (UP0 = true);
        // PLOP3.LUT P0, PT, PT, PT, UP0, 0x80, 0x0 (P0 = UP0)
        let set_uniform = word(&[
            (0, 12, 0x89c),
            (64, 3, 7),
            (68, 3, 7),
            (72, 5, 0x1f),
            (77, 3, 7),
            (81, 3, 0),
            (84, 3, 7),
            (87, 3, 7),
        ]);
        let from_uniform = word(&[
            (0, 12, 0x81c),
            (67, 1, 1),
            (68, 3, 0),
            (72, 5, 0x10),
            (77, 3, 7),
            (81, 3, 0),
            (84, 3, 7),
            (87, 3, 7),
        ]);
        let (_, predicates) = run(&[], &[set_uniform, from_uniform]);
        assert!(predicates[&0], "P0 takes UP0's value");
    }

    #[test]
    fn float_to_half_packing_rounds_by_its_mode() {
        // The halves just below and above a value, from every finite half.
        let around = |value: f32| -> (u16, u16) {
            let mut finite: Vec<u16> = (0u16..0x7c00).chain(0x8001..0xfc00).collect();
            finite.sort_by(|&x, &y| f16_to_f32(x).total_cmp(&f16_to_f32(y)));
            let below = finite
                .iter()
                .rev()
                .copied()
                .find(|&h| f16_to_f32(h) <= value)
                .unwrap_or(0xfc00);
            let above = finite
                .iter()
                .copied()
                .find(|&h| f16_to_f32(h) >= value)
                .unwrap_or(0x7c00);
            (below, above)
        };
        let reference = |value: f32, rounding: u32| -> u16 {
            let (below, above) = around(value);
            match rounding {
                0 => f64_to_f16(f64::from(value)),
                1 => below,
                2 => above,
                _ => {
                    if value >= 0.0 {
                        below
                    } else {
                        above
                    }
                }
            }
        };
        let values = [
            1.0001f32, -1.0001, 3.17, -0.1, 65519.0, 70000.0, -70000.0, 1.0e-6, -1.0e-6, 2.0,
        ];
        for rounding in 0..4u32 {
            for pair in values.chunks(2) {
                let (high, low) = (pair[0], pair[1]);
                // F2FP.PACK_AB R2, R0, R1 (R1 into the low half)
                let pack = word(&[
                    (0, 12, 0x23e),
                    (16, 8, 2),
                    (24, 8, 0),
                    (32, 8, 1),
                    (79, 2, u128::from(rounding)),
                ]);
                let (registers, _) = run(&[high.to_bits(), low.to_bits()], &[pack]);
                let expected = u32::from(reference(low, rounding))
                    | u32::from(reference(high, rounding)) << 16;
                assert_eq!(
                    registers[&2], expected,
                    "rounding {rounding}: {high}, {low}"
                );
            }
        }
    }

    /// The half nearest `value` (ties to an even significand), found among
    /// the finite halves rather than by the simulator's own conversion.
    fn nearest_half(halves: &[(f64, u16)], value: f64) -> u16 {
        if value.is_nan() {
            return 0x7e00;
        }
        // Beyond 65504 + half an ulp (32) rounds to infinity.
        if value.abs() >= 65520.0 {
            return if value > 0.0 { 0x7c00 } else { 0xfc00 };
        }
        let at = halves.partition_point(|&(h, _)| h < value);
        let candidates = [at.checked_sub(1), Some(at)];
        let mut best: Option<(f64, u16)> = None;
        for &(h, bits) in candidates.iter().flatten().filter_map(|&i| halves.get(i)) {
            let distance = (h - value).abs();
            best = match best {
                Some((d, b)) if d < distance || (d == distance && b & 1 == 0) => Some((d, b)),
                _ => Some((distance, bits)),
            };
        }
        let bits = best.unwrap().1;
        // Zero keeps the sign of what rounded to it.
        if bits & 0x7fff == 0 && value.is_sign_negative() {
            0x8000
        } else {
            bits
        }
    }

    #[test]
    fn half_adds_multiplies_and_fused_multiply_adds_round_once() {
        // Every finite half but -0, sorted.
        let mut halves: Vec<(f64, u16)> = (0u16..0x7c00)
            .chain(0x8001..0xfc00)
            .map(|bits| (f64::from(f16_to_f32(bits)), bits))
            .collect();
        halves.sort_by(|x, y| x.0.total_cmp(&y.0));
        let value = |bits: u16| f64::from(f16_to_f32(bits));
        // Halves whose magnitudes lie in [2^-10, 2^10] (so a * b + c is
        // exact in f64), zeros, and halves near the top of the range.
        let mut state = 0x2545_f491u32;
        let mut list: Vec<u16> = vec![0x0000, 0x8000, 0x3c00, 0xbc00, 0x7bff, 0x3555, 0xc0f3];
        while list.len() < 40 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let exponent = 5 + (state >> 8) % 21; // biased 5..=25
            list.push(((state >> 16) & 0x8000 | exponent << 10 | state & 0x3ff) as u16);
        }
        // A source's two halves after its swizzle, absolute value and
        // negation: the pair (0), the low half (2) or the high half (3).
        let operand = |word: u32, swizzle: u32, abs: bool, neg: bool| -> [f64; 2] {
            let (low, high) = (word as u16, (word >> 16) as u16);
            let pair = match swizzle {
                0 => [low, high],
                2 => [low, low],
                _ => [high, high],
            };
            pair.map(|h| {
                let h = if abs { h & 0x7fff } else { h };
                value(if neg { h ^ 0x8000 } else { h })
            })
        };
        let pack = |pair: [u16; 2]| u32::from(pair[0]) | u32::from(pair[1]) << 16;
        let mut case = 0u32;
        for window in list.windows(6).step_by(2) {
            let a = pack([window[0], window[1]]);
            let b = pack([window[2], window[3]]);
            let c = pack([window[4], window[5]]);
            for (swizzles, modifiers) in [
                (0u32, 0u32),
                (0x2_3, 0b01_10),
                (0x3_2, 0b10_01),
                (0x0_0, 0b11_11),
            ] {
                case += 1;
                let (sa, sb) = (swizzles >> 4 & 3, swizzles & 3);
                let (abs_a, neg_a) = (modifiers & 8 != 0, modifiers & 4 != 0);
                let (abs_b, neg_b) = (modifiers & 2 != 0, modifiers & 1 != 0);
                let saturate = case.is_multiple_of(3);
                let x = operand(a, sa, abs_a, neg_a);
                let y = operand(b, sb, abs_b, neg_b);
                let z = operand(c, 0, false, case.is_multiple_of(2));
                let fields = |opcode: u128| {
                    vec![
                        (0, 12, opcode),
                        (16, 8, 3),
                        (24, 8, 0),
                        (32, 8, 1),
                        (60, 2, u128::from(sb)),
                        (62, 1, u128::from(abs_b)),
                        (63, 1, u128::from(neg_b)),
                        (72, 1, u128::from(neg_a)),
                        (73, 1, u128::from(abs_a)),
                        (74, 2, u128::from(sa)),
                        (77, 1, u128::from(saturate)),
                    ]
                };
                let finish = |exact: [f64; 2]| {
                    pack(exact.map(|v| {
                        let v = if saturate { v.clamp(0.0, 1.0) } else { v };
                        // Saturation sends -0 (and NaN) to +0.
                        let v = if saturate && (v == 0.0 || v.is_nan()) {
                            0.0
                        } else {
                            v
                        };
                        nearest_half(&halves, v)
                    }))
                };
                let mut fma = fields(0x231);
                fma.extend([(64, 8, 2), (84, 1, u128::from(case.is_multiple_of(2)))]);
                let checks = [
                    (
                        "HADD2",
                        word(&fields(0x230)),
                        finish([x[0] + y[0], x[1] + y[1]]),
                    ),
                    (
                        "HMUL2",
                        word(&fields(0x232)),
                        finish([x[0] * y[0], x[1] * y[1]]),
                    ),
                    (
                        "HFMA2",
                        word(&fma),
                        finish([x[0] * y[0] + z[0], x[1] * y[1] + z[1]]),
                    ),
                ];
                for (name, instruction, expected) in checks {
                    let (registers, _) = run(&[a, b, c], &[instruction]);
                    assert_eq!(
                        registers[&3], expected,
                        "{name} {a:#010x} {b:#010x} {c:#010x} swizzles {swizzles:#x} \
                         modifiers {modifiers:#06b} saturate {saturate}"
                    );
                }
            }
        }
        // (1 + 2^-10)^2 - (1 + 2^-9) is 2^-20 fused, 0 if the product
        // were rounded first.
        let fma = word(&[
            (0, 12, 0x231),
            (16, 8, 3),
            (24, 8, 0),
            (32, 8, 1),
            (64, 8, 2),
        ]);
        let (registers, _) = run(&[0x3c01_3c01, 0x3c01_3c01, 0xbc02_bc02], &[fma]);
        assert_eq!(registers[&3], 0x0010_0010);
    }

    #[test]
    fn texture_samples_write_their_components_in_mask_order() {
        // TEX.LZ (2D, handle c[0x0][0x40]) with x in R0, y in R1, d = R4,
        // a second destination R8 (or RZ), a mask and packed halves.
        let sample = |mask: u8, d2: u8, half: bool| {
            word(&[
                (0, 12, 0xb60),
                (16, 8, 4),
                (24, 8, 0),
                (32, 8, 1),
                (40, 14, 0x10),
                (54, 5, 0),
                (61, 3, 1),
                (64, 8, u128::from(d2)),
                (72, 4, u128::from(mask)),
                (79, 1, u128::from(half)),
                (81, 3, 7),
                (87, 3, 1),
            ])
        };
        // A 4 x 4 texture whose texel (x, y) holds 1 + x + 4y + 0.25c in
        // component c (exact in halves).
        let component = |x: u32, y: u32, c: u32| 1.0 + (x + 4 * y) as f32 + 0.25 * c as f32;
        let texels: Vec<[u32; 4]> = (0..16u32)
            .map(|n| [0, 1, 2, 3].map(|c| component(n % 4, n / 4, c).to_bits()))
            .collect();
        for mask in 1..16u8 {
            for d2 in [255u8, 8] {
                for half in [false, true] {
                    let program = [
                        mov(0, 0.625f32.to_bits()),
                        mov(1, 0.375f32.to_bits()),
                        sample(mask, d2, half),
                        EXIT,
                    ];
                    let code: Vec<_> = program.iter().map(|&raw| decode(raw)).collect();
                    let layout = Layout::workgroup([1, 1, 1]);
                    let (translated, registers, _) =
                        translate_with_registers(&code, &layout).unwrap();
                    let texture = translated
                        .descriptors
                        .iter()
                        .find(|d| d.cbuf_offset == 0x40)
                        .unwrap();
                    let inputs = Inputs {
                        workgroup_count: [1, 1, 1],
                        images: HashMap::from([(
                            (0, texture.binding),
                            Image {
                                width: 4,
                                height: 4,
                                depth: 1,
                                layers: 1,
                                texels: texels.clone(),
                            },
                        )]),
                        ..Inputs::default()
                    };
                    let machine = Machine::new(&translated.spirv);
                    let report = machine.simulate(&inputs);
                    assert!(report.problems.is_empty(), "{:?}", report.problems);
                    let variables = &report.invocations[0];
                    let register = |n: u8| registers.get(&n).map(|v| variables[v][0] as u32);
                    // Texel (2, 1): the components in mask order, packed
                    // in pairs when half; components 2-3 (the third and
                    // fourth words, or the second packed word) go to R8 when
                    // it is a destination.
                    let mut words: Vec<u32> = (0..4u32)
                        .filter(|c| mask >> c & 1 != 0)
                        .map(|c| component(2, 1, c).to_bits())
                        .collect();
                    if half {
                        words = words
                            .chunks(2)
                            .map(|pair| {
                                let h = |bits: u32| u32::from(f32_to_f16(f32::from_bits(bits)));
                                h(pair[0]) | pair.get(1).map_or(0, |&b| h(b)) << 16
                            })
                            .collect();
                    }
                    let per_register = if half { 1 } else { 2 };
                    for (n, &expected) in words.iter().enumerate() {
                        let target = if d2 == 255 || n < per_register {
                            4 + n as u8
                        } else {
                            d2 + (n - per_register) as u8
                        };
                        assert_eq!(
                            register(target),
                            Some(expected),
                            "mask {mask:#x} d2 {d2} half {half}: word {n}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn divergent_branches_match_a_per_invocation_model() {
        // LOP3.LUT Rd, Ra, imm, RZ, a & b.
        let and = |d: u128, a: u128, imm: u32| {
            word(&[
                (0, 12, 0x812),
                (16, 8, d),
                (24, 8, a),
                (32, 32, u128::from(imm)),
                (64, 8, 255),
                (72, 8, 0xc0),
                (81, 3, 7),
            ])
        };
        let program = [
            // 0: S2R R0, SR_TID.X; 1: R2 = R0 & 7
            word(&[(0, 12, 0x919), (16, 8, 0), (72, 8, 0x21)]),
            and(2, 0, 7),
            // 2: ISETP.LT P0, R1, R2; 3: @!P0 BRA 7
            less(0, 1, 2),
            bra(3, 7, 8),
            // 4: R1 += 1; 5: R3 += 3; 6: BRA 2
            add(1, 1, 1),
            add(3, 3, 3),
            bra(6, 2, 7),
            // 7: R4 = R0 & 1; 8: ISETP.LT P1, RZ, R4; 9: @P1 BRA 11
            and(4, 0, 1),
            less(1, 255, 4),
            bra(9, 11, 1),
            // 10: R3 += 100; 11: R3 += 1
            add(3, 3, 100),
            add(3, 3, 1),
        ];
        // Two subgroups, the second partly filled.
        let (registers, _, problems) = run_group(40, &[], &program);
        assert!(problems.is_empty(), "{problems:?}");
        for (n, registers) in registers.iter().enumerate() {
            let n = n as u32;
            // The same control flow, run for one invocation.
            let mut r3 = 0;
            for _ in 0..n & 7 {
                r3 += 3;
            }
            if n & 1 == 0 {
                r3 += 100;
            }
            r3 += 1;
            assert_eq!(registers[&1], n & 7, "invocation {n}: iterations");
            assert_eq!(registers[&3], r3, "invocation {n}");
        }
    }

    #[test]
    fn texture_queries_give_the_bound_texture_s_dimensions() {
        // TXQ R4, (handle c[0x0][0x40]), level R0 = 0, a mask.
        let query = |mask: u8| {
            word(&[
                (0, 12, 0xb6f),
                (16, 8, 4),
                (24, 8, 0),
                (32, 8, 255),
                (40, 14, 0x10),
                (54, 5, 0),
                (64, 8, 255),
                (72, 4, u128::from(mask)),
            ])
        };
        // Width, height, depth (1 for a 2D texture) and level count.
        let expected = [6u32, 8, 1, 1];
        for mask in 1..16u8 {
            let code: Vec<_> = [mov(0, 0), query(mask), EXIT]
                .iter()
                .map(|&raw| decode(raw))
                .collect();
            let (translated, registers, _) =
                translate_with_registers(&code, &Layout::workgroup([1, 1, 1])).unwrap();
            let texture = translated
                .descriptors
                .iter()
                .find(|d| d.cbuf_offset == 0x40)
                .unwrap();
            let report = Machine::new(&translated.spirv).simulate(&Inputs {
                workgroup_count: [1, 1, 1],
                images: HashMap::from([(
                    (0, texture.binding),
                    Image {
                        width: 6,
                        height: 8,
                        depth: 1,
                        layers: 1,
                        texels: vec![[0; 4]; 48],
                    },
                )]),
                ..Inputs::default()
            });
            assert!(report.problems.is_empty(), "{:?}", report.problems);
            let written: Vec<u32> = (0..4)
                .filter(|c| mask >> c & 1 != 0)
                .map(|c| expected[c])
                .collect();
            for (n, &value) in written.iter().enumerate() {
                let variable = registers[&(4 + n as u8)];
                assert_eq!(
                    report.invocations[0][&variable][0] as u32, value,
                    "mask {mask:#x}: word {n}"
                );
            }
        }
    }

    #[test]
    fn bindless_samples_read_the_texture_pool_entry_their_handle_names() {
        use echora_shader_abi::{
            SAMPLER_INDEX_SHIFT, TEXTURE_POOL_CAPACITY_SPEC_ID, TEXTURE_POOL_SET, texture_pool,
        };
        // TEX.B.LZ R4, R0 (x, y in R0 and R1), R2 (the handle), 2D, xyzw.
        let sample = word(&[
            (0, 12, 0x361),
            (16, 8, 4),
            (24, 8, 0),
            (32, 8, 2),
            (59, 1, 1),
            (61, 3, 1),
            (64, 8, 255),
            (72, 4, 0xf),
            (81, 3, 7),
            (87, 3, 1),
        ]);
        // Entries 5 and 6 of the pool's 2D views: 2 x 2 textures whose
        // texel (x, y) holds the entry, x, y and 7 as floats.
        let entry = |index: u32| Image {
            width: 2,
            height: 2,
            depth: 1,
            layers: 1,
            texels: (0..4u32)
                .map(|n| [index as f32, (n % 2) as f32, (n / 2) as f32, 7.0].map(f32::to_bits))
                .collect(),
        };
        let image_arrays = HashMap::from([
            ((TEXTURE_POOL_SET, texture_pool::SAMPLED_2D, 5), entry(5)),
            ((TEXTURE_POOL_SET, texture_pool::SAMPLED_2D, 6), entry(6)),
            // Entry 6's unsigned view, read when its flag says integer.
            (
                (TEXTURE_POOL_SET, texture_pool::UINT_2D, 6),
                Image {
                    width: 2,
                    height: 2,
                    depth: 1,
                    layers: 1,
                    texels: (0..4u32).map(|n| [60, 61, 62, 60 + n]).collect(),
                },
            ),
        ]);
        let float = |values: [f32; 4]| values.map(f32::to_bits);
        // (handle, the pool's capacity, whether entry 6 is integer, the
        // texel expected at (1, 0)).
        let cases = [
            (
                5 | 3 << SAMPLER_INDEX_SHIFT,
                16,
                false,
                float([5.0, 1.0, 0.0, 7.0]),
            ),
            (6, 16, false, float([6.0, 1.0, 0.0, 7.0])),
            (6, 16, true, [60, 61, 62, 61]),
            // Past the capacity: zeros.
            (6, 6, false, [0; 4]),
        ];
        for (handle, capacity, integer, expected) in cases {
            // The flags buffer: a word per entry, bit 0 for integer.
            let flags: Vec<u8> = (0..16u32)
                .flat_map(|n| u32::from(integer && n == 6).to_le_bytes())
                .collect();
            let program = [
                mov(0, 0.75f32.to_bits()),
                mov(1, 0.25f32.to_bits()),
                mov(2, handle),
                sample,
                EXIT,
            ];
            let code: Vec<_> = program.iter().map(|&raw| decode(raw)).collect();
            let (translated, registers, _) =
                translate_with_registers(&code, &Layout::workgroup([1, 1, 1])).unwrap();
            let report = Machine::new(&translated.spirv).simulate(&Inputs {
                workgroup_count: [1, 1, 1],
                image_arrays: image_arrays.clone(),
                uniforms: HashMap::from([((TEXTURE_POOL_SET, texture_pool::FLAGS), flags)]),
                specialization: HashMap::from([(TEXTURE_POOL_CAPACITY_SPEC_ID, capacity)]),
                ..Inputs::default()
            });
            assert!(report.problems.is_empty(), "{:?}", report.problems);
            for (n, value) in expected.iter().enumerate() {
                let variable = registers[&(4 + n as u8)];
                assert_eq!(
                    report.invocations[0][&variable][0] as u32, *value,
                    "handle {handle:#x} capacity {capacity} integer {integer}: component {n}"
                );
            }
        }
    }

    #[test]
    fn samples_through_one_handle_share_its_pool_loads() {
        use echora_shader_abi::{TEXTURE_POOL_SET, texture_pool};
        // Four TEX.B.LZ R4.., R0 (x, y in R0 and R1), R2 (the handle), each
        // at x += 0.5 (as a MOV to R0 between them).
        let sample = word(&[
            (0, 12, 0x361),
            (16, 8, 4),
            (24, 8, 0),
            (32, 8, 2),
            (59, 1, 1),
            (61, 3, 1),
            (64, 8, 255),
            (72, 4, 0xf),
            (81, 3, 7),
            (87, 3, 1),
        ]);
        // R8 accumulates each sample's first component.
        let accumulate = word(&[
            (0, 12, 0x210),
            (16, 8, 8),
            (24, 8, 8),
            (32, 8, 4),
            (64, 8, 255),
            (77, 3, 7),
            (81, 3, 7),
            (84, 3, 7),
            (87, 3, 7),
        ]);
        let mut program = vec![mov(1, 0.25f32.to_bits()), mov(2, 3)];
        for x in [0.25f32, 0.75, 0.25, 0.75] {
            program.extend([mov(0, x.to_bits()), sample, accumulate]);
        }
        program.push(EXIT);
        let code: Vec<_> = program.iter().map(|&raw| decode(raw)).collect();
        let (translated, registers, _) =
            translate_with_registers(&code, &Layout::workgroup([1, 1, 1])).unwrap();
        assert_eq!(crate::spirv::tests::validate(&translated.spirv), None);
        // The handle's sampler, float view and flag are loaded once; each
        // sample loads the unsigned view in its integer branch.
        let chains = {
            let words = &translated.spirv;
            let mut at = 5;
            let mut count = 0;
            while at < words.len() {
                count += usize::from(words[at] & 0xffff == u32::from(op::ACCESS_CHAIN));
                at += (words[at] >> 16) as usize;
            }
            count
        };
        assert_eq!(chains, 3 + 4);
        // An integer entry 3 holding x + 10 in its first component.
        let image = Image {
            width: 2,
            height: 2,
            depth: 1,
            layers: 1,
            texels: (0..4u32).map(|n| [n % 2 + 10, 0, 0, 0]).collect(),
        };
        let flags: Vec<u8> = (0..4u32)
            .flat_map(|n| u32::from(n == 3).to_le_bytes())
            .collect();
        let report = Machine::new(&translated.spirv).simulate(&Inputs {
            workgroup_count: [1, 1, 1],
            image_arrays: HashMap::from([((TEXTURE_POOL_SET, texture_pool::UINT_2D, 3), image)]),
            uniforms: HashMap::from([((TEXTURE_POOL_SET, texture_pool::FLAGS), flags)]),
            ..Inputs::default()
        });
        assert!(report.problems.is_empty(), "{:?}", report.problems);
        assert_eq!(
            report.invocations[0][&registers[&8]][0] as u32,
            10 + 11 + 10 + 11
        );
    }

    #[test]
    fn a_loop_inside_a_branch_reads_its_registers_afresh_each_pass() {
        // R0 = 0, R1 = 3 (instructions 0 and 1).
        // 2: ISETP.LT P0, R0, R1; 3: @!P0 BRA 8;
        // 4: IADD3 R2, R2, R0 (R0 read before this pass writes it);
        // 5: R0 += 1; 6: ISETP.LT P1, R0, R1; 7: @P1 BRA 4
        let body = [
            less(0, 0, 1),
            bra(3, 8, 8),
            word(&[
                (0, 12, 0x210),
                (16, 8, 2),
                (24, 8, 2),
                (32, 8, 0),
                (64, 8, 255),
                (77, 3, 7),
                (81, 3, 7),
                (84, 3, 7),
                (87, 3, 7),
            ]),
            add(0, 0, 1),
            less(1, 0, 1),
            bra(7, 4, 1),
        ];
        let (r, _) = run(&[0, 3], &body);
        assert_eq!((r[&0], r[&2]), (3, 3));
        let (r, _) = run(&[5, 3], &body);
        assert_eq!((r[&0], r[&2]), (5, 0));
    }

    #[test]
    fn packed_cooperative_multiplies_give_the_same_results() {
        use crate::compute::MatrixPath;
        // Halves with magnitudes in [2^-4, 2^4] from a sequence, as each
        // lane's register bits.
        let halves = |seed: u32, lane: usize, register: u8| {
            let mut state = seed ^ (lane as u32) << 8 ^ u32::from(register) << 16;
            let mut half = || {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let exponent = 11 + (state >> 8) % 9;
                (state >> 16) & 0x8000 | exponent << 10 | state & 0x3ff
            };
            u64::from(half() | half() << 16)
        };
        for (k16, single, pair) in [
            (true, true, true),
            (true, false, true),
            (true, true, false),
            (false, false, false),
            (false, true, true),
        ] {
            let hmma = |d: u128, b: u128, c: u128| {
                word(&[
                    (0, 12, 0x23c),
                    (16, 8, d),
                    (24, 8, 0),
                    (32, 8, b),
                    (64, 8, c),
                    (75, 1, u128::from(k16)),
                    (76, 1, u128::from(single)),
                ])
            };
            // A in R0-R3, B in R4-R5 (and R6-R7), C from R8 (and R12), D
            // from R16 (and R20).
            let mut program = vec![hmma(16, 4, 8)];
            if pair {
                program.push(hmma(20, 6, 12));
            }
            program.push(EXIT);
            let code: Vec<_> = program.iter().map(|&raw| decode(raw)).collect();
            let run = |matrices| {
                let layout = Layout {
                    matrices,
                    ..Layout::workgroup([32, 1, 1])
                };
                let (translated, registers, _) = translate_with_registers(&code, &layout).unwrap();
                let report = Machine::new(&translated.spirv).run_workgroup_with(
                    32,
                    32,
                    |lane| {
                        HashMap::from([
                            (27, vec![lane as u64, 0, 0]),
                            (29, vec![lane as u64]),
                            (40, vec![0]),
                            (41, vec![lane as u64]),
                        ])
                    },
                    |lane| {
                        (0..16u8)
                            .filter_map(|n| {
                                let value = if single && n >= 8 {
                                    // A float C: a half's value widened.
                                    let bits = halves(7, lane, n) as u16;
                                    u64::from(f16_to_f32(bits).to_bits())
                                } else {
                                    halves(7, lane, n)
                                };
                                registers.get(&n).map(|&variable| (variable, vec![value]))
                            })
                            .collect()
                    },
                );
                assert!(
                    report.problems.is_empty(),
                    "{matrices:?}: {:?}",
                    report.problems
                );
                (0..32)
                    .map(|lane| {
                        (16..24u8)
                            .map(|n| {
                                registers
                                    .get(&n)
                                    .map(|v| report.invocations[lane][v][0] as u32)
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            };
            let (plain, packed) = (run(MatrixPath::Cooperative), run(MatrixPath::Cooperative32));
            assert!(plain[0][0].is_some());
            assert_eq!(plain, packed, "k16 {k16} single {single} pair {pair}");
        }
    }

    #[test]
    fn sixty_four_bit_arithmetic_carries_like_the_hardware() {
        /// IADD3 Rd, Pp, Pq, Ra, (-)Rb, Rc with carries in Pr, Ps (.X when
        /// `extended`; negation bits as given; 15 = !PT).
        #[allow(clippy::too_many_arguments)]
        fn iadd3(
            d: u128,
            p: u128,
            q: u128,
            a: u128,
            b: u128,
            negate_b: bool,
            c: u128,
            extended: bool,
            r: u128,
            s: u128,
        ) -> u128 {
            word(&[
                (0, 12, 0x210),
                (16, 8, d),
                (24, 8, a),
                (32, 8, b),
                (63, 1, u128::from(negate_b)),
                (64, 8, c),
                (74, 1, u128::from(extended)),
                (77, 4, s),
                (81, 3, p),
                (84, 3, q),
                (87, 4, r),
            ])
        }
        let pair = |r: &HashMap<u8, u32>, n: u8| u64::from(r[&n]) | u64::from(r[&(n + 1)]) << 32;
        let list = values(16);
        for window in list.windows(6).step_by(2) {
            let inputs = [
                window[0], window[1], window[2], window[3], window[4], window[5],
            ];
            let x = u64::from(inputs[0]) | u64::from(inputs[1]) << 32;
            let y = u64::from(inputs[2]) | u64::from(inputs[3]) << 32;
            let z = u64::from(inputs[4]) | u64::from(inputs[5]) << 32;
            // x + y: IADD3 R6, P0, PT, R0, R2, RZ; IADD3.X R7, R1, R3, RZ,
            // P0, !PT.
            let (r, _) = run(
                &inputs,
                &[
                    iadd3(6, 0, 7, 0, 2, false, 255, false, 7, 7),
                    iadd3(7, 7, 7, 1, 3, false, 255, true, 0, 15),
                ],
            );
            assert_eq!(pair(&r, 6), x.wrapping_add(y), "{x:#x} + {y:#x}");
            // x - y: IADD3 R6, P0, PT, R0, -R2, RZ; IADD3.X R7, R1, ~R3, RZ,
            // P0, !PT.
            let (r, _) = run(
                &inputs,
                &[
                    iadd3(6, 0, 7, 0, 2, true, 255, false, 7, 7),
                    iadd3(7, 7, 7, 1, 3, true, 255, true, 0, 15),
                ],
            );
            assert_eq!(pair(&r, 6), x.wrapping_sub(y), "{x:#x} - {y:#x}");
            // x + y + z: IADD3 R6, P0, P1, R0, R2, R4; IADD3.X R7, R1, R3,
            // R5, P0, P1.
            let (r, _) = run(
                &inputs,
                &[
                    iadd3(6, 0, 1, 0, 2, false, 4, false, 7, 7),
                    iadd3(7, 7, 7, 1, 3, false, 5, true, 0, 1),
                ],
            );
            assert_eq!(
                pair(&r, 6),
                x.wrapping_add(y).wrapping_add(z),
                "{x:#x} + {y:#x} + {z:#x}"
            );
            // y + (x << s): LEA R6, P0, R0, R2, s; LEA.HI.X R7, R0, R3, R1,
            // s, P0.
            for shift in [0u128, 1, 3, 17, 31] {
                let lea = |d: u128, b: u128, c: u128, high: bool, p: u128, carry: u128| {
                    word(&[
                        (0, 12, 0x211),
                        (16, 8, d),
                        (24, 8, 0),
                        (32, 8, b),
                        (64, 8, c),
                        (74, 1, u128::from(high)),
                        (75, 5, shift),
                        (80, 1, u128::from(high)),
                        (81, 3, p),
                        (87, 4, carry),
                    ])
                };
                let (r, _) = run(
                    &inputs,
                    &[lea(6, 2, 255, false, 0, 7), lea(7, 3, 1, true, 7, 0)],
                );
                assert_eq!(
                    pair(&r, 6),
                    y.wrapping_add(x << shift),
                    "{y:#x} + {x:#x} << {shift}"
                );
            }
            // IMAD.WIDE(.U32) R6, R0, R2, R4 and IMAD.HI(.U32) R6, R0, R2,
            // R4: a * b + R5:R4, the pair or its high word.
            for signed in [false, true] {
                let imad = |opcode: u128| {
                    word(&[
                        (0, 12, opcode),
                        (16, 8, 6),
                        (24, 8, 0),
                        (32, 8, 2),
                        (64, 8, 4),
                        (73, 1, u128::from(signed)),
                        (81, 3, 7),
                    ])
                };
                let product = if signed {
                    i64::from(inputs[0] as i32).wrapping_mul(i64::from(inputs[2] as i32)) as u64
                } else {
                    u64::from(inputs[0]) * u64::from(inputs[2])
                };
                let expected = product.wrapping_add(z);
                let (r, _) = run(&inputs, &[imad(0x225)]);
                assert_eq!(pair(&r, 6), expected, "IMAD.WIDE signed {signed}");
                let (r, _) = run(&inputs, &[imad(0x227)]);
                assert_eq!(r[&6], (expected >> 32) as u32, "IMAD.HI signed {signed}");
            }
        }
        // Division by 7 as a compiler writes it: q = hi(n * 0x24924925),
        // then (((n - q) >> 1) + q) >> 2. Here only the high multiply:
        // IMAD.HI.U32 R1, R0, 0x24924925 (immediate form), RZ.
        let high = word(&[
            (0, 12, 0x827),
            (16, 8, 1),
            (24, 8, 0),
            (32, 32, 0x2492_4925),
            (64, 8, 255),
            (81, 3, 7),
        ]);
        for n in values(32) {
            let (r, _) = run(&[n], &[high]);
            let q = r[&1];
            assert_eq!(q, ((u64::from(n) * 0x2492_4925) >> 32) as u32);
            assert_eq!((((n - q) >> 1) + q) >> 2, n / 7, "{n} / 7");
        }
    }

    #[test]
    fn votes_and_reductions_count_only_the_lanes_still_running() {
        // `raw` under guard predicate `guard` (index, plus 8 negated).
        let guarded = |raw: u128, guard: u128| raw & !(0xf << 12) | guard << 12;
        let and = |d: u128, a: u128, imm: u32| {
            word(&[
                (0, 12, 0x812),
                (16, 8, d),
                (24, 8, a),
                (32, 32, u128::from(imm)),
                (64, 8, 255),
                (72, 8, 0xc0),
                (81, 3, 7),
            ])
        };
        // ISETP.EQ.U32.AND P2, PT, R1, R5, PT
        let equal = word(&[
            (0, 12, 0x20c),
            (24, 8, 1),
            (32, 8, 5),
            (76, 3, 2),
            (81, 3, 2),
            (84, 3, 7),
            (87, 3, 7),
        ]);
        // VOTE.mode Rd, Pp, P0
        let vote = |d: u128, p: u128, mode: u128| {
            word(&[
                (0, 12, 0x806),
                (16, 8, d),
                (72, 2, mode),
                (81, 3, p),
                (87, 3, 0),
            ])
        };
        // REDUX.op URd, R0; MOV Rd, URd
        let redux = |d: u128, operation: u128| {
            word(&[(0, 12, 0x3c4), (16, 6, d), (24, 8, 0), (78, 3, operation)])
        };
        let from_uniform =
            |d: u128, u: u128| word(&[(0, 12, 0xc02), (72, 4, 0xf), (16, 8, d), (32, 6, u)]);
        let program = [
            // S2R R0, SR_TID.X; R1 = R0 & 7; @(R1 == 5) EXIT
            word(&[(0, 12, 0x919), (16, 8, 0), (72, 8, 0x21)]),
            and(1, 0, 7),
            equal,
            guarded(EXIT, 2),
            // P0 = R0 odd
            and(2, 0, 1),
            less(0, 255, 2),
            // VOTE.ANY R6, P1, P0; @P1 R7 += 1; VOTE.ALL R8, P3, P0;
            // @P3 R9 += 1
            vote(6, 1, 1),
            guarded(add(7, 7, 1), 1),
            vote(8, 3, 0),
            guarded(add(9, 9, 1), 3),
            // REDUX.SUM, .MAX, .MIN, .OR of R0 into R10-R13
            redux(4, 3),
            from_uniform(10, 4),
            redux(5, 5),
            from_uniform(11, 5),
            redux(6, 4),
            from_uniform(12, 6),
            redux(7, 1),
            from_uniform(13, 7),
        ];
        let (registers, _, problems) = run_group(40, &[0, 0, 0, 0, 0, 5], &program);
        assert!(problems.is_empty(), "{problems:?}");
        for (n, r) in registers.iter().enumerate() {
            if n & 7 == 5 {
                // Exited before voting.
                assert_eq!((r[&6], r[&10]), (0, 0), "invocation {n}");
                continue;
            }
            // The invocations of n's subgroup still running.
            let running: Vec<u32> = (n / 32 * 32..(n / 32 * 32 + 32).min(40))
                .map(|m| m as u32)
                .filter(|m| m & 7 != 5)
                .collect();
            let ballot = running
                .iter()
                .filter(|&&m| m & 1 == 1)
                .fold(0u32, |mask, &m| mask | 1 << (m % 32));
            assert_eq!(r[&6], ballot, "invocation {n}: VOTE.ANY's ballot");
            assert_eq!(r[&8], ballot, "invocation {n}: VOTE.ALL's ballot");
            assert_eq!(r[&7], u32::from(ballot != 0), "invocation {n}: any");
            let all = running.iter().all(|m| m & 1 == 1);
            assert_eq!(r[&9], u32::from(all), "invocation {n}: all");
            assert_eq!(r[&10], running.iter().sum::<u32>(), "invocation {n}: sum");
            assert_eq!(
                r[&11],
                *running.iter().max().unwrap(),
                "invocation {n}: max"
            );
            assert_eq!(
                r[&12],
                *running.iter().min().unwrap(),
                "invocation {n}: min"
            );
            let or = running.iter().fold(0, |x, m| x | m);
            assert_eq!(r[&13], or, "invocation {n}: or");
        }
    }

    #[test]
    fn global_atomics_return_each_invocation_its_own_old_value() {
        use echora_shader_abi::{PAGE_SHIFT, PUSH_PAGE_TABLE_OFFSET};
        // R4:R5 = guest 0x10_0001_0000 (past 32 bits).
        // ATOMG.E.ADD.STRONG.GPU R2, [R4.64+0x8], R6 (R6 = 3), then
        // RED.E.ADD.STRONG.GPU [R4.64+0xc], R6, register address (bit 91
        // clear: .E is bit 72).
        let atom = word(&[
            (0, 12, 0x3a8),
            (16, 8, 2),
            (24, 8, 4),
            (32, 8, 6),
            (40, 24, 8),
            (72, 1, 1),
            (87, 4, 0),
        ]);
        let red = word(&[
            (0, 12, 0x98e),
            (24, 8, 4),
            (32, 8, 6),
            (40, 24, 0xc),
            (72, 1, 1),
            (87, 3, 0),
        ]);
        let guest = 0x10_0001_0000u64;
        let program = [
            mov(4, guest as u32),
            mov(5, (guest >> 32) as u32),
            mov(6, 3),
            atom,
            red,
            EXIT,
        ];
        let code: Vec<_> = program.iter().map(|&raw| decode(raw)).collect();
        let (translated, registers, _) =
            translate_with_registers(&code, &Layout::workgroup([64, 1, 1])).unwrap();
        // The guest page at device 0x3000_0000; words 2 and 3 start at 100
        // and 1000.
        let (table, page) = (0x2000_0000u64, 0x3000_0000u64);
        let mut words = vec![0u8; 64];
        words[8..12].copy_from_slice(&100u32.to_le_bytes());
        words[12..16].copy_from_slice(&1000u32.to_le_bytes());
        let mut push = vec![0u8; 128];
        let at = PUSH_PAGE_TABLE_OFFSET as usize;
        push[at..at + 8].copy_from_slice(&table.to_le_bytes());
        push[at + 8..at + 12].copy_from_slice(&((guest >> PAGE_SHIFT) as u32).to_le_bytes());
        push[at + 12..at + 16].copy_from_slice(&1u32.to_le_bytes());
        let inputs = Inputs {
            workgroup_count: [1, 1, 1],
            push,
            memory: BTreeMap::from([(table, page.to_le_bytes().to_vec()), (page, words)]),
            ..Inputs::default()
        };
        let report = Machine::new(&translated.spirv).simulate(&inputs);
        assert!(report.problems.is_empty(), "{:?}", report.problems);
        assert!(
            report.unmapped_pages.is_empty(),
            "{:?}",
            report.unmapped_pages
        );
        let memory = &report.memory[&page];
        let word = |n: usize| u32::from_le_bytes(memory[4 * n..4 * n + 4].try_into().unwrap());
        assert_eq!((word(2), word(3)), (100 + 64 * 3, 1000 + 64 * 3));
        // Each invocation's old value: 100, 103, ..., in some order.
        let mut olds: Vec<u32> = report
            .invocations
            .iter()
            .map(|variables| variables[&registers[&2]][0] as u32)
            .collect();
        olds.sort_unstable();
        assert_eq!(olds, (0..64).map(|n| 100 + 3 * n).collect::<Vec<u32>>());
    }

    #[test]
    fn each_invocation_s_guest_exit_is_reported() {
        // 0: MOV R1, 40; 1: S2R R0, SR_TID.X; 2: ISETP.GE.U32 P0, R0, R1;
        // 3: @P0 EXIT; 4: (BAR or nothing); 5: EXIT.
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
        let early = EXIT & !(0xf << 12);
        // NOP (0x918) where there is no barrier.
        for (fourth, flag) in [(BAR, true), (0x918 | 7 << 12, false)] {
            let program = [mov(1, 40), s2r, at_least, early, fourth, EXIT];
            let code: Vec<_> = program.iter().map(|&raw| decode(raw)).collect();
            let translated =
                crate::compute::translate(&code, &Layout::workgroup([64, 1, 1])).unwrap();
            assert_eq!(crate::spirv::tests::validate(&translated.spirv), None);
            let report = Machine::new(&translated.spirv).simulate(&Inputs {
                workgroup_count: [1, 1, 1],
                ..Inputs::default()
            });
            assert!(report.problems.is_empty(), "{:?}", report.problems);
            for (n, &exit) in report.exits.iter().enumerate() {
                let expected = if n >= 40 { 3 * 16 } else { 5 * 16 };
                assert_eq!(exit, Some(expected), "barrier {flag}: invocation {n}");
            }
        }
    }

    #[test]
    fn multiplies_with_dnz_make_zero_factors_give_plus_zero() {
        let (inf, nan) = (f32::INFINITY.to_bits(), f32::NAN.to_bits());
        let float = |x: f32| x.to_bits();
        // FMUL(.DNZ) R3, R0, R1 and FFMA(.DNZ) R3, R0, R1, R2.
        let fmul = |dnz: u128| {
            word(&[
                (0, 12, 0x220),
                (16, 8, 3),
                (24, 8, 0),
                (32, 8, 1),
                (76, 1, dnz),
                (84, 3, 4),
            ])
        };
        let ffma = |dnz: u128| {
            word(&[
                (0, 12, 0x223),
                (16, 8, 3),
                (24, 8, 0),
                (32, 8, 1),
                (64, 8, 2),
                (76, 1, dnz),
            ])
        };
        // (a, b, c, a*b with DNZ, a*b + c with DNZ)
        let cases = [
            (0, inf, float(2.5), 0, float(2.5)),
            (inf, float(-0.0), float(-0.0), 0, 0),
            (0, nan, float(-1.0), 0, float(-1.0)),
            (float(-0.0), float(3.0), float(1.0), 0, float(1.0)),
            (
                float(1.5),
                float(-2.0),
                float(0.5),
                float(-3.0),
                float(-2.5),
            ),
            (inf, float(2.0), float(1.0), inf, inf),
        ];
        for (a, b, c, product, sum) in cases {
            let (r, _) = run(&[a, b, c], &[fmul(1)]);
            assert_eq!(r[&3], product, "FMUL.DNZ {a:#x} {b:#x}");
            let (r, _) = run(&[a, b, c], &[ffma(1)]);
            assert_eq!(r[&3], sum, "FFMA.DNZ {a:#x} {b:#x} {c:#x}");
        }
        // Without DNZ, 0 * Inf is NaN.
        let (r, _) = run(&[0, inf], &[fmul(0)]);
        assert!(f32::from_bits(r[&3]).is_nan());
        // HMUL2.DNZ / HFMA2.DNZ R3, R0, R1(, R2), lane by lane: low lanes 0 *
        // Inf (+ 1.0), high lanes 2.0 * 3.0 (+ 1.0).
        let pack = |low: u32, high: u32| low | high << 16;
        let (a, b, c) = (
            pack(0x0000, 0x4000),
            pack(0x7c00, 0x4200),
            pack(0x3c00, 0x3c00),
        );
        let hmul = word(&[
            (0, 12, 0x232),
            (16, 8, 3),
            (24, 8, 0),
            (32, 8, 1),
            (76, 1, 1),
        ]);
        let (r, _) = run(&[a, b, c], &[hmul]);
        assert_eq!(r[&3], pack(0, 0x4600), "HMUL2.DNZ");
        let hfma = word(&[
            (0, 12, 0x231),
            (16, 8, 3),
            (24, 8, 0),
            (32, 8, 1),
            (64, 8, 2),
            (76, 1, 1),
        ]);
        let (r, _) = run(&[a, b, c], &[hfma]);
        assert_eq!(r[&3], pack(0x3c00, 0x4700), "HFMA2.DNZ");
        let code: Vec<_> = [hmul, hfma, ffma(1), fmul(1), EXIT]
            .iter()
            .map(|&raw| decode(raw))
            .collect();
        let translated = crate::compute::translate(&code, &Layout::workgroup([1, 1, 1])).unwrap();
        assert_eq!(crate::spirv::tests::validate(&translated.spirv), None);
    }

    #[test]
    fn saturation_sends_nan_to_zero_and_packing_keeps_specials() {
        let (inf, nan) = (f32::INFINITY.to_bits(), f32::NAN.to_bits());
        // FADD.SAT R2, R0, R1.
        let fadd = word(&[
            (0, 12, 0x221),
            (16, 8, 2),
            (24, 8, 0),
            (32, 8, 1),
            (77, 1, 1),
        ]);
        for (a, b, expected) in [
            (inf, (-f32::INFINITY).to_bits(), 0),
            (nan, 0, 0),
            ((-0.0f32).to_bits(), 0, 0),
            ((-2.0f32).to_bits(), 0, 0),
            (inf, 0, 1.0f32.to_bits()),
            (0.25f32.to_bits(), 0, 0.25f32.to_bits()),
        ] {
            let (r, _) = run(&[a, b], &[fadd]);
            assert_eq!(r[&2], expected, "FADD.SAT {a:#x} + {b:#x}");
        }
        // HADD2.SAT R2, R0, R1: a NaN lane and an Inf lane.
        let hadd = word(&[
            (0, 12, 0x230),
            (16, 8, 2),
            (24, 8, 0),
            (32, 8, 1),
            (77, 1, 1),
        ]);
        let (r, _) = run(&[0x7c00_7e00, 0], &[hadd]);
        assert_eq!(r[&2], 0x3c00_0000, "HADD2.SAT");
        // F2FP.PACK_AB R2, R0, R1 (round to nearest): NaN stays NaN, ±Inf
        // stay, 65520 rounds to +Inf, -1e9 to -Inf.
        let pack = word(&[(0, 12, 0x23e), (16, 8, 2), (24, 8, 0), (32, 8, 1)]);
        for (high, low, expected_high, expected_low) in [
            (inf, (-f32::INFINITY).to_bits(), 0x7c00, 0xfc00),
            (65520.0f32.to_bits(), (-1.0e9f32).to_bits(), 0x7c00, 0xfc00),
        ] {
            let (r, _) = run(&[high, low], &[pack]);
            assert_eq!(
                r[&2],
                expected_low | expected_high << 16,
                "F2FP {high:#x} {low:#x}"
            );
        }
        let (r, _) = run(&[nan, 0], &[pack]);
        assert!(
            r[&2] >> 16 & 0x7c00 == 0x7c00 && r[&2] >> 16 & 0x3ff != 0,
            "{:#x}",
            r[&2]
        );
    }

    #[test]
    fn texel_offsets_move_the_sample_and_never_divide_by_zero() {
        // TEX.LZ (2D, offsets) R4, R0 (x, y), R2 (packed offsets: x in bits
        // 0-3, y in 4-7), handle c[0x0][0x40].
        let sample = word(&[
            (0, 12, 0xb60),
            (16, 8, 4),
            (24, 8, 0),
            (32, 8, 2),
            (40, 14, 0x10),
            (61, 3, 1),
            (64, 8, 255),
            (72, 4, 0xf),
            (76, 1, 1),
            (81, 3, 7),
            (87, 3, 1),
        ]);
        // Texel (2, 1), offset by (+1, -1): texel (3, 0).
        let program = [
            mov(0, 0.625f32.to_bits()),
            mov(1, 0.375f32.to_bits()),
            mov(2, 0x01 | 0xf << 4),
            sample,
            EXIT,
        ];
        let code: Vec<_> = program.iter().map(|&raw| decode(raw)).collect();
        let (translated, registers, _) =
            translate_with_registers(&code, &Layout::workgroup([1, 1, 1])).unwrap();
        assert_eq!(crate::spirv::tests::validate(&translated.spirv), None);
        let texture = translated
            .descriptors
            .iter()
            .find(|d| d.cbuf_offset == 0x40)
            .unwrap();
        let texels = (0..16u32).map(|n| [n, 0, 0, 0]).collect();
        let machine = Machine::new(&translated.spirv);
        let image = Image {
            width: 4,
            height: 4,
            depth: 1,
            layers: 1,
            texels,
        };
        let report = machine.simulate(&Inputs {
            workgroup_count: [1, 1, 1],
            images: HashMap::from([((0, texture.binding), image)]),
            ..Inputs::default()
        });
        assert!(report.problems.is_empty(), "{:?}", report.problems);
        assert_eq!(report.invocations[0][&registers[&4]][0], 3);
        // Unbound (size 0): no NaN coordinate.
        let report = machine.simulate(&Inputs {
            workgroup_count: [1, 1, 1],
            ..Inputs::default()
        });
        assert_eq!(report.first_nan, vec![None]);
    }

    #[test]
    fn mufu_flushes_subnormal_inputs_and_logs_zero_to_minus_infinity() {
        // MUFU.function R1, R0.
        let mufu =
            |function: u128| word(&[(0, 12, 0x308), (16, 8, 1), (32, 8, 0), (74, 4, function)]);
        let tiny = 1.0e-40f32.to_bits();
        let negative_tiny = (-1.0e-40f32).to_bits();
        let cases = [
            (3, 0, f32::NEG_INFINITY),
            (3, (-0.0f32).to_bits(), f32::NEG_INFINITY),
            (3, tiny, f32::NEG_INFINITY),
            (3, 8.0f32.to_bits(), 3.0),
            (4, tiny, f32::INFINITY),
            (4, negative_tiny, f32::NEG_INFINITY),
            (4, 4.0f32.to_bits(), 0.25),
            (5, tiny, f32::INFINITY),
            (2, tiny, 1.0),
        ];
        for (function, input, expected) in cases {
            let (r, _) = run(&[input], &[mufu(function)]);
            assert_eq!(
                f32::from_bits(r[&1]),
                expected,
                "MUFU {function} of {input:#x}"
            );
        }
    }

    #[test]
    fn exact_matrix_multiplies_match_the_tensor_core_model() {
        use crate::compute::MatrixPath;
        use crate::compute::matrix::tests::block_sum;
        // Halves with exponents from 2^-10 to 2^10 (so alignment truncates)
        // from a sequence, by matrix and position.
        let half = |seed: u64, at: u64| -> u16 {
            let mut state =
                seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ at.wrapping_mul(0xbf58_476d_1ce4_e5b9);
            state ^= state >> 29;
            state = state.wrapping_mul(0x94d0_49bb_1331_11eb);
            state ^= state >> 32;
            let exponent = 5 + (state >> 20) % 21;
            ((state >> 8) as u16 & 0x8000) | (exponent as u16) << 10 | (state as u16 & 0x3ff)
        };
        let value = |h: u16| f16_to_f32(h);
        for k16 in [true, false] {
            for single in [true, false] {
                let depth = if k16 { 16 } else { 8 };
                let a = |row: usize, k: usize| half(1, (row * 16 + k) as u64);
                let b =
                    |n: usize, k: usize, column: usize| half(2 + n as u64, (k * 8 + column) as u64);
                let c = |n: usize, row: usize, column: usize| {
                    half(4 + n as u64, (row * 8 + column) as u64)
                };
                let hmma = |d: u128, b: u128, c: u128| {
                    word(&[
                        (0, 12, 0x23c),
                        (16, 8, d),
                        (24, 8, 0),
                        (32, 8, b),
                        (64, 8, c),
                        (75, 1, u128::from(k16)),
                        (76, 1, u128::from(single)),
                    ])
                };
                // A in R0-R3; B in R4-R5 and R6-R7; C from R8 and R12; D
                // from R16 and R20.
                let program = [hmma(16, 4, 8), hmma(20, 6, 12), EXIT];
                let code: Vec<_> = program.iter().map(|&raw| decode(raw)).collect();
                let layout = Layout {
                    matrices: MatrixPath::Exact,
                    ..Layout::workgroup([32, 1, 1])
                };
                let (translated, registers, _) = translate_with_registers(&code, &layout).unwrap();
                let pair = |low: u16, high: u16| u64::from(low) | u64::from(high) << 16;
                let initial = |lane: usize| {
                    let (g, t) = (lane / 4, lane % 4);
                    let mut values: HashMap<u8, u64> = HashMap::new();
                    for j in 0..depth / 4 {
                        let row = g + 8 * (j & 1);
                        let k = 2 * t + 8 * (j >> 1);
                        values.insert(j as u8, pair(a(row, k), a(row, k + 1)));
                    }
                    for n in 0..2 {
                        for j in 0..depth / 8 {
                            let k = 2 * t + 8 * j;
                            values.insert((4 + 2 * n + j) as u8, pair(b(n, k, g), b(n, k + 1, g)));
                        }
                        for i in 0..4 {
                            let (row, column) = (g + 8 * (i >> 1), 2 * t + (i & 1));
                            let h = c(n, row, column);
                            if single {
                                values.insert((8 + 4 * n + i) as u8, u64::from(value(h).to_bits()));
                            } else {
                                let register = (8 + 4 * n + i / 2) as u8;
                                let old = values.get(&register).copied().unwrap_or(0);
                                values.insert(register, old | u64::from(h) << (16 * (i & 1)));
                            }
                        }
                    }
                    values
                        .into_iter()
                        .filter_map(|(register, value)| {
                            registers
                                .get(&register)
                                .map(|&variable| (variable, vec![value]))
                        })
                        .collect()
                };
                let report = Machine::new(&translated.spirv).run_workgroup_with(
                    32,
                    32,
                    |lane| {
                        HashMap::from([
                            (27, vec![lane as u64, 0, 0]),
                            (29, vec![lane as u64]),
                            (40, vec![0]),
                            (41, vec![lane as u64]),
                        ])
                    },
                    initial,
                );
                assert!(report.problems.is_empty(), "{:?}", report.problems);
                for (lane, variables) in report.invocations.iter().enumerate() {
                    let (g, t) = (lane / 4, lane % 4);
                    let register = |n: u8| variables[&registers[&n]][0] as u32;
                    for n in 0..2 {
                        for i in 0..4 {
                            let (row, column) = (g + 8 * (i >> 1), 2 * t + (i & 1));
                            let blocks = depth / 8;
                            let mut sum = value(c(n, row, column));
                            for m in 0..blocks {
                                let products: Vec<f32> = (8 * m..8 * m + 8)
                                    .map(|k| value(a(row, k)) * value(b(n, k, column)))
                                    .collect();
                                sum = block_sum(sum, &products, !single && m + 1 == blocks);
                            }
                            let (got, expected) = if single {
                                (register((16 + 4 * n + i) as u8), sum.to_bits())
                            } else {
                                let word = register((16 + 4 * n + i / 2) as u8);
                                (
                                    word >> (16 * (i & 1)) & 0xffff,
                                    u32::from(f64_to_f16(f64::from(sum))),
                                )
                            };
                            assert_eq!(
                                got, expected,
                                "k16 {k16} single {single} lane {lane} D{n}[{i}]"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn interpolated_attributes_read_as_the_hardware_hands_them_over() {
        use crate::header::{Header, Interpolation};
        use echora_shader_abi::Stage;
        // IPA R0, a[0x7c] (1/w); MUFU.RCP R0, R0 (w); IPA R1, a[0x80];
        // FMUL R1, R1, R0: the program's own perspective division.
        let ipa = |d: u128, address: u128| {
            word(&[
                (0, 12, 0x326),
                (16, 8, d),
                (24, 8, 255),
                (32, 8, 255),
                (64, 8, address / 4),
                (81, 3, 7),
            ])
        };
        let rcp = word(&[(0, 12, 0x308), (16, 8, 0), (32, 8, 0), (74, 4, 4)]);
        let multiply = word(&[
            (0, 12, 0x220),
            (16, 8, 1),
            (24, 8, 1),
            (32, 8, 0),
            (84, 3, 4),
        ]);
        let code: Vec<_> = [ipa(0, 0x7c), rcp, ipa(1, 0x80), multiply, EXIT]
            .iter()
            .map(|&raw| decode(raw))
            .collect();
        // Attribute 0's x is 2.0 (Vulkan's perspective-corrected value), 1/w
        // is 0.25: perspective gives 2.0 back, screen-linear 2.0 * 4.
        for (mode, expected) in [
            (Interpolation::Perspective, 2.0f32),
            (Interpolation::ScreenLinear, 8.0),
        ] {
            let mut imap = [Interpolation::Unused; 128];
            imap[..4].fill(mode);
            let header = Header {
                stage: Stage::Fragment,
                imap,
                targets: 0,
                writes_depth: false,
                writes_sample_mask: false,
                local_bytes: Some(0),
                outputs: [0; 5],
                outputs_beyond: false,
            };
            let layout = Layout {
                stage: Stage::Fragment,
                header: Some(header),
                ..Layout::workgroup([1, 1, 1])
            };
            let (translated, registers, _) = translate_with_registers(&code, &layout).unwrap();
            assert_eq!(crate::spirv::tests::validate(&translated.spirv), None);
            let bits = |values: [f32; 4]| values.iter().map(|v| u64::from(v.to_bits())).collect();
            let report = Machine::new(&translated.spirv).simulate(&Inputs {
                workgroup_count: [1, 1, 1],
                builtins: HashMap::from([(15, bits([10.5, 20.5, 0.5, 0.25]))]),
                locations: HashMap::from([(0, bits([2.0, 0.0, 0.0, 0.0]))]),
                ..Inputs::default()
            });
            assert!(report.problems.is_empty(), "{:?}", report.problems);
            let r1 = report.invocations[0][&registers[&1]][0] as u32;
            assert_eq!(f32::from_bits(r1), expected, "{mode:?}");
        }
    }

    #[test]
    fn a_vector_mixing_interpolations_reads_each_component_as_its_own() {
        use crate::header::{Header, Interpolation};
        use echora_shader_abi::Stage;
        // IPA R1, a[0x80]; IPA R2, a[0x84]: x perspective, y screen-linear.
        let ipa = |d: u128, address: u128| {
            word(&[
                (0, 12, 0x326),
                (16, 8, d),
                (24, 8, 255),
                (32, 8, 255),
                (64, 8, address / 4),
                (81, 3, 7),
            ])
        };
        let code: Vec<_> = [ipa(1, 0x80), ipa(2, 0x84), EXIT]
            .iter()
            .map(|&raw| decode(raw))
            .collect();
        let mut imap = [Interpolation::Unused; 128];
        imap[0] = Interpolation::Perspective;
        imap[1] = Interpolation::ScreenLinear;
        let layout = Layout {
            stage: Stage::Fragment,
            header: Some(Header {
                stage: Stage::Fragment,
                imap,
                targets: 0,
                writes_depth: false,
                writes_sample_mask: false,
                local_bytes: Some(0),
                outputs: [0; 5],
                outputs_beyond: false,
            }),
            ..Layout::workgroup([1, 1, 1])
        };
        let (translated, registers, _) = translate_with_registers(&code, &layout).unwrap();
        assert_eq!(crate::spirv::tests::validate(&translated.spirv), None);
        // Two scalar inputs at location 0, components 0 and 1, the second
        // NoPerspective.
        let machine = Machine::new(&translated.spirv);
        assert_eq!(machine.locations.values().filter(|&&l| l == 0).count(), 2);
        let bits = |values: [f32; 4]| values.iter().map(|v| u64::from(v.to_bits())).collect();
        let report = machine.simulate(&Inputs {
            workgroup_count: [1, 1, 1],
            builtins: HashMap::from([(15, bits([0.5, 0.5, 0.5, 0.25]))]),
            locations: HashMap::from([(0, bits([2.0, 3.0, 0.0, 0.0]))]),
            ..Inputs::default()
        });
        assert!(report.problems.is_empty(), "{:?}", report.problems);
        let read = |n: u8| f32::from_bits(report.invocations[0][&registers[&n]][0] as u32);
        // x: 2.0 * 1/w; y: as delivered.
        assert_eq!((read(1), read(2)), (0.5, 3.0));
    }

    #[test]
    fn a_mesh_program_draws_the_primitives_its_map_counts() {
        use crate::compute::MeshLayout;
        use echora_shader_abi::Stage;
        // ISBEWR.O [Ra + offset], Rdata, with a size and region.
        let isbewr = |a: u128, data: u128, offset: u128, size: u128, region: u128| {
            word(&[
                (0, 12, 0x927),
                (24, 8, a),
                (32, 8, data),
                (40, 16, offset),
                (74, 2, size),
                (76, 2, region),
                (79, 1, 1),
            ])
        };
        // R0 = lane; R1 = lane * 4 (the vertex's word in each line);
        // position (lane, 2, 0, 1); lane 0 writes the count as a byte at
        // map byte 3 and the three indices as one word at byte 4.
        let s2r = word(&[(0, 12, 0x919), (16, 8, 0), (72, 8, 0x21)]);
        let times_four = word(&[
            (0, 12, 0x824),
            (16, 8, 1),
            (24, 8, 0),
            (32, 32, 4),
            (64, 8, 255),
        ]);
        let to_float = word(&[
            (0, 12, 0x306),
            (16, 8, 2),
            (32, 8, 0),
            (75, 2, 2),
            (84, 2, 2),
        ]);
        let first_lane = word(&[
            (0, 12, 0x20c),
            (24, 8, 0),
            (32, 8, 255),
            (76, 3, 2),
            (81, 3, 1),
            (84, 3, 7),
            (87, 3, 7),
        ]);
        let only_first = |raw: u128| raw & !(0xf << 12) | 1 << 12;
        let program = [
            s2r,
            times_four,
            to_float,
            mov(3, 2.0f32.to_bits()),
            mov(4, 0),
            mov(5, 1.0f32.to_bits()),
            isbewr(1, 2, 0x000, 2, 3),
            isbewr(1, 3, 0x080, 2, 3),
            isbewr(1, 4, 0x100, 2, 3),
            isbewr(1, 5, 0x180, 2, 3),
            first_lane,
            mov(6, 1),
            mov(7, 0x0002_0100),
            only_first(isbewr(255, 6, 3, 0, 0)),
            only_first(isbewr(255, 7, 4, 2, 0)),
            EXIT,
        ];
        let code: Vec<_> = program.iter().map(|&raw| decode(raw)).collect();
        let layout = Layout {
            stage: Stage::Mesh,
            mesh: Some(MeshLayout {
                task: false,
                has_task: false,
                threads: 32,
                max_vertices: 3,
                max_primitives: 1,
                vertices_per_primitive: 3,
                payload_words: 32,
                local_words: 0,
                // Position (0x70-0x7c: bits 28-31).
                vertex_outputs: [0xf000_0000, 0, 0, 0, 0],
                primitive_outputs: None,
            }),
            ..Layout::workgroup([32, 1, 1])
        };
        let translated = crate::compute::translate(&code, &layout).unwrap();
        assert_eq!(crate::spirv::tests::validate(&translated.spirv), None);
        let machine = Machine::new(&translated.spirv);
        // The position is Invariant (OpDecorate %position Invariant).
        let position = machine
            .builtins
            .iter()
            .find(|&(_, &builtin)| builtin == 0)
            .map(|(&variable, _)| variable)
            .unwrap();
        let words = &translated.spirv;
        assert!(words.windows(3).any(|w| {
            w[0] == (3 << 16 | u32::from(op::DECORATE)) && w[1] == position && w[2] == 18
        }));
        let report = machine.simulate(&Inputs {
            workgroup_count: [1, 1, 1],
            ..Inputs::default()
        });
        assert!(report.problems.is_empty(), "{:?}", report.problems);
        // One triangle of three vertices, indices 0, 1, 2.
        assert_eq!(report.mesh_outputs, Some((3, 1)));
        let output = |builtin: u32| {
            let (&variable, _) = machine
                .builtins
                .iter()
                .find(|&(&variable, &b)| {
                    b == builtin && report.workgroup_outputs.contains_key(&variable)
                })
                .unwrap();
            report.workgroup_outputs[&variable].clone()
        };
        assert_eq!(output(5296), vec![0, 1, 2]);
        let positions: Vec<f32> = output(0)
            .iter()
            .map(|&w| f32::from_bits(w as u32))
            .collect();
        assert_eq!(
            positions,
            vec![0.0, 2.0, 0.0, 1.0, 1.0, 2.0, 0.0, 1.0, 2.0, 2.0, 0.0, 1.0]
        );
    }

    #[test]
    fn shifts_permutes_and_bit_counts_match_nak_s_hardware_checked_folds() {
        // NAK's `fold` for each op (ir.rs), which its hw_tests check
        // against a GPU, written out for SM70+.
        let list = values(40);
        let mut state = 0x1234_5678u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state
        };
        for _ in 0..24 {
            let (low, high) = (next(), next());
            let shift = next() % 128;
            // SHF: the 32 forms of (type, right, wrap, high word).
            for form in 0..32u32 {
                let kind = form & 3; // 0 I64, 1 U64, 2 I32, 3 U32
                let (right, wrap, high_word) = (form & 4 != 0, form & 8 != 0, form & 16 != 0);
                let shf = word(&[
                    (0, 12, 0x219),
                    (16, 8, 3),
                    (24, 8, 0),
                    (32, 8, 2),
                    (64, 8, 1),
                    (73, 2, u128::from(kind)),
                    (75, 1, u128::from(wrap)),
                    (76, 1, u128::from(right)),
                    (80, 1, u128::from(high_word)),
                ]);
                let bits = if kind < 2 { 64 } else { 32 };
                let amount = if wrap {
                    shift & (bits - 1)
                } else {
                    shift.min(bits)
                };
                let x = u64::from(low) | u64::from(high) << 32;
                let signed = kind == 0 || kind == 2;
                let shifted = if right {
                    if signed {
                        (x as i64).checked_shr(amount).unwrap_or((x as i64) >> 63) as u64
                    } else {
                        x.checked_shr(amount).unwrap_or(0)
                    }
                } else {
                    x.checked_shl(amount).unwrap_or(0)
                };
                let expected = if high_word {
                    (shifted >> 32) as u32
                } else {
                    shifted as u32
                };
                let (r, _) = run(&[low, high, shift], &[shf]);
                assert_eq!(
                    r[&3], expected,
                    "SHF kind {kind} right {right} wrap {wrap} high {high_word}: \
                     {low:#x} {high:#x} by {shift}"
                );
            }
            // PRMT (index mode): byte n from nibble n of the selector (a's
            // bytes 0-3, c's 4-7; bit 3 replicates the byte's sign).
            let selector = next();
            let prmt = word(&[
                (0, 12, 0x216),
                (16, 8, 3),
                (24, 8, 0),
                (32, 8, 2),
                (64, 8, 1),
            ]);
            let source = u64::from(low) | u64::from(high) << 32;
            let expected = (0..4).fold(0u32, |out, n| {
                let nibble = selector >> (4 * n) & 0xf;
                let byte = (source >> (8 * (nibble & 7))) as u8;
                let byte = if nibble & 8 != 0 {
                    if byte & 0x80 != 0 { 0xff } else { 0 }
                } else {
                    byte
                };
                out | u32::from(byte) << (8 * n)
            });
            let (r, _) = run(&[low, high, selector], &[prmt]);
            assert_eq!(r[&3], expected, "PRMT {low:#x} {high:#x} {selector:#x}");
        }
        for &x in &list {
            // FLO(.S32)(.SH) R3, (~)R0; POPC R3, (~)R0.
            for (signed, shift_amount, invert) in
                (0..8).map(|n| (n & 1 != 0, n & 2 != 0, n & 4 != 0))
            {
                let flo = word(&[
                    (0, 12, 0x300),
                    (16, 8, 3),
                    (32, 8, 0),
                    (63, 1, u128::from(invert)),
                    (73, 1, u128::from(signed)),
                    (74, 1, u128::from(shift_amount)),
                ]);
                let src = if invert { !x } else { x };
                let leading = if signed && src & 0x8000_0000 != 0 {
                    (!src).leading_zeros()
                } else {
                    src.leading_zeros()
                };
                // No bit found: all ones in both forms. NAK's fold gives 32
                // with .SH, but its lowering of find_lsb (FLO.SH of the
                // reversed value) relies on all ones, which the Vulkan CTS
                // checks; its hardware test never feeds 0 or -1.
                let expected = if leading == 32 {
                    u32::MAX
                } else if shift_amount {
                    leading
                } else {
                    31 - leading
                };
                let (r, _) = run(&[x], &[flo]);
                assert_eq!(
                    r[&3], expected,
                    "FLO {signed} {shift_amount} {invert} {x:#x}"
                );
                let popc = word(&[
                    (0, 12, 0x309),
                    (16, 8, 3),
                    (32, 8, 0),
                    (63, 1, u128::from(invert)),
                ]);
                let (r, _) = run(&[x], &[popc]);
                assert_eq!(r[&3], src.count_ones(), "POPC {invert} {x:#x}");
            }
        }
        for window in list.windows(2) {
            let (a, b) = (window[0], window[1]);
            // IMNMX(.U32) R3, R0, R1, (!)PT: min where PT, max where !PT.
            for (signed, min) in [(false, true), (false, false), (true, true), (true, false)] {
                let imnmx = word(&[
                    (0, 12, 0x217),
                    (16, 8, 3),
                    (24, 8, 0),
                    (32, 8, 1),
                    (73, 1, u128::from(signed)),
                    (87, 4, if min { 7 } else { 15 }),
                ]);
                let expected = match (signed, min) {
                    (false, true) => a.min(b),
                    (false, false) => a.max(b),
                    (true, true) => (a as i32).min(b as i32) as u32,
                    (true, false) => (a as i32).max(b as i32) as u32,
                };
                let (r, _) = run(&[a, b], &[imnmx]);
                assert_eq!(
                    r[&3], expected,
                    "IMNMX signed {signed} min {min} {a:#x} {b:#x}"
                );
            }
            // SGXT(.U32) R3, R0, bits (0-40).
            for signed in [false, true] {
                let bits = b % 41;
                let sgxt = word(&[
                    (0, 12, 0x21a),
                    (16, 8, 3),
                    (24, 8, 0),
                    (32, 8, 1),
                    (73, 1, u128::from(signed)),
                ]);
                let expected = if bits >= 32 {
                    a
                } else if bits == 0 {
                    0
                } else {
                    let shift = 32 - bits;
                    if signed {
                        (((a << shift) as i32) >> shift) as u32
                    } else {
                        (a << shift) >> shift
                    }
                };
                let (r, _) = run(&[a, bits], &[sgxt]);
                assert_eq!(r[&3], expected, "SGXT signed {signed} {a:#x} {bits}");
            }
        }
    }

    #[test]
    fn float_compares_follow_all_sixteen_codes_with_nan() {
        let specials = [
            0.0f32,
            -0.0,
            1.0,
            -2.5,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::MIN_POSITIVE,
        ];
        for &a in &specials {
            for &b in &specials {
                for code in 0..16u32 {
                    // FSETP.code.AND P0, PT, R0, R1, PT.
                    let fsetp = word(&[
                        (0, 12, 0x20b),
                        (24, 8, 0),
                        (32, 8, 1),
                        (76, 4, u128::from(code)),
                        (81, 3, 0),
                        (84, 3, 7),
                        (87, 3, 7),
                    ]);
                    let unordered = a.is_nan() || b.is_nan();
                    let relation = match code & 7 {
                        0 => false,
                        1 => a < b,
                        2 => a == b,
                        3 => a <= b,
                        4 => a > b,
                        5 => a != b && !unordered,
                        6 => a >= b,
                        _ => !unordered,
                    };
                    // Codes 8-15: the unordered forms (NAN, LTU .. GEU, T).
                    let expected = match code {
                        0..=7 => relation,
                        8 => unordered,
                        15 => true,
                        _ => unordered || relation,
                    };
                    let (_, p) = run(&[a.to_bits(), b.to_bits()], &[fsetp]);
                    assert_eq!(p[&0], expected, "FSETP code {code}: {a} vs {b}");
                }
            }
        }
    }

    #[test]
    fn forms_without_a_translation_are_refused() {
        use crate::decode::Op;
        use crate::header::{Header, Interpolation};
        use echora_shader_abi::Stage;
        let unknown = |raw: u128| matches!(decode(raw).op, Op::Unknown { .. });
        // IMNMX: the 64-bit form (bit 74).
        let imnmx = word(&[
            (0, 12, 0x217),
            (16, 8, 3),
            (24, 8, 0),
            (32, 8, 1),
            (87, 3, 7),
        ]);
        assert!(!unknown(imnmx));
        assert!(unknown(imnmx | 1 << 74));
        // IPA: a predicate output.
        let ipa = |frequency: u128| {
            word(&[
                (0, 12, 0x326),
                (16, 8, 1),
                (24, 8, 255),
                (32, 8, 255),
                (64, 8, 0x80 / 4),
                (78, 2, frequency),
                (81, 3, 7),
            ])
        };
        assert!(!unknown(ipa(0)));
        assert!(unknown(ipa(0) & !(7 << 81)));
        // IPA.CONSTANT: only of a flat input; IPA.STATE not at all.
        let layout = |mode| {
            let mut imap = [Interpolation::Unused; 128];
            imap[0] = mode;
            Layout {
                stage: Stage::Fragment,
                header: Some(Header {
                    stage: Stage::Fragment,
                    imap,
                    targets: 0,
                    writes_depth: false,
                    writes_sample_mask: false,
                    local_bytes: Some(0),
                    outputs: [0; 5],
                    outputs_beyond: false,
                }),
                ..Layout::workgroup([1, 1, 1])
            }
        };
        let translate = |frequency, mode| {
            let code: Vec<_> = [ipa(frequency), EXIT]
                .iter()
                .map(|&raw| decode(raw))
                .collect();
            crate::compute::translate(&code, &layout(mode))
        };
        assert!(translate(1, Interpolation::Constant).is_ok());
        assert!(translate(1, Interpolation::Perspective).is_err());
        assert!(translate(2, Interpolation::Constant).is_err());
        assert!(translate(0, Interpolation::Perspective).is_ok());
    }

    #[test]
    fn integer_and_float_conversions_round_and_saturate_by_their_modes() {
        // An exact value rounded to f32 by mode (0 nearest even, 1 down, 2
        // up, 3 toward zero).
        let round = |x: f64, mode: u32| -> f32 {
            let nearest = x as f32;
            let step = |v: f32, up: bool| {
                if up {
                    if v == 0.0 {
                        f32::from_bits(1)
                    } else if v > 0.0 {
                        f32::from_bits(v.to_bits() + 1)
                    } else {
                        f32::from_bits(v.to_bits() - 1)
                    }
                } else if v == 0.0 {
                    -f32::from_bits(1)
                } else if v > 0.0 {
                    f32::from_bits(v.to_bits() - 1)
                } else {
                    f32::from_bits(v.to_bits() + 1)
                }
            };
            let n = f64::from(nearest);
            match mode {
                1 if n > x => step(nearest, false),
                2 if n < x => step(nearest, true),
                3 if n.abs() > x.abs() => step(nearest, x < 0.0),
                _ => nearest,
            }
        };
        let list = values(48);
        for &value in &list {
            // I2F(.S/.U)(.size).rounding R1, R0 to F32.
            for (signed, size) in [
                (false, 2u32),
                (true, 2),
                (false, 1),
                (true, 1),
                (false, 0),
                (true, 0),
            ] {
                for mode in 0..4u32 {
                    let i2f = word(&[
                        (0, 12, 0x306),
                        (16, 8, 1),
                        (32, 8, 0),
                        (74, 1, u128::from(signed)),
                        (75, 2, 2),
                        (78, 2, u128::from(mode)),
                        (84, 2, u128::from(size)),
                    ]);
                    let bits = 8 << size;
                    let raw = if bits == 32 {
                        value
                    } else {
                        value & ((1 << bits) - 1)
                    };
                    let exact = if signed {
                        f64::from(((raw << (32 - bits)) as i32) >> (32 - bits))
                    } else {
                        f64::from(raw)
                    };
                    let (r, _) = run(&[value], &[i2f]);
                    assert_eq!(
                        r[&1],
                        round(exact, mode).to_bits(),
                        "I2F signed {signed} size {size} mode {mode}: {value:#x}"
                    );
                }
            }
        }
        // F2I(.S32/.U32).rounding R1, R0 from F32: rounded, saturated, NaN 0.
        let floats = [
            0.5f32,
            1.5,
            2.5,
            -0.5,
            -1.5,
            -2.7,
            3.9,
            1.0e10,
            -1.0e10,
            4_294_967_040.0,
            -2_147_483_648.0,
            2_147_483_520.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            -0.0,
            123_456.78,
        ];
        for &x in &floats {
            for signed in [false, true] {
                for mode in 0..4u32 {
                    let f2i = word(&[
                        (0, 12, 0x305),
                        (16, 8, 1),
                        (32, 8, 0),
                        (72, 1, u128::from(signed)),
                        (75, 2, 2),
                        (78, 2, u128::from(mode)),
                        (84, 2, 2),
                    ]);
                    let rounded = match mode {
                        0 => x.round_ties_even(),
                        1 => x.floor(),
                        2 => x.ceil(),
                        _ => x.trunc(),
                    };
                    // Rust's `as` saturates and sends NaN to 0.
                    let expected = if signed {
                        rounded as i32 as u32
                    } else {
                        rounded as u32
                    };
                    let (r, _) = run(&[x.to_bits()], &[f2i]);
                    assert_eq!(r[&1], expected, "F2I signed {signed} mode {mode}: {x}");
                }
            }
        }
    }

    #[test]
    fn vertex_pulling_decodes_every_format_as_the_abi_defines() {
        use echora_shader_abi::{
            PAGE_SHIFT, PUSH_PAGE_TABLE_OFFSET, PUSH_VERTEX_FETCH_OFFSET, Stage, VertexAttribute,
            VertexFormat, VertexLayout as L, VertexNumeric as N,
        };
        // ALD R0..R3, a[0x80]: location 0's four words.
        let ald = word(&[
            (0, 12, 0x321),
            (16, 8, 0),
            (24, 8, 255),
            (32, 8, 255),
            (40, 10, 0x80),
            (74, 2, 3),
        ]);
        let code: Vec<_> = [ald, EXIT].iter().map(|&raw| decode(raw)).collect();
        let layout = Layout {
            stage: Stage::Vertex,
            ..Layout::workgroup([1, 1, 1])
        };
        let (translated, registers, _) = translate_with_registers(&code, &layout).unwrap();
        let machine = Machine::new(&translated.spirv);
        // Guest page 0x10 (vertex data) at device 0x3000_0000; the page
        // table at 0x2000_0000; the attribute table at 0x4000_0000.
        let (page_table, page, table) = (0x2000_0000u64, 0x3000_0000u64, 0x4000_0000u64);
        let guest = 0x10u64 << PAGE_SHIFT;
        let data: Vec<u8> = (0..4096u32)
            .map(|n| (n.wrapping_mul(0x9e) ^ n >> 3) as u8)
            .collect();
        let stride = 24u32;
        let vertex = 5u32;
        // The CPU decode of the ABI's rules.
        let element = &data[(vertex * stride) as usize..];
        let word_at = |at: usize| u32::from_le_bytes(element[at..at + 4].try_into().unwrap());
        let half = |bits: u32| f16_to_f32(bits as u16);
        let small = |raw: u32, mantissa: u32| -> f32 {
            let (e, m) = (raw >> mantissa, raw & ((1 << mantissa) - 1));
            match e {
                0 => m as f32 * 2f32.powi(-14 - mantissa as i32),
                31 => f32::from_bits(0x7f80_0000 | m << (23 - mantissa)),
                _ => f32::from_bits((e + 112) << 23 | m << (23 - mantissa)),
            }
        };
        let expected = |format: VertexFormat| -> [u32; 4] {
            let integer = format.integer();
            let mut out = [0, 0, 0, if integer { 1 } else { 1.0f32.to_bits() }];
            for c in 0..format.components as usize {
                let (raw, bits) = match format.layout {
                    L::Bits8 => (u32::from(element[c]), 8),
                    L::Bits16 => (
                        u32::from(u16::from_le_bytes([element[2 * c], element[2 * c + 1]])),
                        16,
                    ),
                    L::Bits32 => (word_at(4 * c), 32),
                    L::Packed1010102 => {
                        let (shift, bits) = [(0, 10), (10, 10), (20, 10), (30, 2)][c];
                        (word_at(0) >> shift & ((1u64 << bits) - 1) as u32, bits)
                    }
                    L::Packed111110 => {
                        let (shift, bits) = [(0, 11), (11, 11), (22, 10), (0, 0)][c];
                        (word_at(0) >> shift & ((1u32 << bits) - 1), bits)
                    }
                };
                let signed = if bits == 32 {
                    raw as i32
                } else {
                    ((raw << (32 - bits)) as i32) >> (32 - bits)
                };
                let largest = |b: u32| ((1u64 << b) - 1) as f32;
                out[c] = match (format.layout, format.numeric) {
                    (L::Packed111110, _) => small(raw, bits - 5).to_bits(),
                    (_, N::Unorm) => (raw as f32 / largest(bits)).to_bits(),
                    (_, N::Snorm) => (signed as f32 / largest(bits - 1)).max(-1.0).to_bits(),
                    (_, N::Uint) => raw,
                    (_, N::Sint) => signed as u32,
                    (_, N::Uscaled) => (raw as f32).to_bits(),
                    (_, N::Sscaled) => (signed as f32).to_bits(),
                    (L::Bits16, N::Float) => half(raw).to_bits(),
                    (_, N::Float) => raw,
                };
            }
            if format.bgra {
                out.swap(0, 2);
            }
            out
        };
        let formats = [
            (L::Bits8, N::Unorm, 4, false),
            (L::Bits8, N::Unorm, 4, true),
            (L::Bits8, N::Snorm, 3, false),
            (L::Bits8, N::Uint, 2, false),
            (L::Bits8, N::Sint, 4, false),
            (L::Bits8, N::Uscaled, 2, false),
            (L::Bits8, N::Sscaled, 3, false),
            (L::Bits16, N::Unorm, 2, false),
            (L::Bits16, N::Snorm, 4, false),
            (L::Bits16, N::Float, 4, false),
            (L::Bits16, N::Float, 3, false),
            (L::Bits16, N::Sint, 2, false),
            (L::Bits16, N::Uscaled, 4, false),
            (L::Bits32, N::Float, 3, false),
            (L::Bits32, N::Float, 1, false),
            (L::Bits32, N::Uint, 4, false),
            (L::Bits32, N::Sint, 2, false),
            (L::Packed1010102, N::Unorm, 4, false),
            (L::Packed1010102, N::Snorm, 4, false),
            (L::Packed1010102, N::Uint, 4, true),
            (L::Packed111110, N::Float, 3, false),
        ];
        for (layout, numeric, components, bgra) in formats {
            let format = VertexFormat {
                components,
                layout,
                numeric,
                bgra,
            };
            let attribute = VertexAttribute {
                address: guest,
                stride,
                divisor: 0,
                format: format.encode(),
                size: 4096,
                reserved: [0; 2],
            };
            let mut push = vec![0u8; 128];
            let at = PUSH_VERTEX_FETCH_OFFSET as usize;
            push[at..at + 8].copy_from_slice(&table.to_le_bytes());
            push[at + 12..at + 16].copy_from_slice(&1u32.to_le_bytes());
            let at = PUSH_PAGE_TABLE_OFFSET as usize;
            push[at..at + 8].copy_from_slice(&page_table.to_le_bytes());
            push[at + 8..at + 12].copy_from_slice(&0x10u32.to_le_bytes());
            push[at + 12..at + 16].copy_from_slice(&1u32.to_le_bytes());
            let report = machine.simulate(&Inputs {
                workgroup_count: [1, 1, 1],
                push,
                builtins: HashMap::from([(42, vec![u64::from(vertex)]), (43, vec![0])]),
                memory: BTreeMap::from([
                    (page_table, page.to_le_bytes().to_vec()),
                    (page, data.clone()),
                    (table, attribute.to_bytes().to_vec()),
                ]),
                ..Inputs::default()
            });
            assert!(
                report.problems.is_empty(),
                "{format:?}: {:?}",
                report.problems
            );
            let got: Vec<u32> = (0..4u8)
                .map(|n| report.invocations[0][&registers[&n]][0] as u32)
                .collect();
            assert_eq!(got, expected(format), "{format:?}");
        }
        // Per instance: base 2, InstanceIndex 7, divisor 3: element 3 (a
        // 32-bit uint, the first word of element 3).
        let format = VertexFormat {
            components: 1,
            layout: L::Bits32,
            numeric: N::Uint,
            bgra: false,
        };
        let attribute = VertexAttribute {
            address: guest,
            stride,
            divisor: 3,
            format: format.encode(),
            size: 4096,
            reserved: [0; 2],
        };
        let mut push = vec![0u8; 128];
        let at = PUSH_VERTEX_FETCH_OFFSET as usize;
        push[at..at + 8].copy_from_slice(&table.to_le_bytes());
        push[at + 8..at + 12].copy_from_slice(&2u32.to_le_bytes());
        push[at + 12..at + 16].copy_from_slice(&1u32.to_le_bytes());
        let at = PUSH_PAGE_TABLE_OFFSET as usize;
        push[at..at + 8].copy_from_slice(&page_table.to_le_bytes());
        push[at + 8..at + 12].copy_from_slice(&0x10u32.to_le_bytes());
        push[at + 12..at + 16].copy_from_slice(&1u32.to_le_bytes());
        let report = machine.simulate(&Inputs {
            workgroup_count: [1, 1, 1],
            push,
            builtins: HashMap::from([(42, vec![9]), (43, vec![7])]),
            memory: BTreeMap::from([
                (page_table, page.to_le_bytes().to_vec()),
                (page, data.clone()),
                (table, attribute.to_bytes().to_vec()),
            ]),
            ..Inputs::default()
        });
        let at = (3 * stride) as usize;
        let expected = u32::from_le_bytes(data[at..at + 4].try_into().unwrap());
        assert_eq!(report.invocations[0][&registers[&0]][0] as u32, expected);
    }

    #[test]
    fn vertex_ids_include_the_base_vertex_and_instance_ids_exclude_the_base_instance() {
        use echora_shader_abi::Stage;
        // ALD R0, a[0x2fc] (vertex id); ALD R1, a[0x2f8] (instance id).
        let ald = |d: u128, address: u128| {
            word(&[
                (0, 12, 0x321),
                (16, 8, d),
                (24, 8, 255),
                (32, 8, 255),
                (40, 10, address),
            ])
        };
        let code: Vec<_> = [ald(0, 0x2fc), ald(1, 0x2f8), EXIT]
            .iter()
            .map(|&raw| decode(raw))
            .collect();
        let layout = Layout {
            stage: Stage::Vertex,
            ..Layout::workgroup([1, 1, 1])
        };
        let (translated, registers, _) = translate_with_registers(&code, &layout).unwrap();
        assert_eq!(crate::spirv::tests::validate(&translated.spirv), None);
        // Index 10 of an indexed draw with base vertex 5494 (Vulkan's
        // VertexIndex 5504), the first instance of base instance 3.
        let report = Machine::new(&translated.spirv).simulate(&Inputs {
            workgroup_count: [1, 1, 1],
            builtins: HashMap::from([
                (42, vec![5504]),
                (4424, vec![5494]),
                (43, vec![3]),
                (4425, vec![3]),
            ]),
            ..Inputs::default()
        });
        assert!(report.problems.is_empty(), "{:?}", report.problems);
        let read = |n: u8| report.invocations[0][&registers[&n]][0] as u32;
        assert_eq!((read(0), read(1)), (5504, 0));
    }

    #[test]
    fn gathers_pick_their_component_and_offset() {
        // TLD4.c (2D, handle c[0x0][0x40]) R4, R0 (x, then y, or x and y
        // then the packed offsets in R2): component c of the four texels
        // around the point, in Vulkan's order (i0,j1) (i1,j1) (i1,j0)
        // (i0,j0).
        let gather = |component: u128, offsets: bool| {
            word(&[
                (0, 12, 0xb63),
                (16, 8, 4),
                (24, 8, 0),
                (32, 8, if offsets { 2 } else { 1 }),
                (40, 14, 0x10),
                (61, 3, 1),
                (64, 8, 255),
                (72, 4, 0xf),
                (76, 1, u128::from(offsets)),
                (81, 3, 7),
                (87, 2, component),
            ])
        };
        // Texel (x, y), component c: 100c + 10y + x.
        let texels: Vec<[u32; 4]> = (0..16u32)
            .map(|n| [0, 1, 2, 3].map(|c| 100 * c + 10 * (n / 4) + n % 4))
            .collect();
        // (2.0, 1.0) in texels: i0 = 1, j0 = 0.
        let (x, y) = (0.5f32, 0.25f32);
        for (component, offsets, (dx, dy)) in [
            (1u32, false, (0i64, 0i64)),
            (2, false, (0, 0)),
            (1, true, (1, -1)),
        ] {
            let packed = (dx as u32 & 0xff) | (dy as u32 & 0xff) << 8;
            let program = [
                mov(0, x.to_bits()),
                mov(1, y.to_bits()),
                mov(2, packed),
                gather(u128::from(component), offsets),
                EXIT,
            ];
            let code: Vec<_> = program.iter().map(|&raw| decode(raw)).collect();
            let (translated, registers, _) =
                translate_with_registers(&code, &Layout::workgroup([1, 1, 1])).unwrap();
            assert_eq!(crate::spirv::tests::validate(&translated.spirv), None);
            let texture = translated
                .descriptors
                .iter()
                .find(|d| d.cbuf_offset == 0x40)
                .unwrap();
            let report = Machine::new(&translated.spirv).simulate(&Inputs {
                workgroup_count: [1, 1, 1],
                images: HashMap::from([(
                    (0, texture.binding),
                    Image {
                        width: 4,
                        height: 4,
                        depth: 1,
                        layers: 1,
                        texels: texels.clone(),
                    },
                )]),
                ..Inputs::default()
            });
            assert!(report.problems.is_empty(), "{:?}", report.problems);
            let (i0, j0) = (1 + dx, dy);
            let value = |i: i64, j: i64| {
                let (i, j) = (i.clamp(0, 3) as u32, j.clamp(0, 3) as u32);
                100 * component + 10 * j + i
            };
            let expected = [
                value(i0, j0 + 1),
                value(i0 + 1, j0 + 1),
                value(i0 + 1, j0),
                value(i0, j0),
            ];
            let got: Vec<u32> = (4..8u8)
                .map(|n| report.invocations[0][&registers[&n]][0] as u32)
                .collect();
            assert_eq!(got, expected, "component {component} offsets ({dx}, {dy})");
        }
    }

    #[test]
    fn explicit_gradient_samples_read_their_coordinates() {
        // TXD (2D, handle c[0x0][0x40]) R4, R0 (x, y), R2 (dx/dx, dx/dy,
        // dy/dx, dy/dy): the simulator samples level 0 nearest, so the
        // texel at the coordinates whatever the gradients.
        let txd = word(&[
            (0, 12, 0xb6c),
            (16, 8, 4),
            (24, 8, 0),
            (32, 8, 2),
            (40, 14, 0x10),
            (61, 3, 1),
            (64, 8, 255),
            (72, 4, 0x3),
            (81, 3, 7),
        ]);
        let texels: Vec<[u32; 4]> = (0..16u32)
            .map(|n| [n, n + 100, 0, 0].map(|v| (v as f32).to_bits()))
            .collect();
        let gradient = (1.0f32 / 4.0).to_bits();
        let program = [
            mov(0, 0.875f32.to_bits()),
            mov(1, 0.125f32.to_bits()),
            mov(2, gradient),
            mov(3, 0),
            mov(5, 0),
            mov(6, gradient),
            txd,
            EXIT,
        ];
        let code: Vec<_> = program.iter().map(|&raw| decode(raw)).collect();
        let (translated, registers, _) =
            translate_with_registers(&code, &Layout::workgroup([1, 1, 1])).unwrap();
        assert_eq!(crate::spirv::tests::validate(&translated.spirv), None);
        let texture = translated
            .descriptors
            .iter()
            .find(|d| d.cbuf_offset == 0x40)
            .unwrap();
        let report = Machine::new(&translated.spirv).simulate(&Inputs {
            workgroup_count: [1, 1, 1],
            images: HashMap::from([(
                (0, texture.binding),
                Image {
                    width: 4,
                    height: 4,
                    depth: 1,
                    layers: 1,
                    texels,
                },
            )]),
            ..Inputs::default()
        });
        assert!(report.problems.is_empty(), "{:?}", report.problems);
        // Texel (3, 0): 3.0 and 103.0 in components 0 and 1 (mask 0x3).
        let read = |n: u8| f32::from_bits(report.invocations[0][&registers[&n]][0] as u32);
        assert_eq!((read(4), read(5)), (3.0, 103.0));
    }

    #[test]
    fn depth_compares_test_the_reference_by_the_sampler_s_op() {
        // Texel (x, y) of a 4 x 4 depth texture: 0.1 (x + y).
        let depth = |x: u32, y: u32| 0.1 * (x + y) as f32;
        let texels: Vec<[u32; 4]> = (0..16u32)
            .map(|n| [depth(n % 4, n / 4).to_bits(), 0, 0, 0])
            .collect();
        // TEX.LZ.DC R4, R0 (x, y), R2 (reference); TLD4.DC R4, R0, R2.
        let sample = |gather: bool| {
            word(&[
                (0, 12, if gather { 0xb63 } else { 0xb60 }),
                (16, 8, 4),
                (24, 8, 0),
                (32, 8, 2),
                (40, 14, 0x10),
                (61, 3, 1),
                (64, 8, 255),
                (72, 4, if gather { 0xf } else { 0x1 }),
                (78, 1, 1),
                (81, 3, 7),
                (87, 3, if gather { 0 } else { 1 }),
            ])
        };
        let reference = 0.35f32;
        // (2.0, 1.0) in texels: nearest (2, 1) = 0.3; the gather's (1,1)
        // (2,1) (2,0) (1,0) = 0.2 0.3 0.2 0.1.
        let (x, y) = (0.625f32, 0.375f32);
        for compare in [3u32, 4] {
            let holds = |d: f32| {
                let holds = if compare == 3 {
                    reference <= d
                } else {
                    reference > d
                };
                if holds { 1.0f32 } else { 0.0 }
            };
            for gather in [false, true] {
                let (x, y) = if gather { (0.5f32, 0.25f32) } else { (x, y) };
                let program = [
                    mov(0, x.to_bits()),
                    mov(1, y.to_bits()),
                    mov(2, reference.to_bits()),
                    sample(gather),
                    EXIT,
                ];
                let code: Vec<_> = program.iter().map(|&raw| decode(raw)).collect();
                let (translated, registers, _) =
                    translate_with_registers(&code, &Layout::workgroup([1, 1, 1])).unwrap();
                assert_eq!(crate::spirv::tests::validate(&translated.spirv), None);
                let texture = translated
                    .descriptors
                    .iter()
                    .find(|d| d.cbuf_offset == 0x40)
                    .unwrap();
                let report = Machine::new(&translated.spirv).simulate(&Inputs {
                    workgroup_count: [1, 1, 1],
                    images: HashMap::from([(
                        (0, texture.binding),
                        Image {
                            width: 4,
                            height: 4,
                            depth: 1,
                            layers: 1,
                            texels: texels.clone(),
                        },
                    )]),
                    compare_op: Some(compare),
                    ..Inputs::default()
                });
                assert!(report.problems.is_empty(), "{:?}", report.problems);
                let read = |n: u8| f32::from_bits(report.invocations[0][&registers[&n]][0] as u32);
                if gather {
                    let expected = [depth(1, 1), depth(2, 1), depth(2, 0), depth(1, 0)].map(holds);
                    assert_eq!(
                        [read(4), read(5), read(6), read(7)],
                        expected,
                        "gather, op {compare}"
                    );
                } else {
                    assert_eq!(read(4), holds(depth(2, 1)), "sample, op {compare}");
                }
            }
        }
    }

    #[test]
    fn explicit_and_gradient_levels_pick_their_mip() {
        // Level l of a 4 x 4 texture: (4 >> l)^2 texels, texel n holding
        // 1000 l + n.
        let level = |l: u32| {
            let side = 4 >> l;
            Image {
                width: side,
                height: side,
                depth: 1,
                layers: 1,
                texels: (0..side * side).map(|n| [1000 * l + n, 0, 0, 0]).collect(),
            }
        };
        let texture = |raw: u128, registers: &[u32]| {
            let mut program: Vec<u128> = registers
                .iter()
                .enumerate()
                .map(|(n, &value)| mov(n as u128, value))
                .collect();
            program.extend([raw, EXIT]);
            let code: Vec<_> = program.iter().map(|&raw| decode(raw)).collect();
            let (translated, registers, _) =
                translate_with_registers(&code, &Layout::workgroup([1, 1, 1])).unwrap();
            assert_eq!(crate::spirv::tests::validate(&translated.spirv), None);
            let binding = translated
                .descriptors
                .iter()
                .find(|d| d.cbuf_offset == 0x40)
                .unwrap()
                .binding;
            let report = Machine::new(&translated.spirv).simulate(&Inputs {
                workgroup_count: [1, 1, 1],
                images: HashMap::from([((0, binding), level(0))]),
                mips: HashMap::from([((0, binding), vec![level(1), level(2)])]),
                ..Inputs::default()
            });
            assert!(report.problems.is_empty(), "{:?}", report.problems);
            (0..4u8)
                .map(|n| {
                    registers
                        .get(&(4 + n))
                        .map_or(0, |v| report.invocations[0][v][0] as u32)
                })
                .collect::<Vec<u32>>()
        };
        let common = |opcode: u128, mask: u128, lod: u128| {
            word(&[
                (0, 12, opcode),
                (16, 8, 4),
                (24, 8, 0),
                (32, 8, 2),
                (40, 14, 0x10),
                (61, 3, 1),
                (64, 8, 255),
                (72, 4, mask),
                (81, 3, 7),
                (87, 3, lod),
            ])
        };
        let (x, y) = (0.6f32.to_bits(), 0.6f32.to_bits());
        // TEX.LL R4, R0 (x, y), R2 (level): 1.4 is level 1's texel (1, 1),
        // 1.6 level 2's only texel.
        assert_eq!(
            texture(common(0xb60, 1, 3), &[x, y, 1.4f32.to_bits()])[0],
            1003
        );
        assert_eq!(
            texture(common(0xb60, 1, 3), &[x, y, 1.6f32.to_bits()])[0],
            2000
        );
        // TXD R4, R0 (x, y), R2.. (gradients of 0.5: two level-0 texels,
        // level 1).
        let txd = word(&[
            (0, 12, 0xb6c),
            (16, 8, 4),
            (24, 8, 0),
            (32, 8, 2),
            (40, 14, 0x10),
            (61, 3, 1),
            (64, 8, 255),
            (72, 4, 1),
            (81, 3, 7),
        ]);
        let half = 0.5f32.to_bits();
        assert_eq!(texture(txd, &[x, y, half, 0, 0, half])[0], 1003);
        // TLD.LL R4, R0 (x, y integer), R2 (level 1): texel (1, 0) there.
        assert_eq!(texture(common(0xb66, 1, 3), &[1, 0, 1])[0], 1001);
        // TXQ R4, level R0 = 1: 2 x 2, three levels.
        let txq = word(&[
            (0, 12, 0xb6f),
            (16, 8, 4),
            (24, 8, 0),
            (32, 8, 255),
            (40, 14, 0x10),
            (64, 8, 255),
            (72, 4, 0xf),
        ]);
        assert_eq!(texture(txq, &[1]), vec![2, 2, 1, 3]);
    }

    #[test]
    fn half_adds_flush_subnormals_with_ftz() {
        // HADD2(.FTZ) R2, R0, R1: R0 = a subnormal and a normal half, R1 = 0.
        let add = |ftz: u128| {
            word(&[
                (0, 12, 0x230),
                (16, 8, 2),
                (24, 8, 0),
                (32, 8, 1),
                (80, 1, ftz),
            ])
        };
        let input = 0x3c00_8001u32;
        let (registers, _) = run(&[input, 0], &[add(0)]);
        assert_eq!(registers[&2], input);
        // The subnormal flushes to -0, and -0 + +0 is +0.
        let (registers, _) = run(&[input, 0], &[add(1)]);
        assert_eq!(registers[&2], 0x3c00_0000);
    }

    #[test]
    fn the_uniform_words_read_are_reported() {
        // MOV R0, c[0x3][0x10]
        let read = word(&[
            (0, 12, 0xa02),
            (16, 8, 0),
            (38, 16, 0x10),
            (54, 5, 3),
            (72, 4, 0xf),
        ]);
        let code: Vec<_> = [read, EXIT].iter().map(|&raw| decode(raw)).collect();
        let translated = crate::compute::translate(&code, &Layout::workgroup([1, 1, 1])).unwrap();
        let bank = translated.descriptors.iter().find(|d| d.kind == 0).unwrap();
        assert_eq!(bank.index, 3);
        let report = Machine::new(&translated.spirv).simulate(&Inputs {
            workgroup_count: [1, 1, 1],
            ..Inputs::default()
        });
        assert!(report.uniform_reads.contains(&(0, bank.binding, 0x10)));
    }

    #[test]
    fn lane_masks_match_the_hardware_s_numbering() {
        // S2R R1, SR_EQMASK .. R5, SR_GEMASK (0x38-0x3c)
        let masks: Vec<u128> = (0..5u128)
            .map(|n| word(&[(0, 12, 0x919), (16, 8, 1 + n), (72, 8, 0x38 + n)]))
            .collect();
        let (registers, _, problems) = run_group(32, &[], &masks);
        assert!(problems.is_empty(), "{problems:?}");
        let lane = 5;
        let r = &registers[lane];
        assert_eq!(r[&1], 1 << lane, "eq");
        assert_eq!(r[&2], (1 << lane) - 1, "lt");
        assert_eq!(r[&3], (1 << (lane + 1)) - 1, "le");
        assert_eq!(r[&4], !((1u32 << (lane + 1)) - 1), "gt");
        assert_eq!(r[&5], !((1u32 << lane) - 1), "ge");
    }

    #[test]
    fn accesses_to_pages_the_table_does_not_map_are_counted() {
        use echora_shader_abi::{PAGE_SHIFT, PUSH_PAGE_TABLE_OFFSET};
        // MOV R4, 0x10000; MOV R5, 0; LDG.E R2, [R4.64]
        let load = word(&[
            (0, 12, 0x381),
            (16, 8, 2),
            (24, 8, 4),
            (72, 1, 1),
            (73, 3, 4),
        ]);
        let code: Vec<_> = [mov(4, 0x10000), mov(5, 0), load, EXIT]
            .iter()
            .map(|&raw| decode(raw))
            .collect();
        let translated = crate::compute::translate(&code, &Layout::workgroup([1, 1, 1])).unwrap();
        let machine = Machine::new(&translated.spirv);
        let table = 0x2000_0000u64;
        let push = |first: u32, count: u32| {
            let mut push = vec![0u8; 128];
            let at = PUSH_PAGE_TABLE_OFFSET as usize;
            push[at..at + 8].copy_from_slice(&table.to_le_bytes());
            push[at + 8..at + 12].copy_from_slice(&first.to_le_bytes());
            push[at + 12..at + 16].copy_from_slice(&count.to_le_bytes());
            push
        };
        let page = 0x10000u64 >> PAGE_SHIFT;
        // Outside the table (it covers the page after).
        let report = machine.simulate(&Inputs {
            workgroup_count: [1, 1, 1],
            push: push(page as u32 + 1, 1),
            memory: BTreeMap::from([(table, vec![0; 8])]),
            ..Inputs::default()
        });
        assert_eq!(report.unmapped_pages, BTreeMap::from([(page, 1)]));
        // In the table with a zero entry.
        let report = machine.simulate(&Inputs {
            workgroup_count: [1, 1, 1],
            push: push(page as u32, 1),
            memory: BTreeMap::from([(table, vec![0; 8])]),
            ..Inputs::default()
        });
        assert_eq!(report.unmapped_pages, BTreeMap::from([(page, 1)]));
    }
}
