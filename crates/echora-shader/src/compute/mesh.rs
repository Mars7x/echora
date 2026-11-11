//! Task and mesh programs (VK_EXT_mesh_shader).
//!
//! The hardware runs them as a single warp (at most 32 threads) and passes
//! data through the internal stage buffer, which ISBERD and ISBEWR read and
//! write by byte address. The layout is Mesa NAK's
//! (`nak_nir_lower_mesh_intrinsics.c`, `nak_nir_lower_task_intrinsics`):
//!
//! - A task program's output attributes (`.O`, region 3): the mesh
//!   workgroup count at byte 4, its x, y and z at 8, 12 and 16, then the
//!   payload. Here they are the task payload (`TaskPayloadWorkgroupEXT`),
//!   and the program ends with `OpEmitMeshTasksEXT(count, 1, 1)`.
//! - A program's input attributes (region 3, not `.O`): byte 0 its own
//!   workgroup index; a mesh program after a task program reads the
//!   payload at the other bytes, a program without one the draw's
//!   workgroup counts (at 4, the product; at 8, 12 and 16, x, y and z).
//! - A mesh program's output attributes: per-vertex outputs in groups of
//!   32 vertices, one 128-byte line per output word the program's header
//!   lists (in address order), word `v % 32` of the line for vertex `v`;
//!   then the per-primitive outputs, the same way, for the words the
//!   primitive header (a pass-through geometry program's) lists. Here a
//!   workgroup array, sized to the larger of the lines and the local
//!   buffer the pipeline gives (SET_MESH_INIT_SHADER bits 12-20, in
//!   128-byte lines), so other data the program keeps there fits too.
//! - A mesh program's map (region 0, `.O`): the primitive count as a word
//!   written at byte 3, then each primitive's vertex indices as bytes from
//!   byte 4.
//!
//! The translated program runs as a function; the entry point calls it,
//! then (after a workgroup barrier, as every invocation has returned)
//! emits the mesh tasks, or sets the mesh outputs (all `max_vertices`
//! vertices, the written primitive count capped at `max_primitives`) and
//! copies the lines and the indices to the output variables. The
//! `.SKEW` bit is ignored: the addresses the programs compute are taken
//! as linear, as NAK computes them.

use echora_shader_abi::{Stage, Unsupported};

use super::{Emitter, MeshLayout, unsupported};
use crate::spirv::{Function, Type, capability, decoration, execution, glsl, op, storage};

/// Built-ins.
const BUILTIN_POSITION: u32 = 0;
const BUILTIN_POINT_SIZE: u32 = 1;
const BUILTIN_PRIMITIVE_ID: u32 = 7;
const BUILTIN_LAYER: u32 = 9;
const BUILTIN_VIEWPORT_INDEX: u32 = 10;
const BUILTIN_NUM_WORKGROUPS: u32 = 24;
const BUILTIN_WORKGROUP_ID: u32 = 26;
const BUILTIN_LOCAL_INVOCATION_INDEX: u32 = 29;
const BUILTIN_PRIMITIVE_POINT_INDICES: u32 = 5294;
const BUILTIN_PRIMITIVE_LINE_INDICES: u32 = 5295;
const BUILTIN_PRIMITIVE_TRIANGLE_INDICES: u32 = 5296;

/// Decorations.
const DECORATION_LOCATION: u32 = 30;
const DECORATION_PER_PRIMITIVE: u32 = 5271;

/// Elements per group of stage buffer lines.
const GROUP: u32 = 32;

/// A task or mesh program's variables.
#[derive(Debug)]
pub(super) struct MeshState {
    /// The entry point, while the program's own function is emitted.
    main: Option<Function>,
    /// The program's function.
    body: u32,
    /// The task payload (u32 words), when there is one.
    payload: Option<u32>,
    /// Mesh: the stage buffer's output attributes, the map and the
    /// primitive count (workgroup variables).
    attributes: u32,
    attribute_words: u32,
    map: u32,
    map_words: u32,
    count: u32,
}

/// The attribute addresses a header's output map lists, in order.
pub(super) fn lines(outputs: &[u32; 5]) -> Vec<u32> {
    (0..160)
        .filter(|bit| outputs[bit / 32] >> (bit % 32) & 1 != 0)
        .map(|bit| bit as u32 * 4)
        .collect()
}

impl MeshLayout {
    /// Vertices per primitive.
    fn vertices_per_primitive(&self) -> u32 {
        u32::from(self.vertices_per_primitive)
    }

    /// Words of the per-vertex lines.
    fn vertex_words(&self) -> u32 {
        self.max_vertices.div_ceil(GROUP) * lines(&self.vertex_outputs).len() as u32 * GROUP
    }

    fn primitive_words(&self) -> u32 {
        let lines = self
            .primitive_outputs
            .map_or(0, |outputs| lines(&outputs).len());
        self.max_primitives.div_ceil(GROUP) * lines as u32 * GROUP
    }
}

impl Emitter {
    fn mesh_layout(&self) -> MeshLayout {
        self.layout.mesh.expect("a task or mesh layout")
    }

    fn word_array(&mut self, length: u32) -> u32 {
        let length = self.constant(length);
        self.module.ty(Type::Array {
            element: self.u32,
            length,
        })
    }

    fn workgroup_words(&mut self, length: u32) -> u32 {
        let array = self.word_array(length);
        let pointer = self.module.pointer(storage::WORKGROUP, array);
        let variable = self.module.global_variable(pointer, storage::WORKGROUP);
        self.interface.push(variable);
        variable
    }

    /// Sets a task or mesh program up: its execution modes and variables,
    /// and its own function, which the entry point calls.
    pub(super) fn mesh_begin(&mut self) {
        let Some(mesh) = self.layout.mesh else {
            return;
        };
        self.module.capability(capability::MESH_SHADING_EXT);
        self.module.extension("SPV_EXT_mesh_shader");
        let entry = self.entry;
        self.module
            .execution_mode(entry, execution::LOCAL_SIZE, &[mesh.threads, 1, 1]);
        let payload = (mesh.task || mesh.has_task).then(|| {
            let array = self.word_array(mesh.payload_words);
            let pointer = self
                .module
                .pointer(storage::TASK_PAYLOAD_WORKGROUP_EXT, array);
            let variable = self
                .module
                .global_variable(pointer, storage::TASK_PAYLOAD_WORKGROUP_EXT);
            self.interface.push(variable);
            variable
        });
        let (mut attributes, mut attribute_words, mut map, mut map_words, mut count) =
            (0, 0, 0, 0, 0);
        if !mesh.task {
            self.module
                .execution_mode(entry, execution::OUTPUT_VERTICES, &[mesh.max_vertices]);
            self.module.execution_mode(
                entry,
                execution::OUTPUT_PRIMITIVES_EXT,
                &[mesh.max_primitives],
            );
            let topology = match mesh.vertices_per_primitive {
                1 => execution::OUTPUT_POINTS,
                2 => execution::OUTPUT_LINES_EXT,
                _ => execution::OUTPUT_TRIANGLES_EXT,
            };
            self.module.execution_mode(entry, topology, &[]);
            attribute_words = (mesh.vertex_words() + mesh.primitive_words())
                .max(mesh.local_words)
                .max(1);
            attributes = self.workgroup_words(attribute_words);
            map_words = 4 + mesh.vertices_per_primitive() * mesh.max_primitives;
            map = self.workgroup_words(map_words);
            let pointer = self.module.pointer(storage::WORKGROUP, self.u32);
            count = self.module.global_variable(pointer, storage::WORKGROUP);
            self.interface.push(count);
        }
        let function_type = self.module.ty(Type::Function {
            result: self.void,
            parameters: vec![],
        });
        let body = Function::new(&mut self.module, self.void, function_type);
        let id = body.id;
        let main = std::mem::replace(&mut self.function, body);
        self.mesh = Some(MeshState {
            main: Some(main),
            body: id,
            payload,
            attributes,
            attribute_words,
            map,
            map_words,
            count,
        });
    }

    /// Ends a task or mesh program: the program's function is done; the
    /// entry point calls it, then emits the tasks or the mesh outputs.
    /// Returns whether the entry point is already terminated.
    pub(super) fn mesh_end(&mut self) -> Result<(), Unsupported> {
        let Some(state) = self.mesh.as_mut() else {
            return Ok(());
        };
        // VK_EXT_mesh_shader's minimum maxMeshSharedMemorySize, in words.
        let workgroup_words = state.attribute_words + state.map_words + 1;
        if workgroup_words > 7168 {
            return Err(unsupported(
                0,
                format!("a mesh stage buffer of {workgroup_words} words"),
            ));
        }
        let main = state.main.take().expect("the entry point");
        let body_id = state.body;
        let body = std::mem::replace(&mut self.function, main);
        body.finish(&mut self.module);
        self.value(op::FUNCTION_CALL, self.void, &[body_id]);
        self.barrier();
        let mesh = self.mesh_layout();
        if mesh.task {
            let payload = self.mesh.as_ref().and_then(|state| state.payload);
            let payload = payload.expect("a task payload");
            let count = self.word(payload, storage::TASK_PAYLOAD_WORKGROUP_EXT, 1);
            let one = self.constant(1);
            self.function
                .emit(op::EMIT_MESH_TASKS_EXT, &[count, one, one, payload]);
        } else {
            self.vertex_trace_record = None;
            let state = self.mesh.as_ref().expect("mesh variables");
            let (attributes, attribute_words) = (state.attributes, state.attribute_words);
            let vertex_lines = lines(&mesh.vertex_outputs);
            self.vertex_trace_mesh_outputs(
                attributes,
                attribute_words,
                &vertex_lines,
                mesh.max_vertices,
            );
            self.mesh_outputs(&mesh)?;
            self.function.emit(op::RETURN, &[]);
        }
        Ok(())
    }

    /// Word `index` (a constant) of a u32 array variable.
    fn word(&mut self, array: u32, class: u32, index: u32) -> u32 {
        let index = self.constant(index);
        self.word_at(array, class, index)
    }

    fn word_pointer(&mut self, array: u32, class: u32, index: u32) -> u32 {
        let pointer = self.module.pointer(class, self.u32);
        self.value(op::ACCESS_CHAIN, pointer, &[array, index])
    }

    fn word_at(&mut self, array: u32, class: u32, index: u32) -> u32 {
        let pointer = self.word_pointer(array, class, index);
        self.value(op::LOAD, self.u32, &[pointer])
    }

    /// Word `index` (a value) of an array of `length` words, 0 past it.
    pub(super) fn bounded_word(&mut self, array: u32, class: u32, index: u32, length: u32) -> u32 {
        let last = self.constant(length - 1);
        let clamped = self.glsl(self.u32, glsl::U_MIN, &[index, last]);
        let value = self.word_at(array, class, clamped);
        let limit = self.constant(length);
        let inside = self.value(op::U_LESS_THAN, self.bool, &[index, limit]);
        let zero = self.constant(0);
        self.value(op::SELECT, self.u32, &[inside, value, zero])
    }

    /// Stores `value` at word `index` of an array of `length` words; past
    /// it, nothing.
    fn bounded_store(
        &mut self,
        array: u32,
        class: u32,
        index: u32,
        length: u32,
        value: u32,
    ) -> Result<(), Unsupported> {
        let limit = self.constant(length);
        let inside = self.value(op::U_LESS_THAN, self.bool, &[index, limit]);
        self.when(inside, |e| {
            let pointer = e.word_pointer(array, class, index);
            e.function.emit(op::STORE, &[pointer, value]);
            Ok(())
        })
    }

    /// ISBERD and ISBEWR.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn stage_buffer(
        &mut self,
        index: usize,
        write: bool,
        value: u8,
        a: u8,
        offset: u16,
        size: u8,
        region: u8,
        output: bool,
    ) -> Result<(), Unsupported> {
        let Some(mesh) = self.layout.mesh else {
            return Err(unsupported(
                index,
                "stage buffer outside a task or mesh program",
            ));
        };
        let state = self.mesh.as_ref().expect("task or mesh variables");
        let (payload, attributes, attribute_words, map, map_words, count) = (
            state.payload,
            state.attributes,
            state.attribute_words,
            state.map,
            state.map_words,
            state.count,
        );
        let base = self.constant(u32::from(offset));
        let address = if a == 255 {
            base
        } else {
            let register = self.read_register(a);
            self.value(op::I_ADD, self.u32, &[register, base])
        };
        let two = self.constant(2);
        let word_index = self.value(op::SHIFT_RIGHT_LOGICAL, self.u32, &[address, two]);
        let payload_class = storage::TASK_PAYLOAD_WORKGROUP_EXT;
        if write {
            let data = self.read_register(value);
            return match (mesh.task, output, region, size) {
                (true, true, 3, 2) => {
                    let payload = payload.expect("a task payload");
                    self.bounded_store(payload, payload_class, word_index, mesh.payload_words, data)
                }
                (false, true, 3, 2) => self.bounded_store(
                    attributes,
                    storage::WORKGROUP,
                    word_index,
                    attribute_words,
                    data,
                ),
                // The map, by byte: the primitive count at byte 3, the
                // vertex indices from byte 4 (a U16 or 32-bit write sets
                // two or four of them, low byte first).
                (false, true, 0, 0..=2) => {
                    let bytes = 1u32 << size;
                    let index_bytes = |e: &mut Self| -> Result<(), Unsupported> {
                        for n in 0..bytes {
                            let at = e.constant(n);
                            let at = e.value(op::I_ADD, e.u32, &[address, at]);
                            let shift = e.constant(8 * n);
                            let byte = e.value(op::SHIFT_RIGHT_LOGICAL, e.u32, &[data, shift]);
                            let mask = e.constant(0xff);
                            let byte = e.value(op::BITWISE_AND, e.u32, &[byte, mask]);
                            e.bounded_store(map, storage::WORKGROUP, at, map_words, byte)?;
                        }
                        Ok(())
                    };
                    let count_value = |e: &mut Self| match size {
                        0 => {
                            let mask = e.constant(0xff);
                            e.value(op::BITWISE_AND, e.u32, &[data, mask])
                        }
                        1 => {
                            let mask = e.constant(0xffff);
                            e.value(op::BITWISE_AND, e.u32, &[data, mask])
                        }
                        _ => data,
                    };
                    if a == 255 {
                        if offset == 3 {
                            let value = count_value(self);
                            self.function.emit(op::STORE, &[count, value]);
                            Ok(())
                        } else {
                            index_bytes(self)
                        }
                    } else {
                        // A register address: the count where it is byte 3.
                        let three = self.constant(3);
                        let is_count = self.value(op::I_EQUAL, self.bool, &[address, three]);
                        let mut result = Ok(());
                        self.choose(is_count, |e, is_count| {
                            if is_count {
                                let value = count_value(e);
                                e.function.emit(op::STORE, &[count, value]);
                            } else if let Err(error) = index_bytes(e) {
                                result = Err(error);
                            }
                        });
                        result
                    }
                }
                _ => Err(unsupported(
                    index,
                    format!(
                        "ISBEWR region {region} size {size}{}",
                        if output { " .O" } else { "" }
                    ),
                )),
            };
        }
        let result = match (output, region, size) {
            // The program's own outputs, read back.
            (true, 3, 2) if mesh.task => {
                let payload = payload.expect("a task payload");
                self.bounded_word(payload, payload_class, word_index, mesh.payload_words)
            }
            (true, 3, 2) => {
                self.bounded_word(attributes, storage::WORKGROUP, word_index, attribute_words)
            }
            (false, 3, 2) if !mesh.task && mesh.has_task => {
                // The workgroup index at byte 0, the payload after it.
                let id = self.builtin(BUILTIN_WORKGROUP_ID, 0);
                let payload = payload.expect("a task payload");
                let word =
                    self.bounded_word(payload, payload_class, word_index, mesh.payload_words);
                let zero = self.constant(0);
                let first = self.value(op::I_EQUAL, self.bool, &[address, zero]);
                self.value(op::SELECT, self.u32, &[first, id, word])
            }
            (false, 3, 2) if a == 255 => match offset {
                0 => self.builtin(BUILTIN_WORKGROUP_ID, 0),
                4 => {
                    let x = self.builtin(BUILTIN_NUM_WORKGROUPS, 0);
                    let y = self.builtin(BUILTIN_NUM_WORKGROUPS, 1);
                    let z = self.builtin(BUILTIN_NUM_WORKGROUPS, 2);
                    let xy = self.value(op::I_MUL, self.u32, &[x, y]);
                    self.value(op::I_MUL, self.u32, &[xy, z])
                }
                8 | 12 | 16 => self.builtin(BUILTIN_NUM_WORKGROUPS, u32::from(offset - 8) / 4),
                _ => {
                    return Err(unsupported(
                        index,
                        format!("ISBERD of input byte {offset:#x} without a task program"),
                    ));
                }
            },
            _ => {
                return Err(unsupported(
                    index,
                    format!(
                        "ISBERD region {region} size {size}{}{}",
                        if output { " .O" } else { "" },
                        if a == 255 {
                            ""
                        } else {
                            " at a register address"
                        }
                    ),
                ));
            }
        };
        if value != 255 {
            self.write_register(value, result);
        }
        Ok(())
    }

    /// An output array of `count` elements of `ty`.
    fn output_array(&mut self, ty: u32, count: u32) -> u32 {
        let length = self.constant(count);
        let array = self.module.ty(Type::Array {
            element: ty,
            length,
        });
        let pointer = self.module.pointer(storage::OUTPUT, array);
        let variable = self.module.global_variable(pointer, storage::OUTPUT);
        self.interface.push(variable);
        variable
    }

    /// A loop over `first`, `first + step`, ... below `limit` (values).
    fn strided(
        &mut self,
        first: u32,
        step: u32,
        limit: u32,
        mut body: impl FnMut(&mut Self, u32) -> Result<(), Unsupported>,
    ) -> Result<(), Unsupported> {
        let counter = self.function.variable(&mut self.module, self.u32_ptr, None);
        self.function.emit(op::STORE, &[counter, first]);
        let (header, check, inner, continuing, merge) = (
            self.module.id(),
            self.module.id(),
            self.module.id(),
            self.module.id(),
            self.module.id(),
        );
        self.function.emit(op::BRANCH, &[header]);
        self.function.label(header);
        self.function
            .emit(op::LOOP_MERGE, &[merge, continuing, crate::spirv::NONE]);
        self.function.emit(op::BRANCH, &[check]);
        self.function.label(check);
        let value = self.value(op::LOAD, self.u32, &[counter]);
        let more = self.value(op::U_LESS_THAN, self.bool, &[value, limit]);
        self.function
            .emit(op::BRANCH_CONDITIONAL, &[more, inner, merge]);
        self.function.label(inner);
        body(self, value)?;
        self.function.emit(op::BRANCH, &[continuing]);
        self.function.label(continuing);
        let next = self.value(op::I_ADD, self.u32, &[value, step]);
        self.function.emit(op::STORE, &[counter, next]);
        self.function.emit(op::BRANCH, &[header]);
        self.function.label(merge);
        Ok(())
    }

    /// Element `element` of the lines starting at word `start`: its index
    /// in the stage buffer's attributes, line `rank`.
    pub(super) fn line_index(
        &mut self,
        start: u32,
        line_count: u32,
        element: u32,
        rank: u32,
    ) -> u32 {
        let five = self.constant(5);
        let group = self.value(op::SHIFT_RIGHT_LOGICAL, self.u32, &[element, five]);
        let group_words = self.constant(line_count * GROUP);
        let group_start = self.value(op::I_MUL, self.u32, &[group, group_words]);
        let mask = self.constant(GROUP - 1);
        let slot = self.value(op::BITWISE_AND, self.u32, &[element, mask]);
        let fixed = self.constant(start + rank * GROUP);
        let index = self.value(op::I_ADD, self.u32, &[group_start, slot]);
        self.value(op::I_ADD, self.u32, &[index, fixed])
    }

    /// The mesh outputs from the stage buffer.
    fn mesh_outputs(&mut self, mesh: &MeshLayout) -> Result<(), Unsupported> {
        let state = self.mesh.as_ref().expect("mesh variables");
        let (attributes, attribute_words, map, map_words, count) = (
            state.attributes,
            state.attribute_words,
            state.map,
            state.map_words,
            state.count,
        );
        let written = self.value(op::LOAD, self.u32, &[count]);
        let maximum = self.constant(mesh.max_primitives);
        let primitives = self.glsl(self.u32, glsl::U_MIN, &[written, maximum]);
        let vertices = self.constant(mesh.max_vertices);
        self.function
            .emit(op::SET_MESH_OUTPUTS_EXT, &[vertices, primitives]);
        let invocation = self.scalar_builtin(BUILTIN_LOCAL_INVOCATION_INDEX);
        let step = self.constant(mesh.threads);
        // Per-vertex outputs.
        let vertex_lines = lines(&mesh.vertex_outputs);
        let vec4 = self.vec4();
        let mut targets = Vec::new();
        let mut position = None;
        let mut point_size = None;
        let mut generics = std::collections::BTreeMap::new();
        for (rank, &address) in vertex_lines.iter().enumerate() {
            let target = match address {
                0x5c => None,
                0x6c if mesh.vertices_per_primitive == 1 => {
                    let variable = *point_size.get_or_insert_with(|| {
                        let variable = self.output_array(self.f32, mesh.max_vertices);
                        self.module
                            .decorate(variable, decoration::BUILT_IN, &[BUILTIN_POINT_SIZE]);
                        variable
                    });
                    Some((variable, None))
                }
                0x6c => None,
                0x70..=0x7f => {
                    let variable = *position.get_or_insert_with(|| {
                        let variable = self.output_array(vec4, mesh.max_vertices);
                        self.module
                            .decorate(variable, decoration::BUILT_IN, &[BUILTIN_POSITION]);
                        // Invariant, as a vertex program's (a depth
                        // prepass and its lit pass agree).
                        self.module.decorate(variable, decoration::INVARIANT, &[]);
                        variable
                    });
                    Some((variable, Some((address - 0x70) / 4)))
                }
                0x80..=0x27f => {
                    let location = (address - 0x80) / 16;
                    let variable = *generics.entry(location).or_insert_with(|| {
                        let variable = self.output_array(vec4, mesh.max_vertices);
                        self.module
                            .decorate(variable, DECORATION_LOCATION, &[location]);
                        variable
                    });
                    self.io.stored_generics |= 1 << location;
                    Some((variable, Some(address / 4 % 4)))
                }
                _ => {
                    return Err(unsupported(
                        0,
                        format!("mesh per-vertex output {address:#x}"),
                    ));
                }
            };
            if let Some(target) = target {
                let bit = address / 4;
                self.io.stores[(bit / 64) as usize] |= 1 << (bit % 64);
                targets.push((rank as u32, target));
            }
        }
        let line_count = vertex_lines.len() as u32;
        self.strided(invocation, step, vertices, |e, vertex| {
            for &(rank, (variable, component)) in &targets {
                let at = e.line_index(0, line_count, vertex, rank);
                let word = e.bounded_word(attributes, storage::WORKGROUP, at, attribute_words);
                let value = e.value(op::BITCAST, e.f32, &[word]);
                let pointer = e.module.pointer(storage::OUTPUT, e.f32);
                let pointer = match component {
                    Some(component) => {
                        let component = e.constant(component);
                        e.value(op::ACCESS_CHAIN, pointer, &[variable, vertex, component])
                    }
                    None => e.value(op::ACCESS_CHAIN, pointer, &[variable, vertex]),
                };
                e.function.emit(op::STORE, &[pointer, value]);
            }
            Ok(())
        })?;
        // Per-primitive outputs and the indices.
        let primitive_lines = mesh
            .primitive_outputs
            .map_or(Vec::new(), |outputs| lines(&outputs));
        let mut primitive_targets = Vec::new();
        for (rank, &address) in primitive_lines.iter().enumerate() {
            let builtin = match address {
                0x5c => continue,
                0x60 => BUILTIN_PRIMITIVE_ID,
                0x64 => {
                    self.module.capability(capability::SHADER_LAYER);
                    BUILTIN_LAYER
                }
                0x68 => {
                    self.module.capability(capability::SHADER_VIEWPORT_INDEX);
                    BUILTIN_VIEWPORT_INDEX
                }
                _ => {
                    return Err(unsupported(
                        0,
                        format!("mesh per-primitive output {address:#x}"),
                    ));
                }
            };
            let variable = self.output_array(self.u32, mesh.max_primitives);
            self.module
                .decorate(variable, decoration::BUILT_IN, &[builtin]);
            self.module
                .decorate(variable, DECORATION_PER_PRIMITIVE, &[]);
            primitive_targets.push((rank as u32, variable));
        }
        let per_primitive = self.vertices_per_primitive_type(mesh);
        let (index_type, index_builtin) = per_primitive;
        let indices = self.output_array(index_type, mesh.max_primitives);
        self.module
            .decorate(indices, decoration::BUILT_IN, &[index_builtin]);
        let primitive_start = mesh.vertex_words();
        let primitive_line_count = primitive_lines.len() as u32;
        let corners = mesh.vertices_per_primitive();
        self.strided(invocation, step, primitives, |e, primitive| {
            let stride = e.constant(corners);
            let first = e.value(op::I_MUL, e.u32, &[primitive, stride]);
            let four = e.constant(4);
            let first = e.value(op::I_ADD, e.u32, &[first, four]);
            let mut corner_values = Vec::new();
            for corner in 0..corners {
                let offset = e.constant(corner);
                let at = e.value(op::I_ADD, e.u32, &[first, offset]);
                corner_values.push(e.bounded_word(map, storage::WORKGROUP, at, map_words));
            }
            let value = e.vector(e.u32, &corner_values);
            let pointer = e.module.pointer(storage::OUTPUT, index_type);
            let pointer = e.value(op::ACCESS_CHAIN, pointer, &[indices, primitive]);
            e.function.emit(op::STORE, &[pointer, value]);
            for &(rank, variable) in &primitive_targets {
                let at = e.line_index(primitive_start, primitive_line_count, primitive, rank);
                let word = e.bounded_word(attributes, storage::WORKGROUP, at, attribute_words);
                let pointer = e.module.pointer(storage::OUTPUT, e.u32);
                let pointer = e.value(op::ACCESS_CHAIN, pointer, &[variable, primitive]);
                e.function.emit(op::STORE, &[pointer, word]);
            }
            Ok(())
        })
    }

    /// The type and built-in of a primitive's vertex indices.
    fn vertices_per_primitive_type(&mut self, mesh: &MeshLayout) -> (u32, u32) {
        match mesh.vertices_per_primitive {
            1 => (self.u32, BUILTIN_PRIMITIVE_POINT_INDICES),
            count => {
                let ty = self.module.ty(Type::Vector {
                    component: self.u32,
                    count: u32::from(count),
                });
                let builtin = if count == 2 {
                    BUILTIN_PRIMITIVE_LINE_INDICES
                } else {
                    BUILTIN_PRIMITIVE_TRIANGLE_INDICES
                };
                (ty, builtin)
            }
        }
    }

    /// The SPIR-V execution model of a task or mesh stage.
    pub(super) fn mesh_model(stage: Stage) -> Option<u32> {
        match stage {
            Stage::Task => Some(execution::TASK_EXT),
            Stage::Mesh => Some(execution::MESH_EXT),
            _ => None,
        }
    }
}
