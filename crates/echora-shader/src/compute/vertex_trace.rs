//! Debugging: vertex tracing (`Layout::vertex_trace`). Each vertex or
//! mesh program records, for its first invocations, its final position
//! and the raw results of its texture instructions in a buffer at the
//! device address two specialization constants give, so one GPU run shows
//! what a vertex program really computed there.
//!
//! The buffer holds `SLOTS` records of `RECORD_WORDS` u32 words. A vertex
//! program's record `VertexIndex - BaseVertex` (the index the draw read)
//! is written by its invocation of the draw's first instance
//! (`InstanceIndex == BaseInstance`); a mesh program's record `n` by
//! invocation `n` of workgroup (0, 0, 0). Words:
//!
//! - 0: bit 0 set when the invocation reached its end; a mesh program's
//!   bit 1 when the workgroup wrote its outputs.
//! - 1: flags. Bit 0: the position has a NaN component; bit 1: an
//!   infinite one; bit 8 + k: texture entry k's result has a NaN word (as
//!   a float).
//! - 2, 3: a vertex program's VertexIndex and InstanceIndex (raw); a mesh
//!   program's LocalInvocationIndex and WorkgroupId.x.
//! - 4..8: the position as written (x, y, z, w; float bits). A mesh
//!   program's is output vertex `n`'s (from the stage buffer).
//! - 8 + 6k .. 14 + 6k, texture entry k (k < `ENTRIES`, one per texture
//!   instruction in the order the translation emits them): its byte offset
//!   in the program code plus one (0: no entry), the times this invocation
//!   ran it, and the four result words the last run produced (before the
//!   instruction's mask picks the written components; integer bits).
//!
//! Words 104..128 are unused. Nothing is written while the address is 0;
//! the runtime zeroes the buffer before the draw it traces.

use echora_shader_abi::{
    SpecConstant, SpecMeaning, VERTEX_TRACE_HIGH_SPEC_ID, VERTEX_TRACE_LOW_SPEC_ID,
};

use super::{Emitter, Stage};
use crate::spirv::{capability, op};

/// Records in the buffer.
pub const SLOTS: u32 = 64;
/// Words per record.
pub const RECORD_WORDS: u32 = 128;
/// Texture entries per record.
pub const ENTRIES: usize = 16;
const ENTRY_START: u32 = 8;
const ENTRY_WORDS: u32 = 6;

const BUILTIN_POSITION: u32 = 0;
const BUILTIN_WORKGROUP_ID: u32 = 26;
const BUILTIN_LOCAL_INVOCATION_INDEX: u32 = 29;
const BUILTIN_VERTEX_INDEX: u32 = 42;
const BUILTIN_INSTANCE_INDEX: u32 = 43;
const BUILTIN_BASE_VERTEX: u32 = 4424;
const BUILTIN_BASE_INSTANCE: u32 = 4425;

impl Emitter {
    /// Whether this program records a vertex trace.
    fn vertex_traced(&self) -> bool {
        self.layout.vertex_trace
            && match self.layout.mesh {
                Some(mesh) => !mesh.task,
                None => self.stage == Stage::Vertex,
            }
    }

    /// The trace buffer's address (a u64 value), from its two
    /// specialization constants.
    fn vertex_trace_buffer(&mut self) -> u32 {
        let (low, high) = match self.vertex_trace_constants {
            Some(words) => words,
            None => {
                let mut words = [0; 2];
                for (word, (id, meaning)) in words.iter_mut().zip([
                    (VERTEX_TRACE_LOW_SPEC_ID, SpecMeaning::VertexTraceLow),
                    (VERTEX_TRACE_HIGH_SPEC_ID, SpecMeaning::VertexTraceHigh),
                ]) {
                    let constant = self.module.u32_spec_constant(0);
                    self.module.decorate(constant, 1, &[id]);
                    self.spec_constants.push(SpecConstant { id, meaning });
                    *word = constant;
                }
                self.vertex_trace_constants = Some((words[0], words[1]));
                (words[0], words[1])
            }
        };
        self.join(low, high)
    }

    /// This invocation's record (a u64 address; 0: not traced) and its
    /// slot.
    fn vertex_trace_slot(&mut self) -> (u32, u32) {
        let (slot, first) = if self.layout.mesh.is_some() {
            let slot = self.scalar_builtin(BUILTIN_LOCAL_INVOCATION_INDEX);
            let zero = self.constant(0);
            let mut first = self.module.bool_constant(true);
            for component in 0..3 {
                let id = self.builtin(BUILTIN_WORKGROUP_ID, component);
                let is_zero = self.value(op::I_EQUAL, self.bool, &[id, zero]);
                first = self.value(op::LOGICAL_AND, self.bool, &[first, is_zero]);
            }
            (slot, first)
        } else {
            self.module.capability(capability::DRAW_PARAMETERS);
            let vertex = self.scalar_builtin(BUILTIN_VERTEX_INDEX);
            let base = self.scalar_builtin(BUILTIN_BASE_VERTEX);
            let slot = self.value(op::I_SUB, self.u32, &[vertex, base]);
            let instance = self.scalar_builtin(BUILTIN_INSTANCE_INDEX);
            let base = self.scalar_builtin(BUILTIN_BASE_INSTANCE);
            let first = self.value(op::I_EQUAL, self.bool, &[instance, base]);
            (slot, first)
        };
        let buffer = self.vertex_trace_buffer();
        let zero64 = self.constant64(0);
        let bound = self.value(op::I_NOT_EQUAL, self.bool, &[buffer, zero64]);
        let slots = self.constant(SLOTS);
        let inside = self.value(op::U_LESS_THAN, self.bool, &[slot, slots]);
        let traced = self.value(op::LOGICAL_AND, self.bool, &[bound, first]);
        let traced = self.value(op::LOGICAL_AND, self.bool, &[traced, inside]);
        let u64 = self.u64();
        let wide = self.value(op::U_CONVERT, u64, &[slot]);
        let stride = self.constant64(u64::from(RECORD_WORDS) * 4);
        let offset = self.value(op::I_MUL, u64, &[wide, stride]);
        let at = self.value(op::I_ADD, u64, &[buffer, offset]);
        let record = self.value(op::SELECT, u64, &[traced, at, zero64]);
        (record, slot)
    }

    /// At the program's start (in the function that runs its code): the
    /// invocation's record and its identity words.
    pub(super) fn vertex_trace_begin(&mut self) {
        if !self.vertex_traced() {
            return;
        }
        let (record, slot) = self.vertex_trace_slot();
        self.vertex_trace_record = Some(record);
        let (first, second) = if self.layout.mesh.is_some() {
            (slot, self.builtin(BUILTIN_WORKGROUP_ID, 0))
        } else {
            // Declared here so that every exit can read it.
            let vec4 = self.vec4();
            self.builtin_output(BUILTIN_POSITION, vec4);
            (
                self.scalar_builtin(BUILTIN_VERTEX_INDEX),
                self.scalar_builtin(BUILTIN_INSTANCE_INDEX),
            )
        };
        self.trace_atomic(record, 2, op::ATOMIC_EXCHANGE, first);
        self.trace_atomic(record, 3, op::ATOMIC_EXCHANGE, second);
    }

    /// A texture instruction at `index` produced `results`: its entry.
    pub(super) fn vertex_trace_texture(&mut self, index: usize, results: &[u32]) {
        let Some(record) = self.vertex_trace_record else {
            return;
        };
        let entry = match self.vertex_trace_entries.iter().position(|&i| i == index) {
            Some(entry) => entry,
            None if self.vertex_trace_entries.len() < ENTRIES => {
                self.vertex_trace_entries.push(index);
                self.vertex_trace_entries.len() - 1
            }
            None => return,
        };
        let start = ENTRY_START + entry as u32 * ENTRY_WORDS;
        let offset = self.constant(u32::try_from(index * 16 + 1).unwrap_or(u32::MAX));
        self.trace_atomic(record, start, op::ATOMIC_EXCHANGE, offset);
        let one = self.constant(1);
        self.trace_atomic(record, start + 1, op::ATOMIC_I_ADD, one);
        let zero = self.constant(0);
        let mut nan = self.module.bool_constant(false);
        for n in 0..4 {
            let word = results.get(n).copied().unwrap_or(zero);
            self.trace_atomic(record, start + 2 + n as u32, op::ATOMIC_EXCHANGE, word);
            let float = self.value(op::BITCAST, self.f32, &[word]);
            let is_nan = self.value(op::IS_NAN, self.bool, &[float]);
            nan = self.value(op::LOGICAL_OR, self.bool, &[nan, is_nan]);
        }
        let bit = self.constant(1 << (8 + entry));
        let flag = self.value(op::SELECT, self.u32, &[nan, bit, zero]);
        self.trace_atomic(record, 1, op::ATOMIC_OR, flag);
    }

    /// At an exit: a vertex program's position and flags, and the end bit.
    pub(super) fn vertex_trace_end(&mut self) {
        let Some(record) = self.vertex_trace_record else {
            return;
        };
        if self.layout.mesh.is_none() {
            let vec4 = self.vec4();
            let variable = self.builtin_output(BUILTIN_POSITION, vec4);
            let value = self.value(op::LOAD, vec4, &[variable]);
            let words: Vec<u32> = (0..4)
                .map(|n| {
                    let component = self.value(op::COMPOSITE_EXTRACT, self.f32, &[value, n]);
                    self.value(op::BITCAST, self.u32, &[component])
                })
                .collect();
            self.vertex_trace_position(record, &words);
        }
        let one = self.constant(1);
        self.trace_atomic(record, 0, op::ATOMIC_OR, one);
    }

    /// The position words (u32 bits) and their flags.
    fn vertex_trace_position(&mut self, record: u32, words: &[u32]) {
        let zero = self.constant(0);
        let mut flags = zero;
        for (n, &word) in words.iter().enumerate() {
            self.trace_atomic(record, 4 + n as u32, op::ATOMIC_EXCHANGE, word);
            let float = self.value(op::BITCAST, self.f32, &[word]);
            for (test, bit) in [(op::IS_NAN, 1), (op::IS_INF, 2)] {
                let hit = self.value(test, self.bool, &[float]);
                let bit = self.constant(bit);
                let bit = self.value(op::SELECT, self.u32, &[hit, bit, zero]);
                flags = self.value(op::BITWISE_OR, self.u32, &[flags, bit]);
            }
        }
        self.trace_atomic(record, 1, op::ATOMIC_OR, flags);
    }

    /// A mesh program's entry point, after its workgroup's code (the stage
    /// buffer complete): output vertex `n`'s position in record `n`.
    pub(super) fn vertex_trace_mesh_outputs(
        &mut self,
        attributes: u32,
        attribute_words: u32,
        vertex_lines: &[u32],
        max_vertices: u32,
    ) {
        if !self.vertex_traced() {
            return;
        }
        let (record, slot) = self.vertex_trace_slot();
        let line_count = vertex_lines.len() as u32;
        let vertices = self.constant(max_vertices);
        let inside = self.value(op::U_LESS_THAN, self.bool, &[slot, vertices]);
        let last = self.constant(max_vertices.saturating_sub(1));
        let vertex = self.glsl(self.u32, crate::spirv::glsl::U_MIN, &[slot, last]);
        let zero = self.constant(0);
        let words: Vec<u32> = (0..4u32)
            .map(
                |n| match vertex_lines.iter().position(|&line| line == 0x70 + n * 4) {
                    Some(rank) => {
                        let at = self.line_index(0, line_count, vertex, rank as u32);
                        let word = self.bounded_word(
                            attributes,
                            crate::spirv::storage::WORKGROUP,
                            at,
                            attribute_words,
                        );
                        self.value(op::SELECT, self.u32, &[inside, word, zero])
                    }
                    None => zero,
                },
            )
            .collect();
        self.vertex_trace_position(record, &words);
        let two = self.constant(2);
        self.trace_atomic(record, 0, op::ATOMIC_OR, two);
    }
}
