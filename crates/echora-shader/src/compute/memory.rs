//! Loads and stores: global memory through the GPU page table, shared
//! memory as a workgroup array and local memory as a per-invocation array.
//!
//! A global address is looked up in the page table the push constants
//! give (`echora_shader_abi::PageTable`): its page's entry is the device
//! address of that page, 0 for an unmapped one. An access outside the
//! table or on an unmapped page reads zero and writes nothing. Shared and
//! local accesses outside their array do the same. Accesses are naturally
//! aligned and at most 16 bytes, so none crosses a page.

use echora_shader_abi::{PAGE_SHIFT, PAGE_SIZE, PUSH_PAGE_TABLE, PUSH_PAGE_TABLE_OFFSET};

use super::{Emitter, unsupported};
use crate::decode::{Address, Pred, Space};
use crate::spirv::{Type, capability, decoration, glsl, op, storage};
use echora_shader_abi::Unsupported;

/// Memory objects a program has declared so far.
#[derive(Debug, Default)]
pub(super) struct Memory {
    /// Whether global memory (physical pointers) is used.
    pub(super) physical: bool,
    push_constants: Option<u32>,
    shared: Option<u32>,
    local: Option<u32>,
}

/// SPIR-V's memory-access operand for an aligned access.
const ALIGNED: u32 = 2;

/// Scopes and memory semantics for barriers.
const SCOPE_WORKGROUP: u32 = 2;
const ACQUIRE_RELEASE_WORKGROUP_MEMORY: u32 = 0x8 | 0x100;

/// Words a load or store of memory type `size` moves (a sub-word access
/// moves the word that holds it).
fn words(size: u8) -> Option<u32> {
    match size {
        0..=4 => Some(1),
        5 => Some(2),
        6 => Some(4),
        _ => None,
    }
}

impl Emitter {
    /// Runs `body` only when the boolean value `condition` holds.
    fn when(
        &mut self,
        condition: u32,
        body: impl FnOnce(&mut Self) -> Result<(), Unsupported>,
    ) -> Result<(), Unsupported> {
        let then = self.module.id();
        let merge = self.module.id();
        self.function
            .emit(op::SELECTION_MERGE, &[merge, crate::spirv::NONE]);
        self.function
            .emit(op::BRANCH_CONDITIONAL, &[condition, then, merge]);
        self.function.label(then);
        body(self)?;
        self.function.emit(op::BRANCH, &[merge]);
        self.function.label(merge);
        Ok(())
    }

    /// A workgroup barrier that also orders shared memory.
    pub(super) fn barrier(&mut self) {
        let scope = self.constant(SCOPE_WORKGROUP);
        let semantics = self.constant(ACQUIRE_RELEASE_WORKGROUP_MEMORY);
        self.function
            .emit(op::CONTROL_BARRIER, &[scope, scope, semantics]);
    }

    pub(super) fn load(
        &mut self,
        index: usize,
        d: u8,
        space: Space,
        address: Address,
        size: u8,
        condition: Pred,
    ) -> Result<(), Unsupported> {
        let count = words(size).ok_or_else(|| unsupported(index, format!("load size {size}")))?;
        let (values, low_bits) = match space {
            Space::Global => self.global_load(address, count, condition)?,
            Space::Shared | Space::Local => {
                if !condition.is_true() {
                    return Err(unsupported(index, "predicated shared or local load"));
                }
                self.array_load(index, space, address, count)?
            }
        };
        let values = if size < 4 {
            vec![self.sub_word(values[0], low_bits, size)]
        } else {
            values
        };
        if d != 255 {
            for (n, value) in values.into_iter().enumerate() {
                self.write_register(d.saturating_add(n as u8), value);
            }
        }
        Ok(())
    }

    pub(super) fn store(
        &mut self,
        index: usize,
        data: u8,
        space: Space,
        address: Address,
        size: u8,
    ) -> Result<(), Unsupported> {
        let count = match words(size) {
            Some(count) if size >= 4 => count,
            _ => return Err(unsupported(index, format!("store size {size}"))),
        };
        let values: Vec<u32> = (0..count)
            .map(|n| {
                if data == 255 {
                    self.constant(0)
                } else {
                    self.read_register(data.saturating_add(n as u8))
                }
            })
            .collect();
        match space {
            Space::Global => self.global_store(address, &values),
            Space::Shared | Space::Local => self.array_store(index, space, address, &values),
        }
    }

    /// An atomic read-modify-write, or with `compare` a compare-and-swap;
    /// `d` receives the old value (0 where nothing was there).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn atomic(
        &mut self,
        index: usize,
        d: u8,
        space: Space,
        address: Address,
        data: u8,
        compare: Option<u8>,
        operation: u8,
        kind: u8,
    ) -> Result<(), Unsupported> {
        let float = kind == 3;
        let opcode = match (compare.is_some(), operation, kind) {
            (true, _, 0 | 1) => op::ATOMIC_COMPARE_EXCHANGE,
            (false, 0, 0 | 1) => op::ATOMIC_I_ADD,
            (false, 0, 3) => op::ATOMIC_F_ADD,
            (false, 1, 0) => op::ATOMIC_U_MIN,
            (false, 1, 1) => op::ATOMIC_S_MIN,
            (false, 2, 0) => op::ATOMIC_U_MAX,
            (false, 2, 1) => op::ATOMIC_S_MAX,
            (false, 5, 0 | 1) => op::ATOMIC_AND,
            (false, 6, 0 | 1) => op::ATOMIC_OR,
            (false, 7, 0 | 1) => op::ATOMIC_XOR,
            (false, 8, 0 | 1) => op::ATOMIC_EXCHANGE,
            _ => {
                return Err(unsupported(
                    index,
                    format!("atomic operation {operation} on type {kind}"),
                ));
            }
        };
        if float {
            self.module.capability(capability::ATOMIC_FLOAT32_ADD);
            self.module.extension("SPV_EXT_shader_atomic_float_add");
        }
        let value = self.read_register(data);
        let compare = compare.map(|reg| self.read_register(reg));
        let result = self.result_variables(1)[0];
        let scope = self.constant(if space == Space::Global {
            SCOPE_DEVICE
        } else {
            SCOPE_WORKGROUP
        });
        let relaxed = self.constant(0);
        let apply = |e: &mut Self, pointer: u32| {
            let (ty, value) = if float {
                (e.f32, e.value(op::BITCAST, e.f32, &[value]))
            } else {
                (e.u32, value)
            };
            let old = match compare {
                Some(compare) => e.value(
                    opcode,
                    ty,
                    &[pointer, scope, relaxed, relaxed, value, compare],
                ),
                None => e.value(opcode, ty, &[pointer, scope, relaxed, value]),
            };
            let old = if float {
                e.value(op::BITCAST, e.u32, &[old])
            } else {
                old
            };
            e.function.emit(op::STORE, &[result, old]);
        };
        match space {
            Space::Global => {
                let va = self.global_address(address);
                self.mapped(va, None, |e, device| {
                    let pointer = if float {
                        let pointer = e.module.pointer(storage::PHYSICAL_STORAGE_BUFFER, e.f32);
                        e.value(op::CONVERT_U_TO_PTR, pointer, &[device])
                    } else {
                        e.device_pointer(device, 1).0
                    };
                    apply(e, pointer);
                    Ok(())
                })?;
            }
            // Shared memory is an array of words.
            Space::Shared if float => {
                return Err(unsupported(index, "shared float atomic"));
            }
            Space::Shared => {
                let (variable, bytes) = self.array(index, space)?;
                let at = self.array_address(address);
                let (element, inside) = self.array_word(variable, storage::WORKGROUP, bytes, at, 0);
                self.when(inside, |e| {
                    apply(e, element);
                    Ok(())
                })?;
            }
            Space::Local => return Err(unsupported(index, "local atomic")),
        }
        let old = self.value(op::LOAD, self.u32, &[result]);
        if !self.zero_register(d) {
            self.write_register(d, old);
        }
        Ok(())
    }

    /// The `size`-typed value at byte `low_bits` (0-3) of `word`.
    fn sub_word(&mut self, word: u32, low_bits: u32, size: u8) -> u32 {
        let eight = self.constant(8);
        let shift = self.value(op::I_MUL, self.u32, &[low_bits, eight]);
        let bits = self.constant(if size < 2 { 8 } else { 16 });
        let opcode = if size & 1 == 1 {
            op::BIT_FIELD_S_EXTRACT
        } else {
            op::BIT_FIELD_U_EXTRACT
        };
        self.value(opcode, self.u32, &[word, shift, bits])
    }

    /// The push-constant block's page-table fields: the table's device
    /// address, its first page and its page count (as 64-bit values).
    fn page_table(&mut self) -> (u32, u32, u32) {
        let u64 = self.u64();
        let variable = match self.memory.push_constants {
            Some(variable) => variable,
            None => {
                let block = self.module.ty(Type::Struct {
                    members: vec![u64, self.u32, self.u32],
                });
                self.module.decorate(block, decoration::BLOCK, &[]);
                for (member, offset) in [0, 8, 12].into_iter().enumerate() {
                    self.module.member_decorate(
                        block,
                        member as u32,
                        decoration::OFFSET,
                        &[PUSH_PAGE_TABLE_OFFSET + offset],
                    );
                }
                let pointer = self.module.pointer(storage::PUSH_CONSTANT, block);
                let variable = self.module.global_variable(pointer, storage::PUSH_CONSTANT);
                self.interface.push(variable);
                self.memory.push_constants = Some(variable);
                self.push_fields |= echora_shader_abi::PUSH_PAGE_TABLE;
                variable
            }
        };
        let mut fields = [0; 3];
        for (member, field) in fields.iter_mut().enumerate() {
            let ty = if member == 0 { u64 } else { self.u32 };
            let pointer = self.module.pointer(storage::PUSH_CONSTANT, ty);
            let member = self.constant(member as u32);
            let element = self.value(op::ACCESS_CHAIN, pointer, &[variable, member]);
            let value = self.value(op::LOAD, ty, &[element]);
            *field = if ty == u64 {
                value
            } else {
                self.widen(value, false)
            };
        }
        debug_assert_ne!(self.push_fields & PUSH_PAGE_TABLE, 0);
        (fields[0], fields[1], fields[2])
    }

    /// A global address as a 64-bit value.
    fn global_address(&mut self, address: Address) -> u32 {
        let u64 = self.u64();
        let base = if address.wide {
            self.read_pair(address.reg.0)
        } else {
            let low = self.read_register(address.reg.0);
            self.widen(low, false)
        };
        let mut sum = base;
        if let Some((uniform, wide)) = address.uniform {
            let low = self.read_uniform(uniform.0);
            let value = if wide {
                // URZ's pair is zero, and UR62's high word is URZ.
                let high = self.read_uniform(uniform.0.saturating_add(1).min(63));
                self.join(low, high)
            } else {
                self.widen(low, false)
            };
            sum = self.value(op::I_ADD, u64, &[sum, value]);
        }
        if address.offset != 0 {
            let offset = self.constant64(i64::from(address.offset) as u64);
            sum = self.value(op::I_ADD, u64, &[sum, offset]);
        }
        sum
    }

    /// Looks up global address `va` and, where it is mapped and `extra`
    /// holds, runs `access` with the device address of `va`.
    fn mapped(
        &mut self,
        va: u32,
        extra: Option<u32>,
        access: impl FnOnce(&mut Self, u32) -> Result<(), Unsupported>,
    ) -> Result<(), Unsupported> {
        self.memory.physical = true;
        self.module
            .capability(capability::PHYSICAL_STORAGE_BUFFER_ADDRESSES);
        let u64 = self.u64();
        let (table, first, count) = self.page_table();
        let shift = self.constant(PAGE_SHIFT);
        let page = self.value(op::SHIFT_RIGHT_LOGICAL, u64, &[va, shift]);
        let relative = self.value(op::I_SUB, u64, &[page, first]);
        let below = self.value(op::U_GREATER_THAN, self.bool, &[count, relative]);
        // A page before the first wraps to a huge relative page, so one
        // comparison covers both ends. No table bound: address 0.
        let zero = self.constant64(0);
        let bound = self.value(op::I_NOT_EQUAL, self.bool, &[table, zero]);
        let mut inside = self.value(op::LOGICAL_AND, self.bool, &[below, bound]);
        if let Some(extra) = extra {
            inside = self.value(op::LOGICAL_AND, self.bool, &[inside, extra]);
        }
        self.when(inside, |e| {
            let eight = e.constant64(8);
            let step = e.value(op::I_MUL, u64, &[relative, eight]);
            let entry = e.value(op::I_ADD, u64, &[table, step]);
            let pointer = e.module.pointer(storage::PHYSICAL_STORAGE_BUFFER, u64);
            let entry = e.value(op::CONVERT_U_TO_PTR, pointer, &[entry]);
            let base = e.value(op::LOAD, u64, &[entry, ALIGNED, 8]);
            let mapped = e.value(op::I_NOT_EQUAL, e.bool, &[base, zero]);
            e.when(mapped, |e| {
                let mask = e.constant64(PAGE_SIZE - 1);
                let within = e.value(op::BITWISE_AND, u64, &[va, mask]);
                let device = e.value(op::I_ADD, u64, &[base, within]);
                access(e, device)
            })
        })
    }

    /// A pointer to `count` words at a device address.
    fn device_pointer(&mut self, device: u32, count: u32) -> (u32, u32) {
        let ty = if count == 1 {
            self.u32
        } else {
            self.module.ty(Type::Vector {
                component: self.u32,
                count,
            })
        };
        let pointer = self.module.pointer(storage::PHYSICAL_STORAGE_BUFFER, ty);
        (self.value(op::CONVERT_U_TO_PTR, pointer, &[device]), ty)
    }

    /// Zero-initialised variables that receive a conditional load's words.
    fn result_variables(&mut self, count: u32) -> Vec<u32> {
        let zero = self.constant(0);
        (0..count)
            .map(|_| {
                let variable = self.function.variable(&mut self.module, self.u32_ptr, None);
                self.function.emit(op::STORE, &[variable, zero]);
                variable
            })
            .collect()
    }

    /// A global load's words, and the address's low two bits.
    fn global_load(
        &mut self,
        address: Address,
        count: u32,
        condition: Pred,
    ) -> Result<(Vec<u32>, u32), Unsupported> {
        let va = self.global_address(address);
        let extra = (!condition.is_true()).then(|| self.read_predicate(condition));
        let results = self.result_variables(count);
        let u64 = self.u64();
        self.mapped(va, extra, |e, device| {
            // A sub-word load reads the aligned word that holds it.
            let device = if count == 1 {
                let mask = e.constant64(!3);
                e.value(op::BITWISE_AND, u64, &[device, mask])
            } else {
                device
            };
            let (pointer, ty) = e.device_pointer(device, count);
            let value = e.value(op::LOAD, ty, &[pointer, ALIGNED, count * 4]);
            for (n, &result) in results.iter().enumerate() {
                let word = if count == 1 {
                    value
                } else {
                    e.value(op::COMPOSITE_EXTRACT, e.u32, &[value, n as u32])
                };
                e.function.emit(op::STORE, &[result, word]);
            }
            Ok(())
        })?;
        let values = results
            .into_iter()
            .map(|result| self.value(op::LOAD, self.u32, &[result]))
            .collect();
        let low = self.value(op::U_CONVERT, self.u32, &[va]);
        let three = self.constant(3);
        let low_bits = self.value(op::BITWISE_AND, self.u32, &[low, three]);
        Ok((values, low_bits))
    }

    fn global_store(&mut self, address: Address, values: &[u32]) -> Result<(), Unsupported> {
        let va = self.global_address(address);
        let count = values.len() as u32;
        self.mapped(va, None, |e, device| {
            let (pointer, ty) = e.device_pointer(device, count);
            let value = if count == 1 {
                values[0]
            } else {
                let mut operands = vec![ty];
                operands.extend_from_slice(values);
                let id = e.module.id();
                operands.insert(1, id);
                e.function.emit(op::COMPOSITE_CONSTRUCT, &operands);
                id
            };
            e.function
                .emit(op::STORE, &[pointer, value, ALIGNED, count * 4]);
            Ok(())
        })
    }

    /// The shared or local memory array and its size in bytes.
    fn array(&mut self, index: usize, space: Space) -> Result<(u32, u32), Unsupported> {
        let (bytes, existing, name) = match space {
            Space::Shared => (self.layout.shared_bytes, self.memory.shared, "shared"),
            _ => (self.layout.local_bytes, self.memory.local, "local"),
        };
        let bytes =
            bytes.ok_or_else(|| unsupported(index, format!("{name} memory size not given")))?;
        if bytes < 4 {
            return Err(unsupported(
                index,
                format!("{name} memory access with none"),
            ));
        }
        if let Some(variable) = existing {
            return Ok((variable, bytes));
        }
        let length = self.constant(bytes / 4);
        let array = self.module.ty(Type::Array {
            element: self.u32,
            length,
        });
        let variable = if space == Space::Shared {
            let pointer = self.module.pointer(storage::WORKGROUP, array);
            let variable = self.module.global_variable(pointer, storage::WORKGROUP);
            self.interface.push(variable);
            self.memory.shared = Some(variable);
            variable
        } else {
            let pointer = self.module.pointer(storage::FUNCTION, array);
            let variable = self.function.variable(&mut self.module, pointer, None);
            self.memory.local = Some(variable);
            variable
        };
        Ok((variable, bytes))
    }

    /// A shared or local access's byte address.
    fn array_address(&mut self, address: Address) -> u32 {
        let mut sum = self.read_register(address.reg.0);
        if address.scale != 1 {
            let scale = self.constant(u32::from(address.scale));
            sum = self.value(op::I_MUL, self.u32, &[sum, scale]);
        }
        if let Some((uniform, _)) = address.uniform {
            let value = self.read_uniform(uniform.0);
            sum = self.value(op::I_ADD, self.u32, &[sum, value]);
        }
        if address.offset != 0 {
            let offset = self.constant(address.offset as u32);
            sum = self.value(op::I_ADD, self.u32, &[sum, offset]);
        }
        sum
    }

    /// Word `n` of an array access: its pointer (clamped into the array)
    /// and whether it is inside.
    fn array_word(
        &mut self,
        variable: u32,
        storage_class: u32,
        bytes: u32,
        at: u32,
        n: u32,
    ) -> (u32, u32) {
        let offset = self.constant(n * 4);
        let byte = self.value(op::I_ADD, self.u32, &[at, offset]);
        let two = self.constant(2);
        let word = self.value(op::SHIFT_RIGHT_LOGICAL, self.u32, &[byte, two]);
        let last = self.constant(bytes / 4 - 1);
        let inside = self.value(op::U_LESS_THAN_EQUAL, self.bool, &[word, last]);
        let clamped = self.glsl(self.u32, glsl::U_MIN, &[word, last]);
        let pointer = self.module.pointer(storage_class, self.u32);
        let element = self.value(op::ACCESS_CHAIN, pointer, &[variable, clamped]);
        (element, inside)
    }

    fn array_load(
        &mut self,
        index: usize,
        space: Space,
        address: Address,
        count: u32,
    ) -> Result<(Vec<u32>, u32), Unsupported> {
        let (variable, bytes) = self.array(index, space)?;
        let class = storage_class(space);
        let at = self.array_address(address);
        let zero = self.constant(0);
        let mut values = Vec::new();
        for n in 0..count {
            let (element, inside) = self.array_word(variable, class, bytes, at, n);
            let value = self.value(op::LOAD, self.u32, &[element]);
            values.push(self.value(op::SELECT, self.u32, &[inside, value, zero]));
        }
        let three = self.constant(3);
        let low_bits = self.value(op::BITWISE_AND, self.u32, &[at, three]);
        Ok((values, low_bits))
    }

    fn array_store(
        &mut self,
        index: usize,
        space: Space,
        address: Address,
        values: &[u32],
    ) -> Result<(), Unsupported> {
        let (variable, bytes) = self.array(index, space)?;
        let class = storage_class(space);
        let at = self.array_address(address);
        for (n, &value) in values.iter().enumerate() {
            let (element, inside) = self.array_word(variable, class, bytes, at, n as u32);
            self.when(inside, |e| {
                e.function.emit(op::STORE, &[element, value]);
                Ok(())
            })?;
        }
        Ok(())
    }
}

/// Scope and semantics of an atomic: the device for global memory, the
/// workgroup for shared, relaxed either way.
const SCOPE_DEVICE: u32 = 1;

fn storage_class(space: Space) -> u32 {
    if space == Space::Shared {
        storage::WORKGROUP
    } else {
        storage::FUNCTION
    }
}
