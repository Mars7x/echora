//! An indirect mesh draw's task workgroup count, bounded on the GPU.
//!
//! An NVN indirect mesh draw's record holds one group count, written by
//! earlier GPU work (the culling passes). Vulkan's indirect mesh draw reads
//! three, and a count above maxTaskWorkGroupCount is not a valid one: the
//! count is moved into the draw's argument slot by a one-invocation compute
//! program that bounds it by VK_EXT_mesh_shader's minimum limit (65535),
//! and the original count is retained for ECHORA_MESH_COUNT_LOG and
//! device-loss diagnostics.

use ash::vk;
use echora_shader::spirv::{self, Function, Module, Type};

use crate::renderer::{HostBuffer, Renderer, vk_error};

/// VK_EXT_mesh_shader's minimum maxTaskWorkGroupCount per dimension.
pub(crate) const TASK_WORKGROUPS: u32 = 65535;

/// Counts kept in the log, the oldest overwritten.
const LOGGED: u32 = 65536;

/// Words ECHORA_DIAG_PROBE reads through the GPU's view (`probe_device_words`).
const PROBES: u32 = 64;
const PUSH_BYTES: usize = 32;

#[derive(Clone, Copy)]
pub(crate) enum IndirectCounts {
    Mesh,
    Dispatch,
}

impl IndirectCounts {
    fn axes(self) -> usize {
        match self {
            Self::Mesh => 1,
            Self::Dispatch => 3,
        }
    }
}

/// A bound from a diagnostic variable (tighter than the device limit, so a
/// diagnostic run cannot flood the GPU through wrong counts), else `limit`.
fn diagnostic_bound(name: &str, limit: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .map_or(limit, |bound| bound.min(limit))
}

pub(crate) struct MeshCount {
    pipelines: [vk::Pipeline; 2],
    layout: vk::PipelineLayout,
    /// Each draw's count as read (u32), by draw number modulo LOGGED.
    log: HostBuffer,
    log_address: u64,
    /// Draws recorded, and the last one the log was read up to.
    next: u64,
    reported: u64,
    /// ECHORA_DIAG_PROBE: words read through the device addresses the page
    /// table gives (u32 each), and how many are in use.
    probes: HostBuffer,
    probes_address: u64,
    probed: u32,
}

impl MeshCount {
    fn new(renderer: &Renderer) -> Result<Self, String> {
        let mut count = Self {
            pipelines: [vk::Pipeline::null(); 2],
            layout: vk::PipelineLayout::null(),
            log: HostBuffer::empty(),
            log_address: 0,
            next: 0,
            reported: 0,
            probes: HostBuffer::empty(),
            probes_address: 0,
            probed: 0,
        };
        if let Err(error) = count.initialize(renderer) {
            count.destroy(renderer.device());
            return Err(error);
        }
        Ok(count)
    }

    fn initialize(&mut self, renderer: &Renderer) -> Result<(), String> {
        let device = renderer.device();
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER;
        (self.log, self.log_address) =
            renderer.addressable_host_buffer(u64::from(LOGGED) * 4, usage)?;
        // Unwritten entries read u32::MAX: work the GPU never reached.
        self.log.write(0, &vec![0xff; LOGGED as usize * 4]);
        (self.probes, self.probes_address) =
            renderer.addressable_host_buffer(u64::from(PROBES) * 4, usage)?;
        self.probes.write(0, &vec![0xff; PROBES as usize * 4]);
        let range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(PUSH_BYTES as u32);
        let ranges = [range];
        let layout_info = vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&ranges);
        // SAFETY: a valid device and create info; cleaned up by destroy.
        self.layout = unsafe { device.create_pipeline_layout(&layout_info, None) }
            .map_err(vk_error("vkCreatePipelineLayout"))?;
        for kind in [IndirectCounts::Mesh, IndirectCounts::Dispatch] {
            self.pipelines[kind as usize] =
                crate::compute::internal_pipeline(device, self.layout, &bound_program(kind))?;
        }
        Ok(())
    }

    /// The counts read since the last report: each past the bound, then
    /// the largest (ECHORA_MESH_COUNT_LOG; also when a wait stalls or the
    /// device is lost, as the log is readable at any moment).
    pub(crate) fn report_pending(&self, why: &str) {
        let bytes = self.log.read_bytes(LOGGED as usize * 4);
        let counts: Vec<u32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        let from = self
            .reported
            .max(self.next.saturating_sub(u64::from(LOGGED)));
        let mut largest = 0;
        // Counts a diagnostic bound cut (the tighter of the task and
        // per-axis bounds; the log keeps one largest word a record).
        let diagnostic = diagnostic_bound("ECHORA_DIAG_TASK_GROUPS", TASK_WORKGROUPS).min(
            diagnostic_bound("ECHORA_DIAG_DISPATCH_AXIS", TASK_WORKGROUPS),
        );
        let mut cut = 0;
        let mut unrun = 0;
        for draw in from..self.next {
            let count = counts[(draw % u64::from(LOGGED)) as usize];
            if count == u32::MAX {
                unrun += 1;
                continue;
            }
            largest = largest.max(count);
            cut += u32::from(count > diagnostic);
            if count > TASK_WORKGROUPS {
                eprintln!(
                    "gpu: indirect draw or dispatch {draw}: count {count} past {TASK_WORKGROUPS}"
                );
            }
        }
        eprintln!(
            "gpu: indirect counts of draws and dispatches {from}..{} ({why}): largest {largest}, {cut} past the bound {diagnostic}, {unrun} not run by the GPU yet",
            self.next
        );
        if self.probed > 0 {
            let words: Vec<String> = self
                .probes
                .read_bytes(self.probed as usize * 4)
                .as_chunks::<4>()
                .0
                .iter()
                .map(|probe| match u32::from_le_bytes(*probe) {
                    u32::MAX => "not run".to_string(),
                    word => format!("{word:#x}"),
                })
                .collect();
            eprintln!("gpu: probed words as the GPU read them ({why}): {words:?}");
        }
    }

    pub(crate) fn destroy(self, device: &ash::Device) {
        // SAFETY: the renderer is idle when it is destroyed.
        unsafe {
            for pipeline in self.pipelines {
                device.destroy_pipeline(pipeline, None);
            }
            device.destroy_pipeline_layout(self.layout, None);
        }
        self.log.destroy(device);
        self.probes.destroy(device);
    }
}

impl Renderer {
    /// Record, outside a rendering pass and any render condition, the move
    /// of group counts at `source` (its buffer and offset) into `target`.
    /// A mesh draw updates x and the caller initializes y and z to 1. A
    /// dispatch updates all three axes. Each is bounded by TASK_WORKGROUPS (also
    /// the minimum maxComputeWorkGroupCount), ordered after earlier writes
    /// and before the indirect draw or dispatch that reads them. False
    /// when the device cannot address the buffers (the caller copies the
    /// counts unbounded).
    pub(crate) fn bounded_indirect(
        &mut self,
        source: (vk::Buffer, u64),
        target: (vk::Buffer, u64),
        kind: IndirectCounts,
    ) -> Result<bool, String> {
        let (Some(source_address), Some(target_address)) =
            (self.buffer_address(source.0), self.buffer_address(target.0))
        else {
            return Ok(false);
        };
        if self.mesh_count.is_none() {
            self.mesh_count = Some(MeshCount::new(self)?);
        }
        // ECHORA_DIAG_TASK_GROUPS (a mesh draw's task workgroups) and
        // ECHORA_DIAG_DISPATCH_AXIS / ECHORA_DIAG_DISPATCH_GROUPS (a
        // dispatch's per axis and in all): diagnostic runs only.
        static BOUNDS: std::sync::OnceLock<(u32, u32, u32)> = std::sync::OnceLock::new();
        let (task_bound, axis_bound, product_bound) = *BOUNDS.get_or_init(|| {
            (
                diagnostic_bound("ECHORA_DIAG_TASK_GROUPS", TASK_WORKGROUPS),
                diagnostic_bound("ECHORA_DIAG_DISPATCH_AXIS", TASK_WORKGROUPS),
                diagnostic_bound("ECHORA_DIAG_DISPATCH_GROUPS", u32::MAX),
            )
        });
        let (bound, product) = match kind {
            IndirectCounts::Mesh => (task_bound, 0),
            IndirectCounts::Dispatch => (
                axis_bound,
                if product_bound == u32::MAX {
                    0
                } else {
                    product_bound
                },
            ),
        };
        self.record_bound_program(
            source_address + source.1,
            target_address + target.1,
            kind,
            bound,
            product,
            true,
        )?;
        Ok(true)
    }

    /// ECHORA_DIAG_PROBE (diagnostic): record reads of the u32 words at
    /// these device addresses (as the page table resolves guest
    /// addresses), in command order, into the probe buffer; reported with
    /// the counts.
    pub fn probe_device_words(&mut self, addresses: &[u64]) -> Result<(), String> {
        if self.mesh_count.is_none() {
            self.mesh_count = Some(MeshCount::new(self)?);
        }
        for &address in addresses {
            let mesh_count = self.mesh_count.as_mut().expect("made above");
            if mesh_count.probed >= PROBES {
                break;
            }
            let slot = mesh_count.probes_address + u64::from(mesh_count.probed) * 4;
            mesh_count.probed += 1;
            self.record_bound_program(address, slot, IndirectCounts::Mesh, u32::MAX, 0, false)?;
        }
        Ok(())
    }

    /// The bound program over one record (see `bound_program`), outside a
    /// rendering pass and any condition, ordered after earlier writes and
    /// before indirect reads; `logged` keeps the count in the log.
    fn record_bound_program(
        &mut self,
        source: u64,
        target: u64,
        kind: IndirectCounts,
        bound: u32,
        product: u32,
        logged: bool,
    ) -> Result<(), String> {
        let commands = self.commands()?;
        // A condition left active would gate the program as well.
        self.end_condition();
        let mesh_count = self.mesh_count.as_mut().expect("made above");
        let log = if logged {
            let slot = mesh_count.next % u64::from(LOGGED);
            mesh_count.next += 1;
            mesh_count.log_address + slot * 4
        } else {
            target
        };
        let mut push = [0; PUSH_BYTES];
        push[..8].copy_from_slice(&source.to_le_bytes());
        push[8..16].copy_from_slice(&target.to_le_bytes());
        push[16..24].copy_from_slice(&log.to_le_bytes());
        push[24..28].copy_from_slice(&bound.to_le_bytes());
        push[28..32].copy_from_slice(&product.to_le_bytes());
        let (pipeline, layout) = (mesh_count.pipelines[kind as usize], mesh_count.layout);
        // The count an earlier pass wrote is read by the program.
        let written = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        let bounded = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::INDIRECT_COMMAND_READ);
        // SAFETY: recording, outside rendering; the buffers are live and
        // the addresses inside them.
        unsafe {
            let device = &self.gpu.device;
            *self.stats.entry("stat: barriers in draw").or_default() += 2;
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[written],
                &[],
                &[],
            );
            device.cmd_bind_pipeline(commands, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_push_constants(commands, layout, vk::ShaderStageFlags::COMPUTE, 0, &push);
            device.cmd_dispatch(commands, 1, 1, 1);
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::DRAW_INDIRECT,
                vk::DependencyFlags::empty(),
                &[bounded],
                &[],
                &[],
            );
        }
        if logged {
            self.mesh_count_report();
        }
        Ok(())
    }

    /// ECHORA_MESH_COUNT_LOG (debugging): every 30 indirect mesh draws
    /// (about a forest frame's), the counts read since the last report as
    /// far as the GPU has written them.
    fn mesh_count_report(&mut self) {
        if !logging() {
            return;
        }
        let Some(mesh_count) = self.mesh_count.as_mut() else {
            return;
        };
        if mesh_count.next < mesh_count.reported + 30 {
            return;
        }
        mesh_count.report_pending("periodic");
        mesh_count.reported = mesh_count.next;
    }
}

/// Whether ECHORA_MESH_COUNT_LOG is set.
pub(crate) fn logging() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ECHORA_MESH_COUNT_LOG").is_some())
}

/// Read the guest record's one or three axes, clamp them, and log the
/// highest raw axis. A mesh record contains only X. The host record uses
/// one for Y and Z without reading them from the source.
fn bound_program(kind: IndirectCounts) -> Vec<u32> {
    use spirv::{capability, decoration, execution, glsl, op, storage};
    let mut module = Module::new();
    module.capability(capability::SHADER);
    module.capability(capability::INT64);
    module.capability(capability::PHYSICAL_STORAGE_BUFFER_ADDRESSES);
    module.memory_model(capability::ADDRESSING_PHYSICAL_STORAGE_BUFFER64, 1);
    let glsl_set = module.glsl();
    let void = module.void();
    let bool_type = module.bool();
    let u32_type = module.u32();
    let u64_type = module.u64();
    let block = module.ty(Type::Struct {
        members: vec![u64_type, u64_type, u64_type, u32_type, u32_type],
    });
    module.decorate(block, decoration::BLOCK, &[]);
    for (member, offset) in [0, 8, 16, 24, 28].into_iter().enumerate() {
        module.member_decorate(block, member as u32, decoration::OFFSET, &[offset]);
    }
    let push_pointer = module.pointer(storage::PUSH_CONSTANT, block);
    let push = module.global_variable(push_pointer, storage::PUSH_CONSTANT);
    let push_u64 = module.pointer(storage::PUSH_CONSTANT, u64_type);
    let push_u32 = module.pointer(storage::PUSH_CONSTANT, u32_type);
    let global_u32 = module.pointer(storage::PHYSICAL_STORAGE_BUFFER, u32_type);
    let function_type = module.ty(Type::Function {
        result: void,
        parameters: Vec::new(),
    });
    let members = [0, 1, 2, 3, 4].map(|value| module.constant(u32_type, value));
    let one = module.constant(u32_type, 1);
    let zero = module.constant(u32_type, 0);
    let mut main = Function::new(&mut module, void, function_type);
    let field = |main: &mut Function, module: &mut Module, member: usize, pointer, ty| {
        let at = main.value(module, op::ACCESS_CHAIN, pointer, &[push, members[member]]);
        main.value(module, op::LOAD, ty, &[at])
    };
    let source = field(&mut main, &mut module, 0, push_u64, u64_type);
    let target = field(&mut main, &mut module, 1, push_u64, u64_type);
    let log = field(&mut main, &mut module, 2, push_u64, u64_type);
    let bound = field(&mut main, &mut module, 3, push_u32, u32_type);
    let offsets = [0, 4, 8].map(|offset| module.constant64(u64_type, offset));
    let mut largest = zero;
    let mut values = [one; 3];
    for (index, (value, offset)) in values.iter_mut().zip(offsets).take(kind.axes()).enumerate() {
        let source_at = main.value(&mut module, op::I_ADD, u64_type, &[source, offset]);
        // Memory operand Aligned (2), 4.
        let source_at = main.value(&mut module, op::CONVERT_U_TO_PTR, global_u32, &[source_at]);
        let count = main.value(&mut module, op::LOAD, u32_type, &[source_at, 2, 4]);
        *value = main.value(
            &mut module,
            op::EXT_INST,
            u32_type,
            &[glsl_set, glsl::U_MIN, count, bound],
        );
        largest = if index == 0 {
            count
        } else {
            main.value(
                &mut module,
                op::EXT_INST,
                u32_type,
                &[glsl_set, glsl::U_MAX, largest, count],
            )
        };
    }
    if matches!(kind, IndirectCounts::Dispatch) {
        // Each dispatch axis is at most TASK_WORKGROUPS. Their product
        // fits 64 bits and can exceed 32 bits.
        let product = field(&mut main, &mut module, 4, push_u32, u32_type);
        let [x, y, z] =
            values.map(|axis| main.value(&mut module, op::U_CONVERT, u64_type, &[axis]));
        let xy = main.value(&mut module, op::I_MUL, u64_type, &[x, y]);
        let xyz = main.value(&mut module, op::I_MUL, u64_type, &[xy, z]);
        let wide_product = main.value(&mut module, op::U_CONVERT, u64_type, &[product]);
        let capped = main.value(&mut module, op::I_NOT_EQUAL, bool_type, &[product, zero]);
        let over = main.value(
            &mut module,
            op::U_GREATER_THAN,
            bool_type,
            &[xyz, wide_product],
        );
        let limit = main.value(&mut module, op::LOGICAL_AND, bool_type, &[capped, over]);
        let x_capped = main.value(
            &mut module,
            op::EXT_INST,
            u32_type,
            &[glsl_set, glsl::U_MIN, values[0], product],
        );
        for (value, replacement) in values.iter_mut().zip([x_capped, one, one]) {
            *value = main.value(
                &mut module,
                op::SELECT,
                u32_type,
                &[limit, replacement, *value],
            );
        }
    }
    for (offset, value) in offsets.into_iter().zip(values).take(kind.axes()) {
        let target_at = main.value(&mut module, op::I_ADD, u64_type, &[target, offset]);
        let target_at = main.value(&mut module, op::CONVERT_U_TO_PTR, global_u32, &[target_at]);
        main.emit(op::STORE, &[target_at, value, 2, 4]);
    }
    let log = main.value(&mut module, op::CONVERT_U_TO_PTR, global_u32, &[log]);
    main.emit(op::STORE, &[log, largest, 2, 4]);
    main.emit(op::RETURN, &[]);
    let entry_point = main.id;
    main.finish(&mut module);
    module.entry_point(execution::GL_COMPUTE, entry_point, "main", &[push]);
    module.execution_mode(entry_point, execution::LOCAL_SIZE, &[1, 1, 1]);
    module.finish()
}

#[cfg(test)]
mod tests {
    use super::IndirectCounts;

    #[test]
    fn mesh_count_setup_can_fail_before_creating_pipelines() {
        let Ok(mut gpu) = crate::vulkan::Gpu::new() else {
            return;
        };
        if !gpu.buffer_device_address {
            return;
        }
        gpu.buffer_device_address = false;
        let renderer = super::Renderer::new(gpu).unwrap();
        assert!(super::MeshCount::new(&renderer).is_err());
    }

    /// The bound program run once in the simulator: the three target words
    /// and the logged word, for `source` words, a per-word `bound`, the
    /// number of `words` used and a `product` bound (0: none).
    fn bounded(source: [u32; 3], bound: u32, words: u32, product: u32) -> ([u32; 3], u32) {
        use echora_shader::sim::{Inputs, Machine};
        const SOURCE: u64 = 0x1000;
        const TARGET: u64 = 0x2000;
        const LOG: u64 = 0x3000;
        let kind = if words == 1 {
            IndirectCounts::Mesh
        } else {
            IndirectCounts::Dispatch
        };
        let push: Vec<u8> = [SOURCE, TARGET, LOG]
            .iter()
            .flat_map(|address| address.to_le_bytes())
            .chain([bound, product].iter().flat_map(|word| word.to_le_bytes()))
            .collect();
        let bytes = |words: &[u32]| -> Vec<u8> {
            words.iter().flat_map(|word| word.to_le_bytes()).collect()
        };
        let initial_target = match kind {
            IndirectCounts::Mesh => [0xdead_beef, 1, 1],
            IndirectCounts::Dispatch => [0xdead_beef; 3],
        };
        let memory = [
            (SOURCE, bytes(&source[..kind.axes()])),
            (TARGET, bytes(&initial_target)),
            (LOG, bytes(&[0xdead_beef])),
        ]
        .into_iter()
        .collect();
        let report = Machine::new(&super::bound_program(kind)).simulate(&Inputs {
            workgroup_count: [1, 1, 1],
            push,
            memory,
            ..Inputs::default()
        });
        assert!(report.problems.is_empty(), "{:?}", report.problems);
        let word = |address: u64, index: usize| {
            let region = &report.memory[&address];
            u32::from_le_bytes(region[index * 4..index * 4 + 4].try_into().unwrap())
        };
        (
            [word(TARGET, 0), word(TARGET, 1), word(TARGET, 2)],
            word(LOG, 0),
        )
    }

    #[test]
    fn a_mesh_count_is_bounded_and_its_other_groups_are_one() {
        // One word used: the count as it is below the bound, the bound
        // above it; y and z are 1 even with only four source bytes mapped.
        assert_eq!(bounded([5, 77, 88], 65535, 1, 0), ([5, 1, 1], 5));
        assert_eq!(bounded([0, 77, 88], 65535, 1, 0), ([0, 1, 1], 0));
        assert_eq!(bounded([65535, 0, 0], 65535, 1, 0), ([65535, 1, 1], 65535));
        assert_eq!(
            bounded([0x3f80_0000, 9, 9], 65535, 1, 0),
            ([65535, 1, 1], 0x3f80_0000)
        );
        assert_eq!(bounded([u32::MAX, 0, 0], 64, 1, 0), ([64, 1, 1], u32::MAX));
        // A probe: the word as it is.
        assert_eq!(
            bounded([0x1234_5678, 0, 0], u32::MAX, 1, 0),
            ([0x1234_5678, 1, 1], 0x1234_5678)
        );
    }

    #[test]
    fn a_four_byte_mesh_record_expands_to_vulkan_indirect_arguments() {
        let Ok(gpu) = crate::vulkan::Gpu::new() else {
            return;
        };
        if !gpu.buffer_device_address {
            return;
        }
        let mut renderer = super::Renderer::new(gpu).unwrap();
        let source = renderer
            .host_buffer_with(4, ash::vk::BufferUsageFlags::STORAGE_BUFFER, true)
            .unwrap();
        source.write(0, &73u32.to_le_bytes());
        let initial = [0u32, 1, 1].map(u32::to_le);
        let target = renderer
            .ring_upload(crate::draw::words_as_bytes(&initial), 4)
            .unwrap();
        assert!(
            renderer
                .bounded_indirect((source.buffer, 0), target, IndirectCounts::Mesh)
                .unwrap()
        );
        let token = renderer.defer_ring_read(target.1, 12);
        renderer.flush().unwrap();
        let expected: Vec<_> = [73u32, 1, 1]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        assert_eq!(renderer.take_reads(&[token])[&token], expected);
        source.destroy(renderer.device());
    }

    #[test]
    fn dispatch_groups_are_bounded_per_axis_and_in_all() {
        // Three words used, each bounded; the largest raw word is logged.
        assert_eq!(bounded([120, 68, 1], 65535, 3, 0), ([120, 68, 1], 120));
        assert_eq!(
            bounded([70000, 3, u32::MAX], 65535, 3, 0),
            ([65535, 3, 65535], u32::MAX)
        );
        // The diagnostic's bounds: 64 an axis, 256 in all. Within both, as
        // they are; past the product, x alone and y, z 1.
        assert_eq!(bounded([16, 16, 1], 64, 3, 256), ([16, 16, 1], 16));
        assert_eq!(bounded([16, 16, 2], 64, 3, 256), ([16, 1, 1], 16));
        assert_eq!(
            bounded([u32::MAX, u32::MAX, u32::MAX], 64, 3, 256),
            ([64, 1, 1], u32::MAX)
        );
        assert_eq!(bounded([1000, 1, 1], 64, 3, 16), ([16, 1, 1], 1000));
        assert_eq!(
            bounded([4096, 4096, 256], 65535, 3, 32768),
            ([4096, 1, 1], 4096)
        );
        // No group count is ever more than the product bound.
        for source in [
            [64, 64, 64],
            [3, 100, 7],
            [1, 1, 300],
            [0, 5, 5],
            [64, 4, 1],
        ] {
            let ([x, y, z], _) = bounded(source, 64, 3, 256);
            assert!(
                u64::from(x) * u64::from(y) * u64::from(z) <= 256,
                "{source:?}"
            );
        }
    }

    #[test]
    fn the_bound_programs_are_valid_spirv() {
        for kind in [IndirectCounts::Mesh, IndirectCounts::Dispatch] {
            let words = super::bound_program(kind);
            let path =
                std::env::temp_dir().join(format!("echora-mesh-count-{}.spv", std::process::id()));
            let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
            std::fs::write(&path, bytes).unwrap();
            let output = std::process::Command::new("spirv-val")
                .args(["--target-env", "vulkan1.3"])
                .arg(&path)
                .output();
            let _ = std::fs::remove_file(&path);
            if let Ok(output) = output {
                assert!(
                    output.status.success(),
                    "{}",
                    String::from_utf8_lossy(&output.stderr)
                );
            }
        }
    }
}
