//! An indirect mesh draw's task workgroup count, bounded on the GPU.
//!
//! An NVN indirect mesh draw's record holds one group count, written by
//! earlier GPU work (the culling passes). Vulkan's indirect mesh draw reads
//! three, and a count above maxTaskWorkGroupCount is not a valid one: the
//! count is moved into the draw's argument slot by a one-invocation compute
//! program that bounds it by VK_EXT_mesh_shader's minimum limit (65535),
//! and the count as read is kept in a log for ECHORA_MESH_COUNT_LOG. The
//! game's task programs leave at once past their 4096th workgroup, so a
//! larger count draws nothing more; an unbounded one of billions ran the GPU
//! past the kernel's job limit.

use ash::vk;
use echora_shader::spirv::{self, Function, Module, Type};

use crate::renderer::{HostBuffer, Renderer, vk_error};

/// VK_EXT_mesh_shader's minimum maxTaskWorkGroupCount per dimension.
pub(crate) const TASK_WORKGROUPS: u32 = 65535;

/// Counts kept in the log, the oldest overwritten.
const LOGGED: u32 = 65536;

/// Words ECHORA_DIAG_PROBE reads through the GPU's view (`probe_device_words`).
const PROBES: u32 = 64;

/// A bound from a diagnostic variable (tighter than the device limit, so a
/// diagnostic run cannot flood the GPU through wrong counts), else `limit`.
fn diagnostic_bound(name: &str, limit: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .map_or(limit, |bound| bound.min(limit))
}

pub(crate) struct MeshCount {
    pipeline: vk::Pipeline,
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
        let device = renderer.device();
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let log = renderer.host_buffer_with(u64::from(LOGGED) * 4, usage, true)?;
        log.write(0, &vec![0; LOGGED as usize * 4]);
        let log_address = renderer
            .buffer_address(log.buffer)
            .ok_or("bounded mesh counts need buffer device addresses")?;
        // Three words a probe (the program writes x, y and z).
        let probes = renderer.host_buffer_with(u64::from(PROBES) * 12, usage, true)?;
        probes.write(0, &vec![0; PROBES as usize * 12]);
        let probes_address = renderer
            .buffer_address(probes.buffer)
            .ok_or("bounded mesh counts need buffer device addresses")?;
        let range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(40);
        let ranges = [range];
        let layout_info = vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&ranges);
        // SAFETY: a valid device and create info; destroyed in `destroy`.
        let layout = unsafe { device.create_pipeline_layout(&layout_info, None) }
            .map_err(vk_error("vkCreatePipelineLayout"))?;
        let words = bound_program();
        let module_info = vk::ShaderModuleCreateInfo::default().code(&words);
        // SAFETY: valid SPIR-V; destroyed once the pipeline holds it.
        let module = unsafe { device.create_shader_module(&module_info, None) }
            .map_err(vk_error("vkCreateShaderModule"))?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main");
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout);
        // SAFETY: as above.
        let pipelines = unsafe {
            device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
        };
        // SAFETY: the pipeline (made or not) no longer needs the module.
        unsafe { device.destroy_shader_module(module, None) };
        let pipeline =
            pipelines.map_err(|(_, error)| vk_error("vkCreateComputePipelines")(error))?[0];
        Ok(Self {
            pipeline,
            layout,
            log,
            log_address,
            next: 0,
            reported: 0,
            probes,
            probes_address,
            probed: 0,
        })
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
        for draw in from..self.next {
            let count = counts[(draw % u64::from(LOGGED)) as usize];
            largest = largest.max(count);
            cut += u32::from(count > diagnostic);
            if count > TASK_WORKGROUPS {
                eprintln!(
                    "gpu: indirect draw or dispatch {draw}: count {count} past {TASK_WORKGROUPS}"
                );
            }
        }
        eprintln!(
            "gpu: indirect counts of draws and dispatches {from}..{} ({why}): largest {largest}, {cut} past the bound {diagnostic}",
            self.next
        );
        if self.probed > 0 {
            let words: Vec<String> = self
                .probes
                .read_bytes(self.probed as usize * 12)
                .as_chunks::<12>()
                .0
                .iter()
                .map(|probe| format!("{:#x}", u32::from_le_bytes(probe[..4].try_into().unwrap())))
                .collect();
            eprintln!("gpu: probed words as the GPU read them ({why}): {words:?}");
        }
    }

    pub(crate) fn destroy(self, device: &ash::Device) {
        // SAFETY: the renderer is idle when it is destroyed.
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
        }
        self.log.destroy(device);
        self.probes.destroy(device);
    }
}

impl Renderer {
    /// Record, outside a rendering pass and any render condition, the move
    /// of `words` group counts (1: an NVN mesh draw's one, the target's y
    /// and z set to 1; 3: a dispatch's x, y and z) at `source` (its buffer
    /// and offset) into `target`, each bounded by TASK_WORKGROUPS (also
    /// the minimum maxComputeWorkGroupCount), ordered after earlier writes
    /// and before the indirect draw or dispatch that reads them. False
    /// when the device cannot address the buffers (the caller copies the
    /// counts unbounded).
    pub(crate) fn bounded_indirect(
        &mut self,
        source: (vk::Buffer, u64),
        target: (vk::Buffer, u64),
        words: u32,
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
        let (bound, product) = if words == 1 {
            (task_bound, 0)
        } else {
            (
                axis_bound,
                if product_bound == u32::MAX {
                    0
                } else {
                    product_bound
                },
            )
        };
        self.record_bound_program(
            source_address + source.1,
            target_address + target.1,
            words,
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
            let slot = mesh_count.probes_address + u64::from(mesh_count.probed) * 12;
            mesh_count.probed += 1;
            self.record_bound_program(address, slot, 1, u32::MAX, 0, false)?;
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
        words: u32,
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
        let push: Vec<u8> = [source, target, log]
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .chain(bound.to_le_bytes())
            .chain(words.to_le_bytes())
            .chain(product.to_le_bytes())
            .chain([0; 4])
            .collect();
        let (pipeline, layout) = (mesh_count.pipeline, mesh_count.layout);
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

/// One invocation: for each of the 3 words, `target[i] = i < words ?
/// min(source[i], bound) : 1`, and `*log` the largest source word used
/// (u32). The push constants hold the source, target and log addresses
/// (u64), then the bound and the word count.
fn bound_program() -> Vec<u32> {
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
        members: vec![u64_type, u64_type, u64_type, u32_type, u32_type, u32_type],
    });
    module.decorate(block, decoration::BLOCK, &[]);
    for (member, offset) in [0, 8, 16, 24, 28, 32].into_iter().enumerate() {
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
    let members: Vec<u32> = (0..6)
        .map(|value| module.constant(u32_type, value))
        .collect();
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
    let words = field(&mut main, &mut module, 4, push_u32, u32_type);
    let product = field(&mut main, &mut module, 5, push_u32, u32_type);
    let offsets: Vec<u32> = (0..3u64)
        .map(|index| module.constant64(u64_type, index * 4))
        .collect();
    let mut largest = zero;
    let mut values = Vec::new();
    for index in 0..3usize {
        let source_at = main.value(&mut module, op::I_ADD, u64_type, &[source, offsets[index]]);
        // Memory operand Aligned (2), 4.
        let source_at = main.value(&mut module, op::CONVERT_U_TO_PTR, global_u32, &[source_at]);
        let count = main.value(&mut module, op::LOAD, u32_type, &[source_at, 2, 4]);
        let used = main.value(
            &mut module,
            op::U_LESS_THAN,
            bool_type,
            &[members[index], words],
        );
        let bounded = main.value(
            &mut module,
            op::EXT_INST,
            u32_type,
            &[glsl_set, glsl::U_MIN, count, bound],
        );
        values.push(main.value(&mut module, op::SELECT, u32_type, &[used, bounded, one]));
        let raw = main.value(&mut module, op::SELECT, u32_type, &[used, count, zero]);
        largest = main.value(
            &mut module,
            op::EXT_INST,
            u32_type,
            &[glsl_set, glsl::U_MAX, largest, raw],
        );
    }
    // A product bound (nonzero): past it, x alone (bounded by it) and y, z 1.
    let xy = main.value(&mut module, op::I_MUL, u32_type, &[values[0], values[1]]);
    let xyz = main.value(&mut module, op::I_MUL, u32_type, &[xy, values[2]]);
    let capped = main.value(&mut module, op::I_NOT_EQUAL, bool_type, &[product, zero]);
    let over = main.value(&mut module, op::U_GREATER_THAN, bool_type, &[xyz, product]);
    let limit = main.value(&mut module, op::LOGICAL_AND, bool_type, &[capped, over]);
    let x_capped = main.value(
        &mut module,
        op::EXT_INST,
        u32_type,
        &[glsl_set, glsl::U_MIN, values[0], product],
    );
    values[0] = main.value(
        &mut module,
        op::SELECT,
        u32_type,
        &[limit, x_capped, values[0]],
    );
    values[1] = main.value(&mut module, op::SELECT, u32_type, &[limit, one, values[1]]);
    values[2] = main.value(&mut module, op::SELECT, u32_type, &[limit, one, values[2]]);
    for index in 0..3usize {
        let target_at = main.value(&mut module, op::I_ADD, u64_type, &[target, offsets[index]]);
        let target_at = main.value(&mut module, op::CONVERT_U_TO_PTR, global_u32, &[target_at]);
        main.emit(op::STORE, &[target_at, values[index], 2, 4]);
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
    /// The bound program run once in the simulator: the three target words
    /// and the logged word, for `source` words, a per-word `bound`, the
    /// number of `words` used and a `product` bound (0: none).
    fn bounded(source: [u32; 3], bound: u32, words: u32, product: u32) -> ([u32; 3], u32) {
        use echora_shader::sim::{Inputs, Machine};
        const SOURCE: u64 = 0x1000;
        const TARGET: u64 = 0x2000;
        const LOG: u64 = 0x3000;
        let push: Vec<u8> = [SOURCE, TARGET, LOG]
            .iter()
            .flat_map(|address| address.to_le_bytes())
            .chain(
                [bound, words, product]
                    .iter()
                    .flat_map(|word| word.to_le_bytes()),
            )
            .collect();
        let bytes = |words: &[u32]| -> Vec<u8> {
            words.iter().flat_map(|word| word.to_le_bytes()).collect()
        };
        let memory = [
            (SOURCE, bytes(&source)),
            (TARGET, bytes(&[0xdead_beef; 3])),
            (LOG, bytes(&[0xdead_beef])),
        ]
        .into_iter()
        .collect();
        let report = Machine::new(&super::bound_program()).simulate(&Inputs {
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
        // above it; y and z are 1 whatever follows the count in memory.
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
    fn the_bound_program_is_valid_spirv() {
        let words = super::bound_program();
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
