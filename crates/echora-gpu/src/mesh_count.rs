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

pub(crate) struct MeshCount {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    /// Each draw's count as read (u32), by draw number modulo LOGGED.
    log: HostBuffer,
    log_address: u64,
    /// Draws recorded, and the last one the log was read up to.
    next: u64,
    reported: u64,
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
        let range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(32);
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
        for draw in from..self.next {
            let count = counts[(draw % u64::from(LOGGED)) as usize];
            largest = largest.max(count);
            if count > TASK_WORKGROUPS {
                eprintln!(
                    "gpu: indirect draw or dispatch {draw}: count {count} past {TASK_WORKGROUPS}"
                );
            }
        }
        eprintln!(
            "gpu: indirect counts of draws and dispatches {from}..{} ({why}): largest {largest}",
            self.next
        );
    }

    pub(crate) fn destroy(self, device: &ash::Device) {
        // SAFETY: the renderer is idle when it is destroyed.
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
        }
        self.log.destroy(device);
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
        let commands = self.commands()?;
        // A condition left active would gate the program as well.
        self.end_condition();
        let mesh_count = self.mesh_count.as_mut().expect("made above");
        let slot = mesh_count.next % u64::from(LOGGED);
        mesh_count.next += 1;
        let push: Vec<u8> = [
            source_address + source.1,
            target_address + target.1,
            mesh_count.log_address + slot * 4,
        ]
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .chain(TASK_WORKGROUPS.to_le_bytes())
        .chain(words.to_le_bytes())
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
        self.mesh_count_report();
        Ok(true)
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
    let members: Vec<u32> = (0..5)
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
    let offsets: Vec<u32> = (0..3u64)
        .map(|index| module.constant64(u64_type, index * 4))
        .collect();
    let mut largest = zero;
    for index in 0..3u32 {
        let offset = offsets[index as usize];
        let source_at = main.value(&mut module, op::I_ADD, u64_type, &[source, offset]);
        let target_at = main.value(&mut module, op::I_ADD, u64_type, &[target, offset]);
        // Memory operand Aligned (2), 4.
        let source_at = main.value(&mut module, op::CONVERT_U_TO_PTR, global_u32, &[source_at]);
        let count = main.value(&mut module, op::LOAD, u32_type, &[source_at, 2, 4]);
        let used = main.value(
            &mut module,
            op::U_LESS_THAN,
            bool_type,
            &[members[index as usize], words],
        );
        let bounded = main.value(
            &mut module,
            op::EXT_INST,
            u32_type,
            &[glsl_set, glsl::U_MIN, count, bound],
        );
        let value = main.value(&mut module, op::SELECT, u32_type, &[used, bounded, one]);
        let target_at = main.value(&mut module, op::CONVERT_U_TO_PTR, global_u32, &[target_at]);
        main.emit(op::STORE, &[target_at, value, 2, 4]);
        let raw = main.value(&mut module, op::SELECT, u32_type, &[used, count, zero]);
        largest = main.value(
            &mut module,
            op::EXT_INST,
            u32_type,
            &[glsl_set, glsl::U_MAX, largest, raw],
        );
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
