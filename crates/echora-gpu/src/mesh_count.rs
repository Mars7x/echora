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
    /// Record, outside a rendering pass, the move of the group count at
    /// `source` (its buffer and offset) into `target`'s first word,
    /// bounded by TASK_WORKGROUPS, ordered after earlier writes and before
    /// the indirect draw that reads it. False when the device cannot
    /// address the buffers (the caller copies the count unbounded).
    pub(crate) fn bounded_mesh_count(
        &mut self,
        source: (vk::Buffer, u64),
        target: (vk::Buffer, u64),
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
        self.mesh_count_report();
        Ok(true)
    }

    /// ECHORA_MESH_COUNT_LOG (debugging): every 1024 indirect mesh draws,
    /// the largest count read since the last report and how many were past
    /// the bound (from the log the GPU has written so far).
    fn mesh_count_report(&mut self) {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| std::env::var_os("ECHORA_MESH_COUNT_LOG").is_some()) {
            return;
        }
        let Some(mesh_count) = self.mesh_count.as_mut() else {
            return;
        };
        if mesh_count.next < mesh_count.reported + 1024 {
            return;
        }
        let bytes = mesh_count.log.read_bytes(LOGGED as usize * 4);
        let counts: Vec<u32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        let from = mesh_count
            .reported
            .max(mesh_count.next.saturating_sub(u64::from(LOGGED)));
        let mut largest = 0;
        let mut over = 0;
        for draw in from..mesh_count.next {
            let count = counts[(draw % u64::from(LOGGED)) as usize];
            largest = largest.max(count);
            over += u32::from(count > TASK_WORKGROUPS);
        }
        eprintln!(
            "gpu: indirect mesh counts of draws {from}..{}: largest {largest}, {over} past {TASK_WORKGROUPS}",
            mesh_count.next
        );
        mesh_count.reported = mesh_count.next;
    }
}

/// One invocation: `*target = min(*source, bound); *log = *source` (u32).
/// The push constants hold the source, target and log addresses (u64),
/// then the bound.
fn bound_program() -> Vec<u32> {
    use spirv::{capability, decoration, execution, glsl, op, storage};
    let mut module = Module::new();
    module.capability(capability::SHADER);
    module.capability(capability::INT64);
    module.capability(capability::PHYSICAL_STORAGE_BUFFER_ADDRESSES);
    module.memory_model(capability::ADDRESSING_PHYSICAL_STORAGE_BUFFER64, 1);
    let glsl_set = module.glsl();
    let void = module.void();
    let u32_type = module.u32();
    let u64_type = module.u64();
    let block = module.ty(Type::Struct {
        members: vec![u64_type, u64_type, u64_type, u32_type],
    });
    module.decorate(block, decoration::BLOCK, &[]);
    for member in 0..4 {
        module.member_decorate(block, member, decoration::OFFSET, &[member * 8]);
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
    let members: Vec<u32> = (0..4)
        .map(|value| module.constant(u32_type, value))
        .collect();
    let mut main = Function::new(&mut module, void, function_type);
    let field = |main: &mut Function, module: &mut Module, member: usize, pointer, ty| {
        let at = main.value(module, op::ACCESS_CHAIN, pointer, &[push, members[member]]);
        main.value(module, op::LOAD, ty, &[at])
    };
    let source = field(&mut main, &mut module, 0, push_u64, u64_type);
    let target = field(&mut main, &mut module, 1, push_u64, u64_type);
    let log = field(&mut main, &mut module, 2, push_u64, u64_type);
    let bound = field(&mut main, &mut module, 3, push_u32, u32_type);
    // Memory operand Aligned (2), 4.
    let source = main.value(&mut module, op::CONVERT_U_TO_PTR, global_u32, &[source]);
    let count = main.value(&mut module, op::LOAD, u32_type, &[source, 2, 4]);
    let bounded = main.value(
        &mut module,
        op::EXT_INST,
        u32_type,
        &[glsl_set, glsl::U_MIN, count, bound],
    );
    let target = main.value(&mut module, op::CONVERT_U_TO_PTR, global_u32, &[target]);
    main.emit(op::STORE, &[target, bounded, 2, 4]);
    let log = main.value(&mut module, op::CONVERT_U_TO_PTR, global_u32, &[log]);
    main.emit(op::STORE, &[log, count, 2, 4]);
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
