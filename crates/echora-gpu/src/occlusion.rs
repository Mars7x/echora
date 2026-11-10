//! The guest's SAMPLES_PASSED counter, measured with occlusion queries,
//! and the conditions its reports gate rendering with.
//!
//! NVN counts samples that pass the depth and stencil tests from
//! ResetCounter(SAMPLES_PASSED) on, and ReportCounter(SAMPLES_PASSED,
//! address) writes the count so far (NVNcounterData: the counter, then a
//! timestamp). Here one occlusion query is active outside rendering passes
//! from a reset until the next report or the end of the command buffer
//! (queries cannot span command buffers); each such segment's result is
//! added on the GPU to an accumulator, which a report copies to the
//! guest's memory. Nothing waits for the GPU: a condition on a report
//! reads it with VK_EXT_conditional_rendering.

use ash::vk;
use echora_shader::spirv::{self, Function, Module, Type};

use crate::renderer::{HostBuffer, Renderer, vk_error};

/// Queries each submission slot may use (later segments in a slot reuse
/// the last one, which a slot never needs: a segment ends per report).
const QUERIES: u32 = 4096;

pub(crate) struct Occlusion {
    pool: vk::QueryPool,
    /// The fence of each submission slot: slot `i` owns queries from
    /// `i * QUERIES`; the next free one of each.
    fences: [vk::Fence; 2],
    next: [u32; 2],
    /// The query counting now, in the command buffer being recorded.
    active: Option<u32>,
    /// Counting resumes in the next command buffer (a segment ended at a
    /// submission).
    resume: bool,
    /// The count since the last reset (u64), and each query's result as
    /// copied out for adding (u64 each, by query index).
    accumulator: HostBuffer,
    results: HostBuffer,
    accumulator_address: u64,
    results_address: u64,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl Occlusion {
    pub(crate) fn new(renderer: &Renderer, fences: [vk::Fence; 2]) -> Result<Self, String> {
        let device = renderer.device();
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::OCCLUSION)
            .query_count(2 * QUERIES);
        // SAFETY: a valid create info on a live device; destroyed in `destroy`.
        let pool = unsafe { device.create_query_pool(&info, None) }
            .map_err(vk_error("vkCreateQueryPool"))?;
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        let accumulator = renderer.host_buffer_with(16, usage, true)?;
        accumulator.write(0, &[0; 16]);
        let results = renderer.host_buffer_with(u64::from(2 * QUERIES) * 8, usage, true)?;
        let address = |buffer: &HostBuffer| {
            renderer
                .buffer_address(buffer.buffer)
                .ok_or("occlusion counts need buffer device addresses")
        };
        let (accumulator_address, results_address) = (address(&accumulator)?, address(&results)?);
        let range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(16);
        let ranges = [range];
        let layout_info = vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&ranges);
        // SAFETY: as above.
        let layout = unsafe { device.create_pipeline_layout(&layout_info, None) }
            .map_err(vk_error("vkCreatePipelineLayout"))?;
        let words = add_program();
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
            pool,
            fences,
            next: [0; 2],
            active: None,
            resume: false,
            accumulator,
            results,
            accumulator_address,
            results_address,
            layout,
            pipeline,
        })
    }

    /// The slot of the command buffer recorded under `fence`.
    fn slot(&self, fence: vk::Fence) -> usize {
        usize::from(self.fences[1] == fence)
    }

    /// A slot's queries are free again once its fence has signalled
    /// (before it records anew).
    pub(crate) fn slot_reused(&mut self, fence: vk::Fence) {
        let slot = self.slot(fence);
        self.next[slot] = 0;
    }

    /// Start a segment: reset a query of this slot and begin it (outside
    /// a rendering pass).
    fn begin(&mut self, device: &ash::Device, commands: vk::CommandBuffer, fence: vk::Fence) {
        let slot = self.slot(fence);
        let query = slot as u32 * QUERIES + self.next[slot].min(QUERIES - 1);
        self.next[slot] += 1;
        // SAFETY: recording, outside rendering; the query is this slot's.
        unsafe {
            device.cmd_reset_query_pool(commands, self.pool, query, 1);
            device.cmd_begin_query(commands, self.pool, query, vk::QueryControlFlags::PRECISE);
        }
        self.active = Some(query);
    }

    /// End the active segment (outside a rendering pass) and add its
    /// result to the accumulator on the GPU.
    fn end(&mut self, device: &ash::Device, commands: vk::CommandBuffer) {
        let Some(query) = self.active.take() else {
            return;
        };
        let at = u64::from(query) * 8;
        let to_compute = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
        let after = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::SHADER_READ);
        let push: Vec<u8> = [self.accumulator_address, self.results_address + at]
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        // SAFETY: recording, outside rendering; the query has ended before
        // its result is copied, and the copy waits for it.
        unsafe {
            device.cmd_end_query(commands, self.pool, query);
            device.cmd_copy_query_pool_results(
                commands,
                self.pool,
                query,
                1,
                self.results.buffer,
                at,
                8,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            );
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[to_compute],
                &[],
                &[],
            );
            device.cmd_bind_pipeline(commands, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_push_constants(
                commands,
                self.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &push,
            );
            device.cmd_dispatch(commands, 1, 1, 1);
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[after],
                &[],
                &[],
            );
        }
    }

    pub(crate) fn destroy(self, device: &ash::Device) {
        // SAFETY: the renderer is idle when it is destroyed.
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_query_pool(self.pool, None);
        }
        self.accumulator.destroy(device);
        self.results.destroy(device);
    }
}

impl Renderer {
    fn occlusion(&mut self) -> Result<&mut Occlusion, String> {
        if self.occlusion.is_none() {
            let fences = self.slot_fences();
            self.occlusion = Some(Occlusion::new(self, fences)?);
        }
        Ok(self.occlusion.as_mut().expect("made above"))
    }

    /// ResetCounter(SAMPLES_PASSED): the count restarts from 0 here.
    pub fn samples_reset(&mut self) -> Result<(), String> {
        let commands = self.commands()?;
        let fence = self.current_fence();
        let device = self.gpu.device.clone();
        let occlusion = self.occlusion()?;
        occlusion.end(&device, commands);
        let accumulator = occlusion.accumulator.buffer;
        // SAFETY: recording, outside rendering.
        unsafe { device.cmd_fill_buffer(commands, accumulator, 0, 8, 0) };
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
        // SAFETY: as above.
        unsafe {
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
        occlusion.resume = false;
        occlusion.begin(&device, commands, fence);
        Ok(())
    }

    /// ReportCounter(SAMPLES_PASSED, address): the count so far and
    /// `timestamp`, as NVNcounterData, into `buffer` at `offset` (the
    /// guest memory's device copy). Counting continues.
    pub fn samples_report(
        &mut self,
        buffer: vk::Buffer,
        offset: u64,
        timestamp: u64,
    ) -> Result<(), String> {
        let commands = self.commands()?;
        let fence = self.current_fence();
        let device = self.gpu.device.clone();
        let occlusion = self.occlusion()?;
        let counting = occlusion.active.is_some();
        occlusion.end(&device, commands);
        let copy = vk::BufferCopy {
            src_offset: 0,
            dst_offset: offset,
            size: 8,
        };
        let written = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
        // SAFETY: recording, outside rendering; the range is the buffer's.
        unsafe {
            device.cmd_copy_buffer(commands, occlusion.accumulator.buffer, buffer, &[copy]);
            device.cmd_update_buffer(commands, buffer, offset + 8, &timestamp.to_le_bytes());
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[written],
                &[],
                &[],
            );
        }
        if counting || occlusion.resume {
            occlusion.resume = false;
            occlusion.begin(&device, commands, fence);
        }
        Ok(())
    }

    /// Before a command buffer ends: a segment counting in it ends there,
    /// to resume in the next.
    pub(crate) fn occlusion_before_submit(&mut self, commands: vk::CommandBuffer) {
        let device = self.gpu.device.clone();
        if let Some(occlusion) = self.occlusion.as_mut()
            && occlusion.active.is_some()
        {
            occlusion.end(&device, commands);
            occlusion.resume = true;
        }
    }

    /// When a slot begins recording (its fence signalled): its queries
    /// are free, and a segment interrupted by the last submission resumes.
    pub(crate) fn occlusion_after_begin(&mut self, commands: vk::CommandBuffer) {
        let fence = self.current_fence();
        let device = self.gpu.device.clone();
        if let Some(occlusion) = self.occlusion.as_mut() {
            occlusion.slot_reused(fence);
            if occlusion.resume {
                occlusion.resume = false;
                occlusion.begin(&device, commands, fence);
            }
        }
    }
}

/// `acc[0] += seg[0]` on two u64 addresses from the push constants (acc,
/// then seg), one invocation.
fn add_program() -> Vec<u32> {
    use spirv::{capability, decoration, execution, op, storage};
    let mut module = Module::new();
    module.capability(capability::SHADER);
    module.capability(capability::INT64);
    module.capability(capability::PHYSICAL_STORAGE_BUFFER_ADDRESSES);
    module.memory_model(capability::ADDRESSING_PHYSICAL_STORAGE_BUFFER64, 1);
    let void = module.void();
    let u32_type = module.u32();
    let u64_type = module.u64();
    let block = module.ty(Type::Struct {
        members: vec![u64_type, u64_type],
    });
    module.decorate(block, decoration::BLOCK, &[]);
    module.member_decorate(block, 0, decoration::OFFSET, &[0]);
    module.member_decorate(block, 1, decoration::OFFSET, &[8]);
    let push_pointer = module.pointer(storage::PUSH_CONSTANT, block);
    let push = module.global_variable(push_pointer, storage::PUSH_CONSTANT);
    let member_pointer = module.pointer(storage::PUSH_CONSTANT, u64_type);
    let word_pointer = module.pointer(storage::PHYSICAL_STORAGE_BUFFER, u64_type);
    let function_type = module.ty(Type::Function {
        result: void,
        parameters: Vec::new(),
    });
    let (zero, one) = (module.constant(u32_type, 0), module.constant(u32_type, 1));
    let mut main = Function::new(&mut module, void, function_type);
    let address = |main: &mut Function, module: &mut Module, member: u32| {
        let field = main.value(module, op::ACCESS_CHAIN, member_pointer, &[push, member]);
        let value = main.value(module, op::LOAD, u64_type, &[field]);
        main.value(module, op::CONVERT_U_TO_PTR, word_pointer, &[value])
    };
    let accumulator = address(&mut main, &mut module, zero);
    let segment = address(&mut main, &mut module, one);
    // Memory operand Aligned (2), 8 bytes.
    let total = main.value(&mut module, op::LOAD, u64_type, &[accumulator, 2, 8]);
    let part = main.value(&mut module, op::LOAD, u64_type, &[segment, 2, 8]);
    let sum = main.value(&mut module, op::I_ADD, u64_type, &[total, part]);
    main.emit(op::STORE, &[accumulator, sum, 2, 8]);
    main.emit(op::RETURN, &[]);
    let entry = main.id;
    main.finish(&mut module);
    module.entry_point(execution::GL_COMPUTE, entry, "main", &[push]);
    module.execution_mode(entry, execution::LOCAL_SIZE, &[1, 1, 1]);
    module.finish()
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_add_program_is_valid_spirv() {
        let words = super::add_program();
        let path =
            std::env::temp_dir().join(format!("echora-occlusion-{}.spv", std::process::id()));
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
