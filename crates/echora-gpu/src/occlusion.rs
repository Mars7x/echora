//! The guest's SAMPLES_PASSED counter, measured with occlusion queries,
//! and the conditions its reports gate rendering with.
//!
//! NVN counts samples that pass the depth and stencil tests from
//! ResetCounter(SAMPLES_PASSED) on, and ReportCounter(SAMPLES_PASSED,
//! address) writes the count so far (NVNcounterData: the counter, then a
//! timestamp). Here one occlusion query is active outside rendering passes
//! from a reset until the next report or the end of the command buffer
//! (queries cannot span command buffers): each report ends a segment and
//! begins the next. Reading a query's result waits for the work before it,
//! so reports are not resolved one by one: a batch of them is resolved at
//! once, before a condition is begun and before a submission (when the
//! counts can next be read), by one copy of the batch's query results and
//! one dispatch that sums each report's segments, then a copy of each sum
//! to the guest's memory. Nothing waits for the GPU on the host: a
//! condition on a report reads it with VK_EXT_conditional_rendering.

use ash::vk;
use echora_shader::spirv::{self, Function, Module, Type};

use crate::renderer::{HostBuffer, Renderer, vk_error};

/// Queries each submission slot may use (a slot resolves its batch and
/// starts its queries over when it runs out).
const QUERIES: u32 = 4096;

/// A report waiting to be resolved: its segments (queries `from..to` of
/// the slot), whether the count before the batch carries into it (no reset
/// since), and where it goes.
struct Pending {
    from: u32,
    to: u32,
    carry: bool,
    buffer: vk::Buffer,
    offset: u64,
}

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
    /// The batch: its first query, the first query since a reset in it (if
    /// one was), its reports, and the end of its ended segments.
    batch_from: u32,
    reset_from: Option<u32>,
    pending: Vec<Pending>,
    ended: u32,
    /// Each slot's plan entries and sums used so far (both reset with the
    /// slot's queries): a resolve's plan and sums are written past them.
    plan_next: [u32; 2],
    /// The count since the last reset as of the last resolve (u64); each
    /// query's result as copied out (u64 each, by query index); the
    /// resolve plans (four u32 per report: from, to, carry, unused); and
    /// the sums (u64 each, by plan entry).
    accumulator: HostBuffer,
    results: HostBuffer,
    plans: HostBuffer,
    sums: HostBuffer,
    accumulator_address: u64,
    results_address: u64,
    plans_address: u64,
    sums_address: u64,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    /// PRECISE where the device enables occlusionQueryPrecise (counts, as
    /// SAMPLES_PASSED reports them); else none.
    control: vk::QueryControlFlags,
}

/// Plan entries each slot may use: every report and one accumulator entry
/// per resolve.
const PLANS: u32 = 2 * QUERIES;

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
        let plans = renderer.host_buffer_with(u64::from(2 * PLANS) * 16, usage, true)?;
        let sums = renderer.host_buffer_with(u64::from(2 * PLANS) * 8, usage, true)?;
        let address = |buffer: &HostBuffer| {
            renderer
                .buffer_address(buffer.buffer)
                .ok_or("occlusion counts need buffer device addresses")
        };
        let range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(40);
        let ranges = [range];
        let layout_info = vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&ranges);
        // SAFETY: as above.
        let layout = unsafe { device.create_pipeline_layout(&layout_info, None) }
            .map_err(vk_error("vkCreatePipelineLayout"))?;
        let words = sum_program();
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
            batch_from: 0,
            reset_from: None,
            pending: Vec::new(),
            ended: 0,
            plan_next: [0; 2],
            accumulator_address: address(&accumulator)?,
            results_address: address(&results)?,
            plans_address: address(&plans)?,
            sums_address: address(&sums)?,
            accumulator,
            results,
            plans,
            sums,
            layout,
            pipeline,
            control: if renderer.gpu.occlusion_query_precise {
                vk::QueryControlFlags::PRECISE
            } else {
                vk::QueryControlFlags::empty()
            },
        })
    }

    /// The slot of the command buffer recorded under `fence`.
    fn slot(&self, fence: vk::Fence) -> usize {
        usize::from(self.fences[1] == fence)
    }

    /// A slot's queries are free again once its fence has signalled
    /// (before it records anew; the last batch was resolved before its
    /// submission).
    pub(crate) fn slot_reused(&mut self, fence: vk::Fence) {
        let slot = self.slot(fence);
        // ECHORA_OCCLUSION_LOG (debugging): the slot's finished work, its
        // segments' sample counts and its reports' sums (zero and not).
        if std::env::var_os("ECHORA_OCCLUSION_LOG").is_some() && self.next[slot] > 0 {
            let words = |buffer: &HostBuffer, from: u64, count: u64| -> Vec<u64> {
                buffer
                    .read_bytes(((from + count) * 8) as usize)
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .skip(from as usize)
                    .map(|word| u64::from_le_bytes(*word))
                    .collect()
            };
            let segments = words(
                &self.results,
                u64::from(slot as u32 * QUERIES),
                u64::from(self.next[slot]),
            );
            let sums = words(
                &self.sums,
                u64::from(slot as u32 * PLANS),
                u64::from(self.plan_next[slot]),
            );
            let zero = |values: &[u64]| values.iter().filter(|value| **value == 0).count();
            eprintln!(
                "occlusion slot {slot}: {} segments ({} zero) {:?} {} sums ({} zero) {:?}",
                segments.len(),
                zero(&segments),
                &segments[..segments.len().min(12)],
                sums.len(),
                zero(&sums),
                &sums[..sums.len().min(12)]
            );
        }
        self.next[slot] = 0;
        self.plan_next[slot] = 0;
        self.batch_from = slot as u32 * QUERIES;
        self.ended = self.batch_from;
        self.reset_from = self.reset_from.map(|_| self.batch_from);
    }

    /// Whether a segment can begin without resolving first (the slot has
    /// a query left, and plan entries for the batch).
    fn has_room(&self, fence: vk::Fence) -> bool {
        let slot = self.slot(fence);
        self.next[slot] < QUERIES && self.plan_next[slot] + self.pending.len() as u32 + 2 < PLANS
    }

    /// Start a segment: reset a query of this slot and begin it (outside
    /// a rendering pass; the slot has room).
    fn begin(&mut self, device: &ash::Device, commands: vk::CommandBuffer, fence: vk::Fence) {
        let slot = self.slot(fence);
        let query = slot as u32 * QUERIES + self.next[slot];
        self.next[slot] += 1;
        // SAFETY: recording, outside rendering; the query is this slot's
        // and unused since the slot's queries were last freed.
        unsafe {
            device.cmd_reset_query_pool(commands, self.pool, query, 1);
            device.cmd_begin_query(commands, self.pool, query, self.control);
        }
        self.active = Some(query);
    }

    /// End the active segment (outside a rendering pass); it joins the
    /// batch.
    fn end(&mut self, device: &ash::Device, commands: vk::CommandBuffer) {
        if let Some(query) = self.active.take() {
            // SAFETY: recording, outside rendering; the query is active.
            unsafe { device.cmd_end_query(commands, self.pool, query) };
            self.ended = query + 1;
        }
    }

    /// Resolve the batch (outside a rendering pass and any condition; no
    /// segment active): copy its query results, sum each report's
    /// segments and the count so far, and write the reports' counts.
    fn resolve(&mut self, device: &ash::Device, commands: vk::CommandBuffer, fence: vk::Fence) {
        if self.pending.is_empty() && self.reset_from.is_none() && self.ended == self.batch_from {
            return;
        }
        let slot = self.slot(fence);
        // The plan: each report, then the count so far (into the
        // accumulator).
        let mut plan: Vec<[u32; 4]> = self
            .pending
            .iter()
            .map(|report| [report.from, report.to, u32::from(report.carry), 0])
            .collect();
        plan.push([
            self.reset_from.unwrap_or(self.batch_from),
            self.ended,
            u32::from(self.reset_from.is_none()),
            0,
        ]);
        let first = slot as u32 * PLANS + self.plan_next[slot];
        self.plan_next[slot] += plan.len() as u32;
        let bytes: Vec<u8> = plan
            .iter()
            .flatten()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        self.plans.write(first as usize * 16, &bytes);
        let queries = self.ended - self.batch_from;
        let push: Vec<u8> = [
            self.plans_address + u64::from(first) * 16,
            self.results_address,
            self.sums_address + u64::from(first) * 8,
            self.accumulator_address,
        ]
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .chain((plan.len() as u32).to_le_bytes())
        .chain([0; 4])
        .collect();
        let to_compute = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        let to_copy = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
        let written = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(
                vk::AccessFlags::MEMORY_READ
                    | vk::AccessFlags::MEMORY_WRITE
                    | vk::AccessFlags::SHADER_READ,
            );
        let sum_at = |index: u32| u64::from(first + index) * 8;
        // SAFETY: recording, outside rendering and conditions; the batch's
        // queries have ended, and the copy waits for their results.
        unsafe {
            if queries > 0 {
                device.cmd_copy_query_pool_results(
                    commands,
                    self.pool,
                    self.batch_from,
                    queries,
                    self.results.buffer,
                    u64::from(self.batch_from) * 8,
                    8,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                );
            }
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
            device.cmd_dispatch(commands, (plan.len() as u32).div_ceil(64), 1, 1);
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[to_copy],
                &[],
                &[],
            );
            for (index, report) in self.pending.iter().enumerate() {
                let copy = vk::BufferCopy {
                    src_offset: sum_at(index as u32),
                    dst_offset: report.offset,
                    size: 8,
                };
                device.cmd_copy_buffer(commands, self.sums.buffer, report.buffer, &[copy]);
            }
            let total = vk::BufferCopy {
                src_offset: sum_at(plan.len() as u32 - 1),
                dst_offset: 0,
                size: 8,
            };
            device.cmd_copy_buffer(
                commands,
                self.sums.buffer,
                self.accumulator.buffer,
                &[total],
            );
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
        self.pending.clear();
        self.reset_from = None;
        self.batch_from = self.ended;
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
        self.plans.destroy(device);
        self.sums.destroy(device);
    }
}

impl Renderer {
    fn occlusion(&mut self) -> Result<&mut Occlusion, String> {
        if self.occlusion.is_none() {
            let fences = self.slot_fences();
            let mut occlusion = Occlusion::new(self, fences)?;
            // Its batch starts in the current slot.
            occlusion.slot_reused(self.current_fence());
            self.occlusion = Some(occlusion);
        }
        Ok(self.occlusion.as_mut().expect("made above"))
    }

    /// End the active segment and begin the next, resolving the batch
    /// first when the slot has no room for it.
    fn next_segment(&mut self, begin: bool) -> Result<(), String> {
        let commands = self.commands()?;
        let fence = self.current_fence();
        let device = self.gpu.device.clone();
        let occlusion = self.occlusion()?;
        occlusion.end(&device, commands);
        if begin && !occlusion.has_room(fence) {
            self.end_condition();
            let occlusion = self.occlusion.as_mut().expect("made above");
            occlusion.resolve(&device, commands, fence);
            // The slot's queries are all resolved: they start over (each
            // is reset before it begins, after the copy in command order).
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            // SAFETY: recording, outside rendering.
            unsafe {
                device.cmd_pipeline_barrier(
                    commands,
                    vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[barrier],
                    &[],
                    &[],
                )
            };
            occlusion.slot_reused(fence);
        }
        if begin {
            let occlusion = self.occlusion.as_mut().expect("made above");
            occlusion.resume = false;
            occlusion.begin(&device, commands, fence);
        }
        Ok(())
    }

    /// ResetCounter(SAMPLES_PASSED): the count restarts from 0 here.
    pub fn samples_reset(&mut self) -> Result<(), String> {
        self.next_segment(false)?;
        let occlusion = self.occlusion.as_mut().expect("made by next_segment");
        occlusion.reset_from = Some(occlusion.ended);
        self.next_segment(true)?;
        let occlusion = self.occlusion.as_mut().expect("made by next_segment");
        occlusion.reset_from = occlusion.active;
        Ok(())
    }

    /// ReportCounter(SAMPLES_PASSED, address): the count so far and
    /// `timestamp`, as NVNcounterData, into `buffer` at `offset` (the
    /// guest memory's device copy; the count once the batch resolves).
    /// Counting continues.
    pub fn samples_report(
        &mut self,
        buffer: vk::Buffer,
        offset: u64,
        timestamp: u64,
    ) -> Result<(), String> {
        let counting = self.occlusion()?.active.is_some();
        let resume = self.occlusion()?.resume;
        self.next_segment(false)?;
        let commands = self.commands()?;
        let occlusion = self.occlusion.as_mut().expect("made by next_segment");
        occlusion.pending.push(Pending {
            from: occlusion.reset_from.unwrap_or(occlusion.batch_from),
            to: occlusion.ended,
            carry: occlusion.reset_from.is_none(),
            buffer,
            offset,
        });
        let written = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
        // SAFETY: recording, outside rendering; the range is the buffer's.
        unsafe {
            let device = &self.gpu.device;
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
        if counting || resume {
            self.next_segment(true)?;
        }
        Ok(())
    }

    /// Write the batch's reports (outside a rendering pass, before a
    /// condition begins): counting goes on in a new segment.
    pub(crate) fn occlusion_resolve(&mut self, commands: vk::CommandBuffer) {
        let fence = self.current_fence();
        let device = self.gpu.device.clone();
        if let Some(occlusion) = self.occlusion.as_mut()
            && !occlusion.pending.is_empty()
        {
            let counting = occlusion.active.is_some();
            occlusion.end(&device, commands);
            occlusion.resolve(&device, commands, fence);
            if counting && occlusion.has_room(fence) {
                occlusion.begin(&device, commands, fence);
            } else if counting {
                occlusion.resume = true;
            }
        }
    }

    /// Before a command buffer ends (outside rendering and conditions): a
    /// segment counting in it ends there, to resume in the next, and the
    /// batch is resolved.
    pub(crate) fn occlusion_before_submit(&mut self, commands: vk::CommandBuffer) {
        let fence = self.current_fence();
        let device = self.gpu.device.clone();
        if let Some(occlusion) = self.occlusion.as_mut() {
            if occlusion.active.is_some() {
                occlusion.end(&device, commands);
                occlusion.resume = true;
            }
            occlusion.resolve(&device, commands, fence);
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

/// For each of `count` plan entries (four u32: from, to, carry, unused),
/// one invocation each: `sums[i] = (carry ? accumulator : 0) +
/// results[from] + ... + results[to - 1]` (u64). The push constants hold
/// the plan, results, sums and accumulator addresses (u64), then count.
fn sum_program() -> Vec<u32> {
    use spirv::{capability, decoration, execution, op, storage};
    let mut module = Module::new();
    module.capability(capability::SHADER);
    module.capability(capability::INT64);
    module.capability(capability::PHYSICAL_STORAGE_BUFFER_ADDRESSES);
    module.memory_model(capability::ADDRESSING_PHYSICAL_STORAGE_BUFFER64, 1);
    let void = module.void();
    let bool_type = module.bool();
    let u32_type = module.u32();
    let u64_type = module.u64();
    let block = module.ty(Type::Struct {
        members: vec![u64_type, u64_type, u64_type, u64_type, u32_type],
    });
    module.decorate(block, decoration::BLOCK, &[]);
    for member in 0..5 {
        module.member_decorate(block, member, decoration::OFFSET, &[member * 8]);
    }
    let push_pointer = module.pointer(storage::PUSH_CONSTANT, block);
    let push = module.global_variable(push_pointer, storage::PUSH_CONSTANT);
    let push_u64 = module.pointer(storage::PUSH_CONSTANT, u64_type);
    let push_u32 = module.pointer(storage::PUSH_CONSTANT, u32_type);
    let global_u64 = module.pointer(storage::PHYSICAL_STORAGE_BUFFER, u64_type);
    let global_u32 = module.pointer(storage::PHYSICAL_STORAGE_BUFFER, u32_type);
    let vector = module.ty(Type::Vector {
        component: u32_type,
        count: 3,
    });
    let input_vector = module.pointer(storage::INPUT, vector);
    let invocation = module.global_variable(input_vector, storage::INPUT);
    // BuiltIn GlobalInvocationId (28).
    module.decorate(invocation, decoration::BUILT_IN, &[28]);
    let function_type = module.ty(Type::Function {
        result: void,
        parameters: Vec::new(),
    });
    let constants: Vec<u32> = (0..5)
        .map(|value| module.constant(u32_type, value))
        .collect();
    let zero64 = module.constant64(u64_type, 0);
    let (four64, eight64, sixteen64) = (
        module.constant64(u64_type, 4),
        module.constant64(u64_type, 8),
        module.constant64(u64_type, 16),
    );
    let function_u32 = module.pointer(storage::FUNCTION, u32_type);
    let function_u64 = module.pointer(storage::FUNCTION, u64_type);
    let mut main = Function::new(&mut module, void, function_type);
    let query = main.variable(&mut module, function_u32, None);
    let sum = main.variable(&mut module, function_u64, None);
    let field = |main: &mut Function, module: &mut Module, member: usize, pointer, ty| {
        let at = main.value(
            module,
            op::ACCESS_CHAIN,
            pointer,
            &[push, constants[member]],
        );
        main.value(module, op::LOAD, ty, &[at])
    };
    let plans = field(&mut main, &mut module, 0, push_u64, u64_type);
    let results = field(&mut main, &mut module, 1, push_u64, u64_type);
    let sums = field(&mut main, &mut module, 2, push_u64, u64_type);
    let accumulator = field(&mut main, &mut module, 3, push_u64, u64_type);
    let count = field(&mut main, &mut module, 4, push_u32, u32_type);
    let ids = main.value(&mut module, op::LOAD, vector, &[invocation]);
    let index = main.value(&mut module, op::COMPOSITE_EXTRACT, u32_type, &[ids, 0]);
    let inside = main.value(&mut module, op::U_LESS_THAN, bool_type, &[index, count]);
    let (body, end) = (module.id(), module.id());
    main.emit(op::SELECTION_MERGE, &[end, 0]);
    main.emit(op::BRANCH_CONDITIONAL, &[inside, body, end]);
    main.label(body);
    // Memory operand Aligned (2) with the alignment.
    let load = |main: &mut Function, module: &mut Module, address, pointer, ty, alignment| {
        let pointer = main.value(module, op::CONVERT_U_TO_PTR, pointer, &[address]);
        main.value(module, op::LOAD, ty, &[pointer, 2, alignment])
    };
    let index64 = main.value(&mut module, op::U_CONVERT, u64_type, &[index]);
    let entry_offset = main.value(&mut module, op::I_MUL, u64_type, &[index64, sixteen64]);
    let entry = main.value(&mut module, op::I_ADD, u64_type, &[plans, entry_offset]);
    let from = load(&mut main, &mut module, entry, global_u32, u32_type, 4);
    let to_at = main.value(&mut module, op::I_ADD, u64_type, &[entry, four64]);
    let to = load(&mut main, &mut module, to_at, global_u32, u32_type, 4);
    let carry_at = main.value(&mut module, op::I_ADD, u64_type, &[entry, eight64]);
    let carry = load(&mut main, &mut module, carry_at, global_u32, u32_type, 4);
    let before = load(&mut main, &mut module, accumulator, global_u64, u64_type, 8);
    let carried = main.value(
        &mut module,
        op::I_NOT_EQUAL,
        bool_type,
        &[carry, constants[0]],
    );
    let initial = main.value(
        &mut module,
        op::SELECT,
        u64_type,
        &[carried, before, zero64],
    );
    main.emit(op::STORE, &[query, from]);
    main.emit(op::STORE, &[sum, initial]);
    let (header, step, next, merge) = (module.id(), module.id(), module.id(), module.id());
    main.emit(op::BRANCH, &[header]);
    main.label(header);
    let current = main.value(&mut module, op::LOAD, u32_type, &[query]);
    let more = main.value(&mut module, op::U_LESS_THAN, bool_type, &[current, to]);
    main.emit(op::LOOP_MERGE, &[merge, next, 0]);
    main.emit(op::BRANCH_CONDITIONAL, &[more, step, merge]);
    main.label(step);
    let current64 = main.value(&mut module, op::U_CONVERT, u64_type, &[current]);
    let result_offset = main.value(&mut module, op::I_MUL, u64_type, &[current64, eight64]);
    let result_at = main.value(&mut module, op::I_ADD, u64_type, &[results, result_offset]);
    let result = load(&mut main, &mut module, result_at, global_u64, u64_type, 8);
    let total = main.value(&mut module, op::LOAD, u64_type, &[sum]);
    let total = main.value(&mut module, op::I_ADD, u64_type, &[total, result]);
    main.emit(op::STORE, &[sum, total]);
    let following = main.value(&mut module, op::I_ADD, u32_type, &[current, constants[1]]);
    main.emit(op::STORE, &[query, following]);
    main.emit(op::BRANCH, &[next]);
    main.label(next);
    main.emit(op::BRANCH, &[header]);
    main.label(merge);
    let out_offset = main.value(&mut module, op::I_MUL, u64_type, &[index64, eight64]);
    let out_at = main.value(&mut module, op::I_ADD, u64_type, &[sums, out_offset]);
    let out = main.value(&mut module, op::CONVERT_U_TO_PTR, global_u64, &[out_at]);
    let total = main.value(&mut module, op::LOAD, u64_type, &[sum]);
    main.emit(op::STORE, &[out, total, 2, 8]);
    main.emit(op::BRANCH, &[end]);
    main.label(end);
    main.emit(op::RETURN, &[]);
    let entry_point = main.id;
    main.finish(&mut module);
    module.entry_point(
        execution::GL_COMPUTE,
        entry_point,
        "main",
        &[push, invocation],
    );
    module.execution_mode(entry_point, execution::LOCAL_SIZE, &[64, 1, 1]);
    module.finish()
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_sum_program_is_valid_spirv() {
        let words = super::sum_program();
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
