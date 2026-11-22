//! A render condition on a 64-bit word that GPU work is still writing,
//! evaluated on the GPU in command order.
//!
//! NVN's SetRenderEnableConditional compares two 64-bit counters (rendering
//! holds while they are equal, mode 0, or differ, mode 1). Vulkan's
//! conditional rendering reads one 32-bit word. When the first counter is
//! written by earlier GPU work (a culling pass), a one-invocation compute
//! program compares it, after that work, with the second (a value the CPU
//! knows, or another word GPU work writes) and stores the outcome (1:
//! render) where the condition reads it, so the CPU need not wait for the
//! work to finish to compare them itself.

use ash::vk;
use echora_shader::spirv::{self, Function, Module, Type};

use crate::renderer::{HostBuffer, Renderer, vk_error};

/// Outcome words kept, the oldest reused (a few dozen are made a frame;
/// at most two submissions are in flight).
const SLOTS: u64 = 1 << 14;
const PUSH_BYTES: usize = 32;

/// What a word is compared with.
pub enum Comparand {
    /// A value the CPU knows.
    Value(u64),
    /// Another word GPU work writes (a buffer and offset).
    Word(vk::Buffer, u64),
}

pub(crate) struct WordConditions {
    /// Against a pushed value, and against a second word.
    pipelines: [vk::Pipeline; 2],
    layout: vk::PipelineLayout,
    outcomes: HostBuffer,
    outcomes_address: u64,
    next: u64,
}

impl WordConditions {
    fn new(renderer: &Renderer) -> Result<Self, String> {
        let device = renderer.device();
        let outcomes = renderer.host_buffer_with(
            SLOTS * 4,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::CONDITIONAL_RENDERING_EXT,
            true,
        )?;
        let outcomes_address = renderer
            .buffer_address(outcomes.buffer)
            .ok_or("word conditions need buffer device addresses")?;
        let range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(PUSH_BYTES as u32);
        let ranges = [range];
        let layout_info = vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&ranges);
        // SAFETY: a valid device and create info; destroyed in `destroy`.
        let layout = unsafe { device.create_pipeline_layout(&layout_info, None) }
            .map_err(vk_error("vkCreatePipelineLayout"))?;
        let mut pipelines = [vk::Pipeline::null(); 2];
        for (index, second_word) in [false, true].into_iter().enumerate() {
            match create_pipeline(device, layout, second_word) {
                Ok(pipeline) => pipelines[index] = pipeline,
                Err(error) => {
                    // SAFETY: none has been used.
                    unsafe {
                        for pipeline in pipelines {
                            device.destroy_pipeline(pipeline, None);
                        }
                        device.destroy_pipeline_layout(layout, None);
                    }
                    outcomes.destroy(device);
                    return Err(error);
                }
            }
        }
        Ok(Self {
            pipelines,
            layout,
            outcomes,
            outcomes_address,
            next: 0,
        })
    }

    pub(crate) fn destroy(self, device: &ash::Device) {
        // SAFETY: the renderer is idle when it is destroyed.
        unsafe {
            for pipeline in self.pipelines {
                device.destroy_pipeline(pipeline, None);
            }
            device.destroy_pipeline_layout(self.layout, None);
        }
        self.outcomes.destroy(device);
    }
}

fn create_pipeline(
    device: &ash::Device,
    layout: vk::PipelineLayout,
    second_word: bool,
) -> Result<vk::Pipeline, String> {
    let words = compare_program(second_word);
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
    Ok(pipelines.map_err(|(_, error)| vk_error("vkCreateComputePipelines")(error))?[0])
}

impl Renderer {
    /// How many word conditions have been compared (`condition_on_word`).
    /// The outcome of the one made when this was `made` stays in its slot
    /// until `SLOTS` later comparisons reuse it (`word_outcome_held`).
    pub fn word_conditions_made(&self) -> u64 {
        self.word_conditions
            .as_ref()
            .map_or(0, |conditions| conditions.next)
    }

    /// Whether the outcome of the word condition compared when
    /// `word_conditions_made` was `made` (before it) is still in its slot.
    pub fn word_outcome_held(&self, made: u64) -> bool {
        self.word_conditions_made()
            .checked_sub(made)
            .is_some_and(|later| (1..=SLOTS).contains(&later))
    }

    /// The word a render condition reads to hold while the 64-bit word at
    /// `word` (a buffer and offset GPU work writes) equals `other`
    /// (`render_when_equal`) or differs from it: compared on the GPU after
    /// the work recorded so far, outside a rendering pass. None where the
    /// device cannot (no conditional rendering or buffer addresses).
    pub fn condition_on_word(
        &mut self,
        word: (vk::Buffer, u64),
        other: Comparand,
        render_when_equal: bool,
    ) -> Result<Option<(vk::Buffer, u64)>, String> {
        if !self.supports_conditions() {
            return Ok(None);
        }
        let Some(word_address) = self.buffer_address(word.0) else {
            return Ok(None);
        };
        // The comparand pushed: a value, or the second word's address.
        let (second_word, comparand) = match other {
            Comparand::Value(value) => (false, value),
            Comparand::Word(buffer, offset) => match self.buffer_address(buffer) {
                Some(address) => (true, address + offset),
                None => return Ok(None),
            },
        };
        if self.word_conditions.is_none() {
            self.word_conditions = Some(WordConditions::new(self)?);
        }
        let commands = self.commands()?;
        // A condition left active would gate the comparison as well.
        self.end_condition();
        let conditions = self.word_conditions.as_mut().expect("made above");
        let slot = conditions.next % SLOTS;
        conditions.next += 1;
        let outcome = (conditions.outcomes.buffer, slot * 4);
        let mut push = [0; PUSH_BYTES];
        push[..8].copy_from_slice(&(word_address + word.1).to_le_bytes());
        push[8..16].copy_from_slice(&(conditions.outcomes_address + slot * 4).to_le_bytes());
        push[16..24].copy_from_slice(&comparand.to_le_bytes());
        push[24..28].copy_from_slice(&u32::from(render_when_equal).to_le_bytes());
        let (pipeline, layout) = (
            conditions.pipelines[usize::from(second_word)],
            conditions.layout,
        );
        let written = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        let compared = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::CONDITIONAL_RENDERING_READ_EXT);
        // SAFETY: recording, outside rendering; the buffers are live and
        // the addresses inside them.
        unsafe {
            let device = &self.gpu.device;
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
                vk::PipelineStageFlags::CONDITIONAL_RENDERING_EXT,
                vk::DependencyFlags::empty(),
                &[compared],
                &[],
                &[],
            );
        }
        *self.stats.entry("stat: barriers in draw").or_default() += 2;
        *self
            .stats
            .entry("stat: conditions compared on the GPU")
            .or_default() += 1;
        Ok(Some(outcome))
    }
}

/// Load the 64-bit word at the first address (as two 4-byte-aligned
/// halves), compare it with the pushed value (or, with `second_word`, the
/// word at the pushed address) and store 1 at the second address when
/// rendering holds (equal == wanted), 0 otherwise.
fn compare_program(second_word: bool) -> Vec<u32> {
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
    // Word address, outcome address, then the value's low and high halves
    // (or, with `second_word`, the second word's address: a 64-bit member,
    // as loading one through a 32-bit member is invalid and drivers then
    // read only its low half), render when equal, padding.
    let (members, offsets, when_equal_member): (Vec<_>, &[u32], usize) = if second_word {
        (
            vec![u64_type, u64_type, u64_type, u32_type, u32_type],
            &[0, 8, 16, 24, 28],
            3,
        )
    } else {
        (
            vec![u64_type, u64_type, u32_type, u32_type, u32_type, u32_type],
            &[0, 8, 16, 20, 24, 28],
            4,
        )
    };
    let block = module.ty(Type::Struct { members });
    module.decorate(block, decoration::BLOCK, &[]);
    for (member, &offset) in offsets.iter().enumerate() {
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
    let four = module.constant64(u64_type, 4);
    let mut main = Function::new(&mut module, void, function_type);
    let field = |main: &mut Function, module: &mut Module, member: usize, pointer, ty| {
        let at = main.value(module, op::ACCESS_CHAIN, pointer, &[push, members[member]]);
        main.value(module, op::LOAD, ty, &[at])
    };
    let word = field(&mut main, &mut module, 0, push_u64, u64_type);
    let outcome = field(&mut main, &mut module, 1, push_u64, u64_type);
    let when_equal = field(
        &mut main,
        &mut module,
        when_equal_member,
        push_u32,
        u32_type,
    );
    // A 64-bit word's halves at `at` (memory operand Aligned (2), 4).
    let halves = |main: &mut Function, module: &mut Module, at| {
        let high_at = main.value(module, op::I_ADD, u64_type, &[at, four]);
        let low_pointer = main.value(module, op::CONVERT_U_TO_PTR, global_u32, &[at]);
        let high_pointer = main.value(module, op::CONVERT_U_TO_PTR, global_u32, &[high_at]);
        [low_pointer, high_pointer]
            .map(|pointer| main.value(module, op::LOAD, u32_type, &[pointer, 2, 4]))
    };
    let [value_low, value_high] = if second_word {
        let second = field(&mut main, &mut module, 2, push_u64, u64_type);
        halves(&mut main, &mut module, second)
    } else {
        [2, 3].map(|member| field(&mut main, &mut module, member, push_u32, u32_type))
    };
    let [low, high] = halves(&mut main, &mut module, word);
    let low_equal = main.value(&mut module, op::I_EQUAL, bool_type, &[low, value_low]);
    let high_equal = main.value(&mut module, op::I_EQUAL, bool_type, &[high, value_high]);
    let equal = main.value(
        &mut module,
        op::LOGICAL_AND,
        bool_type,
        &[low_equal, high_equal],
    );
    let wanted = main.value(&mut module, op::I_NOT_EQUAL, bool_type, &[when_equal, zero]);
    let renders = main.value(&mut module, op::LOGICAL_EQUAL, bool_type, &[equal, wanted]);
    let stored = main.value(&mut module, op::SELECT, u32_type, &[renders, one, zero]);
    let outcome = main.value(&mut module, op::CONVERT_U_TO_PTR, global_u32, &[outcome]);
    main.emit(op::STORE, &[outcome, stored, 2, 4]);
    main.emit(op::RETURN, &[]);
    let entry_point = main.id;
    main.finish(&mut module);
    module.entry_point(execution::GL_COMPUTE, entry_point, "main", &[push]);
    module.execution_mode(entry_point, execution::LOCAL_SIZE, &[1, 1, 1]);
    module.finish()
}

#[cfg(test)]
mod tests {
    /// The program run once in the simulator: the outcome word for the
    /// 64-bit `word`, the pushed `value` and `render_when_equal`.
    fn outcome(word: u64, value: u64, render_when_equal: bool) -> u32 {
        // Pushed, and read from a second word: the same outcome.
        let pushed = simulate(word, value, render_when_equal, false);
        assert_eq!(simulate(word, value, render_when_equal, true), pushed);
        pushed
    }

    fn simulate(word: u64, value: u64, render_when_equal: bool, second_word: bool) -> u32 {
        use echora_shader::sim::{Inputs, Machine};
        // Above 4 GiB, as device addresses are once a few GiB are in use.
        const WORD: u64 = 0x2_0000_1000;
        const OUTCOME: u64 = 0x3_0000_2000;
        const SECOND: u64 = 0x4_0000_3000;
        let comparand = if second_word { SECOND } else { value };
        let push: Vec<u8> = [WORD, OUTCOME, comparand]
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .chain(u32::from(render_when_equal).to_le_bytes())
            .chain([0; 4])
            .collect();
        let memory = [
            (WORD, word.to_le_bytes().to_vec()),
            (OUTCOME, 0xdead_beef_u32.to_le_bytes().to_vec()),
            (SECOND, value.to_le_bytes().to_vec()),
        ]
        .into_iter()
        .collect();
        let report = Machine::new(&super::compare_program(second_word)).simulate(&Inputs {
            workgroup_count: [1, 1, 1],
            push,
            memory,
            ..Inputs::default()
        });
        assert!(report.problems.is_empty(), "{:?}", report.problems);
        u32::from_le_bytes(report.memory[&OUTCOME][..4].try_into().unwrap())
    }

    #[test]
    fn a_word_condition_is_compared_after_the_work_before_it() {
        let Ok(gpu) = crate::vulkan::Gpu::new() else {
            return;
        };
        let mut renderer = crate::renderer::Renderer::new(gpu).unwrap();
        let word = renderer
            .host_buffer_with(8, ash::vk::BufferUsageFlags::STORAGE_BUFFER, true)
            .unwrap();
        word.write(0, &(7u64 << 32 | 5).to_le_bytes());
        let mut outcomes = Vec::new();
        let second = renderer
            .host_buffer_with(8, ash::vk::BufferUsageFlags::STORAGE_BUFFER, true)
            .unwrap();
        second.write(0, &5u64.to_le_bytes());
        let same = renderer
            .host_buffer_with(8, ash::vk::BufferUsageFlags::STORAGE_BUFFER, true)
            .unwrap();
        same.write(0, &(7u64 << 32 | 5).to_le_bytes());
        for (other, when_equal) in [
            (super::Comparand::Value(7u64 << 32 | 5), true),
            (super::Comparand::Value(5), true),
            (super::Comparand::Value(5), false),
            (super::Comparand::Word(second.buffer, 0), true),
            (super::Comparand::Word(same.buffer, 0), true),
            (super::Comparand::Word(same.buffer, 0), false),
        ] {
            let Some(outcome) = renderer
                .condition_on_word((word.buffer, 0), other, when_equal)
                .unwrap()
            else {
                word.destroy(renderer.device());
                second.destroy(renderer.device());
                same.destroy(renderer.device());
                return;
            };
            outcomes.push(outcome);
        }
        // Each outcome stays in its slot until SLOTS later comparisons.
        assert_eq!(renderer.word_conditions_made(), 6);
        assert!((0..6).all(|made| renderer.word_outcome_held(made)));
        assert!(!renderer.word_outcome_held(6));
        renderer.word_conditions.as_mut().unwrap().next = super::SLOTS + 1;
        assert!(!renderer.word_outcome_held(0));
        assert!(renderer.word_outcome_held(1));
        renderer.word_conditions.as_mut().unwrap().next = 6;
        renderer.flush().unwrap();
        renderer.wait_for(renderer.recorded_serial()).unwrap();
        let held = renderer.word_conditions.as_ref().unwrap();
        let bytes = held.outcomes.read_bytes(4 * outcomes.len());
        let words: Vec<u32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        assert_eq!(
            outcomes.iter().map(|(_, at)| *at).collect::<Vec<_>>(),
            [0, 4, 8, 12, 16, 20]
        );
        assert_eq!(words, [1, 0, 1, 0, 1, 0]);
        word.destroy(renderer.device());
        second.destroy(renderer.device());
        same.destroy(renderer.device());
    }

    #[test]
    fn the_compare_programs_are_valid_spirv() {
        for second_word in [false, true] {
            let words = super::compare_program(second_word);
            let path = std::env::temp_dir().join(format!(
                "echora-word-condition-{second_word}-{}.spv",
                std::process::id()
            ));
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
                    "second word {second_word}: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
            }
        }
    }

    #[test]
    fn a_word_condition_compares_all_64_bits() {
        // Mode 0 (render while equal) and mode 1 (while they differ).
        assert_eq!(outcome(0, 0, true), 1);
        assert_eq!(outcome(5, 0, true), 0);
        assert_eq!(outcome(0, 0, false), 0);
        assert_eq!(outcome(5, 0, false), 1);
        // Only the high half differs: not equal.
        assert_eq!(outcome(1 << 32, 0, true), 0);
        assert_eq!(outcome(1 << 32, 0, false), 1);
        assert_eq!(
            outcome(0x1234_5678_9abc_def0, 0x1234_5678_9abc_def0, true),
            1
        );
    }
}
