//! One 16x16x16 half-precision cooperative-matrix multiply-add on the
//! device, against the CPU: whether VK_KHR_cooperative_matrix works under
//! the subgroup sizes the runtime uses.
//!   coop-check VARIANT ACCUMULATOR
//! VARIANT: `32full` (required size 32, REQUIRE_FULL_SUBGROUPS), `32`
//! (required size 32), `64` (required size 64). ACCUMULATOR: `16` or `32`.
//! The shader is built here with echora-shader's SPIR-V writer, in the
//! shape the translator gives HMMA: the lanes store A, B and C to
//! workgroup memory, the subgroup loads them as cooperative matrices
//! (row-major, stride 16), multiplies, stores D back there, and the lanes
//! copy it out. Exit status 0 when D matches, 1 when it differs or the GPU
//! does not finish within two seconds.

// Records and submits Vulkan work directly (ash), as the library does.
#![allow(unsafe_code)]

use ash::vk;
use echora_gpu::vulkan::Gpu;
use echora_shader::spirv::{
    Function, Module, Type, capability, decoration, execution, op, storage,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let [_, variant, accumulator] = args.as_slice() else {
        eprintln!("usage: coop-check 32full|32|64 16|32");
        std::process::exit(2);
    };
    let wide = accumulator == "32";
    let (subgroup, full) = match variant.as_str() {
        "32full" => (32, true),
        "32" => (32, false),
        "64" => (64, false),
        _ => {
            eprintln!("variant must be 32full, 32 or 64");
            std::process::exit(2);
        }
    };
    let spirv: Vec<u8> = shader(subgroup, wide)
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    // COOP_CHECK_SPIRV=<file>: write the program (for spirv-val) and stop.
    if let Some(file) = std::env::var_os("COOP_CHECK_SPIRV") {
        if let Err(error) = std::fs::write(&file, &spirv) {
            eprintln!("{}: {error}", std::path::Path::new(&file).display());
            std::process::exit(2);
        }
        return;
    }
    let gpu = Gpu::new().unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(2);
    });
    println!(
        "device: {}; variant {variant}, accumulator f{accumulator}",
        gpu.name
    );
    match run(&gpu, &spirv, subgroup, full) {
        Ok(true) => println!("result: matches"),
        Ok(false) => {
            println!("result: DIFFERS");
            std::process::exit(1);
        }
        Err(error) => {
            println!("error: {error}");
            std::process::exit(1);
        }
    }
}

/// The test program for a workgroup of `lanes` (one subgroup), with a
/// half or (`wide`) 32-bit accumulator. Binding 0 holds A, B and C (256
/// floats each), binding 1 receives D (256 floats).
/// OpFConvert.
const F_CONVERT: u16 = 115;

fn shader(lanes: u32, wide: bool) -> Vec<u32> {
    const SCOPE_WORKGROUP: u32 = 2;
    const SCOPE_SUBGROUP: u32 = 3;
    const LOCAL_INVOCATION_INDEX: u32 = 29;
    let mut m = Module::new();
    m.capability(capability::SHADER);
    m.capability(capability::FLOAT16);
    m.capability(capability::COOPERATIVE_MATRIX);
    m.extension("SPV_KHR_cooperative_matrix");
    m.memory_model(0, 1);
    let void = m.void();
    let u32_type = m.u32();
    let f32_type = m.f32();
    let f16 = m.ty(Type::Float { width: 16 });
    let accumulator = if wide { f32_type } else { f16 };
    let c = |m: &mut Module, value: u32| m.u32_constant(value);
    let (n256, n768) = (c(&mut m, 256), c(&mut m, 768));
    // Storage buffers of floats.
    let buffer = |m: &mut Module, length: u32, binding: u32| {
        let array = m.ty(Type::Array {
            element: f32_type,
            length,
        });
        m.decorate(array, decoration::ARRAY_STRIDE, &[4]);
        let block = m.ty(Type::Struct {
            members: vec![array],
        });
        m.decorate(block, decoration::BLOCK, &[]);
        m.member_decorate(block, 0, decoration::OFFSET, &[0]);
        let pointer = m.pointer(storage::STORAGE_BUFFER, block);
        let variable = m.global_variable(pointer, storage::STORAGE_BUFFER);
        m.decorate(variable, decoration::DESCRIPTOR_SET, &[0]);
        m.decorate(variable, decoration::BINDING, &[binding]);
        variable
    };
    let input = buffer(&mut m, n768, 0);
    let output = buffer(&mut m, n256, 1);
    let workgroup = |m: &mut Module, element: u32| {
        let array = m.ty(Type::Array {
            element,
            length: n256,
        });
        let pointer = m.pointer(storage::WORKGROUP, array);
        m.global_variable(pointer, storage::WORKGROUP)
    };
    let (shared_a, shared_b, shared_c) = (
        workgroup(&mut m, f16),
        workgroup(&mut m, f16),
        workgroup(&mut m, accumulator),
    );
    let index_pointer = m.pointer(storage::INPUT, u32_type);
    let index = m.global_variable(index_pointer, storage::INPUT);
    m.decorate(index, decoration::BUILT_IN, &[LOCAL_INVOCATION_INDEX]);
    let function_type = m.ty(Type::Function {
        result: void,
        parameters: Vec::new(),
    });
    let mut f = Function::new(&mut m, void, function_type);
    let lane = f.value(&mut m, op::LOAD, u32_type, &[index]);
    let float_in = m.pointer(storage::STORAGE_BUFFER, f32_type);
    let half_shared = m.pointer(storage::WORKGROUP, f16);
    let wide_shared = m.pointer(storage::WORKGROUP, accumulator);
    let zero = c(&mut m, 0);
    let element = |f: &mut Function, m: &mut Module, ty: u32, base: u32, at: u32| {
        f.value(m, op::ACCESS_CHAIN, ty, &[base, at])
    };
    // In: each lane converts every `lanes`-th element.
    for step in 0..256 / lanes {
        let offset = c(&mut m, step * lanes);
        let at = f.value(&mut m, op::I_ADD, u32_type, &[lane, offset]);
        for (part, shared, ty, element_type) in [
            (0, shared_a, half_shared, f16),
            (256, shared_b, half_shared, f16),
            (512, shared_c, wide_shared, accumulator),
        ] {
            let shift = c(&mut m, part);
            let from = f.value(&mut m, op::I_ADD, u32_type, &[at, shift]);
            let source = f.value(&mut m, op::ACCESS_CHAIN, float_in, &[input, zero, from]);
            let value = f.value(&mut m, op::LOAD, f32_type, &[source]);
            let value = if element_type == f32_type {
                value
            } else {
                f.value(&mut m, F_CONVERT, element_type, &[value])
            };
            let target = element(&mut f, &mut m, ty, shared, at);
            f.emit(op::STORE, &[target, value]);
        }
    }
    let barrier = |f: &mut Function, m: &mut Module| {
        let scope = m.u32_constant(SCOPE_WORKGROUP);
        let semantics = m.u32_constant(0x8 | 0x100);
        f.emit(op::CONTROL_BARRIER, &[scope, scope, semantics]);
    };
    barrier(&mut f, &mut m);
    // The multiply, at subgroup scope: row-major, stride 16.
    let (scope, sixteen, row_major) = (c(&mut m, SCOPE_SUBGROUP), c(&mut m, 16), zero);
    let matrix = |m: &mut Module, component: u32, usage: u32| {
        let usage = m.u32_constant(usage);
        m.ty(Type::CooperativeMatrix {
            component,
            scope,
            rows: sixteen,
            columns: sixteen,
            usage,
        })
    };
    let (a_type, b_type, c_type) = (
        matrix(&mut m, f16, 0),
        matrix(&mut m, f16, 1),
        matrix(&mut m, accumulator, 2),
    );
    let first_a = element(&mut f, &mut m, half_shared, shared_a, zero);
    let first_b = element(&mut f, &mut m, half_shared, shared_b, zero);
    let first_c = element(&mut f, &mut m, wide_shared, shared_c, zero);
    let load = |f: &mut Function, m: &mut Module, ty: u32, pointer: u32| {
        f.value(
            m,
            op::COOPERATIVE_MATRIX_LOAD,
            ty,
            &[pointer, row_major, sixteen],
        )
    };
    let a = load(&mut f, &mut m, a_type, first_a);
    let b = load(&mut f, &mut m, b_type, first_b);
    let c_matrix = load(&mut f, &mut m, c_type, first_c);
    let d = f.value(
        &mut m,
        op::COOPERATIVE_MATRIX_MUL_ADD,
        c_type,
        &[a, b, c_matrix],
    );
    f.emit(
        op::COOPERATIVE_MATRIX_STORE,
        &[first_c, d, row_major, sixteen],
    );
    barrier(&mut f, &mut m);
    // Out: D as floats.
    for step in 0..256 / lanes {
        let offset = c(&mut m, step * lanes);
        let at = f.value(&mut m, op::I_ADD, u32_type, &[lane, offset]);
        let source = element(&mut f, &mut m, wide_shared, shared_c, at);
        let value = f.value(&mut m, op::LOAD, accumulator, &[source]);
        let value = if wide {
            value
        } else {
            f.value(&mut m, F_CONVERT, f32_type, &[value])
        };
        let target = f.value(&mut m, op::ACCESS_CHAIN, float_in, &[output, zero, at]);
        f.emit(op::STORE, &[target, value]);
    }
    f.emit(op::RETURN, &[]);
    let entry = f.id;
    f.finish(&mut m);
    m.entry_point(
        execution::GL_COMPUTE,
        entry,
        "main",
        &[input, output, shared_a, shared_b, shared_c, index],
    );
    m.execution_mode(entry, execution::LOCAL_SIZE, &[lanes, 1, 1]);
    m.finish()
}

/// A host-visible buffer of `bytes`.
fn buffer(gpu: &Gpu, bytes: &[u8]) -> Result<(vk::Buffer, vk::DeviceMemory), String> {
    let device = &gpu.device;
    let info = vk::BufferCreateInfo::default()
        .size(bytes.len().max(4) as u64)
        .usage(vk::BufferUsageFlags::STORAGE_BUFFER);
    // SAFETY: valid create info; destroyed by the caller.
    let buffer = unsafe { device.create_buffer(&info, None) }.map_err(|e| e.to_string())?;
    // SAFETY: a live buffer.
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let properties = gpu.memory_properties();
    let wanted = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let index = (0..properties.memory_type_count)
        .find(|&index| {
            requirements.memory_type_bits & (1 << index) != 0
                && properties.memory_types[index as usize]
                    .property_flags
                    .contains(wanted)
        })
        .ok_or("no host-visible memory")?;
    let allocate = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(index);
    // SAFETY: valid allocate info; the memory is mapped, written and
    // bound once.
    unsafe {
        let memory = device
            .allocate_memory(&allocate, None)
            .map_err(|e| e.to_string())?;
        device
            .bind_buffer_memory(buffer, memory, 0)
            .map_err(|e| e.to_string())?;
        let mapped = device
            .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            .map_err(|e| e.to_string())?;
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped.cast(), bytes.len());
        Ok((buffer, memory))
    }
}

fn run(gpu: &Gpu, spirv: &[u8], subgroup: u32, full: bool) -> Result<bool, String> {
    let device = &gpu.device;
    let objects = gpu.shader_objects()?;
    // A, B: small integers; C: small integers; exact in half and float.
    let a: Vec<f32> = (0..256).map(|i| ((i * 7 + 3) % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..256).map(|i| ((i * 11 + 1) % 5) as f32 - 2.0).collect();
    let c: Vec<f32> = (0..256).map(|i| ((i * 13) % 7) as f32 - 3.0).collect();
    let mut expected = vec![0f32; 256];
    for row in 0..16 {
        for column in 0..16 {
            let mut sum = c[row * 16 + column];
            for k in 0..16 {
                sum += a[row * 16 + k] * b[k * 16 + column];
            }
            expected[row * 16 + column] = sum;
        }
    }
    let floats =
        |values: &[f32]| -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let inputs: Vec<f32> = a.iter().chain(&b).chain(&c).copied().collect();
    let buffers = [buffer(gpu, &floats(&inputs))?, buffer(gpu, &[0u8; 1024])?];
    let bindings: Vec<_> = (0..2)
        .map(|binding| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();
    // SAFETY (this block): valid create infos on a live device; everything
    // is recorded, submitted once and waited for with a timeout.
    unsafe {
        let layout = device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
            .map_err(|e| e.to_string())?;
        let layouts = [layout];
        let pipeline_layout = device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts),
                None,
            )
            .map_err(|e| e.to_string())?;
        let mut size =
            vk::ShaderRequiredSubgroupSizeCreateInfoEXT::default().required_subgroup_size(subgroup);
        let info = vk::ShaderCreateInfoEXT::default()
            .flags(if full {
                vk::ShaderCreateFlagsEXT::REQUIRE_FULL_SUBGROUPS
            } else {
                vk::ShaderCreateFlagsEXT::empty()
            })
            .stage(vk::ShaderStageFlags::COMPUTE)
            .code_type(vk::ShaderCodeTypeEXT::SPIRV)
            .code(spirv)
            .name(c"main")
            .set_layouts(&layouts)
            .push_next(&mut size);
        let shaders = objects
            .create_shaders(&[info], None)
            .map_err(|(_, e)| format!("vkCreateShadersEXT: {e}"))?;
        let sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 2,
        }];
        let pool = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&sizes),
                None,
            )
            .map_err(|e| e.to_string())?;
        let set = device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts),
            )
            .map_err(|e| e.to_string())?[0];
        let infos: Vec<[vk::DescriptorBufferInfo; 1]> = buffers
            .iter()
            .map(|(buffer, _)| {
                [vk::DescriptorBufferInfo {
                    buffer: *buffer,
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                }]
            })
            .collect();
        let writes: Vec<_> = infos
            .iter()
            .enumerate()
            .map(|(binding, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(binding as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(info)
            })
            .collect();
        device.update_descriptor_sets(&writes, &[]);
        let command_pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(gpu.queue_family),
                None,
            )
            .map_err(|e| e.to_string())?;
        let command = device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .command_buffer_count(1),
            )
            .map_err(|e| e.to_string())?[0];
        device
            .begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())
            .map_err(|e| e.to_string())?;
        objects.cmd_bind_shaders(command, &[vk::ShaderStageFlags::COMPUTE], &shaders);
        device.cmd_bind_descriptor_sets(
            command,
            vk::PipelineBindPoint::COMPUTE,
            pipeline_layout,
            0,
            &[set],
            &[],
        );
        device.cmd_dispatch(command, 1, 1, 1);
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ);
        device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
        device
            .end_command_buffer(command)
            .map_err(|e| e.to_string())?;
        let fence = device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .map_err(|e| e.to_string())?;
        let commands = [command];
        device
            .queue_submit(
                gpu.queue,
                &[vk::SubmitInfo::default().command_buffers(&commands)],
                fence,
            )
            .map_err(|e| e.to_string())?;
        device
            .wait_for_fences(&[fence], true, 2_000_000_000)
            .map_err(|e| format!("waiting for the GPU: {e}"))?;
        let mapped = device
            .map_memory(buffers[1].1, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            .map_err(|e| e.to_string())?;
        let bytes = std::slice::from_raw_parts(mapped as *const u8, 1024);
        let got: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let wrong = got.iter().zip(&expected).filter(|(g, e)| g != e).count();
        if wrong != 0 {
            println!(
                "{wrong} of 256 elements differ; first row got {:?}",
                &got[..16]
            );
            println!("expected {:?}", &expected[..16]);
        }
        Ok(wrong == 0)
    }
}
