//! Vulkan device for executing translated NVN work.
//!
//! The device enables the features the recompiler's SPIR-V relies on when the
//! driver offers them, and VK_EXT_shader_object so a single translated stage
//! can be compiled without a full pipeline.

use ash::vk;
use std::ffi::CStr;

use crate::shader::{Descriptor, DescriptorKind, Stage, Translated};

pub struct Gpu {
    pub(crate) entry: ash::Entry,
    pub(crate) instance: ash::Instance,
    pub physical: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub queue_family: u32,
    shader_object: Option<ash::ext::shader_object::Device>,
    /// VK_KHR_swapchain is enabled (windowed presentation is possible).
    pub swapchain: bool,
    /// VK_EXT_depth_clip_control is enabled: draws can take clip-space Z
    /// in [-1, 1] (OpenGL style) instead of [0, 1].
    pub depth_clip_control: bool,
    /// The depthBiasClamp feature is enabled (otherwise a polygon offset's
    /// clamp must be 0).
    pub depth_bias_clamp: bool,
    /// Compute shaders can require 32-lane subgroups (the guest's warp
    /// width; VK_EXT_subgroup_size_control, core in 1.3).
    pub compute_subgroup_32: bool,
    /// Buffers can be given device addresses (bufferDeviceAddress, core in
    /// 1.2): the GPU page table points into them.
    pub buffer_device_address: bool,
    /// VK_EXT_external_memory_host: guest memory can be imported as device
    /// memory, so buffers in it are shared with the GPU without copies.
    pub host_import: Option<ash::ext::external_memory_host::Device>,
    /// VK_KHR_external_memory_fd with VK_EXT_external_memory_dma_buf (Linux):
    /// a dma-buf over guest memory's file (a udmabuf) can be imported as
    /// device memory, so guest buffers are shared with the GPU in place.
    pub dma_buf_import: Option<ash::khr::external_memory_fd::Device>,
    /// VK_EXT_mesh_shader with task and mesh stages: the guest's task and
    /// mesh pipelines can draw.
    pub mesh_shader: Option<ash::ext::mesh_shader::Device>,
    /// VK_EXT_conditional_rendering: draws and dispatches gated on a word
    /// the GPU wrote (the guest's occlusion-conditioned rendering).
    pub conditional_rendering: Option<ash::ext::conditional_rendering::Device>,
    /// VK_KHR_cooperative_matrix is enabled: the shapes and types the
    /// device multiplies (M, N, K, A, B, C, result, saturating, scope).
    pub cooperative_matrix: Vec<CooperativeMatrixShape>,
    /// The texture pool's descriptor set layout (`echora_shader_abi`
    /// `TEXTURE_POOL_SET`), when the device has the descriptor indexing it
    /// needs (with null descriptors, which entries start as): set 1 of
    /// every program that reads the pool.
    pub texture_pool_layout: Option<vk::DescriptorSetLayout>,
    /// The driver's shaderBinaryUUID and shaderBinaryVersion: shader object
    /// binaries it wrote load again only while both are unchanged.
    pub shader_binary_version: ([u8; 16], u32),
    pub name: String,
}

fn vk_error(context: &str) -> impl Fn(vk::Result) -> String + '_ {
    move |error| format!("{context}: {error}")
}

impl Gpu {
    pub fn new() -> Result<Self, String> {
        // SAFETY: loads the system Vulkan loader.
        let entry =
            unsafe { ash::Entry::load() }.map_err(|error| format!("Vulkan loader: {error}"))?;
        let application = vk::ApplicationInfo::default()
            .application_name(c"echora")
            .api_version(vk::API_VERSION_1_3);
        // Window-system surfaces, when the loader offers them, so frames can
        // be presented to a window.
        // SAFETY: the entry is loaded.
        let available = unsafe { entry.enumerate_instance_extension_properties(None) }
            .map_err(vk_error("vkEnumerateInstanceExtensionProperties"))?;
        let surface_extensions: Vec<*const std::ffi::c_char> = [
            ash::khr::surface::NAME,
            ash::khr::wayland_surface::NAME,
            ash::khr::xlib_surface::NAME,
            ash::khr::win32_surface::NAME,
        ]
        .into_iter()
        .filter(|wanted| {
            available
                .iter()
                .any(|extension| extension.extension_name_as_c_str() == Ok(*wanted))
        })
        .map(|name| name.as_ptr())
        .collect();
        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&application)
            .enabled_extension_names(&surface_extensions);
        // SAFETY: valid create info; the instance is destroyed in Drop.
        let instance = unsafe { entry.create_instance(&instance_info, None) }
            .map_err(vk_error("vkCreateInstance"))?;
        Self::with_instance(entry, instance)
    }

    fn with_instance(entry: ash::Entry, instance: ash::Instance) -> Result<Self, String> {
        // SAFETY: the instance is live for these queries.
        let physicals = unsafe { instance.enumerate_physical_devices() }
            .map_err(vk_error("vkEnumeratePhysicalDevices"))?;
        let rank = |physical: &vk::PhysicalDevice| {
            // SAFETY: as above.
            match unsafe { instance.get_physical_device_properties(*physical) }.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 0,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                _ => 3, // CPU implementations (lavapipe) last.
            }
        };
        let physical = *physicals
            .iter()
            .min_by_key(|physical| rank(physical))
            .ok_or("no Vulkan device")?;
        // SAFETY: as above.
        let properties = unsafe { instance.get_physical_device_properties(physical) };
        let name = properties
            .device_name_as_c_str()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        // SAFETY: as above.
        let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let queue_family = families
            .iter()
            .position(|family| {
                family
                    .queue_flags
                    .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
            })
            .ok_or("no graphics+compute queue")? as u32;

        // Enable what the driver supports of the features the SPIR-V may use.
        let mut supported11 = vk::PhysicalDeviceVulkan11Features::default();
        let mut supported12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut supported13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut supported_objects = vk::PhysicalDeviceShaderObjectFeaturesEXT::default();
        let mut supported_clip = vk::PhysicalDeviceDepthClipControlFeaturesEXT::default();
        let mut supported_robust = vk::PhysicalDeviceRobustness2FeaturesEXT::default();
        let mut supported_mesh = vk::PhysicalDeviceMeshShaderFeaturesEXT::default();
        let mut supported_clock = vk::PhysicalDeviceShaderClockFeaturesKHR::default();
        let mut supported_atomic_float = vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default();
        let mut supported_cooperative = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
        let mut supported_conditional =
            vk::PhysicalDeviceConditionalRenderingFeaturesEXT::default();
        let mut supported = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut supported11)
            .push_next(&mut supported12)
            .push_next(&mut supported13)
            .push_next(&mut supported_objects)
            .push_next(&mut supported_clip)
            .push_next(&mut supported_robust)
            .push_next(&mut supported_mesh)
            .push_next(&mut supported_clock)
            .push_next(&mut supported_atomic_float)
            .push_next(&mut supported_cooperative)
            .push_next(&mut supported_conditional);
        // SAFETY: the chain is well formed and outlives the call.
        unsafe { instance.get_physical_device_features2(physical, &mut supported) };
        let core = supported.features;
        // Guest programs assume 32-lane warps: compute shaders require that
        // width where the driver allows it (RADV runs 64 by default).
        let mut size_control = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
        let mut object_properties = vk::PhysicalDeviceShaderObjectPropertiesEXT::default();
        let mut properties2 = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut size_control)
            .push_next(&mut object_properties);
        // SAFETY: the chain is well formed and outlives the call.
        unsafe { instance.get_physical_device_properties2(physical, &mut properties2) };
        let shader_binary_version = (
            object_properties.shader_binary_uuid,
            object_properties.shader_binary_version,
        );
        let buffer_device_address = supported12.buffer_device_address != 0;
        let compute_subgroup_32 = supported13.subgroup_size_control != 0
            && (size_control.min_subgroup_size..=size_control.max_subgroup_size).contains(&32)
            && size_control
                .required_subgroup_size_stages
                .contains(vk::ShaderStageFlags::COMPUTE);
        let (s11, s12, s13, objects) = (
            supported11,
            supported12,
            supported13,
            supported_objects.shader_object,
        );

        // SAFETY: as above.
        let extensions = unsafe { instance.enumerate_device_extension_properties(physical) }
            .map_err(vk_error("vkEnumerateDeviceExtensionProperties"))?;
        let has = |wanted: &CStr| {
            extensions
                .iter()
                .any(|extension| extension.extension_name_as_c_str() == Ok(wanted))
        };
        let use_objects = objects != 0 && has(ash::ext::shader_object::NAME);
        let mut enabled_extensions = Vec::new();
        if use_objects {
            enabled_extensions.push(ash::ext::shader_object::NAME.as_ptr());
        }
        let swapchain = has(ash::khr::swapchain::NAME);
        if swapchain {
            enabled_extensions.push(ash::khr::swapchain::NAME.as_ptr());
        }
        let depth_clip_control = use_objects
            && supported_clip.depth_clip_control != 0
            && has(ash::ext::depth_clip_control::NAME);
        if depth_clip_control {
            enabled_extensions.push(ash::ext::depth_clip_control::NAME.as_ptr());
        }
        // Guest shaders index buffers the way the console allows; reads out
        // of bounds must return zeros, not fault the GPU (a fault hangs the
        // ring and loses the device).
        let robustness2 =
            supported_robust.robust_buffer_access2 != 0 && has(ash::ext::robustness2::NAME);
        if robustness2 {
            enabled_extensions.push(ash::ext::robustness2::NAME.as_ptr());
        }
        let host_import = has(ash::ext::external_memory_host::NAME)
            && std::env::var_os("ECHORA_NO_HOST_IMPORT").is_none();
        if host_import {
            enabled_extensions.push(ash::ext::external_memory_host::NAME.as_ptr());
        }
        // ECHORA_NO_DMA_BUF (diagnostic): guest buffers are copied instead.
        let dma_buf_import = cfg!(target_os = "linux")
            && has(ash::khr::external_memory_fd::NAME)
            && has(ash::ext::external_memory_dma_buf::NAME)
            && std::env::var_os("ECHORA_NO_DMA_BUF").is_none();
        if dma_buf_import {
            enabled_extensions.push(ash::khr::external_memory_fd::NAME.as_ptr());
            enabled_extensions.push(ash::ext::external_memory_dma_buf::NAME.as_ptr());
        }
        // Task and mesh pipelines draw through shader objects of those stages.
        let mesh_shader = use_objects
            && supported_mesh.task_shader != 0
            && supported_mesh.mesh_shader != 0
            && has(ash::ext::mesh_shader::NAME);
        if mesh_shader {
            enabled_extensions.push(ash::ext::mesh_shader::NAME.as_ptr());
        }
        let conditional_rendering = supported_conditional.conditional_rendering != 0
            && has(ash::ext::conditional_rendering::NAME);
        if conditional_rendering {
            enabled_extensions.push(ash::ext::conditional_rendering::NAME.as_ptr());
        }
        // The guest's clock special registers (SPV_KHR_shader_clock) and
        // float atomic adds on storage (SPV_EXT_shader_atomic_float_add).
        let subgroup_clock =
            supported_clock.shader_subgroup_clock != 0 && has(ash::khr::shader_clock::NAME);
        if subgroup_clock {
            enabled_extensions.push(ash::khr::shader_clock::NAME.as_ptr());
        }
        let float_atomic_add = supported_atomic_float.shader_buffer_float32_atomic_add != 0
            && has(ash::ext::shader_atomic_float::NAME);
        if float_atomic_add {
            enabled_extensions.push(ash::ext::shader_atomic_float::NAME.as_ptr());
        }
        // Matrix multiply-add on the GPU's matrix units (guest HMMA).
        let cooperative_matrix = supported_cooperative.cooperative_matrix != 0
            && has(ash::khr::cooperative_matrix::NAME);
        if cooperative_matrix {
            enabled_extensions.push(ash::khr::cooperative_matrix::NAME.as_ptr());
        }

        let features = vk::PhysicalDeviceFeatures::default()
            .robust_buffer_access(core.robust_buffer_access != 0)
            .shader_int64(core.shader_int64 != 0)
            .shader_int16(core.shader_int16 != 0)
            .shader_float64(core.shader_float64 != 0)
            .fragment_stores_and_atomics(core.fragment_stores_and_atomics != 0)
            .vertex_pipeline_stores_and_atomics(core.vertex_pipeline_stores_and_atomics != 0)
            .shader_storage_image_read_without_format(
                core.shader_storage_image_read_without_format != 0,
            )
            .shader_storage_image_write_without_format(
                core.shader_storage_image_write_without_format != 0,
            )
            .image_cube_array(core.image_cube_array != 0)
            .depth_bias_clamp(core.depth_bias_clamp != 0)
            .independent_blend(core.independent_blend != 0)
            .geometry_shader(core.geometry_shader != 0)
            .tessellation_shader(core.tessellation_shader != 0)
            .sample_rate_shading(core.sample_rate_shading != 0)
            .shader_clip_distance(core.shader_clip_distance != 0)
            .multi_viewport(core.multi_viewport != 0)
            .depth_clamp(core.depth_clamp != 0)
            .shader_image_gather_extended(core.shader_image_gather_extended != 0);
        let mut enable11 = vk::PhysicalDeviceVulkan11Features::default()
            .storage_buffer16_bit_access(s11.storage_buffer16_bit_access != 0)
            .uniform_and_storage_buffer16_bit_access(
                s11.uniform_and_storage_buffer16_bit_access != 0,
            )
            .shader_draw_parameters(s11.shader_draw_parameters != 0);
        let mut enable12 = vk::PhysicalDeviceVulkan12Features::default()
            .storage_buffer8_bit_access(s12.storage_buffer8_bit_access != 0)
            .uniform_and_storage_buffer8_bit_access(s12.uniform_and_storage_buffer8_bit_access != 0)
            .shader_float16(s12.shader_float16 != 0)
            .shader_int8(s12.shader_int8 != 0)
            .descriptor_indexing(s12.descriptor_indexing != 0)
            .shader_sampled_image_array_non_uniform_indexing(
                s12.shader_sampled_image_array_non_uniform_indexing != 0,
            )
            .runtime_descriptor_array(s12.runtime_descriptor_array != 0)
            // The texture pool (descriptor set 1): arrays written as
            // textures are registered, partly, while work using them runs.
            .descriptor_binding_partially_bound(s12.descriptor_binding_partially_bound != 0)
            .descriptor_binding_update_unused_while_pending(
                s12.descriptor_binding_update_unused_while_pending != 0,
            )
            .descriptor_binding_sampled_image_update_after_bind(
                s12.descriptor_binding_sampled_image_update_after_bind != 0,
            )
            .descriptor_binding_storage_image_update_after_bind(
                s12.descriptor_binding_storage_image_update_after_bind != 0,
            )
            .descriptor_binding_uniform_texel_buffer_update_after_bind(
                s12.descriptor_binding_uniform_texel_buffer_update_after_bind != 0,
            )
            .scalar_block_layout(s12.scalar_block_layout != 0)
            .shader_output_viewport_index(s12.shader_output_viewport_index != 0)
            .shader_output_layer(s12.shader_output_layer != 0)
            .vulkan_memory_model(s12.vulkan_memory_model != 0)
            .timeline_semaphore(s12.timeline_semaphore != 0)
            .buffer_device_address(s12.buffer_device_address != 0);
        let mut enable13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(s13.dynamic_rendering != 0)
            .synchronization2(s13.synchronization2 != 0)
            .shader_demote_to_helper_invocation(s13.shader_demote_to_helper_invocation != 0)
            .maintenance4(s13.maintenance4 != 0)
            .subgroup_size_control(s13.subgroup_size_control != 0);
        let mut enable_objects =
            vk::PhysicalDeviceShaderObjectFeaturesEXT::default().shader_object(use_objects);
        let priorities = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];
        let mut device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&enabled_extensions)
            .enabled_features(&features)
            .push_next(&mut enable11)
            .push_next(&mut enable12)
            .push_next(&mut enable13);
        let mut enable_clip =
            vk::PhysicalDeviceDepthClipControlFeaturesEXT::default().depth_clip_control(true);
        if use_objects {
            device_info = device_info.push_next(&mut enable_objects);
        }
        if depth_clip_control {
            device_info = device_info.push_next(&mut enable_clip);
        }
        let mut enable_robust = vk::PhysicalDeviceRobustness2FeaturesEXT::default()
            .robust_buffer_access2(true)
            .robust_image_access2(supported_robust.robust_image_access2 != 0)
            .null_descriptor(supported_robust.null_descriptor != 0);
        if robustness2 {
            device_info = device_info.push_next(&mut enable_robust);
        }
        let mut enable_mesh = vk::PhysicalDeviceMeshShaderFeaturesEXT::default()
            .task_shader(true)
            .mesh_shader(true);
        if mesh_shader {
            device_info = device_info.push_next(&mut enable_mesh);
        }
        let mut enable_conditional = vk::PhysicalDeviceConditionalRenderingFeaturesEXT::default()
            .conditional_rendering(true);
        if conditional_rendering {
            device_info = device_info.push_next(&mut enable_conditional);
        }
        let mut enable_clock =
            vk::PhysicalDeviceShaderClockFeaturesKHR::default().shader_subgroup_clock(true);
        if subgroup_clock {
            device_info = device_info.push_next(&mut enable_clock);
        }
        let mut enable_atomic_float = vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default()
            .shader_buffer_float32_atomic_add(true);
        if float_atomic_add {
            device_info = device_info.push_next(&mut enable_atomic_float);
        }
        let mut enable_cooperative =
            vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
        if cooperative_matrix {
            device_info = device_info.push_next(&mut enable_cooperative);
        }
        // SAFETY: every enabled feature was reported as supported.
        let device = unsafe { instance.create_device(physical, &device_info, None) }
            .map_err(vk_error("vkCreateDevice"))?;
        // SAFETY: the family and index exist on this device.
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let shader_object =
            use_objects.then(|| ash::ext::shader_object::Device::new(&instance, &device));
        let host_import =
            host_import.then(|| ash::ext::external_memory_host::Device::new(&instance, &device));
        let dma_buf_import =
            dma_buf_import.then(|| ash::khr::external_memory_fd::Device::new(&instance, &device));
        let mesh_shader =
            mesh_shader.then(|| ash::ext::mesh_shader::Device::new(&instance, &device));
        let conditional_rendering = conditional_rendering
            .then(|| ash::ext::conditional_rendering::Device::new(&instance, &device));
        let texture_pool_layout = (s12.runtime_descriptor_array != 0
            && s12.descriptor_binding_partially_bound != 0
            && s12.descriptor_binding_sampled_image_update_after_bind != 0
            && s12.descriptor_binding_storage_image_update_after_bind != 0
            && s12.descriptor_binding_uniform_texel_buffer_update_after_bind != 0
            && s12.shader_sampled_image_array_non_uniform_indexing != 0
            && robustness2
            && supported_robust.null_descriptor != 0)
            .then(|| texture_pool_layout(&device))
            .transpose()?;
        let cooperative_matrix = if cooperative_matrix {
            let functions = ash::khr::cooperative_matrix::Instance::new(&entry, &instance);
            // SAFETY: a live physical device with the extension.
            unsafe { functions.get_physical_device_cooperative_matrix_properties(physical) }
                .unwrap_or_default()
                .iter()
                .map(|shape| CooperativeMatrixShape {
                    m: shape.m_size,
                    n: shape.n_size,
                    k: shape.k_size,
                    a: shape.a_type,
                    b: shape.b_type,
                    c: shape.c_type,
                    result: shape.result_type,
                    saturating: shape.saturating_accumulation != 0,
                    scope: shape.scope,
                })
                .collect()
        } else {
            Vec::new()
        };
        crate::shader::set_cooperative_matrix(&cooperative_matrix);
        Ok(Self {
            entry,
            instance,
            physical,
            device,
            queue,
            queue_family,
            shader_object,
            swapchain,
            depth_clip_control,
            depth_bias_clamp: core.depth_bias_clamp != 0,
            compute_subgroup_32,
            buffer_device_address,
            texture_pool_layout,
            cooperative_matrix,
            host_import,
            dma_buf_import,
            mesh_shader,
            conditional_rendering,
            shader_binary_version,
            name,
        })
    }

    pub fn memory_properties(&self) -> vk::PhysicalDeviceMemoryProperties {
        // SAFETY: the instance and physical device are live.
        unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical)
        }
    }

    pub fn shader_objects(&self) -> Result<&ash::ext::shader_object::Device, String> {
        self.shader_object
            .as_ref()
            .ok_or_else(|| "driver lacks VK_EXT_shader_object".into())
    }

    /// Descriptor set 0 for one translated stage.
    pub fn descriptor_set_layout(
        &self,
        descriptors: &[Descriptor],
        stages: vk::ShaderStageFlags,
    ) -> Result<vk::DescriptorSetLayout, String> {
        let pairs: Vec<_> = descriptors
            .iter()
            .map(|descriptor| (*descriptor, stages))
            .collect();
        self.descriptor_set_layout_for(&pairs)
    }

    /// Descriptor set 0 for the bindings of several stages, each visible to
    /// its own stage.
    pub fn descriptor_set_layout_for(
        &self,
        descriptors: &[(Descriptor, vk::ShaderStageFlags)],
    ) -> Result<vk::DescriptorSetLayout, String> {
        let bindings: Vec<_> = descriptors
            .iter()
            .map(|(descriptor, stages)| {
                let kind = match descriptor.kind() {
                    DescriptorKind::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
                    DescriptorKind::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
                    DescriptorKind::TextureBuffer => vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
                    DescriptorKind::ImageBuffer => vk::DescriptorType::STORAGE_TEXEL_BUFFER,
                    DescriptorKind::Texture => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    DescriptorKind::Image => vk::DescriptorType::STORAGE_IMAGE,
                };
                // Buffers occupy one binding per element; the others are
                // arrays at a single binding.
                let count = match descriptor.kind() {
                    DescriptorKind::UniformBuffer | DescriptorKind::StorageBuffer => 1,
                    _ => descriptor.count,
                };
                vk::DescriptorSetLayoutBinding::default()
                    .binding(descriptor.binding)
                    .descriptor_type(kind)
                    .descriptor_count(count)
                    .stage_flags(*stages)
            })
            .collect();
        let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY: valid create info on a live device.
        unsafe { self.device.create_descriptor_set_layout(&info, None) }
            .map_err(vk_error("vkCreateDescriptorSetLayout"))
    }

    /// Compile one translated stage on the driver (VK_EXT_shader_object) and
    /// discard it: a check that the SPIR-V is accepted.
    pub fn check_stage(&self, translated: &Translated) -> Result<(), String> {
        let objects = self
            .shader_object
            .as_ref()
            .ok_or("driver lacks VK_EXT_shader_object")?;
        let stage = stage_flags(translated.stage);
        let layout = self.descriptor_set_layout(&translated.descriptors, stage)?;
        let mut layouts = vec![layout];
        if translated.texture_pool_bindings != 0 {
            match self.texture_pool_layout {
                Some(pool) => layouts.push(pool),
                None => {
                    // SAFETY: the layout is not used.
                    unsafe { self.device.destroy_descriptor_set_layout(layout, None) };
                    return Err("the device cannot hold the texture pool".into());
                }
            }
        }
        let next = if translated.stage == Stage::Vertex {
            vk::ShaderStageFlags::FRAGMENT
        } else {
            vk::ShaderStageFlags::empty()
        };
        let code: &[u8] = bytemuck_words(&translated.spirv);
        // As the renderer creates them: compute in 32-lane subgroups.
        let mut subgroup_32 =
            vk::ShaderRequiredSubgroupSizeCreateInfoEXT::default().required_subgroup_size(32);
        let mut info = vk::ShaderCreateInfoEXT::default()
            .stage(stage)
            .next_stage(next)
            .code_type(vk::ShaderCodeTypeEXT::SPIRV)
            .code(code)
            .name(c"main")
            .set_layouts(&layouts);
        if self.compute_subgroup_32 && translated.stage == Stage::Compute {
            info = info.push_next(&mut subgroup_32);
            if whole_subgroups(&translated.spirv) {
                info = info.flags(vk::ShaderCreateFlagsEXT::REQUIRE_FULL_SUBGROUPS);
            }
        }
        // SAFETY: the code and layout outlive the call.
        let result = unsafe { objects.create_shaders(&[info], None) };
        // SAFETY: the layout is no longer used.
        unsafe { self.device.destroy_descriptor_set_layout(layout, None) };
        match result {
            Ok(shaders) => {
                for shader in shaders {
                    // SAFETY: created above and unused.
                    unsafe { objects.destroy_shader(shader, None) };
                }
                Ok(())
            }
            Err((shaders, error)) => {
                for shader in shaders
                    .into_iter()
                    .filter(|shader| *shader != vk::ShaderEXT::null())
                {
                    // SAFETY: as above.
                    unsafe { objects.destroy_shader(shader, None) };
                }
                Err(format!("vkCreateShadersEXT: {error}"))
            }
        }
    }
}

/// Whether a compute program's workgroups split into whole 32-lane
/// subgroups (its LocalSize x is a multiple of 32), so it may be created
/// with REQUIRE_FULL_SUBGROUPS: every lane of a guest warp is then active,
/// as warp-wide operations (subgroup shuffles, cooperative matrices)
/// assume.
pub(crate) fn whole_subgroups(spirv: &[u32]) -> bool {
    const OP_EXECUTION_MODE: u32 = 16;
    const LOCAL_SIZE: u32 = 17;
    let mut at = 5;
    while let Some(&first) = spirv.get(at) {
        let (count, opcode) = ((first >> 16) as usize, first & 0xffff);
        if count == 0 {
            return false;
        }
        if opcode == OP_EXECUTION_MODE && spirv.get(at + 2) == Some(&LOCAL_SIZE) {
            return spirv.get(at + 3).is_some_and(|x| *x != 0 && x % 32 == 0);
        }
        at += count;
    }
    false
}

/// One shape the device's cooperative matrices multiply.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CooperativeMatrixShape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub a: vk::ComponentTypeKHR,
    pub b: vk::ComponentTypeKHR,
    pub c: vk::ComponentTypeKHR,
    pub result: vk::ComponentTypeKHR,
    pub saturating: bool,
    pub scope: vk::ScopeKHR,
}

pub use echora_shader_abi::{SAMPLER_POOL_CAPACITY, TEXTURE_POOL_CAPACITY};

/// The set layout of `echora_shader_abi::texture_pool`: arrays updated
/// after bind and partially bound, for every stage.
fn texture_pool_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout, String> {
    use echora_shader_abi::texture_pool as pool;
    let kind = |binding| match binding {
        pool::SAMPLERS => vk::DescriptorType::SAMPLER,
        pool::SAMPLED_BUFFER => vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
        pool::STORAGE_2D | pool::STORAGE_3D => vk::DescriptorType::STORAGE_IMAGE,
        pool::FLAGS => vk::DescriptorType::STORAGE_BUFFER,
        _ => vk::DescriptorType::SAMPLED_IMAGE,
    };
    let bindings: Vec<_> = (0..pool::COUNT)
        .map(|binding| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(kind(binding))
                .descriptor_count(match binding {
                    pool::SAMPLERS => SAMPLER_POOL_CAPACITY,
                    pool::FLAGS => 1,
                    _ => TEXTURE_POOL_CAPACITY,
                })
                .stage_flags(vk::ShaderStageFlags::ALL)
        })
        .collect();
    // The flags buffer is written once, when the set is made.
    let flags: Vec<_> = (0..pool::COUNT)
        .map(|binding| {
            if binding == pool::FLAGS {
                vk::DescriptorBindingFlags::empty()
            } else {
                vk::DescriptorBindingFlags::PARTIALLY_BOUND
                    | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
                    | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING
            }
        })
        .collect();
    let mut binding_flags =
        vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&flags);
    let info = vk::DescriptorSetLayoutCreateInfo::default()
        .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
        .bindings(&bindings)
        .push_next(&mut binding_flags);
    // SAFETY: valid create info on a live device; destroyed with the Gpu.
    unsafe { device.create_descriptor_set_layout(&info, None) }
        .map_err(vk_error("vkCreateDescriptorSetLayout (texture pool)"))
}

impl Drop for Gpu {
    fn drop(&mut self) {
        // SAFETY: nothing created from the device outlives it.
        unsafe {
            let _ = self.device.device_wait_idle();
            if let Some(layout) = self.texture_pool_layout {
                self.device.destroy_descriptor_set_layout(layout, None);
            }
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

pub fn stage_flags(stage: Stage) -> vk::ShaderStageFlags {
    match stage {
        Stage::Vertex => vk::ShaderStageFlags::VERTEX,
        Stage::TessControl => vk::ShaderStageFlags::TESSELLATION_CONTROL,
        Stage::TessEval => vk::ShaderStageFlags::TESSELLATION_EVALUATION,
        Stage::Geometry => vk::ShaderStageFlags::GEOMETRY,
        Stage::Fragment => vk::ShaderStageFlags::FRAGMENT,
        Stage::Compute => vk::ShaderStageFlags::COMPUTE,
        Stage::Task => vk::ShaderStageFlags::TASK_EXT,
        Stage::Mesh => vk::ShaderStageFlags::MESH_EXT,
    }
}

fn bytemuck_words(words: &[u32]) -> &[u8] {
    // SAFETY: u32 has no padding and u8 has alignment 1.
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast(), words.len() * 4) }
}
