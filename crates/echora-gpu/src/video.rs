//! AV1 decoding on the GPU's video engine (Vulkan Video).
//!
//! The guest's movies are AV1 Main, 8-bit 4:2:0. A parser (`echora-av1`)
//! reads each frame's headers into the Vulkan Video standard structures;
//! this module owns the decode session, its picture buffer (DPB) and the
//! bitstream buffer, records `vkCmdDecodeVideoKHR` on a decode queue and
//! reads finished pictures back as NV12.
//!
//! Decoding has a Vulkan device of its own (`VideoDevice`) with only a
//! decode queue and a copy queue, so it never shares a queue with the
//! renderer and runs on whichever thread decodes the movie.

use std::sync::{Arc, Mutex};

use ash::vk;
use ash::vk::native::{
    StdVideoAV1SequenceHeader, StdVideoDecodeAV1PictureInfo, StdVideoDecodeAV1ReferenceInfo,
};

fn vk_error(context: &str) -> impl Fn(vk::Result) -> String + '_ {
    move |error| format!("{context}: {error}")
}

/// A Vulkan device whose queue families decode AV1 and copy.
pub struct VideoDevice {
    _entry: ash::Entry,
    instance: ash::Instance,
    pub physical: vk::PhysicalDevice,
    pub device: ash::Device,
    pub name: String,
    pub decode_family: u32,
    decode_queue: Mutex<vk::Queue>,
    pub transfer_family: u32,
    transfer_queue: Mutex<vk::Queue>,
    video_instance: ash::khr::video_queue::Instance,
    video: ash::khr::video_queue::Device,
    decode: ash::khr::video_decode_queue::Device,
    memory: vk::PhysicalDeviceMemoryProperties,
}

const EXTENSIONS: [&std::ffi::CStr; 3] = [
    ash::khr::video_queue::NAME,
    ash::khr::video_decode_queue::NAME,
    ash::khr::video_decode_av1::NAME,
];

impl VideoDevice {
    /// The first device (discrete GPUs first) with an AV1 decode queue.
    pub fn new() -> Result<Arc<Self>, String> {
        // SAFETY: loads the system Vulkan loader.
        let entry =
            unsafe { ash::Entry::load() }.map_err(|error| format!("Vulkan loader: {error}"))?;
        let application = vk::ApplicationInfo::default()
            .application_name(c"echora-video")
            .api_version(vk::API_VERSION_1_3);
        let instance_info = vk::InstanceCreateInfo::default().application_info(&application);
        // SAFETY: valid create info; the instance is destroyed in Drop.
        let instance = unsafe { entry.create_instance(&instance_info, None) }
            .map_err(vk_error("vkCreateInstance"))?;
        match Self::with_instance(&instance) {
            Ok((physical, decode_family, transfer_family, name)) => Self::create(
                entry,
                instance,
                physical,
                decode_family,
                transfer_family,
                name,
            ),
            Err(error) => {
                // SAFETY: nothing was created from the instance.
                unsafe { instance.destroy_instance(None) };
                Err(error)
            }
        }
    }

    fn with_instance(
        instance: &ash::Instance,
    ) -> Result<(vk::PhysicalDevice, u32, u32, String), String> {
        // SAFETY: the instance is live for these queries.
        let mut physicals = unsafe { instance.enumerate_physical_devices() }
            .map_err(vk_error("vkEnumeratePhysicalDevices"))?;
        physicals.sort_by_key(|physical| {
            // SAFETY: as above.
            match unsafe { instance.get_physical_device_properties(*physical) }.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 0,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                _ => 2,
            }
        });
        for physical in physicals {
            // SAFETY: as above.
            let extensions = unsafe { instance.enumerate_device_extension_properties(physical) }
                .map_err(vk_error("vkEnumerateDeviceExtensionProperties"))?;
            let has = |name: &std::ffi::CStr| {
                extensions
                    .iter()
                    .any(|extension| extension.extension_name_as_c_str() == Ok(name))
            };
            if !EXTENSIONS.into_iter().all(has) {
                continue;
            }
            // SAFETY: as above.
            let count =
                unsafe { instance.get_physical_device_queue_family_properties2_len(physical) };
            let mut video = vec![vk::QueueFamilyVideoPropertiesKHR::default(); count];
            let mut properties: Vec<_> = video
                .iter_mut()
                .map(|video| vk::QueueFamilyProperties2::default().push_next(video))
                .collect();
            // SAFETY: `properties` has the reported length, each chained to
            // its video properties.
            unsafe {
                instance.get_physical_device_queue_family_properties2(physical, &mut properties)
            };
            let flags: Vec<vk::QueueFlags> = properties
                .iter()
                .map(|family| family.queue_family_properties.queue_flags)
                .collect();
            drop(properties);
            let Some(decode) = (0..count).find(|&index| {
                flags[index].contains(vk::QueueFlags::VIDEO_DECODE_KHR)
                    && video[index]
                        .video_codec_operations
                        .contains(vk::VideoCodecOperationFlagsKHR::DECODE_AV1)
            }) else {
                continue;
            };
            // A copy-only family (the DMA engine) when there is one.
            let copies = |index: &usize| {
                flags[*index].intersects(
                    vk::QueueFlags::TRANSFER | vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE,
                )
            };
            let transfer = (0..count)
                .filter(copies)
                .min_by_key(|&index| {
                    flags[index].intersects(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
                })
                .ok_or("no queue family copies")?;
            // SAFETY: as above.
            let properties = unsafe { instance.get_physical_device_properties(physical) };
            let name = properties
                .device_name_as_c_str()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default();
            return Ok((physical, decode as u32, transfer as u32, name));
        }
        Err("no device decodes AV1 (VK_KHR_video_decode_av1)".into())
    }

    fn create(
        entry: ash::Entry,
        instance: ash::Instance,
        physical: vk::PhysicalDevice,
        decode_family: u32,
        transfer_family: u32,
        name: String,
    ) -> Result<Arc<Self>, String> {
        let priorities = [1.0f32];
        let mut queues = vec![
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(decode_family)
                .queue_priorities(&priorities),
        ];
        if transfer_family != decode_family {
            queues.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(transfer_family)
                    .queue_priorities(&priorities),
            );
        }
        let extensions: Vec<_> = EXTENSIONS.iter().map(|name| name.as_ptr()).collect();
        let mut features13 = vk::PhysicalDeviceVulkan13Features::default().synchronization2(true);
        let info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&extensions)
            .push_next(&mut features13);
        // SAFETY: valid create info; the device is destroyed in Drop.
        let device = match unsafe { instance.create_device(physical, &info, None) } {
            Ok(device) => device,
            Err(error) => {
                // SAFETY: nothing else was created from the instance.
                unsafe { instance.destroy_instance(None) };
                return Err(format!("vkCreateDevice (video): {error}"));
            }
        };
        // SAFETY: both families were requested with one queue each.
        let (decode_queue, transfer_queue) = unsafe {
            (
                device.get_device_queue(decode_family, 0),
                device.get_device_queue(transfer_family, 0),
            )
        };
        // SAFETY: a live physical device.
        let memory = unsafe { instance.get_physical_device_memory_properties(physical) };
        Ok(Arc::new(Self {
            video_instance: ash::khr::video_queue::Instance::new(&entry, &instance),
            video: ash::khr::video_queue::Device::new(&instance, &device),
            decode: ash::khr::video_decode_queue::Device::new(&instance, &device),
            _entry: entry,
            instance,
            physical,
            device,
            name,
            decode_family,
            decode_queue: Mutex::new(decode_queue),
            transfer_family,
            transfer_queue: Mutex::new(transfer_queue),
            memory,
        }))
    }

    fn memory_type(&self, bits: u32, wanted: vk::MemoryPropertyFlags) -> Option<u32> {
        (0..self.memory.memory_type_count).find(|&index| {
            bits & (1 << index) != 0
                && self.memory.memory_types[index as usize]
                    .property_flags
                    .contains(wanted)
        })
    }

    /// Device-local memory for `requirements` (any memory it allows, if
    /// none is device-local).
    fn allocate_local(
        &self,
        requirements: vk::MemoryRequirements,
    ) -> Result<vk::DeviceMemory, String> {
        let bits = requirements.memory_type_bits;
        let index = self
            .memory_type(bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
            .or_else(|| self.memory_type(bits, vk::MemoryPropertyFlags::empty()))
            .ok_or("no memory type for a video resource")?;
        let info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(index);
        // SAFETY: valid allocate info; freed by the owner.
        unsafe { self.device.allocate_memory(&info, None) }
            .map_err(vk_error("vkAllocateMemory (video)"))
    }

    fn submit(
        &self,
        queue: &Mutex<vk::Queue>,
        command: vk::CommandBuffer,
        fence: vk::Fence,
    ) -> Result<(), String> {
        let commands = [vk::CommandBufferSubmitInfo::default().command_buffer(command)];
        let submit = [vk::SubmitInfo2::default().command_buffer_infos(&commands)];
        let queue = queue.lock().unwrap_or_else(|poison| poison.into_inner());
        // SAFETY: the queue is held; the command buffer is recorded and the
        // fence unsignaled.
        unsafe {
            self.device
                .reset_fences(&[fence])
                .map_err(vk_error("vkResetFences"))?;
            self.device
                .queue_submit2(*queue, &submit, fence)
                .map_err(vk_error("vkQueueSubmit2 (video)"))?;
        }
        drop(queue);
        // SAFETY: the fence belongs to this device.
        unsafe { self.device.wait_for_fences(&[fence], true, 5_000_000_000) }
            .map_err(vk_error("vkWaitForFences (video)"))
    }
}

impl Drop for VideoDevice {
    fn drop(&mut self) {
        // SAFETY: decoders hold an Arc, so none outlives the device.
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// What the device's AV1 Main 8-bit 4:2:0 decoding supports.
#[derive(Clone, Copy, Debug)]
pub struct Av1Capabilities {
    pub min_extent: vk::Extent2D,
    pub max_extent: vk::Extent2D,
    pub max_dpb_slots: u32,
    pub max_active_references: u32,
    pub bitstream_offset_alignment: u64,
    pub bitstream_size_alignment: u64,
    pub picture_access_granularity: vk::Extent2D,
    /// Decoded pictures may be written straight into the DPB image
    /// (`DPB_AND_OUTPUT_COINCIDE`).
    pub coincide: bool,
    /// Decoded pictures may be written to an image apart from the DPB
    /// (`DPB_AND_OUTPUT_DISTINCT`).
    pub distinct: bool,
    /// StdVideoAV1Level.
    pub max_level: u32,
    pub std_header_version: vk::ExtensionProperties,
}

/// The pixel format and image flags the driver wants for an image used as
/// `usage` in this profile.
#[derive(Clone, Copy, Debug)]
pub struct PictureFormat {
    pub format: vk::Format,
    pub create_flags: vk::ImageCreateFlags,
    pub tiling: vk::ImageTiling,
    pub usage: vk::ImageUsageFlags,
}

/// AV1 Main profile, 8-bit 4:2:0, without film grain synthesis: the
/// movies carry none.
fn with_profile<R>(f: impl FnOnce(&vk::VideoProfileInfoKHR) -> R) -> R {
    let mut av1 = vk::VideoDecodeAV1ProfileInfoKHR::default()
        .std_profile(vk::native::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_MAIN)
        .film_grain_support(false);
    let profile = vk::VideoProfileInfoKHR::default()
        .video_codec_operation(vk::VideoCodecOperationFlagsKHR::DECODE_AV1)
        .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
        .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
        .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
        .push_next(&mut av1);
    f(&profile)
}

pub fn av1_capabilities(video: &VideoDevice) -> Result<Av1Capabilities, String> {
    with_profile(|profile| {
        let mut av1 = vk::VideoDecodeAV1CapabilitiesKHR::default();
        let mut decode = vk::VideoDecodeCapabilitiesKHR::default();
        let mut capabilities = vk::VideoCapabilitiesKHR::default()
            .push_next(&mut decode)
            .push_next(&mut av1);
        // SAFETY: a live physical device; the profile and the output chain
        // are valid for the call.
        unsafe {
            (video
                .video_instance
                .fp()
                .get_physical_device_video_capabilities_khr)(
                video.physical,
                profile,
                &mut capabilities,
            )
        }
        .result()
        .map_err(vk_error("AV1 decode capabilities"))?;
        let mut result = Av1Capabilities {
            min_extent: capabilities.min_coded_extent,
            max_extent: capabilities.max_coded_extent,
            max_dpb_slots: capabilities.max_dpb_slots,
            max_active_references: capabilities.max_active_reference_pictures,
            bitstream_offset_alignment: capabilities.min_bitstream_buffer_offset_alignment,
            bitstream_size_alignment: capabilities.min_bitstream_buffer_size_alignment,
            picture_access_granularity: capabilities.picture_access_granularity,
            coincide: false,
            distinct: false,
            max_level: 0,
            std_header_version: capabilities.std_header_version,
        };
        let flags = decode.flags;
        result.coincide =
            flags.contains(vk::VideoDecodeCapabilityFlagsKHR::DPB_AND_OUTPUT_COINCIDE);
        result.distinct =
            flags.contains(vk::VideoDecodeCapabilityFlagsKHR::DPB_AND_OUTPUT_DISTINCT);
        result.max_level = av1.max_level as u32;
        Ok(result)
    })
}

/// The formats the driver offers for pictures used as `usage`, best first.
pub fn picture_formats(
    video: &VideoDevice,
    usage: vk::ImageUsageFlags,
) -> Result<Vec<PictureFormat>, String> {
    with_profile(|profile| {
        let profiles = [*profile];
        let mut list = vk::VideoProfileListInfoKHR::default().profiles(&profiles);
        let info = vk::PhysicalDeviceVideoFormatInfoKHR::default()
            .image_usage(usage)
            .push_next(&mut list);
        let query = video
            .video_instance
            .fp()
            .get_physical_device_video_format_properties_khr;
        let mut count = 0;
        // SAFETY: a count query with a valid info chain.
        unsafe { query(video.physical, &info, &mut count, std::ptr::null_mut()) }
            .result()
            .map_err(vk_error("AV1 picture formats"))?;
        let mut properties = vec![vk::VideoFormatPropertiesKHR::default(); count as usize];
        // SAFETY: `properties` holds `count` entries.
        unsafe { query(video.physical, &info, &mut count, properties.as_mut_ptr()) }
            .result()
            .map_err(vk_error("AV1 picture formats"))?;
        properties.truncate(count as usize);
        Ok(properties
            .iter()
            .map(|property| PictureFormat {
                format: property.format,
                create_flags: property.image_create_flags,
                tiling: property.image_tiling,
                usage: property.image_usage_flags,
            })
            .collect())
    })
}

/// One frame to decode, as the parser describes it.
pub struct Av1Frame<'a> {
    /// The frame's OBUs (the frame header and its tile groups), placed in
    /// the bitstream buffer as given.
    pub bitstream: &'a [u8],
    /// Offsets into `bitstream`.
    pub frame_header_offset: u32,
    pub tile_offsets: &'a [u32],
    pub tile_sizes: &'a [u32],
    /// The frame's size (FrameWidth, FrameHeight: before superres upscaling
    /// the decoder applies itself).
    pub width: u32,
    pub height: u32,
    /// Its subordinate pointers must stay valid for the call.
    pub picture: &'a StdVideoDecodeAV1PictureInfo,
    /// The DPB slot each of LAST..ALTREF reads (-1: none).
    pub reference_name_slot_indices: [i32; 7],
    /// Every DPB slot `reference_name_slot_indices` names, with the
    /// reference information the parser keeps for it.
    pub references: &'a [(u32, StdVideoDecodeAV1ReferenceInfo)],
    /// The slot this picture is decoded into, and its reference information
    /// there. It becomes the picture later frames (and `read_nv12`) read.
    pub slot: u32,
    pub slot_info: StdVideoDecodeAV1ReferenceInfo,
}

/// DPB slots: AV1's eight reference frames and the picture being decoded.
pub const SLOTS: u32 = 9;

/// The planes of a picture in NV12 as the guest stores it: Y rows, then
/// interleaved UV rows, both `pitch` bytes apart.
pub fn nv12_size(height: u32, pitch: u32) -> usize {
    pitch as usize * (height as usize + height.div_ceil(2) as usize)
}

/// An AV1 decode session: the DPB (one image, a layer per slot, that
/// decoded pictures are also written to), the bitstream buffer and a
/// read-back buffer. Decoding is synchronous: each call waits for the GPU.
pub struct Av1Decoder {
    video: Arc<VideoDevice>,
    extent: vk::Extent2D,
    session: vk::VideoSessionKHR,
    session_memory: Vec<vk::DeviceMemory>,
    parameters: vk::VideoSessionParametersKHR,
    pictures: vk::Image,
    pictures_memory: vk::DeviceMemory,
    view: vk::ImageView,
    bitstream: Mapped,
    size_alignment: u64,
    readback: Mapped,
    decode_pool: vk::CommandPool,
    transfer_pool: vk::CommandPool,
    decode_command: vk::CommandBuffer,
    transfer_command: vk::CommandBuffer,
    fence: vk::Fence,
    /// No picture was decoded yet: the DPB layers are uninitialized and the
    /// session needs a reset.
    fresh: bool,
    /// Each slot's picture size, once a picture is decoded into it.
    decoded: [Option<vk::Extent2D>; SLOTS as usize],
}

/// A host-visible buffer, persistently mapped.
struct Mapped {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    pointer: *mut u8,
    size: u64,
    coherent: bool,
}

impl Mapped {
    const NONE: Self = Self {
        buffer: vk::Buffer::null(),
        memory: vk::DeviceMemory::null(),
        pointer: std::ptr::null_mut(),
        size: 0,
        coherent: true,
    };

    fn new(
        video: &VideoDevice,
        size: u64,
        usage: vk::BufferUsageFlags,
        read: bool,
    ) -> Result<Self, String> {
        let buffer = with_profile(|profile| {
            let profiles = [*profile];
            let mut list = vk::VideoProfileListInfoKHR::default().profiles(&profiles);
            let mut info = vk::BufferCreateInfo::default().size(size).usage(usage);
            if usage.contains(vk::BufferUsageFlags::VIDEO_DECODE_SRC_KHR) {
                info = info.push_next(&mut list);
            }
            // SAFETY: valid create info; destroyed by `destroy`.
            unsafe { video.device.create_buffer(&info, None) }
        })
        .map_err(vk_error("vkCreateBuffer (video)"))?;
        let mut mapped = Self {
            buffer,
            ..Self::NONE
        };
        // SAFETY: a live buffer.
        let requirements = unsafe { video.device.get_buffer_memory_requirements(buffer) };
        let visible = vk::MemoryPropertyFlags::HOST_VISIBLE;
        let coherent = vk::MemoryPropertyFlags::HOST_COHERENT;
        let cached = vk::MemoryPropertyFlags::HOST_CACHED;
        let bits = requirements.memory_type_bits;
        // Read-backs want cached memory; uploads want coherent memory.
        let index = if read {
            video
                .memory_type(bits, visible | cached)
                .or_else(|| video.memory_type(bits, visible))
        } else {
            video
                .memory_type(bits, visible | coherent)
                .or_else(|| video.memory_type(bits, visible))
        };
        let Some(index) = index else {
            mapped.destroy(video);
            return Err("no host-visible memory for a video buffer".into());
        };
        mapped.coherent = video.memory.memory_types[index as usize]
            .property_flags
            .contains(coherent);
        let info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(index);
        // SAFETY: valid allocate info; the memory is bound, mapped and freed
        // with the buffer.
        let result = unsafe {
            video
                .device
                .allocate_memory(&info, None)
                .and_then(|memory| {
                    mapped.memory = memory;
                    video.device.bind_buffer_memory(buffer, memory, 0)?;
                    video
                        .device
                        .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                })
        };
        match result {
            Ok(pointer) => {
                mapped.pointer = pointer.cast();
                mapped.size = size;
                Ok(mapped)
            }
            Err(error) => {
                mapped.destroy(video);
                Err(format!("video buffer memory: {error}"))
            }
        }
    }

    fn destroy(&mut self, video: &VideoDevice) {
        // SAFETY: no GPU work using the buffer is pending (decoding waits
        // for every submission); freeing memory unmaps it.
        unsafe {
            if self.buffer != vk::Buffer::null() {
                video.device.destroy_buffer(self.buffer, None);
            }
            if self.memory != vk::DeviceMemory::null() {
                video.device.free_memory(self.memory, None);
            }
        }
        *self = Self::NONE;
    }
}

// SAFETY: the mapped pointers are used only through `&mut Av1Decoder`.
unsafe impl Send for Av1Decoder {}

impl Av1Decoder {
    /// A session for pictures up to `width` x `height`.
    pub fn new(video: Arc<VideoDevice>, width: u32, height: u32) -> Result<Self, String> {
        let capabilities = av1_capabilities(&video)?;
        if !capabilities.coincide {
            return Err("the device cannot decode into its DPB (DPB_AND_OUTPUT_COINCIDE)".into());
        }
        if capabilities.max_dpb_slots < SLOTS || capabilities.max_active_references < 7 {
            return Err(format!(
                "the device keeps {} DPB slots and {} references; AV1 needs {SLOTS} and 7",
                capabilities.max_dpb_slots, capabilities.max_active_references
            ));
        }
        let granularity = capabilities.picture_access_granularity;
        let extent = vk::Extent2D {
            width: width.next_multiple_of(granularity.width.max(1)),
            height: height.next_multiple_of(granularity.height.max(1)),
        };
        let max = capabilities.max_extent;
        if extent.width > max.width || extent.height > max.height {
            return Err(format!(
                "{width}x{height} exceeds the decoder's {}x{}",
                max.width, max.height
            ));
        }
        let usage =
            vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR | vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR;
        let format = *picture_formats(&video, usage)?
            .first()
            .ok_or("no picture format decodes AV1 into the DPB")?;
        let mut decoder = Self {
            extent,
            session: vk::VideoSessionKHR::null(),
            session_memory: Vec::new(),
            parameters: vk::VideoSessionParametersKHR::null(),
            pictures: vk::Image::null(),
            pictures_memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            bitstream: Mapped::NONE,
            size_alignment: capabilities.bitstream_size_alignment.max(1),
            readback: Mapped::NONE,
            decode_pool: vk::CommandPool::null(),
            transfer_pool: vk::CommandPool::null(),
            decode_command: vk::CommandBuffer::null(),
            transfer_command: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            fresh: true,
            decoded: [None; SLOTS as usize],
            video,
        };
        decoder.create(format, &capabilities)?;
        Ok(decoder)
    }

    /// Creates the session and its resources (`Drop` releases whatever was
    /// made if this fails part way).
    fn create(
        &mut self,
        format: PictureFormat,
        capabilities: &Av1Capabilities,
    ) -> Result<(), String> {
        let video = self.video.clone();
        let device = &video.device;
        let functions = video.video.fp();
        self.session = with_profile(|profile| {
            let info = vk::VideoSessionCreateInfoKHR::default()
                .queue_family_index(video.decode_family)
                .video_profile(profile)
                .picture_format(format.format)
                .max_coded_extent(self.extent)
                .reference_picture_format(format.format)
                .max_dpb_slots(SLOTS)
                .max_active_reference_pictures(7)
                .std_header_version(&capabilities.std_header_version);
            let mut session = vk::VideoSessionKHR::null();
            // SAFETY: valid create info; destroyed in Drop.
            unsafe {
                (functions.create_video_session_khr)(
                    device.handle(),
                    &info,
                    std::ptr::null(),
                    &mut session,
                )
            }
            .result_with_success(session)
        })
        .map_err(vk_error("vkCreateVideoSessionKHR"))?;

        let mut count = 0;
        // SAFETY: a count query on a live session.
        unsafe {
            (functions.get_video_session_memory_requirements_khr)(
                device.handle(),
                self.session,
                &mut count,
                std::ptr::null_mut(),
            )
        }
        .result()
        .map_err(vk_error("vkGetVideoSessionMemoryRequirementsKHR"))?;
        let mut requirements =
            vec![vk::VideoSessionMemoryRequirementsKHR::default(); count as usize];
        // SAFETY: `requirements` holds `count` entries.
        unsafe {
            (functions.get_video_session_memory_requirements_khr)(
                device.handle(),
                self.session,
                &mut count,
                requirements.as_mut_ptr(),
            )
        }
        .result()
        .map_err(vk_error("vkGetVideoSessionMemoryRequirementsKHR"))?;
        let mut binds = Vec::new();
        for requirement in &requirements[..count as usize] {
            let memory = video.allocate_local(requirement.memory_requirements)?;
            self.session_memory.push(memory);
            binds.push(
                vk::BindVideoSessionMemoryInfoKHR::default()
                    .memory_bind_index(requirement.memory_bind_index)
                    .memory(memory)
                    .memory_size(requirement.memory_requirements.size),
            );
        }
        // SAFETY: each bind names fresh memory of the size asked for.
        unsafe {
            (functions.bind_video_session_memory_khr)(
                device.handle(),
                self.session,
                binds.len() as u32,
                binds.as_ptr(),
            )
        }
        .result()
        .map_err(vk_error("vkBindVideoSessionMemoryKHR"))?;

        // The DPB: one layer per slot, also read back by the copy queue.
        let families = [video.decode_family, video.transfer_family];
        let concurrent = families[0] != families[1];
        self.pictures = with_profile(|profile| {
            let profiles = [*profile];
            let mut list = vk::VideoProfileListInfoKHR::default().profiles(&profiles);
            let mut info = vk::ImageCreateInfo::default()
                .flags(format.create_flags)
                .image_type(vk::ImageType::TYPE_2D)
                .format(format.format)
                .extent(vk::Extent3D {
                    width: self.extent.width,
                    height: self.extent.height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(SLOTS)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(format.tiling)
                .usage(
                    vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR
                        | vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR
                        | vk::ImageUsageFlags::TRANSFER_SRC,
                )
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .push_next(&mut list);
            if concurrent {
                info = info
                    .sharing_mode(vk::SharingMode::CONCURRENT)
                    .queue_family_indices(&families);
            }
            // SAFETY: valid create info; destroyed in Drop.
            unsafe { device.create_image(&info, None) }
        })
        .map_err(vk_error("vkCreateImage (DPB)"))?;
        // SAFETY: a live image.
        let requirements = unsafe { device.get_image_memory_requirements(self.pictures) };
        self.pictures_memory = video.allocate_local(requirements)?;
        // SAFETY: fresh memory of the size the image asks for.
        unsafe { device.bind_image_memory(self.pictures, self.pictures_memory, 0) }
            .map_err(vk_error("vkBindImageMemory (DPB)"))?;
        let mut view_usage = vk::ImageViewUsageCreateInfo::default().usage(
            vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR | vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR,
        );
        let info = vk::ImageViewCreateInfo::default()
            .image(self.pictures)
            .view_type(vk::ImageViewType::TYPE_2D_ARRAY)
            .format(format.format)
            .subresource_range(color_layers(0, SLOTS))
            .push_next(&mut view_usage);
        // SAFETY: valid create info; destroyed in Drop.
        self.view = unsafe { device.create_image_view(&info, None) }
            .map_err(vk_error("vkCreateImageView (DPB)"))?;

        self.readback = Mapped::new(
            &video,
            nv12_size(self.extent.height, self.extent.width.next_multiple_of(2)) as u64,
            vk::BufferUsageFlags::TRANSFER_DST,
            true,
        )?;

        for (family, pool, command) in [
            (
                video.decode_family,
                &mut self.decode_pool,
                &mut self.decode_command,
            ),
            (
                video.transfer_family,
                &mut self.transfer_pool,
                &mut self.transfer_command,
            ),
        ] {
            let info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            // SAFETY: valid create info; destroyed in Drop.
            *pool = unsafe { device.create_command_pool(&info, None) }
                .map_err(vk_error("vkCreateCommandPool (video)"))?;
            let info = vk::CommandBufferAllocateInfo::default()
                .command_pool(*pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            // SAFETY: a live pool.
            *command = unsafe { device.allocate_command_buffers(&info) }
                .map_err(vk_error("vkAllocateCommandBuffers (video)"))?[0];
        }
        // SAFETY: valid create info; destroyed in Drop.
        self.fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(vk_error("vkCreateFence (video)"))?;
        Ok(())
    }

    /// Starts decoding a coded video sequence: its sequence header becomes
    /// the session parameters for the frames that follow. Pointers inside
    /// `header` need only be valid for the call.
    pub fn set_sequence(&mut self, header: &StdVideoAV1SequenceHeader) -> Result<(), String> {
        let device = &self.video.device;
        let functions = self.video.video.fp();
        let mut av1 =
            vk::VideoDecodeAV1SessionParametersCreateInfoKHR::default().std_sequence_header(header);
        let info = vk::VideoSessionParametersCreateInfoKHR::default()
            .video_session(self.session)
            .push_next(&mut av1);
        let mut parameters = vk::VideoSessionParametersKHR::null();
        // SAFETY: valid create info on a live session.
        unsafe {
            (functions.create_video_session_parameters_khr)(
                device.handle(),
                &info,
                std::ptr::null(),
                &mut parameters,
            )
        }
        .result()
        .map_err(vk_error("vkCreateVideoSessionParametersKHR"))?;
        if self.parameters != vk::VideoSessionParametersKHR::null() {
            // SAFETY: no decode using the old parameters is pending.
            unsafe {
                (functions.destroy_video_session_parameters_khr)(
                    device.handle(),
                    self.parameters,
                    std::ptr::null(),
                )
            };
        }
        self.parameters = parameters;
        Ok(())
    }

    fn picture_resource(
        &self,
        slot: u32,
        extent: vk::Extent2D,
    ) -> vk::VideoPictureResourceInfoKHR<'static> {
        vk::VideoPictureResourceInfoKHR::default()
            .coded_extent(extent)
            .base_array_layer(slot)
            .image_view_binding(self.view)
    }

    /// Decodes `frame` into its slot and waits for it.
    pub fn decode(&mut self, frame: &Av1Frame) -> Result<(), String> {
        if self.parameters == vk::VideoSessionParametersKHR::null() {
            return Err("AV1 frame before a sequence header".into());
        }
        if frame.slot >= SLOTS
            || frame.width > self.extent.width
            || frame.height > self.extent.height
        {
            return Err(format!(
                "AV1 frame {}x{} in slot {} does not fit the session ({}x{})",
                frame.width, frame.height, frame.slot, self.extent.width, self.extent.height
            ));
        }
        if frame.tile_offsets.len() != frame.tile_sizes.len() {
            return Err("AV1 frame tile offsets and sizes differ in number".into());
        }
        let extent = vk::Extent2D {
            width: frame.width,
            height: frame.height,
        };
        let mut references = Vec::with_capacity(frame.references.len());
        for (slot, info) in frame.references {
            let decoded = self
                .decoded
                .get(*slot as usize)
                .copied()
                .flatten()
                .ok_or_else(|| format!("AV1 frame references empty DPB slot {slot}"))?;
            if *slot == frame.slot {
                return Err(format!(
                    "AV1 frame decodes into slot {slot}, which it references"
                ));
            }
            references.push((*slot, *info, decoded));
        }

        // The bitstream, padded to the size alignment.
        let range = (frame.bitstream.len() as u64)
            .max(1)
            .next_multiple_of(self.size_alignment);
        if range > self.bitstream.size {
            let video = self.video.clone();
            self.bitstream.destroy(&video);
            self.bitstream = Mapped::new(
                &video,
                range.next_power_of_two().max(1 << 20),
                vk::BufferUsageFlags::VIDEO_DECODE_SRC_KHR,
                false,
            )?;
        }
        // SAFETY: the mapping holds at least `range` bytes and no GPU work
        // reads it now.
        unsafe {
            let pointer = self.bitstream.pointer;
            std::ptr::copy_nonoverlapping(frame.bitstream.as_ptr(), pointer, frame.bitstream.len());
            std::ptr::write_bytes(
                pointer.add(frame.bitstream.len()),
                0,
                range as usize - frame.bitstream.len(),
            );
        }
        if !self.bitstream.coherent {
            let flush = vk::MappedMemoryRange::default()
                .memory(self.bitstream.memory)
                .size(vk::WHOLE_SIZE);
            // SAFETY: the memory is mapped.
            unsafe { self.video.device.flush_mapped_memory_ranges(&[flush]) }
                .map_err(vk_error("vkFlushMappedMemoryRanges (video)"))?;
        }

        let resources: Vec<_> = references
            .iter()
            .map(|(slot, _, decoded)| self.picture_resource(*slot, *decoded))
            .collect();
        let target = self.picture_resource(frame.slot, extent);
        let mut slot_infos: Vec<_> = references
            .iter()
            .map(|(_, info, _)| {
                vk::VideoDecodeAV1DpbSlotInfoKHR::default().std_reference_info(info)
            })
            .collect();
        let mut setup_info =
            vk::VideoDecodeAV1DpbSlotInfoKHR::default().std_reference_info(&frame.slot_info);
        let slots: Vec<_> = references
            .iter()
            .zip(&resources)
            .zip(slot_infos.iter_mut())
            .map(|(((slot, _, _), resource), info)| {
                vk::VideoReferenceSlotInfoKHR::default()
                    .slot_index(*slot as i32)
                    .picture_resource(resource)
                    .push_next(info)
            })
            .collect();
        let setup = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(frame.slot as i32)
            .picture_resource(&target)
            .push_next(&mut setup_info);
        // Coding begins with the references bound and the target to be
        // activated (slot -1).
        let mut bound: Vec<_> = references
            .iter()
            .zip(&resources)
            .map(|((slot, _, _), resource)| {
                vk::VideoReferenceSlotInfoKHR::default()
                    .slot_index(*slot as i32)
                    .picture_resource(resource)
            })
            .collect();
        bound.push(
            vk::VideoReferenceSlotInfoKHR::default()
                .slot_index(-1)
                .picture_resource(&target),
        );
        let mut picture = vk::VideoDecodeAV1PictureInfoKHR::default()
            .std_picture_info(frame.picture)
            .reference_name_slot_indices(frame.reference_name_slot_indices)
            .frame_header_offset(frame.frame_header_offset)
            .tile_offsets(frame.tile_offsets)
            .tile_sizes(frame.tile_sizes);
        let decode = vk::VideoDecodeInfoKHR::default()
            .src_buffer(self.bitstream.buffer)
            .src_buffer_range(range)
            .dst_picture_resource(target)
            .setup_reference_slot(&setup)
            .reference_slots(&slots)
            .push_next(&mut picture);
        let begin = vk::VideoBeginCodingInfoKHR::default()
            .video_session(self.session)
            .video_session_parameters(self.parameters)
            .reference_slots(&bound);

        let device = &self.video.device;
        let functions = self.video.video.fp();
        let command = self.decode_command;
        // SAFETY: the command buffer is idle (every submission is waited
        // for); everything recorded is live and the structures above
        // outlive the recording.
        unsafe {
            device
                .begin_command_buffer(
                    command,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(vk_error("vkBeginCommandBuffer (video)"))?;
            if self.fresh {
                let barrier = [vk::ImageMemoryBarrier2::default()
                    .dst_stage_mask(vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
                    .dst_access_mask(
                        vk::AccessFlags2::VIDEO_DECODE_READ_KHR
                            | vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
                    )
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::VIDEO_DECODE_DPB_KHR)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(self.pictures)
                    .subresource_range(color_layers(0, SLOTS))];
                device.cmd_pipeline_barrier2(
                    command,
                    &vk::DependencyInfo::default().image_memory_barriers(&barrier),
                );
            }
            (functions.cmd_begin_video_coding_khr)(command, &begin);
            if self.fresh {
                let control = vk::VideoCodingControlInfoKHR::default()
                    .flags(vk::VideoCodingControlFlagsKHR::RESET);
                (functions.cmd_control_video_coding_khr)(command, &control);
            }
            (self.video.decode.fp().cmd_decode_video_khr)(command, &decode);
            (functions.cmd_end_video_coding_khr)(command, &vk::VideoEndCodingInfoKHR::default());
            device
                .end_command_buffer(command)
                .map_err(vk_error("vkEndCommandBuffer (video)"))?;
        }
        self.video
            .submit(&self.video.decode_queue, command, self.fence)?;
        self.fresh = false;
        self.decoded[frame.slot as usize] = Some(extent);
        Ok(())
    }

    /// The size of the picture in `slot`, if one was decoded there.
    pub fn picture_size(&self, slot: u32) -> Option<(u32, u32)> {
        self.decoded
            .get(slot as usize)
            .copied()
            .flatten()
            .map(|extent| (extent.width, extent.height))
    }

    /// Copies the picture in `slot` into `out` as NV12 with rows `pitch`
    /// bytes apart (`nv12_size(height, pitch)` bytes).
    pub fn read_nv12(&mut self, slot: u32, pitch: u32, out: &mut [u8]) -> Result<(), String> {
        let extent = self
            .decoded
            .get(slot as usize)
            .copied()
            .flatten()
            .ok_or_else(|| format!("no picture in DPB slot {slot}"))?;
        if pitch < extent.width.next_multiple_of(2) || pitch % 2 != 0 {
            return Err(format!(
                "NV12 pitch {pitch} is too small for width {}",
                extent.width
            ));
        }
        let size = nv12_size(extent.height, pitch);
        if out.len() < size {
            return Err(format!("NV12 output holds {} of {size} bytes", out.len()));
        }
        if size as u64 > self.readback.size {
            let video = self.video.clone();
            self.readback.destroy(&video);
            self.readback = Mapped::new(
                &video,
                size as u64,
                vk::BufferUsageFlags::TRANSFER_DST,
                true,
            )?;
        }
        let layer = |aspect| {
            vk::ImageSubresourceLayers::default()
                .aspect_mask(aspect)
                .base_array_layer(slot)
                .layer_count(1)
        };
        let regions = [
            vk::BufferImageCopy::default()
                .buffer_row_length(pitch)
                .image_subresource(layer(vk::ImageAspectFlags::PLANE_0))
                .image_extent(vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                }),
            // The UV plane's texels are two bytes.
            vk::BufferImageCopy::default()
                .buffer_offset(pitch as u64 * extent.height as u64)
                .buffer_row_length(pitch / 2)
                .image_subresource(layer(vk::ImageAspectFlags::PLANE_1))
                .image_extent(vk::Extent3D {
                    width: extent.width.div_ceil(2),
                    height: extent.height.div_ceil(2),
                    depth: 1,
                }),
        ];
        let transition = |old, new| {
            vk::ImageMemoryBarrier2::default()
                .old_layout(old)
                .new_layout(new)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(self.pictures)
                .subresource_range(color_layers(slot, 1))
        };
        // The decode that wrote the picture was waited for, so the copy
        // needs only the layout changes around it.
        let to_copy = [transition(
            vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        )
        .dst_stage_mask(vk::PipelineStageFlags2::COPY)
        .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)];
        let back = [transition(
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
        )
        .src_stage_mask(vk::PipelineStageFlags2::COPY)];
        let host = [vk::BufferMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(self.readback.buffer)
            .size(vk::WHOLE_SIZE)];
        let device = &self.video.device;
        let command = self.transfer_command;
        // SAFETY: the command buffer is idle; the image and buffer are live.
        unsafe {
            device
                .begin_command_buffer(
                    command,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(vk_error("vkBeginCommandBuffer (video)"))?;
            device.cmd_pipeline_barrier2(
                command,
                &vk::DependencyInfo::default().image_memory_barriers(&to_copy),
            );
            device.cmd_copy_image_to_buffer(
                command,
                self.pictures,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.readback.buffer,
                &regions,
            );
            device.cmd_pipeline_barrier2(
                command,
                &vk::DependencyInfo::default()
                    .image_memory_barriers(&back)
                    .buffer_memory_barriers(&host),
            );
            device
                .end_command_buffer(command)
                .map_err(vk_error("vkEndCommandBuffer (video)"))?;
        }
        self.video
            .submit(&self.video.transfer_queue, command, self.fence)?;
        if !self.readback.coherent {
            let range = vk::MappedMemoryRange::default()
                .memory(self.readback.memory)
                .size(vk::WHOLE_SIZE);
            // SAFETY: the memory is mapped.
            unsafe { device.invalidate_mapped_memory_ranges(&[range]) }
                .map_err(vk_error("vkInvalidateMappedMemoryRanges (video)"))?;
        }
        // SAFETY: the copy finished; the mapping holds `size` bytes.
        let picture = unsafe { std::slice::from_raw_parts(self.readback.pointer, size) };
        out[..size].copy_from_slice(picture);
        Ok(())
    }
}

fn color_layers(base: u32, count: u32) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .base_array_layer(base)
        .layer_count(count)
}

impl Drop for Av1Decoder {
    fn drop(&mut self) {
        let video = self.video.clone();
        let device = &video.device;
        let functions = video.video.fp();
        self.bitstream.destroy(&video);
        self.readback.destroy(&video);
        // SAFETY: every submission was waited for, so nothing below is in
        // use; null handles are skipped.
        unsafe {
            if self.fence != vk::Fence::null() {
                device.destroy_fence(self.fence, None);
            }
            for pool in [self.decode_pool, self.transfer_pool] {
                if pool != vk::CommandPool::null() {
                    device.destroy_command_pool(pool, None);
                }
            }
            if self.view != vk::ImageView::null() {
                device.destroy_image_view(self.view, None);
            }
            if self.pictures != vk::Image::null() {
                device.destroy_image(self.pictures, None);
            }
            if self.pictures_memory != vk::DeviceMemory::null() {
                device.free_memory(self.pictures_memory, None);
            }
            if self.parameters != vk::VideoSessionParametersKHR::null() {
                (functions.destroy_video_session_parameters_khr)(
                    device.handle(),
                    self.parameters,
                    std::ptr::null(),
                );
            }
            if self.session != vk::VideoSessionKHR::null() {
                (functions.destroy_video_session_khr)(
                    device.handle(),
                    self.session,
                    std::ptr::null(),
                );
            }
            for memory in self.session_memory.drain(..) {
                device.free_memory(memory, None);
            }
        }
    }
}
