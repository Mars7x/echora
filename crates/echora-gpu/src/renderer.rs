//! Executes NVN work on the host GPU.
//!
//! Guest render targets are Vulkan images keyed by the guest texture's
//! address. Work is recorded into one command buffer and submitted when the
//! NVN layer needs the results (a present, a readback, a fence), then waited
//! for: simple and ordered, not fast.

use std::collections::HashMap;

use ash::vk;

use crate::draw::ImageShape;
use crate::vulkan::Gpu;

pub struct Target {
    pub image: vk::Image,
    pub view: vk::ImageView,
    /// The view shaders sample: `view`, or for depth-stencil formats a
    /// depth-only view (a sampled view may name one aspect only).
    pub sampled_view: vk::ImageView,
    /// Dimensionality and mip levels the image was created with.
    pub shape: crate::draw::ImageShape,
    pub levels: u32,
    memory: vk::DeviceMemory,
    pub format: vk::Format,
    /// The guest's extent; the image is `scale` times as wide and high.
    pub width: u32,
    pub height: u32,
    /// Resolution scale (none, or the renderer's scale for render targets).
    pub scale: Scale,
    /// Layout the image is in after the commands recorded so far.
    layout: vk::ImageLayout,
}

impl Target {
    /// The image's extent at mip `level`.
    pub fn extent(&self, level: u32) -> (u32, u32) {
        (
            (self.scale.apply(self.width) >> level).max(1),
            (self.scale.apply(self.height) >> level).max(1),
        )
    }

    /// `rect` (x, y, width, height in guest texels) in image texels.
    fn scaled(&self, rect: [i32; 4]) -> [i32; 4] {
        self.scale.rect(rect)
    }
}

/// A resolution scale, numerator over denominator in lowest terms: image
/// texels per guest texel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scale {
    numerator: u32,
    denominator: u32,
}

impl Scale {
    pub const ONE: Self = Self {
        numerator: 1,
        denominator: 1,
    };

    /// `numerator / denominator`, reduced; None outside 1/2 to 4.
    pub fn new(numerator: u32, denominator: u32) -> Option<Self> {
        if denominator == 0 || numerator * 2 < denominator || numerator > denominator * 4 {
            return None;
        }
        let (mut a, mut b) = (numerator, denominator);
        while b != 0 {
            (a, b) = (b, a % b);
        }
        Some(Self {
            numerator: numerator / a,
            denominator: denominator / a,
        })
    }

    /// The scale chosen by ECHORA_RESOLUTION=WIDTHxHEIGHT (its height over
    /// the console's 1080 rows: 2560x1440 is 4/3) or ECHORA_RESOLUTION_SCALE
    /// (an integer, or N/D); none by default or when out of range.
    pub fn from_env() -> Self {
        Self::env_override().unwrap_or(Self::ONE)
    }

    /// The scale the environment asks for, if it asks.
    pub fn env_override() -> Option<Self> {
        let resolution = std::env::var("ECHORA_RESOLUTION").ok().and_then(|value| {
            let (_, height) = value.trim().split_once(['x', 'X'])?;
            Self::new(height.trim().parse().ok()?, 1080)
        });
        let factor = || {
            let value = std::env::var("ECHORA_RESOLUTION_SCALE").ok()?;
            let value = value.trim().trim_end_matches('x');
            match value.split_once('/') {
                Some((numerator, denominator)) => {
                    Self::new(numerator.parse().ok()?, denominator.parse().ok()?)
                }
                None => Self::new(value.parse().ok()?, 1),
            }
        };
        resolution.or_else(factor)
    }

    pub fn is_one(self) -> bool {
        self == Self::ONE
    }

    /// (numerator, denominator), as the shader translator takes them.
    pub fn terms(self) -> (u32, u32) {
        (self.numerator, self.denominator)
    }

    pub fn factor(self) -> f32 {
        self.numerator as f32 / self.denominator as f32
    }

    /// A guest extent in image texels.
    pub fn apply(self, value: u32) -> u32 {
        let scaled = u64::from(value) * u64::from(self.numerator) / u64::from(self.denominator);
        (scaled as u32).max(u32::from(value > 0))
    }

    /// A guest coordinate in image texels (rounded down, as the shaders'
    /// rescaled texel fetches are).
    fn coordinate(self, value: i32) -> i32 {
        (i64::from(value) * i64::from(self.numerator))
            .div_euclid(i64::from(self.denominator))
            .clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
    }

    /// A rect (x, y, width, height) in image texels: its edges scaled, so
    /// neighbouring rects still meet.
    pub fn rect(self, [x, y, width, height]: [i32; 4]) -> [i32; 4] {
        let (left, top) = (self.coordinate(x), self.coordinate(y));
        let right = self.coordinate(x.saturating_add(width));
        let bottom = self.coordinate(y.saturating_add(height));
        [left, top, right - left, bottom - top]
    }
}

/// Allocations at least this large are logged with their caller.
const LARGE_ALLOCATION: u64 = 512 << 20;

/// ECHORA_ALLOC_LOG: running totals of device allocations by caller.
fn log_allocation(caller: &'static std::panic::Location<'static>, size: u64) {
    use std::collections::HashMap;
    use std::sync::Mutex;
    static TOTALS: Mutex<(u64, Option<HashMap<String, (u64, u64)>>)> = Mutex::new((0, None));
    let mut guard = TOTALS.lock().unwrap_or_else(|poison| poison.into_inner());
    let (sum, totals) = &mut *guard;
    let totals = totals.get_or_insert_with(HashMap::new);
    let entry = totals.entry(caller.to_string()).or_default();
    entry.0 += 1;
    entry.1 += size;
    let before = *sum >> 30;
    *sum += size;
    if *sum >> 30 > before {
        let mut ranked: Vec<_> = totals.iter().collect();
        ranked.sort_by_key(|(_, (_, bytes))| std::cmp::Reverse(*bytes));
        let top: Vec<String> = ranked
            .iter()
            .take(6)
            .map(|(site, (count, bytes))| format!("{site} {} MiB/{count}", bytes >> 20))
            .collect();
        eprintln!("gpu alloc: {} MiB total; {}", *sum >> 20, top.join(", "));
    }
}
/// The largest span `import_file` shares: udmabuf's default size limit.
const MAX_SHARED_SPAN: u64 = 64 << 20;
/// Guest memory pinned by shared spans before the least recently used are
/// dropped.
const MAX_SHARED_BYTES: u64 = 1 << 30;
/// The largest texture upload: 1.5 GiB, above the largest the game makes
/// (a 1365x1365, 90-layer ASTC array with 11 levels, 852 MiB as RGBA8).
const MAX_UPLOAD_BYTES: u64 = 1536 << 20;

pub struct Renderer {
    pub gpu: Gpu,
    /// The texture pool's set (`crate::pool`), once a program reads it.
    pub(crate) texture_pool: Option<crate::pool::TexturePool>,
    /// Counts for the runtime's statistics (`take_stats`).
    pub(crate) stats: HashMap<&'static str, u64>,
    /// Debugging buffers shaders write through device addresses
    /// (`debug_buffer`).
    debug_buffers: Vec<HostBuffer>,
    pool: vk::CommandPool,
    commands: vk::CommandBuffer,
    fence: vk::Fence,
    recording: bool,
    memory_types: vk::PhysicalDeviceMemoryProperties,
    targets: HashMap<u64, Target>,
    /// Guest textures uploaded for sampling, with the generation uploaded.
    uploads: HashMap<u64, (u64, Target)>,
    /// Uploads forgotten while recorded work may still sample them:
    /// destroyed after the next flush.
    retired_uploads: Vec<Target>,
    /// Texel buffers made from guest buffer textures (shaders read them as
    /// samplerBuffer), by upload key: the generation copied, the buffer and
    /// its view.
    texel_buffers: HashMap<u64, (u64, HostBuffer, vk::BufferView)>,
    /// Texel buffers replaced while recorded work may still read them.
    retired_texel_buffers: Vec<(HostBuffer, vk::BufferView)>,
    /// Debugging (ECHORA_LAST_SUBMIT / ECHORA_SLOW_SUBMITS files): what
    /// the current submission records, to find work that stalls the GPU.
    labels: Vec<String>,
    /// Draws and dispatches recorded since the last submission.
    work_items: u32,
    samplers: HashMap<SamplerKey, vk::Sampler>,
    /// Views of another type than an image's own (a 2D-array view of a 2D
    /// image, ...), by image and view type.
    extra_views: HashMap<(vk::Image, i32), vk::ImageView>,
    /// Host memory imported as storage buffers (VK_EXT_external_memory_host),
    /// by page-aligned host address and length.
    imported: HashMap<(usize, u64), (vk::Buffer, vk::DeviceMemory)>,
    /// The GPU page table (`set_page_table`): its buffer and the entries
    /// written there.
    page_table: Option<(HostBuffer, Vec<u64>)>,
    /// Spans the driver refused to import (file-backed memory, on drivers
    /// that only import anonymous memory): not tried again.
    import_refused: std::collections::HashSet<(usize, u64)>,
    /// Spans of guest memory's file shared as dma-bufs (see `import_file`),
    /// by page-aligned file offset and length, oldest use first.
    shared: Vec<((u64, u64), vk::Buffer, vk::DeviceMemory)>,
    /// Shared spans the GPU page table points into (`address_span`): kept
    /// until their file range is released, never evicted while the table
    /// may use them. Start, span, buffer, memory and device address.
    addressed: Vec<(u64, u64, vk::Buffer, vk::DeviceMemory, u64)>,
    /// Bytes `shared` pins in host memory (udmabuf pins its pages).
    pub shared_bytes: u64,
    /// /dev/udmabuf, opened on first use; None after it failed to open.
    #[cfg(target_os = "linux")]
    udmabuf: Option<Option<std::fs::File>>,
    /// Spans that could not be shared: not tried again.
    shared_refused: std::collections::HashSet<(u64, u64)>,
    /// Shared spans dropped while recorded work may still read them:
    /// destroyed after the next flush.
    retired_shared: Vec<(vk::Buffer, vk::DeviceMemory)>,
    /// Storage buffers kept on the GPU between dispatches (see
    /// `resident_buffer`), by the caller's key: buffer, memory, size.
    resident: HashMap<u64, (vk::Buffer, vk::DeviceMemory, u64)>,
    /// How each sampled view was made (image, type, format, range), so a
    /// swizzled copy of it can be made.
    view_infos: HashMap<vk::ImageView, ViewInfo>,
    /// Targets created since the caller last took them (to initialize).
    fresh_targets: Vec<u64>,
    /// Host-visible ring for per-draw data, reset after each flush.
    ring: HostBuffer,
    ring_used: u64,
    descriptor_pool: vk::DescriptorPool,
    /// Descriptor sets, and descriptors of each pooled type (uniform
    /// buffer, combined image sampler, storage buffer, storage image),
    /// allocated since the pool was last reset.
    sets_used: u32,
    descriptors_used: [u32; 4],
    /// Submissions that waited for the GPU, and the microseconds waited.
    pub flushes: u64,
    /// Ring ranges (token, offset, length) a dispatch wrote, copied out at
    /// the next flush into `completed_reads` (dispatch results come back
    /// when the work is next waited for, not one wait per dispatch).
    pending_reads: Vec<(u64, u64, usize)>,
    /// The rendering pass left open after a draw (color views, depth view,
    /// render area): the next draw into the same attachments continues it
    /// instead of storing and loading them again. Any other recording
    /// (`commands`) ends it.
    open_pass: Option<(Vec<vk::ImageView>, Option<vk::ImageView>, [i32; 4])>,
    completed_reads: Vec<(u64, Vec<u8>)>,
    next_read_token: u64,
    /// The other submission slot: while one slot's work runs on the GPU,
    /// the CPU records into the other (see `flush_async`).
    spare: Slot,
    /// The recorded slot's previous submission has not been waited for.
    current_in_flight: bool,
    /// Labels of the last submission (ECHORA_SLOW_SUBMITS reports them).
    submitted_labels: Vec<String>,
    /// Render targets created, and recreated because their shape changed.
    pub targets_created: u64,
    /// Changes to the set of targets (created, recreated, moved,
    /// removed): an index of them by key is current while this holds.
    pub target_changes: u64,
    pub targets_recreated: u64,
    /// Scaled targets replaced by guest-resolution ones (`keep_unscaled`).
    pub descaled: u64,
    /// Render target resolution scale (`Scale::from_env`).
    scale: Scale,
    /// Targets kept at the guest's resolution: compute passes write them
    /// as storage images, or they were bound with an unscaled target.
    unscaled: std::collections::HashSet<u64>,
    /// Color clears limited by the scissor to part of the target.
    pub partial_clears: u64,
    pub flush_wait_microseconds: u64,
    /// Images rendered to since the last memory barrier, and the
    /// attachments of the last rendering pass: a pass or transfer that
    /// reads one of them (or another pass that writes one) waits for
    /// those writes first.
    written: std::collections::HashSet<vk::Image>,
    last_attachments: Vec<vk::Image>,
    /// Memory barriers recorded for those hazards.
    pub hazard_barriers: u64,
    /// GPU time per submission (ECHORA_GPU_TIME=1).
    timer: Option<crate::timing::GpuTimer>,
    /// The SAMPLES_PASSED counter and its occlusion queries, once used.
    pub(crate) occlusion: Option<crate::occlusion::Occlusion>,
    /// The condition draws and dispatches are gated on (`set_condition`):
    /// wanted, and active in the command buffer being recorded.
    condition: Option<(vk::Buffer, u64, bool)>,
    condition_active: bool,
    /// A failure that must end the run rather than skip one draw (an upload
    /// over MAX_UPLOAD_BYTES): the runtime takes it after each draw and
    /// dispatch and stops with it.
    pub fatal: Option<String>,
    /// Where linked shader objects' driver binaries are kept between runs
    /// (see `crate::shader_cache`), if anywhere.
    pub(crate) shader_cache: Option<std::path::PathBuf>,
}

type SamplerKey = (i32, i32, i32, bool, [i32; 3], [u32; 3]);

/// What one submission owns until the GPU finishes it: its command
/// buffer and fence, the ring its uploads went to, its descriptor pool,
/// the reads deferred from its ring and the resources retired while it
/// was recorded. The renderer's own fields hold the slot being recorded.
struct Slot {
    commands: vk::CommandBuffer,
    fence: vk::Fence,
    ring: HostBuffer,
    ring_used: u64,
    descriptor_pool: vk::DescriptorPool,
    sets_used: u32,
    descriptors_used: [u32; 4],
    pending_reads: Vec<(u64, u64, usize)>,
    retired_uploads: Vec<Target>,
    retired_texel_buffers: Vec<(HostBuffer, vk::BufferView)>,
    retired_shared: Vec<(vk::Buffer, vk::DeviceMemory)>,
    /// Submitted and not yet waited for.
    in_flight: bool,
}

/// Draws and dispatches one submission may hold (see `bound_submission`).
const MAX_WORK_ITEMS: u32 = 1000;

/// Ring size: everything recorded between flushes must fit.
const RING_SIZE: u64 = 64 << 20;
/// Descriptor sets per flush.
const MAX_SETS: u32 = 4096;

/// A descriptor pool for one submission's sets.
fn create_descriptor_pool(device: &ash::Device) -> Result<vk::DescriptorPool, String> {
    let sizes = [
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: MAX_SETS * 8,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: MAX_SETS * 16,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: MAX_SETS * 4,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_IMAGE,
            descriptor_count: MAX_SETS * 4,
        },
        // Counted with the combined image samplers (`reserve`).
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
            descriptor_count: MAX_SETS * 16,
        },
    ];
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(MAX_SETS)
        .pool_sizes(&sizes);
    // SAFETY: valid create info; destroyed in Drop.
    let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None) }
        .map_err(vk_error("vkCreateDescriptorPool"))?;
    Ok(descriptor_pool)
}

pub(crate) fn vk_error(context: &'static str) -> impl Fn(vk::Result) -> String {
    move |error| format!("{context}: {error}")
}

pub fn is_depth(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::D16_UNORM
            | vk::Format::D32_SFLOAT
            | vk::Format::D24_UNORM_S8_UINT
            | vk::Format::D32_SFLOAT_S8_UINT
    )
}

pub fn has_stencil(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::D24_UNORM_S8_UINT | vk::Format::D32_SFLOAT_S8_UINT
    )
}

fn aspect(format: vk::Format) -> vk::ImageAspectFlags {
    if has_stencil(format) {
        vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
    } else if is_depth(format) {
        vk::ImageAspectFlags::DEPTH
    } else {
        vk::ImageAspectFlags::COLOR
    }
}

impl Renderer {
    pub fn new(gpu: Gpu) -> Result<Self, String> {
        let device = &gpu.device;
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(gpu.queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: valid create infos on a live device; destroyed in Drop.
        let pool = unsafe { device.create_command_pool(&pool_info, None) }
            .map_err(vk_error("vkCreateCommandPool"))?;
        let allocate = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: as above.
        let commands = unsafe { device.allocate_command_buffers(&allocate) }
            .map_err(vk_error("vkAllocateCommandBuffers"))?[0];
        // SAFETY: as above.
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(vk_error("vkCreateFence"))?;
        let memory_types = gpu.memory_properties();
        let descriptor_pool = create_descriptor_pool(device)?;
        let spare_allocate = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: valid create infos on a live device; destroyed in Drop.
        let spare_commands = unsafe { device.allocate_command_buffers(&spare_allocate) }
            .map_err(vk_error("vkAllocateCommandBuffers"))?[0];
        // SAFETY: as above.
        let spare_fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(vk_error("vkCreateFence"))?;
        let spare = Slot {
            commands: spare_commands,
            fence: spare_fence,
            ring: HostBuffer::empty(),
            ring_used: 0,
            descriptor_pool: create_descriptor_pool(device)?,
            sets_used: 0,
            descriptors_used: [0; 4],
            pending_reads: Vec::new(),
            retired_uploads: Vec::new(),
            retired_texel_buffers: Vec::new(),
            retired_shared: Vec::new(),
            in_flight: false,
        };
        // SAFETY: the physical device belongs to the live instance.
        let (period, valid_bits) = unsafe {
            let limits = gpu
                .instance
                .get_physical_device_properties(gpu.physical)
                .limits;
            let families = gpu
                .instance
                .get_physical_device_queue_family_properties(gpu.physical);
            let bits = families
                .get(gpu.queue_family as usize)
                .map_or(0, |family| family.timestamp_valid_bits);
            (limits.timestamp_period, bits)
        };
        let timer =
            crate::timing::GpuTimer::from_env(device, period, valid_bits, [fence, spare.fence]);
        let mut renderer = Self {
            texture_pool: None,
            stats: HashMap::new(),
            debug_buffers: Vec::new(),
            timer,
            shader_cache: None,
            fatal: None,
            gpu,
            pool,
            commands,
            fence,
            recording: false,
            memory_types,
            targets: HashMap::new(),
            uploads: HashMap::new(),
            retired_uploads: Vec::new(),
            texel_buffers: HashMap::new(),
            retired_texel_buffers: Vec::new(),
            labels: Vec::new(),
            work_items: 0,
            samplers: HashMap::new(),
            extra_views: HashMap::new(),
            imported: HashMap::new(),
            addressed: Vec::new(),
            page_table: None,
            import_refused: Default::default(),
            shared: Vec::new(),
            shared_bytes: 0,
            #[cfg(target_os = "linux")]
            udmabuf: None,
            shared_refused: Default::default(),
            retired_shared: Vec::new(),
            resident: HashMap::new(),
            view_infos: HashMap::new(),
            fresh_targets: Vec::new(),
            ring: HostBuffer::empty(),
            ring_used: 0,
            descriptor_pool,
            sets_used: 0,
            descriptors_used: [0; 4],
            flushes: 0,
            pending_reads: Vec::new(),
            open_pass: None,
            completed_reads: Vec::new(),
            next_read_token: 0,
            targets_created: 0,
            target_changes: 0,
            targets_recreated: 0,
            descaled: 0,
            scale: Scale::from_env(),
            unscaled: std::collections::HashSet::new(),
            partial_clears: 0,
            flush_wait_microseconds: 0,
            written: std::collections::HashSet::new(),
            last_attachments: Vec::new(),
            hazard_barriers: 0,
            spare,
            current_in_flight: false,
            submitted_labels: Vec::new(),
            occlusion: None,
            condition: None,
            condition_active: false,
        };
        let usage = vk::BufferUsageFlags::UNIFORM_BUFFER
            | vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::VERTEX_BUFFER
            | vk::BufferUsageFlags::INDEX_BUFFER
            | vk::BufferUsageFlags::INDIRECT_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        // Render conditions are read from a ring copy (`apply_condition`).
        let usage = if renderer.gpu.conditional_rendering.is_some() {
            usage | vk::BufferUsageFlags::CONDITIONAL_RENDERING_EXT
        } else {
            usage
        };
        // Addressable, where the device allows: vertex fetch tables are
        // read through their device address.
        let addressable = renderer.gpu.buffer_device_address;
        renderer.ring = renderer.host_buffer_with(RING_SIZE, usage, addressable)?;
        renderer.spare.ring = renderer.host_buffer_with(RING_SIZE, usage, addressable)?;
        Ok(renderer)
    }

    pub(crate) fn device(&self) -> &ash::Device {
        &self.gpu.device
    }

    /// The command buffer, begun if nothing is recording.
    /// The command buffer, for recording outside a rendering pass (an open
    /// one ends first).
    pub(crate) fn commands(&mut self) -> Result<vk::CommandBuffer, String> {
        let commands = self.recording_commands()?;
        self.end_pass();
        Ok(commands)
    }

    /// End the rendering pass a draw left open, if any.
    pub(crate) fn end_pass(&mut self) {
        if self.open_pass.take().is_some() {
            // SAFETY: recording, inside the pass begun by `begin_rendering`.
            unsafe { self.gpu.device.cmd_end_rendering(self.commands) };
        }
    }

    /// The command buffer, for recording inside the open rendering pass.
    pub(crate) fn recording_commands(&mut self) -> Result<vk::CommandBuffer, String> {
        if !self.recording {
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            // SAFETY: the buffer is idle (its slot's last submission was
            // waited for before it became current again).
            unsafe { self.device().begin_command_buffer(self.commands, &begin) }
                .map_err(vk_error("vkBeginCommandBuffer"))?;
            self.recording = true;
            if let Some(timer) = self.timer.as_mut() {
                timer.begin(&self.gpu.device, self.commands, self.fence);
            }
            if self.spare.in_flight {
                // The other slot's batch may still run: this one starts
                // after it, so batches finish in submission order and each
                // slot's retired resources are free once its fence signals.
                let barrier = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                    .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
                // SAFETY: recording, outside any rendering pass.
                unsafe {
                    self.gpu.device.cmd_pipeline_barrier(
                        self.commands,
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        vk::DependencyFlags::empty(),
                        &[barrier],
                        &[],
                        &[],
                    )
                };
            }
            self.occlusion_after_begin(self.commands);
        }
        Ok(self.commands)
    }

    /// Whether render conditions can be evaluated on the GPU
    /// (VK_EXT_conditional_rendering).
    pub fn supports_conditions(&self) -> bool {
        self.gpu.conditional_rendering.is_some()
    }

    /// The fences of the two submission slots (they keep their handles
    /// when the slots swap), and the current one's.
    pub(crate) fn slot_fences(&self) -> [vk::Fence; 2] {
        [self.fence, self.spare.fence]
    }

    pub(crate) fn current_fence(&self) -> vk::Fence {
        self.fence
    }

    /// Gate the draws and dispatches recorded from now on on the 32-bit
    /// word at `offset` in `buffer` (the guest memory's device copy): they
    /// run while it is nonzero, or zero when `inverted`; None: always.
    /// The word is copied into the ring when gating starts (the GPU's
    /// value then, in command order) and conditional rendering begins
    /// outside a rendering pass, before the next draw or dispatch.
    pub fn set_condition(&mut self, condition: Option<(vk::Buffer, u64, bool)>) {
        if self.condition != condition {
            self.end_condition();
            self.condition = condition;
        }
    }

    /// Begin the wanted condition if it is not active (before recording a
    /// draw or dispatch; ends an open rendering pass to do so).
    pub(crate) fn apply_condition(&mut self) -> Result<(), String> {
        let Some((buffer, offset, inverted)) = self.condition else {
            return Ok(());
        };
        if self.condition_active {
            return Ok(());
        }
        let Some(conditional) = self.gpu.conditional_rendering.clone() else {
            return Ok(());
        };
        let (ring, at) = self.ring_upload(&[0; 4], 4)?;
        let commands = self.commands()?;
        // Reports not yet written are, before the condition reads one.
        self.occlusion_resolve(commands);
        let copy = vk::BufferCopy {
            src_offset: offset,
            dst_offset: at,
            size: 4,
        };
        let before = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
        let after = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::CONDITIONAL_RENDERING_READ_EXT);
        let begin = vk::ConditionalRenderingBeginInfoEXT::default()
            .buffer(ring)
            .offset(at)
            .flags(if inverted {
                vk::ConditionalRenderingFlagsEXT::INVERTED
            } else {
                vk::ConditionalRenderingFlagsEXT::empty()
            });
        // SAFETY: recording, outside rendering; the ranges exist.
        unsafe {
            let device = &self.gpu.device;
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[before],
                &[],
                &[],
            );
            device.cmd_copy_buffer(commands, buffer, ring, &[copy]);
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::CONDITIONAL_RENDERING_EXT,
                vk::DependencyFlags::empty(),
                &[after],
                &[],
                &[],
            );
            (conditional.fp().cmd_begin_conditional_rendering_ext)(commands, &begin);
        }
        self.condition_active = true;
        Ok(())
    }

    /// End the active condition (outside a rendering pass).
    pub(crate) fn end_condition(&mut self) {
        if !self.condition_active {
            return;
        }
        self.end_pass();
        if let Some(conditional) = self.gpu.conditional_rendering.as_ref() {
            // SAFETY: recording; conditional rendering was begun outside a
            // pass in this command buffer, and the pass just ended.
            unsafe { (conditional.fp().cmd_end_conditional_rendering_ext)(self.commands) };
        }
        self.condition_active = false;
    }

    /// Copy `bytes` into the ring at an `alignment`-aligned offset, flushing
    /// first when the ring is full.
    /// A device-local storage buffer of `size` bytes kept under `key`
    /// across dispatches (created, or recreated at another size, as
    /// needed), and whether it is new (its contents undefined).
    #[track_caller]
    pub fn resident_buffer(&mut self, key: u64, size: u64) -> Result<(vk::Buffer, bool), String> {
        let size = size.max(16).next_multiple_of(16);
        if let Some(&(buffer, _, held)) = self.resident.get(&key) {
            if held == size {
                return Ok((buffer, false));
            }
            // Another size: the old one retires after the work using it.
            self.flush()?;
            let (buffer, memory, _) = self.resident.remove(&key).unwrap();
            // SAFETY: the flush finished every use.
            unsafe {
                self.gpu.device.destroy_buffer(buffer, None);
                self.gpu.device.free_memory(memory, None);
            }
        }
        let device = self.gpu.device.clone();
        let addressable = self.gpu.buffer_device_address;
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(addressed_usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
                addressable,
            ))
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: valid create info; destroyed with the renderer.
        let buffer =
            unsafe { device.create_buffer(&info, None) }.map_err(vk_error("vkCreateBuffer"))?;
        // SAFETY: the buffer is live.
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let memory = self.allocate_with(
            requirements,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            addressable,
        )?;
        // SAFETY: fresh buffer and memory of the required size.
        unsafe { device.bind_buffer_memory(buffer, memory, 0) }
            .map_err(vk_error("vkBindBufferMemory"))?;
        self.resident.insert(key, (buffer, memory, size));
        Ok((buffer, true))
    }

    /// The device address of `buffer` (made by this renderer with device
    /// addresses: resident buffers, shared file spans), or None where the
    /// device has none.
    pub fn buffer_address(&self, buffer: vk::Buffer) -> Option<u64> {
        let info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
        // SAFETY: a live buffer created with SHADER_DEVICE_ADDRESS usage
        // (addressed_usage) when the device supports addresses.
        self.gpu
            .buffer_device_address
            .then(|| unsafe { self.gpu.device.get_buffer_device_address(&info) })
    }

    /// The device address of resident buffer `key`.
    pub fn resident_address(&self, key: u64) -> Option<u64> {
        let &(buffer, _, _) = self.resident.get(&key)?;
        self.buffer_address(buffer)
    }

    /// The GPU page table: `entries[i]` is the device address of guest page
    /// `first_page + i` (0: unmapped; see `echora_shader_abi::PageTable`).
    /// Rewritten only when the entries change, after the work that may
    /// still read it; the value is what programs get in their push
    /// constants.
    pub fn set_page_table(
        &mut self,
        first_page: u32,
        entries: &[u64],
    ) -> Result<echora_shader_abi::PageTable, String> {
        if !self.gpu.buffer_device_address {
            return Ok(echora_shader_abi::PageTable::default());
        }
        let bytes = (entries.len().max(1) * 8) as u64;
        let current = self
            .page_table
            .as_ref()
            .is_some_and(|(_, written)| written.as_slice() == entries);
        if !current {
            self.flush()?;
            let fits = self
                .page_table
                .as_ref()
                .is_some_and(|(buffer, _)| buffer.size >= bytes);
            if !fits {
                if let Some((old, _)) = self.page_table.take() {
                    old.destroy(&self.gpu.device);
                }
                let buffer =
                    self.host_buffer_with(bytes, vk::BufferUsageFlags::STORAGE_BUFFER, true)?;
                self.page_table = Some((buffer, Vec::new()));
            }
            let (buffer, written) = self.page_table.as_mut().unwrap();
            let raw: Vec<u8> = entries
                .iter()
                .flat_map(|entry| entry.to_le_bytes())
                .collect();
            buffer.write(0, &raw);
            *written = entries.to_vec();
        }
        let (buffer, _) = self.page_table.as_ref().unwrap();
        Ok(echora_shader_abi::PageTable {
            address: self.buffer_address(buffer.buffer).unwrap_or(0),
            first_page,
            page_count: entries.len() as u32,
        })
    }

    /// The page table's entries as last written (`set_page_table`).
    pub fn page_table_entries(&self) -> &[u64] {
        self.page_table
            .as_ref()
            .map_or(&[], |(_, entries)| entries.as_slice())
    }

    /// Record a copy of `bytes` into resident buffer `key` at `at` (before
    /// work that reads it, after work that used it). Uploads larger than
    /// the ring go in pieces.
    pub fn upload_resident(&mut self, key: u64, at: u64, bytes: &[u8]) -> Result<(), String> {
        const PIECE: usize = 8 << 20;
        if bytes.len() > PIECE {
            for (index, piece) in bytes.chunks(PIECE).enumerate() {
                self.upload_resident(key, at + (index * PIECE) as u64, piece)?;
            }
            return Ok(());
        }
        let (source, offset) = self.ring_upload(bytes, 16)?;
        let &(buffer, _, size) = self.resident.get(&key).ok_or("no such resident buffer")?;
        let commands = self.commands()?;
        let length = (bytes.len() as u64).min(size.saturating_sub(at));
        let before = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
        let after = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
        // SAFETY: recording outside rendering; both buffers are live.
        unsafe {
            let device = &self.gpu.device;
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[before],
                &[],
                &[],
            );
            device.cmd_copy_buffer(
                commands,
                source,
                buffer,
                &[vk::BufferCopy {
                    src_offset: offset,
                    dst_offset: at,
                    size: length,
                }],
            );
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[after],
                &[],
                &[],
            );
        }
        Ok(())
    }

    /// Record a copy of `length` bytes of resident buffer `key` for the
    /// host: its token completes with the next flush (see
    /// `take_reads`).
    pub fn read_resident(&mut self, key: u64, at: u64, length: u64) -> Result<u64, String> {
        let &(buffer, _, size) = self.resident.get(&key).ok_or("no such resident buffer")?;
        let length = length.min(size.saturating_sub(at));
        let (target, offset) = self.ring_alloc(length, 16)?;
        let commands = self.commands()?;
        let before = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
        let after = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ);
        // SAFETY: recording outside rendering; both buffers are live.
        unsafe {
            let device = &self.gpu.device;
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[before],
                &[],
                &[],
            );
            device.cmd_copy_buffer(
                commands,
                buffer,
                target,
                &[vk::BufferCopy {
                    src_offset: at,
                    dst_offset: offset,
                    size: length,
                }],
            );
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[after],
                &[],
                &[],
            );
        }
        Ok(self.defer_ring_read(offset, length as usize))
    }

    /// Room for `size` bytes in the ring (flushing when it is full).
    fn ring_alloc(&mut self, size: u64, alignment: u64) -> Result<(vk::Buffer, u64), String> {
        if size > RING_SIZE {
            return Err(format!("{size}-byte read exceeds the ring"));
        }
        let mut offset = self.ring_used.next_multiple_of(alignment);
        if offset + size > RING_SIZE {
            self.flush()?;
            offset = 0;
        }
        self.ring_used = offset + size;
        Ok((self.ring.buffer, offset))
    }

    pub(crate) fn ring_upload(
        &mut self,
        bytes: &[u8],
        alignment: u64,
    ) -> Result<(vk::Buffer, u64), String> {
        let size = bytes.len() as u64;
        if size > RING_SIZE {
            return Err(format!("{size}-byte upload exceeds the ring"));
        }
        let mut offset = self.ring_used.next_multiple_of(alignment);
        if offset + size > RING_SIZE {
            self.flush()?;
            offset = 0;
        }
        self.ring.write(offset as usize, bytes);
        self.ring_used = offset + size;
        Ok((self.ring.buffer, offset))
    }

    /// Make room, before recording one draw or dispatch, for `bytes` of
    /// ring uploads (alignment included) and one descriptor set holding
    /// `descriptors` of each pooled type, flushing first if either would
    /// overflow. A flush in the middle would reset the pool and the ring
    /// under the work being recorded (its set would be stale and its data
    /// overwritten).
    pub(crate) fn reserve(&mut self, bytes: u64, descriptors: [u32; 4]) -> Result<(), String> {
        // Room for the render condition's word too (`apply_condition`).
        let bytes = bytes + if self.condition.is_some() { 16 } else { 0 };
        if bytes > RING_SIZE {
            return Err(format!("{bytes} bytes of uploads exceed the ring"));
        }
        let capacity = [MAX_SETS * 8, MAX_SETS * 16, MAX_SETS * 4, MAX_SETS * 4];
        let pool_full = self.sets_used + 1 > MAX_SETS
            || (0..4).any(|kind| self.descriptors_used[kind] + descriptors[kind] > capacity[kind]);
        if pool_full || self.ring_used.next_multiple_of(256) + bytes > RING_SIZE {
            self.flush_async()?;
        }
        self.sets_used += 1;
        for (used, more) in self.descriptors_used.iter_mut().zip(descriptors) {
            *used += more;
        }
        Ok(())
    }

    /// `length` bytes of the ring at `offset`, after the work writing them
    /// completed (flushed) and before the ring is reused.
    /// Read `length` ring bytes at `offset` once the recorded work has
    /// completed: returns a token `take_reads` answers after
    /// the next flush.
    pub(crate) fn defer_ring_read(&mut self, offset: u64, length: usize) -> u64 {
        let token = self.next_read_token;
        self.next_read_token += 1;
        self.pending_reads.push((token, offset, length));
        token
    }

    /// Whether deferred reads wait for a flush.
    /// Whether deferred reads wait in either slot.
    pub fn reads_pending(&self) -> bool {
        !self.pending_reads.is_empty() || !self.spare.pending_reads.is_empty()
    }

    /// The completed deferred reads among `tokens`, by token; other
    /// readers' results stay for them.
    pub fn take_reads(&mut self, tokens: &[u64]) -> HashMap<u64, Vec<u8>> {
        let (mine, others) = std::mem::take(&mut self.completed_reads)
            .into_iter()
            .partition(|(token, _)| tokens.contains(token));
        self.completed_reads = others;
        mine.into_iter().collect()
    }

    pub(crate) fn ring_read(&self, offset: u64, length: usize) -> Vec<u8> {
        assert!(offset + length as u64 <= self.ring.size);
        // SAFETY: in bounds of the mapped ring; the GPU work has completed.
        unsafe { std::slice::from_raw_parts(self.ring.mapped.add(offset as usize), length) }
            .to_vec()
    }

    /// A descriptor set of `layout` from this flush's pool.
    pub(crate) fn descriptor_set(
        &mut self,
        layout: vk::DescriptorSetLayout,
    ) -> Result<vk::DescriptorSet, String> {
        let layouts = [layout];
        let info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);
        // SAFETY: the pool and layout are live.
        match unsafe { self.gpu.device.allocate_descriptor_sets(&info) } {
            Ok(sets) => Ok(sets[0]),
            Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY | vk::Result::ERROR_FRAGMENTED_POOL) => {
                // Sets already written stay valid until their work is done.
                self.flush()?;
                // SAFETY: as above, after the pool was reset by the flush.
                unsafe { self.gpu.device.allocate_descriptor_sets(&info) }
                    .map(|sets| sets[0])
                    .map_err(vk_error("vkAllocateDescriptorSets"))
            }
            Err(error) => Err(format!("vkAllocateDescriptorSets: {error}")),
        }
    }

    /// The image view a texture binding of the shader's `texture_type`
    /// (the recompiler's TextureType: 1D, 1D array, 2D, 2D array, 3D, cube,
    /// cube array, buffer, rectangle) samples, uploading a guest texture
    /// when its generation changed.
    /// `texture_view` with `components` applied (a texture view's swizzle)
    /// and, when it is the image's sRGB twin, `format` (a texture view's
    /// format: sRGB decoding of a UNORM texture, or none of an sRGB one).
    pub(crate) fn texture_view_swizzled(
        &mut self,
        source: &crate::draw::TextureSource,
        texture_type: u32,
        components: vk::ComponentMapping,
        format: Option<vk::Format>,
    ) -> Result<vk::ImageView, String> {
        use ash::vk::Handle;
        let plain = self.texture_view(source, texture_type)?;
        let identity = |part: vk::ComponentSwizzle| part == vk::ComponentSwizzle::IDENTITY;
        let Some(info) = self.view_infos.get(&plain).copied() else {
            return Ok(plain);
        };
        let format = format
            .filter(|format| srgb_twin(info.format) == Some(*format))
            .unwrap_or(info.format);
        if format == info.format
            && [components.r, components.g, components.b, components.a]
                .into_iter()
                .all(identity)
        {
            return Ok(plain);
        }
        let cache = (
            info.image,
            i32::MIN
                + (plain.as_raw() as i32 & 0xffff) * 8192
                + i32::from(format != info.format) * 4096
                + [components.r, components.g, components.b, components.a]
                    .iter()
                    .fold(0, |key, part| key * 8 + part.as_raw()),
        );
        if let Some(view) = self.extra_views.get(&cache) {
            return Ok(*view);
        }
        let create = vk::ImageViewCreateInfo::default()
            .image(info.image)
            .view_type(info.view_type)
            .format(format)
            .components(components)
            .subresource_range(info.range);
        // SAFETY: the image is live; the view is destroyed with it.
        let view = unsafe { self.gpu.device.create_image_view(&create, None) }
            .map_err(vk_error("vkCreateImageView"))?;
        self.extra_views.insert(cache, view);
        Ok(view)
    }

    /// Note how a new target's own views were made (see `view_infos`).
    fn register_views(&mut self, target: &Target) {
        let mut range = vk::ImageSubresourceRange {
            level_count: target.levels,
            layer_count: target.shape.layers(),
            ..range(target.format)
        };
        let info = ViewInfo {
            image: target.image,
            view_type: own_view_type(target.shape),
            format: target.format,
            range,
        };
        self.view_infos.insert(target.view, info);
        if has_stencil(target.format) {
            range.aspect_mask = vk::ImageAspectFlags::DEPTH;
        }
        self.view_infos
            .insert(target.sampled_view, ViewInfo { range, ..info });
    }

    pub(crate) fn texture_view(
        &mut self,
        source: &crate::draw::TextureSource,
        texture_type: u32,
    ) -> Result<vk::ImageView, String> {
        use crate::draw::{ImageShape, TextureSource};
        // A 2D upload read by a 1D binding (the shader's texture
        // instruction is 1D): Vulkan has no 1D view of a 2D image, so its
        // first row is uploaded again as a 1D image under its own cache
        // key, which is what a 1D read of the 2D texture returns (row 0;
        // inferred, NVN 1D textures are not told apart from 2D ones of
        // height 1 so far). Each region's copy reads one row from the same
        // texels. Block-compressed formats have no 1D images.
        if let TextureSource::Upload {
            key,
            format,
            shape: ImageShape::D2 { layers },
            ..
        } = source
        {
            if matches!(texture_type, 0 | 1) && !block_compressed(*format) {
                let mut one_dimensional = source.clone();
                if let TextureSource::Upload {
                    key: alias,
                    height,
                    shape,
                    levels,
                    ..
                } = &mut one_dimensional
                {
                    *alias = key ^ 1 << 63;
                    *height = 1;
                    *shape = ImageShape::D1 { layers: *layers };
                    for region in levels {
                        region.height = 1;
                    }
                }
                return self.texture_view(&one_dimensional, texture_type);
            }
            // Likewise a 3D binding (a colour lookup table slot the
            // application fills with an ordinary texture, inferred): a 3D
            // image one slice deep holds the same texels.
            if texture_type == 4 && *layers == 1 && !block_compressed(*format) {
                let mut volume = source.clone();
                if let TextureSource::Upload {
                    key: alias, shape, ..
                } = &mut volume
                {
                    *alias = key ^ 1 << 62;
                    *shape = ImageShape::D3 { depth: 1 };
                }
                return self.texture_view(&volume, texture_type);
            }
        }
        // A 3D upload read by a 2D or 2D-array binding (the instruction
        // decides the type): Vulkan samples no 2D view of a 3D image, so
        // level 0 is uploaded again as a 2D array whose layers are its
        // depth slices, under its own cache key. Each slice's copy reads
        // from the same texels.
        if let TextureSource::Upload {
            key,
            format,
            shape: ImageShape::D3 { depth },
            texels,
            levels,
            ..
        } = source
            && matches!(texture_type, 2 | 3)
            && !block_compressed(*format)
            && *depth > 0
        {
            let mut layered = source.clone();
            let base = levels.iter().find(|region| region.level == 0).copied();
            if let (
                TextureSource::Upload {
                    key: alias,
                    shape,
                    levels: slices,
                    ..
                },
                Some(base),
            ) = (&mut layered, base)
            {
                let end = levels
                    .iter()
                    .map(|region| region.offset)
                    .filter(|&offset| offset > base.offset)
                    .min()
                    .unwrap_or(texels.len() as u64);
                let slice = (end - base.offset) / u64::from(*depth);
                *alias = key ^ 1 << 61;
                *shape = ImageShape::D2 { layers: *depth };
                *slices = (0..*depth)
                    .map(|layer| crate::draw::UploadRegion {
                        layer,
                        depth: 1,
                        offset: base.offset + u64::from(layer) * slice,
                        ..base
                    })
                    .collect();
                return self.texture_view(&layered, texture_type);
            }
        }
        // A view of some mip levels and/or array layers (cube faces) of a
        // render target.
        if let TextureSource::TargetView {
            key,
            base_level,
            level_count,
            base_layer,
            layer_count,
        } = *source
        {
            let whole = TextureSource::Target(key);
            let Some(target) = self.targets.get(&key) else {
                return Err("sampled target does not exist".into());
            };
            let base_level = base_level.min(target.levels - 1);
            let level_count = if level_count == 0 {
                target.levels - base_level
            } else {
                level_count.min(target.levels - base_level)
            };
            let layers = target.shape.layers();
            let base_layer = base_layer.min(layers - 1);
            let layer_count = if layer_count == 0 {
                layers - base_layer
            } else {
                layer_count.min(layers - base_layer)
            };
            let layered = matches!(
                target.shape,
                ImageShape::D2 { .. } | ImageShape::Cube { .. }
            );
            let cube = matches!(target.shape, ImageShape::Cube { .. });
            // The binding's view type over the selected range, when the
            // range can hold it.
            let view = match texture_type {
                2 | 8 if layered => Some((vk::ImageViewType::TYPE_2D, 1)),
                3 if layered => Some((vk::ImageViewType::TYPE_2D_ARRAY, layer_count)),
                5 if cube && layer_count >= 6 => Some((vk::ImageViewType::CUBE, 6)),
                6 if cube && layer_count >= 6 => {
                    Some((vk::ImageViewType::CUBE_ARRAY, layer_count / 6 * 6))
                }
                _ => None,
            };
            let Some((view_type, layer_count)) = view else {
                return self.texture_view(&whole, texture_type);
            };
            if (base_level, level_count, base_layer, layer_count) == (0, target.levels, 0, layers)
                && view_type == own_view_type(target.shape)
            {
                return self.texture_view(&whole, texture_type);
            }
            self.transition(key, vk::ImageLayout::GENERAL)?;
            let target = &self.targets[&key];
            let packed = base_level
                | level_count << 5
                | base_layer << 10
                | layer_count << 17
                | (view_type.as_raw() as u32) << 25;
            let cache = (target.image, -(1 << 30) - packed as i32);
            if let Some(view) = self.extra_views.get(&cache) {
                return Ok(*view);
            }
            let mut subresource = range(target.format);
            if has_stencil(target.format) {
                subresource.aspect_mask = vk::ImageAspectFlags::DEPTH;
            }
            let info = vk::ImageViewCreateInfo::default()
                .image(target.image)
                .view_type(view_type)
                .format(target.format)
                .subresource_range(vk::ImageSubresourceRange {
                    base_mip_level: base_level,
                    level_count,
                    base_array_layer: base_layer,
                    layer_count,
                    ..subresource
                });
            // SAFETY: the image is live; the view is destroyed with it.
            let view = unsafe { self.gpu.device.create_image_view(&info, None) }
                .map_err(vk_error("vkCreateImageView"))?;
            self.extra_views.insert(cache, view);
            self.view_infos.insert(view, ViewInfo::of(&info));
            return Ok(view);
        }
        // A 3D binding of a 2D render target: the application renders a 3D
        // colour lookup table slice by slice, which 2D targets do not hold
        // yet. An identity table of the same size (neutral grading) stands
        // in (an approximation).
        if let TextureSource::Target(key) = source {
            let flat = self.targets.get(key).is_some_and(|target| {
                matches!(target.shape, ImageShape::D2 { .. }) && target.width == target.height
            });
            if texture_type == 4 && flat {
                let size = self.targets[key].width.clamp(2, 64);
                let texels: Vec<u8> = (0..size * size * size)
                    .flat_map(|index| {
                        let channel = |value: u32| (value * 255 / (size - 1)) as u8;
                        [
                            channel(index % size),
                            channel(index / size % size),
                            channel(index / (size * size)),
                            255,
                        ]
                    })
                    .collect();
                let identity = TextureSource::Upload {
                    key: u64::MAX - u64::from(size),
                    generation: 0,
                    width: size,
                    height: size,
                    shape: ImageShape::D3 { depth: size },
                    format: vk::Format::R8G8B8A8_UNORM,
                    texels: std::sync::Arc::new(texels),
                    levels: vec![crate::draw::UploadRegion {
                        level: 0,
                        layer: 0,
                        width: size,
                        height: size,
                        depth: size,
                        offset: 0,
                    }],
                };
                return self.texture_view(&identity, texture_type);
            }
        }
        let key = match source {
            TextureSource::Target(key) | TextureSource::TargetView { key, .. } => {
                self.transition(*key, vk::ImageLayout::GENERAL)?;
                if !self.targets.contains_key(key) {
                    return Err("sampled target does not exist".into());
                }
                *key
            }
            TextureSource::Upload { key, .. } => {
                self.upload(source)?;
                *key
            }
        };
        let target = match source {
            TextureSource::Target(_) | TextureSource::TargetView { .. } => &self.targets[&key],
            TextureSource::Upload { .. } => &self.uploads[&key].1,
        };
        let wanted = match texture_type {
            0 => vk::ImageViewType::TYPE_1D,
            1 => vk::ImageViewType::TYPE_1D_ARRAY,
            3 => vk::ImageViewType::TYPE_2D_ARRAY,
            4 => vk::ImageViewType::TYPE_3D,
            5 => vk::ImageViewType::CUBE,
            6 => vk::ImageViewType::CUBE_ARRAY,
            _ => vk::ImageViewType::TYPE_2D,
        };
        if wanted == own_view_type(target.shape) {
            return Ok(target.sampled_view);
        }
        if let Some(view) = self.extra_views.get(&(target.image, wanted.as_raw())) {
            return Ok(*view);
        }
        let layers = target.shape.layers();
        let layer_count = match (wanted, target.shape) {
            (vk::ImageViewType::TYPE_1D, ImageShape::D1 { .. })
            | (vk::ImageViewType::TYPE_2D, ImageShape::D2 { .. } | ImageShape::Cube { .. }) => 1,
            (vk::ImageViewType::TYPE_1D_ARRAY, ImageShape::D1 { .. })
            | (vk::ImageViewType::TYPE_2D_ARRAY, ImageShape::D2 { .. } | ImageShape::Cube { .. }) => {
                layers
            }
            (vk::ImageViewType::CUBE, ImageShape::Cube { .. }) => 6,
            (vk::ImageViewType::CUBE_ARRAY, ImageShape::Cube { .. }) => layers,
            _ => {
                let what = match source {
                    TextureSource::Target(_) | TextureSource::TargetView { .. } => "render target",
                    TextureSource::Upload { .. } => "upload",
                };
                return Err(format!(
                    "a {wanted:?} binding samples {what} {key:x}, a {:?} image",
                    target.shape
                ));
            }
        };
        let mut subresource = range(target.format);
        if has_stencil(target.format) {
            subresource.aspect_mask = vk::ImageAspectFlags::DEPTH;
        }
        let info = vk::ImageViewCreateInfo::default()
            .image(target.image)
            .view_type(wanted)
            .format(target.format)
            .subresource_range(vk::ImageSubresourceRange {
                level_count: target.levels,
                layer_count,
                ..subresource
            });
        let image = target.image;
        // SAFETY: the image is live; the view is destroyed with it.
        let view = unsafe { self.gpu.device.create_image_view(&info, None) }
            .map_err(vk_error("vkCreateImageView"))?;
        self.extra_views.insert((image, wanted.as_raw()), view);
        self.view_infos.insert(view, ViewInfo::of(&info));
        Ok(view)
    }

    /// Upload a guest texture when it is new or its generation changed.
    fn upload(&mut self, source: &crate::draw::TextureSource) -> Result<(), String> {
        let crate::draw::TextureSource::Upload {
            key,
            generation,
            width,
            height,
            shape,
            format,
            texels,
            levels,
        } = source
        else {
            return Ok(());
        };
        if let Some((uploaded, target)) = self.uploads.get(key) {
            if uploaded == generation
                && (target.width, target.height, target.format, target.shape)
                    == (*width, *height, *format, *shape)
            {
                return Ok(());
            }
        }
        let level_count = levels
            .iter()
            .map(|region| region.level + 1)
            .max()
            .unwrap_or(1);
        // A texture whose contents change keeps its image when its shape
        // does (each movie picture): the new texels are copied over the
        // old, ordered after the work that read them, with no flush and no
        // new allocation. Another shape retires the old image, freed once
        // the work submitted so far has completed.
        let mut reused = None;
        if let Some((_, old)) = self.uploads.remove(key) {
            if (old.width, old.height, old.format, old.shape) == (*width, *height, *format, *shape)
                && old.levels == level_count
            {
                reused = Some(old);
            } else {
                self.retired_uploads.push(old);
            }
        }
        // A guest texture's upload never needs more than this: a larger one
        // means a misread size, shape or format, which stops the run with
        // the texture's description instead of exhausting memory.
        if texels.len() as u64 > MAX_UPLOAD_BYTES {
            let message = format!(
                "upload {key:#x}: {width}x{height} {shape:?} {format:?}, {level_count} levels, is {} MiB, over the {} MiB limit",
                texels.len() >> 20,
                MAX_UPLOAD_BYTES >> 20
            );
            self.fatal = Some(message.clone());
            return Err(message);
        }
        // Each region's bytes: up to the next region's offset, the last to
        // the end.
        let mut by_offset: Vec<&crate::draw::UploadRegion> = levels.iter().collect();
        by_offset.sort_by_key(|region| region.offset);
        let region_bytes = |index: usize| {
            by_offset
                .get(index + 1)
                .map_or(texels.len() as u64, |next| next.offset)
                - by_offset[index].offset
        };
        // An upload larger than the ring (a big ASTC array decoded to
        // RGBA8) whose regions each fit goes through the ring a batch of
        // regions at a time, with no staging buffer of its own (two of
        // 852 MiB pushed host memory past its limit).
        let chunked = texels.len() as u64 > RING_SIZE
            && (0..by_offset.len()).all(|index| region_bytes(index) <= RING_SIZE);
        // The texels are staged before the image exists, so a failure
        // leaves nothing behind. Other uploads larger than the ring get a
        // staging buffer of their own, destroyed with the submission's
        // retired buffers.
        let (buffer, offset) = if chunked {
            (vk::Buffer::null(), 0)
        } else if texels.len() as u64 > RING_SIZE {
            let staging =
                self.host_buffer(texels.len() as u64, vk::BufferUsageFlags::TRANSFER_SRC)?;
            staging.write(0, texels);
            let buffer = staging.buffer;
            self.retired_texel_buffers
                .push((staging, vk::BufferView::null()));
            (buffer, 0)
        } else {
            self.ring_upload(texels, 16)?
        };
        let reusing = reused.is_some();
        let mut target = match reused {
            Some(target) => target,
            None => self.create_shaped_image(
                *width,
                *height,
                *shape,
                *format,
                vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::TRANSFER_SRC,
                level_count,
            )?,
        };
        let commands = match self.commands() {
            Ok(commands) => commands,
            Err(error) => {
                self.retired_uploads.push(target);
                return Err(error);
            }
        };
        let barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(if reusing {
                vk::AccessFlags::SHADER_READ
            } else {
                vk::AccessFlags::empty()
            })
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(target.image)
            .subresource_range(vk::ImageSubresourceRange {
                level_count: target.levels,
                layer_count: shape.layers(),
                ..range(*format)
            });
        let after = vk::ImageMemoryBarrier {
            src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            dst_access_mask: vk::AccessFlags::SHADER_READ,
            old_layout: vk::ImageLayout::GENERAL,
            ..barrier
        };
        // Levels the image cannot hold (create_shaped_image clamps) are
        // not copied.
        let copies: Vec<vk::BufferImageCopy> = levels
            .iter()
            .filter(|region| region.level < target.levels && region.layer < shape.layers())
            .map(|region| vk::BufferImageCopy {
                buffer_offset: offset + region.offset,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: region.level,
                    base_array_layer: region.layer,
                    layer_count: 1,
                },
                image_extent: vk::Extent3D {
                    width: region.width,
                    height: region.height,
                    depth: region.depth,
                },
                ..Default::default()
            })
            .collect();
        // SAFETY: recording; the image and ring are live.
        unsafe {
            let device = &self.gpu.device;
            // A reused image: after every earlier read of it.
            device.cmd_pipeline_barrier(
                commands,
                if reusing {
                    vk::PipelineStageFlags::ALL_COMMANDS
                } else {
                    vk::PipelineStageFlags::TOP_OF_PIPE
                },
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
            if !chunked {
                device.cmd_copy_buffer_to_image(
                    commands,
                    buffer,
                    target.image,
                    vk::ImageLayout::GENERAL,
                    &copies,
                );
            }
        }
        if chunked {
            // Batches of consecutive regions that fill the ring; a batch
            // may flush first (the barrier above is then submitted before
            // its copies).
            let mut start = 0;
            while start < by_offset.len() {
                let mut end = start;
                let mut bytes = 0;
                while end < by_offset.len() && bytes + region_bytes(end) <= RING_SIZE {
                    bytes += region_bytes(end);
                    end += 1;
                }
                let first = by_offset[start].offset;
                let slice = &texels[first as usize..(first + bytes) as usize];
                let (ring, at) = self.ring_upload(slice, 16)?;
                let batch: Vec<vk::BufferImageCopy> = copies
                    .iter()
                    .filter(|copy| {
                        let source = copy.buffer_offset - offset;
                        source >= first && source < first + bytes
                    })
                    .map(|copy| vk::BufferImageCopy {
                        buffer_offset: at + (copy.buffer_offset - offset - first),
                        ..*copy
                    })
                    .collect();
                let commands = self.commands()?;
                // SAFETY: recording; the image and ring are live.
                unsafe {
                    self.gpu.device.cmd_copy_buffer_to_image(
                        commands,
                        ring,
                        target.image,
                        vk::ImageLayout::GENERAL,
                        &batch,
                    );
                }
                start = end;
            }
        }
        let commands = self.commands()?;
        // SAFETY: recording; the image is live.
        unsafe {
            self.gpu.device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_GRAPHICS | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[after],
            );
        }
        target.layout = vk::ImageLayout::GENERAL;
        if !reusing {
            self.register_views(&target);
        }
        self.uploads.insert(*key, (*generation, target));
        Ok(())
    }

    pub(crate) fn sampler(
        &mut self,
        sampler: &crate::draw::Sampler,
    ) -> Result<vk::Sampler, String> {
        let key: SamplerKey = (
            sampler.min_filter.as_raw(),
            sampler.mag_filter.as_raw(),
            sampler.mipmap.as_raw(),
            sampler.base_level_only,
            sampler.address.map(|mode| mode.as_raw()),
            sampler.lod.map(f32::to_bits),
        );
        if let Some(existing) = self.samplers.get(&key) {
            return Ok(*existing);
        }
        let info = vk::SamplerCreateInfo::default()
            .min_filter(sampler.min_filter)
            .mag_filter(sampler.mag_filter)
            .mipmap_mode(sampler.mipmap)
            .address_mode_u(sampler.address[0])
            .address_mode_v(sampler.address[1])
            .address_mode_w(sampler.address[2])
            .min_lod(sampler.lod[0])
            .max_lod(if sampler.base_level_only {
                0.25
            } else {
                sampler.lod[1].max(sampler.lod[0])
            })
            .mip_lod_bias(sampler.lod[2]);
        // SAFETY: valid create info; destroyed in Drop.
        let created = unsafe { self.gpu.device.create_sampler(&info, None) }
            .map_err(vk_error("vkCreateSampler"))?;
        self.samplers.insert(key, created);
        Ok(created)
    }

    /// Submit what is recorded and wait for it.
    /// Submit what was recorded and wait until all submitted work has
    /// completed: deferred reads are then available and retired resources
    /// are freed.
    pub fn flush(&mut self) -> Result<(), String> {
        if self.spare.in_flight {
            self.swap_slot();
            self.finish_current()?;
            self.swap_slot();
        }
        if self.recording {
            self.submit_current()?;
            self.finish_current()?;
        }
        // No work reads the texture pool any more.
        if let Some(pool) = self.texture_pool.as_mut() {
            pool.in_use = false;
        }
        Ok(())
    }

    /// End of a frame: submit what was recorded without waiting for it, so
    /// the GPU renders this frame while the CPU prepares the next. Reads of
    /// results still go through `flush`, which waits.
    pub fn submit_frame(&mut self) -> Result<(), String> {
        self.flush_async()
    }

    /// Submit what was recorded and continue in the other slot, waiting
    /// only for that slot's previous submission: the GPU runs one batch
    /// while the CPU records the next. For capacity limits (work items,
    /// ring and descriptor space), where nothing reads results.
    pub(crate) fn flush_async(&mut self) -> Result<(), String> {
        if !self.recording {
            return Ok(());
        }
        self.submit_current()?;
        self.swap_slot();
        // Its previous submission (if any) must finish before reuse.
        self.finish_current()
    }

    /// Exchange the recorded slot with the spare one.
    fn swap_slot(&mut self) {
        let spare = &mut self.spare;
        std::mem::swap(&mut self.commands, &mut spare.commands);
        std::mem::swap(&mut self.fence, &mut spare.fence);
        std::mem::swap(&mut self.ring, &mut spare.ring);
        std::mem::swap(&mut self.ring_used, &mut spare.ring_used);
        std::mem::swap(&mut self.descriptor_pool, &mut spare.descriptor_pool);
        std::mem::swap(&mut self.sets_used, &mut spare.sets_used);
        std::mem::swap(&mut self.descriptors_used, &mut spare.descriptors_used);
        std::mem::swap(&mut self.pending_reads, &mut spare.pending_reads);
        std::mem::swap(&mut self.retired_uploads, &mut spare.retired_uploads);
        std::mem::swap(
            &mut self.retired_texel_buffers,
            &mut spare.retired_texel_buffers,
        );
        std::mem::swap(&mut self.retired_shared, &mut spare.retired_shared);
        std::mem::swap(&mut self.current_in_flight, &mut spare.in_flight);
    }

    /// End and submit the current slot's command buffer.
    fn submit_current(&mut self) -> Result<(), String> {
        self.end_pass();
        self.end_condition();
        self.occlusion_before_submit(self.commands);
        // Ring bytes the host reads back (storage a draw wrote) are made
        // visible to it once the submission completes.
        if !self.pending_reads.is_empty() {
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ);
            // SAFETY: recording, outside rendering (the pass just ended).
            unsafe {
                self.gpu.device.cmd_pipeline_barrier(
                    self.commands,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[barrier],
                    &[],
                    &[],
                );
            }
        }
        self.recording = false;
        self.flushes += 1;
        self.work_items = 0;
        let labels = std::mem::take(&mut self.labels);
        // The one before stays beside it (`.prev`): with two submissions in
        // flight, a hang can be in either.
        if let Some(path) = std::env::var_os("ECHORA_LAST_SUBMIT") {
            let mut previous = path.clone();
            previous.push(".prev");
            let _ = std::fs::rename(&path, previous);
            let _ = std::fs::write(path, labels.join("\n"));
        }
        // ECHORA_SUBMIT_LOG=<file> (debugging): every submission's labels
        // from present ECHORA_SUBMIT_LOG_FROM on, appended.
        if let Some(path) = std::env::var_os("ECHORA_SUBMIT_LOG") {
            let from = std::env::var("ECHORA_SUBMIT_LOG_FROM")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(0);
            let present = |label: &String| {
                label
                    .strip_prefix("present ")
                    .and_then(|rest| rest.split(' ').next())
                    .and_then(|number| number.parse::<u64>().ok())
            };
            let lines: Vec<&String> = labels
                .iter()
                .filter(|label| present(label).is_some_and(|number| number >= from))
                .collect();
            if !lines.is_empty() {
                use std::io::Write as _;
                if let Ok(mut file) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                {
                    for line in lines {
                        let _ = writeln!(file, "{line}");
                    }
                    let _ = writeln!(file, "-- submit");
                }
            }
        }
        self.submitted_labels = labels;
        if let Some(timer) = &self.timer {
            timer.end(&self.gpu.device, self.commands, self.fence);
        }
        let device = &self.gpu.device;
        let buffers = [self.commands];
        let submit = [vk::SubmitInfo::default().command_buffers(&buffers)];
        // SAFETY: the buffer was begun; the fence is unsignaled.
        unsafe {
            device
                .end_command_buffer(self.commands)
                .map_err(vk_error("vkEndCommandBuffer"))?;
            device
                .queue_submit(self.gpu.queue, &submit, self.fence)
                .map_err(vk_error("vkQueueSubmit"))?;
        }
        self.current_in_flight = true;
        Ok(())
    }

    /// Wait for the current slot's submission, then free what it retired,
    /// collect its deferred reads and make the slot ready to record.
    fn finish_current(&mut self) -> Result<(), String> {
        if !self.current_in_flight {
            return Ok(());
        }
        let wait_start = std::time::Instant::now();
        let device = &self.gpu.device;
        // SAFETY: the fence belongs to a submitted batch.
        unsafe {
            device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .map_err(vk_error("vkWaitForFences"))?;
            device
                .reset_fences(&[self.fence])
                .map_err(vk_error("vkResetFences"))?;
            device
                .reset_command_buffer(self.commands, vk::CommandBufferResetFlags::empty())
                .map_err(vk_error("vkResetCommandBuffer"))?;
            device
                .reset_descriptor_pool(self.descriptor_pool, vk::DescriptorPoolResetFlags::empty())
                .map_err(vk_error("vkResetDescriptorPool"))?;
        }
        self.current_in_flight = false;
        self.flush_wait_microseconds += wait_start.elapsed().as_micros() as u64;
        if let Some(timer) = self.timer.as_mut() {
            timer.finished(&self.gpu.device, self.fence, self.flush_wait_microseconds);
        }
        // ECHORA_SLOW_SUBMITS_MS: the wait that counts as slow (1000).
        let slow = std::env::var("ECHORA_SLOW_SUBMITS_MS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(1000);
        if wait_start.elapsed() > std::time::Duration::from_millis(slow) {
            if let Some(path) = std::env::var_os("ECHORA_SLOW_SUBMITS") {
                use std::io::Write;
                if let Ok(mut file) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                {
                    let _ = writeln!(
                        file,
                        "submission waited {:?}:\n{}\n",
                        wait_start.elapsed(),
                        self.submitted_labels.join("\n")
                    );
                }
            }
        }
        // The work has completed: forgotten uploads can go.
        for target in std::mem::take(&mut self.retired_uploads) {
            self.destroy_target(target);
        }
        for (buffer, view) in std::mem::take(&mut self.retired_texel_buffers) {
            self.destroy_texel_buffer(buffer, view);
        }
        for (buffer, memory) in std::mem::take(&mut self.retired_shared) {
            // SAFETY: the work reading it has completed.
            unsafe {
                self.gpu.device.destroy_buffer(buffer, None);
                self.gpu.device.free_memory(memory, None);
            }
        }
        // Deferred results leave the ring before its next use overwrites
        // them.
        for (token, offset, length) in std::mem::take(&mut self.pending_reads) {
            let bytes = self.ring_read(offset, length);
            self.completed_reads.push((token, bytes));
        }
        self.ring_used = 0;
        self.sets_used = 0;
        self.descriptors_used = [0; 4];
        Ok(())
    }

    /// The texel-buffer view of guest buffer texture `source`: its level-0
    /// texels copied into a buffer once per generation. Buffer textures
    /// can be far longer than the largest 1D image (light lists a million
    /// texels long), which only a texel buffer reads correctly.
    pub(crate) fn texel_buffer_view(
        &mut self,
        source: &crate::draw::TextureSource,
    ) -> Result<vk::BufferView, String> {
        let crate::draw::TextureSource::Upload {
            key,
            generation,
            format,
            texels,
            levels,
            ..
        } = source
        else {
            return Err("texel buffers from render targets are not modeled".into());
        };
        if let Some((cached, _, view)) = self.texel_buffers.get(key) {
            if cached == generation {
                return Ok(*view);
            }
        }
        let start = levels.first().map_or(0, |region| region.offset as usize);
        let end = levels
            .get(1)
            .map_or(texels.len(), |region| region.offset as usize)
            .min(texels.len());
        let texel_size = format_bytes(*format) as usize;
        if texel_size == 0 {
            return Err(format!("{format:?} has no texel-buffer form"));
        }
        let bytes = &texels[start.min(end)..end];
        let length = bytes.len() / texel_size * texel_size;
        let buffer = self.host_buffer(
            length.max(texel_size) as u64,
            vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER,
        )?;
        buffer.write(0, &bytes[..length]);
        let info = vk::BufferViewCreateInfo::default()
            .buffer(buffer.buffer)
            .format(bit_preserving_float(*format))
            .offset(0)
            .range(length.max(texel_size) as u64);
        // SAFETY: the buffer is live; the view is destroyed with it.
        let view = match unsafe { self.device().create_buffer_view(&info, None) } {
            Ok(view) => view,
            Err(error) => {
                buffer.destroy(self.device());
                return Err(format!("vkCreateBufferView {format:?}: {error}"));
            }
        };
        if let Some((_, old, old_view)) =
            self.texel_buffers.insert(*key, (*generation, buffer, view))
        {
            self.retired_texel_buffers.push((old, old_view));
        }
        Ok(view)
    }

    fn destroy_texel_buffer(&self, buffer: HostBuffer, view: vk::BufferView) {
        // SAFETY: the work reading it has completed.
        unsafe { self.device().destroy_buffer_view(view, None) };
        buffer.destroy(self.device());
    }

    /// `length` bytes of guest memory's file `file` at `offset` as a storage
    /// buffer the GPU reads and writes in place: the buffer and the offset
    /// in it. The page-aligned span is shared as a dma-buf (a udmabuf of the
    /// file, which must be sealed against shrinking), so the GPU sees the
    /// CPU's writes without flushes, as the console's does. `None` where
    /// that is unavailable (no dma-buf import, no /dev/udmabuf, a span over
    /// udmabuf's 64 MiB limit, an import the driver refuses): the caller
    /// copies instead. Spans are kept for reuse, the least recently used
    /// dropped beyond MAX_SHARED_BYTES (udmabuf pins their pages).
    pub fn import_file(
        &mut self,
        file: i32,
        offset: u64,
        length: u64,
    ) -> Result<Option<(vk::Buffer, u64)>, String> {
        const PAGE: u64 = 4096;
        let start = offset & !(PAGE - 1);
        let span = (offset - start + length.max(1)).next_multiple_of(PAGE);
        let key = (start, span);
        if let Some(index) = self.shared.iter().position(|(known, _, _)| *known == key) {
            let entry = self.shared.remove(index);
            let buffer = entry.1;
            self.shared.push(entry);
            return Ok(Some((buffer, offset - start)));
        }
        if span > MAX_SHARED_SPAN || self.shared_refused.contains(&key) {
            return Ok(None);
        }
        let Some(imported) = self.share_span(file, start, span)? else {
            self.shared_refused.insert(key);
            return Ok(None);
        };
        self.shared.push((key, imported.0, imported.1));
        self.shared_bytes += span;
        while self.shared_bytes > MAX_SHARED_BYTES && self.shared.len() > 1 {
            let ((_, dropped), buffer, memory) = self.shared.remove(0);
            self.shared_bytes -= dropped;
            self.retired_shared.push((buffer, memory));
        }
        Ok(Some((imported.0, offset - start)))
    }

    /// The device address of `length` bytes of guest memory's file `file`
    /// at `offset`, shared in place (a udmabuf, as `import_file`) for the
    /// GPU page table, at most `MAX_SHARED_SPAN` bytes. The span stays until
    /// `release_file_range` covers it. None where sharing or device
    /// addresses are unavailable.
    pub fn address_span(
        &mut self,
        file: i32,
        offset: u64,
        length: u64,
    ) -> Result<Option<u64>, String> {
        const PAGE: u64 = 4096;
        let start = offset & !(PAGE - 1);
        let span = (offset - start + length.max(1)).next_multiple_of(PAGE);
        if let Some(&(known, _, _, _, address)) = self
            .addressed
            .iter()
            .find(|(known, held, ..)| *known <= start && start + span <= known + held)
        {
            return Ok(Some(address + (offset - known)));
        }
        if span > MAX_SHARED_SPAN || !self.gpu.buffer_device_address {
            self.stat("stat: spans too large to share", 1);
            return Ok(None);
        }
        let Some((buffer, memory)) = self.share_span(file, start, span)? else {
            return Ok(None);
        };
        let Some(address) = self.buffer_address(buffer) else {
            self.stat("stat: shared spans without a device address", 1);
            self.retired_shared.push((buffer, memory));
            return Ok(None);
        };
        self.addressed.push((start, span, buffer, memory, address));
        Ok(Some(address + (offset - start)))
    }

    /// Bytes of guest memory the page table's spans pin.
    pub fn addressed_bytes(&self) -> u64 {
        self.addressed.iter().map(|(_, span, ..)| span).sum()
    }

    /// Stop sharing file range `[offset, offset + length)`: its pages may be
    /// released (a udmabuf would keep the old ones).
    pub fn release_file_range(&mut self, offset: u64, length: u64) {
        let end = offset + length;
        let (gone, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut self.addressed)
            .into_iter()
            .partition(|(start, span, ..)| *start < end && offset < start + span);
        self.addressed = kept;
        for (_, _, buffer, memory, _) in gone {
            self.retired_shared.push((buffer, memory));
        }
        let (dropped, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut self.shared)
            .into_iter()
            .partition(|((start, span), _, _)| *start < end && offset < start + span);
        self.shared = kept;
        for ((_, span), buffer, memory) in dropped {
            self.shared_bytes -= span;
            self.retired_shared.push((buffer, memory));
        }
        self.shared_refused
            .retain(|(start, span)| !(*start < end && offset < start + span));
    }

    /// A udmabuf of `span` bytes of `file` at `start`, imported as a storage
    /// buffer, or None where that is unavailable.
    #[cfg(target_os = "linux")]
    fn share_span(
        &mut self,
        file: i32,
        start: u64,
        span: u64,
    ) -> Result<Option<(vk::Buffer, vk::DeviceMemory)>, String> {
        use std::os::fd::AsRawFd;
        let Some(external) = self.gpu.dma_buf_import.clone() else {
            return Ok(None);
        };
        let udmabuf = self.udmabuf.get_or_insert_with(|| {
            std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open("/dev/udmabuf")
                .ok()
        });
        let Some(udmabuf) = udmabuf else {
            self.stat("stat: udmabuf device unavailable", 1);
            return Ok(None);
        };
        // struct udmabuf_create { u32 memfd; u32 flags; u64 offset; u64 size; }
        // and UDMABUF_CREATE = _IOW('u', 0x42, struct udmabuf_create).
        #[repr(C)]
        struct Create {
            memfd: u32,
            flags: u32,
            offset: u64,
            size: u64,
        }
        const UDMABUF_CREATE: libc::c_ulong = 0x4018_7542;
        const UDMABUF_FLAGS_CLOEXEC: u32 = 1;
        let mut create = Create {
            memfd: file as u32,
            flags: UDMABUF_FLAGS_CLOEXEC,
            offset: start,
            size: span,
        };
        // SAFETY: a valid request for the udmabuf device; it returns a new
        // descriptor (or -1) and does not keep the pointer.
        let dma_buf = unsafe {
            libc::ioctl(
                udmabuf.as_raw_fd(),
                UDMABUF_CREATE as _,
                std::ptr::addr_of_mut!(create),
            )
        };
        if dma_buf < 0 {
            // The kernel's reason, the first few times (an import the page
            // table needs that fails leaves its pages unmapped).
            self.stat("stat: udmabuf creations refused", 1);
            if self.stats.get("stat: udmabuf creations refused") <= Some(&4) {
                eprintln!(
                    "gpu: udmabuf of {span:#x} bytes at file offset {start:#x} refused: {}",
                    std::io::Error::last_os_error()
                );
            }
            return Ok(None);
        }
        let handle = vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;
        let mut properties = vk::MemoryFdPropertiesKHR::default();
        // SAFETY: `dma_buf` is a dma-buf descriptor we own.
        let queried =
            unsafe { external.get_memory_fd_properties(handle, dma_buf, &mut properties) };
        let device = self.gpu.device.clone();
        let close = || {
            // SAFETY: ours, and not imported.
            unsafe { libc::close(dma_buf) };
        };
        if queried.is_err() {
            close();
            self.stat("stat: dma-buf memory properties refused", 1);
            return Ok(None);
        }
        let addressable = self.gpu.buffer_device_address;
        let mut external_info = vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle);
        let info = vk::BufferCreateInfo::default()
            .size(span)
            .usage(addressed_usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
                addressable,
            ))
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut external_info);
        // SAFETY: valid create info; destroyed with the span.
        let buffer = match unsafe { device.create_buffer(&info, None) } {
            Ok(buffer) => buffer,
            Err(error) => {
                close();
                return Err(format!("vkCreateBuffer: {error}"));
            }
        };
        // SAFETY: the buffer is live.
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let bits = properties.memory_type_bits & requirements.memory_type_bits;
        let refuse = |close: &dyn Fn()| {
            close();
            // SAFETY: unused buffer.
            unsafe { device.destroy_buffer(buffer, None) };
            Ok(None)
        };
        if bits == 0 || requirements.size > span {
            self.stat("stat: dma-buf imports refused (memory type or size)", 1);
            return refuse(&close);
        }
        let mut import = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(handle)
            .fd(dma_buf);
        let mut address_flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let mut allocate = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(bits.trailing_zeros())
            .push_next(&mut import);
        if addressable {
            allocate = allocate.push_next(&mut address_flags);
        }
        // SAFETY: a successful import takes ownership of `dma_buf`.
        let memory = match unsafe { device.allocate_memory(&allocate, None) } {
            Ok(memory) => memory,
            Err(_) => {
                self.stat("stat: dma-buf imports refused (allocation)", 1);
                return refuse(&close);
            }
        };
        // SAFETY: fresh buffer and memory of the required size.
        if unsafe { device.bind_buffer_memory(buffer, memory, 0) }.is_err() {
            // SAFETY: unused buffer and memory.
            unsafe {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
            }
            return Ok(None);
        }
        Ok(Some((buffer, memory)))
    }

    #[cfg(not(target_os = "linux"))]
    fn share_span(
        &mut self,
        _file: i32,
        _start: u64,
        _span: u64,
    ) -> Result<Option<(vk::Buffer, vk::DeviceMemory)>, String> {
        Ok(None)
    }

    /// `length` bytes of host memory at `pointer` as a storage buffer the
    /// GPU reads and writes in place (no copies): the buffer and the offset
    /// of `pointer` in it. `None` without host-memory import. The memory
    /// must stay mapped while the renderer lives; imports are kept per
    /// page-aligned span and reused.
    pub fn import_host(
        &mut self,
        pointer: *mut u8,
        length: u64,
    ) -> Result<Option<(vk::Buffer, u64)>, String> {
        let Some(host) = self.gpu.host_import.clone() else {
            return Ok(None);
        };
        const PAGE: usize = 4096;
        let start = pointer as usize & !(PAGE - 1);
        let offset = (pointer as usize - start) as u64;
        let span = (offset + length.max(1)).next_multiple_of(PAGE as u64);
        if let Some((buffer, _)) = self.imported.get(&(start, span)) {
            return Ok(Some((*buffer, offset)));
        }
        if self.import_refused.contains(&(start, span)) {
            return Ok(None);
        }
        let handle = vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;
        let mut properties = vk::MemoryHostPointerPropertiesEXT::default();
        // SAFETY: `start` is a page-aligned host address the caller keeps
        // mapped (guest memory).
        unsafe {
            (host.fp().get_memory_host_pointer_properties_ext)(
                host.device(),
                handle,
                start as *const std::ffi::c_void,
                &mut properties,
            )
        }
        .result()
        .map_err(vk_error("vkGetMemoryHostPointerPropertiesEXT"))?;
        let device = self.gpu.device.clone();
        let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle);
        let info = vk::BufferCreateInfo::default()
            .size(span)
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut external);
        // SAFETY: valid create info; destroyed with the renderer.
        let buffer =
            unsafe { device.create_buffer(&info, None) }.map_err(vk_error("vkCreateBuffer"))?;
        // SAFETY: the buffer is live.
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let bits = properties.memory_type_bits & requirements.memory_type_bits;
        let Some(index) =
            (0..self.memory_types.memory_type_count).find(|index| bits & (1 << index) != 0)
        else {
            // SAFETY: unused buffer.
            unsafe { device.destroy_buffer(buffer, None) };
            return Err("no memory type can import host memory".into());
        };
        let mut import = vk::ImportMemoryHostPointerInfoEXT::default()
            .handle_type(handle)
            .host_pointer(start as *mut std::ffi::c_void);
        if span >= LARGE_ALLOCATION {
            eprintln!("gpu: {} MiB host import at {start:#x}", span >> 20);
        }
        let allocate = vk::MemoryAllocateInfo::default()
            .allocation_size(span)
            .memory_type_index(index)
            .push_next(&mut import);
        // SAFETY: the pointer is page aligned and `span` bytes stay mapped.
        let memory = match unsafe { device.allocate_memory(&allocate, None) } {
            Ok(memory) => memory,
            Err(_) => {
                // The caller copies instead.
                // SAFETY: unused buffer.
                unsafe { device.destroy_buffer(buffer, None) };
                self.import_refused.insert((start, span));
                return Ok(None);
            }
        };
        // SAFETY: fresh buffer and memory of the required size.
        unsafe { device.bind_buffer_memory(buffer, memory, 0) }
            .map_err(vk_error("vkBindBufferMemory"))?;
        self.imported.insert((start, span), (buffer, memory));
        Ok(Some((buffer, offset)))
    }

    #[track_caller]
    pub(crate) fn allocate(
        &self,
        requirements: vk::MemoryRequirements,
        flags: vk::MemoryPropertyFlags,
    ) -> Result<vk::DeviceMemory, String> {
        self.allocate_with(requirements, flags, false)
    }

    /// `allocate`, and with `addressable` memory buffers bound to it can
    /// have device addresses (`buffer_address`).
    #[track_caller]
    pub(crate) fn allocate_with(
        &self,
        requirements: vk::MemoryRequirements,
        flags: vk::MemoryPropertyFlags,
        addressable: bool,
    ) -> Result<vk::DeviceMemory, String> {
        // ECHORA_ALLOC_LOG (debugging): device memory allocated, summed by
        // caller, printed at each further GiB (frees are not counted).
        if std::env::var_os("ECHORA_ALLOC_LOG").is_some() {
            log_allocation(std::panic::Location::caller(), requirements.size);
        }
        // Allocations this large are unusual for guest-sized resources:
        // named with their caller, so a runaway size can be traced.
        if requirements.size >= LARGE_ALLOCATION {
            eprintln!(
                "gpu: {} MiB {flags:?} allocation at {}",
                requirements.size >> 20,
                std::panic::Location::caller()
            );
        }
        let index = (0..self.memory_types.memory_type_count)
            .find(|index| {
                requirements.memory_type_bits & (1 << index) != 0
                    && self.memory_types.memory_types[*index as usize]
                        .property_flags
                        .contains(flags)
            })
            .ok_or_else(|| format!("no memory type with {flags:?}"))?;
        let mut address_flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let mut info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(index);
        if addressable {
            info = info.push_next(&mut address_flags);
        }
        // SAFETY: valid allocate info; freed by the owner.
        unsafe { self.device().allocate_memory(&info, None) }.map_err(vk_error("vkAllocateMemory"))
    }

    /// The target for guest texture `key`, created (or recreated when its
    /// size or format changed) as needed.
    pub fn target(
        &mut self,
        key: u64,
        width: u32,
        height: u32,
        format: vk::Format,
    ) -> Result<&Target, String> {
        // Keeps the mip levels an existing target was created with.
        let levels = self.targets.get(&key).map_or(1, |target| target.levels);
        self.target_with_levels(key, width, height, format, levels)
    }

    /// The target for guest texture `key` with `levels` mip levels (draws
    /// and compute passes may render into any of them).
    pub fn target_with_levels(
        &mut self,
        key: u64,
        width: u32,
        height: u32,
        format: vk::Format,
        levels: u32,
    ) -> Result<&Target, String> {
        let shape = crate::draw::ImageShape::D2 { layers: 1 };
        self.shaped_target(key, width, height, shape, format, levels)
    }

    /// The target for guest texture `key` as a `shape` image: a 3D one for
    /// volumes compute passes write (colour lookup tables).
    pub fn shaped_target(
        &mut self,
        key: u64,
        width: u32,
        height: u32,
        shape: crate::draw::ImageShape,
        format: vk::Format,
        levels: u32,
    ) -> Result<&Target, String> {
        let largest = width.max(height).max(shape.depth()).max(1);
        let levels = levels.clamp(1, 32 - largest.leading_zeros());
        let stale = self.targets.get(&key).is_some_and(|target| {
            (
                target.width,
                target.height,
                target.format,
                target.levels,
                target.shape,
            ) != (width, height, format, levels, shape)
        });
        if stale {
            self.targets_recreated += 1;
            if std::env::var_os("ECHORA_GPU_FLOW").is_some() {
                let old = &self.targets[&key];
                eprintln!(
                    "target {key:x} recreated: {}x{} {:?} levels {} {:?} -> {width}x{height} {format:?} levels {levels} {shape:?}",
                    old.width, old.height, old.format, old.levels, old.shape
                );
            }
            self.flush()?;
            self.target_changes += 1;
            let old = self.targets.remove(&key).unwrap();
            // The same memory taller or shorter, as the application
            // re-initializes a texture over it (a shadow map 2048 wide and
            // 2048 or 4096 high): rows both shapes have keep their texels,
            // as block-linear rows of one width lie alike in memory.
            let flat = |shape: ImageShape| matches!(shape, ImageShape::D2 { layers: 1 });
            if (old.width, old.format) == (width, format) && flat(old.shape) && flat(shape) {
                self.create_target_keeping(key, &old, height, levels)?;
                self.flush()?;
                self.destroy_target(old);
                return Ok(&self.targets[&key]);
            }
            self.destroy_target(old);
        }
        if !self.targets.contains_key(&key) {
            let scale = self.scale_for(key, width, height, shape);
            let target = self.create_target(width, height, shape, format, levels, scale)?;
            // Tiny targets (exposure values adapting from their previous
            // frame) start from guest memory, as on the console; mip chains
            // (bloom) start zeroed, as levels sampled before they are drawn
            // showed garbage. Other targets keep the device's initial
            // contents: zeroing them (or copying their guest bytes) lost the
            // 3D scene (measured on the trainer selection screen).
            self.targets_created += 1;
            let tiny = target.width * target.height <= 64;
            let chain = target.levels > 1;
            self.register_views(&target);
            self.target_changes += 1;
            self.targets.insert(key, target);
            if tiny {
                self.fresh_targets.push(key);
            } else if chain || std::env::var_os("ECHORA_ZERO_TARGETS").is_some() {
                self.zero_target(key)?;
            }
        }
        Ok(&self.targets[&key])
    }

    /// Create target `key` `height` rows high in place of `old` (same
    /// width and format), copying the rows both have.
    fn create_target_keeping(
        &mut self,
        key: u64,
        old: &Target,
        height: u32,
        levels: u32,
    ) -> Result<(), String> {
        let shape = ImageShape::D2 { layers: 1 };
        let target = self.create_target(old.width, height, shape, old.format, levels, old.scale)?;
        self.targets_created += 1;
        self.register_views(&target);
        self.target_changes += 1;
        self.targets.insert(key, target);
        self.transition(key, vk::ImageLayout::GENERAL)?;
        let commands = self.commands()?;
        let layers = vk::ImageSubresourceLayers {
            aspect_mask: range(old.format).aspect_mask,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };
        let region = vk::ImageCopy {
            src_subresource: layers,
            dst_subresource: layers,
            extent: vk::Extent3D {
                width: old.extent(0).0,
                height: old.extent(0).1.min(old.scale.apply(height)),
                depth: 1,
            },
            ..Default::default()
        };
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
        let image = self.targets[&key].image;
        // SAFETY: recording; both images are live and in GENERAL layout.
        unsafe {
            let device = &self.gpu.device;
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
            device.cmd_copy_image(
                commands,
                old.image,
                vk::ImageLayout::GENERAL,
                image,
                vk::ImageLayout::GENERAL,
                &[region],
            );
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
        Ok(())
    }

    /// Clear every mip level of a target to zero: images start with
    /// undefined contents, and levels the application samples before
    /// writing (a bloom chain's) would show garbage.
    fn zero_target(&mut self, key: u64) -> Result<(), String> {
        self.transition(key, vk::ImageLayout::GENERAL)?;
        let commands = self.commands()?;
        let target = &self.targets[&key];
        let subresource = vk::ImageSubresourceRange {
            level_count: target.levels,
            layer_count: target.shape.layers(),
            ..range(target.format)
        };
        // SAFETY: recording; the image is live and in GENERAL layout.
        unsafe {
            if is_depth(target.format) {
                self.gpu.device.cmd_clear_depth_stencil_image(
                    commands,
                    target.image,
                    vk::ImageLayout::GENERAL,
                    &vk::ClearDepthStencilValue::default(),
                    &[subresource],
                );
            } else {
                self.gpu.device.cmd_clear_color_image(
                    commands,
                    target.image,
                    vk::ImageLayout::GENERAL,
                    &vk::ClearColorValue::default(),
                    &[subresource],
                );
            }
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
            self.gpu.device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
        Ok(())
    }

    pub fn existing_target(&self, key: u64) -> Option<&Target> {
        self.targets.get(&key)
    }

    /// Keys of the guest render targets (not internal staging images).
    /// Before recording a draw or dispatch: submit once a submission holds
    /// MAX_WORK_ITEMS of them. One submission is one GPU job, and the
    /// kernel resets the whole GPU when a job runs past its 10 s limit
    /// (headless runs once gathered 270 frames of work into one).
    pub(crate) fn bound_submission(&mut self) -> Result<(), String> {
        if self.work_items >= MAX_WORK_ITEMS {
            self.flush_async()?;
        }
        self.work_items += 1;
        Ok(())
    }

    /// Whether submissions are labelled (`label`).
    pub fn labelling() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| {
            std::env::var_os("ECHORA_LAST_SUBMIT").is_some()
                || std::env::var_os("ECHORA_SLOW_SUBMITS").is_some()
                || std::env::var_os("ECHORA_SUBMIT_LOG").is_some()
                || std::env::var_os("ECHORA_GPU_TIME").is_some_and(|value| value == "items")
        })
    }

    /// Debugging (ECHORA_GPU_TIME=items): time the draw or dispatch about
    /// to be recorded (`start`), or end its timing; it is named by the
    /// latest label without its present number.
    pub(crate) fn time_item(&mut self, start: bool) {
        let Some(timer) = self.timer.as_mut() else {
            return;
        };
        if !timer.timing_items() {
            return;
        }
        let name = self
            .labels
            .last()
            .map(|label| {
                let words: Vec<&str> = label.split_whitespace().collect();
                // "present N dispatch HASH ..." or "present N draw A/B ...".
                words.get(2..4).map_or(label.clone(), |kind| kind.join(" "))
            })
            .unwrap_or_default();
        timer.item(&self.gpu.device, self.commands, self.fence, &name, start);
    }

    /// Debugging: name a piece of work in the current submission.
    pub fn label(&mut self, label: String) {
        if self.labels.len() < 4096 {
            self.labels.push(label);
        }
    }

    /// Debugging: the sizes of the renderer's caches.
    pub fn cache_sizes(&self) -> Vec<(&'static str, usize)> {
        vec![
            ("targets", self.targets.len()),
            ("uploads", self.uploads.len()),
            ("retired uploads", self.retired_uploads.len()),
            ("samplers", self.samplers.len()),
            ("extra views", self.extra_views.len()),
            ("view infos", self.view_infos.len()),
            ("written", self.written.len()),
            ("pending reads", self.pending_reads.len()),
            ("completed reads", self.completed_reads.len()),
            ("fresh targets", self.fresh_targets.len()),
        ]
    }

    pub fn target_keys(&self) -> Vec<u64> {
        let mut keys: Vec<u64> = self
            .targets
            .keys()
            .copied()
            .filter(|key| *key < STAGING_KEYS)
            .collect();
        keys.sort_unstable();
        keys
    }

    /// The largest 1D/2D image extent the device supports.
    pub fn max_image_extent(&self) -> u32 {
        // SAFETY: the instance and physical device are live.
        let limits = unsafe {
            self.gpu
                .instance
                .get_physical_device_properties(self.gpu.physical)
        }
        .limits;
        limits
            .max_image_dimension1_d
            .min(limits.max_image_dimension2_d)
    }

    /// Whether images of `format` can be storage images (compute writes).
    /// The UNORM twin an sRGB format stores through, when that twin
    /// supports storage.
    pub fn storage_twin(&self, format: vk::Format) -> Option<vk::Format> {
        let srgb = matches!(
            format,
            vk::Format::R8G8B8A8_SRGB | vk::Format::B8G8R8A8_SRGB
        );
        srgb_twin(format).filter(|twin| srgb && self.supports_storage(*twin))
    }

    /// The format a storage image binding views `format` as: the format
    /// the program declares (an unsigned or signed word format of the same
    /// texel size), else `format` itself or its storage twin.
    pub(crate) fn storage_view_format(
        &self,
        format: vk::Format,
        declared: Option<vk::Format>,
    ) -> Option<vk::Format> {
        if let Some(declared) = declared.filter(|declared| *declared != format) {
            return (uint_alias(format) == uint_alias(declared)).then_some(declared);
        }
        if self.supports_storage(format) {
            Some(format)
        } else {
            self.storage_twin(format)
        }
    }

    pub fn supports_storage(&self, format: vk::Format) -> bool {
        // SAFETY: the instance and physical device are live.
        unsafe {
            self.gpu
                .instance
                .get_physical_device_format_properties(self.gpu.physical, format)
        }
        .optimal_tiling_features
        .contains(vk::FormatFeatureFlags::STORAGE_IMAGE)
    }

    /// The resolution scale a new target `key` of this extent gets: the
    /// renderer's for 2D targets larger than a few texels (tiny ones hold
    /// values such as exposure, not images).
    fn scale_for(&self, key: u64, width: u32, height: u32, shape: ImageShape) -> Scale {
        let scalable = matches!(shape, ImageShape::D2 { .. })
            && width * height > 64
            && key < STAGING_KEYS
            && !self.unscaled.contains(&key)
            && self.scale.apply(width.max(height)) <= self.max_image_extent();
        if scalable { self.scale } else { Scale::ONE }
    }

    /// The common scale of a pass's attachments: targets bound with ones
    /// of another scale are kept unscaled from then on.
    pub(crate) fn attachment_scale(
        &mut self,
        colors: &[u64],
        depth: Option<u64>,
    ) -> Result<Scale, String> {
        if self.scale.is_one() {
            return Ok(Scale::ONE);
        }
        let scales = || {
            colors
                .iter()
                .chain(depth.iter())
                .filter_map(|key| self.targets.get(key).map(|target| target.scale))
        };
        let first = scales().next().unwrap_or(Scale::ONE);
        if scales().all(|scale| scale == first) {
            return Ok(first);
        }
        for key in colors.iter().chain(depth.iter()) {
            self.keep_unscaled(*key)?;
        }
        Ok(Scale::ONE)
    }

    /// Whether a texture binding of `source` samples a scaled target.
    /// The format a texture source's texels are in (a target's, or the
    /// decoded upload's).
    pub fn source_format(&self, source: &crate::draw::TextureSource) -> Option<vk::Format> {
        use crate::draw::TextureSource;
        match source {
            TextureSource::Target(key) | TextureSource::TargetView { key, .. } => {
                self.targets.get(key).map(|target| target.format)
            }
            TextureSource::Upload { format, .. } => Some(*format),
        }
    }

    pub(crate) fn samples_scaled(&self, source: &crate::draw::TextureSource) -> bool {
        use crate::draw::TextureSource;
        match source {
            TextureSource::Target(key) | TextureSource::TargetView { key, .. } => self
                .targets
                .get(key)
                .is_some_and(|target| !target.scale.is_one()),
            TextureSource::Upload { .. } => false,
        }
    }

    /// The renderer's resolution scale (shaders are translated for it).
    /// Chooses the render target scale. Call before any target exists.
    pub fn set_resolution_scale(&mut self, scale: Scale) {
        self.scale = scale;
    }

    /// Keep compiled shader binaries in `directory` and load them from
    /// there in later runs instead of compiling the same SPIR-V again.
    pub fn set_shader_cache(&mut self, directory: std::path::PathBuf) {
        self.shader_cache = Some(directory);
    }

    pub fn resolution_scale(&self) -> Scale {
        self.scale
    }

    /// Keep target `key` at the guest's resolution from now on, replacing
    /// a scaled image by a downscaled copy (compute passes write storage
    /// images at guest coordinates).
    pub fn keep_unscaled(&mut self, key: u64) -> Result<(), String> {
        if !self.unscaled.insert(key) {
            return Ok(());
        }
        let Some(old) = self.targets.get(&key) else {
            return Ok(());
        };
        if old.scale.is_one() {
            return Ok(());
        }
        let (width, height, shape, format, levels) =
            (old.width, old.height, old.shape, old.format, old.levels);
        self.descaled += 1;
        self.transition(key, vk::ImageLayout::GENERAL)?;
        self.order_target(key)?;
        let target = self.create_target(width, height, shape, format, levels, Scale::ONE)?;
        self.register_views(&target);
        let old = self.targets.insert(key, target).unwrap();
        self.target_changes += 1;
        self.transition(key, vk::ImageLayout::GENERAL)?;
        let commands = self.commands()?;
        let filter = if is_depth(format) {
            vk::Filter::NEAREST
        } else {
            vk::Filter::LINEAR
        };
        let new = &self.targets[&key];
        let blits: Vec<vk::ImageBlit> = (0..levels)
            .map(|level| {
                let layers = vk::ImageSubresourceLayers {
                    aspect_mask: range(format).aspect_mask,
                    mip_level: level,
                    base_array_layer: 0,
                    layer_count: shape.layers(),
                };
                let corner = |(width, height): (u32, u32)| vk::Offset3D {
                    x: width as i32,
                    y: height as i32,
                    z: 1,
                };
                vk::ImageBlit {
                    src_subresource: layers,
                    src_offsets: [vk::Offset3D::default(), corner(old.extent(level))],
                    dst_subresource: layers,
                    dst_offsets: [vk::Offset3D::default(), corner(new.extent(level))],
                }
            })
            .collect();
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
        // SAFETY: recording; both images are live and in GENERAL layout.
        unsafe {
            let device = &self.gpu.device;
            device.cmd_blit_image(
                commands,
                old.image,
                vk::ImageLayout::GENERAL,
                new.image,
                vk::ImageLayout::GENERAL,
                &blits,
                filter,
            );
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
        self.flush()?;
        self.destroy_target(old);
        Ok(())
    }

    fn create_target(
        &self,
        width: u32,
        height: u32,
        shape: crate::draw::ImageShape,
        format: vk::Format,
        levels: u32,
        scale: Scale,
    ) -> Result<Target, String> {
        let storage = if self.supports_storage(format) || self.storage_twin(format).is_some() {
            vk::ImageUsageFlags::STORAGE
        } else {
            vk::ImageUsageFlags::empty()
        };
        let usage = if is_depth(format) {
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
        } else {
            vk::ImageUsageFlags::COLOR_ATTACHMENT | storage
        } | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST;
        let mut target = self.create_shaped_image(
            scale.apply(width),
            scale.apply(height),
            shape,
            format,
            usage,
            levels,
        )?;
        (target.width, target.height, target.scale) = (width, height, scale);
        Ok(target)
    }

    /// Keep target `from` (if any) under key `to` instead, dropping what
    /// `to` held.
    pub fn move_target(&mut self, from: u64, to: u64) -> Result<(), String> {
        let Some(target) = self.targets.remove(&from) else {
            return Ok(());
        };
        self.target_changes += 1;
        if let Some(replaced) = self.targets.remove(&to) {
            self.flush()?;
            self.destroy_target(replaced);
        }
        self.targets.insert(to, target);
        self.fresh_targets.retain(|fresh| *fresh != from);
        Ok(())
    }

    /// Targets created since the last call.
    pub fn take_fresh_targets(&mut self) -> Vec<u64> {
        std::mem::take(&mut self.fresh_targets)
    }

    /// Target `key` gets its contents otherwise (a clear or a copy): it
    /// needs no initialization.
    pub fn defined(&mut self, key: u64) {
        self.fresh_targets.retain(|fresh| *fresh != key);
    }

    /// Copy an upload's level 0 into level 0 of target `key` (same format
    /// and size only).
    pub fn initialize_target(
        &mut self,
        key: u64,
        source: &crate::draw::TextureSource,
    ) -> Result<(), String> {
        let crate::draw::TextureSource::Upload {
            key: upload,
            width,
            height,
            format,
            shape: crate::draw::ImageShape::D2 { layers: 1 },
            ..
        } = source
        else {
            return Err("only single 2D uploads initialize targets".into());
        };
        let target = self.targets.get(&key).ok_or("no such target")?;
        if (target.width, target.height, target.format) != (*width, *height, *format) {
            return Err("upload and target differ".into());
        }
        self.upload(source)?;
        self.transition(key, vk::ImageLayout::GENERAL)?;
        let commands = self.commands()?;
        let (from, to) = (self.uploads[upload].1.image, self.targets[&key].image);
        let (scaled_width, scaled_height) = self.targets[&key].extent(0);
        let layers = vk::ImageSubresourceLayers {
            aspect_mask: range(*format).aspect_mask,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };
        // A scaled target: the guest's texels, scaled up.
        let corner = |width: u32, height: u32| vk::Offset3D {
            x: width as i32,
            y: height as i32,
            z: 1,
        };
        let blit = vk::ImageBlit {
            src_subresource: layers,
            src_offsets: [vk::Offset3D::default(), corner(*width, *height)],
            dst_subresource: layers,
            dst_offsets: [vk::Offset3D::default(), corner(scaled_width, scaled_height)],
        };
        let scaled = (scaled_width, scaled_height) != (*width, *height);
        let region = vk::ImageCopy::default()
            .src_subresource(layers)
            .dst_subresource(layers)
            .extent(vk::Extent3D {
                width: *width,
                height: *height,
                depth: 1,
            });
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
        // SAFETY: recording; both images are live and in GENERAL layout.
        unsafe {
            if scaled {
                self.gpu.device.cmd_blit_image(
                    commands,
                    from,
                    vk::ImageLayout::GENERAL,
                    to,
                    vk::ImageLayout::GENERAL,
                    &[blit],
                    if is_depth(*format) {
                        vk::Filter::NEAREST
                    } else {
                        vk::Filter::LINEAR
                    },
                );
            } else {
                self.gpu.device.cmd_copy_image(
                    commands,
                    from,
                    vk::ImageLayout::GENERAL,
                    to,
                    vk::ImageLayout::GENERAL,
                    &[region],
                );
            }
            self.gpu.device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
        Ok(())
    }

    /// A 2D view of one mip level of target `key` (the target's own view
    /// when it has a single level), for attachments and storage images.
    /// A view of mip level `level` of target `key` in `format` (a format
    /// the target was created to allow), for storage through a twin.
    pub(crate) fn level_view_as(
        &mut self,
        key: u64,
        level: u32,
        format: vk::Format,
    ) -> Result<vk::ImageView, String> {
        let target = self.targets.get(&key).ok_or("no such target")?;
        if format == target.format {
            return self.level_view(key, level);
        }
        if level >= target.levels {
            return Err(format!("level {level} of a {}-level target", target.levels));
        }
        // Keyed apart from level_view's (i32::MAX - level) entries.
        let cache = (target.image, i32::MIN + format.as_raw() * 32 + level as i32);
        if let Some(view) = self.extra_views.get(&cache) {
            return Ok(*view);
        }
        let mut usage = vk::ImageViewUsageCreateInfo::default().usage(vk::ImageUsageFlags::STORAGE);
        let info = vk::ImageViewCreateInfo::default()
            .image(target.image)
            .view_type(own_view_type(target.shape))
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                base_mip_level: level,
                level_count: 1,
                layer_count: target.shape.layers(),
                ..range(target.format)
            })
            .push_next(&mut usage);
        // SAFETY: the image is live; the view is destroyed with it.
        let view = unsafe { self.gpu.device.create_image_view(&info, None) }
            .map_err(vk_error("vkCreateImageView"))?;
        self.extra_views.insert(cache, view);
        Ok(view)
    }

    pub(crate) fn level_view(&mut self, key: u64, level: u32) -> Result<vk::ImageView, String> {
        let target = self.targets.get(&key).ok_or("no such target")?;
        if target.levels == 1 && level == 0 {
            return Ok(target.view);
        }
        if level >= target.levels {
            return Err(format!("level {level} of a {}-level target", target.levels));
        }
        let cache = (target.image, i32::MAX - level as i32);
        if let Some(view) = self.extra_views.get(&cache) {
            return Ok(*view);
        }
        let info = vk::ImageViewCreateInfo::default()
            .image(target.image)
            .view_type(own_view_type(target.shape))
            .format(target.format)
            .subresource_range(vk::ImageSubresourceRange {
                base_mip_level: level,
                layer_count: target.shape.layers(),
                ..range(target.format)
            });
        // SAFETY: the image is live; the view is destroyed with it.
        let view = unsafe { self.gpu.device.create_image_view(&info, None) }
            .map_err(vk_error("vkCreateImageView"))?;
        self.extra_views.insert(cache, view);
        Ok(view)
    }

    /// A 2D view of mip level `level` of array layer (or cube face)
    /// `layer` of target `key`, for attachments: `level_view` for targets
    /// of one layer.
    pub(crate) fn attachment_view(
        &mut self,
        key: u64,
        level: u32,
        layer: u32,
    ) -> Result<vk::ImageView, String> {
        let target = self.targets.get(&key).ok_or("no such target")?;
        let layers = target.shape.layers();
        if layers == 1 || matches!(target.shape, ImageShape::D3 { .. }) {
            return self.level_view(key, level);
        }
        if level >= target.levels || layer >= layers {
            return Err(format!(
                "level {level} layer {layer} of a {}-level {layers}-layer target",
                target.levels
            ));
        }
        let cache = (target.image, i32::MIN + 1 + (layer << 5 | level) as i32);
        if let Some(view) = self.extra_views.get(&cache) {
            return Ok(*view);
        }
        let info = vk::ImageViewCreateInfo::default()
            .image(target.image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(target.format)
            .subresource_range(vk::ImageSubresourceRange {
                base_mip_level: level,
                base_array_layer: layer,
                ..range(target.format)
            });
        // SAFETY: the image is live; the view is destroyed with it.
        let view = unsafe { self.gpu.device.create_image_view(&info, None) }
            .map_err(vk_error("vkCreateImageView"))?;
        self.extra_views.insert(cache, view);
        Ok(view)
    }

    #[track_caller]
    fn create_shaped_image(
        &self,
        width: u32,
        height: u32,
        shape: crate::draw::ImageShape,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        levels: u32,
    ) -> Result<Target, String> {
        use crate::draw::ImageShape;
        let largest = width.max(height).max(shape.depth()).max(1);
        let levels = levels.clamp(1, 32 - largest.leading_zeros());
        let (image_type, flags) = match shape {
            ImageShape::D1 { .. } => (vk::ImageType::TYPE_1D, vk::ImageCreateFlags::empty()),
            ImageShape::D2 { .. } => (vk::ImageType::TYPE_2D, vk::ImageCreateFlags::empty()),
            ImageShape::D3 { .. } => (vk::ImageType::TYPE_3D, vk::ImageCreateFlags::empty()),
            ImageShape::Cube { .. } => (
                vk::ImageType::TYPE_2D,
                vk::ImageCreateFlags::CUBE_COMPATIBLE,
            ),
        };
        // Formats with an sRGB twin may be viewed as either (an NVN texture
        // view's format): listed, so the driver keeps its compression.
        // Storage images stay single-format: sRGB formats have no storage
        // use, which a listed view format would need.
        // An sRGB format stores through its UNORM twin (image stores do
        // no sRGB encoding): STORAGE then applies to the twin's view only,
        // which EXTENDED_USAGE allows.
        let store_twin = self.storage_twin(format);
        let twin = srgb_twin(format)
            .filter(|_| !usage.contains(vk::ImageUsageFlags::STORAGE) || store_twin.is_some());
        // A storage target may also be written as unsigned words of its
        // texel size (a shader's R32UI view of an R11G11B10 or RGBA8
        // target): that alias is listed too.
        let alias = uint_alias(format)
            .filter(|alias| *alias != format && usage.contains(vk::ImageUsageFlags::STORAGE));
        let mut listed = vec![format];
        listed.extend(twin);
        listed.extend(alias);
        // R8 and R16 words may also be declared signed.
        listed.extend(alias.and_then(|alias| match alias {
            vk::Format::R8_UINT => Some(vk::Format::R8_SINT),
            vk::Format::R16_UINT => Some(vk::Format::R16_SINT),
            _ => None,
        }));
        listed.retain({
            let mut seen = Vec::new();
            move |format| {
                let new = !seen.contains(format);
                seen.push(*format);
                new
            }
        });
        let mut format_list = vk::ImageFormatListCreateInfo::default().view_formats(&listed);
        let twin = twin.or(alias);
        let flags = if twin.is_some() {
            flags | vk::ImageCreateFlags::MUTABLE_FORMAT
        } else {
            flags
        };
        let flags = if store_twin.is_some() && usage.contains(vk::ImageUsageFlags::STORAGE) {
            flags | vk::ImageCreateFlags::EXTENDED_USAGE
        } else {
            flags
        };
        let mut info = vk::ImageCreateInfo::default()
            .flags(flags)
            .image_type(image_type)
            .format(format)
            .extent(vk::Extent3D {
                width,
                height: if image_type == vk::ImageType::TYPE_1D {
                    1
                } else {
                    height
                },
                depth: shape.depth(),
            })
            .mip_levels(levels)
            .array_layers(shape.layers())
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        if twin.is_some() {
            info = info.push_next(&mut format_list);
        }
        let device = self.device();
        // SAFETY: valid create info; destroyed by destroy_target.
        let image =
            unsafe { device.create_image(&info, None) }.map_err(vk_error("vkCreateImage"))?;
        // SAFETY: the image is live.
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let memory = self.allocate(requirements, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        // SAFETY: fresh image and memory of the required size.
        unsafe { device.bind_image_memory(image, memory, 0) }
            .map_err(vk_error("vkBindImageMemory"))?;
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(own_view_type(shape))
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                level_count: levels,
                layer_count: shape.layers(),
                ..range(format)
            });
        // SAFETY: as above.
        let view = unsafe { device.create_image_view(&view_info, None) }
            .map_err(vk_error("vkCreateImageView"))?;
        let sampled_view = if has_stencil(format) {
            let depth_only = view_info.subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                level_count: levels,
                ..range(format)
            });
            // SAFETY: as above.
            unsafe { device.create_image_view(&depth_only, None) }
                .map_err(vk_error("vkCreateImageView"))?
        } else {
            view
        };
        Ok(Target {
            image,
            view,
            sampled_view,
            shape,
            levels,
            memory,
            format,
            width,
            height,
            scale: Scale::ONE,
            layout: vk::ImageLayout::UNDEFINED,
        })
    }

    fn destroy_target(&mut self, target: Target) {
        self.view_infos.retain(|_, info| info.image != target.image);
        let mut extra = Vec::new();
        self.extra_views.retain(|(image, _), view| {
            let other = *image != target.image;
            if !other {
                extra.push(*view);
            }
            other
        });
        let mut views = extra.clone();
        views.extend([target.view, target.sampled_view]);
        self.forget_pool_views(&views);
        // SAFETY: the caller flushed, so no work uses the target.
        unsafe {
            for view in extra {
                self.device().destroy_image_view(view, None);
            }
            if target.sampled_view != target.view {
                self.device().destroy_image_view(target.sampled_view, None);
            }
            self.device().destroy_image_view(target.view, None);
            self.device().destroy_image(target.image, None);
            self.device().free_memory(target.memory, None);
        }
    }

    /// Record a full barrier moving `key`'s image to `layout`.
    pub(crate) fn transition(&mut self, key: u64, layout: vk::ImageLayout) -> Result<(), String> {
        let commands = self.commands()?;
        let target = self.targets.get_mut(&key).ok_or("no such target")?;
        if target.layout == layout {
            return Ok(());
        }
        let barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
            .old_layout(target.layout)
            .new_layout(layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(target.image)
            // Every level and layer: the whole image changes layout (other
            // levels and layers were left in UNDEFINED before).
            .subresource_range(vk::ImageSubresourceRange {
                level_count: vk::REMAINING_MIP_LEVELS,
                layer_count: vk::REMAINING_ARRAY_LAYERS,
                ..range(target.format)
            });
        target.layout = layout;
        // SAFETY: recording; the image is live.
        unsafe {
            self.gpu.device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            )
        };
        Ok(())
    }

    /// Make earlier rendering visible before work that reads (or writes)
    /// `images`: one memory barrier when any of them was rendered to since
    /// the last one. Rendering passes are recorded one per draw, and
    /// nothing else orders a pass's attachment writes before a later
    /// sample, copy or read back of the same image (stale tiles otherwise).
    pub(crate) fn order_after_rendering(&mut self, images: &[vk::Image]) -> Result<(), String> {
        if !images.iter().any(|image| self.written.contains(image)) {
            return Ok(());
        }
        self.memory_barrier()
    }

    /// A full memory barrier; everything written before is visible after.
    pub(crate) fn memory_barrier(&mut self) -> Result<(), String> {
        let commands = self.commands()?;
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
        // SAFETY: recording outside rendering.
        unsafe {
            self.gpu.device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            )
        };
        self.written.clear();
        self.hazard_barriers += 1;
        Ok(())
    }

    /// Order a transfer reading or writing target `key` after rendering.
    fn order_target(&mut self, key: u64) -> Result<(), String> {
        let image = self.targets.get(&key).ok_or("no such target")?.image;
        self.order_after_rendering(&[image])
    }

    /// Copy `source` rect (x, y, width, height) of target `from` into
    /// `destination` rect of target `to`, scaling (linear filter) when the
    /// sizes differ. Rects are clipped to their targets.
    pub fn copy_target(
        &mut self,
        from: u64,
        source: [i32; 4],
        to: u64,
        destination: [i32; 4],
    ) -> Result<(), String> {
        if from == to {
            return Err("copy within one target".into());
        }
        self.transition(from, vk::ImageLayout::GENERAL)?;
        self.transition(to, vk::ImageLayout::GENERAL)?;
        self.order_target(from)?;
        self.order_target(to)?;
        let commands = self.commands()?;
        let (source_target, destination_target) = (&self.targets[&from], &self.targets[&to]);
        let bounds = |target: &Target| [0, 0, target.width as i32, target.height as i32];
        let corners = |[x, y, width, height]: [i32; 4]| {
            [
                vk::Offset3D { x, y, z: 0 },
                vk::Offset3D {
                    x: x + width,
                    y: y + height,
                    z: 1,
                },
            ]
        };
        let source = source_target.scaled(clip(source, bounds(source_target)));
        let destination = destination_target.scaled(clip(destination, bounds(destination_target)));
        if source[2] <= 0 || source[3] <= 0 || destination[2] <= 0 || destination[3] <= 0 {
            return Ok(());
        }
        let layers = |format| {
            let range = range(format);
            vk::ImageSubresourceLayers {
                aspect_mask: range.aspect_mask,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            }
        };
        let region = vk::ImageBlit::default()
            .src_subresource(layers(source_target.format))
            .src_offsets(corners(source))
            .dst_subresource(layers(destination_target.format))
            .dst_offsets(corners(destination));
        let filter = if is_depth(source_target.format) || source[2..] == destination[2..] {
            vk::Filter::NEAREST
        } else {
            vk::Filter::LINEAR
        };
        // SAFETY: recording; both images are live and in GENERAL layout.
        unsafe {
            self.gpu.device.cmd_blit_image(
                commands,
                source_target.image,
                vk::ImageLayout::GENERAL,
                destination_target.image,
                vk::ImageLayout::GENERAL,
                &[region],
                filter,
            )
        };
        Ok(())
    }

    /// Clear `rect` (x, y, width, height; the whole target when None) of a
    /// color target to `color`, written only where `mask` (bit per channel)
    /// is set. Masked clears are not modeled.
    pub fn clear_color(
        &mut self,
        key: u64,
        rect: Option<[i32; 4]>,
        color: vk::ClearColorValue,
    ) -> Result<(), String> {
        self.clear_color_level(key, 0, 0, rect, color)
    }

    /// Clear `rect` of mip level `level` of array layer `layer` of a color
    /// target (see `clear_color`).
    pub fn clear_color_level(
        &mut self,
        key: u64,
        level: u32,
        layer: u32,
        rect: Option<[i32; 4]>,
        color: vk::ClearColorValue,
    ) -> Result<(), String> {
        self.transition(key, vk::ImageLayout::GENERAL)?;
        let commands = self.commands()?;
        let target = &self.targets[&key];
        if level >= target.levels || layer >= target.shape.layers() {
            return Err(format!(
                "level {level} layer {layer} of a {}-level {}-layer target",
                target.levels,
                target.shape.layers()
            ));
        }
        let (whole_width, whole_height) = target.extent(level);
        let whole = [0, 0, whole_width as i32, whole_height as i32];
        let rect = rect.map(|rect| target.scaled(rect));
        let [x, y, width, height] = clip(rect.unwrap_or(whole), whole);
        if [x, y, width, height] != whole {
            self.partial_clears += 1;
        }
        if width <= 0 || height <= 0 {
            return Ok(());
        }
        if [x, y, width, height] == whole {
            // SAFETY: recording; the image is in GENERAL layout.
            unsafe {
                self.gpu.device.cmd_clear_color_image(
                    commands,
                    target.image,
                    vk::ImageLayout::GENERAL,
                    &color,
                    &[vk::ImageSubresourceRange {
                        base_mip_level: level,
                        base_array_layer: layer,
                        ..range(target.format)
                    }],
                )
            };
            return Ok(());
        }
        // A partial clear runs as an attachment clear in a render pass.
        self.begin_rendering(&[key], &[level], &[layer], None, 0)?;
        let commands = self.recording_commands()?;
        let attachment = vk::ClearAttachment {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            color_attachment: 0,
            clear_value: vk::ClearValue { color },
        };
        let rect = vk::ClearRect {
            rect: vk::Rect2D {
                offset: vk::Offset2D { x, y },
                extent: vk::Extent2D {
                    width: width as u32,
                    height: height as u32,
                },
            },
            base_array_layer: 0,
            layer_count: 1,
        };
        // SAFETY: inside the rendering begun above.
        unsafe {
            self.gpu
                .device
                .cmd_clear_attachments(commands, &[attachment], &[rect]);
        }
        Ok(())
    }

    /// Clear the depth and/or stencil of array layer `layer` of a depth
    /// target within `rect`.
    pub fn clear_depth_stencil(
        &mut self,
        key: u64,
        layer: u32,
        rect: Option<[i32; 4]>,
        depth: Option<f32>,
        stencil: Option<u32>,
    ) -> Result<(), String> {
        let format = self.targets.get(&key).ok_or("no such target")?.format;
        let mut aspects = vk::ImageAspectFlags::empty();
        if depth.is_some() {
            aspects |= vk::ImageAspectFlags::DEPTH;
        }
        if stencil.is_some() && has_stencil(format) {
            aspects |= vk::ImageAspectFlags::STENCIL;
        }
        if aspects.is_empty() {
            return Ok(());
        }
        let target = &self.targets[&key];
        let (whole_width, whole_height) = target.extent(0);
        let whole = [0, 0, whole_width as i32, whole_height as i32];
        let rect = rect.map(|rect| target.scaled(rect));
        let [x, y, width, height] = clip(rect.unwrap_or(whole), whole);
        if width <= 0 || height <= 0 {
            return Ok(());
        }
        self.begin_rendering(&[], &[], &[], Some(key), layer)?;
        let commands = self.recording_commands()?;
        let attachment = vk::ClearAttachment {
            aspect_mask: aspects,
            color_attachment: 0,
            clear_value: vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: depth.unwrap_or(0.0),
                    stencil: stencil.unwrap_or(0),
                },
            },
        };
        let rect = vk::ClearRect {
            rect: vk::Rect2D {
                offset: vk::Offset2D { x, y },
                extent: vk::Extent2D {
                    width: width as u32,
                    height: height as u32,
                },
            },
            base_array_layer: 0,
            layer_count: 1,
        };
        // SAFETY: inside the rendering begun above.
        unsafe {
            self.gpu
                .device
                .cmd_clear_attachments(commands, &[attachment], &[rect]);
        }
        Ok(())
    }

    /// Begin dynamic rendering into color targets `colors` and an optional
    /// depth target, limited to `area`.
    /// Each color target renders into its mip level in `levels` and its
    /// layer in `layers` (0 when missing); the depth target into layer
    /// `depth_layer`.
    pub(crate) fn begin_rendering(
        &mut self,
        colors: &[u64],
        levels: &[u32],
        layers: &[u32],
        depth: Option<u64>,
        depth_layer: u32,
    ) -> Result<(), String> {
        let level = |index: usize| levels.get(index).copied().unwrap_or(0);
        let mut views = Vec::with_capacity(colors.len());
        for (index, key) in colors.iter().enumerate() {
            let layer = layers.get(index).copied().unwrap_or(0);
            views.push(self.attachment_view(*key, level(index), layer)?);
        }
        let depth_view = match depth {
            Some(key) => Some(self.attachment_view(key, 0, depth_layer)?),
            None => None,
        };
        // The render area is the attachments' common extent (the
        // scissor limits each draw, a clear its rectangle), so draws with
        // other scissors continue one pass.
        let extent = |key: &u64, level: u32| {
            let (width, height) = self.targets[key].extent(level);
            (width as i32, height as i32)
        };
        let (smallest_width, smallest_height) = colors
            .iter()
            .enumerate()
            .map(|(index, key)| extent(key, level(index)))
            .chain(depth.iter().map(|key| extent(key, 0)))
            .fold((i32::MAX, i32::MAX), |(w, h), (tw, th)| {
                (w.min(tw), h.min(th))
            });
        let whole = [0, 0, smallest_width, smallest_height];
        if self
            .open_pass
            .as_ref()
            .is_some_and(|(open, open_depth, open_area)| {
                *open == views && *open_depth == depth_view && *open_area == whole
            })
        {
            return Ok(());
        }
        self.end_pass();
        for key in colors.iter().chain(depth.iter()) {
            self.transition(*key, vk::ImageLayout::GENERAL)?;
        }
        // Another pass's writes to these attachments come first.
        let attachments: Vec<vk::Image> = colors
            .iter()
            .chain(depth.iter())
            .map(|key| self.targets[key].image)
            .collect();
        if attachments != self.last_attachments {
            self.order_after_rendering(&attachments)?;
        }
        self.written.extend(attachments.iter().copied());
        self.last_attachments = attachments;
        let commands = self.commands()?;
        let attachments: Vec<_> = views
            .iter()
            .map(|view| {
                vk::RenderingAttachmentInfo::default()
                    .image_view(*view)
                    .image_layout(vk::ImageLayout::GENERAL)
                    .load_op(vk::AttachmentLoadOp::LOAD)
                    .store_op(vk::AttachmentStoreOp::STORE)
            })
            .collect();
        let depth_attachment = depth_view.map(|view| {
            vk::RenderingAttachmentInfo::default()
                .image_view(view)
                .image_layout(vk::ImageLayout::GENERAL)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE)
        });
        // The render area may not exceed any attachment (the guest binds
        // targets of different sizes together; it renders within each).
        let [x, y, width, height] = whole;
        let mut info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x, y },
                extent: vk::Extent2D {
                    width: width.max(0) as u32,
                    height: height.max(0) as u32,
                },
            })
            .layer_count(1)
            .color_attachments(&attachments);
        if let Some(depth_attachment) = depth_attachment.as_ref() {
            info = info.depth_attachment(depth_attachment);
            if depth.is_some_and(|key| has_stencil(self.targets[&key].format)) {
                info = info.stencil_attachment(depth_attachment);
            }
        }
        // SAFETY: recording; attachments are live and in GENERAL layout.
        unsafe { self.gpu.device.cmd_begin_rendering(commands, &info) };
        self.open_pass = Some((views, depth_view, whole));
        Ok(())
    }

    /// Read a color target back as RGBA f32 texels (row-major).
    /// Record a scaled blit of color target `key` into a swapchain image,
    /// leaving that image ready to present.
    pub(crate) fn blit_to_swapchain(
        &mut self,
        key: u64,
        image: vk::Image,
        (width, height): (u32, u32),
        extent: vk::Extent2D,
    ) -> Result<(), String> {
        self.transition(key, vk::ImageLayout::GENERAL)?;
        let commands = self.commands()?;
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let to_transfer = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(image)
            .subresource_range(range);
        let to_present = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::empty())
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .image(image)
            .subresource_range(range);
        let layers = vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };
        // By default the frame keeps its aspect ratio: scaled to fit and
        // centred, with black bars filling the rest of the window
        // (ECHORA_SCALING_MODE chooses another fit).
        let (mode, filter) = *output_options();
        let fitted = present_rect(mode, (width, height), (extent.width, extent.height));
        let blit = vk::ImageBlit {
            src_subresource: layers,
            // Targets keep the console's row order: row 0 is the top.
            src_offsets: [
                vk::Offset3D::default(),
                vk::Offset3D {
                    x: width as i32,
                    y: height as i32,
                    z: 1,
                },
            ],
            dst_subresource: layers,
            dst_offsets: [
                vk::Offset3D {
                    x: fitted[0],
                    y: fitted[1],
                    z: 0,
                },
                vk::Offset3D {
                    x: fitted[2],
                    y: fitted[3],
                    z: 1,
                },
            ],
        };
        let letterboxed = fitted != [0, 0, extent.width as i32, extent.height as i32];
        let source = self.targets[&key].image;
        // SAFETY: recording; the target is in GENERAL layout and the
        // swapchain image was acquired.
        unsafe {
            let device = &self.gpu.device;
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)],
                &[],
                &[to_transfer],
            );
            if letterboxed {
                device.cmd_clear_color_image(
                    commands,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                    &[range],
                );
                device.cmd_pipeline_barrier(
                    commands,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)],
                    &[],
                    &[],
                );
            }
            device.cmd_blit_image(
                commands,
                source,
                vk::ImageLayout::GENERAL,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[blit],
                filter,
            );
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_present],
            );
        }
        Ok(())
    }

    pub fn read_target(&mut self, key: u64) -> Result<Vec<[f32; 4]>, String> {
        self.read_target_level(key, 0)
    }

    /// Release the upload kept for `key` (and its one-dimensional alias):
    /// the texture it held was finalized.
    pub fn forget_upload(&mut self, key: u64) {
        for key in [key, key ^ 1 << 63] {
            if let Some((_, target)) = self.uploads.remove(&key) {
                self.retired_uploads.push(target);
            }
        }
    }

    /// Debugging: a zeroed host-visible buffer of `size` bytes that shaders
    /// can write through its device address; returns its index (for
    /// `read_debug_buffer`) and that address.
    pub fn debug_buffer(&mut self, size: u64) -> Result<(usize, u64), String> {
        let buffer = self.host_buffer_with(size, vk::BufferUsageFlags::STORAGE_BUFFER, true)?;
        buffer.write(0, &vec![0; size as usize]);
        let address = self
            .buffer_address(buffer.buffer)
            .ok_or("debug buffers need buffer device addresses")?;
        self.debug_buffers.push(buffer);
        Ok((self.debug_buffers.len() - 1, address))
    }

    /// Debugging: a `debug_buffer`'s bytes as they are now.
    pub fn read_debug_buffer(&self, index: usize) -> Vec<u8> {
        self.debug_buffers
            .get(index)
            .map_or_else(Vec::new, |buffer| buffer.read_bytes(buffer.size as usize))
    }

    /// Make sure a guest texture is uploaded (debugging reads).
    pub fn texture_view_for(&mut self, source: &crate::draw::TextureSource) -> Result<(), String> {
        self.texture_view(source, 2).map(|_| ())
    }

    /// Debugging: read mip level `level` of upload `key` back as RGBA f32
    /// texels (as a target would be).
    pub fn read_upload_level(&mut self, key: u64, level: u32) -> Result<Vec<[f32; 4]>, String> {
        self.read_upload_slice(key, level, 0)
    }

    /// Debugging: as `read_upload_level`, slice `slice` of a 3D upload.
    pub fn read_upload_slice(
        &mut self,
        key: u64,
        level: u32,
        slice: u32,
    ) -> Result<Vec<[f32; 4]>, String> {
        let (generation, target) = self.uploads.remove(&key).ok_or("no such upload")?;
        let temporary = u64::MAX - 1;
        self.targets.insert(temporary, target);
        let texels = self
            .read_level(temporary, level, false, slice)
            .map(|(texels, ..)| texels);
        let target = self.targets.remove(&temporary).unwrap();
        self.uploads.insert(key, (generation, target));
        texels
    }

    /// Read mip level `level` of a color target back as RGBA f32 texels
    /// (at the guest's resolution).
    pub fn read_target_level(&mut self, key: u64, level: u32) -> Result<Vec<[f32; 4]>, String> {
        self.read_level(key, level, false, 0)
            .map(|(texels, ..)| texels)
    }

    /// Level 0 of a color target at the image's own (scaled) resolution,
    /// with that width and height: the frame as the window shows it.
    pub fn read_target_scaled(&mut self, key: u64) -> Result<(Vec<[f32; 4]>, u32, u32), String> {
        self.read_level(key, 0, true, 0)
    }

    /// Level `level` (slice `slice` of a 3D target) as RGBA f32 texels.
    fn read_level(
        &mut self,
        key: u64,
        level: u32,
        full: bool,
        slice: u32,
    ) -> Result<(Vec<[f32; 4]>, u32, u32), String> {
        let (width, height, format, scaled) = {
            let target = self.targets.get(&key).ok_or("no such target")?;
            if level >= target.levels {
                return Err(format!("level {level} of a {}-level target", target.levels));
            }
            let scaled = target.extent(level);
            let (width, height) = if full {
                scaled
            } else {
                (
                    (target.width >> level).max(1),
                    (target.height >> level).max(1),
                )
            };
            (width, height, target.format, scaled)
        };
        // Depth targets cannot be blitted to a color image: their depth
        // aspect is copied as it is and read as (depth, 0, 0, 1).
        if is_depth(format) {
            if level != 0 || slice != 0 || (full && scaled != (width, height)) {
                return Err("a depth target reads back at level 0, unscaled".into());
            }
            let bytes = self.read_target_bytes(key, 4)?;
            let depth = |bytes: &[u8]| match format {
                vk::Format::D16_UNORM => {
                    f32::from(u16::from_le_bytes([bytes[0], bytes[1]])) / 65535.0
                }
                vk::Format::D24_UNORM_S8_UINT | vk::Format::X8_D24_UNORM_PACK32 => {
                    let word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                    (word & 0xff_ffff) as f32 / 16_777_215.0
                }
                _ => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            };
            let size = if format == vk::Format::D16_UNORM {
                2
            } else {
                4
            };
            let texels = bytes
                .chunks_exact(size)
                .map(|texel| [depth(texel), 0.0, 0.0, 1.0])
                .collect();
            return Ok((texels, width, height));
        }
        // Blit to an RGBA32F image so every color format reads back alike.
        let staging_key = u64::MAX;
        self.target(staging_key, width, height, vk::Format::R32G32B32A32_SFLOAT)?;
        self.transition(key, vk::ImageLayout::GENERAL)?;
        self.order_target(key)?;
        self.transition(staging_key, vk::ImageLayout::GENERAL)?;
        let size = u64::from(width) * u64::from(height) * 16;
        let buffer = self.host_buffer(size, vk::BufferUsageFlags::TRANSFER_DST)?;
        let commands = self.commands()?;
        let layers = vk::ImageSubresourceLayers {
            aspect_mask: aspect(format),
            mip_level: level,
            base_array_layer: 0,
            layer_count: 1,
        };
        let corner = vk::Offset3D {
            x: width as i32,
            y: height as i32,
            z: 1,
        };
        let blit = vk::ImageBlit {
            src_subresource: layers,
            src_offsets: [
                vk::Offset3D {
                    x: 0,
                    y: 0,
                    z: slice as i32,
                },
                vk::Offset3D {
                    x: scaled.0 as i32,
                    y: scaled.1 as i32,
                    z: slice as i32 + 1,
                },
            ],
            dst_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                ..layers
            },
            dst_offsets: [vk::Offset3D::default(), corner],
        };
        let copy = vk::BufferImageCopy {
            image_subresource: blit.dst_subresource,
            image_extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            ..Default::default()
        };
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
        let (source, staging) = (self.targets[&key].image, self.targets[&staging_key].image);
        // SAFETY: recording; images in GENERAL layout; the buffer fits.
        unsafe {
            let device = &self.gpu.device;
            device.cmd_blit_image(
                commands,
                source,
                vk::ImageLayout::GENERAL,
                staging,
                vk::ImageLayout::GENERAL,
                &[blit],
                vk::Filter::NEAREST,
            );
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
            device.cmd_copy_image_to_buffer(
                commands,
                staging,
                vk::ImageLayout::GENERAL,
                buffer.buffer,
                &[copy],
            );
        }
        self.flush()?;
        let texels = buffer.read_f32x4((width * height) as usize);
        buffer.destroy(self.device());
        Ok((texels, width, height))
    }

    /// Level 0 of a color target as its own format's bytes, rows packed:
    /// `texel_bytes` is that format's texel size, which the caller knows.
    pub fn read_target_bytes(&mut self, key: u64, texel_bytes: u32) -> Result<Vec<u8>, String> {
        let key = self.unscaled_copy(key)?;
        let (image, width, height, format) = {
            let target = self.targets.get(&key).ok_or("no such target")?;
            (target.image, target.width, target.height, target.format)
        };
        // Depth targets: their depth aspect (2 bytes per texel for D16, 4
        // for the others: a float for the D32 formats, the depth in the
        // low 24 bits for D24).
        let (aspect_mask, texel_bytes) = if is_depth(format) {
            let bytes = if format == vk::Format::D16_UNORM {
                2
            } else {
                4
            };
            (vk::ImageAspectFlags::DEPTH, bytes)
        } else {
            (vk::ImageAspectFlags::COLOR, texel_bytes)
        };
        self.transition(key, vk::ImageLayout::GENERAL)?;
        self.order_target(key)?;
        let size = u64::from(width) * u64::from(height) * u64::from(texel_bytes);
        let buffer = self.host_buffer(size, vk::BufferUsageFlags::TRANSFER_DST)?;
        let commands = self.commands()?;
        let copy = vk::BufferImageCopy {
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            ..Default::default()
        };
        // SAFETY: recording; the image is in GENERAL layout; the buffer fits.
        unsafe {
            self.gpu.device.cmd_copy_image_to_buffer(
                commands,
                image,
                vk::ImageLayout::GENERAL,
                buffer.buffer,
                &[copy],
            );
        }
        self.flush()?;
        // SAFETY: in bounds; the copy has completed.
        let bytes =
            unsafe { std::slice::from_raw_parts(buffer.mapped.cast::<u8>(), size as usize) }
                .to_vec();
        buffer.destroy(self.device());
        Ok(bytes)
    }

    /// `key`, or for a scaled target a copy of its level 0 at the guest's
    /// resolution (a staging target), for reading back the guest's texels.
    fn unscaled_copy(&mut self, key: u64) -> Result<u64, String> {
        let target = self.targets.get(&key).ok_or("no such target")?;
        if target.scale.is_one() {
            return Ok(key);
        }
        let (width, height, format) = (target.width, target.height, target.format);
        let (scaled_width, scaled_height) = target.extent(0);
        let staging_key = STAGING_KEYS + 1;
        self.target_with_levels(staging_key, width, height, format, 1)?;
        self.transition(key, vk::ImageLayout::GENERAL)?;
        self.order_target(key)?;
        self.transition(staging_key, vk::ImageLayout::GENERAL)?;
        let commands = self.commands()?;
        let layers = vk::ImageSubresourceLayers {
            aspect_mask: range(format).aspect_mask,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };
        let corner = |width: u32, height: u32| vk::Offset3D {
            x: width as i32,
            y: height as i32,
            z: 1,
        };
        let blit = vk::ImageBlit {
            src_subresource: layers,
            src_offsets: [vk::Offset3D::default(), corner(scaled_width, scaled_height)],
            dst_subresource: layers,
            dst_offsets: [vk::Offset3D::default(), corner(width, height)],
        };
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
        let (source, staging) = (self.targets[&key].image, self.targets[&staging_key].image);
        // SAFETY: recording; both images are live and in GENERAL layout.
        unsafe {
            let device = &self.gpu.device;
            device.cmd_blit_image(
                commands,
                source,
                vk::ImageLayout::GENERAL,
                staging,
                vk::ImageLayout::GENERAL,
                &[blit],
                vk::Filter::NEAREST,
            );
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
        Ok(staging_key)
    }

    /// A host-visible, coherent buffer.
    #[track_caller]
    pub(crate) fn host_buffer(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
    ) -> Result<HostBuffer, String> {
        self.host_buffer_with(size, usage, false)
    }

    /// `host_buffer`, with a device address when `addressable`.
    #[track_caller]
    pub(crate) fn host_buffer_with(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
        addressable: bool,
    ) -> Result<HostBuffer, String> {
        let usage = addressed_usage(usage, addressable);
        let info = vk::BufferCreateInfo::default()
            .size(size.max(4))
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let device = self.device();
        // SAFETY: valid create info; destroyed by HostBuffer::destroy.
        let buffer =
            unsafe { device.create_buffer(&info, None) }.map_err(vk_error("vkCreateBuffer"))?;
        // SAFETY: the buffer is live.
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        // Host-cached memory when there is some: the CPU reads results back
        // from these buffers, and reads from uncached (often device-local,
        // PCIe-mapped) memory are an order of magnitude slower.
        let coherent =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let memory = self
            .allocate_with(
                requirements,
                coherent | vk::MemoryPropertyFlags::HOST_CACHED,
                addressable,
            )
            .or_else(|_| self.allocate_with(requirements, coherent, addressable))?;
        // SAFETY: fresh buffer and memory; mapped for the buffer's lifetime.
        let mapped = unsafe {
            device
                .bind_buffer_memory(buffer, memory, 0)
                .map_err(vk_error("vkBindBufferMemory"))?;
            device
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .map_err(vk_error("vkMapMemory"))?
        };
        Ok(HostBuffer {
            buffer,
            memory,
            mapped: mapped.cast(),
            size: size.max(4),
        })
    }
}

/// A mapped host-visible buffer.
pub(crate) struct HostBuffer {
    pub buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
    pub size: u64,
}

// SAFETY: the mapping belongs to this buffer alone; whoever owns the buffer
// may use it from any thread.
unsafe impl Send for HostBuffer {}

impl HostBuffer {
    /// A placeholder owning nothing.
    fn empty() -> Self {
        Self {
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            mapped: std::ptr::null_mut(),
            size: 0,
        }
    }

    pub fn write(&self, offset: usize, bytes: &[u8]) {
        assert!(offset + bytes.len() <= self.size as usize);
        // SAFETY: in bounds of the mapping, which lives as long as self.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.mapped.add(offset), bytes.len())
        };
    }

    /// The first `length` bytes as they are now (debugging: work may still
    /// be writing them).
    pub(crate) fn read_bytes(&self, length: usize) -> Vec<u8> {
        let length = length.min(self.size as usize);
        // SAFETY: in bounds of the mapping.
        unsafe { std::slice::from_raw_parts(self.mapped, length) }.to_vec()
    }

    fn read_f32x4(&self, count: usize) -> Vec<[f32; 4]> {
        assert!(count * 16 <= self.size as usize);
        // SAFETY: in bounds; the GPU work writing it has completed.
        let floats = unsafe { std::slice::from_raw_parts(self.mapped.cast::<f32>(), count * 4) };
        floats.as_chunks::<4>().0.to_vec()
    }

    pub fn destroy(self, device: &ash::Device) {
        if self.buffer == vk::Buffer::null() {
            return;
        }
        // SAFETY: no pending work uses the buffer (callers flush first).
        unsafe {
            device.unmap_memory(self.memory);
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

fn range(format: vk::Format) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: aspect(format),
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

/// `rect` limited to `bounds`, both (x, y, width, height).
fn clip(rect: [i32; 4], bounds: [i32; 4]) -> [i32; 4] {
    let x0 = rect[0].max(bounds[0]);
    let y0 = rect[1].max(bounds[1]);
    let x1 = (rect[0] + rect[2]).min(bounds[0] + bounds[2]);
    let y1 = (rect[1] + rect[3]).min(bounds[1] + bounds[3]);
    [x0, y0, x1 - x0, y1 - y0]
}

impl Drop for Renderer {
    fn drop(&mut self) {
        let _ = self.flush();
        if let Some(occlusion) = self.occlusion.take() {
            occlusion.destroy(&self.gpu.device);
        }
        if let Some(timer) = self.timer.take() {
            timer.destroy(&self.gpu.device);
        }
        for (_, (buffer, memory, _)) in std::mem::take(&mut self.resident) {
            // SAFETY: the flush above finished every use of the buffer.
            unsafe {
                self.gpu.device.destroy_buffer(buffer, None);
                self.gpu.device.free_memory(memory, None);
            }
        }
        if let Some((buffer, _)) = self.page_table.take() {
            buffer.destroy(&self.gpu.device);
        }
        for (_, (buffer, memory)) in std::mem::take(&mut self.imported) {
            // SAFETY: the flush above finished every use of the buffer.
            unsafe {
                self.gpu.device.destroy_buffer(buffer, None);
                self.gpu.device.free_memory(memory, None);
            }
        }
        let shared = std::mem::take(&mut self.shared)
            .into_iter()
            .map(|(_, buffer, memory)| (buffer, memory))
            .chain(
                std::mem::take(&mut self.addressed)
                    .into_iter()
                    .map(|(_, _, buffer, memory, _)| (buffer, memory)),
            );
        let retired = std::mem::take(&mut self.retired_shared)
            .into_iter()
            .chain(std::mem::take(&mut self.spare.retired_shared));
        for (buffer, memory) in shared.chain(retired) {
            // SAFETY: the flush above finished every use of the buffer.
            unsafe {
                self.gpu.device.destroy_buffer(buffer, None);
                self.gpu.device.free_memory(memory, None);
            }
        }
        for (_, target) in std::mem::take(&mut self.targets) {
            self.destroy_target(target);
        }
        for (_, (_, target)) in std::mem::take(&mut self.uploads) {
            self.destroy_target(target);
        }
        for target in std::mem::take(&mut self.retired_uploads) {
            self.destroy_target(target);
        }
        for (_, (_, buffer, view)) in std::mem::take(&mut self.texel_buffers) {
            self.destroy_texel_buffer(buffer, view);
        }
        for (buffer, view) in std::mem::take(&mut self.retired_texel_buffers) {
            self.destroy_texel_buffer(buffer, view);
        }
        let ring = std::mem::replace(&mut self.ring, HostBuffer::empty());
        ring.destroy(&self.gpu.device);
        let ring = std::mem::replace(&mut self.spare.ring, HostBuffer::empty());
        ring.destroy(&self.gpu.device);
        self.destroy_texture_pool();
        for buffer in std::mem::take(&mut self.debug_buffers) {
            buffer.destroy(&self.gpu.device);
        }
        // SAFETY: all work has completed.
        unsafe {
            for (_, sampler) in std::mem::take(&mut self.samplers) {
                self.gpu.device.destroy_sampler(sampler, None);
            }
            self.gpu
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.gpu
                .device
                .destroy_descriptor_pool(self.spare.descriptor_pool, None);
            self.gpu.device.destroy_fence(self.fence, None);
            self.gpu.device.destroy_fence(self.spare.fence, None);
            self.gpu.device.destroy_command_pool(self.pool, None);
        }
    }
}

/// Whether `format` stores texels in compressed blocks (BC, ETC2, EAC,
/// ASTC), which 1D images cannot hold.
fn block_compressed(format: vk::Format) -> bool {
    let raw = format.as_raw();
    (131..=184).contains(&raw) || (1_000_066_000..=1_000_066_013).contains(&raw)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn renderer() -> Option<Renderer> {
        // No device (CI): the GPU tests have nothing to run on.
        Gpu::new().ok().map(|gpu| Renderer::new(gpu).unwrap())
    }

    #[test]
    fn the_page_table_and_resident_buffers_have_device_addresses() {
        let Some(mut renderer) = renderer() else {
            return;
        };
        if !renderer.gpu.buffer_device_address {
            return;
        }
        let (_, created) = renderer.resident_buffer(7, 4096).unwrap();
        assert!(created);
        let resident = renderer.resident_address(7).unwrap();
        assert_ne!(resident, 0);
        let table = renderer
            .set_page_table(0x100, &[resident, 0, resident + 4096])
            .unwrap();
        assert_ne!(table.address, 0);
        assert_eq!((table.first_page, table.page_count), (0x100, 3));
        // The same entries: the same table, not rewritten.
        let again = renderer
            .set_page_table(0x100, &[resident, 0, resident + 4096])
            .unwrap();
        assert_eq!(again, table);
    }

    #[test]
    fn clip_limits_to_bounds() {
        assert_eq!(clip([-5, 2, 20, 4], [0, 0, 10, 10]), [0, 2, 10, 4]);
        assert_eq!(clip([12, 0, 4, 4], [0, 0, 10, 10])[2], -2);
    }

    #[test]
    fn whole_and_partial_clears_read_back() {
        let Some(mut renderer) = renderer() else {
            return;
        };
        renderer
            .target(1, 8, 4, vk::Format::R8G8B8A8_UNORM)
            .unwrap();
        renderer
            .clear_color(
                1,
                None,
                vk::ClearColorValue {
                    float32: [1.0, 0.0, 0.0, 1.0],
                },
            )
            .unwrap();
        renderer
            .clear_color(
                1,
                Some([4, 0, 4, 4]),
                vk::ClearColorValue {
                    float32: [0.0, 0.0, 1.0, 1.0],
                },
            )
            .unwrap();
        let texels = renderer.read_target(1).unwrap();
        assert_eq!(texels.len(), 32);
        assert_eq!(texels[0], [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(texels[7], [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(texels[3 * 8 + 3], [1.0, 0.0, 0.0, 1.0]);
    }
}

/// The view type an image of `shape` is created with.
fn own_view_type(shape: crate::draw::ImageShape) -> vk::ImageViewType {
    use crate::draw::ImageShape;
    match shape {
        ImageShape::D1 { layers } if layers > 1 => vk::ImageViewType::TYPE_1D_ARRAY,
        ImageShape::D1 { .. } => vk::ImageViewType::TYPE_1D,
        ImageShape::D2 { layers } if layers > 1 => vk::ImageViewType::TYPE_2D_ARRAY,
        ImageShape::D2 { .. } => vk::ImageViewType::TYPE_2D,
        ImageShape::D3 { .. } => vk::ImageViewType::TYPE_3D,
        ImageShape::Cube { layers } if layers > 6 => vk::ImageViewType::CUBE_ARRAY,
        ImageShape::Cube { .. } => vk::ImageViewType::CUBE,
    }
}

/// The parameters a view was created with.
#[derive(Clone, Copy, Debug)]
struct ViewInfo {
    image: vk::Image,
    view_type: vk::ImageViewType,
    format: vk::Format,
    range: vk::ImageSubresourceRange,
}

impl ViewInfo {
    fn of(info: &vk::ImageViewCreateInfo) -> Self {
        Self {
            image: info.image,
            view_type: info.view_type,
            format: info.format,
            range: info.subresource_range,
        }
    }
}

/// The sRGB or UNORM counterpart of `format` (the pairs NVN texture views
/// reinterpret between), if it has one.
pub fn srgb_twin(format: vk::Format) -> Option<vk::Format> {
    use vk::Format as F;
    let pairs = [
        (F::R8G8B8A8_UNORM, F::R8G8B8A8_SRGB),
        (F::B8G8R8A8_UNORM, F::B8G8R8A8_SRGB),
        (F::BC1_RGBA_UNORM_BLOCK, F::BC1_RGBA_SRGB_BLOCK),
        (F::BC2_UNORM_BLOCK, F::BC2_SRGB_BLOCK),
        (F::BC3_UNORM_BLOCK, F::BC3_SRGB_BLOCK),
        (F::BC7_UNORM_BLOCK, F::BC7_SRGB_BLOCK),
    ];
    pairs.iter().find_map(|&(unorm, srgb)| {
        if format == unorm {
            Some(srgb)
        } else if format == srgb {
            Some(unorm)
        } else {
            None
        }
    })
}

/// The float format of the same texel size for a 32-bit integer format.
/// Translated shaders declare every texture with a float sampled type and
/// reinterpret the bits they read, as the console's texel fetch returns raw
/// 32-bit values; reading an integer format through a float sampler is
/// undefined in Vulkan (RADV converts the value, so light-list indices came
/// back as garbage). A float view of the same bits keeps them intact.
pub(crate) fn bit_preserving_float(format: vk::Format) -> vk::Format {
    use vk::Format as F;
    match format {
        F::R32_UINT | F::R32_SINT => F::R32_SFLOAT,
        F::R32G32_UINT | F::R32G32_SINT => F::R32G32_SFLOAT,
        F::R32G32B32_UINT | F::R32G32B32_SINT => F::R32G32B32_SFLOAT,
        F::R32G32B32A32_UINT | F::R32G32B32A32_SINT => F::R32G32B32A32_SFLOAT,
        other => other,
    }
}

/// Bytes per texel of the uncompressed formats buffer textures use (0 for
/// others).
pub(crate) fn format_bytes(format: vk::Format) -> u32 {
    use vk::Format as F;
    match format {
        F::R8_UNORM | F::R8_SNORM | F::R8_UINT | F::R8_SINT => 1,
        F::R8G8_UNORM | F::R8G8_SNORM | F::R8G8_UINT | F::R8G8_SINT => 2,
        F::R16_UNORM | F::R16_SNORM | F::R16_UINT | F::R16_SINT | F::R16_SFLOAT => 2,
        F::R8G8B8A8_UNORM
        | F::R8G8B8A8_SNORM
        | F::R8G8B8A8_UINT
        | F::R8G8B8A8_SINT
        | F::R8G8B8A8_SRGB
        | F::B8G8R8A8_UNORM
        | F::A2B10G10R10_UNORM_PACK32
        | F::B10G11R11_UFLOAT_PACK32 => 4,
        F::R16G16_UNORM | F::R16G16_SNORM | F::R16G16_UINT | F::R16G16_SINT | F::R16G16_SFLOAT => 4,
        F::R32_UINT | F::R32_SINT | F::R32_SFLOAT => 4,
        F::R16G16B16A16_UNORM
        | F::R16G16B16A16_SNORM
        | F::R16G16B16A16_UINT
        | F::R16G16B16A16_SINT
        | F::R16G16B16A16_SFLOAT => 8,
        F::R32G32_UINT | F::R32G32_SINT | F::R32G32_SFLOAT => 8,
        F::R32G32B32_UINT | F::R32G32B32_SINT | F::R32G32B32_SFLOAT => 12,
        F::R32G32B32A32_UINT | F::R32G32B32A32_SINT | F::R32G32B32A32_SFLOAT => 16,
        _ => 0,
    }
}

/// The rectangle `[x0, y0, x1, y1]` that shows a `source`-sized frame in a
/// `window`-sized surface at its own aspect ratio: as large as fits,
/// centred (letterboxed or pillarboxed).
/// Keys from here up are the renderer's own staging targets (never scaled).
const STAGING_KEYS: u64 = u64::MAX - 0x10;

/// How a presented frame fills the window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScalingMode {
    /// Largest size with the frame's aspect ratio, centred (black bars).
    Fit,
    /// The whole window, whatever its aspect ratio.
    Stretch,
    /// The largest whole multiple of the frame's size that fits, centred
    /// (sharp pixels); Fit when even one does not fit.
    Integer,
}

/// ECHORA_SCALING_MODE (fit, stretch or integer; default fit) and
/// ECHORA_SCALING_FILTER (linear or nearest; default linear).
fn output_options() -> &'static (ScalingMode, vk::Filter) {
    static OPTIONS: std::sync::OnceLock<(ScalingMode, vk::Filter)> = std::sync::OnceLock::new();
    OPTIONS.get_or_init(|| {
        let var = |name| std::env::var(name).unwrap_or_default().to_ascii_lowercase();
        let mode = match var("ECHORA_SCALING_MODE").as_str() {
            "stretch" => ScalingMode::Stretch,
            "integer" => ScalingMode::Integer,
            _ => ScalingMode::Fit,
        };
        let filter = match var("ECHORA_SCALING_FILTER").as_str() {
            "nearest" => vk::Filter::NEAREST,
            _ => vk::Filter::LINEAR,
        };
        (mode, filter)
    })
}

/// The window rect (x0, y0, x1, y1) a `source`-sized frame is shown in.
pub(crate) fn present_rect(mode: ScalingMode, source: (u32, u32), window: (u32, u32)) -> [i32; 4] {
    match mode {
        ScalingMode::Fit => fit_rect(source, window),
        ScalingMode::Stretch => [0, 0, window.0 as i32, window.1 as i32],
        ScalingMode::Integer => {
            let factor = (window.0 / source.0.max(1)).min(window.1 / source.1.max(1));
            if factor == 0 {
                return fit_rect(source, window);
            }
            let (width, height) = (source.0 * factor, source.1 * factor);
            let (x, y) = ((window.0 - width) / 2, (window.1 - height) / 2);
            [x as i32, y as i32, (x + width) as i32, (y + height) as i32]
        }
    }
}

pub(crate) fn fit_rect(source: (u32, u32), window: (u32, u32)) -> [i32; 4] {
    let (sw, sh) = (u64::from(source.0.max(1)), u64::from(source.1.max(1)));
    let (ww, wh) = (u64::from(window.0), u64::from(window.1));
    // Fit the width unless that makes the frame taller than the window.
    let (w, h) = if ww * sh <= wh * sw {
        (ww, (ww * sh + sw / 2) / sw)
    } else {
        ((wh * sw + sh / 2) / sh, wh)
    };
    let (x, y) = ((ww - w) / 2, (wh - h) / 2);
    [x as i32, y as i32, (x + w) as i32, (y + h) as i32]
}

#[cfg(test)]
mod fit_tests {
    use super::{ScalingMode, fit_rect, present_rect};

    #[test]
    fn scales_are_rational() {
        use super::Scale;
        let qhd = Scale::new(1440, 1080).unwrap();
        assert_eq!(qhd.terms(), (4, 3));
        assert_eq!((qhd.apply(1920), qhd.apply(1080)), (2560, 1440));
        // Edges scale, so rects that meet still meet.
        assert_eq!(qhd.rect([0, 0, 960, 540]), [0, 0, 1280, 720]);
        assert_eq!(qhd.rect([1, 1, 2, 2]), [1, 1, 3, 3]);
        assert_eq!(Scale::new(2160, 1080).unwrap().terms(), (2, 1));
        assert_eq!(Scale::new(720, 1080).unwrap().apply(1920), 1280);
        assert!(Scale::new(500, 1080).is_none());
        assert!(Scale::new(5000, 1080).is_none());
    }

    #[test]
    fn frames_keep_their_aspect_ratio() {
        // Same shape: the whole window.
        assert_eq!(fit_rect((1920, 1080), (2560, 1440)), [0, 0, 2560, 1440]);
        // Wider window: bars at the sides.
        assert_eq!(fit_rect((1920, 1080), (2560, 1080)), [320, 0, 2240, 1080]);
        // Taller window: bars above and below.
        assert_eq!(fit_rect((1920, 1080), (1280, 1024)), [0, 152, 1280, 872]);
        let integer = |source, window| present_rect(ScalingMode::Integer, source, window);
        assert_eq!(integer((1920, 1080), (3840, 2160)), [0, 0, 3840, 2160]);
        assert_eq!(integer((1920, 1080), (2560, 1440)), [320, 180, 2240, 1260]);
        assert_eq!(integer((1920, 1080), (1280, 720)), [0, 0, 1280, 720]);
        assert_eq!(
            present_rect(ScalingMode::Stretch, (1920, 1080), (3440, 1440)),
            [0, 0, 3440, 1440]
        );
        assert_eq!(fit_rect((1920, 1080), (0, 0)), [0, 0, 0, 0]);
    }
}

/// The unsigned word format with `format`'s texel size, which a storage
/// image of it can be viewed as (None: no such size, or compressed).
pub fn uint_alias(format: vk::Format) -> Option<vk::Format> {
    use vk::Format as F;
    let bytes = match format {
        F::R8_UNORM | F::R8_SNORM | F::R8_UINT | F::R8_SINT => 1,
        F::R16_SFLOAT | F::R16_UNORM | F::R16_SNORM | F::R16_UINT | F::R16_SINT => 2,
        F::R8G8_UNORM | F::R8G8_SNORM | F::R8G8_UINT | F::R8G8_SINT => 2,
        F::R32_SFLOAT | F::R32_UINT | F::R32_SINT => 4,
        F::R16G16_SFLOAT | F::R16G16_UNORM | F::R16G16_SNORM | F::R16G16_UINT => 4,
        F::R16G16_SINT => 4,
        F::R8G8B8A8_UNORM | F::R8G8B8A8_SNORM | F::R8G8B8A8_UINT | F::R8G8B8A8_SINT => 4,
        F::R8G8B8A8_SRGB | F::B8G8R8A8_UNORM | F::B8G8R8A8_SRGB => 4,
        F::A2B10G10R10_UNORM_PACK32 | F::A2B10G10R10_UINT_PACK32 => 4,
        F::B10G11R11_UFLOAT_PACK32 | F::E5B9G9R9_UFLOAT_PACK32 => 4,
        F::R32G32_SFLOAT | F::R32G32_UINT | F::R32G32_SINT => 8,
        F::R16G16B16A16_SFLOAT | F::R16G16B16A16_UNORM | F::R16G16B16A16_SNORM => 8,
        F::R16G16B16A16_UINT | F::R16G16B16A16_SINT => 8,
        F::R32G32B32A32_SFLOAT | F::R32G32B32A32_UINT | F::R32G32B32A32_SINT => 16,
        _ => return None,
    };
    Some(match bytes {
        1 => F::R8_UINT,
        2 => F::R16_UINT,
        4 => F::R32_UINT,
        8 => F::R32G32_UINT,
        _ => F::R32G32B32A32_UINT,
    })
}

/// The format a storage image descriptor declares (`Descriptor::flags` bits
/// 8-11: the recompiler's ImageFormat), None when typeless.
pub(crate) fn declared_image_format(flags: u32) -> Option<vk::Format> {
    use vk::Format as F;
    Some(match flags >> 8 & 0xf {
        1 => F::R8_UINT,
        2 => F::R8_SINT,
        3 => F::R16_UINT,
        4 => F::R16_SINT,
        5 => F::R32_UINT,
        6 => F::R32G32_UINT,
        7 => F::R32G32B32A32_UINT,
        _ => return None,
    })
}

/// Whether texels of `format` are integers (read through int or uint
/// images), as SpecMeaning::IntegerFormat asks.
pub fn integer_format(format: vk::Format) -> bool {
    use vk::Format as F;
    matches!(
        format,
        F::R8_UINT
            | F::R8_SINT
            | F::R8G8_UINT
            | F::R8G8_SINT
            | F::R8G8B8A8_UINT
            | F::R8G8B8A8_SINT
            | F::R16_UINT
            | F::R16_SINT
            | F::R16G16_UINT
            | F::R16G16_SINT
            | F::R16G16B16A16_UINT
            | F::R16G16B16A16_SINT
            | F::R32_UINT
            | F::R32_SINT
            | F::R32G32_UINT
            | F::R32G32_SINT
            | F::R32G32B32_UINT
            | F::R32G32B32_SINT
            | F::R32G32B32A32_UINT
            | F::R32G32B32A32_SINT
            | F::A2B10G10R10_UINT_PACK32
            | F::S8_UINT
    )
}

/// `usage`, plus device addresses when the device supports them.
fn addressed_usage(usage: vk::BufferUsageFlags, addressable: bool) -> vk::BufferUsageFlags {
    if addressable {
        usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
    } else {
        usage
    }
}
