//! Presentation to a host window: a Vulkan swapchain on a Wayland, Xlib or
//! Win32 surface, filled by blitting the presented render target (scaled to the
//! window).

use std::ffi::c_void;
use std::time::{Duration, Instant};

use ash::vk;

use crate::renderer::Renderer;
use crate::vulkan::Gpu;

/// Allow several refresh intervals for an image without relying on WSI
/// forward progress while the surface is occluded.
const ACQUIRE_TIMEOUT_NS: u64 = 50_000_000;

/// The native window to present to.
#[derive(Clone, Copy, Debug)]
pub enum WindowHandle {
    Wayland {
        display: *mut c_void,
        surface: *mut c_void,
    },
    Xlib {
        display: *mut c_void,
        /// An Xlib `Window` (C `unsigned long`).
        window: vk::Window,
    },
    Win32 {
        hinstance: vk::HINSTANCE,
        hwnd: vk::HWND,
    },
}

// SAFETY: the handles are only passed to Vulkan, which accepts them from any
// thread; the window outlives the presenter (its owner keeps it open).
unsafe impl Send for WindowHandle {}

pub struct Presenter {
    surface_loader: ash::khr::surface::Instance,
    swapchain_loader: ash::khr::swapchain::Device,
    surface: vk::SurfaceKHR,
    swapchain: vk::SwapchainKHR,
    images: Vec<PresentImage>,
    extent: vk::Extent2D,
    format: vk::Format,
    fence: vk::Fence,
    acquisition: Option<Acquisition>,
    /// The swapchain no longer matches the surface (resize).
    stale: bool,
    /// Window size the swapchain was built for.
    requested: (u32, u32),
    /// The swapchain stores sRGB-encoded texels: presented images with an
    /// sRGB format keep their encoded bytes (the console scans out the
    /// display texture's bytes; a UNORM swapchain showed them decoded).
    srgb: bool,
    /// Whether this presenter can retire the synchronization objects used
    /// by its queued window presents.
    queued: bool,
    preferred_modes: &'static [vk::PresentModeKHR],
}

struct PresentImage {
    image: vk::Image,
    ready: Option<PresentSync>,
}

struct Acquisition {
    index: u32,
    ready: bool,
}

impl Acquisition {
    fn wait(
        &mut self,
        device: &ash::Device,
        fence: vk::Fence,
        timeout: u64,
    ) -> Result<bool, String> {
        if !self.ready {
            self.ready = wait_and_reset(device, fence, timeout, "waiting for the acquired image")?;
        }
        Ok(self.ready)
    }
}

struct PresentSync {
    blitted: vk::Semaphore,
    presented: vk::Fence,
    pending: bool,
}

impl PresentSync {
    fn new(device: &ash::Device) -> Result<Self, String> {
        // SAFETY: valid creation parameters and a live device.
        unsafe {
            let blitted = device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                .map_err(vk_error("vkCreateSemaphore"))?;
            let presented = match device.create_fence(&vk::FenceCreateInfo::default(), None) {
                Ok(fence) => fence,
                Err(error) => {
                    device.destroy_semaphore(blitted, None);
                    return Err(vk_error("vkCreateFence")(error));
                }
            };
            Ok(Self {
                blitted,
                presented,
                pending: false,
            })
        }
    }

    fn wait(&mut self, device: &ash::Device, timeout: u64) -> Result<bool, String> {
        if self.pending
            && wait_and_reset(device, self.presented, timeout, "waiting for presentation")?
        {
            self.pending = false;
        }
        Ok(!self.pending)
    }

    fn destroy(self, device: &ash::Device) {
        // SAFETY: both the blit and all presents using it have finished.
        unsafe {
            device.destroy_fence(self.presented, None);
            device.destroy_semaphore(self.blitted, None);
        }
    }
}

/// The caller retains ownership of the pending fence on timeout.
fn wait_and_reset(
    device: &ash::Device,
    fence: vk::Fence,
    timeout: u64,
    context: &'static str,
) -> Result<bool, String> {
    // SAFETY: the caller owns the fence and holds exclusive host access.
    unsafe {
        match device.wait_for_fences(&[fence], true, timeout) {
            Ok(()) => device
                .reset_fences(&[fence])
                .map(|()| true)
                .map_err(vk_error("vkResetFences")),
            Err(vk::Result::TIMEOUT) => Ok(false),
            Err(error) => Err(vk_error(context)(error)),
        }
    }
}

fn remaining_budget(deadline: Instant) -> u64 {
    deadline
        .saturating_duration_since(Instant::now())
        .as_nanos() as u64
}

fn vk_error(context: &'static str) -> impl Fn(vk::Result) -> String {
    move |error| format!("{context}: {error}")
}

fn select_present_mode(
    preferred: &[vk::PresentModeKHR],
    supported: &[vk::PresentModeKHR],
) -> vk::PresentModeKHR {
    preferred
        .iter()
        .copied()
        .find(|mode| supported.contains(mode))
        .unwrap_or(vk::PresentModeKHR::FIFO)
}

fn select_surface_format(
    formats: &[vk::SurfaceFormatKHR],
    srgb: bool,
) -> Option<vk::SurfaceFormatKHR> {
    use vk::Format;

    formats
        .iter()
        .min_by_key(|format| {
            let other_color_space = format.color_space != vk::ColorSpaceKHR::SRGB_NONLINEAR;
            let rank = match (srgb, format.format) {
                (true, Format::B8G8R8A8_SRGB) | (false, Format::B8G8R8A8_UNORM) => 0,
                (true, Format::R8G8B8A8_SRGB) | (false, Format::R8G8B8A8_UNORM) => 1,
                (_, Format::B8G8R8A8_UNORM | Format::B8G8R8A8_SRGB) => 2,
                (_, Format::R8G8B8A8_UNORM | Format::R8G8B8A8_SRGB) => 3,
                _ => 4,
            };
            (other_color_space, rank)
        })
        .copied()
}

impl Presenter {
    pub fn new(
        gpu: &Gpu,
        window: WindowHandle,
        preferred_modes: &'static [vk::PresentModeKHR],
    ) -> Result<Self, String> {
        if !gpu.swapchain {
            return Err("the device has no swapchain support".into());
        }
        let (system, create_surface) = match window {
            WindowHandle::Wayland { .. } => ("Wayland", c"vkCreateWaylandSurfaceKHR"),
            WindowHandle::Xlib { .. } => ("X11", c"vkCreateXlibSurfaceKHR"),
            WindowHandle::Win32 { .. } => ("Win32", c"vkCreateWin32SurfaceKHR"),
        };
        // SAFETY: the instance is live and the command name is NUL terminated.
        let available = unsafe {
            gpu.entry
                .get_instance_proc_addr(gpu.instance.handle(), create_surface.as_ptr())
        };
        if available.is_none() {
            return Err(format!(
                "the Vulkan instance cannot create {system} surfaces"
            ));
        }
        // SAFETY: the handles name a live window (see WindowHandle).
        let surface = unsafe {
            match window {
                WindowHandle::Wayland { display, surface } => {
                    let info = vk::WaylandSurfaceCreateInfoKHR::default()
                        .display(display)
                        .surface(surface);
                    ash::khr::wayland_surface::Instance::new(&gpu.entry, &gpu.instance)
                        .create_wayland_surface(&info, None)
                        .map_err(vk_error("vkCreateWaylandSurfaceKHR"))?
                }
                WindowHandle::Xlib { display, window } => {
                    let info = vk::XlibSurfaceCreateInfoKHR::default()
                        .dpy(display.cast())
                        .window(window);
                    ash::khr::xlib_surface::Instance::new(&gpu.entry, &gpu.instance)
                        .create_xlib_surface(&info, None)
                        .map_err(vk_error("vkCreateXlibSurfaceKHR"))?
                }
                WindowHandle::Win32 { hinstance, hwnd } => {
                    let info = vk::Win32SurfaceCreateInfoKHR::default()
                        .hinstance(hinstance)
                        .hwnd(hwnd);
                    ash::khr::win32_surface::Instance::new(&gpu.entry, &gpu.instance)
                        .create_win32_surface(&info, None)
                        .map_err(vk_error("vkCreateWin32SurfaceKHR"))?
                }
            }
        };
        let surface_loader = ash::khr::surface::Instance::new(&gpu.entry, &gpu.instance);
        // SAFETY: live physical device, surface and queue family.
        let supported = unsafe {
            surface_loader.get_physical_device_surface_support(
                gpu.physical,
                gpu.queue_family,
                surface,
            )
        };
        if !matches!(supported, Ok(true)) {
            // SAFETY: the surface was just created and is unused.
            unsafe { surface_loader.destroy_surface(surface, None) };
            return Err(supported.err().map_or_else(
                || "the queue cannot present to this window".into(),
                vk_error("vkGetPhysicalDeviceSurfaceSupportKHR"),
            ));
        }
        // SAFETY: valid create info.
        let fence = unsafe {
            gpu.device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        };
        let fence = match fence {
            Ok(fence) => fence,
            Err(error) => {
                // SAFETY: the new surface has no swapchain.
                unsafe { surface_loader.destroy_surface(surface, None) };
                return Err(vk_error("vkCreateFence")(error));
            }
        };
        Ok(Self {
            surface_loader,
            swapchain_loader: ash::khr::swapchain::Device::new(&gpu.instance, &gpu.device),
            surface,
            swapchain: vk::SwapchainKHR::null(),
            images: Vec::new(),
            extent: vk::Extent2D::default(),
            format: vk::Format::UNDEFINED,
            fence,
            acquisition: None,
            stale: true,
            requested: (0, 0),
            srgb: false,
            queued: gpu.swapchain_maintenance
                && std::env::var_os("ECHORA_PRESENT_CPU_WAIT").is_none(),
            preferred_modes,
        })
    }

    /// (Re)create the swapchain for the surface's current size. `fallback`
    /// is used when the surface leaves the size to the swapchain (Wayland).
    fn rebuild(
        &mut self,
        gpu: &Gpu,
        fallback: vk::Extent2D,
        deadline: Instant,
    ) -> Result<bool, String> {
        // SAFETY: live physical device and surface.
        let (capabilities, formats, modes) = unsafe {
            (
                self.surface_loader
                    .get_physical_device_surface_capabilities(gpu.physical, self.surface)
                    .map_err(vk_error("vkGetPhysicalDeviceSurfaceCapabilitiesKHR"))?,
                self.surface_loader
                    .get_physical_device_surface_formats(gpu.physical, self.surface)
                    .map_err(vk_error("vkGetPhysicalDeviceSurfaceFormatsKHR"))?,
                self.surface_loader
                    .get_physical_device_surface_present_modes(gpu.physical, self.surface)
                    .map_err(vk_error("vkGetPhysicalDeviceSurfacePresentModesKHR"))?,
            )
        };
        let format =
            select_surface_format(&formats, self.srgb).ok_or("the surface has no formats")?;
        let extent = if capabilities.current_extent.width != u32::MAX {
            capabilities.current_extent
        } else {
            vk::Extent2D {
                width: fallback.width.clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: fallback.height.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            }
        };
        if extent.width == 0 || extent.height == 0 {
            return Ok(false);
        }
        if !self.wait_for_presents(gpu, Some(deadline))? {
            return Ok(false);
        }
        let mut count = capabilities.min_image_count + 1;
        if capabilities.max_image_count != 0 {
            count = count.min(capabilities.max_image_count);
        }
        let info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface)
            .min_image_count(count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(select_present_mode(self.preferred_modes, &modes))
            .clipped(true)
            .old_swapchain(self.swapchain);
        // SAFETY: rendering and pending present operations on the old
        // swapchain have finished or handed their resources to WSI.
        let swapchain = unsafe { self.swapchain_loader.create_swapchain(&info, None) };
        // An old swapchain is retired even when creation fails.
        self.destroy_swapchain(gpu);
        self.swapchain = swapchain.map_err(vk_error("vkCreateSwapchainKHR"))?;
        // SAFETY: the new swapchain is live.
        let images = unsafe { self.swapchain_loader.get_swapchain_images(self.swapchain) }
            .map_err(vk_error("vkGetSwapchainImagesKHR"))?;
        for image in images {
            let ready = self
                .queued
                .then(|| PresentSync::new(&gpu.device))
                .transpose()?;
            self.images.push(PresentImage { image, ready });
        }
        self.extent = extent;
        self.format = format.format;
        self.stale = false;
        Ok(true)
    }

    fn wait_for_presents(&mut self, gpu: &Gpu, deadline: Option<Instant>) -> Result<bool, String> {
        for image in &mut self.images {
            if let Some(ready) = &mut image.ready
                && !ready.wait(&gpu.device, deadline.map_or(u64::MAX, remaining_budget))?
            {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn destroy_swapchain(&mut self, gpu: &Gpu) {
        for image in self.images.drain(..) {
            if let Some(ready) = image.ready {
                ready.destroy(&gpu.device);
            }
        }
        if self.swapchain != vk::SwapchainKHR::null() {
            // SAFETY: there are no outstanding rendering or present
            // references. The swapchain's images belong to it.
            unsafe {
                self.swapchain_loader
                    .destroy_swapchain(self.swapchain, None)
            };
            self.swapchain = vk::SwapchainKHR::null();
        }
    }

    pub fn destroy(mut self, gpu: &Gpu) {
        // SAFETY: the renderer has drained the submission thread before
        // destruction and will submit no later work.
        unsafe {
            let _ = gpu.device.device_wait_idle();
        }
        if self.acquisition.as_ref().is_some_and(|image| !image.ready) {
            // Queue idle cannot complete an acquire the presentation engine
            // still owns. Wait before destroying its fence or swapchain.
            loop {
                // SAFETY: this presenter owns the fence and no other host uses it.
                match unsafe { gpu.device.wait_for_fences(&[self.fence], true, u64::MAX) } {
                    Ok(()) | Err(vk::Result::ERROR_DEVICE_LOST) => break,
                    Err(_) => std::thread::sleep(Duration::from_millis(10)),
                }
            }
        }
        let _ = self.wait_for_presents(gpu, None);
        self.destroy_swapchain(gpu);
        // SAFETY: no swapchain or pending acquire references the surface/fence.
        unsafe {
            self.surface_loader.destroy_surface(self.surface, None);
            gpu.device.destroy_fence(self.fence, None);
        }
    }
}

impl Renderer {
    /// Show render target `key` in the window, scaled to fill it. `size` is
    /// the window's size in pixels, used when the surface does not fix one.
    pub fn present(
        &mut self,
        presenter: &mut Presenter,
        key: u64,
        size: (u32, u32),
    ) -> Result<(), String> {
        if size.0 == 0 || size.1 == 0 {
            return Ok(());
        }
        let format = self.existing_target(key).ok_or("no such target")?.format;
        if crate::renderer::is_depth(format) {
            return Err("cannot present a depth target".into());
        }
        let fallback = vk::Extent2D {
            width: size.0,
            height: size.1,
        };
        let srgb = matches!(
            format,
            vk::Format::R8G8B8A8_SRGB | vk::Format::B8G8R8A8_SRGB
        );
        let deadline = Instant::now() + Duration::from_nanos(ACQUIRE_TIMEOUT_NS);
        let changed = presenter.srgb != srgb || presenter.requested != size;
        let retry = presenter.acquisition.is_some();
        if !retry {
            if presenter.stale || changed {
                self.stat(
                    if presenter.srgb != srgb && presenter.swapchain != vk::SwapchainKHR::null() {
                        "stat: swapchain rebuilds for a presented encoding change"
                    } else if presenter.requested != size {
                        "stat: swapchain rebuilds for a window size change"
                    } else {
                        "stat: swapchain rebuilds for a stale swapchain"
                    },
                    1,
                );
                presenter.srgb = srgb;
                presenter.stale = true;
                self.flush()?;
                if !presenter.rebuild(&self.gpu, fallback, deadline)? {
                    return Ok(());
                }
                presenter.requested = size;
            }
            // SAFETY: live swapchain and an unsignaled fence with no pending acquire.
            let acquired = unsafe {
                presenter.swapchain_loader.acquire_next_image(
                    presenter.swapchain,
                    remaining_budget(deadline),
                    vk::Semaphore::null(),
                    presenter.fence,
                )
            };
            match acquired {
                Ok((index, suboptimal)) => {
                    presenter.stale |= suboptimal;
                    presenter.acquisition = Some(Acquisition {
                        index,
                        ready: false,
                    });
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    presenter.stale = true;
                    return Ok(());
                }
                Err(vk::Result::TIMEOUT | vk::Result::NOT_READY) => {
                    self.stat("stat: presents skipped waiting for a window image", 1);
                    return Ok(());
                }
                Err(error) => return Err(format!("vkAcquireNextImageKHR: {error}")),
            }
        }
        let acquired = presenter.acquisition.as_mut().unwrap();
        if !acquired.wait(
            &self.gpu.device,
            presenter.fence,
            remaining_budget(deadline),
        )? {
            self.stat("stat: presents skipped waiting for acquire fence", 1);
            return Ok(());
        }
        if retry && changed {
            // The acquire has finished and the stale swapchain can be retired.
            presenter.acquisition = None;
            presenter.stale = true;
            return Ok(());
        }
        let index = acquired.index;
        if let Some(ready) = &mut presenter.images[index as usize].ready
            && !ready.wait(&self.gpu.device, remaining_budget(deadline))?
        {
            self.stat("stat: presents skipped waiting for presentation fence", 1);
            return Ok(());
        }
        presenter.acquisition = None;
        let shown = self.present_acquired(presenter, key, index);
        if shown.is_err() {
            self.abandon_present(presenter);
        }
        shown
    }

    /// Blit target `key` into acquired swapchain image `index` and queue its
    /// present.
    fn present_acquired(
        &mut self,
        presenter: &mut Presenter,
        key: u64,
        index: u32,
    ) -> Result<(), String> {
        let image = &mut presenter.images[index as usize];
        self.blit_to_swapchain(key, image.image, presenter.format, presenter.extent)?;
        let signal = if let Some(ready) = &image.ready {
            self.submit_for_present(ready.blitted)?;
            Some(ready.blitted)
        } else {
            self.flush()?;
            None
        };
        let swapchains = [presenter.swapchain];
        let indices = [index];
        let mut info = vk::PresentInfoKHR::default()
            .swapchains(&swapchains)
            .image_indices(&indices)
            .wait_semaphores(signal.as_slice());
        let fence = image.ready.as_ref().map(|ready| ready.presented);
        let mut completion = vk::SwapchainPresentFenceInfoEXT::default().fences(fence.as_slice());
        if fence.is_some() {
            info = info.push_next(&mut completion);
        }
        let held = self
            .queue_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // SAFETY: the blit either completed or signals the wait semaphore.
        // It transitions the image to PRESENT_SRC_KHR before WSI reads it.
        let presented = unsafe {
            presenter
                .swapchain_loader
                .queue_present(self.gpu.queue, &info)
        };
        drop(held);
        if let Some(ready) = &mut image.ready {
            ready.pending = present_was_queued(presented);
        }
        match presented {
            Ok(suboptimal) => {
                self.stat("stat: window presents queued", 1);
                presenter.stale |= suboptimal;
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.stat("stat: window presents out of date", 1);
                presenter.stale = true;
            }
            Err(error) => return Err(format!("vkQueuePresentKHR: {error}")),
        }
        Ok(())
    }

    /// After an error that followed a successful acquire: the image stays
    /// acquired and its semaphore may be signaled with no waiter. Rebuilding
    /// the swapchain releases the image and replaces each image's
    /// semaphore, which is only safe once the queue has signaled it.
    fn abandon_present(&mut self, presenter: &mut Presenter) {
        presenter.stale = true;
        let _ = self.flush();
        let _held = self
            .queue_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // SAFETY: the queue is externally synchronized by the held lock.
        unsafe {
            let _ = self.gpu.device.queue_wait_idle(self.gpu.queue);
        }
    }
}

/// Whether a present operation was queued (and so waits on its semaphore
/// and later signals its fence) given `vkQueuePresentKHR`'s result. WSI may
/// reject a surface without cancelling the queue wait.
fn present_was_queued(result: Result<bool, vk::Result>) -> bool {
    matches!(
        result,
        Ok(_)
            | Err(vk::Result::ERROR_OUT_OF_DATE_KHR
                | vk::Result::ERROR_SURFACE_LOST_KHR
                | vk::Result::ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use vk::PresentModeKHR as Mode;

    #[test]
    fn acquisition_timeout_keeps_the_image_until_its_fence_signals() {
        let Ok(gpu) = Gpu::new() else { return };
        let device = &gpu.device;
        // SAFETY: the test owns the device, queue, and fence. The submission
        // signals the fence without referencing any other resources.
        unsafe {
            let fence = device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .unwrap();
            let mut acquisition = Acquisition {
                index: 2,
                ready: false,
            };
            assert!(!acquisition.wait(device, fence, 0).unwrap());
            assert_eq!(acquisition.index, 2);
            device
                .queue_submit(gpu.queue, &[vk::SubmitInfo::default()], fence)
                .unwrap();
            assert!(acquisition.wait(device, fence, 1_000_000_000).unwrap());
            assert!(!device.get_fence_status(fence).unwrap());
            assert!(acquisition.wait(device, fence, 0).unwrap());
            device.destroy_fence(fence, None);
        }
    }

    #[test]
    fn rejected_surfaces_still_queue_the_present() {
        assert!(present_was_queued(Ok(false)));
        assert!(present_was_queued(Ok(true)));
        assert!(present_was_queued(Err(vk::Result::ERROR_OUT_OF_DATE_KHR)));
        assert!(present_was_queued(Err(vk::Result::ERROR_SURFACE_LOST_KHR)));
        // Other errors leave the semaphore signaled: the present is
        // abandoned and the queue drained before the swapchain is rebuilt.
        assert!(!present_was_queued(Err(vk::Result::ERROR_DEVICE_LOST)));
        assert!(!present_was_queued(Err(
            vk::Result::ERROR_OUT_OF_HOST_MEMORY
        )));
    }

    #[test]
    fn presentation_uses_the_first_supported_preference() {
        let preferred = [Mode::IMMEDIATE, Mode::MAILBOX];
        assert_eq!(
            select_present_mode(&preferred, &[Mode::FIFO, Mode::MAILBOX, Mode::IMMEDIATE]),
            Mode::IMMEDIATE
        );
        assert_eq!(
            select_present_mode(&preferred, &[Mode::FIFO, Mode::MAILBOX]),
            Mode::MAILBOX
        );
        assert_eq!(select_present_mode(&preferred, &[Mode::FIFO]), Mode::FIFO);
    }

    #[test]
    fn presentation_preserves_encoding_with_either_channel_order() {
        let nonlinear = vk::ColorSpaceKHR::SRGB_NONLINEAR;
        let surface_format = |format, color_space| vk::SurfaceFormatKHR {
            format,
            color_space,
        };
        let bgra = surface_format(vk::Format::B8G8R8A8_UNORM, nonlinear);
        let rgba_srgb = surface_format(vk::Format::R8G8B8A8_SRGB, nonlinear);
        let bgra_hdr = surface_format(
            vk::Format::B8G8R8A8_SRGB,
            vk::ColorSpaceKHR::HDR10_ST2084_EXT,
        );
        let supported = [bgra_hdr, bgra, rgba_srgb];
        assert_eq!(select_surface_format(&supported, true), Some(rgba_srgb));
        assert_eq!(select_surface_format(&supported, false), Some(bgra));
        assert_eq!(select_surface_format(&[rgba_srgb], false), Some(rgba_srgb));
    }

    #[test]
    fn scanout_preserves_sdr_bytes_on_oppositely_encoded_surfaces() {
        use crate::renderer::ScalingMode;
        use vk::Format as F;

        let Ok(gpu) = Gpu::new() else { return };
        let mut renderer = Renderer::new(gpu).unwrap();
        for (source, same_order, swapped, srgb) in [
            (F::R8G8B8A8_SRGB, F::R8G8B8A8_UNORM, F::B8G8R8A8_UNORM, true),
            (F::R8G8B8A8_UNORM, F::R8G8B8A8_SRGB, F::B8G8R8A8_SRGB, false),
            (F::B8G8R8A8_SRGB, F::B8G8R8A8_UNORM, F::R8G8B8A8_UNORM, true),
            (F::B8G8R8A8_UNORM, F::B8G8R8A8_SRGB, F::R8G8B8A8_SRGB, false),
        ] {
            renderer.target(1, 2, 1, source).unwrap();
            let green = if srgb { 128 } else { 55 };
            for (x, rgba) in [[0.0, 0.21586, 1.0, 1.0], [1.0, 0.21586, 0.0, 0.5]]
                .into_iter()
                .enumerate()
            {
                renderer
                    .clear_color(
                        1,
                        Some([x as i32, 0, 1, 1]),
                        vk::ClearColorValue { float32: rgba },
                    )
                    .unwrap();
            }

            for format in [same_order, swapped, source] {
                for (width, height, filter) in [
                    (2, 1, vk::Filter::NEAREST),
                    (5, 5, vk::Filter::NEAREST),
                    (5, 5, vk::Filter::LINEAR),
                ] {
                    let extent = vk::Extent2D { width, height };
                    let image = renderer.target(2, width, height, format).unwrap().image;
                    // Mirror the final layout in this offscreen target's tracker.
                    renderer
                        .transition(2, vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .unwrap();
                    renderer
                        .write_present_image(1, image, format, extent, (ScalingMode::Fit, filter))
                        .unwrap();
                    let bytes = renderer.read_target_bytes(2, 4).unwrap();
                    let pixel = |x, y| {
                        let at = ((y * width + x) * 4) as usize;
                        let mut rgba: [u8; 4] = bytes[at..at + 4].try_into().unwrap();
                        if matches!(format, F::B8G8R8A8_SRGB | F::B8G8R8A8_UNORM) {
                            rgba.swap(0, 2);
                        }
                        rgba
                    };
                    let y = height / 2;
                    assert_eq!(pixel(0, y), [0, green, 255, 255]);
                    assert_eq!(pixel(width - 1, y), [255, green, 0, 128]);
                    if height == 5 {
                        for x in 0..width {
                            assert_eq!(pixel(x, 0), [0, 0, 0, 255]);
                            assert_eq!(pixel(x, 4), [0, 0, 0, 255]);
                        }
                        if filter == vk::Filter::NEAREST {
                            assert_eq!(pixel(1, y), pixel(0, y));
                            assert_eq!(pixel(3, y), pixel(width - 1, y));
                        } else {
                            let half = if srgb { 188 } else { 128 };
                            let mid = pixel(2, y);
                            for (actual, expected) in mid.into_iter().zip([half, green, half, 191])
                            {
                                assert!(
                                    actual.abs_diff(expected) <= 1,
                                    "{source:?} -> {format:?}: {mid:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
