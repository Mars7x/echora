//! Presentation to a host window: a Vulkan swapchain on a Wayland, Xlib or
//! Win32 surface, filled by blitting the presented render target (scaled to the
//! window) and presented synchronously.

use std::ffi::c_void;

use ash::vk;

use crate::renderer::Renderer;
use crate::vulkan::Gpu;

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
    images: Vec<vk::Image>,
    extent: vk::Extent2D,
    format: vk::Format,
    fence: vk::Fence,
    /// The swapchain no longer matches the surface (resize).
    stale: bool,
    /// Window size the swapchain was built for.
    requested: (u32, u32),
    /// The swapchain stores sRGB-encoded texels: presented images with an
    /// sRGB format keep their encoded bytes (the console scans out the
    /// display texture's bytes; a UNORM swapchain showed them decoded).
    srgb: bool,
}

fn vk_error(context: &'static str) -> impl Fn(vk::Result) -> String {
    move |error| format!("{context}: {error}")
}

impl Presenter {
    pub fn new(gpu: &Gpu, window: WindowHandle) -> Result<Self, String> {
        if !gpu.swapchain {
            return Err("the device has no swapchain support".into());
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
        }
        .map_err(vk_error("vkGetPhysicalDeviceSurfaceSupportKHR"))?;
        if !supported {
            // SAFETY: the surface was just created and is unused.
            unsafe { surface_loader.destroy_surface(surface, None) };
            return Err("the queue cannot present to this window".into());
        }
        // SAFETY: valid create info.
        let fence = unsafe {
            gpu.device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        }
        .map_err(vk_error("vkCreateFence"))?;
        Ok(Self {
            surface_loader,
            swapchain_loader: ash::khr::swapchain::Device::new(&gpu.instance, &gpu.device),
            surface,
            swapchain: vk::SwapchainKHR::null(),
            images: Vec::new(),
            extent: vk::Extent2D::default(),
            format: vk::Format::UNDEFINED,
            fence,
            stale: true,
            requested: (0, 0),
            srgb: false,
        })
    }

    /// (Re)create the swapchain for the surface's current size. `fallback`
    /// is used when the surface leaves the size to the swapchain (Wayland).
    fn rebuild(&mut self, gpu: &Gpu, fallback: vk::Extent2D) -> Result<(), String> {
        // SAFETY: live physical device and surface.
        let (capabilities, formats) = unsafe {
            (
                self.surface_loader
                    .get_physical_device_surface_capabilities(gpu.physical, self.surface)
                    .map_err(vk_error("vkGetPhysicalDeviceSurfaceCapabilitiesKHR"))?,
                self.surface_loader
                    .get_physical_device_surface_formats(gpu.physical, self.surface)
                    .map_err(vk_error("vkGetPhysicalDeviceSurfaceFormatsKHR"))?,
            )
        };
        let wanted = if self.srgb {
            vk::Format::B8G8R8A8_SRGB
        } else {
            vk::Format::B8G8R8A8_UNORM
        };
        let format = formats
            .iter()
            .find(|format| format.format == wanted)
            .or_else(|| {
                formats
                    .iter()
                    .find(|format| format.format == vk::Format::B8G8R8A8_UNORM)
            })
            .or_else(|| formats.first())
            .copied()
            .ok_or("the surface has no formats")?;
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
            return Err("the window is minimized".into());
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
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true)
            .old_swapchain(self.swapchain);
        // SAFETY: the device is idle with respect to the old swapchain (every
        // present waits for its blit), which is retired here.
        unsafe {
            let swapchain = self
                .swapchain_loader
                .create_swapchain(&info, None)
                .map_err(vk_error("vkCreateSwapchainKHR"))?;
            if self.swapchain != vk::SwapchainKHR::null() {
                self.swapchain_loader
                    .destroy_swapchain(self.swapchain, None);
            }
            self.swapchain = swapchain;
            self.images = self
                .swapchain_loader
                .get_swapchain_images(swapchain)
                .map_err(vk_error("vkGetSwapchainImagesKHR"))?;
        }
        self.extent = extent;
        self.format = format.format;
        self.stale = false;
        Ok(())
    }

    pub fn destroy(self, gpu: &Gpu) {
        // SAFETY: nothing uses the swapchain after the device is idle.
        unsafe {
            let _ = gpu.device.device_wait_idle();
            if self.swapchain != vk::SwapchainKHR::null() {
                self.swapchain_loader
                    .destroy_swapchain(self.swapchain, None);
            }
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
        let (width, height, format) = {
            let target = self.existing_target(key).ok_or("no such target")?;
            let (width, height) = target.extent(0);
            (width, height, target.format)
        };
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
        if presenter.srgb != srgb {
            presenter.srgb = srgb;
            presenter.stale = true;
        }
        if presenter.stale || presenter.requested != size {
            self.flush()?;
            presenter.rebuild(&self.gpu, fallback)?;
            presenter.requested = size;
        }
        // SAFETY: live swapchain; the fence is unsignaled.
        let acquired = unsafe {
            presenter.swapchain_loader.acquire_next_image(
                presenter.swapchain,
                u64::MAX,
                vk::Semaphore::null(),
                presenter.fence,
            )
        };
        let index = match acquired {
            Ok((index, suboptimal)) => {
                presenter.stale |= suboptimal;
                index
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                presenter.stale = true;
                return Ok(());
            }
            Err(error) => return Err(format!("vkAcquireNextImageKHR: {error}")),
        };
        // SAFETY: the fence was submitted by the acquire above.
        unsafe {
            let device = &self.gpu.device;
            device
                .wait_for_fences(&[presenter.fence], true, u64::MAX)
                .map_err(vk_error("vkWaitForFences"))?;
            device
                .reset_fences(&[presenter.fence])
                .map_err(vk_error("vkResetFences"))?;
        }
        let image = presenter.images[index as usize];
        self.blit_to_swapchain(key, image, (width, height), presenter.extent)?;
        self.flush()?;
        let swapchains = [presenter.swapchain];
        let indices = [index];
        let info = vk::PresentInfoKHR::default()
            .swapchains(&swapchains)
            .image_indices(&indices);
        // SAFETY: the blit finished (flush waits) and left the image in
        // PRESENT_SRC_KHR.
        match unsafe {
            presenter
                .swapchain_loader
                .queue_present(self.gpu.queue, &info)
        } {
            Ok(suboptimal) => presenter.stale |= suboptimal,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => presenter.stale = true,
            Err(error) => return Err(format!("vkQueuePresentKHR: {error}")),
        }
        Ok(())
    }
}
