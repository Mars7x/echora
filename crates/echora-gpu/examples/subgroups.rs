#![allow(unsafe_code)]
//! Print each Vulkan device's subgroup sizes (default, and the range
//! VK_EXT_subgroup_size_control can require), for checking that guest
//! 32-lane warp code fits the host.
use ash::vk;

fn main() {
    // SAFETY: plain instance creation and property queries.
    unsafe {
        let entry = ash::Entry::load().expect("load Vulkan");
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let instance = entry
            .create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app),
                None,
            )
            .expect("create instance");
        for physical in instance
            .enumerate_physical_devices()
            .expect("enumerate devices")
        {
            let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
            let mut control = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
            let mut properties = vk::PhysicalDeviceProperties2::default()
                .push_next(&mut subgroup)
                .push_next(&mut control);
            instance.get_physical_device_properties2(physical, &mut properties);
            let name = std::ffi::CStr::from_ptr(properties.properties.device_name.as_ptr());
            println!(
                "{}: subgroup {} (stages {:?}, ops {:?}); size control {}..{} for {:?}",
                name.to_string_lossy(),
                subgroup.subgroup_size,
                subgroup.supported_stages,
                subgroup.supported_operations,
                control.min_subgroup_size,
                control.max_subgroup_size,
                control.required_subgroup_size_stages
            );
        }
        instance.destroy_instance(None);
    }
}
