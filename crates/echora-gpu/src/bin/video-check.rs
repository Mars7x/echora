//! Reports the device's AV1 decode support (Vulkan Video).

use ash::vk;
use echora_gpu::video;

fn main() {
    let device = match video::VideoDevice::new() {
        Ok(device) => device,
        Err(error) => {
            println!("{error}");
            return;
        }
    };
    println!(
        "device: {}; decode queue family {}, copy queue family {}",
        device.name, device.decode_family, device.transfer_family
    );
    match video::av1_capabilities(&device) {
        Ok(capabilities) => println!("{capabilities:#?}"),
        Err(error) => println!("{error}"),
    }
    for (name, usage) in [
        ("dpb", vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR),
        ("output", vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR),
        (
            "dpb+output",
            vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR | vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR,
        ),
    ] {
        match video::picture_formats(&device, usage) {
            Ok(formats) => println!("{name}: {formats:?}"),
            Err(error) => println!("{name}: {error}"),
        }
    }
    match video::Av1Decoder::new(device.clone(), 1920, 1080) {
        Ok(_) => println!("1920x1080 session: created"),
        Err(error) => println!("1920x1080 session: {error}"),
    }
}
