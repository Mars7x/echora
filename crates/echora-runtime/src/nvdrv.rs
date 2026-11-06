//! NVIDIA driver device ioctls behind `nvdrv:gsv` (INvDrv2User). Each ioctl
//! takes the request's input bytes and the sizes of its output buffers and
//! returns the bytes to write; the service layer owns descriptors and IPC.
//! Ioctls are added as the application reaches them; anything else is an
//! explicit error naming the device and code.

/// Bytes written to the main and inline output buffers of an ioctl.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct IoctlOutput {
    pub output: Vec<u8>,
    pub inline_output: Vec<u8>,
    /// NvResult returned in the reply's Err word (0 is Success).
    pub result: u32,
}

/// NvResult::NotImplemented.
const NOT_IMPLEMENTED: u32 = 1;

/// NVGPU_GPU_IOCTL_GET_CHARACTERISTICS with the 0xD8-byte native structure
/// (`0xC0E84705`: group 'G', number 5, in/out, 0xE8 bytes).
const GET_CHARACTERISTICS_NATIVE: u32 = 0xc0e8_4705;
/// NVGPU_GPU_IOCTL_GET_TPC_MASKS (in/out, 0x18 bytes).
const GET_TPC_MASKS: u32 = 0xc018_4706;
/// NVGPU_GPU_IOCTL_ZCULL_GET_CTX_SIZE (out, 4 bytes).
const ZCULL_GET_CTX_SIZE: u32 = 0x8004_4701;
/// NVGPU_GPU_IOCTL_ZCULL_GET_INFO (out, 0x28 bytes).
const ZCULL_GET_INFO: u32 = 0x8028_4702;
/// NVGPU_GPU_IOCTL_FLUSH_L2 (in, 8 bytes).
const FLUSH_L2: u32 = 0x4008_4707;
/// NVGPU_GPU_IOCTL_GET_ACTIVE_SLOT_MASK (out, 8 bytes).
const GET_ACTIVE_SLOT_MASK: u32 = 0x8008_4714;
/// Group 'G' number 0x13, in/out, 8 bytes.
const GPU_IOCTL_13: u32 = 0xc008_4713;
/// TPC mask of the single GPC: bits for its two TPCs.
const TPC_MASK: u32 = 0x3;
/// Native GPU characteristics size for that code. Eden 1a6d7d4 handles the
/// 0xD0-byte form (`0xC0E04705`); this SDK's is eight bytes longer.
const NATIVE_CHARACTERISTICS_SIZE: usize = 0xd8;

/// Eden 1a6d7d4 (`nvhost_ctrl_gpu.cpp` GetCharacteristics and
/// `native_gpu_characteristics.h` MakeNativeGpuCharacteristics): the legacy
/// Maxwell values with the native Ampere overrides. Bytes Eden leaves out
/// (0x82.., including the eight this SDK adds) stay zero. Provisional.
fn native_gpu_characteristics() -> Vec<u8> {
    let mut bytes = vec![0u8; NATIVE_CHARACTERISTICS_SIZE];
    let mut put32 = |at: usize, value: u32| bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
    put32(0x00, 0x170); // arch
    put32(0x04, 0xe); // impl
    put32(0x08, 0xa1); // rev
    put32(0x0c, 0x1); // num_gpc
    put32(0x20, 0x2); // num_tpc_per_gpc
    put32(0x24, 0x20); // bus_type: AXI
    put32(0x28, 0x1000); // preferred page size
    put32(0x2c, 0x20000); // compression_page_size
    put32(0x30, 0x1b); // pde_coverage_bit_count
    put32(0x34, 0x10000 | 0x1000); // available page sizes
    put32(0x38, 0x1); // gpc_mask
    put32(0x3c, 0x807); // SM version
    put32(0x40, 0x807); // SPA version
    put32(0x44, 0x80); // sm_arch_warp_count
    put32(0x48, 0x28); // gpu_va_bit_count
    put32(0x58, 0x902d); // twod_class
    put32(0x5c, 0xb197); // threed_class
    put32(0x60, 0xb1c0); // compute_class
    put32(0x64, 0xb06f); // gpfifo_class
    put32(0x68, 0xa140); // inline_to_memory_class
    put32(0x6c, 0xb0b5); // dma_copy_class
    let flags: u64 = 0x55 | 1 << 8 | 1 << 26 | 1 << 34 | 0xf << 40;
    bytes[0x10..0x18].copy_from_slice(&0x40000u64.to_le_bytes()); // l2_cache_size
    bytes[0x50..0x58].copy_from_slice(&flags.to_le_bytes());
    // Interface ioctl numbers: GetGpuTime, TSG bind, no debugger, channel
    // SetTimeslice, address-space Remap, then four unimplemented.
    for (index, value) in [0x1ci16, 2, -1, 0x1d, 0x14, 0, 0, 0, 0]
        .into_iter()
        .enumerate()
    {
        let at = 0x70 + index * 2;
        bytes[at..at + 2].copy_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Ioctl (inline size 0) and Ioctl3: one input buffer, a main and an
/// optional inline output buffer.
pub fn ioctl(
    device: &str,
    code: u32,
    input: &[u8],
    output_size: usize,
    inline_size: usize,
) -> Result<IoctlOutput, String> {
    match (device, code) {
        // Output: u64 native size, the input's second word, then the native
        // structure; the inline output gets the structure alone (Eden's
        // native path).
        ("/dev/nvhost-ctrl-gpu", GET_CHARACTERISTICS_NATIVE) => {
            let packet = 0x10 + NATIVE_CHARACTERISTICS_SIZE;
            if input.len() < packet
                || output_size < packet
                || inline_size < NATIVE_CHARACTERISTICS_SIZE
            {
                return Err(format!(
                    "GetCharacteristics buffers {:#x}/{output_size:#x}/{inline_size:#x} are too small",
                    input.len()
                ));
            }
            let native = native_gpu_characteristics();
            let mut output = (NATIVE_CHARACTERISTICS_SIZE as u64).to_le_bytes().to_vec();
            output.extend_from_slice(&input[8..0x10]);
            output.extend_from_slice(&native);
            Ok(IoctlOutput {
                output,
                inline_output: native,
                result: 0,
            })
        }
        // GetTPCMasks (Eden 1a6d7d4 GetTPCMasks3): u32 mask_buffer_size,
        // padding, u64 mask_buffer_address, u32 tpc_mask, padding. A nonzero
        // buffer size gets mask 3 (two TPCs in the one GPC), also written as
        // the inline output's first u32.
        ("/dev/nvhost-ctrl-gpu", GET_TPC_MASKS) => {
            if input.len() < 0x18 || output_size < 0x18 {
                return Err(format!(
                    "GetTPCMasks buffers {:#x}/{output_size:#x} are too small",
                    input.len()
                ));
            }
            let mut output = input[..0x18].to_vec();
            let requested = u32::from_le_bytes([input[0], input[1], input[2], input[3]]);
            let mut inline_output = Vec::new();
            if requested != 0 {
                output[0x10..0x14].copy_from_slice(&TPC_MASK.to_le_bytes());
                if inline_size >= 4 {
                    inline_output = TPC_MASK.to_le_bytes().to_vec();
                }
            }
            Ok(IoctlOutput {
                output,
                inline_output,
                result: 0,
            })
        }
        // ZCULL_GET_CTX_SIZE (out, u32): Eden 1a6d7d4 answers 1.
        ("/dev/nvhost-ctrl-gpu", ZCULL_GET_CTX_SIZE) if output_size >= 4 => Ok(IoctlOutput {
            output: 1u32.to_le_bytes().to_vec(),
            ..IoctlOutput::default()
        }),
        // ZCULL_GET_INFO (out, ten u32s): Eden 1a6d7d4's values.
        ("/dev/nvhost-ctrl-gpu", ZCULL_GET_INFO) if output_size >= 0x28 => Ok(IoctlOutput {
            output: [
                0x20u32, 0x20, 0x400, 0x800, 0x20, 0x20, 0xc0, 0x20, 0x40, 0x10,
            ]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
            ..IoctlOutput::default()
        }),
        // GET_ACTIVE_SLOT_MASK (out, u32 slot, u32 mask): slot 7, mask 1.
        ("/dev/nvhost-ctrl-gpu", GET_ACTIVE_SLOT_MASK) if output_size >= 8 => Ok(IoctlOutput {
            output: [7u32, 1]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
            ..IoctlOutput::default()
        }),
        // FLUSH_L2 (in, u32 flags, u32 reserved): nothing is cached on the
        // host side, so there is nothing to flush.
        ("/dev/nvhost-ctrl-gpu", FLUSH_L2) if input.len() >= 8 => Ok(IoctlOutput::default()),
        // Eden 1a6d7d4 stubs Ioctl3 0x13 on this device as NotImplemented
        // and writes no output; the SDK continues past that result.
        ("/dev/nvhost-ctrl-gpu", GPU_IOCTL_13) => Ok(IoctlOutput {
            result: NOT_IMPLEMENTED,
            ..IoctlOutput::default()
        }),
        _ => Err(format!(
            "ioctl {code:#010x} on {device} is not implemented (input {:#x}, outputs {output_size:#x}/{inline_size:#x})",
            input.len()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_characteristics_report_ampere_with_the_input_address_echoed() {
        let mut input = vec![0u8; 0xe8];
        input[..8].copy_from_slice(&0xe8u64.to_le_bytes());
        input[8..16].copy_from_slice(&0x1234_5678u64.to_le_bytes());
        let out = ioctl("/dev/nvhost-ctrl-gpu", 0xc0e8_4705, &input, 0xe8, 0xd8).unwrap();
        assert_eq!(out.output.len(), 0xe8);
        assert_eq!(out.output[..8], 0xd8u64.to_le_bytes());
        assert_eq!(out.output[8..16], 0x1234_5678u64.to_le_bytes());
        assert_eq!(out.output[0x10..], out.inline_output[..]);
        let word =
            |at: usize| u32::from_le_bytes(out.inline_output[at..at + 4].try_into().unwrap());
        assert_eq!((word(0), word(4), word(0x3c)), (0x170, 0xe, 0x807));
        assert_eq!(word(0x5c), 0xb197);
        assert!(ioctl("/dev/nvhost-ctrl-gpu", 0xc0e8_4705, &input, 0xe8, 0xd0).is_err());
        assert!(ioctl("/dev/nvmap", 0xc0e8_4705, &input, 0xe8, 0xd8).is_err());
    }

    #[test]
    fn tpc_masks_are_written_only_for_a_nonzero_buffer_size() {
        let mut input = vec![0u8; 0x18];
        input[8..16].copy_from_slice(&0xdead_0000u64.to_le_bytes());
        let empty = ioctl("/dev/nvhost-ctrl-gpu", 0xc018_4706, &input, 0x18, 4).unwrap();
        assert_eq!(empty.output, input);
        assert!(empty.inline_output.is_empty());
        input[..4].copy_from_slice(&4u32.to_le_bytes());
        let out = ioctl("/dev/nvhost-ctrl-gpu", 0xc018_4706, &input, 0x18, 4).unwrap();
        assert_eq!(out.output[0x10..0x14], 3u32.to_le_bytes());
        assert_eq!(out.output[8..16], input[8..16]);
        assert_eq!(out.inline_output, 3u32.to_le_bytes());
        assert_eq!(out.result, 0);
        let stub = ioctl("/dev/nvhost-ctrl-gpu", 0xc008_4713, &[0; 8], 8, 8).unwrap();
        assert_eq!((stub.result, stub.output.len()), (1, 0));
        let size = ioctl("/dev/nvhost-ctrl-gpu", 0x8004_4701, &[], 4, 0).unwrap();
        assert_eq!((size.output, size.result), (1u32.to_le_bytes().to_vec(), 0));
        let info = ioctl("/dev/nvhost-ctrl-gpu", 0x8028_4702, &[], 0x28, 0).unwrap();
        assert_eq!((info.output.len(), info.output[8]), (0x28, 0x00));
        assert_eq!(info.output[4 * 9], 0x10);
        let slots = ioctl("/dev/nvhost-ctrl-gpu", 0x8008_4714, &[], 8, 0).unwrap();
        assert_eq!(slots.output, [7, 0, 0, 0, 1, 0, 0, 0]);
        assert!(ioctl("/dev/nvhost-ctrl-gpu", 0x8028_4702, &[], 0x20, 0).is_err());
    }
}
