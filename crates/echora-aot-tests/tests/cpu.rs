//! Integer, flag, call/return, PAuth-hint, budget and SIMD stack programs,
//! executed natively.
#![cfg(any(target_os = "linux", windows))]
#![allow(clippy::unwrap_used, unsafe_code)]

use echora_aot_tests::*;

#[test]
fn memory_and_flags() {
    let mut m = Machine::new();
    assert!(matches!(
        memory_and_flags::run(&mut m),
        Err(Trap::Exit { code: 0 })
    ));
    assert_eq!(m.memory.read_u32(0x8000).unwrap(), 42);
    assert_eq!(m.core.cpu.regs[2], 42);
    assert_eq!(m.core.cpu.nzcv, 6);
}

#[test]
fn calls_return_with_and_without_pauth_hints() {
    for run in [call_return::run, pauth_disabled_call_return::run] {
        let mut m = Machine::new();
        assert!(matches!(run(&mut m), Err(Trap::Exit { code: 0 })));
        assert_eq!(m.core.cpu.regs[3], 9);
        assert_eq!(m.core.cpu.regs[30], 0x1004);
    }
}

#[test]
fn budget_stops_an_infinite_loop() {
    let mut m = Machine::new();
    m.core.max_steps = 5;
    assert!(matches!(
        bounded_loop::run(&mut m),
        Err(Trap::Budget { steps: 5, .. })
    ));
}

#[test]
fn long_multiply_and_conditional_compare() {
    let mut m = Machine::new();
    m.core.cpu.nzcv = 4;
    m.core.cpu.regs[1] = 0xffff_ffff;
    m.core.cpu.regs[2] = 2;
    m.core.cpu.regs[3] = 1;
    m.core.cpu.regs[4] = 0x1_ffff_ffff;
    m.core.cpu.regs[5] = 0;
    assert!(matches!(
        long_multiply_and_conditional_compare::run(&mut m),
        Err(Trap::Exit { code: 0 })
    ));
    assert_eq!(m.core.cpu.regs[0], 0x1ffffffff);
    assert_eq!(m.core.cpu.nzcv, 0xa);
}

#[test]
fn vector_pair_stack_round_trip() {
    let mut m = Machine::new();
    m.core.cpu.sp = 0x9000;
    m.core.cpu.vectors[0] = 0x0011_2233_4455_6677_8899_aabb_ccdd_eeff;
    m.core.cpu.vectors[1] = 0xfedc_ba98_7654_3210_0123_4567_89ab_cdef;
    assert!(matches!(
        vector_stack_roundtrip::run(&mut m),
        Err(Trap::Exit { code: 0 })
    ));
    assert_eq!(m.core.cpu.vectors[2], m.core.cpu.vectors[0]);
    assert_eq!(m.core.cpu.vectors[3], m.core.cpu.vectors[1]);
    assert_eq!(m.core.cpu.sp, 0x9000);
}

#[test]
fn xpaclri_strips_the_link_register_only() {
    let mut m = Machine::new();
    m.core.cpu.regs[30] = 0xab00007100001234;
    m.core.cpu.nzcv = 9;
    let result = strip_lr_pac::run(&mut m);
    assert!(matches!(result, Err(Trap::Exit { code: 0 })), "{result:?}");
    assert_eq!(m.core.cpu.regs[30], 0x7100001234);
    assert_eq!(m.core.cpu.nzcv, 9);
}

#[test]
fn unsupported_system_register_stops_before_effects() {
    let mut m = Machine::new();
    m.core.cpu.regs[2] = 0xfeed;
    m.core.cpu.nzcv = 7;
    assert!(matches!(
        unsupported_virtual_counter::run(&mut m),
        Err(Trap::UnsupportedInstruction {
            pc: 0x1000,
            word: 0xd53be042
        })
    ));
    assert_eq!(m.core.cpu.regs[2], 0xfeed);
    assert_eq!(m.core.cpu.nzcv, 7);
    assert_eq!(m.core.steps, 1);
}

#[test]
fn undiscovered_indirect_target_reports_missing_block() {
    let mut m = Machine::new();
    m.core.cpu.regs[1] = 0x1800;
    assert!(matches!(
        indirect_missing_block::run(&mut m),
        Err(Trap::MissingBlock { pc: 0x1800 })
    ));
}

#[test]
fn calls_across_dispatch_windows() {
    let mut m = Machine::new();
    assert!(matches!(
        sharded_call_return::run(&mut m),
        Err(Trap::Exit { code: 0 })
    ));
    assert_eq!(m.core.cpu.regs[3], 17);
    assert_eq!(m.core.cpu.regs[30], 0x1004);
    assert_eq!(m.core.steps, 4);
    let mut m = Machine::new();
    m.core.cpu.pc = 0x1100;
    assert!(matches!(
        sharded_hole::run(&mut m),
        Err(Trap::MissingBlock { pc: 0x1100 })
    ));
    assert_eq!(m.core.steps, 0);
}

/// Guest loads and stores go straight through the guest view, so an access
/// the guest may not make faults in the host. The fault handler names the
/// guest address and the process ends; it is not a recoverable trap. The
/// faulting run happens in a child process.
#[test]
fn unmapped_store_ends_the_process_naming_the_guest_address() {
    if std::env::var_os("ECHORA_FAULT_CHILD").is_some() {
        // No core dump for this intentional crash.
        #[cfg(unix)]
        {
            let none = libc::rlimit {
                rlim_cur: 0,
                rlim_max: 0,
            };
            // SAFETY: setrlimit with a valid struct; affects only this process.
            assert_eq!(unsafe { libc::setrlimit(libc::RLIMIT_CORE, &none) }, 0);
        }
        // No Windows Error Reporting dialog for this intentional crash.
        #[cfg(windows)]
        // SAFETY: changes only this process's error mode.
        unsafe {
            use windows_sys::Win32::System::Diagnostics::Debug::{
                SEM_NOGPFAULTERRORBOX, SetErrorMode,
            };
            SetErrorMode(SEM_NOGPFAULTERRORBOX);
        }
        let mut m = Machine::new();
        m.memory.unmap_exact(0x8000, 0x2000).unwrap();
        let _ = memory_and_flags::run(&mut m);
        unreachable!("the store to an unmapped page must fault");
    }
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "unmapped_store_ends_the_process_naming_the_guest_address",
            "--nocapture",
        ])
        .env("ECHORA_FAULT_CHILD", "1")
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("guest memory fault at 0x0000000000008000"),
        "{stderr}"
    );
}

#[test]
fn crc32_crc32c_and_mrs_nzcv() {
    let mut m = Machine::new();
    let regs = &mut m.core.cpu.regs;
    regs[0] = 0xffff_ffff;
    regs[1] = 0xdead_beef_0000_0031; // only the low byte ('1') is used
    regs[5] = 0xffff_ffff_ffff_ffff; // only W5 is the accumulator
    regs[6] = 0x3231; // "12"
    regs[2] = 0xffff_ffff;
    regs[3] = 0x3837_3635_3433_3231; // "12345678"
    m.core.cpu.nzcv = 0b1010;
    assert!(matches!(
        crc_and_flags::run(&mut m),
        Err(Trap::Exit { code: 0 })
    ));
    // Expected values: zlib's CRC32 and a CRC32C checked against its
    // standard "123456789" value, both without the final inversion.
    assert_eq!(m.core.cpu.regs[0], 0x7c23_1048);
    assert_eq!(m.core.cpu.regs[5], 0xb0ac_bb32);
    assert_eq!(m.core.cpu.regs[2], 0x9f78_7f65);
    assert_eq!(m.core.cpu.regs[4], 0xa000_0000);
    assert_eq!(m.core.cpu.nzcv, 0b1010);
}

fn lanes(values: [u32; 4]) -> u128 {
    values
        .iter()
        .rev()
        .fold(0u128, |acc, &v| acc << 32 | u128::from(v))
}

#[test]
fn fp_across_lane_reductions() {
    // [1.0, -2.0, quiet NaN, 3.0]: FMAXV/FMINV propagate the NaN, the
    // NM forms ignore it.
    let mut m = Machine::new();
    m.core.cpu.vectors[1] = lanes([0x3f80_0000, 0xc000_0000, 0x7fc0_0001, 0x4040_0000]);
    for v in [0, 2, 3, 4] {
        m.core.cpu.vectors[v] = u128::MAX;
    }
    assert!(matches!(
        fp_across::run(&mut m),
        Err(Trap::Exit { code: 0 })
    ));
    let v = m.core.cpu.vectors;
    assert_eq!(v[0], 0x7fc0_0001);
    assert_eq!(v[3], 0x7fc0_0001);
    assert_eq!(v[4], 0x4040_0000);
    assert_eq!(v[2], 0xc000_0000);

    // [1.5, -0.0, 7.25, 2.0]: the minimum is negative zero.
    let mut m = Machine::new();
    m.core.cpu.vectors[1] = lanes([0x3fc0_0000, 0x8000_0000, 0x40e8_0000, 0x4000_0000]);
    assert!(matches!(
        fp_across::run(&mut m),
        Err(Trap::Exit { code: 0 })
    ));
    let v = m.core.cpu.vectors;
    assert_eq!(
        (v[0], v[3], v[4], v[2]),
        (0x40e8_0000, 0x8000_0000, 0x40e8_0000, 0x8000_0000)
    );
}

#[test]
fn absolute_difference_long_and_dot_products() {
    // Inputs and expected values were computed independently in Python,
    // running the four instructions in program order (UDOT reads the V0
    // that SABAL just wrote).
    let mut m = Machine::new();
    let v = &mut m.core.cpu.vectors;
    v[0] = 0xffffffff000000030000000200000001;
    v[2] = 0x80007ffffffb0064;
    v[3] = 0x20001fff900077fff8000000aff9c;
    v[4] = 0x280000001e000000140000000a;
    v[6] = 0x4030201c80aff00;
    v[7] = 0x1020304641400ff;
    v[8] = 0xbeef0000deadfffffffb00000005;
    v[9] = 0xfa05fc0302ff7f80;
    v[10] = 0x7f9070702ff8080;
    assert!(matches!(
        abd_long_and_dot::run(&mut m),
        Err(Trap::Exit { code: 0 })
    ));
    let v = m.core.cpu.vectors;
    assert_eq!(v[0], 0xfffe0001000200000011000000c9);
    assert_eq!(v[5], 0x30001000100030064000a00ff00ff);
    assert_eq!(v[4], 0x126000001250000001400007a86);
    // The 2S form clears the upper half of the destination.
    assert_eq!(v[8], 0xffffffa70000008a);
}

#[test]
fn msr_nzcv_takes_bits_31_to_28() {
    let mut m = Machine::new();
    m.core.cpu.regs[6] = 0xffff_ffff_5fff_ffff; // N=0 Z=1 C=0 V=1; other bits ignored
    assert!(matches!(
        nzcv_round_trip::run(&mut m),
        Err(Trap::Exit { code: 0 })
    ));
    assert_eq!(m.core.cpu.nzcv, 0b0101);
    assert_eq!(m.core.cpu.regs[7], 0x5000_0000);
}
