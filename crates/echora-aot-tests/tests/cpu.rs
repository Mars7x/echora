//! Integer, flag, call/return, PAuth-hint, budget and SIMD stack programs,
//! executed natively. Expected values follow pokopia-recomp
//! `tests/aot_execution.rs`.
#![cfg(target_os = "linux")]
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
        let none = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        // SAFETY: setrlimit with a valid struct; affects only this process.
        assert_eq!(unsafe { libc::setrlimit(libc::RLIMIT_CORE, &none) }, 0);
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
