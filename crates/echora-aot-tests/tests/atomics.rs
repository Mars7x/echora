//! Load/store-exclusive, compare-and-swap, atomic add, load-acquire and
//! store-release semantics, executed natively. The body is transcribed from
//! pokopia-recomp `tests/aot_execution.rs`.
#![cfg(target_os = "linux")]
#![allow(
    clippy::unwrap_used,
    clippy::type_complexity,
    clippy::single_element_loop,
    unused_variables,
    unused_mut
)]

use echora_aot_tests::*;

fn machine() -> Machine {
    Machine::new()
}

#[test]
fn atomics_and_ordered_accesses() {
    let cas_cases: &[(fn(&mut Machine) -> Result<(), Trap>, u8, usize, usize)] = &[
        (cas_byte::run, 1, 1, 2),
        (cas_halfword::run, 2, 1, 2),
        (cas_word::run, 4, 8, 20),
        (cas_doubleword::run, 8, 1, 2),
    ];
    for &(run, bytes, rs, rt) in cas_cases {
        let mask = u64::MAX >> (64 - bytes * 8);
        let initial = 0xfedcba9876543210;
        let desired = 0x123456789abcdef0;
        for success in [true, false] {
            let mut m = machine();
            m.core.cpu.regs[0] = 0x8000;
            m.core.cpu.nzcv = 0xb;
            m.core.cpu.regs[rs] =
                ((initial ^ u64::from(!success)) & mask) | (0xabcdefff12345678 & !mask);
            m.core.cpu.regs[rt] = desired;
            m.memory.write_u64(0x8000, initial).unwrap();
            assert!(matches!(run(&mut m), Err(Trap::Exit { code: 0 })));
            assert_eq!(m.core.cpu.regs[rs], initial & mask);
            assert_eq!(m.core.cpu.regs[rt], desired);
            assert_eq!(m.core.cpu.nzcv, 0xb);
            assert_eq!(m.core.steps, 2);
            assert_eq!(
                m.memory.read_u64(0x8000).unwrap(),
                if success {
                    (initial & !mask) | (desired & mask)
                } else {
                    initial
                }
            );
        }
    }
    for success in [true, false] {
        let mut m = machine();
        m.core.cpu.regs[0] = 0x8000;
        m.core.cpu.regs[1] = 17;
        let initial = if success { 17 } else { 18 };
        m.memory.write_u64(0x8000, initial).unwrap();
        assert!(matches!(
            cas_alias_compare_store::run(&mut m),
            Err(Trap::Exit { code: 0 })
        ));
        assert_eq!(m.core.cpu.regs[1], initial);
        assert_eq!(m.memory.read_u64(0x8000).unwrap(), initial);
    }
    for success in [true, false] {
        let mut m = machine();
        m.core.cpu.regs[0] = 0x8000;
        m.core.cpu.regs[1] = 0x1234;
        let initial = if success { 0x8000 } else { 0xdeadbeef };
        m.memory.write_u64(0x8000, initial).unwrap();
        assert!(matches!(
            cas_alias_compare_base::run(&mut m),
            Err(Trap::Exit { code: 0 })
        ));
        assert_eq!(m.core.cpu.regs[0], initial);
        assert_eq!(
            m.memory.read_u64(0x8000).unwrap(),
            if success { 0x1234 } else { initial }
        );
    }
    let mut m = machine();
    m.core.cpu.sp = 0x8000;
    m.core.cpu.regs[2] = 0xdeadbeef12345678;
    m.core.cpu.nzcv = 5;
    assert!(matches!(
        cas_zero_compare_sp::run(&mut m),
        Err(Trap::Exit { code: 0 })
    ));
    assert_eq!(m.memory.read_u32(0x8000).unwrap(), 0x12345678);
    assert_eq!(m.core.cpu.sp, 0x8000);
    assert_eq!(m.core.cpu.nzcv, 5);
    let mut m = machine();
    m.core.cpu.sp = 0x8000;
    m.core.cpu.regs[1] = 0xabcdef1200001234;
    m.memory.write_u32(0x8000, 0x1234).unwrap();
    assert!(matches!(
        cas_zero_store_sp::run(&mut m),
        Err(Trap::Exit { code: 0 })
    ));
    assert_eq!(m.memory.read_u32(0x8000).unwrap(), 0);
    assert_eq!(m.core.cpu.regs[1], 0x1234);
    assert_eq!(m.core.cpu.sp, 0x8000);
    for address in [0x8002] {
        let mut m = machine();
        m.core.cpu.regs[0] = address;
        m.core.cpu.regs[8] = 0xabcdef1200001234;
        m.core.cpu.regs[20] = 0x5678;
        m.core.cpu.nzcv = 0xd;
        m.memory.write_u64(0x8000, u64::MAX).unwrap();
        let before = m.core.cpu.clone();
        assert!(matches!(
            cas_word::run(&mut m),
            Err(Trap::MemoryFault { .. })
        ));
        assert_eq!(m.core.cpu, before);
        assert_eq!(m.core.steps, 1);
        assert_eq!(m.memory.read_u64(0x8000).unwrap(), u64::MAX);
    }
    let mut m = machine();
    m.core.cpu.sp = 0x8004;
    m.core.cpu.regs[1] = 0x1234;
    m.core.cpu.nzcv = 7;
    m.memory.write_u32(0x8004, 0x1234).unwrap();
    let before = m.core.cpu.clone();
    assert!(matches!(
        cas_zero_store_sp::run(&mut m),
        Err(Trap::InvalidState { .. })
    ));
    assert_eq!(m.core.cpu, before);
    assert_eq!(m.memory.read_u32(0x8004).unwrap(), 0x1234);
    let ldadd_cases: &[(fn(&mut Machine) -> Result<(), Trap>, u8, usize, usize)] = &[
        (ldadd_byte::run, 1, 1, 2),
        (ldadd_halfword::run, 2, 1, 2),
        (ldadd_word_alias::run, 4, 8, 8),
        (ldadd_doubleword::run, 8, 1, 2),
    ];
    for &(run, bytes, rs, rt) in ldadd_cases {
        let mask = u64::MAX >> (64 - bytes * 8);
        let initial = (0x1122334455667788 & !mask) | mask;
        let addend = (0xfedcba9876543200 & !mask) | 1;
        let mut m = machine();
        m.core.cpu.regs[0] = 0x8000;
        m.core.cpu.nzcv = 0xd;
        m.core.cpu.regs[rt] = u64::MAX;
        m.core.cpu.regs[rs] = addend;
        m.memory.write_u64(0x8000, initial).unwrap();
        assert!(matches!(run(&mut m), Err(Trap::Exit { code: 0 })));
        assert_eq!(m.core.cpu.regs[rt], mask);
        if rs != rt {
            assert_eq!(m.core.cpu.regs[rs], addend);
        }
        assert_eq!(m.core.cpu.regs[0], 0x8000);
        assert_eq!(m.core.cpu.nzcv, 0xd);
        assert_eq!(m.core.steps, 2);
        assert_eq!(m.memory.read_u64(0x8000).unwrap(), initial & !mask);
    }
    let mut m = machine();
    m.core.cpu.regs[0] = 0x8001;
    m.core.cpu.regs[1] = 0xabcdef1234567801;
    m.memory.write_u64(0x8000, 0x112233445566ff88).unwrap();
    assert!(matches!(
        ldadd_byte::run(&mut m),
        Err(Trap::Exit { code: 0 })
    ));
    assert_eq!(m.core.cpu.regs[2], 0xff);
    assert_eq!(m.core.cpu.regs[1], 0xabcdef1234567801);
    assert_eq!(m.memory.read_u64(0x8000).unwrap(), 0x1122334455660088);
    let mut m = machine();
    m.core.cpu.regs[0] = 0x8000;
    m.core.cpu.regs[1] = 9;
    m.core.cpu.nzcv = 7;
    m.memory.write_u64(0x8000, 17).unwrap();
    assert!(matches!(
        ldadd_alias_base_result::run(&mut m),
        Err(Trap::Exit { code: 0 })
    ));
    assert_eq!(m.core.cpu.regs[0], 17);
    assert_eq!(m.core.cpu.regs[1], 9);
    assert_eq!(m.memory.read_u64(0x8000).unwrap(), 26);
    assert_eq!(m.core.cpu.nzcv, 7);
    let mut m = machine();
    m.core.cpu.regs[0] = 0x8000;
    m.core.cpu.regs[2] = u64::MAX;
    m.core.cpu.nzcv = 3;
    m.memory.write_u64(0x8000, 16).unwrap();
    assert!(matches!(
        ldadd_alias_base_source::run(&mut m),
        Err(Trap::Exit { code: 0 })
    ));
    assert_eq!(m.core.cpu.regs[0], 0x8000);
    assert_eq!(m.core.cpu.regs[2], 16);
    assert_eq!(m.memory.read_u64(0x8000).unwrap(), 0x8010);
    assert_eq!(m.core.cpu.nzcv, 3);
    let mut m = machine();
    m.core.cpu.sp = 0x8000;
    m.core.cpu.regs[2] = u64::MAX;
    m.core.cpu.nzcv = 5;
    m.memory.write_u64(0x8000, 0x8877665544332211).unwrap();
    assert!(matches!(
        ldadd_zero_source_sp::run(&mut m),
        Err(Trap::Exit { code: 0 })
    ));
    assert_eq!(m.core.cpu.regs[2], 0x8877665544332211);
    assert_eq!(m.core.cpu.sp, 0x8000);
    assert_eq!(m.memory.read_u64(0x8000).unwrap(), 0x8877665544332211);
    assert_eq!(m.core.cpu.nzcv, 5);
    let mut m = machine();
    m.core.cpu.sp = 0x8000;
    m.core.cpu.regs[1] = 1;
    m.core.cpu.nzcv = 9;
    m.memory.write_u64(0x8000, u64::MAX).unwrap();
    let before = m.core.cpu.regs;
    assert!(matches!(
        ldadd_discard_sp::run(&mut m),
        Err(Trap::Exit { code: 0 })
    ));
    assert_eq!(m.core.cpu.regs, before);
    assert_eq!(m.core.cpu.sp, 0x8000);
    assert_eq!(m.core.cpu.nzcv, 9);
    assert_eq!(m.memory.read_u64(0x8000).unwrap(), 0);
    for address in [0x8002] {
        let mut m = machine();
        m.core.cpu.regs[0] = address;
        m.core.cpu.regs[8] = 0xabcdef1200000001;
        m.core.cpu.nzcv = 0xf;
        m.memory.write_u64(0x8000, u64::MAX).unwrap();
        let before = m.core.cpu.clone();
        assert!(matches!(
            ldadd_word_alias::run(&mut m),
            Err(Trap::MemoryFault { .. })
        ));
        assert_eq!(m.core.cpu, before);
        assert_eq!(m.core.steps, 1);
        assert_eq!(m.memory.read_u64(0x8000).unwrap(), u64::MAX);
    }
    let mut m = machine();
    m.core.cpu.sp = 0x8008;
    m.core.cpu.regs[1] = 1;
    m.core.cpu.nzcv = 0xa;
    m.memory.write_u64(0x8008, 17).unwrap();
    let before = m.core.cpu.clone();
    assert!(matches!(
        ldadd_discard_sp::run(&mut m),
        Err(Trap::InvalidState { .. })
    ));
    assert_eq!(m.core.cpu, before);
    assert_eq!(m.memory.read_u64(0x8008).unwrap(), 17);
    // LDAXR/STLXR retry loop: one pass succeeds with no interference.
    let mut m = machine();
    m.core.cpu.regs[0] = 0x8000;
    m.core.cpu.regs[2] = 0xffffffff_ffffffff;
    m.core.cpu.nzcv = 0xd;
    m.memory.write_u64(0x8000, 41).unwrap();
    m.memory.write_u64(0x8008, 7).unwrap();
    assert!(matches!(
        exclusive_increment_loop::run(&mut m),
        Err(Trap::Exit { code: 0 })
    ));
    assert_eq!(m.memory.read_u64(0x8000).unwrap(), 42);
    assert_eq!(m.memory.read_u64(0x8008).unwrap(), 7);
    assert_eq!(
        (
            m.core.cpu.regs[1],
            m.core.cpu.regs[2],
            m.core.cpu.nzcv,
            m.core.steps
        ),
        (42, 0, 0xd, 5)
    );
    assert_eq!(m.core.memory.exclusive_reservation(), None);
    // A plain store to the reserved bytes between LDXR and STXR.
    let mut m = machine();
    m.core.cpu.regs[0] = 0x8000;
    m.core.cpu.regs[3] = 0x99;
    m.core.cpu.regs[2] = 0xabcd_0000_0000_0005;
    m.memory.write_u64(0x8000, 5).unwrap();
    // Whether the same PE's plain store clears its local monitor is
    // IMPLEMENTATION DEFINED (Arm ARM B2.9): generated stores go straight to
    // the guest view and leave it, so the STXR may store X1 and succeed.
    assert!(matches!(
        exclusive_plain_store_interference::run(&mut m),
        Err(Trap::Exit { code: 0 })
    ));
    let outcome = (
        m.core.cpu.regs[1],
        m.core.cpu.regs[2],
        m.memory.read_u64(0x8000).unwrap(),
    );
    assert!(
        outcome == (5, 1, 0x99) || outcome == (5, 0, 5),
        "{outcome:x?}"
    );
    // CLREX opens the monitor. (The SVC case needs the runtime and is tested there.)
    let mut m = machine();
    m.core.cpu.regs[0] = 0x8000;
    m.memory.write_u64(0x8000, 5).unwrap();
    assert!(matches!(
        exclusive_clrex::run(&mut m),
        Err(Trap::Exit { code: 0 })
    ));
    assert_eq!(m.core.cpu.regs[2], 1);
    assert_eq!(m.memory.read_u64(0x8000).unwrap(), 5);
    // Byte width through 16-byte-aligned SP: W results zero-extend to X.
    let mut m = machine();
    m.core.cpu.sp = 0x8ff0;
    m.core.cpu.regs[5] = u64::MAX;
    m.core.cpu.regs[6] = u64::MAX;
    m.core.cpu.regs[7] = 0xffff_ffff_ffff_ff3c;
    m.memory.write_u64(0x8ff0, 0x1122_3344_5566_77a5).unwrap();
    assert!(matches!(
        exclusive_byte_sp::run(&mut m),
        Err(Trap::Exit { code: 0 })
    ));
    assert_eq!(
        (m.core.cpu.regs[5], m.core.cpu.regs[6], m.core.cpu.sp),
        (0xa5, 0, 0x8ff0)
    );
    assert_eq!(m.memory.read_u64(0x8ff0).unwrap(), 0x1122_3344_5566_773c);
    let mut m = machine();
    m.core.cpu.sp = 0x8ff8;
    m.core.cpu.regs[5] = 3;
    let before = m.core.cpu.clone();
    assert!(matches!(
        exclusive_byte_sp::run(&mut m),
        Err(Trap::InvalidState { .. })
    ));
    assert_eq!(m.core.cpu, before);
    // STLXRH with WZR data stores zero and preserves neighboring bytes.
    let mut m = machine();
    m.core.cpu.regs[0] = 0x8002;
    m.memory.write_u64(0x8000, 0x1111_2222_3333_4444).unwrap();
    assert!(matches!(
        exclusive_halfword_zero::run(&mut m),
        Err(Trap::Exit { code: 0 })
    ));
    assert_eq!((m.core.cpu.regs[1], m.core.cpu.regs[2]), (0x3333, 0));
    assert_eq!(m.memory.read_u64(0x8000).unwrap(), 0x1111_2222_0000_4444);
    // Rt == Rn: the base is captured before the destination is written.
    let mut m = machine();
    m.core.cpu.regs[0] = 0x8010;
    m.memory.write_u64(0x8010, 0xfeed).unwrap();
    assert!(matches!(
        exclusive_load_base_alias::run(&mut m),
        Err(Trap::Exit { code: 0 })
    ));
    assert_eq!(m.core.cpu.regs[0], 0xfeed);
    // The terminating SVC is an exception, which clears the local monitor.
    assert_eq!(m.core.memory.exclusive_reservation(), None);
    let ldapr_cases: &[(fn(&mut Machine) -> Result<(), Trap>, u8, usize, usize)] = &[
        (ldapr_byte_alias::run, 1, 8, 8),
        (ldapr_halfword::run, 2, 0, 2),
        (ldapr_word::run, 4, 0, 2),
        (ldapr_doubleword::run, 8, 0, 2),
    ];
    for &(run, bytes, rn, rt) in ldapr_cases {
        let mut m = machine();
        m.core.cpu.regs[rt] = u64::MAX;
        m.core.cpu.regs[rn] = 0x8000;
        m.core.cpu.nzcv = 0xd;
        let initial = 0xfedcba9876543210;
        m.memory.write_u64(0x8000, initial).unwrap();
        m.memory.set_permissions(0x8000, 0x1000, READ).unwrap();
        assert!(matches!(run(&mut m), Err(Trap::Exit { code: 0 })));
        assert_eq!(
            m.core.cpu.regs[rt],
            initial & (u64::MAX >> (64 - bytes * 8))
        );
        assert_eq!(m.core.cpu.nzcv, 0xd);
        assert_eq!(m.core.steps, 2);
        assert_eq!(m.memory.read_u64(0x8000).unwrap(), initial);
    }
    let mut m = machine();
    m.core.cpu.regs[0] = 0x8000;
    m.core.cpu.nzcv = 3;
    let before = m.core.cpu.regs;
    assert!(matches!(
        ldapr_discard::run(&mut m),
        Err(Trap::Exit { code: 0 })
    ));
    assert_eq!(m.core.cpu.regs, before);
    assert_eq!(m.core.cpu.nzcv, 3);
    let mut m = machine();
    m.core.cpu.sp = 0x8000;
    m.memory.write_u64(0x8000, 0x123456789abcdef0).unwrap();
    assert!(matches!(ldapr_sp::run(&mut m), Err(Trap::Exit { code: 0 })));
    assert_eq!(m.core.cpu.regs[2], 0x123456789abcdef0);
    assert_eq!(m.core.cpu.sp, 0x8000);
    let mut m = machine();
    m.core.cpu.regs[0] = 0x8002;
    m.core.cpu.regs[2] = u64::MAX;
    let before = m.core.cpu.clone();
    assert!(matches!(
        ldapr_word::run(&mut m),
        Err(Trap::MemoryFault {
            address: 0x8002,
            ..
        })
    ));
    assert_eq!(m.core.cpu, before);
    let mut m = machine();
    m.core.cpu.sp = 0x8008;
    m.core.cpu.regs[2] = u64::MAX;
    let before = m.core.cpu.clone();
    assert!(matches!(
        ldapr_sp::run(&mut m),
        Err(Trap::InvalidState { .. })
    ));
    assert_eq!(m.core.cpu, before);
    let ldar_cases: &[(fn(&mut Machine) -> Result<(), Trap>, u8, usize, usize)] = &[
        (ldar_byte_alias::run, 1, 3, 3),
        (ldar_halfword::run, 2, 1, 5),
        (ldar_word::run, 4, 1, 5),
        (ldar_doubleword::run, 8, 1, 5),
    ];
    for &(run, bytes, rn, rt) in ldar_cases {
        let mut m = machine();
        m.core.cpu.regs[rt] = u64::MAX;
        m.core.cpu.regs[rn] = 0x8000;
        m.core.cpu.nzcv = 0xb;
        let initial = 0xf0e1d2c3b4a59687;
        m.memory.write_u64(0x8000, initial).unwrap();
        m.memory.set_permissions(0x8000, 0x1000, READ).unwrap();
        assert!(matches!(run(&mut m), Err(Trap::Exit { code: 0 })));
        assert_eq!(
            m.core.cpu.regs[rt],
            initial & (u64::MAX >> (64 - bytes * 8))
        );
        if rn != rt {
            assert_eq!(m.core.cpu.regs[rn], 0x8000);
        }
        assert_eq!(m.core.cpu.nzcv, 0xb);
        assert_eq!(m.core.steps, 2);
        assert_eq!(m.memory.read_u64(0x8000).unwrap(), initial);
    }
    let mut m = machine();
    m.core.cpu.regs[3] = 0x8001;
    m.core.cpu.nzcv = 9;
    m.memory.write_u16(0x8000, 0x96ff).unwrap();
    assert!(matches!(
        ldar_byte_alias::run(&mut m),
        Err(Trap::Exit { code: 0 })
    ));
    assert_eq!(m.core.cpu.regs[3], 0x96);
    assert_eq!(m.core.cpu.nzcv, 9);
    let mut m = machine();
    m.core.cpu.regs[1] = 0x8000;
    m.core.cpu.nzcv = 5;
    let before = m.core.cpu.regs;
    assert!(matches!(
        ldar_discard::run(&mut m),
        Err(Trap::Exit { code: 0 })
    ));
    assert_eq!(m.core.cpu.regs, before);
    assert_eq!(m.core.cpu.nzcv, 5);
    let mut m = machine();
    m.core.cpu.sp = 0x8000;
    m.core.cpu.nzcv = 7;
    m.memory.write_u64(0x8000, 0x8070605040302010).unwrap();
    assert!(matches!(ldar_sp::run(&mut m), Err(Trap::Exit { code: 0 })));
    assert_eq!(m.core.cpu.regs[5], 0x8070605040302010);
    assert_eq!(m.core.cpu.sp, 0x8000);
    assert_eq!(m.core.cpu.nzcv, 7);
    let mut m = machine();
    m.core.cpu.regs[1] = 0x8002;
    m.core.cpu.regs[5] = u64::MAX;
    m.core.cpu.nzcv = 0xf;
    let before = m.core.cpu.clone();
    assert!(matches!(
        ldar_word::run(&mut m),
        Err(Trap::MemoryFault {
            address: 0x8002,
            ..
        })
    ));
    assert_eq!(m.core.cpu, before);
    assert_eq!(m.core.steps, 1);
    let mut m = machine();
    m.core.cpu.sp = 0x8008;
    m.core.cpu.regs[5] = u64::MAX;
    m.core.cpu.nzcv = 0xa;
    let before = m.core.cpu.clone();
    assert!(matches!(
        ldar_sp::run(&mut m),
        Err(Trap::InvalidState { .. })
    ));
    assert_eq!(m.core.cpu, before);
    assert_eq!(m.core.steps, 1);
    let stlr_cases: &[(fn(&mut Machine) -> Result<(), Trap>, u8, usize)] = &[
        (stlr_byte::run, 1, 21),
        (stlr_halfword::run, 2, 2),
        (stlr_word::run, 4, 2),
        (stlr_doubleword::run, 8, 2),
    ];
    for &(run, bytes, rt) in stlr_cases {
        let mut m = machine();
        m.core.cpu.regs[0] = 0x8000;
        m.core.cpu.nzcv = 0xd;
        let initial = 0x1122334455667788;
        let value = 0xfedcba9876543210;
        let mask = u64::MAX >> (64 - bytes * 8);
        m.core.cpu.regs[rt] = value;
        m.memory.write_u64(0x8000, initial).unwrap();
        let before = m.core.cpu.regs;
        assert!(matches!(run(&mut m), Err(Trap::Exit { code: 0 })));
        assert_eq!(m.core.cpu.regs, before);
        assert_eq!(m.core.cpu.nzcv, 0xd);
        assert_eq!(m.core.steps, 2);
        m.memory.set_permissions(0x8000, 0x1000, READ).unwrap();
        assert_eq!(
            m.memory.read_u64(0x8000).unwrap(),
            (initial & !mask) | (value & mask)
        );
    }
    let mut m = machine();
    m.core.cpu.regs[0] = 0x8001;
    m.core.cpu.regs[21] = 0xaabbccdddddd9999;
    m.memory.write_u64(0x8000, 0x1122334455667788).unwrap();
    assert!(matches!(
        stlr_byte::run(&mut m),
        Err(Trap::Exit { code: 0 })
    ));
    assert_eq!(m.memory.read_u64(0x8000).unwrap(), 0x1122334455669988);
    assert_eq!(m.core.cpu.regs[21], 0xaabbccdddddd9999);
    let mut m = machine();
    m.core.cpu.regs[0] = 0x8000;
    m.core.cpu.nzcv = 7;
    assert!(matches!(
        stlr_alias::run(&mut m),
        Err(Trap::Exit { code: 0 })
    ));
    assert_eq!(m.memory.read_u64(0x8000).unwrap(), 0x8000);
    assert_eq!(m.core.cpu.regs[0], 0x8000);
    assert_eq!(m.core.cpu.nzcv, 7);
    let mut m = machine();
    m.core.cpu.sp = 0x8000;
    m.core.cpu.nzcv = 3;
    m.memory.write_u64(0x8000, u64::MAX).unwrap();
    assert!(matches!(
        stlr_zero_sp::run(&mut m),
        Err(Trap::Exit { code: 0 })
    ));
    assert_eq!(m.memory.read_u64(0x8000).unwrap(), 0);
    assert_eq!(m.core.cpu.sp, 0x8000);
    assert_eq!(m.core.cpu.nzcv, 3);
    for address in [0x8002] {
        let mut m = machine();
        m.core.cpu.regs[0] = address;
        m.core.cpu.regs[2] = 0xabcdef1200001234;
        m.core.cpu.nzcv = 9;
        m.memory.write_u64(0x8000, u64::MAX).unwrap();
        let before = m.core.cpu.clone();
        assert!(matches!(
            stlr_word::run(&mut m),
            Err(Trap::MemoryFault { .. })
        ));
        assert_eq!(m.core.cpu, before);
        assert_eq!(m.core.steps, 1);
        assert_eq!(m.memory.read_u64(0x8000).unwrap(), u64::MAX);
    }
    let mut m = machine();
    m.core.cpu.sp = 0x8008;
    m.core.cpu.nzcv = 5;
    m.memory.write_u64(0x8008, u64::MAX).unwrap();
    let before = m.core.cpu.clone();
    assert!(matches!(
        stlr_zero_sp::run(&mut m),
        Err(Trap::InvalidState { .. })
    ));
    assert_eq!(m.core.cpu, before);
    assert_eq!(m.memory.read_u64(0x8008).unwrap(), u64::MAX);
    let mut m = machine();
}
