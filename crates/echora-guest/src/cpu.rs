//! Integer register and flag semantics used by statically generated AArch64 code.

use crate::Trap;
use crate::view::GuestView;
use serde::Serialize;

/// The prototype's explicit instruction-pointer stripping configuration.
pub const PAUTH_ADDRESS_BITS: u8 = 39;
pub const PAUTH_TOP_BYTE_IGNORE: bool = false;

/// This prototype models instruction-address authentication as disabled for
/// both keys (SCTLR.EnIA = SCTLR.EnIB = 0). There is deliberately no enabled
/// policy: it does not generate or verify pointer-authentication codes. XPACLRI
/// stripping is modeled separately with the fixed VA-width/TBI configuration.
/// BTI, guarded control stacks, and additional PAuth extensions are not modeled.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PointerAuthenticationPolicy {
    #[default]
    Disabled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum InstructionAuthKey {
    A,
    B,
}

/// Only the four LR/SP hint forms currently have explicit disabled-key semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum PAuthHint {
    PaciaSp,
    PacibSp,
    AutiaSp,
    AutibSp,
    /// The X17-by-X16 forms; with keys disabled they leave X17 unchanged.
    Pacia1716,
    Pacib1716,
    Autia1716,
    Autib1716,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Cpu {
    pub regs: [u64; 31],
    /// Raw Q-register bits only; this does not implement SIMD/FP arithmetic.
    /// Q31 is an ordinary register, unlike the encoding of XZR in the GPR bank.
    pub vectors: [u128; 32],
    pub sp: u64,
    pub pc: u64,
    /// N, Z, C, V in bits 3, 2, 1, 0.
    pub nzcv: u8,
    pub tpidr_el0: u64,
    pub tpidrro_el0: u64,
    /// Raw status/control storage for thread-context inheritance. FP arithmetic
    /// and access instructions for these registers are not implemented yet.
    pub fpcr: u32,
    pub fpsr: u32,
}

#[inline(always)]
pub const fn width_mask(bits: u8) -> u64 {
    match bits {
        32 => u32::MAX as u64,
        64 => u64::MAX,
        _ => panic!("register width must be 32 or 64"),
    }
}

pub const fn sign_extend(value: u64, bits: u8) -> u64 {
    assert!(bits > 0 && bits <= 64);
    if bits == 64 {
        value
    } else {
        (((value << (64 - bits)) as i64) >> (64 - bits)) as u64
    }
}

/// Shift types use the architectural encoding: LSL, LSR, ASR, ROR.
#[inline]
pub fn shift(value: u64, kind: u8, amount: u8, bits: u8) -> u64 {
    assert!(amount < bits);
    let value = value & width_mask(bits);
    match kind {
        0 => value.wrapping_shl(amount.into()) & width_mask(bits),
        1 => value >> amount,
        2 => ((sign_extend(value, bits) as i64) >> amount) as u64 & width_mask(bits),
        3 if bits == 32 => (value as u32).rotate_right(amount.into()) as u64,
        3 => value.rotate_right(amount.into()),
        _ => panic!("invalid shift type"),
    }
}

/// UXTB/UXTH/UXTW/UXTX/SXTB/SXTH/SXTW/SXTX, followed by an optional LSL.
pub fn extend(value: u64, option: u8, amount: u8) -> u64 {
    assert!(option < 8 && amount < 64);
    let bits = 8_u8 << (option & 3);
    let result = if option & 4 != 0 {
        sign_extend(value, bits)
    } else if bits == 64 {
        value
    } else {
        value & ((1_u64 << bits) - 1)
    };
    result.wrapping_shl(amount.into())
}

/// UMADDL multiplies two unsigned W operands and adds a 64-bit accumulator.
/// Only the low 32 bits of each multiplicand participate; the sum wraps at 64.
pub fn unsigned_multiply_add_long(lhs: u64, rhs: u64, addend: u64) -> u64 {
    (lhs as u32 as u64)
        .wrapping_mul(rhs as u32 as u64)
        .wrapping_add(addend)
}

impl Cpu {
    /// Fixed architectural policy, exposed for runtime reports. Adding another
    /// policy requires implementing it in the dedicated helpers below.
    pub const fn pointer_authentication_policy(&self) -> PointerAuthenticationPolicy {
        PointerAuthenticationPolicy::Disabled
    }

    /// With instruction-address authentication disabled, PAC/AUT hints leave LR
    /// unchanged. XPACLRI has separate VA-width/TBI-dependent stripping semantics.
    pub fn pauth_disabled_hint(&mut self, hint: PAuthHint) {
        match (self.pointer_authentication_policy(), hint) {
            (
                PointerAuthenticationPolicy::Disabled,
                PAuthHint::PaciaSp
                | PAuthHint::PacibSp
                | PAuthHint::AutiaSp
                | PAuthHint::AutibSp
                | PAuthHint::Pacia1716
                | PAuthHint::Pacib1716
                | PAuthHint::Autia1716
                | PAuthHint::Autib1716,
            ) => {}
        }
    }

    /// RETAA/RETAB authenticate through key A/B respectively, then return. With
    /// both authentication enables clear, the target is the unchanged LR. Do not
    /// mask its high bits: disabled authentication does not remove an existing PAC.
    pub fn pauth_disabled_return(&self, key: InstructionAuthKey) -> u64 {
        match (self.pointer_authentication_policy(), key) {
            (
                PointerAuthenticationPolicy::Disabled,
                InstructionAuthKey::A | InstructionAuthKey::B,
            ) => self.regs[30],
        }
    }

    /// XPACLRI strips instruction PAC bits without authenticating. Under the
    /// fixed 39-bit, TBI-disabled model, bit 55 selects the extension of bits
    /// 63:39; bit 38 is part of the address and must not select sign extension.
    /// Reject another memory model before modifying any architectural state.
    pub fn strip_lr_instruction_pac(&mut self, address_space_end: u64) -> Result<(), Trap> {
        if PAUTH_TOP_BYTE_IGNORE || address_space_end != (1_u64 << PAUTH_ADDRESS_BITS) {
            return Err(Trap::InvalidState {
                message: "XPACLRI requires the modeled 39-bit address space with TBI disabled"
                    .into(),
            });
        }
        let pointer = self.regs[30];
        let mask = (1_u64 << PAUTH_ADDRESS_BITS) - 1;
        let extension = 0_u64.wrapping_sub((pointer >> 55) & 1);
        self.regs[30] = (pointer & mask) | (extension & !mask);
        Ok(())
    }

    /// XPACI/XPACD Xd: the same stripping as XPACLRI applied to Xd (with TBI
    /// disabled, instruction and data pointers strip identically).
    pub fn strip_pac(&mut self, rd: u8, address_space_end: u64) -> Result<(), Trap> {
        if PAUTH_TOP_BYTE_IGNORE || address_space_end != (1_u64 << PAUTH_ADDRESS_BITS) {
            return Err(Trap::InvalidState {
                message: "XPAC requires the modeled 39-bit address space with TBI disabled".into(),
            });
        }
        let pointer = self.read_reg(rd, 64, false);
        let mask = (1_u64 << PAUTH_ADDRESS_BITS) - 1;
        let extension = 0_u64.wrapping_sub((pointer >> 55) & 1);
        self.write_reg(rd, 64, false, (pointer & mask) | (extension & !mask));
        Ok(())
    }

    /// Store raw Q bits in architectural register order and little-endian byte
    /// order. Memory validates all 32 bytes before committing any of the store.
    pub fn store_q_pair(
        &self,
        memory: &mut GuestView,
        address: u64,
        rt: u8,
        rt2: u8,
    ) -> Result<(), Trap> {
        let mut bytes = [0_u8; 32];
        bytes[..16].copy_from_slice(&self.vectors[rt as usize].to_le_bytes());
        bytes[16..].copy_from_slice(&self.vectors[rt2 as usize].to_le_bytes());
        memory.write(address, &bytes)
    }

    /// Load both Q registers only after Memory has checked and read the entire
    /// pair. General-purpose address writeback is performed by emitted code only
    /// after this succeeds; vector and GPR indices do not alias one another.
    pub fn load_q_pair(
        &mut self,
        memory: &GuestView,
        address: u64,
        rt: u8,
        rt2: u8,
    ) -> Result<(), Trap> {
        let mut bytes = [0_u8; 32];
        memory.read(address, &mut bytes)?;
        let first = u128::from_le_bytes(bytes[..16].try_into().expect("fixed 16-byte Q register"));
        let second = u128::from_le_bytes(bytes[16..].try_into().expect("fixed 16-byte Q register"));
        self.vectors[rt as usize] = first;
        self.vectors[rt2 as usize] = second;
        Ok(())
    }

    /// Register 31 is SP only in instruction forms that explicitly permit it.
    #[inline]
    pub fn read_reg(&self, index: u8, bits: u8, stack_pointer: bool) -> u64 {
        let value = if index == 31 {
            if stack_pointer { self.sp } else { 0 }
        } else {
            self.regs[index as usize]
        };
        value & width_mask(bits)
    }

    /// Every W-register write clears the corresponding X-register's upper half.
    #[inline]
    pub fn write_reg(&mut self, index: u8, bits: u8, stack_pointer: bool, value: u64) {
        let value = value & width_mask(bits);
        if index == 31 {
            if stack_pointer {
                self.sp = value;
            }
        } else {
            self.regs[index as usize] = value;
        }
    }

    #[inline(always)]
    pub fn add_with_carry(
        &mut self,
        lhs: u64,
        rhs: u64,
        carry: bool,
        bits: u8,
        flags: bool,
    ) -> u64 {
        let mask = width_mask(bits);
        let lhs = lhs & mask;
        let rhs = rhs & mask;
        let wide = lhs as u128 + rhs as u128 + u128::from(carry);
        let result = wide as u64 & mask;
        if flags {
            let sign = 1_u64 << (bits - 1);
            let n = result & sign != 0;
            let z = result == 0;
            let c = (wide >> bits) != 0;
            let v = (!(lhs ^ rhs) & (lhs ^ result) & sign) != 0;
            self.nzcv = ((n as u8) << 3) | ((z as u8) << 2) | ((c as u8) << 1) | v as u8;
        }
        result
    }

    #[inline]
    pub fn add_sub(&mut self, lhs: u64, rhs: u64, subtract: bool, bits: u8, flags: bool) -> u64 {
        if subtract {
            self.add_with_carry(lhs, !rhs, true, bits, flags)
        } else {
            self.add_with_carry(lhs, rhs, false, bits, flags)
        }
    }

    /// CCMP tests the incoming NZCV before replacing it. When the predicate is
    /// false, all four flags come directly from the instruction's NZCV literal.
    pub fn conditional_compare(&mut self, lhs: u64, rhs: u64, bits: u8, condition: u8, nzcv: u8) {
        if self.condition(condition) {
            self.add_sub(lhs, rhs, true, bits, true);
        } else {
            self.nzcv = nzcv & 15;
        }
    }

    /// CCMN: as CCMP, with flags from Rn + operand.
    pub fn conditional_compare_negative(
        &mut self,
        lhs: u64,
        rhs: u64,
        bits: u8,
        condition: u8,
        nzcv: u8,
    ) {
        if self.condition(condition) {
            self.add_sub(lhs, rhs, false, bits, true);
        } else {
            self.nzcv = nzcv & 15;
        }
    }

    pub fn logical_flags(&mut self, value: u64, bits: u8) {
        let value = value & width_mask(bits);
        self.nzcv = (((value >> (bits - 1)) as u8) << 3) | (u8::from(value == 0) << 2);
    }

    #[inline]
    pub fn condition(&self, condition: u8) -> bool {
        let n = self.nzcv & 8 != 0;
        let z = self.nzcv & 4 != 0;
        let c = self.nzcv & 2 != 0;
        let v = self.nzcv & 1 != 0;
        let result = match (condition >> 1) & 7 {
            0 => z,
            1 => c,
            2 => n,
            3 => v,
            4 => c && !z,
            5 => n == v,
            6 => n == v && !z,
            7 => true,
            _ => unreachable!(),
        };
        if condition & 1 != 0 && condition != 15 {
            !result
        } else {
            result
        }
    }
}

/// One CRC32 (`castagnoli` false, polynomial 0x04C11DB7) or CRC32C
/// (polynomial 0x1EDC6F41) step over the low `bytes` bytes of `data`, in the
/// reflected bit order the instructions use, without inversion.
pub fn crc32(acc: u32, data: u64, bytes: u8, castagnoli: bool) -> u32 {
    let poly: u32 = if castagnoli { 0x82f6_3b78 } else { 0xedb8_8320 };
    let mut crc = acc;
    for byte in 0..bytes {
        crc ^= u32::from((data >> (8 * byte)) as u8);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (poly & (crc & 1).wrapping_neg());
        }
    }
    crc
}

#[cfg(test)]
mod tests {

    #[test]
    fn crc32_matches_standard_check_values() {
        for (castagnoli, check) in [(false, 0xcbf4_3926u32), (true, 0xe306_9283)] {
            let crc = b"123456789".iter().fold(!0u32, |acc, &b| {
                super::crc32(acc, u64::from(b), 1, castagnoli)
            });
            assert_eq!(!crc, check);
            // Wider steps equal the same bytes one at a time.
            let word = u64::from_le_bytes(*b"12345678");
            let bytewise = (0..8).fold(!0u32, |acc, i| {
                super::crc32(acc, word >> (8 * i), 1, castagnoli)
            });
            assert_eq!(super::crc32(!0, word, 8, castagnoli), bytewise);
        }
    }
    use super::*;
    #[cfg(target_os = "linux")]
    use crate::memory::{Memory, READ, STATE_STACK, WRITE};

    #[test]
    fn xpaclri_uses_bit55_and_preserves_low39_bits_and_other_state() {
        let mut cpu = Cpu {
            nzcv: 0b1011,
            sp: 0x7000,
            ..Cpu::default()
        };
        cpu.regs[17] = 0x1234_5678;
        // Bit38 is set in the normal low address range; it does not request a
        // negative extension. Nonzero top-byte and PAC bits are stripped.
        for pointer in [0x7100001234, 0xab7f_ff7100001234] {
            cpu.regs[30] = pointer;
            cpu.strip_lr_instruction_pac(1_u64 << 39).unwrap();
            assert_eq!(cpu.regs[30], 0x7100001234);
        }
        cpu.regs[30] = 0x1280_0000_1234_5678; // Original bit55=1.
        cpu.strip_lr_instruction_pac(1_u64 << 39).unwrap();
        assert_eq!(cpu.regs[30], 0xffff_ff80_1234_5678);
        assert_eq!(cpu.nzcv, 0b1011);
        assert_eq!(cpu.sp, 0x7000);
        assert_eq!(cpu.regs[17], 0x1234_5678);
        let original = cpu.clone();
        assert!(cpu.strip_lr_instruction_pac(1_u64 << 48).is_err());
        assert_eq!(cpu, original);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn q_pair_stack_roundtrip_preserves_all128_bits_and_register31() {
        let mut memory = Memory::new(0x10000);
        memory
            .map(0x1000, 0x1000, READ | WRITE, STATE_STACK, "test stack")
            .unwrap();
        let mut view = memory.view();
        let mut cpu = Cpu {
            sp: 0x2000,
            nzcv: 0b1011,
            ..Cpu::default()
        };
        for index in 0..8 {
            cpu.vectors[index] = (0x8877_6655_4433_2200_u128 + index as u128) << 64
                | (0xffee_ddcc_bbaa_9900 + index as u128);
        }
        let original = cpu.vectors;
        for first in [6, 4, 2, 0] {
            let address = cpu.sp - 32;
            cpu.store_q_pair(&mut view, address, first, first + 1)
                .unwrap();
            cpu.sp = address;
        }
        assert_eq!(cpu.sp, 0x1f80);
        assert_eq!(memory.read_u64(cpu.sp).unwrap(), original[0] as u64);
        assert_eq!(
            memory.read_u64(cpu.sp + 8).unwrap(),
            (original[0] >> 64) as u64
        );
        cpu.vectors[..8].fill(0);
        for first in [0, 2, 4, 6] {
            cpu.load_q_pair(&view, cpu.sp, first, first + 1).unwrap();
            cpu.sp += 32;
        }
        assert_eq!(cpu.vectors, original);
        assert_eq!(cpu.sp, 0x2000);
        assert_eq!(cpu.nzcv, 0b1011);
        cpu.vectors[31] = u128::MAX;
        cpu.store_q_pair(&mut view, 0x1000, 31, 0).unwrap();
        cpu.vectors[31] = 0;
        cpu.load_q_pair(&view, 0x1000, 31, 0).unwrap();
        assert_eq!(cpu.vectors[31], u128::MAX);
    }

    #[test]
    fn unsigned_long_multiply_uses_w_operands_and_a_wrapping_x_accumulator() {
        assert_eq!(
            unsigned_multiply_add_long(0xffff_ffff, 0xffff_ffff, 0),
            0xffff_fffe_0000_0001
        );
        assert_eq!(
            unsigned_multiply_add_long(0xffff_ffff, 0xffff_ffff, 0x1_ffff_ffff),
            0
        );
        assert_eq!(
            unsigned_multiply_add_long(0xabcd_0000_ffff_fffe, 0x9999_0000_0000_0003, 1),
            0x2_ffff_fffb
        );
        let mut cpu = Cpu {
            sp: 0x1234_5000,
            nzcv: 0b1011,
            ..Cpu::default()
        };
        cpu.regs[1] = 0xffff_ffff;
        cpu.regs[2] = 2;
        let result = unsigned_multiply_add_long(
            cpu.read_reg(1, 32, false),
            cpu.read_reg(2, 32, false),
            cpu.read_reg(31, 64, false),
        );
        cpu.write_reg(0, 64, false, result);
        assert_eq!(cpu.regs[0], 0x1_ffff_fffe);
        assert_eq!(cpu.nzcv, 0b1011);
        assert_eq!(cpu.sp, 0x1234_5000);
        assert_eq!(
            unsigned_multiply_add_long(cpu.read_reg(31, 32, false), 99, 7),
            7
        );
    }

    #[test]
    fn conditional_compare_uses_old_flags_and_preserves_registers() {
        let mut cpu = Cpu::default();
        cpu.regs[0] = 0xaabb_ccdd_eeff_0000;
        cpu.sp = 0x5678_9000;
        let registers = cpu.regs;
        for fallback in 0..16 {
            cpu.nzcv = 0b0100; // Z=1, so NE is false.
            cpu.conditional_compare(0, 1, 64, 1, fallback);
            assert_eq!(cpu.nzcv, fallback);
        }
        cpu.nzcv = 0; // NE true; 32-bit comparison ignores the high word.
        cpu.conditional_compare(0x1_0000_0000, 0, 32, 1, 15);
        assert_eq!(cpu.nzcv, 0b0110);
        // The previous comparison changed Z. The next NE must now be false.
        cpu.conditional_compare(0, 1, 32, 1, 0b1011);
        assert_eq!(cpu.nzcv, 0b1011);
        // Unconditional 64-bit comparison must retain the high word.
        cpu.conditional_compare(0x1_0000_0000, 0, 64, 14, 15);
        assert_eq!(cpu.nzcv, 0b0010);
        cpu.conditional_compare(0x8000_0000_0000_0000, 1, 64, 14, 0);
        assert_eq!(cpu.nzcv, 0b0011); // Positive result, no borrow, signed overflow.
        assert_eq!(cpu.regs, registers);
        assert_eq!(cpu.sp, 0x5678_9000);
    }

    #[test]
    fn disabled_pauth_hints_preserve_every_register_and_flag() {
        let mut cpu = Cpu {
            regs: std::array::from_fn(|index| 0xa500_0000_0000_0000 | index as u64),
            vectors: std::array::from_fn(|index| u128::MAX - index as u128),
            sp: 0x1234_5678_9000,
            pc: 0x8765_4320,
            nzcv: 0b1011,
            tpidr_el0: 0x2000,
            tpidrro_el0: 0x3000,
            fpcr: 0x00400000,
            fpsr: 1,
        };
        let original = cpu.clone();
        assert_eq!(
            cpu.pointer_authentication_policy(),
            PointerAuthenticationPolicy::Disabled
        );
        for hint in [
            PAuthHint::PaciaSp,
            PAuthHint::PacibSp,
            PAuthHint::AutiaSp,
            PAuthHint::AutibSp,
        ] {
            cpu.pauth_disabled_hint(hint);
            assert_eq!(cpu, original, "{hint:?}");
        }
    }

    #[test]
    fn disabled_pauth_returns_use_lr_without_signing_stripping_or_mutation() {
        let mut cpu = Cpu::default();
        cpu.regs[30] = 0xa5ff_1234_5678_9000;
        cpu.sp = 0x1230;
        cpu.nzcv = 0b1101;
        let original = cpu.clone();
        for key in [InstructionAuthKey::A, InstructionAuthKey::B] {
            assert_eq!(cpu.pauth_disabled_return(key), 0xa5ff_1234_5678_9000);
            assert_eq!(cpu, original);
        }
    }

    #[test]
    fn w_writes_zero_extend_and_sp_is_distinct_from_zero_register() {
        let mut cpu = Cpu::default();
        cpu.regs[3] = u64::MAX;
        cpu.write_reg(3, 32, false, 0x1234_5678_8765_4321);
        assert_eq!(cpu.regs[3], 0x8765_4321);
        cpu.sp = 0xffff_ffff_0000_1000;
        assert_eq!(cpu.read_reg(31, 64, false), 0);
        cpu.write_reg(31, 64, false, 55);
        assert_eq!(cpu.sp, 0xffff_ffff_0000_1000);
        cpu.write_reg(31, 32, true, 0xffff_ffff_1234_5678);
        assert_eq!(cpu.sp, 0x1234_5678);
        assert_eq!(cpu.read_reg(31, 64, true), 0x1234_5678);
    }

    #[test]
    fn arithmetic_flags_cover_carry_borrow_and_signed_overflow() {
        let mut cpu = Cpu::default();
        assert_eq!(cpu.add_sub(u32::MAX as u64, 1, false, 32, true), 0);
        assert_eq!(cpu.nzcv, 0b0110);
        assert_eq!(cpu.add_sub(0x7fff_ffff, 1, false, 32, true), 0x8000_0000);
        assert_eq!(cpu.nzcv, 0b1001);
        assert_eq!(cpu.add_sub(0, 1, true, 64, true), u64::MAX);
        assert_eq!(cpu.nzcv, 0b1000);
        assert_eq!(cpu.add_sub(5, 5, true, 64, true), 0);
        assert_eq!(cpu.nzcv, 0b0110);
        assert_eq!(
            cpu.add_sub(0x8000_0000_0000_0000, 1, true, 64, true),
            0x7fff_ffff_ffff_ffff
        );
        assert_eq!(cpu.nzcv, 0b0011);
        cpu.nzcv = 0b1010;
        cpu.add_sub(0, 0, false, 64, false);
        assert_eq!(cpu.nzcv, 0b1010);
    }

    #[test]
    fn condition_codes_follow_unsigned_and_signed_flags() {
        let mut cpu = Cpu::default();
        cpu.add_sub(1, 2, true, 32, true);
        assert!(!cpu.condition(0)); // EQ
        assert!(cpu.condition(1)); // NE
        assert!(cpu.condition(3)); // CC/LO
        assert!(cpu.condition(11)); // LT
        assert!(!cpu.condition(12)); // GT
        assert!(cpu.condition(13)); // LE
        for flags in 0..16 {
            cpu.nzcv = flags;
            for cond in (0..14).step_by(2) {
                assert_ne!(cpu.condition(cond), cpu.condition(cond + 1));
            }
            assert!(cpu.condition(14));
            assert!(cpu.condition(15));
        }
    }

    #[test]
    fn extension_and_shifts_preserve_architectural_width() {
        assert_eq!(sign_extend(0x80, 8), 0xffff_ffff_ffff_ff80);
        assert_eq!(sign_extend(0x8000_0000, 32), 0xffff_ffff_8000_0000);
        assert_eq!(extend(0xffff_ff80, 0, 0), 0x80);
        assert_eq!(extend(0xffff_ff80, 4, 2), 0xffff_ffff_ffff_fe00);
        assert_eq!(shift(0x8000_0000, 2, 1, 32), 0xc000_0000);
        assert_eq!(shift(0x8000_0000, 1, 1, 32), 0x4000_0000);
        assert_eq!(shift(1, 3, 1, 32), 0x8000_0000);
        assert_eq!(shift(1, 3, 0, 64), 1);
    }

    #[test]
    fn add_with_carry_consumes_carry_without_losing_unsigned_overflow() {
        let mut cpu = Cpu::default();
        assert_eq!(cpu.add_with_carry(u64::MAX, 0, true, 64, true), 0);
        assert_eq!(cpu.nzcv, 0b0110);
        assert_eq!(cpu.add_with_carry(0, !0_u64, false, 64, true), u64::MAX);
        assert_eq!(cpu.nzcv, 0b1000);
    }
}
