//! `check_effects`: `echora_lift::defuse` effects against lifted code,
//! shared by the semantic suites (`#[path]`-included by each).
#![allow(dead_code)]

use echora_lift::aarch64::Instruction;

/// The generator of the values `check_effects` changes registers to.
pub struct EffectRandom(pub u64);

impl EffectRandom {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn value(&mut self) -> u64 {
        self.next()
    }
}

/// Checks `echora_lift::defuse::effects` of `instruction`, lifted at
/// `entry` (followed by the SVCs ending the run), against what its lifted
/// code does, from the state the caller put in `m` (the `Cpu` and the
/// data at `data`, of `size` bytes). It runs twice: from that state, and
/// with every general register, SP and NZCV the effects say it does not
/// read set to other values. Window calls keep registers outside a
/// callee's summary in the caller's locals, so these sets must be exact:
/// the trap, PC, vectors, FPCR, FPSR and the data must come out the same,
/// every register the effects say it may write the same in both runs, and
/// every other one unchanged from where its run started.
pub fn check_effects(
    m: &mut echora_aot_tests::Machine,
    dispatch: fn(&mut echora_aot_tests::Core) -> Result<(), echora_aot_tests::Trap>,
    instruction: &Instruction,
    entry: u64,
    data: u64,
    size: u64,
    random: &mut EffectRandom,
) -> Result<(), String> {
    use echora_aot_tests::Trap;
    let effects = echora_lift::defuse::effects(instruction);
    let start = m.core.cpu.clone();
    let mut memory = vec![0; size as usize];
    m.memory.read(data, &mut memory).expect("data is mapped");
    let run = |m: &mut echora_aot_tests::Machine, cpu: &echora_guest::cpu::Cpu| {
        m.core.cpu = cpu.clone();
        m.core.cpu.pc = entry;
        m.memory.write(data, &memory).expect("data is mapped");
        m.core.memory.clear_exclusive();
        m.core.spin_polls = 0;
        let trap = loop {
            match dispatch(&mut m.core) {
                Ok(()) => continue,
                Err(Trap::HostBoundary) => {
                    m.core.spin_polls = 0;
                    m.core.host_check_at = u64::MAX;
                    m.core.rearm();
                }
                Err(trap) => break trap,
            }
        };
        m.core.settle_block_trap();
        let mut after = vec![0; size as usize];
        m.memory.read(data, &mut after).expect("data is mapped");
        (format!("{trap:?}"), m.core.cpu.clone(), after)
    };
    // Another value: pointers into the data and small indices stay in
    // range (and aligned as they were), so an access through a register
    // the effects miss reads or writes elsewhere instead of faulting.
    let mut change = |value: u64| {
        if (data..data + size).contains(&value) {
            value ^ 0x40
        } else if value < 0x1000 {
            value ^ 0x8
        } else {
            random.value()
        }
    };
    let mut other = start.clone();
    for n in 0..31u8 {
        if !effects.reads.has_gpr(n) {
            other.regs[n as usize] = change(other.regs[n as usize]);
        }
    }
    if !effects.reads.sp {
        other.sp = change(other.sp) & !15;
    }
    if !effects.reads.nzcv {
        other.nzcv = !other.nzcv & 15;
    }
    let (trap, first, written) = run(m, &start);
    let (trap2, second, written2) = run(m, &other);
    let mut problems = Vec::new();
    if trap != trap2 {
        problems.push(format!("trap {trap} became {trap2}"));
    }
    if written != written2 {
        problems.push("memory differs".to_owned());
    }
    if first.pc != second.pc || first.vectors != second.vectors {
        problems.push("PC or vectors differ".to_owned());
    }
    if (first.fpcr, first.fpsr) != (second.fpcr, second.fpsr) {
        problems.push("FPCR or FPSR differ".to_owned());
    }
    let mut check = |name: String, written: bool, values: [u64; 4]| {
        let [a, b, start_a, start_b] = values;
        if written && a != b {
            problems.push(format!("{name} (a write) depends on registers not read"));
        }
        if !written && (a != start_a || b != start_b) {
            problems.push(format!("{name} changed outside the writes"));
        }
    };
    for n in 0..31 {
        check(
            format!("x{n}"),
            effects.writes.has_gpr(n as u8),
            [first.regs[n], second.regs[n], start.regs[n], other.regs[n]],
        );
    }
    check(
        "sp".to_owned(),
        effects.writes.sp,
        [first.sp, second.sp, start.sp, other.sp],
    );
    check(
        "nzcv".to_owned(),
        effects.writes.nzcv,
        [first.nzcv, second.nzcv, start.nzcv, other.nzcv].map(u64::from),
    );
    if problems.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{instruction:?} (reads {:?}, writes {:?}): {}",
            effects.reads,
            effects.writes,
            problems.join("; ")
        ))
    }
}

/// States per instruction the semantic suites check `check_effects` from.
pub const EFFECT_STATES: usize = 4;

/// The `check_effects` results of a suite: printed, and none may fail.
pub fn assert_effects(checks: usize, failures: &[String]) {
    println!("{checks} register effect checks against echora_lift::defuse");
    assert!(checks > 0, "no effect checks ran");
    assert!(
        failures.is_empty(),
        "{} register effects differ from echora_lift::defuse:\n{}",
        failures.len(),
        failures[..failures.len().min(40)].join("\n")
    );
}
