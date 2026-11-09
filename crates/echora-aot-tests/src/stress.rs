// The concurrency stress programs of `tests/stress.rs`, as AArch64 code
// assembled by dynasm-rs: the patterns a job system uses, run on several
// Cores sharing one Memory, and lifted by `build.rs`. Original code.
//
// On entry: x0 = shared data (page aligned, zeroed), x1 = this thread's
// index, x2 = thread count, x3 = iterations (or items). Each ends with
// SVC #7; x10 and x11 hold what the harness checks.

use dynasmrt::aarch64::Aarch64Relocation;
use dynasmrt::{DynasmApi, DynasmLabelApi, VecAssembler, dynasm};

type Assembler = VecAssembler<Aarch64Relocation>;

/// NOP, which pads the code to an alignment.
const NOP: u32 = 0xd503_201f;

/// Pads with NOPs to a multiple of `to` bytes.
fn align(ops: &mut Assembler, to: usize) {
    while !ops.offset().0.is_multiple_of(to) {
        ops.push_u32(NOP);
    }
}

/// Where the stress programs are loaded.
pub const STRESS_BASE: u64 = 0x300_0000;

/// Every program by name, with its entry offset from `STRESS_BASE`.
pub const STRESS_ENTRIES: &[(&str, u64)] = &[
    ("claim_cas", 0x0),
    ("claim_llsc", 0x400),
    ("claim_ldadd", 0x800),
    ("ticket", 0xc00),
    ("seqlock", 0x1000),
    ("pass_acquire", 0x1400),
    ("pass_dmb", 0x1800),
    ("casp_stack", 0x1c00),
    ("pair_halves", 0x2000),
    ("vector_halves", 0x2400),
];

/// The image as words: the programs, NOP padded between them, and zeros
/// to a whole page.
#[allow(clippy::unwrap_used)]
pub fn stress_words() -> Vec<u32> {
    let mut bytes = assemble().finalize().unwrap();
    bytes.resize(bytes.len().next_multiple_of(0x1000), 0);
    let (words, _) = bytes.as_chunks::<4>();
    words.iter().map(|word| u32::from_le_bytes(*word)).collect()
}

/// The programs, assembled (labels still resolvable).
fn assemble() -> Assembler {
    let mut ops = Assembler::new(0);
    dynasm!(ops
        ; .arch aarch64
        ; ->claim_cas:
        ; mov x10, 0
        ; add x12, x0, 1, lsl 12
        ; l1:
        ; ldr x4, [x0]
        ; l2:
        ; cmp x4, x3
        ; b.hs >l9
        ; add x5, x4, 1
        ; mov x6, x4
        ; casal x6, x5, [x0]
        ; cmp x6, x4
        ; b.eq >l3
        ; mov x4, x6
        ; yield
        ; b <l2
        ; l3:
        ; add x9, x12, x4, lsl 2
        ; ldr w8, [x9]
        ; add w8, w8, 1
        ; str w8, [x9]
        ; add x10, x10, 1
        ; b <l1
        ; l9:
        ; svc 7
        ; udf 0

    );
    align(&mut ops, 1024);
    dynasm!(ops
        ; .arch aarch64
        // As claim_cas, with LDAXR/STLXR.
        ; ->claim_llsc:
        ; mov x10, 0
        ; add x12, x0, 1, lsl 12
        ; l1:
        ; ldaxr x4, [x0]
        ; cmp x4, x3
        ; b.hs >l8
        ; add x5, x4, 1
        ; stlxr w6, x5, [x0]
        ; cbnz w6, <l1
        ; add x9, x12, x4, lsl 2
        ; ldr w8, [x9]
        ; add w8, w8, 1
        ; str w8, [x9]
        ; add x10, x10, 1
        ; b <l1
        ; l8:
        ; clrex
        ; svc 7
        ; udf 0

    );
    align(&mut ops, 1024);
    dynasm!(ops
        ; .arch aarch64
        // As claim_cas, with LDADDAL handing out the indices.
        ; ->claim_ldadd:
        ; mov x10, 0
        ; add x12, x0, 1, lsl 12
        ; mov x5, 1
        ; l1:
        ; ldaddal x5, x4, [x0]
        ; cmp x4, x3
        ; b.hs >l9
        ; add x9, x12, x4, lsl 2
        ; ldr w8, [x9]
        ; add w8, w8, 1
        ; str w8, [x9]
        ; add x10, x10, 1
        ; b <l1
        ; l9:
        ; svc 7
        ; udf 0

    );
    align(&mut ops, 1024);
    dynasm!(ops
        ; .arch aarch64
        // A ticket lock: take a ticket with LDADD, wait (SEVL, WFE, LDAR) until
        // it is served, then add one to a pair of plain words, which must stay
        // equal, and serve the next ticket with STLR and SEV. x10 = times the
        // pair was seen unequal inside the lock.
        ; ->ticket:
        ; mov x10, 0
        ; add x12, x0, 0x40
        ; add x13, x0, 0x80
        ; mov w7, 1
        ; l1:
        ; cbz x3, >l9
        ; ldadd w7, w4, [x0]
        ; sevl
        ; l2:
        ; wfe
        ; ldar w5, [x12]
        ; cmp w5, w4
        ; b.ne <l2
        ; ldr x8, [x13]
        ; ldr x9, [x13, 8]
        ; cmp x8, x9
        ; cinc x10, x10, ne
        ; add x8, x8, 1
        ; str x8, [x13]
        ; str x8, [x13, 8]
        ; add w5, w5, 1
        ; stlr w5, [x12]
        ; sev
        ; sub x3, x3, 1
        ; b <l1
        ; l9:
        ; svc 7
        ; udf 0

    );
    align(&mut ops, 1024);
    dynasm!(ops
        ; .arch aarch64
        // A seqlock. Thread 0 writes x3 values (i, !i) under the sequence count
        // at x0, then sets x0 + 0x80. The others read until that is set and the
        // last read is done; x10 = torn reads (a pair that is not (i, !i)),
        // x11 = consistent reads.
        ; ->seqlock:
        ; add x12, x0, 0x40
        ; add x13, x0, 0x80
        ; cbnz x1, >l5
        ; mov x4, 0
        ; l1:
        ; cmp x4, x3
        ; b.hs >l4
        ; ldr x5, [x0]
        ; add x5, x5, 1
        ; str x5, [x0]
        ; dmb ish
        ; add x4, x4, 1
        ; str x4, [x12]
        ; mvn x6, x4
        ; str x6, [x12, 8]
        ; dmb ish
        ; add x5, x5, 1
        ; str x5, [x0]
        ; b <l1
        ; l4:
        ; mov x5, 1
        ; stlr x5, [x13]
        ; svc 7
        ; l5:
        ; mov x10, 0
        ; mov x11, 0
        ; l6:
        ; ldar x9, [x13]
        ; l7:
        ; ldar x5, [x0]
        // Odd: a write in progress; zero: nothing written yet (the pair is
        // not (i, !i) before the first write).
        ; tbnz x5, 0, >l71
        ; cbnz x5, >l8
        ; l71:
        ; yield
        ; b <l7
        ; l8:
        ; ldr x6, [x12]
        ; ldr x7, [x12, 8]
        ; dmb ishld
        ; ldr x8, [x0]
        ; cmp x8, x5
        ; b.ne <l6
        ; mvn x7, x7
        ; cmp x6, x7
        ; cinc x10, x10, ne
        ; add x11, x11, 1
        ; cbz x9, <l6
        ; svc 7
        ; udf 0

    );
    align(&mut ops, 1024);
    dynasm!(ops
        ; .arch aarch64
        // Message passing with STLR and LDAR. Thread 0 sends x3 messages: plain
        // stores of (k, 3k) to x0 + 0x40, then STLR k to x0; it waits for the
        // acknowledgement at x0 + 0x80. Thread 1 waits with LDAR, reads the pair
        // with plain loads and acknowledges with STLR. x10 = messages seen wrong.
        ; ->pass_acquire:
        ; add x12, x0, 0x40
        ; add x13, x0, 0x80
        ; mov x10, 0
        ; mov x4, 0
        ; cbnz x1, >l5
        ; l1:
        ; cmp x4, x3
        ; b.hs >l9
        ; add x4, x4, 1
        ; add x6, x4, x4, lsl 1
        ; str x4, [x12]
        ; str x6, [x12, 8]
        ; stlr x4, [x0]
        ; sevl
        ; l2:
        ; wfe
        ; ldar x5, [x13]
        ; cmp x5, x4
        ; b.ne <l2
        ; b <l1
        ; l5:
        ; cmp x4, x3
        ; b.hs >l9
        ; add x4, x4, 1
        ; sevl
        ; l6:
        ; wfe
        ; ldar x5, [x0]
        ; cmp x5, x4
        ; b.ne <l6
        ; ldr x6, [x12]
        ; ldr x7, [x12, 8]
        ; add x8, x4, x4, lsl 1
        ; cmp x6, x4
        ; ccmp x7, x8, 0, eq
        ; cinc x10, x10, ne
        ; stlr x4, [x13]
        ; sev
        ; b <l5
        ; l9:
        ; svc 7
        ; udf 0

    );
    align(&mut ops, 1024);
    dynasm!(ops
        ; .arch aarch64
        // As pass_acquire with plain stores and loads ordered by DMB ISH
        // (store data, DMB, store flag; load flag, DMB, load data).
        ; ->pass_dmb:
        ; add x12, x0, 0x40
        ; add x13, x0, 0x80
        ; mov x10, 0
        ; mov x4, 0
        ; cbnz x1, >l5
        ; l1:
        ; cmp x4, x3
        ; b.hs >l9
        ; add x4, x4, 1
        ; add x6, x4, x4, lsl 1
        ; str x4, [x12]
        ; str x6, [x12, 8]
        ; dmb ish
        ; str x4, [x0]
        ; l2:
        ; ldr x5, [x13]
        ; cmp x5, x4
        ; b.ne <l2
        ; dmb ish
        ; b <l1
        ; l5:
        ; cmp x4, x3
        ; b.hs >l9
        ; add x4, x4, 1
        ; l6:
        ; ldr x5, [x0]
        ; cmp x5, x4
        ; b.ne <l6
        ; dmb ish
        ; ldr x6, [x12]
        ; ldr x7, [x12, 8]
        ; add x8, x4, x4, lsl 1
        ; cmp x6, x4
        ; ccmp x7, x8, 0, eq
        ; cinc x10, x10, ne
        ; dmb ish
        ; str x4, [x13]
        ; b <l5
        ; l9:
        ; svc 7
        ; udf 0

    );
    align(&mut ops, 1024);
    dynasm!(ops
        ; .arch aarch64
        // A Treiber stack of nodes (x0 + 0x1000 + 64 * i: next at 0, busy at
        // 8) under a {pointer, tag} head at x0, pushed and popped with CASPAL;
        // the tag makes it ABA-safe. Each thread pops a node, marks it busy
        // with LDADDAL (it must have been 0), clears the mark and pushes it
        // back, x3 times. x10 = nodes seen busy when popped.
        ; ->casp_stack:
        ; mov x10, 0
        ; mov x12, 1
        ; movn x14, 0
        ; l1:
        ; cbz x3, >l9
        ; l2:
        ; ldp x4, x5, [x0]
        ; cbnz x4, >l3
        ; yield
        ; b <l2
        ; l3:
        ; ldr x6, [x4]
        ; add x7, x5, 1
        ; mov x8, x4
        ; mov x9, x5
        ; caspal x8, x9, x6, x7, [x0]
        ; cmp x8, x4
        ; ccmp x9, x5, 0, eq
        ; b.ne <l2
        ; add x11, x4, 8
        ; ldaddal x12, x13, [x11]
        ; cbz x13, >l4
        ; add x10, x10, 1
        ; l4:
        ; ldaddal x14, x13, [x11]
        ; l5:
        ; ldp x8, x9, [x0]
        ; str x8, [x4]
        ; mov x16, x8
        ; mov x17, x9
        ; mov x6, x4
        ; add x7, x9, 1
        ; caspal x8, x9, x6, x7, [x0]
        ; cmp x8, x16
        ; ccmp x9, x17, 0, eq
        ; b.ne <l5
        ; sub x3, x3, 1
        ; b <l1
        ; l9:
        ; svc 7
        ; udf 0

    );
    align(&mut ops, 1024);
    dynasm!(ops
        ; .arch aarch64
        // 64-bit halves of pair stores. Thread 0 stores x3 pairs with STP to
        // x0 + 0x40: (k:k, !k:!k), each half two equal 32-bit words. The others
        // read with LDP until thread 0 sets x0 + 0x80, and count halves whose
        // 32-bit words differ (a torn 64-bit half). x10 = torn halves, x11 =
        // reads.
        ; ->pair_halves:
        ; add x12, x0, 0x40
        ; add x13, x0, 0x80
        ; cbnz x1, >l5
        ; mov x4, 0
        ; l1:
        ; cmp x4, x3
        ; b.hs >l4
        ; add x4, x4, 1
        ; orr x5, x4, x4, lsl 32
        ; mvn x6, x5
        ; stp x5, x6, [x12]
        ; b <l1
        ; l4:
        ; mov x5, 1
        ; stlr x5, [x13]
        ; svc 7
        ; l5:
        ; mov x10, 0
        ; mov x11, 0
        ; l6:
        ; ldar x9, [x13]
        ; ldp x5, x6, [x12]
        ; eor x7, x5, x5, ror 32
        ; cmp x7, 0
        ; cinc x10, x10, ne
        ; eor x7, x6, x6, ror 32
        ; cmp x7, 0
        ; cinc x10, x10, ne
        ; add x11, x11, 1
        ; cbz x9, <l6
        ; svc 7
        ; udf 0

    );
    align(&mut ops, 1024);
    dynasm!(ops
        ; .arch aarch64
        // As pair_halves with a Q register: STR Q / LDR Q, whose two 64-bit
        // halves are each single-copy atomic.
        ; ->vector_halves:
        ; add x12, x0, 0x40
        ; add x13, x0, 0x80
        ; cbnz x1, >l5
        ; mov x4, 0
        ; l1:
        ; cmp x4, x3
        ; b.hs >l4
        ; add x4, x4, 1
        ; orr x5, x4, x4, lsl 32
        ; mvn x6, x5
        ; fmov d0, x5
        ; mov v0.d[1], x6
        ; str q0, [x12]
        ; b <l1
        ; l4:
        ; mov x5, 1
        ; stlr x5, [x13]
        ; svc 7
        ; l5:
        ; mov x10, 0
        ; mov x11, 0
        ; l6:
        ; ldar x9, [x13]
        ; ldr q1, [x12]
        ; fmov x5, d1
        ; mov x6, v1.d[1]
        ; eor x7, x5, x5, ror 32
        ; cmp x7, 0
        ; cinc x10, x10, ne
        ; eor x7, x6, x6, ror 32
        ; cmp x7, 0
        ; cinc x10, x10, ne
        ; add x11, x11, 1
        ; cbz x9, <l6
        ; svc 7
        ; udf 0
    );

    ops
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use dynasmrt::components::StaticLabel;

    #[test]
    fn the_entry_table_matches_the_labels() {
        let ops = assemble();
        for &(name, offset) in STRESS_ENTRIES {
            let label = ops.labels().resolve_static(&StaticLabel::global(name)).unwrap();
            assert_eq!(label.0 as u64, offset, "{name}");
        }
    }
}
