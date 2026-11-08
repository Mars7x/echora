// Concurrency stress programs for `tests/stress.rs`: the patterns a job
// system uses, run on several Cores sharing one Memory. Original code.
// `assemble_stress.py` assembles this file and checks the words in as
// `src/stress.rs`, so building needs no assembler.
//
// On entry: x0 = shared data (page aligned, zeroed), x1 = this thread's
// index, x2 = thread count, x3 = iterations (or items). Each ends with
// SVC #7; x10 and x11 hold what the harness checks.

.text
// Claims items 0..x3 by advancing a shared index with CASAL, and marks
// each claimed item with a plain load and store at x0 + 0x1000 + 4 * i.
// x10 = items claimed. A doubly claimed item shows as a count other than 1.
claim_cas:
    mov x10, #0
    add x12, x0, #0x1000
1:  ldr x4, [x0]
2:  cmp x4, x3
    b.hs 9f
    add x5, x4, #1
    mov x6, x4
    casal x6, x5, [x0]
    cmp x6, x4
    b.eq 3f
    mov x4, x6
    yield
    b 2b
3:  add x9, x12, x4, lsl #2
    ldr w8, [x9]
    add w8, w8, #1
    str w8, [x9]
    add x10, x10, #1
    b 1b
9:  svc #7
    udf #0

.p2align 10
// As claim_cas, with LDAXR/STLXR.
claim_llsc:
    mov x10, #0
    add x12, x0, #0x1000
1:  ldaxr x4, [x0]
    cmp x4, x3
    b.hs 8f
    add x5, x4, #1
    stlxr w6, x5, [x0]
    cbnz w6, 1b
    add x9, x12, x4, lsl #2
    ldr w8, [x9]
    add w8, w8, #1
    str w8, [x9]
    add x10, x10, #1
    b 1b
8:  clrex
    svc #7
    udf #0

.p2align 10
// As claim_cas, with LDADDAL handing out the indices.
claim_ldadd:
    mov x10, #0
    add x12, x0, #0x1000
    mov x5, #1
1:  ldaddal x5, x4, [x0]
    cmp x4, x3
    b.hs 9f
    add x9, x12, x4, lsl #2
    ldr w8, [x9]
    add w8, w8, #1
    str w8, [x9]
    add x10, x10, #1
    b 1b
9:  svc #7
    udf #0

.p2align 10
// A ticket lock: take a ticket with LDADD, wait (SEVL, WFE, LDAR) until
// it is served, then add one to a pair of plain words, which must stay
// equal, and serve the next ticket with STLR and SEV. x10 = times the
// pair was seen unequal inside the lock.
ticket:
    mov x10, #0
    add x12, x0, #0x40
    add x13, x0, #0x80
    mov w7, #1
1:  cbz x3, 9f
    ldadd w7, w4, [x0]
    sevl
2:  wfe
    ldar w5, [x12]
    cmp w5, w4
    b.ne 2b
    ldr x8, [x13]
    ldr x9, [x13, #8]
    cmp x8, x9
    cinc x10, x10, ne
    add x8, x8, #1
    str x8, [x13]
    str x8, [x13, #8]
    add w5, w5, #1
    stlr w5, [x12]
    sev
    sub x3, x3, #1
    b 1b
9:  svc #7
    udf #0

.p2align 10
// A seqlock. Thread 0 writes x3 values (i, !i) under the sequence count
// at x0, then sets x0 + 0x80. The others read until that is set and the
// last read is done; x10 = torn reads (a pair that is not (i, !i)),
// x11 = consistent reads.
seqlock:
    add x12, x0, #0x40
    add x13, x0, #0x80
    cbnz x1, 5f
    mov x4, #0
1:  cmp x4, x3
    b.hs 4f
    ldr x5, [x0]
    add x5, x5, #1
    str x5, [x0]
    dmb ish
    add x4, x4, #1
    str x4, [x12]
    mvn x6, x4
    str x6, [x12, #8]
    dmb ish
    add x5, x5, #1
    str x5, [x0]
    b 1b
4:  mov x5, #1
    stlr x5, [x13]
    svc #7
5:  mov x10, #0
    mov x11, #0
6:  ldar x9, [x13]
7:  ldar x5, [x0]
    // Odd: a write in progress; zero: nothing written yet (the pair is
    // not (i, !i) before the first write).
    tbnz x5, #0, 71f
    cbnz x5, 8f
71: yield
    b 7b
8:  ldr x6, [x12]
    ldr x7, [x12, #8]
    dmb ishld
    ldr x8, [x0]
    cmp x8, x5
    b.ne 6b
    mvn x7, x7
    cmp x6, x7
    cinc x10, x10, ne
    add x11, x11, #1
    cbz x9, 6b
    svc #7
    udf #0

.p2align 10
// Message passing with STLR and LDAR. Thread 0 sends x3 messages: plain
// stores of (k, 3k) to x0 + 0x40, then STLR k to x0; it waits for the
// acknowledgement at x0 + 0x80. Thread 1 waits with LDAR, reads the pair
// with plain loads and acknowledges with STLR. x10 = messages seen wrong.
pass_acquire:
    add x12, x0, #0x40
    add x13, x0, #0x80
    mov x10, #0
    mov x4, #0
    cbnz x1, 5f
1:  cmp x4, x3
    b.hs 9f
    add x4, x4, #1
    add x6, x4, x4, lsl #1
    str x4, [x12]
    str x6, [x12, #8]
    stlr x4, [x0]
    sevl
2:  wfe
    ldar x5, [x13]
    cmp x5, x4
    b.ne 2b
    b 1b
5:  cmp x4, x3
    b.hs 9f
    add x4, x4, #1
    sevl
6:  wfe
    ldar x5, [x0]
    cmp x5, x4
    b.ne 6b
    ldr x6, [x12]
    ldr x7, [x12, #8]
    add x8, x4, x4, lsl #1
    cmp x6, x4
    ccmp x7, x8, #0, eq
    cinc x10, x10, ne
    stlr x4, [x13]
    sev
    b 5b
9:  svc #7
    udf #0

.p2align 10
// As pass_acquire with plain stores and loads ordered by DMB ISH
// (store data, DMB, store flag; load flag, DMB, load data).
pass_dmb:
    add x12, x0, #0x40
    add x13, x0, #0x80
    mov x10, #0
    mov x4, #0
    cbnz x1, 5f
1:  cmp x4, x3
    b.hs 9f
    add x4, x4, #1
    add x6, x4, x4, lsl #1
    str x4, [x12]
    str x6, [x12, #8]
    dmb ish
    str x4, [x0]
2:  ldr x5, [x13]
    cmp x5, x4
    b.ne 2b
    dmb ish
    b 1b
5:  cmp x4, x3
    b.hs 9f
    add x4, x4, #1
6:  ldr x5, [x0]
    cmp x5, x4
    b.ne 6b
    dmb ish
    ldr x6, [x12]
    ldr x7, [x12, #8]
    add x8, x4, x4, lsl #1
    cmp x6, x4
    ccmp x7, x8, #0, eq
    cinc x10, x10, ne
    dmb ish
    str x4, [x13]
    b 5b
9:  svc #7
    udf #0

.p2align 10
// A Treiber stack of nodes (x0 + 0x1000 + 64 * i: next at 0, busy at
// 8) under a {pointer, tag} head at x0, pushed and popped with CASPAL;
// the tag makes it ABA-safe. Each thread pops a node, marks it busy
// with LDADDAL (it must have been 0), clears the mark and pushes it
// back, x3 times. x10 = nodes seen busy when popped.
casp_stack:
    mov x10, #0
    mov x12, #1
    mov x14, #-1
1:  cbz x3, 9f
2:  ldp x4, x5, [x0]
    cbnz x4, 3f
    yield
    b 2b
3:  ldr x6, [x4]
    add x7, x5, #1
    mov x8, x4
    mov x9, x5
    caspal x8, x9, x6, x7, [x0]
    cmp x8, x4
    ccmp x9, x5, #0, eq
    b.ne 2b
    add x11, x4, #8
    ldaddal x12, x13, [x11]
    cbz x13, 4f
    add x10, x10, #1
4:  ldaddal x14, x13, [x11]
5:  ldp x8, x9, [x0]
    str x8, [x4]
    mov x16, x8
    mov x17, x9
    mov x6, x4
    add x7, x9, #1
    caspal x8, x9, x6, x7, [x0]
    cmp x8, x16
    ccmp x9, x17, #0, eq
    b.ne 5b
    sub x3, x3, #1
    b 1b
9:  svc #7
    udf #0

.p2align 10
// 64-bit halves of pair stores. Thread 0 stores x3 pairs with STP to
// x0 + 0x40: (k:k, !k:!k), each half two equal 32-bit words. The others
// read with LDP until thread 0 sets x0 + 0x80, and count halves whose
// 32-bit words differ (a torn 64-bit half). x10 = torn halves, x11 =
// reads.
pair_halves:
    add x12, x0, #0x40
    add x13, x0, #0x80
    cbnz x1, 5f
    mov x4, #0
1:  cmp x4, x3
    b.hs 4f
    add x4, x4, #1
    orr x5, x4, x4, lsl #32
    mvn x6, x5
    stp x5, x6, [x12]
    b 1b
4:  mov x5, #1
    stlr x5, [x13]
    svc #7
5:  mov x10, #0
    mov x11, #0
6:  ldar x9, [x13]
    ldp x5, x6, [x12]
    eor x7, x5, x5, ror #32
    cmp x7, #0
    cinc x10, x10, ne
    eor x7, x6, x6, ror #32
    cmp x7, #0
    cinc x10, x10, ne
    add x11, x11, #1
    cbz x9, 6b
    svc #7
    udf #0

.p2align 10
// As pair_halves with a Q register: STR Q / LDR Q, whose two 64-bit
// halves are each single-copy atomic.
vector_halves:
    add x12, x0, #0x40
    add x13, x0, #0x80
    cbnz x1, 5f
    mov x4, #0
1:  cmp x4, x3
    b.hs 4f
    add x4, x4, #1
    orr x5, x4, x4, lsl #32
    mvn x6, x5
    fmov d0, x5
    mov v0.d[1], x6
    str q0, [x12]
    b 1b
4:  mov x5, #1
    stlr x5, [x13]
    svc #7
5:  mov x10, #0
    mov x11, #0
6:  ldar x9, [x13]
    ldr q1, [x12]
    fmov x5, d1
    mov x6, v1.d[1]
    eor x7, x5, x5, ror #32
    cmp x7, #0
    cinc x10, x10, ne
    eor x7, x6, x6, ror #32
    cmp x7, #0
    cinc x10, x10, ne
    add x11, x11, #1
    cbz x9, 6b
    svc #7
    udf #0
