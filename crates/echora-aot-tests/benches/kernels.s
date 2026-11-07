// Benchmark kernels for `benches/kernels.rs`. Original code, never copied
// from a game. `assemble_kernels.py` assembles this file with llvm-mc and
// checks the words in as `src/kernels.rs`, so building needs no assembler.
// Each kernel starts its own 1 KiB window and ends with SVC #7, followed by
// a UDF so that discovery does not run on past it.

.text
// x0 = iterations. Integer arithmetic, flags and a loop branch.
arith:
    mov x1, #0
    mov x2, #1
1:  add x1, x1, x2
    eor x2, x2, x1, lsl #3
    madd x3, x1, x2, x3
    ror x4, x3, #13
    add x5, x5, x4, lsr #7
    subs x0, x0, #1
    b.ne 1b
    svc #7
    udf #0

.p2align 10
// x0 = repetitions, x1 = destination, x2 = source, x3 = bytes (64n).
memcpy:
1:  mov x6, x1
    mov x7, x2
    mov x8, x3
2:  ldp q0, q1, [x7], #32
    ldp q2, q3, [x7], #32
    stp q0, q1, [x6], #32
    stp q2, q3, [x6], #32
    subs x8, x8, #64
    b.ne 2b
    subs x0, x0, #1
    b.ne 1b
    svc #7
    udf #0

.p2align 10
// x0 = repetitions, x1 = destination, x3 = bytes (16n), x4 = value.
memset:
1:  mov x6, x1
    mov x8, x3
2:  stp x4, x4, [x6], #16
    subs x8, x8, #16
    b.ne 2b
    subs x0, x0, #1
    b.ne 1b
    svc #7
    udf #0

.p2align 10
// x0 = iterations. Two BLs per iteration to leaf functions in the next
// window.
calls:
1:  bl leaf_add
    bl leaf_mix
    subs x0, x0, #1
    b.ne 1b
    svc #7
    udf #0

.p2align 10
leaf_add:
    add x1, x1, #1
    ret
leaf_mix:
    eor x2, x2, x1, lsl #1
    add x2, x2, #3
    ret

.p2align 10
// x0 = iterations, x1 = seed. An eight-way switch through a halfword jump
// table, bounded by masking the index.
switch:
1:  and w17, w1, #7
    adr x16, table
    ldrh w9, [x16, x17, lsl #1]
    adr x3, cases
    add x3, x3, x9, lsl #2
    br x3
cases:
    add x2, x2, #1
    b 2f
    add x2, x2, #2
    b 2f
    eor x2, x2, #3
    b 2f
    sub x2, x2, #4
    b 2f
    lsl x2, x2, #1
    b 2f
    add x2, x2, x1
    b 2f
    eor x2, x2, x1
    b 2f
    orr x2, x2, #0x10
    b 2f
2:  add x1, x1, #5
    subs x0, x0, #1
    b.ne 1b
    svc #7
    udf #0
.p2align 3
table:
    .hword 0, 2, 4, 6, 8, 10, 12, 14

.p2align 10
// x0 = repetitions, x1 = a, x2 = b, x3 = 16-byte elements (nonzero).
vector:
1:  mov x6, x1
    mov x7, x2
    mov x8, x3
2:  ldr q0, [x6], #16
    ldr q1, [x7], #16
    fmla v2.4s, v0.4s, v1.4s
    fadd v3.4s, v3.4s, v1.4s
    fmul v4.4s, v0.4s, v0.4s
    subs x8, x8, #1
    b.ne 2b
    subs x0, x0, #1
    b.ne 1b
    svc #7
    udf #0

.p2align 10
// x0 = address of a word another thread sets nonzero.
poll:
1:  ldar w8, [x0]
    cbz w8, 1b
    svc #7
    udf #0
