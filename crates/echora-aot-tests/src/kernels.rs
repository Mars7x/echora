// The benchmark kernels, as AArch64 code assembled by dynasm-rs:
// shared by `build.rs`, which lifts them, and the benchmarks and tests,
// which run them. Original code, never copied from a game. Each kernel
// starts its own 1 KiB window and ends with SVC #7, followed by a UDF so
// that discovery does not run on past it.

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

/// Where the kernel image is loaded: page and window aligned.
pub const KERNEL_BASE: u64 = 0x10_0000;

/// Every kernel by name, with its entry offset from `KERNEL_BASE`.
pub const KERNEL_ENTRIES: &[(&str, u64)] = &[
    ("arith", 0x0),
    ("memcpy", 0x400),
    ("memset", 0x800),
    ("calls", 0xc00),
    ("switch", 0x1400),
    ("vector", 0x1800),
    ("poll", 0x1c00),
    ("scalar", 0x2000),
    ("ldst", 0x2400),
];

/// Bytes of the image: whole 1 KiB windows.
pub const KERNEL_SIZE: u64 = 0x2800;

/// Where the leaf functions `calls` branches to start.
const LEAVES: u64 = 0x1000;

/// The kernels' code as assembled, NOP padded between kernels.
#[allow(clippy::unwrap_used)]
fn code() -> Vec<u8> {
    assemble().finalize().unwrap()
}

/// The kernels, assembled (labels still resolvable).
fn assemble() -> Assembler {
    let mut ops = Assembler::new(0);
    dynasm!(ops
        ; .arch aarch64
        ; ->arith:
        ; mov x1, 0
        ; mov x2, 1
        ; l1:
        ; add x1, x1, x2
        ; eor x2, x2, x1, lsl 3
        ; madd x3, x1, x2, x3
        ; ror x4, x3, 13
        ; add x5, x5, x4, lsr 7
        ; subs x0, x0, 1
        ; b.ne <l1
        ; svc 7
        ; udf 0

    );
    align(&mut ops, 1024);
    dynasm!(ops
        ; .arch aarch64
        // x0 = repetitions, x1 = destination, x2 = source, x3 = bytes (64n).
        ; ->memcpy:
        ; l1:
        ; mov x6, x1
        ; mov x7, x2
        ; mov x8, x3
        ; l2:
        ; ldp q0, q1, [x7], 32
        ; ldp q2, q3, [x7], 32
        ; stp q0, q1, [x6], 32
        ; stp q2, q3, [x6], 32
        ; subs x8, x8, 64
        ; b.ne <l2
        ; subs x0, x0, 1
        ; b.ne <l1
        ; svc 7
        ; udf 0

    );
    align(&mut ops, 1024);
    dynasm!(ops
        ; .arch aarch64
        // x0 = repetitions, x1 = destination, x3 = bytes (16n), x4 = value.
        ; ->memset:
        ; l1:
        ; mov x6, x1
        ; mov x8, x3
        ; l2:
        ; stp x4, x4, [x6], 16
        ; subs x8, x8, 16
        ; b.ne <l2
        ; subs x0, x0, 1
        ; b.ne <l1
        ; svc 7
        ; udf 0

    );
    align(&mut ops, 1024);
    dynasm!(ops
        ; .arch aarch64
        // x0 = iterations. Two BLs per iteration to leaf functions in the next
        // window.
        ; ->calls:
        ; l1:
        ; bl ->leaf_add
        ; bl ->leaf_mix
        ; subs x0, x0, 1
        ; b.ne <l1
        ; svc 7
        ; udf 0

    );
    align(&mut ops, 1024);
    dynasm!(ops
        ; .arch aarch64
        ; ->leaf_add:
        ; add x1, x1, 1
        ; ret
        ; ->leaf_mix:
        ; eor x2, x2, x1, lsl 1
        ; add x2, x2, 3
        ; ret

    );
    align(&mut ops, 1024);
    dynasm!(ops
        ; .arch aarch64
        // x0 = iterations, x1 = seed. An eight-way switch through a halfword jump
        // table, bounded by masking the index.
        ; ->switch:
        ; l1:
        ; and w17, w1, 7
        ; adr x16, ->table
        ; ldrh w9, [x16, x17, lsl 1]
        ; adr x3, ->cases
        ; add x3, x3, x9, lsl 2
        ; br x3
        ; ->cases:
        ; add x2, x2, 1
        ; b >l2
        ; add x2, x2, 2
        ; b >l2
        ; eor x2, x2, 3
        ; b >l2
        ; sub x2, x2, 4
        ; b >l2
        ; lsl x2, x2, 1
        ; b >l2
        ; add x2, x2, x1
        ; b >l2
        ; eor x2, x2, x1
        ; b >l2
        ; orr x2, x2, 0x10
        ; b >l2
        ; l2:
        ; add x1, x1, 5
        ; subs x0, x0, 1
        ; b.ne <l1
        ; svc 7
        ; udf 0
    );
    align(&mut ops, 8);
    dynasm!(ops
        ; .arch aarch64
        ; ->table:
        ; .u16 0, 2, 4, 6, 8, 10, 12, 14

    );
    align(&mut ops, 1024);
    dynasm!(ops
        ; .arch aarch64
        // x0 = repetitions, x1 = a, x2 = b, x3 = 16-byte elements (nonzero).
        ; ->vector:
        ; l1:
        ; mov x6, x1
        ; mov x7, x2
        ; mov x8, x3
        ; l2:
        ; ldr q0, [x6], 16
        ; ldr q1, [x7], 16
        ; fmla v2.s4, v0.s4, v1.s4
        ; fadd v3.s4, v3.s4, v1.s4
        ; fmul v4.s4, v0.s4, v0.s4
        ; subs x8, x8, 1
        ; b.ne <l2
        ; subs x0, x0, 1
        ; b.ne <l1
        ; svc 7
        ; udf 0

    );
    align(&mut ops, 1024);
    dynasm!(ops
        ; .arch aarch64
        // x0 = address of a word another thread sets nonzero.
        ; ->poll:
        ; l1:
        ; ldar w8, [x0]
        ; cbz w8, <l1
        ; svc 7
        ; udf 0

    );
    align(&mut ops, 1024);
    dynasm!(ops
        ; .arch aarch64
        // x0 = repetitions, x1 = source, x2 = destination, x3 = floats (n).
        // Scalar single precision as games use it: loads, a multiply, a fused
        // multiply-add, a compare and a select, and a store.
        ; ->scalar:
        ; l1:
        ; mov x6, x1
        ; mov x7, x2
        ; mov x8, x3
        ; fmov s2, 1.0
        ; l2:
        ; ldr s0, [x6], 4
        ; fmul s1, s0, s0
        ; fmadd s2, s0, s1, s2
        ; fcmp s2, s0
        ; fcsel s3, s2, s0, gt
        ; str s3, [x7], 4
        ; subs x8, x8, 1
        ; b.ne <l2
        ; subs x0, x0, 1
        ; b.ne <l1
        ; svc 7
        ; udf 0

    );
    align(&mut ops, 1024);
    dynasm!(ops
        ; .arch aarch64
        // Integer loads and stores as data structures use them: a load, an add and
        // a store per word. x0 = repetitions, x1 = source, x2 = destination,
        // x3 = words.
        ; ->ldst:
        ; l1:
        ; mov x6, x1
        ; mov x7, x2
        ; mov x8, x3
        ; l2:
        ; ldr x9, [x6], 8
        ; add x10, x10, x9
        ; str x10, [x7], 8
        ; subs x8, x8, 1
        ; b.ne <l2
        ; subs x0, x0, 1
        ; b.ne <l1
        ; svc 7
        ; udf 0
    );
    ops
}

/// The image as bytes from `KERNEL_BASE`: each kernel (and the leaf
/// functions) without its trailing padding, zeros (UDF) in the gaps.
pub fn image() -> Vec<u8> {
    let code = code();
    let mut starts: Vec<u64> = KERNEL_ENTRIES.iter().map(|&(_, offset)| offset).collect();
    starts.push(LEAVES);
    starts.sort_unstable();
    let mut bytes = vec![0; KERNEL_SIZE as usize];
    for (index, &start) in starts.iter().enumerate() {
        let end = starts.get(index + 1).map_or(code.len(), |&next| next as usize);
        let mut run = &code[start as usize..end];
        while let Some((rest, last)) = run.split_last_chunk::<4>() {
            let word = u32::from_le_bytes(*last);
            if word != 0 && word != NOP {
                break;
            }
            run = rest;
        }
        bytes[start as usize..start as usize + run.len()].copy_from_slice(run);
    }
    bytes
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use dynasmrt::components::StaticLabel;

    #[test]
    fn the_entry_table_matches_the_labels() {
        let ops = assemble();
        for &(name, offset) in KERNEL_ENTRIES.iter().chain(&[("leaf_add", LEAVES)]) {
            let label = ops.labels().resolve_static(&StaticLabel::global(name)).unwrap();
            assert_eq!(label.0 as u64, offset, "{name}");
        }
        assert!(code().len() as u64 <= KERNEL_SIZE);
    }
}
