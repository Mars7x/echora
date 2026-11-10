# Shader translator (`echora-shader`)

The game's GPU programs are NVN2 programs: NVIDIA Ampere (SM 8.x, the
Switch 2's T239) machine code with a control section. `echora-shader` is
Echora's own translator, in Rust. Until it covers every program, an
external translator plugin loaded at run time handles the rest
([shader-plugin.md](shader-plugin.md)): a program the Rust translator
handles uses its SPIR-V, anything else goes to the plugin, and a
differential mode compares the two.

## Design stance

- **Our own architecture.** References are used for facts only (instruction
  encodings and exact semantics), never for structure or code: Lichii's
  Ampere front end (GPL-3.0-or-later, the main reference, because it already
  handles the forms the game uses) and Mesa's NAK (MIT, for clean bit
  layouts). Both are credited in NOTICE.md and docs/research.md and in the
  headers of files that use their facts.
- **Ampere straight into our IR.** No lowering to Maxwell first.
- **A GPU page table for global memory** (guest GPU virtual address to
  device address, set up by the runtime and renderer), in place of tracking
  where storage buffers are: no heuristic search of driver tables.
- **Safety built in**: one shared per-invocation loop budget for every loop,
  and memory accesses outside mapped pages reading zero and dropping writes,
  rather than hangs or device loss.

## Ahead-of-time translation

All of a game's programs ship in its RomFS (about 10,000 for the target
game). The plan is to translate them all at the first start and fill the
driver's pipeline cache, so nothing compiles during play. That makes three
goals first-class, next to correctness:

1. **No runtime state in a translation** wherever possible. Storage-image
   integer-ness, texture view types, render-target formats, vertex input
   types and the resolution scale are runtime inputs (specialization
   constants, push constants, or paths chosen by descriptors), not baked-in
   variants. The page table removes buffer tracking entirely; texture and
   image handles index descriptor arrays at run time.
2. **Explicit, enumerable variants.** Where a variant cannot be avoided, the
   translation reports it, and the runtime records the variants it meets so
   the next start precompiles them too.
3. **Throughput.** Translation is a pure function of the program and runs
   in parallel per program: fast over ~10,000 programs.

## Interface (agreed with the GPU side)

- `echora-shader-abi`, a small dependency-free crate, holds the shared
  types (translation result, descriptors, stages, the push-constant block
  and `ABI_VERSION`, which is part of every shader cache key). The plugin
  interface fills the same types.
- `echora_shader::translate(program, &Parameters) -> Result<Translated,
  Unsupported>`: pure, no callbacks. `Parameters` carries what the program
  does not (only the stage slot a task or mesh program is bound as);
  `Unsupported` names the first unhandled instruction or feature and its
  offset. `Translated` lists the specialization constants and push
  constants it reads.
- `ECHORA_SHADER_TRANSLATOR` selects `plugin` (the default with a plugin),
  `rust` (Rust where supported, the plugin otherwise; the default without
  one) or `compare`.
- **Page table**: 4 KiB pages (`PAGE_SHIFT` 12, the granularity of NVN
  memory pools, so no page spans two pools); a buffer of u64 device
  addresses (0 unmapped) for `[va_base, va_base + pages * 4 KiB)`; the
  table's address, `va_base >> 12` and the page count in bytes 112-128 of
  the push-constant block. A global access reads
  `table[(va >> 12) - base] + (va & 0xfff)`
  through PhysicalStorageBuffer64. Accesses are naturally aligned (a
  misaligned global access faults on the hardware) and at most 16 bytes, so
  none crosses a page: the first byte's page decides. An unmapped or
  out-of-range page reads 0 and drops writes.
- **Vertex pulling**: a vertex program fetches its generic inputs from
  guest memory itself, through the page table, so one translation serves
  every attribute format. Bytes 80-96 of the push-constant block
  (`VertexFetch`) hold the address of the draw's table of
  `VertexAttribute`s (one per input location: guest address, stride,
  instance divisor, format, readable size), the base instance and a mask
  of present entries. The format is decoded in the shader
  (`VertexFormat`: component count, 8/16/32-bit or packed layout, and
  normalized, integer, float or scaled numbers), bit-exact for integers.
  Index buffers stay fixed-function.

## Plan

1. **Decoder**: 128-bit words into typed instructions; unsupported forms are
   `Op::Unknown` with the raw bits, never a panic.
2. **IR** and lowering of the common ALU, memory and control forms.
3. **SPIR-V** for simple compute programs (a structurizer for the control
   flow; an emitter, `rspirv` or a small hand-written writer).
4. **Graphics stages** and textures.
5. **HMMA**: a subgroup path, VK_KHR_cooperative_matrix, and an exact tensor-core model.
6. **Task and mesh programs** (VK_EXT_mesh_shader), with the stage buffer.

**Textures.** A texture whose handle is a constant-bank word becomes a
combined image sampler, one per handle word, view type and depth
comparison, reported as a descriptor naming that word; the runtime binds
the texture and sampler the word names. Bindings follow the uniform
buffers. A bindless form (0xf60, 0xf63, 0xf66, 0xf6f: the handle in the
uniform register at bits 40-45) binds the same way when that register
only ever holds one constant-bank word (every write a ULDC of it); a
handle computed at run time indexes the texture pool instead: set 1, an
unsized array of sampled images per view type (bindings 1-7) and one of
samplers (binding 0), by the handle's low 20 bits and high 12, with
non-uniform indexing where the handle can differ across the warp (a
register; a uniform register's handle is the same for all lanes, so its
accesses are plain); `Translated::texture_pool_bindings` names the
bindings a program reads. A texture index at or past the pool's capacity
(the specialization constant `SpecMeaning::TexturePoolCapacity`, id 31,
by default 2^20, every index a handle carries) reads entry 0 and gives
zeros, as an unwritten entry does, so the arrays are never indexed past
their size. The samplers' 12-bit index needs 4096 entries. A 2D, 2D
array or 3D pool texture may be integer: bit 0 of its word in the
flags buffer (binding 14, one u32 per texture index) says so at run
time, and the read (fetch, sample or gather, not a depth comparison or
a per-tap gather) then goes through its uint view (bindings 11-13).
Texel buffers and storage images through the pool (bindings 8-10) are
not used yet: the decoder knows no buffer textures or bindless surfaces. TEX.B (0x361) has the handle among its values,
after the coordinates (all coordinates in Ra unless packed, bit 60, with
four values or fewer), and always reads the pool. Programs without derivatives sample at an explicit level (a
bias over level zero is that level). Vulkan takes only constant texel
offsets for a sample, so a sample adds the offsets divided by the
level's size to its coordinates; a fetch adds them to its integer
coordinates, and a gather passes them on.

**Graphics programs.** A graphics program is a 128-byte header (the
first 0x50 bytes the shader program header, SPH: its type at bits 0-4,
the stage at 10-13) and the code. ALD, AST and IPA address the attribute
space: 0x6c point size, 0x70-0x7c position (the fragment coordinate in a
fragment program, scaled back to the guest's resolution by the push
constants' inverse resolution scale), 0x80 + 16n generic vector n,
0x2f8 and 0x2fc the instance and vertex ids (without the draw's base),
0x3fc front facing (all ones or zero); writes to 0x5c (a per-primitive
hint) are dropped.
Generic vectors between stages are `vec4`s at location n; a fragment
input's interpolation comes from the SPH's input map (two bits per
component: constant, perspective, screen-linear), centroid from IPA.
Vertex inputs are fetched by the program from its vertex buffers through
the page table, by the draw's fetch table (`VertexFetch` at push constant
80, one `VertexAttribute` per location: address, stride, divisor, format
and size), so translations do not depend on attribute formats: a helper
function decodes any format at run time (8-, 16- and 32-bit components,
10/10/10/2 and 11/11/10 float; normalized, integer, float and scaled;
BGRA order), each word through its own page lookup and reading zero past
the stream's size, and the program's prologue fetches each location it
reads once. A
fragment program leaves its outputs in R0.. at exit: four registers per
render target the SPH enables (four component bits each at bits
576-607), then a register for the sample mask (bit 608) and the depth
(bit 609). KILL demotes the invocation to a helper. A graphics program's local memory size is the SPH's ShaderLocalMemoryLowSize (common word 1, bits 0-23); a nonzero high size (common word 2, bits 32 and up of the size) is refused. `stores` has bit
(address / 4) set for every attribute word written.

**Storage images.** A formatted image access (SULD.P, SUST.P) is typeless
(the bound image's format, no format in the SPIR-V). Whether the image is
integer is run-time state, so each image has a float and an unsigned view
at one binding, and a boolean specialization constant
(`SpecMeaning::IntegerFormat(binding)`, ids from 32) picks the view the
access uses; the runtime precompiles both values. SURED reduces into a texel
through a descriptor and binding of its own (even for a handle SULD or
SUST also uses), an R32ui image declared R32_UINT in the descriptor's
format bits (flags 8-11 = 5): Vulkan's image atomics require a format,
and the runtime views a 4-byte-texel image as R32_UINT for it. The operation numbering is inferred from a Maxwell lowering,
not measured.

**Task and mesh programs.** NVN binds a task program in the vertex slot
and a mesh program in the vertex or tessellation evaluation slot; the
runtime says which (`Parameters::stage_slot`) and passes the mesh
registers (`Parameters::mesh`), from which come the workgroup size (at
most 32 threads: the hardware runs one warp), the output limits (at most
256 vertices and primitives), the topology and the task payload's size
(the larger of the line counts in SET_MESH_INIT_SHADER bits 12-20 and
22-30, times 128 bytes). The field widths are Lichii's decoding.

The programs talk through the internal stage buffer, which ISBERD and
ISBEWR address by byte, laid out as Mesa's NAK lays it out. A task
program's output attributes become the task payload (the mesh workgroup
count at byte 4, then the counts per dimension and the payload proper);
its function returns, a barrier follows, and `OpEmitMeshTasksEXT` emits
that many mesh workgroups in x. A program's input attributes give its
workgroup index at byte 0; a mesh program after a task program reads the
payload at the other bytes, and one without a task program reads the
draw's workgroup counts at bytes 4-16. A mesh program's output attributes
are a workgroup array: per-vertex lines, then per-primitive lines, in
groups of 32 elements, one 128-byte line per output word its header's
output map lists (the primitive header, a pass-through geometry
program's, lists the per-primitive ones). Its map holds the primitive
count (a word at byte 3) and the vertex indices (bytes from byte 4). At
the end, after a barrier, the program sets its outputs (all its
`max_vertices` vertices, the written primitive count capped at
`max_primitives`) and copies the lines to the position, point size (for
points) and generic vector arrays and to the per-primitive id, layer and
viewport index, and the indices to the primitive index built-in.
Per-primitive generic vectors are refused, as the fragment program would
have to declare them per-primitive; so are 16-bit stage buffer accesses
and outputs past the generic vectors. The `.SKEW` bit is ignored (NAK
computes linear addresses with it set). None of this has run on a GPU
yet.

**Matrix multiply.** HMMA runs on the subgroup: each lane gathers its A
row from its own quad and its B columns from the quads of lanes 8t and
8t + 4. Half products are exact in 32-bit floats and are summed in k
order with 32-bit round-to-nearest; a half result is rounded once at the
end. Published measurements of NVIDIA's tensor cores (Fasi, Higham,
Mikaitis and Pranesh, 2021) show an internal summation that aligns and
truncates, so the last bits of a long sum can differ from the hardware.
With cooperative matrices (`MatrixPath::Cooperative`,
VK_KHR_cooperative_matrix at 16x16x16 and subgroup scope), HMMA.16816
instead writes the lanes' fragments to per-warp workgroup scratch in the
PTX layout (B's upper eight columns zero), multiplies with
OpCooperativeMatrixMulAddKHR (a half or 32-bit accumulator) and reads D
back; the device's matrix units then decide the summation's rounding.
Two HMMA.16816s in a row that share A and the accumulator type (both
unguarded, in one block, the second reading nothing the first writes)
run as one multiply: the second's B and C fill columns 8-15 and its D is
read from there, halving the multiplies and the scratch traffic.
With `MatrixPath::Exact` (`Parameters::matrix_mode` `MatrixMode::Exact`)
the subgroup path sums like the A100's tensor cores as published (Fasi,
Higham, Mikaitis and Pranesh, 2021; Khattak and Mikaitis, 2025): each
block of eight k adds its products and the accumulator in one step, every
addend aligned to the largest exponent and truncated to 24 bits below its
leading bit, the exact sum normalised once by truncation (a half result's
last block rounded to the nearest half instead, through a round-to-odd
32-bit value). It is done in integers from the addends' bits. The
papers do not settle the sign-magnitude truncation, the chaining of the
two blocks of HMMA.16816 through a 32-bit value, the results past the
32-bit range (the largest finite value) and below it (truncated
subnormals), or blocks with an infinity or NaN (their plain sum); these
are inferred, and none of it has been compared with hardware results.

**Exits and barriers.** On the hardware an exited thread stops counting
toward BAR.SYNC and the others synchronise; in SPIR-V a barrier must sit
in uniform control flow. So in a program with BAR, an exit whose other
arm goes on (`@P EXIT` then more code) sets a per-invocation `exited`
flag instead of returning, and the program goes on at the same level:
every run of instructions between barriers is skipped by an exited
invocation, the barriers are not, so every invocation that has not
returned reaches each one. A subgroup whose invocations have all exited
returns at once (a subgroup vote), and an exited invocation leaves any
loop at its next iteration, so stale state cannot keep it looping. An
exited invocation's later stores and shuffles do not happen; a shuffle
that reads one gets whatever it last held (the hardware's value there is
undefined too). Exits anywhere else return as before.

**Simulator.** `echora_shader::sim` (the `sim` feature; the tests use it
too) runs every invocation of a workgroup of a translated compute
program on the CPU: invocations run one at a time to each barrier or
subgroup operation, which completes when every running invocation of its
scope is there; shared memory is the workgroup's, global memory the
dumped device memory reached through the program's own page table. It
reports runaway loops, barriers reached at different places (a
deadlock) or after other invocations returned, reads and writes outside
the dumped memory (faults), indices past an array, shuffles from lanes
not running there, and uses of what SPIR-V leaves undefined. The
`simulate` example runs it on a dump directory (its documentation gives
the format). Bound textures and images come from dumped level-0 texels:
fetches and storage reads and writes are exact, samples are nearest at
level 0 (enough for control flow), gathers take the four texels around
the point; depth comparisons, level queries and texture pool entries
read zeros.

**Exactness.** Float multiplies and adds are marked NoContraction, as
the hardware rounds each one. Directed rounding (`.RM`, `.RP`, `.RZ`) is
exact: the nearest result's error is computed without rounding (two-sum
for an add, a fused multiply-add for a product) and the result steps one
ulp where its sign says the exact value lies. Lane ids, lane masks and the
warp operations assume 32-lane subgroups, which the runtime requests
(VK_EXT_subgroup_size_control) for every translated program.

**Emission quality.** The driver's optimiser (NIR, ACO) does best on
plain structured code, so the output aims for that:

- structured control flow: real if/else, loops and merges, built after
  Ramsey's "Beyond Relooper" over the dominator tree (exits to outer
  constructs pass a target through each merge); the block dispatcher (a
  switch inside one loop) is kept for irreducible graphs and computed
  branches (BRX, RET);
- native subgroup operations for SHFL, VOTE and friends;
- fp16 types where the source works on halves;
- PhysicalStorageBuffer pointers through the page table;
- cooperative matrices for HMMA, with a subgroup fallback;
- specialization constants for run-time settings.

An optional spirv-opt pass (its performance recipe) sits behind a switch
and is measured against the unoptimised output before it becomes a
default.

**Verification.** Unit tests per form (words built from known fields,
decoded and compared); on the user's machine, nvdisasm (CUDA toolkit) as a
decode oracle, and a differential translation of the ~6,200 dumped game
stages against Lichii (spirv-val, a driver compile, and where the two
differ, finding out which is right). No game data is in the repository.

## Decoded so far

Field layout: bits 0-8 the operation, 9-11 the operand form, 12-15 the
guard predicate (15 negates; 7 is PT), 16-23 the destination (255 RZ),
24-31 the first source. The form places the wide operand: forms 4, 5 and 6
put an immediate (bits 32-63), a constant (bank 54-58, byte offset 38-53)
or a uniform register (32-37, with bit 91) in the second source; forms 2, 3 and 7 in the
third; register sources sit in 64-71, or the second in 32-39 in form 1. A
constant form with bit 91 takes its buffer from a uniform register (32-37)
instead of a bank. Banks 18 and up are never bound and read zero (fullscreen
programs read bank 24). The layouts were cross-checked against NAK's encoder.

| Operation | Opcodes (bits 0-11) | Notes |
|---|---|---|
| MOV | 0x202, 0x802, 0xa02, 0xc02 | register, immediate, constant, uniform; quad-lane write mask 72-75 (lane n of each quad written when bit n is set, 0xf in ordinary code; UMOV has none) |
| IADD3 | 0x010 in forms 1-7 | carry outputs 81-86, inputs 87-90 and 77-80 |
| IMAD | 0x024 in forms 1-7 | low word |
| LOP3.LUT | 0x012 in forms 1-7 | table 72-79, `.PAND` 80, predicate output 81-83 |
| FFMA, FMUL, FADD | 0x023, 0x020, 0x021 | saturate 77, rounding 78-79, FTZ 80; FFMA and FMUL `.FMZ` 76; FMUL scale 84-86 (1-3 divide by 8, 4, 2; 4 none; 5-7 multiply by 2, 4, 8) |
| FSETP, ISETP | 0x00b, 0x00c in forms 1, 4, 5, 6 | compare 76-79/76-78, outputs 81 and 84, accumulator 87-90, combine 74-75, ISETP signed 73 and `.EX` 72 |
| S2R | 0x919 | special register 72-79 |
| BRA | 0x947 | 50-bit byte offset 32-81 (low two bits the convergence mode), condition 87-90 |
| EXIT, NOP | 0x94d, 0x918 | |
| BRX, BRXU | 0x949, 0x958 | to the byte offset from the code's start in register 24 (BRXU: a uniform register); blocks start after every unconditional transfer, and the dispatcher selects blocks by their first instruction's index |
| CALL, RET | 0x944, 0x950 | CALL.REL to its 50-bit offset at 32-81 (the return address put in a register by a MOV before it); RET.REL to the byte offset (from the code's start) in register 24, like BRX. A program ends at an unconditional EXIT or RET that no earlier branch or call reaches past |
| FSWZADD | 0x822 | a 24, b 64, pattern 32-39: per quad lane n, bits 2n give a + b, a - b, b - a or b |
| IADD3 carries | 0x010 | `.X` 74: negation is inversion and the carries at 87-90, 77-80 add in |
| IMAD.WIDE, IMAD.HI | 0x025, 0x027 in forms 1-7 | signed 73; C a register pair; carry out 81-83 |
| IABS | 0x013 | the source in the second slot |
| HADD2, HMUL2, HFMA2, HMNMX2 | 0x030 (forms 1-3), 0x032 (1, 4, 5), 0x031 (1-5), 0x040 (1, 4) | half pairs on SPIR-V's 16-bit floats; A at 24 (swizzle 74-75, abs 73, neg 72), B at 32 (swizzle 60-61, abs 62, neg 63) or 64 (81-82, 83, 84) in forms 2 and 3, or wide; swizzle 0 the pair, 1 a float in both, 2 the low half, 3 the high; saturate 77; HFMA2 `.RELU` (79) clamps each lane, where the predicate at 87-90 holds, negative to +0 and NaN to 0x7fff; HMNMX2's minimum predicate 87-90; HADD2's second source in the third slot in forms 2 and 3; HADD2.F32 (78) adds the lane-0 halves as one float |
| HSETP2 | 0x034 (forms 1, 2) | P (81-83) lane 0's comparison, Q (84-86) lane 1's, each combined (69-70) with the accumulator (87-90); compare 76-79; not `.H_AND` (71) |
| UFLO, UBREV, UPOPC | 0x0bd, 0x0be, 0x0bf | uniform forms of FLO, BREV, POPC |
| F2FP.PACK_AB | 0x03e (forms 1, 5) | b's half low, a's high, or with MERGE_C (78) register 64's low half high |
| POPC, FLO, BREV | 0x109, 0x100, 0x101 | invert 63 (form 1); FLO signed 73, shift amount 74 |
| SHFL | 0x389, 0x589, 0x989, 0xf89 | lane register 32 or immediate 53-57, clamp and segment mask register 64 or immediate 40-52, mode 58-59, in-range predicate 81-83 |
| REDUX | 0x3c4 | uniform destination 16-21, register 24, operation 78-80 (AND, OR, XOR, SUM, MIN, MAX), signed 73 |
| VOTEU | 0x886 | as VOTE, into a uniform register and uniform predicate |
| BREAK, WARPSYNC | 0x942, 0x948 | no effect per invocation |
| VOTE | 0x806 | mode 72-73 (ALL, ANY, EQ), condition 87-90, vote predicate 81-83, ballot in the destination |
| P2R, R2P, UP2UR | 0x803, 0x804, 0x883 | immediate mask 32-63 (P0-P6), byte 76-77; UP2UR is P2R of the uniform predicates into a uniform register |
| LEA | 0x011 in forms 1-7 | shift 75-79, `.HI` 80, negate shifted 72, `.X` 74, carry out 81, in 87-90 |
| SHF | 0x019 in forms 1-7 | type 73-74 (S64, U64, S32, U32), wrap 75, right 76, high word 80 |
| PRMT, SGXT | 0x016, 0x01a | PRMT mode 72-74; SGXT signed 73, wrap 75 |
| SEL, FSEL | 0x007, 0x008 | condition 87-90 |
| FMNMX, IMNMX | 0x009, 0x017 | minimum when the predicate at 87-90 holds; IMNMX signed 73 |
| MUFU | 0x108 | function 74-79, f16 73; SIN and COS take turns |
| F2I, I2F, I2FP, FRND | 0x105, 0x106, 0x045, 0x107 | sizes 75-76 and 84-85 (log2 bytes), rounding 78-79, F2I signed 72, I2F 74 |
| F2IP | 0x243 | `.U8.F32` with the third source RZ and bit 74 set (other mode bits refused); b and a rounded by 78-79, saturated to 0-255 with NaN to 0, packed as b in bits 0-7 and a in 8-15. The saturation and NaN results are inferred from F2I, not measured |
| FSET.BF | 0x00a | as FSETP, with 1.0 or 0.0 in a register |
| PLOP3 | 0x81c | inputs 87-90, 77-80, 68-71; tables 64-66 with 72-76, and 16-23 |
| CS2R | 0x805 | special register 72-79 (255 SRZ), pair 80 |
| S2R specials | | 0x13 THREAD_KILL (helper or demoted: 1, else 0), 0x84 VARIABLE_RATE (X rate in bits 8-15, Y in 16-23; always 1x1 here), 0 lane id, 0x21-0x23 thread id, 0x25-0x27 workgroup id, 0x38-0x3c lane masks (EQ, LT, LE, GT, GE), 0x50-0x51 clock |
| LDC, ULDC | 0xb82, 0xab9, 0xabb (ULDC indexed by a uniform register) | index register 24, bank 54-58, offset 38-53, size 73-75, LDC mode 78-79; ULDC destination 16-21 |
| LDG, STG | 0x381, 0x981, 0x386, 0x986 | address register 24 (pair: bit 72 in 0x3xx, 90 in 0x9xx), uniform register 32 (loads) or 64 (stores) with bit 91 (pair 72), offset 40-63 signed, type 73-75, data 32; 0x981's condition at 64-66 stored as 7 - P (negate 67); memory order 77-80: .STRONG.GPU (7) and .STRONG.SYS (0xa) as Volatile accesses, so they reach memory other workgroups see; .STRONG.SM (5, what compilers emit for ordinary accesses), weak (0) and constant (4) as plain ones |
| Byte and half-word stores | STG, STS, STL types 0-3 | into the word holding them: global and shared memory clear and set the bytes with an atomic AND and OR, so neighbouring bytes written by others survive; local memory reads, merges and writes |
| LDS, STS, LDL, STL | 0x984, 0x388/0x988, 0x983, 0x387/0x987 | as global, 32-bit; shared register scale 78-79 (1, 4, 8, 16) |
| RED, ATOMG, ATOMS | 0x98e, 0x9a8/0x3a8, 0x38c/0x98c; CAS 0x3a9, 0x38d | operation 87-90 (RED 87-89), type 73-75; address as loads (ATOMG 0x9a8's pair flag at 70); data 32, or compare 32 and data 64 for CAS |
| TXD | 0xb6c | 2D and 2D array; coordinates (x, y, then the layer) from Ra, gradients from Rb as d/dx and d/dy of x, then of y; results as TEX |
| TEX, TLD.SCR, TLD4, TXQ, TMML | 0xb60, 0xb66, 0xb63, 0xb6f, 0xb69 (bindless 0xf6x) | handle word c[54-58][40-53 × 4], dimension 61-63 (1D, 2D, 3D, cube, 1D array, 2D array, -, cube array), level mode 87-89 (implicit, zero, bias, explicit), TLD4 component 87-88, offsets 76 (TLD4 per-tap with 76-77 = 2: signed 6-bit x, y per tap in the bytes of two registers, one gather per tap), TXQ and TMML have no fault predicate at 81-83, TXQ only the dimension query (76-89 clear), depth compare 78, half results 79, mask 72-75, second destination 64; values (array index first, coordinates, level, offsets, reference) half in Ra and half in Rb up to four, else four and the rest |
| TXQ.B | 0x370 | as Lichii decodes it: bit 59 alone of 59-63, nothing at 32-58, 76-89 or 91; the handle in Ra, the level in Ra+1, through the texture pool; the dimensions of a 2D view (width, height, 1, levels), as TXQ |
| SULD.P, SUST.P | 0xb97, 0xb9b | handle word c[54-58][40-53 × 4], dimension 61-63 (1D, buffer, -, 2D, 2D array, 3D), component mask 72-75, coordinates from register 24, SULD destination 16, SUST data 32 |
| SURED | 0x59f | `.D.U32` (bits 64-95 other than the operation exactly `0x0010e000`), buffer or 2D; handle and coordinates as SULD, data 32, operation 87-89 (0 add, 1 min, 2 max, 5 and, 6 or, 7 xor; 3 and 4 refused); an atomic on its own R32_UINT descriptor |
| ALD, AST | 0x321, 0x322 | address 40-49, words 74-75 (plus one), index register 24, vertex register 32 (ALD) or 64 (AST), patch 76, ALD output 79; AST data 32 |
| IPA | 0x326, 0x526 (address plus the uniform register at 24: a select over the generic components the SPH uses) | address 64-71 (× 4), location 76-77 (default, centroid, offset), mode 78-79 (interpolated, constant), offset register 32 |
| KILL | 0x95b | condition 87-90 |
| LDSM | 0x83b | matrices 1 << 72-73, `.MT88` 78; address register 24, uniform register 32 with bit 91, offset 40-63; row addresses shuffled from the lanes that supply them, shared words read (halves for `.MT88`) |
| MOVM | 0x23a | `.16.MT88` only (bits 32-95 zero); the 8x8 half matrix transposed in the same lane layout: two subgroup shuffles from lanes 8(t%4) + t/8 and +4, the half picked by (t/4)%2 |
| ISBERD, ISBEWR | 0x923, 0x927 | NAK's fields: address register 24, offset 40-55 (56-73 must be clear), size 74-75, region 76-77 (0 the map, 3 attributes), `.SKEW` 78, `.O` 79; ISBERD destination 16, ISBEWR data 32. 32-bit attribute reads and writes, U8 map writes and the primitive count; see Task and mesh programs |
| HMMA | 0x23c | d 16, a 24, b 32, c 64; m16n8k16 with bit 75 (else m16n8k8; 78 the k4 form, not decoded), 32-bit accumulator with bit 76 (else halves), half sources (82-83 zero); PTX fragment layout per lane, rows and columns gathered by subgroup shuffles |
| BAR.SYNC | 0xb1d | mode 77-78 (0 SYNC); a workgroup control barrier that orders uniform (buffer and physical), workgroup and image memory, as bar.sync orders all memory for the CTA |
| MEMBAR | 0x992 | scope 76-78 (0 CTA, 1 SM, 2 GPU, 3 system; .MMIO 72 and other bits but .SC 80 refused); OpMemoryBarrier over all memory at the workgroup (CTA, SM) or the device (GPU, system: Vulkan has nothing wider) |
| ERRBAR | 0x9ab | no operands; nothing (it orders no memory) |
| LDGSTS, LDGDEPBAR | 0xfae, 0x9af | as Lichii decodes it (NAK has none): 16 bytes from [R24.64 + bits 32-43] to shared [R16 + bits 48-63 × 16], or zeros where the enable predicate (87-89, negated by 90, or by 70 in the form with a cache-policy descriptor at 64-69) fails; copied at once, so LDGDEPBAR and DEPBAR wait for nothing |
| Uniform ALU | 0x082, 0x087, 0x08c, 0x090-0x092, 0x096, 0x099, 0x09a, 0x0a4, 0x0a5 in forms 1, 2, 4 (UMOV 4, 6); UPLOP3 0x89c | the per-thread operation 0x080 lower, on uniform registers (63 URZ) and uniform predicates (7 UPT), the guard included; ULDC and S2UR (0x9c3) are uniform too |
| R2UR | 0x3c2 | uniform destination 16-21, register 24 |
| ISETP.EX | 0x00c bit 72 | the low words' result at 68-71 decides when the high words are equal |
| BSSY, BSYNC, DEPBAR | 0x945, 0x941, 0x91a | no effect per invocation |
