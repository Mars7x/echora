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
5. **HMMA**: a subgroup path, and VK_KHR_cooperative_matrix.

**Textures.** A texture whose handle is a constant-bank word becomes a
combined image sampler, one per handle word, view type and depth
comparison, reported as a descriptor naming that word; the runtime binds
the texture and sampler the word names. Bindings follow the uniform
buffers. Programs without derivatives sample at an explicit level (a
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
0x2f8 and 0x2fc the instance and vertex ids (without the draw's base).
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
(bit 609). KILL demotes the invocation to a helper. `stores` has bit
(address / 4) set for every attribute word written.

**Exactness.** Float multiplies and adds are marked NoContraction, as
the hardware rounds each one. Directed rounding (`.RM`, `.RP`, `.RZ`) is
exact: the nearest result's error is computed without rounding (two-sum
for an add, a fused multiply-add for a product) and the result steps one
ulp where its sign says the exact value lies. Lane ids, lane masks and the
warp operations assume 32-lane subgroups, which the runtime requests
(VK_EXT_subgroup_size_control) for every translated program.

**Emission quality.** The driver's optimiser (NIR, ACO) does best on
plain structured code, so the output aims for that:

- a structurizer that emits real if/else, loops and merges, with the
  block dispatcher (a switch inside one loop, the first translation's
  form) kept only for irreducible regions;
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
| MOV | 0x202, 0x802, 0xa02, 0xc02 | register, immediate, constant, uniform |
| IADD3 | 0x010 in forms 1-7 | carry outputs 81-86, inputs 87-90 and 77-80 |
| IMAD | 0x024 in forms 1-7 | low word |
| LOP3.LUT | 0x012 in forms 1-7 | table 72-79, `.PAND` 80, predicate output 81-83 |
| FFMA, FMUL, FADD | 0x023, 0x020, 0x021 | saturate 77, rounding 78-79, FTZ 80; FFMA and FMUL `.FMZ` 76; FMUL scale 84-86 (1-3 divide by 8, 4, 2; 4 none; 5-7 multiply by 2, 4, 8) |
| FSETP, ISETP | 0x00b, 0x00c in forms 1, 4, 5, 6 | compare 76-79/76-78, outputs 81 and 84, accumulator 87-90, combine 74-75, ISETP signed 73 and `.EX` 72 |
| S2R | 0x919 | special register 72-79 |
| BRA | 0x947 | 50-bit byte offset 32-81 (low two bits the convergence mode), condition 87-90 |
| EXIT, NOP | 0x94d, 0x918 | |
| IADD3 carries | 0x010 | `.X` 74: negation is inversion and the carries at 87-90, 77-80 add in |
| IMAD.WIDE, IMAD.HI | 0x025, 0x027 in forms 1-7 | signed 73; C a register pair; carry out 81-83 |
| IABS | 0x013 | the source in the second slot |
| POPC, FLO, BREV | 0x109, 0x100, 0x101 | invert 63 (form 1); FLO signed 73, shift amount 74 |
| SHFL | 0x389, 0x589, 0x989, 0xf89 | lane register 32 or immediate 53-57, clamp and segment mask register 64 or immediate 40-52, mode 58-59, in-range predicate 81-83 |
| REDUX | 0x3c4 | uniform destination 16-21, register 24, operation 78-80 (AND, OR, XOR, SUM, MIN, MAX), signed 73 |
| VOTEU | 0x886 | as VOTE, into a uniform register and uniform predicate |
| BREAK, WARPSYNC | 0x942, 0x948 | no effect per invocation |
| VOTE | 0x806 | mode 72-73 (ALL, ANY, EQ), condition 87-90, vote predicate 81-83, ballot in the destination |
| P2R, R2P | 0x803, 0x804 | immediate mask 32-63 (P0-P6), byte 76-77 |
| LEA | 0x011 in forms 1-7 | shift 75-79, `.HI` 80, negate shifted 72, `.X` 74, carry out 81, in 87-90 |
| SHF | 0x019 in forms 1-7 | type 73-74 (S64, U64, S32, U32), wrap 75, right 76, high word 80 |
| PRMT, SGXT | 0x016, 0x01a | PRMT mode 72-74; SGXT signed 73, wrap 75 |
| SEL, FSEL | 0x007, 0x008 | condition 87-90 |
| FMNMX, IMNMX | 0x009, 0x017 | minimum when the predicate at 87-90 holds; IMNMX signed 73 |
| MUFU | 0x108 | function 74-79, f16 73; SIN and COS take turns |
| F2I, I2F, I2FP, FRND | 0x105, 0x106, 0x045, 0x107 | sizes 75-76 and 84-85 (log2 bytes), rounding 78-79, F2I signed 72, I2F 74 |
| FSET.BF | 0x00a | as FSETP, with 1.0 or 0.0 in a register |
| PLOP3 | 0x81c | inputs 87-90, 77-80, 68-71; tables 64-66 with 72-76, and 16-23 |
| CS2R | 0x805 | special register 72-79 (255 SRZ), pair 80 |
| S2R specials | | 0 lane id, 0x21-0x23 thread id, 0x25-0x27 workgroup id, 0x38-0x3c lane masks (EQ, LT, LE, GT, GE), 0x50-0x51 clock |
| LDC, ULDC | 0xb82, 0xab9, 0xabb (ULDC indexed by a uniform register) | index register 24, bank 54-58, offset 38-53, size 73-75, LDC mode 78-79; ULDC destination 16-21 |
| LDG, STG | 0x381, 0x981, 0x386, 0x986 | address register 24 (pair: bit 72 in 0x3xx, 90 in 0x9xx), uniform register 32 (loads) or 64 (stores) with bit 91 (pair 72), offset 40-63 signed, type 73-75, data 32; 0x981's condition at 64-66 stored as 7 - P (negate 67) |
| LDS, STS, LDL, STL | 0x984, 0x388/0x988, 0x983, 0x387/0x987 | as global, 32-bit; shared register scale 78-79 (1, 4, 8, 16) |
| RED, ATOMG, ATOMS | 0x98e, 0x9a8/0x3a8, 0x38c/0x98c; CAS 0x3a9, 0x38d | operation 87-90 (RED 87-89), type 73-75; address as loads (ATOMG 0x9a8's pair flag at 70); data 32, or compare 32 and data 64 for CAS |
| TEX, TLD.SCR, TLD4, TXQ | 0xb60, 0xb66, 0xb63, 0xb6f | handle word c[54-58][40-53 × 4], dimension 61-63 (1D, 2D, 3D, cube, 1D array, 2D array, -, cube array), level mode 87-89 (implicit, zero, bias, explicit), TLD4 component 87-88, offsets 76, depth compare 78, half results 79, mask 72-75, second destination 64; values (array index first, coordinates, level, offsets, reference) half in Ra and half in Rb up to four, else four and the rest |
| ALD, AST | 0x321, 0x322 | address 40-49, words 74-75 (plus one), index register 24, vertex register 32 (ALD) or 64 (AST), patch 76, ALD output 79; AST data 32 |
| IPA | 0x326 | address 64-71 (× 4), location 76-77 (default, centroid, offset), mode 78-79 (interpolated, constant), offset register 32 |
| KILL | 0x95b | condition 87-90 |
| BAR.SYNC | 0xb1d | mode 77-78 (0 SYNC) |
| Uniform ALU | 0x082, 0x087, 0x08c, 0x090-0x092, 0x096, 0x099, 0x09a, 0x0a4, 0x0a5 in forms 1, 2, 4 (UMOV 4, 6); UPLOP3 0x89c | the per-thread operation 0x080 lower, on uniform registers (63 URZ) and uniform predicates (7 UPT), the guard included; ULDC and S2UR (0x9c3) are uniform too |
| R2UR | 0x3c2 | uniform destination 16-21, register 24 |
| ISETP.EX | 0x00c bit 72 | the low words' result at 68-71 decides when the high words are equal |
| BSSY, BSYNC, DEPBAR | 0x945, 0x941, 0x91a | no effect per invocation |
