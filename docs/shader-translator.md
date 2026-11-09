# Shader translator (`echora-shader`)

The game's GPU programs are NVN2 programs: NVIDIA Ampere (SM 8.x, the
Switch 2's T239) machine code with a control section. Today Lichii's
recompiler (C++, behind `echora-gpu/native/shader`) turns them into
SPIR-V, lowering Ampere to Maxwell first. `echora-shader` is Echora's own
translator, in Rust, growing beside it: a program the Rust translator
handles uses its SPIR-V, anything else still goes to Lichii, and a
differential mode compares the two. Lichii stays until the Rust translator
covers every program.

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
  and `ABI_VERSION`, which is part of every shader cache key). Lichii's
  bridge fills the same types.
- `echora_shader::translate(program, &Parameters) -> Result<Translated,
  Unsupported>`: pure, no callbacks. `Parameters` carries what the program
  does not (only the stage slot a task or mesh program is bound as);
  `Unsupported` names the first unhandled instruction or feature and its
  offset. `Translated` lists the specialization constants and push
  constants it reads.
- `ECHORA_SHADER_TRANSLATOR` selects `lichii` (default), `rust` (Rust where
  supported, Lichii otherwise) or `compare`.
- **Page table**: 64 KiB pages; a buffer of u64 device addresses (0
  unmapped) for `[va_base, va_base + pages * 64 KiB)`; the table's address,
  `va_base >> 16` and the page count in bytes 112-128 of the push-constant
  block. A global access reads `table[(va >> 16) - base] + (va & 0xffff)`
  through PhysicalStorageBuffer64. Accesses are naturally aligned (a
  misaligned global access faults on the hardware) and at most 16 bytes, so
  none crosses a page: the first byte's page decides. An unmapped or
  out-of-range page reads 0 and drops writes.

## Plan

1. **Decoder**: 128-bit words into typed instructions; unsupported forms are
   `Op::Unknown` with the raw bits, never a panic.
2. **IR** and lowering of the common ALU, memory and control forms.
3. **SPIR-V** for simple compute programs (a structurizer for the control
   flow; an emitter, `rspirv` or a small hand-written writer).
4. **Graphics stages** and textures.
5. **HMMA**: a subgroup path, and VK_KHR_cooperative_matrix.

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
instead of a bank. The layouts were cross-checked against NAK's encoder.

| Operation | Opcodes (bits 0-11) | Notes |
|---|---|---|
| MOV | 0x202, 0x802, 0xa02, 0xc02 | register, immediate, constant, uniform |
| IADD3 | 0x010 in forms 1-7 | carry outputs 81-86, inputs 87-90 and 77-80 |
| IMAD | 0x024 in forms 1-7 | low word |
| LOP3.LUT | 0x012 in forms 1-7 | table 72-79, `.PAND` 80, predicate output 81-83 |
| FFMA, FMUL, FADD | 0x023, 0x020, 0x021 | saturate 77, rounding 78-79, FTZ 80; FFMA and FMUL `.FMZ` 76; FMUL scale 84-86 |
| FSETP, ISETP | 0x00b, 0x00c in forms 1, 4, 5, 6 | compare 76-79/76-78, outputs 81 and 84, accumulator 87-90, combine 74-75, ISETP signed 73 and `.EX` 72 |
| S2R | 0x919 | special register 72-79 |
| BRA | 0x947 | 50-bit byte offset 32-81 (low two bits the convergence mode), condition 87-90 |
| EXIT, NOP | 0x94d, 0x918 | |
