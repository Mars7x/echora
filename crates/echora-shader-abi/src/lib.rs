//! The interface between Echora's shader translators and the runtime that
//! runs their programs: what a translator is given, what it returns, and
//! the run-time inputs (push constants, the GPU page table) its SPIR-V may
//! read. Both translators produce these types: the C++ recompiler through
//! `echora-gpu`'s bridge, and the Rust translator (`echora-shader`) directly.
//! No dependencies, so a translator need not link the Vulkan backend.

/// Changes whenever anything a translated program relies on changes (the
/// push-constant layout, the page-table format, descriptor numbering). It
/// is stored in every `Translated` and is part of every shader cache key.
pub const ABI_VERSION: u32 = 4;

/// A program's pipeline stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stage {
    Vertex,
    TessControl,
    TessEval,
    Geometry,
    Fragment,
    Compute,
    /// Switch 2 task and mesh programs (VK_EXT_mesh_shader stages).
    Task,
    Mesh,
}

impl Stage {
    /// The C++ recompiler's stage number (Shader::Stage).
    pub fn from_raw(raw: u32) -> Option<Self> {
        Some(match raw {
            0 => Self::Vertex,
            1 => Self::TessControl,
            2 => Self::TessEval,
            3 => Self::Geometry,
            4 => Self::Fragment,
            5 => Self::Compute,
            7 => Self::Task,
            8 => Self::Mesh,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DescriptorKind {
    UniformBuffer,
    StorageBuffer,
    TextureBuffer,
    ImageBuffer,
    Texture,
    Image,
}

/// One set-0 binding, in the order the program's SPIR-V numbers them. The
/// layout matches `EchoraShaderDescriptor` in the C++ bridge.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Descriptor {
    pub kind: u32,
    pub binding: u32,
    pub count: u32,
    /// Guest constant buffer index of a uniform buffer.
    pub index: u32,
    pub cbuf_index: u32,
    pub cbuf_offset: u32,
    pub shift_left: u32,
    pub has_secondary: u32,
    pub secondary_cbuf_index: u32,
    pub secondary_cbuf_offset: u32,
    pub secondary_shift_left: u32,
    pub size_shift: u32,
    pub texture_type: u32,
    /// Bit 0 written, 1 read, 2 integer, 3 depth, 4 multisample, 5 a
    /// storage buffer at a raw 8-byte pointer (no size word: the binding
    /// gives the bounds). Images: bits 8-11 the declared format
    /// (Shader::ImageFormat; 0 typeless).
    pub flags: u32,
    /// Storage buffers: the pointer is read from memory this many bytes
    /// past the constant buffer pointer (`u32::MAX`: the constant buffer
    /// word is the pointer).
    pub indirect_offset: u32,
}

impl Descriptor {
    pub fn kind(&self) -> DescriptorKind {
        match self.kind {
            0 => DescriptorKind::UniformBuffer,
            1 => DescriptorKind::StorageBuffer,
            2 => DescriptorKind::TextureBuffer,
            3 => DescriptorKind::ImageBuffer,
            4 => DescriptorKind::Texture,
            _ => DescriptorKind::Image,
        }
    }
}

/// What ahead-of-time translation needs beyond the program itself. All
/// other pipeline state reaches a translated program at run time (spec
/// constants, push constants), so one translation serves every pipeline.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Parameters {
    /// The stage the program is bound as when it is not the stage its
    /// header names: task or mesh.
    pub stage_slot: Option<Stage>,
    /// Compute: workgroup size, when the control section does not give it.
    pub workgroup_size: Option<[u32; 3]>,
    /// Compute: shared memory bytes, when the control section does not.
    pub shared_memory_size: Option<u32>,
    /// Local (per-invocation stack) memory bytes, when not given.
    pub local_memory_size: Option<u32>,
    /// How HMMA (the warp's half matrix multiply) translates.
    pub matrix_mode: MatrixMode,
    /// Task and mesh programs: the pipeline state they are translated
    /// against (`stage_slot` names which of the two).
    pub mesh: Option<MeshParameters>,
}

/// The task or mesh pipeline state a program is translated against, as the
/// runtime derives it from the programs' control sections (`bound_mesh_program`
/// in the runtime).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MeshParameters {
    /// The task stage (else the mesh stage).
    pub task: bool,
    /// The hardware's SET_MESH_SHADER_A, SET_MESH_SHADER_B and
    /// SET_MESH_INIT_SHADER words:
    /// - A: bits 0..4 the output topology (4, triangles: the only one the
    ///   game's mesh programs use), bits 4..13 the maximum vertices, bits
    ///   13..23 the maximum primitives. The two counts come from the mesh
    ///   program's control section word +0x60 (vertices in its low half,
    ///   primitives in its high half).
    /// - B: bits 20..26 the mesh program's threads per workgroup (its
    ///   control word +0x58, bits 0..6).
    /// - INIT: the task program's control word +0x58 when a task program
    ///   is bound, else the mesh program's +0x58 with bits 0..6 cleared.
    pub registers: [u32; 3],
    /// Mesh programs: the program header (0x50 bytes) of the pass-through
    /// geometry program bound with them, whose output map lists the
    /// per-primitive outputs.
    pub primitive_header: Option<[u8; 0x50]>,
}

/// HMMA's translation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MatrixMode {
    /// Subgroup shuffles; products summed in k order with 32-bit
    /// round-to-nearest (any device).
    #[default]
    Subgroup,
    /// Cooperative matrices: only for a device that multiplies 16x16x16
    /// half matrices into half and 32-bit accumulators at subgroup scope
    /// (VK_KHR_cooperative_matrix). The device's rounding.
    Cooperative,
    /// As `Cooperative`, staging the fragments as whole 32-bit registers
    /// (half matrices loaded through word pointers): half the stores.
    /// Under test (ECHORA_HMMA=coop32) until it matches `Cooperative`.
    Cooperative32,
    /// Subgroup shuffles with the tensor cores' published arithmetic:
    /// blocks of eight products and the accumulator aligned to the
    /// largest exponent, truncated, and normalised once. Slower; closest
    /// to the hardware's results.
    Exact,
}

/// A program a translator does not handle (yet).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unsupported {
    pub reason: String,
    /// Byte offset of the first instruction it could not translate.
    pub offset: Option<u32>,
}

/// A translator: a pure function of the program bytes (as NVN stores them
/// at the stage's GPU address) and `Parameters`, callable from many threads
/// at once.
pub type Translate = fn(&[u8], &Parameters) -> Result<Translated, Unsupported>;

/// What a run-time specialization constant a program declares stands for;
/// the pipeline builder fills it from the bound state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpecMeaning {
    /// Vertex attribute `index`'s type: 0 float, 1 signed, 2 unsigned.
    GenericInputType(u8),
    /// The sampled texture or image at descriptor `binding` is integer.
    IntegerFormat(u32),
    /// The entries the texture pool's arrays (set `TEXTURE_POOL_SET`)
    /// hold: a handle whose texture index is this or more reads as an
    /// unwritten entry (zeros), without indexing past the arrays. Its
    /// default is `TEXTURE_INDEX_MASK + 1`, every index a handle carries.
    TexturePoolCapacity,
    /// Debugging (programs translated with loop tracing): the low and high
    /// words of the device address of the loop trace buffer (u32 words; 0:
    /// no tracing). See the translator's documentation of the layout.
    LoopTraceLow,
    LoopTraceHigh,
    /// Debugging (programs translated with vertex tracing): the low and
    /// high words of the device address of the vertex trace buffer (0: no
    /// tracing). See the translator's documentation of the layout.
    VertexTraceLow,
    VertexTraceHigh,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpecConstant {
    pub id: u32,
    pub meaning: SpecMeaning,
}

/// A translated program.
#[derive(Clone, Debug)]
pub struct Translated {
    /// `ABI_VERSION` of the translator that made it.
    pub abi_version: u32,
    pub stage: Stage,
    pub spirv: Vec<u32>,
    pub descriptors: Vec<Descriptor>,
    pub loaded_generics: u32,
    pub stored_generics: u32,
    /// Every attribute written, as a Shader IR::Attribute bit mask.
    pub stores: [u64; 8],
    pub frag_colors: u32,
    pub cbuf_used_sizes: [u32; 18],
    /// Specialization constants the pipeline builder must set.
    pub spec_constants: Vec<SpecConstant>,
    /// `PushConstants` fields the program reads (`PUSH_*` bits).
    pub push_fields: u32,
    /// The texture pool bindings (descriptor set `TEXTURE_POOL_SET`) the
    /// program reads, bit n for binding n (`texture_pool::*`); 0 when it
    /// uses only set 0.
    pub texture_pool_bindings: u32,
}

/// The push-constant block every translated program shares, 128 bytes at
/// offset 0 (the size Vulkan guarantees). Field offsets are fixed by the
/// ABI; a program declares only the ranges it reads.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PushConstants {
    /// Bytes 0..80: words of the translator's own (the C++ recompiler's
    /// texture rescaling bits, say).
    pub translator: [u32; 20],
    /// Bytes 80..96: where a vertex program's generic inputs come from
    /// (`VertexFetch`).
    pub vertex_fetch: VertexFetch,
    /// Bytes 96..112: render-target resolution scale (x, y, inverse x,
    /// inverse y).
    pub resolution_scale: [f32; 4],
    /// Bytes 112..128: the GPU page table (`PageTable`).
    pub page_table: PageTable,
}

impl PushConstants {
    /// The block as pushed (little-endian, the layout above).
    pub fn to_bytes(&self) -> [u8; PUSH_CONSTANTS_SIZE as usize] {
        let mut bytes = [0u8; PUSH_CONSTANTS_SIZE as usize];
        let mut words = Vec::with_capacity(32);
        words.extend_from_slice(&self.translator);
        let fetch = &self.vertex_fetch;
        words.extend([
            fetch.table as u32,
            (fetch.table >> 32) as u32,
            fetch.base_instance,
            fetch.attributes,
        ]);
        words.extend(self.resolution_scale.map(f32::to_bits));
        let table = &self.page_table;
        words.extend([
            table.address as u32,
            (table.address >> 32) as u32,
            table.first_page,
            table.page_count,
        ]);
        for (chunk, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

/// `Translated::push_fields` bits.
pub const PUSH_TRANSLATOR: u32 = 1;
pub const PUSH_RESOLUTION_SCALE: u32 = 2;
pub const PUSH_PAGE_TABLE: u32 = 4;
pub const PUSH_VERTEX_FETCH: u32 = 8;
/// The fields of the ABI block (all but the translator's words).
pub const PUSH_ABI_FIELDS: u32 = PUSH_RESOLUTION_SCALE | PUSH_PAGE_TABLE | PUSH_VERTEX_FETCH;

pub const PUSH_TRANSLATOR_OFFSET: u32 = 0;
pub const PUSH_VERTEX_FETCH_OFFSET: u32 = 80;
pub const PUSH_RESOLUTION_SCALE_OFFSET: u32 = 96;
pub const PUSH_PAGE_TABLE_OFFSET: u32 = 112;
pub const PUSH_CONSTANTS_SIZE: u32 = 128;

/// Global memory a program reaches through pointers. The table is a buffer
/// of u64 device addresses, one per `PAGE_SIZE` page of guest GPU virtual
/// address space from page `first_page`; 0 marks an unmapped page. Pages
/// are 4 KiB, the granularity of NVN memory pools, so no page spans two
/// pools. An
/// access to guest address `va` reads `table[(va >> PAGE_SHIFT) -
/// first_page] + (va & (PAGE_SIZE - 1))` through PhysicalStorageBuffer64.
///
/// Only the first byte's page is looked up: guest global accesses are
/// naturally aligned and at most 16 bytes, so none crosses a 4 KiB page. An
/// address outside the table or on an unmapped page reads 0, drops a
/// write, and an atomic there returns 0. Shared and local memory do not
/// go through the table.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PageTable {
    /// Device address of the table buffer (0: no table bound).
    pub address: u64,
    /// The first guest page the table covers (guest address >> PAGE_SHIFT).
    pub first_page: u32,
    /// Pages in the table.
    pub page_count: u32,
}

/// Bindless textures: a texture or image whose handle a program computes
/// at run time is read from the bound NVN texture and sampler pools,
/// which every program sees as arrays in descriptor set 1, indexed by the
/// handle: bits 0..20 the texture's index in the texture pool
/// (`TEXTURE_INDEX_MASK`), bits 20..32 the sampler's in the sampler pool
/// (`handle >> SAMPLER_INDEX_SHIFT`), as NVN forms handles.
///
/// Each binding is an unsized array (runtimeDescriptorArray, partially
/// bound, indexed with NonUniform); its entries are the pool's descriptors
/// in that view type. An entry the runtime has not written, or a texture
/// of another view type, reads as a null descriptor (zeros). Sampling
/// combines `texture_pool::SAMPLERS[sampler index]` with the image; a
/// sampler with depth compare comes from the guest's sampler state, so
/// depth-compare reads need no other binding. Integer textures are read
/// from the `UINT_*` arrays (a uint view of the same entry), chosen at run
/// time by the entry's bit in `texture_pool::FLAGS`.
pub const TEXTURE_POOL_SET: u32 = 1;
// Specialization constant ids, one table for every program the translator
// makes (each program's `spec_constants` lists the ones it declares):
//
// - 0..27: unused.
// - 27, 28: `VertexTraceLow`, `VertexTraceHigh` (ECHORA_VERTEX_TRACE).
// - 29, 30: `LoopTraceLow`, `LoopTraceHigh` (ECHORA_LOOP_TRACE).
// - 31: `TexturePoolCapacity`.
// - 32 and up: `IntegerFormat`, one per sampled texture or storage image
//   binding the program reads, numbered from `INTEGER_FORMAT_SPEC_BASE`
//   in the order the translation meets them.

/// The specialization constant ids of `SpecMeaning::VertexTraceLow` and
/// `VertexTraceHigh`.
pub const VERTEX_TRACE_LOW_SPEC_ID: u32 = 27;
pub const VERTEX_TRACE_HIGH_SPEC_ID: u32 = 28;
/// The specialization constant ids of `SpecMeaning::LoopTraceLow` and
/// `LoopTraceHigh`.
pub const LOOP_TRACE_LOW_SPEC_ID: u32 = 29;
pub const LOOP_TRACE_HIGH_SPEC_ID: u32 = 30;
/// The specialization constant id of `SpecMeaning::TexturePoolCapacity`.
pub const TEXTURE_POOL_CAPACITY_SPEC_ID: u32 = 31;
/// The first `SpecMeaning::IntegerFormat` constant's id; the others follow.
pub const INTEGER_FORMAT_SPEC_BASE: u32 = 32;
pub const TEXTURE_INDEX_MASK: u32 = 0xf_ffff;
pub const SAMPLER_INDEX_SHIFT: u32 = 20;
/// Entries per texture array, which the renderer gives programs as
/// `SpecMeaning::TexturePoolCapacity`. Pools this size cover every pool
/// the game makes.
pub const TEXTURE_POOL_CAPACITY: u32 = 1 << 16;
/// Entries in the sampler array: every 12-bit sampler index.
pub const SAMPLER_POOL_CAPACITY: u32 = 1 << 12;

/// Set-1 bindings (`TEXTURE_POOL_SET`).
pub mod texture_pool {
    /// VkSampler array (`OpTypeSampler`).
    pub const SAMPLERS: u32 = 0;
    /// Sampled-image arrays, float results, by view type.
    pub const SAMPLED_1D: u32 = 1;
    pub const SAMPLED_1D_ARRAY: u32 = 2;
    pub const SAMPLED_2D: u32 = 3;
    pub const SAMPLED_2D_ARRAY: u32 = 4;
    pub const SAMPLED_3D: u32 = 5;
    pub const SAMPLED_CUBE: u32 = 6;
    pub const SAMPLED_CUBE_ARRAY: u32 = 7;
    /// Uniform texel buffers.
    pub const SAMPLED_BUFFER: u32 = 8;
    /// Storage images, format Unknown (typeless).
    pub const STORAGE_2D: u32 = 9;
    pub const STORAGE_3D: u32 = 10;
    /// Sampled-image arrays, uint results.
    pub const UINT_2D: u32 = 11;
    pub const UINT_2D_ARRAY: u32 = 12;
    pub const UINT_3D: u32 = 13;
    /// A storage buffer of one u32 per texture index (`uint flags[]`):
    /// bit 0 set when the entry is integer, so programs read it through
    /// the `UINT_*` view (only 2D, 2D array and 3D have one).
    pub const FLAGS: u32 = 14;
    pub const COUNT: u32 = 15;
}

/// Vertex pulling: a vertex program reads its generic inputs from guest
/// memory itself, through the page table, as the console's hardware
/// fetches them; so one translation serves every vertex format. The draw's
/// fetch table is a buffer of `VertexAttribute`s, one per generic input
/// location (up to `VERTEX_ATTRIBUTES`), read through
/// PhysicalStorageBuffer64 at `table`.
///
/// For the input at location `n`, `entry = table[n]`:
/// - element index: `VertexIndex` (Vulkan's, base vertex included) when
///   `divisor == 0`, else `base_instance + (InstanceIndex - base_instance) /
///   divisor`;
/// - its bytes start at guest address `entry.address + element *
///   entry.stride`; each 32-bit word (or smaller component, inside its
///   aligned word) is looked up in the page table on its own, so an
///   element may straddle pages; bytes at or past `entry.size` from
///   `entry.address` read as 0;
/// - `entry.format` (`VertexFormat`) says how to decode them. The result is
///   four 32-bit words as the guest's ALD reads them: float bits for
///   normalized, scaled and float formats, the integer for integer formats
///   (sign-extended for signed). Components the format lacks read (0, 0,
///   0, 1), with 1 as 1.0 for float results and 1 for integer ones. An
///   entry with no components (unbound or unformatted) reads (0.0, 0.0,
///   0.0, 1.0).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VertexFetch {
    /// Device address of the draw's `VertexAttribute` table (0: no table;
    /// every input reads as unbound).
    pub table: u64,
    /// The draw's first instance (`InstanceIndex` of instance 0).
    pub base_instance: u32,
    /// Bit n: `table[n]` is present (locations past the table are unbound).
    pub attributes: u32,
}

pub const VERTEX_ATTRIBUTES: usize = 32;

impl VertexAttribute {
    /// The entry as the table stores it.
    pub fn to_bytes(&self) -> [u8; 32] {
        let words = [
            self.address as u32,
            (self.address >> 32) as u32,
            self.stride,
            self.divisor,
            self.format,
            self.size,
            self.reserved[0],
            self.reserved[1],
        ];
        let mut bytes = [0u8; 32];
        for (chunk, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

/// One generic input's source, 32 bytes, std430.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VertexAttribute {
    /// Guest GPU address of element 0's first byte (stream base plus the
    /// attribute's offset).
    pub address: u64,
    /// Bytes from one element to the next (0: every element is the same).
    pub stride: u32,
    /// 0: per vertex; n: per instance, advancing every n instances.
    pub divisor: u32,
    /// `VertexFormat::encode`.
    pub format: u32,
    /// Bytes readable from `address` (the bound stream's size past it).
    pub size: u32,
    pub reserved: [u32; 2],
}

/// How the bytes of one vertex element decode. Encoded as bits 0..3
/// components (0-4), 4..8 `Layout`, 8..12 `Numeric`, bit 12 BGRA order
/// (components 0 and 2 swapped after decoding).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VertexFormat {
    pub components: u32,
    pub layout: VertexLayout,
    pub numeric: VertexNumeric,
    pub bgra: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VertexLayout {
    /// Components of 8, 16 or 32 bits each.
    #[default]
    Bits8 = 0,
    Bits16 = 1,
    Bits32 = 2,
    /// One 32-bit word: 10, 10, 10, 2 bits (RGB10A2), from bit 0.
    Packed1010102 = 3,
    /// One 32-bit word: unsigned 11-, 11-, 10-bit floats (R11G11B10F).
    Packed111110 = 4,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VertexNumeric {
    /// Unsigned normalized to [0, 1].
    #[default]
    Unorm = 0,
    /// Signed normalized to [-1, 1] (the most negative value clamps to -1).
    Snorm = 1,
    /// Unsigned integer, kept as an integer.
    Uint = 2,
    /// Signed integer, kept as an integer.
    Sint = 3,
    /// IEEE half (16-bit) or single (32-bit) float.
    Float = 4,
    /// Unsigned integer converted to float.
    Uscaled = 5,
    /// Signed integer converted to float.
    Sscaled = 6,
}

impl VertexFormat {
    pub fn encode(self) -> u32 {
        self.components & 7
            | (self.layout as u32) << 4
            | (self.numeric as u32) << 8
            | u32::from(self.bgra) << 12
    }

    pub fn decode(word: u32) -> Option<Self> {
        let layout = match word >> 4 & 0xf {
            0 => VertexLayout::Bits8,
            1 => VertexLayout::Bits16,
            2 => VertexLayout::Bits32,
            3 => VertexLayout::Packed1010102,
            4 => VertexLayout::Packed111110,
            _ => return None,
        };
        let numeric = match word >> 8 & 0xf {
            0 => VertexNumeric::Unorm,
            1 => VertexNumeric::Snorm,
            2 => VertexNumeric::Uint,
            3 => VertexNumeric::Sint,
            4 => VertexNumeric::Float,
            5 => VertexNumeric::Uscaled,
            6 => VertexNumeric::Sscaled,
            _ => return None,
        };
        Some(Self {
            components: word & 7,
            layout,
            numeric,
            bgra: word >> 12 & 1 != 0,
        })
    }

    /// Whether components decode to integers (else to float bits).
    pub fn integer(self) -> bool {
        matches!(self.numeric, VertexNumeric::Uint | VertexNumeric::Sint)
    }
}

pub const PAGE_SHIFT: u32 = 12;
pub const PAGE_SIZE: u64 = 1 << PAGE_SHIFT;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_constant_fields_sit_at_their_documented_offsets() {
        assert_eq!(
            std::mem::size_of::<PushConstants>(),
            PUSH_CONSTANTS_SIZE as usize
        );
        assert_eq!(
            std::mem::offset_of!(PushConstants, translator),
            PUSH_TRANSLATOR_OFFSET as usize
        );
        assert_eq!(
            std::mem::offset_of!(PushConstants, vertex_fetch),
            PUSH_VERTEX_FETCH_OFFSET as usize
        );
        assert_eq!(
            std::mem::offset_of!(PushConstants, resolution_scale),
            PUSH_RESOLUTION_SCALE_OFFSET as usize
        );
        assert_eq!(
            std::mem::offset_of!(PushConstants, page_table),
            PUSH_PAGE_TABLE_OFFSET as usize
        );
        assert_eq!(std::mem::size_of::<PageTable>(), 16);
        assert_eq!(std::mem::size_of::<VertexFetch>(), 16);
        assert_eq!(std::mem::size_of::<VertexAttribute>(), 32);
    }

    #[test]
    fn push_bytes_follow_the_struct_layout() {
        let constants = PushConstants {
            vertex_fetch: VertexFetch {
                table: 0x1122_3344_5566_7788,
                base_instance: 9,
                attributes: 3,
            },
            resolution_scale: [2.0, 2.0, 0.5, 0.5],
            page_table: PageTable {
                address: 0xaabb_ccdd_0000_1000,
                first_page: 5,
                page_count: 6,
            },
            ..Default::default()
        };
        let bytes = constants.to_bytes();
        let at = |offset: u32, length: usize| &bytes[offset as usize..offset as usize + length];
        assert_eq!(
            at(PUSH_VERTEX_FETCH_OFFSET, 16),
            [
                0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, 9, 0, 0, 0, 3, 0, 0, 0
            ]
        );
        assert_eq!(at(PUSH_RESOLUTION_SCALE_OFFSET, 4), 2.0f32.to_le_bytes());
        assert_eq!(
            at(PUSH_RESOLUTION_SCALE_OFFSET + 8, 4),
            0.5f32.to_le_bytes()
        );
        assert_eq!(
            at(PUSH_PAGE_TABLE_OFFSET, 16),
            [
                0, 0x10, 0, 0, 0xdd, 0xcc, 0xbb, 0xaa, 5, 0, 0, 0, 6, 0, 0, 0
            ]
        );
        assert!(
            at(0, PUSH_VERTEX_FETCH_OFFSET as usize)
                .iter()
                .all(|&b| b == 0)
        );
    }

    #[test]
    fn vertex_formats_round_trip() {
        let format = VertexFormat {
            components: 4,
            layout: VertexLayout::Bits8,
            numeric: VertexNumeric::Unorm,
            bgra: true,
        };
        assert_eq!(VertexFormat::decode(format.encode()), Some(format));
        assert_eq!(VertexFormat::decode(0).map(|f| f.components), Some(0));
    }

    #[test]
    fn descriptors_match_the_bridge_layout() {
        // 15 u32 fields, as EchoraShaderDescriptor.
        assert_eq!(std::mem::size_of::<Descriptor>(), 60);
    }
}
