//! The interface between Echora's shader translators and the runtime that
//! runs their programs: what a translator is given, what it returns, and
//! the run-time inputs (push constants, the GPU page table) its SPIR-V may
//! read. Both translators produce these types: the C++ recompiler through
//! `echora-gpu`'s bridge, and the Rust translator (`echora-shader`) directly.
//! No dependencies, so a translator need not link the Vulkan backend.

/// Changes whenever anything a translated program relies on changes (the
/// push-constant layout, the page-table format, descriptor numbering). It
/// is stored in every `Translated` and is part of every shader cache key.
pub const ABI_VERSION: u32 = 1;

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
}

/// The push-constant block every translated program shares, 128 bytes at
/// offset 0 (the size Vulkan guarantees). Field offsets are fixed by the
/// ABI; a program declares only the ranges it reads.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PushConstants {
    /// Bytes 0..96: words of the translator's own (the C++ recompiler's
    /// texture rescaling bits, say).
    pub translator: [u32; 24],
    /// Bytes 96..112: render-target resolution scale (x, y, inverse x,
    /// inverse y).
    pub resolution_scale: [f32; 4],
    /// Bytes 112..128: the GPU page table (`PageTable`).
    pub page_table: PageTable,
}

/// `Translated::push_fields` bits.
pub const PUSH_TRANSLATOR: u32 = 1;
pub const PUSH_RESOLUTION_SCALE: u32 = 2;
pub const PUSH_PAGE_TABLE: u32 = 4;

pub const PUSH_TRANSLATOR_OFFSET: u32 = 0;
pub const PUSH_RESOLUTION_SCALE_OFFSET: u32 = 96;
pub const PUSH_PAGE_TABLE_OFFSET: u32 = 112;
pub const PUSH_CONSTANTS_SIZE: u32 = 128;

/// Global memory a program reaches through pointers. The table is a buffer
/// of u64 device addresses, one per `PAGE_SIZE` page of guest GPU virtual
/// address space from page `first_page`; 0 marks an unmapped page. An
/// access to guest address `va` reads `table[(va >> PAGE_SHIFT) -
/// first_page] + (va & (PAGE_SIZE - 1))` through PhysicalStorageBuffer64.
///
/// Only the first byte's page is looked up: guest global accesses are
/// naturally aligned and at most 16 bytes, so none crosses a page. An
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

pub const PAGE_SHIFT: u32 = 16;
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
            std::mem::offset_of!(PushConstants, resolution_scale),
            PUSH_RESOLUTION_SCALE_OFFSET as usize
        );
        assert_eq!(
            std::mem::offset_of!(PushConstants, page_table),
            PUSH_PAGE_TABLE_OFFSET as usize
        );
        assert_eq!(std::mem::size_of::<PageTable>(), 16);
    }

    #[test]
    fn descriptors_match_the_bridge_layout() {
        // 15 u32 fields, as EchoraShaderDescriptor.
        assert_eq!(std::mem::size_of::<Descriptor>(), 60);
    }
}
