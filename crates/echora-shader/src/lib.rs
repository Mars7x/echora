//! Translation of the Switch 2 GPU's shader programs (NVIDIA Ampere, SM 8.x
//! SASS) to SPIR-V, ahead of time and without runtime state. See
//! `docs/shader-translator.md` for the design and its status.
//!
//! So far: [`decode`], 128-bit instruction words into typed instructions.
//! [`translate`] has the runtime's signature (`echora_shader_abi::Translate`)
//! and translates no program yet, so every program still goes to Lichii.

pub mod decode;
pub mod spirv;

pub use decode::{Instruction, Op, decode, decode_program};
use echora_shader_abi::{Parameters, Translate, Translated, Unsupported};

/// Translates one program (its bytes as NVN stores them at the stage's GPU
/// address) to SPIR-V: a pure function, callable from many threads at once.
/// Not implemented yet: every program is `Unsupported`.
pub fn translate(_program: &[u8], _parameters: &Parameters) -> Result<Translated, Unsupported> {
    Err(Unsupported {
        reason: "echora-shader does not translate programs yet (decoder only)".into(),
        offset: None,
    })
}

/// `translate` is the runtime's translator signature.
const _: Translate = translate;
