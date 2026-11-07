//! Ahead-of-time translation of AArch64 guest code.
//!
//! [`aarch64`] decodes instructions into a semantic form (scalar forms here,
//! SIMD/FP forms in [`simd`]); [`codegen`] discovers reachable code from
//! entry points, function prologues, relocation targets and jump tables,
//! builds basic blocks, and emits Rust that operates on
//! [`echora_guest::Core`]. The host toolchain compiles that Rust to native
//! code for whichever architecture it targets.

// Ported code keeps its original structure; these lints flag style only.
#![allow(missing_debug_implementations)]

pub mod aarch64;
pub mod codegen;
pub mod simd;
