//! Ahead-of-time translation of AArch64 guest code.
//!
//! [`aarch64`] decodes instructions into a semantic form (scalar forms here,
//! SIMD/FP forms in [`simd`]); [`codegen`] discovers reachable code from
//! entry points, function prologues, relocation targets and jump tables,
//! builds basic blocks, and emits Rust that operates on
//! [`echora_guest::Core`]. The host toolchain compiles that Rust to native
//! code for whichever architecture it targets.
//!
//! Analyses over a translation, none of which changes what is emitted
//! unless codegen asks for it:
//! - [`defuse`]: the registers, flags and memory each instruction reads
//!   and writes;
//! - [`spin`]: loops that only another thread can end (pure polls), which
//!   codegen marks with a spin hint;
//! - [`structure`]: call sites and their targets, natural loops, and a
//!   structured (loop, if, block) tree for every window, which codegen's
//!   optional structured emission writes out;
//! - [`dispatch`]: how every BR and BLR gets its target, and which look
//!   like jump-table dispatches whose table was not recovered.

// Ported code keeps its original structure; these lints flag style only.
#![allow(missing_debug_implementations)]

// The tests execute guest SIMD/FP operations, which echora-simd provides.
#[cfg(test)]
extern crate echora_simd;

pub mod aarch64;
pub mod codegen;
pub mod defuse;
pub mod dispatch;
pub mod liveness;
pub mod simd;
pub mod spin;
pub mod structure;
