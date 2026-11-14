//! The default emission (and inline_vector alone, as game builds use it)
//! must not change by accident: a change to region code rebuilds every
//! region of the game (one to the root module only its dispatcher crate,
//! as `lift` rewrites only files that changed). build.rs hashes the Rust that split emission writes for every
//! program it lifts. If this fails after an intended change to default
//! output, update the constants, and say so in the commit message.
//! Options that are off by default (structured) must leave these
//! unchanged.

use echora_aot_tests::golden;

#[test]
fn default_emission_is_unchanged() {
    assert_eq!(
        golden::DEFAULT_OUTPUT,
        0x7e13_40d3_890b_63a1,
        "{:#018x}",
        golden::DEFAULT_OUTPUT
    );
}

#[test]
fn inline_vector_emission_is_unchanged() {
    assert_eq!(
        golden::INLINE_VECTOR_OUTPUT,
        0x6110_55b0_1d16_950e,
        "{:#018x}",
        golden::INLINE_VECTOR_OUTPUT
    );
}
