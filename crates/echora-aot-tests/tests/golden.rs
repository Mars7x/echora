//! The default emission (and inline_vector alone, as game builds use it)
//! must not change by accident: a change rebuilds every region of the
//! game. build.rs hashes the Rust that split emission writes for every
//! program it lifts. If this fails after an intended change to default
//! output, update the constants, and say so in the commit message.
//! Options that are off by default (structured) must leave these
//! unchanged.

use echora_aot_tests::golden;

#[test]
fn default_emission_is_unchanged() {
    assert_eq!(
        golden::DEFAULT_OUTPUT,
        0xa37a_b0a5_259f_63a4,
        "{:#018x}",
        golden::DEFAULT_OUTPUT
    );
}

#[test]
fn inline_vector_emission_is_unchanged() {
    assert_eq!(
        golden::INLINE_VECTOR_OUTPUT,
        0xd413_6437_5dd2_f425,
        "{:#018x}",
        golden::INLINE_VECTOR_OUTPUT
    );
}
