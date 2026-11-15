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
        0xffb3_c2a1_43ad_3f4e,
        "{:#018x}",
        golden::DEFAULT_OUTPUT
    );
}

#[test]
fn inline_vector_emission_is_unchanged() {
    assert_eq!(
        golden::INLINE_VECTOR_OUTPUT,
        0xdef6_1b9e_c945_4e73,
        "{:#018x}",
        golden::INLINE_VECTOR_OUTPUT
    );
}
