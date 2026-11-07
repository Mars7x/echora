// Original synthetic instruction sequences, never copied from a game.
// Shared by `build.rs`
// (which lifts them) and the tests (which run them).

/// `(name, words)`; each program is loaded at `CODE_BASE` and entered there.
pub const PROGRAMS: &[(&str, &[u32])] = &[
    (
        "memory_and_flags",
        &[
            0xd2900000, 0xd2800521, 0x11000421, 0xb9000001, 0xb9400002, 0x7100a85f, 0x54000041,
            0xd40000e1, 0xd4200000,
        ],
    ),
    (
        "call_return",
        &[0x94000004, 0x14000006, 0xd4200000, 0xd4200000, 0x91002463, 0xd65f03c0, 0xd4200000, 0xd40000e1],
    ),
    (
        "pauth_disabled_call_return",
        &[
            0x94000004, 0x14000008, 0xd4200000, 0xd4200000, 0xd503237f, 0x91002463, 0xd50323ff,
            0xd65f0fff, 0xd4200000, 0xd40000e1,
        ],
    ),
    ("bounded_loop", &[0x14000000]),
    (
        "long_multiply_and_conditional_compare",
        &[0x9ba20c20, 0xfa440000, 0x54000061, 0x7a45180a, 0xd40000e1, 0xd4200000],
    ),
    ("vector_stack_roundtrip", &[0xadbf07e0, 0xacc10fe2, 0xd40000e1]),
    ("strip_lr_pac", &[0xd50320ff, 0xd40000e1]),
    ("unsupported_virtual_counter", &[0xd53be042, 0xd40000e1]),
    // BR X1: the target is chosen at run time and was never discovered.
    ("indirect_missing_block", &[0xd61f0020, 0xd40000e1]),
    ("cas_byte", &[0x08e1fc02, 0xd40000e1]),
    ("cas_halfword", &[0x48a1fc02, 0xd40000e1]),
    ("cas_word", &[0x88e87c14, 0xd40000e1]),
    ("cas_doubleword", &[0xc8e1fc02, 0xd40000e1]),
    ("cas_alias_compare_store", &[0xc8a17c01, 0xd40000e1]),
    ("cas_alias_compare_base", &[0xc8a07c01, 0xd40000e1]),
    ("cas_zero_compare_sp", &[0x88ffffe2, 0xd40000e1]),
    ("cas_zero_store_sp", &[0x88a1ffff, 0xd40000e1]),
    ("ldadd_byte", &[0x38210002, 0xd40000e1]),
    ("ldadd_halfword", &[0x78a10002, 0xd40000e1]),
    ("ldadd_word_alias", &[0xb8e80008, 0xd40000e1]),
    ("ldadd_doubleword", &[0xf8610002, 0xd40000e1]),
    ("ldadd_alias_base_result", &[0xf8210000, 0xd40000e1]),
    ("ldadd_alias_base_source", &[0xf8200002, 0xd40000e1]),
    ("ldadd_zero_source_sp", &[0xf8ff03e2, 0xd40000e1]),
    ("ldadd_discard_sp", &[0xf8e103ff, 0xd40000e1]),
    ("exclusive_increment_loop", &[0xc85ffc01, 0x91000421, 0xc802fc01, 0x35ffffa2, 0xd40000e1]),
    ("exclusive_plain_store_interference", &[0xc85f7c01, 0xf9000003, 0xc8027c01, 0xd40000e1]),
    ("exclusive_clrex", &[0xc85f7c01, 0xd503305f, 0xc8027c01, 0xd40000e1]),
    ("exclusive_byte_sp", &[0x085f7fe5, 0x08067fe7, 0xd40000e1]),
    ("exclusive_halfword_zero", &[0x485ffc01, 0x4802fc1f, 0xd40000e1]),
    ("exclusive_load_base_alias", &[0xc85f7c00, 0xd40000e1]),
    ("ldapr_byte_alias", &[0x38bfc108, 0xd40000e1]),
    ("ldapr_halfword", &[0x78bfc002, 0xd40000e1]),
    ("ldapr_word", &[0xb8bfc002, 0xd40000e1]),
    ("ldapr_doubleword", &[0xf8bfc002, 0xd40000e1]),
    ("ldapr_discard", &[0xb8bfc01f, 0xd40000e1]),
    ("ldapr_sp", &[0xf8bfc3e2, 0xd40000e1]),
    ("ldar_byte_alias", &[0x08dffc63, 0xd40000e1]),
    ("ldar_halfword", &[0x48dffc25, 0xd40000e1]),
    ("ldar_word", &[0x88dffc25, 0xd40000e1]),
    ("ldar_doubleword", &[0xc8dffc25, 0xd40000e1]),
    ("ldar_discard", &[0x88dffc3f, 0xd40000e1]),
    ("ldar_sp", &[0xc8dfffe5, 0xd40000e1]),
    ("stlr_byte", &[0x089ffc15, 0xd40000e1]),
    ("stlr_halfword", &[0x489ffc02, 0xd40000e1]),
    ("stlr_word", &[0x889ffc02, 0xd40000e1]),
    ("stlr_doubleword", &[0xc89ffc02, 0xd40000e1]),
    ("stlr_alias", &[0xc89ffc00, 0xd40000e1]),
    ("stlr_zero_sp", &[0xc89fffff, 0xd40000e1]),
    // Forms found unsupported by `echora analyze` on the real game.
    (
        "crc_and_flags",
        &[
            0x1ac14000, // CRC32B W0, W0, W1
            0x1ac644a5, // CRC32H W5, W5, W6
            0x9ac35c42, // CRC32CX W2, W2, X3
            0xd53b4204, // MRS X4, NZCV
            0xd40000e1,
        ],
    ),
    // MSR NZCV, X6; MRS X7, NZCV
    ("nzcv_round_trip", &[0xd51b4206, 0xd53b4207, 0xd40000e1]),
    ("casp_x", &[0x48227c24, 0xd40000e1]),   // CASP X2, X3, X4, X5, [X1]
    ("caspal_w", &[0x0862fc24, 0xd40000e1]), // CASPAL W2, W3, W4, W5, [X1]
    ("caspa_alias", &[0x48627c22, 0xd40000e1]), // CASPA X2, X3, X2, X3, [X1]
    (
        "abd_long_and_dot",
        &[
            0x0e635040, // SABAL V0.4S, V2.4H, V3.4H
            0x2e2770c5, // UABDL V5.8H, V6.8B, V7.8B
            0x6e839404, // UDOT V4.4S, V0.16B, V3.16B
            0x0e8a9528, // SDOT V8.2S, V9.8B, V10.8B
            0xd40000e1,
        ],
    ),
    (
        "fp_across",
        &[
            0x6e30f820, // FMAXV S0, V1.4S
            0x6eb0f823, // FMINV S3, V1.4S
            0x6e30c824, // FMAXNMV S4, V1.4S
            0x6eb0c822, // FMINNMV S2, V1.4S
            0xd40000e1,
        ],
    ),
];

pub const CODE_BASE: u64 = 0x1000;
