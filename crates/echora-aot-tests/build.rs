//! Lifts the synthetic programs into Rust modules compiled with this crate.

use std::fmt::Write as _;
use std::path::PathBuf;

use echora_lift::codegen::{
    CodeRegion, discover, discover_with_tables, emit_rust, emit_rust_split,
};

include!("src/programs.rs");

mod kernels {
    include!("src/kernels.rs");
}

fn region(words: &[u32]) -> CodeRegion {
    CodeRegion {
        name: "synthetic".into(),
        base: CODE_BASE,
        bytes: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

fn module(out: &mut String, name: &str, generated: &str) {
    writeln!(
        out,
        "pub mod {name} {{\n{generated}\n\
         pub fn run(m: &mut crate::Machine) -> Result<(), Trap> {{ m.run(dispatch) }}\n}}"
    )
    .expect("writing to a String cannot fail");
}

fn main() {
    println!("cargo::rerun-if-changed=src/programs.rs");
    println!("cargo::rerun-if-changed=src/kernels.rs");
    let mut out = String::new();
    for (name, words) in PROGRAMS {
        let translation = discover(&[region(words)], &[CODE_BASE], 100)
            .unwrap_or_else(|e| panic!("lifting {name}: {e:#}"));
        module(&mut out, name, &emit_rust(&translation));
    }

    // A call across a 4 KiB dispatch window, entered from every address.
    let mut words = vec![0xd503201f; 300];
    words[0] = 0x94000100; // BL to a block in the next window.
    words[1] = 0xd40000e1;
    words[256] = 0x91004463; // ADD X3, X3, #17.
    words[257] = 0xd65f03c0;
    let entries: Vec<u64> = (0..words.len()).map(|i| CODE_BASE + i as u64 * 4).collect();
    let translation = discover(&[region(&words)], &entries, 400).expect("sharded_call_return");
    module(&mut out, "sharded_call_return", &emit_rust(&translation));

    // The same program through split emission (the layout real builds use):
    // region sources become child modules named as the root expects.
    let (root, regions) = emit_rust_split(&translation, 10);
    writeln!(out, "pub mod split_call_return {{").expect("write");
    for (region, source) in regions {
        writeln!(out, "pub mod gen_r_{region:x} {{\n{source}\n}}").expect("write");
    }
    writeln!(
        out,
        "{root}\npub fn run(m: &mut crate::Machine) -> Result<(), Trap> {{ install(); m.run(dispatch) }}\n}}"
    )
    .expect("write");

    // A hole inside a window must trap, not fall through.
    let words = vec![0x14000000; 300];
    let entries: Vec<u64> = (0..words.len())
        .filter(|i| *i != 64)
        .map(|i| CODE_BASE + i as u64 * 4)
        .collect();
    let translation = discover(&[region(&words)], &entries, 400).expect("sharded_hole");
    module(&mut out, "sharded_hole", &emit_rust(&translation));

    kernel_code(&mut out);

    let path = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR")).join("programs.rs");
    std::fs::write(path, out).expect("write generated programs");
}

/// The benchmark kernels, lifted together through split emission (as real
/// builds are) and discovered with their jump table.
fn kernel_code(out: &mut String) {
    let image = CodeRegion {
        name: "kernels".into(),
        base: kernels::KERNEL_BASE,
        bytes: kernels::image(),
    };
    let entries: Vec<u64> = kernels::KERNEL_ENTRIES
        .iter()
        .map(|(_, offset)| kernels::KERNEL_BASE + offset)
        .collect();
    let translation = discover_with_tables(
        std::slice::from_ref(&image),
        std::slice::from_ref(&image),
        &entries,
        &[],
        &Default::default(),
        4096,
    )
    .expect("lifting the kernels");
    let report = translation.report();
    assert!(
        report.unsupported.is_empty() && report.unresolved_targets.is_empty(),
        "kernels must lift completely: {:?} {:?}",
        report.unsupported,
        report.unresolved_targets
    );
    assert_eq!(report.jump_tables, 1, "the switch kernel's jump table");
    let (root, regions) = emit_rust_split(&translation, 10);
    writeln!(out, "pub mod kernel_code {{").expect("write");
    for (region, source) in regions {
        writeln!(out, "pub mod gen_r_{region:x} {{\n{source}\n}}").expect("write");
    }
    writeln!(out, "{root}\n}}").expect("write");
}
