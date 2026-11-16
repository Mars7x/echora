//! The census of the benchmark kernels' emitted code
//! (`echora_lift::census`, `kernel_code::KERNEL_CENSUS`): printed with
//! `--nocapture`, and checked where the counts are known from the kernels'
//! source (`src/kernels.rs`).
#![allow(clippy::unwrap_used)]

use echora_aot_tests::kernel_code::KERNEL_CENSUS;

/// The census row of `label`, by column.
fn row(label: &str) -> Vec<&'static str> {
    KERNEL_CENSUS
        .lines()
        .find(|line| line.starts_with(&format!("| {label} |")))
        .unwrap_or_else(|| panic!("no row for {label}"))
        .split('|')
        .map(str::trim)
        .filter(|cell| !cell.is_empty())
        .collect()
}

#[test]
fn the_kernel_census_matches_the_kernels() {
    println!("{KERNEL_CENSUS}");
    // ldst: LDR X, ADD, STR X per iteration: one 8-byte load and store.
    let ldst = row("ldst");
    assert_eq!(ldst[1], "0/0/0/1");
    assert_eq!(ldst[2], "0/0/0/1");
    // calls: two BLs, both through `Core::call`.
    assert_eq!(row("calls")[4], "2");
    // intvec: no helper call left in the default kernel lift is required
    // here; its inlined operations are counted by option only when the
    // features inline them, so only the row's presence is checked.
    assert_eq!(row("intvec").len(), 12);
}
