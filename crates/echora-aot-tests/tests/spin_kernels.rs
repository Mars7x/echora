//! The spin-loop detector on the benchmark kernels: of the seven, only the
//! polling loop waits for another thread.

use std::collections::BTreeSet;

use echora_aot_tests::kernels::{self, KERNEL_BASE, KERNEL_ENTRIES};
use echora_lift::codegen::{CodeRegion, discover_with_tables};
use echora_lift::spin::{SpinLoop, find_spin_loops};

#[test]
fn only_the_polling_kernel_spins() {
    let image = CodeRegion {
        name: "kernels".into(),
        base: KERNEL_BASE,
        bytes: kernels::image(),
    };
    let entries: Vec<u64> = KERNEL_ENTRIES
        .iter()
        .map(|(_, offset)| KERNEL_BASE + offset)
        .collect();
    let translation = discover_with_tables(
        std::slice::from_ref(&image),
        std::slice::from_ref(&image),
        &entries,
        &[],
        &BTreeSet::new(),
        4096,
    )
    .expect("lifting the kernels");
    let poll = KERNEL_ENTRIES
        .iter()
        .find(|(name, _)| *name == "poll")
        .map(|(_, offset)| KERNEL_BASE + offset)
        .expect("a poll kernel");
    // poll: ldar w8, [x0]; cbz w8, poll.
    assert_eq!(
        find_spin_loops(&translation.blocks),
        vec![SpinLoop {
            head: poll,
            back_edges: vec![poll + 4],
        }]
    );
}
