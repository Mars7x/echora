//! Size and codec distribution of a package's RomFS files.
#![allow(clippy::unwrap_used)]

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: romfs_stats GAME.nup");
    let a = echora_nup::NupArchive::open(path).unwrap();
    for p in a.packages() {
        let Some(romfs) = p.romfs else { continue };
        let files = a.walk_files(romfs);
        let mut sizes: Vec<(u64, bool)> = files
            .iter()
            .map(|(_, id)| {
                let f = a.file(*id);
                (f.size, f.codec == echora_nup::Codec::Zstd)
            })
            .collect();
        sizes.sort_unstable();
        let zstd = sizes.iter().filter(|s| s.1).count();
        let pct = |q: f64| sizes[((sizes.len() - 1) as f64 * q) as usize].0;
        println!("files {} (zstd {zstd})", sizes.len());
        println!(
            "size p50 {} p90 {} p99 {} max {}",
            pct(0.5),
            pct(0.9),
            pct(0.99),
            pct(1.0)
        );
        let big_zstd: Vec<_> = sizes.iter().filter(|s| s.1 && s.0 > 64 << 20).collect();
        println!("zstd files over 64 MiB: {}", big_zstd.len());
        let mut largest: Vec<_> = files
            .iter()
            .map(|(p, id)| (a.file(*id).size, a.file(*id).codec, p))
            .collect();
        largest.sort_unstable_by_key(|x| std::cmp::Reverse(x.0));
        for (size, codec, path) in largest.iter().take(8) {
            println!("  {size:>12} {codec:?} {path}");
        }
    }
}
