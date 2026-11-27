//! Measure alternating page-table lengths on a Vulkan device. Pass --whole to
//! supply complete tables, as runtimes without resize patches do.
use echora_gpu::renderer::Renderer;
use echora_gpu::vulkan::Gpu;
use std::time::Instant;

fn main() -> Result<(), String> {
    let whole = std::env::args().skip(1).any(|arg| arg == "--whole");
    let mut renderer = Renderer::new(Gpu::new()?)?;
    let small = 1 << 19;
    let large = 1 << 20;
    // Four and eight MiB exactly, with one remapped entry near the start
    // and one near the end of the larger table. The entries are symbolic
    // addresses; this measures table updates without submitting a shader.
    let mut entries = vec![0; large];
    let (table, _) = renderer.set_page_table(0x100, &entries[..small])?;
    if table.address == 0 {
        return Err("Vulkan buffer device addresses are required".into());
    }
    let iterations = 80;
    let start = Instant::now();
    for step in 0..iterations {
        let count = if step % 2 == 0 { large } else { small };
        entries[17] = (step + 1) as u64 * 4096;
        if count == large {
            entries[large - 1] = entries[17];
        }
        if whole {
            renderer.set_page_table(0x100, &entries[..count])?;
        } else {
            let prefix = (17, &entries[17..18]);
            let tail = (large - 1, &entries[large - 1..]);
            if count == large {
                renderer.patch_page_table(0x100, count, &[prefix, tail])?;
            } else {
                renderer.patch_page_table(0x100, count, &[prefix])?;
            }
        }
        renderer.flush()?;
    }
    let elapsed = start.elapsed();
    let stats = renderer.take_stats();
    let stat = |name| stats.get(name).copied().unwrap_or_default();
    println!(
        "{iterations} resizes in {:.3} ms, {} buffers allocated, {} reused",
        elapsed.as_secs_f64() * 1000.0,
        stat("stat: page table buffers allocated"),
        stat("stat: page table buffers reused"),
    );
    Ok(())
}
