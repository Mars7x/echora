//! GPU time per submission (ECHORA_GPU_TIME=1): a timestamp at the start
//! and end of each batch, and every few seconds a line on stderr with the
//! elapsed time covered by submissions, against wall time and CPU waits.
//! ECHORA_GPU_TIME=items adds a ranking of draw and dispatch intervals,
//! which may overlap and include time waiting for earlier commands.

use ash::vk;
use std::collections::BTreeMap;

/// How often a summary line is written.
const REPORT_EVERY: std::time::Duration = std::time::Duration::from_secs(5);

/// Items timed per submission slot (more are not timed).
const ITEMS: u32 = 1024;

/// Per-item timing (ECHORA_GPU_TIME=items).
struct ItemTimes {
    pool: vk::QueryPool,
    /// Each slot's items in recording order: slot `i` owns queries from
    /// `i * 2 * ITEMS`, two per item.
    names: [Vec<String>; crate::renderer::SLOTS],
    /// Whether an item is open (started, not ended) in each slot.
    open: [bool; crate::renderer::SLOTS],
    totals: std::collections::HashMap<String, (f64, u64)>,
}

struct BatchTimes {
    mask: u64,
    last_end: Option<u64>,
    next_serial: u64,
    /// A fence can be collected out of submission order. Account for a
    /// batch once the previous batches have been collected.
    pending: BTreeMap<u64, Option<[u64; 2]>>,
    ticks: u64,
    submits: u64,
}

impl BatchTimes {
    fn new(valid_bits: u32) -> Self {
        Self {
            mask: u64::MAX >> (64 - valid_bits),
            last_end: None,
            next_serial: 1,
            pending: BTreeMap::new(),
            ticks: 0,
            submits: 0,
        }
    }

    fn record(&mut self, serial: u64, interval: Option<[u64; 2]>) {
        if serial != self.next_serial {
            self.pending.insert(serial, interval);
            return;
        }
        self.count(interval);
        while let Some(interval) = self.pending.remove(&self.next_serial) {
            self.count(interval);
        }
    }

    fn count(&mut self, interval: Option<[u64; 2]>) {
        self.next_serial += 1;
        let Some([start, end]) = interval else {
            return;
        };
        let duration = end.wrapping_sub(start) & self.mask;
        // Count only the part beyond the previous end if batches overlap.
        // The elapsed time between compared samples must fit the counter.
        let advance = self
            .last_end
            .map_or(duration, |last| end.wrapping_sub(last) & self.mask);
        self.ticks += duration.min(advance);
        self.last_end = Some(end);
        self.submits += 1;
    }
}

pub(crate) struct GpuTimer {
    pool: vk::QueryPool,
    /// Nanoseconds per timestamp tick.
    period: f64,
    /// The fence of each submission slot: slot `i` owns queries 2i, 2i+1.
    fences: [vk::Fence; crate::renderer::SLOTS],
    batches: BatchTimes,
    since: std::time::Instant,
    waited_at_start: u64,
    items: Option<ItemTimes>,
}

impl GpuTimer {
    /// A timer when ECHORA_GPU_TIME is set and the queue has timestamps.
    pub(crate) fn from_env(
        device: &ash::Device,
        period: f32,
        valid_bits: u32,
        fences: [vk::Fence; crate::renderer::SLOTS],
    ) -> Option<Self> {
        if std::env::var_os("ECHORA_GPU_TIME").is_none_or(|value| value == "0") || valid_bits == 0 {
            return None;
        }
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(2 * crate::renderer::SLOTS as u32);
        // SAFETY: a valid create info on a live device; destroyed in Drop.
        let pool = unsafe { device.create_query_pool(&info, None) }.ok()?;
        let items = (std::env::var_os("ECHORA_GPU_TIME").is_some_and(|value| value == "items"))
            .then(|| {
                let info = vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(2 * crate::renderer::SLOTS as u32 * ITEMS);
                // SAFETY: as above.
                unsafe { device.create_query_pool(&info, None) }.ok()
            })
            .flatten()
            .map(|pool| ItemTimes {
                pool,
                names: std::array::from_fn(|_| Vec::new()),
                open: [false; crate::renderer::SLOTS],
                totals: std::collections::HashMap::new(),
            });
        Some(Self {
            pool,
            period: f64::from(period),
            fences,
            batches: BatchTimes::new(valid_bits),
            since: std::time::Instant::now(),
            waited_at_start: 0,
            items,
        })
    }

    /// Whether draws and dispatches are timed one by one.
    pub(crate) fn timing_items(&self) -> bool {
        self.items.is_some()
    }

    /// Start (`start`) or end the timing of one draw or dispatch named
    /// `name` (ignored when ending). Outside a rendering pass, or inside
    /// one: timestamps may be written in either.
    pub(crate) fn item(
        &mut self,
        device: &ash::Device,
        commands: vk::CommandBuffer,
        fence: vk::Fence,
        name: &str,
        start: bool,
    ) {
        let Some(first) = self.first_query(fence) else {
            return;
        };
        let slot = (first / 2) as usize;
        let Some(items) = self.items.as_mut() else {
            return;
        };
        let index = items.names[slot].len() as u32;
        if start {
            if index >= ITEMS || items.open[slot] {
                return;
            }
            items.names[slot].push(name.to_owned());
            items.open[slot] = true;
        } else if !items.open[slot] {
            return;
        } else {
            items.open[slot] = false;
        }
        let query = slot as u32 * 2 * ITEMS + (index - u32::from(!start)) * 2 + u32::from(!start);
        // SAFETY: recording; the query was reset when the slot began.
        unsafe {
            device.cmd_write_timestamp(
                commands,
                if start {
                    vk::PipelineStageFlags::TOP_OF_PIPE
                } else {
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE
                },
                items.pool,
                query,
            );
        }
    }

    fn first_query(&self, fence: vk::Fence) -> Option<u32> {
        self.fences
            .iter()
            .position(|known| *known == fence)
            .map(|slot| slot as u32 * 2)
    }

    /// At the start of the slot's recording.
    pub(crate) fn begin(
        &mut self,
        device: &ash::Device,
        commands: vk::CommandBuffer,
        fence: vk::Fence,
    ) {
        let Some(first) = self.first_query(fence) else {
            return;
        };
        if let Some(items) = self.items.as_mut() {
            let slot = (first / 2) as usize;
            items.names[slot].clear();
            items.open[slot] = false;
            // SAFETY: recording, outside any rendering pass.
            unsafe {
                device.cmd_reset_query_pool(
                    commands,
                    items.pool,
                    slot as u32 * 2 * ITEMS,
                    2 * ITEMS,
                );
            }
        }
        // SAFETY: recording, outside any rendering pass.
        unsafe {
            device.cmd_reset_query_pool(commands, self.pool, first, 2);
            device.cmd_write_timestamp(
                commands,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                self.pool,
                first,
            );
        }
    }

    /// Just before the slot's command buffer ends.
    pub(crate) fn end(&self, device: &ash::Device, commands: vk::CommandBuffer, fence: vk::Fence) {
        let Some(first) = self.first_query(fence) else {
            return;
        };
        // SAFETY: recording, outside any rendering pass.
        unsafe {
            device.cmd_write_timestamp(
                commands,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.pool,
                first + 1,
            );
        }
    }

    /// After the slot's fence signaled: count its batch, and report when
    /// due. `waited_us` is the renderer's running total of fence waits.
    pub(crate) fn finished(
        &mut self,
        device: &ash::Device,
        fence: vk::Fence,
        serial: u64,
        waited_us: u64,
    ) {
        let Some(first) = self.first_query(fence) else {
            return;
        };
        let mut ticks = [0u64; 2];
        // SAFETY: both queries were written by the batch the fence covers.
        let read = unsafe {
            device.get_query_pool_results(
                self.pool,
                first,
                &mut ticks,
                vk::QueryResultFlags::TYPE_64,
            )
        };
        self.batches.record(serial, read.ok().map(|()| ticks));
        if let Some(items) = self.items.as_mut() {
            let slot = (first / 2) as usize;
            let names = std::mem::take(&mut items.names[slot]);
            // An item left open (its end not recorded) is not counted.
            let complete = names.len() - usize::from(items.open[slot]);
            if complete > 0 {
                let mut ticks = vec![0u64; complete * 2];
                // SAFETY: the batch the fence covers wrote these queries.
                let read = unsafe {
                    device.get_query_pool_results(
                        items.pool,
                        slot as u32 * 2 * ITEMS,
                        &mut ticks,
                        vk::QueryResultFlags::TYPE_64,
                    )
                };
                if read.is_ok() {
                    for (name, pair) in names.iter().zip(ticks.chunks_exact(2)) {
                        let total = items.totals.entry(name.clone()).or_default();
                        let duration = pair[1].wrapping_sub(pair[0]) & self.batches.mask;
                        total.0 += duration as f64 * self.period;
                        total.1 += 1;
                    }
                }
            }
        }
        let elapsed = self.since.elapsed();
        if elapsed >= REPORT_EVERY {
            if let Some(items) = self.items.as_mut() {
                let mut ranked: Vec<_> = items.totals.drain().collect();
                ranked.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
                let top: Vec<String> = ranked
                    .iter()
                    .take(10)
                    .map(|(name, (ns, count))| format!("{name} {:.1} ms/{count}", ns / 1e6))
                    .collect();
                eprintln!("gpu items: overlapping intervals: {}", top.join(", "));
            }
            let wall_ms = elapsed.as_secs_f64() * 1e3;
            let busy_ms = std::mem::take(&mut self.batches.ticks) as f64 * self.period / 1e6;
            let submits = std::mem::take(&mut self.batches.submits);
            let waited_ms = waited_us.saturating_sub(self.waited_at_start) as f64 / 1e3;
            eprintln!(
                "gpu time: busy {busy_ms:.0} ms of {wall_ms:.0} ms ({:.0}%), {submits} submits, CPU waited {waited_ms:.0} ms",
                busy_ms * 100.0 / wall_ms
            );
            self.since = std::time::Instant::now();
            self.waited_at_start = waited_us;
        }
    }

    pub(crate) fn destroy(&self, device: &ash::Device) {
        // SAFETY: no submission using the pool is pending (Drop flushed).
        unsafe {
            device.destroy_query_pool(self.pool, None);
            if let Some(items) = &self.items {
                device.destroy_query_pool(items.pool, None);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BatchTimes;

    #[test]
    fn batch_time_follows_queue_order_across_a_timestamp_wrap() {
        let mut times = BatchTimes::new(8);
        times.record(2, Some([252, 6]));
        assert_eq!(times.ticks, 0);
        times.record(1, Some([240, 254]));
        assert_eq!(times.ticks, 22);
        assert_eq!(times.submits, 2);

        // A report can start while the following submission overlaps it.
        times.ticks = 0;
        times.record(3, Some([4, 12]));
        times.record(5, Some([20, 25]));
        // Advance past a submission whose query results were unavailable.
        times.record(4, None);
        assert_eq!(times.ticks, 11);
        assert_eq!(times.submits, 4);
    }
}
