//! GPU time per submission (ECHORA_GPU_TIME=1): a timestamp at the start
//! and end of each batch, and every few seconds a line on stderr with the
//! time the GPU was busy, against wall time and against the time the CPU
//! waited for it. Tells whether a phase is limited by the GPU or the CPU.

use ash::vk;

/// How often a summary line is written.
const REPORT_EVERY: std::time::Duration = std::time::Duration::from_secs(5);

pub(crate) struct GpuTimer {
    pool: vk::QueryPool,
    /// Nanoseconds per timestamp tick.
    period: f64,
    /// The fence of each submission slot: slot `i` owns queries 2i, 2i+1.
    fences: [vk::Fence; 2],
    /// End of the last batch counted, in ticks: batches finish in order,
    /// and one may start before the previous one ends, so busy time is
    /// the union of the intervals.
    last_end: u64,
    busy_ns: f64,
    submits: u64,
    since: std::time::Instant,
    waited_at_start: u64,
}

impl GpuTimer {
    /// A timer when ECHORA_GPU_TIME is set and the queue has timestamps.
    pub(crate) fn from_env(
        device: &ash::Device,
        period: f32,
        valid_bits: u32,
        fences: [vk::Fence; 2],
    ) -> Option<Self> {
        if std::env::var_os("ECHORA_GPU_TIME").is_none_or(|value| value == "0") || valid_bits == 0 {
            return None;
        }
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(4);
        // SAFETY: a valid create info on a live device; destroyed in Drop.
        let pool = unsafe { device.create_query_pool(&info, None) }.ok()?;
        Some(Self {
            pool,
            period: f64::from(period),
            fences,
            last_end: 0,
            busy_ns: 0.0,
            submits: 0,
            since: std::time::Instant::now(),
            waited_at_start: 0,
        })
    }

    fn first_query(&self, fence: vk::Fence) -> Option<u32> {
        self.fences
            .iter()
            .position(|known| *known == fence)
            .map(|slot| slot as u32 * 2)
    }

    /// At the start of the slot's recording.
    pub(crate) fn begin(
        &self,
        device: &ash::Device,
        commands: vk::CommandBuffer,
        fence: vk::Fence,
    ) {
        let Some(first) = self.first_query(fence) else {
            return;
        };
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
    pub(crate) fn finished(&mut self, device: &ash::Device, fence: vk::Fence, waited_us: u64) {
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
        if read.is_ok() && ticks[1] >= ticks[0] {
            let start = ticks[0].max(self.last_end);
            if ticks[1] > start {
                self.busy_ns += (ticks[1] - start) as f64 * self.period;
            }
            self.last_end = self.last_end.max(ticks[1]);
            self.submits += 1;
        }
        let elapsed = self.since.elapsed();
        if elapsed >= REPORT_EVERY {
            let wall_ms = elapsed.as_secs_f64() * 1e3;
            let busy_ms = self.busy_ns / 1e6;
            let waited_ms = waited_us.saturating_sub(self.waited_at_start) as f64 / 1e3;
            eprintln!(
                "gpu time: busy {busy_ms:.0} ms of {wall_ms:.0} ms ({:.0}%), {} submits, CPU waited {waited_ms:.0} ms",
                busy_ms * 100.0 / wall_ms,
                self.submits
            );
            self.since = std::time::Instant::now();
            self.busy_ns = 0.0;
            self.submits = 0;
            self.waited_at_start = waited_us;
        }
    }

    pub(crate) fn destroy(&self, device: &ash::Device) {
        // SAFETY: no submission using the pool is pending (Drop flushed).
        unsafe { device.destroy_query_pool(self.pool, None) };
    }
}
