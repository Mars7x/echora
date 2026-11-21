//! Host CPU placement for the GPU replay thread.
//!
//! The replay thread turns every submitted command list into Vulkan work
//! and is the longest chain of a 3D frame. On a host with simultaneous
//! multithreading it shares a core with whichever guest worker the
//! scheduler puts on the sibling, and runs slower for it. One core (both of
//! its logical CPUs) is kept for it: the replay thread runs there, guest
//! workers and the threads they start run on the other cores. Forest scene,
//! Ryzen 7 5800X3D: 19.2-19.4 frames a second without, 19.8-20.0 with.
//!
//! Only on Linux, with at least eight logical CPUs (a smaller host has no
//! core to spare). ECHORA_REPLAY_CORE=0 leaves placement to the scheduler.
// sched_setaffinity is a libc call.
#![allow(unsafe_code)]

/// The logical CPUs of the core kept for the replay thread, and every
/// other online CPU; `None` when no core is kept.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Reserved {
    pub kept: Vec<usize>,
    pub others: Vec<usize>,
}

/// The core to keep: the one holding the highest-numbered online CPU (CPU 0
/// takes most interrupts), given the online CPUs and each CPU's siblings.
fn choose(online: &[usize], siblings: impl Fn(usize) -> Option<Vec<usize>>) -> Option<Reserved> {
    if online.len() < 8 {
        return None;
    }
    let last = *online.iter().max()?;
    let mut kept = siblings(last).unwrap_or_else(|| vec![last]);
    kept.retain(|cpu| online.contains(cpu));
    kept.sort_unstable();
    let others: Vec<usize> = online
        .iter()
        .copied()
        .filter(|cpu| !kept.contains(cpu))
        .collect();
    (!kept.is_empty() && !others.is_empty()).then_some(Reserved { kept, others })
}

/// A CPU list as sysfs writes it: "0-3,8,10-11".
fn parse_list(text: &str) -> Option<Vec<usize>> {
    let mut cpus = Vec::new();
    for part in text.trim().split(',').filter(|part| !part.is_empty()) {
        match part.split_once('-') {
            Some((first, last)) => {
                let (first, last): (usize, usize) = (first.parse().ok()?, last.parse().ok()?);
                if last < first || last - first > 4096 {
                    return None;
                }
                cpus.extend(first..=last);
            }
            None => cpus.push(part.parse().ok()?),
        }
    }
    Some(cpus)
}

/// The core kept for the replay thread on this host.
pub(crate) fn reserved() -> Option<Reserved> {
    static RESERVED: std::sync::OnceLock<Option<Reserved>> = std::sync::OnceLock::new();
    RESERVED
        .get_or_init(|| {
            if !cfg!(target_os = "linux")
                || crate::nvn::debug_var_os("ECHORA_REPLAY_CORE").is_some_and(|value| value == "0")
            {
                return None;
            }
            let read = |path: &str| std::fs::read_to_string(path).ok();
            let online = parse_list(&read("/sys/devices/system/cpu/online")?)?;
            choose(&online, |cpu| {
                parse_list(&read(&format!(
                    "/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
                ))?)
            })
        })
        .clone()
}

/// Restrict the calling thread to `cpus` (threads it starts inherit that).
/// A refusal (a container's CPU limit) leaves placement as it was.
pub(crate) fn run_on(cpus: &[usize]) {
    #[cfg(target_os = "linux")]
    // SAFETY: the set is zeroed and filled within its capacity, and the
    // call reads it for the calling thread only.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        let capacity = 8 * std::mem::size_of::<libc::cpu_set_t>();
        for &cpu in cpus.iter().filter(|&&cpu| cpu < capacity) {
            libc::CPU_SET(cpu, &mut set);
        }
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
    #[cfg(not(target_os = "linux"))]
    let _ = cpus;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sysfs_lists_parse() {
        assert_eq!(
            parse_list("0-3,8,10-11\n"),
            Some(vec![0, 1, 2, 3, 8, 10, 11])
        );
        assert_eq!(parse_list("5"), Some(vec![5]));
        assert_eq!(parse_list("3-1"), None);
        assert_eq!(parse_list("x"), None);
    }

    #[test]
    fn the_last_core_is_kept_with_its_sibling() {
        let online: Vec<usize> = (0..16).collect();
        let siblings = |cpu: usize| Some(vec![cpu % 8, cpu % 8 + 8]);
        let reserved = choose(&online, siblings).unwrap();
        assert_eq!(reserved.kept, vec![7, 15]);
        assert_eq!(reserved.others.len(), 14);
        assert!(!reserved.others.contains(&7) && !reserved.others.contains(&15));
    }

    #[test]
    fn small_hosts_keep_no_core() {
        let online: Vec<usize> = (0..6).collect();
        assert_eq!(choose(&online, |cpu| Some(vec![cpu])), None);
    }

    #[test]
    fn a_host_without_smt_keeps_one_cpu() {
        let online: Vec<usize> = (0..8).collect();
        let reserved = choose(&online, |_| None).unwrap();
        assert_eq!(reserved.kept, vec![7]);
        assert_eq!(reserved.others, (0..7).collect::<Vec<_>>());
    }
}
