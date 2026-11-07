//! Process creation: map the executable modules, the main stack and thread
//! local storage, and set the registers for the first instruction (rtld's
//! entry). Relocation and module initialization are left to the guest rtld.
//!
//! Module placement comes from `echora_loader::process`, the same
//! placement the lifter used.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use echora_loader::process::{self, PlacedModule};

use crate::memory::{
    EXECUTE, PAGE_SIZE, READ, STATE_CODE, STATE_CODE_DATA, STATE_STACK, STATE_TLS, WRITE,
};
use crate::npdm::Npdm;
use crate::runtime::{Layout, Machine, RemoteMemoryModel, Trap};

pub struct LoadedProgram {
    pub exefs_directory: PathBuf,
    pub modules: Vec<PlacedModule>,
    pub npdm: Npdm,
    pub machine: Machine,
}

fn page_up(value: u64) -> Result<u64> {
    Ok(value
        .checked_add(PAGE_SIZE - 1)
        .context("page rounding overflow")?
        & !(PAGE_SIZE - 1))
}

fn process_scheduling_masks(npdm: &Npdm) -> Result<(u64, u64)> {
    let capabilities = &npdm.kernel_capabilities;
    ensure!(
        capabilities.kernel_flags_ceiling_validated
            && capabilities.aci0.first_kernel_flags.is_some()
            && capabilities.acid.first_kernel_flags.is_some(),
        "bootstrap requires an ACI0 KernelFlags grant validated against its ACID ceiling"
    );
    let core_mask = capabilities
        .active_core_mask
        .filter(|mask| *mask != 0)
        .context("bootstrap requires a nonzero validated process core mask")?;
    let priority_mask = capabilities
        .active_priority_mask
        .filter(|mask| *mask != 0)
        .context("bootstrap requires a nonzero validated process priority mask")?;
    ensure!(
        npdm.main_thread_core < 64 && core_mask & (1u64 << npdm.main_thread_core) != 0,
        "main thread core is not permitted by the NPDM"
    );
    ensure!(
        npdm.main_thread_priority < 64 && priority_mask & (1u64 << npdm.main_thread_priority) != 0,
        "main thread priority is not permitted by the NPDM"
    );
    Ok((core_mask, priority_mask))
}

fn env_number<T: std::str::FromStr>(name: &str) -> Option<T> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
}

/// Loads the ExeFS in `exefs` into a new process with an instruction budget
/// of `max_steps`.
/// Addresses of the exports the runtime replaces with host functions
/// (`nvn::HOOKED_EXPORTS`). A hook only runs where no translated block
/// exists, so translation must leave these addresses out.
pub fn hooked_export_addresses(placed: &[PlacedModule]) -> Result<Vec<u64>> {
    let mut addresses = Vec::new();
    for name in crate::nvn::HOOKED_EXPORTS {
        for p in placed {
            if let Some(offset) = echora_loader::defined_symbol(&p.module, name)? {
                addresses.push(p.base + offset);
            }
        }
    }
    Ok(addresses)
}

/// Guest functions replaced whole by host code (`crate::hle`): translation
/// leaves them out, so calls reach the host entry. Unless ECHORA_NO_HLE is
/// set when translating, the translation can only run with them replaced.
pub fn hle_function_addresses(placed: &[PlacedModule]) -> Vec<u64> {
    if std::env::var_os("ECHORA_NO_HLE").is_some() {
        return Vec::new();
    }
    placed
        .iter()
        .flat_map(|p| {
            [
                crate::hle::zlib_uncompress_offset(&p.module),
                crate::hle::zstd_decompress_offset(&p.module),
            ]
            .into_iter()
            .flatten()
            .map(|offset| p.base + offset)
        })
        .collect()
}

pub fn load_program(exefs: &Path, max_steps: u64) -> Result<LoadedProgram> {
    let modules = echora_loader::load_exefs(exefs)?;
    let npdm = crate::npdm::load_npdm(&exefs.join("main.npdm"))?;
    if !npdm.is_64bit {
        bail!("only AArch64 programs are supported");
    }
    if npdm.address_space_type != 3 {
        bail!("only NPDM address-space type 3 (39-bit) is supported");
    }
    if npdm.flags0 & 0x40 != 0 {
        bail!("NPDM address-sanitizer alias extensions are not implemented");
    }
    let (core_mask, priority_mask) = process_scheduling_masks(&npdm)?;

    let mut layout = Layout::default();
    ensure!(
        layout.code_base == process::FIRST_MODULE_BASE,
        "runtime and lifter disagree on the first module base"
    );
    let stack_size = page_up(u64::from(npdm.main_thread_stack_size))?;
    if !(PAGE_SIZE..=16 * 1024 * 1024).contains(&stack_size) {
        bail!("NPDM main stack size is outside the supported range");
    }
    layout.stack_size = stack_size;
    let system_resource = u64::from(npdm.system_resource_size);
    ensure!(
        system_resource.is_multiple_of(PAGE_SIZE) && system_resource < layout.total_memory,
        "NPDM SystemResourceSize {system_resource:#x} is not a page multiple below total memory"
    );
    layout.system_resource_size = system_resource;
    layout.stack_base = 0x7ff0000000 - stack_size;
    layout.processor_id = npdm.main_thread_core;
    layout.process_ideal_core_id = npdm.main_thread_core;
    layout.main_thread_priority = npdm.main_thread_priority;
    layout.process_core_mask = Some(core_mask);
    layout.process_priority_mask = Some(priority_mask);
    if npdm.acid_layout == "extended" {
        layout.remote_memory_model = RemoteMemoryModel::AbsentRegionCompatibility;
    }
    // Any block boundary is a valid preemption point; switching at every
    // one makes the host scheduler most of the run time.
    layout.scheduler_slice = env_number("ECHORA_SCHEDULER_SLICE").unwrap_or(20_000);
    layout.workers = env_number::<usize>("ECHORA_WORKERS").unwrap_or(1).max(1);

    let mut machine = Machine::new(layout, max_steps);
    // ECHORA_VIRTUAL_CLOCK=N: the guest counter advances by executed
    // instructions at N per second instead of by host time.
    machine.core.physical_counter.instructions_per_second = env_number("ECHORA_VIRTUAL_CLOCK");
    machine
        .initialize_entropy_from_os()
        .context("initialize process entropy")?;

    let placed = process::place_modules(modules);
    for p in &placed {
        let module = &p.module;
        for (segment, label, permissions, state, bss) in [
            (&module.text, "text", READ | EXECUTE, STATE_CODE, 0),
            (&module.rodata, "rodata", READ, STATE_CODE, 0),
            (
                &module.data,
                "data",
                READ | WRITE,
                STATE_CODE_DATA,
                u64::from(module.bss_size),
            ),
        ] {
            if !u64::from(segment.memory_offset).is_multiple_of(PAGE_SIZE) {
                bail!("{} {label} memory offset is not page aligned", module.name);
            }
            let size = page_up(segment.bytes.len() as u64 + bss)?;
            if size == 0 {
                continue;
            }
            let address = p
                .base
                .checked_add(u64::from(segment.memory_offset))
                .context("segment address overflow")?;
            machine.memory.map(
                address,
                size,
                permissions,
                state,
                format!("{}:{label}", module.name),
            )?;
            machine.memory.initialize(address, &segment.bytes)?;
            if permissions & EXECUTE != 0 {
                machine.memory.protect_translated_code(address, size)?;
            }
            machine.layout.initial_used_memory += size;
        }
    }

    let layout = &machine.layout;
    let (stack_base, stack_size, tls_base) =
        (layout.stack_base, layout.stack_size, layout.tls_base);
    let process_local = layout.process_local_region();
    let main_handle = layout.main_thread_handle;
    machine.memory.map(
        stack_base,
        stack_size,
        READ | WRITE,
        STATE_STACK,
        "main stack",
    )?;
    machine.memory.map(
        tls_base,
        PAGE_SIZE,
        READ | WRITE,
        STATE_TLS,
        "main thread local region",
    )?;
    // The process local region: the process's first thread-local slot, its
    // own zeroed page below the main thread's.
    machine.memory.map(
        process_local,
        PAGE_SIZE,
        READ | WRITE,
        STATE_TLS,
        "process local region",
    )?;
    machine
        .memory
        .write_u32(tls_base + 0x110, main_handle as u32)?;
    // GetInfo used-memory counts code, the main stack and Normal memory.
    machine.layout.initial_used_memory += stack_size;
    machine.register_main_thread_tls()?;

    // Replace the NVN driver's bootstrap export with a host entry; the thunk
    // range stays unmapped.
    for address in hooked_export_addresses(&placed)? {
        machine
            .nvn
            .add_hook(address, crate::nvn::HostEntry::BootstrapLoader);
    }
    // Host replacements for known library functions, matched by exact code.
    if std::env::var_os("ECHORA_NO_HLE").is_none() {
        for p in &placed {
            if let Some(offset) = crate::hle::zlib_uncompress_offset(&p.module) {
                machine
                    .nvn
                    .add_hook(p.base + offset, crate::nvn::HostEntry::ZlibUncompress);
            }
            if let Some(offset) = crate::hle::zstd_decompress_offset(&p.module) {
                machine
                    .nvn
                    .add_hook(p.base + offset, crate::nvn::HostEntry::ZstdDecompress);
            }
            for offset in crate::hle::inflate_finish_sites(&p.module) {
                machine
                    .nvn
                    .add_hook(p.base + offset, crate::nvn::HostEntry::InflateFinishCall);
            }
        }
    }

    let entry = placed
        .first()
        .context("no executable modules")?
        .text_range()
        .start;
    let cpu = &mut machine.core.cpu;
    cpu.sp = stack_base + stack_size;
    cpu.pc = entry;
    cpu.regs[0] = 0;
    cpu.regs[1] = main_handle;
    cpu.tpidrro_el0 = tls_base;
    machine
        .services
        .initialize_main_application(
            machine.layout.process_id,
            npdm.program_id,
            &npdm.file_sha256,
        )
        .context("register the main application identity")?;
    machine
        .services
        .set_service_access(npdm.service_access.clone());
    // ECHORA_ROMFS: the application's data, as an extracted RomFS directory
    // or a NUP read in place (ECHORA_ROMFS_PACKAGE selects the package when
    // more than one has a RomFS).
    if let Some(path) = std::env::var_os("ECHORA_ROMFS") {
        machine
            .services
            .set_data_romfs(open_romfs(Path::new(&path))?);
    }
    Ok(LoadedProgram {
        exefs_directory: exefs
            .canonicalize()
            .context("resolve the ExeFS directory")?,
        modules: placed,
        npdm,
        machine,
    })
}

fn open_romfs(path: &Path) -> Result<crate::romfs::VirtualRomfs> {
    use crate::romfs::{FileSource, NupFiles, VirtualRomfs};
    if path.is_dir() {
        let listing = crate::romfs::list_directory(path)?;
        return VirtualRomfs::build(path, &listing).context("build the RomFS image");
    }
    let archive = std::sync::Arc::new(echora_nup::NupArchive::open(path)?);
    let with_romfs: Vec<usize> = archive
        .packages()
        .iter()
        .filter(|p| p.romfs.is_some())
        .map(|p| p.index)
        .collect();
    let package = match env_number::<usize>("ECHORA_ROMFS_PACKAGE") {
        Some(index) => index,
        None => match with_romfs.as_slice() {
            [only] => *only,
            _ => bail!(
                "the NUP has {} packages with a RomFS; set ECHORA_ROMFS_PACKAGE",
                with_romfs.len()
            ),
        },
    };
    let (files, listing) = NupFiles::open(archive, package)?;
    VirtualRomfs::build_from(FileSource::Nup(files), &listing).context("build the RomFS image")
}

impl LoadedProgram {
    /// Module and offset of a guest address.
    pub fn describe_address(&self, address: u64) -> Option<serde_json::Value> {
        self.modules
            .iter()
            .find(|m| address >= m.base && address < m.base + m.module.image_size)
            .map(|m| {
                serde_json::json!({
                    "module": m.module.name,
                    "base": m.base,
                    "offset": address - m.base,
                })
            })
    }

    /// The machine's report for `stop`, with code locations resolved.
    pub fn run_report(&self, stop: &Trap) -> serde_json::Value {
        let mut report = self.machine.report(stop);
        report["pc_location"] = self
            .describe_address(self.machine.core.cpu.pc)
            .unwrap_or(serde_json::Value::Null);
        report["pc_samples"] = self
            .machine
            .top_pc_samples(64)
            .into_iter()
            .map(|(thread, pc, hits)| {
                serde_json::json!({
                    "thread": thread,
                    "pc": pc,
                    "hits": hits,
                    "location": self.describe_address(pc),
                })
            })
            .collect();
        report
    }
}
