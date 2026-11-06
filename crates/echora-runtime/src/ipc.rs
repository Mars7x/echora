//! Bounded HIPC framing for bootstrap services, including CMIF domains.
//!
//! This module parses messages and constructs replies. It does not allocate
//! handles, trust a client-supplied PID, dispatch services, or execute IPC.
//! One CMIF input AutoSelect buffer is represented by its X/A descriptor pair;
//! this module never reads the referenced guest memory. Bounded CMIF request
//! copy handles are preserved as untrusted values for service-layer validation;
//! this codec never resolves or transfers them. Request move handles,
//! receive/exchange buffers, receive lists, session-close messages, and legacy
//! protocols are explicitly outside this subset. Domain framing is
//! selected by established session state, never guessed from message bytes.
//! CMIF controls remain non-domain frames; the dispatcher chooses which
//! individual controls, object lookups, and object lifetimes it implements.
//! CMIF service replies may contain one copied kernel handle. This only
//! constructs its wire slot; alias ownership remains the caller's responsibility.
//!
//! CMIF replies use the generic SF CMIF output policy. Atmosphere's legacy SM
//! compatibility shim instead uses minimal fixed replies, including an invalid
//! moved-handle slot on GetService errors. That separate compatibility layout
//! is not implemented here; live SM CMIF compatibility remains unverified.
//!
//! Wire references:
//! - <https://switchbrew.org/wiki/IPC_Marshalling>
//! - <https://switchbrew.org/wiki/Services_API>
//! - <https://github.com/switchbrew/libnx/blob/master/nx/include/switch/sf/hipc.h>
//! - <https://github.com/switchbrew/libnx/blob/master/nx/include/switch/sf/cmif.h>
//! - <https://github.com/switchbrew/libnx/blob/master/nx/include/switch/sf/tipc.h>
//! - <https://github.com/switchbrew/libnx/blob/b156b4774fb2df7b93402169509266da7547fc4c/nx/include/switch/sf/hipc.h>
//! - <https://github.com/switchbrew/libnx/blob/b156b4774fb2df7b93402169509266da7547fc4c/nx/include/switch/sf/cmif.h>
//! - <https://github.com/Atmosphere-NX/Atmosphere/blob/6b0c077b4c608a3a2ee712b8dc35fee87ceb62e8/libraries/libstratosphere/include/stratosphere/tipc/impl/tipc_impl_command_serialization.hpp>
//! - <https://github.com/Atmosphere-NX/Atmosphere/blob/6b0c077b4c608a3a2ee712b8dc35fee87ceb62e8/libraries/libstratosphere/include/stratosphere/sf/impl/sf_impl_command_serialization.hpp>
//! - <https://github.com/Atmosphere-NX/Atmosphere/blob/6b0c077b4c608a3a2ee712b8dc35fee87ceb62e8/libraries/libstratosphere/source/sf/cmif/sf_cmif_service_dispatch.cpp>
//! - <https://github.com/Atmosphere-NX/Atmosphere/blob/6b0c077b4c608a3a2ee712b8dc35fee87ceb62e8/libraries/libstratosphere/source/sf/cmif/sf_cmif_domain_service_object.cpp>
//! - <https://github.com/Atmosphere-NX/Atmosphere/blob/6b0c077b4c608a3a2ee712b8dc35fee87ceb62e8/libraries/libstratosphere/include/stratosphere/sf/cmif/sf_cmif_domain_service_object.hpp>
//! - <https://github.com/Atmosphere-NX/Atmosphere/blob/6b0c077b4c608a3a2ee712b8dc35fee87ceb62e8/libraries/libstratosphere/source/sf/hipc/sf_hipc_server_session_manager.cpp>
//! - <https://github.com/Atmosphere-NX/Atmosphere/blob/6b0c077b4c608a3a2ee712b8dc35fee87ceb62e8/stratosphere/sm/source/sm_user_service.cpp>

use anyhow::{Context, Result, bail, ensure};
use serde::Serialize;

pub const TLS_MESSAGE_BUFFER_SIZE: usize = 0x100;
/// Framing limit, not a claim about a console's service-specific buffer limit.
pub const MAX_INPUT_BUFFER_SIZE: u64 = 16 * 1024 * 1024;
/// Bounded domain-object table capacity, matching the inspected SF processor.
/// This is separate from the number of objects a service session can own.
pub const MAX_DOMAIN_OBJECTS: usize = 8;
/// Local codec limit for request copy handles, not a console ABI maximum.
pub const MAX_COPY_HANDLES: usize = 8;
const GUEST_ADDRESS_SPACE_END: u64 = 1 << 39;
/// First four SHA256 bytes of `nn::sm::detail::IUserInterface`, read as LE u32.
pub const SM_CMIF_INTERFACE_ID: u32 = 0xcac4_d930;
const HIPC_HEADER_SIZE: usize = 8;
const CMIF_HEADER_SIZE: usize = 16;
const CMIF_DOMAIN_HEADER_SIZE: usize = 16;
const CMIF_PADDING_SIZE: usize = 16;
const TIPC_COMMAND_BASE: u16 = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Protocol {
    Cmif,
    Tipc,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestKind {
    Request,
    Control,
    /// Release one domain object; this frame has no SFCI or service method ID.
    DomainClose,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DomainCommand {
    SendMessage,
    Close,
}

/// Domain IDs are wire values only. The service dispatcher must look them up in
/// the particular session's object table; they are not kernel handles.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct DomainRequest {
    pub command: DomainCommand,
    pub object_id: u32,
    pub input_objects: Vec<u32>,
    pub header_offset: usize,
    /// Exact SFCI-plus-argument size; excludes the following object-ID table.
    pub data_size: u16,
    /// Preserved for diagnostics, not interpreted as an object or pointer.
    pub reserved: u32,
    /// Domain context token. The inner SFCI token is recorded separately.
    pub token: u32,
    pub object_table_offset: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InputBufferKind {
    Pointer,
    MapAlias,
    /// Two separate inputs, an X (pointer) and an A (mapped) buffer, both
    /// active: not an AutoSelect pair (which has exactly one active), as
    /// prepo's SaveReport sends its event name and report. `address` and
    /// `size` are the A buffer's; `pointer` holds the X buffer.
    PointerAndMapAlias,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PointerDescriptor {
    pub offset: usize,
    pub raw_words: [u32; 2],
    pub index: u8,
    pub address: u64,
    pub size: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct MapAliasDescriptor {
    pub offset: usize,
    pub raw_words: [u32; 3],
    pub mode: u8,
    pub address: u64,
    pub size: u64,
}

/// One nonempty input buffer selected from exactly one X and one A descriptor.
/// The other descriptor must be inactive (address and size both zero).
/// Parsing a Pointer selection does not advertise or implement a server pointer
/// buffer: a runtime with zero pointer capacity must reject that selection.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct InputBuffer {
    pub kind: InputBufferKind,
    pub address: u64,
    pub size: u64,
    /// The X and A descriptors of an AutoSelect pair; a lone X (pointer) or
    /// lone A (mapped) input has only one.
    pub pointer: Option<PointerDescriptor>,
    pub map_alias: Option<MapAliasDescriptor>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct HipcMetadata {
    pub header_words: [u32; 2],
    pub tag: u16,
    pub special_header: Option<u32>,
    pub raw_data_words: u16,
    pub raw_data_offset: usize,
    pub raw_data_end: usize,
    pub arguments_offset: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PidDescriptor {
    /// Offset of the eight-byte PID slot relative to the message-buffer base.
    pub offset: usize,
    /// Untrusted bytes from the request, not an authenticated process identity.
    /// The transport must supply its own sender identity to a service handler.
    pub raw_value: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CmifMetadata {
    pub header_offset: usize,
    /// The combined version/reserved word must be zero or one in this subset.
    pub version: u32,
    pub token: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Request {
    pub protocol: Protocol,
    pub kind: RequestKind,
    /// Service/control method ID. Zero is a placeholder for DomainClose, whose
    /// wire frame has no method ID; dispatch must check `kind` first.
    pub command_id: u32,
    pub pid: Option<PidDescriptor>,
    /// Raw copied-handle slots in wire order, after the optional PID slot.
    /// Parsing neither validates their object types nor grants access to them.
    pub copy_handles: Vec<u32>,
    /// Raw arguments after framing removal. A domain header supplies their
    /// exact byte length; other frames retain any final word padding. Command
    /// validation still needs to check their length and interpretation. Padding
    /// inside a command argument structure is not inferred or removed here.
    pub arguments: Vec<u8>,
    pub input_buffer: Option<InputBuffer>,
    /// X descriptors after the first when a request sends several pointer
    /// inputs (index 1, 2, ...), as hid SendVibrationValues does.
    pub extra_pointers: Vec<InputBuffer>,
    /// B (receive) descriptors in wire order: (address, size, mode).
    pub receive_buffers: Vec<(u64, u64, u8)>,
    /// C receive-list entries after the raw data: (address, size).
    pub receive_list: Vec<(u64, u64)>,
    /// CMIF output-pointer size table (u16 per C buffer the client sized),
    /// found after a domain message's payload and object table.
    pub out_pointer_sizes: Vec<u16>,
    pub hipc: HipcMetadata,
    pub cmif: Option<CmifMetadata>,
    pub domain: Option<DomainRequest>,
}

impl Request {
    /// One-line shape of the request for reports on unmodeled commands: raw
    /// argument bytes, copied handles and buffer sizes.
    pub fn describe(&self) -> String {
        let arguments: String = self
            .arguments
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        let mut text = format!("arguments {} bytes [{arguments}]", self.arguments.len());
        if !self.copy_handles.is_empty() {
            text += &format!(", copied handles {:#x?}", self.copy_handles);
        }
        if let Some(buffer) = &self.input_buffer {
            text += &format!(", input buffer {:#x} bytes", buffer.size);
        }
        for buffer in &self.extra_pointers {
            text += &format!(", input buffer {:#x} bytes", buffer.size);
        }
        for (_, size, _) in &self.receive_buffers {
            text += &format!(", output buffer {size:#x} bytes");
        }
        for (_, size) in &self.receive_list {
            text += &format!(", receive list {size:#x} bytes");
        }
        text
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reply {
    /// A complete TLS message buffer. Constructed replies are zero-initialized;
    /// domain Close preserves the supplied request snapshot instead.
    pub bytes: Vec<u8>,
    /// End of the encoded HIPC message, including its raw-section padding.
    pub used_bytes: usize,
    /// Domain Close has no Result field; ordinary replies always have one.
    pub result_offset: Option<usize>,
    pub moved_handle_offset: Option<usize>,
}

fn extent(bytes: &[u8], offset: usize, size: usize, field: &str) -> Result<std::ops::Range<usize>> {
    let end = offset.checked_add(size).context("IPC extent overflow")?;
    ensure!(
        end <= bytes.len() && end <= TLS_MESSAGE_BUFFER_SIZE,
        "truncated or oversized IPC {field}: offset {offset:#x}, size {size:#x}, buffer {:#x}",
        bytes.len()
    );
    Ok(offset..end)
}

fn read_u32(bytes: &[u8], offset: usize, field: &str) -> Result<u32> {
    let range = extent(bytes, offset, 4, field)?;
    Ok(crate::bytes::le_u32(&bytes[range]))
}

fn read_u64(bytes: &[u8], offset: usize, field: &str) -> Result<u64> {
    let range = extent(bytes, offset, 8, field)?;
    Ok(crate::bytes::le_u64(&bytes[range]))
}

fn align_16(offset: usize) -> Result<usize> {
    Ok(offset.checked_add(15).context("IPC alignment overflow")? & !15)
}

fn validate_input_range(address: u64, size: u64, kind: &str) -> Result<bool> {
    // A zero-size descriptor transfers nothing whatever its address (SDK
    // ioctl wrappers pass their buffer pointer with size 0 for output-only
    // ioctls).
    if size == 0 {
        return Ok(false);
    }
    ensure!(
        address != 0,
        "unsupported IPC {kind} input descriptor with a size but no address"
    );
    ensure!(
        size <= MAX_INPUT_BUFFER_SIZE,
        "IPC {kind} input buffer exceeds the 16 MiB framing limit"
    );
    let end = address
        .checked_add(size)
        .context("IPC input buffer address overflow")?;
    ensure!(
        address < GUEST_ADDRESS_SPACE_END && end <= GUEST_ADDRESS_SPACE_END,
        "IPC {kind} input buffer exceeds the fixed 39-bit guest address space"
    );
    Ok(true)
}

fn parse_auto_select_input(bytes: &[u8], offset: usize) -> Result<InputBuffer> {
    let pointer = parse_pointer_descriptor(bytes, offset)?;
    let map_alias = parse_map_alias_input(bytes, offset + 8)?;
    let pointer_active = validate_input_range(pointer.address, pointer.size, "X")?;
    let map_active = validate_input_range(map_alias.address, map_alias.size, "A")?;
    // Both empty is an empty input (an output-only ioctl, for example).
    let (kind, address, size) = if !pointer_active && !map_active {
        (InputBufferKind::Pointer, pointer.address, 0)
    } else if pointer_active && map_active {
        (
            InputBufferKind::PointerAndMapAlias,
            map_alias.address,
            map_alias.size,
        )
    } else if pointer_active {
        (InputBufferKind::Pointer, pointer.address, pointer.size)
    } else {
        (InputBufferKind::MapAlias, map_alias.address, map_alias.size)
    };
    Ok(InputBuffer {
        kind,
        address,
        size,
        pointer: Some(pointer),
        map_alias: Some(map_alias),
    })
}

fn parse_pointer_descriptor(bytes: &[u8], offset: usize) -> Result<PointerDescriptor> {
    parse_indexed_pointer_descriptor(bytes, offset, 0)
}

fn parse_indexed_pointer_descriptor(
    bytes: &[u8],
    offset: usize,
    expected_index: u8,
) -> Result<PointerDescriptor> {
    let pointer_words = [
        read_u32(bytes, offset, "X descriptor word 0")?,
        read_u32(bytes, offset + 4, "X descriptor word 1")?,
    ];
    let pointer = PointerDescriptor {
        offset,
        raw_words: pointer_words,
        index: (pointer_words[0] & 0x3f) as u8,
        address: u64::from(pointer_words[1])
            | (u64::from((pointer_words[0] >> 12) & 0xf) << 32)
            | (u64::from((pointer_words[0] >> 6) & 0x3f) << 36),
        size: u64::from(pointer_words[0] >> 16),
    };
    ensure!(
        pointer.index == expected_index,
        "unsupported IPC X descriptor index"
    );
    Ok(pointer)
}

/// A lone A (map-alias) input descriptor, as sent for mapped input buffers.
fn parse_mapped_input(bytes: &[u8], offset: usize) -> Result<InputBuffer> {
    let map_alias = parse_map_alias_input(bytes, offset)?;
    validate_input_range(map_alias.address, map_alias.size, "A")?;
    Ok(InputBuffer {
        kind: InputBufferKind::MapAlias,
        address: map_alias.address,
        size: map_alias.size,
        pointer: None,
        map_alias: Some(map_alias),
    })
}

/// A lone X (pointer) input descriptor; the session must report a pointer
/// buffer for the client to send one.
fn parse_pointer_input(bytes: &[u8], offset: usize) -> Result<InputBuffer> {
    parse_indexed_pointer_input(bytes, offset, 0)
}

fn parse_indexed_pointer_input(bytes: &[u8], offset: usize, index: u8) -> Result<InputBuffer> {
    let pointer = parse_indexed_pointer_descriptor(bytes, offset, index)?;
    validate_input_range(pointer.address, pointer.size, "X")?;
    Ok(InputBuffer {
        kind: InputBufferKind::Pointer,
        address: pointer.address,
        size: pointer.size,
        pointer: Some(pointer),
        map_alias: None,
    })
}

fn parse_map_alias_input(bytes: &[u8], offset: usize) -> Result<MapAliasDescriptor> {
    let map_words = [
        read_u32(bytes, offset, "A descriptor size word")?,
        read_u32(bytes, offset + 4, "A descriptor address word")?,
        read_u32(bytes, offset + 8, "A descriptor attributes word")?,
    ];
    let map_alias = MapAliasDescriptor {
        offset,
        raw_words: map_words,
        mode: (map_words[2] & 3) as u8,
        address: u64::from(map_words[1])
            | (u64::from(map_words[2] >> 28) << 32)
            | (u64::from((map_words[2] >> 2) & 0x3f_ffff) << 36),
        size: u64::from(map_words[0]) | (u64::from((map_words[2] >> 24) & 0xf) << 32),
    };
    // Modes 0 (normal), 1 (non-secure) and 3 (non-device) differ only in
    // how the kernel maps the pages for the server; the bytes are the same.
    // Mode 2 is not a valid Horizon mode.
    ensure!(map_alias.mode != 2, "invalid IPC A descriptor mode 2");
    Ok(map_alias)
}

/// Parse a request contained in at most one 0x100-byte TLS message buffer.
/// Bytes after the declared message are ignored; they can be stale TLS data.
/// No argument or padding bytes are interpreted as pointers or handles here.
/// This wrapper requires a non-domain session. Use [`parse_request_with_domain`]
/// only after the service transport has established that session's domain state.
pub fn parse_request(bytes: &[u8]) -> Result<Request> {
    parse_request_with_domain(bytes, false)
}

fn parse_cmif_header(
    bytes: &[u8],
    header_offset: usize,
    message_end: usize,
) -> Result<(u32, usize, CmifMetadata)> {
    let arguments_offset = header_offset
        .checked_add(CMIF_HEADER_SIZE)
        .context("CMIF header extent overflow")?;
    ensure!(
        arguments_offset <= message_end,
        "truncated CMIF header or padding"
    );
    ensure!(
        &bytes[extent(bytes, header_offset, 4, "CMIF signature")?] == b"SFCI",
        "expected SFCI at {header_offset:#x}; domain framing requires the established session context"
    );
    let version = read_u32(bytes, header_offset + 4, "CMIF version")?;
    ensure!(
        version <= 1,
        "unsupported CMIF version/reserved word {version:#x}"
    );
    let command_id = read_u32(bytes, header_offset + 8, "CMIF command ID")?;
    let token = read_u32(bytes, header_offset + 12, "CMIF token")?;
    Ok((
        command_id,
        arguments_offset,
        CmifMetadata {
            header_offset,
            version,
            token,
        },
    ))
}

/// Parse using the transport's established per-session domain state. Request
/// tags 4/6 carry a domain header only when `is_domain` is true. Control tags
/// 5/7 always use ordinary CMIF framing, including on a domain session. TIPC
/// cannot be used as a domain protocol. Copy handles are accepted only on CMIF
/// requests, excluding domain Close. Parsing allocates no service objects.
/// Local bound on B descriptors and C receive-list entries per request.
const MAX_RECEIVE_BUFFERS: usize = 4;

pub fn parse_request_with_domain(bytes: &[u8], is_domain: bool) -> Result<Request> {
    ensure!(
        bytes.len() <= TLS_MESSAGE_BUFFER_SIZE,
        "IPC input exceeds the 0x100-byte TLS message buffer"
    );
    let word0 = read_u32(bytes, 0, "header word 0")?;
    let word1 = read_u32(bytes, 4, "header word 1")?;
    let tag = word0 as u16;
    let (protocol, mut kind) = match tag {
        4 | 6 => (Protocol::Cmif, RequestKind::Request),
        5 | 7 => (Protocol::Cmif, RequestKind::Control),
        TIPC_COMMAND_BASE..=u16::MAX => (Protocol::Tipc, RequestKind::Request),
        _ => bail!("unsupported IPC tag {tag:#x}: close and legacy frames are not implemented"),
    };
    ensure!(
        !is_domain || protocol == Protocol::Cmif,
        "TIPC cannot be used in a CMIF domain session"
    );
    let pointer_count = (word0 >> 16) & 0xf;
    let send_count = (word0 >> 20) & 0xf;
    let receive_count = ((word0 >> 24) & 0xf) as usize;
    ensure!(
        word0 >> 28 == 0,
        "unsupported IPC exchange (W) buffer descriptors"
    );
    ensure!(
        receive_count <= MAX_RECEIVE_BUFFERS,
        "IPC receive buffer count exceeds the local codec limit"
    );
    ensure!(
        receive_count == 0 || kind == RequestKind::Request,
        "IPC receive buffers require a service request"
    );
    // (X, A): none, one AutoSelect pair, one lone mapped (A) input, or
    // one or more lone pointer (X) inputs with consecutive indices.
    let has_input_buffer = match (pointer_count, send_count) {
        (0, 0) => false,
        (1, 1) | (0, 1) | (1..=4, 0) => true,
        _ => {
            bail!("unsupported IPC input buffer descriptors: require one X/A pair, one X or one A")
        }
    };
    // TIPC requests may send a lone mapped (A) buffer (the Switch 2
    // video decoder's Configure does); pointers stay CMIF-only here.
    ensure!(
        !has_input_buffer
            || (kind == RequestKind::Request
                && (protocol == Protocol::Cmif || (pointer_count, send_count) == (0, 1))),
        "unsupported IPC input buffer descriptors for this protocol or control frame"
    );
    // Receive-list mode: 0 none, 2 one entry, N > 2 N-2 entries (1 would
    // place received data inline and is not implemented).
    let list_mode = (word1 >> 10) & 0xf;
    ensure!(list_mode != 1, "unsupported inline IPC receive-list mode");
    let list_count = match list_mode {
        0 => 0,
        2 => 1,
        mode => (mode - 2) as usize,
    };
    ensure!(
        list_count <= MAX_RECEIVE_BUFFERS,
        "IPC receive-list length exceeds the local codec limit"
    );
    ensure!(
        word1 & 0x000f_c000 == 0,
        "nonzero reserved bits in IPC header word 1"
    );
    ensure!(
        word1 & 0x7ff0_0000 == 0,
        "unsupported explicit IPC receive-list offset"
    );

    let mut raw_start = HIPC_HEADER_SIZE;
    let mut pid = None;
    let mut copy_handles = Vec::new();
    let special_header = if word1 & 0x8000_0000 != 0 {
        let special = read_u32(bytes, raw_start, "special header")?;
        raw_start += 4;
        ensure!(
            special & !0x1ff == 0,
            "nonzero reserved bits in IPC special header"
        );
        ensure!(special & 0x1e0 == 0, "unsupported IPC request move handles");
        let copy_count = ((special >> 1) & 0xf) as usize;
        ensure!(
            copy_count <= MAX_COPY_HANDLES,
            "IPC request copy handle count exceeds the local codec limit"
        );
        ensure!(
            copy_count == 0 || kind == RequestKind::Request,
            "IPC copy handles require a CMIF or TIPC request"
        );
        if special & 1 != 0 {
            pid = Some(PidDescriptor {
                offset: raw_start,
                raw_value: read_u64(bytes, raw_start, "PID slot")?,
            });
            raw_start += 8;
        }
        for _ in 0..copy_count {
            copy_handles.push(read_u32(bytes, raw_start, "request copy handle")?);
            raw_start += 4;
        }
        Some(special)
    } else {
        None
    };

    let mut extra_pointers = Vec::new();
    let input_buffer = if has_input_buffer {
        if pointer_count == 1 && send_count == 1 {
            let input = parse_auto_select_input(bytes, raw_start)?;
            raw_start += 20;
            Some(input)
        } else if pointer_count >= 1 && send_count == 0 {
            let input = parse_pointer_input(bytes, raw_start)?;
            raw_start += 8;
            for index in 1..pointer_count {
                extra_pointers.push(parse_indexed_pointer_input(bytes, raw_start, index as u8)?);
                raw_start += 8;
            }
            Some(input)
        } else {
            let input = parse_mapped_input(bytes, raw_start)?;
            raw_start += 12;
            Some(input)
        }
    } else {
        None
    };
    let mut receive_buffers = Vec::with_capacity(receive_count);
    for _ in 0..receive_count {
        let words = [
            read_u32(bytes, raw_start, "B descriptor size word")?,
            read_u32(bytes, raw_start + 4, "B descriptor address word")?,
            read_u32(bytes, raw_start + 8, "B descriptor attributes word")?,
        ];
        let address = u64::from(words[1])
            | (u64::from(words[2] >> 28) << 32)
            | (u64::from((words[2] >> 2) & 0x3f_ffff) << 36);
        let size = u64::from(words[0]) | (u64::from((words[2] >> 24) & 0xf) << 32);
        receive_buffers.push((address, size, (words[2] & 3) as u8));
        raw_start += 12;
    }

    let raw_data_words = (word1 & 0x3ff) as u16;
    let raw_size = usize::from(raw_data_words) * 4;
    let raw_range = extent(bytes, raw_start, raw_size, "raw data")?;
    let raw_end = raw_range.end;
    let mut receive_list = Vec::with_capacity(list_count);
    for index in 0..list_count {
        let offset = raw_end + index * 8;
        let low = read_u32(bytes, offset, "C descriptor address word")?;
        let high = read_u32(bytes, offset + 4, "C descriptor size word")?;
        receive_list.push((
            u64::from(low) | (u64::from(high & 0xffff) << 32),
            u64::from(high >> 16),
        ));
    }

    let mut domain = None;
    let mut out_pointer_sizes = Vec::new();
    let (command_id, arguments_offset, arguments_end, cmif) = match protocol {
        Protocol::Tipc => (u32::from(tag - TIPC_COMMAND_BASE), raw_start, raw_end, None),
        Protocol::Cmif => {
            ensure!(
                raw_size >= CMIF_HEADER_SIZE + CMIF_PADDING_SIZE,
                "CMIF raw data is too short for its header and mandatory split padding"
            );
            let frame_offset = align_16(raw_start)?;
            let leading_padding = frame_offset - raw_start;
            let trailing_padding = CMIF_PADDING_SIZE - leading_padding;
            let frame_end = raw_end
                .checked_sub(trailing_padding)
                .context("CMIF padding exceeds raw data")?;
            if is_domain && kind == RequestKind::Request {
                let header = &bytes[extent(
                    bytes,
                    frame_offset,
                    CMIF_DOMAIN_HEADER_SIZE,
                    "CMIF domain header",
                )?];
                let command = match header[0] {
                    1 => DomainCommand::SendMessage,
                    2 => DomainCommand::Close,
                    value => bail!("unsupported CMIF domain command {value:#x}"),
                };
                let object_count = usize::from(header[1]);
                ensure!(
                    object_count <= MAX_DOMAIN_OBJECTS,
                    "CMIF domain input object count exceeds the supported capacity"
                );
                let data_size = crate::bytes::le_u16(&header[2..4]);
                let message_offset = frame_offset + CMIF_DOMAIN_HEADER_SIZE;
                let object_table_offset = message_offset
                    .checked_add(usize::from(data_size))
                    .context("CMIF domain data extent overflow")?;
                let object_table_end = object_table_offset
                    .checked_add(object_count * 4)
                    .context("CMIF domain object table extent overflow")?;
                ensure!(
                    object_table_end <= frame_end,
                    "CMIF domain data or object table exceeds the declared raw section"
                );
                // After the payload and object table, the client may place a
                // u16 output-pointer size per C receive-list buffer (Switchbrew
                // "IPC Marshalling"), then pad to a word. Anything longer than
                // that table is rejected.
                let trailing = frame_end - object_table_end;
                let table_limit = (list_count * 2).div_ceil(4) * 4;
                ensure!(
                    trailing < 4 || trailing <= table_limit,
                    "unsupported trailing CMIF domain raw data"
                );
                if trailing >= 4 {
                    for index in 0..list_count.min(trailing / 2) {
                        out_pointer_sizes.push(crate::bytes::le_u16(
                            &bytes[object_table_end + index * 2..object_table_end + index * 2 + 2],
                        ));
                    }
                }
                let mut input_objects = Vec::with_capacity(object_count);
                for index in 0..object_count {
                    input_objects.push(read_u32(
                        bytes,
                        object_table_offset + index * 4,
                        "CMIF domain input object ID",
                    )?);
                }
                domain = Some(DomainRequest {
                    command,
                    object_id: read_u32(bytes, frame_offset + 4, "CMIF domain target object")?,
                    input_objects,
                    header_offset: frame_offset,
                    data_size,
                    reserved: read_u32(bytes, frame_offset + 8, "CMIF domain reserved word")?,
                    token: read_u32(bytes, frame_offset + 12, "CMIF domain token")?,
                    object_table_offset,
                });
                match command {
                    DomainCommand::SendMessage => {
                        let (command_id, arguments_offset, cmif) =
                            parse_cmif_header(bytes, message_offset, object_table_offset)?;
                        (
                            command_id,
                            arguments_offset,
                            object_table_offset,
                            Some(cmif),
                        )
                    }
                    DomainCommand::Close => {
                        ensure!(
                            data_size == 0 && object_count == 0,
                            "CMIF domain Close must not contain a message or input objects"
                        );
                        ensure!(
                            pid.is_none() && copy_handles.is_empty() && input_buffer.is_none(),
                            "unsupported PID, copy handles, or input buffer on CMIF domain Close"
                        );
                        kind = RequestKind::DomainClose;
                        (0, message_offset, message_offset, None)
                    }
                }
            } else {
                let (command_id, arguments_offset, cmif) =
                    parse_cmif_header(bytes, frame_offset, frame_end)?;
                (command_id, arguments_offset, frame_end, Some(cmif))
            }
        }
    };

    Ok(Request {
        protocol,
        kind,
        command_id,
        pid,
        copy_handles,
        arguments: bytes[arguments_offset..arguments_end].to_vec(),
        input_buffer,
        extra_pointers,
        receive_buffers,
        receive_list,
        out_pointer_sizes,
        hipc: HipcMetadata {
            header_words: [word0, word1],
            tag,
            special_header,
            raw_data_words,
            raw_data_offset: raw_start,
            raw_data_end: raw_end,
            arguments_offset,
        },
        cmif,
        domain,
    })
}

fn write_u32(bytes: &mut [u8], offset: usize, value: u32) -> Result<()> {
    let range = extent(bytes, offset, 4, "reply field")?;
    bytes[range].copy_from_slice(&value.to_le_bytes());
    Ok(())
}

/// The bounded output subset has one moved handle or up to two copied
/// handles. Domain service-object IDs are separate from kernel handles.
#[derive(Clone, Copy)]
enum ReplyHandle {
    Move(u32),
    Copy(u32),
    CopyPair(u32, u32),
}

impl ReplyHandle {
    fn count(self) -> usize {
        match self {
            Self::Move(_) | Self::Copy(_) => 1,
            Self::CopyPair(..) => 2,
        }
    }
}

/// Build a reply containing a u32 service Result and an optional moved handle.
/// This constructs wire bytes only; handle allocation/ownership and the SVC's
/// transport result remain the caller's responsibility.
///
/// TIPC replies retain the encoded request tag, including on service errors.
/// Their output handle layout is the command's declared signature: a failed
/// TIPC GetServiceHandle still needs `Some(0)` for its invalid moved-handle slot.
/// Under this builder's generic SF CMIF policy, service errors have no output
/// handles, and `Some(_)` is rejected then. This is not a universal CMIF rule:
/// Atmosphere's legacy SM compatibility shim retains an invalid moved-handle
/// slot on GetService errors and uses a separate minimal fixed reply layout.
/// This is distinct from malformed/unsupported requests, which this module
/// rejects instead of fabricating a successful service or transport response.
///
/// `cmif_interface_id` is written to SFCO+0x0c; pass `SM_CMIF_INTERFACE_ID` for
/// the SM user interface. It is ignored for TIPC. CMIF response version is zero;
/// neither the request version nor its context token is echoed into the reply.
pub fn build_reply(
    request: &Request,
    result: u32,
    moved_handle: Option<u32>,
    cmif_interface_id: u32,
) -> Result<Reply> {
    build_reply_with_data(request, result, moved_handle, cmif_interface_id, &[])
}

/// As [`build_reply`], with explicit raw output arguments following the result
/// framing. The caller supplies the command's correctly ordered argument bytes;
/// this function adds zero padding to a word boundary and, for CMIF, retains the
/// sixteen total framing-padding bytes. Output always fits one TLS buffer.
pub fn build_reply_with_data(
    request: &Request,
    result: u32,
    moved_handle: Option<u32>,
    cmif_interface_id: u32,
    data: &[u8],
) -> Result<Reply> {
    build_reply_with_output_handle(
        request,
        result,
        moved_handle.map(ReplyHandle::Move),
        cmif_interface_id,
        data,
    )
}

/// Build a CMIF service reply with one copied kernel handle and optional raw
/// output data. This local subset excludes TIPC, controls, and domain Close.
/// The handle is only a wire value here: the caller must stage a valid alias
/// and publish its ownership only after the complete reply write succeeds.
/// Generic SF service errors cannot contain a copy slot, even a null one.
/// Copied handles are never reported through `Reply::moved_handle_offset`.
pub fn build_copy_handle_reply_with_data(
    request: &Request,
    result: u32,
    copy_handle: u32,
    cmif_interface_id: u32,
    data: &[u8],
) -> Result<Reply> {
    ensure!(
        request.protocol == Protocol::Cmif && request.kind == RequestKind::Request,
        "copied output handles require a CMIF service request"
    );
    build_reply_with_output_handle(
        request,
        result,
        Some(ReplyHandle::Copy(copy_handle)),
        cmif_interface_id,
        data,
    )
}

fn build_reply_with_output_handle(
    request: &Request,
    result: u32,
    output_handle: Option<ReplyHandle>,
    cmif_interface_id: u32,
    data: &[u8],
) -> Result<Reply> {
    ensure!(
        request.domain.is_none(),
        "CMIF domain requests require a domain reply builder"
    );
    ensure!(
        data.len() <= TLS_MESSAGE_BUFFER_SIZE,
        "IPC reply output data exceeds the TLS message buffer"
    );
    let data_size = data.len().div_ceil(4) * 4;
    let (tag, raw_size) = match request.protocol {
        Protocol::Cmif => {
            let tag_matches_kind = match request.kind {
                RequestKind::Request => matches!(request.hipc.tag, 4 | 6),
                RequestKind::Control => matches!(request.hipc.tag, 5 | 7),
                RequestKind::DomainClose => false,
            };
            ensure!(
                tag_matches_kind
                    && request
                        .cmif
                        .as_ref()
                        .is_some_and(|header| header.version <= 1),
                "inconsistent CMIF request metadata for reply"
            );
            ensure!(
                result == 0 || (output_handle.is_none() && data.is_empty()),
                "the generic SF CMIF service-error policy does not support output handles or data"
            );
            // Atmosphere's generic SF PrepareForReply/ErrorReply use type Invalid (0)
            // for CMIF responses, with sixteen bytes of total raw padding.
            (0, CMIF_HEADER_SIZE + CMIF_PADDING_SIZE + data_size)
        }
        Protocol::Tipc => {
            ensure!(
                request.kind == RequestKind::Request
                    && request.hipc.tag >= TIPC_COMMAND_BASE
                    && request.command_id == u32::from(request.hipc.tag - TIPC_COMMAND_BASE)
                    && request.cmif.is_none(),
                "inconsistent TIPC request metadata for reply"
            );
            // Atmosphere's InMessageHeader and OutMessageHeader use the same
            // CommandId. SerializeResults writes that header on success/error.
            (u32::from(request.hipc.tag), 4 + data_size)
        }
    };

    let raw_start = output_handle.map_or(8usize, |handle| 12 + 4 * handle.count());
    let used_bytes = raw_start
        .checked_add(raw_size)
        .context("IPC reply extent overflow")?;
    ensure!(
        used_bytes <= TLS_MESSAGE_BUFFER_SIZE,
        "IPC reply exceeds the TLS message buffer"
    );
    let mut bytes = vec![0; TLS_MESSAGE_BUFFER_SIZE];
    write_u32(&mut bytes, 0, tag)?;
    write_u32(
        &mut bytes,
        4,
        (raw_size / 4) as u32 | if output_handle.is_some() { 1 << 31 } else { 0 },
    )?;
    let moved_handle_offset = match output_handle {
        Some(ReplyHandle::Move(handle)) => {
            write_u32(&mut bytes, 8, 1 << 5)?; // one move, no PID/copies
            write_u32(&mut bytes, 12, handle)?;
            Some(12)
        }
        Some(ReplyHandle::Copy(handle)) => {
            write_u32(&mut bytes, 8, 1 << 1)?; // one copy, no PID/moves
            write_u32(&mut bytes, 12, handle)?;
            None
        }
        Some(ReplyHandle::CopyPair(first, second)) => {
            write_u32(&mut bytes, 8, 2 << 1)?; // two copies, no PID/moves
            write_u32(&mut bytes, 12, first)?;
            write_u32(&mut bytes, 16, second)?;
            None
        }
        None => None,
    };

    let (result_offset, data_offset) = match request.protocol {
        Protocol::Tipc => {
            write_u32(&mut bytes, raw_start, result)?;
            (raw_start, raw_start + 4)
        }
        Protocol::Cmif => {
            let header_offset = align_16(raw_start)?;
            let signature_range = extent(&bytes, header_offset, 4, "SFCO signature")?;
            bytes[signature_range].copy_from_slice(b"SFCO");
            write_u32(&mut bytes, header_offset + 4, 0)?;
            write_u32(&mut bytes, header_offset + 8, result)?;
            write_u32(&mut bytes, header_offset + 12, cmif_interface_id)?;
            (header_offset + 8, header_offset + CMIF_HEADER_SIZE)
        }
    };
    let data_range = extent(&bytes, data_offset, data.len(), "reply output data")?;
    bytes[data_range].copy_from_slice(data);
    Ok(Reply {
        bytes,
        used_bytes,
        result_offset: Some(result_offset),
        moved_handle_offset,
    })
}

/// Build a successful TIPC reply that moves several kernel handles, followed
/// by raw output data after the result word. The caller publishes the moved
/// objects only after the complete reply has been written.
pub fn build_tipc_reply_with_moves(request: &Request, moves: &[u32], data: &[u8]) -> Result<Reply> {
    ensure!(
        request.protocol == Protocol::Tipc
            && request.kind == RequestKind::Request
            && request.hipc.tag >= TIPC_COMMAND_BASE
            && request.command_id == u32::from(request.hipc.tag - TIPC_COMMAND_BASE),
        "moved-handle TIPC replies require a TIPC request"
    );
    ensure!(
        !moves.is_empty() && moves.len() <= 15,
        "TIPC replies move between one and fifteen handles"
    );
    let raw_start = 12 + moves.len() * 4;
    let raw_size = 4 + data.len().div_ceil(4) * 4;
    let used_bytes = raw_start + raw_size;
    ensure!(
        used_bytes <= TLS_MESSAGE_BUFFER_SIZE,
        "IPC reply exceeds the TLS message buffer"
    );
    let mut bytes = vec![0; TLS_MESSAGE_BUFFER_SIZE];
    write_u32(&mut bytes, 0, u32::from(request.hipc.tag))?;
    write_u32(&mut bytes, 4, (raw_size / 4) as u32 | (1 << 31))?;
    write_u32(&mut bytes, 8, (moves.len() as u32) << 5)?;
    for (index, handle) in moves.iter().enumerate() {
        write_u32(&mut bytes, 12 + index * 4, *handle)?;
    }
    write_u32(&mut bytes, raw_start, 0)?;
    let data_range = extent(&bytes, raw_start + 4, data.len(), "reply output data")?;
    bytes[data_range].copy_from_slice(data);
    Ok(Reply {
        bytes,
        used_bytes,
        result_offset: Some(raw_start),
        moved_handle_offset: None,
    })
}

/// Build a domain SendMessage response using the generic SF CMIF policy.
/// Successful returned service objects are IDs in this session's domain table,
/// never moved kernel handles. The caller owns object allocation and must not
/// commit those objects until it can publish the complete reply.
///
/// `data` is the command's exact marshalled output structure. Object IDs follow
/// it immediately, even when its length is not a multiple of four; only the
/// final raw section is word-rounded. Zero object IDs can represent null output
/// objects. On a service error the domain header remains present, but no output
/// data or objects are allowed. Dispatch/transport failures are not fabricated
/// into service replies by this function.
pub fn build_domain_reply_with_data(
    request: &Request,
    result: u32,
    cmif_interface_id: u32,
    data: &[u8],
    output_object_ids: &[u32],
) -> Result<Reply> {
    build_domain_reply_with_output_copy(
        request,
        result,
        &[],
        cmif_interface_id,
        data,
        output_object_ids,
    )
}

/// Build a domain SendMessage reply with one copied kernel handle. A copy
/// remains a HIPC handle even in a domain and does not add a domain object ID.
/// Scalar outputs and domain objects may coexist; their layout follows
/// [`build_domain_reply_with_data`]. The caller owns all alias/object lifetimes.
/// The generic SF error policy rejects any copied slot, including handle zero.
/// Build a CMIF reply that copies two kernel handles (for example a pair of
/// readable events), in a domain or on a plain session.
pub fn build_copy_handle_pair_reply(
    request: &Request,
    handles: [u32; 2],
    cmif_interface_id: u32,
) -> Result<Reply> {
    if request.domain.is_some() {
        return build_domain_reply_with_output_copy(
            request,
            0,
            &handles,
            cmif_interface_id,
            &[],
            &[],
        );
    }
    ensure!(
        request.protocol == Protocol::Cmif && request.kind == RequestKind::Request,
        "copied output handles require a CMIF service request"
    );
    build_reply_with_output_handle(
        request,
        0,
        Some(ReplyHandle::CopyPair(handles[0], handles[1])),
        cmif_interface_id,
        &[],
    )
}

pub fn build_domain_copy_handle_reply_with_data(
    request: &Request,
    result: u32,
    copy_handle: u32,
    cmif_interface_id: u32,
    data: &[u8],
    output_object_ids: &[u32],
) -> Result<Reply> {
    build_domain_reply_with_output_copy(
        request,
        result,
        &[copy_handle],
        cmif_interface_id,
        data,
        output_object_ids,
    )
}

fn build_domain_reply_with_output_copy(
    request: &Request,
    result: u32,
    copy_handles: &[u32],
    cmif_interface_id: u32,
    data: &[u8],
    output_object_ids: &[u32],
) -> Result<Reply> {
    ensure!(
        request.protocol == Protocol::Cmif
            && request.kind == RequestKind::Request
            && matches!(request.hipc.tag, 4 | 6)
            && request
                .cmif
                .as_ref()
                .is_some_and(|header| header.version <= 1)
            && request
                .domain
                .as_ref()
                .is_some_and(|domain| domain.command == DomainCommand::SendMessage),
        "inconsistent CMIF domain SendMessage metadata for reply"
    );
    ensure!(
        output_object_ids.len() <= MAX_DOMAIN_OBJECTS,
        "CMIF domain output object count exceeds the supported capacity"
    );
    ensure!(
        data.len() <= TLS_MESSAGE_BUFFER_SIZE,
        "IPC reply output data exceeds the TLS message buffer"
    );
    ensure!(
        result == 0 || (copy_handles.is_empty() && data.is_empty() && output_object_ids.is_empty()),
        "the generic SF CMIF domain service-error policy does not support output handles, data or objects"
    );
    let output_size = data
        .len()
        .checked_add(output_object_ids.len() * 4)
        .context("CMIF domain reply output extent overflow")?;
    let raw_size = CMIF_PADDING_SIZE
        + CMIF_DOMAIN_HEADER_SIZE
        + CMIF_HEADER_SIZE
        + output_size.div_ceil(4) * 4;
    ensure!(copy_handles.len() <= 2, "at most two copied reply handles");
    let raw_start = if copy_handles.is_empty() {
        HIPC_HEADER_SIZE
    } else {
        HIPC_HEADER_SIZE + 4 + 4 * copy_handles.len()
    };
    let used_bytes = raw_start
        .checked_add(raw_size)
        .context("CMIF domain reply extent overflow")?;
    ensure!(
        used_bytes <= TLS_MESSAGE_BUFFER_SIZE,
        "CMIF domain reply exceeds the TLS message buffer"
    );

    let mut bytes = vec![0; TLS_MESSAGE_BUFFER_SIZE];
    write_u32(&mut bytes, 0, 0)?;
    write_u32(
        &mut bytes,
        4,
        (raw_size / 4) as u32 | if copy_handles.is_empty() { 0 } else { 1 << 31 },
    )?;
    if !copy_handles.is_empty() {
        write_u32(&mut bytes, 8, (copy_handles.len() as u32) << 1)?; // copies only
        for (index, handle) in copy_handles.iter().enumerate() {
            write_u32(&mut bytes, 12 + 4 * index, *handle)?;
        }
    }
    let domain_offset = align_16(raw_start)?;
    write_u32(&mut bytes, domain_offset, output_object_ids.len() as u32)?;
    let header_offset = domain_offset + CMIF_DOMAIN_HEADER_SIZE;
    let signature_range = extent(&bytes, header_offset, 4, "domain SFCO signature")?;
    bytes[signature_range].copy_from_slice(b"SFCO");
    write_u32(&mut bytes, header_offset + 4, 0)?;
    write_u32(&mut bytes, header_offset + 8, result)?;
    write_u32(&mut bytes, header_offset + 12, cmif_interface_id)?;
    let data_offset = header_offset + CMIF_HEADER_SIZE;
    let data_range = extent(&bytes, data_offset, data.len(), "domain reply output data")?;
    bytes[data_range].copy_from_slice(data);
    let object_table_offset = data_offset + data.len();
    for (index, object_id) in output_object_ids.iter().enumerate() {
        write_u32(&mut bytes, object_table_offset + index * 4, *object_id)?;
    }
    Ok(Reply {
        bytes,
        used_bytes,
        result_offset: Some(header_offset + 8),
        moved_handle_offset: None,
    })
}

/// Preserve the request-shaped buffer for a descriptor-free domain Close.
/// The inspected SF domain dispatcher unregisters the object without writing a
/// reply frame; its outer session manager sends the same buffer. There is no
/// SFCO header, service Result, or output object table to invent here.
///
/// The original snapshot is required to preserve nonsemantic padding and stale
/// TLS bytes exactly. It is reparsed in domain context and must match `request`.
/// The dispatcher must publish this reply before committing the object release.
/// Object lookup/release semantics remain the dispatcher's responsibility.
pub fn build_domain_close_reply(request: &Request, original_bytes: &[u8]) -> Result<Reply> {
    ensure!(
        request.protocol == Protocol::Cmif
            && request.kind == RequestKind::DomainClose
            && request
                .domain
                .as_ref()
                .is_some_and(|domain| domain.command == DomainCommand::Close),
        "domain Close reply requires a parsed domain Close request"
    );
    let original_request = parse_request_with_domain(original_bytes, true)?;
    ensure!(
        original_request == *request,
        "domain Close reply snapshot does not match the parsed request"
    );
    let mut bytes = vec![0; TLS_MESSAGE_BUFFER_SIZE];
    bytes[..original_bytes.len()].copy_from_slice(original_bytes);
    Ok(Reply {
        bytes,
        used_bytes: request.hipc.raw_data_end,
        result_offset: None,
        moved_handle_offset: None,
    })
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn get_u32(bytes: &[u8], offset: usize) -> u32 {
        crate::bytes::le_u32(&bytes[offset..offset + 4])
    }

    fn tipc(command: u16, arguments: &[u8], pid: Option<u64>) -> Vec<u8> {
        assert!(arguments.len().is_multiple_of(4));
        let raw_start = if pid.is_some() { 20 } else { 8 };
        let mut bytes = vec![0; raw_start + arguments.len()];
        put_u32(&mut bytes, 0, u32::from(command) + 16);
        put_u32(
            &mut bytes,
            4,
            (arguments.len() / 4) as u32 | if pid.is_some() { 1 << 31 } else { 0 },
        );
        if let Some(pid) = pid {
            put_u32(&mut bytes, 8, 1);
            bytes[12..20].copy_from_slice(&pid.to_le_bytes());
        }
        bytes[raw_start..].copy_from_slice(arguments);
        bytes
    }

    fn cmif(tag: u16, command: u32, arguments: &[u8], pid: Option<u64>) -> Vec<u8> {
        assert!(arguments.len().is_multiple_of(4));
        let raw_start = if pid.is_some() { 20 } else { 8 };
        let raw_size = 32 + arguments.len();
        // Nonzero padding is intentional: unused SDK padding is not semantic.
        let mut bytes = vec![0xa5; raw_start + raw_size];
        put_u32(&mut bytes, 0, u32::from(tag));
        put_u32(
            &mut bytes,
            4,
            (raw_size / 4) as u32 | if pid.is_some() { 1 << 31 } else { 0 },
        );
        if let Some(pid) = pid {
            put_u32(&mut bytes, 8, 1);
            bytes[12..20].copy_from_slice(&pid.to_le_bytes());
        }
        let header = (raw_start + 15) & !15;
        bytes[header..header + 4].copy_from_slice(b"SFCI");
        put_u32(&mut bytes, header + 4, 1);
        put_u32(&mut bytes, header + 8, command);
        put_u32(&mut bytes, header + 12, 0x1234_5678);
        bytes[header + 16..header + 16 + arguments.len()].copy_from_slice(arguments);
        bytes
    }

    fn auto_select(pointer: (u64, u64), map_alias: (u64, u64), pid: Option<u64>) -> Vec<u8> {
        assert!(pointer.1 <= u64::from(u16::MAX));
        let descriptor_start = if pid.is_some() { 20 } else { 8 };
        let raw_start = descriptor_start + 20;
        let mut bytes = vec![0xa5; raw_start + 32];
        put_u32(&mut bytes, 0, 0x0011_0004);
        put_u32(&mut bytes, 4, 8 | if pid.is_some() { 1 << 31 } else { 0 });
        if let Some(pid) = pid {
            put_u32(&mut bytes, 8, 1);
            bytes[12..20].copy_from_slice(&pid.to_le_bytes());
        }
        put_u32(
            &mut bytes,
            descriptor_start,
            ((pointer.1 as u32) << 16)
                | (((pointer.0 >> 32) as u32 & 0xf) << 12)
                | (((pointer.0 >> 36) as u32 & 0x3f) << 6),
        );
        put_u32(&mut bytes, descriptor_start + 4, pointer.0 as u32);
        put_u32(&mut bytes, descriptor_start + 8, map_alias.1 as u32);
        put_u32(&mut bytes, descriptor_start + 12, map_alias.0 as u32);
        put_u32(
            &mut bytes,
            descriptor_start + 16,
            (((map_alias.0 >> 36) as u32 & 0x3f_ffff) << 2)
                | (((map_alias.1 >> 32) as u32 & 0xf) << 24)
                | (((map_alias.0 >> 32) as u32 & 0xf) << 28),
        );
        let header = (raw_start + 15) & !15;
        bytes[header..header + 4].copy_from_slice(b"SFCI");
        put_u32(&mut bytes, header + 4, 1);
        put_u32(&mut bytes, header + 8, 0);
        put_u32(&mut bytes, header + 12, 0);
        bytes
    }

    /// Add copied slots to a synthetic CMIF fixture, preserving its framed
    /// payload while independently realigning the sixteen split-padding bytes.
    fn with_copy_handles(original: &[u8], handles: &[u32]) -> Vec<u8> {
        assert!(handles.len() <= 15);
        let word0 = get_u32(original, 0);
        let word1 = get_u32(original, 4);
        assert!(matches!(word0 as u16, 4..=7));
        assert_eq!(word0 >> 24, 0);
        let old_special = if word1 >> 31 != 0 {
            get_u32(original, 8)
        } else {
            0
        };
        assert_eq!(old_special & !1, 0);
        let pid_size = if old_special & 1 != 0 { 8 } else { 0 };
        let old_descriptor_start = 8 + if word1 >> 31 != 0 { 4 + pid_size } else { 0 };
        let descriptor_size =
            ((word0 >> 16) & 0xf) as usize * 8 + ((word0 >> 20) & 0xf) as usize * 12;
        let old_raw_start = old_descriptor_start + descriptor_size;
        let old_frame = (old_raw_start + 15) & !15;
        let raw_size = (word1 & 0x3ff) as usize * 4;
        let frame_size = raw_size - 16;

        let descriptor_start = 12 + pid_size + handles.len() * 4;
        let raw_start = descriptor_start + descriptor_size;
        let frame = (raw_start + 15) & !15;
        let mut bytes = vec![0xa5; raw_start + raw_size];
        put_u32(&mut bytes, 0, word0);
        put_u32(&mut bytes, 4, word1 | 0x8000_0000);
        put_u32(&mut bytes, 8, old_special | ((handles.len() as u32) << 1));
        if pid_size != 0 {
            bytes[12..20].copy_from_slice(&original[12..20]);
        }
        for (index, handle) in handles.iter().enumerate() {
            put_u32(&mut bytes, 12 + pid_size + index * 4, *handle);
        }
        bytes[descriptor_start..raw_start]
            .copy_from_slice(&original[old_descriptor_start..old_raw_start]);
        bytes[frame..frame + frame_size]
            .copy_from_slice(&original[old_frame..old_frame + frame_size]);
        bytes
    }

    fn domain_send(
        tag: u16,
        command: u32,
        arguments: &[u8],
        pid: Option<u64>,
        object_id: u32,
        input_objects: &[u32],
    ) -> Vec<u8> {
        let raw_start = if pid.is_some() { 20 } else { 8 };
        let raw_size = (48 + arguments.len() + input_objects.len() * 4).div_ceil(4) * 4;
        let mut bytes = vec![0xa5; raw_start + raw_size];
        put_u32(&mut bytes, 0, u32::from(tag));
        put_u32(
            &mut bytes,
            4,
            (raw_size / 4) as u32 | if pid.is_some() { 1 << 31 } else { 0 },
        );
        if let Some(pid) = pid {
            put_u32(&mut bytes, 8, 1);
            bytes[12..20].copy_from_slice(&pid.to_le_bytes());
        }
        let domain = (raw_start + 15) & !15;
        bytes[domain] = 1;
        bytes[domain + 1] = input_objects.len() as u8;
        bytes[domain + 2..domain + 4]
            .copy_from_slice(&((16 + arguments.len()) as u16).to_le_bytes());
        put_u32(&mut bytes, domain + 4, object_id);
        put_u32(&mut bytes, domain + 8, 0x2468_1357);
        put_u32(&mut bytes, domain + 12, 0x1234_5678);
        let cmif = domain + 16;
        bytes[cmif..cmif + 4].copy_from_slice(b"SFCI");
        put_u32(&mut bytes, cmif + 4, 1);
        put_u32(&mut bytes, cmif + 8, command);
        put_u32(&mut bytes, cmif + 12, 0);
        let arguments_offset = cmif + 16;
        bytes[arguments_offset..arguments_offset + arguments.len()].copy_from_slice(arguments);
        for (index, object_id) in input_objects.iter().enumerate() {
            put_u32(
                &mut bytes,
                arguments_offset + arguments.len() + index * 4,
                *object_id,
            );
        }
        bytes
    }

    fn domain_close(object_id: u32) -> Vec<u8> {
        let mut bytes = vec![0xa5; 40];
        put_u32(&mut bytes, 0, 4);
        put_u32(&mut bytes, 4, 8);
        put_u32(&mut bytes, 16, 2);
        put_u32(&mut bytes, 20, object_id);
        put_u32(&mut bytes, 24, 0);
        put_u32(&mut bytes, 28, 0);
        bytes
    }

    #[test]
    fn tipc_initialize_has_a_pid_slot_and_no_raw_argument() {
        let bytes = tipc(0, &[], Some(0xfeed_cafe_1234_5678));
        let request = parse_request(&bytes).unwrap();
        assert_eq!(request.protocol, Protocol::Tipc);
        assert_eq!(request.command_id, 0);
        assert_eq!(request.hipc.raw_data_offset, 20);
        assert_eq!(request.hipc.raw_data_words, 0);
        assert_eq!(request.pid.as_ref().unwrap().offset, 12);
        assert_eq!(
            request.pid.as_ref().unwrap().raw_value,
            0xfeed_cafe_1234_5678
        );
        assert!(request.arguments.is_empty());
        assert!(request.cmif.is_none());
    }

    #[test]
    fn get_service_name_is_eight_direct_tipc_argument_bytes() {
        let name = *b"lm:\0\0\0\0\0";
        let request = parse_request(&tipc(1, &name, None)).unwrap();
        assert_eq!(request.command_id, 1);
        assert_eq!(request.hipc.arguments_offset, 8);
        assert_eq!(request.arguments, name);
        assert!(request.pid.is_none());
    }

    #[test]
    fn cmif_pid_and_context_do_not_move_padding_into_arguments() {
        let name = *b"lm:\0\0\0\0\0";
        let without_pid = parse_request(&cmif(4, 1, &name, None)).unwrap();
        assert_eq!(without_pid.arguments, name);
        assert_eq!(without_pid.hipc.raw_data_offset, 8);
        assert_eq!(without_pid.hipc.arguments_offset, 32);
        assert_eq!(without_pid.hipc.raw_data_end, 48);
        assert_eq!(without_pid.cmif.as_ref().unwrap().header_offset, 16);

        let with_pid = parse_request(&cmif(6, 0, &[0; 8], Some(99))).unwrap();
        assert_eq!(with_pid.arguments, [0; 8]);
        assert_eq!(with_pid.hipc.raw_data_offset, 20);
        assert_eq!(with_pid.hipc.arguments_offset, 48);
        assert_eq!(with_pid.hipc.raw_data_end, 60);
        let metadata = with_pid.cmif.as_ref().unwrap();
        assert_eq!(metadata.header_offset, 32);
        assert_eq!(metadata.version, 1);
        assert_eq!(metadata.token, 0x1234_5678);
        assert_eq!(with_pid.pid.as_ref().unwrap().offset, 12);
    }

    #[test]
    fn an_empty_special_header_still_changes_the_raw_offset() {
        let name = *b"lm:\0\0\0\0\0";
        let mut tiny = vec![0; 20];
        put_u32(&mut tiny, 0, 17);
        put_u32(&mut tiny, 4, 0x8000_0002);
        tiny[12..20].copy_from_slice(&name);
        let request = parse_request(&tiny).unwrap();
        assert_eq!(request.arguments, name);
        assert_eq!(request.hipc.raw_data_offset, 12);
        assert_eq!(request.hipc.special_header, Some(0));
        assert!(request.pid.is_none());

        let mut normal = cmif(4, 1, &name, None);
        normal.resize(52, 0xa5);
        put_u32(&mut normal, 4, 0x8000_000a);
        put_u32(&mut normal, 8, 0);
        let request = parse_request(&normal).unwrap();
        assert_eq!(request.arguments, name);
        assert_eq!(request.hipc.raw_data_offset, 12);
        assert_eq!(request.hipc.arguments_offset, 32);
        assert_eq!(request.hipc.raw_data_end, 52);
    }

    #[test]
    fn copy_handles_preserve_untrusted_slots_after_pid_and_before_cmif() {
        let handles = [0, 0xffff_8000, 0xffff_8001, 0x200, 5, 6, 7, u32::MAX];
        for pid in [None, Some(0x1122_3344_5566_7788)] {
            for count in 0..=MAX_COPY_HANDLES {
                let original = cmif(6, 10, &[0x5a; 16], pid);
                let bytes = with_copy_handles(&original, &handles[..count]);
                let request = parse_request(&bytes).unwrap();
                let raw_start = 12 + if pid.is_some() { 8 } else { 0 } + count * 4;
                assert_eq!(request.copy_handles, handles[..count]);
                assert_eq!(request.pid.as_ref().map(|slot| slot.raw_value), pid);
                if let Some(slot) = request.pid.as_ref() {
                    assert_eq!(slot.offset, 12);
                }
                assert_eq!(request.hipc.raw_data_offset, raw_start);
                assert_eq!(request.hipc.raw_data_end, raw_start + 48);
                assert_eq!(
                    request.cmif.as_ref().unwrap().header_offset,
                    (raw_start + 15) & !15
                );
                assert_eq!(request.arguments, [0x5a; 16]);
                assert_eq!(request.command_id, 10);
            }
        }
    }

    #[test]
    fn domain_pid_and_copy_handle_frame_has_the_observed_prefix_shape() {
        let arguments = [0x35; 16];
        let original = domain_send(6, 10, &arguments, Some(77), 1, &[]);
        let bytes = with_copy_handles(&original, &[0xffff_8001]);
        let request = parse_request_with_domain(&bytes, true).unwrap();
        assert_eq!(request.copy_handles, [0xffff_8001]);
        assert_eq!(request.pid.as_ref().unwrap().offset, 12);
        assert_eq!(request.pid.as_ref().unwrap().raw_value, 77);
        assert_eq!(request.hipc.special_header, Some(3));
        assert_eq!(request.hipc.raw_data_offset, 0x18);
        assert_eq!(request.hipc.raw_data_words, 16);
        assert_eq!(request.hipc.raw_data_end, 0x58);
        assert_eq!(request.hipc.arguments_offset, 0x40);
        assert_eq!(request.cmif.as_ref().unwrap().header_offset, 0x30);
        assert_eq!(request.cmif.as_ref().unwrap().version, 1);
        assert_eq!(request.command_id, 10);
        assert_eq!(request.arguments, arguments);
        let domain = request.domain.as_ref().unwrap();
        assert_eq!(domain.header_offset, 0x20);
        assert_eq!(domain.object_id, 1);
        assert_eq!(domain.data_size, 32);
        assert_eq!(domain.object_table_offset, 0x50);
        assert!(domain.input_objects.is_empty());

        // Request copies do not become response handles or domain output IDs.
        let ordinary = parse_request_with_domain(&original, true).unwrap();
        assert_eq!(
            build_domain_reply_with_data(&request, 0, 123, &[], &[2]).unwrap(),
            build_domain_reply_with_data(&ordinary, 0, 123, &[], &[2]).unwrap()
        );
    }

    #[test]
    fn copy_handle_slots_and_following_raw_data_require_complete_tls_extents() {
        let original = domain_send(6, 10, &[0; 16], Some(77), 1, &[9]);
        let bytes = with_copy_handles(&original, &[0, 0xffff_8000, 0xffff_8001]);
        for length in 0..bytes.len() {
            assert!(parse_request_with_domain(&bytes[..length], true).is_err());
        }
        let request = parse_request_with_domain(&bytes, true).unwrap();
        assert_eq!(request.domain.as_ref().unwrap().input_objects, [9]);
        let mut full_tls = bytes;
        full_tls.resize(TLS_MESSAGE_BUFFER_SIZE, 0xcc);
        assert_eq!(parse_request_with_domain(&full_tls, true).unwrap(), request);

        let exact = with_copy_handles(&cmif(4, 0, &[0x5a; 180], None), &[7; MAX_COPY_HANDLES]);
        assert_eq!(exact.len(), TLS_MESSAGE_BUFFER_SIZE);
        assert!(parse_request(&exact).is_ok());
        let excessive = with_copy_handles(&cmif(4, 0, &[0x5a; 184], None), &[7; MAX_COPY_HANDLES]);
        assert!(parse_request(&excessive).is_err());
        let excessive_count = with_copy_handles(&cmif(4, 0, &[], None), &[7; MAX_COPY_HANDLES + 1]);
        assert!(
            parse_request(&excessive_count)
                .unwrap_err()
                .to_string()
                .contains("copy handle count")
        );
    }

    #[test]
    fn copy_handles_precede_auto_select_descriptors_without_echoing_in_replies() {
        for pid in [None, Some(77)] {
            let original = auto_select((0, 0), (0x41_1234_5000, 24), pid);
            let bytes = with_copy_handles(&original, &[0xffff_8001, 0]);
            let request = parse_request(&bytes).unwrap();
            let descriptor_start = if pid.is_some() { 28 } else { 20 };
            let input = request.input_buffer.as_ref().unwrap();
            assert_eq!(request.copy_handles, [0xffff_8001, 0]);
            assert_eq!(input.pointer.as_ref().unwrap().offset, descriptor_start);
            assert_eq!(
                input.map_alias.as_ref().unwrap().offset,
                descriptor_start + 8
            );
            assert_eq!(input.kind, InputBufferKind::MapAlias);
            assert_eq!(input.address, 0x41_1234_5000);
            assert_eq!(input.size, 24);
            assert_eq!(request.hipc.raw_data_offset, descriptor_start + 20);
            assert!(request.arguments.is_empty());
            let ordinary = parse_request(&original).unwrap();
            for output_handle in [None, Some(0x345)] {
                assert_eq!(
                    build_reply(&request, 0, output_handle, 123).unwrap(),
                    build_reply(&ordinary, 0, output_handle, 123).unwrap()
                );
            }
        }
    }

    #[test]
    fn copied_handles_reject_controls_tipc_and_all_request_move_counts() {
        let original = cmif(4, 0, &[], None);
        let copied = with_copy_handles(&original, &[0xffff_8001]);
        for tag in [5, 7] {
            let mut bytes = copied.clone();
            put_u32(&mut bytes, 0, tag);
            for domain in [false, true] {
                assert!(parse_request_with_domain(&bytes, domain).is_err());
            }
        }
        for handles in [&[][..], &[0xffff_8001][..]] {
            for move_count in [1, 15] {
                let mut bytes = with_copy_handles(&original, handles);
                let special = get_u32(&bytes, 8);
                put_u32(&mut bytes, 8, special | (move_count << 5));
                assert!(
                    parse_request(&bytes)
                        .unwrap_err()
                        .to_string()
                        .contains("request move handles")
                );
            }
        }
    }

    #[test]
    fn tipc_requests_may_send_one_mapped_buffer() {
        // Header, one A descriptor (size 0x40 at 0x1000), one raw word.
        let mut bytes = vec![0u8; 24];
        put_u32(&mut bytes, 0, (101 + 16) | 1 << 20);
        put_u32(&mut bytes, 4, 1);
        put_u32(&mut bytes, 8, 0x40);
        put_u32(&mut bytes, 12, 0x1000);
        put_u32(&mut bytes, 20, 7);
        let request = parse_request(&bytes).unwrap();
        assert_eq!(request.protocol, Protocol::Tipc);
        assert_eq!(request.command_id, 101);
        let input = request.input_buffer.unwrap();
        assert_eq!((input.address, input.size), (0x1000, 0x40));
        assert_eq!(request.arguments, 7u32.to_le_bytes());
    }

    #[test]
    fn unsupported_frames_and_descriptors_are_rejected_explicitly() {
        let ordinary = tipc(1, b"lm:\0\0\0\0\0", None);
        for tag in [0, 1, 2, 3, 15] {
            let mut bytes = ordinary.clone();
            put_u32(&mut bytes, 0, tag);
            assert!(
                parse_request(&bytes)
                    .unwrap_err()
                    .to_string()
                    .contains("unsupported IPC tag")
            );
        }
        // X pointers and W exchanges (a lone A buffer is valid TIPC).
        for descriptor_bit in [16, 28] {
            let mut bytes = ordinary.clone();
            put_u32(&mut bytes, 0, 17 | (1 << descriptor_bit));
            assert!(
                parse_request(&bytes)
                    .unwrap_err()
                    .to_string()
                    .contains("buffer descriptors")
            );
        }
        // One B (receive) descriptor ahead of the raw data is parsed.
        let mut with_receive = vec![0u8; 0x100];
        put_u32(&mut with_receive, 0, 17 | (1 << 24));
        put_u32(&mut with_receive, 4, 2);
        put_u32(&mut with_receive, 8, 0x40); // size
        put_u32(&mut with_receive, 12, 0x5000_1000); // address low
        put_u32(&mut with_receive, 16, 0x1000_0000); // address bits 32-35 = 1
        let request = parse_request(&with_receive).unwrap();
        assert_eq!(request.receive_buffers, vec![(0x1_5000_1000, 0x40, 0)]);
        for header_bit in [10, 14, 20] {
            let mut bytes = ordinary.clone();
            put_u32(&mut bytes, 4, 2 | (1 << header_bit));
            assert!(parse_request(&bytes).is_err());
        }
        // TIPC requests may copy handles (the graphics supervisor does);
        // request moves and reserved bits are still rejected.
        let mut copied_tipc = tipc(0, &[], Some(0));
        put_u32(&mut copied_tipc, 8, 2);
        let request = parse_request(&copied_tipc).unwrap();
        assert_eq!(request.copy_handles, vec![0]);
        for special in [1 << 5, 1 << 9] {
            let mut bytes = tipc(0, &[], Some(0));
            put_u32(&mut bytes, 8, special);
            assert!(parse_request(&bytes).is_err());
        }
    }

    #[test]
    fn non_domain_cmif_rejects_domain_headers_bad_signatures_versions_and_short_padding() {
        let ordinary = cmif(4, 1, b"lm:\0\0\0\0\0", None);
        for signature in [*b"SFCO", [1, 0, 0x10, 0]] {
            let mut bytes = ordinary.clone();
            bytes[16..20].copy_from_slice(&signature);
            assert!(
                parse_request(&bytes)
                    .unwrap_err()
                    .to_string()
                    .contains("domain framing requires the established session context")
            );
        }
        for version in [2, 0x1_0000] {
            let mut bytes = ordinary.clone();
            put_u32(&mut bytes, 20, version);
            assert!(parse_request(&bytes).is_err());
        }
        let mut short_raw = ordinary;
        put_u32(&mut short_raw, 4, 4);
        assert!(
            parse_request(&short_raw)
                .unwrap_err()
                .to_string()
                .contains("mandatory split padding")
        );
    }

    #[test]
    fn every_required_byte_is_bounded_and_trailing_tls_bytes_are_ignored() {
        for bytes in [
            tipc(1, b"lm:\0\0\0\0\0", None),
            tipc(0, &[], Some(7)),
            cmif(4, 1, b"lm:\0\0\0\0\0", None),
            cmif(6, 0, &[0; 8], Some(7)),
        ] {
            for truncated_size in 0..bytes.len() {
                assert!(
                    parse_request(&bytes[..truncated_size]).is_err(),
                    "accepted truncated packet at {truncated_size}/{}",
                    bytes.len()
                );
            }
            let expected = parse_request(&bytes).unwrap();
            let mut full_tls = bytes;
            full_tls.resize(TLS_MESSAGE_BUFFER_SIZE, 0xcc);
            assert_eq!(parse_request(&full_tls).unwrap(), expected);
            full_tls.push(0);
            assert!(parse_request(&full_tls).is_err());
        }
        let mut oversized_raw = tipc(1, b"lm:\0\0\0\0\0", None);
        put_u32(&mut oversized_raw, 4, 0x3ff);
        assert!(parse_request(&oversized_raw).is_err());
    }

    #[test]
    fn tipc_replies_echo_the_tag_and_put_a_single_result_after_handles() {
        let initialize = parse_request(&tipc(0, &[], Some(u64::MAX))).unwrap();
        let reply = build_reply(&initialize, 0, None, SM_CMIF_INTERFACE_ID).unwrap();
        assert_eq!(reply.bytes.len(), TLS_MESSAGE_BUFFER_SIZE);
        assert_eq!(get_u32(&reply.bytes, 0), 16);
        assert_eq!(get_u32(&reply.bytes, 4), 1);
        assert_eq!(reply.result_offset, Some(8));
        assert_eq!(reply.used_bytes, 12);
        assert!(reply.bytes[8..].iter().all(|byte| *byte == 0));

        let get_service = parse_request(&tipc(1, b"lm:\0\0\0\0\0", None)).unwrap();
        for (result, handle) in [(0, 0x345), (0x1234, 0)] {
            let reply = build_reply(&get_service, result, Some(handle), 0).unwrap();
            assert_eq!(get_u32(&reply.bytes, 0), 17);
            assert_eq!(get_u32(&reply.bytes, 4), 0x8000_0001);
            assert_eq!(get_u32(&reply.bytes, 8), 1 << 5);
            assert_eq!(get_u32(&reply.bytes, 12), handle);
            assert_eq!(get_u32(&reply.bytes, 16), result);
            assert_eq!(reply.moved_handle_offset, Some(12));
            assert_eq!(reply.result_offset, Some(16));
            assert_eq!(reply.used_bytes, 20);
            assert!(reply.bytes[20..].iter().all(|byte| *byte == 0));
        }
    }

    #[test]
    fn cmif_replies_have_sfco_version_zero_interface_id_and_split_padding() {
        let request = parse_request(&cmif(6, 1, b"lm:\0\0\0\0\0", None)).unwrap();
        for (result, handle, expected_size) in [(0, Some(0x345), 48), (0x1234, None, 40)] {
            let reply = build_reply(&request, result, handle, SM_CMIF_INTERFACE_ID).unwrap();
            assert_eq!(get_u32(&reply.bytes, 0), 0);
            assert_eq!(
                get_u32(&reply.bytes, 4),
                8 | if handle.is_some() { 1 << 31 } else { 0 }
            );
            assert_eq!(&reply.bytes[16..20], b"SFCO");
            assert_eq!(get_u32(&reply.bytes, 20), 0);
            assert_eq!(get_u32(&reply.bytes, 24), result);
            assert_eq!(get_u32(&reply.bytes, 28), SM_CMIF_INTERFACE_ID);
            assert_eq!(reply.result_offset, Some(24));
            assert_eq!(reply.used_bytes, expected_size);
            if let Some(handle) = handle {
                assert_eq!(get_u32(&reply.bytes, 8), 1 << 5);
                assert_eq!(get_u32(&reply.bytes, 12), handle);
            } else {
                assert!(reply.bytes[8..16].iter().all(|byte| *byte == 0));
            }
            assert!(reply.bytes[32..].iter().all(|byte| *byte == 0));
        }
        assert!(build_reply(&request, 0x1234, Some(0), SM_CMIF_INTERFACE_ID).is_err());
    }

    #[test]
    fn copied_handle_replies_have_exact_frames_without_echoing_request_metadata() {
        for domain in [false, true] {
            for tag in [4, 6] {
                for pid in [None, Some(0xfeed_cafe_1234_5678)] {
                    let original = if domain {
                        domain_send(tag, 91, &[], pid, 7, &[])
                    } else {
                        cmif(tag, 91, &[], pid)
                    };
                    let bytes = with_copy_handles(&original, &[0xffff_8001, 0x432]);
                    let parsed = parse_request_with_domain(&bytes, domain).unwrap();
                    for version in [0, 1] {
                        let mut request = parsed.clone();
                        request.cmif.as_mut().unwrap().version = version;
                        for handle in [0, 0xfedc_ba98] {
                            let reply = if domain {
                                build_domain_copy_handle_reply_with_data(
                                    &request,
                                    0,
                                    handle,
                                    0x56ef_eaa1,
                                    &[],
                                    &[],
                                )
                            } else {
                                build_copy_handle_reply_with_data(
                                    &request,
                                    0,
                                    handle,
                                    0x56ef_eaa1,
                                    &[],
                                )
                            }
                            .unwrap();
                            let mut expected = vec![0; 256];
                            let (word1, cmif_offset, used_bytes) = if domain {
                                (0x8000_000cu32, 0x20, 0x40)
                            } else {
                                (0x8000_0008u32, 0x10, 0x30)
                            };
                            put_u32(&mut expected, 4, word1);
                            put_u32(&mut expected, 8, 2);
                            put_u32(&mut expected, 12, handle);
                            expected[cmif_offset..cmif_offset + 4].copy_from_slice(b"SFCO");
                            put_u32(&mut expected, cmif_offset + 12, 0x56ef_eaa1);
                            // This compares all padding and unused TLS bytes,
                            // including zero domain objects and absent PID.
                            assert_eq!(reply.bytes, expected);
                            assert_eq!(reply.used_bytes, used_bytes);
                            assert_eq!(reply.result_offset, Some(cmif_offset + 8));
                            assert_eq!(reply.moved_handle_offset, None);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn copied_handles_coexist_with_data_and_unaligned_domain_object_tables() {
        let ordinary = parse_request(&cmif(4, 91, &[], None)).unwrap();
        let reply =
            build_copy_handle_reply_with_data(&ordinary, 0, 0x345, 0x1234, &[0x34, 0x12]).unwrap();
        let mut expected = vec![0; 256];
        put_u32(&mut expected, 4, 0x8000_0009);
        put_u32(&mut expected, 8, 2);
        put_u32(&mut expected, 12, 0x345);
        expected[16..20].copy_from_slice(b"SFCO");
        put_u32(&mut expected, 28, 0x1234);
        expected[32..34].copy_from_slice(&[0x34, 0x12]);
        assert_eq!(reply.bytes, expected);
        assert_eq!(reply.used_bytes, 52);
        assert_eq!(reply.result_offset, Some(24));
        assert_eq!(reply.moved_handle_offset, None);

        let domain =
            parse_request_with_domain(&domain_send(6, 91, &[], None, 7, &[]), true).unwrap();
        let reply = build_domain_copy_handle_reply_with_data(
            &domain,
            0,
            0x345,
            0x1234,
            &[0x34, 0x12],
            &[2, 0],
        )
        .unwrap();
        let mut expected = vec![0; 256];
        put_u32(&mut expected, 4, 0x8000_000f);
        put_u32(&mut expected, 8, 2);
        put_u32(&mut expected, 12, 0x345);
        put_u32(&mut expected, 16, 2); // Only the two objects, not the handle.
        expected[32..36].copy_from_slice(b"SFCO");
        put_u32(&mut expected, 44, 0x1234);
        expected[48..50].copy_from_slice(&[0x34, 0x12]);
        put_u32(&mut expected, 50, 2); // No gap after the two scalar bytes.
        put_u32(&mut expected, 54, 0);
        assert_eq!(reply.bytes, expected);
        assert_eq!(reply.used_bytes, 76);
        assert_eq!(reply.result_offset, Some(40));
        assert_eq!(reply.moved_handle_offset, None);
    }

    #[test]
    fn two_pointer_inputs_parse_in_index_order() {
        // hid SendNativeVibrationValues (214): u16 flags, aruid, then X0
        // (vibration handles) and X1 (values), from a recorded request.
        let mut bytes = vec![0u8; 0x100];
        for (index, word) in [
            0x0002_0006u32,
            0x0000_000c,
            0x0030_11c0,
            0x04d2_6fbc,
            0x00c0_11c1,
            0x04d2_6fec,
            0,
            0,
            u32::from_le_bytes(*b"SFCI"),
            1,
            214,
        ]
        .iter()
        .enumerate()
        {
            put_u32(&mut bytes, index * 4, *word);
        }
        let request = parse_request(&bytes).unwrap();
        assert_eq!(request.command_id, 214);
        let first = request.input_buffer.unwrap();
        assert_eq!((first.address, first.size), (0x71_04d2_6fbc, 0x30));
        assert_eq!(request.extra_pointers.len(), 1);
        let second = &request.extra_pointers[0];
        assert_eq!((second.address, second.size), (0x71_04d2_6fec, 0xc0));
    }

    #[test]
    fn copy_handle_pair_replies_carry_both_handles() {
        let ordinary = parse_request(&cmif(4, 91, &[], None)).unwrap();
        let reply = build_copy_handle_pair_reply(&ordinary, [0x345, 0x346], 0x1234).unwrap();
        assert_eq!(crate::bytes::le_u32(&reply.bytes[4..8]), 0x8000_0008);
        assert_eq!(crate::bytes::le_u32(&reply.bytes[8..12]), 2 << 1);
        assert_eq!(crate::bytes::le_u32(&reply.bytes[12..16]), 0x345);
        assert_eq!(crate::bytes::le_u32(&reply.bytes[16..20]), 0x346);
        assert_eq!(&reply.bytes[32..36], b"SFCO");
        assert_eq!(reply.result_offset, Some(40));

        let domain =
            parse_request_with_domain(&domain_send(6, 91, &[], None, 7, &[]), true).unwrap();
        let reply = build_copy_handle_pair_reply(&domain, [0x345, 0x346], 0x1234).unwrap();
        assert_eq!(crate::bytes::le_u32(&reply.bytes[4..8]), 0x8000_000c);
        assert_eq!(crate::bytes::le_u32(&reply.bytes[8..12]), 2 << 1);
        assert_eq!(crate::bytes::le_u32(&reply.bytes[16..20]), 0x346);
        assert_eq!(crate::bytes::le_u32(&reply.bytes[32..36]), 0);
        assert_eq!(&reply.bytes[48..52], b"SFCO");
        assert_eq!(reply.result_offset, Some(56));
    }

    #[test]
    fn copied_handle_replies_stop_at_tls_and_domain_object_limits() {
        let ordinary = parse_request(&cmif(4, 91, &[], None)).unwrap();
        let reply = build_copy_handle_reply_with_data(&ordinary, 0, 1, 0, &[0x5a; 208]).unwrap();
        assert_eq!(reply.used_bytes, 256);
        assert_eq!(reply.bytes.len(), 256);
        assert!(build_copy_handle_reply_with_data(&ordinary, 0, 1, 0, &[0; 209]).is_err());

        let domain =
            parse_request_with_domain(&domain_send(4, 91, &[], None, 7, &[]), true).unwrap();
        for (object_count, maximum_data) in [(0, 192), (1, 188), (8, 160)] {
            let objects = vec![2; object_count];
            let reply = build_domain_copy_handle_reply_with_data(
                &domain,
                0,
                1,
                0,
                &vec![0x5a; maximum_data],
                &objects,
            )
            .unwrap();
            assert_eq!(reply.used_bytes, 256);
            assert_eq!(reply.bytes.len(), 256);
            assert!(
                build_domain_copy_handle_reply_with_data(
                    &domain,
                    0,
                    1,
                    0,
                    &vec![0; maximum_data + 1],
                    &objects,
                )
                .is_err()
            );
        }
        assert!(build_domain_copy_handle_reply_with_data(&domain, 0, 1, 0, &[], &[2; 9]).is_err());
    }

    #[test]
    fn copied_handle_replies_reject_wrong_metadata_and_generic_service_errors() {
        let ordinary = parse_request(&cmif(4, 91, &[], None)).unwrap();
        let domain =
            parse_request_with_domain(&domain_send(4, 91, &[], None, 7, &[]), true).unwrap();
        for result in [1, 0xe401] {
            for handle in [0, 0x345] {
                assert!(
                    build_copy_handle_reply_with_data(&ordinary, result, handle, 0, &[],).is_err()
                );
                assert!(
                    build_domain_copy_handle_reply_with_data(&domain, result, handle, 0, &[], &[],)
                        .is_err()
                );
            }
        }
        assert!(build_copy_handle_reply_with_data(&domain, 0, 1, 0, &[]).is_err());
        assert!(build_domain_copy_handle_reply_with_data(&ordinary, 0, 1, 0, &[], &[]).is_err());

        let mut invalid = vec![
            parse_request(&tipc(1, &[0; 8], None)).unwrap(),
            parse_request(&cmif(5, 3, &[], None)).unwrap(),
            parse_request(&cmif(7, 3, &[], None)).unwrap(),
            parse_request_with_domain(&domain_close(7), true).unwrap(),
        ];
        for original in [&ordinary, &domain] {
            let mut bad = (*original).clone();
            bad.protocol = Protocol::Tipc;
            invalid.push(bad);
            let mut bad = (*original).clone();
            bad.kind = RequestKind::Control;
            invalid.push(bad);
            let mut bad = (*original).clone();
            bad.hipc.tag = 5;
            invalid.push(bad);
            let mut bad = (*original).clone();
            bad.cmif = None;
            invalid.push(bad);
            let mut bad = (*original).clone();
            bad.cmif.as_mut().unwrap().version = 2;
            invalid.push(bad);
        }
        let mut wrong_domain_command = domain.clone();
        wrong_domain_command.domain.as_mut().unwrap().command = DomainCommand::Close;
        invalid.push(wrong_domain_command);
        for request in invalid {
            assert!(build_copy_handle_reply_with_data(&request, 0, 1, 0, &[]).is_err());
            assert!(build_domain_copy_handle_reply_with_data(&request, 0, 1, 0, &[], &[]).is_err());
        }
    }

    #[test]
    fn legacy_reply_builders_keep_their_copy_free_framing() {
        let request = parse_request(&cmif(6, 1, &[], Some(0x1234))).unwrap();
        for handle in [None, Some(0x345)] {
            let reply = build_reply_with_data(&request, 0, handle, 0x1234, &[0x34, 0x12]).unwrap();
            let mut expected = vec![0; 256];
            put_u32(
                &mut expected,
                4,
                if handle.is_some() { 0x8000_0009 } else { 9 },
            );
            if let Some(handle) = handle {
                put_u32(&mut expected, 8, 32);
                put_u32(&mut expected, 12, handle);
            }
            expected[16..20].copy_from_slice(b"SFCO");
            put_u32(&mut expected, 28, 0x1234);
            expected[32..34].copy_from_slice(&[0x34, 0x12]);
            assert_eq!(reply.bytes, expected);
            assert_eq!(reply.used_bytes, if handle.is_some() { 52 } else { 44 });
            assert_eq!(reply.moved_handle_offset, handle.map(|_| 12));
        }

        let tipc_request = parse_request(&tipc(1, &[0; 8], None)).unwrap();
        let reply = build_reply(&tipc_request, 0x415, Some(0), 0).unwrap();
        let mut expected = vec![0; 256];
        put_u32(&mut expected, 0, 17);
        put_u32(&mut expected, 4, 0x8000_0001);
        put_u32(&mut expected, 8, 32);
        put_u32(&mut expected, 16, 0x415);
        assert_eq!(reply.bytes, expected);
        assert_eq!(reply.used_bytes, 20);
        assert_eq!(reply.moved_handle_offset, Some(12));

        let domain =
            parse_request_with_domain(&domain_send(4, 1, &[], None, 7, &[]), true).unwrap();
        let reply =
            build_domain_reply_with_data(&domain, 0, 0x1234, &[0x34, 0x12], &[2, 0]).unwrap();
        let mut expected = vec![0; 256];
        put_u32(&mut expected, 4, 15);
        put_u32(&mut expected, 16, 2);
        expected[32..36].copy_from_slice(b"SFCO");
        put_u32(&mut expected, 44, 0x1234);
        expected[48..50].copy_from_slice(&[0x34, 0x12]);
        put_u32(&mut expected, 50, 2);
        assert_eq!(reply.bytes, expected);
        assert_eq!(reply.used_bytes, 68);
        assert_eq!(reply.moved_handle_offset, None);
    }

    #[test]
    fn control_frames_keep_cmif_alignment_and_u16_output_word_padding() {
        for tag in [5, 7] {
            let mut bytes = cmif(tag, 3, &[], None);
            if tag == 5 {
                put_u32(&mut bytes, 20, 0);
            }
            let request = parse_request(&bytes).unwrap();
            assert_eq!(request.kind, RequestKind::Control);
            assert_eq!(request.protocol, Protocol::Cmif);
            assert_eq!(request.command_id, 3);
            assert!(request.arguments.is_empty());
            assert!(request.input_buffer.is_none());
            let reply =
                build_reply_with_data(&request, 0, None, 0xec6b_e3ff, &0x400_u16.to_le_bytes())
                    .unwrap();
            assert_eq!(get_u32(&reply.bytes, 0), 0);
            assert_eq!(get_u32(&reply.bytes, 4), 9);
            assert_eq!(&reply.bytes[16..20], b"SFCO");
            assert_eq!(get_u32(&reply.bytes, 28), 0xec6b_e3ff);
            assert_eq!(&reply.bytes[32..34], &0x400_u16.to_le_bytes());
            assert!(reply.bytes[34..].iter().all(|byte| *byte == 0));
            assert_eq!(reply.used_bytes, 44);
            assert!(build_reply_with_data(&request, 1, None, 0, &[0, 0]).is_err());
        }
        assert_eq!(
            parse_request(&tipc(0, &[], Some(1))).unwrap().kind,
            RequestKind::Request
        );
    }

    #[test]
    fn extra_reply_data_cannot_overflow_a_tls_message() {
        let cmif_request = parse_request(&cmif(4, 1, &[], None)).unwrap();
        let tipc_request = parse_request(&tipc(1, &[], None)).unwrap();
        for (request, handle, maximum) in [
            (&cmif_request, None, 216),
            (&cmif_request, Some(1), 208),
            (&tipc_request, None, 244),
            (&tipc_request, Some(1), 236),
        ] {
            let reply = build_reply_with_data(request, 0, handle, 0, &vec![0x5a; maximum]).unwrap();
            assert_eq!(reply.bytes.len(), TLS_MESSAGE_BUFFER_SIZE);
            assert_eq!(reply.used_bytes, TLS_MESSAGE_BUFFER_SIZE);
            assert!(
                build_reply_with_data(request, 0, handle, 0, &vec![0x5a; maximum + 1]).is_err()
            );
        }
    }

    #[test]
    fn auto_select_decodes_both_choices_and_pid_adjusted_descriptor_offsets() {
        let address = 0x41_2345_6780;
        for (pointer, map_alias, kind, pid) in [
            ((0, 0), (address, 48), InputBufferKind::MapAlias, None),
            ((address, 48), (0, 0), InputBufferKind::Pointer, None),
            ((0, 0), (address, 48), InputBufferKind::MapAlias, Some(7)),
        ] {
            let request = parse_request(&auto_select(pointer, map_alias, pid)).unwrap();
            let buffer = request.input_buffer.as_ref().unwrap();
            assert_eq!(buffer.kind, kind);
            assert_eq!(buffer.address, address);
            assert_eq!(buffer.size, 48);
            assert_eq!(buffer.pointer.as_ref().unwrap().address, pointer.0);
            assert_eq!(buffer.map_alias.as_ref().unwrap().address, map_alias.0);
            assert_eq!(
                buffer.pointer.as_ref().unwrap().offset,
                if pid.is_some() { 20 } else { 8 }
            );
            assert_eq!(
                buffer.map_alias.as_ref().unwrap().offset,
                buffer.pointer.as_ref().unwrap().offset + 8
            );
            assert_eq!(
                request.hipc.raw_data_offset,
                buffer.pointer.as_ref().unwrap().offset + 20
            );
            assert_eq!(
                request.hipc.arguments_offset,
                if pid.is_some() { 64 } else { 48 }
            );
            assert!(request.arguments.is_empty());
        }
        assert!(
            parse_request(&auto_select(
                (0, 0),
                (GUEST_ADDRESS_SPACE_END - 16, 16),
                None,
            ))
            .is_ok()
        );
        assert!(
            parse_request(&auto_select((0, 0), (0x1000, MAX_INPUT_BUFFER_SIZE), None,)).is_ok()
        );
    }

    #[test]
    fn auto_select_rejects_ambiguous_unsafe_or_unsupported_descriptors() {
        for (pointer, map_alias) in [
            ((0, 0), (0, 32)),
            ((0, 32), (0, 0)),
            ((0, 0), (GUEST_ADDRESS_SPACE_END, 32)),
            ((GUEST_ADDRESS_SPACE_END, 32), (0, 0)),
            ((0, 0), (GUEST_ADDRESS_SPACE_END - 15, 16)),
            ((0, 0), (0x1000, MAX_INPUT_BUFFER_SIZE + 1)),
            ((0, 0), (0x1000, 1 << 32)),
        ] {
            assert!(parse_request(&auto_select(pointer, map_alias, None)).is_err());
        }
        // Zero-size halves are empty whatever their address; both empty is
        // an empty input.
        for (pointer, map_alias) in [((0, 0), (0, 0)), ((0x1000, 0), (0x2000, 0))] {
            let empty = parse_request(&auto_select(pointer, map_alias, None)).unwrap();
            assert_eq!(empty.input_buffer.unwrap().size, 0);
        }
        let half = parse_request(&auto_select((0x1000, 0), (0x2000, 16), None)).unwrap();
        let half = half.input_buffer.unwrap();
        assert_eq!(
            (half.kind, half.address, half.size),
            (InputBufferKind::MapAlias, 0x2000, 16)
        );
        // Both active: two inputs, not an AutoSelect choice.
        let both = parse_request(&auto_select((0x1000, 32), (0x2000, 48), None)).unwrap();
        let both = both.input_buffer.unwrap();
        assert_eq!(both.kind, InputBufferKind::PointerAndMapAlias);
        assert_eq!((both.address, both.size), (0x2000, 48));
        let ordinary = auto_select((0, 0), (0x1000, 32), None);
        // Non-secure (1) and non-device (3) buffers read like normal ones;
        // mode 2 does not exist.
        for (mode, valid) in [(1, true), (2, false), (3, true)] {
            let mut bytes = ordinary.clone();
            put_u32(&mut bytes, 24, mode);
            assert_eq!(parse_request(&bytes).is_ok(), valid);
        }
        let mut bad_index = ordinary.clone();
        put_u32(&mut bad_index, 8, 1);
        assert!(parse_request(&bad_index).is_err());
        for word0 in [0x0012_0004, 0x0021_0004, 0x0011_0005, 0x0011_0010] {
            let mut bytes = ordinary.clone();
            put_u32(&mut bytes, 0, word0);
            assert!(parse_request(&bytes).is_err());
        }
    }

    #[test]
    fn descriptor_and_following_raw_extents_are_both_required() {
        let bytes = auto_select((0, 0), (0x1000, 32), None);
        for size in 0..bytes.len() {
            assert!(parse_request(&bytes[..size]).is_err());
        }
        assert!(parse_request(&bytes).is_ok());
        let mut excessive_raw = bytes;
        put_u32(&mut excessive_raw, 4, 0x3ff);
        assert!(parse_request(&excessive_raw).is_err());
    }

    #[test]
    fn domain_data_length_separates_arguments_object_ids_and_context_tokens() {
        for (tag, pid, domain_offset, arguments_offset) in
            [(4, None, 16, 48), (6, Some(77), 32, 64)]
        {
            let bytes = domain_send(tag, 19, &[0xde, 0xad, 0xbe], pid, 0x42, &[9, 0]);
            let request = parse_request_with_domain(&bytes, true).unwrap();
            assert_eq!(request.kind, RequestKind::Request);
            assert_eq!(request.command_id, 19);
            assert_eq!(request.arguments, [0xde, 0xad, 0xbe]);
            assert_eq!(request.hipc.arguments_offset, arguments_offset);
            let domain = request.domain.as_ref().unwrap();
            assert_eq!(domain.command, DomainCommand::SendMessage);
            assert_eq!(domain.object_id, 0x42);
            assert_eq!(domain.input_objects, [9, 0]);
            assert_eq!(domain.header_offset, domain_offset);
            assert_eq!(domain.data_size, 19);
            assert_eq!(domain.object_table_offset, arguments_offset + 3);
            assert_eq!(domain.reserved, 0x2468_1357);
            assert_eq!(domain.token, 0x1234_5678);
            let cmif = request.cmif.as_ref().unwrap();
            assert_eq!(cmif.header_offset, domain_offset + 16);
            assert_eq!(cmif.version, 1);
            assert_eq!(cmif.token, 0);
            assert_eq!(request.pid.as_ref().map(|value| value.raw_value), pid);
        }
    }

    #[test]
    fn domain_context_is_explicit_and_controls_remain_non_domain() {
        let domain = domain_send(4, 0, &[], None, 1, &[]);
        assert!(parse_request(&domain).is_err());
        assert!(parse_request_with_domain(&cmif(4, 0, &[], None), true).is_err());
        assert!(parse_request_with_domain(&tipc(0, &[], Some(1)), true).is_err());
        for tag in [5, 7] {
            let ordinary = cmif(tag, 3, &[], None);
            let request = parse_request_with_domain(&ordinary, true).unwrap();
            assert_eq!(request, parse_request(&ordinary).unwrap());
            assert_eq!(request.kind, RequestKind::Control);
            assert!(request.domain.is_none());
            assert!(build_reply_with_data(&request, 0, None, 0xec6b_e3ff, &[0, 0]).is_ok());
            assert!(build_domain_reply_with_data(&request, 0, 0, &[], &[]).is_err());
            assert!(
                parse_request_with_domain(&domain_send(tag, 3, &[], None, 1, &[]), true).is_err()
            );
        }
    }

    #[test]
    fn domain_header_and_object_table_cannot_escape_the_declared_raw_extent() {
        let bytes = domain_send(6, 2, &[0x11, 0x22], Some(7), 1, &[2]);
        for truncated_size in 0..bytes.len() {
            assert!(parse_request_with_domain(&bytes[..truncated_size], true).is_err());
        }
        let expected = parse_request_with_domain(&bytes, true).unwrap();
        let mut full_tls = bytes.clone();
        full_tls.resize(TLS_MESSAGE_BUFFER_SIZE, 0xcc);
        assert_eq!(
            parse_request_with_domain(&full_tls, true).unwrap(),
            expected
        );
        full_tls.push(0);
        assert!(parse_request_with_domain(&full_tls, true).is_err());

        for data_size in [0_u16, 15, 21, u16::MAX] {
            let mut bad = bytes.clone();
            bad[34..36].copy_from_slice(&data_size.to_le_bytes());
            assert!(parse_request_with_domain(&bad, true).is_err());
        }
        for command in [0, 3, 255] {
            let mut bad = bytes.clone();
            bad[32] = command;
            assert!(parse_request_with_domain(&bad, true).is_err());
        }
        let mut missing_object = bytes.clone();
        missing_object[33] = 2;
        assert!(parse_request_with_domain(&missing_object, true).is_err());
        let mut extra_raw_word = bytes;
        let word1 = get_u32(&extra_raw_word, 4);
        put_u32(&mut extra_raw_word, 4, word1 + 1);
        extra_raw_word.extend_from_slice(&[0; 4]);
        assert!(parse_request_with_domain(&extra_raw_word, true).is_err());

        let within_cap = domain_send(4, 0, &[], None, 1, &[2; MAX_DOMAIN_OBJECTS]);
        assert!(parse_request_with_domain(&within_cap, true).is_ok());
        let over_cap = domain_send(4, 0, &[], None, 1, &[2; MAX_DOMAIN_OBJECTS + 1]);
        assert!(parse_request_with_domain(&over_cap, true).is_err());
    }

    #[test]
    fn domain_output_pointer_size_table_follows_the_payload() {
        // Domain SendMessage with one B buffer, one C receive-list entry and
        // the client's u16 output-pointer size table after the arguments.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(6u32 | (1 << 24)).to_le_bytes());
        bytes.extend_from_slice(&(15u32 | (3 << 10)).to_le_bytes());
        bytes.extend_from_slice(&0x88u32.to_le_bytes());
        bytes.extend_from_slice(&0x1234_5000u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.resize(32, 0);
        bytes.extend_from_slice(&[1, 0, 0x18, 0]);
        bytes.extend_from_slice(&0xfu32.to_le_bytes());
        bytes.extend_from_slice(&[0; 8]);
        bytes.extend_from_slice(b"SFCI");
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&11u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0x10u64.to_le_bytes());
        bytes.extend_from_slice(&0x88u16.to_le_bytes());
        bytes.resize(80, 0);
        bytes.extend_from_slice(&0x1234_6000u32.to_le_bytes());
        bytes.extend_from_slice(&(0x88u32 << 16).to_le_bytes());
        let request = parse_request_with_domain(&bytes, true).unwrap();
        assert_eq!(request.command_id, 11);
        assert_eq!(request.arguments, 0x10u64.to_le_bytes());
        assert_eq!(request.receive_buffers, [(0x1234_5000, 0x88, 0)]);
        assert_eq!(request.receive_list, [(0x1234_6000, 0x88)]);
        assert_eq!(request.out_pointer_sizes, [0x88]);
        // Without a C descriptor there is no table, so the word is rejected.
        let mut no_list = bytes[..80].to_vec();
        put_u32(&mut no_list, 4, 15);
        assert!(parse_request_with_domain(&no_list, true).is_err());
    }

    #[test]
    fn domain_auto_select_offsets_follow_the_complete_descriptor_prefix() {
        for pid in [None, Some(77)] {
            let mut bytes = auto_select((0, 0), (0x41_1234_5000, 24), pid);
            let descriptor_start = if pid.is_some() { 20 } else { 8 };
            let raw_start = descriptor_start + 20;
            let domain_offset = (raw_start + 15) & !15;
            bytes.splice(domain_offset..domain_offset, [0; 16]);
            let word1 = get_u32(&bytes, 4);
            put_u32(&mut bytes, 4, word1 + 4);
            put_u32(&mut bytes, domain_offset, 1 | (16 << 16));
            put_u32(&mut bytes, domain_offset + 4, 2);
            let request = parse_request_with_domain(&bytes, true).unwrap();
            let buffer = request.input_buffer.as_ref().unwrap();
            assert_eq!(buffer.kind, InputBufferKind::MapAlias);
            assert_eq!(buffer.pointer.as_ref().unwrap().offset, descriptor_start);
            assert_eq!(
                buffer.map_alias.as_ref().unwrap().offset,
                descriptor_start + 8
            );
            assert_eq!(buffer.address, 0x41_1234_5000);
            assert_eq!(buffer.size, 24);
            assert_eq!(request.hipc.raw_data_offset, raw_start);
            assert_eq!(request.hipc.arguments_offset, domain_offset + 32);
            assert_eq!(request.domain.as_ref().unwrap().object_id, 2);
            assert!(request.arguments.is_empty());
        }
    }

    #[test]
    fn domain_reply_objects_immediately_follow_unrounded_output_data() {
        let request =
            parse_request_with_domain(&domain_send(6, 0, &[], Some(7), 1, &[]), true).unwrap();
        let reply =
            build_domain_reply_with_data(&request, 0, 0xe1d7_f748, &[0x34, 0x12], &[2, 0]).unwrap();
        assert_eq!(reply.bytes.len(), TLS_MESSAGE_BUFFER_SIZE);
        assert_eq!(get_u32(&reply.bytes, 0), 0);
        assert_eq!(get_u32(&reply.bytes, 4), 15);
        assert_eq!(reply.used_bytes, 68);
        assert_eq!(reply.result_offset, Some(40));
        assert_eq!(reply.moved_handle_offset, None);
        assert!(reply.bytes[8..16].iter().all(|byte| *byte == 0));
        assert_eq!(get_u32(&reply.bytes, 16), 2);
        assert!(reply.bytes[20..32].iter().all(|byte| *byte == 0));
        assert_eq!(&reply.bytes[32..36], b"SFCO");
        assert_eq!(get_u32(&reply.bytes, 36), 0);
        assert_eq!(get_u32(&reply.bytes, 40), 0);
        assert_eq!(get_u32(&reply.bytes, 44), 0xe1d7_f748);
        assert_eq!(&reply.bytes[48..50], &[0x34, 0x12]);
        assert_eq!(get_u32(&reply.bytes, 50), 2);
        assert_eq!(get_u32(&reply.bytes, 54), 0);
        assert!(reply.bytes[58..].iter().all(|byte| *byte == 0));
        assert!(build_reply(&request, 0, None, 0).is_err());
        assert!(build_reply(&request, 0, Some(1), 0).is_err());
    }

    #[test]
    fn domain_error_replies_have_a_zero_object_header_and_no_outputs() {
        let request =
            parse_request_with_domain(&domain_send(4, 0, &[], None, 1, &[]), true).unwrap();
        let reply = build_domain_reply_with_data(&request, 0x1234, 0xe1d7_f748, &[], &[]).unwrap();
        assert_eq!(reply.used_bytes, 56);
        assert_eq!(get_u32(&reply.bytes, 4), 12);
        assert!(reply.bytes[16..32].iter().all(|byte| *byte == 0));
        assert_eq!(&reply.bytes[32..36], b"SFCO");
        assert_eq!(get_u32(&reply.bytes, 40), 0x1234);
        assert_eq!(get_u32(&reply.bytes, 44), 0xe1d7_f748);
        assert!(reply.bytes[48..].iter().all(|byte| *byte == 0));
        assert!(build_domain_reply_with_data(&request, 1, 0, &[0], &[]).is_err());
        assert!(build_domain_reply_with_data(&request, 1, 0, &[], &[0]).is_err());
        let ordinary = parse_request(&cmif(4, 0, &[], None)).unwrap();
        assert!(build_domain_reply_with_data(&ordinary, 0, 0, &[], &[]).is_err());
    }

    #[test]
    fn domain_request_and_reply_sizes_stop_at_the_tls_boundary() {
        for (pid, maximum) in [(None, 196), (Some(1), 184)] {
            let exact = domain_send(4, 0, &vec![0x5a; maximum], pid, 1, &[2]);
            assert_eq!(exact.len(), TLS_MESSAGE_BUFFER_SIZE);
            assert!(parse_request_with_domain(&exact, true).is_ok());
            let too_large = domain_send(4, 0, &vec![0x5a; maximum + 1], pid, 1, &[2]);
            assert!(parse_request_with_domain(&too_large, true).is_err());
        }
        let request =
            parse_request_with_domain(&domain_send(4, 0, &[], None, 1, &[]), true).unwrap();
        for objects in [&[][..], &[2_u32][..]] {
            let maximum = 200 - objects.len() * 4;
            let reply =
                build_domain_reply_with_data(&request, 0, 0, &vec![0; maximum], objects).unwrap();
            assert_eq!(reply.used_bytes, TLS_MESSAGE_BUFFER_SIZE);
            assert!(
                build_domain_reply_with_data(&request, 0, 0, &vec![0; maximum + 1], objects)
                    .is_err()
            );
        }
        assert!(
            build_domain_reply_with_data(&request, 0, 0, &[], &[2; MAX_DOMAIN_OBJECTS]).is_ok()
        );
        assert!(
            build_domain_reply_with_data(&request, 0, 0, &[], &[2; MAX_DOMAIN_OBJECTS + 1])
                .is_err()
        );
    }

    #[test]
    fn domain_close_preserves_the_request_without_inventing_a_result() {
        for object_id in [0, 1, u32::MAX] {
            let mut original = domain_close(object_id);
            let request = parse_request_with_domain(&original, true).unwrap();
            assert_eq!(request.kind, RequestKind::DomainClose);
            assert_eq!(request.command_id, 0);
            assert_eq!(request.cmif, None);
            assert!(request.arguments.is_empty());
            let domain = request.domain.as_ref().unwrap();
            assert_eq!(domain.command, DomainCommand::Close);
            assert_eq!(domain.object_id, object_id);
            assert_eq!(domain.data_size, 0);
            assert_eq!(domain.object_table_offset, 32);
            assert!(domain.input_objects.is_empty());
            original.resize(TLS_MESSAGE_BUFFER_SIZE, 0xcc);
            let reply = build_domain_close_reply(&request, &original).unwrap();
            assert_eq!(reply.bytes, original);
            assert_eq!(reply.used_bytes, 40);
            assert_eq!(reply.result_offset, None);
            assert_eq!(reply.moved_handle_offset, None);
            assert!(build_domain_reply_with_data(&request, 0, 0, &[], &[]).is_err());
            assert!(build_reply(&request, 0, None, 0).is_err());
            let different = domain_close(object_id ^ 1);
            assert!(build_domain_close_reply(&request, &different).is_err());
            assert!(build_domain_close_reply(&request, &original[..39]).is_err());
        }
    }

    #[test]
    fn domain_close_rejects_embedded_messages_objects_pid_copies_and_buffers() {
        let original = domain_close(1);
        assert!(parse_request(&original).is_err());
        for size in 0..original.len() {
            assert!(parse_request_with_domain(&original[..size], true).is_err());
        }
        for (object_count, data_size) in [(1, 0), (0, 4)] {
            let mut bytes = original.clone();
            bytes.resize(44, 0xa5);
            put_u32(&mut bytes, 4, 9);
            put_u32(&mut bytes, 16, 2 | (object_count << 8) | (data_size << 16));
            assert!(parse_request_with_domain(&bytes, true).is_err());
        }
        let mut with_pid = domain_send(4, 0, &[], Some(7), 1, &[]);
        put_u32(&mut with_pid, 4, 0x8000_0008);
        put_u32(&mut with_pid, 32, 2);
        with_pid.truncate(52);
        assert!(parse_request_with_domain(&with_pid, true).is_err());

        for handles in [&[0][..], &[0xffff_8001; MAX_COPY_HANDLES][..]] {
            let with_copies = with_copy_handles(&original, handles);
            assert!(
                parse_request_with_domain(&with_copies, true)
                    .unwrap_err()
                    .to_string()
                    .contains("copy handles")
            );
        }

        let mut with_buffer = auto_select((0, 0), (0x1000, 32), None);
        put_u32(&mut with_buffer, 32, 2);
        put_u32(&mut with_buffer, 36, 1);
        put_u32(&mut with_buffer, 40, 0);
        put_u32(&mut with_buffer, 44, 0);
        assert!(parse_request_with_domain(&with_buffer, true).is_err());
        let ordinary = parse_request(&cmif(4, 0, &[], None)).unwrap();
        assert!(build_domain_close_reply(&ordinary, &original).is_err());
    }
}
