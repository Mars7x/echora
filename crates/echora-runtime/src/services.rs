//! Local service-manager, logging and typed applet endpoints for the bootstrap.
//!
//! CMIF output uses generic SF framing, not Atmosphere's minimal SM compatibility
//! shim. The logger is a bounded local diagnostic sink, not a TMA/UART transport.
//! LM, logger and applet sessions may use bounded CMIF domains. appletOE
//! command 10 supports only the supplied patch's flags-zero policy and opens
//! a distinct proxy bound to an immutable bootstrap application identity.
//! Audited proxy getters create distinct typed endpoints.
//! WindowController can return its bound process ID; its other methods and
//! window/display/lifecycle/winding/debug backends remain unsupported.
//! CommonStateGetter can enable a shared exit-request configuration flag and
//! copy the application-owned lifecycle event. Its explicit foreground-launch
//! state starts with a pending focus notification and a signaled event, without
//! message consumption, dynamic lifecycle changes or event wait/reset operations.
//! SelfController additionally initializes one application-owned
//! logical display manager under an explicit bootstrap launch policy; it does
//! not bind VI services or create display resources. Command 91 copies a readable
//! alias of the application-owned, initially clear suspended-tick event; no
//! suspension accounting, wait, reset, signal or scheduler operations are modeled.
//! ApplicationFunctions command 130 copies a distinct, initially clear GPU-error
//! event under a no-submitted-GPU-workloads model, not a GPU-health guarantee.
//! GPU submission, error detection and event signaling remain unsupported.
//! AudioController retains independent
//! reference constructor state per endpoint, without child methods or a backend.
//! Root ID 1
//! and monotonically increasing, non-reused child IDs are local allocation
//! policies, not console ABI guarantees. Nested conversion and controls that
//! clone/copy shared object holders remain explicit unsupported operations.

use crate::ipc::{self, InputBufferKind, Protocol, Request, RequestKind, TLS_MESSAGE_BUFFER_SIZE};
use crate::logging::{LocalLogSink, LoggerContext};
use crate::memory::Memory;
use crate::runtime::Trap;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

/// Horizon's application handle table holds 1024 entries.
const MAX_LIVE_HANDLES: usize = 1024;
const FIRST_SESSION_HANDLE: u32 = 0x200;
const MAX_READABLE_EVENTS: usize = 1024;
const SUSPENDED_TICK_EVENT_ID: ReadableEventId = ReadableEventId(1);
const GPU_ERROR_EVENT_ID: ReadableEventId = ReadableEventId(2);
const LIFECYCLE_EVENT_ID: ReadableEventId = ReadableEventId(3);
const ADD_ON_CONTENT_EVENT_ID: ReadableEventId = ReadableEventId(4);
/// Provisional GetGraphicsScalingParameters reply (Eden 1a6d7d4 prototype).
const PROVISIONAL_GRAPHICS_SCALING_PARAMETERS: [u32; 3] = [10, 1, 1];
const APPLICATION_UNKNOWN_210_EVENT_ID: ReadableEventId = ReadableEventId(5);
const FRIEND_INVITATION_EVENT_ID: ReadableEventId = ReadableEventId(6);
const MAX_DOMAIN_OBJECTS: usize = 128;
const MAX_TRACE: usize = 1024;
const MAX_PRIVATE_PARSE_FAILURES: usize = 8;
const MAX_PARSE_FAILURE_REASON_BYTES: usize = 512;
const LM_INTERFACE_ID: u32 = 0xe1d7_f748;
const LOGGER_INTERFACE_ID: u32 = 0x3e81_dad2;
const HIPC_MANAGER_INTERFACE_ID: u32 = 0xec6b_e3ff;
// First four SHA256 bytes of nn::am::service::IApplicationProxyService as LE u32.
// This is the replying interface's debug ID, not the returned proxy's ID.
const APPLICATION_PROXY_SERVICE_INTERFACE_ID: u32 = 0xbb83_bce3;
// Replying nn::am::service::IApplicationProxy interface, not the child interface.
const APPLICATION_PROXY_INTERFACE_ID: u32 = 0x7d59_c154;
// First four SHA256 bytes of nn::am::service::IWindowController as LE u32.
const WINDOW_CONTROLLER_INTERFACE_ID: u32 = 0x0279_422d;
// First four SHA256 bytes of nn::am::service::ICommonStateGetter as LE u32.
const COMMON_STATE_GETTER_INTERFACE_ID: u32 = 0x32f2_7d77;
/// Reply interface token for aoc:u. CMIF replies do not carry an interface ID
/// on the wire in this prototype's reply format beyond its context field.
const AOC_INTERFACE_ID: u32 = 0;
// First four SHA256 bytes of nn::am::service::ISelfController as LE u32.
const SELF_CONTROLLER_INTERFACE_ID: u32 = 0x56ef_eaa1;
// First four SHA256 bytes of nn::am::service::IApplicationFunctions as LE u32.
const APPLICATION_FUNCTIONS_INTERFACE_ID: u32 = 0x21f9_610e;

/// The role is declared by this bootstrap, not inferred from an unvalidated
/// NPDM ProgramType grant. Copies carry identical immutable identity/provenance;
/// they do not represent separate applications or mutable lifecycle state.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct MainApplicationIdentity {
    process_id: u64,
    #[serde(rename = "program_id_hex", serialize_with = "serialize_program_id_hex")]
    program_id: u64,
    npdm_sha256: String,
}

/// Explicit reference-launcher policy, not a decoded NPDM grant or a claimed
/// console default. Eden application_creator.cpp chooses Application (1) and
/// AllForeground (0); am_types.h defines these numeric values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
struct ApplicationLaunchPolicy {
    applet_id: u32,
    library_applet_mode: u32,
}

impl ApplicationLaunchPolicy {
    const REFERENCE_APPLICATION: Self = Self {
        applet_id: 1,
        library_applet_mode: 0,
    };
}

/// Logical parameters from DisplayLayerManager::Initialize. No VI references,
/// visibility/focus state, display resources or graphics-supervisor session are
/// modeled. None denotes an unallocated ID, never a fabricated valid handle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct LogicalDisplayLayerManager {
    owner: MainApplicationIdentity,
    applet_id: u32,
    library_applet_mode: u32,
    shared_buffer_id: Option<u64>,
    shared_layer_id: Option<u64>,
    buffer_sharing_enabled: bool,
    blending_enabled: bool,
    managed_display_layers: BTreeSet<u64>,
    managed_recording_layers: BTreeSet<u64>,
}

/// Initial application state under the pinned frontend's foreground policy.
/// AppletManager::SetWindowSystem calls SetFocusState(InFocus) before Run;
/// LifecycleManager's pending focus flag signals its event despite an empty
/// unordered queue. Acknowledged raw zero is an unacknowledged constructor
/// value, not the named NotInFocus value (2). No dynamic producers/consumers or
/// display focus backend are implemented here.
#[derive(Clone, Debug, PartialEq, Eq)]
struct LogicalApplicationLifecycle {
    is_application: bool,
    requested_focus_state: u8,
    acknowledged_focus_state: u8,
    focus_state_changed_notification_enabled: bool,
    has_focus_state_changed: bool,
    applet_message_available: bool,
    unordered_messages: Vec<u32>,
    /// SelfController SetFocusHandlingMode (13) inputs. Eden 1a6d7d4 feeds
    /// `suspend` into a focus-handling state machine that only changes the
    /// focus state while backgrounded, which this foreground model never is.
    focus_handling: Option<FocusHandlingRequest>,
}

impl LogicalApplicationLifecycle {
    fn reference_foreground() -> Self {
        Self {
            is_application: true,
            requested_focus_state: 1,
            acknowledged_focus_state: 0,
            focus_state_changed_notification_enabled: true,
            has_focus_state_changed: true,
            applet_message_available: true,
            unordered_messages: Vec::new(),
            focus_handling: None,
        }
    }

    /// Eden PopMessageInOrderOfPriority restricted to the modeled sources:
    /// an application focus change, then the unordered queue. Resume, exit,
    /// sleep and mode-change messages are never produced here.
    fn pop_message(&mut self) -> Option<u32> {
        if self.is_application
            && self.focus_state_changed_notification_enabled
            && self.has_focus_state_changed
        {
            self.has_focus_state_changed = false;
            return Some(APPLET_MESSAGE_FOCUS_STATE_CHANGED);
        }
        if self.unordered_messages.is_empty() {
            None
        } else {
            Some(self.unordered_messages.remove(0))
        }
    }

    /// Eden ShouldSignalSystemEvent restricted to the modeled state.
    fn should_signal(&self) -> bool {
        (self.is_application
            && self.focus_state_changed_notification_enabled
            && self.has_focus_state_changed)
            || !self.unordered_messages.is_empty()
    }

    /// Eden SignalSystemEventIfNeeded caches the evaluation in
    /// m_applet_message_available; the event bit follows the cache.
    fn refresh_availability(&mut self) -> bool {
        self.applet_message_available = self.should_signal();
        self.applet_message_available
    }

    fn report(&self) -> serde_json::Value {
        serde_json::json!({
            "is_application":self.is_application,
            "requested_focus_state":self.requested_focus_state,
            "acknowledged_focus_state":self.acknowledged_focus_state,
            "focus_state_changed_notification_enabled":self.focus_state_changed_notification_enabled,
            "has_focus_state_changed":self.has_focus_state_changed,
            "applet_message_available":self.applet_message_available,
            "unordered_message_count":self.unordered_messages.len(),
            "focus_handling":self.focus_handling,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct MainApplicationState {
    launch_policy: ApplicationLaunchPolicy,
    display_layer_manager: Option<LogicalDisplayLayerManager>,
    lifecycle: LogicalApplicationLifecycle,
    lifecycle_event: ReadableEventId,
    request_exit_to_library_applet_at_execute_next_program_enabled: bool,
    suspended_ticks: u64,
    accumulated_suspended_tick_changed_event: ReadableEventId,
    gpu_error_detected_event: ReadableEventId,
    add_on_content_list_changed_event: ReadableEventId,
    application_unknown_210_event: ReadableEventId,
    friend_invitation_storage_channel_event: ReadableEventId,
    /// WindowController AcquireForegroundRights (10) was called. Eden 1a6d7d4
    /// treats it as success with no window-system effect; under the foreground
    /// launch policy the application already owns the foreground.
    foreground_rights_acquired: bool,
    /// SelfController setters reached at startup; values stored as given.
    self_settings: SelfControllerSettings,
    /// ApplicationFunctions NotifyRunning (40) was called.
    running_notified: bool,
}

/// Native-PC launch policy: the port presents as docked (TV output), the
/// console mode with the larger render targets, and therefore reports the
/// docked performance profile. nn::oe OperationMode::Docked = 1 and APM
/// PerformanceMode::Boost = 1 (Eden 1a6d7d4 am_types.h / apm_controller.h).
const OPERATION_MODE_DOCKED: u8 = 1;
const PERFORMANCE_MODE_BOOST: i32 = 1;

/// Stored SelfController settings (Eden 1a6d7d4 ISelfController setters).
/// These change only application-owned configuration: no screenshot, album,
/// display-request or suspension backend exists to act on them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
struct SelfControllerSettings {
    screenshot_permission: Option<u32>,
    operation_mode_changed_notification: Option<bool>,
    performance_mode_changed_notification: Option<bool>,
    restart_message_enabled: Option<bool>,
    out_of_focus_suspending_enabled: Option<bool>,
    album_image_orientation: Option<u32>,
    handles_request_to_display: Option<bool>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
struct FocusHandlingRequest {
    notify: bool,
    background: bool,
    suspend: bool,
}

/// Progress of the application's graphics-supervisor bootstrap.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum GraphicsSupervisorPhase {
    Created,
    Initialized,
    ProcessBound,
    Ready,
}

/// A KTransferMemory: the owner's locked range and its permission while
/// locked. There is one reference (the handle); closing it unlocks the range.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub(crate) struct TransferMemory {
    pub address: u64,
    pub size: u64,
    pub permission: u8,
}

/// Syncpoint cache shared through two transfer memories (IGraphicsSupervisor
/// command 204). Undocumented on Switchbrew; the layout below is the one the
/// pinned Eden 1a6d7d4 graphics_syncpoint_cache.h serves to this application:
/// a 0x1000 request page written by the application and a 0x2000 page the
/// server fills (header, 192 u64 counter values, group/slot acknowledgements).
const SYNCPOINT_CACHE_COUNTERS: u64 = 192;
const SYNCPOINT_CACHE_GROUPS: u32 = 3;
const SYNCPOINT_CACHE_VALUES: u64 = 0x40;
const SYNCPOINT_CACHE_GROUP_REQUEST: u64 = 0;
const SYNCPOINT_CACHE_SLOT_REQUEST: u64 = 8;
const SYNCPOINT_CACHE_GROUP_ACK: u64 = 0x640;
const SYNCPOINT_CACHE_SLOT_ACK: u64 = 0x648;
const SYNCPOINT_UNALLOCATED: u64 = u64::MAX;
const SYNCPOINT_MAX_EVENT_INDEX: u32 = 63;
const GPU_NOTIFICATION_MAX_ID: u32 = 63;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct SyncpointCache {
    rw: TransferMemory,
    ro: TransferMemory,
    rw_handle: u32,
    ro_handle: u32,
    request_event: ReadableEventId,
    completed_event: ReadableEventId,
    refreshes: u64,
}

/// Interface tag written into this prototype's nvdrv CMIF replies.
const NVDRV_INTERFACE_ID: u32 = 0x6e76_6472;
const NVDRV_MAX_FDS: usize = 64;
/// NvResult codes for Open of an unavailable device.
const NV_NOT_SUPPORTED: u32 = 0x2;
const NV_FILE_OPERATION_FAILED: u32 = 0x3_0003;
/// Eden 1a6d7d4 extended GPU address-space layout (40-bit VA).
const GPU_VA_USER_START: u64 = 0x800_0000;
const GPU_VA_SPLIT: u64 = 3 << 38;
const GPU_VA_END: u64 = 1 << 40;
/// Device paths listed on Switchbrew "NV services".
/// Work buffer size answered by IAudioRendererManager GetWorkBufferSize.
const AUDIO_RENDERER_WORK_BUFFER_SIZE: u64 = 0x1000;

const NVDRV_DEVICES: &[&str] = &[
    "/dev/nvhost-ctrl",
    "/dev/nvmap",
    "/dev/nvdisp-ctrl",
    "/dev/nvdisp-disp0",
    "/dev/nvdisp-disp1",
    "/dev/nvhdcp_up-ctrl",
    "/dev/nvcec-ctrl",
    "/dev/nvdcutil-disp0",
    "/dev/nvdcutil-disp1",
    "/dev/nvsched-ctrl",
    "/dev/nverpt-ctrl",
    "/dev/nvhost-as-gpu",
    "/dev/nvhost-dbg-gpu",
    "/dev/nvhost-prof-gpu",
    "/dev/nvhost-ctrl-gpu",
    "/dev/nvhost-tsg-gpu",
    "/dev/nvhost-prof-dev-gpu",
    "/dev/nvhost-gpu",
    "/dev/nvhost-msenc",
    "/dev/nvhost-nvdec",
    "/dev/nvhost-nvjpg",
    "/dev/nvhost-vic",
    "/dev/nvhost-tsec",
    "/dev/nvhost-display",
];

/// Interface tag written into this prototype's fsp-srv CMIF replies.
const FSP_INTERFACE_ID: u32 = 0x6673_7073;
/// Local bound on one storage read.
const MAX_STORAGE_READ: u64 = 64 << 20;

/// Horizon pseudo-handle for the current process.
const CURRENT_PROCESS_PSEUDO_HANDLE: u32 = 0xffff_8001;

/// nn::am AppletMessage::FocusStateChanged (Eden 1a6d7d4 am_types.h).
const APPLET_MESSAGE_FOCUS_STATE_CHANGED: u32 = 15;
/// AM ResultNoMessages: module 128, description 3 (Eden am_results.h).
const AM_RESULT_NO_MESSAGES: u32 = 128 | (3 << 9);
/// AM ResultNoDataInChannel (module 128, description 2; Eden 1a6d7d4).
const AM_RESULT_NO_DATA_IN_CHANNEL: u32 = 128 | (2 << 9);
/// AM ResultInvalidOffset (module 128, description 503; Eden 1a6d7d4).
const AM_RESULT_INVALID_OFFSET: u32 = 128 | (503 << 9);
/// LaunchParameterAccountPreselectedUser magic (Eden 1a6d7d4 applet_manager).
const PRESELECTED_USER_MAGIC: u32 = 0xc794_97ca;
/// Account 2124-0200, network service account registration required
/// (Switchbrew "Error codes").
const ACCOUNT_RESULT_NSA_REGISTRATION_REQUIRED: u32 = 124 | (200 << 9);

/// Local backing-object identity, distinct from a guest handle or domain ID.
/// Each modeled event is owned by the application for its whole lifetime.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
struct ReadableEventId(u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ReadableEventKind {
    AccumulatedSuspendedTickChanged,
    GpuErrorDetected,
    LifecycleSystem,
    /// it is never signaled.
    AddOnContentListChanged,
    /// Syncpoint-cache refresh request (the application holds the writable
    /// side and signals it after writing request bits).
    SyncpointCacheRequest,
    /// ApplicationFunctions command 210's event (Switchbrew: "[20.0.0+]",
    /// unnamed). Provisional, from Eden 1a6d7d4 (Applet::unknown_event): an
    /// applet-owned event the reference never signals.
    ApplicationUnknown210,
    /// ApplicationFunctions GetFriendInvitationStorageChannelEvent (140):
    /// signaled when a friend invitation delivers data. Nothing sends
    /// invitations here, so it is never signaled.
    FriendInvitationStorageChannel,
    /// Signaled after the server refreshes the syncpoint cache.
    SyncpointCacheCompleted,
    /// An async context's completion event (account, play report); the
    /// contexts finish at creation, so it starts signaled.
    AsyncContextDone,
    /// DisplaySession notification event (GetNotificationEvent, 11). No
    /// producer exists yet, so it is never signaled.
    DisplayNotification,
    /// DisplaySession command 31's event (undocumented; u32 index 0, moved
    /// like command 11's, released by 32). Provisional and never signaled.
    DisplayEvent31,
    /// NVIDIA driver device event (QueryEvent): never signaled, as nothing
    /// produces GPU errors or semaphore notifications here.
    NvdrvEvent {
        device: &'static str,
        id: u32,
    },
    /// GpuSession notification event by index (CreateEvent, command 12).
    GpuNotification {
        index: u32,
    },
    /// IAudioDevice QueryAudioDeviceOutputEvent (12): signaled when the
    /// output device changes. There is one fixed output, so it never is.
    AudioDeviceOutput,
    /// IAudioOut RegisterBufferEvent (4): signaled when a buffer is released.
    AudioOutBuffer,
    /// IAudioRenderer QuerySystemEvent (7): signaled as each frame renders.
    AudioRendererFrame,
    /// Internal: a parked bsd:u Poll, signaled when it completes.
    SocketPoll,
    /// nfp IUser AttachAvailabilityChangeEvent (23); never signaled.
    NfpAvailability,
    /// Syncpoint supervisor command 22's event for an index.
    SyncpointEvent {
        index: u32,
    },
    /// Video decoder GetInputEvent (3): signaled while input is accepted.
    VideoDecoderInput,
    /// Library applet state-changed and pop-out-data events.
    LibraryApplet,
    /// nifm IRequest state-change and second event.
    NetworkRequest,
    /// hid AcquireNpadStyleSetUpdateEventHandle (106); no controller
    /// connects, so it never signals.
    NpadStyleSetUpdate,
    /// bcat IDeliveryCacheProgressService event; signaled because the sync
    /// is already done.
    BcatProgress,
}

/// One audio output. Appended buffers play back to back in real time at the
/// stream's sample rate once started; finished buffers move to the released
/// list and signal the buffer event. With ECHORA_AUDIO set, each appended
/// buffer's samples also go to the host (`crate::audio`).
#[derive(Clone, Debug, Serialize)]
struct AudioOutState {
    sample_rate: u32,
    channels: u32,
    started: bool,
    volume: f32,
    /// (client tag, sample frames) in play order.
    queue: std::collections::VecDeque<(u64, u64)>,
    released: std::collections::VecDeque<u64>,
    played_samples: u64,
    /// Host time at which the front buffer finishes, while started.
    #[serde(skip)]
    front_end: Option<std::time::Instant>,
    event: Option<ReadableEventId>,
    /// Host playback stream, when host audio is on.
    #[serde(skip)]
    host: Option<std::sync::mpsc::Sender<Vec<u8>>>,
}

/// Samples per channel at `sample_rate` that an Opus packet decodes to, from
/// its TOC byte (RFC 6716 section 3.1): the configuration's frame duration
/// times the frame count (code 0: one, 1-2: two, 3: the count byte).
fn opus_packet_samples(packet: &[u8], sample_rate: u32) -> Option<u64> {
    let toc = *packet.first()?;
    let config = toc >> 3;
    // Frame duration in units of 2.5 ms.
    let units: u64 = match config {
        0..=11 => [4, 8, 16, 24][usize::from(config % 4)],
        12..=15 => [4, 8][usize::from(config % 2)],
        _ => [1, 2, 4, 8][usize::from(config % 4)],
    };
    let frames = match toc & 3 {
        0 => 1,
        1 | 2 => 2,
        _ => u64::from(*packet.get(1)? & 0x3f),
    };
    if frames == 0 || frames * units > 48 {
        // A packet holds at most 120 ms of audio.
        return None;
    }
    Some(frames * units * u64::from(sample_rate) / 400)
}

/// Local bound on queued plus released buffers per audio output.
const MAX_AUDIO_OUT_BUFFERS: usize = 64;

impl AudioOutState {
    fn duration(&self, frames: u64) -> std::time::Duration {
        std::time::Duration::from_nanos(frames * 1_000_000_000 / u64::from(self.sample_rate))
    }

    /// Release every buffer that has finished by `now`; returns whether any
    /// did.
    fn advance(&mut self, now: std::time::Instant) -> bool {
        let mut released = false;
        while self.started {
            let Some(&(tag, frames)) = self.queue.front() else {
                self.front_end = None;
                break;
            };
            let end = match self.front_end {
                Some(end) => end,
                None => {
                    let end = now + self.duration(frames);
                    self.front_end = Some(end);
                    end
                }
            };
            if end > now {
                break;
            }
            self.queue.pop_front();
            self.released.push_back(tag);
            self.played_samples += frames;
            released = true;
            // The next buffer starts where this one ended.
            self.front_end = self
                .queue
                .front()
                .map(|&(_, next)| end + self.duration(next));
        }
        released
    }
}

/// Aliases of an event reference one record, never cloned per-alias signal bits.
/// Suspended/GPU events start clear; the lifecycle event starts signaled under
/// the explicit foreground-launch policy. Its bit remains independent of the
/// lifecycle's cached availability. No dynamic producer, wait/reset API,
/// suspension accounting or automatic-clear behavior is implemented.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct ReadableEvent {
    owner: MainApplicationIdentity,
    kind: ReadableEventKind,
    signaled: bool,
}

/// Per-object defaults from Eden 1a6d7d4's IAudioController constructor/header,
/// not verified console volume or timing behavior. Raw IEEE-754 binary32 bits
/// retain exact values and Eq semantics without introducing host audio effects.
/// The signed nanosecond field is reference storage only: no timer is scheduled
/// and command 3's console argument interpretation remains unsupported.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
struct AudioControllerState {
    main_applet_volume_bits: u32,
    library_applet_volume_bits: u32,
    transparent_volume_rate_bits: u32,
    reference_fade_duration_ns: i64,
}

impl AudioControllerState {
    const REFERENCE_DEFAULTS: Self = Self {
        main_applet_volume_bits: 0x3e80_0000,
        library_applet_volume_bits: 0x3f80_0000,
        transparent_volume_rate_bits: 0,
        reference_fade_duration_ns: 0,
    };
}

fn serialize_program_id_hex<S: serde::Serializer>(
    value: &u64,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(&format!("{value:016x}"))
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ServiceObject {
    ServiceManager {
        registered_process_id: Option<u64>,
    },
    LogManager,
    Logger {
        process_id: u64,
    },
    ApplicationProxyService,
    ApplicationProxy {
        application: MainApplicationIdentity,
    },
    CommonStateGetter {
        application: MainApplicationIdentity,
    },
    SelfController {
        application: MainApplicationIdentity,
    },
    WindowController {
        application: MainApplicationIdentity,
    },
    AudioController {
        state: AudioControllerState,
    },
    DisplayController {
        application: MainApplicationIdentity,
    },
    ProcessWindingController {
        application: MainApplicationIdentity,
    },
    LibraryAppletCreator {
        application: MainApplicationIdentity,
    },
    /// ILibraryAppletAccessor; its state is `Services::library_applets[id]`.
    LibraryAppletAccessor {
        id: usize,
    },
    /// A writable IStorage made by the library applet creator or an
    /// applet, and its IStorageAccessor; contents in
    /// `Services::applet_storages[id]`.
    LibraryStorage {
        id: usize,
    },
    LibraryStorageAccessor {
        id: usize,
    },
    ApplicationFunctions {
        application: MainApplicationIdentity,
    },
    PerformanceFunctions,
    GraphicsScalingParametersFunctions {
        application: MainApplicationIdentity,
    },
    DebugFunctions,
    /// acc:u0 IAccountServiceForApplication.
    AccountService,
    /// AM IStorage holding an immutable launch-parameter buffer, and the
    /// IStorageAccessor opened on it.
    AppletStorage {
        data: Vec<u8>,
    },
    AppletStorageAccessor {
        data: Vec<u8>,
    },
    /// IManagerForApplication (BaaS account manager) for one local user.
    AccountManager {
        uid: [u8; 16],
    },
    /// IAsyncNetworkServiceLicenseKindContext from
    /// LoadNetworkServiceLicenseKindAsync (170): finished at creation.
    AccountLicenseContext,
    /// IProfile for one local user.
    AccountProfile {
        uid: [u8; 16],
    },
    /// aoc:u IAddOnContentManager. Stateless: answers from the trusted main
    /// application identity and the selected add-on inventory.
    AddOnContentManager,
    /// ectx:aw error-context writer and the registrar it creates.
    ErrorContextWriter,
    ErrorContextRegistrar,
    /// pctl IParentalControlServiceFactory and the service it creates.
    ParentalControlFactory,
    ParentalControl {
        initialized: bool,
    },
    /// time:u IStaticService and the clock / time-zone objects it returns.
    TimeStatic,
    SystemClock {
        clock: u32,
    },
    SteadyClock,
    TimeZone,
    /// ISelfController GetGraphicsSupervisor (2000) session: a separate
    /// client session copied to the guest, not a domain object.
    GraphicsSupervisor {
        application: MainApplicationIdentity,
    },
    /// IGraphicsSupervisor GetNvidiaDriver (1402): the process-bound NVIDIA
    /// driver session ("nvdrv:gsv" in Eden 1a6d7d4).
    NvidiaDriver {
        application: MainApplicationIdentity,
    },
    /// IGraphicsSupervisor GetSyncpointSession (202) session (TIPC).
    SyncpointSupervisor,
    /// VideoDecoderCreatorSession from graphics supervisor 906 (TIPC).
    VideoDecoderCreator,
    /// VideoEncoderCreatorSession from graphics supervisor 1002 (TIPC).
    VideoEncoderCreator,
    /// VideoDecoderSession / VideoEncoderSession from the creators'
    /// command 1 (TIPC; Switchbrew names their commands only).
    VideoDecoder,
    VideoEncoder,
    /// DisplaySession from IGraphicsSupervisor InitializeDisplayContext (305).
    DisplaySession {
        state: crate::display::DisplaySessionState,
    },
    /// IGraphicsSupervisor CreateGpuSessionWithRemoteMemory (497) session.
    GpuSession,
    /// GpuSession GetKUserWorkerSession (22): nn::gpuk KUserWorkerSession.
    GpuWorker,
    /// fsp-srv IFileSystemProxy.
    FileSystemProxy,
    /// IStorage over the application's RomFS (OpenDataStorageByCurrentProcess).
    DataStorage,
    /// A host-backed IFileSystem (`Services::file_systems[id]`) and the
    /// IFile / IDirectory objects opened on it.
    FileSystem {
        id: usize,
    },
    File {
        id: usize,
    },
    Directory {
        id: usize,
    },
    /// aud:u, the target's audio entry point. Inferred from Eden's NS2
    /// INativeAudioManager: GetAudioInManager (0), GetAudioOutManager (1) and
    /// GetAudioRendererManager (2) return the audin:u, audout:u and audren:u
    /// manager interfaces.
    AudioManager,
    AudioInManager,
    AudioOutManager,
    AudioRendererManager,
    /// auddmg:u, nn::audio::devmgr::IUserSessionManager ([S2], Switchbrew
    /// Audio_services), and the IUserSession its command 0 returns.
    AudioDeviceSessionManager,
    AudioDeviceSession,
    /// hwopus IHardwareOpusDecoderManager.
    OpusDecoderManager,
    /// bsd:u ISocketClient with no network: sockets cannot be created.
    SocketClient,
    /// ssl ISslService and an ISslContext. Contexts only record options and
    /// PKI imports; with no network no connection is ever established.
    SslService,
    SslContext,
    /// nsd:u, network service discovery (Eden-NS2 1a6d7d4 nsd.cpp).
    NsdUser,
    /// nfp:user (amiibo) IUserManager and the IUser it creates; the host
    /// has no NFC devices.
    NfpUserManager,
    NfpUser {
        initialized: bool,
    },
    /// nifm:u IStaticService, IGeneralService and IRequest with no network.
    NifmStatic,
    NifmGeneral,
    /// Its events are `Services::nifm_requests[id]`.
    NifmRequest {
        id: usize,
    },
    /// bcat:u with an empty delivery cache: IServiceCreator, IBcatService,
    /// IDeliveryCacheStorageService, its file and directory services, and
    /// IDeliveryCacheProgressService.
    BcatCreator,
    BcatService,
    BcatStorage,
    BcatFile,
    BcatDirectory,
    BcatProgress,
    /// hid IHidServer and its IAppletResource.
    HidServer,
    HidAppletResource,
    /// hid IActiveVibrationDeviceList (CreateActiveVibrationDeviceList).
    HidVibrationDeviceList,
    /// stpl:u nn::streamplay::IUserService and the IUserShimScopedObject
    /// its command 0 returns (Switchbrew "Switch 2: Chat services").
    StreamplayUser,
    StreamplayShim,
    /// prepo:u nn::prepo::detail::ipc::IPrepoService (play reports).
    PlayReport,
    /// nn::prepo::detail::ipc::IAsyncContext, from a report sent with
    /// command 10500 (SDK 22's PlayReport::Send).
    PlayReportContext,
    /// IHardwareOpusDecoder: decodes packets to silence of the right length.
    OpusDecoder {
        sample_rate: u32,
        channels: u32,
    },
    /// IAudioRenderer from OpenAudioRenderer; its state is
    /// `Services::audio_renderers[id]`.
    AudioRenderer {
        id: usize,
    },
    /// IAudioOut from OpenAudioOut; its state is `Services::audio_outs[id]`.
    AudioOut {
        id: usize,
    },
    /// IAudioDevice from GetAudioDeviceServiceWithRevisionInfo.
    AudioDevice {
        revision: u32,
    },
}

impl ServiceObject {
    fn service(&self) -> &'static str {
        match self {
            Self::ServiceManager { .. } => "sm:",
            Self::LogManager => "lm",
            Self::Logger { .. } => "lm:logger",
            Self::ApplicationProxyService => "appletOE",
            Self::AddOnContentManager => "aoc:u",
            Self::ErrorContextWriter => "ectx:aw",
            Self::ErrorContextRegistrar => "ectx:aw:context_registrar",
            Self::ParentalControlFactory => "pctl",
            Self::ParentalControl { .. } => "pctl:service",
            Self::TimeStatic => "time:u",
            Self::AccountService => "acc:u0",
            Self::AppletStorage { .. } => "appletOE:storage",
            Self::AppletStorageAccessor { .. } => "appletOE:storage_accessor",
            Self::AccountProfile { .. } => "acc:u0:profile",
            Self::AccountManager { .. } => "acc:u0:manager_for_application",
            Self::AccountLicenseContext => "acc:u0:async_license_kind",
            Self::SystemClock { .. } => "time:u:system_clock",
            Self::SteadyClock => "time:u:steady_clock",
            Self::TimeZone => "time:u:time_zone",
            Self::GraphicsSupervisor { .. } => "appletOE:graphics_supervisor",
            Self::NvidiaDriver { .. } => "nvdrv:gsv",
            Self::SyncpointSupervisor => "appletOE:syncpoint_supervisor",
            Self::VideoDecoderCreator => "appletOE:video_decoder_creator",
            Self::VideoEncoderCreator => "appletOE:video_encoder_creator",
            Self::VideoDecoder => "appletOE:video_decoder",
            Self::VideoEncoder => "appletOE:video_encoder",
            Self::GpuSession => "appletOE:gpu_session",
            Self::DisplaySession { .. } => "appletOE:display_session",
            Self::GpuWorker => "appletOE:gpu_worker",
            Self::FileSystemProxy => "fsp-srv",
            Self::DataStorage => "fsp-srv:data_storage",
            Self::FileSystem { .. } => "fsp-srv:file_system",
            Self::File { .. } => "fsp-srv:file",
            Self::Directory { .. } => "fsp-srv:directory",
            Self::AudioManager => "aud:u",
            Self::AudioInManager => "aud:u:audin",
            Self::AudioOutManager => "aud:u:audout",
            Self::AudioRendererManager => "aud:u:audren",
            Self::AudioDevice { .. } => "aud:u:audio_device",
            Self::AudioOut { .. } => "aud:u:audio_out",
            Self::AudioRenderer { .. } => "aud:u:audio_renderer",
            Self::OpusDecoderManager => "hwopus",
            Self::AudioDeviceSessionManager => "auddmg:u",
            Self::AudioDeviceSession => "auddmg:u:session",
            Self::SocketClient => "bsd:u",
            Self::SslService => "ssl",
            Self::SslContext => "ssl:context",
            Self::NifmStatic => "nifm:u",
            Self::NsdUser => "nsd:u",
            Self::NfpUserManager => "nfp:user",
            Self::NfpUser { .. } => "nfp:user:user",
            Self::NifmGeneral => "nifm:u:general",
            Self::NifmRequest { .. } => "nifm:u:request",
            Self::BcatCreator => "bcat:u",
            Self::BcatService => "bcat:u:service",
            Self::BcatStorage => "bcat:u:storage",
            Self::BcatFile => "bcat:u:file",
            Self::BcatDirectory => "bcat:u:directory",
            Self::BcatProgress => "bcat:u:progress",
            Self::StreamplayUser => "stpl:u",
            Self::PlayReport => "prepo:u",
            Self::PlayReportContext => "prepo:u:async_context",
            Self::HidServer => "hid",
            Self::HidAppletResource => "hid:applet_resource",
            Self::HidVibrationDeviceList => "hid:vibration_device_list",
            Self::StreamplayShim => "stpl:u:shim",
            Self::OpusDecoder { .. } => "hwopus:decoder",
            Self::ApplicationProxy { .. } => "appletOE:application_proxy",
            Self::CommonStateGetter { .. } => "appletOE:common_state_getter",
            Self::SelfController { .. } => "appletOE:self_controller",
            Self::WindowController { .. } => "appletOE:window_controller",
            Self::AudioController { .. } => "appletOE:audio_controller",
            Self::DisplayController { .. } => "appletOE:display_controller",
            Self::ProcessWindingController { .. } => "appletOE:process_winding_controller",
            Self::LibraryAppletCreator { .. } => "appletOE:library_applet_creator",
            Self::LibraryAppletAccessor { .. } => "appletOE:library_applet_accessor",
            Self::LibraryStorage { .. } => "appletOE:library_storage",
            Self::LibraryStorageAccessor { .. } => "appletOE:library_storage_accessor",
            Self::ApplicationFunctions { .. } => "appletOE:application_functions",
            Self::PerformanceFunctions => "appletOE:performance_functions",
            Self::GraphicsScalingParametersFunctions { .. } => {
                "appletOE:graphics_scaling_parameters_functions"
            }
            Self::DebugFunctions => "appletOE:debug_functions",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
enum Session {
    Regular {
        object: ServiceObject,
    },
    Domain {
        original_service: &'static str,
        next_object_id: u32,
        objects: BTreeMap<u32, ServiceObject>,
    },
}

impl Session {
    fn regular(object: ServiceObject) -> Self {
        Self::Regular { object }
    }

    fn service(&self) -> &'static str {
        match self {
            Self::Regular { object } => object.service(),
            Self::Domain {
                original_service, ..
            } => original_service,
        }
    }

    fn is_domain(&self) -> bool {
        matches!(self, Self::Domain { .. })
    }

    fn object(&self, object_id: Option<u32>) -> Option<&ServiceObject> {
        match (self, object_id) {
            (Self::Regular { object }, None) => Some(object),
            (Self::Domain { objects, .. }, Some(id)) => objects.get(&id),
            _ => None,
        }
    }

    fn object_mut(&mut self, object_id: Option<u32>) -> Option<&mut ServiceObject> {
        match (self, object_id) {
            (Self::Regular { object }, None) => Some(object),
            (Self::Domain { objects, .. }, Some(id)) => objects.get_mut(&id),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct IpcEvent {
    pub pc: u64,
    pub handle: u32,
    pub service: String,
    pub protocol: String,
    pub request_kind: String,
    pub domain_object_id: Option<u32>,
    pub input_domain_objects: Vec<u32>,
    pub created_domain_objects: Vec<u32>,
    pub command_id: u32,
    pub has_pid_descriptor: bool,
    /// Raw descriptor values remain untrusted; ordinary traces retain only the count.
    pub input_copy_handle_count: usize,
    /// Published copied handles only; failed requests leave this count zero.
    pub output_copy_handle_count: usize,
    pub argument_bytes: usize,
    pub input_buffer: Option<ipc::InputBuffer>,
    pub requested_service: Option<String>,
    pub completed: bool,
    pub service_result: Option<u32>,
    pub log_disposition: Option<String>,
}

/// Private diagnostic data from an already-read TLS message. These snapshots
/// are deliberately separate from the ordinary event trace and may contain
/// guest data; they belong only in the full local report, never a checkpoint.
#[derive(Clone, Debug, Serialize)]
struct PrivateIpcParseFailure {
    pc: u64,
    handle: u32,
    service: String,
    tls_address: u64,
    domain_session: bool,
    reason: String,
    reason_truncated: bool,
    tls_snapshot_hex: String,
}

pub struct Services {
    next_handle: u32,
    sessions: BTreeMap<u32, Session>,
    /// Commands answered by an explicit provisional (non-reference) policy.
    provisional_inputs: BTreeMap<&'static str, Vec<u8>>,
    /// Selected add-on content IDs (aoc:u). Empty unless a content manifest
    /// selects DLC; an ExeFS-only run genuinely has none installed.
    add_on_content: Vec<u64>,
    error_contexts_completed: u64,
    error_context_bytes: u64,
    parental_control_grants: BTreeSet<u32>,
    /// ACI0 service access control; `None` (tests) disables the check.
    service_access: Option<Vec<String>>,
    denied_services: BTreeSet<String>,
    /// Handle of the one GetGraphicsSupervisor client session, once created.
    graphics_supervisor_handle: Option<u32>,
    graphics_supervisor_phase: Option<GraphicsSupervisorPhase>,
    nvdrv_firmware_memory_margin: Option<bool>,
    /// Open NVIDIA driver file descriptors and their device paths.
    nvdrv_fds: BTreeMap<u32, String>,
    /// Page ranges SetGpuMemoryAttribute marked GpuSharable (0x20), kept
    /// here rather than in the guest mappings, as (start, end).
    pub(crate) gpu_sharable: BTreeMap<u64, u64>,
    /// KUserWorkerSession GPU memory by handle: (start, end, two u8 flags).
    gpu_memory: BTreeMap<u32, (u64, u64, [u8; 2])>,
    /// NormalMapGpuMemory4K mappings by GPU address: (address space, last
    /// GPU address, memory handle, memory offset).
    gpu_mappings: BTreeMap<u64, (u32, u64, u32, u64)>,
    /// Per-descriptor NVIDIA device state.
    nvdrv_devices: crate::nvdrv::Devices,
    /// QueryEvent backings by (device path, event ID); each is created once.
    nvdrv_events: BTreeMap<(String, u32), ReadableEventId>,
    /// Live transfer-memory handles (CreateTransferMemory).
    transfer_memories: BTreeMap<u32, TransferMemory>,
    /// Transfer memories referenced by a server object after their handle
    /// closed; they stay locked while that object lives.
    retained_transfer_memories: Vec<TransferMemory>,
    /// Handles to the writable side of an event (SignalEvent/ClearEvent).
    writable_event_handles: BTreeMap<u32, ReadableEventId>,
    syncpoint_cache: Option<SyncpointCache>,
    syncpoint_event_count: Option<u32>,
    /// Graphics-supervisor context setup commands completed (create/init).
    graphics_contexts: BTreeSet<u32>,
    gpu_session_version: Option<u64>,
    gpu_events: BTreeMap<u32, ReadableEventId>,
    /// GpuSession graphics devices by ID: (packed name, parameter, label).
    graphics_devices: BTreeMap<u8, [u64; 3]>,
    /// GpuSession address spaces by handle, with their creation flags.
    gpu_address_spaces: BTreeMap<u32, u32>,
    gpu_worker_ready: bool,
    graphics_scale_changed_notification: Option<bool>,
    exit_locked: bool,
    /// Cloned client handles and the primary handle of their server session.
    session_aliases: BTreeMap<u32, u32>,
    fs_current_process: Option<u64>,
    fs_access_log_mode: u32,
    data_romfs: Option<crate::romfs::VirtualRomfs>,
    /// Host-backed file systems, open files and open directory listings
    /// (entries plus read cursor).
    file_systems: Vec<crate::fs::HostFileSystem>,
    open_files: Vec<std::path::PathBuf>,
    open_directories: Vec<(Vec<crate::fs::Entry>, usize)>,
    data_storage_bytes_read: u64,
    /// Diagnostics: read count, last offset, and the 64 KiB RomFS chunks
    /// touched, to tell progressing loads from repeated reads.
    data_storage_reads: u64,
    data_storage_last_offset: u64,
    data_storage_chunks: std::collections::BTreeSet<u64>,
    cpu_boost_mode: Option<u32>,
    syncpoint_context_initialized: bool,
    steady_epoch: std::time::Instant,
    steady_clock_source: [u8; 16],
    /// The single local user (native-PC configuration, see `LocalUser`).
    local_user: LocalUser,
    /// acc:u0 InitializeApplicationInfo was called by the main application.
    account_application_info_initialized: bool,
    /// The AccountPreselectedUser launch parameter was popped.
    preselected_user_popped: bool,
    /// NACP UserAccountSwitchLock (offset 0x3026, Switchbrew "NACP") when
    /// `ECHORA_NACP` names the extracted control.nacp; otherwise the error
    /// that reading it produced, or None when no NACP was supplied.
    nacp_user_account_switch_lock: Option<Result<u8, String>>,
    shared_memories: BTreeMap<u32, SharedMemoryObject>,
    shared_memory_handles: BTreeMap<u32, u32>,
    provisional_commands: BTreeSet<&'static str>,
    readable_event_handles: BTreeMap<u32, ReadableEventId>,
    // Runtime owns each thread record/TLS; this table owns its dynamic handle.
    thread_handles: BTreeMap<u32, u64>,
    readable_events: BTreeMap<ReadableEventId, ReadableEvent>,
    trace: std::collections::VecDeque<IpcEvent>,
    trace_omitted: u64,
    private_parse_failures: Vec<PrivateIpcParseFailure>,
    session_close_requests: u64,
    parse_failures_omitted: u64,
    log_sink: LocalLogSink,
    main_application: Option<MainApplicationIdentity>,
    audio_device_output_event: Option<ReadableEventId>,
    audio_outs: Vec<AudioOutState>,
    /// Renderer sessions and their system events.
    audio_renderers: Vec<(crate::audio_renderer::Renderer, Option<ReadableEventId>)>,
    /// IAudioDevice output volume auto-tune setting (commands 19/20).
    audio_volume_auto_tune: bool,
    /// IAudioDevice output volumes by device name (f32 bits).
    audio_device_volumes: BTreeMap<String, u32>,
    /// Device functionalities the application listed virtual devices for.
    audio_device_list_requests: BTreeSet<u32>,
    /// bsd:u EventFd descriptors and their counter values.
    socket_event_fds: BTreeMap<i32, u64>,
    /// bsd:u Poll calls by requested timeout (the service never blocks).
    socket_poll_timeouts: BTreeMap<i32, u64>,
    /// Polls parked with nothing ready, by their internal event handle.
    pending_polls: BTreeMap<u32, PendingPoll>,
    /// Set by a request that parks its thread on the internal event handle
    /// (taken by the runtime after SendSyncRequest).
    blocked_request: Option<u32>,
    /// SendSyncRequest's own result for the last request, when not success.
    transport_result: Option<u32>,
    /// nfp availability-change event, created once.
    nfp_availability_event: Option<ReadableEventId>,
    /// Npad ids the application disconnected (hid DisconnectNpad).
    pub(crate) hid_disconnected_npads: BTreeSet<u32>,
    /// hid SetSupportedNpadStyleSet's style set.
    hid_supported_style_set: u32,
    /// ISelfController idle time detection extension (62/63).
    idle_time_detection_extension: u32,
    /// Syncpoint supervisor command 22 events by index.
    syncpoint_events: BTreeMap<u32, ReadableEventId>,
    /// Syncpoint event indices armed by 201 and not yet disarmed by 202.
    armed_syncpoint_events: BTreeSet<u32>,
    /// Decoder output fences: the value of each slot's syncpoint token (see
    /// `decoder_fence_token`), advanced when a frame is decoded into it.
    decoder_fences: BTreeMap<u32, u64>,
    /// Syncpoint events armed (201) on a decoder fence that has not reached
    /// its value yet: event index -> (token, value).
    decoder_fence_waits: BTreeMap<u32, (u32, u64)>,
    /// Set by Stop (207) and Finalize (115) until the next Start (6) or
    /// decoder (201): no frame will arrive, so every decoder fence counts
    /// as reached (the movie library waits on them while tearing down).
    decoder_fences_released: bool,
    /// Initialized video decoders: transfer memory size and parameters.
    video_decoders: Vec<(u64, Vec<u32>)>,
    /// Graphics supervisor native memories (111) by handle: kind, size.
    native_memories: BTreeMap<u32, (u8, u64)>,
    /// Application addresses bound to native memories (114).
    native_memory_addresses: BTreeMap<u32, u64>,
    /// Decoder image pools (911) by their first native image handle: the
    /// native memory handle and the number of images.
    decoder_image_pools: BTreeMap<u32, (u32, u32)>,
    /// The picture last decoded into each output slot (native image
    /// handle), handed out by GetFrame (213) and read through 961.
    slot_pictures: BTreeMap<u32, Box<crate::av1::Picture>>,
    /// Each 911 request in order: its preference, u32 and native memory
    /// (address and size), as hex for the report.
    decoder_image_pool_requests: Vec<String>,
    /// Whether the video decoder was started (6) and not stopped (207).
    video_decoder_running: bool,
    /// Frames handed back to the decoder (command 314), output slots
    /// waiting for a frame (311/314), and frames reported decoded (213).
    returned_frames: u64,
    decoder_slots: std::collections::VecDeque<u32>,
    decoded_frames: u64,
    /// Submissions refused while the decoder's input was full.
    refused_video_submissions: u64,
    /// Decoded frames waiting for GetFrame, with the slot each fills.
    decoded_video_frames: std::collections::VecDeque<(u32, VideoFrame)>,
    /// The decoder's input event (GetInputEvent), for back-pressure.
    video_input_event: Option<ReadableEventId>,
    /// GetFrame calls answered "no output" (213).
    empty_frame_requests: u64,
    /// Video decoder submissions so far and the queued input, by
    /// submission id.
    video_submissions: u64,
    video_input: Vec<(u64, VideoFrame)>,
    /// The host AV1 decoder of the current decoder session (created by
    /// 201, dropped by 115), and its failures.
    av1_decoder: Option<crate::av1::Decoder>,
    /// Pictures decoded and when the first was (ECHORA_MOVIE_RATE).
    av1_pictures: (u64, Option<std::time::Instant>, std::time::Instant),
    av1_errors: Vec<String>,
    /// The last 600 video decoder requests: command and leading argument
    /// bytes (hex); GetFrame (213) and syncpoint waits add what they did.
    video_log: std::collections::VecDeque<String>,
    /// Library applets and applet storages.
    library_applets: Vec<LibraryApplet>,
    applet_storages: Vec<Vec<u8>>,
    /// ISelfController GetLibraryAppletLaunchableEvent (9), always signaled.
    library_applet_launchable_event: Option<ReadableEventId>,
    terminate_result: Option<u32>,
    /// hid applet resource shared memory, once created.
    hid_memory: Option<u32>,
    /// stpl GetGuestHidSharedMemory's shared memory, once created.
    streamplay_hid_memory: Option<u32>,
    /// Last object id handed out by ISslContext imports.
    ssl_next_id: u64,
    /// nifm IRequest event pairs, created by GetSystemEventReadableHandles.
    nifm_requests: Vec<Option<(ReadableEventId, ReadableEventId)>>,
    // One record per modeled application, not per session or cloned endpoint.
    main_application_state: Option<MainApplicationState>,
}

impl Default for Services {
    fn default() -> Self {
        Self {
            next_handle: FIRST_SESSION_HANDLE,
            sessions: BTreeMap::new(),
            provisional_inputs: BTreeMap::new(),
            add_on_content: Vec::new(),
            error_contexts_completed: 0,
            error_context_bytes: 0,
            parental_control_grants: BTreeSet::new(),
            service_access: None,
            denied_services: BTreeSet::new(),
            graphics_supervisor_handle: None,
            graphics_supervisor_phase: None,
            nvdrv_firmware_memory_margin: None,
            nvdrv_fds: BTreeMap::new(),
            gpu_sharable: BTreeMap::new(),
            gpu_memory: BTreeMap::new(),
            gpu_mappings: BTreeMap::new(),
            nvdrv_devices: Default::default(),
            nvdrv_events: BTreeMap::new(),
            transfer_memories: BTreeMap::new(),
            retained_transfer_memories: Vec::new(),
            writable_event_handles: BTreeMap::new(),
            syncpoint_cache: None,
            syncpoint_event_count: None,
            graphics_contexts: BTreeSet::new(),
            gpu_session_version: None,
            gpu_events: BTreeMap::new(),
            graphics_devices: BTreeMap::new(),
            gpu_address_spaces: BTreeMap::new(),
            gpu_worker_ready: false,
            graphics_scale_changed_notification: None,
            exit_locked: false,
            session_aliases: BTreeMap::new(),
            fs_current_process: None,
            fs_access_log_mode: 0,
            data_romfs: None,
            file_systems: Vec::new(),
            open_files: Vec::new(),
            open_directories: Vec::new(),
            data_storage_bytes_read: 0,
            data_storage_reads: 0,
            data_storage_last_offset: 0,
            data_storage_chunks: Default::default(),
            cpu_boost_mode: None,
            syncpoint_context_initialized: false,
            steady_epoch: std::time::Instant::now(),
            steady_clock_source: random_clock_source(),
            local_user: LocalUser::from_environment(),
            account_application_info_initialized: false,
            preselected_user_popped: false,
            nacp_user_account_switch_lock: std::env::var_os("ECHORA_NACP").map(|path| {
                let bytes =
                    std::fs::read(&path).map_err(|error| format!("read ECHORA_NACP: {error}"))?;
                if bytes.len() != crate::content_metadata::NACP_SIZE {
                    return Err("ECHORA_NACP is not a 0x4000-byte control.nacp".into());
                }
                Ok(bytes[0x3026])
            }),
            shared_memories: BTreeMap::new(),
            shared_memory_handles: BTreeMap::new(),
            provisional_commands: BTreeSet::new(),
            readable_event_handles: BTreeMap::new(),
            thread_handles: BTreeMap::new(),
            readable_events: BTreeMap::new(),
            trace: Default::default(),
            trace_omitted: 0,
            private_parse_failures: Vec::new(),
            session_close_requests: 0,
            parse_failures_omitted: 0,
            log_sink: LocalLogSink::default(),
            main_application: None,
            audio_device_output_event: None,
            audio_outs: Vec::new(),
            audio_renderers: Vec::new(),
            audio_volume_auto_tune: false,
            audio_device_volumes: BTreeMap::new(),
            audio_device_list_requests: BTreeSet::new(),
            socket_event_fds: BTreeMap::new(),
            socket_poll_timeouts: BTreeMap::new(),
            pending_polls: BTreeMap::new(),
            blocked_request: None,
            transport_result: None,
            nfp_availability_event: None,
            hid_disconnected_npads: BTreeSet::new(),
            hid_supported_style_set: 0,
            idle_time_detection_extension: 0,
            syncpoint_events: BTreeMap::new(),
            armed_syncpoint_events: BTreeSet::new(),
            decoder_fences: BTreeMap::new(),
            decoder_fence_waits: BTreeMap::new(),
            decoder_fences_released: false,
            video_decoders: Vec::new(),
            native_memories: BTreeMap::new(),
            native_memory_addresses: BTreeMap::new(),
            decoder_image_pools: BTreeMap::new(),
            slot_pictures: BTreeMap::new(),
            decoder_image_pool_requests: Vec::new(),
            video_decoder_running: false,
            returned_frames: 0,
            decoder_slots: Default::default(),
            decoded_frames: 0,
            refused_video_submissions: 0,
            decoded_video_frames: Default::default(),
            video_input_event: None,
            empty_frame_requests: 0,
            video_submissions: 0,
            video_input: Vec::new(),
            av1_decoder: None,
            av1_pictures: (0, None, std::time::Instant::now()),
            av1_errors: Vec::new(),
            video_log: Default::default(),
            library_applets: Vec::new(),
            applet_storages: Vec::new(),
            library_applet_launchable_event: None,
            terminate_result: None,
            ssl_next_id: 0,
            streamplay_hid_memory: None,
            hid_memory: None,
            nifm_requests: Vec::new(),
            main_application_state: None,
        }
    }
}

/// An exclusive, mutation-free reservation in the shared dynamic handle table.
/// Runtime prepares the thread/TLS while this guard keeps all other handle
/// allocation unavailable. Dropping it consumes neither a handle nor capacity.
pub(crate) struct ReservedThreadHandle<'a> {
    services: &'a mut Services,
    handle: u32,
    thread_id: u64,
}

impl ReservedThreadHandle<'_> {
    pub(crate) fn handle(&self) -> u32 {
        self.handle
    }

    /// Preparation already checked capacity, uniqueness and the increment.
    /// The exclusive borrow prevents those conditions changing before commit.
    pub(crate) fn commit(self) -> u32 {
        self.services
            .thread_handles
            .insert(self.handle, self.thread_id);
        self.services.next_handle = self.handle + 1;
        self.handle
    }
}

fn unsupported(pc: u64, service: &str, command_id: Option<u32>, message: String) -> Trap {
    Trap::UnsupportedService {
        pc,
        service: service.into(),
        command_id,
        message,
    }
}

/// A kernel shared-memory object: page-multiple size, the permission granted
/// to this (non-owner) process, and its initial contents.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SharedMemoryObject {
    pub(crate) size: u64,
    pub(crate) permission: u8,
    pub(crate) contents: Vec<u8>,
    pub(crate) label: &'static str,
}

const TIME_SHARED_MEMORY_ID: u32 = 1;

/// Per-launch steady-clock source ID from host randomness (zero on failure,
/// which is still a single consistent ID for the session).
fn random_clock_source() -> [u8; 16] {
    let mut id = [0u8; 16];
    let _ = getrandom::fill(&mut id);
    id
}

fn invalid(message: impl Into<String>) -> Trap {
    Trap::InvalidState {
        message: message.into(),
    }
}

/// The one local user of this native-PC port. The Uid is a fixed, nonzero
/// local identifier (not a console account); the nickname comes from
/// `ECHORA_USER_NAME` (UTF-8, at most 32 bytes) and defaults to "Player".
#[derive(Clone, Debug)]
struct LocalUser {
    uid: [u8; 16],
    nickname: String,
}

impl LocalUser {
    const UID: [u8; 16] = *b"echora-runtime\x00\x01";

    fn from_environment() -> Self {
        let nickname = std::env::var("ECHORA_USER_NAME")
            .ok()
            .filter(|name| !name.is_empty() && name.len() <= 32 && !name.contains('\0'))
            .unwrap_or_else(|| "Player".into());
        Self {
            uid: Self::UID,
            nickname,
        }
    }

    /// ProfileBase (0x38): Uid, u64 last-edit POSIX time (0: never edited),
    /// 0x20-byte UTF-8 nickname, zero padded.
    fn profile_base(&self, uid: &[u8; 16]) -> Vec<u8> {
        let mut base = vec![0u8; 0x38];
        base[..16].copy_from_slice(uid);
        let name = self.nickname.as_bytes();
        base[0x18..0x18 + name.len()].copy_from_slice(name);
        base
    }

    /// UserData (0x80): icon ID 0 (Mii) with every other field zero; no Mii
    /// or character icon is configured.
    fn user_data(&self) -> Vec<u8> {
        vec![0u8; 0x80]
    }
}

fn service_name(arguments: &[u8]) -> Result<String, Trap> {
    if arguments.len() != 8 {
        return Err(invalid(
            "SM GetServiceHandle requires exactly eight argument bytes",
        ));
    }
    let length = arguments.iter().position(|byte| *byte == 0).unwrap_or(8);
    if length == 0
        || !arguments[..length]
            .iter()
            .all(|byte| (b' '..=b'~').contains(byte))
        || arguments[length..].iter().any(|byte| *byte != 0)
    {
        return Err(invalid(
            "SM service name is not a nonempty zero-padded ASCII name",
        ));
    }
    Ok(String::from_utf8(arguments[..length].to_vec()).expect("checked to be ASCII above"))
}

struct Call<'a> {
    memory: &'a mut Memory,
    buffer_address: u64,
    handle: u32,
    process_id: u64,
    pc: u64,
    request: &'a Request,
    original_bytes: &'a [u8],
}

impl Call<'_> {
    fn shape(&self, arguments: usize, pid: bool, input_buffer: bool) -> Result<(), Trap> {
        self.shape_with_copies(arguments, pid, input_buffer, 0)
    }

    fn shape_with_copies(
        &self,
        arguments: usize,
        pid: bool,
        input_buffer: bool,
        copies: usize,
    ) -> Result<(), Trap> {
        if self.request.arguments.len() != arguments
            || self.request.pid.is_some() != pid
            || self.request.input_buffer.is_some() != input_buffer
            || self.request.copy_handles.len() != copies
            || !self.request.receive_buffers.is_empty()
            || !self.request.receive_list.is_empty()
            || self
                .request
                .domain
                .as_ref()
                .is_some_and(|domain| !domain.input_objects.is_empty())
        {
            return Err(invalid(format!(
                "IPC command {} expects {arguments} argument bytes, PID={pid}, input buffer={input_buffer}, {copies} copied handles, and no input domain objects; got {} argument bytes {:02x?}, PID={}, input buffer={}, {} copied handles, {} receive buffers, {} receive-list entries",
                self.request.command_id,
                self.request.arguments.len(),
                &self.request.arguments[..self.request.arguments.len().min(32)],
                self.request.pid.is_some(),
                self.request.input_buffer.is_some(),
                self.request.copy_handles.len(),
                self.request.receive_buffers.len(),
                self.request.receive_list.len(),
            )));
        }
        Ok(())
    }

    /// Shape of a command with one AutoSelect output buffer (a B descriptor,
    /// optionally paired with a C receive-list entry) and no other buffers.
    fn shape_output(&self, arguments: usize, pid: bool) -> Result<(), Trap> {
        if self.request.arguments.len() != arguments
            || self.request.pid.is_some() != pid
            || self.request.input_buffer.is_some()
            || !self.request.copy_handles.is_empty()
            || self.request.receive_buffers.len() != 1
            || self.request.receive_list.len() > 1
            || self
                .request
                .domain
                .as_ref()
                .is_some_and(|domain| !domain.input_objects.is_empty())
        {
            return Err(invalid(format!(
                "IPC command {} expects {arguments} argument bytes, PID={pid} and one output buffer",
                self.request.command_id
            )));
        }
        Ok(())
    }

    /// The selected AutoSelect output buffer: the B descriptor when it has a
    /// size, otherwise the receive-list entry (this server has no pointer
    /// buffer, so clients normally use B).
    /// Auto-select output `index`: its B descriptor, or the C receive-list
    /// entry of the same index when the B descriptor is empty.
    fn output_buffer_at(&self, index: usize) -> Result<(u64, u64), Trap> {
        let &(address, size, mode) = self
            .request
            .receive_buffers
            .get(index)
            .ok_or_else(|| invalid("missing output buffer descriptor"))?;
        if mode == 2 {
            return Err(invalid("invalid B descriptor mode 2"));
        }
        if size != 0 {
            return Ok((address, size));
        }
        match self.request.receive_list.get(index) {
            Some(&(address, size)) if size != 0 => Ok((address, size)),
            _ => Ok((address, 0)),
        }
    }

    fn output_buffer(&self) -> Result<(u64, u64), Trap> {
        // A request with only a C (receive-list) buffer uses that.
        let Some(&(address, size, mode)) = self.request.receive_buffers.first() else {
            return match self.request.receive_list.first() {
                Some(&(address, size)) => Ok((address, size)),
                None => Err(invalid("missing output buffer descriptor")),
            };
        };
        // Modes 0 (normal), 1 (non-secure) and 3 (non-device) only differ in
        // which memory states the kernel may map; this server writes the
        // client's memory directly. Mode 2 is invalid.
        if mode == 2 {
            return Err(invalid("invalid B descriptor mode 2"));
        }
        if size != 0 {
            return Ok((address, size));
        }
        match self.request.receive_list.first() {
            Some(&(address, size)) if size != 0 => Ok((address, size)),
            _ => Ok((address, 0)),
        }
    }

    fn prepare_reply(
        &self,
        result: u32,
        handle: Option<u32>,
        interface_id: u32,
        data: &[u8],
        output_objects: &[u32],
    ) -> Result<ipc::Reply, Trap> {
        let reply = if self.request.domain.is_some() {
            if handle.is_some() {
                return Err(invalid(
                    "domain object replies cannot substitute kernel handles",
                ));
            }
            ipc::build_domain_reply_with_data(
                self.request,
                result,
                interface_id,
                data,
                output_objects,
            )
        } else {
            if !output_objects.is_empty() {
                return Err(invalid(
                    "non-domain replies cannot contain domain object IDs",
                ));
            }
            ipc::build_reply_with_data(self.request, result, handle, interface_id, data)
        };
        reply.map_err(|error| invalid(format!("build service reply: {error:#}")))
    }

    fn write_reply(&mut self, reply: &ipc::Reply) -> Result<(), Trap> {
        self.memory.write(self.buffer_address, &reply.bytes)
    }

    fn prepare_copy_handle_reply(
        &self,
        handle: u32,
        interface_id: u32,
    ) -> Result<ipc::Reply, Trap> {
        self.prepare_copy_handle_reply_with_data(handle, interface_id, &[])
    }

    fn prepare_copy_handle_reply_with_data(
        &self,
        handle: u32,
        interface_id: u32,
        data: &[u8],
    ) -> Result<ipc::Reply, Trap> {
        // A readable-event alias remains a kernel handle inside a domain. It
        // is never an output domain object or a moved service-session handle.
        let reply = if self.request.domain.is_some() {
            ipc::build_domain_copy_handle_reply_with_data(
                self.request,
                0,
                handle,
                interface_id,
                data,
                &[],
            )
        } else {
            ipc::build_copy_handle_reply_with_data(self.request, 0, handle, interface_id, data)
        };
        reply.map_err(|error| invalid(format!("build copied-event reply: {error:#}")))
    }

    fn reply(
        &mut self,
        result: u32,
        handle: Option<u32>,
        interface_id: u32,
        data: &[u8],
    ) -> Result<(), Trap> {
        let reply = self.prepare_reply(result, handle, interface_id, data, &[])?;
        self.write_reply(&reply)
    }

    fn unsupported(&self, service: &str, message: impl Into<String>) -> Trap {
        unsupported(
            self.pc,
            service,
            Some(self.request.command_id),
            message.into(),
        )
    }
}

/// A bsd:u Poll parked until a descriptor is ready: the calling thread's
/// message buffer, its prepared reply (the ready count goes at
/// `data_offset`), the pollfds and their output buffer.
#[derive(Clone, Debug)]
struct PendingPoll {
    tls: u64,
    reply: Vec<u8>,
    data_offset: usize,
    entries: Vec<u8>,
    output: Option<(u64, u64)>,
    event: ReadableEventId,
}

/// Fill each pollfd's revents for the EventFds: writable (POLLOUT 4) always,
/// readable (POLLIN 1) with a nonzero counter, POLLNVAL (0x20) for unknown
/// descriptors. Returns the number with any revents.
fn poll_event_fds(event_fds: &BTreeMap<i32, u64>, entries: &mut [u8]) -> i32 {
    let mut ready = 0;
    for entry in entries.chunks_mut(8) {
        let fd = crate::bytes::le_u32(&entry[..4]) as i32;
        let events = crate::bytes::le_u16(&entry[4..6]);
        let revents = match event_fds.get(&fd) {
            Some(&counter) => events & (u16::from(counter != 0) | 4),
            None => 0x20,
        };
        entry[6..8].copy_from_slice(&revents.to_le_bytes());
        ready += i32::from(revents != 0);
    }
    ready
}

/// Submissions the host video decoder holds before its input event clears.
const VIDEO_INPUT_DEPTH: usize = 8;

/// A video decoder submission, and the frame it becomes.
#[derive(Debug, Default)]
struct VideoFrame {
    /// The submission's presentation time (microseconds).
    timestamp: u64,
    /// The stream's last submission: its frame reports the end of the
    /// stream (the SDK's output property `last`).
    last: bool,
    /// The picture the submission decoded to.
    picture: Option<Box<crate::av1::Picture>>,
}

/// The syncpoint token of a decoder output slot's fence: a range no
/// syncpoint the SDK opens reaches (its tokens are small inverted ids).
fn decoder_fence_token(slot: u32) -> u32 {
    0x5644_0000 | (slot & 0xffff)
}

/// AppletId of the controller applet (yuzu applet AppletId::Controller).
const LIBRARY_APPLET_CONTROLLER: u32 = 0x0c;

/// A library applet created through ILibraryAppletCreator: its id and
/// mode, the storages pushed in and waiting to be popped, and its events.
#[derive(Clone, Debug, Default, Serialize)]
struct LibraryApplet {
    applet_id: u32,
    mode: u32,
    in_data: Vec<usize>,
    out_data: std::collections::VecDeque<usize>,
    started: bool,
    completed: bool,
    #[serde(skip)]
    state_event: Option<ReadableEventId>,
    #[serde(skip)]
    pop_out_event: Option<ReadableEventId>,
    #[serde(skip)]
    event_170: Option<ReadableEventId>,
}

struct Outcome {
    result: Option<u32>,
    log_disposition: Option<String>,
    created_domain_objects: Vec<u32>,
    output_copy_handle_count: usize,
}

impl Outcome {
    fn result(result: u32) -> Self {
        Self {
            result: Some(result),
            log_disposition: None,
            created_domain_objects: Vec::new(),
            output_copy_handle_count: 0,
        }
    }
    fn success() -> Self {
        Self::result(0)
    }
}

impl Services {
    /// Bind the one main application before any dynamic handle is allocated.
    /// This consumes only trusted loader/runtime metadata, never IPC PID bytes.
    /// Enforce the loaded NPDM's ACI0 service access control in sm: GetService.
    pub(crate) fn set_service_access(&mut self, names: Vec<String>) {
        self.service_access = Some(names);
    }

    fn service_allowed(&self, name: &str) -> bool {
        self.service_access.as_ref().is_none_or(|names| {
            names.iter().any(|allowed| match allowed.strip_suffix('*') {
                Some(prefix) => name.starts_with(prefix),
                None => allowed == name,
            })
        })
    }

    pub(crate) fn initialize_main_application(
        &mut self,
        process_id: u64,
        program_id: u64,
        npdm_sha256: &str,
    ) -> Result<(), Trap> {
        if self.main_application.is_some()
            || self.main_application_state.is_some()
            || !self.sessions.is_empty()
            || !self.readable_event_handles.is_empty()
            || !self.thread_handles.is_empty()
            || !self.readable_events.is_empty()
            || self.next_handle != FIRST_SESSION_HANDLE
        {
            return Err(invalid(
                "main application identity must be initialized once before dynamic handles exist",
            ));
        }
        if npdm_sha256.len() != 64 || !npdm_sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(invalid(
                "main application NPDM provenance must be a 64-digit ASCII SHA256",
            ));
        }
        let application = MainApplicationIdentity {
            process_id,
            program_id,
            npdm_sha256: npdm_sha256.into(),
        };
        // Eden 1a6d7d4's Applet owns these distinct events before controller
        // getters run. Local backing IDs are not console ABI values; registering
        // them consumes neither guest handles nor domain object IDs. The GPU
        // event has no producer here because GPU workloads/detection are absent;
        // its clear state is not a claim about the host or emulated GPU's health.
        let events = BTreeMap::from([
            (
                SUSPENDED_TICK_EVENT_ID,
                ReadableEvent {
                    owner: application.clone(),
                    kind: ReadableEventKind::AccumulatedSuspendedTickChanged,
                    signaled: false,
                },
            ),
            (
                GPU_ERROR_EVENT_ID,
                ReadableEvent {
                    owner: application.clone(),
                    kind: ReadableEventKind::GpuErrorDetected,
                    signaled: false,
                },
            ),
            (
                LIFECYCLE_EVENT_ID,
                ReadableEvent {
                    owner: application.clone(),
                    kind: ReadableEventKind::LifecycleSystem,
                    // AppletManager::SetWindowSystem sets InFocus before Run.
                    // Initial pending focus makes availability true without
                    // adding an unordered message; the getter never signals it.
                    signaled: true,
                },
            ),
            (
                ADD_ON_CONTENT_EVENT_ID,
                ReadableEvent {
                    owner: application.clone(),
                    kind: ReadableEventKind::AddOnContentListChanged,
                    signaled: false,
                },
            ),
            (
                APPLICATION_UNKNOWN_210_EVENT_ID,
                ReadableEvent {
                    owner: application.clone(),
                    kind: ReadableEventKind::ApplicationUnknown210,
                    signaled: false,
                },
            ),
            (
                FRIEND_INVITATION_EVENT_ID,
                ReadableEvent {
                    owner: application.clone(),
                    kind: ReadableEventKind::FriendInvitationStorageChannel,
                    signaled: false,
                },
            ),
        ]);
        let state = MainApplicationState {
            launch_policy: ApplicationLaunchPolicy::REFERENCE_APPLICATION,
            display_layer_manager: None,
            lifecycle: LogicalApplicationLifecycle::reference_foreground(),
            lifecycle_event: LIFECYCLE_EVENT_ID,
            request_exit_to_library_applet_at_execute_next_program_enabled: false,
            suspended_ticks: 0,
            accumulated_suspended_tick_changed_event: SUSPENDED_TICK_EVENT_ID,
            gpu_error_detected_event: GPU_ERROR_EVENT_ID,
            add_on_content_list_changed_event: ADD_ON_CONTENT_EVENT_ID,
            application_unknown_210_event: APPLICATION_UNKNOWN_210_EVENT_ID,
            friend_invitation_storage_channel_event: FRIEND_INVITATION_EVENT_ID,
            foreground_rights_acquired: false,
            self_settings: SelfControllerSettings::default(),
            running_notified: false,
        };
        // Complete all construction before publishing the trusted binding.
        self.main_application = Some(application);
        self.main_application_state = Some(state);
        self.readable_events = events;
        Ok(())
    }

    fn record_parse_failure(
        &mut self,
        pc: u64,
        handle: u32,
        session: &Session,
        tls_address: u64,
        reason: &str,
        bytes: &[u8; TLS_MESSAGE_BUFFER_SIZE],
    ) {
        if self.private_parse_failures.len() >= MAX_PRIVATE_PARSE_FAILURES {
            self.parse_failures_omitted = self.parse_failures_omitted.saturating_add(1);
            return;
        }
        let mut reason_end = reason.len().min(MAX_PARSE_FAILURE_REASON_BYTES);
        while !reason.is_char_boundary(reason_end) {
            reason_end -= 1;
        }
        use std::fmt::Write;
        let mut tls_snapshot_hex = String::with_capacity(TLS_MESSAGE_BUFFER_SIZE * 2);
        for byte in bytes {
            write!(tls_snapshot_hex, "{byte:02x}").expect("writing to a String cannot fail");
        }
        self.private_parse_failures.push(PrivateIpcParseFailure {
            pc,
            handle,
            service: session.service().into(),
            tls_address,
            domain_session: session.is_domain(),
            reason: reason[..reason_end].into(),
            reason_truncated: reason_end != reason.len(),
            tls_snapshot_hex,
        });
    }

    fn next_available_handle(&self) -> Result<u32, Trap> {
        if self
            .sessions
            .len()
            .saturating_add(self.readable_event_handles.len())
            .saturating_add(self.thread_handles.len())
            .saturating_add(self.shared_memory_handles.len())
            .saturating_add(self.transfer_memories.len())
            .saturating_add(self.writable_event_handles.len())
            .saturating_add(self.session_aliases.len())
            >= MAX_LIVE_HANDLES
        {
            return Err(invalid(
                "prototype combined session/event/thread handle capacity exhausted",
            ));
        }
        if !(FIRST_SESSION_HANDLE..0x7fff_ffff).contains(&self.next_handle) {
            return Err(invalid("dynamic handle range exhausted"));
        }
        if self.sessions.contains_key(&self.next_handle)
            || self.session_aliases.contains_key(&self.next_handle)
            || self.readable_event_handles.contains_key(&self.next_handle)
            || self.thread_handles.contains_key(&self.next_handle)
            || self.shared_memory_handles.contains_key(&self.next_handle)
            || self.transfer_memories.contains_key(&self.next_handle)
            || self.writable_event_handles.contains_key(&self.next_handle)
        {
            return Err(invalid("dynamic allocation would replace a live handle"));
        }
        Ok(self.next_handle)
    }

    /// Reserve a typed handle without publishing it or advancing the allocator.
    /// Thread IDs are logical identities, not raw handles or pseudo-handles.
    /// Publish a transfer memory whose range the runtime has already locked.
    pub(crate) fn insert_transfer_memory(&mut self, memory: TransferMemory) -> Result<u32, Trap> {
        let handle = self.next_available_handle()?;
        self.transfer_memories.insert(handle, memory);
        self.next_handle = handle + 1;
        Ok(handle)
    }

    pub(crate) fn transfer_memory(&self, handle: u32) -> Option<TransferMemory> {
        self.transfer_memories.get(&handle).copied()
    }

    /// Remove a transfer-memory handle. Returns the memory to unlock, or
    /// `None` when a server object still references it (it stays locked).
    pub(crate) fn remove_transfer_memory(&mut self, handle: u32) -> Option<TransferMemory> {
        let memory = self.transfer_memories.remove(&handle)?;
        let referenced = self
            .syncpoint_cache
            .as_ref()
            .is_some_and(|cache| cache.rw_handle == handle || cache.ro_handle == handle);
        if referenced {
            self.retained_transfer_memories.push(memory);
            None
        } else {
            Some(memory)
        }
    }

    pub(crate) fn reserve_thread_handle(
        &mut self,
        thread_id: u64,
    ) -> Result<ReservedThreadHandle<'_>, Trap> {
        if thread_id == 0 || self.thread_handles.values().any(|id| *id == thread_id) {
            return Err(invalid("thread handle requires a nonzero unique thread ID"));
        }
        let handle = self.next_available_handle()?;
        Ok(ReservedThreadHandle {
            services: self,
            handle,
            thread_id,
        })
    }

    /// Signal state of the backing event behind a live readable-event alias.
    /// `None` means the handle is not a readable-event alias.
    pub(crate) fn readable_event_signaled(&self, handle: u32) -> Result<Option<bool>, Trap> {
        let Some(id) = self.readable_event_handles.get(&handle) else {
            return Ok(None);
        };
        // An armed syncpoint event stays signaled until disarmed: its
        // target value counts as reached (host GPU work completes at
        // submission), even after the client clears the event.
        self.readable_events
            .get(id)
            .map(|event| {
                let armed = matches!(event.kind, ReadableEventKind::SyncpointEvent { index }
                    if self.armed_syncpoint_events.contains(&index));
                Some(event.signaled || armed)
            })
            .ok_or_else(|| invalid("readable-event alias has no backing event"))
    }

    /// The readable-event kind behind a handle, for diagnostics.
    pub(crate) fn readable_event_kind(&self, handle: u32) -> Option<String> {
        let id = self.readable_event_handles.get(&handle)?;
        self.readable_events
            .get(id)
            .map(|event| format!("{:?}", event.kind))
    }

    pub(crate) fn thread_id(&self, handle: u32) -> Option<u64> {
        self.thread_handles.get(&handle).copied()
    }

    /// Runtime must first destroy the dormant thread and release its owned TLS.
    pub(crate) fn release_thread_handle(&mut self, handle: u32) -> Option<u64> {
        self.thread_handles.remove(&handle)
    }

    pub fn connect_named_port(&mut self, name: &str, pc: u64) -> Result<u32, Trap> {
        if name != "sm:" {
            return Err(unsupported(
                pc,
                name,
                None,
                "named port has no implemented backend".into(),
            ));
        }
        let handle = self.next_available_handle()?;
        self.sessions.insert(
            handle,
            Session::regular(ServiceObject::ServiceManager {
                registered_process_id: None,
            }),
        );
        self.next_handle += 1;
        Ok(handle)
    }

    /// Close one session or readable alias; the application-owned event remains.
    /// Pseudo-handles belong to the SVC layer.
    pub fn close(&mut self, handle: u32) -> bool {
        if self.session_aliases.remove(&handle).is_some() {
            return true;
        }
        if let Some(session) = self.sessions.remove(&handle) {
            // The server session lives while any cloned client handle does:
            // the lowest alias becomes its primary handle.
            let aliases: Vec<u32> = self
                .session_aliases
                .iter()
                .filter(|(_, primary)| **primary == handle)
                .map(|(alias, _)| *alias)
                .collect();
            if let Some(&successor) = aliases.first() {
                self.sessions.insert(successor, session);
                for alias in aliases {
                    if alias == successor {
                        self.session_aliases.remove(&alias);
                    } else {
                        self.session_aliases.insert(alias, successor);
                    }
                }
            }
            return true;
        }
        self.readable_event_handles.remove(&handle).is_some()
            || self.shared_memory_handles.remove(&handle).is_some()
            || self.writable_event_handles.remove(&handle).is_some()
    }

    /// SignalEvent: only the writable side may signal. `None` is not a
    /// writable-event handle.
    pub(crate) fn signal_event(&mut self, handle: u32) -> Option<()> {
        let id = *self.writable_event_handles.get(&handle)?;
        self.readable_events.get_mut(&id)?.signaled = true;
        Some(())
    }

    /// ClearEvent accepts either side of an event.
    pub(crate) fn clear_event(&mut self, handle: u32) -> Option<()> {
        let id = *self
            .writable_event_handles
            .get(&handle)
            .or_else(|| self.readable_event_handles.get(&handle))?;
        self.readable_events.get_mut(&id)?.signaled = false;
        Some(())
    }

    /// ResetSignal (readable side only): `Some(false)` when not signaled.
    pub(crate) fn reset_signal(&mut self, handle: u32) -> Option<bool> {
        let id = *self.readable_event_handles.get(&handle)?;
        let event = self.readable_events.get_mut(&id)?;
        let was = event.signaled;
        event.signaled = false;
        Some(was)
    }

    /// Serve pending syncpoint-cache requests: for each requested slot not
    /// yet acknowledged, publish the counter value (no syncpoint is
    /// allocated by this model yet, so every value reads as unallocated),
    /// then acknowledge, clear the request event and signal completion.
    /// Returns whether any request was served.
    pub(crate) fn refresh_syncpoint_cache(&mut self, memory: &mut Memory) -> Result<bool, Trap> {
        let Some(cache) = self.syncpoint_cache.as_mut() else {
            return Ok(false);
        };
        let rw = cache.rw.address;
        let ro = cache.ro.address;
        let requests = memory.read_u64(rw + SYNCPOINT_CACHE_GROUP_REQUEST)?;
        let acknowledged = memory.read_u64(ro + SYNCPOINT_CACHE_GROUP_ACK)?;
        let group_mask = (1u64 << SYNCPOINT_CACHE_GROUPS) - 1;
        let mut groups = (requests ^ acknowledged) & group_mask;
        if groups == 0 {
            return Ok(false);
        }
        while groups != 0 {
            let group = groups.trailing_zeros() as u64;
            groups &= groups - 1;
            let requested = memory.read_u64(rw + SYNCPOINT_CACHE_SLOT_REQUEST + group * 8)?;
            let mut slots =
                requested ^ memory.read_u64(ro + SYNCPOINT_CACHE_SLOT_ACK + group * 8)?;
            while slots != 0 {
                let bit = slots.trailing_zeros() as u64;
                slots &= slots - 1;
                let id = group * 64 + bit;
                if id >= SYNCPOINT_CACHE_COUNTERS {
                    continue;
                }
                memory.initialize(
                    ro + SYNCPOINT_CACHE_VALUES + id * 8,
                    &SYNCPOINT_UNALLOCATED.to_le_bytes(),
                )?;
            }
            memory.initialize(
                ro + SYNCPOINT_CACHE_SLOT_ACK + group * 8,
                &requested.to_le_bytes(),
            )?;
        }
        memory.initialize(ro + SYNCPOINT_CACHE_GROUP_ACK, &requests.to_le_bytes())?;
        cache.refreshes = cache.refreshes.saturating_add(1);
        let (request, completed) = (cache.request_event, cache.completed_event);
        if let Some(event) = self.readable_events.get_mut(&request) {
            event.signaled = false;
        }
        if let Some(event) = self.readable_events.get_mut(&completed) {
            event.signaled = true;
        }
        Ok(true)
    }
    /// IPC type predicate: readable event and thread handles are excluded.
    pub fn contains(&self, handle: u32) -> bool {
        self.sessions.contains_key(&handle) || self.session_aliases.contains_key(&handle)
    }

    pub fn send_sync_request(
        &mut self,
        memory: &mut Memory,
        buffer_address: u64,
        handle: u32,
        process_id: u64,
        pc: u64,
    ) -> Result<(), Trap> {
        // A cloned client handle reaches the same server session.
        let handle = self.session_aliases.get(&handle).copied().unwrap_or(handle);
        let session = self
            .sessions
            .get(&handle)
            .ok_or_else(|| {
                invalid(format!(
                    "IPC handle {handle:#x} is not a live service session"
                ))
            })?
            .clone();
        let mut bytes = [0; TLS_MESSAGE_BUFFER_SIZE];
        memory.read(buffer_address, &mut bytes)?;
        // A bare CMIF Close (HIPC tag 2, no data or descriptors) on a
        // non-domain session announces that the client is about to close its
        // handle (libnx serviceClose: send, ignore the result, CloseHandle).
        // The session itself is released by that CloseHandle; the request
        // succeeds without a reply. A TIPC Close (type 15, also bare) does
        // the same for TIPC sessions.
        // For TIPC the server closes its side, so the request itself fails
        // with ResultSessionClosed (0xf601), which the SDK requires before
        // closing its handle (sdk+0x1a5794).
        let tipc_close = bytes[..8] == [15, 0, 0, 0, 0, 0, 0, 0];
        if !session.is_domain() && (bytes[..8] == [2, 0, 0, 0, 0, 0, 0, 0] || tipc_close) {
            self.session_close_requests += 1;
            if tipc_close {
                self.transport_result = Some(0xf601);
            }
            return Ok(());
        }
        let request = match ipc::parse_request_with_domain(&bytes, session.is_domain()) {
            Ok(request) => request,
            Err(error) => {
                let reason = format!("{error:#}");
                // No extra memory is read and no request is dispatched. The
                // bounded diagnostic changes neither objects nor guest bytes.
                self.record_parse_failure(pc, handle, &session, buffer_address, &reason, &bytes);
                return Err(invalid(format!(
                    "unsupported or malformed {} IPC; bounded details are in private_parse_failures",
                    session.service()
                )));
            }
        };
        let object_id = request.domain.as_ref().map(|domain| domain.object_id);
        let service = session
            .object(object_id)
            .map_or_else(|| session.service(), ServiceObject::service);
        let requested_service = if service == "sm:"
            && request.kind == RequestKind::Request
            && request.command_id == 1
        {
            service_name(&request.arguments).ok()
        } else {
            None
        };
        // The trace keeps the most recent MAX_TRACE requests; older ones are
        // dropped and counted in `trace_omitted`.
        let event_index = {
            if self.trace.len() >= MAX_TRACE {
                self.trace.pop_front();
                self.trace_omitted += 1;
            }
            let index = self.trace.len();
            self.trace.push_back(IpcEvent {
                pc,
                handle,
                service: service.into(),
                protocol: format!("{:?}", request.protocol),
                request_kind: format!("{:?}", request.kind),
                domain_object_id: object_id,
                input_domain_objects: request
                    .domain
                    .as_ref()
                    .map_or_else(Vec::new, |domain| domain.input_objects.clone()),
                created_domain_objects: Vec::new(),
                command_id: request.command_id,
                has_pid_descriptor: request.pid.is_some(),
                input_copy_handle_count: request.copy_handles.len(),
                output_copy_handle_count: 0,
                argument_bytes: request.arguments.len(),
                input_buffer: request.input_buffer.clone(),
                requested_service,
                completed: false,
                service_result: None,
                log_disposition: None,
            });
            Some(index)
        };
        let mut call = Call {
            memory,
            buffer_address,
            handle,
            process_id,
            pc,
            request: &request,
            original_bytes: &bytes,
        };
        let outcome = self.dispatch(&session, &mut call)?;
        if let Some(index) = event_index {
            self.trace[index].completed = true;
            self.trace[index].service_result = outcome.result;
            self.trace[index].log_disposition = outcome.log_disposition;
            self.trace[index].created_domain_objects = outcome.created_domain_objects;
            self.trace[index].output_copy_handle_count = outcome.output_copy_handle_count;
        }
        Ok(())
    }

    /// Reply with a copy of a new, signaled completion event for an async
    /// context that finished when it was created.
    fn reply_done_event(&mut self, call: &mut Call<'_>) -> Result<Outcome, Trap> {
        let application = self
            .main_application
            .clone()
            .ok_or_else(|| invalid("async context requires the bootstrap main application"))?;
        if self.readable_events.len() >= MAX_READABLE_EVENTS {
            return Err(invalid("readable event backing capacity exhausted"));
        }
        let id = ReadableEventId(
            self.readable_events
                .keys()
                .map(|id| id.0)
                .max()
                .unwrap_or(0)
                + 1,
        );
        let handle = self.next_available_handle()?;
        let reply = call.prepare_copy_handle_reply(handle, AOC_INTERFACE_ID)?;
        call.write_reply(&reply)?;
        self.readable_events.insert(
            id,
            ReadableEvent {
                owner: application,
                kind: ReadableEventKind::AsyncContextDone,
                signaled: true,
            },
        );
        self.readable_event_handles.insert(handle, id);
        self.next_handle = handle + 1;
        Ok(Outcome {
            output_copy_handle_count: 1,
            ..Outcome::success()
        })
    }

    fn reply_new_object(
        &mut self,
        call: &mut Call<'_>,
        object: ServiceObject,
        interface_id: u32,
    ) -> Result<Outcome, Trap> {
        self.reply_new_object_with(call, object, interface_id, &[])
    }

    /// As [`Self::reply_new_object`], with raw output data before the object.
    fn reply_new_object_with(
        &mut self,
        call: &mut Call<'_>,
        object: ServiceObject,
        interface_id: u32,
        data: &[u8],
    ) -> Result<Outcome, Trap> {
        if call.request.domain.is_some() {
            let mut updated = self.sessions[&call.handle].clone();
            let Session::Domain {
                next_object_id,
                objects,
                ..
            } = &mut updated
            else {
                return Err(invalid(
                    "domain object allocation requires a domain session",
                ));
            };
            if objects.len() >= MAX_DOMAIN_OBJECTS {
                return Err(invalid("prototype domain object capacity exhausted"));
            }
            let id = *next_object_id;
            let next = id
                .checked_add(1)
                .filter(|_| id != 0)
                .ok_or_else(|| invalid("prototype domain object ID range exhausted"))?;
            if objects.contains_key(&id) {
                return Err(invalid(
                    "prototype domain object ID would replace a live object",
                ));
            }
            // Prepare the whole new table before publishing its ID. Object IDs
            // belong to this session's domain and are never kernel handles.
            objects.insert(id, object);
            *next_object_id = next;
            let reply = call.prepare_reply(0, None, interface_id, data, &[id])?;
            call.write_reply(&reply)?;
            self.sessions.insert(call.handle, updated);
            return Ok(Outcome {
                result: Some(0),
                log_disposition: None,
                created_domain_objects: vec![id],
                output_copy_handle_count: 0,
            });
        }
        let handle = self.next_available_handle()?;
        // The reply must be writable before the new handle becomes live.
        call.reply(0, Some(handle), interface_id, data)?;
        self.sessions.insert(handle, Session::regular(object));
        self.next_handle += 1;
        Ok(Outcome::success())
    }

    fn dispatch(&mut self, session: &Session, call: &mut Call<'_>) -> Result<Outcome, Trap> {
        if call.request.kind == RequestKind::Control {
            if session.service() == "sm:" {
                return Err(call.unsupported("sm:", "SM CMIF controls are not implemented"));
            }
            if call.request.command_id == 0 {
                call.shape(0, false, false)?;
                let Session::Regular { object } = session else {
                    // Atmosphere can wrap a domain in another domain. This
                    // prototype has no nested-domain holder/dispatch model.
                    return Err(call.unsupported(
                        session.service(),
                        "repeated domain conversion requires unsupported nested domains",
                    ));
                };
                let converted = Session::Domain {
                    original_service: object.service(),
                    next_object_id: 2,
                    objects: BTreeMap::from([(1, object.clone())]),
                };
                call.reply(0, None, HIPC_MANAGER_INTERFACE_ID, &1u32.to_le_bytes())?;
                self.sessions.insert(call.handle, converted);
                return Ok(Outcome {
                    result: Some(0),
                    log_disposition: None,
                    created_domain_objects: vec![1],
                    output_copy_handle_count: 0,
                });
            }
            if call.request.command_id == 3 {
                call.shape(0, false, false)?;
                // Every session reports yuzu's 0x8000-byte pointer buffer:
                // hwopus, hid and other clients abort with sf::hipc
                // PointerBufferTooSmall (11-141) when it is smaller than an
                // X or C buffer they send. This server reads and writes
                // client memory directly, so the capacity only admits X/C
                // descriptors.
                let capacity: u16 = 0x8000;
                call.reply(0, None, HIPC_MANAGER_INTERFACE_ID, &capacity.to_le_bytes())?;
                return Ok(Outcome::success());
            }
            // CloneCurrentObject (2) / CloneCurrentObjectEx (4, u32 tag):
            // another client handle to the same server session (regular or
            // domain), so state and domain object IDs are shared.
            if matches!(call.request.command_id, 2 | 4) {
                let arguments = if call.request.command_id == 4 { 4 } else { 0 };
                call.shape(arguments, false, false)?;
                let handle = self.next_available_handle()?;
                call.reply(0, Some(handle), HIPC_MANAGER_INTERFACE_ID, &[])?;
                self.session_aliases.insert(handle, call.handle);
                self.next_handle = handle + 1;
                return Ok(Outcome::success());
            }
            return Err(call.unsupported(
                session.service(),
                "CMIF control operation is not implemented",
            ));
        }
        if call.request.kind == RequestKind::DomainClose {
            return self.close_domain_object(session, call);
        }
        let object_id = call.request.domain.as_ref().map(|domain| domain.object_id);
        let object = session.object(object_id).ok_or_else(|| {
            call.unsupported(
                session.service(),
                format!(
                    "domain target {object_id:?} does not exist in this session; the server-close transport for TargetNotFound is not modeled"
                ),
            )
        })?;
        if matches!(object, ServiceObject::VideoDecoder) {
            let arguments: String = call.request.arguments[..call.request.arguments.len().min(16)]
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect();
            self.note_video(format!("{} {arguments}", call.request.command_id));
        }
        match object {
            ServiceObject::ServiceManager {
                registered_process_id,
            } => self.service_manager(*registered_process_id, call),
            ServiceObject::GraphicsSupervisor { application } => {
                let application = application.clone();
                self.graphics_supervisor(&application, call)
            }
            ServiceObject::SyncpointSupervisor => self.syncpoint_supervisor(call),
            // CreateVideoDecoderSession / CreateVideoEncoderSession (1): no
            // input, moves the session (Switchbrew "Switch 2: GSV services").
            ServiceObject::VideoDecoderCreator | ServiceObject::VideoEncoderCreator
                if call.request.command_id == 1 =>
            {
                call.shape(0, false, false)?;
                let decoder = matches!(object, ServiceObject::VideoDecoderCreator);
                let handle = self.next_available_handle()?;
                call.reply(0, Some(handle), 0, &[])?;
                let session = if decoder {
                    ServiceObject::VideoDecoder
                } else {
                    ServiceObject::VideoEncoder
                };
                self.sessions.insert(handle, Session::regular(session));
                self.next_handle = handle + 1;
                Ok(Outcome::success())
            }
            // VideoDecoderSession 201 (TIPC 217; Switchbrew's
            // InitializeVideoDecoderWithTransferMemory, listed as 200):
            // u64 transfer memory size, four u32s (OoT's AV1 decoder sends
            // 3, 3, 3, 1) and the copied transfer memory; no reply data
            // (client sdk+0x330d6c). The parameters are recorded. No host
            // video decoder exists yet. Provisional.
            ServiceObject::VideoDecoder if call.request.command_id == 201 => {
                call.shape_with_copies(24, false, false, 1)?;
                let arguments = &call.request.arguments;
                let size = crate::bytes::le_u64(&arguments[..8]);
                let parameters: Vec<u32> = (0..4)
                    .map(|index| crate::bytes::le_u32(&arguments[8 + index * 4..12 + index * 4]))
                    .collect();
                call.reply(0, None, 0, &[])?;
                self.video_decoders.push((size, parameters));
                self.av1_decoder = None;
                self.decoder_fences.clear();
                self.decoder_fences_released = false;
                self.provisional_commands
                    .insert("appletOE:video_decoder:201");
                Ok(Outcome::success())
            }
            // VideoDecoderSession 3 (GetInputEvent): no input; moves an
            // event the SDK wraps as an auto-clear system event (client
            // sdk+0x33109c). Signaled while the decoder accepts input; with
            // no host decoder it never holds input, so it starts signaled.
            ServiceObject::VideoDecoder if call.request.command_id == 3 => {
                call.shape(0, false, false)?;
                let application = self
                    .main_application
                    .clone()
                    .ok_or_else(|| invalid("video decoding requires the main application"))?;
                if self.readable_events.len() >= MAX_READABLE_EVENTS {
                    return Err(invalid("readable event backing capacity exhausted"));
                }
                let event = ReadableEventId(
                    self.readable_events
                        .keys()
                        .map(|id| id.0)
                        .max()
                        .unwrap_or(0)
                        + 1,
                );
                let handle = self.next_available_handle()?;
                call.reply(0, Some(handle), 0, &[])?;
                self.readable_events.insert(
                    event,
                    ReadableEvent {
                        owner: application,
                        kind: ReadableEventKind::VideoDecoderInput,
                        signaled: true,
                    },
                );
                self.video_input_event = Some(event);
                self.readable_event_handles.insert(handle, event);
                self.next_handle = handle + 1;
                self.provisional_commands.insert("appletOE:video_decoder:3");
                Ok(Outcome::success())
            }
            // VideoDecoderSession 311 (BindFrame; the SDK's
            // RegisterAv1DecoderBuffer, client sdk+0x330854) and 314
            // (Switchbrew's DestroyFrame; the SDK's
            // ReleaseAv1DecoderOutputBuffer, client sdk+0x331730): u32 slot
            // handle the SDK keeps per output slot, then 12 (311) or 28 (314:
            // u32 0, the slot's previous sync handle, u64 0) bytes -> the
            // slot's 16-byte NativeSyncHandle, which the SDK stores before
            // putting the slot on its ready ring (sdk+0x331810), from which
            // AcquireAv1DecoderOutputBuffer hands it out. The slot is queued
            // for the next decoded frame and the handle is a fence on that
            // frame: the slot's decoder fence token and the value it takes
            // once the frame is decoded (syncpoint supervisor 103/201/202
            // answer from it). Provisional.
            ServiceObject::VideoDecoder if matches!(call.request.command_id, 311 | 314) => {
                let register = call.request.command_id == 311;
                call.shape(if register { 16 } else { 32 }, false, false)?;
                let slot = crate::bytes::le_u32(&call.request.arguments[..4]);
                if self.decoder_slots.len() >= 64 {
                    return Err(invalid("decoder output slot queue overflow"));
                }
                // The slot's sync handle (u32 token, u8 64-bit flag, u64
                // threshold; layout from sdk+0x5d3a50) completes when the
                // next frame is decoded into the slot.
                let token = decoder_fence_token(slot);
                let threshold = self.decoder_fences.get(&token).copied().unwrap_or(0) + 1;
                self.decoder_fences.entry(token).or_insert(0);
                let mut reply = token.to_le_bytes().to_vec();
                reply.extend_from_slice(&1u32.to_le_bytes());
                reply.extend_from_slice(&threshold.to_le_bytes());
                call.reply(0, None, 0, &reply)?;
                // The decoder holds each slot once: a slot handed back while
                // it already waits for a frame (after Stop the SDK keeps
                // returning its slots) stays in its place.
                if !self.decoder_slots.contains(&slot) {
                    self.decoder_slots.push_back(slot);
                }
                self.decode_video_frames();
                if !register {
                    self.returned_frames += 1;
                }
                self.provisional_commands
                    .insert("appletOE:video_decoder:311/314");
                Ok(Outcome::success())
            }
            // VideoDecoderSession 213 (Switchbrew's GetFrame; read by the
            // SDK's GetAv1DecoderOutputBufferProperties, sdk+0x331a24): the
            // next decoded frame: u32 slot handle, four u32 sizes the SDK
            // keeps as u16 (width, height, then two more; answered as the
            // configured size twice), a u32, the u64 timestamp and a u32
            // whose bit 0 is the SDK's `last` output property: the frame of
            // the stream's last submission, which ends a movie. Frames fill
            // slots in the order they were handed to the decoder, each
            // taking the oldest submission's timestamp and picture. With no
            // slot or no
            // submission waiting there is no frame: the result is
            // 0x1c286f, which the SDK's AcquireAv1DecoderOutputBuffer itself
            // returns when no output is ready (sdk+0x330c80). Provisional.
            ServiceObject::VideoDecoder if call.request.command_id == 213 => {
                call.shape(0, false, false)?;
                self.decode_video_frames();
                let Some((slot, frame)) = self.decoded_video_frames.pop_front() else {
                    const NO_OUTPUT: u32 = 0x1c_286f;
                    call.reply(NO_OUTPUT, None, 0, &[])?;
                    self.empty_frame_requests += 1;
                    self.note_video("213 -> no output".into());
                    return Ok(Outcome::result(NO_OUTPUT));
                };
                let (width, height) = self
                    .video_decoders
                    .last()
                    .and_then(|(_, parameters)| Some((*parameters.get(4)?, *parameters.get(5)?)))
                    .unwrap_or((1920, 1080));
                let mut data = Vec::with_capacity(0x24);
                for value in [slot, width, height, width, height, 0] {
                    data.extend_from_slice(&value.to_le_bytes());
                }
                data.extend_from_slice(&frame.timestamp.to_le_bytes());
                data.extend_from_slice(&u32::from(frame.last).to_le_bytes());
                call.reply(0, None, 0, &data)?;
                match frame.picture {
                    Some(picture) => {
                        self.slot_pictures.insert(slot, picture);
                    }
                    None => {
                        self.slot_pictures.remove(&slot);
                    }
                }
                self.provisional_commands
                    .insert("appletOE:video_decoder:213");
                Ok(Outcome::success())
            }
            // VideoDecoderSession 109 (Decode; the SDK's
            // SubmitAv1DecoderInputBuffer, client sdk+0x330b00): the u64
            // presentation time and u32 flags (bit 0: `last`), and the AV1
            // bitstream (low-overhead OBUs) in a mapped buffer -> u64
            // submission id (sdk+0x330b44). Each is decoded on the host and
            // queued for a slot. Ids count from 1. Provisional.
            ServiceObject::VideoDecoder if call.request.command_id == 109 => {
                let Some(input) = call
                    .request
                    .input_buffer
                    .as_ref()
                    .map(|buffer| (buffer.address, buffer.size))
                else {
                    return Err(call.unsupported(object.service(), "Decode without input"));
                };
                if call.request.arguments.len() != 12 || input.1 > 0x100_0000 {
                    return Err(call.unsupported(
                        object.service(),
                        format!("Decode shape ({})", call.request.describe()),
                    ));
                }
                // A full decoder refuses the submission with 0x1c266f, which
                // OoT's player treats as "try again" (subsdk0+0xfd80 maps it
                // to 122).
                if self.video_input.len() >= VIDEO_INPUT_DEPTH {
                    const QUEUE_FULL: u32 = 0x1c_266f;
                    call.reply(QUEUE_FULL, None, 0, &[])?;
                    self.refused_video_submissions += 1;
                    return Ok(Outcome::result(QUEUE_FULL));
                }
                let mut bytes = vec![0u8; input.1 as usize];
                call.memory.read(input.0, &mut bytes)?;
                // u64 presentation time (microseconds), then u32 flags whose
                // bit 0 marks the stream's last submission (an empty one
                // with time -1 at the end of a movie).
                let timestamp = crate::bytes::le_u64(&call.request.arguments[..8]);
                let last = call.request.arguments[8] & 1 != 0;
                self.video_submissions += 1;
                let id = self.video_submissions;
                // GetFrame (213) consumes submissions in order.
                if self.video_input.len() >= 4096 {
                    return Err(invalid("video submission backlog overflow"));
                }
                // ECHORA_VIDEO_DUMP_DIR: the submissions of each decoder go
                // to decoder-<n>.obu there (n counts initializations), each
                // behind its length as a little-endian u32, for offline
                // decoding work.
                if let Some(directory) = std::env::var_os("ECHORA_VIDEO_DUMP_DIR") {
                    use std::io::Write;
                    let path = std::path::Path::new(&directory)
                        .join(format!("decoder-{}.obu", self.video_decoders.len()));
                    if let Ok(mut file) = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(path)
                    {
                        let _ = file.write_all(&(bytes.len() as u32).to_le_bytes());
                        let _ = file.write_all(&bytes);
                    }
                }
                let picture = self.decode_av1(&bytes);
                self.video_input.push((
                    id,
                    VideoFrame {
                        timestamp,
                        last,
                        picture,
                    },
                ));
                self.decode_video_frames();
                call.reply(0, None, 0, &id.to_le_bytes())?;
                self.provisional_commands
                    .insert("appletOE:video_decoder:109");
                Ok(Outcome::success())
            }
            // VideoDecoderSession 115 (FinalizeVideoDecoder; Switchbrew
            // name): no input or reply data (inferred). Queued slots and
            // submissions are dropped.
            ServiceObject::VideoDecoder if call.request.command_id == 115 => {
                call.shape(0, false, false)?;
                self.decoder_slots.clear();
                self.video_input.clear();
                self.decoded_video_frames.clear();
                self.video_input_event = None;
                self.video_decoder_running = false;
                self.av1_decoder = None;
                self.release_decoder_fences();
                call.reply(0, None, 0, &[])?;
                self.provisional_commands
                    .insert("appletOE:video_decoder:115");
                Ok(Outcome::success())
            }
            // VideoDecoderSession Start (6), Flush (8) and Stop (207)
            // (Switchbrew names): no input observed for Start; no reply data
            // (inferred). The decoder's running state is recorded.
            ServiceObject::VideoDecoder if matches!(call.request.command_id, 6 | 8 | 207) => {
                call.shape(0, false, false)?;
                match call.request.command_id {
                    6 => {
                        self.video_decoder_running = true;
                        self.decoder_fences_released = false;
                    }
                    207 => {
                        self.video_decoder_running = false;
                        self.release_decoder_fences();
                    }
                    _ => {}
                }
                call.reply(0, None, 0, &[])?;
                self.provisional_commands
                    .insert("appletOE:video_decoder:start_stop");
                Ok(Outcome::success())
            }
            // VideoDecoderSession 101 (Configure): u32 width, u32 height,
            // u32 frame rate (OoT: 1920, 1080, 60), four more words and a
            // mapped codec configuration buffer; no reply data (client
            // sdk+0x330fe4). Recorded with the last initialized decoder.
            ServiceObject::VideoDecoder if call.request.command_id == 101 => {
                if call.request.arguments.len() != 32 || call.request.input_buffer.is_none() {
                    return Err(call.unsupported(
                        object.service(),
                        format!("Configure shape ({})", call.request.describe()),
                    ));
                }
                let words: Vec<u32> = call
                    .request
                    .arguments
                    .chunks(4)
                    .map(crate::bytes::le_u32)
                    .collect();
                call.reply(0, None, 0, &[])?;
                if let Some((_, parameters)) = self.video_decoders.last_mut() {
                    parameters.extend(&words[..3]);
                }
                self.provisional_commands
                    .insert("appletOE:video_decoder:101");
                Ok(Outcome::success())
            }
            ServiceObject::VideoDecoderCreator
            | ServiceObject::VideoEncoderCreator
            | ServiceObject::VideoDecoder
            | ServiceObject::VideoEncoder => Err(call.unsupported(
                object.service(),
                format!("video command ({})", call.request.describe()),
            )),
            ServiceObject::GpuSession => self.gpu_session(call),
            ServiceObject::DisplaySession { .. } => self.display_session(call),
            ServiceObject::FileSystemProxy => self.file_system_proxy(call),
            ServiceObject::DataStorage => self.data_storage(call),
            ServiceObject::FileSystem { .. }
            | ServiceObject::File { .. }
            | ServiceObject::Directory { .. } => self.host_file_system(&object.clone(), call),
            ServiceObject::AudioManager
            | ServiceObject::AudioInManager
            | ServiceObject::AudioOutManager
            | ServiceObject::AudioRendererManager
            | ServiceObject::AudioDevice { .. }
            | ServiceObject::AudioOut { .. }
            | ServiceObject::AudioRenderer { .. }
            | ServiceObject::AudioDeviceSessionManager
            | ServiceObject::AudioDeviceSession
            | ServiceObject::OpusDecoderManager
            | ServiceObject::OpusDecoder { .. } => self.audio(&object.clone(), call),
            ServiceObject::SocketClient => self.socket_client(call),
            ServiceObject::SslService | ServiceObject::SslContext => {
                self.ssl(&object.clone(), call)
            }
            ServiceObject::NifmStatic
            | ServiceObject::NifmGeneral
            | ServiceObject::NifmRequest { .. } => self.nifm(&object.clone(), call),
            ServiceObject::NsdUser => self.nsd(call),
            ServiceObject::NfpUserManager | ServiceObject::NfpUser { .. } => {
                self.nfp(&object.clone(), call)
            }
            ServiceObject::BcatCreator
            | ServiceObject::BcatService
            | ServiceObject::BcatStorage
            | ServiceObject::BcatFile
            | ServiceObject::BcatDirectory
            | ServiceObject::BcatProgress => self.bcat(&object.clone(), call),
            ServiceObject::HidServer
            | ServiceObject::HidAppletResource
            | ServiceObject::HidVibrationDeviceList => self.hid(&object.clone(), call),
            ServiceObject::StreamplayUser | ServiceObject::StreamplayShim => {
                self.streamplay(&object.clone(), call)
            }
            ServiceObject::PlayReport => self.play_report(call),
            ServiceObject::PlayReportContext => self.play_report_context(call),
            ServiceObject::GpuWorker => self.gpu_worker(call),
            ServiceObject::NvidiaDriver { application }
                if call.request.protocol == Protocol::Cmif =>
            {
                let application = application.clone();
                self.nvidia_driver(&application, call)
            }
            _ if call.request.protocol != Protocol::Cmif => {
                Err(call.unsupported(object.service(), "this endpoint requires CMIF"))
            }
            ServiceObject::LogManager if call.request.command_id == 0 => {
                call.shape(8, true, false)?;
                self.reply_new_object(
                    call,
                    ServiceObject::Logger {
                        process_id: call.process_id,
                    },
                    LM_INTERFACE_ID,
                )
            }
            ServiceObject::Logger { process_id } => self.logger(*process_id, call),
            ServiceObject::AddOnContentManager => self.add_on_content(call),
            ServiceObject::AppletStorage { .. } | ServiceObject::AppletStorageAccessor { .. } => {
                self.applet_storage(&object.clone(), call)
            }
            ServiceObject::AccountService
            | ServiceObject::AccountProfile { .. }
            | ServiceObject::AccountManager { .. }
            | ServiceObject::AccountLicenseContext => self.account(&object.clone(), call),
            // Eden 1a6d7d4 glue/ectx.cpp: CreateContextRegistrar (0) returns a
            // registrar; its Complete (0) takes a u32 and an input buffer and
            // replies u32 0. Only the completion count and buffer size are kept
            // locally; the error-context payload is neither parsed nor retained.
            ServiceObject::TimeStatic
            | ServiceObject::SystemClock { .. }
            | ServiceObject::SteadyClock
            | ServiceObject::TimeZone => self.time(&object.clone(), call),
            ServiceObject::ParentalControlFactory | ServiceObject::ParentalControl { .. } => {
                self.parental_control(&object.clone(), call)
            }
            ServiceObject::ErrorContextWriter if call.request.command_id == 0 => {
                call.shape(0, false, false)?;
                self.reply_new_object(call, ServiceObject::ErrorContextRegistrar, AOC_INTERFACE_ID)
            }
            ServiceObject::ErrorContextRegistrar if call.request.command_id == 0 => {
                call.shape(4, false, true)?;
                let size = call
                    .request
                    .input_buffer
                    .as_ref()
                    .map_or(0, |buffer| buffer.size);
                call.reply(0, None, AOC_INTERFACE_ID, &0u32.to_le_bytes())?;
                self.error_contexts_completed += 1;
                self.error_context_bytes = self.error_context_bytes.saturating_add(size);
                Ok(Outcome::success())
            }
            ServiceObject::ApplicationProxyService if call.request.command_id == 10 => {
                self.open_application_proxy(call)
            }
            ServiceObject::LibraryAppletCreator { .. }
            | ServiceObject::LibraryAppletAccessor { .. }
            | ServiceObject::LibraryStorage { .. }
            | ServiceObject::LibraryStorageAccessor { .. } => {
                self.library_applet(&object.clone(), call)
            }
            ServiceObject::ApplicationProxy { application } if call.request.command_id == 1 => {
                self.open_self_controller(application, call)
            }
            ServiceObject::SelfController { application }
                if matches!(call.request.command_id, 10 | 11 | 12 | 14 | 16 | 19 | 50) =>
            {
                self.self_controller_setting(application, call)
            }
            // SetGraphicsScaleChangedNotificationEnabled(bool), named in Eden
            // 1a6d7d4; Switchbrew lists no command table for this interface.
            // No graphics-scale change source exists, so the flag is stored.
            ServiceObject::GraphicsScalingParametersFunctions { application }
                if call.request.command_id == 1050 =>
            {
                let length = if call.request.arguments.len() == 4 {
                    4
                } else {
                    1
                };
                call.shape(length, false, false)?;
                if self.main_application.as_ref() != Some(application) {
                    return Err(invalid(
                        "graphics-scaling owner must match the trusted bootstrap main application",
                    ));
                }
                let bytes = &call.request.arguments;
                if bytes[0] > 1 || bytes[1..].iter().any(|byte| *byte != 0) {
                    return Err(call.unsupported(
                        "appletOE:graphics_scaling_parameters_functions",
                        "notification flag is not a boolean",
                    ));
                }
                call.reply(0, None, SELF_CONTROLLER_INTERFACE_ID, &[])?;
                self.graphics_scale_changed_notification = Some(bytes[0] == 1);
                Ok(Outcome::success())
            }
            // GetGraphicsScalingParameters (10): a 12-byte {graphics_scale,
            // screen_ratio_numerator, screen_ratio_denominator}. Undocumented
            // on Switchbrew; Eden 1a6d7d4 returns {10, 1, 1} under its own
            // "prototype policy" (lowest SDK scale, unit screen ratio). This
            // follows that value provisionally; no scale change is produced.
            ServiceObject::GraphicsScalingParametersFunctions { application }
                if call.request.command_id == 10 =>
            {
                call.shape(0, false, false)?;
                if self.main_application.as_ref() != Some(application) {
                    return Err(invalid(
                        "graphics-scaling owner must match the trusted bootstrap main application",
                    ));
                }
                let mut data = Vec::with_capacity(12);
                for value in PROVISIONAL_GRAPHICS_SCALING_PARAMETERS {
                    data.extend_from_slice(&value.to_le_bytes());
                }
                call.reply(0, None, SELF_CONTROLLER_INTERFACE_ID, &data)?;
                Ok(Outcome::success())
            }
            // GetLibraryAppletLaunchableEvent (9; Switchbrew "Applet Manager
            // services"): a copied event, signaled while library applets can
            // be launched, which on the host is always (Eden signals it too).
            ServiceObject::SelfController { application } if call.request.command_id == 9 => {
                call.shape(0, false, false)?;
                if self.main_application.as_ref() != Some(application) {
                    return Err(invalid(
                        "self-controller owner must match the trusted bootstrap main application",
                    ));
                }
                let slot = self.library_applet_launchable_event;
                let (event, outcome) = self.reply_applet_event(call, slot, true)?;
                self.library_applet_launchable_event = Some(event);
                Ok(outcome)
            }
            // SetIdleTimeDetectionExtension (62, u32) and
            // GetIdleTimeDetectionExtension (63) (Switchbrew "Applet Manager
            // services"): the extension is stored; there is no idle
            // detection on the host for it to extend.
            ServiceObject::SelfController { application }
                if matches!(call.request.command_id, 62 | 63) =>
            {
                let set = call.request.command_id == 62;
                call.shape(if set { 4 } else { 0 }, false, false)?;
                if self.main_application.as_ref() != Some(application) {
                    return Err(invalid(
                        "self-controller owner must match the trusted bootstrap main application",
                    ));
                }
                if set {
                    self.idle_time_detection_extension =
                        crate::bytes::le_u32(&call.request.arguments[..4]);
                    call.reply(0, None, SELF_CONTROLLER_INTERFACE_ID, &[])?;
                } else {
                    let value = self.idle_time_detection_extension;
                    call.reply(0, None, SELF_CONTROLLER_INTERFACE_ID, &value.to_le_bytes())?;
                }
                Ok(Outcome::success())
            }
            // LockExit (1) / UnlockExit (2) (Switchbrew "Applet Manager
            // services"): defer or allow system exit requests. No exit
            // request source exists here; the lock state is stored.
            ServiceObject::SelfController { application }
                if matches!(call.request.command_id, 1 | 2) =>
            {
                call.shape(0, false, false)?;
                if self.main_application.as_ref() != Some(application) {
                    return Err(invalid(
                        "self-controller owner must match the trusted bootstrap main application",
                    ));
                }
                call.reply(0, None, SELF_CONTROLLER_INTERFACE_ID, &[])?;
                self.exit_locked = call.request.command_id == 1;
                Ok(Outcome::success())
            }
            // SetCpuBoostMode (66, u32; Switchbrew "Applet Manager
            // services"): a performance hint. The native host has no CPU
            // boost configuration, so the requested mode is stored.
            ServiceObject::CommonStateGetter { application } if call.request.command_id == 66 => {
                call.shape(4, false, false)?;
                if self.main_application.as_ref() != Some(application) {
                    return Err(invalid(
                        "lifecycle endpoint owner must match the trusted bootstrap main application",
                    ));
                }
                let mode = crate::bytes::le_u32(&call.request.arguments[..4]);
                if mode > 2 {
                    return Err(call.unsupported(
                        "appletOE:common_state_getter",
                        "CPU boost mode outside the documented values",
                    ));
                }
                call.reply(0, None, COMMON_STATE_GETTER_INTERFACE_ID, &[])?;
                self.cpu_boost_mode = Some(mode);
                Ok(Outcome::success())
            }
            ServiceObject::SelfController { application } if call.request.command_id == 2000 => {
                self.get_graphics_supervisor(application, call)
            }
            ServiceObject::SelfController { application } if call.request.command_id == 13 => {
                self.set_focus_handling_mode(application, call)
            }
            ServiceObject::SelfController { application } if call.request.command_id == 91 => self
                .copy_application_event(
                    application,
                    call,
                    ReadableEventKind::AccumulatedSuspendedTickChanged,
                    SELF_CONTROLLER_INTERFACE_ID,
                ),
            // GetScreenShotAndContinuousRecordingInitiallyEnabled: Eden 1a6d7d4
            // derives screenshots from the stored SelfController permission
            // (enabled unless Disable = 2) and recording from gameplay-recording
            // support plus Enabled state. Recording is never initialized here.
            // Command 321 is Switch 2-only and undocumented (Switchbrew lists it
            // as "[S2] [20.0.0+]" with no name or I/O; the pinned Eden snapshot
            // does not define it). Provisional, explicitly reported policy:
            // success with eight zero output bytes, so any small scalar output
            // reads as 0/false ("feature off"). Not reference-backed.
            ServiceObject::ApplicationFunctions { application }
                if call.request.command_id == 321 =>
            {
                self.provisional_reply(
                    application,
                    call,
                    "appletOE:application_functions:321",
                    0,
                    APPLICATION_FUNCTIONS_INTERFACE_ID,
                )
            }
            // SelfController 2400 is Switch 2-only and undocumented (absent from
            // the pinned Eden snapshot and Switchbrew). Observed input: one byte.
            ServiceObject::SelfController { application } if call.request.command_id == 2400 => {
                self.provisional_reply(
                    application,
                    call,
                    "appletOE:self_controller:2400",
                    1,
                    SELF_CONTROLLER_INTERFACE_ID,
                )
            }
            ServiceObject::ApplicationFunctions { application }
                if matches!(call.request.command_id, 40 | 330) =>
            {
                call.shape(0, false, false)?;
                if self.main_application.as_ref() != Some(application) {
                    return Err(invalid(
                        "application-functions owner must match the trusted main application",
                    ));
                }
                // NotifyRunning returns became_running = true (Eden 1a6d7d4).
                // Command 330 is unnamed; the reference replies with u8 0.
                let notify = call.request.command_id == 40;
                let data = [u8::from(notify)];
                call.reply(0, None, APPLICATION_FUNCTIONS_INTERFACE_ID, &data)?;
                if notify {
                    if let Some(state) = self.main_application_state.as_mut() {
                        state.running_notified = true;
                    }
                }
                Ok(Outcome::success())
            }
            ServiceObject::ApplicationFunctions { application }
                if call.request.command_id == 320 =>
            {
                call.shape(0, false, false)?;
                if self.main_application.as_ref() != Some(application) {
                    return Err(invalid(
                        "application-functions owner must match the trusted main application",
                    ));
                }
                let state = self
                    .main_application_state
                    .as_ref()
                    .ok_or_else(|| invalid("query requires initialized application state"))?;
                let screenshot = state.self_settings.screenshot_permission.unwrap_or(0) != 2;
                call.reply(
                    0,
                    None,
                    APPLICATION_FUNCTIONS_INTERFACE_ID,
                    &[u8::from(screenshot), 0],
                )?;
                Ok(Outcome::success())
            }
            ServiceObject::ApplicationFunctions { application }
                if call.request.command_id == 130 =>
            {
                self.copy_application_event(
                    application,
                    call,
                    ReadableEventKind::GpuErrorDetected,
                    APPLICATION_FUNCTIONS_INTERFACE_ID,
                )
            }
            // PopLaunchParameter(u32 kind) -> IStorage. UserChannel (1) is empty
            // for a frontend launch. AccountPreselectedUser (2) holds one
            // 0x88-byte record {magic, is_account_selected = 1, Uid, zero
            // padding} for the local user, as Eden 1a6d7d4 pushes at launch;
            // popping consumes it. Empty channels reply ResultNoDataInChannel.
            ServiceObject::ApplicationFunctions { application } if call.request.command_id == 1 => {
                call.shape(4, false, false)?;
                if self.main_application.as_ref() != Some(application) {
                    return Err(invalid(
                        "application-functions owner must match the trusted main application",
                    ));
                }
                let kind = crate::bytes::le_u32(&call.request.arguments[..4]);
                match kind {
                    2 if !self.preselected_user_popped => {
                        let mut data = vec![0u8; 0x88];
                        data[..4].copy_from_slice(&PRESELECTED_USER_MAGIC.to_le_bytes());
                        data[4..8].copy_from_slice(&1u32.to_le_bytes());
                        data[8..24].copy_from_slice(&self.local_user.uid);
                        let outcome = self.reply_new_object(
                            call,
                            ServiceObject::AppletStorage { data },
                            APPLICATION_FUNCTIONS_INTERFACE_ID,
                        )?;
                        self.preselected_user_popped = true;
                        Ok(outcome)
                    }
                    1 | 2 => {
                        call.reply(
                            AM_RESULT_NO_DATA_IN_CHANNEL,
                            None,
                            APPLICATION_FUNCTIONS_INTERFACE_ID,
                            &[],
                        )?;
                        Ok(Outcome::result(AM_RESULT_NO_DATA_IN_CHANNEL))
                    }
                    _ => Err(call.unsupported(
                        "appletOE:application_functions",
                        format!("launch parameter kind {kind} is not defined"),
                    )),
                }
            }
            // GetDisplayVersion -> DisplayVersion (0x10-byte NUL-padded
            // string): the NACP DisplayVersion (offset 0x3060, Switchbrew
            // "NACP") when ECHORA_NACP names the control.nacp. Without it,
            // native-PC policy: "1.0.0" (the dump here has no control data;
            // the application reaches this after the trainer is confirmed).
            ServiceObject::ApplicationFunctions { .. } if call.request.command_id == 23 => {
                call.shape(0, false, false)?;
                let mut version = [0u8; 0x10];
                let from_nacp = std::env::var_os("ECHORA_NACP")
                    .and_then(|path| std::fs::read(path).ok())
                    .filter(|bytes| bytes.len() == crate::content_metadata::NACP_SIZE);
                match from_nacp {
                    Some(bytes) => version.copy_from_slice(&bytes[0x3060..0x3070]),
                    None => version[..5].copy_from_slice(b"1.0.0"),
                }
                call.reply(0, None, APPLICATION_FUNCTIONS_INTERFACE_ID, &version)?;
                Ok(Outcome::success())
            }
            // Command 210 is undocumented ("[20.0.0+]" on Switchbrew). Eden
            // 1a6d7d4 copies the applet's never-signaled unknown_event; this
            // follows that reference (provisional).
            ServiceObject::ApplicationFunctions { application }
                if call.request.command_id == 210 =>
            {
                self.copy_application_event(
                    application,
                    call,
                    ReadableEventKind::ApplicationUnknown210,
                    APPLICATION_FUNCTIONS_INTERFACE_ID,
                )
            }
            ServiceObject::ApplicationFunctions { application }
                if call.request.command_id == 140 =>
            {
                self.copy_application_event(
                    application,
                    call,
                    ReadableEventKind::FriendInvitationStorageChannel,
                    APPLICATION_FUNCTIONS_INTERFACE_ID,
                )
            }
            // TryPopFromFriendInvitationStorageChannel (141) (output buffer):
            // nothing is ever queued, so 128-2 (no data in the channel), which
            // nn::friends reads as "no invitation" (Eden-NS2).
            ServiceObject::ApplicationFunctions { .. } if call.request.command_id == 141 => {
                call.shape_output(0, false)?;
                const APPLET_RESULT_NO_DATA_IN_CHANNEL: u32 = 128 | (2 << 9);
                call.reply(
                    APPLET_RESULT_NO_DATA_IN_CHANNEL,
                    None,
                    APPLICATION_FUNCTIONS_INTERFACE_ID,
                    &[],
                )?;
                Ok(Outcome::result(APPLET_RESULT_NO_DATA_IN_CHANNEL))
            }
            ServiceObject::WindowController { application } if call.request.command_id == 10 => {
                call.shape(0, false, false)?;
                if self.main_application.as_ref() != Some(application) {
                    return Err(invalid(
                        "window endpoint owner must match the trusted bootstrap main application",
                    ));
                }
                call.reply(0, None, WINDOW_CONTROLLER_INTERFACE_ID, &[])?;
                if let Some(state) = self.main_application_state.as_mut() {
                    state.foreground_rights_acquired = true;
                }
                Ok(Outcome::success())
            }
            ServiceObject::WindowController { application } if call.request.command_id == 1 => {
                call.shape(0, false, false)?;
                // IWindowController copies its stored Applet.aruid; Applet
                // initializes that ProcessId from its owned process. Do not
                // substitute the request sender, program ID or global binding.
                // This query does not allocate an ID or register with HID/VI.
                call.reply(
                    0,
                    None,
                    WINDOW_CONTROLLER_INTERFACE_ID,
                    &application.process_id.to_le_bytes(),
                )?;
                Ok(Outcome::success())
            }
            ServiceObject::CommonStateGetter { application } if call.request.command_id == 0 => {
                self.copy_application_event(
                    application,
                    call,
                    ReadableEventKind::LifecycleSystem,
                    COMMON_STATE_GETTER_INTERFACE_ID,
                )
            }
            ServiceObject::CommonStateGetter { application }
                if matches!(call.request.command_id, 5 | 6) =>
            {
                call.shape(0, false, false)?;
                if self.main_application.as_ref() != Some(application) {
                    return Err(invalid(
                        "lifecycle endpoint owner must match the trusted bootstrap main application",
                    ));
                }
                let data = if call.request.command_id == 5 {
                    vec![OPERATION_MODE_DOCKED]
                } else {
                    PERFORMANCE_MODE_BOOST.to_le_bytes().to_vec()
                };
                call.reply(0, None, COMMON_STATE_GETTER_INTERFACE_ID, &data)?;
                Ok(Outcome::success())
            }
            ServiceObject::CommonStateGetter { application }
                if matches!(call.request.command_id, 1 | 9) =>
            {
                self.common_state_lifecycle(application, call)
            }
            ServiceObject::CommonStateGetter { application } if call.request.command_id == 900 => {
                call.shape(0, false, false)?;
                if self.main_application.as_ref() != Some(application) {
                    return Err(invalid(
                        "CommonStateGetter owner must match the trusted bootstrap main application",
                    ));
                }
                let mut updated = self.main_application_state.clone().ok_or_else(|| {
                    invalid(
                        "CommonStateGetter configuration requires initialized application state",
                    )
                })?;
                // The reference setter changes only this Applet-owned bool.
                // Repeated calls are idempotent; no exit, next-program action
                // or lifecycle/event side effect is modeled by this command.
                updated.request_exit_to_library_applet_at_execute_next_program_enabled = true;
                call.reply(0, None, COMMON_STATE_GETTER_INTERFACE_ID, &[])?;
                self.main_application_state = Some(updated);
                Ok(Outcome::success())
            }
            ServiceObject::ApplicationProxy { application }
                if matches!(
                    call.request.command_id,
                    0 | 2 | 3 | 4 | 10 | 11 | 20 | 40 | 60 | 1000
                ) =>
            {
                call.shape(0, false, false)?;
                // These constructors only retain references or per-object
                // default values and register handlers in the audited source.
                // No window/event/display/audio/lifecycle backend is created.
                // The guard preserves unsupported-command behavior.
                let child = match call.request.command_id {
                    0 => ServiceObject::CommonStateGetter {
                        application: application.clone(),
                    },
                    2 => ServiceObject::WindowController {
                        application: application.clone(),
                    },
                    // Value state belongs to each fresh endpoint; the source
                    // constructor has no Applet/process or shared audio owner.
                    3 => ServiceObject::AudioController {
                        state: AudioControllerState::REFERENCE_DEFAULTS,
                    },
                    4 => ServiceObject::DisplayController {
                        application: application.clone(),
                    },
                    10 => ServiceObject::ProcessWindingController {
                        application: application.clone(),
                    },
                    11 => ServiceObject::LibraryAppletCreator {
                        application: application.clone(),
                    },
                    20 => ServiceObject::ApplicationFunctions {
                        application: application.clone(),
                    },
                    40 => ServiceObject::PerformanceFunctions,
                    60 => ServiceObject::GraphicsScalingParametersFunctions {
                        application: application.clone(),
                    },
                    // ApplicationProxy has no appletAE debug-setting gate. The
                    // constructor has no Applet binding or mutable debug state.
                    1000 => ServiceObject::DebugFunctions,
                    _ => unreachable!("proxy getter guard and construction must agree"),
                };
                self.reply_new_object(call, child, APPLICATION_PROXY_INTERFACE_ID)
            }
            // IApplicationFunctions SetTerminateResult (22): u32 result the
            // application reports for its termination (an abort reason). It
            // is recorded for the report; termination itself is not modeled.
            ServiceObject::ApplicationFunctions { .. } if call.request.command_id == 22 => {
                call.shape(4, false, false)?;
                let result = crate::bytes::le_u32(&call.request.arguments[..4]);
                self.terminate_result = Some(result);
                call.reply(0, None, APPLICATION_FUNCTIONS_INTERFACE_ID, &[])?;
                Ok(Outcome::success())
            }
            // IApplicationFunctions EnsureSaveData (20): Uid (16 bytes) ->
            // u64 required additional size, 0 when the save data exists.
            // Reported as present; mounting it is the file system service's job.
            ServiceObject::ApplicationFunctions { .. } if call.request.command_id == 20 => {
                call.shape(16, false, false)?;
                call.reply(
                    0,
                    None,
                    APPLICATION_FUNCTIONS_INTERFACE_ID,
                    &0u64.to_le_bytes(),
                )?;
                Ok(Outcome::success())
            }
            // IApplicationFunctions GetSaveDataSize (26): SaveDataType (u8,
            // padded to 8 bytes) and Uid (16 bytes) -> i64 size and i64
            // journal size. Like Eden-NS2 with no recorded size: both zero.
            ServiceObject::ApplicationFunctions { .. } if call.request.command_id == 26 => {
                call.shape(24, false, false)?;
                call.reply(0, None, APPLICATION_FUNCTIONS_INTERFACE_ID, &[0u8; 16])?;
                Ok(Outcome::success())
            }
            // IApplicationFunctions GetSaveDataSizeMax (28) -> i64 maximum
            // size and i64 maximum journal size. Eden-NS2 answers 0xFFFFFFF
            // for both (no limit worth checking).
            ServiceObject::ApplicationFunctions { .. } if call.request.command_id == 28 => {
                call.shape(0, false, false)?;
                let mut reply = Vec::with_capacity(16);
                for value in [0xFFF_FFFFu64, 0xFFF_FFFF] {
                    reply.extend_from_slice(&value.to_le_bytes());
                }
                call.reply(0, None, APPLICATION_FUNCTIONS_INTERFACE_ID, &reply)?;
                Ok(Outcome::success())
            }
            // IApplicationFunctions ExtendSaveData (25): SaveDataType (u8,
            // padded to 8 bytes), Uid (16 bytes), i64 size, i64 journal
            // size -> u64 additional space required: 0, the save data
            // always fits (as for EnsureSaveData).
            ServiceObject::ApplicationFunctions { .. } if call.request.command_id == 25 => {
                call.shape(40, false, false)?;
                call.reply(
                    0,
                    None,
                    APPLICATION_FUNCTIONS_INTERFACE_ID,
                    &0u64.to_le_bytes(),
                )?;
                Ok(Outcome::success())
            }
            // IApplicationFunctions GetDesiredLanguage (21): the LanguageCode
            // (an ASCII tag in a u64) chosen from the application's supported
            // languages and the system setting. This host reports "en-US".
            ServiceObject::ApplicationFunctions { .. } if call.request.command_id == 21 => {
                call.shape(0, false, false)?;
                let mut language = [0u8; 8];
                language[..5].copy_from_slice(b"en-US");
                call.reply(0, None, APPLICATION_FUNCTIONS_INTERFACE_ID, &language)?;
                Ok(Outcome::success())
            }
            _ => Err(call.unsupported(object.service(), "service command is not implemented")),
        }
    }

    /// aoc:u IAddOnContentManager, following the pinned Eden-NS2
    /// addon_content_manager.cpp: CountAddOnContent (2) counts selected add-ons
    /// whose base title matches the caller; GetAddOnContentBaseId (5) is the
    /// caller's base program ID + 0x1000 (no control metadata is loaded);
    /// GetAddOnContentListChangedEvent (8) copies an application-owned event.
    /// PID inputs must be the trusted main application's transport identity.
    fn add_on_content(&mut self, call: &mut Call<'_>) -> Result<Outcome, Trap> {
        let application = self
            .main_application
            .clone()
            .ok_or_else(|| invalid("aoc:u requires the bootstrap main application"))?;
        match call.request.command_id {
            2 | 5 => {
                call.shape(8, true, false)?;
                if application.process_id != call.process_id {
                    return Err(invalid("aoc:u caller PID is not the main application"));
                }
                // Eden FileSys common_funcs.h: base = id & !0x1fff; add-on base
                // = base + 0x1000; an add-on matches when its base equals ours.
                const BASE_TITLE_ID_MASK: u64 = 0xffff_ffff_ffff_e000;
                let base = application.program_id & BASE_TITLE_ID_MASK;
                let data = if call.request.command_id == 2 {
                    let count = self
                        .add_on_content
                        .iter()
                        .filter(|id| *id & BASE_TITLE_ID_MASK == base)
                        .count() as u32;
                    count.to_le_bytes().to_vec()
                } else {
                    (base + 0x1000).to_le_bytes().to_vec()
                };
                call.reply(0, None, AOC_INTERFACE_ID, &data)?;
                Ok(Outcome::success())
            }
            8 => self.copy_application_event(
                &application,
                call,
                ReadableEventKind::AddOnContentListChanged,
                AOC_INTERFACE_ID,
            ),
            // CheckAddOnContentMountStatus (50): PID + u64 placeholder; fails
            // only when mounted add-on content became unavailable. No add-on
            // content is mounted by this runtime, and nothing is ever removed
            // while it runs, so the check succeeds.
            50 => {
                call.shape(8, true, false)?;
                if application.process_id != call.process_id {
                    return Err(invalid("aoc:u caller PID is not the main application"));
                }
                call.reply(0, None, AOC_INTERFACE_ID, &[])?;
                Ok(Outcome::success())
            }
            _ => Err(call.unsupported("aoc:u", "add-on content command is not implemented")),
        }
    }

    /// AM IStorage Open (0) and IStorageAccessor GetSize (0) / Read (11),
    /// following Eden 1a6d7d4: reads outside the buffer return
    /// ResultInvalidOffset. Write stops: launch parameters are read-only here.
    fn applet_storage(
        &mut self,
        object: &ServiceObject,
        call: &mut Call<'_>,
    ) -> Result<Outcome, Trap> {
        const SERVICE: &str = "appletOE:storage";
        match (object, call.request.command_id) {
            (ServiceObject::AppletStorage { data }, 0) => {
                call.shape(0, false, false)?;
                self.reply_new_object(
                    call,
                    ServiceObject::AppletStorageAccessor { data: data.clone() },
                    APPLICATION_FUNCTIONS_INTERFACE_ID,
                )
            }
            (ServiceObject::AppletStorageAccessor { data }, 0) => {
                call.shape(0, false, false)?;
                let size = data.len() as i64;
                call.reply(
                    0,
                    None,
                    APPLICATION_FUNCTIONS_INTERFACE_ID,
                    &size.to_le_bytes(),
                )?;
                Ok(Outcome::success())
            }
            (ServiceObject::AppletStorageAccessor { data }, 11) => {
                call.shape_output(8, false)?;
                let offset = crate::bytes::le_i64(&call.request.arguments[..8]);
                let (address, size) = call.output_buffer()?;
                let in_range = u64::try_from(offset)
                    .ok()
                    .and_then(|begin| begin.checked_add(size).map(|end| (begin, end)))
                    .filter(|&(_, end)| end <= data.len() as u64);
                let Some((begin, end)) = in_range else {
                    call.reply(
                        AM_RESULT_INVALID_OFFSET,
                        None,
                        APPLICATION_FUNCTIONS_INTERFACE_ID,
                        &[],
                    )?;
                    return Ok(Outcome::result(AM_RESULT_INVALID_OFFSET));
                };
                call.memory
                    .write(address, &data[begin as usize..end as usize])?;
                call.reply(0, None, APPLICATION_FUNCTIONS_INTERFACE_ID, &[])?;
                Ok(Outcome::success())
            }
            _ => Err(call.unsupported(SERVICE, "storage command is not implemented")),
        }
    }

    /// acc:u0 (Switchbrew "Account services"). Native-PC policy: exactly one
    /// local user exists, is open, and is the preselected user, as when an
    /// application that requires a startup account is launched for that
    /// user. Commands are added as the application reaches them.
    fn account(&mut self, object: &ServiceObject, call: &mut Call<'_>) -> Result<Outcome, Trap> {
        const SERVICE: &str = "acc:u0";
        let application = self
            .main_application
            .clone()
            .ok_or_else(|| invalid("acc:u0 requires the bootstrap main application"))?;
        let uid = self.local_user.uid;
        match (object, call.request.command_id) {
            // InitializeApplicationInfoV0 (100), InitializeApplicationInfo
            // (140) and InitializeApplicationInfoV2 (160, listed without a
            // description; same PID + u64 placeholder shape): no output.
            (ServiceObject::AccountService, 100 | 140 | 160) => {
                call.shape(8, true, false)?;
                if application.process_id != call.process_id {
                    return Err(invalid("acc:u0 caller PID is not the main application"));
                }
                call.reply(0, None, AOC_INTERFACE_ID, &[])?;
                self.account_application_info_initialized = true;
                Ok(Outcome::success())
            }
            // IsUserAccountSwitchLocked -> bool: the NACP UserAccountSwitchLock
            // value when the control.nacp is supplied. Without it, native-PC
            // policy: the single local user cannot be switched, so switching
            // is reported as locked.
            (ServiceObject::AccountService, 150) => {
                call.shape(0, false, false)?;
                let locked = match &self.nacp_user_account_switch_lock {
                    Some(Ok(value @ (0 | 1))) => *value,
                    Some(Ok(value)) => {
                        return Err(call.unsupported(
                            SERVICE,
                            format!("NACP UserAccountSwitchLock {value} is not a boolean"),
                        ));
                    }
                    Some(Err(error)) => return Err(call.unsupported(SERVICE, error.clone())),
                    None => 1,
                };
                call.reply(0, None, AOC_INTERFACE_ID, &[locked])?;
                Ok(Outcome::success())
            }
            // GetUserCount: s32.
            (ServiceObject::AccountService, 0) => {
                call.shape(0, false, false)?;
                call.reply(0, None, AOC_INTERFACE_ID, &1i32.to_le_bytes())?;
                Ok(Outcome::success())
            }
            // GetUserExistence(Uid) -> bool.
            (ServiceObject::AccountService, 1) => {
                call.shape(16, false, false)?;
                let exists = call.request.arguments[..] == uid;
                call.reply(0, None, AOC_INTERFACE_ID, &[u8::from(exists)])?;
                Ok(Outcome::success())
            }
            // ListAllUsers (2) / ListOpenUsers (3): Uid array in a type-0xA
            // (C pointer) output buffer, unused slots zero (invalid Uid).
            (ServiceObject::AccountService, 2 | 3) => {
                if !call.request.arguments.is_empty()
                    || call.request.pid.is_some()
                    || call.request.input_buffer.is_some()
                    || !call.request.copy_handles.is_empty()
                    || !call.request.receive_buffers.is_empty()
                    || call.request.receive_list.len() != 1
                {
                    return Err(invalid(
                        "account user list expects one pointer output buffer",
                    ));
                }
                let (address, size) = call.request.receive_list[0];
                let mut list = vec![0u8; size as usize];
                if size >= 16 {
                    list[..16].copy_from_slice(&uid);
                }
                call.memory.write(address, &list)?;
                call.reply(0, None, AOC_INTERFACE_ID, &[])?;
                Ok(Outcome::success())
            }
            // GetLastOpenedUser -> Uid.
            (ServiceObject::AccountService, 4) => {
                call.shape(0, false, false)?;
                call.reply(0, None, AOC_INTERFACE_ID, &uid)?;
                Ok(Outcome::success())
            }
            // GetProfile(Uid) -> IProfile. Unknown users stop explicitly: the
            // documented page lists no result code for them.
            (ServiceObject::AccountService, 5) => {
                call.shape(16, false, false)?;
                if call.request.arguments[..] != uid {
                    return Err(call.unsupported(SERVICE, "GetProfile for an unknown user"));
                }
                self.reply_new_object(
                    call,
                    ServiceObject::AccountProfile { uid },
                    AOC_INTERFACE_ID,
                )
            }
            // GetBaasAccountManagerForApplication(Uid) -> IManagerForApplication.
            (ServiceObject::AccountService, 101) => {
                call.shape(16, false, false)?;
                if call.request.arguments[..] != uid {
                    return Err(call.unsupported(
                        SERVICE,
                        "GetBaasAccountManagerForApplication for an unknown user",
                    ));
                }
                self.reply_new_object(
                    call,
                    ServiceObject::AccountManager { uid },
                    AOC_INTERFACE_ID,
                )
            }
            // IManagerForApplication CheckAvailability (0) and GetAccountId
            // (1): the local user has no network service account, so both
            // report Switchbrew 2124-0200 "network service account must be
            // registered first" (0x1907C).
            (ServiceObject::AccountManager { .. }, 0 | 1) => {
                call.shape(0, false, false)?;
                call.reply(
                    ACCOUNT_RESULT_NSA_REGISTRATION_REQUIRED,
                    None,
                    AOC_INTERFACE_ID,
                    &[],
                )?;
                Ok(Outcome::result(ACCOUNT_RESULT_NSA_REGISTRATION_REQUIRED))
            }
            // LoadIdTokenCacheDeprecated (3) / LoadIdTokenCache (4) (output
            // buffer): no network service account, so no ID token either;
            // the SDK returns the result to the game.
            (ServiceObject::AccountManager { .. }, 3 | 4) => {
                call.shape_output(0, false)?;
                call.reply(
                    ACCOUNT_RESULT_NSA_REGISTRATION_REQUIRED,
                    None,
                    AOC_INTERFACE_ID,
                    &[],
                )?;
                Ok(Outcome::result(ACCOUNT_RESULT_NSA_REGISTRATION_REQUIRED))
            }
            // LoadNetworkServiceLicenseKindAsync (170) -> an async context.
            // Offline there is no Switch Online licence: the context is done
            // at once, its GetResult fails and the kind is 0 (none).
            (ServiceObject::AccountManager { .. }, 170) => {
                call.shape(0, false, false)?;
                self.reply_new_object(call, ServiceObject::AccountLicenseContext, AOC_INTERFACE_ID)
            }
            // IAsyncContext GetSystemEvent (0) -> copied event, signaled.
            (ServiceObject::AccountLicenseContext, 0) => {
                call.shape(0, false, false)?;
                self.reply_done_event(call)
            }
            // Cancel (1): nothing is in flight.
            (ServiceObject::AccountLicenseContext, 1) => {
                call.shape(0, false, false)?;
                call.reply(0, None, AOC_INTERFACE_ID, &[])?;
                Ok(Outcome::success())
            }
            // HasDone (2) -> bool: always done.
            (ServiceObject::AccountLicenseContext, 2) => {
                call.shape(0, false, false)?;
                call.reply(0, None, AOC_INTERFACE_ID, &[1])?;
                Ok(Outcome::success())
            }
            // GetResult (3): 110-300, no network connection (offline).
            (ServiceObject::AccountLicenseContext, 3) => {
                call.shape(0, false, false)?;
                const NETWORK_RESULT_OFFLINE: u32 = 110 | (300 << 9);
                call.reply(NETWORK_RESULT_OFFLINE, None, AOC_INTERFACE_ID, &[])?;
                Ok(Outcome::result(NETWORK_RESULT_OFFLINE))
            }
            // GetNetworkServiceLicenseKind (100) -> u32: 0, no licence.
            (ServiceObject::AccountLicenseContext, 100) => {
                call.shape(0, false, false)?;
                call.reply(0, None, AOC_INTERFACE_ID, &0u32.to_le_bytes())?;
                Ok(Outcome::success())
            }
            // IsUserRegistrationRequestPermitted(PID, u64) -> bool. There is
            // no registration UI on this host.
            (ServiceObject::AccountService, 50) => {
                call.shape(8, true, false)?;
                call.reply(0, None, AOC_INTERFACE_ID, &[0])?;
                Ok(Outcome::success())
            }
            // TrySelectUserWithoutInteraction(bool network_required) -> Uid.
            // With a single local user and no network service account, it is
            // selected unless a network service account is required.
            (ServiceObject::AccountService, 51 | 52) => {
                call.shape(1, false, false)?;
                let required = call.request.arguments[0];
                if required > 1 {
                    return Err(invalid("network-required flag is not a boolean"));
                }
                let selected = if required == 0 { uid } else { [0; 16] };
                call.reply(0, None, AOC_INTERFACE_ID, &selected)?;
                Ok(Outcome::success())
            }
            // IProfile GetBase (1) -> ProfileBase, Get (0) additionally fills
            // a 0x80-byte UserData pointer buffer.
            (ServiceObject::AccountProfile { uid }, 0 | 1) => {
                let get = call.request.command_id == 0;
                if !call.request.arguments.is_empty()
                    || call.request.pid.is_some()
                    || call.request.input_buffer.is_some()
                    || !call.request.copy_handles.is_empty()
                    || !call.request.receive_buffers.is_empty()
                    || call.request.receive_list.len() != usize::from(get)
                {
                    return Err(invalid("IProfile Get/GetBase buffer shape mismatch"));
                }
                if get {
                    let (address, size) = call.request.receive_list[0];
                    if size < 0x80 {
                        return Err(invalid("UserData buffer is smaller than 0x80 bytes"));
                    }
                    call.memory.write(address, &self.local_user.user_data())?;
                }
                let base = self.local_user.profile_base(uid);
                call.reply(0, None, AOC_INTERFACE_ID, &base)?;
                Ok(Outcome::success())
            }
            _ => Err(call.unsupported(SERVICE, "account command is not implemented")),
        }
    }

    /// time:u, following the pinned Eden-NS2 glue/psc time sources' shapes.
    /// Native-PC policy: the host UTC clock is the source of truth for the
    /// user, network and local system clocks; the steady clock counts host
    /// monotonic seconds from session start with a per-launch random clock
    /// source ID (as after an RTC reset). SystemClockContext is
    /// {offset: s64, steady time point: s64, clock source: [u8; 16]}.
    /// aud:u and the audio managers it returns. No audio backend exists:
    /// manager methods stay unsupported until a run reaches them.
    fn audio(&mut self, object: &ServiceObject, call: &mut Call<'_>) -> Result<Outcome, Trap> {
        let command = call.request.command_id;
        match (object, command) {
            (ServiceObject::AudioManager, 0..=2) => {
                call.shape(0, false, false)?;
                let manager = match command {
                    0 => ServiceObject::AudioInManager,
                    1 => ServiceObject::AudioOutManager,
                    _ => ServiceObject::AudioRendererManager,
                };
                self.reply_new_object(call, manager, AOC_INTERFACE_ID)
            }
            // IAudioRendererManager GetWorkBufferSize (1): the 0x34-byte
            // AudioRendererParameterInternal (Eden-NS2 1a6d7d4
            // audio_renderer_parameter.h; revision tag at 0x30) -> u64.
            // Tags are "REV" plus '0' + version; OoT sends "REv@", version
            // 16 with a lowercase 'v' (observed), so either case is taken.
            // The size is the transfer memory OpenAudioRenderer receives.
            // The host renderer keeps its state in host memory, so one page
            // is enough (Eden sizes Nintendo's internal structures instead).
            (ServiceObject::AudioRendererManager, 1) => {
                call.shape(0x34, false, false)?;
                let revision = crate::bytes::le_u32(&call.request.arguments[0x30..0x34]);
                let [r, e, v, version] = revision.to_le_bytes();
                if (r, e) != (b'R', b'E') || !matches!(v, b'V' | b'v') || version < b'0' {
                    return Err(call.unsupported(
                        object.service(),
                        format!("renderer revision {revision:#x} is not a REV tag"),
                    ));
                }
                call.reply(
                    0,
                    None,
                    AOC_INTERFACE_ID,
                    &AUDIO_RENDERER_WORK_BUFFER_SIZE.to_le_bytes(),
                )?;
                Ok(Outcome::success())
            }
            // IAudioRendererManager GetAudioDeviceServiceWithRevisionInfo (4):
            // u32 revision magic ("REV" plus a version byte, padded) and the
            // u64 applet resource user ID, without a PID descriptor (observed
            // from the guest; yuzu audio_renderer_manager.cpp names the fields).
            (ServiceObject::AudioRendererManager, 4) => {
                call.shape(16, false, false)?;
                let revision = crate::bytes::le_u32(&call.request.arguments[..4]);
                self.reply_new_object(
                    call,
                    ServiceObject::AudioDevice { revision },
                    AOC_INTERFACE_ID,
                )
            }
            // IAudioDevice ListAudioOutputDeviceName (14), as Eden-NS2
            // 1a6d7d4 audio_device.cpp: the three output devices' 0x100-byte
            // names into a mapped output buffer, and the s32 count.
            (ServiceObject::AudioDevice { .. }, 14) => {
                call.shape_output(0, false)?;
                let (address, size) = call.output_buffer()?;
                let names = [
                    "AudioBuiltInSpeakerOutput",
                    "AudioTvOutput",
                    "AudioExternalOutput",
                ];
                let count = names.len().min((size / 0x100) as usize);
                for (index, name) in names[..count].iter().enumerate() {
                    let mut entry = [0u8; 0x100];
                    entry[..name.len()].copy_from_slice(name.as_bytes());
                    call.memory.write(address + index as u64 * 0x100, &entry)?;
                }
                call.reply(0, None, AOC_INTERFACE_ID, &(count as u32).to_le_bytes())?;
                Ok(Outcome::success())
            }
            // Set/GetAudioDeviceOutputVolume (1/2, mapped buffer; 7/8
            // AutoSelect): the device name in the input buffer, with an f32
            // volume for Set. Volumes are kept per name (1.0 until set);
            // the host output does not apply them.
            (ServiceObject::AudioDevice { .. }, 1 | 2 | 7 | 8) => {
                let set = matches!(command, 1 | 7);
                call.shape(if set { 4 } else { 0 }, false, true)?;
                let (address, size) = call
                    .request
                    .input_buffer
                    .as_ref()
                    .map(|buffer| (buffer.address, buffer.size))
                    .ok_or_else(|| invalid("device volume without a name"))?;
                let mut name = vec![0u8; size.min(0x100) as usize];
                call.memory.read(address, &mut name)?;
                let name = String::from_utf8_lossy(
                    &name[..name.iter().position(|&b| b == 0).unwrap_or(name.len())],
                )
                .into_owned();
                if set {
                    let volume = crate::bytes::le_u32(&call.request.arguments[..4]);
                    self.audio_device_volumes.insert(name, volume);
                    call.reply(0, None, AOC_INTERFACE_ID, &[])?;
                } else {
                    let volume = self
                        .audio_device_volumes
                        .get(&name)
                        .copied()
                        .unwrap_or(1.0f32.to_bits());
                    call.reply(0, None, AOC_INTERFACE_ID, &volume.to_le_bytes())?;
                }
                Ok(Outcome::success())
            }
            // GetActiveChannelCount (5) -> u32: the host output's channels
            // (Eden-NS2 reports its sink's), stereo.
            (ServiceObject::AudioDevice { .. }, 5) => {
                call.shape(0, false, false)?;
                call.reply(0, None, AOC_INTERFACE_ID, &2u32.to_le_bytes())?;
                Ok(Outcome::success())
            }
            // ReleaseAudioOutputDeviceNotification (18, [17.0.0+]): the u64
            // from Acquire; nothing to release.
            (ServiceObject::AudioDevice { .. }, 18) => {
                call.shape(8, false, false)?;
                call.reply(0, None, AOC_INTERFACE_ID, &[])?;
                Ok(Outcome::success())
            }
            // IAudioDevice SetAudioDeviceOutputVolumeAutoTuneEnabled (19,
            // bool) and IsAudioDeviceOutputVolumeAutoTuneEnabled (20): the
            // setting is kept and reported (Eden-NS2 reports false). The
            // host applies no automatic volume tuning either way.
            (ServiceObject::AudioDevice { .. }, 19) => {
                // The bool arrives padded to a word (observed).
                call.shape(4, false, false)?;
                self.audio_volume_auto_tune = call.request.arguments[0] != 0;
                call.reply(0, None, AOC_INTERFACE_ID, &[])?;
                Ok(Outcome::success())
            }
            (ServiceObject::AudioDevice { .. }, 20) => {
                call.shape(0, false, false)?;
                let enabled = u32::from(self.audio_volume_auto_tune);
                call.reply(0, None, AOC_INTERFACE_ID, &enabled.to_le_bytes())?;
                Ok(Outcome::success())
            }
            // IAudioDevice QueryAudioDeviceOutputEvent (12): copies the
            // readable side of one application-owned event; repeated calls
            // alias it.
            // AcquireAudioOutputDeviceNotification (17, [17.0.0+]): takes a
            // u64 and returns a handle (Switchbrew Audio_services). The
            // output device never changes here, so it is the same
            // never-signaled event.
            (ServiceObject::AudioDevice { .. }, 12 | 17) => {
                call.shape(if command == 17 { 8 } else { 0 }, false, false)?;
                let application = self
                    .main_application
                    .clone()
                    .ok_or_else(|| invalid("aud:u requires the bootstrap main application"))?;
                let id = match self.audio_device_output_event {
                    Some(id) => id,
                    None => {
                        if self.readable_events.len() >= MAX_READABLE_EVENTS {
                            return Err(invalid("readable event backing capacity exhausted"));
                        }
                        ReadableEventId(
                            self.readable_events
                                .keys()
                                .map(|id| id.0)
                                .max()
                                .unwrap_or(0)
                                + 1,
                        )
                    }
                };
                let handle = self.next_available_handle()?;
                // A copied handle, as for the audio-out buffer event.
                let reply = call.prepare_copy_handle_reply(handle, 0)?;
                call.write_reply(&reply)?;
                self.readable_events.entry(id).or_insert(ReadableEvent {
                    owner: application,
                    kind: ReadableEventKind::AudioDeviceOutput,
                    signaled: false,
                });
                self.audio_device_output_event = Some(id);
                self.readable_event_handles.insert(handle, id);
                self.next_handle = handle + 1;
                Ok(Outcome {
                    output_copy_handle_count: 1,
                    ..Outcome::success()
                })
            }
            // IAudioOutManager OpenAudioOut (1), from yuzu audio_out_manager.cpp
            // and the observed request: AudioOutParameter {u32 sample_rate,
            // u16 channel_count, u16 reserved} and the u64 applet resource
            // user ID; the device name in an A buffer, its echo in a B buffer,
            // the PID and the process handle. Replies AudioOutParameterInternal
            // {u32 sample_rate, u32 channel_count, u32 sample_format (2,
            // PCM16), u32 state (1, stopped)} and the IAudioOut object. Zero
            // rate or channels select the 48 kHz stereo default.
            (ServiceObject::AudioOutManager, 1) => {
                let request = &call.request;
                if request.arguments.len() != 16
                    || request.pid.is_none()
                    || request.input_buffer.is_none()
                    || request.copy_handles.len() != 1
                    || request.receive_buffers.len() != 1
                {
                    return Err(call.unsupported(object.service(), "unexpected OpenAudioOut shape"));
                }
                let arguments = &request.arguments;
                let sample_rate = match crate::bytes::le_u32(&arguments[..4]) {
                    0 => 48_000,
                    rate => rate,
                };
                let channels = match crate::bytes::le_u16(&arguments[4..6]) {
                    0 => 2,
                    count => u32::from(count),
                };
                if sample_rate != 48_000 || !matches!(channels, 1 | 2 | 6) {
                    return Err(call.unsupported(object.service(), "unsupported audio-out format"));
                }
                let input = request
                    .input_buffer
                    .clone()
                    .expect("input buffer presence checked by the request shape");
                let (out_address, out_size, _) = request.receive_buffers[0];
                let mut name = vec![0u8; input.size.min(0x100) as usize];
                call.memory.read(input.address, &mut name)?;
                if name.iter().all(|byte| *byte == 0) {
                    name = b"DeviceOut".to_vec();
                }
                name.resize(out_size.min(0x100) as usize, 0);
                call.memory.write(out_address, &name)?;
                let mut reply = Vec::with_capacity(16);
                for value in [sample_rate, channels, 2, 1] {
                    reply.extend_from_slice(&value.to_le_bytes());
                }
                if self.audio_outs.len() >= 8 {
                    return Err(invalid("audio output capacity exhausted"));
                }
                let id = self.audio_outs.len();
                self.audio_outs.push(AudioOutState {
                    sample_rate,
                    channels,
                    started: false,
                    volume: 1.0,
                    queue: Default::default(),
                    released: Default::default(),
                    played_samples: 0,
                    front_end: None,
                    event: None,
                    host: crate::audio::open(sample_rate, channels),
                });
                self.reply_new_object_with(
                    call,
                    ServiceObject::AudioOut { id },
                    AOC_INTERFACE_ID,
                    &reply,
                )
            }
            (ServiceObject::AudioOut { id }, _) => self.audio_out(*id, call),
            // auddmg:u command 0: "Takes a PID, an u32, an u64 pid_reserved,
            // returns an output IUserSession" (Switchbrew Audio_services).
            // OoT sends 24 bytes, a further u64 before the reserved one
            // (observed: u32 2, u64 1, u64 0); none affects the session.
            // IUserSession command 0 is devmgr ListVirtualDeviceNames
            // (client stub sdk+0x2ade20, reached from ListVirtualInputDevices):
            // u8 flag and u32 device functionality, names (0x100 bytes each)
            // into an output buffer, s32 count. The host offers no virtual
            // devices through this interface, so the count is 0.
            (ServiceObject::AudioDeviceSession, 0) => {
                call.shape_output(8, false)?;
                call.output_buffer()?;
                let functionality = crate::bytes::le_u32(&call.request.arguments[4..8]);
                self.audio_device_list_requests.insert(functionality);
                call.reply(0, None, AOC_INTERFACE_ID, &0u32.to_le_bytes())?;
                self.provisional_commands.insert("auddmg:u:session:0");
                Ok(Outcome::success())
            }
            (ServiceObject::AudioDeviceSessionManager, 0) => {
                call.shape(24, true, false)?;
                self.reply_new_object(call, ServiceObject::AudioDeviceSession, AOC_INTERFACE_ID)
            }
            // IAudioRendererManager OpenAudioRenderer (0), as Eden-NS2
            // 1a6d7d4 audio_renderer_manager.cpp: the 0x34-byte parameters
            // (padded to 0x38), u64 transfer memory size and the applet
            // resource user ID, with the PID and two copied handles
            // (transfer memory, process).
            (ServiceObject::AudioRendererManager, 0) => {
                let request = &call.request;
                if request.arguments.len() < 0x48
                    || request.pid.is_none()
                    || request.copy_handles.len() != 2
                {
                    return Err(call.unsupported(
                        object.service(),
                        format!("OpenAudioRenderer shape ({})", request.describe()),
                    ));
                }
                let parameters =
                    crate::audio_renderer::Parameters::parse(&request.arguments[..0x34])
                        .map_err(|message| call.unsupported(object.service(), message))?;
                let work_size = crate::bytes::le_u64(&request.arguments[0x38..0x40]);
                if work_size < AUDIO_RENDERER_WORK_BUFFER_SIZE {
                    return Err(call.unsupported(
                        object.service(),
                        format!("renderer work buffer of {work_size:#x} bytes"),
                    ));
                }
                if self.audio_renderers.len() >= 2 {
                    return Err(invalid("audio renderer capacity exhausted"));
                }
                let id = self.audio_renderers.len();
                self.audio_renderers
                    .push((crate::audio_renderer::Renderer::new(parameters), None));
                self.reply_new_object(call, ServiceObject::AudioRenderer { id }, AOC_INTERFACE_ID)
            }
            (ServiceObject::AudioRenderer { id }, _) => self.audio_renderer(*id, call),
            (ServiceObject::OpusDecoderManager, 0..=8) => self.opus_manager(call),
            (
                ServiceObject::OpusDecoder {
                    sample_rate,
                    channels,
                },
                _,
            ) => self.opus_decode(*sample_rate, *channels, call),
            // IHardwareOpusDecoderManager GetWorkBufferSizeForMultiStreamExEx
            // (9, Switchbrew numbering): OpusMultiStreamParametersEx {u32
            // sample_rate, u32 channel_count, u32 total_stream_count, u32
            // stereo_stream_count, u8 mappings[256], ...} in an X buffer;
            // replies u32 size. Decoding keeps no state in the work buffer,
            // so the size is a local value: 64 KiB per stream plus 4 KiB.
            (ServiceObject::OpusDecoderManager, 9) => {
                call.shape(0, false, true)?;
                let input = call
                    .request
                    .input_buffer
                    .clone()
                    .expect("input buffer presence checked by the request shape");
                if input.size < 16 {
                    return Err(call.unsupported(object.service(), "short Opus parameters"));
                }
                let mut parameters = [0u8; 16];
                call.memory.read(input.address, &mut parameters)?;
                let streams = crate::bytes::le_u32(&parameters[8..12]);
                if streams == 0 || streams > 255 {
                    return Err(call.unsupported(object.service(), "invalid Opus stream count"));
                }
                let size = 0x1000 + streams * 0x10000;
                call.reply(0, None, AOC_INTERFACE_ID, &size.to_le_bytes())?;
                Ok(Outcome::success())
            }
            // IAudioDevice GetActiveAudioOutputDeviceName (13): one 0x100-byte
            // AudioDeviceName in a mapped output buffer (yuzu audio_device.cpp).
            // The fixed docked output is reported as "AudioTvOutput".
            (ServiceObject::AudioDevice { .. }, 13) => {
                call.shape_output(0, false)?;
                let (address, size) = call.output_buffer()?;
                let mut name = [0u8; 0x100];
                name[..13].copy_from_slice(b"AudioTvOutput");
                call.memory
                    .write(address, &name[..size.min(0x100) as usize])?;
                call.reply(0, None, AOC_INTERFACE_ID, &[])?;
                Ok(Outcome::success())
            }
            _ => Err(call.unsupported(object.service(), "audio command is not implemented")),
        }
    }

    /// A copied handle to `slot`'s event (created on first use, never
    /// auto-cleared), signaled now when `signaled`.
    fn reply_applet_event(
        &mut self,
        call: &mut Call<'_>,
        slot: Option<ReadableEventId>,
        signaled: bool,
    ) -> Result<(ReadableEventId, Outcome), Trap> {
        let application = self
            .main_application
            .clone()
            .ok_or_else(|| invalid("applet events require the main application"))?;
        let event = match slot {
            Some(event) => event,
            None => {
                if self.readable_events.len() >= MAX_READABLE_EVENTS {
                    return Err(invalid("readable event backing capacity exhausted"));
                }
                ReadableEventId(
                    self.readable_events
                        .keys()
                        .map(|id| id.0)
                        .max()
                        .unwrap_or(0)
                        + 1,
                )
            }
        };
        let handle = self.next_available_handle()?;
        let reply = call.prepare_copy_handle_reply(handle, 0)?;
        call.write_reply(&reply)?;
        let entry = self.readable_events.entry(event).or_insert(ReadableEvent {
            owner: application,
            kind: ReadableEventKind::LibraryApplet,
            signaled: false,
        });
        entry.signaled |= signaled;
        self.readable_event_handles.insert(handle, event);
        self.next_handle = handle + 1;
        Ok((
            event,
            Outcome {
                output_copy_handle_count: 1,
                ..Outcome::success()
            },
        ))
    }

    /// Library applets (Switchbrew "Applet Manager services"): the creator,
    /// ILibraryAppletAccessor, IStorage and IStorageAccessor. Applets run
    /// on the host: the controller applet completes as soon as it starts,
    /// reporting the one connected controller; others stop when started.
    fn library_applet(
        &mut self,
        object: &ServiceObject,
        call: &mut Call<'_>,
    ) -> Result<Outcome, Trap> {
        let command = call.request.command_id;
        let service = object.service();
        let reply_u64 =
            |call: &mut Call<'_>, value: u64| call.reply(0, None, 0, &value.to_le_bytes());
        match (object, command) {
            // CreateLibraryAppletOld (0: u32 AppletId, u32 mode) and
            // CreateLibraryApplet (3, [20.0.0+]: the same and a u64 thread
            // id) -> ILibraryAppletAccessor.
            (ServiceObject::LibraryAppletCreator { .. }, 0 | 3) => {
                call.shape(if command == 0 { 8 } else { 16 }, false, false)?;
                let arguments = &call.request.arguments;
                let applet = LibraryApplet {
                    applet_id: crate::bytes::le_u32(&arguments[..4]),
                    mode: crate::bytes::le_u32(&arguments[4..8]),
                    ..LibraryApplet::default()
                };
                if self.library_applets.len() >= 64 {
                    return Err(invalid("library applet capacity exhausted"));
                }
                let id = self.library_applets.len();
                self.library_applets.push(applet);
                self.reply_new_object(call, ServiceObject::LibraryAppletAccessor { id }, 0)
            }
            // CreateStorage (10): s64 size -> IStorage of that many zeros.
            (ServiceObject::LibraryAppletCreator { .. }, 10) => {
                call.shape(8, false, false)?;
                let size = crate::bytes::le_u64(&call.request.arguments[..8]);
                if size > 0x100_0000 || self.applet_storages.len() >= 4096 {
                    return Err(call.unsupported(service, format!("storage of {size:#x} bytes")));
                }
                let id = self.applet_storages.len();
                self.applet_storages.push(vec![0; size as usize]);
                self.reply_new_object(call, ServiceObject::LibraryStorage { id }, 0)
            }
            // IStorage Open (0) -> IStorageAccessor.
            (ServiceObject::LibraryStorage { id }, 0) => {
                call.shape(0, false, false)?;
                let id = *id;
                self.reply_new_object(call, ServiceObject::LibraryStorageAccessor { id }, 0)
            }
            // IStorageAccessor GetSize (0) -> s64; Write (10: s64 offset,
            // input buffer); Read (11: s64 offset, output buffer).
            (ServiceObject::LibraryStorageAccessor { id }, 0) => {
                call.shape(0, false, false)?;
                let size = self.applet_storages[*id].len() as u64;
                reply_u64(call, size)?;
                Ok(Outcome::success())
            }
            (ServiceObject::LibraryStorageAccessor { id }, 10 | 11) => {
                if call.request.arguments.len() != 8 {
                    return Err(call.unsupported(
                        service,
                        format!("storage access ({})", call.request.describe()),
                    ));
                }
                let offset = crate::bytes::le_u64(&call.request.arguments[..8]) as usize;
                let (address, size) = if command == 10 {
                    call.request
                        .input_buffer
                        .as_ref()
                        .map(|buffer| (buffer.address, buffer.size))
                        .ok_or_else(|| invalid("storage Write without data"))?
                } else {
                    call.output_buffer()?
                };
                let storage = &mut self.applet_storages[*id];
                let end = offset
                    .checked_add(size as usize)
                    .filter(|end| *end <= storage.len());
                let Some(end) = end else {
                    // ResultObjectInvalid-style range error: am result 503.
                    call.reply(0x3e880, None, 0, &[])?;
                    return Ok(Outcome::result(0x3e880));
                };
                if command == 10 {
                    call.memory.read(address, &mut storage[offset..end])?;
                } else {
                    call.memory.write(address, &storage[offset..end])?;
                }
                call.reply(0, None, 0, &[])?;
                Ok(Outcome::success())
            }
            // GetAppletStateChangedEvent (0): signaled once the applet ends.
            (ServiceObject::LibraryAppletAccessor { id }, 0) => {
                call.shape(0, false, false)?;
                let id = *id;
                let applet = &self.library_applets[id];
                let (event, outcome) =
                    self.reply_applet_event(call, applet.state_event, applet.completed)?;
                self.library_applets[id].state_event = Some(event);
                Ok(outcome)
            }
            // 170 ([22.0.0+], unnamed): no input; a copied event the SDK
            // wraps as a system event next to the state-changed one when it
            // creates the applet handle (sdk+0x1e3954). What signals it is
            // not known; it is never signaled here. Provisional.
            (ServiceObject::LibraryAppletAccessor { id }, 170) => {
                call.shape(0, false, false)?;
                let id = *id;
                let slot = self.library_applets[id].event_170;
                let (event, outcome) = self.reply_applet_event(call, slot, false)?;
                self.library_applets[id].event_170 = Some(event);
                self.provisional_commands
                    .insert("appletOE:library_applet_accessor:170");
                Ok(outcome)
            }
            // IsCompleted (1) -> u8.
            (ServiceObject::LibraryAppletAccessor { id }, 1) => {
                call.shape(0, false, false)?;
                let completed = u32::from(self.library_applets[*id].completed);
                call.reply(0, None, 0, &completed.to_le_bytes())?;
                Ok(Outcome::success())
            }
            // Start (10), and 90 ([20.0.0+], unnamed), which the SDK's
            // StartLibraryApplet sends instead with 32 bytes (sdk+0x1e3b94;
            // OoT's are zero): the applet runs to completion on the host.
            (ServiceObject::LibraryAppletAccessor { id }, 10 | 90) => {
                call.shape(if command == 90 { 32 } else { 0 }, false, false)?;
                let id = *id;
                // The low byte is the AppletId; OoT's controller applet is
                // 0x0100000c (the high byte's meaning is not known).
                let output = match self.library_applets[id].applet_id & 0xff {
                    LIBRARY_APPLET_CONTROLLER => {
                        // ControllerSupportResultInfo (yuzu applet_controller.h):
                        // s8 player count, 3 pad bytes, u32 selected npad id,
                        // u32 result. One controller, Player 1, success.
                        let mut info = vec![0u8; 0xc];
                        info[0] = 1;
                        info
                    }
                    other => {
                        return Err(call.unsupported(
                            service,
                            format!("library applet {other:#x} is not modeled"),
                        ));
                    }
                };
                let storage = self.applet_storages.len();
                self.applet_storages.push(output);
                let applet = &mut self.library_applets[id];
                applet.started = true;
                applet.completed = true;
                applet.out_data.push_back(storage);
                for event in [applet.state_event, applet.pop_out_event]
                    .into_iter()
                    .flatten()
                {
                    if let Some(event) = self.readable_events.get_mut(&event) {
                        event.signaled = true;
                    }
                }
                call.reply(0, None, 0, &[])?;
                Ok(Outcome::success())
            }
            // RequestExit (20) / Terminate (25): the applet has ended.
            // GetResult (30): success once completed. Focus and GPU slice
            // settings (50, 60) have nothing to adjust on the host.
            (ServiceObject::LibraryAppletAccessor { id }, 20 | 25 | 30 | 50 | 60) => {
                if call.request.arguments.len() > 4 {
                    return Err(call.unsupported(
                        service,
                        format!("applet command ({})", call.request.describe()),
                    ));
                }
                let applet = &mut self.library_applets[*id];
                if matches!(command, 20 | 25) {
                    applet.completed = true;
                }
                call.reply(0, None, 0, &[])?;
                Ok(Outcome::success())
            }
            // PushInData (100): an IStorage domain object of this session.
            (ServiceObject::LibraryAppletAccessor { id }, 100) => {
                let id = *id;
                let input = call
                    .request
                    .domain
                    .as_ref()
                    .and_then(|domain| domain.input_objects.first().copied());
                let storage = input.and_then(|object| {
                    match self.sessions[&call.handle].object(Some(object)) {
                        Some(ServiceObject::LibraryStorage { id }) => Some(*id),
                        _ => None,
                    }
                });
                let Some(storage) = storage else {
                    return Err(call.unsupported(
                        service,
                        format!("PushInData ({})", call.request.describe()),
                    ));
                };
                self.library_applets[id].in_data.push(storage);
                call.reply(0, None, 0, &[])?;
                Ok(Outcome::success())
            }
            // PopOutData (101) -> IStorage, or am ResultNoDataInChannel
            // (module 128, description 2) when empty.
            (ServiceObject::LibraryAppletAccessor { id }, 101) => {
                call.shape(0, false, false)?;
                match self.library_applets[*id].out_data.pop_front() {
                    Some(storage) => self.reply_new_object(
                        call,
                        ServiceObject::LibraryStorage { id: storage },
                        0,
                    ),
                    None => {
                        call.reply(0x480, None, 0, &[])?;
                        Ok(Outcome::result(0x480))
                    }
                }
            }
            // GetPopOutDataEvent (105): signaled while output is pending.
            (ServiceObject::LibraryAppletAccessor { id }, 105) => {
                call.shape(0, false, false)?;
                let id = *id;
                let applet = &self.library_applets[id];
                let pending = !applet.out_data.is_empty();
                let (event, outcome) =
                    self.reply_applet_event(call, applet.pop_out_event, pending)?;
                self.library_applets[id].pop_out_event = Some(event);
                Ok(outcome)
            }
            _ => Err(call.unsupported(
                service,
                format!("library applet command ({})", call.request.describe()),
            )),
        }
    }

    /// nfp:user (yuzu/Eden numbering) with no NFC devices.
    fn nfp(&mut self, object: &ServiceObject, call: &mut Call<'_>) -> Result<Outcome, Trap> {
        const SERVICE: &str = "nfp:user";
        match (object, call.request.command_id) {
            // IUserManager CreateUserInterface (0) -> IUser.
            (ServiceObject::NfpUserManager, 0) => {
                call.shape(0, false, false)?;
                self.reply_new_object(
                    call,
                    ServiceObject::NfpUser { initialized: false },
                    AOC_INTERFACE_ID,
                )
            }
            // IUser Initialize (0): u64 applet resource user ID, u64 unused,
            // the PID and an input buffer of MCU versions; Finalize (1).
            (ServiceObject::NfpUser { .. }, 0 | 1) => {
                let initialize = call.request.command_id == 0;
                if initialize && (call.request.arguments.len() != 16 || call.request.pid.is_none())
                {
                    return Err(call.unsupported(
                        SERVICE,
                        format!("Initialize shape ({})", call.request.describe()),
                    ));
                }
                let mut updated = self.sessions[&call.handle].clone();
                if let Some(ServiceObject::NfpUser { initialized }) =
                    updated.object_mut(call.request.domain.as_ref().map(|domain| domain.object_id))
                {
                    *initialized = initialize;
                }
                self.sessions.insert(call.handle, updated);
                call.reply(0, None, AOC_INTERFACE_ID, &[])?;
                Ok(Outcome::success())
            }
            // AttachAvailabilityChangeEvent (23): one copied event, never
            // signaled (NFC availability does not change on the host).
            (ServiceObject::NfpUser { .. }, 23) => {
                call.shape(0, false, false)?;
                let application = self
                    .main_application
                    .clone()
                    .ok_or_else(|| invalid("nfp requires the bootstrap main application"))?;
                let event = match self.nfp_availability_event {
                    Some(event) => event,
                    None => {
                        if self.readable_events.len() >= MAX_READABLE_EVENTS {
                            return Err(invalid("readable event backing capacity exhausted"));
                        }
                        ReadableEventId(
                            self.readable_events
                                .keys()
                                .map(|id| id.0)
                                .max()
                                .unwrap_or(0)
                                + 1,
                        )
                    }
                };
                let handle = self.next_available_handle()?;
                let reply = call.prepare_copy_handle_reply(handle, 0)?;
                call.write_reply(&reply)?;
                self.readable_events.entry(event).or_insert(ReadableEvent {
                    owner: application,
                    kind: ReadableEventKind::NfpAvailability,
                    signaled: false,
                });
                self.nfp_availability_event = Some(event);
                self.readable_event_handles.insert(handle, event);
                self.next_handle = handle + 1;
                Ok(Outcome {
                    output_copy_handle_count: 1,
                    ..Outcome::success()
                })
            }
            // GetState (19) -> u32: 0 not initialized, 1 initialized.
            (ServiceObject::NfpUser { initialized }, 19) => {
                call.shape(0, false, false)?;
                call.reply(
                    0,
                    None,
                    AOC_INTERFACE_ID,
                    &u32::from(*initialized).to_le_bytes(),
                )?;
                Ok(Outcome::success())
            }
            _ => Err(call.unsupported(SERVICE, "nfp command is not implemented")),
        }
    }

    /// A transport result for the last request (instead of success), once.
    pub(crate) fn take_transport_result(&mut self) -> Option<u32> {
        self.transport_result.take()
    }

    /// Pair output slots handed to the decoder with the oldest waiting
    /// submissions: each pair is a decoded frame (slot, timestamp) for
    /// GetFrame. No pixels are produced yet. The input event is signaled
    /// while fewer than VIDEO_INPUT_DEPTH submissions wait, the decoder's
    /// back-pressure on SubmitAv1DecoderInputBuffer.
    fn decode_video_frames(&mut self) {
        while !self.decoder_slots.is_empty() && !self.video_input.is_empty() {
            let slot = self.decoder_slots.pop_front().unwrap_or_default();
            let (_, frame) = self.video_input.remove(0);
            self.decoded_video_frames.push_back((slot, frame));
            self.decoded_frames += 1;
            *self
                .decoder_fences
                .entry(decoder_fence_token(slot))
                .or_insert(0) += 1;
        }
        let reached: Vec<u32> = self
            .decoder_fence_waits
            .iter()
            .filter(|(_, (token, value))| self.decoder_fence_reached(*token, *value))
            .map(|(&index, _)| index)
            .collect();
        for index in reached {
            self.decoder_fence_waits.remove(&index);
            if let Some(event) = self
                .syncpoint_events
                .get(&index)
                .and_then(|event| self.readable_events.get_mut(event))
            {
                event.signaled = true;
            }
        }
        let accepting = self.video_input.len() < VIDEO_INPUT_DEPTH;
        if let Some(event) = self
            .video_input_event
            .and_then(|id| self.readable_events.get_mut(&id))
        {
            event.signaled = accepting;
        }
    }

    /// Add to the video decoder log, keeping the last 600 entries.
    fn note_video(&mut self, entry: String) {
        if self.video_log.len() == 600 {
            self.video_log.pop_front();
        }
        self.video_log.push_back(entry);
    }

    /// Whether decoder fence `token` (or its complement, as some SDK paths
    /// pass it) has reached `value`; tokens that are not decoder fences
    /// always have (host GPU work completes at submission).
    fn decoder_fence_reached(&self, token: u32, value: u64) -> bool {
        self.decoder_fence_value(token)
            .is_none_or(|current| current >= value)
    }

    fn decoder_fence_value(&self, token: u32) -> Option<u64> {
        let value = self
            .decoder_fences
            .get(&token)
            .or_else(|| self.decoder_fences.get(&!token))
            .copied()?;
        Some(if self.decoder_fences_released {
            u64::MAX >> 1
        } else {
            value
        })
    }

    /// Stop or Finalize: release every decoder fence and signal the events
    /// armed on them.
    fn release_decoder_fences(&mut self) {
        self.decoder_fences_released = true;
        for (index, _) in std::mem::take(&mut self.decoder_fence_waits) {
            if let Some(event) = self
                .syncpoint_events
                .get(&index)
                .and_then(|event| self.readable_events.get_mut(event))
            {
                event.signaled = true;
            }
        }
    }

    /// Decode one submission with the session's host decoder (created on
    /// first use): the last picture it completes, if any. Failures are
    /// recorded (the first 16) and give no picture.
    fn decode_av1(&mut self, bytes: &[u8]) -> Option<Box<crate::av1::Picture>> {
        let result = match &mut self.av1_decoder {
            Some(decoder) => decoder.decode(bytes),
            None => crate::av1::Decoder::new().and_then(|mut decoder| {
                let result = decoder.decode(bytes);
                self.av1_decoder = Some(decoder);
                result
            }),
        };
        if let Ok(pictures) = &result {
            // ECHORA_MOVIE_RATE: decoded movie pictures per host second, a
            // speed measure tied to content rather than to presents.
            if !pictures.is_empty() && crate::nvn::debug_var_os("ECHORA_MOVIE_RATE").is_some() {
                let (count, first, boot) = &mut self.av1_pictures;
                let first = *first.get_or_insert_with(std::time::Instant::now);
                *count += pictures.len() as u64;
                if *count % 60 == 0 {
                    let seconds = first.elapsed().as_secs_f64();
                    eprintln!(
                        "movie: {count} pictures in {seconds:.2} s ({:.1}/s), {:.2} s since boot",
                        *count as f64 / seconds.max(1e-9),
                        boot.elapsed().as_secs_f64()
                    );
                }
            }
        }
        match result {
            Ok(mut pictures) => pictures.pop().map(Box::new),
            Err(error) => {
                if self.av1_errors.len() < 16 {
                    self.av1_errors.push(error);
                }
                None
            }
        }
    }

    /// The internal handle of a request that parked its thread, once.
    pub(crate) fn take_blocked_request(&mut self) -> Option<u32> {
        self.blocked_request.take()
    }

    /// Finish parked polls that now have a ready descriptor: write their
    /// pollfds and reply, then signal their internal event.
    fn complete_ready_polls(&mut self, memory: &mut Memory) -> Result<(), Trap> {
        let handles: Vec<u32> = self.pending_polls.keys().copied().collect();
        for handle in handles {
            let mut poll = self.pending_polls[&handle].clone();
            let ready = poll_event_fds(&self.socket_event_fds, &mut poll.entries);
            if ready == 0 {
                continue;
            }
            if let Some((address, size)) = poll.output {
                let length = poll.entries.len().min(size as usize);
                memory.write(address, &poll.entries[..length])?;
            }
            poll.reply[poll.data_offset..poll.data_offset + 4]
                .copy_from_slice(&ready.to_le_bytes());
            memory.write(poll.tls, &poll.reply)?;
            if let Some(event) = self.readable_events.get_mut(&poll.event) {
                event.signaled = true;
            }
            self.pending_polls.remove(&handle);
        }
        Ok(())
    }

    /// nsd:u as Eden-NS2 1a6d7d4 nsd.cpp answers it: environment "lp1",
    /// FQDNs resolved to themselves (nothing is contacted offline) and
    /// application server environment Lp (1).
    fn nsd(&mut self, call: &mut Call<'_>) -> Result<Outcome, Trap> {
        const SERVICE: &str = "nsd:u";
        match call.request.command_id {
            // GetEnvironmentIdentifier: 8 bytes into the output buffer.
            11 => {
                call.shape_output(0, false)?;
                let (address, size) = call.output_buffer()?;
                if size < 8 {
                    return Err(call.unsupported(SERVICE, "environment buffer under 8 bytes"));
                }
                call.memory.write(address, b"lp1\0\0\0\0\0")?;
                call.reply(0, None, AOC_INTERFACE_ID, &[])?;
            }
            // Resolve / ResolveEx: FQDN in, the same FQDN (0x100 bytes) out;
            // ResolveEx adds a success result word.
            20 | 21 => {
                let request = &call.request;
                let input = request.input_buffer.as_ref().map(|b| (b.address, b.size));
                let Some((input_address, input_size)) = input else {
                    return Err(call.unsupported(SERVICE, "Resolve without an FQDN"));
                };
                let (address, size) = call.output_buffer()?;
                let mut fqdn = vec![0u8; input_size.min(0x100) as usize];
                call.memory.read(input_address, &mut fqdn)?;
                let length = fqdn.iter().position(|&b| b == 0).unwrap_or(fqdn.len());
                if length >= 0x100 || size < 0x100 {
                    // ResultOverflow (module NSD 141, description 6).
                    call.reply(0x0c8d, None, AOC_INTERFACE_ID, &[])?;
                    return Ok(Outcome::result(0x0c8d));
                }
                let mut out = [0u8; 0x100];
                out[..length].copy_from_slice(&fqdn[..length]);
                call.memory.write(address, &out)?;
                let reply = if call.request.command_id == 21 {
                    0u32.to_le_bytes().to_vec()
                } else {
                    Vec::new()
                };
                call.reply(0, None, AOC_INTERFACE_ID, &reply)?;
            }
            // GetApplicationServerEnvironmentType -> u32 (Lp).
            100 => {
                call.shape(0, false, false)?;
                call.reply(0, None, AOC_INTERFACE_ID, &1u32.to_le_bytes())?;
            }
            _ => return Err(call.unsupported(SERVICE, "nsd command is not implemented")),
        }
        Ok(Outcome::success())
    }

    /// bsd:u ISocketClient (Switchbrew numbering). The host offers no
    /// network: RegisterClient (0) and StartMonitoring (1) succeed, and
    /// Socket (2) returns -1 with errno ENETDOWN (100; nn::socket errno
    /// values follow Linux numbering, per yuzu's Errno enum).
    fn socket_client(&mut self, call: &mut Call<'_>) -> Result<Outcome, Trap> {
        const SERVICE: &str = "bsd:u";
        const ENETDOWN: u32 = 100;
        match call.request.command_id {
            // RegisterClient: LibraryConfigData, PID placeholder and the
            // transfer-memory size, with the PID and the transfer memory
            // (copied handle, unused here). Replies s32 0.
            0 => {
                if call.request.pid.is_none() || call.request.copy_handles.len() != 1 {
                    return Err(call.unsupported(SERVICE, "RegisterClient without PID and memory"));
                }
                call.reply(0, None, AOC_INTERFACE_ID, &0u32.to_le_bytes())?;
                Ok(Outcome::success())
            }
            // 36 [18.0.0+]: Switchbrew documents it as the Switch 2
            // RegisterClient variant (PID, u32 version, u64 PID placeholder;
            // no transfer memory). Replies s32 0.
            36 => {
                call.shape(16, true, false)?;
                call.reply(0, None, AOC_INTERFACE_ID, &0u32.to_le_bytes())?;
                Ok(Outcome::success())
            }
            // StartMonitoring: PID placeholder plus the PID descriptor.
            1 => {
                call.shape(8, true, false)?;
                call.reply(0, None, AOC_INTERFACE_ID, &[])?;
                Ok(Outcome::success())
            }
            // Socket / SocketExempt (domain, type, protocol): {s32 ret, u32 errno}.
            2 | 3 => {
                call.shape(12, false, false)?;
                let mut reply = Vec::with_capacity(8);
                reply.extend_from_slice(&(-1i32).to_le_bytes());
                reply.extend_from_slice(&ENETDOWN.to_le_bytes());
                call.reply(0, None, AOC_INTERFACE_ID, &reply)?;
                Ok(Outcome::success())
            }
            // Poll (6): s32 nfds, s32 timeout (ms), pollfd {s32 fd, s16
            // events, s16 revents} array in and out (Eden-NS2 bsd.cpp) ->
            // {s32 ready, u32 errno}. EventFds are writable (POLLOUT 4) and
            // readable (POLLIN 1) while their counter is nonzero; other
            // descriptors report POLLNVAL (0x20). The service does not
            // block: with nothing ready it returns 0 at once, whatever the
            // timeout (each timeout is recorded for the report).
            6 => {
                let request = &call.request;
                if request.arguments.len() != 8
                    || request.pid.is_some()
                    || !request.copy_handles.is_empty()
                {
                    return Err(
                        call.unsupported(SERVICE, format!("Poll shape ({})", request.describe()))
                    );
                }
                let count = crate::bytes::le_u32(&request.arguments[..4]) as i32;
                let timeout = crate::bytes::le_u32(&request.arguments[4..8]) as i32;
                let input = request
                    .input_buffer
                    .as_ref()
                    .map(|buffer| (buffer.address, buffer.size))
                    .ok_or_else(|| invalid("Poll without pollfds"))?;
                if !(0..=64).contains(&count) || (count as u64) * 8 > input.1 {
                    return Err(call.unsupported(SERVICE, format!("Poll of {count} descriptors")));
                }
                let mut entries = vec![0u8; count as usize * 8];
                call.memory.read(input.0, &mut entries)?;
                let output = if count != 0 {
                    Some(call.output_buffer()?)
                } else {
                    None
                };
                *self.socket_poll_timeouts.entry(timeout).or_default() += 1;
                let ready = poll_event_fds(&self.socket_event_fds, &mut entries);
                if ready == 0 && timeout < 0 {
                    // Nothing ready and no timeout: the calling thread parks
                    // on an internal event (no handle reaches the guest)
                    // until an EventFd write makes a descriptor ready.
                    let mut first = 1i32.to_le_bytes().to_vec();
                    first.extend_from_slice(&0u32.to_le_bytes());
                    let mut second = 2i32.to_le_bytes().to_vec();
                    second.extend_from_slice(&0u32.to_le_bytes());
                    let reply = call.prepare_reply(0, None, AOC_INTERFACE_ID, &first, &[])?;
                    let other = call.prepare_reply(0, None, AOC_INTERFACE_ID, &second, &[])?;
                    let data_offset = reply
                        .bytes
                        .iter()
                        .zip(&other.bytes)
                        .position(|(a, b)| a != b)
                        .ok_or_else(|| invalid("Poll reply has no data field"))?;
                    let application = self
                        .main_application
                        .clone()
                        .ok_or_else(|| invalid("bsd:u requires the bootstrap main application"))?;
                    if self.readable_events.len() >= MAX_READABLE_EVENTS {
                        return Err(invalid("readable event backing capacity exhausted"));
                    }
                    let event = ReadableEventId(
                        self.readable_events
                            .keys()
                            .map(|id| id.0)
                            .max()
                            .unwrap_or(0)
                            + 1,
                    );
                    self.readable_events.insert(
                        event,
                        ReadableEvent {
                            owner: application,
                            kind: ReadableEventKind::SocketPoll,
                            signaled: false,
                        },
                    );
                    let handle = self.next_available_handle()?;
                    self.readable_event_handles.insert(handle, event);
                    self.next_handle = handle + 1;
                    self.pending_polls.insert(
                        handle,
                        PendingPoll {
                            tls: call.buffer_address,
                            reply: reply.bytes,
                            data_offset,
                            entries,
                            output,
                            event,
                        },
                    );
                    self.blocked_request = Some(handle);
                    return Ok(Outcome::success());
                }
                if let Some((address, size)) = output {
                    let length = entries.len().min(size as usize);
                    call.memory.write(address, &entries[..length])?;
                }
                let mut reply = ready.to_le_bytes().to_vec();
                reply.extend_from_slice(&0u32.to_le_bytes());
                call.reply(0, None, AOC_INTERFACE_ID, &reply)?;
                Ok(Outcome::success())
            }
            // Read (25): s32 fd, the data into an output buffer -> {s32
            // ret, u32 errno}. An EventFd yields its 8-byte counter and
            // resets it; at zero it fails with EAGAIN (11) instead of
            // blocking.
            25 => {
                if call.request.arguments.len() != 4 || call.request.pid.is_some() {
                    return Err(call.unsupported(
                        SERVICE,
                        format!("Read shape ({})", call.request.describe()),
                    ));
                }
                let fd = crate::bytes::le_u32(&call.request.arguments[..4]) as i32;
                let (address, size) = call.output_buffer()?;
                let (ret, errno) = match self.socket_event_fds.get_mut(&fd) {
                    Some(_) if size < 8 => (-1i32, 22u32),
                    Some(0) => (-1, 11),
                    Some(counter) => {
                        call.memory.write(address, &counter.to_le_bytes())?;
                        *counter = 0;
                        (8, 0)
                    }
                    None => (-1, 9),
                };
                let mut reply = ret.to_le_bytes().to_vec();
                reply.extend_from_slice(&errno.to_le_bytes());
                call.reply(0, None, AOC_INTERFACE_ID, &reply)?;
                Ok(Outcome::success())
            }
            // Write (24): s32 fd, the data in an input buffer -> {s32 ret,
            // u32 errno}. For an EventFd the 8-byte value adds to its
            // counter (Linux eventfd semantics), completing parked polls.
            24 => {
                if call.request.arguments.len() != 4 || call.request.pid.is_some() {
                    return Err(call.unsupported(
                        SERVICE,
                        format!("Write shape ({})", call.request.describe()),
                    ));
                }
                let fd = crate::bytes::le_u32(&call.request.arguments[..4]) as i32;
                let input = call
                    .request
                    .input_buffer
                    .as_ref()
                    .map(|buffer| (buffer.address, buffer.size))
                    .ok_or_else(|| invalid("Write without data"))?;
                let (ret, errno) = match self.socket_event_fds.get_mut(&fd) {
                    Some(counter) if input.1 >= 8 => {
                        let value = call.memory.read_u64(input.0)?;
                        *counter = counter.saturating_add(value);
                        (8i32, 0u32)
                    }
                    // EINVAL (22) for a short eventfd write.
                    Some(_) => (-1, 22),
                    None => (-1, 9),
                };
                let mut reply = ret.to_le_bytes().to_vec();
                reply.extend_from_slice(&errno.to_le_bytes());
                call.reply(0, None, AOC_INTERFACE_ID, &reply)?;
                self.complete_ready_polls(call.memory)?;
                Ok(Outcome::success())
            }
            // Close (26): s32 fd -> {s32 ret, u32 errno}. Only EventFd
            // descriptors exist; others fail with EBADF (9).
            26 => {
                call.shape(4, false, false)?;
                let fd = crate::bytes::le_u32(&call.request.arguments[..4]) as i32;
                let (ret, errno) = match self.socket_event_fds.remove(&fd) {
                    Some(_) => (0i32, 0u32),
                    None => (-1, 9),
                };
                let mut reply = ret.to_le_bytes().to_vec();
                reply.extend_from_slice(&errno.to_le_bytes());
                call.reply(0, None, AOC_INTERFACE_ID, &reply)?;
                Ok(Outcome::success())
            }
            // EventFd (31, [7.0.0+]: u64 initial value, u32 flags (Eden-NS2
            // bsd.cpp) -> {s32 fd, u32 errno}. A local counter, not a
            // socket: it gets a descriptor holding its value. Descriptors
            // start at 3, after the standard streams.
            31 => {
                call.shape(16, false, false)?;
                let initial = crate::bytes::le_u64(&call.request.arguments[..8]);
                let fd = self
                    .socket_event_fds
                    .keys()
                    .next_back()
                    .map_or(3, |fd| fd + 1);
                self.socket_event_fds.insert(fd, initial);
                let mut reply = fd.to_le_bytes().to_vec();
                reply.extend_from_slice(&0u32.to_le_bytes());
                call.reply(0, None, AOC_INTERFACE_ID, &reply)?;
                Ok(Outcome::success())
            }
            _ => Err(call.unsupported(SERVICE, "socket command is not implemented")),
        }
    }

    /// A copied handle to a shared memory, created on first use and kept
    /// in `slot`, with optional reply data.
    fn reply_shared_memory(
        &mut self,
        call: &mut Call<'_>,
        slot: fn(&mut Self) -> &mut Option<u32>,
        object: SharedMemoryObject,
        data: &[u8],
        interface_id: u32,
    ) -> Result<Outcome, Trap> {
        let id = match *slot(self) {
            Some(id) => id,
            None => {
                let id = self.shared_memories.keys().max().copied().unwrap_or(0) + 1;
                self.shared_memories.insert(id, object);
                *slot(self) = Some(id);
                id
            }
        };
        let handle = self.next_available_handle()?;
        let reply = if call.request.domain.is_some() {
            ipc::build_domain_copy_handle_reply_with_data(
                call.request,
                0,
                handle,
                interface_id,
                data,
                &[],
            )
        } else {
            ipc::build_copy_handle_reply_with_data(call.request, 0, handle, interface_id, data)
        }
        .map_err(|error| invalid(format!("build shared-memory reply: {error:#}")))?;
        call.write_reply(&reply)?;
        self.shared_memory_handles.insert(handle, id);
        self.next_handle = handle + 1;
        Ok(Outcome {
            output_copy_handle_count: 1,
            ..Outcome::success()
        })
    }

    /// hid IHidServer (Switchbrew / yuzu numbering) with no controllers
    /// connected. The applet resource's 0x40000-byte shared memory stays
    /// zeroed: empty rings and no npad style sets. Setup commands that
    /// carry only scalars (or an X list of npad ids) and return nothing
    /// are accepted.
    fn hid(&mut self, object: &ServiceObject, call: &mut Call<'_>) -> Result<Outcome, Trap> {
        const NO_OUTPUT: &[u32] = &[
            1,    // ActivateDebugPad
            11,   // ActivateTouchScreen
            21,   // ActivateMouse
            31,   // ActivateKeyboard
            51,   // ActivateXpad
            56,   // ActivateJoyXpad
            66,   // StartSixAxisSensor
            67,   // StopSixAxisSensor
            69,   // EnableSixAxisSensorFusion
            70,   // SetSixAxisSensorFusionParameters
            72,   // ResetSixAxisSensorFusionParameters
            74,   // SetAccelerometerParameters
            76,   // SetAccelerometerPlayMode
            79,   // SetGyroscopeZeroDriftMode
            91,   // ActivateGesture
            102,  // SetSupportedNpadIdType
            103,  // ActivateNpad
            104,  // DeactivateNpad
            109,  // ActivateNpadWithRevision
            120,  // SetNpadJoyHoldType
            122,  // SetNpadJoyAssignmentModeSingleByDefault
            124,  // SetNpadJoyAssignmentModeDual
            128,  // SetNpadHandheldActivationMode
            134,  // SetNpadAnalogStickUseCenterClamp
            135,  // SetNpadCaptureButtonAssignment
            136,  // ClearNpadCaptureButtonAssignment
            300,  // ActivateConsoleSixAxisSensor
            301,  // StartConsoleSixAxisSensor
            302,  // StopConsoleSixAxisSensor
            213,  // Switch 2, unnamed vibration command (inert here)
            214,  // SendNativeVibrationValues
            1000, // SetNpadCommunicationMode
            1002, // SetTouchScreenConfiguration
            // Switch 2 (Eden-NS2): native input / Joy-Con mouse memory is
            // retained by the guest; no mouse samples arrive while those
            // devices are disconnected.
            3110, // RegisterJcmsMemory
            3136, // SetNmouseButtonConfigJoyLeftJcms
            3137, // SetNmouseButtonConfigJoyRightJcms
            3138, // SetMergedNmouseSourceTypes
            3139, // RegisterNativeInputMemory
            3150, // SetMouseLibraryVersion
        ];
        let command = call.request.command_id;
        match (object, command) {
            // CreateAppletResource(u64 aruid, PID).
            (ServiceObject::HidServer, 0) => {
                call.shape(8, true, false)?;
                self.reply_new_object(call, ServiceObject::HidAppletResource, AOC_INTERFACE_ID)
            }
            // GetSharedMemoryHandle (0): read-only for the application.
            (ServiceObject::HidAppletResource, 0) => {
                call.shape(0, false, false)?;
                self.reply_shared_memory(
                    call,
                    |services| &mut services.hid_memory,
                    SharedMemoryObject {
                        size: 0x40000,
                        permission: 1,
                        contents: Vec::new(),
                        label: "hid",
                    },
                    &[],
                    AOC_INTERFACE_ID,
                )
            }
            // SetSupportedNpadStyleSet (100): u32 style set, u64 applet
            // resource user ID, PID; GetSupportedNpadStyleSet (101): u64
            // applet resource user ID, PID -> u32 (0 until set).
            (ServiceObject::HidServer, 100) => {
                call.shape(16, true, false)?;
                self.hid_supported_style_set = crate::bytes::le_u32(&call.request.arguments[..4]);
                call.reply(0, None, AOC_INTERFACE_ID, &[])?;
                Ok(Outcome::success())
            }
            (ServiceObject::HidServer, 101) => {
                call.shape(8, true, false)?;
                let styles = self.hid_supported_style_set;
                call.reply(0, None, AOC_INTERFACE_ID, &styles.to_le_bytes())?;
                Ok(Outcome::success())
            }
            // DisconnectNpad (107): u32 npad id, u64 applet resource user
            // ID and the PID. The disconnected ids are recorded (reported).
            (ServiceObject::HidServer, 107) => {
                call.shape(16, true, false)?;
                let npad = crate::bytes::le_u32(&call.request.arguments[..4]);
                self.hid_disconnected_npads.insert(npad);
                call.reply(0, None, AOC_INTERFACE_ID, &[])?;
                Ok(Outcome::success())
            }
            // AcquireNpadStyleSetUpdateEventHandle(u32 npad id, u64 aruid,
            // u64 event cookie, PID) -> copied event.
            (ServiceObject::HidServer, 106) => {
                let application = self
                    .main_application
                    .clone()
                    .ok_or_else(|| invalid("hid requires the bootstrap main application"))?;
                if self.readable_events.len() >= MAX_READABLE_EVENTS {
                    return Err(invalid("readable event backing capacity exhausted"));
                }
                let id = ReadableEventId(
                    self.readable_events
                        .keys()
                        .map(|id| id.0)
                        .max()
                        .unwrap_or(0)
                        + 1,
                );
                let handle = self.next_available_handle()?;
                let reply = call.prepare_copy_handle_reply(handle, AOC_INTERFACE_ID)?;
                call.write_reply(&reply)?;
                self.readable_events.insert(
                    id,
                    ReadableEvent {
                        owner: application,
                        kind: ReadableEventKind::NpadStyleSetUpdate,
                        signaled: false,
                    },
                );
                self.readable_event_handles.insert(handle, id);
                self.next_handle = handle + 1;
                Ok(Outcome {
                    output_copy_handle_count: 1,
                    ..Outcome::success()
                })
            }
            // GetVibrationDeviceInfo(u32 handle) -> VibrationDeviceInfo
            // {u32 type, u32 position}: yuzu reports a linear resonant
            // actuator (1) on the side the handle's device index names.
            (ServiceObject::HidServer, 200) => {
                call.shape(4, false, false)?;
                let device_index = call.request.arguments[3];
                let position = if device_index == 1 { 2u32 } else { 1 };
                let mut info = Vec::with_capacity(8);
                info.extend_from_slice(&1u32.to_le_bytes());
                info.extend_from_slice(&position.to_le_bytes());
                call.reply(0, None, AOC_INTERFACE_ID, &info)?;
                Ok(Outcome::success())
            }
            (ServiceObject::HidServer, 203) => {
                call.shape(0, false, false)?;
                self.reply_new_object(
                    call,
                    ServiceObject::HidVibrationDeviceList,
                    AOC_INTERFACE_ID,
                )
            }
            // IsVibrationDeviceMounted(handle, aruid) -> bool: the modeled
            // controller has its motors (host gamepads play them).
            (ServiceObject::HidServer, 211) => {
                call.reply(0, None, AOC_INTERFACE_ID, &[1])?;
                Ok(Outcome::success())
            }
            // SendVibrationValue(u32 handle, VibrationValue, u64 aruid):
            // VibrationValue is f32 low amplitude, low frequency, high
            // amplitude, high frequency.
            (ServiceObject::HidServer, 201) => {
                let arguments = &call.request.arguments;
                if arguments.len() >= 20 {
                    let float = |at: usize| crate::bytes::le_f32(&arguments[at..at + 4]);
                    crate::settings::rumble(vibration_device(&arguments[..4]), float(4), float(12));
                }
                call.reply(0, None, AOC_INTERFACE_ID, &[])?;
                Ok(Outcome::success())
            }
            // SendVibrationValues(u64 aruid, X handles, X values): u32
            // handles and 16-byte VibrationValues, pairwise.
            (ServiceObject::HidServer, 206) => {
                if let (Some(handles), Some(values)) = (
                    call.request.input_buffer.clone(),
                    call.request.extra_pointers.first().cloned(),
                ) {
                    let count = (handles.size / 4).min(values.size / 16).min(8) as usize;
                    let mut handle_bytes = vec![0u8; count * 4];
                    let mut value_bytes = vec![0u8; count * 16];
                    call.memory.read(handles.address, &mut handle_bytes)?;
                    call.memory.read(values.address, &mut value_bytes)?;
                    for index in 0..count {
                        let value = &value_bytes[index * 16..];
                        let float = |at: usize| crate::bytes::le_f32(&value[at..at + 4]);
                        crate::settings::rumble(
                            vibration_device(&handle_bytes[index * 4..index * 4 + 4]),
                            float(0),
                            float(8),
                        );
                    }
                }
                call.reply(0, None, AOC_INTERFACE_ID, &[])?;
                Ok(Outcome::success())
            }
            // SendVibrationValueInBool(u32 handle, bool, u64 aruid): an
            // on/off motor, played as full low band.
            (ServiceObject::HidServer, 212) => {
                let arguments = &call.request.arguments;
                if arguments.len() >= 5 {
                    let on = f32::from(u8::from(arguments[4] != 0));
                    crate::settings::rumble(vibration_device(&arguments[..4]), on, 0.0);
                }
                call.reply(0, None, AOC_INTERFACE_ID, &[])?;
                Ok(Outcome::success())
            }
            // ActivateVibrationDevice(u32 handle).
            (ServiceObject::HidVibrationDeviceList, 0) => {
                call.shape(4, false, false)?;
                call.reply(0, None, AOC_INTERFACE_ID, &[])?;
                Ok(Outcome::success())
            }
            (ServiceObject::HidServer, command) if NO_OUTPUT.contains(&command) => {
                if !call.request.receive_buffers.is_empty() || !call.request.receive_list.is_empty()
                {
                    return Err(call.unsupported(object.service(), "hid setter with outputs"));
                }
                call.reply(0, None, AOC_INTERFACE_ID, &[])?;
                Ok(Outcome::success())
            }
            _ => Err(call.unsupported(object.service(), "hid command is not implemented")),
        }
    }

    /// IAsyncContext of a play report sent with 10500: the report was
    /// discarded, so the context is done from the start.
    fn play_report_context(&mut self, call: &mut Call<'_>) -> Result<Outcome, Trap> {
        match call.request.command_id {
            // GetSystemEvent (0; PlayReport's AsyncContext::GetCompletionEvent)
            // -> copied event, signaled.
            0 => {
                call.shape(0, false, false)?;
                self.reply_done_event(call)
            }
            // Cancel (1; inferred from the SDK's method order, between the
            // event and the result): nothing is in flight. GetResult (2):
            // the discarded report succeeded.
            1 | 2 => {
                call.shape(0, false, false)?;
                call.reply(0, None, AOC_INTERFACE_ID, &[])?;
                Ok(Outcome::success())
            }
            _ => Err(call.unsupported(
                "prepo:u:async_context",
                "play report async context command is not implemented",
            )),
        }
    }

    /// stpl:u (GameShare streaming). Only command 0 is documented; the
    /// shim's commands are added as the game uses them.
    /// prepo:u (Switchbrew "Play Report services"). Native-PC policy: play
    /// reports are telemetry for Nintendo's servers, so the reports are
    /// accepted and discarded; nothing is ever pending transmission.
    fn play_report(&mut self, call: &mut Call<'_>) -> Result<Outcome, Trap> {
        match call.request.command_id {
            // SaveReport* (10100..=10107; OoT sends 10107, a newer variant)
            // and SaveSystemReport* (20100, 20101): event name and
            // MessagePack buffers, optionally a user and the PID. Discarded.
            10100..=10107 | 20100 | 20101 => {
                call.reply(0, None, AOC_INTERFACE_ID, &[])?;
                Ok(Outcome::success())
            }
            // RequestImmediateTransmission: nothing to send.
            10200 => {
                call.shape(0, false, false)?;
                call.reply(0, None, AOC_INTERFACE_ID, &[])?;
                Ok(Outcome::success())
            }
            // GetTransmissionStatus -> u32 (0: idle).
            10300 => {
                call.shape(0, false, false)?;
                call.reply(0, None, AOC_INTERFACE_ID, &0u32.to_le_bytes())?;
                Ok(Outcome::success())
            }
            // GetSystemSessionId -> u64.
            10400 => {
                call.shape(0, false, false)?;
                call.reply(0, None, AOC_INTERFACE_ID, &0u64.to_le_bytes())?;
                Ok(Outcome::success())
            }
            // 10500 (SDK 22's PlayReport::Send(Uid): PID, the user and
            // event in the arguments, the report data through a copied
            // handle, the event name in a pointer buffer) -> IAsyncContext.
            // Discarded like the reports above; the context's commands are
            // added as the game uses them.
            10500 => {
                self.reply_new_object(call, ServiceObject::PlayReportContext, AOC_INTERFACE_ID)
            }
            // IsUserAgreementCheckEnabled -> bool (no agreement to check).
            10600 => {
                call.shape(0, false, false)?;
                call.reply(0, None, AOC_INTERFACE_ID, &[0])?;
                Ok(Outcome::success())
            }
            _ => Err(call.unsupported("prepo:u", "play report command is not implemented")),
        }
    }

    fn streamplay(&mut self, object: &ServiceObject, call: &mut Call<'_>) -> Result<Outcome, Trap> {
        match (object, call.request.command_id) {
            // Cmd0 (u32 stpl version, 0x12 observed) -> the shim object.
            (ServiceObject::StreamplayUser, 0) => {
                call.shape(4, false, false)?;
                self.reply_new_object(call, ServiceObject::StreamplayShim, AOC_INTERFACE_ID)
            }
            // GetRequiredMemorySize (1): u8 type, u8 mode, s32 guest count
            // -> u32. Following Eden-NS2: one page of session state for a
            // console with no GameShare guests.
            (ServiceObject::StreamplayShim, 1) => {
                call.shape(8, false, false)?;
                call.reply(0, None, AOC_INTERFACE_ID, &0x1000u32.to_le_bytes())?;
                Ok(Outcome::success())
            }
            // GetGuestHidSharedMemory (300) -> {u64 size, u64 offset} and a
            // copied shared memory (Eden-NS2): 0x84000 bytes of guest pad,
            // touch and Joy-Con mouse rings, all empty with no guests.
            (ServiceObject::StreamplayShim, 300) => {
                call.shape(0, false, false)?;
                const SIZE: u64 = 0x84000;
                let mut data = Vec::with_capacity(16);
                data.extend_from_slice(&SIZE.to_le_bytes());
                data.extend_from_slice(&0u64.to_le_bytes());
                self.reply_shared_memory(
                    call,
                    |services| &mut services.streamplay_hid_memory,
                    SharedMemoryObject {
                        size: SIZE,
                        permission: 1,
                        contents: Vec::new(),
                        label: "stpl guest hid",
                    },
                    &data,
                    AOC_INTERFACE_ID,
                )
            }
            _ => Err(call.unsupported(object.service(), "streamplay command is not implemented")),
        }
    }

    /// bcat:u (Switchbrew numbering; layouts and results from yuzu's bcat
    /// module, module 122) with an empty delivery cache. Syncs finish at
    /// once (Done, result 0), the cache has no directories, and opening any
    /// entry fails with FailedOpenEntity (2).
    fn bcat(&mut self, object: &ServiceObject, call: &mut Call<'_>) -> Result<Outcome, Trap> {
        const FAILED_OPEN_ENTITY: u32 = 122 | (2 << 9);
        const NO_OPEN_ENTITY: u32 = 122 | (7 << 9);
        const PROGRESS_DONE: u32 = 9;
        let command = call.request.command_id;
        let child = match (object, command) {
            // CreateBcatService / CreateDeliveryCacheStorageService (u64
            // process id, PID) and ...WithApplicationId (u64).
            (ServiceObject::BcatCreator, 0) => Some(ServiceObject::BcatService),
            (ServiceObject::BcatCreator, 1 | 2) => Some(ServiceObject::BcatStorage),
            // RequestSyncDeliveryCache(WithDirectoryName) -> progress.
            (ServiceObject::BcatService, 10100 | 10101) => Some(ServiceObject::BcatProgress),
            (ServiceObject::BcatStorage, 0) => Some(ServiceObject::BcatFile),
            (ServiceObject::BcatStorage, 1) => Some(ServiceObject::BcatDirectory),
            _ => None,
        };
        if let Some(child) = child {
            if call.request.arguments.len() > 8
                || call.request.input_buffer.is_some()
                || !call.request.receive_buffers.is_empty()
            {
                return Err(
                    call.unsupported(object.service(), "bcat creator with unexpected inputs")
                );
            }
            return self.reply_new_object(call, child, AOC_INTERFACE_ID);
        }
        match (object, command) {
            // EnumerateDeliveryCacheDirectory: DirectoryName[] into B -> s32 0.
            (ServiceObject::BcatStorage, 10) => {
                call.shape_output(0, false)?;
                call.reply(0, None, AOC_INTERFACE_ID, &0u32.to_le_bytes())?;
                Ok(Outcome::success())
            }
            // Open (directory and/or file name): nothing exists.
            (ServiceObject::BcatFile | ServiceObject::BcatDirectory, 0) => {
                call.reply(FAILED_OPEN_ENTITY, None, AOC_INTERFACE_ID, &[])?;
                Ok(Outcome::result(FAILED_OPEN_ENTITY))
            }
            (ServiceObject::BcatFile | ServiceObject::BcatDirectory, 1..=3) => {
                call.reply(NO_OPEN_ENTITY, None, AOC_INTERFACE_ID, &[])?;
                Ok(Outcome::result(NO_OPEN_ENTITY))
            }
            // GetEvent: a signaled event (the sync has completed).
            (ServiceObject::BcatProgress, 0) => {
                call.shape(0, false, false)?;
                let application = self
                    .main_application
                    .clone()
                    .ok_or_else(|| invalid("bcat requires the bootstrap main application"))?;
                if self.readable_events.len() >= MAX_READABLE_EVENTS {
                    return Err(invalid("readable event backing capacity exhausted"));
                }
                let id = ReadableEventId(
                    self.readable_events
                        .keys()
                        .map(|id| id.0)
                        .max()
                        .unwrap_or(0)
                        + 1,
                );
                let handle = self.next_available_handle()?;
                let reply = call.prepare_copy_handle_reply(handle, AOC_INTERFACE_ID)?;
                call.write_reply(&reply)?;
                self.readable_events.insert(
                    id,
                    ReadableEvent {
                        owner: application,
                        kind: ReadableEventKind::BcatProgress,
                        signaled: true,
                    },
                );
                self.readable_event_handles.insert(handle, id);
                self.next_handle = handle + 1;
                Ok(Outcome {
                    output_copy_handle_count: 1,
                    ..Outcome::success()
                })
            }
            // GetImpl: DeliveryCacheProgressImpl (0x200 bytes: s32 status,
            // Result, names and byte counts) into a C buffer.
            (ServiceObject::BcatProgress, 1) => {
                let Some(&(address, size)) = call.request.receive_list.first() else {
                    return Err(call.unsupported(object.service(), "GetImpl without a buffer"));
                };
                let mut progress = vec![0u8; 0x200.min(size as usize)];
                if progress.len() >= 4 {
                    progress[..4].copy_from_slice(&PROGRESS_DONE.to_le_bytes());
                }
                call.memory.write(address, &progress)?;
                call.reply(0, None, AOC_INTERFACE_ID, &[])?;
                Ok(Outcome::success())
            }
            _ => Err(call.unsupported(object.service(), "bcat command is not implemented")),
        }
    }

    /// nifm:u (Switchbrew numbering) for a console with no network. A
    /// submitted IRequest settles at once in state Free (1) and GetResult
    /// reports yuzu's ResultNetworkCommunicationDisabled (module 110,
    /// description 1111); the state-change event is signaled on Submit.
    fn nifm(&mut self, object: &ServiceObject, call: &mut Call<'_>) -> Result<Outcome, Trap> {
        const NETWORK_DISABLED: u32 = 110 | (1111 << 9);
        const REQUEST_FREE: u32 = 1;
        let command = call.request.command_id;
        match (object, command) {
            // CreateGeneralServiceOld (4) / CreateGeneralService (5: PID, u64).
            (ServiceObject::NifmStatic, 4) => {
                call.shape(0, false, false)?;
                self.reply_new_object(call, ServiceObject::NifmGeneral, AOC_INTERFACE_ID)
            }
            (ServiceObject::NifmStatic, 5) => {
                call.shape(8, true, false)?;
                self.reply_new_object(call, ServiceObject::NifmGeneral, AOC_INTERFACE_ID)
            }
            // GetClientId: ClientId {u32} into a C (0x1A) buffer.
            (ServiceObject::NifmGeneral, 1) => {
                let Some(&(address, size)) = call.request.receive_list.first() else {
                    return Err(call.unsupported(object.service(), "GetClientId without a buffer"));
                };
                if size >= 4 {
                    call.memory.write(address, &1u32.to_le_bytes())?;
                }
                call.reply(0, None, AOC_INTERFACE_ID, &[])?;
                Ok(Outcome::success())
            }
            // CreateRequest (s32 RequirementPreset).
            (ServiceObject::NifmGeneral, 4) => {
                call.shape(4, false, false)?;
                let id = self.nifm_requests.len();
                self.nifm_requests.push(None);
                self.reply_new_object(call, ServiceObject::NifmRequest { id }, AOC_INTERFACE_ID)
            }
            // GetCurrentIpAddress / GetInternetConnectionStatus: not connected.
            (ServiceObject::NifmGeneral, 12 | 18) => {
                call.shape(0, false, false)?;
                call.reply(NETWORK_DISABLED, None, AOC_INTERFACE_ID, &[])?;
                Ok(Outcome::result(NETWORK_DISABLED))
            }
            // GetCurrentNetworkProfile (5, and 101 with NetworkProfileDataV2:
            // the SDK's nn::nifm::GetCurrentNetworkProfile sends 101): the
            // profile goes to an output buffer; there is no current network.
            (ServiceObject::NifmGeneral, 5 | 101) => {
                // The buffer is a B or C descriptor; nothing is written.
                if !call.request.arguments.is_empty() || call.request.input_buffer.is_some() {
                    return Err(call.unsupported(
                        object.service(),
                        format!(
                            "GetCurrentNetworkProfile shape ({})",
                            call.request.describe()
                        ),
                    ));
                }
                call.reply(NETWORK_DISABLED, None, AOC_INTERFACE_ID, &[])?;
                Ok(Outcome::result(NETWORK_DISABLED))
            }
            // IsWirelessCommunicationEnabled / IsAnyInternetRequestAccepted.
            (ServiceObject::NifmGeneral, 17 | 21) => {
                call.reply(0, None, AOC_INTERFACE_ID, &[0])?;
                Ok(Outcome::success())
            }
            (ServiceObject::NifmRequest { .. }, 0) => {
                call.shape(0, false, false)?;
                call.reply(0, None, AOC_INTERFACE_ID, &REQUEST_FREE.to_le_bytes())?;
                Ok(Outcome::success())
            }
            (ServiceObject::NifmRequest { .. }, 1) => {
                call.shape(0, false, false)?;
                call.reply(NETWORK_DISABLED, None, AOC_INTERFACE_ID, &[])?;
                Ok(Outcome::result(NETWORK_DISABLED))
            }
            // GetSystemEventReadableHandles: two copied events.
            (ServiceObject::NifmRequest { id }, 2) => {
                call.shape(0, false, false)?;
                let id = *id;
                let application = self
                    .main_application
                    .clone()
                    .ok_or_else(|| invalid("nifm requires the bootstrap main application"))?;
                let events = match self.nifm_requests[id] {
                    Some(events) => events,
                    None => {
                        if self.readable_events.len() + 2 > MAX_READABLE_EVENTS {
                            return Err(invalid("readable event backing capacity exhausted"));
                        }
                        let next = self
                            .readable_events
                            .keys()
                            .map(|id| id.0)
                            .max()
                            .unwrap_or(0)
                            + 1;
                        let events = (ReadableEventId(next), ReadableEventId(next + 1));
                        for event in [events.0, events.1] {
                            self.readable_events.insert(
                                event,
                                ReadableEvent {
                                    owner: application.clone(),
                                    kind: ReadableEventKind::NetworkRequest,
                                    signaled: false,
                                },
                            );
                        }
                        self.nifm_requests[id] = Some(events);
                        events
                    }
                };
                let first = self.next_available_handle()?;
                let second = first
                    .checked_add(1)
                    .filter(|handle| !self.sessions.contains_key(handle))
                    .ok_or_else(|| invalid("no second handle for the nifm events"))?;
                let reply = ipc::build_copy_handle_pair_reply(
                    call.request,
                    [first, second],
                    AOC_INTERFACE_ID,
                )
                .map_err(|error| invalid(format!("build nifm event reply: {error:#}")))?;
                call.write_reply(&reply)?;
                self.readable_event_handles.insert(first, events.0);
                self.readable_event_handles.insert(second, events.1);
                self.next_handle = second + 1;
                Ok(Outcome {
                    output_copy_handle_count: 2,
                    ..Outcome::success()
                })
            }
            // Submit: the request settles immediately (Free).
            (ServiceObject::NifmRequest { id }, 4) => {
                call.shape(0, false, false)?;
                if let Some((event, _)) = self.nifm_requests[*id] {
                    if let Some(event) = self.readable_events.get_mut(&event) {
                        event.signaled = true;
                    }
                }
                call.reply(0, None, AOC_INTERFACE_ID, &[])?;
                Ok(Outcome::success())
            }
            // Cancel and the scalar setters (SetRequirementPreset,
            // SetPriority, SetRejectable, SetConnectionConfirmationOption,
            // SetPersistent, SetInstant, SetSustainable, SetRawPriority,
            // SetGreedy, SetSharable, SetKeptInSleep): accepted, no effect.
            (ServiceObject::NifmRequest { .. }, 3 | 6 | 8 | 10..=17 | 23) => {
                if call.request.input_buffer.is_some() || !call.request.receive_buffers.is_empty() {
                    return Err(call.unsupported(object.service(), "nifm setter with buffers"));
                }
                call.reply(0, None, AOC_INTERFACE_ID, &[])?;
                Ok(Outcome::success())
            }
            _ => Err(call.unsupported(object.service(), "nifm command is not implemented")),
        }
    }

    /// ssl ISslService / ISslContext (Switchbrew numbering).
    fn ssl(&mut self, object: &ServiceObject, call: &mut Call<'_>) -> Result<Outcome, Trap> {
        let command = call.request.command_id;
        match (object, command) {
            // CreateContext: PID, u32 SslVersion, u64 PID placeholder.
            (ServiceObject::SslService, 0) => {
                call.shape(16, true, false)?;
                self.reply_new_object(call, ServiceObject::SslContext, AOC_INTERFACE_ID)
            }
            // SetInterfaceVersion (u32).
            (ServiceObject::SslService, 5) => {
                call.shape(4, false, false)?;
                call.reply(0, None, AOC_INTERFACE_ID, &[])?;
                Ok(Outcome::success())
            }
            // SetOption (ContextOption, s32) and AddPolicyOid (X string).
            (ServiceObject::SslContext, 0) => {
                call.shape(8, false, false)?;
                call.reply(0, None, AOC_INTERFACE_ID, &[])?;
                Ok(Outcome::success())
            }
            (ServiceObject::SslContext, 9) => {
                call.shape(0, false, true)?;
                call.reply(0, None, AOC_INTERFACE_ID, &[])?;
                Ok(Outcome::success())
            }
            // ImportServerPki (X certs, u32 format), ImportCrl (X DER) and
            // RegisterInternalPki (u32): reply a fresh u64 object id.
            (ServiceObject::SslContext, 4 | 8 | 10) => {
                let (arguments, buffer) = match command {
                    4 => (4, true),
                    8 => (4, false),
                    _ => (0, true),
                };
                call.shape(arguments, false, buffer)?;
                self.ssl_next_id += 1;
                call.reply(0, None, AOC_INTERFACE_ID, &self.ssl_next_id.to_le_bytes())?;
                Ok(Outcome::success())
            }
            _ => Err(call.unsupported(object.service(), "ssl command is not implemented")),
        }
    }

    /// IHardwareOpusDecoderManager (Switchbrew / yuzu numbering). Work
    /// buffers hold no decoder state here, so sizes are a local value.
    fn opus_manager(&mut self, call: &mut Call<'_>) -> Result<Outcome, Trap> {
        const SERVICE: &str = "hwopus";
        let command = call.request.command_id;
        let arguments = call.request.arguments.clone();
        let word = |index: usize| -> Option<u32> {
            arguments
                .get(index * 4..index * 4 + 4)
                .map(|bytes| crate::bytes::le_u32(bytes))
        };
        // (sample_rate, channels) from inline OpusParameters(Ex) or the X
        // buffer's OpusMultiStreamParameters(Ex) (which start the same way).
        let parameters = |call: &mut Call<'_>| -> Result<(u32, u32), Trap> {
            if matches!(command, 2 | 3 | 6 | 7) {
                let input = call
                    .request
                    .input_buffer
                    .clone()
                    .ok_or_else(|| call.unsupported(SERVICE, "missing Opus parameters"))?;
                let mut bytes = [0u8; 8];
                call.memory.read(input.address, &mut bytes)?;
                Ok((
                    crate::bytes::le_u32(&bytes[..4]),
                    crate::bytes::le_u32(&bytes[4..]),
                ))
            } else {
                match (word(0), word(1)) {
                    (Some(rate), Some(channels)) => Ok((rate, channels)),
                    _ => Err(call.unsupported(SERVICE, "short Opus parameters")),
                }
            }
        };
        let (sample_rate, channels) = parameters(call)?;
        if !matches!(sample_rate, 8_000 | 12_000 | 16_000 | 24_000 | 48_000)
            || !(1..=255).contains(&channels)
        {
            return Err(call.unsupported(SERVICE, "invalid Opus decoder parameters"));
        }
        match command {
            // GetWorkBufferSize / Ex / ExEx and the multi-stream forms.
            1 | 3 | 5 | 7 | 8 => {
                let size = 0x1000 + channels * 0x10000;
                call.reply(0, None, AOC_INTERFACE_ID, &size.to_le_bytes())?;
                Ok(Outcome::success())
            }
            // Open* (0, 2, 4, 6): the work buffer arrives as transfer memory
            // (copied handle); it is not used.
            _ => {
                if call.request.copy_handles.len() != 1 {
                    return Err(call.unsupported(SERVICE, "Opus open without work-buffer memory"));
                }
                self.reply_new_object(
                    call,
                    ServiceObject::OpusDecoder {
                        sample_rate,
                        channels,
                    },
                    AOC_INTERFACE_ID,
                )
            }
        }
    }

    /// IHardwareOpusDecoder decode commands (0, 2, 4-9): the input holds
    /// OpusPacketHeader {u32 size (big-endian), u32 final_range} and the
    /// packet. The output receives the packet's duration of interleaved
    /// PCM16 silence; no audio is produced anywhere, so decoded content is
    /// unobservable. Replies u32 consumed bytes and u32 samples per channel,
    /// plus u64 decode time for the "Perf" forms (4-9).
    fn opus_decode(
        &mut self,
        sample_rate: u32,
        channels: u32,
        call: &mut Call<'_>,
    ) -> Result<Outcome, Trap> {
        const SERVICE: &str = "hwopus:decoder";
        let command = call.request.command_id;
        if !matches!(command, 0 | 2 | 4..=9) {
            return Err(call.unsupported(SERVICE, "Opus decoder command is not implemented"));
        }
        let input = call
            .request
            .input_buffer
            .clone()
            .ok_or_else(|| call.unsupported(SERVICE, "decode without input"))?;
        if call.request.receive_buffers.len() != 1 {
            return Err(call.unsupported(SERVICE, "decode without one output buffer"));
        }
        let (output, capacity) = call.output_buffer()?;
        let mut header = [0u8; 8];
        if input.size < 8 {
            return Err(call.unsupported(SERVICE, "short Opus packet header"));
        }
        call.memory.read(input.address, &mut header)?;
        let size = u64::from(crate::bytes::be_u32(&header[..4]));
        if size == 0 || 8 + size > input.size {
            return Err(call.unsupported(SERVICE, "Opus packet exceeds its input buffer"));
        }
        let mut packet = vec![0u8; size.min(2) as usize];
        call.memory.read(input.address + 8, &mut packet)?;
        let samples = opus_packet_samples(&packet, sample_rate)
            .ok_or_else(|| call.unsupported(SERVICE, "malformed Opus packet"))?;
        let bytes = samples * u64::from(channels) * 2;
        if bytes > capacity {
            return Err(call.unsupported(SERVICE, "Opus output buffer too small"));
        }
        call.memory.write(output, &vec![0u8; bytes as usize])?;
        let mut reply = Vec::with_capacity(16);
        reply.extend_from_slice(&((8 + size) as u32).to_le_bytes());
        reply.extend_from_slice(&(samples as u32).to_le_bytes());
        if command >= 4 {
            reply.extend_from_slice(&0u64.to_le_bytes());
        }
        call.reply(0, None, AOC_INTERFACE_ID, &reply)?;
        self.provisional_commands.insert(SERVICE);
        Ok(Outcome::success())
    }

    /// Release finished audio-out buffers and signal their events. Returns
    /// whether any event was signaled.
    pub(crate) fn advance_audio(&mut self, now: std::time::Instant) -> bool {
        let mut signaled = false;
        for (renderer, event) in &mut self.audio_renderers {
            if renderer.advance(now) {
                if let Some(event) = event.and_then(|id| self.readable_events.get_mut(&id)) {
                    event.signaled = true;
                    signaled = true;
                }
            }
        }
        for out in &mut self.audio_outs {
            if out.advance(now) {
                if let Some(event) = out.event.and_then(|id| self.readable_events.get_mut(&id)) {
                    event.signaled = true;
                    signaled = true;
                }
            }
        }
        signaled
    }

    /// When the next audio-out buffer finishes, if any is playing.
    pub(crate) fn next_audio_deadline(&self) -> Option<std::time::Instant> {
        self.audio_outs
            .iter()
            .filter(|out| out.started)
            .filter_map(|out| out.front_end)
            .chain(
                self.audio_renderers
                    .iter()
                    .filter_map(|(renderer, _)| renderer.next_deadline()),
            )
            .min()
    }

    /// IAudioRenderer (Eden-NS2 1a6d7d4 audio_renderer.cpp numbering). The
    /// renderer state is `audio_renderer::Renderer`; nothing is mixed.
    fn audio_renderer(&mut self, id: usize, call: &mut Call<'_>) -> Result<Outcome, Trap> {
        const SERVICE: &str = "aud:u:audio_renderer";
        let now = std::time::Instant::now();
        let command = call.request.command_id;
        let renderer = &mut self.audio_renderers[id].0;
        renderer.advance(now);
        let parameters = renderer.parameters;
        let word = |value: u32| value.to_le_bytes();
        match command {
            // GetSampleRate, GetSampleCount, GetMixBufferCount, GetState (0
            // started, 1 stopped), GetRenderingTimeLimit.
            0..=3 | 9 => {
                call.shape(0, false, false)?;
                let value = match command {
                    0 => parameters.sample_rate,
                    1 => parameters.sample_count,
                    2 => parameters.mixes,
                    3 => u32::from(!renderer.started),
                    _ => renderer.rendering_time_limit,
                };
                call.reply(0, None, AOC_INTERFACE_ID, &word(value))?;
            }
            // RequestUpdate (4): input in an A buffer, output and
            // performance in B buffers; RequestUpdateAuto (10) sends the
            // same three as AutoSelect buffers.
            4 | 10 => {
                let request = &call.request;
                let input = request
                    .input_buffer
                    .as_ref()
                    .map(|buffer| (buffer.address, buffer.size));
                let output = request
                    .receive_buffers
                    .first()
                    .map(|&(address, size, _)| (address, size));
                let output = output.or_else(|| request.receive_list.first().copied());
                let (Some((input_address, input_size)), Some((output_address, output_size))) =
                    (input, output)
                else {
                    return Err(call.unsupported(
                        SERVICE,
                        format!("RequestUpdate buffers ({})", request.describe()),
                    ));
                };
                if input_size > 0x10_0000 || output_size > 0x10_0000 {
                    return Err(invalid("renderer update buffers exceed 1 MiB"));
                }
                let mut bytes = vec![0u8; input_size as usize];
                call.memory.read(input_address, &mut bytes)?;
                let output = self.audio_renderers[id]
                    .0
                    .update(&bytes, output_size as usize)
                    .map_err(|message| call.unsupported(SERVICE, message))?;
                call.memory.write(output_address, &output)?;
                call.reply(0, None, AOC_INTERFACE_ID, &[])?;
            }
            // Start (5) / Stop (6).
            5 | 6 => {
                call.shape(0, false, false)?;
                self.audio_renderers[id].0.set_started(command == 5, now);
                call.reply(0, None, AOC_INTERFACE_ID, &[])?;
            }
            // QuerySystemEvent (7): one copied event, signaled per frame.
            7 => {
                call.shape(0, false, false)?;
                let application = self
                    .main_application
                    .clone()
                    .ok_or_else(|| invalid("aud:u requires the bootstrap main application"))?;
                let event = match self.audio_renderers[id].1 {
                    Some(event) => event,
                    None => {
                        if self.readable_events.len() >= MAX_READABLE_EVENTS {
                            return Err(invalid("readable event backing capacity exhausted"));
                        }
                        ReadableEventId(
                            self.readable_events
                                .keys()
                                .map(|id| id.0)
                                .max()
                                .unwrap_or(0)
                                + 1,
                        )
                    }
                };
                let handle = self.next_available_handle()?;
                let reply = call.prepare_copy_handle_reply(handle, 0)?;
                call.write_reply(&reply)?;
                self.readable_events.entry(event).or_insert(ReadableEvent {
                    owner: application,
                    kind: ReadableEventKind::AudioRendererFrame,
                    signaled: false,
                });
                self.audio_renderers[id].1 = Some(event);
                self.readable_event_handles.insert(handle, event);
                self.next_handle = handle + 1;
                return Ok(Outcome {
                    output_copy_handle_count: 1,
                    ..Outcome::success()
                });
            }
            // SetVoiceDropParameter (12, f32) / GetVoiceDropParameter (13).
            12 => {
                call.shape(4, false, false)?;
                let value = crate::bytes::le_u32(&call.request.arguments[..4]);
                self.audio_renderers[id].0.voice_drop_parameter = value;
                call.reply(0, None, AOC_INTERFACE_ID, &[])?;
            }
            13 => {
                call.shape(0, false, false)?;
                let value = self.audio_renderers[id].0.voice_drop_parameter;
                call.reply(0, None, AOC_INTERFACE_ID, &value.to_le_bytes())?;
            }
            // SetRenderingTimeLimit (8): u32 percent.
            8 => {
                call.shape(4, false, false)?;
                let limit = crate::bytes::le_u32(&call.request.arguments[..4]);
                self.audio_renderers[id].0.rendering_time_limit = limit;
                call.reply(0, None, AOC_INTERFACE_ID, &[])?;
            }
            _ => {
                return Err(call.unsupported(SERVICE, "audio renderer command is not implemented"));
            }
        }
        Ok(Outcome::success())
    }

    /// IAudioOut (yuzu audio_out.cpp command numbering).
    fn audio_out(&mut self, id: usize, call: &mut Call<'_>) -> Result<Outcome, Trap> {
        const SERVICE: &str = "aud:u:audio_out";
        let now = std::time::Instant::now();
        let command = call.request.command_id;
        if crate::nvn::debug_var_os("ECHORA_AUDIO_TRACE").is_some() {
            let out = &self.audio_outs[id];
            eprintln!(
                "audio-out {id} command {command} tls {:#x} pc {:#x} queued {} released {} played {}",
                call.buffer_address,
                call.pc,
                out.queue.len(),
                out.released.len(),
                out.played_samples
            );
        }
        match command {
            // GetAudioOutState: 0 started, 1 stopped.
            0 => {
                call.shape(0, false, false)?;
                let state = if self.audio_outs[id].started { 0u32 } else { 1 };
                call.reply(0, None, AOC_INTERFACE_ID, &state.to_le_bytes())?;
            }
            // Start / Stop.
            1 | 2 => {
                call.shape(0, false, false)?;
                let out = &mut self.audio_outs[id];
                out.advance(now);
                out.started = command == 1;
                out.front_end = None;
                call.reply(0, None, AOC_INTERFACE_ID, &[])?;
            }
            // AppendAudioOutBuffer(u64 tag, AudioOutBuffer in A/AutoSelect):
            // AudioOutBuffer {u64 next, u64 samples, u64 capacity, u64 size,
            // u64 offset}; `size` bytes of PCM16 interleaved frames.
            3 | 7 => {
                call.shape(8, false, true)?;
                let tag = crate::bytes::le_u64(&call.request.arguments[..8]);
                let input = call
                    .request
                    .input_buffer
                    .clone()
                    .expect("input buffer presence checked by the request shape");
                if input.size < 0x28 {
                    return Err(call.unsupported(SERVICE, "short AudioOutBuffer"));
                }
                let mut record = [0u8; 0x28];
                call.memory.read(input.address, &mut record)?;
                let size = crate::bytes::le_u64(&record[0x18..0x20]);
                if let Some(host) = &self.audio_outs[id].host {
                    // PCM at samples + offset; a bounded copy per buffer.
                    let samples = crate::bytes::le_u64(&record[0x08..0x10]);
                    let offset = crate::bytes::le_u64(&record[0x20..0x28]);
                    let mut pcm = vec![0u8; size.min(1 << 20) as usize];
                    if call
                        .memory
                        .read(samples.wrapping_add(offset), &mut pcm)
                        .is_ok()
                    {
                        crate::audio::apply_volume(
                            &mut pcm,
                            self.audio_outs[id].volume * crate::settings::master_volume(),
                        );
                        let _ = host.send(pcm);
                    }
                }
                let out = &mut self.audio_outs[id];
                if out.queue.len() + out.released.len() >= MAX_AUDIO_OUT_BUFFERS {
                    return Err(call.unsupported(SERVICE, "audio-out buffer queue full"));
                }
                out.advance(now);
                let frames = size / (u64::from(out.channels) * 2);
                if out.queue.is_empty() {
                    out.front_end = None;
                }
                out.queue.push_back((tag, frames));
                if out.started && out.front_end.is_none() {
                    out.advance(now);
                }
                call.reply(0, None, AOC_INTERFACE_ID, &[])?;
            }
            // RegisterBufferEvent: the readable side of the buffer event.
            4 => {
                call.shape(0, false, false)?;
                let application = self
                    .main_application
                    .clone()
                    .ok_or_else(|| invalid("aud:u requires the bootstrap main application"))?;
                let id_event = match self.audio_outs[id].event {
                    Some(event) => event,
                    None => {
                        if self.readable_events.len() >= MAX_READABLE_EVENTS {
                            return Err(invalid("readable event backing capacity exhausted"));
                        }
                        ReadableEventId(
                            self.readable_events
                                .keys()
                                .map(|id| id.0)
                                .max()
                                .unwrap_or(0)
                                + 1,
                        )
                    }
                };
                let handle = self.next_available_handle()?;
                // A copied handle (yuzu audio_out.cpp PushCopyObjects): sent
                // as a moved handle, the client read 0 from its copy slot
                // and every wait on it failed with InvalidHandle.
                let reply = call.prepare_copy_handle_reply(handle, 0)?;
                call.write_reply(&reply)?;
                self.readable_events
                    .entry(id_event)
                    .or_insert(ReadableEvent {
                        owner: application,
                        kind: ReadableEventKind::AudioOutBuffer,
                        signaled: false,
                    });
                self.audio_outs[id].event = Some(id_event);
                self.readable_event_handles.insert(handle, id_event);
                self.next_handle = handle + 1;
                self.provisional_commands.insert(SERVICE);
                return Ok(Outcome {
                    output_copy_handle_count: 1,
                    ..Outcome::success()
                });
            }
            // GetReleasedAudioOutBuffers(Auto): u64 tags into the output
            // buffer, u32 count reply. The game sends 4 argument bytes with
            // command 8 (observed; their meaning is not known, so they are
            // ignored and the output buffer bounds the count).
            5 | 8 => {
                if call.shape_output(0, false).is_err() {
                    call.shape_output(4, false)?;
                }
                if crate::nvn::debug_var_os("ECHORA_AUDIO_TRACE").is_some()
                    && !call.request.arguments.is_empty()
                {
                    eprintln!(
                        "audio-out {id} command {command} argument {:02x?}",
                        call.request.arguments
                    );
                }
                let (address, size) = call.output_buffer()?;
                let out = &mut self.audio_outs[id];
                out.advance(now);
                let count = (size / 8).min(out.released.len() as u64) as usize;
                let tags: Vec<u8> = out
                    .released
                    .drain(..count)
                    .flat_map(u64::to_le_bytes)
                    .collect();
                call.memory.write(address, &tags)?;
                call.reply(0, None, AOC_INTERFACE_ID, &(count as u32).to_le_bytes())?;
            }
            // ContainsAudioOutBuffer(u64 tag) -> bool.
            6 => {
                call.shape(8, false, false)?;
                let tag = crate::bytes::le_u64(&call.request.arguments[..8]);
                let out = &self.audio_outs[id];
                let present = out.queue.iter().any(|(queued, _)| *queued == tag)
                    || out.released.contains(&tag);
                call.reply(0, None, AOC_INTERFACE_ID, &[u8::from(present)])?;
            }
            // GetAudioOutBufferCount -> u32 (queued, not yet released).
            9 => {
                call.shape(0, false, false)?;
                let out = &mut self.audio_outs[id];
                out.advance(now);
                call.reply(
                    0,
                    None,
                    AOC_INTERFACE_ID,
                    &(out.queue.len() as u32).to_le_bytes(),
                )?;
            }
            // GetAudioOutPlayedSampleCount -> u64.
            10 => {
                call.shape(0, false, false)?;
                let out = &mut self.audio_outs[id];
                out.advance(now);
                call.reply(0, None, AOC_INTERFACE_ID, &out.played_samples.to_le_bytes())?;
            }
            // FlushAudioOutBuffers -> bool: release everything queued now.
            11 => {
                call.shape(0, false, false)?;
                let out = &mut self.audio_outs[id];
                let flushed = !out.queue.is_empty();
                while let Some((tag, _)) = out.queue.pop_front() {
                    out.released.push_back(tag);
                }
                out.front_end = None;
                if let Some(event) = out.event.and_then(|id| self.readable_events.get_mut(&id)) {
                    event.signaled |= flushed;
                }
                call.reply(0, None, AOC_INTERFACE_ID, &[u8::from(flushed)])?;
            }
            // SetAudioOutVolume(f32) / GetAudioOutVolume -> f32.
            12 => {
                call.shape(4, false, false)?;
                let volume = crate::bytes::le_f32(&call.request.arguments[..4]);
                self.audio_outs[id].volume = volume;
                call.reply(0, None, AOC_INTERFACE_ID, &[])?;
            }
            13 => {
                call.shape(0, false, false)?;
                let volume = self.audio_outs[id].volume;
                call.reply(0, None, AOC_INTERFACE_ID, &volume.to_le_bytes())?;
            }
            _ => return Err(call.unsupported(SERVICE, "audio-out command is not implemented")),
        }
        self.provisional_commands.insert(SERVICE);
        Ok(Outcome::success())
    }

    fn time(&mut self, object: &ServiceObject, call: &mut Call<'_>) -> Result<Outcome, Trap> {
        let command = call.request.command_id;
        let steady = self.steady_epoch.elapsed().as_secs() as i64;
        let posix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|span| span.as_secs() as i64)
            .unwrap_or(0);
        match (object, command) {
            (ServiceObject::TimeStatic, 0 | 1 | 4 | 5) => {
                call.shape(0, false, false)?;
                self.reply_new_object(
                    call,
                    ServiceObject::SystemClock { clock: command },
                    AOC_INTERFACE_ID,
                )
            }
            (ServiceObject::TimeStatic, 2) => {
                call.shape(0, false, false)?;
                self.reply_new_object(call, ServiceObject::SteadyClock, AOC_INTERFACE_ID)
            }
            (ServiceObject::TimeStatic, 3) => {
                call.shape(0, false, false)?;
                self.reply_new_object(call, ServiceObject::TimeZone, AOC_INTERFACE_ID)
            }
            (ServiceObject::SystemClock { .. }, 0) => {
                call.shape(0, false, false)?;
                call.reply(0, None, AOC_INTERFACE_ID, &posix.to_le_bytes())?;
                Ok(Outcome::success())
            }
            (ServiceObject::SystemClock { .. }, 2) => {
                call.shape(0, false, false)?;
                let mut context = Vec::with_capacity(0x20);
                context.extend_from_slice(&(posix - steady).to_le_bytes());
                context.extend_from_slice(&steady.to_le_bytes());
                context.extend_from_slice(&self.steady_clock_source);
                call.reply(0, None, AOC_INTERFACE_ID, &context)?;
                Ok(Outcome::success())
            }
            (ServiceObject::SteadyClock, 0) => {
                call.shape(0, false, false)?;
                let mut point = Vec::with_capacity(0x18);
                point.extend_from_slice(&steady.to_le_bytes());
                point.extend_from_slice(&self.steady_clock_source);
                call.reply(0, None, AOC_INTERFACE_ID, &point)?;
                Ok(Outcome::success())
            }
            (ServiceObject::TimeStatic, 20) => {
                call.shape(0, false, false)?;
                if !self.shared_memories.contains_key(&TIME_SHARED_MEMORY_ID) {
                    let object = self.time_shared_memory(posix, steady);
                    self.shared_memories.insert(TIME_SHARED_MEMORY_ID, object);
                }
                let handle = self.next_available_handle()?;
                let reply = call.prepare_copy_handle_reply(handle, AOC_INTERFACE_ID)?;
                call.write_reply(&reply)?;
                self.shared_memory_handles
                    .insert(handle, TIME_SHARED_MEMORY_ID);
                self.next_handle = handle + 1;
                Ok(Outcome {
                    output_copy_handle_count: 1,
                    ..Outcome::success()
                })
            }
            // GetClockSnapshot (400): u8 type -> ClockSnapshot (0xD0 bytes,
            // yuzu/Eden layout) in an output buffer: user and network
            // SystemClockContexts, both POSIX times, CalendarTimes and their
            // CalendarAdditionalInfos, the steady clock time point, the
            // location name, automatic correction and the type. With no
            // network the network clock equals the user clock.
            (ServiceObject::TimeStatic, 400) => {
                // The u8 arrives padded to a word; the buffer may be B or C.
                if !(1..=4).contains(&call.request.arguments.len())
                    || call.request.pid.is_some()
                    || call.request.input_buffer.is_some()
                {
                    return Err(call.unsupported(
                        "time:u",
                        format!("GetClockSnapshot shape ({})", call.request.describe()),
                    ));
                }
                let kind = call.request.arguments[0];
                let (address, size) = call.output_buffer()?;
                if size < 0xd0 {
                    return Err(call.unsupported("time:u", "ClockSnapshot buffer under 0xd0 bytes"));
                }
                let mut context = Vec::with_capacity(0x20);
                context.extend_from_slice(&(posix - steady).to_le_bytes());
                context.extend_from_slice(&steady.to_le_bytes());
                context.extend_from_slice(&self.steady_clock_source);
                let calendar = local_calendar_time(posix);
                let mut snapshot = vec![0u8; 0xd0];
                snapshot[0..0x20].copy_from_slice(&context);
                snapshot[0x20..0x40].copy_from_slice(&context);
                snapshot[0x40..0x48].copy_from_slice(&posix.to_le_bytes());
                snapshot[0x48..0x50].copy_from_slice(&posix.to_le_bytes());
                snapshot[0x50..0x58].copy_from_slice(&calendar[..8]);
                snapshot[0x58..0x60].copy_from_slice(&calendar[..8]);
                snapshot[0x60..0x78].copy_from_slice(&calendar[8..]);
                snapshot[0x78..0x90].copy_from_slice(&calendar[8..]);
                snapshot[0x90..0x98].copy_from_slice(&steady.to_le_bytes());
                snapshot[0x98..0xa8].copy_from_slice(&self.steady_clock_source);
                snapshot[0xa8..0xab].copy_from_slice(b"UTC");
                snapshot[0xcd] = kind;
                call.memory.write(address, &snapshot)?;
                call.reply(0, None, AOC_INTERFACE_ID, &[])?;
                Ok(Outcome::success())
            }
            // ITimeZoneService. The device location is UTC (inference: no
            // rule data is loaded, so every conversion uses offset 0).
            (ServiceObject::TimeZone, 0) => {
                call.shape(0, false, false)?;
                let mut name = [0u8; 0x24];
                name[..3].copy_from_slice(b"UTC");
                call.reply(0, None, AOC_INTERFACE_ID, &name)?;
                Ok(Outcome::success())
            }
            (ServiceObject::TimeZone, 100 | 101) => {
                call.shape(8, false, command == 100)?;
                let time = crate::bytes::le_i64(&call.request.arguments[..8]);
                call.reply(0, None, AOC_INTERFACE_ID, &local_calendar_time(time))?;
                Ok(Outcome::success())
            }
            // ToPosixTime (201, with a rule buffer) / ToPosixTimeWithMyRule
            // (202): CalendarTime -> u32 count and PosixTime[] in a C buffer.
            // In UTC every calendar time has exactly one POSIX time.
            (ServiceObject::TimeZone, 201 | 202) => {
                if call.request.arguments.len() != 8 {
                    return Err(call.unsupported("time:u", "ToPosixTime expects a CalendarTime"));
                }
                let Some(&(address, size)) = call.request.receive_list.first() else {
                    return Err(call.unsupported("time:u", "ToPosixTime without an output buffer"));
                };
                let mut calendar = [0u8; 8];
                calendar.copy_from_slice(&call.request.arguments[..8]);
                let count: u32 = if size >= 8 {
                    call.memory
                        .write(address, &local_posix_time(calendar).to_le_bytes())?;
                    1
                } else {
                    0
                };
                call.reply(0, None, AOC_INTERFACE_ID, &count.to_le_bytes())?;
                Ok(Outcome::success())
            }
            _ => Err(call.unsupported("time:u", "time command is not implemented")),
        }
    }

    /// The 4 KiB time shared memory (Eden psc/time/shared_memory.h layout).
    /// Each field is {u32 counter; T value[2]}; counter 0 selects value[0].
    /// Steady base is 0 ns because the steady clock and the guest tick
    /// counter share the session epoch; contexts carry steady seconds.
    fn time_shared_memory(&self, posix: i64, steady: i64) -> SharedMemoryObject {
        let mut bytes = vec![0u8; 0x1000];
        let source = self.steady_clock_source;
        let put = |bytes: &mut Vec<u8>, at: usize, data: &[u8]| {
            bytes[at..at + data.len()].copy_from_slice(data);
        };
        // steady_time_points @0x00: value[0] @0x08 = {time_point: 0, source}.
        put(&mut bytes, 0x08, &0i64.to_le_bytes());
        put(&mut bytes, 0x10, &source);
        // local (0x38) / network (0x80) contexts: value[0] @+0x08.
        for base in [0x38usize, 0x80] {
            put(&mut bytes, base + 0x08, &(posix - steady).to_le_bytes());
            put(&mut bytes, base + 0x10, &steady.to_le_bytes());
            put(&mut bytes, base + 0x18, &source);
        }
        // automatic_corrections @0xC8: value[0] @0xCC = false.
        // continuous adjustment @0xD0: value[0] @0xD8 = {rtc_offset: uptime
        // ns, diff_scale 0, shift 0, lower 0, upper 0, source}.
        // Fields: rtc_offset @0xD8, diff_scale @0xE0, shift_amount @0xE8,
        // lower @0xF0, upper @0xF8 (both the 0 ns base), source @0x100.
        let uptime = self.steady_epoch.elapsed().as_nanos() as i64;
        put(&mut bytes, 0xd8, &uptime.to_le_bytes());
        put(&mut bytes, 0x100, &source);
        SharedMemoryObject {
            size: 0x1000,
            permission: crate::memory::READ,
            contents: bytes,
            label: "time shared memory",
        }
    }

    pub(crate) fn shared_memory(&self, handle: u32) -> Option<&SharedMemoryObject> {
        let id = self.shared_memory_handles.get(&handle)?;
        self.shared_memories.get(id)
    }

    /// pctl, following the pinned Eden-NS2 pctl sources. Plain `pctl` carries
    /// Application | SnsPost | Status | StereoVision capability. Native-PC
    /// policy: no parental-control PIN or restriction is configured, so the
    /// permission checks the reference grants without a PIN succeed and
    /// IsRestrictionEnabled is false. Granted permissions are recorded.
    fn parental_control(
        &mut self,
        object: &ServiceObject,
        call: &mut Call<'_>,
    ) -> Result<Outcome, Trap> {
        let command = call.request.command_id;
        match (object, command) {
            (ServiceObject::ParentalControlFactory, 0 | 1) => {
                call.shape(8, true, false)?;
                let application = self
                    .main_application
                    .as_ref()
                    .ok_or_else(|| invalid("pctl requires the bootstrap main application"))?;
                if application.process_id != call.process_id {
                    return Err(invalid("pctl caller PID is not the main application"));
                }
                self.reply_new_object(
                    call,
                    ServiceObject::ParentalControl {
                        initialized: command == 0,
                    },
                    AOC_INTERFACE_ID,
                )
            }
            (ServiceObject::ParentalControl { .. }, 1 | 1001 | 1013 | 1018) => {
                call.shape(0, false, false)?;
                call.reply(0, None, AOC_INTERFACE_ID, &[])?;
                // Initialize (1) is recorded alongside the granted checks; the
                // object's creation-time `initialized` flag is not rewritten.
                self.parental_control_grants.insert(command);
                Ok(Outcome::success())
            }
            (ServiceObject::ParentalControl { .. }, 1031 | 1032) => {
                call.shape(0, false, false)?;
                // IsRestrictionEnabled: false (no PIN); GetSafetyLevel: u32 0.
                let data: &[u8] = if command == 1031 { &[0] } else { &[0, 0, 0, 0] };
                call.reply(0, None, AOC_INTERFACE_ID, data)?;
                Ok(Outcome::success())
            }
            _ => Err(call.unsupported("pctl", "parental-control command is not implemented")),
        }
    }

    /// Explicit provisional policy for undocumented Switch 2 commands: require
    /// the observed argument size (a plain-CMIF request may pad it to a word),
    /// reply success with eight zero output bytes so small scalar outputs read
    /// as 0/false, and record the command and raw input in the report. This is
    /// not reference-backed behavior and is reported as such.
    fn provisional_reply(
        &mut self,
        application: &MainApplicationIdentity,
        call: &mut Call<'_>,
        name: &'static str,
        arguments: usize,
        interface_id: u32,
    ) -> Result<Outcome, Trap> {
        let padded = arguments.next_multiple_of(4);
        let length = if arguments != 0 && call.request.arguments.len() == padded {
            padded
        } else {
            arguments
        };
        call.shape(length, false, false)?;
        if self.main_application.as_ref() != Some(application) {
            return Err(invalid(
                "provisional command owner must match the main application",
            ));
        }
        let input = call.request.arguments[..arguments].to_vec();
        call.reply(0, None, interface_id, &[0; 8])?;
        self.provisional_commands.insert(name);
        self.provisional_inputs.insert(name, input);
        Ok(Outcome::success())
    }

    /// One-argument SelfController setters. A bool is one byte (domain) or a
    /// zero-padded word (plain CMIF); u32 enums are one word. Values that are
    /// not booleans, or nonzero padding, stop explicitly.
    fn self_controller_setting(
        &mut self,
        application: &MainApplicationIdentity,
        call: &mut Call<'_>,
    ) -> Result<Outcome, Trap> {
        let command = call.request.command_id;
        let is_u32 = matches!(command, 10 | 19);
        let length = if is_u32 || call.request.arguments.len() == 4 {
            4
        } else {
            1
        };
        call.shape(length, false, false)?;
        if self.main_application.as_ref() != Some(application) {
            return Err(invalid(
                "self-controller owner must match the trusted bootstrap main application",
            ));
        }
        let arguments = &call.request.arguments;
        let value = u32::from_le_bytes([
            arguments[0],
            *arguments.get(1).unwrap_or(&0),
            *arguments.get(2).unwrap_or(&0),
            *arguments.get(3).unwrap_or(&0),
        ]);
        if !is_u32 && value > 1 {
            return Err(invalid("SelfController boolean argument is not 0 or 1"));
        }
        let state = self
            .main_application_state
            .as_ref()
            .ok_or_else(|| invalid("self-controller setting requires application state"))?;
        let mut settings = state.self_settings;
        let flag = value != 0;
        match command {
            10 => settings.screenshot_permission = Some(value),
            11 => settings.operation_mode_changed_notification = Some(flag),
            12 => settings.performance_mode_changed_notification = Some(flag),
            14 => settings.restart_message_enabled = Some(flag),
            16 => settings.out_of_focus_suspending_enabled = Some(flag),
            19 => settings.album_image_orientation = Some(value),
            _ => settings.handles_request_to_display = Some(flag),
        }
        call.reply(0, None, SELF_CONTROLLER_INTERFACE_ID, &[])?;
        // No operation/performance-mode change, resume or suspension source is
        // modeled, so none of these settings produces a lifecycle message.
        if let Some(state) = self.main_application_state.as_mut() {
            state.self_settings = settings;
        }
        Ok(Outcome::success())
    }

    /// SetFocusHandlingMode (13): three u8 bools (notify, background, suspend).
    /// Eden 1a6d7d4 sets focus-change notification to `notify`, updates the
    /// focus-handling mode from `suspend`, then re-evaluates the lifecycle
    /// system event. Non-boolean bytes stop explicitly instead of guessing.
    fn set_focus_handling_mode(
        &mut self,
        application: &MainApplicationIdentity,
        call: &mut Call<'_>,
    ) -> Result<Outcome, Trap> {
        // Domain messages carry the exact 3-byte size; plain CMIF rounds raw
        // data up to a whole word, so one trailing padding byte is accepted.
        let length = if call.request.arguments.len() == 4 {
            4
        } else {
            3
        };
        call.shape(length, false, false)?;
        if self.main_application.as_ref() != Some(application) {
            return Err(invalid(
                "self-controller owner must match the trusted bootstrap main application",
            ));
        }
        let bytes = &call.request.arguments[..3];
        if bytes.iter().any(|byte| *byte > 1) {
            return Err(invalid(
                "SetFocusHandlingMode argument is not a boolean byte",
            ));
        }
        let request = FocusHandlingRequest {
            notify: bytes[0] != 0,
            background: bytes[1] != 0,
            suspend: bytes[2] != 0,
        };
        let state = self
            .main_application_state
            .as_ref()
            .ok_or_else(|| invalid("focus handling requires initialized application state"))?;
        let event_id = state.lifecycle_event;
        if !self.readable_events.contains_key(&event_id) {
            return Err(invalid("lifecycle event backing is missing"));
        }
        let mut lifecycle = state.lifecycle.clone();
        lifecycle.focus_state_changed_notification_enabled = request.notify;
        lifecycle.focus_handling = Some(request);
        call.reply(0, None, SELF_CONTROLLER_INTERFACE_ID, &[])?;
        let signaled = lifecycle.refresh_availability();
        if let Some(state) = self.main_application_state.as_mut() {
            state.lifecycle = lifecycle;
        }
        if let Some(event) = self.readable_events.get_mut(&event_id) {
            event.signaled = signaled;
        }
        Ok(Outcome::success())
    }

    /// GetGraphicsSupervisor (2000): creates the application's graphics
    /// supervisor session and copies its client handle. Eden 1a6d7d4 caches the
    /// client session and copies the same session on later calls; a second
    /// handle aliasing one session is not modeled, so a repeat call stops.
    fn get_graphics_supervisor(
        &mut self,
        application: &MainApplicationIdentity,
        call: &mut Call<'_>,
    ) -> Result<Outcome, Trap> {
        call.shape(0, false, false)?;
        if self.main_application.as_ref() != Some(application) {
            return Err(invalid(
                "self-controller owner must match the trusted bootstrap main application",
            ));
        }
        if self.graphics_supervisor_handle.is_some() {
            return Err(call.unsupported(
                "appletOE:self_controller",
                "a second graphics-supervisor handle aliasing one session is not modeled",
            ));
        }
        let handle = self.next_available_handle()?;
        let reply = call.prepare_copy_handle_reply(handle, SELF_CONTROLLER_INTERFACE_ID)?;
        call.write_reply(&reply)?;
        self.sessions.insert(
            handle,
            Session::regular(ServiceObject::GraphicsSupervisor {
                application: application.clone(),
            }),
        );
        self.graphics_supervisor_handle = Some(handle);
        self.graphics_supervisor_phase = Some(GraphicsSupervisorPhase::Created);
        self.next_handle = handle + 1;
        Ok(Outcome {
            output_copy_handle_count: 1,
            ..Outcome::success()
        })
    }

    /// IGraphicsSupervisor speaks TIPC. Only the observed bootstrap sequence is
    /// modeled; every other input stops explicitly rather than guessing a
    /// result code for a reverse-engineered protocol.
    /// DisplaySession (TIPC; Switchbrew lists command IDs without names or
    /// I/O). Commands are implemented as the application reaches them.
    fn display_session(&mut self, call: &mut Call<'_>) -> Result<Outcome, Trap> {
        const SERVICE: &str = "appletOE:display_session";
        if call.request.protocol != Protocol::Tipc {
            return Err(call.unsupported(SERVICE, "display session commands are TIPC"));
        }
        let mut updated = self.sessions[&call.handle].clone();
        let Some(ServiceObject::DisplaySession { state }) = updated.object_mut(None) else {
            return Err(invalid(
                "display session target is not a regular display session",
            ));
        };
        // GetNotifications (21): copies the 32 pending notification words to
        // one output buffer and clears them and the event (Eden 1a6d7d4).
        if call.request.command_id == 21 {
            call.shape_output(0, false)?;
            let (address, size) = call.output_buffer()?;
            if size < 128 {
                return Err(call.unsupported(SERVICE, "notification buffer is under 128 bytes"));
            }
            let mut words = state.notifications.clone();
            words.resize(32, 0);
            let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
            call.memory.write(address, &bytes)?;
            call.reply(0, None, 0, &[])?;
            state.notifications.clear();
            if let Some(id) = state.notification_event {
                if let Some(event) = self.readable_events.get_mut(&ReadableEventId(id)) {
                    event.signaled = false;
                }
            }
            self.sessions.insert(call.handle, updated);
            self.provisional_commands.insert("appletOE:display_session");
            return Ok(Outcome::success());
        }
        // 32: u32 index and a copy of command 31's event for it; moves another
        // handle to that same event. Undocumented, inferred from the SDK
        // wrapper (sdk+0x5e3af0) that replaces its stored handle with the
        // reply's. Provisional.
        if call.request.command_id == 32 {
            call.shape_with_copies(4, false, false, 1)?;
            let index = crate::bytes::le_u32(&call.request.arguments[..4]);
            let copied = self
                .readable_event_handles
                .get(&call.request.copy_handles[0])
                .copied();
            let Some(id) = copied.filter(|id| state.events_31.get(&index) == Some(&id.0)) else {
                return Err(call.unsupported(
                    SERVICE,
                    "command 32 handle is not this session's command 31 event",
                ));
            };
            let handle = self.next_available_handle()?;
            call.reply(0, Some(handle), 0, &[])?;
            self.readable_event_handles.insert(handle, id);
            self.next_handle = handle + 1;
            self.sessions.insert(call.handle, updated);
            self.provisional_commands.insert("appletOE:display_session");
            return Ok(Outcome::success());
        }
        if call.request.pid.is_some()
            || !call.request.copy_handles.is_empty()
            || call.request.input_buffer.is_some()
            || !call.request.receive_buffers.is_empty()
            || !call.request.receive_list.is_empty()
        {
            return Err(call.unsupported(
                SERVICE,
                format!(
                    "display session command carries buffers ({})",
                    call.request.describe()
                ),
            ));
        }
        // GetNotificationEvent (11) / ReleaseNotificationEvent (13): u32 index
        // 0 (Eden 1a6d7d4). 11 moves a readable event, created once per
        // session and signaled only while notification bits are pending.
        // 31 is undocumented with 11's shape and an event index below 8: the
        // SDK keeps one moved handle per index (sdk+0x5ead6c stores it in
        // an array by index) and closes it on failure; provisional.
        if matches!(call.request.command_id, 11 | 13 | 31) {
            let index = if call.request.arguments.len() == 4 {
                crate::bytes::le_u32(&call.request.arguments[..4])
            } else {
                u32::MAX
            };
            let limit = if call.request.command_id == 31 { 8 } else { 1 };
            if index >= limit {
                return Err(call.unsupported(
                    SERVICE,
                    format!("display event index {index} is not modeled"),
                ));
            }
            let notification = matches!(call.request.command_id, 11 | 13);
            if call.request.command_id == 13 {
                call.reply(0, None, 0, &[])?;
                self.provisional_commands.insert("appletOE:display_session");
                return Ok(Outcome::success());
            }
            let application = self
                .main_application
                .clone()
                .ok_or_else(|| invalid("display session requires the main application"))?;
            let slot = if notification {
                state.notification_event
            } else {
                state.events_31.get(&index).copied()
            };
            let id = match slot {
                Some(id) => ReadableEventId(id),
                None => {
                    if self.readable_events.len() >= MAX_READABLE_EVENTS {
                        return Err(invalid("readable event backing capacity exhausted"));
                    }
                    ReadableEventId(
                        self.readable_events
                            .keys()
                            .map(|id| id.0)
                            .max()
                            .unwrap_or(0)
                            + 1,
                    )
                }
            };
            let pending = notification && state.notifications.iter().any(|bits| *bits != 0);
            if notification {
                state.notification_event = Some(id.0);
            } else {
                state.events_31.insert(index, id.0);
            }
            let handle = self.next_available_handle()?;
            call.reply(0, Some(handle), 0, &[])?;
            let event = self.readable_events.entry(id).or_insert(ReadableEvent {
                owner: application,
                kind: if notification {
                    ReadableEventKind::DisplayNotification
                } else {
                    ReadableEventKind::DisplayEvent31
                },
                signaled: false,
            });
            event.signaled |= pending;
            self.readable_event_handles.insert(handle, id);
            self.next_handle = handle + 1;
            self.sessions.insert(call.handle, updated);
            self.provisional_commands.insert("appletOE:display_session");
            return Ok(Outcome::success());
        }
        let data = state
            .handle(call.request.command_id, &call.request.arguments)
            .map_err(|message| call.unsupported(SERVICE, message))?;
        call.reply(0, None, 0, &data)?;
        self.sessions.insert(call.handle, updated);
        self.provisional_commands.insert("appletOE:display_session");
        Ok(Outcome::success())
    }

    fn graphics_supervisor(
        &mut self,
        application: &MainApplicationIdentity,
        call: &mut Call<'_>,
    ) -> Result<Outcome, Trap> {
        const SERVICE: &str = "appletOE:graphics_supervisor";
        if call.request.protocol != Protocol::Tipc {
            return Err(call.unsupported(SERVICE, "graphics supervisor commands are TIPC"));
        }
        if self.main_application.as_ref() != Some(application) {
            return Err(invalid(
                "graphics supervisor owner must match the trusted bootstrap main application",
            ));
        }
        match call.request.command_id {
            // Initialize: copied process handle, then the applet resource user
            // ID, bootstrap flags and four remote-memory parameters (Eden
            // 1a6d7d4 graphics_supervisor_protocol.h). The flags select
            // bootstrap features that change nothing modeled here (titles send
            // 0x8000000000000005 and ...07); all-zero parameters request no
            // remote memory, so success only advances the session phase.
            4 => {
                call.shape_with_copies(48, false, false, 1)?;
                let word = |at: usize| crate::bytes::le_u64(&call.request.arguments[at..at + 8]);
                let (aruid, flags) = (word(0), word(8));
                let parameters = [word(16), word(24), word(32), word(40)];
                if call.request.copy_handles[0] != CURRENT_PROCESS_PSEUDO_HANDLE {
                    return Err(call.unsupported(
                        SERVICE,
                        "Initialize process handle is not the current-process pseudo-handle",
                    ));
                }
                if self.graphics_supervisor_phase != Some(GraphicsSupervisorPhase::Created) {
                    return Err(call.unsupported(SERVICE, "repeated graphics Initialize"));
                }
                if aruid != application.process_id {
                    return Err(call.unsupported(
                        SERVICE,
                        "Initialize applet resource user ID is not the application's",
                    ));
                }
                if parameters != [0; 4] {
                    return Err(call.unsupported(
                        SERVICE,
                        format!(
                            "Initialize remote-memory parameters {parameters:#x?} (flags \
                             {flags:#x}) are not modeled"
                        ),
                    ));
                }
                call.reply(0, None, 0, &[])?;
                self.graphics_supervisor_phase = Some(GraphicsSupervisorPhase::Initialized);
                Ok(Outcome::success())
            }
            // BindProcess: the copied current-process handle and no data.
            11 => {
                call.shape_with_copies(0, false, false, 1)?;
                if call.request.copy_handles[0] != CURRENT_PROCESS_PSEUDO_HANDLE {
                    return Err(call.unsupported(
                        SERVICE,
                        "BindProcess handle is not the current-process pseudo-handle",
                    ));
                }
                if self.graphics_supervisor_phase != Some(GraphicsSupervisorPhase::Initialized) {
                    return Err(
                        call.unsupported(SERVICE, "BindProcess outside the Initialized phase")
                    );
                }
                call.reply(0, None, 0, &[])?;
                self.graphics_supervisor_phase = Some(GraphicsSupervisorPhase::ProcessBound);
                Ok(Outcome::success())
            }
            // CompleteBootstrap: no data; the supervisor becomes ready.
            101 => {
                call.shape(0, false, false)?;
                if self.graphics_supervisor_phase != Some(GraphicsSupervisorPhase::ProcessBound) {
                    return Err(call
                        .unsupported(SERVICE, "CompleteBootstrap outside the ProcessBound phase"));
                }
                call.reply(0, None, 0, &[])?;
                self.graphics_supervisor_phase = Some(GraphicsSupervisorPhase::Ready);
                Ok(Outcome::success())
            }
            // Context create/initialize pairs with no input or output. 301
            // (CreateDisplayContext) and 401/402 (Create/InitializeGpuContext)
            // are documented on Switchbrew "Switch 2: GSV services"; 1101/1102
            // and 1501/1502 (CreateInterModuleImageContext / Initialize...) are
            // listed without I/O and follow the same pattern, as do 901
            // (CreateVideoDecoderContext) and 1001 (CreateVideoEncoderContext;
            // Switchbrew: "No input/output", unofficial names). No backend object
            // exists for these contexts yet, so each is recorded and reported;
            // the undocumented pairs as provisional. An initialize must follow
            // its create.
            301 | 401 | 402 | 901 | 1001 | 1101 | 1102 | 1501 | 1502 => {
                call.shape(0, false, false)?;
                let command = call.request.command_id;
                if self.graphics_supervisor_phase != Some(GraphicsSupervisorPhase::Ready) {
                    return Err(
                        call.unsupported(SERVICE, "context setup before bootstrap completed")
                    );
                }
                let create = match command {
                    402 => Some(401),
                    1102 => Some(1101),
                    1502 => Some(1501),
                    _ => None,
                };
                if let Some(create) = create {
                    if !self.graphics_contexts.contains(&create) {
                        return Err(
                            call.unsupported(SERVICE, "context initialize before its create")
                        );
                    }
                }
                if self.graphics_contexts.contains(&command) {
                    return Err(call.unsupported(SERVICE, "repeated context setup command"));
                }
                call.reply(0, None, 0, &[])?;
                self.graphics_contexts.insert(command);
                if matches!(command, 1101 | 1102 | 1501 | 1502) {
                    let name: &'static str = match command {
                        1101 => "appletOE:graphics_supervisor:1101",
                        1102 => "appletOE:graphics_supervisor:1102",
                        1501 => "appletOE:graphics_supervisor:1501",
                        _ => "appletOE:graphics_supervisor:1502",
                    };
                    self.provisional_commands.insert(name);
                    self.provisional_inputs.insert(name, Vec::new());
                }
                Ok(Outcome::success())
            }
            // 111 (nn::gs::sdk native memory creation, client sdk+0x48f744):
            // u8 kind and u64 size -> u32 NativeMemoryHandle (sdk+0x48f7cc).
            // Handles count from 1; kind and size are recorded. The memory
            // itself is the application's (CreateNativeMemory takes its
            // pointer); nothing is reserved on the host. Provisional.
            111 => {
                call.shape(12, false, false)?;
                let arguments = &call.request.arguments;
                let kind = arguments[0];
                let size = crate::bytes::le_u64(&arguments[4..12]);
                let handle = self
                    .native_memories
                    .keys()
                    .next_back()
                    .map_or(1, |handle| handle + 1);
                self.native_memories.insert(handle, (kind, size));
                call.reply(0, None, 0, &handle.to_le_bytes())?;
                self.provisional_commands
                    .insert("appletOE:graphics_supervisor:111");
                Ok(Outcome::success())
            }
            // 112: u32 native memory handle from 111; no reply data. It
            // follows the decoder's teardown, the counterpart of 111
            // (inferred): the handle and its bound address are forgotten.
            112 => {
                call.shape(4, false, false)?;
                let handle = crate::bytes::le_u32(&call.request.arguments[..4]);
                if self.native_memories.remove(&handle).is_none() {
                    return Err(
                        call.unsupported(SERVICE, format!("destroy of native memory {handle}"))
                    );
                }
                self.native_memory_addresses.remove(&handle);
                call.reply(0, None, 0, &[])?;
                self.provisional_commands
                    .insert("appletOE:graphics_supervisor:112");
                Ok(Outcome::success())
            }
            // 114 (CreateNativeMemory, client sdk+0x48ec88): u32 handle from
            // 111, u64 address and u64 size of the application's memory, and
            // 16 bytes OoT leaves zero; no reply data. The address is
            // recorded with the handle; the size must match 111's.
            114 => {
                call.shape(36, false, false)?;
                let arguments = &call.request.arguments;
                let handle = crate::bytes::le_u32(&arguments[..4]);
                let address = crate::bytes::le_u64(&arguments[4..12]);
                let size = crate::bytes::le_u64(&arguments[12..20]);
                if self
                    .native_memories
                    .get(&handle)
                    .map(|&(_, expected)| expected)
                    != Some(size)
                {
                    return Err(call.unsupported(
                        SERVICE,
                        format!("native memory {handle} of {size:#x} bytes at {address:#x}"),
                    ));
                }
                self.native_memory_addresses.insert(handle, address);
                call.reply(0, None, 0, &[])?;
                self.provisional_commands
                    .insert("appletOE:graphics_supervisor:114");
                Ok(Outcome::success())
            }
            // InitializeDecoderImagePool (911; Switchbrew name): the 0x40-byte
            // VideoDecoderImagePreference, a u32 and the u32 native memory
            // handle (from 111/114) -> u32 pool (client sdk+0x331fc0).
            // Pools count from 1. FinalizeDecoderImagePool (912): u32 pool,
            // no reply data (inferred from its name and 911's result).
            911 => {
                call.shape(0x48, false, false)?;
                let images = crate::bytes::le_u32(&call.request.arguments[0x40..0x44]);
                let memory = crate::bytes::le_u32(&call.request.arguments[0x44..0x48]);
                if !self.native_memory_addresses.contains_key(&memory)
                    || !(1..=64).contains(&images)
                {
                    return Err(call.unsupported(
                        SERVICE,
                        format!("image pool of {images} images on native memory {memory}"),
                    ));
                }
                // The SDK names image i of the pool by base + i
                // (CalculateNativeImageHandle, sdk+0x477d40), so pools
                // take disjoint handle ranges.
                let pool = self
                    .decoder_image_pools
                    .iter()
                    .next_back()
                    .map_or(1, |(base, (_, count))| base + count);
                self.decoder_image_pools.insert(pool, (memory, images));
                if self.decoder_image_pool_requests.len() < 16 {
                    let preference: String = call.request.arguments[..0x44]
                        .iter()
                        .map(|byte| format!("{byte:02x}"))
                        .collect();
                    let address = self.native_memory_addresses[&memory];
                    let size = self
                        .native_memories
                        .get(&memory)
                        .map_or(0, |&(_, size)| size);
                    self.decoder_image_pool_requests
                        .push(format!("{preference} memory {address:#x}+{size:#x}"));
                }
                call.reply(0, None, 0, &pool.to_le_bytes())?;
                self.provisional_commands
                    .insert("appletOE:graphics_supervisor:911");
                Ok(Outcome::success())
            }
            // 961 (the SDK's ReadDecoderOutputImageNv12 asks it before
            // ReadUserImageNv12, client sdk+0x3321e0): u32 native image
            // handle -> the image's 0x80-byte CpuUserImageDescriptor, read
            // relative to the pool's memory. The picture last decoded into
            // that image is written there as pitch-linear NV12 (Y rows, then
            // interleaved UV rows, each the picture's width apart) and the
            // descriptor says so: kind 0x14 with u32 Y offset at +8, u64 Y
            // pitch at +0x10, u32 UV offset at +0x18 and u64 UV pitch at
            // +0x20 (the kind 0x14 copy at sdk+0x478d9c). Images are equal
            // shares of their pool's memory. Provisional.
            961 => {
                call.shape(4, false, false)?;
                let image = crate::bytes::le_u32(&call.request.arguments[..4]);
                let Some((&base, &(memory, images))) = self
                    .decoder_image_pools
                    .range(..=image)
                    .next_back()
                    .filter(|(base, (_, count))| image - *base < *count)
                else {
                    return Err(call.unsupported(SERVICE, format!("native image {image}")));
                };
                let address = self.native_memory_addresses[&memory];
                let size = self
                    .native_memories
                    .get(&memory)
                    .map_or(0, |&(_, size)| size);
                let share = size / u64::from(images);
                let offset = share * u64::from(image - base);
                let mut descriptor = [0u8; 0x80];
                descriptor[0] = 0x14;
                if let Some(picture) = self.slot_pictures.get(&image) {
                    let (width, height) = (picture.width as usize, picture.height as usize);
                    let chroma_width = width.div_ceil(2);
                    let mut uv = Vec::with_capacity(chroma_width * 2 * height.div_ceil(2));
                    for (u, v) in picture.u.iter().zip(&picture.v) {
                        uv.push(*u);
                        uv.push(*v);
                    }
                    let pitch = chroma_width as u64 * 2;
                    let uv_offset = offset + pitch * height as u64;
                    if uv_offset + uv.len() as u64 > offset + share {
                        return Err(call.unsupported(
                            SERVICE,
                            format!("{width}x{height} picture in a {share:#x}-byte image"),
                        ));
                    }
                    let mut y = Vec::with_capacity(pitch as usize * height);
                    for row in picture.y.chunks(width) {
                        y.extend_from_slice(row);
                        y.resize(y.len() + pitch as usize - width, 0);
                    }
                    call.memory.write(address + offset, &y)?;
                    call.memory.write(address + uv_offset, &uv)?;
                    descriptor[8..12].copy_from_slice(&(offset as u32).to_le_bytes());
                    descriptor[0x10..0x18].copy_from_slice(&pitch.to_le_bytes());
                    descriptor[0x18..0x1c].copy_from_slice(&(uv_offset as u32).to_le_bytes());
                    descriptor[0x20..0x28].copy_from_slice(&pitch.to_le_bytes());
                } else {
                    // No decoded picture (the decoder failed): a zero-pitch
                    // image reads as its first bytes repeated.
                    descriptor[8..12].copy_from_slice(&(offset as u32).to_le_bytes());
                    descriptor[0x18..0x1c].copy_from_slice(&(offset as u32).to_le_bytes());
                }
                call.reply(0, None, 0, &descriptor)?;
                self.provisional_commands
                    .insert("appletOE:graphics_supervisor:961");
                Ok(Outcome::success())
            }
            912 => {
                call.shape(4, false, false)?;
                let pool = crate::bytes::le_u32(&call.request.arguments[..4]);
                if self.decoder_image_pools.remove(&pool).is_none() {
                    return Err(call.unsupported(SERVICE, format!("finalize of image pool {pool}")));
                }
                call.reply(0, None, 0, &[])?;
                self.provisional_commands
                    .insert("appletOE:graphics_supervisor:912");
                Ok(Outcome::success())
            }
            // 906 (undocumented; Switchbrew lists 901..905): u8 flag and the
            // u64 VideoDecoderLibraryVersion (the SDK's
            // InitializeServerContext, client at sdk+0x3379f0); moves a
            // VideoDecoderCreatorSession like 905. Requires 901's context.
            906 => {
                call.shape(12, false, false)?;
                if !self.graphics_contexts.contains(&901) {
                    return Err(call.unsupported(SERVICE, "906 before CreateVideoDecoderContext"));
                }
                let handle = self.next_available_handle()?;
                call.reply(0, Some(handle), 0, &[])?;
                self.sessions
                    .insert(handle, Session::regular(ServiceObject::VideoDecoderCreator));
                self.next_handle = handle + 1;
                self.provisional_commands
                    .insert("appletOE:graphics_supervisor:906");
                Ok(Outcome::success())
            }
            // InitializeVideoEncoderContext (1002): no input; moves a
            // VideoEncoderCreatorSession (Switchbrew, unofficial name).
            1002 => {
                call.shape(0, false, false)?;
                if !self.graphics_contexts.contains(&1001) {
                    return Err(call.unsupported(SERVICE, "1002 before CreateVideoEncoderContext"));
                }
                let handle = self.next_available_handle()?;
                call.reply(0, Some(handle), 0, &[])?;
                self.sessions
                    .insert(handle, Session::regular(ServiceObject::VideoEncoderCreator));
                self.next_handle = handle + 1;
                Ok(Outcome::success())
            }
            // 991: the SDK's nn::codec::CalculateDecoderImagePoolBuffer
            // (client sdk+0x331d24): 0x40-byte VideoDecoderImagePreference
            // and a u32 -> u64 size, u64 alignment (the SDK rounds the size
            // up to the alignment, at least 4 KiB). OoT asks for one
            // 1920x1080 image (words 1 and 2: count, then height - 1 and
            // width - 1 as u16s). No host video decoder exists yet; the size
            // is a host value for 32-bit pixels. Provisional.
            991 => {
                call.shape(0x44, false, false)?;
                let word = |at: usize| crate::bytes::le_u32(&call.request.arguments[at..at + 4]);
                let count = u64::from(word(4).max(1));
                let extent = word(8);
                let (width, height) = (u64::from(extent & 0xffff) + 1, u64::from(extent >> 16) + 1);
                let size = (count * width * height * 4).next_multiple_of(0x1000);
                let mut reply = size.to_le_bytes().to_vec();
                reply.extend_from_slice(&0x1000u64.to_le_bytes());
                call.reply(0, None, 0, &reply)?;
                self.provisional_commands
                    .insert("appletOE:graphics_supervisor:991");
                Ok(Outcome::success())
            }
            // DestroyVideoDecoderContext (903) / DestroyVideoEncoderContext
            // (1003): no input/output (Switchbrew); undo 901 / 1001.
            903 | 1003 => {
                call.shape(0, false, false)?;
                let create = call.request.command_id - 2;
                if !self.graphics_contexts.remove(&create) {
                    return Err(call.unsupported(SERVICE, "destroy of a context never created"));
                }
                call.reply(0, None, 0, &[])?;
                Ok(Outcome::success())
            }
            // GetNvidiaDriver: after image-format setup, moves a new session
            // to the process-bound NVIDIA driver.
            1402 => {
                call.shape(0, false, false)?;
                if self.graphics_supervisor_phase != Some(GraphicsSupervisorPhase::Ready)
                    || !self.graphics_contexts.contains(&1502)
                {
                    return Err(call.unsupported(
                        SERVICE,
                        "GetNvidiaDriver before bootstrap and image-format setup",
                    ));
                }
                let handle = self.next_available_handle()?;
                call.reply(0, Some(handle), 0, &[])?;
                self.sessions.insert(
                    handle,
                    Session::regular(ServiceObject::NvidiaDriver {
                        application: application.clone(),
                    }),
                );
                self.next_handle = handle + 1;
                Ok(Outcome::success())
            }
            // InitializeSyncpointContext: no data; enables GetSyncpointSession.
            201 => {
                call.shape(0, false, false)?;
                if self.graphics_supervisor_phase != Some(GraphicsSupervisorPhase::Ready) {
                    return Err(call.unsupported(SERVICE, "syncpoint context before bootstrap"));
                }
                call.reply(0, None, 0, &[])?;
                self.syncpoint_context_initialized = true;
                Ok(Outcome::success())
            }
            // GetSyncpointSession: moves a new syncpoint-supervisor session.
            202 => {
                call.shape(0, false, false)?;
                if !self.syncpoint_context_initialized || self.nvdrv_sessions() == 0 {
                    return Err(call.unsupported(
                        SERVICE,
                        "GetSyncpointSession before the syncpoint context and NVIDIA driver",
                    ));
                }
                let handle = self.next_available_handle()?;
                call.reply(0, Some(handle), 0, &[])?;
                self.sessions
                    .insert(handle, Session::regular(ServiceObject::SyncpointSupervisor));
                self.next_handle = handle + 1;
                Ok(Outcome::success())
            }
            // BindSyncpointMemory: two copied transfer memories (request page
            // 0x1000, value page 0x2000) and their sizes; moves the request
            // event's writable side and the completion event's readable side.
            204 => {
                call.shape_with_copies(24, false, false, 2)?;
                let arguments = &call.request.arguments;
                let word = |at: usize| crate::bytes::le_u32(&arguments[at..at + 4]);
                let quad = |at: usize| crate::bytes::le_u64(&arguments[at..at + 8]);
                if (word(0), word(4), quad(8), quad(16)) != (1, 0x1ff, 0x1000, 0x2000) {
                    return Err(call.unsupported(
                        SERVICE,
                        "syncpoint memory parameters are not the observed layout",
                    ));
                }
                if !self.syncpoint_context_initialized
                    || self.nvdrv_sessions() == 0
                    || self.syncpoint_cache.is_some()
                {
                    return Err(call.unsupported(SERVICE, "BindSyncpointMemory in the wrong state"));
                }
                let (rw_handle, ro_handle) =
                    (call.request.copy_handles[0], call.request.copy_handles[1]);
                let (Some(rw), Some(ro)) = (
                    self.transfer_memory(rw_handle),
                    self.transfer_memory(ro_handle),
                ) else {
                    return Err(call.unsupported(
                        SERVICE,
                        "syncpoint memory handles are not transfer memories",
                    ));
                };
                if rw_handle == ro_handle || rw.size != 0x1000 || ro.size != 0x2000 {
                    return Err(call
                        .unsupported(SERVICE, "syncpoint transfer memories have the wrong sizes"));
                }
                if self.readable_events.len() + 2 > MAX_READABLE_EVENTS {
                    return Err(invalid("readable event backing capacity exhausted"));
                }
                let request_handle = self.next_available_handle()?;
                let completed_handle = request_handle
                    .checked_add(1)
                    .filter(|handle| !self.sessions.contains_key(handle))
                    .ok_or_else(|| invalid("no second handle for the syncpoint events"))?;
                let reply = ipc::build_tipc_reply_with_moves(
                    call.request,
                    &[request_handle, completed_handle],
                    &[],
                )
                .map_err(|error| invalid(format!("build syncpoint reply: {error:#}")))?;
                // Server initialization of both pages precedes the reply.
                call.memory.initialize(rw.address, &[0; 0x1000])?;
                call.memory.initialize(ro.address, &[0; 0x2000])?;
                let mut header = Vec::with_capacity(32);
                for value in [
                    1u32,
                    0,
                    (SYNCPOINT_CACHE_COUNTERS - 1) as u32,
                    SYNCPOINT_CACHE_VALUES as u32,
                    SYNCPOINT_CACHE_GROUP_REQUEST as u32,
                    SYNCPOINT_CACHE_GROUP_ACK as u32,
                    SYNCPOINT_CACHE_SLOT_REQUEST as u32,
                    SYNCPOINT_CACHE_SLOT_ACK as u32,
                ] {
                    header.extend_from_slice(&value.to_le_bytes());
                }
                call.memory.initialize(ro.address, &header)?;
                let values: Vec<u8> = (0..SYNCPOINT_CACHE_COUNTERS)
                    .flat_map(|_| SYNCPOINT_UNALLOCATED.to_le_bytes())
                    .collect();
                call.memory
                    .initialize(ro.address + SYNCPOINT_CACHE_VALUES, &values)?;
                call.write_reply(&reply)?;
                let next_id = self
                    .readable_events
                    .keys()
                    .map(|id| id.0)
                    .max()
                    .unwrap_or(0)
                    + 1;
                let (request_event, completed_event) =
                    (ReadableEventId(next_id), ReadableEventId(next_id + 1));
                for (id, kind) in [
                    (request_event, ReadableEventKind::SyncpointCacheRequest),
                    (completed_event, ReadableEventKind::SyncpointCacheCompleted),
                ] {
                    self.readable_events.insert(
                        id,
                        ReadableEvent {
                            owner: application.clone(),
                            kind,
                            signaled: false,
                        },
                    );
                }
                self.writable_event_handles
                    .insert(request_handle, request_event);
                self.readable_event_handles
                    .insert(completed_handle, completed_event);
                self.next_handle = completed_handle + 1;
                self.syncpoint_cache = Some(SyncpointCache {
                    rw,
                    ro,
                    rw_handle,
                    ro_handle,
                    request_event,
                    completed_event,
                    refreshes: 0,
                });
                Ok(Outcome::success())
            }
            // InitializeDisplayContext (Switchbrew "Switch 2: GSV services"):
            // u64 Version, moves a DisplaySession; 19.0.0 requires 0x300010000.
            // 0x300020000 is what newer SDKs send (Eden 1a6d7d4) and OoT sends
            // 0x300040000; both are provisional, as each display-session
            // command is still checked on its own.
            // Requires CreateDisplayContext (301).
            305 => {
                call.shape(8, false, false)?;
                let version = crate::bytes::le_u64(&call.request.arguments[..8]);
                if !matches!(version, 0x3_0001_0000 | 0x3_0002_0000 | 0x3_0004_0000) {
                    return Err(call.unsupported(
                        SERVICE,
                        format!("display session version {version:#x} is not known"),
                    ));
                }
                if !self.graphics_contexts.contains(&301) {
                    return Err(call.unsupported(SERVICE, "display session before context 301"));
                }
                let handle = self.next_available_handle()?;
                call.reply(0, Some(handle), 0, &[])?;
                self.sessions.insert(
                    handle,
                    Session::regular(ServiceObject::DisplaySession {
                        state: Default::default(),
                    }),
                );
                self.next_handle = handle + 1;
                if version != 0x3_0001_0000 {
                    self.provisional_commands
                        .insert("appletOE:graphics_supervisor:305");
                }
                Ok(Outcome::success())
            }
            // CreateGpuSessionWithRemoteMemory (Switchbrew "Switch 2: GSV
            // services"): u64 version, moves a GpuSession. Version 2 is
            // documented for 19.0.0; 4 is what newer firmware's SDK sends
            // (Eden 1a6d7d4). This application opens it without the
            // documented 401/402 GPU-context pair.
            497 => {
                call.shape(8, false, false)?;
                let version = crate::bytes::le_u64(&call.request.arguments[..8]);
                if !matches!(version, 2 | 4) {
                    return Err(call.unsupported(SERVICE, "GPU session version is not 2 or 4"));
                }
                if self.graphics_supervisor_phase != Some(GraphicsSupervisorPhase::Ready)
                    || self.nvdrv_sessions() == 0
                {
                    return Err(call.unsupported(
                        SERVICE,
                        "GPU session before bootstrap and the NVIDIA driver",
                    ));
                }
                let handle = self.next_available_handle()?;
                call.reply(0, Some(handle), 0, &[])?;
                self.sessions
                    .insert(handle, Session::regular(ServiceObject::GpuSession));
                self.next_handle = handle + 1;
                self.gpu_session_version = Some(version);
                Ok(Outcome::success())
            }
            // Command 115 (undocumented on Switchbrew) takes a u64 memory-mode
            // request and returns the granted mode. Replying 0 clears the
            // optional kernel-GPU ("gpuk") memory mode, so the application
            // uses the NVIDIA driver's nvmap path, which Switchbrew documents.
            // Same policy as Eden 1a6d7d4; reported as provisional.
            115 => {
                call.shape(8, false, false)?;
                let requested = crate::bytes::le_u64(&call.request.arguments[..8]);
                if !matches!(requested, 0 | 2)
                    || self.graphics_supervisor_phase != Some(GraphicsSupervisorPhase::Ready)
                {
                    return Err(call.unsupported(
                        SERVICE,
                        "native memory mode request is not the observed one",
                    ));
                }
                call.reply(0, None, 0, &0u64.to_le_bytes())?;
                self.provisional_commands
                    .insert("appletOE:graphics_supervisor:115");
                self.provisional_inputs.insert(
                    "appletOE:graphics_supervisor:115",
                    requested.to_le_bytes().to_vec(),
                );
                Ok(Outcome::success())
            }
            // ExistsAddressSpace (434, Switchbrew "Switch 2: GSV services"):
            // a u32 GpuSession address-space handle; the SDK (sdk+0x46c428)
            // reads only the result. Success for a created address space.
            434 => {
                call.shape(4, false, false)?;
                let handle = crate::bytes::le_u32(&call.request.arguments[..4]);
                if !self.gpu_address_spaces.contains_key(&handle) {
                    return Err(call.unsupported(
                        SERVICE,
                        format!("ExistsAddressSpace for unknown handle {handle}"),
                    ));
                }
                call.reply(0, None, 0, &[])?;
                Ok(Outcome::success())
            }
            // Undocumented query with no input and a u32 output. The SDK
            // (OoT, sdk+0x491660) issues command 120 next only when the
            // value is 1; 0 skips that undocumented follow-up. Provisional.
            126 => {
                call.shape(0, false, false)?;
                if self.graphics_supervisor_phase != Some(GraphicsSupervisorPhase::Ready) {
                    return Err(call.unsupported(SERVICE, "command 126 before bootstrap completes"));
                }
                call.reply(0, None, 0, &0u32.to_le_bytes())?;
                self.provisional_commands
                    .insert("appletOE:graphics_supervisor:126");
                Ok(Outcome::success())
            }
            _ => Err(call.unsupported(
                SERVICE,
                format!(
                    "graphics supervisor command is not implemented ({})",
                    call.request.describe()
                ),
            )),
        }
    }

    /// GpuSession (TIPC; Switchbrew "Switch 2: GSV services"): notification
    /// events by ID and the KUserWorker session handshake.
    fn gpu_session(&mut self, call: &mut Call<'_>) -> Result<Outcome, Trap> {
        const SERVICE: &str = "appletOE:gpu_session";
        if call.request.protocol != Protocol::Tipc {
            return Err(call.unsupported(SERVICE, "GPU session commands are TIPC"));
        }
        let application = self
            .main_application
            .clone()
            .ok_or_else(|| invalid("GPU session requires the main application"))?;
        let event_id_argument = |call: &Call<'_>| -> Result<u32, Trap> {
            call.shape(4, false, false)?;
            let id = crate::bytes::le_u32(&call.request.arguments[..4]);
            if id > GPU_NOTIFICATION_MAX_ID {
                return Err(call.unsupported(SERVICE, "event ID outside GetEventIdRange"));
            }
            Ok(id)
        };
        match call.request.command_id {
            // GetEventIdRange: two u32s (min, max). The bounds are not
            // documented; 0..=63 matches the 64 event slots of the nvhost-ctrl
            // event interface and Eden 1a6d7d4. Reported as provisional.
            11 => {
                call.shape(0, false, false)?;
                let mut data = 0u32.to_le_bytes().to_vec();
                data.extend_from_slice(&GPU_NOTIFICATION_MAX_ID.to_le_bytes());
                call.reply(0, None, 0, &data)?;
                self.provisional_commands.insert("appletOE:gpu_session:11");
                self.provisional_inputs
                    .insert("appletOE:gpu_session:11", Vec::new());
                Ok(Outcome::success())
            }
            // CreateEvent(u32 EventId): moves the readable side of that ID's
            // event; repeated calls alias the same event.
            12 => {
                let index = event_id_argument(call)?;
                let id = match self.gpu_events.get(&index) {
                    Some(id) => *id,
                    None => {
                        if self.readable_events.len() >= MAX_READABLE_EVENTS {
                            return Err(invalid("readable event backing capacity exhausted"));
                        }
                        ReadableEventId(
                            self.readable_events
                                .keys()
                                .map(|id| id.0)
                                .max()
                                .unwrap_or(0)
                                + 1,
                        )
                    }
                };
                let handle = self.next_available_handle()?;
                call.reply(0, Some(handle), 0, &[])?;
                self.readable_events.entry(id).or_insert(ReadableEvent {
                    owner: application,
                    kind: ReadableEventKind::GpuNotification { index },
                    signaled: false,
                });
                self.gpu_events.insert(index, id);
                self.readable_event_handles.insert(handle, id);
                self.next_handle = handle + 1;
                Ok(Outcome::success())
            }
            // RequestKUserWorkerSession(u32 EventId from CreateEvent): the
            // event is signaled when the worker session is ready. This server
            // creates it immediately.
            21 => {
                let index = event_id_argument(call)?;
                let Some(&id) = self.gpu_events.get(&index) else {
                    return Err(
                        call.unsupported(SERVICE, "worker requested with an uncreated event ID")
                    );
                };
                call.reply(0, None, 0, &[])?;
                self.gpu_worker_ready = true;
                if let Some(event) = self.readable_events.get_mut(&id) {
                    event.signaled = true;
                }
                Ok(Outcome::success())
            }
            // GetKUserWorkerSession: moves the worker session once ready.
            22 => {
                call.shape(0, false, false)?;
                if !self.gpu_worker_ready {
                    return Err(call
                        .unsupported(SERVICE, "worker session acquired before it was requested"));
                }
                let handle = self.next_available_handle()?;
                call.reply(0, Some(handle), 0, &[])?;
                self.sessions
                    .insert(handle, Session::regular(ServiceObject::GpuWorker));
                self.next_handle = handle + 1;
                Ok(Outcome::success())
            }
            // CreateGpuAddressSpace (33; reached from
            // nn::gpu::sdk::CreateGpuAddressSpace, IPC at sdk+0x46c360): a u32
            // (0 here, read as creation flags) in, a u32 address-space handle
            // out. Handles count from 1. Provisional.
            33 => {
                call.shape(4, false, false)?;
                let flags = crate::bytes::le_u32(&call.request.arguments[..4]);
                if flags != 0 {
                    return Err(call.unsupported(
                        SERVICE,
                        format!("address space creation flags {flags:#x} are not modeled"),
                    ));
                }
                if self.gpu_address_spaces.len() >= 64 {
                    return Err(call.unsupported(SERVICE, "too many GPU address spaces"));
                }
                let handle = self
                    .gpu_address_spaces
                    .keys()
                    .next_back()
                    .map_or(1, |last| last + 1);
                call.reply(0, None, 0, &handle.to_le_bytes())?;
                self.gpu_address_spaces.insert(handle, flags);
                self.provisional_commands.insert("appletOE:gpu_session:33");
                Ok(Outcome::success())
            }
            // GetGsvPartitionGpuAddress (202) and 203 (the SDK reads it as
            // the user partition's range list, sdk+0x46c4d0): a u32 address-
            // space handle in. 202 returns one (start, end) u64 pair; 203 a
            // u64 count (under 5) and four pairs, of which the SDK uses the
            // first. Ranges follow Eden 1a6d7d4's extended GPU address-space
            // layout (gpu_address_space_layout.h): user [0x8000000, split),
            // supervisor [split, 1 << 40). Provisional.
            202 | 203 => {
                call.shape(4, false, false)?;
                let handle = crate::bytes::le_u32(&call.request.arguments[..4]);
                if !self.gpu_address_spaces.contains_key(&handle) {
                    return Err(call.unsupported(
                        SERVICE,
                        format!("GPU address space {handle} does not exist"),
                    ));
                }
                let mut data = Vec::new();
                if call.request.command_id == 202 {
                    data.extend_from_slice(&GPU_VA_SPLIT.to_le_bytes());
                    data.extend_from_slice(&GPU_VA_END.to_le_bytes());
                } else {
                    data.extend_from_slice(&1u64.to_le_bytes());
                    data.extend_from_slice(&GPU_VA_USER_START.to_le_bytes());
                    data.extend_from_slice(&GPU_VA_SPLIT.to_le_bytes());
                    data.resize(8 + 4 * 16, 0);
                }
                call.reply(0, None, 0, &data)?;
                self.provisional_commands
                    .insert(if call.request.command_id == 202 {
                        "appletOE:gpu_session:202"
                    } else {
                        "appletOE:gpu_session:203"
                    });
                Ok(Outcome::success())
            }
            // CreateGraphicsDevice (41; nn::gpu::sdk::CreateGraphicsDevice,
            // sdk+0x467ff0): the device name packed into a u64 ("NvRmGpu"),
            // a u64 parameter and a u64 debug label; returns a u8 device ID
            // that the SDK stores in its GraphicsDeviceType. IDs count from 1.
            // Undocumented beyond the name; provisional.
            41 => {
                call.shape(24, false, false)?;
                let word = |at: usize| crate::bytes::le_u64(&call.request.arguments[at..at + 8]);
                let device = [word(0), word(8), word(16)];
                if device[0] != u64::from_le_bytes(*b"NvRmGpu\0") {
                    return Err(call.unsupported(
                        SERVICE,
                        format!("graphics device name {:#x} is not NvRmGpu", device[0]),
                    ));
                }
                let id = (1..=u8::MAX)
                    .find(|id| !self.graphics_devices.contains_key(id))
                    .ok_or_else(|| call.unsupported(SERVICE, "graphics device IDs exhausted"))?;
                call.reply(0, None, 0, &u32::from(id).to_le_bytes())?;
                self.graphics_devices.insert(id, device);
                self.provisional_commands.insert("appletOE:gpu_session:41");
                Ok(Outcome::success())
            }
            _ => Err(call.unsupported(
                SERVICE,
                format!(
                    "GPU session command is not implemented ({})",
                    call.request.describe()
                ),
            )),
        }
    }

    /// fsp-srv IFileSystemProxy (Switchbrew "Filesystem services").
    fn file_system_proxy(&mut self, call: &mut Call<'_>) -> Result<Outcome, Trap> {
        const SERVICE: &str = "fsp-srv";
        match call.request.command_id {
            // SetCurrentProcess(u64 ProcessId) with the PID descriptor; the
            // kernel-supplied PID identifies the caller.
            1 => {
                call.shape(8, true, false)?;
                let application = self
                    .main_application
                    .as_ref()
                    .ok_or_else(|| invalid("fsp-srv requires the main application"))?;
                if call.process_id != application.process_id {
                    return Err(call.unsupported(SERVICE, "SetCurrentProcess from another process"));
                }
                call.reply(0, None, FSP_INTERFACE_ID, &[])?;
                self.fs_current_process = Some(call.process_id);
                Ok(Outcome::success())
            }
            // OpenDataStorageByCurrentProcess: an IStorage over the
            // application's RomFS (the selected update's resolved RomFS).
            200 => {
                call.shape(0, false, false)?;
                if self.fs_current_process.is_none() {
                    return Err(call.unsupported(SERVICE, "data storage before SetCurrentProcess"));
                }
                if self.data_romfs.is_none() {
                    return Err(call.unsupported(
                        SERVICE,
                        "no RomFS is configured (set ECHORA_ROMFS to the extracted RomFS)",
                    ));
                }
                self.reply_new_object(call, ServiceObject::DataStorage, FSP_INTERFACE_ID)
            }
            // OpenSaveDataFileSystem (51): u8 SaveDataSpaceId, padding, then
            // SaveDataAttribute {u64 program id, Uid user (16), u64 system
            // save id, u8 type, u8 rank, u16 index, ...}. Each save is a
            // host directory under ECHORA_SAVE_DIR (default
            // $ECHORA_WORK/save): <program>/<type>-<user>-<index>.
            51 => {
                call.shape(72, false, false)?;
                let arguments = &call.request.arguments;
                let attribute = &arguments[8..];
                let program = crate::bytes::le_u64(&attribute[0..8]);
                let user = crate::bytes::le_u128(&attribute[8..24]);
                let (kind, index) = (
                    attribute[32],
                    u16::from_le_bytes([attribute[34], attribute[35]]),
                );
                let program = if program == 0 {
                    self.main_application
                        .as_ref()
                        .map(|application| application.program_id)
                        .unwrap_or(0)
                } else {
                    program
                };
                let base = std::env::var_os("ECHORA_SAVE_DIR")
                    .map(std::path::PathBuf::from)
                    .or_else(|| {
                        std::env::var_os("ECHORA_WORK")
                            .map(|work| std::path::PathBuf::from(work).join("save"))
                    })
                    .ok_or_else(|| {
                        call.unsupported(
                            SERVICE,
                            "set ECHORA_SAVE_DIR or ECHORA_WORK for save data",
                        )
                    })?;
                let root = base
                    .join(format!("{program:016x}"))
                    .join(format!("{kind}-{user:032x}-{index}"));
                let file_system = crate::fs::HostFileSystem::new(&root).map_err(|error| {
                    invalid(format!("create save directory {}: {error}", root.display()))
                })?;
                let id = self.file_systems.len();
                self.file_systems.push(file_system);
                self.reply_new_object(call, ServiceObject::FileSystem { id }, FSP_INTERFACE_ID)
            }
            // SetGlobalAccessLogMode (1004, u32) / GetGlobalAccessLogMode
            // (1005, u32 out): the filesystem access-log mode, 0 (off) by
            // default. No access log is produced here; the mode is stored.
            1004 => {
                call.shape(4, false, false)?;
                let mode = crate::bytes::le_u32(&call.request.arguments[..4]);
                call.reply(0, None, FSP_INTERFACE_ID, &[])?;
                self.fs_access_log_mode = mode;
                Ok(Outcome::success())
            }
            1005 => {
                call.shape(0, false, false)?;
                call.reply(
                    0,
                    None,
                    FSP_INTERFACE_ID,
                    &self.fs_access_log_mode.to_le_bytes(),
                )?;
                Ok(Outcome::success())
            }
            _ => Err(call.unsupported(SERVICE, "filesystem proxy command is not implemented")),
        }
    }

    /// IFileSystem / IFile / IDirectory over a host directory (yuzu
    /// numbering). Paths arrive in X buffers; failures reply fs results.
    fn host_file_system(
        &mut self,
        object: &ServiceObject,
        call: &mut Call<'_>,
    ) -> Result<Outcome, Trap> {
        use crate::fs;
        let command = call.request.command_id;
        let fail = |call: &mut Call<'_>, code: u32| -> Result<Outcome, Trap> {
            call.reply(code, None, FSP_INTERFACE_ID, &[])?;
            Ok(Outcome::result(code))
        };
        let path = |call: &Call<'_>| -> Result<String, Trap> {
            let input = call
                .request
                .input_buffer
                .as_ref()
                .ok_or_else(|| invalid("file-system command without a path buffer"))?;
            let mut bytes = vec![0u8; input.size.min(0x301) as usize];
            call.memory.read(input.address, &mut bytes)?;
            Ok(fs::guest_path(&bytes))
        };
        let word = |call: &Call<'_>, offset: usize| {
            crate::bytes::le_u32(&call.request.arguments[offset..offset + 4])
        };
        let dword = |call: &Call<'_>, offset: usize| {
            crate::bytes::le_u64(&call.request.arguments[offset..offset + 8])
        };
        match (object, command) {
            (ServiceObject::FileSystem { id }, _) => {
                let file_system = self.file_systems[*id].clone();
                let result = match command {
                    // CreateFile(u32 option, u64 size; X path).
                    0 => {
                        call.shape(16, false, true)?;
                        file_system.create_file(&path(call)?, dword(call, 8))
                    }
                    1 => {
                        call.shape(0, false, true)?;
                        file_system.delete_file(&path(call)?)
                    }
                    2 => {
                        call.shape(0, false, true)?;
                        file_system.create_directory(&path(call)?)
                    }
                    3 | 4 => {
                        call.shape(0, false, true)?;
                        file_system.delete_directory(&path(call)?, command == 4)
                    }
                    // GetEntryType(X path) -> u32.
                    7 => {
                        call.shape(0, false, true)?;
                        match file_system.entry_type(&path(call)?) {
                            Ok(kind) => {
                                call.reply(0, None, FSP_INTERFACE_ID, &kind.to_le_bytes())?;
                                return Ok(Outcome::success());
                            }
                            Err(code) => Err(code),
                        }
                    }
                    // OpenFile(u32 mode; X path) -> IFile.
                    8 => {
                        call.shape(4, false, true)?;
                        match file_system.open_file(&path(call)?) {
                            Ok(host) => {
                                let id = self.open_files.len();
                                self.open_files.push(host);
                                return self.reply_new_object(
                                    call,
                                    ServiceObject::File { id },
                                    FSP_INTERFACE_ID,
                                );
                            }
                            Err(code) => Err(code),
                        }
                    }
                    // OpenDirectory(u32 filter; X path) -> IDirectory.
                    9 => {
                        call.shape(4, false, true)?;
                        match file_system.list(&path(call)?, word(call, 0)) {
                            Ok(entries) => {
                                let id = self.open_directories.len();
                                self.open_directories.push((entries, 0));
                                return self.reply_new_object(
                                    call,
                                    ServiceObject::Directory { id },
                                    FSP_INTERFACE_ID,
                                );
                            }
                            Err(code) => Err(code),
                        }
                    }
                    // Commit: host writes are already durable.
                    10 => {
                        call.shape(0, false, false)?;
                        Ok(())
                    }
                    // GetFreeSpaceSize / GetTotalSpaceSize(X path) -> u64:
                    // a fixed 512 MiB (inference; saves have no host quota).
                    11 | 12 => {
                        call.shape(0, false, true)?;
                        call.reply(0, None, FSP_INTERFACE_ID, &(512u64 << 20).to_le_bytes())?;
                        return Ok(Outcome::success());
                    }
                    13 => {
                        call.shape(0, false, true)?;
                        file_system.clean_directory(&path(call)?)
                    }
                    _ => {
                        return Err(call.unsupported(
                            object.service(),
                            "file-system command is not implemented",
                        ));
                    }
                };
                match result {
                    Ok(()) => {
                        call.reply(0, None, FSP_INTERFACE_ID, &[])?;
                        Ok(Outcome::success())
                    }
                    Err(code) => fail(call, code),
                }
            }
            (ServiceObject::File { id }, _) => {
                let host = self.open_files[*id].clone();
                match command {
                    // Read(u32 option, u64 offset, u64 size; B out) -> u64.
                    0 => {
                        call.shape_output(24, false)?;
                        let (offset, size) = (dword(call, 8), dword(call, 16));
                        let (address, capacity) = call.output_buffer()?;
                        let length = size.min(capacity) as usize;
                        match fs::read_at(&host, offset, length) {
                            Ok(bytes) => {
                                call.memory.write(address, &bytes)?;
                                call.reply(
                                    0,
                                    None,
                                    FSP_INTERFACE_ID,
                                    &(bytes.len() as u64).to_le_bytes(),
                                )?;
                                Ok(Outcome::success())
                            }
                            Err(code) => fail(call, code),
                        }
                    }
                    // Write(u32 option, u64 offset, u64 size; A in).
                    1 => {
                        call.shape(24, false, true)?;
                        let (offset, size) = (dword(call, 8), dword(call, 16));
                        let input = call
                            .request
                            .input_buffer
                            .clone()
                            .expect("input buffer presence checked by the request shape");
                        let mut bytes = vec![0u8; size.min(input.size) as usize];
                        call.memory.read(input.address, &mut bytes)?;
                        match fs::write_at(&host, offset, &bytes) {
                            Ok(()) => {
                                call.reply(0, None, FSP_INTERFACE_ID, &[])?;
                                Ok(Outcome::success())
                            }
                            Err(code) => fail(call, code),
                        }
                    }
                    2 => {
                        call.shape(0, false, false)?;
                        call.reply(0, None, FSP_INTERFACE_ID, &[])?;
                        Ok(Outcome::success())
                    }
                    3 => {
                        call.shape(8, false, false)?;
                        match fs::set_size(&host, dword(call, 0)) {
                            Ok(()) => {
                                call.reply(0, None, FSP_INTERFACE_ID, &[])?;
                                Ok(Outcome::success())
                            }
                            Err(code) => fail(call, code),
                        }
                    }
                    4 => {
                        call.shape(0, false, false)?;
                        match fs::size(&host) {
                            Ok(size) => {
                                call.reply(0, None, FSP_INTERFACE_ID, &size.to_le_bytes())?;
                                Ok(Outcome::success())
                            }
                            Err(code) => fail(call, code),
                        }
                    }
                    _ => Err(call.unsupported(object.service(), "file command is not implemented")),
                }
            }
            (ServiceObject::Directory { id }, _) => {
                let id = *id;
                match command {
                    // Read(B out of DirectoryEntry) -> u64 count.
                    0 => {
                        call.shape_output(0, false)?;
                        let (address, capacity) = call.output_buffer()?;
                        let (entries, cursor) = &mut self.open_directories[id];
                        let room = capacity as usize / fs::DIRECTORY_ENTRY_SIZE;
                        let batch: Vec<u8> = entries
                            .iter()
                            .skip(*cursor)
                            .take(room)
                            .flat_map(|entry| entry.encode())
                            .collect();
                        let count = batch.len() / fs::DIRECTORY_ENTRY_SIZE;
                        *cursor += count;
                        call.memory.write(address, &batch)?;
                        call.reply(0, None, FSP_INTERFACE_ID, &(count as u64).to_le_bytes())?;
                        Ok(Outcome::success())
                    }
                    1 => {
                        call.shape(0, false, false)?;
                        let count = self.open_directories[id].0.len() as u64;
                        call.reply(0, None, FSP_INTERFACE_ID, &count.to_le_bytes())?;
                        Ok(Outcome::success())
                    }
                    _ => {
                        Err(call
                            .unsupported(object.service(), "directory command is not implemented"))
                    }
                }
            }
            _ => unreachable!("host_file_system dispatch"),
        }
    }

    /// IStorage over the synthesized RomFS image.
    fn data_storage(&mut self, call: &mut Call<'_>) -> Result<Outcome, Trap> {
        const SERVICE: &str = "fsp-srv:data_storage";
        let romfs = self
            .data_romfs
            .as_ref()
            .ok_or_else(|| invalid("data storage has no RomFS"))?;
        match call.request.command_id {
            // Read(u64 offset, u64 size) into one output buffer.
            0 => {
                call.shape_output(16, false)?;
                let offset = crate::bytes::le_u64(&call.request.arguments[..8]);
                let size = crate::bytes::le_u64(&call.request.arguments[8..16]);
                let (address, capacity) = call.output_buffer()?;
                if size > capacity {
                    return Err(call.unsupported(SERVICE, "read larger than its output buffer"));
                }
                if offset
                    .checked_add(size)
                    .is_none_or(|end| end > romfs.size())
                {
                    return Err(call.unsupported(SERVICE, "read past the end of the RomFS image"));
                }
                if size > MAX_STORAGE_READ {
                    return Err(
                        call.unsupported(SERVICE, "single read above the local 64 MiB bound")
                    );
                }
                let bytes = romfs.read(offset, size as usize).map_err(|error| {
                    call.unsupported(SERVICE, format!("RomFS data unavailable: {error:#}"))
                })?;
                call.memory.write(address, &bytes)?;
                call.reply(0, None, FSP_INTERFACE_ID, &[])?;
                self.data_storage_bytes_read = self.data_storage_bytes_read.saturating_add(size);
                self.data_storage_reads += 1;
                self.data_storage_last_offset = offset;
                if size > 0 {
                    self.data_storage_chunks
                        .extend(offset >> 16..=(offset + size - 1) >> 16);
                }
                Ok(Outcome::success())
            }
            // GetSize: u64.
            4 => {
                call.shape(0, false, false)?;
                call.reply(0, None, FSP_INTERFACE_ID, &romfs.size().to_le_bytes())?;
                Ok(Outcome::success())
            }
            _ => Err(call.unsupported(SERVICE, "storage command is not implemented")),
        }
    }

    /// Configure the application's RomFS for OpenDataStorageByCurrentProcess.
    pub fn set_data_romfs(&mut self, romfs: crate::romfs::VirtualRomfs) {
        self.data_romfs = Some(romfs);
    }

    fn nvdrv_sessions(&self) -> usize {
        self.sessions
            .values()
            .filter(|session| {
                matches!(
                    session,
                    Session::Regular {
                        object: ServiceObject::NvidiaDriver { .. }
                    }
                )
            })
            .count()
    }

    /// ISyncpointSupervisor (TIPC). Only commands the application has reached
    /// are modeled; syncpoint counters need the GPU model and stop explicitly.
    fn syncpoint_supervisor(&mut self, call: &mut Call<'_>) -> Result<Outcome, Trap> {
        const SERVICE: &str = "appletOE:syncpoint_supervisor";
        if call.request.protocol != Protocol::Tipc {
            return Err(call.unsupported(SERVICE, "syncpoint supervisor commands are TIPC"));
        }
        match call.request.command_id {
            // Command 1 returns the largest event index the session accepts.
            // Undocumented by name; 63 matches the documented nvhost-ctrl
            // event-slot range 0x00-0x3F (Switchbrew "NV services") and the
            // value Eden 1a6d7d4 returns. Reported as provisional.
            1 => {
                call.shape(0, false, false)?;
                call.reply(0, None, 0, &SYNCPOINT_MAX_EVENT_INDEX.to_le_bytes())?;
                self.provisional_commands
                    .insert("appletOE:syncpoint_supervisor:1");
                self.provisional_inputs
                    .insert("appletOE:syncpoint_supervisor:1", Vec::new());
                Ok(Outcome::success())
            }
            // Command 21 sets how many events the client will use (u32 at
            // most the value from command 1).
            21 => {
                call.shape(4, false, false)?;
                let count = crate::bytes::le_u32(&call.request.arguments[..4]);
                if count > SYNCPOINT_MAX_EVENT_INDEX {
                    return Err(call.unsupported(SERVICE, "event count above the maximum"));
                }
                call.reply(0, None, 0, &[])?;
                self.syncpoint_event_count = Some(count);
                Ok(Outcome::success())
            }
            // Command 103 (the SDK's TryWaitSyncPoint, client sdk+0x5d3e0c):
            // u32 inverted syncpoint handle -> u64 current value and u8
            // "64-bit" flag; the SDK counts its threshold as reached unless
            // the value is below it, or for some callers unless the 32-bit
            // difference value - threshold is negative (sdk+0x5d3e9c,
            // sdk+0x5d3ea8). Syncpoint values are not tracked and host GPU
            // work completes at submission, so every threshold counts as
            // reached: 0x7fffffff_7fffffff passes both comparisons for any
            // threshold up to 0x7fffffff. Provisional.
            103 => {
                call.shape(4, false, false)?;
                let token = crate::bytes::le_u32(&call.request.arguments[..4]);
                let value = self
                    .decoder_fence_value(token)
                    .unwrap_or(0x7fff_ffff_7fff_ffff);
                let mut reply = value.to_le_bytes().to_vec();
                reply.extend_from_slice(&1u32.to_le_bytes());
                call.reply(0, None, 0, &reply)?;
                self.provisional_commands
                    .insert("appletOE:syncpoint_supervisor:103");
                Ok(Outcome::success())
            }
            // Command 201 (the SDK's RequestSyncPointEvent, client
            // sdk+0x5d4eb0): u32 event index (from 22), u32 syncpoint, u64
            // value, u8 flag; no reply data. It arms the index's event for
            // the syncpoint reaching the value. Host GPU work completes at
            // submission, so the value counts as reached: the event is
            // signaled at once. Provisional.
            201 => {
                call.shape(20, false, false)?;
                let index = crate::bytes::le_u32(&call.request.arguments[..4]);
                let Some(&event) = self.syncpoint_events.get(&index) else {
                    return Err(call.unsupported(
                        SERVICE,
                        format!("syncpoint event index {index} was not acquired"),
                    ));
                };
                call.reply(0, None, 0, &[])?;
                // A decoder fence short of its value arms the event for the
                // frame that reaches it (see decode_video_frames).
                let token = crate::bytes::le_u32(&call.request.arguments[4..8]);
                let value = crate::bytes::le_u64(&call.request.arguments[8..16]);
                self.note_video(format!(
                    "syncpoint 201 index {index} token {token:#x} value {value}"
                ));
                if !self.decoder_fence_reached(token, value) {
                    // An index armed before for a reached target would read
                    // as signaled (see readable_event_signaled).
                    self.armed_syncpoint_events.remove(&index);
                    self.decoder_fence_waits.insert(index, (token, value));
                    if let Some(event) = self.readable_events.get_mut(&event) {
                        event.signaled = false;
                    }
                    self.provisional_commands
                        .insert("appletOE:syncpoint_supervisor:201");
                    return Ok(Outcome::success());
                }
                if let Some(event) = self.readable_events.get_mut(&event) {
                    event.signaled = true;
                }
                self.armed_syncpoint_events.insert(index);
                self.provisional_commands
                    .insert("appletOE:syncpoint_supervisor:201");
                Ok(Outcome::success())
            }
            // Command 202: u32 event index; no reply data. The SDK's
            // syncpoint-event wait sends it when the event is not yet
            // signaled and then waits on the event (sdk+0x5d4b48); it also
            // follows successful waits (sdk+0x5d5138). It asks for the event
            // once the armed target is reached, which on the host it always
            // is: the event is signaled. Provisional.
            202 => {
                call.shape(4, false, false)?;
                let index = crate::bytes::le_u32(&call.request.arguments[..4]);
                let Some(&event) = self.syncpoint_events.get(&index) else {
                    return Err(call.unsupported(
                        SERVICE,
                        format!("syncpoint event index {index} was not acquired"),
                    ));
                };
                call.reply(0, None, 0, &[])?;
                let pending = self
                    .decoder_fence_waits
                    .get(&index)
                    .is_some_and(|&(token, value)| !self.decoder_fence_reached(token, value));
                if !pending {
                    self.decoder_fence_waits.remove(&index);
                    if let Some(event) = self.readable_events.get_mut(&event) {
                        event.signaled = true;
                    }
                }
                self.provisional_commands
                    .insert("appletOE:syncpoint_supervisor:202");
                Ok(Outcome::success())
            }
            // Command 22: u32 event index -> a moved event handle. The SDK's
            // syncpt InitializeEx tries indices in turn and wraps each
            // returned handle as a system event (sdk+0x5d41f4 reads the
            // moved handle), skipping indices that fail. One event per
            // index, below 1024 (the SDK's slot limit); a taken index fails.
            // When the events should be signaled is not modeled yet.
            22 => {
                call.shape(4, false, false)?;
                let index = crate::bytes::le_u32(&call.request.arguments[..4]);
                if index >= 1024 || self.syncpoint_events.contains_key(&index) {
                    return Err(call.unsupported(SERVICE, format!("syncpoint event index {index}")));
                }
                let application = self
                    .main_application
                    .clone()
                    .ok_or_else(|| invalid("syncpoint events require the main application"))?;
                if self.readable_events.len() >= MAX_READABLE_EVENTS {
                    return Err(invalid("readable event backing capacity exhausted"));
                }
                let event = ReadableEventId(
                    self.readable_events
                        .keys()
                        .map(|id| id.0)
                        .max()
                        .unwrap_or(0)
                        + 1,
                );
                let handle = self.next_available_handle()?;
                call.reply(0, Some(handle), 0, &[])?;
                self.readable_events.insert(
                    event,
                    ReadableEvent {
                        owner: application,
                        kind: ReadableEventKind::SyncpointEvent { index },
                        signaled: false,
                    },
                );
                self.syncpoint_events.insert(index, event);
                self.readable_event_handles.insert(handle, event);
                self.next_handle = handle + 1;
                self.provisional_commands
                    .insert("appletOE:syncpoint_supervisor:22");
                Ok(Outcome::success())
            }
            _ => Err(call.unsupported(
                SERVICE,
                format!(
                    "syncpoint supervisor command is not implemented ({})",
                    call.request.describe()
                ),
            )),
        }
    }

    /// KUserWorkerSession (Switchbrew "Switch 2: GSV services",
    /// nn::gpuk::impl::kuser::KUserWorkerSession), TIPC.
    fn gpu_worker(&mut self, call: &mut Call<'_>) -> Result<Outcome, Trap> {
        const SERVICE: &str = "appletOE:gpu_worker";
        match call.request.command_id {
            // CreateGpuMemory: u64 MemoryAddressStart, u64 MemoryAddressEnd
            // (its last byte: the SDK sends start + size - 1) and two u8s;
            // returns a u32 GpuMemoryHandle. The range must
            // carry GpuSharable (SetGpuMemoryAttribute 0x20). Handles count
            // from 1. Provisional beyond the documented layout.
            101 => {
                let arguments = &call.request.arguments;
                if !(18..=24).contains(&arguments.len()) {
                    return Err(call.unsupported(
                        SERVICE,
                        format!("CreateGpuMemory shape ({})", call.request.describe()),
                    ));
                }
                let start = crate::bytes::le_u64(&arguments[..8]);
                let last = crate::bytes::le_u64(&arguments[8..16]);
                let end = last.wrapping_add(1);
                let flags = [arguments[16], arguments[17]];
                if last < start
                    || end == 0
                    || start % 0x1000 != 0
                    || end % 0x1000 != 0
                    || !self.is_gpu_sharable(start, end - start)
                {
                    return Err(call.unsupported(
                        SERVICE,
                        format!("CreateGpuMemory range {start:#x}..{end:#x} is not GpuSharable"),
                    ));
                }
                if self.gpu_memory.len() >= 4096 {
                    return Err(call.unsupported(SERVICE, "too many GPU memory handles"));
                }
                let handle = self
                    .gpu_memory
                    .keys()
                    .next_back()
                    .map_or(1, |last| last + 1);
                call.reply(0, None, 0, &handle.to_le_bytes())?;
                self.gpu_memory.insert(handle, (start, end, flags));
                self.provisional_commands.insert("appletOE:gpu_worker:101");
                Ok(Outcome::success())
            }
            // DestroyGpuMemory: u32 GpuMemoryHandle and u32 EventId, no
            // output; the event is signaled when the memory is released
            // (Switchbrew). Memory still mapped is refused.
            102 => {
                call.shape(8, false, false)?;
                let memory = crate::bytes::le_u32(&call.request.arguments[..4]);
                let event_index = crate::bytes::le_u32(&call.request.arguments[4..8]);
                if !self.gpu_memory.contains_key(&memory) {
                    return Err(call
                        .unsupported(SERVICE, format!("destroy of unknown GPU memory {memory}")));
                }
                if self
                    .gpu_mappings
                    .values()
                    .any(|&(_, _, mapped, _)| mapped == memory)
                {
                    return Err(
                        call.unsupported(SERVICE, "destroy of GPU memory that is still mapped")
                    );
                }
                let Some(&event) = self.gpu_events.get(&event_index) else {
                    return Err(call.unsupported(SERVICE, "release event was not created"));
                };
                call.reply(0, None, 0, &[])?;
                self.gpu_memory.remove(&memory);
                if let Some(event) = self.readable_events.get_mut(&event) {
                    event.signaled = true;
                }
                Ok(Outcome::success())
            }
            // NormalMapGpuMemory4K ([20.0.0+] 211). Layout from the SDK
            // (sdk+0x4691c0): u32 address space, u64 GPU address, u64 last
            // GPU address, u32 GpuMemoryHandle, u64 memory offset, seven u8
            // settings each in its own word, and the u32 ID of a GpuSession
            // event the SDK then waits on. Returns a u8 the SDK overwrites.
            // The map completes at once and signals that event. The host
            // renders through NVN, so nothing is mapped. Provisional.
            211 => {
                let arguments = &call.request.arguments;
                if arguments.len() != 0x40 {
                    return Err(call.unsupported(
                        SERVICE,
                        format!("NormalMapGpuMemory4K shape ({})", call.request.describe()),
                    ));
                }
                let word = |at: usize| crate::bytes::le_u32(&arguments[at..at + 4]);
                let dword = |at: usize| crate::bytes::le_u64(&arguments[at..at + 8]);
                let (space, address, last) = (word(0), dword(4), dword(0xc));
                let (memory, offset, event_index) = (word(0x14), dword(0x18), word(0x3c));
                let Some(&(start, end, _)) = self.gpu_memory.get(&memory) else {
                    return Err(
                        call.unsupported(SERVICE, format!("map of unknown GPU memory {memory}"))
                    );
                };
                if !self.gpu_address_spaces.contains_key(&space)
                    || last < address
                    || address % 0x1000 != 0
                    || (last + 1) % 0x1000 != 0
                    || offset
                        .checked_add(last - address)
                        .is_none_or(|top| top >= end - start)
                {
                    return Err(call.unsupported(
                        SERVICE,
                        format!(
                            "map {address:#x}..={last:#x} of memory {memory} offset {offset:#x} in space {space} is not modeled"
                        ),
                    ));
                }
                if self.gpu_mappings.range(..=last).next_back().is_some_and(
                    |(_, &(other, other_last, ..))| other == space && other_last >= address,
                ) {
                    return Err(call.unsupported(SERVICE, "overlapping GPU mapping"));
                }
                let Some(&event) = self.gpu_events.get(&event_index) else {
                    return Err(call.unsupported(SERVICE, "map completion event was not created"));
                };
                call.reply(0, None, 0, &0u32.to_le_bytes())?;
                self.gpu_mappings
                    .insert(address, (space, last, memory, offset));
                if let Some(event) = self.readable_events.get_mut(&event) {
                    event.signaled = true;
                }
                self.provisional_commands.insert("appletOE:gpu_worker:211");
                Ok(Outcome::success())
            }
            // NormalUnmapGpuMemory4K ([20.0.0+] 213). Layout from the SDK
            // (sdk+0x469558): u32 address space, u64 GPU address, u64 last
            // GPU address, u8 sync flag, u32 event ID the SDK then waits on.
            // Returns the u64 byte count unmapped (sdk+0x4696d8 subtracts it
            // from the space's mapped total). Removes the mappings inside the
            // range and signals the event. Provisional.
            213 => {
                let arguments = &call.request.arguments;
                if arguments.len() != 0x1c {
                    return Err(call.unsupported(
                        SERVICE,
                        format!("NormalUnmapGpuMemory4K shape ({})", call.request.describe()),
                    ));
                }
                let word = |at: usize| crate::bytes::le_u32(&arguments[at..at + 4]);
                let dword = |at: usize| crate::bytes::le_u64(&arguments[at..at + 8]);
                let (space, address, last, event_index) =
                    (word(0), dword(4), dword(0xc), word(0x18));
                if !self.gpu_address_spaces.contains_key(&space) || last < address {
                    return Err(call.unsupported(
                        SERVICE,
                        format!("unmap {address:#x}..={last:#x} in space {space} is not modeled"),
                    ));
                }
                let inside: Vec<(u64, u64)> = self
                    .gpu_mappings
                    .range(address..=last)
                    .filter(|&(_, &(other, other_last, ..))| other == space && other_last <= last)
                    .map(|(&start, &(_, other_last, ..))| (start, other_last))
                    .collect();
                // The reply is the byte count unmapped; the SDK subtracts it
                // from the address space's mapped total, which must reach
                // zero before DestroyGpuAddressSpace.
                let unmapped: u64 = inside.iter().map(|&(start, end)| end - start + 1).sum();
                let Some(&event) = self.gpu_events.get(&event_index) else {
                    return Err(call.unsupported(SERVICE, "unmap completion event was not created"));
                };
                call.reply(0, None, 0, &unmapped.to_le_bytes())?;
                for (start, _) in inside {
                    self.gpu_mappings.remove(&start);
                }
                if let Some(event) = self.readable_events.get_mut(&event) {
                    event.signaled = true;
                }
                self.provisional_commands.insert("appletOE:gpu_worker:213");
                Ok(Outcome::success())
            }
            _ => Err(call.unsupported(
                SERVICE,
                format!(
                    "GPU worker command is not implemented ({})",
                    call.request.describe()
                ),
            )),
        }
    }

    /// Whether every page of `[address, address + size)` is GpuSharable.
    pub(crate) fn is_gpu_sharable(&self, address: u64, size: u64) -> bool {
        let end = address.saturating_add(size);
        let mut cursor = address;
        while cursor < end {
            match self.gpu_sharable.range(..=cursor).next_back() {
                Some((_, &run_end)) if run_end > cursor => cursor = run_end,
                _ => return false,
            }
        }
        true
    }

    /// The process-bound NVIDIA driver session. Commands are added as the
    /// application reaches them; everything else stops explicitly.
    fn nvidia_driver(
        &mut self,
        application: &MainApplicationIdentity,
        call: &mut Call<'_>,
    ) -> Result<Outcome, Trap> {
        const SERVICE: &str = "nvdrv:gsv";
        if self.main_application.as_ref() != Some(application) {
            return Err(invalid(
                "NVIDIA driver owner must match the trusted bootstrap main application",
            ));
        }
        match call.request.command_id {
            // INvDrv2User Open (0): a type-0x5 input buffer Path; returns u32
            // FdOut then u32 Err (Switchbrew "NV services" classic Open; the
            // S2 table names the command without details). Only device paths
            // the page documents open; descriptors count up from 1.
            0 => {
                call.shape(0, false, true)?;
                let buffer = call
                    .request
                    .input_buffer
                    .clone()
                    .ok_or_else(|| invalid("Open shape check guarantees a path buffer"))?;
                if buffer.size == 0 || buffer.size > 0x100 {
                    return Err(call.unsupported(SERVICE, "Open path size is not 1..=0x100"));
                }
                let mut bytes = vec![0u8; buffer.size as usize];
                call.memory.read(buffer.address, &mut bytes)?;
                let end = bytes
                    .iter()
                    .position(|byte| *byte == 0)
                    .unwrap_or(bytes.len());
                let path = String::from_utf8_lossy(&bytes[..end]).into_owned();
                if !NVDRV_DEVICES.contains(&path.as_str()) {
                    return Err(
                        call.unsupported(SERVICE, format!("Open of undocumented device {path:?}"))
                    );
                }
                // No GPU debugger or profiler exists (the characteristics
                // report the debugger unavailable). As Eden 1a6d7d4: the
                // profiler answers NotSupported and an absent device
                // FileOperationFailed, both with fd 0; the driver then takes
                // its no-debugger path.
                let unavailable = match path.as_str() {
                    "/dev/nvhost-prof-gpu" => Some(NV_NOT_SUPPORTED),
                    "/dev/nvhost-dbg-gpu" | "/dev/nvhost-prof-dev-gpu" => {
                        Some(NV_FILE_OPERATION_FAILED)
                    }
                    _ => None,
                };
                if let Some(error) = unavailable {
                    let mut reply = 0u32.to_le_bytes().to_vec();
                    reply.extend_from_slice(&error.to_le_bytes());
                    call.reply(0, None, NVDRV_INTERFACE_ID, &reply)?;
                    return Ok(Outcome::success());
                }
                if self.nvdrv_fds.len() >= NVDRV_MAX_FDS {
                    return Err(call.unsupported(SERVICE, "too many open NVIDIA driver devices"));
                }
                let fd = self.nvdrv_fds.keys().next_back().map_or(1, |last| last + 1);
                let mut reply = fd.to_le_bytes().to_vec();
                reply.extend_from_slice(&0u32.to_le_bytes());
                call.reply(0, None, NVDRV_INTERFACE_ID, &reply)?;
                self.nvdrv_fds.insert(fd, path);
                Ok(Outcome::success())
            }
            // INvDrv2User Ioctl (1) and Ioctl3 (12): u32 Fd, u32 Iocode, an
            // input buffer and one output buffer (Ioctl3: main and inline);
            // returns u32 Err (classic layout, Eden 1a6d7d4). Device ioctls
            // live in nvdrv.rs.
            command @ (1 | 12) => {
                let outputs = if command == 1 { 1 } else { 2 };
                let request = &call.request;
                if request.arguments.len() < 8
                    || request.pid.is_some()
                    || !request.copy_handles.is_empty()
                    || request.input_buffer.is_none()
                    || request.receive_buffers.len() != outputs
                {
                    return Err(call.unsupported(
                        SERVICE,
                        format!(
                            "ioctl shape is not the documented one ({})",
                            request.describe()
                        ),
                    ));
                }
                let word = |at: usize| crate::bytes::le_u32(&request.arguments[at..at + 4]);
                let (fd, code) = (word(0), word(4));
                let Some(device) = self.nvdrv_fds.get(&fd).cloned() else {
                    return Err(call.unsupported(SERVICE, format!("ioctl on unopened fd {fd}")));
                };
                let input = request
                    .input_buffer
                    .clone()
                    .ok_or_else(|| invalid("checked above"))?;
                let mut input_bytes = vec![
                    0u8;
                    usize::try_from(input.size)
                        .unwrap_or(usize::MAX)
                        .min(0x10000)
                ];
                call.memory.read(input.address, &mut input_bytes)?;
                let main = call.output_buffer_at(0)?;
                let inline = if outputs == 2 {
                    call.output_buffer_at(1)?
                } else {
                    (0, 0)
                };
                let result = crate::nvdrv::ioctl(
                    &mut self.nvdrv_devices,
                    fd,
                    &device,
                    code,
                    &input_bytes,
                    main.1 as usize,
                    inline.1 as usize,
                )
                .map_err(|message| call.unsupported(SERVICE, message))?;
                if result.output.len() as u64 > main.1
                    || result.inline_output.len() as u64 > inline.1
                {
                    return Err(invalid("ioctl output exceeds its buffer"));
                }
                call.memory.write(main.0, &result.output)?;
                call.memory.write(inline.0, &result.inline_output)?;
                call.reply(0, None, NVDRV_INTERFACE_ID, &result.result.to_le_bytes())?;
                self.provisional_commands.insert("nvdrv:gsv:ioctl");
                Ok(Outcome::success())
            }
            // INvDrv2User Close (2): u32 Fd; returns u32 Err (classic layout).
            2 => {
                call.shape(4, false, false)?;
                let fd = crate::bytes::le_u32(&call.request.arguments[..4]);
                if !self.nvdrv_fds.contains_key(&fd) {
                    return Err(call.unsupported(SERVICE, format!("Close of unopened fd {fd}")));
                }
                call.reply(0, None, NVDRV_INTERFACE_ID, &0u32.to_le_bytes())?;
                self.nvdrv_fds.remove(&fd);
                self.nvdrv_devices.close(fd);
                Ok(Outcome::success())
            }
            // INvDrv2User QueryEvent (4): u32 Fd, u32 EvtId; copies the
            // readable event and returns u32 Err (classic layout). Only the
            // nvhost-ctrl-gpu events Switchbrew documents (1 ErrorEvent, 2
            // SemaphoreEvent) exist; repeated queries alias one backing.
            4 => {
                call.shape(8, false, false)?;
                let word = |at: usize| crate::bytes::le_u32(&call.request.arguments[at..at + 4]);
                let (fd, event) = (word(0), word(4));
                let Some(device) = self.nvdrv_fds.get(&fd).cloned() else {
                    return Err(
                        call.unsupported(SERVICE, format!("QueryEvent on unopened fd {fd}"))
                    );
                };
                let device: &'static str = match (device.as_str(), event) {
                    ("/dev/nvhost-ctrl-gpu", 1 | 2) => "/dev/nvhost-ctrl-gpu",
                    _ => {
                        return Err(call.unsupported(
                            SERVICE,
                            format!("QueryEvent {event} on {device} is not modeled"),
                        ));
                    }
                };
                let key = (device.to_string(), event);
                let id = match self.nvdrv_events.get(&key) {
                    Some(id) => *id,
                    None => {
                        if self.readable_events.len() >= MAX_READABLE_EVENTS {
                            return Err(invalid("readable event backing capacity exhausted"));
                        }
                        ReadableEventId(
                            self.readable_events
                                .keys()
                                .map(|id| id.0)
                                .max()
                                .unwrap_or(0)
                                + 1,
                        )
                    }
                };
                let handle = self.next_available_handle()?;
                let reply = call.prepare_copy_handle_reply_with_data(
                    handle,
                    NVDRV_INTERFACE_ID,
                    &0u32.to_le_bytes(),
                )?;
                call.write_reply(&reply)?;
                self.readable_events.entry(id).or_insert(ReadableEvent {
                    owner: application.clone(),
                    kind: ReadableEventKind::NvdrvEvent { device, id: event },
                    signaled: false,
                });
                self.nvdrv_events.insert(key, id);
                self.readable_event_handles.insert(handle, id);
                self.next_handle = handle + 1;
                Ok(Outcome::success())
            }
            // INvDrv2User SetConfiguration (13): a u64 whose bit 0 is the old
            // GraphicsFirmwareMemoryMarginEnabled option (Switchbrew "NV
            // services", [S2]). Other bits are undocumented and stop. There is
            // no firmware memory pool here, so the value is stored and reported.
            13 => {
                call.shape(16, false, false)
                    .or_else(|_| call.shape(8, false, false))?;
                let arguments = &call.request.arguments;
                let enabled = arguments[0];
                if enabled > 1 || arguments[1..8].iter().any(|byte| *byte != 0) {
                    return Err(call.unsupported(
                        SERVICE,
                        "margin input is not a boolean followed by zero padding",
                    ));
                }
                if arguments.len() == 16 {
                    let aruid = crate::bytes::le_u64(&arguments[8..16]);
                    if aruid != application.process_id {
                        return Err(
                            call.unsupported(SERVICE, "margin aruid is not the application's")
                        );
                    }
                }
                call.reply(0, None, NVDRV_INTERFACE_ID, &[])?;
                self.nvdrv_firmware_memory_margin = Some(enabled == 1);
                Ok(Outcome::success())
            }
            _ => Err(call.unsupported(
                SERVICE,
                format!(
                    "NVIDIA driver command is not implemented ({})",
                    call.request.describe()
                ),
            )),
        }
    }

    /// ReceiveMessage (1) and GetCurrentFocusState (9) on the lifecycle record.
    /// Eden 1a6d7d4 PopMessage pops by priority and then re-evaluates the
    /// system event (clearing it when nothing remains); an empty queue is AM
    /// ResultNoMessages. GetAndClearFocusState acknowledges the requested
    /// focus and returns it as a u8. State and the event bit are published
    /// only after the complete reply is written.
    fn common_state_lifecycle(
        &mut self,
        application: &MainApplicationIdentity,
        call: &mut Call<'_>,
    ) -> Result<Outcome, Trap> {
        call.shape(0, false, false)?;
        if self.main_application.as_ref() != Some(application) {
            return Err(invalid(
                "lifecycle endpoint owner must match the trusted bootstrap main application",
            ));
        }
        let state = self
            .main_application_state
            .as_ref()
            .ok_or_else(|| invalid("lifecycle requires initialized application state"))?;
        let mut lifecycle = state.lifecycle.clone();
        let event_id = state.lifecycle_event;
        if !self.readable_events.contains_key(&event_id) {
            return Err(invalid("lifecycle event backing is missing"));
        }
        let (result, data) = if call.request.command_id == 1 {
            match lifecycle.pop_message() {
                Some(message) => (0, message.to_le_bytes().to_vec()),
                None => (AM_RESULT_NO_MESSAGES, Vec::new()),
            }
        } else {
            lifecycle.acknowledged_focus_state = lifecycle.requested_focus_state;
            (0, vec![lifecycle.acknowledged_focus_state])
        };
        call.reply(result, None, COMMON_STATE_GETTER_INTERFACE_ID, &data)?;
        let signaled = lifecycle.refresh_availability();
        if let Some(state) = self.main_application_state.as_mut() {
            state.lifecycle = lifecycle;
        }
        if let Some(event) = self.readable_events.get_mut(&event_id) {
            event.signaled = signaled;
        }
        Ok(Outcome::success())
    }

    fn copy_application_event(
        &mut self,
        application: &MainApplicationIdentity,
        call: &mut Call<'_>,
        kind: ReadableEventKind,
        interface_id: u32,
    ) -> Result<Outcome, Trap> {
        call.shape(0, false, false)?;
        if self.main_application.as_ref() != Some(application) {
            return Err(invalid(
                "event endpoint owner must match the trusted bootstrap main application",
            ));
        }
        let state = self
            .main_application_state
            .as_ref()
            .ok_or_else(|| invalid("owned event requires initialized application state"))?;
        let event_id = match kind {
            ReadableEventKind::AccumulatedSuspendedTickChanged => {
                state.accumulated_suspended_tick_changed_event
            }
            ReadableEventKind::GpuErrorDetected => state.gpu_error_detected_event,
            ReadableEventKind::LifecycleSystem => state.lifecycle_event,
            ReadableEventKind::AddOnContentListChanged => state.add_on_content_list_changed_event,
            ReadableEventKind::ApplicationUnknown210 => state.application_unknown_210_event,
            ReadableEventKind::FriendInvitationStorageChannel => {
                state.friend_invitation_storage_channel_event
            }
            ReadableEventKind::SyncpointCacheRequest
            | ReadableEventKind::SyncpointCacheCompleted
            | ReadableEventKind::AsyncContextDone
            | ReadableEventKind::GpuNotification { .. }
            | ReadableEventKind::AudioDeviceOutput
            | ReadableEventKind::AudioOutBuffer
            | ReadableEventKind::AudioRendererFrame
            | ReadableEventKind::SocketPoll
            | ReadableEventKind::NfpAvailability
            | ReadableEventKind::SyncpointEvent { .. }
            | ReadableEventKind::VideoDecoderInput
            | ReadableEventKind::LibraryApplet
            | ReadableEventKind::NetworkRequest
            | ReadableEventKind::BcatProgress
            | ReadableEventKind::NpadStyleSetUpdate
            | ReadableEventKind::DisplayNotification
            | ReadableEventKind::DisplayEvent31
            | ReadableEventKind::NvdrvEvent { .. } => {
                return Err(invalid(
                    "syncpoint-cache events are not application lifecycle events",
                ));
            }
        };
        let event = self
            .readable_events
            .get(&event_id)
            .ok_or_else(|| invalid("application-owned event backing is missing"))?;
        if event.owner != *application || event.kind != kind {
            return Err(invalid(
                "application-owned event backing owner/type mismatch",
            ));
        }
        let handle = self.next_available_handle()?;
        let mut aliases = self.readable_event_handles.clone();
        aliases.insert(handle, event_id);
        // A fresh alias references the existing event without signaling,
        // clearing, acknowledging/consuming notifications or changing
        // suspended_ticks. Libnx autoclear is a client wait/reset policy,
        // not a property of this getter/event.
        // Stage all allocations and construct the complete reply before write;
        // faults leave the alias table, allocator and application state intact.
        let reply = call.prepare_copy_handle_reply(handle, interface_id)?;
        call.write_reply(&reply)?;
        self.readable_event_handles = aliases;
        self.next_handle = handle + 1;
        Ok(Outcome {
            output_copy_handle_count: 1,
            ..Outcome::success()
        })
    }

    fn open_self_controller(
        &mut self,
        application: &MainApplicationIdentity,
        call: &mut Call<'_>,
    ) -> Result<Outcome, Trap> {
        call.shape(0, false, false)?;
        if self.main_application.as_ref() != Some(application) {
            return Err(invalid(
                "SelfController owner must match the trusted bootstrap main application",
            ));
        }
        let mut updated = self.main_application_state.clone().ok_or_else(|| {
            invalid("SelfController requires the bootstrap application launch policy")
        })?;
        let policy = updated.launch_policy;
        let manager =
            updated
                .display_layer_manager
                .get_or_insert_with(|| LogicalDisplayLayerManager {
                    owner: application.clone(),
                    applet_id: policy.applet_id,
                    library_applet_mode: policy.library_applet_mode,
                    shared_buffer_id: None,
                    shared_layer_id: None,
                    buffer_sharing_enabled: false,
                    blending_enabled: false,
                    managed_display_layers: BTreeSet::new(),
                    managed_recording_layers: BTreeSet::new(),
                });
        // Each SelfController constructor initializes the same Applet manager.
        // Reference Initialize resets these scalar fields but does not clear
        // managed sets or change visibility. Our supported child methods never
        // allocate layers; the sets therefore stay empty during guest execution.
        manager.owner = application.clone();
        manager.applet_id = policy.applet_id;
        manager.library_applet_mode = policy.library_applet_mode;
        manager.shared_buffer_id = None;
        manager.shared_layer_id = None;
        manager.buffer_sharing_enabled = false;
        manager.blending_enabled = matches!(policy.library_applet_mode, 1 | 3);
        let child = ServiceObject::SelfController {
            application: application.clone(),
        };
        // Prepare all application state before the fallible reply/object path.
        // A shape, capacity or reply-write fault publishes neither change.
        let outcome = self.reply_new_object(call, child, APPLICATION_PROXY_INTERFACE_ID)?;
        self.main_application_state = Some(updated);
        // No Drop hook belongs on the cloned ServiceObject enum. Reference
        // Finalize returns immediately without a VI manager, and retains these
        // logical parameters even when a bound backend is finalized. Closing a
        // supported endpoint therefore leaves this application-owned record;
        // this is not an implementation of VI/layer/supervisor teardown.
        Ok(outcome)
    }

    fn open_application_proxy(&mut self, call: &mut Call<'_>) -> Result<Outcome, Trap> {
        call.shape_with_copies(16, true, false, 1)?;
        let flags = crate::bytes::le_u32(&call.request.arguments[..4]);
        if flags != 0 {
            return Err(call.unsupported(
                "appletOE",
                "command 10 supports only the supplied patch's flags-zero compatibility policy",
            ));
        }
        let application = self.main_application.as_ref().ok_or_else(|| {
            call.unsupported(
                "appletOE",
                "main application identity was not initialized by bootstrap",
            )
        })?;
        if application.process_id != call.process_id {
            return Err(call.unsupported(
                "appletOE",
                "trusted sender does not match the bootstrap main application process",
            ));
        }
        if call.request.copy_handles[0] != 0xffff8001 {
            return Err(call.unsupported(
                "appletOE",
                "application proxy requires a typed borrow of the current modeled process; other copied objects are not implemented",
            ));
        }
        // SF ClientProcessId marshalling overwrites the raw u64 placeholder at
        // argument+8 from the trusted sender; argument+4 is alignment padding.
        // Neither those bytes nor the raw PID slot establish process identity.
        // HIPC permits other copied object kinds/null slots; this handler's
        // narrow typed borrow is not a general destination handle-table transfer
        // and never consumes the caller's process pseudo-handle.
        self.reply_new_object(
            call,
            ServiceObject::ApplicationProxy {
                application: application.clone(),
            },
            APPLICATION_PROXY_SERVICE_INTERFACE_ID,
        )
    }

    fn close_domain_object(
        &mut self,
        session: &Session,
        call: &mut Call<'_>,
    ) -> Result<Outcome, Trap> {
        call.shape(0, false, false)?;
        let id = call
            .request
            .domain
            .as_ref()
            .ok_or_else(|| invalid("domain close has no domain metadata"))?
            .object_id;
        let mut updated = session.clone();
        let Session::Domain { objects, .. } = &mut updated else {
            return Err(invalid("domain close requires a converted session"));
        };
        // Primary SF UnregisterObject ignores missing IDs and treats root and
        // child entries alike. The wrapper/session and sibling objects survive.
        objects.remove(&id);
        let reply = ipc::build_domain_close_reply(call.request, call.original_bytes)
            .map_err(|error| invalid(format!("build domain close reply: {error:#}")))?;
        call.write_reply(&reply)?;
        self.sessions.insert(call.handle, updated);
        Ok(Outcome {
            result: None,
            log_disposition: None,
            created_domain_objects: Vec::new(),
            output_copy_handle_count: 0,
        })
    }

    fn service_manager(
        &mut self,
        registered_process_id: Option<u64>,
        call: &mut Call<'_>,
    ) -> Result<Outcome, Trap> {
        match call.request.command_id {
            0 => {
                let expected = match call.request.protocol {
                    Protocol::Tipc => 0,
                    Protocol::Cmif => 8,
                };
                call.shape(expected, true, false)?;
                let mut updated = self.sessions[&call.handle].clone();
                let object_id = call.request.domain.as_ref().map(|domain| domain.object_id);
                let Some(ServiceObject::ServiceManager {
                    registered_process_id,
                }) = updated.object_mut(object_id)
                else {
                    return Err(invalid(
                        "registration target is not a service-manager object",
                    ));
                };
                *registered_process_id = Some(call.process_id);
                call.reply(0, None, ipc::SM_CMIF_INTERFACE_ID, &[])?;
                // The descriptor and raw payload PID bytes are not authenticated.
                self.sessions.insert(call.handle, updated);
                Ok(Outcome::success())
            }
            1 => {
                call.shape(8, false, false)?;
                if registered_process_id != Some(call.process_id) {
                    // TIPC declares a move-handle slot even on service errors.
                    // CMIF follows this prototype's explicit generic SF policy.
                    let empty_handle = (call.request.protocol == Protocol::Tipc).then_some(0);
                    call.reply(0x415, empty_handle, ipc::SM_CMIF_INTERFACE_ID, &[])?;
                    return Ok(Outcome::result(0x415));
                }
                let name = service_name(&call.request.arguments)?;
                if !self.service_allowed(&name) {
                    // sm ResultNotAllowed (module 21, description 8): the
                    // NPDM's service access control does not list this name.
                    let empty_handle = (call.request.protocol == Protocol::Tipc).then_some(0);
                    call.reply(0x1015, empty_handle, ipc::SM_CMIF_INTERFACE_ID, &[])?;
                    self.denied_services.insert(name);
                    return Ok(Outcome::result(0x1015));
                }
                let object = match name.as_str() {
                    "lm" => ServiceObject::LogManager,
                    "appletOE" => ServiceObject::ApplicationProxyService,
                    "aoc:u" => ServiceObject::AddOnContentManager,
                    "ectx:aw" => ServiceObject::ErrorContextWriter,
                    "pctl" => ServiceObject::ParentalControlFactory,
                    "time:u" => ServiceObject::TimeStatic,
                    "fsp-srv" => ServiceObject::FileSystemProxy,
                    "acc:u0" => ServiceObject::AccountService,
                    "aud:u" => ServiceObject::AudioManager,
                    "hwopus" => ServiceObject::OpusDecoderManager,
                    "auddmg:u" => ServiceObject::AudioDeviceSessionManager,
                    "bsd:u" => ServiceObject::SocketClient,
                    "ssl" => ServiceObject::SslService,
                    "nifm:u" => ServiceObject::NifmStatic,
                    "nsd:u" => ServiceObject::NsdUser,
                    "nfp:user" => ServiceObject::NfpUserManager,
                    "bcat:u" => ServiceObject::BcatCreator,
                    "stpl:u" => ServiceObject::StreamplayUser,
                    "prepo:u" => ServiceObject::PlayReport,
                    "hid" => ServiceObject::HidServer,
                    _ => {
                        return Err(call.unsupported(
                            "sm:",
                            format!("requested service {name:?} has no implemented backend"),
                        ));
                    }
                };
                self.reply_new_object(call, object, ipc::SM_CMIF_INTERFACE_ID)
            }
            _ => Err(call.unsupported("sm:", "service-manager command is not implemented")),
        }
    }

    fn logger(&mut self, process_id: u64, call: &mut Call<'_>) -> Result<Outcome, Trap> {
        match call.request.command_id {
            0 => {
                call.shape(0, false, true)?;
                let buffer = call
                    .request
                    .input_buffer
                    .as_ref()
                    .expect("input buffer presence checked by the request shape");
                if buffer.kind != InputBufferKind::MapAlias {
                    return Err(call.unsupported(
                        "lm:logger",
                        "active pointer descriptors require an allocated server pointer buffer",
                    ));
                }
                let size = usize::try_from(buffer.size)
                    .ok()
                    .filter(|size| *size <= self.log_sink.max_packet_bytes())
                    .ok_or_else(|| {
                        invalid("logger packet exceeds the local sink's per-packet byte limit")
                    })?;
                let mut packet = vec![0; size];
                call.memory.read(buffer.address, &mut packet)?;
                // Construct the full domain/non-domain reply before log state
                // changes: neither framing failure nor a write fault may retain
                // a packet for a request that could not receive its reply.
                let reply = call.prepare_reply(0, None, LOGGER_INTERFACE_ID, &[], &[])?;
                // This single-thread model cannot change mappings between this
                // preflight and the reply. A reply fault cannot mutate log state.
                call.memory
                    .check_write(call.buffer_address, TLS_MESSAGE_BUFFER_SIZE)?;
                let context = LoggerContext::new(process_id);
                let disposition = self
                    .log_sink
                    .log(&context, &packet)
                    .map_err(|error| invalid(format!("local log sink: {error:#}")))?;
                call.write_reply(&reply)?;
                Ok(Outcome {
                    result: Some(0),
                    log_disposition: Some(format!("{disposition:?}")),
                    created_domain_objects: Vec::new(),
                    output_copy_handle_count: 0,
                })
            }
            1 => {
                call.shape(4, false, false)?;
                let destination = crate::bytes::le_u32(&call.request.arguments[..4]);
                call.reply(0, None, LOGGER_INTERFACE_ID, &[])?;
                self.log_sink.set_destination(destination);
                Ok(Outcome::success())
            }
            _ => Err(call.unsupported("lm:logger", "logger command is not implemented")),
        }
    }

    pub fn report(&self) -> serde_json::Value {
        // Keep each macro expansion bounded while preserving one flat schema.
        let mut report = serde_json::json!({
            "available_named_ports":["sm:"], "process_model":"single_local_process",
            "sessions":self.sessions, "trace":self.trace,
            "trace_limit":MAX_TRACE, "trace_omitted":self.trace_omitted,
            "supported_sm_commands":[0,1], "service_lookup_decoding_only":false,
            "cmif_reply_profile":"generic_sf", "live_sm_cmif_compatibility_verified":false,
            "service_backends":["lm","appletOE"], "filesystem_services_implemented":false,
            "session_close_requests":self.session_close_requests,
            "audio_outs":self.audio_outs,
            "audio_device_list_requests":self.audio_device_list_requests,
            "socket_poll_timeouts":self.socket_poll_timeouts,
            "hid_disconnected_npads":self.hid_disconnected_npads,
            "hid_supported_style_set":self.hid_supported_style_set,
            "video_decoders":self.video_decoders,
            "video_decoding_implemented":false,
            "video_submissions":self.video_submissions,
            "returned_frames":self.returned_frames,
            "decoded_frames":self.decoded_frames,
            "av1_errors":self.av1_errors,
            "video_log":self.video_log,
            "decoder_image_pool_requests":self.decoder_image_pool_requests,
            "refused_video_submissions":self.refused_video_submissions,
            "empty_frame_requests":self.empty_frame_requests,
            "library_applets":self.library_applets,
            "audio_renderers":self.audio_renderers.iter().map(|(renderer, _)| serde_json::json!({
                "sample_rate":renderer.parameters.sample_rate,
                "sample_count":renderer.parameters.sample_count,
                "voices":renderer.parameters.voices,
                "started":renderer.started,
                "frames_rendered":renderer.frames,
                "updates":renderer.updates,
                "mixing_implemented":false,
            })).collect::<Vec<_>>(),
            "terminate_result":self.terminate_result,
            "data_storage_reads":self.data_storage_reads,
            "data_storage_bytes_read":self.data_storage_bytes_read,
            "data_storage_distinct_64k_chunks":self.data_storage_chunks.len(),
            "data_storage_last_offset":self.data_storage_last_offset,
        });
        let application_report = serde_json::json!({
            "main_application":self.main_application,
            "application_identity_source":"loaded_npdm_and_modeled_process",
            "main_application_role_policy":"bootstrap_declared_main_application_npdm_program_type_not_validated",
            "application_proxy_service_model":"command_10_flags_zero_patch_compatibility",
            "application_proxy_open_implemented":true,
            "supported_application_proxy_service_commands":[10],
            "application_proxy_supported_open_flags":[0],
            "supported_application_proxy_commands":[0,1,2,3,4,10,11,20,40,60,1000],
            "application_functions_endpoint_implemented":true,
            "application_functions_model":"immutable_identity_with_application_owned_gpu_error_event",
            "supported_application_functions_commands":[20,21,22,40,130,140,210,320,321,330],
            "application_lifecycle_implemented":false,
            "applet_events_implemented":true,
            "applet_events_complete":false,
            "applet_event_scope":["accumulated_suspended_tick_changed","gpu_error_detected","lifecycle_system"],
            "provisional_unknown_commands":self.provisional_commands,
            "error_contexts_completed":self.error_contexts_completed,
            "parental_control_policy":"no_pin_no_restriction_configured",
            "service_access_enforced":self.service_access.is_some(),
            "time_policy":"host_utc_system_clocks_and_session_monotonic_steady_clock",
            "denied_services":self.denied_services,
            "parental_control_granted_commands":self.parental_control_grants,
            "error_context_bytes":self.error_context_bytes,
            "provisional_unknown_command_inputs":self.provisional_inputs,
            "native_launch_policy":{"operation_mode":"docked","performance_mode":"boost"},
            "application_process_reference_policy":"direct_current_process_typed_borrow_no_general_handle_transfer",
            "pointer_buffer_capacity":0, "logging":self.log_sink.report(),
        });
        let graphics_report = serde_json::json!({
            "graphics_scaling_parameters_functions_endpoint_implemented":true,
            "graphics_scaling_parameters_functions_model":"immutable_identity_only",
            "supported_graphics_scaling_parameters_functions_commands":[10, 1050],
            "graphics_scaling_parameters_policy":"provisional_eden_1a6d7d4_prototype",
            "graphics_rendering_implemented":false,
            "graphics_scaling_implemented":false,
        });
        let performance_report = serde_json::json!({
            "performance_functions_endpoint_implemented":true,
            "performance_functions_model":"stateless_endpoint_only",
            "supported_performance_functions_commands":[],
            "performance_control_implemented":false,
        });
        let library_report = serde_json::json!({
            "library_applet_creator_endpoint_implemented":true,
            "library_applet_creator_model":"host_applets_complete_on_start",
            "supported_library_applet_creator_commands":[0, 3, 10],
            "library_applet_creation_implemented":true,
            "library_applet_storage_implemented":true,
            "library_applet_window_backend_implemented":false,
        });
        let common_report = serde_json::json!({
            "common_state_getter_endpoint_implemented":true,
            "common_state_getter_model":"immutable_identity_with_shared_exit_request_config_and_lifecycle_event",
            "supported_common_state_getter_commands":[0,1,5,6,9,900],
            "request_exit_to_library_applet_at_execute_next_program_enabled":self.main_application_state.as_ref().map(|state| state.request_exit_to_library_applet_at_execute_next_program_enabled),
            // Command 900's exit-request configuration, not the whole endpoint.
            "common_state_exit_request_config_only":true,
            "program_switching_implemented":false,
            "library_applet_exit_implemented":false,
            "common_state_events_implemented":true,
            "common_state_event_scope":"copied_lifecycle_system_event_handle_only",
        });
        let lifecycle_report = serde_json::json!({
            "lifecycle_event_implemented":true,
            "lifecycle_event_model":"application_owned_initially_signaled_for_reference_foreground_focus_notification",
            "main_application_lifecycle_event_id":self.main_application_state.as_ref().map(|state| state.lifecycle_event),
            "main_application_lifecycle_state":self.main_application_state.as_ref().map(|state| state.lifecycle.report()),
            "lifecycle_initialization_policy":"eden_1a6d7d4_foreground_main_application_before_run_not_console_verified",
            "lifecycle_acknowledged_zero_semantics":"unacknowledged_constructor_value_not_a_named_focus_state",
            "lifecycle_message_receiving_implemented":true,
            "lifecycle_focus_query_implemented":true,
            "lifecycle_notification_consumption_implemented":true,
            "lifecycle_dynamic_transitions_implemented":false,
        });
        let self_report = serde_json::json!({
            "self_controller_endpoint_implemented":true,
            "self_controller_model":"application_owned_logical_display_manager_and_suspended_tick_event",
            "supported_self_controller_commands":[10,11,12,13,14,16,19,50,91,2000,2400],
            "main_application_launch_policy":self.main_application_state.as_ref().map(|state| &state.launch_policy),
            "main_application_launch_policy_source":"reference_launcher_application_all_foreground_not_npdm_derived",
            "main_application_display_manager":self.main_application_state.as_ref().and_then(|state| state.display_layer_manager.as_ref()),
            "vi_services_implemented":false,
            "graphics_supervisor_implemented":false,
            "display_visibility_implemented":false,
            "focus_state_implemented":false,
            "display_manager_close_policy":"retain_logical_fields_no_bound_resources_no_full_vi_teardown",
        });
        let event_report = serde_json::json!({
            "suspended_tick_changed_event_implemented":true,
            "suspended_tick_changed_event_model":"application_owned_initially_clear_readable_event_aliases_no_accounting",
            "main_application_suspended_ticks":self.main_application_state.as_ref().map(|state| state.suspended_ticks),
            "main_application_suspended_tick_changed_event_id":self.main_application_state.as_ref().map(|state| state.accumulated_suspended_tick_changed_event),
            "readable_events":self.readable_events,
            "readable_event_aliases":self.readable_event_handles,
            "live_readable_event_aliases":self.readable_event_handles.len(),
            "live_service_sessions":self.sessions.len(),
            "live_service_and_event_handle_limit":MAX_LIVE_HANDLES,
            "readable_event_backing_limit":MAX_READABLE_EVENTS,
            "readable_event_handle_policy":"monotonic_aliases_shared_dynamic_capacity_application_owned_backing",
        });
        let event_limits_report = serde_json::json!({
            "suspension_intervals_implemented":false,
            "suspension_accounting_implemented":false,
            "event_wait_implemented":false,
            "event_reset_signal_implemented":false,
            "event_signal_implemented":false,
            "event_scheduler_implemented":false,
        });
        let handle_report = serde_json::json!({
            "live_thread_handles":self.thread_handles.len(),
            "live_dynamic_handles":self.sessions.len() + self.readable_event_handles.len() + self.thread_handles.len(),
            "dynamic_handle_limit":MAX_LIVE_HANDLES,
            "dynamic_handle_limit_scope":"service_sessions_readable_aliases_and_threads",
            "main_thread_handle_in_dynamic_limit":false,
            "live_service_and_event_handle_limit_scope":"shared_dynamic_limit_including_threads",
        });
        let gpu_event_report = serde_json::json!({
            "gpu_error_detected_event_implemented":true,
            "gpu_error_detected_event_model":"application_owned_initially_clear_no_gpu_workloads_submitted",
            "main_application_gpu_error_detected_event_id":self.main_application_state.as_ref().map(|state| state.gpu_error_detected_event),
            "gpu_error_detection_implemented":false,
            "gpu_workload_submission_implemented":false,
            "gpu_health_verified":false,
        });
        let window_report = serde_json::json!({
            "window_controller_endpoint_implemented":true,
            "window_controller_model":"immutable_identity_with_process_id_query",
            "supported_window_controller_commands":[1,10],
            "window_controller_aruid_source":"addressed_application_process_id",
            "graphics_supervisor_phase":self.graphics_supervisor_phase,
            "nvdrv_open_devices":self.nvdrv_fds,
            "graphics_devices":self.graphics_devices.keys().collect::<Vec<_>>(),
            "gpu_address_spaces":self.gpu_address_spaces.keys().collect::<Vec<_>>(),
            "hid_registration_implemented":false,
            "window_system_backend_implemented":false,
        });
        let audio_report = serde_json::json!({
            "audio_controller_endpoint_implemented":true,
            "audio_controller_model":"per_endpoint_reference_constructor_state",
            "supported_audio_controller_commands":[],
            "audio_controller_state_representation":"ieee754_f32_bits_and_reference_i64_nanoseconds",
            "audio_controller_defaults_source":"eden_1a6d7d4bf2b55feba32f56d2119aeeeeaab19823_constructor_policy_not_console_verified",
            "audio_controller_reference_defaults":AudioControllerState::REFERENCE_DEFAULTS,
            "audio_backend_implemented":false,
            "audio_fade_scheduling_implemented":false,
        });
        let display_report = serde_json::json!({
            "display_controller_endpoint_implemented":true,
            "display_controller_model":"immutable_identity_only",
            "supported_display_controller_commands":[],
            "display_control_implemented":false,
        });
        let winding_report = serde_json::json!({
            "process_winding_controller_endpoint_implemented":true,
            "process_winding_controller_model":"immutable_identity_only",
            "supported_process_winding_controller_commands":[],
            "process_winding_implemented":false,
        });
        let debug_report = serde_json::json!({
            "debug_functions_endpoint_implemented":true,
            "debug_functions_model":"stateless_endpoint_only",
            "supported_debug_functions_commands":[],
            "debug_backend_implemented":false,
        });
        let domain_services = [
            "lm",
            "lm:logger",
            "appletOE",
            "appletOE:application_proxy",
            "appletOE:common_state_getter",
            "appletOE:self_controller",
            "appletOE:window_controller",
            "appletOE:audio_controller",
            "appletOE:display_controller",
            "appletOE:process_winding_controller",
            "appletOE:library_applet_creator",
            "appletOE:application_functions",
            "appletOE:performance_functions",
            "appletOE:graphics_scaling_parameters_functions",
            "appletOE:debug_functions",
        ];
        let domain_report = serde_json::json!({
            "domain_services":domain_services,
            "domain_objects_per_session_limit":MAX_DOMAIN_OBJECTS,
            "domain_id_policy":"local_root_1_monotonic_children_without_reuse",
            "domain_close_policy":"remove_one_entry_missing_ids_noop_request_shaped_reply",
            "domain_missing_target_policy":"explicit_stop_server_close_transport_unmodeled",
            "nested_domain_conversion_implemented":false,
            "domain_copy_clone_controls_implemented":false,
            "private_parse_failures":{
                "records":self.private_parse_failures,
                "omitted":self.parse_failures_omitted,
                "limit":MAX_PRIVATE_PARSE_FAILURES,
                "reason_limit_bytes":MAX_PARSE_FAILURE_REASON_BYTES,
                "tls_snapshot_bytes":TLS_MESSAGE_BUFFER_SIZE,
                "snapshot_scope":"full_already_read_tls_message_buffer_including_potentially_stale_tail",
            },
        });
        let target = report
            .as_object_mut()
            .expect("the report is built as a JSON object");
        for part in [
            application_report,
            graphics_report,
            performance_report,
            library_report,
            common_report,
            lifecycle_report,
            self_report,
            event_report,
            event_limits_report,
            handle_report,
            gpu_event_report,
            window_report,
            audio_report,
            display_report,
            winding_report,
            debug_report,
            domain_report,
        ] {
            let serde_json::Value::Object(fields) = part else {
                unreachable!("service report fragments are object literals");
            };
            target.extend(fields);
        }
        report
    }
}

/// POSIX seconds for a UTC CalendarTime (year u16, month, day, hour,
/// minute, second), via days-from-civil (H. Hinnant).
fn utc_posix_time(calendar: [u8; 8]) -> i64 {
    let year = i64::from(u16::from_le_bytes([calendar[0], calendar[1]]));
    let (month, day) = (i64::from(calendar[2]), i64::from(calendar[3]));
    let year = if month <= 2 { year - 1 } else { year };
    let era = year.div_euclid(400);
    let yoe = year - era * 400;
    let mp = (month + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    days * 86_400
        + i64::from(calendar[4]) * 3600
        + i64::from(calendar[5]) * 60
        + i64::from(calendar[6])
}

/// ToCalendarTime output for a POSIX time in UTC: CalendarTime (year u16,
/// month, day, hour, minute, second, pad) then CalendarAdditionalInfo
/// (day of week u32, day of year u32, zone name [8], DST u32, offset i32).
/// The host's UTC offset (seconds), daylight-saving flag and zone
/// abbreviation at POSIX `time`. Native-PC policy: the game's local time
/// (its day and night) follows the host's time zone, as a console's does
/// its user's.
fn host_local_zone(time: i64) -> (i64, bool, [u8; 8]) {
    static ZONE: std::sync::OnceLock<jiff::tz::TimeZone> = std::sync::OnceLock::new();
    let zone = ZONE.get_or_init(jiff::tz::TimeZone::system);
    let Ok(timestamp) = jiff::Timestamp::from_second(time) else {
        return (0, false, *b"UTC\0\0\0\0\0");
    };
    let info = zone.to_offset_info(timestamp);
    let mut abbreviation = [0u8; 8];
    let name = info.abbreviation().as_bytes();
    let length = name.len().min(7);
    abbreviation[..length].copy_from_slice(&name[..length]);
    (
        i64::from(info.offset().seconds()),
        info.dst().is_dst(),
        abbreviation,
    )
}

/// CalendarTime and CalendarAdditionalInfo of POSIX `time` in the host's
/// local time zone.
fn local_calendar_time(time: i64) -> [u8; 0x20] {
    let (offset, dst, abbreviation) = host_local_zone(time);
    let mut out = utc_calendar_time(time + offset);
    out[16..24].copy_from_slice(&abbreviation);
    out[24..28].copy_from_slice(&u32::from(dst).to_le_bytes());
    out[28..32].copy_from_slice(&(offset as i32).to_le_bytes());
    out
}

/// The POSIX time of a local CalendarTime in the host's time zone.
fn local_posix_time(calendar: [u8; 8]) -> i64 {
    let wall = utc_posix_time(calendar);
    let first = wall - host_local_zone(wall).0;
    wall - host_local_zone(first).0
}

fn utc_calendar_time(time: i64) -> [u8; 0x20] {
    let days = time.div_euclid(86_400);
    let seconds = time.rem_euclid(86_400);
    // Civil-from-days (H. Hinnant), with 0000-03-01 as the era origin.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    const BEFORE: [i64; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let year_day = BEFORE[(month - 1) as usize] + day - 1 + i64::from(leap && month > 2);
    let week_day = (days + 4).rem_euclid(7);
    let mut out = [0u8; 0x20];
    out[0..2].copy_from_slice(&(year as u16).to_le_bytes());
    out[2] = month as u8;
    out[3] = day as u8;
    out[4] = (seconds / 3600) as u8;
    out[5] = (seconds / 60 % 60) as u8;
    out[6] = (seconds % 60) as u8;
    out[8..12].copy_from_slice(&(week_day as u32).to_le_bytes());
    out[12..16].copy_from_slice(&(year_day as u32).to_le_bytes());
    out[16..19].copy_from_slice(b"UTC");
    out
}

/// A VibrationDeviceHandle (u8 style, player, device index, padding) as a
/// motor key: one per side of each player's controller.
fn vibration_device(handle: &[u8]) -> u32 {
    u32::from(handle[1]) << 8 | u32::from(handle[2])
}

#[allow(clippy::unwrap_used)]
#[cfg(all(test, any(target_os = "linux", windows)))]
mod tests {
    use super::*;

    #[test]
    fn decoder_slots_pair_with_submissions_and_input_back_pressure() {
        let mut services = Services::default();
        services.readable_events.insert(
            ReadableEventId(1),
            ReadableEvent {
                owner: MainApplicationIdentity {
                    process_id: 1,
                    program_id: 1,
                    npdm_sha256: String::new(),
                },
                kind: ReadableEventKind::VideoDecoderInput,
                signaled: true,
            },
        );
        services.video_input_event = Some(ReadableEventId(1));
        for id in 0..VIDEO_INPUT_DEPTH as u64 {
            services.video_input.push((
                id,
                VideoFrame {
                    timestamp: id * 100,
                    ..Default::default()
                },
            ));
        }
        services.decode_video_frames();
        // No slot yet: nothing decodes and the full input clears the event.
        assert!(services.decoded_video_frames.is_empty());
        assert!(!services.readable_events[&ReadableEventId(1)].signaled);
        services.decoder_slots.extend([7, 9]);
        services.decode_video_frames();
        assert_eq!(
            services
                .decoded_video_frames
                .iter()
                .map(|(slot, frame)| (*slot, frame.timestamp))
                .collect::<Vec<_>>(),
            [(7, 0), (9, 100)]
        );
        assert_eq!(services.video_input.len(), VIDEO_INPUT_DEPTH - 2);
        assert!(services.readable_events[&ReadableEventId(1)].signaled);
    }

    #[test]
    fn the_last_submission_reaches_its_slot_marked_last() {
        let mut services = Services::default();
        services.video_input.push((1, VideoFrame::default()));
        services.video_input.push((
            2,
            VideoFrame {
                timestamp: u64::MAX,
                last: true,
                picture: None,
            },
        ));
        services.decoder_slots.extend([3, 4]);
        services.decode_video_frames();
        assert_eq!(
            services
                .decoded_video_frames
                .iter()
                .map(|(slot, frame)| (*slot, frame.last))
                .collect::<Vec<_>>(),
            [(3, false), (4, true)]
        );
    }

    #[test]
    fn decoder_fences_complete_with_their_slots_frame_or_on_stop() {
        let mut services = Services::default();
        let owner = MainApplicationIdentity {
            process_id: 1,
            program_id: 1,
            npdm_sha256: String::new(),
        };
        services.readable_events.insert(
            ReadableEventId(1),
            ReadableEvent {
                owner,
                kind: ReadableEventKind::SyncpointEvent { index: 0 },
                signaled: false,
            },
        );
        services.syncpoint_events.insert(0, ReadableEventId(1));
        let token = decoder_fence_token(1);
        services.decoder_fences.insert(token, 0);
        // A wait armed on slot 1's first frame is pending until the frame.
        services.decoder_fence_waits.insert(0, (token, 1));
        assert!(!services.decoder_fence_reached(token, 1));
        services.decoder_slots.push_back(1);
        services.video_input.push((1, VideoFrame::default()));
        services.decode_video_frames();
        assert_eq!(services.decoder_fence_value(token), Some(1));
        assert!(services.decoder_fence_waits.is_empty());
        assert!(services.readable_events[&ReadableEventId(1)].signaled);
        // The next frame is never decoded; stopping releases the fence.
        services
            .readable_events
            .get_mut(&ReadableEventId(1))
            .unwrap()
            .signaled = false;
        services.decoder_fence_waits.insert(0, (token, 2));
        services.release_decoder_fences();
        assert!(services.decoder_fence_reached(token, 2));
        assert!(services.readable_events[&ReadableEventId(1)].signaled);
        // Tokens that are not decoder fences always count as reached.
        assert!(services.decoder_fence_reached(7, u64::MAX));
    }

    #[test]
    fn event_fds_poll_readable_only_with_a_nonzero_counter() {
        let mut fds = BTreeMap::new();
        fds.insert(3, 0u64);
        let pollfd = |fd: i32, events: u16| {
            let mut entry = fd.to_le_bytes().to_vec();
            entry.extend_from_slice(&events.to_le_bytes());
            entry.extend_from_slice(&0u16.to_le_bytes());
            entry
        };
        let revents = |entries: &[u8], index: usize| {
            crate::bytes::le_u16(&entries[index * 8 + 6..index * 8 + 8])
        };
        // POLLIN on an empty eventfd is not ready; an unknown fd is invalid.
        let mut entries = [pollfd(3, 1), pollfd(9, 1)].concat();
        assert_eq!(poll_event_fds(&fds, &mut entries), 1);
        assert_eq!((revents(&entries, 0), revents(&entries, 1)), (0, 0x20));
        fds.insert(3, 2);
        let mut entries = pollfd(3, 1 | 4);
        assert_eq!(poll_event_fds(&fds, &mut entries), 1);
        assert_eq!(revents(&entries, 0), 1 | 4);
    }

    #[test]
    fn opus_packet_durations_follow_the_toc_byte() {
        // CELT 20 ms (config 31), one frame: 960 samples at 48 kHz.
        assert_eq!(opus_packet_samples(&[31 << 3], 48_000), Some(960));
        // SILK 60 ms (config 3), two frames (code 1): 120 ms.
        assert_eq!(opus_packet_samples(&[(3 << 3) | 1], 48_000), Some(5760));
        // CELT 2.5 ms (config 16), code 3 with 4 frames, at 24 kHz.
        assert_eq!(opus_packet_samples(&[(16 << 3) | 3, 4], 24_000), Some(240));
        // More than 120 ms, zero frames and an empty packet are malformed.
        assert_eq!(opus_packet_samples(&[(3 << 3) | 3, 3], 48_000), None);
        assert_eq!(opus_packet_samples(&[(16 << 3) | 3, 0], 48_000), None);
        assert_eq!(opus_packet_samples(&[], 48_000), None);
    }

    #[test]
    fn audio_out_buffers_release_back_to_back_in_real_time() {
        let mut out = AudioOutState {
            sample_rate: 48_000,
            channels: 2,
            started: false,
            volume: 1.0,
            queue: Default::default(),
            released: Default::default(),
            played_samples: 0,
            front_end: None,
            event: None,
            host: None,
        };
        let start = std::time::Instant::now();
        let ms = std::time::Duration::from_millis;
        // 10 ms and 20 ms buffers; nothing plays before Start.
        out.queue.extend([(1, 480), (2, 960)]);
        assert!(!out.advance(start + ms(100)));
        out.started = true;
        assert!(!out.advance(start));
        assert_eq!(out.front_end, Some(start + ms(10)));
        assert!(out.advance(start + ms(10)));
        assert_eq!(out.released, [1]);
        // The second buffer started when the first ended.
        assert_eq!(out.front_end, Some(start + ms(30)));
        assert!(!out.advance(start + ms(29)));
        assert!(out.advance(start + ms(30)));
        assert_eq!(
            (out.released.len(), out.played_samples, out.front_end),
            (2, 1440, None)
        );
    }
    use crate::memory::{READ, STATE_TLS, WRITE};

    #[test]
    fn thread_handle_reservation_is_transactional_and_preserves_handle_types() {
        let mut services = Services::default();
        let before = services.report();
        {
            let reserved = services.reserve_thread_handle(2).unwrap();
            assert_eq!(reserved.handle(), 0x200);
        }
        assert_eq!(services.report(), before);
        assert_eq!(services.next_handle, 0x200);
        assert!(services.reserve_thread_handle(0).is_err());
        assert_eq!(services.report(), before);

        let thread = services.reserve_thread_handle(2).unwrap().commit();
        assert_eq!(thread, 0x200);
        assert_eq!(services.thread_id(thread), Some(2));
        assert!(!services.contains(thread));
        assert!(!services.close(thread));
        assert_eq!(services.thread_id(thread), Some(2));
        assert!(services.reserve_thread_handle(2).is_err());
        let session = services.connect_named_port("sm:", 0).unwrap();
        assert_eq!(session, 0x201);
        assert!(services.contains(session));
        assert_eq!(services.thread_id(session), None);
        assert_eq!(services.release_thread_handle(session), None);
        assert!(services.contains(session));
        assert_eq!(services.release_thread_handle(thread), Some(2));
        assert_eq!(services.release_thread_handle(thread), None);
        assert_eq!(services.thread_id(thread), None);
        assert_eq!(services.reserve_thread_handle(3).unwrap().commit(), 0x202);
        assert_eq!(services.report()["live_thread_handles"], 1);
        assert_eq!(services.report()["live_dynamic_handles"], 2);
        assert_eq!(services.report()["dynamic_handle_limit"], MAX_LIVE_HANDLES);
        assert_eq!(
            services.report()["main_thread_handle_in_dynamic_limit"],
            false
        );
        assert!(services.close(session));
        assert_eq!(services.release_thread_handle(0x202), Some(3));
        // Once a dynamic handle was allocated, closing it does not make the
        // trusted bootstrap registration boundary available again.
        assert!(
            services
                .initialize_main_application(1, 2, &"a".repeat(64))
                .is_err()
        );
    }

    #[test]
    fn thread_handles_share_capacity_with_sessions_and_readable_aliases() {
        let (mut services, mut memory, controller, target) = self_controller_fixture(true);
        let event = copy_suspended_tick_event(&mut services, &mut memory, controller, target);
        let first_thread = services.reserve_thread_handle(2).unwrap().commit();
        assert!(!services.contains(event));
        assert!(!services.contains(first_thread));
        while services.sessions.len()
            + services.readable_event_handles.len()
            + services.thread_handles.len()
            < MAX_LIVE_HANDLES - 1
        {
            services.connect_named_port("sm:", 0).unwrap();
        }
        let last_thread = services.reserve_thread_handle(3).unwrap().commit();
        assert_eq!(services.report()["live_dynamic_handles"], MAX_LIVE_HANDLES);
        let sessions = services.sessions.clone();
        let aliases = services.readable_event_handles.clone();
        let threads = services.thread_handles.clone();
        let backing = services.readable_events.clone();
        let next = services.next_handle;
        assert!(services.reserve_thread_handle(4).is_err());
        assert!(services.connect_named_port("sm:", 0).is_err());
        let packet = proxy_child_packet(target, 91);
        assert!(transact(&mut services, &mut memory, controller, &packet).is_err());
        assert_eq!(services.sessions, sessions);
        assert_eq!(services.readable_event_handles, aliases);
        assert_eq!(services.thread_handles, threads);
        assert_eq!(services.readable_events, backing);
        assert_eq!(services.next_handle, next);
        assert_eq!(services.trace.back().unwrap().output_copy_handle_count, 0);
        let mut unchanged = [0; TLS_MESSAGE_BUFFER_SIZE];
        memory.read(0x1000, &mut unchanged).unwrap();
        assert_eq!(unchanged, packet);

        assert_eq!(services.release_thread_handle(first_thread), Some(2));
        assert_eq!(
            copy_suspended_tick_event(&mut services, &mut memory, controller, target),
            next
        );
        assert!(services.close(event));
        let replacement = services.reserve_thread_handle(4).unwrap().commit();
        assert_eq!(replacement, next + 1);
        assert_ne!(replacement, first_thread);
        assert_eq!(services.thread_id(last_thread), Some(3));
        assert_eq!(services.report()["live_dynamic_handles"], MAX_LIVE_HANDLES);
    }

    #[test]
    fn thread_handle_range_exhaustion_and_live_collisions_never_overwrite() {
        let mut services = Services {
            next_handle: 0x7fff_fffe,
            ..Services::default()
        };
        {
            let reserved = services.reserve_thread_handle(2).unwrap();
            assert_eq!(reserved.handle(), 0x7fff_fffe);
        }
        assert_eq!(services.next_handle, 0x7fff_fffe);
        let handle = services.reserve_thread_handle(2).unwrap().commit();
        assert_eq!(services.next_handle, 0x7fff_ffff);
        let before = services.report();
        assert!(services.reserve_thread_handle(3).is_err());
        assert!(services.connect_named_port("sm:", 0).is_err());
        assert_eq!(services.report(), before);
        assert_eq!(services.release_thread_handle(handle), Some(2));
        assert!(services.reserve_thread_handle(3).is_err());
        assert_eq!(services.next_handle, 0x7fff_ffff);

        let mut services = Services::default();
        let thread = services.reserve_thread_handle(2).unwrap().commit();
        services.next_handle = thread;
        let before = services.report();
        assert!(services.reserve_thread_handle(3).is_err());
        assert!(services.connect_named_port("sm:", 0).is_err());
        assert_eq!(services.report(), before);
        assert_eq!(services.thread_id(thread), Some(2));
    }

    fn memory() -> Memory {
        let mut memory = Memory::new(0x10000);
        memory
            .map(0x1000, 0x1000, READ | WRITE, STATE_TLS, "TLS")
            .unwrap();
        memory
    }

    fn register_packet() -> [u8; TLS_MESSAGE_BUFFER_SIZE] {
        let mut bytes = [0; TLS_MESSAGE_BUFFER_SIZE];
        bytes[..4].copy_from_slice(&0x10u32.to_le_bytes());
        bytes[4..8].copy_from_slice(&0x80000000u32.to_le_bytes());
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
        bytes[12..20].copy_from_slice(&999u64.to_le_bytes());
        bytes
    }

    fn lookup_packet(name: &[u8]) -> [u8; TLS_MESSAGE_BUFFER_SIZE] {
        assert!(name.len() <= 8);
        let mut bytes = [0; TLS_MESSAGE_BUFFER_SIZE];
        bytes[..4].copy_from_slice(&0x11u32.to_le_bytes());
        bytes[4..8].copy_from_slice(&2u32.to_le_bytes());
        bytes[8..8 + name.len()].copy_from_slice(name);
        bytes
    }

    fn cmif_packet(
        tag: u32,
        command: u32,
        arguments: &[u8],
        pid: bool,
    ) -> [u8; TLS_MESSAGE_BUFFER_SIZE] {
        assert!(arguments.len().is_multiple_of(4));
        let mut bytes = [0; TLS_MESSAGE_BUFFER_SIZE];
        bytes[..4].copy_from_slice(&tag.to_le_bytes());
        let raw_words = 8 + arguments.len() as u32 / 4;
        bytes[4..8].copy_from_slice(&(raw_words | if pid { 1 << 31 } else { 0 }).to_le_bytes());
        let header = if pid {
            bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
            bytes[12..20].copy_from_slice(&999u64.to_le_bytes());
            32
        } else {
            16
        };
        bytes[header..header + 4].copy_from_slice(b"SFCI");
        bytes[header + 8..header + 12].copy_from_slice(&command.to_le_bytes());
        bytes[header + 16..header + 16 + arguments.len()].copy_from_slice(arguments);
        bytes
    }

    fn logger_packet(address: u32, size: u32) -> [u8; TLS_MESSAGE_BUFFER_SIZE] {
        let mut bytes = [0; TLS_MESSAGE_BUFFER_SIZE];
        // CMIF request, one inactive X descriptor and one active A descriptor.
        bytes[..4].copy_from_slice(&0x00110004u32.to_le_bytes());
        bytes[4..8].copy_from_slice(&8u32.to_le_bytes());
        bytes[16..20].copy_from_slice(&size.to_le_bytes());
        bytes[20..24].copy_from_slice(&address.to_le_bytes());
        bytes[32..36].copy_from_slice(b"SFCI");
        bytes
    }

    fn domain_packet(
        object_id: u32,
        command: u32,
        arguments: &[u8],
        pid: bool,
        input_objects: &[u32],
    ) -> [u8; TLS_MESSAGE_BUFFER_SIZE] {
        assert!(arguments.len().is_multiple_of(4));
        let mut bytes = [0; TLS_MESSAGE_BUFFER_SIZE];
        bytes[..4].copy_from_slice(&4u32.to_le_bytes());
        let raw_words = 12 + arguments.len() as u32 / 4 + input_objects.len() as u32;
        bytes[4..8].copy_from_slice(&(raw_words | if pid { 1 << 31 } else { 0 }).to_le_bytes());
        let domain = if pid {
            bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
            bytes[12..20].copy_from_slice(&999u64.to_le_bytes());
            32
        } else {
            16
        };
        bytes[domain] = 1;
        bytes[domain + 1] = input_objects.len() as u8;
        bytes[domain + 2..domain + 4]
            .copy_from_slice(&((16 + arguments.len()) as u16).to_le_bytes());
        bytes[domain + 4..domain + 8].copy_from_slice(&object_id.to_le_bytes());
        let cmif = domain + 16;
        bytes[cmif..cmif + 4].copy_from_slice(b"SFCI");
        bytes[cmif + 8..cmif + 12].copy_from_slice(&command.to_le_bytes());
        let data = cmif + 16;
        bytes[data..data + arguments.len()].copy_from_slice(arguments);
        for (index, id) in input_objects.iter().enumerate() {
            let offset = data + arguments.len() + index * 4;
            bytes[offset..offset + 4].copy_from_slice(&id.to_le_bytes());
        }
        bytes
    }

    fn domain_log_packet(object_id: u32, address: u32, size: u32) -> [u8; TLS_MESSAGE_BUFFER_SIZE] {
        let mut bytes = logger_packet(address, size);
        bytes[4..8].copy_from_slice(&12u32.to_le_bytes());
        bytes[32..48].fill(0);
        bytes[32] = 1;
        bytes[34..36].copy_from_slice(&16u16.to_le_bytes());
        bytes[36..40].copy_from_slice(&object_id.to_le_bytes());
        bytes[48..52].copy_from_slice(b"SFCI");
        bytes
    }

    fn domain_close_packet(object_id: u32) -> [u8; TLS_MESSAGE_BUFFER_SIZE] {
        // Padding and TLS tail are deliberately nonzero: close preserves the
        // request-shaped buffer instead of constructing a Result/SFCO response.
        let mut bytes = [0xa5; TLS_MESSAGE_BUFFER_SIZE];
        bytes[..4].copy_from_slice(&4u32.to_le_bytes());
        bytes[4..8].copy_from_slice(&8u32.to_le_bytes());
        bytes[16..32].fill(0);
        bytes[16] = 2;
        bytes[20..24].copy_from_slice(&object_id.to_le_bytes());
        bytes
    }

    fn transact(
        services: &mut Services,
        memory: &mut Memory,
        handle: u32,
        packet: &[u8],
    ) -> Result<(), Trap> {
        memory.write(0x1000, packet)?;
        services.send_sync_request(memory, 0x1000, handle, 7, 0)
    }

    fn open_lm(services: &mut Services, memory: &mut Memory) -> u32 {
        let sm = services.connect_named_port("sm:", 0).unwrap();
        transact(services, memory, sm, &register_packet()).unwrap();
        transact(services, memory, sm, &lookup_packet(b"lm")).unwrap();
        assert_eq!(memory.read_u32(0x1000).unwrap(), 0x11);
        assert_eq!(memory.read_u32(0x1008).unwrap(), 0x20);
        assert_eq!(memory.read_u32(0x1010).unwrap(), 0);
        memory.read_u32(0x100c).unwrap()
    }

    fn open_applet_oe(services: &mut Services, memory: &mut Memory) -> u32 {
        let sm = services.connect_named_port("sm:", 0).unwrap();
        transact(services, memory, sm, &register_packet()).unwrap();
        transact(services, memory, sm, &lookup_packet(b"appletOE")).unwrap();
        assert_eq!(memory.read_u32(0x1000).unwrap(), 0x11);
        assert_eq!(memory.read_u32(0x1008).unwrap(), 0x20);
        assert_eq!(memory.read_u32(0x1010).unwrap(), 0);
        let handle = memory.read_u32(0x100c).unwrap();
        assert_eq!(
            services.sessions[&handle].object(None),
            Some(&ServiceObject::ApplicationProxyService)
        );
        handle
    }

    fn application_services() -> Services {
        let mut services = Services::default();
        services
            .initialize_main_application(7, 0x0100000000123000, &"ab".repeat(32))
            .unwrap();
        services
    }

    fn application_open_arguments() -> [u8; 16] {
        let mut args = [0; 16];
        // The alignment bytes and ClientProcessId placeholder are deliberately
        // nonzero and differ from the trusted sender and raw PID descriptor.
        args[4..8].copy_from_slice(&0xa5b6c7d8u32.to_le_bytes());
        args[8..16].copy_from_slice(&0xfedcba9876543210u64.to_le_bytes());
        args
    }

    fn application_open_packet(
        domain_target: Option<u32>,
        args: &[u8],
        copies: &[u32],
        input_objects: &[u32],
        input_buffer: bool,
    ) -> [u8; TLS_MESSAGE_BUFFER_SIZE] {
        assert!(args.len().is_multiple_of(4));
        assert!(domain_target.is_some() || input_objects.is_empty());
        let mut bytes = [0; TLS_MESSAGE_BUFFER_SIZE];
        let tag = 6u32 | if input_buffer { 0x00110000 } else { 0 };
        bytes[..4].copy_from_slice(&tag.to_le_bytes());
        bytes[8..12].copy_from_slice(&(1 | ((copies.len() as u32) << 1)).to_le_bytes());
        bytes[12..20].copy_from_slice(&0x1122334455667788u64.to_le_bytes());
        let mut raw_start = 20;
        for handle in copies {
            bytes[raw_start..raw_start + 4].copy_from_slice(&handle.to_le_bytes());
            raw_start += 4;
        }
        if input_buffer {
            // Inactive X plus active A, with a bounded unmapped input address.
            // A forbidden buffer must be rejected without reading that address.
            bytes[raw_start + 8..raw_start + 12].copy_from_slice(&1u32.to_le_bytes());
            bytes[raw_start + 12..raw_start + 16].copy_from_slice(&0x3000u32.to_le_bytes());
            raw_start += 20;
        }
        let domain_size = if domain_target.is_some() { 16 } else { 0 };
        let raw_size = 16 + domain_size + 16 + args.len() + input_objects.len() * 4;
        assert!(raw_start + raw_size <= TLS_MESSAGE_BUFFER_SIZE);
        bytes[4..8].copy_from_slice(&(0x80000000 | (raw_size / 4) as u32).to_le_bytes());
        let frame = (raw_start + 15) & !15;
        if let Some(id) = domain_target {
            bytes[frame] = 1;
            bytes[frame + 1] = input_objects.len() as u8;
            bytes[frame + 2..frame + 4].copy_from_slice(&((16 + args.len()) as u16).to_le_bytes());
            bytes[frame + 4..frame + 8].copy_from_slice(&id.to_le_bytes());
        }
        let cmif = frame + domain_size;
        bytes[cmif..cmif + 4].copy_from_slice(b"SFCI");
        bytes[cmif + 4..cmif + 8].copy_from_slice(&1u32.to_le_bytes());
        bytes[cmif + 8..cmif + 12].copy_from_slice(&10u32.to_le_bytes());
        bytes[cmif + 12..cmif + 16].copy_from_slice(&0x98765432u32.to_le_bytes());
        let data = cmif + 16;
        bytes[data..data + args.len()].copy_from_slice(args);
        for (index, id) in input_objects.iter().enumerate() {
            let offset = data + args.len() + index * 4;
            bytes[offset..offset + 4].copy_from_slice(&id.to_le_bytes());
        }
        bytes
    }

    fn open_application_proxy(
        services: &mut Services,
        memory: &mut Memory,
        endpoint: u32,
        domain_target: Option<u32>,
    ) -> u32 {
        let packet = application_open_packet(
            domain_target,
            &application_open_arguments(),
            &[0xffff8001],
            &[],
            false,
        );
        transact(services, memory, endpoint, &packet).unwrap();
        let event = services.trace.back().unwrap();
        assert!(event.completed);
        assert_eq!(event.service_result, Some(0));
        assert_eq!(event.input_copy_handle_count, 1);
        if domain_target.is_some() {
            assert_eq!(memory.read_u32(0x1004).unwrap(), 13);
            assert_eq!(memory.read_u32(0x1010).unwrap(), 1);
            assert_eq!(
                memory.read_u32(0x102c).unwrap(),
                APPLICATION_PROXY_SERVICE_INTERFACE_ID
            );
            memory.read_u32(0x1030).unwrap()
        } else {
            assert_eq!(memory.read_u32(0x1008).unwrap(), 0x20);
            assert_eq!(
                memory.read_u32(0x101c).unwrap(),
                APPLICATION_PROXY_SERVICE_INTERFACE_ID
            );
            memory.read_u32(0x100c).unwrap()
        }
    }

    fn proxy_identity(
        services: &Services,
        handle: u32,
        object_id: Option<u32>,
    ) -> &MainApplicationIdentity {
        match services.sessions[&handle].object(object_id) {
            Some(ServiceObject::ApplicationProxy { application }) => application,
            _ => panic!("expected an application proxy object"),
        }
    }

    #[derive(Clone, Copy)]
    struct ProxyGetterExpectation {
        command: u32,
        service: &'static str,
        kind: &'static str,
        bound_identity: bool,
    }

    // Independent wire/report expectations from the audited interface signatures;
    // do not derive these cases from the production getter factory.
    const PROXY_GETTERS: [ProxyGetterExpectation; 11] = [
        ProxyGetterExpectation {
            command: 0,
            service: "appletOE:common_state_getter",
            kind: "common_state_getter",
            bound_identity: true,
        },
        ProxyGetterExpectation {
            command: 1,
            service: "appletOE:self_controller",
            kind: "self_controller",
            bound_identity: true,
        },
        ProxyGetterExpectation {
            command: 2,
            service: "appletOE:window_controller",
            kind: "window_controller",
            bound_identity: true,
        },
        ProxyGetterExpectation {
            command: 3,
            service: "appletOE:audio_controller",
            kind: "audio_controller",
            bound_identity: false,
        },
        ProxyGetterExpectation {
            command: 4,
            service: "appletOE:display_controller",
            kind: "display_controller",
            bound_identity: true,
        },
        ProxyGetterExpectation {
            command: 10,
            service: "appletOE:process_winding_controller",
            kind: "process_winding_controller",
            bound_identity: true,
        },
        ProxyGetterExpectation {
            command: 11,
            service: "appletOE:library_applet_creator",
            kind: "library_applet_creator",
            bound_identity: true,
        },
        ProxyGetterExpectation {
            command: 20,
            service: "appletOE:application_functions",
            kind: "application_functions",
            bound_identity: true,
        },
        ProxyGetterExpectation {
            command: 40,
            service: "appletOE:performance_functions",
            kind: "performance_functions",
            bound_identity: false,
        },
        ProxyGetterExpectation {
            command: 60,
            service: "appletOE:graphics_scaling_parameters_functions",
            kind: "graphics_scaling_parameters_functions",
            bound_identity: true,
        },
        ProxyGetterExpectation {
            command: 1000,
            service: "appletOE:debug_functions",
            kind: "debug_functions",
            bound_identity: false,
        },
    ];

    fn proxy_child_packet(target: Option<u32>, getter: u32) -> [u8; TLS_MESSAGE_BUFFER_SIZE] {
        let mut packet = if let Some(id) = target {
            domain_packet(id, getter, &[], false, &[])
        } else {
            cmif_packet(6, getter, &[], false)
        };
        packet[..4].copy_from_slice(&6u32.to_le_bytes());
        let cmif = if target.is_some() { 32 } else { 16 };
        packet[cmif + 4..cmif + 8].copy_from_slice(&1u32.to_le_bytes());
        packet[cmif + 12..cmif + 16].copy_from_slice(&0x13579bdfu32.to_le_bytes());
        packet
    }

    fn get_proxy_child(
        services: &mut Services,
        memory: &mut Memory,
        handle: u32,
        target: Option<u32>,
        getter: u32,
    ) -> u32 {
        transact(
            services,
            memory,
            handle,
            &proxy_child_packet(target, getter),
        )
        .unwrap();
        let event = services.trace.back().unwrap();
        assert!(event.completed);
        assert_eq!(event.command_id, getter);
        assert_eq!(event.service_result, Some(0));
        assert_eq!(event.input_copy_handle_count, 0);
        assert!(!event.has_pid_descriptor);
        assert_eq!(event.argument_bytes, 0);
        let cmif = if target.is_some() { 0x1020 } else { 0x1010 };
        assert_eq!(memory.read_u32(0x1000).unwrap(), 0);
        assert_eq!(memory.read_u32(cmif).unwrap(), u32::from_le_bytes(*b"SFCO"));
        assert_eq!(memory.read_u32(cmif + 4).unwrap(), 0);
        assert_eq!(memory.read_u32(cmif + 8).unwrap(), 0);
        assert_eq!(
            memory.read_u32(cmif + 12).unwrap(),
            0x7d59_c154 // Replying IApplicationProxy, not the returned interface.
        );
        if target.is_some() {
            assert_eq!(memory.read_u32(0x1004).unwrap(), 13); // No move-handle header.
            assert_eq!(memory.read_u32(0x1010).unwrap(), 1);
            let id = memory.read_u32(0x1030).unwrap();
            assert_eq!(event.created_domain_objects, [id]);
            id
        } else {
            assert_eq!(memory.read_u32(0x1004).unwrap(), 0x80000008);
            assert_eq!(memory.read_u32(0x1008).unwrap(), 0x20);
            assert!(event.created_domain_objects.is_empty());
            memory.read_u32(0x100c).unwrap()
        }
    }

    fn assert_proxy_child(
        services: &Services,
        handle: u32,
        object_id: Option<u32>,
        expected_kind: &str,
        identity: Option<&MainApplicationIdentity>,
    ) {
        let actual = services.sessions[&handle].object(object_id).unwrap();
        let expected = if expected_kind == "audio_controller" {
            assert!(identity.is_none());
            // Literal constructor expectation, independent of the production
            // default constant; there must be no application identity field.
            serde_json::json!({
                "kind":"audio_controller",
                "state":{
                    "main_applet_volume_bits":0x3e80_0000u32,
                    "library_applet_volume_bits":0x3f80_0000u32,
                    "transparent_volume_rate_bits":0,
                    "reference_fade_duration_ns":0,
                },
            })
        } else if let Some(application) = identity {
            serde_json::json!({"kind":expected_kind,"application":application})
        } else {
            serde_json::json!({"kind":expected_kind})
        };
        assert_eq!(serde_json::to_value(actual).unwrap(), expected);
    }

    fn assert_window_aruid_reply(memory: &Memory, domain: bool, process_id: u64) {
        // Independent literal CMIF layout: no special/handle header, no output
        // objects, exactly one u64. Request context tokens are not reply IIDs.
        let mut expected = [0; TLS_MESSAGE_BUFFER_SIZE];
        expected[4..8].copy_from_slice(&(if domain { 14u32 } else { 10 }).to_le_bytes());
        let cmif = if domain { 32 } else { 16 };
        expected[cmif..cmif + 4].copy_from_slice(b"SFCO");
        expected[cmif + 12..cmif + 16].copy_from_slice(&0x0279_422du32.to_le_bytes());
        expected[cmif + 16..cmif + 24].copy_from_slice(&process_id.to_le_bytes());
        let mut actual = [0; TLS_MESSAGE_BUFFER_SIZE];
        memory.read(0x1000, &mut actual).unwrap();
        assert_eq!(actual, expected);
    }

    fn assert_common_exit_request_reply(memory: &Memory, domain: bool) {
        let mut expected = [0; TLS_MESSAGE_BUFFER_SIZE];
        // Empty success: no special header, handles, scalar outputs or domain
        // objects. CMIF version/result and all framing padding remain zero.
        expected[4..8].copy_from_slice(&(if domain { 12u32 } else { 8 }).to_le_bytes());
        let cmif = if domain { 32 } else { 16 };
        expected[cmif..cmif + 4].copy_from_slice(b"SFCO");
        expected[cmif + 12..cmif + 16].copy_from_slice(&0x32f2_7d77u32.to_le_bytes());
        let mut actual = [0; TLS_MESSAGE_BUFFER_SIZE];
        memory.read(0x1000, &mut actual).unwrap();
        assert_eq!(actual, expected);
    }

    fn application_child_fixture(
        domain: bool,
        getter: u32,
    ) -> (Services, Memory, u32, Option<u32>) {
        let mut services = application_services();
        let mut memory = memory();
        let endpoint = open_applet_oe(&mut services, &mut memory);
        let proxy = open_application_proxy(&mut services, &mut memory, endpoint, None);
        let controller = get_proxy_child(&mut services, &mut memory, proxy, None, getter);
        if domain {
            assert_eq!(convert_domain(&mut services, &mut memory, controller), 1);
        }
        (services, memory, controller, domain.then_some(1))
    }

    fn self_controller_fixture(domain: bool) -> (Services, Memory, u32, Option<u32>) {
        application_child_fixture(domain, 1)
    }

    fn copy_suspended_tick_event(
        services: &mut Services,
        memory: &mut Memory,
        handle: u32,
        target: Option<u32>,
    ) -> u32 {
        copy_event_reply(
            services,
            memory,
            handle,
            target,
            91,
            0x56ef_eaa1,
            "appletOE:self_controller",
        )
    }

    fn copy_gpu_error_event(
        services: &mut Services,
        memory: &mut Memory,
        handle: u32,
        target: Option<u32>,
    ) -> u32 {
        copy_event_reply(
            services,
            memory,
            handle,
            target,
            130,
            0x21f9_610e,
            "appletOE:application_functions",
        )
    }

    fn copy_lifecycle_event(
        services: &mut Services,
        memory: &mut Memory,
        handle: u32,
        target: Option<u32>,
    ) -> u32 {
        copy_event_reply(
            services,
            memory,
            handle,
            target,
            0,
            0x32f2_7d77,
            "appletOE:common_state_getter",
        )
    }

    #[test]
    fn receive_message_pops_the_launch_focus_change_then_reports_no_messages() {
        for domain in [false, true] {
            let mut services = application_services();
            let mut memory = memory();
            let endpoint = open_applet_oe(&mut services, &mut memory);
            if domain {
                convert_domain(&mut services, &mut memory, endpoint);
            }
            let proxy =
                open_application_proxy(&mut services, &mut memory, endpoint, domain.then_some(1));
            let (proxy_handle, proxy_target) = if domain {
                (endpoint, Some(proxy))
            } else {
                (proxy, None)
            };
            let common = get_proxy_child(&mut services, &mut memory, proxy_handle, proxy_target, 0);
            let (handle, target) = if domain {
                (endpoint, Some(common))
            } else {
                (common, None)
            };
            let cmif = if domain { 32 } else { 16 } + 0x1000;
            let signaled =
                |services: &Services| services.readable_events[&LIFECYCLE_EVENT_ID].signaled;
            assert!(signaled(&services));
            let sessions = services.sessions.clone();
            // First ReceiveMessage: FocusStateChanged, after which the event clears.
            transact(
                &mut services,
                &mut memory,
                handle,
                &proxy_child_packet(target, 1),
            )
            .unwrap();
            assert_eq!(memory.read_u32(cmif).unwrap(), u32::from_le_bytes(*b"SFCO"));
            assert_eq!(memory.read_u32(cmif + 8).unwrap(), 0);
            assert_eq!(
                memory.read_u32(cmif + 12).unwrap(),
                COMMON_STATE_GETTER_INTERFACE_ID
            );
            assert_eq!(
                memory.read_u32(cmif + 16).unwrap(),
                APPLET_MESSAGE_FOCUS_STATE_CHANGED
            );
            assert!(!signaled(&services));
            let lifecycle = services
                .main_application_state
                .as_ref()
                .unwrap()
                .lifecycle
                .clone();
            assert!(!lifecycle.has_focus_state_changed && !lifecycle.applet_message_available);
            // Second ReceiveMessage: nothing pending, AM ResultNoMessages.
            transact(
                &mut services,
                &mut memory,
                handle,
                &proxy_child_packet(target, 1),
            )
            .unwrap();
            assert_eq!(memory.read_u32(cmif + 8).unwrap(), 0x680);
            assert!(!signaled(&services));
            // GetCurrentFocusState acknowledges and returns InFocus as a u8.
            assert_eq!(lifecycle.acknowledged_focus_state, 0);
            transact(
                &mut services,
                &mut memory,
                handle,
                &proxy_child_packet(target, 9),
            )
            .unwrap();
            assert_eq!(memory.read_u32(cmif + 8).unwrap(), 0);
            assert_eq!(memory.read_u8(cmif + 16).unwrap(), 1);
            let lifecycle = services
                .main_application_state
                .as_ref()
                .unwrap()
                .lifecycle
                .clone();
            assert_eq!(lifecycle.acknowledged_focus_state, 1);
            assert!(!signaled(&services));
            assert_eq!(services.sessions, sessions);
        }
    }

    #[test]
    fn utc_calendar_time_converts_posix_seconds() {
        let out = super::utc_calendar_time(1_709_210_096); // 2024-02-29 12:34:56 Thu
        assert_eq!(&out[..7], &[0xe8, 0x07, 2, 29, 12, 34, 56]);
        assert_eq!(crate::bytes::le_u32(&out[8..12]), 4);
        assert_eq!(crate::bytes::le_u32(&out[12..16]), 59);
        for time in [0, 951_782_400, 1_709_210_096, 4_102_444_799] {
            let calendar = super::utc_calendar_time(time);
            assert_eq!(
                super::utc_posix_time(calendar[..8].try_into().unwrap()),
                time
            );
        }
        let epoch = super::utc_calendar_time(0);
        assert_eq!(&epoch[..7], &[0xb2, 0x07, 1, 1, 0, 0, 0]);
        assert_eq!(crate::bytes::le_u32(&epoch[8..12]), 4);
    }

    #[test]
    fn time_shared_memory_matches_the_reference_layout() {
        let mut services = application_services();
        services.steady_clock_source = [0x5a; 16];
        let object = services.time_shared_memory(1_700_000_100, 100);
        let bytes = &object.contents;
        assert_eq!(
            (object.size, object.permission, bytes.len()),
            (0x1000, 1, 0x1000)
        );
        let i64_at = |at: usize| crate::bytes::le_i64(&bytes[at..at + 8]);
        // Counters are zero, so slot 0 of every field is the live value.
        for counter in [0x00usize, 0x38, 0x80, 0xc8, 0xd0] {
            assert_eq!(&bytes[counter..counter + 4], &[0; 4]);
        }
        assert_eq!(i64_at(0x08), 0);
        assert_eq!(&bytes[0x10..0x20], &[0x5a; 16]);
        for base in [0x38usize, 0x80] {
            assert_eq!(i64_at(base + 0x08), 1_700_000_000);
            assert_eq!(i64_at(base + 0x10), 100);
            assert_eq!(&bytes[base + 0x18..base + 0x28], &[0x5a; 16]);
        }
        assert_eq!(bytes[0xcc], 0);
        for field in [0xe0usize, 0xe8, 0xf0, 0xf8] {
            assert_eq!(i64_at(field), 0, "{field:#x}");
        }
        assert_eq!(&bytes[0x100..0x110], &[0x5a; 16]);
        assert!(bytes[0x110..].iter().all(|byte| *byte == 0));
    }

    fn copy_event_reply(
        services: &mut Services,
        memory: &mut Memory,
        handle: u32,
        target: Option<u32>,
        command: u32,
        interface_id: u32,
        service: &str,
    ) -> u32 {
        transact(
            services,
            memory,
            handle,
            &proxy_child_packet(target, command),
        )
        .unwrap();
        let alias = memory.read_u32(0x100c).unwrap();
        let mut expected = [0; TLS_MESSAGE_BUFFER_SIZE];
        // One COPY slot, no move slots/PID/scalar data/domain objects. The
        // special header and copied handle lie outside the declared raw words.
        expected[4..8].copy_from_slice(
            &(if target.is_some() {
                0x8000_000cu32
            } else {
                0x8000_0008
            })
            .to_le_bytes(),
        );
        expected[8..12].copy_from_slice(&2u32.to_le_bytes());
        expected[12..16].copy_from_slice(&alias.to_le_bytes());
        let cmif = if target.is_some() { 32 } else { 16 };
        expected[cmif..cmif + 4].copy_from_slice(b"SFCO");
        expected[cmif + 12..cmif + 16].copy_from_slice(&interface_id.to_le_bytes());
        let mut actual = [0; TLS_MESSAGE_BUFFER_SIZE];
        memory.read(0x1000, &mut actual).unwrap();
        assert_eq!(actual, expected);
        let event = services.trace.back().unwrap();
        assert_eq!(event.service, service);
        assert_eq!(event.command_id, command);
        assert!(event.completed);
        assert_eq!(event.service_result, Some(0));
        assert_eq!(event.output_copy_handle_count, 1);
        assert!(event.created_domain_objects.is_empty());
        alias
    }

    fn open_logger(services: &mut Services, memory: &mut Memory, lm: u32) -> u32 {
        transact(
            services,
            memory,
            lm,
            &cmif_packet(4, 0, &888u64.to_le_bytes(), true),
        )
        .unwrap();
        memory.read_u32(0x100c).unwrap()
    }

    fn convert_domain(services: &mut Services, memory: &mut Memory, handle: u32) -> u32 {
        transact(services, memory, handle, &cmif_packet(5, 0, &[], false)).unwrap();
        assert_eq!(memory.read_u32(0x1000).unwrap(), 0);
        assert_eq!(memory.read_u32(0x1004).unwrap(), 9);
        assert_eq!(
            memory.read_u32(0x1010).unwrap(),
            u32::from_le_bytes(*b"SFCO")
        );
        assert_eq!(memory.read_u32(0x1018).unwrap(), 0);
        assert_eq!(memory.read_u32(0x101c).unwrap(), HIPC_MANAGER_INTERFACE_ID);
        memory.read_u32(0x1020).unwrap()
    }

    fn open_domain_logger(
        services: &mut Services,
        memory: &mut Memory,
        lm: u32,
        root_object: u32,
    ) -> u32 {
        transact(
            services,
            memory,
            lm,
            &domain_packet(root_object, 0, &888u64.to_le_bytes(), true, &[]),
        )
        .unwrap();
        assert_eq!(memory.read_u32(0x1000).unwrap(), 0);
        assert_eq!(memory.read_u32(0x1004).unwrap(), 13); // No special/move-handle header.
        assert_eq!(memory.read_u32(0x1010).unwrap(), 1); // One output domain object.
        assert_eq!(
            memory.read_u32(0x1020).unwrap(),
            u32::from_le_bytes(*b"SFCO")
        );
        assert_eq!(memory.read_u32(0x1028).unwrap(), 0);
        assert_eq!(memory.read_u32(0x102c).unwrap(), LM_INTERFACE_ID);
        memory.read_u32(0x1030).unwrap()
    }

    fn domain_state(services: &Services, handle: u32) -> (&BTreeMap<u32, ServiceObject>, u32) {
        match &services.sessions[&handle] {
            Session::Domain {
                objects,
                next_object_id,
                ..
            } => (objects, *next_object_id),
            Session::Regular { .. } => panic!("expected a converted domain session"),
        }
    }

    fn registration(services: &Services, handle: u32) -> Option<u64> {
        match services.sessions[&handle].object(None) {
            Some(ServiceObject::ServiceManager {
                registered_process_id,
            }) => *registered_process_id,
            _ => panic!("expected an ordinary service-manager session"),
        }
    }

    fn valid_log_packet() -> Vec<u8> {
        let mut packet = vec![0; 27];
        packet[..8].copy_from_slice(&999u64.to_le_bytes());
        packet[8..16].copy_from_slice(&42u64.to_le_bytes());
        packet[16] = 7;
        packet[20..24].copy_from_slice(&3u32.to_le_bytes());
        packet[24..].copy_from_slice(b"abc");
        packet
    }

    #[test]
    fn main_application_identity_is_bounded_one_time_bootstrap_metadata() {
        for hash in [
            String::new(),
            "a".repeat(63),
            "a".repeat(65),
            "g".repeat(64),
            "é".repeat(32),
        ] {
            let mut services = Services::default();
            assert!(services.initialize_main_application(7, 1, &hash).is_err());
            assert!(services.main_application.is_none());
            assert!(services.main_application_state.is_none());
            assert!(services.sessions.is_empty());
            assert!(services.readable_events.is_empty());
            assert!(services.readable_event_handles.is_empty());
            assert_eq!(services.next_handle, FIRST_SESSION_HANDLE);
        }
        let mut services = application_services();
        let identity = services.main_application.clone();
        let application_state = services.main_application_state.clone();
        for (pid, program) in [(7, 0x0100000000123000), (8, 2)] {
            assert!(
                services
                    .initialize_main_application(pid, program, &"ab".repeat(32))
                    .is_err()
            );
            assert_eq!(services.main_application, identity);
            assert_eq!(services.main_application_state, application_state);
        }
        let report = services.report();
        assert_eq!(report["main_application"]["process_id"], 7);
        assert_eq!(
            report["main_application"]["program_id_hex"],
            "0100000000123000"
        );
        assert_eq!(report["main_application"]["npdm_sha256"], "ab".repeat(32));
        assert_eq!(
            report["application_identity_source"],
            "loaded_npdm_and_modeled_process"
        );
        assert_eq!(
            report["supported_application_proxy_commands"],
            serde_json::json!([0, 1, 2, 3, 4, 10, 11, 20, 40, 60, 1000])
        );
        assert_eq!(
            report["main_application_launch_policy"],
            serde_json::json!({"applet_id":1,"library_applet_mode":0})
        );
        assert!(report["main_application_display_manager"].is_null());
        assert_eq!(
            report["supported_application_functions_commands"],
            serde_json::json!([20, 21, 22, 40, 130, 140, 210, 320, 321, 330])
        );
        assert_eq!(report["application_lifecycle_implemented"], false);
        assert_eq!(report["applet_events_implemented"], true);
        assert_eq!(report["applet_events_complete"], false);
        assert_eq!(
            report["applet_event_scope"],
            serde_json::json!([
                "accumulated_suspended_tick_changed",
                "gpu_error_detected",
                "lifecycle_system"
            ])
        );
        for close_first in [false, true] {
            let mut services = Services::default();
            let handle = services.connect_named_port("sm:", 0).unwrap();
            if close_first {
                assert!(services.close(handle));
            }
            assert!(
                services
                    .initialize_main_application(7, 1, &"ab".repeat(32))
                    .is_err()
            );
            assert!(services.main_application.is_none());
            assert!(services.main_application_state.is_none());
        }
    }

    #[test]
    fn regular_application_proxies_share_identity_without_parent_handle_lifetime() {
        let mut services = application_services();
        let mut memory = memory();
        let endpoint = open_applet_oe(&mut services, &mut memory);
        let identity = services.main_application.clone().unwrap();
        let first = open_application_proxy(&mut services, &mut memory, endpoint, None);
        let second = open_application_proxy(&mut services, &mut memory, endpoint, None);
        assert_ne!(first, second);
        assert_ne!(first, endpoint);
        assert_eq!(proxy_identity(&services, first, None), &identity);
        assert_eq!(proxy_identity(&services, second, None), &identity);
        assert!(services.close(endpoint));
        assert!(services.contains(first) && services.contains(second));
        let packet = cmif_packet(4, 21, &[], false);
        let before = services.sessions.clone();
        assert!(matches!(
            transact(&mut services, &mut memory, first, &packet),
            Err(Trap::UnsupportedService { ref service, command_id: Some(21), .. })
                if service == "appletOE:application_proxy"
        ));
        assert_eq!(services.sessions, before);
        assert!(services.close(first));
        assert!(services.contains(second));
        assert_eq!(services.main_application, Some(identity));
    }

    #[test]
    fn domain_application_proxies_survive_root_close_and_share_boot_identity() {
        let mut services = application_services();
        let mut memory = memory();
        let endpoint = open_applet_oe(&mut services, &mut memory);
        convert_domain(&mut services, &mut memory, endpoint);
        let identity = services.main_application.clone().unwrap();
        let next_handle = services.next_handle;
        let first = open_application_proxy(&mut services, &mut memory, endpoint, Some(1));
        let second = open_application_proxy(&mut services, &mut memory, endpoint, Some(1));
        assert_eq!((first, second), (2, 3));
        assert_eq!(services.next_handle, next_handle);
        assert_eq!(proxy_identity(&services, endpoint, Some(first)), &identity);
        assert_eq!(proxy_identity(&services, endpoint, Some(second)), &identity);
        transact(
            &mut services,
            &mut memory,
            endpoint,
            &domain_close_packet(1),
        )
        .unwrap();
        let packet = domain_packet(first, 21, &[], false, &[]);
        let before = services.sessions.clone();
        assert!(matches!(
            transact(&mut services, &mut memory, endpoint, &packet),
            Err(Trap::UnsupportedService { ref service, command_id: Some(21), .. })
                if service == "appletOE:application_proxy"
        ));
        assert_eq!(services.sessions, before);
        transact(
            &mut services,
            &mut memory,
            endpoint,
            &domain_close_packet(first),
        )
        .unwrap();
        assert_eq!(domain_state(&services, endpoint).0.len(), 1);
        assert_eq!(proxy_identity(&services, endpoint, Some(second)), &identity);
        assert!(services.close(endpoint));
        assert!(!services.contains(endpoint));
        assert_eq!(services.main_application, Some(identity));
    }

    #[test]
    fn regular_proxy_children_reply_and_identity_survive_parent_close() {
        for expected in PROXY_GETTERS {
            let getter = expected.command;
            let service_name = expected.service;
            let mut services = application_services();
            let mut memory = memory();
            let endpoint = open_applet_oe(&mut services, &mut memory);
            let proxy = open_application_proxy(&mut services, &mut memory, endpoint, None);
            let identity = proxy_identity(&services, proxy, None).clone();
            let expected_identity = expected.bound_identity.then_some(&identity);
            let first = get_proxy_child(&mut services, &mut memory, proxy, None, getter);
            let second = get_proxy_child(&mut services, &mut memory, proxy, None, getter);
            assert_ne!(first, second);
            assert_ne!(first, proxy);
            assert_proxy_child(&services, first, None, expected.kind, expected_identity);
            assert_proxy_child(&services, second, None, expected.kind, expected_identity);
            assert!(services.close(endpoint));
            assert!(services.close(proxy));
            assert!(services.contains(first) && services.contains(second));
            for command in [
                0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 20, 40, 60, 91, 130, 900, 1000, 1050, 2000,
            ] {
                if (expected.command == 2 && matches!(command, 1 | 10))
                    || (expected.command == 0 && matches!(command, 0 | 1 | 5 | 6 | 9 | 900))
                    || (expected.command == 1
                        && matches!(
                            command,
                            1 | 2 | 9 | 10 | 11 | 12 | 13 | 14 | 16 | 19 | 50 | 91 | 2000 | 2400
                        ))
                    || (expected.command == 11 && matches!(command, 0 | 3 | 10))
                    || (expected.command == 60 && matches!(command, 10 | 1050))
                    || (expected.command == 20
                        && matches!(command, 1 | 20 | 40 | 130 | 320 | 321 | 330))
                {
                    continue; // Only these named child methods are supported.
                }
                let packet = cmif_packet(4, command, &[], false);
                let before = services.sessions.clone();
                assert!(matches!(
                    transact(&mut services, &mut memory, first, &packet),
                    Err(Trap::UnsupportedService { ref service, command_id: Some(id), .. })
                        if service == service_name && id == command
                ));
                assert_eq!(services.sessions, before);
                assert!(!services.trace.back().unwrap().completed);
                assert_eq!(services.trace.back().unwrap().service_result, None);
            }
            assert!(services.close(first));
            assert!(services.contains(second));
            assert_proxy_child(&services, second, None, expected.kind, expected_identity);
            let report = services.report();
            assert_eq!(
                report["supported_application_proxy_commands"],
                serde_json::json!([0, 1, 2, 3, 4, 10, 11, 20, 40, 60, 1000])
            );
            assert_eq!(report["application_functions_endpoint_implemented"], true);
            assert_eq!(
                report["supported_application_functions_commands"],
                serde_json::json!([20, 21, 22, 40, 130, 140, 210, 320, 321, 330])
            );
            assert_eq!(report["application_lifecycle_implemented"], false);
            assert_eq!(report["applet_events_implemented"], true);
            assert_eq!(report["applet_events_complete"], false);
            assert_eq!(
                report["applet_event_scope"],
                serde_json::json!([
                    "accumulated_suspended_tick_changed",
                    "gpu_error_detected",
                    "lifecycle_system"
                ])
            );
            assert_eq!(
                report["graphics_scaling_parameters_functions_endpoint_implemented"],
                true
            );
            assert_eq!(
                report["graphics_scaling_parameters_functions_model"],
                "immutable_identity_only"
            );
            assert_eq!(
                report["supported_graphics_scaling_parameters_functions_commands"],
                serde_json::json!([10, 1050])
            );
            assert_eq!(report["graphics_rendering_implemented"], false);
            assert_eq!(report["graphics_scaling_implemented"], false);
            assert_eq!(report["performance_functions_endpoint_implemented"], true);
            assert_eq!(
                report["performance_functions_model"],
                "stateless_endpoint_only"
            );
            assert_eq!(
                report["supported_performance_functions_commands"],
                serde_json::json!([])
            );
            assert_eq!(report["performance_control_implemented"], false);
            assert_eq!(report["library_applet_creator_endpoint_implemented"], true);
            assert_eq!(
                report["library_applet_creator_model"],
                "host_applets_complete_on_start"
            );
            assert_eq!(
                report["supported_library_applet_creator_commands"],
                serde_json::json!([0, 3, 10])
            );
            assert_eq!(report["library_applet_creation_implemented"], true);
            assert_eq!(report["library_applet_storage_implemented"], true);
            assert_eq!(report["library_applet_window_backend_implemented"], false);
            assert_eq!(report["window_controller_endpoint_implemented"], true);
            assert_eq!(
                report["window_controller_model"],
                "immutable_identity_with_process_id_query"
            );
            assert_eq!(
                report["supported_window_controller_commands"],
                serde_json::json!([1, 10])
            );
            assert_eq!(
                report["window_controller_aruid_source"],
                "addressed_application_process_id"
            );
            assert_eq!(report["hid_registration_implemented"], false);
            assert_eq!(report["window_system_backend_implemented"], false);
            assert_eq!(report["common_state_getter_endpoint_implemented"], true);
            assert_eq!(
                report["common_state_getter_model"],
                "immutable_identity_with_shared_exit_request_config_and_lifecycle_event"
            );
            assert_eq!(
                report["supported_common_state_getter_commands"],
                serde_json::json!([0, 1, 5, 6, 9, 900])
            );
            assert_eq!(report["common_state_exit_request_config_only"], true);
            assert_eq!(report["program_switching_implemented"], false);
            assert_eq!(report["library_applet_exit_implemented"], false);
            assert_eq!(report["common_state_events_implemented"], true);
            for (endpoint, model, commands, expected_model, backend) in [
                (
                    "display_controller_endpoint_implemented",
                    "display_controller_model",
                    "supported_display_controller_commands",
                    "immutable_identity_only",
                    "display_control_implemented",
                ),
                (
                    "process_winding_controller_endpoint_implemented",
                    "process_winding_controller_model",
                    "supported_process_winding_controller_commands",
                    "immutable_identity_only",
                    "process_winding_implemented",
                ),
                (
                    "debug_functions_endpoint_implemented",
                    "debug_functions_model",
                    "supported_debug_functions_commands",
                    "stateless_endpoint_only",
                    "debug_backend_implemented",
                ),
            ] {
                assert_eq!(report[endpoint], true);
                assert_eq!(report[model], expected_model);
                assert_eq!(report[commands], serde_json::json!([]));
                assert_eq!(report[backend], false);
            }
        }
    }

    #[test]
    fn domain_proxy_children_use_object_ids_and_keep_identity_after_parent_close() {
        for expected in PROXY_GETTERS {
            let getter = expected.command;
            let service_name = expected.service;
            let mut services = application_services();
            let mut memory = memory();
            let endpoint = open_applet_oe(&mut services, &mut memory);
            convert_domain(&mut services, &mut memory, endpoint);
            let proxy = open_application_proxy(&mut services, &mut memory, endpoint, Some(1));
            let identity = proxy_identity(&services, endpoint, Some(proxy)).clone();
            let expected_identity = expected.bound_identity.then_some(&identity);
            let next_handle = services.next_handle;
            let first = get_proxy_child(&mut services, &mut memory, endpoint, Some(proxy), getter);
            let second = get_proxy_child(&mut services, &mut memory, endpoint, Some(proxy), getter);
            assert_eq!((first, second), (3, 4));
            assert_eq!(services.next_handle, next_handle);
            for id in [1, proxy] {
                transact(
                    &mut services,
                    &mut memory,
                    endpoint,
                    &domain_close_packet(id),
                )
                .unwrap();
            }
            assert_proxy_child(
                &services,
                endpoint,
                Some(first),
                expected.kind,
                expected_identity,
            );
            assert_proxy_child(
                &services,
                endpoint,
                Some(second),
                expected.kind,
                expected_identity,
            );
            let before = services.sessions.clone();
            for command in [
                0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 20, 40, 60, 91, 130, 900, 1000, 1050, 2000,
            ] {
                if (expected.command == 2 && matches!(command, 1 | 10))
                    || (expected.command == 0 && matches!(command, 0 | 1 | 5 | 6 | 9 | 900))
                    || (expected.command == 1
                        && matches!(
                            command,
                            1 | 2 | 9 | 10 | 11 | 12 | 13 | 14 | 16 | 19 | 50 | 91 | 2000 | 2400
                        ))
                    || (expected.command == 11 && matches!(command, 0 | 3 | 10))
                    || (expected.command == 60 && matches!(command, 10 | 1050))
                    || (expected.command == 20
                        && matches!(command, 1 | 20 | 40 | 130 | 320 | 321 | 330))
                {
                    continue; // Only these named child methods are supported.
                }
                let packet = domain_packet(first, command, &[], false, &[]);
                assert!(matches!(
                    transact(&mut services, &mut memory, endpoint, &packet),
                    Err(Trap::UnsupportedService { ref service, command_id: Some(id), .. })
                        if service == service_name && id == command
                ));
                assert_eq!(services.sessions, before);
            }
            transact(
                &mut services,
                &mut memory,
                endpoint,
                &domain_close_packet(first),
            )
            .unwrap();
            assert_eq!(domain_state(&services, endpoint).0.len(), 1);
            assert_proxy_child(
                &services,
                endpoint,
                Some(second),
                expected.kind,
                expected_identity,
            );
            assert!(services.close(endpoint));
            assert!(!services.contains(endpoint));
            assert_eq!(services.main_application, Some(identity));
        }
    }

    #[test]
    fn proxy_child_getters_reject_all_unexpected_request_inputs() {
        for expected in PROXY_GETTERS {
            let getter = expected.command;
            let mut services = application_services();
            let mut memory = memory();
            let endpoint = open_applet_oe(&mut services, &mut memory);
            convert_domain(&mut services, &mut memory, endpoint);
            let proxy = open_application_proxy(&mut services, &mut memory, endpoint, Some(1));
            let mut copy = proxy_child_packet(Some(proxy), getter);
            copy[4..8].copy_from_slice(&0x8000000cu32.to_le_bytes());
            copy[8..12].copy_from_slice(&2u32.to_le_bytes()); // One copy, no PID.
            copy[12..16].copy_from_slice(&0xffff8001u32.to_le_bytes());
            let mut buffer = domain_log_packet(proxy, 0x3000, 1);
            buffer[56..60].copy_from_slice(&getter.to_le_bytes());
            let before = services.sessions.clone();
            let next = services.next_handle;
            let application_state = services.main_application_state.clone();
            for packet in [
                domain_packet(proxy, getter, &[0; 4], false, &[]),
                domain_packet(proxy, getter, &[], true, &[]),
                domain_packet(proxy, getter, &[], false, &[proxy]),
                copy,
                buffer,
            ] {
                ipc::parse_request_with_domain(&packet, true).unwrap();
                assert!(matches!(
                    transact(&mut services, &mut memory, endpoint, &packet),
                    Err(Trap::InvalidState { .. })
                ));
                assert_eq!(services.sessions, before);
                assert_eq!(services.next_handle, next);
                assert_eq!(services.main_application_state, application_state);
                let mut after = [0; TLS_MESSAGE_BUFFER_SIZE];
                memory.read(0x1000, &mut after).unwrap();
                assert_eq!(after, packet);
            }
            assert!(services.private_parse_failures.is_empty());
            // Unknown/excluded getters remain unsupported even with unexpected
            // inputs; they must not enter the supported getter shape validator.
            for command in [21, 999, 1001] {
                for packet in [
                    domain_packet(proxy, command, &[], false, &[]),
                    domain_packet(proxy, command, &[0; 4], true, &[]),
                ] {
                    assert!(matches!(
                        transact(&mut services, &mut memory, endpoint, &packet),
                        Err(Trap::UnsupportedService { .. })
                    ));
                    assert_eq!(services.sessions, before);
                }
            }
        }
    }

    #[test]
    fn proxy_child_reply_faults_do_not_publish_children_or_consume_ids() {
        for expected in PROXY_GETTERS {
            let getter = expected.command;
            for domain in [false, true] {
                let mut services = application_services();
                let mut memory = memory();
                let endpoint = open_applet_oe(&mut services, &mut memory);
                if domain {
                    convert_domain(&mut services, &mut memory, endpoint);
                }
                let proxy = open_application_proxy(
                    &mut services,
                    &mut memory,
                    endpoint,
                    domain.then_some(1),
                );
                let (handle, target) = if domain {
                    (endpoint, Some(proxy))
                } else {
                    (proxy, None)
                };
                let next_handle = services.next_handle;
                let expected_id = if domain {
                    domain_state(&services, endpoint).1
                } else {
                    next_handle
                };
                let before = services.sessions.clone();
                let identity = services.main_application.clone();
                let application_state = services.main_application_state.clone();
                let packet = proxy_child_packet(target, getter);
                memory.write(0x1000, &packet).unwrap();
                memory.set_permissions(0x1000, 0x1000, READ).unwrap();
                assert!(matches!(
                    services.send_sync_request(&mut memory, 0x1000, handle, 7, 0),
                    Err(Trap::MemoryFault { .. })
                ));
                assert_eq!(services.sessions, before);
                assert_eq!(services.next_handle, next_handle);
                assert_eq!(services.main_application, identity);
                assert_eq!(services.main_application_state, application_state);
                let mut after = [0; TLS_MESSAGE_BUFFER_SIZE];
                memory.read(0x1000, &mut after).unwrap();
                assert_eq!(after, packet);
                memory
                    .set_permissions(0x1000, 0x1000, READ | WRITE)
                    .unwrap();
                assert_eq!(
                    get_proxy_child(&mut services, &mut memory, handle, target, getter),
                    expected_id
                );
            }
        }
    }

    #[test]
    fn common_exit_request_config_is_shared_idempotent_and_survives_endpoint_lifetimes() {
        const FLAG: &str = "request_exit_to_library_applet_at_execute_next_program_enabled";
        assert!(Services::default().report()[FLAG].is_null());
        for domain in [false, true] {
            let mut services = application_services();
            assert_eq!(services.report()[FLAG], false);
            let mut memory = memory();
            let endpoint = open_applet_oe(&mut services, &mut memory);
            if domain {
                convert_domain(&mut services, &mut memory, endpoint);
            }
            let proxy =
                open_application_proxy(&mut services, &mut memory, endpoint, domain.then_some(1));
            let (proxy_handle, proxy_target) = if domain {
                (endpoint, Some(proxy))
            } else {
                (proxy, None)
            };
            let first = get_proxy_child(&mut services, &mut memory, proxy_handle, proxy_target, 0);
            let second = get_proxy_child(&mut services, &mut memory, proxy_handle, proxy_target, 0);
            assert_eq!(services.report()[FLAG], false);
            let mut enabled = services.main_application_state.clone().unwrap();
            enabled.request_exit_to_library_applet_at_execute_next_program_enabled = true;
            let sessions = services.sessions.clone();
            let next_handle = services.next_handle;
            let identity = services.main_application.clone();
            // Repeating on one endpoint and using a sibling both address the
            // same application flag, without creating objects or other state.
            for child in [first, first, second] {
                let (handle, target) = if domain {
                    (endpoint, Some(child))
                } else {
                    (child, None)
                };
                let packet = proxy_child_packet(target, 900);
                transact(&mut services, &mut memory, handle, &packet).unwrap();
                assert_common_exit_request_reply(&memory, domain);
                assert_eq!(services.main_application_state.as_ref(), Some(&enabled));
                assert_eq!(services.main_application, identity);
                assert_eq!(services.sessions, sessions);
                assert_eq!(services.next_handle, next_handle);
                let event = services.trace.back().unwrap();
                assert_eq!(event.service, "appletOE:common_state_getter");
                assert_eq!(event.command_id, 900);
                assert!(event.completed);
                assert_eq!(event.service_result, Some(0));
                assert!(event.created_domain_objects.is_empty());
            }
            let self_first =
                get_proxy_child(&mut services, &mut memory, proxy_handle, proxy_target, 1);
            assert_eq!(services.report()[FLAG], true);
            let self_second =
                get_proxy_child(&mut services, &mut memory, proxy_handle, proxy_target, 1);
            assert_eq!(services.report()[FLAG], true);
            let initialized = services.main_application_state.clone();
            let third = get_proxy_child(&mut services, &mut memory, proxy_handle, proxy_target, 0);
            assert_eq!(services.main_application_state, initialized);
            if domain {
                for object in [1, proxy, first, second, self_first, self_second] {
                    transact(
                        &mut services,
                        &mut memory,
                        endpoint,
                        &domain_close_packet(object),
                    )
                    .unwrap();
                    assert_eq!(services.main_application_state, initialized);
                }
            } else {
                for handle in [endpoint, proxy, first, second, self_first, self_second] {
                    assert!(services.close(handle));
                    assert_eq!(services.main_application_state, initialized);
                }
            }
            let (handle, target) = if domain {
                (endpoint, Some(third))
            } else {
                (third, None)
            };
            transact(
                &mut services,
                &mut memory,
                handle,
                &proxy_child_packet(target, 900),
            )
            .unwrap();
            assert_common_exit_request_reply(&memory, domain);
            assert_eq!(services.main_application_state, initialized);
            if domain {
                transact(
                    &mut services,
                    &mut memory,
                    endpoint,
                    &domain_close_packet(third),
                )
                .unwrap();
                assert!(domain_state(&services, endpoint).0.is_empty());
                assert!(services.close(endpoint));
            } else {
                assert!(services.close(third));
            }
            assert_eq!(services.main_application_state, initialized);
            assert_eq!(services.report()[FLAG], true);
            let report = services.report();
            assert_eq!(
                report["supported_common_state_getter_commands"],
                serde_json::json!([0, 1, 5, 6, 9, 900])
            );
            assert_eq!(report["common_state_exit_request_config_only"], true);
            for key in [
                "program_switching_implemented",
                "library_applet_exit_implemented",
                "application_lifecycle_implemented",
            ] {
                assert_eq!(report[key], false);
            }
        }
    }

    #[test]
    fn common_exit_request_config_rejects_shapes_and_publishes_only_after_a_valid_reply() {
        for domain in [false, true] {
            let mut services = application_services();
            let mut memory = memory();
            let endpoint = open_applet_oe(&mut services, &mut memory);
            if domain {
                convert_domain(&mut services, &mut memory, endpoint);
            }
            let proxy =
                open_application_proxy(&mut services, &mut memory, endpoint, domain.then_some(1));
            let (proxy_handle, proxy_target) = if domain {
                (endpoint, Some(proxy))
            } else {
                (proxy, None)
            };
            get_proxy_child(&mut services, &mut memory, proxy_handle, proxy_target, 1);
            let common = get_proxy_child(&mut services, &mut memory, proxy_handle, proxy_target, 0);
            let (handle, target) = if domain {
                (endpoint, Some(common))
            } else {
                (common, None)
            };
            let mut copy = proxy_child_packet(target, 900);
            copy[4..8]
                .copy_from_slice(&(if domain { 0x8000000cu32 } else { 0x80000008 }).to_le_bytes());
            copy[8..12].copy_from_slice(&2u32.to_le_bytes());
            copy[12..16].copy_from_slice(&0xffff8001u32.to_le_bytes());
            let mut buffer = if domain {
                domain_log_packet(common, 0x3000, 1)
            } else {
                logger_packet(0x3000, 1)
            };
            let command_offset = if domain { 56 } else { 40 };
            buffer[command_offset..command_offset + 4].copy_from_slice(&900u32.to_le_bytes());
            let mut malformed = vec![copy, buffer];
            for (arguments, pid) in [(&[0; 4][..], false), (&[][..], true)] {
                malformed.push(if domain {
                    domain_packet(common, 900, arguments, pid, &[])
                } else {
                    cmif_packet(4, 900, arguments, pid)
                });
            }
            if domain {
                malformed.push(domain_packet(common, 900, &[], false, &[common]));
            }
            let state = services.main_application_state.clone();
            let sessions = services.sessions.clone();
            let identity = services.main_application.clone();
            let next_handle = services.next_handle;
            for packet in malformed {
                ipc::parse_request_with_domain(&packet, domain).unwrap();
                assert!(matches!(
                    transact(&mut services, &mut memory, handle, &packet),
                    Err(Trap::InvalidState { .. })
                ));
                assert_eq!(services.main_application_state, state);
                assert_eq!(services.sessions, sessions);
                assert_eq!(services.main_application, identity);
                assert_eq!(services.next_handle, next_handle);
                let mut after = [0; TLS_MESSAGE_BUFFER_SIZE];
                memory.read(0x1000, &mut after).unwrap();
                assert_eq!(after, packet);
            }
            assert!(services.private_parse_failures.is_empty());
            let packet = proxy_child_packet(target, 900);
            memory.write(0x1000, &packet).unwrap();
            memory.set_permissions(0x1000, 0x1000, READ).unwrap();
            assert!(matches!(
                services.send_sync_request(&mut memory, 0x1000, handle, 7, 0),
                Err(Trap::MemoryFault { .. })
            ));
            assert_eq!(services.main_application_state, state);
            assert_eq!(services.sessions, sessions);
            assert_eq!(services.main_application, identity);
            assert_eq!(services.next_handle, next_handle);
            let mut after = [0; TLS_MESSAGE_BUFFER_SIZE];
            memory.read(0x1000, &mut after).unwrap();
            assert_eq!(after, packet);
            memory
                .set_permissions(0x1000, 0x1000, READ | WRITE)
                .unwrap();

            // Missing local bootstrap metadata and a foreign addressed owner
            // are explicit failures, not successful writes into a replacement.
            services.main_application_state = None;
            assert!(matches!(
                transact(&mut services, &mut memory, handle, &packet),
                Err(Trap::InvalidState { .. })
            ));
            assert!(services.main_application_state.is_none());
            assert_eq!(services.sessions, sessions);
            services.main_application_state = state.clone();
            let ServiceObject::CommonStateGetter { application } = services
                .sessions
                .get_mut(&handle)
                .unwrap()
                .object_mut(target)
                .unwrap()
            else {
                panic!("expected the addressed common-state getter");
            };
            application.process_id = 8;
            let foreign_sessions = services.sessions.clone();
            assert!(matches!(
                transact(&mut services, &mut memory, handle, &packet),
                Err(Trap::InvalidState { .. })
            ));
            assert_eq!(services.main_application_state, state);
            assert_eq!(services.sessions, foreign_sessions);
            assert_eq!(services.next_handle, next_handle);
            memory.read(0x1000, &mut after).unwrap();
            assert_eq!(after, packet);
            let ServiceObject::CommonStateGetter { application } = services
                .sessions
                .get_mut(&handle)
                .unwrap()
                .object_mut(target)
                .unwrap()
            else {
                panic!("expected the addressed common-state getter");
            };
            *application = identity.clone().unwrap();
            transact(&mut services, &mut memory, handle, &packet).unwrap();
            assert_common_exit_request_reply(&memory, domain);
            let mut enabled = state.unwrap();
            enabled.request_exit_to_library_applet_at_execute_next_program_enabled = true;
            assert_eq!(services.main_application_state, Some(enabled));
            assert_eq!(services.sessions, sessions);
            assert_eq!(services.main_application, identity);
            assert_eq!(services.next_handle, next_handle);
        }
    }

    #[test]
    fn audio_constructor_state_is_independent_and_survives_conversion_and_parent_close() {
        let mut services = application_services();
        let mut memory = memory();
        let endpoint = open_applet_oe(&mut services, &mut memory);
        let proxy = open_application_proxy(&mut services, &mut memory, endpoint, None);
        get_proxy_child(&mut services, &mut memory, proxy, None, 1);
        let application = services.main_application.clone();
        let application_state = services.main_application_state.clone();
        let first = get_proxy_child(&mut services, &mut memory, proxy, None, 3);
        let second = get_proxy_child(&mut services, &mut memory, proxy, None, 3);
        assert_ne!(first, second);
        assert_proxy_child(&services, first, None, "audio_controller", None);
        assert_proxy_child(&services, second, None, "audio_controller", None);

        // Synthetic private-state markers only; no supported guest method can
        // set these. Negative zero, a NaN payload and a negative reference
        // duration test exact raw-bit/signed preservation, not console policy.
        let custom = AudioControllerState {
            main_applet_volume_bits: 0x8000_0000,
            library_applet_volume_bits: 0x7fc0_1234,
            transparent_volume_rate_bits: 0x3e00_0000,
            reference_fade_duration_ns: -17,
        };
        let ServiceObject::AudioController { state } = services
            .sessions
            .get_mut(&first)
            .unwrap()
            .object_mut(None)
            .unwrap()
        else {
            panic!("expected an audio controller");
        };
        *state = custom;
        assert_proxy_child(&services, second, None, "audio_controller", None);
        let third = get_proxy_child(&mut services, &mut memory, proxy, None, 3);
        assert_ne!(third, first);
        assert_ne!(third, second);
        assert_proxy_child(&services, third, None, "audio_controller", None);
        assert_eq!(services.main_application, application);
        assert_eq!(services.main_application_state, application_state);

        let report = services.report();
        assert_eq!(report["audio_controller_endpoint_implemented"], true);
        assert_eq!(
            report["audio_controller_model"],
            "per_endpoint_reference_constructor_state"
        );
        assert_eq!(
            report["supported_audio_controller_commands"],
            serde_json::json!([])
        );
        assert_eq!(
            report["audio_controller_state_representation"],
            "ieee754_f32_bits_and_reference_i64_nanoseconds"
        );
        assert_eq!(
            report["audio_controller_defaults_source"],
            "eden_1a6d7d4bf2b55feba32f56d2119aeeeeaab19823_constructor_policy_not_console_verified"
        );
        assert_eq!(
            report["audio_controller_reference_defaults"],
            serde_json::json!({
                "main_applet_volume_bits":0x3e80_0000u32,
                "library_applet_volume_bits":0x3f80_0000u32,
                "transparent_volume_rate_bits":0,
                "reference_fade_duration_ns":0,
            })
        );
        assert_eq!(report["audio_backend_implemented"], false);
        assert_eq!(report["audio_fade_scheduling_implemented"], false);
        assert!(services.close(endpoint));
        assert!(services.close(proxy));
        let next_handle = services.next_handle;
        assert_eq!(convert_domain(&mut services, &mut memory, first), 1);
        assert_eq!(
            services.sessions[&first].object(Some(1)),
            Some(&ServiceObject::AudioController { state: custom })
        );
        assert_eq!(services.next_handle, next_handle);
        assert!(services.close(second));
        assert_eq!(
            services.sessions[&first].object(Some(1)),
            Some(&ServiceObject::AudioController { state: custom })
        );
        assert_proxy_child(&services, third, None, "audio_controller", None);
        transact(&mut services, &mut memory, first, &domain_close_packet(1)).unwrap();
        assert!(domain_state(&services, first).0.is_empty());
        assert!(services.close(first));
        assert_proxy_child(&services, third, None, "audio_controller", None);
        assert!(services.close(third));
        assert_eq!(services.main_application, application);
        assert_eq!(services.main_application_state, application_state);
        assert_eq!(services.next_handle, next_handle);
    }

    #[test]
    fn window_aruid_returns_the_full_bound_identity_after_parent_close() {
        for domain in [false, true] {
            for bound_process_id in [0x1234_5678_9abc_def0u64, 0xfedc_ba98_7654_3210] {
                let mut services = Services::default();
                services
                    .initialize_main_application(
                        bound_process_id,
                        0x0100_0000_0012_3000,
                        &"ab".repeat(32),
                    )
                    .unwrap();
                let original_identity = services.main_application.clone().unwrap();
                let mut memory = memory();
                let endpoint = open_applet_oe(&mut services, &mut memory);
                if domain {
                    convert_domain(&mut services, &mut memory, endpoint);
                }
                let open = application_open_packet(
                    domain.then_some(1),
                    &application_open_arguments(),
                    &[0xffff8001],
                    &[],
                    false,
                );
                memory.write(0x1000, &open).unwrap();
                services
                    .send_sync_request(&mut memory, 0x1000, endpoint, bound_process_id, 0)
                    .unwrap();
                let proxy = memory
                    .read_u32(if domain { 0x1030 } else { 0x100c })
                    .unwrap();
                let (proxy_handle, proxy_target) = if domain {
                    (endpoint, Some(proxy))
                } else {
                    (proxy, None)
                };
                let window =
                    get_proxy_child(&mut services, &mut memory, proxy_handle, proxy_target, 2);
                let (handle, target) = if domain {
                    for id in [1, proxy] {
                        transact(
                            &mut services,
                            &mut memory,
                            endpoint,
                            &domain_close_packet(id),
                        )
                        .unwrap();
                    }
                    (endpoint, Some(window))
                } else {
                    assert!(services.close(endpoint));
                    assert!(services.close(proxy));
                    (window, None)
                };
                assert_proxy_child(
                    &services,
                    handle,
                    target,
                    "window_controller",
                    Some(&original_identity),
                );
                // Synthetic replacement of the private global fixture: there
                // is no supported rebinding API. Existing immutable endpoints
                // must keep their own identity even if that fixture differs.
                let global = services.main_application.as_mut().unwrap();
                global.process_id = 0xdead_beef_0123_4567;
                global.program_id = 0x0100_0000_0099_9000;
                let sessions = services.sessions.clone();
                let application = services.main_application.clone();
                let state = services.main_application_state.clone();
                let next_handle = services.next_handle;
                let packet = proxy_child_packet(target, 1);
                for caller in [0xabcd_ef01_2345_6789, 7] {
                    memory.write(0x1000, &packet).unwrap();
                    services
                        .send_sync_request(&mut memory, 0x1000, handle, caller, 0)
                        .unwrap();
                    assert_window_aruid_reply(&memory, domain, bound_process_id);
                    assert_eq!(services.sessions, sessions);
                    assert_eq!(services.main_application, application);
                    assert_eq!(services.main_application_state, state);
                    assert_eq!(services.next_handle, next_handle);
                    let event = services.trace.back().unwrap();
                    assert!(event.completed);
                    assert_eq!(event.service, "appletOE:window_controller");
                    assert_eq!(event.command_id, 1);
                    assert_eq!(event.service_result, Some(0));
                    assert_eq!(event.domain_object_id, target);
                    assert!(event.created_domain_objects.is_empty());
                    assert_eq!(event.input_copy_handle_count, 0);
                }
            }
        }
    }

    #[test]
    fn window_aruid_shape_and_reply_faults_preserve_objects_and_application_state() {
        for domain in [false, true] {
            let mut services = application_services();
            let mut memory = memory();
            let endpoint = open_applet_oe(&mut services, &mut memory);
            if domain {
                convert_domain(&mut services, &mut memory, endpoint);
            }
            let proxy =
                open_application_proxy(&mut services, &mut memory, endpoint, domain.then_some(1));
            let (proxy_handle, proxy_target) = if domain {
                (endpoint, Some(proxy))
            } else {
                (proxy, None)
            };
            // Keep a real initialized shared record to detect accidental
            // display reinitialization while answering this pure query.
            get_proxy_child(&mut services, &mut memory, proxy_handle, proxy_target, 1);
            let window = get_proxy_child(&mut services, &mut memory, proxy_handle, proxy_target, 2);
            let (handle, target) = if domain {
                (endpoint, Some(window))
            } else {
                (window, None)
            };
            let mut copy = proxy_child_packet(target, 1);
            copy[4..8]
                .copy_from_slice(&(if domain { 0x8000000cu32 } else { 0x80000008 }).to_le_bytes());
            copy[8..12].copy_from_slice(&2u32.to_le_bytes()); // One copy, no PID.
            copy[12..16].copy_from_slice(&0xffff8001u32.to_le_bytes());
            let mut buffer = if domain {
                domain_log_packet(window, 0x3000, 1)
            } else {
                logger_packet(0x3000, 1)
            };
            let command_offset = if domain { 56 } else { 40 };
            buffer[command_offset..command_offset + 4].copy_from_slice(&1u32.to_le_bytes());
            let mut malformed = vec![copy, buffer];
            for (arguments, pid) in [(&[0; 4][..], false), (&[][..], true)] {
                malformed.push(if domain {
                    domain_packet(window, 1, arguments, pid, &[])
                } else {
                    cmif_packet(4, 1, arguments, pid)
                });
            }
            if domain {
                malformed.push(domain_packet(window, 1, &[], false, &[window]));
            }
            let sessions = services.sessions.clone();
            let application = services.main_application.clone();
            let state = services.main_application_state.clone();
            let next_handle = services.next_handle;
            for packet in malformed {
                ipc::parse_request_with_domain(&packet, domain).unwrap();
                assert!(matches!(
                    transact(&mut services, &mut memory, handle, &packet),
                    Err(Trap::InvalidState { .. })
                ));
                assert_eq!(services.sessions, sessions);
                assert_eq!(services.main_application, application);
                assert_eq!(services.main_application_state, state);
                assert_eq!(services.next_handle, next_handle);
                let mut after = [0; TLS_MESSAGE_BUFFER_SIZE];
                memory.read(0x1000, &mut after).unwrap();
                assert_eq!(after, packet);
            }
            assert!(services.private_parse_failures.is_empty());
            let caller_query = proxy_child_packet(target, 2);
            assert!(matches!(
                transact(&mut services, &mut memory, handle, &caller_query),
                Err(Trap::UnsupportedService {
                    command_id: Some(2),
                    ..
                })
            ));
            let packet = proxy_child_packet(target, 1);
            memory.write(0x1000, &packet).unwrap();
            memory.set_permissions(0x1000, 0x1000, READ).unwrap();
            assert!(matches!(
                services.send_sync_request(&mut memory, 0x1000, handle, 7, 0),
                Err(Trap::MemoryFault { .. })
            ));
            assert_eq!(services.sessions, sessions);
            assert_eq!(services.main_application, application);
            assert_eq!(services.main_application_state, state);
            assert_eq!(services.next_handle, next_handle);
            let mut after = [0; TLS_MESSAGE_BUFFER_SIZE];
            memory.read(0x1000, &mut after).unwrap();
            assert_eq!(after, packet);
            memory
                .set_permissions(0x1000, 0x1000, READ | WRITE)
                .unwrap();
            transact(&mut services, &mut memory, handle, &packet).unwrap();
            assert_window_aruid_reply(&memory, domain, 7);
            assert_eq!(services.sessions, sessions);
            assert_eq!(services.main_application, application);
            assert_eq!(services.main_application_state, state);
            assert_eq!(services.next_handle, next_handle);
        }
    }

    #[test]
    fn suspended_tick_event_is_created_at_bootstrap_and_repeated_copies_share_one_backing() {
        let empty = Services::default().report();
        assert!(empty["main_application_suspended_ticks"].is_null());
        assert!(empty["main_application_suspended_tick_changed_event_id"].is_null());
        assert!(empty["main_application_gpu_error_detected_event_id"].is_null());
        assert_eq!(empty["readable_events"], serde_json::json!({}));
        let initialized = application_services();
        assert_eq!(initialized.next_handle, 0x200);
        assert!(initialized.sessions.is_empty());
        assert!(initialized.readable_event_handles.is_empty());
        let report = initialized.report();
        assert_eq!(report["main_application_suspended_ticks"], 0);
        assert_eq!(report["main_application_gpu_error_detected_event_id"], 2);
        assert_eq!(
            report["main_application_suspended_tick_changed_event_id"],
            1
        );
        assert_eq!(
            report["readable_events"],
            serde_json::json!({
                "1":{
                    "owner":initialized.main_application,
                    "kind":"accumulated_suspended_tick_changed",
                    "signaled":false,
                },
                "2":{
                    "owner":initialized.main_application,
                    "kind":"gpu_error_detected",
                    "signaled":false,
                },
                "3":{
                    "owner":initialized.main_application,
                    "kind":"lifecycle_system",
                    "signaled":true,
                },
                "4":{
                    "owner":initialized.main_application,
                    "kind":"add_on_content_list_changed",
                    "signaled":false,
                },
                "5":{
                    "owner":initialized.main_application,
                    "kind":"application_unknown210",
                    "signaled":false,
                },
                "6":{
                    "owner":initialized.main_application,
                    "kind":"friend_invitation_storage_channel",
                    "signaled":false,
                },
            })
        );
        assert_eq!(report["suspended_tick_changed_event_implemented"], true);
        assert_eq!(report["readable_event_backing_limit"], MAX_READABLE_EVENTS);
        assert_eq!(
            report["live_service_and_event_handle_limit"],
            MAX_LIVE_HANDLES
        );
        assert_eq!(report["applet_events_complete"], false);
        assert_eq!(
            report["applet_event_scope"],
            serde_json::json!([
                "accumulated_suspended_tick_changed",
                "gpu_error_detected",
                "lifecycle_system"
            ])
        );
        for key in [
            "suspension_intervals_implemented",
            "suspension_accounting_implemented",
            "event_wait_implemented",
            "event_reset_signal_implemented",
            "event_signal_implemented",
            "event_scheduler_implemented",
            "application_lifecycle_implemented",
        ] {
            assert_eq!(report[key], false);
        }
        for domain in [false, true] {
            let (mut services, mut memory, controller, target) = self_controller_fixture(domain);
            let sessions = services.sessions.clone();
            let state = services.main_application_state.clone();
            let backing = services.readable_events.clone();
            let expected = services.next_handle;
            let first = copy_suspended_tick_event(&mut services, &mut memory, controller, target);
            let second = copy_suspended_tick_event(&mut services, &mut memory, controller, target);
            assert_eq!((first, second), (expected, expected + 1));
            assert_eq!(
                services.readable_event_handles,
                BTreeMap::from([(first, ReadableEventId(1)), (second, ReadableEventId(1)),])
            );
            assert_eq!(services.sessions, sessions); // Domain IDs/capacity unchanged too.
            assert_eq!(services.main_application_state, state);
            assert_eq!(services.readable_events, backing);
            assert!(services.contains(controller));
            assert!(!services.contains(first) && !services.contains(second));
            assert_eq!(services.report()["live_readable_event_aliases"], 2);
            assert!(matches!(
                services.send_sync_request(&mut memory, 0x3000, first, 7, 0),
                Err(Trap::InvalidState { .. })
            )); // Session type rejection occurs before reading unmapped TLS.
            assert!(matches!(
                transact(
                    &mut services,
                    &mut memory,
                    controller,
                    &proxy_child_packet(target, 90)
                ),
                Err(Trap::UnsupportedService {
                    command_id: Some(90),
                    ..
                })
            ));
            assert_eq!(services.trace.back().unwrap().output_copy_handle_count, 0);
            assert!(services.close(first));
            assert!(!services.close(first));
            assert!(services.readable_event_handles.contains_key(&second));
            assert!(services.close(second));
            assert!(services.readable_event_handles.is_empty());
            assert_eq!(services.readable_events, backing);
            assert_eq!(services.main_application_state, state);
        }
    }

    #[test]
    fn suspended_tick_aliases_outlive_sessions_and_shared_state_initializers_preserve_backing() {
        let mut services = application_services();
        let mut memory = memory();
        let endpoint = open_applet_oe(&mut services, &mut memory);
        let regular_proxy = open_application_proxy(&mut services, &mut memory, endpoint, None);
        let regular_self = get_proxy_child(&mut services, &mut memory, regular_proxy, None, 1);
        convert_domain(&mut services, &mut memory, endpoint);
        let domain_proxy = open_application_proxy(&mut services, &mut memory, endpoint, Some(1));
        let domain_self =
            get_proxy_child(&mut services, &mut memory, endpoint, Some(domain_proxy), 1);
        let common = get_proxy_child(&mut services, &mut memory, endpoint, Some(domain_proxy), 0);
        // Synthetic markers distinguish sharing/preservation from reconstruction
        // with zero/clear defaults. No implemented guest method produces these.
        services
            .main_application_state
            .as_mut()
            .unwrap()
            .suspended_ticks = 0x1234_5678_9abc_def0;
        services
            .readable_events
            .get_mut(&ReadableEventId(1))
            .unwrap()
            .signaled = true;
        let backing = services.readable_events.clone();
        let first = copy_suspended_tick_event(&mut services, &mut memory, regular_self, None);
        let second =
            copy_suspended_tick_event(&mut services, &mut memory, endpoint, Some(domain_self));
        assert_eq!(
            services.readable_event_handles[&first],
            services.readable_event_handles[&second]
        );
        assert_eq!(services.readable_events, backing);
        transact(
            &mut services,
            &mut memory,
            endpoint,
            &proxy_child_packet(Some(common), 900),
        )
        .unwrap();
        get_proxy_child(&mut services, &mut memory, regular_proxy, None, 1);
        get_proxy_child(&mut services, &mut memory, endpoint, Some(domain_proxy), 1);
        let state = services.main_application_state.clone();
        assert_eq!(
            state.as_ref().unwrap().suspended_ticks,
            0x1234_5678_9abc_def0
        );
        assert_eq!(
            state
                .as_ref()
                .unwrap()
                .accumulated_suspended_tick_changed_event,
            ReadableEventId(1)
        );
        assert!(
            state
                .as_ref()
                .unwrap()
                .request_exit_to_library_applet_at_execute_next_program_enabled
        );
        assert_eq!(services.readable_events, backing);
        for object in [1, domain_proxy, domain_self] {
            transact(
                &mut services,
                &mut memory,
                endpoint,
                &domain_close_packet(object),
            )
            .unwrap();
            assert_eq!(services.readable_events, backing);
        }
        for handle in services.sessions.keys().copied().collect::<Vec<_>>() {
            assert!(services.close(handle));
        }
        assert!(services.sessions.is_empty());
        assert_eq!(services.readable_event_handles.len(), 2);
        assert_eq!(services.readable_events, backing);
        assert_eq!(services.main_application_state, state);
        assert!(services.close(first));
        assert!(!services.close(first));
        assert!(services.readable_event_handles.contains_key(&second));
        assert!(services.close(second));
        assert!(services.readable_event_handles.is_empty());
        assert_eq!(services.readable_events, backing);
        // New endpoints after all previous sessions/aliases closed still refer
        // to the application-owned object; neither constructor resets it.
        let endpoint = open_applet_oe(&mut services, &mut memory);
        let proxy = open_application_proxy(&mut services, &mut memory, endpoint, None);
        let controller = get_proxy_child(&mut services, &mut memory, proxy, None, 1);
        let last = copy_suspended_tick_event(&mut services, &mut memory, controller, None);
        assert!(last > second);
        assert_eq!(services.readable_event_handles[&last], ReadableEventId(1));
        assert_eq!(services.main_application_state, state);
        assert_eq!(services.readable_events, backing);
    }

    #[test]
    fn suspended_tick_event_shape_and_reply_faults_publish_no_alias_or_allocation() {
        for domain in [false, true] {
            let (mut services, mut memory, handle, target) = self_controller_fixture(domain);
            let mut copy = proxy_child_packet(target, 91);
            copy[4..8].copy_from_slice(
                &(if domain { 0x8000_000cu32 } else { 0x8000_0008 }).to_le_bytes(),
            );
            copy[8..12].copy_from_slice(&2u32.to_le_bytes());
            copy[12..16].copy_from_slice(&0xffff8001u32.to_le_bytes());
            let mut buffer = if domain {
                domain_log_packet(1, 0x3000, 1)
            } else {
                logger_packet(0x3000, 1)
            };
            let command_offset = if domain { 56 } else { 40 };
            buffer[command_offset..command_offset + 4].copy_from_slice(&91u32.to_le_bytes());
            let mut malformed = vec![copy, buffer];
            for (args, pid) in [(&[0; 4][..], false), (&[][..], true)] {
                malformed.push(if domain {
                    domain_packet(1, 91, args, pid, &[])
                } else {
                    cmif_packet(4, 91, args, pid)
                });
            }
            if domain {
                malformed.push(domain_packet(1, 91, &[], false, &[1]));
            }
            let state = services.main_application_state.clone();
            let sessions = services.sessions.clone();
            let backing = services.readable_events.clone();
            let next = services.next_handle;
            for packet in malformed {
                ipc::parse_request_with_domain(&packet, domain).unwrap();
                assert!(matches!(
                    transact(&mut services, &mut memory, handle, &packet),
                    Err(Trap::InvalidState { .. })
                ));
                assert!(services.readable_event_handles.is_empty());
                assert_eq!(services.next_handle, next);
                assert_eq!(services.main_application_state, state);
                assert_eq!(services.sessions, sessions);
                assert_eq!(services.readable_events, backing);
                assert!(!services.trace.back().unwrap().completed);
                assert_eq!(services.trace.back().unwrap().output_copy_handle_count, 0);
                let mut actual = [0; TLS_MESSAGE_BUFFER_SIZE];
                memory.read(0x1000, &mut actual).unwrap();
                assert_eq!(actual, packet);
            }
            let packet = proxy_child_packet(target, 91);
            memory.write(0x1000, &packet).unwrap();
            memory.set_permissions(0x1000, 0x1000, READ).unwrap();
            assert!(matches!(
                services.send_sync_request(&mut memory, 0x1000, handle, 7, 0),
                Err(Trap::MemoryFault { .. })
            ));
            assert!(services.readable_event_handles.is_empty());
            assert_eq!(services.next_handle, next);
            assert_eq!(services.main_application_state, state);
            assert_eq!(services.sessions, sessions);
            assert_eq!(services.readable_events, backing);
            assert!(!services.trace.back().unwrap().completed);
            assert_eq!(services.trace.back().unwrap().output_copy_handle_count, 0);
            let mut actual = [0; TLS_MESSAGE_BUFFER_SIZE];
            memory.read(0x1000, &mut actual).unwrap();
            assert_eq!(actual, packet);
            memory
                .set_permissions(0x1000, 0x1000, READ | WRITE)
                .unwrap();
            assert_eq!(
                copy_suspended_tick_event(&mut services, &mut memory, handle, target),
                next
            );
            assert_eq!(services.main_application_state, state);
            assert_eq!(services.sessions, sessions);
            assert_eq!(services.readable_events, backing);
        }
    }

    #[test]
    fn suspended_tick_event_requires_complete_trusted_owner_and_existing_backing() {
        for domain in [false, true] {
            let (mut services, mut memory, handle, target) = self_controller_fixture(domain);
            let owner = services.main_application.clone();
            let sessions = services.sessions.clone();
            let state = services.main_application_state.clone();
            let backing = services.readable_events.clone();
            let next = services.next_handle;
            for scenario in 0..7 {
                services.sessions = sessions.clone();
                services.main_application = owner.clone();
                services.main_application_state = state.clone();
                services.readable_events = backing.clone();
                match scenario {
                    0..=2 => {
                        let ServiceObject::SelfController { application } = services
                            .sessions
                            .get_mut(&handle)
                            .unwrap()
                            .object_mut(target)
                            .unwrap()
                        else {
                            panic!("expected SelfController")
                        };
                        match scenario {
                            0 => application.process_id ^= 1,
                            1 => application.program_id ^= 1,
                            2 => application.npdm_sha256 = "cd".repeat(32),
                            _ => unreachable!(),
                        }
                    }
                    3 => services.main_application = None,
                    4 => services.main_application_state = None,
                    5 => services.readable_events.clear(),
                    6 => {
                        services
                            .readable_events
                            .get_mut(&ReadableEventId(1))
                            .unwrap()
                            .owner
                            .program_id ^= 1
                    }
                    _ => unreachable!(),
                }
                let before_owner = services.main_application.clone();
                let before_sessions = services.sessions.clone();
                let before_state = services.main_application_state.clone();
                let before_backing = services.readable_events.clone();
                let packet = proxy_child_packet(target, 91);
                assert!(matches!(
                    transact(&mut services, &mut memory, handle, &packet),
                    Err(Trap::InvalidState { .. })
                ));
                assert_eq!(services.main_application, before_owner);
                assert_eq!(services.sessions, before_sessions);
                assert_eq!(services.main_application_state, before_state);
                assert_eq!(services.readable_events, before_backing);
                assert!(services.readable_event_handles.is_empty());
                assert_eq!(services.next_handle, next);
                assert_eq!(services.trace.back().unwrap().output_copy_handle_count, 0);
                assert!(!services.trace.back().unwrap().completed);
                let mut actual = [0; TLS_MESSAGE_BUFFER_SIZE];
                memory.read(0x1000, &mut actual).unwrap();
                assert_eq!(actual, packet);
            }
            services.sessions = sessions;
            services.main_application = owner;
            services.main_application_state = state;
            services.readable_events = backing;
            assert_eq!(
                copy_suspended_tick_event(&mut services, &mut memory, handle, target),
                next
            );
        }
    }

    #[test]
    fn suspended_tick_aliases_share_handle_capacity_without_consuming_domain_slots_or_wrapping() {
        for domain in [false, true] {
            let mut services = application_services();
            let mut memory = memory();
            let endpoint = open_applet_oe(&mut services, &mut memory);
            if domain {
                convert_domain(&mut services, &mut memory, endpoint);
            }
            let proxy =
                open_application_proxy(&mut services, &mut memory, endpoint, domain.then_some(1));
            let (proxy_handle, proxy_target) = if domain {
                (endpoint, Some(proxy))
            } else {
                (proxy, None)
            };
            let controller =
                get_proxy_child(&mut services, &mut memory, proxy_handle, proxy_target, 1);
            let (handle, target) = if domain {
                (endpoint, Some(controller))
            } else {
                (controller, None)
            };
            if domain {
                while domain_state(&services, endpoint).0.len() < MAX_DOMAIN_OBJECTS {
                    get_proxy_child(&mut services, &mut memory, endpoint, Some(proxy), 0);
                }
                assert_eq!(domain_state(&services, endpoint).0.len(), 128);
            }
            let prior_controller_session = services.sessions[&handle].clone();
            let state = services.main_application_state.clone();
            let backing = services.readable_events.clone();
            while services.sessions.len() + services.readable_event_handles.len()
                < MAX_LIVE_HANDLES - 1
            {
                services.connect_named_port("sm:", 0).unwrap();
            }
            let first = copy_suspended_tick_event(&mut services, &mut memory, handle, target);
            assert_eq!(
                services.sessions.len() + services.readable_event_handles.len(),
                MAX_LIVE_HANDLES
            );
            assert_eq!(services.sessions[&handle], prior_controller_session);
            let sessions = services.sessions.clone();
            let aliases = services.readable_event_handles.clone();
            let next = services.next_handle;
            let packet = proxy_child_packet(target, 91);
            assert!(matches!(
                transact(&mut services, &mut memory, handle, &packet),
                Err(Trap::InvalidState { .. })
            ));
            assert!(matches!(
                services.connect_named_port("sm:", 0),
                Err(Trap::InvalidState { .. })
            ));
            assert_eq!(services.sessions, sessions);
            assert_eq!(services.readable_event_handles, aliases);
            assert_eq!(services.next_handle, next);
            assert_eq!(services.main_application_state, state);
            assert_eq!(services.readable_events, backing);
            assert_eq!(services.trace.back().unwrap().output_copy_handle_count, 0);
            let mut actual = [0; TLS_MESSAGE_BUFFER_SIZE];
            memory.read(0x1000, &mut actual).unwrap();
            assert_eq!(actual, packet);
            // Freeing a session allows an event alias, and freeing an alias
            // allows a session. Numeric handles are never reused.
            assert!(services.close(0x200));
            let second = copy_suspended_tick_event(&mut services, &mut memory, handle, target);
            assert_eq!(second, next);
            assert_ne!(second, 0x200);
            assert!(services.close(first));
            let session = services.connect_named_port("sm:", 0).unwrap();
            assert!(session > second);
            assert_eq!(
                services.sessions.len() + services.readable_event_handles.len(),
                MAX_LIVE_HANDLES
            );
            assert_eq!(services.sessions[&handle], prior_controller_session);
            assert_eq!(services.main_application_state, state);
            assert_eq!(services.readable_events, backing);

            let (mut services, mut memory, handle, target) = self_controller_fixture(domain);
            services.next_handle = 0x7fff_fffe;
            let last = copy_suspended_tick_event(&mut services, &mut memory, handle, target);
            assert_eq!(last, 0x7fff_fffe);
            assert_eq!(services.next_handle, 0x7fff_ffff);
            let aliases = services.readable_event_handles.clone();
            let packet = proxy_child_packet(target, 91);
            assert!(matches!(
                transact(&mut services, &mut memory, handle, &packet),
                Err(Trap::InvalidState { .. })
            ));
            assert_eq!(services.readable_event_handles, aliases);
            assert_eq!(services.next_handle, 0x7fff_ffff);
            assert!(services.close(last));
            assert!(matches!(
                transact(&mut services, &mut memory, handle, &packet),
                Err(Trap::InvalidState { .. })
            ));
            assert!(services.readable_event_handles.is_empty());
            assert_eq!(services.next_handle, 0x7fff_ffff);
            assert_eq!(services.readable_events.len(), 6);
        }
    }

    #[test]
    fn gpu_error_aliases_are_distinct_from_suspension_and_survive_controller_lifetimes() {
        for domain in [false, true] {
            let mut services = application_services();
            let mut memory = memory();
            let endpoint = open_applet_oe(&mut services, &mut memory);
            if domain {
                convert_domain(&mut services, &mut memory, endpoint);
            }
            let proxy =
                open_application_proxy(&mut services, &mut memory, endpoint, domain.then_some(1));
            let (proxy_handle, proxy_target) = if domain {
                (endpoint, Some(proxy))
            } else {
                (proxy, None)
            };
            let controller =
                get_proxy_child(&mut services, &mut memory, proxy_handle, proxy_target, 1);
            let functions =
                get_proxy_child(&mut services, &mut memory, proxy_handle, proxy_target, 20);
            let sibling =
                get_proxy_child(&mut services, &mut memory, proxy_handle, proxy_target, 20);
            let common = get_proxy_child(&mut services, &mut memory, proxy_handle, proxy_target, 0);
            let (self_handle, self_target) = if domain {
                (endpoint, Some(controller))
            } else {
                (controller, None)
            };
            let (functions_handle, functions_target) = if domain {
                (endpoint, Some(functions))
            } else {
                (functions, None)
            };
            let (sibling_handle, sibling_target) = if domain {
                (endpoint, Some(sibling))
            } else {
                (sibling, None)
            };
            let sessions = services.sessions.clone();
            let state = services.main_application_state.clone();
            let backing = services.readable_events.clone();
            let suspended =
                copy_suspended_tick_event(&mut services, &mut memory, self_handle, self_target);
            let first = copy_gpu_error_event(
                &mut services,
                &mut memory,
                functions_handle,
                functions_target,
            );
            let second = copy_gpu_error_event(
                &mut services,
                &mut memory,
                functions_handle,
                functions_target,
            );
            let third =
                copy_gpu_error_event(&mut services, &mut memory, sibling_handle, sibling_target);
            assert_eq!(
                services.readable_event_handles,
                BTreeMap::from([
                    (suspended, ReadableEventId(1)),
                    (first, ReadableEventId(2)),
                    (second, ReadableEventId(2)),
                    (third, ReadableEventId(2)),
                ])
            );
            assert!(suspended < first && first < second && second < third);
            assert_eq!(services.sessions, sessions); // Includes domain object table and next ID.
            assert_eq!(services.main_application_state, state);
            assert_eq!(services.readable_events, backing);
            assert!(!services.readable_events[&ReadableEventId(1)].signaled);
            assert!(!services.readable_events[&ReadableEventId(2)].signaled);
            assert_eq!(
                services
                    .main_application_state
                    .as_ref()
                    .unwrap()
                    .suspended_ticks,
                0
            );
            let report = services.report();
            assert_eq!(
                report["supported_application_functions_commands"],
                serde_json::json!([20, 21, 22, 40, 130, 140, 210, 320, 321, 330])
            );
            assert_eq!(
                report["application_functions_model"],
                "immutable_identity_with_application_owned_gpu_error_event"
            );
            assert_eq!(report["gpu_error_detected_event_implemented"], true);
            assert_eq!(
                report["gpu_error_detected_event_model"],
                "application_owned_initially_clear_no_gpu_workloads_submitted"
            );
            assert_eq!(report["main_application_gpu_error_detected_event_id"], 2);
            for key in [
                "gpu_error_detection_implemented",
                "gpu_workload_submission_implemented",
                "gpu_health_verified",
                "graphics_rendering_implemented",
            ] {
                assert_eq!(report[key], false);
            }
            // Conversion retains the existing endpoint and application event;
            // copied aliases are not converted into domain objects.
            if !domain {
                assert_eq!(convert_domain(&mut services, &mut memory, functions), 1);
                let converted =
                    copy_gpu_error_event(&mut services, &mut memory, functions, Some(1));
                assert_eq!(
                    services.readable_event_handles[&converted],
                    ReadableEventId(2)
                );
                assert_eq!(domain_state(&services, functions).0.len(), 1);
                assert_eq!(domain_state(&services, functions).1, 2);
                assert!(services.close(converted));
            }
            // A synthetic signal marker proves shared-state constructors and
            // getters preserve the GPU object, rather than reconstructing clear
            // state. There is no guest GPU producer or signal API in this model.
            services
                .readable_events
                .get_mut(&ReadableEventId(2))
                .unwrap()
                .signaled = true;
            let backing = services.readable_events.clone();
            let (common_handle, common_target) = if domain {
                (endpoint, Some(common))
            } else {
                (common, None)
            };
            transact(
                &mut services,
                &mut memory,
                common_handle,
                &proxy_child_packet(common_target, 900),
            )
            .unwrap();
            get_proxy_child(&mut services, &mut memory, proxy_handle, proxy_target, 1);
            let latest =
                copy_gpu_error_event(&mut services, &mut memory, sibling_handle, sibling_target);
            assert_eq!(services.readable_event_handles[&latest], ReadableEventId(2));
            assert_eq!(services.readable_events, backing);
            let state = services.main_application_state.clone();
            assert_eq!(state.as_ref().unwrap().suspended_ticks, 0);
            assert_eq!(
                state
                    .as_ref()
                    .unwrap()
                    .accumulated_suspended_tick_changed_event,
                ReadableEventId(1)
            );
            if domain {
                for object in [1, proxy, functions, sibling, controller] {
                    transact(
                        &mut services,
                        &mut memory,
                        endpoint,
                        &domain_close_packet(object),
                    )
                    .unwrap();
                    assert_eq!(services.readable_events, backing);
                }
            }
            for handle in services.sessions.keys().copied().collect::<Vec<_>>() {
                assert!(services.close(handle));
            }
            for alias in [first, second, third, latest] {
                assert!(!services.contains(alias));
                assert!(services.close(alias));
                assert!(!services.close(alias));
                assert_eq!(services.readable_events, backing);
            }
            assert_eq!(
                services.readable_event_handles,
                BTreeMap::from([(suspended, ReadableEventId(1))])
            );
            assert!(services.close(suspended));
            assert!(services.readable_event_handles.is_empty());
            assert_eq!(services.main_application_state, state);
            assert_eq!(services.readable_events, backing);
            let endpoint = open_applet_oe(&mut services, &mut memory);
            let proxy = open_application_proxy(&mut services, &mut memory, endpoint, None);
            let functions = get_proxy_child(&mut services, &mut memory, proxy, None, 20);
            let last = copy_gpu_error_event(&mut services, &mut memory, functions, None);
            assert!(last > latest);
            assert_eq!(services.readable_event_handles[&last], ReadableEventId(2));
            assert_eq!(services.main_application_state, state);
            assert_eq!(services.readable_events, backing);
        }
    }

    #[test]
    fn gpu_error_event_rejects_shapes_foreign_owners_wrong_kinds_and_failed_replies() {
        for domain in [false, true] {
            let (mut services, mut memory, handle, target) = application_child_fixture(domain, 20);
            let mut copy = proxy_child_packet(target, 130);
            copy[4..8].copy_from_slice(
                &(if domain { 0x8000_000cu32 } else { 0x8000_0008 }).to_le_bytes(),
            );
            copy[8..12].copy_from_slice(&2u32.to_le_bytes());
            copy[12..16].copy_from_slice(&0xffff8001u32.to_le_bytes());
            let mut buffer = if domain {
                domain_log_packet(1, 0x3000, 1)
            } else {
                logger_packet(0x3000, 1)
            };
            let command_offset = if domain { 56 } else { 40 };
            buffer[command_offset..command_offset + 4].copy_from_slice(&130u32.to_le_bytes());
            let mut malformed = vec![copy, buffer];
            for (args, pid) in [(&[0; 4][..], false), (&[][..], true)] {
                malformed.push(if domain {
                    domain_packet(1, 130, args, pid, &[])
                } else {
                    cmif_packet(4, 130, args, pid)
                });
            }
            if domain {
                malformed.push(domain_packet(1, 130, &[], false, &[1]));
            }
            let state = services.main_application_state.clone();
            let sessions = services.sessions.clone();
            let owner = services.main_application.clone();
            let backing = services.readable_events.clone();
            let next = services.next_handle;
            for packet in malformed {
                ipc::parse_request_with_domain(&packet, domain).unwrap();
                assert!(matches!(
                    transact(&mut services, &mut memory, handle, &packet),
                    Err(Trap::InvalidState { .. })
                ));
                assert_eq!(services.sessions, sessions);
                assert_eq!(services.main_application_state, state);
                assert_eq!(services.readable_events, backing);
                assert!(services.readable_event_handles.is_empty());
                assert_eq!(services.next_handle, next);
                assert!(!services.trace.back().unwrap().completed);
                assert_eq!(services.trace.back().unwrap().output_copy_handle_count, 0);
                let mut actual = [0; TLS_MESSAGE_BUFFER_SIZE];
                memory.read(0x1000, &mut actual).unwrap();
                assert_eq!(actual, packet);
            }
            let packet = proxy_child_packet(target, 130);
            for scenario in 0..9 {
                services.sessions = sessions.clone();
                services.main_application = owner.clone();
                services.main_application_state = state.clone();
                services.readable_events = backing.clone();
                match scenario {
                    0..=2 => {
                        let ServiceObject::ApplicationFunctions { application } = services
                            .sessions
                            .get_mut(&handle)
                            .unwrap()
                            .object_mut(target)
                            .unwrap()
                        else {
                            panic!("expected ApplicationFunctions")
                        };
                        match scenario {
                            0 => application.process_id ^= 1,
                            1 => application.program_id ^= 1,
                            2 => application.npdm_sha256 = "cd".repeat(32),
                            _ => unreachable!(),
                        }
                    }
                    3 => services.main_application = None,
                    4 => services.main_application_state = None,
                    5 => {
                        services.readable_events.remove(&ReadableEventId(2));
                    }
                    6 => {
                        services
                            .readable_events
                            .get_mut(&ReadableEventId(2))
                            .unwrap()
                            .owner
                            .program_id ^= 1
                    }
                    7 => {
                        services
                            .main_application_state
                            .as_mut()
                            .unwrap()
                            .gpu_error_detected_event = ReadableEventId(1)
                    }
                    8 => {
                        services
                            .readable_events
                            .get_mut(&ReadableEventId(2))
                            .unwrap()
                            .kind = ReadableEventKind::AccumulatedSuspendedTickChanged
                    }
                    _ => unreachable!(),
                }
                let before_owner = services.main_application.clone();
                let before_sessions = services.sessions.clone();
                let before_state = services.main_application_state.clone();
                let before_backing = services.readable_events.clone();
                assert!(matches!(
                    transact(&mut services, &mut memory, handle, &packet),
                    Err(Trap::InvalidState { .. })
                ));
                assert_eq!(services.main_application, before_owner);
                assert_eq!(services.sessions, before_sessions);
                assert_eq!(services.main_application_state, before_state);
                assert_eq!(services.readable_events, before_backing);
                assert!(services.readable_event_handles.is_empty());
                assert_eq!(services.next_handle, next);
                assert!(!services.trace.back().unwrap().completed);
                assert_eq!(services.trace.back().unwrap().output_copy_handle_count, 0);
                let mut actual = [0; TLS_MESSAGE_BUFFER_SIZE];
                memory.read(0x1000, &mut actual).unwrap();
                assert_eq!(actual, packet);
            }
            services.sessions = sessions.clone();
            services.main_application = owner;
            services.main_application_state = state.clone();
            services.readable_events = backing.clone();
            memory.write(0x1000, &packet).unwrap();
            memory.set_permissions(0x1000, 0x1000, READ).unwrap();
            assert!(matches!(
                services.send_sync_request(&mut memory, 0x1000, handle, 7, 0),
                Err(Trap::MemoryFault { .. })
            ));
            assert_eq!(services.sessions, sessions);
            assert_eq!(services.main_application_state, state);
            assert_eq!(services.readable_events, backing);
            assert!(services.readable_event_handles.is_empty());
            assert_eq!(services.next_handle, next);
            assert!(!services.trace.back().unwrap().completed);
            assert_eq!(services.trace.back().unwrap().output_copy_handle_count, 0);
            let mut actual = [0; TLS_MESSAGE_BUFFER_SIZE];
            memory.read(0x1000, &mut actual).unwrap();
            assert_eq!(actual, packet);
            memory
                .set_permissions(0x1000, 0x1000, READ | WRITE)
                .unwrap();
            assert_eq!(
                copy_gpu_error_event(&mut services, &mut memory, handle, target),
                next
            );
            assert_eq!(services.readable_event_handles[&next], ReadableEventId(2));
            assert_eq!(services.sessions, sessions);
            assert_eq!(services.main_application_state, state);
            assert_eq!(services.readable_events, backing);
        }
    }

    #[test]
    fn lifecycle_bootstrap_has_pending_focus_and_a_signaled_event_with_no_unordered_messages() {
        let empty = Services::default().report();
        assert!(empty["main_application_lifecycle_event_id"].is_null());
        assert!(empty["main_application_lifecycle_state"].is_null());
        let services = application_services();
        let report = services.report();
        assert_eq!(services.next_handle, 0x200);
        assert!(services.sessions.is_empty());
        assert!(services.readable_event_handles.is_empty());
        assert_eq!(
            services.readable_events.keys().copied().collect::<Vec<_>>(),
            [
                ReadableEventId(1),
                ReadableEventId(2),
                ReadableEventId(3),
                ReadableEventId(4),
                ReadableEventId(5),
                ReadableEventId(6)
            ]
        );
        assert!(!services.readable_events[&ReadableEventId(4)].signaled);
        assert!(!services.readable_events[&ReadableEventId(1)].signaled);
        assert!(!services.readable_events[&ReadableEventId(2)].signaled);
        assert!(services.readable_events[&ReadableEventId(3)].signaled);
        assert_eq!(report["main_application_lifecycle_event_id"], 3);
        assert_eq!(
            report["main_application_lifecycle_state"],
            serde_json::json!({
                "is_application":true,
                "requested_focus_state":1,
                "acknowledged_focus_state":0,
                "focus_state_changed_notification_enabled":true,
                "has_focus_state_changed":true,
                "applet_message_available":true,
                "unordered_message_count":0,
                "focus_handling":null,
            })
        );
        assert_eq!(report["lifecycle_event_implemented"], true);
        assert_eq!(
            report["lifecycle_event_model"],
            "application_owned_initially_signaled_for_reference_foreground_focus_notification"
        );
        assert_eq!(
            report["lifecycle_initialization_policy"],
            "eden_1a6d7d4_foreground_main_application_before_run_not_console_verified"
        );
        assert_eq!(
            report["lifecycle_acknowledged_zero_semantics"],
            "unacknowledged_constructor_value_not_a_named_focus_state"
        );
        assert_eq!(report["common_state_events_implemented"], true);
        assert_eq!(
            report["common_state_event_scope"],
            "copied_lifecycle_system_event_handle_only"
        );
        assert_eq!(
            report["supported_common_state_getter_commands"],
            serde_json::json!([0, 1, 5, 6, 9, 900])
        );
        for key in [
            "lifecycle_message_receiving_implemented",
            "lifecycle_focus_query_implemented",
            "lifecycle_notification_consumption_implemented",
        ] {
            assert_eq!(report[key], true, "{key}");
        }
        for key in [
            "lifecycle_dynamic_transitions_implemented",
            "application_lifecycle_implemented",
            "focus_state_implemented",
            "event_wait_implemented",
            "event_reset_signal_implemented",
            "event_signal_implemented",
            "event_scheduler_implemented",
        ] {
            assert_eq!(report[key], false, "{key}");
        }
    }

    #[test]
    fn lifecycle_event_copies_preserve_notifications_and_backing_across_endpoint_lifetimes() {
        for domain in [false, true] {
            let mut services = application_services();
            let mut memory = memory();
            let endpoint = open_applet_oe(&mut services, &mut memory);
            if domain {
                convert_domain(&mut services, &mut memory, endpoint);
            }
            let proxy =
                open_application_proxy(&mut services, &mut memory, endpoint, domain.then_some(1));
            let (proxy_handle, proxy_target) = if domain {
                (endpoint, Some(proxy))
            } else {
                (proxy, None)
            };
            let first = get_proxy_child(&mut services, &mut memory, proxy_handle, proxy_target, 0);
            let second = get_proxy_child(&mut services, &mut memory, proxy_handle, proxy_target, 0);
            let initial = services.main_application_state.clone();
            let backing = services.readable_events.clone();
            let sessions = services.sessions.clone();
            let next = services.next_handle;
            let mut aliases = Vec::new();
            for child in [first, first, second] {
                let (handle, target) = if domain {
                    (endpoint, Some(child))
                } else {
                    (child, None)
                };
                let alias = copy_lifecycle_event(&mut services, &mut memory, handle, target);
                assert_eq!(alias, next + aliases.len() as u32);
                assert_eq!(services.readable_event_handles[&alias], ReadableEventId(3));
                assert!(!services.contains(alias));
                aliases.push(alias);
                assert_eq!(services.sessions, sessions); // No domain ID or object allocation.
                assert_eq!(services.main_application_state, initial);
                assert_eq!(services.readable_events, backing);
            }
            let (first_handle, first_target) = if domain {
                (endpoint, Some(first))
            } else {
                assert_eq!(convert_domain(&mut services, &mut memory, first), 1);
                let before = services.sessions.clone();
                aliases.push(copy_lifecycle_event(
                    &mut services,
                    &mut memory,
                    first,
                    Some(1),
                ));
                assert_eq!(services.sessions, before);
                assert_eq!(services.main_application_state, initial);
                (first, Some(1))
            };

            // Synthetic divergence proves the getter does not derive the event
            // bit from cached availability. No guest reset operation is exposed.
            services
                .readable_events
                .get_mut(&ReadableEventId(3))
                .unwrap()
                .signaled = false;
            let backing = services.readable_events.clone();
            let lifecycle = services
                .main_application_state
                .as_ref()
                .unwrap()
                .lifecycle
                .clone();
            assert!(lifecycle.applet_message_available && lifecycle.has_focus_state_changed);
            assert!(lifecycle.unordered_messages.is_empty());
            let self_first =
                get_proxy_child(&mut services, &mut memory, proxy_handle, proxy_target, 1);
            let self_second =
                get_proxy_child(&mut services, &mut memory, proxy_handle, proxy_target, 1);
            transact(
                &mut services,
                &mut memory,
                first_handle,
                &proxy_child_packet(first_target, 900),
            )
            .unwrap();
            assert_eq!(
                services.main_application_state.as_ref().unwrap().lifecycle,
                lifecycle
            );
            assert_eq!(services.readable_events, backing);
            assert!(
                services
                    .main_application_state
                    .as_ref()
                    .unwrap()
                    .request_exit_to_library_applet_at_execute_next_program_enabled
            );
            let third = get_proxy_child(&mut services, &mut memory, proxy_handle, proxy_target, 0);
            let retained = services.main_application_state.clone();
            if domain {
                for object in [1, proxy, first, second, self_first, self_second] {
                    transact(
                        &mut services,
                        &mut memory,
                        endpoint,
                        &domain_close_packet(object),
                    )
                    .unwrap();
                    assert_eq!(services.main_application_state, retained);
                    assert_eq!(services.readable_events, backing);
                }
            } else {
                for handle in [endpoint, proxy, first, second, self_first, self_second] {
                    assert!(services.close(handle));
                    assert_eq!(services.main_application_state, retained);
                    assert_eq!(services.readable_events, backing);
                }
            }
            let (handle, target) = if domain {
                (endpoint, Some(third))
            } else {
                (third, None)
            };
            aliases.push(copy_lifecycle_event(
                &mut services,
                &mut memory,
                handle,
                target,
            ));
            assert_eq!(services.main_application_state, retained);
            assert_eq!(services.readable_events, backing);
            let before_aliases = services.readable_event_handles.clone();
            let next = services.next_handle;
            for command in [7, 8] {
                assert!(matches!(
                    transact(&mut services, &mut memory, handle, &proxy_child_packet(target, command)),
                    Err(Trap::UnsupportedService { command_id: Some(id), .. }) if id == command
                ));
                assert_eq!(services.main_application_state, retained);
                assert_eq!(services.readable_events, backing);
                assert_eq!(services.readable_event_handles, before_aliases);
                assert_eq!(services.next_handle, next);
            }
            if domain {
                transact(
                    &mut services,
                    &mut memory,
                    endpoint,
                    &domain_close_packet(third),
                )
                .unwrap();
                assert!(domain_state(&services, endpoint).0.is_empty());
            }
            for session in services.sessions.keys().copied().collect::<Vec<_>>() {
                assert!(services.close(session));
            }
            assert_eq!(services.readable_event_handles.len(), aliases.len());
            for alias in aliases {
                assert_eq!(services.readable_event_handles[&alias], ReadableEventId(3));
                assert!(services.close(alias));
                assert!(!services.close(alias));
                assert_eq!(services.main_application_state, retained);
                assert_eq!(services.readable_events, backing);
            }
            assert!(services.readable_event_handles.is_empty());
            assert_eq!(services.readable_events.len(), 6);
        }
    }

    #[test]
    fn lifecycle_event_rejects_shapes_owners_types_and_reply_faults_without_publication() {
        for domain in [false, true] {
            let (mut services, mut memory, handle, target) = application_child_fixture(domain, 0);
            let existing = copy_lifecycle_event(&mut services, &mut memory, handle, target);
            let mut copy = proxy_child_packet(target, 0);
            copy[4..8].copy_from_slice(
                &(if domain { 0x8000_000cu32 } else { 0x8000_0008 }).to_le_bytes(),
            );
            copy[8..12].copy_from_slice(&2u32.to_le_bytes());
            copy[12..16].copy_from_slice(&0xffff8001u32.to_le_bytes());
            let buffer = if domain {
                domain_log_packet(1, 0x3000, 1)
            } else {
                logger_packet(0x3000, 1)
            };
            let mut malformed = vec![copy, buffer];
            for (args, pid) in [(&[0; 4][..], false), (&[][..], true)] {
                malformed.push(if domain {
                    domain_packet(1, 0, args, pid, &[])
                } else {
                    cmif_packet(4, 0, args, pid)
                });
            }
            if domain {
                malformed.push(domain_packet(1, 0, &[], false, &[1]));
            }
            let state = services.main_application_state.clone();
            let sessions = services.sessions.clone();
            let owner = services.main_application.clone();
            let backing = services.readable_events.clone();
            let aliases = services.readable_event_handles.clone();
            let next = services.next_handle;
            for packet in malformed {
                ipc::parse_request_with_domain(&packet, domain).unwrap();
                assert!(matches!(
                    transact(&mut services, &mut memory, handle, &packet),
                    Err(Trap::InvalidState { .. })
                ));
                assert_eq!(services.sessions, sessions);
                assert_eq!(services.main_application_state, state);
                assert_eq!(services.readable_events, backing);
                assert_eq!(services.readable_event_handles, aliases);
                assert_eq!(services.next_handle, next);
                assert!(!services.trace.back().unwrap().completed);
                assert_eq!(services.trace.back().unwrap().output_copy_handle_count, 0);
                let mut actual = [0; TLS_MESSAGE_BUFFER_SIZE];
                memory.read(0x1000, &mut actual).unwrap();
                assert_eq!(actual, packet);
            }
            let packet = proxy_child_packet(target, 0);
            for scenario in 0..9 {
                services.sessions = sessions.clone();
                services.main_application = owner.clone();
                services.main_application_state = state.clone();
                services.readable_events = backing.clone();
                match scenario {
                    0..=2 => {
                        let ServiceObject::CommonStateGetter { application } = services
                            .sessions
                            .get_mut(&handle)
                            .unwrap()
                            .object_mut(target)
                            .unwrap()
                        else {
                            panic!("expected CommonStateGetter")
                        };
                        match scenario {
                            0 => application.process_id ^= 1,
                            1 => application.program_id ^= 1,
                            2 => application.npdm_sha256 = "cd".repeat(32),
                            _ => unreachable!(),
                        }
                    }
                    3 => services.main_application = None,
                    4 => services.main_application_state = None,
                    5 => {
                        services.readable_events.remove(&ReadableEventId(3));
                    }
                    6 => {
                        services
                            .readable_events
                            .get_mut(&ReadableEventId(3))
                            .unwrap()
                            .owner
                            .program_id ^= 1
                    }
                    7 => {
                        services
                            .main_application_state
                            .as_mut()
                            .unwrap()
                            .lifecycle_event = ReadableEventId(1)
                    }
                    8 => {
                        services
                            .readable_events
                            .get_mut(&ReadableEventId(3))
                            .unwrap()
                            .kind = ReadableEventKind::GpuErrorDetected
                    }
                    _ => unreachable!(),
                }
                let before_owner = services.main_application.clone();
                let before_sessions = services.sessions.clone();
                let before_state = services.main_application_state.clone();
                let before_backing = services.readable_events.clone();
                assert!(matches!(
                    transact(&mut services, &mut memory, handle, &packet),
                    Err(Trap::InvalidState { .. })
                ));
                assert_eq!(services.main_application, before_owner);
                assert_eq!(services.sessions, before_sessions);
                assert_eq!(services.main_application_state, before_state);
                assert_eq!(services.readable_events, before_backing);
                assert_eq!(services.readable_event_handles, aliases);
                assert_eq!(services.next_handle, next);
                assert!(!services.trace.back().unwrap().completed);
                assert_eq!(services.trace.back().unwrap().output_copy_handle_count, 0);
                let mut actual = [0; TLS_MESSAGE_BUFFER_SIZE];
                memory.read(0x1000, &mut actual).unwrap();
                assert_eq!(actual, packet);
            }
            services.sessions = sessions.clone();
            services.main_application = owner;
            services.main_application_state = state.clone();
            services.readable_events = backing.clone();
            memory.write(0x1000, &packet).unwrap();
            memory.set_permissions(0x1000, 0x1000, READ).unwrap();
            assert!(matches!(
                services.send_sync_request(&mut memory, 0x1000, handle, 7, 0),
                Err(Trap::MemoryFault { .. })
            ));
            assert_eq!(services.sessions, sessions);
            assert_eq!(services.main_application_state, state);
            assert_eq!(services.readable_events, backing);
            assert_eq!(services.readable_event_handles, aliases);
            assert_eq!(services.next_handle, next);
            assert!(!services.trace.back().unwrap().completed);
            assert_eq!(services.trace.back().unwrap().output_copy_handle_count, 0);
            let mut actual = [0; TLS_MESSAGE_BUFFER_SIZE];
            memory.read(0x1000, &mut actual).unwrap();
            assert_eq!(actual, packet);
            memory
                .set_permissions(0x1000, 0x1000, READ | WRITE)
                .unwrap();
            assert_eq!(
                copy_lifecycle_event(&mut services, &mut memory, handle, target),
                next
            );
            assert_eq!(
                services.readable_event_handles[&existing],
                ReadableEventId(3)
            );
            assert_eq!(services.readable_event_handles[&next], ReadableEventId(3));
            assert_eq!(services.sessions, sessions);
            assert_eq!(services.main_application_state, state);
            assert_eq!(services.readable_events, backing);
        }
    }

    #[test]
    fn lifecycle_aliases_share_session_event_and_thread_capacity_without_domain_allocation() {
        for domain in [false, true] {
            let (mut services, mut memory, handle, target) = application_child_fixture(domain, 0);
            let first = copy_lifecycle_event(&mut services, &mut memory, handle, target);
            let thread = services.reserve_thread_handle(2).unwrap().commit();
            while services.sessions.len()
                + services.readable_event_handles.len()
                + services.thread_handles.len()
                < MAX_LIVE_HANDLES - 1
            {
                services.connect_named_port("sm:", 0).unwrap();
            }
            let before_session = services.sessions[&handle].clone();
            let last = copy_lifecycle_event(&mut services, &mut memory, handle, target);
            assert_eq!(services.sessions[&handle], before_session);
            assert_eq!(services.report()["live_dynamic_handles"], MAX_LIVE_HANDLES);
            assert_eq!(services.thread_id(thread), Some(2));
            assert!(!services.contains(thread));
            assert!(!services.contains(first));
            assert!(!services.contains(last));
            let sessions = services.sessions.clone();
            let aliases = services.readable_event_handles.clone();
            let threads = services.thread_handles.clone();
            let state = services.main_application_state.clone();
            let backing = services.readable_events.clone();
            let next = services.next_handle;
            let packet = proxy_child_packet(target, 0);
            assert!(matches!(
                transact(&mut services, &mut memory, handle, &packet),
                Err(Trap::InvalidState { .. })
            ));
            assert!(services.reserve_thread_handle(3).is_err());
            assert!(services.connect_named_port("sm:", 0).is_err());
            assert_eq!(services.sessions, sessions);
            assert_eq!(services.readable_event_handles, aliases);
            assert_eq!(services.thread_handles, threads);
            assert_eq!(services.main_application_state, state);
            assert_eq!(services.readable_events, backing);
            assert_eq!(services.next_handle, next);
            assert_eq!(services.trace.back().unwrap().output_copy_handle_count, 0);
            let mut actual = [0; TLS_MESSAGE_BUFFER_SIZE];
            memory.read(0x1000, &mut actual).unwrap();
            assert_eq!(actual, packet);

            assert_eq!(services.release_thread_handle(thread), Some(2));
            let retried = copy_lifecycle_event(&mut services, &mut memory, handle, target);
            assert_eq!(retried, next);
            assert_ne!(retried, thread); // Freed numeric handles are not reused.
            assert_eq!(
                services.readable_event_handles[&retried],
                ReadableEventId(3)
            );
            assert_eq!(services.sessions, sessions);
            assert_eq!(services.report()["live_dynamic_handles"], MAX_LIVE_HANDLES);
            assert!(services.close(first));
            let new_thread = services.reserve_thread_handle(3).unwrap().commit();
            assert!(new_thread > retried);
            assert_eq!(services.thread_id(new_thread), Some(3));
            assert!(!services.readable_event_handles.contains_key(&new_thread));
            assert_eq!(services.sessions[&handle], before_session);
            assert_eq!(services.main_application_state, state);
            assert_eq!(services.readable_events, backing);
            assert_eq!(services.report()["live_dynamic_handles"], MAX_LIVE_HANDLES);
        }
    }

    #[test]
    fn self_controllers_share_application_state_across_sessions_and_closes() {
        let mut services = application_services();
        let mut memory = memory();
        let identity = services.main_application.clone().unwrap();
        assert!(
            services
                .main_application_state
                .as_ref()
                .unwrap()
                .display_layer_manager
                .is_none()
        );
        let endpoint = open_applet_oe(&mut services, &mut memory);
        let regular_proxy = open_application_proxy(&mut services, &mut memory, endpoint, None);
        convert_domain(&mut services, &mut memory, endpoint);
        let domain_proxy = open_application_proxy(&mut services, &mut memory, endpoint, Some(1));
        let regular_child = get_proxy_child(&mut services, &mut memory, regular_proxy, None, 1);
        let initialized = services.main_application_state.clone();
        let domain_child =
            get_proxy_child(&mut services, &mut memory, endpoint, Some(domain_proxy), 1);
        // Different sessions/endpoints use the one application record. The
        // second initialization does not allocate a second display world.
        assert_eq!(services.main_application_state, initialized);
        assert_proxy_child(
            &services,
            regular_child,
            None,
            "self_controller",
            Some(&identity),
        );
        assert_proxy_child(
            &services,
            endpoint,
            Some(domain_child),
            "self_controller",
            Some(&identity),
        );
        let report = services.report();
        assert_eq!(report["self_controller_endpoint_implemented"], true);
        assert_eq!(
            report["self_controller_model"],
            "application_owned_logical_display_manager_and_suspended_tick_event"
        );
        assert_eq!(
            report["supported_self_controller_commands"],
            serde_json::json!([10, 11, 12, 13, 14, 16, 19, 50, 91, 2000, 2400])
        );
        assert_eq!(
            report["main_application_launch_policy_source"],
            "reference_launcher_application_all_foreground_not_npdm_derived"
        );
        assert_eq!(
            report["main_application_display_manager"],
            serde_json::json!({
                "owner":identity,
                "applet_id":1,"library_applet_mode":0,
                "shared_buffer_id":null,"shared_layer_id":null,
                "buffer_sharing_enabled":false,"blending_enabled":false,
                "managed_display_layers":[],"managed_recording_layers":[],
            })
        );
        for key in [
            "vi_services_implemented",
            "graphics_supervisor_implemented",
            "display_visibility_implemented",
            "focus_state_implemented",
            "graphics_rendering_implemented",
        ] {
            assert_eq!(report[key], false);
        }
        assert_eq!(
            report["display_manager_close_policy"],
            "retain_logical_fields_no_bound_resources_no_full_vi_teardown"
        );
        assert!(services.close(regular_proxy));
        for object_id in [1, domain_proxy] {
            transact(
                &mut services,
                &mut memory,
                endpoint,
                &domain_close_packet(object_id),
            )
            .unwrap();
            assert_eq!(services.main_application_state, initialized);
        }
        assert!(services.close(regular_child));
        assert_proxy_child(
            &services,
            endpoint,
            Some(domain_child),
            "self_controller",
            Some(&identity),
        );
        assert_eq!(services.main_application_state, initialized);
        transact(
            &mut services,
            &mut memory,
            endpoint,
            &domain_close_packet(domain_child),
        )
        .unwrap();
        assert!(domain_state(&services, endpoint).0.is_empty());
        assert!(services.close(endpoint));
        // With no bound VI manager, reference Finalize is a no-op; owner and
        // policy do not disappear with the last SelfController endpoint.
        assert_eq!(services.main_application_state, initialized);
        assert_eq!(services.main_application, Some(identity));
    }

    #[test]
    fn self_controller_reinitialization_is_transactional_and_preserves_managed_sets() {
        for domain in [false, true] {
            let mut services = application_services();
            let mut memory = memory();
            let endpoint = open_applet_oe(&mut services, &mut memory);
            if domain {
                convert_domain(&mut services, &mut memory, endpoint);
            }
            let proxy =
                open_application_proxy(&mut services, &mut memory, endpoint, domain.then_some(1));
            let (handle, target) = if domain {
                (endpoint, Some(proxy))
            } else {
                (proxy, None)
            };
            get_proxy_child(&mut services, &mut memory, handle, target, 1);
            let manager = services
                .main_application_state
                .as_mut()
                .unwrap()
                .display_layer_manager
                .as_mut()
                .unwrap();
            // Synthetic logical markers, not real resources: no supported
            // guest method can produce this state. They distinguish the
            // reference's scalar reset from its preservation of managed sets.
            manager.applet_id = 99;
            manager.library_applet_mode = 3;
            manager.shared_buffer_id = Some(17);
            manager.shared_layer_id = Some(19);
            manager.buffer_sharing_enabled = true;
            manager.blending_enabled = true;
            manager.managed_display_layers.insert(23);
            manager.managed_recording_layers.insert(29);
            let prior_state = services.main_application_state.clone();
            let prior_sessions = services.sessions.clone();
            let next_handle = services.next_handle;
            let expected_id = if domain {
                domain_state(&services, endpoint).1
            } else {
                next_handle
            };
            let packet = proxy_child_packet(target, 1);
            memory.write(0x1000, &packet).unwrap();
            memory.set_permissions(0x1000, 0x1000, READ).unwrap();
            assert!(matches!(
                services.send_sync_request(&mut memory, 0x1000, handle, 7, 0),
                Err(Trap::MemoryFault { .. })
            ));
            assert_eq!(services.main_application_state, prior_state);
            assert_eq!(services.sessions, prior_sessions);
            assert_eq!(services.next_handle, next_handle);
            let mut after = [0; TLS_MESSAGE_BUFFER_SIZE];
            memory.read(0x1000, &mut after).unwrap();
            assert_eq!(after, packet);
            memory
                .set_permissions(0x1000, 0x1000, READ | WRITE)
                .unwrap();
            assert_eq!(
                get_proxy_child(&mut services, &mut memory, handle, target, 1),
                expected_id
            );
            let manager = services
                .main_application_state
                .as_ref()
                .unwrap()
                .display_layer_manager
                .as_ref()
                .unwrap();
            assert_eq!(Some(&manager.owner), services.main_application.as_ref());
            assert_eq!((manager.applet_id, manager.library_applet_mode), (1, 0));
            assert_eq!(
                (manager.shared_buffer_id, manager.shared_layer_id),
                (None, None)
            );
            assert!(!manager.buffer_sharing_enabled && !manager.blending_enabled);
            assert_eq!(manager.managed_display_layers, BTreeSet::from([23]));
            assert_eq!(manager.managed_recording_layers, BTreeSet::from([29]));

            // A corrupted local endpoint must not bind a foreign owner to the
            // one trusted application. There is no raw IPC PID to trust here.
            let ServiceObject::ApplicationProxy { application } = services
                .sessions
                .get_mut(&handle)
                .unwrap()
                .object_mut(target)
                .unwrap()
            else {
                panic!("expected the addressed application proxy");
            };
            application.process_id = 8;
            let prior_state = services.main_application_state.clone();
            let prior_sessions = services.sessions.clone();
            let next_handle = services.next_handle;
            assert!(matches!(
                transact(&mut services, &mut memory, handle, &packet),
                Err(Trap::InvalidState { .. })
            ));
            assert_eq!(services.main_application_state, prior_state);
            assert_eq!(services.sessions, prior_sessions);
            assert_eq!(services.next_handle, next_handle);
        }
    }

    #[test]
    fn application_open_rejects_absent_identity_wrong_sender_flags_and_wrong_object_types() {
        let mut services = Services::default();
        let mut memory = memory();
        let endpoint = open_applet_oe(&mut services, &mut memory);
        let packet = application_open_packet(
            None,
            &application_open_arguments(),
            &[0xffff8001],
            &[],
            false,
        );
        let before = services.sessions.clone();
        assert!(matches!(
            transact(&mut services, &mut memory, endpoint, &packet),
            Err(Trap::UnsupportedService { .. })
        ));
        assert_eq!(services.sessions, before);
        let mut services = application_services();
        let endpoint = open_applet_oe(&mut services, &mut memory);
        let before = services.sessions.clone();
        let identity = services.main_application.clone();
        let next = services.next_handle;
        // None of these is a supported KProcess borrow. Some are valid generic
        // HIPC copied object/empty-slot forms; no console AM Result is invented.
        for copied_handle in [0, 0xffff8000, 0x100, endpoint, 0x12345678] {
            let packet = application_open_packet(
                None,
                &application_open_arguments(),
                &[copied_handle],
                &[],
                false,
            );
            assert!(matches!(
                transact(&mut services, &mut memory, endpoint, &packet),
                Err(Trap::UnsupportedService { .. })
            ));
            assert_eq!(services.sessions, before);
            assert_eq!(services.next_handle, next);
        }
        let mut args = application_open_arguments();
        args[..4].copy_from_slice(&1u32.to_le_bytes());
        let packet = application_open_packet(None, &args, &[0xffff8001], &[], false);
        assert!(matches!(
            transact(&mut services, &mut memory, endpoint, &packet),
            Err(Trap::UnsupportedService { .. })
        ));
        let packet = application_open_packet(
            None,
            &application_open_arguments(),
            &[0xffff8001],
            &[],
            false,
        );
        memory.write(0x1000, &packet).unwrap();
        assert!(matches!(
            services.send_sync_request(&mut memory, 0x1000, endpoint, 8, 0),
            Err(Trap::UnsupportedService { .. })
        ));
        assert_eq!(services.sessions, before);
        assert_eq!(services.next_handle, next);
        assert_eq!(services.main_application, identity);
        let mut after = [0; TLS_MESSAGE_BUFFER_SIZE];
        memory.read(0x1000, &mut after).unwrap();
        assert_eq!(after, packet);
        // The raw PID and ClientProcessId placeholders in this same packet are
        // untrusted and nonzero; the correct trusted sender still succeeds.
        services
            .send_sync_request(&mut memory, 0x1000, endpoint, 7, 0)
            .unwrap();
        let proxy = memory.read_u32(0x100c).unwrap();
        assert_eq!(
            proxy_identity(&services, proxy, None),
            identity.as_ref().unwrap()
        );
    }

    #[test]
    fn application_open_and_existing_methods_enforce_copy_and_argument_shapes() {
        let mut services = application_services();
        let mut memory = memory();
        let endpoint = open_applet_oe(&mut services, &mut memory);
        convert_domain(&mut services, &mut memory, endpoint);
        let args = application_open_arguments();
        let before = services.sessions.clone();
        let next = services.next_handle;
        let mut extra_args = args.to_vec();
        extra_args.extend_from_slice(&[0; 4]);
        for packet in [
            application_open_packet(Some(1), &args, &[], &[], false),
            application_open_packet(Some(1), &args, &[0xffff8001, 0xffff8001], &[], false),
            application_open_packet(Some(1), &extra_args, &[0xffff8001], &[], false),
            application_open_packet(Some(1), &args, &[0xffff8001], &[1], false),
            application_open_packet(Some(1), &args, &[0xffff8001], &[], true),
            domain_packet(1, 10, &args, false, &[]),
        ] {
            // These are valid codec frames, rejected by this service's shape.
            ipc::parse_request_with_domain(&packet, true).unwrap();
            assert!(matches!(
                transact(&mut services, &mut memory, endpoint, &packet),
                Err(Trap::InvalidState { .. })
            ));
            assert_eq!(services.sessions, before);
            assert_eq!(services.next_handle, next);
            let mut after = [0; TLS_MESSAGE_BUFFER_SIZE];
            memory.read(0x1000, &mut after).unwrap();
            assert_eq!(after, packet);
        }
        assert!(services.private_parse_failures.is_empty());
        let lm = open_lm(&mut services, &mut memory);
        let mut packet = application_open_packet(None, &[0; 8], &[0xffff8001], &[], false);
        let request = ipc::parse_request(&packet).unwrap();
        let cmif = request.cmif.unwrap().header_offset;
        packet[cmif + 8..cmif + 12].copy_from_slice(&0u32.to_le_bytes());
        let before = services.sessions.clone();
        assert!(matches!(
            transact(&mut services, &mut memory, lm, &packet),
            Err(Trap::InvalidState { .. })
        ));
        assert_eq!(services.sessions, before);
    }

    #[test]
    fn application_proxy_reply_faults_publish_neither_session_nor_domain_object() {
        for domain in [false, true] {
            let mut services = application_services();
            let mut memory = memory();
            let endpoint = open_applet_oe(&mut services, &mut memory);
            let target = domain.then_some(1);
            if domain {
                convert_domain(&mut services, &mut memory, endpoint);
            }
            let before = services.sessions.clone();
            let next_handle = services.next_handle;
            let identity = services.main_application.clone();
            let packet = application_open_packet(
                target,
                &application_open_arguments(),
                &[0xffff8001],
                &[],
                false,
            );
            memory.write(0x1000, &packet).unwrap();
            memory.set_permissions(0x1000, 0x1000, READ).unwrap();
            assert!(matches!(
                services.send_sync_request(&mut memory, 0x1000, endpoint, 7, 0),
                Err(Trap::MemoryFault { .. })
            ));
            assert_eq!(services.sessions, before);
            assert_eq!(services.next_handle, next_handle);
            assert_eq!(services.main_application, identity);
            let mut after = [0; TLS_MESSAGE_BUFFER_SIZE];
            memory.read(0x1000, &mut after).unwrap();
            assert_eq!(after, packet);
            memory
                .set_permissions(0x1000, 0x1000, READ | WRITE)
                .unwrap();
            let created = open_application_proxy(&mut services, &mut memory, endpoint, target);
            assert_eq!(created, if domain { 2 } else { next_handle });
        }
    }

    #[test]
    fn application_proxy_capacity_is_explicit_and_freed_slots_do_not_reuse_ids() {
        for domain in [false, true] {
            let mut services = application_services();
            let mut memory = memory();
            let endpoint = open_applet_oe(&mut services, &mut memory);
            let target = domain.then_some(1);
            if domain {
                convert_domain(&mut services, &mut memory, endpoint);
            }
            let mut last = 0;
            while if domain {
                domain_state(&services, endpoint).0.len() < MAX_DOMAIN_OBJECTS
            } else {
                services.sessions.len() < MAX_LIVE_HANDLES
            } {
                last = open_application_proxy(&mut services, &mut memory, endpoint, target);
            }
            let next = if domain {
                domain_state(&services, endpoint).1
            } else {
                services.next_handle
            };
            let before = services.sessions.clone();
            let packet = application_open_packet(
                target,
                &application_open_arguments(),
                &[0xffff8001],
                &[],
                false,
            );
            assert!(matches!(
                transact(&mut services, &mut memory, endpoint, &packet),
                Err(Trap::InvalidState { .. })
            ));
            assert_eq!(services.sessions, before);
            // The same exhausted object/session capacity rejects both shared
            // logical-state and per-object-state getters without publication.
            let application_state = services.main_application_state.clone();
            assert!(
                application_state
                    .as_ref()
                    .unwrap()
                    .display_layer_manager
                    .is_none()
            );
            let (child_handle, child_target) = if domain {
                (endpoint, Some(last))
            } else {
                (last, None)
            };
            for getter in [1, 3] {
                let packet = proxy_child_packet(child_target, getter);
                assert!(matches!(
                    transact(&mut services, &mut memory, child_handle, &packet),
                    Err(Trap::InvalidState { .. })
                ));
                assert_eq!(services.main_application_state, application_state);
                assert_eq!(services.sessions, before);
                let mut after = [0; TLS_MESSAGE_BUFFER_SIZE];
                memory.read(0x1000, &mut after).unwrap();
                assert_eq!(after, packet);
            }
            if domain {
                transact(
                    &mut services,
                    &mut memory,
                    endpoint,
                    &domain_close_packet(last),
                )
                .unwrap();
            } else {
                assert!(services.close(last));
            }
            let created = open_application_proxy(&mut services, &mut memory, endpoint, target);
            assert_eq!(created, next);
            assert!(created > last);
            assert!(services.main_application.is_some());
        }
    }

    #[test]
    fn applet_oe_has_a_typed_session_controls_and_bounded_object_lifetime() {
        let mut services = Services::default();
        let mut memory = memory();
        let applet = open_applet_oe(&mut services, &mut memory);
        let lookup = services.trace.back().unwrap();
        assert_eq!(lookup.requested_service.as_deref(), Some("appletOE"));
        assert!(lookup.completed);
        let next_handle = services.next_handle;
        transact(
            &mut services,
            &mut memory,
            applet,
            &cmif_packet(5, 3, &[], false),
        )
        .unwrap();
        assert_eq!(memory.read_u32(0x101c).unwrap(), HIPC_MANAGER_INTERFACE_ID);
        assert_eq!(memory.read_u32(0x1020).unwrap(), 0x8000);
        assert_eq!(services.next_handle, next_handle);
        assert_eq!(convert_domain(&mut services, &mut memory, applet), 1);
        let (objects, next_object) = domain_state(&services, applet);
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[&1], ServiceObject::ApplicationProxyService);
        assert_eq!(next_object, 2);
        let packet = domain_close_packet(1);
        transact(&mut services, &mut memory, applet, &packet).unwrap();
        assert!(domain_state(&services, applet).0.is_empty());
        assert!(services.contains(applet));
        assert_eq!(services.next_handle, next_handle);
        let mut after = [0; TLS_MESSAGE_BUFFER_SIZE];
        memory.read(0x1000, &mut after).unwrap();
        assert_eq!(after, packet);
        let close = services.trace.back().unwrap();
        assert!(close.completed);
        assert_eq!(close.service_result, None);
        assert!(services.close(applet));
        assert!(!services.close(applet));
        assert!(!services.contains(applet));
        let report = services.report();
        assert_eq!(report["application_proxy_open_implemented"], true);
        assert_eq!(
            report["application_proxy_service_model"],
            "command_10_flags_zero_patch_compatibility"
        );
    }

    #[test]
    fn applet_oe_lookup_requires_registration_exact_name_and_writable_reply() {
        let mut services = Services::default();
        let mut memory = memory();
        let sm = services.connect_named_port("sm:", 0).unwrap();
        let next = services.next_handle;
        transact(&mut services, &mut memory, sm, &lookup_packet(b"appletOE")).unwrap();
        assert_eq!(memory.read_u32(0x1010).unwrap(), 0x415);
        assert_eq!(services.next_handle, next);
        assert_eq!(services.sessions.len(), 1);
        transact(&mut services, &mut memory, sm, &register_packet()).unwrap();
        let before = services.sessions.clone();
        assert!(matches!(
            transact(&mut services, &mut memory, sm, &lookup_packet(b"appletoe")),
            Err(Trap::UnsupportedService { .. })
        ));
        assert_eq!(services.sessions, before);
        let packet = lookup_packet(b"appletOE");
        memory.write(0x1000, &packet).unwrap();
        memory.set_permissions(0x1000, 0x1000, READ).unwrap();
        assert!(matches!(
            services.send_sync_request(&mut memory, 0x1000, sm, 7, 0),
            Err(Trap::MemoryFault { .. })
        ));
        assert_eq!(services.sessions, before);
        assert_eq!(services.next_handle, next);
        let mut after = [0; TLS_MESSAGE_BUFFER_SIZE];
        memory.read(0x1000, &mut after).unwrap();
        assert_eq!(after, packet);
        memory
            .set_permissions(0x1000, 0x1000, READ | WRITE)
            .unwrap();
        transact(&mut services, &mut memory, sm, &packet).unwrap();
        assert_eq!(memory.read_u32(0x100c).unwrap(), next);
        assert_eq!(
            services.sessions[&next].object(None),
            Some(&ServiceObject::ApplicationProxyService)
        );
    }

    #[test]
    fn applet_oe_methods_stop_without_proxy_success_or_object_mutation() {
        for domain in [false, true] {
            let mut services = Services::default();
            let mut memory = memory();
            let applet = open_applet_oe(&mut services, &mut memory);
            if domain {
                convert_domain(&mut services, &mut memory, applet);
            }
            let before = services.sessions.clone();
            let next = services.next_handle;
            for command in [0, 11] {
                // Only the observed command 10 is implemented in this stage.
                let packet = if domain {
                    domain_packet(1, command, &[0; 8], true, &[])
                } else {
                    cmif_packet(4, command, &[0; 8], true)
                };
                let error = transact(&mut services, &mut memory, applet, &packet).unwrap_err();
                assert!(matches!(
                    error,
                    Trap::UnsupportedService { ref service, command_id: Some(id), .. }
                        if service == "appletOE" && id == command
                ));
                assert_eq!(services.sessions, before);
                assert_eq!(services.next_handle, next);
                let mut after = [0; TLS_MESSAGE_BUFFER_SIZE];
                memory.read(0x1000, &mut after).unwrap();
                assert_eq!(after, packet);
                let event = services.trace.back().unwrap();
                assert!(!event.completed);
                assert_eq!(event.service_result, None);
            }
            assert!(services.private_parse_failures.is_empty());
            if !domain {
                let error = transact(
                    &mut services,
                    &mut memory,
                    applet,
                    &lookup_packet(b"unused"),
                )
                .unwrap_err();
                assert!(matches!(
                    error,
                    Trap::UnsupportedService { ref service, ref message, .. }
                        if service == "appletOE" && message == "this endpoint requires CMIF"
                ));
                assert_eq!(services.sessions, before);
            }
        }
    }

    #[test]
    fn applet_oe_control_reply_faults_do_not_convert_or_close_objects() {
        let mut services = Services::default();
        let mut memory = memory();
        let applet = open_applet_oe(&mut services, &mut memory);
        for packet in [cmif_packet(5, 3, &[], false), cmif_packet(5, 0, &[], false)] {
            let before = services.sessions.clone();
            memory.write(0x1000, &packet).unwrap();
            memory.set_permissions(0x1000, 0x1000, READ).unwrap();
            assert!(matches!(
                services.send_sync_request(&mut memory, 0x1000, applet, 7, 0),
                Err(Trap::MemoryFault { .. })
            ));
            assert_eq!(services.sessions, before);
            memory
                .set_permissions(0x1000, 0x1000, READ | WRITE)
                .unwrap();
        }
        convert_domain(&mut services, &mut memory, applet);
        let before = services.sessions.clone();
        memory.write(0x1000, &domain_close_packet(1)).unwrap();
        memory.set_permissions(0x1000, 0x1000, READ).unwrap();
        assert!(matches!(
            services.send_sync_request(&mut memory, 0x1000, applet, 7, 0),
            Err(Trap::MemoryFault { .. })
        ));
        assert_eq!(services.sessions, before);
    }

    #[test]
    fn private_parse_failures_capture_only_bounded_tls_and_leave_service_state_unchanged() {
        let mut services = Services::default();
        let mut memory = memory();
        let applet = open_applet_oe(&mut services, &mut memory);
        let mut packet = cmif_packet(4, 0, &[0; 8], true);
        packet[8..12].copy_from_slice(&0x21u32.to_le_bytes()); // PID plus one moved handle.
        packet[20..24].copy_from_slice(&0xffff8001u32.to_le_bytes());
        packet[200..].fill(0xa5);
        memory.write(0x1000, &packet).unwrap();
        memory
            .write(0x1100, &[0x3c; TLS_MESSAGE_BUFFER_SIZE])
            .unwrap();
        memory.set_permissions(0x1000, 0x1000, READ).unwrap();
        let before = services.sessions.clone();
        let next = services.next_handle;
        let log_before = services.log_sink.report();
        let trace_before = services.trace.len();
        for index in 0..MAX_PRIVATE_PARSE_FAILURES + 3 {
            let error = services
                .send_sync_request(&mut memory, 0x1000, applet, 7, 0x8000 + index as u64)
                .unwrap_err();
            assert!(matches!(
                error,
                Trap::InvalidState { ref message }
                    if message == "unsupported or malformed appletOE IPC; bounded details are in private_parse_failures"
            ));
        }
        assert_eq!(services.sessions, before);
        assert_eq!(services.next_handle, next);
        assert_eq!(services.log_sink.report(), log_before);
        assert_eq!(services.trace.len(), trace_before);
        let mut after = [0; TLS_MESSAGE_BUFFER_SIZE];
        memory.read(0x1000, &mut after).unwrap();
        assert_eq!(after, packet);
        assert_eq!(services.private_parse_failures.len(), 8);
        assert_eq!(services.parse_failures_omitted, 3);
        for (index, failure) in services.private_parse_failures.iter().enumerate() {
            assert_eq!(failure.pc, 0x8000 + index as u64);
            assert_eq!(failure.handle, applet);
            assert_eq!(failure.service, "appletOE");
            assert_eq!(failure.tls_address, 0x1000);
            assert!(!failure.domain_session);
            assert!(failure.reason.contains("handle"));
            assert!(!failure.reason_truncated);
            assert_eq!(failure.tls_snapshot_hex.len(), TLS_MESSAGE_BUFFER_SIZE * 2);
            let decoded: Vec<_> = (0..TLS_MESSAGE_BUFFER_SIZE)
                .map(|index| {
                    u8::from_str_radix(&failure.tls_snapshot_hex[index * 2..index * 2 + 2], 16)
                        .unwrap()
                })
                .collect();
            assert_eq!(decoded, packet);
        }
        let report = services.report();
        assert_eq!(
            report["private_parse_failures"]["records"]
                .as_array()
                .unwrap()
                .len(),
            8
        );
        assert_eq!(report["private_parse_failures"]["omitted"], 3);
        services.parse_failures_omitted = u64::MAX;
        assert!(
            services
                .send_sync_request(&mut memory, 0x1000, applet, 7, 0)
                .is_err()
        );
        assert_eq!(services.parse_failures_omitted, u64::MAX);

        // A failed TLS read cannot create a snapshot from zeroes/partial data.
        let mut empty = Services::default();
        let sm = empty.connect_named_port("sm:", 0).unwrap();
        assert!(matches!(
            empty.send_sync_request(&mut memory, 0x3000, sm, 7, 0),
            Err(Trap::MemoryFault { .. })
        ));
        assert!(empty.private_parse_failures.is_empty());
        assert_eq!(empty.parse_failures_omitted, 0);
        // The diagnostic's reason cap is byte bounded and UTF-8 safe.
        let session = empty.sessions[&sm].clone();
        empty.record_parse_failure(
            0,
            sm,
            &session,
            0x1000,
            &format!("a{}", "é".repeat(300)),
            &packet,
        );
        assert_eq!(
            empty.private_parse_failures[0].reason.len(),
            MAX_PARSE_FAILURE_REASON_BYTES - 1
        );
        assert!(empty.private_parse_failures[0].reason_truncated);
    }

    #[test]
    fn domain_conversion_returns_a_plain_control_reply_and_keeps_controls_plain() {
        let mut services = Services::default();
        let mut memory = memory();
        let lm = open_lm(&mut services, &mut memory);
        let next_handle = services.next_handle;
        assert_eq!(convert_domain(&mut services, &mut memory, lm), 1);
        let (objects, next_id) = domain_state(&services, lm);
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[&1], ServiceObject::LogManager);
        assert_eq!(next_id, 2);
        assert_eq!(services.next_handle, next_handle);
        let event = services.trace.back().unwrap();
        assert_eq!(event.request_kind, "Control");
        assert_eq!(event.domain_object_id, None);
        assert_eq!(event.created_domain_objects, [1]);
        assert_eq!(event.service_result, Some(0));
        transact(
            &mut services,
            &mut memory,
            lm,
            &cmif_packet(5, 3, &[], false),
        )
        .unwrap();
        assert_eq!(
            memory.read_u32(0x1010).unwrap(),
            u32::from_le_bytes(*b"SFCO")
        );
        assert_eq!(memory.read_u32(0x1020).unwrap(), 0x8000);
        assert_eq!(domain_state(&services, lm).0.len(), 1);
    }

    #[test]
    fn domain_logger_objects_survive_root_close_and_close_has_no_service_result() {
        let mut services = Services::default();
        let mut memory = memory();
        memory
            .map(0x2000, 0x1000, READ | WRITE, STATE_TLS, "log input")
            .unwrap();
        let lm = open_lm(&mut services, &mut memory);
        let root = convert_domain(&mut services, &mut memory, lm);
        let next_handle = services.next_handle;
        let first = open_domain_logger(&mut services, &mut memory, lm, root);
        let second = open_domain_logger(&mut services, &mut memory, lm, root);
        assert_eq!((first, second), (2, 3));
        assert_eq!(services.next_handle, next_handle);
        assert_eq!(services.sessions.len(), 2);
        assert_eq!(
            domain_state(&services, lm).0[&first],
            ServiceObject::Logger { process_id: 7 }
        );
        transact(
            &mut services,
            &mut memory,
            lm,
            &domain_packet(first, 1, &0xfedcba98u32.to_le_bytes(), false, &[]),
        )
        .unwrap();
        assert_eq!(services.log_sink.destination(), 0xfedcba98);
        assert_eq!(memory.read_u32(0x1010).unwrap(), 0);
        assert_eq!(memory.read_u32(0x102c).unwrap(), LOGGER_INTERFACE_ID);

        let close = domain_close_packet(root);
        transact(&mut services, &mut memory, lm, &close).unwrap();
        let mut after = [0; TLS_MESSAGE_BUFFER_SIZE];
        memory.read(0x1000, &mut after).unwrap();
        assert_eq!(after, close);
        assert!(services.contains(lm));
        assert!(!domain_state(&services, lm).0.contains_key(&root));
        let event = services.trace.back().unwrap();
        assert!(event.completed);
        assert_eq!(event.request_kind, "DomainClose");
        assert_eq!(event.service_result, None);
        assert_eq!(event.domain_object_id, Some(root));

        let packet = valid_log_packet();
        memory.write(0x2000, &packet).unwrap();
        transact(
            &mut services,
            &mut memory,
            lm,
            &domain_log_packet(first, 0x2000, packet.len() as u32),
        )
        .unwrap();
        assert_eq!(services.log_sink.records()[0].process_id, 7);
        assert_eq!(
            services.log_sink.records()[0].untrusted_packet_process_id,
            999
        );
        assert_eq!(services.log_sink.records()[0].payload, b"abc");
        assert_eq!(
            services.trace.back().unwrap().log_disposition.as_deref(),
            Some("Stored")
        );
        assert_eq!(memory.read_u32(0x1028).unwrap(), 0);
        assert_eq!(memory.read_u64(0x2000).unwrap(), 999);

        for id in [first, first, 0, 999] {
            let close = domain_close_packet(id);
            transact(&mut services, &mut memory, lm, &close).unwrap();
            memory.read(0x1000, &mut after).unwrap();
            assert_eq!(after, close);
            assert!(services.trace.back().unwrap().completed);
            assert_eq!(services.trace.back().unwrap().service_result, None);
        }
        assert_eq!(domain_state(&services, lm).0.len(), 1);
        assert!(domain_state(&services, lm).0.contains_key(&second));
        transact(
            &mut services,
            &mut memory,
            lm,
            &domain_packet(second, 1, &1u32.to_le_bytes(), false, &[]),
        )
        .unwrap();
        transact(
            &mut services,
            &mut memory,
            lm,
            &cmif_packet(5, 3, &[], false),
        )
        .unwrap();
        assert!(services.close(lm));
        assert!(!services.contains(lm));
        assert_eq!(services.sessions.len(), 1);
    }

    #[test]
    fn domain_targets_shapes_and_missing_domain_headers_never_fake_dispatch() {
        let mut services = Services::default();
        let mut memory = memory();
        let lm = open_lm(&mut services, &mut memory);
        convert_domain(&mut services, &mut memory, lm);
        let before = services.sessions[&lm].clone();
        let next_handle = services.next_handle;
        for id in [0, 2, u32::MAX] {
            let packet = domain_packet(id, 0, &[0; 8], true, &[]);
            assert!(matches!(
                transact(&mut services, &mut memory, lm, &packet),
                Err(Trap::UnsupportedService { .. })
            ));
            let mut after = [0; TLS_MESSAGE_BUFFER_SIZE];
            memory.read(0x1000, &mut after).unwrap();
            assert_eq!(after, packet);
            assert_eq!(services.sessions[&lm], before);
            assert!(!services.trace.back().unwrap().completed);
        }
        // Object 1 is an LM object, not a Logger; command 1 must not change the sink.
        assert!(matches!(
            transact(
                &mut services,
                &mut memory,
                lm,
                &domain_packet(1, 1, &42u32.to_le_bytes(), false, &[])
            ),
            Err(Trap::UnsupportedService { .. })
        ));
        assert_eq!(services.log_sink.destination(), 1);
        for packet in [
            cmif_packet(4, 0, &[0; 8], true),
            domain_packet(1, 0, &[0; 8], true, &[1]),
        ] {
            assert!(matches!(
                transact(&mut services, &mut memory, lm, &packet),
                Err(Trap::InvalidState { .. })
            ));
            assert_eq!(services.sessions[&lm], before);
        }
        assert_eq!(services.next_handle, next_handle);
    }

    #[test]
    fn domain_object_ids_are_scoped_to_their_session() {
        let mut services = Services::default();
        let mut memory = memory();
        let first = open_lm(&mut services, &mut memory);
        let second = open_lm(&mut services, &mut memory);
        convert_domain(&mut services, &mut memory, first);
        convert_domain(&mut services, &mut memory, second);
        let object = open_domain_logger(&mut services, &mut memory, first, 1);
        assert!(matches!(
            transact(
                &mut services,
                &mut memory,
                second,
                &domain_packet(object, 1, &42u32.to_le_bytes(), false, &[])
            ),
            Err(Trap::UnsupportedService { .. })
        ));
        transact(
            &mut services,
            &mut memory,
            second,
            &domain_close_packet(object),
        )
        .unwrap();
        assert!(domain_state(&services, first).0.contains_key(&object));
        assert_eq!(
            open_domain_logger(&mut services, &mut memory, second, 1),
            object
        );
        transact(
            &mut services,
            &mut memory,
            first,
            &domain_close_packet(object),
        )
        .unwrap();
        assert!(!domain_state(&services, first).0.contains_key(&object));
        assert!(domain_state(&services, second).0.contains_key(&object));
        transact(
            &mut services,
            &mut memory,
            second,
            &domain_packet(object, 1, &42u32.to_le_bytes(), false, &[]),
        )
        .unwrap();
        assert_eq!(services.log_sink.destination(), 42);
    }

    #[test]
    fn repeated_conversion_and_copy_clone_controls_stop_without_mutation() {
        let mut services = Services::default();
        let mut memory = memory();
        let lm = open_lm(&mut services, &mut memory);
        convert_domain(&mut services, &mut memory, lm);
        let before = services.sessions[&lm].clone();
        let next_handle = services.next_handle;
        for command in [0, 1] {
            let packet = cmif_packet(5, command, &[], false);
            assert!(matches!(
                transact(&mut services, &mut memory, lm, &packet),
                Err(Trap::UnsupportedService { .. })
            ));
            assert_eq!(services.sessions[&lm], before);
            assert_eq!(services.next_handle, next_handle);
            let mut after = [0; TLS_MESSAGE_BUFFER_SIZE];
            memory.read(0x1000, &mut after).unwrap();
            assert_eq!(after, packet);
        }
        // CloneCurrentObject aliases the same server session (domain table
        // included); closing the original keeps the session through the alias.
        transact(
            &mut services,
            &mut memory,
            lm,
            &cmif_packet(5, 2, &[], false),
        )
        .unwrap();
        let clone = next_handle;
        assert_eq!(services.session_aliases.get(&clone), Some(&lm));
        assert!(services.contains(clone));
        assert!(services.close(lm));
        assert_eq!(services.sessions[&clone], before);
        assert!(services.session_aliases.is_empty());
        assert!(services.close(clone));
        assert!(!services.contains(clone));
    }

    #[test]
    fn domain_capacity_and_id_exhaustion_are_explicit_and_ids_are_not_reused() {
        let mut services = Services::default();
        let mut memory = memory();
        let lm = open_lm(&mut services, &mut memory);
        convert_domain(&mut services, &mut memory, lm);
        for expected in 2..=MAX_DOMAIN_OBJECTS as u32 {
            assert_eq!(
                open_domain_logger(&mut services, &mut memory, lm, 1),
                expected
            );
        }
        assert_eq!(domain_state(&services, lm).0.len(), MAX_DOMAIN_OBJECTS);
        let before = services.sessions[&lm].clone();
        let packet = domain_packet(1, 0, &[0; 8], true, &[]);
        assert!(matches!(
            transact(&mut services, &mut memory, lm, &packet),
            Err(Trap::InvalidState { .. })
        ));
        assert_eq!(services.sessions[&lm], before);
        let next = domain_state(&services, lm).1;
        transact(&mut services, &mut memory, lm, &domain_close_packet(2)).unwrap();
        assert_eq!(open_domain_logger(&mut services, &mut memory, lm, 1), next);
        assert!(!domain_state(&services, lm).0.contains_key(&2));
        // Reach the allocation-range boundary without billions of requests.
        if let Session::Domain {
            objects,
            next_object_id,
            ..
        } = services.sessions.get_mut(&lm).unwrap()
        {
            objects.remove(&3);
            *next_object_id = u32::MAX;
        }
        let before = services.sessions[&lm].clone();
        assert!(matches!(
            transact(&mut services, &mut memory, lm, &packet),
            Err(Trap::InvalidState { .. })
        ));
        assert_eq!(services.sessions[&lm], before);
    }

    #[test]
    fn reply_write_faults_prevent_domain_conversion_and_child_allocation() {
        let mut services = Services::default();
        let mut memory = memory();
        let lm = open_lm(&mut services, &mut memory);
        let before = services.sessions[&lm].clone();
        let next_handle = services.next_handle;
        memory
            .write(0x1000, &cmif_packet(5, 0, &[], false))
            .unwrap();
        memory.set_permissions(0x1000, 0x1000, READ).unwrap();
        assert!(matches!(
            services.send_sync_request(&mut memory, 0x1000, lm, 7, 0),
            Err(Trap::MemoryFault { .. })
        ));
        assert_eq!(services.sessions[&lm], before);
        assert_eq!(services.next_handle, next_handle);
        memory
            .set_permissions(0x1000, 0x1000, READ | WRITE)
            .unwrap();
        convert_domain(&mut services, &mut memory, lm);
        let before = services.sessions[&lm].clone();
        memory
            .write(0x1000, &domain_packet(1, 0, &[0; 8], true, &[]))
            .unwrap();
        memory.set_permissions(0x1000, 0x1000, READ).unwrap();
        assert!(matches!(
            services.send_sync_request(&mut memory, 0x1000, lm, 7, 0),
            Err(Trap::MemoryFault { .. })
        ));
        assert_eq!(services.sessions[&lm], before);
        assert_eq!(services.next_handle, next_handle);
        memory
            .set_permissions(0x1000, 0x1000, READ | WRITE)
            .unwrap();
        assert_eq!(open_domain_logger(&mut services, &mut memory, lm, 1), 2);
    }

    #[test]
    fn reply_write_faults_prevent_domain_close_destination_changes_and_log_retention() {
        let mut services = Services::default();
        let mut memory = memory();
        memory
            .map(0x2000, 0x1000, READ | WRITE, STATE_TLS, "log input")
            .unwrap();
        let lm = open_lm(&mut services, &mut memory);
        convert_domain(&mut services, &mut memory, lm);
        let logger = open_domain_logger(&mut services, &mut memory, lm, 1);
        let log_packet = valid_log_packet();
        memory.write(0x2000, &log_packet).unwrap();
        for packet in [
            domain_close_packet(logger),
            domain_packet(logger, 1, &42u32.to_le_bytes(), false, &[]),
            domain_log_packet(logger, 0x2000, log_packet.len() as u32),
        ] {
            let before = services.sessions[&lm].clone();
            let log_before = services.log_sink.report();
            memory.write(0x1000, &packet).unwrap();
            memory.set_permissions(0x1000, 0x1000, READ).unwrap();
            assert!(matches!(
                services.send_sync_request(&mut memory, 0x1000, lm, 7, 0),
                Err(Trap::MemoryFault { .. })
            ));
            assert_eq!(services.sessions[&lm], before);
            assert_eq!(services.log_sink.report(), log_before);
            let mut after = [0; TLS_MESSAGE_BUFFER_SIZE];
            memory.read(0x1000, &mut after).unwrap();
            assert_eq!(after, packet);
            memory
                .set_permissions(0x1000, 0x1000, READ | WRITE)
                .unwrap();
        }
    }

    #[test]
    fn lm_open_logger_global_destination_and_map_alias_log_have_real_state() {
        let mut services = Services::default();
        let mut memory = memory();
        memory
            .map(0x2000, 0x1000, READ | WRITE, STATE_TLS, "log input")
            .unwrap();
        let lm = open_lm(&mut services, &mut memory);
        // This session really owns no server-side pointer-buffer capacity.
        transact(
            &mut services,
            &mut memory,
            lm,
            &cmif_packet(5, 3, &[], false),
        )
        .unwrap();
        assert_eq!(memory.read_u32(0x1004).unwrap(), 9);
        assert_eq!(memory.read_u32(0x101c).unwrap(), HIPC_MANAGER_INTERFACE_ID);
        assert_eq!(memory.read_u32(0x1020).unwrap(), 0x8000);
        let logger = open_logger(&mut services, &mut memory, lm);
        assert!(matches!(
            services.sessions[&logger].object(None),
            Some(ServiceObject::Logger { process_id: 7 })
        ));
        assert_eq!(memory.read_u32(0x101c).unwrap(), LM_INTERFACE_ID);
        transact(
            &mut services,
            &mut memory,
            logger,
            &cmif_packet(4, 1, &0xfedcba98u32.to_le_bytes(), false),
        )
        .unwrap();
        assert_eq!(services.log_sink.destination(), 0xfedcba98);
        let second = open_logger(&mut services, &mut memory, lm);
        assert_eq!(services.log_sink.destination(), 0xfedcba98);
        let packet = valid_log_packet();
        memory.write(0x2000, &packet).unwrap();
        transact(
            &mut services,
            &mut memory,
            second,
            &logger_packet(0x2000, packet.len() as u32),
        )
        .unwrap();
        assert_eq!(services.log_sink.records().len(), 1);
        assert_eq!(
            services.trace.back().unwrap().log_disposition.as_deref(),
            Some("Stored")
        );
        assert_eq!(memory.read_u32(0x1018).unwrap(), 0);
        assert_eq!(memory.read_u32(0x101c).unwrap(), LOGGER_INTERFACE_ID);
        assert_eq!(memory.read_u64(0x2000).unwrap(), 999);
        assert!(services.close(lm));
        assert!(services.contains(logger));
        assert!(services.contains(second));
    }

    #[test]
    fn unregistered_sm_returns_invalid_client_with_the_declared_tipc_handle_slot() {
        let mut services = Services::default();
        let mut memory = memory();
        let handle = services.connect_named_port("sm:", 0).unwrap();
        // Session initialization is checked before interpreting the empty name.
        transact(&mut services, &mut memory, handle, &lookup_packet(&[])).unwrap();
        assert_eq!(memory.read_u32(0x1000).unwrap(), 0x11);
        assert_eq!(memory.read_u32(0x1004).unwrap(), 0x80000001);
        assert_eq!(memory.read_u32(0x1008).unwrap(), 0x20);
        assert_eq!(memory.read_u32(0x100c).unwrap(), 0);
        assert_eq!(memory.read_u32(0x1010).unwrap(), 0x415);
        assert!(services.trace[0].completed);
        assert_eq!(services.trace[0].service_result, Some(0x415));
        assert_eq!(registration(&services, handle), None);
        assert_eq!(services.sessions.len(), 1);
    }

    #[test]
    fn reply_faults_prevent_handle_allocation_destination_changes_and_log_retention() {
        let mut services = Services::default();
        let mut memory = memory();
        memory
            .map(0x2000, 0x1000, READ | WRITE, STATE_TLS, "log input")
            .unwrap();
        let lm = open_lm(&mut services, &mut memory);
        let next = services.next_handle;
        memory
            .write(0x1000, &cmif_packet(4, 0, &[0; 8], true))
            .unwrap();
        memory.set_permissions(0x1000, 0x1000, READ).unwrap();
        assert!(matches!(
            services.send_sync_request(&mut memory, 0x1000, lm, 7, 0),
            Err(Trap::MemoryFault { .. })
        ));
        assert_eq!(services.next_handle, next);
        assert_eq!(services.sessions.len(), 2);
        memory
            .set_permissions(0x1000, 0x1000, READ | WRITE)
            .unwrap();
        let logger = open_logger(&mut services, &mut memory, lm);
        assert_eq!(logger, next);
        memory
            .write(0x1000, &cmif_packet(4, 1, &42u32.to_le_bytes(), false))
            .unwrap();
        memory.set_permissions(0x1000, 0x1000, READ).unwrap();
        assert!(matches!(
            services.send_sync_request(&mut memory, 0x1000, logger, 7, 0),
            Err(Trap::MemoryFault { .. })
        ));
        assert_eq!(services.log_sink.destination(), 1);
        memory
            .set_permissions(0x1000, 0x1000, READ | WRITE)
            .unwrap();
        let packet = valid_log_packet();
        memory.write(0x2000, &packet).unwrap();
        memory
            .write(0x1000, &logger_packet(0x2000, packet.len() as u32))
            .unwrap();
        memory.set_permissions(0x1000, 0x1000, READ).unwrap();
        assert!(matches!(
            services.send_sync_request(&mut memory, 0x1000, logger, 7, 0),
            Err(Trap::MemoryFault { .. })
        ));
        assert!(services.log_sink.records().is_empty());
    }

    #[test]
    fn sm_registration_uses_kernel_identity_and_lookup_stops_without_a_backend() {
        let mut services = Services::default();
        let handle = services.connect_named_port("sm:", 0).unwrap();
        let mut memory = memory();
        memory.write(0x1000, &register_packet()).unwrap();
        services
            .send_sync_request(&mut memory, 0x1000, handle, 7, 0)
            .unwrap();
        assert_eq!(registration(&services, handle), Some(7));
        assert!(services.trace[0].completed);
        let mut lookup = [0; TLS_MESSAGE_BUFFER_SIZE];
        lookup[..4].copy_from_slice(&0x11u32.to_le_bytes());
        lookup[4..8].copy_from_slice(&2u32.to_le_bytes());
        lookup[8..15].copy_from_slice(b"missing");
        memory.write(0x1000, &lookup).unwrap();
        assert!(matches!(
            services.send_sync_request(&mut memory, 0x1000, handle, 7, 0),
            Err(Trap::UnsupportedService {
                command_id: Some(1),
                ..
            })
        ));
        let mut after = [0; TLS_MESSAGE_BUFFER_SIZE];
        memory.read(0x1000, &mut after).unwrap();
        assert_eq!(lookup, after);
        assert_eq!(
            services.trace[1].requested_service.as_deref(),
            Some("missing")
        );
        assert!(!services.trace[1].completed);
        assert!(services.close(handle));
        assert!(!services.close(handle));
        assert!(!services.contains(handle));
        assert!(
            services
                .send_sync_request(&mut memory, 0x1000, handle, 7, 0)
                .is_err()
        );
    }

    #[test]
    fn reply_memory_fault_does_not_register_the_session() {
        let mut services = Services::default();
        let handle = services.connect_named_port("sm:", 0).unwrap();
        let mut memory = memory();
        memory.write(0x1000, &register_packet()).unwrap();
        memory.set_permissions(0x1000, 0x1000, READ).unwrap();
        assert!(matches!(
            services.send_sync_request(&mut memory, 0x1000, handle, 7, 0),
            Err(Trap::MemoryFault { .. })
        ));
        assert_eq!(registration(&services, handle), None);
    }

    #[test]
    fn local_user_profile_base_follows_the_documented_layout() {
        let user = LocalUser {
            uid: LocalUser::UID,
            nickname: "Ash".into(),
        };
        let base = user.profile_base(&user.uid);
        assert_eq!(base.len(), 0x38);
        assert_eq!(&base[..16], &LocalUser::UID);
        assert_eq!(&base[0x10..0x18], &[0; 8]);
        assert_eq!(&base[0x18..0x1b], b"Ash");
        assert!(base[0x1b..].iter().all(|byte| *byte == 0));
        assert_ne!(LocalUser::UID, [0; 16]);
        assert_eq!(user.user_data().len(), 0x80);
    }

    #[test]
    fn cmif_register_uses_descriptor_identity_and_explicit_generic_reply_profile() {
        let mut services = Services::default();
        let handle = services.connect_named_port("sm:", 0).unwrap();
        let mut memory = memory();
        let mut packet = [0; TLS_MESSAGE_BUFFER_SIZE];
        packet[..4].copy_from_slice(&6u32.to_le_bytes());
        packet[4..8].copy_from_slice(&0x8000000au32.to_le_bytes());
        packet[8..12].copy_from_slice(&1u32.to_le_bytes());
        packet[12..20].copy_from_slice(&999u64.to_le_bytes());
        packet[32..36].copy_from_slice(b"SFCI");
        packet[36..40].copy_from_slice(&1u32.to_le_bytes());
        packet[44..48].copy_from_slice(&0xdeadbeefu32.to_le_bytes());
        // The reserved u64 is not the authenticated client identity.
        packet[48..56].copy_from_slice(&888u64.to_le_bytes());
        memory.write(0x1000, &packet).unwrap();
        services
            .send_sync_request(&mut memory, 0x1000, handle, 7, 0)
            .unwrap();
        assert_eq!(registration(&services, handle), Some(7));
        assert_eq!(memory.read_u32(0x1000).unwrap(), 0);
        assert_eq!(memory.read_u32(0x1004).unwrap(), 8);
        assert_eq!(
            memory.read_u32(0x1010).unwrap(),
            u32::from_le_bytes(*b"SFCO")
        );
        assert_eq!(memory.read_u32(0x1014).unwrap(), 0);
        assert_eq!(memory.read_u32(0x1018).unwrap(), 0);
        assert_eq!(memory.read_u32(0x101c).unwrap(), ipc::SM_CMIF_INTERFACE_ID);
        assert_eq!(services.report()["cmif_reply_profile"], "generic_sf");
    }
}
