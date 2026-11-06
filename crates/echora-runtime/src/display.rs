//! DisplaySession (IGraphicsSupervisor InitializeDisplayContext, 305).
//!
//! Switchbrew "Switch 2: GSV services" documents the session as a TIPC
//! interface and lists its command IDs without names or I/O. Names, argument
//! layouts and limits here follow the pinned Eden 1a6d7d4 IVi2Session and are
//! provisional. Only the default display ("default") is modeled; GameShare
//! player displays, notification events and buffer presentation stop
//! explicitly until implemented.

use serde::Serialize;

/// The eight-byte display name "default" (zero padded, little endian).
pub const DEFAULT_DISPLAY_NAME: u64 = 0x0074_6c75_6166_6564;
/// Client-state records per session (Eden 1a6d7d4).
const MAX_CLIENT_STATES: usize = 16;
/// Display references per session (Eden 1a6d7d4).
const MAX_DISPLAY_REFERENCES: u32 = 1024;
/// Presentation buffer slots per client, all free at registration.
const ALL_BUFFER_SLOTS: u64 = 0x7fff;
/// SetClientOutputEncoding: the only encoding the SDK sends (Eden 1a6d7d4).
const DEFAULT_OUTPUT_ENCODING: u64 = 0x14;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct ClientState {
    pub token: u64,
    pub retire_notification: u32,
    pub available_notification: u32,
    pub bound_to_default_display: bool,
    /// A secondary display this client is bound to instead.
    pub secondary_display: Option<u64>,
    pub output_encoding: u64,
    /// Free presentation buffer slots, one bit each (15 slots, Eden 1a6d7d4).
    pub available: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct DisplaySessionState {
    pub request_group_limit: Option<u32>,
    pub display_references: u32,
    pub display_notification: u32,
    pub clients: Vec<ClientState>,
    pub client_slots: u32,
    pub default_endpoint: Option<u64>,
    /// Local backing ID of the session's notification event, once created.
    pub notification_event: Option<u32>,
    /// Readable events moved by command 31, by event index, each created
    /// once.
    pub events_31: std::collections::BTreeMap<u32, u32>,
    /// Command 31 event indices bound by the odd commands 261..=271, with
    /// the command.
    pub event_bindings: std::collections::BTreeMap<u32, u32>,
    /// Pending notification bits (32 x u32), read by GetNotifications.
    pub notifications: Vec<u32>,
    /// Other displays opened by name (for example "player1", a GameShare
    /// player display), with their notification flags. Nothing is shown on
    /// them.
    pub secondary_displays: Vec<(u64, u32)>,
    /// Endpoints named after a secondary display: (name, target display,
    /// client token). The game opens "player1" on "default", so the target
    /// may be the default display (inference: the player view is routed
    /// to the main output).
    pub secondary_endpoints: Vec<(u64, u64, u64)>,
    /// Command 301's two values for the default display (unnamed).
    pub display_values: Option<(u64, u64)>,
}

/// A successful reply's raw data, or an explicit stop.
pub type Handled = Result<Vec<u8>, String>;

fn word(arguments: &[u8], index: usize) -> u32 {
    crate::bytes::le_u32(&arguments[index * 4..index * 4 + 4])
}

fn dword(arguments: &[u8], index: usize) -> u64 {
    u64::from(word(arguments, index)) | (u64::from(word(arguments, index + 1)) << 32)
}

impl DisplaySessionState {
    fn client(&mut self, token: u64) -> Option<&mut ClientState> {
        if token == 0 {
            return None;
        }
        self.clients.iter_mut().find(|client| client.token == token)
    }

    pub fn handle(&mut self, command: u32, arguments: &[u8]) -> Handled {
        let expect = |words: usize| {
            if arguments.len() == words * 4 {
                Ok(())
            } else {
                Err(format!(
                    "display session command {command} expects {words} argument words, got {} bytes",
                    arguments.len()
                ))
            }
        };
        match command {
            // GetNotificationLimit (1), GetSessionLimit (3) and
            // GetRequestGroupLimit (5): u32. 7 is the inclusive upper request
            // group index the SDK turns into 32 client slots.
            1 | 3 | 5 => {
                expect(0)?;
                Ok((if command == 5 { 7u32 } else { 0 }).to_le_bytes().to_vec())
            }
            // ConfigureRequestGroups(u32 limit < 8).
            51 => {
                expect(1)?;
                let limit = word(arguments, 0);
                if limit >= 8 {
                    return Err(format!("request group limit {limit} is not below 8"));
                }
                self.request_group_limit = Some(limit);
                Ok(Vec::new())
            }
            // OpenDisplay(u64 name, u32 notification flags).
            101 => {
                expect(3)?;
                let (name, flags) = (dword(arguments, 0), word(arguments, 2));
                if name != DEFAULT_DISPLAY_NAME {
                    // Inference: a secondary display opens like the default
                    // one; the console has no such output, so it stays dark.
                    if self.secondary_displays.len() >= MAX_DISPLAY_REFERENCES as usize {
                        return Err("display reference limit reached".into());
                    }
                    self.secondary_displays.push((name, flags));
                    return Ok(Vec::new());
                }
                if flags != 1 {
                    return Err(format!("default display flags {flags:#x} are not 1"));
                }
                if self.display_references >= MAX_DISPLAY_REFERENCES {
                    return Err("display reference limit reached".into());
                }
                self.display_notification = flags;
                self.display_references += 1;
                Ok(Vec::new())
            }
            // RegisterClientState(u64 token, u32 retire slot, u32 available slot).
            201 => {
                expect(4)?;
                let token = dword(arguments, 0);
                let (first, second) = (word(arguments, 2), word(arguments, 3));
                if token == 0 || first >= 32 || second >= 32 || first == second {
                    return Err("client state token or notification slots are invalid".into());
                }
                let slots = (1u32 << first) | (1u32 << second);
                if self.client(token).is_some() || self.client_slots & slots != 0 {
                    return Err("client state token or slots are already registered".into());
                }
                if self.clients.len() >= MAX_CLIENT_STATES {
                    return Err("client state capacity exhausted".into());
                }
                self.clients.push(ClientState {
                    token,
                    retire_notification: first,
                    available_notification: second,
                    bound_to_default_display: false,
                    secondary_display: None,
                    output_encoding: DEFAULT_OUTPUT_ENCODING,
                    available: ALL_BUFFER_SLOTS,
                });
                self.client_slots |= slots;
                Ok(Vec::new())
            }
            // BindClientDisplay(u64 token, u64 display name).
            251 => {
                expect(4)?;
                let (token, name) = (dword(arguments, 0), dword(arguments, 2));
                if self
                    .secondary_displays
                    .iter()
                    .any(|&(open, _)| open == name)
                {
                    let client = self
                        .client(token)
                        .ok_or_else(|| "bind for an unregistered client".to_string())?;
                    if client.bound_to_default_display || client.secondary_display.is_some() {
                        return Err("client is already bound to a display".into());
                    }
                    client.secondary_display = Some(name);
                    return Ok(Vec::new());
                }
                if name != DEFAULT_DISPLAY_NAME || self.display_references == 0 {
                    return Err(format!("bind to display {name:#x} that is not open"));
                }
                let client = self
                    .client(token)
                    .ok_or_else(|| "bind for an unregistered client".to_string())?;
                if client.bound_to_default_display || client.secondary_display.is_some() {
                    return Err("client is already bound to a display".into());
                }
                client.bound_to_default_display = true;
                Ok(Vec::new())
            }
            // SetClientOutputEncoding(u64 token, u64 encoding).
            351 => {
                expect(4)?;
                let (token, encoding) = (dword(arguments, 0), dword(arguments, 2));
                if encoding != DEFAULT_OUTPUT_ENCODING {
                    return Err(format!("output encoding {encoding:#x} is not modeled"));
                }
                let client = self
                    .client(token)
                    .ok_or_else(|| "output encoding for an unregistered client".to_string())?;
                client.output_encoding = encoding;
                Ok(Vec::new())
            }
            // OpenEndpoint(u64 name, u64 display name, u64 client token).
            111 => {
                expect(6)?;
                let name = dword(arguments, 0);
                let display = dword(arguments, 2);
                let token = dword(arguments, 4);
                let secondary = self
                    .client(token)
                    .and_then(|client| client.secondary_display);
                let target_open = display == name
                    || (display == DEFAULT_DISPLAY_NAME && self.display_references != 0);
                if secondary == Some(name) && target_open {
                    if self
                        .secondary_endpoints
                        .iter()
                        .any(|&(open, ..)| open == name)
                    {
                        return Err(format!("endpoint {name:#x} is already open"));
                    }
                    self.secondary_endpoints.push((name, display, token));
                    return Ok(Vec::new());
                }
                if name != DEFAULT_DISPLAY_NAME || display != name {
                    return Err(format!(
                        "endpoint {name:#x}/{display:#x} is not the default"
                    ));
                }
                let bound = self
                    .client(token)
                    .is_some_and(|client| client.bound_to_default_display);
                if !bound {
                    return Err("endpoint client is not bound to the default display".into());
                }
                if self.default_endpoint.is_some() {
                    return Err("default endpoint is already open".into());
                }
                self.default_endpoint = Some(token);
                Ok(Vec::new())
            }
            // GetClientAvailability(u64 token) -> u64 free buffer slot mask,
            // for a registered client while the default display is open.
            212 => {
                expect(2)?;
                let open = self.display_references != 0;
                let client = self
                    .client(dword(arguments, 0))
                    .ok_or_else(|| "availability for an unregistered client".to_string())?;
                if !open {
                    return Err("availability query without an open display".into());
                }
                Ok(client.available.to_le_bytes().to_vec())
            }
            // 261..=271 odd (u64 display name, u32 event index) -> ():
            // undocumented. The SDK allocates the index for command 31's
            // event first and releases it if these fail (sdk+0x5eb338,
            // sdk+0x5ebb68, sdk+0x5ebdf8, sdk+0x5ec428), so each binds that
            // event to one property's change notification: 263 from
            // QueryRefreshRate, 265 QueryHdrStatus, 267 QueryHdrPaperWhite,
            // 271 QueryVrrSupported. None of them changes here, so bound
            // events are never signaled (provisional).
            261 | 263 | 265 | 267 | 269 | 271 => {
                expect(3)?;
                let name = dword(arguments, 0);
                let index = word(arguments, 2);
                if name != DEFAULT_DISPLAY_NAME || index >= 8 {
                    return Err(format!(
                        "command {command} for display {name:#x} index {index}"
                    ));
                }
                self.event_bindings.insert(index, command);
                Ok(Vec::new())
            }
            // 301(u64 display name, u64, u64) -> (): unnamed and absent from
            // Eden 1a6d7d4. The SDK's wrapper (sdk+0x5623dc) sends the
            // default display and 0x8000000000000000 for both values, reads
            // no reply data and treats any failure as an error. Recorded as a
            // display setting with no further effect (provisional).
            301 => {
                expect(6)?;
                let name = dword(arguments, 0);
                if name != DEFAULT_DISPLAY_NAME || self.display_references == 0 {
                    return Err(format!(
                        "command 301 for display {name:#x} that is not open"
                    ));
                }
                self.display_values = Some((dword(arguments, 2), dword(arguments, 4)));
                Ok(Vec::new())
            }
            _ => Err(format!(
                "display session command {command} is not implemented (arguments {} bytes {:02x?})",
                arguments.len(),
                arguments
            )),
        }
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    fn args(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|word| word.to_le_bytes()).collect()
    }

    fn name() -> [u32; 2] {
        [
            DEFAULT_DISPLAY_NAME as u32,
            (DEFAULT_DISPLAY_NAME >> 32) as u32,
        ]
    }

    #[test]
    fn secondary_display_binds_a_client_and_opens_an_endpoint() {
        let player1 = u64::from_le_bytes(*b"player1\0");
        let display = [player1 as u32, (player1 >> 32) as u32];
        let mut state = DisplaySessionState::default();
        state
            .handle(101, &args(&[display[0], display[1], 0]))
            .unwrap();
        state.handle(201, &args(&[9, 0, 2, 3])).unwrap();
        assert!(
            state
                .handle(251, &args(&[8, 0, display[0], display[1]]))
                .is_err()
        );
        state
            .handle(251, &args(&[9, 0, display[0], display[1]]))
            .unwrap();
        assert!(
            state
                .handle(251, &args(&[9, 0, display[0], display[1]]))
                .is_err()
        );
        let endpoint = [display[0], display[1], display[0], display[1], 9, 0];
        state.handle(111, &args(&endpoint)).unwrap();
        assert!(state.handle(111, &args(&endpoint)).is_err());
        assert_eq!(state.secondary_endpoints, [(player1, player1, 9)]);
        assert_eq!(state.default_endpoint, None);
    }

    #[test]
    fn notification_commands_bind_event_indices_of_the_default_display() {
        let mut state = DisplaySessionState::default();
        let [low, high] = name();
        assert!(state.handle(261, &args(&[low, high])).is_err());
        assert!(state.handle(263, &args(&[low + 1, high, 0])).is_err());
        assert!(state.handle(265, &args(&[low, high, 8])).is_err());
        assert!(state.event_bindings.is_empty());
        for (command, index) in [(261, 0), (263, 1), (271, 2)] {
            let reply = state.handle(command, &args(&[low, high, index]));
            assert!(reply.unwrap().is_empty());
        }
        assert_eq!(
            state.event_bindings.iter().collect::<Vec<_>>(),
            [(&0, &261), (&1, &263), (&2, &271)]
        );
    }

    #[test]
    fn default_display_client_and_endpoint_sequence() {
        let mut state = DisplaySessionState::default();
        assert_eq!(state.handle(5, &[]).unwrap(), 7u32.to_le_bytes());
        assert_eq!(state.handle(3, &[]).unwrap(), 0u32.to_le_bytes());
        state.handle(51, &args(&[7])).unwrap();
        assert!(state.handle(51, &args(&[8])).is_err());
        let [low, high] = name();
        // Binding before the display is open stops.
        state.handle(201, &args(&[0x1000, 0, 0, 1])).unwrap();
        assert!(state.handle(251, &args(&[0x1000, 0, low, high])).is_err());
        state.handle(101, &args(&[low, high, 1])).unwrap();
        // Another name opens a secondary display, not the default one.
        state.handle(101, &args(&[low + 1, high, 1])).unwrap();
        assert_eq!(state.secondary_displays, [(DEFAULT_DISPLAY_NAME + 1, 1)]);
        // Overlapping slots or a repeated token are rejected.
        assert!(state.handle(201, &args(&[0x2000, 0, 1, 2])).is_err());
        assert!(state.handle(201, &args(&[0x1000, 0, 4, 5])).is_err());
        assert!(
            state
                .handle(111, &args(&[low, high, low, high, 0x1000, 0]))
                .is_err()
        );
        state.handle(251, &args(&[0x1000, 0, low, high])).unwrap();
        state.handle(351, &args(&[0x1000, 0, 0x14, 0])).unwrap();
        assert!(state.handle(351, &args(&[0x1000, 0, 0x15, 0])).is_err());
        state
            .handle(111, &args(&[low, high, low, high, 0x1000, 0]))
            .unwrap();
        assert!(
            state
                .handle(111, &args(&[low, high, low, high, 0x1000, 0]))
                .is_err()
        );
        assert_eq!(state.default_endpoint, Some(0x1000));
        assert_eq!(
            state.handle(212, &args(&[0x1000, 0])).unwrap(),
            0x7fffu64.to_le_bytes()
        );
        assert!(state.handle(212, &args(&[0x3000, 0])).is_err());
        state
            .handle(301, &args(&[low, high, 0, 1 << 31, 0, 1 << 31]))
            .unwrap();
        assert_eq!(state.display_values, Some((1 << 63, 1 << 63)));
        assert!(state.handle(229, &[]).is_err());
    }
}
